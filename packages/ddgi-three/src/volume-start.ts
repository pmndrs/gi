/**
 * What `GIVolume.start(renderer, scene)` does (T1.14, `docs/design/three-composite.md`,
 * "The simple path"): everything between "I have a volume" and "GI is on",
 * so the simple path takes as many lines as three's `LightProbeGrid`
 * (construct, place, add, bake):
 *
 * 1. the GPU device is three's own (`renderer.backend.device`), so GI and
 *    three share it (SPEC: the host owns the device);
 * 2. the platform and a DDGI technique on the volume's probe grid, prepared;
 * 3. `ThreeGI` on the scene, with `updateOnRender`: the GI light's node runs
 *    each GI frame from inside three's render (gi-light.ts), so there is no
 *    per-frame call;
 * 4. the warm-up (below), so the first frame the user sees has converged GI
 *    instead of probes filling in from black;
 * 5. the device's limits: a scene that doesn't fit them throws
 *    `GILimitError` naming the limit, with the fix for three.
 *
 * **The warm-up** (moved from the viewer, `apps/viewer/src/techniques/ddgi-technique.ts`):
 * - no probe work until the ray-tracing trees exist (rays would see through
 *   every wall of a half-built scene to the sky);
 * - then up to {@link DEFAULT_WARM_UP}.passes passes over every probe, as
 *   many probes per step as the technique allows, with a running average:
 *   pass k keeps k/(k+1) of the old value, so after n passes the field is
 *   the plain mean of n estimates, the fastest way to average out ray noise.
 *   It stops early after `maxSeconds` (with at least `minPasses` passes), so
 *   a slow GPU isn't kept waiting. The normal hysteresis carries on from
 *   there. What a game does behind its loading screen: at 2 ms a frame,
 *   Sponza's 19,425 probes would take minutes to fill from black;
 * - then the platform's budget calibration (budget v2): frames with the
 *   millisecond budget, GPU-bound with two in flight, until the platform
 *   reports it calibrated and confirmed, or {@link CALIBRATION_MAX_STEPS}.
 *   Without GPU timestamps the platform can't calibrate and says so at once.
 */

import { PerspectiveCamera, type Camera, type Scene, type WebGPURenderer } from 'three/webgpu';
import { createGI, DDGITechnique, GILimitError, type DDGISettingsInput, type EnvironmentDescriptor, type FrameBudget, type GIPlatform } from '@pmndrs/ddgi-core';
import type { PmndrsSkyLike } from './environment.ts';
import type { GILightOptions } from './gi-light.ts';
import type { GIVolume } from './gi-volume.ts';
import { ThreeGI } from './three-gi.ts';

export interface WarmUpSettings {
    /** Full passes over every probe with a running average. Default 16. */
    readonly passes: number;
    /** Stop after this long even if the passes aren't done... Default 8. */
    readonly maxSeconds: number;
    /** ...but not before every probe has been updated this many times. Default 2. */
    readonly minPasses: number;
    /** Run the budget calibration after the passes (needs `timestamp-query`). Default true. */
    readonly calibrate: boolean;
}

export const DEFAULT_WARM_UP: WarmUpSettings = { passes: 16, maxSeconds: 8, minPasses: 2, calibrate: true };

/**
 * The calibration stops after this many frames even if the platform is
 * still measuring: a burst gives up after 400 frames and its confirmation
 * after 2 × 300 (`CALIBRATION` in ddgi-core's work-budget.ts).
 */
export const CALIBRATION_MAX_STEPS = 1000;
/** Calibration frames queued at once, so the GPU never waits for the CPU between them. */
const CALIBRATION_FRAMES_IN_FLIGHT = 2;
/** Warm-up steps recorded before waiting for the GPU. Each can be a few thousand probes. */
const WARM_UP_STEPS_PER_BATCH = 4;
/** The DDGI technique's counter of probes it updated in the last frame. */
const PROBES_UPDATED_COUNTER = 'probes updated';

export const DEFAULT_GI_BUDGET: FrameBudget = { gpuMs: 2 };

export interface GIVolumeStartOptions {
    /** GPU work per frame for GI updates after the warm-up. Default `{ gpuMs: 2 }`. */
    readonly budget?: FrameBudget;
    /** DDGI settings over the defaults (rays per probe, hysteresis, ...). The probe grid comes from the volume. */
    readonly ddgi?: Partial<Omit<DDGISettingsInput, 'volume'>>;
    /** An environment for GI rays instead of the scene's (environment.ts). */
    readonly environment?: EnvironmentDescriptor | null;
    /** A pmndrs/sky `Sky` to take GI's sky from (environment.ts). */
    readonly sky?: PmndrsSkyLike | null;
    /** Settings for the GI light. */
    readonly light?: GILightOptions;
    /** The warm-up before `start` resolves (file comment). `false` skips it: GI then fills in live. */
    readonly warmUp?: boolean | Partial<WarmUpSettings>;
    /** Where warnings go (each distinct message once). Default `console.warn`. */
    readonly warn?: (message: string) => void;
}

/** What `start` made; the volume keeps it. */
export interface StartedGI {
    readonly platform: GIPlatform;
    readonly technique: DDGITechnique;
    readonly threeGI: ThreeGI;
}

/**
 * A `GILimitError` with the fix for three: the scene needs more than the
 * renderer's device allows. three makes its device with default limits (a
 * 128 MiB storage binding; NOTES), which large scenes outgrow.
 */
export class ThreeGILimitError extends GILimitError {
    constructor(source: GILimitError) {
        super(source.limit, source.needed, source.available);
        this.message =
            `@pmndrs/ddgi/three: the scene needs ${source.limit} = ${source.needed}, but the renderer's device allows ${source.available}. ` +
            'Make the renderer with the platform\'s limits: `const { renderer } = await createGIRenderer()`, or ' +
            '`new WebGPURenderer({ requiredLimits: requiredLimits(await navigator.gpu.requestAdapter()) })`.';
    }
}

/** The parts of the platform the warm-up reads that the contract doesn't name yet (reported for the platform). */
interface TreesBuiltSignal {
    rays?: { whenBuilt?: () => Promise<void> };
}

function camerasAt(volume: GIVolume): Camera {
    // The warm-up updates every probe, so its camera only orders the work; any view will do.
    const camera = new PerspectiveCamera(60, 1, 0.1, 1000);
    const { center } = volume.ddgiVolume();
    camera.position.set(center[0], center[1], center[2]);
    camera.updateMatrixWorld();
    return camera;
}

async function runWarmUp(started: StartedGI, camera: Camera, budget: FrameBudget, settings: WarmUpSettings): Promise<void> {
    const { platform, technique, threeGI } = started;
    const probeCount = technique.field?.grid.probeCount ?? 0;
    const targetHysteresis = technique.settings.hysteresis;
    const startedAt = performance.now();
    let probeUpdates = 0;
    const passes = () => (probeCount > 0 ? probeUpdates / probeCount : settings.passes);
    try {
        while (passes() < settings.passes) {
            for (let step = 0; step < WARM_UP_STEPS_PER_BATCH && passes() < settings.passes; step++) {
                const completed = Math.floor(passes());
                technique.settings.hysteresis = Math.min(targetHysteresis, completed / (completed + 1));
                threeGI.budget = { units: Math.min(probeCount, technique.settings.maxProbesPerFrame) };
                threeGI.update(camera, 0);
                probeUpdates += technique.counters[PROBES_UPDATED_COUNTER] ?? 0;
            }
            await platform.device.queue.onSubmittedWorkDone();
            const outOfTime = performance.now() - startedAt > settings.maxSeconds * 1000 && passes() >= settings.minPasses;
            if (outOfTime) break;
        }
    } finally {
        technique.settings.hysteresis = targetHysteresis;
    }

    threeGI.budget = budget;
    if (!settings.calibrate) return;
    const budgetCounter = `${technique.id} budget`;
    const queued: Promise<void>[] = [];
    for (let step = 0; step < CALIBRATION_MAX_STEPS; step++) {
        threeGI.update(camera, 0);
        const report = platform.stats.counters[budgetCounter];
        if (report?.calibrating !== 1 && report?.confirming !== 1) break;
        // Two frames in flight: wait for the one before this, never for this one.
        queued.push(platform.device.queue.onSubmittedWorkDone());
        if (queued.length >= CALIBRATION_FRAMES_IN_FLIGHT) await queued.shift();
        // Lets the timestamp read-backs land before the next frame collects them.
        await new Promise((resolve) => setTimeout(resolve, 0));
    }
    await Promise.all(queued);
}

/** `GIVolume.start` (file comment). Throws `ThreeGILimitError` when the scene outgrows the device's limits. */
export async function startVolume(volume: GIVolume, renderer: WebGPURenderer, scene: Scene, options: GIVolumeStartOptions = {}): Promise<StartedGI> {
    const backend = renderer.backend as unknown as { device?: GPUDevice };
    if (!backend.device) await renderer.init();
    const device = backend.device;
    if (!device) throw new Error('@pmndrs/ddgi/three: GIVolume.start needs a WebGPURenderer on WebGPU (it got no GPU device; is it the WebGL fallback?).');
    if (volume.parent === null) scene.add(volume);
    scene.updateMatrixWorld();

    const platform = createGI(device, { gpuTiming: device.features.has('timestamp-query'), label: volume.name || 'GIVolume' });
    let threeGI: ThreeGI | null = null;
    try {
        const technique = platform.addTechnique(new DDGITechnique({ ...options.ddgi, volume: volume.ddgiVolume() }));
        platform.setOutput(technique);
        await platform.prepare();
        threeGI = new ThreeGI(platform, scene, {
            // No probe work until the trees exist (file comment); the warm-up sets the real budget.
            budget: { paused: true },
            volume,
            environment: options.environment,
            sky: options.sky,
            light: options.light,
            warn: options.warn,
            // Switched on after the warm-up: a render loop already running must not add GI frames to it.
            updateOnRender: false,
        });
        const started: StartedGI = { platform, technique, threeGI };
        const camera = camerasAt(volume);
        // The first update copies the scene in and starts the tree builds; the second sends the finished trees.
        threeGI.update(camera, 0);
        await (platform as unknown as TreesBuiltSignal).rays?.whenBuilt?.();
        threeGI.update(camera, 0);
        const budget = options.budget ?? DEFAULT_GI_BUDGET;
        if (options.warmUp === false) {
            threeGI.budget = budget;
        } else {
            await runWarmUp(started, camera, budget, { ...DEFAULT_WARM_UP, ...(typeof options.warmUp === 'object' ? options.warmUp : {}) });
        }
        threeGI.updateOnRender = true;
        return started;
    } catch (error) {
        threeGI?.dispose();
        platform.dispose();
        if (error instanceof GILimitError) throw new ThreeGILimitError(error);
        throw error;
    }
}
