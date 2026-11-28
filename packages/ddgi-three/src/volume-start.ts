/**
 * What `DDGIVolume.start(renderer, scene)` does (T1.14, `docs/design/three-composite.md`,
 * "The simple path"): everything between "I have a volume" and "GI is on",
 * so the simple path takes as many lines as three's `LightProbeGrid`
 * (construct, place, add, bake):
 *
 * 1. the GPU device is three's own (`renderer.backend.device`), so GI and
 *    three share it (SPEC: the host owns the device);
 * 2. the platform and a DDGI technique on the volume's probe grid, prepared,
 *    with the scene copied in while the kernels compile (T2.14);
 * 3. `ThreeGI` on the scene, with `updateOnRender`: the GI light's node runs
 *    each GI frame from inside three's render (gi-light.ts), so there is no
 *    per-frame call;
 * 4. the start-up frames (below), so the first frame the user sees has
 *    converged GI instead of probes filling in from black;
 * 5. the device's limits: a scene that doesn't fit them throws
 *    `GILimitError` naming the limit, with the fix for three.
 *
 * **The start-up frames.** The warm-up itself is the DDGI technique's since
 * T2.4b (`warmUpPasses`, `warmUpMaxSeconds`, `warmUpMinPasses`; ddgi-core's
 * ddgi-technique.ts): it waits for the ray-tracing trees, then makes up to
 * {@link DEFAULT_WARM_UP}.passes passes over every probe whatever the
 * frame's budget, stopping early after `maxSeconds` once `minPasses` are
 * made, so a slow GPU isn't kept waiting. What a game does behind its
 * loading screen: at 2 ms a frame, Sponza's 19,425 probes would take a
 * minute to fill from black. Here `start` only runs the frames, back to back:
 * - frames until the technique says it is warm, a few at a time, then a
 *   wait for the GPU;
 * - then the platform's budget calibration (budget v2): frames with the
 *   millisecond budget, GPU-bound with two in flight, until the platform
 *   reports it calibrated and confirmed, or {@link CALIBRATION_MAX_STEPS}.
 *   Without GPU timestamps the platform can't calibrate and says so at once.
 * `DDGIVolume.restart` runs the same frames again after forgetting the field
 * ("start the probes over").
 */

import { PerspectiveCamera, type Camera, type Scene, type WebGPURenderer } from 'three/webgpu';
import {
    createGI,
    ddgiPreset,
    DDGITechnique,
    DEFAULT_WARM_UP_MAX_SECONDS,
    DEFAULT_WARM_UP_MIN_PASSES,
    DEFAULT_WARM_UP_PASSES,
    GILimitError,
    type DDGIPreset,
    type DDGIPresetName,
    type DDGISettingsInput,
    type EnvironmentDescriptor,
    type FrameBudget,
    type GIPlatform,
    type LoadedField,
    type LoadFieldOptions,
} from '@pmndrs/gi-core';
import { ThreeGI, type GILightOptions, type PmndrsSkyLike } from '@pmndrs/gi-three';
import type { DDGIVolume } from './ddgi-volume.ts';

/** The warm-up's settings, as `start` takes them; the first three become the technique's `warmUpPasses`, `warmUpMaxSeconds`, `warmUpMinPasses`. */
export interface WarmUpSettings {
    /** Full passes over every probe, with no time between them for the history. Default 16. */
    readonly passes: number;
    /** Stop after this long even if the passes aren't done... Default 8. */
    readonly maxSeconds: number;
    /** ...but not before every probe has been updated this many times. Default 2. */
    readonly minPasses: number;
    /** Run the budget calibration after the passes (needs `timestamp-query`). Default true. */
    readonly calibrate: boolean;
}

export const DEFAULT_WARM_UP: WarmUpSettings = { passes: DEFAULT_WARM_UP_PASSES, maxSeconds: DEFAULT_WARM_UP_MAX_SECONDS, minPasses: DEFAULT_WARM_UP_MIN_PASSES, calibrate: true };

/**
 * The calibration stops after this many frames even if the platform is
 * still measuring: a burst gives up after 400 frames and its confirmation
 * after 2 × 300 (`CALIBRATION` in ddgi-core's work-budget.ts).
 */
export const CALIBRATION_MAX_STEPS = 1000;
/** Calibration frames queued at once, so the GPU never waits for the CPU between them. */
const CALIBRATION_FRAMES_IN_FLIGHT = 2;
/** Warm-up frames recorded before waiting for the GPU. Each can be a few thousand probes. */
const WARM_UP_STEPS_PER_BATCH = 4;
/**
 * The warm-up stops after this many frames even if the technique still says
 * it is warming: a guard for a host whose trees never finish building.
 * Sponza's 16 passes take 80 frames.
 */
const WARM_UP_MAX_STEPS = 100_000;

export const DEFAULT_GI_BUDGET: FrameBudget = { gpuMs: 2 };

export interface DDGIVolumeStartOptions {
    /**
     * A quality preset (`'low'`, `'medium'`, `'high'`, or a preset object from
     * `ddgiPreset` with fields changed): its rays and tile sizes, its budget,
     * its texture store size, and its probe spacing for a volume made without
     * probe counts. `budget` and `ddgi` below win over it. Default: none, which
     * is the same as `'high'`. `recommendPreset(adapter.info, adapter.limits)`
     * suggests one.
     */
    readonly preset?: DDGIPresetName | DDGIPreset;
    /** GPU work per frame for GI updates after the warm-up. Default the preset's, or `{ gpuMs: 2 }`. */
    readonly budget?: FrameBudget;
    /** DDGI settings over the preset's and the defaults (rays per probe, hysteresis, ...). The probe grid comes from the volume. */
    readonly ddgi?: Partial<Omit<DDGISettingsInput, 'volume'>>;
    /** An environment for GI rays instead of the scene's (environment.ts). */
    readonly environment?: EnvironmentDescriptor | null;
    /** A pmndrs/sky `Sky` to take GI's sky from (environment.ts). */
    readonly sky?: PmndrsSkyLike | null;
    /** Settings for the GI light. */
    readonly light?: GILightOptions;
    /** The warm-up before `start` resolves (file comment). `false` skips it: GI then fills in live. */
    readonly warmUp?: boolean | Partial<WarmUpSettings>;
    /**
     * The camera the start-up frames run with. The warm-up updates every probe
     * whatever the view, so for one volume this only orders the work; but a
     * camera-following volume's window follows it from the first frame, so a
     * host with such a volume passes the camera it will render with (it may
     * move meanwhile: it is read each frame). Otherwise the window warms up
     * around the volume's centre and jumps to the host's camera on its first
     * frame, which resets every slot and warms the whole field again
     * (several seconds of full-rate updates; NOTES). Default: a camera at the
     * volume's centre.
     */
    readonly camera?: Camera;
    /**
     * A saved bake (`DDGIVolume.saveBake`, a `.ddgibake` file) to start from:
     * loaded right after `prepare`, so there are no warm-up passes (`start`
     * still waits for the ray-tracing trees and runs the calibration). A
     * bake whose layout doesn't fit this volume's DDGI throws
     * `SavedFieldError`, naming what differs; one saved for a scene that
     * changed since loads as stale unless `bakeIfSceneChanged` is `refuse`
     * (`DDGIVolume.loadedBake` says which).
     */
    readonly bake?: Uint8Array | ArrayBuffer;
    /** What a stale `bake` does: `load stale` (the default) or `refuse` (DDGITechnique.loadField). */
    readonly bakeIfSceneChanged?: LoadFieldOptions['ifSceneChanged'];
    /** Where warnings go (each distinct message once). Default `console.warn`. */
    readonly warn?: (message: string) => void;
    /** A label for the platform's GPU objects. Default the volume's name, or `DDGIVolume`. */
    readonly label?: string;
    /**
     * Called once the platform, the technique and `ThreeGI` exist, before the
     * start-up frames run: a host that shows progress (the technique's "warming
     * up" and "warm-up passes" counters) or attaches debug views reads them
     * from here. `start` still owns the frames until it resolves.
     */
    readonly onCreated?: (started: StartedGI) => void;
}

/** What `start` made; the volume keeps it. */
export interface StartedGI {
    readonly platform: GIPlatform;
    readonly technique: DDGITechnique;
    readonly threeGI: ThreeGI;
    /** What `start`'s `bake` loaded (stale or not), or null without one. */
    readonly loadedBake: LoadedField | null;
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
            `@pmndrs/gi/three: the scene needs ${source.limit} = ${source.needed}, but the renderer's device allows ${source.available}. ` +
            'Make the renderer with the platform\'s limits: `const { renderer } = await createGIRenderer()`, or ' +
            '`new WebGPURenderer({ requiredLimits: requiredLimits(await navigator.gpu.requestAdapter()) })` with `requiredLimits` from `@pmndrs/gi/core`.';
    }
}

/** The parts of the platform the warm-up reads that the contract doesn't name yet (reported for the platform). */
interface TreesBuiltSignal {
    rays?: { whenBuilt?: () => Promise<void> };
}

function camerasAt(volume: DDGIVolume): Camera {
    // The warm-up updates every probe, so its camera only orders the work; any view will do
    // (a camera-following volume excepted: `DDGIVolumeStartOptions.camera`).
    const camera = new PerspectiveCamera(60, 1, 0.1, 1000);
    const { center } = volume.ddgiVolume();
    camera.position.set(center[0], center[1], center[2]);
    camera.updateMatrixWorld();
    return camera;
}

/** The warm-up options as the technique's settings. */
function warmUpTechniqueSettings(warmUp: DDGIVolumeStartOptions['warmUp']): Pick<DDGISettingsInput, 'warmUpPasses' | 'warmUpMaxSeconds' | 'warmUpMinPasses'> {
    if (warmUp === false) return { warmUpPasses: 0 };
    const settings = { ...DEFAULT_WARM_UP, ...(typeof warmUp === 'object' ? warmUp : {}) };
    return { warmUpPasses: settings.passes, warmUpMaxSeconds: settings.maxSeconds, warmUpMinPasses: settings.minPasses };
}

/** Resolves in a later JavaScript task, after the browser has sent this task's WebGPU calls. */
function nextTask(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

/** The start-up frames (file comment): the technique's warm-up, then the calibration. */
async function runStartUpFrames(started: StartedGI, camera: Camera, budget: FrameBudget, calibrate: boolean): Promise<void> {
    const { platform, technique, threeGI } = started;
    threeGI.budget = budget;
    for (let step = 1; technique.warmingUp && step <= WARM_UP_MAX_STEPS; step++) {
        threeGI.update(camera, 0);
        if (step % WARM_UP_STEPS_PER_BATCH === 0) await platform.device.queue.onSubmittedWorkDone();
    }
    await platform.device.queue.onSubmittedWorkDone();
    if (!calibrate) return;
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

/**
 * `DDGIVolume.restart` ("start the probes over"): forgets the field and the
 * calibration, then runs the start-up frames again (the warm-up, if the
 * technique's `warmUpPasses` isn't 0, and the calibration). The render-time
 * update is off meanwhile and comes back as it was.
 */
export async function restartVolume(volume: DDGIVolume, started: StartedGI, options: Pick<DDGIVolumeStartOptions, 'budget' | 'warmUp' | 'camera'> = {}): Promise<void> {
    const { platform, threeGI } = started;
    const updateOnRender = threeGI.updateOnRender;
    threeGI.updateOnRender = false;
    try {
        platform.resetHistory();
        platform.recalibrate();
        const calibrate = typeof options.warmUp === 'object' ? options.warmUp.calibrate !== false : true;
        await runStartUpFrames(started, options.camera ?? camerasAt(volume), options.budget ?? threeGI.budget, calibrate);
    } finally {
        threeGI.updateOnRender = updateOnRender;
    }
}

/** `DDGIVolume.start` (file comment). Throws `ThreeGILimitError` when the scene outgrows the device's limits. */
export async function startVolume(volume: DDGIVolume, renderer: WebGPURenderer, scene: Scene, options: DDGIVolumeStartOptions = {}): Promise<StartedGI> {
    const backend = renderer.backend as unknown as { device?: GPUDevice };
    if (!backend.device) await renderer.init();
    const device = backend.device;
    if (!device) throw new Error('@pmndrs/gi/three: DDGIVolume.start needs a WebGPURenderer on WebGPU (it got no GPU device; is it the WebGL fallback?).');
    if (volume.parent === null) scene.add(volume);
    scene.updateMatrixWorld();

    const preset = typeof options.preset === 'string' ? ddgiPreset(options.preset) : (options.preset ?? null);
    if (preset) volume.useDefaultSpacing(preset.probeSpacing);
    const platform = createGI(device, {
        gpuTiming: device.features.has('timestamp-query'),
        label: options.label ?? (volume.name || 'DDGIVolume'),
        textureLayerSize: preset?.textureLayerSize,
    });
    let threeGI: ThreeGI | null = null;
    try {
        // The preset first, so the host's own `ddgi` settings win over it.
        const settings = { ...preset?.ddgi, ...options.ddgi, ...warmUpTechniqueSettings(options.warmUp), volume: volume.ddgiVolume() };
        const technique = platform.addTechnique(new DDGITechnique(settings));
        platform.setOutput(technique);
        // The scene is copied in while the kernels compile (T2.14): `prepare` asks for every
        // compile at once, and the copy is CPU work on this thread, so the two overlap.
        const preparing = platform.prepare();
        // Its failure is thrown by the `await preparing` below; until then it isn't unhandled.
        preparing.catch(() => {});
        // Chrome sends WebGPU calls to its GPU process when the JavaScript task ends, so a
        // long copy in this task would hold the compile requests back until it's done (NOTES).
        await nextTask();
        threeGI = new ThreeGI(platform, scene, {
            // No probe work until the trees exist (file comment); the warm-up sets the real budget.
            budget: { paused: true },
            // Copy the scene in while the kernels compile (three-gi.ts, T2.14).
            preparing,
            volume,
            environment: options.environment,
            sky: options.sky,
            light: options.light,
            warn: options.warn,
            // Switched on after the warm-up: a render loop already running must not add GI frames to it.
            updateOnRender: false,
        });
        const camera = options.camera ?? camerasAt(volume);
        // The scene copy, while the kernels compile (the platform isn't ready: no GI frame yet).
        threeGI.update(camera, 0);
        await preparing;
        // The bake goes in before the first GI frame, so the technique skips its warm-up passes. The scene copy above
        // made the scene the fingerprint is checked against.
        const loadedBake = options.bake ? technique.loadField(options.bake, { ifSceneChanged: options.bakeIfSceneChanged }) : null;
        const started: StartedGI = { platform, technique, threeGI, loadedBake };
        options.onCreated?.(started);
        // The first GI frame uploads the scene and starts the tree builds; the one after them sends the finished trees.
        threeGI.update(camera, 0);
        await (platform as unknown as TreesBuiltSignal).rays?.whenBuilt?.();
        threeGI.update(camera, 0);
        const budget = options.budget ?? (preset ? { ...preset.budget } : DEFAULT_GI_BUDGET);
        if (options.warmUp === false) {
            threeGI.budget = budget;
        } else {
            const calibrate = typeof options.warmUp === 'object' ? options.warmUp.calibrate !== false : DEFAULT_WARM_UP.calibrate;
            await runStartUpFrames(started, camera, budget, calibrate);
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
