/**
 * What DDGI's update costs against how many probes it updates in a frame,
 * whether budget v2 holds (docs/design/ddgi.md, "Budget v2"), and how long
 * start-up takes. Scenes go through the three adapter (`ThreeGI`) exactly as
 * the viewer loads them (`loadScene`), so Sponza and Bistro are the viewer's.
 * First built in T1.9a; the budget sections were rewritten for budget v2.
 *
 * Per scene, in this order:
 *
 * 1. **startup**: load the scene's files, `gi.prepare()` (kernel compiles),
 *    the scene copy and the mesh trees (`rays.whenBuilt`), then the viewer's
 *    warm-up rule (DECISIONS 2026-11-05): steps of `maxProbesPerFrame`
 *    probes with a running-average hysteresis, at most 16 passes over every
 *    probe, stopping after 8 s once 2 passes are done, 4 steps then a wait for
 *    the GPU. Then, as the viewer ends its warm-up, frames with
 *    `budget.gpuMs`, GPU-bound (two in flight, not paced by the display),
 *    until the platform's calibration burst is done
 *    and confirmed (its batch timed over 300 more back-to-back frames). Reported: when each stage was reached, from the start of
 *    the load (file fetch included) and from the technique's creation (where
 *    the viewer's own clock starts): the scene first drawn without GI; the
 *    first warm-up step finished on the GPU; the first frame drawn with GI
 *    (three's pipelines compiled for the GI light, then drawn and finished);
 *    every probe updated once; the warm-up done; the calibration done (its
 *    frames, batch, cost line, the spread of the timings it settled on, and
 *    its filler frames). Kernel compiles are cold only for the
 *    first scene of a run (each run has a fresh browser profile; the driver's
 *    own shader cache may still be warm), so time one scene per run for cold
 *    numbers.
 * 2. **rates**: the calibrated budget at 30, 60 and 120 Hz (frames paced
 *    with timers, nothing waits for the GPU, `deltaTime` the real interval,
 *    no `targetFps`: the platform observes the rate). Judged after the first
 *    RATE_SETTLE_SECONDS: the probes per frame stay within ±5% of their
 *    median, never below the floor, and never step down while frames are on
 *    time. The update's GPU ms here is information only (light load, low
 *    clocks).
 * 3. **budget**: GPU-bound and uncapped (two frames in flight, so the GPU
 *    never waits; as Dennis asked, benchmarks run with the GPU loaded), with
 *    `budget.gpuMs` (default 2). The page fails unless the median update (trace + blend + write) of
 *    frames at the calibrated batch is within ±10% of the budget (DECISIONS,
 *    "Defined measurements", plus a lower bound after the budget v2 review,
 *    so a batch calibrated at low clocks fails too; no lower bound when every
 *    probe fits). On the Intel UHD this is reported, not judged (a
 *    correctness target only, DECISIONS 2026-11-05). Reported: probes per frame, the calibrated
 *    batch and the floor, and the time for a full sweep at 60 Hz.
 * 4. **curve**: fixed batches of 32 ... 4,096 probes, each frame waited for,
 *    one sweep up and one down so a clock drift shows; per batch the median
 *    `ddgi/trace`, `ddgi/blend`, `ddgi/write` and `ddgi/census` GPU ms, and a
 *    fit of the update (trace + blend + write) as fixed + per-probe.
 *
 * Three more sections on request:
 * - **recheck**: `recalibrate()` after the budget section, and the new batch
 *   against the first;
 * - **steady**: a fixed batch at 60 Hz and back to back, with each spike's
 *   start time, to tell our own spikes from other programs' GPU work;
 * - **breakdown**: test-only copies of `ddgi/trace` with parts of the
 *   shading taken out, timed on the same probes (TRACE_VARIANTS).
 *
 * Choose with environment variables (vite passes `VITE_*` to the page):
 *   VITE_DDGI_SCENES=cornell-box,two-room,sponza,bistro-exterior   (default cornell-box,two-room)
 *   VITE_DDGI_SECTIONS=startup,rates,budget,curve,recheck,steady,breakdown   (default startup,rates,budget,curve)
 *   VITE_DDGI_BUDGET_MS=2
 *   VITE_DDGI_STEADY_UNITS=16,512                                    (default: the budget's median batch)
 *
 * Manual (Sponza and Bistro need the downloads). Run it under the bench lock:
 *   node tools/reference/bench-lock.ts run --what "batch cost" -- pnpm test:gpu --filter ddgi-batch-cost --json <file>
 * and turn the JSON into rows with `node tools/gpu/pages/ddgi-batch-cost/to-csv.ts`.
 */

import { PerspectiveCamera, type Scene, type WebGPURenderer } from 'three/webgpu';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { EnvironmentDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { costOnCurve } from '../../../../packages/ddgi-core/src/platform/work-budget.ts';
import { DDGI_TRACE_WGSL, TRACE_WORKGROUP_SIZE } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/trace.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { assert } from '../../client.ts';
import { runThreePage } from '../three-shared/harness.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SCENES = listFromEnv(env.VITE_DDGI_SCENES, ['cornell-box', 'two-room']);
const SECTIONS = new Set(listFromEnv(env.VITE_DDGI_SECTIONS, ['startup', 'rates', 'budget', 'curve']));
const BUDGET_MS = typeof env.VITE_DDGI_BUDGET_MS === 'string' ? Number(env.VITE_DDGI_BUDGET_MS) : 2;
/** Fixed batch sizes for the `steady` section; by default the budget's median batch. */
const STEADY_UNITS = listFromEnv(env.VITE_DDGI_STEADY_UNITS, []).map(Number).filter((units) => units > 0);

/** The viewer's warm-up rule (apps/viewer/src/techniques/ddgi-technique.ts, DECISIONS 2026-11-05). */
const WARMUP = { passes: 16, maxSeconds: 8, minPasses: 2, stepsPerBatch: 4 } as const;

/** Batch sizes for the cost curve, in probes per frame. */
const BATCH_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096] as const;
/** Frames per batch size and sweep: untimed first, then timed. */
const CURVE_UNTIMED_FRAMES = 6;
const CURVE_TIMED_FRAMES = 30;

/** GPU-bound budget frames, each waited for (budget v2: the calibrated batch's cost with the clocks up). */
const BUDGET_FRAMES = 600;
/**
 * The calibrated batch's median update must be within this share of the
 * budget, either way (docs/design/ddgi.md, "Defined measurements": ≤ gpuMs
 * +10% with the clocks up; the lower bound added after the budget v2 review).
 */
const BUDGET_TOLERANCE = 0.1;
/** The calibration frames after the warm-up (burst, then its confirmation) stop after this many, as the viewer's do. */
const CALIBRATION_MAX_STEPS = 600;

/** The platform is measuring: a burst, or the back-to-back frames that confirm it. */
function stillCalibrating(scene: Pick<SceneUnderTest, 'platform' | 'ddgi'>): boolean {
    const report = scene.platform.budgetReport(scene.ddgi);
    return report?.state === 'calibrating' || report?.confirming === true;
}

/** Host frame rates for the `rates` section, and how long each runs. */
const RATES_HZ = [30, 60, 120] as const;
const RATE_SECONDS = 8;
/**
 * The first seconds of each rate are not judged: the platform needs a full
 * second of frames to observe the rate (and so the floor), and the last rate's
 * window to clear.
 */
const RATE_SETTLE_SECONDS = 2.5;
/** "Stable": every judged frame's probes within this share of the median. */
const RATE_STABLE_SHARE = 0.05;

/** The update's own passes; the census is a counter for the stats, reported apart. */
const UPDATE_PASSES = ['ddgi/trace', 'ddgi/blend', 'ddgi/write'] as const;
const CENSUS_PASS = 'ddgi/census';
const PROBES_UPDATED = 'probes updated';

function median(values: readonly number[]): number {
    if (values.length === 0) return NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Nearest-rank percentile, as the viewer's benchmark takes it. */
function percentile(values: readonly number[], share: number): number {
    if (values.length === 0) return NaN;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(share * sorted.length) - 1))];
}

function round(value: number, digits = 4): number {
    return Number.isFinite(value) ? Number(value.toFixed(digits)) : value;
}

/** Least squares `ms = fixedMs + perProbeMs × probes`. */
function fitLine(points: ReadonlyArray<{ probes: number; ms: number }>): { fixedMs: number; perProbeMs: number } {
    const count = points.length;
    const meanX = points.reduce((sum, point) => sum + point.probes, 0) / count;
    const meanY = points.reduce((sum, point) => sum + point.ms, 0) / count;
    let covariance = 0;
    let variance = 0;
    for (const point of points) {
        covariance += (point.probes - meanX) * (point.ms - meanY);
        variance += (point.probes - meanX) ** 2;
    }
    const perProbeMs = variance > 0 ? covariance / variance : NaN;
    return { fixedMs: meanY - perProbeMs * meanX, perProbeMs };
}

function nextAnimationFrame(): Promise<void> {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/**
 * GPU-bound frames: at most this many queued, so the GPU never waits for the
 * CPU between them. Waiting for each frame left the RTX 3060 about 40% busy,
 * and its clocks wandered between power states by up to 2× over seconds
 * (budget v2 rework: the bench lock's clock log had 10 of 439 samples at 70%
 * use or more).
 */
const FRAMES_IN_FLIGHT = 2;

/** Keeps up to FRAMES_IN_FLIGHT frames queued: call `after` once per frame, `drain` at the end. */
class GpuBoundPacing {
    private readonly pending: Promise<void>[] = [];
    private readonly device: GPUDevice;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    async after(): Promise<void> {
        this.pending.push(this.device.queue.onSubmittedWorkDone());
        if (this.pending.length >= FRAMES_IN_FLIGHT) await this.pending.shift();
        await yieldToCallbacks();
    }

    async drain(): Promise<void> {
        await Promise.all(this.pending.splice(0));
    }
}

/** Lets mapAsync callbacks (timestamps, census) run before the next encode collects them. */
function yieldToCallbacks(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

/** The manifest's environment as the viewer gives it to the platform (techniques/ddgi-technique.ts). */
function environmentFor(manifest: SceneManifest): EnvironmentDescriptor {
    const environment = manifest.environment;
    switch (environment.type) {
        case 'none':
            return { type: 'none' };
        case 'color':
            return { type: 'color', color: environment.color, intensity: environment.intensity };
        case 'procedural':
            return platformSkyEnvironment(manifest);
    }
}

/** Every pass timing seen so far, by platform frame and pass name. */
class FrameTimings {
    readonly byFrame = new Map<number, Map<string, number>>();
    readonly unitsByFrame = new Map<number, number>();

    /** After an update: the units it used, and whatever timings have arrived. */
    note(platform: GIPlatformImpl, ddgi: DDGITechnique): void {
        const stats = platform.stats;
        this.unitsByFrame.set(stats.frame, ddgi.counters[PROBES_UPDATED] ?? 0);
        for (const pass of stats.passes) {
            if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
            let passes = this.byFrame.get(pass.frame);
            if (!passes) {
                passes = new Map();
                this.byFrame.set(pass.frame, passes);
            }
            passes.set(pass.name, pass.gpuMs);
        }
    }

    /** The update's GPU ms (trace + blend + write) of a frame, or undefined while not all three are in. */
    updateMs(frame: number): number | undefined {
        const passes = this.byFrame.get(frame);
        if (!passes || !UPDATE_PASSES.every((name) => passes.has(name))) return undefined;
        return UPDATE_PASSES.reduce((sum, name) => sum + (passes.get(name) ?? 0), 0);
    }

    pass(frame: number, name: string): number | undefined {
        return this.byFrame.get(frame)?.get(name);
    }
}

interface SceneUnderTest {
    readonly manifest: SceneManifest;
    readonly platform: GIPlatformImpl;
    readonly ddgi: DDGITechnique;
    readonly threeGI: ThreeGI;
    readonly camera: PerspectiveCamera;
    readonly probeCount: number;
}

/** One frame through the adapter, as the viewer runs it. `deltaTime` 0: back to back, as the viewer's warm-up. */
function frame(scene: SceneUnderTest, budget: ThreeGI['budget'], deltaTime = 1 / 60): void {
    scene.threeGI.budget = budget;
    scene.threeGI.update(scene.camera, deltaTime);
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));
}

/**
 * How long each compute pipeline took to compile, by label: from the
 * `createComputePipelineAsync` call until its promise settled. Pipelines made
 * together compile side by side, so these overlap; the longest one bounds a
 * `prepare`. A pipeline the device already made (same code, earlier scene)
 * comes back almost at once.
 */
const pipelineCompileMs = new Map<string, number>();

function timePipelineCompiles(device: GPUDevice): void {
    const original = device.createComputePipelineAsync.bind(device);
    device.createComputePipelineAsync = async (descriptor: GPUComputePipelineDescriptor) => {
        const startedAt = performance.now();
        try {
            return await original(descriptor);
        } finally {
            const label = descriptor.label ?? '(no label)';
            pipelineCompileMs.set(label, Math.max(pipelineCompileMs.get(label) ?? 0, performance.now() - startedAt));
        }
    };
}

/** Draws one frame of the scene with three, its pipelines compiled first, and waits for the GPU. */
async function drawFrame(device: GPUDevice, renderer: WebGPURenderer, scene: Scene, camera: PerspectiveCamera): Promise<void> {
    await renderer.compileAsync(scene, camera);
    renderer.render(scene, camera);
    await device.queue.onSubmittedWorkDone();
}

async function startScene(
    device: GPUDevice,
    renderer: WebGPURenderer,
    manifest: SceneManifest,
    log: (line: string) => void,
): Promise<{ scene: SceneUnderTest; startup: Record<string, number> }> {
    pipelineCompileMs.clear();
    const loadStartedAt = performance.now();
    const loaded = await loadScene(manifest, () => {});
    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld();
    // The scene on screen without GI, as the viewer shows it while DDGI starts.
    await drawFrame(device, renderer, loaded.scene, camera);
    const sceneShownAt = performance.now();
    const createdAt = performance.now();

    const platform = new GIPlatformImpl(device, { label: `batch cost ${manifest.id}`, gpuTiming: true });
    const volume = manifest.volume;
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing } }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const preparedAt = performance.now();

    const threeGI = new ThreeGI(platform, loaded.scene, { budget: { paused: true }, environment: environmentFor(manifest), warn: () => {} });
    threeGI.update(camera, 0);
    const copiedAt = performance.now();
    await platform.rays.whenBuilt();
    threeGI.update(camera, 0);
    await device.queue.onSubmittedWorkDone();
    const treesAt = performance.now();

    const probeCount = ddgi.field?.grid.probeCount ?? 0;
    const scene: SceneUnderTest = { manifest, platform, ddgi, threeGI, camera, probeCount };

    // The viewer's warm-up (see the file comment).
    const targetHysteresis = ddgi.settings.hysteresis;
    const stepUnits = Math.min(probeCount, ddgi.settings.maxProbesPerFrame);
    let updates = 0;
    let steps = 0;
    let firstGiAt = NaN;
    let firstGiShownAt = NaN;
    let firstPassAt = NaN;
    const warmupStartedAt = performance.now();
    for (;;) {
        const passes = updates / probeCount;
        const completed = Math.floor(passes);
        ddgi.settings.hysteresis = Math.min(targetHysteresis, completed / (completed + 1));
        frame(scene, { units: stepUnits });
        updates += ddgi.counters[PROBES_UPDATED] ?? 0;
        steps++;
        const done = updates / probeCount >= WARMUP.passes || (performance.now() - warmupStartedAt > WARMUP.maxSeconds * 1000 && updates / probeCount >= WARMUP.minPasses);
        // Wait where the viewer waits, and also after the first step and the first full pass to time them.
        const firstStep = steps === 1;
        const firstPass = Number.isNaN(firstPassAt) && updates >= probeCount;
        if (done || firstStep || firstPass || steps % WARMUP.stepsPerBatch === 0) {
            await device.queue.onSubmittedWorkDone();
            if (firstStep) {
                firstGiAt = performance.now();
                // The first frame drawn with GI: three compiles the materials again for the GI light.
                await drawFrame(device, renderer, loaded.scene, camera);
                firstGiShownAt = performance.now();
            }
            if (firstPass) firstPassAt = performance.now();
            await yieldToCallbacks();
        }
        if (done) break;
    }
    ddgi.settings.hysteresis = targetHysteresis;
    const settledAt = performance.now();

    // The calibration (budget v2), as the viewer ends its warm-up: frames with the
    // millisecond budget, GPU-bound (two in flight, deltaTime 0), until the burst is done.
    let calibrationSteps = 0;
    const calibrationPacing = new GpuBoundPacing(device);
    for (;;) {
        frame(scene, { gpuMs: BUDGET_MS }, 0);
        calibrationSteps++;
        if (!stillCalibrating(scene) || calibrationSteps >= CALIBRATION_MAX_STEPS) break;
        await calibrationPacing.after();
    }
    await calibrationPacing.drain();
    await device.queue.onSubmittedWorkDone();
    const calibratedAt = performance.now();
    const calibration = platform.budgetReport(ddgi)?.calibration;
    assert(calibration?.measured, `${manifest.id}: the platform did not calibrate within ${calibrationSteps} steps`);

    const fromLoad = (at: number) => round(at - loadStartedAt, 1);
    const fromCreation = (at: number) => round(at - createdAt, 1);
    const startup = {
        probes: probeCount,
        // From the start of the load: files fetched and parsed, then the first frame drawn without GI.
        load_files_ms: round(sceneShownAt - loadStartedAt, 1),
        scene_shown_from_load_ms: fromLoad(sceneShownAt),
        prepare_ms: round(preparedAt - createdAt, 1),
        scene_copy_cpu_ms: round(copiedAt - preparedAt, 1),
        trees_ms: round(treesAt - copiedAt, 1),
        first_gi_step_from_load_ms: fromLoad(firstGiAt),
        first_gi_shown_from_load_ms: fromLoad(firstGiShownAt),
        first_pass_from_load_ms: fromLoad(firstPassAt),
        settled_from_load_ms: fromLoad(settledAt),
        first_gi_step_from_creation_ms: fromCreation(firstGiAt),
        first_gi_shown_from_creation_ms: fromCreation(firstGiShownAt),
        first_pass_from_creation_ms: fromCreation(firstPassAt),
        settled_from_creation_ms: fromCreation(settledAt),
        warmup_ms: round(settledAt - warmupStartedAt, 1),
        warmup_steps: steps,
        warmup_passes: round(updates / probeCount, 2),
        calibrated_from_load_ms: fromLoad(calibratedAt),
        calibration_ms: round(calibratedAt - settledAt, 1),
        calibration_steps: calibrationSteps,
        calibration_frames: calibration.frames,
        calibrated_probes: calibration.units,
        calibration_fixed_ms: round(calibration.model?.fixedMs ?? NaN),
        calibration_us_per_probe: round((calibration.model?.msPerUnit ?? NaN) * 1000, 3),
        calibration_spread: round(calibration.spread),
        calibration_sizes: calibration.curve.length,
        calibration_filler_frames: calibration.fillerFrames,
        calibration_confirm_factor: round(calibration.confirmFactor ?? NaN),
    };
    // The slowest pipelines this scene's start-up compiled.
    for (const [label, ms] of [...pipelineCompileMs].sort((a, b) => b[1] - a[1]).slice(0, 6)) {
        (startup as Record<string, number>)[`compile_ms/${label}`] = round(ms, 1);
    }
    log(
        `${manifest.id}: ${probeCount} probes; files ${startup.load_files_ms} ms, prepare ${startup.prepare_ms}, copy ${startup.scene_copy_cpu_ms}, trees ${startup.trees_ms}; ` +
            `first GI drawn ${startup.first_gi_shown_from_load_ms} ms from load (${startup.first_gi_shown_from_creation_ms} after creation), first pass ${startup.first_pass_from_creation_ms}, ` +
            `settled ${startup.settled_from_creation_ms} (${startup.warmup_passes} passes in ${startup.warmup_ms} ms); ` +
            `calibrated in ${startup.calibration_ms} ms (${calibration.frames} frames, ${calibration.curve.length} sizes, spread ${startup.calibration_spread}): ${calibration.units} probes, ` +
            `${startup.calibration_fixed_ms} ms + ${startup.calibration_us_per_probe} µs/probe`,
    );
    return { scene, startup };
}

/** The probes a calibrated platform hands out at `framesPerSecond`: the calibrated batch or the floor. */
function probesAt(scene: SceneUnderTest, framesPerSecond: number, calibratedUnits: number): number {
    return Math.max(calibratedUnits, scene.ddgi.minUnitsPerFrame({ framesPerSecond, calibratedUnits }));
}

/**
 * Budget v2, GPU-bound: frames with `budget.gpuMs` back to back, each waited
 * for, `deltaTime` the real interval (a host running uncapped). The check:
 * the calibrated batch's median update is within the budget +10%.
 */
async function measureBudget(device: GPUDevice, scene: SceneUnderTest, log: (line: string) => void): Promise<Record<string, number>> {
    const timings = new FrameTimings();
    const firstFrame = scene.platform.stats.frame + 1;
    const intervals: number[] = [];
    const states = new Map<number, string>();
    let previousAt = performance.now();
    const pacing = new GpuBoundPacing(device);
    for (let index = 0; index < BUDGET_FRAMES; index++) {
        const now = performance.now();
        frame(scene, { gpuMs: BUDGET_MS }, (now - previousAt) / 1000);
        if (index > 0) intervals.push(now - previousAt);
        previousAt = now;
        states.set(scene.platform.stats.frame, scene.platform.budgetReport(scene.ddgi)?.state ?? '');
        timings.note(scene.platform, scene.ddgi);
        await pacing.after();
    }
    await pacing.drain();
    for (let extra = 0; extra < 6; extra++) {
        frame(scene, { paused: true });
        timings.note(scene.platform, scene.ddgi);
        await device.queue.onSubmittedWorkDone();
        await yieldToCallbacks();
    }

    const report = scene.platform.budgetReport(scene.ddgi);
    const calibrated = report?.calibratedUnits;
    assert(calibrated !== undefined, `${scene.manifest.id}: no calibrated batch`);
    const units: number[] = [];
    let calibrating = 0;
    for (let platformFrame = firstFrame; platformFrame < firstFrame + BUDGET_FRAMES; platformFrame++) {
        units.push(timings.unitsByFrame.get(platformFrame) ?? 0);
        if (states.get(platformFrame) === 'calibrating') calibrating++;
    }
    // What the technique ran: the calibrated batch, or fewer when every probe fits (its
    // capacity), or more when the floor binds (then the batch's cost isn't the budget's).
    const ran = median(units);
    const atBatch: number[] = [];
    for (let platformFrame = firstFrame; platformFrame < firstFrame + BUDGET_FRAMES; platformFrame++) {
        const ms = timings.updateMs(platformFrame);
        if (ms !== undefined && timings.unitsByFrame.get(platformFrame) === ran) atBatch.push(ms);
    }
    const steppedDown = units.filter((used) => used < ran).length;
    assert(atBatch.length >= BUDGET_FRAMES / 2, `${scene.manifest.id}: only ${atBatch.length} timed frames at the batch of ${ran}`);
    const atCalibrated = atBatch;
    const probesAt60 = probesAt(scene, 60, calibrated);
    // The most the technique can update in a frame: all its probes, or maxProbesPerFrame.
    const techniqueMax = Math.min(scene.probeCount, scene.ddgi.settings.maxProbesPerFrame);
    const result = {
        budget_ms: BUDGET_MS,
        calibrated_probes: calibrated,
        probes_run: ran,
        // Every probe the technique can update fits in the budget: no lower bound then. A batch
        // that ran below the calibrated one because it stepped down doesn't count as that.
        all_probes_fit: ran >= techniqueMax && calibrated >= techniqueMax ? 1 : 0,
        // The floor is above the calibrated batch: its cost is the floor's, not judged.
        floor_binds: ran > calibrated ? 1 : 0,
        frame_interval_ms_median: round(median(intervals), 3),
        update_ms_median: round(median(atCalibrated)),
        update_ms_p05: round(percentile(atCalibrated, 0.05)),
        update_ms_p95: round(percentile(atCalibrated, 0.95)),
        update_over_budget_share: round(median(atCalibrated) / BUDGET_MS - 1, 4),
        timed_frames: atCalibrated.length,
        probes_median: median(units),
        probes_p05: percentile(units, 0.05),
        probes_p95: percentile(units, 0.95),
        frames_stepped_down: steppedDown,
        frames_calibrating: calibrating,
        floor_probes_at_60hz: scene.ddgi.minUnitsPerFrame({ framesPerSecond: 60, calibratedUnits: calibrated }),
        probes_per_frame_at_60hz: probesAt60,
        // A full pass over every probe at 60 Hz with this batch (or the floor).
        sweep_s_at_60hz: round(Math.ceil(scene.probeCount / probesAt60) / 60, 2),
        calibration_fixed_ms: round(report?.calibration?.model?.fixedMs ?? NaN),
        calibration_us_per_probe: round((report?.calibration?.model?.msPerUnit ?? NaN) * 1000, 3),
    };
    log(
        `${scene.manifest.id}, budget ${BUDGET_MS} ms, GPU-bound: calibrated ${calibrated} probes (ran ${ran}) cost ${result.update_ms_median} ms median ` +
            `(p05 ${result.update_ms_p05}, p95 ${result.update_ms_p95}; ${Math.round(result.update_over_budget_share * 1000) / 10}% against the budget); ` +
            `probes per frame ${result.probes_median} (p05 ${result.probes_p05}, p95 ${result.probes_p95}), ${steppedDown} frames stepped down; ` +
            `at 60 Hz ${probesAt60} probes a frame (floor ${result.floor_probes_at_60hz}), a sweep every ${result.sweep_s_at_60hz} s`,
    );
    return result;
}

/**
 * Opt-in `recheck`: `recalibrate()` after the GPU-bound frames, the burst run
 * the same way as at start-up, and the new batch and its cost on the new
 * curve, next to what the first calibration chose. Tells whether a
 * calibration right after the warm-up reads the GPU differently from one
 * later on.
 */
async function recheckCalibration(device: GPUDevice, scene: SceneUnderTest, firstUnits: number, log: (line: string) => void): Promise<Record<string, number>> {
    scene.platform.recalibrate();
    let steps = 0;
    const pacing = new GpuBoundPacing(device);
    for (;;) {
        frame(scene, { gpuMs: BUDGET_MS }, 0);
        steps++;
        if (!stillCalibrating(scene) || steps >= CALIBRATION_MAX_STEPS) break;
        await pacing.after();
    }
    await pacing.drain();
    const calibration = scene.platform.budgetReport(scene.ddgi)?.calibration;
    assert(calibration?.measured, `${scene.manifest.id}: the re-calibration did not finish in ${steps} frames`);
    const result = {
        first_calibrated_probes: firstUnits,
        recalibrated_probes: calibration.units,
        recalibration_frames: calibration.frames,
        recalibration_spread: round(calibration.spread),
        recalibration_confirm_factor: round(calibration.confirmFactor ?? NaN),
        // The first batch's cost on the new curve: below the budget when the first calibration read slow.
        first_batch_ms_on_new_curve: round(costOnCurve(calibration.curve, firstUnits) ?? NaN),
    };
    log(`${scene.manifest.id} re-check: first ${firstUnits} probes, now ${calibration.units} (${calibration.frames} frames); the first batch now reads ${result.first_batch_ms_on_new_curve} ms`);
    return result;
}

/**
 * Budget v2 at a host frame rate: frames paced by timers at `hz` (nothing
 * waits for the GPU, as in a browser at that refresh rate), `deltaTime` the
 * real interval, no `targetFps`. Returns the numbers and what failed.
 */
async function measureRate(device: GPUDevice, scene: SceneUnderTest, hz: number, log: (line: string) => void): Promise<{ result: Record<string, number>; failures: string[] }> {
    const timings = new FrameTimings();
    const firstFrame = scene.platform.stats.frame + 1;
    const frames = Math.round(hz * RATE_SECONDS);
    const settleFrames = Math.round(hz * RATE_SETTLE_SECONDS);
    const reports: Array<{ steppedDown: boolean; framesLate: boolean; floor: number; units: number; fps: number }> = [];
    const intervals: number[] = [];
    let previousAt = performance.now();
    let due = previousAt;
    for (let index = 0; index < frames; index++) {
        due += 1000 / hz;
        await sleep(due - performance.now());
        const now = performance.now();
        frame(scene, { gpuMs: BUDGET_MS }, (now - previousAt) / 1000);
        intervals.push(now - previousAt);
        previousAt = now;
        const report = scene.platform.budgetReport(scene.ddgi);
        reports.push({ steppedDown: report?.steppedDown ?? false, framesLate: report?.framesLate ?? false, floor: report?.floorUnits ?? 0, units: report?.units ?? 0, fps: report?.framesPerSecond ?? NaN });
        timings.note(scene.platform, scene.ddgi);
    }
    for (let extra = 0; extra < 6; extra++) {
        await device.queue.onSubmittedWorkDone();
        await yieldToCallbacks();
        frame(scene, { paused: true });
        timings.note(scene.platform, scene.ddgi);
    }

    const judged = reports.slice(settleFrames);
    const units = judged.map((report) => report.units);
    const typical = median(units);
    const ms = Array.from({ length: frames - settleFrames }, (_, index) => timings.updateMs(firstFrame + settleFrames + index)).filter((value): value is number => value !== undefined);
    const late = judged.filter((report) => report.framesLate).length;
    const belowFloor = judged.filter((report) => report.units < report.floor).length;
    const steppedDown = judged.filter((report) => report.steppedDown).length;
    const unstable = units.filter((value) => Math.abs(value - typical) > RATE_STABLE_SHARE * typical).length;
    const result = {
        hz,
        frame_interval_ms_median: round(median(intervals.slice(settleFrames)), 3),
        observed_fps: round(judged.at(-1)?.fps ?? NaN, 2),
        probes_median: typical,
        probes_min: Math.min(...units),
        probes_max: Math.max(...units),
        floor_probes: judged.at(-1)?.floor ?? 0,
        calibrated_probes: scene.platform.budgetReport(scene.ddgi)?.calibratedUnits ?? NaN,
        frames_unstable: unstable,
        frames_below_floor: belowFloor,
        frames_stepped_down: steppedDown,
        frames_late: late,
        sweep_s: round(Math.ceil(scene.probeCount / typical) * (median(intervals.slice(settleFrames)) / 1000), 2),
        // Information only: at a light load the GPU clocks down, so these read high.
        update_ms_median: round(median(ms)),
        update_ms_p95: round(percentile(ms, 0.95)),
    };
    const failures: string[] = [];
    if (belowFloor > 0) failures.push(`${scene.manifest.id} at ${hz} Hz: ${belowFloor} frames below the floor`);
    // With frames on time the number must hold; with late frames a step-down is the design working.
    if (late === 0 && (unstable > 0 || steppedDown > 0)) {
        failures.push(`${scene.manifest.id} at ${hz} Hz: frames on time, but ${unstable} frames off the median batch by more than ${RATE_STABLE_SHARE * 100}% and ${steppedDown} stepped down`);
    }
    log(
        `${scene.manifest.id} at ${hz} Hz (observed ${result.observed_fps}): ${typical} probes a frame (${result.probes_min}-${result.probes_max}; calibrated ${result.calibrated_probes}, floor ${result.floor_probes}), ` +
            `a sweep every ${result.sweep_s} s; ${late} late, ${steppedDown} stepped down; update ${result.update_ms_median} ms median (light load, information only)`,
    );
    return { result, failures };
}

/**
 * Spikes with a fixed batch: the same probes-per-frame every frame, so any
 * frame much slower than the median is not the budget's doing. Run at the
 * display's frame rate (`raf`) and back to back with a wait per frame
 * (`waited`). Each frame's start time is kept, so the spikes' spacing in
 * wall-clock time shows whether they follow our frames or something else's
 * clock (another program's GPU work).
 */
async function measureSteady(device: GPUDevice, scene: SceneUnderTest, units: number, log: (line: string) => void): Promise<Record<string, unknown>> {
    const result: Record<string, unknown> = { units };
    for (const cadence of ['raf', 'waited'] as const) {
        const timings = new FrameTimings();
        const startedAt = new Map<number, number>();
        const firstFrame = scene.platform.stats.frame + 1;
        const frames = 360;
        for (let index = 0; index < frames; index++) {
            if (cadence === 'raf') await nextAnimationFrame();
            startedAt.set(scene.platform.stats.frame + 1, performance.now());
            frame(scene, { units });
            timings.note(scene.platform, scene.ddgi);
            if (cadence === 'waited') {
                await device.queue.onSubmittedWorkDone();
                await yieldToCallbacks();
            }
        }
        for (let extra = 0; extra < 6; extra++) {
            await device.queue.onSubmittedWorkDone();
            await yieldToCallbacks();
            frame(scene, { paused: true });
            timings.note(scene.platform, scene.ddgi);
        }
        const samples: Array<{ frame: number; at: number; total: number; trace: number; blend: number; write: number }> = [];
        for (let platformFrame = firstFrame; platformFrame < firstFrame + frames; platformFrame++) {
            const total = timings.updateMs(platformFrame);
            if (total === undefined) continue;
            samples.push({
                frame: platformFrame - firstFrame,
                at: startedAt.get(platformFrame) ?? NaN,
                total,
                trace: timings.pass(platformFrame, 'ddgi/trace') ?? NaN,
                blend: timings.pass(platformFrame, 'ddgi/blend') ?? NaN,
                write: timings.pass(platformFrame, 'ddgi/write') ?? NaN,
            });
        }
        const typical = median(samples.map((sample) => sample.total));
        const spikes = samples.filter((sample) => sample.total > 1.5 * typical + 0.2);
        const gaps = spikes.slice(1).map((spike, index) => spike.at - spikes[index].at);
        const extra = (pass: 'trace' | 'blend' | 'write') => round(median(spikes.map((spike) => spike[pass])) - median(samples.map((sample) => sample[pass])));
        const frameGaps = samples.slice(1).map((sample, index) => sample.at - samples[index].at);
        result[cadence] = {
            timed_frames: samples.length,
            frame_interval_ms_median: round(median(frameGaps), 2),
            update_ms_median: round(typical),
            update_ms_p95: round(percentile(samples.map((sample) => sample.total), 0.95)),
            // Each pass on its own, for the 20-probe breakdown (T1.9).
            trace_ms_median: round(median(samples.map((sample) => sample.trace))),
            blend_ms_median: round(median(samples.map((sample) => sample.blend))),
            write_ms_median: round(median(samples.map((sample) => sample.write))),
            trace_ms_p95: round(percentile(samples.map((sample) => sample.trace), 0.95)),
            blend_ms_p95: round(percentile(samples.map((sample) => sample.blend), 0.95)),
            write_ms_p95: round(percentile(samples.map((sample) => sample.write), 0.95)),
            spike_count: spikes.length,
            spike_share: round(spikes.length / samples.length, 3),
            spike_extra_ms_median: round(median(spikes.map((spike) => spike.total)) - typical),
            spike_extra_trace_ms: extra('trace'),
            spike_extra_blend_ms: extra('blend'),
            spike_extra_write_ms: extra('write'),
            spike_spacing_ms_median: round(median(gaps), 2),
            spike_spacing_ms_p25: round(percentile(gaps, 0.25), 2),
            spike_spacing_ms_p75: round(percentile(gaps, 0.75), 2),
            spike_frames: spikes.slice(0, 30).map((spike) => spike.frame).join(' '),
        };
        log(`${scene.manifest.id} steady ${units} probes (${cadence}): ${JSON.stringify(result[cadence])}`);
    }
    return result;
}

/** Fixed batch sizes, each frame waited for, one sweep up and one down. */
async function measureCurve(device: GPUDevice, scene: SceneUnderTest, log: (line: string) => void): Promise<Record<string, unknown>> {
    const sizes = BATCH_SIZES.filter((size) => size <= scene.probeCount);
    const framesBySize = new Map<number, number[]>(sizes.map((size) => [size, []]));
    const timings = new FrameTimings();
    for (const sweep of [sizes, [...sizes].reverse()]) {
        for (const size of sweep) {
            for (let index = 0; index < CURVE_UNTIMED_FRAMES + CURVE_TIMED_FRAMES; index++) {
                frame(scene, { units: size });
                timings.note(scene.platform, scene.ddgi);
                if (index >= CURVE_UNTIMED_FRAMES) framesBySize.get(size)!.push(scene.platform.stats.frame);
                // Waiting for each frame keeps one batch per measurement. A frame's
                // timestamps are read back from the next encode on and noted then.
                await device.queue.onSubmittedWorkDone();
                await yieldToCallbacks();
            }
        }
    }
    // Let the last frames' timings in.
    for (let extra = 0; extra < 6; extra++) {
        frame(scene, { paused: true });
        timings.note(scene.platform, scene.ddgi);
        await device.queue.onSubmittedWorkDone();
        await yieldToCallbacks();
    }

    const points: Array<{ probes: number; ms: number }> = [];
    const perSize: Record<string, unknown> = {};
    for (const size of sizes) {
        const list = framesBySize.get(size)!.flatMap((platformFrame) => {
            const trace = timings.pass(platformFrame, 'ddgi/trace');
            const blend = timings.pass(platformFrame, 'ddgi/blend');
            const write = timings.pass(platformFrame, 'ddgi/write');
            if (trace === undefined || blend === undefined || write === undefined) return [];
            return [{ trace, blend, write, census: timings.pass(platformFrame, CENSUS_PASS) ?? NaN }];
        });
        assert(list.length >= CURVE_TIMED_FRAMES, `${scene.manifest.id}: only ${list.length} timed frames at ${size} probes`);
        const totals = list.map((sample) => sample.trace + sample.blend + sample.write);
        const total = median(totals);
        points.push({ probes: size, ms: total });
        perSize[String(size)] = {
            trace_ms: round(median(list.map((sample) => sample.trace))),
            blend_ms: round(median(list.map((sample) => sample.blend))),
            write_ms: round(median(list.map((sample) => sample.write))),
            census_ms: round(median(list.map((sample) => sample.census).filter(Number.isFinite))),
            update_ms: round(total),
            update_ms_p95: round(percentile(totals, 0.95)),
            us_per_probe: round((total / size) * 1000, 3),
            frames: list.length,
        };
    }
    const all = fitLine(points);
    const small = fitLine(points.filter((point) => point.probes <= 512));
    const result = {
        sizes: perSize,
        fit_all_fixed_ms: round(all.fixedMs),
        fit_all_us_per_probe: round(all.perProbeMs * 1000, 3),
        fit_small_fixed_ms: round(small.fixedMs),
        fit_small_us_per_probe: round(small.perProbeMs * 1000, 3),
    };
    log(
        `${scene.manifest.id} curve: ` +
            sizes.map((size) => `${size}: ${(perSize[String(size)] as { update_ms: number }).update_ms} ms`).join(', ') +
            `; fit (all) ${result.fit_all_fixed_ms} ms + ${result.fit_all_us_per_probe} µs/probe, (≤ 512) ${result.fit_small_fixed_ms} ms + ${result.fit_small_us_per_probe} µs/probe`,
    );
    return result;
}

/**
 * Where `ddgi/trace`'s time goes: the trace kernel's source with parts of
 * the shading taken out, each timed on the same probes. Test-only copies of
 * the kernel, made by replacing source text; the technique's own kernel is
 * untouched. `full` is the unchanged source, as a check against the
 * technique's own `ddgi/trace` time.
 */
const TRACE_VARIANTS: ReadonlyArray<{ readonly name: string; readonly replace: ReadonlyArray<readonly [string, string]>; readonly workgroupSize?: number }> = [
    { name: 'full', replace: [] },
    // Closest hits only: no surface, no direct light, no bounce (a miss still reads the sky).
    { name: 'closest_only', replace: [['radiance = ddgiShadeRay(ray, hit, random);', 'radiance = vec3f(hit.t, random);']] },
    // Everything but the direct light (its shadow rays).
    { name: 'no_direct', replace: [['let direct = giDirectIrradiance(surface, DDGI_ALL_LIGHT_GROUPS, random);', 'let direct = vec3f(random, 0.0);']] },
    // Everything but the bounce light read from the field.
    {
        name: 'no_bounce',
        replace: [
            [
                `let bounce = ddgiSampleIrradiance(
        surface.position, surface.normal, ray.direction,
        ddgiIrradianceAtlas, ddgiAtlasSampler, ddgiDistanceAtlas, ddgiAtlasSampler, ddgiProbeData,
        ddgi.volumeOrigin, ddgi.volumeSpacing, ddgi.volumeCounts, ddgi.volumeTexels,
    );`,
                'let bounce = vec3f(0.0);',
            ],
        ],
    },
];
const BREAKDOWN_SIZES = [128, 512, 2048] as const;
const BREAKDOWN_REPEATS = 24;

/** `ddgi/trace` group 1, as in ddgi-technique.ts (equal descriptors make interchangeable layouts). */
const TRACE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'breakdown trace group 1',
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
    ],
};

/** The technique's private bind group for `ddgi/trace` (a test tool reaching in; nothing in the library does). */
function techniqueTraceGroup(ddgi: DDGITechnique): GPUBindGroup {
    const resources = (ddgi as unknown as { resources: { rays: { traceGroup: GPUBindGroup } } | null }).resources;
    assert(resources, 'DDGI is not prepared');
    return resources.rays.traceGroup;
}

async function measureTraceBreakdown(device: GPUDevice, scene: SceneUnderTest, log: (line: string) => void): Promise<Record<string, unknown>> {
    const kernels = await Promise.all(
        TRACE_VARIANTS.map(async (variant) => {
            let code = DDGI_TRACE_WGSL;
            for (const [from, to] of variant.replace) {
                assert(code.includes(from), `trace variant ${variant.name}: the source no longer contains its replaced text`);
                code = code.replace(from, to);
            }
            const kernel = await scene.platform.kernels.computeKernel({
                label: `breakdown ${variant.name}`,
                modules: [scene.platform.shade.module],
                code,
                entryPoint: 'ddgiTrace',
                ownGroups: [TRACE_GROUP],
            });
            return { name: variant.name, kernel, workgroupSize: variant.workgroupSize ?? TRACE_WORKGROUP_SIZE };
        }),
    );
    const querySet = device.createQuerySet({ type: 'timestamp', count: kernels.length * 2 });
    const resolveBuffer = device.createBuffer({ size: kernels.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readBuffer = device.createBuffer({ size: kernels.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const raysPerProbe = scene.ddgi.settings.raysPerProbe;
    const result: Record<string, unknown> = {};
    for (const size of BREAKDOWN_SIZES.filter((candidate) => candidate <= scene.probeCount)) {
        const samples = new Map<string, number[]>(kernels.map(({ name }) => [name, []]));
        for (let repeat = 0; repeat < BREAKDOWN_REPEATS; repeat++) {
            // A real frame sets the update list and uniforms for `size` probes...
            frame(scene, { units: size });
            // ...then every variant traces those same probes, one pass each, in rotating order.
            const encoder = device.createCommandEncoder({ label: 'trace breakdown' });
            const group = techniqueTraceGroup(scene.ddgi);
            const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
            const probesX = Math.min(size, maxPerDimension);
            for (let offset = 0; offset < kernels.length; offset++) {
                const index = (offset + repeat) % kernels.length;
                const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: index * 2, endOfPassWriteIndex: index * 2 + 1 } });
                kernels[index].kernel.bind(pass);
                pass.setBindGroup(1, group);
                pass.dispatchWorkgroups(Math.ceil(raysPerProbe / kernels[index].workgroupSize), probesX, Math.ceil(size / probesX));
                pass.end();
            }
            encoder.resolveQuerySet(querySet, 0, kernels.length * 2, resolveBuffer, 0);
            encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, kernels.length * 16);
            device.queue.submit([encoder.finish()]);
            await readBuffer.mapAsync(GPUMapMode.READ);
            const stamps = new BigUint64Array(readBuffer.getMappedRange().slice(0));
            readBuffer.unmap();
            kernels.forEach(({ name }, index) => {
                const ms = Number(stamps[index * 2 + 1] - stamps[index * 2]) / 1e6;
                if (ms >= 0) samples.get(name)!.push(ms);
            });
        }
        const medians = Object.fromEntries([...samples].map(([name, values]) => [`${name}_ms`, round(median(values))]));
        result[String(size)] = medians;
        log(`${scene.manifest.id} trace breakdown at ${size} probes: ${Object.entries(medians).map(([name, ms]) => `${name} ${ms}`).join(', ')}`);
    }
    querySet.destroy();
    resolveBuffer.destroy();
    readBuffer.destroy();
    return result;
}

await runThreePage(async ({ device, renderer, data, log }) => {
    assert(device.features.has('timestamp-query'), 'timing needs the timestamp-query feature');
    timePipelineCompiles(device);
    data.sections = [...SECTIONS].join(',');
    data.userAgent = navigator.userAgent;
    data.browserVersion = await fullBrowserVersion();
    const manifests = loadAllManifests();
    const results: Record<string, unknown> = {};
    data.scenes = results;
    const budgetFailures: string[] = [];
    // The cost check is a performance target, and performance is judged on the target hardware
    // only (an RTX 3060 and up, an M5 Pro). The Intel UHD is a correctness target (DECISIONS
    // 2026-11-05): its numbers are reported, not judged.
    const judgeCost = device.adapterInfo.vendor !== 'intel';
    data.costJudged = judgeCost;
    for (const id of SCENES) {
        const manifest = manifests.find((candidate) => candidate.id === id);
        assert(manifest, `no manifest '${id}'`);
        const { scene, startup } = await startScene(device, renderer, manifest, log);
        const sceneResult: Record<string, unknown> = { probes: scene.probeCount, raysPerProbe: scene.ddgi.settings.raysPerProbe };
        results[id] = sceneResult;
        if (SECTIONS.has('startup')) sceneResult.startup = startup;
        // Rates first, in rising order: the platform remembers the fastest on-time rate it saw for
        // 10 s, so a slower rate straight after the GPU-bound frames would read as late frames.
        if (SECTIONS.has('rates')) {
            const rates: Record<string, unknown> = {};
            for (const hz of RATES_HZ) {
                const { result, failures } = await measureRate(device, scene, hz, log);
                rates[`${hz}hz`] = result;
                budgetFailures.push(...failures);
            }
            sceneResult.rates = rates;
        }
        if (SECTIONS.has('budget')) {
            const budget = await measureBudget(device, scene, log);
            sceneResult.budgetV2 = budget;
            // Checked after the numbers are in the report, so a failed run can still be read.
            const tooSlow = budget.floor_binds !== 1 && budget.update_ms_median > BUDGET_MS * (1 + BUDGET_TOLERANCE);
            const tooFast = budget.floor_binds !== 1 && budget.all_probes_fit !== 1 && budget.update_ms_median < BUDGET_MS * (1 - BUDGET_TOLERANCE);
            if (judgeCost && (tooSlow || tooFast)) {
                budgetFailures.push(`${id}: the calibrated batch (${budget.calibrated_probes} probes) took ${budget.update_ms_median} ms back to back against a ${BUDGET_MS} ms budget (±${BUDGET_TOLERANCE * 100}%)`);
            }
            if (budget.frames_calibrating > 0) budgetFailures.push(`${id}: the platform calibrated again during ${budget.frames_calibrating} GPU-bound frames`);
        }
        if (SECTIONS.has('recheck')) {
            const firstUnits = (startup as { calibrated_probes: number }).calibrated_probes;
            sceneResult.recheck = await recheckCalibration(device, scene, firstUnits, log);
        }
        if (SECTIONS.has('steady')) {
            const budget = sceneResult.budgetV2 as { probes_median?: number } | undefined;
            const sizes = STEADY_UNITS.length > 0 ? STEADY_UNITS : [Math.round(budget?.probes_median ?? 256)];
            const steady: Record<string, unknown> = {};
            for (const units of sizes) steady[String(units)] = await measureSteady(device, scene, units, log);
            sceneResult.steady = steady;
        }
        if (SECTIONS.has('curve')) sceneResult.curve = await measureCurve(device, scene, log);
        if (SECTIONS.has('breakdown')) sceneResult.breakdown = await measureTraceBreakdown(device, scene, log);
        // Probes per state and rays traced, from the GPU census (a few frames old).
        sceneResult.counters = { ...scene.ddgi.counters };
        scene.threeGI.dispose();
        scene.platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
    assert(budgetFailures.length === 0, budgetFailures.join('; '));
});
