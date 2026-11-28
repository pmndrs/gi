/**
 * What DDGI's update costs against how many probes it updates in a frame,
 * whether budget v2 holds (docs/design/ddgi.md, "Budget v2"), and how long
 * start-up takes. Scenes go through the three adapter (`ThreeGI`) exactly as
 * the viewer loads them (`loadScene`), so Sponza and Bistro are the viewer's.
 * First built in T1.9a; the budget sections were rewritten for budget v2.
 *
 * Per scene, in this order:
 *
 * 1. **startup**: load the scene's files, `gi.prepare()` (kernel compiles)
 *    with the scene copy's CPU part alongside it (T2.14, as `volume.start`),
 *    the uploads and the mesh trees (`rays.whenBuilt`: every mesh's first
 *    tree, the GPU's since T2.12), then the warm-up,
 *    the technique's own since T2.4b (`warmUpPasses`; DECISIONS 2026-11-05):
 *    steps of `maxProbesPerFrame` probes with no time between them, at most
 *    16 passes over every probe, stopping after 8 s once 2 passes are done,
 *    4 steps then a wait for the GPU, as `volume.start` runs it. Then frames with
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
 * Four more sections on request:
 * - **saturation**: the step-down on real hardware: at 60 Hz, a synthetic
 *   GPU load (a page-only busy compute pass) makes frames late for 10 s,
 *   then goes; the batch must step down within 4 s, never below the floor,
 *   and recover (SATURATION_TEST);
 * - **recheck**: `recalibrate()` after the budget section, and the new batch
 *   against the first;
 * - **steady**: a fixed batch at 60 Hz and back to back, with each spike's
 *   start time, to tell our own spikes from other programs' GPU work;
 * - **breakdown**: test-only copies of `ddgi/trace` with parts of the
 *   shading taken out, timed on the same probes (TRACE_VARIANTS).
 *
 * Each scene runs once per direct-light path in `VITE_DDGI_DIRECT` (T2.2):
 * `deferred` (the default: `ddgi/trace` plus the platform's `gi/direct`,
 * timed as `ddgi/gi-direct`, both counted in the update) and `inline` (M1's
 * trace with the shadow rays inside). With both listed the paths run one
 * after the other on fresh platforms, inline first, and results are keyed
 * `<scene>/<path>`. With direct light deferred, the budget section also
 * reports the hits `gi/direct` lit per probe and per second (one shadow ray
 * each for a scene lit by one shadow-casting sun, as Sponza is).
 *
 * Choose with environment variables (vite passes `VITE_*` to the page):
 *   VITE_DDGI_SCENES=cornell-box,two-room,sponza,bistro-exterior   (default cornell-box,two-room)
 *   VITE_DDGI_DIRECT=inline,deferred                                (default deferred)
 *   VITE_DDGI_SECTIONS=startup,rates,budget,curve,saturation,recheck,steady,breakdown   (default startup,rates,budget,curve)
 *   VITE_DDGI_BUDGET_MS=2
 *   VITE_DDGI_STEADY_UNITS=16,512                                    (default: the budget's median batch)
 *   VITE_DDGI_MESH_TREES=tinybvh                                     (default gpu: first trees from the GPU, T2.12)
 *   VITE_DDGI_THREE_COMPILE=when-light-joins                         (default after-first-step; see THREE_COMPILE)
 *   VITE_DDGI_RAMP_SECONDS=8                                         (default 0: off; see rampAndRecalibrate)
 *   VITE_DDGI_MAX_TEXTURE_SIZE=1024                                  (default: textures as the files have them; T2.17,
 *                                                                     Bistro exterior's take 5.1 GB, more than a 6 GB GPU holds with DDGI)
 *
 * Manual (Sponza and Bistro need the downloads). Run it under the bench lock:
 *   node tools/reference/bench-lock.ts run --what "batch cost" -- pnpm test:gpu --filter ddgi-batch-cost --json <file>
 * and turn the JSON into rows with `node tools/gpu/pages/ddgi-batch-cost/to-csv.ts`.
 */

import { PerspectiveCamera, type Lighting, type Scene, type WebGPURenderer } from 'three/webgpu';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import type { EnvironmentDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import { DDGITechnique, type DirectLightPath } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { costOnCurve, SATURATION } from '../../../../packages/core/src/platform/work-budget.ts';
import { DDGI_TRACE_WGSL, TRACE_WORKGROUP_SIZE } from '../../../../packages/core/src/techniques/ddgi/wgsl/trace.ts';
import { ThreeGI } from '../../../../packages/three/src/three-gi.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { sceneLighting } from '../../../../apps/viewer/src/render/scene-lighting.ts';
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
/** The direct-light paths to measure, in this order (see the file comment). */
const DIRECT_PATHS: DirectLightPath[] = listFromEnv(env.VITE_DDGI_DIRECT, ['deferred']).map((path) => {
    if (path !== 'inline' && path !== 'deferred') throw new Error(`VITE_DDGI_DIRECT: '${path}' is neither inline nor deferred`);
    return path;
});
/** Fixed batch sizes for the `steady` section; by default the budget's median batch. */
const STEADY_UNITS = listFromEnv(env.VITE_DDGI_STEADY_UNITS, []).map(Number).filter((units) => units > 0);
/** T2.12: `gpu` (default) builds the first mesh trees on the GPU; `tinybvh` is M1's path, for a comparison. */
const MESH_TREES = env.VITE_DDGI_MESH_TREES === 'tinybvh' ? 'tinybvh' : 'gpu';
/** T2.10: seconds of GPU-bound frames after the start-up, then a re-calibration, before the other sections (0: off). */
const RAMP_SECONDS = typeof env.VITE_DDGI_RAMP_SECONDS === 'string' ? Number(env.VITE_DDGI_RAMP_SECONDS) : 0;
/** The fixed batch of the ramp's frames: a heavy, steady load. */
const RAMP_UNITS = 1024;
/** T2.17: scale model textures down to this side at load (`LoadSceneOptions.maxTextureSize`); undefined keeps them. */
const MAX_TEXTURE_SIZE = typeof env.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' ? Number(env.VITE_DDGI_MAX_TEXTURE_SIZE) : undefined;
/**
 * When three compiles its materials for the GI light. `after-first-step`
 * (default): after the first warm-up step, in series, as T2.12's check 1 is
 * judged (lead ruling 2026-11-08: `DDGIVolume.start` can't compile for the
 * host's own render passes, so hosts get it in series). `when-light-joins`:
 * as soon as the light joins the scene (`prepare` resolved), alongside the
 * tree builds, as a host that draws every frame and renders straight to the
 * canvas would; reported for information.
 */
const THREE_COMPILE = env.VITE_DDGI_THREE_COMPILE === 'when-light-joins' ? 'when-light-joins' : 'after-first-step';

/** Warm-up frames recorded before waiting for the GPU, as `volume.start` does (the warm-up itself is the technique's). */
const WARMUP = { stepsPerBatch: 4 } as const;

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

/**
 * The update's own passes; the census is a counter for the stats, reported
 * apart. With direct light deferred, the platform's `gi/direct` pass (timed
 * as DDGI's `ddgi/gi-direct`) is part of the update.
 */
const DIRECT_PASS = 'ddgi/gi-direct';
function updatePasses(direct: DirectLightPath | undefined): readonly string[] {
    return direct === 'inline' ? ['ddgi/trace', 'ddgi/blend', 'ddgi/write'] : ['ddgi/trace', DIRECT_PASS, 'ddgi/blend', 'ddgi/write'];
}
const CENSUS_PASS = 'ddgi/census';
/** DDGI's counter of the hits gi/direct lit in a frame (from the census, a few frames late). */
const DIRECT_HITS = 'hits lit by gi/direct';
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
    private updatePassNames: readonly string[] = updatePasses(undefined);

    /**
     * After an update: the units it used, and whatever timings have arrived.
     * Passes of one name in a frame add up (the deferred trace runs one
     * `ddgi/trace` and one `ddgi/gi-direct` per round of 1,024 probes). The
     * newest frame's list is complete each time, so its sums replace earlier ones.
     */
    note(platform: GIPlatformImpl, ddgi: DDGITechnique): void {
        const stats = platform.stats;
        this.updatePassNames = updatePasses(ddgi.directLight);
        this.unitsByFrame.set(stats.frame, ddgi.counters[PROBES_UPDATED] ?? 0);
        const sums = new Map<number, Map<string, number>>();
        for (const pass of stats.passes) {
            if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
            let passes = sums.get(pass.frame);
            if (!passes) {
                passes = new Map();
                sums.set(pass.frame, passes);
            }
            passes.set(pass.name, (passes.get(pass.name) ?? 0) + pass.gpuMs);
        }
        for (const [platformFrame, passes] of sums) this.byFrame.set(platformFrame, passes);
    }

    /** The update's GPU ms (trace, gi/direct when deferred, blend, write) of a frame, or undefined while not all are in. */
    updateMs(frame: number): number | undefined {
        const passes = this.byFrame.get(frame);
        if (!passes || !this.updatePassNames.every((name) => passes.has(name))) return undefined;
        return this.updatePassNames.reduce((sum, name) => sum + (passes.get(name) ?? 0), 0);
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

/** Draws one frame of the scene with three, its pipelines compiled first, and waits for the GPU. Returns when the compile ended. */
async function drawFrame(device: GPUDevice, renderer: WebGPURenderer, scene: Scene, camera: PerspectiveCamera): Promise<number> {
    await renderer.compileAsync(scene, camera);
    const compiledAt = performance.now();
    renderer.render(scene, camera);
    await device.queue.onSubmittedWorkDone();
    return compiledAt;
}

/** three's own lighting, kept so a scene drawn with clustered lighting doesn't pass it on to the next scene. */
let defaultLighting: Lighting | null = null;

async function startScene(
    device: GPUDevice,
    renderer: WebGPURenderer,
    manifest: SceneManifest,
    direct: DirectLightPath,
    log: (line: string) => void,
): Promise<{ scene: SceneUnderTest; startup: Record<string, number> }> {
    pipelineCompileMs.clear();
    const loadStartedAt = performance.now();
    const loaded = await loadScene(manifest, () => {}, MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE });
    const filesLoadedAt = performance.now();
    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld();
    // three draws the scene's lights as the viewer does: clustered for many shadowless lights
    // (sponza-candles' 1,024 candles). With three's default lighting every material's shader holds
    // every light, and compiling those for 1,024 lights took minutes (T2.3).
    defaultLighting ??= renderer.lighting;
    renderer.lighting = sceneLighting(loaded.directLights, defaultLighting);
    // The scene on screen without GI, as the viewer shows it while DDGI starts.
    const threeCompiledAt = await drawFrame(device, renderer, loaded.scene, camera);
    const sceneShownAt = performance.now();
    const createdAt = performance.now();

    // T2.12: first trees from the GPU (the default), or tinybvh only (M1's path) for a comparison.
    const meshTrees = MESH_TREES === 'tinybvh' ? { gpuFirst: false } : {};
    const platform = new GIPlatformImpl(device, { label: `batch cost ${manifest.id}`, gpuTiming: true, meshTrees });
    const volume = manifest.volume;
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing } }, { directLight: direct }));
    platform.setOutput(ddgi);
    // As `volume.start` runs it since T2.14: the scene copied in (CPU) while the kernels
    // compile, after a task boundary so Chrome has sent the compile requests (NOTES).
    const preparing = platform.prepare();
    preparing.catch(() => {});
    await yieldToCallbacks();
    const threeGI = new ThreeGI(platform, loaded.scene, { budget: { paused: true }, preparing, environment: environmentFor(manifest), warn: () => {} });
    const copyStartedAt = performance.now();
    threeGI.update(camera, 0);
    const copiedWhileCompilingAt = performance.now();
    const copyWhileCompilingMs = copiedWhileCompilingAt - copyStartedAt;
    // T2.10: the scene copy's staging buffers (T2.13), meshes waiting for the GPU to pack them. They
    // peak once the copy has staged every mesh and before the first encode packs them.
    const stagingAfterCopyBytes = platform.sceneCopy.memory.stagingBuffers;
    await preparing;
    const preparedAt = performance.now();
    const stagingBeforeFirstEncodeBytes = platform.sceneCopy.memory.stagingBuffers;

    // The first GI frame: the scene's uploads, and the tree builds start (the GPU's first, T2.12).
    threeGI.update(camera, 0);
    const copiedAt = performance.now();
    const stagingAfterFirstEncodeBytes = platform.sceneCopy.memory.stagingBuffers;
    // The GI light joined the scene when `prepare` resolved (ThreeGI). With `when-light-joins`, three's
    // materials for it start compiling now, alongside the tree builds (after the uploads, which start the
    // GPU's build); by default they compile in series, after the first warm-up step (THREE_COMPILE).
    let threeGiCompileStartedAt = performance.now();
    let threeGiCompiled: Promise<number> | null = THREE_COMPILE === 'when-light-joins' ? renderer.compileAsync(loaded.scene, camera).then(() => performance.now()) : null;
    await platform.rays.whenBuilt();
    const firstTreesAt = performance.now();
    threeGI.update(camera, 0);
    await device.queue.onSubmittedWorkDone();
    const treesAt = performance.now();
    const treesAtFirstGi = platform.rays.meshTreeCounts;

    const probeCount = ddgi.field?.grid.probeCount ?? 0;
    const scene: SceneUnderTest = { manifest, platform, ddgi, threeGI, camera, probeCount };

    // The warm-up (see the file comment): the technique's own since T2.4b, run as volume.start runs it.
    let updates = 0;
    let steps = 0;
    let firstGiAt = NaN;
    let firstGiShownAt = NaN;
    let firstPassAt = NaN;
    const warmupStartedAt = performance.now();
    for (;;) {
        frame(scene, { gpuMs: BUDGET_MS }, 0);
        updates += ddgi.counters[PROBES_UPDATED] ?? 0;
        steps++;
        const done = !ddgi.warmingUp;
        // Wait where the viewer waits, and also after the first step and the first full pass to time them.
        const firstStep = steps === 1;
        const firstPass = Number.isNaN(firstPassAt) && updates >= probeCount;
        if (done || firstStep || firstPass || steps % WARMUP.stepsPerBatch === 0) {
            await device.queue.onSubmittedWorkDone();
            if (firstStep) {
                firstGiAt = performance.now();
                // The first frame drawn with GI: three's materials for the GI light (compiled now, or since the light joined).
                if (!threeGiCompiled) {
                    threeGiCompileStartedAt = performance.now();
                    threeGiCompiled = renderer.compileAsync(loaded.scene, camera).then(() => performance.now());
                }
                await threeGiCompiled;
                await drawFrame(device, renderer, loaded.scene, camera);
                firstGiShownAt = performance.now();
            }
            if (firstPass) firstPassAt = performance.now();
            await yieldToCallbacks();
        }
        if (done) break;
    }
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

    const threeGiCompileEndedAt = threeGiCompiled ? await threeGiCompiled : NaN;
    const fromLoad = (at: number) => round(at - loadStartedAt, 1);
    const fromCreation = (at: number) => round(at - createdAt, 1);
    const startup = {
        probes: probeCount,
        // From the start of the load: files fetched and parsed, then the first frame drawn without GI.
        // load_files_ms is the whole stage (as before T2.3); its three parts follow.
        load_files_ms: round(sceneShownAt - loadStartedAt, 1),
        load_fetch_parse_ms: round(filesLoadedAt - loadStartedAt, 1),
        load_three_compile_ms: round(threeCompiledAt - filesLoadedAt, 1),
        load_three_first_draw_ms: round(sceneShownAt - threeCompiledAt, 1),
        scene_shown_from_load_ms: fromLoad(sceneShownAt),
        // Since T2.14 the scene copy's CPU part overlaps the compiles: `prepare_ms` is from the
        // platform's creation to `prepare` resolved, that copy included; `compile_wait_ms` is
        // what was left to wait for after it; `scene_copy_cpu_ms` is the copy while compiling
        // plus the first GI frame (the uploads) after `prepare`.
        prepare_ms: round(preparedAt - createdAt, 1),
        compile_wait_ms: round(preparedAt - copiedWhileCompilingAt, 1),
        scene_copy_while_compiling_ms: round(copyWhileCompilingMs, 1),
        scene_copy_cpu_ms: round(copyWhileCompilingMs + (copiedAt - preparedAt), 1),
        // T2.10: SceneMemory.stagingBuffers (MB, 1e6 bytes) after the copy, before and after the first encode.
        staging_after_copy_mb: round(stagingAfterCopyBytes / 1e6, 1),
        staging_before_first_encode_mb: round(stagingBeforeFirstEncodeBytes / 1e6, 1),
        staging_after_first_encode_mb: round(stagingAfterFirstEncodeBytes / 1e6, 1),
        // T2.12's stages from `prepare` to the first frame drawn with GI. The first GI frame (the
        // uploads, the GPU's tree build submitted, tinybvh's copies sent to the worker); then the wait
        // for every mesh's first tree; then the frame that places them; then the first warm-up step
        // on the GPU; then three's first frame with the GI light (`first_gi_draw_ms`), which includes
        // its material compile (`three_gi_compile_ms`) in series by default, or only what the draw still
        // waited for it (`three_gi_compile_wait_ms`) when it started as the light joined.
        first_gi_frame_ms: round(copiedAt - preparedAt, 1),
        trees_ms: round(treesAt - copiedAt, 1),
        first_trees_wait_ms: round(firstTreesAt - copiedAt, 1),
        trees_placed_frame_ms: round(treesAt - firstTreesAt, 1),
        first_gi_step_ms: round(firstGiAt - treesAt, 1),
        first_gi_draw_ms: round(firstGiShownAt - firstGiAt, 1),
        three_gi_compile_ms: round(threeGiCompileEndedAt - threeGiCompileStartedAt, 1),
        three_gi_compile_wait_ms: round(Math.max(0, threeGiCompileEndedAt - firstGiAt), 1),
        three_gi_compile_when_light_joins: THREE_COMPILE === 'when-light-joins' ? 1 : 0,
        gpu_first_trees: platform.rays.gpuFirstTrees ? 1 : 0,
        gpu_tree_build_ms: round(treesAtFirstGi.lastGpuBuildMs, 1),
        meshes_on_gpu_trees_at_first_gi: treesAtFirstGi.onGpuTrees,
        meshes_on_tinybvh_trees_at_first_gi: treesAtFirstGi.onTinybvhTrees,
        gpu_trees_refused: treesAtFirstGi.gpuTreesRefused,
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
    // The slowest pipelines this scene's start-up compiled, and every sky bake kernel (T2.6a's check 3).
    const sorted = [...pipelineCompileMs].sort((a, b) => b[1] - a[1]);
    // How many compute pipelines the start-up made (T2.14: only what the configuration uses).
    (startup as Record<string, number>).pipelines_compiled = sorted.filter(([label]) => label.startsWith(`batch cost ${manifest.id} `)).length;
    for (const [label, ms] of [...sorted.slice(0, 6), ...sorted.slice(6).filter(([label]) => label.includes(' sky '))]) {
        (startup as Record<string, number>)[`compile_ms/${label}`] = round(ms, 1);
    }
    log(
        `${manifest.id} (${direct} direct light): ${probeCount} probes; files ${startup.load_files_ms} ms (fetch and parse ${startup.load_fetch_parse_ms}, three compile ${startup.load_three_compile_ms}, first draw ${startup.load_three_first_draw_ms}), prepare ${startup.prepare_ms}, copy ${startup.scene_copy_cpu_ms}, trees ${startup.trees_ms}; ` +
            `T2.12 stages: first GI frame ${startup.first_gi_frame_ms}, first trees ${startup.first_trees_wait_ms} (GPU build ${startup.gpu_tree_build_ms}; ${startup.meshes_on_gpu_trees_at_first_gi} on GPU trees, ${startup.meshes_on_tinybvh_trees_at_first_gi} on tinybvh's (the fallback), ${startup.gpu_trees_refused} refused), ` +
            `placed ${startup.trees_placed_frame_ms}, first step ${startup.first_gi_step_ms}, draw ${startup.first_gi_draw_ms} (three's GI compile ${startup.three_gi_compile_ms}, waited ${startup.three_gi_compile_wait_ms}); ` +
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
    // gi/direct at the batch, and the hits it lit (the census's latest frame, which ran the same batch).
    const directMs: number[] = [];
    for (let platformFrame = firstFrame; platformFrame < firstFrame + BUDGET_FRAMES; platformFrame++) {
        const ms = timings.pass(platformFrame, DIRECT_PASS);
        if (ms !== undefined && timings.unitsByFrame.get(platformFrame) === ran) directMs.push(ms);
    }
    const directHits = scene.ddgi.counters[DIRECT_HITS];
    const directMedianMs = directMs.length > 0 ? median(directMs) : NaN;
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
        gi_direct_ms_median: round(directMedianMs),
        gi_direct_hits_per_probe: directHits === undefined ? NaN : round(directHits / ran, 2),
        // Hits lit per second inside gi/direct (one shadow ray each under a single shadow-casting sun).
        gi_direct_mhits_per_s: directHits === undefined ? NaN : round(directHits / (directMedianMs * 1000), 2),
    };
    log(
        `${scene.manifest.id}, budget ${BUDGET_MS} ms, GPU-bound: calibrated ${calibrated} probes (ran ${ran}) cost ${result.update_ms_median} ms median ` +
            `(p05 ${result.update_ms_p05}, p95 ${result.update_ms_p95}; ${Math.round(result.update_over_budget_share * 1000) / 10}% against the budget); ` +
            `probes per frame ${result.probes_median} (p05 ${result.probes_p05}, p95 ${result.probes_p95}), ${steppedDown} frames stepped down; ` +
            `at 60 Hz ${probesAt60} probes a frame (floor ${result.floor_probes_at_60hz}), a sweep every ${result.sweep_s_at_60hz} s` +
            (Number.isFinite(directMedianMs) ? `; gi/direct ${result.gi_direct_ms_median} ms, ${result.gi_direct_hits_per_probe} hits a probe, ${result.gi_direct_mhits_per_s} M hits/s` : ''),
    );
    return result;
}

/**
 * Opt-in ramp (T2.10, `VITE_DDGI_RAMP_SECONDS`): GPU-bound frames of RAMP_UNITS
 * probes for that many seconds, then `recalibrate()` with its burst run as at
 * start-up, so the sections after it run at a batch calibrated with the clocks
 * up. *Why:* on the RTX laptop under Chrome 157 the start-up's bursty load
 * leaves the GPU in a low-power state (about 780 MHz under load, nvidia-smi),
 * and the start-up's calibration then reads it slow (Sponza 154 probes at
 * 2 ms against 640 at 1,800-1,900 MHz in T2.2). Budget v2 calibrates with the
 * clocks up (DECISIONS 2026-11-05), and SPEC's timings are GPU-bound. The
 * start-up's own calibration is still reported in `startup`.
 */
async function rampAndRecalibrate(device: GPUDevice, scene: SceneUnderTest, firstUnits: number, log: (line: string) => void): Promise<Record<string, number>> {
    const pacing = new GpuBoundPacing(device);
    const startedAt = performance.now();
    let rampFrames = 0;
    while (performance.now() - startedAt < RAMP_SECONDS * 1000) {
        frame(scene, { units: RAMP_UNITS }, 0);
        rampFrames++;
        await pacing.after();
    }
    await pacing.drain();
    const rampMs = performance.now() - startedAt;
    const recheck = await recheckCalibration(device, scene, firstUnits, log);
    log(`${scene.manifest.id} ramp: ${rampFrames} frames of ${RAMP_UNITS} probes in ${round(rampMs / 1000, 1)} s, then re-calibrated: ${recheck.recalibrated_probes} probes (start-up's ${firstUnits})`);
    return { ramp_seconds: round(rampMs / 1000, 2), ramp_frames: rampFrames, ramp_units: RAMP_UNITS, ...recheck };
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
 * The `saturation` section's phases: on time, then a synthetic GPU load that
 * makes frames late, then on time again.
 *
 * - The rate is the first of `rates` the page's own loop holds without the
 *   load (a 2 s probe, no `targetFps`, so nothing steps down): on the Intel
 *   UHD the loop itself takes about 21-23 ms a frame, so 60 Hz would be late
 *   from the start and the recovery could never count as on time.
 * - The load is sized in the frame loop, from measured frame times, until
 *   loaded frames take `loadIntervals` display intervals (within
 *   `loadTolerance`, two rounds in a row). Why: sized alone on an idle GPU,
 *   the RTX's load read 30 ms at low clocks and a few ms once its clocks
 *   rose, so loaded frames took only 18 ms (first RTX run). While it is
 *   sized, `targetFps` is a third of the rate, so nothing steps down yet.
 * - The load lasts `loadSeconds`, enough steps to reach the floor on every
 *   test scene.
 * - The recovery has as long as the controller needs from the lowest batch
 *   to `recoveredShare` × the calibrated one, from its own constants
 *   (SATURATION: `recoverAfterSeconds`, then ×`recovery` a second), plus
 *   `recoveryMarginSeconds`; at most `maxAfterSeconds`.
 */
const SATURATION_TEST = {
    rates: [60, 30, 20],
    probeSeconds: 2,
    /** A rate is held when the probe's median frame is within this share of its interval. */
    heldShare: 0.05,
    beforeSeconds: 3,
    loadSeconds: 10,
    /** Frame time under the load, in display intervals: two, so every frame is late. */
    loadIntervals: 2,
    loadTolerance: 0.15,
    sizingRoundSeconds: 0.5,
    maxSizingRounds: 20,
    /** Loaded frames must reach at least this many intervals, or the test can't judge anything. */
    minLoadIntervals: 1.5,
    /** Recovered: the batch back to at least this share of the calibrated one (its ceiling is 0.9). */
    recoveredShare: 0.85,
    recoveryMarginSeconds: 10,
    maxAfterSeconds: 90,
    /** The step-down must start within this long of the load. */
    stepDownWithinSeconds: 4,
} as const;

type SaturationPhase = 'probe' | 'sizing' | 'before' | 'load' | 'after';
interface SaturationSample {
    readonly seconds: number;
    readonly phase: SaturationPhase;
    readonly units: number;
    readonly floor: number;
    readonly frameMs: number;
}

/**
 * Frames at `hz` for `seconds`, each waited for as a host's display would make
 * it wait, with the synthetic load when `load` is given, and `targetFps` when
 * given (the probe runs without one).
 */
async function saturationFrames(
    device: GPUDevice,
    scene: SceneUnderTest,
    hz: number,
    seconds: number,
    phase: SaturationPhase,
    options: { readonly load?: SyntheticGpuLoad; readonly targetFps?: number; readonly startedAt: number },
    samples: SaturationSample[],
): Promise<void> {
    const endAt = performance.now() + seconds * 1000;
    let previousAt = performance.now();
    let due = previousAt;
    while (performance.now() < endAt) {
        // The next display slot; a frame that ran past it starts at once (late), and the
        // slots go on from there. A frame that fits waits for its slot.
        due = Math.max(due + 1000 / hz, performance.now());
        await sleep(due - performance.now());
        const now = performance.now();
        frame(scene, options.targetFps === undefined ? { gpuMs: BUDGET_MS } : { gpuMs: BUDGET_MS, targetFps: options.targetFps }, (now - previousAt) / 1000);
        options.load?.submit();
        // A display makes the host wait for its GPU: a frame that takes longer than the interval is late.
        await device.queue.onSubmittedWorkDone();
        await yieldToCallbacks();
        const report = scene.platform.budgetReport(scene.ddgi);
        samples.push({ seconds: (now - options.startedAt) / 1000, phase, units: report?.units ?? 0, floor: report?.floorUnits ?? 0, frameMs: now - previousAt });
        previousAt = now;
    }
}

/** A page-only compute pass that keeps the GPU busy for about `loadMs`: the synthetic load. */
const BUSY_LOAD_WGSL = /* wgsl */ `
struct BusyParams { iterations: u32 }
@group(0) @binding(0) var<uniform> params: BusyParams;
@group(0) @binding(1) var<storage, read_write> sink: array<u32>;

// A chain of dependent integer steps per thread, so the work can't be skipped or shortened.
@compute @workgroup_size(64)
fn busy(@builtin(global_invocation_id) id: vec3u) {
    var value = id.x;
    for (var step = 0u; step < params.iterations; step++) {
        value = value * 1664525u + 1013904223u;
    }
    sink[id.x] = value;
}
`;
const BUSY_LOAD_WORKGROUPS = 1024;

/** The synthetic load: one dispatch per call to `submit`, sized at start to take about `loadMs`. */
class SyntheticGpuLoad {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly params: GPUBuffer;
    private readonly sink: GPUBuffer;
    private readonly group: GPUBindGroup;
    iterations = 4096;

    constructor(device: GPUDevice) {
        this.device = device;
        const module = device.createShaderModule({ label: 'saturation test: busy load', code: BUSY_LOAD_WGSL });
        this.pipeline = device.createComputePipeline({ label: 'saturation test: busy load', layout: 'auto', compute: { module, entryPoint: 'busy' } });
        this.params = device.createBuffer({ label: 'saturation test: busy params', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.sink = device.createBuffer({ label: 'saturation test: busy sink', size: BUSY_LOAD_WORKGROUPS * 64 * 4, usage: GPUBufferUsage.STORAGE });
        this.group = device.createBindGroup({
            label: 'saturation test: busy load',
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.params } },
                { binding: 1, resource: { buffer: this.sink } },
            ],
        });
    }

    submit(): void {
        this.device.queue.writeBuffer(this.params, 0, new Uint32Array([this.iterations]));
        const encoder = this.device.createCommandEncoder({ label: 'saturation test: busy load' });
        const pass = encoder.beginComputePass({ label: 'saturation test: busy load' });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.group);
        pass.dispatchWorkgroups(BUSY_LOAD_WORKGROUPS);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    /** A first guess: scales the iterations until one dispatch alone takes about `targetMs` of wall time (median of 3). The frame loop then sizes it for real. */
    async size(targetMs: number): Promise<number> {
        let ms = 0;
        for (let round = 0; round < 4; round++) {
            const times: number[] = [];
            for (let repeat = 0; repeat < 3; repeat++) {
                const startedAt = performance.now();
                this.submit();
                await this.device.queue.onSubmittedWorkDone();
                times.push(performance.now() - startedAt);
            }
            ms = median(times);
            if (Math.abs(ms - targetMs) < 0.15 * targetMs) break;
            this.iterations = Math.max(64, Math.round((this.iterations * targetMs) / Math.max(ms, 0.05)));
        }
        return ms;
    }

    dispose(): void {
        this.params.destroy();
        this.sink.destroy();
    }
}

/**
 * Budget v2's saturation step-down on real hardware (the measurement "a
 * saturation step-down that recovers", docs/design/ddgi.md). At the first
 * rate the loop holds (SATURATION_TEST; timer-paced, with `targetFps`: timer
 * pacing isn't steady enough to be learned as a display rate), each frame
 * waited for as a host's display would make it wait: on time, then with the
 * synthetic load (frames late), then without it. Judged: the load made frames
 * late (at least `minLoadIntervals`), no step-down before the load, a
 * step-down within `stepDownWithinSeconds` of it, never below the floor, and
 * back to `recoveredShare` of the calibrated batch within the time the
 * controller's own constants allow.
 */
async function measureSaturation(device: GPUDevice, scene: SceneUnderTest, log: (line: string) => void): Promise<{ result: Record<string, unknown>; failures: string[] }> {
    const calibrated = scene.platform.budgetReport(scene.ddgi)?.calibratedUnits;
    assert(calibrated !== undefined, `${scene.manifest.id}: no calibrated batch before the saturation test`);
    const samples: SaturationSample[] = [];
    const startedAt = performance.now();
    // The rate: the first one the loop holds without the load (see SATURATION_TEST).
    let hz: number = SATURATION_TEST.rates[SATURATION_TEST.rates.length - 1];
    for (const rate of SATURATION_TEST.rates) {
        const probe: SaturationSample[] = [];
        await saturationFrames(device, scene, rate, SATURATION_TEST.probeSeconds, 'probe', { startedAt }, probe);
        samples.push(...probe);
        if (median(probe.slice(1).map((sample) => sample.frameMs)) <= (1000 / rate) * (1 + SATURATION_TEST.heldShare)) {
            hz = rate;
            break;
        }
    }
    const intervalMs = 1000 / hz;
    const load = new SyntheticGpuLoad(device);
    const loadAloneMs = await load.size(SATURATION_TEST.loadIntervals * intervalMs);
    // Size the load from loaded frames' times (see SATURATION_TEST), with a target of a third
    // of the rate so nothing steps down while it is sized.
    const wantedMs = SATURATION_TEST.loadIntervals * intervalMs;
    let sizingRounds = 0;
    let agreeingRounds = 0;
    let sizedFrameMs = 0;
    while (sizingRounds < SATURATION_TEST.maxSizingRounds && agreeingRounds < 2) {
        sizingRounds++;
        const round: SaturationSample[] = [];
        await saturationFrames(device, scene, hz, SATURATION_TEST.sizingRoundSeconds, 'sizing', { load, targetFps: hz / (SATURATION_TEST.loadIntervals + 1), startedAt }, round);
        samples.push(...round);
        sizedFrameMs = median(round.slice(1).map((sample) => sample.frameMs));
        if (Math.abs(sizedFrameMs - wantedMs) <= SATURATION_TEST.loadTolerance * wantedMs) {
            agreeingRounds++;
        } else {
            agreeingRounds = 0;
            load.iterations = Math.max(64, Math.round((load.iterations * wantedMs) / Math.max(sizedFrameMs, 1)));
        }
    }
    // Two of the controller's windows without the load, still at the lenient target: the window
    // open when the sizing ends holds loaded frames, which against the real target read as late
    // (the first Intel run of this version stepped down before the load that way).
    await saturationFrames(device, scene, hz, 2 * SATURATION.windowSeconds, 'sizing', { targetFps: hz / (SATURATION_TEST.loadIntervals + 1), startedAt }, samples);
    await saturationFrames(device, scene, hz, SATURATION_TEST.beforeSeconds, 'before', { targetFps: hz, startedAt }, samples);
    await saturationFrames(device, scene, hz, SATURATION_TEST.loadSeconds, 'load', { load, targetFps: hz, startedAt }, samples);
    load.dispose();
    // As long as the controller needs to recover from the lowest batch under the load (SATURATION).
    const lowestUnderLoad = Math.max(1, Math.min(...samples.filter((sample) => sample.phase === 'load').map((sample) => sample.units)));
    const recoverySteps = Math.max(0, Math.ceil(Math.log((SATURATION_TEST.recoveredShare * calibrated) / lowestUnderLoad) / Math.log(SATURATION.recovery)));
    const recoverySeconds = SATURATION.recoverAfterSeconds + recoverySteps * SATURATION.windowSeconds;
    const afterSeconds = Math.min(SATURATION_TEST.maxAfterSeconds, recoverySeconds + SATURATION_TEST.recoveryMarginSeconds);
    await saturationFrames(device, scene, hz, afterSeconds, 'after', { targetFps: hz, startedAt }, samples);

    const inPhase = (phase: SaturationPhase) => samples.filter((sample) => sample.phase === phase);
    const loadStart = inPhase('load')[0]?.seconds ?? Number.NaN;
    const judged = samples.filter((sample) => sample.phase !== 'probe' && sample.phase !== 'sizing');
    const steppedBeforeLoad = inPhase('before').filter((sample) => sample.units < calibrated).length;
    const firstStep = samples.find((sample) => (sample.phase === 'load' || sample.phase === 'after') && sample.units < calibrated);
    const lowest = Math.min(...judged.map((sample) => sample.units));
    const belowFloor = judged.filter((sample) => sample.units < sample.floor).length;
    const after = inPhase('after');
    const recoveredAt = after.find((sample) => sample.units >= SATURATION_TEST.recoveredShare * calibrated);
    const result = {
        hz,
        calibrated_probes: calibrated,
        floor_probes: samples.at(-1)?.floor ?? 0,
        frames_stepped_down_before_load: steppedBeforeLoad,
        load_alone_ms: round(loadAloneMs, 2),
        load_iterations: load.iterations,
        sizing_rounds: sizingRounds,
        sizing_frame_ms: round(sizedFrameMs, 2),
        // What the load achieved: the median frame under it, against `loadIntervals` × the interval.
        load_frame_intervals: round(median(inPhase('load').map((sample) => sample.frameMs)) / intervalMs, 2),
        recovery_allowed_s: round(afterSeconds, 1),
        frame_ms_median_before: round(median(inPhase('before').map((sample) => sample.frameMs)), 2),
        frame_ms_median_load: round(median(inPhase('load').map((sample) => sample.frameMs)), 2),
        frame_ms_median_after: round(median(after.slice(Math.floor(after.length / 2)).map((sample) => sample.frameMs)), 2),
        step_down_after_s: firstStep ? round(firstStep.seconds - loadStart, 2) : -1,
        lowest_probes: lowest,
        frames_below_floor: belowFloor,
        recovered_after_s: recoveredAt ? round(recoveredAt.seconds - (after[0]?.seconds ?? 0), 2) : -1,
        final_probes: samples.at(-1)?.units ?? 0,
        // The batch once a second: what happened over time.
        probes_per_second: judged.filter((_, index) => index % hz === 0).map((sample) => `${Math.round(sample.seconds)}:${sample.units}`).join(' '),
    };
    const failures: string[] = [];
    if (result.load_frame_intervals < SATURATION_TEST.minLoadIntervals) {
        failures.push(`${scene.manifest.id}: the synthetic load made frames only ${result.load_frame_intervals} intervals long (needs ${SATURATION_TEST.minLoadIntervals}); nothing to judge`);
    }
    if (steppedBeforeLoad > 0) failures.push(`${scene.manifest.id}: stepped down before the load, at ${hz} Hz on time (frames ${result.frame_ms_median_before} ms)`);
    if (!firstStep || firstStep.seconds - loadStart > SATURATION_TEST.stepDownWithinSeconds) {
        failures.push(`${scene.manifest.id}: no step-down within ${SATURATION_TEST.stepDownWithinSeconds} s of the load (frames ${result.frame_ms_median_load} ms against ${round(1000 / hz, 2)})`);
    }
    if (belowFloor > 0) failures.push(`${scene.manifest.id}: ${belowFloor} frames below the floor during the saturation test`);
    if (!recoveredAt) failures.push(`${scene.manifest.id}: no recovery to ${SATURATION_TEST.recoveredShare} × the calibrated batch within ${result.recovery_allowed_s} s of the load ending (from ${lowestUnderLoad}; ended at ${result.final_probes})`);
    log(
        `${scene.manifest.id} saturation at ${hz} Hz (load sized in ${sizingRounds} rounds, frames under it ${result.load_frame_intervals} intervals): frames ${result.frame_ms_median_before} → ${result.frame_ms_median_load} → ${result.frame_ms_median_after} ms; ` +
            `stepped down after ${result.step_down_after_s} s to as low as ${lowest} (floor ${result.floor_probes}), recovered after ${result.recovered_after_s} s; ${result.probes_per_second}`,
    );
    return { result, failures };
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
            const total = timings.updateMs(platformFrame);
            const trace = timings.pass(platformFrame, 'ddgi/trace');
            const blend = timings.pass(platformFrame, 'ddgi/blend');
            const write = timings.pass(platformFrame, 'ddgi/write');
            if (total === undefined || trace === undefined || blend === undefined || write === undefined) return [];
            return [{ total, trace, direct: timings.pass(platformFrame, DIRECT_PASS) ?? NaN, blend, write, census: timings.pass(platformFrame, CENSUS_PASS) ?? NaN }];
        });
        assert(list.length >= CURVE_TIMED_FRAMES, `${scene.manifest.id}: only ${list.length} timed frames at ${size} probes`);
        const totals = list.map((sample) => sample.total);
        const total = median(totals);
        points.push({ probes: size, ms: total });
        perSize[String(size)] = {
            trace_ms: round(median(list.map((sample) => sample.trace))),
            gi_direct_ms: round(median(list.map((sample) => sample.direct).filter(Number.isFinite))),
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
                `let bounce = ddgiSampleVolumes(
        surface.position, surface.normal, ray.direction,
        ddgiIrradianceAtlas, ddgiAtlasSampler, ddgiDistanceAtlas, ddgiAtlasSampler, ddgiProbeData, ddgiVolumeTable,
    ).irradiance;`,
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
        { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
    ],
};

/** The technique's private bind group for `ddgi/trace`, its first volume's (a test tool reaching in; nothing in the library does). */
function techniqueTraceGroup(ddgi: DDGITechnique): GPUBindGroup {
    const resources = (ddgi as unknown as { volumes: readonly { rays: { traceGroup: GPUBindGroup } }[] }).volumes[0];
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
    data.directPaths = DIRECT_PATHS.join(',');
    // Textures scaled down at load change the GPU memory the scene takes (T2.17), so the run says so.
    if (MAX_TEXTURE_SIZE !== undefined) data.maxTextureSize = MAX_TEXTURE_SIZE;
    const runs = SCENES.flatMap((sceneId) => DIRECT_PATHS.map((direct) => ({ sceneId, direct })));
    for (const { sceneId, direct } of runs) {
        const id = sceneId;
        const manifest = manifests.find((candidate) => candidate.id === id);
        assert(manifest, `no manifest '${id}'`);
        const { scene, startup } = await startScene(device, renderer, manifest, direct, log);
        const sceneResult: Record<string, unknown> = { scene: id, directLight: direct, probes: scene.probeCount, raysPerProbe: scene.ddgi.settings.raysPerProbe };
        results[DIRECT_PATHS.length > 1 ? `${id}/${direct}` : id] = sceneResult;
        if (SECTIONS.has('startup')) sceneResult.startup = startup;
        if (RAMP_SECONDS > 0) sceneResult.ramp = await rampAndRecalibrate(device, scene, startup.calibrated_probes, log);
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
        if (SECTIONS.has('saturation')) {
            const { result, failures } = await measureSaturation(device, scene, log);
            sceneResult.saturation = result;
            budgetFailures.push(...failures);
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
        // The breakdown's variants edit M1's inline trace source; they mean nothing for the deferred kernel.
        if (SECTIONS.has('breakdown') && direct === 'inline') sceneResult.breakdown = await measureTraceBreakdown(device, scene, log);
        // Probes per state and rays traced, from the GPU census (a few frames old).
        sceneResult.counters = { ...scene.ddgi.counters };
        scene.threeGI.dispose();
        scene.platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
    assert(budgetFailures.length === 0, budgetFailures.join('; '));
});
