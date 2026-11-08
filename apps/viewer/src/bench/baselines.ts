/**
 * The baseline benchmark: for each scene and each technique, how long the
 * set-up takes and what a frame costs on the GPU. The techniques are three's
 * `LightProbeGrid` and VXGI (the baselines) and our DDGI (T1.8), so T1.9 can
 * measure DDGI's frame cost the same way the baselines were measured.
 *
 * Definitions (they are written into every result so nobody has to guess).
 * The view is 1280x720 at the scene's first bookmark, nothing moving.
 *
 * Set-up ("bake"), for the light probe grid (the whole bake with its bounce
 * passes) and VXGI (one re-voxelization: triangle collection, voxelization,
 * light injection and bounces, run outside a frame):
 *  - `warmup_wall_ms`: the throwaway set-up slice that compiles the bake's
 *    shaders (probe grid: one probe per pass; VXGI: one whole
 *    re-voxelization), until the GPU finished it. Kept apart so compile time
 *    is in no other number.
 *  - `bake_wall_ms`: wall-clock time from the first bake step until the GPU
 *    has finished the last, with timestamps off, nothing read back and a
 *    single wait at the very end.
 *  - `bake_gpu_pass_sum_ms`: the sum of the bake passes' GPU durations
 *    (three's timestamps), from a second, separate bake that reads the
 *    timestamps back whenever three's query pool fills, so it is not
 *    wall-timed. Passes may overlap on the GPU, so this sum can exceed the
 *    wall time; it is not "GPU busy time".
 *  Each bake is repeated `bakeRuns` times; the median is reported.
 *
 * DDGI's set-up is not a bake but a start-up, timed from when the technique
 * is created (the scene is already loaded):
 *  - `prepared_wall_ms`: the platform's kernels compiled (`gi.prepare()`);
 *  - `first_gi_wall_ms`: the scene copied into the platform and its
 *    ray-tracing trees built, so the next frame shows real GI;
 *  - `warmed_wall_ms`: the viewer's warm-up done (up to 16 full passes over
 *    every probe; `techniques/ddgi-technique.ts`), after which the frame
 *    budget applies. Frames are only measured after this.
 *
 * Frames:
 *  - `first_frame_wall_ms`: the first frame after the technique appears,
 *    until the GPU finished it. Contains shader compilation.
 *  - `frame_gpu_ms`: GPU time of every render and compute pass in one frame
 *    (shadow maps included), median over the sampled frames, with GI on.
 *    For DDGI it includes the platform's own passes (the probe update and
 *    the top-level refit), which it records in an encoder of its own that
 *    three's timestamps don't see. Those come from the platform's newest
 *    timed frame, one or two frames behind the frame they're added to
 *    (timestamps are read back late); on a still view with a fixed budget
 *    the same work runs every frame. DDGI's update runs within its default
 *    2 ms budget.
 *  - `frame_gpu_p95_ms`: the same frames' 95th percentile (nearest rank),
 *    per round, then the median over rounds: whether some frames spike.
 *  - `gi_off_frame_gpu_ms`: the same, through the same technique's pipeline
 *    (same MSAA, same TRAA, same targets) with only the GI taken out: for the
 *    probe grid, the grid is hidden from the camera; for VXGI, the gather and
 *    its injection are dropped while the pre-pass, scene pass and TRAA stay.
 *  - `gi_gpu_ms`: `frame_gpu_ms` minus `gi_off_frame_gpu_ms`, per round, then
 *    the mean, with a 95% confidence interval from the round-to-round spread
 *    (`_ci_low`, `_ci_high`). This is what the GI costs, in the technique's own
 *    pipeline. When the interval contains 0 the cost is below what the run can
 *    resolve: the value is written as measured, negative included, and marked
 *    `_below_noise`. Only an interval entirely below 0 is a failed
 *    measurement. See `measurement.ts`.
 *  - `frame_wall_ms` and `frame_wall_p95_ms`: median and 95th percentile of
 *    render-then-wait-for-the-GPU wall time, GI on; `gi_off_frame_wall_ms`
 *    and `gi_off_frame_wall_p95_ms` with GI off. CPU work counts here (for
 *    DDGI, the adapter's scene sync and the platform's encode), GPU work in
 *    parallel with it doesn't.
 *  GI on and GI off are measured in `frameRounds` alternating rounds
 *  (off-on, on-off, ...), so GPU clock drift cancels out of the difference.
 *  "No GI" has no GI to take out; its `frame_gpu_ms` is the plain frame of
 *  the probe grid's pipeline.
 *
 * Quality of a number:
 *  - Every repeated number has a coefficient of variation (`_cv`: standard
 *    deviation over the repeats divided by the mean). Above
 *    `NOISY_CV_THRESHOLD` (10%) it is marked noisy.
 *  - A time that comes out negative, or any value that is not finite, is never
 *    reported as a number. It is a failed measurement, with the reason.
 *    Differences follow the confidence-interval rule above.
 *
 * It drives `ViewerCore` directly, with no page UI, so the numbers are the
 * numbers of the viewer's own render path.
 */

import { isHeavyScene, type SceneManifest } from '../../../../scenes/src/manifest';
import { ViewerCore } from '../core/viewer-core';
import { resolveGpuTotals, timestampPoolsFillingUp } from '../render/gpu-time';
import type { GiMode } from '../scene/layers';
import { loadAllManifests } from '../scene/manifests';
import { DdgiTechnique } from '../techniques/ddgi-technique';
import type { GiTechnique, TechniqueId } from '../techniques/types';
import { failedMeasurement, median, NOISY_CV_THRESHOLD, percentile, summarize, summarizeDifference, type Measurement } from './measurement';

export { NOISY_CV_THRESHOLD, type Measurement } from './measurement';

export interface BaselineOptions {
    /** Scene ids; null means every manifest scene except the heavy ones (`isHeavyScene`). */
    scenes: string[] | null;
    /** Live techniques only: the reference image (`cycles`) has nothing to time. */
    techniques: Exclude<TechniqueId, 'cycles'>[];
    /** Frames timed per round and GI mode. */
    sampleFrames: number;
    /** Rounds of GI-off and GI-on frame timing. Their spread is the noise figure of the frame numbers. */
    frameRounds: number;
    /** How often each bake is repeated (each repeat is one wall-timed and one GPU-timed bake). */
    bakeRuns: number;
    width: number;
    height: number;
    /** Scale model textures down to this side at load (`LoadSceneOptions.maxTextureSize`); null keeps them as the files have them. */
    maxTextureSize: number | null;
    log(message: string): void;
}

export const DEFAULT_BASELINE_OPTIONS: Omit<BaselineOptions, 'log'> = {
    scenes: null,
    techniques: ['none', 'lpg', 'vxgi', 'ddgi'],
    sampleFrames: 20,
    frameRounds: 5,
    bakeRuns: 3,
    width: 1280,
    height: 720,
    maxTextureSize: null,
};

export interface BaselineRow {
    scene: string;
    technique: TechniqueId;
    triangles: number;
    measurements: Measurement[];
    details: Record<string, number | string>;
    /** Why the whole technique failed on this scene (load failure, exception, GPU error). */
    error: string | null;
}

export interface BaselineResult {
    rows: BaselineRow[];
    adapter: { vendor: string; architecture: string; description: string; device: string; features: string[] };
    hasTimestamps: boolean;
    /** Uncaptured GPU errors and device loss seen during the run, in order. */
    gpuErrors: string[];
    options: Omit<BaselineOptions, 'log'>;
    noisyCvThreshold: number;
    userAgent: string;
}

/** Places the viewer camera at a manifest's first bookmark. */
function useFirstBookmark(core: ViewerCore, manifest: SceneManifest): void {
    const bookmark = manifest.cameras[0];
    core.camera.position.set(...bookmark.position);
    core.camera.fov = bookmark.fovYDeg;
    core.camera.lookAt(...bookmark.target);
    core.camera.updateProjectionMatrix();
    core.camera.updateMatrixWorld(true);
}

/** The 95th percentile used for frame spikes. */
const P95 = 0.95;

interface FrameSample {
    wallMs: number;
    /** Null when the frame had an impossible timestamp, or the device has no timestamps. */
    gpuMs: number | null;
    passes: number;
    drawCalls: number;
}

/**
 * GPU work a technique records outside three's passes, for this frame, in ms; 0 when it has
 * none, null when it has some but no timing for it yet (the sample is then left out).
 */
function workOutsideThree(core: ViewerCore): number | null {
    const ddgi = core.activeTechniques().find((technique): technique is DdgiTechnique => technique instanceof DdgiTechnique);
    if (!ddgi || core.layout.giMode === 'off' || !ddgi.running) return 0;
    return ddgi.platformGpuMs();
}

/** Draws one frame and waits until the GPU has finished it. */
async function timedFrame(core: ViewerCore): Promise<FrameSample> {
    const startedAt = await core.renderThroughLoop();
    // Read straight after drawing: three resets the counters at the start of each frame.
    const drawCalls = core.renderer.info.render.drawCalls;
    await core.device.queue.onSubmittedWorkDone();
    const wallMs = performance.now() - startedAt;
    if (!core.hasTimestamps) return { wallMs, gpuMs: null, passes: 0, drawCalls };
    const totals = await resolveGpuTotals(core.renderer);
    // A frame with an impossible timestamp is left out of the median rather than poisoning it.
    const outside = workOutsideThree(core);
    const gpuMs = totals.invalidPasses > 0 || outside === null ? null : totals.renderMs + totals.computeMs + outside;
    return { wallMs, gpuMs, passes: totals.passCount, drawCalls };
}

/** Most extra warm-up frames spent waiting for shaders to finish compiling. */
const MAX_SETTLE_FRAMES = 1200;
/** Frames in a row that must look the same before timing starts. */
const SETTLE_WINDOW = 8;
/** Fewest frames drawn the first time a pipeline is shown, so a short plateau during compilation is not mistaken for done. */
const FIRST_SETTLE_FRAMES = 30;
/** Fewest frames drawn when going back to a pipeline that has been shown before (its shaders exist). */
const AGAIN_SETTLE_FRAMES = SETTLE_WINDOW;

/**
 * three compiles pipelines in the background and skips objects whose pipeline is not ready
 * yet, so the first frames after a technique appears can be nearly empty (a cold VXGI frame
 * once measured 0.03 ms, and a Bistro frame 3 ms instead of 40 ms). Keep drawing until the
 * number of draw calls has stopped growing (everything compiled) and the frame time has
 * stopped changing.
 */
async function settle(core: ViewerCore, minimumFrames: number): Promise<void> {
    const drawCalls: number[] = [];
    const gpuTimes: number[] = [];
    for (let i = 0; i < MAX_SETTLE_FRAMES; i++) {
        const sample = await timedFrame(core);
        drawCalls.push(sample.drawCalls);
        gpuTimes.push(sample.gpuMs ?? sample.wallMs);
        if (drawCalls.length < minimumFrames) continue;
        const recentCalls = drawCalls.slice(-SETTLE_WINDOW);
        const recentTimes = gpuTimes.slice(-SETTLE_WINDOW);
        const callsSteady = recentCalls.every((count) => count === recentCalls[0]);
        const typical = median(recentTimes);
        const timesSteady = recentTimes.every((ms) => Math.abs(ms - typical) <= 0.25 * typical + 0.02);
        if (callsSteady && timesSteady) return;
    }
}

interface SampledFrames {
    /** Median and 95th percentile of the frames' GPU time; null when no frame had a valid one. */
    gpuMs: number | null;
    gpuP95Ms: number | null;
    wallMs: number;
    wallP95Ms: number;
    passes: number;
}

/** Medians and 95th percentiles of `count` timed frames. */
async function sampleFrames(core: ViewerCore, count: number): Promise<SampledFrames> {
    const gpu: number[] = [];
    const wall: number[] = [];
    const passes: number[] = [];
    for (let i = 0; i < count; i++) {
        const sample = await timedFrame(core);
        wall.push(sample.wallMs);
        passes.push(sample.passes);
        if (sample.gpuMs !== null) gpu.push(sample.gpuMs);
    }
    return {
        gpuMs: gpu.length > 0 ? median(gpu) : null,
        gpuP95Ms: gpu.length > 0 ? percentile(gpu, P95) : null,
        wallMs: median(wall),
        wallP95Ms: percentile(wall, P95),
        passes: median(passes),
    };
}

/** One whole bake, timed on the wall clock: timestamps off, no read-back, one wait at the end. */
async function bakeWallRun(core: ViewerCore, technique: GiTechnique): Promise<number> {
    // Start from an idle GPU, so earlier work isn't counted.
    await core.device.queue.onSubmittedWorkDone();
    core.setTimestampTracking(false);
    try {
        technique.restartPrepare();
        const startedAt = performance.now();
        while (!technique.encodePrepareStep()) {
            // Each step only records and submits; the GPU works through them while the CPU records more.
        }
        await core.device.queue.onSubmittedWorkDone();
        return performance.now() - startedAt;
    } finally {
        core.setTimestampTracking(true);
    }
}

/** One whole bake, timed by GPU timestamps. Returns the sum over its passes, or the reason there is none. */
async function bakeGpuRun(core: ViewerCore, technique: GiTechnique): Promise<number | string> {
    if (!core.hasTimestamps) return 'the device has no timestamp queries';
    // Empty the timing window, so only the bake's passes are in it.
    await resolveGpuTotals(core.renderer);
    technique.restartPrepare();
    let totalMs = 0;
    let passes = 0;
    let invalid = 0;
    const collect = async () => {
        const totals = await resolveGpuTotals(core.renderer);
        totalMs += totals.renderMs + totals.computeMs;
        passes += totals.passCount;
        invalid += totals.invalidPasses;
    };
    let finished = false;
    while (!finished) {
        finished = technique.encodePrepareStep();
        // Past full, three would resolve on its own and those passes would be lost.
        if (timestampPoolsFillingUp(core.renderer)) await collect();
    }
    await collect();
    if (invalid > 0) return `${invalid} of ${passes} passes had impossible timestamps`;
    if (passes === 0) return 'no passes were timed';
    return totalMs;
}

interface TechniqueResult {
    measurements: Measurement[];
    details: Record<string, number | string>;
}

async function measureBake(core: ViewerCore, technique: GiTechnique, options: BaselineOptions): Promise<Measurement[]> {
    const warmupStartedAt = performance.now();
    await technique.warmUp();
    const measurements: Measurement[] = [summarize('warmup_wall_ms', [performance.now() - warmupStartedAt])];

    const wall: number[] = [];
    const gpu: number[] = [];
    let gpuProblem: string | null = null;
    for (let run = 0; run < options.bakeRuns; run++) {
        wall.push(await bakeWallRun(core, technique));
        const gpuResult = await bakeGpuRun(core, technique);
        if (typeof gpuResult === 'number') gpu.push(gpuResult);
        else gpuProblem ??= gpuResult;
    }
    measurements.push(summarize('bake_wall_ms', wall));
    measurements.push(gpuProblem ? failedMeasurement('bake_gpu_pass_sum_ms', gpuProblem, options.bakeRuns) : summarize('bake_gpu_pass_sum_ms', gpu));
    return measurements;
}

/**
 * DDGI's start-up: waits for the platform and the trees, then runs the warm-up the way the
 * viewer does (a few steps, then one wait for the GPU), and reports when each stage was
 * reached, counted from the technique's creation.
 */
async function measureDdgiStartup(core: ViewerCore, technique: DdgiTechnique): Promise<TechniqueResult> {
    await technique.warmUp();
    if (technique.failed) throw new Error(`DDGI could not start: ${technique.failed}`);
    const stepsPerBatch = technique.prepareStepsPerBatch;
    let steps = 0;
    while (!technique.ready) {
        for (let step = 0; step < stepsPerBatch && !technique.ready; step++, steps++) technique.encodePrepareStep();
        await core.device.queue.onSubmittedWorkDone();
    }
    const { preparedMs, firstGiMs, warmedMs } = technique.startup;
    const stage = (name: string, value: number | null) => (value === null ? failedMeasurement(name, 'not reached') : summarize(name, [value]));
    return {
        measurements: [stage('prepared_wall_ms', preparedMs), stage('first_gi_wall_ms', firstGiMs), stage('warmed_wall_ms', warmedMs)],
        details: { warmup_steps: steps, warmup_passes_made: Number(technique.warmupPasses.toFixed(2)) },
    };
}

async function measureFrames(core: ViewerCore, id: TechniqueId, options: BaselineOptions): Promise<TechniqueResult> {
    // "No GI" has nothing to take out: its GI-on frame is the plain frame.
    const modes: GiMode[] = id === 'none' ? ['on'] : ['off', 'on'];
    const gpu: Record<string, Array<number | null>> = { on: [], off: [] };
    const gpuP95: Record<string, number[]> = { on: [], off: [] };
    const wall: Record<string, number[]> = { on: [], off: [] };
    const wallP95: Record<string, number[]> = { on: [], off: [] };
    const passes: number[] = [];
    const shown = new Set<GiMode>();
    for (let round = 0; round < options.frameRounds; round++) {
        // Alternate the order (off-on, on-off, ...), so a steady drift cancels out of the difference.
        const order = round % 2 === 0 ? modes : [...modes].reverse();
        for (const mode of order) {
            core.setLayout({ giMode: mode });
            await settle(core, shown.has(mode) ? AGAIN_SETTLE_FRAMES : FIRST_SETTLE_FRAMES);
            shown.add(mode);
            const sampled = await sampleFrames(core, options.sampleFrames);
            gpu[mode].push(sampled.gpuMs);
            if (sampled.gpuP95Ms !== null) gpuP95[mode].push(sampled.gpuP95Ms);
            wall[mode].push(sampled.wallMs);
            wallP95[mode].push(sampled.wallP95Ms);
            if (mode === 'on') passes.push(sampled.passes);
        }
    }
    core.setLayout({ giMode: 'on' });

    const valid = (values: Array<number | null>) => values.filter((value): value is number => value !== null);
    const measurements: Measurement[] = [];
    if (!core.hasTimestamps) {
        measurements.push(failedMeasurement('frame_gpu_ms', 'the device has no timestamp queries'));
    } else {
        measurements.push(summarize('frame_gpu_ms', valid(gpu.on)));
        measurements.push(summarize('frame_gpu_p95_ms', gpuP95.on));
        if (id !== 'none') {
            measurements.push(summarize('gi_off_frame_gpu_ms', valid(gpu.off)));
            measurements.push(summarize('gi_off_frame_gpu_p95_ms', gpuP95.off));
            // Paired per round; a round with a missing side is left out.
            const differences = gpu.on.flatMap((on, round) => {
                const off = gpu.off[round];
                return on !== null && off !== null ? [on - off] : [];
            });
            measurements.push(summarizeDifference('gi_gpu_ms', differences));
        }
    }
    measurements.push(summarize('frame_wall_ms', wall.on));
    measurements.push(summarize('frame_wall_p95_ms', wallP95.on));
    if (id !== 'none') {
        measurements.push(summarize('gi_off_frame_wall_ms', wall.off));
        measurements.push(summarize('gi_off_frame_wall_p95_ms', wallP95.off));
    }
    return { measurements, details: { timed_passes_per_frame: median(passes) } };
}

async function measureTechnique(core: ViewerCore, id: TechniqueId, options: BaselineOptions): Promise<TechniqueResult> {
    core.setLayout({ left: id, right: null, giMode: 'on', split: 0.5 });
    const technique = core.getTechnique(id);
    const firstFrame = await timedFrame(core);
    const measurements: Measurement[] = [summarize('first_frame_wall_ms', [firstFrame.wallMs])];
    // DDGI starts up rather than bakes; the baselines bake (or voxelize).
    let startupDetails: Record<string, number | string> = {};
    if (technique instanceof DdgiTechnique) {
        const startup = await measureDdgiStartup(core, technique);
        measurements.push(...startup.measurements);
        startupDetails = startup.details;
    } else if (id !== 'none') {
        measurements.push(...(await measureBake(core, technique, options)));
    }
    const frames = await measureFrames(core, id, options);
    measurements.push(...frames.measurements);
    return {
        measurements,
        details: {
            ...technique.describeSettings(),
            ...startupDetails,
            ...frames.details,
            frame_rounds: options.frameRounds,
            bake_runs: options.bakeRuns,
            sample_frames: options.sampleFrames,
        },
    };
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function describeRow(row: BaselineRow): string {
    if (row.error) return `ERROR ${row.error}`;
    return row.measurements
        .map((measurement) => {
            if (measurement.value === null) return `${measurement.name} FAILED (${measurement.problem})`;
            const cv = measurement.cv === null ? '' : ` cv ${(measurement.cv * 100).toFixed(0)}%`;
            const interval = measurement.confidenceInterval
                ? ` [${measurement.confidenceInterval.low.toFixed(2)}, ${measurement.confidenceInterval.high.toFixed(2)}]`
                : '';
            const flags = `${measurement.noisy ? ' NOISY' : ''}${measurement.belowNoise ? ' BELOW-NOISE' : ''}`;
            return `${measurement.name} ${measurement.value.toFixed(2)}${interval}${cv}${flags}`;
        })
        .join(', ');
}

export async function runBaselines(canvas: HTMLCanvasElement, options: BaselineOptions): Promise<BaselineResult> {
    canvas.width = options.width;
    canvas.height = options.height;
    canvas.style.width = `${options.width}px`;
    canvas.style.height = `${options.height}px`;
    const core = await ViewerCore.create(canvas);
    core.resize(options.width, options.height);

    // "Every scene" leaves out the heavy ones (sponza-full); they run only when named.
    const manifests = loadAllManifests().filter((manifest) => (options.scenes === null ? !isHeavyScene(manifest) : options.scenes.includes(manifest.id)));
    if (options.scenes !== null && manifests.length !== options.scenes.length) {
        throw new Error(`Unknown scene id in ${options.scenes.join(',')}; known: ${loadAllManifests().map((m) => m.id).join(', ')}`);
    }

    const rows: BaselineRow[] = [];
    for (const manifest of manifests) {
        options.log(`scene ${manifest.id}: loading`);
        // Start every scene from "no GI": loading creates the technique the layout shows, and
        // DDGI's start-up times count from its creation, so it must be created when measured.
        core.layout = { left: 'none', right: null, giMode: 'on', split: 0.5 };
        let triangles = 0;
        try {
            const loaded = await core.loadScene(manifest, () => {}, options.maxTextureSize === null ? {} : { maxTextureSize: options.maxTextureSize });
            triangles = loaded.stats.triangles;
            useFirstBookmark(core, manifest);
        } catch (error) {
            // A missing download (not fetched yet) fails every technique of the scene, and says why.
            for (const technique of options.techniques) {
                rows.push({ scene: manifest.id, technique, triangles, measurements: [], details: {}, error: `load failed: ${errorMessage(error)}` });
            }
            options.log(`scene ${manifest.id}: load failed: ${errorMessage(error)}`);
            continue;
        }

        for (const technique of options.techniques) {
            options.log(`scene ${manifest.id}: ${technique}`);
            const errorsBefore = core.gpuErrors.length;
            let row: BaselineRow;
            try {
                const measured = await measureTechnique(core, technique, options);
                await core.device.queue.onSubmittedWorkDone();
                // Late validation errors show up a moment after the queue drains.
                await new Promise((resolve) => setTimeout(resolve, 100));
                const newErrors = core.gpuErrors.slice(errorsBefore);
                row = { scene: manifest.id, technique, triangles, ...measured, error: newErrors.length > 0 ? `GPU error: ${newErrors[0]}` : null };
            } catch (error) {
                row = { scene: manifest.id, technique, triangles, measurements: [], details: {}, error: errorMessage(error) };
            }
            rows.push(row);
            options.log(`  ${technique}: ${describeRow(row)}`);
        }
    }

    const info = core.adapterInfo;
    const result: BaselineResult = {
        rows,
        adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description, device: info.device, features: [...core.device.features].sort() },
        hasTimestamps: core.hasTimestamps,
        gpuErrors: [...core.gpuErrors],
        options: {
            scenes: options.scenes,
            techniques: options.techniques,
            sampleFrames: options.sampleFrames,
            frameRounds: options.frameRounds,
            bakeRuns: options.bakeRuns,
            width: options.width,
            height: options.height,
            maxTextureSize: options.maxTextureSize,
        },
        noisyCvThreshold: NOISY_CV_THRESHOLD,
        userAgent: navigator.userAgent,
    };
    core.unloadScene();
    return result;
}

/** Every failed measurement of a run, as `scene/technique/metric: reason` lines. */
export function failedMeasurements(result: BaselineResult): string[] {
    const problems: string[] = [];
    for (const row of result.rows) {
        if (row.error) problems.push(`${row.scene}/${row.technique}: ${row.error}`);
        for (const measurement of row.measurements) {
            if (measurement.problem) problems.push(`${row.scene}/${row.technique}/${measurement.name}: ${measurement.problem}`);
        }
    }
    return problems;
}
