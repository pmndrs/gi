/**
 * T1.12: how long DDGI's field takes to settle when each probe is updated
 * rarely (Sponza under budget v2: about one update every 73 frames), for each
 * hysteresis rule. Manual; for hand use under the bench lock.
 *
 * The GI image is the engine-free test view's indirect image
 * (ddgi-shared/harness.ts `TestView`, the same as ddgi-flicker): per pixel,
 * albedo / π × the field's irradiance. Its mean over the whole image is the
 * number followed here.
 *
 * Steps (one scene and camera per run):
 * 1. **The batch.** Start the scene as the viewer does
 *    (ddgi-shared/viewer-scene.ts), run the viewer's warm-up, then the
 *    platform's calibration as the viewer ends its warm-up (budget v2: the
 *    2 ms budget, frames back to back with two in flight, `deltaTime` 0)
 *    until it is calibrated and confirmed. Its batch is used for every rule
 *    below. `VITE_DDGI_SETTLE_UNITS` skips the calibration and uses that
 *    number instead.
 * 2. **The converged reference.** Every probe back to "not updated yet", the
 *    warm-up, then full passes over every probe with fixed hysteresis
 *    REFERENCE_FAST (fast, noisier) in blocks of REFERENCE_BLOCK_PASSES
 *    until the image mean moves less than REFERENCE_TOLERANCE between two
 *    blocks, then REFERENCE_QUIET_PASSES passes at REFERENCE_QUIET, and the
 *    reference is the mean of the image means after each of the last
 *    REFERENCE_AVERAGE_PASSES passes. Full passes settle bounce light in far
 *    fewer frames than the sparse batch; the field's settled value doesn't
 *    depend on the hysteresis (each rule is a weighted mean of unbiased
 *    estimates), only how fast and how quietly it gets there. The images of
 *    those last passes, averaged, are the reference image.
 *    For the `light` run, a second reference the same way with the sun at
 *    LIGHT_CHANGE_SCALE of its intensity.
 * 3. **Per rule** (`VITE_DDGI_HYSTERESIS`, default
 *    `fixed:0.97,tau:2,tau:5,tau:10`; ddgi-shared/hysteresis-modes.ts), up to
 *    three runs (`VITE_DDGI_SETTLE_RUNS`), each starting with every probe
 *    back to "not updated yet":
 *    - `warmup`: the viewer's warm-up, then frames at the batch;
 *    - `restart`: frames at the batch straight away, as after "Start the
 *      probes over" with no warm-up (a correction the field must recover from);
 *    - `light`: the warm-up, LIGHT_SETTLE_SECONDS at the batch (so each
 *      probe's history reaches its steady count), then the sun drops to
 *      LIGHT_CHANGE_SCALE and the run is timed against the second reference:
 *      how fast the field follows a light change with nothing resetting it
 *      (M2's SceneChanges will reset the probes near a change; this is the
 *      field on its own). Skipped in a scene with no directional light.
 *    Frames run back to back (two in flight); DDGI's clock advances
 *    FRAME_SECONDS a frame (a 60 Hz host, `VITE_DDGI_SETTLE_FPS`), so
 *    seconds below are frames × FRAME_SECONDS. The image mean is taken every
 *    SAMPLE_SECONDS. A run ends once the mean has stayed within 1% of the
 *    reference for HOLD_SECONDS, or at MAX_SECONDS (`VITE_DDGI_SETTLE_MAX_SECONDS`).
 *
 * Reported per rule and run, for 5% and 1%: the frames and seconds from the
 * end of the warm-up (or the restart, or the light change) until the image mean is within that
 * share of the reference and stays there for the rest of the run (null if it
 * never does), and the first time it gets there. Also the mean over the last
 * HOLD_SECONDS against the reference (the level it settles at). And, as
 * information, the per-pixel error: sum over pixels and channels of
 * |GI - reference image| / sum of the reference image. It holds the noise a
 * rule leaves in the field as well as any bias, so it shows what a lower
 * hysteresis costs in steadiness.
 *
 *   VITE_DDGI_SCENES=sponza   VITE_DDGI_CAMERA=1   VITE_DDGI_WIDTH=640 VITE_DDGI_HEIGHT=360
 */

import type { DirectionalLight } from 'three/webgpu';
import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { imageMean, TestView } from '../ddgi-shared/harness.ts';
import { applyHysteresisMode, parseHysteresisMode, parseHysteresisModes, type HysteresisMode } from '../ddgi-shared/hysteresis-modes.ts';
import { startViewerScene, warmUp, yieldToCallbacks, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}
function numberFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' ? Number(value) : NaN;
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

const SCENE = listFromEnv(env.VITE_DDGI_SCENES, ['sponza'])[0];
const CAMERA = numberFromEnv(env.VITE_DDGI_CAMERA, 1);
const WIDTH = numberFromEnv(env.VITE_DDGI_WIDTH, 640);
const HEIGHT = numberFromEnv(env.VITE_DDGI_HEIGHT, 360);
const MODES = parseHysteresisModes(env.VITE_DDGI_HYSTERESIS, ['fixed:0.97', 'tau:2', 'tau:5', 'tau:10']);
const UNITS_OVERRIDE = numberFromEnv(env.VITE_DDGI_SETTLE_UNITS, 0);
const RUNS = listFromEnv(env.VITE_DDGI_SETTLE_RUNS, ['warmup', 'restart', 'light']);
/** The `light` run: the sun drops to this share of its intensity... */
const LIGHT_CHANGE_SCALE = 0.5;
/** ...after this long at the batch with the normal sun (4 memory times of the longest rule measured, 10 s). */
const LIGHT_SETTLE_SECONDS = 40;

/** DDGI's clock step per frame: the host's frame interval. */
const FRAME_SECONDS = 1 / numberFromEnv(env.VITE_DDGI_SETTLE_FPS, 60);
/** How often the image mean is taken. */
const SAMPLE_SECONDS = 0.5;
/** A run ends once the image mean has stayed within the tightest tolerance this long... */
const HOLD_SECONDS = 60;
/** ...or after this long. */
const MAX_SECONDS = numberFromEnv(env.VITE_DDGI_SETTLE_MAX_SECONDS, 600);
const TOLERANCES = [0.05, 0.01] as const;

/** The budget the calibration solves for (the viewer's default). */
const BUDGET_MS = 2;
/** As the viewer: the calibration's burst and confirmation stop after this many frames. */
const CALIBRATION_MAX_STEPS = 1000;
/** Frames in flight when running back to back, as the viewer's calibration. */
const FRAMES_IN_FLIGHT = 2;
const MAX_PROBES_PER_FRAME = 4096;

/** The converged reference (step 2 of the file comment). */
const REFERENCE_FAST = parseHysteresisMode('fixed:0.8');
const REFERENCE_QUIET = parseHysteresisMode('fixed:0.98');
const REFERENCE_BLOCK_PASSES = 8;
const REFERENCE_TOLERANCE = 0.0005;
const REFERENCE_MIN_BLOCKS = 3;
const REFERENCE_MAX_BLOCKS = 60;
const REFERENCE_QUIET_PASSES = 32;
const REFERENCE_AVERAGE_PASSES = 16;

const PROBES_UPDATED = 'probes updated';

function mean(values: readonly number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function standardDeviation(values: readonly number[]): number {
    const centre = mean(values);
    return Math.sqrt(mean(values.map((value) => (value - centre) ** 2)));
}
function round(value: number, significant = 5): number {
    return Number.isFinite(value) ? Number(value.toPrecision(significant)) : value;
}

/** Back-to-back frames with a few in flight, so the GPU never waits for the page. */
class GpuBoundPacing {
    private readonly pending: Promise<void>[] = [];
    constructor(private readonly device: GPUDevice) {}

    async after(): Promise<void> {
        this.pending.push(this.device.queue.onSubmittedWorkDone());
        if (this.pending.length >= FRAMES_IN_FLIGHT) await this.pending.shift();
        await yieldToCallbacks();
    }

    async drain(): Promise<void> {
        await Promise.all(this.pending.splice(0));
    }
}

interface Context {
    readonly scene: ViewerScene;
    readonly view: TestView;
    readonly bookmark: ViewerScene['manifest']['cameras'][number];
    readonly log: (line: string) => void;
}

async function giImage(context: Context): Promise<Float32Array> {
    const field = context.scene.ddgi.field;
    assert(field, 'DDGI has no field');
    return (await context.view.render(field, context.bookmark, WIDTH, HEIGHT)).indirect;
}

async function giImageMean(context: Context): Promise<number> {
    return imageMean(await giImage(context));
}

/** Sum of |image - reference| over pixels and channels, over the sum of the reference. */
function pixelError(image: Float32Array, reference: Float32Array): number {
    let difference = 0;
    let total = 0;
    for (let index = 0; index < reference.length; index++) {
        difference += Math.abs(image[index] - reference[index]);
        total += reference[index];
    }
    return difference / total;
}

/** Step 1: the batch budget v2 gives this scene with the clocks up, as the viewer gets it. */
async function calibratedBatch(context: Context): Promise<Record<string, number>> {
    const { scene } = context;
    if (UNITS_OVERRIDE > 0) return { units: Math.round(UNITS_OVERRIDE), from_override: 1 };
    await warmUp(scene);
    const pacing = new GpuBoundPacing(scene.device);
    let steps = 0;
    for (;;) {
        scene.frame({ gpuMs: BUDGET_MS }, 0);
        steps++;
        const report = scene.platform.budgetReport(scene.ddgi);
        const calibrating = report?.state === 'calibrating' || report?.confirming === true;
        if (!calibrating || steps >= CALIBRATION_MAX_STEPS) break;
        await pacing.after();
    }
    await pacing.drain();
    const calibration = scene.platform.budgetReport(scene.ddgi)?.calibration;
    assert(calibration?.measured, `${SCENE}: the platform did not calibrate within ${steps} frames`);
    return {
        units: calibration.units,
        from_override: 0,
        calibration_steps: steps,
        confirm_factor: calibration.confirmFactor ?? NaN,
        ms_at_batch: calibration.model ? calibration.model.fixedMs + calibration.model.msPerUnit * calibration.units : NaN,
    };
}

/** Full passes over every probe, split into equal frames; returns the image after each pass when asked. */
async function fullPasses(context: Context, passes: number, imageEachPass: boolean): Promise<Float32Array[]> {
    const { scene } = context;
    const framesPerPass = Math.ceil(scene.probeCount / MAX_PROBES_PER_FRAME);
    const unitsPerFrame = Math.ceil(scene.probeCount / framesPerPass);
    const images: Float32Array[] = [];
    for (let pass = 0; pass < passes; pass++) {
        for (let frame = 0; frame < framesPerPass; frame++) scene.frame({ units: unitsPerFrame }, FRAME_SECONDS);
        await scene.device.queue.onSubmittedWorkDone();
        if (imageEachPass) images.push(await giImage(context));
    }
    return images;
}

interface Reference {
    readonly image: Float32Array;
    readonly report: Record<string, number>;
}

/** Step 2: the image mean of a fully converged field, and its image. */
async function convergedReference(context: Context): Promise<Reference> {
    const { scene, log } = context;
    applyHysteresisMode(scene.ddgi.settings, REFERENCE_FAST);
    scene.ddgi.resetHistory();
    await warmUp(scene);
    let blockMean = await giImageMean(context);
    let blocks = 0;
    let settled = false;
    for (let block = 1; block <= REFERENCE_MAX_BLOCKS; block++) {
        await fullPasses(context, REFERENCE_BLOCK_PASSES, false);
        const next = await giImageMean(context);
        const change = next / blockMean - 1;
        blockMean = next;
        blocks = block;
        log(`reference block ${block} (${REFERENCE_FAST.name}): image mean ${next.toPrecision(6)}, ${(change * 100).toFixed(3)}% since the last block`);
        if (block >= REFERENCE_MIN_BLOCKS && Math.abs(change) < REFERENCE_TOLERANCE) {
            settled = true;
            break;
        }
    }
    assert(settled, `the reference did not settle within ±${REFERENCE_TOLERANCE * 100}% in ${REFERENCE_MAX_BLOCKS} blocks`);
    applyHysteresisMode(scene.ddgi.settings, REFERENCE_QUIET);
    const quietImages = (await fullPasses(context, REFERENCE_QUIET_PASSES, true)).slice(-REFERENCE_AVERAGE_PASSES);
    const averaged = quietImages.map(imageMean);
    const reference = mean(averaged);
    const image = new Float32Array(quietImages[0].length);
    for (const quiet of quietImages) for (let index = 0; index < image.length; index++) image[index] += quiet[index] / quietImages.length;
    // How far one quiet pass's image is from the averaged one: the floor the per-pixel error can reach at REFERENCE_QUIET.
    const quietPixelError = mean(quietImages.map((quiet) => pixelError(quiet, image)));
    log(`reference: ${reference.toPrecision(6)} (fast blocks ${blocks}; quiet passes' sd ${((standardDeviation(averaged) / reference) * 100).toFixed(3)}%; last fast block ${blockMean.toPrecision(6)})`);
    return {
        image,
        report: {
            image_mean: reference,
            fast_blocks: blocks,
            fast_passes: blocks * REFERENCE_BLOCK_PASSES,
            quiet_passes: REFERENCE_QUIET_PASSES,
            quiet_sd_share: standardDeviation(averaged) / reference,
            quiet_against_fast: reference / blockMean - 1,
            quiet_pixel_error: quietPixelError,
        },
    };
}

interface Sample {
    readonly frame: number;
    /** Image mean against the reference's, minus 1. */
    readonly error: number;
    readonly pixelError: number;
}

/** When the error gets within `tolerance` for the first time, and from when it stays there (null if it never does). */
function settleTimes(samples: readonly Sample[], tolerance: number): { first: number | null; stays: number | null } {
    const first = samples.find((sample) => Math.abs(sample.error) <= tolerance)?.frame ?? null;
    let stays: number | null = null;
    for (let index = samples.length - 1; index >= 0; index--) {
        if (Math.abs(samples[index].error) > tolerance) break;
        stays = samples[index].frame;
    }
    return { first, stays };
}

/** The scene's first directional light (the sun), if it has one. */
function findSun(scene: ViewerScene): DirectionalLight | undefined {
    let sun: DirectionalLight | undefined;
    scene.loaded.scene.traverse((object) => {
        if (!sun && (object as DirectionalLight).isDirectionalLight) sun = object as DirectionalLight;
    });
    return sun;
}

/** Frames at the batch, back to back, not measured. */
async function runAtBatch(scene: ViewerScene, units: number, seconds: number): Promise<void> {
    const pacing = new GpuBoundPacing(scene.device);
    const frames = Math.round(seconds / FRAME_SECONDS);
    for (let frame = 0; frame < frames; frame++) {
        scene.frame({ units }, FRAME_SECONDS);
        await pacing.after();
    }
    await pacing.drain();
}

interface References {
    readonly normal: Reference;
    /** With the sun at LIGHT_CHANGE_SCALE; null without a sun. */
    readonly dimmed: Reference | null;
    readonly sun: { readonly light: DirectionalLight; readonly intensity: number } | null;
}

/** Step 3: one rule, one run. */
async function settleRun(context: Context, mode: HysteresisMode, run: string, units: number, references: References): Promise<Record<string, number | string | null>> {
    const { scene, log } = context;
    applyHysteresisMode(scene.ddgi.settings, mode);
    scene.ddgi.resetHistory();
    let reference = references.normal;
    if (run === 'warmup') {
        await warmUp(scene);
    } else if (run === 'light') {
        assert(references.sun && references.dimmed, 'the light run needs a directional light');
        await warmUp(scene);
        await runAtBatch(scene, units, LIGHT_SETTLE_SECONDS);
        references.sun.light.intensity = references.sun.intensity * LIGHT_CHANGE_SCALE;
        reference = references.dimmed;
    } else {
        assert(run === 'restart', `unknown run '${run}': warmup, restart or light`);
    }

    const sampleFrames = Math.max(1, Math.round(SAMPLE_SECONDS / FRAME_SECONDS));
    const holdFrames = Math.round(HOLD_SECONDS / FRAME_SECONDS);
    const maxFrames = Math.round(MAX_SECONDS / FRAME_SECONDS);
    const tightest = Math.min(...TOLERANCES);
    const referenceMean = reference.report.image_mean;
    const sample = async (frame: number): Promise<Sample> => {
        const image = await giImage(context);
        return { frame, error: imageMean(image) / referenceMean - 1, pixelError: pixelError(image, reference.image) };
    };
    const samples: Sample[] = [await sample(0)];
    const probesPerFrame: number[] = [];
    const pacing = new GpuBoundPacing(scene.device);
    let frame = 0;
    let withinSince: number | null = Math.abs(samples[0].error) <= tightest ? 0 : null;
    while (frame < maxFrames) {
        scene.frame({ units }, FRAME_SECONDS);
        frame++;
        if (frame % sampleFrames !== 0) {
            await pacing.after();
            continue;
        }
        await pacing.drain();
        probesPerFrame.push(scene.ddgi.counters[PROBES_UPDATED] ?? 0);
        const latest = await sample(frame);
        samples.push(latest);
        const error = latest.error;
        if (Math.abs(error) > tightest) withinSince = null;
        else withinSince ??= frame;
        if (withinSince !== null && frame - withinSince >= holdFrames) break;
    }
    await pacing.drain();
    if (references.sun) references.sun.light.intensity = references.sun.intensity;

    const holdSamples = samples.filter((sample) => sample.frame >= frame - holdFrames);
    const result: Record<string, number | string | null> = {
        hysteresis: mode.name,
        run,
        frames_run: frame,
        seconds_run: round(frame * FRAME_SECONDS),
        probes_per_frame: mean(probesPerFrame),
        start_error: round(samples[0].error),
        end_error: round(samples[samples.length - 1].error),
        hold_mean_error: round(mean(holdSamples.map((sample) => sample.error))),
        hold_sd_error: round(standardDeviation(holdSamples.map((sample) => sample.error))),
        start_pixel_error: round(samples[0].pixelError),
        hold_pixel_error: round(mean(holdSamples.map((sample) => sample.pixelError))),
    };
    for (const tolerance of TOLERANCES) {
        const percent = `${tolerance * 100}pct`;
        const { first, stays } = settleTimes(samples, tolerance);
        result[`frames_to_${percent}`] = stays;
        result[`seconds_to_${percent}`] = stays === null ? null : round(stays * FRAME_SECONDS);
        result[`first_frames_to_${percent}`] = first;
        result[`first_seconds_to_${percent}`] = first === null ? null : round(first * FRAME_SECONDS);
    }
    // The curve, coarsely (every 5 s): seconds, then image-mean error / per-pixel error in percent.
    const curveEvery = Math.max(1, Math.round(5 / SAMPLE_SECONDS));
    result.curve = samples
        .filter((_, index) => index % curveEvery === 0)
        .map((sample) => `${round(sample.frame * FRAME_SECONDS, 4)}:${round(sample.error * 100, 3)}/${round(sample.pixelError * 100, 3)}`)
        .join(' ');
    const show = (key: string) => (result[key] === null ? 'never' : `${result[key]} s`);
    log(
        `${mode.name} ${run}: start ${(samples[0].error * 100).toFixed(2)}%; within 5% from ${show('seconds_to_5pct')}, within 1% from ${show('seconds_to_1pct')} ` +
            `(${result.frames_to_1pct ?? 'never'} frames); level over the last ${HOLD_SECONDS} s ${((result.hold_mean_error as number) * 100).toFixed(3)}%, ` +
            `per-pixel error ${((result.hold_pixel_error as number) * 100).toFixed(2)}%; ran ${result.seconds_run} s`,
    );
    return result;
}

await runThreePage(async ({ device, data, log }) => {
    data.userAgent = navigator.userAgent;
    data.browserVersion = await fullBrowserVersion();
    const manifest = loadAllManifests().find((candidate) => candidate.id === SCENE);
    assert(manifest, `no manifest '${SCENE}'`);
    const bookmark = manifest.cameras[CAMERA - 1];
    assert(bookmark, `${SCENE} has no camera ${CAMERA}`);
    Object.assign(data, { scene: SCENE, camera: bookmark.name, width: WIDTH, height: HEIGHT, frame_seconds: FRAME_SECONDS });

    const scene = await startViewerScene(device, manifest, { gpuTiming: true, label: `settle ${SCENE}` });
    try {
        data.probes = scene.probeCount;
        const context: Context = { scene, view: await TestView.create(device, scene.platform), bookmark, log };
        const batch = await calibratedBatch(context);
        data.batch = batch;
        data.sweep_frames = round(scene.probeCount / batch.units);
        log(`batch ${batch.units} probes a frame (${batch.from_override ? 'given' : 'calibrated'}): every probe once per ${data.sweep_frames} frames`);
        const normal = await convergedReference(context);
        data.reference = normal.report;
        const sunLight = findSun(scene);
        const sun = sunLight ? { light: sunLight, intensity: sunLight.intensity } : null;
        let dimmed: Reference | null = null;
        if (sun && RUNS.includes('light')) {
            sun.light.intensity = sun.intensity * LIGHT_CHANGE_SCALE;
            dimmed = await convergedReference(context);
            sun.light.intensity = sun.intensity;
            data.reference_dimmed = dimmed.report;
            data.light_change_scale = LIGHT_CHANGE_SCALE;
        }
        const references: References = { normal, dimmed, sun };
        const runs: Record<string, unknown> = {};
        data.runs = runs;
        for (const mode of MODES) {
            for (const run of RUNS) {
                if (run === 'light' && !sun) {
                    log(`${mode.name} light: skipped, ${SCENE} has no directional light`);
                    continue;
                }
                runs[`${mode.name}/${run}`] = await settleRun(context, mode, run, batch.units, references);
            }
        }
    } finally {
        scene.dispose();
    }
});
