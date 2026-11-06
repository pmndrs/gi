/**
 * T1.9 stability: how much the GI image changes from one frame to the next on
 * a still, converged scene while DDGI keeps updating at its normal budget.
 * The number Dennis sets a threshold on by eye (SPEC, "How we'll judge it").
 *
 * The GI image is the engine-free test view's indirect image
 * (ddgi-shared/harness.ts `TestView`): per pixel, albedo / π × the field's
 * irradiance at the primary hit. It holds no noise of its own (no shadow
 * rays, no texture), so every change between two frames is the probes
 * changing. Camera and scene don't move.
 *
 * Steps per run (one scene, one camera):
 * 1. Start the scene as the viewer does (ddgi-shared/viewer-scene.ts) and run
 *    the viewer's warm-up.
 * 2. Converge: blocks of full passes over every probe with the fixed 0.97
 *    hysteresis (the rule before T1.12, whatever is measured after: with
 *    every probe updated often it settles bounce light fastest, and the
 *    flicker number is about the settled state), until the image mean moves
 *    less than CONVERGED_TOLERANCE between two blocks (at least MIN_BLOCKS;
 *    fails after MAX_BLOCKS). This is the accuracy page's rule, so
 *    "converged" means the same thing.
 * 3. For each hysteresis rule (`VITE_DDGI_FLICKER_HYSTERESIS`, default the
 *    technique's defaults; ddgi-shared/hysteresis-modes.ts) and each budget
 *    setting (`VITE_DDGI_FLICKER_BUDGETS`, default `gpuMs:2,units:256`):
 *    settle frames at that budget, not measured (SETTLE_FRAMES: the
 *    millisecond budget needs a few frames to find its batch; an adaptive
 *    rule gets at least 4 memory times, so each probe's history reaches its
 *    steady count), then MEASURE_FRAMES frames, each one platform update
 *    followed by one test-view render (this waits for the GPU, so frames run
 *    back to back; DDGI's clock still advances 1/60 s a frame, a 60 Hz host).
 *
 * The measure, for each pair of consecutive frames (t-1, t), over the
 * "surface" pixels (primary ray hit something; sky pixels are 0 in both
 * frames and would only dilute it):
 *   d(pixel) = mean over rgb of |GI(t) - GI(t-1)|  /  mean(GI(t))
 * where mean(GI(t)) is the mean of rgb over the same pixels. Reported: the
 * mean of d over pixels, and its 95th percentile over pixels, each then
 * averaged over the frame pairs. Also: how much the whole image's mean moves
 * between frames (brightness flicker), the probes updated per frame, and the
 * drift of the image mean from the first to the last 10 frames (a check that
 * the scene was converged).
 *
 * Frame sequences for looking at (`VITE_DDGI_FLICKER_IMAGES=1`): from each
 * setting, IMAGE_SEQUENCES runs of IMAGE_SEQUENCE_FRAMES consecutive frames as
 * PNG (same exposure within a scene: the converged mean maps to 0.25, then
 * sRGB), and for each run the difference between its first and last frame,
 * |a - b| / mean x DIFFERENCE_GAIN as grey. Written by to-csv.ts.
 *
 *   VITE_DDGI_SCENES=cornell-box   (one scene per run)
 *   VITE_DDGI_CAMERA=1             (1-based bookmark)
 *   VITE_DDGI_FLICKER_BUDGETS=gpuMs:2,units:256
 *   VITE_DDGI_FLICKER_HYSTERESIS=fixed:0.97,tau:2,tau:5,tau:10   (results keyed '<rule>/<budget>' when given)
 *   VITE_DDGI_FLICKER_AFTER_WARMUP=400   (instead of steps 1-2: for each rule, every probe back to "not
 *       updated yet", the warm-up, then this many frames at the budget with deltaTime 0, as the viewer's
 *       calibration runs, and the measurement starts at once: what a user sees right after loading)
 *   VITE_DDGI_FLICKER_FRAMES=120
 *   VITE_DDGI_FLICKER_IMAGES=1
 *   VITE_DDGI_WIDTH=640 VITE_DDGI_HEIGHT=360
 *
 * Manual; run it under the bench lock, three times per scene and GPU, then
 * `node tools/gpu/pages/ddgi-flicker/to-csv.ts`.
 */

import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { imageMean, TestView } from '../ddgi-shared/harness.ts';
import { applyHysteresisMode, FIXED_097, parseHysteresisModes, type HysteresisMode } from '../ddgi-shared/hysteresis-modes.ts';
import { startViewerScene, warmUp, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}
function countFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' ? Number(value) : NaN;
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

const SCENE = listFromEnv(env.VITE_DDGI_SCENES, ['cornell-box'])[0];
const CAMERA = countFromEnv(env.VITE_DDGI_CAMERA, 1);
const BUDGETS = listFromEnv(env.VITE_DDGI_FLICKER_BUDGETS, ['gpuMs:2', 'units:256']);
/** Hysteresis rules to measure; without the variable, the defaults only and results keyed by budget as before. */
const HYSTERESIS_GIVEN = typeof env.VITE_DDGI_FLICKER_HYSTERESIS === 'string' && env.VITE_DDGI_FLICKER_HYSTERESIS.trim() !== '';
const HYSTERESIS_MODES = parseHysteresisModes(env.VITE_DDGI_FLICKER_HYSTERESIS, ['default']);
const MEASURE_FRAMES = countFromEnv(env.VITE_DDGI_FLICKER_FRAMES, 120);
const SAVE_IMAGES = env.VITE_DDGI_FLICKER_IMAGES === '1';
const WIDTH = countFromEnv(env.VITE_DDGI_WIDTH, 640);
const HEIGHT = countFromEnv(env.VITE_DDGI_HEIGHT, 360);

const SETTLE_FRAMES = 60;
/** Frames after the warm-up before measuring, when the run measures right after loading (see the file comment); -1 when not. */
const AFTER_WARMUP_FRAMES = typeof env.VITE_DDGI_FLICKER_AFTER_WARMUP === 'string' && env.VITE_DDGI_FLICKER_AFTER_WARMUP !== '' ? Math.max(0, Math.round(Number(env.VITE_DDGI_FLICKER_AFTER_WARMUP))) : -1;
/** DDGI's clock step per frame: a 60 Hz host. */
const FRAME_SECONDS = 1 / 60;
/** An adaptive rule settles for at least this many memory times (each probe's history count reaches its steady value within e^-4). */
const SETTLE_MEMORY_TIMES = 4;
/** The accuracy page's convergence rule (ddgi-sponza-accuracy). */
const BLOCK_PASSES = 32;
const CONVERGED_TOLERANCE = 0.005;
const MIN_BLOCKS = 2;
const MAX_BLOCKS = 12;
const MAX_PROBES_PER_FRAME = 4096;
const PROBES_UPDATED = 'probes updated';
const DRIFT_FRAMES = 10;

const IMAGE_SEQUENCES = 2;
const IMAGE_SEQUENCE_FRAMES = 4;
/** Where in the measured frames the sequences start (as a share), so they are not the first, still settling, ones. */
const IMAGE_SEQUENCE_STARTS = [0.25, 0.7];
/** A difference of 1 / DIFFERENCE_GAIN of the mean shows as white. */
const DIFFERENCE_GAIN = 10;
/** The converged image mean maps to this display value before sRGB. */
const DISPLAY_MEAN = 0.25;

type Budget = { gpuMs: number } | { units: number };

function parseBudget(text: string): Budget {
    const [kind, amount] = text.split(':');
    const value = Number(amount);
    assert(Number.isFinite(value) && value > 0, `bad budget '${text}'`);
    if (kind === 'gpuMs') return { gpuMs: value };
    if (kind === 'units') return { units: value };
    throw new Error(`bad budget '${text}': use gpuMs:<ms> or units:<probes>`);
}

function percentile(sorted: Float32Array, share: number): number {
    return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(share * sorted.length) - 1))];
}
function mean(values: readonly number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function standardDeviation(values: readonly number[]): number {
    const centre = mean(values);
    return Math.sqrt(mean(values.map((value) => (value - centre) ** 2)));
}
function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
/** Five significant digits: Sponza's changes are around 0.00002, which fixed decimals would cut to nothing. */
function round(value: number, significant = 5): number {
    return Number.isFinite(value) ? Number(value.toPrecision(significant)) : value;
}

interface Comparison {
    /** Mean over surface pixels of the per-pixel relative change. */
    readonly mean: number;
    readonly p95: number;
    /** |mean(t) - mean(t-1)| / mean(t-1) over the whole image. */
    readonly imageMeanChange: number;
    readonly surfaceShare: number;
    readonly surfaceMean: number;
}

/** One frame pair, as defined in the file comment. `surface` marks the pixels whose primary ray hit something. */
function compare(previous: Float32Array, current: Float32Array, surface: Uint8Array): Comparison {
    const pixels = surface.length;
    let surfaceCount = 0;
    let surfaceSum = 0;
    for (let pixel = 0; pixel < pixels; pixel++) {
        if (!surface[pixel]) continue;
        surfaceCount++;
        surfaceSum += current[pixel * 3] + current[pixel * 3 + 1] + current[pixel * 3 + 2];
    }
    const surfaceMean = surfaceSum / (3 * surfaceCount);
    const changes = new Float32Array(surfaceCount);
    let next = 0;
    for (let pixel = 0; pixel < pixels; pixel++) {
        if (!surface[pixel]) continue;
        const change = (Math.abs(current[pixel * 3] - previous[pixel * 3]) + Math.abs(current[pixel * 3 + 1] - previous[pixel * 3 + 1]) + Math.abs(current[pixel * 3 + 2] - previous[pixel * 3 + 2])) / 3;
        changes[next++] = change / surfaceMean;
    }
    let sum = 0;
    for (const change of changes) sum += change;
    changes.sort();
    const previousMean = imageMean(previous);
    return {
        mean: sum / surfaceCount,
        p95: percentile(changes, 0.95),
        imageMeanChange: Math.abs(imageMean(current) - previousMean) / previousMean,
        surfaceShare: surfaceCount / pixels,
        surfaceMean,
    };
}

function toSrgb8(linear: number): number {
    const clamped = Math.min(1, Math.max(0, linear));
    const encoded = clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
}

async function pngBase64(rgba: Uint8ClampedArray<ArrayBuffer>, width: number, height: number): Promise<string> {
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext('2d');
    assert(context, 'no 2d canvas for PNG encoding');
    context.putImageData(new ImageData(rgba, width, height), 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
    return btoa(binary);
}

function exposedImage(image: Float32Array, scale: number, width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let pixel = 0; pixel < width * height; pixel++) {
        for (let channel = 0; channel < 3; channel++) rgba[pixel * 4 + channel] = toSrgb8(image[pixel * 3 + channel] * scale);
        rgba[pixel * 4 + 3] = 255;
    }
    return rgba;
}

function differenceImage(first: Float32Array, last: Float32Array, referenceMean: number, width: number, height: number): Uint8ClampedArray<ArrayBuffer> {
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let pixel = 0; pixel < width * height; pixel++) {
        const change = (Math.abs(first[pixel * 3] - last[pixel * 3]) + Math.abs(first[pixel * 3 + 1] - last[pixel * 3 + 1]) + Math.abs(first[pixel * 3 + 2] - last[pixel * 3 + 2])) / 3;
        const grey = Math.round(Math.min(1, (change / referenceMean) * DIFFERENCE_GAIN) * 255);
        rgba.set([grey, grey, grey, 255], pixel * 4);
    }
    return rgba;
}

/** Frames to run before measuring a rule: SETTLE_FRAMES, and for an adaptive rule at least SETTLE_MEMORY_TIMES memory times. */
function settleFrames(mode: HysteresisMode): number {
    const { adaptiveHysteresis, hysteresisMemorySeconds } = mode.settings;
    if (!adaptiveHysteresis || !Number.isFinite(hysteresisMemorySeconds)) return SETTLE_FRAMES;
    return Math.max(SETTLE_FRAMES, Math.ceil((SETTLE_MEMORY_TIMES * hysteresisMemorySeconds) / FRAME_SECONDS));
}

/** One hysteresis rule and budget setting: settle, then measure frame by frame. */
async function measureSetting(
    scene: ViewerScene,
    view: TestView,
    bookmark: ViewerScene['manifest']['cameras'][number],
    mode: HysteresisMode,
    budgetText: string,
    log: (line: string) => void,
): Promise<{ result: Record<string, unknown>; images: Array<{ name: string; base64: string }> }> {
    const budget = parseBudget(budgetText);
    const field = scene.ddgi.field;
    assert(field, 'DDGI has no field');
    applyHysteresisMode(scene.ddgi.settings, mode);
    const afterWarmup = AFTER_WARMUP_FRAMES >= 0;
    if (afterWarmup) {
        scene.ddgi.resetHistory();
        await warmUp(scene);
        for (let index = 0; index < AFTER_WARMUP_FRAMES; index++) {
            scene.frame(budget, 0);
            if (index % 8 === 7) await scene.device.queue.onSubmittedWorkDone();
        }
    }
    const renderFrame = async (): Promise<Float32Array> => {
        scene.frame(budget, FRAME_SECONDS);
        const { indirect } = await view.render(field, bookmark, WIDTH, HEIGHT);
        return indirect;
    };
    // Settling needs no images; the GPU is waited for every few frames so the page never sits in one long task.
    const settle = afterWarmup ? 0 : settleFrames(mode);
    for (let index = 0; index < settle; index++) {
        scene.frame(budget, FRAME_SECONDS);
        if (index % 8 === 7) await scene.device.queue.onSubmittedWorkDone();
    }

    const comparisons: Comparison[] = [];
    const probesPerFrame: number[] = [];
    const imageMeans: number[] = [];
    const kept = new Map<number, Float32Array>();
    const sequenceFrames = new Set<number>();
    if (SAVE_IMAGES) {
        for (const start of IMAGE_SEQUENCE_STARTS.slice(0, IMAGE_SEQUENCES)) {
            const first = Math.floor(start * MEASURE_FRAMES);
            for (let offset = 0; offset < IMAGE_SEQUENCE_FRAMES; offset++) sequenceFrames.add(first + offset);
        }
    }
    let previous = await renderFrame();
    let surface: Uint8Array | null = null;
    for (let index = 0; index < MEASURE_FRAMES; index++) {
        const current = await renderFrame();
        probesPerFrame.push(scene.ddgi.counters[PROBES_UPDATED] ?? 0);
        if (!surface) {
            // The same pixels hit a surface in every frame (nothing moves); the first frame decides.
            surface = new Uint8Array(WIDTH * HEIGHT);
            for (let pixel = 0; pixel < surface.length; pixel++) surface[pixel] = current[pixel * 3] + current[pixel * 3 + 1] + current[pixel * 3 + 2] > 0 ? 1 : 0;
            assert(surface.some((value) => value === 1), 'the test view is black');
        }
        let nonFinite = 0;
        for (const value of current) if (!Number.isFinite(value)) nonFinite++;
        assert(nonFinite === 0, `${nonFinite} non-finite GI values at measured frame ${index}`);
        comparisons.push(compare(previous, current, surface));
        imageMeans.push(imageMean(current));
        if (sequenceFrames.has(index)) kept.set(index, current);
        previous = current;
    }

    const perFrameMean = comparisons.map((comparison) => comparison.mean);
    const perFrameP95 = comparisons.map((comparison) => comparison.p95);
    const driftFirst = mean(imageMeans.slice(0, DRIFT_FRAMES));
    const driftLast = mean(imageMeans.slice(-DRIFT_FRAMES));
    const result = {
        hysteresis: mode.name,
        budget: budgetText,
        settle_frames: settle,
        frames: comparisons.length,
        // The headline: the two numbers, each averaged over the frame pairs.
        rel_change_mean: round(mean(perFrameMean)),
        rel_change_p95: round(mean(perFrameP95)),
        rel_change_mean_sd: round(standardDeviation(perFrameMean)),
        rel_change_p95_sd: round(standardDeviation(perFrameP95)),
        rel_change_p95_max: round(Math.max(...perFrameP95)),
        image_mean_change: round(mean(comparisons.map((comparison) => comparison.imageMeanChange))),
        image_mean_drift: round(driftLast / driftFirst - 1),
        surface_share: round(comparisons[0].surfaceShare),
        surface_mean_gi: round(mean(comparisons.map((comparison) => comparison.surfaceMean))),
        probes_per_frame_mean: round(mean(probesPerFrame)),
        probes_per_frame_median: median(probesPerFrame),
        probes_share_per_frame: round(mean(probesPerFrame) / scene.probeCount, 5),
    };
    log(
        `${scene.manifest.id} ${mode.name} ${budgetText}: mean ${(result.rel_change_mean * 100).toFixed(3)}%, p95 ${(result.rel_change_p95 * 100).toFixed(3)}% per pixel; ` +
            `image mean moves ${(result.image_mean_change * 100).toFixed(4)}% a frame, drift ${(result.image_mean_drift * 100).toFixed(3)}%; ${result.probes_per_frame_mean} probes a frame (${(result.probes_share_per_frame * 100).toFixed(2)}% of ${scene.probeCount})`,
    );

    const images: Array<{ name: string; base64: string }> = [];
    if (SAVE_IMAGES) {
        const scale = DISPLAY_MEAN / result.surface_mean_gi;
        const ruleName = HYSTERESIS_GIVEN ? `${mode.name.replaceAll(':', '-').replaceAll('/', '-')}-` : '';
        const safeBudget = `${ruleName}${budgetText.replace(':', '-')}`;
        let sequence = 0;
        for (const start of IMAGE_SEQUENCE_STARTS.slice(0, IMAGE_SEQUENCES)) {
            sequence++;
            const first = Math.floor(start * MEASURE_FRAMES);
            const frames = Array.from({ length: IMAGE_SEQUENCE_FRAMES }, (_, offset) => kept.get(first + offset)!);
            for (let offset = 0; offset < frames.length; offset++) {
                images.push({ name: `${safeBudget}-seq${sequence}-frame${offset + 1}`, base64: await pngBase64(exposedImage(frames[offset], scale, WIDTH, HEIGHT), WIDTH, HEIGHT) });
            }
            images.push({
                name: `${safeBudget}-seq${sequence}-difference-first-to-last-x${DIFFERENCE_GAIN}`,
                base64: await pngBase64(differenceImage(frames[0], frames[frames.length - 1], result.surface_mean_gi, WIDTH, HEIGHT), WIDTH, HEIGHT),
            });
        }
    }
    return { result, images };
}

await runThreePage(async ({ device, data, log }) => {
    data.userAgent = navigator.userAgent;
    data.browserVersion = await fullBrowserVersion();
    const manifest = loadAllManifests().find((candidate) => candidate.id === SCENE);
    assert(manifest, `no manifest '${SCENE}'`);
    const bookmark = manifest.cameras[CAMERA - 1];
    assert(bookmark, `${SCENE} has no camera ${CAMERA}`);
    data.scene = SCENE;
    data.camera = bookmark.name;
    data.width = WIDTH;
    data.height = HEIGHT;

    const scene = await startViewerScene(device, manifest, { gpuTiming: true, label: `flicker ${SCENE}` });
    try {
        const field = scene.ddgi.field;
        assert(field, 'DDGI has no field');
        data.probes = scene.probeCount;
        const view = await TestView.create(device, scene.platform);

        const warm = AFTER_WARMUP_FRAMES >= 0 ? { passes: 0 } : await warmUp(scene);
        data.warmup_passes = round(warm.passes);
        data.after_warmup_frames = AFTER_WARMUP_FRAMES;

        // Converge with the fixed 0.97 rule (see the file comment) with whole passes split into equal frames.
        applyHysteresisMode(scene.ddgi.settings, FIXED_097);
        const framesPerPass = Math.ceil(scene.probeCount / MAX_PROBES_PER_FRAME);
        const unitsPerFrame = Math.ceil(scene.probeCount / framesPerPass);
        let blockMean = imageMean((await view.render(field, bookmark, WIDTH, HEIGHT)).indirect);
        let blocks = 0;
        let settled = false;
        for (let block = 1; block <= (AFTER_WARMUP_FRAMES >= 0 ? 0 : MAX_BLOCKS); block++) {
            for (let frame = 0; frame < BLOCK_PASSES * framesPerPass; frame++) {
                scene.frame({ units: unitsPerFrame }, FRAME_SECONDS);
                if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
            }
            const next = imageMean((await view.render(field, bookmark, WIDTH, HEIGHT)).indirect);
            const change = next / blockMean - 1;
            blockMean = next;
            blocks = block;
            log(`converge block ${block}: image mean ${next.toFixed(6)}, ${(change * 100).toFixed(3)}% since the last block`);
            if (block >= MIN_BLOCKS && Math.abs(change) < CONVERGED_TOLERANCE) {
                settled = true;
                break;
            }
        }
        data.converge_blocks = blocks;
        data.converge_passes = warm.passes + blocks * BLOCK_PASSES;
        data.converged_image_mean = round(blockMean);
        assert(settled || AFTER_WARMUP_FRAMES >= 0, `the GI image mean did not settle within ±${CONVERGED_TOLERANCE * 100}% in ${MAX_BLOCKS} blocks`);

        const settings: Record<string, unknown> = {};
        const images: Array<{ name: string; base64: string }> = [];
        data.settings = settings;
        data.images = images;
        for (const mode of HYSTERESIS_MODES) {
            for (const budgetText of BUDGETS) {
                const { result, images: settingImages } = await measureSetting(scene, view, bookmark, mode, budgetText, log);
                settings[HYSTERESIS_GIVEN ? `${mode.name}/${budgetText}` : budgetText] = result;
                images.push(...settingImages);
            }
        }
    } finally {
        scene.dispose();
    }
});
