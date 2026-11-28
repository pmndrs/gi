/**
 * pmndrs/gi#54's comparison on the cost page (`VITE_RC_SHARED_BOUNCE_BENCH=1`):
 * the shared-cell second bounce against GI-1.2's rays on a budget of 20k cells,
 * at equal GPU time. GPU time is the sum of the technique's pass timestamps.
 *
 * Steps (`VITE_RC_SHARED_BOUNCE_STEP`), so each timed run is short enough for
 * the clock rule (a laptop GPU throttles within a minute of load):
 *
 * - `all` (the default): the whole comparison in one run, as first written.
 *   The calibration and both trials of both modes: well over a minute of load.
 * - `calibrate`: budgeted GI-1.2 at 20k (the target ms per frame), then a
 *   bisection for the shared-cell budget that takes as long. No reference.
 *   With `VITE_RC_SHARED_BOUNCE_CALIBRATE_POINT=mode:budget`, that one point
 *   (Sponza's bisection in one run throttled the laptop GPU).
 * - `trial`: one exposure of one mode (`VITE_RC_SHARED_BOUNCE_MODE`; the
 *   shared budget from `VITE_RC_SHARED_BOUNCE_BUDGET`) for 64 × the target
 *   (`VITE_RC_SHARED_BOUNCE_TARGET_MS`) of cumulative GPU time from a cold
 *   cache, against the per-hit reference made in the same run; on two-room
 *   then 12 × the target after the lamp doubles and the crate moves.
 * - `images` (two-room): per-hit GI-1.2, budgeted GI-1.2 at 20k and the shared
 *   bounce at the doorway views, each a 24-frame mean after 96 frames, drawn
 *   on the page for the runner's screenshot, with numbers against per hit and
 *   the light in the views' darkest blocks (where a leak would show).
 *
 * The reference is per-hit GI-1.2 (96 warm frames, then a 24-frame mean of its
 * denoised irradiance), not Cycles: the numbers say how far each estimator is
 * from GI-1.2's own answer.
 */
import type { GpuTimer } from '../../../../packages/ddgi-core/src/platform/timing/gpu-timer.ts';
import type { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { accuracyNumbers, rgbOfRgba, type RgbImage } from '../radiance-cache-accuracy/accuracy.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import type { PageCamera } from '../radiance-cache-probes/screen.ts';

type Mode = 'gi1-rays' | 'shared-cell';
type Step = 'all' | 'calibrate' | 'trial' | 'images';

const env = import.meta.env;
const STEP = (env.VITE_RC_SHARED_BOUNCE_STEP as Step | undefined) ?? 'all';
if (!['all', 'calibrate', 'trial', 'images'].includes(STEP)) throw new Error(`VITE_RC_SHARED_BOUNCE_STEP is all, calibrate, trial or images, got ${STEP}`);

/** The scenes the comparison runs (`VITE_RC_SHARED_BOUNCE_SCENES`, comma-separated). */
export function sharedBounceScenes(): string[] {
    const listed = (env.VITE_RC_SHARED_BOUNCE_SCENES as string | undefined)?.split(',').map((scene) => scene.trim()).filter(Boolean);
    return listed ?? (STEP === 'images' ? ['two-room'] : ['two-room', 'sponza']);
}

/** Budgeted GI-1.2's budget, what the shared bounce is timed against. */
const BASELINE_BUDGET = 20000;
const STILL_TARGETS = 64;
const CHANGED_TARGETS = 12;
/** Two-room's doorway views (scenes/manifests/two-room.json). */
const DOORWAY_VIEWS = ['through-the-doorway', 'dark-room-leak-view'] as const;

function numberFromEnv(name: string): number {
    const text = env[name] as string | undefined;
    const value = Number(text);
    if (text === undefined || !Number.isFinite(value) || value <= 0) throw new Error(`${name} must be a positive number, got ${String(text)}`);
    return value;
}

export interface SharedBounceScene {
    readonly device: GPUDevice;
    readonly technique: RadianceCacheTechnique;
    readonly timer: GpuTimer;
    readonly frame: () => void;
    readonly width: number;
    readonly height: number;
    /** Two-room: the lamp doubled and the crate moved 1 m (true), or back. */
    readonly changeScene?: (changed: boolean) => void;
    /** The scene's bookmarks, and a way to look through one. */
    readonly cameras: readonly { readonly name: string; readonly camera: PageCamera }[];
    readonly setCamera: (camera: PageCamera) => void;
    readonly id: string;
}

export async function compareSharedBounce(scene: SharedBounceScene): Promise<Record<string, unknown>> {
    const { device, technique, timer, frame, width, height, changeScene } = scene;
    const sample = async (): Promise<number> => {
        frame();
        await timer.drain();
        const frames = timer.takeSamples();
        if (frames.length !== 1) throw new Error('expected one GPU timestamp sample');
        const ms = frames[0].passes.reduce((sum, pass) => sum + pass.gpuMs, 0);
        if (!(ms > 0)) throw new Error('no GPU time measured');
        return ms;
    };
    await timer.drain();
    timer.takeSamples();
    const batch = async (count: number): Promise<number> => {
        let sum = 0;
        for (let i = 0; i < count; i++) sum += await sample();
        return sum / count;
    };
    const image = async (): Promise<RgbImage> => {
        const texture = technique.probes?.denoisedIrradiance;
        if (!texture) throw new Error('missing denoised irradiance');
        const values = halves(await readTexture(device, texture, 8, width, height));
        if (!values.every(Number.isFinite)) throw new Error('nonfinite irradiance');
        return rgbOfRgba(values, width, height);
    };
    const set = (mode: Mode, cellShading: 'per-hit' | 'budgeted', budget: number): void => {
        technique.settings.secondBounce = mode;
        technique.settings.cellShading = cellShading;
        technique.settings.cellShadingBudget = budget;
    };
    /** 96 frames from a cold cache, then the mean of 24 frames' denoised irradiance. */
    const meanImage = async (): Promise<RgbImage> => {
        technique.resetHistory();
        await batch(96);
        const sum = new Float32Array(width * height * 3);
        for (let i = 0; i < 24; i++) {
            await sample();
            const current = await image();
            for (let j = 0; j < sum.length; j++) sum[j] += current.rgb[j] / 24;
        }
        if (!sum.some((value) => value > 0)) throw new Error('black image');
        return { width, height, rgb: sum };
    };
    const reference = async (): Promise<RgbImage> => {
        set('gi1-rays', 'per-hit', BASELINE_BUDGET);
        return meanImage();
    };
    const calibration: { mode: string; budget: number; ms: number }[] = [];
    const calibrate = async (mode: Mode, budget: number): Promise<number> => {
        set(mode, 'budgeted', budget);
        technique.resetHistory();
        await batch(24);
        const ms = await batch(20);
        calibration.push({ mode, budget, ms });
        return ms;
    };
    const calibrateBoth = async (): Promise<{ target: number; bestBudget: number }> => {
        const target = await calibrate('gi1-rays', BASELINE_BUDGET);
        let bestBudget = BASELINE_BUDGET;
        let gap = Infinity;
        let low = 0;
        let high = 160000;
        let budget = BASELINE_BUDGET;
        for (let attempt = 0; attempt < 7; attempt++) {
            const ms = await calibrate('shared-cell', budget);
            if (Math.abs(target - ms) < gap) {
                bestBudget = budget;
                gap = Math.abs(target - ms);
            }
            if (gap <= 0.025 * target) break;
            if (ms > target) high = budget;
            else low = budget;
            budget = Math.round((low + high) / 2);
        }
        return { target, bestBudget };
    };
    // Stop at the first complete frame reaching the target. The overshoot and frame count stay
    // visible with each error measurement.
    const exposure = async (targetMs: number, ref: RgbImage) => {
        let gpuMs = 0;
        let frames = 0;
        while (gpuMs < targetMs) {
            gpuMs += await sample();
            frames++;
        }
        return { targetMs, gpuMs, frames, msPerFrame: gpuMs / frames, ...accuracyNumbers(ref, await image()) };
    };
    const measure = async (mode: Mode, budget: number, target: number, before: RgbImage, after: RgbImage | undefined) => {
        changeScene?.(false);
        set(mode, 'budgeted', budget);
        technique.resetHistory();
        const still = await exposure(STILL_TARGETS * target, before);
        if (technique.worldCache?.secondBounceInUse !== mode) throw new Error('wrong second-bounce estimator');
        let changed;
        if (after && changeScene) {
            changeScene(true);
            changed = await exposure(CHANGED_TARGETS * target, after);
            changeScene(false);
        }
        return { mode, budget, still, changed };
    };
    const references = async (): Promise<{ before: RgbImage; after: RgbImage | undefined }> => {
        const before = await reference();
        let after: RgbImage | undefined;
        if (changeScene) {
            changeScene(true);
            after = await reference();
            changeScene(false);
        }
        return { before, after };
    };
    const referenceNote = '96 warm frames plus a 24-frame mean of per-hit GI-1.2 denoised irradiance';

    if (STEP === 'calibrate') {
        // One point (`mode:budget`), when the whole bisection is too long for the clock rule (Sponza).
        const point = env.VITE_RC_SHARED_BOUNCE_CALIBRATE_POINT as string | undefined;
        if (point) {
            const [mode, budgetText] = point.split(':');
            const budget = Number(budgetText);
            if ((mode !== 'gi1-rays' && mode !== 'shared-cell') || !Number.isInteger(budget) || budget <= 0) throw new Error(`VITE_RC_SHARED_BOUNCE_CALIBRATE_POINT is mode:budget, got ${point}`);
            const ms = await calibrate(mode, budget);
            return { step: STEP, calibration, point: { mode, budget, ms } };
        }
        const { target, bestBudget } = await calibrateBoth();
        return { step: STEP, calibration, targetMs: target, sharedBudget: bestBudget };
    }
    if (STEP === 'trial') {
        const mode = env.VITE_RC_SHARED_BOUNCE_MODE as Mode | undefined;
        if (mode !== 'gi1-rays' && mode !== 'shared-cell') throw new Error(`VITE_RC_SHARED_BOUNCE_MODE is gi1-rays or shared-cell, got ${String(mode)}`);
        const target = numberFromEnv('VITE_RC_SHARED_BOUNCE_TARGET_MS');
        const budget = mode === 'gi1-rays' ? BASELINE_BUDGET : numberFromEnv('VITE_RC_SHARED_BOUNCE_BUDGET');
        const { before, after } = await references();
        const measurement = await measure(mode, budget, target, before, after);
        return { step: STEP, trial: (env.VITE_RC_SHARED_BOUNCE_TRIAL as string | undefined) ?? '1', measurement, reference: referenceNote };
    }
    if (STEP === 'images') return doorwayImages(scene, meanImage, set);

    const { before, after } = await references();
    const { target, bestBudget } = await calibrateBoth();
    const measurements = [];
    // Reverse the order for the replicate to reduce clock/thermal order effects.
    for (const mode of ['gi1-rays', 'shared-cell', 'shared-cell', 'gi1-rays'] as const) {
        measurements.push(await measure(mode, mode === 'gi1-rays' ? BASELINE_BUDGET : bestBudget, target, before, after));
    }
    return { calibration, measurements, reference: referenceNote };
}

// ---- The doorway images ------------------------------------------------------------------------------

const PANEL_SCALE = 3;

interface DarkBlocks {
    /** The 8 × 8 blocks in the per-hit image's darkest quarter (by mean rgb). */
    readonly blocks: number[];
    readonly blocksX: number;
}

function blockMeans(image: RgbImage): Float64Array {
    const blocksX = Math.floor(image.width / 8);
    const blocksY = Math.floor(image.height / 8);
    const means = new Float64Array(blocksX * blocksY);
    for (let y = 0; y < blocksY * 8; y++) {
        for (let x = 0; x < blocksX * 8; x++) {
            const pixel = (y * image.width + x) * 3;
            means[Math.floor(y / 8) * blocksX + Math.floor(x / 8)] += (image.rgb[pixel] + image.rgb[pixel + 1] + image.rgb[pixel + 2]) / 192;
        }
    }
    return means;
}

function darkBlocks(reference: RgbImage): DarkBlocks {
    const means = blockMeans(reference);
    const order = [...means.keys()].sort((a, b) => means[a] - means[b]);
    return { blocks: order.slice(0, Math.floor(order.length / 4)), blocksX: Math.floor(reference.width / 8) };
}

/** Light in the reference's darkest blocks: the image's mean over them, and that over the reference's. */
function darkLight(reference: RgbImage, test: RgbImage, dark: DarkBlocks): { referenceMean: number; ratio: number } {
    const ref = blockMeans(reference);
    const tst = blockMeans(test);
    let refSum = 0;
    let testSum = 0;
    for (const block of dark.blocks) {
        refSum += ref[block];
        testSum += tst[block];
    }
    return { referenceMean: refSum / dark.blocks.length, ratio: testSum / Math.max(refSum, 1e-30) };
}

async function doorwayImages(
    scene: SharedBounceScene,
    meanImage: () => Promise<RgbImage>,
    set: (mode: Mode, cellShading: 'per-hit' | 'budgeted', budget: number) => void,
): Promise<Record<string, unknown>> {
    if (scene.id !== 'two-room') throw new Error('the doorway images are of two-room');
    const sharedBudget = numberFromEnv('VITE_RC_SHARED_BOUNCE_BUDGET');
    const modes = [
        { name: 'per-hit GI-1.2', apply: () => set('gi1-rays', 'per-hit', BASELINE_BUDGET) },
        { name: `budgeted GI-1.2, ${BASELINE_BUDGET / 1000}k`, apply: () => set('gi1-rays', 'budgeted', BASELINE_BUDGET) },
        { name: `shared-cell, ${Math.round(sharedBudget / 1000)}k`, apply: () => set('shared-cell', 'budgeted', sharedBudget) },
    ];
    const canvas = document.createElement('canvas');
    const panelWidth = Math.floor(scene.width / PANEL_SCALE);
    const panelHeight = Math.floor(scene.height / PANEL_SCALE);
    canvas.width = panelWidth * modes.length;
    canvas.height = panelHeight * (DOORWAY_VIEWS.length + 1);
    // Scaled to the viewport, so the runner's screenshot holds all three rows.
    canvas.style.cssText = 'position:fixed;left:0;top:0;z-index:10;background:#000;width:auto;height:auto;max-width:100vw;max-height:100vh';
    document.body.style.overflow = 'hidden';
    document.body.append(canvas);
    const context = canvas.getContext('2d');
    if (!context) throw new Error('no 2d context');
    const results: Record<string, unknown> = {};
    for (const [row, view] of DOORWAY_VIEWS.entries()) {
        const bookmark = scene.cameras.find((camera) => camera.name === view);
        if (!bookmark) throw new Error(`two-room has no ${view} camera`);
        scene.setCamera(bookmark.camera);
        const images: RgbImage[] = [];
        for (const mode of modes) {
            mode.apply();
            images.push(await meanImage());
        }
        const reference = images[0];
        const dark = darkBlocks(reference);
        const exposure = 0.18 / Math.max(meanOf(reference), 1e-6);
        const viewResults: Record<string, unknown> = {};
        images.forEach((picture, column) => {
            drawPanel(context, picture, column * panelWidth, row * panelHeight, exposure, `${view}: ${modes[column].name}`);
            viewResults[modes[column].name] = { ...accuracyNumbers(reference, picture), meanIrradiance: meanOf(picture), darkQuarter: darkLight(reference, picture, dark) };
        });
        // The last row: the leak view's differences from per hit (red: more light, blue: less).
        if (view === 'dark-room-leak-view') {
            for (let column = 1; column < images.length; column++) {
                drawDifference(context, reference, images[column], column * panelWidth, DOORWAY_VIEWS.length * panelHeight, exposure, `${modes[column].name} - per hit`);
            }
        }
        results[view] = viewResults;
    }
    return { step: 'images', views: results, note: 'denoised irradiance, 24-frame means after 96 frames; darkQuarter: light in the per-hit image\'s darkest quarter of 8x8 blocks, over per hit\'s' };
}

function meanOf(image: RgbImage): number {
    let sum = 0;
    for (const value of image.rgb) sum += value;
    return sum / image.rgb.length;
}

function drawPanel(context: CanvasRenderingContext2D, image: RgbImage, left: number, top: number, exposure: number, label: string): void {
    const panelWidth = Math.floor(image.width / PANEL_SCALE);
    const panelHeight = Math.floor(image.height / PANEL_SCALE);
    const pixels = context.createImageData(panelWidth, panelHeight);
    for (let y = 0; y < panelHeight; y++) {
        for (let x = 0; x < panelWidth; x++) {
            // The texture's first row is the image's top (its readback order).
            const source = ((y * PANEL_SCALE) * image.width + x * PANEL_SCALE) * 3;
            const target = (y * panelWidth + x) * 4;
            for (let channel = 0; channel < 3; channel++) {
                const value = image.rgb[source + channel] * exposure;
                pixels.data[target + channel] = Math.round(255 * Math.pow(value / (1 + value), 1 / 2.2));
            }
            pixels.data[target + 3] = 255;
        }
    }
    context.putImageData(pixels, left, top);
    label2d(context, label, left, top);
}

function drawDifference(context: CanvasRenderingContext2D, reference: RgbImage, test: RgbImage, left: number, top: number, exposure: number, label: string): void {
    const panelWidth = Math.floor(reference.width / PANEL_SCALE);
    const panelHeight = Math.floor(reference.height / PANEL_SCALE);
    const pixels = context.createImageData(panelWidth, panelHeight);
    for (let y = 0; y < panelHeight; y++) {
        for (let x = 0; x < panelWidth; x++) {
            const source = ((y * PANEL_SCALE) * reference.width + x * PANEL_SCALE) * 3;
            const target = (y * panelWidth + x) * 4;
            let difference = 0;
            for (let channel = 0; channel < 3; channel++) difference += (test.rgb[source + channel] - reference.rgb[source + channel]) / 3;
            const value = Math.min(1, Math.abs(difference) * exposure * 4);
            pixels.data[target] = difference > 0 ? Math.round(255 * value) : 0;
            pixels.data[target + 2] = difference < 0 ? Math.round(255 * value) : 0;
            pixels.data[target + 3] = 255;
        }
    }
    context.putImageData(pixels, left, top);
    label2d(context, label, left, top);
}

function label2d(context: CanvasRenderingContext2D, label: string, left: number, top: number): void {
    context.font = '12px sans-serif';
    context.fillStyle = 'rgba(0,0,0,0.6)';
    context.fillRect(left, top, context.measureText(label).width + 8, 16);
    context.fillStyle = '#fff';
    context.fillText(label, left + 4, top + 12);
}
