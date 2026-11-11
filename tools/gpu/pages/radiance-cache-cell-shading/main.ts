/**
 * RC-L1: lever 1, budgeted cell shading (`cellShading: 'budget'`, id Tech 8's
 * way), against Capsaicin's per-hit shading (`perHit`), at 1280 × 720 through
 * the engine-free page screen (radiance-cache-probes' `PageScreen`). Sections
 * 6 and 7 of `radiance-cache-world-cache` (constant sky, closed box) run in
 * both modes on that page; this one has the rest of the row's done-when.
 *
 * 1. **Converged image, both modes** (Cornell, two-room and Sponza from their
 *    first bookmark; the budget above the cells in use, max age 1, one
 *    sample): from an empty cache, 300 still frames, then 32 measured. The
 *    mean interpolated E (before the denoiser) and the mean denoised E over
 *    the surface pixels, each averaged over the 32 frames; on a budget each
 *    within 3% of `perHit`'s (the gate, stated before the run). Flicker before
 *    and after the denoiser (T1.12's measure, as `radiance-cache-denoiser`)
 *    beside `perHit`'s, as information. No NaN, infinite or negative value in
 *    either image.
 * 2. **The counters on a smaller budget, exactly** (the sunlit corner of
 *    `radiance-cache-world-cache`, black sky, so every draw finds the sun and
 *    traces a shadow ray): budget 5,000 (the cells in use are about 65,000),
 *    3 samples, max age 4, after 12 frames; then the budget above the cells in
 *    use. Lit cells = min(budget, cells in use) and shadow rays = lit cells × 3,
 *    exactly. The cells to light, read back, are distinct cells in use, and
 *    their age classes (from the last-lit frames read before the frame) are
 *    what the CPU twin's choice says: every cell of the classes before the
 *    cutoff, the twin's count of the cutoff class, none after.
 * 3. **Lag** (two-room, Capsaicin's defaults otherwise, max age 8, one
 *    sample): `perHit`, and the budget at 1, 1/4 and 1/16 of the cells in use
 *    (counted on the lit room). Its lamp off for 30 frames, then on, for 720
 *    frames: frames until the mean interpolated and denoised E are within 10%
 *    of the converged value, first, and for good (read every frame for the
 *    first 128, then every 8th, so a lag past 128 frames is known to 8). The converged value is
 *    `perHit`'s (the mean of its last 32 frames): every mode converges to the
 *    same image (section 1), but a small budget isn't there yet by the end,
 *    so its own end value would understate its lag (a first run measured
 *    against each run's own end did: 1/16's lamp-on end value was 4.8% below
 *    `perHit`'s). Each run's own end value is reported beside it. Then a cut
 *    to the second bookmark (`resetScreenProbes`, the world cache warm): the
 *    same. Information.
 *
 * Correctness only, no timing (`radiance-cache-timing` times both modes).
 * Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc-l1 cell shading" -- pnpm test:gpu --filter radiance-cache-cell-shading`.
 */

import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { halfBitsToFloat } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { cellShadingClass, chooseCells } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/cell-shading-cpu.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { assert, drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest } from '../ddgi-shared/harness.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';
import { readRange, runFrames, startScenario, sun, SUN_TRAVEL } from '../radiance-cache-world-cache/corner.ts';

const WIDTH = 1280;
const HEIGHT = 720;

/** Section 1: frames before the measured window (the denoiser's history converges by about 192), the window, the tolerance. */
const CONVERGED_FROM_FRAME = 300;
const MEASURED_FRAMES = 32;
const MEAN_TOLERANCE = 0.03;
/** "A budget at least the cells in use, max age 1, one sample" (the done-when). */
const AMPLE_BUDGET: Partial<RadianceCacheSettings> = { cellShading: 'budget', cellShadingBudget: 1_000_000, cellShadingMaxAge: 1, cellShadingSamples: 1 };

/** Section 2. */
const SMALL_BUDGET = 5000;
const COUNTER_SAMPLES = 3;
const COUNTER_MAX_AGE = 4;
const COUNTER_WARM_FRAMES = 12;

/** Section 3: frames before the lamp comes on, frames after each event, the converged window at their end, the lag's tolerance. */
const LAMP_OFF_FRAMES = 30;
const EVENT_FRAMES = 720;
const LAG_TOLERANCE = 0.1;
const LAG_BUDGET_SHARES = [1, 1 / 4, 1 / 16] as const;
/** Section 3 reads the images every frame up to this one after an event, then every LAG_COARSE_STEP-th. */
const LAG_FINE_FRAMES = 128;
const LAG_COARSE_STEP = 8;


// ---- A run: a platform with the radiance cache over one scene, through the page screen -----------

interface Run {
    readonly platform: GIPlatformImpl;
    readonly technique: RadianceCacheTechnique;
    readonly screen: PageScreen;
    /** The camera the next frames are drawn from (a cut changes it). */
    camera: PageCamera;
    /** The scene's lights, for the pages that switch them (procedural scenes only). */
    readonly lights: readonly LightHandle[];
    frame(): Promise<void>;
    dispose(): void;
}

function bookmark(manifest: SceneManifest, index: number, far: number): PageCamera {
    const camera = manifest.cameras[index];
    return { position: [...camera.position], target: [...camera.target], fovYDeg: camera.fovYDeg, near: 0.05, far };
}

type LoadedScene = Awaited<ReturnType<typeof loadScene>>;
const loadedScenes = new Map<string, LoadedScene>();

/** Sponza through the viewer's loader once (the model only is GI geometry), reused by every run. */
async function loadedSponza(manifest: SceneManifest): Promise<LoadedScene> {
    const cached = loadedScenes.get(manifest.id);
    if (cached) return cached;
    const loaded = await loadScene(manifest, () => {});
    const content = new Set<object>();
    loaded.content.traverse((object) => content.add(object));
    loaded.scene.traverse((object) => {
        if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    loaded.scene.updateMatrixWorld();
    loadedScenes.set(manifest.id, loaded);
    return loaded;
}

async function startRun(device: GPUDevice, sceneId: string, settings: Partial<RadianceCacheSettings>, options: { readonly lampIntensity?: number } = {}): Promise<Run> {
    const manifest = loadManifest(sceneId);
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-cell-shading ${sceneId}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    const lights: LightHandle[] = [];
    let beforeFrame = (): void => {};
    if (sceneId === 'sponza') {
        const loaded = await loadedSponza(manifest);
        const sync = new SceneSync(platform.scene, { device, warn: () => {} });
        sync.environment = platformSkyEnvironment(manifest);
        beforeFrame = () => sync.update(loaded.scene);
    } else {
        describeScene(platform, manifest, {}, lights);
        if (options.lampIntensity !== undefined) for (const light of lights) platform.scene.updateLight(light, { intensity: options.lampIntensity });
    }
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const run: Run = {
        platform,
        technique,
        screen,
        camera: bookmark(manifest, 0, sceneId === 'sponza' ? 1000 : 100),
        lights,
        async frame() {
            beforeFrame();
            const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(run.camera, WIDTH, HEIGHT) };
            const encoder = device.createCommandEncoder({ label: `radiance-cache-cell-shading ${sceneId}` });
            platform.encode(encoder, input);
            screen.encode(encoder, run.camera);
            platform.encodeScreen(encoder, screen.screen);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        },
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
    // A frame places the scene; then the mesh trees, and an empty cache to start from.
    await run.frame();
    await platform.rays.whenBuilt();
    await run.frame();
    technique.resetHistory();
    return run;
}

// ---- Images ----------------------------------------------------------------------------------------

async function readHalf4(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
    return halves(await readTexture(device, texture, 8, WIDTH, HEIGHT));
}

/** Pixels with a surface (depth below 1). */
async function surfacePixels(device: GPUDevice, run: Run): Promise<Uint32Array> {
    const bytes = await readTexture(device, run.screen.screen.depth, 4, WIDTH, HEIGHT);
    const depth = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    return Uint32Array.from({ length: WIDTH * HEIGHT }, (_, pixel) => pixel).filter((pixel) => depth[pixel] !== 1);
}

interface Images {
    readonly interpolated: Float32Array;
    readonly denoised: Float32Array;
}

async function readImages(device: GPUDevice, run: Run): Promise<Images> {
    const probes = run.technique.probes;
    const denoised = run.technique.denoiser?.output;
    assert(probes && denoised, 'no interpolated or denoised irradiance after a frame');
    return { interpolated: await readHalf4(device, probes.irradiance), denoised: await readHalf4(device, denoised) };
}

function imageMean(image: Float32Array, surface: Uint32Array): number {
    let sum = 0;
    for (const pixel of surface) sum += (image[pixel * 4] + image[pixel * 4 + 1] + image[pixel * 4 + 2]) / 3;
    return sum / Math.max(surface.length, 1);
}

/** T1.12's measure between two frames, over surface pixels: mean and p95 of |ΔE| (rgb mean) over the image's mean (as radiance-cache-denoiser). */
function frameChange(previous: Float32Array, current: Float32Array, surface: Uint32Array): { mean: number; p95: number } {
    const mean = imageMean(current, surface);
    const changes = new Float32Array(surface.length);
    let total = 0;
    surface.forEach((pixel, index) => {
        let change = 0;
        for (let channel = 0; channel < 3; channel++) change += Math.abs(current[pixel * 4 + channel] - previous[pixel * 4 + channel]) / 3;
        changes[index] = change / mean;
        total += changes[index];
    });
    changes.sort();
    return { mean: total / Math.max(surface.length, 1), p95: changes.length === 0 ? 0 : changes[Math.min(changes.length - 1, Math.floor(0.95 * changes.length))] };
}

function badValues(image: Float32Array, surface: Uint32Array): number {
    let bad = 0;
    for (const pixel of surface) {
        for (let channel = 0; channel < 3; channel++) {
            const value = image[pixel * 4 + channel];
            if (!Number.isFinite(value) || value < 0) bad++;
        }
    }
    return bad;
}

const average = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

async function readCounters(device: GPUDevice, run: Run): Promise<{ usedCells: number; litCells: number; shadowRays: number } | null> {
    const cache = run.technique.worldCache;
    if (!cache || cache.preparedCellShading !== 'budget') return null;
    const counters = new Uint32Array(await readRange(device, cache.buffers.words, 0, 64 * 4));
    return { usedCells: counters[HASH_GRID_COUNTER.usedCellCount], litCells: counters[HASH_GRID_COUNTER.litCellCount], shadowRays: counters[HASH_GRID_COUNTER.cellShadowRayCount] };
}

// ---- Section 1: the converged image in both modes ----------------------------------------------------

interface ConvergedImage {
    readonly meanInterpolated: number;
    readonly meanDenoised: number;
    readonly interpolatedFlickerMean: number;
    readonly interpolatedFlickerP95: number;
    readonly denoisedFlickerMean: number;
    readonly denoisedFlickerP95: number;
    readonly badValues: number;
    readonly counters: Awaited<ReturnType<typeof readCounters>>;
}

async function convergedImage(device: GPUDevice, sceneId: string, settings: Partial<RadianceCacheSettings>): Promise<ConvergedImage> {
    const run = await startRun(device, sceneId, settings);
    try {
        const surface = await surfacePixels(device, run);
        for (let frame = 0; frame < CONVERGED_FROM_FRAME; frame++) await run.frame();
        let previous = await readImages(device, run);
        const flicker = { interpolated: [] as { mean: number; p95: number }[], denoised: [] as { mean: number; p95: number }[] };
        const means = { interpolated: [] as number[], denoised: [] as number[] };
        let bad = 0;
        for (let frame = 0; frame < MEASURED_FRAMES; frame++) {
            await run.frame();
            const current = await readImages(device, run);
            bad += badValues(current.interpolated, surface) + badValues(current.denoised, surface);
            flicker.interpolated.push(frameChange(previous.interpolated, current.interpolated, surface));
            flicker.denoised.push(frameChange(previous.denoised, current.denoised, surface));
            means.interpolated.push(imageMean(current.interpolated, surface));
            means.denoised.push(imageMean(current.denoised, surface));
            previous = current;
        }
        return {
            meanInterpolated: average(means.interpolated),
            meanDenoised: average(means.denoised),
            interpolatedFlickerMean: average(flicker.interpolated.map((change) => change.mean)),
            interpolatedFlickerP95: average(flicker.interpolated.map((change) => change.p95)),
            denoisedFlickerMean: average(flicker.denoised.map((change) => change.mean)),
            denoisedFlickerP95: average(flicker.denoised.map((change) => change.p95)),
            badValues: bad,
            counters: await readCounters(device, run),
        };
    } finally {
        run.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

async function checkConverged(device: GPUDevice, failures: string[]): Promise<Record<string, unknown>> {
    const results: Record<string, unknown> = {};
    for (const sceneId of ['cornell-box', 'two-room', 'sponza']) {
        const perHit = await convergedImage(device, sceneId, {});
        const budget = await convergedImage(device, sceneId, AMPLE_BUDGET);
        const interpolatedRatio = budget.meanInterpolated / perHit.meanInterpolated;
        const denoisedRatio = budget.meanDenoised / perHit.meanDenoised;
        if (!(Math.abs(interpolatedRatio - 1) <= MEAN_TOLERANCE)) failures.push(`${sceneId}: the budget's mean interpolated E is ${interpolatedRatio.toFixed(4)} x perHit's`);
        if (!(Math.abs(denoisedRatio - 1) <= MEAN_TOLERANCE)) failures.push(`${sceneId}: the budget's mean denoised E is ${denoisedRatio.toFixed(4)} x perHit's`);
        if (perHit.badValues + budget.badValues > 0) failures.push(`${sceneId}: ${perHit.badValues} (perHit) and ${budget.badValues} (budget) NaN, infinite or negative values`);
        if (budget.counters && budget.counters.litCells !== budget.counters.usedCells) failures.push(`${sceneId}: the ample budget lit ${budget.counters.litCells} of ${budget.counters.usedCells} cells in use`);
        results[sceneId] = { perHit, budget, interpolatedRatio, denoisedRatio };
    }
    return results;
}

// ---- Section 2: the counters on a smaller budget ------------------------------------------------------

async function checkCounters(device: GPUDevice, failures: string[]): Promise<Record<string, unknown>> {
    const results: Record<string, unknown> = {};
    for (const budget of [SMALL_BUDGET, 1_000_000]) {
        const label = `counters at budget ${budget}`;
        const fail = (message: string): void => void failures.push(`${label}: ${message}`);
        const scenario = await startScenario(device, `cell shading ${budget}`, [sun(SUN_TRAVEL)], false, {
            cellShading: 'budget',
            cellShadingBudget: budget,
            cellShadingSamples: COUNTER_SAMPLES,
            cellShadingMaxAge: COUNTER_MAX_AGE,
        });
        try {
            await runFrames(device, scenario, COUNTER_WARM_FRAMES);
            const cache = scenario.cache;
            const lastLitWord = cache.layout.words.cellShadingLastLitBuffer;
            assert(lastLitWord !== null, 'a budget cache without its last-lit words');
            // The last-lit frames before the frame the choice is checked on (the frame overwrites the lit cells').
            const lastLitBefore = new Uint32Array(await readRange(device, cache.buffers.words, lastLitWord * 4, cache.capacity.numCells * 4));
            await runFrames(device, scenario, 1);

            const counters = new Uint32Array(await readRange(device, cache.buffers.words, 0, 64 * 4));
            const usedCells = counters[HASH_GRID_COUNTER.usedCellCount];
            const litCells = counters[HASH_GRID_COUNTER.litCellCount];
            const shadowRays = counters[HASH_GRID_COUNTER.cellShadowRayCount];
            if (litCells !== Math.min(budget, usedCells)) fail(`${litCells} cells lit, min(budget, ${usedCells} in use) is ${Math.min(budget, usedCells)}`);
            if (shadowRays !== litCells * COUNTER_SAMPLES) fail(`${shadowRays} shadow rays for ${litCells} lit cells × ${COUNTER_SAMPLES} samples`);

            const lists = cache.rayLists;
            assert(lists, 'no per-ray lists after a frame');
            const words = new Uint32Array(await readRange(device, lists.buffer, 0, lists.buffer.size));
            const layout = lists.layout;
            const cellOf = (visibility: number): number => words[layout.visibilityCellBuffer + visibility];
            const used = words.slice(layout.usedCellBuffer, layout.usedCellBuffer + usedCells);
            const chosen = words.slice(layout.chosenCellBuffer, layout.chosenCellBuffer + litCells);
            const usedCellSet = new Set(Array.from(used, cellOf));
            if (usedCellSet.size !== usedCells) fail(`the ${usedCells} cells in use are ${usedCellSet.size} distinct cells`);
            const chosenCells = Array.from(chosen, cellOf);
            if (new Set(chosenCells).size !== litCells) fail(`the ${litCells} cells to light are ${new Set(chosenCells).size} distinct cells`);
            const notInUse = chosenCells.filter((cell) => !usedCellSet.has(cell)).length;
            if (notInUse > 0) fail(`${notInUse} cells to light aren't in use`);

            // This frame's index: a cell lit this frame holds it plus one.
            const lastLitAfter = new Uint32Array(await readRange(device, cache.buffers.words, (lastLitWord + chosenCells[0]) * 4, 4))[0];
            const frameIndex = lastLitAfter - 1;
            const classOf = (cell: number): number => cellShadingClass(lastLitBefore[cell], frameIndex, COUNTER_MAX_AGE);
            const twin = chooseCells(Array.from(used, (visibility) => classOf(cellOf(visibility))), COUNTER_MAX_AGE + 1, budget);
            const chosenByClass = new Array<number>(COUNTER_MAX_AGE + 1).fill(0);
            for (const cell of chosenCells) chosenByClass[classOf(cell)]++;
            const expectedByClass = twin.histogram.map((count, ageClass) => (ageClass < twin.cutoffClass ? count : ageClass === twin.cutoffClass ? twin.litInCutoffClass : 0));
            if (chosenByClass.some((count, ageClass) => count !== expectedByClass[ageClass])) fail(`lit cells by age class ${chosenByClass.join(', ')}, the twin's choice ${expectedByClass.join(', ')}`);
            // The list is in rank order: classes never go back down along it.
            let outOfOrder = 0;
            for (let index = 1; index < chosenCells.length; index++) if (classOf(chosenCells[index]) < classOf(chosenCells[index - 1])) outOfOrder++;
            if (outOfOrder > 0) fail(`${outOfOrder} cells to light come after a cell of a later class`);
            results[`budget ${budget}`] = { usedCells, litCells, shadowRays, samples: COUNTER_SAMPLES, frameIndex, histogram: twin.histogram, litByClass: chosenByClass };
        } finally {
            scenario.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }
    return results;
}

// ---- Section 3: lag -------------------------------------------------------------------------------------

/** Every half's value, for means over whole images without converting them (the lag section reads thousands). */
const HALF_VALUES = (() => {
    const values = new Float32Array(65536);
    for (let bits = 0; bits < 65536; bits++) values[bits] = halfBitsToFloat(bits);
    return values;
})();

/** The mean rgb E over `surface` of an `rgba16float` image as read back (four halves a pixel). */
function halfImageMean(bytes: Uint8Array, surface: Uint32Array): number {
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    let sum = 0;
    for (const pixel of surface) sum += HALF_VALUES[words[pixel * 4]] + HALF_VALUES[words[pixel * 4 + 1]] + HALF_VALUES[words[pixel * 4 + 2]];
    return sum / 3 / Math.max(surface.length, 1);
}

interface Lag {
    /** Frames after the event until the mean is first within 10% of the converged value, and until it stays there; -1 for never. */
    readonly firstWithin: number;
    readonly within: number;
    /** The converged value it is measured against (perHit's), and this run's own mean over its last 32 frames. */
    readonly converged: number;
    readonly ownEnd: number;
}

/** The mean over an event's last 32 frames (its last readings: every 8th frame by then, so 4 of them). */
function endValue(means: readonly number[]): number {
    return average(means.slice(means.length - MEASURED_FRAMES / LAG_COARSE_STEP));
}

/** An event's means and the frames (counted from 1, the first frame after the event) they were read at. */
interface EventMeans {
    readonly frames: number[];
    readonly interpolated: number[];
    readonly denoised: number[];
}

function lagOf(frames: readonly number[], means: readonly number[], converged: number): Lag {
    const near = means.map((mean) => Math.abs(mean - converged) <= LAG_TOLERANCE * converged);
    const firstWithin = near.indexOf(true);
    let within = -1;
    for (let index = near.length - 1; index >= 0 && near[index]; index--) within = index;
    return { firstWithin: firstWithin < 0 ? -1 : frames[firstWithin], within: within < 0 ? -1 : frames[within], converged, ownEnd: endValue(means) };
}

/** perHit's converged values after each event: what every run's lag is measured against. */
interface LagReference {
    readonly lampOn: { readonly interpolated: number; readonly denoised: number };
    readonly cut: { readonly interpolated: number; readonly denoised: number };
}

async function eventMeans(device: GPUDevice, run: Run, surface: Uint32Array): Promise<EventMeans> {
    const means: EventMeans = { frames: [], interpolated: [], denoised: [] };
    const probes = run.technique.probes;
    const denoised = run.technique.denoiser?.output;
    assert(probes && denoised, 'no interpolated or denoised irradiance');
    for (let frame = 1; frame <= EVENT_FRAMES; frame++) {
        await run.frame();
        if (frame > LAG_FINE_FRAMES && frame % LAG_COARSE_STEP !== 0) continue;
        means.frames.push(frame);
        means.interpolated.push(halfImageMean(await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT), surface));
        means.denoised.push(halfImageMean(await readTexture(device, denoised, 8, WIDTH, HEIGHT), surface));
    }
    return means;
}

async function measureLag(device: GPUDevice, label: string, settings: Partial<RadianceCacheSettings>, given?: LagReference): Promise<{ result: Record<string, unknown>; reference: LagReference }> {
    const manifest = loadManifest('two-room');
    const lampIntensity = manifest.lights[0].intensity;
    const run = await startRun(device, 'two-room', settings, { lampIntensity: 0 });
    try {
        for (let frame = 0; frame < LAMP_OFF_FRAMES; frame++) await run.frame();
        const surfaceA = await surfacePixels(device, run);
        for (const light of run.lights) run.platform.scene.updateLight(light, { intensity: lampIntensity });
        const lampOn = await eventMeans(device, run, surfaceA);
        const counters = await readCounters(device, run);

        // A cut: the second bookmark, the screen probes and the denoiser start afresh, the world cache stays.
        run.camera = bookmark(manifest, 1, 100);
        run.technique.resetScreenProbes();
        await run.frame();
        const surfaceB = await surfacePixels(device, run);
        const cut = await eventMeans(device, run, surfaceB);
        const reference = given ?? {
            lampOn: { interpolated: endValue(lampOn.interpolated), denoised: endValue(lampOn.denoised) },
            cut: { interpolated: endValue(cut.interpolated), denoised: endValue(cut.denoised) },
        };
        return {
            reference,
            result: {
                label,
                counters,
                lampOn: { interpolated: lagOf(lampOn.frames, lampOn.interpolated, reference.lampOn.interpolated), denoised: lagOf(lampOn.frames, lampOn.denoised, reference.lampOn.denoised) },
                cut: { interpolated: lagOf(cut.frames, cut.interpolated, reference.cut.interpolated), denoised: lagOf(cut.frames, cut.denoised, reference.cut.denoised) },
                // Every 8th frame's mean, for plotting.
                lampOnInterpolatedMeans: lampOn.interpolated.filter((_, index) => lampOn.frames[index] % 8 === 0),
                cutInterpolatedMeans: cut.interpolated.filter((_, index) => cut.frames[index] % 8 === 0),
            },
        };
    } finally {
        run.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

async function checkLag(device: GPUDevice): Promise<Record<string, unknown>> {
    const results: Record<string, unknown> = {};
    const perHit = await measureLag(device, 'perHit', {});
    results.perHit = perHit.result;
    results.reference = perHit.reference;
    // The cells in use on the lit room, from a budget run above them.
    const ample = await measureLag(device, 'budget, all cells', { cellShading: 'budget', cellShadingBudget: 1_000_000 }, perHit.reference);
    const usedCells = (ample.result.counters as { usedCells: number } | null)?.usedCells ?? 0;
    assert(usedCells > 0, 'two-room: no cells in use on a budget');
    results['budget 1'] = ample.result;
    for (const share of LAG_BUDGET_SHARES.slice(1)) {
        const budget = Math.max(1, Math.round(usedCells * share));
        const label = `budget ${budget} (1/${Math.round(1 / share)} of ${usedCells})`;
        results[`budget 1/${Math.round(1 / share)}`] = (await measureLag(device, label, { cellShading: 'budget', cellShadingBudget: budget }, perHit.reference)).result;
    }
    return results;
}

// ---- The page --------------------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    // The platform's 8 storage buffers per stage, whatever the adapter allows (budget-device.ts).
    const { device, adapterStorageBuffersPerStage } = await requestBudgetDevice(adapter);
    window.addEventListener('pagehide', () => device.destroy(), { once: true });
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    // VITE_RC_CELL_SHADING_SECTIONS=counters,converged,lag (default all) runs some sections only, for hand use.
    const env = import.meta.env as Record<string, unknown>;
    const sections = typeof env.VITE_RC_CELL_SHADING_SECTIONS === 'string' && env.VITE_RC_CELL_SHADING_SECTIONS.trim() !== '' ? env.VITE_RC_CELL_SHADING_SECTIONS.split(',').map((name) => name.trim()) : ['counters', 'converged', 'lag'];
    const counters = sections.includes('counters') ? await checkCounters(device, failures) : 'not run';
    const converged = sections.includes('converged') ? await checkConverged(device, failures) : 'not run';
    const lag = sections.includes('lag') ? await checkLag(device) : 'not run';
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.slice(0, 20).join(' | '),
        data: { width: WIDTH, height: HEIGHT, adapterStorageBuffersPerStage, counters, converged, lag },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}

