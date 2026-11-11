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
 *    traces a shadow ray): budget 5,000 (the cells in use are about 48,000),
 *    3 samples, max age 4, after 12 frames; then the budget above the cells in
 *    use. Lit cells = min(budget, cells in use), shadow rays and samples
 *    written = lit cells × 3, exactly. The cells to light, read back, are
 *    distinct cells in use, and their age classes (from the last-lit frames
 *    read before the frame) are what the CPU twin's choice says: every cell
 *    of the classes before the cutoff, the twin's count of the cutoff class,
 *    none after. After the frame every kept-hit word is 0 again (UpdateTiles
 *    resets them). And the samples reach the cells: one frame from an empty
 *    cache (multibounce off, the budget above the cells in use), every
 *    finest cell with samples counts exactly 3, and there are as many as lit
 *    cells.
 * 2b. **No cell starves** (the corner, budget 5,000, one sample, multibounce
 *    on, max age 8; after 40 frames, 16 frames read back): ranking by age goes
 *    round all the cells in use. The cells in use move: on the corner about
 *    11,000 of 47,000 each frame weren't in use the frame before (the probes
 *    are re-placed every frame), so the cells sharing the budget are those in
 *    use over the last frames, N' of them over the last ceil(N / B) frames (N
 *    the most in use in one frame). After each frame every cell in use was lit
 *    within ceil(N' / B) + 1 frames, and before the lighting the oldest cell
 *    in use the frame before too is at most ceil(N' / B) + 2 frames old. The
 *    first run, which bounded it by ceil(N / B), measured 14 against 10.
 * 3. **Lag** (two-room, Capsaicin's defaults otherwise, max age 8, one
 *    sample; multibounce on, then off): `perHit`, and the budget at 1, 1/4
 *    and 1/16 of the cells in use (counted on the lit room). Its lamp off for
 *    30 frames, then on, for 720 frames; a cut; then the lamp off again for
 *    720 frames. For each event, frames until the mean interpolated and
 *    denoised E are within 10% of the step from the value before the event to
 *    the converged value, first, and for good (read every frame for the first
 *    128, then every 8th, so a lag past 128 frames is known to 8). The lamp-off
 *    event runs until both means are under 1% of their lamp-on value, or 4,096
 *    frames (read every 32nd past 720), and reports the frames to 1% too.
 *    Gates (a light switched off must leave the cells): per hit and with the
 *    budget above the cells in use, under 1% within 720 frames; with the
 *    smaller budgets, the mean keeps falling (at the end at most half of
 *    what it was at frame 720, or under 1%). A small budget fades slowly by
 *    design: a lit cell takes 1/16 of the way to 0 each time it is lit, and
 *    at 1/16 of the cells it is lit only every 30-40 frames (its second
 *    bounce a third as often), so 1% takes thousands of frames. The first
 *    run, which asked for 1% within 720 frames of every budget, read 4% at
 *    1/4 and 37% at 1/16 with the multibounce on. The converged value is
 *    `perHit`'s (the mean of its last 32 frames): every mode converges to the
 *    same image (section 1), but a small budget isn't there yet by the end,
 *    so its own end value would understate its lag (a first run measured
 *    against each run's own end did: 1/16's lamp-on end value was 4.8% below
 *    `perHit`'s, of the same multibounce setting). Each run's own end value
 *    is reported beside it. The cut is to the second bookmark
 *    (`resetScreenProbes`, the world cache warm). Information, but for the
 *    lamp-off gate.
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
import { CELL_SHADING_CLASSES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
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
import { cellValue, readRange, readTiles, runFrames, startScenario, sun, SUN_TRAVEL } from '../radiance-cache-world-cache/corner.ts';

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
/** Section 2b. */
const STARVATION_WARM_FRAMES = 40;
const STARVATION_FRAMES = 16;
/** Section 3's lamp-off event: "gone" is under this share of the lamp-on value; it runs until then, or this many frames, read every LAMP_OFF_COARSE_STEP-th past EVENT_FRAMES. */
const LAMP_OFF_REMAINDER = 0.01;
const LAMP_OFF_MAX_FRAMES = 4096;
const LAMP_OFF_COARSE_STEP = 32;

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
            // The tile slots before and after: a slot claimed this frame (its checksum changed) had its
            // cells' last-lit words cleared by AddNewTile, so the GPU ranked them as new.
            const hashWord = cache.layout.words.hashBuffer;
            const slotsBefore = new Uint32Array(await readRange(device, cache.buffers.words, hashWord * 4, cache.capacity.numTiles * 4));
            await runFrames(device, scenario, 1);
            const slotsAfter = new Uint32Array(await readRange(device, cache.buffers.words, hashWord * 4, cache.capacity.numTiles * 4));
            let claimedTiles = 0;
            for (let tile = 0; tile < cache.capacity.numTiles; tile++) {
                if (slotsAfter[tile] === slotsBefore[tile] || slotsAfter[tile] === 0) continue;
                claimedTiles++;
                lastLitBefore.fill(0, tile * cache.capacity.numCellsPerTile, (tile + 1) * cache.capacity.numCellsPerTile);
            }

            const counters = new Uint32Array(await readRange(device, cache.buffers.words, 0, 64 * 4));
            const usedCells = counters[HASH_GRID_COUNTER.usedCellCount];
            const litCells = counters[HASH_GRID_COUNTER.litCellCount];
            const shadowRays = counters[HASH_GRID_COUNTER.cellShadowRayCount];
            if (litCells !== Math.min(budget, usedCells)) fail(`${litCells} cells lit, min(budget, ${usedCells} in use) is ${Math.min(budget, usedCells)}`);
            if (shadowRays !== litCells * COUNTER_SAMPLES) fail(`${shadowRays} shadow rays for ${litCells} lit cells × ${COUNTER_SAMPLES} samples`);
            const samples = counters[HASH_GRID_COUNTER.cellSampleCount];
            if (samples !== litCells * COUNTER_SAMPLES) fail(`${samples} samples written for ${litCells} lit cells × ${COUNTER_SAMPLES} samples`);
            // UpdateTiles reset every kept-hit word for the next frame's offers.
            const keptWord = cache.layout.words.cellShadingKeptHitBuffer;
            assert(keptWord !== null, 'a budget cache without its kept-hit words');
            const kept = new Uint32Array(await readRange(device, cache.buffers.words, keptWord * 4, cache.capacity.numCells * 4));
            const keptLeft = kept.reduce((count, word) => count + (word !== 0 ? 1 : 0), 0);
            if (keptLeft > 0) fail(`${keptLeft} kept-hit words still set after the frame`);

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
            const classOf = (cell: number): number => cellShadingClass(lastLitBefore[cell], frameIndex);
            const twin = chooseCells(Array.from(used, (visibility) => classOf(cellOf(visibility))), CELL_SHADING_CLASSES, budget);
            const chosenByClass = new Array<number>(CELL_SHADING_CLASSES).fill(0);
            for (const cell of chosenCells) chosenByClass[classOf(cell)]++;
            const expectedByClass = twin.histogram.map((count, ageClass) => (ageClass < twin.cutoffClass ? count : ageClass === twin.cutoffClass ? twin.litInCutoffClass : 0));
            if (chosenByClass.some((count, ageClass) => count !== expectedByClass[ageClass])) fail(`lit cells by age class ${chosenByClass.join(', ')}, the twin's choice ${expectedByClass.join(', ')}`);
            // The list is in rank order: classes never go back down along it.
            let outOfOrder = 0;
            for (let index = 1; index < chosenCells.length; index++) if (classOf(chosenCells[index]) < classOf(chosenCells[index - 1])) outOfOrder++;
            if (outOfOrder > 0) fail(`${outOfOrder} cells to light come after a cell of a later class`);
            // Only the classes with cells, as "class: in use / lit".
            const byClass = Object.fromEntries(twin.histogram.flatMap((count, ageClass) => (count > 0 ? [[ageClass, `${count} / ${chosenByClass[ageClass]}`]] : [])));
            results[`budget ${budget}`] = { usedCells, litCells, shadowRays, claimedTiles, samplesWritten: samples, samplesPerCell: COUNTER_SAMPLES, staleCells: counters[HASH_GRID_COUNTER.staleCellCount], frameIndex, inUseOverLitByClass: byClass };
        } finally {
            scenario.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }
    results.samplesInCells = await checkSamplesInCells(device, failures);
    return results;
}

/** Section 2's last check: one frame from an empty cache, every finest cell with samples counts exactly the samples per lit cell. */
async function checkSamplesInCells(device: GPUDevice, failures: string[]): Promise<Record<string, number>> {
    const fail = (message: string): void => void failures.push(`samples in the cells: ${message}`);
    const scenario = await startScenario(device, 'cell shading samples', [sun(SUN_TRAVEL)], false, {
        useMultibounce: false,
        cellShading: 'budget',
        cellShadingBudget: 1_000_000,
        cellShadingSamples: COUNTER_SAMPLES,
    });
    try {
        // startScenario empties the cache at the next frame: this one.
        await runFrames(device, scenario, 1);
        const counters = new Uint32Array(await readRange(device, scenario.cache.buffers.words, 0, 64 * 4));
        const litCells = counters[HASH_GRID_COUNTER.litCellCount];
        const packed = scenario.cache.packedTiles;
        const packedCount = new Uint32Array(await readRange(device, packed.count.buffer, packed.count.offset, 4))[0];
        const tiles = Array.from(new Uint32Array(await readRange(device, scenario.cache.buffers.words, packed.firstWord * 4, packedCount * 4)));
        const values = await readTiles(device, scenario.cache, tiles);
        const capacity = scenario.cache.capacity;
        let cellsWithSamples = 0;
        let otherCounts = 0;
        tiles.forEach((_, slot) => {
            for (let cell = 0; cell < capacity.numCellsPerTileMip[0]; cell++) {
                const count = cellValue(values, capacity, slot, cell)[3];
                if (count > 0) cellsWithSamples++;
                if (count > 0 && count !== COUNTER_SAMPLES) otherCounts++;
            }
        });
        if (litCells === 0) fail('nothing lit');
        if (cellsWithSamples !== litCells) fail(`${cellsWithSamples} cells have samples, ${litCells} were lit`);
        if (otherCounts > 0) fail(`${otherCounts} cells count other than ${COUNTER_SAMPLES} samples`);
        return { litCells, cellsWithSamples, cellsWithOtherCounts: otherCounts };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- Section 2b: no cell starves ---------------------------------------------------------------------

async function checkStarvation(device: GPUDevice, failures: string[]): Promise<Record<string, unknown>> {
    const fail = (message: string): void => void failures.push(`starvation: ${message}`);
    const scenario = await startScenario(device, 'cell shading starvation', [sun(SUN_TRAVEL)], false, { cellShading: 'budget', cellShadingBudget: SMALL_BUDGET, cellShadingSamples: 1 });
    try {
        await runFrames(device, scenario, STARVATION_WARM_FRAMES);
        const cache = scenario.cache;
        const lastLitWord = cache.layout.words.cellShadingLastLitBuffer;
        assert(lastLitWord !== null, 'a budget cache without its last-lit words');
        let previousUsed = new Set<number>();
        let previousLastLit: Uint32Array | null = null;
        const usedByFrame: number[][] = [];
        const frames: { usedCells: number; oldestAfter: number; oldestBefore: number; returning: number; staleCells: number }[] = [];
        let maxUsed = 0;
        for (let frame = 0; frame < STARVATION_FRAMES; frame++) {
            await runFrames(device, scenario, 1);
            const counters = new Uint32Array(await readRange(device, cache.buffers.words, 0, 64 * 4));
            const usedCells = counters[HASH_GRID_COUNTER.usedCellCount];
            maxUsed = Math.max(maxUsed, usedCells);
            const lists = cache.rayLists;
            assert(lists, 'no per-ray lists after a frame');
            const layout = lists.layout;
            const words = new Uint32Array(await readRange(device, lists.buffer, 0, lists.buffer.size));
            const usedCellList = Array.from(words.slice(layout.usedCellBuffer, layout.usedCellBuffer + usedCells), (visibility) => words[layout.visibilityCellBuffer + visibility]);
            const lastLit = new Uint32Array(await readRange(device, cache.buffers.words, lastLitWord * 4, cache.capacity.numCells * 4));
            // This frame's index: the newest last-lit frame among the cells in use, minus one.
            const frameIndex = usedCellList.reduce((newest, cell) => Math.max(newest, lastLit[cell]), 0) - 1;
            let oldestAfter = 0;
            let oldestBefore = 0;
            let returning = 0;
            for (const cell of usedCellList) {
                // Frames since lit after this frame's lighting (0: lit this frame), never-lit counted as the frame count.
                oldestAfter = Math.max(oldestAfter, lastLit[cell] === 0 ? frameIndex + 1 : frameIndex + 1 - lastLit[cell]);
                if (!previousUsed.has(cell)) {
                    returning++;
                    continue;
                }
                if (previousLastLit) oldestBefore = Math.max(oldestBefore, previousLastLit[cell] === 0 ? frameIndex + 1 : frameIndex + 1 - previousLastLit[cell]);
            }
            frames.push({ usedCells, oldestAfter, oldestBefore, returning, staleCells: counters[HASH_GRID_COUNTER.staleCellCount] });
            usedByFrame.push(usedCellList);
            previousUsed = new Set(usedCellList);
            previousLastLit = lastLit;
        }
        const rounds = Math.ceil(maxUsed / SMALL_BUDGET);
        // The cells sharing the budget: those in use over the last `rounds` frames (the most over any such window).
        let sharing = 0;
        for (let end = rounds; end <= usedByFrame.length; end++) sharing = Math.max(sharing, new Set(usedByFrame.slice(end - rounds, end).flat()).size);
        const sharedRounds = Math.ceil(sharing / SMALL_BUDGET);
        const worstAfter = Math.max(...frames.map((entry) => entry.oldestAfter));
        const worstBefore = Math.max(...frames.slice(1).map((entry) => entry.oldestBefore));
        if (worstAfter > sharedRounds + 1) fail(`a cell in use was last lit ${worstAfter} frames ago; with ${sharing} cells in use over ${rounds} frames and a budget of ${SMALL_BUDGET} every one is lit within ${sharedRounds} (+1)`);
        if (worstBefore > sharedRounds + 2) fail(`before the lighting, a cell in use since the frame before was ${worstBefore} frames old (ceil(N' / B) = ${sharedRounds}, + 2)`);
        // No starvation: the ages don't grow over the frames read.
        const firstHalf = Math.max(...frames.slice(0, frames.length / 2).map((entry) => entry.oldestAfter));
        const secondHalf = Math.max(...frames.slice(frames.length / 2).map((entry) => entry.oldestAfter));
        if (secondHalf > firstHalf + 1) fail(`the oldest age grows: ${firstHalf} in the first ${frames.length / 2} frames read, ${secondHalf} in the last`);
        return {
            budget: SMALL_BUDGET,
            maxUsedCells: maxUsed,
            ceilUsedOverBudget: rounds,
            cellsInUseOverThoseFrames: sharing,
            ceilSharingOverBudget: sharedRounds,
            worstFramesSinceLitAfter: worstAfter,
            worstAgeBefore: worstBefore,
            frames,
        };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
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

/** Frames until the means are within 10% of the step from `before` to `converged` (perHit's values). */
function lagOf(frames: readonly number[], means: readonly number[], converged: number, before: number): Lag {
    const tolerance = LAG_TOLERANCE * Math.abs(converged - before);
    const near = means.map((mean) => Math.abs(mean - converged) <= tolerance);
    const firstWithin = near.indexOf(true);
    let within = -1;
    for (let index = near.length - 1; index >= 0 && near[index]; index--) within = index;
    return { firstWithin: firstWithin < 0 ? -1 : frames[firstWithin], within: within < 0 ? -1 : frames[within], converged, ownEnd: endValue(means) };
}

interface EventValues {
    readonly interpolated: number;
    readonly denoised: number;
}

/** perHit's converged values after each event (of the same multibounce setting): what every run's lag is measured against. */
interface LagReference {
    readonly lampOn: EventValues;
    readonly cut: EventValues;
    readonly lampOff: EventValues;
}

const ends = (means: EventMeans): EventValues => ({ interpolated: endValue(means.interpolated), denoised: endValue(means.denoised) });

/**
 * The means after an event, EVENT_FRAMES of them; with `goneBelow`, on past
 * that until both are under it (or LAMP_OFF_MAX_FRAMES), every
 * LAMP_OFF_COARSE_STEP-th frame read.
 */
async function eventMeans(device: GPUDevice, run: Run, surface: Uint32Array, goneBelow?: EventValues): Promise<EventMeans> {
    const means: EventMeans = { frames: [], interpolated: [], denoised: [] };
    const probes = run.technique.probes;
    const denoised = run.technique.denoiser?.output;
    assert(probes && denoised, 'no interpolated or denoised irradiance');
    const lastFrame = goneBelow ? LAMP_OFF_MAX_FRAMES : EVENT_FRAMES;
    for (let frame = 1; frame <= lastFrame; frame++) {
        await run.frame();
        const step = frame <= LAG_FINE_FRAMES ? 1 : frame <= EVENT_FRAMES ? LAG_COARSE_STEP : LAMP_OFF_COARSE_STEP;
        if (frame % step !== 0) continue;
        means.frames.push(frame);
        const interpolated = halfImageMean(await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT), surface);
        const denoisedMean = halfImageMean(await readTexture(device, denoised, 8, WIDTH, HEIGHT), surface);
        means.interpolated.push(interpolated);
        means.denoised.push(denoisedMean);
        if (goneBelow && frame >= EVENT_FRAMES && interpolated <= goneBelow.interpolated && denoisedMean <= goneBelow.denoised) break;
    }
    return means;
}

/** The first frame (counted from 1) whose mean is at most `level`, and the mean at a given frame (the last reading at or before it). */
function firstFrameAtMost(means: EventMeans, image: 'interpolated' | 'denoised', level: number): number {
    const index = means[image].findIndex((mean) => mean <= level);
    return index < 0 ? -1 : means.frames[index];
}

function meanAt(means: EventMeans, image: 'interpolated' | 'denoised', frame: number): number {
    let value = means[image][0];
    means.frames.forEach((at, index) => {
        if (at <= frame) value = means[image][index];
    });
    return value;
}

async function measureLag(
    device: GPUDevice,
    label: string,
    settings: Partial<RadianceCacheSettings>,
    failures: string[],
    given?: LagReference,
): Promise<{ result: Record<string, unknown>; reference: LagReference; usedCells: number }> {
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

        // The lamp off again, at the second bookmark: its light must leave the cells.
        const cutEnd = ends(cut);
        const gone: EventValues = { interpolated: LAMP_OFF_REMAINDER * cutEnd.interpolated, denoised: LAMP_OFF_REMAINDER * cutEnd.denoised };
        for (const light of run.lights) run.platform.scene.updateLight(light, { intensity: 0 });
        const lampOff = await eventMeans(device, run, surfaceB, gone);
        const reference = given ?? { lampOn: ends(lampOn), cut: cutEnd, lampOff: ends(lampOff) };
        const lampOffEnd: EventValues = { interpolated: lampOff.interpolated[lampOff.interpolated.length - 1], denoised: lampOff.denoised[lampOff.denoised.length - 1] };
        const ample = settings.cellShading !== 'budget' || (settings.cellShadingBudget ?? 0) >= 1_000_000;
        const framesToGone = { interpolated: firstFrameAtMost(lampOff, 'interpolated', gone.interpolated), denoised: firstFrameAtMost(lampOff, 'denoised', gone.denoised) };
        for (const image of ['interpolated', 'denoised'] as const) {
            const atEventEnd = meanAt(lampOff, image, EVENT_FRAMES);
            const percent = (value: number): string => `${((100 * value) / cutEnd[image]).toFixed(2)}%`;
            if (ample && !(framesToGone[image] > 0 && framesToGone[image] <= EVENT_FRAMES)) {
                failures.push(`lag ${label}: with the lamp off again the mean ${image} E is ${percent(atEventEnd)} of its lamp-on value at frame ${EVENT_FRAMES} (under 1% by then asked)`);
            }
            if (!ample && !(lampOffEnd[image] <= gone[image] || lampOffEnd[image] <= 0.5 * atEventEnd)) {
                failures.push(`lag ${label}: with the lamp off again the mean ${image} E went from ${percent(atEventEnd)} at frame ${EVENT_FRAMES} to ${percent(lampOffEnd[image])} at ${lampOff.frames[lampOff.frames.length - 1]}: it doesn't keep fading`);
            }
        }
        const lags = (means: EventMeans, converged: EventValues, before: EventValues): Record<string, Lag> => ({
            interpolated: lagOf(means.frames, means.interpolated, converged.interpolated, before.interpolated),
            denoised: lagOf(means.frames, means.denoised, converged.denoised, before.denoised),
        });
        const none: EventValues = { interpolated: 0, denoised: 0 };
        return {
            reference,
            usedCells: counters?.usedCells ?? 0,
            result: {
                label,
                counters,
                // From the dark room (0), from the first view's light (the step to the second's: its own value), and from the lamp-on value.
                lampOn: lags(lampOn, reference.lampOn, none),
                cut: lags(cut, reference.cut, none),
                lampOff: lags(lampOff, reference.lampOff, reference.cut),
                lampOffFramesToOnePercent: framesToGone,
                lampOffRemainderAt720: { interpolated: meanAt(lampOff, 'interpolated', EVENT_FRAMES) / cutEnd.interpolated, denoised: meanAt(lampOff, 'denoised', EVENT_FRAMES) / cutEnd.denoised },
                lampOffRemainderAtEnd: { frame: lampOff.frames[lampOff.frames.length - 1], interpolated: lampOffEnd.interpolated / cutEnd.interpolated, denoised: lampOffEnd.denoised / cutEnd.denoised },
                // Every 8th frame's mean, for plotting.
                lampOnInterpolatedMeans: lampOn.interpolated.filter((_, index) => lampOn.frames[index] % 8 === 0),
                lampOffInterpolatedMeans: lampOff.interpolated.filter((_, index) => lampOff.frames[index] % 8 === 0),
            },
        };
    } finally {
        run.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

async function checkLag(device: GPUDevice, failures: string[]): Promise<Record<string, unknown>> {
    const results: Record<string, unknown> = {};
    for (const useMultibounce of [true, false]) {
        const setting = `multibounce ${useMultibounce ? 'on' : 'off'}`;
        const perSetting: Record<string, unknown> = {};
        const perHit = await measureLag(device, `perHit, ${setting}`, { useMultibounce }, failures);
        perSetting.perHit = perHit.result;
        perSetting.reference = perHit.reference;
        // The cells in use on the lit room, from a budget run above them.
        const ample = await measureLag(device, `budget, all cells, ${setting}`, { useMultibounce, cellShading: 'budget', cellShadingBudget: 1_000_000 }, failures, perHit.reference);
        assert(ample.usedCells > 0, 'two-room: no cells in use on a budget');
        perSetting['budget 1'] = ample.result;
        for (const share of LAG_BUDGET_SHARES.slice(1)) {
            const budget = Math.max(1, Math.round(ample.usedCells * share));
            const label = `budget ${budget} (1/${Math.round(1 / share)} of ${ample.usedCells}), ${setting}`;
            perSetting[`budget 1/${Math.round(1 / share)}`] = (await measureLag(device, label, { useMultibounce, cellShading: 'budget', cellShadingBudget: budget }, failures, perHit.reference)).result;
        }
        results[setting] = perSetting;
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
    const sections =
        typeof env.VITE_RC_CELL_SHADING_SECTIONS === 'string' && env.VITE_RC_CELL_SHADING_SECTIONS.trim() !== ''
            ? env.VITE_RC_CELL_SHADING_SECTIONS.split(',').map((name) => name.trim())
            : ['counters', 'starvation', 'converged', 'lag'];
    const counters = sections.includes('counters') ? await checkCounters(device, failures) : 'not run';
    const starvation = sections.includes('starvation') ? await checkStarvation(device, failures) : 'not run';
    const converged = sections.includes('converged') ? await checkConverged(device, failures) : 'not run';
    const lag = sections.includes('lag') ? await checkLag(device, failures) : 'not run';
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.slice(0, 20).join(' | '),
        data: { width: WIDTH, height: HEIGHT, adapterStorageBuffersPerStage, counters, starvation, converged, lag },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}

