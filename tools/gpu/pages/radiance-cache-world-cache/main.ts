/**
 * RC3.2: the radiance cache's world cache (Capsaicin's hash grid) populated
 * and resolved through the technique, on the GPU.
 *
 * The scene is a corner: a 400 m floor (y = 0, facing up) and a 40 × 10 m
 * wall (z = 0, facing +z), both grey (albedo 0.5), under a black sky. The
 * camera at (0, 2, 6) looks at the wall's foot, so probes sit on the floor
 * and the wall: the floor's probe rays hit the wall, the wall's hit the floor,
 * and those near the corner are shorter than their cell. The screen comes
 * from primary rays through the platform's tracer (radiance-cache-probes'
 * `PageScreen`). The scene and the readbacks are in `corner.ts`.
 *
 * Every section runs Capsaicin's single-bounce configuration
 * (`useMultibounce` off, RC3.3): they check the first bounce alone. The
 * second bounce is checked on `radiance-cache-multibounce`.
 *
 * 1. **A lit plane converges** (sun along (0, -1, -1), intensity 8, after
 *    40 frames from an empty cache): floor and wall both get E = 8 cos 45°,
 *    so the light leaving any hit toward any probe is albedo / π × E. Every
 *    probe ray that hit reads it within 2%, and so does every finest-level
 *    cell with samples (rgb / count). The GPU's bypass flags match the
 *    CPU's "ray shorter than its cell" (rays within 0.1% of a cell's size
 *    are counted, not checked), some rays take each path, and every hit was
 *    inserted, resolved and given a shadow ray (no bucket overflowed); the
 *    update list holds every touched tile once (register D32).
 * 2. **UpdateTiles** (same frame): no finest-level count above 16 and some
 *    at 16; every coarser cell of every live tile is the sum of its 2 × 2
 *    finer cells (rgb and count, within half precision).
 * 3. **ResolveCells** (a point light at (0, 1.5, 2), one frame on an empty
 *    cache, so most cells have fewer than 16 samples): every resolved ray
 *    reads what Capsaicin's filtered readback picks from the cells as read
 *    back (the CPU's `hashGridCacheFilteredRadiance`); some rays read a
 *    coarser level, and some of those differ from their own cell's mean by
 *    more than 5% (so the check can tell the levels apart).
 * 4. **Emissive hits skip the cache** (the sun and an emissive patch on the
 *    floor, emission (3, 2, 1)): every ray that hit the patch's front reads
 *    its emission alone (a back-face hit is lit as any surface, as Capsaicin's); each was inserted (it is on the visibility list) and
 *    none was resolved or given a shadow ray. (Short rays are section 1.)
 * 5. **The light's weight** (two suns, red along (0, -1, -1) and blue along
 *    (0, -1, -0.5), intensity 8 each, 40 frames): one light is drawn per hit
 *    in proportion to its brightness, so each sample lights one channel; the
 *    mean over every hit ray of red and of blue is within 2% of albedo / π ×
 *    the light's E on that surface, and green stays 0.
 * 6. **Sky light at cache hits** (RC6.1: no lights, a constant sky of
 *    radiance L = (1, 0.8, 0.6), 60 frames from an empty cache): with
 *    Capsaicin's environment light every hit draws the sky (all but the
 *    rare direction whose packed numbers land on the horizon, weight 0), and
 *    a cell's radiance is albedo / π × E with E = π L V, V the share of the
 *    sky its hits see (`sky-view.ts`: the wall hides part of the floor's sky,
 *    the floor part of the wall's). The sum over cells of the cells' rgb /
 *    count against the sum of albedo × L × V (V the mean over the cell's hits
 *    that frame), over four readings 16 frames apart (one history each; the
 *    GPU's append order reseeds the random numbers, so one reading's open
 *    floor moved by 1.3% between runs), is within the Lambert check's 2.5%
 *    (DECISIONS 2026-11-10) for the open floor (V ≥ 0.99, the done-when's
 *    E = πL), the whole floor and the wall, in every channel.
 * 6b. **The sky against a light** (the same sky and section 1's sun, 60
 *    frames): the draw picks between the sky and the sun by their weights,
 *    so a cell's mean is albedo × L × V + albedo / π × the sun's E (the sun
 *    reaches every floor and wall hit in front of the wall); the same sums
 *    within 2.5% for the three groups, every channel, and both lights are
 *    drawn.
 * 7. **A closed box with no lights** (the same sky outside, 20 frames; one
 *    mesh, large enough that the camera sees one wall and no edge, so every
 *    probe starts inside: on an edge pixel an origin rebuilt from depth can
 *    land a hair outside, register D17): no probe ray starts outside or
 *    misses, every hit draws the sky as above, and its shadow ray is
 *    blocked, so every cell with samples and every hit ray reads 0.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc3.2 world cache" -- pnpm test:gpu --filter radiance-cache-world-cache --gpu nvidia` (or intel).
 */

import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import {
    hashGridCacheCellIndex,
    hashGridCacheCellOffsetMip0,
    hashGridCacheCellSizeConstant,
    hashGridCacheFilteredRadiance,
    hashGridCacheGetCellSize,
    verticalFieldOfView,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import { BYPASS_CACHE_BIT } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { RESERVOIR_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import {
    ALBEDO,
    BLUE_SUN_TRAVEL,
    CAMERA,
    cellValue,
    CLOSED_BOX,
    CONVERGENCE_FRAMES,
    EMISSION,
    FRAME_CAMERA,
    HALF_TOLERANCE,
    HEIGHT,
    LEVEL_SIDES,
    LIT_RADIANCE,
    MAX_SAMPLE_COUNT,
    readFrameState,
    readTiles,
    relativeError,
    runFrames,
    startScenario,
    sun,
    SUN_INTENSITY,
    SUN_TRAVEL,
    TOLERANCE,
    WIDTH,
    type Scenario,
    type ScenarioWorld,
} from './corner.ts';
import { cornerSkyVisibility } from './sky-view.ts';

// ---- Section 1 and 2: the lit plane and UpdateTiles ----------------------------------------------

interface Checks {
    readonly failures: string[];
    readonly data: Record<string, unknown>;
}

function cellSizeConstants(): { cellSize: number; minCellSize: number } {
    return { cellSize: hashGridCacheCellSizeConstant(verticalFieldOfView(FRAME_CAMERA.projection), WIDTH, HEIGHT, 32), minCellSize: 0.1 };
}

async function checkLitPlane(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`lit plane: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const hits = state.rays.filter((ray) => ray.hitT >= 0);
    const counters = state.counters;

    // Every hit inserted, resolved and lit (the sun reaches both surfaces), nothing refused.
    const overflow = counters[HASH_GRID_COUNTER.bucketOverflowCount];
    if (overflow !== 0) fail(`${overflow} inserts refused (bucket overflow)`);
    if (counters[HASH_GRID_COUNTER.visibilityCount0] !== hits.length) fail(`visibility list holds ${counters[HASH_GRID_COUNTER.visibilityCount0]} hits, the rays ${hits.length}`);
    if (state.resolve.length !== hits.length) fail(`resolve list holds ${state.resolve.length}, hits ${hits.length}`);
    if (state.visibilityRay.length !== hits.length) fail(`${state.visibilityRay.length} shadow rays for ${hits.length} hits`);
    // The update list: each tile touched this frame, once (register D32).
    const touchedTiles = new Set(Array.from(state.visibilityCell, (cell) => Math.floor(cell / scenario.cache.capacity.numCellsPerTile)));
    const listedTiles = new Set(state.updateTiles);
    if (listedTiles.size !== state.updateTiles.length) fail(`the update list holds ${state.updateTiles.length} entries for ${listedTiles.size} tiles`);
    if (listedTiles.size !== touchedTiles.size || [...touchedTiles].some((tile) => !listedTiles.has(tile))) fail(`the update list holds ${listedTiles.size} tiles, the hits touched ${touchedTiles.size}`);
    const others = hits.filter((ray) => ray.surface !== 'floor' && ray.surface !== 'wall').length;
    if (others > 0) fail(`${others} hits on neither the floor nor the wall`);

    // Each hit ray reads albedo / π × E.
    let worstRay = 0;
    let raysOff = 0;
    state.rays.forEach((ray, index) => {
        if (ray.hitT < 0) return;
        const error = Math.max(...[0, 1, 2].map((channel) => relativeError(state.radiance[index * 4 + channel], LIT_RADIANCE)));
        worstRay = Math.max(worstRay, error);
        if (error > TOLERANCE) {
            raysOff++;
            if (raysOff <= 3) fail(`ray ${index} (${ray.surface}, ${ray.hitT.toFixed(3)} m) reads ${[0, 1, 2].map((channel) => state.radiance[index * 4 + channel].toFixed(4)).join(', ')}, expected ${LIT_RADIANCE.toFixed(4)}`);
        }
    });
    if (raysOff > 0) fail(`${raysOff} of ${hits.length} hit rays are more than 2% off`);

    // The bypass: the GPU's flag on each resolve entry against the CPU's "shorter than its cell".
    const constants = cellSizeConstants();
    let bypassed = 0;
    let resolved = 0;
    let undecided = 0;
    let flagMismatches = 0;
    for (const entry of state.resolve) {
        const visibility = entry & ~BYPASS_CACHE_BIT;
        const ray = state.rays[state.visibilityQuery[visibility]];
        const gpuBypass = (entry & BYPASS_CACHE_BIT) !== 0;
        const cellSize = hashGridCacheGetCellSize(ray.hit, CAMERA.position, constants);
        const rayLength = Math.hypot(ray.hit[0] - ray.origin[0], ray.hit[1] - ray.origin[1], ray.hit[2] - ray.origin[2]);
        if (gpuBypass) bypassed++;
        else resolved++;
        if (Math.abs(rayLength - cellSize) < 1e-3 * cellSize) {
            undecided++;
            continue;
        }
        if (gpuBypass !== rayLength < cellSize) {
            flagMismatches++;
            if (flagMismatches <= 3) fail(`ray ${state.visibilityQuery[visibility]}: ${rayLength.toFixed(4)} m long in a ${cellSize} m cell, GPU bypass ${gpuBypass}`);
        }
    }
    if (bypassed === 0 || resolved === 0) fail(`the rays should take both paths: ${bypassed} bypassed, ${resolved} resolved`);
    // A shadow ray's bypass flag is its resolve entry's.
    const resolveFlag = new Map<number, boolean>();
    for (const entry of state.resolve) resolveFlag.set(entry & ~BYPASS_CACHE_BIT, (entry & BYPASS_CACHE_BIT) !== 0);
    const shadowFlagMismatches = Array.from(state.visibilityRay).filter((entry) => resolveFlag.get(entry & ~BYPASS_CACHE_BIT) !== ((entry & BYPASS_CACHE_BIT) !== 0)).length;
    if (shadowFlagMismatches > 0) fail(`${shadowFlagMismatches} shadow rays carry another bypass flag than their resolve entry`);

    // Section 2 and the cells: every live tile, all four levels.
    const capacity = scenario.cache.capacity;
    const tiles = Array.from(state.packedTiles);
    const values = await readTiles(device, scenario.cache, tiles);
    let cellsWithSamples = 0;
    let worstCell = 0;
    let cellsOff = 0;
    let largestCount = 0;
    let countsAtCap = 0;
    let levelChecks = 0;
    let levelMismatches = 0;
    let worstLevelError = 0;
    tiles.forEach((tile, slot) => {
        for (let cell = 0; cell < capacity.numCellsPerTileMip[0]; cell++) {
            const value = cellValue(values, capacity, slot, cell);
            largestCount = Math.max(largestCount, value[3]);
            if (value[3] === MAX_SAMPLE_COUNT) countsAtCap++;
            if (!(value[3] > 0)) continue;
            cellsWithSamples++;
            const error = Math.max(...[0, 1, 2].map((channel) => relativeError(value[channel] / value[3], LIT_RADIANCE)));
            worstCell = Math.max(worstCell, error);
            if (error > TOLERANCE) cellsOff++;
        }
        // Each coarser cell against the sum of its 2 x 2 finer cells.
        for (let level = 1; level < LEVEL_SIDES.length; level++) {
            const side = LEVEL_SIDES[level];
            for (let y = 0; y < side; y++) {
                for (let x = 0; x < side; x++) {
                    const offset = (cellX: number, cellY: number, cellLevel: number): number =>
                        hashGridCacheCellIndex([cellX << cellLevel, cellY << cellLevel], tile, cellLevel, capacity) - tile * capacity.numCellsPerTile;
                    const coarse = cellValue(values, capacity, slot, offset(x, y, level));
                    const sum = [0, 0, 0, 0];
                    for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
                        const fine = cellValue(values, capacity, slot, offset(2 * x + dx, 2 * y + dy, level - 1));
                        for (let channel = 0; channel < 4; channel++) sum[channel] += fine[channel];
                    }
                    levelChecks++;
                    const error = Math.max(...[0, 1, 2, 3].map((channel) => Math.abs(coarse[channel] - sum[channel]) / Math.max(Math.abs(sum[channel]), 1e-3)));
                    worstLevelError = Math.max(worstLevelError, error);
                    if (error > HALF_TOLERANCE) {
                        levelMismatches++;
                        if (levelMismatches <= 3) fail(`tile ${tile} level ${level} cell (${x}, ${y}) holds ${coarse.map((v) => v.toFixed(4)).join(', ')}, its 2x2 sum ${sum.map((v) => v.toFixed(4)).join(', ')}`);
                    }
                }
            }
        }
    });
    if (cellsOff > 0) fail(`${cellsOff} of ${cellsWithSamples} cells are more than 2% off albedo / pi x E`);
    if (largestCount > MAX_SAMPLE_COUNT) fail(`a finest-level cell counts ${largestCount} samples, above the cap of ${MAX_SAMPLE_COUNT}`);
    if (countsAtCap === 0) fail(`no cell reached the cap of ${MAX_SAMPLE_COUNT} samples after ${CONVERGENCE_FRAMES} frames`);
    if (levelMismatches > 0) fail(`${levelMismatches} of ${levelChecks} coarser cells aren't the sum of their 2x2 cells`);

    return {
        failures,
        data: {
            expectedRadiance: LIT_RADIANCE,
            rays: state.rays.length,
            hits: hits.length,
            worstRayError: worstRay,
            bypassed,
            resolved,
            undecidedBypass: undecided,
            bypassFlagMismatches: flagMismatches,
            bucketOverflow: overflow,
            updateListEntries: state.updateTiles.length,
            updateListTiles: listedTiles.size,
            liveTiles: tiles.length,
            cellsWithSamples,
            worstCellError: worstCell,
            largestCount,
            cellsAtCap: countsAtCap,
            coarserCellsChecked: levelChecks,
            coarserCellMismatches: levelMismatches,
            worstCoarserCellError: worstLevelError,
        },
    };
}

// ---- Section 3: ResolveCells ----------------------------------------------------------------------

async function checkResolve(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`resolve: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const capacity = scenario.cache.capacity;
    const entries = Array.from(state.resolve).filter((entry) => (entry & BYPASS_CACHE_BIT) === 0);
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);
    const tiles = [...new Set(entries.map((entry) => tileOf(state.visibilityCell[entry])))];
    const slotOfTile = new Map(tiles.map((tile, slot) => [tile, slot]));
    const values = await readTiles(device, scenario.cache, tiles);

    const levelCounts = [0, 0, 0, 0];
    let mismatches = 0;
    let worst = 0;
    let coarserAndDifferent = 0;
    for (const entry of entries) {
        const cell = state.visibilityCell[entry];
        const query = state.visibilityQuery[entry];
        const { tileIndex, cellOffset } = hashGridCacheCellOffsetMip0(cell, capacity);
        const slot = slotOfTile.get(tileIndex) ?? 0;
        const levels = [0, 1, 2, 3].map((level) =>
            capacity.sizeTileMip[level] > 0 ? cellValue(values, capacity, slot, hashGridCacheCellIndex(cellOffset, tileIndex, level, capacity) - tileIndex * capacity.numCellsPerTile) : null,
        );
        const picked = hashGridCacheFilteredRadiance(levels, MAX_SAMPLE_COUNT);
        levelCounts[picked.level]++;
        const expected = [0, 1, 2].map((channel) => picked.value[channel] / Math.max(picked.value[3], 1));
        const got = [0, 1, 2].map((channel) => state.radiance[query * 4 + channel]);
        const error = Math.max(...[0, 1, 2].map((channel) => Math.abs(got[channel] - expected[channel]) / Math.max(Math.abs(expected[channel]), 1e-4)));
        worst = Math.max(worst, error);
        if (error > 2 * HALF_TOLERANCE) {
            mismatches++;
            if (mismatches <= 3) fail(`ray ${query} reads ${got.map((v) => v.toFixed(5)).join(', ')}, the level-${picked.level} pick gives ${expected.map((v) => v.toFixed(5)).join(', ')}`);
        }
        const own = levels[0];
        if (picked.level > 0 && own && own[3] > 0) {
            const ownMean = own[0] / own[3];
            if (Math.abs(ownMean - expected[0]) > 0.05 * Math.max(ownMean, expected[0])) coarserAndDifferent++;
        }
    }
    const coarser = levelCounts[1] + levelCounts[2] + levelCounts[3];
    if (entries.length === 0) fail('no resolved rays');
    if (mismatches > 0) fail(`${mismatches} of ${entries.length} resolved rays don't read Capsaicin's pick`);
    if (coarser === 0) fail('no ray read a coarser level: the check can\'t see the pick');
    if (coarserAndDifferent === 0) fail('no coarser pick differs from its own cell by 5%: the check can\'t tell the levels apart');
    return { failures, data: { resolvedRays: entries.length, tiles: tiles.length, picksByLevel: levelCounts, coarserPicksThatDiffer: coarserAndDifferent, worstError: worst } };
}

// ---- Section 4: emissive hits --------------------------------------------------------------------

async function checkEmissive(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`emissive: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    // Capsaicin bypasses the cache on an emissive front face only (gi1.comp:2444): the floor's
    // probes beside the patch can catch its edge from below, and those hits are lit as any other.
    const patchRays = new Set<number>();
    let patchBackFaces = 0;
    state.rays.forEach((ray, index) => {
        if (ray.hitT < 0 || ray.surface !== 'patch') return;
        if (ray.frontFace) patchRays.add(index);
        else patchBackFaces++;
    });
    let worst = 0;
    for (const index of patchRays) {
        const error = Math.max(...[0, 1, 2].map((channel) => relativeError(state.radiance[index * 4 + channel], EMISSION[channel])));
        worst = Math.max(worst, error);
    }
    if (patchRays.size === 0) fail('no probe ray hit the emissive patch');
    if (worst > 2 * HALF_TOLERANCE) fail(`a ray on the patch reads ${(worst * 100).toFixed(3)}% off its emission alone`);
    // Inserted (on the visibility list), but neither resolved nor given a shadow ray.
    const visibilityOfPatch = new Set<number>();
    state.visibilityQuery.forEach((query, visibility) => {
        if (patchRays.has(query)) visibilityOfPatch.add(visibility);
    });
    if (visibilityOfPatch.size !== patchRays.size) fail(`${visibilityOfPatch.size} of ${patchRays.size} patch hits are on the visibility list`);
    const resolvedPatch = Array.from(state.resolve).filter((entry) => visibilityOfPatch.has(entry & ~BYPASS_CACHE_BIT)).length;
    const shadowPatch = Array.from(state.visibilityRay).filter((entry) => visibilityOfPatch.has(entry & ~BYPASS_CACHE_BIT)).length;
    if (resolvedPatch > 0) fail(`${resolvedPatch} patch hits were resolved from the cache`);
    if (shadowPatch > 0) fail(`${shadowPatch} patch hits got a shadow ray`);
    return { failures, data: { patchRays: patchRays.size, patchBackFaces, worstEmissionError: worst, inserted: visibilityOfPatch.size, resolved: resolvedPatch, shadowRays: shadowPatch } };
}

// ---- Section 5: two coloured suns -----------------------------------------------------------------

function cosineTo(travel: Vec3, normal: Vec3): number {
    const length = Math.hypot(...travel);
    return Math.max(0, -(travel[0] * normal[0] + travel[1] * normal[1] + travel[2] * normal[2]) / length);
}

async function checkTwoSuns(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`two suns: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const sums = { red: 0, redExpected: 0, green: 0, blue: 0, blueExpected: 0 };
    let hits = 0;
    state.rays.forEach((ray, index) => {
        if (ray.hitT < 0 || (ray.surface !== 'floor' && ray.surface !== 'wall')) return;
        hits++;
        const normal: Vec3 = ray.surface === 'floor' ? [0, 1, 0] : [0, 0, 1];
        sums.red += state.radiance[index * 4];
        sums.green += state.radiance[index * 4 + 1];
        sums.blue += state.radiance[index * 4 + 2];
        sums.redExpected += (ALBEDO / Math.PI) * SUN_INTENSITY * cosineTo(SUN_TRAVEL, normal);
        sums.blueExpected += (ALBEDO / Math.PI) * SUN_INTENSITY * cosineTo(BLUE_SUN_TRAVEL, normal);
    });
    const redRatio = sums.red / sums.redExpected;
    const blueRatio = sums.blue / sums.blueExpected;
    const greenMean = sums.green / Math.max(hits, 1);
    if (hits === 0) fail('no hits');
    if (Math.abs(redRatio - 1) > TOLERANCE) fail(`the mean red is ${redRatio.toFixed(4)} x albedo / pi x E`);
    if (Math.abs(blueRatio - 1) > TOLERANCE) fail(`the mean blue is ${blueRatio.toFixed(4)} x albedo / pi x E`);
    if (greenMean > 1e-3) fail(`green reads ${greenMean} on average; neither sun has any`);
    return { failures, data: { hits, meanRedOverExpected: redRatio, meanBlueOverExpected: blueRatio, meanGreen: greenMean } };
}

// ---- Sections 6 and 7: the sky as a light (RC6.1) ------------------------------------------------

/** Section 6's constant sky radiance, a colour so each channel is checked on its own. */
const SKY_RADIANCE: Vec3 = [1, 0.8, 0.6];
/** The Lambert check's threshold (DECISIONS 2026-11-10, RC2.2). */
const LAMBERT_TOLERANCE = 0.025;
/** Floor cells whose hits see at least this share of the sky are the "open floor". */
const OPEN_FLOOR_VISIBILITY = 0.99;
/** `GI_DIRECT_SKY_LIGHT` (platform/shade/wgsl.ts): the reservoir's light word when it drew the sky. */
const SKY_LIGHT = 0xfffffffe;
/**
 * Hits allowed to draw nothing: the sky's direction is two numbers kept at 16 bits, and one in
 * about 131,000 rounds to 1, a direction on the horizon whose weight is 0.
 */
const MAX_HITS_WITHOUT_LIGHT_SHARE = 1e-4;

/** How the frame's hits drew their lights: how many drew the sky, and how many drew nothing. */
function skyDraws(state: Awaited<ReturnType<typeof readFrameState>>): { hits: number; skyDraws: number; otherDraws: number; withoutLight: number } {
    const hits = state.counters[HASH_GRID_COUNTER.visibilityCount0];
    let sky = 0;
    for (let ray = 0; ray < state.visibilityRay.length; ray++) {
        if (state.reservoirs[ray * RESERVOIR_WORDS] === SKY_LIGHT) sky++;
    }
    return { hits, skyDraws: sky, otherDraws: state.visibilityRay.length - sky, withoutLight: hits - state.visibilityRay.length };
}

function checkSkyDraws(draws: ReturnType<typeof skyDraws>, withSun: boolean, fail: (message: string) => void): void {
    if (draws.hits === 0) fail('no hits');
    if (!withSun && draws.otherDraws > 0) fail(`${draws.otherDraws} hits drew something other than the sky, with no other light in the scene`);
    if (withSun && (draws.otherDraws === 0 || draws.skyDraws === 0)) fail(`with the sun and the sky, ${draws.skyDraws} hits drew the sky and ${draws.otherDraws} the sun: both should be drawn`);
    if (draws.withoutLight > MAX_HITS_WITHOUT_LIGHT_SHARE * draws.hits) fail(`${draws.withoutLight} of ${draws.hits} hits drew no light; the sky reaches every one`);
}

interface SkyGroup {
    cells: number;
    measured: [number, number, number];
    expected: [number, number, number];
    ratios: number[];
    visibilitySum: number;
}

function newSkyGroup(): SkyGroup {
    return { cells: 0, measured: [0, 0, 0], expected: [0, 0, 0], ratios: [], visibilitySum: 0 };
}

function percentile(values: readonly number[], share: number): number {
    if (values.length === 0) return Number.NaN;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
}

/** Section 6 reads the cells this many times, a history's length (16 frames) apart, and sums the readings. */
const SKY_READINGS = 4;

/**
 * Section 6 (and 6b with `sunRadiance`, the light the sun adds to every
 * cell: section 1's `LIT_RADIANCE`, 0 without the sun).
 */
function checkSkyCorner(sunRadiance: number): (device: GPUDevice, scenario: Scenario) => Promise<Checks> {
    return (device, scenario) => checkSkyCornerWith(device, scenario, sunRadiance);
}

async function checkSkyCornerWith(device: GPUDevice, scenario: Scenario, sunRadiance: number): Promise<Checks> {
    const withSun = sunRadiance > 0;
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`${withSun ? 'sky and sun' : 'sky corner'}: ${message}`);
    };
    const groups = { openFloor: newSkyGroup(), floor: newSkyGroup(), wall: newSkyGroup() };
    let mixedCells = 0;
    let cellsWithoutSamples = 0;
    let draws = { hits: 0, skyDraws: 0, otherDraws: 0, withoutLight: 0 };
    for (let reading = 0; reading < SKY_READINGS; reading++) {
        if (reading > 0) await runFrames(device, scenario, MAX_SAMPLE_COUNT);
        const counts = await readSkyCorner(device, scenario, groups, sunRadiance, fail);
        mixedCells += counts.mixedCells;
        cellsWithoutSamples += counts.cellsWithoutSamples;
        draws = { hits: draws.hits + counts.draws.hits, skyDraws: draws.skyDraws + counts.draws.skyDraws, otherDraws: draws.otherDraws + counts.draws.otherDraws, withoutLight: draws.withoutLight + counts.draws.withoutLight };
    }

    const summary: Record<string, unknown> = {};
    for (const [name, group] of Object.entries(groups)) {
        const ratio = [0, 1, 2].map((channel) => group.measured[channel] / group.expected[channel]);
        if (group.cells === 0) fail(`no ${name} cells with samples`);
        else if (ratio.some((value) => !(Math.abs(value - 1) <= LAMBERT_TOLERANCE))) fail(`${name}: the cells read ${ratio.map((value) => value.toFixed(4)).join(', ')} x the expected (rgb), over ${group.cells} cell readings`);
        summary[name] = {
            cellReadings: group.cells,
            meanVisibility: group.visibilitySum / Math.max(group.cells, 1),
            overExpected: ratio,
            perCellRedRatioP5: percentile(group.ratios, 0.05),
            perCellRedRatioP50: percentile(group.ratios, 0.5),
            perCellRedRatioP95: percentile(group.ratios, 0.95),
        };
    }
    return { failures, data: { skyRadiance: SKY_RADIANCE, sunRadiance, readings: SKY_READINGS, ...draws, skyDrawShare: draws.skyDraws / Math.max(draws.skyDraws + draws.otherDraws, 1), mixedCells, cellsWithoutSamples, ...summary } };
}

/** One reading of section 6: this frame's hits by cell, and each cell's value against albedo × L × V (+ the sun's), added into `groups`. */
async function readSkyCorner(
    device: GPUDevice,
    scenario: Scenario,
    groups: Record<'openFloor' | 'floor' | 'wall', SkyGroup>,
    sunRadiance: number,
    fail: (message: string) => void,
): Promise<{ draws: ReturnType<typeof skyDraws>; mixedCells: number; cellsWithoutSamples: number }> {
    const state = await readFrameState(device, scenario);
    const draws = skyDraws(state);
    checkSkyDraws(draws, sunRadiance > 0, fail);

    // This frame's hits by cell: which surface, and the share of the sky each sees.
    const hitsByCell = new Map<number, { surface: 'floor' | 'wall' | 'mixed'; visibility: number[] }>();
    state.visibilityCell.forEach((cell, visibility) => {
        const ray = state.rays[state.visibilityQuery[visibility]];
        if (ray.surface !== 'floor' && ray.surface !== 'wall') return;
        const entry = hitsByCell.get(cell) ?? { surface: ray.surface, visibility: [] };
        if (entry.surface !== ray.surface) entry.surface = 'mixed';
        entry.visibility.push(cornerSkyVisibility(ray.hit, ray.surface));
        hitsByCell.set(cell, entry);
    });

    const capacity = scenario.cache.capacity;
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);
    const tiles = [...new Set([...hitsByCell.keys()].map(tileOf))];
    const slotOfTile = new Map(tiles.map((tile, slot) => [tile, slot]));
    const values = await readTiles(device, scenario.cache, tiles);

    let mixedCells = 0;
    let cellsWithoutSamples = 0;
    for (const [cell, hits] of hitsByCell) {
        if (hits.surface === 'mixed') {
            mixedCells++;
            continue;
        }
        const tile = tileOf(cell);
        const value = cellValue(values, capacity, slotOfTile.get(tile) ?? 0, cell - tile * capacity.numCellsPerTile);
        if (!(value[3] > 0)) {
            cellsWithoutSamples++;
            continue;
        }
        const visibility = hits.visibility.reduce((sum, share) => sum + share, 0) / hits.visibility.length;
        const into = hits.surface === 'wall' ? [groups.wall] : visibility >= OPEN_FLOOR_VISIBILITY ? [groups.floor, groups.openFloor] : [groups.floor];
        for (const group of into) {
            group.cells++;
            group.visibilitySum += visibility;
            for (let channel = 0; channel < 3; channel++) {
                group.measured[channel] += value[channel] / value[3];
                group.expected[channel] += ALBEDO * SKY_RADIANCE[channel] * visibility + sunRadiance;
            }
            group.ratios.push(value[0] / value[3] / (ALBEDO * SKY_RADIANCE[0] * visibility + sunRadiance));
        }
    }
    return { draws, mixedCells, cellsWithoutSamples };
}

async function checkClosedBox(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`closed box: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const draws = skyDraws(state);
    checkSkyDraws(draws, false, fail);
    const misses = state.rays.filter((ray) => ray.hitT < 0).length;
    if (misses > 0) fail(`${misses} probe rays left the box`);
    // Every probe sits on the back wall, inside the box (the camera sees no edge: CLOSED_BOX).
    const { minX, maxX, minY, maxY, minZ, maxZ } = CLOSED_BOX;
    const outsideOrigins = state.rays.filter(({ origin: [x, y, z] }) => !(x > minX && x < maxX && y > minY && y < maxY && z > minZ && z < maxZ)).length;
    if (outsideOrigins > 0) fail(`${outsideOrigins} probe rays start outside the box`);

    // Every hit ray, and every cell with samples, reads 0: the sky can't be seen from inside.
    let litRays = 0;
    let brightestRay = 0;
    state.rays.forEach((ray, index) => {
        if (ray.hitT < 0) return;
        const brightest = Math.max(state.radiance[index * 4], state.radiance[index * 4 + 1], state.radiance[index * 4 + 2]);
        brightestRay = Math.max(brightestRay, brightest);
        if (brightest !== 0) litRays++;
    });
    if (litRays > 0) fail(`${litRays} hit rays read light (brightest ${brightestRay})`);

    const capacity = scenario.cache.capacity;
    const tiles = Array.from(state.packedTiles);
    const values = await readTiles(device, scenario.cache, tiles);
    let cellsWithSamples = 0;
    let litCells = 0;
    let brightestCell = 0;
    tiles.forEach((_, slot) => {
        for (let cell = 0; cell < capacity.numCellsPerTileMip[0]; cell++) {
            const value = cellValue(values, capacity, slot, cell);
            if (!(value[3] > 0)) continue;
            cellsWithSamples++;
            const brightest = Math.max(value[0], value[1], value[2]);
            brightestCell = Math.max(brightestCell, brightest);
            if (brightest !== 0) litCells++;
        }
    });
    if (cellsWithSamples === 0) fail('no cell has samples');
    if (litCells > 0) fail(`${litCells} of ${cellsWithSamples} cells read light (brightest ${brightestCell})`);
    return { failures, data: { ...draws, misses, outsideOrigins, litRays, cellsWithSamples, litCells } };
}

// ---- The page --------------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    // As a host would ask (requestGIDevice): the adapter's storage limits, and its optional features;
    // but the platform's 8 storage buffers per stage, whatever the adapter allows (budget-device.ts).
    const { device, adapterStorageBuffersPerStage } = await requestBudgetDevice(adapter);
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);

    const failures: string[] = [];
    const data: Record<string, unknown> = { adapterStorageBuffersPerStage };
    // Capsaicin's single-bounce configuration (gi1_use_multibounce off): the checks below expect the
    // first bounce alone (RC3.3's multibounce has its own page, radiance-cache-multibounce).
    const run = async (
        name: string,
        lights: readonly LightDescriptor[],
        withPatch: boolean,
        frames: number,
        check: (device: GPUDevice, scenario: Scenario) => Promise<Checks>,
        world: ScenarioWorld = {},
    ): Promise<void> => {
        const scenario = await startScenario(device, name, lights, withPatch, { useMultibounce: false }, world);
        try {
            await runFrames(device, scenario, frames);
            const checks = await check(device, scenario);
            failures.push(...checks.failures);
            data[name] = checks.data;
        } finally {
            scenario.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    };

    await run('litPlane', [sun(SUN_TRAVEL)], false, CONVERGENCE_FRAMES, checkLitPlane);
    await run('resolve', [{ type: 'point', position: [0, 1.5, 2], range: 0, color: [1, 1, 1], intensity: 10, label: 'point' }], false, 1, checkResolve);
    await run('emissive', [sun(SUN_TRAVEL)], true, 1, checkEmissive);
    await run('twoSuns', [sun(SUN_TRAVEL, [1, 0, 0]), sun(BLUE_SUN_TRAVEL, [0, 0, 1])], false, CONVERGENCE_FRAMES, checkTwoSuns);
    await run('skyCorner', [], false, 60, checkSkyCorner(0), { sky: SKY_RADIANCE });
    await run('skyAndSun', [sun(SUN_TRAVEL)], false, 60, checkSkyCorner(LIT_RADIANCE), { sky: SKY_RADIANCE });
    await run('closedBox', [], false, 20, checkClosedBox, { sky: SKY_RADIANCE, closedBox: true });

    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data,
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
