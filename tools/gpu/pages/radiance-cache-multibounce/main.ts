/**
 * RC3.3: GI-1.2's multibounce cells, on the GPU, through the technique
 * (Capsaicin's default: `useMultibounce` on).
 *
 * The sunlit corner of `radiance-cache-world-cache` (`corner.ts`): floor and
 * wall, albedo 0.5, a sun along (0, -1, -1), so both get E = 8 cos 45° and
 * nothing shadows them. Every surface a second-bounce ray can reach (the
 * floor in front of the wall, the wall's front) leaves the same direct
 * radiance L = albedo / π × E toward any direction, and every light sample
 * is exact (one sun, weight 1). That makes Capsaicin's estimator easy to
 * state: each second-bounce sample is L × brdf / pdf / (1 - 0.7) = albedo × L
 * / 0.3 (Lambert: brdf / pdf = albedo), and a cell's indirect value is the
 * mean of its samples, so albedo × L / 0.3 = 1.667 L.
 *
 * 1. **The kernels, one frame** (the seventh frame from an empty cache,
 *    against what the GPU held after the sixth; the platform's frame index,
 *    which seeds the random numbers and the page reports as `frame`, is 9
 *    then, as the scenario's two set-up frames come first):
 *    - `PopulateMultibounceCells`: each first-bounce entry is kept exactly
 *      when the CPU's port of `rand()` (seed = its visibility index, the
 *      frame) is above 0.7; a kept entry has a second-bounce entry exactly
 *      when the CPU's ray (`sampleBRDFAndEvaluateDiffuse` with the next two
 *      numbers, about the base surface's normal) hits the floor or the wall
 *      (rays within 1 mm of an edge, and hits on the corner's line, are
 *      counted, not checked); the GPU's
 *      direction is the CPU's within half precision, and brdf and pdf are
 *      albedo / π × cos and cos / π.
 *    - `UpdateMultibounceCells`: each first-bounce cell's indirect scratch
 *      after the frame holds the sum of its second-bounce samples, computed
 *      on the CPU from the read-back cells (the filtered direct value), brdf
 *      and pdf (within one quantisation step per sample), and its count;
 *      cells with no sample hold 0.
 *    - The lag: each first-bounce cell's indirect value after the seventh
 *      frame is `UpdateTiles`' accumulation of its value after the sixth and
 *      the scratch the sixth left (within half precision).
 *    - Resolve: a probe ray that doesn't bypass the cache reads its cell's
 *      filtered direct plus filtered indirect value; one that bypasses reads
 *      its own light sample plus its own second-bounce sample, if it has one
 *      (register D40). The bypass flags are the resolve list's.
 *    - Reported, not checked: how many probe rays Capsaicin's own indices
 *      would have patched by mistake (D40).
 * 2. **Converged** (40 frames from an empty cache): every first-bounce
 *    finest-level cell with indirect samples reads albedo × L / 0.3 within 2%,
 *    and every resolved probe ray whose filtered indirect value has samples
 *    reads L × (1 + albedo / 0.3) within 2% (the rest L). The physical second
 *    bounce here is L × albedo × F, F the share of the cosine-weighted
 *    hemisphere the lit surfaces cover (at most 1/2 for the floor beside an
 *    endless wall), so Capsaicin's estimator reads above it: its weight
 *    1 / (1 - 0.7) undoes a discard that its mean (over the kept samples only)
 *    already ignores, and a ray that misses adds no sample. Reported: the mean
 *    of the bypassed rays, whose estimate counts the discarded and missed
 *    rays as zero and so stays unbiased.
 * 3. **Two-room's dark room** (room B, no lamp, lit through the doorway; a
 *    camera inside it looking at its east wall, so everything in view is
 *    room B): the mean interpolated irradiance E over the picture with the
 *    multibounce off (30 frames), on (30 frames, the cache emptied when the
 *    option changes), and off again. Stated before the run: E rises, and by
 *    a lot (more than half again), since nearly every second-bounce ray in a
 *    closed room hits and each sample weighs 1 / 0.3; and off again returns
 *    to the first value (within 10%: a single frame's probes are noisy). The
 *    three pictures (E, tone-mapped at one exposure) are drawn for the
 *    screenshot.
 *    The radiance cache is prepared with the multibounce off, so its world
 *    cache holds no indirect values (register D42): turning it on prepares
 *    again (`platform.prepare()`), turning it off is live. Checked: the
 *    platform's memory for the technique drops by at least the indirect
 *    arrays (534,773,760 bytes at Capsaicin's capacity) without the
 *    multibounce, and comes back when it is prepared on; off, then prepared
 *    again, it drops once more; the world cache's stats say which.
 *
 * 4. **The sky at second-bounce hits** (RC6.1: the corner with no lights
 *    under a constant sky, 6 frames): `GenerateMultibounceReservoirs` draws
 *    the sky as `GenerateReservoirs` does (Capsaicin's environment light,
 *    register D33). With the sky the only light, every second-bounce entry
 *    that gets a shadow ray drew the sky (`GI_DIRECT_SKY_LIGHT`), so do the
 *    first-bounce ones, and all but the rare one whose sky direction's packed
 *    numbers land on the horizon (weight 0) get one; some second-bounce
 *    entries exist, so the check can fail.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc3.3 multibounce" -- pnpm test:gpu --filter radiance-cache-multibounce`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { floatToHalfBits, halfBitsToFloat, makeRandom, rand, rand2, sampleBRDFAndEvaluateDiffuse } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { HASH_GRID_COUNTER, type HashGridCacheCapacity } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import type { HashGridCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cache.ts';
import {
    hashGridCacheCellIndex,
    hashGridCacheCellOffsetMip0,
    FLOAT_QUANTIZE,
    hashGridCacheFilteredRadiance,
    updateTilesAccumulate,
    type CellValue,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import { BYPASS_CACHE_BIT, RESERVOIR_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { GI_DIRECT_SKY_LIGHT } from '../../../../packages/ddgi-core/src/platform/contract/shade.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';
import {
    ALBEDO,
    cellValue,
    LIT_RADIANCE,
    MAX_SAMPLE_COUNT,
    readFrameState,
    readIndirectScratch,
    readTiles,
    relativeError,
    runFrames,
    startScenario,
    sun,
    SUN_TRAVEL,
    TOLERANCE,
    type FrameState,
    type ProbeRay,
    type Scenario,
    type ScenarioWorld,
} from '../radiance-cache-world-cache/corner.ts';

/** `discard_multibounce_ray_probability` (gi1.h:64), as the GPU compares it: in f32. */
const DISCARD = Math.fround(0.7);
/** `max_multibounce_sample_count` (gi1.h:65). */
const MAX_MULTIBOUNCE_SAMPLE_COUNT = 16;
/** One half-precision step, relative. */
const HALF_STEP = 2 ** -10;
/** Capsaicin's second-bounce sample on the corner: albedo × L / (1 - 0.7). */
const INDIRECT_RADIANCE = (ALBEDO * LIT_RADIANCE) / (1 - DISCARD);
/** A CPU ray this close to an edge of the floor or the wall is counted, not checked (m). */
const EDGE_MARGIN = 1e-3;
/** How far the GPU's half-packed direction may sit from the CPU's, per component. */
const DIRECTION_TOLERANCE = 2e-3;
const KERNEL_FRAMES = 6;
const CONVERGENCE_FRAMES = 40;

interface Checks {
    readonly failures: string[];
    readonly data: Record<string, unknown>;
}

function checker(section: string): { failures: string[]; fail: (message: string) => void } {
    const failures: string[] = [];
    return {
        failures,
        fail: (message: string) => {
            if (failures.length < 12) failures.push(`${section}: ${message}`);
        },
    };
}

// ---- The corner's geometry on the CPU ----------------------------------------------------------

/** The normal of the surface a probe ray hit (the corner's quads are flat; their front faces face the camera's side). */
function surfaceNormal(ray: ProbeRay): Vec3 | null {
    if (ray.surface === 'floor') return [0, 1, 0];
    if (ray.surface === 'wall') return [0, 0, 1];
    return null;
}

/** Where a ray from `origin` along `direction` first meets the floor (|x|, |z| ≤ 200 at y = 0) or the wall's front (|x| ≤ 20, 0 ≤ y ≤ 10 at z = 0). */
function traceCorner(origin: Vec3, direction: Vec3): { hit: boolean; nearEdge: boolean; point: Vec3 | null } {
    let best = Infinity;
    let point: Vec3 | null = null;
    let nearEdge = false;
    const at = (t: number): Vec3 => [origin[0] + t * direction[0], origin[1] + t * direction[1], origin[2] + t * direction[2]];
    if (direction[2] < 0 && origin[2] > 0) {
        const t = -origin[2] / direction[2];
        const p = at(t);
        const margin = Math.min(20 - Math.abs(p[0]), p[1], 10 - p[1]);
        if (Math.abs(margin) < EDGE_MARGIN) nearEdge = true;
        if (margin >= 0 && t < best) {
            best = t;
            point = p;
        }
    }
    if (direction[1] < 0 && origin[1] > 0) {
        const t = -origin[1] / direction[1];
        const p = at(t);
        const margin = Math.min(200 - Math.abs(p[0]), 200 - Math.abs(p[2]));
        if (Math.abs(margin) < EDGE_MARGIN) nearEdge = true;
        if (margin >= 0 && t < best) {
            best = t;
            point = p;
        }
    }
    return { hit: point !== null, nearEdge, point };
}

// ---- Cell values -------------------------------------------------------------------------------

/** Read-back values of whole tiles, direct or indirect, by tile index. */
class TileValues {
    private constructor(
        private readonly capacity: HashGridCacheCapacity,
        private readonly slots: Map<number, number>,
        private readonly values: Float32Array,
    ) {}

    static async read(device: GPUDevice, cache: HashGridCache, tiles: Iterable<number>, which: 'direct' | 'indirect'): Promise<TileValues> {
        const list = [...new Set(tiles)];
        const values = await readTiles(device, cache, list, which);
        return new TileValues(cache.capacity, new Map(list.map((tile, slot) => [tile, slot])), values);
    }

    /** The value of a cell (any level) of a tile that was read. */
    cell(cellIndex: number): CellValue {
        const tile = Math.floor(cellIndex / this.capacity.numCellsPerTile);
        const slot = this.slots.get(tile);
        if (slot === undefined) throw new Error(`tile ${tile} was not read back`);
        return cellValue(this.values, this.capacity, slot, cellIndex - tile * this.capacity.numCellsPerTile);
    }

    /** `HashGridCache_FilteredRadiance*` of a finest-level cell: its value or a coarser level's, by `maxCount`. */
    filtered(cellIndexMip0: number, maxCount: number): CellValue {
        const { tileIndex, cellOffset } = hashGridCacheCellOffsetMip0(cellIndexMip0, this.capacity);
        const levels = [0, 1, 2, 3].map((level) => (this.capacity.sizeTileMip[level] > 0 ? this.cell(hashGridCacheCellIndex(cellOffset, tileIndex, level, this.capacity)) : null));
        return hashGridCacheFilteredRadiance(levels, maxCount).value;
    }
}

const tileOf = (cell: number, capacity: HashGridCacheCapacity): number => Math.floor(cell / capacity.numCellsPerTile);

/** The mean rgb of a stored value (radiance × count, count). */
const meanOf = (value: CellValue): [number, number, number] => [0, 1, 2].map((channel) => value[channel] / Math.max(value[3], 1)) as [number, number, number];

/** A float as the nearest half, as the GPU packs a cell value (register D31). */
const toHalf = (value: number): number => halfBitsToFloat(floatToHalfBits(value));

/** UpdateMultibounceCells' sample on the CPU: the second-bounce cell's filtered direct mean × brdf / pdf / (1 - p). */
function multibounceSample(direct: TileValues, cell: number, info: Float32Array, entry: number): [number, number, number] {
    const mean = meanOf(direct.filtered(cell, MAX_SAMPLE_COUNT));
    const pdf = info[entry * 4 + 3];
    return [0, 1, 2].map((channel) => (mean[channel] * info[entry * 4 + channel]) / pdf / (1 - DISCARD)) as [number, number, number];
}

// ---- Section 1: the kernels, one frame -----------------------------------------------------------

async function checkKernels(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const { failures, fail } = checker('kernels');
    const capacity = scenario.cache.capacity;

    // After the sixth frame from an empty cache: the first-bounce cells' indirect values and the scratch the frame left.
    const before = await readFrameState(device, scenario);
    const beforeTiles = new Set(Array.from(before.visibilityCell, (cell) => tileOf(cell, capacity)));
    const beforeIndirect = await TileValues.read(device, scenario.cache, beforeTiles, 'indirect');
    const beforeCells = [...beforeTiles].flatMap((tile) => Array.from({ length: capacity.numCellsPerTileMip[0] }, (_, cell) => tile * capacity.numCellsPerTile + cell));
    const beforeScratch = await readIndirectScratch(device, scenario.cache, beforeCells);

    await runFrames(device, scenario, 1);
    const state = await readFrameState(device, scenario);
    const frame = scenario.technique.probes?.frameIndex ?? 0;
    const second = state.secondBounce;
    const count0 = state.visibilityCell.length;

    // PopulateMultibounceCells: the discard, the rays and their brdf and pdf.
    const entryOfBase = new Map<number, number>();
    second.query.forEach((base, entry) => {
        if (entryOfBase.has(base)) fail(`first-bounce entry ${base} traced two second-bounce rays`);
        entryOfBase.set(base, entry);
    });
    let kept = 0;
    let keptAndHit = 0;
    let nearEdges = 0;
    let discardMismatches = 0;
    let hitMismatches = 0;
    let worstDirection = 0;
    let worstBrdfOverPdf = 0;
    let wrongSecondHits = 0;
    for (let base = 0; base < count0; base++) {
        const ray = state.rays[state.visibilityQuery[base]];
        const normal = surfaceNormal(ray);
        const random = makeRandom(base, frame);
        const isKept = rand(random) > DISCARD;
        const entry = entryOfBase.get(base);
        if (!isKept) {
            if (entry !== undefined) {
                discardMismatches++;
                if (discardMismatches <= 3) fail(`entry ${base} traced although the CPU's rand() discards it`);
            }
            continue;
        }
        kept++;
        if (!normal) continue;
        // A hit on the corner's line (on the floor and the wall at once, within the classification's
        // 1 mm) can't tell its surface from the hit point: counted with the edges.
        if (Math.abs(ray.hit[1]) < EDGE_MARGIN && Math.abs(ray.hit[2]) < EDGE_MARGIN) {
            nearEdges++;
            continue;
        }
        const sample = sampleBRDFAndEvaluateDiffuse([ALBEDO, ALBEDO, ALBEDO], normal, rand2(random));
        const traced = traceCorner(ray.hit, sample.direction);
        if (traced.nearEdge) {
            nearEdges++;
            continue;
        }
        if (traced.hit) keptAndHit++;
        if (traced.hit !== (entry !== undefined)) {
            hitMismatches++;
            const gpu = entry === undefined ? '' : ` at t ${second.hitT[entry].toFixed(4)} from ${second.origin[entry].map((v) => v.toFixed(5)).join(', ')} along ${second.direction[entry].map((v) => v.toFixed(4)).join(', ')}`;
            if (hitMismatches <= 3) fail(`entry ${base} (${ray.surface} at ${ray.hit.map((v) => v.toFixed(5)).join(', ')}): the CPU's ray along ${sample.direction.map((v) => v.toFixed(4)).join(', ')} ${traced.hit ? 'hits' : 'misses'}, the GPU ${entry !== undefined ? 'has' : 'has no'} second-bounce entry${gpu}`);
            continue;
        }
        if (entry === undefined) continue;
        const direction = second.direction[entry];
        worstDirection = Math.max(worstDirection, ...[0, 1, 2].map((axis) => Math.abs(direction[axis] - sample.direction[axis])));
        const brdf = second.info[entry * 4];
        const pdf = second.info[entry * 4 + 3];
        worstBrdfOverPdf = Math.max(worstBrdfOverPdf, relativeError(brdf / pdf, ALBEDO), Math.abs(pdf - sample.pdf) / Math.max(sample.pdf, 1e-3));
        const end = second.origin[entry].map((value, axis) => value + second.hitT[entry] * direction[axis]);
        if (traced.point && Math.hypot(...end.map((value, axis) => value - (traced.point as Vec3)[axis])) > 1e-2 * Math.max(1, second.hitT[entry])) wrongSecondHits++;
    }
    if (discardMismatches > 0) fail(`${discardMismatches} discarded entries traced`);
    if (hitMismatches > 0) fail(`${hitMismatches} of ${kept} kept entries disagree with the CPU on hitting`);
    if (worstDirection > DIRECTION_TOLERANCE) fail(`a second-bounce direction is ${worstDirection} off the CPU's`);
    if (worstBrdfOverPdf > 4 * HALF_STEP) fail(`brdf / pdf or pdf is ${(worstBrdfOverPdf * 100).toFixed(3)}% off Lambert's`);
    if (wrongSecondHits > 0) fail(`${wrongSecondHits} second-bounce hits are not where the CPU's ray meets the corner`);
    if (second.count === 0) fail('no second-bounce entries');
    if (second.count !== state.counters[HASH_GRID_COUNTER.visibilityCount1]) fail('the second-bounce count and its lists disagree');

    // UpdateMultibounceCells: each first-bounce cell's indirect scratch against the CPU's sum.
    const secondTiles = Array.from(second.cell, (cell) => tileOf(cell, capacity));
    const direct = await TileValues.read(device, scenario.cache, [...secondTiles, ...Array.from(state.visibilityCell, (cell) => tileOf(cell, capacity))], 'direct');
    const indirect = await TileValues.read(device, scenario.cache, Array.from(state.visibilityCell, (cell) => tileOf(cell, capacity)), 'indirect');
    const expected = new Map<number, { sum: [number, number, number]; count: number }>();
    const sampleOf = new Map<number, [number, number, number]>();
    second.query.forEach((base, entry) => {
        const sample = multibounceSample(direct, second.cell[entry], second.info, entry);
        sampleOf.set(base, sample);
        const baseCell = state.visibilityCell[base];
        const sum = expected.get(baseCell) ?? { sum: [0, 0, 0], count: 0 };
        sample.forEach((value, channel) => (sum.sum[channel] += Math.round(FLOAT_QUANTIZE * value)));
        sum.count++;
        expected.set(baseCell, sum);
    });
    const baseCells = [...new Set(state.visibilityCell)];
    const scratch = await readIndirectScratch(device, scenario.cache, baseCells);
    let scratchMismatches = 0;
    let emptyCellsChecked = 0;
    for (const cell of baseCells) {
        const gpu = scratch.get(cell) as Uint32Array;
        const cpu = expected.get(cell) ?? { sum: [0, 0, 0], count: 0 };
        if (cpu.count === 0) emptyCellsChecked++;
        const off = gpu[3] !== cpu.count || [0, 1, 2].some((channel) => Math.abs(gpu[channel] - cpu.sum[channel]) > cpu.count + 1);
        if (off) {
            scratchMismatches++;
            if (scratchMismatches <= 3) fail(`cell ${cell}'s indirect scratch is ${Array.from(gpu).join(', ')}, the CPU's ${cpu.sum.join(', ')}, ${cpu.count}`);
        }
    }
    if (scratchMismatches > 0) fail(`${scratchMismatches} of ${baseCells.length} first-bounce cells' indirect scratch differ from the CPU's sum`);

    // The lag: the indirect value now is UpdateTiles' accumulation of the previous frame's value and scratch.
    const updated = new Set(state.updateTiles);
    let lagChecked = 0;
    let lagMismatches = 0;
    let worstLag = 0;
    for (const cell of baseCells) {
        const tile = tileOf(cell, capacity);
        if (!beforeTiles.has(tile) || !updated.has(tile)) continue;
        const previous = beforeIndirect.cell(cell);
        const quantized = beforeScratch.get(cell) as Uint32Array;
        const samples: CellValue = [quantized[0] / FLOAT_QUANTIZE, quantized[1] / FLOAT_QUANTIZE, quantized[2] / FLOAT_QUANTIZE, quantized[3]];
        const accumulated = updateTilesAccumulate(previous, samples, MAX_MULTIBOUNCE_SAMPLE_COUNT).map(toHalf);
        const now = indirect.cell(cell);
        lagChecked++;
        const error = Math.max(...[0, 1, 2, 3].map((channel) => Math.abs(now[channel] - accumulated[channel]) / Math.max(Math.abs(accumulated[channel]), 1e-2)));
        worstLag = Math.max(worstLag, error);
        if (error > 2 * HALF_STEP) {
            lagMismatches++;
            if (lagMismatches <= 3) fail(`cell ${cell}'s indirect value is ${now.map((v) => v.toFixed(4)).join(', ')}, UpdateTiles' accumulation of last frame's gives ${accumulated.map((v) => v.toFixed(4)).join(', ')}`);
        }
    }
    if (lagChecked === 0) fail('no cell to check the lag on');
    if (lagMismatches > 0) fail(`${lagMismatches} of ${lagChecked} cells don't hold last frame's scratch accumulated`);

    // Resolve: filtered direct + indirect, or (bypassing) the own light sample + own second bounce.
    const resolveFlag = new Map<number, boolean>();
    for (const entry of state.resolve) resolveFlag.set(entry & ~BYPASS_CACHE_BIT, (entry & BYPASS_CACHE_BIT) !== 0);
    let flagMismatches = 0;
    let resolvedChecked = 0;
    let bypassedChecked = 0;
    let bypassedWithSecondBounce = 0;
    let rayMismatches = 0;
    let worstRay = 0;
    for (let base = 0; base < count0; base++) {
        const bypass = resolveFlag.get(base);
        if (bypass === undefined) continue;
        if ((state.visibilityBypass[base] !== 0) !== bypass) flagMismatches++;
        const query = state.visibilityQuery[base];
        let expectedRadiance: number[];
        if (bypass) {
            const own = sampleOf.get(base) ?? [0, 0, 0];
            if (sampleOf.has(base)) bypassedWithSecondBounce++;
            expectedRadiance = own.map((value) => LIT_RADIANCE + value);
            bypassedChecked++;
        } else {
            const cell = state.visibilityCell[base];
            const directMean = meanOf(direct.filtered(cell, MAX_SAMPLE_COUNT));
            const indirectMean = meanOf(indirect.filtered(cell, MAX_MULTIBOUNCE_SAMPLE_COUNT));
            expectedRadiance = directMean.map((value, channel) => value + indirectMean[channel]);
            resolvedChecked++;
        }
        const error = Math.max(...[0, 1, 2].map((channel) => relativeError(state.radiance[query * 4 + channel], expectedRadiance[channel])));
        worstRay = Math.max(worstRay, error);
        if (error > 4 * HALF_STEP) {
            rayMismatches++;
            if (rayMismatches <= 3) fail(`probe ray ${query} (${bypass ? 'bypassing' : 'resolved'}) reads ${[0, 1, 2].map((c) => state.radiance[query * 4 + c].toFixed(4)).join(', ')}, expected ${expectedRadiance.map((v) => v.toFixed(4)).join(', ')}`);
        }
    }
    if (flagMismatches > 0) fail(`${flagMismatches} bypass flags differ from the resolve list's`);
    if (rayMismatches > 0) fail(`${rayMismatches} of ${resolvedChecked + bypassedChecked} probe rays read other than the CPU's`);
    if (bypassedWithSecondBounce === 0) fail('no bypassing probe ray had a second bounce: the D40 path is not exercised');

    return {
        failures,
        data: {
            frame,
            firstBounceEntries: count0,
            kept,
            keptShare: kept / Math.max(count0, 1),
            keptAndHit,
            secondBounceEntries: second.count,
            raysNearAnEdge: nearEdges,
            worstDirectionError: worstDirection,
            worstBrdfOverPdfError: worstBrdfOverPdf,
            baseCells: baseCells.length,
            baseCellsWithoutSamples: emptyCellsChecked,
            scratchMismatches,
            lagCellsChecked: lagChecked,
            worstLagError: worstLag,
            resolvedRaysChecked: resolvedChecked,
            bypassedRaysChecked: bypassedChecked,
            bypassedRaysWithSecondBounce: bypassedWithSecondBounce,
            worstRayError: worstRay,
            capsaicinIndices: capsaicinMisroutes(state),
        },
    };
}

/**
 * What Capsaicin's own indices would do on this frame (register D40), read
 * from the lists: `PopulateCells` patches the "probe ray" of every
 * second-bounce shadow ray with the bypass bit, which is a first-bounce
 * visibility index (gi1.comp:2100-2106); `UpdateMultibounceCells` reads its
 * flag from `VisibilityRayBuffer[count0 + i]`, a shadow-ray entry
 * (gi1.comp:1977-1981).
 */
function capsaicinMisroutes(state: FrameState): Record<string, number> {
    const count0 = state.visibilityCell.length;
    const secondBounceShadowRays = Array.from(state.visibilityRay).filter((entry) => (entry & ~BYPASS_CACHE_BIT) >= count0);
    const populateCellsPatches = secondBounceShadowRays.filter((entry) => (entry & BYPASS_CACHE_BIT) !== 0).length;
    let updatePatches = 0;
    let updateReadsPastTheList = 0;
    for (let entry = 0; entry < state.secondBounce.count; entry++) {
        const index = count0 + entry;
        if (index >= state.visibilityRay.length) {
            updateReadsPastTheList++;
            continue;
        }
        if ((state.visibilityRay[index] & BYPASS_CACHE_BIT) !== 0) updatePatches++;
    }
    const bypassingWithSecondBounce = state.secondBounce.query.filter((base) => state.visibilityBypass[base] !== 0).length;
    return {
        populateCellsPatchesOfUnrelatedRays: populateCellsPatches,
        updateMultibounceCellsPatchesOfUnrelatedRays: updatePatches,
        updateMultibounceCellsFlagsReadPastTheShadowRays: updateReadsPastTheList,
        bypassingRaysWhoseSecondBounceCapsaicinDrops: bypassingWithSecondBounce,
    };
}

// ---- Section 2: converged ------------------------------------------------------------------------

async function checkConverged(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const { failures, fail } = checker('converged');
    const capacity = scenario.cache.capacity;
    const state = await readFrameState(device, scenario);
    const tiles = Array.from(state.visibilityCell, (cell) => tileOf(cell, capacity));
    const indirect = await TileValues.read(device, scenario.cache, tiles, 'indirect');
    const baseCells = [...new Set(state.visibilityCell)];
    let cellsWithSamples = 0;
    let cellsOff = 0;
    let worstCell = 0;
    for (const cell of baseCells) {
        const value = indirect.cell(cell);
        if (!(value[3] > 0)) continue;
        cellsWithSamples++;
        const error = Math.max(...meanOf(value).map((mean) => relativeError(mean, INDIRECT_RADIANCE)));
        worstCell = Math.max(worstCell, error);
        if (error > TOLERANCE) cellsOff++;
    }
    if (cellsWithSamples === 0) fail('no first-bounce cell has indirect samples');
    if (cellsOff > 0) fail(`${cellsOff} of ${cellsWithSamples} cells' indirect values are more than 2% off albedo x L / 0.3 = ${INDIRECT_RADIANCE.toFixed(4)} (worst ${(worstCell * 100).toFixed(2)}%)`);

    const resolveFlag = new Map<number, boolean>();
    for (const entry of state.resolve) resolveFlag.set(entry & ~BYPASS_CACHE_BIT, (entry & BYPASS_CACHE_BIT) !== 0);
    let withIndirect = 0;
    let withoutIndirect = 0;
    let raysOff = 0;
    let worstRay = 0;
    let bypassedSum = 0;
    let bypassedRays = 0;
    let resolvedSum = 0;
    for (const [base, bypass] of resolveFlag) {
        const query = state.visibilityQuery[base];
        const red = state.radiance[query * 4];
        if (bypass) {
            bypassedSum += red;
            bypassedRays++;
            continue;
        }
        resolvedSum += red;
        const hasIndirect = indirect.filtered(state.visibilityCell[base], MAX_MULTIBOUNCE_SAMPLE_COUNT)[3] > 0;
        const expected = LIT_RADIANCE + (hasIndirect ? INDIRECT_RADIANCE : 0);
        if (hasIndirect) withIndirect++;
        else withoutIndirect++;
        const error = Math.max(...[0, 1, 2].map((channel) => relativeError(state.radiance[query * 4 + channel], expected)));
        worstRay = Math.max(worstRay, error);
        if (error > TOLERANCE) {
            raysOff++;
            if (raysOff <= 3) fail(`probe ray ${query} reads ${state.radiance[query * 4].toFixed(4)}, expected ${expected.toFixed(4)}`);
        }
    }
    if (raysOff > 0) fail(`${raysOff} of ${withIndirect + withoutIndirect} resolved probe rays are more than 2% off`);
    if (withIndirect === 0) fail('no resolved probe ray has a second bounce');
    return {
        failures,
        data: {
            directRadiance: LIT_RADIANCE,
            capsaicinIndirectRadiance: INDIRECT_RADIANCE,
            physicalIndirectAtMost: ALBEDO * LIT_RADIANCE * 0.5,
            cellsWithIndirectSamples: cellsWithSamples,
            worstCellError: worstCell,
            resolvedRaysWithIndirect: withIndirect,
            resolvedRaysWithoutIndirect: withoutIndirect,
            worstResolvedRayError: worstRay,
            meanResolvedRayOverDirect: resolvedSum / Math.max(withIndirect + withoutIndirect, 1) / LIT_RADIANCE,
            bypassedRays,
            meanBypassedRayOverDirect: bypassedSum / Math.max(bypassedRays, 1) / LIT_RADIANCE,
        },
    };
}

// ---- Section 3: two-room's dark room ------------------------------------------------------------

const ROOM_WIDTH = 1280;
const ROOM_HEIGHT = 720;
/** Inside room B (x 4.2-8.2 m, z 0-4 m), near the partition, looking at the east wall: everything in view is room B. */
const DARK_ROOM_CAMERA: PageCamera = { position: [4.6, 1.5, 3.4], target: [8.2, 0.9, 1.2], fovYDeg: 70, near: 0.05, far: 100 };
/**
 * Frames per setting. 30 until RC4.2; since then the screen probes' history is the filtered atlas
 * (Capsaicin's, FilterScreenProbes), which holds the multibounce's light longer after the cache is
 * emptied: at 30 frames the single bounce read 0.0628 again against 0.0534 the first time (M5), at
 * 120 frames 0.0539 against 0.0540.
 */
const ROOM_FRAMES = 120;

/** Mean luminance (Rec. 709) of the interpolated irradiance, and the picture as halves. */
async function readIrradiance(device: GPUDevice, technique: RadianceCacheTechnique): Promise<{ mean: number; pixels: Float32Array }> {
    const probes = technique.probes;
    if (!probes) throw new Error('two-room: no probes after encodeScreen');
    const pixels = halves(await readTexture(device, probes.irradiance, 8, ROOM_WIDTH, ROOM_HEIGHT));
    let sum = 0;
    for (let pixel = 0; pixel < ROOM_WIDTH * ROOM_HEIGHT; pixel++) {
        sum += 0.2126 * pixels[pixel * 4] + 0.7152 * pixels[pixel * 4 + 1] + 0.0722 * pixels[pixel * 4 + 2];
    }
    return { mean: sum / (ROOM_WIDTH * ROOM_HEIGHT), pixels };
}

/** E as a picture: `exposure × E` tone-mapped by x / (1 + x), then sRGB-ish gamma 1/2.2. */
function drawIrradiance(pixels: Float32Array, exposure: number, caption: string, gallery: HTMLElement): void {
    const image = new ImageData(ROOM_WIDTH, ROOM_HEIGHT);
    for (let pixel = 0; pixel < ROOM_WIDTH * ROOM_HEIGHT; pixel++) {
        for (let channel = 0; channel < 3; channel++) {
            const value = Math.max(0, pixels[pixel * 4 + channel] * exposure);
            image.data[pixel * 4 + channel] = Math.round(255 * Math.pow(value / (1 + value), 1 / 2.2));
        }
        image.data[pixel * 4 + 3] = 255;
    }
    const canvas = document.createElement('canvas');
    canvas.width = ROOM_WIDTH;
    canvas.height = ROOM_HEIGHT;
    canvas.style.width = '420px';
    canvas.style.display = 'block';
    canvas.getContext('2d')?.putImageData(image, 0, 0);
    const label = document.createElement('div');
    label.textContent = caption;
    label.style.font = '11px sans-serif';
    label.style.color = '#ddd';
    const column = document.createElement('div');
    column.append(label, canvas);
    gallery.append(column);
}

async function checkTwoRoom(device: GPUDevice, gallery: HTMLElement): Promise<Checks> {
    const { failures, fail } = checker('two-room');
    const manifest = loadManifest('two-room');
    const platform = new GIPlatformImpl(device, { label: 'radiance-cache-multibounce two-room' });
    const technique = platform.addTechnique(new RadianceCacheTechnique({ useMultibounce: false }));
    platform.setOutput(technique);
    await platform.prepare();
    describeScene(platform, manifest);
    const screen = await PageScreen.create(platform, device, ROOM_WIDTH, ROOM_HEIGHT);
    const camera = frameCamera(DARK_ROOM_CAMERA, ROOM_WIDTH, ROOM_HEIGHT);
    const frameInput: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera };
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: 'radiance-cache-multibounce two-room' });
        platform.encode(encoder, frameInput);
        screen.encode(encoder, DARK_ROOM_CAMERA);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    try {
        frame();
        await platform.rays.whenBuilt();
        frame();
        await device.queue.onSubmittedWorkDone();
        technique.resetHistory();
        // The technique's GPU bytes as the platform reports them, and whether its world cache holds the indirect arrays.
        const memory = (): { bytes: number; multibounceArrays: number } => ({
            bytes: platform.stats.memory.techniques[technique.id] ?? 0,
            multibounceArrays: technique.counters['world cache multibounce arrays'] ?? -1,
        });
        const memorySteps: Record<string, { bytes: number; multibounceArrays: number }> = { preparedOff: memory() };
        const run = async (useMultibounce: boolean): Promise<{ mean: number; pixels: Float32Array }> => {
            // Turning it off is read every frame; on needs the indirect arrays, made at prepare (register
            // D42). Either change empties the cache (Capsaicin's needs_hash_grid_clear).
            technique.settings.useMultibounce = useMultibounce;
            if (useMultibounce) {
                await platform.prepare();
                frame();
                await device.queue.onSubmittedWorkDone();
                memorySteps.preparedOn = memory();
            }
            for (let index = 0; index < ROOM_FRAMES; index++) frame();
            await device.queue.onSubmittedWorkDone();
            return readIrradiance(device, technique);
        };
        const single = await run(false);
        const multi = await run(true);
        const singleAgain = await run(false);
        memorySteps.offLive = memory();
        await platform.prepare();
        frame();
        await device.queue.onSubmittedWorkDone();
        memorySteps.preparedOffAgain = memory();
        // gi1.cpp:497-552: a uint2 and four uints per cell, at Capsaicin's 22,282,240 cells.
        const indirectArrays = 534_773_760;
        const freed = (memorySteps.preparedOn?.bytes ?? 0) - memorySteps.preparedOff.bytes;
        const freedAgain = (memorySteps.preparedOn?.bytes ?? 0) - memorySteps.preparedOffAgain.bytes;
        if (!(freed >= indirectArrays)) fail(`memory: prepared without the multibounce the technique holds ${freed} bytes less than with it, expected at least ${indirectArrays}`);
        if (freedAgain !== freed) fail(`memory: prepared off again it holds ${freedAgain} bytes less, the first time ${freed}`);
        if (memorySteps.offLive.bytes !== memorySteps.preparedOn?.bytes) fail('memory: turning the multibounce off changed the memory before the next prepare');
        const arrays = [memorySteps.preparedOff, memorySteps.preparedOn, memorySteps.offLive, memorySteps.preparedOffAgain].map((step) => step?.multibounceArrays);
        if (arrays.join() !== '0,1,1,0') fail(`memory: the world cache's 'multibounce arrays' stat read ${arrays.join()}, expected 0,1,1,0`);
        const rise = multi.mean / single.mean;
        if (!(single.mean > 0)) fail(`the dark room reads no light with a single bounce (${single.mean})`);
        if (!(rise > 1.5)) fail(`the multibounce raises the dark room's mean E by ${rise.toFixed(3)}x, expected more than 1.5x`);
        if (Math.abs(singleAgain.mean / single.mean - 1) > 0.1) fail(`single bounce again reads ${singleAgain.mean.toPrecision(4)}, first ${single.mean.toPrecision(4)}`);
        const nonFinite = [single, multi, singleAgain].reduce((sum, picture) => sum + picture.pixels.filter((value) => !Number.isFinite(value) || value < 0).length, 0);
        if (nonFinite > 0) fail(`${nonFinite} negative or non-finite irradiance values`);
        const exposure = 1 / Math.max(multi.mean, 1e-6);
        drawIrradiance(single.pixels, exposure, `two-room, dark room: E, single bounce (mean ${single.mean.toPrecision(3)})`, gallery);
        drawIrradiance(multi.pixels, exposure, `two-room, dark room: E, multibounce (mean ${multi.mean.toPrecision(3)}, ${rise.toFixed(2)}x)`, gallery);
        drawIrradiance(singleAgain.pixels, exposure, `two-room, dark room: E, single bounce again (mean ${singleAgain.mean.toPrecision(3)})`, gallery);
        return {
            failures,
            data: { camera: DARK_ROOM_CAMERA, frames: ROOM_FRAMES, meanSingle: single.mean, meanMulti: multi.mean, meanSingleAgain: singleAgain.mean, rise, nonFinite, memory: { ...memorySteps, freed, freedAgain } },
        };
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- Section 4: the sky at second-bounce hits (RC6.1) ----------------------------------------------

/** Section 4's constant sky, the only light. */
const SKY_RADIANCE: Vec3 = [1, 0.8, 0.6];
/** Entries allowed to draw nothing: the sky direction's two numbers are kept at 16 bits, and one in about 131,000 rounds onto the horizon (weight 0). */
const MAX_WITHOUT_LIGHT_SHARE = 1e-4;

async function checkSkySecondBounce(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const { failures, fail } = checker('sky at second-bounce hits');
    const state = await readFrameState(device, scenario);
    const count0 = state.counters[HASH_GRID_COUNTER.visibilityCount0];
    const count1 = state.secondBounce.count;
    const draws = { first: { shadowRays: 0, sky: 0 }, second: { shadowRays: 0, sky: 0 } };
    state.visibilityRay.forEach((entry, ray) => {
        const bounce = (entry & ~BYPASS_CACHE_BIT) >= count0 ? draws.second : draws.first;
        bounce.shadowRays++;
        if (state.reservoirs[ray * RESERVOIR_WORDS] === GI_DIRECT_SKY_LIGHT) bounce.sky++;
    });
    if (count1 === 0) fail('no second-bounce entries');
    for (const [name, bounce, entries] of [['first-bounce', draws.first, count0], ['second-bounce', draws.second, count1]] as const) {
        if (bounce.sky !== bounce.shadowRays) fail(`${bounce.shadowRays - bounce.sky} of ${bounce.shadowRays} ${name} shadow rays drew something other than the sky, the only light`);
        if (entries - bounce.shadowRays > MAX_WITHOUT_LIGHT_SHARE * entries) fail(`${entries - bounce.shadowRays} of ${entries} ${name} entries drew no light; the sky reaches every one`);
    }
    return { failures, data: { skyRadiance: SKY_RADIANCE, firstBounceEntries: count0, secondBounceEntries: count1, draws } };
}

// ---- The page --------------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const { device, adapterStorageBuffersPerStage } = await requestBudgetDevice(adapter);
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    const gallery = document.createElement('div');
    gallery.style.display = 'flex';
    gallery.style.gap = '8px';
    document.body.style.background = '#222';
    document.body.style.margin = '0';
    document.body.append(gallery);

    const failures: string[] = [];
    const data: Record<string, unknown> = { adapterStorageBuffersPerStage };
    const corner = async (
        name: string,
        frames: number,
        check: (device: GPUDevice, scenario: Scenario) => Promise<Checks>,
        lights = [sun(SUN_TRAVEL)],
        world: ScenarioWorld = {},
    ): Promise<void> => {
        // Capsaicin's default configuration: multibounce on.
        const scenario = await startScenario(device, name, lights, false, {}, world);
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
    await corner('kernels', KERNEL_FRAMES, checkKernels);
    await corner('converged', CONVERGENCE_FRAMES, checkConverged);
    await corner('skySecondBounce', KERNEL_FRAMES, checkSkySecondBounce, [], { sky: SKY_RADIANCE });
    const room = await checkTwoRoom(device, gallery);
    failures.push(...room.failures);
    data.twoRoom = room.data;

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
