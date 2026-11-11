/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/

/**
 * The CPU twin of the hash grid's shared WGSL (`hash-grid-wgsl.ts`): the
 * cell descriptor and both hashes, the cell index of the tile mip layout,
 * and the decay age, ported from Capsaicin v1.3's
 * `src/core/src/render_techniques/gi1/hash_grid_cache.hlsl` and
 * `src/core/src/math/hash.hlsl`, with the same names in camelCase.
 *
 * Every float step is rounded to f32 (`Math.fround`) as the GPU rounds it,
 * so for a hit whose floors aren't within a rounding step of an integer the
 * CPU gives the GPU's bits (the `radiance-cache-hash-grid` GPU page checks
 * it). Near such a boundary the two may land on different sides: WGSL lets
 * `log2`, `/` and `distance` be a few ULP off (WGSL §15.7.4), and so would
 * another D3D12 driver for Capsaicin.
 */

import type { Vec3 } from '../../../platform/contract/math.ts';
import type { HashGridCacheCapacity } from './capacity.ts';

const f32 = Math.fround;

/** `[f(v.x), f(v.y), f(v.z)]`: HLSL's per-component math on a float3. */
function map3(vector: Vec3, f: (value: number) => number): [number, number, number] {
    return [f(vector[0]), f(vector[1]), f(vector[2])];
}

/** `kGI1_InvalidId` (gi1.hlsl:30): no tile, no cell. */
export const INVALID_ID = 0xffffffff;
/** `kHashGridCache_TileDecay` (hash_grid_cache.hlsl:29): frames a tile may go untouched before `PurgeTiles` frees it. */
export const TILE_DECAY = 50;
/** `kHashGridCache_FloatQuantize` (hash_grid_cache.hlsl:32): fixed-point scale for atomic accumulation. */
export const FLOAT_QUANTIZE = 1e3;
/** `HASHGRIDCACHE_STEP_FACTOR` (hash_grid_cache.hlsl:92): cell sizes are powers of two in millimetres. */
export const STEP_FACTOR = 1e3;
/** `HASHGRIDCACHE_SIZE_FACTOR` (hash_grid_cache.hlsl:93). */
export const SIZE_FACTOR = f32(1e-3);

// ---- math/hash.hlsl ----------------------------------------------------------------------------

/** `pcgHash(uint)` (hash.hlsl:33-39): PCG's output permutation of one LCG step. */
export function pcgHash(value: number): number {
    const state = (Math.imul(value, 747796405) + 2891336453) >>> 0;
    // The shift is 4..19 bits, chosen by the state's top four bits.
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}

/** `xxHash(uint)` (hash.hlsl:102-112): xxHash32's round and avalanche on one word. */
export function xxHash(value: number): number {
    const prime32_2 = 2246822519;
    const prime32_3 = 3266489917;
    const prime32_4 = 668265263;
    const prime32_5 = 374761393;
    let ret = (value + prime32_5) >>> 0;
    ret = Math.imul(prime32_4, ((ret << 17) | (ret >>> (32 - 17))) >>> 0) >>> 0;
    ret = Math.imul(prime32_2, (ret ^ (ret >>> 15)) >>> 0) >>> 0;
    ret = Math.imul(prime32_3, (ret ^ (ret >>> 13)) >>> 0) >>> 0;
    return (ret ^ (ret >>> 16)) >>> 0;
}

// ---- hash_grid_cache.hlsl ----------------------------------------------------------------------

/** The per-frame constants the descriptor reads (`HashGridCacheConstants`, gi1_shared.h:59-92, and `g_Eye`). */
export interface HashGridCacheDescConstants {
    /** `cell_size`: tan of the angle one cell spans (gi1.cpp:1856-1859); see {@link hashGridCacheCellSizeConstant}. */
    readonly cellSize: number;
    readonly minCellSize: number;
    readonly tileCellRatio: number;
    readonly numBuckets: number;
}

/** `HashGridCache_Data` (hash_grid_cache.hlsl:111-117): what a hit tells the cache. */
export interface HashGridCacheData {
    readonly eyePosition: Vec3;
    readonly hitPosition: Vec3;
    /** The ray's direction (normalised). */
    readonly direction: Vec3;
    /** The ray's length from its origin to the hit. */
    readonly hitDistance: number;
}

/**
 * `HashGridCache_Desc` (hash_grid_cache.hlsl:119-130), plus the parts of the
 * key it is made from (Capsaicin keeps those only with `DEBUG_HASH_CELLS`),
 * which the tests use to pick hits.
 */
export interface HashGridCacheDesc {
    /** Which bucket the tile goes in: the pcg chain modulo `num_buckets`. */
    readonly bucketIndex: number;
    /** The tile's checksum: the xxHash chain, never 0 (0 marks an empty slot). */
    readonly tileHash: number;
    /** The cell inside the tile's finest level: the hit's cell, projected along the ray's dominant axis. */
    readonly cellOffset: readonly [number, number];
    /** `l`: log2 of the cell size in millimetres. */
    readonly level: number;
    readonly cellSize: number;
    readonly tileSize: number;
    /** `signed_c`: the tile's integer coordinate. */
    readonly tileCoordinate: Vec3;
    /** `signed_d`: the direction's bin on each axis, 0..4. */
    readonly directionBin: Vec3;
    /** `t`: 1 when the ray is shorter than a tile. */
    readonly shortRay: number;
}

/** `distance(a, b)` in f32. */
function distanceF32(a: Vec3, b: Vec3): number {
    const dx = f32(a[0] - b[0]);
    const dy = f32(a[1] - b[1]);
    const dz = f32(a[2] - b[2]);
    return f32(Math.sqrt(f32(f32(f32(dx * dx) + f32(dy * dy)) + f32(dz * dz))));
}

/** HLSL's `uint(float)`: toward zero, saturated (the GPU's conversion; negative gives 0). */
function floatToUint(value: number): number {
    if (!(value > 0)) return 0;
    return Math.min(Math.trunc(value), 0xffffffff);
}

/** `asuint(int(float))`: toward zero, saturated to i32, then the bits as u32. */
function floatToIntBits(value: number): number {
    const clamped = Math.max(-2147483648, Math.min(2147483647, Math.trunc(value)));
    return clamped >>> 0;
}

/**
 * `HashGridCache_GetCellSize` (hash_grid_cache.hlsl:96-102): the cell size at
 * a position, a power of two in millimetres that doubles at discrete
 * distances from the eye.
 */
export function hashGridCacheGetCellSize(position: Vec3, eyePosition: Vec3, constants: Pick<HashGridCacheDescConstants, 'cellSize' | 'minCellSize'>): number {
    const cellSizeStep = Math.max(f32(distanceF32(eyePosition, position) * f32(constants.cellSize)), f32(constants.minCellSize));
    const logStepMultiplier = floatToUint(Math.log2(f32(STEP_FACTOR * cellSizeStep)));
    return f32(SIZE_FACTOR * 2 ** logStepMultiplier);
}

/**
 * `HashGridCache_GetDesc` (hash_grid_cache.hlsl:133-192): the tile's bucket
 * and checksum and the cell's offset inside it, from the key (level, tile
 * coordinate, direction bin, short-ray flag).
 */
export function hashGridCacheGetDesc(data: HashGridCacheData, constants: HashGridCacheDescConstants): HashGridCacheDesc {
    const hit = map3(data.hitPosition, f32);
    const direction = map3(data.direction, f32);
    const tileCellRatio = f32(constants.tileCellRatio);

    // :140-143
    const cellSizeStep = Math.max(f32(distanceF32(data.eyePosition, hit) * f32(constants.cellSize)), f32(constants.minCellSize));
    const logStepMultiplier = Math.floor(Math.log2(f32(STEP_FACTOR * cellSizeStep)));
    const hitCellSize = f32(SIZE_FACTOR * 2 ** logStepMultiplier);
    const hitTileSize = f32(hitCellSize * tileCellRatio);

    // :145-146
    const signedC = map3(hit, (value) => Math.floor(f32(value / hitTileSize)));
    const signedD = map3(direction, (value) => Math.floor(f32(0.5 + f32(f32(f32(0.5 * value) + 0.5) * 4))));

    // :148-151
    const l = floatToUint(logStepMultiplier);
    const c = map3(signedC, floatToIntBits);
    const d = map3(signedD, floatToIntBits);
    const t = f32(data.hitDistance) < hitTileSize ? 1 : 0;

    // :153-156: pcgHash(l + pcgHash(c.x + ... pcgHash(d.z + pcgHash(t)))), built innermost first.
    const keyOutermostLast = [d[2], d[1], d[0], c[2], c[1], c[0], l];
    let bucket = pcgHash(t);
    for (const component of keyOutermostLast) bucket = pcgHash((component + bucket) >>> 0);
    const bucketIndex = bucket % constants.numBuckets;

    // :158-162: the xxHash chain over the same key; max(1, ...) keeps 0 for "empty".
    let checksum = xxHash(t);
    for (const component of keyOutermostLast) checksum = xxHash((component + checksum) >>> 0);
    const tileHash = Math.max(1, checksum);

    // :164-179: the hit's cell inside its tile, on the plane across the ray's dominant axis.
    const e = map3(hit, (value) => f32(Math.floor(f32(value / hitCellSize)) - f32(Math.floor(f32(value / hitTileSize)) * tileCellRatio)));
    const absDirection = map3(direction, Math.abs);
    const maxDirection = Math.max(absDirection[0], absDirection[1], absDirection[2]);
    let cellOffset: [number, number];
    if (absDirection[0] === maxDirection) cellOffset = [floatToUint(e[1]), floatToUint(e[2])];
    else if (absDirection[1] === maxDirection) cellOffset = [floatToUint(e[0]), floatToUint(e[2])];
    else cellOffset = [floatToUint(e[0]), floatToUint(e[1])];

    return { bucketIndex, tileHash, cellOffset, level: l, cellSize: hitCellSize, tileSize: hitTileSize, tileCoordinate: signedC, directionBin: signedD, shortRay: t };
}

/** `HashGridCache_CellIndex(uint2, tile_index, mip_level)` (hash_grid_cache.hlsl:207-221). */
export function hashGridCacheCellIndex(cellOffsetMip0: readonly [number, number], tileIndex: number, mipLevel: number, capacity: HashGridCacheCapacity): number {
    const mipSize = capacity.sizeTileMip[0] >> mipLevel;
    const x = cellOffsetMip0[0] >> mipLevel;
    const y = cellOffsetMip0[1] >> mipLevel;
    return tileIndex * capacity.numCellsPerTile + capacity.firstCellOffsetTileMip[mipLevel] + x + y * mipSize;
}

/** `HashGridCache_CellOffsetMip0` (hash_grid_cache.hlsl:195-205): the tile and the finest-level offset of a finest-level cell index. */
export function hashGridCacheCellOffsetMip0(cellIndexMip0: number, capacity: HashGridCacheCapacity): { readonly tileIndex: number; readonly cellOffset: readonly [number, number] } {
    const tileIndex = Math.floor(cellIndexMip0 / capacity.numCellsPerTile);
    let cellLinearOffset = cellIndexMip0 % capacity.numCellsPerTile;
    const yStep = capacity.sizeTileMip[0];
    const y = Math.floor(cellLinearOffset / yStep);
    cellLinearOffset -= y * yStep;
    return { tileIndex, cellOffset: [cellLinearOffset, y] };
}

/**
 * Frames since a tile was last touched, as `PurgeTiles` works it out
 * (gi1.comp:1726-1738), across the frame counter's wrap: a stamp above the
 * frame means the counter wrapped since.
 */
export function hashGridCacheTileAge(frameIndex: number, tileDecay: number): number {
    const frame = frameIndex >>> 0;
    const stamp = tileDecay >>> 0;
    if (frame < stamp) return (0xffffffff - stamp + frame + 1) >>> 0;
    return frame - stamp;
}

/**
 * `cell_size` for `HashGridCacheConstants` (gi1.cpp:1856-1859): tan of the
 * angle `cellSizeOption` pixels span, from the vertical field of view and the
 * render size. `max(1/h, h/w²)` is the larger of two per-pixel factors:
 * `1/h` (a pixel's share of the screen's height) and `h/w²` (which wins only
 * on a screen taller than it is wide); Capsaicin's own expression.
 */
export function hashGridCacheCellSizeConstant(fovY: number, width: number, height: number, cellSizeOption: number): number {
    return Math.tan(fovY * cellSizeOption * Math.max(1 / height, height / (width * width)));
}

/** The vertical field of view of a perspective projection (column-major): `projection[5]` is `1 / tan(fovY / 2)`. */
export function verticalFieldOfView(projection: ArrayLike<number>): number {
    return 2 * Math.atan(1 / projection[5]);
}

// ---- UpdateTiles and the filtered readback -------------------------------------------------------

/** Four floats: rgb radiance (times the sample count, for a stored cell value) and a sample count. */
export type CellValue = readonly [number, number, number, number];

/**
 * `UpdateTiles`' temporal accumulation of one cell (gi1.comp:2162-2177): the
 * stored value (radiance times its count, and the count) and this frame's
 * samples (their sum and count), each turned into a mean, blended with weight
 * 1 / count, the count capped at `maxCount`, and stored times the count. In
 * f32 steps, as the GPU computes it (before the value is packed as halves).
 */
export function updateTilesAccumulate(stored: CellValue, newSamples: CellValue, maxCount: number, keepWithoutSamples = false, budget = keepWithoutSamples): [number, number, number, number] {
    // Lever 1's budget build (hash-grid-population-wgsl.ts, updateTilesWgsl, `budget`): a cell in use this
    // frame (keepWithoutSamples) with no new sample keeps its value; a cell nobody used fades as
    // Capsaicin's, but stops at a count of 1 instead of emptying.
    if (newSamples[3] <= 0 && (keepWithoutSamples || (budget && stored[3] <= 1))) return [stored[0], stored[1], stored[2], stored[3]];
    const sampleCount = f32(Math.min(f32(stored[3] + newSamples[3]), f32(maxCount)));
    const storedScale = f32(Math.max(stored[3], 1));
    const newScale = f32(Math.max(newSamples[3], 1));
    const radiance = stored.map((value) => f32(value / storedScale));
    const newRadiance = newSamples.map((value) => f32(value / newScale));
    const blended = radiance[3] <= 0 ? newRadiance : radiance.map((value, channel) => f32(value + f32(f32(newRadiance[channel] - value) * f32(1 / sampleCount))));
    return [f32(blended[0] * sampleCount), f32(blended[1] * sampleCount), f32(blended[2] * sampleCount), f32(blended[3] * sampleCount)];
}

/**
 * Which level `HashGridCache_FilteredRadianceDirect` (and `...Indirect`)
 * reads (hash_grid_cache.hlsl:434-494): the finest cell's value, replaced by
 * the next coarser level's while the value in hand has fewer than `maxCount`
 * samples and that level exists. `levels[0]` is the finest cell's value,
 * then its three coarser cells' (`null` where a level has no cells). Returns
 * the level read and its value; the resolve divides rgb by `max(count, 1)`.
 */
export function hashGridCacheFilteredRadiance(levels: readonly (CellValue | null)[], maxCount: number): { readonly level: number; readonly value: CellValue } {
    let level = 0;
    let value = levels[0];
    if (!value) throw new RangeError("@pmndrs/ddgi: the finest level's value is needed");
    for (let next = 1; next < levels.length; next++) {
        const coarser = levels[next];
        if (value[3] < maxCount && coarser) {
            level = next;
            value = coarser;
        }
    }
    return { level, value };
}
