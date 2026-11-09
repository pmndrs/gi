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
 * The hash grid's shared WGSL: one module every hash-grid pass includes,
 * ported from Capsaicin v1.3's `src/core/src/render_techniques/gi1/hash_grid_cache.hlsl`
 * (the descriptor, insert, find, cell indices, value packing, filtered
 * readback), `src/core/src/math/hash.hlsl` (`pcgHash`, `xxHash`) and
 * `src/core/src/math/pack.hlsl` (`packHalf4`), with the cache kernels of
 * `gi1.comp` (`ClearCounters`' cache half, `PurgeTiles`). Functions keep
 * Capsaicin's names and its variable names, so a line here can be read next
 * to its source line.
 *
 * What differs, and why:
 * - Capsaicin's ~30 buffers are three packed storage buffers (register D6,
 *   `capacity.ts`); a buffer is reached at a fixed word offset
 *   (`HASHGRIDCACHE_HASHBUFFER + tile_index`) where Capsaicin indexes its
 *   buffer array (`g_HashGridCache_BuffersUint[HASHGRIDCACHE_HASHBUFFER][tile_index]`).
 * - HLSL `out` parameters become returned structs; HLSL overloads get
 *   distinct names (`HashGridCache_CellIndexLinear` for the `uint1` forms).
 * - `InterlockedCompareExchange` is strong; WGSL has only
 *   `atomicCompareExchangeWeak`, which may fail with the slot still empty, so
 *   the insert retries that slot until the exchange happens or the slot holds
 *   a tile (register D22; same result).
 * - The filtered readbacks drop their debug-view argument (Capsaicin's debug
 *   views are not part of the baseline).
 * - `g_Eye` and `g_FrameIndex` (GI1's own constants) ride in the hash grid's
 *   constant block.
 */

import type { HashGridCacheBufferLayout } from './capacity.ts';
import { HASH_GRID_COUNTER } from './capacity.ts';
import { FLOAT_QUANTIZE, INVALID_ID, SIZE_FACTOR, STEP_FACTOR, TILE_DECAY } from './hash-grid-cpu.ts';

/** `GPUShaderStage.COMPUTE` (the global doesn't exist in Node, where the tests import this file). */
const SHADER_STAGE_COMPUTE = 0x4;

/** The hash grid's bindings inside the group a kernel puts it in. */
export const HASH_GRID_CACHE_BINDING = {
    /** `HashGridCacheConstants`, uniform. */
    constants: 0,
    /** The tile words (counters, hash, decay, packed tile lists), `array<atomic<u32>>`. */
    tileWords: 1,
    /** The cell values (direct, then indirect), `array<vec2u>`. */
    cellValues: 2,
    /** The update scratch (direct, then indirect), `array<atomic<u32>>`. */
    updateScratch: 3,
} as const;

/** Layout entries of the hash grid's group: a uniform and three storage buffers. */
export const HASH_GRID_CACHE_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    { binding: HASH_GRID_CACHE_BINDING.constants, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { binding: HASH_GRID_CACHE_BINDING.tileWords, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    { binding: HASH_GRID_CACHE_BINDING.cellValues, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    { binding: HASH_GRID_CACHE_BINDING.updateScratch, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
];

/**
 * Word offsets of `HashGridCacheConstants`' fields (`gi1_shared.h:59-92`,
 * debug fields left out, `g_Eye` and `g_FrameIndex` added at the front and
 * `g_UseDirectLighting` in the first padding word), for
 * the TypeScript that writes the block. 32 words, 128 bytes.
 */
export const HASH_GRID_CONSTANTS_WORD = {
    eyePosition: 0,
    frameIndex: 3,
    cellSize: 4,
    minCellSize: 5,
    tileSize: 6,
    tileCellRatio: 7,
    numBuckets: 8,
    numTiles: 9,
    numCells: 10,
    numTilesPerBucket: 11,
    sizeTileMip0: 12,
    numCellsPerTileMip0: 16,
    numCellsPerTile: 20,
    firstCellOffsetTileMip0: 21,
    bufferPingPong: 25,
    maxSampleCount: 26,
    discardMultibounceRayProbability: 27,
    maxMultibounceSampleCount: 28,
    useDirectLighting: 29,
} as const;

/** Bytes of `HashGridCacheConstants`. */
export const HASH_GRID_CONSTANTS_BYTES = 128;

/** A float literal WGSL accepts for any finite number (`1e-3` stays `1e-3`, `50` becomes `50.0`). */
function floatLiteral(value: number): string {
    const text = String(value);
    return /[.e]/.test(text) ? text : `${text}.0`;
}

const HASH_GRID_CONSTANTS_STRUCT_WGSL = /* wgsl */ `
// HashGridCacheConstants (gi1_shared.h:59-92), without the debug fields, with GI1's g_Eye and
// g_FrameIndex (gi1.comp:33-36) at the front and its g_UseDirectLighting after the options.
struct HashGridCacheConstants {
    eye_position: vec3f,
    frame_index: u32,
    cell_size: f32,                 // tan of the angle a cell spans (gi1.cpp:1856-1859)
    min_cell_size: f32,
    tile_size: f32,
    tile_cell_ratio: f32,           // tile_size / cell_size
    num_buckets: u32,
    num_tiles: u32,
    num_cells: u32,
    num_tiles_per_bucket: u32,
    size_tile_mip0: u32,
    size_tile_mip1: u32,
    size_tile_mip2: u32,
    size_tile_mip3: u32,
    num_cells_per_tile_mip0: u32,
    num_cells_per_tile_mip1: u32,
    num_cells_per_tile_mip2: u32,
    num_cells_per_tile_mip3: u32,
    num_cells_per_tile: u32,        // sum all available mips
    first_cell_offset_tile_mip0: u32,
    first_cell_offset_tile_mip1: u32,
    first_cell_offset_tile_mip2: u32,
    first_cell_offset_tile_mip3: u32,
    buffer_ping_pong: u32,
    max_sample_count: f32,
    discard_multibounce_ray_probability: f32,
    max_multibounce_sample_count: f32,
    use_direct_lighting: u32,       // GI1's g_UseDirectLighting (gi1.comp:35; gi1_use_direct_lighting)
    padding1: u32,
    padding2: u32,
}
`;

/** `math/hash.hlsl` and `math/pack.hlsl`: the pieces the hash grid uses. */
const MATH_WGSL = /* wgsl */ `
// pcgHash(uint) (math/hash.hlsl:33-39). https://www.pcg-random.org/
fn pcgHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// xxHash(uint) (math/hash.hlsl:102-112). xxhash (https://github.com/Cyan4973/xxHash)
fn xxHash(value: u32) -> u32 {
    const prime32_2 = 2246822519u;
    const prime32_3 = 3266489917u;
    const prime32_4 = 668265263u;
    const prime32_5 = 374761393u;
    var ret = value + prime32_5;
    ret = prime32_4 * ((ret << 17u) | (ret >> (32u - 17u)));
    ret = prime32_2 * (ret ^ (ret >> 15u));
    ret = prime32_3 * (ret ^ (ret >> 13u));
    return ret ^ (ret >> 16u);
}

// packHalf4 / unpackHalf4 (math/pack.hlsl:423-428, 456-460): x and y in the first word, z and w
// in the second, the first of each pair in the low half (register D11: pack2x16float for f32tof16).
fn packHalf4(value: vec4f) -> vec2u {
    return vec2u(pack2x16float(value.xy), pack2x16float(value.zw));
}

fn unpackHalf4(packed_value: vec2u) -> vec4f {
    return vec4f(unpack2x16float(packed_value.x), unpack2x16float(packed_value.y));
}
`;

/**
 * The shared module, with the hash grid's bindings in `group` (entries
 * {@link HASH_GRID_CACHE_GROUP_ENTRIES}). Every hash-grid pass includes it;
 * it declares no entry point.
 */
export function hashGridCacheWgsl(group: number, layout: HashGridCacheBufferLayout): string {
    const words = layout.tileWords;
    return /* wgsl */ `
// ---- The hash-grid radiance cache (Capsaicin v1.3, hash_grid_cache.hlsl; AMD, MIT) ----

const kGI1_InvalidId: u32 = ${INVALID_ID}u;                                        // gi1.hlsl:30
// The number of frames before an unused tile is evicted and returned to the pool (hash_grid_cache.hlsl:29).
const kHashGridCache_TileDecay: u32 = ${TILE_DECAY}u;
// The amount of float quantization for atomic updates of the hash cells as integer (:32).
const kHashGridCache_FloatQuantize: f32 = ${floatLiteral(FLOAT_QUANTIZE)};
const HASHGRIDCACHE_STEP_FACTOR: f32 = ${floatLiteral(STEP_FACTOR)};                // :92
const HASHGRIDCACHE_SIZE_FACTOR: f32 = 1e-3;                                         // :93 (f32 ${SIZE_FACTOR})

// Where Capsaicin's buffers are in the packed tile words (register D6; capacity.ts): word offsets.
const HASHGRIDCACHE_PACKEDTILECOUNTBUFFER0: u32 = ${HASH_GRID_COUNTER.packedTileCount0}u;
const HASHGRIDCACHE_PACKEDTILECOUNTBUFFER1: u32 = ${HASH_GRID_COUNTER.packedTileCount1}u;
const HASHGRIDCACHE_UPDATETILECOUNTBUFFER: u32 = ${HASH_GRID_COUNTER.updateTileCount}u;
const HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0: u32 = ${HASH_GRID_COUNTER.visibilityCount0}u;
const HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1: u32 = ${HASH_GRID_COUNTER.visibilityCount1}u;
const HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER: u32 = ${HASH_GRID_COUNTER.visibilityRayCount}u;
const HASHGRIDCACHE_RESOLVECOUNTBUFFER: u32 = ${HASH_GRID_COUNTER.resolveCount}u;
const HASHGRIDCACHE_BUCKETOVERFLOWCOUNT: u32 = ${HASH_GRID_COUNTER.bucketOverflowCount}u;  // ours (register D23)
const HASHGRIDCACHE_HASHBUFFER: u32 = ${words.hashBuffer}u;
const HASHGRIDCACHE_DECAYTILEBUFFER: u32 = ${words.decayTileBuffer}u;
const HASHGRIDCACHE_PACKEDTILEINDEXBUFFER0: u32 = ${words.packedTileIndexBuffer0}u;
const HASHGRIDCACHE_PACKEDTILEINDEXBUFFER1: u32 = ${words.packedTileIndexBuffer1}u;
const HASHGRIDCACHE_UPDATELISTEDTILEBUFFER: u32 = ${words.updateListedTileBuffer}u;      // ours (register D32)
// In the cell values (vec2u elements) and the update scratch (words).
const HASHGRIDCACHE_VALUEBUFFER: u32 = ${layout.cellValues.valueBuffer}u;
const HASHGRIDCACHE_VALUEINDIRECTBUFFER: u32 = ${layout.cellValues.valueIndirectBuffer}u;
const HASHGRIDCACHE_UPDATECELLVALUEBUFFER: u32 = ${layout.updateScratch.updateCellValueBuffer}u;
const HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER: u32 = ${layout.updateScratch.updateCellValueIndirectBuffer}u;

${HASH_GRID_CONSTANTS_STRUCT_WGSL}
@group(${group}) @binding(${HASH_GRID_CACHE_BINDING.constants}) var<uniform> g_HashGridCacheConstants: HashGridCacheConstants;
@group(${group}) @binding(${HASH_GRID_CACHE_BINDING.tileWords}) var<storage, read_write> g_HashGridCache_TileWords: array<atomic<u32>>;
@group(${group}) @binding(${HASH_GRID_CACHE_BINDING.cellValues}) var<storage, read_write> g_HashGridCache_CellValues: array<vec2u>;
@group(${group}) @binding(${HASH_GRID_CACHE_BINDING.updateScratch}) var<storage, read_write> g_HashGridCache_UpdateScratch: array<atomic<u32>>;
${MATH_WGSL}
// ---- The ping-pong packed tile lists (hash_grid_cache.hlsl:75-78) ----
// buffer_ping_pong picks this frame's list (written) and last frame's (read by PurgeTiles);
// Capsaicin flips it at the end of every frame (gi1.cpp:3098-3099).

fn HashGridCache_PackedTileCountWord() -> u32 {
    return HASHGRIDCACHE_PACKEDTILECOUNTBUFFER0 + g_HashGridCacheConstants.buffer_ping_pong;
}

fn HashGridCache_PreviousPackedTileCountWord() -> u32 {
    return HASHGRIDCACHE_PACKEDTILECOUNTBUFFER1 - g_HashGridCacheConstants.buffer_ping_pong;
}

fn HashGridCache_PackedTileIndexWord(packed_tile_index: u32) -> u32 {
    return HASHGRIDCACHE_PACKEDTILEINDEXBUFFER0 + g_HashGridCacheConstants.buffer_ping_pong * g_HashGridCacheConstants.num_tiles + packed_tile_index;
}

fn HashGridCache_PreviousPackedTileIndexWord(packed_tile_index: u32) -> u32 {
    return HASHGRIDCACHE_PACKEDTILEINDEXBUFFER1 - g_HashGridCacheConstants.buffer_ping_pong * g_HashGridCacheConstants.num_tiles + packed_tile_index;
}

// ---- Hash-grid radiance caching common functions (hash_grid_cache.hlsl:88-246) ----

// Gets the size of the hash cell for the given world-space position (:96-102).
fn HashGridCache_GetCellSize(position: vec3f) -> f32 {
    let cell_size_step = max(distance(g_HashGridCacheConstants.eye_position, position) * g_HashGridCacheConstants.cell_size, g_HashGridCacheConstants.min_cell_size);
    let log_step_multiplier = u32(log2(HASHGRIDCACHE_STEP_FACTOR * cell_size_step));
    let hit_cell_size = HASHGRIDCACHE_SIZE_FACTOR * exp2(f32(log_step_multiplier));
    return hit_cell_size;
}

// :104-109
fn HashGridCache_GetTileSize(position: vec3f) -> f32 {
    let hit_cell_size = HashGridCache_GetCellSize(position);
    let hit_tile_size = hit_cell_size * g_HashGridCacheConstants.tile_cell_ratio;
    return hit_tile_size;
}

// :111-117
struct HashGridCache_Data {
    eye_position: vec3f,
    hit_position: vec3f,
    direction: vec3f,
    hit_distance: f32,
}

// :119-130
struct HashGridCache_Desc {
    bucket_index: u32,   // bucket index
    tile_hash: u32,      // tile hash
    cell_offset: vec2u,  // cell offset within the tile
}

// Gets the description for the radiance cell in the current frame structure (:132-192).
//
// The key is the cell size's level l (cells are powers of two in millimetres, doubling with
// distance from the eye), the tile's integer coordinate c, the ray direction's bin d (5 per
// axis) and t, whether the ray is shorter than a tile. The bucket is a pcg chain over the key;
// the tile's checksum, stored in its slot, an xxHash chain over the same key, never 0 (0 marks
// an empty slot). The cell is the hit's cell inside its tile, projected along the ray's
// dominant axis, so a tile is a 2D slab of cells facing the rays that made it.
fn HashGridCache_GetDesc(data: HashGridCache_Data) -> HashGridCache_Desc {
    let eye_position = data.eye_position;
    let hit_position = data.hit_position;
    let direction = data.direction;
    let hit_distance = data.hit_distance;

    let cell_size_step = max(distance(eye_position, hit_position) * g_HashGridCacheConstants.cell_size, g_HashGridCacheConstants.min_cell_size);
    let log_step_multiplier = floor(log2(HASHGRIDCACHE_STEP_FACTOR * cell_size_step));
    let hit_cell_size = HASHGRIDCACHE_SIZE_FACTOR * exp2(log_step_multiplier);
    let hit_tile_size = hit_cell_size * g_HashGridCacheConstants.tile_cell_ratio;

    let signed_c = floor(hit_position / hit_tile_size);
    let signed_d = floor(0.5 + (0.5 * direction + 0.5) * 4.0);

    let l = u32(log_step_multiplier);
    let c = bitcast<vec3u>(vec3i(signed_c));     // asuint because pcg and xxhash32 use uint and it matters for negative values
    let d = bitcast<vec3u>(vec3i(signed_d));     //
    let t = select(0u, 1u, hit_distance < hit_tile_size);

    let bucket_index = pcgHash(l +
                       pcgHash(c.x + pcgHash(c.y + pcgHash(c.z +
                       pcgHash(d.x + pcgHash(d.y + pcgHash(d.z +
                       pcgHash(t)))))))) % g_HashGridCacheConstants.num_buckets;

    let tile_hash  = max(1u,
                     xxHash(l +
                     xxHash(c.x + xxHash(c.y + xxHash(c.z +
                     xxHash(d.x + xxHash(d.y + xxHash(d.z +
                     xxHash(t)))))))));

    let e = floor(hit_position / hit_cell_size) - floor(hit_position / hit_tile_size) * g_HashGridCacheConstants.tile_cell_ratio;
    var cell_offset: vec2u;
    let abs_direction = abs(direction);
    let max_direction = max(max(abs_direction.x, abs_direction.y), abs_direction.z);
    if (abs_direction.x == max_direction) {
        cell_offset = vec2u(u32(e.y), u32(e.z));
    } else if (abs_direction.y == max_direction) {
        cell_offset = vec2u(u32(e.x), u32(e.z));
    } else {
        cell_offset = vec2u(u32(e.x), u32(e.y));
    }

    return HashGridCache_Desc(bucket_index, tile_hash, cell_offset);
}

// A tile index and a finest-level cell offset inside it.
struct HashGridCache_TileCell {
    tile_index: u32,
    cell_offset_mip0: vec2u,
}

// Used with HashGridCache_CellIndex for climbing mip levels (:194-205).
fn HashGridCache_CellOffsetMip0(cell_index_mip0: u32) -> HashGridCache_TileCell {
    let tile_index = cell_index_mip0 / g_HashGridCacheConstants.num_cells_per_tile;    // cell_index_mip0 considers if tiles use mipmaps
    var cell_linear_offset = cell_index_mip0 % g_HashGridCacheConstants.num_cells_per_tile;
    let mip_size = g_HashGridCacheConstants.size_tile_mip0;
    let y_step = mip_size;
    var cell_offset_mip0: vec2u;
    cell_offset_mip0.y = cell_linear_offset / y_step;
    cell_linear_offset -= cell_offset_mip0.y * y_step;
    cell_offset_mip0.x = cell_linear_offset;
    return HashGridCache_TileCell(tile_index, cell_offset_mip0);
}

// first_cell_offset_tile_mip_level[mip_level] (:209-214), as a switch: a private array indexed at
// run time would go to local memory.
fn HashGridCache_FirstCellOffsetTileMip(mip_level: u32) -> u32 {
    switch mip_level {
        case 0u: { return g_HashGridCacheConstants.first_cell_offset_tile_mip0; }
        case 1u: { return g_HashGridCacheConstants.first_cell_offset_tile_mip1; }
        case 2u: { return g_HashGridCacheConstants.first_cell_offset_tile_mip2; }
        default: { return g_HashGridCacheConstants.first_cell_offset_tile_mip3; }
    }
}

// HashGridCache_CellIndex(uint2 cell_offset_mip0, uint tile_index, uint mip_level) (:207-221):
// the cell over a finest-level offset at a coarser level. A tile's cells are its four levels one
// after another (64 + 16 + 4 + 1 by default), each row-major.
fn HashGridCache_CellIndex(cell_offset_mip0: vec2u, tile_index: u32, mip_level: u32) -> u32 {
    let mip_size = g_HashGridCacheConstants.size_tile_mip0 >> mip_level;
    let cell_offset = cell_offset_mip0 >> vec2u(mip_level);
    return tile_index * g_HashGridCacheConstants.num_cells_per_tile +
           HashGridCache_FirstCellOffsetTileMip(mip_level) +
           cell_offset.x + cell_offset.y * mip_size;
}

// HashGridCache_CellIndex(uint1 cell_offset_mip_level, uint tile_index, uint mip_level) (:228-241):
// used for loops where the tile grid doesn't matter.
fn HashGridCache_CellIndexLinear(cell_offset_mip_level: u32, tile_index: u32, mip_level: u32) -> u32 {
    return tile_index * g_HashGridCacheConstants.num_cells_per_tile +
           HashGridCache_FirstCellOffsetTileMip(mip_level) +
           cell_offset_mip_level;
}

// What HashGridCache_InsertCell returns through its out parameters, with its result.
struct HashGridCache_InsertResult {
    cell_index: u32,     // kGI1_InvalidId when the bucket is full
    tile_index: u32,
    is_new_tile: bool,
}

// InterlockedCompareExchange(g_HashGridCache_HashBuffer[tile_index], 0, tile_hash, previous_hash):
// the slot's value before the call. WGSL's exchange is weak (it may fail although the slot held
// 0), so a failure that still saw 0 tries again (register D22).
fn HashGridCache_CompareExchangeEmptySlot(tile_index: u32, tile_hash: u32) -> u32 {
    loop {
        let exchange = atomicCompareExchangeWeak(&g_HashGridCache_TileWords[HASHGRIDCACHE_HASHBUFFER + tile_index], 0u, tile_hash);
        if (exchange.exchanged || exchange.old_value != 0u) {
            return exchange.old_value;
        }
    }
}

// Inserts a new radiance cell inside the current frame hash-grid cache (:248-278): the first
// slot of the bucket that is empty (claimed) or already holds this tile (found). A bucket full of
// other tiles refuses the cell, and the refusal is counted (register D23).
fn HashGridCache_InsertCell(data: HashGridCache_Data) -> HashGridCache_InsertResult {
    var bucket_offset: u32;
    var is_new_tile = false;
    var tile_index = kGI1_InvalidId;
    let desc = HashGridCache_GetDesc(data);
    for (bucket_offset = 0u; bucket_offset < g_HashGridCacheConstants.num_tiles_per_bucket; bucket_offset++) {
        tile_index = bucket_offset + desc.bucket_index * g_HashGridCacheConstants.num_tiles_per_bucket;
        let previous_hash = HashGridCache_CompareExchangeEmptySlot(tile_index, desc.tile_hash);
        if (previous_hash == 0u) {
            is_new_tile = true;
            break;  // inserted new tile and cell
        }
        if (previous_hash == desc.tile_hash) {
            break;  // found existing tile and cell
        }
    }
    if (bucket_offset >= g_HashGridCacheConstants.num_tiles_per_bucket) {
        atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_BUCKETOVERFLOWCOUNT], 1u);
        return HashGridCache_InsertResult(kGI1_InvalidId, tile_index, false);  // too much collisions, out of tiles :(
    }

    return HashGridCache_InsertResult(HashGridCache_CellIndex(desc.cell_offset, tile_index, 0u), tile_index, is_new_tile);
}

// What HashGridCache_FindCell returns through its out parameter, with its result.
struct HashGridCache_FindResult {
    cell_index: u32,     // kGI1_InvalidId when the tile isn't in the cache
    tile_index: u32,
}

// Finds a radiance cell entry inside the current frame hash-grid cache (:280-298). Read-only; it
// stops at the bucket's first empty slot.
fn HashGridCache_FindCell(data: HashGridCache_Data) -> HashGridCache_FindResult {
    var bucket_offset: u32;
    var tile_index = kGI1_InvalidId;
    let desc = HashGridCache_GetDesc(data);
    for (bucket_offset = 0u; bucket_offset < g_HashGridCacheConstants.num_tiles_per_bucket; bucket_offset++) {
        tile_index = bucket_offset + desc.bucket_index * g_HashGridCacheConstants.num_tiles_per_bucket;
        let previous_hash = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_HASHBUFFER + tile_index]);
        if (previous_hash == 0u) {
            return HashGridCache_FindResult(kGI1_InvalidId, tile_index);  // no other tile in bucket
        }
        if (previous_hash == desc.tile_hash) {
            break;  // found tile and cell
        }
    }
    if (bucket_offset >= g_HashGridCacheConstants.num_tiles_per_bucket) {
        return HashGridCache_FindResult(kGI1_InvalidId, tile_index);  // not found in bucket
    }
    return HashGridCache_FindResult(HashGridCache_CellIndex(desc.cell_offset, tile_index, 0u), tile_index);  // found tile and cell
}

// Bump the tile's decay to the max. now that it's been 'touched': InterlockedExchange of the
// frame index into its decay stamp (gi1.comp:1000-1002, 1775-1777, 2886-2888). Returns the
// previous stamp; equal to the frame index means the tile was touched already this frame.
fn HashGridCache_TouchTile(tile_index: u32) -> u32 {
    return atomicExchange(&g_HashGridCache_TileWords[HASHGRIDCACHE_DECAYTILEBUFFER + tile_index], g_HashGridCacheConstants.frame_index);
}

// "If this cell is inside a new tile, we need to add the tile to the packed storage and clear its
// cells" (gi1.comp:1020-1036 and 1798-1812, the same block in both population passes): the tile
// joins this frame's packed list, and its finest-level values start at zero (the coarser levels
// are rewritten by UpdateTiles). With GI-1.2's multibounce on (Capsaicin's default) the indirect
// values are cleared too.
fn HashGridCache_AddNewTile(tile_index: u32) {
    let packed_tile_index = atomicAdd(&g_HashGridCache_TileWords[HashGridCache_PackedTileCountWord()], 1u);
    atomicStore(&g_HashGridCache_TileWords[HashGridCache_PackedTileIndexWord(packed_tile_index)], tile_index);

    // Clear mip0 cells (others will be reset anyways by UpdateTiles)
    for (var cell_offset = 0u; cell_offset < g_HashGridCacheConstants.num_cells_per_tile_mip0; cell_offset++) {
        let cell_index = HashGridCache_CellIndexLinear(cell_offset, tile_index, 0u);
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = vec2u(0u, 0u);
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index] = vec2u(0u, 0u);
    }
}

// Whether this call puts the tile on this frame's update list (g_HashGridCache_UpdateTileBuffer),
// asked only by a thread Capsaicin's rule would have append it (gi1.comp:1043-1048, 1819-1824:
// the tile is new, or this thread touched it first this frame). That rule can pick two threads
// for a new tile: the one that claimed its slot, and one that found the tile and touched it before
// the claimer did. Two UpdateTiles workgroups then update the tile at once and race on its scratch
// (register D32). The tile's own word, set to the frame plus one, lets exactly one of them append
// it: plus one, so the zeroed words of a new or reset cache never read as frame 0.
fn HashGridCache_ListTileForUpdate(tile_index: u32) -> bool {
    let listed = g_HashGridCacheConstants.frame_index + 1u;
    return atomicExchange(&g_HashGridCache_TileWords[HASHGRIDCACHE_UPDATELISTEDTILEBUFFER + tile_index], listed) != listed;
}

// Quantizes the radiance value so it can be blended atomically (:300-306). WGSL's round, like
// the HLSL round DXC compiles (Round_ne), rounds halfway cases to even.
fn HashGridCache_QuantizeRadiance(radiance: vec3f) -> vec4u {
    return vec4u(u32(round(kHashGridCache_FloatQuantize * radiance.x)),
                 u32(round(kHashGridCache_FloatQuantize * radiance.y)),
                 u32(round(kHashGridCache_FloatQuantize * radiance.z)), 1u);
}

// Recovers the previously quantized radiance value (:308-315).
fn HashGridCache_RecoverRadiance(quantized_radiance: vec4u) -> vec4f {
    return vec4f(f32(quantized_radiance.x) / kHashGridCache_FloatQuantize,
                 f32(quantized_radiance.y) / kHashGridCache_FloatQuantize,
                 f32(quantized_radiance.z) / kHashGridCache_FloatQuantize,
                 f32(quantized_radiance.w));
}

// The largest finite half.
const HASHGRIDCACHE_LARGEST_HALF: f32 = 65504.0;

// Two floats as halves, as HLSL's f32tof16 makes them (register D31): rounded to the nearest
// half (pack2x16float truncates on both our GPUs, NOTES, and UpdateTiles re-packs every value
// each frame, where truncation would pull a cell's running mean down by up to 16 half steps),
// and clamped to the largest finite half (beyond it pack2x16float is indeterminate in WGSL,
// where f32tof16 gives infinity).
fn HashGridCache_PackHalf2(value: vec2f) -> u32 {
    let clamped = clamp(value, vec2f(-HASHGRIDCACHE_LARGEST_HALF), vec2f(HASHGRIDCACHE_LARGEST_HALF));
    let packed = pack2x16float(clamped);
    return HashGridCache_RoundHalfBits(clamped.x, packed & 0xffffu) | (HashGridCache_RoundHalfBits(clamped.y, packed >> 16u) << 16u);
}

// The half 'bits' pack2x16float gave for 'value', or the next half away from zero when that one
// is nearer (ties keep 'bits').
fn HashGridCache_RoundHalfBits(value: f32, bits: u32) -> u32 {
    let magnitude = abs(value);
    let packed_magnitude = abs(unpack2x16float(bits).x);
    // Already the nearer one (or above), or the largest finite half: nothing further out to take.
    if (!(magnitude > packed_magnitude) || (bits & 0x7fffu) >= 0x7bffu) {
        return bits;
    }
    let next_magnitude = abs(unpack2x16float(bits + 1u).x);
    return select(bits, bits + 1u, next_magnitude - magnitude < magnitude - packed_magnitude);
}

// Packs the radiance for storing inside the hash grid (:355-359): rgb and a sample count as halves
// (packHalf4, rounded and clamped as f32tof16 would; HashGridCache_PackHalf2).
fn HashGridCache_PackRadiance(radiance: vec4f) -> vec2u {
    return vec2u(HashGridCache_PackHalf2(radiance.xy), HashGridCache_PackHalf2(radiance.zw));
}

// Unpacks the radiance from its packed format inside the hash grid (:361-365).
fn HashGridCache_UnpackRadiance(packed_radiance: vec2u) -> vec4f {
    return unpackHalf4(packed_radiance);
}

// The cell indices of a finest-level cell's three coarser levels, kGI1_InvalidId where a level
// has no cells (hash_grid_cache.hlsl:437-440).
struct HashGridCache_MipCells {
    mip1: u32,
    mip2: u32,
    mip3: u32,
}

fn HashGridCache_MipCellsOf(cell_index_mip0: u32) -> HashGridCache_MipCells {
    let tile_cell = HashGridCache_CellOffsetMip0(cell_index_mip0);
    let cell_offset_mip0 = tile_cell.cell_offset_mip0;
    let tile_index = tile_cell.tile_index;
    let cell_index_mip1 = select(kGI1_InvalidId, HashGridCache_CellIndex(cell_offset_mip0, tile_index, 1u), g_HashGridCacheConstants.size_tile_mip1 > 0u);
    let cell_index_mip2 = select(kGI1_InvalidId, HashGridCache_CellIndex(cell_offset_mip0, tile_index, 2u), g_HashGridCacheConstants.size_tile_mip2 > 0u);
    let cell_index_mip3 = select(kGI1_InvalidId, HashGridCache_CellIndex(cell_offset_mip0, tile_index, 3u), g_HashGridCacheConstants.size_tile_mip3 > 0u);
    return HashGridCache_MipCells(cell_index_mip1, cell_index_mip2, cell_index_mip3);
}

// HashGridCache_FilteredRadianceIndirect (:434-463), without its debug view: start at the cell,
// and while the value has fewer samples than the cap and a coarser level exists, take the
// coarser level's value instead.
fn HashGridCache_FilteredRadianceIndirect(cell_index_mip0: u32) -> vec4f {
    let mips = HashGridCache_MipCellsOf(cell_index_mip0);
    let max_count = g_HashGridCacheConstants.max_multibounce_sample_count;

    // Mip 0
    var indirect_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index_mip0]);
    // Mip 1
    if (indirect_radiance.w < max_count && mips.mip1 != kGI1_InvalidId) {
        indirect_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + mips.mip1]);
    }
    // Mip 2
    if (indirect_radiance.w < max_count && mips.mip2 != kGI1_InvalidId) {
        indirect_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + mips.mip2]);
    }
    // Mip 3
    if (indirect_radiance.w < max_count && mips.mip3 != kGI1_InvalidId) {
        indirect_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + mips.mip3]);
    }
    return indirect_radiance;
}

// HashGridCache_FilteredRadianceDirect (:465-494), without its debug view: as above, on the
// direct values and their cap.
fn HashGridCache_FilteredRadianceDirect(cell_index_mip0: u32) -> vec4f {
    let mips = HashGridCache_MipCellsOf(cell_index_mip0);
    let max_count = g_HashGridCacheConstants.max_sample_count;

    // Mip 0
    var direct_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index_mip0]);
    // Mip 1
    if (direct_radiance.w < max_count && mips.mip1 != kGI1_InvalidId) {
        direct_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + mips.mip1]);
    }
    // Mip 2
    if (direct_radiance.w < max_count && mips.mip2 != kGI1_InvalidId) {
        direct_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + mips.mip2]);
    }
    // Mip 3
    if (direct_radiance.w < max_count && mips.mip3 != kGI1_InvalidId) {
        direct_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + mips.mip3]);
    }
    return direct_radiance;
}

// ---- The world-cache lookup (world-cache.ts, WorldCacheLookup) ----

// A ray length no tile reaches: the key's short-ray flag is 0 for it (register D24).
const HASHGRIDCACHE_LOOKUP_HIT_DISTANCE: f32 = 3.0e38;

// As Capsaicin's reflections read the cache (gi1.comp:2866-2899): find the cell, touch its tile,
// and take the filtered direct value plus, with multibounce, the filtered indirect value, each
// divided by its sample count. Unlike there, the position is not jittered across the cell (the
// caller has the random numbers for that), and the ray is taken as longer than a tile, since the
// lookup isn't given its length (register D24). The normal is for a cache that needs it; the hash
// grid's key has none. Weight 1 where the cell is in the cache (even with no samples yet, as
// Capsaicin uses it), 0 where it isn't.
fn worldCacheLookup(position: vec3f, normal: vec3f, direction: vec3f) -> vec4f {
    let data = HashGridCache_Data(g_HashGridCacheConstants.eye_position, position, direction, HASHGRIDCACHE_LOOKUP_HIT_DISTANCE);
    let found = HashGridCache_FindCell(data);
    if (found.cell_index == kGI1_InvalidId) {
        return vec4f(0.0);
    }
    HashGridCache_TouchTile(found.tile_index);
    let direct_radiance = HashGridCache_FilteredRadianceDirect(found.cell_index);
    let indirect_radiance = HashGridCache_FilteredRadianceIndirect(found.cell_index);
    let radiance = direct_radiance.rgb / max(direct_radiance.w, 1.0) + indirect_radiance.rgb / max(indirect_radiance.w, 1.0);
    return vec4f(radiance, 1.0);
}
`;
}

// ---- Kernels ------------------------------------------------------------------------------------

/** Entry points of the cache's own maintenance kernels. */
export const HASH_GRID_ENTRY_POINT = {
    clearCounters: 'ClearCounters',
    purgeTiles: 'PurgeTiles',
} as const;

/** `PurgeTiles`' threads per workgroup (`[numthreads(64, 1, 1)]`, gi1.comp:1714). */
export const PURGE_TILES_WORKGROUP_SIZE = 64;

/** The group the maintenance kernels put the hash grid in. */
export const HASH_GRID_KERNEL_GROUP = 1;

/**
 * `ClearCounters` (gi1.comp:162-185), the hash grid's half: this frame's
 * packed tile count and the per-frame counters. Capsaicin's kernel clears the
 * screen probes' and reflections' counters in the same thread; the screen
 * cache has its own. The overflow counter is ours (register D23).
 */
export function clearCountersWgsl(layout: HashGridCacheBufferLayout): string {
    return /* wgsl */ `
${hashGridCacheWgsl(HASH_GRID_KERNEL_GROUP, layout)}
@compute @workgroup_size(1)
fn ${HASH_GRID_ENTRY_POINT.clearCounters}() {
    atomicStore(&g_HashGridCache_TileWords[HashGridCache_PackedTileCountWord()], 0u);
    atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_UPDATETILECOUNTBUFFER], 0u);
    atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0], 0u);
    atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER], 0u);
    // USE_MULTI_BOUNCE, on by default (gi1.h:55)
    atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1], 0u);
    atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_RESOLVECOUNTBUFFER], 0u);
    atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_BUCKETOVERFLOWCOUNT], 0u);
}
`;
}

/**
 * `PurgeTiles` (gi1.comp:1712-1758): every tile in last frame's packed list
 * that went `kHashGridCache_TileDecay` frames untouched is freed (its slot's
 * checksum zeroed); the rest join this frame's list. Dispatched indirectly
 * from last frame's count (gi1.cpp:2238-2253), so it needs the platform's
 * `indirectModule` (`giIndirectItemIndex`) before it.
 */
export function purgeTilesWgsl(layout: HashGridCacheBufferLayout): string {
    return /* wgsl */ `
${hashGridCacheWgsl(HASH_GRID_KERNEL_GROUP, layout)}
@compute @workgroup_size(${PURGE_TILES_WORKGROUP_SIZE})
fn ${HASH_GRID_ENTRY_POINT.purgeTiles}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = giIndirectItemIndex(workgroup, workgroups, local_index, ${PURGE_TILES_WORKGROUP_SIZE}u);
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HashGridCache_PreviousPackedTileCountWord()])) {
        return; // out of bounds
    }

    // We opt for calculating the decay of each tile from the
    // current frame index, as it saves many writes per frame
    // and is quite a bit faster than having to decrement all
    // cell's decay values.
    let tile_index = atomicLoad(&g_HashGridCache_TileWords[HashGridCache_PreviousPackedTileIndexWord(did)]);
    var tile_decay = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_DECAYTILEBUFFER + tile_index]);

    if (g_HashGridCacheConstants.frame_index < tile_decay) {   // account for integer wraparound case
        tile_decay = ((0xFFFFFFFFu - tile_decay) + g_HashGridCacheConstants.frame_index + 1u);
    } else {
        tile_decay = (g_HashGridCacheConstants.frame_index - tile_decay);
    }

    if (tile_decay >= kHashGridCache_TileDecay) {
        atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_HASHBUFFER + tile_index], 0u);
        return; // kill the tile
    }

    let packed_tile_index = atomicAdd(&g_HashGridCache_TileWords[HashGridCache_PackedTileCountWord()], 1u);
    atomicStore(&g_HashGridCache_TileWords[HashGridCache_PackedTileIndexWord(packed_tile_index)], tile_index);
}
`;
}
