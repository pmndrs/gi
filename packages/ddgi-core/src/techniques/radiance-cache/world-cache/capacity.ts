/**
 * The hash-grid world cache's options, the capacity Capsaicin derives from
 * them, and how we pack its arrays into a few storage buffers (register D6).
 *
 * Capsaicin v1.3 sizes the cache in `GI1::HashGridCache::ensureMemoryIsAllocated`
 * (`gi1.cpp:438-547`) from `GI1::RenderOptions` (`gi1.h:58-65`). The defaults
 * give 2^14 buckets × 16 tiles = 262,144 tiles, each a slab of 8 × 8 cells
 * with three coarser levels (4 × 4, 2 × 2, 1 × 1): 85 cells, so 22,282,240
 * cells. Every cell has a direct and an indirect value (a `uint2` of four
 * halves each) and two update scratch arrays (four `uint`s each): about 1 GiB
 * (`research/gi-1.0/source-map.md` §4.2; docs/design/radiance-cache.md,
 * "World-cache memory").
 *
 * Capsaicin keeps about 30 separate buffers in unbounded arrays
 * (`hash_grid_cache.hlsl:52-86`); WebGPU limits the storage buffers a shader
 * stage binds (register D6), so the arrays are packed into three buffers at
 * fixed offsets, one per element type and use:
 *
 * - **tile words** (`u32`, atomics): the cache's counters, then per tile the
 *   hash (checksum) slot, the decay stamp, the two ping-pong packed tile
 *   lists and the frame it was last put on the update list (register D32).
 *   A few MiB.
 * - **cell values** (`vec2u`): `ValueBuffer` then `ValueIndirectBuffer`.
 * - **update scratch** (`u32`, atomics): `UpdateCellValueBuffer` then
 *   `UpdateCellValueIndirectBuffer`, four words per cell each.
 *
 * The per-ray lists Capsaicin sizes from the screen (visibility, query and
 * resolve lists, `gi1.cpp:637-695`) are in a fourth buffer, made for the
 * screen's probe-ray count (`hash-grid-population-wgsl.ts`).
 */

import { GILimitError } from '../../../platform/contract/errors.ts';

/**
 * The hash grid's options, named as `RadianceCacheSettings` names them
 * (Capsaicin's `gi1_hash_grid_cache_*` options without the `gi1_` prefix),
 * so the technique's settings object can be passed as it is.
 *
 * The capacity fields (`NumBuckets`, `NumTilesPerBucket`, `TileCellRatio`)
 * take effect at `prepare`; the others are read every frame.
 */
export interface HashGridCacheOptions {
    /** `gi1_hash_grid_cache_cell_size` (gi1.h:58, default 32): a cell's size, in pixels of projected size at its distance. */
    hashGridCacheCellSize: number;
    /** `gi1_hash_grid_cache_min_cell_size` (gi1.h:59, default 0.1): the smallest cell, in scene units (metres). */
    hashGridCacheMinCellSize: number;
    /** `gi1_hash_grid_cache_tile_cell_ratio` (gi1.h:60, default 8): cells across a tile's finest level (8 × 8). */
    hashGridCacheTileCellRatio: number;
    /** `gi1_hash_grid_cache_num_buckets` (gi1.h:61, default 14): log2 of the bucket count (16,384). */
    hashGridCacheNumBuckets: number;
    /** `gi1_hash_grid_cache_num_tiles_per_bucket` (gi1.h:62, default 4): log2 of the tile slots per bucket (16). */
    hashGridCacheNumTilesPerBucket: number;
    /** `gi1_hash_grid_cache_max_sample_count` (gi1.h:63, default 16): a cell's temporal sample cap; below it the resolve reads a coarser level. */
    hashGridCacheMaxSampleCount: number;
    /**
     * Lever L1 (TASKS P6.2, id Tech 8's "shade cells, not hits"): `'per-hit'`
     * lights every hit (GI-1.2, the default); `'budgeted'` lights at most
     * `cellShadingBudget` cells a frame, one light sample each, chosen by
     * priority (`cell-shading-wgsl.ts`). Read every frame; switching doesn't
     * empty the cache (a cell's value is the same estimate either way).
     */
    cellShading: 'per-hit' | 'budgeted';
    /** Cells lit a frame when `cellShading` is `'budgeted'`: a whole number, 0 or more. Read every frame. */
    cellShadingBudget: number;
    /**
     * `gi1_use_multibounce` (gi1.h:55, default true): GI-1.2's second bounce.
     * With it, `PopulateMultibounceCells`, `GenerateMultibounceReservoirs` and
     * `UpdateMultibounceCells` run and the indirect values are read; without
     * it the cache is GI-1.0's single bounce (register D35). Changing it
     * empties the cache, as Capsaicin's `needs_hash_grid_clear` does
     * (gi1.cpp:1740-1745).
     */
    useMultibounce: boolean;
    /** `gi1_hash_grid_cache_discard_multibounce_ray_probability` (gi1.h:64, default 0.7): the share of first-bounce hits that trace no second-bounce ray. */
    hashGridCacheDiscardMultibounceRayProbability: number;
    /** `gi1_hash_grid_cache_max_multibounce_sample_count` (gi1.h:65, default 16): the same cap for the indirect values. */
    hashGridCacheMaxMultibounceSampleCount: number;
    /**
     * `gi1_use_direct_lighting` (gi1.h:53, default true), GI-1's own option,
     * which `GenerateReservoirs` reads: with it, a hit on an emissive front
     * face gives the probe ray its emission and bypasses the cache.
     */
    useDirectLighting: boolean;
}

/** Capsaicin's defaults (gi1.h:58-65). */
export const DEFAULT_HASH_GRID_CACHE_OPTIONS: Readonly<HashGridCacheOptions> = Object.freeze({
    hashGridCacheCellSize: 32,
    hashGridCacheMinCellSize: 0.1,
    hashGridCacheTileCellRatio: 8,
    hashGridCacheNumBuckets: 14,
    hashGridCacheNumTilesPerBucket: 4,
    hashGridCacheMaxSampleCount: 16,
    cellShading: 'per-hit',
    cellShadingBudget: 20000,
    useMultibounce: true,
    hashGridCacheDiscardMultibounceRayProbability: 0.7,
    hashGridCacheMaxMultibounceSampleCount: 16,
    useDirectLighting: true,
});

/** Mip levels in a tile: Capsaicin's constants have four, and it asserts the chain ends there (gi1.cpp:448). */
export const TILE_MIP_LEVELS = 4;

/**
 * The capacity Capsaicin derives from the options (`gi1.cpp:442-461`), with
 * its names. These are also the fields of `HashGridCacheConstants`
 * (`gi1_shared.h:59-92`) that don't change from frame to frame.
 */
export interface HashGridCacheCapacity {
    readonly numBuckets: number;
    readonly numTilesPerBucket: number;
    /** Cells across a tile at each level: 8, 4, 2, 1 by default (`size_tile_mip0..3`). */
    readonly sizeTileMip: readonly [number, number, number, number];
    /** Cells in a tile at each level: 64, 16, 4, 1 (`num_cells_per_tile_mip0..3`). */
    readonly numCellsPerTileMip: readonly [number, number, number, number];
    /** All levels: 85 (`num_cells_per_tile`). */
    readonly numCellsPerTile: number;
    /** Where each level starts inside a tile's cells: 0, 64, 80, 84 (`first_cell_offset_tile_mip0..3`). */
    readonly firstCellOffsetTileMip: readonly [number, number, number, number];
    /** `num_tiles_per_bucket × num_buckets`: 262,144. */
    readonly numTiles: number;
    /** `num_cells_per_tile × num_tiles`: 22,282,240. */
    readonly numCells: number;
}

/** Throws a RangeError unless `value` is a whole number in [min, max]. */
function checkWhole(name: string, value: number, min: number, max: number): void {
    if (!Number.isInteger(value) || value < min || value > max) {
        throw new RangeError(`@pmndrs/gi: the world cache's ${name} must be a whole number from ${min} to ${max}, got ${value}`);
    }
}

/** The capacity for `options` (`gi1.cpp:442-461`). */
export function hashGridCacheCapacity(options: Pick<HashGridCacheOptions, 'hashGridCacheNumBuckets' | 'hashGridCacheNumTilesPerBucket' | 'hashGridCacheTileCellRatio'>): HashGridCacheCapacity {
    checkWhole('hashGridCacheNumBuckets (log2)', options.hashGridCacheNumBuckets, 0, 24);
    checkWhole('hashGridCacheNumTilesPerBucket (log2)', options.hashGridCacheNumTilesPerBucket, 0, 8);
    // Capsaicin asserts the mip chain stops after four levels: (size_tile_mip3 >> 1) == 0 (gi1.cpp:448),
    // so the ratio is below 16; and a level with no cells would break the cell index.
    checkWhole('hashGridCacheTileCellRatio', options.hashGridCacheTileCellRatio, 1, 15);
    const numBuckets = 1 << options.hashGridCacheNumBuckets;
    const numTilesPerBucket = 1 << options.hashGridCacheNumTilesPerBucket;
    const size0 = options.hashGridCacheTileCellRatio;
    const sizeTileMip = [size0, size0 >> 1, size0 >> 2, size0 >> 3] as const;
    const numCellsPerTileMip = [sizeTileMip[0] ** 2, sizeTileMip[1] ** 2, sizeTileMip[2] ** 2, sizeTileMip[3] ** 2] as const;
    const numCellsPerTile = numCellsPerTileMip[0] + numCellsPerTileMip[1] + numCellsPerTileMip[2] + numCellsPerTileMip[3];
    const firstCellOffsetTileMip = [
        0,
        numCellsPerTileMip[0],
        numCellsPerTileMip[0] + numCellsPerTileMip[1],
        numCellsPerTileMip[0] + numCellsPerTileMip[1] + numCellsPerTileMip[2],
    ] as const;
    const numTiles = numTilesPerBucket * numBuckets;
    const numCells = numCellsPerTile * numTiles;
    // Cell indices are u32 on the GPU, and the scratch holds four words per cell.
    if (numCells * 4 * 2 > 0xffffffff) throw new RangeError(`@pmndrs/gi: the world cache's ${numCells} cells are too many to index with 32 bits`);
    return { numBuckets, numTilesPerBucket, sizeTileMip, numCellsPerTileMip, numCellsPerTile, firstCellOffsetTileMip, numTiles, numCells };
}

// ---- The packed buffers (register D6) ----------------------------------------------------------

/** Bytes in a `u32`. */
const WORD_BYTES = 4;

/**
 * Words of the counter block at the start of the tile words. Capsaicin has
 * one single-`uint` buffer per counter (`gi1_shared.h:100-127`); here each is
 * a word. The block is 64 words (256 bytes) so the per-tile arrays after it
 * start on WebGPU's storage-binding alignment, should a pass ever bind one
 * alone.
 */
export const COUNTER_WORDS = 64;

/**
 * Word offsets of the counters in the tile words, named after Capsaicin's
 * buffers (`HashGridBufferNamesUint`, gi1_shared.h:100-127). The two packed
 * tile counts are a ping-pong pair, selected by `buffer_ping_pong`
 * (`hash_grid_cache.hlsl:75-76`).
 */
export const HASH_GRID_COUNTER = {
    packedTileCount0: 0,
    packedTileCount1: 1,
    updateTileCount: 2,
    visibilityCount0: 3,
    visibilityCount1: 4,
    visibilityRayCount: 5,
    resolveCount: 6,
    /**
     * Inserts refused this frame because every slot of their bucket held
     * another tile (`hash_grid_cache.hlsl:268-275`). Capsaicin counts them per
     * bucket, and only with `DEBUG_HASH_STATS`; ours is one counter, always
     * on, cleared with the others each frame (register D23).
     */
    bucketOverflowCount: 7,
    /**
     * The frame's probe-ray count, copied from the screen cache's buffer at
     * the start of the cache's work, where the indirect arguments of the
     * insert pass (made once, at `prepare`) can read it.
     */
    probeRayCount: 8,
    /**
     * Lever L1 (`cellShading: 'budgeted'`, cell-shading-wgsl.ts): the cells
     * hit this frame, each listed once as a candidate for lighting. Cleared
     * by the cache's encoder at the start of a budgeted frame.
     */
    cellCandidateCount: 9,
    /** Lever L1: the candidates lit this frame, min(candidates, budget), written by `PrioritizeCells`. */
    cellShadedCount: 10,
    /**
     * Lever L1: frames between a settled cell's turns, for the next frame
     * (`PrioritizeCells` writes it from this frame's counts). Kept across
     * frames; 0 (an empty cache) reads as 1.
     */
    cellSchedulePeriod: 11,
    /** Lever L1: the first of `CELL_SHADING_BINS` words, the candidates per priority bin (then each bin's first slot). */
    cellPriorityBins: 32,
} as const;

/** Lever L1's priority classes: new or changed, still converging, its turn, the rest (cell-shading-wgsl.ts). */
export const CELL_SHADING_CLASSES = 4;
/** Random sub-bins per class, so the class the budget cuts through is cut at random places. */
export const CELL_SHADING_SUB_BINS = 8;
/** Priority bins: class × sub-bin, in the counter block from `HASH_GRID_COUNTER.cellPriorityBins` (32 words, its end). */
export const CELL_SHADING_BINS = CELL_SHADING_CLASSES * CELL_SHADING_SUB_BINS;

/** Where each array lives: word (or element) offsets inside its packed buffer, and the buffers' sizes. */
export interface HashGridCacheBufferLayout {
    /** Tile words: counters, then hash, decay, packed tile index 0 and 1, `numTiles` words each. */
    readonly tileWords: {
        readonly hashBuffer: number;
        readonly decayTileBuffer: number;
        readonly packedTileIndexBuffer0: number;
        readonly packedTileIndexBuffer1: number;
        /** Ours (register D32): per tile, the frame (plus one) it was last put on the update list. */
        readonly updateListedTileBuffer: number;
        readonly bytes: number;
    };
    /** Cell values, in `vec2u` elements: `ValueBuffer` at 0, `ValueIndirectBuffer` at `numCells`. */
    readonly cellValues: {
        readonly valueBuffer: number;
        readonly valueIndirectBuffer: number;
        readonly bytes: number;
    };
    /** Update scratch, in words: `UpdateCellValueBuffer` at 0, `UpdateCellValueIndirectBuffer` at `4 × numCells`. */
    readonly updateScratch: {
        readonly updateCellValueBuffer: number;
        readonly updateCellValueIndirectBuffer: number;
        readonly bytes: number;
    };
    /** All three. */
    readonly totalBytes: number;
}

/** Bytes of a cell value: a `uint2` holding four halves, rgb radiance and a sample count (`hash_grid_cache.hlsl:356-365`). */
export const CELL_VALUE_BYTES = 8;
/** Words of a cell's update scratch: quantised rgb and a sample count (`HashGridCache_QuantizeRadiance`, hash_grid_cache.hlsl:301-306; `num_cells << 2`, gi1.cpp:527). */
export const UPDATE_SCRATCH_WORDS_PER_CELL = 4;

export function hashGridCacheBufferLayout(capacity: HashGridCacheCapacity): HashGridCacheBufferLayout {
    const { numTiles, numCells } = capacity;
    const hashBuffer = COUNTER_WORDS;
    const decayTileBuffer = hashBuffer + numTiles;
    const packedTileIndexBuffer0 = decayTileBuffer + numTiles;
    const packedTileIndexBuffer1 = packedTileIndexBuffer0 + numTiles;
    const updateListedTileBuffer = packedTileIndexBuffer1 + numTiles;
    const tileWordsBytes = (updateListedTileBuffer + numTiles) * WORD_BYTES;
    const cellValuesBytes = 2 * numCells * CELL_VALUE_BYTES;
    const scratchWordsPerArray = UPDATE_SCRATCH_WORDS_PER_CELL * numCells;
    const updateScratchBytes = 2 * scratchWordsPerArray * WORD_BYTES;
    return {
        tileWords: { hashBuffer, decayTileBuffer, packedTileIndexBuffer0, packedTileIndexBuffer1, updateListedTileBuffer, bytes: tileWordsBytes },
        cellValues: { valueBuffer: 0, valueIndirectBuffer: numCells, bytes: cellValuesBytes },
        updateScratch: { updateCellValueBuffer: 0, updateCellValueIndirectBuffer: scratchWordsPerArray, bytes: updateScratchBytes },
        totalBytes: tileWordsBytes + cellValuesBytes + updateScratchBytes,
    };
}

/**
 * Throws `GILimitError` naming the first limit the packed buffers pass: each
 * is bound whole as one storage binding, so it must fit both the binding
 * limit and the buffer limit. A device made with WebGPU's default limits
 * (128 MiB bindings, 256 MiB buffers; three's own device, register D14)
 * fails on `maxStorageBufferBindingSize`.
 */
export function checkHashGridCacheLimits(layout: HashGridCacheBufferLayout, limits: Pick<GPUSupportedLimits, 'maxStorageBufferBindingSize' | 'maxBufferSize'>): void {
    const largest = Math.max(layout.tileWords.bytes, layout.cellValues.bytes, layout.updateScratch.bytes);
    if (largest > limits.maxStorageBufferBindingSize) throw new GILimitError('maxStorageBufferBindingSize', largest, limits.maxStorageBufferBindingSize);
    if (largest > limits.maxBufferSize) throw new GILimitError('maxBufferSize', largest, limits.maxBufferSize);
}
