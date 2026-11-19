/**
 * What lever E8 (cells lit on a budget, `cell-shading-wgsl.ts`) and
 * experiment E9 (per-hit change reset) share with the GI-1.2 population
 * passes: E8's counters and its offer of a hit as its cell's sample, and the
 * scene's change events with the per-cell memory of when each cell last
 * forgot its history. Kept apart so the population passes
 * (`hash-grid-population-wgsl.ts`) and E8's kernels (which include the
 * population passes' head) don't import each other.
 *
 * `CELL_CHANGES_WGSL` is part of the population passes' ray-data group
 * (group 2, bindings 4-6), so every population pass can read it.
 * `CELL_SHADING_COMMON_WGSL` needs, before it, the hash grid
 * (`hashGridCacheWgsl`), the ray data and Capsaicin's hash (`CAPSAICIN_WGSL`:
 * `rcPcgHash`).
 */

import { CELL_SHADING_BINS, CELL_SHADING_SUB_BINS, HASH_GRID_COUNTER } from './capacity.ts';

/**
 * Salts for lever E8's random streams, so none of them shares its numbers
 * with `GenerateReservoirs`' stream (seeded with the visibility index) or with
 * each other. Each is above any visibility index and any cell index this
 * cache can have, so a salted seed never equals another stream's plain seed.
 */
export const CELL_SHADING_SALT = {
    /** The key an entry offers for its cell (hashed, not a stream). */
    select: 0x5bd1e995,
    /** The light sample a lit cell draws. */
    shade: 0x27d4eb2d,
    /** A candidate's random sub-bin. */
    subBin: 0x68e31da4,
    /** Where the budget cuts a sub-bin: which of its cells are lit. */
    thin: 0x1b873593,
} as const;

/**
 * The most change events passed to the GPU: the technique's tracker keeps
 * at most one fewer with a box, plus the one with no bounds
 * (`ChangeEventTracker`, platform contract).
 */
export const CELL_SHADING_MAX_CHANGE_EVENTS = 16;

/**
 * Experiment E9: the top bit of a cell's direct scratch count, set by
 * `GenerateReservoirs` and `GenerateMultibounceReservoirs` when a hit's cell
 * is due to forget its history; `UpdateTiles` takes it off. Per-hit counts
 * never reach it.
 */
export const CELL_CHANGE_RESET_BIT = 0x80000000;

/** Bindings of the change data in the population passes' ray-data group (group 2). */
export const CELL_CHANGES_BINDING = {
    /** `HashGridCacheCellShading`: E8's budget and the change events. Uniform. */
    uniform: 4,
    /** Per finest-level cell, the frame (low 16 bits) it last forgot its history: two to a word. Storage. */
    resetFrames: 5,
    /** E8's candidate and lit lists and its priority bins (`cellShadingListLayout`). Storage. */
    cellLists: 6,
} as const;

/**
 * `HashGridCacheCellShading` and the change test with its per-cell memory.
 * A cell near an event forgets its history at most once per epoch of the
 * event (`ChangeEvent.epoch`): due when the frame it last forgot is before
 * the epoch's start. 16 bits of frame are kept per cell (two cells a word):
 * an epoch starts at most `quietFrames + rearmFrames` frames back, so the
 * difference is read modulo 2^16 without ambiguity (a cell idle for exactly
 * a multiple of 65,536 frames reads as reset; it forgets at its next epoch).
 */
export const CELL_CHANGES_WGSL = /* wgsl */ `
// A change event (ChangeEventTracker, platform contract): its box (the whole world for a change
// with no bounds) and the start of its current epoch, plus one (0: no event).
struct HashGridCacheChangeEvent {
    lower: vec3f,
    epoch_plus_one: u32,
    upper: vec3f,
    padding0: u32,
}

// The cache's per-frame cell values (hash-grid-cache.ts writes them).
struct HashGridCacheCellShading {
    budget: u32,                    // E8: cells lit this frame at most
    event_count: u32,
    padding0: u32,
    padding1: u32,
    events: array<HashGridCacheChangeEvent, ${CELL_SHADING_MAX_CHANGE_EVENTS}>,
}

@group(2) @binding(${CELL_CHANGES_BINDING.uniform}) var<uniform> g_HashGridCache_CellShading: HashGridCacheCellShading;
@group(2) @binding(${CELL_CHANGES_BINDING.resetFrames}) var<storage, read_write> g_HashGridCache_ResetFrames: array<atomic<u32>>;
@group(2) @binding(${CELL_CHANGES_BINDING.cellLists}) var<storage, read_write> g_HashGridCache_CellLists: array<atomic<u32>>;

// The latest epoch (plus one) among the events within 'margin' of 'position'; 0 for none.
fn HashGridCache_ChangeEpoch(position: vec3f, margin: f32) -> u32 {
    var latest = 0u;
    let count = min(g_HashGridCache_CellShading.event_count, ${CELL_SHADING_MAX_CHANGE_EVENTS}u);
    for (var index = 0u; index < count; index++) {
        let event = g_HashGridCache_CellShading.events[index];
        if (all(position >= event.lower - vec3f(margin)) && all(position <= event.upper + vec3f(margin))) {
            latest = max(latest, event.epoch_plus_one);
        }
    }
    return latest;
}

// Where a finest-level cell's last-reset frame is: its word and the shift of its 16 bits.
fn HashGridCache_ResetSlot(cell_index: u32) -> vec2u {
    let tile_index = cell_index / g_HashGridCacheConstants.num_cells_per_tile;
    let linear = tile_index * g_HashGridCacheConstants.num_cells_per_tile_mip0 + (cell_index % g_HashGridCacheConstants.num_cells_per_tile);
    return vec2u(linear >> 1u, (linear & 1u) * 16u);
}

// Whether the cell at 'position' is due to forget its history (near an event whose current epoch
// began after it last forgot), and if so, records that it forgets now. Several threads may find
// it due in one frame; each records the same frame, and each may act on it.
fn HashGridCache_ClaimChangeReset(cell_index: u32, position: vec3f) -> bool {
    let epoch_plus_one = HashGridCache_ChangeEpoch(position, HashGridCache_GetCellSize(position));
    if (epoch_plus_one == 0u) {
        return false;
    }
    let frame = g_HashGridCacheConstants.frame_index;
    let since_epoch = frame - (epoch_plus_one - 1u);
    let slot = HashGridCache_ResetSlot(cell_index);
    let stored = (atomicLoad(&g_HashGridCache_ResetFrames[slot.x]) >> slot.y) & 0xffffu;
    let since_reset = (frame - stored) & 0xffffu;
    if (since_reset <= since_epoch) {
        return false; // already forgot within this epoch
    }
    atomicAnd(&g_HashGridCache_ResetFrames[slot.x], ~(0xffffu << slot.y));
    atomicOr(&g_HashGridCache_ResetFrames[slot.x], (frame & 0xffffu) << slot.y);
    return true;
}
`;

/** A cell count that marks a changed cell (exact as a half; no sum of whole counts reaches it). */
export const CELL_CHANGED_COUNT = 1 + 2 ** -10;
/** A cell count for a cell lit with no light to draw: the smallest normal half. */
export const CELL_DARK_COUNT = 2 ** -14;

export const CELL_SHADING_COMMON_WGSL = /* wgsl */ `
// ---- Lever E8: cells lit on a budget (cell-shading-wgsl.ts) ----
const HASHGRIDCACHE_CELLCANDIDATECOUNT: u32 = ${HASH_GRID_COUNTER.cellCandidateCount}u;
const HASHGRIDCACHE_CELLSHADEDCOUNT: u32 = ${HASH_GRID_COUNTER.cellShadedCount}u;
const HASHGRIDCACHE_CELLSCHEDULEPERIOD: u32 = ${HASH_GRID_COUNTER.cellSchedulePeriod}u;
const HASHGRIDCACHE_CELLSHADEDTARGET: u32 = ${HASH_GRID_COUNTER.cellShadedTarget}u;
const HASHGRIDCACHE_CELL_BINS: u32 = ${CELL_SHADING_BINS}u;
const HASHGRIDCACHE_CELL_SUB_BINS: u32 = ${CELL_SHADING_SUB_BINS}u;
const HASHGRIDCACHE_CELL_SELECT_SALT: u32 = ${CELL_SHADING_SALT.select}u;
// A cell's sample count that marks it changed: its next sample replaces it (its mean is kept until
// then). Not a whole number, so no count reaches it by adding samples; exact as a half.
const HASHGRIDCACHE_CELL_CHANGED_COUNT: f32 = ${CELL_CHANGED_COUNT};
// A cell's count when it was lit and had no light to draw (per hit, such a hit adds no sample): not
// new, so not lit again first, yet light enough to leave its tile's coarser levels as they are.
// The smallest normal half.
const HASHGRIDCACHE_CELL_DARK_COUNT: f32 = ${CELL_DARK_COUNT};

// The key a visibility entry offers for its cell this frame: random, never 0 (the scratch's empty
// value). Recomputed by ClaimCells to find the entry that offered the largest. Two rounds of PCG's
// hash, the frame's hashed in between: a PCG stream's first number, seeded per entry, made the
// largest key lean toward lower entries (0.486 of the way through a cell's entries on the GPU).
fn HashGridCache_CellSampleKey(visibility_index: u32) -> u32 {
    return rcPcgHash(rcPcgHash(visibility_index ^ HASHGRIDCACHE_CELL_SELECT_SALT) + rcPcgHash(g_HashGridCacheConstants.frame_index)) | 1u;
}

// Offers this entry as its cell's sample this frame. The entry with the largest key wins, so the
// cell's sample is drawn uniformly from the entries that hit it, which is what makes its mean the
// mean per-hit lighting gives (every hit lit, each a sample). The key goes in the first word of
// the cell's direct update scratch, which is zero between frames (UpdateTiles clears what it reads)
// and which, with cells lit on the budget, nothing else writes until ClaimCells.
fn HashGridCache_OfferCellSample(visibility_index: u32, cell_index: u32) {
    atomicMax(&g_HashGridCache_UpdateScratch[HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index], HashGridCache_CellSampleKey(visibility_index));
}

// A ray from 'origin' to 'position' shorter than the cell at 'position': it reads no cell and is
// lit on its own (Capsaicin's bypass test, GenerateReservoirs gi1.comp:2555-2566).
fn HashGridCache_ShorterThanCell(origin: vec3f, position: vec3f) -> bool {
    return distance(origin, position) < HashGridCache_GetCellSize(position);
}

// Experiment E9: a hit near a change marks its cell (once per epoch), so UpdateTiles takes this
// frame's samples whole (with cellChangeResetHistory samples of the history beside them).
fn HashGridCache_MarkChangeReset(cell_index: u32, position: vec3f) {
    if (HashGridCache_ChangeResetPerHit() && HashGridCache_ClaimChangeReset(cell_index, position)) {
        atomicOr(&g_HashGridCache_UpdateScratch[HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index + 3u], HASHGRIDCACHE_CHANGE_RESET_BIT);
    }
}
`;

/** E8's lists in their own buffer, for a screen of `maxCombinedRayCount` visibility entries: word offsets. */
export interface CellShadingListLayout {
    /** Two words per candidate: its visibility entry with the changed bit (bit 31), and its bin. */
    readonly candidates: number;
    /** One word per lit cell: its visibility entry with the changed bit. */
    readonly lit: number;
    /** `CELL_SHADING_BINS` words of first slots, then as many of counts, then as many of cursors (the next slot). */
    readonly bins: number;
    readonly bytes: number;
}

export function cellShadingListLayout(maxCombinedRayCount: number): CellShadingListLayout {
    const candidates = 0;
    const lit = candidates + 2 * maxCombinedRayCount;
    const bins = lit + maxCombinedRayCount;
    return { candidates, lit, bins, bytes: (bins + 3 * CELL_SHADING_BINS) * 4 };
}
