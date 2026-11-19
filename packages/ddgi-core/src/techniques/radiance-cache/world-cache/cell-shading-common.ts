/**
 * Lever E8's pieces that the GI-1.2 population passes share with its own
 * kernels (`cell-shading-wgsl.ts`): where its counters are, and how a hit
 * offers itself as its cell's one sample this frame. Kept apart so the
 * population passes (`hash-grid-population-wgsl.ts`) and the lever's kernels
 * (which include the population passes' head) don't import each other.
 *
 * Needs, before it: the hash grid (`hashGridCacheWgsl`) and Capsaicin's hash
 * (`CAPSAICIN_WGSL`: `rcPcgHash`).
 */

import { CELL_SHADING_BINS, CELL_SHADING_SUB_BINS, HASH_GRID_COUNTER } from './capacity.ts';

/**
 * Salts for lever E8's random streams, so none of them shares its numbers
 * with `GenerateReservoirs`' stream (seeded with the visibility index) or with
 * each other. Each is above any visibility index (fewer than 2^23) and any
 * cell index, so a salted seed never equals another stream's plain seed.
 */
export const CELL_SHADING_SALT = {
    /** The key an entry offers for its cell. */
    select: 0x5bd1e995,
    /** The light sample a lit cell draws. */
    shade: 0x27d4eb2d,
    /** A candidate's random sub-bin. */
    subBin: 0x68e31da4,
} as const;

/** A cell count that marks a changed cell (exact as a half; no sum of whole counts reaches it). */
export const CELL_CHANGED_COUNT = 1 + 2 ** -10;
/** A cell count for a cell lit with no light to draw: the smallest normal half. */
export const CELL_DARK_COUNT = 2 ** -14;

export const CELL_SHADING_COMMON_WGSL = /* wgsl */ `
// ---- Lever E8: cells lit on a budget (cell-shading-wgsl.ts) ----
const HASHGRIDCACHE_CELLCANDIDATECOUNT: u32 = ${HASH_GRID_COUNTER.cellCandidateCount}u;
const HASHGRIDCACHE_CELLSHADEDCOUNT: u32 = ${HASH_GRID_COUNTER.cellShadedCount}u;
const HASHGRIDCACHE_CELLSCHEDULEPERIOD: u32 = ${HASH_GRID_COUNTER.cellSchedulePeriod}u;
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
`;
