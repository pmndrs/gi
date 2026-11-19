/**
 * Lever L1's pieces that the GI-1.2 population passes share with its own
 * kernels (`cell-shading-wgsl.ts`): where its counters are, and how a hit
 * offers itself as its cell's one sample this frame. Kept apart so the
 * population passes (`hash-grid-population-wgsl.ts`) and the lever's kernels
 * (which include the population passes' head) don't import each other.
 *
 * Needs, before it: the hash grid (`hashGridCacheWgsl`) and Capsaicin's random
 * numbers (`CAPSAICIN_WGSL`: `rcMakeRandom`, `rcRandInt`).
 */

import { CELL_SHADING_BINS, CELL_SHADING_SUB_BINS, HASH_GRID_COUNTER } from './capacity.ts';

/**
 * Salts for lever L1's random streams, so none of them shares its numbers
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

export const CELL_SHADING_COMMON_WGSL = /* wgsl */ `
// ---- Lever L1: cells lit on a budget (cell-shading-wgsl.ts) ----
const HASHGRIDCACHE_CELLCANDIDATECOUNT: u32 = ${HASH_GRID_COUNTER.cellCandidateCount}u;
const HASHGRIDCACHE_CELLSHADEDCOUNT: u32 = ${HASH_GRID_COUNTER.cellShadedCount}u;
const HASHGRIDCACHE_CELLSCHEDULEPERIOD: u32 = ${HASH_GRID_COUNTER.cellSchedulePeriod}u;
const HASHGRIDCACHE_CELLPRIORITYBINS: u32 = ${HASH_GRID_COUNTER.cellPriorityBins}u;
const HASHGRIDCACHE_CELL_BINS: u32 = ${CELL_SHADING_BINS}u;
const HASHGRIDCACHE_CELL_SUB_BINS: u32 = ${CELL_SHADING_SUB_BINS}u;
const HASHGRIDCACHE_CELL_SELECT_SALT: u32 = ${CELL_SHADING_SALT.select}u;

// The key a visibility entry offers for its cell this frame: random, never 0 (the scratch's empty
// value). Recomputed by ClaimCells to find the entry that offered the largest.
fn HashGridCache_CellSampleKey(visibility_index: u32) -> u32 {
    var random = rcMakeRandom(visibility_index ^ HASHGRIDCACHE_CELL_SELECT_SALT, g_HashGridCacheConstants.frame_index);
    return rcRandInt(&random) | 1u;
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
