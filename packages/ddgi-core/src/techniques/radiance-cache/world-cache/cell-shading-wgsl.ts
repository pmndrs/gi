/**
 * Lever E8 (TASKS P6.2): id Tech 8's "shade cells, not hits" (Sousa,
 * SIGGRAPH 2025, slides 17-20; research/gi-1.0/idtech8-performance.md, T1-T3)
 * on GI-1.2's hash grid, behind `cellShading: 'budgeted'`
 * (docs/design/radiance-cache.md, "Lever E8").
 *
 * Per hit (GI-1.2, the default), `GenerateReservoirs` and
 * `GenerateMultibounceReservoirs` draw a light for every probe-ray and
 * second-bounce hit and `PopulateCells` traces its shadow ray into the cell:
 * one shadow ray per hit. Budgeted, a hit only finds its cell and offers
 * itself as the cell's sample (`HashGridCache_OfferCellSample`, a random key
 * into the cell's scratch), and at most `cellShadingBudget` cells a frame are
 * lit, one light sample and shadow ray each, by these kernels, run after
 * `PopulateCells` (which then traces only the shadow rays of rays shorter than
 * their cell) and before `UpdateTiles`:
 *
 * 1. `ClaimCells` (over the visibility entries, first and second bounce): the
 *    entry whose key is the cell's largest claims the cell, so each cell hit
 *    this frame is listed once, its sample drawn uniformly from its hits. The
 *    claim puts the cell in a priority class and a random sub-bin, and
 *    counts the bin.
 * 2. `PrioritizeCells` (one workgroup, a thread per bin): the bins' counts
 *    become their first slots (a counting sort by priority over 256 bins, 4
 *    classes × 64 random sub-bins, a workgroup scan), the frame's lit target
 *    is min(candidates, budget), and the period of the settled cells' turns
 *    is set for the next frame.
 * 3. `ScatterCells` (over the candidates): a bin wholly under the target is
 *    lit, a bin wholly past it is not, and the one bin the target cuts is
 *    thinned by a hash of the cell (each lit with the share of the bin the
 *    target leaves), so which of its cells are lit doesn't follow the order
 *    the claims ran in. A lit cell goes on the lit list; one not lit clears
 *    the key from its scratch.
 * 4. `ShadeCells` (over the shade list): one light and shadow ray from the
 *    cell's sample hit, stored as the cell's one sample in its scratch, where
 *    `UpdateTiles` blends it in as it blends per-hit samples.
 *
 * The priority classes, highest first:
 * - **new or changed:** no sample yet, or the sample hit is near a change
 *   event whose current epoch began since the cell last forgot its history
 *   (`ChangeEventTracker` in the platform contract; the cell's last reset
 *   frame, `HashGridCache_ClaimChangeReset`): once per event, re-armed every
 *   8 frames while it goes on and once when it stops. A changed cell keeps its
 *   mean (nothing goes dark) but is marked by its count (`CELL_CHANGED_COUNT`),
 *   so whenever it is next lit, this frame or later, its new sample replaces
 *   its value; it stays in this class until then;
 * - **converging:** fewer samples than `max_sample_count`;
 * - **its turn:** settled cells, each once every `period` frames by a hash of
 *   the cell, the period being the settled cells over the budget they left
 *   last frame, so every settled cell hit each frame is relit at least that
 *   often;
 * - **the rest**, at random.
 *
 * A cell not lit keeps its value (`UpdateTiles` leaves a budgeted cell with no
 * sample alone), and its rays read that value: the cell's last estimate.
 *
 * Unbiasedness: the cell's sample hit is uniform over its hits this frame, and
 * its one light sample is an unbiased estimate of the light leaving that hit,
 * so its expectation is the mean per-hit lighting would add over those hits.
 * Which cells are lit depends on counts, hashes and random numbers, never on
 * a sample's value, and the running mean of unbiased samples is unbiased. The
 * cell's mean converges to per-hit's; it gets one sample a lighting instead of
 * one per hit, so its value is noisier and older. Per hit, a hit with no light
 * to draw adds a sample of no light (register D63,
 * `HashGridCache_AddSampleOfNoLight`); so here a lit cell whose sample hit has
 * no light to draw takes a sample of 0, and its mean is over all its hits as
 * per hit's is.
 */

import { CAPSAICIN_WGSL } from '../wgsl/capsaicin.ts';
import { CELL_SHADING_BINS, type HashGridCacheBufferLayout } from './capacity.ts';
import { CELL_SHADING_COMMON_WGSL, CELL_SHADING_MAX_CHANGE_EVENTS, CELL_SHADING_SALT } from './cell-shading-common.ts';
import { LIGHT_MASK_WGSL, populationHeadWgsl, PROBE_RAY_SURFACE_WGSL } from './hash-grid-population-wgsl.ts';

/** Bytes of `HashGridCacheCellShading`: four words, then 32 bytes an event. */
export const CELL_SHADING_UNIFORM_BYTES = 16 + 32 * CELL_SHADING_MAX_CHANGE_EVENTS;

/** A candidate's (and a lit cell's) first word: its visibility entry, with this bit when the cell changed. The second word of a candidate is its bin. */
export const CELL_CANDIDATE_CHANGED_BIT = 0x80000000;
export const CELL_CANDIDATE_ENTRY_MASK = 0x7fffffff;

export const CELL_SHADING_WORKGROUP_SIZE = {
    claimCells: 64,
    scatterCells: 64,
    /** As `PopulateCells`, which traces the same kind of shadow ray (gi1.comp:2110). */
    shadeCells: 32,
} as const;

export const CELL_SHADING_ENTRY_POINT = {
    claimFirstBounceCells: 'ClaimFirstBounceCells',
    claimSecondBounceCells: 'ClaimSecondBounceCells',
    prioritizeCells: 'PrioritizeCells',
    scatterCells: 'ScatterCells',
    shadeCells: 'ShadeCells',
} as const;

/** The class numbers, as `ClaimCells` assigns them (lower is lit first). */
export const CELL_PRIORITY_CLASS = { newOrChanged: 0, converging: 1, turn: 2, rest: 3 } as const;

const CELL_LIST_WGSL = /* wgsl */ `
const HASHGRIDCACHE_CELL_ENTRY_MASK: u32 = ${CELL_CANDIDATE_ENTRY_MASK}u;
const HASHGRIDCACHE_CELL_CHANGED_BIT: u32 = ${CELL_CANDIDATE_CHANGED_BIT}u;

fn HashGridCache_CellListWord(offset: u32, index: u32) -> u32 {
    return atomicLoad(&g_HashGridCache_CellLists[offset + index]);
}

fn HashGridCache_SetCellListWord(offset: u32, index: u32, value: u32) {
    atomicStore(&g_HashGridCache_CellLists[offset + index], value);
}
`;

/** The shared head of the lever's kernels. */
function cellShadingHeadWgsl(layout: HashGridCacheBufferLayout): string {
    return `${CAPSAICIN_WGSL}\n${populationHeadWgsl(layout)}\n${CELL_SHADING_COMMON_WGSL}\n${CELL_LIST_WGSL}`;
}

/**
 * `ClaimFirstBounceCells` and `ClaimSecondBounceCells`: the same claim over
 * the first-bounce entries (dispatched on `VisibilityCountBuffer0`) and the
 * second-bounce ones (on `VisibilityCountBuffer1`, numbered after them).
 */
export function claimCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = CELL_SHADING_WORKGROUP_SIZE.claimCells;
    return /* wgsl */ `
${cellShadingHeadWgsl(layout)}

const HASHGRIDCACHE_CELL_SUB_BIN_SALT: u32 = ${CELL_SHADING_SALT.subBin}u;

// An indirect value whose history is cut to one sample: its mean kept, its count 1 (a value with at
// most one sample is left as it is). The next sample then weighs as much as all the history.
fn HashGridCache_KeepOneSample(value_index: u32) {
    let value = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[value_index]);
    if (value.w > 1.0) {
        g_HashGridCache_CellValues[value_index] = HashGridCache_PackRadiance(vec4f(value.rgb / value.w, 1.0));
    }
}

// A changed cell's direct value: its mean kept (so nothing goes dark) with the changed count, so that
// whenever it is next lit its sample replaces it (ShadeCells). A cell with no samples (new) is
// emptied: it is lit again first.
fn HashGridCache_MarkChanged(cell_index: u32) {
    let value = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index]);
    var marked = vec4f(0.0);
    if (value.w >= 1.0) {
        marked = vec4f(value.rgb / value.w * HASHGRIDCACHE_CELL_CHANGED_COUNT, HASHGRIDCACHE_CELL_CHANGED_COUNT);
    }
    g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = HashGridCache_PackRadiance(marked);
}

// The claim of one visibility entry: whether it claimed its cell (flags), and then the cell's
// candidate word (a word can be 0: entry 0 in bin 0, so the claim is a flag of its own).
const HASHGRIDCACHE_CELL_CLAIMED: u32 = 1u;

struct HashGridCache_Claim {
    flags: u32,
    word: u32,
    bin: u32,
}

fn HashGridCache_ClaimCell(visibility_index: u32) -> HashGridCache_Claim {
    var none: HashGridCache_Claim;
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index);
    if (cell_index == kGI1_InvalidId) {
        return none; // a second-bounce ray that missed: no cell
    }

    // The entry that offered the cell's largest key claims it. Two entries with the same key (one
    // pair in billions) are told apart by the exchange: the first one there wins.
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;
    if (atomicLoad(&g_HashGridCache_UpdateScratch[scratch + 0u]) != HashGridCache_CellSampleKey(visibility_index)) {
        return none;
    }
    if (atomicExchange(&g_HashGridCache_UpdateScratch[scratch + 1u], 1u) != 0u) {
        return none;
    }

    let ray = HashGridCache_VisibilityRay(visibility_index);
    let position = ray.origin + ray.hitT * worldCacheProbeRayDirection(ray);
    let changed = HashGridCache_ClaimChangeReset(cell_index, position);
    let direct = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index]);

    if (changed) {
        HashGridCache_MarkChanged(cell_index);
        if (g_HashGridCacheConstants.use_multibounce != 0u) {
            HashGridCache_KeepOneSample(HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index);
        }
    }

    // New (no count), changed (now or on an earlier frame, not lit since), converging (whole counts
    // under the cap), settled (at the cap).
    var priority = ${CELL_PRIORITY_CLASS.rest}u;
    if (changed || direct.w <= 0.0 || direct.w == HASHGRIDCACHE_CELL_CHANGED_COUNT) {
        priority = ${CELL_PRIORITY_CLASS.newOrChanged}u;
    } else if (direct.w >= 1.0 && direct.w < g_HashGridCacheConstants.max_sample_count) {
        priority = ${CELL_PRIORITY_CLASS.converging}u;
    } else {
        let period = max(atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSCHEDULEPERIOD]), 1u);
        if ((rcPcgHash(cell_index) + g_HashGridCacheConstants.frame_index) % period == 0u) {
            priority = ${CELL_PRIORITY_CLASS.turn}u;
        }
    }
    var random = rcMakeRandom(cell_index ^ HASHGRIDCACHE_CELL_SUB_BIN_SALT, g_HashGridCacheConstants.frame_index);
    let bin = priority * HASHGRIDCACHE_CELL_SUB_BINS + rcRandIntBelow(&random, HASHGRIDCACHE_CELL_SUB_BINS);
    atomicAdd(&g_HashGridCache_CellLists[g_HashGridCache_RayLists.cell_priority_bins + HASHGRIDCACHE_CELL_BINS + bin], 1u);

    let changed_bit = select(0u, HASHGRIDCACHE_CELL_CHANGED_BIT, changed);
    return HashGridCache_Claim(HASHGRIDCACHE_CELL_CLAIMED, visibility_index | changed_bit, bin);
}

// The workgroup's claims go on the candidate list with one global atomic: each claim takes a place
// in the group's count, then one thread adds the count to the list's (on the GPU, one atomic per
// claim on one word was most of ClaimCells' 1 ms on Sponza). Every thread reaches the barriers.
var<workgroup> lds_ClaimCells_Count: atomic<u32>;
var<workgroup> lds_ClaimCells_First: u32;

fn HashGridCache_AppendClaims(claim: HashGridCache_Claim, local_index: u32) {
    if (local_index == 0u) {
        atomicStore(&lds_ClaimCells_Count, 0u);
    }
    workgroupBarrier();
    var place = 0u;
    if (claim.flags != 0u) {
        place = atomicAdd(&lds_ClaimCells_Count, 1u);
    }
    workgroupBarrier();
    if (local_index == 0u) {
        lds_ClaimCells_First = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLCANDIDATECOUNT], atomicLoad(&lds_ClaimCells_Count));
    }
    let first = workgroupUniformLoad(&lds_ClaimCells_First);
    if (claim.flags != 0u) {
        HashGridCache_SetCellListWord(g_HashGridCache_RayLists.cell_candidate_buffer, 2u * (first + place), claim.word);
        HashGridCache_SetCellListWord(g_HashGridCache_RayLists.cell_candidate_buffer, 2u * (first + place) + 1u, claim.bin);
    }
}

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.claimFirstBounceCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    var claim: HashGridCache_Claim;
    if (did < atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        claim = HashGridCache_ClaimCell(did);
    }
    HashGridCache_AppendClaims(claim, local_index);
}

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.claimSecondBounceCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    var claim: HashGridCache_Claim;
    if (did < atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1])) {
        claim = HashGridCache_ClaimCell(atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]) + did);
    }
    HashGridCache_AppendClaims(claim, local_index);
}
`;
}

/**
 * `PrioritizeCells`: a workgroup of a thread per bin turns the bins' counts
 * into their first slots (an exclusive prefix sum over 256 words; one thread
 * looping over them took about 0.25 ms), sets the frame's lit count
 * and the next frame's turn period. CPU twin `prioritizeCells` (cell-shading-cpu.ts).
 */
export function prioritizeCellsWgsl(layout: HashGridCacheBufferLayout): string {
    return /* wgsl */ `
${cellShadingHeadWgsl(layout)}

var<workgroup> lds_PrioritizeCells_Sums: array<u32, HASHGRIDCACHE_CELL_BINS>;

@compute @workgroup_size(${CELL_SHADING_BINS})
fn ${CELL_SHADING_ENTRY_POINT.prioritizeCells}(@builtin(local_invocation_index) bin: u32) {
    // An inclusive scan of the bins' counts in the group's memory (Hillis-Steele: 8 steps for 256).
    let count = HashGridCache_CellListWord(g_HashGridCache_RayLists.cell_priority_bins, HASHGRIDCACHE_CELL_BINS + bin);
    lds_PrioritizeCells_Sums[bin] = count;
    workgroupBarrier();
    for (var offset = 1u; offset < HASHGRIDCACHE_CELL_BINS; offset <<= 1u) {
        var before = 0u;
        if (bin >= offset) {
            before = lds_PrioritizeCells_Sums[bin - offset];
        }
        workgroupBarrier();
        lds_PrioritizeCells_Sums[bin] += before;
        workgroupBarrier();
    }
    // Each bin's first slot: the candidates in the bins before it.
    HashGridCache_SetCellListWord(g_HashGridCache_RayLists.cell_priority_bins, bin, lds_PrioritizeCells_Sums[bin] - count);
    // Each bin's cursor starts at its first slot (ScatterCells takes slots from it).
    HashGridCache_SetCellListWord(g_HashGridCache_RayLists.cell_priority_bins, 2u * HASHGRIDCACHE_CELL_BINS + bin, lds_PrioritizeCells_Sums[bin] - count);

    if (bin == 0u) {
        let candidates = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLCANDIDATECOUNT]);
        let budget = g_HashGridCache_CellShading.budget;
        atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSHADEDTARGET], min(candidates, budget));

        // The settled cells take what the urgent ones (new, changed, converging) leave of the budget;
        // a period of settled / left frames gives each its turn once in that many frames (the turns'
        // expected count is then at most what is left). With nothing left this frame, a turn comes
        // once in 'settled' frames.
        let urgent = lds_PrioritizeCells_Sums[${CELL_PRIORITY_CLASS.turn}u * HASHGRIDCACHE_CELL_SUB_BINS - 1u];
        let settled = candidates - urgent;
        let left = max(budget - min(budget, urgent), 1u);
        atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSCHEDULEPERIOD], max((settled + left - 1u) / left, 1u));
    }
}
`;
}

/** `ScatterCells`: each candidate takes its bin's next slot; under the budget it goes on the shade list, past it its scratch is cleared. */
export function scatterCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = CELL_SHADING_WORKGROUP_SIZE.scatterCells;
    return /* wgsl */ `
${cellShadingHeadWgsl(layout)}

const HASHGRIDCACHE_CELL_THIN_SALT: u32 = ${CELL_SHADING_SALT.thin}u;

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.scatterCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLCANDIDATECOUNT])) {
        return; // out of bounds
    }
    let candidate = HashGridCache_CellListWord(g_HashGridCache_RayLists.cell_candidate_buffer, 2u * did);
    let bin = HashGridCache_CellListWord(g_HashGridCache_RayLists.cell_candidate_buffer, 2u * did + 1u);
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, candidate & HASHGRIDCACHE_CELL_ENTRY_MASK);
    let first = HashGridCache_CellListWord(g_HashGridCache_RayLists.cell_priority_bins, bin);
    let count = HashGridCache_CellListWord(g_HashGridCache_RayLists.cell_priority_bins, HASHGRIDCACHE_CELL_BINS + bin);
    let lit_target = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSHADEDTARGET]);

    // A bin wholly under the target is lit; the one bin the target cuts is thinned by a hash of the
    // cell, each lit with the share of the bin the target leaves (lighting it in claim order leaned
    // the drawn hits toward lower entries). Slots come from the candidate's own bin, so the bins
    // under the target fill their slots whatever the order, and the cut bin's lit cells follow them
    // up to the target; the lit count is what ShadeCells reads.
    var lit = first + count <= lit_target;
    if (!lit && first < lit_target) {
        let share = f32(lit_target - first) / f32(count);
        lit = f32(rcPcgHash(rcPcgHash(cell_index ^ HASHGRIDCACHE_CELL_THIN_SALT) + rcPcgHash(g_HashGridCacheConstants.frame_index)) >> 8u) * (1.0 / 16777216.0) < share;
    }
    if (lit) {
        let slot = atomicAdd(&g_HashGridCache_CellLists[g_HashGridCache_RayLists.cell_priority_bins + 2u * HASHGRIDCACHE_CELL_BINS + bin], 1u);
        if (slot < lit_target) {
            HashGridCache_SetCellListWord(g_HashGridCache_RayLists.cell_shade_buffer, slot, candidate);
            atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSHADEDCOUNT], 1u);
            return;
        }
        // Past the target (the thinned bin drew more than its share): not lit.
    }
    // Not lit this frame: the key and the claim leave the scratch, which UpdateTiles then reads as
    // no sample (the cell keeps its value).
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 0u], 0u);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 1u], 0u);
}
`;
}

/**
 * `ShadeCells`: over the shade list, the cell's sample hit draws one light and
 * traces its shadow ray, as `GenerateReservoirs` and `PopulateCells` do for a
 * hit (the platform's light choice, register D4, with the sky as
 * `HashGridCache_LightMask` sets it, D62; Lambert, D8), and the result
 * is the cell's one sample this frame, stored over the key in its scratch. A
 * changed cell's value is emptied first, so the sample replaces it.
 *
 * Built with the platform's hit-shading module (group 0).
 */
export function shadeCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = CELL_SHADING_WORKGROUP_SIZE.shadeCells;
    return /* wgsl */ `
${cellShadingHeadWgsl(layout)}
${PROBE_RAY_SURFACE_WGSL}
${LIGHT_MASK_WGSL}
const HASHGRIDCACHE_CELL_SHADE_SALT: u32 = ${CELL_SHADING_SALT.shade}u;
// INV_PI (math/math_constants.hlsl).
const HASHGRIDCACHE_SHADE_INV_PI: f32 = 0.318309886183791;

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.shadeCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    // The lit count can pass the target where the thinned bin drew more than its share; those weren't listed.
    if (did >= min(atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSHADEDCOUNT]), atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSHADEDTARGET]))) {
        return; // out of bounds
    }
    let shaded = HashGridCache_CellListWord(g_HashGridCache_RayLists.cell_shade_buffer, did);
    let visibility_index = shaded & HASHGRIDCACHE_CELL_ENTRY_MASK;
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index);
    let surface = HashGridCache_ProbeSurface(HashGridCache_VisibilityRay(visibility_index));

    // One light, drawn from the lights GenerateReservoirs draws from (HashGridCache_LightMask: every
    // group, and the sky unless E10's fallback holds it, register D62), and its shadow ray, as
    // PopulateCells traces it.
    var random = rcMakeRandom(visibility_index ^ HASHGRIDCACHE_CELL_SHADE_SALT, g_HashGridCacheConstants.frame_index);
    let reservoir = giDirectChooseLight(surface, HashGridCache_LightMask(), rcRand(&random));
    let sample_params = rcRand2(&random);
    // A changed cell, or one with no whole sample (new), starts again from this sample (UpdateTiles
    // takes a sample whole into a value with no count).
    let value = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index]);
    let replaces = (shaded & HASHGRIDCACHE_CELL_CHANGED_BIT) != 0u || value.w < 1.0 || value.w == HASHGRIDCACHE_CELL_CHANGED_COUNT;
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;

    // No light to draw (it faces away from every light, and the sky is black or not a candidate): a
    // sample of 0, as per hit (HashGridCache_AddSampleOfNoLight, register D63), so a cell's mean is
    // over all its hits, as per hit's is.
    var lighting = vec3f(0.0);
    if (reservoir.light != GI_DIRECT_NO_LIGHT) {
        let light_ray = giDirectLightRay(surface, reservoir.light, sample_params);
        let occluded = light_ray.castsShadows && giTraceAny(light_ray.shadowRay);
        if (!occluded) {
            lighting = surface.albedo * HASHGRIDCACHE_SHADE_INV_PI * light_ray.irradiance * reservoir.weight;
        }
    }
    if (replaces) {
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = vec2u(0u);
    }

    // The cell's one sample, over the key and the claim.
    let quantized_radiance = HashGridCache_QuantizeRadiance(lighting);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 0u], quantized_radiance.x);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 1u], quantized_radiance.y);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 2u], quantized_radiance.z);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 3u], quantized_radiance.w);
}
`;
}
