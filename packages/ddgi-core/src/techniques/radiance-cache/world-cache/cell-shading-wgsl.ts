/**
 * Lever L1 (TASKS P6.2): id Tech 8's "shade cells, not hits" (Sousa,
 * SIGGRAPH 2025, slides 17-20; research/gi-1.0/idtech8-performance.md, T1-T3)
 * on GI-1.2's hash grid, behind `cellShading: 'budgeted'`
 * (docs/design/radiance-cache.md, "Lever L1").
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
 * 2. `PrioritizeCells` (one thread): the bins' counts become their first
 *    slots (a counting sort by priority over 32 bins, too few for a scan
 *    kernel), the cells lit this frame are min(candidates, budget), and the
 *    period of the settled cells' turns is set for the next frame.
 * 3. `ScatterCells` (over the candidates): each takes the next slot of its
 *    bin; a slot under the budget goes on the shade list, and a candidate
 *    past it clears the key from its scratch (it is not lit this frame).
 * 4. `ShadeCells` (over the shade list): one light and shadow ray from the
 *    cell's sample hit, stored as the cell's one sample in its scratch, where
 *    `UpdateTiles` blends it in as it blends per-hit samples.
 *
 * The priority classes, highest first:
 * - **new or changed:** no sample yet, or the sample hit is near one of this
 *   frame's scene changes (`SceneChanges.regions`, or every cell for a change
 *   with no bounds). A changed cell keeps its mean (nothing goes dark) but is
 *   marked by its count (`CELL_CHANGED_COUNT`), so whenever it is next lit,
 *   this frame or later, its new sample replaces its value; it stays in this
 *   class until then;
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
 * to draw adds no sample; so here: a lit cell whose sample hit has no light
 * adds none, and a cell with no estimate is marked dark (`CELL_DARK_COUNT`, a
 * count too small to move its tile's coarser levels) so it isn't taken as
 * new every frame. (A sample of 0 instead dimmed the coarser levels where
 * floor and ceiling share a tile, and the furnace's second bounce read low.)
 */

import { CAPSAICIN_WGSL } from '../wgsl/capsaicin.ts';
import type { HashGridCacheBufferLayout } from './capacity.ts';
import { CELL_SHADING_COMMON_WGSL, CELL_SHADING_SALT } from './cell-shading-common.ts';
import { ALL_LIGHT_GROUPS, populationHeadWgsl, PROBE_RAY_SURFACE_WGSL } from './hash-grid-population-wgsl.ts';

/** `GPUShaderStage.COMPUTE` (the global doesn't exist in Node, where the tests import this file). */
const SHADER_STAGE_COMPUTE = 0x4;

/** The group the lever's kernels put their uniform in (the hash grid is group 1, the ray data group 2). */
export const CELL_SHADING_GROUP = 3;

export const CELL_SHADING_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [{ binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } }];

/** The most change boxes a frame passes; more count as a change everywhere (as the screen probes' history counts them, D46). */
export const CELL_SHADING_MAX_CHANGE_BOXES = 16;

/** Bytes of `HashGridCacheCellShading`: four words, then 32 bytes a box. */
export const CELL_SHADING_UNIFORM_BYTES = 16 + 32 * CELL_SHADING_MAX_CHANGE_BOXES;

/** A candidate's word: its visibility entry in the low 23 bits, the changed bit, and its priority bin in the top 8. */
export const CELL_CANDIDATE_ENTRY_BITS = 23;
export const CELL_CANDIDATE_CHANGED_BIT = 1 << CELL_CANDIDATE_ENTRY_BITS;
export const CELL_CANDIDATE_BIN_SHIFT = 24;

/** The most visibility entries a frame can have with the candidates' word: a screen of at most 2^22 probe rays (4K has 2.1 M). */
export const CELL_SHADING_MAX_ENTRIES = 2 ** CELL_CANDIDATE_ENTRY_BITS;

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

const CELL_SHADING_UNIFORM_WGSL = /* wgsl */ `
struct HashGridCacheChangeBox {
    lower: vec3f,
    padding0: f32,
    upper: vec3f,
    padding1: f32,
}

// Lever L1's per-frame values (hash-grid-cache.ts writes them).
struct HashGridCacheCellShading {
    budget: u32,                    // cells lit this frame at most
    change_everything: u32,         // 1: a change with no bounds (the sun, the sky) or too many boxes
    change_box_count: u32,
    padding0: u32,
    change_boxes: array<HashGridCacheChangeBox, ${CELL_SHADING_MAX_CHANGE_BOXES}>,
}

@group(${CELL_SHADING_GROUP}) @binding(0) var<uniform> g_HashGridCache_CellShading: HashGridCacheCellShading;

const HASHGRIDCACHE_CELL_ENTRY_MASK: u32 = ${CELL_SHADING_MAX_ENTRIES - 1}u;
const HASHGRIDCACHE_CELL_CHANGED_BIT: u32 = ${CELL_CANDIDATE_CHANGED_BIT}u;
const HASHGRIDCACHE_CELL_BIN_SHIFT: u32 = ${CELL_CANDIDATE_BIN_SHIFT}u;
`;

/** The shared head of the lever's kernels. */
function cellShadingHeadWgsl(layout: HashGridCacheBufferLayout): string {
    return `${CAPSAICIN_WGSL}\n${populationHeadWgsl(layout)}\n${CELL_SHADING_COMMON_WGSL}\n${CELL_SHADING_UNIFORM_WGSL}`;
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

// Near one of this frame's scene changes: inside a change box grown by the cell's size (the hit
// stands for its cell), or anywhere for a change with no bounds.
fn HashGridCache_NearChange(position: vec3f, margin: f32) -> bool {
    if (g_HashGridCache_CellShading.change_everything != 0u) {
        return true;
    }
    let count = min(g_HashGridCache_CellShading.change_box_count, ${CELL_SHADING_MAX_CHANGE_BOXES}u);
    for (var box_index = 0u; box_index < count; box_index++) {
        let change = g_HashGridCache_CellShading.change_boxes[box_index];
        if (all(position >= change.lower - vec3f(margin)) && all(position <= change.upper + vec3f(margin))) {
            return true;
        }
    }
    return false;
}

// An indirect value whose history is cut to one sample: its mean kept, its count 1 (a value with at
// most one sample is left as it is). The next sample then weighs as much as all the history.
fn HashGridCache_KeepOneSample(value_index: u32) {
    let value = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[value_index]);
    if (value.w > 1.0) {
        g_HashGridCache_CellValues[value_index] = HashGridCache_PackRadiance(vec4f(value.rgb / value.w, 1.0));
    }
}

// A changed cell's direct value: its mean kept (so nothing goes dark) with the changed count, so that
// whenever it is next lit its sample replaces it (ShadeCells). A cell with no samples (new, or lit
// with no light) is emptied: it is lit again first.
fn HashGridCache_MarkChanged(cell_index: u32) {
    let value = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index]);
    var marked = vec4f(0.0);
    if (value.w >= 1.0) {
        marked = vec4f(value.rgb / value.w * HASHGRIDCACHE_CELL_CHANGED_COUNT, HASHGRIDCACHE_CELL_CHANGED_COUNT);
    }
    g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = HashGridCache_PackRadiance(marked);
}

fn HashGridCache_ClaimCell(visibility_index: u32) {
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index);
    if (cell_index == kGI1_InvalidId) {
        return; // a second-bounce ray that missed: no cell
    }

    // The entry that offered the cell's largest key claims it. Two entries with the same key (one
    // pair in billions) are told apart by the exchange: the first one there wins.
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;
    if (atomicLoad(&g_HashGridCache_UpdateScratch[scratch + 0u]) != HashGridCache_CellSampleKey(visibility_index)) {
        return;
    }
    if (atomicExchange(&g_HashGridCache_UpdateScratch[scratch + 1u], 1u) != 0u) {
        return;
    }

    let ray = HashGridCache_VisibilityRay(visibility_index);
    let position = ray.origin + ray.hitT * worldCacheProbeRayDirection(ray);
    let changed = HashGridCache_NearChange(position, HashGridCache_GetCellSize(position));
    let direct = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index]);

    if (changed) {
        HashGridCache_MarkChanged(cell_index);
        if (g_HashGridCacheConstants.use_multibounce != 0u) {
            HashGridCache_KeepOneSample(HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index);
        }
    }

    // New (no count), changed (now or on an earlier frame, not lit since), converging (whole counts
    // under the cap), settled (at the cap, or lit with no light: the dark count).
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
    atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLPRIORITYBINS + bin], 1u);

    let candidate = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLCANDIDATECOUNT], 1u);
    let changed_bit = select(0u, HASHGRIDCACHE_CELL_CHANGED_BIT, changed);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.cell_candidate_buffer, candidate, visibility_index | changed_bit | (bin << HASHGRIDCACHE_CELL_BIN_SHIFT));
}

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.claimFirstBounceCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        return; // out of bounds
    }
    HashGridCache_ClaimCell(did);
}

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.claimSecondBounceCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1])) {
        return; // out of bounds
    }
    HashGridCache_ClaimCell(atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]) + did);
}
`;
}

/**
 * `PrioritizeCells`: one thread turns the bins' counts into their first
 * slots (an exclusive prefix sum over 32 words), sets the frame's lit count
 * and the next frame's turn period. CPU twin `prioritizeCells` (cell-shading-cpu.ts).
 */
export function prioritizeCellsWgsl(layout: HashGridCacheBufferLayout): string {
    return /* wgsl */ `
${cellShadingHeadWgsl(layout)}

@compute @workgroup_size(1)
fn ${CELL_SHADING_ENTRY_POINT.prioritizeCells}() {
    let candidates = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLCANDIDATECOUNT]);
    let budget = g_HashGridCache_CellShading.budget;

    var first_slot = 0u;
    var urgent = 0u; // candidates new, changed or converging
    for (var bin = 0u; bin < HASHGRIDCACHE_CELL_BINS; bin++) {
        let count = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLPRIORITYBINS + bin]);
        atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLPRIORITYBINS + bin], first_slot);
        first_slot += count;
        if (bin < ${CELL_PRIORITY_CLASS.turn}u * HASHGRIDCACHE_CELL_SUB_BINS) {
            urgent += count;
        }
    }
    atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSHADEDCOUNT], min(candidates, budget));

    // The settled cells take what the urgent ones leave of the budget; a period of settled / left
    // frames gives each its turn once in that many frames (the turns' expected count is then at
    // most what is left). With nothing left this frame, a turn comes once in 'settled' frames.
    let settled = candidates - urgent;
    let left = max(budget - min(budget, urgent), 1u);
    atomicStore(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSCHEDULEPERIOD], max((settled + left - 1u) / left, 1u));
}
`;
}

/** `ScatterCells`: each candidate takes its bin's next slot; under the budget it goes on the shade list, past it its scratch is cleared. */
export function scatterCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = CELL_SHADING_WORKGROUP_SIZE.scatterCells;
    return /* wgsl */ `
${cellShadingHeadWgsl(layout)}

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
    let candidate = HashGridCache_RayListWord(g_HashGridCache_RayLists.cell_candidate_buffer, did);
    let bin = candidate >> HASHGRIDCACHE_CELL_BIN_SHIFT;
    let slot = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLPRIORITYBINS + bin], 1u);
    if (slot < atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSHADEDCOUNT])) {
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.cell_shade_buffer, slot, candidate & (HASHGRIDCACHE_CELL_ENTRY_MASK | HASHGRIDCACHE_CELL_CHANGED_BIT));
        return;
    }
    // Not lit this frame: the key and the claim leave the scratch, which UpdateTiles then reads as
    // no sample (the cell keeps its value).
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, candidate & HASHGRIDCACHE_CELL_ENTRY_MASK);
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 0u], 0u);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 1u], 0u);
}
`;
}

/**
 * `ShadeCells`: over the shade list, the cell's sample hit draws one light and
 * traces its shadow ray, as `GenerateReservoirs` and `PopulateCells` do for a
 * hit (the platform's light choice, register D4; Lambert, D8), and the result
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

const HASHGRIDCACHE_SHADE_LIGHT_GROUPS: u32 = ${ALL_LIGHT_GROUPS}u;
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
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_CELLSHADEDCOUNT])) {
        return; // out of bounds
    }
    let shaded = HashGridCache_RayListWord(g_HashGridCache_RayLists.cell_shade_buffer, did);
    let visibility_index = shaded & HASHGRIDCACHE_CELL_ENTRY_MASK;
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index);
    let surface = HashGridCache_ProbeSurface(HashGridCache_VisibilityRay(visibility_index));

    // One light, as GenerateReservoirs draws it, and its shadow ray, as PopulateCells traces it.
    var random = rcMakeRandom(visibility_index ^ HASHGRIDCACHE_CELL_SHADE_SALT, g_HashGridCacheConstants.frame_index);
    let reservoir = giDirectChooseLight(surface, HASHGRIDCACHE_SHADE_LIGHT_GROUPS, rcRand(&random));
    let sample_params = rcRand2(&random);
    // A changed cell, or one with no whole sample (new, or dark), starts again from this sample
    // (UpdateTiles takes a sample whole into a value with no count).
    let value = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index]);
    let replaces = (shaded & HASHGRIDCACHE_CELL_CHANGED_BIT) != 0u || value.w < 1.0 || value.w == HASHGRIDCACHE_CELL_CHANGED_COUNT;
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;

    if (reservoir.light == GI_DIRECT_NO_LIGHT) {
        // No light to draw: no sample, as per hit (GenerateReservoirs adds none for such a hit), so a
        // cell's mean is over the same hits per hit's is. A cell with no estimate is marked dark (not
        // new, so it doesn't take the budget every frame); one with samples keeps them.
        if (replaces) {
            g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = HashGridCache_PackRadiance(vec4f(0.0, 0.0, 0.0, HASHGRIDCACHE_CELL_DARK_COUNT));
        }
        atomicStore(&g_HashGridCache_UpdateScratch[scratch + 0u], 0u);
        atomicStore(&g_HashGridCache_UpdateScratch[scratch + 1u], 0u);
        return;
    }
    let light_ray = giDirectLightRay(surface, reservoir.light, sample_params);
    let occluded = light_ray.castsShadows && giTraceAny(light_ray.shadowRay);
    var lighting = vec3f(0.0);
    if (!occluded) {
        lighting = surface.albedo * HASHGRIDCACHE_SHADE_INV_PI * light_ray.irradiance * reservoir.weight;
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
