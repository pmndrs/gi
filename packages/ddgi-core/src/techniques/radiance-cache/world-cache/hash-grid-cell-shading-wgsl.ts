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
 * Lever 1, budgeted cell shading the id Tech 8 way (RC-L1;
 * docs/design/radiance-cache.md, "Budgeted cell shading, id Tech 8's way";
 * Sousa, SIGGRAPH 2025, S15-S20): the four kernels `cellShading: 'budget'`
 * runs in place of Capsaicin's per-hit `GenerateReservoirs`. AMD's notice is
 * kept because the light draw, the shadow ray and the bypass cases are
 * `GenerateReservoirs` and `PopulateCells` (gi1.comp:2400-2600, 2040-2106).
 *
 * 1. `CellShadingPick`, over the first-bounce hits: what Capsaicin's
 *    `GenerateReservoirs` does for the cases that bypass the cache (an
 *    emissive front face gives its emission; a ray shorter than its cell
 *    draws its own light and shadow ray, which `PopulateCells` traces into
 *    its probe ray, and on a budget not into its cell), and every
 *    non-emissive hit, the bypassing ones too, offers itself to its cell: the
 *    cell keeps the hit with the largest key, a hash of (hit, frame), so a
 *    uniformly random one. Every non-emissive hit joins the resolve list, as
 *    with the multibounce.
 * 2. `CellShadingRank`, over the same hits: each cell's kept hit counts its
 *    cell in the histogram of age classes and puts it on the list of cells in
 *    use. The platform's scan then turns the histogram into each class's first
 *    rank (in the same pass).
 * 3. `CellShadingChoose`, over the cells in use: each takes the next rank of
 *    its class; the first `cell_shading_budget` ranks go on the list of cells
 *    to light, in rank order, and their kept hits are flagged as lit picks
 *    (the second bounce leaves from them). No sort. The kept-hit word then
 *    says "in use this frame" until `UpdateTiles` reads and resets it
 *    (`CELL_SHADING_MARK`).
 * 4. `CellShadingLight`, over the cells to light: the kept hit's surface
 *    through `giSurfaceAt` (the hit record, so lever 27's real textures reach
 *    it too), then `cell_shading_samples` times `GenerateReservoirs`' light
 *    draw (the sky among the lights) and `PopulateCells`' shadow ray, into the
 *    cell's update scratch, each a sample even when it finds no light or its
 *    shadow ray is blocked (so a light switched off fades from the cell);
 *    `UpdateTiles` blends it as it blends hits' samples.
 *
 * The age classes: 0 for a cell never lit (new), then one class per age,
 * oldest first: class 1 for 255 frames or more since lit, class 255 for lit
 * last frame. Ranking by the real age (not one shared "stale" rank) lets the
 * budget go round the cells in use: with N cells in use and a budget B, every
 * one is lit within about ceil(N / B) frames. `cell_shading_max_age` doesn't
 * change the order; it marks cells as stale for the stale-cell counter (a
 * frame budget, lever 22, would read it to raise the budget).
 */

import { CAPSAICIN_WGSL } from '../wgsl/capsaicin.ts';
import type { HashGridCacheBufferLayout } from './capacity.ts';
import {
    ALL_LIGHT_GROUPS,
    CELL_SHADING_CLASSES,
    INV_PI_WGSL,
    LIGHT_DRAW_WGSL,
    populationHeadWgsl,
    RESERVOIR_WORDS,
    CELL_SHADING_MARK,
    VISIBILITY_FLAG,
    VISIBILITY_SURFACE_WGSL,
    type PopulationBindings,
} from './hash-grid-population-wgsl.ts';
import { HASH_GRID_CACHE_WORDS_ONLY } from './hash-grid-wgsl.ts';

/** What each budget kernel binds (the hash grid in group 1, the ray data in group 2), counted against the platform's 8 storage buffers. */
export const CELL_SHADING_BINDINGS = {
    /** Hit shading (5), the cache words (the kept-hit words, counters), the radiance slots (emission), the lists: 8. */
    cellShadingPick: { cache: HASH_GRID_CACHE_WORDS_ONLY, rayData: { probeRays: false, radiance: true } },
    /** The cache words (kept hits, last-lit frames, counters), the lists, the age classes: 3. */
    cellShadingRank: { cache: HASH_GRID_CACHE_WORDS_ONLY, rayData: { probeRays: false, radiance: false, cellShadingClasses: true } },
    /** As the rank: 3. */
    cellShadingChoose: { cache: HASH_GRID_CACHE_WORDS_ONLY, rayData: { probeRays: false, radiance: false, cellShadingClasses: true } },
    /** Hit shading (5), the cache words (update scratch, last-lit frames, counters), the lists: 7. */
    cellShadingLight: { cache: HASH_GRID_CACHE_WORDS_ONLY, rayData: { probeRays: false, radiance: false } },
} as const satisfies Record<string, PopulationBindings>;

export type CellShadingPassName = keyof typeof CELL_SHADING_BINDINGS;

export const CELL_SHADING_ENTRY_POINT: Readonly<Record<CellShadingPassName, string>> = {
    cellShadingPick: 'CellShadingPick',
    cellShadingRank: 'CellShadingRank',
    cellShadingChoose: 'CellShadingChoose',
    cellShadingLight: 'CellShadingLight',
};

/** Threads per workgroup: the pick as `GenerateReservoirs` (64), the light as `PopulateCells` (32). */
export const CELL_SHADING_WORKGROUP_SIZE: Readonly<Record<CellShadingPassName, number>> = {
    cellShadingPick: 64,
    cellShadingRank: 64,
    cellShadingChoose: 64,
    cellShadingLight: 32,
};

/**
 * The key a hit offers its cell, the age class and the counters' workgroup
 * sums, shared by the kernels.
 */
/** The oldest age the ranking tells apart (`CELL_SHADING_CLASSES` classes: never lit, then ages 255 down to 1). */
export const CELL_SHADING_OLDEST_AGE = CELL_SHADING_CLASSES - 1;

const CELL_SHADING_COMMON_WGSL = /* wgsl */ `
const HASHGRIDCACHE_CELLSHADING_OLDEST_AGE: u32 = ${CELL_SHADING_OLDEST_AGE}u;
// The key a first-bounce hit offers its cell this frame. It must be different for every hit (so
// the cell's largest key names one hit) and look random (so the kept hit is a uniform pick):
// the hit's index XORed with a hash of the frame, then Chris Wellons' lowbias32 mixer
// (https://nullprogram.com/blog/2018/07/31/). Every step is invertible on u32 (an XOR, an
// xorshift, a multiply by an odd number), so two hits never share a key. 0 means "no hit kept",
// so the one hit a frame whose key comes out 0 (one value in 2^32) doesn't offer itself.
fn HashGridCache_CellShadingKey(visibility_index: u32) -> u32 {
    var x = visibility_index ^ pcgHash(g_HashGridCacheConstants.frame_index);
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Frames since the cell was last lit by the budget, at least 1 (it is read before this frame's
// lighting). The last-lit word holds the frame plus one, 0 for never; the subtraction wraps with
// the frame counter, as PurgeTiles' decay does.
fn HashGridCache_CellShadingAge(last_lit: u32) -> u32 {
    return max((g_HashGridCacheConstants.frame_index + 1u) - last_lit, 1u);
}

// The cell's age class: 0 never lit (new), then by age, oldest first: 255 frames or more is
// class 1, lit last frame class 255. Every age has its own class, so cells lit long ago always go
// ahead of cells lit recently and the budget goes round all the cells in use.
fn HashGridCache_CellShadingClass(last_lit: u32) -> u32 {
    if (last_lit == 0u) {
        return 0u;
    }
    return 1u + (HASHGRIDCACHE_CELLSHADING_OLDEST_AGE - min(HashGridCache_CellShadingAge(last_lit), HASHGRIDCACHE_CELLSHADING_OLDEST_AGE));
}

// Stale: never lit, or lit cell_shading_max_age frames ago or more (the stale-cell counter).
fn HashGridCache_CellShadingIsStale(last_lit: u32) -> bool {
    return last_lit == 0u || HashGridCache_CellShadingAge(last_lit) >= g_HashGridCacheConstants.cell_shading_max_age;
}

// Flags of a first-bounce entry in g_HashGridCache_RayLists.visibility_bypass_buffer: its probe
// ray bypasses the cache (register D40, as per hit), and, on a budget, it is its cell's lit pick.
const HASHGRIDCACHE_VISIBILITY_FLAG_BYPASS: u32 = ${VISIBILITY_FLAG.bypass}u;
const HASHGRIDCACHE_VISIBILITY_FLAG_LIT_PICK: u32 = ${VISIBILITY_FLAG.litPick}u;
`;

// ---- CellShadingPick -----------------------------------------------------------------------------

/**
 * `CellShadingPick`: `GenerateReservoirs` (gi1.comp:2400-2600) for the hits
 * that bypass the cache, and the offer of every other hit to its cell. Built
 * with the platform's hit-shading module (group 0).
 */
export function cellShadingPickWgsl(layout: HashGridCacheBufferLayout): string {
    const size = CELL_SHADING_WORKGROUP_SIZE.cellShadingPick;
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${populationHeadWgsl(layout, CELL_SHADING_BINDINGS.cellShadingPick)}
${VISIBILITY_SURFACE_WGSL}
${CELL_SHADING_COMMON_WGSL}
const HASHGRIDCACHE_LIGHT_GROUPS: u32 = ${ALL_LIGHT_GROUPS}u;
${LIGHT_DRAW_WGSL}
@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.cellShadingPick}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        return; // out of bounds
    }

    let visibility = HashGridCache_LoadVisibility(did);
    let surface = HashGridCache_VisibilitySurface(visibility);
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did);

    // As GenerateReservoirs (gi1.comp:2439-2452): an emissive front face gives the probe ray its
    // emission and nothing more; it neither offers itself nor is resolved.
    if (g_HashGridCacheConstants.use_direct_lighting != 0u && surface.frontFace && dot(surface.emission, surface.emission) > 0.0) {
        ScreenProbes_AccumulateRadiance(query_index, surface.emission);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_bypass_buffer, did, 0u);
        return;
    }

    // As GenerateReservoirs (gi1.comp:2531-2561, register D37): a ray shorter than its cell
    // bypasses the cell's filtered value.
    let cell_size  = HashGridCache_GetCellSize(surface.position);
    let ray_length = distance(visibility.origin, surface.position);
    let bypass_cache = ray_length < cell_size;
    let bypass_bit = select(0u, HASHGRIDCACHE_BYPASS_BIT, bypass_cache);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_bypass_buffer, did, select(0u, HASHGRIDCACHE_VISIBILITY_FLAG_BYPASS, bypass_cache));

    // Offer this hit to its cell, the bypassing ones too: the cell keeps the largest key (a
    // uniformly random hit, see the key), and its light is that pick's alone (a bypassing hit's own
    // sample below goes to its probe ray only). The words are 0 between frames (UpdateTiles resets
    // each cell it saw in use), so 0 is "none".
    let key = HashGridCache_CellShadingKey(did);
    if (key != 0u) {
        let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did);
        atomicMax(&g_HashGridCache_Words[HASHGRIDCACHE_CELLSHADINGKEPTHITBUFFER + cell_index], key);
    }

    // Every other hit reads its cell (or, bypassing, takes its own light): the resolve list as
    // GI-1.2's multibounce builds it (gi1.comp:2563-2569). With the multibounce off, Capsaicin
    // resolves only the hits that drew a light (its shadow-ray list); on a budget a hit draws no
    // light of its own, so every hit is resolved in both settings.
    let resolve_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_RESOLVECOUNTBUFFER], 1u);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.resolve_buffer, resolve_index, did | bypass_bit);

    if (bypass_cache) {
        // The bypass's probe ray is still lit per hit, as Capsaicin lights it (few hits):
        // GenerateReservoirs' light draw and its shadow ray on the list PopulateCells traces (which,
        // on a budget, adds it to the probe ray and not to the cell).
        var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);
        let draw = HashGridCache_DrawLight(surface, &random);
        if (draw.choice.light == GI_DIRECT_NO_LIGHT) {
            return;
        }
        let ray_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER], 1u);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_ray_buffer, ray_index, did | bypass_bit);
        let reservoir_word = g_HashGridCache_RayLists.reservoir_buffer + ${RESERVOIR_WORDS}u * ray_index;
        HashGridCache_SetRayListWord(reservoir_word, 0u, draw.choice.light);
        HashGridCache_SetRayListWord(reservoir_word, 1u, bitcast<u32>(draw.choice.weight));
        HashGridCache_SetRayListWord(reservoir_word, 2u, pack2x16unorm(draw.sample_params));
    }
}
`;
}

// ---- CellShadingRank -----------------------------------------------------------------------------

/**
 * `CellShadingRank`: each cell's kept hit (the one hit whose key the cell
 * holds) counts the cell in its age class and lists it as in use. Both the
 * histogram and the list's counter are summed per workgroup first, so a frame
 * where every cell falls in one class makes one global atomic per workgroup.
 * The scan of the histogram runs after it, in the same pass.
 */
export function cellShadingRankWgsl(layout: HashGridCacheBufferLayout): string {
    const size = CELL_SHADING_WORKGROUP_SIZE.cellShadingRank;
    return /* wgsl */ `
${populationHeadWgsl(layout, CELL_SHADING_BINDINGS.cellShadingRank)}
${CELL_SHADING_COMMON_WGSL}
var<workgroup> lds_CellShading_Histogram: array<atomic<u32>, ${CELL_SHADING_CLASSES}>;
var<workgroup> lds_CellShading_UsedCount: atomic<u32>;
var<workgroup> lds_CellShading_UsedBase: u32;
var<workgroup> lds_CellShading_StaleCount: atomic<u32>;

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.cellShadingRank}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    for (var bin = local_index; bin < ${CELL_SHADING_CLASSES}u; bin += ${size}u) {
        atomicStore(&lds_CellShading_Histogram[bin], 0u);
    }
    if (local_index == 0u) {
        atomicStore(&lds_CellShading_UsedCount, 0u);
        atomicStore(&lds_CellShading_StaleCount, 0u);
    }
    workgroupBarrier();

    // Every thread reaches the barriers below; one past the hits, or whose hit isn't its cell's
    // kept one, just adds nothing.
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    var cell_index = 0u;
    var is_kept = false;
    var slot = 0u;
    if (did < atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did);
        let key = HashGridCache_CellShadingKey(did);
        is_kept = key != 0u && atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_CELLSHADINGKEPTHITBUFFER + cell_index]) == key;
    }
    if (is_kept) {
        let last_lit = atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_CELLSHADINGLASTLITBUFFER + cell_index]);
        atomicAdd(&lds_CellShading_Histogram[HashGridCache_CellShadingClass(last_lit)], 1u);
        slot = atomicAdd(&lds_CellShading_UsedCount, 1u);
        if (HashGridCache_CellShadingIsStale(last_lit)) {
            atomicAdd(&lds_CellShading_StaleCount, 1u);
        }
    }
    workgroupBarrier();

    // One thread takes the workgroup's room on the list; the bins go out one per thread.
    if (local_index == 0u) {
        lds_CellShading_UsedBase = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_USEDCELLCOUNT], atomicLoad(&lds_CellShading_UsedCount));
        atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_STALECELLCOUNT], atomicLoad(&lds_CellShading_StaleCount));
    }
    for (var bin = local_index; bin < ${CELL_SHADING_CLASSES}u; bin += ${size}u) {
        let count = atomicLoad(&lds_CellShading_Histogram[bin]);
        if (count > 0u) {
            atomicAdd(&g_HashGridCache_CellShadingClasses[HASHGRIDCACHE_CELLSHADING_HISTOGRAM + bin], count);
        }
    }
    workgroupBarrier();
    if (is_kept) {
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.used_cell_buffer, lds_CellShading_UsedBase + slot, did);
    }
}
`;
}

// ---- CellShadingChoose ---------------------------------------------------------------------------

/**
 * `CellShadingChoose`: each cell in use takes the next rank of its class (the
 * scanned histogram holds each class's first rank); ranks below the budget go
 * on the list of cells to light at their rank, so the list is in rank order
 * and has exactly min(budget, cells in use) entries, and their kept hits get
 * the lit-pick flag. The lit count is the
 * number of cells that got such a rank, summed per workgroup, so the page's
 * check of it tests the ranks rather than a min() written by hand.
 */
export function cellShadingChooseWgsl(layout: HashGridCacheBufferLayout): string {
    const size = CELL_SHADING_WORKGROUP_SIZE.cellShadingChoose;
    return /* wgsl */ `
${populationHeadWgsl(layout, CELL_SHADING_BINDINGS.cellShadingChoose)}
${CELL_SHADING_COMMON_WGSL}
var<workgroup> lds_CellShading_LitCount: atomic<u32>;

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.cellShadingChoose}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    if (local_index == 0u) {
        atomicStore(&lds_CellShading_LitCount, 0u);
    }
    workgroupBarrier();

    let used_index = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (used_index < atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_USEDCELLCOUNT])) {
        let did = HashGridCache_RayListWord(g_HashGridCache_RayLists.used_cell_buffer, used_index);
        let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did);
        let age_class = HashGridCache_CellShadingClass(atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_CELLSHADINGLASTLITBUFFER + cell_index]));
        let rank = atomicLoad(&g_HashGridCache_CellShadingClasses[HASHGRIDCACHE_CELLSHADING_HISTOGRAM + age_class]) +
                   atomicAdd(&g_HashGridCache_CellShadingClasses[HASHGRIDCACHE_CELLSHADING_CURSOR + age_class], 1u);
        // The key has done its work (Rank); from here the word says "in use this frame" (CELL_SHADING_MARK).
        atomicStore(&g_HashGridCache_Words[HASHGRIDCACHE_CELLSHADINGKEPTHITBUFFER + cell_index], ${CELL_SHADING_MARK.inUse}u);
        if (rank < g_HashGridCacheConstants.cell_shading_budget) {
            HashGridCache_SetRayListWord(g_HashGridCache_RayLists.chosen_cell_buffer, rank, did);
            // The second bounce leaves from this hit (PopulateMultibounceCells, lit cells).
            atomicOr(&g_HashGridCache_RayListWords[g_HashGridCache_RayLists.visibility_bypass_buffer + did], HASHGRIDCACHE_VISIBILITY_FLAG_LIT_PICK);
            atomicAdd(&lds_CellShading_LitCount, 1u);
        }
    }
    workgroupBarrier();
    if (local_index == 0u) {
        let lit = atomicLoad(&lds_CellShading_LitCount);
        if (lit > 0u) {
            atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_LITCELLCOUNT], lit);
        }
    }
}
`;
}

// ---- CellShadingLight ----------------------------------------------------------------------------

/**
 * `CellShadingLight`: for each cell to light, its kept hit's surface, then
 * `cell_shading_samples` light draws, each as `GenerateReservoirs` draws one
 * (gi1.comp:2500-2525; the sky among the lights, RC6.1) and lit as
 * `PopulateCells` lights it (gi1.comp:2040-2098): a shadow ray, and if it is
 * clear the BRDF toward the probe ray times the light's radiance and weight.
 * Each sample is quantised as `PopulateCells` quantises a hit's and counts one
 * in the cell's sample count, so `UpdateTiles` blends a lit cell's samples as
 * it would the same number of hits'. A draw that finds no light, or whose
 * shadow ray is blocked, is a sample of 0: per hit the cell's other hits
 * keep it current, here only the lit cell's own samples do, and a cell whose
 * light went out must fade to it. The sample's light point goes through the
 * reservoir's 16-bit packing first, as `PopulateCells` reads it back. Built
 * with the platform's hit-shading module (group 0).
 */
export function cellShadingLightWgsl(layout: HashGridCacheBufferLayout): string {
    const size = CELL_SHADING_WORKGROUP_SIZE.cellShadingLight;
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${populationHeadWgsl(layout, CELL_SHADING_BINDINGS.cellShadingLight)}
${VISIBILITY_SURFACE_WGSL}
${INV_PI_WGSL}
const HASHGRIDCACHE_LIGHT_GROUPS: u32 = ${ALL_LIGHT_GROUPS}u;
${LIGHT_DRAW_WGSL}
var<workgroup> lds_CellShading_ShadowRays: atomic<u32>;
var<workgroup> lds_CellShading_Samples: atomic<u32>;

@compute @workgroup_size(${size})
fn ${CELL_SHADING_ENTRY_POINT.cellShadingLight}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    if (local_index == 0u) {
        atomicStore(&lds_CellShading_ShadowRays, 0u);
        atomicStore(&lds_CellShading_Samples, 0u);
    }
    workgroupBarrier();

    let lit_index = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (lit_index < atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_LITCELLCOUNT])) {
        let did = HashGridCache_RayListWord(g_HashGridCache_RayLists.chosen_cell_buffer, lit_index);
        let surface = HashGridCache_VisibilitySurface(HashGridCache_LoadVisibility(did));
        let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did);

        // The random numbers GenerateReservoirs would draw for this hit, then on for more samples.
        var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);
        var quantized_sum = vec4u(0u);
        var shadow_rays = 0u;
        for (var sample_index = 0u; sample_index < g_HashGridCacheConstants.cell_shading_samples; sample_index++) {
            let draw = HashGridCache_DrawLight(surface, &random);
            if (draw.choice.light == GI_DIRECT_NO_LIGHT) {
                quantized_sum += vec4u(0u, 0u, 0u, 1u);   // no light, no shadow ray: a sample of 0
                continue;
            }
            let sample_params = unpack2x16unorm(pack2x16unorm(draw.sample_params));
            let light_ray = giDirectLightRay(surface, draw.choice.light, sample_params);
            var occluded = false;
            if (light_ray.castsShadows) {
                occluded = giTraceAny(light_ray.shadowRay);
                shadow_rays += 1u;
            }
            var lighting = vec3f(0.0);
            if (!occluded) {
                lighting = surface.albedo * HASHGRIDCACHE_INV_PI * light_ray.irradiance * draw.choice.weight;
            }
            quantized_sum += HashGridCache_QuantizeRadiance(lighting);
        }

        // Into the cell's update scratch, as PopulateCells adds a hit's sample.
        let scratch = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;
        if (any(quantized_sum.xyz > vec3u(0u))) {
            atomicAdd(&g_HashGridCache_Words[scratch + 0u], quantized_sum.x);
            atomicAdd(&g_HashGridCache_Words[scratch + 1u], quantized_sum.y);
            atomicAdd(&g_HashGridCache_Words[scratch + 2u], quantized_sum.z);
        }
        if (quantized_sum.w > 0u) {
            atomicAdd(&g_HashGridCache_Words[scratch + 3u], quantized_sum.w);
        }
        // Lit this frame: the frame plus one (0 is "never"; the one frame in 2^32 where that
        // wraps to 0 stores 1, a hair old, rather than "never").
        atomicStore(&g_HashGridCache_Words[HASHGRIDCACHE_CELLSHADINGLASTLITBUFFER + cell_index], max(g_HashGridCacheConstants.frame_index + 1u, 1u));
        atomicAdd(&lds_CellShading_ShadowRays, shadow_rays);
        atomicAdd(&lds_CellShading_Samples, quantized_sum.w);
    }
    workgroupBarrier();
    if (local_index == 0u) {
        let rays = atomicLoad(&lds_CellShading_ShadowRays);
        if (rays > 0u) {
            atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_CELLSHADOWRAYCOUNT], rays);
        }
        let samples = atomicLoad(&lds_CellShading_Samples);
        if (samples > 0u) {
            atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_CELLSAMPLECOUNT], samples);
        }
    }
}
`;
}
