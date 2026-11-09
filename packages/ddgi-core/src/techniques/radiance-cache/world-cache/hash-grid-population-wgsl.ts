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
 * The hash grid's population and resolve (RC3.2): the passes Capsaicin v1.3
 * runs between tracing the screen probes and blending them (`GI1::render`,
 * gi1.cpp:2335-2528), ported from `gi1.comp` with its names:
 *
 * 1. `PopulateScreenProbesHandleHit` (gi1.comp:985-1076), the cache half of
 *    `PopulateScreenProbes`: each probe ray's hit inserts its cell, refreshes
 *    its tile's decay stamp, joins the visibility list, and a new tile joins
 *    the packed list with its cells cleared; a tile first touched this frame
 *    joins the update list.
 * 2. `GenerateReservoirs` (gi1.comp:2400-2600): an emissive front face gives
 *    its emission to the probe ray and stops there; any other hit draws one
 *    light (the platform's light choice, register D4) and joins the resolve
 *    list, flagged to bypass the cache when its ray is shorter than its cell.
 * 3. `PopulateCells` (gi1.comp:2040-2106): one shadow ray per drawn light; the
 *    light reflected toward the ray's origin is added, quantised, into the
 *    cell's update scratch, and straight into a bypassing ray's radiance.
 * 4. `GenerateUpdateTilesDispatch` and `UpdateTiles` (gi1.comp:2130-2349):
 *    each updated tile's finest cells blend this frame's samples into their
 *    history (at most `max_sample_count` samples), and its coarser levels are
 *    rewritten as 2 × 2 sums.
 * 5. `ResolveCells` (gi1.comp:2351-2385): each resolved ray that doesn't
 *    bypass the cache adds its cell's filtered radiance.
 *
 * Capsaicin compiles GI-1.2's multibounce in by default (`USE_MULTI_BOUNCE`,
 * gi1_use_multibounce true, gi1.h:55), so this is that path: the resolve list
 * holds every non-emissive hit, `UpdateTiles` and `ResolveCells` carry the
 * indirect values, and the visibility lists have room for the second-bounce
 * hits RC3.3 adds. Until RC3.3 nothing writes the indirect scratch, so the
 * indirect values stay 0.
 *
 * How the passes find their data (register D30): Capsaicin copies each hit
 * into its own visibility buffer (`HashGridCache_PackVisibility`) and keeps
 * the reservoir's surface and material in their own lists; here a visibility
 * entry keeps its probe ray's index (`VisibilityQueryBuffer`, which Capsaicin
 * also keeps) and its cell, and each pass reads the hit from the probe ray and
 * rebuilds the surface with the platform's `giSurfaceAt`. The lists live in
 * one packed buffer (register D6) at the word offsets of `HashGridCacheRayLists`.
 */

import { CAPSAICIN_WGSL } from '../wgsl/capsaicin.ts';
import type { HashGridCacheBufferLayout } from './capacity.ts';
import { HASH_GRID_KERNEL_GROUP, hashGridCacheWgsl } from './hash-grid-wgsl.ts';
import { WORLD_CACHE_PROBE_RAYS_WGSL } from './world-cache.ts';

/** `GPUShaderStage.COMPUTE` (the global doesn't exist in Node, where the tests import this file). */
const SHADER_STAGE_COMPUTE = 0x4;

/** The group the population kernels put the frame's rays and lists in (the hash grid is group 1). */
export const RAY_DATA_GROUP = 2;

/** Bindings of the ray-data group. */
export const RAY_DATA_BINDING = {
    /** `HashGridCacheRayLists`: where each list starts. Uniform. */
    rayLists: 0,
    /** The screen cache's probe rays (`WorldCacheProbeRays`). Read-only storage. */
    probeRays: 1,
    /** The probe rays' radiance slots (`array<vec2u>`, four halves each). Storage. */
    radiance: 2,
    /** The per-ray lists, packed (`g_HashGridCache_RayListWords`). Storage. */
    listWords: 3,
} as const;

export const RAY_DATA_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    { binding: RAY_DATA_BINDING.rayLists, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { binding: RAY_DATA_BINDING.probeRays, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: RAY_DATA_BINDING.radiance, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    { binding: RAY_DATA_BINDING.listWords, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
];

/** Bytes of `HashGridCacheRayLists`. */
export const RAY_LISTS_UNIFORM_BYTES = 32;

/** Words of a reservoir: the light, its weight (f32 bits), the point on the light (two unorm16). */
export const RESERVOIR_WORDS = 3;

/** The top bit of a visibility-ray or resolve entry: the ray bypasses the cache (gi1.comp:2566, 2582). */
export const BYPASS_CACHE_BIT = 0x80000000;

/** The light groups whose lights light the cache's hits: every group. */
export const ALL_LIGHT_GROUPS = 0xf;

/** Threads per workgroup, as Capsaicin's `[numthreads]`. */
export const POPULATION_WORKGROUP_SIZE = {
    /** PopulateScreenProbesMain (gi1.comp:1160): 32. */
    populateScreenProbes: 32,
    /** GenerateReservoirs (gi1.comp:2400): 64. */
    generateReservoirs: 64,
    /** PopulateCellsMain (gi1.comp:2110): 32. */
    populateCells: 32,
    /** UpdateTiles (gi1.comp:2120-2122, 2142): 8 × 8, one tile of 8 × 8 cells per workgroup at the default ratio. */
    updateTilesSide: 8,
    /** ResolveCells (gi1.comp:2350): 64. */
    resolveCells: 64,
} as const;

/** Entry points: Capsaicin's kernel names (the cache half of PopulateScreenProbes by its hit handler's name). */
export const POPULATION_ENTRY_POINT = {
    populateScreenProbes: 'PopulateScreenProbesHandleHit',
    generateReservoirs: 'GenerateReservoirs',
    populateCells: 'PopulateCells',
    updateTiles: 'UpdateTiles',
    resolveCells: 'ResolveCells',
} as const;

/** Where each per-ray list starts in the packed list words, for a screen of `maxRayCount` probe rays. */
export interface HashGridCacheRayListLayout {
    /** `max_ray_count` (gi1.cpp:129): the probe rays the screen can have. */
    readonly maxRayCount: number;
    /** `max_combined_ray_count` (gi1.cpp:637): with multibounce, room for a second-bounce hit per ray. */
    readonly maxCombinedRayCount: number;
    /** Word offsets. */
    readonly updateTileBuffer: number;
    readonly visibilityCellBuffer: number;
    readonly visibilityQueryBuffer: number;
    readonly visibilityRayBuffer: number;
    readonly resolveBuffer: number;
    readonly reservoirBuffer: number;
    readonly bytes: number;
}

/**
 * The lists Capsaicin sizes from the screen (`ensureMemoryIsAllocated`,
 * gi1.cpp:637-695): the update tiles (at most one per cell), the visibility
 * cell, query and ray lists (`max_combined_ray_count` each), the resolve list
 * (`max_ray_count`, first bounce only) and a reservoir per visibility ray
 * (`g_Reservoir_IndirectSampleReservoirBuffer`).
 */
export function hashGridCacheRayListLayout(maxRayCount: number, numCells: number): HashGridCacheRayListLayout {
    // USE_MULTI_BOUNCE is Capsaicin's default (gi1.h:55): bounce 0 and bounce 1 share the visibility lists.
    const maxCombinedRayCount = 2 * maxRayCount;
    const updateTileBuffer = 0;
    const visibilityCellBuffer = updateTileBuffer + Math.min(maxCombinedRayCount, numCells);
    const visibilityQueryBuffer = visibilityCellBuffer + maxCombinedRayCount;
    const visibilityRayBuffer = visibilityQueryBuffer + maxCombinedRayCount;
    const resolveBuffer = visibilityRayBuffer + maxCombinedRayCount;
    const reservoirBuffer = resolveBuffer + maxRayCount;
    const words = reservoirBuffer + RESERVOIR_WORDS * maxCombinedRayCount;
    return {
        maxRayCount,
        maxCombinedRayCount,
        updateTileBuffer,
        visibilityCellBuffer,
        visibilityQueryBuffer,
        visibilityRayBuffer,
        resolveBuffer,
        reservoirBuffer,
        bytes: Math.max(words, 1) * 4,
    };
}

/** The ray-data group's declarations, and Capsaicin's screen-probe helper the population passes call. */
const RAY_DATA_WGSL = /* wgsl */ `
${WORLD_CACHE_PROBE_RAYS_WGSL}

// Where each per-ray list starts in g_HashGridCache_RayListWords (gi1.cpp:637-695), in words.
struct HashGridCacheRayLists {
    max_ray_count: u32,
    max_combined_ray_count: u32,
    update_tile_buffer: u32,        // g_HashGridCache_UpdateTileBuffer: tiles first touched this frame
    visibility_cell_buffer: u32,    // g_HashGridCache_VisibilityCellBuffer: each visibility entry's cell
    visibility_query_buffer: u32,   // g_HashGridCache_VisibilityQueryBuffer: each visibility entry's probe ray
    visibility_ray_buffer: u32,     // g_HashGridCache_VisibilityRayBuffer: entries with a shadow ray (top bit: bypass)
    resolve_buffer: u32,            // g_HashGridCache_ResolveBuffer: entries to resolve (top bit: bypass)
    reservoir_buffer: u32,          // g_Reservoir_IndirectSampleReservoirBuffer: 3 words per visibility ray
}

@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.rayLists}) var<uniform> g_HashGridCache_RayLists: HashGridCacheRayLists;
@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.probeRays}) var<storage, read> g_ProbeRays: WorldCacheProbeRays;
// g_ScreenProbes_ProbeSpawnRadianceBuffer: each probe ray's radiance and hit distance as four halves.
@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.radiance}) var<storage, read_write> g_ScreenProbes_ProbeSpawnRadianceBuffer: array<vec2u>;
@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.listWords}) var<storage, read_write> g_HashGridCache_RayListWords: array<atomic<u32>>;

const HASHGRIDCACHE_BYPASS_BIT: u32 = ${BYPASS_CACHE_BIT}u;

// ScreenProbes_AccumulateRadiance (screen_probes.hlsl:248-259): adds a contribution to a probe
// ray's radiance, keeping its hit distance (packed as f32tof16 would, HashGridCache_PackHalf2).
// Not atomic, as in Capsaicin: each probe ray has one hit, so one visibility entry, and the
// passes that add to it run one after another.
fn ScreenProbes_AccumulateRadiance(query_index: u32, radiance: vec3f) {
    if (dot(radiance, radiance) > 0.0) {  // avoid accumulating null contribution(s)
        let packed_radiance = g_ScreenProbes_ProbeSpawnRadianceBuffer[query_index];
        var radiance_and_ray_distance = vec4f(unpack2x16float(packed_radiance.x), unpack2x16float(packed_radiance.y));
        radiance_and_ray_distance = vec4f(radiance_and_ray_distance.xyz + radiance, radiance_and_ray_distance.w);  // accumulate contribution
        g_ScreenProbes_ProbeSpawnRadianceBuffer[query_index] = vec2u(HashGridCache_PackHalf2(radiance_and_ray_distance.xy), HashGridCache_PackHalf2(radiance_and_ray_distance.zw));
    }
}

fn HashGridCache_RayListWord(offset: u32, index: u32) -> u32 {
    return atomicLoad(&g_HashGridCache_RayListWords[offset + index]);
}

fn HashGridCache_SetRayListWord(offset: u32, index: u32, value: u32) {
    atomicStore(&g_HashGridCache_RayListWords[offset + index], value);
}
`;

/**
 * The surface a probe ray hit, rebuilt from its hit record with the
 * platform's `giSurfaceAt` (for kernels built with the hit-shading module).
 * Capsaicin rebuilds it from its packed visibility the same way
 * (`fetchVerticesNormUV`, `interpolate`; gi1.comp:2412-2427, 2050-2062).
 */
const PROBE_RAY_SURFACE_WGSL = /* wgsl */ `
fn HashGridCache_ProbeRay(probe_ray: WorldCacheProbeRay) -> GiRay {
    return GiRay(probe_ray.origin, 0.0, worldCacheProbeRayDirection(probe_ray), max(probe_ray.hitT, 0.0));
}

fn HashGridCache_ProbeSurface(probe_ray: WorldCacheProbeRay) -> GiSurface {
    let hit = GiHit(probe_ray.hitT, probe_ray.hitInstance, probe_ray.hitTriangle, probe_ray.hitBarycentrics, probe_ray.hitFrontFace != 0u);
    return giSurfaceAt(HashGridCache_ProbeRay(probe_ray), hit);
}
`;

/** The workgroup a 2D-split dispatch is on, counted in rows, as the platform's indirect arguments and `dispatchSplit` lay them out. */
const LINEAR_WORKGROUP_WGSL = /* wgsl */ `
fn HashGridCache_WorkgroupIndex(workgroup: vec3u, workgroups: vec3u) -> u32 {
    return workgroup.y * workgroups.x + workgroup.x;
}
`;

/** The shared head of every population kernel: the hash grid (group 1) and the ray data (group 2). */
function populationHeadWgsl(layout: HashGridCacheBufferLayout): string {
    return `${hashGridCacheWgsl(HASH_GRID_KERNEL_GROUP, layout)}\n${RAY_DATA_WGSL}\n${LINEAR_WORKGROUP_WGSL}`;
}

// ---- PopulateScreenProbes, the cache half ---------------------------------------------------------

/**
 * `PopulateScreenProbesHandleHit` (gi1.comp:985-1076). Capsaicin runs it
 * inside `PopulateScreenProbes`, right after the probe ray's closest hit; the
 * world-cache slot hands the traced rays over instead, so here it is a kernel
 * of its own over the probe rays, run after the screen cache's trace
 * (register D30). Dispatched indirectly on the probe-ray count.
 */
export function populateScreenProbesWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.populateScreenProbes;
    return /* wgsl */ `
${populationHeadWgsl(layout)}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.populateScreenProbes}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= g_ProbeRays.count) {
        return; // out of bounds
    }
    let probe_ray = g_ProbeRays.rays[did];
    // A miss is the screen cache's (PopulateScreenProbesHandleMiss reads the sky), and a ray the
    // screen cache lit itself carries the bypass flag: neither reaches the cache.
    if (probe_ray.hitT < 0.0 || (probe_ray.flags & WORLD_CACHE_PROBE_RAY_BYPASS) != 0u) {
        return;
    }

    let direction = worldCacheProbeRayDirection(probe_ray);
    var data: HashGridCache_Data;
    data.eye_position = g_HashGridCacheConstants.eye_position;
    data.hit_position = probe_ray.origin + (probe_ray.hitT * direction);
    data.direction    = direction;
    data.hit_distance = probe_ray.hitT;

    let inserted = HashGridCache_InsertCell(data);
    let tile_index = inserted.tile_index;
    let is_new_tile = inserted.is_new_tile;
    let cell_index = inserted.cell_index;

    if (cell_index != kGI1_InvalidId) {
        // Bump the cell's decay to the max. now that it's been 'touched'
        let previous_tile_decay = HashGridCache_TouchTile(tile_index);

        // We update the cell index for later passes. The hit itself stays in the probe ray,
        // which the visibility entry finds through its query index (register D30).
        let visibility_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0], 1u);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index, cell_index);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index, did);

        // requestLightSampleLocation(data.hit_position) bounds Capsaicin's light sampler grid
        // to the hits; the platform's light grid covers the scene's bounds instead (register D4).

        // If this cell is inside a new tile, we need to add the tile to the packed storage and clear its cells.
        if (is_new_tile) {
            HashGridCache_AddNewTile(tile_index);
        }

        // If we're the 1st invocation touching this cell (this frame), we want to clear the
        // scratch storage that'll be used for atomically updating the radiance.
        // The accumulation will be resolved in the 'UpdateTiles()' kernel to
        // avoid integer overflow.
        if (is_new_tile || previous_tile_decay != g_HashGridCacheConstants.frame_index) {
            let update_tile_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_UPDATETILECOUNTBUFFER], 1u);
            HashGridCache_SetRayListWord(g_HashGridCache_RayLists.update_tile_buffer, update_tile_index, tile_index);
        }
    }
}
`;
}

// ---- GenerateReservoirs ------------------------------------------------------------------------------

/**
 * `GenerateReservoirs` (gi1.comp:2400-2600), with Capsaicin's light
 * reservoir as the platform's one-light choice (register D4): Capsaicin's
 * `sampleLightsListCone` resamples `kReservoir_SampleCount` lights streamed
 * from its light sampler grid into one reservoir (a light, a point on it and
 * its weight W); `giDirectChooseLight` draws one of the platform's candidates
 * for the hit (the directional lights, then the point and spot lights or the
 * light grid's picks for the hit's cell) in proportion to the luminance of its
 * unshadowed irradiance times its share, and its weight is share over that
 * probability: the same role as W, `light_radiance × W` estimating the
 * candidates' sum. The point on the light is two random numbers drawn here,
 * as Capsaicin's `sampleParams` are, and used by `PopulateCells`.
 *
 * Built with the platform's hit-shading module (group 0).
 */
export function generateReservoirsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.generateReservoirs;
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${populationHeadWgsl(layout)}
${PROBE_RAY_SURFACE_WGSL}

const HASHGRIDCACHE_LIGHT_GROUPS: u32 = ${ALL_LIGHT_GROUPS}u;

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.generateReservoirs}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        return; // out of bounds
    }

    // Load our visibility sample, and reconstruct the world-space position and normal
    // (the surface facing the ray's side, as Capsaicin flips its normal by the hit's face).
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did);
    let probe_ray = g_ProbeRays.rays[query_index];
    let surface = HashGridCache_ProbeSurface(probe_ray);

    // Patch the screen probes with some emissivity information:
    // We bypass the hash-grid cache entirely here, as adding emissive information to the cells
    // effectively enlarges the area light (due to the spatial nature of the grid) and leads to
    // light leaks and generally poorer visuals.
    if (g_HashGridCacheConstants.use_direct_lighting != 0u && surface.frontFace && dot(surface.emission, surface.emission) > 0.0) {
        ScreenProbes_AccumulateRadiance(query_index, surface.emission);
        return; // do not continue past an emissive surface
    }

    // Temporal radiance feedback (gi1.comp:2454-2496) runs only with gi1_use_temporal_feedback,
    // off by default (gi1.h:54): not part of the baseline.

    // Sample new lights. Capsaicin forces a diffuse material on this bounce (gi1.comp:2506-2510);
    // the platform's materials are diffuse already (register D8).
    var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);
    let reservoir = giDirectChooseLight(surface, HASHGRIDCACHE_LIGHT_GROUPS, rcRand(&random));
    let sample_params = rcRand2(&random);

    // Resolve cells (no reservoir only discards nee, not second bounce contrib we need to resolve into screen probes)
    let resolve_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_RESOLVECOUNTBUFFER], 1u);

    // Here, we check that the ray has traversed at least a full hash cell
    // before hitting something.
    // If it hasn't, we bypass the filtering from the cache (although we
    // still accumulate the contribution into the cell).
    // This avoids possible light leaks where a bounce close to a wall shares the cell of a
    // bright event on its other side (the diagram at gi1.comp:2533-2551).
    // Capsaicin measures from the probe's pixel on the depth buffer; the probe ray's origin is
    // that point pushed off the surface (register D17).
    let cell_size  = HashGridCache_GetCellSize(surface.position);
    let ray_length = distance(probe_ray.origin, surface.position);

    let bypass_cache = ray_length < cell_size;
    let bypass_bit = select(0u, HASHGRIDCACHE_BYPASS_BIT, bypass_cache);

    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.resolve_buffer, resolve_index, did | bypass_bit);

    if (reservoir.light == GI_DIRECT_NO_LIGHT) {
        return;
    }

    // Append our resampled shadow ray
    let ray_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER], 1u);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_ray_buffer, ray_index, did | bypass_bit);   // compact the surviving shadow rays

    // Write the results out to memory: the reservoir (packReservoir). The ray's origin and the
    // material (g_Reservoir_IndirectSampleBuffer, ...MaterialBuffer) are read again from the
    // probe ray by PopulateCells (register D30).
    let reservoir_word = g_HashGridCache_RayLists.reservoir_buffer + ${RESERVOIR_WORDS}u * ray_index;
    HashGridCache_SetRayListWord(reservoir_word, 0u, reservoir.light);
    HashGridCache_SetRayListWord(reservoir_word, 1u, bitcast<u32>(reservoir.weight));
    HashGridCache_SetRayListWord(reservoir_word, 2u, pack2x16unorm(sample_params));
}
`;
}
// ---- PopulateCells -----------------------------------------------------------------------------------

/**
 * `PopulateCells` (gi1.comp:2040-2106) with `PopulateCellsTraceRay`,
 * `PopulateCellsHandleHit` and `PopulateCellsHandleMiss` (gi1.comp:1988-2038):
 * one shadow ray through the platform's `giTraceAny` toward the reservoir's
 * light; if nothing blocks it, the light reflected toward the probe ray's
 * origin is `BRDF × light_radiance × W`, Lambert's albedo / π with the cosine
 * in the platform's irradiance (register D8: Capsaicin evaluates GGX at
 * roughness 1 with its diffuse compensation here). Added, quantised, into the
 * cell's update scratch; a ray that bypasses the cache also gets it directly.
 *
 * Built with the platform's hit-shading module (group 0).
 */
export function populateCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.populateCells;
    return /* wgsl */ `
${populationHeadWgsl(layout)}
${PROBE_RAY_SURFACE_WGSL}

// INV_PI (math/math_constants.hlsl).
const HASHGRIDCACHE_INV_PI: f32 = 0.318309886183791;

// GIDenoiser_RemoveNaNs (gi_denoiser.hlsl:74-81): maps the colour to [0, 1), clamps it and maps
// it back, so a NaN becomes 0 and anything past about 1e4 is held there. HLSL's saturate turns a
// NaN into 0; WGSL's clamp leaves that to the implementation, so the NaN test is on the bits.
fn GIDenoiser_RemoveNaNs(color_in: vec3f) -> vec3f {
    var color = color_in / (1.0 + color_in);
    let is_nan = (bitcast<vec3u>(color) & vec3u(0x7fffffffu)) > vec3u(0x7f800000u);
    color = select(saturate(color), vec3f(0.0), is_nan);
    color /= max(1.0 - color, vec3f(1e-4));
    return color;
}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.populateCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER])) {
        return; // out of bounds
    }

    let visibility_ray   = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_ray_buffer, did);
    let visibility_index = visibility_ray & ~HASHGRIDCACHE_BYPASS_BIT;

    // Reconstruct world-space position and normal
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index);
    let probe_ray = g_ProbeRays.rays[query_index];
    let surface = HashGridCache_ProbeSurface(probe_ray);

    // Retrieve the light sample that we should use for our shadow ray
    let reservoir_word = g_HashGridCache_RayLists.reservoir_buffer + ${RESERVOIR_WORDS}u * did;
    let light = HashGridCache_RayListWord(reservoir_word, 0u);
    let light_weight = bitcast<f32>(HashGridCache_RayListWord(reservoir_word, 1u));
    let sample_params = unpack2x16unorm(HashGridCache_RayListWord(reservoir_word, 2u));

    // Decode the light sample from our reservoir, and traverse the BVH
    let light_ray = giDirectLightRay(surface, light, sample_params);
    let occluded = light_ray.castsShadows && giTraceAny(light_ray.shadowRay);

    // PopulateCellsHandleHit: an occluded light leaves nothing. PopulateCellsHandleMiss: the
    // BRDF toward the probe ray's origin, times the light's radiance and the reservoir's weight.
    var lighting = vec3f(0.0);
    if (!occluded) {
        lighting = surface.albedo * HASHGRIDCACHE_INV_PI * light_ray.irradiance * light_weight;
    }

    // And update the hash-grid cell payload
    let cell_index         = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index);
    let quantized_radiance = HashGridCache_QuantizeRadiance(lighting);
    let scratch            = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;

    if (dot(lighting, lighting) > 0.0) {
        atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 0u], quantized_radiance.x);
        atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 1u], quantized_radiance.y);
        atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 2u], quantized_radiance.z);
    }
    atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 3u], quantized_radiance.w);

    // In some cases, we want to bypass the cache to avoid light leaks;
    // so we simply patch the screen probes directly and invalidate the
    // cache connection.
    let bypass_cache = (visibility_ray >> 31u) != 0u;

    if (bypass_cache) {
        ScreenProbes_AccumulateRadiance(query_index, GIDenoiser_RemoveNaNs(lighting));
    }
}
`;
}

// ---- GenerateUpdateTilesDispatch and UpdateTiles -----------------------------------------------

/**
 * Tiles one `UpdateTiles` workgroup covers: `UPDATE_TILES_GROUP_SIZE /
 * num_cells_per_tile_mip0` (`GenerateUpdateTilesDispatch`, gi1.comp:2129-2140,
 * whose arguments the platform's indirect-arguments writer makes from the
 * update-tile count). Capsaicin assumes the 8 × 8 group holds whole tiles
 * (gi1.comp:2116-2118), so the tile's side must divide 8.
 */
export function updateTilesPerWorkgroup(tileCellRatio: number): number {
    const side = POPULATION_WORKGROUP_SIZE.updateTilesSide;
    if (!Number.isInteger(tileCellRatio) || tileCellRatio < 1 || side % tileCellRatio !== 0) {
        throw new RangeError(`@pmndrs/ddgi: UpdateTiles covers whole tiles in an ${side} × ${side} workgroup, so hashGridCacheTileCellRatio must divide ${side}, got ${tileCellRatio}`);
    }
    return (side * side) / (tileCellRatio * tileCellRatio);
}

/**
 * `UpdateTiles` (gi1.comp:2142-2348). One difference in form: Capsaicin's
 * threads past the update list return before the group's barriers; WGSL
 * needs every thread at a barrier, so they skip the work instead (the same
 * result).
 */
export function updateTilesWgsl(layout: HashGridCacheBufferLayout): string {
    const side = POPULATION_WORKGROUP_SIZE.updateTilesSide;
    return /* wgsl */ `
${populationHeadWgsl(layout)}

const UPDATE_TILES_GROUP_X: u32 = ${side}u;
const UPDATE_TILES_GROUP_Y: u32 = ${side}u;

var<workgroup> lds_UpdateTiles_ValueBuffer: array<array<vec2u, ${side}>, ${side}>;
var<workgroup> lds_UpdateTiles_ValueIndirectBuffer: array<array<vec2u, ${side}>, ${side}>;

// Temporal accumulation of one cell (gi1.comp:2162-2177 for the direct values, 2179-2195 for
// the indirect ones): the history and this frame's samples, each as a mean, blended with weight
// 1 / sample_count, the count capped at max_count; stored times the count, which later passes
// read as a hint for how much to prefilter.
fn UpdateTiles_Accumulate(previous_radiance: vec4f, new_samples: vec4f, max_count: f32) -> vec4f {
    var radiance = previous_radiance;
    var new_radiance = new_samples;
    let sample_count = min(radiance.w + new_radiance.w, max_count);
    radiance /= max(radiance.w, 1.0);
    new_radiance /= max(new_radiance.w, 1.0);
    if (radiance.w <= 0.0) {
        radiance = new_radiance;
    } else {
        radiance = mix(radiance, new_radiance, 1.0 / sample_count);
    }
    radiance *= sample_count; // sample count is used as a hint for picking prefiltering amount
    return radiance;
}

// This frame's quantised samples of a cell (an update scratch array at 'scratch'), recovered,
// and the scratch cleared for the next frame (gi1.comp:2164-2168, 2210-2224).
fn UpdateTiles_TakeScratch(scratch: u32) -> vec4f {
    let quantized = vec4u(atomicLoad(&g_HashGridCache_UpdateScratch[scratch + 0u]),
                          atomicLoad(&g_HashGridCache_UpdateScratch[scratch + 1u]),
                          atomicLoad(&g_HashGridCache_UpdateScratch[scratch + 2u]),
                          atomicLoad(&g_HashGridCache_UpdateScratch[scratch + 3u]));
    // Clear scratch
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 0u], 0u);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 1u], 0u);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 2u], 0u);
    atomicStore(&g_HashGridCache_UpdateScratch[scratch + 3u], 0u);
    return HashGridCache_RecoverRadiance(quantized);
}

// The box filter of a coarser level (gi1.comp:2230-2348): the sum of a 2 x 2 block of the level
// below, read from the group's memory 'step' threads apart.
fn UpdateTiles_SumDirect(position: vec2u, step: u32) -> vec4f {
    var radiance = vec4f(0.0);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueBuffer[position.x + 0u][position.y + 0u]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueBuffer[position.x + step][position.y + 0u]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueBuffer[position.x + 0u][position.y + step]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueBuffer[position.x + step][position.y + step]);
    return radiance;
}

fn UpdateTiles_SumIndirect(position: vec2u, step: u32) -> vec4f {
    var radiance = vec4f(0.0);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueIndirectBuffer[position.x + 0u][position.y + 0u]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueIndirectBuffer[position.x + step][position.y + 0u]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueIndirectBuffer[position.x + 0u][position.y + step]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueIndirectBuffer[position.x + step][position.y + step]);
    return radiance;
}

// One coarser level (MIP 1, 2 or 3): a thread on the level's grid ('step' threads apart) sums
// its 2 x 2 block of the level below and writes it to the tile and to the group's memory, where
// the next level reads it.
fn UpdateTiles_FilterLevel(group_thread_id: vec2u, tile_cell_offset: vec2u, tile_index: u32, mip_level: u32, step: u32) {
    let cell_index = HashGridCache_CellIndex(tile_cell_offset, tile_index, mip_level);

    // Box filter
    let packed_radiance = HashGridCache_PackRadiance(UpdateTiles_SumDirect(group_thread_id, step));
    lds_UpdateTiles_ValueBuffer[group_thread_id.x][group_thread_id.y] = packed_radiance;
    g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = packed_radiance;

    // USE_MULTI_BOUNCE
    let packed_radiance_indirect = HashGridCache_PackRadiance(UpdateTiles_SumIndirect(group_thread_id, step));
    lds_UpdateTiles_ValueIndirectBuffer[group_thread_id.x][group_thread_id.y] = packed_radiance_indirect;
    g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index] = packed_radiance_indirect;
}

@compute @workgroup_size(${side}, ${side})
fn ${POPULATION_ENTRY_POINT.updateTiles}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
) {
    let group_id = HashGridCache_WorkgroupIndex(workgroup, workgroups);
    let group_thread_id = local_id.xy;

    // A group cover from 1 tile/subgroup (8x8) to 64 tiles/subgroups (1x1). Capsaicin's
    // subgroup_id, subgroup_thread_id and subgroup_size: a tile of the group, not a wave.
    let size_tile_mip0   = g_HashGridCacheConstants.size_tile_mip0;
    let tile_in_group    = group_thread_id / size_tile_mip0;
    let tile_cell_offset = group_thread_id % size_tile_mip0;
    let tiles_in_group   = vec2u(UPDATE_TILES_GROUP_X, UPDATE_TILES_GROUP_Y) / size_tile_mip0;
    let update_tile_index = group_id * tiles_in_group.x * tiles_in_group.y +
                            tile_in_group.y * tiles_in_group.x +
                            tile_in_group.x;

    let is_updated = update_tile_index < atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_UPDATETILECOUNTBUFFER]);
    var tile_index = 0u;

    // MIP 0
    if (is_updated) {
        tile_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.update_tile_buffer, update_tile_index);
        let cell_index = HashGridCache_CellIndex(tile_cell_offset, tile_index, 0u);

        // Temporal accumulation
        let direct_radiance = UpdateTiles_Accumulate(
            HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index]),
            UpdateTiles_TakeScratch(HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index),
            g_HashGridCacheConstants.max_sample_count);
        // USE_MULTI_BOUNCE
        let indirect_radiance = UpdateTiles_Accumulate(
            HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index]),
            UpdateTiles_TakeScratch(HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER + 4u * cell_index),
            g_HashGridCacheConstants.max_multibounce_sample_count);

        // Pack
        let packed_direct_radiance = HashGridCache_PackRadiance(direct_radiance);
        lds_UpdateTiles_ValueBuffer[group_thread_id.x][group_thread_id.y] = packed_direct_radiance;
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = packed_direct_radiance;

        let packed_indirect_radiance = HashGridCache_PackRadiance(indirect_radiance);
        lds_UpdateTiles_ValueIndirectBuffer[group_thread_id.x][group_thread_id.y] = packed_indirect_radiance;
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index] = packed_indirect_radiance;
    }

    // MIP 1
    workgroupBarrier();
    if (is_updated && g_HashGridCacheConstants.size_tile_mip1 > 0u && all((group_thread_id % 2u) == vec2u(0u))) {
        UpdateTiles_FilterLevel(group_thread_id, tile_cell_offset, tile_index, 1u, 1u);
    }

    // MIP 2
    workgroupBarrier();
    if (is_updated && g_HashGridCacheConstants.size_tile_mip2 > 0u && all((group_thread_id % 4u) == vec2u(0u))) {
        UpdateTiles_FilterLevel(group_thread_id, tile_cell_offset, tile_index, 2u, 2u);
    }

    // MIP 3
    workgroupBarrier();
    if (is_updated && g_HashGridCacheConstants.size_tile_mip3 > 0u && all((group_thread_id % 8u) == vec2u(0u))) {
        UpdateTiles_FilterLevel(group_thread_id, tile_cell_offset, tile_index, 3u, 4u);
    }
}
`;
}

// ---- ResolveCells ------------------------------------------------------------------------------------

/**
 * `ResolveCells` (gi1.comp:2350-2385), the `USE_MULTI_BOUNCE` path: each
 * entry of the resolve list without the bypass bit adds its cell's filtered
 * direct radiance (a coarser level while the cell has fewer than
 * `max_sample_count` samples, `HashGridCache_FilteredRadianceDirect`) and its
 * filtered indirect radiance to its probe ray.
 */
export function resolveCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.resolveCells;
    return /* wgsl */ `
${populationHeadWgsl(layout)}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.resolveCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_RESOLVECOUNTBUFFER])) {
        return; // out of bounds
    }

    let visibility_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.resolve_buffer, did);

    if ((visibility_index >> 31u) != 0u) {
        return; // do not use filtered radiance
    }

    let cell_index  = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index);
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index);

    let direct_radiance = HashGridCache_FilteredRadianceDirect(cell_index);
    var radiance = (direct_radiance.rgb / max(direct_radiance.w, 1.0));
    // USE_MULTI_BOUNCE
    let indirect_radiance = HashGridCache_FilteredRadianceIndirect(cell_index);
    radiance += (indirect_radiance.rgb / max(indirect_radiance.w, 1.0));
    ScreenProbes_AccumulateRadiance(query_index, radiance);
}
`;
}
