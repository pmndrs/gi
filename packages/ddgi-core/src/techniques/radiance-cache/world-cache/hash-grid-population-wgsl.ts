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
 * And GI-1.2's multibounce (RC3.3; `USE_MULTI_BOUNCE`, gi1_use_multibounce
 * true by default, gi1.h:55), in Capsaicin's order (gi1.cpp:2363-2512):
 *
 * - `PopulateMultibounceCells` (gi1.comp:1761-1946), after the insert: 70% of
 *   the first-bounce hits stop there (`discard_multibounce_ray_probability`);
 *   the rest trace one cosine-sampled ray (the second vertex forced diffuse,
 *   Lambert here, register D8) and insert its hit as a second-bounce entry of
 *   the visibility list (after the first-bounce ones), with the brdf and pdf
 *   it was drawn with (`g_HashGridCache_MultibounceInfoBuffer`).
 * - `GenerateMultibounceReservoirs` (gi1.comp:2601-2695), after
 *   `GenerateReservoirs`: one light per second-bounce hit, its shadow ray on
 *   the same list as the first bounce's, so `PopulateCells` lights both.
 * - `UpdateMultibounceCells` (gi1.comp:1948-1983), after `UpdateTiles`: each
 *   second-bounce hit's filtered direct radiance × brdf / pdf / (1 - 0.7)
 *   goes into its first-bounce cell's indirect update scratch. `UpdateTiles`
 *   has already run this frame, so the next frame's takes it: the indirect
 *   values lag the direct ones by a frame ("using last frame", gi1.cpp:2505).
 *
 * With the multibounce on, the resolve list holds every non-emissive hit and
 * `UpdateTiles` and `ResolveCells` carry the indirect values. With it off
 * (Capsaicin's single-bounce build), the three passes don't run and the
 * resolve list holds only the hits that drew a light, as Capsaicin's
 * `ResolveCells` then reads its shadow-ray list (gi1.comp:2353-2360).
 *
 * How the passes find their data (register D30): as in Capsaicin, the insert
 * copies each hit into the visibility list (`g_HashGridCache_VisibilityBuffer`,
 * `HashGridCache_PackVisibility`), with its probe ray's index
 * (`VisibilityQueryBuffer`) and its cell; ours also keeps the ray (origin,
 * direction, length), from which `GenerateReservoirs` measures the short-ray
 * test. `GenerateReservoirs` and `PopulateCells` rebuild the surface from the
 * copy with the platform's `giSurfaceAt`, and don't keep Capsaicin's
 * per-shadow-ray surface and material lists. The lists live in one packed
 * buffer (register D6) at the word offsets of `HashGridCacheRayLists`.
 *
 * Which buffers each pass binds ({@link POPULATION_BINDINGS}): the two that
 * light hits bind the platform's hit shading (5 storage buffers: the
 * tracer's 3 and the scene copy's 2) and so have 3 of the platform's 8 left
 * (`MIN_STORAGE_BUFFERS_PER_STAGE`): the cache words, the probe rays'
 * radiance slots and the per-ray lists. That is why the insert copies the hit
 * (they don't bind the probe rays) and why the counters share a buffer with
 * the update scratch (`capacity.ts`).
 */

import { CAPSAICIN_WGSL } from '../wgsl/capsaicin.ts';
import type { HashGridCacheBufferLayout } from './capacity.ts';
import {
    HASH_GRID_CACHE_ALL_BUFFERS,
    HASH_GRID_CACHE_WORDS_ONLY,
    HASH_GRID_KERNEL_GROUP,
    hashGridCacheWgsl,
    type HashGridCacheBindingSet,
} from './hash-grid-wgsl.ts';
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
    /** Lever 1 (`cellShading: 'budget'`): the age-class histogram and its cursors (`CELL_SHADING_CLASS_WORD`). Storage. */
    cellShadingClasses: 4,
} as const;

/**
 * Which of the ray-data group's buffers a pass binds beside the list offsets
 * and the per-ray lists, which every pass binds. The layout entries
 * ({@link rayDataGroupEntries}) and the WGSL follow it.
 */
export interface RayDataBindingSet {
    /** The screen cache's probe rays: only the insert reads them; it copies each hit into the visibility list. */
    readonly probeRays: boolean;
    /** The probe rays' radiance slots: the passes that add light to a probe ray. */
    readonly radiance: boolean;
    /** Lever 1's age-class histogram and cursors: the passes that rank and choose the cells to light. */
    readonly cellShadingClasses?: boolean;
}

/** Layout entries of the ray-data group for `set`: a uniform and one to three storage buffers. */
export function rayDataGroupEntries(set: RayDataBindingSet): GPUBindGroupLayoutEntry[] {
    const entries: GPUBindGroupLayoutEntry[] = [{ binding: RAY_DATA_BINDING.rayLists, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } }];
    if (set.probeRays) entries.push({ binding: RAY_DATA_BINDING.probeRays, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } });
    if (set.radiance) entries.push({ binding: RAY_DATA_BINDING.radiance, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } });
    entries.push({ binding: RAY_DATA_BINDING.listWords, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } });
    if (set.cellShadingClasses) entries.push({ binding: RAY_DATA_BINDING.cellShadingClasses, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } });
    return entries;
}

/** The buffers one population pass binds: its hash-grid set (group 1) and its ray-data set (group 2). */
export interface PopulationBindings {
    readonly cache: HashGridCacheBindingSet;
    readonly rayData: RayDataBindingSet;
}

/**
 * What each population pass binds. `GenerateReservoirs`, `PopulateCells` and
 * `PopulateMultibounceCells` also bind the platform's hit shading (5 storage
 * buffers), so theirs come to exactly the platform's 8: the cache words and
 * the lists, with the radiance slots for the first two and the cell values
 * for the third (it inserts cells, and a new tile's values are cleared).
 * `GenerateMultibounceReservoirs` shades but writes no probe ray: 7.
 */
export const POPULATION_BINDINGS = {
    /** Inserts cells (and clears a new tile's values), and copies each hit from the probe rays. */
    populateScreenProbes: { cache: HASH_GRID_CACHE_ALL_BUFFERS, rayData: { probeRays: true, radiance: false } },
    /** Counters and lists; an emissive hit's emission into its radiance slot. */
    generateReservoirs: { cache: HASH_GRID_CACHE_WORDS_ONLY, rayData: { probeRays: false, radiance: true } },
    /** The update scratch (in the cache words); a bypassing ray's light into its radiance slot. */
    populateCells: { cache: HASH_GRID_CACHE_WORDS_ONLY, rayData: { probeRays: false, radiance: true } },
    /** The update scratch into the cell values. */
    updateTiles: { cache: HASH_GRID_CACHE_ALL_BUFFERS, rayData: { probeRays: false, radiance: false } },
    /** The cell values into the radiance slots. */
    resolveCells: { cache: HASH_GRID_CACHE_ALL_BUFFERS, rayData: { probeRays: false, radiance: true } },
    /** Traces from the first-bounce hits and inserts the second-bounce cells (clearing a new tile's values). */
    populateMultibounceCells: { cache: HASH_GRID_CACHE_ALL_BUFFERS, rayData: { probeRays: false, radiance: false } },
    /** Counters and lists: a light per second-bounce hit. */
    generateMultibounceReservoirs: { cache: HASH_GRID_CACHE_WORDS_ONLY, rayData: { probeRays: false, radiance: false } },
    /** The second-bounce cells' values into the first-bounce cells' indirect scratch; a bypassing ray's into its radiance slot. */
    updateMultibounceCells: { cache: HASH_GRID_CACHE_ALL_BUFFERS, rayData: { probeRays: false, radiance: true } },
} as const satisfies Record<string, PopulationBindings>;

/**
 * Bytes of `HashGridCacheRayLists`: thirteen words (lever 1 added two), the
 * uniform buffer rounded up to 16 bytes.
 */
export const RAY_LISTS_UNIFORM_BYTES = 64;

/** Words of a second-bounce entry's brdf and pdf (`HashGridCache_PackBRDF`: four halves). */
export const MULTIBOUNCE_INFO_WORDS = 2;

/**
 * Words of one visibility entry (`g_HashGridCache_VisibilityBuffer`), at
 * these offsets. Capsaicin's `HashGridCache_PackVisibility` keeps the hit in
 * four (instance with the back-face bit, geometry, primitive, barycentrics as
 * halves; `hash_grid_cache.hlsl:379-384`); ours keeps the barycentrics as
 * f32, as the probe ray has them, and also the ray, whose origin the
 * short-ray test measures from (register D30).
 */
export const VISIBILITY_WORD = {
    /** The ray's origin, three f32. */
    origin: 0,
    /** The ray's direction as halves, two words (as `WorldCacheProbeRay.direction`). */
    direction: 3,
    /** The hit's distance along the ray, f32. */
    hitDistance: 5,
    /** The instance slot, with {@link VISIBILITY_BACK_FACE_BIT} set for a back-face hit (Capsaicin's packing). */
    instance: 6,
    /** The triangle in the instance's mesh (Capsaicin's `primitive_index`). */
    triangle: 7,
    /** The barycentric weights of vertices 1 and 2, two f32. */
    barycentrics: 8,
} as const;

export const VISIBILITY_WORDS = 10;

/** The top bit of a visibility entry's instance word: the hit is on a back face (`HashGridCache_PackVisibility`). */
export const VISIBILITY_BACK_FACE_BIT = 0x80000000;

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
    /** PopulateMultibounceCellsMain (gi1.comp:1942): 32. */
    populateMultibounceCells: 32,
    /** GenerateMultibounceReservoirs (gi1.comp:2602): 64. */
    generateMultibounceReservoirs: 64,
    /** UpdateMultibounceCellsMain (gi1.comp:1985): 32. */
    updateMultibounceCells: 32,
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
    populateMultibounceCells: 'PopulateMultibounceCells',
    generateMultibounceReservoirs: 'GenerateMultibounceReservoirs',
    updateMultibounceCells: 'UpdateMultibounceCells',
} as const;

/** Where each per-ray list starts in the packed list words, for a screen of `maxRayCount` probe rays. */
export interface HashGridCacheRayListLayout {
    /** `max_ray_count` (gi1.cpp:129): the probe rays the screen can have. */
    readonly maxRayCount: number;
    /** `max_combined_ray_count` (gi1.cpp:637): with multibounce, room for a second-bounce hit per ray. */
    readonly maxCombinedRayCount: number;
    /** Word offsets. */
    readonly updateTileBuffer: number;
    /** `VISIBILITY_WORDS` per entry. */
    readonly visibilityBuffer: number;
    readonly visibilityCellBuffer: number;
    readonly visibilityQueryBuffer: number;
    readonly visibilityRayBuffer: number;
    readonly resolveBuffer: number;
    readonly reservoirBuffer: number;
    /** `g_HashGridCache_MultibounceInfoBuffer`: {@link MULTIBOUNCE_INFO_WORDS} per second-bounce entry, `max_ray_count` of them. */
    readonly multibounceInfoBuffer: number;
    /** Ours (register D40): per first-bounce entry, 1 when its probe ray bypasses the cache's filtered values. */
    readonly visibilityBypassBuffer: number;
    /** Lever 1 (`budget`; empty in `perHit`): the visibility entry each cell in use kept this frame, one per cell. */
    readonly usedCellBuffer: number;
    /** Lever 1: the kept entries of the cells to light, most urgent first (`max_ray_count`; empty in `perHit`). */
    readonly chosenCellBuffer: number;
    readonly bytes: number;
}

/**
 * The lists Capsaicin sizes from the screen (`ensureMemoryIsAllocated`,
 * gi1.cpp:637-695): the update tiles (at most one per cell), the visibility
 * entries, cell, query and ray lists (`max_combined_ray_count` each), the resolve list
 * (`max_ray_count`, first bounce only), a reservoir per visibility ray
 * (`g_Reservoir_IndirectSampleReservoirBuffer`), the second-bounce entries'
 * brdf and pdf (`max_ray_count`, gi1.cpp:688; none without multibounce) and,
 * ours, a bypass flag per first-bounce entry (`max_ray_count`, register D40).
 * Without multibounce the resolve list stays: ours lists the lit hits there
 * where Capsaicin's `ResolveCells` reads the shadow-ray list (register D42).
 */
export function hashGridCacheRayListLayout(maxRayCount: number, numCells: number, multibounce: boolean, cellShading = false): HashGridCacheRayListLayout {
    // With USE_MULTI_BOUNCE (Capsaicin's default, gi1.h:55) bounce 0 and bounce 1 share the
    // visibility lists; without it they hold the first bounce alone (gi1.cpp:637).
    const maxCombinedRayCount = multibounce ? 2 * maxRayCount : maxRayCount;
    const updateTileBuffer = 0;
    const visibilityBuffer = updateTileBuffer + Math.min(maxCombinedRayCount, numCells);
    const visibilityCellBuffer = visibilityBuffer + VISIBILITY_WORDS * maxCombinedRayCount;
    const visibilityQueryBuffer = visibilityCellBuffer + maxCombinedRayCount;
    const visibilityRayBuffer = visibilityQueryBuffer + maxCombinedRayCount;
    const resolveBuffer = visibilityRayBuffer + maxCombinedRayCount;
    const reservoirBuffer = resolveBuffer + maxRayCount;
    const multibounceInfoBuffer = reservoirBuffer + RESERVOIR_WORDS * maxCombinedRayCount;
    const visibilityBypassBuffer = multibounceInfoBuffer + (multibounce ? MULTIBOUNCE_INFO_WORDS * maxRayCount : 0);
    // Lever 1's two lists: at most one entry per first-bounce hit each (a cell in use has a hit).
    const usedCellBuffer = visibilityBypassBuffer + maxRayCount;
    const chosenCellBuffer = usedCellBuffer + (cellShading ? maxRayCount : 0);
    const words = chosenCellBuffer + (cellShading ? maxRayCount : 0);
    return {
        maxRayCount,
        maxCombinedRayCount,
        updateTileBuffer,
        visibilityBuffer,
        visibilityCellBuffer,
        visibilityQueryBuffer,
        visibilityRayBuffer,
        resolveBuffer,
        reservoirBuffer,
        multibounceInfoBuffer,
        visibilityBypassBuffer,
        usedCellBuffer,
        chosenCellBuffer,
        bytes: Math.max(words, 1) * 4,
    };
}

/**
 * The ray-data group's declarations for `set` ({@link RayDataBindingSet}):
 * the list offsets and the per-ray lists always, with the visibility entries'
 * loader; the probe rays and the insert's copy into the visibility list, and
 * the radiance slots and Capsaicin's screen-probe helper, when bound.
 */
function rayDataWgsl(set: RayDataBindingSet): string {
    const probeRays = /* wgsl */ `
@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.probeRays}) var<storage, read> g_ProbeRays: WorldCacheProbeRays;

// A probe ray's hit into g_HashGridCache_VisibilityBuffer, with the ray beside it (register D30).
fn HashGridCache_StoreProbeRayVisibility(visibility_index: u32, probe_ray: WorldCacheProbeRay) {
    HashGridCache_StoreVisibility(visibility_index, probe_ray.origin, probe_ray.direction, probe_ray.hitT,
                                  probe_ray.hitInstance, probe_ray.hitFrontFace != 0u, probe_ray.hitTriangle, probe_ray.hitBarycentrics);
}
`;
    const radiance = /* wgsl */ `
// g_ScreenProbes_ProbeSpawnRadianceBuffer: each probe ray's radiance and hit distance as four halves.
@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.radiance}) var<storage, read_write> g_ScreenProbes_ProbeSpawnRadianceBuffer: array<vec2u>;

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
`;
    return /* wgsl */ `
${WORLD_CACHE_PROBE_RAYS_WGSL}

// Where each per-ray list starts in g_HashGridCache_RayListWords (gi1.cpp:637-695), in words.
struct HashGridCacheRayLists {
    max_ray_count: u32,
    max_combined_ray_count: u32,
    update_tile_buffer: u32,        // g_HashGridCache_UpdateTileBuffer: tiles first touched this frame
    visibility_buffer: u32,         // g_HashGridCache_VisibilityBuffer: each visibility entry's hit and ray (${VISIBILITY_WORDS} words)
    visibility_cell_buffer: u32,    // g_HashGridCache_VisibilityCellBuffer: each visibility entry's cell
    visibility_query_buffer: u32,   // g_HashGridCache_VisibilityQueryBuffer: each visibility entry's probe ray
    visibility_ray_buffer: u32,     // g_HashGridCache_VisibilityRayBuffer: entries with a shadow ray (top bit: bypass)
    resolve_buffer: u32,            // g_HashGridCache_ResolveBuffer: entries to resolve (top bit: bypass)
    reservoir_buffer: u32,          // g_Reservoir_IndirectSampleReservoirBuffer: 3 words per visibility ray
    multibounce_info_buffer: u32,   // g_HashGridCache_MultibounceInfoBuffer: brdf and pdf as four halves per second-bounce entry
    visibility_bypass_buffer: u32,  // ours (register D40): per first-bounce entry, 1 when its probe ray bypasses the cache
    used_cell_buffer: u32,          // ours, lever 1: the visibility entry each cell in use kept this frame
    chosen_cell_buffer: u32,        // ours, lever 1: the kept entries of the cells to light, most urgent first
}

@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.rayLists}) var<uniform> g_HashGridCache_RayLists: HashGridCacheRayLists;
@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.listWords}) var<storage, read_write> g_HashGridCache_RayListWords: array<atomic<u32>>;

const HASHGRIDCACHE_BYPASS_BIT: u32 = ${BYPASS_CACHE_BIT}u;
const HASHGRIDCACHE_VISIBILITY_BACK_FACE_BIT: u32 = ${VISIBILITY_BACK_FACE_BIT}u;

fn HashGridCache_RayListWord(offset: u32, index: u32) -> u32 {
    return atomicLoad(&g_HashGridCache_RayListWords[offset + index]);
}

fn HashGridCache_SetRayListWord(offset: u32, index: u32, value: u32) {
    atomicStore(&g_HashGridCache_RayListWords[offset + index], value);
}

// HashGridCache_PackVisibility (hash_grid_cache.hlsl:379-384) into g_HashGridCache_VisibilityBuffer,
// with the ray beside the hit (VISIBILITY_WORD in hash-grid-population-wgsl.ts). The direction is
// three halves, as WorldCacheProbeRay.direction holds it.
fn HashGridCache_StoreVisibility(visibility_index: u32, origin: vec3f, direction: vec2u, hit_distance: f32,
                                 instance_index: u32, is_front_face: bool, primitive_index: u32, barycentrics: vec2f) {
    let entry = g_HashGridCache_RayLists.visibility_buffer + ${VISIBILITY_WORDS}u * visibility_index;
    let back_face_bit = select(HASHGRIDCACHE_VISIBILITY_BACK_FACE_BIT, 0u, is_front_face);
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.origin}u + 0u, bitcast<u32>(origin.x));
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.origin}u + 1u, bitcast<u32>(origin.y));
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.origin}u + 2u, bitcast<u32>(origin.z));
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.direction}u + 0u, direction.x);
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.direction}u + 1u, direction.y);
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.hitDistance}u, bitcast<u32>(hit_distance));
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.instance}u, instance_index | back_face_bit);
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.triangle}u, primitive_index);
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.barycentrics}u + 0u, bitcast<u32>(barycentrics.x));
    HashGridCache_SetRayListWord(entry, ${VISIBILITY_WORD.barycentrics}u + 1u, bitcast<u32>(barycentrics.y));
}

// HashGridCache_Visibility (hash_grid_cache.hlsl:39-46), with the ray it was hit by.
struct HashGridCache_Visibility {
    origin: vec3f,
    direction: vec3f,
    hit_distance: f32,
    is_front_face: bool,
    instance_index: u32,
    primitive_index: u32,
    barycentrics: vec2f,
}

// HashGridCache_UnpackVisibility (hash_grid_cache.hlsl:387-397) of g_HashGridCache_VisibilityBuffer[visibility_index].
fn HashGridCache_LoadVisibility(visibility_index: u32) -> HashGridCache_Visibility {
    let entry = g_HashGridCache_RayLists.visibility_buffer + ${VISIBILITY_WORDS}u * visibility_index;
    let origin = vec3f(bitcast<f32>(HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.origin}u + 0u)),
                       bitcast<f32>(HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.origin}u + 1u)),
                       bitcast<f32>(HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.origin}u + 2u)));
    let direction = vec3f(unpack2x16float(HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.direction}u + 0u)),
                          unpack2x16float(HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.direction}u + 1u)).x);
    let instance_word = HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.instance}u);
    let barycentrics = vec2f(bitcast<f32>(HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.barycentrics}u + 0u)),
                             bitcast<f32>(HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.barycentrics}u + 1u)));
    return HashGridCache_Visibility(
        origin,
        direction,
        bitcast<f32>(HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.hitDistance}u)),
        (instance_word & HASHGRIDCACHE_VISIBILITY_BACK_FACE_BIT) == 0u,
        instance_word & ~HASHGRIDCACHE_VISIBILITY_BACK_FACE_BIT,
        HashGridCache_RayListWord(entry, ${VISIBILITY_WORD.triangle}u),
        barycentrics);
}
${set.probeRays ? probeRays : ''}${set.radiance ? radiance : ''}${set.cellShadingClasses ? CELL_SHADING_CLASSES_WGSL : ''}`;
}

/**
 * Lever 1's age-class histogram (`cellShading: 'budget'`), in its own small
 * buffer so the platform's scan can run over it from its start: per class
 * (`CELL_SHADING_CLASSES` of them) the cells in use of that class, scanned in
 * place into each class's first rank; then per class a cursor the choice
 * counts up as it hands out ranks.
 */
export const CELL_SHADING_CLASSES = 256;
export const CELL_SHADING_CLASS_WORD = {
    /** The histogram, then (after the scan) each class's first rank. */
    histogram: 0,
    /** Ranks handed out in each class so far. */
    cursor: CELL_SHADING_CLASSES,
} as const;
export const CELL_SHADING_CLASS_WORDS = 2 * CELL_SHADING_CLASSES;

const CELL_SHADING_CLASSES_WGSL = /* wgsl */ `
// Lever 1: the age-class histogram (scanned into first ranks), then the per-class cursors.
@group(${RAY_DATA_GROUP}) @binding(${RAY_DATA_BINDING.cellShadingClasses}) var<storage, read_write> g_HashGridCache_CellShadingClasses: array<atomic<u32>>;
const HASHGRIDCACHE_CELLSHADING_HISTOGRAM: u32 = ${CELL_SHADING_CLASS_WORD.histogram}u;
const HASHGRIDCACHE_CELLSHADING_CURSOR: u32 = ${CELL_SHADING_CLASS_WORD.cursor}u;
`;

/**
 * The surface a visibility entry's hit is on, rebuilt with the platform's
 * `giSurfaceAt` (for kernels built with the hit-shading module). Capsaicin
 * rebuilds it from its packed visibility the same way (`fetchVerticesNormUV`,
 * `interpolate`; gi1.comp:2412-2427, 2050-2062).
 */
export const VISIBILITY_SURFACE_WGSL = /* wgsl */ `
fn HashGridCache_VisibilitySurface(visibility: HashGridCache_Visibility) -> GiSurface {
    let ray = GiRay(visibility.origin, 0.0, visibility.direction, max(visibility.hit_distance, 0.0));
    let hit = GiHit(visibility.hit_distance, visibility.instance_index, visibility.primitive_index, visibility.barycentrics, visibility.is_front_face);
    return giSurfaceAt(ray, hit);
}
`;

/**
 * The light draw `GenerateReservoirs` and `GenerateMultibounceReservoirs`
 * share (both call `sampleLightsListCone<kReservoir_SampleCount>`,
 * gi1.comp:2513, 2659), on the platform's light choice (register D4). With
 * the light builder's `environment_light_enable` the sky is one of the
 * candidates, as Capsaicin's environment light is the first light of its
 * list (light_builder.cpp:131, 200-207) and so a candidate of every cell of
 * its light sampler (light_sampler_grid_stream_build.comp:120 streams light
 * 0 into each cell's first reservoir): its direction uniform over the
 * shading normal's hemisphere, pdf 1 / 2π (`sampleEnvironmentLight`,
 * light_sampling.hlsl:164-180, Capsaicin's default `Uniform` mode), weighed by
 * that direction's light (register D33 for what remains different).
 *
 * The random numbers: one for the draw, two for the point on a drawn light
 * (Capsaicin's `sampleParams`, which the reservoir keeps), then two for the
 * sky's direction, so a scene without a sky draws exactly as before. The
 * sky's numbers go through the reservoir's 16-bit packing before they are
 * used, so `PopulateCells` rebuilds the very direction that was weighed
 * (Capsaicin packs them as halves, `packReservoir`, reservoir.hlsl:545-551,
 * and rebuilds the direction from those, a hair off the one it weighed).
 */
export const LIGHT_DRAW_WGSL = /* wgsl */ `
struct HashGridCache_LightDraw {
    choice: GiDirectLightChoice,
    sample_params: vec2f,   // what the reservoir keeps: the point on the light, or the sky's direction
}

fn HashGridCache_DrawLight(surface: GiSurface, random: ptr<function, RcRandom>) -> HashGridCache_LightDraw {
    let draw_random = rcRand(random);
    let light_params = rcRand2(random);
    if (g_HashGridCacheConstants.use_environment_light == 0u) {
        return HashGridCache_LightDraw(giDirectChooseLight(surface, HASHGRIDCACHE_LIGHT_GROUPS, draw_random), light_params);
    }
    let sky_params = unpack2x16unorm(pack2x16unorm(rcRand2(random)));
    let choice = giDirectChooseLightWithSky(surface, HASHGRIDCACHE_LIGHT_GROUPS, draw_random, sky_params);
    return HashGridCache_LightDraw(choice, select(light_params, sky_params, choice.light == GI_DIRECT_SKY_LIGHT));
}
`;

/** INV_PI (math/math_constants.hlsl). */
export const INV_PI_WGSL = /* wgsl */ `
const HASHGRIDCACHE_INV_PI: f32 = 0.318309886183791;
`;

/** For the two passes that patch a bypassing probe ray directly (`PopulateCells`, `UpdateMultibounceCells`). */
export const REMOVE_NANS_WGSL = /* wgsl */ `
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
`;

/** The workgroup a 2D-split dispatch is on, counted in rows, as the platform's indirect arguments and `dispatchSplit` lay them out. */
const LINEAR_WORKGROUP_WGSL = /* wgsl */ `
fn HashGridCache_WorkgroupIndex(workgroup: vec3u, workgroups: vec3u) -> u32 {
    return workgroup.y * workgroups.x + workgroup.x;
}
`;

/** The shared head of a population kernel: the hash grid (group 1) and the ray data (group 2), each with the buffers the pass binds. */
export function populationHeadWgsl(layout: HashGridCacheBufferLayout, bindings: PopulationBindings): string {
    return `${hashGridCacheWgsl(HASH_GRID_KERNEL_GROUP, layout, bindings.cache)}\n${rayDataWgsl(bindings.rayData)}\n${LINEAR_WORKGROUP_WGSL}`;
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
${populationHeadWgsl(layout, POPULATION_BINDINGS.populateScreenProbes)}

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

        // We update the cell index for later passes (the visibility entry keeps the hit and,
        // ours, the ray it came along: register D30).
        let visibility_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0], 1u);
        HashGridCache_StoreProbeRayVisibility(visibility_index, probe_ray);
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
        // (Capsaicin's condition, with a once-per-frame check so a new tile is listed once: register D32.)
        if ((is_new_tile || previous_tile_decay != g_HashGridCacheConstants.frame_index) && HashGridCache_ListTileForUpdate(tile_index)) {
            let update_tile_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_UPDATETILECOUNTBUFFER], 1u);
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
${populationHeadWgsl(layout, POPULATION_BINDINGS.generateReservoirs)}
${VISIBILITY_SURFACE_WGSL}

const HASHGRIDCACHE_LIGHT_GROUPS: u32 = ${ALL_LIGHT_GROUPS}u;
${LIGHT_DRAW_WGSL}
// g_HashGridCache_ResolveBuffer[resolve_index] = entry, appended.
fn HashGridCache_AppendResolve(entry: u32) {
    let resolve_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_RESOLVECOUNTBUFFER], 1u);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.resolve_buffer, resolve_index, entry);
}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.generateReservoirs}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        return; // out of bounds
    }

    // Load our visibility sample, and reconstruct the world-space position and normal
    // (the surface facing the ray's side, as Capsaicin flips its normal by the hit's face).
    let visibility = HashGridCache_LoadVisibility(did);
    let surface = HashGridCache_VisibilitySurface(visibility);

    // Recover the ray origin: Capsaicin rebuilds it from the probe's pixel on the depth buffer
    // (gi1.comp:2429-2436, 2500); the visibility entry keeps the probe ray's origin, that point
    // pushed off the surface (registers D17, D30).
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did);

    // Patch the screen probes with some emissivity information:
    // We bypass the hash-grid cache entirely here, as adding emissive information to the cells
    // effectively enlarges the area light (due to the spatial nature of the grid) and leads to
    // light leaks and generally poorer visuals.
    if (g_HashGridCacheConstants.use_direct_lighting != 0u && surface.frontFace && dot(surface.emission, surface.emission) > 0.0) {
        ScreenProbes_AccumulateRadiance(query_index, surface.emission);
        // Ours (register D40): the probe ray takes nothing more, so no second bounce is added to it directly.
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_bypass_buffer, did, 0u);
        return; // do not continue past an emissive surface
    }

    // Temporal radiance feedback (gi1.comp:2454-2496) runs only with gi1_use_temporal_feedback,
    // off by default (gi1.h:54): not part of the baseline.

    // Sample new lights, the sky among them (LIGHT_DRAW_WGSL). Capsaicin forces a diffuse material
    // on this bounce (gi1.comp:2506-2510); the platform's materials are diffuse already (register D8).
    var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);
    let draw = HashGridCache_DrawLight(surface, &random);
    let reservoir = draw.choice;
    let sample_params = draw.sample_params;

    // Here, we check that the ray has traversed at least a full hash cell
    // before hitting something.
    // If it hasn't, we bypass the filtering from the cache (although we
    // still accumulate the contribution into the cell).
    // This avoids possible light leaks where a bounce close to a wall shares the cell of a
    // bright event on its other side (the diagram at gi1.comp:2533-2551).
    // Measured from the probe ray's origin, not Capsaicin's unpushed depth point (register D37).
    let cell_size  = HashGridCache_GetCellSize(surface.position);
    let ray_length = distance(visibility.origin, surface.position);

    let bypass_cache = ray_length < cell_size;
    let bypass_bit = select(0u, HASHGRIDCACHE_BYPASS_BIT, bypass_cache);

    // Ours (register D40): whether this probe ray bypasses the cache, by visibility entry, for
    // UpdateMultibounceCells, which adds the second bounce to such a ray directly.
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_bypass_buffer, did, select(0u, 1u, bypass_cache));

    // USE_MULTI_BOUNCE: resolve cells (no reservoir only discards nee, not second bounce contrib
    // we need to resolve into screen probes).
    let use_multibounce = g_HashGridCacheConstants.use_multibounce != 0u;
    if (use_multibounce) {
        HashGridCache_AppendResolve(did | bypass_bit);
    }

    if (reservoir.light == GI_DIRECT_NO_LIGHT) {
        return;
    }

    // Append our resampled shadow ray
    let ray_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER], 1u);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_ray_buffer, ray_index, did | bypass_bit);   // compact the surviving shadow rays

    // Without USE_MULTI_BOUNCE, ResolveCells reads the shadow-ray list (gi1.comp:2353-2360): only
    // the hits that drew a light are resolved. Ours lists the same entries on the resolve list.
    if (!use_multibounce) {
        HashGridCache_AppendResolve(did | bypass_bit);
    }

    // Write the results out to memory: the reservoir (packReservoir). The ray's origin and the
    // material (g_Reservoir_IndirectSampleBuffer, ...MaterialBuffer) are read again from the
    // visibility entry by PopulateCells (register D30).
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
${populationHeadWgsl(layout, POPULATION_BINDINGS.populateCells)}
${VISIBILITY_SURFACE_WGSL}

${INV_PI_WGSL}
${REMOVE_NANS_WGSL}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.populateCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER])) {
        return; // out of bounds
    }

    let visibility_ray   = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_ray_buffer, did);
    let visibility_index = visibility_ray & ~HASHGRIDCACHE_BYPASS_BIT;

    // Reconstruct world-space position and normal
    let surface = HashGridCache_VisibilitySurface(HashGridCache_LoadVisibility(visibility_index));
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index);

    // Retrieve the light sample that we should use for our shadow ray
    let reservoir_word = g_HashGridCache_RayLists.reservoir_buffer + ${RESERVOIR_WORDS}u * did;
    let light = HashGridCache_RayListWord(reservoir_word, 0u);
    let light_weight = bitcast<f32>(HashGridCache_RayListWord(reservoir_word, 1u));
    let sample_params = unpack2x16unorm(HashGridCache_RayListWord(reservoir_word, 2u));

    // Decode the light sample from our reservoir, and traverse the BVH. The platform's light ray
    // leaves along the geometric normal, as Capsaicin's (MakeRayInfoShadow), but by its own
    // offset, and takes the light's cosine on the shading normal where Capsaicin's BRDF takes it
    // on the geometric one (gi1.comp:2062-2063, 2073-2076, 2018; register D36). For the sky
    // (GI_DIRECT_SKY_LIGHT) the reservoir's sample_params are its direction, rebuilt about the
    // same normal GenerateReservoirs drew it about (Capsaicin's evaluateLightConeSampledAM,
    // light_evaluation.hlsl:1039-1051, at the sharpest level its solid angle gives,
    // gi1.comp:2071), and the shadow ray is unbounded (MakeRayInfoShadowDir, trace_ray.hlsl:147-150).
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
        atomicAdd(&g_HashGridCache_Words[scratch + 0u], quantized_radiance.x);
        atomicAdd(&g_HashGridCache_Words[scratch + 1u], quantized_radiance.y);
        atomicAdd(&g_HashGridCache_Words[scratch + 2u], quantized_radiance.z);
    }
    atomicAdd(&g_HashGridCache_Words[scratch + 3u], quantized_radiance.w);

    // In some cases, we want to bypass the cache to avoid light leaks;
    // so we simply patch the screen probes directly and invalidate the
    // cache connection.
    let bypass_cache = (visibility_ray >> 31u) != 0u;

    // Only a first-bounce entry's query is a probe ray. Capsaicin patches a second-bounce entry's
    // too (gi1.comp:2100-2106), whose query is the first-bounce entry it was traced from
    // (gi1.comp:1794), so its light lands on an unrelated probe ray (register D40).
    let is_first_bounce = visibility_index < atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);

    if (bypass_cache && is_first_bounce) {
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
    // USE_MULTI_BOUNCE (register D42): the indirect values' share, only in a cache that has them.
    const multibounce = (code: string): string => (layout.multibounce ? code : '');
    // Lever 1 (cellShading 'budget'): a cell keeps its light until it is lit again. Capsaicin's
    // blend below takes a cell of an updated tile that got no sample this frame one sample closer
    // to empty (the mean kept, the count down by one) and clears it when the count reaches 0
    // (gi1.comp:2170-2177): in per-hit shading every cell in use gets a sample each frame, so only
    // cells nobody looks at fade. On a budget most cells in use go unlit for frames, and would fade
    // too; so in that mode a cell with no new sample is left as it is.
    const keepUnlitCells = layout.cellShading
        ? `
    if (new_samples.w <= 0.0) {
        return previous_radiance;   // lever 1: no new sample, the light is reused until the cell is lit again
    }`
        : '';
    return /* wgsl */ `
${populationHeadWgsl(layout, POPULATION_BINDINGS.updateTiles)}

const UPDATE_TILES_GROUP_X: u32 = ${side}u;
const UPDATE_TILES_GROUP_Y: u32 = ${side}u;

var<workgroup> lds_UpdateTiles_ValueBuffer: array<array<vec2u, ${side}>, ${side}>;
${multibounce(`var<workgroup> lds_UpdateTiles_ValueIndirectBuffer: array<array<vec2u, ${side}>, ${side}>;`)}

// Temporal accumulation of one cell (gi1.comp:2162-2177 for the direct values, 2179-2195 for
// the indirect ones): the history and this frame's samples, each as a mean, blended with weight
// 1 / sample_count, the count capped at max_count; stored times the count, which later passes
// read as a hint for how much to prefilter.
fn UpdateTiles_Accumulate(previous_radiance: vec4f, new_samples: vec4f, max_count: f32) -> vec4f {${keepUnlitCells}
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
    let quantized = vec4u(atomicLoad(&g_HashGridCache_Words[scratch + 0u]),
                          atomicLoad(&g_HashGridCache_Words[scratch + 1u]),
                          atomicLoad(&g_HashGridCache_Words[scratch + 2u]),
                          atomicLoad(&g_HashGridCache_Words[scratch + 3u]));
    // Clear scratch
    atomicStore(&g_HashGridCache_Words[scratch + 0u], 0u);
    atomicStore(&g_HashGridCache_Words[scratch + 1u], 0u);
    atomicStore(&g_HashGridCache_Words[scratch + 2u], 0u);
    atomicStore(&g_HashGridCache_Words[scratch + 3u], 0u);
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

${multibounce(`fn UpdateTiles_SumIndirect(position: vec2u, step: u32) -> vec4f {
    var radiance = vec4f(0.0);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueIndirectBuffer[position.x + 0u][position.y + 0u]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueIndirectBuffer[position.x + step][position.y + 0u]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueIndirectBuffer[position.x + 0u][position.y + step]);
    radiance += HashGridCache_UnpackRadiance(lds_UpdateTiles_ValueIndirectBuffer[position.x + step][position.y + step]);
    return radiance;
}`)}

// One coarser level (MIP 1, 2 or 3): a thread on the level's grid ('step' threads apart) sums
// its 2 x 2 block of the level below and writes it to the tile and to the group's memory, where
// the next level reads it.
fn UpdateTiles_FilterLevel(group_thread_id: vec2u, tile_cell_offset: vec2u, tile_index: u32, mip_level: u32, step: u32) {
    let cell_index = HashGridCache_CellIndex(tile_cell_offset, tile_index, mip_level);

    // Box filter
    let packed_radiance = HashGridCache_PackRadiance(UpdateTiles_SumDirect(group_thread_id, step));
    lds_UpdateTiles_ValueBuffer[group_thread_id.x][group_thread_id.y] = packed_radiance;
    g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = packed_radiance;

    ${multibounce(`// USE_MULTI_BOUNCE
    let packed_radiance_indirect = HashGridCache_PackRadiance(UpdateTiles_SumIndirect(group_thread_id, step));
    lds_UpdateTiles_ValueIndirectBuffer[group_thread_id.x][group_thread_id.y] = packed_radiance_indirect;
    g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index] = packed_radiance_indirect;`)}
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

    let is_updated = update_tile_index < atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_UPDATETILECOUNTBUFFER]);
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
        ${multibounce(`// USE_MULTI_BOUNCE
        let indirect_radiance = UpdateTiles_Accumulate(
            HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index]),
            UpdateTiles_TakeScratch(HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER + 4u * cell_index),
            g_HashGridCacheConstants.max_multibounce_sample_count);`)}

        // Pack
        let packed_direct_radiance = HashGridCache_PackRadiance(direct_radiance);
        lds_UpdateTiles_ValueBuffer[group_thread_id.x][group_thread_id.y] = packed_direct_radiance;
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = packed_direct_radiance;

        ${multibounce(`let packed_indirect_radiance = HashGridCache_PackRadiance(indirect_radiance);
        lds_UpdateTiles_ValueIndirectBuffer[group_thread_id.x][group_thread_id.y] = packed_indirect_radiance;
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index] = packed_indirect_radiance;`)}
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
${populationHeadWgsl(layout, POPULATION_BINDINGS.resolveCells)}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.resolveCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_RESOLVECOUNTBUFFER])) {
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
    ${
        layout.multibounce
            ? `// USE_MULTI_BOUNCE
    let indirect_radiance = HashGridCache_FilteredRadianceIndirect(cell_index);
    radiance += (indirect_radiance.rgb / max(indirect_radiance.w, 1.0));`
            : '// Without USE_MULTI_BOUNCE the cache has no indirect values (register D42).'
    }
    ScreenProbes_AccumulateRadiance(query_index, radiance);
}
`;
}

// ---- GI-1.2's multibounce (RC3.3) ---------------------------------------------------------------

/**
 * The second-bounce entries' brdf and pdf (`g_HashGridCache_MultibounceInfoBuffer`,
 * `HashGridCache_PackBRDF` / `HashGridCache_UnpackBRDF`, hash_grid_cache.hlsl:409-419):
 * four halves, rounded as `f32tof16` would (register D31).
 */
export const MULTIBOUNCE_INFO_WGSL = /* wgsl */ `
fn HashGridCache_StoreMultibounceInfo(multibounce_index: u32, brdf: vec3f, pdf: f32) {
    let entry = g_HashGridCache_RayLists.multibounce_info_buffer + ${MULTIBOUNCE_INFO_WORDS}u * multibounce_index;
    let packed = HashGridCache_PackRadiance(vec4f(brdf, pdf));    // packHalf4(float4(brdf, pdf))
    HashGridCache_SetRayListWord(entry, 0u, packed.x);
    HashGridCache_SetRayListWord(entry, 1u, packed.y);
}

fn HashGridCache_LoadMultibounceInfo(multibounce_index: u32) -> vec4f {
    let entry = g_HashGridCache_RayLists.multibounce_info_buffer + ${MULTIBOUNCE_INFO_WORDS}u * multibounce_index;
    return unpackHalf4(vec2u(HashGridCache_RayListWord(entry, 0u), HashGridCache_RayListWord(entry, 1u)));
}
`;

/**
 * `sampleBRDFAndEvaluateDiffuse` (materials/material_sampling.hlsl:741-773)
 * with its helpers: `QuaternionRotationZ` and `Quaternion::transform`
 * (math/quaternion.hlsl:46-64), `sampleLambert` / `mapToCosineHemisphere`
 * (material_sampling.hlsl:295-299, math/sampling.hlsl:161-169) and
 * `sampleLambertPDF` (material_sampling.hlsl:306-309).
 *
 * The direction is Capsaicin's for the same random pair: a cosine-weighted
 * point of the +z hemisphere, turned so +z is the normal by the inverse of
 * the quaternion that turns the normal to +z. The reflectance is Lambert's,
 * albedo / π × cos (register D8; Capsaicin forces the second vertex to
 * roughness 1 and metallicity 0, gi1.comp:1909-1914, and evaluates its GGX
 * with the diffuse compensation there). The view direction only enters
 * Capsaicin's GGX half vector, so Lambert doesn't take it.
 */
export const SAMPLE_DIFFUSE_WGSL = /* wgsl */ `
const HASHGRIDCACHE_TWO_PI: f32 = 6.28318530717959;

// Quaternion::transform (quaternion.hlsl:46-51): the quaternion q = (axis xyz, w) applied to a direction.
fn HashGridCache_QuaternionTransform(q: vec4f, direction: vec3f) -> vec3f {
    let q_axis = q.xyz;
    return 2.0 * dot(q_axis, direction) * q_axis + (q.w * q.w - dot(q_axis, q_axis)) * direction + 2.0 * q.w * cross(q_axis, direction);
}

// QuaternionRotationZ (quaternion.hlsl:58-64): the rotation taking 'direction' to +z, with the
// special case of a direction at (or near) -z.
fn HashGridCache_QuaternionRotationZ(direction: vec3f) -> vec4f {
    return select(vec4f(1.0, 0.0, 0.0, 0.0), normalize(vec4f(direction.y, -direction.x, 0.0, 1.0 + direction.z)), direction.z >= -0.99999);
}

struct HashGridCache_DiffuseSample {
    direction: vec3f,   // the new ray's direction, world space
    brdf: vec3f,        // the reflectance toward it, the cosine included
    pdf: f32,           // its solid-angle density
}

fn HashGridCache_SampleBRDFAndEvaluateDiffuse(albedo: vec3f, normal: vec3f, samples: vec2f) -> HashGridCache_DiffuseSample {
    // Transform the view direction into the surfaces tangent coordinate space (oriented so that z axis is aligned to normal)
    let local_rotation = HashGridCache_QuaternionRotationZ(normal);

    // Sample diffuse BRDF component (mapToCosineHemisphere: Ray Tracing Gems, "Sampling Transformations Zoo", Shirley)
    let sin_theta = sqrt(samples.x);
    let phi = HASHGRIDCACHE_TWO_PI * samples.y;
    let new_light = vec3f(sin_theta * cos(phi), sin_theta * sin(phi), sqrt(1.0 - samples.x));

    // Evaluate BRDF for new light direction (Lambert, register D8)
    let dot_nl = clamp(new_light.z, -1.0, 1.0);
    let brdf = albedo * HASHGRIDCACHE_INV_PI * saturate(dot_nl);

    // Calculate combined PDF for current sample
    let pdf = saturate(dot_nl) * HASHGRIDCACHE_INV_PI;

    // Transform the new direction back into world space
    let inverse_rotation = vec4f(-local_rotation.xyz, local_rotation.w);
    let light_direction = normalize(HashGridCache_QuaternionTransform(inverse_rotation, new_light));
    return HashGridCache_DiffuseSample(light_direction, brdf, pdf);
}
`;

/** Which first-bounce entries `PopulateMultibounceCells` traces from: every hit (Capsaicin's), or lever 1's lit cells. */
export type MultibounceSource = 'everyHit' | 'litCells';

/**
 * `PopulateMultibounceCells` (gi1.comp:1869-1939) with its hit handler
 * (`PopulateMultibounceCellsHandleHit`, gi1.comp:1761-1852): for each
 * first-bounce visibility entry, a random number keeps 30% of them
 * (`discard_multibounce_ray_probability`, 0.7); each kept one traces one
 * cosine-sampled ray from its hit, and the ray's hit is inserted as a
 * second-bounce entry: its brdf and pdf at `multibounce_index` (the
 * second-bounce count), its hit, cell and query (the first-bounce entry) at
 * `multibounce_index + visibility_count0`. Dispatched on the first-bounce
 * count, after `PopulateScreenProbes`' cache half (gi1.cpp:2363-2393).
 *
 * Built with the platform's hit-shading module (group 0): `giSurfaceAt`
 * rebuilds the first-bounce hit, `giTraceClosest` traces the new ray.
 */
export function populateMultibounceCellsWgsl(layout: HashGridCacheBufferLayout, source: MultibounceSource = 'everyHit'): string {
    const size = POPULATION_WORKGROUP_SIZE.populateMultibounceCells;
    // Which first-bounce entries trace: every hit (Capsaicin's), or, with lever 1's budget, the
    // hit each lit cell kept (one per lit cell, docs/design/radiance-cache.md, "Budgeted cell
    // shading", step 5), so the second bounce is on the same budget.
    const firstBounceEntry =
        source === 'everyHit'
            ? `    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        return; // out of bounds
    }`
            : `    let lit_index = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (lit_index >= atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_LITCELLCOUNT])) {
        return; // out of bounds
    }
    // Lever 1: the first-bounce entry the lit cell kept. Its random numbers are the ones Capsaicin's
    // pass would draw for that entry, so the discard keeps the same share of lit cells as of hits.
    let did = HashGridCache_RayListWord(g_HashGridCache_RayLists.chosen_cell_buffer, lit_index);`;
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${populationHeadWgsl(layout, POPULATION_BINDINGS.populateMultibounceCells)}
${VISIBILITY_SURFACE_WGSL}
${INV_PI_WGSL}
${MULTIBOUNCE_INFO_WGSL}
${SAMPLE_DIFFUSE_WGSL}

// PopulateMultibounceCellsHandleHit (gi1.comp:1761-1852): as the first bounce's insert
// (PopulateScreenProbesHandleHit), into the second-bounce half of the visibility lists.
fn PopulateMultibounceCellsHandleHit(did: u32, brdf: vec3f, pdf: f32, ray: GiRay, hit: GiHit) {
    var data: HashGridCache_Data;
    data.eye_position = g_HashGridCacheConstants.eye_position;
    data.hit_position = ray.origin + (hit.t * ray.direction);
    data.direction    = ray.direction;
    data.hit_distance = hit.t;

    let inserted = HashGridCache_InsertCell(data);
    let tile_index = inserted.tile_index;
    let is_new_tile = inserted.is_new_tile;
    let cell_index = inserted.cell_index;

    if (cell_index != kGI1_InvalidId) {
        // Bump the cell's decay to the max. now that it's been 'touched'
        let previous_tile_decay = HashGridCache_TouchTile(tile_index);

        // We update the cell index for later passes
        let multibounce_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1], 1u);
        HashGridCache_StoreMultibounceInfo(multibounce_index, brdf, pdf);

        // The visibility entry keeps the ray beside the hit, as the first bounce's does (register D30),
        // its direction as three halves (Capsaicin's packHalf3).
        let visibility_index = multibounce_index + atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
        let direction_halves = vec2u(pack2x16float(ray.direction.xy), pack2x16float(vec2f(ray.direction.z, 0.0)));
        HashGridCache_StoreVisibility(visibility_index, ray.origin, direction_halves, hit.t, hit.instance, hit.frontFace, hit.triangle, hit.barycentrics);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index, cell_index);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index, did);   // BE CAREFUL: a visibility entry, not a probe ray

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
        // (Capsaicin's condition, with a once-per-frame check so a new tile is listed once: register D32.)
        if ((is_new_tile || previous_tile_decay != g_HashGridCacheConstants.frame_index) && HashGridCache_ListTileForUpdate(tile_index)) {
            let update_tile_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_UPDATETILECOUNTBUFFER], 1u);
            HashGridCache_SetRayListWord(g_HashGridCache_RayLists.update_tile_buffer, update_tile_index, tile_index);
        }
    }
}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.populateMultibounceCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
${firstBounceEntry}

    // Random sample
    var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);

    if (rcRand(&random) <= g_HashGridCacheConstants.discard_multibounce_ray_probability) {
        return;
    }

    // Load our visibility sample, and reconstruct world-space position and normal (the shading
    // normal, turned to the side the ray arrived on, as Capsaicin's).
    let visibility = HashGridCache_LoadVisibility(did);
    let surface = HashGridCache_VisibilitySurface(visibility);

    // Force diffuse lighting on secondary bounce to reduce noise (gi1.comp:1909-1914): the
    // platform's materials are Lambert already (register D8).

    // Recover the ray origin: Capsaicin rebuilds the probe's point from the depth buffer for the
    // view direction (gi1.comp:1919-1925), which only its GGX lobe reads; Lambert doesn't.

    // Sample diffuse direction
    let diffuse_sample = HashGridCache_SampleBRDFAndEvaluateDiffuse(surface.albedo, surface.normal, rcRand2(&random));

    // Trace: MakeRayInfoClosest(world, normal, direction, MAX_HIT_DISTANCE) (trace_ray.hlsl:112-119,
    // gi1.comp:1933) pushes the origin off the surface with offsetPosition along the same normal
    // (register D39).
    let ray = GiRay(rcOffsetPosition(surface.position, surface.normal), 0.0, diffuse_sample.direction, RC_MAX_HIT_DISTANCE);
    let hit = giTraceClosest(ray);

    // If we hit some geometry, we append a new world-space hash-grid cache query
    if (hit.t >= 0.0) {
        PopulateMultibounceCellsHandleHit(did, diffuse_sample.brdf, diffuse_sample.pdf, ray, hit);
    }
}
`;
}

/**
 * `GenerateMultibounceReservoirs` (gi1.comp:2601-2695): one light per
 * second-bounce hit (the platform's choice, register D4, as
 * `GenerateReservoirs`), its shadow ray appended to the same list as the
 * first bounce's, so `PopulateCells` lights both; flagged to bypass when the
 * second-bounce ray is shorter than its cell. Unlike the first bounce, no
 * emissive test and no resolve entry: the second-bounce cells are only read
 * by `UpdateMultibounceCells`. Dispatched on the second-bounce count
 * (gi1.cpp:2425-2433).
 *
 * Built with the platform's hit-shading module (group 0).
 */
export function generateMultibounceReservoirsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.generateMultibounceReservoirs;
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${populationHeadWgsl(layout, POPULATION_BINDINGS.generateMultibounceReservoirs)}
${VISIBILITY_SURFACE_WGSL}

const HASHGRIDCACHE_LIGHT_GROUPS: u32 = ${ALL_LIGHT_GROUPS}u;
${LIGHT_DRAW_WGSL}
@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.generateMultibounceReservoirs}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    var did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1])) {
        return; // out of bounds
    }

    did += atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);

    // Load our visibility sample, and reconstruct world-space position and normal. Capsaicin's
    // normal here is interpolated from vertex normals it doesn't transform (gi1.comp:2626);
    // giSurfaceAt's is in world space (register D41).
    let visibility = HashGridCache_LoadVisibility(did);
    let surface = HashGridCache_VisibilitySurface(visibility);

    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did);

    // Recover the ray origin: the first-bounce hit this ray left from, rebuilt from its visibility
    // entry (gi1.comp:2635-2646 interpolates the triangle; giSurfaceAt does the same).
    let origin = HashGridCache_VisibilitySurface(HashGridCache_LoadVisibility(query_index)).position;

    // Sample new lights, the sky among them (LIGHT_DRAW_WGSL). Capsaicin forces a diffuse material
    // on this bounce (gi1.comp:2650-2656); the platform's materials are diffuse already (register D8).
    var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);
    let draw = HashGridCache_DrawLight(surface, &random);
    let reservoir = draw.choice;
    let sample_params = draw.sample_params;

    if (reservoir.light == GI_DIRECT_NO_LIGHT) {
        return;
    }

    // Append our resampled shadow ray
    let ray_index = atomicAdd(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER], 1u);

    let cell_size  = HashGridCache_GetCellSize(surface.position);
    let ray_length = distance(origin, surface.position);

    let bypass_cache = ray_length < cell_size;
    let bypass_bit = select(0u, HASHGRIDCACHE_BYPASS_BIT, bypass_cache);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_ray_buffer, ray_index, did | bypass_bit);   // compact the surviving shadow rays

    // Write the results out to memory: the reservoir (packReservoir), as GenerateReservoirs.
    let reservoir_word = g_HashGridCache_RayLists.reservoir_buffer + ${RESERVOIR_WORDS}u * ray_index;
    HashGridCache_SetRayListWord(reservoir_word, 0u, reservoir.light);
    HashGridCache_SetRayListWord(reservoir_word, 1u, bitcast<u32>(reservoir.weight));
    HashGridCache_SetRayListWord(reservoir_word, 2u, pack2x16unorm(sample_params));
}
`;
}

/**
 * `UpdateMultibounceCells` (gi1.comp:1948-1983): for each second-bounce
 * entry, its cell's filtered direct radiance (this frame's, `UpdateTiles` has
 * run) × brdf / pdf, divided by the share of rays kept (1 - 0.7), is added,
 * quantised, to the indirect update scratch of the first-bounce cell it was
 * traced from. The next frame's `UpdateTiles` takes it into that cell's
 * indirect value: the one-frame lag. A first-bounce ray that bypasses the
 * cache gets the sample straight away, as `PopulateCells` gives it its direct
 * light (register D40 for which ray that is). Dispatched on the second-bounce
 * count after `UpdateTiles` (gi1.cpp:2505-2516).
 */
export function updateMultibounceCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.updateMultibounceCells;
    return /* wgsl */ `
${populationHeadWgsl(layout, POPULATION_BINDINGS.updateMultibounceCells)}
${MULTIBOUNCE_INFO_WGSL}
${REMOVE_NANS_WGSL}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.updateMultibounceCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1])) {
        return; // out of bounds
    }

    let did2 = did + atomicLoad(&g_HashGridCache_Words[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did2);
    let base_visibility_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did2);
    let cell_index_base = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, base_visibility_index);   // BE CAREFUL: also a cell
    let query_info = HashGridCache_LoadMultibounceInfo(did);
    let brdf = query_info.xyz;
    let pdf = query_info.w;
    let direct_radiance = HashGridCache_FilteredRadianceDirect(cell_index);

    var radiance = (direct_radiance.rgb / max(direct_radiance.w, 1.0));
    radiance = (radiance * brdf) / pdf;
    radiance /= 1.0 - g_HashGridCacheConstants.discard_multibounce_ray_probability;
    let quantized_radiance = HashGridCache_QuantizeRadiance(radiance);
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER + 4u * cell_index_base;

    if (dot(radiance, radiance) > 0.0) {
        atomicAdd(&g_HashGridCache_Words[scratch + 0u], quantized_radiance.x);
        atomicAdd(&g_HashGridCache_Words[scratch + 1u], quantized_radiance.y);
        atomicAdd(&g_HashGridCache_Words[scratch + 2u], quantized_radiance.z);
    }
    atomicAdd(&g_HashGridCache_Words[scratch + 3u], quantized_radiance.w);

    // A first-bounce ray that bypasses the cache's filtered values (ResolveCells skips it) gets its
    // second bounce here. Capsaicin reads the flag from g_HashGridCache_VisibilityRayBuffer[did2], a
    // list indexed by shadow ray, and patches the probe ray g_HashGridCache_VisibilityQueryBuffer[did2],
    // which is the first-bounce entry, not its probe ray (gi1.comp:1977-1981). Ours reads the
    // first-bounce entry's own flag and its probe ray (register D40). Each first-bounce entry traced
    // at most one second-bounce ray, so no two threads add to the same probe ray.
    let bypass_cache = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_bypass_buffer, base_visibility_index) != 0u;
    if (bypass_cache) {
        let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, base_visibility_index);
        ScreenProbes_AccumulateRadiance(query_index, GIDenoiser_RemoveNaNs(radiance.rgb));
    }
}
`;
}
