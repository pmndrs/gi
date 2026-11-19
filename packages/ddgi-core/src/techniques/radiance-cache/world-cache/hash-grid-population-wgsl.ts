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
 * GI-1.2's multibounce (RC3.3; `USE_MULTI_BOUNCE`, on by default, gi1.h:55)
 * adds three passes in the slots Capsaicin runs them (gi1.cpp:2363-2390,
 * 2427-2436, 2506-2516):
 *
 * - `PopulateMultibounceCells` (gi1.comp:1761-1946), after the probe rays'
 *   insert: 70% of the first-bounce hits stop there (the discard,
 *   `discard_multibounce_ray_probability`); the rest trace one ray in a
 *   cosine-weighted direction from the hit, and that ray's hit inserts its
 *   cell, joins the visibility lists after the first-bounce entries, and
 *   keeps the BRDF and pdf of its direction (`MultibounceInfoBuffer`); a ray
 *   that misses joins them too, with no cell (register D37).
 * - `GenerateMultibounceReservoirs` (gi1.comp:2603-2700), after
 *   `GenerateReservoirs`: one light per second-bounce hit, its shadow ray
 *   appended to the same list, so `PopulateCells` lights the second-bounce
 *   cells with the first-bounce ones.
 * - `UpdateMultibounceCells` (gi1.comp:1948-1989), after `UpdateTiles`: each
 *   second-bounce hit reads its cell's filtered direct radiance (its own light
 *   where its segment is shorter than its cell), weighs it by BRDF / pdf and
 *   adds it into the first-bounce cell's indirect scratch; a miss adds a
 *   sample of no light. `UpdateTiles` blends that scratch next frame (the
 *   one-frame lag), and `ResolveCells` reads the indirect values with the
 *   direct ones.
 *
 * RC3.5 fixes Capsaicin's errors in these passes (DECISIONS 2026-11-19): the
 * second bounce's estimate (register D37: no 1 / (1 - p) on a mean over the
 * kept rays, and misses count) and the bypass paths (register D38: the light
 * goes to the right probe ray, weighted, or stays with the second bounce).
 *
 * Off (`useMultibounce` false) is GI-1.0's single bounce: those passes don't
 * run, the indirect values are neither updated nor read, and the resolve list
 * holds only the hits with a light, as Capsaicin's build without
 * `USE_MULTI_BOUNCE` resolves its shadow-ray list (register D35).
 *
 * How the passes find their data (register D30): Capsaicin copies each hit
 * into its own visibility buffer (`HashGridCache_PackVisibility`) and keeps
 * the reservoir's surface and material in their own lists; here a visibility
 * entry keeps its probe ray's index (`VisibilityQueryBuffer`, which Capsaicin
 * also keeps) and its cell, and each pass reads the hit from the probe ray and
 * rebuilds the surface with the platform's `giSurfaceAt`. A second-bounce
 * entry has no probe ray, so its ray is kept in the lists in the probe rays'
 * own record (`MultibounceRayBuffer`, register D36). The lists live in one
 * packed buffer (register D6) at the word offsets of `HashGridCacheRayLists`.
 */

import { GILimitError } from '../../../platform/contract/errors.ts';
import { CAPSAICIN_WGSL } from '../wgsl/capsaicin.ts';
import type { HashGridCacheBufferLayout } from './capacity.ts';
import { CELL_CHANGE_RESET_BIT, CELL_CHANGES_BINDING, CELL_CHANGES_WGSL, CELL_SHADING_COMMON_WGSL } from './cell-shading-common.ts';
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
    // Lever E8 and experiment E9 (cell-shading-common.ts): the change events, each cell's last reset, E8's lists.
    { binding: CELL_CHANGES_BINDING.uniform, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { binding: CELL_CHANGES_BINDING.resetFrames, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    { binding: CELL_CHANGES_BINDING.cellLists, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
];

/** Bytes of `HashGridCacheRayLists`: sixteen words (the neural tail's last, P6.4), padded to 16 bytes. */
export const RAY_LISTS_UNIFORM_BYTES = 80;

/** Word of `HashGridCacheRayLists` that says how cells are lit: bit 0 budgeted (lever E8), bit 1 per-hit change reset (E9). Written every frame. */
export const RAY_LISTS_CELL_SHADING_WORD = 14;
/** Bits of that word; bits 8-15 hold E9's history kept (`cellChangeResetHistory`). */
export const CELL_SHADING_FLAG = { budgeted: 1, changeReset: 2, historyShift: 8 } as const;

/** Words of a reservoir: the light, its weight (f32 bits), the point on the light (two unorm16). */
export const RESERVOIR_WORDS = 3;

/** Words of a second-bounce entry's neural-tail answer (P6.4, register E11): rgb and a valid flag as four halves. */
export const NEURAL_TAIL_WORDS = 2;

/** Words of a second-bounce entry's `MultibounceInfoBuffer` value: BRDF and pdf as four halves (`HashGridCache_PackBRDF`). */
export const MULTIBOUNCE_INFO_WORDS = 2;

/** Words of a second-bounce ray kept in the lists: a `WorldCacheProbeRay`, word for word (register D36). */
export const MULTIBOUNCE_RAY_WORDS = 12;

/** Words of a second-bounce entry's own direct light, for a segment shorter than its cell: rgb as halves (register D38). */
export const MULTIBOUNCE_DIRECT_WORDS = 2;

/** A second-bounce entry that is a miss (register D37): its cell word holds kGI1_InvalidId. */
export const MULTIBOUNCE_MISS_CELL = 0xffffffff;

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
    /** GenerateMultibounceReservoirs (gi1.comp:2603): 64. */
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
    readonly visibilityCellBuffer: number;
    readonly visibilityQueryBuffer: number;
    readonly visibilityRayBuffer: number;
    readonly resolveBuffer: number;
    readonly reservoirBuffer: number;
    /** `g_HashGridCache_MultibounceInfoBuffer`: a second-bounce entry's BRDF and pdf (`max_ray_count` × uint2, gi1.cpp:688). */
    readonly multibounceInfoBuffer: number;
    /** Ours (register D36): a second-bounce entry's ray, as a probe-ray record (`max_ray_count` of them). */
    readonly multibounceRayBuffer: number;
    /** Ours (register D38): per first-bounce entry, 1 when its probe ray is shorter than its cell (it bypasses the cache). */
    readonly firstBounceBypassBuffer: number;
    /** Ours (register D38): per second-bounce entry, its own direct light where its segment bypasses the cache. */
    readonly multibounceDirectBuffer: number;
    /**
     * P6.4 (register E11): per second-bounce entry, the neural tail's answer
     * (`NEURAL_TAIL_WORDS`), which `UpdateMultibounceCells` adds to the second
     * hit's light when the tail is in use. Only made with a tail: without
     * one it is 0 and holds no words.
     */
    readonly neuralTailBuffer: number;
    readonly bytes: number;
}

/**
 * The lists Capsaicin sizes from the screen (`ensureMemoryIsAllocated`,
 * gi1.cpp:637-695): the update tiles (at most one per cell), the visibility
 * cell, query and ray lists (`max_combined_ray_count` each), the resolve list
 * (`max_ray_count`, first bounce only), a reservoir per visibility ray
 * (`g_Reservoir_IndirectSampleReservoirBuffer`), and for each second-bounce
 * entry (at most one per first-bounce entry, so `max_ray_count`) its BRDF and
 * pdf (`MultibounceInfoBuffer`) and its ray (ours, register D36).
 */
export function hashGridCacheRayListLayout(maxRayCount: number, numCells: number, secondBounceRays = true, neuralTail = false): HashGridCacheRayListLayout {
    // USE_MULTI_BOUNCE is Capsaicin's default (gi1.h:55): bounce 0 and bounce 1 share the visibility lists.
    // Without second-bounce rays (multibounce off, or the second bounce from a fallback, register
    // E10) the lists hold the first bounce only and the second-bounce lists are empty: 9 words a
    // probe ray instead of 32. The offsets are uniforms, so the lists are made again on a switch.
    const secondBounceCount = secondBounceRays ? maxRayCount : 0;
    const maxCombinedRayCount = maxRayCount + secondBounceCount;
    const updateTileBuffer = 0;
    const visibilityCellBuffer = updateTileBuffer + Math.min(maxCombinedRayCount, numCells);
    const visibilityQueryBuffer = visibilityCellBuffer + maxCombinedRayCount;
    const visibilityRayBuffer = visibilityQueryBuffer + maxCombinedRayCount;
    const resolveBuffer = visibilityRayBuffer + maxCombinedRayCount;
    const reservoirBuffer = resolveBuffer + maxRayCount;
    const multibounceInfoBuffer = reservoirBuffer + RESERVOIR_WORDS * maxCombinedRayCount;
    const multibounceRayBuffer = multibounceInfoBuffer + MULTIBOUNCE_INFO_WORDS * secondBounceCount;
    const firstBounceBypassBuffer = multibounceRayBuffer + MULTIBOUNCE_RAY_WORDS * secondBounceCount;
    const multibounceDirectBuffer = firstBounceBypassBuffer + maxRayCount;
    const tailStart = multibounceDirectBuffer + MULTIBOUNCE_DIRECT_WORDS * secondBounceCount;
    // P6.4 (register E11): the neural tail's answers, one per second-bounce entry, only with a tail.
    const words = tailStart + (neuralTail ? NEURAL_TAIL_WORDS * secondBounceCount : 0);
    return {
        maxRayCount,
        maxCombinedRayCount,
        updateTileBuffer,
        visibilityCellBuffer,
        visibilityQueryBuffer,
        visibilityRayBuffer,
        resolveBuffer,
        reservoirBuffer,
        multibounceInfoBuffer,
        multibounceRayBuffer,
        firstBounceBypassBuffer,
        multibounceDirectBuffer,
        neuralTailBuffer: neuralTail && secondBounceRays ? tailStart : 0,
        bytes: Math.max(words, 1) * 4,
    };
}

/**
 * Throws `GILimitError` naming the limit the per-ray lists pass. They are one
 * buffer bound whole as one storage binding, so it must fit both the binding
 * limit and the buffer limit. It grows with the screen (32 words per probe
 * ray: 29 MB at 1280 × 720, 265 MB at 3840 × 2160, past WebGPU's default
 * 128 MiB binding; lever E8's lists are a buffer of their own, made when it
 * is used), so a larger screen or a device with smaller limits can't
 * bind it where the cache itself fits (register D14's rule: name the limit,
 * no fallback).
 */
export function checkHashGridRayListLimits(layout: HashGridCacheRayListLayout, limits: Pick<GPUSupportedLimits, 'maxStorageBufferBindingSize' | 'maxBufferSize'>): void {
    if (layout.bytes > limits.maxStorageBufferBindingSize) throw new GILimitError('maxStorageBufferBindingSize', layout.bytes, limits.maxStorageBufferBindingSize);
    if (layout.bytes > limits.maxBufferSize) throw new GILimitError('maxBufferSize', layout.bytes, limits.maxBufferSize);
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
    multibounce_info_buffer: u32,   // g_HashGridCache_MultibounceInfoBuffer: BRDF and pdf per second-bounce entry, 2 words
    multibounce_ray_buffer: u32,    // ours (register D36): each second-bounce entry's ray, a WorldCacheProbeRay in 12 words
    first_bounce_bypass_buffer: u32,  // ours (register D38): 1 where a first-bounce entry's probe ray bypasses the cache
    multibounce_direct_buffer: u32,   // ours (register D38): a bypassing second-bounce entry's own direct light, 2 words
    cell_candidate_buffer: u32,     // lever E8, in g_HashGridCache_CellLists: the cells hit this frame, two words each
    cell_shade_buffer: u32,         // lever E8, in g_HashGridCache_CellLists: the cells lit this frame
    cell_shading: u32,              // bit 0: lever E8, cells lit on a budget; bit 1: E9, per-hit change reset; bits 8-15: E9's history kept
    cell_priority_bins: u32,        // lever E8, in g_HashGridCache_CellLists: each bin's first slot, then its count
    neural_tail_buffer: u32,          // P6.4 (register E11): the neural tail's answer per second-bounce entry, 2 words
}

${CELL_CHANGES_WGSL}

// Lever E8: hits only find their cells, and the cells are lit on a budget (cell-shading-wgsl.ts).
fn HashGridCache_CellShadingBudgeted() -> bool {
    return (g_HashGridCache_RayLists.cell_shading & ${CELL_SHADING_FLAG.budgeted}u) != 0u;
}

// Experiment E9: per hit, a cell hit near a recent scene change takes the frame's samples whole.
fn HashGridCache_ChangeResetPerHit() -> bool {
    return (g_HashGridCache_RayLists.cell_shading & ${CELL_SHADING_FLAG.changeReset}u) != 0u;
}

// E9: the samples of a marked cell's history it keeps (0: the frame's samples replace it).
fn HashGridCache_ChangeResetHistory() -> f32 {
    return f32((g_HashGridCache_RayLists.cell_shading >> ${CELL_SHADING_FLAG.historyShift}u) & 0xffu);
}

// E9's mark in a cell's direct scratch count (PopulateCells sets it, UpdateTiles takes it off).
const HASHGRIDCACHE_CHANGE_RESET_BIT: u32 = ${CELL_CHANGE_RESET_BIT}u;

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

// ---- GI-1.2's second-bounce entries (USE_MULTI_BOUNCE) ----

// HashGridCache_PackBRDF / HashGridCache_UnpackBRDF (hash_grid_cache.hlsl:410-419): the BRDF and
// the pdf of a second-bounce ray's direction as four halves (packHalf4, rounded as f32tof16 would).
fn HashGridCache_PackBRDF(brdf: vec3f, pdf: f32) -> vec2u {
    return HashGridCache_PackRadiance(vec4f(brdf, pdf));
}

fn HashGridCache_UnpackBRDF(packed_data: vec2u) -> vec4f {
    return unpackHalf4(packed_data);
}

// A second-bounce entry's ray (register D36): Capsaicin keeps the hit in its visibility buffer
// (HashGridCache_PackVisibility, gi1.comp:1781-1793); the lists keep the whole ray here, in the
// probe rays' own record, so the passes rebuild its surface as they rebuild a probe ray's.
// 'index' counts second-bounce entries (the visibility index less VisibilityCountBuffer0).
fn HashGridCache_StoreMultibounceRay(index: u32, ray: WorldCacheProbeRay) {
    let base = g_HashGridCache_RayLists.multibounce_ray_buffer + ${MULTIBOUNCE_RAY_WORDS}u * index;
    HashGridCache_SetRayListWord(base, 0u, bitcast<u32>(ray.origin.x));
    HashGridCache_SetRayListWord(base, 1u, bitcast<u32>(ray.origin.y));
    HashGridCache_SetRayListWord(base, 2u, bitcast<u32>(ray.origin.z));
    HashGridCache_SetRayListWord(base, 3u, ray.flags);
    HashGridCache_SetRayListWord(base, 4u, ray.direction.x);
    HashGridCache_SetRayListWord(base, 5u, ray.direction.y);
    HashGridCache_SetRayListWord(base, 6u, bitcast<u32>(ray.hitT));
    HashGridCache_SetRayListWord(base, 7u, ray.hitInstance);
    HashGridCache_SetRayListWord(base, 8u, ray.hitTriangle);
    HashGridCache_SetRayListWord(base, 9u, ray.hitFrontFace);
    HashGridCache_SetRayListWord(base, 10u, bitcast<u32>(ray.hitBarycentrics.x));
    HashGridCache_SetRayListWord(base, 11u, bitcast<u32>(ray.hitBarycentrics.y));
}

fn HashGridCache_LoadMultibounceRay(index: u32) -> WorldCacheProbeRay {
    let base = g_HashGridCache_RayLists.multibounce_ray_buffer + ${MULTIBOUNCE_RAY_WORDS}u * index;
    var ray: WorldCacheProbeRay;
    ray.origin = vec3f(bitcast<f32>(HashGridCache_RayListWord(base, 0u)),
                       bitcast<f32>(HashGridCache_RayListWord(base, 1u)),
                       bitcast<f32>(HashGridCache_RayListWord(base, 2u)));
    ray.flags = HashGridCache_RayListWord(base, 3u);
    ray.direction = vec2u(HashGridCache_RayListWord(base, 4u), HashGridCache_RayListWord(base, 5u));
    ray.hitT = bitcast<f32>(HashGridCache_RayListWord(base, 6u));
    ray.hitInstance = HashGridCache_RayListWord(base, 7u);
    ray.hitTriangle = HashGridCache_RayListWord(base, 8u);
    ray.hitFrontFace = HashGridCache_RayListWord(base, 9u);
    ray.hitBarycentrics = vec2f(bitcast<f32>(HashGridCache_RayListWord(base, 10u)), bitcast<f32>(HashGridCache_RayListWord(base, 11u)));
    return ray;
}

// The ray whose hit a visibility entry stands for: a first-bounce entry (below
// VisibilityCountBuffer0) is a probe ray's hit, found through its query index; a second-bounce
// entry's ray is in the lists (register D36).
fn HashGridCache_VisibilityRay(visibility_index: u32) -> WorldCacheProbeRay {
    let first_bounce_count = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
    if (visibility_index < first_bounce_count) {
        return g_ProbeRays.rays[HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index)];
    }
    return HashGridCache_LoadMultibounceRay(visibility_index - first_bounce_count);
}
`;

/**
 * The surface a probe ray hit, rebuilt from its hit record with the
 * platform's `giSurfaceAt` (for kernels built with the hit-shading module).
 * Capsaicin rebuilds it from its packed visibility the same way
 * (`fetchVerticesNormUV`, `interpolate`; gi1.comp:2412-2427, 2050-2062).
 */
export const PROBE_RAY_SURFACE_WGSL = /* wgsl */ `
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

/**
 * The shared head of every population kernel: the hash grid (group 1) and
 * the ray data (group 2). Exported for kernels that run inside the grid's
 * frame with its data (E10's pass, P6.4's neural tail).
 */
export function populationHeadWgsl(layout: HashGridCacheBufferLayout): string {
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
        // (Capsaicin's condition, with a once-per-frame check so a new tile is listed once: register D32.)
        if ((is_new_tile || previous_tile_decay != g_HashGridCacheConstants.frame_index) && HashGridCache_ListTileForUpdate(tile_index)) {
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
${CELL_SHADING_COMMON_WGSL}

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

    // The entry's bypass flag by visibility index, for UpdateMultibounceCells (register D38):
    // 0 unless set below (the list outlives the frame, so every entry writes it).
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.first_bounce_bypass_buffer, did, 0u);

    // Patch the screen probes with some emissivity information:
    // We bypass the hash-grid cache entirely here, as adding emissive information to the cells
    // effectively enlarges the area light (due to the spatial nature of the grid) and leads to
    // light leaks and generally poorer visuals.
    if (g_HashGridCacheConstants.use_direct_lighting != 0u && surface.frontFace && dot(surface.emission, surface.emission) > 0.0) {
        ScreenProbes_AccumulateRadiance(query_index, surface.emission);
        return; // do not continue past an emissive surface
    }

    // Lever E8 (cells lit on a budget): the hit offers itself as its cell's sample and, if its ray
    // is at least a cell long, joins the resolve list with no light of its own; its cell is lit by
    // ClaimCells-ShadeCells. A shorter ray reads no cell, so it is still lit per hit below.
    // Unlike per hit with multibounce off, a hit is resolved without knowing whether it has a light.
    if (HashGridCache_CellShadingBudgeted()) {
        HashGridCache_OfferCellSample(did, HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did));
        if (!HashGridCache_ShorterThanCell(probe_ray.origin, surface.position)) {
            let budgeted_resolve_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_RESOLVECOUNTBUFFER], 1u);
            HashGridCache_SetRayListWord(g_HashGridCache_RayLists.resolve_buffer, budgeted_resolve_index, did);
            return;
        }
    }

    // Experiment E9: a hit near a change marks its cell here, where every hit passes (with a light or
    // not, so a light switched off is followed too), not where samples are added.
    HashGridCache_MarkChangeReset(HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did), surface.position);

    // Temporal radiance feedback (gi1.comp:2454-2496) runs only with gi1_use_temporal_feedback,
    // off by default (gi1.h:54): not part of the baseline.

    // Sample new lights. Capsaicin forces a diffuse material on this bounce (gi1.comp:2506-2510);
    // the platform's materials are diffuse already (register D8). Capsaicin's light list also
    // holds the environment map; the platform's doesn't (register D33).
    var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);
    let reservoir = giDirectChooseLight(surface, HASHGRIDCACHE_LIGHT_GROUPS, rcRand(&random));
    let sample_params = rcRand2(&random);

    // Without USE_MULTI_BOUNCE Capsaicin resolves the shadow-ray list (gi1.comp:2363-2368, 2514-2521),
    // so a hit with no light isn't resolved; the resolve list then holds the same entries (register D35).
    if (g_HashGridCacheConstants.use_multibounce == 0u && reservoir.light == GI_DIRECT_NO_LIGHT) {
        return;
    }

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
    if (bypass_cache) {
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.first_bounce_bypass_buffer, did, 1u);
    }

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

    // Reconstruct world-space position and normal: a first-bounce entry's probe ray, or a
    // second-bounce entry's own ray (GenerateMultibounceReservoirs appends those to this list).
    let surface = HashGridCache_ProbeSurface(HashGridCache_VisibilityRay(visibility_index));

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

    // Lever E8 (cells lit on a budget): only rays shorter than their cell have shadow rays here, and
    // their samples stay with their rays; the cell's sample is ShadeCells', and until ClaimCells the
    // scratch's first words hold the entries' keys.
    if (!HashGridCache_CellShadingBudgeted()) {
        if (dot(lighting, lighting) > 0.0) {
            atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 0u], quantized_radiance.x);
            atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 1u], quantized_radiance.y);
            atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 2u], quantized_radiance.z);
        }
        atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 3u], quantized_radiance.w);
    }

    // In some cases, we want to bypass the cache to avoid light leaks;
    // so we simply patch the screen probes directly and invalidate the
    // cache connection.
    let bypass_cache = (visibility_ray >> 31u) != 0u;

    if (bypass_cache) {
        let first_bounce_count = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
        if (visibility_index < first_bounce_count) {
            // A first-bounce entry: its query is its probe ray, which reads this sample instead of its cell.
            let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index);
            ScreenProbes_AccumulateRadiance(query_index, GIDenoiser_RemoveNaNs(lighting));
        } else {
            // A second-bounce entry (register D38): Capsaicin adds its light to the probe ray numbered
            // as its first-bounce entry (gi1.comp:2102-2105 with 1793), unweighted by the first
            // bounce's BRDF / pdf. The light is the second hit's, and what the bypass guards against
            // is reading it from a cell wider than the segment; so it is kept here, in place of the
            // cell's filtered value, for UpdateMultibounceCells to weigh into the first bounce.
            let base = g_HashGridCache_RayLists.multibounce_direct_buffer + ${MULTIBOUNCE_DIRECT_WORDS}u * (visibility_index - first_bounce_count);
            HashGridCache_SetRayListWord(base, 0u, HashGridCache_PackHalf2(lighting.xy));
            HashGridCache_SetRayListWord(base, 1u, HashGridCache_PackHalf2(vec2f(lighting.z, 0.0)));
        }
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
 * result). The indirect values (`USE_MULTI_BOUNCE`) are updated only with
 * multibounce on (register D35).
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
    if (g_HashGridCacheConstants.use_multibounce != 0u) {
        let packed_radiance_indirect = HashGridCache_PackRadiance(UpdateTiles_SumIndirect(group_thread_id, step));
        lds_UpdateTiles_ValueIndirectBuffer[group_thread_id.x][group_thread_id.y] = packed_radiance_indirect;
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index] = packed_radiance_indirect;
    }
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

        // Temporal accumulation. Lever E8 (cells lit on a budget): a cell not lit this frame keeps its
        // value and count, as the indirect values do (register D37); Capsaicin's blend would step the
        // count down each such frame and read 0 after 16, though nothing was learned about the cell.
        // Experiment E9: a cell marked near a recent change takes this frame's samples whole.
        let direct_scratch = HASHGRIDCACHE_UPDATECELLVALUEBUFFER + 4u * cell_index;
        let change_reset = (atomicAnd(&g_HashGridCache_UpdateScratch[direct_scratch + 3u], ~HASHGRIDCACHE_CHANGE_RESET_BIT) & HASHGRIDCACHE_CHANGE_RESET_BIT) != 0u;
        let previous_direct_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index]);
        let new_direct_samples = UpdateTiles_TakeScratch(direct_scratch);
        var direct_radiance = previous_direct_radiance;
        if (change_reset && new_direct_samples.w <= 0.0) {
            // Marked, and every hit in it had no light to draw (a light switched off): its light is none now.
            direct_radiance = vec4f(0.0);
        } else if (change_reset) {
            // The history counts as at most cellChangeResetHistory samples beside the frame's (0: replaced).
            let kept = min(HashGridCache_ChangeResetHistory(), previous_direct_radiance.w);
            let total = kept + new_direct_samples.w;
            let mean = (previous_direct_radiance.rgb / max(previous_direct_radiance.w, 1.0) * kept + new_direct_samples.rgb) / total;
            let count = min(total, g_HashGridCacheConstants.max_sample_count);
            direct_radiance = vec4f(mean * count, count);
        } else if (!HashGridCache_CellShadingBudgeted() || new_direct_samples.w > 0.0) {
            direct_radiance = UpdateTiles_Accumulate(previous_direct_radiance, new_direct_samples, g_HashGridCacheConstants.max_sample_count);
        }

        // Pack
        let packed_direct_radiance = HashGridCache_PackRadiance(direct_radiance);
        lds_UpdateTiles_ValueBuffer[group_thread_id.x][group_thread_id.y] = packed_direct_radiance;
        g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEBUFFER + cell_index] = packed_direct_radiance;

        // USE_MULTI_BOUNCE: the scratch UpdateMultibounceCells filled last frame (its one-frame lag).
        // A frame with no indirect sample leaves the value and its count as they are (register
        // D37): Capsaicin's blend takes such a frame as a step down (the mean kept, the count one
        // less, and at 0 the value reads 0), and with only a cell's kept rays counting (about 0.3
        // of its hits a frame) a sparse cell would spend many frames at 0. The direct values keep
        // Capsaicin's blend.
        if (g_HashGridCacheConstants.use_multibounce != 0u) {
            var previous_indirect_radiance = HashGridCache_UnpackRadiance(g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index]);
            // E9: the second bounce's history is cut to one sample (its mean kept), so it follows too.
            // One whatever the direct value keeps: its samples come at about 0.3 a frame per hit,
            // and kept as 3 samples the furnace's second bounce followed no faster than per hit.
            if (change_reset && previous_indirect_radiance.w > 1.0) {
                previous_indirect_radiance = vec4f(previous_indirect_radiance.rgb / previous_indirect_radiance.w, 1.0);
            }
            let new_indirect_samples = UpdateTiles_TakeScratch(HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER + 4u * cell_index);
            var indirect_radiance = previous_indirect_radiance;
            if (new_indirect_samples.w > 0.0) {
                indirect_radiance = UpdateTiles_Accumulate(previous_indirect_radiance, new_indirect_samples, g_HashGridCacheConstants.max_multibounce_sample_count);
            }
            let packed_indirect_radiance = HashGridCache_PackRadiance(indirect_radiance);
            lds_UpdateTiles_ValueIndirectBuffer[group_thread_id.x][group_thread_id.y] = packed_indirect_radiance;
            g_HashGridCache_CellValues[HASHGRIDCACHE_VALUEINDIRECTBUFFER + cell_index] = packed_indirect_radiance;
        }
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
 * `ResolveCells` (gi1.comp:2350-2385): each entry of the resolve list without
 * the bypass bit adds its cell's filtered direct radiance (a coarser level
 * while the cell has fewer than `max_sample_count` samples,
 * `HashGridCache_FilteredRadianceDirect`) and, with multibounce
 * (`USE_MULTI_BOUNCE`), its filtered indirect radiance to its probe ray.
 * Without multibounce the list holds the hits with a light only, Capsaicin's
 * shadow-ray list (register D35).
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
    if (g_HashGridCacheConstants.use_multibounce != 0u) {  // USE_MULTI_BOUNCE
        let indirect_radiance = HashGridCache_FilteredRadianceIndirect(cell_index);
        radiance += (indirect_radiance.rgb / max(indirect_radiance.w, 1.0));
    }
    ScreenProbes_AccumulateRadiance(query_index, radiance);
}
`;
}

// ---- GI-1.2's multibounce: PopulateMultibounceCells, GenerateMultibounceReservoirs, UpdateMultibounceCells ----

/**
 * Sampling a second-bounce direction (`sampleBRDFAndEvaluateDiffuse`,
 * material_sampling.hlsl:741-773) with Capsaicin's quaternion frame
 * (`QuaternionRotationZ` and `Quaternion`, quaternion.hlsl) and
 * `mapToCosineHemisphere` (sampling.hlsl:161-169), so the same two random
 * numbers give the same direction. The reflectance is Lambert's, albedo / π ×
 * cos (register D8): Capsaicin forces the second vertex diffuse (metallicity
 * 0, roughness 1, gi1.comp:1904-1908) but its default build still evaluates
 * GGX at roughness 1 with the diffuse compensation there. The CPU twin is
 * `multibounce-cpu.ts`.
 */
const MULTIBOUNCE_SAMPLING_WGSL = /* wgsl */ `
// INV_PI and TWO_PI (math/math_constants.hlsl).
const HASHGRIDCACHE_MULTIBOUNCE_INV_PI: f32 = 0.318309886183791;
const HASHGRIDCACHE_MULTIBOUNCE_TWO_PI: f32 = 6.28318530717959;

// QuaternionRotationZ (quaternion.hlsl:58-64): the rotation taking 'direction' to +z, with the
// special case of a direction at or near -z.
fn HashGridCache_QuaternionRotationZ(direction: vec3f) -> vec4f {
    if (direction.z >= -0.99999) {
        return normalize(vec4f(direction.y, -direction.x, 0.0, 1.0 + direction.z));
    }
    return vec4f(1.0, 0.0, 0.0, 0.0);
}

// Quaternion::inverse (quaternion.hlsl:37-42).
fn HashGridCache_QuaternionInverse(quaternion: vec4f) -> vec4f {
    return vec4f(-quaternion.xyz, quaternion.w);
}

// Quaternion::transform (quaternion.hlsl:49-55).
fn HashGridCache_QuaternionTransform(quaternion: vec4f, direction: vec3f) -> vec3f {
    let q_axis = quaternion.xyz;
    return 2.0 * dot(q_axis, direction) * q_axis + (quaternion.w * quaternion.w - dot(q_axis, q_axis)) * direction + 2.0 * quaternion.w * cross(q_axis, direction);
}

// mapToCosineHemisphere (sampling.hlsl:161-169): Ray Tracing Gems, "Sampling Transformations Zoo".
fn HashGridCache_MapToCosineHemisphere(samples: vec2f) -> vec3f {
    let sin_theta = sqrt(samples.x);
    let phi = HASHGRIDCACHE_MULTIBOUNCE_TWO_PI * samples.y;
    return vec3f(sin_theta * cos(phi), sin_theta * sin(phi), sqrt(1.0 - samples.x));
}

struct HashGridCache_BRDFSample {
    direction: vec3f,
    brdf: vec3f,    // reflectance toward the sampled direction, the cosine included
    pdf: f32,
}

// sampleBRDFAndEvaluateDiffuse (material_sampling.hlsl:741-773) for a Lambert surface: the view
// direction it also takes only matters to the GGX term, which the platform's materials don't have.
fn HashGridCache_SampleBRDFAndEvaluateDiffuse(albedo: vec3f, samples: vec2f, normal: vec3f) -> HashGridCache_BRDFSample {
    // Transform into the surface's tangent space (oriented so that z axis is aligned to normal)
    let local_rotation = HashGridCache_QuaternionRotationZ(normal);

    // Sample diffuse BRDF component (sampleLambert, material_sampling.hlsl:295-299)
    let new_light = HashGridCache_MapToCosineHemisphere(samples);

    // Evaluate BRDF for new light direction: evaluateLambert x saturate(dotNL) (material_evaluation.hlsl:157-160)
    let dot_nl = clamp(new_light.z, -1.0, 1.0);
    let reflectance = albedo * HASHGRIDCACHE_MULTIBOUNCE_INV_PI * saturate(dot_nl);

    // sampleLambertPDF (material_sampling.hlsl:306-309)
    let pdf = saturate(dot_nl) * HASHGRIDCACHE_MULTIBOUNCE_INV_PI;

    // Transform the new direction back into world space
    let light_direction = normalize(HashGridCache_QuaternionTransform(HashGridCache_QuaternionInverse(local_rotation), new_light));
    return HashGridCache_BRDFSample(light_direction, reflectance, pdf);
}
`;

/**
 * `PopulateMultibounceCells` (gi1.comp:1869-1946) with its hit handler
 * (`PopulateMultibounceCellsHandleHit`, gi1.comp:1761-1852): over the
 * first-bounce visibility entries (dispatched on `VisibilityCountBuffer0`,
 * gi1.cpp:2386-2390), each draws whether to stop (the discard, probability
 * `discard_multibounce_ray_probability`, 0.7), and otherwise traces one
 * closest-hit ray from its hit in a cosine-weighted direction. That ray's hit
 * is inserted as `PopulateScreenProbesHandleHit` inserts a probe ray's: its
 * cell, its tile's decay stamp, the packed and update lists; and it becomes a
 * second-bounce visibility entry, after the first-bounce ones, whose query is
 * the first-bounce entry and whose `MultibounceInfoBuffer` value is the
 * direction's BRDF and pdf.
 *
 * Built with the platform's hit-shading module (group 0), which brings the
 * tracer's `giTraceClosest`.
 */
export function populateMultibounceCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.populateMultibounceCells;
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${populationHeadWgsl(layout)}
${PROBE_RAY_SURFACE_WGSL}
${MULTIBOUNCE_SAMPLING_WGSL}

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
        var visibility_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1], 1u);
        let packed_brdf = HashGridCache_PackBRDF(brdf, pdf);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_info_buffer, ${MULTIBOUNCE_INFO_WORDS}u * visibility_index + 0u, packed_brdf.x);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_info_buffer, ${MULTIBOUNCE_INFO_WORDS}u * visibility_index + 1u, packed_brdf.y);

        // The hit (Capsaicin's HashGridCache_PackVisibility): the whole ray, in a probe ray's record (register D36).
        var multibounce_ray: WorldCacheProbeRay;
        multibounce_ray.origin = ray.origin;
        multibounce_ray.flags = 0u;
        multibounce_ray.direction = vec2u(pack2x16float(ray.direction.xy), pack2x16float(vec2f(ray.direction.z, 0.0)));
        multibounce_ray.hitT = hit.t;
        multibounce_ray.hitInstance = hit.instance;
        multibounce_ray.hitTriangle = hit.triangle;
        multibounce_ray.hitFrontFace = select(0u, 1u, hit.frontFace);
        multibounce_ray.hitBarycentrics = hit.barycentrics;
        HashGridCache_StoreMultibounceRay(visibility_index, multibounce_ray);

        visibility_index += atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index, cell_index);
        HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index, did);

        // requestLightSampleLocation(data.hit_position): the platform's light grid covers the
        // scene's bounds instead (register D4).

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
            let update_tile_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_UPDATETILECOUNTBUFFER], 1u);
            HashGridCache_SetRayListWord(g_HashGridCache_RayLists.update_tile_buffer, update_tile_index, tile_index);
        }
    }
}

// A kept second-bounce ray that hit nothing (register D37): an entry with no cell, so that
// UpdateMultibounceCells counts it as a sample of no light in the first-bounce cell.
fn PopulateMultibounceCellsHandleMiss(did: u32, brdf: vec3f, pdf: f32, ray: GiRay) {
    var visibility_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1], 1u);
    let packed_brdf = HashGridCache_PackBRDF(brdf, pdf);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_info_buffer, ${MULTIBOUNCE_INFO_WORDS}u * visibility_index + 0u, packed_brdf.x);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_info_buffer, ${MULTIBOUNCE_INFO_WORDS}u * visibility_index + 1u, packed_brdf.y);
    var multibounce_ray: WorldCacheProbeRay;
    multibounce_ray.origin = ray.origin;
    multibounce_ray.flags = 0u;
    multibounce_ray.direction = vec2u(pack2x16float(ray.direction.xy), pack2x16float(vec2f(ray.direction.z, 0.0)));
    multibounce_ray.hitT = -1.0;
    multibounce_ray.hitInstance = 0u;
    multibounce_ray.hitTriangle = 0u;
    multibounce_ray.hitFrontFace = 0u;
    multibounce_ray.hitBarycentrics = vec2f(0.0);
    HashGridCache_StoreMultibounceRay(visibility_index, multibounce_ray);
    visibility_index += atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, visibility_index, kGI1_InvalidId);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_query_buffer, visibility_index, did);
}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.populateMultibounceCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        return; // out of bounds
    }

    // Random sample
    var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);

    if (rcRand(&random) <= g_HashGridCacheConstants.discard_multibounce_ray_probability) {
        return;
    }

    // Load our visibility sample, and reconstruct the world-space position and normal (the
    // shading normal facing the ray's side, as Capsaicin flips it by the hit's face).
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did);
    let surface = HashGridCache_ProbeSurface(g_ProbeRays.rays[query_index]);

    // Force diffuse lighting on secondary bounce to reduce noise: the platform's materials are
    // diffuse already (register D8). Capsaicin recovers the probe's pixel here for the view
    // direction, which only its GGX term reads.

    // Sample diffuse direction
    let brdf_sample = HashGridCache_SampleBRDFAndEvaluateDiffuse(surface.albedo, rcRand2(&random), surface.normal);

    // Trace (MakeRayInfoClosest, trace_ray.hlsl:95-118: the origin pushed off the surface, here as
    // the platform pushes a ray's origin, on the side the probe ray arrived from; register D36)
    let ray = GiRay(giShadeOffsetRayOrigin(surface.position, giShadeRaySide(surface)), 0.0, brdf_sample.direction, RC_MAX_HIT_DISTANCE);
    let hit = giTraceClosest(ray);

    // If we hit some geometry, we append a new world-space hash-grid cache query. A kept ray that
    // hits nothing is an entry too, with no cell: it is a sample of no light that still counts
    // (register D37; Capsaicin's miss shader is empty, gi1.rt:120-123). The sky at the first hit
    // is the first hit's light, not this bounce's (register D33).
    if (hit.t >= 0.0) {
        PopulateMultibounceCellsHandleHit(did, brdf_sample.brdf, brdf_sample.pdf, ray, hit);
    } else {
        PopulateMultibounceCellsHandleMiss(did, brdf_sample.brdf, brdf_sample.pdf, ray);
    }
}
`;
}

/**
 * `GenerateMultibounceReservoirs` (gi1.comp:2603-2700): over the
 * second-bounce entries (dispatched on `VisibilityCountBuffer1`,
 * gi1.cpp:2427-2436), one light each (the platform's light choice, register
 * D4, as `GenerateReservoirs`), its shadow ray appended to the same list as
 * the first bounce's, flagged to bypass the cache when the second-bounce ray
 * is shorter than its cell. Capsaicin checks no emissive surface here: a
 * second-bounce hit is lit by its light sample only.
 *
 * Built with the platform's hit-shading module (group 0).
 */
export function generateMultibounceReservoirsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.generateMultibounceReservoirs;
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${populationHeadWgsl(layout)}
${PROBE_RAY_SURFACE_WGSL}
${CELL_SHADING_COMMON_WGSL}

const HASHGRIDCACHE_LIGHT_GROUPS: u32 = ${ALL_LIGHT_GROUPS}u;

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.generateMultibounceReservoirs}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    var did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1])) {
        return; // out of bounds
    }

    let first_bounce_count = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
    did += first_bounce_count;

    // A miss has no surface and no light (register D37).
    if (HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did) == kGI1_InvalidId) {
        return;
    }

    // Load our visibility sample (the second-bounce ray, register D36), and reconstruct the
    // world-space position and normal
    let ray = HashGridCache_LoadMultibounceRay(did - first_bounce_count);
    let surface = HashGridCache_ProbeSurface(ray);

    // Lever E8 (cells lit on a budget): the second hit offers itself as its cell's sample. A segment
    // at least a cell long reads its cell (UpdateMultibounceCells), which ShadeCells lights, so it
    // draws no light here; its entry's flags are written as below. A shorter segment keeps its own
    // light, so it goes on as per hit.
    if (HashGridCache_CellShadingBudgeted()) {
        HashGridCache_OfferCellSample(did, HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did));
        if (!HashGridCache_ShorterThanCell(ray.origin, surface.position)) {
            let budgeted_entry = did - first_bounce_count;
            HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_ray_buffer + ${MULTIBOUNCE_RAY_WORDS}u * budgeted_entry, 3u, 0u);
            HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_direct_buffer + ${MULTIBOUNCE_DIRECT_WORDS}u * budgeted_entry, 0u, 0u);
            HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_direct_buffer + ${MULTIBOUNCE_DIRECT_WORDS}u * budgeted_entry, 1u, 0u);
            return;
        }
    }

    // Experiment E9: a second hit near a change marks its cell (with a light or not).
    HashGridCache_MarkChangeReset(HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did), surface.position);

    // Recover the ray origin: the first-bounce hit. Capsaicin rebuilds that hit's point from its
    // visibility entry; the second-bounce ray starts there, pushed off by the platform's offset
    // (256 rounding steps of its largest coordinate: hundredths of a millimetre near the origin).
    let origin = ray.origin;

    // Sample new lights. Capsaicin forces a diffuse material on this bounce (gi1.comp:2662-2667);
    // the platform's materials are diffuse already (register D8). Its light list also holds the
    // environment map; the platform's doesn't (register D33).
    var random = rcMakeRandom(did, g_HashGridCacheConstants.frame_index);
    let reservoir = giDirectChooseLight(surface, HASHGRIDCACHE_LIGHT_GROUPS, rcRand(&random));
    let sample_params = rcRand2(&random);

    // The segment shorter than its cell: UpdateMultibounceCells then reads this hit's own light
    // instead of its cell's (register D38), kept on the entry's ray (its flags word) for every
    // hit entry, lit or not. Its own light starts at none; PopulateCells writes it if a light is drawn.
    let cell_size  = HashGridCache_GetCellSize(surface.position);
    let ray_length = distance(origin, surface.position);

    let bypass_cache = ray_length < cell_size;
    let bypass_bit = select(0u, HASHGRIDCACHE_BYPASS_BIT, bypass_cache);
    let entry = did - first_bounce_count;
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_ray_buffer + ${MULTIBOUNCE_RAY_WORDS}u * entry, 3u, select(0u, WORLD_CACHE_PROBE_RAY_BYPASS, bypass_cache));
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_direct_buffer + ${MULTIBOUNCE_DIRECT_WORDS}u * entry, 0u, 0u);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.multibounce_direct_buffer + ${MULTIBOUNCE_DIRECT_WORDS}u * entry, 1u, 0u);

    if (reservoir.light == GI_DIRECT_NO_LIGHT) {
        return;
    }

    // Append our resampled shadow ray
    let ray_index = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYRAYCOUNTBUFFER], 1u);
    HashGridCache_SetRayListWord(g_HashGridCache_RayLists.visibility_ray_buffer, ray_index, did | bypass_bit);   // compact the surviving shadow rays

    // Write the results out to memory: the reservoir (packReservoir), as GenerateReservoirs does.
    let reservoir_word = g_HashGridCache_RayLists.reservoir_buffer + ${RESERVOIR_WORDS}u * ray_index;
    HashGridCache_SetRayListWord(reservoir_word, 0u, reservoir.light);
    HashGridCache_SetRayListWord(reservoir_word, 1u, bitcast<u32>(reservoir.weight));
    HashGridCache_SetRayListWord(reservoir_word, 2u, pack2x16unorm(sample_params));
}
`;
}

/**
 * `UpdateMultibounceCells` (gi1.comp:1948-1983), run after `UpdateTiles`
 * (gi1.cpp:2506-2516) over the second-bounce entries: the light reaching the
 * first hit from the second (the second-bounce cell's filtered direct
 * radiance, this frame's since `UpdateTiles` just ran, or the hit's own light
 * where its segment is shorter than its cell) times BRDF / pdf is added,
 * quantised, into the indirect scratch of the first-bounce cell (the query's
 * cell); a miss adds a sample of no light. `UpdateTiles` blends that scratch
 * into the indirect values next frame: the one-frame lag.
 *
 * Two of Capsaicin's errors are fixed here (DECISIONS 2026-11-19):
 * - **The estimate** (register D37): Capsaicin divides each sample by
 *   1 - p and counts only kept rays that hit, so the cell's mean is the
 *   second bounce times 1 / (1 - p) times 1 / P(hit). Here a kept ray counts
 *   whether it hits or not, and the cell's mean over kept rays needs no 1 / (1 - p).
 * - **The bypass** (register D38): where the first bounce's own probe ray is
 *   shorter than its cell (its probe ray reads no cell), this entry's sample
 *   goes to that probe ray, weighted by 1 / (1 - p) since it is that ray's
 *   only draw; Capsaicin reads the flag from the shadow-ray list at a
 *   visibility index and adds to the probe ray numbered as the first-bounce
 *   entry. Each first-bounce entry has at most one second-bounce entry, so no
 *   two threads add to one probe ray.
 */
export function updateMultibounceCellsWgsl(layout: HashGridCacheBufferLayout): string {
    const size = POPULATION_WORKGROUP_SIZE.updateMultibounceCells;
    return /* wgsl */ `
${populationHeadWgsl(layout)}

@compute @workgroup_size(${size})
fn ${POPULATION_ENTRY_POINT.updateMultibounceCells}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1])) {
        return; // out of bounds
    }

    let did2 = did + atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did2);
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did2);   // the first-bounce entry
    let cell_index_base = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, query_index);   // BE CAREFUL: also a cell

    // The light arriving from the second hit, as BRDF / pdf weighs it (no light for a miss).
    var radiance = vec3f(0.0);
    // Lever E8 (cells lit on a budget): a second-hit cell no level of which has been lit yet holds
    // no estimate, not darkness, so it adds no sample (counting it as 0 would pull the first-bounce
    // cell's mean down until 16 later samples wash it out).
    var adds_sample = true;
    if (cell_index != kGI1_InvalidId) {
        let packed_info = vec2u(HashGridCache_RayListWord(g_HashGridCache_RayLists.multibounce_info_buffer, ${MULTIBOUNCE_INFO_WORDS}u * did + 0u),
                                HashGridCache_RayListWord(g_HashGridCache_RayLists.multibounce_info_buffer, ${MULTIBOUNCE_INFO_WORDS}u * did + 1u));
        let query_info = HashGridCache_UnpackBRDF(packed_info);
        let brdf = query_info.xyz;
        let pdf = query_info.w;

        var second_radiance = vec3f(0.0);
        let segment_bypasses = (HashGridCache_RayListWord(g_HashGridCache_RayLists.multibounce_ray_buffer + ${MULTIBOUNCE_RAY_WORDS}u * did, 3u) & WORLD_CACHE_PROBE_RAY_BYPASS) != 0u;
        if (segment_bypasses) {
            // The second hit's own light (PopulateCells), not its cell's (register D38).
            let base = g_HashGridCache_RayLists.multibounce_direct_buffer + ${MULTIBOUNCE_DIRECT_WORDS}u * did;
            second_radiance = vec3f(unpack2x16float(HashGridCache_RayListWord(base, 0u)), unpack2x16float(HashGridCache_RayListWord(base, 1u)).x);
        } else {
            let direct_radiance = HashGridCache_FilteredRadianceDirect(cell_index);
            second_radiance = direct_radiance.rgb / max(direct_radiance.w, 1.0);
            adds_sample = !HashGridCache_CellShadingBudgeted() || direct_radiance.w > 0.0;
        }
        // P6.4 (register E11): the neural tail's answer for the second hit, the light it reflects
        // from indirect light (every later bounce, as the network has learned the grid's indirect
        // values), so the first-bounce cell's indirect value carries every bounce in expectation.
        if (g_HashGridCacheConstants.use_neural_tail != 0u) {
            let tail_base = g_HashGridCache_RayLists.neural_tail_buffer + ${NEURAL_TAIL_WORDS}u * did;
            let tail = vec4f(unpack2x16float(HashGridCache_RayListWord(tail_base, 0u)), unpack2x16float(HashGridCache_RayListWord(tail_base, 1u)));
            if (tail.w > 0.0) {
                second_radiance += tail.rgb;
            }
        }
        radiance = (second_radiance * brdf) / pdf;
    }
    // No 1 / (1 - p): the cell's mean is over the kept rays, every one of which counts (register D37).
    let quantized_radiance = HashGridCache_QuantizeRadiance(radiance);
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER + 4u * cell_index_base;

    if (adds_sample) {
        if (dot(radiance, radiance) > 0.0) {
            atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 0u], quantized_radiance.x);
            atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 1u], quantized_radiance.y);
            atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 2u], quantized_radiance.z);
        }
        atomicAdd(&g_HashGridCache_UpdateScratch[scratch + 3u], quantized_radiance.w);
    }

    // The first bounce's probe ray reads no cell when it is shorter than its cell (register D38):
    // its indirect light is this draw, weighted by 1 / (1 - p) for the draws discarded.
    let first_bounce_bypasses = HashGridCache_RayListWord(g_HashGridCache_RayLists.first_bounce_bypass_buffer, query_index) != 0u;
    if (first_bounce_bypasses) {
        let probe_ray_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, query_index);
        ScreenProbes_AccumulateRadiance(probe_ray_index, GIDenoiser_RemoveNaNs(radiance / (1.0 - g_HashGridCacheConstants.discard_multibounce_ray_probability)));
    }
}
`;
}
