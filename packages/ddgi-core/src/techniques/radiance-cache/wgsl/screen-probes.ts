/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.comp
 * (ClearCounters, ClearProbeMask, SpawnScreenProbes, CompactScreenProbes,
 * SampleScreenProbes, PopulateScreenProbes, BlendScreenProbes).
 *
 * Copyright © 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files(the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions :
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * The screen-probe kernels, one WGSL entry point per Capsaicin kernel, named
 * as Capsaicin names it, with the lines it ports. Each kernel's own resources
 * are in group 1 (the platform's are group 0); the layouts are next to the
 * code.
 *
 * Phase 2 (RC2.1) built the spawn, the compaction, the sampling, the trace
 * and the blend for a frame with no history. Phase 4 (RC4.1) adds the
 * history: `CompactScreenProbes` lists the override tiles for
 * `PatchScreenProbes`, `SampleScreenProbes` reads the 3 × 3 reprojected
 * probes and the off-screen probe cache and guides its rays by the luminance
 * CDF, and `BlendScreenProbes` blends with Capsaicin's biased hysteresis and
 * writes the evicted and updated probes into the cache. The history kernels
 * themselves (`ReprojectScreenProbes`, the cache's count, scatter and reorder,
 * the patch) are in `screen-probe-history.ts`. Since RC3.2 the probe rays'
 * hits are lit by the world cache between `PopulateScreenProbes` and
 * `BlendScreenProbes`.
 */

import { CAPSAICIN_WGSL } from './capsaicin.ts';

/** `GPUShaderStage.COMPUTE`, as a number: the unit tests import this file in Node, which has no `GPUShaderStage`. */
const COMPUTE = 0x4;

/** Workgroup sizes, as Capsaicin's `[numthreads]`. */
export const WORKGROUP_SIZE = {
    /** ClearProbeMask (gi1.comp:242): 8 × 8. */
    clearProbeMask: 8,
    /** SpawnScreenProbes, CompactScreenProbes (gi1.comp:555, 581). */
    spawn: 64,
    /** SampleScreenProbes, BlendScreenProbes (gi1.comp:646, 1166): one probe's 64 rays per workgroup. */
    probe: 64,
    /** PopulateScreenProbesMain (gi1.comp:1160): half a probe's rays per workgroup. */
    populate: 32,
} as const;

/** Entry points: Capsaicin's kernel names. */
export const ENTRY_POINT = {
    clearCounters: 'ClearCounters',
    clearProbeMask: 'ClearProbeMask',
    spawn: 'SpawnScreenProbes',
    compact: 'CompactScreenProbes',
    sample: 'SampleScreenProbes',
    populate: 'PopulateScreenProbesMain',
    blend: 'BlendScreenProbes',
} as const;

/** Bytes of `RcFrame`. */
export const FRAME_UNIFORM_BYTES = 288;

/**
 * Words of `RcCounters`: Capsaicin's append counters (`ClearCounters`,
 * gi1.comp:163-185), one word each, in its order. Phase 4 uses the screen
 * probes' five: the empty and override tiles and the probe cache's LRU, MRU
 * and list-element counts. The world cache keeps its own counters.
 */
export const COUNTER_NAMES = [
    'reservoirHashListCount',
    'emptyTileCount',
    'overrideTileCount',
    'cachedTileLRUCount',
    'cachedTileMRUCount',
    'cachedTileListElementCount',
    'packedTileCount',
    'updateTileCount',
    'visibilityCount0',
    'visibilityRayCount',
    'visibilityCount1',
    'resolveCount',
    'rtSampleCount',
] as const;

export const COUNTERS_BYTES = 64;

/** The word of each counter in `RcCounters`. */
export const COUNTER_INDEX = Object.fromEntries(COUNTER_NAMES.map((name, index) => [name, index])) as { readonly [Name in (typeof COUNTER_NAMES)[number]]: number };

/** Byte offset of a counter, for indirect arguments sized by it. */
export function counterByteOffset(name: (typeof COUNTER_NAMES)[number]): number {
    return COUNTER_INDEX[name] * 4;
}

/** The counters' words as WGSL constants (`RC_COUNTER_EMPTY_TILE` and so on), for the kernels that bind them. */
export const COUNTERS_WGSL = /* wgsl */ `
const RC_COUNTER_WORDS: u32 = ${COUNTERS_BYTES / 4}u;
const RC_COUNTER_EMPTY_TILE: u32 = ${COUNTER_NAMES.indexOf('emptyTileCount')}u;               // g_ScreenProbes_EmptyTileCountBuffer
const RC_COUNTER_OVERRIDE_TILE: u32 = ${COUNTER_NAMES.indexOf('overrideTileCount')}u;         // g_ScreenProbes_OverrideTileCountBuffer
const RC_COUNTER_CACHED_TILE_LRU: u32 = ${COUNTER_NAMES.indexOf('cachedTileLRUCount')}u;      // g_ScreenProbes_ProbeCachedTileLRUCountBuffer
const RC_COUNTER_CACHED_TILE_MRU: u32 = ${COUNTER_NAMES.indexOf('cachedTileMRUCount')}u;      // g_ScreenProbes_ProbeCachedTileMRUCountBuffer
const RC_COUNTER_CACHED_TILE_LIST_ELEMENT: u32 = ${COUNTER_NAMES.indexOf('cachedTileListElementCount')}u; // g_ScreenProbes_ProbeCachedTileListElementCountBuffer
`;

/**
 * The frame's constants: Capsaicin's `g_Eye`, `g_PreviousEye`,
 * `g_FrameIndex`, `g_BufferDimensions`, `g_UseDirectLighting`, the
 * `GI1Constants` matrices (gi1.comp:33-41, 95-98) and
 * `ScreenProbesConstants.cell_size` (gi1.cpp:1839-1843), plus how the host's
 * screen buffers are encoded and where the probe cache's arrays start in its
 * packed buffer (`probe-cache.ts`, register D6).
 *
 * The matrices are camera-relative: the inverse view-projections give a
 * point relative to their frame's eye (added after the divide), and the
 * view-projection takes a point relative to `eye`. The same positions as
 * Capsaicin's `transformPointProjection`, without the f32 cancellation of a
 * camera far from the origin. The previous frame's are the previous camera's
 * (Capsaicin's `view_proj_inv_prev`, gi1.cpp:1820), kept by the technique
 * (register D49).
 */
export const FRAME_WGSL = /* wgsl */ `
struct RcFrame {
    inverseViewProjection: mat4x4f,          // inverse(projection x view), with the view's translation taken out
    previousInverseViewProjection: mat4x4f,  // g_PreviousViewProjectionInverse, relative to previousEye
    viewProjection: mat4x4f,                 // g_ViewProjection, for points relative to eye
    eye: vec3f,                              // g_Eye
    frameIndex: u32,                         // g_FrameIndex
    previousEye: vec3f,                      // g_PreviousEye
    cellSize: f32,                           // ScreenProbesConstants.cell_size, per unit distance
    bufferDimensions: vec2u,                 // g_BufferDimensions: the screen in pixels
    probeCount: vec2u,                       // probe tiles (8 x 8 pixels) across and down
    maxProbeSpawnCount: u32,                 // spawn tiles (16 x 16 pixels)
    normalUnorm: u32,                        // 1: normals are 0..1 encoded (decoded as 2n - 1), 0: signed floats
    reversedDepth: u32,                      // 1: depth 1 is near, so the sky is at 0
    useDirectLighting: u32,                  // g_UseDirectLighting (gi1_use_direct_lighting)
    cachedTileLRUOffset: u32,                // word of g_ScreenProbes_ProbeCachedTileLRUBuffer (this frame's half)
    previousCachedTileLRUOffset: u32,        // word of g_ScreenProbes_PreviousProbeCachedTileLRUBuffer (the half Reorder writes)
    cachedTileMRUOffset: u32,                // word of g_ScreenProbes_ProbeCachedTileMRUBuffer
    cachedTileListOffset: u32,               // word of g_ScreenProbes_ProbeCachedTileListBuffer
    cachedTileListElementOffset: u32,        // word of g_ScreenProbes_ProbeCachedTileListElementBuffer (uint4 each)
    emptyTileOffset: u32,                    // word of g_ScreenProbes_EmptyTileBuffer
    overrideTileOffset: u32,                 // word of g_ScreenProbes_OverrideTileBuffer
    spawnProbeOffset: u32,                   // word of g_ScreenProbes_ProbeSpawnProbeBuffer (uint2 each)
}
@group(1) @binding(0) var<uniform> rcFrame: RcFrame;
`;

/**
 * Byte offsets of `RcFrame`'s fields, by WGSL's layout rules (a `mat4x4f` is
 * 64 bytes; a `vec3f` aligns to 16 and the scalar after it packs into its
 * last 4 bytes). `writeFrameUniforms` writes each field by name; a unit test
 * checks these against the struct above.
 */
export const FRAME_UNIFORM_OFFSETS = {
    inverseViewProjection: 0,
    previousInverseViewProjection: 64,
    viewProjection: 128,
    eye: 192,
    frameIndex: 204,
    previousEye: 208,
    cellSize: 220,
    bufferDimensions: 224,
    probeCount: 232,
    maxProbeSpawnCount: 240,
    normalUnorm: 244,
    reversedDepth: 248,
    useDirectLighting: 252,
    cachedTileLRUOffset: 256,
    previousCachedTileLRUOffset: 260,
    cachedTileMRUOffset: 264,
    cachedTileListOffset: 268,
    cachedTileListElementOffset: 272,
    emptyTileOffset: 276,
    overrideTileOffset: 280,
    spawnProbeOffset: 284,
} as const;

/** The host's depth (`FrameScreen.depth`) and the reads Capsaicin makes of its own depth buffer. */
export function screenDepthWgsl(binding: number): string {
    return /* wgsl */ `
@group(1) @binding(${binding}) var rcDepth: texture_2d<f32>;

fn rcLoadDepth(pixel: vec2u) -> f32 {
    return textureLoad(rcDepth, pixel, 0).x;
}

// Capsaicin tells sky pixels by a zero geometry normal (gi1.comp:570-571). The host's
// screen buffers say it by depth: a pixel at the far plane drew nothing (register D17).
fn rcIsSkyPixel(pixel: vec2u) -> bool {
    let farDepth = select(1.0, 0.0, rcFrame.reversedDepth != 0u);
    return rcLoadDepth(pixel) == farDepth;
}

// transformPointProjection(uv, depth, g_ViewProjectionInverse) (transform.hlsl:110-113, 94-99):
// the world position seen through the pixel at this depth (camera-relative, see RcFrame).
fn rcTransformPointProjection(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(2.0 * uv.x - 1.0, 2.0 * (1.0 - uv.y) - 1.0, depth, 1.0);
    let homogeneous = rcFrame.inverseViewProjection * ndc;
    return rcFrame.eye + homogeneous.xyz / homogeneous.w;
}
`;
}

/** The host's normals (`FrameScreen.normal`), as Capsaicin reads its geometry normals. */
export function screenNormalWgsl(binding: number): string {
    return /* wgsl */ `
@group(1) @binding(${binding}) var rcNormal: texture_2d<f32>;

// normalize(2.0f * g_GeometryNormalBuffer.Load(...).xyz - 1.0f) (gi1.comp:665), for either
// encoding FrameScreen allows.
fn rcLoadNormal(pixel: vec2u) -> vec3f {
    let stored = textureLoad(rcNormal, pixel, 0).xyz;
    return normalize(select(stored, 2.0 * stored - 1.0, rcFrame.normalUnorm != 0u));
}
`;
}

/** The workgroup a 2D-split dispatch is on, counted in rows (the platform's indirect arguments split the same way). */
export const WORKGROUP_INDEX_WGSL = /* wgsl */ `
fn rcWorkgroupIndex(workgroup: vec3u, workgroups: vec3u) -> u32 {
    return workgroup.y * workgroups.x + workgroup.x;
}
`;

/**
 * The off-screen probe cache's entries and the forward projection that finds
 * them on the screen, shared by every kernel that touches the cache. Needs
 * `FRAME_WGSL` and `CAPSAICIN_WGSL` in the same module.
 */
export const PROBE_CACHE_WGSL = /* wgsl */ `
// High bit of a g_ScreenProbes_ProbeCachedTileLRUBuffer entry: the slot holds no cached probe yet
// (InitCachedTileLRU, gi1.comp:287-290); in ProbeSpawnProbeBuffer, a slot allocated this frame.
const RC_CACHED_TILE_UNALLOCATED: u32 = 0x80000000u;
// The normal test for reusing a probe or a cached tile (gi1.comp:340, 851, 878).
const RC_REUSE_NORMAL_COSINE: f32 = 0.95;

// transformPointProjection(world_pos, g_ViewProjection) (transform.hlsl:94-99), with the
// camera-relative matrix (RcFrame): clip space, divided by w.
fn rcTransformPointProjectionForward(world: vec3f) -> vec3f {
    let homogeneous = rcFrame.viewProjection * vec4f(world - rcFrame.eye, 1.0);
    return homogeneous.xyz / homogeneous.w;
}

// A g_ScreenProbes_ProbeCachedTileIndexBuffer texel: the cached probe's world position and its
// packed normal (packNormal, which Capsaicin stores with asfloat in the w channel, gi1.comp:1306),
// held as integer bits in an rgba32uint texture (register D47).
fn rcCachedTileEntry(world: vec3f, normal: vec3f) -> vec4u {
    return vec4u(bitcast<vec3u>(world), rcPackNormal(normal));
}

fn rcCachedTilePosition(entry: vec4u) -> vec3f {
    return bitcast<vec3f>(entry.xyz);
}

fn rcCachedTileNormal(entry: vec4u) -> vec3f {
    return normalize(rcUnpackNormal(entry.w));
}
`;

/**
 * What the reprojecting kernels share: positions rebuilt from the screen's
 * depth at a probe's seed, and the cell directions of the hit-point
 * reprojection (gi1.comp:740-772, repeated in ReprojectScreenProbes,
 * SampleScreenProbes' taps and its cached tiles). Needs `screenDepthWgsl`,
 * `screenNormalWgsl` and `PROBE_CACHE_WGSL` in the same module.
 */
export const HISTORY_WGSL = /* wgsl */ `
${PROBE_CACHE_WGSL}
// A probe's world position: transformPointProjection((seed + 0.5) / g_BufferDimensions,
// g_DepthBuffer[seed], g_ViewProjectionInverse) (gi1.comp:669-670, 708-710).
fn rcProbeWorldPosition(seed: vec2u) -> vec3f {
    let uv = (vec2f(seed) + 0.5) / vec2f(rcFrame.bufferDimensions);
    return rcTransformPointProjection(uv, rcLoadDepth(seed));
}

// The direction through a probe cell's centre, in world space, for a probe of this normal:
// mapToHemiOctahedron((cell + 0.5) / probe_size) turned by GetOrthoVectors (gi1.comp:719-724).
fn rcCellDirection(cell: vec2u, probeNormal: vec3f) -> vec3f {
    return rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(cell) + 0.5) / f32(RC_PROBE_SIZE)), probeNormal);
}

// The probe cell a direction falls in, for a probe of this normal:
// uint2(mapToHemiOctahedronInverse(mul(direction, CreateTBN(normal))) * probe_size) (gi1.comp:762-764).
// Kept inside the probe (register D51): the map gives exactly 1 on the square's edge, where
// Capsaicin's index would leave the 64 cells of its workgroup arrays.
fn rcRemapCellIndex(direction: vec3f, normal: vec3f) -> u32 {
    let remapCell = min(vec2u(rcMapToHemiOctahedronInverse(rcToProbeFrame(direction, normal)) * f32(RC_PROBE_SIZE)), vec2u(RC_PROBE_SIZE - 1u));
    return remapCell.x + remapCell.y * RC_PROBE_SIZE;
}
`;

export const UNIFORM_ENTRY: GPUBindGroupLayoutEntry = { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform', minBindingSize: FRAME_UNIFORM_BYTES } };
const SCREEN_TEXTURE: GPUTextureBindingLayout = { sampleType: 'unfilterable-float', viewDimension: '2d' };

export function storage(binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, buffer: { type } };
}

export function texture(binding: number): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, texture: SCREEN_TEXTURE };
}

export function uintTexture(binding: number): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } };
}

export function storageTexture(binding: number, format: GPUTextureFormat, access: GPUStorageTextureAccess = 'write-only'): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, storageTexture: { access, format, viewDimension: '2d' } };
}

// ---- ClearCounters ------------------------------------------------------------------------------

export const CLEAR_COUNTERS_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ClearCounters group 1',
    entries: [storage(0, 'storage')],
};

export const CLEAR_COUNTERS_WGSL = /* wgsl */ `
// Capsaicin's append counters, in the order ClearCounters resets them (gi1.comp:166-184).
struct RcCounters {
    words: array<u32, 16>,
}
@group(1) @binding(0) var<storage, read_write> rcCounters: RcCounters;

// ClearCounters (gi1.comp:163-185): the frame starts with every list empty.
@compute @workgroup_size(1)
fn ${ENTRY_POINT.clearCounters}() {
    for (var counter = 0u; counter < ${COUNTER_NAMES.length}u; counter++) {
        rcCounters.words[counter] = 0u;
    }
}
`;

// ---- ClearProbeMask -----------------------------------------------------------------------------

export const CLEAR_PROBE_MASK_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ClearProbeMask group 1',
    entries: [storageTexture(0, 'r32uint')],
};

export const CLEAR_PROBE_MASK_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
// One mip level of a probe mask.
@group(1) @binding(0) var rcProbeMask: texture_storage_2d<r32uint, write>;

// ClearProbeMask (gi1.comp:242-246): run on every level of both probe masks when they are
// made (gi1.cpp:153-178), and on resetHistory. The texture's own size bounds it: Capsaicin
// dispatches whole groups and D3D drops the writes past the edge.
@compute @workgroup_size(8, 8)
fn ${ENTRY_POINT.clearProbeMask}(@builtin(global_invocation_id) did: vec3u) {
    if (any(did.xy >= textureDimensions(rcProbeMask))) {
        return;
    }
    textureStore(rcProbeMask, did.xy, vec4u(RC_INVALID_ID, 0u, 0u, 0u));
}
`;

// ---- SpawnScreenProbes --------------------------------------------------------------------------

export const SPAWN_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/SpawnScreenProbes group 1',
    entries: [UNIFORM_ENTRY, texture(1), texture(2), storage(3, 'storage'), storage(4, 'storage')],
};

export const SPAWN_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenDepthWgsl(1)}
${screenNormalWgsl(2)}
// g_ScreenProbes_PreviousProbeSpawnBuffer: each spawn tile's seed, packed.
@group(1) @binding(3) var<storage, read_write> rcPreviousProbeSpawn: array<u32>;
// g_ScreenProbes_ProbeSpawnScanBuffer: 1 for a tile that spawns a probe (the compaction's flags).
@group(1) @binding(4) var<storage, read_write> rcProbeSpawnScan: array<u32>;

// SpawnScreenProbes (gi1.comp:555-579): one candidate probe per 16 x 16 tile, at this
// frame's Halton jitter; a tile whose pixel is sky spawns none.
@compute @workgroup_size(64)
fn ${ENTRY_POINT.spawn}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.x;
    let maxProbeSpawnWidth = (rcFrame.bufferDimensions.x + RC_PROBE_SPAWN_TILE_SIZE - 1u) / RC_PROBE_SPAWN_TILE_SIZE;
    let maxProbeSpawnHeight = (rcFrame.bufferDimensions.y + RC_PROBE_SPAWN_TILE_SIZE - 1u) / RC_PROBE_SPAWN_TILE_SIZE;
    if (did >= maxProbeSpawnWidth * maxProbeSpawnHeight) {
        return; // out of bounds
    }

    let probe = vec2u(did % maxProbeSpawnWidth, did / maxProbeSpawnWidth);
    let tileSize = f32(RC_PROBE_SPAWN_TILE_SIZE);
    let jitter = vec2u(min(rcCalculateHaltonSequence(rcFrame.frameIndex) * tileSize, vec2f(tileSize - 1.0)));
    let seed = min(probe * RC_PROBE_SPAWN_TILE_SIZE + jitter, rcFrame.bufferDimensions - 1u);

    let isSkyPixel = rcIsSkyPixel(seed);
    if (!isSkyPixel) {
        rcPreviousProbeSpawn[did] = rcPackSeed(seed);
    }
    rcProbeSpawnScan[did] = select(0u, 1u, !isSkyPixel);
}
`;

// ---- CompactScreenProbes ------------------------------------------------------------------------

export const COMPACT_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/CompactScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        storage(1, 'read-only-storage'),
        storage(2, 'read-only-storage'),
        storage(3, 'read-only-storage'),
        storage(4, 'storage'),
        uintTexture(5),
        storage(6, 'storage'),
        storage(7, 'storage'),
    ],
};

/** The compacted probe list: the count in word 0 (copied from the compaction), then one packed seed per probe. */
export const PROBE_SPAWN_STRUCT_WGSL = /* wgsl */ `
// g_ScreenProbes_ProbeSpawnBuffer, with the probe count Capsaicin reads from the end of
// its scan (scan[last] + index[last], gi1.comp:651-652) kept in front of it.
struct RcProbeSpawn {
    count: u32,
    seeds: array<u32>,
}
`;

/** Byte offset of `RcProbeSpawn.seeds`. */
export const PROBE_SPAWN_SEEDS_OFFSET = 4;

/** Bytes before the first `RcProbeRay` (the count and its padding). */
export const PROBE_RAYS_HEADER_BYTES = 16;
/** Bytes of one `RcProbeRay`. */
export const PROBE_RAY_BYTES = 48;
/** `RcProbeRay.flags`: the ray is lit already, so the world cache adds nothing to it (no ray sets it since RC3.2). */
export const PROBE_RAY_BYPASS = 1;

/**
 * The frame's probe rays, shaped for phase 3's world-cache interface
 * (docs/design/radiance-cache.md, "The world-cache interface"): per ray its
 * origin, direction, hit record and bypass flag, with the ray count on the
 * GPU. The radiance slot is the ray scratch's result for the same ray, which
 * gi/direct adds into and `BlendScreenProbes` reads.
 */
const PROBE_RAYS_STRUCT_WGSL = /* wgsl */ `
// RcProbeRay.flags: already lit by the screen cache, so the world cache leaves it alone. Since
// RC3.2 no ray sets it: Capsaicin's bypass cases (gi1.comp:2439-2452, 2533-2583) are decided
// and lit inside the world cache.
const RC_PROBE_RAY_BYPASS: u32 = ${PROBE_RAY_BYPASS}u;

struct RcProbeRay {
    origin: vec3f,           // where the ray starts, pushed off the surface
    flags: u32,              // RC_PROBE_RAY_BYPASS
    direction: vec2u,        // g_ScreenProbes_ProbeSpawnSampleBuffer: three halves (packHalf3)
    // The hit, in the platform's GiHitRecord order: distance (negative for a miss),
    // instance, triangle, front face (1) or back (0), barycentrics of vertices 1 and 2.
    hitT: f32,
    hitInstance: u32,
    hitTriangle: u32,
    hitFrontFace: u32,
    hitBarycentrics: vec2f,
}

struct RcProbeRays {
    count: u32,              // this frame's rays: 64 x the probe count (SampleScreenProbes writes it)
    reserved: array<u32, 3>, // scalars, so the rays start at byte 16 (a vec3u would align to 16 and push them to 32)
    rays: array<RcProbeRay>,
}
`;

/** Built with the platform's `compute.indirectModule` (`GiIndirectArgs`, `giIndirectItemIndex`). */
export const COMPACT_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${COUNTERS_WGSL}
${PROBE_SPAWN_STRUCT_WGSL}
// The compaction's dispatch arguments, holding the kept count.
@group(1) @binding(1) var<storage, read> rcCompactArgs: GiIndirectArgs;
// The platform's stream compaction of the spawn flags: kept spawn tiles in increasing order.
@group(1) @binding(2) var<storage, read> rcKeptSpawnTiles: array<u32>;
@group(1) @binding(3) var<storage, read> rcPreviousProbeSpawn: array<u32>;
@group(1) @binding(4) var<storage, read_write> rcProbeSpawn: RcProbeSpawn;
// g_ScreenProbes_ProbeMaskBuffer, level 0, as ReprojectScreenProbes left it this frame.
@group(1) @binding(5) var rcProbeMaskBuffer: texture_2d<u32>;
@group(1) @binding(6) var<storage, read_write> rcCounters: array<atomic<u32>, RC_COUNTER_WORDS>;
// The probe cache's packed words (probe-cache.ts): here g_ScreenProbes_OverrideTileBuffer.
@group(1) @binding(7) var<storage, read_write> rcProbeCacheWords: array<u32>;

// CompactScreenProbes (gi1.comp:581-613). Capsaicin runs a thread per spawn tile and
// places each kept one at its exclusive prefix (g_ScreenProbes_ProbeSpawnIndexBuffer,
// from gfxCommandScanSum); the platform's compaction has listed the kept tiles in the
// same order, so a thread per kept tile writes the same list.
@compute @workgroup_size(64)
fn ${ENTRY_POINT.compact}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let probeIndex = giIndirectItemIndex(workgroup, workgroups, localIndex, 64u);
    if (probeIndex >= rcCompactArgs.count) {
        return; // out of bounds
    }
    let did = rcKeptSpawnTiles[probeIndex];
    let probeSeed = rcPreviousProbeSpawn[did];
    let probeMask = textureLoad(rcProbeMaskBuffer, rcUnpackSeed(probeSeed) / RC_PROBE_SIZE, 0).x;

    // If we're not filling a hole from the reprojection, we can append this tile to the
    // list of "overridable tiles"; that is, this is a valid candidate to be un-spawned in
    // favor of a disoccluded "empty tile" during the patching of the probes (gi1.comp:601-610).
    if (probeMask != RC_INVALID_ID) {
        let overrideTileIndex = atomicAdd(&rcCounters[RC_COUNTER_OVERRIDE_TILE], 1u);
        rcProbeCacheWords[rcFrame.overrideTileOffset + overrideTileIndex] = probeIndex;
    }

    rcProbeSpawn.seeds[probeIndex] = probeSeed;
}
`;

// ---- SampleScreenProbes -------------------------------------------------------------------------

export const SAMPLE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/SampleScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        texture(1),
        texture(2),
        uintTexture(3),
        texture(4),
        texture(5),
        uintTexture(6),
        storage(7, 'read-only-storage'),
        storage(8, 'storage'),
        storage(9, 'storage'),
        storage(10, 'storage'),
        storage(11, 'read-only-storage'),
        storage(12, 'read-only-storage'),
        storage(13, 'storage'),
        storageTexture(14, 'rgba16float'),
        storageTexture(15, 'rgba16float'),
    ],
};

/**
 * The workgroup arrays the reprojecting kernels accumulate into
 * (screen_probes.hlsl:82-86), and the atomic sum of a reprojected cell into
 * them (gi1.comp:762-772).
 */
export const RADIANCE_ACCUMULATION_WGSL = /* wgsl */ `
// lds_ScreenProbes_RadianceValues and lds_ScreenProbes_RadianceSampleCounts: per-cell sums of
// quantized radiance and hit distance, and how many samples each cell got.
var<workgroup> rcRadianceValues: array<atomic<u32>, 256>;
var<workgroup> rcRadianceSampleCounts: array<atomic<u32>, 64>;

fn rcClearRadianceValues(localId: u32) {
    atomicStore(&rcRadianceValues[(localId << 2u) + 0u], 0u);
    atomicStore(&rcRadianceValues[(localId << 2u) + 1u], 0u);
    atomicStore(&rcRadianceValues[(localId << 2u) + 2u], 0u);
    atomicStore(&rcRadianceValues[(localId << 2u) + 3u], 0u);
    atomicStore(&rcRadianceSampleCounts[localId], 0u);
}

// The radiance (and length) one sample brings into its cell.
fn rcAccumulateRadiance(cellIndex: u32, radianceAndLength: vec4f) {
    let quantized = rcQuantizeRadiance(radianceAndLength);
    atomicAdd(&rcRadianceValues[(cellIndex << 2u) + 0u], quantized.x);
    atomicAdd(&rcRadianceValues[(cellIndex << 2u) + 1u], quantized.y);
    atomicAdd(&rcRadianceValues[(cellIndex << 2u) + 2u], quantized.z);
    atomicAdd(&rcRadianceValues[(cellIndex << 2u) + 3u], quantized.w);
    atomicAdd(&rcRadianceSampleCounts[cellIndex], 1u);
}

fn rcLoadRadianceValues(localId: u32) -> vec4f {
    return rcRecoverRadiance(vec4u(
        atomicLoad(&rcRadianceValues[(localId << 2u) + 0u]),
        atomicLoad(&rcRadianceValues[(localId << 2u) + 1u]),
        atomicLoad(&rcRadianceValues[(localId << 2u) + 2u]),
        atomicLoad(&rcRadianceValues[(localId << 2u) + 3u]),
    ));
}
`;

/** The radiance backup for cells no sample reached (gi1.comp:1212-1234; also 412-434): needs `RADIANCE_ACCUMULATION_WGSL`. */
export const RADIANCE_BACKUP_WGSL = /* wgsl */ `
// lds_ScreenProbes_RadianceBackup.
var<workgroup> rcRadianceBackup: array<vec4f, 64>;

// Calculate the radiance backup value to be used for untraced cells: the mean of the reached
// cells' sums, divided among the empty ones. Every thread of the workgroup calls it (barriers).
// Returns the backup (rgb, and MAX_HIT_DISTANCE as the hit distance).
fn rcRadianceBackupValue(localId: u32) -> vec4f {
    let ownSampleCount = atomicLoad(&rcRadianceSampleCounts[localId]);
    rcRadianceBackup[localId] = vec4f(rcLoadRadianceValues(localId).xyz, select(0.0, 1.0, ownSampleCount > 0u));
    workgroupBarrier();

    for (var stride = 1u; stride < 64u; stride <<= 1u) {
        if (localId < 64u / (2u * stride)) {
            rcRadianceBackup[2u * (localId + 1u) * stride - 1u] += rcRadianceBackup[(2u * localId + 1u) * stride - 1u];
        }
        workgroupBarrier();
    }

    if (localId == 0u) {
        let totalRadiance = rcRadianceBackup[64u - 1u];
        let radiance = totalRadiance.xyz / max(totalRadiance.w, 1.0);
        let emptyCellCount = f32(RC_PROBE_CELLS) - totalRadiance.w;
        rcRadianceBackup[0] = vec4f(radiance / max(emptyCellCount, 1.0), RC_MAX_HIT_DISTANCE);
    }
    workgroupBarrier();
    return rcRadianceBackup[0];
}
`;

export const SAMPLE_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${COUNTERS_WGSL}
${screenDepthWgsl(1)}
${screenNormalWgsl(2)}
${HISTORY_WGSL}
${PROBE_SPAWN_STRUCT_WGSL}
${PROBE_RAYS_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
${RADIANCE_ACCUMULATION_WGSL}
// g_ScreenProbes_ProbeMaskBuffer (level 0) and g_ScreenProbes_ProbeBuffer as ReprojectScreenProbes
// left them: this frame's reprojected probes.
@group(1) @binding(3) var rcProbeMaskBuffer: texture_2d<u32>;
@group(1) @binding(4) var rcProbeBuffer: texture_2d<f32>;
// The off-screen probe cache: g_ScreenProbes_ProbeCachedTileBuffer (an atlas of cached probes)
// and g_ScreenProbes_ProbeCachedTileIndexBuffer (each cached probe's position and normal).
@group(1) @binding(5) var rcProbeCachedTileBuffer: texture_2d<f32>;
@group(1) @binding(6) var rcProbeCachedTileIndexBuffer: texture_2d<u32>;
@group(1) @binding(7) var<storage, read> rcProbeSpawn: RcProbeSpawn;
// The probe rays: this kernel writes their count and directions.
@group(1) @binding(8) var<storage, read_write> rcProbeRays: RcProbeRays;
@group(1) @binding(9) var<storage, read_write> rcCounters: array<atomic<u32>, RC_COUNTER_WORDS>;
// The probe cache's packed words (probe-cache.ts): the LRU, the lists, ProbeSpawnProbeBuffer.
@group(1) @binding(10) var<storage, read_write> rcProbeCacheWords: array<u32>;
@group(1) @binding(11) var<storage, read> rcProbeCachedTileListCountBuffer: array<u32>;
@group(1) @binding(12) var<storage, read> rcProbeCachedTileListIndexBuffer: array<u32>;
@group(1) @binding(13) var<storage, read_write> rcProbeCachedTileLRUFlagBuffer: array<atomic<u32>>;
// g_ScreenProbes_PreviousProbeBuffer: each spawned probe's reprojected history, for BlendScreenProbes.
@group(1) @binding(14) var rcPreviousProbeBuffer: texture_storage_2d<rgba16float, write>;
// The reprojected radiance of the probe this one replaces, for BlendScreenProbes' eviction (register D46).
@group(1) @binding(15) var rcEvictedProbeBuffer: texture_storage_2d<rgba16float, write>;

// lds_ScreenProbes_RadianceReuseSampleCount and lds_ScreenProbes_Radiance (the sampling CDF).
var<workgroup> rcRadianceReuseSampleCount: atomic<u32>;
var<workgroup> rcRadianceCdf: array<f32, 64>;

// InterlockedCompareExchange(g_ScreenProbes_ProbeCachedTileLRUFlagBuffer[tile], 1, 0, flag)
// (gi1.comp:906-907, 917-918): claims a cached tile for this probe and returns the flag it held
// (1: it was free this frame). WGSL has only the weak exchange, so a spurious failure is
// retried while the flag still reads 1 (as register D22).
fn rcClaimCachedTile(tile: u32) -> u32 {
    loop {
        let result = atomicCompareExchangeWeak(&rcProbeCachedTileLRUFlagBuffer[tile], 1u, 0u);
        if (result.exchanged || result.old_value != 1u) {
            return result.old_value;
        }
    }
}

// The cell's radiance at a reprojected probe seen from worldPos (gi1.comp:740-772): the point the
// cell's centre ray hit, as a direction and a length from worldPos. A zero length (the cell's hit
// distance 0, seen from the probe's own point) is skipped, where Capsaicin's normalize gives NaN
// (register D51).
struct RcReprojectedSample {
    direction: vec3f,
    length: f32,
}

fn rcReprojectSample(probePosition: vec3f, probeNormal: vec3f, cell: vec2u, hitDistance: f32, worldPos: vec3f) -> RcReprojectedSample {
    let hitPoint = probePosition + rcCellDirection(cell, probeNormal) * hitDistance;
    let toHit = hitPoint - worldPos;
    let reprojectedLength = length(toHit);
    return RcReprojectedSample(toHit / max(reprojectedLength, 1e-30), reprojectedLength);
}

// ScreenProbes_ScanRadiance (screen_probes.hlsl:309-351): an exclusive scan of the 64 cells'
// weights, normalised by their total, so the CDF is cell i's start in [0, 1). Every thread of the
// workgroup calls it (barriers).
fn rcScanRadiance(localId: u32, value: f32) {
    rcRadianceCdf[localId] = value;
    workgroupBarrier();

    let blockSize = RC_PROBE_CELLS;
    let firstLane = (localId / blockSize) * blockSize;
    let localLane = localId - firstLane;

    var total = rcRadianceCdf[firstLane + blockSize - 1u];
    workgroupBarrier();

    for (var stride = 1u; stride <= (blockSize >> 1u); stride <<= 1u) {
        if (localLane < blockSize / (2u * stride)) {
            rcRadianceCdf[firstLane + 2u * (localLane + 1u) * stride - 1u] += rcRadianceCdf[firstLane + (2u * localLane + 1u) * stride - 1u];
        }
        workgroupBarrier();
    }

    if (localLane == 0u) {
        rcRadianceCdf[firstLane + blockSize - 1u] = 0.0;
    }
    workgroupBarrier();

    for (var stride = blockSize >> 1u; stride > 0u; stride >>= 1u) {
        if (localLane < blockSize / (2u * stride)) {
            let carried = rcRadianceCdf[firstLane + (2u * localLane + 1u) * stride - 1u];
            rcRadianceCdf[firstLane + (2u * localLane + 1u) * stride - 1u] = rcRadianceCdf[firstLane + 2u * (localLane + 1u) * stride - 1u];
            rcRadianceCdf[firstLane + 2u * (localLane + 1u) * stride - 1u] += carried;
        }
        workgroupBarrier();
    }

    total += rcRadianceCdf[firstLane + blockSize - 1u];
    workgroupBarrier();

    rcRadianceCdf[localId] /= max(total, 1e-5);
    workgroupBarrier();
}

// ScreenProbes_FindCellIndex (screen_probes.hlsl:283-306): the last cell whose CDF start is at
// most s (a binary search). Cells of weight 0 have an empty interval and are never picked, unless
// every weight is 0: then it is the last cell.
fn rcFindCellIndex(localId: u32, s: f32) -> u32 {
    var index = 0u;
    var count = RC_PROBE_CELLS;
    let start = (localId / RC_PROBE_CELLS) * RC_PROBE_CELLS;
    while (0u < count) {
        let count2 = count >> 1u;
        let mid = index + count2;
        if (rcRadianceCdf[mid + start] > s) {
            count = count2;
        } else {
            index = mid + 1u;
            count -= count2 + 1u;
        }
    }
    return max(index, 1u) - 1u;
}

// SampleScreenProbes (gi1.comp:646-983). Each workgroup is one probe, each thread one cell.
//
// 1. The 3 x 3 reprojected probes around the probe's tile bring their cells' radiance into this
//    probe's cells by hit-point reprojection, behind a plane test and a hemisphere test
//    (gi1.comp:687-781). If the tile's own reprojected probe gives nothing, it is the probe this
//    one replaces: it is evicted into the off-screen cache.
// 2. The cached probes listed in the 3 x 3 tiles (CountScreenProbes, ScatterScreenProbes) bring
//    theirs too; the nearest cached probe on this probe's plane is the one to update, and the
//    nearest on the evicted probe's plane the slot to evict it into (gi1.comp:784-934).
// 3. Each cell's history is the mean of what reached it; the luminance of the history, times the
//    cell's cosine, is the CDF the rays are guided by (gi1.comp:937-948). A cell with history
//    sends its ray to a cell drawn from the CDF; a cell without sends it into itself
//    (gi1.comp:968-978).
//
// The specular branch (GGX by the surface's BRDF probability, gi1.comp:952-965) is not taken:
// the platform's materials are diffuse-only (registers D8, D16).
@compute @workgroup_size(64)
fn ${ENTRY_POINT.sample}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localId: u32,
) {
    let did = rcWorkgroupIndex(workgroup, workgroups) * 64u + localId;
    let probeCount = rcProbeSpawn.count;
    let cellAndProbeIndex = rcGetCellAndProbeIndex(did);
    let cellIndex = cellAndProbeIndex.x;
    let probeIndex = cellAndProbeIndex.y;
    // The ray count for the passes after this one (and the world cache), on the GPU.
    if (did == 0u) {
        rcProbeRays.count = probeCount * RC_PROBE_CELLS;
    }
    let isProbe = probeIndex < probeCount;

    let cell = vec2u(cellIndex % RC_PROBE_SIZE, cellIndex / RC_PROBE_SIZE);
    var seed = vec2u(0xffffffffu);
    if (isProbe) {
        seed = rcUnpackSeed(rcProbeSpawn.seeds[probeIndex]);
    }
    let probe = seed / RC_PROBE_SIZE;
    let pos = probe * RC_PROBE_SIZE + cell;

    // gi1.comp:662-674. Threads past the last probe read nothing (Capsaicin reads at seed
    // (-1, -1), which D3D answers with 0, and uses none of it).
    var normal = vec3f(0.0, 0.0, 1.0);
    var worldPos = rcFrame.eye;
    if (isProbe) {
        normal = rcLoadNormal(seed);
        worldPos = rcProbeWorldPosition(seed);
    }
    var previousProbeMask = RC_INVALID_ID;
    let cellSize = distance(rcFrame.eye, worldPos) * rcFrame.cellSize;
    let probeRes = vec2i(rcFrame.probeCount);
    // g_ScreenProbes_ProbeCachedTileIndexBuffer's size: one entry per probe tile.
    let dims = rcFrame.probeCount;

    if (localId == 0u) {
        atomicStore(&rcRadianceReuseSampleCount, 0u);
    }
    rcClearRadianceValues(localId);
    workgroupBarrier();

    if (isProbe) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let tap = vec2i(probe) + vec2i(x, y);
                if (any(tap < vec2i(0)) || any(tap >= probeRes)) {
                    continue; // out of bounds
                }
                let probeMask = textureLoad(rcProbeMaskBuffer, vec2u(tap), 0).x;
                if (probeMask == RC_INVALID_ID) {
                    continue; // invalid probe
                }
                if (x == 0 && y == 0) {
                    previousProbeMask = probeMask;
                }
                let probeSeed = rcUnpackSeed(probeMask);
                let probePos = rcProbeWorldPosition(probeSeed);
                if (abs(dot(probePos - worldPos, normal)) > cellSize) {
                    continue; // prevent leaks from faraway probes
                }
                let probeNormal = rcLoadNormal(probeSeed);
                let probeRadiance = textureLoad(rcProbeBuffer, (probeSeed / RC_PROBE_SIZE) * RC_PROBE_SIZE + cell, 0);
                let reprojected = rcReprojectSample(probePos, probeNormal, cell, probeRadiance.w, worldPos);
                if (dot(normal, reprojected.direction) < 0.0 || reprojected.length == 0.0) {
                    continue; // oriented hemispheres do not overlap
                }
                if (x == 0 && y == 0) {
                    atomicAdd(&rcRadianceReuseSampleCount, 1u);
                }
                rcAccumulateRadiance(rcRemapCellIndex(reprojected.direction, normal), vec4f(probeRadiance.xyz, reprojected.length));
            }
        }
    }
    workgroupBarrier();

    // The tile's own reprojected probe gave this probe something: it is not replaced.
    if (atomicLoad(&rcRadianceReuseSampleCount) > 0u) {
        previousProbeMask = RC_INVALID_ID;
    }

    if (isProbe) {
        var previousProbePos = vec3f(0.0);
        var previousProbeNormal = vec3f(0.0, 0.0, 1.0);
        if (previousProbeMask != RC_INVALID_ID) {
            let previousSeed = rcUnpackSeed(previousProbeMask);
            previousProbePos = rcProbeWorldPosition(previousSeed);
            previousProbeNormal = rcLoadNormal(previousSeed);
        }

        var evictedProbeMask = RC_INVALID_ID;
        var updatedProbeMask = RC_INVALID_ID;
        var bestEvictedProbeScore = cellSize;
        var bestUpdatedProbeScore = cellSize;

        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let tap = vec2i(probe) + vec2i(x, y);
                if (any(tap < vec2i(0)) || any(tap >= probeRes)) {
                    continue; // out of bounds
                }
                let tapIndex = u32(tap.x + tap.y * probeRes.x);
                let cachedTileCount = rcProbeCachedTileListCountBuffer[tapIndex];
                let cachedTileIndex = rcProbeCachedTileListIndexBuffer[tapIndex];
                for (var i = 0u; i < cachedTileCount; i++) {
                    let cachedTile = rcProbeCacheWords[rcFrame.cachedTileListOffset + cachedTileIndex + i];
                    let cachedIndex = rcProbeCacheWords[rcFrame.cachedTileLRUOffset + cachedTile];
                    let cachedProbe = vec2u(cachedIndex % dims.x, cachedIndex / dims.x);
                    let cachedPos = cachedProbe * RC_PROBE_SIZE + cell;
                    let cachedEntry = textureLoad(rcProbeCachedTileIndexBuffer, cachedProbe, 0);
                    let cachedProbePos = rcCachedTilePosition(cachedEntry);
                    let cachedProbeNormal = rcCachedTileNormal(cachedEntry);

                    if (abs(dot(cachedProbePos - worldPos, normal)) < cellSize) {
                        let updatedProbeScore = distance(cachedProbePos, worldPos);
                        if (updatedProbeScore < bestUpdatedProbeScore && dot(normal, cachedProbeNormal) > RC_REUSE_NORMAL_COSINE) {
                            updatedProbeMask = cachedTile;
                            bestUpdatedProbeScore = updatedProbeScore;
                        }
                        let cachedProbeRadiance = textureLoad(rcProbeCachedTileBuffer, cachedPos, 0);
                        let reprojected = rcReprojectSample(cachedProbePos, cachedProbeNormal, cell, cachedProbeRadiance.w, worldPos);
                        if (dot(normal, reprojected.direction) > 0.0 && reprojected.length > 0.0) {
                            rcAccumulateRadiance(rcRemapCellIndex(reprojected.direction, normal), vec4f(cachedProbeRadiance.xyz, reprojected.length));
                        }
                    }

                    if (previousProbeMask != RC_INVALID_ID) {
                        if (abs(dot(cachedProbePos - previousProbePos, previousProbeNormal)) < cellSize) {
                            let evictedProbeScore = distance(cachedProbePos, previousProbePos);
                            if (evictedProbeScore < bestEvictedProbeScore && dot(previousProbeNormal, cachedProbeNormal) > RC_REUSE_NORMAL_COSINE) {
                                evictedProbeMask = cachedTile;
                                bestEvictedProbeScore = evictedProbeScore;
                            }
                        }
                    }
                }
            }
        }

        if (cellIndex == 0u) {
            if (previousProbeMask != RC_INVALID_ID) {
                if (evictedProbeMask == RC_INVALID_ID) {
                    // A new slot at the LRU's front: flag it as newly allocated.
                    evictedProbeMask = atomicAdd(&rcCounters[RC_COUNTER_CACHED_TILE_LRU], 1u) | RC_CACHED_TILE_UNALLOCATED;
                } else if (rcClaimCachedTile(evictedProbeMask) != 1u) {
                    evictedProbeMask = RC_INVALID_ID;
                }
            }
            if (updatedProbeMask != RC_INVALID_ID) {
                if (rcClaimCachedTile(updatedProbeMask) != 1u) {
                    updatedProbeMask = RC_INVALID_ID;
                }
            }
            rcProbeCacheWords[rcFrame.spawnProbeOffset + 2u * probeIndex + 0u] = evictedProbeMask;
            rcProbeCacheWords[rcFrame.spawnProbeOffset + 2u * probeIndex + 1u] = updatedProbeMask;
        }
    }
    workgroupBarrier();

    let totalWeight = f32(atomicLoad(&rcRadianceSampleCounts[localId]));
    var previousRadiance = rcLoadRadianceValues(localId);
    if (totalWeight > 0.0) {
        previousRadiance /= totalWeight;
    } else {
        previousRadiance.w = -1.0;
    }

    // Build our sampling CDF: the history's luminance times the cell's cosine (gi1.comp:944-947).
    let cellDirection = rcMapToHemiOctahedron((vec2f(cell) + 0.5) / f32(RC_PROBE_SIZE));
    rcScanRadiance(localId, rcLuminance(previousRadiance.xyz) * cellDirection.z);

    if (!isProbe) {
        return; // out of bounds
    }

    // The draws in the order of Capsaicin's DISABLE_SPECULAR_MATERIALS build (register D16; the
    // default build draws one more rand(), the BRDF choice at gi1.comp:960): rand2() for the GGX
    // sample (gi1.comp:950), the CDF draw of ray guiding (gi1.comp:971), then the point inside the
    // cell (gi1.comp:973).
    var random = rcMakeRandom(did, rcFrame.frameIndex);
    _ = rcRand2(&random);
    let sampledCellIndex = rcFindCellIndex(localId, rcRand(&random));
    var sampledCell = cell;
    if (totalWeight > 0.0) {
        sampledCell = vec2u(sampledCellIndex % RC_PROBE_SIZE, sampledCellIndex / RC_PROBE_SIZE);
    }
    let direction = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(sampledCell) + rcRand2(&random)) / f32(RC_PROBE_SIZE)), normal);

    textureStore(rcPreviousProbeBuffer, pos, previousRadiance);
    textureStore(rcEvictedProbeBuffer, pos, textureLoad(rcProbeBuffer, pos, 0));
    rcProbeRays.rays[did].direction = rcPackSample(direction);
}
`;

// ---- PopulateScreenProbes -----------------------------------------------------------------------

export const POPULATE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/PopulateScreenProbes group 1',
    entries: [UNIFORM_ENTRY, texture(1), texture(2), storage(3, 'read-only-storage'), storage(4, 'storage'), storage(5, 'storage')],
};

/** `PopulateScreenProbesMain`, around the platform's side of the ray scratch (`HitShading.direct.traceWgsl('rcRayScratch', ...)`). */
export function populateWgsl(directScratchWgsl: string): string {
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenDepthWgsl(1)}
${screenNormalWgsl(2)}
${PROBE_SPAWN_STRUCT_WGSL}
${PROBE_RAYS_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
@group(1) @binding(3) var<storage, read> rcProbeSpawn: RcProbeSpawn;
// The probe rays: this kernel reads the directions and writes the origins, hits and flags.
@group(1) @binding(4) var<storage, read_write> rcProbeRays: RcProbeRays;
// g_ScreenProbes_ProbeSpawnRadianceBuffer, inside the platform's ray scratch: each ray's
// radiance and hit distance as four halves (packHalf4), which the world cache adds into.
@group(1) @binding(5) var<storage, read_write> rcRayScratch: GiRayScratch;

${directScratchWgsl}

// The pixel's position on the screen's depth, pushed off the surface for a ray's origin.
// Capsaicin starts its rays at the visibility buffer's exact point on the triangle,
// moved off it by offsetPosition (gi1.comp:1135-1150, geometry.hlsl:35-47). The host's
// depth only gives the point to the depth's precision, so the push also covers what four
// steps of the stored depth move the point (register D17): one step for the depth's own
// rounding, and margin for the reconstruction's float error and the depth's interpolation
// across the triangle, which can each be about a step (chosen, not measured).
const RC_DEPTH_STEP: f32 = 1.0 / 16777216.0;  // 2^-24: one step of a 24-bit depth, and of a 32-bit float just below 1

fn rcRayOrigin(seed: vec2u, normal: vec3f) -> vec3f {
    let uv = (vec2f(seed) + 0.5) / vec2f(rcFrame.bufferDimensions);
    let depth = rcLoadDepth(seed);
    let world = rcTransformPointProjection(uv, depth);
    // Stepped towards the near plane, which is always in front of the camera.
    let towardsNear = select(-1.0, 1.0, rcFrame.reversedDepth != 0u);
    let depthError = distance(world, rcTransformPointProjection(uv, depth + towardsNear * 4.0 * RC_DEPTH_STEP));
    return giShadeOffsetRayOrigin(world, normal) + normal * depthError;
}

// PopulateScreenProbesMain (gi1.comp:1159-1164) running PopulateScreenProbes
// (gi1.comp:1107-1157): one closest-hit ray per probe cell.
//
// A miss reads the sky (PopulateScreenProbesHandleMiss, gi1.comp:1078-1084). A hit starts
// at no radiance (payload.sky_sample stays 0, gi1.comp:1153) and is the world cache's:
// PopulateScreenProbesHandleHit's insert (gi1.comp:985-1076) and the passes after it run in
// the world cache's encodeFrame over the probe rays recorded here (world-cache/
// hash-grid-population-wgsl.ts). They add the hit's light into the ray's result: its emission
// for an emissive front face, its direct light for a ray shorter than its cell (the bypass),
// the cell's filtered radiance for the rest.
// The ray result's fourth half is the hit distance (gi1.comp:1156).
@compute @workgroup_size(32)
fn ${ENTRY_POINT.populate}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let did = rcWorkgroupIndex(workgroup, workgroups) * 32u + localIndex;
    let probeIndex = rcGetCellAndProbeIndex(did).y;

    if (probeIndex < rcProbeSpawn.count) {
        let seed = rcUnpackSeed(rcProbeSpawn.seeds[probeIndex]);
        // Sky pixels never reach the compacted list; Capsaicin checks again (gi1.comp:1127-1130).
        if (!rcIsSkyPixel(seed)) {
            let direction = rcUnpackSample(rcProbeRays.rays[did].direction);
            let normal = rcLoadNormal(seed);
            let ray = GiRay(rcRayOrigin(seed, normal), 0.0, direction, RC_MAX_HIT_DISTANCE);
            let hit = giTraceClosest(ray);
            rcProbeRays.rays[did].origin = ray.origin;
            rcProbeRays.rays[did].hitT = hit.t;
            rcProbeRays.rays[did].hitInstance = hit.instance;
            rcProbeRays.rays[did].hitTriangle = hit.triangle;
            rcProbeRays.rays[did].hitFrontFace = select(0u, 1u, hit.frontFace);
            rcProbeRays.rays[did].hitBarycentrics = hit.barycentrics;
            // No ray is lit here, so none carries the bypass flag: the world cache takes every
            // hit (and leaves the misses alone).
            rcProbeRays.rays[did].flags = 0u;

            var radiance = vec3f(0.0);
            var hitDistance = RC_LARGEST_HALF;
            if (hit.t < 0.0) {
                if (rcFrame.useDirectLighting != 0u) {
                    radiance = giSkyRadiance(direction);
                }
            } else {
                hitDistance = min(hit.t, RC_LARGEST_HALF);
            }
            giStoreRayResult(did, radiance, hitDistance);
        }
    }
}
`;
}

// ---- BlendScreenProbes --------------------------------------------------------------------------

export const BLEND_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/BlendScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        texture(1),
        texture(2),
        texture(3),
        texture(4),
        storage(5, 'read-only-storage'),
        storage(6, 'read-only-storage'),
        storage(7, 'read-only-storage'),
        storage(8, 'storage'),
        storage(9, 'storage'),
        storage(10, 'storage'),
        storageTexture(11, 'rgba16float'),
        storageTexture(12, 'r32uint', 'read-write'),
        storageTexture(13, 'rgba16float'),
        storageTexture(14, 'rgba32uint'),
    ],
};

export const BLEND_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${COUNTERS_WGSL}
${screenNormalWgsl(1)}
${screenDepthWgsl(2)}
${HISTORY_WGSL}
${PROBE_SPAWN_STRUCT_WGSL}
${PROBE_RAYS_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
${RADIANCE_ACCUMULATION_WGSL}
${RADIANCE_BACKUP_WGSL}
// g_ScreenProbes_PreviousProbeBuffer: each probe's history, as SampleScreenProbes wrote it.
@group(1) @binding(3) var rcPreviousProbeBuffer: texture_2d<f32>;
// The reprojected radiance of the probe this one replaces (register D46).
@group(1) @binding(4) var rcEvictedProbeBuffer: texture_2d<f32>;
@group(1) @binding(5) var<storage, read> rcProbeSpawn: RcProbeSpawn;
@group(1) @binding(6) var<storage, read> rcProbeRays: RcProbeRays;
// The ray scratch's results alone (RayScratch.rayResults): g_ScreenProbes_ProbeSpawnRadianceBuffer.
@group(1) @binding(7) var<storage, read> rcProbeSpawnRadiance: array<vec2u>;
@group(1) @binding(8) var<storage, read_write> rcCounters: array<atomic<u32>, RC_COUNTER_WORDS>;
// The probe cache's packed words (probe-cache.ts): the LRU, the MRU, ProbeSpawnProbeBuffer.
@group(1) @binding(9) var<storage, read_write> rcProbeCacheWords: array<atomic<u32>>;
@group(1) @binding(10) var<storage, read_write> rcProbeCachedTileLRUFlagBuffer: array<atomic<u32>>;
// g_ScreenProbes_ProbeBuffer: the probe atlas, 8 x 8 cells per probe tile; rgb radiance, a hit distance.
@group(1) @binding(11) var rcProbeBuffer: texture_storage_2d<rgba16float, write>;
// g_ScreenProbes_ProbeMaskBuffer, level 0: each probe tile's seed, or RC_INVALID_ID.
@group(1) @binding(12) var rcProbeMaskBuffer: texture_storage_2d<r32uint, read_write>;
// The off-screen probe cache (SampleScreenProbes reads it).
@group(1) @binding(13) var rcProbeCachedTileBuffer: texture_storage_2d<rgba16float, write>;
@group(1) @binding(14) var rcProbeCachedTileIndexBuffer: texture_storage_2d<rgba32uint, write>;

// Shadow-preserving biased temporal hysteresis (gi1.comp:1262-1268, after
// https://www.youtube.com/watch?v=WzpLWzGvFK4&t=630s): a cell that got much brighter than its
// history keeps most of the history (up to 0.95^2 of it); one that got darker, or brighter by at
// most its history again, takes the new value.
fn rcTemporalBlend(lumaA: f32, lumaB: f32) -> f32 {
    return rcSquared(clamp(max(lumaA - lumaB - min(lumaA, lumaB), 0.0) / max(max(lumaA, lumaB), 1e-4), 0.0, 0.95));
}

// The texel of a cached probe's cell: g_ScreenProbes_ProbeCachedTileLRUBuffer[slot] names a probe
// tile of the cache's atlas (gi1.comp:1287-1289).
fn rcCachedTilePos(slot: u32, cell: vec2u) -> vec2u {
    let dims = rcFrame.probeCount;
    let cachedIndex = atomicLoad(&rcProbeCacheWords[rcFrame.cachedTileLRUOffset + slot]) & ~RC_CACHED_TILE_UNALLOCATED;
    return vec2u(cachedIndex % dims.x, cachedIndex / dims.x) * RC_PROBE_SIZE + cell;
}

// BlendScreenProbes (gi1.comp:1166-1331): the probe's new rays into its cells, the backup for
// cells no ray reached, the hysteresis against the cell's history, then the probe cache: the
// replaced probe into its slot, this probe into the cached probe it updates.
@compute @workgroup_size(64)
fn ${ENTRY_POINT.blend}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localId: u32,
) {
    let did = rcWorkgroupIndex(workgroup, workgroups) * 64u + localId;
    let probeCount = rcProbeSpawn.count;
    let cellAndProbeIndex = rcGetCellAndProbeIndex(did);
    let cellIndex = cellAndProbeIndex.x;
    let probeIndex = cellAndProbeIndex.y;

    let cell = vec2u(cellIndex % RC_PROBE_SIZE, cellIndex / RC_PROBE_SIZE);
    var seed = vec2u(0xffffffffu);
    if (probeIndex < probeCount) {
        seed = rcUnpackSeed(rcProbeSpawn.seeds[probeIndex]);
    }
    let probe = seed / RC_PROBE_SIZE;
    let pos = probe * RC_PROBE_SIZE + cell;

    // Initialize our LDS... (gi1.comp:1183-1191)
    rcClearRadianceValues(localId);
    workgroupBarrier();

    // ... and accumulate into the sampled cells (gi1.comp:1193-1210). Each ray goes to the
    // cell its direction falls in.
    if (probeIndex < probeCount) {
        let normal = rcLoadNormal(seed);
        let direction = rcUnpackSample(rcProbeRays.rays[did].direction);
        let radiance = rcUnpackRadiance(rcProbeSpawnRadiance[did]);
        let sampledCell = vec2u(rcMapToHemiOctahedronInverse(rcToProbeFrame(direction, normal)) * f32(RC_PROBE_SIZE));
        rcAccumulateRadiance(sampledCell.x + sampledCell.y * RC_PROBE_SIZE, radiance);
    }
    workgroupBarrier();

    // The radiance backup value to be used for untraced cells (gi1.comp:1212-1234).
    let backup = rcRadianceBackupValue(localId);

    // We can now map back to our regular probe grid (gi1.comp:1236-1256).
    if (probeIndex >= probeCount) {
        return; // out of bounds
    }

    var radiance = rcLoadRadianceValues(localId);
    let sampleCount = atomicLoad(&rcRadianceSampleCounts[localId]);
    if (sampleCount > 0u) {
        radiance /= f32(sampleCount);
    } else {
        radiance = backup;
    }

    let previousRadiance = textureLoad(rcPreviousProbeBuffer, pos, 0);
    if (previousRadiance.w > 0.0) {
        let temporalBlend = rcTemporalBlend(rcLuminance(radiance.xyz), rcLuminance(previousRadiance.xyz));
        radiance = mix(radiance, previousRadiance, temporalBlend);
    }
    // Stored as halves (register D19): the hit distance kept below the half range.
    let stored = vec4f(radiance.xyz, min(radiance.w, RC_LARGEST_HALF));

    let lruCount = atomicLoad(&rcCounters[RC_COUNTER_CACHED_TILE_LRU]);
    var evictedProbeMask = atomicLoad(&rcProbeCacheWords[rcFrame.spawnProbeOffset + 2u * probeIndex + 0u]);
    let updatedProbeMask = atomicLoad(&rcProbeCacheWords[rcFrame.spawnProbeOffset + 2u * probeIndex + 1u]);

    // The replaced probe goes into its slot (gi1.comp:1275-1310). A slot below the LRU count was
    // taken by a new allocation this frame, which wins (the flag bit makes a new one pass).
    if (evictedProbeMask != RC_INVALID_ID && evictedProbeMask >= lruCount) {
        let isNewlyAllocated = (evictedProbeMask & RC_CACHED_TILE_UNALLOCATED) != 0u;
        evictedProbeMask &= ~RC_CACHED_TILE_UNALLOCATED; // unmask
        let cachedPos = rcCachedTilePos(evictedProbeMask, cell);

        if (cellIndex == 0u) {
            // Capsaicin's every thread clears the slot's flag bit and its LRU flag (gi1.comp:1281-1284);
            // here one thread does, with atomics, as the others read the slot (register D53).
            if (isNewlyAllocated) {
                atomicAnd(&rcProbeCacheWords[rcFrame.cachedTileLRUOffset + evictedProbeMask], ~RC_CACHED_TILE_UNALLOCATED);
                atomicStore(&rcProbeCachedTileLRUFlagBuffer[evictedProbeMask], 0u);
            }
            let previousProbeMask = textureLoad(rcProbeMaskBuffer, probe).x;
            let cachedTileMRUIndex = atomicAdd(&rcCounters[RC_COUNTER_CACHED_TILE_MRU], 1u);
            atomicStore(&rcProbeCacheWords[rcFrame.cachedTileMRUOffset + evictedProbeMask], cachedTileMRUIndex);

            let previousSeed = rcUnpackSeed(previousProbeMask);
            let cachedProbe = cachedPos / RC_PROBE_SIZE;
            textureStore(rcProbeCachedTileIndexBuffer, cachedProbe, rcCachedTileEntry(rcProbeWorldPosition(previousSeed), rcLoadNormal(previousSeed)));
        }
        textureStore(rcProbeCachedTileBuffer, cachedPos, textureLoad(rcEvictedProbeBuffer, pos, 0));
    }

    // This probe refreshes the cached probe it found (gi1.comp:1312-1323); the slot keeps its place
    // in the LRU order (its flag goes back to 1).
    if (updatedProbeMask != RC_INVALID_ID && updatedProbeMask >= lruCount) {
        let cachedPos = rcCachedTilePos(updatedProbeMask, cell);
        if (cellIndex == 0u) {
            atomicStore(&rcProbeCachedTileLRUFlagBuffer[updatedProbeMask], 1u);
        }
        textureStore(rcProbeCachedTileBuffer, cachedPos, stored);
    }

    // gi1.comp:1325-1330.
    if (cellIndex == 0u) {
        textureStore(rcProbeMaskBuffer, probe, vec4u(rcPackSeed(seed), 0u, 0u, 0u));
    }
    textureStore(rcProbeBuffer, pos, stored);
}
`;
