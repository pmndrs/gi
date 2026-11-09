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
 * The screen-probe kernels of phase 2 (RC2.1), one WGSL entry point per
 * Capsaicin kernel, named as Capsaicin names it, with the lines it ports.
 * Each kernel's own resources are in group 1 (the platform's are group 0);
 * the layouts are next to the code.
 *
 * What phase 2 leaves out, and where it comes back (docs/design/radiance-cache.md,
 * "Phase 2"): `ReprojectScreenProbes` and every history read (phase 4), so a
 * frame starts with the probe mask cleared, as reprojection leaves a tile with
 * no previous probe (gi1.comp:457); the override-tile list and
 * `PatchScreenProbes` (phase 4); ray guiding and the previous-probe taps in
 * `SampleScreenProbes` (phase 4); the world cache, so every probe-ray hit is
 * lit on Capsaicin's cache-bypass path (phase 3 sends the rest to the cache);
 * temporal blending and the off-screen probe cache in `BlendScreenProbes`
 * (phase 4).
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
export const FRAME_UNIFORM_BYTES = 112;

/**
 * Words of `RcCounters`: Capsaicin's append counters (`ClearCounters`,
 * gi1.comp:163-185), one word each, in its order. Phase 2 appends to none of
 * them; the lists that do arrive with phases 3 and 4, which then only add
 * their passes.
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

/**
 * The frame's constants: Capsaicin's `g_Eye`, `g_FrameIndex`,
 * `g_BufferDimensions`, `g_UseDirectLighting` and `GI1Constants.view_proj_inv`
 * (gi1.comp:33-41, 97), plus how the host's screen buffers are encoded.
 *
 * The inverse view-projection is camera-relative (`eye` is added after the
 * divide): the same position as Capsaicin's `transformPointProjection(uv,
 * depth, g_ViewProjectionInverse)`, without the f32 cancellation of a camera
 * far from the origin.
 */
const FRAME_WGSL = /* wgsl */ `
struct RcFrame {
    inverseViewProjection: mat4x4f,  // inverse(projection x view), with the view's translation taken out
    eye: vec3f,                      // g_Eye
    frameIndex: u32,                 // g_FrameIndex
    bufferDimensions: vec2u,         // g_BufferDimensions: the screen in pixels
    probeCount: vec2u,               // probe tiles (8 x 8 pixels) across and down
    maxProbeSpawnCount: u32,         // spawn tiles (16 x 16 pixels)
    normalUnorm: u32,                // 1: normals are 0..1 encoded (decoded as 2n - 1), 0: signed floats
    reversedDepth: u32,              // 1: depth 1 is near, so the sky is at 0
    useDirectLighting: u32,          // g_UseDirectLighting (gi1_use_direct_lighting)
}
@group(1) @binding(0) var<uniform> rcFrame: RcFrame;
`;

/** The host's depth (`FrameScreen.depth`) and the reads Capsaicin makes of its own depth buffer. */
function screenDepthWgsl(binding: number): string {
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
function screenNormalWgsl(binding: number): string {
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
const WORKGROUP_INDEX_WGSL = /* wgsl */ `
fn rcWorkgroupIndex(workgroup: vec3u, workgroups: vec3u) -> u32 {
    return workgroup.y * workgroups.x + workgroup.x;
}
`;

const UNIFORM_ENTRY: GPUBindGroupLayoutEntry = { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform', minBindingSize: FRAME_UNIFORM_BYTES } };
const SCREEN_TEXTURE: GPUTextureBindingLayout = { sampleType: 'unfilterable-float', viewDimension: '2d' };

function storage(binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, buffer: { type } };
}

function texture(binding: number): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, texture: SCREEN_TEXTURE };
}

function storageTexture(binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, storageTexture: { access: 'write-only', format, viewDimension: '2d' } };
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
    entries: [UNIFORM_ENTRY, storageTexture(1, 'r32uint')],
};

export const CLEAR_PROBE_MASK_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
@group(1) @binding(1) var rcProbeMask: texture_storage_2d<r32uint, write>;

// ClearProbeMask (gi1.comp:242-246). Capsaicin runs it when the mask is made; in phase 2
// it runs every frame, leaving each tile as ReprojectScreenProbes leaves a tile that
// found no previous probe (gi1.comp:457). Phase 4's reprojection takes its place.
@compute @workgroup_size(8, 8)
fn ${ENTRY_POINT.clearProbeMask}(@builtin(global_invocation_id) did: vec3u) {
    if (any(did.xy >= rcFrame.probeCount)) {
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
    entries: [storage(0, 'read-only-storage'), storage(1, 'read-only-storage'), storage(2, 'read-only-storage'), storage(3, 'storage')],
};

/** The compacted probe list: the count in word 0 (copied from the compaction), then one packed seed per probe. */
const PROBE_SPAWN_STRUCT_WGSL = /* wgsl */ `
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
/** `RcProbeRay.flags`: the ray is lit already, so the world cache adds nothing to it. */
export const PROBE_RAY_BYPASS = 1;

/**
 * The frame's probe rays, shaped for phase 3's world-cache interface
 * (docs/design/radiance-cache.md, "The world-cache interface"): per ray its
 * origin, direction, hit record and bypass flag, with the ray count on the
 * GPU. The radiance slot is the ray scratch's result for the same ray, which
 * gi/direct adds into and `BlendScreenProbes` reads.
 */
const PROBE_RAYS_STRUCT_WGSL = /* wgsl */ `
// RcProbeRay.flags: already lit (Capsaicin's bypass flag, gi1.comp:2561-2583); the world cache leaves it alone.
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
${PROBE_SPAWN_STRUCT_WGSL}
// The compaction's dispatch arguments, holding the kept count.
@group(1) @binding(0) var<storage, read> rcCompactArgs: GiIndirectArgs;
// The platform's stream compaction of the spawn flags: kept spawn tiles in increasing order.
@group(1) @binding(1) var<storage, read> rcKeptSpawnTiles: array<u32>;
@group(1) @binding(2) var<storage, read> rcPreviousProbeSpawn: array<u32>;
@group(1) @binding(3) var<storage, read_write> rcProbeSpawn: RcProbeSpawn;

// CompactScreenProbes (gi1.comp:581-613). Capsaicin runs a thread per spawn tile and
// places each kept one at its exclusive prefix (g_ScreenProbes_ProbeSpawnIndexBuffer,
// from gfxCommandScanSum); the platform's compaction has listed the kept tiles in the
// same order, so a thread per kept tile writes the same list.
//
// The override-tile list it also fills (gi1.comp:601-610) feeds PatchScreenProbes,
// which arrives with reprojection in phase 4. Without reprojection the probe mask
// is cleared at this point, so that list would be empty anyway.
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
    rcProbeSpawn.seeds[probeIndex] = rcPreviousProbeSpawn[did];
}
`;

// ---- SampleScreenProbes -------------------------------------------------------------------------

export const SAMPLE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/SampleScreenProbes group 1',
    entries: [UNIFORM_ENTRY, texture(1), storage(2, 'read-only-storage'), storage(3, 'storage')],
};

export const SAMPLE_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenNormalWgsl(1)}
${PROBE_SPAWN_STRUCT_WGSL}
${PROBE_RAYS_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
@group(1) @binding(2) var<storage, read> rcProbeSpawn: RcProbeSpawn;
// The probe rays: this kernel writes their count and directions.
@group(1) @binding(3) var<storage, read_write> rcProbeRays: RcProbeRays;

// SampleScreenProbes (gi1.comp:646-983), as Capsaicin runs it for a probe with no history.
//
// With no previous probe near (phase 2 has no reprojection), every 3 x 3 tap and every
// cached tile is skipped (gi1.comp:687-898), each cell's previous radiance is
// (0, 0, 0, -1) (gi1.comp:901-914), and the luminance CDF is zero, so the ray-guiding
// branch samples the thread's own cell: sampled_cell = cell when total_weight is 0
// (gi1.comp:972). Each ray is then a uniform point of its own cell
// (gi1.comp:973-977), so the 64 rays cover the hemisphere one per cell. Phase 4 adds the
// taps, the CDF scan and the PreviousProbeBuffer write (gi1.comp:981).
//
// The specular branch (GGX by the surface's BRDF probability, gi1.comp:952-965) is not
// taken: the platform's materials are diffuse-only (registers D8, D16).
@compute @workgroup_size(64)
fn ${ENTRY_POINT.sample}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localId: u32,
) {
    let did = rcWorkgroupIndex(workgroup, workgroups) * 64u + localId;
    let cellAndProbeIndex = rcGetCellAndProbeIndex(did);
    let cellIndex = cellAndProbeIndex.x;
    let probeIndex = cellAndProbeIndex.y;
    // The ray count for the passes after this one (and phase 3's world cache), on the GPU.
    if (did == 0u) {
        rcProbeRays.count = rcProbeSpawn.count * RC_PROBE_CELLS;
    }
    if (probeIndex >= rcProbeSpawn.count) {
        return; // out of bounds
    }

    let cell = vec2u(cellIndex % RC_PROBE_SIZE, cellIndex / RC_PROBE_SIZE);
    let seed = rcUnpackSeed(rcProbeSpawn.seeds[probeIndex]);
    let normal = rcLoadNormal(seed);

    // Capsaicin's draws, in its order: rand2() for the GGX sample (gi1.comp:950), the
    // CDF draw of ray guiding (gi1.comp:971), then the point inside the cell (gi1.comp:973).
    var random = rcMakeRandom(did, rcFrame.frameIndex);
    _ = rcRand2(&random);
    _ = rcRand(&random);
    let sampledCell = cell;
    let direction = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(sampledCell) + rcRand2(&random)) / f32(RC_PROBE_SIZE)), normal);

    rcProbeRays.rays[did].direction = rcPackSample(direction);
}
`;

// ---- PopulateScreenProbes -----------------------------------------------------------------------

export const POPULATE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/PopulateScreenProbes group 1',
    entries: [UNIFORM_ENTRY, texture(1), texture(2), storage(3, 'read-only-storage'), storage(4, 'storage'), storage(5, 'storage')],
};

/** The light groups whose lights reach the probes' hits: every group. */
export const ALL_LIGHT_GROUPS = 0xf;

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
// radiance and hit distance as four halves (packHalf4), which gi/direct adds the
// direct light into.
@group(1) @binding(5) var<storage, read_write> rcRayScratch: GiRayScratch;

${directScratchWgsl}

// The pixel's position on the screen's depth, pushed off the surface for a ray's origin.
// Capsaicin starts its rays at the visibility buffer's exact point on the triangle,
// moved off it by offsetPosition (gi1.comp:1135-1150, geometry.hlsl:35-47). The host's
// depth only gives the point to the depth's precision, so the push also covers what one
// step of the stored depth moves the point (register D17).
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
// Phase 2 has no world cache, so every hit is lit as Capsaicin lights a hit that bypasses
// it (the staging in docs/design/radiance-cache.md, "Phase 2"):
//  - a miss reads the sky (PopulateScreenProbesHandleMiss, gi1.comp:1078-1084);
//  - an emissive front face gives its emission and nothing more (GenerateReservoirs,
//    gi1.comp:2439-2452: "do not continue past an emissive surface");
//  - any other hit gives the light it reflects towards the probe: albedo / pi x the
//    direct irradiance from one light sample with its shadow ray (GenerateReservoirs,
//    gi1.comp:2500-2598, then PopulateCells, gi1.comp:2040-2106, which adds it straight
//    to the probe ray when the bypass flag is set). Here the platform's gi/direct does
//    that pass over the hits listed with giDeferDirect, and adds albedo / pi x direct
//    into the ray's result.
// The ray result's fourth half is the hit distance (gi1.comp:1156).
@compute @workgroup_size(32)
fn ${ENTRY_POINT.populate}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let did = rcWorkgroupIndex(workgroup, workgroups) * 32u + localIndex;
    let probeIndex = rcGetCellAndProbeIndex(did).y;

    // No early return: every thread reaches giDeferDirect, whose work list is filled
    // with subgroup operations.
    var deferred = false;
    var record = GiSurfaceRecord();
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
            // Phase 2 lights every ray here (the staging above), so every ray bypasses the
            // world cache. Phase 3 keeps the flag for emissive hits and rays shorter than a
            // cell (gi1.comp:2558-2561) and leaves the rest to the cache.
            rcProbeRays.rays[did].flags = RC_PROBE_RAY_BYPASS;

            var radiance = vec3f(0.0);
            var hitDistance: f32;
            if (hit.t < 0.0) {
                hitDistance = RC_LARGEST_HALF;
                if (rcFrame.useDirectLighting != 0u) {
                    radiance = giSkyRadiance(direction);
                }
            } else {
                hitDistance = min(hit.t, RC_LARGEST_HALF);
                let surface = giSurfaceAt(ray, hit);
                let emissive = dot(surface.emission, surface.emission) > 0.0;
                if (rcFrame.useDirectLighting != 0u && surface.frontFace && emissive) {
                    radiance = surface.emission;
                } else {
                    // gi/direct adds albedo / pi x direct (one light sample, its shadow ray).
                    deferred = giNeedsDirect(surface);
                    if (deferred) {
                        record = giSurfaceRecord(surface);
                    }
                }
            }
            giStoreRayResult(did, radiance, hitDistance);
        }
    }
    giDeferDirect(deferred, record, did);
}
`;
}

// ---- BlendScreenProbes --------------------------------------------------------------------------

export const BLEND_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/BlendScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        texture(1),
        storage(2, 'read-only-storage'),
        storage(3, 'read-only-storage'),
        storage(4, 'read-only-storage'),
        storageTexture(5, 'rgba16float'),
        storageTexture(6, 'r32uint'),
    ],
};

export const BLEND_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenNormalWgsl(1)}
${PROBE_SPAWN_STRUCT_WGSL}
${PROBE_RAYS_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
@group(1) @binding(2) var<storage, read> rcProbeSpawn: RcProbeSpawn;
@group(1) @binding(3) var<storage, read> rcProbeRays: RcProbeRays;
// The ray scratch's results alone (RayScratch.rayResults): g_ScreenProbes_ProbeSpawnRadianceBuffer.
@group(1) @binding(4) var<storage, read> rcProbeSpawnRadiance: array<vec2u>;
// g_ScreenProbes_ProbeBuffer: the probe atlas, 8 x 8 cells per probe tile; rgb radiance, a hit distance.
@group(1) @binding(5) var rcProbeBuffer: texture_storage_2d<rgba16float, write>;
// g_ScreenProbes_ProbeMaskBuffer, level 0: each probe tile's seed, or RC_INVALID_ID.
@group(1) @binding(6) var rcProbeMaskBuffer: texture_storage_2d<r32uint, write>;

// screen_probes.hlsl:82-86: per-cell sums of quantized radiance and hit distance, sample
// counts, and the backup value for cells no ray reached.
var<workgroup> rcRadianceValues: array<atomic<u32>, 256>;
var<workgroup> rcRadianceSampleCounts: array<atomic<u32>, 64>;
var<workgroup> rcRadianceBackup: array<vec4f, 64>;

// BlendScreenProbes (gi1.comp:1166-1331), without temporal history: the biased hysteresis
// against the previous probe (gi1.comp:1258-1269) and the off-screen probe cache
// (gi1.comp:1271-1323) arrive in phase 4.
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
    atomicStore(&rcRadianceValues[(localId << 2u) + 0u], 0u);
    atomicStore(&rcRadianceValues[(localId << 2u) + 1u], 0u);
    atomicStore(&rcRadianceValues[(localId << 2u) + 2u], 0u);
    atomicStore(&rcRadianceValues[(localId << 2u) + 3u], 0u);
    atomicStore(&rcRadianceSampleCounts[localId], 0u);
    workgroupBarrier();

    // ... and accumulate into the sampled cells (gi1.comp:1193-1210). Each ray goes to the
    // cell its direction falls in, which is its own cell unless half precision moved it
    // across an edge.
    if (probeIndex < probeCount) {
        let normal = rcLoadNormal(seed);
        let direction = rcUnpackSample(rcProbeRays.rays[did].direction);
        let radiance = rcUnpackRadiance(rcProbeSpawnRadiance[did]);

        let quantizedRadiance = rcQuantizeRadiance(radiance);
        let sampledCell = vec2u(rcMapToHemiOctahedronInverse(rcToProbeFrame(direction, normal)) * f32(RC_PROBE_SIZE));
        let sampledCellIndex = sampledCell.x + sampledCell.y * RC_PROBE_SIZE;

        atomicAdd(&rcRadianceValues[(sampledCellIndex << 2u) + 0u], quantizedRadiance.x);
        atomicAdd(&rcRadianceValues[(sampledCellIndex << 2u) + 1u], quantizedRadiance.y);
        atomicAdd(&rcRadianceValues[(sampledCellIndex << 2u) + 2u], quantizedRadiance.z);
        atomicAdd(&rcRadianceValues[(sampledCellIndex << 2u) + 3u], quantizedRadiance.w);
        atomicAdd(&rcRadianceSampleCounts[sampledCellIndex], 1u);
    }
    workgroupBarrier();

    // Calculate the radiance backup value to be used for untraced cells (gi1.comp:1212-1234):
    // the mean of the traced cells, divided among the empty ones.
    let ownSampleCount = atomicLoad(&rcRadianceSampleCounts[localId]);
    rcRadianceBackup[localId] = vec4f(
        rcRecoverRadiance(vec4u(
            atomicLoad(&rcRadianceValues[(localId << 2u) + 0u]),
            atomicLoad(&rcRadianceValues[(localId << 2u) + 1u]),
            atomicLoad(&rcRadianceValues[(localId << 2u) + 2u]),
            0u,
        )).xyz,
        select(0.0, 1.0, ownSampleCount > 0u),
    );
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

    // We can now map back to our regular probe grid (gi1.comp:1236-1256).
    if (probeIndex >= probeCount) {
        return; // out of bounds
    }

    var radiance = rcRecoverRadiance(vec4u(
        atomicLoad(&rcRadianceValues[(localId << 2u) + 0u]),
        atomicLoad(&rcRadianceValues[(localId << 2u) + 1u]),
        atomicLoad(&rcRadianceValues[(localId << 2u) + 2u]),
        atomicLoad(&rcRadianceValues[(localId << 2u) + 3u]),
    ));
    if (ownSampleCount > 0u) {
        radiance /= f32(ownSampleCount);
    } else {
        radiance = rcRadianceBackup[0];
    }

    // gi1.comp:1325-1330. The hit distance is kept below the half range (register D19).
    if (cellIndex == 0u) {
        textureStore(rcProbeMaskBuffer, probe, vec4u(rcPackSeed(seed), 0u, 0u, 0u));
    }
    textureStore(rcProbeBuffer, pos, vec4f(radiance.xyz, min(radiance.w, RC_LARGEST_HALF)));
}
`;
