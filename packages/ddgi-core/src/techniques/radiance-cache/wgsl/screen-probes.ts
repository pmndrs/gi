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
 * The screen-probe kernels (RC2.1, with history since RC4.1), one WGSL entry
 * point per Capsaicin kernel, named as Capsaicin names it, with the lines it
 * ports. Each kernel's own resources are in group 1 (the platform's are group
 * 0); the layouts are next to the code.
 *
 * The kernels that carry probes between frames (`ReprojectScreenProbes`, the
 * off-screen cache's lookup and reorder, `PatchScreenProbes`) are in
 * temporal-probes.ts; `SampleScreenProbes` and `BlendScreenProbes` here read
 * and write that history (the taps, ray guiding, the hysteresis, the cache).
 * Since RC3.2 the probe rays' hits are lit by the world cache between
 * `PopulateScreenProbes` and `BlendScreenProbes`.
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
export const FRAME_UNIFORM_BYTES = 256;

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

/** Word index of each counter in `RcCounters` (`COUNTER_NAMES`' order). */
export const COUNTER = Object.fromEntries(COUNTER_NAMES.map((name, index) => [name, index])) as { readonly [Name in (typeof COUNTER_NAMES)[number]]: number };

/** The counters as WGSL constants, for kernels that bind `RcCounters` as atomics. */
export const COUNTERS_WGSL = /* wgsl */ `
// Capsaicin's append counters (ClearCounters, gi1.comp:163-185), one word each.
const RC_COUNTER_EMPTY_TILES: u32 = ${COUNTER.emptyTileCount}u;               // g_ScreenProbes_EmptyTileCountBuffer
const RC_COUNTER_OVERRIDE_TILES: u32 = ${COUNTER.overrideTileCount}u;         // g_ScreenProbes_OverrideTileCountBuffer
const RC_COUNTER_CACHED_TILE_LRU: u32 = ${COUNTER.cachedTileLRUCount}u;       // g_ScreenProbes_ProbeCachedTileLRUCountBuffer
const RC_COUNTER_CACHED_TILE_MRU: u32 = ${COUNTER.cachedTileMRUCount}u;       // g_ScreenProbes_ProbeCachedTileMRUCountBuffer
const RC_COUNTER_CACHED_TILE_LIST_ELEMENTS: u32 = ${COUNTER.cachedTileListElementCount}u; // g_ScreenProbes_ProbeCachedTileListElementCountBuffer
`;

/**
 * The frame's constants: Capsaicin's `g_Eye`, `g_FrameIndex`,
 * `g_BufferDimensions`, `g_UseDirectLighting`, `g_PreviousEye`,
 * `GI1Constants.view_proj`, `view_proj_inv` and `view_proj_inv_prev`
 * (gi1.comp:33-41, 95-98) and `ScreenProbesConstants.cell_size`
 * (gi1.cpp:1839-1843), plus how the host's screen buffers are encoded.
 *
 * The matrices are camera-relative (each camera's eye is added after the
 * divide, or taken off before the multiply): the same positions as
 * Capsaicin's `transformPointProjection`, without the f32 cancellation of a
 * camera far from the origin.
 */
export const FRAME_WGSL = /* wgsl */ `
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
    previousInverseViewProjection: mat4x4f, // g_PreviousViewProjectionInverse, relative to previousEye
    viewProjection: mat4x4f,         // g_ViewProjection, relative to eye: projection x view x translate(eye)
    previousEye: vec3f,              // g_PreviousEye
    cellSize: f32,                   // ScreenProbesConstants.cell_size, per unit distance from the eye
}
@group(1) @binding(0) var<uniform> rcFrame: RcFrame;
`;

/**
 * Byte offsets of `RcFrame`'s fields, by WGSL's layout rules (a `mat4x4f` is
 * 64 bytes; a `vec3f` aligns to 16 and the `u32` after it packs into its last
 * 4 bytes). `writeFrameUniforms` writes each field by name; a unit test checks
 * these against the struct above.
 */
export const FRAME_UNIFORM_OFFSETS = {
    inverseViewProjection: 0,
    eye: 64,
    frameIndex: 76,
    bufferDimensions: 80,
    probeCount: 88,
    maxProbeSpawnCount: 96,
    normalUnorm: 100,
    reversedDepth: 104,
    useDirectLighting: 108,
    previousInverseViewProjection: 112,
    viewProjection: 176,
    previousEye: 240,
    cellSize: 252,
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

/**
 * The previous frame's depth and normals (the platform's `GpuScreenHistory`:
 * Capsaicin's `g_PreviousDepthBuffer` and `g_PreviousNormalBuffer`), and the
 * previous camera's reconstruction.
 */
export function previousScreenWgsl(depthBinding: number, normalBinding: number): string {
    return /* wgsl */ `
@group(1) @binding(${depthBinding}) var rcPreviousDepth: texture_2d<f32>;
@group(1) @binding(${normalBinding}) var rcPreviousNormal: texture_2d<f32>;

fn rcLoadPreviousDepth(pixel: vec2u) -> f32 {
    return textureLoad(rcPreviousDepth, pixel, 0).x;
}

// normalize(2.0f * g_PreviousNormalBuffer.Load(...).xyz - 1.0f) (gi1.comp:341): the history
// holds the normals decoded already.
fn rcLoadPreviousNormal(pixel: vec2u) -> vec3f {
    return normalize(textureLoad(rcPreviousNormal, pixel, 0).xyz);
}

// transformPointProjection(uv, depth, g_PreviousViewProjectionInverse) (gi1.comp:342).
fn rcTransformPreviousPointProjection(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(2.0 * uv.x - 1.0, 2.0 * (1.0 - uv.y) - 1.0, depth, 1.0);
    let homogeneous = rcFrame.previousInverseViewProjection * ndc;
    return rcFrame.previousEye + homogeneous.xyz / homogeneous.w;
}
`;
}

/**
 * The host's motion (`FrameScreen.velocity`): NDC, current − previous, y up.
 * Capsaicin's velocity is in uv, current − previous (`uv - velocity`,
 * gi1.comp:330); uv's y runs down, so an NDC step of (x, y) is a uv step of
 * (x / 2, −y / 2).
 */
export function screenVelocityWgsl(binding: number): string {
    return /* wgsl */ `
@group(1) @binding(${binding}) var rcVelocity: texture_2d<f32>;

// previous_uv = uv - velocity (gi1.comp:329-330, 361-362).
fn rcPreviousUv(pixel: vec2u, uv: vec2f) -> vec2f {
    let ndcVelocity = textureLoad(rcVelocity, pixel, 0).xy;
    return uv - vec2f(0.5 * ndcVelocity.x, -0.5 * ndcVelocity.y);
}
`;
}

/**
 * The per-cell sums of quantised radiance and hit distance that Capsaicin
 * accumulates with `InterlockedAdd` in workgroup memory
 * (`lds_ScreenProbes_RadianceValues`, screen_probes.hlsl:82), kept as 64-bit
 * sums: a low word and a carry word per channel (register D40).
 *
 * Capsaicin's 32-bit sums wrap: a miss's hit distance is `FLT_MAX` packed as
 * a half (infinity), which `uint(round(1e4 × ∞))` turns into 0xffffffff, so
 * every miss subtracts 1 and the sum stays "far" by luck. Ours stores 65,504
 * (D19), 6.55e8 once quantised, which a 32-bit sum wraps at 7 rays in a cell;
 * ray guiding sends up to 64 rays to one cell, and `SampleScreenProbes`' taps
 * add up to 9 probes (plus cached ones) into a cell. With the carry the sum
 * is exact up to 2^64, far past anything a workgroup adds.
 */
export const WIDE_RADIANCE_SUMS_WGSL = /* wgsl */ `
// Per cell, per channel (radiance rgb, hit distance): the low word, then the carries.
var<workgroup> rcRadianceValues: array<atomic<u32>, 512>;
var<workgroup> rcRadianceSampleCounts: array<atomic<u32>, 64>;

fn rcClearRadianceSums(localId: u32) {
    for (var word = 0u; word < 8u; word++) {
        atomicStore(&rcRadianceValues[(localId << 3u) + word], 0u);
    }
    atomicStore(&rcRadianceSampleCounts[localId], 0u);
}

// One sample into a cell: InterlockedAdd of each quantised channel, plus the carry when the
// low word wraps (the old value plus the addend is below the old value exactly then).
fn rcAccumulateRadiance(cellIndex: u32, quantized: vec4u) {
    for (var channel = 0u; channel < 4u; channel++) {
        let word = (cellIndex << 3u) + (channel << 1u);
        let previous = atomicAdd(&rcRadianceValues[word], quantized[channel]);
        if (previous + quantized[channel] < previous) {
            atomicAdd(&rcRadianceValues[word + 1u], 1u);
        }
    }
    atomicAdd(&rcRadianceSampleCounts[cellIndex], 1u);
}

// ScreenProbes_RecoverRadiance (screen_probes.hlsl:237-246) of a cell's 64-bit sums.
fn rcRecoverRadianceSum(cellIndex: u32) -> vec4f {
    var sum = vec4f(0.0);
    for (var channel = 0u; channel < 4u; channel++) {
        let word = (cellIndex << 3u) + (channel << 1u);
        sum[channel] = f32(atomicLoad(&rcRadianceValues[word + 1u])) * 4294967296.0 + f32(atomicLoad(&rcRadianceValues[word]));
    }
    return sum / RC_FLOAT_QUANTIZE;
}

fn rcRadianceSampleCount(cellIndex: u32) -> u32 {
    return atomicLoad(&rcRadianceSampleCounts[cellIndex]);
}
`;

/**
 * Capsaicin's backup for cells no sample reached (`lds_ScreenProbes_RadianceBackup`,
 * gi1.comp:407-429 and 1212-1234): every thread puts in its own cell's
 * radiance and whether it was reached, a tree sum adds them up, and thread 0
 * leaves the result in element 0.
 *
 * Two fixes (register D41). Capsaicin puts in each cell's *summed* samples, so
 * a cell that guiding sent 30 rays to weighs 30 times, and then divides the
 * mean by the number of empty cells, so the more cells go unsampled the
 * darker each reads. We put in each cell's mean and keep the mean of the
 * reached cells.
 */
export const RADIANCE_BACKUP_WGSL = /* wgsl */ `
var<workgroup> rcRadianceBackup: array<vec4f, 64>;

// Every thread of the workgroup calls it (it has barriers); the result is in rcRadianceBackup[0].
fn rcComputeRadianceBackup(localId: u32) {
    let samples = rcRadianceSampleCount(localId);
    rcRadianceBackup[localId] = vec4f(rcRecoverRadianceSum(localId).xyz / f32(max(samples, 1u)), select(0.0, 1.0, samples > 0u));
    workgroupBarrier();
    for (var stride = 1u; stride < 64u; stride <<= 1u) {
        if (localId < 64u / (2u * stride)) {
            rcRadianceBackup[2u * (localId + 1u) * stride - 1u] += rcRadianceBackup[(2u * localId + 1u) * stride - 1u];
        }
        workgroupBarrier();
    }
    if (localId == 0u) {
        let totalRadiance = rcRadianceBackup[64u - 1u];
        // The reached cells' mean radiance. Capsaicin divides it by the number of empty cells
        // too (radiance / max(empty_cell_count, 1.0f)), so the more cells go unsampled the darker
        // each of them reads: under a constant sky with history, about a third of the cells
        // (register D41).
        rcRadianceBackup[0] = vec4f(totalRadiance.xyz / max(totalRadiance.w, 1.0), RC_MAX_HIT_DISTANCE);
    }
    workgroupBarrier();
}
`;

/** The workgroup a 2D-split dispatch is on, counted in rows (the platform's indirect arguments split the same way). */
export const WORKGROUP_INDEX_WGSL = /* wgsl */ `
fn rcWorkgroupIndex(workgroup: vec3u, workgroups: vec3u) -> u32 {
    return workgroup.y * workgroups.x + workgroup.x;
}
`;

export const UNIFORM_ENTRY: GPUBindGroupLayoutEntry = { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform', minBindingSize: FRAME_UNIFORM_BYTES } };
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

/** An `r32uint` texture read with `textureLoad` (the probe mask). */
function maskTexture(binding: number): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } };
}

/** Layout-entry helpers for the temporal kernels (wgsl/temporal-probes.ts). */
export const LAYOUT = { storage, texture, storageTexture, maskTexture } as const;

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

// ClearProbeMask (gi1.comp:242-246). Capsaicin runs it when the mask is made; we run it on
// last frame's mask when there is no history (the first frame, a resize, resetScreenHistory),
// so ReprojectScreenProbes finds no previous probe anywhere.
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
    entries: [
        storage(0, 'read-only-storage'),
        storage(1, 'read-only-storage'),
        storage(2, 'read-only-storage'),
        storage(3, 'storage'),
        maskTexture(4),
        storage(5, 'storage'),
        storage(6, 'storage'),
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
${PROBE_SPAWN_STRUCT_WGSL}
// The compaction's dispatch arguments, holding the kept count.
@group(1) @binding(0) var<storage, read> rcCompactArgs: GiIndirectArgs;
// The platform's stream compaction of the spawn flags: kept spawn tiles in increasing order.
@group(1) @binding(1) var<storage, read> rcKeptSpawnTiles: array<u32>;
@group(1) @binding(2) var<storage, read> rcPreviousProbeSpawn: array<u32>;
@group(1) @binding(3) var<storage, read_write> rcProbeSpawn: RcProbeSpawn;
// g_ScreenProbes_ProbeMaskBuffer, level 0, as ReprojectScreenProbes left it.
@group(1) @binding(4) var rcProbeMask: texture_2d<u32>;
@group(1) @binding(5) var<storage, read_write> rcCounters: array<atomic<u32>, 16>;
// g_ScreenProbes_OverrideTileBuffer: probes that may give their rays to an empty tile.
@group(1) @binding(6) var<storage, read_write> rcOverrideTiles: array<u32>;

${COUNTERS_WGSL}

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
    let probeMask = textureLoad(rcProbeMask, rcUnpackSeed(probeSeed) / RC_PROBE_SIZE, 0).x;

    // If we're not filling a hole from the reprojection, we can append this tile to the
    // list of "overridable tile"; that is, this is a valid candidate to be un-spawned in
    // favor of a disoccluded "empty tile" during the patching of the probes (gi1.comp:601-610).
    if (probeMask != RC_INVALID_ID) {
        let overrideTileIndex = atomicAdd(&rcCounters[RC_COUNTER_OVERRIDE_TILES], 1u);
        rcOverrideTiles[overrideTileIndex] = probeIndex;
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
        storage(2, 'read-only-storage'),
        storage(3, 'storage'),
        texture(4),
        texture(5),
        maskTexture(6),
        storageTexture(7, 'rgba16float'),
        storage(8, 'read-only-storage'),
        storage(9, 'read-only-storage'),
        storage(10, 'read-only-storage'),
        storage(11, 'read-only-storage'),
        storage(12, 'read-only-storage'),
        texture(13),
        storage(14, 'storage'),
        storage(15, 'storage'),
        storage(16, 'storage'),
        storage(17, 'storage'),
    ],
};

/**
 * The share of a probe's rays that sample their own cell even when ray
 * guiding has a CDF (register D42): a floor under the luminance CDF, so a
 * cell that reads black is still refreshed now and then.
 */
export const RAY_GUIDING_UNIFORM_SHARE = 0.125;

/** `RcProbeSpawnProbe`: per spawned probe, the cached tiles it evicts into and refreshes, and the probe it replaces. */
export const PROBE_SPAWN_PROBE_BYTES = 16;

/** The off-screen probe cache's flag for an LRU entry that was never used (`InitCachedTileLRU`, gi1.comp:289) or is newly allocated (gi1.comp:871). */
export const CACHED_TILE_NEW = 0x80000000;

/** Shared by Sample and Blend: a cell of a probe frame from a world direction, as Capsaicin finds it. */
const PROBE_CELL_WGSL = /* wgsl */ `
// uint2(mapToHemiOctahedronInverse(mul(direction, CreateTBN(normal))) * probe_size)
// (gi1.comp:747-749, 1201-1202), as a cell index. A direction exactly on the square's far
// edge would give cell 8, past the probe; it is kept in the last cell (register D45).
fn rcProbeCellIndex(direction: vec3f, normal: vec3f) -> u32 {
    let cell = min(vec2u(rcMapToHemiOctahedronInverse(rcToProbeFrame(direction, normal)) * f32(RC_PROBE_SIZE)), vec2u(RC_PROBE_SIZE - 1u));
    return cell.x + cell.y * RC_PROBE_SIZE;
}

// packHalf4 (pack.hlsl): radiance and hit distance as four halves.
fn rcPackRadiance(radiance: vec4f) -> vec2u {
    return vec2u(pack2x16float(radiance.xy), pack2x16float(radiance.zw));
}

const RC_CACHED_TILE_NEW: u32 = ${CACHED_TILE_NEW}u;
// The normal agreement for reusing a cached probe (gi1.comp:344, 812, 852).
const RC_REUSE_NORMAL_THRESHOLD: f32 = 0.95;
`;

/** The probe this one replaces and the cached tiles it touches (`g_ScreenProbes_ProbeSpawnProbeBuffer`), and the replaced probe's radiance. */
const PROBE_SPAWN_PROBE_WGSL = /* wgsl */ `
// g_ScreenProbes_ProbeSpawnProbeBuffer (gi1.comp:896), per spawned probe: x the LRU entry its
// tile's old probe goes to (RC_CACHED_TILE_NEW set when newly allocated), y the cached tile it
// refreshes, each RC_INVALID_ID for none; z the old probe's seed, which BlendScreenProbes reads
// from the probe mask in Capsaicin (gi1.comp:1293) and which SampleScreenProbes knows already.
struct RcProbeSpawnProbe {
    evicted: u32,
    updated: u32,
    previousSeed: u32,
    reserved: u32,
}
`;

export const SAMPLE_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenNormalWgsl(1)}
${screenDepthWgsl(4)}
${PROBE_SPAWN_STRUCT_WGSL}
${PROBE_RAYS_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
${COUNTERS_WGSL}
${WIDE_RADIANCE_SUMS_WGSL}
${PROBE_CELL_WGSL}
${PROBE_SPAWN_PROBE_WGSL}
@group(1) @binding(2) var<storage, read> rcProbeSpawn: RcProbeSpawn;
// The probe rays: this kernel writes their count and directions.
@group(1) @binding(3) var<storage, read_write> rcProbeRays: RcProbeRays;
// g_ScreenProbes_ProbeBuffer: the atlas as ReprojectScreenProbes left it (last frame's probes, moved).
@group(1) @binding(5) var rcProbeBuffer: texture_2d<f32>;
// g_ScreenProbes_ProbeMaskBuffer, level 0: the reprojected probes' seeds.
@group(1) @binding(6) var rcProbeMask: texture_2d<u32>;
// g_ScreenProbes_PreviousProbeBuffer: free after the reprojection, so each probe's cells'
// previous radiance goes here for BlendScreenProbes (gi1.comp:981).
@group(1) @binding(7) var rcPreviousProbeBuffer: texture_storage_2d<rgba16float, write>;
// The off-screen probe cache: the cached tiles that land in each probe tile this frame
// (CountScreenProbes, ScatterScreenProbes), the LRU list, each cached tile's position and normal,
// its radiance, and the LRU entries' flags.
@group(1) @binding(8) var<storage, read> rcCachedTileListCount: array<u32>;
@group(1) @binding(9) var<storage, read> rcCachedTileListIndex: array<u32>;
@group(1) @binding(10) var<storage, read> rcCachedTileList: array<u32>;
@group(1) @binding(11) var<storage, read> rcCachedTileLRU: array<u32>;
@group(1) @binding(12) var<storage, read> rcCachedTileIndex: array<vec4u>;
@group(1) @binding(13) var rcCachedTileBuffer: texture_2d<f32>;
@group(1) @binding(14) var<storage, read_write> rcCachedTileLRUFlags: array<atomic<u32>>;
@group(1) @binding(15) var<storage, read_write> rcCounters: array<atomic<u32>, 16>;
@group(1) @binding(16) var<storage, read_write> rcProbeSpawnProbe: array<RcProbeSpawnProbe>;
// The radiance of the probe each spawned probe replaces, a cell per ray, for BlendScreenProbes
// to copy into the cache (register D43).
@group(1) @binding(17) var<storage, read_write> rcEvictedProbeRadiance: array<vec2u>;

const RC_RAY_GUIDING_UNIFORM_SHARE: f32 = ${RAY_GUIDING_UNIFORM_SHARE};

var<workgroup> rcRadianceReuseSampleCount: atomic<u32>;
// lds_ScreenProbes_Radiance: the cells' sampling CDF (exclusive sums over the total).
var<workgroup> rcCellCdf: array<f32, 64>;
var<workgroup> rcCellCdfTotal: f32;

// A cell of a nearby or cached probe, moved into this probe's frame (gi1.comp:725-756 and
// 819-843): the point its ray hit, seen from this probe, lands in one of this probe's cells,
// which takes the cell's radiance and the new hit distance. 'strict' is the cached probes'
// test (dot > 0); the screen taps take dot >= 0.
fn rcReuseCell(radiance: vec4f, probePosition: vec3f, probeNormal: vec3f, cell: vec2u, worldPosition: vec3f, normal: vec3f, strict: bool) -> bool {
    let probeDirection = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(cell) + 0.5) / f32(RC_PROBE_SIZE)), probeNormal);
    let hitPoint = probePosition + probeDirection * radiance.w;
    let reprojected = hitPoint - worldPosition;
    let reprojectedLength = length(reprojected);
    if (!(reprojectedLength > 0.0)) {
        return false; // the hit is the probe itself: no direction to take (D45)
    }
    let reprojectedDirection = reprojected / reprojectedLength; // normalize
    let facing = dot(normal, reprojectedDirection);
    if (facing < 0.0 || (strict && facing == 0.0)) {
        return false; // oriented hemispheres do not overlap
    }
    let quantized = rcQuantizeRadiance(vec4f(radiance.xyz, min(reprojectedLength, RC_LARGEST_HALF)));
    rcAccumulateRadiance(rcProbeCellIndex(reprojectedDirection, normal), quantized);
    return true;
}

// The probe's world position and normal from the screen at a seed.
fn rcProbePosition(seed: vec2u) -> vec3f {
    return rcTransformPointProjection((vec2f(seed) + 0.5) / vec2f(rcFrame.bufferDimensions), rcLoadDepth(seed));
}

// InterlockedCompareExchange(g_ScreenProbes_ProbeCachedTileLRUFlagBuffer[entry], 1, 0, flag)
// with flag == 1 (gi1.comp:876, 888): claims a cached tile for this probe alone. WGSL's
// exchange is weak, so a spurious failure (the flag still reads 1) is retried (as D22).
fn rcClaimCachedTile(entry: u32) -> bool {
    loop {
        let result = atomicCompareExchangeWeak(&rcCachedTileLRUFlags[entry], 1u, 0u);
        if (result.exchanged) {
            return true;
        }
        if (result.old_value != 1u) {
            return false;
        }
    }
}

// ScreenProbes_ScanRadiance (screen_probes.hlsl:282-338): every thread puts in its cell's weight;
// the workgroup leaves each cell's exclusive prefix sum over the total in rcCellCdf and the
// total in rcCellCdfTotal. Capsaicin divides by max(total, 1e-5); a total of 0 is kept as 0 here
// and the caller samples its own cell instead (register D42).
fn rcScanCellRadiance(localId: u32, weight: f32) {
    rcCellCdf[localId] = weight;
    workgroupBarrier();
    for (var offset = 1u; offset < 64u; offset <<= 1u) {
        var addend = 0.0;
        if (localId >= offset) {
            addend = rcCellCdf[localId - offset];
        }
        workgroupBarrier();
        rcCellCdf[localId] += addend;
        workgroupBarrier();
    }
    // Inclusive sums now; the exclusive one is the previous cell's inclusive sum.
    let total = rcCellCdf[63];
    let exclusive = select(0.0, rcCellCdf[max(localId, 1u) - 1u], localId > 0u);
    workgroupBarrier();
    rcCellCdf[localId] = select(0.0, exclusive / total, total > 0.0);
    if (localId == 0u) {
        rcCellCdfTotal = total;
    }
    workgroupBarrier();
}

// ScreenProbes_FindCellIndex (screen_probes.hlsl:257-280): the last cell whose CDF is at most s.
fn rcFindCellIndex(s: f32) -> u32 {
    var index = 0u;
    var count = RC_PROBE_CELLS;
    while (0u < count) {
        let count2 = count >> 1u;
        let mid = index + count2;
        if (rcCellCdf[mid] > s) {
            count = count2;
        } else {
            index = mid + 1u;
            count -= count2 + 1u;
        }
    }
    return max(index, 1u) - 1u;
}

// SampleScreenProbes (gi1.comp:646-983).
//
// Each cell's previous radiance: the probes of the 3 x 3 tiles around this one (as reprojected
// this frame) and the cached probes landing there, each cell moved into this probe's frame by
// its hit distance (gi1.comp:687-761, 782-861). The cells' luminance x cos theta is the CDF
// rays are guided by (gi1.comp:901-919, 969-978). The probe also picks the cached tile its
// tile's old probe is evicted into and the cached tile it refreshes (gi1.comp:767-897).
//
// Ours differs in the guiding (register D42): a share of the rays (RC_RAY_GUIDING_UNIFORM_SHARE)
// samples its own cell, as a probe without history does, so a black cell is still refreshed;
// and a probe whose cells all read black samples its own cells (Capsaicin's empty CDF sends
// every ray to the last cell).
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
    let probeCount = rcProbeSpawn.count;
    let cellAndProbeIndex = rcGetCellAndProbeIndex(did);
    let cellIndex = cellAndProbeIndex.x;
    let probeIndex = cellAndProbeIndex.y;
    // The ray count for the passes after this one (and the world cache), on the GPU.
    if (did == 0u) {
        rcProbeRays.count = probeCount * RC_PROBE_CELLS;
    }
    // One probe per workgroup, so this is the same for every thread of it.
    let isProbe = probeIndex < probeCount;

    let cell = vec2u(cellIndex % RC_PROBE_SIZE, cellIndex / RC_PROBE_SIZE);
    var seed = vec2u(0u);
    var normal = vec3f(0.0, 0.0, 1.0);
    var worldPosition = vec3f(0.0);
    if (isProbe) {
        seed = rcUnpackSeed(rcProbeSpawn.seeds[probeIndex]);
        normal = rcLoadNormal(seed);
        worldPosition = rcProbePosition(seed);
    }
    let probe = seed / RC_PROBE_SIZE;
    let pos = probe * RC_PROBE_SIZE + cell;
    let cellSize = distance(rcFrame.eye, worldPosition) * rcFrame.cellSize;
    let probeRes = vec2i(rcFrame.probeCount);

    if (localId == 0u) {
        atomicStore(&rcRadianceReuseSampleCount, 0u);
    }
    rcClearRadianceSums(localId);
    workgroupBarrier();

    // The 3 x 3 screen taps (gi1.comp:687-759).
    var previousProbeMask = RC_INVALID_ID;
    if (isProbe) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let tap = vec2i(probe) + vec2i(x, y);
                if (any(tap < vec2i(0)) || any(tap >= probeRes)) {
                    continue; // out of bounds
                }
                let probeMask = textureLoad(rcProbeMask, vec2u(tap), 0).x;
                if (probeMask == RC_INVALID_ID) {
                    continue; // invalid probe
                }
                let isCentre = x == 0 && y == 0;
                if (isCentre) {
                    previousProbeMask = probeMask;
                }
                let probeSeed = rcUnpackSeed(probeMask);
                let probePosition = rcProbePosition(probeSeed);
                if (abs(dot(probePosition - worldPosition, normal)) > cellSize) {
                    continue; // prevent leaks from faraway probes
                }
                let probeNormal = rcLoadNormal(probeSeed);
                let probeRadiance = textureLoad(rcProbeBuffer, (probeSeed / RC_PROBE_SIZE) * RC_PROBE_SIZE + cell, 0);
                if (rcReuseCell(probeRadiance, probePosition, probeNormal, cell, worldPosition, normal, false) && isCentre) {
                    atomicAdd(&rcRadianceReuseSampleCount, 1u);
                }
            }
        }
    }
    workgroupBarrier();

    // The tile's old probe is evicted only when this probe could not reuse it (gi1.comp:762-765).
    if (atomicLoad(&rcRadianceReuseSampleCount) > 0u) {
        previousProbeMask = RC_INVALID_ID;
    }

    // The cached probes landing in the 3 x 3 tiles (gi1.comp:767-897).
    if (isProbe) {
        var previousProbePosition = vec3f(0.0);
        var previousProbeNormal = vec3f(0.0, 0.0, 1.0);
        if (previousProbeMask != RC_INVALID_ID) {
            let previousProbeSeed = rcUnpackSeed(previousProbeMask);
            previousProbePosition = rcProbePosition(previousProbeSeed);
            previousProbeNormal = rcLoadNormal(previousProbeSeed);
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
                let cachedTileCount = rcCachedTileListCount[tapIndex];
                let cachedTileStart = rcCachedTileListIndex[tapIndex];
                for (var i = 0u; i < cachedTileCount; i++) {
                    let cachedTile = rcCachedTileList[cachedTileStart + i];
                    let cachedIndex = rcCachedTileLRU[cachedTile];
                    let cachedProbe = vec2u(cachedIndex % rcFrame.probeCount.x, cachedIndex / rcFrame.probeCount.x);
                    let cachedEntry = rcCachedTileIndex[cachedIndex];
                    let cachedProbePosition = bitcast<vec3f>(cachedEntry.xyz);
                    let cachedProbeNormal = normalize(rcUnpackNormal(cachedEntry.w));

                    if (abs(dot(cachedProbePosition - worldPosition, normal)) < cellSize) {
                        let updatedProbeScore = distance(cachedProbePosition, worldPosition);
                        if (updatedProbeScore < bestUpdatedProbeScore && dot(normal, cachedProbeNormal) > RC_REUSE_NORMAL_THRESHOLD) {
                            updatedProbeMask = cachedTile;
                            bestUpdatedProbeScore = updatedProbeScore;
                        }
                        let cachedProbeRadiance = textureLoad(rcCachedTileBuffer, cachedProbe * RC_PROBE_SIZE + cell, 0);
                        _ = rcReuseCell(cachedProbeRadiance, cachedProbePosition, cachedProbeNormal, cell, worldPosition, normal, true);
                    }

                    if (previousProbeMask != RC_INVALID_ID) {
                        if (abs(dot(cachedProbePosition - previousProbePosition, previousProbeNormal)) < cellSize) {
                            let evictedProbeScore = distance(cachedProbePosition, previousProbePosition);
                            if (evictedProbeScore < bestEvictedProbeScore && dot(previousProbeNormal, cachedProbeNormal) > RC_REUSE_NORMAL_THRESHOLD) {
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
                    // Allocate the least recently used entry (the LRU list is in that order).
                    evictedProbeMask = atomicAdd(&rcCounters[RC_COUNTER_CACHED_TILE_LRU], 1u) | RC_CACHED_TILE_NEW;
                } else if (!rcClaimCachedTile(evictedProbeMask)) {
                    evictedProbeMask = RC_INVALID_ID;
                }
            }
            if (updatedProbeMask != RC_INVALID_ID && !rcClaimCachedTile(updatedProbeMask)) {
                updatedProbeMask = RC_INVALID_ID;
            }
            rcProbeSpawnProbe[probeIndex] = RcProbeSpawnProbe(evictedProbeMask, updatedProbeMask, previousProbeMask, 0u);
        }
    }
    workgroupBarrier();

    // Each cell's previous radiance (gi1.comp:901-914), and the CDF (gi1.comp:916-919).
    let totalWeight = f32(rcRadianceSampleCount(localId));
    var previousRadiance = rcRecoverRadianceSum(localId);
    if (totalWeight > 0.0) {
        previousRadiance /= totalWeight;
    } else {
        previousRadiance.w = -1.0;
    }
    let cellDirection = rcMapToHemiOctahedron((vec2f(cell) + 0.5) / f32(RC_PROBE_SIZE));
    rcScanCellRadiance(localId, rcLuminance(previousRadiance.xyz) * cellDirection.z);

    if (!isProbe) {
        return; // out of bounds
    }

    // BlendScreenProbes copies the probe this one replaces into the cache (g_ScreenProbes_ProbeBuffer[pos],
    // gi1.comp:1308); it writes the atlas itself, so the cell is read here (register D43).
    rcEvictedProbeRadiance[did] = rcPackRadiance(textureLoad(rcProbeBuffer, pos, 0));

    // Capsaicin's draws, in its order: rand2() for the GGX sample (gi1.comp:950), the
    // CDF draw of ray guiding (gi1.comp:971), then the point inside the cell (gi1.comp:973).
    var random = rcMakeRandom(did, rcFrame.frameIndex);
    _ = rcRand2(&random);
    let guide = rcRand(&random);
    var sampledCell = cell;
    if (totalWeight > 0.0 && rcCellCdfTotal > 0.0 && guide >= RC_RAY_GUIDING_UNIFORM_SHARE) {
        let sampledCellIndex = rcFindCellIndex((guide - RC_RAY_GUIDING_UNIFORM_SHARE) / (1.0 - RC_RAY_GUIDING_UNIFORM_SHARE));
        sampledCell = vec2u(sampledCellIndex % RC_PROBE_SIZE, sampledCellIndex / RC_PROBE_SIZE);
    }
    let direction = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(sampledCell) + rcRand2(&random)) / f32(RC_PROBE_SIZE)), normal);

    textureStore(rcPreviousProbeBuffer, pos, previousRadiance);
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
        storage(2, 'read-only-storage'),
        storage(3, 'read-only-storage'),
        storage(4, 'read-only-storage'),
        storageTexture(5, 'rgba16float'),
        storageTexture(6, 'r32uint'),
        texture(7),
        texture(8),
        storage(9, 'read-only-storage'),
        storage(10, 'read-only-storage'),
        storage(11, 'storage'),
        storage(12, 'storage'),
        storage(13, 'storage'),
        storage(14, 'storage'),
        storage(15, 'storage'),
        storageTexture(16, 'rgba16float'),
    ],
};

/**
 * `BlendScreenProbes`' biased temporal hysteresis (gi1.comp:1262-1268): how
 * much of the previous radiance a cell keeps, from the new and the previous
 * luminance. It keeps nothing when the new value is darker or less than twice
 * as bright (shadows arrive at once), and up to 0.95² of the previous value
 * against a sudden brightening (fireflies). The CPU twin is
 * `temporalBlendFactor` (screen-probe-history.ts).
 */
export const TEMPORAL_BLEND_WGSL = /* wgsl */ `
fn rcTemporalBlend(lumaA: f32, lumaB: f32) -> f32 {
    let t = clamp(max(lumaA - lumaB - min(lumaA, lumaB), 0.0) / max(max(lumaA, lumaB), 1e-4), 0.0, 0.95);
    return t * t; // squared
}
`;

export const BLEND_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenNormalWgsl(1)}
${screenDepthWgsl(7)}
${PROBE_SPAWN_STRUCT_WGSL}
${PROBE_RAYS_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
${COUNTERS_WGSL}
${WIDE_RADIANCE_SUMS_WGSL}
${RADIANCE_BACKUP_WGSL}
${PROBE_CELL_WGSL}
${PROBE_SPAWN_PROBE_WGSL}
${TEMPORAL_BLEND_WGSL}
@group(1) @binding(2) var<storage, read> rcProbeSpawn: RcProbeSpawn;
@group(1) @binding(3) var<storage, read> rcProbeRays: RcProbeRays;
// The ray scratch's results alone (RayScratch.rayResults): g_ScreenProbes_ProbeSpawnRadianceBuffer.
@group(1) @binding(4) var<storage, read> rcProbeSpawnRadiance: array<vec2u>;
// g_ScreenProbes_ProbeBuffer: the probe atlas, 8 x 8 cells per probe tile; rgb radiance, a hit distance.
@group(1) @binding(5) var rcProbeBuffer: texture_storage_2d<rgba16float, write>;
// g_ScreenProbes_ProbeMaskBuffer, level 0: each probe tile's seed, or RC_INVALID_ID.
@group(1) @binding(6) var rcProbeMaskBuffer: texture_storage_2d<r32uint, write>;
// g_ScreenProbes_PreviousProbeBuffer: each cell's previous radiance, from SampleScreenProbes.
@group(1) @binding(8) var rcPreviousProbeBuffer: texture_2d<f32>;
@group(1) @binding(9) var<storage, read> rcProbeSpawnProbe: array<RcProbeSpawnProbe>;
@group(1) @binding(10) var<storage, read> rcEvictedProbeRadiance: array<vec2u>;
@group(1) @binding(11) var<storage, read_write> rcCounters: array<atomic<u32>, 16>;
// The off-screen probe cache: the LRU list and its flags and MRU order, each cached tile's
// position and normal, and its radiance.
@group(1) @binding(12) var<storage, read_write> rcCachedTileLRU: array<u32>;
@group(1) @binding(13) var<storage, read_write> rcCachedTileLRUFlags: array<atomic<u32>>;
@group(1) @binding(14) var<storage, read_write> rcCachedTileMRU: array<u32>;
@group(1) @binding(15) var<storage, read_write> rcCachedTileIndex: array<vec4u>;
@group(1) @binding(16) var rcCachedTileBuffer: texture_storage_2d<rgba16float, write>;

// BlendScreenProbes (gi1.comp:1166-1331).
//
// A cell no ray reached this frame keeps its previous radiance, and only a cell with no
// history either takes the backup (register D41). Capsaicin gives every unreached cell the
// backup, divided by the number of empty cells; with ray guiding about a third of a probe's
// cells go unreached each frame, and the hysteresis below takes a darker value at once, so
// a constant sky read 1.29 pi L at frame 1 falls to 0.87 pi L.
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
    let isProbe = probeIndex < probeCount;

    let cell = vec2u(cellIndex % RC_PROBE_SIZE, cellIndex / RC_PROBE_SIZE);
    var seed = vec2u(0xffffffffu);
    if (isProbe) {
        seed = rcUnpackSeed(rcProbeSpawn.seeds[probeIndex]);
    }
    let probe = seed / RC_PROBE_SIZE;
    let pos = probe * RC_PROBE_SIZE + cell;

    // Initialize our LDS... (gi1.comp:1183-1191)
    rcClearRadianceSums(localId);
    workgroupBarrier();

    // ... and accumulate into the sampled cells (gi1.comp:1193-1210). Each ray goes to the
    // cell its direction falls in.
    if (isProbe) {
        let normal = rcLoadNormal(seed);
        let direction = rcUnpackSample(rcProbeRays.rays[did].direction);
        let radiance = rcUnpackRadiance(rcProbeSpawnRadiance[did]);
        rcAccumulateRadiance(rcProbeCellIndex(direction, normal), rcQuantizeRadiance(radiance));
    }
    workgroupBarrier();

    // The backup for cells with neither a sample nor history (gi1.comp:1212-1234; D41).
    rcComputeRadianceBackup(localId);

    // We can now map back to our regular probe grid (gi1.comp:1236-1256).
    if (!isProbe) {
        return; // out of bounds
    }

    let sampleCount = rcRadianceSampleCount(localId);
    let previousRadiance = textureLoad(rcPreviousProbeBuffer, pos, 0);
    var radiance = rcRecoverRadianceSum(localId);
    if (sampleCount > 0u) {
        radiance /= f32(sampleCount);
    } else if (previousRadiance.w > 0.0) {
        radiance = previousRadiance; // no ray this frame: the cell keeps what it held (D41)
    } else {
        radiance = rcRadianceBackup[0];
    }

    // Shadow-preserving biased temporal hysteresis (gi1.comp:1258-1269).
    if (previousRadiance.w > 0.0) {
        radiance = mix(radiance, previousRadiance, rcTemporalBlend(rcLuminance(radiance.xyz), rcLuminance(previousRadiance.xyz)));
    }

    // The off-screen probe cache (gi1.comp:1271-1323).
    let spawnProbe = rcProbeSpawnProbe[probeIndex];
    let cachedTileLRUCount = atomicLoad(&rcCounters[RC_COUNTER_CACHED_TILE_LRU]);

    // The tile's old probe goes into the cache, unless its entry is being reused this frame.
    let evictedProbeMask = spawnProbe.evicted;
    if (evictedProbeMask != RC_INVALID_ID && evictedProbeMask >= cachedTileLRUCount) {
        let entry = evictedProbeMask & ~RC_CACHED_TILE_NEW;
        // Every thread takes the slot with the flag cleared; one clears it in the list.
        let cachedIndex = rcCachedTileLRU[entry] & ~RC_CACHED_TILE_NEW;
        if (cellIndex == 0u) {
            if ((evictedProbeMask & RC_CACHED_TILE_NEW) != 0u) {
                rcCachedTileLRU[entry] = cachedIndex; // unmask
                atomicStore(&rcCachedTileLRUFlags[entry], 0u);
            }
            rcCachedTileMRU[entry] = atomicAdd(&rcCounters[RC_COUNTER_CACHED_TILE_MRU], 1u);
            let previousSeed = rcUnpackSeed(spawnProbe.previousSeed);
            let previousUv = (vec2f(previousSeed) + 0.5) / vec2f(rcFrame.bufferDimensions);
            let previousWorld = rcTransformPointProjection(previousUv, rcLoadDepth(previousSeed));
            let previousNormal = rcLoadNormal(previousSeed);
            rcCachedTileIndex[cachedIndex] = vec4u(bitcast<vec3u>(previousWorld), rcPackNormal(previousNormal));
        }
        let cachedProbe = vec2u(cachedIndex % rcFrame.probeCount.x, cachedIndex / rcFrame.probeCount.x);
        textureStore(rcCachedTileBuffer, cachedProbe * RC_PROBE_SIZE + cell, rcUnpackRadiance(rcEvictedProbeRadiance[did]));
    }

    // The cached probe this one stands on gets its new radiance.
    let updatedProbeMask = spawnProbe.updated;
    if (updatedProbeMask != RC_INVALID_ID && updatedProbeMask >= cachedTileLRUCount) {
        let cachedIndex = rcCachedTileLRU[updatedProbeMask];
        if (cellIndex == 0u) {
            atomicStore(&rcCachedTileLRUFlags[updatedProbeMask], 1u);
        }
        let cachedProbe = vec2u(cachedIndex % rcFrame.probeCount.x, cachedIndex / rcFrame.probeCount.x);
        textureStore(rcCachedTileBuffer, cachedProbe * RC_PROBE_SIZE + cell, vec4f(radiance.xyz, min(radiance.w, RC_LARGEST_HALF)));
    }

    // gi1.comp:1325-1330. The hit distance is kept below the half range (register D19).
    if (cellIndex == 0u) {
        textureStore(rcProbeMaskBuffer, probe, vec4u(rcPackSeed(seed), 0u, 0u, 0u));
    }
    textureStore(rcProbeBuffer, pos, vec4f(radiance.xyz, min(radiance.w, RC_LARGEST_HALF)));
}
`;
