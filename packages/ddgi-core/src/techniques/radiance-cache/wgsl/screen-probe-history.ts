/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.comp
 * (InitCachedTileLRU, ReprojectScreenProbes, CountScreenProbes, ScatterScreenProbes,
 * PatchScreenProbes, ReorderScreenProbes).
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * Phase 4 (RC4.1): the screen probes' history, the kernels Capsaicin runs
 * around the spawn (`GI1::render`, gi1.cpp:2256-2318, 2544-2556):
 *
 * 1. `ReprojectScreenProbes` (gi1.comp:292-499), a thread per pixel and a
 *    workgroup per probe tile: each tile takes the previous frame's probe
 *    that best matches one of its pixels (plane and normal tests), its cells
 *    re-aimed from that pixel at the points their rays hit; a tile that finds
 *    none is a disocclusion, listed as an empty tile;
 * 2. `CountScreenProbes` (gi1.comp:501-539), the platform's scan and
 *    `ScatterScreenProbes` (gi1.comp:541-553): the off-screen probe cache's
 *    entries listed by the probe tile they project into;
 * 3. `PatchScreenProbes` (gi1.comp:615-644): each empty tile takes over a
 *    random spawned probe whose tile already had one (adaptive sampling);
 * 4. after `BlendScreenProbes`, the platform's scan of the LRU flags and
 *    `ReorderScreenProbes` (gi1.comp:1333-1354): the cache's slots reordered,
 *    the ones untouched this frame first (in their old order), then the ones
 *    used, most recent last, so new allocations take the least recently used.
 *
 * `InitCachedTileLRU` (gi1.comp:286-290) fills the LRU when the cache is
 * made. `CopyPreviousScreen` is ours: Capsaicin's framework copies the depth
 * and geometry normals it keeps as history (`PrevVisibilityDepth`,
 * `PrevGeometryNormal`, gi1.cpp:1368-1369, capsaicin_internal.cpp:879-889);
 * we copy the host's at the end of the screen stage (register D49).
 *
 * The probe cache's arrays live in one packed buffer (`probe-cache.ts`,
 * register D6) at the word offsets `RcFrame` carries.
 */

import { CAPSAICIN_WGSL } from './capsaicin.ts';
import {
    COUNTERS_WGSL,
    FRAME_WGSL,
    HISTORY_WGSL,
    RADIANCE_ACCUMULATION_WGSL,
    RADIANCE_BACKUP_WGSL,
    screenDepthWgsl,
    screenNormalWgsl,
    storage,
    storageTexture,
    texture,
    UNIFORM_ENTRY,
    uintTexture,
    PROBE_CACHE_WGSL,
} from './screen-probes.ts';
import { SPHERICAL_HARMONICS_WGSL } from './spherical-harmonics.ts';

/** Entry points: Capsaicin's kernel names (and ours for the screen copy). */
export const HISTORY_ENTRY_POINT = {
    initCachedTileLRU: 'InitCachedTileLRU',
    reproject: 'ReprojectScreenProbes',
    count: 'CountScreenProbes',
    scatter: 'ScatterScreenProbes',
    patch: 'PatchScreenProbes',
    reorder: 'ReorderScreenProbes',
    copyPreviousScreen: 'CopyPreviousScreen',
} as const;

/** Workgroup sizes, as Capsaicin's `[numthreads]`. */
export const HISTORY_WORKGROUP_SIZE = {
    /** ReprojectScreenProbes (gi1.comp:292): 8 × 8, one probe tile. Also the screen copy. */
    tile: 8,
    /** InitCachedTileLRU, CountScreenProbes, ScatterScreenProbes, PatchScreenProbes, ReorderScreenProbes: 64. */
    list: 64,
} as const;

/** The format the technique keeps the previous frame's depth in (`CopyPreviousScreen`). */
export const PREVIOUS_DEPTH_FORMAT: GPUTextureFormat = 'r32float';
/** The format of the previous frame's normals: decoded to signed floats. */
export const PREVIOUS_NORMAL_FORMAT: GPUTextureFormat = 'rgba16float';

/** The probe cache's packed words, plain `u32`s. */
const PROBE_CACHE_WORDS_WGSL = /* wgsl */ `
@group(1) @binding(BINDING) var<storage, read_write> rcProbeCacheWords: array<u32>;
`;

function probeCacheWords(binding: number): string {
    return PROBE_CACHE_WORDS_WGSL.replace('BINDING', String(binding));
}

/** The number of probe tiles: the probe cache's slots, as Capsaicin's `max_probe_count` (gi1.cpp:125). */
const PROBE_TILE_COUNT_WGSL = /* wgsl */ `
fn rcProbeTileCount() -> u32 {
    return rcFrame.probeCount.x * rcFrame.probeCount.y;
}
`;

// ---- InitCachedTileLRU --------------------------------------------------------------------------

export const INIT_CACHED_TILE_LRU_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/InitCachedTileLRU group 1',
    entries: [UNIFORM_ENTRY, storage(1, 'storage')],
};

export const INIT_CACHED_TILE_LRU_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${PROBE_CACHE_WGSL}
${PROBE_TILE_COUNT_WGSL}
${probeCacheWords(1)}

// InitCachedTileLRU (gi1.comp:286-290): every slot of this frame's LRU holds its own cached tile,
// flagged as holding no probe yet. Capsaicin's dispatch is whole groups over a buffer of
// max_probe_count, so its writes past the end are dropped; here the count bounds it.
@compute @workgroup_size(64)
fn ${HISTORY_ENTRY_POINT.initCachedTileLRU}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.x;
    if (did >= rcProbeTileCount()) {
        return;
    }
    rcProbeCacheWords[rcFrame.cachedTileLRUOffset + did] = did | RC_CACHED_TILE_UNALLOCATED;
}
`;

// ---- ReprojectScreenProbes ----------------------------------------------------------------------

export const REPROJECT_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ReprojectScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        texture(1),
        texture(2),
        texture(3),
        texture(4),
        texture(5),
        uintTexture(6),
        texture(7),
        uintTexture(8),
        storageTexture(9, 'rgba16float'),
        storageTexture(10, 'r32uint'),
        storageTexture(11, 'rg32uint'),
        storage(12, 'storage'),
        storage(13, 'storage'),
    ],
};

export const REPROJECT_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${COUNTERS_WGSL}
${SPHERICAL_HARMONICS_WGSL}
${screenDepthWgsl(1)}
${screenNormalWgsl(2)}
${HISTORY_WGSL}
${RADIANCE_ACCUMULATION_WGSL}
${RADIANCE_BACKUP_WGSL}
// g_VelocityBuffer: FrameScreen.velocity, the NDC motion (current - previous) in rg.
@group(1) @binding(3) var rcVelocity: texture_2d<f32>;
// g_PreviousDepthBuffer and g_PreviousNormalBuffer: last frame's host depth and normals, as
// CopyPreviousScreen kept them (normals already decoded to signed floats).
@group(1) @binding(4) var rcPreviousDepth: texture_2d<f32>;
@group(1) @binding(5) var rcPreviousNormal: texture_2d<f32>;
// g_ScreenProbes_PreviousProbeMaskBuffer (level 0), g_ScreenProbes_PreviousProbeBuffer and
// g_ScreenProbes_PreviousProbeSHBuffer: last frame's probes.
@group(1) @binding(6) var rcPreviousProbeMaskBuffer: texture_2d<u32>;
@group(1) @binding(7) var rcPreviousProbeBuffer: texture_2d<f32>;
@group(1) @binding(8) var rcPreviousProbeSHBuffer: texture_2d<u32>;
// This frame's probes, as reprojection leaves them for the spawn and the sampling.
@group(1) @binding(9) var rcProbeBuffer: texture_storage_2d<rgba16float, write>;
@group(1) @binding(10) var rcProbeMaskBuffer: texture_storage_2d<r32uint, write>;
@group(1) @binding(11) var rcProbeSHBuffer: texture_storage_2d<rg32uint, write>;
@group(1) @binding(12) var<storage, read_write> rcCounters: array<atomic<u32>, RC_COUNTER_WORDS>;
${probeCacheWords(13)}

// lds_ScreenProbes_Reprojection[0]: the best candidate's score, half(distance / cell size) << 16
// | the lane that found it. One segment: probe_size 8 is the workgroup's size.
var<workgroup> rcReprojection: atomic<u32>;

// (f32tof16(65504.0f) << 16) | 0xFFFF (gi1.comp:311): no candidate yet.
const RC_REPROJECTION_NONE: u32 = 0x7bffffffu;
const RC_NO_LANE: u32 = 0xffffu;
// The half that f32tof16 gives past the half range: infinity.
const RC_HALF_INFINITY_BITS: u32 = 0x7c00u;

// f32tof16 of a non-negative score (gi1.comp:343). pack2x16float's result past the half range is
// indeterminate in WGSL (register D52), so a score past it is infinity's bits, as f32tof16 gives.
fn rcScoreHalfBits(value: f32) -> u32 {
    if (!(value <= 65504.0)) {
        return RC_HALF_INFINITY_BITS;
    }
    return pack2x16float(vec2f(value, 0.0)) & 0xffffu;
}

// FrameScreen.velocity as Capsaicin's: a uv delta, y down (geometry.hlsl:104-110 writes
// (current_uv - previous_uv) * (1, -1) with uv = 0.5 * ndc). Ours is an NDC delta (register D50).
fn rcLoadVelocity(pixel: vec2u) -> vec2f {
    return textureLoad(rcVelocity, pixel, 0).xy * vec2f(0.5, -0.5);
}

// transformPointProjection(uv, depth, g_PreviousViewProjectionInverse) (gi1.comp:337), relative to
// the previous eye.
fn rcPreviousWorldPosition(seed: vec2u) -> vec3f {
    let uv = (vec2f(seed) + 0.5) / vec2f(rcFrame.bufferDimensions);
    let depth = textureLoad(rcPreviousDepth, seed, 0).x;
    let ndc = vec4f(2.0 * uv.x - 1.0, 2.0 * (1.0 - uv.y) - 1.0, depth, 1.0);
    let homogeneous = rcFrame.previousInverseViewProjection * ndc;
    return rcFrame.previousEye + homogeneous.xyz / homogeneous.w;
}

fn rcLoadPreviousNormal(seed: vec2u) -> vec3f {
    return normalize(textureLoad(rcPreviousNormal, seed, 0).xyz);
}

// The previous frame's pixel under this one, and whether it is on the screen
// (int2(previous_uv * g_BufferDimensions), gi1.comp:328-330).
struct RcPreviousPixel {
    pixel: vec2i,
    onScreen: bool,
}

fn rcPreviousPixel(pixel: vec2u) -> RcPreviousPixel {
    let uv = (vec2f(pixel) + 0.5) / vec2f(rcFrame.bufferDimensions);
    let previousUV = uv - rcLoadVelocity(pixel);
    // Far off screen (or not a number), it can't land on it: kept out of i32's range edge.
    let scaled = clamp(previousUV, vec2f(-1.0), vec2f(2.0)) * vec2f(rcFrame.bufferDimensions);
    let previousPos = vec2i(scaled);
    return RcPreviousPixel(previousPos, all(previousPos >= vec2i(0)) && all(previousPos < vec2i(rcFrame.bufferDimensions)));
}

// ReprojectScreenProbes (gi1.comp:292-499). A workgroup is one probe tile, a thread one of its
// pixels (finding the previous probe) and one of its cells (reprojecting it).
@compute @workgroup_size(8, 8)
fn ${HISTORY_ENTRY_POINT.reproject}(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(workgroup_id) groupId: vec3u,
    @builtin(local_invocation_id) localId: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let did = id.xy;
    // Capsaicin reads a zero normal past the screen's edge, so those threads are sky (gi1.comp:295-297).
    let isSkyPixel = any(did >= rcFrame.bufferDimensions) || rcIsSkyPixel(did);

    var normal = vec3f(0.0, 0.0, 1.0);
    var worldPos = rcFrame.eye;
    if (!isSkyPixel) {
        normal = rcLoadNormal(did);
        worldPos = rcProbeWorldPosition(did);
    }
    let cellSize = distance(rcFrame.eye, worldPos) * rcFrame.cellSize;

    let cell = localId.xy;
    let cellIndex = cell.x + cell.y * RC_PROBE_SIZE;

    if (cellIndex == 0u) {
        atomicStore(&rcReprojection, RC_REPROJECTION_NONE);
    }
    rcClearRadianceValues(localIndex);
    workgroupBarrier();

    // Try and find the closest matching probe in the previous frame. Each probe is assigned a
    // score and we atomically pick the best candidate out of all the 8x8, i.e., 64, potential
    // positions (gi1.comp:322-348).
    if (!isSkyPixel) {
        let previous = rcPreviousPixel(did);
        if (previous.onScreen) {
            let probeMask = textureLoad(rcPreviousProbeMaskBuffer, vec2u(previous.pixel) / RC_PROBE_SIZE, 0).x;
            if (probeMask != RC_INVALID_ID) {
                let probeSeed = rcUnpackSeed(probeMask);
                let probePos = rcPreviousWorldPosition(probeSeed);
                let probeNormal = rcLoadPreviousNormal(probeSeed);
                if (abs(dot(probePos - worldPos, normal)) < cellSize && dot(normal, probeNormal) > RC_REUSE_NORMAL_COSINE) {
                    let probeScore = (rcScoreHalfBits(distance(probePos, worldPos) / cellSize) << 16u) | localIndex;
                    atomicMin(&rcReprojection, probeScore);
                }
            }
        }
    }
    workgroupBarrier();

    // Now we map each lane to a cell in the probe and re-use the radiance values temporally
    // (gi1.comp:350-391).
    let localLane = atomicLoad(&rcReprojection) & 0xffffu;
    let seed = groupId.xy * RC_PROBE_SIZE + vec2u(localLane & 7u, localLane >> 3u);
    var probeMask = RC_INVALID_ID;
    if (localLane != RC_NO_LANE) {
        // The winning lane's pixel: on the screen, not sky, its previous pixel on the screen.
        probeMask = textureLoad(rcPreviousProbeMaskBuffer, vec2u(rcPreviousPixel(seed).pixel) / RC_PROBE_SIZE, 0).x;
        let seedNormal = rcLoadNormal(seed);
        let seedWorldPos = rcProbeWorldPosition(seed);

        let probeSeed = rcUnpackSeed(probeMask);
        let probePos = rcPreviousWorldPosition(probeSeed);
        let probeNormal = rcLoadPreviousNormal(probeSeed);
        let probeRadiance = textureLoad(rcPreviousProbeBuffer, (probeSeed / RC_PROBE_SIZE) * RC_PROBE_SIZE + cell, 0);

        let hitPoint = probePos + rcCellDirection(cell, probeNormal) * probeRadiance.w;
        let toHit = hitPoint - seedWorldPos;
        let reprojectedLength = length(toHit);
        let reprojectedDir = toHit / max(reprojectedLength, 1e-30);
        // A zero length (the probe's own point, a hit distance of 0) is skipped (register D51).
        if (dot(seedNormal, reprojectedDir) > 0.0 && reprojectedLength > 0.0) {
            rcAccumulateRadiance(rcRemapCellIndex(reprojectedDir, seedNormal), vec4f(probeRadiance.xyz, reprojectedLength));
        }
    }
    workgroupBarrier();

    // Calculate the radiance backup value to be used for unvisited cells (gi1.comp:393-416).
    let backup = rcRadianceBackupValue(localIndex);

    let probe = did / RC_PROBE_SIZE;

    // No probe found, this is a disocclusion (gi1.comp:418-450).
    if (localLane == RC_NO_LANE) {
        if (cellIndex == 0u) {
            if (!isSkyPixel) {
                let tileSize = f32(RC_PROBE_SPAWN_TILE_SIZE);
                let jitter = vec2u(min(rcCalculateHaltonSequence(rcFrame.frameIndex) * tileSize, vec2f(tileSize - 1.0)));
                // Check whether this disoccluded tile won't be filled during the probe spawning pass.
                // If it won't be spawned, we append it to the list of "empty tiles" so the ray
                // distribution can get re-balanced to achieve adaptive sampling.
                if (any((jitter / RC_PROBE_SIZE) != (probe % (RC_PROBE_SPAWN_TILE_SIZE / RC_PROBE_SIZE)))) {
                    let emptyTileIndex = atomicAdd(&rcCounters[RC_COUNTER_EMPTY_TILE], 1u);
                    rcProbeCacheWords[rcFrame.emptyTileOffset + emptyTileIndex] = probe.x + probe.y * rcFrame.probeCount.x;
                }
            }
            textureStore(rcProbeMaskBuffer, probe, vec4u(RC_INVALID_ID, 0u, 0u, 0u));
        }
        return; // reprojection failed :'(
    }

    // Reproject the spherical harmonics (gi1.comp:452-468): the previous probe's nine coefficients.
    if (cellIndex < RC_SH_COEFFICIENTS) {
        let previousProbe = rcUnpackSeed(probeMask) / RC_PROBE_SIZE;
        if (cellIndex == 0u) {
            textureStore(rcProbeMaskBuffer, probe, vec4u(rcPackSeed(seed), 0u, 0u, 0u));
        }
        textureStore(rcProbeSHBuffer, rcProbeSHTexel(probe, cellIndex), textureLoad(rcPreviousProbeSHBuffer, rcProbeSHTexel(previousProbe, cellIndex), 0));
    }

    // And reproject the radiance (gi1.comp:470-486).
    var radiance = rcLoadRadianceValues(localIndex);
    let sampleCount = atomicLoad(&rcRadianceSampleCounts[localIndex]);
    if (sampleCount > 0u) {
        radiance /= f32(sampleCount);
    } else {
        radiance = backup;
    }
    // Stored as halves (register D19): the hit distance kept below the half range, every value
    // rounded to its nearest half (register D59).
    textureStore(rcProbeBuffer, did, rcNearestHalf(vec4f(radiance.xyz, min(radiance.w, RC_LARGEST_HALF))));
}
`;

// ---- CountScreenProbes --------------------------------------------------------------------------

export const COUNT_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/CountScreenProbes group 1',
    entries: [UNIFORM_ENTRY, uintTexture(1), storage(2, 'storage'), storage(3, 'storage'), storage(4, 'storage'), storage(5, 'storage')],
};

export const COUNT_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${COUNTERS_WGSL}
${PROBE_CACHE_WGSL}
${PROBE_TILE_COUNT_WGSL}
@group(1) @binding(1) var rcProbeCachedTileIndexBuffer: texture_2d<u32>;
${probeCacheWords(2)}
@group(1) @binding(3) var<storage, read_write> rcCounters: array<atomic<u32>, RC_COUNTER_WORDS>;
@group(1) @binding(4) var<storage, read_write> rcProbeCachedTileListCountBuffer: array<atomic<u32>>;
@group(1) @binding(5) var<storage, read_write> rcProbeCachedTileLRUFlagBuffer: array<u32>;

// CountScreenProbes (gi1.comp:501-539): each cached probe that lies on the screen (in front of the
// camera, inside the depth range) is counted into the probe tile it projects into. Every slot's
// LRU flag is set: free to be claimed this frame.
@compute @workgroup_size(64)
fn ${HISTORY_ENTRY_POINT.count}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.x;
    let dims = rcFrame.probeCount;
    if (did >= rcProbeTileCount()) {
        return; // out of bounds
    }

    let cachedTileIndex = rcProbeCacheWords[rcFrame.cachedTileLRUOffset + did];
    if ((cachedTileIndex & RC_CACHED_TILE_UNALLOCATED) == 0u) {
        let cachedProbe = vec2u(cachedTileIndex % dims.x, cachedTileIndex / dims.x);
        let worldPos = rcCachedTilePosition(textureLoad(rcProbeCachedTileIndexBuffer, cachedProbe, 0));
        var homogeneous = rcTransformPointProjectionForward(worldPos);
        homogeneous = vec3f(0.5 * vec2f(homogeneous.x, -homogeneous.y) + 0.5, homogeneous.z);

        if (all(homogeneous > vec3f(0.0)) && all(homogeneous < vec3f(1.0))) {
            let probe = vec2u(homogeneous.xy * vec2f(dims));
            let probeIndex = probe.x + probe.y * dims.x;
            let elementIndex = atomicAdd(&rcCounters[RC_COUNTER_CACHED_TILE_LIST_ELEMENT], 1u);
            let indexInTile = atomicAdd(&rcProbeCachedTileListCountBuffer[probeIndex], 1u);
            let element = rcFrame.cachedTileListElementOffset + 4u * elementIndex;
            rcProbeCacheWords[element + 0u] = did;
            rcProbeCacheWords[element + 1u] = probeIndex;
            rcProbeCacheWords[element + 2u] = indexInTile;
            rcProbeCacheWords[element + 3u] = 0u;
        }
    }

    rcProbeCachedTileLRUFlagBuffer[did] = 1u;
}
`;

// ---- ScatterScreenProbes ------------------------------------------------------------------------

export const SCATTER_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ScatterScreenProbes group 1',
    entries: [UNIFORM_ENTRY, storage(1, 'read-only-storage'), storage(2, 'storage'), storage(3, 'read-only-storage')],
};

/** Built with the platform's `compute.indirectModule` (`GiIndirectArgs`, `giIndirectItemIndex`). */
export const SCATTER_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
// The dispatch's arguments, holding g_ScreenProbes_ProbeCachedTileListElementCountBuffer[0].
@group(1) @binding(1) var<storage, read> rcScatterArgs: GiIndirectArgs;
${probeCacheWords(2)}
// The platform's exclusive scan of g_ScreenProbes_ProbeCachedTileListCountBuffer.
@group(1) @binding(3) var<storage, read> rcProbeCachedTileListIndexBuffer: array<u32>;

// ScatterScreenProbes (gi1.comp:541-553): each listed cached probe into its tile's run of the list.
@compute @workgroup_size(64)
fn ${HISTORY_ENTRY_POINT.scatter}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let did = giIndirectItemIndex(workgroup, workgroups, localIndex, 64u);
    if (did >= rcScatterArgs.count) {
        return; // out of bounds
    }
    let element = rcFrame.cachedTileListElementOffset + 4u * did;
    let listElement = vec3u(rcProbeCacheWords[element + 0u], rcProbeCacheWords[element + 1u], rcProbeCacheWords[element + 2u]);
    let scatterIndex = rcProbeCachedTileListIndexBuffer[listElement.y] + listElement.z;
    rcProbeCacheWords[rcFrame.cachedTileListOffset + scatterIndex] = listElement.x;
}
`;

// ---- PatchScreenProbes --------------------------------------------------------------------------

export const PATCH_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/PatchScreenProbes group 1',
    entries: [UNIFORM_ENTRY, texture(1), storage(2, 'read-only-storage'), storage(3, 'read-only-storage'), storage(4, 'storage'), storage(5, 'storage')],
};

/** Built with the platform's `compute.indirectModule`. */
export const PATCH_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${COUNTERS_WGSL}
${screenDepthWgsl(1)}
// The dispatch's arguments, holding g_ScreenProbes_EmptyTileCountBuffer[0].
@group(1) @binding(2) var<storage, read> rcPatchArgs: GiIndirectArgs;
@group(1) @binding(3) var<storage, read> rcCounters: array<u32, RC_COUNTER_WORDS>;
${probeCacheWords(4)}
// g_ScreenProbes_ProbeSpawnBuffer, with its count in front (RcProbeSpawn), as atomics.
@group(1) @binding(5) var<storage, read_write> rcProbeSpawnWords: array<atomic<u32>>;

// PatchScreenProbes (gi1.comp:615-644): stochastically patch the overridable tiles using empty
// ones (a.k.a., adaptive sampling). Each empty tile that isn't sky at this frame's jitter takes
// over a random override tile's probe.
@compute @workgroup_size(64)
fn ${HISTORY_ENTRY_POINT.patch}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let did = giIndirectItemIndex(workgroup, workgroups, localIndex, 64u);
    let overrideTileCount = rcCounters[RC_COUNTER_OVERRIDE_TILE];
    if (overrideTileCount == 0u || did >= rcPatchArgs.count) {
        return; // out of bounds
    }

    let probeIndex = rcProbeCacheWords[rcFrame.emptyTileOffset + did];
    let probe = vec2u(probeIndex % rcFrame.probeCount.x, probeIndex / rcFrame.probeCount.x);
    let probeSize = f32(RC_PROBE_SIZE);
    let jitter = vec2u(min(rcCalculateHaltonSequence(rcFrame.frameIndex) * probeSize, vec2f(probeSize - 1.0)));
    let seed = min(probe * RC_PROBE_SIZE + jitter, rcFrame.bufferDimensions - 1u);
    if (rcIsSkyPixel(seed)) {
        return; // sky pixel
    }

    var random = rcMakeRandom(did, rcFrame.frameIndex);
    let index = rcRandInt(&random) % overrideTileCount;
    // The seeds start after the count (RcProbeSpawn).
    atomicExchange(&rcProbeSpawnWords[1u + rcProbeCacheWords[rcFrame.overrideTileOffset + index]], rcPackSeed(seed));
}
`;

// ---- ReorderScreenProbes ------------------------------------------------------------------------

export const REORDER_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ReorderScreenProbes group 1',
    entries: [UNIFORM_ENTRY, storage(1, 'storage'), storage(2, 'read-only-storage'), storage(3, 'read-only-storage')],
};

export const REORDER_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${PROBE_TILE_COUNT_WGSL}
${probeCacheWords(1)}
@group(1) @binding(2) var<storage, read> rcProbeCachedTileLRUFlagBuffer: array<u32>;
// The platform's exclusive scan of the flags: a slot's place among the untouched ones, and at
// index max_probe_count their number.
@group(1) @binding(3) var<storage, read> rcProbeCachedTileLRUIndexBuffer: array<u32>;

// ReorderScreenProbes (gi1.comp:1333-1354): re-order the cached probes so the least-recently used
// entries are evicted first. Slots untouched this frame (flag 1) keep their order at the front;
// the ones used (flag 0) follow in the order they were used (their MRU index).
@compute @workgroup_size(64)
fn ${HISTORY_ENTRY_POINT.reorder}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.x;
    let slotCount = rcProbeTileCount();
    if (did >= slotCount) {
        return; // out of bounds
    }
    let cachedTileIndex = rcProbeCacheWords[rcFrame.cachedTileLRUOffset + did];
    let cachedTileFlag = rcProbeCachedTileLRUFlagBuffer[did];
    var cachedTileOffset = rcProbeCachedTileLRUIndexBuffer[did];
    if (cachedTileFlag == 0u) {
        cachedTileOffset = rcProbeCacheWords[rcFrame.cachedTileMRUOffset + did] + rcProbeCachedTileLRUIndexBuffer[slotCount];
    }
    rcProbeCacheWords[rcFrame.previousCachedTileLRUOffset + cachedTileOffset] = cachedTileIndex;
}
`;

// ---- CopyPreviousScreen -------------------------------------------------------------------------

export const COPY_PREVIOUS_SCREEN_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/CopyPreviousScreen group 1',
    entries: [UNIFORM_ENTRY, texture(1), texture(2), storageTexture(3, PREVIOUS_DEPTH_FORMAT), storageTexture(4, PREVIOUS_NORMAL_FORMAT)],
};

export const COPY_PREVIOUS_SCREEN_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenDepthWgsl(1)}
${screenNormalWgsl(2)}
@group(1) @binding(3) var rcPreviousDepth: texture_storage_2d<r32float, write>;
@group(1) @binding(4) var rcPreviousNormal: texture_storage_2d<rgba16float, write>;

// Keeps this frame's depth and normals for the next frame's ReprojectScreenProbes (register D49),
// the normals decoded as ReprojectScreenProbes reads them (2n - 1 for a unorm encoding).
@compute @workgroup_size(8, 8)
fn ${HISTORY_ENTRY_POINT.copyPreviousScreen}(@builtin(global_invocation_id) id: vec3u) {
    let pixel = id.xy;
    if (any(pixel >= rcFrame.bufferDimensions)) {
        return;
    }
    textureStore(rcPreviousDepth, pixel, vec4f(rcLoadDepth(pixel), 0.0, 0.0, 0.0));
    textureStore(rcPreviousNormal, pixel, vec4f(rcLoadNormal(pixel), 0.0));
}
`;

/** Every history kernel, for the technique's `prepare` and the binding-budget test. */
export const HISTORY_KERNELS = [
    { entryPoint: HISTORY_ENTRY_POINT.initCachedTileLRU, code: INIT_CACHED_TILE_LRU_WGSL, group: INIT_CACHED_TILE_LRU_GROUP, indirect: false },
    { entryPoint: HISTORY_ENTRY_POINT.reproject, code: REPROJECT_WGSL, group: REPROJECT_GROUP, indirect: false },
    { entryPoint: HISTORY_ENTRY_POINT.count, code: COUNT_WGSL, group: COUNT_GROUP, indirect: false },
    { entryPoint: HISTORY_ENTRY_POINT.scatter, code: SCATTER_WGSL, group: SCATTER_GROUP, indirect: true },
    { entryPoint: HISTORY_ENTRY_POINT.patch, code: PATCH_WGSL, group: PATCH_GROUP, indirect: true },
    { entryPoint: HISTORY_ENTRY_POINT.reorder, code: REORDER_WGSL, group: REORDER_GROUP, indirect: false },
    { entryPoint: HISTORY_ENTRY_POINT.copyPreviousScreen, code: COPY_PREVIOUS_SCREEN_WGSL, group: COPY_PREVIOUS_SCREEN_GROUP, indirect: false },
] as const;
