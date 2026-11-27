/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.comp
 * (InitCachedTileLRU, ReprojectScreenProbes, CountScreenProbes, ScatterScreenProbes,
 * PatchScreenProbes, ReorderScreenProbes).
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
 * The screen probes' history (RC4.1): the kernels around spawn, sample and
 * blend that carry probes from one frame to the next, one WGSL entry point per
 * Capsaicin kernel, in the order `GI1::render` runs them (gi1.cpp:2256-2557):
 *
 * 1. `ReprojectScreenProbes` (gi1.comp:292-499): every probe tile looks for
 *    last frame's probe that its pixels moved from (velocity), with the plane
 *    and normal tests; the best pixel becomes the tile's probe and last
 *    frame's cells are moved into its frame by their hit distances. Tiles
 *    with none are disocclusions: listed as empty tiles for the patch.
 * 2. `CountScreenProbes` (gi1.comp:501-539), the platform's scan and
 *    `ScatterScreenProbes` (gi1.comp:541-553): the off-screen cache's tiles
 *    that project onto the screen, listed per probe tile.
 * 3. `PatchScreenProbes` (gi1.comp:615-644), after the spawn and compaction:
 *    each empty tile takes the rays of a random spawned probe whose tile
 *    already had a reprojected probe (adaptive sampling).
 * 4. `ReorderScreenProbes` (gi1.comp:1333-1354), after the blend, with the
 *    platform's scan of the LRU flags: entries used this frame move to the
 *    back of the LRU list, in the order they were used.
 *
 * `InitCachedTileLRU` (gi1.comp:286-290) runs when the cache is made or its
 * history dropped. Sampling and blending, which read and write the cache too,
 * are in screen-probes.ts.
 */

import { CAPSAICIN_WGSL } from './capsaicin.ts';
import {
    COUNTERS_WGSL,
    FRAME_WGSL,
    LAYOUT,
    previousScreenWgsl,
    HISTORY_WGSL,
    historyWordsWgsl,
    RADIANCE_BACKUP_WGSL,
    screenDepthWgsl,
    screenNormalWgsl,
    screenVelocityWgsl,
    UNIFORM_ENTRY,
    WIDE_RADIANCE_SUMS_WGSL,
} from './screen-probes.ts';
import { SH_COEFFICIENTS } from './spherical-harmonics.ts';

const { storage, texture, storageTexture, maskTexture } = LAYOUT;
const COMPUTE = 0x4;

/** Entry points: Capsaicin's kernel names. */
export const TEMPORAL_ENTRY_POINT = {
    initCachedTileLRU: 'InitCachedTileLRU',
    reproject: 'ReprojectScreenProbes',
    count: 'CountScreenProbes',
    scatter: 'ScatterScreenProbes',
    patch: 'PatchScreenProbes',
    reorder: 'ReorderScreenProbes',
} as const;

/** Workgroup sizes, as Capsaicin's `[numthreads]`. */
export const TEMPORAL_WORKGROUP = {
    /** ReprojectScreenProbes (gi1.comp:292): one 8 × 8 probe tile per workgroup. */
    reproject: 8,
    /** InitCachedTileLRU, CountScreenProbes, ScatterScreenProbes, PatchScreenProbes, ReorderScreenProbes: 64. */
    list: 64,
} as const;

/** An `rg32uint` texture read with `textureLoad` (the SH of last frame). */
function uintTexture(binding: number): GPUBindGroupLayoutEntry {
    return { binding, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } };
}

/** Thread `did` of a 1D dispatch split into rows of `workgroups.x` workgroups (dispatchSplit). */
const LINEAR_INDEX_WGSL = /* wgsl */ `
fn rcLinearIndex(workgroup: vec3u, workgroups: vec3u, localIndex: u32) -> u32 {
    return (workgroup.y * workgroups.x + workgroup.x) * 64u + localIndex;
}

// The probe tiles of the screen: g_ScreenProbes_ProbeCachedTileIndexBuffer's size, and the
// length of the LRU lists (max_probe_count, gi1.cpp:303).
fn rcProbeTileCount() -> u32 {
    return rcFrame.probeCount.x * rcFrame.probeCount.y;
}
`;

// ---- InitCachedTileLRU ---------------------------------------------------------------------------

export const INIT_CACHED_TILE_LRU_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/InitCachedTileLRU group 1',
    entries: [UNIFORM_ENTRY, storage(1, 'storage')],
};

export const INIT_CACHED_TILE_LRU_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${LINEAR_INDEX_WGSL}
${historyWordsWgsl(1)}

// InitCachedTileLRU (gi1.comp:286-290): every entry points at its own slot, flagged unused.
@compute @workgroup_size(64)
fn ${TEMPORAL_ENTRY_POINT.initCachedTileLRU}(@builtin(workgroup_id) workgroup: vec3u, @builtin(num_workgroups) workgroups: vec3u, @builtin(local_invocation_index) localIndex: u32) {
    let did = rcLinearIndex(workgroup, workgroups, localIndex);
    if (did >= rcProbeTileCount()) {
        return; // out of bounds
    }
    rcHistoryWords[rcCachedTileLRUWord(did)] = did | 0x80000000u;
}
`;

// ---- ReprojectScreenProbes -----------------------------------------------------------------------

export const REPROJECT_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ReprojectScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        texture(1),
        texture(2),
        texture(3),
        texture(4),
        texture(5),
        maskTexture(6),
        texture(7),
        storageTexture(8, 'rgba16float'),
        storageTexture(9, 'r32uint'),
        uintTexture(10),
        storageTexture(11, 'rg32uint'),
        storage(12, 'storage'),
        storage(13, 'storage'),
        storage(14, 'read-only-storage'),
        storage(15, 'storage'),
    ],
};

export const REPROJECT_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenDepthWgsl(1)}
${screenNormalWgsl(2)}
${screenVelocityWgsl(3)}
${previousScreenWgsl(4, 5)}
${COUNTERS_WGSL}
${WIDE_RADIANCE_SUMS_WGSL}
${HISTORY_WGSL}
${RADIANCE_BACKUP_WGSL}
// g_ScreenProbes_PreviousProbeMaskBuffer and g_ScreenProbes_PreviousProbeBuffer: last frame's probes.
@group(1) @binding(6) var rcPreviousProbeMask: texture_2d<u32>;
@group(1) @binding(7) var rcPreviousProbeBuffer: texture_2d<f32>;
// g_ScreenProbes_ProbeBuffer and g_ScreenProbes_ProbeMaskBuffer (level 0): this frame's, written
// for every probe tile.
@group(1) @binding(8) var rcProbeBuffer: texture_storage_2d<rgba16float, write>;
@group(1) @binding(9) var rcProbeMaskBuffer: texture_storage_2d<r32uint, write>;
// g_ScreenProbes_PreviousProbeSHBuffer and g_ScreenProbes_ProbeSHBuffer as textures (register D26).
@group(1) @binding(10) var rcPreviousProbeSH: texture_2d<u32>;
@group(1) @binding(11) var rcProbeSH: texture_storage_2d<rg32uint, write>;
// g_ScreenProbes_EmptyTileBuffer: disoccluded tiles the spawn will not fill.
@group(1) @binding(12) var<storage, read_write> rcEmptyTiles: array<u32>;
@group(1) @binding(13) var<storage, read_write> rcCounters: array<atomic<u32>, 16>;
// The history sample counts per atlas cell (D46): last frame's, and this frame's (written here, in
// the history words).
@group(1) @binding(14) var<storage, read> rcPreviousCellCounts: array<u32>;
${historyWordsWgsl(15)}

// lds_ScreenProbes_Reprojection: the best pixel's score (half of distance / cell size in the
// high 16 bits) and its lane (the low 16). One 8 x 8 probe per workgroup, so one segment.
var<workgroup> rcReprojection: atomic<u32>;

// The probe-tile pixel this thread is, if it is on the screen and not sky.
fn rcIsSurfacePixel(pixel: vec2u) -> bool {
    return all(pixel < rcFrame.bufferDimensions) && !rcIsSkyPixel(pixel);
}

// The probe of last frame a pixel's motion leads to (gi1.comp:329-336, 361-365): its seed in
// last frame's mask, or RC_INVALID_ID when the pixel came from off the screen.
fn rcPreviousProbeMaskAt(pixel: vec2u) -> u32 {
    let uv = (vec2f(pixel) + 0.5) / vec2f(rcFrame.bufferDimensions);
    // int2(previous_uv * g_BufferDimensions): truncated towards zero, so a point up to a pixel
    // off the left or top edge still reads the first column or row.
    let previousPos = vec2i(rcPreviousUv(pixel, uv) * vec2f(rcFrame.bufferDimensions));
    if (any(previousPos < vec2i(0)) || any(previousPos >= vec2i(rcFrame.bufferDimensions))) {
        return RC_INVALID_ID;
    }
    return textureLoad(rcPreviousProbeMask, vec2u(previousPos) / RC_PROBE_SIZE, 0).x;
}

// A probe of last frame: its position from last frame's depth and camera, and its normal.
fn rcPreviousProbePosition(probeSeed: vec2u) -> vec3f {
    let probeUv = (vec2f(probeSeed) + 0.5) / vec2f(rcFrame.bufferDimensions);
    return rcTransformPreviousPointProjection(probeUv, rcLoadPreviousDepth(probeSeed));
}

// ReprojectScreenProbes (gi1.comp:292-499).
@compute @workgroup_size(8, 8)
fn ${TEMPORAL_ENTRY_POINT.reproject}(
    @builtin(global_invocation_id) globalId: vec3u,
    @builtin(workgroup_id) groupId: vec3u,
    @builtin(local_invocation_id) localId: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let did = globalId.xy;
    // A pixel past the screen's edge (the last tiles are cut) reads as sky, as Capsaicin's
    // out-of-bounds normal load of 0 does (gi1.comp:296-297).
    let isSkyPixel = !rcIsSurfacePixel(did);
    var normal = vec3f(0.0, 0.0, 1.0);
    var worldPos = vec3f(0.0);
    if (!isSkyPixel) {
        normal = rcLoadNormal(did);
        worldPos = rcTransformPointProjection((vec2f(did) + 0.5) / vec2f(rcFrame.bufferDimensions), rcLoadDepth(did));
    }
    let cellSize = distance(rcFrame.eye, worldPos) * rcFrame.cellSize;
    let cell = localId.xy % RC_PROBE_SIZE;
    let cellIndex = cell.x + cell.y * RC_PROBE_SIZE;

    if (cellIndex == 0u) {
        atomicStore(&rcReprojection, (rcF32ToF16Bits(65504.0) << 16u) | 0xffffu);
    }
    rcClearRadianceSums(localIndex);
    workgroupBarrier();

    // Try and find the closest matching probe in the previous frame. Each probe is assigned a
    // score and we atomically pick the best candidate out of all the 8x8, i.e., 64, potential
    // positions (gi1.comp:324-352).
    if (!isSkyPixel) {
        let probeMask = rcPreviousProbeMaskAt(did);
        if (probeMask != RC_INVALID_ID) {
            let probeSeed = rcUnpackSeed(probeMask);
            let probePos = rcPreviousProbePosition(probeSeed);
            let probeNormal = rcLoadPreviousNormal(probeSeed);
            if (abs(dot(probePos - worldPos, normal)) < cellSize && dot(normal, probeNormal) > 0.95) {
                let probeScore = (rcF32ToF16Bits(distance(probePos, worldPos) / cellSize) << 16u) | localIndex;
                atomicMin(&rcReprojection, probeScore);
            }
        }
    }
    workgroupBarrier();

    // Now we map each lane to a cell in the probe and re-use the radiance values temporally
    // (gi1.comp:355-404).
    let localLane = atomicLoad(&rcReprojection) & 0xffffu;
    let found = localLane != 0xffffu;
    let seed = (groupId.xy << vec2u(3u)) + vec2u(localLane & 7u, localLane >> 3u);
    var seedNormal = vec3f(0.0, 0.0, 1.0);
    var previousProbeMask = RC_INVALID_ID;
    if (found) {
        previousProbeMask = rcPreviousProbeMaskAt(seed);
        seedNormal = rcLoadNormal(seed);
        let seedPos = rcTransformPointProjection((vec2f(seed) + 0.5) / vec2f(rcFrame.bufferDimensions), rcLoadDepth(seed));

        let probeSeed = rcUnpackSeed(previousProbeMask);
        let probePos = rcPreviousProbePosition(probeSeed);
        let probeNormal = rcLoadPreviousNormal(probeSeed);
        let probeCell = (probeSeed / RC_PROBE_SIZE) * RC_PROBE_SIZE + cell;
        let probeRadiance = textureLoad(rcPreviousProbeBuffer, probeCell, 0);
        let probeDirection = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(cell) + 0.5) / f32(RC_PROBE_SIZE)), probeNormal);

        let hitPoint = probePos + probeDirection * probeRadiance.w;
        let reprojected = hitPoint - seedPos;
        let reprojectedLen = length(reprojected);
        let reprojectedDir = reprojected / max(reprojectedLen, 1e-30); // normalize

        // A backup fill (negative hit distance) is no history: it is made again, never moved (D47).
        if (probeRadiance.w >= 0.0 && dot(seedNormal, reprojectedDir) > 0.0) {
            // Kept in the probe: a direction on the square's far edge would give cell 8 (D45).
            let remapCell = min(vec2u(rcMapToHemiOctahedronInverse(rcToProbeFrame(reprojectedDir, seedNormal)) * f32(RC_PROBE_SIZE)), vec2u(RC_PROBE_SIZE - 1u));
            let remapCellIndex = remapCell.x + remapCell.y * RC_PROBE_SIZE;
            rcAccumulateRadiance(remapCellIndex, rcQuantizeRadiance(vec4f(probeRadiance.xyz, min(reprojectedLen, RC_LARGEST_HALF))), rcPreviousCellCounts[rcAtlasIndex(probeCell)]);
        }
    }
    workgroupBarrier();

    // Calculate the radiance backup value to be used for unvisited cells (gi1.comp:407-429; D41).
    rcComputeRadianceBackup(localIndex);

    let probe = did / RC_PROBE_SIZE;

    // No probe found, this is a disocclusion (gi1.comp:431-461).
    if (!found) {
        if (cellIndex == 0u) {
            if (!isSkyPixel) {
                let jitter = vec2u(min(rcCalculateHaltonSequence(rcFrame.frameIndex) * f32(RC_PROBE_SPAWN_TILE_SIZE), vec2f(f32(RC_PROBE_SPAWN_TILE_SIZE) - 1.0)));
                // Check whether this disoccluded tile won't be filled during the probe spawning pass.
                // If it won't be spawned, we append it to the list of "empty tiles" so the ray
                // distribution can get re-balanced to achieve adaptive sampling.
                if (any((jitter / RC_PROBE_SIZE) != (probe % (RC_PROBE_SPAWN_TILE_SIZE / RC_PROBE_SIZE)))) {
                    let emptyTileIndex = atomicAdd(&rcCounters[RC_COUNTER_EMPTY_TILES], 1u);
                    rcEmptyTiles[emptyTileIndex] = probe.x + probe.y * rcFrame.probeCount.x;
                }
            }
            textureStore(rcProbeMaskBuffer, probe, vec4u(RC_INVALID_ID, 0u, 0u, 0u));
        }
        return; // reprojection failed :'(
    }

    // Reproject the spherical harmonics (gi1.comp:463-479).
    if (cellIndex < ${SH_COEFFICIENTS}u) {
        let previousProbe = rcUnpackSeed(previousProbeMask) / RC_PROBE_SIZE;
        if (cellIndex == 0u) {
            textureStore(rcProbeMaskBuffer, probe, vec4u(rcPackSeed(seed), 0u, 0u, 0u));
        }
        let texel = vec2u(cellIndex % 3u, cellIndex / 3u);
        textureStore(rcProbeSH, probe * 3u + texel, textureLoad(rcPreviousProbeSH, previousProbe * 3u + texel, 0));
    }

    // And reproject the radiance (gi1.comp:481-498).
    var radiance = rcRecoverRadianceSum(cellIndex);
    let sampleCount = rcRadianceSampleCount(cellIndex);
    var count = 0u;
    if (sampleCount > 0u) {
        radiance /= f32(sampleCount);
        count = rcMeanHistoryCount(cellIndex);
    } else {
        radiance = rcRadianceBackup[0]; // a fill (D47)
    }
    textureStore(rcProbeBuffer, did, rcNearestHalf4(vec4f(radiance.xyz, min(radiance.w, RC_LARGEST_HALF))));
    rcHistoryWords[rcCellCountWord(rcAtlasIndex(did))] = count;
}
`;

// ---- CountScreenProbes and ScatterScreenProbes ---------------------------------------------------

export const COUNT_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/CountScreenProbes group 1',
    // Binding 1 is the history words (read-write: other passes write them).
    entries: [UNIFORM_ENTRY, storage(1, 'storage'), storage(2, 'read-only-storage'), storage(3, 'storage'), storage(4, 'storage'), storage(5, 'storage'), storage(6, 'storage')],
};

export const COUNT_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${LINEAR_INDEX_WGSL}
${COUNTERS_WGSL}
${historyWordsWgsl(1)}
// g_ScreenProbes_ProbeCachedTileIndexBuffer: each cached slot's position (f32 bits) and packed normal,
// as words of a buffer rather than an rgba32float texture (register D45).
@group(1) @binding(2) var<storage, read> rcCachedTileIndex: array<vec4u>;
@group(1) @binding(3) var<storage, read_write> rcCachedTileListCount: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> rcCachedTileListElements: array<vec4u>;
@group(1) @binding(5) var<storage, read_write> rcCachedTileLRUFlags: array<u32>;
@group(1) @binding(6) var<storage, read_write> rcCounters: array<atomic<u32>, 16>;

// CountScreenProbes (gi1.comp:501-539): each used cache entry whose probe projects onto the
// screen is listed under the probe tile it lands in; every entry's flag is set (not used yet).
@compute @workgroup_size(64)
fn ${TEMPORAL_ENTRY_POINT.count}(@builtin(workgroup_id) workgroup: vec3u, @builtin(num_workgroups) workgroups: vec3u, @builtin(local_invocation_index) localIndex: u32) {
    let did = rcLinearIndex(workgroup, workgroups, localIndex);
    if (did >= rcProbeTileCount()) {
        return; // out of bounds
    }

    let cachedTileIndex = rcHistoryWords[rcCachedTileLRUWord(did)];
    if ((cachedTileIndex & 0x80000000u) == 0u) {
        let worldPos = bitcast<vec3f>(rcCachedTileIndex[cachedTileIndex].xyz);
        // transformPointProjection(world_pos, g_ViewProjection), camera-relative. A point behind
        // the camera (w <= 0) is never on the screen; Capsaicin relies on its depth test for that (D45).
        let clip = rcFrame.viewProjection * vec4f(worldPos - rcFrame.eye, 1.0);
        var homogeneous = clip.xyz / clip.w;
        homogeneous = vec3f(0.5 * vec2f(homogeneous.x, -homogeneous.y) + 0.5, homogeneous.z);

        if (clip.w > 0.0 && all(homogeneous > vec3f(0.0)) && all(homogeneous < vec3f(1.0))) {
            let probe = vec2u(homogeneous.xy * vec2f(rcFrame.probeCount));
            let probeIndex = probe.x + probe.y * rcFrame.probeCount.x;
            let elementIndex = atomicAdd(&rcCounters[RC_COUNTER_CACHED_TILE_LIST_ELEMENTS], 1u);
            let indexInTile = atomicAdd(&rcCachedTileListCount[probeIndex], 1u);
            rcCachedTileListElements[elementIndex] = vec4u(did, probeIndex, indexInTile, 0u);
        }
    }

    rcCachedTileLRUFlags[did] = 1u;
}
`;

export const SCATTER_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ScatterScreenProbes group 1',
    entries: [storage(0, 'read-only-storage'), storage(1, 'read-only-storage'), storage(2, 'read-only-storage'), storage(3, 'storage')],
};

/** Built with the platform's `compute.indirectModule` (`GiIndirectArgs`, `giIndirectItemIndex`). */
export const SCATTER_WGSL = /* wgsl */ `
// The dispatch's arguments, sized on the list's element count.
@group(1) @binding(0) var<storage, read> rcScatterArgs: GiIndirectArgs;
@group(1) @binding(1) var<storage, read> rcCachedTileListElements: array<vec4u>;
// The platform's exclusive scan of the per-tile counts: where each tile's list starts.
@group(1) @binding(2) var<storage, read> rcCachedTileListIndex: array<u32>;
${historyWordsWgsl(3, false)}

// ScatterScreenProbes (gi1.comp:541-553).
@compute @workgroup_size(64)
fn ${TEMPORAL_ENTRY_POINT.scatter}(@builtin(workgroup_id) workgroup: vec3u, @builtin(num_workgroups) workgroups: vec3u, @builtin(local_invocation_index) localIndex: u32) {
    let did = giIndirectItemIndex(workgroup, workgroups, localIndex, 64u);
    if (did >= rcScatterArgs.count) {
        return; // out of bounds
    }
    let listElement = rcCachedTileListElements[did].xyz;
    rcHistoryWords[rcCachedTileListWord(rcCachedTileListIndex[listElement.y] + listElement.z)] = listElement.x;
}
`;

// ---- PatchScreenProbes ---------------------------------------------------------------------------

export const PATCH_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/PatchScreenProbes group 1',
    entries: [UNIFORM_ENTRY, texture(1), storage(2, 'read-only-storage'), storage(3, 'storage'), storage(4, 'read-only-storage'), storage(5, 'read-only-storage'), storage(6, 'storage')],
};

/** Built with the platform's `compute.indirectModule`. */
export const PATCH_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenDepthWgsl(1)}
${COUNTERS_WGSL}
// The dispatch's arguments, sized on the empty-tile count.
@group(1) @binding(2) var<storage, read> rcPatchArgs: GiIndirectArgs;
@group(1) @binding(3) var<storage, read_write> rcCounters: array<atomic<u32>, 16>;
@group(1) @binding(4) var<storage, read> rcEmptyTiles: array<u32>;
@group(1) @binding(5) var<storage, read> rcOverrideTiles: array<u32>;
// g_ScreenProbes_ProbeSpawnBuffer, with the count in front (RcProbeSpawn); its seeds are swapped atomically.
struct RcProbeSpawnAtomic {
    count: u32,
    seeds: array<atomic<u32>>,
}
@group(1) @binding(6) var<storage, read_write> rcProbeSpawn: RcProbeSpawnAtomic;

// PatchScreenProbes (gi1.comp:615-644): stochastically patch the overridable tiles using
// empty ones (a.k.a., adaptive sampling).
@compute @workgroup_size(64)
fn ${TEMPORAL_ENTRY_POINT.patch}(@builtin(workgroup_id) workgroup: vec3u, @builtin(num_workgroups) workgroups: vec3u, @builtin(local_invocation_index) localIndex: u32) {
    let did = giIndirectItemIndex(workgroup, workgroups, localIndex, 64u);
    let overrideTileCount = atomicLoad(&rcCounters[RC_COUNTER_OVERRIDE_TILES]);
    if (overrideTileCount == 0u || did >= rcPatchArgs.count) {
        return; // out of bounds
    }

    let probeIndex = rcEmptyTiles[did];
    let probe = vec2u(probeIndex % rcFrame.probeCount.x, probeIndex / rcFrame.probeCount.x);
    let jitter = vec2u(min(rcCalculateHaltonSequence(rcFrame.frameIndex) * f32(RC_PROBE_SIZE), vec2f(f32(RC_PROBE_SIZE) - 1.0)));
    let seed = min(probe * RC_PROBE_SIZE + jitter, rcFrame.bufferDimensions - 1u);
    if (rcIsSkyPixel(seed)) {
        return; // sky pixel
    }

    var random = rcMakeRandom(did, rcFrame.frameIndex);
    let index = rcRandIntBelow(&random, overrideTileCount);
    atomicStore(&rcProbeSpawn.seeds[rcOverrideTiles[index]], rcPackSeed(seed));
}
`;

// ---- ReorderScreenProbes -------------------------------------------------------------------------

export const REORDER_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ReorderScreenProbes group 1',
    // Binding 1 is the history words (the LRU list and the MRU order; read-write: other passes write them).
    entries: [UNIFORM_ENTRY, storage(1, 'storage'), storage(2, 'read-only-storage'), storage(3, 'read-only-storage'), storage(5, 'storage')],
};

export const REORDER_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${LINEAR_INDEX_WGSL}
${historyWordsWgsl(1)}
@group(1) @binding(2) var<storage, read> rcCachedTileLRUFlags: array<u32>;
// The platform's exclusive scan of the flags over max_probe_count + 1 entries: an entry not used
// this frame keeps its place among those (flag 1); the last word counts them.
@group(1) @binding(3) var<storage, read> rcCachedTileLRUIndex: array<u32>;
// g_ScreenProbes_PreviousProbeCachedTileLRUBuffer: next frame's LRU list.
@group(1) @binding(5) var<storage, read_write> rcNextCachedTileLRU: array<u32>;

// ReorderScreenProbes (gi1.comp:1333-1354): the entries used this frame (flag 0) go after the
// others, in the order they were used, so the least recently used come first.
@compute @workgroup_size(64)
fn ${TEMPORAL_ENTRY_POINT.reorder}(@builtin(workgroup_id) workgroup: vec3u, @builtin(num_workgroups) workgroups: vec3u, @builtin(local_invocation_index) localIndex: u32) {
    let did = rcLinearIndex(workgroup, workgroups, localIndex);
    let count = rcProbeTileCount();
    if (did >= count) {
        return; // out of bounds
    }

    let cachedTileIndex = rcHistoryWords[rcCachedTileLRUWord(did)];
    var cachedTileOffset = rcCachedTileLRUIndex[did];
    if (rcCachedTileLRUFlags[did] == 0u) {
        cachedTileOffset = rcHistoryWords[rcCachedTileMRUWord(did)] + rcCachedTileLRUIndex[count];
    }
    // Every offset is below count when the flags and the MRU order agree, which the GPU page checks.
    if (cachedTileOffset < count) {
        rcNextCachedTileLRU[cachedTileOffset] = cachedTileIndex;
    }
}
`;
