/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.comp
 * (FilterProbeMask, FilterScreenProbes, ProjectScreenProbes, InterpolateScreenProbes) and
 * src/core/src/render_techniques/gi1/screen_probes.hlsl (ScreenProbes_FindClosestProbe).
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
 * Phase 2's second half (RC2.2): from the blended probe atlas to irradiance
 * per pixel, the three kernels Capsaicin runs after `BlendScreenProbes`
 * (`GI1::render`, gi1.cpp:2561-2636), each named after its entry point:
 *
 * 1. `FilterProbeMask` (gi1.comp:248-284), once per mip level of the probe
 *    mask: each texel keeps the first probe of its 2 × 2 children, so any
 *    pixel can find a probe nearby in a few loads (`FindClosestProbe`).
 * 2. `ProjectScreenProbes` (gi1.comp:1445-1512): each probe's 64 cells to L2
 *    SH (nine coefficients), one probe per workgroup.
 * 3. `InterpolateScreenProbes` (gi1.comp:1514-1640): per pixel, up to four
 *    nearby probes weighted by plane distance, depth and normal, and their SH
 *    evaluated for the pixel's normal: irradiance E, which `ResolveGI1`
 *    (gi1.frag:76-150) turns into reflected light with `albedo / pi`, as
 *    three's composite does.
 *
 * Since RC4.2, `FilterScreenProbes` (gi1.comp:1356-1442, at the end of this
 * file) runs twice between the mask and the projection, across then down,
 * and the interpolation reads the platform's SSGI (Capsaicin's
 * `HAS_OCCLUSION` branch). The GI denoiser that reads the interpolated image
 * is RC4.3; until then the shading module reads this kernel's output.
 *
 * Differences from Capsaicin, in the register (docs/design/radiance-cache.md):
 * the projection weight is a cell's solid angle (D25) and the basis the
 * cell's mean (D34), the SH live in a
 * texture (D26), the interpolation's jitter is white noise (D27), SH
 * coefficients are clamped to the half range (D28), the filter keeps every
 * finite neighbour value (D49) and leaves backup fills fills (D50), and the
 * interpolation writes the lit image next frame's SSGI reads (D51). Without
 * SSGI it runs Capsaicin's `#ifndef HAS_OCCLUSION` branch.
 */

import { CAPSAICIN_WGSL } from './capsaicin.ts';
import { FRAME_WGSL, HISTORY_WGSL, PROBE_SPAWN_STRUCT_WGSL, screenDepthWgsl, screenNormalWgsl, UNIFORM_ENTRY, WORKGROUP_INDEX_WGSL } from './screen-probes.ts';
import { shIrradianceWgsl, SPHERICAL_HARMONICS_WGSL } from './spherical-harmonics.ts';

const COMPUTE = 0x4;

/** Entry points: Capsaicin's kernel names. */
export const SCREEN_IRRADIANCE_ENTRY_POINT = {
    filterProbeMask: 'FilterProbeMask',
    filter: 'FilterScreenProbes',
    project: 'ProjectScreenProbes',
    interpolate: 'InterpolateScreenProbes',
} as const;

/** Workgroup sizes, as Capsaicin's `[numthreads]`. */
export const SCREEN_IRRADIANCE_WORKGROUP = {
    /** FilterProbeMask, InterpolateScreenProbes (gi1.comp:248, 1514): 8 × 8. */
    tile: 8,
    /** ProjectScreenProbes (gi1.comp:1445): one probe's 64 cells per workgroup. */
    probe: 64,
} as const;

/** Bytes of `RcScreenProbesConstants`. */
export const SCREEN_PROBES_CONSTANTS_BYTES = 32;

/**
 * Byte offsets of `RcScreenProbesConstants`' fields by WGSL's layout rules
 * (checked against the struct by a unit test).
 */
export const SCREEN_PROBES_CONSTANTS_OFFSETS = {
    viewForward: 0,
    cellSize: 12,
    probeMaskMipCount: 16,
    interpolationJitter: 20,
    hasOcclusion: 24,
    writeCombined: 28,
} as const;

/**
 * Capsaicin's `ScreenProbesConstants` (gi1_shared.h:38-46) beyond what
 * `RcFrame` has, and `g_NearFar`'s use: the interpolation's linear depth.
 *
 * Capsaicin linearises its reversed-Z depth with the camera planes
 * (`toLinearDepth`, transform.hlsl:144-148): the distance along the view
 * axis. We take the same distance from the reconstructed position, the dot
 * product with the camera's forward axis, which holds for any depth
 * convention the host draws with.
 */
const SCREEN_PROBES_CONSTANTS_WGSL = /* wgsl */ `
struct RcScreenProbesConstants {
    viewForward: vec3f,         // the camera's forward axis, world space: linear depth is dot(p - eye, viewForward)
    cellSize: f32,              // cell_size: tan(fovY * probe_size * max(1 / h, h / w^2)) (gi1.cpp:1839-1843), per unit distance
    probeMaskMipCount: u32,     // probe_mask_mip_count (gi1.cpp:1846)
    interpolationJitter: u32,   // 1: the lookup position is jittered (Capsaicin's); 0: not (RadianceCacheSettings.interpolationJitter)
    hasOcclusion: u32,          // 1: SSGI's occlusion, bent normal and near-field light are bound (Capsaicin's HAS_OCCLUSION)
    writeCombined: u32,         // 1: write this frame's lit image (lighting + albedo E / pi) for next frame's SSGI
}
@group(1) @binding(1) var<uniform> rcScreenProbes: RcScreenProbesConstants;
`;

/**
 * `ScreenProbes_FindClosestProbe(pos)` (screen_probes.hlsl:110-131) over a
 * probe mask given as a parameter, its sizes and mip count from the texture
 * itself: the probe of `position`'s tile, else the first probe found going up
 * the mip chain. For the debug views, which have no `RcScreenProbesConstants`
 * (the interpolation's own copy below reads them).
 */
export function findClosestProbeWgsl(functionName: string, maskParameter: string): string {
    return /* wgsl */ `
fn ${functionName}(position: vec2u, ${maskParameter}: texture_2d<u32>) -> u32 {
    var dims = textureDimensions(${maskParameter}, 0);
    var pos = min(position / RC_PROBE_SIZE, dims - 1u);
    let levels = textureNumLevels(${maskParameter});
    for (var i = 0u; i < levels; i++) {
        let probe = textureLoad(${maskParameter}, pos, i32(i)).x;
        if (probe != RC_INVALID_ID) {
            return probe;
        }
        dims = max(dims >> vec2u(1u), vec2u(1u));
        pos = min(pos >> vec2u(1u), dims - 1u);
    }
    return RC_INVALID_ID;
}
`;
}

const CONSTANTS_ENTRY: GPUBindGroupLayoutEntry = { binding: 1, visibility: COMPUTE, buffer: { type: 'uniform', minBindingSize: SCREEN_PROBES_CONSTANTS_BYTES } };

// ---- FilterProbeMask ----------------------------------------------------------------------------

export const FILTER_PROBE_MASK_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/FilterProbeMask group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } },
        { binding: 1, visibility: COMPUTE, storageTexture: { access: 'write-only', format: 'r32uint', viewDimension: '2d' } },
    ],
};

export const FILTER_PROBE_MASK_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
// g_ScreenProbes_InProbeMaskBuffer: the probe mask's level i - 1.
@group(1) @binding(0) var rcInProbeMaskBuffer: texture_2d<u32>;
// g_ScreenProbes_OutProbeMaskBuffer: level i of the same texture.
@group(1) @binding(1) var rcOutProbeMaskBuffer: texture_storage_2d<r32uint, write>;

// FilterProbeMask (gi1.comp:248-284), run once per level from 1 (gi1.cpp:2572-2584).
@compute @workgroup_size(8, 8)
fn ${SCREEN_IRRADIANCE_ENTRY_POINT.filterProbeMask}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.xy;
    let dims = textureDimensions(rcInProbeMaskBuffer);

    if (any((did << vec2u(1u)) >= dims)) {
        return; // out of bounds
    }

    var probeMask = RC_INVALID_ID;

    // Here, we look for a valid probe in the 2x2 region from the upper mip in
    // order to populate the lower mips with as many valid probes as possible.
    for (var y = 0u; y < 2u; y++) {
        for (var x = 0u; x < 2u; x++) {
            let pos = (did << vec2u(1u)) + vec2u(x, y);

            probeMask = RC_INVALID_ID;
            if (all(pos < dims)) {
                probeMask = textureLoad(rcInProbeMaskBuffer, pos, 0).x;
            }

            if (probeMask != RC_INVALID_ID) {
                break;
            }
        }

        if (probeMask != RC_INVALID_ID) {
            break;
        }
    }

    textureStore(rcOutProbeMaskBuffer, did, vec4u(probeMask, 0u, 0u, 0u));
}
`;

// ---- ProjectScreenProbes ------------------------------------------------------------------------

export const PROJECT_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/ProjectScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        { binding: 1, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 4, visibility: COMPUTE, storageTexture: { access: 'write-only', format: 'rg32uint', viewDimension: '2d' } },
    ],
};

/** The solid angle of one probe cell: the hemisphere's 2 pi over 64 equal-area cells (register D25). */
export const SH_CELL_SOLID_ANGLE = (2 * Math.PI) / 64;
/** Points across a cell, each way, that the projection averages the SH basis over (register D34). */
export const SH_CELL_SUBSAMPLES = 4;

export const PROJECT_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${SPHERICAL_HARMONICS_WGSL}
${FRAME_WGSL}
${screenNormalWgsl(1)}
${PROBE_SPAWN_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
@group(1) @binding(2) var<storage, read> rcProbeSpawn: RcProbeSpawn;
// g_ScreenProbes_ProbeBuffer: the blended probe atlas.
@group(1) @binding(3) var rcProbeBuffer: texture_2d<f32>;
// g_ScreenProbes_ProbeSHBuffer, as a texture (register D26): nine packHalf4 coefficients per probe tile.
@group(1) @binding(4) var rcProbeSHBuffer: texture_storage_2d<rg32uint, write>;

// How a cell counts in the projection (register D25 for the weight, D34 for the basis). The coefficients are meant to be the SH of
// the radiance, c_i = the integral of L Y_i over the hemisphere, here a sum over 64 equal-area
// cells of the cell's radiance times the integral of Y_i over the cell: its solid angle times
// the basis averaged over the cell, from RC_SH_CELL_SUBSAMPLES^2 points spread evenly over it.
// Capsaicin takes 1 / probe_size and the basis at the cell centre (gi1.comp:1466-1474): 4 / pi
// times the solid angle, and a centre point that leans towards the normal, so a constant sky
// of radiance L reads about 4.08 L where irradiance is pi L (1.30 x) for one frame's probes with
// no history; Capsaicin's steady state with history is not measured yet. The cell-mean basis reads
// within 0.2% of pi L (screen-irradiance.test.ts).
const RC_SH_CELL_SOLID_ANGLE: f32 = ${SH_CELL_SOLID_ANGLE};
const RC_SH_CELL_SUBSAMPLES: u32 = ${SH_CELL_SUBSAMPLES}u;

// The SH basis averaged over a cell of the probe about 'normal' (see above).
fn rcProbeCellSH(cell: vec2u, normal: vec3f) -> array<f32, 9> {
    var cellSH: array<f32, 9>;
    for (var row = 0u; row < RC_SH_CELL_SUBSAMPLES; row++) {
        for (var column = 0u; column < RC_SH_CELL_SUBSAMPLES; column++) {
            let inCell = (vec2f(f32(column), f32(row)) + 0.5) / f32(RC_SH_CELL_SUBSAMPLES);
            let direction = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(cell) + inCell) / f32(RC_PROBE_SIZE)), normal);
            let basis = rcSHGetCoefficients(direction);
            for (var j = 0u; j < RC_SH_COEFFICIENTS; j++) {
                cellSH[j] += basis[j];
            }
        }
    }
    for (var j = 0u; j < RC_SH_COEFFICIENTS; j++) {
        cellSH[j] /= f32(RC_SH_CELL_SUBSAMPLES * RC_SH_CELL_SUBSAMPLES);
    }
    return cellSH;
}

// lds_ScreenProbes_ProbeSHBuffer (screen_probes.hlsl:81): each cell's nine weighted terms.
var<workgroup> rcProbeSHValues: array<vec4f, ${9 * 64}>;

// ProjectScreenProbes (gi1.comp:1445-1512): one probe's 64 cells to SH. Dispatched over every
// possible ray (gi1.cpp:2618-2625); workgroups past the probe count return after the barrier.
@compute @workgroup_size(64)
fn ${SCREEN_IRRADIANCE_ENTRY_POINT.project}(
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
    let probePos = probe * RC_PROBE_SIZE + cell;

    // Compute our SH probe (gi1.comp:1462-1481): the cell's radiance times the cell's SH basis,
    // in world space through the probe's frame (the cell's mean basis, register D34).
    if (probeIndex < probeCount) {
        let normal = rcLoadNormal(seed);
        let radiance = textureLoad(rcProbeBuffer, probePos, 0).xyz * RC_SH_CELL_SOLID_ANGLE;

        let directionSH = rcProbeCellSH(cell, normal);

        for (var j = 0u; j < RC_SH_COEFFICIENTS; j++) {
            rcProbeSHValues[RC_SH_COEFFICIENTS * localId + j] = vec4f(directionSH[j] * radiance, 1.0);
        }
    }
    workgroupBarrier();

    if (probeIndex >= probeCount) {
        return; // out of bounds
    }

    // Resolve the probe and write out the SH coefficients (gi1.comp:1488-1511): the first nine
    // threads each sum one coefficient over the 64 cells. The offset is Capsaicin's, 0 with a
    // probe per 64-thread workgroup.
    let shIndex = cell.x + cell.y * RC_PROBE_SIZE;

    if (shIndex < RC_SH_COEFFICIENTS) {
        var irradianceSH = vec4f(0.0);

        let offset = (vec2u(localId & 7u, localId >> 3u) / RC_PROBE_SIZE) * RC_PROBE_SIZE;

        for (var y = 0u; y < RC_PROBE_SIZE; y++) {
            for (var x = 0u; x < RC_PROBE_SIZE; x++) {
                let pos = vec2u(x, y) + offset;
                let index = pos.x + (pos.y << 3u);

                irradianceSH += rcProbeSHValues[RC_SH_COEFFICIENTS * index + shIndex];
            }
        }

        textureStore(rcProbeSHBuffer, rcProbeSHTexel(probe, shIndex), vec4u(rcPackSHColor(irradianceSH), 0u, 0u));
    }
}
`;

/**
 * The lookups `InterpolateScreenProbes` and `FilterScreenProbes` share, over
 * `rcProbeMask` (the probe mask with its mip chain), `RcFrame`,
 * `RcScreenProbesConstants` and the host's depth.
 */
const PROBE_LOOKUP_WGSL = /* wgsl */ `
// The probe tiles across and down: the probe mask's level 0.
fn rcProbeMaskDimensions() -> vec2u {
    return (rcFrame.bufferDimensions + RC_PROBE_SIZE - 1u) / RC_PROBE_SIZE;
}

// ScreenProbes_FindClosestProbe(pos) (screen_probes.hlsl:110-131): the probe of pos's tile,
// else the first probe found going up the mask's mips.
fn rcFindClosestProbe(position: vec2u) -> u32 {
    var dims = rcProbeMaskDimensions();
    var pos = min(position / RC_PROBE_SIZE, dims - 1u);

    for (var i = 0u; i < rcScreenProbes.probeMaskMipCount; i++) {
        let probe = textureLoad(rcProbeMask, pos, i32(i)).x;

        if (probe != RC_INVALID_ID) {
            return probe; // found a close-by probe :)
        }

        dims = max(dims >> vec2u(1u), vec2u(1u));

        pos = min(pos >> vec2u(1u), dims - 1u);
    }

    return RC_INVALID_ID;
}

// ScreenProbes_FindClosestProbe(pos, offset) (screen_probes.hlsl:137-166): the same with an
// offset applied at every level, to find the neighbour probe in a given direction.
fn rcFindClosestProbeWithOffset(position: vec2u, offset: vec2i) -> u32 {
    var dims = rcProbeMaskDimensions();
    var pos = min(position / RC_PROBE_SIZE, dims - 1u);

    for (var i = 0u; i < rcScreenProbes.probeMaskMipCount; i++) {
        let loc = vec2i(pos) + offset;

        if (any(loc < vec2i(0)) || any(loc >= vec2i(dims))) {
            break; // out of bounds
        }

        let probe = textureLoad(rcProbeMask, vec2u(loc), i32(i)).x;

        if (probe != RC_INVALID_ID) {
            return probe; // found a close-by probe :)
        }

        dims = max(dims >> vec2u(1u), vec2u(1u));

        pos = min(pos >> vec2u(1u), dims - 1u);
    }

    return RC_INVALID_ID;
}

// toLinearDepth(depth, g_NearFar) (transform.hlsl:144-148) for a reconstructed point: its
// distance along the camera's view axis (RcScreenProbesConstants).
fn rcLinearDepth(worldPosition: vec3f) -> f32 {
    return dot(worldPosition - rcFrame.eye, rcScreenProbes.viewForward);
}

// The world position the host's depth gives at a pixel (gi1.comp:1533-1535).
fn rcWorldPositionAt(pixel: vec2u) -> vec3f {
    let uv = (vec2f(pixel) + 0.5) / vec2f(rcFrame.bufferDimensions);
    return rcTransformPointProjection(uv, rcLoadDepth(pixel));
}
`;

// ---- InterpolateScreenProbes --------------------------------------------------------------------

export const INTERPOLATE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/InterpolateScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        CONSTANTS_ENTRY,
        { binding: 2, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 3, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 4, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } },
        { binding: 5, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } },
        { binding: 6, visibility: COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
        { binding: 7, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 8, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 9, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 10, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 11, visibility: COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
    ],
};

export const INTERPOLATE_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${SPHERICAL_HARMONICS_WGSL}
${FRAME_WGSL}
${SCREEN_PROBES_CONSTANTS_WGSL}
${screenDepthWgsl(2)}
${screenNormalWgsl(3)}
// g_ScreenProbes_ProbeMask: the probe mask with its whole mip chain (FilterProbeMask's).
@group(1) @binding(4) var rcProbeMask: texture_2d<u32>;
// g_ScreenProbes_ProbeSHBuffer, as a texture (register D26).
@group(1) @binding(5) var rcProbeSHBuffer: texture_2d<u32>;
// g_GIDenoiser_ColorBuffer: irradiance in rgb, the denoiser hint in alpha.
@group(1) @binding(6) var rcColorBuffer: texture_storage_2d<rgba16float, write>;
// g_OcclusionAndBentNormalBuffer and g_NearFieldGlobalIlluminationBuffer: the platform's SSGI
// (gi/ssgi, RC4.2), the bent normal signed in xyz and the visibility in w; 1 x 1 placeholders
// when SSGI is off (hasOcclusion 0).
@group(1) @binding(7) var rcOcclusionAndBentNormal: texture_2d<f32>;
@group(1) @binding(8) var rcNearFieldGlobalIllumination: texture_2d<f32>;
// FrameScreen.lighting and FrameScreen.albedo, and the lit image they make with this frame's
// irradiance: Capsaicin's PrevCombinedIllumination for next frame's SSGI (register D51).
@group(1) @binding(9) var rcLighting: texture_2d<f32>;
@group(1) @binding(10) var rcAlbedo: texture_2d<f32>;
@group(1) @binding(11) var rcCombinedIllumination: texture_storage_2d<rgba16float, write>;

${shIrradianceWgsl('rcCalculateSHIrradianceBentCone', 'probeSH')}

${PROBE_LOOKUP_WGSL}
// The pixel's result: irradiance and the denoiser's hint, and with writeCombined the lit image
// next frame's SSGI reads as its lighting (Capsaicin's combine pass writes it from the resolved GI,
// combine.comp:74): the host's own light plus albedo / pi x E, nothing for a sky pixel.
fn rcStoreResult(did: vec2u, irradiance: vec3f, denoiserHint: f32, isSky: bool) {
    textureStore(rcColorBuffer, did, vec4f(irradiance, denoiserHint));
    if (rcScreenProbes.writeCombined != 0u) {
        var lit = vec3f(0.0);
        if (!isSky) {
            lit = textureLoad(rcLighting, did, 0).rgb + textureLoad(rcAlbedo, did, 0).rgb * irradiance * RC_INV_PI;
        }
        textureStore(rcCombinedIllumination, did, vec4f(lit, 1.0));
    }
}

// InterpolateScreenProbes (gi1.comp:1514-1640), the diffuse part. Capsaicin's GI-1.1 half
// (the glossy reflections' queue and buffers, gi1.comp:1528-1530, 1640 on) is left out with
// the reflections (register D8).
@compute @workgroup_size(8, 8)
fn ${SCREEN_IRRADIANCE_ENTRY_POINT.interpolate}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.xy;
    if (any(did >= rcFrame.bufferDimensions)) {
        return; // outside the screen: Capsaicin writes black past the buffer, which has no texels here
    }
    // A sky pixel is one at the far plane (register D17; Capsaicin: a zero geometry normal).
    if (rcIsSkyPixel(did)) {
        rcStoreResult(did, vec3f(0.0), 1.0, true);
        return; // discard sky pixels
    }

    var normal = rcLoadNormal(did);
    let depth = rcLoadDepth(did);

    var pos = did;
    let worldPos = rcWorldPositionAt(did);
    let cellSize = distance(rcFrame.eye, worldPos) * rcScreenProbes.cellSize;

    // Capsaicin draws the jitter from its blue-noise sampler (MakeBlueNoiseSampler(did,
    // g_FrameIndex, 1), gi1.comp:1540-1543); ours from the PCG (register D27).
    var random = rcMakeRandom(did.x + did.y * rcFrame.bufferDimensions.x, rcFrame.frameIndex);
    let s = rcRand2(&random);
    let jitter = vec2i((2.0 * s - 1.0) * f32(RC_PROBE_SPAWN_TILE_SIZE)) * i32(rcScreenProbes.interpolationJitter);
    let newPos = vec2u(clamp(vec2i(did) + jitter, vec2i(0), vec2i(rcFrame.bufferDimensions) - 1));
    let newWorldPos = rcWorldPositionAt(newPos);

    if (abs(dot(newWorldPos - worldPos, normal)) < 0.5 * cellSize) {
        pos = newPos; // only apply the jitter if the new position lies in the original pixel plane
    }

    var probes: vec4u; // locate nearby probes for interpolation

    probes.x = rcFindClosestProbe(pos);

    if (probes.x == RC_INVALID_ID) {
        rcStoreResult(did, vec3f(0.0), 1.0, false);
        return; // couldn't find any nearby probe...
    }

    let seed = rcUnpackSeed(probes.x);
    let offset = vec2i(select(1, -1, pos.x < seed.x), select(1, -1, pos.y < seed.y));

    probes.y = rcFindClosestProbeWithOffset(pos, vec2i(offset.x, 0));
    probes.z = rcFindClosestProbeWithOffset(pos, vec2i(0, offset.y));
    probes.w = rcFindClosestProbeWithOffset(pos, offset);

    if (probes.y == probes.x) {
        probes.y = RC_INVALID_ID;
    }
    if (probes.z == probes.y || probes.z == probes.x) {
        probes.z = RC_INVALID_ID;
    }
    if (probes.w == probes.z || probes.w == probes.y || probes.w == probes.x) {
        probes.w = RC_INVALID_ID;
    }

    var w = vec4f(0.0); // calculate per-probe blending weights
    let linearDepth = rcLinearDepth(worldPos);

    for (var i = 0u; i < 4u; i++) {
        if (probes[i] != RC_INVALID_ID) {
            let probeSeed = rcUnpackSeed(probes[i]);
            let probePos = rcWorldPositionAt(probeSeed);

            if (abs(dot(probePos - worldPos, normal)) > cellSize) {
                w[i] = 0.0; // prevent probes ahead of pixel plane to leak radiance into occluded background
            } else {
                w[i] = saturate(1.0 - abs(rcLinearDepth(probePos) - linearDepth) / linearDepth);
                w[i] *= max(dot(normal, rcLoadNormal(probeSeed)), 0.0);
                w[i] = pow(w[i], 8.0); // make it steep
            }
        }
    }

    var useBackup = false;

    if (dot(w, w) == 0.0) {
        w = vec4f(1.0, select(0.0, 1.0, probes.y != RC_INVALID_ID), select(0.0, 1.0, probes.z != RC_INVALID_ID), select(0.0, 1.0, probes.w != RC_INVALID_ID));

        useBackup = true; // for 'relaxed' interpolation in failure cases
    }

    w /= w.x + w.y + w.z + w.w; // weights must sum up to 1

    // Capsaicin's '#ifndef HAS_OCCLUSION' branch (gi1.comp:1617-1620), what it runs without its
    // SSGI technique: no ambient occlusion (ao 1, so the cone is the whole hemisphere), no
    // near-field light to start from, and the shading normal. Our one normal is
    // FrameScreen.normal, the host's shading normal (register D17).
    var ao = 1.0;
    var irradiance = vec3f(0.0);
    normal = rcLoadNormal(did);
    // The HAS_OCCLUSION branch (gi1.comp:1621-1627): SSGI's visibility narrows each probe's SH
    // lookup to the cone around the bent normal, and its near-field light, irradiance already
    // (register D52), is what the probes' sum starts from.
    if (rcScreenProbes.hasOcclusion != 0u) {
        let aoAndBentNormal = textureLoad(rcOcclusionAndBentNormal, did, 0);
        ao = aoAndBentNormal.w;
        irradiance = textureLoad(rcNearFieldGlobalIllumination, did, 0).rgb;
        normal = normalize(aoAndBentNormal.xyz);
    }

    for (var j = 0u; j < 4u; j++) {
        // A missing neighbour has weight 0; its 'probe' is a tile past the screen, so it is
        // skipped rather than read (Capsaicin reads it and multiplies by 0).
        if (w[j] > 0.0) {
            let probe = rcUnpackSeed(probes[j]) / RC_PROBE_SIZE;

            irradiance += w[j] * rcCalculateSHIrradianceBentCone(normal, ao, probe, rcProbeSHBuffer);
        }
    }

    let denoiserHint = select(1.0, 0.0, useBackup); // hint to the denoiser that we should ideally not keep this sample...

    rcStoreResult(did, irradiance, denoiserHint, false);
}
`;

// ---- FilterScreenProbes -------------------------------------------------------------------------

/** `kGI1_AngleThreshold` (screen_probes.hlsl:30): cos(0.02 pi), a neighbour's hit seen from this probe within 3.6 degrees of the cell's direction. */
export const FILTER_ANGLE_THRESHOLD = Math.cos(2e-2 * Math.PI);
/** `kRadius` (gi1.comp:1393): neighbour probes on each side, along the pass's direction. */
export const FILTER_RADIUS = 3;
/** Bytes of `RcBlurDirection`. */
export const BLUR_DIRECTION_BYTES = 16;

export const FILTER_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/FilterScreenProbes group 1',
    entries: [
        UNIFORM_ENTRY,
        CONSTANTS_ENTRY,
        { binding: 2, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 3, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 4, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } },
        { binding: 5, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 7, visibility: COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '2d' } },
        { binding: 8, visibility: COMPUTE, buffer: { type: 'uniform', minBindingSize: BLUR_DIRECTION_BYTES } },
    ],
};

export const FILTER_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${SCREEN_PROBES_CONSTANTS_WGSL}
${screenDepthWgsl(2)}
${screenNormalWgsl(3)}
${PROBE_SPAWN_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
${HISTORY_WGSL}
// g_ScreenProbes_ProbeMask, with its mip chain (FilterProbeMask runs first, gi1.cpp:2561-2585).
@group(1) @binding(4) var rcProbeMask: texture_2d<u32>;
@group(1) @binding(5) var<storage, read> rcProbeSpawn: RcProbeSpawn;
// g_ScreenProbes_PreviousProbeBuffer: the pass's input (the blended atlas, then the first pass's output).
@group(1) @binding(6) var rcPreviousProbeBuffer: texture_2d<f32>;
// g_ScreenProbes_ProbeBuffer: the pass's output.
@group(1) @binding(7) var rcProbeBuffer: texture_storage_2d<rgba16float, write>;
// g_BlurDirection: (1, 0) for the first pass, (0, 1) for the second (gi1.cpp:2601-2603).
struct RcBlurDirection {
    direction: vec2i,
}
@group(1) @binding(8) var<uniform> rcBlurDirection: RcBlurDirection;

${PROBE_LOOKUP_WGSL}

const RC_FILTER_ANGLE_THRESHOLD: f32 = ${FILTER_ANGLE_THRESHOLD};
const RC_FILTER_RADIUS: i32 = ${FILTER_RADIUS};

// GIDenoiser_RemoveNaNs (gi_denoiser.hlsl:75-82) on a neighbour's radiance: x / (1 + x), saturate,
// then back, which sends NaN, infinity and negatives to 0. Ours sends exactly those to 0 and keeps
// every finite value: Capsaicin's form also caps at 1e4 - 1 (the max(1 - x, 1e-4)), on the
// neighbours only (register D49). WGSL may assume no NaN in comparisons, so the bits are checked.
fn rcRemoveNaNs(color: vec3f) -> vec3f {
    let finite = (bitcast<vec3u>(color) & vec3u(0x7f800000u)) != vec3u(0x7f800000u);
    return select(vec3f(0.0), max(color, vec3f(0.0)), finite);
}

// FilterScreenProbes (gi1.comp:1356-1442): one pass of the separable probe filter. Each cell of
// each probe in this frame's list averages the matching cell of up to 2 x RC_FILTER_RADIUS
// neighbour probes along the pass's direction (found through the mask, nearest first, alternating
// sides), each taken only if its plane is within a cell of this probe's, its hemisphere holds the
// cell's direction, and its hit, seen from this probe, lies within 3.6 degrees of that direction;
// weighted by how close its linear depth is, to the 8th power. Dispatched over every possible ray
// (gi1.cpp:2592-2593).
@compute @workgroup_size(64)
fn ${SCREEN_IRRADIANCE_ENTRY_POINT.filter}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localId: u32,
) {
    let did = rcWorkgroupIndex(workgroup, workgroups) * 64u + localId;
    let probeCount = rcProbeSpawn.count;

    let cellAndProbeIndex = rcGetCellAndProbeIndex(did);
    let cellIndex = cellAndProbeIndex.x;
    let probeIndex = cellAndProbeIndex.y;

    if (probeIndex >= probeCount) {
        return; // out of bounds
    }

    let cell = vec2u(cellIndex % RC_PROBE_SIZE, cellIndex / RC_PROBE_SIZE);
    let seed = rcUnpackSeed(rcProbeSpawn.seeds[probeIndex]);
    let probe = seed / RC_PROBE_SIZE;
    let pos = probe * RC_PROBE_SIZE + cell;

    let normal = rcLoadNormal(seed);
    let worldPos = rcWorldPositionAt(seed);
    let linearDepth = rcLinearDepth(worldPos);
    let cellSize = distance(rcFrame.eye, worldPos) * rcScreenProbes.cellSize;

    var radiance = textureLoad(rcPreviousProbeBuffer, pos, 0);
    // A backup fill (D47) has no measured hit distance: Capsaicin's fill holds MAX_HIT_DISTANCE, so
    // the neighbours' own distances decide its reprojection test, and it stays a fill (register D50).
    let isFill = radiance.w < 0.0;
    var hitDistance = select(radiance.w, RC_LARGEST_HALF, isFill);
    radiance.w = hitDistance;
    var totalWeight = 1.0;

    let direction = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(cell) + 0.5) / f32(RC_PROBE_SIZE)), normal);

    for (var i = 0; i < 2 * RC_FILTER_RADIUS; i++) {
        let stepOffset = (((i & 1) << 1u) - 1) * ((i >> 1u) + 1); // -1, +1, -2, +2, -3, +3
        let probeMask = rcFindClosestProbeWithOffset(seed, stepOffset * rcBlurDirection.direction);

        if (probeMask == RC_INVALID_ID) {
            continue; // invalid probe
        }

        let probeSeed = rcUnpackSeed(probeMask);
        let probeNormal = rcLoadNormal(probeSeed);
        let probeWorld = rcWorldPositionAt(probeSeed);

        if (abs(dot(probeWorld - worldPos, normal)) > cellSize || dot(direction, probeNormal) < 0.0) {
            continue; // oriented hemispheres do not overlap
        }

        // The neighbour's cell for this direction, kept in the probe (D45).
        let probeCell = min(vec2u(rcMapToHemiOctahedronInverse(rcToProbeFrame(direction, probeNormal)) * f32(RC_PROBE_SIZE)), vec2u(RC_PROBE_SIZE - 1u));
        let probePos = (probeSeed / RC_PROBE_SIZE) * RC_PROBE_SIZE + probeCell;
        let probeRadiance = textureLoad(rcPreviousProbeBuffer, probePos, 0);

        // A neighbour's backup fill is no measurement: skipped, as the reprojection and the taps skip it (D47, register D50).
        if (probeRadiance.w < 0.0) {
            continue;
        }

        let probeDirection = normalize(rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(probeCell) + 0.5) / f32(RC_PROBE_SIZE)), probeNormal));

        let probeHitDistance = min(probeRadiance.w, hitDistance);
        let hitPoint = probeWorld + probeDirection * probeHitDistance;
        let reprojectedDir = normalize(hitPoint - worldPos);

        if (dot(direction, reprojectedDir) < RC_FILTER_ANGLE_THRESHOLD) {
            continue; // skip probes with high angle error after hit reprojection
        }

        let weight = pow(saturate(1.0 - abs(rcLinearDepth(probeWorld) - linearDepth) / linearDepth), 8.0);

        radiance += weight * vec4f(rcRemoveNaNs(probeRadiance.xyz), probeHitDistance);
        totalWeight += weight;

        if (!isFill) {
            hitDistance = radiance.w / totalWeight;
        }
    }

    var filtered = radiance / totalWeight;
    if (isFill) {
        filtered.w = RC_FILL_HIT_DISTANCE;
    }
    // Stored to the nearest half (D48), the hit distance within the half range (D19).
    textureStore(rcProbeBuffer, pos, rcNearestHalf4(vec4f(filtered.xyz, min(filtered.w, RC_LARGEST_HALF))));
}
`;
