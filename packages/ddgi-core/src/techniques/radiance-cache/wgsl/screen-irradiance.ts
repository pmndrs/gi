/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.comp
 * (FilterProbeMask, ProjectScreenProbes, InterpolateScreenProbes) and
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
 * From the blended probe atlas to irradiance per pixel: the kernels
 * Capsaicin runs after `BlendScreenProbes` (`GI1::render`,
 * gi1.cpp:2561-2636), each named after its entry point:
 *
 * 1. `FilterProbeMask` (gi1.comp:248-284; RC2.2), once per mip level of the
 *    probe mask: each texel keeps the first probe of its 2 × 2 children, so
 *    any pixel can find a probe nearby in a few loads (`FindClosestProbe`).
 * 2. `FilterScreenProbes` (gi1.comp:1356-1443; RC4.2), twice: a separable
 *    blur of each probe cell with the same direction's cell of up to three
 *    neighbour probes either side, first across the screen, then down it,
 *    accepting a neighbour only on the probe's plane, with an overlapping
 *    hemisphere and a hit point that reprojects within 3.6 degrees, weighted
 *    by depth.
 * 3. `ProjectScreenProbes` (gi1.comp:1445-1512; RC2.2): each probe's 64
 *    cells to L2 SH (nine coefficients), one probe per workgroup.
 * 4. `InterpolateScreenProbes` (gi1.comp:1514-1640; RC2.2, SSGI's inputs
 *    RC4.2): per pixel, up to four nearby probes weighted by plane distance,
 *    depth and normal, and their SH evaluated about SSGI's bent normal through
 *    a cone narrowed by its occlusion, added to SSGI's near-field light:
 *    irradiance E, which `ResolveGI1` (gi1.frag:76-150) turns into reflected
 *    light with `albedo / pi`, as three's composite does.
 *
 * The GI denoiser that reads the interpolated image is RC4.3; until then the
 * shading module reads this kernel's output directly.
 *
 * Differences from Capsaicin, in the register (docs/design/radiance-cache.md):
 * the projection weight is a cell's solid angle (D25), the SH live in a
 * texture (D26), the interpolation's jitter is white noise (D27), SH
 * coefficients are clamped to the half range (D28), linear depth comes from
 * the rebuilt point (D29), a missing neighbour probe is skipped rather than
 * read and weighted 0 (D38), a neighbour's cell is kept inside its probe
 * (D51), and one normal, the host's, does the geometry normal's work too
 * (D17).
 */

import { CAPSAICIN_WGSL } from './capsaicin.ts';
import { FRAME_WGSL, PROBE_SPAWN_STRUCT_WGSL, screenDepthWgsl, screenNormalWgsl, UNIFORM_ENTRY, WORKGROUP_INDEX_WGSL } from './screen-probes.ts';
import { shIrradianceWgsl, SPHERICAL_HARMONICS_WGSL } from './spherical-harmonics.ts';

const COMPUTE = 0x4;

/** Entry points: Capsaicin's kernel names. */
export const SCREEN_IRRADIANCE_ENTRY_POINT = {
    filterProbeMask: 'FilterProbeMask',
    filterProbes: 'FilterScreenProbes',
    project: 'ProjectScreenProbes',
    interpolate: 'InterpolateScreenProbes',
} as const;

/** Workgroup sizes, as Capsaicin's `[numthreads]`. */
export const SCREEN_IRRADIANCE_WORKGROUP = {
    /** FilterProbeMask, InterpolateScreenProbes (gi1.comp:248, 1514): 8 × 8. */
    tile: 8,
    /** ProjectScreenProbes, FilterScreenProbes (gi1.comp:1445, 1356): one probe's 64 cells per workgroup. */
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
} as const;

/**
 * Capsaicin's `ScreenProbesConstants` (gi1_shared.h:38-46) beyond what
 * `RcFrame` has, and `g_NearFar`'s use: the interpolation's linear depth.
 *
 * Capsaicin linearises its reversed-Z depth with the camera planes
 * (`toLinearDepth`, transform.hlsl:144-148): the distance along the view
 * axis. We take the same distance from the reconstructed position, the dot
 * product with the camera's forward axis, which holds for any depth
 * convention the host draws with (register D29; the two agree, by a unit
 * test in screen-irradiance.test.ts).
 */
const SCREEN_PROBES_CONSTANTS_WGSL = /* wgsl */ `
struct RcScreenProbesConstants {
    viewForward: vec3f,         // the camera's forward axis, world space: linear depth is dot(p - eye, viewForward)
    cellSize: f32,              // cell_size: tan(fovY * probe_size * max(1 / h, h / w^2)) (gi1.cpp:1839-1843), per unit distance
    probeMaskMipCount: u32,     // probe_mask_mip_count (gi1.cpp:1846)
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


/**
 * What `FilterScreenProbes` and `InterpolateScreenProbes` share: finding
 * probes in the mask's mips, linear depth and positions from the host's
 * depth. Needs `FRAME_WGSL`, `SCREEN_PROBES_CONSTANTS_WGSL`,
 * `screenDepthWgsl` and the mask bound as `rcProbeMask` in the same module.
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
// distance along the camera's view axis (RcScreenProbesConstants; register D29).
fn rcLinearDepth(worldPosition: vec3f) -> f32 {
    return dot(worldPosition - rcFrame.eye, rcScreenProbes.viewForward);
}

// The world position the host's depth gives at a pixel (gi1.comp:1533-1535).
fn rcWorldPositionAt(pixel: vec2u) -> vec3f {
    let uv = (vec2f(pixel) + 0.5) / vec2f(rcFrame.bufferDimensions);
    return rcTransformPointProjection(uv, rcLoadDepth(pixel));
}
`;

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

// ---- FilterScreenProbes -------------------------------------------------------------------------

/** Bytes of `RcBlurDirection`: `g_BlurDirection` (an `int2`), padded to 16. */
export const BLUR_DIRECTION_BYTES = 16;

/** `g_BlurDirection` of the two passes (gi1.cpp:2597-2603): across the screen, then down it. */
export const BLUR_DIRECTIONS: readonly (readonly [number, number])[] = [
    [1, 0],
    [0, 1],
];

/** `kRadius` (gi1.comp:1393): neighbour probes either side in each pass. */
export const FILTER_RADIUS = 3;

/** `kGI1_AngleThreshold` (screen_probes.hlsl:30): cos(2e-2 pi), a reprojected hit within 3.6 degrees. */
export const FILTER_ANGLE_THRESHOLD = Math.cos(2e-2 * Math.PI);

export const FILTER_PROBES_GROUP: GPUBindGroupLayoutDescriptor = {
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

export const FILTER_PROBES_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${SCREEN_PROBES_CONSTANTS_WGSL}
${screenDepthWgsl(2)}
${screenNormalWgsl(3)}
${PROBE_SPAWN_STRUCT_WGSL}
${WORKGROUP_INDEX_WGSL}
// g_ScreenProbes_ProbeMask: the probe mask with its whole mip chain (FilterProbeMask's).
@group(1) @binding(4) var rcProbeMask: texture_2d<u32>;
@group(1) @binding(5) var<storage, read> rcProbeSpawn: RcProbeSpawn;
// g_ScreenProbes_PreviousProbeBuffer as the pass binds it (gi1.cpp:2608-2610): the atlas it reads.
@group(1) @binding(6) var rcInputProbeBuffer: texture_2d<f32>;
// g_ScreenProbes_ProbeBuffer as the pass binds it (gi1.cpp:2605-2607): the atlas it writes.
@group(1) @binding(7) var rcOutputProbeBuffer: texture_storage_2d<rgba16float, write>;

struct RcBlurDirection {
    direction: vec2i, // g_BlurDirection: (1, 0) the first pass, (0, 1) the second
}
@group(1) @binding(8) var<uniform> rcBlurDirection: RcBlurDirection;

${PROBE_LOOKUP_WGSL}

const RC_FILTER_RADIUS: i32 = ${FILTER_RADIUS};
const RC_ANGLE_THRESHOLD: f32 = ${FILTER_ANGLE_THRESHOLD};

// GIDenoiser_RemoveNaNs (gi_denoiser.hlsl:75-82): through x / (1 + x) and back, clamped, so a
// huge value comes back near 1e4. D3D's saturate takes a NaN to 0; WGSL leaves a NaN's result
// indeterminate (15.7.2), so here it only bounds large values.
fn rcRemoveNaNs(color: vec3f) -> vec3f {
    var c = color / (1.0 + color);
    c = saturate(c);
    c /= max(1.0 - c, vec3f(1e-4));
    return c;
}

// FilterScreenProbes (gi1.comp:1356-1443): a thread per probe cell, over every possible ray
// (gi1.cpp:2593-2595), run once per blur direction.
@compute @workgroup_size(64)
fn ${SCREEN_IRRADIANCE_ENTRY_POINT.filterProbes}(@builtin(workgroup_id) workgroup: vec3u, @builtin(num_workgroups) workgroups: vec3u, @builtin(local_invocation_index) localId: u32) {
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

    // Capsaicin's geometry normal; ours is FrameScreen.normal (register D17).
    let normal = rcLoadNormal(seed);
    let worldPos = rcWorldPositionAt(seed);
    let cellSize = distance(rcFrame.eye, worldPos) * rcScreenProbes.cellSize;
    let linearDepth = rcLinearDepth(worldPos);

    var radiance = textureLoad(rcInputProbeBuffer, pos, 0);
    // The cell's centre direction, in world space through the probe's frame (GetOrthoVectors).
    let direction = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(cell) + 0.5) / f32(RC_PROBE_SIZE)), normal);
    var hitDistance = radiance.w;
    var totalWeight = 1.0;

    // kSize = kRadius << 1 neighbours, nearest first and alternating sides: steps -1, 1, -2, 2, -3, 3.
    for (var i = 0; i < (RC_FILTER_RADIUS << 1u); i++) {
        let blurStep = (((i & 1) << 1u) - 1) * ((i >> 1u) + 1);
        let probeMask = rcFindClosestProbeWithOffset(seed, blurStep * rcBlurDirection.direction);

        if (probeMask == RC_INVALID_ID) {
            continue; // invalid probe
        }

        let probeSeed = rcUnpackSeed(probeMask);
        let probeNormal = rcLoadNormal(probeSeed);
        let probeWorld = rcWorldPositionAt(probeSeed);

        if (abs(dot(probeWorld - worldPos, normal)) > cellSize || dot(direction, probeNormal) < 0.0) {
            continue; // oriented hemispheres do not overlap
        }

        // The neighbour's cell that holds this direction; kept inside its probe (register D51).
        let probeCell = min(vec2u(rcMapToHemiOctahedronInverse(rcToProbeFrame(direction, probeNormal)) * f32(RC_PROBE_SIZE)), vec2u(RC_PROBE_SIZE - 1u));
        let probeTile = probeSeed / RC_PROBE_SIZE;
        let probePos = probeTile * RC_PROBE_SIZE + probeCell;

        let probeDirection = normalize(rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(probeCell) + 0.5) / f32(RC_PROBE_SIZE)), probeNormal));

        let neighbour = textureLoad(rcInputProbeBuffer, probePos, 0);
        let probeHitDistance = min(neighbour.w, hitDistance);
        let hitPoint = probeWorld + probeDirection * probeHitDistance;
        let reprojectedDir = normalize(hitPoint - worldPos);

        if (dot(direction, reprojectedDir) < RC_ANGLE_THRESHOLD) {
            continue; // skip probes with high angle error after hit reprojection
        }

        let weight = pow(saturate(1.0 - abs(rcLinearDepth(probeWorld) - linearDepth) / linearDepth), 8.0);

        radiance += weight * vec4f(rcRemoveNaNs(neighbour.xyz), probeHitDistance);
        totalWeight += weight;

        hitDistance = radiance.w / totalWeight;
    }

    textureStore(rcOutputProbeBuffer, pos, radiance / totalWeight);
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

// How a cell counts in the projection (register D25). The coefficients are meant to be the SH of
// the radiance, c_i = the integral of L Y_i over the hemisphere: here a sum over the 64
// equal-area cells of the cell's radiance, its solid angle 2 pi / 64 and the basis at the cell's
// centre, as Capsaicin takes it (gi1.comp:1466-1474). Capsaicin's weight is 1 / probe_size, which
// is 4 / pi times the solid angle; nothing downstream divides that back out (the GI denoiser and
// ResolveGI1 keep the scale), so a constant sky of radiance L reads 1.30 pi L there. With the solid
// angle it reads 1.021 pi L: the 2.1% left is the centre point leaning towards the normal, which
// Capsaicin has too (screen-irradiance.test.ts; the design doc's levers list a cell-mean basis).
const RC_SH_CELL_SOLID_ANGLE: f32 = ${SH_CELL_SOLID_ANGLE};

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

    // Compute our SH probe (gi1.comp:1462-1481): the cell's radiance times the SH basis at the
    // cell's centre, in world space through the probe's frame (the weight is register D25).
    if (probeIndex < probeCount) {
        let normal = rcLoadNormal(seed);
        let radiance = textureLoad(rcProbeBuffer, probePos, 0).xyz * RC_SH_CELL_SOLID_ANGLE;
        let direction = rcFromProbeFrame(rcMapToHemiOctahedron((vec2f(cell) + 0.5) / f32(RC_PROBE_SIZE)), normal);

        let directionSH = rcSHGetCoefficients(direction);

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
// SSGI's outputs (wgsl/ssgi.ts): g_OcclusionAndBentNormalBuffer (rgb 0.5 n + 0.5, a the occlusion)
// and g_NearFieldGlobalIlluminationBuffer (rgb).
@group(1) @binding(7) var rcOcclusionAndBentNormalBuffer: texture_2d<f32>;
@group(1) @binding(8) var rcNearFieldGlobalIlluminationBuffer: texture_2d<f32>;

${shIrradianceWgsl('rcCalculateSHIrradianceBentCone', 'probeSH')}

${PROBE_LOOKUP_WGSL}

// InterpolateScreenProbes (gi1.comp:1514-1640), the diffuse part, with SSGI (HAS_OCCLUSION). Capsaicin's GI-1.1 half
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
        textureStore(rcColorBuffer, did, vec4f(0.0, 0.0, 0.0, 1.0));
        return; // discard sky pixels
    }

    // Capsaicin's geometry normal here (gi1.comp:1518, 1598); ours is FrameScreen.normal, the host's
    // shading normal, for the plane tests and weights below as well as the SH (register D17).
    var normal = rcLoadNormal(did);

    var pos = did;
    let worldPos = rcWorldPositionAt(did);
    let cellSize = distance(rcFrame.eye, worldPos) * rcScreenProbes.cellSize;

    // Capsaicin draws the jitter from its blue-noise sampler (MakeBlueNoiseSampler(did,
    // g_FrameIndex, 1), gi1.comp:1540-1543); ours from the PCG (register D27).
    var random = rcMakeRandom(did.x + did.y * rcFrame.bufferDimensions.x, rcFrame.frameIndex);
    let s = rcRand2(&random);
    let jitter = vec2i((2.0 * s - 1.0) * f32(RC_PROBE_SPAWN_TILE_SIZE));
    let newPos = vec2u(clamp(vec2i(did) + jitter, vec2i(0), vec2i(rcFrame.bufferDimensions) - 1));
    let newWorldPos = rcWorldPositionAt(newPos);

    if (abs(dot(newWorldPos - worldPos, normal)) < 0.5 * cellSize) {
        pos = newPos; // only apply the jitter if the new position lies in the original pixel plane
    }

    var probes: vec4u; // locate nearby probes for interpolation

    probes.x = rcFindClosestProbe(pos);

    if (probes.x == RC_INVALID_ID) {
        textureStore(rcColorBuffer, did, vec4f(0.0, 0.0, 0.0, 1.0));
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

    // Capsaicin's HAS_OCCLUSION branch (gi1.comp:1621-1626), what it runs with its SSGI technique
    // (RC4.2): the probes' SH evaluated about SSGI's bent normal, through a cone its occlusion
    // narrows, added to SSGI's near-field light.
    let aoAndBentNormal = textureLoad(rcOcclusionAndBentNormalBuffer, did, 0);

    let ao = aoAndBentNormal.w;
    var irradiance = textureLoad(rcNearFieldGlobalIlluminationBuffer, did, 0).rgb;
    normal = normalize(2.0 * aoAndBentNormal.xyz - 1.0);

    for (var j = 0u; j < 4u; j++) {
        // A missing neighbour has weight 0. Capsaicin reads it anyway, a tile past the buffer that
        // D3D12 reads as 0, and multiplies by 0 (gi1.comp:1629-1634). WGSL leaves an out-of-bounds
        // textureLoad to the implementation (a texel inside, or 0; WGSL 17.7.4), so it is skipped
        // rather than read (register D38).
        if (w[j] > 0.0) {
            let probe = rcUnpackSeed(probes[j]) / RC_PROBE_SIZE;

            irradiance += w[j] * rcCalculateSHIrradianceBentCone(normal, ao, probe, rcProbeSHBuffer);
        }
    }

    let denoiserHint = select(1.0, 0.0, useBackup); // hint to the denoiser that we should ideally not keep this sample...

    textureStore(rcColorBuffer, did, vec4f(irradiance, denoiserHint));
}
`;
