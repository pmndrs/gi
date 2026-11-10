/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/ssgi/ssgi.comp
 * (Main, IntersectDirPlaneOneSided, RotFromToMatrix, IntegrateHalfArc, IntegrateBentNormal)
 * and ssgi_debug.comp's views (drawn in wgsl/debug.ts).
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
 *
 * Capsaicin's IntersectDirPlaneOneSided and the horizon truncation come from
 * h3r2tic's rtoy-samples (assets/shaders/ssgi/ssgi.glsl), and RotFromToMatrix
 * from the Open Asset Import Library (include/assimp/matrix3x3.inl), as
 * Capsaicin's comments say.
 */

/**
 * Capsaicin's SSGI technique (RC4.2): screen-space horizons around each
 * pixel, GTAO-style, giving three things `InterpolateScreenProbes` reads
 * (`HAS_OCCLUSION`, gi1.comp:1621-1626):
 *
 * - the **ambient occlusion**: the cosine-weighted share of the hemisphere
 *   no nearby depth sample blocks (1 = open), which narrows the cone the
 *   probes' SH are evaluated over;
 * - the **bent normal**: the mean open direction, along which they are
 *   evaluated;
 * - the **near-field light**: last frame's lit colour of the samples that
 *   raise a horizon, times the share of the hemisphere they cover, which the
 *   probes are too coarse to see.
 *
 * One kernel, `SSGI` (Capsaicin's `Main`, ssgi.comp:132-252), a thread per
 * pixel, run before GI-1's passes as Capsaicin's renderer runs the SSGI
 * technique before GI-1 (gi1_renderer.cpp:60-61).
 *
 * Differences from Capsaicin, in the register (docs/design/radiance-cache.md):
 * white noise for the blue noise (D55), last frame's lighting from
 * `FrameScreen.previousLighting` (D56), the technique a module of the
 * radiance cache (D57), two guards where D3D's NaN rules give the answer
 * (D58), sky pixels told by depth and one normal for both of Capsaicin's
 * (D17), and linear depth from the rebuilt point (D29).
 */

import { CAPSAICIN_WGSL } from './capsaicin.ts';
import { FRAME_WGSL, screenDepthWgsl, screenNormalWgsl, UNIFORM_ENTRY } from './screen-probes.ts';

const COMPUTE = 0x4;

/** The kernel's entry point: Capsaicin's `Main` in `ssgi.comp`, named after its technique. */
export const SSGI_ENTRY_POINT = 'SSGI';

/** `[numthreads(8, 8, 1)]` (ssgi.comp:132). */
export const SSGI_WORKGROUP_SIZE = 8;

/** Bytes of `RcSSGIConstants`. */
export const SSGI_CONSTANTS_BYTES = 96;

/**
 * Byte offsets of `RcSSGIConstants`' fields by WGSL's layout rules (a
 * `mat4x4f` is 64 bytes; a `vec3f` aligns to 16 and the scalar after it
 * packs into its last 4 bytes). Checked against the struct by a unit test.
 */
export const SSGI_CONSTANTS_OFFSETS = {
    inverseView: 0,
    forward: 64,
    uvRadius: 76,
    falloffMul: 80,
    falloffAdd: 84,
    sliceCount: 88,
    stepCount: 92,
} as const;

/** The two outputs' format: Capsaicin's `DXGI_FORMAT_R16G16B16A16_FLOAT` (ssgi.cpp:74-77). */
export const SSGI_OUTPUT_FORMAT: GPUTextureFormat = 'rgba16float';

export const SSGI_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/SSGI group 1',
    entries: [
        UNIFORM_ENTRY,
        { binding: 1, visibility: COMPUTE, buffer: { type: 'uniform', minBindingSize: SSGI_CONSTANTS_BYTES } },
        { binding: 2, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 3, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 4, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 5, visibility: COMPUTE, storageTexture: { access: 'write-only', format: SSGI_OUTPUT_FORMAT, viewDimension: '2d' } },
        { binding: 6, visibility: COMPUTE, storageTexture: { access: 'write-only', format: SSGI_OUTPUT_FORMAT, viewDimension: '2d' } },
    ],
};

export const SSGI_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${FRAME_WGSL}
${screenDepthWgsl(2)}
${screenNormalWgsl(3)}

// SSGIConstants (ssgi_shared.h:34-51), the fields Main reads beyond what RcFrame has
// (inv_view_proj, eye, buffer_dimensions and frame_index are RcFrame's).
struct RcSSGIConstants {
    inverseView: mat4x4f,   // inv_view: view-space directions to world space
    forward: vec3f,         // forward: normalize(center - eye), the camera's view axis
    uvRadius: f32,          // uv_radius: view_radius * 0.5 * max(proj[0][0], proj[1][1]) (ssgi.cpp:128-129)
    falloffMul: f32,        // falloff_mul: -1 / (view_radius * falloff_range) (ssgi.cpp:130-133)
    falloffAdd: f32,        // falloff_add: view_radius * (1 - falloff_range) / (view_radius * falloff_range) + 1
    sliceCount: u32,        // slice_count (ssgi_slice_count_)
    stepCount: u32,         // step_count (ssgi_step_count_)
}
@group(1) @binding(1) var<uniform> rcSSGI: RcSSGIConstants;
// g_LightingBuffer: last frame's lit colour, PrevCombinedIllumination (FrameScreen.previousLighting, D56).
@group(1) @binding(4) var rcLightingBuffer: texture_2d<f32>;
// g_OcclusionAndBentNormalBuffer: rgb the bent normal as 0.5 n + 0.5, a the ambient occlusion.
@group(1) @binding(5) var rcOcclusionAndBentNormalBuffer: texture_storage_2d<${SSGI_OUTPUT_FORMAT}, write>;
// g_NearFieldGlobalIlluminationBuffer: rgb the near-field light.
@group(1) @binding(6) var rcNearFieldGlobalIlluminationBuffer: texture_storage_2d<${SSGI_OUTPUT_FORMAT}, write>;

const RC_SSGI_PI: f32 = 3.14159265358979;
const RC_SSGI_HALF_PI: f32 = 1.57079632679490;
// normalize(2 * 0 - 1): what Capsaicin's decode makes of a sky pixel's cleared normal (D58).
const RC_SSGI_SKY_NORMAL: vec3f = vec3f(-0.577350269189626);

// g_PointSampler (MIN_MAG_MIP_POINT, clamped to the edge; ssgi.cpp:217-218): the texel under uv
// of a texture of any size, as SampleLevel(g_PointSampler, uv, 0) picks it.
fn rcSSGIPointTexel(uv: vec2f, size: vec2u) -> vec2u {
    let texel = vec2i(floor(uv * vec2f(size)));
    return vec2u(clamp(texel, vec2i(0), vec2i(size) - 1));
}

// The shading normal at a sample (ssgi.comp:212). Capsaicin's sky pixels hold a cleared normal,
// which normalize(2 * normal - 1) turns into -(1, 1, 1) / sqrt(3); FrameScreen's sky is told by
// depth (D17), and a signed-float host's cleared normal would normalise to NaN, so a sky sample
// gets Capsaicin's value (D58). Its weight is 0 anyway: the far plane is past the view radius.
fn rcSSGISampleNormal(pixel: vec2u) -> vec3f {
    if (rcIsSkyPixel(pixel)) {
        return RC_SSGI_SKY_NORMAL;
    }
    return rcLoadNormal(pixel);
}

// IntersectDirPlaneOneSided (ssgi.comp:72-77), from h3r2tic's demo: the distance along dir, from
// the origin, to the plane through pt with this normal; facing away from it, a large one.
fn rcIntersectDirPlaneOneSided(dir: vec3f, normal: vec3f, pt: vec3f) -> f32 {
    let d = -dot(pt, normal);
    let t = d / max(1e-5, -dot(dir, normal));
    return t;
}

// RotFromToMatrix (ssgi.comp:81-111), from the Open Asset Import Library: the rotation taking
// the unit vector fromDirection onto toDirection. Capsaicin fills an HLSL float3x3 by rows (mtx[row][column]) and
// applies it as mul(mtx, v); WGSL's constructor takes columns, hence the transpose.
fn rcRotFromToMatrix(fromDirection: vec3f, toDirection: vec3f) -> mat3x3f {
    let e = dot(fromDirection, toDirection);
    let f = abs(e);

    if (f > 1.0 - 0.0003) {
        return mat3x3f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    }

    let v = cross(fromDirection, toDirection);
    let h = 1.0 / (1.0 + e); // optimization by Gottfried Chen
    let hvx = h * v.x;
    let hvz = h * v.z;
    let hvxy = hvx * v.y;
    let hvxz = hvx * v.z;
    let hvyz = hvz * v.y;

    let row0 = vec3f(e + hvx * v.x, hvxy - v.z, hvxz + v.y);
    let row1 = vec3f(hvxy + v.z, e + h * v.y * v.y, hvyz - v.x);
    let row2 = vec3f(hvxz - v.y, hvyz + v.x, e + hvz * v.z);

    return transpose(mat3x3f(row0, row1, row2));
}

// IntegrateHalfArc (ssgi.comp:113-116): GTAO's cosine-weighted visibility of one side of a slice,
// from the normal's angle to the horizon's, so the two sides of an open slice sum to 1.
fn rcIntegrateHalfArc(horizonAngle: f32, normalAngle: f32) -> f32 {
    return (cos(normalAngle) + 2.0 * horizonAngle * sin(normalAngle) - cos(2.0 * horizonAngle - normalAngle)) / 4.0;
}

// IntegrateBentNormal (ssgi.comp:118-130): the open directions between the two horizons of a
// slice, integrated in view space, taken to world space and turned from the view axis to this
// pixel's own direction to the eye.
fn rcIntegrateBentNormal(horizonAngle0: f32, horizonAngle1: f32, normalAngle: f32, worldEyeDir: vec3f, sliceViewDir: vec3f) -> vec3f {
    let t0 = (6.0 * sin(horizonAngle0 - normalAngle) - sin(3.0 * horizonAngle0 - normalAngle)
            + 6.0 * sin(horizonAngle1 - normalAngle) - sin(3.0 * horizonAngle1 - normalAngle)
            + 16.0 * sin(normalAngle) - 3.0 * (sin(horizonAngle0 + normalAngle) + sin(horizonAngle1 + normalAngle))) / 12.0;
    let t1 = (-cos(3.0 * horizonAngle0 - normalAngle)
            - cos(3.0 * horizonAngle1 - normalAngle)
            + 8.0 * cos(normalAngle) - 3.0 * (cos(horizonAngle0 + normalAngle) + cos(horizonAngle1 + normalAngle))) / 12.0;

    let viewBentNormal = vec3f(sliceViewDir.x * t0, sliceViewDir.y * t0, t1);
    let worldBentNormal = (rcSSGI.inverseView * vec4f(viewBentNormal, 0.0)).xyz;
    return rcRotFromToMatrix(-rcSSGI.forward, worldEyeDir) * worldBentNormal;
}

// Main (ssgi.comp:132-252): a thread per pixel.
@compute @workgroup_size(${SSGI_WORKGROUP_SIZE}, ${SSGI_WORKGROUP_SIZE})
fn ${SSGI_ENTRY_POINT}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.xy;
    let dims = rcFrame.bufferDimensions;
    if (any(did >= dims)) {
        return; // out of bounds
    }

    let uv = (vec2f(did) + 0.5) / vec2f(dims);
    // Capsaicin tells a sky pixel by a zero shading normal (ssgi.comp:142); FrameScreen by depth (D17).
    if (rcIsSkyPixel(did)) {
        textureStore(rcOcclusionAndBentNormalBuffer, did, vec4f(0.0));
        textureStore(rcNearFieldGlobalIlluminationBuffer, did, vec4f(0.0));
        return; // discard sky pixels
    }

    let depth = rcLoadDepth(did);
    let worldPos = rcTransformPointProjection(uv, depth);
    // toLinearDepth(depth, g_NearFar): the distance along the view axis, from the rebuilt point (D29).
    let linearDepth = dot(worldPos - rcFrame.eye, rcSSGI.forward);
    let worldNormal = rcLoadNormal(did);
    let worldEyeDir = normalize(rcFrame.eye - worldPos);

    // MakeBlueNoiseSampler(did, frame_index).rand2() (ssgi.comp:156-157): the probes' PCG instead
    // (D55), from its own seed so the interpolation's jitter (seeded by the pixel index) doesn't
    // repeat these numbers; Capsaicin keeps the two apart by blue-noise dimension 0 and 1.
    var random = rcMakeRandom(did.x + did.y * dims.x + dims.x * dims.y, rcFrame.frameIndex);
    let noise = rcRand2(&random); // in [0, 1]

    let sliceCount = rcSSGI.sliceCount;
    let stepCount = rcSSGI.stepCount;
    let sliceUvRadius = rcSSGI.uvRadius / linearDepth;
    var ambientOcclusion = 0.0;
    var bentNormal = vec3f(0.0);
    var globalLighting = vec3f(0.0);
    for (var sliceIndex = 0u; sliceIndex < sliceCount; sliceIndex++) {
        let sliceAngle = ((f32(sliceIndex) + noise.x) / f32(sliceCount)) * RC_SSGI_PI;
        let sliceCos = cos(sliceAngle);
        let sliceSin = sin(sliceAngle);
        // uv runs down the screen, view space up it.
        let sliceUvDir = vec2f(sliceCos, -sliceSin) * sliceUvRadius;

        let sliceViewDir = vec3f(sliceCos, sliceSin, 0.0);
        let sliceWorldDir = (rcSSGI.inverseView * vec4f(sliceViewDir, 0.0)).xyz;
        let orthoWorldDir = sliceWorldDir - dot(sliceWorldDir, worldEyeDir) * worldEyeDir;
        let projAxisDir = normalize(cross(orthoWorldDir, worldEyeDir));
        let projWorldNormal = worldNormal - dot(worldNormal, projAxisDir) * projAxisDir;
        let projWorldNormalLen = length(projWorldNormal);
        let projWorldNormalCos = saturate(dot(projWorldNormal, worldEyeDir) / projWorldNormalLen);
        let projWorldNormalAngle = sign(dot(orthoWorldDir, projWorldNormal)) * acos(projWorldNormalCos);

        var horizonAngles: array<f32, 2>;
        for (var sideIndex = 0u; sideIndex < 2u; sideIndex++) {
            // side_signs = {1, -1}
            let sideSign = select(-1.0, 1.0, sideIndex == 0u);
            let horizonMin = cos(projWorldNormalAngle + sideSign * RC_SSGI_HALF_PI);
            var horizonCos = horizonMin;
            var prevSampleWorldPos = vec3f(0.0);
            for (var stepIndex = 0u; stepIndex < stepCount; stepIndex++) {
                let sampleStep = (f32(stepIndex) + noise.y) / f32(stepCount);
                let sampleUvOffset = sampleStep * sliceUvDir;

                let sampleUv = uv + sideSign * sampleUvOffset;
                let samplePixel = rcSSGIPointTexel(sampleUv, dims);
                let sampleDepth = rcLoadDepth(samplePixel);
                // At the sample's own uv, not its texel's centre, as Capsaicin rebuilds it.
                let sampleWorldPos = rcTransformPointProjection(sampleUv, sampleDepth);

                var horizonWorldDir = sampleWorldPos - worldPos;
                let horizonWorldLen = length(horizonWorldDir); // BE CAREFUL: main source of precision errors
                // A sample in this pixel's own texel is this point: Capsaicin divides by its zero
                // length, and D3D's max() and >= then drop the NaN, so the sample changes nothing.
                // WGSL gives no such NaN a defined result, so the sample is skipped (D58).
                if (horizonWorldLen > 0.0) {
                    horizonWorldDir /= horizonWorldLen;
                    let sampleWeight = saturate(horizonWorldLen * rcSSGI.falloffMul + rcSSGI.falloffAdd);
                    let sampleCos = mix(horizonMin, dot(horizonWorldDir, worldEyeDir), sampleWeight);
                    var prevHorizonCos = horizonCos;
                    horizonCos = max(horizonCos, sampleCos);

                    if (sampleCos >= prevHorizonCos) {
                        // From h3r2tic's demo: the light of the sample that raised the horizon.
                        let sampleLighting = textureLoad(rcLightingBuffer, rcSSGIPointTexel(sampleUv, textureDimensions(rcLightingBuffer)), 0).rgb;
                        let sampleWorldNormal = rcSSGISampleNormal(samplePixel);

                        if (stepIndex > 0u) {
                            // Capsaicin's positions are world space where h3r2tic's are view space, so
                            // this intersects from the world's origin, not the eye: kept as Capsaicin
                            // has it (the design doc's "worth telling AMD").
                            let closestWorldPos = prevSampleWorldPos * min(
                                rcIntersectDirPlaneOneSided(prevSampleWorldPos, sampleWorldNormal, sampleWorldPos),
                                rcIntersectDirPlaneOneSided(prevSampleWorldPos, worldNormal, worldPos),
                            );

                            prevHorizonCos = clamp(dot(normalize(closestWorldPos - worldPos), worldEyeDir), prevHorizonCos, horizonCos);
                        }

                        let horizonAngle0 = projWorldNormalAngle + max(sideSign * acos(prevHorizonCos) - projWorldNormalAngle, -RC_SSGI_HALF_PI);
                        let horizonAngle1 = projWorldNormalAngle + min(sideSign * acos(horizonCos) - projWorldNormalAngle, RC_SSGI_HALF_PI);
                        let sampleOcclusion = rcIntegrateHalfArc(horizonAngle0, projWorldNormalAngle) - rcIntegrateHalfArc(horizonAngle1, projWorldNormalAngle);

                        globalLighting += sampleLighting * sampleOcclusion * step(0.0, dot(-horizonWorldDir, sampleWorldNormal));
                    }
                }

                prevSampleWorldPos = sampleWorldPos;
            }

            // Ambient occlusion
            horizonAngles[sideIndex] = sideSign * acos(horizonCos);
            ambientOcclusion += projWorldNormalLen * rcIntegrateHalfArc(horizonAngles[sideIndex], projWorldNormalAngle);

            // Global lighting: Capsaicin scales the sum so far on each side, so the first side's
            // light (and any earlier slice's) is scaled more than once; kept as it is (the design
            // doc's "worth telling AMD").
            globalLighting *= projWorldNormalLen;
        }

        // Bent normal
        bentNormal += projWorldNormalLen * rcIntegrateBentNormal(horizonAngles[0], horizonAngles[1], projWorldNormalAngle, worldEyeDir, sliceViewDir);
    }

    ambientOcclusion /= f32(sliceCount);
    bentNormal = normalize(bentNormal);
    globalLighting /= f32(sliceCount);

    textureStore(rcOcclusionAndBentNormalBuffer, did, vec4f(0.5 * bentNormal + 0.5, ambientOcclusion));
    textureStore(rcNearFieldGlobalIlluminationBuffer, did, vec4f(globalLighting, 0.0));
}
`;
