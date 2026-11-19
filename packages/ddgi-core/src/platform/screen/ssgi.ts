/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/ssgi/ssgi.comp
 * (Main, IntegrateHalfArc, IntegrateBentNormal, RotFromToMatrix, IntersectDirPlaneOneSided)
 * and ssgi.cpp (the constants). ssgi.comp in turn takes IntersectDirPlaneOneSided and the
 * near-field light from h3r2tic's rtoy-samples (ssgi.glsl) and RotFromToMatrix from the
 * Open Asset Import Library, as its comments say.
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
 * Screen-space GI (RC4.2, `GpuScreenSpaceGI` in platform/contract/compute.ts):
 * Capsaicin's SSGI technique, which GI-1 reads in `InterpolateScreenProbes`.
 * Per pixel, a few depth samples along random slices around the view axis
 * give the horizon on each side (a horizon-based ambient occlusion, GTAO's
 * integrals), from which come the cosine-weighted visibility, the bent
 * normal, and the light the occluders' pixels send through the covered
 * part of the hemisphere (h3r2tic's near-field light).
 *
 * In the platform, not a technique: it reads only `FrameScreen` and the
 * camera, and DDGI could narrow its own lookup with the same occlusion. One
 * kernel, `Main` (ssgi.comp:132-252), a thread per pixel, Capsaicin's 8 × 8.
 *
 * Differences from Capsaicin, in the radiance cache's register
 * (docs/design/radiance-cache.md): the near-field light is scaled by π to be
 * irradiance (D52), its projected-normal weight is applied once per slice
 * (D53), the occluder-thickness step works around the eye as h3r2tic's
 * view-space original does (D54), the per-pixel random numbers are white
 * noise (D55), a depth sample is rebuilt at its texel's centre (D56), and
 * the slices are taken relative to the same slices with nothing in the way
 * (D57).
 * The arithmetic runs in view space (rotations and
 * dot products are the same; positions are relative to the eye, which keeps
 * them small) and the bent normal is stored signed (Capsaicin's 0..1
 * encoding in a float texture changes nothing).
 */

import type { GpuScreenSpaceGI, ScreenSpaceGIInputs, ScreenSpaceGIOptions, ScreenSpaceGITextures } from '../contract/compute.ts';
import type { FrameCamera } from '../contract/frame.ts';

const SHADER_STAGE_COMPUTE = 0x4;
const TEXTURE_USAGE = { copySrc: 0x01, copyDst: 0x02, textureBinding: 0x04, storageBinding: 0x08 } as const;
const BUFFER_USAGE = { uniform: 0x40, copyDst: 0x08 } as const;
/** `rgba16float`: four halves a texel. */
const TEXEL_BYTES = 8;
/** `[numthreads(8, 8, 1)]` (ssgi.comp:132). */
const WORKGROUP = 8;
/** Bytes of `GiSsgi`. */
const UNIFORM_BYTES = 176;

/** Byte offsets of `GiSsgi`'s fields (WGSL layout; a `mat4x4f` is 64 bytes). */
export const SSGI_UNIFORM_OFFSETS = {
    inverseProjection: 0,
    view: 64,
    size: 128,
    frameIndex: 136,
    sliceCount: 140,
    stepCount: 144,
    uvRadius: 148,
    falloffMul: 152,
    falloffAdd: 156,
    normalUnorm: 160,
    reversedDepth: 164,
    hasLighting: 168,
} as const;

export const SSGI_WGSL = /* wgsl */ `
// SSGIConstants (ssgi_shared.h:34-50), the matrices in view space (file comment).
struct GiSsgi {
    inverseProjection: mat4x4f, // inverse of this frame's (jittered) projection: NDC to view space
    view: mat4x4f,              // world to view: its rotation turns normals into view space, its transpose back
    size: vec2u,                // buffer_dimensions
    frameIndex: u32,            // frame_index
    sliceCount: u32,            // slice_count
    stepCount: u32,             // step_count
    uvRadius: f32,              // uv_radius: view_radius * 0.5 * max(P[0][0], P[1][1]) (ssgi.cpp:132-133)
    falloffMul: f32,            // falloff_mul: -1 / falloff_range (ssgi.cpp:134-137)
    falloffAdd: f32,            // falloff_add: falloff_from / falloff_range + 1
    normalUnorm: u32,           // 1: FrameScreen's normals are 0..1 encoded
    reversedDepth: u32,         // 1: depth 1 is near, so the sky is at 0
    hasLighting: u32,           // 0: no lighting image, the near-field light is zero
    pad: u32,
}
@group(0) @binding(0) var<uniform> giSsgi: GiSsgi;
@group(0) @binding(1) var giSsgiDepth: texture_2d<f32>;
@group(0) @binding(2) var giSsgiNormal: texture_2d<f32>;
// g_LightingBuffer: Capsaicin binds last frame's combined image (PrevCombinedIllumination, ssgi.cpp:150-151).
@group(0) @binding(3) var giSsgiLighting: texture_2d<f32>;
// g_OcclusionAndBentNormalBuffer and g_NearFieldGlobalIlluminationBuffer.
@group(0) @binding(4) var giSsgiOcclusion: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var giSsgiNearField: texture_storage_2d<rgba16float, write>;

const GI_SSGI_PI: f32 = 3.14159265358979;
const GI_SSGI_HALF_PI: f32 = 1.57079632679490;

// g_PointSampler (clamp, ssgi.cpp:218-219): the texel under uv, clamped to the screen.
fn giSsgiTexel(uv: vec2f) -> vec2u {
    let size = vec2i(giSsgi.size);
    return vec2u(clamp(vec2i(floor(uv * vec2f(size))), vec2i(0), size - 1));
}

fn giSsgiDepthAt(uv: vec2f) -> f32 {
    return textureLoad(giSsgiDepth, giSsgiTexel(uv), 0).x;
}

// The view-space normal under uv, unit length; a pixel nothing was drawn on stays zero.
fn giSsgiNormalAt(uv: vec2f) -> vec3f {
    let stored = textureLoad(giSsgiNormal, giSsgiTexel(uv), 0).xyz;
    if (all(stored == vec3f(0.0))) {
        return vec3f(0.0);
    }
    let world = select(stored, 2.0 * stored - 1.0, giSsgi.normalUnorm != 0u);
    return normalize((giSsgi.view * vec4f(world, 0.0)).xyz);
}

// transformPointProjection(uv, depth, inv_view_proj) (transform.hlsl:110-113), in view space.
fn giSsgiViewPosition(uv: vec2f, depth: f32) -> vec3f {
    let ndc = vec4f(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, depth, 1.0);
    let homogeneous = giSsgi.inverseProjection * ndc;
    return homogeneous.xyz / homogeneous.w;
}

// A view-space direction in world space: the view rotation's transpose.
fn giSsgiToWorld(direction: vec3f) -> vec3f {
    let v = giSsgi.view;
    return vec3f(dot(v[0].xyz, direction), dot(v[1].xyz, direction), dot(v[2].xyz, direction));
}

// Two uniform numbers per pixel and frame (Capsaicin's blue-noise sampler, MakeBlueNoiseSampler(did,
// frame_index).rand2(), ssgi.comp:156-157; white noise here for the reason of register D27).
fn giSsgiHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
fn giSsgiNoise(pixel: vec2u) -> vec2f {
    let first = giSsgiHash(pixel.x + pixel.y * giSsgi.size.x + giSsgiHash(giSsgi.frameIndex));
    let second = giSsgiHash(first);
    return vec2f(f32(first >> 8u), f32(second >> 8u)) / 16777216.0;
}

// IntersectDirPlaneOneSided (ssgi.comp:72-77, from h3r2tic's ssgi.glsl): the distance along 'dir'
// from the origin to the plane through 'pt' with normal 'normal'. The origin is the eye in view
// space, as in h3r2tic's original (register D54).
fn giSsgiIntersectDirPlaneOneSided(dir: vec3f, normal: vec3f, pt: vec3f) -> f32 {
    let d = -dot(pt, normal);
    return d / max(1e-5, -dot(dir, normal));
}

// RotFromToMatrix (ssgi.comp:81-111, from assimp's FromToMatrix) applied to 'v': the rotation taking
// unit 'from' to unit 'to'.
fn giSsgiRotateFromTo(fromDir: vec3f, toDir: vec3f, v: vec3f) -> vec3f {
    let e = dot(fromDir, toDir);
    let f = abs(e);

    if (f > 1.0 - 0.0003) {
        return v;
    }

    let c = cross(fromDir, toDir);
    let h = 1.0 / (1.0 + e); // optimization by Gottfried Chen
    let hvx = h * c.x;
    let hvz = h * c.z;
    let hvxy = hvx * c.y;
    let hvxz = hvx * c.z;
    let hvyz = hvz * c.y;

    // mtx[row][column] as ssgi.comp writes it; mul(mtx, v) takes rows.
    let row0 = vec3f(e + hvx * c.x, hvxy - c.z, hvxz + c.y);
    let row1 = vec3f(hvxy + c.z, e + h * c.y * c.y, hvyz - c.x);
    let row2 = vec3f(hvxz - c.y, hvyz + c.x, e + hvz * c.z);
    return vec3f(dot(row0, v), dot(row1, v), dot(row2, v));
}

// IntegrateHalfArc (ssgi.comp:113-116): GTAO's cosine-weighted integral of one side of a slice,
// from the normal's angle to the horizon's.
fn giSsgiIntegrateHalfArc(horizonAngle: f32, normalAngle: f32) -> f32 {
    return (cos(normalAngle) + 2.0 * horizonAngle * sin(normalAngle) - cos(2.0 * horizonAngle - normalAngle)) / 4.0;
}

// IntegrateBentNormal (ssgi.comp:118-130): the slice's unoccluded directions' mean, in view space.
// Capsaicin turns it to world space and rotates it from the view axis (-forward) to this pixel's
// eye direction; in view space the view axis is +z.
fn giSsgiIntegrateBentNormal(horizonAngle0: f32, horizonAngle1: f32, normalAngle: f32, eyeDir: vec3f, sliceViewDir: vec3f) -> vec3f {
    let t0 = (6.0 * sin(horizonAngle0 - normalAngle) - sin(3.0 * horizonAngle0 - normalAngle) +
              6.0 * sin(horizonAngle1 - normalAngle) - sin(3.0 * horizonAngle1 - normalAngle) +
              16.0 * sin(normalAngle) - 3.0 * (sin(horizonAngle0 + normalAngle) + sin(horizonAngle1 + normalAngle))) / 12.0;
    let t1 = (-cos(3.0 * horizonAngle0 - normalAngle)
              - cos(3.0 * horizonAngle1 - normalAngle) +
              8.0 * cos(normalAngle) - 3.0 * (cos(horizonAngle0 + normalAngle) + cos(horizonAngle1 + normalAngle))) / 12.0;

    let viewBentNormal = vec3f(sliceViewDir.x * t0, sliceViewDir.y * t0, t1);
    return giSsgiRotateFromTo(vec3f(0.0, 0.0, 1.0), eyeDir, viewBentNormal);
}

// acos of a cosine that rounding may have pushed just past [-1, 1] (WGSL leaves acos undefined there).
fn giSsgiAcos(cosine: f32) -> f32 {
    return acos(clamp(cosine, -1.0, 1.0));
}

// Main (ssgi.comp:132-252).
@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
    let did = id.xy;
    if (any(did >= giSsgi.size)) {
        return; // out of bounds
    }

    let uv = (vec2f(did) + 0.5) / vec2f(giSsgi.size);
    let normal = giSsgiNormalAt(uv);
    let depth = giSsgiDepthAt(uv);
    let farDepth = select(1.0, 0.0, giSsgi.reversedDepth != 0u);
    // Capsaicin tells the sky by a zero normal; the host's screen also by the far plane (register D17).
    if (all(normal == vec3f(0.0)) || depth == farDepth) {
        textureStore(giSsgiOcclusion, did, vec4f(0.0));
        textureStore(giSsgiNearField, did, vec4f(0.0));
        return; // discard sky pixels
    }

    let position = giSsgiViewPosition(uv, depth);
    let linearDepth = -position.z;
    let eyeDir = normalize(-position);

    let noise = giSsgiNoise(did); // in [0, 1)

    let sliceUvRadius = giSsgi.uvRadius / linearDepth;
    var ambientOcclusion = 0.0;
    var bentNormal = vec3f(0.0);
    var globalLighting = vec3f(0.0);
    // What the same slices give with nothing in the way: the visibility and bent normal are taken
    // relative to them (register D57).
    var openVisibility = 0.0;
    var openBentNormal = vec3f(0.0);
    let sliceCount = giSsgi.sliceCount;
    let stepCount = giSsgi.stepCount;
    for (var sliceIndex = 0u; sliceIndex < sliceCount; sliceIndex++) {
        let sliceAngle = ((f32(sliceIndex) + noise.x) / f32(sliceCount)) * GI_SSGI_PI;
        let sliceCos = cos(sliceAngle);
        let sliceSin = sin(sliceAngle);
        let sliceUvDir = vec2f(sliceCos, -sliceSin) * sliceUvRadius;

        // The view-space direction the uv step (cos, -sin) moves along on the image plane: the
        // projection's scales turn it (register D58: Capsaicin takes (cos, sin, 0), which is that
        // direction only for square pixels and equal scales, so its samples leave the slice).
        let sliceViewDir = normalize(vec3f(sliceCos * giSsgi.inverseProjection[0][0], sliceSin * giSsgi.inverseProjection[1][1], 0.0));
        let orthoDir = sliceViewDir - dot(sliceViewDir, eyeDir) * eyeDir;
        let projAxisDir = normalize(cross(orthoDir, eyeDir));
        let projNormal = normal - dot(normal, projAxisDir) * projAxisDir;
        let projNormalLen = length(projNormal);
        let projNormalCos = saturate(dot(projNormal, eyeDir) / max(projNormalLen, 1e-8));
        let projNormalAngle = sign(dot(orthoDir, projNormal)) * giSsgiAcos(projNormalCos);

        // Both sides' near-field light, weighted by the projected normal's length once for the
        // slice (register D53: Capsaicin multiplies the running total after each side).
        var sliceLighting = vec3f(0.0);
        var horizonAngles: array<f32, 2>;
        for (var sideIndex = 0u; sideIndex < 2u; sideIndex++) {
            let sideSign = select(-1.0, 1.0, sideIndex == 0u);
            let horizonMin = cos(projNormalAngle + sideSign * GI_SSGI_HALF_PI);
            var horizonCos = horizonMin;
            var prevSamplePos = vec3f(0.0);
            for (var stepIndex = 0u; stepIndex < stepCount; stepIndex++) {
                let sampleStep = (f32(stepIndex) + noise.y) / f32(stepCount);
                let sampleUvOffset = sampleStep * sliceUvDir;

                // The sample is where the view ray through its own uv meets the plane of the texel it
                // reads (that texel's depth, at its centre, and its normal): on the surface and in the
                // slice (register D56). Capsaicin puts the texel's depth at the unsnapped uv, off the
                // surface by up to half a texel's depth change, which on a floor seen at a grazing
                // angle raises the horizon; the texel's centre alone would leave the slice sideways.
                let rawUv = uv + sideSign * sampleUvOffset;
                let sampleUv = (vec2f(giSsgiTexel(rawUv)) + 0.5) / vec2f(giSsgi.size);
                let sampleDepth = giSsgiDepthAt(sampleUv);
                let texelPos = giSsgiViewPosition(sampleUv, sampleDepth);
                let texelNormal = giSsgiNormalAt(sampleUv);
                let rayDir = normalize(giSsgiViewPosition(rawUv, sampleDepth));
                let rayCos = dot(rayDir, texelNormal);
                let texelDistance = length(texelPos);
                var samplePos = texelPos;
                if (abs(rayCos) > 1e-3) {
                    // Kept within a factor of two of the texel's own distance, against normals that don't fit the depth.
                    samplePos = rayDir * clamp(dot(texelPos, texelNormal) / rayCos, 0.5 * texelDistance, 2.0 * texelDistance);
                }

                let horizonLen = length(samplePos - position); // BE CAREFUL: main source of precision errors
                // A sample at an infinite far plane (or on the pixel itself) is no occluder: no direction, no weight.
                let finite = horizonLen > 0.0 && horizonLen < 3.0e38;
                let horizonDir = select(vec3f(0.0), (samplePos - position) / horizonLen, finite);
                let sampleWeight = select(0.0, saturate(horizonLen * giSsgi.falloffMul + giSsgi.falloffAdd), finite);
                let sampleCos = mix(horizonMin, dot(horizonDir, eyeDir), sampleWeight);
                var prevHorizonCos = horizonCos;
                horizonCos = max(horizonCos, sampleCos);

                if (sampleCos >= prevHorizonCos) {
                    // From h3r2tic's demo (ssgi.comp:209-228).
                    var sampleLighting = vec3f(0.0);
                    if (giSsgi.hasLighting != 0u) {
                        sampleLighting = textureLoad(giSsgiLighting, giSsgiTexel(sampleUv), 0).rgb;
                    }
                    let sampleNormal = texelNormal;

                    if (stepIndex > 0u) {
                        let closestPos = prevSamplePos * min(
                            giSsgiIntersectDirPlaneOneSided(prevSamplePos, sampleNormal, samplePos),
                            giSsgiIntersectDirPlaneOneSided(prevSamplePos, normal, position),
                        );
                        let toClosest = closestPos - position;
                        if (dot(toClosest, toClosest) > 0.0) {
                            prevHorizonCos = clamp(dot(normalize(toClosest), eyeDir), prevHorizonCos, horizonCos);
                        }
                    }

                    let horizonAngle0 = projNormalAngle + max(sideSign * giSsgiAcos(prevHorizonCos) - projNormalAngle, -GI_SSGI_HALF_PI);
                    let horizonAngle1 = projNormalAngle + min(sideSign * giSsgiAcos(horizonCos) - projNormalAngle, GI_SSGI_HALF_PI);
                    // An arc is never negative; rounding and the thickness step can make it so (register D57).
                    let sampleOcclusion = max(giSsgiIntegrateHalfArc(horizonAngle0, projNormalAngle) - giSsgiIntegrateHalfArc(horizonAngle1, projNormalAngle), 0.0);

                    sliceLighting += sampleLighting * sampleOcclusion * step(0.0, dot(-horizonDir, sampleNormal));
                }

                prevSamplePos = samplePos;
            }

            // Ambient occlusion.
            horizonAngles[sideIndex] = sideSign * giSsgiAcos(horizonCos);
            ambientOcclusion += projNormalLen * giSsgiIntegrateHalfArc(horizonAngles[sideIndex], projNormalAngle);
        }
        globalLighting += projNormalLen * sliceLighting;

        // Bent normal.
        bentNormal += projNormalLen * giSsgiIntegrateBentNormal(horizonAngles[0], horizonAngles[1], projNormalAngle, eyeDir, sliceViewDir);

        // The open slice: horizons at the normal's own hemisphere boundary.
        let openAngle0 = projNormalAngle + GI_SSGI_HALF_PI;
        let openAngle1 = projNormalAngle - GI_SSGI_HALF_PI;
        openVisibility += projNormalLen * (giSsgiIntegrateHalfArc(openAngle0, projNormalAngle) + giSsgiIntegrateHalfArc(openAngle1, projNormalAngle));
        openBentNormal += projNormalLen * giSsgiIntegrateBentNormal(openAngle0, openAngle1, projNormalAngle, eyeDir, sliceViewDir);
    }

    // Capsaicin averages the slices (ambient_occlusion / SLICE_COUNT, normalize(bent_normal)), which is
    // right only on average over slice angles: one slice (the default) of an open surface reads
    // len (cos n + n sin n), anywhere from 0 to above 1, and leans the bent normal towards the eye,
    // and the SH cone lookup is not linear in either. Ours: the slices' visibility over what the
    // same slices give open, and the bent normal as the pixel's normal moved by what the occluders
    // took away, so an open surface reads 1 and its own normal in every frame (register D57).
    var visibility = 1.0;
    var bentView = normal;
    if (openVisibility > 0.0) {
        visibility = saturate(ambientOcclusion / openVisibility);
        let moved = normal * length(openBentNormal) + (bentNormal - openBentNormal);
        let movedLength = length(moved);
        bentView = select(normal, moved / movedLength, movedLength > 0.0);
    }
    // The arcs are shares of the cosine-weighted hemisphere (pi), so the light they carry is E / pi:
    // times pi it is irradiance, which the GI it is added to is (register D52). Taken over the same
    // open total as the visibility (D57), so far field and near field share one hemisphere.
    globalLighting *= GI_SSGI_PI / max(openVisibility, 1e-8);

    textureStore(giSsgiOcclusion, did, vec4f(giSsgiToWorld(bentView), visibility));
    textureStore(giSsgiNearField, did, vec4f(globalLighting, 0.0));
}
`;

const SSGI_LAYOUT: GPUBindGroupLayoutDescriptor = {
    label: 'gi/ssgi',
    entries: [
        { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform', minBindingSize: UNIFORM_BYTES } },
        { binding: 1, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 4, visibility: SHADER_STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 5, visibility: SHADER_STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
};

export interface ScreenSpaceGIKernel {
    readonly layout: GPUBindGroupLayout;
    readonly pipeline: GPUComputePipeline;
}

/** Compiles the SSGI kernel (once per platform; every `ScreenSpaceGI` shares it). */
export async function compileScreenSpaceGIKernel(device: GPUDevice, label: string): Promise<ScreenSpaceGIKernel> {
    const layout = device.createBindGroupLayout({ ...SSGI_LAYOUT, label: `${label} gi/ssgi` });
    const pipeline = await device.createComputePipelineAsync({
        label: `${label} gi/ssgi`,
        layout: device.createPipelineLayout({ label: `${label} gi/ssgi`, bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ label: `${label} gi/ssgi`, code: SSGI_WGSL }), entryPoint: 'main' },
    });
    return { layout, pipeline };
}

interface Sized extends ScreenSpaceGITextures {
    readonly width: number;
    readonly height: number;
}

/** A depth texture is read through its depth aspect; anything else whole. */
function screenView(texture: GPUTexture): GPUTextureView {
    const isDepth = texture.format.startsWith('depth');
    return texture.createView({ aspect: isDepth ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
}

export class ScreenSpaceGI implements GpuScreenSpaceGI {
    private sized: Sized | null = null;
    private readonly uniforms: GPUBuffer;
    private readonly uniformWords = new ArrayBuffer(UNIFORM_BYTES);
    /** Bound when there is no lighting image: the near-field light is then zero (`hasLighting` 0). */
    private readonly noLighting: GPUTexture;
    private group: { readonly key: readonly unknown[]; readonly bindGroup: GPUBindGroup } | null = null;

    constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ScreenSpaceGIKernel,
        private readonly options: ScreenSpaceGIOptions,
    ) {
        this.uniforms = device.createBuffer({ label: `${options.label} ssgi`, size: UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        this.noLighting = device.createTexture({ label: `${options.label} ssgi no lighting`, size: [1, 1], format: 'rgba16float', usage: TEXTURE_USAGE.textureBinding });
    }

    get memoryBytes(): number {
        return UNIFORM_BYTES + (this.sized ? 2 * this.sized.width * this.sized.height * TEXEL_BYTES : 0);
    }

    texturesFor(width: number, height: number): ScreenSpaceGITextures {
        const current = this.sized;
        if (current && current.width === width && current.height === height) return current;
        this.releaseSized();
        const usage = TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.copySrc;
        const make = (name: string) => this.device.createTexture({ label: `${this.options.label} ssgi ${name}`, size: [width, height], format: 'rgba16float', usage });
        const sized: Sized = { width, height, occlusion: make('occlusion and bent normal'), nearField: make('near-field light') };
        this.sized = sized;
        return sized;
    }

    encode(encoder: GPUCommandEncoder, inputs: ScreenSpaceGIInputs): ScreenSpaceGITextures {
        const { screen } = inputs;
        const sized = this.texturesFor(screen.width, screen.height) as Sized;
        this.writeUniforms(inputs);
        const lighting = inputs.lighting ?? this.noLighting;
        const key = [sized, screen.depth, screen.normal, lighting];
        if (!this.group || this.group.key.some((value, index) => value !== key[index])) {
            this.group = {
                key,
                bindGroup: this.device.createBindGroup({
                    label: `${this.options.label} ssgi`,
                    layout: this.kernel.layout,
                    entries: [
                        { binding: 0, resource: { buffer: this.uniforms } },
                        { binding: 1, resource: screenView(screen.depth) },
                        { binding: 2, resource: screenView(screen.normal) },
                        { binding: 3, resource: screenView(lighting) },
                        { binding: 4, resource: sized.occlusion.createView() },
                        { binding: 5, resource: sized.nearField.createView() },
                    ],
                }),
            };
        }
        const pass = encoder.beginComputePass({ label: `${this.options.label} ssgi` });
        pass.setPipeline(this.kernel.pipeline);
        pass.setBindGroup(0, this.group.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(screen.width / WORKGROUP), Math.ceil(screen.height / WORKGROUP));
        pass.end();
        return sized;
    }

    dispose(): void {
        this.releaseSized();
        this.uniforms.destroy();
        this.noLighting.destroy();
    }

    /** `SSGIConstants` (ssgi.cpp:118-138), for view space. */
    private writeUniforms(inputs: ScreenSpaceGIInputs): void {
        const { screen, camera, settings } = inputs;
        for (const [name, value] of [['sliceCount', settings.sliceCount], ['stepCount', settings.stepCount]] as const) {
            if (!Number.isInteger(value) || value < 1) throw new RangeError(`@pmndrs/gi: SSGI's ${name} must be a whole number of at least 1, got ${value}`);
        }
        if (!(settings.viewRadius > 0) || !(settings.falloffRange > 0 && settings.falloffRange <= 1)) {
            throw new RangeError(`@pmndrs/gi: SSGI needs a radius above 0 and a falloff range in (0, 1], got ${settings.viewRadius} and ${settings.falloffRange}`);
        }
        const inverseProjection = invert4(jitteredProjection(camera));
        if (!inverseProjection) throw new Error('@pmndrs/gi: SSGI needs a camera projection that can be inverted');
        const words = this.uniformWords;
        const floats = new Float32Array(words);
        const uints = new Uint32Array(words);
        const at = (offset: number) => offset / 4;
        const offsets = SSGI_UNIFORM_OFFSETS;
        floats.set(inverseProjection, at(offsets.inverseProjection));
        floats.set(Array.from(camera.view), at(offsets.view));
        uints[at(offsets.size)] = screen.width;
        uints[at(offsets.size) + 1] = screen.height;
        uints[at(offsets.frameIndex)] = inputs.frame >>> 0;
        uints[at(offsets.sliceCount)] = settings.sliceCount;
        uints[at(offsets.stepCount)] = settings.stepCount;
        // ssgi.cpp:132-137; projection[0][0] and [1][1] in glm are elements 0 and 5 column-major.
        floats[at(offsets.uvRadius)] = settings.viewRadius * 0.5 * Math.max(camera.projection[0], camera.projection[5]);
        const falloffRange = settings.viewRadius * settings.falloffRange;
        const falloffFrom = settings.viewRadius * (1 - settings.falloffRange);
        floats[at(offsets.falloffMul)] = -1 / falloffRange;
        floats[at(offsets.falloffAdd)] = falloffFrom / falloffRange + 1;
        uints[at(offsets.normalUnorm)] = screen.normalEncoding === 'unorm' ? 1 : 0;
        uints[at(offsets.reversedDepth)] = screen.reversedDepth ? 1 : 0;
        uints[at(offsets.hasLighting)] = inputs.lighting ? 1 : 0;
        this.device.queue.writeBuffer(this.uniforms, 0, words);
    }

    private releaseSized(): void {
        if (!this.sized) return;
        this.sized.occlusion.destroy();
        this.sized.nearField.destroy();
        this.sized = null;
        this.group = null;
    }
}

/**
 * The projection the screen was drawn with: `translate(jitter) × projection`
 * (`FrameCamera.jitter`), so positions rebuilt from depth land where the depth was drawn.
 */
export function jitteredProjection(camera: Pick<FrameCamera, 'projection' | 'jitter'>): number[] {
    const projection = Array.from(camera.projection);
    const [jitterX, jitterY] = camera.jitter ?? [0, 0];
    for (let column = 0; column < 4; column++) {
        const w = projection[column * 4 + 3];
        projection[column * 4 + 0] += jitterX * w;
        projection[column * 4 + 1] += jitterY * w;
    }
    return projection;
}

/** The inverse of a column-major 4 × 4 matrix (cofactors), or null if it has none. */
export function invert4(m: ArrayLike<number>): number[] | null {
    const out = new Array<number>(16);
    // Each entry of the adjugate: the signed minor of the transposed position.
    const minor = (skipRow: number, skipColumn: number): number => {
        const rows = [0, 1, 2, 3].filter((row) => row !== skipRow);
        const columns = [0, 1, 2, 3].filter((column) => column !== skipColumn);
        const at = (row: number, column: number) => m[columns[column] * 4 + rows[row]];
        return (
            at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) -
            at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
            at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0))
        );
    };
    let determinant = 0;
    for (let column = 0; column < 4; column++) determinant += m[column * 4] * (column % 2 === 0 ? 1 : -1) * minor(0, column);
    if (!Number.isFinite(determinant) || determinant === 0) return null;
    for (let row = 0; row < 4; row++) {
        for (let column = 0; column < 4; column++) {
            // inverse[row][column] = cofactor[column][row] / determinant.
            out[column * 4 + row] = (((row + column) % 2 === 0 ? 1 : -1) * minor(column, row)) / determinant;
        }
    }
    return out;
}
