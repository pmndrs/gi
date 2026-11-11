/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.comp
 * (ReprojectGI, FilterGI) and src/core/src/render_techniques/gi1/gi_denoiser.hlsl.
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
 * GI-1's image-space GI denoiser, the kernels Capsaicin runs after
 * `InterpolateScreenProbes` and before `ResolveGI1` (`GI1::render`,
 * gi1.cpp:2882-2927), each named after its entry point:
 *
 * 1. `ReprojectGI` (gi1.comp:3996-4102): per pixel, last frame's history
 *    (irradiance × samples in rgb, the sample count in alpha) found through
 *    the motion vector with a 3 × 3 tent, kept where the surface is the same
 *    (distance within a cell, normals within 0.95), then this frame's sample
 *    appended and the history shortened where the light changes. It writes
 *    the blur mask: 8 minus the history's length, so a pixel with less than
 *    8 frames of history is blurred, and more the less it has.
 * 2. `FilterGI` (gi1.comp:4104-4154), twice: horizontally, then vertically
 *    (gi1.cpp:2907-2926), over the mask's radius, weighted by depth and
 *    normal. The second pass's output is next frame's history, and its mean
 *    (rgb / count) is the denoised irradiance (`g_IrradianceBuffer`).
 *
 * `KeepGuides` is ours: Capsaicin's framework keeps last frame's depth and
 * shading normals for it (`PrevVisibilityDepth`, `PrevShadingNormal`,
 * gi1.cpp:2010-2014); this module keeps its own copies.
 *
 * The constants are Capsaicin's (constants.ts, with their lines). Where this
 * differs from Capsaicin, the reason is in the radiance cache's divergence
 * register (webgpu-ddgi `docs/design/radiance-cache-divergences.md`, D65 onwards):
 * texture formats (Capsaicin's with `texture-formats-tier1`, else a fallback: D65), the input in its own texture (D66), sky by depth
 * (D67), the guides kept here (D68), points relative to the eye (D69),
 * linear depth from the rebuilt point (D70), NaN removal by the bits (D71),
 * a point sampler's texel by hand (D72), the delta's sign by comparisons (D75).
 */

import {
    COLOR_DELTA_SMOOTHING,
    FILTER_DEPTH_SHARPNESS_WITH_HISTORY,
    FILTER_DEPTH_SHARPNESS_WITHOUT_HISTORY,
    GRAZING_CELL_EXPONENT,
    GRAZING_CELL_SCALE,
    HISTORY_SAMPLES_PER_COUNT,
    LUMINANCE_WEIGHTS,
    MAX_BLUR_MASK,
    MIN_HISTORY_LUMINANCE,
    MIN_HISTORY_SAMPLES,
    ONE_OVER_SQRT_OF_TWO,
    REMOVE_NANS_FLOOR,
    REPROJECTION_NORMAL_COSINE,
    VIGNETTE_EXPONENT,
    VIGNETTE_SCALE,
    WORKGROUP_SIZE,
} from './constants.ts';

const COMPUTE = 0x4;

/** Entry points: Capsaicin's kernel names, and ours for the guide copy. */
export const ENTRY_POINT = {
    reproject: 'ReprojectGI',
    filter: 'FilterGI',
    keepGuides: 'KeepGuides',
} as const;

/**
 * Texture formats. Capsaicin's history colour and output are
 * `R16G16B16A16_FLOAT` (gi1.cpp:1210-1212, 1707), as here.
 */
export const FORMAT = {
    color: 'rgba16float',
    irradiance: 'rgba16float',
    previousDepth: 'r32float',
    previousNormal: 'rgba16float',
} as const satisfies Record<string, GPUTextureFormat>;

/** The blur mask's and the colour deltas' formats: Capsaicin's where the device allows them, else the fallback. */
export interface GI1DenoiserFormats {
    readonly blurMask: 'r8snorm' | 'rgba8snorm';
    readonly colorDelta: 'r16float' | 'r32float';
}

/** The feature that makes `r8snorm` and `r16float` storage formats. */
export const TEXTURE_FORMATS_TIER1: GPUFeatureName = 'texture-formats-tier1';

/** Capsaicin's: `R8_SNORM` and `R16_FLOAT` (gi1.cpp:1200, 1222). Storage formats only with `texture-formats-tier1`. */
export const CAPSAICIN_FORMATS: GI1DenoiserFormats = { blurMask: 'r8snorm', colorDelta: 'r16float' };

/**
 * The one declared fallback, in core WebGPU (register D65): the mask in an
 * `rgba8snorm` texel's r (the same snorm steps), the delta in f32.
 */
export const CORE_FORMATS: GI1DenoiserFormats = { blurMask: 'rgba8snorm', colorDelta: 'r32float' };

/** Capsaicin's formats when the device has `texture-formats-tier1`, otherwise the fallback. */
export function formatsFor(features: { has(feature: string): boolean }): GI1DenoiserFormats {
    return features.has(TEXTURE_FORMATS_TIER1) ? CAPSAICIN_FORMATS : CORE_FORMATS;
}

/** Bytes of `GIDenoiserConstants`, and its fields' byte offsets (written by name in gi1-denoiser.ts). */
export const CONSTANTS_BYTES = 160;
export const CONSTANTS_OFFSETS = {
    inverseViewProjection: 0,
    previousInverseViewProjection: 64,
    viewForward: 128,
    cellSize: 140,
    bufferDimensions: 144,
    reversedDepth: 152,
    normalUnorm: 156,
} as const;

/** Bytes of `GIDenoiserFilterPass`: the blur direction (`g_BlurDirection`, gi1.comp:38) and padding. */
export const FILTER_PASS_BYTES = 16;

// ---- Shared declarations ------------------------------------------------------------------------

const UNIFORM = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: COMPUTE, buffer: { type: 'uniform' } });
const TEXTURE = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float' } });
const STORAGE_TEXTURE = (binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: COMPUTE,
    storageTexture: { access: 'write-only', format },
});

const CONSTANTS_WGSL = /* wgsl */ `
struct GIDenoiserConstants {
    // inverse(projection × view × translate(eye)), with this frame's jitter: a pixel's uv and
    // depth to its point relative to this frame's eye (Capsaicin's view_proj_inv, gi1.comp:4023).
    inverseViewProjection: mat4x4f,
    // The previous frame's, with its jitter, relative to this frame's eye too
    // (g_PreviousViewProjectionInverse, gi1.comp:4049), so the two frames' points subtract directly.
    previousInverseViewProjection: mat4x4f,
    // The camera's forward axis: linear depth is measured along it (FilterGI's toLinearDepth).
    viewForward: vec3f,
    // g_ScreenProbesConstants.cell_size (gi1.cpp:1839-1843): a cell's size per unit distance.
    cellSize: f32,
    bufferDimensions: vec2u,
    // The depth is reversed-Z (1 = near): the far plane, where sky pixels sit, is then 0.
    reversedDepth: u32,
    // FrameGuides.normalEncoding is 'unorm': normals are 0..1 encoded.
    normalUnorm: u32,
}
@group(0) @binding(0) var<uniform> giConstants: GIDenoiserConstants;

const MAX_BLUR_MASK: f32 = ${MAX_BLUR_MASK.toFixed(1)};
`;

/** Helpers every kernel uses. Needs `giDepth` and `giNormal` declared. */
const SCREEN_WGSL = /* wgsl */ `
fn giLoadDepth(pixel: vec2u) -> f32 {
    return textureLoad(giDepth, pixel, 0).x;
}

// Capsaicin tells a sky pixel by its zero shading normal (gi1.comp:4011). The guides say it by
// depth: a pixel at the far plane drew nothing (register D67).
fn giIsSkyDepth(depth: f32) -> bool {
    let farDepth = select(1.0, 0.0, giConstants.reversedDepth != 0u);
    return depth == farDepth;
}

// 2.0f * g_ShadingNormalBuffer.Load(...).xyz - 1.0f (gi1.comp:4019, 4121), for either encoding.
fn giLoadNormal(pixel: vec2u) -> vec3f {
    let stored = textureLoad(giNormal, pixel, 0).xyz;
    return select(stored, 2.0 * stored - 1.0, giConstants.normalUnorm != 0u);
}

// transformPointProjection(uv, depth, inverse) (src/core/src/math/transform.hlsl:94-113): the
// point seen through uv at this depth, relative to this frame's eye (register D69).
fn giTransformPointProjection(uv: vec2f, depth: f32, inverseViewProjection: mat4x4f) -> vec3f {
    let ndc = vec4f(2.0 * uv.x - 1.0, 2.0 * (1.0 - uv.y) - 1.0, depth, 1.0);
    let homogeneous = inverseViewProjection * ndc;
    return homogeneous.xyz / homogeneous.w;
}
`;

const REMOVE_NANS_WGSL = /* wgsl */ `
// D3D's saturate takes NaN to 0, so on D3D Capsaicin's formula turns NaN and both infinities into 0
// (inf / (1 + inf) is NaN there too). WGSL's saturate doesn't promise that, and on the Apple M5 a NaN
// came through (radiance-cache-denoiser check 4), so the non-finite case is told by its bits,
// which no float assumption can fold away (register D71).
fn giIsFinite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// GIDenoiser_RemoveNaNs (gi_denoiser.hlsl:64-92): x / (1 + x), saturated, mapped back. The identity
// on finite values up to about 1e4, which it caps; 0 for NaN and the infinities, as on D3D.
fn giRemoveNaNs(color: f32) -> f32 {
    if (!giIsFinite(color)) {
        return 0.0;
    }
    let compressed = saturate(color / (1.0 + color));
    return compressed / max(1.0 - compressed, ${REMOVE_NANS_FLOOR});
}

fn giRemoveNaNs4(color: vec4f) -> vec4f {
    return vec4f(giRemoveNaNs(color.x), giRemoveNaNs(color.y), giRemoveNaNs(color.z), giRemoveNaNs(color.w));
}
`;

// ---- ReprojectGI --------------------------------------------------------------------------------

export function reprojectLayout(formats: GI1DenoiserFormats): GPUBindGroupLayoutDescriptor {
    return {
        label: 'gi1-denoiser/ReprojectGI',
        entries: [
            UNIFORM(0),
            TEXTURE(1),
            TEXTURE(2),
            TEXTURE(3),
            TEXTURE(4),
            TEXTURE(5),
            TEXTURE(6),
            TEXTURE(7),
            TEXTURE(8),
            STORAGE_TEXTURE(9, formats.blurMask),
            STORAGE_TEXTURE(10, FORMAT.color),
            STORAGE_TEXTURE(11, formats.colorDelta),
        ],
    };
}

export function reprojectWgsl(formats: GI1DenoiserFormats): string {
    return /* wgsl */ `
${CONSTANTS_WGSL}
// This frame's noisy irradiance and Capsaicin's denoiser hint in alpha: what
// InterpolateScreenProbes left in g_GIDenoiser_ColorBuffer (gi1.comp:1638), in a texture of its
// own (register D66).
@group(0) @binding(1) var giColorInput: texture_2d<f32>;
@group(0) @binding(2) var giDepth: texture_2d<f32>;
@group(0) @binding(3) var giNormal: texture_2d<f32>;
// g_VelocityBuffer: FrameGuides.motion, the NDC motion (current - previous) in rg.
@group(0) @binding(4) var giMotion: texture_2d<f32>;
// g_GIDenoiser_PreviousColorBuffer, g_GIDenoiser_PreviousColorDeltaBuffer: last frame's history.
@group(0) @binding(5) var giPreviousColor: texture_2d<f32>;
@group(0) @binding(6) var giPreviousColorDelta: texture_2d<f32>;
// g_PreviousDepthBuffer, g_PreviousDetailsBuffer: last frame's depth and normals as KeepGuides kept
// them (normals decoded, 0 on sky pixels, as Capsaicin's zero normal there).
@group(0) @binding(7) var giPreviousDepth: texture_2d<f32>;
@group(0) @binding(8) var giPreviousNormal: texture_2d<f32>;
// g_GIDenoiser_BlurMask, g_GIDenoiser_ColorBuffer, g_GIDenoiser_ColorDeltaBuffer.
// (r8snorm and r16float with texture-formats-tier1, else rgba8snorm and r32float: register D65.)
@group(0) @binding(9) var giBlurMask: texture_storage_2d<${formats.blurMask}, write>;
@group(0) @binding(10) var giColor: texture_storage_2d<${FORMAT.color}, write>;
@group(0) @binding(11) var giColorDelta: texture_storage_2d<${formats.colorDelta}, write>;

${SCREEN_WGSL}
${REMOVE_NANS_WGSL}

fn giLuminance(color: vec3f) -> f32 {
    return dot(color, vec3f(${LUMINANCE_WEIGHTS.join(', ')}));
}

// The texel g_NearestSampler (point filtering, clamped) picks at st: SampleLevel at a uv off the
// pixel grid, as a load (register D72).
fn giNearestTexel(st: vec2f, dimensions: vec2u) -> vec2i {
    return clamp(vec2i(floor(st * vec2f(dimensions))), vec2i(0), vec2i(dimensions) - 1);
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn ${ENTRY_POINT.reproject}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.xy;
    let dimensions = giConstants.bufferDimensions;
    if (any(did >= dimensions)) {
        return; // out of bounds
    }

    let color = textureLoad(giColorInput, did, 0);
    var lighting = vec4f(0.0);
    let depth = giLoadDepth(did);

    var alphaBlend = 1.0;
    var colorDelta = 0.0;
    let uv = (vec2f(did) + 0.5) / vec2f(dimensions);
    let isSkyPixel = giIsSkyDepth(depth);

    if (!isSkyPixel) {
        // g_VelocityBuffer holds the uv motion with y down (geometry.hlsl:104-110); the guides hold
        // NDC motion with y up: half of it, y flipped.
        let velocity = textureLoad(giMotion, did, 0).xy * vec2f(0.5, -0.5);
        let previousUv = uv - velocity;
        let normal = giLoadNormal(did);

        if (all(previousUv > vec2f(0.0)) && all(previousUv < vec2f(1.0))) {
            let world = giTransformPointProjection(uv, depth, giConstants.inverseViewProjection);
            // distance(g_Eye, world): the points are relative to the eye.
            var cellSize = length(world) * giConstants.cellSize;
            // Grow the tolerance up to five times at grazing angles, where neighbouring pixels sit
            // far apart along the surface.
            let facing = max(dot(normalize(-world), normalize(normal)), 0.0);
            cellSize *= mix(1.0, ${GRAZING_CELL_SCALE.toFixed(1)}, pow(1.0 - facing, ${GRAZING_CELL_EXPONENT.toFixed(1)}));

            var weight = 0.0;
            let texelSize = 1.0 / vec2f(dimensions);

            // A 3 × 3 neighbourhood round the reprojected position, each tap weighted by a tent of its
            // texel centre's distance from it (0 at √2 pixels), kept only on the same surface.
            for (var y = -1; y <= 1; y++) {
                for (var x = -1; x <= 1; x++) {
                    let st = previousUv + vec2f(f32(x), f32(y)) * texelSize;
                    let texel = giNearestTexel(st, dimensions);
                    let c = textureLoad(giPreviousColor, texel, 0);
                    let previousNormal = textureLoad(giPreviousNormal, texel, 0).xyz;

                    if (c.w < 1.0 || dot(previousNormal, previousNormal) == 0.0) {
                        continue; // skip invalid sample(s)
                    }

                    let previousDepth = textureLoad(giPreviousDepth, texel, 0).x;
                    let previousWorld = giTransformPointProjection(st, previousDepth, giConstants.previousInverseViewProjection);

                    if (distance(world, previousWorld) < cellSize && dot(normal, previousNormal) > ${REPROJECTION_NORMAL_COSINE}) {
                        let subpixelDistance = distance(floor(st * vec2f(dimensions)) + 0.5, previousUv * vec2f(dimensions));
                        let w = saturate(1.0 - subpixelDistance * ${ONE_OVER_SQRT_OF_TWO});

                        colorDelta += w * textureLoad(giPreviousColorDelta, texel, 0).x;
                        lighting += w * c;
                        weight += w;
                    }
                }
            }

            if (weight > 0.0) {
                // HLSL's sign returns an int: 0 for NaN, whose delta RemoveNaNs then zeroes. WGSL's
                // float sign(NaN) has no defined value, so the comparisons say it (register D75).
                let deltaSign = select(select(0.0, -1.0, colorDelta < 0.0), 1.0, colorDelta > 0.0);
                colorDelta = giRemoveNaNs(abs(colorDelta) / weight);
                colorDelta *= deltaSign; // restore delta

                lighting = giRemoveNaNs4(lighting / weight);
            }

            if (color.w > 0.0) {
                let lumaA = giLuminance(color.xyz);
                let lumaB = giLuminance(lighting.xyz / max(lighting.w, 1.0));

                colorDelta = mix(colorDelta, lumaA - lumaB, ${COLOR_DELTA_SMOOTHING}); // smoothed delta
                alphaBlend = saturate(1.0 - abs(colorDelta) / max(lumaB, ${MIN_HISTORY_LUMINANCE}));
            }
        }
    }

    // Fewer than MAX_BLUR_MASK frames of history: blur, the more the fewer (sky: never).
    let blurMask = select(-1.0, max(MAX_BLUR_MASK - lighting.w, 0.0), !isSkyPixel);

    // Append this frame's sample. A sample the interpolation marked weak (hint 0) is taken only
    // when there is no history, and then counted as -1, so it is replaced as soon as possible.
    if (color.w > 0.0 || lighting.w < 1.0) {
        lighting += vec4f(color.xyz, select(-1.0, 1.0, color.w > 0.0));
    }

    // Shorten the history where the light changes (alphaBlend falls) and towards the screen's
    // edges (the vignette), where reprojection fails first.
    let vignetteUv = uv * (1.0 - uv.yx);
    let vignette = pow(${VIGNETTE_SCALE.toFixed(1)} * vignetteUv.x * vignetteUv.y, ${VIGNETTE_EXPONENT});
    let maxSampleCount = max(mix(${MIN_HISTORY_SAMPLES.toFixed(1)}, ${HISTORY_SAMPLES_PER_COUNT.toFixed(1)} * min(abs(lighting.w), MAX_BLUR_MASK), alphaBlend) * vignette, 1.0);

    if (lighting.w > maxSampleCount) {
        lighting *= maxSampleCount / lighting.w; // evict old samples from the history
    }

    textureStore(giBlurMask, did, vec4f(blurMask / MAX_BLUR_MASK, 0.0, 0.0, 0.0));
    textureStore(giColor, did, lighting);
    textureStore(giColorDelta, did, vec4f(colorDelta, 0.0, 0.0, 0.0));
}
`;
}

// ---- FilterGI -----------------------------------------------------------------------------------

export const FILTER_LAYOUT: GPUBindGroupLayoutDescriptor = {
    label: 'gi1-denoiser/FilterGI',
    entries: [UNIFORM(0), UNIFORM(1), TEXTURE(2), TEXTURE(3), TEXTURE(4), TEXTURE(5), STORAGE_TEXTURE(6, FORMAT.color), STORAGE_TEXTURE(7, FORMAT.irradiance)],
};

export const FILTER_WGSL = /* wgsl */ `
${CONSTANTS_WGSL}
struct GIDenoiserFilterPass {
    // g_BlurDirection (gi1.comp:38): (1, 0) on the first pass, (0, 1) on the second (gi1.cpp:2911).
    blurDirection: vec2i,
    padding: vec2u,
}
@group(0) @binding(1) var<uniform> giFilterPass: GIDenoiserFilterPass;
// g_GIDenoiser_PreviousColorBuffer in this pass: ReprojectGI's history (first pass) or the first
// pass's result (second pass), gi1.cpp:2917-2922.
@group(0) @binding(2) var giFilterInput: texture_2d<f32>;
// g_GIDenoiser_BlurMask, read.
@group(0) @binding(3) var giBlurMask: texture_2d<f32>;
@group(0) @binding(4) var giDepth: texture_2d<f32>;
@group(0) @binding(5) var giNormal: texture_2d<f32>;
// g_GIDenoiser_ColorBuffer in this pass, and g_IrradianceBuffer (written by the second pass only).
@group(0) @binding(6) var giFilterOutput: texture_storage_2d<${FORMAT.color}, write>;
@group(0) @binding(7) var giIrradiance: texture_storage_2d<${FORMAT.irradiance}, write>;

${SCREEN_WGSL}

// GIDenoiser_GetBlurRadius (gi_denoiser.hlsl:43-54): the mask back in pixels, at least 1 where
// it is above 0, rounded.
fn giGetBlurRadius(pixel: vec2u) -> i32 {
    var blurRadius = 0;
    let blurMask = textureLoad(giBlurMask, pixel, 0).x * MAX_BLUR_MASK;

    if (blurMask > 0.0) {
        blurRadius = i32(max(blurMask, 1.0) + 0.5);
    }

    return blurRadius;
}

// toLinearDepth(g_DepthBuffer.Load(...), g_NearFar) (transform.hlsl:144-148, reversed-Z): the
// distance along the view axis, here of the point rebuilt from the depth (register D70).
fn giLinearDepth(pixel: vec2u) -> f32 {
    let uv = (vec2f(pixel) + 0.5) / vec2f(giConstants.bufferDimensions);
    let point = giTransformPointProjection(uv, giLoadDepth(pixel), giConstants.inverseViewProjection);
    return dot(point, giConstants.viewForward);
}

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn ${ENTRY_POINT.filter}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.xy;
    let dimensions = giConstants.bufferDimensions;
    if (any(did >= dimensions)) {
        return; // out of bounds
    }

    var lighting = textureLoad(giFilterInput, did, 0);
    let blurRadius = giGetBlurRadius(did);
    let blurDirection = giFilterPass.blurDirection;

    if (blurRadius > 0) {
        var weight = 1.0;
        var color = lighting.xyz / max(lighting.w, 1.0);

        let centerDepth = giLinearDepth(did);
        let centerNormal = giLoadNormal(did);
        // A pixel with history samples (count above 0) is filtered sharply in depth and by
        // normal; one without, more widely and by depth only.
        let hasSamples = lighting.w > 0.0;
        let depthSharpness = select(${FILTER_DEPTH_SHARPNESS_WITHOUT_HISTORY.toFixed(1)}, ${FILTER_DEPTH_SHARPNESS_WITH_HISTORY.toFixed(1)}, hasSamples);

        for (var r = -blurRadius; r <= blurRadius; r++) {
            let pos = vec2u(clamp(vec2i(did) + r * blurDirection, vec2i(0), vec2i(dimensions) - 1));
            let c = textureLoad(giFilterInput, pos, 0);

            if (c.w > 0.0) {
                let depth = giLinearDepth(pos);
                let normal = giLoadNormal(pos);

                let depthDiff = 1.0 - (centerDepth / depth);
                let depthFactor = exp2(-depthSharpness * abs(depthDiff));
                // max(dot(n, centre n), 0) to the fourth: squared twice, as Capsaicin writes it.
                var normalFactor = max(dot(normal, centerNormal), 0.0);
                normalFactor *= normalFactor;
                normalFactor *= normalFactor;

                let w = depthFactor * select(1.0, normalFactor, hasSamples);

                color += w * (c.xyz / max(c.w, 1.0));
                weight += w;
            }
        }

        lighting = vec4f((color / weight) * max(lighting.w, 1.0), lighting.w);
    }

    if (blurDirection.y > 0) {
        textureStore(giIrradiance, did, vec4f(lighting.xyz / max(lighting.w, 1.0), 1.0));
    }

    textureStore(giFilterOutput, did, lighting);
}
`;

// ---- KeepGuides (ours) --------------------------------------------------------------------------

export const KEEP_GUIDES_LAYOUT: GPUBindGroupLayoutDescriptor = {
    label: 'gi1-denoiser/KeepGuides',
    entries: [UNIFORM(0), TEXTURE(1), TEXTURE(2), STORAGE_TEXTURE(3, FORMAT.previousDepth), STORAGE_TEXTURE(4, FORMAT.previousNormal)],
};

export const KEEP_GUIDES_WGSL = /* wgsl */ `
${CONSTANTS_WGSL}
@group(0) @binding(1) var giDepth: texture_2d<f32>;
@group(0) @binding(2) var giNormal: texture_2d<f32>;
@group(0) @binding(3) var giPreviousDepth: texture_storage_2d<${FORMAT.previousDepth}, write>;
@group(0) @binding(4) var giPreviousNormal: texture_storage_2d<${FORMAT.previousNormal}, write>;

${SCREEN_WGSL}

// The next frame's g_PreviousDepthBuffer and g_PreviousDetailsBuffer (Capsaicin's framework keeps
// PrevVisibilityDepth and PrevShadingNormal, register D68): the depth as it is, the normal decoded,
// and 0 on sky pixels, where Capsaicin's normal buffer holds 0.
@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE}, 1)
fn ${ENTRY_POINT.keepGuides}(@builtin(global_invocation_id) id: vec3u) {
    let did = id.xy;
    if (any(did >= giConstants.bufferDimensions)) {
        return;
    }
    let depth = giLoadDepth(did);
    let normal = select(giLoadNormal(did), vec3f(0.0), giIsSkyDepth(depth));
    textureStore(giPreviousDepth, did, vec4f(depth, 0.0, 0.0, 0.0));
    textureStore(giPreviousNormal, did, vec4f(normal, 0.0));
}
`;

/** Every kernel for these formats: entry point, WGSL and its one bind group's layout (the budget tests read these). */
export function denoiserKernels(formats: GI1DenoiserFormats): readonly { readonly entryPoint: string; readonly code: string; readonly layout: GPUBindGroupLayoutDescriptor }[] {
    return [
        { entryPoint: ENTRY_POINT.reproject, code: reprojectWgsl(formats), layout: reprojectLayout(formats) },
        { entryPoint: ENTRY_POINT.filter, code: FILTER_WGSL, layout: FILTER_LAYOUT },
        { entryPoint: ENTRY_POINT.keepGuides, code: KEEP_GUIDES_WGSL, layout: KEEP_GUIDES_LAYOUT },
    ];
}
