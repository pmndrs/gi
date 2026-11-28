/**
 * Capsaicin's temporal radiance feedback (`gi1_use_temporal_feedback`,
 * gi1.comp:2454-2496 in `GenerateReservoirs`; RC3.6, register D68): a probe
 * ray whose hit was on last frame's screen takes last frame's lit image there
 * (the paper's §2.2.4: direct light plus every bounce the image holds) instead
 * of its world-cache cell.
 *
 * What is Capsaicin's: where it applies (a first-bounce hit that isn't
 * emissive, with direct lighting on), the lit image it reads (last frame's
 * final colour; here the host's lit colour plus albedo / π × E, register D51),
 * the probe ray reading that value whole, and a normal test (its 0.5).
 *
 * What differs (register D68):
 * - **The reprojection.** Capsaicin projects the hit with this frame's camera
 *   and follows the velocity stored at that pixel, which belongs to whatever
 *   surface the camera sees there now, then compares linear depth within 5%.
 *   Here the hit is projected with last frame's camera straight into last
 *   frame's screen (a world-space hit is where it was, for a still surface),
 *   and the point last frame's pixel showed must lie on the hit's plane within
 *   2% of the distance (the platform's screen-history rule, as SSGI's D60).
 * - **The cell.** Capsaicin also adds the lit value into the hit's cell as a
 *   direct-light sample and skips the hit's shadow ray, so the cell's direct
 *   value mixes direct light with every bounce, and `ResolveCells` and the
 *   second bounce then add the cell's indirect value on top: the later bounces
 *   counted twice for every reader of that cell. Here the cell keeps its own
 *   light sample (the shadow ray is traced as without feedback) and the lit
 *   value reaches only this probe ray.
 *
 * Engine-free WGSL for `GenerateReservoirs`' group 3: the uniform with last
 * frame's camera, last frame's lit image, depth and normals.
 */

import type { FrameCamera } from '@pmndrs/gi-core/platform/contract/frame';
import { cameraRelativeInverseViewProjection, cameraRelativeViewProjection } from '../camera.ts';

/** `GPUShaderStage.COMPUTE` (the global doesn't exist in Node, where the tests import this file). */
const SHADER_STAGE_COMPUTE = 0x4;

/** The group `GenerateReservoirs` reads last frame's screen from (the hash grid is 1, the ray data 2). */
export const TEMPORAL_FEEDBACK_GROUP = 3;

/** Bindings of the group. */
export const TEMPORAL_FEEDBACK_BINDING = {
    /** `HashGridCacheTemporalFeedback`: last frame's camera and screen size. Uniform. */
    uniform: 0,
    /** Last frame's lit image (`rgba16float`, the radiance cache's combined image, register D51). */
    lighting: 1,
    /** Last frame's `FrameScreen.depth` (`r32float`, the platform's screen-history copy). */
    depth: 2,
    /** Last frame's normals (`rgba16float`, decoded, the platform's screen-history copy). */
    normal: 3,
} as const;

export const TEMPORAL_FEEDBACK_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    { binding: TEMPORAL_FEEDBACK_BINDING.uniform, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { binding: TEMPORAL_FEEDBACK_BINDING.lighting, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
    { binding: TEMPORAL_FEEDBACK_BINDING.depth, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
    { binding: TEMPORAL_FEEDBACK_BINDING.normal, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
];

/** Bytes of `HashGridCacheTemporalFeedback`. */
export const TEMPORAL_FEEDBACK_UNIFORM_BYTES = 160;

/** Word offsets in the uniform. */
export const TEMPORAL_FEEDBACK_WORD = {
    previousViewProjection: 0,
    previousInverseViewProjection: 16,
    previousEye: 32,
    planeTolerance: 35,
    size: 36,
    normalThreshold: 38,
} as const;

/**
 * The plane test: the point last frame's pixel showed lies within this share
 * of the hit's distance from the hit's plane (the platform's screen-history
 * rule, SSGI's D60 and the probes' reprojection).
 */
export const TEMPORAL_FEEDBACK_PLANE_TOLERANCE = 0.02;

/** The normal test: last frame's normal there · the hit's normal above this (Capsaicin's 5e-1, gi1.comp:2475). */
export const TEMPORAL_FEEDBACK_NORMAL_THRESHOLD = 0.5;

/** Last frame's screen, as `GenerateReservoirs` reads it (`WorldCacheFrame.temporalFeedback`). */
export interface WorldCacheTemporalFeedback {
    /** Last frame's camera: the one its screen was drawn with. */
    readonly camera: Pick<FrameCamera, 'position' | 'view' | 'projection' | 'jitter'>;
    /** Last frame's lit image (`rgba16float`, `TEXTURE_BINDING`), `width` × `height`. */
    readonly lighting: GPUTexture;
    /** Last frame's depth (`r32float`) and normals (`rgba16float`), the same size. */
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    readonly width: number;
    readonly height: number;
}

/** The uniform's words for last frame's camera and screen. */
export function temporalFeedbackUniformWords(feedback: Pick<WorldCacheTemporalFeedback, 'camera' | 'width' | 'height'>): ArrayBuffer {
    const bytes = new ArrayBuffer(TEMPORAL_FEEDBACK_UNIFORM_BYTES);
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    const word = TEMPORAL_FEEDBACK_WORD;
    floats.set(cameraRelativeViewProjection(feedback.camera), word.previousViewProjection);
    floats.set(cameraRelativeInverseViewProjection(feedback.camera), word.previousInverseViewProjection);
    floats.set(feedback.camera.position, word.previousEye);
    floats[word.planeTolerance] = TEMPORAL_FEEDBACK_PLANE_TOLERANCE;
    words[word.size] = feedback.width;
    words[word.size + 1] = feedback.height;
    floats[word.normalThreshold] = TEMPORAL_FEEDBACK_NORMAL_THRESHOLD;
    return bytes;
}

/**
 * The group's declarations and `HashGridCache_TemporalFeedback(surface)`:
 * last frame's lit radiance leaving `surface` in rgb, and 1 in w where last
 * frame's screen showed that surface there (0 otherwise). `surface.normal`
 * faces the probe ray's side, so a hit on the side the camera didn't see
 * fails the normal test.
 */
export const TEMPORAL_FEEDBACK_WGSL = /* wgsl */ `
struct HashGridCacheTemporalFeedback {
    previous_view_projection: mat4x4f,          // last frame's projection x view x T(eye): camera-relative
    previous_inverse_view_projection: mat4x4f,  // its inverse: NDC to a point relative to last frame's eye
    previous_eye: vec3f,
    plane_tolerance: f32,                       // the shown point within this share of the distance from the hit's plane
    size: vec2u,                                // last frame's screen in pixels
    normal_threshold: f32,                      // last frame's normal there . the hit's normal above this
    padding: u32,
}

@group(${TEMPORAL_FEEDBACK_GROUP}) @binding(${TEMPORAL_FEEDBACK_BINDING.uniform}) var<uniform> g_HashGridCache_TemporalFeedback: HashGridCacheTemporalFeedback;
@group(${TEMPORAL_FEEDBACK_GROUP}) @binding(${TEMPORAL_FEEDBACK_BINDING.lighting}) var g_HashGridCache_PreviousLighting: texture_2d<f32>;
@group(${TEMPORAL_FEEDBACK_GROUP}) @binding(${TEMPORAL_FEEDBACK_BINDING.depth}) var g_HashGridCache_PreviousDepth: texture_2d<f32>;
@group(${TEMPORAL_FEEDBACK_GROUP}) @binding(${TEMPORAL_FEEDBACK_BINDING.normal}) var g_HashGridCache_PreviousNormal: texture_2d<f32>;

// Last frame's lit radiance at the surface where last frame's screen showed it (w 1), else w 0.
fn HashGridCache_TemporalFeedback(surface: GiSurface) -> vec4f {
    let feedback = g_HashGridCache_TemporalFeedback;
    let relative = surface.position - feedback.previous_eye;
    let clip = feedback.previous_view_projection * vec4f(relative, 1.0);
    if (clip.w <= 0.0) {
        return vec4f(0.0); // behind last frame's camera
    }
    let uv = vec2f(0.5 * clip.x / clip.w + 0.5, 0.5 - 0.5 * clip.y / clip.w);
    if (any(uv <= vec2f(0.0)) || any(uv >= vec2f(1.0))) {
        return vec4f(0.0); // off last frame's screen
    }
    let texel = min(vec2u(uv * vec2f(feedback.size)), feedback.size - vec2u(1u));
    let texel_uv = (vec2f(texel) + 0.5) / vec2f(feedback.size);

    // The point last frame's pixel showed (sky pixels sit on the far plane and fail the plane test).
    let depth = textureLoad(g_HashGridCache_PreviousDepth, texel, 0).x;
    let homogeneous = feedback.previous_inverse_view_projection * vec4f(2.0 * texel_uv.x - 1.0, 1.0 - 2.0 * texel_uv.y, depth, 1.0);
    if (homogeneous.w == 0.0) {
        return vec4f(0.0);
    }
    let shown = homogeneous.xyz / homogeneous.w;
    let same_plane = abs(dot(shown - relative, surface.normal)) <= feedback.plane_tolerance * length(relative);
    // A pixel with nothing drawn stores a zero normal, which fails this too.
    let same_normal = dot(textureLoad(g_HashGridCache_PreviousNormal, texel, 0).xyz, surface.normal) > feedback.normal_threshold;
    if (!same_plane || !same_normal) {
        return vec4f(0.0);
    }
    return vec4f(textureLoad(g_HashGridCache_PreviousLighting, texel, 0).rgb, 1.0);
}
`;
