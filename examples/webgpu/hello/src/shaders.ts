/**
 * WGSL for the hello renderer: a forward pass that shades with GI from the
 * platform's shading module, a shadow-map pass for its one point light, and
 * a display pass that tone-maps to the canvas.
 *
 * The struct layouts here and the offsets in renderer.ts (`FRAME_FLOAT`,
 * `DRAW_FLOAT`) describe the same bytes; change them together.
 */

/** How the forward pass lights a pixel; `viewMode` in the frame uniforms. */
export const VIEW_MODE = {
    /** emission + albedo / π × (direct + GI), plus the environment where coverage is low. */
    full: 0,
    /** albedo / π × GI: the indirect light only (what the engine-free test view calls "indirect"). */
    gi: 1,
    /** The shading module's own debug colour for `debugView` (giDebugColor). */
    debug: 2,
} as const;

/** Where the forward pass gets irradiance from. */
export type GiSource =
    /** The platform's shading module, through its `globalsCode` at group 1. */
    | { readonly kind: 'module'; readonly globalsCode: string }
    /** A constant irradiance (`flatIrradiance` in the frame uniforms): GI off, or a cost baseline. */
    | { readonly kind: 'flat' };

/** The cube shadow map's faces, in array-layer order; renderer.ts builds one view-projection per face. */
export const SHADOW_FACES = [
    { name: 'positive x', direction: [1, 0, 0], up: [0, 1, 0] },
    { name: 'negative x', direction: [-1, 0, 0], up: [0, 1, 0] },
    { name: 'positive y', direction: [0, 1, 0], up: [0, 0, 1] },
    { name: 'negative y', direction: [0, -1, 0], up: [0, 0, -1] },
    { name: 'positive z', direction: [0, 0, 1], up: [0, 1, 0] },
    { name: 'negative z', direction: [0, 0, -1], up: [0, 1, 0] },
] as const;

const COMMON_WGSL = /* wgsl */ `
const HELLO_PI: f32 = 3.14159265358979;
const HELLO_INV_PI: f32 = 0.318309886183791;

struct FrameUniforms {
    viewProjection: mat4x4f,
    cameraPosition: vec3f,
    viewMode: u32,
    lightPosition: vec3f,
    // Metres; 0 means no range window.
    lightRange: f32,
    // Colour times intensity (candela).
    lightIntensity: vec3f,
    lightEnabled: u32,
    // Irradiance from the environment, used where GI coverage is low.
    fallbackIrradiance: vec3f,
    // How far a shadow lookup moves off the surface, metres.
    shadowNormalOffset: f32,
    // The constant irradiance of the 'flat' GI source.
    flatIrradiance: vec3f,
    debugView: u32,
    inverseViewportSize: vec2f,
    shadowViewProjection: array<mat4x4f, ${SHADOW_FACES.length}>,
}

struct DrawUniforms {
    model: mat4x4f,
    // inverse(transpose(model)), for normals.
    normalMatrix: mat3x3f,
    albedo: vec3f,
    // 1 when the model matrix mirrors, which flips triangle winding on screen.
    mirrored: u32,
    emission: vec3f,
}
`;

/** The forward pass's own bindings, group 0. */
export const FORWARD_BINDING = { frame: 0, draw: 1, shadowMap: 2, shadowSampler: 3 } as const;
/** The group `globalsCode` is written at. */
export const GI_GROUP = 1;

function giSourceWgsl(source: GiSource): string {
    if (source.kind === 'module') {
        return /* wgsl */ `
${source.globalsCode}
fn helloGiIrradiance(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f) -> vec3f {
    return giIrradianceAt(position, normal, viewDirection, screenUV);
}
fn helloGiCoverage(position: vec3f, normal: vec3f) -> f32 {
    return giCoverageAt(position, normal);
}
fn helloGiDebugColor(position: vec3f, normal: vec3f, view: u32, screenUV: vec2f) -> vec3f {
    return giDebugColorAt(position, normal, view, screenUV);
}
`;
    }
    return /* wgsl */ `
fn helloGiIrradiance(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f) -> vec3f {
    return frame.flatIrradiance;
}
fn helloGiCoverage(position: vec3f, normal: vec3f) -> f32 {
    return 1.0;
}
fn helloGiDebugColor(position: vec3f, normal: vec3f, view: u32, screenUV: vec2f) -> vec3f {
    return frame.flatIrradiance * HELLO_INV_PI;
}
`;
}

export function forwardShader(source: GiSource): string {
    return /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(${FORWARD_BINDING.frame}) var<uniform> frame: FrameUniforms;
@group(0) @binding(${FORWARD_BINDING.draw}) var<uniform> draw: DrawUniforms;
@group(0) @binding(${FORWARD_BINDING.shadowMap}) var shadowMap: texture_depth_2d_array;
@group(0) @binding(${FORWARD_BINDING.shadowSampler}) var shadowSampler: sampler_comparison;

const VIEW_MODE_FULL: u32 = ${VIEW_MODE.full}u;
const VIEW_MODE_GI: u32 = ${VIEW_MODE.gi}u;
const SHADOW_FACE_POSITIVE_X: u32 = 0u;
const SHADOW_FACE_NEGATIVE_X: u32 = 1u;
const SHADOW_FACE_POSITIVE_Y: u32 = 2u;
const SHADOW_FACE_NEGATIVE_Y: u32 = 3u;
const SHADOW_FACE_POSITIVE_Z: u32 = 4u;
const SHADOW_FACE_NEGATIVE_Z: u32 = 5u;

${giSourceWgsl(source)}

struct VertexOutput {
    // Invariant: the depth prepass computes the same position the same way,
    // and the forward pass tests depth for equality against it.
    @builtin(position) @invariant clipPosition: vec4f,
    @location(0) worldPosition: vec3f,
    @location(1) worldNormal: vec3f,
}

@vertex
fn helloVertex(@location(0) position: vec3f, @location(1) normal: vec3f) -> VertexOutput {
    let world = draw.model * vec4f(position, 1.0);
    var output: VertexOutput;
    output.clipPosition = frame.viewProjection * world;
    output.worldPosition = world.xyz;
    output.worldNormal = draw.normalMatrix * normal;
    return output;
}

// The cube face whose frustum holds this direction: the axis it points along most.
fn helloShadowFace(direction: vec3f) -> u32 {
    let size = abs(direction);
    if (size.x >= size.y && size.x >= size.z) {
        return select(SHADOW_FACE_NEGATIVE_X, SHADOW_FACE_POSITIVE_X, direction.x > 0.0);
    }
    if (size.y >= size.z) {
        return select(SHADOW_FACE_NEGATIVE_Y, SHADOW_FACE_POSITIVE_Y, direction.y > 0.0);
    }
    return select(SHADOW_FACE_NEGATIVE_Z, SHADOW_FACE_POSITIVE_Z, direction.z > 0.0);
}

// 1 where the light sees the point, 0 in shadow (2x2 filtered at edges).
// The point is projected with the same matrix its face was rendered with,
// so no cube-map orientation convention is involved.
fn helloShadow(position: vec3f, normal: vec3f) -> f32 {
    let lookup = position + normal * frame.shadowNormalOffset;
    let face = helloShadowFace(lookup - frame.lightPosition);
    let clip = frame.shadowViewProjection[face] * vec4f(lookup, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return textureSampleCompareLevel(shadowMap, shadowSampler, uv, i32(face), ndc.z);
}

// The same falloff the platform's hit shading uses (three.js r187, decay 2):
// 1 / max(d^2, 0.01), times (clamp(1 - (d / range)^4, 0, 1))^2 when range > 0.
fn helloDistanceAttenuation(lightDistance: f32, range: f32) -> f32 {
    let inverseSquare = 1.0 / max(lightDistance * lightDistance, 0.01);
    if (range <= 0.0) {
        return inverseSquare;
    }
    let ratio = lightDistance / range;
    let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    return inverseSquare * window * window;
}

// Irradiance straight from the point light: I / d^2 x cos, shadowed.
fn helloDirectIrradiance(position: vec3f, normal: vec3f) -> vec3f {
    if (frame.lightEnabled == 0u) {
        return vec3f(0.0);
    }
    let toLight = frame.lightPosition - position;
    let lightDistance = length(toLight);
    let cosine = dot(normal, toLight / max(lightDistance, 1.0e-20));
    if (cosine <= 0.0) {
        return vec3f(0.0);
    }
    return frame.lightIntensity * (helloDistanceAttenuation(lightDistance, frame.lightRange) * cosine * helloShadow(position, normal));
}

@fragment
fn helloFragment(input: VertexOutput, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4f {
    // The side of the surface we see, as the tracer reports it: the authored
    // front unless winding was flipped by a mirroring transform.
    let seesAuthoredFront = frontFacing != (draw.mirrored != 0u);
    let interpolated = normalize(input.worldNormal);
    let normal = select(-interpolated, interpolated, seesAuthoredFront);
    let position = input.worldPosition;
    // From the eye towards the point, as the shading module expects.
    let viewDirection = normalize(position - frame.cameraPosition);
    let screenUV = input.clipPosition.xy * frame.inverseViewportSize;

    if (frame.viewMode == VIEW_MODE_GI) {
        return vec4f(draw.albedo * HELLO_INV_PI * helloGiIrradiance(position, normal, viewDirection, screenUV), 1.0);
    }
    if (frame.viewMode != VIEW_MODE_FULL) {
        return vec4f(helloGiDebugColor(position, normal, frame.debugView, screenUV), 1.0);
    }

    var irradiance = helloDirectIrradiance(position, normal) + helloGiIrradiance(position, normal, viewDirection, screenUV);
    // Where GI isn't sure of its answer (volume edges), fade to the environment.
    if (any(frame.fallbackIrradiance > vec3f(0.0))) {
        irradiance += (1.0 - helloGiCoverage(position, normal)) * frame.fallbackIrradiance;
    }
    return vec4f(draw.emission + draw.albedo * HELLO_INV_PI * irradiance, 1.0);
}
`;
}

/**
 * The depth prepass: depth only, so the forward pass shades each pixel once
 * (depth test 'equal'), whatever order the draws come in. The position is
 * computed exactly as in helloVertex, both @invariant.
 */
export const DEPTH_PREPASS_SHADER = /* wgsl */ `
${COMMON_WGSL}
@group(0) @binding(${FORWARD_BINDING.frame}) var<uniform> frame: FrameUniforms;
@group(0) @binding(${FORWARD_BINDING.draw}) var<uniform> draw: DrawUniforms;

struct DepthOutput {
    @builtin(position) @invariant clipPosition: vec4f,
}

@vertex
fn helloDepthVertex(@location(0) position: vec3f) -> DepthOutput {
    let world = draw.model * vec4f(position, 1.0);
    var output: DepthOutput;
    output.clipPosition = frame.viewProjection * world;
    return output;
}
`;

/** The shadow pass's bindings, group 0 (both with dynamic offsets). */
export const SHADOW_BINDING = { face: 0, draw: 1 } as const;

export const SHADOW_SHADER = /* wgsl */ `
${COMMON_WGSL}
struct ShadowFaceUniforms {
    viewProjection: mat4x4f,
}
@group(0) @binding(${SHADOW_BINDING.face}) var<uniform> face: ShadowFaceUniforms;
@group(0) @binding(${SHADOW_BINDING.draw}) var<uniform> draw: DrawUniforms;

@vertex
fn helloShadowVertex(@location(0) position: vec3f) -> @builtin(position) vec4f {
    return face.viewProjection * (draw.model * vec4f(position, 1.0));
}
`;

/** The display pass: the HDR image, exposed and tone-mapped, written to the canvas. */
export const DISPLAY_SHADER = /* wgsl */ `
struct DisplayUniforms {
    // 2^EV.
    exposureScale: f32,
}
@group(0) @binding(0) var hdrImage: texture_2d<f32>;
@group(0) @binding(1) var<uniform> display: DisplayUniforms;

// One triangle that covers the screen.
@vertex
fn displayVertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let corner = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

// Khronos PBR Neutral tone mapping: leaves colours alone up to about 0.8,
// then compresses highlights towards white without shifting hue much.
fn toneMapNeutral(input: vec3f) -> vec3f {
    let startCompression = 0.8 - 0.04;
    let desaturation = 0.15;
    let lowest = min(input.r, min(input.g, input.b));
    let offset = select(0.04, lowest - 6.25 * lowest * lowest, lowest < 0.08);
    var color = input - offset;
    let peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) {
        return color;
    }
    let headroom = 1.0 - startCompression;
    let newPeak = 1.0 - headroom * headroom / (peak + headroom - startCompression);
    color *= newPeak / peak;
    let blend = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3f(newPeak), blend);
}

fn encodeSrgb(linear: vec3f) -> vec3f {
    let low = linear * 12.92;
    let high = 1.055 * pow(linear, vec3f(1.0 / 2.4)) - 0.055;
    return select(high, low, linear <= vec3f(0.0031308));
}

@fragment
fn displayFragment(@builtin(position) pixel: vec4f) -> @location(0) vec4f {
    let hdr = textureLoad(hdrImage, vec2i(pixel.xy), 0).rgb * display.exposureScale;
    return vec4f(encodeSrgb(saturate(toneMapNeutral(max(hdr, vec3f(0.0))))), 1.0);
}
`;
