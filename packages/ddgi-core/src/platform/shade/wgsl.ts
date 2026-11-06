/**
 * WGSL of the hit-shading module (contract/shade.ts): the surface at a ray
 * hit, direct light with shadow rays, and the sky.
 *
 * ```wgsl
 * fn giSurfaceAt(ray: GiRay, hit: GiHit) -> GiSurface
 * fn giDirectIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> vec3f
 * fn giDirectReaches(surface: GiSurface, groupMask: u32) -> bool   // any light in the groups lights this side (no shadow rays)
 * fn giDirectRandom(ray: u32, frame: u32) -> vec2f                  // gi/direct's random numbers for a ray
 * fn giSkyRadiance(direction: vec3f) -> vec3f
 * fn giSkyIrradiance(surface: GiSurface) -> vec3f
 * fn giSkyIrradianceAt(normal: vec3f) -> vec3f        // the same, for any unit normal
 * ```
 *
 * It depends on the scene copy (lights, materials, triangles) and the `rays`
 * module (`giTraceAny` for shadow rays), and binds no storage buffer of its
 * own: one uniform buffer (settings and the sky's irradiance SH), the sky's
 * radiance map and a sampler. So the scene copy and hit shading together stay
 * at the scene copy's 2 storage buffers (docs/api/platform.md, binding budget).
 *
 * Light units are the platform's (contract/scene.ts), which are three.js's
 * physically based ones; point and spot falloff and spot cones follow three.js
 * r187 exactly, so the three adapter matches what three renders.
 */

import type { BindingSlot, ModuleBindingEntry } from '../contract/wgsl.ts';
import { RADIANCE_SH_COUNT, SKY_HARMONICS_LAYOUT } from '../sky/projection-wgsl.ts';
import { SH_BASIS_WGSL, SH_COEFFICIENT_COUNT } from '../sky/spherical-harmonics.ts';

/** `GPUShaderStage.COMPUTE`. The global does not exist in Node, where the tests import this file. */
const SHADER_STAGE_COMPUTE = 0x4;

/** The module's bindings, in order: uniforms (settings and sky SH), the sky's radiance map, its sampler. No storage buffers. */
export const SHADE_MODULE_ENTRIES: readonly ModuleBindingEntry[] = [
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
    { visibility: SHADER_STAGE_COMPUTE, sampler: { type: 'filtering' } },
];

/**
 * Byte layout of `GiShadeUniforms`: the sky's harmonics first (radiance SH
 * bands 0-2, then irradiance SH; written by the sky bake, which binds that
 * range as storage, `SKY_HARMONICS_LAYOUT`), then the live settings (written
 * with `queue.writeBuffer`).
 */
export const SHADE_UNIFORM_LAYOUT = {
    skyHarmonics: 0,
    skyRadiance: SKY_HARMONICS_LAYOUT.radiance,
    skyIrradiance: SKY_HARMONICS_LAYOUT.irradiance,
    maxShadowedLightsPerHit: SKY_HARMONICS_LAYOUT.byteLength,
    bounceAlbedoCap: SKY_HARMONICS_LAYOUT.byteLength + 4,
    /** Rounded up to the struct's 16-byte alignment. */
    byteLength: SKY_HARMONICS_LAYOUT.byteLength + 16,
} as const;

/** The module's source, with its bindings at `slot`. */
export function shadeModuleWgsl(slot: BindingSlot): string {
    const group = slot.group;
    const uniformsBinding = slot.firstBinding;
    const radianceMapBinding = slot.firstBinding + 1;
    const samplerBinding = slot.firstBinding + 2;
    const skyIrradianceTerms = Array.from({ length: SH_COEFFICIENT_COUNT }, (_, index) => `        giShade.skyIrradiance[${index}].xyz * basis[${index}]`).join(' +\n');

    return /* wgsl */ `
// ---- Hit shading (platform/shade) ------------------------------------------------------

struct GiSurface {
    position: vec3f,        // world space, on the triangle, nudged 64 rounding steps in from its edges (see giSurfaceAt)
    normal: vec3f,          // shading normal: the interpolated vertex normal if the mesh has normals, facing the ray's side
    geometricNormal: vec3f, // the triangle's own unit normal, out of the authored front side (for front/back decisions)
    uv: vec2f,              // uv set 0, interpolated; (0, 0) without uvs
    albedo: vec3f,          // diffuse albedo, capped per channel at bounceAlbedoCap
    emission: vec3f,        // emitted radiance
    material: u32,
    instance: u32,
    frontFace: bool,        // the ray hit the authored front side (also on mirrored instances)
}

struct GiShadeUniforms {
    // The sky's radiance as SH, bands 0-2 (platform/sky), for techniques that
    // follow the sky's own shape (the instant sky). rgb per coefficient; w unused.
    skyRadiance: array<vec4f, ${RADIANCE_SH_COUNT}>,
    // The sky's irradiance as SH, bands 0, 1, 2 and 4, already convolved with
    // the cosine lobe (platform/sky). rgb per coefficient; w unused.
    skyIrradiance: array<vec4f, ${SH_COEFFICIENT_COUNT}>,
    maxShadowedLightsPerHit: u32,
    bounceAlbedoCap: f32,
}

@group(${group}) @binding(${uniformsBinding}) var<uniform> giShade: GiShadeUniforms;
// The sky seen by rays that miss: equirect, first row the nadir (three.js's equirectUV).
@group(${group}) @binding(${radianceMapBinding}) var giSkyRadianceMap: texture_2d<f32>;
// Linear, repeating across the -X seam, clamped at the poles.
@group(${group}) @binding(${samplerBinding}) var giSkySampler: sampler;

const GI_SHADE_INV_PI: f32 = 0.318309886183791;
const GI_SHADE_INV_TWO_PI: f32 = 0.159154943091895;
const GI_SHADE_TWO_PI: f32 = 6.28318530717959;
// tMax of a shadow ray towards a directional light: everything up to "infinitely" far blocks it.
const GI_SHADE_UNBOUNDED_DISTANCE: f32 = 1.0e27;
// One ulp of 1.0 in f32, 2^-23.
const GI_SHADE_ULP: f32 = 1.1920929e-7;
// How far a shading point moves into its triangle (giSurfaceAt), and a shadow
// ray's origin off the surface (giShadeOffsetRayOrigin), in rounding steps of
// the point's largest coordinate. Both well above the few steps of error in
// the point; the in-surface nudge smaller, as it moves where shading happens.
const GI_SHADE_IN_SURFACE_STEPS: f32 = 64.0;
const GI_SHADE_NORMAL_STEPS: f32 = 256.0;

// ---- The surface at a hit ---------------------------------------------------------------

fn giSurfaceAt(ray: GiRay, hit: GiHit) -> GiSurface {
    // The point from the barycentrics on the world-space triangle rather than
    // ray.origin + t * ray.direction: its error depends on the triangle's own
    // coordinates, not on how far the ray travelled.
    let corners = giInstanceTriangle(hit.instance, hit.triangle);
    let weights = clamp(vec3f(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y), vec3f(0.0), vec3f(1.0));
    let onTriangle = corners.vertex0 * weights.x + corners.vertex1 * weights.y + corners.vertex2 * weights.z;
    // Then moved a few dozen rounding steps into the triangle, towards its
    // centroid. Why: at an edge or corner where faces of a closed shell meet,
    // the rounded point can land a hair outside the neighbouring face, and a
    // shadow ray from there escapes the shell. On the trace oracle's thin box
    // that let 25 of 1,200 corner hits see a light outside. Inside its own
    // triangle by more than the rounding, the point is inside the shell.
    let toCentroid = (corners.vertex0 + corners.vertex1 + corners.vertex2) / 3.0 - onTriangle;
    let centroidDistance = length(toCentroid);
    let nudge = min(GI_SHADE_IN_SURFACE_STEPS * giShadeRoundingStep(onTriangle), centroidDistance);
    let position = onTriangle + toCentroid * (nudge / max(centroidDistance, 1.0e-30));

    let geometricNormal = giInstanceGeometricNormal(hit.instance, hit.triangle);
    // Vertex normals point out of the authored front side, like the geometric
    // normal; on a back-face hit they are turned to the side the ray came from.
    let interpolated = giInstanceNormalAt(hit.instance, hit.triangle, hit.barycentrics);
    let normal = select(-interpolated, interpolated, hit.frontFace);

    let materialIndex = giMaterialAt(hit.instance, hit.triangle);
    let material = giMaterial(materialIndex);
    return GiSurface(
        position,
        normal,
        geometricNormal,
        giInstanceUvAt(hit.instance, hit.triangle, hit.barycentrics, 0u),
        min(material.albedo, vec3f(giShade.bounceAlbedoCap)),
        material.emission,
        materialIndex,
        hit.instance,
        hit.frontFace,
    );
}

// The geometric normal turned to the side of the surface the ray arrived on.
// Shadow rays leave from that side, and light from the other side can't reach it.
fn giShadeRaySide(surface: GiSurface) -> vec3f {
    return select(-surface.geometricNormal, surface.geometricNormal, surface.frontFace);
}

// The size of one rounding step (an ulp) of the largest coordinate of a
// point: the scale of the error in a hit point reconstructed there. Below
// 1/16 m from the origin it stays at that of 1/16, as ulps vanish at 0.
fn giShadeRoundingStep(point: vec3f) -> f32 {
    let magnitude = max(max(abs(point.x), abs(point.y)), max(abs(point.z), 0.0625));
    return magnitude * GI_SHADE_ULP;
}

// Where a shadow ray starts: the shading point pushed off the surface along
// the geometric normal of the side the ray arrived on.
//
// Why: the point carries rounding error, so a ray started exactly on it can
// hit its own triangle and shadow a lit point. The tracer's front faces follow
// the authored side (DECISIONS 2026-11-04), which on a mirrored instance or a
// back-face hit is not the side the ray is on, so the push uses the side from
// frontFace (giShadeRaySide), not the authored normal.
//
// How far: the error grows with the size of the coordinates, so the push is
// counted in rounding steps of the largest coordinate (256 of them), after
// Wachter and Binder, "A Fast and Robust Method for Avoiding
// Self-Intersection" (Ray Tracing Gems, 2019, chapter 6). Unlike their
// per-coordinate integer offset, the push here is along the normal exactly:
// theirs scales each coordinate's step by that coordinate's own size, which
// tilts the push along the surface when the coordinates differ in size, and
// on the oracle's thin box that tilt carried 8 corner points past the
// neighbouring face. 256 steps are 0.06 mm at 2 m from the origin, 0.3 mm at
// 10 m, 3 mm at 100 m: below the oracle's 1 mm wall pairs near the origin,
// not below them 100 m out.
fn giShadeOffsetRayOrigin(position: vec3f, normal: vec3f) -> vec3f {
    return position + normal * (GI_SHADE_NORMAL_STEPS * giShadeRoundingStep(position));
}

// ---- Lights -------------------------------------------------------------------------------

// One light's unshadowed contribution at a surface, and where to aim its shadow ray.
struct GiShadeLightSample {
    irradiance: vec3f,  // unshadowed irradiance on the surface; 0 when the light can't reach this side
    direction: vec3f,   // unit, from the surface towards the light (its centre)
}

// giShadeDistanceAttenuation and giShadeSpotAttenuation (three's point and spot falloff)
// come from the light-falloff module (light-falloff.ts), which this module depends on,
// so DDGI's change estimate uses exactly the same falloff.

fn giShadeLightSample(light: GiLight, surface: GiSurface, raySide: vec3f) -> GiShadeLightSample {
    var direction: vec3f;
    var irradiance: vec3f;
    if (light.kind == GI_LIGHT_DIRECTIONAL) {
        // Directional intensity is irradiance at normal incidence.
        direction = -light.direction;
        irradiance = light.intensity;
    } else {
        let toLight = light.position - surface.position;
        let lightDistance = length(toLight);
        direction = toLight / max(lightDistance, 1.0e-20);
        // Point and spot intensity is radiant intensity: irradiance = I / d^2, windowed.
        var attenuation = giShadeDistanceAttenuation(lightDistance, light.range);
        if (light.kind == GI_LIGHT_SPOT) {
            attenuation *= giShadeSpotAttenuation(dot(-direction, light.direction), light.cosOuterCone, light.cosInnerCone);
        }
        irradiance = light.intensity * attenuation;
    }
    // Light from behind the surface (as seen from the ray's side) can't reach
    // it. Checked on the geometric normal too, so a smooth shading normal
    // can't pull light through a wall.
    let cosine = dot(surface.normal, direction);
    let reachesSide = dot(raySide, direction) > 0.0;
    return GiShadeLightSample(select(vec3f(0.0), irradiance * max(cosine, 0.0), reachesSide), direction);
}

fn giShadeInGroup(light: GiLight, groupMask: u32) -> bool {
    return ((groupMask >> light.group) & 1u) != 0u;
}

// Rec. 709 luminance: how much a light is estimated to contribute, for the stochastic choice.
fn giShadeLuminance(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

// PCG hash (Jarzynski and Olano, "Hash Functions for GPU Rendering", JCGT 2020).
fn giShadeHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Two numbers in [0, 1) for one light's soft-shadow sample, decorrelated from
// the caller's random (used as is for the stochastic light choice) and between lights.
fn giShadeLightRandom(random: vec2f, light: u32) -> vec2f {
    let first = giShadeHash(bitcast<u32>(random.x) ^ giShadeHash(bitcast<u32>(random.y) + light));
    let second = giShadeHash(first);
    return vec2f(f32(first >> 8u), f32(second >> 8u)) * (1.0 / 16777216.0);
}

// Two unit vectors perpendicular to n and to each other (Duff et al.,
// "Building an Orthonormal Basis, Revisited", JCGT 2017).
fn giShadeTangents(n: vec3f) -> mat2x3f {
    let signZ = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (signZ + n.z);
    let b = n.x * n.y * a;
    return mat2x3f(vec3f(1.0 + signZ * n.x * n.x * a, signZ * b, -signZ * n.x), vec3f(b, signZ + n.y * n.y * a, -n.y));
}

// Is the light visible from the shadow-ray origin? Lights that don't cast
// shadows skip the ray. Lights with a size (a directional light's angular
// radius, a point or spot light's radius) aim at a random point of it, so
// many samples average to a soft shadow.
fn giShadeVisibility(light: GiLight, lightIndex: u32, lightSample: GiShadeLightSample, origin: vec3f, random: vec2f) -> f32 {
    if (!light.castsShadows) {
        return 1.0;
    }
    let jitter = giShadeLightRandom(random, lightIndex);
    let tangents = giShadeTangents(lightSample.direction);
    let angle = GI_SHADE_TWO_PI * jitter.y;
    var direction: vec3f;
    var rayLength: f32;
    if (light.kind == GI_LIGHT_DIRECTIONAL) {
        // Uniform over the cone of half-angle 'radius' around the light's direction.
        let cosTheta = 1.0 - jitter.x * (1.0 - cos(light.radius));
        let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
        direction = normalize(lightSample.direction * cosTheta + (tangents[0] * cos(angle) + tangents[1] * sin(angle)) * sinTheta);
        rayLength = GI_SHADE_UNBOUNDED_DISTANCE;
    } else {
        // A uniform point on the disc of radius 'radius' facing the surface.
        let discRadius = light.radius * sqrt(jitter.x);
        let aimPoint = light.position + (tangents[0] * cos(angle) + tangents[1] * sin(angle)) * discRadius;
        let toAimPoint = aimPoint - origin;
        rayLength = length(toAimPoint);
        direction = toAimPoint / max(rayLength, 1.0e-20);
    }
    let blocked = giTraceAny(GiRay(origin, 0.0, direction, rayLength));
    return select(1.0, 0.0, blocked);
}

// Direct light from the lights whose group bit is set in groupMask, as
// irradiance, shadow rays included.
//
// Directional lights are always evaluated. If more point and spot lights reach
// the surface than giShade.maxShadowedLightsPerHit (K), K of them are chosen at
// random with probability proportional to their unshadowed contribution
// (luminance), by systematic sampling: one random number r = random.x places K
// evenly spaced points (k + r) / K on the lights' cumulative distribution. A
// light picked n times counts n * f / (K p), where p = its share of the total,
// so the sum is unbiased; a light with p >= 1/K is always picked. With K or
// fewer contributing lights every one is evaluated and the answer has no noise.
fn giDirectIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> vec3f {
    let raySide = giShadeRaySide(surface);
    let origin = giShadeOffsetRayOrigin(surface.position, raySide);
    let lightCount = giLightCount();
    let directionalCount = min(giDirectionalLightCount(), lightCount);

    // First pass over the local lights: how many reach the surface, and their total estimated contribution.
    var contributing = 0u;
    var weightSum = 0.0;
    for (var index = directionalCount; index < lightCount; index++) {
        let light = giLight(index);
        if (!giShadeInGroup(light, groupMask)) {
            continue;
        }
        let weight = giShadeLuminance(giShadeLightSample(light, surface, raySide).irradiance);
        if (weight > 0.0) {
            contributing++;
            weightSum += weight;
        }
    }
    let budget = max(giShade.maxShadowedLightsPerHit, 1u);
    let chooseLocalLights = contributing > budget;
    let picksPerHit = f32(budget);

    // Second pass, over every light; one shadow-ray call site keeps the tracer inlined once.
    var total = vec3f(0.0);
    var cumulative = 0.0;
    var seen = 0u;
    for (var index = 0u; index < lightCount; index++) {
        let light = giLight(index);
        if (!giShadeInGroup(light, groupMask)) {
            continue;
        }
        let lightSample = giShadeLightSample(light, surface, raySide);
        let weight = giShadeLuminance(lightSample.irradiance);
        if (!(weight > 0.0)) {
            continue;
        }
        var scale = 1.0;
        if (index >= directionalCount && chooseLocalLights) {
            seen++;
            let start = cumulative / weightSum;
            cumulative += weight;
            // The last light's interval ends at exactly 1, whatever the rounding of the running sum.
            let end = select(cumulative / weightSum, 1.0, seen == contributing);
            // Points (k + r) / K in [start, end): k from ceil(start K - r) up to ceil(end K - r).
            let picks = ceil(end * picksPerHit - random.x) - ceil(start * picksPerHit - random.x);
            if (picks <= 0.0) {
                continue;
            }
            scale = picks * weightSum / (picksPerHit * weight);
        }
        total += lightSample.irradiance * (scale * giShadeVisibility(light, index, lightSample, origin, random));
    }
    return total;
}

// Can any light whose group bit is set in groupMask light this surface at
// all? True when one of them reaches the side the ray arrived on with a
// non-zero unshadowed contribution: exactly the lights giDirectIrradiance
// would evaluate. No shadow rays. A tracing kernel asks this before putting
// a hit on gi/direct's work list, so hits facing away from every light
// (about half of Sponza's, which face away from the sun; T2.0) cost nothing
// there.
fn giDirectReaches(surface: GiSurface, groupMask: u32) -> bool {
    let raySide = giShadeRaySide(surface);
    let lightCount = giLightCount();
    for (var index = 0u; index < lightCount; index++) {
        let light = giLight(index);
        if (giShadeInGroup(light, groupMask) && giShadeLuminance(giShadeLightSample(light, surface, raySide).irradiance) > 0.0) {
            return true;
        }
    }
    return false;
}

// The two numbers in [0, 1) gi/direct hands giDirectIrradiance for the hit of
// ray 'ray' (an index into a ray scratch's results) in frame 'frame'. They
// follow the ray, not the hit's place on the work list, so the subgroup and
// the fallback work lists (which order hits differently) give the same light.
// A kernel that calls giDirectIrradiance itself with these numbers gets
// gi/direct's answer for the same surface.
fn giDirectRandom(ray: u32, frame: u32) -> vec2f {
    let first = giShadeHash(ray ^ giShadeHash(frame));
    let second = giShadeHash(first ^ 0x9e3779b9u);
    return vec2f(f32(first >> 8u), f32(second >> 8u)) * (1.0 / 16777216.0);
}

// ---- Sky ----------------------------------------------------------------------------------

${SH_BASIS_WGSL}

// Radiance seen along a ray that missed everything.
fn giSkyRadiance(direction: vec3f) -> vec3f {
    let d = normalize(direction);
    let uv = vec2f(atan2(d.z, d.x) * GI_SHADE_INV_TWO_PI + 0.5, asin(clamp(d.y, -1.0, 1.0)) * GI_SHADE_INV_PI + 0.5);
    return textureSampleLevel(giSkyRadianceMap, giSkySampler, uv, 0.0).rgb;
}

// Irradiance from the whole sky on a surface facing 'normal' (unit), with
// nothing in the way: the SH sum. Truncating at band 4 leaves under 0.35% on
// flat ground under the shared skies; clamped at 0 because a truncated series
// can dip below it where the sky is black.
fn giSkyIrradianceAt(normal: vec3f) -> vec3f {
    let basis = giSkyShBasis(normal);
    let irradiance =
${skyIrradianceTerms};
    return max(irradiance, vec3f(0.0));
}

// Irradiance from the sky alone at a surface (for the sky light group), unshadowed.
fn giSkyIrradiance(surface: GiSurface) -> vec3f {
    return giSkyIrradianceAt(surface.normal);
}
`;
}
