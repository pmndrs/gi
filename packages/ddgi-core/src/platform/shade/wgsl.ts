/**
 * WGSL of the hit-shading module (contract/shade.ts): the surface at a ray
 * hit, direct light with shadow rays, and the sky.
 *
 * ```wgsl
 * fn giSurfaceAt(ray: GiRay, hit: GiHit) -> GiSurface
 * fn giDirectIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> vec3f
 * fn giDirectGroupIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> GiDirectGroups  // the same, split by light group 0-3
 * fn giDirectSplitIrradiance(surface: GiSurface, groupMask: u32, sunApart: bool, random: vec2f) -> GiDirectGroups  // and the sun apart, in white
 * fn giDirectReaches(surface: GiSurface, groupMask: u32) -> bool   // any light in the groups lights this side (no shadow rays)
 * fn giDirectReachesSplit(surface: GiSurface, groupMask: u32, sunApart: bool) -> bool  // the same, the sun in white when apart
 * fn giDirectRandom(ray: u32, frame: u32) -> vec2f                  // gi/direct's random numbers for a ray
 * fn giDirectChooseLight(surface: GiSurface, groupMask: u32, random: f32) -> GiDirectLightChoice  // one light drawn by weight, and its weight
 * fn giDirectChooseLightWithSky(surface: GiSurface, groupMask: u32, random: f32, skyRandom: vec2f) -> GiDirectLightChoice  // the same, the sky one more candidate
 * fn giDirectLightRay(surface: GiSurface, light: u32, random: vec2f) -> GiDirectLightRay           // its unshadowed irradiance and shadow ray
 * fn giSkyRadiance(direction: vec3f) -> vec3f
 * fn giSkyLightSample(surface: GiSurface, random: vec2f) -> GiSkyLightSample  // the sky as a light: one direction, its irradiance estimate and shadow ray
 * fn giSkyIrradiance(surface: GiSurface) -> vec3f
 * fn giSkyIrradianceAt(normal: vec3f) -> vec3f        // the same, for any unit normal
 * ```
 *
 * It depends on the scene copy (lights, materials, triangles) and the `rays`
 * module (`giTraceAny` for shadow rays), and binds no storage buffer of its
 * own: one uniform buffer (settings, the light grid's place and the sky's
 * irradiance SH), the sky's radiance map, a sampler and the light grid's
 * picks (a texture, light-grid.ts). So the scene copy and hit shading
 * together stay at the scene copy's 2 storage buffers (docs/api/platform.md,
 * binding budget).
 *
 * Light units are the platform's (contract/scene.ts), which are three.js's
 * physically based ones; point and spot falloff and spot cones follow three.js
 * r187 exactly, so the three adapter matches what three renders.
 */

import { GI_DIRECT_SKY_LIGHT } from '../contract/shade.ts';
import type { BindingSlot, ModuleBindingEntry } from '../contract/wgsl.ts';
import { lightGridLookupWgsl } from './light-grid.ts';
import { RADIANCE_SH_COUNT, SKY_HARMONICS_LAYOUT } from '../sky/projection-wgsl.ts';
import { SH_BASIS_WGSL, SH_COEFFICIENT_COUNT } from '../sky/spherical-harmonics.ts';

/** `GPUShaderStage.COMPUTE`. The global does not exist in Node, where the tests import this file. */
const SHADER_STAGE_COMPUTE = 0x4;

/** The module's bindings, in order: uniforms (settings and sky SH), the sky's radiance map, its sampler, the light grid's picks. No storage buffers. */
export const SHADE_MODULE_ENTRIES: readonly ModuleBindingEntry[] = [
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
    { visibility: SHADER_STAGE_COMPUTE, sampler: { type: 'filtering' } },
    { visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } },
];

/**
 * Byte layout of `GiShadeUniforms`: the sky's harmonics first (radiance SH
 * bands 0-2, irradiance SH, the radiance map's scale; written by the sky
 * bake, which binds that range as storage, `SKY_HARMONICS_LAYOUT`), then the
 * live settings (written with `queue.writeBuffer`).
 */
export const SHADE_UNIFORM_LAYOUT = {
    skyHarmonics: 0,
    skyRadiance: SKY_HARMONICS_LAYOUT.radiance,
    skyIrradiance: SKY_HARMONICS_LAYOUT.irradiance,
    skyRadianceMapScale: SKY_HARMONICS_LAYOUT.radianceMapScale,
    maxShadowedLightsPerHit: SKY_HARMONICS_LAYOUT.byteLength,
    bounceAlbedoCap: SKY_HARMONICS_LAYOUT.byteLength + 4,
    /** The light grid's lookup (light-grid.ts), 40 bytes from here: active, shadowed lights per hit with the grid, origin and 1 / cell size, cells and the outside cell. */
    lightGrid: SKY_HARMONICS_LAYOUT.byteLength + 8,
    /** Rounded up to the struct's 16-byte alignment. */
    byteLength: SKY_HARMONICS_LAYOUT.byteLength + 48,
} as const;

/** The module's source, with its bindings at `slot`. */
export function shadeModuleWgsl(slot: BindingSlot): string {
    const group = slot.group;
    const uniformsBinding = slot.firstBinding;
    const radianceMapBinding = slot.firstBinding + 1;
    const samplerBinding = slot.firstBinding + 2;
    const lightGridBinding = slot.firstBinding + 3;
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
    // x: what the radiance map is read times (1 but for a procedural sky,
    // whose map is written before its calibration is known; platform/sky).
    skyRadianceMapScale: vec4f,
    maxShadowedLightsPerHit: u32,
    bounceAlbedoCap: f32,
    // The light grid (light-grid.ts): 1 when this frame's point and spot
    // lights are drawn from it (more of them than exactLightLimit), and where
    // its cells are.
    lightGridActive: u32,
    lightGridShadowedLightsPerHit: u32,  // K with the grid (the lightGridShadowedLightsPerHit setting)
    lightGridOrigin: vec3f,
    lightGridInverseCellSize: f32,
    lightGridCells: vec3u,
    lightGridOutsideCell: u32,
}

@group(${group}) @binding(${uniformsBinding}) var<uniform> giShade: GiShadeUniforms;
// The sky seen by rays that miss: equirect, first row the nadir (three.js's equirectUV).
@group(${group}) @binding(${radianceMapBinding}) var giSkyRadianceMap: texture_2d<f32>;
// Linear, repeating across the -X seam, clamped at the poles.
@group(${group}) @binding(${samplerBinding}) var giSkySampler: sampler;
// The light grid's picks, R per cell: (light index, its share at a hit as f32 bits).
@group(${group}) @binding(${lightGridBinding}) var giLightGridPicks: texture_2d<u32>;

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
    let blocked = giTraceAny(giShadeShadowRay(light, lightIndex, lightSample, origin, random));
    return select(1.0, 0.0, blocked);
}

// The shadow ray giShadeVisibility traces: from the origin towards a random
// point of the light (its cone for a directional light, its disc for a point
// or spot light), so many samples average to a soft shadow.
fn giShadeShadowRay(light: GiLight, lightIndex: u32, lightSample: GiShadeLightSample, origin: vec3f, random: vec2f) -> GiRay {
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
    return GiRay(origin, 0.0, direction, rayLength);
}

${lightGridLookupWgsl('giLightGridPicks')}

// Is this frame's point and spot light drawn from the light grid? Set by
// the platform when the scene has more of them than exactLightLimit.
fn giShadeUsesLightGrid() -> bool {
    return giShade.lightGridActive != 0u;
}

// One of the point and spot lights a hit weighs: the light, and what its
// unshadowed contribution counts for in the sum (share; 0 = skip it).
struct GiShadeCandidate {
    light: u32,
    share: f32,
}

// The hit's local candidate number 'candidate'. Without the grid, every
// point and spot light, each counting once. With it, the R picks of the
// hit's cell, each counting the share the build stored: a pick drawn with
// probability p is a fair one-sample estimate of the cell's light as f / p,
// so R picks estimate it as the sum of f / (R p); a light picked m times is
// one candidate counting m / (R p) (its other picks store 0), so it is never
// traced twice for one hit.
fn giShadeLocalCandidate(candidate: u32, directionalCount: u32, useGrid: bool, cell: u32) -> GiShadeCandidate {
    if (!useGrid) {
        return GiShadeCandidate(directionalCount + candidate, 1.0);
    }
    let pick = giLightGridPick(cell, candidate);
    if (pick.light == GI_LIGHT_GRID_NO_LIGHT) {
        return GiShadeCandidate(0u, 0.0);
    }
    return GiShadeCandidate(pick.light, pick.share);
}

// Direct light from the lights whose group bit is set in groupMask, as
// irradiance, shadow rays included: the sum of giDirectGroupIrradiance's four
// groups. With every light in one group (the usual case) the sum is exactly
// that group's, light by light in the same order as before light groups.
fn giDirectIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> vec3f {
    let groups = giDirectGroupIrradiance(surface, groupMask, random);
    return groups.group0 + groups.group1 + groups.group2 + groups.group3;
}

// Direct irradiance split by the contract's light groups 0-3 (a light's
// 'group'), for techniques that keep one layer per group (DDGI's light
// groups, T2.5). The light choice and the shadow rays are shared: each
// chosen light's contribution goes to its own group, so every group's sum
// stays unbiased (a group with few lights is noisier).
//
// 'sun' is the sun kept apart (giDirectSplitIrradiance with sunApart; T2.6b):
// the first directional light, at a white reference (colour 1, intensity 1)
// and in no group. A technique that keeps it in a layer of its own weighs
// that layer by the sun's real colour x intensity, so a change of the sun's
// colour or brightness needs no new rays. 0 when not asked for.
struct GiDirectGroups {
    group0: vec3f,
    group1: vec3f,
    group2: vec3f,
    group3: vec3f,
    sun: vec3f,
}

// giDirectSplitIrradiance with every light in its own group, the sun included.
fn giDirectGroupIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> GiDirectGroups {
    return giDirectSplitIrradiance(surface, groupMask, false, random);
}

// Is evaluation 'evaluation' of the light loops below the sun kept apart? The
// sun is the first directional light: directional lights come first in the
// light table (giDirectionalLightCount), and the loops evaluate them first,
// so it is evaluation 0 when the scene has one.
fn giShadeIsSunApart(sunApart: bool, evaluation: u32, directionalCount: u32) -> bool {
    return sunApart && evaluation == 0u && directionalCount > 0u;
}

// A light as the sun's own layer sees it: the same light with its colour x
// intensity replaced by white 1, so the result is per unit of the sun's
// light. The light choice never weighs a directional light (they are always
// evaluated), so this changes no other light's odds; and a sun at intensity
// 0 still lights its layer, ready for the moment it comes back.
fn giShadeAtWhite(light: GiLight) -> GiLight {
    var white = light;
    white.intensity = vec3f(1.0);
    return white;
}

// Directional lights are always evaluated. Point and spot lights are first
// gathered as candidates (giShadeLocalCandidate): every one of them when the
// scene has up to exactLightLimit (the light grid is off), or the R picks of
// the hit's cell from the light grid (light-grid.ts) when it has more. Each
// candidate's unshadowed contribution at the surface, times its share, is
// its weight. If more candidates have a weight than K (maxShadowedLightsPerHit,
// or lightGridShadowedLightsPerHit with the grid), K of them are chosen at
// random with probability proportional to their weight (luminance), by
// systematic sampling: one random number r = random.x places K evenly spaced
// points (k + r) / K on the candidates' cumulative distribution. A candidate
// picked n times counts n * share * f / (K q), where q = its share of the
// total weight, so the sum is unbiased; a candidate with q >= 1/K is always
// picked. With K or fewer weighted candidates every one is traced: without
// the grid the answer then has no noise; with it, what is left is the
// grid's own sampling, which a new grid every frame averages out.
//
// With the grid this is ReGIR's two stages (Boksansky et al., "Rendering
// Many Lights with Grid-Based Reservoirs", Ray Tracing Gems II, 2021): the
// cell's picks follow a bound that knows neither the surface's normal nor
// its exact place, and the second stage spends the shadow rays where the
// real contribution is, so lights behind the surface or at grazing angles
// cost no rays. Each chosen light's contribution goes to its own group, and
// a candidate outside groupMask counts 0, which keeps each group's sum
// unbiased. With sunApart the sun (light 0, directional) is evaluated at its
// white reference and goes into 'sun' instead of its group.
fn giDirectSplitIrradiance(surface: GiSurface, groupMask: u32, sunApart: bool, random: vec2f) -> GiDirectGroups {
    let raySide = giShadeRaySide(surface);
    let origin = giShadeOffsetRayOrigin(surface.position, raySide);
    let lightCount = giLightCount();
    let directionalCount = min(giDirectionalLightCount(), lightCount);
    let useGrid = giShadeUsesLightGrid();
    let cell = select(0u, giLightGridCellAt(surface.position), useGrid);
    let localCandidates = select(lightCount - directionalCount, GI_LIGHT_GRID_PICKS, useGrid);
    // K: maxShadowedLightsPerHit on the exact path, lightGridShadowedLightsPerHit with the grid.
    let budget = max(select(giShade.maxShadowedLightsPerHit, giShade.lightGridShadowedLightsPerHit, useGrid), 1u);
    let picksPerHit = f32(budget);

    // First pass over the local candidates: how many have a weight, and the total.
    var contributing = 0u;
    var weightSum = 0.0;
    for (var candidateNumber = 0u; candidateNumber < localCandidates; candidateNumber++) {
        let candidate = giShadeLocalCandidate(candidateNumber, directionalCount, useGrid, cell);
        if (!(candidate.share > 0.0)) {
            continue;
        }
        let light = giLight(candidate.light);
        if (!giShadeInGroup(light, groupMask)) {
            continue;
        }
        let weight = giShadeLuminance(giShadeLightSample(light, surface, raySide).irradiance) * candidate.share;
        if (weight > 0.0) {
            contributing++;
            weightSum += weight;
        }
    }
    let chooseLocalLights = contributing > budget;

    // Second pass, over the directional lights and the local candidates again;
    // one shadow-ray call site keeps the tracer inlined once.
    let evaluations = directionalCount + localCandidates;
    var totals = GiDirectGroups(vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    var cumulative = 0.0;
    var seen = 0u;
    for (var evaluation = 0u; evaluation < evaluations; evaluation++) {
        var index = evaluation;
        var share = 1.0;
        let isLocal = evaluation >= directionalCount;
        if (isLocal) {
            let candidate = giShadeLocalCandidate(evaluation - directionalCount, directionalCount, useGrid, cell);
            if (!(candidate.share > 0.0)) {
                continue;
            }
            index = candidate.light;
            share = candidate.share;
        }
        let isSun = giShadeIsSunApart(sunApart, evaluation, directionalCount);
        var light = giLight(index);
        if (!giShadeInGroup(light, groupMask)) {
            continue;
        }
        if (isSun) {
            light = giShadeAtWhite(light);
        }
        let lightSample = giShadeLightSample(light, surface, raySide);
        let weight = giShadeLuminance(lightSample.irradiance) * share;
        if (!(weight > 0.0)) {
            continue;
        }
        var scale = share;
        if (isLocal && chooseLocalLights) {
            seen++;
            let start = cumulative / weightSum;
            cumulative += weight;
            // The last candidate's interval ends at exactly 1, whatever the rounding of the running sum.
            let end = select(cumulative / weightSum, 1.0, seen == contributing);
            // Points (k + r) / K in [start, end): k from ceil(start K - r) up to ceil(end K - r).
            let picks = ceil(end * picksPerHit - random.x) - ceil(start * picksPerHit - random.x);
            if (picks <= 0.0) {
                continue;
            }
            scale = share * picks * weightSum / (picksPerHit * weight);
        }
        let contribution = lightSample.irradiance * (scale * giShadeVisibility(light, index, lightSample, origin, random));
        // Selects, not an array indexed by the group: a private array indexed
        // at run time goes to local memory, which is slow (NOTES).
        totals.group0 += select(vec3f(0.0), contribution, light.group == 0u && !isSun);
        totals.group1 += select(vec3f(0.0), contribution, light.group == 1u && !isSun);
        totals.group2 += select(vec3f(0.0), contribution, light.group == 2u && !isSun);
        totals.group3 += select(vec3f(0.0), contribution, light.group == 3u && !isSun);
        totals.sun += select(vec3f(0.0), contribution, isSun);
    }
    return totals;
}

// Can any light whose group bit is set in groupMask light this surface at
// all? True when one of them reaches the side the ray arrived on with a
// non-zero unshadowed contribution: exactly the lights giDirectIrradiance
// would weigh (with the light grid, the directional lights and the cell's R
// picks, so a hit it says no to gets nothing from giDirectIrradiance this
// frame either). No shadow rays. A tracing kernel asks this before putting
// a hit on gi/direct's work list, so hits facing away from every light
// (about half of Sponza's, which face away from the sun; T2.0) cost nothing
// there.
fn giDirectReaches(surface: GiSurface, groupMask: u32) -> bool {
    return giDirectReachesSplit(surface, groupMask, false);
}

// giDirectReaches for giDirectSplitIrradiance: with sunApart the sun is
// weighed at its white reference, so a hit it faces is listed even while the
// sun's own intensity is 0 (its layer must keep updating for when it comes
// back).
fn giDirectReachesSplit(surface: GiSurface, groupMask: u32, sunApart: bool) -> bool {
    let raySide = giShadeRaySide(surface);
    let lightCount = giLightCount();
    let useGrid = giShadeUsesLightGrid();
    let directionalCount = min(giDirectionalLightCount(), lightCount);
    let cell = select(0u, giLightGridCellAt(surface.position), useGrid);
    let candidates = select(lightCount, directionalCount + GI_LIGHT_GRID_PICKS, useGrid);
    for (var candidate = 0u; candidate < candidates; candidate++) {
        var index = candidate;
        if (useGrid && candidate >= directionalCount) {
            let pick = giLightGridPick(cell, candidate - directionalCount);
            // A light's repeated picks (share 0) were weighed with its first.
            if (pick.light == GI_LIGHT_GRID_NO_LIGHT || !(pick.share > 0.0)) {
                continue;
            }
            index = pick.light;
        }
        var light = giLight(index);
        if (giShadeIsSunApart(sunApart, candidate, directionalCount)) {
            light = giShadeAtWhite(light);
        }
        if (giShadeInGroup(light, groupMask) && giShadeLuminance(giShadeLightSample(light, surface, raySide).irradiance) > 0.0) {
            return true;
        }
    }
    return false;
}

// ---- One light per hit, its shadow ray traced by the caller ------------------------------
//
// For a technique that draws one light for a hit in one pass and traces its
// shadow ray in a later one, as Capsaicin's GI-1 does (a light reservoir per
// hit in GenerateReservoirs, its shadow ray in PopulateCells; the radiance
// cache, RC3.2). The candidates are those giDirectIrradiance weighs (the
// directional lights; then every point and spot light, or with the light grid
// the R picks of the hit's cell), each weighed the same way: the luminance of
// its unshadowed irradiance at the surface times its share. One candidate is
// drawn with probability q = its weight / the total, and its 'weight' is
// share / q, so irradiance x weight is a one-sample estimate of the sum
// giDirectIrradiance's lights converge to (unbiased; noisier, since one light
// stands for all of them, directional lights included).

// What giDirectChooseLight returns when no light of the groups reaches the surface.
const GI_DIRECT_NO_LIGHT: u32 = 0xffffffffu;
// What giDirectChooseLightWithSky returns when it draws the sky (see giSkyLightSample).
const GI_DIRECT_SKY_LIGHT: u32 = ${GI_DIRECT_SKY_LIGHT}u;

struct GiDirectLightChoice {
    light: u32,     // index into the light table, GI_DIRECT_SKY_LIGHT, or GI_DIRECT_NO_LIGHT
    weight: f32,    // multiplies the light's unshadowed irradiance: its share over the probability it was drawn with
}

// Candidate 'candidate' of giDirectChooseLight: the directional lights first
// (each counting once), then giDirectIrradiance's local candidates.
fn giShadeChoiceCandidate(candidate: u32, directionalCount: u32, useGrid: bool, cell: u32) -> GiShadeCandidate {
    if (candidate < directionalCount) {
        return GiShadeCandidate(candidate, 1.0);
    }
    return giShadeLocalCandidate(candidate - directionalCount, directionalCount, useGrid, cell);
}

// One light for the surface from the lights whose group bit is set in
// groupMask, drawn with 'random' (uniform in [0, 1)) in proportion to its
// weight (see above). No shadow ray: trace giDirectLightRay's.
fn giDirectChooseLight(surface: GiSurface, groupMask: u32, random: f32) -> GiDirectLightChoice {
    return giShadeChooseLight(surface, groupMask, random, 0.0);
}

// giDirectChooseLight with the sky as one more candidate, ahead of the
// lights, as Capsaicin keeps its environment light first in its light list
// (light_builder.cpp:200-207). The sky's candidate is one direction,
// giSkyLightSample's for 'skyRandom', weighed as a light is: the luminance of
// its unshadowed irradiance, here that direction's one-sample estimate. When
// the sky is drawn the choice is GI_DIRECT_SKY_LIGHT; trace
// giDirectLightRay's for it with the same skyRandom. A black sky weighs 0
// and is never drawn, and the choice is then giDirectChooseLight's for the
// same 'random'. gi/direct doesn't use it: there the sky reaches a technique
// through the rays that miss.
fn giDirectChooseLightWithSky(surface: GiSurface, groupMask: u32, random: f32, skyRandom: vec2f) -> GiDirectLightChoice {
    let skyWeight = giShadeLuminance(giSkyLightSample(surface, skyRandom).irradiance);
    return giShadeChooseLight(surface, groupMask, random, skyWeight);
}

// The draw behind both: the lights' candidates, after a leading candidate
// of weight 'skyWeight' that stands for the sky (0: no sky).
fn giShadeChooseLight(surface: GiSurface, groupMask: u32, random: f32, skyWeight: f32) -> GiDirectLightChoice {
    let raySide = giShadeRaySide(surface);
    let lightCount = giLightCount();
    let directionalCount = min(giDirectionalLightCount(), lightCount);
    let useGrid = giShadeUsesLightGrid();
    let cell = select(0u, giLightGridCellAt(surface.position), useGrid);
    let candidates = directionalCount + select(lightCount - directionalCount, GI_LIGHT_GRID_PICKS, useGrid);

    // First pass: the total weight, the sky's first. A NaN from the sky counts as 0.
    let sky = select(0.0, skyWeight, skyWeight > 0.0);
    var weightSum = sky;
    for (var candidateNumber = 0u; candidateNumber < candidates; candidateNumber++) {
        let candidate = giShadeChoiceCandidate(candidateNumber, directionalCount, useGrid, cell);
        if (!(candidate.share > 0.0)) {
            continue;
        }
        let light = giLight(candidate.light);
        if (!giShadeInGroup(light, groupMask)) {
            continue;
        }
        weightSum += giShadeLuminance(giShadeLightSample(light, surface, raySide).irradiance) * candidate.share;
    }
    if (!(weightSum > 0.0)) {
        return GiDirectLightChoice(GI_DIRECT_NO_LIGHT, 0.0);
    }

    // Second pass: the candidate whose stretch of the running total holds random x total.
    // Each weighted candidate is kept as it passes, so if rounding leaves the running
    // total a hair short of the first pass's, the last one is drawn.
    let drawn = random * weightSum;
    var cumulative = sky;
    var chosen = GiDirectLightChoice(GI_DIRECT_NO_LIGHT, 0.0);
    if (sky > 0.0) {
        // The sky counts once (share 1): its weight is 1 / q with q = sky / weightSum.
        chosen = GiDirectLightChoice(GI_DIRECT_SKY_LIGHT, weightSum / sky);
        if (drawn < cumulative) {
            return chosen;
        }
    }
    for (var candidateNumber = 0u; candidateNumber < candidates; candidateNumber++) {
        let candidate = giShadeChoiceCandidate(candidateNumber, directionalCount, useGrid, cell);
        if (!(candidate.share > 0.0)) {
            continue;
        }
        let light = giLight(candidate.light);
        if (!giShadeInGroup(light, groupMask)) {
            continue;
        }
        let luminance = giShadeLuminance(giShadeLightSample(light, surface, raySide).irradiance);
        let weight = luminance * candidate.share;
        if (!(weight > 0.0)) {
            continue;
        }
        cumulative += weight;
        // share / q with q = weight / weightSum.
        chosen = GiDirectLightChoice(candidate.light, weightSum / luminance);
        if (drawn < cumulative) {
            break;
        }
    }
    return chosen;
}

// A light's unshadowed irradiance at the surface and the shadow ray that
// decides whether it arrives: from just off the surface (the side the ray
// came from) towards a random point of the light, as giDirectIrradiance
// aims its own. A light that casts no shadows needs no ray.
struct GiDirectLightRay {
    irradiance: vec3f,   // unshadowed, cosine included; 0 when the light can't reach this side
    castsShadows: bool,  // false: nothing to trace, the light always arrives
    shadowRay: GiRay,
}

// For GI_DIRECT_SKY_LIGHT, 'random' is the sky's direction (giSkyLightSample)
// and 'irradiance' that direction's one-sample estimate; the sky always
// casts shadows.
fn giDirectLightRay(surface: GiSurface, light: u32, random: vec2f) -> GiDirectLightRay {
    if (light == GI_DIRECT_SKY_LIGHT) {
        let sky = giSkyLightSample(surface, random);
        return GiDirectLightRay(sky.irradiance, true, sky.shadowRay);
    }
    let raySide = giShadeRaySide(surface);
    let origin = giShadeOffsetRayOrigin(surface.position, raySide);
    let lightData = giLight(light);
    let lightSample = giShadeLightSample(lightData, surface, raySide);
    return GiDirectLightRay(lightSample.irradiance, lightData.castsShadows, giShadeShadowRay(lightData, light, lightSample, origin, random));
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
    return textureSampleLevel(giSkyRadianceMap, giSkySampler, uv, 0.0).rgb * giShade.skyRadianceMapScale.x;
}

// The sky as a light at a surface (the radiance cache's light sampling at its
// hits, RC6.1): one direction drawn from 'random' (uniform in [0, 1)^2),
// uniform over the hemisphere about the shading normal, pdf 1 / (2π). That
// is Capsaicin's environment light at its default sampling mode
// (EnvironmentSamplingMode::Uniform, light_builder.h:54-65;
// sampleEnvironmentLight, light_sampling.hlsl:164-180, through
// mapToHemisphere, sampling.hlsl:230-248, with the platform's own tangent
// basis: the same distribution).
struct GiSkyLightSample {
    direction: vec3f,   // unit, from the surface
    // The sky's radiance along 'direction' x its cosine on the shading normal
    // / the pdf: one direction's estimate of the unshadowed irradiance from
    // the whole sky. 0 when the direction is below the side the ray arrived
    // on, so a smooth shading normal can't pull sky light through a wall (as
    // giShadeLightSample's side check).
    irradiance: vec3f,
    // From just off the surface (giShadeOffsetRayOrigin) along 'direction',
    // unbounded, as a directional light's: anything it hits blocks the sky.
    shadowRay: GiRay,
}

fn giSkyLightSample(surface: GiSurface, random: vec2f) -> GiSkyLightSample {
    let tangents = giShadeTangents(surface.normal);
    // Uniform in cos(theta) is uniform in solid angle over the hemisphere.
    let cosTheta = 1.0 - random.y;
    let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    let angle = GI_SHADE_TWO_PI * random.x;
    let direction = normalize((tangents[0] * cos(angle) + tangents[1] * sin(angle)) * sinTheta + surface.normal * cosTheta);
    let raySide = giShadeRaySide(surface);
    let reachesSide = dot(raySide, direction) > 0.0;
    let cosine = max(dot(surface.normal, direction), 0.0);
    let irradiance = select(vec3f(0.0), giSkyRadiance(direction) * (cosine * GI_SHADE_TWO_PI), reachesSide);
    let origin = giShadeOffsetRayOrigin(surface.position, raySide);
    return GiSkyLightSample(direction, irradiance, GiRay(origin, 0.0, direction, GI_SHADE_UNBOUNDED_DISTANCE));
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
