/**
 * WGSL of the hit-shading module (contract/shade.ts): the surface at a ray
 * hit, direct light with shadow rays, and the sky.
 *
 * ```wgsl
 * fn giSurfaceAt(ray: GiRay, hit: GiHit) -> GiSurface
 * fn giSurfaceAtCone(ray: GiRay, hit: GiHit, spread: f32) -> GiSurface   // textures at a ray cone's footprint
 * fn giDirectIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> vec3f
 * fn giDirectGroupIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> GiDirectGroups  // the same, split by light group 0-3
 * fn giDirectSplitIrradiance(surface: GiSurface, groupMask: u32, sunApart: bool, random: vec2f) -> GiDirectGroups  // and the sun apart, in white
 * fn giDirectReaches(surface: GiSurface, groupMask: u32) -> bool   // any light in the groups lights this side (no shadow rays)
 * fn giDirectReachesSplit(surface: GiSurface, groupMask: u32, sunApart: bool) -> bool  // the same, the sun in white when apart
 * fn giDirectRandom(ray: u32, frame: u32) -> vec2f                  // gi/direct's random numbers for a ray
 * fn giDirectChooseLight(surface: GiSurface, groupMask: u32, random: f32) -> GiDirectLightChoice  // one light drawn by weight, and its weight; the sky too with GI_DIRECT_SKY_LIGHT_BIT
 * fn giDirectLightRay(surface: GiSurface, light: u32, random: vec2f) -> GiDirectLightRay           // its unshadowed irradiance and shadow ray
 * fn giSkyLightWeight(surface: GiSurface) -> f32                     // the sky as a light: its weight in the choice
 * fn giSkyLightRay(surface: GiSurface, random: vec2f) -> GiDirectLightRay  // and one cosine-weighted sample of it, with its shadow ray
 * fn giSkyRadiance(direction: vec3f) -> vec3f
 * fn giSkyIrradiance(surface: GiSurface) -> vec3f
 * fn giSkyIrradianceAt(normal: vec3f) -> vec3f        // the same, for any unit normal
 * ```
 *
 * It depends on the scene copy (lights, materials, triangles), the `rays`
 * module (`giTraceAny` for shadow rays) and the texture store's sampling
 * module (`giTextureSampleMapped`, textures/texture-sampling-wgsl.ts, which
 * binds the store's array and its sampler and which the walk's alpha rules
 * share), and binds no storage buffer of its own: one uniform buffer
 * (settings, the light grid's place and the sky's irradiance SH), the sky's
 * radiance map, a sampler, and the light grid's picks (a texture,
 * light-grid.ts). So the scene copy and hit shading together stay at the
 * scene copy's 2 storage buffers (docs/api/platform.md, binding budget).
 *
 * Light units are the platform's (contract/scene.ts), which are three.js's
 * physically based ones; point and spot falloff and spot cones follow three.js
 * r187 exactly, so the three adapter matches what three renders.
 */

import type { BindingSlot, ModuleBindingEntry } from '../contract/wgsl.ts';
import { lightGridLookupWgsl } from './light-grid.ts';
import { RADIANCE_SH_COUNT, SKY_HARMONICS_LAYOUT } from '../sky/projection-wgsl.ts';
import { SH_BASIS_WGSL, SH_COEFFICIENT_COUNT } from '../sky/spherical-harmonics.ts';

/** `GPUShaderStage.COMPUTE`. The global does not exist in Node, where the tests import this file. */
const SHADER_STAGE_COMPUTE = 0x4;

/**
 * The module's bindings, in order: uniforms (settings and sky SH), the sky's
 * radiance map, its sampler, the light grid's picks. No storage buffers. The
 * texture store and its sampler are the sampling module's.
 */
export const SHADE_MODULE_ENTRIES: readonly ModuleBindingEntry[] = [
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'float', viewDimension: '2d' } },
    { visibility: SHADER_STAGE_COMPUTE, sampler: { type: 'filtering' } },
    { visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } },
];

/**
 * The ray cone spread `giSurfaceAt` uses: a probe ray of DDGI's default 256
 * rays a probe stands for 1/256 of the sphere, a cone sqrt(4π / 256) = 0.22
 * radians wide. Techniques with their own rays pass theirs to `giSurfaceAtCone`.
 */
export const DEFAULT_CONE_SPREAD = Math.sqrt((4 * Math.PI) / 256);

/**
 * Below this |cos θ| between the ray and the surface, the footprint stops
 * growing: an edge-on hit's footprint would be endless, and the mip it asks
 * for would average the whole texture. 0.1 caps the stretch at 10×.
 */
const MIN_FOOTPRINT_COSINE = 0.1;

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
    /** The `sampleTextures` setting, 1 or 0. */
    sampleTextures: SKY_HARMONICS_LAYOUT.byteLength + 48,
    /** Rounded up to the struct's 16-byte alignment. */
    byteLength: SKY_HARMONICS_LAYOUT.byteLength + 64,
} as const;

/**
 * The bit of `giDirectChooseLight`'s `groupMask` that makes the sky one of
 * its candidates (`GI_DIRECT_SKY_LIGHT_BIT`), above the four light groups'.
 */
export const DIRECT_SKY_LIGHT_BIT = 0x10;

/** The light index `giDirectChooseLight` returns for the sky (`GI_DIRECT_SKY_LIGHT`), which `giDirectLightRay` takes. */
export const DIRECT_SKY_LIGHT = 0xfffffffe;

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
    albedo: vec3f,          // diffuse albedo: colour x albedo texture x (1 - metalness x metalness texture), capped per channel at bounceAlbedoCap
    emission: vec3f,        // emitted radiance: emissive x intensity x emissive texture
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
    // 1: hits read their materials' textures (the sampleTextures setting); 0: flat colours, as before M4.
    sampleTextures: u32,
}

@group(${group}) @binding(${uniformsBinding}) var<uniform> giShade: GiShadeUniforms;
// The sky seen by rays that miss: equirect, first row the nadir (three.js's equirectUV).
@group(${group}) @binding(${radianceMapBinding}) var giSkyRadianceMap: texture_2d<f32>;
// Linear, repeating across the -X seam, clamped at the poles.
@group(${group}) @binding(${samplerBinding}) var giSkySampler: sampler;
// The light grid's picks, R per cell: (light index, its share at a hit as f32 bits).
@group(${group}) @binding(${lightGridBinding}) var giLightGridPicks: texture_2d<u32>;

const GI_SHADE_PI: f32 = 3.14159265358979;
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
// giSurfaceAt's ray cone spread, radians (DEFAULT_CONE_SPREAD in wgsl.ts).
const GI_SHADE_DEFAULT_CONE_SPREAD: f32 = ${DEFAULT_CONE_SPREAD};
// The smallest |cos| between ray and surface the footprint counts (MIN_FOOTPRINT_COSINE in wgsl.ts).
const GI_SHADE_MIN_FOOTPRINT_COSINE: f32 = ${MIN_FOOTPRINT_COSINE};

// ---- The surface at a hit ---------------------------------------------------------------

// The surface at a hit, its textures read at the footprint of a ray cone of
// the platform's default spread (a probe ray of 256 per sphere).
fn giSurfaceAt(ray: GiRay, hit: GiHit) -> GiSurface {
    return giSurfaceAtCone(ray, hit, GI_SHADE_DEFAULT_CONE_SPREAD);
}

// The surface at a hit of a ray standing for a cone 'spread' radians wide (a
// technique's rays: sqrt(4 pi / rays per sphere)). The cone picks the mip of
// each texture, so a ray reads the average of what it stands for, which is
// right and less noisy than one texel (ray cones; Akenine-Moller et al.,
// "Improved Shader and Texture Level of Detail Using Ray Cones", JCGT 2021,
// and Ray Tracing Gems ch. 20). spread 0 reads the finest mip.
fn giSurfaceAtCone(ray: GiRay, hit: GiHit, spread: f32) -> GiSurface {
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

    // The material's colours times its textures (a texture word is GI_NO_TEXTURE
    // until the texture's layer is made: the colour alone, as if it were white;
    // a pending metalness map's record carries metalness 0 meanwhile, so a
    // metal-mapped material doesn't go black while it loads).
    // With sampleTextures off, each texture is read at its last mip, one texel:
    // its mean, the same at every point (the flat colour M1's adapter folded in).
    var albedo = material.albedo;
    var emission = material.emission;
    var metalness = material.metalness;
    var footprint = GiShadeFootprint(0.0, 0.0);
    if (giShade.sampleTextures != 0u) {
        footprint = giShadeTextureFootprint(ray, hit, corners, geometricNormal, spread);
    }
    if (material.albedoTexture != GI_NO_TEXTURE) {
        albedo *= giShadeTextureAtHit(material.albedoTexture, material.albedoUv, hit, weights, footprint).rgb;
    }
    if (material.emissiveTexture != GI_NO_TEXTURE) {
        emission *= giShadeTextureAtHit(material.emissiveTexture, material.emissiveUv, hit, weights, footprint).rgb;
    }
    if (material.metalnessTexture != GI_NO_TEXTURE) {
        metalness *= giShadeChannel(giShadeTextureAtHit(material.metalnessTexture, material.metalnessUv, hit, weights, footprint), material.metalnessChannel);
    }
    // A metal reflects no diffuse light (three and glTF: diffuse = colour x (1 - metalness)).
    albedo *= 1.0 - metalness;

    return GiSurface(
        position,
        normal,
        geometricNormal,
        giInstanceUvAt(hit.instance, hit.triangle, hit.barycentrics, 0u),
        min(albedo, vec3f(giShade.bounceAlbedoCap)),
        emission,
        materialIndex,
        hit.instance,
        hit.frontFace,
    );
}

// ---- Textures at a hit --------------------------------------------------------------------

// What a ray cone covers where it meets the surface, for the mip a texture is read at.
struct GiShadeFootprint {
    // The cone's width there, along the surface: spread x distance, stretched by
    // 1 / |cos| as a slanted surface spreads the cone over more of itself.
    width: f32,
    // Twice the hit triangle's world-space area (its uv area is compared with it).
    doubleWorldArea: f32,
}

fn giShadeTextureFootprint(ray: GiRay, hit: GiHit, corners: GiTriangle, geometricNormal: vec3f, spread: f32) -> GiShadeFootprint {
    let cosine = max(abs(dot(ray.direction, geometricNormal)), GI_SHADE_MIN_FOOTPRINT_COSINE);
    let doubleWorldArea = length(cross(corners.vertex1 - corners.vertex0, corners.vertex2 - corners.vertex0));
    return GiShadeFootprint(spread * hit.t / cosine, doubleWorldArea);
}

// The mip a texture is read at: log2 of the footprint's width in texels.
// Texels per metre come from the triangle's area in texels (its uv area,
// scaled by the uv transform's determinant, times the layer's texels)
// against its area in metres. A triangle with no uv area (no uvs) reads mip 0.
fn giShadeTextureLod(footprint: GiShadeFootprint, uvs: GiTriangleUvs, mapping: GiUvMapping) -> f32 {
    let uvEdge1 = uvs.uv1 - uvs.uv0;
    let uvEdge2 = uvs.uv2 - uvs.uv0;
    let doubleUvArea = abs(uvEdge1.x * uvEdge2.y - uvEdge1.y * uvEdge2.x);
    let transformArea = abs(determinant(mat2x2f(mapping.transform[0], mapping.transform[1])));
    let texelsSquaredPerSquareMetre = doubleUvArea * transformArea * GI_TEXTURE_SIZE * GI_TEXTURE_SIZE / max(footprint.doubleWorldArea, 1.0e-30);
    let footprintTexels = footprint.width * sqrt(texelsSquaredPerSquareMetre);
    return clamp(log2(max(footprintTexels, 1.0e-30)), 0.0, GI_TEXTURE_LAST_LOD);
}

// A material texture at the hit: its uv set at the hit point (clamped
// barycentric 'weights', as the surface point), its 3x2 transform, its wrap
// (giTextureSampleMapped, the store's sampling module, which the walk's alpha
// test shares), at the footprint's mip. Linear light (the store is sRGB-encoded).
// A material texture at a hit: at the ray cone's mip (giShadeSampleTexture),
// or with sampleTextures off at the last mip, which is the texture's mean
// wherever it is read ('footprint' is unused then).
fn giShadeTextureAtHit(layer: u32, mapping: GiUvMapping, hit: GiHit, weights: vec3f, footprint: GiShadeFootprint) -> vec4f {
    if (giShade.sampleTextures == 0u) {
        return textureSampleLevel(giTextureStore, giTextureSampler, vec2f(0.5), layer, GI_TEXTURE_LAST_LOD);
    }
    return giShadeSampleTexture(layer, mapping, hit, weights, footprint);
}

fn giShadeSampleTexture(layer: u32, mapping: GiUvMapping, hit: GiHit, weights: vec3f, footprint: GiShadeFootprint) -> vec4f {
    let uvs = giInstanceTriangleUvs(hit.instance, hit.triangle, mapping.channel);
    return giTextureSampleMapped(layer, mapping, uvs, weights, giShadeTextureLod(footprint, uvs, mapping));
}

// Channel 0-3 (r, g, b, a) of a sample. Selects rather than indexing the
// vector at run time, which can put it in slow per-thread memory (NOTES, "Tracing").
fn giShadeChannel(value: vec4f, channel: u32) -> f32 {
    return select(select(value.x, value.y, channel == 1u), select(value.z, value.w, channel == 3u), channel >= 2u);
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
// For a technique that chooses a light for a hit in one pass and traces its
// shadow ray later (it keeps the choice per hit between the two passes). The
// candidates are those giDirectIrradiance weighs (the
// directional lights; then every point and spot light, or with the light grid
// the R picks of the hit's cell), each weighed the same way: the luminance of
// its unshadowed irradiance at the surface times its share; and, when the
// caller sets GI_DIRECT_SKY_LIGHT_BIT in groupMask, the sky (below). One
// candidate is drawn with probability q = its weight / the total, and its
// 'weight' is share / q, so irradiance x weight is a one-sample estimate of
// the sum giDirectIrradiance's lights converge to, plus the sky's irradiance
// with that bit (unbiased; noisier, since one light stands for all of them,
// directional lights included).

// What giDirectChooseLight returns when no light of the groups reaches the surface.
const GI_DIRECT_NO_LIGHT: u32 = 0xffffffffu;

// The sky as a light: giDirectChooseLight weighs it as a candidate when this
// bit of groupMask is set (above the four light groups' bits, so a caller
// that passes only group bits, as gi/direct does, never gets it), and returns
// GI_DIRECT_SKY_LIGHT for it, which giDirectLightRay takes like a light index.
//
// Why opt-in: a technique whose bounce source already holds the sky (DDGI's
// probes see it through the rays that miss, and their irradiance lights
// every hit) would count it twice. A technique that caches the light leaving
// a hit and never looks the sky up at that hit (the radiance cache's world
// cache, register D33) needs it here.
const GI_DIRECT_SKY_LIGHT_BIT: u32 = ${DIRECT_SKY_LIGHT_BIT}u;
const GI_DIRECT_SKY_LIGHT: u32 = ${DIRECT_SKY_LIGHT}u;

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
// groupMask, and the sky with GI_DIRECT_SKY_LIGHT_BIT, drawn with 'random'
// (uniform in [0, 1)) in proportion to its weight (see above). No shadow ray:
// trace giDirectLightRay's.
fn giDirectChooseLight(surface: GiSurface, groupMask: u32, random: f32) -> GiDirectLightChoice {
    let raySide = giShadeRaySide(surface);
    let lightCount = giLightCount();
    let directionalCount = min(giDirectionalLightCount(), lightCount);
    let useGrid = giShadeUsesLightGrid();
    let cell = select(0u, giLightGridCellAt(surface.position), useGrid);
    let candidates = directionalCount + select(lightCount - directionalCount, GI_LIGHT_GRID_PICKS, useGrid);

    // First pass: the lights' total weight; then the sky's, which comes after them.
    var lightWeightSum = 0.0;
    for (var candidateNumber = 0u; candidateNumber < candidates; candidateNumber++) {
        let candidate = giShadeChoiceCandidate(candidateNumber, directionalCount, useGrid, cell);
        if (!(candidate.share > 0.0)) {
            continue;
        }
        let light = giLight(candidate.light);
        if (!giShadeInGroup(light, groupMask)) {
            continue;
        }
        lightWeightSum += giShadeLuminance(giShadeLightSample(light, surface, raySide).irradiance) * candidate.share;
    }
    var skyWeight = 0.0;
    if ((groupMask & GI_DIRECT_SKY_LIGHT_BIT) != 0u) {
        skyWeight = giSkyLightWeight(surface);
    }
    let weightSum = lightWeightSum + skyWeight;
    if (!(weightSum > 0.0)) {
        return GiDirectLightChoice(GI_DIRECT_NO_LIGHT, 0.0);
    }

    // The sky's stretch of the running total is the last one: [the lights' total, the total).
    let drawn = random * weightSum;
    if (skyWeight > 0.0 && !(drawn < lightWeightSum)) {
        // share / q, with share 1 and q = skyWeight / weightSum.
        return GiDirectLightChoice(GI_DIRECT_SKY_LIGHT, weightSum / skyWeight);
    }

    // Second pass: the light whose stretch of the running total holds random x total.
    // Each weighted candidate is kept as it passes, so if rounding leaves the running
    // total a hair short of the first pass's, the last one is drawn.
    var cumulative = 0.0;
    var chosen = GiDirectLightChoice(GI_DIRECT_NO_LIGHT, 0.0);
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
// aims its own. A light that casts no shadows needs no ray. For
// GI_DIRECT_SKY_LIGHT, one sample of the sky (giSkyLightRay).
struct GiDirectLightRay {
    irradiance: vec3f,   // unshadowed, cosine included; 0 when the light can't reach this side
    castsShadows: bool,  // false: nothing to trace, the light always arrives
    shadowRay: GiRay,
}

fn giDirectLightRay(surface: GiSurface, light: u32, random: vec2f) -> GiDirectLightRay {
    if (light == GI_DIRECT_SKY_LIGHT) {
        return giSkyLightRay(surface, random);
    }
    let raySide = giShadeRaySide(surface);
    let origin = giShadeOffsetRayOrigin(surface.position, raySide);
    let lightData = giLight(light);
    let lightSample = giShadeLightSample(lightData, surface, raySide);
    return GiDirectLightRay(lightSample.irradiance, lightData.castsShadows, giShadeShadowRay(lightData, light, lightSample, origin, random));
}

// ---- The sky as a light -------------------------------------------------------------------
//
// The sky's irradiance at a surface, E = the integral over the hemisphere of
// L(w) cos dw with the sky's radiance L as giSkyRadiance reads it, estimated
// from one direction drawn with probability density cos / pi about the
// shading normal: L(w) cos / (cos / pi) = pi L(w), times whether a ray along
// it escapes (the shadow ray). Unbiased, and with no noise under a constant
// sky with nothing in the way.
//
// Why cosine-weighted rather than drawn by the sky map's brightness: the
// platform's skies have no sun disc (the sun is a directional light, already
// a candidate of its own; calibrated-sky.ts), so the map is smooth and the
// cosine is most of the integrand's shape; it needs no table (no new binding,
// so no technique's bind group changes) and costs one map read. Capsaicin
// offers it as its environment light's cosine mode (light_sampling.hlsl:164-182;
// its default draws the hemisphere uniformly, pdf 1 / (2 pi): the same mean,
// noisier). A host sky with a sharp sun painted in would want the map's
// brightness drawn too (the radiance cache's register D49, its way forward).

// The sky's weight among giDirectChooseLight's candidates, in the same units
// as a light's (the luminance of unshadowed irradiance): the sky's irradiance
// on the shading normal from its SH (giSkyIrradianceAt). Floored at 1% of
// the irradiance averaged over every normal (the band-0 term), so the sky is
// drawn wherever it can give light even where the truncated SH reads 0 (an
// estimate is unbiased only if every candidate that can light the surface
// can be drawn); 0 only when the whole sky is black.
const GI_SKY_LIGHT_WEIGHT_FLOOR: f32 = 0.01;
// The band-0 basis function, Y00 = 1 / (2 sqrt(pi)).
const GI_SKY_SH_Y00: f32 = 0.28209479177387814;

fn giSkyLightWeight(surface: GiSurface) -> f32 {
    let onNormal = giShadeLuminance(giSkyIrradianceAt(surface.normal));
    let averageOverNormals = giShadeLuminance(max(giShade.skyIrradiance[0].xyz * GI_SKY_SH_Y00, vec3f(0.0)));
    return max(onNormal, GI_SKY_LIGHT_WEIGHT_FLOOR * averageOverNormals);
}

// One sample of the sky's irradiance at the surface and its shadow ray: the
// direction drawn cosine-weighted about the shading normal from 'random'
// (uniform in [0, 1)^2, used as is, so a caller that keeps the two numbers
// can rebuild the direction), the estimate pi L(w); 0 when the direction is
// behind the side the ray arrived on (as for a light behind it,
// giShadeLightSample). The shadow ray leaves from just off that side and is
// unbounded, as a directional light's: anything in the way blocks the sky.
fn giSkyLightRay(surface: GiSurface, random: vec2f) -> GiDirectLightRay {
    let raySide = giShadeRaySide(surface);
    let origin = giShadeOffsetRayOrigin(surface.position, raySide);
    let tangents = giShadeTangents(surface.normal);
    let radius = sqrt(random.x);
    let angle = GI_SHADE_TWO_PI * random.y;
    let direction = normalize(tangents[0] * (radius * cos(angle)) + tangents[1] * (radius * sin(angle)) + surface.normal * sqrt(max(1.0 - random.x, 0.0)));
    let reachesSide = dot(raySide, direction) > 0.0;
    let irradiance = select(vec3f(0.0), GI_SHADE_PI * giSkyRadiance(direction), reachesSide);
    return GiDirectLightRay(irradiance, true, GiRay(origin, 0.0, direction, GI_SHADE_UNBOUNDED_DISTANCE));
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
