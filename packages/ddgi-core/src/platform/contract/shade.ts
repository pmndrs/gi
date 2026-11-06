/**
 * Hit shading: lighting the point a ray hit.
 *
 * Every ray-based technique needs the same answer at a hit: what is the
 * material there, how much direct light (with shadows) reaches it, how much
 * does it emit, and what does a ray that misses see. `@gi/shade` writes this
 * once. DDGI, surfels, cascades, RT reflections and the reference checks all
 * reuse it.
 *
 * Bounce light (light that has already bounced at least once) is not part of
 * hit shading. It comes from a *bounce source*: a technique's shading module,
 * usually the active technique's own previous answer. That is how DDGI gets
 * unlimited bounces, and how a surfel technique can fall back to DDGI.
 */

import type { ShadingModule } from './shading-module';
import type { WgslModule } from './wgsl';

/**
 * The hit-shading module defines (and depends on the scene and `rays` modules):
 *
 * ```wgsl
 * struct GiSurface {
 *     position: vec3f,       // world space, on the triangle (moved 64 rounding steps in from its edges, so a shadow ray can't start outside a closed shell)
 *     normal: vec3f,         // shading normal (interpolated vertex normal if the mesh has one), facing the ray's origin side
 *     geometricNormal: vec3f, // the triangle's own normal; used for front/back decisions
 *     uv: vec2f,
 *     albedo: vec3f,         // diffuse albedo, cutout already applied, capped per channel at bounceAlbedoCap
 *     emission: vec3f,       // emitted radiance
 *     material: u32,
 *     instance: u32,
 *     frontFace: bool,
 * }
 *
 * // Reconstruct the surface at a hit from the scene copy.
 * fn giSurfaceAt(ray: GiRay, hit: GiHit) -> GiSurface
 *
 * // Direct light arriving at the surface from the lights whose group bit is
 * // set in `groupMask` (bits 0-3), as irradiance (shadow rays included).
 * // Lights only: the sky reaches a technique through its rays that miss.
 * // `random`, uniform in [0, 1)^2, decorrelates stochastic light selection
 * // (random.x) and soft-shadow samples between calls.
 * fn giDirectIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> vec3f
 *
 * // Radiance seen along a ray that missed everything.
 * fn giSkyRadiance(direction: vec3f) -> vec3f
 *
 * // Irradiance from the sky alone at a surface (for the sky light group),
 * // with nothing in the way (no visibility).
 * fn giSkyIrradiance(surface: GiSurface) -> vec3f
 * // The same for any unit normal.
 * fn giSkyIrradianceAt(normal: vec3f) -> vec3f
 * ```
 *
 * Point and spot lights fall off exactly as in three.js r187 (inverse square,
 * d^2 clamped at 0.01, windowed to zero at `range`; spot cones by smoothstep
 * between the outer and inner cone), so the three adapter matches what three
 * renders. Implemented in `platform/shade/`.
 *
 * A technique adds bounce light itself, from its bounce source:
 * `radiance = surface.emission + surface.albedo / π * (direct + bounce)`.
 * Keeping that sum in the technique lets light groups decide which layer
 * each term goes into.
 */
export interface HitShading {
    readonly module: WgslModule;
    readonly settings: HitShadingSettings;
}

/** Live settings, read every frame. */
export interface HitShadingSettings {
    /**
     * Shadowed lights evaluated per hit. Directional lights are always
     * evaluated; local lights beyond this count are chosen stochastically,
     * weighted by their estimated contribution, and the result is rescaled so
     * it stays unbiased. Keeps hit cost flat as light count grows toward the
     * 32-light minimum capacity. At least 1 (smaller values count as 1).
     * Defaults to 4.
     */
    maxShadowedLightsPerHit: number;
    /**
     * Cap on albedo used for bounces. NVIDIA's native RTXGI uses 0.9; matching
     * it matters when comparing against the native harness (about 23%
     * difference in indirect light). Defaults to 1.0.
     */
    bounceAlbedoCap: number;
}

/**
 * A bounce source: whose shading module answers "how much bounce light
 * reaches this hit point?". Usually the technique itself (its previous
 * frame's field), or another technique when stacking.
 */
export interface BounceSource {
    readonly shading: ShadingModule;
}
