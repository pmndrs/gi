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

import { LIGHT_GROUP_COUNT } from './scene';
import type { ShadingModule } from './shading-module';
import type { GpuTimerScope } from './timing';
import type { WgslModule } from './wgsl';

/**
 * The bit of `giDirectChooseLight`'s `groupMask` that makes the sky one of its
 * candidates (`GI_DIRECT_SKY_LIGHT_BIT`): the first bit above the light groups'
 * (`1 << LIGHT_GROUP_COUNT`), so a mask of group bits alone never has it.
 */
export const DIRECT_SKY_LIGHT_BIT = 1 << LIGHT_GROUP_COUNT;

/** The light index `giDirectChooseLight` returns for the sky (`GI_DIRECT_SKY_LIGHT`), which `giDirectLightRay` takes. */
export const DIRECT_SKY_LIGHT = 0xfffffffe;

/**
 * The hit-shading module defines (and depends on the scene and `rays` modules):
 *
 * ```wgsl
 * struct GiSurface {
 *     position: vec3f,       // world space, on the triangle (moved 64 rounding steps in from its edges, so a shadow ray can't start outside a closed shell)
 *     normal: vec3f,         // shading normal (interpolated vertex normal if the mesh has one), facing the ray's origin side
 *     geometricNormal: vec3f, // the triangle's own normal; used for front/back decisions
 *     uv: vec2f,             // uv set 0 (texture lookups use each texture's own mapping)
 *     albedo: vec3f,         // colour x albedo texture x (1 - metalness x metalness texture), capped per channel at bounceAlbedoCap
 *     emission: vec3f,       // emitted radiance: emissive x intensity x emissive texture
 *     material: u32,
 *     instance: u32,
 *     frontFace: bool,
 * }
 *
 * // Reconstruct the surface at a hit from the scene copy. Textures are read
 * // at the mip a ray cone's footprint covers (ray cones, Ray Tracing Gems
 * // ch. 20): log2(spread x t / max(|cos|, 0.1) x texels per metre). A
 * // technique passes its rays' spread, sqrt(4 pi / rays per sphere) (DDGI:
 * // 0.22 radians at 256 rays); giSurfaceAt uses that 0.22. Spread 0 reads mip 0.
 * fn giSurfaceAt(ray: GiRay, hit: GiHit) -> GiSurface
 * fn giSurfaceAtCone(ray: GiRay, hit: GiHit, spread: f32) -> GiSurface
 *
 * // Direct light arriving at the surface from the lights whose group bit is
 * // set in `groupMask` (bits 0-3), as irradiance (shadow rays included).
 * // Lights only: the sky reaches a technique through its rays that miss.
 * // `random`, uniform in [0, 1)^2, decorrelates stochastic light selection
 * // (random.x) and soft-shadow samples between calls.
 * fn giDirectIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> vec3f
 *
 * // The same light choice and shadow rays, the result split by the lights'
 * // group (struct GiDirectGroups { group0, group1, group2, group3, sun: vec3f }).
 * // giDirectIrradiance is the sum of the four groups; `sun` is 0 here.
 * fn giDirectGroupIrradiance(surface: GiSurface, groupMask: u32, random: vec2f) -> GiDirectGroups
 *
 * // The same, and with `sunApart` the sun kept apart (T2.6b): the first
 * // directional light (light 0) is evaluated at a white reference (colour 1,
 * // intensity 1) and goes into `sun` instead of its group, for a technique
 * // that keeps the sun in a layer of its own and weighs that layer by the
 * // sun's colour x intensity.
 * fn giDirectSplitIrradiance(surface: GiSurface, groupMask: u32, sunApart: bool, random: vec2f) -> GiDirectGroups
 *
 * // Whether any light in the groups lights the surface's side at all
 * // (giDirectIrradiance would evaluate it); no shadow rays.
 * fn giDirectReaches(surface: GiSurface, groupMask: u32) -> bool
 * // The same for giDirectSplitIrradiance: with `sunApart` the sun is weighed
 * // in white, so a sun at intensity 0 still lists the hits it faces.
 * fn giDirectReachesSplit(surface: GiSurface, groupMask: u32, sunApart: bool) -> bool
 *
 * // The random numbers gi/direct uses for a ray of a ray scratch (see `direct`).
 * fn giDirectRandom(ray: u32, frame: u32) -> vec2f
 *
 * // One light per hit, its shadow ray traced by the caller (for a technique
 * // that chooses a light in one pass and traces its shadow ray later). The
 * // candidates and their weights are those
 * // giDirectIrradiance uses (directional lights, then every point and spot
 * // light or the light grid's picks for the hit's cell; weight = luminance of
 * // the unshadowed irradiance x share). One is drawn with probability
 * // q = weight / total, `random` uniform in [0, 1); `weight` is share / q, so
 * // `irradiance x weight x visibility` estimates giDirectIrradiance's sum
 * // without bias. `light` is GI_DIRECT_NO_LIGHT when nothing reaches the surface.
 * // With GI_DIRECT_SKY_LIGHT_BIT (DIRECT_SKY_LIGHT_BIT below, 1 << LIGHT_GROUP_COUNT) set in
 * // `groupMask`, the sky is a candidate too, weighed by its irradiance on the
 * // normal (giSkyLightWeight), and drawn as `light` GI_DIRECT_SKY_LIGHT; the
 * // estimate then includes the sky's irradiance. Opt-in: a technique whose
 * // bounce light already holds the sky (DDGI's probes) leaves the bit clear.
 * struct GiDirectLightChoice { light: u32, weight: f32 }
 * fn giDirectChooseLight(surface: GiSurface, groupMask: u32, random: f32) -> GiDirectLightChoice
 * // The light's unshadowed irradiance at the surface (cosine included) and the
 * // shadow ray towards a random point of it (`random` uniform in [0, 1)^2);
 * // trace it with giTraceAny unless `castsShadows` is false. For
 * // GI_DIRECT_SKY_LIGHT, giSkyLightRay's sample.
 * struct GiDirectLightRay { irradiance: vec3f, castsShadows: bool, shadowRay: GiRay }
 * fn giDirectLightRay(surface: GiSurface, light: u32, random: vec2f) -> GiDirectLightRay
 * // The sky as a light, for any technique: its weight in the choice (the
 * // luminance of its SH irradiance on the shading normal, floored at 1% of
 * // its average over normals), and one sample of its irradiance, the
 * // direction drawn cosine-weighted about the shading normal from `random`
 * // (used as is), the estimate pi x giSkyRadiance(direction), with an
 * // unbounded shadow ray along it.
 * fn giSkyLightWeight(surface: GiSurface) -> f32
 * fn giSkyLightRay(surface: GiSurface, random: vec2f) -> GiDirectLightRay
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
    /**
     * The point and spot falloff `module` uses, on its own and with no
     * bindings, for kernels that need a light's unshadowed irradiance at a
     * point without tracing (DDGI's change estimate):
     *
     * ```wgsl
     * fn giShadeDistanceAttenuation(lightDistance: f32, range: f32) -> f32  // three r187: 1 / max(d², 0.01), windowed at range (0: no window)
     * fn giShadeSpotAttenuation(cosine: f32, cosOuterCone: f32, cosInnerCone: f32) -> f32
     * ```
     *
     * `module` depends on it, so a kernel may list both.
     */
    readonly lightFalloff: WgslModule;
    /**
     * How the sky changed, for kernels that size a sky change on the GPU
     * (DDGI's change estimate; the instant sky's nudge): the sky's radiance
     * SH, bands 0-2, now and as it was when the techniques were last handed
     * an environment change in `SceneChanges`. Two uniform bindings, no
     * storage buffers:
     *
     * ```wgsl
     * fn giSkyRadianceShNow(index: u32) -> vec3f     // coefficient index 0..8, rgb
     * fn giSkyRadianceShBefore(index: u32) -> vec3f
     * ```
     *
     * The coefficients are the real SH of `platform/sky/spherical-harmonics.ts`
     * (z polar), unconvolved: the sky's radiance is about
     * `sum of c_k Y_k(direction)`. A `setEnvironment` that changed nothing,
     * or turned a procedural sun too little to re-bake (0.05 degrees), gives
     * now = before.
     */
    readonly skyChange: WgslModule;
    readonly settings: HitShadingSettings;
    /**
     * Deferred direct light (`gi/direct`, docs/design/m2-lights.md §2): a
     * tracing kernel writes a surface record for each hit that needs direct
     * light instead of tracing shadow rays itself, and `gi/direct` traces
     * them over that compacted list only.
     */
    readonly direct: DeferredDirectLight;
}

/**
 * How a tracing kernel fills `gi/direct`'s work list: `subgroups` (one
 * atomic per subgroup, each thread's place from `subgroupExclusiveAdd`), or
 * the declared fallback `atomics` (one atomic per hit). Both give the same
 * light: `gi/direct`'s random numbers follow the ray, not the list's order.
 */
export type DirectWorkList = 'subgroups' | 'atomics';

/**
 * A technique's ray scratch: one storage buffer holding a header (the work
 * list's count and this frame's light groups), the rays' results and the
 * work list of surface records. One buffer, so a tracing kernel binds all
 * of it as one storage buffer (the default limit is 8 per kernel).
 *
 * A ray's result is two `u32`s: radiance rg as two halves, then b and one
 * half that belongs to the technique (DDGI: the signed hit distance).
 * `gi/direct` adds `albedo / π × direct` into the radiance and leaves the
 * technique's half alone. The technique owns the scratch: it creates it with
 * {@link DeferredDirectLight.createScratch} and destroys it.
 *
 * A scratch may hold several results per ray, its `layers` (T2.5, for a
 * technique that keeps light groups apart): ray `i`'s layer `l` is result
 * `i × layers + l`, and each light group's direct light goes into the layer
 * {@link DirectLightFrame.groupLayers} names.
 */
export interface RayScratch {
    readonly buffer: GPUBuffer;
    /** Rays it holds results for. */
    readonly rayCapacity: number;
    /** Work items (hits needing direct light) one tracing dispatch may append. */
    readonly itemCapacity: number;
    /** Results per ray (1 unless the technique asked for more). */
    readonly layers: number;
    /**
     * The ray results alone, as a binding: kernels that only read finished
     * radiance (a technique's blend) bind this range as `array<vec2u>`,
     * ray `i`'s layer `l` at index `i × layers + l`.
     */
    readonly rayResults: GPUBufferBinding;
    /**
     * Where the header's per-frame counts are, for a technique that copies
     * them into its counters (the buffer has `COPY_SRC`): two `u32`s at this
     * byte offset, the hits `gi/direct` lit this frame and the hits dropped
     * because the list was full (0 when the scratch is sized as asked).
     */
    readonly countsOffset: number;
    /** GPU bytes. */
    readonly bytes: number;
    destroy(): void;
}

/** What {@link DeferredDirectLight.encode} needs each time it runs. */
export interface DirectLightFrame {
    /** The frame number: `gi/direct`'s random numbers (`giDirectRandom`). */
    readonly frame: number;
    /** The light groups whose lights count, a bit per group (bits 0-3). */
    readonly groupMask: number;
    /**
     * For light groups 0-3, the result layer that gets the group's direct
     * light, each below the scratch's `layers`. Omitted: every group goes into
     * layer 0.
     */
    readonly groupLayers?: readonly number[];
    /**
     * The result layer that gets the sun's direct light at a white reference
     * (colour 1, intensity 1), the sun kept apart from its group
     * (`giDirectSplitIrradiance`; T2.6b, a technique's sun layer). The sun is
     * the scene's first directional light (`SceneCopy.firstDirectionalLight`).
     * Omitted: the sun is a light like any other, in its group's layer.
     */
    readonly sunLayer?: number;
    /** The calling technique's id: the pass is timed as `<owner>/gi-direct`, so it counts towards the technique's budget. */
    readonly owner: string;
    readonly timer: GpuTimerScope;
}

/**
 * The platform's deferred direct light. A tracing kernel:
 *
 * 1. binds its {@link RayScratch} as one `read_write` storage buffer named
 *    by the technique, and includes {@link traceWgsl} for that name;
 * 2. stores each ray's result with `giStoreRayResult` (layer 0), or each
 *    layer's with `giStoreRayLayerResult`;
 * 3. for a hit that should get direct light, asks `giNeedsDirect(surface)`
 *    (any light of this frame's groups reaching it), makes its record with
 *    `giSurfaceRecord(surface)`, and every thread of the workgroup calls
 *    `giDeferDirect(deferred, record, ray)` in uniform control flow (the
 *    subgroup version needs it):
 *
 * ```wgsl
 * struct GiSurfaceRecord { ... }   // opaque: build it with giSurfaceRecord
 * fn giSurfaceRecord(surface: GiSurface) -> GiSurfaceRecord
 * fn giStoreRayResult(ray: u32, radiance: vec3f, techniqueHalf: f32)
 * fn giStoreRayLayerResult(ray: u32, layer: u32, radiance: vec3f, techniqueHalf: f32)
 * fn giNeedsDirect(surface: GiSurface) -> bool
 * fn giDeferDirect(deferred: bool, record: GiSurfaceRecord, ray: u32)
 * ```
 *
 * Then the technique calls {@link encode} once per tracing dispatch, after
 * it: `gi/direct` adds the direct light into the listed rays' results (each
 * light group's into its layer) and empties the list for the next dispatch.
 * Kernels after it read finished radiance.
 */
export interface DeferredDirectLight {
    /** The work list a kernel should use on this device: `subgroups` when it has the feature. */
    readonly defaultWorkList: DirectWorkList;
    /** Compiles `gi/direct` (once; later calls return the same promise). Before {@link createScratch}. */
    prepare(): Promise<void>;
    /** The tracing kernel's side, for a scratch bound as `scratchVariable` (includes `enable subgroups;` for `subgroups`). */
    traceWgsl(scratchVariable: string, workList: DirectWorkList): string;
    /** A ray scratch for `rays` rays of `layers` results each (default 1, at most 8) and `workItems` work items. */
    createScratch(options: { readonly label: string; readonly rays: number; readonly workItems: number; readonly layers?: number }): RayScratch;
    /** Records `gi/direct` over the scratch's list into the encoder (one compute pass). */
    encode(encoder: GPUCommandEncoder, scratch: RayScratch, frame: DirectLightFrame): void;
}

/** Live settings, read every frame. */
export interface HitShadingSettings {
    /**
     * Shadowed lights evaluated per hit. Directional lights are always
     * evaluated; local lights beyond this count are chosen stochastically,
     * weighted by their estimated contribution, and the result is rescaled so
     * it stays unbiased. With the light grid (more point and spot lights
     * than `exactLightLimit`), `lightGridShadowedLightsPerHit` counts
     * instead. At least 1 (smaller values count as 1). Defaults to 4.
     */
    maxShadowedLightsPerHit: number;
    /**
     * Cap on albedo used for bounces. NVIDIA's native RTXGI uses 0.9; matching
     * it matters when comparing against the native harness (about 23%
     * difference in indirect light). Defaults to 1.0.
     */
    bounceAlbedoCap: number;
    /**
     * With up to this many point and spot lights in the scene, every hit
     * weighs all of them (M1's path; noise-free when no more than
     * `maxShadowedLightsPerHit` reach it). With more, hits draw their lights
     * from the light grid (docs/design/m2-lights.md §3), so a hit costs the
     * same whatever the light count, with some noise that averages out over
     * frames. `Infinity` keeps every scene on the exact path. Defaults to 8.
     */
    exactLightLimit: number;
    /**
     * The most cells the light grid may divide the scene's bounds into
     * (cubic cells, at most 32,768). More cells fit each cell's light picks
     * closer to its hits (less noise) and cost more to build each frame.
     * Defaults to 2,048 (on Sponza, cells of about 2 m: two probe spacings).
     */
    lightGridMaxCells: number;
    /**
     * Shadowed lights evaluated per hit with the light grid: of the hit's
     * cell's 16 light picks, weighed at the hit, this many get a shadow ray,
     * chosen by their weight. Each shadow ray costs far more than the
     * weighing (T2.3: on `sponza-candles` 1 instead of 4 halved the update's
     * cost per probe, for a little more noise). At least 1. Defaults to 1.
     */
    lightGridShadowedLightsPerHit: number;
    /**
     * Whether ray hits read their materials' textures where they hit (albedo,
     * emissive and metalness maps, M4). Off, each texture is read at its last
     * mip, its mean, the same over the whole surface: the flat colours of
     * before M4 (the adapter's folded averages), for comparisons. Changing it
     * doesn't log a scene change; call
     * `resetHistory` to see the difference at once. Defaults to true.
     */
    sampleTextures: boolean;
}

/**
 * A bounce source: whose shading module answers "how much bounce light
 * reaches this hit point?". Usually the technique itself (its previous
 * frame's field), or another technique when stacking.
 */
export interface BounceSource {
    readonly shading: ShadingModule;
}
