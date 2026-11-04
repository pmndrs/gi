/**
 * Scene queries: asking questions about the scene's geometry on the GPU.
 *
 * DDGI asks "what does this ray hit first?" and "is anything in the way?".
 * Other techniques ask other things: a voxel technique wants a voxelized view
 * of the scene, an SDF technique wants distances. Each kind of question is a
 * query backend over the same scene copy and handles, so adding a technique
 * never means teaching adapters a new scene format.
 *
 * v1 ships one backend, `rays`, implemented by `@gi/trace` (a two-level BVH).
 * A host may replace it, for example with another BVH library or, one day,
 * hardware ray tracing, as long as the WGSL contract below holds.
 */

import type { WgslModule } from './wgsl';

/**
 * The `rays` backend.
 *
 * Its module defines, at minimum:
 *
 * ```wgsl
 * struct GiRay {
 *     origin: vec3f,
 *     tMin: f32,
 *     direction: vec3f,   // normalized
 *     tMax: f32,
 * }
 *
 * struct GiHit {
 *     t: f32,              // distance along the ray; negative means no hit
 *     instance: u32,       // platform instance index (the scene copy's instance slot)
 *     triangle: u32,       // triangle index within the instance's mesh, as the host gave it
 *     barycentrics: vec2f, // weights of vertices 1 and 2
 *     frontFace: bool,     // true on the side the mesh was authored as front (counter-clockwise
 *                          // corners), also through a mirrored transform (DECISIONS 2026-11-04)
 * }
 *
 * // Nearest hit, honouring alpha cutouts and `blocksLight`.
 * fn giTraceClosest(ray: GiRay) -> GiHit
 *
 * // True if anything blocks the ray. May stop at the first blocker found,
 * // so it is cheaper than giTraceClosest. Used for shadow rays.
 * fn giTraceAny(ray: GiRay) -> bool
 * ```
 *
 * Calling these from a technique's own kernel ("inline" tracing) lets the
 * kernel trace, shade and accumulate without writing rays and hits to memory.
 * Inline is the default, and DDGI uses it. Inline tracing binds 3 storage
 * buffers (measured in T0.10), counted in `module.storageBufferCount`.
 * For techniques that prefer separate passes, {@link RayQueries.encodeBatch}
 * runs a batch from buffers. Its kernel binds 5 (the same 3, plus rays and hits).
 *
 * `GiHit` is for function results only: WGSL `bool` can't be stored in a
 * buffer. Batches write `GiHitRecord` instead (24 bytes, T0.10). `frontFace`
 * comes before `barycentrics` so the `vec2f` lands on an 8-byte boundary
 * without padding:
 *
 * ```wgsl
 * struct GiHitRecord {
 *     t: f32,              // word 0
 *     instance: u32,       // word 1
 *     triangle: u32,       // word 2
 *     frontFace: u32,      // word 3: 1 = front, 0 = back
 *     barycentrics: vec2f, // words 4 and 5
 * }
 * ```
 */
export interface RayQueries {
    readonly kind: 'rays';
    readonly module: WgslModule;
    readonly capabilities: {
        /** Cutout holes are skipped during traversal (M4). */
        readonly alphaCutouts: boolean;
        /** `GiHit.frontFace` is meaningful. Required by DDGI probe classification. */
        readonly backFaceHits: true;
    };
    /**
     * Trace `count` rays from a storage buffer of `GiRay` into a storage buffer
     * of `GiHitRecord` (or of `u32` 0/1 for `any`). Recorded into the given encoder.
     */
    encodeBatch(
        encoder: GPUCommandEncoder,
        batch: {
            readonly mode: 'closest' | 'any';
            readonly rays: GPUBuffer;
            readonly hits: GPUBuffer;
            readonly count: number;
        },
    ): void;
    /** GPU bytes held by acceleration structures. */
    readonly memoryBytes: number;
    /** True while a mesh BVH is still building in a worker. Rays meanwhile see its previous shape, or miss it if it is new. */
    readonly building: boolean;
}

/**
 * Placeholder for later backends, listed so the shape is visible.
 * Each is a separate module over the same scene copy.
 *
 * - `voxels`: a dense or sparse occupancy/albedo grid of the scene, revoxelized
 *   inside dirty regions only.
 * - `distance`: signed distance fields per mesh plus a scene-level field.
 */
export type QueryKind = 'rays' | 'voxels' | 'distance';

/** The query backends a platform instance offers. Techniques ask by kind. */
export interface SceneQueries {
    readonly rays: RayQueries;
    /** Whether a backend is available, so a technique can fail clearly in `prepare`. */
    has(kind: QueryKind): boolean;
}
