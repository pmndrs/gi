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
 *     instance: u32,       // platform instance index
 *     triangle: u32,       // triangle index within the instance's mesh
 *     barycentrics: vec2f, // weights of vertices 1 and 2
 *     frontFace: bool,     // false when the ray hit the back side
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
 * For techniques that prefer separate passes, {@link RayQueries.encodeBatch}
 * runs a batch from buffers.
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
     * of `GiHit` (or of `u32` 0/1 for `any`). Recorded into the given encoder.
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
