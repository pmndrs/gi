/**
 * The world-cache slot (docs/design/radiance-cache.md, "The world-cache
 * interface"): where the screen cache hands its traced probe rays to a world
 * cache, after tracing and before blending. Capsaicin's hash grid is the
 * default and, in phase 3, the only implementation (`HashGridCache`); a
 * Brixelizer brick cache, a surface cache or the DDGI field can take its
 * place without the screen cache changing.
 *
 * - **Input:** the frame's probe rays in one GPU buffer ({@link WorldCacheProbeRays}):
 *   per ray its origin, direction, hit record and a bypass flag, with the
 *   ray count on the GPU.
 * - **Output:** for every ray without the bypass flag that hit something, the
 *   world cache **adds** the radiance leaving the hit toward the ray's origin
 *   into that ray's radiance slot. Misses (the sky) and bypassed rays are the
 *   screen cache's.
 * - **WGSL for other readers:** an optional {@link WorldCacheLookup}.
 */

import type { GpuWord } from '../../../platform/contract/compute.ts';
import type { FrameCamera } from '../../../platform/contract/frame.ts';
import type { ChangeEvent } from '../../../platform/contract/change-events.ts';
import type { Aabb } from '../../../platform/contract/math.ts';
import type { TechniqueContext } from '../../../platform/contract/technique.ts';
import type { WorldCacheTemporalFeedback } from './temporal-feedback.ts';

// ---- The probe-ray buffer ------------------------------------------------------------------------

/** Bytes before the first ray: the ray count in word 0, then three reserved words. */
export const WORLD_CACHE_PROBE_RAYS_HEADER_BYTES = 16;
/** Bytes of one `WorldCacheProbeRay`. */
export const WORLD_CACHE_PROBE_RAY_BYTES = 48;
/** `WorldCacheProbeRay.flags` bit: the ray is lit already (Capsaicin's bypass, gi1.comp:2533-2561); the world cache adds nothing to it. */
export const WORLD_CACHE_PROBE_RAY_BYPASS = 1;
/** Bytes of one radiance slot: four halves. */
export const WORLD_CACHE_RADIANCE_SLOT_BYTES = 8;

/**
 * The probe-ray layout in WGSL, for the screen cache that writes it and the
 * world cache that reads it. It is RC2.1's `RcProbeRays` byte for byte
 * (`wgsl/screen-probes.ts` on branch `rc2-1`), so the screen cache's buffer is
 * handed over as it is.
 *
 * ```wgsl
 * struct WorldCacheProbeRay {        // 48 bytes
 *     origin: vec3f,                 // bytes 0-11: where the ray starts (pushed off its surface)
 *     flags: u32,                    // byte 12: WORLD_CACHE_PROBE_RAY_BYPASS
 *     direction: vec2u,              // bytes 16-23: x, y, z as halves (pack2x16float(xy), pack2x16float(z, 0))
 *     hitT: f32,                     // byte 24: the hit's distance; negative for a miss
 *     hitInstance: u32,              // byte 28: the platform's instance slot (GiHitRecord.instance)
 *     hitTriangle: u32,              // byte 32: the triangle in the instance's mesh
 *     hitFrontFace: u32,             // byte 36: 1 front, 0 back
 *     hitBarycentrics: vec2f,        // bytes 40-47: weights of vertices 1 and 2
 * }
 * struct WorldCacheProbeRays {
 *     count: u32,                    // this frame's rays
 *     reserved: array<u32, 3>,       // scalars: a vec3u would align to 16 and push the rays to byte 32
 *     rays: array<WorldCacheProbeRay>,
 * }
 * ```
 *
 * The radiance slots are a separate `array<vec2u>`, ray `i` at index `i`:
 * four halves as Capsaicin's `packHalf4` packs them (`pack2x16float(rg)`,
 * `pack2x16float(b, w)`), rgb the ray's radiance and w the screen cache's
 * own (the hit distance, which the world cache leaves alone). That is
 * Capsaicin's `g_ScreenProbes_ProbeSpawnRadianceBuffer` (gi1.comp:1156).
 */
export const WORLD_CACHE_PROBE_RAYS_WGSL = /* wgsl */ `
// WorldCacheProbeRay.flags: already lit (Capsaicin's bypass flag); the world cache leaves it alone.
const WORLD_CACHE_PROBE_RAY_BYPASS: u32 = ${WORLD_CACHE_PROBE_RAY_BYPASS}u;

struct WorldCacheProbeRay {
    origin: vec3f,
    flags: u32,
    direction: vec2u,           // x, y, z as halves: pack2x16float(xy), pack2x16float(vec2f(z, 0))
    hitT: f32,                  // negative for a miss
    hitInstance: u32,
    hitTriangle: u32,
    hitFrontFace: u32,
    hitBarycentrics: vec2f,
}

struct WorldCacheProbeRays {
    count: u32,
    reserved: array<u32, 3>,    // scalars, so the rays start at byte 16 (a vec3u would align to 16 and push them to 32)
    rays: array<WorldCacheProbeRay>,
}

fn worldCacheProbeRayDirection(ray: WorldCacheProbeRay) -> vec3f {
    return vec3f(unpack2x16float(ray.direction.x), unpack2x16float(ray.direction.y).x);
}
`;

/** One frame's traced probe rays, as the screen cache hands them over. */
export interface WorldCacheProbeRays {
    /** `WorldCacheProbeRays` (above): the header, then `capacity` rays. `STORAGE` usage. */
    readonly buffer: GPUBuffer;
    /** The rays this frame (word 0 of `buffer`), written on the GPU; sizes the world cache's dispatches through indirect arguments. */
    readonly count: GpuWord;
    /** The most rays `buffer` holds. */
    readonly capacity: number;
    /** The radiance slots: `capacity` × `vec2u` (four halves), ray `i` at index `i`. `STORAGE` usage. */
    readonly radiance: GPUBufferBinding;
}

// ---- The world cache --------------------------------------------------------------------------------

/**
 * What a world cache uses from the technique's context. A `TechniqueContext`
 * is one: the hash grid lights its hits through the platform's hit shading
 * (`shade`) and times its passes (`timer`). `fallback` and `warn` are optional
 * (experiment E10): the technique stacked under this one, whose shading module
 * the hash grid can read as its second bounce (`secondBounce: 'ddgi-volume'`).
 */
export type WorldCacheContext = Pick<TechniqueContext, 'device' | 'kernels' | 'compute' | 'shade' | 'timer'> & Partial<Pick<TechniqueContext, 'fallback' | 'warn'>>;

/** Per-frame inputs. */
export interface WorldCacheFrame {
    /**
     * The frame number. Used as a 32-bit counter that wraps (Capsaicin's
     * `g_FrameIndex`), so any non-negative whole number works, past 2^32 too.
     */
    readonly frame: number;
    readonly camera: FrameCamera;
    /** The render size in pixels: cells are sized by how many pixels they cover. */
    readonly width: number;
    readonly height: number;
    /**
     * Optional: the scene's changes. `events` (P6.2): the change events at
     * this frame (the platform's `ChangeEventTracker` over `SceneChanges`),
     * each with its box and the start of its current epoch, for a cache that
     * forgets history once per epoch (the hash grid's E8 and E9).
     * `everything` / `boxes` (P6.1): the events armed this frame, as the screen
     * probes' running means get them (D46): their boxes, or `everything` for
     * a change with no bounds. Absent: no change.
     */
    readonly changes?: WorldCacheChanges;
    /**
     * Optional (RC3.6, register D68): last frame's camera, lit image, depth and
     * normals, for `useTemporalFeedback`. Absent on a frame with no valid last
     * frame (the first, after a resize or a reset) or without a lit image.
     */
    readonly temporalFeedback?: WorldCacheTemporalFeedback;
}

/** The scene's changes for a world cache: the change events, and those armed this frame as boxes (the radiance cache's `HistoryChanges`). */
export interface WorldCacheChanges {
    readonly events: readonly Pick<ChangeEvent, 'box' | 'epoch'>[];
    readonly everything: boolean;
    readonly boxes: readonly Aabb[];
}

/**
 * WGSL for readers that look radiance up without inserting (Capsaicin's
 * reflections read `HashGridCache_FindCell`, gi1.comp:2866-2899):
 *
 * ```wgsl
 * // Radiance leaving `position` along -`direction` (toward a viewer looking along `direction`),
 * // in rgb, and a validity weight in w: 1 where the cache has the cell, 0 where it can't answer.
 * fn worldCacheLookup(position: vec3f, normal: vec3f, direction: vec3f) -> vec4f
 * ```
 *
 * The code declares the cache's bindings in the group the caller picks, and
 * the caller binds `createBindGroup`'s group there.
 */
export interface WorldCacheLookup {
    code(group: number): string;
    /** The group's layout entries, for the caller's layout descriptor (and its storage-buffer count). */
    readonly entries: readonly GPUBindGroupLayoutEntry[];
    /** A bind group for a layout made from `entries`. */
    createBindGroup(layout: GPUBindGroupLayout): GPUBindGroup;
}

/** A world cache the technique owns. */
export interface WorldCache {
    /** Short stable id: `hash-grid` (Capsaicin's) or `sharc` (P6.1). */
    readonly id: string;
    /** Allocates and compiles. May throw `GILimitError` naming the device limit it needs. Call again after capacity settings change. */
    prepare(context: WorldCacheContext): Promise<void>;
    /**
     * Records all of the cache's per-frame work into the encoder: purge,
     * insert, populate, update, resolve into `rays.radiance`. One call per
     * frame and per queue submit (it writes the frame's constants with
     * `queue.writeBuffer`).
     */
    encodeFrame(encoder: GPUCommandEncoder, frame: WorldCacheFrame, rays: WorldCacheProbeRays): void;
    /**
     * Optional (P6.1): makes what the cache needs per probe ray for a screen
     * of up to `maxRayCount` rays, and throws `GILimitError` naming the limit
     * when the device can't hold it. The technique calls it where it makes
     * its own screen resources, so a resize fails there and never inside
     * `encodeFrame`. A cache without it makes its per-ray data in
     * `encodeFrame`.
     */
    prepareScreen?(maxRayCount: number): void;
    /**
     * Records emptying the cache into the encoder: everything it has learned
     * is forgotten, as at `prepare` (a technique's `resetHistory`: a camera
     * cut or a scene swap). Allocates nothing.
     */
    reset(encoder: GPUCommandEncoder): void;
    /** GPU bytes the cache allocated. */
    readonly memoryBytes: number;
    /** Plain-language figures: capacity and memory. */
    readonly stats: Readonly<Record<string, number>>;
    /** WGSL lookups for other readers; absent when the cache can't answer them. */
    readonly lookup?: WorldCacheLookup;
    dispose(): void;
}
