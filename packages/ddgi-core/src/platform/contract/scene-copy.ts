/**
 * The platform's GPU copy of the scene, as techniques see it, and the changes
 * since the last frame.
 *
 * Techniques never talk to the host. They read the scene through this copy,
 * the query backends and hit shading, and they learn what moved from
 * {@link SceneChanges}.
 */

import type { InstanceHandle, LightHandle } from './handles';
import type { LightDescriptor } from './scene';
import type { Aabb } from './math';
import type { WgslModule } from './wgsl';

/**
 * The scene-data module defines accessors over the platform's tables, so
 * techniques never depend on buffer layouts:
 *
 * ```wgsl
 * fn giInstanceCount() -> u32
 * fn giInstanceTransform(instance: u32) -> mat4x4f
 * fn giInstanceBounds(instance: u32) -> GiAabb
 * fn giInstanceTriangleCount(instance: u32) -> u32
 * // World-space corners of one triangle, for voxelizers and debug drawing.
 * fn giInstanceTriangle(instance: u32, triangle: u32) -> GiTriangle
 * fn giMaterialAt(instance: u32, triangle: u32) -> u32
 * fn giLightCount() -> u32
 * ```
 *
 * The table layouts behind these functions are private and may change.
 */
export interface SceneCopy {
    readonly module: WgslModule;
    /** Counts of live objects. */
    readonly counts: {
        readonly meshes: number;
        readonly instances: number;
        readonly triangles: number;
        readonly lights: number;
        readonly materials: number;
    };
    /** GPU bytes held by the copy (geometry, tables, atlas). */
    readonly memoryBytes: number;
}

/**
 * Why a region changed. Techniques may treat these differently: a lighting
 * change needs probes re-traced, a geometry change may also need probes
 * re-classified or moved.
 */
export type ChangeReason = 'geometry' | 'material' | 'light';

/** A part of the world whose lighting is out of date. */
export interface DirtyRegion {
    readonly bounds: Aabb;
    readonly reason: ChangeReason;
    /** The object behind the change, when there is one, for debug output. */
    readonly source?: InstanceHandle | LightHandle;
}

/**
 * One light that changed since the previous `encode`: what it was and what it
 * is now, in the platform's light units (contract/scene.ts). Technique-facing;
 * hosts never see it. Several updates of one light in a frame arrive as one
 * change (the state before the first, the state after the last), and a light
 * whose state came back to where it started is left out.
 *
 * Why the light itself and not just a box: a technique can then work out how
 * much the light at a point changed from the light's own numbers (DDGI's
 * `ddgi/mark`), so noise in its own estimates can never fake a change
 * (docs/design/m2-lights.md §1, §4.2).
 */
export interface LightChange {
    readonly light: LightHandle;
    /**
     * The light's index in the packed light table after this frame's changes (`giLight(index)`); -1 when it was destroyed.
     * For changes carried over from frames a technique didn't see (`mergeSceneChanges`), it is the index after the frame the
     * change came from, and can be stale if a later light was destroyed. Use `light` to identify it; `before`/`after` carry the values.
     */
    readonly index: number;
    /** The light before this frame's changes; null when it was created this frame. */
    readonly before: LightDescriptor | null;
    /** The light now; null when it was destroyed this frame. */
    readonly after: LightDescriptor | null;
    /**
     * How far from the light, in metres, a change to it can matter, for the
     * light before and after (0 for a missing side; Infinity for a directional
     * light). It is the distance where the light's irradiance, without three's
     * range window, falls below 1/1,000 of the scene's reference irradiance:
     * the brightest directional light's, or, with none, the brightest local
     * light's at 1 m. So a dim lamp gets a small radius and a bright one a
     * large one. A light's `range`, when set, caps it. Used only to bound
     * which probes a change can touch, never in shading.
     */
    readonly influenceRadius: { readonly before: number; readonly after: number };
}

/**
 * Everything that changed since the previous `encode`, already merged.
 * A moved object produces two regions (where it was, and where it is now).
 */
export interface SceneChanges {
    readonly regions: readonly DirtyRegion[];
    /**
     * A change with no useful bounds: a directional light or the sun moved,
     * the environment changed, a light group's contents changed, or more
     * lights changed than are listed one by one (`lightsOverflowed`).
     */
    readonly global: readonly ('sun' | 'environment' | 'lights')[];
    /** Instances whose transform changed but which don't contribute to GI. Receivers only; usually ignorable. */
    readonly movedReceivers: number;
    /**
     * Every light that was created, changed or destroyed, with its old and new
     * state, up to 1,024 a frame (`MAX_LISTED_LIGHT_CHANGES` in
     * platform/scene/change-log.ts). Empty when more changed: then
     * `lightsOverflowed` is true and the frame counts as a global light change.
     * Directional lights are listed too, and also flag `'sun'` in `global`.
     */
    readonly lights: readonly LightChange[];
    /** More lights changed this frame than are listed: `lights` is empty and `global` has `'lights'`. */
    readonly lightsOverflowed: boolean;
}
