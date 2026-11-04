/**
 * The platform's GPU copy of the scene, as techniques see it, and the changes
 * since the last frame.
 *
 * Techniques never talk to the host. They read the scene through this copy,
 * the query backends and hit shading, and they learn what moved from
 * {@link SceneChanges}.
 */

import type { InstanceHandle, LightHandle } from './handles';
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
 * Everything that changed since the previous `encode`, already merged.
 * A moved object produces two regions (where it was, and where it is now).
 */
export interface SceneChanges {
    readonly regions: readonly DirtyRegion[];
    /**
     * A change with no useful bounds: a directional light or the sun moved,
     * the environment changed, or a light group's contents changed.
     */
    readonly global: readonly ('sun' | 'environment' | 'lights')[];
    /** Instances whose transform changed but which don't contribute to GI. Receivers only; usually ignorable. */
    readonly movedReceivers: number;
}
