/**
 * The platform's GPU copy of the scene, as techniques see it, and the changes
 * since the last frame.
 *
 * Techniques never talk to the host. They read the scene through this copy,
 * the query backends and hit shading, and they learn what moved from
 * {@link SceneChanges}.
 */

import type { InstanceHandle, LightHandle } from './handles';
import type { DirectionalLightDescriptor, LightDescriptor } from './scene';
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
    /**
     * The scene's first directional light as it is now: light 0 of the light
     * table (directional lights come first, `giDirectionalLightCount`), which
     * hit shading's `sunApart` treats as the sun. Null without directional
     * lights. It stays the same light until it is destroyed (then the last
     * directional light takes its place) or, with none before, one is created.
     *
     * Why techniques read it: a sun layer is weighed by the sun's current
     * colour x intensity (DDGI, T2.6b), and has to know when the sun became
     * another light.
     */
    readonly firstDirectionalLight: { readonly light: LightHandle; readonly descriptor: DirectionalLightDescriptor } | null;
    /**
     * Content hashes of the scene as it is now (SPEC, "content fingerprints
     * for saved bakes"): one per part, so a saved field can say what changed
     * since it was baked. platform/scene/scene-fingerprint.ts lists what each
     * part covers. Worked out on the CPU when called (a few ms, for a save
     * or a load); never in a frame.
     */
    fingerprint(): SceneFingerprint;
}

/**
 * What {@link SceneCopy.fingerprint} gives: a hash (16 hex digits) per part
 * of the scene, and the version of the rules that made them. Two
 * fingerprints of different versions are never compared part by part.
 */
export interface SceneFingerprint {
    readonly version: number;
    /** Each GI instance's mesh content, transform and visibility. */
    readonly geometry: string;
    /** Each GI instance's materials, with their textures by content. */
    readonly materials: string;
    /** The content of every texture those materials use. */
    readonly textures: string;
    /** Every light's numbers and shadow parameters. */
    readonly lights: string;
    /** The environment's settings. */
    readonly sky: string;
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
    /**
     * For a geometry region from a move (`setTransform`, `setTransforms`):
     * how far the instance moved, in metres, as the largest distance any
     * corner of its mesh's box travelled (so a turn in place counts too).
     * Both regions of a move (where it was and where it is) carry it.
     * A deforming mesh's region (new positions from `setDeformedPositions`)
     * carries the largest distance any vertex travelled since the positions
     * before, in world metres (its instance's largest scale applied). For
     * positions given on the GPU, which the platform can't read, it is the
     * diagonal of the mesh's `deformBounds` box instead: the most any vertex
     * inside the box can travel.
     * Undefined for everything else: an instance created, destroyed, shown,
     * hidden or given a new mesh is a whole change there.
     *
     * Why: a technique can then size a move's effect by how far the object
     * went, so a drag that moves a little every frame isn't a full change
     * every frame (docs/design/m3-moving.md §1, §5.1).
     */
    readonly moved?: number;
    /**
     * For a geometry region about one instance: its slot, the index rays
     * report it by (`GiHit.instance`). Lets a technique follow an object
     * across frames and recognise its hits (DDGI's mover mask). Regions
     * merged because a frame had more than `MAX_DIRTY_REGIONS` have none.
     */
    readonly instance?: number;
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
    /**
     * Where receivers are now: the world box of each visible instance that
     * doesn't contribute to GI (`contributesToGI: false`: characters, small
     * props) and was moved, created, shown or deformed. Rays never see these
     * objects, so nothing about the lighting changed; but they need GI where
     * they are, and a technique may have to wake probes there (DDGI's probes
     * in empty space, docs/design/m3-moving.md §5.3). Only the new place is
     * listed: where a receiver left needs nothing.
     *
     * Up to `MAX_RECEIVER_BOXES` (platform/scene/change-log.ts) a frame are
     * listed one by one; past that they are merged into one box.
     */
    readonly receivers: readonly Aabb[];
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
