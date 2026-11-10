/**
 * Mirrors a three.js scene into the platform's scene API (`gi.scene`), and
 * each frame sends only what changed, with a per-frame cost that doesn't
 * grow with the number of objects.
 *
 * What it mirrors (conversions in convert.ts):
 * - **Meshes:** one platform mesh per `BufferGeometry` (shared geometries stay
 *   shared), with positions, indices (made up for non-indexed geometry),
 *   vertex normals, uv, uv1 (only when a material samples channel 1) and
 *   groups.
 * - **Materials:** one platform material per three material (convert.ts,
 *   `materialDescriptor`); one platform texture per three texture and
 *   version, from its image, 8-bit data or compressed mips (`textureSource`).
 *   The platform samples them at ray hits.
 * - **Instances:** one per `Mesh`, and one per instance of an
 *   `InstancedMesh` (its world matrix times the instance matrix; instances
 *   past `count` are hidden). Visibility includes the parents', as three
 *   draws it. `userData.giContributes` says whether an object takes part in GI
 *   rays (user-data.ts).
 * - **See-through materials** (glass, alpha cutouts, blended decals) are
 *   the platform's business: its rays honour `blocksLight`, `alphaMode` and
 *   single-sidedness triangle by triangle (M4), so every mesh is sent whole.
 * - **Lights:** directional, point and spot, with `userData.giLightGroup`.
 *   Hidden lights are sent with intensity 0.
 * - **Environment:** environment.ts, fed with the `SkyMesh` and the sun found.
 *
 * **How changes are found.** three keeps no version for world matrices or
 * visibility, so finding a moved object means looking at it. Looking at every
 * object every frame costs in proportion to the scene (1.2 ms a frame on
 * Bistro interior's 2,062 meshes, T1.7), so the work is bounded instead.
 * Since T3.7 three tells the sync about the two commonest changes itself
 * (change-watch.ts):
 * - **Added and removed objects:** three's `childadded` and `childremoved`
 *   events, heard on every object of the scene, mark the child, so it is
 *   created or destroyed in the next `update`, as with `markChanged`.
 * - **Moves:** each mirrored mesh's `updateMatrixWorld` is hooked. Right
 *   after three computes the matrix (in its render), the hook compares it with
 *   the one last sent and, if it differs, puts the mesh in the hot set. So an
 *   unmarked move reaches GI in the next `update`, and the walk no longer
 *   compares transforms of hooked meshes while three updates the scene's
 *   matrices itself (`scene.matrixWorldAutoUpdate`, three's default).
 *   That assumes three renders this scene (or something calls its
 *   `updateMatrixWorld`) every frame. Two moves are then missed until the
 *   mesh is visited for another reason: a host that leaves auto-update on
 *   but never renders the scene with three, and a mesh under an ancestor
 *   whose own `updateMatrixWorld` doesn't call its children's. Such hosts
 *   call `markChanged`, or turn `scene.matrixWorldAutoUpdate` off, which
 *   brings the walk's transform compares back.
 * - **Instance matrices:** each `InstancedMesh`'s `instanceMatrix.version`
 *   is compared every update (an integer each), so edited instances reach GI
 *   in the next update, with three's update ranges (below).
 * - **Deforming meshes** (deforming.ts): skinned, morphed and
 *   `userData.giDeforming` meshes that contribute to GI are created
 *   `deforming`, each with its own platform mesh, and every update sends a
 *   new pose through `setDeformedPositions` when the pose's inputs changed.
 *
 * What three doesn't announce is still found by bounded work:
 * - **The walk:** a depth-first walk over the scene that resumes where it
 *   stopped, `objectsPerFrame` objects a frame. It finds new objects, checks
 *   each object's transform (unless its hook reports moves), visibility,
 *   materials and GI flag, and at the end of each full pass removes what it
 *   didn't meet. Since T3.7 it is the safety net for what three's events and
 *   hooks miss: a parent's `visible`, a matrix written by hand on a scene
 *   that doesn't update its own, material swaps. So a static object's first
 *   move, a new object and a removal are seen within one pass:
 *   ceil(objects in the scene / objectsPerFrame) frames (with the default
 *   64: Sponza with the curtains, 596 objects, 10 frames; Bistro interior,
 *   2,622 objects, 41 frames, about 0.7 s at 60 fps; a removal that happens
 *   mid-pass can take up to two passes). The first pass over a scene is done whole, so a scene
 *   is complete from the first frame.
 * - **The hot set:** an object found moved (or hidden or shown) is checked
 *   every frame until it has been still for {@link HOT_FRAMES} frames, so
 *   motion reaches GI every frame once it has been seen.
 * - **Moves go out in one batch:** every transform found changed in an
 *   `update` is collected and sent with one `setTransforms` call at its end
 *   (T2.M1; docs/design/m3-moving.md §6), so many movers cost the platform
 *   one call, not one each.
 * - **`InstancedMesh` update ranges:** when only the instance matrices
 *   changed and three's `instanceMatrix.updateRanges` name the instances
 *   that did, only those are looked at and sent. three clears the ranges
 *   when it uploads them during `render`, and `ThreeGI.update` runs before
 *   `render` (the documented order), so they are still there. They're used
 *   only when the mesh was also checked in the previous update; otherwise
 *   ranges from frames in between were already cleared, and every instance
 *   is compared.
 * - **`markChanged(object)`:** the host's shortcut. The object and everything
 *   under it are checked in the next `update` and join the hot set: added,
 *   moved, edited and removed objects all reach GI in that same update.
 * - **Lights** (T2.1, m2-lights.md §1): the first {@link ALWAYS_CHECKED_LIGHTS}
 *   lights and every directional light are checked every frame; the rest
 *   share the walk's rules: `lightsPerFrame` of them a frame, round robin,
 *   and a light found changed (or marked) is checked every frame until it
 *   has been still for {@link HOT_FRAMES} frames. A scene with a few lamps
 *   behaves as before; one with 1,024 candles costs 64 + 64 light checks a
 *   still frame, not 1,024. An unmarked change to a light past the first 64
 *   is seen within ceil((lights - 64) / lightsPerFrame) frames; a hidden or
 *   removed one too. `markChanged(light)` sends it the next update.
 * - **Edits:** geometry attribute versions and material values
 *   ({@link MaterialSnapshot}; three bumps `material.version` only on
 *   `needsUpdate`), `editChecksPerFrame` of each a frame, round robin.
 *
 * Not mirrored, with a one-time warning: `BatchedMesh`, ambient, hemisphere,
 * light-probe and rect-area lights (the last three also add diffuse light on
 * top of GI in three), point and spot lights whose `decay` isn't 2.
 * Morph targets and skinning of meshes that don't contribute to GI use the
 * bind pose (deforming.ts).
 * Camera layers aren't mirrored either: GI follows `visible` only, so an
 * object on a layer the camera doesn't see still blocks and bounces light
 * (hide it, or set `userData.giContributes = false`).
 *
 * It reads world matrices as they are: `ThreeGI.update` leaves them as
 * three's last render computed them (three-gi.ts says why).
 */

import { Matrix4, type Layers, type BufferAttribute, type BufferGeometry, type DirectionalLight, type InstancedMesh, type InterleavedBufferAttribute, type Light, type Material, type Mesh, type Object3D, type PointLight, type Scene, type Texture } from 'three';
import type { Aabb, EnvironmentDescriptor, InstanceHandle, LightChanges, LightDescriptor, LightHandle, MaterialDescriptor, MaterialHandle, MeshHandle, SceneApi, TextureHandle } from '@pmndrs/ddgi-core';
import { LIGHT_NUMBER_COUNT, MaterialSnapshot, PHYSICAL_DECAY, readLightNumbers, lightDescriptor, luminance, materialDescriptor, materialSlotCount, meshDescriptor, textureDescriptor, unreadableTextureReason, usesUvChannel1 } from './convert.ts';
import { EnvironmentSync, type EnvironmentSources, type PmndrsSkyLike, type SkyMeshLike, type Warn } from './environment.ts';
import { hookMatrixWorld, matrixHookInPlace, StructureWatch, unhookMatrixWorld, type MatrixHook } from './change-watch.ts';
import { deformBoundsFor, deformKind, poseSourceFor, type DeformKind, type PoseSource } from './deforming.ts';
import type { GpuPoseContext } from './gpu-skinning.ts';
import type { WebGPURenderer } from 'three/webgpu';
import { contributesFlag, contributesToGI } from './user-data.ts';

/** Floats in a 4x4 matrix. */
const MATRIX_FLOATS = 16;

/**
 * Objects the walk looks at per frame. A frame's sync starts with its data
 * cold in the cache (three has just rendered), so each object costs about
 * 0.7 µs on the RTX 3060 laptop; 64 keep a still frame of Bistro interior
 * near 0.12 ms. The trade is how late a static object's first unmarked move,
 * a new object or a removal is seen: one pass, ceil(objects / 64) frames
 * (Sponza with the curtains, 596 objects: 10 frames; Bistro interior, 2,622:
 * 41 frames, about 0.7 s at 60 fps). `markChanged` makes it the next update.
 */
export const DEFAULT_OBJECTS_PER_FRAME = 64;

/**
 * Geometries, materials and objects' GI flags checked for edits per frame,
 * each. An unmarked edit is seen within (records / this) frames: on Bistro
 * interior about 260 frames for a geometry, 9 for a material. Hosts that
 * edit call `markChanged` to have it the next update.
 */
export const DEFAULT_EDIT_CHECKS_PER_FRAME = 8;

/** An object stays in the hot set (checked every frame) until it has been still this many frames: a second at 60 fps. */
export const HOT_FRAMES = 60;

/**
 * Lights checked every frame whatever else happens: the first this many in
 * the order they were found, and every directional light (the sun is picked
 * from them each frame). A scene with a few lamps sees every light change in
 * the frame it happens, as before T2.1. 64 is the light table's starting
 * size (m2-lights.md §1).
 */
export const ALWAYS_CHECKED_LIGHTS = 64;

/**
 * Lights past {@link ALWAYS_CHECKED_LIGHTS} checked per frame, round robin.
 * A light check costs about as much as an object's (a world matrix and a
 * few numbers read), so 64 keeps a still frame with 1,024 candles near the
 * cost of one with 128 lights; an unmarked candle change is seen within
 * 15 frames (a quarter of a second at 60 fps).
 */
export const DEFAULT_LIGHTS_PER_FRAME = 64;

type Attribute = BufferAttribute | InterleavedBufferAttribute;

interface GeometryRecord {
    readonly geometry: BufferGeometry;
    handle: MeshHandle;
    /** Object records using it. */
    users: number;
    /** Whether `uv1` was passed. */
    uv1: boolean;
    slotCount: number;
    /** Attributes and their versions when last sent: position, index, normal, uv, uv1. */
    readonly attributes: (Attribute | null)[];
    readonly versions: number[];
    /** Where it sits in the edit-check list. */
    listIndex: number;
    /**
     * A deforming mesh's shape (deforming.ts), or null. A deforming record
     * belongs to one three mesh, never shared: each mesh has its own pose.
     */
    readonly deform: DeformingShape | null;
}

/** What a deforming geometry record keeps: how it deforms, its pose box, and where its poses come from. */
interface DeformingShape {
    readonly kind: DeformKind;
    readonly bounds: Aabb;
    readonly pose: PoseSource;
}

interface MaterialRecord {
    readonly material: Material;
    handle: MaterialHandle;
    /** The descriptor last sent to the platform (see `sentMaterial`). */
    sent: MaterialDescriptor;
    users: number;
    snapshot: MaterialSnapshot;
    scratch: MaterialSnapshot;
    /** Texture handles the platform material currently uses. */
    textureHandles: TextureHandle[];
    /** Where it sits in the edit-check list. */
    listIndex: number;
}

interface TextureRecord {
    readonly texture: Texture;
    readonly handle: TextureHandle;
    readonly version: number;
}

interface ObjectRecord {
    readonly object: Mesh;
    readonly geometry: GeometryRecord;
    /** `geometry.geometry`, kept here so the identity check reads one record, not two. */
    readonly geometryObject: BufferGeometry;
    materials: MaterialRecord[];
    /** `mesh.material` when the material list was last checked: unchanged identity means unchanged list (arrays are re-read anyway). */
    materialKey: Material | Material[];
    readonly instances: InstanceHandle[];
    /** InstancedMesh: world matrices last sent, 16 per instance. (A plain mesh's is in the shared pool, at `matrixSlot`.) */
    matrices: Float64Array;
    /** Plain mesh: where its last-sent world matrix sits in the shared pool. -1 for an InstancedMesh. */
    matrixSlot: number;
    /** Per instance: last visibility sent. */
    readonly visible: boolean[];
    /** The object's GI flag (`userData.giContributes`, user-data.ts) when it was created. */
    readonly contributes: boolean;
    /** Fixed facts about the object, read once. */
    readonly instanced: boolean;
    readonly skinned: boolean;
    /** InstancedMesh: the mesh's own world matrix and the instance-matrix version the matrices were made from. */
    readonly parentMatrix: Float64Array;
    instanceMatrixVersion: number;
    /** InstancedMesh: the sync's frame when its instances were last checked (update ranges are trusted only from the next frame). */
    instancesCheckedFrame: number;
    /** The walk pass that last met it; at the end of a pass, records it didn't meet are removed. */
    seenPass: number;
    /** While in the hot set: the frame it may leave if nothing changes before. */
    hotUntil: number;
    /** In the hot list now. */
    hot: boolean;
    /** Plain mesh: the visibility last sent (an InstancedMesh keeps one per instance in `visible`). */
    shown: boolean;
    /** Where it sits in the object list the GI-flag rotation walks. */
    listIndex: number;
    destroyed: boolean;
    /** The `updateMatrixWorld` hook that reports its moves (change-watch.ts). */
    matrixHook: MatrixHook | null;
}

interface LightRecord {
    readonly light: Light;
    readonly handle: LightHandle;
    descriptor: LightDescriptor;
    /** The light's numbers when last sent, and a second array this frame's are read into (readLightNumbers). */
    numbers: Float64Array;
    scratch: Float64Array;
    seenPass: number;
    /** Where it sits in the light list. */
    listIndex: number;
    readonly directional: boolean;
    /** Whether it and its parents were visible when last checked (for picking the sun). */
    visible: boolean;
    /** The frame it was last checked in, so no light is checked twice in one update. */
    checkedFrame: number;
    /** In the hot list until this frame (checked every frame). */
    hotUntil: number;
    hot: boolean;
    destroyed: boolean;
}

/** What the last `update` did, for stats and tests. */
export interface SyncFrameReport {
    created: number;
    destroyed: number;
    transforms: number;
    visibility: number;
    materials: number;
    meshes: number;
    lights: number;
    /** Lights looked at (the first 64 and directional ones, hot ones, and this frame's share of the rest). */
    lightsChecked: number;
    /** Objects the walk looked at. */
    walked: number;
    /** Objects checked because they were in the hot set or marked changed. */
    hot: number;
    /** Deformed poses sent (`setDeformedPositions`). */
    poses: number;
}

export interface SceneSyncOptions {
    /** Who made the device: needed to upload environment textures. */
    readonly device: GPUDevice;
    /**
     * The renderer that draws the scene, on `device`. Given, skinned meshes
     * that contribute to GI are posed on the GPU by three's own skinning
     * (gpu-skinning.ts); without it they are posed on the CPU.
     */
    readonly renderer?: WebGPURenderer;
    /** Where warnings go (each distinct message once). Default `console.warn`. */
    readonly warn?: (message: string) => void;
    /** Objects the walk looks at per frame ({@link DEFAULT_OBJECTS_PER_FRAME}). */
    readonly objectsPerFrame?: number;
    /** Geometries, and separately materials, checked for edits per frame ({@link DEFAULT_EDIT_CHECKS_PER_FRAME}). */
    readonly editChecksPerFrame?: number;
    /** Lights past the first {@link ALWAYS_CHECKED_LIGHTS} checked per frame ({@link DEFAULT_LIGHTS_PER_FRAME}). */
    readonly lightsPerFrame?: number;
}

/** Zeroes a report in place (no allocation each frame). */
function resetReport(report: SyncFrameReport): void {
    report.created = 0;
    report.destroyed = 0;
    report.transforms = 0;
    report.visibility = 0;
    report.materials = 0;
    report.meshes = 0;
    report.lights = 0;
    report.lightsChecked = 0;
    report.walked = 0;
    report.hot = 0;
    report.poses = 0;
}

function emptyReport(): SyncFrameReport {
    return { created: 0, destroyed: 0, transforms: 0, visibility: 0, materials: 0, meshes: 0, lights: 0, lightsChecked: 0, walked: 0, hot: 0, poses: 0 };
}

function positiveInteger(value: number | undefined, fallback: number, name: string): number {
    const chosen = value ?? fallback;
    if (!(Number.isInteger(chosen) && chosen >= 1)) throw new RangeError(`${name} must be a whole number, at least 1; got ${chosen}`);
    return chosen;
}

/** Where each attribute sits in a geometry record's `attributes` and `versions`. */
const GEOMETRY_ATTRIBUTE = { position: 0, index: 1, normal: 2, uv: 3, uv1: 4 } as const;

/** Whether the attribute in a slot was replaced or edited since it was sent. */
function attributeChanged(current: Attribute | null | undefined, record: { readonly attributes: (Attribute | null)[]; readonly versions: number[] }, slot: number): boolean {
    const attribute = current ?? null;
    return attribute !== record.attributes[slot] || attributeVersion(attribute) !== record.versions[slot];
}

function attributeVersion(attribute: Attribute | null): number {
    if (!attribute) return -1;
    return (attribute as InterleavedBufferAttribute).isInterleavedBufferAttribute ? (attribute as InterleavedBufferAttribute).data.version : (attribute as BufferAttribute).version;
}

/** Who is visiting a mesh: the walk, the hot list, or a `markChanged` call. */
type VisitSource = 'walk' | 'hot' | 'marked';

type ObjectFlags = Object3D & {
    isMesh?: boolean;
    isLight?: boolean;
    isSkyMesh?: boolean;
    isBatchedMesh?: boolean;
    isGILight?: boolean;
    isGIVolume?: boolean;
    isLightProbeGrid?: boolean;
    isAmbientLight?: boolean;
    isHemisphereLight?: boolean;
    isLightProbe?: boolean;
    isRectAreaLight?: boolean;
    isPointLight?: boolean;
    isSpotLight?: boolean;
    isDirectionalLight?: boolean;
};

export class SceneSync {
    /**
     * An environment descriptor that replaces whatever three's scene says
     * (environment.ts, source 1). `null` reads it from the scene.
     */
    environment: EnvironmentDescriptor | null = null;
    /** A pmndrs/sky `Sky` to take the sky from (environment.ts, source 2). */
    sky: PmndrsSkyLike | null = null;
    /**
     * `LightProbeGrid`s three will light with: visible, intensity above 0 and,
     * when `update` is given the camera's layers, on one of them (three only
     * gathers lights the camera's layers include). They share three's
     * light-probe slot with the GI light.
     */
    otherProbeLights = 0;
    /**
     * Mirrored materials with their own `envMap`. three lights those with it
     * (diffuse included) even without `scene.environment`, so the GI light's
     * environment-diffuse workaround needs coverage then too.
     */
    materialsWithEnvMap = 0;

    private readonly api: SceneApi;
    private readonly warnOnce: Warn;
    private readonly environmentSync: EnvironmentSync;
    private readonly objectsPerFrame: number;
    private readonly editChecksPerFrame: number;
    private readonly lightsPerFrame: number;

    private readonly geometries = new Map<BufferGeometry, GeometryRecord>();
    private readonly materials = new Map<Material, MaterialRecord>();
    private readonly textures = new Map<Texture, TextureRecord>();
    private readonly textureHandleUsers = new Map<TextureHandle, number>();
    private readonly objects = new Map<Object3D, ObjectRecord>();
    private readonly lights = new Map<Light, LightRecord>();
    /** Arrays beside the maps, walked by index each frame: a Map or Set iterator is an allocation. */
    private readonly objectList: ObjectRecord[] = [];
    private readonly lightList: LightRecord[] = [];
    /** Directional lights, also in lightList: checked every frame, and the sun is picked from them. */
    private readonly directionalLights: LightRecord[] = [];
    /** Lights found changed or marked, checked every frame until still for HOT_FRAMES. */
    private readonly hotLights: LightRecord[] = [];
    /** Where the round robin over the lights past ALWAYS_CHECKED_LIGHTS goes on. */
    private lightCursor = 0;
    private readonly probeGrids: Light[] = [];
    private objectCursor = 0;

    /** Round-robin edit checks: the records in a list, and where the next check starts. */
    private readonly geometryList: GeometryRecord[] = [];
    private readonly materialList: MaterialRecord[] = [];
    private geometryCursor = 0;
    private materialCursor = 0;

    private readonly hotList: ObjectRecord[] = [];
    private readonly marked = new Set<Object3D>();
    /** three's `childadded` and `childremoved` on the walked scene, marking the child (change-watch.ts). */
    private readonly structure = new StructureWatch((child) => this.marked.add(child));
    /** The renderer and device for skinned meshes' GPU poses, or null to pose them on the CPU. */
    private readonly gpuPoses: GpuPoseContext | null;
    /** Objects whose meshes deform (deforming.ts): their poses are checked every update. */
    private readonly deformingList: ObjectRecord[] = [];
    /** InstancedMeshes: their instance matrices' version is compared every update (checkInstancedMeshes). */
    private readonly instancedList: ObjectRecord[] = [];
    /**
     * three's render updates the walked scene's world matrices itself
     * (`scene.matrixWorldAutoUpdate`), so every hooked mesh's move reaches its
     * hook, and the walk can skip comparing their transforms. Read each update.
     */
    private hooksSeeEveryMove = false;

    /** The resumable walk: its scene, a stack of objects still to visit with their parents' visibility, and the pass number. */
    private walkScene: Scene | null = null;
    private readonly walkStack: Object3D[] = [];
    private readonly walkParentVisible: boolean[] = [];
    private pass = 0;
    /**
     * Object records in the order the current pass met them so far, and the
     * previous pass's beyond that. A scene that didn't change shape is walked
     * in the same order, so the next mesh's record is usually the next entry:
     * read in sequence instead of looked up in a Map.
     */
    private readonly passOrder: ObjectRecord[] = [];
    private passCursor = 0;
    private passSkyMesh: SkyMeshLike | null = null;
    private skyMesh: SkyMeshLike | null = null;

    private frame = 0;
    /** The camera's layers this update, if given (for `otherProbeLights`). */
    private cameraLayers: Layers | null = null;
    /** Set by {@link placement}: whether the object and all its parents are visible. */
    private placementVisible = false;
    private sun: DirectionalLight | null = null;
    private readonly scratchMatrix = new Matrix4();
    /** This update's changed transforms, sent with one `setTransforms` call (TransformBatch). */
    private readonly transforms = new TransformBatch();
    /** Plain meshes' last-sent world matrices, one block. */
    private readonly matrixPool = new MatrixPool();
    private readonly warned = new Set<string>();

    /** What the last `update` did. */
    readonly report: SyncFrameReport = emptyReport();
    /** Filled in place each frame for the environment sync. */
    private readonly environmentSources: { -readonly [Key in keyof EnvironmentSources]: EnvironmentSources[Key] } = {};

    constructor(api: SceneApi, options: SceneSyncOptions) {
        this.api = api;
        const warn = options.warn ?? ((message: string) => console.warn(`@pmndrs/ddgi/three: ${message}`));
        this.warnOnce = (message) => {
            if (this.warned.has(message)) return;
            this.warned.add(message);
            warn(message);
        };
        this.environmentSync = new EnvironmentSync(api, options.device, this.warnOnce);
        this.gpuPoses = options.renderer ? { device: options.device, renderer: options.renderer } : null;
        this.objectsPerFrame = positiveInteger(options.objectsPerFrame, DEFAULT_OBJECTS_PER_FRAME, 'objectsPerFrame');
        this.editChecksPerFrame = positiveInteger(options.editChecksPerFrame, DEFAULT_EDIT_CHECKS_PER_FRAME, 'editChecksPerFrame');
        this.lightsPerFrame = positiveInteger(options.lightsPerFrame, DEFAULT_LIGHTS_PER_FRAME, 'lightsPerFrame');
    }

    /** How many things are mirrored. */
    get counts(): { meshes: number; materials: number; textures: number; objects: number; instances: number; lights: number } {
        let instances = 0;
        for (const record of this.objects.values()) instances += record.instances.length;
        return { meshes: this.geometries.size, materials: this.materials.size, textures: this.textures.size, objects: this.objects.size, instances, lights: this.lights.size };
    }

    /**
     * The GI material last sent to the platform for a three material, or
     * undefined if no mirrored mesh uses it: what GI actually sees, for
     * checks that must give another renderer the same values.
     */
    sentMaterial(material: Material): MaterialDescriptor | undefined {
        return this.materials.get(material)?.sent;
    }

    /**
     * Whether a mirrored mesh's instances were sent into GI's rays
     * (`contributesToGI`), or undefined if the mesh isn't mirrored. Since M4
     * that is `userData.giContributes` alone: no mesh is left out for its
     * materials.
     */
    sentContributes(mesh: Mesh): boolean | undefined {
        return this.objects.get(mesh)?.contributes;
    }

    /** The platform texture last made for a three texture, or undefined if none is (not used, or not readable). */
    sentTexture(texture: Texture): TextureHandle | undefined {
        return this.textures.get(texture)?.handle;
    }

    /** How many times the environment was sent, and how many textures uploaded for it. */
    get environmentUpdates(): { updates: number; uploads: number } {
        return { updates: this.environmentSync.updates, uploads: this.environmentSync.uploads };
    }

    /**
     * Says that `object` (and everything under it) was added, moved, edited
     * or removed, so the next `update` checks it at once instead of when the
     * walk gets to it. Cheap; call it as often as you like.
     */
    markChanged(object: Object3D): void {
        this.marked.add(object);
    }

    /** Sends what changed: marked objects, the hot set, this frame's share of the walk, lights, edits, environment. */
    update(scene: Scene, cameraLayers?: Layers): SyncFrameReport {
        this.cameraLayers = cameraLayers ?? null;
        this.frame++;
        resetReport(this.report);
        // A new scene is walked whole at once; afterwards, a share each frame.
        const fresh = scene !== this.walkScene;
        // An assumption, not a test: three must also render this scene (file header, "Moves").
        this.hooksSeeEveryMove = scene.matrixWorldAutoUpdate;
        if (fresh) this.startOver(scene);

        if (this.marked.size > 0) this.checkMarked(scene);
        this.checkHot(scene);
        this.checkInstancedMeshes(scene);
        if (!fresh) this.walk();
        this.sendPoses();
        this.checkLights(scene);
        this.checkEdits();
        this.flushTransforms();
        const sources = this.environmentSources;
        sources.override = this.environment;
        sources.sky = this.sky;
        sources.skyMesh = this.skyMesh;
        sources.sun = this.sun;
        this.environmentSync.update(scene, sources);
        return this.report;
    }

    /** Sends the transforms collected so far in one `setTransforms` call. */
    private flushTransforms(): void {
        this.transforms.send(this.api);
    }

    /** Destroys everything this sync created in the platform scene. */
    dispose(): void {
        this.transforms.clear();
        for (const record of [...this.objects.values()]) this.destroyObject(record);
        for (const record of this.lightList) this.api.destroyLight(record.handle);
        this.lights.clear();
        this.lightList.length = 0;
        this.directionalLights.length = 0;
        this.hotLights.length = 0;
        this.hotList.length = 0;
        this.passOrder.length = 0;
        this.marked.clear();
        this.walkStack.length = 0;
        this.walkParentVisible.length = 0;
        this.walkScene = null;
        this.structure.detach();
        this.deformingList.length = 0;
        this.instancedList.length = 0;
        this.environmentSync.dispose();
    }

    // -----------------------------------------------------------------------
    // The walk
    // -----------------------------------------------------------------------

    /** A new scene: walk all of it now, so it is complete from the first frame. */
    private startOver(scene: Scene): void {
        this.walkScene = scene;
        // Listening starts before the walk, so nothing added from here on is missed.
        this.structure.attach(scene);
        this.walkStack.length = 0;
        this.walkParentVisible.length = 0;
        this.beginPass();
        this.walk(Number.POSITIVE_INFINITY);
    }

    private beginPass(): void {
        this.pass++;
        this.passCursor = 0;
        this.passSkyMesh = null;
        this.walkStack.push(this.walkScene!);
        this.walkParentVisible.push(true);
    }

    /** Visits up to `budget` objects where the walk stopped; at the end of a pass, removes what it didn't meet. */
    private walk(budget: number = this.objectsPerFrame): void {
        if (this.walkStack.length === 0) this.beginPass();
        const stack = this.walkStack;
        const parentVisible = this.walkParentVisible;
        let visited = 0;
        while (visited < budget && stack.length > 0) {
            const object = stack.pop()!;
            const visible = parentVisible.pop()! && object.visible;
            visited++;
            if (!this.visitObject(object, visible, 'walk')) continue;
            // Children in reverse, so they come off the stack in order.
            const children = object.children;
            for (let index = children.length - 1; index >= 0; index--) {
                stack.push(children[index]);
                parentVisible.push(visible);
            }
        }
        this.report.walked += visited;
        if (stack.length === 0) this.endPass();
    }

    /**
     * Removes what the pass didn't meet, unless it is still in the scene: an
     * object moved mid-pass into a part already walked is missed by this pass
     * but not gone, so its parents are followed up to the scene first.
     * Deleting from a Map while iterating it is safe, and allocates nothing.
     */
    private endPass(): void {
        const scene = this.walkScene!;
        for (const record of this.objects.values()) {
            if (record.seenPass === this.pass) continue;
            if (this.placement(record.object, scene)) record.seenPass = this.pass;
            else this.destroyObject(record);
        }
        // Backwards, since destroying swaps the last record into the hole.
        for (let index = this.lightList.length - 1; index >= 0; index--) {
            const record = this.lightList[index];
            if (record.seenPass === this.pass) continue;
            if (this.placement(record.light, scene)) record.seenPass = this.pass;
            else this.destroyLight(record);
        }
        this.passOrder.length = this.passCursor;
        this.skyMesh = this.passSkyMesh;
    }

    /**
     * One object, met by the walk, the hot set or because it was marked.
     * Returns whether to go on into its children.
     */
    private visitObject(object: Object3D, visible: boolean, source: VisitSource): boolean {
        // Flags are tested most common first: a missing property costs a walk up the prototypes.
        const flags = object as ObjectFlags;
        if (flags.isMesh) {
            if (flags.isSkyMesh) {
                // The sky itself, not geometry: it becomes the environment, and its box must not wall the scene in.
                if (visible) {
                    this.passSkyMesh = object as unknown as SkyMeshLike;
                    this.skyMesh = this.passSkyMesh;
                }
                return false;
            }
            if (flags.isBatchedMesh) this.warnOnce('BatchedMesh is not mirrored into GI yet; its objects neither block nor bounce light.');
            else this.visitMesh(object as Mesh, visible, source);
        } else if (flags.isLight) {
            this.visitLight(object as Light, visible, source);
        } else if (flags.isSkyMesh) {
            if (visible) {
                this.passSkyMesh = object as unknown as SkyMeshLike;
                this.skyMesh = this.passSkyMesh;
            }
            return false;
        }
        return true;
    }

    /**
     * Whether `object` is still in `scene` (following parents up to it); also
     * sets {@link placementVisible} to whether it and all its parents are
     * visible. Without allocating.
     */
    private placement(object: Object3D, scene: Scene): boolean {
        let visible = true;
        let node: Object3D | null = object;
        while (node) {
            visible = visible && node.visible;
            if (node === scene) {
                this.placementVisible = visible;
                return true;
            }
            node = node.parent;
        }
        this.placementVisible = false;
        return false;
    }

    // -----------------------------------------------------------------------
    // Marked objects and the hot set
    // -----------------------------------------------------------------------

    /** Marked objects and everything under them, now: in the scene, they join the hot set; gone, they are removed. */
    private checkMarked(scene: Scene): void {
        for (const root of this.marked) {
            if (!this.placement(root, scene)) {
                root.traverse((object) => {
                    const record = this.objects.get(object);
                    if (record) this.destroyObject(record);
                    const light = this.lights.get(object as Light);
                    if (light) this.destroyLight(light);
                });
                continue;
            }
            const parentVisible = root.parent ? this.placement(root.parent, scene) && this.placementVisible : true;
            this.visitMarked(root, parentVisible && root.visible);
        }
        this.marked.clear();
    }

    private visitMarked(object: Object3D, visible: boolean): void {
        this.report.hot++;
        if (!this.visitObject(object, visible, 'marked')) return;
        const record = this.objects.get(object);
        if (record) {
            // Edits too, now rather than when the round robin gets to them.
            this.checkGeometry(record.geometry);
            for (const material of record.materials) this.checkMaterial(material);
        }
        for (const child of object.children) this.visitMarked(child, visible && child.visible);
    }

    /** Every hot object, every frame; each leaves the list after HOT_FRAMES frames without a change. */
    private checkHot(scene: Scene): void {
        const list = this.hotList;
        // Backwards, so a removal (the last entry swapped in) never skips one.
        for (let index = list.length - 1; index >= 0; index--) {
            const record = list[index];
            if (!record.destroyed) {
                this.report.hot++;
                if (!this.placement(record.object, scene)) this.destroyObject(record);
                else this.visitMesh(record.object, this.placementVisible, 'hot');
            }
            if (record.destroyed || this.frame > record.hotUntil) {
                record.hot = false;
                const last = list.pop()!;
                if (index < list.length) list[index] = last;
            }
        }
    }

    private makeHot(record: ObjectRecord): void {
        record.hotUntil = this.frame + HOT_FRAMES;
        if (record.hot) return;
        record.hot = true;
        this.hotList.push(record);
    }

    // -----------------------------------------------------------------------
    // Meshes and instances
    // -----------------------------------------------------------------------

    /**
     * One mesh: its geometry and materials by identity, its transform and
     * visibility. Its GI flag (a `userData` read, one more cold cache line per
     * mesh) is read here only for hot and marked objects; the rest get it in
     * the edit-check rotation (checkEdits).
     */
    private visitMesh(mesh: Mesh, visible: boolean, source: VisitSource): void {
        let record: ObjectRecord | undefined;
        if (source === 'walk') {
            const expected = this.passOrder[this.passCursor];
            record = expected !== undefined && expected.object === mesh && !expected.destroyed ? expected : this.objects.get(mesh);
        } else {
            record = this.objects.get(mesh);
        }
        if (record && record.geometryObject !== mesh.geometry) {
            // A new geometry: the contract has no setter for an instance's mesh, so start over.
            this.destroyObject(record);
            record = undefined;
        } else if (record && source !== 'walk' && (record.contributes !== contributesFlag(mesh, record.skinned) || deformingChanged(record))) {
            // A changed GI flag, or a mesh that starts or stops deforming (`giDeforming`): new instances, and a deforming one its own mesh.
            record = this.restartObject(record, visible);
            if (!record) return;
        }
        if (!record) {
            // The walk's stack can hold an object removed since it was pushed (its removal was already
            // handled through three's event); creating it would put it back in GI until the pass ends.
            if (source === 'walk' && !this.placement(mesh, this.walkScene!)) return;
            record = this.createObject(mesh, visible, contributesToGI(mesh));
            if (!record) return;
            if (source !== 'walk') this.makeHot(record);
        } else {
            if (mesh.material !== record.materialKey || Array.isArray(mesh.material)) this.checkMaterials(record);
            if (this.checkInstances(record, visible, source !== 'walk' || !this.hookReportsMoves(record)) || source === 'marked') {
                this.makeHot(record);
            }
        }
        record.seenPass = this.pass;
        if (source === 'walk') this.passOrder[this.passCursor++] = record;
    }

    private materialForSlot(mesh: Mesh, slot: number): Material | undefined {
        const material = mesh.material;
        if (Array.isArray(material)) return material[slot] ?? material[0];
        return material;
    }

    private createObject(mesh: Mesh, visible: boolean, contributes: boolean): ObjectRecord | undefined {
        // Only a mesh that contributes deforms in GI: a receiver's shape isn't traced (deforming.ts).
        const geometryRecord = this.acquireGeometry(mesh, contributes ? deformKind(mesh) : null);
        if (!geometryRecord) return undefined;
        const materials: MaterialRecord[] = [];
        for (let slot = 0; slot < geometryRecord.slotCount; slot++) {
            const material = this.materialForSlot(mesh, slot);
            if (!material) {
                this.releaseGeometry(geometryRecord);
                for (const acquired of materials) this.releaseMaterial(acquired);
                return undefined;
            }
            materials.push(this.acquireMaterial(material));
        }
        const instanced = (mesh as InstancedMesh).isInstancedMesh ? (mesh as InstancedMesh) : null;
        const record: ObjectRecord = {
            object: mesh,
            geometry: geometryRecord,
            geometryObject: geometryRecord.geometry,
            materials,
            materialKey: mesh.material,
            instances: [],
            matrices: new Float64Array(0),
            matrixSlot: -1,
            visible: [],
            contributes,
            instanced: instanced !== null,
            skinned: (mesh as Mesh & { isSkinnedMesh?: boolean }).isSkinnedMesh === true,
            parentMatrix: new Float64Array(MATRIX_FLOATS),
            instanceMatrixVersion: -1,
            instancesCheckedFrame: -1,
            seenPass: this.pass,
            hotUntil: 0,
            hot: false,
            shown: visible,
            listIndex: this.objectList.length,
            destroyed: false,
            matrixHook: null,
        };
        this.objects.set(mesh, record);
        this.objectList.push(record);
        if (instanced) {
            this.syncInstancedMesh(record, instanced, visible, true);
        } else {
            record.matrixSlot = this.matrixPool.take();
            this.matrixPool.values.set(mesh.matrixWorld.elements, record.matrixSlot * MATRIX_FLOATS);
            this.createInstance(record, mesh.matrixWorld.elements, visible);
        }
        record.matrixHook = hookMatrixWorld(mesh, () => this.matrixWorldUpdated(record));
        if (geometryRecord.deform) this.deformingList.push(record);
        if (instanced) this.instancedList.push(record);
        return record;
    }

    // -----------------------------------------------------------------------
    // Moves reported by three (change-watch.ts)
    // -----------------------------------------------------------------------

    /**
     * Runs inside three's `updateMatrixWorld`, right after it wrote this
     * mesh's world matrix (usually in its render). Compares and queues only:
     * a mesh whose matrix differs from the one last sent joins the hot set,
     * and the next `update` sends it. A hot mesh is skipped: `update`
     * compares it anyway. (An InstancedMesh's own instance matrices aren't
     * part of its world matrix: checkInstancedMeshes looks at those.)
     */
    private matrixWorldUpdated(record: ObjectRecord): void {
        if (record.hot || record.destroyed) return;
        const sent = record.instanced ? record.parentMatrix : this.matrixPool.values;
        const offset = record.instanced ? 0 : record.matrixSlot * MATRIX_FLOATS;
        if (differs(record.object.matrixWorld.elements, sent, offset)) this.makeHot(record);
    }

    /**
     * Every InstancedMesh not already checked this update (the hot ones
     * were): a new `instanceMatrix.version` means edited instances, sent now
     * and the mesh made hot. Instance matrices need no world-matrix update to
     * be read, so an edit reaches GI in the next update, and with three's
     * update ranges from the first edit: a mesh whose version hasn't changed
     * is marked checked this frame, and in the documented order (update, then
     * render) every edit until three's next upload is then in its ranges
     * (syncInstancedMesh). One integer compare per InstancedMesh per update.
     */
    private checkInstancedMeshes(scene: Scene): void {
        const list = this.instancedList;
        for (let index = 0; index < list.length; index++) {
            const record = list[index];
            if (record.instancesCheckedFrame === this.frame) continue;
            const mesh = record.object as InstancedMesh;
            if (mesh.instanceMatrix.version === record.instanceMatrixVersion) {
                record.instancesCheckedFrame = this.frame;
                continue;
            }
            // Gone from the scene: the event or the walk removes it.
            if (!this.placement(mesh, scene)) continue;
            this.syncInstancedMesh(record, mesh, this.placementVisible, true);
            this.makeHot(record);
        }
    }

    /** Whether the record's hook sees every move, so the walk needn't compare its transform. */
    private hookReportsMoves(record: ObjectRecord): boolean {
        return this.hooksSeeEveryMove && record.matrixHook !== null && matrixHookInPlace(record.matrixHook);
    }

    // -----------------------------------------------------------------------
    // Deforming meshes (deforming.ts)
    // -----------------------------------------------------------------------

    /** Every shown deforming mesh: its pose source says whether there is a new pose, and it goes out. */
    private sendPoses(): void {
        const list = this.deformingList;
        for (let index = 0; index < list.length; index++) {
            const record = list[index];
            if (!record.shown) continue;
            const pose = record.geometry.deform!.pose.nextPose();
            if (!pose) continue;
            this.api.setDeformedPositions(record.geometry.handle, pose);
            this.report.poses++;
        }
    }

    private createInstance(record: ObjectRecord, transform: ArrayLike<number>, visible: boolean): void {
        const handle = this.api.createInstance({
            mesh: record.geometry.handle,
            materials: record.materials.map((material) => material.handle),
            transform,
            contributesToGI: record.contributes,
            visible,
            label: record.object.name || undefined,
        });
        record.instances.push(handle);
        record.visible.push(visible);
        this.report.created++;
    }

    /**
     * Destroys and re-creates an object whose instances' GI flag changed (the
     * contract has no setter for `contributesToGI`); the new record is hot.
     * Its platform mesh and materials are held across the restart, so the
     * mesh's tree isn't rebuilt and textures aren't uploaded again.
     */
    private restartObject(record: ObjectRecord, visible: boolean): ObjectRecord | undefined {
        const mesh = record.object;
        const geometry = record.geometry;
        const materials = record.materials;
        geometry.users++;
        for (const material of materials) material.users++;
        this.destroyObject(record);
        const restarted = this.createObject(mesh, visible, contributesToGI(mesh));
        this.releaseGeometry(geometry);
        for (const material of materials) this.releaseMaterial(material);
        if (restarted) this.makeHot(restarted);
        return restarted;
    }

    private destroyObject(record: ObjectRecord): void {
        // Moves collected for its instances go out first, while their handles are still live.
        this.flushTransforms();
        record.destroyed = true;
        if (record.matrixHook) unhookMatrixWorld(record.matrixHook);
        record.matrixHook = null;
        if (record.geometry.deform) this.deformingList.splice(this.deformingList.indexOf(record), 1);
        if (record.instanced) this.instancedList.splice(this.instancedList.indexOf(record), 1);
        removeFromList(this.objectList, record);
        if (record.matrixSlot >= 0) this.matrixPool.release(record.matrixSlot);
        record.matrixSlot = -1;
        for (const instance of record.instances) this.api.destroyInstance(instance);
        this.report.destroyed += record.instances.length;
        record.instances.length = 0;
        this.objects.delete(record.object);
        for (const material of record.materials) this.releaseMaterial(material);
        this.releaseGeometry(record.geometry);
    }

    /** Material per slot by identity; a changed slot means `setMaterials` on every instance. Returns whether it changed. */
    private checkMaterials(record: ObjectRecord): boolean {
        const mesh = record.object;
        const materials = record.materials;
        record.materialKey = mesh.material;
        let changed = false;
        for (let slot = 0; slot < materials.length; slot++) {
            const material = this.materialForSlot(mesh, slot);
            if (material && material !== materials[slot].material) changed = true;
        }
        if (!changed) return false;
        // Acquire the new list before releasing the old, so a material in both keeps its platform copy; and point the
        // instances at the new list before releasing, since the platform refuses to destroy a material instances still use.
        const next: MaterialRecord[] = [];
        for (let slot = 0; slot < materials.length; slot++) {
            next.push(this.acquireMaterial(this.materialForSlot(mesh, slot) ?? materials[slot].material));
        }
        record.materials = next;
        const handles = next.map((material) => material.handle);
        for (const instance of record.instances) this.api.setMaterials(instance, handles);
        for (const material of materials) this.releaseMaterial(material);
        return true;
    }

    /**
     * Visibility, and the transform when `compareTransforms` (the walk skips
     * it for meshes whose hook reports moves); returns whether anything was sent.
     */
    private checkInstances(record: ObjectRecord, visible: boolean, compareTransforms: boolean): boolean {
        const mesh = record.object;
        if (record.instanced) return this.syncInstancedMesh(record, mesh as InstancedMesh, visible, compareTransforms);
        let changed = false;
        if (record.shown !== visible) {
            this.api.setVisible(record.instances[0], visible);
            record.shown = visible;
            this.report.visibility++;
            changed = true;
        }
        if (compareTransforms && copyIfChanged(mesh.matrixWorld.elements, this.matrixPool.values, record.matrixSlot * MATRIX_FLOATS)) {
            this.transforms.add(record.instances[0], mesh.matrixWorld.elements);
            this.report.transforms++;
            changed = true;
        }
        return changed;
    }

    /**
     * An InstancedMesh: instance i's world matrix is the mesh's world matrix
     * times `instanceMatrix[i]`. Only recomputed when the mesh moved or the
     * instance matrices' version changed; then only changed instances are
     * sent. When only the instance matrices changed and three's update ranges
     * name the instances that did (see the file header), only those are
     * looked at. Instances past `count` are hidden, not destroyed, so a count
     * that goes down and up again costs nothing. Without `compareTransforms`
     * (the walk, when the hook reports moves) only the count and visibility
     * are looked at. Returns whether anything was sent.
     */
    private syncInstancedMesh(record: ObjectRecord, mesh: InstancedMesh, visible: boolean, compareTransforms: boolean): boolean {
        const before = this.report.transforms + this.report.visibility;
        const count = mesh.count;
        const parentMoved = compareTransforms && copyIfChanged(mesh.matrixWorld.elements, record.parentMatrix, 0);
        const instancesChanged = compareTransforms && mesh.instanceMatrix.version !== record.instanceMatrixVersion;
        const grew = count > record.instances.length;
        if (grew) {
            const matrices = new Float64Array(count * MATRIX_FLOATS);
            matrices.set(record.matrices);
            record.matrices = matrices;
        }
        // The ranges cover every change since the last render only if this mesh was also checked in the previous update.
        const ranges = mesh.instanceMatrix.updateRanges;
        const useRanges = instancesChanged && !parentMoved && !grew && ranges.length > 0 && record.instancesCheckedFrame === this.frame - 1;
        if (compareTransforms) record.instancesCheckedFrame = this.frame;
        if (useRanges) {
            record.instanceMatrixVersion = mesh.instanceMatrix.version;
            for (const range of ranges) {
                // Ranges count array elements; an instance is 16 of them.
                const first = Math.max(0, Math.floor(range.start / MATRIX_FLOATS));
                const end = Math.min(count, Math.ceil((range.start + range.count) / MATRIX_FLOATS));
                for (let instance = first; instance < end; instance++) this.syncInstance(record, mesh, instance, visible);
            }
        } else if (parentMoved || instancesChanged || grew) {
            record.instanceMatrixVersion = mesh.instanceMatrix.version;
            for (let instance = 0; instance < count; instance++) this.syncInstance(record, mesh, instance, visible);
        }
        for (let instance = 0; instance < record.instances.length; instance++) {
            const shown = visible && instance < count;
            if (record.visible[instance] === shown) continue;
            this.api.setVisible(record.instances[instance], shown);
            record.visible[instance] = shown;
            this.report.visibility++;
        }
        return this.report.transforms + this.report.visibility > before;
    }

    /** One instance of an InstancedMesh: created if it is new, its transform collected if it changed. */
    private syncInstance(record: ObjectRecord, mesh: InstancedMesh, instance: number, visible: boolean): void {
        const matrix = this.scratchMatrix;
        matrix.fromArray(mesh.instanceMatrix.array, instance * MATRIX_FLOATS).premultiply(mesh.matrixWorld);
        if (instance >= record.instances.length) {
            record.matrices.set(matrix.elements, instance * MATRIX_FLOATS);
            this.createInstance(record, matrix.elements, visible);
        } else if (copyIfChanged(matrix.elements, record.matrices, instance * MATRIX_FLOATS)) {
            this.transforms.add(record.instances[instance], matrix.elements);
            this.report.transforms++;
        }
    }

    // -----------------------------------------------------------------------
    // Geometry
    // -----------------------------------------------------------------------

    /** The attributes a geometry's platform mesh is made from, in GEOMETRY_ATTRIBUTE order. */
    private geometryAttributes(geometry: BufferGeometry, uv1: boolean): (Attribute | null)[] {
        const attributes = geometry.attributes;
        return [attributes.position ?? null, geometry.index, attributes.normal ?? null, attributes.uv ?? null, uv1 ? (attributes.uv1 ?? null) : null];
    }

    /**
     * The platform mesh for a three mesh's geometry: shared by every mesh
     * with that geometry, except a deforming one (`deform` given), which gets
     * its own, created `deforming` with its pose box, since each mesh has its
     * own pose.
     */
    private acquireGeometry(mesh: Mesh, deform: DeformKind | null): GeometryRecord | undefined {
        const geometry = mesh.geometry;
        const existing = deform ? undefined : this.geometries.get(geometry);
        if (existing) {
            existing.users++;
            return existing;
        }
        const slotCount = materialSlotCount(geometry);
        let uv1 = false;
        for (let slot = 0; slot < slotCount; slot++) {
            const material = this.materialForSlot(mesh, slot);
            if (material && usesUvChannel1(material)) uv1 = true;
        }
        const plain = meshDescriptor(geometry, { includeUv1: uv1, label: geometry.name || mesh.name || undefined });
        if (!plain) {
            this.warnOnce(`a mesh without positions ("${mesh.name || geometry.uuid}") is left out of GI.`);
            return undefined;
        }
        const shape = deform ? deformingShape(mesh, deform, this.gpuPoses) : null;
        const descriptor = shape ? { ...plain, deforming: true, deformBounds: shape.bounds } : plain;
        const attributes = this.geometryAttributes(geometry, uv1);
        const record: GeometryRecord = {
            geometry,
            handle: this.api.createMesh(descriptor),
            users: 1,
            uv1,
            slotCount,
            attributes,
            versions: attributes.map(attributeVersion),
            listIndex: this.geometryList.length,
            deform: shape,
        };
        if (!shape) this.geometries.set(geometry, record);
        this.geometryList.push(record);
        return record;
    }

    /**
     * A replaced or edited attribute means `updateMesh` (which rebuilds its
     * BVH). Doesn't allocate unless something changed. For a deforming mesh,
     * an edit of the positions alone is a new pose, which its pose source
     * sends (sendPoses); only a replaced position attribute is a new shape.
     */
    private checkGeometry(record: GeometryRecord): void {
        const geometry = record.geometry;
        const attributes = geometry.attributes;
        const positionChanged = record.deform
            ? (attributes.position ?? null) !== record.attributes[GEOMETRY_ATTRIBUTE.position]
            : attributeChanged(attributes.position, record, GEOMETRY_ATTRIBUTE.position);
        const changed =
            positionChanged ||
            attributeChanged(geometry.index, record, GEOMETRY_ATTRIBUTE.index) ||
            attributeChanged(attributes.normal, record, GEOMETRY_ATTRIBUTE.normal) ||
            attributeChanged(attributes.uv, record, GEOMETRY_ATTRIBUTE.uv) ||
            (record.uv1 && attributeChanged(attributes.uv1, record, GEOMETRY_ATTRIBUTE.uv1));
        if (!changed) return;
        const plain = meshDescriptor(geometry, { includeUv1: record.uv1, label: geometry.name || undefined });
        if (!plain) {
            this.warnOnce(`geometry "${geometry.name || geometry.uuid}" lost its positions; GI keeps its previous shape.`);
            return;
        }
        const descriptor = record.deform ? { ...plain, deforming: true, deformBounds: record.deform.bounds } : plain;
        this.api.updateMesh(record.handle, descriptor);
        // The platform now holds the new rest pose; the next pose is worked out from it.
        record.deform?.pose.reset();
        const current = this.geometryAttributes(geometry, record.uv1);
        for (let index = 0; index < current.length; index++) {
            record.attributes[index] = current[index];
            record.versions[index] = attributeVersion(current[index]);
        }
        this.report.meshes++;
    }

    private releaseGeometry(record: GeometryRecord): void {
        record.users--;
        if (record.users > 0) return;
        this.api.destroyMesh(record.handle);
        record.deform?.pose.dispose?.();
        // A deforming record isn't in the map (it isn't shared); a shared one for the same geometry may be.
        if (this.geometries.get(record.geometry) === record) this.geometries.delete(record.geometry);
        removeFromList(this.geometryList, record);
    }

    /** This frame's share of the edit checks: `editChecksPerFrame` geometries, materials and objects' GI flags, round robin. */
    private checkEdits(): void {
        const geometries = this.geometryList;
        for (let check = 0; check < Math.min(this.editChecksPerFrame, geometries.length); check++) {
            this.geometryCursor = (this.geometryCursor + 1) % geometries.length;
            this.checkGeometry(geometries[this.geometryCursor]);
        }
        const materials = this.materialList;
        for (let check = 0; check < Math.min(this.editChecksPerFrame, materials.length); check++) {
            this.materialCursor = (this.materialCursor + 1) % materials.length;
            this.checkMaterial(materials[this.materialCursor]);
        }
        const objects = this.objectList;
        for (let check = 0; check < Math.min(this.editChecksPerFrame, objects.length); check++) {
            this.objectCursor = (this.objectCursor + 1) % objects.length;
            const record = objects[this.objectCursor];
            // A changed flag, or a mesh that starts or stops deforming: start the object over, through the marked path, next update.
            if (record.contributes !== contributesFlag(record.object, record.skinned) || deformingChanged(record)) this.marked.add(record.object);
        }
    }

    // -----------------------------------------------------------------------
    // Materials and textures
    // -----------------------------------------------------------------------

    private acquireMaterial(material: Material): MaterialRecord {
        const existing = this.materials.get(material);
        if (existing) {
            existing.users++;
            return existing;
        }
        const snapshot = new MaterialSnapshot();
        snapshot.read(material);
        const textureHandles: TextureHandle[] = [];
        const descriptor = materialDescriptor(material, (texture) => this.textureHandleFor(texture, textureHandles), { warn: this.warnOnce });
        this.useTextureHandles(textureHandles);
        const record: MaterialRecord = {
            material,
            handle: this.api.createMaterial(descriptor),
            sent: descriptor,
            users: 1,
            snapshot,
            scratch: new MaterialSnapshot(),
            textureHandles,
            listIndex: this.materialList.length,
        };
        if (snapshot.envMap) this.materialsWithEnvMap++;
        this.materials.set(material, record);
        this.materialList.push(record);
        return record;
    }

    /** Compares the material's GI values with the last ones sent. */
    private checkMaterial(record: MaterialRecord): void {
        record.scratch.read(record.material);
        if (record.scratch.equals(record.snapshot)) return;
        this.materialsWithEnvMap += (record.scratch.envMap ? 1 : 0) - (record.snapshot.envMap ? 1 : 0);
        [record.snapshot, record.scratch] = [record.scratch, record.snapshot];
        const textureHandles: TextureHandle[] = [];
        const descriptor = materialDescriptor(record.material, (texture) => this.textureHandleFor(texture, textureHandles), { warn: this.warnOnce });
        this.useTextureHandles(textureHandles);
        // Every field, so a removed texture or cutoff is cleared too.
        this.api.updateMaterial(record.handle, descriptor);
        record.sent = descriptor;
        this.releaseTextureHandles(record.textureHandles);
        record.textureHandles = textureHandles;
        this.report.materials++;
    }

    private releaseMaterial(record: MaterialRecord): void {
        record.users--;
        if (record.users > 0) return;
        this.api.destroyMaterial(record.handle);
        if (record.snapshot.envMap) this.materialsWithEnvMap--;
        this.releaseTextureHandles(record.textureHandles);
        this.materials.delete(record.material);
        removeFromList(this.materialList, record);
    }

    /**
     * The platform texture for a three texture, made when first needed and
     * again when the texture's version changes (new pixels: three bumps it on
     * `needsUpdate`, and the material snapshot holds it, so the edit checks
     * find it). Undefined while it has no readable pixels: still loading, or
     * a kind GI doesn't read (a render target, say), with a warning.
     */
    private textureHandleFor(texture: Texture, collect: TextureHandle[]): TextureHandle | undefined {
        const existing = this.textures.get(texture);
        if (existing && existing.version === texture.version) {
            collect.push(existing.handle);
            return existing.handle;
        }
        const descriptor = textureDescriptor(texture);
        if (!descriptor) {
            const reason = unreadableTextureReason(texture);
            if (reason) this.warnOnce(reason);
            return undefined;
        }
        const handle = this.api.createTexture(descriptor);
        this.textures.set(texture, { texture, handle, version: texture.version });
        // A handle no material uses yet would never be destroyed: count it as used until the caller takes it.
        this.textureHandleUsers.set(handle, 0);
        if (existing && (this.textureHandleUsers.get(existing.handle) ?? 0) === 0) this.destroyTexture(existing.handle);
        collect.push(handle);
        return handle;
    }

    private useTextureHandles(handles: readonly TextureHandle[]): void {
        for (const handle of handles) this.textureHandleUsers.set(handle, (this.textureHandleUsers.get(handle) ?? 0) + 1);
    }

    private releaseTextureHandles(handles: readonly TextureHandle[]): void {
        for (const handle of handles) {
            const users = (this.textureHandleUsers.get(handle) ?? 1) - 1;
            this.textureHandleUsers.set(handle, users);
            if (users <= 0) this.destroyTexture(handle);
        }
    }

    private destroyTexture(handle: TextureHandle): void {
        this.api.destroyTexture(handle);
        this.textureHandleUsers.delete(handle);
        for (const [texture, record] of this.textures) {
            if (record.handle === handle) this.textures.delete(texture);
        }
    }

    // -----------------------------------------------------------------------
    // Lights
    // -----------------------------------------------------------------------

    /**
     * A light met by the walk or marked: warnings for kinds GI doesn't mirror,
     * a record for those it does. A marked light joins the hot lights, so
     * checkLights looks at it in this same update.
     */
    private visitLight(light: Light, visible: boolean, source: VisitSource): void {
        const flags = light as ObjectFlags;
        if (flags.isLightProbeGrid) {
            if (!this.probeGrids.includes(light)) this.probeGrids.push(light);
            return;
        }
        if (flags.isGILight || flags.isGIVolume) return;
        if (flags.isAmbientLight || flags.isHemisphereLight || flags.isLightProbe) {
            if (visible && light.intensity > 0) this.warnOnce(`${light.type} adds diffuse light on top of GI in three (it isn't part of GI's light): light is counted twice where GI covers.`);
            return;
        }
        if (flags.isRectAreaLight) {
            if (visible && light.intensity > 0) this.warnOnce('RectAreaLight is not mirrored into GI; give its surface an emissive material to have it light GI.');
            return;
        }
        if ((flags.isPointLight || flags.isSpotLight) && (light as PointLight).decay !== PHYSICAL_DECAY) {
            this.warnOnce(`GI lights fall off with distance squared (three's decay = 2); "${light.name || light.type}" has decay ${(light as PointLight).decay}, so GI uses 2 for it.`);
        }
        const existing = this.lights.get(light);
        if (existing) {
            existing.seenPass = this.pass;
            if (source === 'marked') this.makeLightHot(existing);
            return;
        }
        // As for meshes (visitMesh): the walk's stack can hold a light removed since it was pushed.
        if (source === 'walk' && !this.placement(light, this.walkScene!)) return;
        const descriptor = lightDescriptor(light, visible ? 1 : 0);
        if (!descriptor) return;
        const numbers = new Float64Array(LIGHT_NUMBER_COUNT);
        readLightNumbers(light, visible ? 1 : 0, numbers);
        const record: LightRecord = {
            light,
            handle: this.api.createLight(descriptor),
            descriptor,
            numbers,
            scratch: new Float64Array(LIGHT_NUMBER_COUNT),
            seenPass: this.pass,
            listIndex: this.lightList.length,
            directional: (light as DirectionalLight).isDirectionalLight === true,
            visible,
            // Just read: nothing to check again this update.
            checkedFrame: this.frame,
            hotUntil: 0,
            hot: false,
            destroyed: false,
        };
        this.lights.set(light, record);
        this.lightList.push(record);
        if (record.directional) this.directionalLights.push(record);
        this.report.created++;
    }

    /**
     * This frame's light checks (the file header): the first
     * ALWAYS_CHECKED_LIGHTS and every directional light, the hot lights, and
     * `lightsPerFrame` of the rest, round robin. Then picks the sun from the
     * directional lights, which are all current.
     */
    private checkLights(scene: Scene): void {
        // Backwards, since destroying swaps the last record into the hole.
        for (let index = Math.min(this.lightList.length, ALWAYS_CHECKED_LIGHTS) - 1; index >= 0; index--) this.checkLight(this.lightList[index], scene);
        for (let index = this.directionalLights.length - 1; index >= 0; index--) this.checkLight(this.directionalLights[index], scene);
        const hot = this.hotLights;
        for (let index = hot.length - 1; index >= 0; index--) {
            const record = hot[index];
            if (!record.destroyed) this.checkLight(record, scene);
            if (record.destroyed || this.frame > record.hotUntil) {
                record.hot = false;
                const last = hot.pop()!;
                if (index < hot.length) hot[index] = last;
            }
        }
        const rest = this.lightList.length - ALWAYS_CHECKED_LIGHTS;
        for (let count = 0; count < Math.min(rest, this.lightsPerFrame); count++) {
            // The list can shrink while this runs (a light found removed): wrap on its current length.
            const remaining = this.lightList.length - ALWAYS_CHECKED_LIGHTS;
            if (remaining <= 0) break;
            this.lightCursor = this.lightCursor % remaining;
            this.checkLight(this.lightList[ALWAYS_CHECKED_LIGHTS + this.lightCursor], scene);
            this.lightCursor++;
        }

        this.sun = null;
        let sunIlluminance = 0;
        for (const record of this.directionalLights) {
            if (!record.visible) continue;
            const illuminance = record.light.intensity * luminance(record.light.color);
            if (!this.sun || illuminance > sunIlluminance) {
                this.sun = record.light as DirectionalLight;
                sunIlluminance = illuminance;
            }
        }
        this.otherProbeLights = 0;
        const grids = this.probeGrids;
        for (let index = grids.length - 1; index >= 0; index--) {
            if (!this.placement(grids[index], scene)) grids.splice(index, 1);
            else if (this.placementVisible && grids[index].intensity > 0 && (!this.cameraLayers || grids[index].layers.test(this.cameraLayers))) this.otherProbeLights++;
        }
    }

    /** One light, at most once an update: gone, removed; otherwise only changed fields sent, and a changed light turns hot. */
    private checkLight(record: LightRecord, scene: Scene): void {
        if (record.checkedFrame === this.frame) return;
        record.checkedFrame = this.frame;
        this.report.lightsChecked++;
        const light = record.light;
        if (!this.placement(light, scene)) {
            this.destroyLight(record);
            return;
        }
        const visible = this.placementVisible;
        record.visible = visible;
        // Numbers first: a descriptor (and its arrays) is built only when something changed.
        readLightNumbers(light, visible ? 1 : 0, record.scratch);
        if (sameNumbers(record.scratch, record.numbers)) return;
        [record.numbers, record.scratch] = [record.scratch, record.numbers];
        this.makeLightHot(record);
        const descriptor = lightDescriptor(light, visible ? 1 : 0)!;
        // A three light never changes type, so the descriptor types always match.
        const changes = lightChanges(record.descriptor, descriptor);
        if (changes) {
            this.api.updateLight(record.handle, changes);
            record.descriptor = descriptor;
            this.report.lights++;
        }
    }

    private makeLightHot(record: LightRecord): void {
        record.hotUntil = this.frame + HOT_FRAMES;
        if (record.hot) return;
        record.hot = true;
        this.hotLights.push(record);
    }

    private destroyLight(record: LightRecord): void {
        this.api.destroyLight(record.handle);
        this.lights.delete(record.light);
        removeFromList(this.lightList, record);
        if (record.directional) this.directionalLights.splice(this.directionalLights.indexOf(record), 1);
        // The hot list drops it on its next pass.
        record.destroyed = true;
        this.report.destroyed++;
    }
}

/** Removes a record from an edit-check list by swapping the last one into its place. */
function removeFromList<Entry extends { listIndex: number }>(list: Entry[], entry: Entry): void {
    const last = list.pop()!;
    if (last !== entry) {
        list[entry.listIndex] = last;
        last.listIndex = entry.listIndex;
    }
}

/**
 * Transforms collected during one `update`, sent with one `setTransforms`
 * call: handles and matrices in typed arrays that grow as needed and are
 * reused frame after frame, so a frame with movers allocates nothing once
 * they are big enough.
 */
class TransformBatch {
    private handles = new Uint32Array(16);
    private matrices = new Float32Array(16 * MATRIX_FLOATS);
    private count = 0;

    add(instance: InstanceHandle, matrix: ArrayLike<number>): void {
        if (this.count === this.handles.length) this.grow();
        this.handles[this.count] = instance;
        const base = this.count * MATRIX_FLOATS;
        for (let index = 0; index < MATRIX_FLOATS; index++) this.matrices[base + index] = matrix[index];
        this.count++;
    }

    /** One `setTransforms` call with everything collected, if anything was. */
    send(api: SceneApi): void {
        if (this.count === 0) return;
        const count = this.count;
        this.count = 0;
        api.setTransforms(this.handles.subarray(0, count), this.matrices.subarray(0, count * MATRIX_FLOATS));
    }

    clear(): void {
        this.count = 0;
    }

    private grow(): void {
        const handles = new Uint32Array(this.handles.length * 2);
        handles.set(this.handles);
        this.handles = handles;
        const matrices = new Float32Array(this.matrices.length * 2);
        matrices.set(this.matrices);
        this.matrices = matrices;
    }
}

/**
 * World matrices in one growing Float64Array, 16 numbers per slot. Plain
 * meshes keep their last-sent matrix here rather than in an array each, so
 * comparisons read memory in order (meshes take slots in the order the walk
 * first meets them) instead of a scattered array per mesh.
 */
class MatrixPool {
    values = new Float64Array(64 * MATRIX_FLOATS);
    private readonly free: number[] = [];
    private used = 0;

    take(): number {
        const reused = this.free.pop();
        if (reused !== undefined) return reused;
        if ((this.used + 1) * MATRIX_FLOATS > this.values.length) {
            const grown = new Float64Array(this.values.length * 2);
            grown.set(this.values);
            this.values = grown;
        }
        return this.used++;
    }

    release(slot: number): void {
        this.free.push(slot);
    }
}

/** Copies `source` (16 numbers) into `target` at `offset` if any differs; returns whether it did. */
function copyIfChanged(source: ArrayLike<number>, target: Float64Array, offset: number): boolean {
    let changed = false;
    for (let index = 0; index < MATRIX_FLOATS; index++) {
        if (target[offset + index] !== source[index]) {
            target[offset + index] = source[index];
            changed = true;
        }
    }
    return changed;
}

/**
 * Whether the mesh would now be created deforming or not, unlike when it
 * was: `userData.giDeforming` set or cleared, or morph targets or a skeleton
 * added. Read where the GI flag is (hot and marked visits, the edit
 * rotation), and handled the same way: the object starts over.
 */
function deformingChanged(record: ObjectRecord): boolean {
    const deformsNow = record.contributes && deformKind(record.object) !== null;
    return deformsNow !== (record.geometry.deform !== null);
}

/** Whether `source` (16 numbers) differs from `target` at `offset`; reads only, so a hook can compare without changing what was sent. */
function differs(source: ArrayLike<number>, target: Float64Array, offset: number): boolean {
    for (let index = 0; index < MATRIX_FLOATS; index++) if (target[offset + index] !== source[index]) return true;
    return false;
}

/** How a deforming mesh deforms, its pose box and its pose source (deforming.ts). */
function deformingShape(mesh: Mesh, kind: DeformKind, gpu: GpuPoseContext | null): DeformingShape {
    const bounds = deformBoundsFor(mesh, kind);
    return { kind, bounds, pose: poseSourceFor(mesh, kind, bounds, gpu) };
}

function sameNumbers(a: Float64Array, b: Float64Array): boolean {
    for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return false;
    return true;
}

function sameTuple(a: readonly number[] | undefined, b: readonly number[] | undefined): boolean {
    if (a === b) return true;
    if (!a || !b || a.length !== b.length) return false;
    for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return false;
    return true;
}

/** The fields of `next` that differ from `previous` (same type), or null if none. */
export function lightChanges(previous: LightDescriptor, next: LightDescriptor): LightChanges | null {
    const changes: Record<string, unknown> = {};
    let any = false;
    const before = previous as unknown as Record<string, unknown>;
    for (const [key, value] of Object.entries(next)) {
        if (key === 'type') continue;
        const old = before[key];
        const same = Array.isArray(value) ? sameTuple(value as number[], old as number[] | undefined) : value === old;
        if (!same) {
            changes[key] = value;
            any = true;
        }
    }
    return any ? (changes as LightChanges) : null;
}
