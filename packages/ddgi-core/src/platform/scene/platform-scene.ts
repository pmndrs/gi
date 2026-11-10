/**
 * The platform's scene: the host-facing `SceneApi` and the technique-facing
 * `SceneCopy`, over two GPU storage buffers (see layout.ts).
 *
 * Calls from the host change CPU state: meshes are checked and laid out,
 * records are written into a CPU mirror of the table buffer, and what moved
 * is logged. The one exception is a mesh's raw arrays: once the platform has
 * called {@link PlatformScene.prepareGpuPacking}, `createMesh` and
 * `updateMesh` copy them straight into a staging buffer, and the next
 * {@link PlatformScene.encode} packs them on the GPU (T2.13,
 * gpu-mesh-packing.ts). Without that call, or for a mesh too large to stage,
 * the CPU packs the mesh (mesh-packing.ts) and `encode` uploads its words.
 * Everything else reaches the GPU in `encode`, which the platform calls at
 * the start of its own.
 *
 * **How uploads reach the GPU, and why `queue.writeBuffer`.** Uploads use
 * `queue.writeBuffer`. It needs no extra submit and no mapping round trips,
 * the browser stages the data itself, and a write issued during `encode`
 * lands on the queue after every earlier submit and before the host submits
 * this frame's commands, so this frame's passes see it. The one GPU-side copy
 * is when the vertex buffer grows: the meshes already on the GPU are copied
 * into the bigger buffer by commands in the host's encoder (no CPU copy of
 * the geometry is kept). Those copies run when the host submits, *after* the
 * writes, so only ranges that hold resident meshes are copied, never a range
 * a new mesh is being written into this frame.
 *
 * **Nothing recompiles.** The WGSL is fixed for the module's lifetime; counts
 * and section offsets live in the table header. When a buffer is reallocated,
 * `module.version` increments so kernels rebuild their bind group, never
 * their pipeline. An old buffer is destroyed at the next `encode`, once the
 * frame whose commands copied out of it has been submitted.
 */

import { DEFAULT_TURBIDITY } from '../sky/calibrated-sky.ts';
import { GILimitError, GINotReadyError } from '../contract/errors.ts';
import type { InstanceHandle, LightHandle, MaterialHandle, MeshHandle, TextureHandle } from '../contract/handles.ts';
import type { Aabb, ColorRGB, Mat4, Vec3 } from '../contract/math.ts';
import type {
    AlphaMode,
    DeformedPositions,
    EnvironmentDescriptor,
    InstanceDescriptor,
    LightChanges,
    LightDescriptor,
    MaterialDescriptor,
    MeshDescriptor,
    SceneApi,
    TextureDescriptor,
    TextureSource,
    UvMapping,
} from '../contract/scene.ts';
import { PlatformTextureStore, type TextureLayerRequest, type TextureLayerResults, type TextureLayerStore } from '../textures/texture-store.ts';
import type { SceneChanges, SceneCopy } from '../contract/scene-copy.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { BindingSlot, WgslModule } from '../contract/wgsl.ts';
import { invertAffineInto } from '../trace/top-level.ts';
import { ChangeLog, copyLight, toAabb, type LightLookup } from './change-log.ts';
import { boxDiagonal, boxInside, checkedDeformBounds, copyCheckedPositions, DeformState, growBox, largestScale, scanPose } from './deformed-positions.ts';
import { checkedGpuPose, GpuPoseCopier, type PoseCopy } from './gpu-pose-copy.ts';
import { BOX_NUMBERS, InstanceTransforms, MATRIX_NUMBERS } from './instance-transforms.ts';
import {
    ALPHA_MODE,
    DEFAULT_ALPHA_CUTOFF,
    ENVIRONMENT_TYPE,
    HEADER_VEC4S,
    HEADER_WORD,
    INSTANCE_FLAG,
    INSTANCE_WORD,
    LIGHT_FLAG,
    LIGHT_GROUP_COUNT,
    LIGHT_TYPE,
    LIGHT_WORD,
    MATERIAL_FLAG,
    MATERIAL_RECORD_VEC4S,
    MATERIAL_WORD,
    MESH_FLAG,
    MESH_WORD,
    NONE,
    TEXTURE_CHANNEL,
    UV_PACK_SHIFT,
    UV_WRAP,
} from './layout.ts';
import { GpuMeshPacker, type MeshToPack, type StagedMesh } from './gpu-mesh-packing.ts';
import { packMeshWords, planMesh, type MeshLayout } from './mesh-packing.ts';
import { RangeAllocator } from './range-allocator.ts';
import { SlotAllocator } from './slot-allocator.ts';
import { TableMirror } from './table-mirror.ts';
import { SCENE_MODULE_ENTRIES, sceneModuleWgsl } from './wgsl.ts';

/** Starting sizes. Every table doubles when it fills; none of this is a hard limit. */
export interface SceneCapacities {
    readonly instances: number;
    readonly meshes: number;
    readonly materials: number;
    /**
     * 64 records (4 KB) to start (m2-lights.md §1). Like every table it
     * doubles when it fills, by reallocating the table buffer: records are
     * read by index from the same binding, so no kernel is rebuilt (the
     * `platform-recompiles` page churns 4,096 lights to check it).
     */
    readonly lights: number;
    /** Words of instance material lists (one per material slot per instance). */
    readonly materialListWords: number;
    /** Words of mesh data in the vertex buffer. */
    readonly vertexWords: number;
}

export const DEFAULT_SCENE_CAPACITIES: SceneCapacities = {
    instances: 64,
    meshes: 64,
    materials: 64,
    lights: 64,
    materialListWords: 256,
    vertexWords: 256 * 1024,
};

export interface PlatformSceneOptions {
    /** Prefix for GPU buffer labels. */
    readonly label?: string;
    readonly initialCapacities?: Partial<SceneCapacities>;
    /**
     * Where textures become layers (textures/texture-store.ts). The platform
     * passes its own, which hit shading also binds; without one the scene
     * makes a store of its own (unit tests pass a stand-in).
     */
    readonly textureStore?: TextureLayerStore;
    /** Where one-time warnings go (a texture that can't be used; deformed positions outside `deformBounds`). Defaults to `console.warn`. */
    readonly warn?: (message: string) => void;
    /**
     * Whether the tracer can follow deformed positions (it refits mesh trees
     * on the GPU, which needs subgroups). False: `setDeformedPositions` is
     * checked and then ignored, with one warning, so the scene copy keeps the
     * rest pose the tree was built for (rays and hit points agree). Default
     * true.
     */
    readonly posesTraced?: boolean;
}

/**
 * A mesh's positions and indices on the CPU, for tree builders: copies the
 * scene made when the mesh was created (the host may reuse its arrays after
 * the call). The indices keep the host's type, whatever width the vertex
 * buffer stores them in.
 */
export interface MeshGeometryCopy {
    readonly positions: Float32Array<ArrayBuffer>;
    readonly indices: Uint16Array<ArrayBuffer> | Uint32Array<ArrayBuffer>;
}

/**
 * A mesh whose geometry went to the GPU in this `encode`, for builders that
 * need it (the tracer's BVH).
 *
 * Since T2.13 it carries no packed words, because the GPU usually packs the
 * mesh. Read it on the CPU from `geometry`, or on the GPU from the vertex
 * buffer at `vertexOffset` + `layout.offsets` (positions 3 floats per vertex;
 * indices 16-bit two to a word when `layout.indices16`, else 32-bit) in
 * commands recorded after this `encode`'s.
 */
export interface UploadedMesh {
    readonly slot: number;
    readonly handle: MeshHandle;
    readonly layout: MeshLayout;
    /** Word offset of the mesh's first word in the vertex buffer. */
    readonly vertexOffset: number;
    /** Handed over once: the receiver owns these arrays and may transfer them (to a worker). */
    readonly geometry: MeshGeometryCopy;
}

/**
 * A deforming mesh given new positions in this `encode`: the scene copy's
 * vertex buffer holds them (written before, or recorded ahead of, anything
 * the tracer records afterwards), so the tracer refits the mesh's tree from
 * it. Positions at `vertexOffset + layout.offsets.positions`, 3 floats a
 * vertex; indices at `layout.offsets.indices`.
 */
export interface PosedMesh {
    readonly slot: number;
    readonly layout: MeshLayout;
    readonly vertexOffset: number;
}

/** One instance as the tracer's top level needs it. */
export interface InstanceSnapshot {
    readonly slot: number;
    readonly handle: InstanceHandle;
    readonly meshSlot: number;
    readonly objectToWorld: Float64Array;
    readonly contributesToGi: boolean;
    readonly visible: boolean;
}

/** Every instance that contributes to GI, in flat arrays (see {@link PlatformScene.packGiInstances}). */
export interface PackedGiInstances {
    readonly slots: Uint32Array;
    readonly meshSlots: Uint32Array;
    readonly visible: Uint8Array;
    /** Object-to-world, 16 numbers per instance (column-major). */
    readonly objectToWorld: Float64Array;
}

/** What one {@link PlatformScene.encode} did. */
export interface SceneUpload {
    readonly changes: SceneChanges;
    /** Meshes created or reshaped since the last encode. A slot may also be in `meshesRemoved` (reused). */
    readonly meshesUploaded: readonly UploadedMesh[];
    /**
     * Deforming meshes given a new pose (`setDeformedPositions`) applied in
     * this encode. Their `layout.bounds` is the box every pose stays in; it
     * only grows (a CPU pose that left it).
     */
    readonly meshesPosed: readonly PosedMesh[];
    /** Mesh slots freed since the last encode. Handle these before `meshesUploaded`. */
    readonly meshesRemoved: readonly number[];
    /** The traceable instances, one of their transforms, or one of their meshes changed: a top level is stale. */
    readonly instancesChanged: boolean;
    /**
     * An instance that contributes to GI was created or destroyed (shown or
     * hidden doesn't count). Moves and mesh updates leave it false, so the
     * tracer's GPU top level (B1a) knows when it needs a new shape without
     * looking at every instance.
     */
    readonly instanceSetChanged: boolean;
    /** A scene buffer was reallocated; `module.version` has changed too. */
    readonly reallocated: boolean;
    /** Bytes written with `queue.writeBuffer` since the last encode (a mesh staged for the GPU counts when it was staged). */
    readonly bytesWritten: number;
}

/** GPU memory of the scene copy, in bytes. */
export interface SceneMemory {
    readonly vertexBuffer: number;
    /** Of `vertexBuffer`, the bytes holding live meshes. */
    readonly vertexBytesUsed: number;
    readonly tableBuffer: number;
    /** Staging buffers of meshes waiting for the GPU to pack them (T2.13); 0 once a load is packed. */
    readonly stagingBuffers: number;
    readonly total: number;
}

/** A mesh's data on its way to the vertex buffer: packed on the CPU, or staged for the GPU. Dropped once uploaded. */
interface PendingMeshData {
    /** The CPU packer's words (the fallback), or null when `staged`. */
    readonly words: Uint32Array<ArrayBuffer> | null;
    /** The raw arrays in a staging buffer, or null when packed on the CPU. */
    readonly staged: StagedMesh | null;
    readonly geometry: MeshGeometryCopy;
}

interface MeshEntry {
    readonly handle: MeshHandle;
    readonly slot: number;
    label: string | undefined;
    layout: MeshLayout;
    /** What the next encode uploads; null once uploaded. */
    pending: PendingMeshData | null;
    /** Word offset in the vertex buffer, or -1 until uploaded. */
    vertexOffset: number;
    readonly instances: Set<InstanceEntry>;
    /** For a mesh created `deforming`: its poses (deformed-positions.ts); null for the others. */
    deform: DeformState | null;
}

/**
 * Where a texture is on its way to the store: waiting for the next `encode`,
 * made (its layer is its slot), or refused (the store warned why; its
 * materials keep their flat colour).
 */
type TextureState = 'pending' | 'built' | 'failed';

interface TextureEntry {
    readonly handle: TextureHandle;
    /** Also its layer in the texture store. */
    readonly slot: number;
    readonly srgb: boolean;
    readonly label: string | undefined;
    /** The host's source, dropped once the layer is made (the host may then free it). */
    source: TextureSource | null;
    state: TextureState;
    /** Material texture references to it (a material using it twice counts twice). */
    users: number;
}

interface MaterialEntry {
    readonly handle: MaterialHandle;
    readonly slot: number;
    descriptor: MaterialDescriptor;
    albedoTexture: TextureEntry | undefined;
    emissiveTexture: TextureEntry | undefined;
    metalnessTexture: TextureEntry | undefined;
    readonly users: Set<InstanceEntry>;
}

interface InstanceEntry {
    readonly handle: InstanceHandle;
    readonly slot: number;
    label: string | undefined;
    mesh: MeshEntry;
    materials: MaterialEntry[];
    contributesToGi: boolean;
    visible: boolean;
    /** Word offset of the material list inside the list section. */
    listOffset: number;
    // Its transform, world box and frame are in the scene's InstanceTransforms, at its slot.
}

interface LightEntry {
    readonly handle: LightHandle;
    /** Position in the packed light table. */
    index: number;
    descriptor: LightDescriptor;
}

const IDENTITY_UV: readonly number[] = [1, 0, 0, 1, 0, 0];

/** The rows `invertAffineInto` would give the identity: what a singular transform's normal matrix is written as. */
const IDENTITY_ROWS: readonly number[] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

function isFiniteVector(value: ArrayLike<number> | undefined, length: number): boolean {
    if (value === undefined || value.length !== length) return false;
    for (let index = 0; index < length; index++) if (!Number.isFinite(value[index])) return false;
    return true;
}

function normalized(vector: Vec3, what: string): [number, number, number] {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    if (!(length > 0) || !Number.isFinite(length)) throw new RangeError(`${what} must be a finite, non-zero vector`);
    return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function checkColor(color: ColorRGB | undefined, what: string): void {
    if (color !== undefined && !isFiniteVector(color, 3)) throw new RangeError(`${what} must be three finite numbers`);
}

function checkNonNegative(value: number | undefined, what: string): void {
    if (value !== undefined && !(Number.isFinite(value) && value >= 0)) throw new RangeError(`${what} must be a finite number >= 0, got ${value}`);
}

export class PlatformScene implements SceneApi, SceneCopy {
    readonly module: WgslModule;

    private readonly device: GPUDevice;
    private readonly label: string;
    private disposed = false;

    private vertexBuffer: GPUBuffer;
    private readonly vertexAllocator: RangeAllocator;
    private tableBuffer: GPUBuffer;
    private uploadedLayoutVersion: number;
    private readonly mirror: TableMirror;
    /** Buffers replaced in the last `encode`, destroyed in the next one (after the host submitted). */
    private retired: GPUBuffer[] = [];
    private bufferVersion = 0;

    private nextHandle = 1;
    private readonly meshes = new Map<number, MeshEntry>();
    private readonly textures = new Map<number, TextureEntry>();
    private readonly materials = new Map<number, MaterialEntry>();
    private readonly instances = new Map<number, InstanceEntry>();
    /** Every instance's matrix, world box and frame, by slot (instance-transforms.ts says why flat). */
    private readonly transforms = new InstanceTransforms();
    private readonly lightsByHandle = new Map<number, LightEntry>();
    /** Packed: directional lights first, then point and spot lights. */
    private readonly lights: LightEntry[] = [];
    private directionalLightCount = 0;
    private environmentDescriptor: EnvironmentDescriptor = { type: 'none' };
    /** Bumped by every `setEnvironment`, also with the same object: the host's way to say a texture's contents changed. */
    private environmentChanges = 0;

    private readonly meshSlots = new SlotAllocator();
    private readonly textureSlots = new SlotAllocator();
    private readonly materialSlots = new SlotAllocator();
    private readonly instanceSlots = new SlotAllocator();

    /** Where textures become layers; `encode` hands it the textures created since the last one. */
    readonly textureStore: TextureLayerStore;
    private readonly pendingTextures = new Set<TextureEntry>();

    private readonly pendingMeshes = new Set<MeshEntry>();
    /** Deforming meshes with a new pose waiting for `encode`. */
    private readonly posesWaiting = new Set<MeshEntry>();
    /** A pose's box, worked out in `applyPoses` (scratch). */
    private readonly poseBounds = new Float32Array(BOX_NUMBERS);
    /** Copies GPU poses into the vertex buffer (gpu-pose-copy.ts); compiled by {@link prepareGpuPoses}. */
    private readonly poseCopier: GpuPoseCopier;
    /** See {@link PlatformSceneOptions.posesTraced}. */
    private readonly posesTraced: boolean;
    private warnedRestPose = false;
    private readonly warn: (message: string) => void;
    /** Packs meshes on the GPU once {@link prepareGpuPacking} was called (T2.13). */
    private readonly packer: GpuMeshPacker;
    private gpuPacking = false;
    /** Bytes staged for the GPU since the last encode, reported in its `bytesWritten`. */
    private stagedBytes = 0;
    private removedMeshSlots: number[] = [];
    private readonly changeLog = new ChangeLog();
    /** How the change log sees the live lights when it builds a frame's light changes. */
    private readonly lightLookup: LightLookup = {
        current: (light) => {
            const entry = this.lightsByHandle.get(light);
            return entry ? { index: entry.index, descriptor: entry.descriptor } : undefined;
        },
        all: () => this.lights.map((entry) => entry.descriptor),
    };
    private headerDirty = true;
    private topLevelDirty = false;
    private instanceSetDirty = false;
    private triangleTotal = 0;
    private cachedBounds: Aabb | null = null;
    /** The inverse transform while an instance record is written (invertAffineInto), so writing one allocates nothing. */
    private readonly inverseScratch = new Float64Array(12);
    /** checkedTransform's copy of a host's transform. */
    private readonly transformScratch = new Float64Array(16);
    /** An instance's world box before a move, while it is written (moveInstance). */
    private readonly boundsBefore = new Float32Array(6);
    /**
     * The last `setTransforms` call's instances, looked up before anything
     * changed, and the handles they were for. A host that moves the same
     * objects frame after frame (an animation, a crowd) sends the same list;
     * while no instance was created or destroyed since (`instanceSetVersion`),
     * the lookups are reused instead of made again, which was a sixth of the
     * call's time at 10,000 instances.
     */
    private batchEntries: InstanceEntry[] = [];
    private batchHandles = new Uint32Array(0);
    private batchVersion = -1;
    /** Bumped by every instance created or destroyed. */
    private instanceSetVersion = 0;

    constructor(device: GPUDevice, options: PlatformSceneOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'gi';
        this.warn = options.warn ?? ((message) => console.warn(message));
        const capacities = { ...DEFAULT_SCENE_CAPACITIES, ...options.initialCapacities };
        this.mirror = new TableMirror({
            instanceCapacity: capacities.instances,
            meshCapacity: capacities.meshes,
            materialCapacity: capacities.materials,
            lightCapacity: capacities.lights,
            materialListCapacity: capacities.materialListWords,
        });
        this.vertexAllocator = new RangeAllocator(capacities.vertexWords);
        this.vertexBuffer = this.createStorageBuffer('scene vertices', capacities.vertexWords * 4);
        this.tableBuffer = this.createStorageBuffer('scene tables', this.mirror.byteLength);
        this.uploadedLayoutVersion = -1;
        this.packer = new GpuMeshPacker(device, { label: this.label });
        this.textureStore = options.textureStore ?? new PlatformTextureStore(device, { label: this.label, warn: options.warn });
        this.poseCopier = new GpuPoseCopier(device, this.label);
        this.posesTraced = options.posesTraced ?? true;

        const currentVersion = () => this.bufferVersion;
        this.module = {
            name: 'gi-scene',
            dependencies: [],
            code: (slot: BindingSlot) => sceneModuleWgsl(slot),
            entries: SCENE_MODULE_ENTRIES,
            resources: () => [{ buffer: this.vertexBuffer }, { buffer: this.tableBuffer }],
            get version() {
                return currentVersion();
            },
            storageBufferCount: SCENE_MODULE_ENTRIES.length,
        };
    }

    // -----------------------------------------------------------------------
    // SceneCopy
    // -----------------------------------------------------------------------

    get counts(): SceneCopy['counts'] {
        return {
            meshes: this.meshes.size,
            instances: this.instances.size,
            triangles: this.triangleTotal,
            lights: this.lights.length,
            materials: this.materials.size,
        };
    }

    get firstDirectionalLight(): SceneCopy['firstDirectionalLight'] {
        // Directional lights are packed first, so light 0 is the first of them when there is one.
        const first = this.directionalLightCount > 0 ? this.lights[0] : undefined;
        if (!first || first.descriptor.type !== 'directional') return null;
        return { light: first.handle, descriptor: first.descriptor };
    }

    /** Point and spot lights (every light that isn't directional): hit shading picks its light path from this (the light grid, T2.3). */
    get localLightCount(): number {
        return this.lights.length - this.directionalLightCount;
    }

    get memoryBytes(): number {
        return this.memory.total;
    }

    get memory(): SceneMemory {
        const vertexBuffer = this.vertexBuffer.size;
        const tableBuffer = this.tableBuffer.size;
        const stagingBuffers = this.packer.stagingBytes;
        return { vertexBuffer, vertexBytesUsed: this.vertexAllocator.used * 4, tableBuffer, stagingBuffers, total: vertexBuffer + tableBuffer + stagingBuffers };
    }

    get environment(): EnvironmentDescriptor {
        return this.environmentDescriptor;
    }

    /**
     * How many times `setEnvironment` has been called. Hit shading re-projects
     * the sky when this changes, so calling `setEnvironment` again with the
     * same object (after changing its texture's contents) re-projects it (T1.4).
     */
    get environmentVersion(): number {
        return this.environmentChanges;
    }

    /** World-space bounds of every visible instance that contributes to GI. A zero box at the origin when there are none. */
    get bounds(): Aabb {
        if (this.cachedBounds) return this.cachedBounds;
        const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
        const worldBounds = this.transforms.bounds;
        for (const instance of this.instances.values()) {
            if (!instance.contributesToGi || !instance.visible) continue;
            const at = instance.slot * BOX_NUMBERS;
            for (let axis = 0; axis < 3; axis++) {
                box[axis] = Math.min(box[axis], worldBounds[at + axis]);
                box[axis + 3] = Math.max(box[axis + 3], worldBounds[at + 3 + axis]);
            }
        }
        this.cachedBounds = box[0] === Infinity ? { min: [0, 0, 0], max: [0, 0, 0] } : toAabb(box);
        return this.cachedBounds;
    }

    /** Slot (the GPU index, and `GiHit.instance`) of an instance. */
    instanceSlot(instance: InstanceHandle): number {
        return this.instance(instance).slot;
    }

    meshSlot(mesh: MeshHandle): number {
        return this.mesh(mesh).slot;
    }

    materialSlot(material: MaterialHandle): number {
        return this.material(material).slot;
    }

    /**
     * An instance's mesh slot, and the record of each material it uses in
     * slot order, as the GPU will read them (views of the CPU mirror: read
     * them at once). For the tracer's triangle-flag upkeep (T4.2,
     * trace/ray-queries.ts), which rewrites a mesh's flags only when the
     * words the walk's material rules read have changed. Null for a
     * destroyed instance.
     */
    instanceMaterialRecords(instance: InstanceHandle): { readonly meshSlot: number; readonly records: readonly Uint32Array[] } | null {
        const entry = this.instances.get(instance);
        if (!entry) return null;
        const recordWords = MATERIAL_RECORD_VEC4S * 4;
        const records = entry.materials.map((material) => {
            const base = this.mirror.recordWord('materials', material.slot);
            return this.mirror.words.subarray(base, base + recordWords);
        });
        return { meshSlot: entry.mesh.slot, records };
    }

    /** Index of a light in the packed light table. Changes when other lights are added or removed. */
    lightIndex(light: LightHandle): number {
        return this.light(light).index;
    }

    /**
     * Every instance that contributes to GI, hidden ones included, packed for
     * the tracer's top-level shape builds: one pass, typed arrays, no object
     * per instance (10,000 instances move often, B1a).
     */
    packGiInstances(): PackedGiInstances {
        let count = 0;
        for (const instance of this.instances.values()) if (instance.contributesToGi) count++;
        const packed = {
            slots: new Uint32Array(count),
            meshSlots: new Uint32Array(count),
            visible: new Uint8Array(count),
            objectToWorld: new Float64Array(count * 16),
        };
        let index = 0;
        for (const instance of this.instances.values()) {
            if (!instance.contributesToGi) continue;
            packed.slots[index] = instance.slot;
            packed.meshSlots[index] = instance.mesh.slot;
            packed.visible[index] = instance.visible ? 1 : 0;
            packed.objectToWorld.set(this.transforms.matrix(instance.slot), index * 16);
            index++;
        }
        return packed;
    }

    /** Every live instance, for the CPU top level (the tracer's fallback without subgroups). */
    instanceSnapshot(): InstanceSnapshot[] {
        return [...this.instances.values()].map((instance) => ({
            slot: instance.slot,
            handle: instance.handle,
            meshSlot: instance.mesh.slot,
            // A copy: the flat arrays are replaced when they grow.
            objectToWorld: this.transforms.matrix(instance.slot).slice(),
            contributesToGi: instance.contributesToGi,
            visible: instance.visible,
        }));
    }

    // -----------------------------------------------------------------------
    // Meshes
    // -----------------------------------------------------------------------

    /**
     * From now on meshes are packed on the GPU (T2.13): `createMesh` and
     * `updateMesh` stage the host's raw arrays, and `encode` packs them. Asks
     * for the packing kernel's compile and resolves when it is compiled;
     * until then `encode` refuses staged meshes (`GINotReadyError`). The
     * platform calls this first in `prepare`, so a host that copies its scene
     * in while preparing stages it at once, alongside the compiles. Meshes
     * created before the call stay with the CPU packer, as does a standalone
     * scene that never calls it (tools and tests).
     */
    prepareGpuPacking(): Promise<void> {
        this.checkLive();
        this.gpuPacking = true;
        return this.packer.prepare();
    }

    /**
     * Compiles the kernel that copies deformed positions given on the GPU
     * into the vertex buffer (gpu-pose-copy.ts). The platform calls it in
     * `prepare`; until it resolves, GPU poses wait in the scene.
     */
    /**
     * The vertex buffer now. It is replaced when it grows (`module.version`
     * changes then), so read it each frame. For the tracer: it refits and
     * rebuilds deforming meshes' trees from the positions in it.
     */
    get vertexStorage(): GPUBuffer {
        return this.vertexBuffer;
    }

    prepareGpuPoses(): Promise<void> {
        this.checkLive();
        return this.poseCopier.prepare();
    }

    createMesh(descriptor: MeshDescriptor): MeshHandle {
        this.checkLive();
        // Checked before the mesh is staged, so a bad box leaves nothing behind.
        const declared = this.declaredDeformBounds(descriptor, this.name('mesh', this.nextHandle, descriptor.label));
        const prepared = this.prepareMeshData(descriptor);
        const pending = prepared.pending;
        const handle = this.nextHandle++ as MeshHandle;
        const { layout, deform } = this.deformingLayout(prepared.layout, descriptor, declared);
        const slot = this.meshSlots.take();
        this.mirror.ensureRecord('meshes', slot);
        const entry: MeshEntry = {
            handle,
            slot,
            label: descriptor.label,
            layout,
            pending,
            vertexOffset: -1,
            instances: new Set(),
            deform,
        };
        this.meshes.set(handle, entry);
        this.pendingMeshes.add(entry);
        this.triangleTotal += layout.triangleCount;
        this.headerDirty = true;
        return handle;
    }

    updateMesh(mesh: MeshHandle, descriptor: MeshDescriptor): void {
        this.checkLive();
        const entry = this.mesh(mesh);
        const declared = this.declaredDeformBounds(descriptor, this.name('mesh', entry.handle, descriptor.label ?? entry.label));
        const prepared = this.prepareMeshData(descriptor);
        const pending = prepared.pending;
        const { layout, deform } = this.deformingLayout(prepared.layout, descriptor, declared);
        this.releaseVertexRange(entry);
        this.forgetPending(entry);
        this.triangleTotal += layout.triangleCount - entry.layout.triangleCount;
        entry.layout = layout;
        entry.pending = pending;
        // A new shape replaces any pose still waiting: it may not even have the same vertex count.
        entry.deform = deform;
        this.posesWaiting.delete(entry);
        entry.label = descriptor.label ?? entry.label;
        this.pendingMeshes.add(entry);
        for (const instance of entry.instances) {
            const traceable = this.isTraceable(instance);
            if (traceable) this.changeLog.addRegion(this.transforms.box(instance.slot), 'geometry', instance.handle, undefined, instance.slot, 0, this.emits(instance));
            this.transforms.setMeshBounds(instance.slot, layout.bounds);
            this.writeInstanceRecord(instance);
            if (traceable) {
                this.changeLog.addRegion(this.transforms.box(instance.slot), 'geometry', instance.handle, undefined, instance.slot, 0, this.emits(instance));
                this.topLevelDirty = true;
            }
        }
        this.cachedBounds = null;
    }

    /**
     * Keeps a copy of the pose for the next `encode` (applyPoses), replacing
     * any pose still waiting. Everything is checked here, so a bad call
     * throws at the call and changes nothing.
     */
    setDeformedPositions(mesh: MeshHandle, positions: Float32Array | DeformedPositions): void {
        this.checkLive();
        const entry = this.mesh(mesh);
        const what = this.name('mesh', entry.handle, entry.label);
        const deform = entry.deform;
        if (!deform) throw new Error(`@pmndrs/ddgi: setDeformedPositions: ${what} was not created with deforming: true.`);
        if (positions instanceof Float32Array) {
            if (positions.length !== deform.current.length) {
                throw new RangeError(`@pmndrs/ddgi: setDeformedPositions: ${what} has ${deform.vertexCount} vertices, but ${positions.length} numbers were given (3 per vertex).`);
            }
            copyCheckedPositions(positions, deform.waiting, `setDeformedPositions for ${what}`);
            deform.waitingOnGpu = null;
        } else {
            // The platform can't read GPU positions, so it can't grow the box to hold them: the host must give one.
            if (!deform.declaredBounds) {
                throw new Error(`@pmndrs/ddgi: setDeformedPositions: ${what} needs deformBounds (given at createMesh) for positions on the GPU, which the platform can't read to find their box.`);
            }
            deform.waitingOnGpu = checkedGpuPose(positions, deform.vertexCount, `setDeformedPositions for ${what}`);
        }
        if (!this.posesTraced) {
            // Checked like any other call, then left out: the tree can't follow it (no subgroups for the refit),
            // and hit points from moved positions on an unmoved tree would be wrong.
            deform.waitingOnGpu = null;
            if (this.warnedRestPose) return;
            this.warnedRestPose = true;
            this.warn('@pmndrs/ddgi: this device has no subgroups, so deforming meshes stay in their rest pose for GI (the mesh-tree refit needs subgroups).');
            return;
        }
        deform.hasWaiting = true;
        this.posesWaiting.add(entry);
    }

    destroyMesh(mesh: MeshHandle): void {
        this.checkLive();
        const entry = this.mesh(mesh);
        if (entry.instances.size > 0) {
            throw new Error(`@pmndrs/ddgi: ${this.name('mesh', entry.handle, entry.label)} is still used by ${entry.instances.size} instance(s); destroy them first.`);
        }
        this.releaseVertexRange(entry);
        this.forgetPending(entry);
        this.pendingMeshes.delete(entry);
        this.posesWaiting.delete(entry);
        this.mirror.clearRecord('meshes', entry.slot);
        this.meshSlots.release(entry.slot);
        this.removedMeshSlots.push(entry.slot);
        this.triangleTotal -= entry.layout.triangleCount;
        this.meshes.delete(mesh);
        this.headerDirty = true;
    }

    // -----------------------------------------------------------------------
    // Textures and materials
    // -----------------------------------------------------------------------

    /**
     * Registers a texture and queues it for the texture store: the next
     * `encode` makes its layer (a reference to the source is kept until
     * then). Until that layer exists its materials shade with their flat
     * colour; when it lands, their records are rewritten and a material
     * change is logged over the instances using them.
     */
    createTexture(descriptor: TextureDescriptor): TextureHandle {
        this.checkLive();
        if (!descriptor || !descriptor.source) throw new TypeError('@pmndrs/ddgi: a texture needs a source');
        const handle = this.nextHandle++ as TextureHandle;
        const entry: TextureEntry = { handle, slot: this.textureSlots.take(), srgb: descriptor.srgb, label: descriptor.label, source: descriptor.source, state: 'pending', users: 0 };
        this.textures.set(handle, entry);
        this.pendingTextures.add(entry);
        return handle;
    }

    destroyTexture(texture: TextureHandle): void {
        this.checkLive();
        const entry = this.texture(texture);
        if (entry.users > 0) throw new Error(`@pmndrs/ddgi: texture ${texture} is still used by ${entry.users} material(s).`);
        // Its layer is free for the next texture; nothing reads it until that one's layer is made.
        this.pendingTextures.delete(entry);
        this.textureSlots.release(entry.slot);
        this.textures.delete(texture);
    }

    /** Textures whose layer is made, waiting, or refused (for tests and tools). */
    textureCounts(): { readonly built: number; readonly pending: number; readonly failed: number } {
        const counts = { built: 0, pending: 0, failed: 0 };
        for (const entry of this.textures.values()) counts[entry.state]++;
        return counts;
    }

    /** A texture's layer in the store once it is made; null while it waits or when it was refused. */
    textureLayer(texture: TextureHandle): number | null {
        const entry = this.texture(texture);
        return entry.state === 'built' ? entry.slot : null;
    }

    createMaterial(descriptor: MaterialDescriptor): MaterialHandle {
        this.checkLive();
        this.checkMaterial(descriptor);
        const handle = this.nextHandle++ as MaterialHandle;
        const slot = this.materialSlots.take();
        this.mirror.ensureRecord('materials', slot);
        const entry: MaterialEntry = {
            handle,
            slot,
            descriptor: { ...descriptor },
            albedoTexture: undefined,
            emissiveTexture: undefined,
            metalnessTexture: undefined,
            users: new Set(),
        };
        this.attachTextures(entry);
        this.materials.set(handle, entry);
        this.writeMaterialRecord(entry);
        this.headerDirty = true;
        return handle;
    }

    updateMaterial(material: MaterialHandle, changes: Partial<MaterialDescriptor>): void {
        this.checkLive();
        const entry = this.material(material);
        const descriptor = { ...entry.descriptor, ...changes };
        this.checkMaterial(descriptor);
        this.detachTextures(entry);
        entry.descriptor = descriptor;
        this.attachTextures(entry);
        this.writeMaterialRecord(entry);
        for (const instance of entry.users) {
            if (this.isTraceable(instance)) this.changeLog.addRegion(this.transforms.box(instance.slot), 'material', instance.handle);
        }
    }

    destroyMaterial(material: MaterialHandle): void {
        this.checkLive();
        const entry = this.material(material);
        if (entry.users.size > 0) {
            throw new Error(`@pmndrs/ddgi: ${this.name('material', entry.handle, entry.descriptor.label)} is still used by ${entry.users.size} instance(s).`);
        }
        this.detachTextures(entry);
        this.mirror.clearRecord('materials', entry.slot);
        this.materialSlots.release(entry.slot);
        this.materials.delete(material);
        this.headerDirty = true;
    }

    // -----------------------------------------------------------------------
    // Instances
    // -----------------------------------------------------------------------

    createInstance(descriptor: InstanceDescriptor): InstanceHandle {
        this.checkLive();
        const mesh = this.mesh(descriptor.mesh);
        const materials = this.materialList(descriptor.materials, mesh);
        const objectToWorld = this.checkedTransform(descriptor.transform);
        const handle = this.nextHandle++ as InstanceHandle;
        const slot = this.instanceSlots.take();
        this.mirror.ensureRecord('instances', slot);
        const entry: InstanceEntry = {
            handle,
            slot,
            label: descriptor.label,
            mesh,
            materials,
            contributesToGi: descriptor.contributesToGI !== false,
            visible: descriptor.visible !== false,
            listOffset: this.mirror.allocateList(materials.length),
        };
        this.transforms.ensure(slot);
        this.transforms.setMatrix(slot, objectToWorld, 0);
        this.transforms.setMeshBounds(slot, mesh.layout.bounds);
        this.instances.set(handle, entry);
        this.instanceSetVersion++;
        mesh.instances.add(entry);
        for (const material of materials) material.users.add(entry);
        this.mirror.writeList(entry.listOffset, materials.map((material) => material.slot));
        this.writeInstanceRecord(entry);
        if (this.isTraceable(entry)) {
            this.changeLog.addRegion(this.transforms.box(slot), 'geometry', handle, undefined, slot, 0, this.emits(entry));
            this.topLevelDirty = true;
        } else if (this.isReceiver(entry)) {
            this.changeLog.addReceiver(this.transforms.bounds, slot * BOX_NUMBERS);
        }
        if (entry.contributesToGi) this.instanceSetDirty = true;
        this.cachedBounds = null;
        this.headerDirty = true;
        return handle;
    }

    setTransform(instance: InstanceHandle, transform: Mat4): void {
        this.checkLive();
        const entry = this.instance(instance);
        const objectToWorld = this.checkedTransform(transform);
        this.moveInstance(entry, objectToWorld, 0);
        this.mirror.markRecord('instances', entry.slot);
    }

    /**
     * Many transforms at once (SceneApi): every handle and number is checked
     * first, so a bad entry changes nothing; then each instance is moved as
     * `setTransform` moves it, writing straight into the table mirror, and
     * the records are marked for upload as one span. Nothing is allocated
     * per instance.
     */
    setTransforms(instances: Uint32Array, matrices: Float32Array): void {
        this.checkLive();
        const count = instances.length;
        if (matrices.length < count * 16) {
            throw new RangeError(`@pmndrs/ddgi: setTransforms needs 16 numbers per instance: ${count} instances, ${matrices.length} numbers.`);
        }
        const entries = this.batchEntriesFor(instances);
        // Any non-finite number makes the sum of number × 0 NaN: one pass with no branch per number.
        // Four sums, one per matrix column, so the additions don't wait on each other.
        let column0 = 0;
        let column1 = 0;
        let column2 = 0;
        let column3 = 0;
        for (let index = 0; index < count * 16; index += 4) {
            column0 += matrices[index] * 0;
            column1 += matrices[index + 1] * 0;
            column2 += matrices[index + 2] * 0;
            column3 += matrices[index + 3] * 0;
        }
        if (column0 + column1 + column2 + column3 !== 0) {
            const bad = matrices.subarray(0, count * 16).findIndex((value) => !Number.isFinite(value));
            throw new RangeError(`@pmndrs/ddgi: a transform must be finite (setTransforms entry ${Math.floor(bad / 16)}).`);
        }
        let firstSlot = Infinity;
        let lastSlot = -1;
        for (let index = 0; index < count; index++) {
            const entry = entries[index];
            // A slide in one pass when it is one (slideInstance), else the whole work.
            if (!(this.isTraceable(entry) && this.slideInstance(entry, matrices, index * 16))) this.moveInstance(entry, matrices, index * 16);
            if (entry.slot < firstSlot) firstSlot = entry.slot;
            if (entry.slot > lastSlot) lastSlot = entry.slot;
        }
        if (lastSlot >= 0) this.mirror.markRecordSpan('instances', firstSlot, lastSlot);
    }

    /** The instances `setTransforms` names, looked up (throwing for an unknown one), or the last call's when they're the same (batchEntries). */
    private batchEntriesFor(instances: Uint32Array): InstanceEntry[] {
        const count = instances.length;
        const previous = this.batchHandles;
        if (this.batchVersion === this.instanceSetVersion && previous.length === count) {
            let same = true;
            for (let index = 0; index < count && same; index++) same = previous[index] === instances[index];
            if (same) return this.batchEntries;
        }
        const entries: InstanceEntry[] = new Array(count);
        for (let index = 0; index < count; index++) entries[index] = this.instance(instances[index] as InstanceHandle);
        this.batchEntries = entries;
        this.batchHandles = instances.slice();
        this.batchVersion = this.instanceSetVersion;
        return entries;
    }

    /**
     * Moves an instance to the transform at `matrices[offset ... offset + 16)`
     * and logs it: where it was and where it is, both with how far it moved
     * (`DirtyRegion.moved`), or, for a visible receiver, where it is now
     * (`SceneChanges.receivers`). Its record is written but
     * not marked for upload; the caller marks it, alone or in a span.
     */
    private moveInstance(entry: InstanceEntry, matrices: ArrayLike<number>, offset: number): void {
        const transforms = this.transforms;
        const slot = entry.slot;
        const linearChanged = transforms.setMatrix(slot, matrices, offset);
        if (!this.isTraceable(entry)) {
            this.writeInstanceWords(entry, linearChanged);
            if (this.isReceiver(entry)) this.changeLog.addReceiver(transforms.bounds, slot * BOX_NUMBERS);
            return;
        }
        // Where it was, kept before the record is rewritten (copied number by number:
        // TypedArray.set costs more than six on this path).
        const boundsBefore = this.boundsBefore;
        const at = slot * BOX_NUMBERS;
        for (let index = 0; index < BOX_NUMBERS; index++) boundsBefore[index] = transforms.bounds[at + index];
        const moved = this.writeInstanceWords(entry, linearChanged);
        this.changeLog.addRegion(boundsBefore, 'geometry', entry.handle, moved, slot, 0, this.emits(entry));
        this.changeLog.addRegion(transforms.bounds, 'geometry', entry.handle, moved, slot, at, this.emits(entry));
        this.topLevelDirty = true;
        this.cachedBounds = null;
    }

    /**
     * `moveInstance` for a traceable instance whose move only translates, in
     * one pass: the matrix is read once, the frame slides (InstanceTransforms
     * .slide), and only what a slide changes in the record is rewritten, the
     * translation column and the world box (the normal matrix, the flags, the
     * mesh and material words stay). Returns false, having changed nothing,
     * when the linear part changed; `moveInstance` then does the whole work.
     * *Why a path of its own:* most moves are slides (drags, B1a's bench), and
     * at 10,000 a frame the general path's three reads of each matrix and its
     * full record were most of `setTransforms`' time (T2.M1).
     * Only `setTransforms` uses it, always with a Float32Array: fed
     * `setTransform`'s Float64Array matrices as well, Chrome compiled its
     * reads for both kinds and the call took 2.65 ms instead of 1.6.
     */
    private slideInstance(entry: InstanceEntry, matrices: Float32Array, offset: number): boolean {
        const transforms = this.transforms;
        const slot = entry.slot;
        const stored = transforms.matrices;
        const m = slot * MATRIX_NUMBERS;
        for (let index = 0; index < 12; index++) if (matrices[offset + index] !== stored[m + index]) return false;
        const bounds = transforms.bounds;
        const at = slot * BOX_NUMBERS;
        const boundsBefore = this.boundsBefore;
        for (let index = 0; index < BOX_NUMBERS; index++) boundsBefore[index] = bounds[at + index];
        stored[m + 12] = matrices[offset + 12];
        stored[m + 13] = matrices[offset + 13];
        stored[m + 14] = matrices[offset + 14];
        stored[m + 15] = matrices[offset + 15];
        const moved = transforms.slide(slot);
        const floats = this.mirror.floats;
        const base = this.mirror.recordWord('instances', slot);
        // Column-major input: the translation is elements 12-14, each row's last number.
        floats[base + INSTANCE_WORD.objectToWorldRow0 + 3] = stored[m + 12];
        floats[base + INSTANCE_WORD.objectToWorldRow1 + 3] = stored[m + 13];
        floats[base + INSTANCE_WORD.objectToWorldRow2 + 3] = stored[m + 14];
        for (let axis = 0; axis < 3; axis++) {
            floats[base + INSTANCE_WORD.boundsMin + axis] = bounds[at + axis];
            floats[base + INSTANCE_WORD.boundsMax + axis] = bounds[at + 3 + axis];
        }
        this.changeLog.addRegion(boundsBefore, 'geometry', entry.handle, moved, slot, 0, this.emits(entry));
        this.changeLog.addRegion(bounds, 'geometry', entry.handle, moved, slot, at, this.emits(entry));
        this.topLevelDirty = true;
        this.cachedBounds = null;
        return true;
    }

    setVisible(instance: InstanceHandle, visible: boolean): void {
        this.checkLive();
        const entry = this.instance(instance);
        if (entry.visible === visible) return;
        entry.visible = visible;
        this.writeInstanceRecord(entry);
        if (entry.contributesToGi) {
            this.changeLog.addRegion(this.transforms.box(entry.slot), 'geometry', instance, undefined, entry.slot, 0, this.emits(entry));
            this.topLevelDirty = true;
            this.cachedBounds = null;
        } else if (visible) {
            // A receiver that appears needs GI where it is, as one that moved there does.
            this.changeLog.addReceiver(this.transforms.bounds, entry.slot * BOX_NUMBERS);
        }
    }

    setMaterials(instance: InstanceHandle, materials: readonly MaterialHandle[]): void {
        this.checkLive();
        const entry = this.instance(instance);
        const next = this.materialList(materials, entry.mesh);
        for (const material of entry.materials) material.users.delete(entry);
        if (next.length !== entry.materials.length) {
            this.mirror.releaseList(entry.listOffset, entry.materials.length);
            entry.listOffset = this.mirror.allocateList(next.length);
        }
        entry.materials = next;
        for (const material of next) material.users.add(entry);
        this.mirror.writeList(entry.listOffset, next.map((material) => material.slot));
        this.writeInstanceRecord(entry);
        if (this.isTraceable(entry)) this.changeLog.addRegion(this.transforms.box(entry.slot), 'material', instance);
    }

    destroyInstance(instance: InstanceHandle): void {
        this.checkLive();
        const entry = this.instance(instance);
        if (this.isTraceable(entry)) {
            this.changeLog.addRegion(this.transforms.box(entry.slot), 'geometry', instance, undefined, entry.slot, 0, this.emits(entry));
            this.topLevelDirty = true;
        }
        if (entry.contributesToGi) this.instanceSetDirty = true;
        entry.mesh.instances.delete(entry);
        for (const material of entry.materials) material.users.delete(entry);
        this.mirror.releaseList(entry.listOffset, entry.materials.length);
        this.mirror.clearRecord('instances', entry.slot);
        this.instanceSlots.release(entry.slot);
        this.instances.delete(instance);
        this.instanceSetVersion++;
        // The last batch may name it: drop the reused lookups, and the instance with them.
        this.batchEntries = [];
        this.batchVersion = -1;
        this.cachedBounds = null;
        this.headerDirty = true;
    }

    // -----------------------------------------------------------------------
    // Lights and environment
    // -----------------------------------------------------------------------

    createLight(descriptor: LightDescriptor): LightHandle {
        this.checkLive();
        const checked = this.checkedLight(descriptor);
        const handle = this.nextHandle++ as LightHandle;
        const entry: LightEntry = { handle, index: -1, descriptor: checked };
        this.changeLog.noteLight(handle, null);
        // Directional lights stay at the front: the first point or spot light moves to the end to make room.
        if (checked.type === 'directional') {
            const displaced = this.lights[this.directionalLightCount];
            if (displaced) this.placeLight(displaced, this.lights.length);
            this.placeLight(entry, this.directionalLightCount);
            this.directionalLightCount++;
        } else {
            this.placeLight(entry, this.lights.length);
        }
        this.lightsByHandle.set(handle, entry);
        this.headerDirty = true;
        return handle;
    }

    updateLight(light: LightHandle, changes: LightChanges): void {
        this.checkLive();
        const entry = this.light(light);
        if ('type' in changes && (changes as { type?: unknown }).type !== entry.descriptor.type) {
            throw new Error('@pmndrs/ddgi: a light cannot change type; destroy it and create a new one.');
        }
        const next = this.checkedLight({ ...entry.descriptor, ...changes } as LightDescriptor);
        // The log keeps the state the frame started with; it works out boxes and flags at encode.
        this.changeLog.noteLight(light, entry.descriptor);
        entry.descriptor = next;
        this.writeLightRecord(entry);
    }

    destroyLight(light: LightHandle): void {
        this.checkLive();
        const entry = this.light(light);
        this.changeLog.noteLight(light, entry.descriptor);
        const last = this.lights.length - 1;
        if (entry.descriptor.type === 'directional') {
            // Fill the hole with the last directional light, and that light's place with the last light.
            const lastDirectional = this.directionalLightCount - 1;
            if (entry.index !== lastDirectional) this.placeLight(this.lights[lastDirectional], entry.index);
            if (lastDirectional !== last) this.placeLight(this.lights[last], lastDirectional);
            this.directionalLightCount--;
        } else if (entry.index !== last) {
            this.placeLight(this.lights[last], entry.index);
        }
        this.lights.length = last;
        this.mirror.clearRecord('lights', last);
        this.lightsByHandle.delete(light);
        this.headerDirty = true;
    }

    /**
     * Stores the environment. The projection to a small engine-neutral form
     * (SH plus an octahedral map) belongs to hit shading (T1.4); the scene copy
     * keeps the descriptor and the plain values in its header.
     */
    setEnvironment(environment: EnvironmentDescriptor): void {
        this.checkLive();
        if (environment.type === 'color') {
            checkColor(environment.color, 'environment color');
            checkNonNegative(environment.intensity, 'environment intensity');
        } else if (environment.type === 'procedural') {
            normalized(environment.sunDirection, 'sunDirection');
            checkNonNegative(environment.turbidity, 'turbidity');
            checkNonNegative(environment.sunIlluminance, 'sunIlluminance');
            checkNonNegative(environment.intensity, 'environment intensity');
        } else if (environment.type === 'texture') {
            if (!environment.texture) throw new TypeError('@pmndrs/ddgi: a texture environment needs a texture');
            checkNonNegative(environment.intensity, 'environment intensity');
        } else if (environment.type !== 'none') {
            throw new TypeError(`@pmndrs/ddgi: unknown environment type ${(environment as { type: string }).type}`);
        }
        this.environmentDescriptor = environment;
        this.environmentChanges++;
        this.changeLog.addGlobal('environment');
        this.headerDirty = true;
    }

    // -----------------------------------------------------------------------
    // Per frame
    // -----------------------------------------------------------------------

    /**
     * Uploads everything queued since the last call, into the host's encoder
     * (see the top of this file for the order of writes and copies). Call once
     * per frame, and submit the encoder before the next call: buffers replaced
     * here are destroyed at the next call.
     */
    encode(encoder: GPUCommandEncoder, timer?: GpuTimerScope): SceneUpload {
        this.checkLive();
        if (!this.packer.ready) {
            // Checked before anything changes, so staged meshes stay queued for the encode after prepare.
            for (const mesh of this.pendingMeshes) if (mesh.pending?.staged) throw new GINotReadyError('preparing');
        }
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];

        const versionBefore = this.bufferVersion;
        // Staged meshes' bytes were written when they were staged.
        let bytesWritten = this.stagedBytes;
        this.stagedBytes = 0;
        // Poses first: they write into meshes already in the vertex buffer (CPU poses with writeBuffer,
        // GPU poses with commands recorded here), which a growth below copies into the new buffer (its
        // copies are recorded after both).
        const poses = this.applyPoses(encoder);
        bytesWritten += poses.bytesWritten;
        const meshes = this.uploadMeshes(encoder, timer);
        bytesWritten += meshes.bytesWritten;
        // Before the tables: the records of materials whose textures land now are rewritten.
        this.buildTextures(encoder, timer);
        bytesWritten += this.uploadTables();

        const upload: SceneUpload = {
            changes: this.changeLog.take(this.lightLookup),
            meshesUploaded: meshes.uploaded,
            meshesPosed: poses.posed,
            meshesRemoved: this.removedMeshSlots,
            instancesChanged: this.topLevelDirty,
            instanceSetChanged: this.instanceSetDirty,
            reallocated: this.bufferVersion !== versionBefore,
            bytesWritten,
        };
        this.removedMeshSlots = [];
        this.topLevelDirty = false;
        this.instanceSetDirty = false;
        return upload;
    }

    /** Frees the GPU buffers. The scene can't be used afterwards. */
    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
        this.vertexBuffer.destroy();
        this.tableBuffer.destroy();
        this.packer.dispose();
        this.poseCopier.dispose();
        this.textureStore.dispose();
    }

    // -----------------------------------------------------------------------
    // Textures
    // -----------------------------------------------------------------------

    /**
     * Hands the store every texture created since the last encode, once its
     * kernel is compiled (before that they wait). Textures whose layer was
     * made get their materials' records rewritten, and a material change is
     * logged over the traceable instances using those materials, so
     * techniques refresh what they saw of them (DDGI's probes).
     */
    private buildTextures(encoder: GPUCommandEncoder, timer: GpuTimerScope | undefined): void {
        if (!this.textureStore.ready) return;
        const byLayer = new Map<number, TextureEntry>();
        const requests: TextureLayerRequest[] = [];
        for (const entry of this.pendingTextures) {
            if (!entry.source) continue;
            byLayer.set(entry.slot, entry);
            requests.push({ layer: entry.slot, source: entry.source, srgb: entry.srgb, label: entry.label });
        }
        this.pendingTextures.clear();
        // A source that throws is refused alone inside the store (texture-store.ts, stageCatching).
        // If the store's encode fails as a whole, this frame's textures fail with it rather than
        // staying pending for good (they were taken off the pending set above).
        let results: TextureLayerResults;
        try {
            results = this.textureStore.encode(encoder, requests, timer);
        } catch (error) {
            for (const entry of byLayer.values()) {
                entry.state = 'failed';
                entry.source = null;
            }
            throw error;
        }
        for (const layer of results.failed) {
            const entry = byLayer.get(layer);
            if (!entry) continue;
            entry.state = 'failed';
            entry.source = null;
        }
        // Sources the store is still checking (a host's GPUTexture) are asked for again next time.
        for (const layer of results.waiting) {
            const entry = byLayer.get(layer);
            if (entry) this.pendingTextures.add(entry);
        }
        const landed = new Set<TextureEntry>();
        for (const layer of results.built) {
            const entry = byLayer.get(layer);
            if (!entry) continue;
            entry.state = 'built';
            entry.source = null;
            landed.add(entry);
        }
        if (landed.size === 0) return;
        for (const material of this.materials.values()) {
            const uses = [material.albedoTexture, material.emissiveTexture, material.metalnessTexture].some((texture) => texture !== undefined && landed.has(texture));
            if (!uses) continue;
            this.writeMaterialRecord(material);
            for (const instance of material.users) {
                if (this.isTraceable(instance)) this.changeLog.addRegion(this.transforms.box(instance.slot), 'material', instance.handle);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Uploads
    // -----------------------------------------------------------------------

    /**
     * Applies the waiting poses of deforming meshes (`setDeformedPositions`):
     * writes each into its mesh's positions in the vertex buffer (a CPU pose
     * with `writeBuffer`, a GPU pose with the pose copy recorded into
     * `encoder`), reports the mesh in `meshesPosed` so the tracer refits its
     * tree from those positions in the same frame, and logs it: per instance,
     * the mesh's deform box in world space, with how far the vertices
     * travelled (`DirtyRegion.moved`), or a receiver box. When a CPU pose
     * leaves the box, the box grows: the instances' boxes change, so the box
     * before is logged too. A GPU pose can't be read here, so its box stays
     * the declared one and its `moved` is that box's diagonal (the most any
     * vertex inside the box can travel).
     *
     * A mesh whose words haven't reached the GPU yet (created or replaced
     * this frame) keeps its pose waiting for the next encode: its data is
     * written by this encode's upload, after these writes. A GPU pose that
     * can't be copied in this encode (its mesh isn't on the GPU yet, or the
     * copy kernel is still compiling) is dropped instead: the host only
     * promises its buffer or texture until it submits this frame, and may
     * destroy it after (the contract), so keeping a reference for later
     * could read a destroyed buffer. The host's next pose takes its place.
     */
    private applyPoses(encoder: GPUCommandEncoder): { readonly posed: PosedMesh[]; readonly bytesWritten: number } {
        const posed: PosedMesh[] = [];
        const copies: PoseCopy[] = [];
        let bytesWritten = 0;
        for (const mesh of this.posesWaiting) {
            const deform = mesh.deform;
            if (!deform || !deform.hasWaiting) {
                this.posesWaiting.delete(mesh);
                continue;
            }
            const onGpu = mesh.pending === null && mesh.vertexOffset >= 0;
            if (deform.waitingOnGpu && !(onGpu && this.poseCopier.ready)) {
                deform.waitingOnGpu = null;
                deform.hasWaiting = false;
                this.posesWaiting.delete(mesh);
                continue;
            }
            if (!onGpu) continue;
            this.posesWaiting.delete(mesh);

            const positionsWord = mesh.vertexOffset + mesh.layout.offsets.positions;
            let travelled: number;
            let grew = false;
            const gpuPose = deform.waitingOnGpu ? deform.takeGpuPose() : null;
            if (gpuPose) {
                copies.push({ pose: gpuPose, vertexCount: deform.vertexCount, destinationWord: positionsWord });
                travelled = boxDiagonal(mesh.layout.bounds);
            } else {
                const known = deform.currentKnown;
                travelled = scanPose(deform.waiting, deform.current, this.poseBounds);
                deform.swap();
                // After a GPU pose the CPU's copy of the positions is out of date: the box's diagonal bounds the travel.
                if (!known) travelled = Math.max(travelled, boxDiagonal(mesh.layout.bounds));
                this.device.queue.writeBuffer(this.vertexBuffer, positionsWord * 4, deform.current);
                bytesWritten += deform.current.byteLength;

                if (deform.declaredBounds && !deform.warnedOutside && !boxInside(this.poseBounds, deform.declaredBounds)) {
                    deform.warnedOutside = true;
                    this.warn(`@pmndrs/ddgi: ${this.name('mesh', mesh.handle, mesh.label)}: deformed positions left its deformBounds; the box grows to hold them (this warning is shown once).`);
                }
                const grownBounds = Float32Array.from(mesh.layout.bounds);
                grew = growBox(grownBounds, this.poseBounds);
                if (grew) {
                    mesh.layout = { ...mesh.layout, bounds: grownBounds };
                    this.writeMeshRecord(mesh);
                    this.cachedBounds = null;
                }
            }
            for (const instance of mesh.instances) {
                const traceable = this.isTraceable(instance);
                const slot = instance.slot;
                if (grew) {
                    if (traceable) this.changeLog.addRegion(this.transforms.box(slot), 'geometry', instance.handle, undefined, slot, 0, this.emits(instance));
                    this.transforms.setMeshBounds(slot, mesh.layout.bounds);
                    this.writeInstanceRecord(instance);
                    if (traceable) this.topLevelDirty = true;
                }
                if (traceable) {
                    const moved = travelled * largestScale(this.transforms.matrices, slot);
                    this.changeLog.addRegion(this.transforms.box(slot), 'geometry', instance.handle, moved, slot, 0, this.emits(instance));
                } else if (this.isReceiver(instance)) {
                    this.changeLog.addReceiver(this.transforms.bounds, slot * BOX_NUMBERS);
                }
            }
            posed.push({ slot: mesh.slot, layout: mesh.layout, vertexOffset: mesh.vertexOffset });
        }
        this.poseCopier.encode(encoder, this.vertexBuffer, copies);
        return { posed, bytesWritten };
    }

    /** A deforming descriptor's `deformBounds`, checked; null when it gives none or isn't deforming. */
    private declaredDeformBounds(descriptor: MeshDescriptor, what: string): Float32Array | null {
        return descriptor.deforming === true && descriptor.deformBounds ? checkedDeformBounds(descriptor.deformBounds, what) : null;
    }

    /**
     * For a `deforming` descriptor: its layout with the box every pose stays
     * in (the rest pose's, grown to `declared`, its checked `deformBounds`),
     * so its instances' boxes don't change from pose to pose; and the state
     * its poses need. Other meshes keep their layout and get no state.
     */
    private deformingLayout(layout: MeshLayout, descriptor: MeshDescriptor, declared: Float32Array | null): { layout: MeshLayout; deform: DeformState | null } {
        if (descriptor.deforming !== true) return { layout, deform: null };
        const bounds = Float32Array.from(layout.bounds);
        if (declared) growBox(bounds, declared);
        return { layout: { ...layout, bounds }, deform: new DeformState(descriptor.positions, declared) };
    }

    /**
     * Places the queued meshes in the vertex buffer and uploads them: the CPU
     * packer's words with `writeBuffer`, staged meshes with the packing
     * kernel, recorded after any growth copies so it writes into the new
     * buffer. `bytesWritten` counts the CPU-packed meshes' writes.
     */
    private uploadMeshes(encoder: GPUCommandEncoder, timer: GpuTimerScope | undefined): { readonly uploaded: UploadedMesh[]; readonly bytesWritten: number } {
        if (this.pendingMeshes.size === 0) {
            // Still called: it destroys the staging buffers the last encode packed from.
            this.packer.encode(encoder, this.vertexBuffer, [], timer);
            return { uploaded: [], bytesWritten: 0 };
        }
        const pending = [...this.pendingMeshes];
        this.pendingMeshes.clear();

        const unplaced: MeshEntry[] = [];
        for (const mesh of pending) {
            const offset = this.vertexAllocator.allocate(mesh.layout.wordCount);
            if (offset === null) unplaced.push(mesh);
            else mesh.vertexOffset = offset;
        }
        if (unplaced.length > 0) {
            const wordsNeeded = unplaced.reduce((sum, mesh) => sum + mesh.layout.wordCount, 0);
            try {
                this.growVertexBuffer(encoder, this.vertexAllocator.capacityToFit(wordsNeeded), new Set(pending));
            } catch (error) {
                // Over the device's limit: leave every mesh queued, as before this call.
                for (const mesh of pending) {
                    this.releaseVertexRange(mesh);
                    this.pendingMeshes.add(mesh);
                }
                throw error;
            }
            for (const mesh of unplaced) {
                const offset = this.vertexAllocator.allocate(mesh.layout.wordCount);
                if (offset === null) throw new Error('vertex allocation failed after growing the buffer');
                mesh.vertexOffset = offset;
            }
        }

        const uploaded: UploadedMesh[] = [];
        const toPack: MeshToPack[] = [];
        let bytesWritten = 0;
        for (const mesh of pending) {
            const data = mesh.pending;
            if (!data) throw new Error('a queued mesh has no data to upload');
            if (data.staged) {
                toPack.push({ staged: data.staged, vertexOffset: mesh.vertexOffset });
            } else if (data.words) {
                this.device.queue.writeBuffer(this.vertexBuffer, mesh.vertexOffset * 4, data.words);
                bytesWritten += data.words.byteLength;
            }
            this.writeMeshRecord(mesh);
            uploaded.push({ slot: mesh.slot, handle: mesh.handle, layout: mesh.layout, vertexOffset: mesh.vertexOffset, geometry: data.geometry });
            // The GPU has the words, or will once the kernel below runs; keep only the layout.
            mesh.pending = null;
        }
        this.packer.encode(encoder, this.vertexBuffer, toPack, timer);
        return { uploaded, bytesWritten };
    }
    /**
     * Replaces the vertex buffer with one of at least `minimumWords`, and
     * records copies of every mesh already on the GPU into it. Meshes in
     * `beingWritten` are skipped: their words go straight into the new buffer.
     */
    private growVertexBuffer(encoder: GPUCommandEncoder, minimumWords: number, beingWritten: Set<MeshEntry>): void {
        const limits = this.device.limits;
        const limitBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
        const limitWords = Math.floor(limitBytes / 4);
        if (minimumWords > limitWords) {
            const limit = limits.maxStorageBufferBindingSize <= limits.maxBufferSize ? 'maxStorageBufferBindingSize' : 'maxBufferSize';
            throw new GILimitError(limit, minimumWords * 4, limitBytes);
        }
        // Doubling keeps the number of reallocations logarithmic while a scene loads.
        const words = Math.min(limitWords, Math.max(minimumWords, this.vertexAllocator.capacity * 2));
        const buffer = this.createStorageBuffer('scene vertices', words * 4);

        const resident = [...this.meshes.values()]
            .filter((mesh) => mesh.vertexOffset >= 0 && !beingWritten.has(mesh))
            .sort((a, b) => a.vertexOffset - b.vertexOffset);
        // Neighbouring meshes are copied as one range.
        let runStart = -1;
        let runEnd = -1;
        const flush = () => {
            if (runStart >= 0) encoder.copyBufferToBuffer(this.vertexBuffer, runStart * 4, buffer, runStart * 4, (runEnd - runStart) * 4);
        };
        for (const mesh of resident) {
            if (mesh.vertexOffset === runEnd) {
                runEnd += mesh.layout.wordCount;
            } else {
                flush();
                runStart = mesh.vertexOffset;
                runEnd = mesh.vertexOffset + mesh.layout.wordCount;
            }
        }
        flush();

        this.retired.push(this.vertexBuffer);
        this.vertexBuffer = buffer;
        this.vertexAllocator.grow(words);
        this.bufferVersion++;
    }

    /** Uploads the table mirror's dirty ranges, reallocating when the layout grew. Returns bytes written. */
    private uploadTables(): number {
        if (this.headerDirty || this.mirror.layoutVersion !== this.uploadedLayoutVersion) this.writeHeader();
        const ranges = this.mirror.takeDirtyRanges();

        if (this.mirror.layoutVersion !== this.uploadedLayoutVersion) {
            const bindingLimit = this.device.limits.maxStorageBufferBindingSize;
            if (this.mirror.byteLength > bindingLimit) {
                throw new GILimitError('maxStorageBufferBindingSize', this.mirror.byteLength, bindingLimit);
            }
            if (this.mirror.byteLength > this.tableBuffer.size) {
                this.retired.push(this.tableBuffer);
                this.tableBuffer = this.createStorageBuffer('scene tables', this.mirror.byteLength);
                this.bufferVersion++;
            }
            this.uploadedLayoutVersion = this.mirror.layoutVersion;
            this.device.queue.writeBuffer(this.tableBuffer, 0, this.mirror.words);
            return this.mirror.byteLength;
        }

        let bytes = 0;
        for (const [start, end] of ranges) {
            // For typed arrays, writeBuffer's data offset and size count elements (words).
            this.device.queue.writeBuffer(this.tableBuffer, start * 16, this.mirror.words, start * 4, (end - start) * 4);
            bytes += (end - start) * 16;
        }
        return bytes;
    }

    private createStorageBuffer(name: string, size: number): GPUBuffer {
        return this.device.createBuffer({
            label: `${this.label} ${name}`,
            size,
            // COPY_SRC: the vertex buffer is copied out of when it grows (and tests read both back).
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }

    private releaseVertexRange(mesh: MeshEntry): void {
        if (mesh.vertexOffset < 0) return;
        this.vertexAllocator.release(mesh.vertexOffset, mesh.layout.wordCount);
        mesh.vertexOffset = -1;
    }

    /**
     * Checks and lays out a mesh, copies what the scene keeps of the host's
     * arrays, and stages them for the GPU or packs them on the CPU.
     */
    private prepareMeshData(given: MeshDescriptor): { layout: MeshLayout; pending: PendingMeshData } {
        // A deforming mesh is stored without normals: they would be the rest pose's. Hit shading then
        // uses each triangle's geometric normal (giInstanceNormalAt's fallback), worked out from the
        // positions the scene copy holds now, so it follows every pose (docs/design/m3-moving.md §3).
        const descriptor = given.deforming === true && given.normals ? { ...given, normals: undefined } : given;
        const plan = planMesh(descriptor);
        // Tree builders read positions and indices after this call has returned, when
        // the host may have changed its arrays: they get copies, made now.
        const geometry: MeshGeometryCopy = { positions: descriptor.positions.slice(), indices: descriptor.indices.slice() };
        if (this.gpuPacking && this.packer.fits(descriptor, plan.layout)) {
            const staged = this.packer.stage(descriptor, plan.layout, plan.materialSlotWords);
            this.stagedBytes += staged.bytes;
            return { layout: plan.layout, pending: { words: null, staged, geometry } };
        }
        return { layout: plan.layout, pending: { words: packMeshWords(descriptor, plan), staged: null, geometry } };
    }

    /** Drops a mesh's data that was never uploaded (the mesh was replaced or destroyed first). */
    private forgetPending(mesh: MeshEntry): void {
        if (mesh.pending?.staged) this.packer.forget(mesh.pending.staged);
        mesh.pending = null;
    }

    // -----------------------------------------------------------------------
    // Records
    // -----------------------------------------------------------------------

    private writeHeader(): void {
        const words = this.mirror.words;
        const floats = this.mirror.floats;
        const layout = this.mirror.layout;
        words.fill(0, 0, HEADER_VEC4S * 4);
        words[HEADER_WORD.instanceSlots] = this.instanceSlots.slotCount;
        words[HEADER_WORD.meshSlots] = this.meshSlots.slotCount;
        words[HEADER_WORD.materialSlots] = this.materialSlots.slotCount;
        words[HEADER_WORD.lightCount] = this.lights.length;
        words[HEADER_WORD.instancesBase] = layout.instancesBase;
        words[HEADER_WORD.meshesBase] = layout.meshesBase;
        words[HEADER_WORD.materialsBase] = layout.materialsBase;
        words[HEADER_WORD.lightsBase] = layout.lightsBase;
        words[HEADER_WORD.materialListsBase] = layout.materialListsBase;
        words[HEADER_WORD.directionalLightCount] = this.directionalLightCount;

        const environment = this.environmentDescriptor;
        words[HEADER_WORD.environmentType] = ENVIRONMENT_TYPE[environment.type];
        if (environment.type === 'color') {
            const intensity = environment.intensity ?? 1;
            for (let channel = 0; channel < 3; channel++) floats[HEADER_WORD.environmentColor + channel] = environment.color[channel] * intensity;
            floats[HEADER_WORD.environmentIntensity] = intensity;
        } else if (environment.type === 'procedural') {
            floats.set(normalized(environment.sunDirection, 'sunDirection'), HEADER_WORD.environmentSunDirection);
            // One turbidity default for the platform and the sky files (T1.4).
            floats[HEADER_WORD.environmentTurbidity] = environment.turbidity ?? DEFAULT_TURBIDITY;
            // The header's one brightness factor: the sun the sky is calibrated to, times the multiplier.
            floats[HEADER_WORD.environmentIntensity] = (environment.sunIlluminance ?? 1) * (environment.intensity ?? 1);
        } else if (environment.type === 'texture') {
            floats[HEADER_WORD.environmentIntensity] = environment.intensity ?? 1;
            floats[HEADER_WORD.environmentRotationY] = environment.rotationY ?? 0;
        }
        this.mirror.markHeader();
        this.headerDirty = false;
    }

    private writeMeshRecord(mesh: MeshEntry): void {
        const words = this.mirror.words;
        const floats = this.mirror.floats;
        const base = this.mirror.recordWord('meshes', mesh.slot);
        const packed = mesh.layout;
        const absolute = (offset: number) => (offset === NONE ? NONE : mesh.vertexOffset + offset);
        words[base + MESH_WORD.indices] = absolute(packed.offsets.indices);
        words[base + MESH_WORD.positions] = absolute(packed.offsets.positions);
        words[base + MESH_WORD.triangleCount] = packed.triangleCount;
        words[base + MESH_WORD.vertexCount] = packed.vertexCount;
        words[base + MESH_WORD.normals] = absolute(packed.offsets.normals);
        words[base + MESH_WORD.uvs0] = absolute(packed.offsets.uvs0);
        words[base + MESH_WORD.uvs1] = absolute(packed.offsets.uvs1);
        words[base + MESH_WORD.materialSlots] = absolute(packed.offsets.materialSlots);
        floats.set(packed.bounds.subarray(0, 3), base + MESH_WORD.boundsMin);
        words[base + MESH_WORD.constantMaterialSlot] = packed.constantMaterialSlot === NONE ? 0 : packed.constantMaterialSlot;
        floats.set(packed.bounds.subarray(3, 6), base + MESH_WORD.boundsMax);
        words[base + MESH_WORD.flags] = MESH_FLAG.live | (packed.indices16 ? MESH_FLAG.indices16 : 0) | (packed.deforming ? MESH_FLAG.deforming : 0);
        this.mirror.markRecord('meshes', mesh.slot);
    }

    /** Rewrites an instance's record, updates its world bounds and marks the record for upload. */
    private writeInstanceRecord(instance: InstanceEntry): void {
        this.writeInstanceWords(instance);
        this.mirror.markRecord('instances', instance.slot);
    }

    /**
     * Rewrites an instance's record in the mirror and updates its world box
     * and frame, without marking it; allocates nothing. Returns how far its
     * corners moved since the record was last written (transformFrame).
     * `linearChanged` false says only the translation changed since then, so
     * the normal matrix and the mirrored and singular flags in the record are
     * still right and aren't worked out again.
     */
    private writeInstanceWords(instance: InstanceEntry, linearChanged = true): number {
        const words = this.mirror.words;
        const floats = this.mirror.floats;
        const base = this.mirror.recordWord('instances', instance.slot);
        const transforms = this.transforms;
        const slot = instance.slot;
        if (linearChanged) this.writeNormalMatrix(slot, base);
        const matrix = transforms.matrices;
        const m = slot * MATRIX_NUMBERS;
        // Written out rather than looped: this runs for every instance `setTransforms` moves.
        // Column-major input: row r of the 3x4 matrix is elements r, 4 + r, 8 + r, 12 + r.
        const row0 = base + INSTANCE_WORD.objectToWorldRow0;
        const row1 = base + INSTANCE_WORD.objectToWorldRow1;
        const row2 = base + INSTANCE_WORD.objectToWorldRow2;
        floats[row0] = matrix[m + 0];
        floats[row0 + 1] = matrix[m + 4];
        floats[row0 + 2] = matrix[m + 8];
        floats[row0 + 3] = matrix[m + 12];
        floats[row1] = matrix[m + 1];
        floats[row1 + 1] = matrix[m + 5];
        floats[row1 + 2] = matrix[m + 9];
        floats[row1 + 3] = matrix[m + 13];
        floats[row2] = matrix[m + 2];
        floats[row2 + 1] = matrix[m + 6];
        floats[row2 + 2] = matrix[m + 10];
        floats[row2 + 3] = matrix[m + 14];
        words[base + INSTANCE_WORD.materialListStart] = instance.listOffset;
        words[base + INSTANCE_WORD.materialListCount] = instance.materials.length;
        words[base + INSTANCE_WORD.reserved] = 0;

        // A slide keeps the frame's edges: only its corner moves (InstanceTransforms.slide).
        const moved = linearChanged ? transforms.place(slot) : transforms.slide(slot);
        const bounds = transforms.bounds;
        const at = slot * BOX_NUMBERS;

        for (let axis = 0; axis < 3; axis++) {
            floats[base + INSTANCE_WORD.boundsMin + axis] = bounds[at + axis];
            floats[base + INSTANCE_WORD.boundsMax + axis] = bounds[at + 3 + axis];
        }
        words[base + INSTANCE_WORD.mesh] = instance.mesh.slot;
        words[base + INSTANCE_WORD.flags] =
            INSTANCE_FLAG.live |
            (instance.contributesToGi ? INSTANCE_FLAG.contributesToGi : 0) |
            (instance.visible ? INSTANCE_FLAG.visible : 0) |
            transforms.flags[slot];
        return moved;
    }

    /** The record's normal matrix (the inverse's linear part), and whether the transform mirrors or is singular. */
    private writeNormalMatrix(slot: number, base: number): void {
        const floats = this.mirror.floats;
        // A singular transform has no normal matrix; identity keeps the accessors finite.
        const inverse = this.inverseScratch;
        const determinant = invertAffineInto(this.transforms.matrices, inverse, slot * MATRIX_NUMBERS);
        const invertible = determinant !== 0;
        if (!invertible) inverse.set(IDENTITY_ROWS);
        // The normal matrix's rows: the inverse's linear part (its rows 0-2, columns 0-2).
        const normal0 = base + INSTANCE_WORD.normalRow0;
        const normal1 = base + INSTANCE_WORD.normalRow1;
        const normal2 = base + INSTANCE_WORD.normalRow2;
        floats[normal0] = inverse[0];
        floats[normal0 + 1] = inverse[1];
        floats[normal0 + 2] = inverse[2];
        floats[normal1] = inverse[4];
        floats[normal1 + 1] = inverse[5];
        floats[normal1 + 2] = inverse[6];
        floats[normal2] = inverse[8];
        floats[normal2 + 1] = inverse[9];
        floats[normal2 + 2] = inverse[10];
        this.transforms.flags[slot] = (invertible && determinant < 0 ? INSTANCE_FLAG.mirrored : 0) | (invertible ? 0 : INSTANCE_FLAG.singular);
    }

    private writeMaterialRecord(material: MaterialEntry): void {
        const words = this.mirror.words;
        const floats = this.mirror.floats;
        const base = this.mirror.recordWord('materials', material.slot);
        const descriptor = material.descriptor;
        // White by default, as three.js materials are: albedo multiplies the texture.
        const albedo = descriptor.albedo ?? [1, 1, 1];
        const emissive = descriptor.emissive ?? [0, 0, 0];
        const emissiveIntensity = descriptor.emissiveIntensity ?? 1;
        const alphaMode = resolvedAlphaMode(descriptor);
        // The record's cutoff is 0 unless the mode is cutout, as before alpha modes (0 = no holes).
        const alphaCutoff = alphaMode === 'cutout' ? (descriptor.alphaCutoff ?? DEFAULT_ALPHA_CUTOFF) : 0;
        for (let channel = 0; channel < 3; channel++) {
            floats[base + MATERIAL_WORD.albedo + channel] = albedo[channel];
            floats[base + MATERIAL_WORD.emission + channel] = emissive[channel] * emissiveIntensity;
        }
        floats[base + MATERIAL_WORD.alphaCutoff] = alphaCutoff;
        words[base + MATERIAL_WORD.flags] =
            MATERIAL_FLAG.live |
            (descriptor.blocksLight !== false ? MATERIAL_FLAG.blocksLight : 0) |
            (descriptor.doubleSided !== false ? MATERIAL_FLAG.doubleSided : 0) |
            (alphaCutoff > 0 && material.albedoTexture ? MATERIAL_FLAG.alphaTest : 0);
        this.writeUvMapping(base + MATERIAL_WORD.albedoUvMatrix, base + MATERIAL_WORD.albedoUvTranslation, descriptor.albedoUv);
        words[base + MATERIAL_WORD.albedoTexture] = textureLayer(material.albedoTexture);
        words[base + MATERIAL_WORD.albedoUvPacked] = packUvMapping(descriptor.albedoUv);
        this.writeUvMapping(base + MATERIAL_WORD.emissiveUvMatrix, base + MATERIAL_WORD.emissiveUvTranslation, descriptor.emissiveUv);
        words[base + MATERIAL_WORD.emissiveTexture] = textureLayer(material.emissiveTexture);
        words[base + MATERIAL_WORD.emissiveUvPacked] = packUvMapping(descriptor.emissiveUv);
        // While a metalness map's layer is still being made, the material is not metal at all: glTF
        // materials put metalness 1 in the factor and the real values in the map, so the factor alone
        // would make every such material black (albedo x (1 - 1)) until its map lands.
        const metalnessMapPending = material.metalnessTexture !== undefined && material.metalnessTexture.state !== 'built';
        floats[base + MATERIAL_WORD.metalness] = metalnessMapPending ? 0 : (descriptor.metalness ?? 0);
        floats[base + MATERIAL_WORD.opacity] = descriptor.opacity ?? 1;
        words[base + MATERIAL_WORD.alphaMode] = ALPHA_MODE[alphaMode];
        words[base + MATERIAL_WORD.reserved] = 0;
        const metalnessMap = descriptor.metalnessTexture;
        this.writeUvMapping(base + MATERIAL_WORD.metalnessUvMatrix, base + MATERIAL_WORD.metalnessUvTranslation, metalnessMap?.uv);
        words[base + MATERIAL_WORD.metalnessTexture] = textureLayer(material.metalnessTexture);
        words[base + MATERIAL_WORD.metalnessUvPacked] = packUvMapping(metalnessMap?.uv) | (TEXTURE_CHANNEL[metalnessMap?.channel ?? 'b'] << UV_PACK_SHIFT.textureChannel);
        this.mirror.markRecord('materials', material.slot);
    }

    private writeUvMapping(matrixWord: number, translationWord: number, mapping: UvMapping | undefined): void {
        const matrix = mapping?.matrix ?? IDENTITY_UV;
        const floats = this.mirror.floats;
        for (let index = 0; index < 4; index++) floats[matrixWord + index] = matrix[index];
        floats[translationWord] = matrix[4];
        floats[translationWord + 1] = matrix[5];
    }

    /** Puts a light at `index` in the packed table and writes its record there. */
    private placeLight(light: LightEntry, index: number): void {
        this.mirror.ensureRecord('lights', index);
        this.lights[index] = light;
        light.index = index;
        this.writeLightRecord(light);
    }

    private writeLightRecord(light: LightEntry): void {
        const words = this.mirror.words;
        const floats = this.mirror.floats;
        const base = this.mirror.recordWord('lights', light.index);
        const descriptor = light.descriptor;
        words.fill(0, base, base + LIGHT_WORD.group + 1);
        words[base + LIGHT_WORD.type] = LIGHT_TYPE[descriptor.type];
        if (descriptor.type !== 'directional') floats.set(descriptor.position, base + LIGHT_WORD.position);
        if (descriptor.type !== 'point') floats.set(normalized(descriptor.direction, 'light direction'), base + LIGHT_WORD.direction);
        floats[base + LIGHT_WORD.range] = descriptor.type === 'directional' ? 0 : descriptor.range;
        for (let channel = 0; channel < 3; channel++) floats[base + LIGHT_WORD.intensity + channel] = descriptor.color[channel] * descriptor.intensity;
        words[base + LIGHT_WORD.flags] = descriptor.castsShadows !== false ? LIGHT_FLAG.castsShadows : 0;
        // Point and directional lights shine in every direction: cos = -1 puts every angle inside the cone.
        floats[base + LIGHT_WORD.cosInnerCone] = descriptor.type === 'spot' ? Math.cos(descriptor.innerConeAngle) : -1;
        floats[base + LIGHT_WORD.cosOuterCone] = descriptor.type === 'spot' ? Math.cos(descriptor.outerConeAngle) : -1;
        floats[base + LIGHT_WORD.radius] = descriptor.type === 'directional' ? (descriptor.angularRadius ?? 0) : (descriptor.radius ?? 0);
        words[base + LIGHT_WORD.group] = descriptor.group ?? 0;
        this.mirror.markRecord('lights', light.index);
    }

    // -----------------------------------------------------------------------
    // Checks and lookups
    // -----------------------------------------------------------------------

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/ddgi: this scene was disposed.');
    }

    private name(kind: string, handle: number, label: string | undefined): string {
        return label ? `${kind} ${handle} "${label}"` : `${kind} ${handle}`;
    }

    private lookup<Entry>(map: Map<number, Entry>, handle: number, kind: string): Entry {
        const entry = map.get(handle);
        if (!entry) throw new Error(`@pmndrs/ddgi: unknown or destroyed ${kind} handle ${handle}.`);
        return entry;
    }

    private mesh(handle: MeshHandle): MeshEntry {
        return this.lookup(this.meshes, handle, 'mesh');
    }

    private texture(handle: TextureHandle): TextureEntry {
        return this.lookup(this.textures, handle, 'texture');
    }

    private material(handle: MaterialHandle): MaterialEntry {
        return this.lookup(this.materials, handle, 'material');
    }

    private instance(handle: InstanceHandle): InstanceEntry {
        return this.lookup(this.instances, handle, 'instance');
    }

    private light(handle: LightHandle): LightEntry {
        return this.lookup(this.lightsByHandle, handle, 'light');
    }

    private isTraceable(instance: InstanceEntry): boolean {
        return instance.contributesToGi && instance.visible;
    }

    /**
     * Whether an instance gives off light (`DirtyRegion.emits`): one of its
     * materials has a non-black emissive colour and a positive intensity
     * (emission is emissive × intensity × texture, and emissive defaults to
     * black). Read from the materials as they are now, so a material edited
     * to glow counts from its next region on.
     */
    private emits(instance: InstanceEntry): boolean {
        return instance.materials.some(({ descriptor }) => {
            const emissive = descriptor.emissive;
            const intensity = descriptor.emissiveIntensity ?? 1;
            return emissive !== undefined && intensity > 0 && (emissive[0] > 0 || emissive[1] > 0 || emissive[2] > 0);
        });
    }

    /** Shown, but left out of rays: it only receives GI (`SceneChanges.receivers`). */
    private isReceiver(instance: InstanceEntry): boolean {
        return !instance.contributesToGi && instance.visible;
    }

    private materialList(handles: readonly MaterialHandle[], mesh: MeshEntry): MaterialEntry[] {
        if (handles.length < mesh.layout.materialSlotCount) {
            throw new RangeError(
                `@pmndrs/ddgi: ${this.name('mesh', mesh.handle, mesh.label)} uses ${mesh.layout.materialSlotCount} material slot(s), but the instance gives ${handles.length} material(s).`,
            );
        }
        return handles.map((handle) => this.material(handle));
    }

    /**
     * A checked copy of a host's transform, in a scratch the caller copies
     * from at once (into InstanceTransforms), so a call allocates nothing.
     */
    private checkedTransform(transform: Mat4): Float64Array {
        if (!transform || transform.length < 16) throw new RangeError('@pmndrs/ddgi: a transform needs 16 numbers (column-major 4x4).');
        const matrix = this.transformScratch;
        for (let index = 0; index < 16; index++) {
            if (!Number.isFinite(transform[index])) throw new RangeError('@pmndrs/ddgi: a transform must be finite.');
            matrix[index] = transform[index];
        }
        return matrix;
    }

    private checkMaterial(descriptor: MaterialDescriptor): void {
        checkColor(descriptor.albedo, 'albedo');
        checkColor(descriptor.emissive, 'emissive');
        checkNonNegative(descriptor.emissiveIntensity, 'emissiveIntensity');
        checkUnitInterval(descriptor.alphaCutoff, 'alphaCutoff');
        checkUnitInterval(descriptor.metalness, 'metalness');
        checkUnitInterval(descriptor.opacity, 'opacity');
        if (descriptor.alphaMode !== undefined && !(descriptor.alphaMode in ALPHA_MODE)) throw new RangeError(`alphaMode must be opaque, cutout or stochastic, got ${descriptor.alphaMode}`);
        const metalnessMap = descriptor.metalnessTexture;
        if (metalnessMap !== undefined && !(metalnessMap.channel in TEXTURE_CHANNEL)) throw new RangeError(`metalnessTexture.channel must be r, g, b or a, got ${metalnessMap.channel}`);
        const mappings = [
            ['albedoUv', descriptor.albedoUv],
            ['emissiveUv', descriptor.emissiveUv],
            ['metalnessTexture.uv', metalnessMap?.uv],
        ] as const;
        for (const [name, mapping] of mappings) {
            if (!mapping) continue;
            if (mapping.matrix !== undefined && !isFiniteVector(mapping.matrix, 6)) throw new RangeError(`${name}.matrix must be six finite numbers`);
            if (mapping.channel !== undefined && mapping.channel !== 0 && mapping.channel !== 1) throw new RangeError(`${name}.channel must be 0 or 1`);
            for (const wrap of [mapping.wrapU, mapping.wrapV]) {
                if (wrap !== undefined && !(wrap in UV_WRAP)) throw new RangeError(`${name} wrap must be repeat, clamp or mirror`);
            }
        }
        if (descriptor.albedoTexture !== undefined) this.texture(descriptor.albedoTexture);
        if (descriptor.emissiveTexture !== undefined) this.texture(descriptor.emissiveTexture);
        if (metalnessMap !== undefined) this.texture(metalnessMap.texture);
    }

    private attachTextures(material: MaterialEntry): void {
        const { albedoTexture, emissiveTexture, metalnessTexture } = material.descriptor;
        material.albedoTexture = albedoTexture !== undefined ? this.texture(albedoTexture) : undefined;
        material.emissiveTexture = emissiveTexture !== undefined ? this.texture(emissiveTexture) : undefined;
        material.metalnessTexture = metalnessTexture !== undefined ? this.texture(metalnessTexture.texture) : undefined;
        for (const texture of [material.albedoTexture, material.emissiveTexture, material.metalnessTexture]) if (texture) texture.users++;
    }

    private detachTextures(material: MaterialEntry): void {
        for (const texture of [material.albedoTexture, material.emissiveTexture, material.metalnessTexture]) if (texture) texture.users--;
        material.albedoTexture = undefined;
        material.emissiveTexture = undefined;
        material.metalnessTexture = undefined;
    }

    /** A checked copy of a light descriptor. Throws on anything the shaders couldn't use. */
    private checkedLight(descriptor: LightDescriptor): LightDescriptor {
        checkColor(descriptor.color, 'light color');
        if (descriptor.color === undefined) throw new RangeError('a light needs a color');
        checkNonNegative(descriptor.intensity, 'light intensity');
        if (descriptor.intensity === undefined) throw new RangeError('a light needs an intensity');
        const group = descriptor.group ?? 0;
        if (!Number.isInteger(group) || group < 0 || group >= LIGHT_GROUP_COUNT) {
            throw new RangeError(`light group must be an integer from 0 to ${LIGHT_GROUP_COUNT - 1}, got ${group}`);
        }
        if (descriptor.type === 'directional') {
            normalized(descriptor.direction, 'light direction');
            checkNonNegative(descriptor.angularRadius, 'angularRadius');
        } else if (descriptor.type === 'point' || descriptor.type === 'spot') {
            if (!isFiniteVector(descriptor.position, 3)) throw new RangeError('light position must be three finite numbers');
            if (!(Number.isFinite(descriptor.range) && descriptor.range >= 0)) {
                throw new RangeError(`light range must be a finite number >= 0 (0 means no limit), got ${descriptor.range}`);
            }
            checkNonNegative(descriptor.radius, 'light radius');
            if (descriptor.type === 'spot') {
                normalized(descriptor.direction, 'light direction');
                const { innerConeAngle, outerConeAngle } = descriptor;
                if (!(innerConeAngle >= 0 && innerConeAngle <= outerConeAngle && outerConeAngle <= Math.PI / 2)) {
                    throw new RangeError('spot cone angles must satisfy 0 <= innerConeAngle <= outerConeAngle <= pi / 2');
                }
            }
        } else {
            throw new TypeError(`@pmndrs/ddgi: unknown light type ${(descriptor as { type: string }).type}`);
        }
        // Its own arrays: the change log compares states, and a host editing a vector in place would change the stored one too.
        return copyLight(descriptor);
    }
}

/** A material texture word: the texture's store layer once it is made, else NONE (the material shades as if it were white). */
function textureLayer(texture: TextureEntry | undefined): number {
    return texture !== undefined && texture.state === 'built' ? texture.slot : NONE;
}

/** The alpha mode a material has: its own, else cutout when it names a cutoff above 0 (as before alpha modes), else opaque. */
function resolvedAlphaMode(descriptor: MaterialDescriptor): AlphaMode {
    if (descriptor.alphaMode !== undefined) return descriptor.alphaMode;
    return (descriptor.alphaCutoff ?? 0) > 0 ? 'cutout' : 'opaque';
}

function checkUnitInterval(value: number | undefined, what: string): void {
    if (value !== undefined && !(value >= 0 && value <= 1)) throw new RangeError(`${what} must be in [0, 1], got ${value}`);
}

function packUvMapping(mapping: UvMapping | undefined): number {
    return (
        ((mapping?.channel ?? 0) << UV_PACK_SHIFT.channel) |
        (UV_WRAP[mapping?.wrapU ?? 'repeat'] << UV_PACK_SHIFT.wrapU) |
        (UV_WRAP[mapping?.wrapV ?? 'repeat'] << UV_PACK_SHIFT.wrapV)
    );
}
