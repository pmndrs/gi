/**
 * The platform's scene: the host-facing `SceneApi` and the technique-facing
 * `SceneCopy`, over two GPU storage buffers (see layout.ts).
 *
 * Calls from the host only change CPU state: meshes are checked and packed,
 * records are written into a CPU mirror of the table buffer, and what moved
 * is logged. Nothing touches the GPU until {@link PlatformScene.encode},
 * which the platform calls at the start of its own `encode`.
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
import { GILimitError } from '../contract/errors.ts';
import type { InstanceHandle, LightHandle, MaterialHandle, MeshHandle, TextureHandle } from '../contract/handles.ts';
import type { Aabb, ColorRGB, Mat4, Vec3 } from '../contract/math.ts';
import type {
    DeformedPositions,
    EnvironmentDescriptor,
    InstanceDescriptor,
    LightChanges,
    LightDescriptor,
    MaterialDescriptor,
    MeshDescriptor,
    SceneApi,
    TextureDescriptor,
    UvMapping,
} from '../contract/scene.ts';
import type { SceneChanges, SceneCopy } from '../contract/scene-copy.ts';
import type { BindingSlot, WgslModule } from '../contract/wgsl.ts';
import { invertAffine, transformBounds } from '../trace/top-level.ts';
import { ChangeLog, toAabb } from './change-log.ts';
import {
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
    MATERIAL_WORD,
    MESH_FLAG,
    MESH_WORD,
    NONE,
    UV_PACK_SHIFT,
    UV_WRAP,
} from './layout.ts';
import { packMesh, type PackedMesh } from './mesh-packing.ts';
import { RangeAllocator } from './range-allocator.ts';
import { SlotAllocator } from './slot-allocator.ts';
import { TableMirror } from './table-mirror.ts';
import { SCENE_MODULE_ENTRIES, sceneModuleWgsl } from './wgsl.ts';

/** Starting sizes. Every table doubles when it fills; none of this is a hard limit. */
export interface SceneCapacities {
    readonly instances: number;
    readonly meshes: number;
    readonly materials: number;
    /** 32 is the light count hit shading is designed to keep flat cost up to. */
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
    lights: 32,
    materialListWords: 256,
    vertexWords: 256 * 1024,
};

export interface PlatformSceneOptions {
    /** Prefix for GPU buffer labels. */
    readonly label?: string;
    readonly initialCapacities?: Partial<SceneCapacities>;
}

/** A mesh whose geometry went to the GPU in this `encode`, for builders that need it (the tracer's BVH). */
export interface UploadedMesh {
    readonly slot: number;
    readonly handle: MeshHandle;
    /** The packed mesh, words included. Read positions and indices with `packedPositions` / `packedIndices`. */
    readonly packed: PackedMesh;
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
    /** Bytes written with `queue.writeBuffer` this frame. */
    readonly bytesWritten: number;
}

/** GPU memory of the scene copy, in bytes. */
export interface SceneMemory {
    readonly vertexBuffer: number;
    /** Of `vertexBuffer`, the bytes holding live meshes. */
    readonly vertexBytesUsed: number;
    readonly tableBuffer: number;
    readonly total: number;
}

interface MeshEntry {
    readonly handle: MeshHandle;
    readonly slot: number;
    label: string | undefined;
    packed: PackedMesh;
    /** Words the mesh occupies; kept apart from `packed.words`, which is dropped after upload. */
    wordCount: number;
    /** Word offset in the vertex buffer, or -1 until uploaded. */
    vertexOffset: number;
    readonly instances: Set<InstanceEntry>;
}

interface TextureEntry {
    readonly handle: TextureHandle;
    readonly slot: number;
    readonly descriptor: TextureDescriptor;
    users: number;
}

interface MaterialEntry {
    readonly handle: MaterialHandle;
    readonly slot: number;
    descriptor: MaterialDescriptor;
    albedoTexture: TextureEntry | undefined;
    emissiveTexture: TextureEntry | undefined;
    readonly users: Set<InstanceEntry>;
}

interface InstanceEntry {
    readonly handle: InstanceHandle;
    readonly slot: number;
    label: string | undefined;
    mesh: MeshEntry;
    materials: MaterialEntry[];
    readonly objectToWorld: Float64Array;
    contributesToGi: boolean;
    visible: boolean;
    /** Word offset of the material list inside the list section. */
    listOffset: number;
    /** World box `[min, max]`, padded like the tracer's. */
    readonly worldBounds: Float32Array;
}

interface LightEntry {
    readonly handle: LightHandle;
    /** Position in the packed light table. */
    index: number;
    descriptor: LightDescriptor;
}

const IDENTITY_UV: readonly number[] = [1, 0, 0, 1, 0, 0];

const EMPTY_WORDS: Uint32Array<ArrayBuffer> = new Uint32Array(0);

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

    private readonly pendingMeshes = new Set<MeshEntry>();
    private removedMeshSlots: number[] = [];
    private readonly changeLog = new ChangeLog();
    private headerDirty = true;
    private topLevelDirty = false;
    private instanceSetDirty = false;
    private triangleTotal = 0;
    private cachedBounds: Aabb | null = null;

    constructor(device: GPUDevice, options: PlatformSceneOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'gi';
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

    get memoryBytes(): number {
        return this.memory.total;
    }

    get memory(): SceneMemory {
        const vertexBuffer = this.vertexBuffer.size;
        const tableBuffer = this.tableBuffer.size;
        return { vertexBuffer, vertexBytesUsed: this.vertexAllocator.used * 4, tableBuffer, total: vertexBuffer + tableBuffer };
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
        for (const instance of this.instances.values()) {
            if (!instance.contributesToGi || !instance.visible) continue;
            for (let axis = 0; axis < 3; axis++) {
                box[axis] = Math.min(box[axis], instance.worldBounds[axis]);
                box[axis + 3] = Math.max(box[axis + 3], instance.worldBounds[axis + 3]);
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

    /** Index of a light in the packed light table. Changes when other lights are added or removed. */
    lightIndex(light: LightHandle): number {
        return this.light(light).index;
    }

    /** Every live instance, for building the tracer's top level. */
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
            packed.objectToWorld.set(instance.objectToWorld, index * 16);
            index++;
        }
        return packed;
    }

    instanceSnapshot(): InstanceSnapshot[] {
        return [...this.instances.values()].map((instance) => ({
            slot: instance.slot,
            handle: instance.handle,
            meshSlot: instance.mesh.slot,
            objectToWorld: instance.objectToWorld,
            contributesToGi: instance.contributesToGi,
            visible: instance.visible,
        }));
    }

    // -----------------------------------------------------------------------
    // Meshes
    // -----------------------------------------------------------------------

    createMesh(descriptor: MeshDescriptor): MeshHandle {
        this.checkLive();
        const packed = packMesh(descriptor);
        const handle = this.nextHandle++ as MeshHandle;
        const slot = this.meshSlots.take();
        this.mirror.ensureRecord('meshes', slot);
        const entry: MeshEntry = {
            handle,
            slot,
            label: descriptor.label,
            packed,
            wordCount: packed.words.length,
            vertexOffset: -1,
            instances: new Set(),
        };
        this.meshes.set(handle, entry);
        this.pendingMeshes.add(entry);
        this.triangleTotal += packed.triangleCount;
        this.headerDirty = true;
        return handle;
    }

    updateMesh(mesh: MeshHandle, descriptor: MeshDescriptor): void {
        this.checkLive();
        const entry = this.mesh(mesh);
        const packed = packMesh(descriptor);
        this.releaseVertexRange(entry);
        this.triangleTotal += packed.triangleCount - entry.packed.triangleCount;
        entry.packed = packed;
        entry.wordCount = packed.words.length;
        entry.label = descriptor.label ?? entry.label;
        this.pendingMeshes.add(entry);
        for (const instance of entry.instances) {
            const traceable = this.isTraceable(instance);
            if (traceable) this.changeLog.addRegion(instance.worldBounds, 'geometry', instance.handle);
            this.writeInstanceRecord(instance);
            if (traceable) {
                this.changeLog.addRegion(instance.worldBounds, 'geometry', instance.handle);
                this.topLevelDirty = true;
            }
        }
        this.cachedBounds = null;
    }

    setDeformedPositions(mesh: MeshHandle, _positions: DeformedPositions): void {
        this.checkLive();
        this.mesh(mesh);
        throw new Error('@pmndrs/ddgi: setDeformedPositions is not supported yet (deforming meshes arrive in M3).');
    }

    destroyMesh(mesh: MeshHandle): void {
        this.checkLive();
        const entry = this.mesh(mesh);
        if (entry.instances.size > 0) {
            throw new Error(`@pmndrs/ddgi: ${this.name('mesh', entry.handle, entry.label)} is still used by ${entry.instances.size} instance(s); destroy them first.`);
        }
        this.releaseVertexRange(entry);
        this.pendingMeshes.delete(entry);
        this.mirror.clearRecord('meshes', entry.slot);
        this.meshSlots.release(entry.slot);
        this.removedMeshSlots.push(entry.slot);
        this.triangleTotal -= entry.packed.triangleCount;
        this.meshes.delete(mesh);
        this.headerDirty = true;
    }

    // -----------------------------------------------------------------------
    // Textures and materials
    // -----------------------------------------------------------------------

    /**
     * Registers a texture. M1 keeps only the handle and the descriptor (a
     * reference to the source, not a copy); the atlas copy arrives in M4.
     */
    createTexture(descriptor: TextureDescriptor): TextureHandle {
        this.checkLive();
        if (!descriptor || !descriptor.source) throw new TypeError('@pmndrs/ddgi: a texture needs a source');
        const handle = this.nextHandle++ as TextureHandle;
        this.textures.set(handle, { handle, slot: this.textureSlots.take(), descriptor, users: 0 });
        return handle;
    }

    destroyTexture(texture: TextureHandle): void {
        this.checkLive();
        const entry = this.texture(texture);
        if (entry.users > 0) throw new Error(`@pmndrs/ddgi: texture ${texture} is still used by ${entry.users} material(s).`);
        this.textureSlots.release(entry.slot);
        this.textures.delete(texture);
    }

    createMaterial(descriptor: MaterialDescriptor): MaterialHandle {
        this.checkLive();
        this.checkMaterial(descriptor);
        const handle = this.nextHandle++ as MaterialHandle;
        const slot = this.materialSlots.take();
        this.mirror.ensureRecord('materials', slot);
        const entry: MaterialEntry = { handle, slot, descriptor: { ...descriptor }, albedoTexture: undefined, emissiveTexture: undefined, users: new Set() };
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
            if (this.isTraceable(instance)) this.changeLog.addRegion(instance.worldBounds, 'material', instance.handle);
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
            objectToWorld,
            contributesToGi: descriptor.contributesToGI !== false,
            visible: descriptor.visible !== false,
            listOffset: this.mirror.allocateList(materials.length),
            worldBounds: new Float32Array(6),
        };
        this.instances.set(handle, entry);
        mesh.instances.add(entry);
        for (const material of materials) material.users.add(entry);
        this.mirror.writeList(entry.listOffset, materials.map((material) => material.slot));
        this.writeInstanceRecord(entry);
        if (this.isTraceable(entry)) {
            this.changeLog.addRegion(entry.worldBounds, 'geometry', handle);
            this.topLevelDirty = true;
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
        const traceable = this.isTraceable(entry);
        if (traceable) this.changeLog.addRegion(entry.worldBounds, 'geometry', instance);
        entry.objectToWorld.set(objectToWorld);
        this.writeInstanceRecord(entry);
        if (traceable) {
            this.changeLog.addRegion(entry.worldBounds, 'geometry', instance);
            this.topLevelDirty = true;
            this.cachedBounds = null;
        } else {
            this.changeLog.addMovedReceiver();
        }
    }

    setVisible(instance: InstanceHandle, visible: boolean): void {
        this.checkLive();
        const entry = this.instance(instance);
        if (entry.visible === visible) return;
        entry.visible = visible;
        this.writeInstanceRecord(entry);
        if (entry.contributesToGi) {
            this.changeLog.addRegion(entry.worldBounds, 'geometry', instance);
            this.topLevelDirty = true;
            this.cachedBounds = null;
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
        if (this.isTraceable(entry)) this.changeLog.addRegion(entry.worldBounds, 'material', instance);
    }

    destroyInstance(instance: InstanceHandle): void {
        this.checkLive();
        const entry = this.instance(instance);
        if (this.isTraceable(entry)) {
            this.changeLog.addRegion(entry.worldBounds, 'geometry', instance);
            this.topLevelDirty = true;
        }
        if (entry.contributesToGi) this.instanceSetDirty = true;
        entry.mesh.instances.delete(entry);
        for (const material of entry.materials) material.users.delete(entry);
        this.mirror.releaseList(entry.listOffset, entry.materials.length);
        this.mirror.clearRecord('instances', entry.slot);
        this.instanceSlots.release(entry.slot);
        this.instances.delete(instance);
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
        this.logLight(entry);
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
        this.logLight(entry);
        if ((next.group ?? 0) !== (entry.descriptor.group ?? 0)) this.changeLog.addGlobal('lights');
        entry.descriptor = next;
        this.writeLightRecord(entry);
        this.logLight(entry);
    }

    destroyLight(light: LightHandle): void {
        this.checkLive();
        const entry = this.light(light);
        this.logLight(entry);
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
    encode(encoder: GPUCommandEncoder): SceneUpload {
        this.checkLive();
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];

        let bytesWritten = 0;
        const versionBefore = this.bufferVersion;
        const uploaded = this.uploadMeshes(encoder);
        for (const mesh of uploaded) bytesWritten += mesh.packed.words.byteLength;
        bytesWritten += this.uploadTables();

        const upload: SceneUpload = {
            changes: this.changeLog.take(),
            meshesUploaded: uploaded,
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
    }

    // -----------------------------------------------------------------------
    // Uploads
    // -----------------------------------------------------------------------

    private uploadMeshes(encoder: GPUCommandEncoder): UploadedMesh[] {
        if (this.pendingMeshes.size === 0) return [];
        const pending = [...this.pendingMeshes];
        this.pendingMeshes.clear();

        const unplaced: MeshEntry[] = [];
        for (const mesh of pending) {
            const offset = this.vertexAllocator.allocate(mesh.wordCount);
            if (offset === null) unplaced.push(mesh);
            else mesh.vertexOffset = offset;
        }
        if (unplaced.length > 0) {
            const wordsNeeded = unplaced.reduce((sum, mesh) => sum + mesh.wordCount, 0);
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
                const offset = this.vertexAllocator.allocate(mesh.wordCount);
                if (offset === null) throw new Error('vertex allocation failed after growing the buffer');
                mesh.vertexOffset = offset;
            }
        }

        const uploaded: UploadedMesh[] = [];
        for (const mesh of pending) {
            this.device.queue.writeBuffer(this.vertexBuffer, mesh.vertexOffset * 4, mesh.packed.words);
            this.writeMeshRecord(mesh);
            uploaded.push({ slot: mesh.slot, handle: mesh.handle, packed: mesh.packed });
            // The GPU has the words now; keep only the layout.
            mesh.packed = { ...mesh.packed, words: EMPTY_WORDS };
        }
        return uploaded;
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
                runEnd += mesh.wordCount;
            } else {
                flush();
                runStart = mesh.vertexOffset;
                runEnd = mesh.vertexOffset + mesh.wordCount;
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
        this.vertexAllocator.release(mesh.vertexOffset, mesh.wordCount);
        mesh.vertexOffset = -1;
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
        const packed = mesh.packed;
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

    /** Rewrites an instance's record and updates its world bounds. */
    private writeInstanceRecord(instance: InstanceEntry): void {
        const words = this.mirror.words;
        const floats = this.mirror.floats;
        const base = this.mirror.recordWord('instances', instance.slot);
        const matrix = instance.objectToWorld;
        const rowWords = [INSTANCE_WORD.objectToWorldRow0, INSTANCE_WORD.objectToWorldRow1, INSTANCE_WORD.objectToWorldRow2];
        const normalWords = [INSTANCE_WORD.normalRow0, INSTANCE_WORD.normalRow1, INSTANCE_WORD.normalRow2];
        const inverse = invertAffine(matrix);
        for (let row = 0; row < 3; row++) {
            // Column-major input: row r of the 3x4 matrix is elements r, 4 + r, 8 + r, 12 + r.
            const at = base + rowWords[row];
            floats[at] = matrix[row];
            floats[at + 1] = matrix[4 + row];
            floats[at + 2] = matrix[8 + row];
            floats[at + 3] = matrix[12 + row];
            // A singular transform has no normal matrix; identity keeps the accessors finite.
            const normalAt = base + normalWords[row];
            for (let column = 0; column < 3; column++) {
                floats[normalAt + column] = inverse ? inverse.rows[row * 4 + column] : row === column ? 1 : 0;
            }
        }
        words[base + INSTANCE_WORD.materialListStart] = instance.listOffset;
        words[base + INSTANCE_WORD.materialListCount] = instance.materials.length;
        words[base + INSTANCE_WORD.reserved] = 0;

        transformBounds(instance.mesh.packed.bounds, matrix, instance.worldBounds, 0);
        floats.set(instance.worldBounds.subarray(0, 3), base + INSTANCE_WORD.boundsMin);
        words[base + INSTANCE_WORD.mesh] = instance.mesh.slot;
        floats.set(instance.worldBounds.subarray(3, 6), base + INSTANCE_WORD.boundsMax);
        words[base + INSTANCE_WORD.flags] =
            INSTANCE_FLAG.live |
            (instance.contributesToGi ? INSTANCE_FLAG.contributesToGi : 0) |
            (instance.visible ? INSTANCE_FLAG.visible : 0) |
            (inverse && inverse.determinant < 0 ? INSTANCE_FLAG.mirrored : 0) |
            (inverse ? 0 : INSTANCE_FLAG.singular);
        this.mirror.markRecord('instances', instance.slot);
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
        const alphaCutoff = descriptor.alphaCutoff ?? 0;
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
        words[base + MATERIAL_WORD.albedoTexture] = material.albedoTexture ? material.albedoTexture.slot : NONE;
        words[base + MATERIAL_WORD.albedoUvPacked] = packUvMapping(descriptor.albedoUv);
        this.writeUvMapping(base + MATERIAL_WORD.emissiveUvMatrix, base + MATERIAL_WORD.emissiveUvTranslation, descriptor.emissiveUv);
        words[base + MATERIAL_WORD.emissiveTexture] = material.emissiveTexture ? material.emissiveTexture.slot : NONE;
        words[base + MATERIAL_WORD.emissiveUvPacked] = packUvMapping(descriptor.emissiveUv);
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

    private materialList(handles: readonly MaterialHandle[], mesh: MeshEntry): MaterialEntry[] {
        if (handles.length < mesh.packed.materialSlotCount) {
            throw new RangeError(
                `@pmndrs/ddgi: ${this.name('mesh', mesh.handle, mesh.label)} uses ${mesh.packed.materialSlotCount} material slot(s), but the instance gives ${handles.length} material(s).`,
            );
        }
        return handles.map((handle) => this.material(handle));
    }

    private checkedTransform(transform: Mat4): Float64Array {
        if (!transform || transform.length < 16) throw new RangeError('@pmndrs/ddgi: a transform needs 16 numbers (column-major 4x4).');
        const matrix = new Float64Array(16);
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
        const cutoff = descriptor.alphaCutoff;
        if (cutoff !== undefined && !(cutoff >= 0 && cutoff <= 1)) throw new RangeError(`alphaCutoff must be in [0, 1], got ${cutoff}`);
        for (const [name, mapping] of [['albedoUv', descriptor.albedoUv], ['emissiveUv', descriptor.emissiveUv]] as const) {
            if (!mapping) continue;
            if (mapping.matrix !== undefined && !isFiniteVector(mapping.matrix, 6)) throw new RangeError(`${name}.matrix must be six finite numbers`);
            if (mapping.channel !== undefined && mapping.channel !== 0 && mapping.channel !== 1) throw new RangeError(`${name}.channel must be 0 or 1`);
            for (const wrap of [mapping.wrapU, mapping.wrapV]) {
                if (wrap !== undefined && !(wrap in UV_WRAP)) throw new RangeError(`${name} wrap must be repeat, clamp or mirror`);
            }
        }
        if (descriptor.albedoTexture !== undefined) this.texture(descriptor.albedoTexture);
        if (descriptor.emissiveTexture !== undefined) this.texture(descriptor.emissiveTexture);
    }

    private attachTextures(material: MaterialEntry): void {
        const { albedoTexture, emissiveTexture } = material.descriptor;
        material.albedoTexture = albedoTexture !== undefined ? this.texture(albedoTexture) : undefined;
        material.emissiveTexture = emissiveTexture !== undefined ? this.texture(emissiveTexture) : undefined;
        if (material.albedoTexture) material.albedoTexture.users++;
        if (material.emissiveTexture) material.emissiveTexture.users++;
    }

    private detachTextures(material: MaterialEntry): void {
        if (material.albedoTexture) material.albedoTexture.users--;
        if (material.emissiveTexture) material.emissiveTexture.users--;
        material.albedoTexture = undefined;
        material.emissiveTexture = undefined;
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
        return { ...descriptor };
    }

    /** Logs where a light's influence is: a box for ranged lights, a global change otherwise. */
    private logLight(light: LightEntry): void {
        const descriptor = light.descriptor;
        if (descriptor.type === 'directional') {
            this.changeLog.addGlobal('sun');
        } else if (descriptor.range === 0) {
            this.changeLog.addGlobal('lights');
        } else {
            const [x, y, z] = descriptor.position;
            const range = descriptor.range;
            this.changeLog.addRegion([x - range, y - range, z - range, x + range, y + range, z + range], 'light', light.handle);
        }
    }
}

function packUvMapping(mapping: UvMapping | undefined): number {
    return (
        ((mapping?.channel ?? 0) << UV_PACK_SHIFT.channel) |
        (UV_WRAP[mapping?.wrapU ?? 'repeat'] << UV_PACK_SHIFT.wrapU) |
        (UV_WRAP[mapping?.wrapV ?? 'repeat'] << UV_PACK_SHIFT.wrapV)
    );
}
