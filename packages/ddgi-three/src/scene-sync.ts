/**
 * Mirrors a three.js scene into the platform's scene API (`gi.scene`), and
 * each frame sends only what changed.
 *
 * What it mirrors (conversions in convert.ts):
 * - **Meshes:** one platform mesh per `BufferGeometry` (shared geometries stay
 *   shared), with positions, indices (made up for non-indexed geometry),
 *   vertex normals, uv, uv1 (only when a material samples channel 1) and
 *   groups.
 * - **Materials:** one platform material per three material; textures as
 *   handles, one per three texture.
 * - **Instances:** one per `Mesh`, and one per instance of an
 *   `InstancedMesh` (its world matrix times the instance matrix; instances
 *   past `count` are hidden). Visibility includes the parents', as three
 *   draws it. `userData.giContributes` says whether an object takes part in GI
 *   rays (user-data.ts).
 * - **Lights:** directional, point and spot, with `userData.giLightGroup`.
 *   Hidden lights are sent with intensity 0.
 * - **Environment:** environment.ts, fed with the `SkyMesh` and the sun this
 *   walk finds.
 *
 * Change tracking:
 * - world matrices and visibility every frame, compared with the last ones
 *   sent (three keeps no version for them), so a still scene sends no
 *   transforms;
 * - each geometry's attribute versions and each material's GI-relevant
 *   values ({@link MaterialSnapshot}; three bumps `material.version` only on
 *   `needsUpdate`), each once every `editCheckInterval` frames
 *   ({@link DEFAULT_EDIT_CHECK_INTERVAL} says why);
 * - light descriptors field by field;
 * - added and removed objects, by a frame stamp.
 *
 * Not mirrored, with a one-time warning: `BatchedMesh`, ambient, hemisphere,
 * light-probe and rect-area lights (the last three also add diffuse light on
 * top of GI in three), point and spot lights whose `decay` isn't 2.
 * Morph targets and skinning use the bind pose (deforming meshes are M3).
 *
 * It reads world matrices as they are: `ThreeGI.update` leaves them as
 * three's last render computed them (three-gi.ts says why).
 */

import { Matrix4, type BufferAttribute, type BufferGeometry, type DirectionalLight, type InstancedMesh, type InterleavedBufferAttribute, type Light, type Material, type Mesh, type Object3D, type PointLight, type Scene, type Texture } from 'three';
import type { EnvironmentDescriptor, InstanceHandle, LightChanges, LightDescriptor, LightHandle, MaterialHandle, MeshHandle, SceneApi, TextureHandle } from '@pmndrs/ddgi-core';
import { MaterialSnapshot, PHYSICAL_DECAY, isSrgbTexture, lightDescriptor, luminance, materialDescriptor, materialSlotCount, meshDescriptor, textureSource, usesUvChannel1 } from './convert.ts';
import { EnvironmentSync, type PmndrsSkyLike, type SkyMeshLike, type Warn } from './environment.ts';
import { contributesFlag, contributesToGI } from './user-data.ts';

/** Floats in a 4x4 matrix. */
const MATRIX_FLOATS = 16;

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
    /** Which frame of the edit-check cycle this record is checked on. */
    readonly checkPhase: number;
}

interface MaterialRecord {
    readonly material: Material;
    handle: MaterialHandle;
    users: number;
    snapshot: MaterialSnapshot;
    scratch: MaterialSnapshot;
    /** Texture handles the platform material currently uses. */
    textureHandles: TextureHandle[];
    /** Which frame of the edit-check cycle this record is checked on. */
    readonly checkPhase: number;
}

interface TextureRecord {
    readonly texture: Texture;
    readonly handle: TextureHandle;
    readonly version: number;
}

interface ObjectRecord {
    readonly object: Mesh;
    readonly geometry: GeometryRecord;
    /** `geometry.geometry`, kept here so the per-frame identity check reads one record, not two. */
    readonly geometryObject: BufferGeometry;
    materials: MaterialRecord[];
    /** `mesh.material` when the material list was last checked: unchanged identity means unchanged list (arrays are also re-read on the edit-check frame). */
    materialKey: Material | Material[];
    /** Which frame of the edit-check cycle the object's GI flag (and a material array's contents) is checked on. */
    readonly checkPhase: number;
    readonly instances: InstanceHandle[];
    /** InstancedMesh: world matrices last sent, 16 per instance. (A plain mesh's is in the shared pool, at `matrixSlot`.) */
    matrices: Float64Array;
    /** Plain mesh: where its last-sent world matrix sits in the shared pool. -1 for an InstancedMesh. */
    matrixSlot: number;
    /** Per instance: last visibility sent. */
    readonly visible: boolean[];
    readonly contributes: boolean;
    /** Fixed facts about the object, read once (a missing flag costs a prototype walk every frame). */
    readonly instanced: boolean;
    readonly skinned: boolean;
    /** InstancedMesh: the mesh's own world matrix and the instance-matrix version the matrices were made from. */
    readonly parentMatrix: Float64Array;
    instanceMatrixVersion: number;
    seenFrame: number;
    /** Set when its platform objects are destroyed, so a stale entry in the walk order is never used. */
    destroyed: boolean;
}

interface LightRecord {
    readonly light: Light;
    readonly handle: LightHandle;
    descriptor: LightDescriptor;
    seenFrame: number;
}

/** What the last `update` sent, for stats and tests. */
export interface SyncFrameReport {
    created: number;
    destroyed: number;
    transforms: number;
    visibility: number;
    materials: number;
    meshes: number;
    lights: number;
}

/**
 * Geometry and material edits are looked for once every this many frames,
 * each record on its own frame of the cycle, so a frame checks only a share
 * of them. Why: on Sponza those checks touch most of the memory the sync
 * reads, and a frame's sync starts with all of it cold in the cache (three
 * has just rendered). Transforms and visibility are still checked every
 * frame, so motion is never late. An edit reaches GI up to this many frames
 * later, which is short next to what follows it: a geometry edit rebuilds
 * that mesh's BVH in a worker, and probes blend new light over many frames.
 */
export const DEFAULT_EDIT_CHECK_INTERVAL = 4;

export interface SceneSyncOptions {
    /** Where warnings go (each distinct message once). Default `console.warn`. */
    readonly warn?: (message: string) => void;
    /** Look for geometry and material edits every this many frames ({@link DEFAULT_EDIT_CHECK_INTERVAL}); 1 checks every frame. */
    readonly editCheckInterval?: number;
    /** Who made the device: needed to upload environment textures. */
    readonly device: GPUDevice;
}

function emptyReport(): SyncFrameReport {
    return { created: 0, destroyed: 0, transforms: 0, visibility: 0, materials: 0, meshes: 0, lights: 0 };
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

export class SceneSync {
    /**
     * An environment descriptor that replaces whatever three's scene says
     * (environment.ts, source 1). `null` reads it from the scene.
     */
    environment: EnvironmentDescriptor | null = null;
    /** A pmndrs/sky `Sky` to take the sky from (environment.ts, source 2). */
    sky: PmndrsSkyLike | null = null;

    private readonly api: SceneApi;
    private readonly warnOnce: Warn;
    private readonly environmentSync: EnvironmentSync;

    private readonly geometries = new Map<BufferGeometry, GeometryRecord>();
    private readonly materials = new Map<Material, MaterialRecord>();
    private readonly textures = new Map<Texture, TextureRecord>();
    private readonly textureHandleUsers = new Map<TextureHandle, number>();
    private readonly objects = new Map<Object3D, ObjectRecord>();
    private readonly lights = new Map<Light, LightRecord>();

    /** Visible `LightProbeGrid`s seen in the last update: they share three's light-probe slot with the GI light. */
    otherProbeLights = 0;

    private readonly editCheckInterval: number;
    private phaseCounter = 0;
    /** Geometry and material records by edit-check phase: a frame checks one bucket of each, without touching the others. */
    private readonly geometryBuckets: Set<GeometryRecord>[];
    private readonly materialBuckets: Set<MaterialRecord>[];
    private frame = 0;
    private seenObjects = 0;
    /**
     * Object records in the order the last walk met them. A scene that didn't
     * change shape is walked in the same order, so the record for the next
     * mesh is usually the next entry: read in sequence instead of hashed in a
     * Map, which keeps a frame's walk cheap when its data is cold in the
     * cache (it is, after three has rendered).
     */
    private readonly walkOrder: ObjectRecord[] = [];
    /** Plain meshes' last-sent world matrices, one block: slots taken in walk order are read in sequence each frame. */
    private readonly matrixPool = new MatrixPool();
    private walkCursor = 0;
    private seenLights = 0;
    private skyMesh: SkyMeshLike | null = null;
    private sun: DirectionalLight | null = null;
    private sunIlluminance = 0;
    private readonly scratchMatrix = new Matrix4();
    private readonly warned = new Set<string>();

    /** What the last `update` sent. */
    readonly report: SyncFrameReport = emptyReport();

    constructor(api: SceneApi, options: SceneSyncOptions) {
        this.api = api;
        const warn = options.warn ?? ((message: string) => console.warn(`@pmndrs/ddgi/three: ${message}`));
        this.warnOnce = (message) => {
            if (this.warned.has(message)) return;
            this.warned.add(message);
            warn(message);
        };
        this.environmentSync = new EnvironmentSync(api, options.device, this.warnOnce);
        const interval = options.editCheckInterval ?? DEFAULT_EDIT_CHECK_INTERVAL;
        if (!(Number.isInteger(interval) && interval >= 1)) throw new RangeError(`editCheckInterval must be a whole number of frames, at least 1; got ${interval}`);
        this.editCheckInterval = interval;
        this.geometryBuckets = Array.from({ length: interval }, () => new Set<GeometryRecord>());
        this.materialBuckets = Array.from({ length: interval }, () => new Set<MaterialRecord>());
    }

    /** Whether a record on this phase looks for edits this frame. */
    private editCheckDue(phase: number): boolean {
        return phase === this.frame % this.editCheckInterval;
    }

    /** Spreads new records over the edit-check cycle. */
    private nextCheckPhase(): number {
        this.phaseCounter = (this.phaseCounter + 1) % this.editCheckInterval;
        return this.phaseCounter;
    }

    /** How many things are mirrored. */
    get counts(): { meshes: number; materials: number; textures: number; objects: number; instances: number; lights: number } {
        let instances = 0;
        for (const record of this.objects.values()) instances += record.instances.length;
        return { meshes: this.geometries.size, materials: this.materials.size, textures: this.textures.size, objects: this.objects.size, instances, lights: this.lights.size };
    }

    /** How many times the environment was sent, and how many textures uploaded for it. */
    get environmentUpdates(): { updates: number; uploads: number } {
        return { updates: this.environmentSync.updates, uploads: this.environmentSync.uploads };
    }

    /** Walks the scene and sends what changed since the last call, reading world matrices as they are. */
    update(scene: Scene): SyncFrameReport {
        this.frame++;
        Object.assign(this.report, emptyReport());
        this.seenObjects = 0;
        this.seenLights = 0;
        this.otherProbeLights = 0;
        this.skyMesh = null;
        this.sun = null;
        this.sunIlluminance = 0;

        this.walkCursor = 0;
        this.visit(scene, true);
        this.walkOrder.length = this.walkCursor;

        if (this.seenObjects < this.objects.size) this.removeUnseenObjects();
        if (this.seenLights < this.lights.size) this.removeUnseenLights();
        this.checkEdits();
        this.environmentSync.update(scene, { override: this.environment, sky: this.sky, skyMesh: this.skyMesh, sun: this.sun });
        return this.report;
    }

    /** This frame's share of the geometry and material edit checks (DEFAULT_EDIT_CHECK_INTERVAL). */
    private checkEdits(): void {
        const bucket = this.frame % this.editCheckInterval;
        for (const geometry of this.geometryBuckets[bucket]) this.checkGeometry(geometry);
        for (const material of this.materialBuckets[bucket]) this.checkMaterial(material);
    }

    /** Destroys everything this sync created in the platform scene. */
    dispose(): void {
        for (const record of [...this.objects.values()]) this.destroyObject(record);
        for (const record of [...this.lights.values()]) this.api.destroyLight(record.handle);
        this.lights.clear();
        this.environmentSync.dispose();
    }

    // -----------------------------------------------------------------------
    // The walk
    // -----------------------------------------------------------------------

    private visit(object: Object3D, parentVisible: boolean): void {
        const visible = parentVisible && object.visible;
        // Flags are tested most common first: most objects are groups or meshes, and a
        // missing property (isSkyMesh on an ordinary mesh) costs a walk up the prototypes.
        const flags = object as Object3D & { isMesh?: boolean; isLight?: boolean; isSkyMesh?: boolean; isBatchedMesh?: boolean };
        if (flags.isMesh) {
            if (flags.isSkyMesh) {
                // The sky itself, not geometry: it becomes the environment, and its box must not wall the scene in.
                if (visible) this.skyMesh = object as unknown as SkyMeshLike;
                return;
            }
            if (flags.isBatchedMesh) this.warnOnce('BatchedMesh is not mirrored into GI yet; its objects neither block nor bounce light.');
            else this.visitMesh(object as Mesh, visible);
        } else if (flags.isLight) {
            this.visitLight(object as Light, visible);
        } else if (flags.isSkyMesh) {
            if (visible) this.skyMesh = object as unknown as SkyMeshLike;
            return;
        }
        const children = object.children;
        for (let index = 0; index < children.length; index++) this.visit(children[index], visible);
    }

    // -----------------------------------------------------------------------
    // Meshes and instances
    // -----------------------------------------------------------------------

    /**
     * Per mesh, every frame: its geometry and material by identity, its world
     * matrix and visibility. Its GI flag and a material array's contents are
     * read on its edit-check frame only (both change rarely, and each read is
     * another cold cache line).
     */
    private visitMesh(mesh: Mesh, visible: boolean): void {
        const expected = this.walkOrder[this.walkCursor];
        let record = expected !== undefined && expected.object === mesh && !expected.destroyed ? expected : this.objects.get(mesh);
        if (record) {
            const due = this.editCheckDue(record.checkPhase);
            if (record.geometryObject !== mesh.geometry || (due && record.contributes !== contributesFlag(mesh, record.skinned))) {
                // A new geometry or a changed GI flag: the contract has no setter for either, so start over.
                this.destroyObject(record);
                record = undefined;
            } else {
                if (mesh.material !== record.materialKey || (due && Array.isArray(mesh.material))) this.checkMaterials(record);
                this.checkInstances(record, visible);
            }
        }
        if (!record) {
            record = this.createObject(mesh, visible, contributesToGI(mesh));
            if (!record) return;
        }
        record.seenFrame = this.frame;
        this.seenObjects++;
        this.walkOrder[this.walkCursor++] = record;
    }

    private materialForSlot(mesh: Mesh, slot: number): Material | undefined {
        const material = mesh.material;
        if (Array.isArray(material)) return material[slot] ?? material[0];
        return material;
    }

    private createObject(mesh: Mesh, visible: boolean, contributes: boolean): ObjectRecord | undefined {
        const geometryRecord = this.acquireGeometry(mesh);
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
            checkPhase: this.nextCheckPhase(),
            instances: [],
            matrices: new Float64Array(0),
            matrixSlot: -1,
            visible: [],
            contributes,
            instanced: instanced !== null,
            skinned: (mesh as Mesh & { isSkinnedMesh?: boolean }).isSkinnedMesh === true,
            parentMatrix: new Float64Array(MATRIX_FLOATS),
            instanceMatrixVersion: -1,
            seenFrame: this.frame,
            destroyed: false,
        };
        this.objects.set(mesh, record);
        if (instanced) {
            this.syncInstancedMesh(record, instanced, visible);
        } else {
            record.matrixSlot = this.matrixPool.take();
            this.matrixPool.values.set(mesh.matrixWorld.elements, record.matrixSlot * MATRIX_FLOATS);
            this.createInstance(record, mesh.matrixWorld.elements, visible);
        }
        return record;
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

    private destroyObject(record: ObjectRecord): void {
        record.destroyed = true;
        if (record.matrixSlot >= 0) this.matrixPool.release(record.matrixSlot);
        record.matrixSlot = -1;
        for (const instance of record.instances) this.api.destroyInstance(instance);
        this.report.destroyed += record.instances.length;
        record.instances.length = 0;
        this.objects.delete(record.object);
        for (const material of record.materials) this.releaseMaterial(material);
        this.releaseGeometry(record.geometry);
    }

    private removeUnseenObjects(): void {
        for (const record of [...this.objects.values()]) {
            if (record.seenFrame !== this.frame) this.destroyObject(record);
        }
    }

    /** Material per slot by identity; a changed slot means `setMaterials` on every instance. */
    private checkMaterials(record: ObjectRecord): void {
        const mesh = record.object;
        const materials = record.materials;
        record.materialKey = mesh.material;
        let changed = false;
        for (let slot = 0; slot < materials.length; slot++) {
            const material = this.materialForSlot(mesh, slot);
            if (material && material !== materials[slot].material) changed = true;
        }
        if (!changed) return;
        // Acquire the new list before releasing the old, so a material in both keeps its platform copy.
        const next: MaterialRecord[] = [];
        for (let slot = 0; slot < materials.length; slot++) {
            next.push(this.acquireMaterial(this.materialForSlot(mesh, slot) ?? materials[slot].material));
        }
        for (const material of materials) this.releaseMaterial(material);
        record.materials = next;
        const handles = next.map((material) => material.handle);
        for (const instance of record.instances) this.api.setMaterials(instance, handles);
    }

    private checkInstances(record: ObjectRecord, visible: boolean): void {
        const mesh = record.object;
        if (record.instanced) {
            this.syncInstancedMesh(record, mesh as InstancedMesh, visible);
            return;
        }
        if (record.visible[0] !== visible) {
            this.api.setVisible(record.instances[0], visible);
            record.visible[0] = visible;
            this.report.visibility++;
        }
        if (copyIfChanged(mesh.matrixWorld.elements, this.matrixPool.values, record.matrixSlot * MATRIX_FLOATS)) {
            this.api.setTransform(record.instances[0], mesh.matrixWorld.elements);
            this.report.transforms++;
        }
    }

    /**
     * An InstancedMesh: instance i's world matrix is the mesh's world matrix
     * times `instanceMatrix[i]`. Only recomputed when the mesh moved or the
     * instance matrices' version changed; then only changed instances are
     * sent. Instances past `count` are hidden, not destroyed, so a count that
     * goes down and up again costs nothing.
     */
    private syncInstancedMesh(record: ObjectRecord, mesh: InstancedMesh, visible: boolean): void {
        const count = mesh.count;
        const parentMoved = copyIfChanged(mesh.matrixWorld.elements, record.parentMatrix, 0);
        const instancesChanged = mesh.instanceMatrix.version !== record.instanceMatrixVersion;
        const grew = count > record.instances.length;
        if (grew) {
            const matrices = new Float64Array(count * MATRIX_FLOATS);
            matrices.set(record.matrices);
            record.matrices = matrices;
        }
        if (parentMoved || instancesChanged || grew) {
            record.instanceMatrixVersion = mesh.instanceMatrix.version;
            const instanceMatrices = mesh.instanceMatrix.array;
            const matrix = this.scratchMatrix;
            for (let instance = 0; instance < count; instance++) {
                matrix.fromArray(instanceMatrices, instance * MATRIX_FLOATS).premultiply(mesh.matrixWorld);
                if (instance >= record.instances.length) {
                    record.matrices.set(matrix.elements, instance * MATRIX_FLOATS);
                    this.createInstance(record, matrix.elements, visible);
                } else if (copyIfChanged(matrix.elements, record.matrices, instance * MATRIX_FLOATS)) {
                    this.api.setTransform(record.instances[instance], matrix.elements);
                    this.report.transforms++;
                }
            }
        }
        for (let instance = 0; instance < record.instances.length; instance++) {
            const shown = visible && instance < count;
            if (record.visible[instance] === shown) continue;
            this.api.setVisible(record.instances[instance], shown);
            record.visible[instance] = shown;
            this.report.visibility++;
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

    private acquireGeometry(mesh: Mesh): GeometryRecord | undefined {
        const geometry = mesh.geometry;
        const existing = this.geometries.get(geometry);
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
        const descriptor = meshDescriptor(geometry, { includeUv1: uv1, label: geometry.name || mesh.name || undefined });
        if (!descriptor) {
            this.warnOnce(`a mesh without positions ("${mesh.name || geometry.uuid}") is left out of GI.`);
            return undefined;
        }
        const attributes = this.geometryAttributes(geometry, uv1);
        const record: GeometryRecord = {
            geometry,
            handle: this.api.createMesh(descriptor),
            users: 1,
            uv1,
            slotCount,
            attributes,
            versions: attributes.map(attributeVersion),
            checkPhase: this.nextCheckPhase(),
        };
        this.geometries.set(geometry, record);
        this.geometryBuckets[record.checkPhase].add(record);
        return record;
    }

    /** On its edit-check frame: a replaced or edited attribute means `updateMesh` (which rebuilds its BVH). */
    private checkGeometry(record: GeometryRecord): void {
        // Without allocating: this runs for every geometry every frame.
        const geometry = record.geometry;
        const attributes = geometry.attributes;
        const changed =
            attributeChanged(attributes.position, record, GEOMETRY_ATTRIBUTE.position) ||
            attributeChanged(geometry.index, record, GEOMETRY_ATTRIBUTE.index) ||
            attributeChanged(attributes.normal, record, GEOMETRY_ATTRIBUTE.normal) ||
            attributeChanged(attributes.uv, record, GEOMETRY_ATTRIBUTE.uv) ||
            (record.uv1 && attributeChanged(attributes.uv1, record, GEOMETRY_ATTRIBUTE.uv1));
        if (!changed) return;
        const descriptor = meshDescriptor(geometry, { includeUv1: record.uv1, label: geometry.name || undefined });
        if (!descriptor) {
            this.warnOnce(`geometry "${geometry.name || geometry.uuid}" lost its positions; GI keeps its previous shape.`);
            return;
        }
        this.api.updateMesh(record.handle, descriptor);
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
        this.geometries.delete(record.geometry);
        this.geometryBuckets[record.checkPhase].delete(record);
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
        const descriptor = materialDescriptor(material, (texture) => this.textureHandleFor(texture, textureHandles));
        this.useTextureHandles(textureHandles);
        const record: MaterialRecord = {
            material,
            handle: this.api.createMaterial(descriptor),
            users: 1,
            snapshot,
            scratch: new MaterialSnapshot(),
            textureHandles,
            checkPhase: this.nextCheckPhase(),
        };
        this.materials.set(material, record);
        this.materialBuckets[record.checkPhase].add(record);
        return record;
    }

    /** On its edit-check frame: compare the material's GI values with the last ones sent. */
    private checkMaterial(record: MaterialRecord): void {
        record.scratch.read(record.material);
        if (record.scratch.equals(record.snapshot)) return;
        [record.snapshot, record.scratch] = [record.scratch, record.snapshot];
        const textureHandles: TextureHandle[] = [];
        const descriptor = materialDescriptor(record.material, (texture) => this.textureHandleFor(texture, textureHandles));
        this.useTextureHandles(textureHandles);
        // Every field, so a removed texture or cutoff is cleared too.
        this.api.updateMaterial(record.handle, descriptor);
        this.releaseTextureHandles(record.textureHandles);
        record.textureHandles = textureHandles;
        this.report.materials++;
    }

    private releaseMaterial(record: MaterialRecord): void {
        record.users--;
        if (record.users > 0) return;
        this.api.destroyMaterial(record.handle);
        this.releaseTextureHandles(record.textureHandles);
        this.materials.delete(record.material);
        this.materialBuckets[record.checkPhase].delete(record);
    }

    /**
     * The platform texture for a three texture, made when first needed and
     * again when the texture's version changes (new pixels). Undefined while
     * it has no readable pixels (still loading, compressed, a render target).
     */
    private textureHandleFor(texture: Texture, collect: TextureHandle[]): TextureHandle | undefined {
        const existing = this.textures.get(texture);
        if (existing && existing.version === texture.version) {
            collect.push(existing.handle);
            return existing.handle;
        }
        const source = textureSource(texture);
        if (!source) {
            const flags = texture as Texture & { isCompressedTexture?: boolean };
            if (flags.isCompressedTexture) this.warnOnce('compressed textures are not read for GI yet (M4); those materials use their colour only.');
            return undefined;
        }
        const handle = this.api.createTexture({ source, srgb: isSrgbTexture(texture), label: texture.name || undefined });
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

    private visitLight(light: Light, visible: boolean): void {
        const flags = light as Light & {
            isGILight?: boolean;
            isLightProbeGrid?: boolean;
            isAmbientLight?: boolean;
            isHemisphereLight?: boolean;
            isLightProbe?: boolean;
            isRectAreaLight?: boolean;
            isPointLight?: boolean;
            isSpotLight?: boolean;
            isDirectionalLight?: boolean;
        };
        if (flags.isLightProbeGrid && visible && light.intensity > 0) this.otherProbeLights++;
        if (flags.isGILight || flags.isLightProbeGrid) return;
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
        const descriptor = lightDescriptor(light, visible ? 1 : 0);
        if (!descriptor) return;

        if (flags.isDirectionalLight && visible) {
            const illuminance = light.intensity * luminance(light.color);
            if (!this.sun || illuminance > this.sunIlluminance) {
                this.sun = light as DirectionalLight;
                this.sunIlluminance = illuminance;
            }
        }

        const record = this.lights.get(light);
        if (!record) {
            this.lights.set(light, { light, handle: this.api.createLight(descriptor), descriptor, seenFrame: this.frame });
            this.report.created++;
        } else {
            // A three light never changes type, so the descriptor types always match.
            const changes = lightChanges(record.descriptor, descriptor);
            if (changes) {
                this.api.updateLight(record.handle, changes);
                record.descriptor = descriptor;
                this.report.lights++;
            }
            record.seenFrame = this.frame;
        }
        this.seenLights++;
    }

    private removeUnseenLights(): void {
        for (const [light, record] of [...this.lights]) {
            if (record.seenFrame === this.frame) continue;
            this.api.destroyLight(record.handle);
            this.lights.delete(light);
            this.report.destroyed++;
        }
    }
}

/**
 * World matrices in one growing Float64Array, 16 numbers per slot. Plain
 * meshes keep their last-sent matrix here rather than in an array each, so
 * the per-frame comparison reads memory in order (meshes take slots in the
 * order the walk first meets them) instead of a scattered array per mesh.
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
