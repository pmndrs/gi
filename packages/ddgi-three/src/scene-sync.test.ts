import {
    Bone,
    BoxGeometry,
    BufferAttribute,
    BufferGeometry,
    CompressedTexture,
    DataTexture,
    Color,
    DirectionalLight,
    Float32BufferAttribute,
    Group,
    InstancedMesh,
    Matrix4,
    Mesh,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    Object3D,
    PlaneGeometry,
    PointLight,
    RGBA_BPTC_Format,
    RenderTarget,
    SRGBColorSpace,
    Scene,
    ShaderMaterial,
    SkinnedMesh,
    Skeleton,
    Texture,
    Uint16BufferAttribute,
    Vector3,
} from 'three';
import { beforeEach, describe, expect, it } from 'vitest';
import type {
    Aabb,
    CompressedPixelData,
    DeformedPositions,
    EnvironmentDescriptor,
    InstanceDescriptor,
    InstanceHandle,
    LightChanges,
    LightDescriptor,
    LightHandle,
    Mat4,
    MaterialDescriptor,
    PixelData,
    MaterialHandle,
    MeshDescriptor,
    MeshHandle,
    SceneApi,
    TextureDescriptor,
    TextureHandle,
} from '@pmndrs/ddgi-core';
import { ALWAYS_CHECKED_LIGHTS, DEFAULT_LIGHTS_PER_FRAME, HOT_FRAMES, SceneSync } from './scene-sync.ts';

/** A SceneApi that records every call and keeps the live objects, for checking what the sync sent. */
class RecordingSceneApi implements SceneApi {
    readonly calls: string[] = [];
    readonly meshes = new Map<number, MeshDescriptor>();
    readonly materials = new Map<number, MaterialDescriptor>();
    readonly textures = new Map<number, TextureDescriptor>();
    readonly instances = new Map<number, Omit<InstanceDescriptor, 'materials'> & { materials: readonly MaterialHandle[]; transform: number[]; visible: boolean }>();
    readonly lights = new Map<number, LightDescriptor>();
    environment: EnvironmentDescriptor | null = null;
    private next = 1;

    get bounds(): Aabb {
        return { min: [0, 0, 0], max: [0, 0, 0] };
    }

    private log(name: string): void {
        this.calls.push(name);
    }

    createMesh(descriptor: MeshDescriptor): MeshHandle {
        this.log('createMesh');
        const handle = this.next++;
        this.meshes.set(handle, descriptor);
        return handle as MeshHandle;
    }
    updateMesh(mesh: MeshHandle, descriptor: MeshDescriptor): void {
        this.log('updateMesh');
        this.meshes.set(mesh, descriptor);
    }
    /** Every pose sent, in order: the mesh and a copy of its positions (the platform copies at the call too). */
    readonly poses: { mesh: number; positions: Float32Array }[] = [];
    setDeformedPositions(mesh: MeshHandle, positions: Float32Array | DeformedPositions): void {
        this.log('setDeformedPositions');
        if (!(positions instanceof Float32Array)) throw new Error('the CPU form is the only one the adapter sends until T3.3');
        if (!this.meshes.get(mesh)?.deforming) throw new Error('setDeformedPositions on a mesh not created deforming');
        this.poses.push({ mesh, positions: positions.slice() });
    }
    destroyMesh(mesh: MeshHandle): void {
        this.log('destroyMesh');
        for (const instance of this.instances.values()) if (instance.mesh === mesh) throw new Error('mesh still used');
        this.meshes.delete(mesh);
    }
    createTexture(descriptor: TextureDescriptor): TextureHandle {
        this.log('createTexture');
        const handle = this.next++;
        this.textures.set(handle, descriptor);
        return handle as TextureHandle;
    }
    destroyTexture(texture: TextureHandle): void {
        this.log('destroyTexture');
        this.textures.delete(texture);
    }
    createMaterial(descriptor: MaterialDescriptor): MaterialHandle {
        this.log('createMaterial');
        const handle = this.next++;
        this.materials.set(handle, descriptor);
        return handle as MaterialHandle;
    }
    updateMaterial(material: MaterialHandle, changes: Partial<MaterialDescriptor>): void {
        this.log('updateMaterial');
        this.materials.set(material, { ...this.materials.get(material), ...changes });
    }
    destroyMaterial(material: MaterialHandle): void {
        this.log('destroyMaterial');
        for (const instance of this.instances.values()) if (instance.materials.includes(material)) throw new Error('material still used');
        this.materials.delete(material);
    }
    createInstance(descriptor: InstanceDescriptor): InstanceHandle {
        this.log('createInstance');
        const handle = this.next++;
        this.instances.set(handle, { ...descriptor, transform: Array.from(descriptor.transform as ArrayLike<number>), visible: descriptor.visible !== false });
        return handle as InstanceHandle;
    }
    setTransform(instance: InstanceHandle, transform: Mat4): void {
        this.log('setTransform');
        this.instances.get(instance)!.transform = Array.from(transform);
    }
    /** Every setTransforms call's instances, in order (one 'setTransforms' entry in `calls` per batch). */
    readonly transformBatches: number[][] = [];
    setTransforms(instances: Uint32Array, matrices: Float32Array): void {
        this.log('setTransforms');
        expect(matrices.length).toBe(instances.length * 16);
        this.transformBatches.push(Array.from(instances));
        instances.forEach((instance, index) => {
            const record = this.instances.get(instance);
            if (!record) throw new Error(`setTransforms on unknown instance ${instance}`);
            record.transform = Array.from(matrices.subarray(index * 16, index * 16 + 16));
        });
    }
    setVisible(instance: InstanceHandle, visible: boolean): void {
        this.log('setVisible');
        this.instances.get(instance)!.visible = visible;
    }
    setMaterials(instance: InstanceHandle, materials: readonly MaterialHandle[]): void {
        this.log('setMaterials');
        this.instances.get(instance)!.materials = materials;
    }
    destroyInstance(instance: InstanceHandle): void {
        this.log('destroyInstance');
        this.instances.delete(instance);
    }
    instanceIndex(instance: InstanceHandle): number {
        if (!this.instances.has(instance)) throw new Error('destroyed instance');
        // Handles are unique here, so the handle itself serves as the index.
        return instance;
    }
    createLight(descriptor: LightDescriptor): LightHandle {
        this.log('createLight');
        const handle = this.next++;
        this.lights.set(handle, descriptor);
        return handle as LightHandle;
    }
    updateLight(light: LightHandle, changes: LightChanges): void {
        this.log(`updateLight ${Object.keys(changes).sort().join(',')}`);
        this.lights.set(light, { ...this.lights.get(light)!, ...changes } as LightDescriptor);
    }
    destroyLight(light: LightHandle): void {
        this.log('destroyLight');
        this.lights.delete(light);
    }
    setEnvironment(environment: EnvironmentDescriptor): void {
        this.log('setEnvironment');
        this.environment = environment;
    }

    take(): string[] {
        return this.calls.splice(0);
    }
}

const NO_DEVICE = {} as GPUDevice;

let api: RecordingSceneApi;
let sync: SceneSync;
let warnings: string[];

function frame(scene: Scene): string[] {
    scene.updateMatrixWorld();
    sync.update(scene);
    return api.take();
}

beforeEach(() => {
    api = new RecordingSceneApi();
    warnings = [];
    // Small scenes: the default budgets cover them whole every frame, so each test sees a change on the next frame.
    sync = new SceneSync(api, { device: NO_DEVICE, warn: (message) => warnings.push(message) });
});

describe('a still scene', () => {
    it('is sent once, then nothing', () => {
        const scene = new Scene();
        const geometry = new BoxGeometry();
        const material = new MeshStandardMaterial({ color: 0x808080 });
        for (let index = 0; index < 3; index++) {
            const mesh = new Mesh(geometry, material);
            mesh.position.x = index;
            scene.add(mesh);
        }
        scene.add(new PointLight(0xffffff, 10, 0));
        const first = frame(scene);
        expect(first.filter((call) => call === 'createMesh')).toHaveLength(1);
        expect(first.filter((call) => call === 'createMaterial')).toHaveLength(1);
        expect(first.filter((call) => call === 'createInstance')).toHaveLength(3);
        expect(first).toContain('createLight');
        expect(first).toContain('setEnvironment');
        expect(frame(scene)).toEqual([]);
        expect(frame(scene)).toEqual([]);
        expect(sync.report).toMatchObject({ created: 0, destroyed: 0, transforms: 0, visibility: 0, materials: 0, meshes: 0, lights: 0 });
    });
});

describe('changes', () => {
    let scene: Scene;
    let moving: Mesh;
    let material: MeshStandardMaterial;

    beforeEach(() => {
        scene = new Scene();
        material = new MeshStandardMaterial({ color: new Color().setRGB(0.5, 0.5, 0.5) });
        moving = new Mesh(new BoxGeometry(), material);
        scene.add(moving, new Mesh(new BoxGeometry(), material));
        frame(scene);
    });

    it('sends one transform when one object moves', () => {
        moving.position.set(1, 2, 3);
        expect(frame(scene)).toEqual(['setTransforms']);
        const instance = [...api.instances.values()][0];
        expect(instance.transform.slice(12, 15)).toEqual([1, 2, 3]);
    });

    it('follows a moving parent', () => {
        const parent = new Group();
        scene.add(parent);
        moving.removeFromParent();
        parent.add(moving);
        frame(scene);
        parent.position.y = 5;
        expect(frame(scene)).toEqual(['setTransforms']);
    });

    it('sends a colour edit (three bumps no version for it), with metalness apart', () => {
        material.color.setRGB(1, 0, 0);
        material.metalness = 0.5;
        expect(frame(scene)).toEqual(['updateMaterial']);
        const descriptor = [...api.materials.values()][0];
        expect(descriptor.albedo).toEqual([1, 0, 0]);
        expect(descriptor.metalness).toBe(0.5);
    });

    it('reports the GI material it last sent for a three material', () => {
        expect(sync.sentMaterial(material)?.albedo).toEqual([0.5, 0.5, 0.5]);
        material.color.setRGB(0.25, 0, 0);
        frame(scene);
        expect(sync.sentMaterial(material)).toEqual([...api.materials.values()][0]);
        expect(sync.sentMaterial(material)?.albedo).toEqual([0.25, 0, 0]);
        expect(sync.sentMaterial(new MeshStandardMaterial())).toBeUndefined();
    });

    it('hides and shows with the parent', () => {
        const parent = new Group();
        scene.add(parent);
        moving.removeFromParent();
        parent.add(moving);
        frame(scene);
        parent.visible = false;
        expect(frame(scene)).toEqual(['setVisible']);
        expect([...api.instances.values()].filter((instance) => !instance.visible)).toHaveLength(1);
        parent.visible = true;
        expect(frame(scene)).toEqual(['setVisible']);
    });

    it('swaps materials', () => {
        moving.material = new MeshStandardMaterial({ color: 0xffffff });
        expect(frame(scene)).toEqual(['createMaterial', 'setMaterials']);
    });

    it('rebuilds a mesh when its positions change', () => {
        const position = moving.geometry.getAttribute('position') as BufferAttribute;
        position.setX(0, 5);
        position.needsUpdate = true;
        expect(frame(scene)).toEqual(['updateMesh']);
    });

    it('destroys what leaves the scene', () => {
        moving.removeFromParent();
        expect(frame(scene)).toEqual(['destroyInstance', 'destroyMesh']);
        scene.clear();
        expect(frame(scene)).toEqual(['destroyInstance', 'destroyMaterial', 'destroyMesh']);
        expect(api.instances.size + api.meshes.size + api.materials.size).toBe(0);
    });
});

describe('the walk', () => {
    /** A scene of `count` meshes in a flat row, each with its own material, walked whole once. */
    function rowScene(sync: SceneSync, count: number): { scene: Scene; meshes: Mesh[]; materials: MeshStandardMaterial[] } {
        const scene = new Scene();
        const materials: MeshStandardMaterial[] = [];
        const meshes: Mesh[] = [];
        const geometry = new BoxGeometry();
        for (let index = 0; index < count; index++) {
            const material = new MeshStandardMaterial();
            const mesh = new Mesh(geometry, material);
            mesh.position.x = index;
            materials.push(material);
            meshes.push(mesh);
            scene.add(mesh);
        }
        scene.updateMatrixWorld();
        sync.update(scene);
        api.take();
        return { scene, meshes, materials };
    }

    it('looks at a bounded number of objects per frame, however big the scene', () => {
        const bounded = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 16 });
        const { scene } = rowScene(bounded, 200);
        // The first frame takes the whole scene, so it is complete at once.
        expect(api.instances.size).toBe(200);
        for (let frame = 0; frame < 30; frame++) {
            bounded.update(scene);
            expect(bounded.report.walked).toBeLessThanOrEqual(16);
            expect(api.take()).toEqual([]);
        }
    });

    it('sees a static object\'s first move within a pass, then every frame', () => {
        const bounded = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 16 });
        const { scene, meshes } = rowScene(bounded, 100);
        const moving = meshes[57];
        moving.position.y = 1;
        scene.updateMatrixWorld();
        // 101 objects (the scene and 100 meshes) at 16 a frame: within ceil(101 / 16) = 7 frames.
        let framesUntilSeen = 0;
        for (; framesUntilSeen < 10; framesUntilSeen++) {
            bounded.update(scene);
            if (api.take().includes('setTransforms')) break;
        }
        expect(framesUntilSeen).toBeLessThan(Math.ceil(101 / 16));
        // Now hot: each further move is sent in the frame it happens.
        for (let frame = 2; frame < 6; frame++) {
            moving.position.y = frame;
            scene.updateMatrixWorld();
            bounded.update(scene);
            expect(api.take()).toEqual(['setTransforms']);
        }
    });

    it('picks up a marked object in the same update: moved, added and removed', () => {
        const bounded = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 1 });
        const { scene, meshes } = rowScene(bounded, 50);
        meshes[30].position.y = 3;
        scene.updateMatrixWorld();
        bounded.markChanged(meshes[30]);
        bounded.update(scene);
        expect(api.take()).toContain('setTransforms');

        const added = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        scene.add(added);
        scene.updateMatrixWorld();
        bounded.markChanged(added);
        bounded.update(scene);
        expect(api.take()).toEqual(['createMesh', 'createMaterial', 'createInstance']);

        added.removeFromParent();
        bounded.markChanged(added);
        bounded.update(scene);
        expect(api.take()).toEqual(['destroyInstance', 'destroyMaterial', 'destroyMesh']);
    });

    it('finds unmarked additions and removals within a pass', () => {
        const bounded = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 8 });
        const { scene, meshes } = rowScene(bounded, 40);
        meshes[5].removeFromParent();
        scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        scene.updateMatrixWorld();
        const calls: string[] = [];
        // Two passes at most: 41 objects at 8 a frame.
        for (let frame = 0; frame < 2 * Math.ceil(41 / 8); frame++) {
            bounded.update(scene);
            calls.push(...api.take());
        }
        expect(calls.filter((call) => call === 'createInstance')).toHaveLength(1);
        expect(calls.filter((call) => call === 'destroyInstance')).toHaveLength(1);
        expect(api.instances.size).toBe(40);
    });

    it('checks a bounded number of materials for edits per frame and never misses one', () => {
        const bounded = new SceneSync(api, { device: NO_DEVICE, editChecksPerFrame: 3 });
        const { scene, materials } = rowScene(bounded, 9);
        for (const material of materials) material.color.setRGB(0.1, 0.2, 0.3);
        const updatesPerFrame: number[] = [];
        for (let frame = 0; frame < 5; frame++) {
            bounded.update(scene);
            updatesPerFrame.push(api.take().filter((call) => call === 'updateMaterial').length);
        }
        expect(updatesPerFrame).toEqual([3, 3, 3, 0, 0]);
    });

    it('follows a hidden parent through the hot set', () => {
        const bounded = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 4 });
        const scene = new Scene();
        const parent = new Group();
        const child = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        parent.add(child);
        scene.add(parent);
        for (let index = 0; index < 20; index++) scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        scene.updateMatrixWorld();
        bounded.update(scene);
        child.position.x = 1;
        scene.updateMatrixWorld();
        bounded.markChanged(child);
        bounded.update(scene);
        api.take();
        // The child is hot now, so hiding its parent shows up in the next frame although the walk is elsewhere.
        parent.visible = false;
        bounded.update(scene);
        expect(api.take()).toEqual(['setVisible']);
    });

    it('keeps an object moved mid-pass into a part already walked', () => {
        const bounded = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 4 });
        const scene = new Scene();
        // scene → early, filler (20 meshes), holder → late: holder is expanded only near the end of a pass.
        const early = new Group();
        const filler = new Group();
        const holder = new Group();
        scene.add(early, filler, holder);
        for (let index = 0; index < 20; index++) filler.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        const late = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        holder.add(late);
        scene.updateMatrixWorld();
        bounded.update(scene);
        api.take();
        // One frame in, the walk has passed `early` but not reached `holder`; `late` moves into `early`.
        bounded.update(scene);
        early.add(late);
        scene.updateMatrixWorld();
        const calls: string[] = [];
        for (let frame = 0; frame < 2 * Math.ceil(25 / 4); frame++) {
            bounded.update(scene);
            calls.push(...api.take());
        }
        expect(calls).not.toContain('destroyInstance');
        expect(api.instances.size).toBe(21);
    });

    it('counts materials with their own envMap (they need GI coverage for the workaround)', () => {
        const scene = new Scene();
        const material = new MeshStandardMaterial();
        scene.add(new Mesh(new BoxGeometry(), material));
        frame(scene);
        expect(sync.materialsWithEnvMap).toBe(0);
        material.envMap = new Texture();
        frame(scene);
        expect(sync.materialsWithEnvMap).toBe(1);
        scene.clear();
        frame(scene);
        expect(sync.materialsWithEnvMap).toBe(0);
    });

    it('sends a texture\'s pixels as they are (no average folded in), and again when its version changes', () => {
        const scene = new Scene();
        const pixels = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
        const map = new DataTexture(pixels, 2, 1);
        map.needsUpdate = true;
        const material = new MeshStandardMaterial({ color: new Color().setRGB(1, 1, 0.5), map });
        // A second material on the same texture: one platform texture for both.
        const other = new MeshStandardMaterial({ map });
        scene.add(new Mesh(new BoxGeometry(), material), new Mesh(new BoxGeometry(), other));
        frame(scene);
        expect(api.textures.size).toBe(1);
        const [first] = [...api.textures.keys()];
        expect((api.textures.get(first)!.source as PixelData).data).toBe(pixels);
        const sent = sync.sentMaterial(material)!;
        expect(sent.albedo).toEqual([1, 1, 0.5]);
        expect(sent.albedoTexture).toBe(first);
        expect(sync.sentTexture(map)).toBe(first);
        // New pixels (a new version): found by the edit checks, sent as a new texture; the old one goes when no material uses it.
        pixels.set([0, 255, 0, 255, 0, 255, 0, 255]);
        map.needsUpdate = true;
        const calls = frame(scene);
        expect(calls.filter((call) => call === 'createTexture')).toHaveLength(1);
        expect(calls.filter((call) => call === 'updateMaterial')).toHaveLength(2);
        expect(calls).toContain('destroyTexture');
        const [second] = [...api.textures.keys()];
        expect(api.textures.size).toBe(1);
        expect(second).not.toBe(first);
        expect(sync.sentMaterial(material)!.albedoTexture).toBe(second);
        expect(sync.sentMaterial(other)!.albedoTexture).toBe(second);
        // A still frame after: nothing.
        expect(frame(scene)).toEqual([]);
    });

    it('sends a CompressedTexture map as its compressed mips', () => {
        const scene = new Scene();
        const mipmaps = [256, 128].map((size) => ({ data: new Uint8Array((size / 4) * (size / 4) * 16), width: size, height: size }));
        const map = new CompressedTexture(mipmaps, 256, 256, RGBA_BPTC_Format);
        map.colorSpace = SRGBColorSpace;
        map.needsUpdate = true;
        scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial({ map })));
        frame(scene);
        const [texture] = [...api.textures.values()];
        expect((texture.source as CompressedPixelData).format).toBe('bc7-rgba-unorm-srgb');
        expect((texture.source as CompressedPixelData).levels[0].data).toBe(mipmaps[0].data);
        expect(warnings).toEqual([]);
    });

    it('warns once about textures GI can\'t read, and once per custom-shaded material', () => {
        const scene = new Scene();
        const target = new RenderTarget(4, 4);
        const shader = new ShaderMaterial({ name: 'water' });
        scene.add(
            new Mesh(new BoxGeometry(), new MeshStandardMaterial({ map: target.texture })),
            new Mesh(new BoxGeometry(), new MeshStandardMaterial({ map: target.texture, color: 0xff0000 })),
            new Mesh(new BoxGeometry(), shader),
            new Mesh(new BoxGeometry(), shader),
        );
        frame(scene);
        frame(scene);
        expect(api.textures.size).toBe(0);
        expect(warnings.filter((warning) => warning.includes('render-target'))).toHaveLength(1);
        expect(warnings.filter((warning) => warning.includes('"water"'))).toHaveLength(1);
    });

    it('picks up a changed GI flag through the edit rotation, and at once when marked', () => {
        const scene = new Scene();
        const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        scene.add(mesh);
        frame(scene);
        mesh.userData.giContributes = false;
        frame(scene);
        frame(scene);
        expect([...api.instances.values()].map((instance) => instance.contributesToGI)).toEqual([false]);
        mesh.userData.giContributes = true;
        sync.markChanged(mesh);
        frame(scene);
        expect([...api.instances.values()].map((instance) => instance.contributesToGI)).toEqual([true]);
    });

    it('rejects budgets below one', () => {
        expect(() => new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 0 })).toThrow(RangeError);
        expect(() => new SceneSync(api, { device: NO_DEVICE, editChecksPerFrame: 0 })).toThrow(RangeError);
    });
});

describe('see-through materials (per triangle since M4)', () => {
    /** contributesToGI of each instance, by label (the mesh's name). */
    function contributions(): Record<string, boolean | undefined> {
        const result: Record<string, boolean | undefined> = {};
        for (const instance of api.instances.values()) result[instance.label ?? '?'] = instance.contributesToGI;
        return result;
    }

    /** A decal as GLTFLoader makes glTF `alphaMode: BLEND` with no metallicFactor (Sponza's dirt_decal). */
    function blendedDecal(): MeshStandardMaterial {
        return new MeshStandardMaterial({ transparent: true, metalness: 1, name: 'dirt_decal' });
    }

    it('sends every mesh into GI\'s rays: decals, glass and meshes that mix them with opaque materials, with no warning', () => {
        const scene = new Scene();
        const decalMaterial = blendedDecal();
        const glass = new MeshPhysicalMaterial({ transmission: 1 });
        // BoxGeometry has six groups; with two materials, face 1 is glass and the rest are opaque.
        const facade = new Mesh(new BoxGeometry(), [new MeshStandardMaterial(), glass]);
        scene.add(
            Object.assign(new Mesh(new BoxGeometry(), decalMaterial), { name: 'decal' }),
            Object.assign(new Mesh(new BoxGeometry(), glass), { name: 'pane' }),
            Object.assign(facade, { name: 'facade' }),
        );
        frame(scene);
        expect(contributions()).toEqual({ decal: true, pane: true, facade: true });
        expect([sync.sentContributes(facade), sync.sentContributes(new Mesh())]).toEqual([true, undefined]);
        // The decal blocks light by its alpha; the glass lets it through, triangle by triangle in the platform.
        expect(sync.sentMaterial(decalMaterial)).toMatchObject({ alphaMode: 'stochastic', blocksLight: true });
        expect(sync.sentMaterial(glass)?.blocksLight).toBe(false);
        expect(warnings).toEqual([]);
    });

    it('sends a material edit or swap that makes a mesh see-through as a material change only, with the same instances', () => {
        const scene = new Scene();
        const material = new MeshStandardMaterial();
        const mesh = Object.assign(new Mesh(new BoxGeometry(), material), { name: 'pane' });
        scene.add(mesh);
        frame(scene);
        material.transparent = true;
        expect(frame(scene)).toEqual(['updateMaterial']);
        expect(sync.sentMaterial(material)?.alphaMode).toBe('stochastic');
        mesh.material = new MeshPhysicalMaterial({ transmission: 1 });
        expect(frame(scene)).toEqual(['createMaterial', 'setMaterials', 'destroyMaterial']);
        expect(contributions()).toEqual({ pane: true });
        expect(api.instances.size).toBe(1);
    });
});

describe('instances', () => {
    it('lists each mirrored mesh with its instances, and counts creations and destructions (RC1.1)', () => {
        const scene = new Scene();
        const plain = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        const instanced = new InstancedMesh(new BoxGeometry(), new MeshStandardMaterial(), 3);
        scene.add(plain, instanced);
        const before = sync.instanceVersion;
        frame(scene);
        const listed = new Map(sync.mirroredInstances());
        expect(listed.get(plain)).toHaveLength(1);
        expect(listed.get(instanced)).toHaveLength(3);
        // Instance i of the InstancedMesh is entry i: the order the screen inputs' table relies on.
        expect([...api.instances.keys()].slice(1)).toEqual(listed.get(instanced));
        const afterCreate = sync.instanceVersion;
        expect(afterCreate).not.toBe(before);

        // Moves and visibility don't count; creations and destructions do.
        plain.position.x = 4;
        instanced.count = 2;
        frame(scene);
        expect(sync.instanceVersion).toBe(afterCreate);
        plain.removeFromParent();
        frame(scene);
        expect(sync.instanceVersion).not.toBe(afterCreate);
        expect(new Map(sync.mirroredInstances()).has(plain)).toBe(false);
    });

    it('sends each InstancedMesh instance with its world matrix', () => {
        const scene = new Scene();
        const instanced = new InstancedMesh(new BoxGeometry(), new MeshStandardMaterial(), 3);
        instanced.position.set(10, 0, 0);
        for (let index = 0; index < 3; index++) instanced.setMatrixAt(index, new Matrix4().makeTranslation(0, index, 0));
        scene.add(instanced);
        const first = frame(scene);
        expect(first.filter((call) => call === 'createInstance')).toHaveLength(3);
        const transforms = [...api.instances.values()].map((instance) => instance.transform.slice(12, 15));
        expect(transforms).toEqual([
            [10, 0, 0],
            [10, 1, 0],
            [10, 2, 0],
        ]);
        expect(frame(scene)).toEqual([]);

        instanced.setMatrixAt(1, new Matrix4().makeTranslation(0, 7, 0));
        instanced.instanceMatrix.needsUpdate = true;
        expect(frame(scene)).toEqual(['setTransforms']);

        instanced.count = 2;
        expect(frame(scene)).toEqual(['setVisible']);
        expect([...api.instances.values()][2].visible).toBe(false);
    });

    it('sends only the instances an update range names, and every changed one without ranges (T2.M1)', () => {
        const scene = new Scene();
        const count = 100;
        const instanced = new InstancedMesh(new BoxGeometry(), new MeshStandardMaterial(), count);
        for (let index = 0; index < count; index++) instanced.setMatrixAt(index, new Matrix4().makeTranslation(index, 0, 0));
        scene.add(instanced);
        frame(scene);
        const handles = [...api.instances.keys()];
        // Checked last frame, so this frame's ranges are trusted: instances 40 and 41 are named (as three's own
        // addUpdateRange counts them, in array elements). Instance 90 also changes, outside the ranges: the range
        // says it didn't, so it isn't even looked at. (Three would not upload it either.)
        for (const index of [40, 41, 90]) instanced.setMatrixAt(index, new Matrix4().makeTranslation(index, 5, 0));
        instanced.instanceMatrix.addUpdateRange(40 * 16, 2 * 16);
        instanced.instanceMatrix.needsUpdate = true;
        expect(frame(scene)).toEqual(['setTransforms']);
        expect(api.transformBatches.at(-1)).toEqual([handles[40], handles[41]]);
        instanced.instanceMatrix.clearUpdateRanges();

        // No ranges: every instance is compared, and the changed one (90, still unsent) goes out.
        instanced.setMatrixAt(3, new Matrix4().makeTranslation(3, 5, 0));
        instanced.instanceMatrix.needsUpdate = true;
        expect(frame(scene)).toEqual(['setTransforms']);
        expect(api.transformBatches.at(-1)).toEqual([handles[3], handles[90]]);
    });

    it('sends every moved object of a frame in one setTransforms call (T2.M1)', () => {
        const scene = new Scene();
        const geometry = new BoxGeometry();
        const material = new MeshStandardMaterial();
        const meshes = Array.from({ length: 5 }, (_, index) => {
            const mesh = new Mesh(geometry, material);
            mesh.position.x = index;
            scene.add(mesh);
            return mesh;
        });
        const instanced = new InstancedMesh(geometry, material, 4);
        for (let index = 0; index < 4; index++) instanced.setMatrixAt(index, new Matrix4().makeTranslation(0, index, 0));
        scene.add(instanced);
        frame(scene);
        for (const mesh of meshes.slice(1, 4)) mesh.position.y += 1;
        instanced.position.z = 2;
        const calls = frame(scene);
        expect(calls).toEqual(['setTransforms']);
        // Three meshes and the four instances under the moved InstancedMesh.
        expect(api.transformBatches.at(-1)).toHaveLength(3 + 4);
        expect(sync.report.transforms).toBe(7);
    });

    it('shares one platform mesh between meshes that share a geometry', () => {
        const scene = new Scene();
        const geometry = new BoxGeometry();
        scene.add(new Mesh(geometry, new MeshStandardMaterial()), new Mesh(geometry, new MeshStandardMaterial()));
        const calls = frame(scene);
        expect(calls.filter((call) => call === 'createMesh')).toHaveLength(1);
        expect(calls.filter((call) => call === 'createMaterial')).toHaveLength(2);
    });

    it('gives each group its material slot', () => {
        const scene = new Scene();
        const materials = Array.from({ length: 6 }, () => new MeshStandardMaterial());
        scene.add(new Mesh(new BoxGeometry(), materials));
        frame(scene);
        const instance = [...api.instances.values()][0];
        expect(instance.materials).toHaveLength(6);
        expect(new Set(instance.materials).size).toBe(6);
    });

    it('handles non-indexed geometry', () => {
        const scene = new Scene();
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(new Float32Array(9), 3));
        scene.add(new Mesh(geometry, new MeshStandardMaterial()));
        frame(scene);
        expect(Array.from([...api.meshes.values()][0].indices)).toEqual([0, 1, 2]);
    });

    it('reads the contributes-to-GI flag, off by default for skinned meshes', () => {
        const scene = new Scene();
        const prop = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        prop.userData.giContributes = false;
        const character = new SkinnedMesh(new BoxGeometry(), new MeshStandardMaterial());
        character.bind(new Skeleton([]));
        const tree = new SkinnedMesh(new BoxGeometry(), new MeshStandardMaterial());
        tree.bind(new Skeleton([]));
        tree.userData.giContributes = true;
        scene.add(prop, character, tree);
        frame(scene);
        expect([...api.instances.values()].map((instance) => instance.contributesToGI)).toEqual([false, false, true]);
    });

    it('leaves a SkyMesh out of the geometry', () => {
        const scene = new Scene();
        const sky = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        Object.assign(sky, { isSkyMesh: true, turbidity: { value: 2 }, sunPosition: { value: { x: 0, y: 1, z: 0 } } });
        scene.add(sky);
        frame(scene);
        expect(api.instances.size).toBe(0);
        expect(api.environment?.type).toBe('procedural');
    });
});

describe('lights', () => {
    it('sends a point light with distance 0 as range 0, and only changed fields after', () => {
        const scene = new Scene();
        const light = new PointLight(0xffffff, 15, 0, 2);
        light.position.set(0, 3, 0);
        scene.add(light);
        frame(scene);
        const descriptor = [...api.lights.values()][0];
        expect(descriptor.type === 'point' && descriptor.range).toBe(0);
        light.intensity = 20;
        expect(frame(scene)).toEqual(['updateLight intensity']);
        light.position.x = 1;
        expect(frame(scene)).toEqual(['updateLight position']);
        light.visible = false;
        expect(frame(scene)).toEqual(['updateLight intensity']);
        expect([...api.lights.values()][0].intensity).toBe(0);
    });

    it('takes the sun for a SkyMesh from the brightest directional light', () => {
        const scene = new Scene();
        scene.add(new DirectionalLight(0xffffff, 2), new DirectionalLight(0xffffff, 50000));
        const sky = new Group() as Group & Record<string, unknown>;
        Object.assign(sky, { isSkyMesh: true, turbidity: { value: 2 }, sunPosition: { value: { x: 0, y: 1, z: 0 } } });
        scene.add(sky);
        frame(scene);
        expect(api.environment?.type === 'procedural' && api.environment.sunIlluminance).toBeCloseTo(50000);
    });

    it('warns once about lights that add diffuse light on top of GI', () => {
        const scene = new Scene();
        const ambient = new (class extends PointLight {
            readonly isAmbientLight = true;
        })();
        scene.add(ambient);
        frame(scene);
        frame(scene);
        expect(warnings).toHaveLength(1);
        expect(api.lights.size).toBe(0);
    });

    it('destroys removed lights', () => {
        const scene = new Scene();
        const light = new PointLight();
        scene.add(light);
        frame(scene);
        light.removeFromParent();
        expect(frame(scene)).toEqual(['destroyLight']);
    });
});

describe('the light walk (T2.1)', () => {
    const CANDLES = 1024;
    let scene: Scene;
    let candles: PointLight[];
    let sun: DirectionalLight;
    const roundRobinFrames = Math.ceil((CANDLES + 1 - ALWAYS_CHECKED_LIGHTS) / DEFAULT_LIGHTS_PER_FRAME);

    beforeEach(() => {
        scene = new Scene();
        candles = [];
        for (let index = 0; index < CANDLES; index++) {
            const candle = new PointLight(0xff3c00, 1, 0, 2);
            candle.position.set(index % 32, 1, Math.floor(index / 32));
            candles.push(candle);
            scene.add(candle);
        }
        // Found after the candles: past the first 64 in the light list.
        sun = new DirectionalLight(0xffffff, 3);
        scene.add(sun);
        frame(scene);
    });

    /** Frames until the next call shows up, up to `limit`. */
    function framesUntilSent(limit: number): { frames: number; calls: string[] } {
        for (let count = 1; count <= limit; count++) {
            const calls = frame(scene);
            if (calls.length > 0) return { frames: count, calls };
        }
        return { frames: Infinity, calls: [] };
    }

    it('checks a bounded number of lights in a still frame, however many there are', () => {
        expect(api.lights.size).toBe(CANDLES + 1);
        expect(frame(scene)).toEqual([]);
        // The first 64, the sun (directional lights are always checked) and this frame's share of the rest.
        expect(sync.report.lightsChecked).toBe(ALWAYS_CHECKED_LIGHTS + 1 + DEFAULT_LIGHTS_PER_FRAME);
    });

    it('sends a change to one of the first 64 lights, or to a directional light, the next frame', () => {
        candles[10].intensity = 2;
        expect(frame(scene)).toEqual(['updateLight intensity']);
        sun.intensity = 4;
        expect(frame(scene)).toEqual(['updateLight intensity']);
    });

    it('sees an unmarked change past the first 64 within one round of the round robin, then every frame while it is hot', () => {
        const candle = candles[700];
        candle.intensity = 0.5;
        const seen = framesUntilSent(roundRobinFrames + 1);
        expect(seen.calls).toEqual(['updateLight intensity']);
        expect(seen.frames).toBeLessThanOrEqual(roundRobinFrames);
        // Hot now: a flicker reaches GI every frame.
        for (let step = 0; step < 5; step++) {
            candle.intensity = 0.6 + step * 0.1;
            expect(frame(scene)).toEqual(['updateLight intensity']);
        }
        // Still for HOT_FRAMES, it goes back to the round robin.
        for (let step = 0; step <= HOT_FRAMES; step++) frame(scene);
        expect(sync.report.lightsChecked).toBe(ALWAYS_CHECKED_LIGHTS + 1 + DEFAULT_LIGHTS_PER_FRAME);
    });

    it('sends a marked light the next update, wherever it is in the list', () => {
        const candle = candles[900];
        candle.position.y = 2;
        candle.updateMatrixWorld();
        sync.markChanged(candle);
        expect(frame(scene)).toEqual(['updateLight position']);
        candle.position.y = 2.5;
        expect(frame(scene)).toEqual(['updateLight position']);
    });

    it('sees lights past the first 64 hidden or removed within one round', () => {
        candles[600].visible = false;
        const hidden = framesUntilSent(roundRobinFrames);
        expect(hidden.calls).toEqual(['updateLight intensity']);
        expect([...api.lights.values()].filter((light) => light.intensity === 0)).toHaveLength(1);
        candles[800].removeFromParent();
        const removed = framesUntilSent(roundRobinFrames);
        expect(removed.calls).toEqual(['destroyLight']);
        expect(api.lights.size).toBe(CANDLES);
    });

    it('takes the sun from the directional lights even when they are far down the list', () => {
        const sky = new Group() as Group & Record<string, unknown>;
        Object.assign(sky, { isSkyMesh: true, turbidity: { value: 2 }, sunPosition: { value: { x: 0, y: 1, z: 0 } } });
        scene.add(sky);
        sync.markChanged(sky);
        frame(scene);
        expect(api.environment?.type === 'procedural' && api.environment.sunIlluminance).toBeCloseTo(3);
        sun.visible = false;
        frame(scene);
        // No visible sun left: the sky takes none.
        expect(api.environment?.type === 'procedural' && api.environment.sunIlluminance).not.toBeCloseTo(3);
    });
});

/**
 * The adapter's `childadded` and `childremoved` listeners left on a tree. three keeps listeners in an
 * EventDispatcher's `_listeners` (EventDispatcher.js), and gives no public way to count them.
 */
function structureListeners(tree: Object3D): number {
    let count = 0;
    tree.traverse((object) => {
        const listeners = (object as unknown as { _listeners?: Record<string, unknown[]> })._listeners;
        count += (listeners?.childadded?.length ?? 0) + (listeners?.childremoved?.length ?? 0);
    });
    return count;
}

describe('three reports changes itself (T3.7)', () => {
    /** A scene of `count` meshes under one group, walked whole once, with a walk of one object a frame: the walk alone would take a pass of count + 2 frames. */
    function slowWalkScene(count: number): { sync: SceneSync; scene: Scene; group: Group; meshes: Mesh[] } {
        const slow = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 1, editChecksPerFrame: 1 });
        const scene = new Scene();
        const group = new Group();
        scene.add(group);
        const geometry = new BoxGeometry();
        const material = new MeshStandardMaterial();
        const meshes: Mesh[] = [];
        for (let index = 0; index < count; index++) {
            const mesh = new Mesh(geometry, material);
            mesh.position.x = index;
            group.add(mesh);
            meshes.push(mesh);
        }
        scene.updateMatrixWorld();
        slow.update(scene);
        api.take();
        return { sync: slow, scene, group, meshes };
    }

    /** One frame in the documented order: three's render (it updates world matrices) has run since the last update. */
    function renderThenUpdate(slow: SceneSync, scene: Scene): string[] {
        scene.updateMatrixWorld();
        slow.update(scene);
        return api.take();
    }

    it('creates an added object in the next update, wherever it is, without markChanged', () => {
        const { sync: slow, scene, meshes } = slowWalkScene(200);
        const added = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        // Deep in the tree: under one of the meshes, which the walk wouldn't reach for about 200 frames.
        meshes[150].add(added);
        expect(renderThenUpdate(slow, scene)).toEqual(['createMesh', 'createMaterial', 'createInstance']);
        // A whole subtree added at once.
        const subtree = new Group();
        subtree.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()), new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        scene.add(subtree);
        expect(renderThenUpdate(slow, scene).filter((call) => call === 'createInstance')).toHaveLength(2);
        // Children added to that subtree later are heard too: the listeners followed it in.
        subtree.children[0].add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        expect(renderThenUpdate(slow, scene).filter((call) => call === 'createInstance')).toHaveLength(1);
        expect(api.instances.size).toBe(200 + 1 + 3);
    });

    it('destroys a removed object in the next update, and keeps one moved between parents', () => {
        const { sync: slow, scene, group, meshes } = slowWalkScene(200);
        meshes[120].removeFromParent();
        expect(renderThenUpdate(slow, scene)).toEqual(['destroyInstance']);
        // attach() removes and adds in one call: the object stays in GI, at its new place.
        const holder = new Group();
        holder.position.y = 10;
        scene.add(holder);
        renderThenUpdate(slow, scene);
        holder.attach(meshes[30]);
        expect(renderThenUpdate(slow, scene)).not.toContain('destroyInstance');
        // A cleared group takes its meshes with it.
        group.clear();
        const calls = renderThenUpdate(slow, scene);
        expect(calls.filter((call) => call === 'destroyInstance')).toHaveLength(198);
        expect(api.instances.size).toBe(1);
    });

    it('does not bring back a removed object the walk had already queued', () => {
        const slow = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 2 });
        const scene = new Scene();
        const geometry = new BoxGeometry();
        const material = new MeshStandardMaterial();
        const meshes = Array.from({ length: 10 }, () => new Mesh(geometry, material));
        scene.add(...meshes);
        scene.updateMatrixWorld();
        slow.update(scene);
        // Two frames: the walk's stack now holds the scene's remaining children, including meshes[9].
        slow.update(scene);
        slow.update(scene);
        api.take();
        meshes[9].removeFromParent();
        const calls: string[] = [];
        for (let frame = 0; frame < 12; frame++) calls.push(...renderThenUpdate(slow, scene));
        expect(calls).toEqual(['destroyInstance']);
        expect(api.instances.size).toBe(9);
    });

    it('sends a still object\'s first unmarked move in the next update, through its hook', () => {
        const { sync: slow, scene, group, meshes } = slowWalkScene(200);
        meshes[170].position.y = 2;
        expect(renderThenUpdate(slow, scene)).toEqual(['setTransforms']);
        expect(slow.report.transforms).toBe(1);
        // A moved parent: each of its meshes reports its own move.
        group.position.z = 1;
        renderThenUpdate(slow, scene);
        expect(slow.report.transforms).toBe(200);
        // A matrix written outside updateMatrixWorld (getWorldPosition calls updateWorldMatrix) still differs
        // from the matrix last sent, so the render's update after it reports it.
        meshes[60].position.y = 3;
        meshes[60].getWorldPosition(new Vector3());
        expect(renderThenUpdate(slow, scene)).toEqual(['setTransforms']);
    });

    it('falls back to the walk for transforms when the scene leaves its matrices to the host', () => {
        const { sync: slow, scene, meshes } = slowWalkScene(20);
        scene.matrixWorldAutoUpdate = false;
        // A matrix set by hand, with no updateMatrixWorld call: no hook can see it, the walk must.
        meshes[10].matrixWorld.setPosition(0, 5, 0);
        const calls: string[] = [];
        for (let frame = 0; frame < 2 * 22; frame++) {
            slow.update(scene);
            calls.push(...api.take());
        }
        expect(calls).toContain('setTransforms');
    });

    it('puts the original updateMatrixWorld back by assignment, never by delete', () => {
        const { sync: slow, scene, meshes } = slowWalkScene(3);
        const own = (object: Object3D) => Object.prototype.hasOwnProperty.call(object, 'updateMatrixWorld');
        expect(own(meshes[0])).toBe(true);
        expect(meshes[0].updateMatrixWorld).not.toBe(Object3D.prototype.updateMatrixWorld);
        meshes[0].removeFromParent();
        renderThenUpdate(slow, scene);
        // Still an own property (deleting it made three 3x slower, T3.0), but three's method again.
        expect(own(meshes[0])).toBe(true);
        expect(meshes[0].updateMatrixWorld).toBe(Object3D.prototype.updateMatrixWorld);
        slow.dispose();
        expect(meshes.every((mesh) => mesh.updateMatrixWorld === Object3D.prototype.updateMatrixWorld)).toBe(true);
        // Disposed: no listener is left on the scene's objects. (An update can't show it: dispose forgets the
        // scene, so the next update would walk it whole again, and marks outside the walked scene do nothing.)
        expect(structureListeners(scene)).toBe(0);
    });

    it('stops listening to a scene it no longer walks, and hears the new one', () => {
        const { sync: slow, scene: first } = slowWalkScene(3);
        const second = new Scene();
        slow.update(second);
        api.take();
        expect(structureListeners(first)).toBe(0);
        // The first scene's changes reach nothing; the second's are created in the next update.
        first.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        second.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        expect(renderThenUpdate(slow, second)).toEqual(['createMesh', 'createMaterial', 'createInstance']);
        expect(api.instances.size).toBe(1);
    });

    it('keeps an object removed and added again before the next update, and sends its move', () => {
        const { sync: slow, scene, group, meshes } = slowWalkScene(10);
        const created = api.instances.size;
        meshes[4].removeFromParent();
        meshes[4].position.y = 7;
        group.add(meshes[4]);
        expect(renderThenUpdate(slow, scene)).toEqual(['setTransforms']);
        expect(api.instances.size).toBe(created);
        // Still heard after the round trip: listeners and hook were put back on it.
        meshes[4].add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        meshes[4].position.y = 8;
        expect(renderThenUpdate(slow, scene)).toEqual(['createMesh', 'createMaterial', 'createInstance', 'setTransforms']);
    });

    it("wraps a SkinnedMesh's own updateMatrixWorld, so three's skinning bookkeeping still runs", () => {
        const scene = new Scene();
        const skinned = new SkinnedMesh(new BoxGeometry(), new MeshStandardMaterial());
        skinned.bind(new Skeleton([]));
        scene.add(skinned);
        frame(scene);
        expect(skinned.updateMatrixWorld).not.toBe(SkinnedMesh.prototype.updateMatrixWorld);
        // SkinnedMesh.updateMatrixWorld keeps bindMatrixInverse at the inverse of matrixWorld (attached mode).
        skinned.position.set(1, 2, 3);
        expect(frame(scene)).toEqual(['setTransforms']);
        expect(skinned.bindMatrixInverse.equals(skinned.matrixWorld.clone().invert())).toBe(true);
        sync.dispose();
        expect(skinned.updateMatrixWorld).toBe(SkinnedMesh.prototype.updateMatrixWorld);
    });

    it('trusts update ranges on the first edit of an InstancedMesh the walk is far from', () => {
        const { sync: slow, scene } = slowWalkScene(200);
        const count = 50;
        const instanced = new InstancedMesh(new BoxGeometry(), new MeshStandardMaterial(), count);
        for (let index = 0; index < count; index++) instanced.setMatrixAt(index, new Matrix4().makeTranslation(index, 0, 0));
        scene.add(instanced);
        renderThenUpdate(slow, scene);
        const handles = [...api.instances.keys()].slice(-count);
        // Let it go cold: the hot set drops it after HOT_FRAMES still frames.
        for (let frame = 0; frame < HOT_FRAMES + 2; frame++) renderThenUpdate(slow, scene);
        // Documented order: update, then render; the host edits after the render, naming instances 7 and 8.
        // Instance 30 changes outside the ranges: the ranges say it didn't, so it isn't even looked at.
        for (const index of [7, 8, 30]) instanced.setMatrixAt(index, new Matrix4().makeTranslation(index, 4, 0));
        instanced.instanceMatrix.addUpdateRange(7 * 16, 2 * 16);
        instanced.instanceMatrix.needsUpdate = true;
        slow.update(scene);
        expect(api.take()).toEqual(['setTransforms']);
        expect(api.transformBatches.at(-1)).toEqual([handles[7], handles[8]]);
    });
});

describe('deforming meshes (T3.7)', () => {
    /** A 1 x 1 sheet in the xy plane, `cells` squares a side, with a morph target that bends its top half forward (z). */
    function morphedSheet(cells: number): Mesh {
        const geometry = new PlaneGeometry(1, 1, cells, cells);
        const position = geometry.getAttribute('position');
        const offsets = new Float32Array(position.count * 3);
        for (let vertex = 0; vertex < position.count; vertex++) offsets[vertex * 3 + 2] = Math.max(0, position.getY(vertex)) * 2;
        geometry.morphAttributes.position = [new Float32BufferAttribute(offsets, 3)];
        geometry.morphTargetsRelative = true;
        const mesh = new Mesh(geometry, new MeshStandardMaterial());
        mesh.updateMorphTargets();
        return mesh;
    }

    /** A 0.2 x 2 x 0.2 column of 8 segments, skinned to two bones: the lower half to the root, the upper half to a bone at mid-height. */
    function skinnedColumn(): { mesh: SkinnedMesh; root: Bone; upper: Bone } {
        const geometry = new BoxGeometry(0.2, 2, 0.2, 1, 8, 1);
        const position = geometry.getAttribute('position');
        const skinIndex: number[] = [];
        const skinWeight: number[] = [];
        for (let vertex = 0; vertex < position.count; vertex++) {
            // A soft joint: the weights blend across the middle segments.
            const upperWeight = Math.min(1, Math.max(0, (position.getY(vertex) + 0.25) / 0.5));
            skinIndex.push(0, 1, 0, 0);
            skinWeight.push(1 - upperWeight, upperWeight, 0, 0);
        }
        geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndex, 4));
        geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeight, 4));
        const root = new Bone();
        root.position.y = -1;
        const upper = new Bone();
        upper.position.y = 1;
        root.add(upper);
        const mesh = new SkinnedMesh(geometry, new MeshStandardMaterial());
        mesh.add(root);
        mesh.bind(new Skeleton([root, upper]));
        return { mesh, root, upper };
    }

    /** three's own CPU pose for every vertex (Mesh.getVertexPosition, with skinning for a SkinnedMesh). */
    function threePose(mesh: Mesh): Float32Array {
        const position = mesh.geometry.getAttribute('position');
        const out = new Float32Array(position.count * 3);
        const vertex = new Vector3();
        for (let index = 0; index < position.count; index++) mesh.getVertexPosition(index, vertex).toArray(out, index * 3);
        return out;
    }

    function largestDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
        let largest = 0;
        for (let index = 0; index < a.length; index++) largest = Math.max(largest, Math.abs(a[index] - b[index]));
        return largest;
    }

    it('creates a morphed mesh deforming, with a box that holds every pose, and sends its poses', () => {
        const scene = new Scene();
        const sheet = morphedSheet(4);
        scene.add(sheet);
        expect(frame(scene)).not.toContain('setDeformedPositions');
        const descriptor = [...api.meshes.values()][0];
        expect(descriptor.deforming).toBe(true);
        // Exact for influences in 0..1: the rest box, plus the top edge's reach forward (2 x 0.5 m).
        expect(descriptor.deformBounds).toEqual({ min: [-0.5, -0.5, 0], max: [0.5, 0.5, 1] });
        sheet.morphTargetInfluences![0] = 0.5;
        expect(frame(scene)).toEqual(['setDeformedPositions']);
        expect(largestDifference(api.poses.at(-1)!.positions, threePose(sheet))).toBeLessThan(1e-6);
        // A still pose sends nothing.
        expect(frame(scene)).toEqual([]);
        expect(sync.report.poses).toBe(0);
    });

    it('sends a skinned mesh that contributes to GI as three poses it, and leaves a character in its bind pose', () => {
        const scene = new Scene();
        const tree = skinnedColumn();
        tree.mesh.userData.giContributes = true;
        const character = skinnedColumn();
        scene.add(tree.mesh, character.mesh);
        frame(scene);
        const deforming = [...api.meshes.values()].filter((mesh) => mesh.deforming);
        expect(deforming).toHaveLength(1);
        // The default box: the bind pose's, grown by half its largest side (2 m) on every side.
        expect(deforming[0].deformBounds!.min[1]).toBeCloseTo(-2);
        expect(deforming[0].deformBounds!.max[2]).toBeCloseTo(1.1);
        tree.upper.rotation.x = Math.PI / 2;
        character.upper.rotation.x = Math.PI / 2;
        expect(frame(scene)).toEqual(['setDeformedPositions']);
        expect(largestDifference(api.poses.at(-1)!.positions, threePose(tree.mesh))).toBeLessThan(1e-5);
        // The bent top really moved: by about a metre.
        expect(largestDifference(api.poses.at(-1)!.positions, tree.mesh.geometry.getAttribute('position').array)).toBeGreaterThan(0.9);
    });

    it('does not send a pose when a skinned rig only moves as a whole', () => {
        const scene = new Scene();
        const rig = new Group();
        const tree = skinnedColumn();
        tree.mesh.userData.giContributes = true;
        rig.add(tree.mesh);
        scene.add(rig);
        frame(scene);
        tree.upper.rotation.x = 0.3;
        expect(frame(scene)).toContain('setDeformedPositions');
        rig.position.set(3, 0, 1);
        rig.rotation.y = 0.7;
        // The bones' world matrices all changed, but the mesh-space pose didn't: only the transform goes out.
        expect(frame(scene)).toEqual(['setTransforms']);
    });

    it('sends a CPU-animated mesh (giDeforming) as poses, not new shapes, and a pose for each mesh sharing a geometry', () => {
        const scene = new Scene();
        const geometry = new PlaneGeometry(1, 1, 2, 2);
        const left = new Mesh(geometry, new MeshStandardMaterial());
        const right = new Mesh(geometry, new MeshStandardMaterial());
        left.userData.giDeforming = true;
        right.userData.giDeforming = true;
        right.userData.giDeformBounds = { min: [-1, -1, -1], max: [1, 1, 1] };
        scene.add(left, right);
        frame(scene);
        expect([...api.meshes.values()].map((mesh) => mesh.deforming)).toEqual([true, true]);
        expect([...api.meshes.values()][1].deformBounds).toEqual({ min: [-1, -1, -1], max: [1, 1, 1] });
        const position = geometry.getAttribute('position') as BufferAttribute;
        position.setZ(4, 0.3);
        position.needsUpdate = true;
        // Each mesh gets the new pose; the geometry edit isn't also sent as a new shape.
        const calls: string[] = [];
        for (let frame = 0; frame < 5; frame++) {
            sync.update(scene);
            calls.push(...api.take());
        }
        expect(calls).toEqual(['setDeformedPositions', 'setDeformedPositions']);
        expect(new Set(api.poses.map((pose) => pose.mesh)).size).toBe(2);
    });

    it('starts an object over when giDeforming changes, as for giContributes', () => {
        const scene = new Scene();
        const sheet = morphedSheet(2);
        scene.add(sheet);
        frame(scene);
        expect([...api.meshes.values()].map((mesh) => mesh.deforming === true)).toEqual([true]);
        sheet.userData.giDeforming = false;
        // Found by the edit rotation (this small scene is covered every frame), restarted the update after.
        frame(scene);
        frame(scene);
        expect([...api.meshes.values()].map((mesh) => mesh.deforming === true)).toEqual([false]);
        sheet.userData.giDeforming = true;
        sync.markChanged(sheet);
        frame(scene);
        expect([...api.meshes.values()].map((mesh) => mesh.deforming === true)).toEqual([true]);
        expect(api.instances.size).toBe(1);
    });

    it('keeps a mesh with giDeforming false in its bind pose, and sends no poses while hidden', () => {
        const scene = new Scene();
        const still = morphedSheet(2);
        still.userData.giDeforming = false;
        const hidden = morphedSheet(2);
        scene.add(still, hidden);
        frame(scene);
        expect([...api.meshes.values()].map((mesh) => mesh.deforming === true)).toEqual([false, true]);
        hidden.visible = false;
        frame(scene);
        hidden.morphTargetInfluences![0] = 1;
        still.morphTargetInfluences![0] = 1;
        expect(frame(scene)).toEqual([]);
        // Shown again: the pose it missed goes out.
        hidden.visible = true;
        expect(frame(scene)).toEqual(['setVisible', 'setDeformedPositions']);
    });
});
