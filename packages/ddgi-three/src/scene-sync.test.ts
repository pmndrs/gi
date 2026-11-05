import {
    BoxGeometry,
    BufferAttribute,
    BufferGeometry,
    DataTexture,
    Color,
    DirectionalLight,
    Group,
    InstancedMesh,
    Matrix4,
    Mesh,
    MeshStandardMaterial,
    PointLight,
    Scene,
    SkinnedMesh,
    Skeleton,
    Texture,
} from 'three';
import { beforeEach, describe, expect, it } from 'vitest';
import type {
    Aabb,
    EnvironmentDescriptor,
    InstanceDescriptor,
    InstanceHandle,
    LightChanges,
    LightDescriptor,
    LightHandle,
    Mat4,
    MaterialDescriptor,
    MaterialHandle,
    MeshDescriptor,
    MeshHandle,
    SceneApi,
    TextureDescriptor,
    TextureHandle,
} from '@pmndrs/ddgi-core';
import { SceneSync } from './scene-sync.ts';

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
    setDeformedPositions(): void {
        throw new Error('not used');
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
        expect(frame(scene)).toEqual(['setTransform']);
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
        expect(frame(scene)).toEqual(['setTransform']);
    });

    it('sends a colour edit (three bumps no version for it), with metalness folded in', () => {
        material.color.setRGB(1, 0, 0);
        material.metalness = 0.5;
        expect(frame(scene)).toEqual(['updateMaterial']);
        const descriptor = [...api.materials.values()][0];
        expect(descriptor.albedo).toEqual([0.5, 0, 0]);
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
            if (api.take().includes('setTransform')) break;
        }
        expect(framesUntilSeen).toBeLessThan(Math.ceil(101 / 16));
        // Now hot: each further move is sent in the frame it happens.
        for (let frame = 2; frame < 6; frame++) {
            moving.position.y = frame;
            scene.updateMatrixWorld();
            bounded.update(scene);
            expect(api.take()).toEqual(['setTransform']);
        }
    });

    it('picks up a marked object in the same update: moved, added and removed', () => {
        const bounded = new SceneSync(api, { device: NO_DEVICE, objectsPerFrame: 1 });
        const { scene, meshes } = rowScene(bounded, 50);
        meshes[30].position.y = 3;
        scene.updateMatrixWorld();
        bounded.markChanged(meshes[30]);
        bounded.update(scene);
        expect(api.take()).toContain('setTransform');

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

    it('folds a texture\'s average colour into the GI albedo, once per texture version', () => {
        const scene = new Scene();
        // Half red, half blue, stored linear.
        const pixels = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
        const map = new DataTexture(pixels, 2, 1);
        map.needsUpdate = true;
        const material = new MeshStandardMaterial({ color: new Color().setRGB(1, 1, 0.5), map });
        scene.add(new Mesh(new BoxGeometry(), material));
        frame(scene);
        const albedo = [...api.materials.values()][0].albedo!;
        expect(albedo[0]).toBeCloseTo(0.5);
        expect(albedo[1]).toBeCloseTo(0);
        expect(albedo[2]).toBeCloseTo(0.25);
        // New pixels (a new version): the average is made again.
        pixels.set([0, 255, 0, 255, 0, 255, 0, 255]);
        map.needsUpdate = true;
        frame(scene);
        expect([...api.materials.values()][0].albedo![1]).toBeCloseTo(1);
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

describe('instances', () => {
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
        expect(frame(scene)).toEqual(['setTransform']);

        instanced.count = 2;
        expect(frame(scene)).toEqual(['setVisible']);
        expect([...api.instances.values()][2].visible).toBe(false);
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
