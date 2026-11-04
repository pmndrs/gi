import {
    BoxGeometry,
    BufferAttribute,
    BufferGeometry,
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
    // Every frame, so each test sees an edit on the next frame; the spreading has its own test.
    sync = new SceneSync(api, { device: NO_DEVICE, warn: (message) => warnings.push(message), editCheckInterval: 1 });
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
        expect(sync.report).toEqual({ created: 0, destroyed: 0, transforms: 0, visibility: 0, materials: 0, meshes: 0, lights: 0 });
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
    it('spreads edit checks over frames but never misses or repeats an edit', () => {
        const spread = new SceneSync(api, { device: NO_DEVICE, editCheckInterval: 4 });
        const scene = new Scene();
        const materials = Array.from({ length: 8 }, () => new MeshStandardMaterial());
        for (const material of materials) scene.add(new Mesh(new BoxGeometry(), material));
        scene.updateMatrixWorld();
        spread.update(scene);
        api.take();
        for (const material of materials) material.color.setRGB(0.1, 0.2, 0.3);
        const updatesPerFrame: number[] = [];
        for (let frame = 0; frame < 6; frame++) {
            spread.update(scene);
            updatesPerFrame.push(api.take().filter((call) => call === 'updateMaterial').length);
        }
        // Every material once within 4 frames, a share of them each frame, none twice.
        expect(updatesPerFrame.slice(0, 4).reduce((sum, count) => sum + count, 0)).toBe(8);
        expect(Math.max(...updatesPerFrame)).toBeLessThan(8);
        expect(updatesPerFrame.slice(4)).toEqual([0, 0]);
    });

    it('sends moves every frame whatever the edit-check interval', () => {
        const spread = new SceneSync(api, { device: NO_DEVICE, editCheckInterval: 8 });
        const scene = new Scene();
        const mesh = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        scene.add(mesh);
        scene.updateMatrixWorld();
        spread.update(scene);
        api.take();
        for (let frame = 1; frame <= 3; frame++) {
            mesh.position.x = frame;
            scene.updateMatrixWorld();
            spread.update(scene);
            expect(api.take()).toEqual(['setTransform']);
        }
    });

    it('copes with children reordered, removed and added back', () => {
        const scene = new Scene();
        const meshes = Array.from({ length: 4 }, () => new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        scene.add(...meshes);
        frame(scene);
        scene.children.reverse();
        expect(frame(scene)).toEqual([]);
        const [first] = scene.children;
        first.removeFromParent();
        expect(frame(scene)).toEqual(['destroyInstance', 'destroyMaterial', 'destroyMesh']);
        scene.add(first);
        expect(frame(scene)).toEqual(['createMesh', 'createMaterial', 'createInstance']);
        expect(frame(scene)).toEqual([]);
        expect(api.instances.size).toBe(4);
    });

    it('rejects an edit-check interval below one frame', () => {
        expect(() => new SceneSync(api, { device: NO_DEVICE, editCheckInterval: 0 })).toThrow(RangeError);
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
