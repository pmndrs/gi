/**
 * The scene copy on a fake device: what reaches the two buffers is read back
 * on the CPU (testing/scene-reader.ts mirrors the WGSL accessors) and compared
 * with the inputs. The GPU page `tools/gpu/pages/scene-copy` runs the same
 * scenes through the real WGSL.
 */

import { describe, expect, it } from 'vitest';
import type { MaterialHandle } from '../contract/handles.ts';
import { PlatformScene, type SceneUpload } from './platform-scene.ts';
import { createFakeGpu, type FakeBuffer, type FakeGpu } from './testing/fake-gpu.ts';
import { buildBasicScene, checkReadback, churnScene, composeTransform, cubeMesh, editBasicScene, makeQueries, materialIndices, SceneModel } from './testing/fixture.ts';
import { SceneReader } from './testing/scene-reader.ts';
import { createRandom } from '../trace/oracle-fixtures.ts';

/** Tables and vertex space small enough that the basic scene and churn reallocate everything. */
const TINY = { instances: 2, meshes: 2, materials: 2, lights: 1, materialListWords: 4, vertexWords: 64 };

function setup(initialCapacities = {}): { gpu: FakeGpu; scene: PlatformScene; model: SceneModel } {
    const gpu = createFakeGpu();
    const scene = new PlatformScene(gpu.device, { initialCapacities });
    return { gpu, scene, model: new SceneModel(scene) };
}

function encodeAndSubmit(gpu: FakeGpu, scene: PlatformScene): SceneUpload {
    const encoder = gpu.createEncoder();
    const upload = scene.encode(encoder);
    gpu.submit(encoder);
    return upload;
}

function readBack(scene: PlatformScene, model: SceneModel): string[] {
    const [vertices, tables] = scene.module.resources().map((resource) => (resource as GPUBufferBinding).buffer as unknown as FakeBuffer);
    const reader = new SceneReader(vertices.bytes, tables.bytes);
    const queries = makeQueries(model);
    const materials = materialIndices(model);
    return checkReadback(model, queries, materials, reader.readAll(queries, materials));
}

describe('scene copy accessors against the inputs', () => {
    it('reads back every accessor of the basic scene', () => {
        const { gpu, scene, model } = setup();
        buildBasicScene(model);
        encodeAndSubmit(gpu, scene);
        expect(readBack(scene, model)).toEqual([]);
    });

    it('reads back correctly after edits', () => {
        const { gpu, scene, model } = setup();
        const basic = buildBasicScene(model);
        encodeAndSubmit(gpu, scene);
        editBasicScene(model, basic);
        encodeAndSubmit(gpu, scene);
        expect(readBack(scene, model)).toEqual([]);
    });

    it('reads back correctly after every buffer was reallocated, with uploads every round', () => {
        const { gpu, scene, model } = setup(TINY);
        buildBasicScene(model);
        churnScene(model, 4, 11, () => encodeAndSubmit(gpu, scene));
        encodeAndSubmit(gpu, scene);
        expect(readBack(scene, model)).toEqual([]);
    });

    it('reads back correctly when everything arrives in one frame', () => {
        const { gpu, scene, model } = setup(TINY);
        const basic = buildBasicScene(model);
        editBasicScene(model, basic);
        churnScene(model, 3);
        encodeAndSubmit(gpu, scene);
        expect(readBack(scene, model)).toEqual([]);
    });
});

describe('adding and removing never recompiles', () => {
    it('creates no shader modules or pipelines, keeps its WGSL and layout, and bumps version on reallocation', () => {
        const { gpu, scene, model } = setup(TINY);
        const codeBefore = scene.module.code({ group: 0, firstBinding: 3 });
        const entriesBefore = JSON.stringify(scene.module.entries);
        const versionBefore = scene.module.version;
        let reallocations = 0;
        buildBasicScene(model);
        churnScene(model, 3, 5, () => {
            if (encodeAndSubmit(gpu, scene).reallocated) reallocations++;
        });
        encodeAndSubmit(gpu, scene);
        expect(gpu.counters.shaderModules).toBe(0);
        expect(gpu.counters.pipelines).toBe(0);
        expect(scene.module.code({ group: 0, firstBinding: 3 })).toBe(codeBefore);
        expect(JSON.stringify(scene.module.entries)).toBe(entriesBefore);
        expect(scene.module.storageBufferCount).toBe(2);
        expect(reallocations).toBeGreaterThan(0);
        expect(scene.module.version).toBeGreaterThan(versionBefore);
    });

    it('destroys replaced buffers at the next encode, not before', () => {
        const { gpu, scene, model } = setup(TINY);
        buildBasicScene(model);
        const firstBuffers = scene.module.resources().map((resource) => (resource as GPUBufferBinding).buffer as unknown as FakeBuffer);
        const encoder = gpu.createEncoder();
        expect(scene.encode(encoder).reallocated).toBe(true);
        // The copy out of the old vertex buffer runs at submit, so it must still exist now.
        expect(firstBuffers.every((buffer) => !buffer.destroyed)).toBe(true);
        gpu.submit(encoder);
        encodeAndSubmit(gpu, scene);
        expect(firstBuffers.every((buffer) => buffer.destroyed)).toBe(true);
    });

    it('uploads nothing when nothing changed', () => {
        const { gpu, scene, model } = setup();
        buildBasicScene(model);
        expect(encodeAndSubmit(gpu, scene).bytesWritten).toBeGreaterThan(0);
        const still = encodeAndSubmit(gpu, scene);
        expect(still.bytesWritten).toBe(0);
        expect(still.changes.regions).toEqual([]);
        expect(still.changes.global).toEqual([]);
    });
});

describe('memory and counts', () => {
    it('reports the GPU bytes it holds and the live objects', () => {
        const { gpu, scene, model } = setup();
        const basic = buildBasicScene(model);
        encodeAndSubmit(gpu, scene);
        const live = gpu.buffers.filter((buffer) => !buffer.destroyed);
        expect(scene.memoryBytes).toBe(live.reduce((sum, buffer) => sum + buffer.size, 0));
        expect(scene.memory.vertexBytesUsed).toBeGreaterThan(70000 * 12); // the wide mesh's positions alone
        expect(scene.counts).toEqual({ meshes: 4, instances: 8, triangles: 12 + 41 + 7 + 1, lights: 5, materials: 5 });
        model.destroyInstance(basic.instances[3]);
        model.destroyMesh(basic.meshes.wide);
        encodeAndSubmit(gpu, scene);
        expect(scene.counts.triangles).toBe(12 + 7 + 1);
        expect(scene.memory.vertexBytesUsed).toBeLessThan(70000 * 12);
    });
});

describe('changes and bounds', () => {
    it('reports where things moved, and global changes', () => {
        const { gpu, scene, model } = setup();
        const basic = buildBasicScene(model);
        const first = encodeAndSubmit(gpu, scene);
        expect(first.instancesChanged).toBe(true);
        expect(first.meshesUploaded.map((mesh) => mesh.slot).sort()).toEqual([0, 1, 2, 3]);
        expect(first.changes.global).toEqual(expect.arrayContaining(['sun', 'lights', 'environment']));

        model.setTransform(basic.instances[0], composeTransform([10, 0, 0], [0, 1, 0], 0, [1, 1, 1]));
        const moved = encodeAndSubmit(gpu, scene);
        expect(moved.instancesChanged).toBe(true);
        expect(moved.changes.regions).toHaveLength(2); // where it was, and where it is
        expect(moved.changes.regions[1].bounds.min[0]).toBeCloseTo(9.5, 4);

        // A receiver (contributesToGI: false) moving changes no GI.
        model.setTransform(basic.instances[5], composeTransform([0, 9, 0], [0, 1, 0], 0, [1, 1, 1]));
        const receiver = encodeAndSubmit(gpu, scene);
        expect(receiver.instancesChanged).toBe(false);
        expect(receiver.changes.regions).toEqual([]);
        expect(receiver.changes.movedReceivers).toBe(1);

        model.updateLight(basic.lights[4], { position: [0, 0, 0] });
        const light = encodeAndSubmit(gpu, scene);
        expect(light.changes.regions.map((region) => region.reason)).toEqual(['light', 'light']);
    });

    it('bounds cover visible instances that contribute, and nothing else', () => {
        const { scene, model } = setup();
        const random = createRandom(2);
        const mesh = model.createMesh(cubeMesh(random));
        const materials: MaterialHandle[] = [model.createMaterial({}), model.createMaterial({})];
        expect(scene.bounds).toEqual({ min: [0, 0, 0], max: [0, 0, 0] });
        model.createInstance({ mesh, materials, transform: composeTransform([1, 2, 3], [0, 1, 0], 0, [2, 2, 2]) });
        model.createInstance({ mesh, materials, transform: composeTransform([100, 0, 0], [0, 1, 0], 0, [1, 1, 1]), contributesToGI: false });
        const hidden = model.createInstance({ mesh, materials, transform: composeTransform([-100, 0, 0], [0, 1, 0], 0, [1, 1, 1]), visible: false });
        expect(scene.bounds.min[0]).toBeCloseTo(0, 4);
        expect(scene.bounds.max[1]).toBeCloseTo(3, 4);
        model.setVisible(hidden, true);
        expect(scene.bounds.min[0]).toBeCloseTo(-100.5, 3);
    });
});

describe('checks at the call', () => {
    it('refuses bad input and objects still in use', () => {
        const { scene, model } = setup();
        const random = createRandom(3);
        const mesh = model.createMesh(cubeMesh(random));
        const material = model.createMaterial({});
        expect(() => scene.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) })).toThrow(/2 material slot/);
        const instance = model.createInstance({ mesh, materials: [material, material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        expect(() => scene.destroyMesh(mesh)).toThrow(/still used/);
        expect(() => scene.destroyMaterial(material)).toThrow(/still used/);
        expect(() => scene.createMesh({ positions: new Float32Array(9), indices: new Uint32Array([0, 1, 3]) })).toThrow(/3 vertices/);
        expect(() => scene.createMesh({ positions: new Float32Array(9), indices: new Uint32Array([0, 1, 2]), normals: new Float32Array(3) })).toThrow(/normals/);
        expect(() => scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 1, position: [0, 0, 0], range: -1 })).toThrow(/range/);
        expect(() => scene.createLight({ type: 'directional', color: [1, 1, 1], intensity: 1, direction: [0, 0, 0] })).toThrow(/direction/);
        expect(() => scene.createLight({ type: 'directional', color: [1, 1, 1], intensity: 1, direction: [0, -1, 0], group: 4 })).toThrow(/group/);
        const light = scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 1, position: [0, 0, 0], range: 0 });
        expect(() => scene.updateLight(light, { type: 'spot' } as never)).toThrow(/change type/);
        model.destroyInstance(instance);
        expect(() => scene.setVisible(instance, false)).toThrow(/destroyed instance/);
        model.destroyMesh(mesh);
        expect(() => scene.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) })).toThrow(/destroyed mesh/);
    });
});

describe('device limits', () => {
    it('throws GILimitError naming the limit when the vertex data outgrows a binding, and keeps the meshes queued', () => {
        const gpu = createFakeGpu({ maxStorageBufferBindingSize: 64 * 1024 });
        const scene = new PlatformScene(gpu.device, { initialCapacities: { vertexWords: 1024 } });
        const random = createRandom(4);
        const model = new SceneModel(scene);
        for (let index = 0; index < 4; index++) model.createMesh({ positions: new Float32Array(3000 * 3).map(() => random()), indices: Uint32Array.from({ length: 3000 }, (_, i) => i) });
        expect(() => encodeAndSubmit(gpu, scene)).toThrow(/maxStorageBufferBindingSize/);
        for (const handle of [...model.meshes.keys()].slice(1)) model.destroyMesh(handle);
        expect(encodeAndSubmit(gpu, scene).meshesUploaded).toHaveLength(1);
    });
});
