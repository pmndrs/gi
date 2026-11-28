/**
 * The scene copy on a fake device: what reaches the two buffers is read back
 * on the CPU (testing/scene-reader.ts mirrors the WGSL accessors) and compared
 * with the inputs. The GPU page `tools/gpu/pages/scene-copy` runs the same
 * scenes through the real WGSL.
 */

import { describe, expect, it } from 'vitest';
import type { InstanceHandle, MaterialHandle } from '../contract/handles.ts';
import type { Aabb, ColorRGB } from '../contract/math.ts';
import type { DirectionalLightDescriptor, LightChanges, LightDescriptor, PointLightDescriptor, SpotLightDescriptor } from '../contract/scene.ts';
import { MAX_LISTED_LIGHT_CHANGES } from './change-log.ts';
import { PlatformScene, type SceneUpload } from './platform-scene.ts';
import { transformFrame } from './instance-transforms.ts';
import { transformBounds } from '../trace/top-level.ts';
import { createFakeGpu, type FakeBuffer, type FakeGpu, type FakeKernel } from './testing/fake-gpu.ts';
import { NONE } from './layout.ts';
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
        expect(first.changes.global).toEqual(expect.arrayContaining(['sun', 'environment']));
        // A light with no range window (range 0) is local now: its influence radius bounds it.
        expect(first.changes.global).not.toContain('lights');
        expect(first.changes.lights.map((change) => change.before)).toEqual([null, null, null, null, null]);
        expect(first.changes.lights.map((change) => change.after?.label)).toEqual(['sun', 'unlimited', 'spot', 'moon', 'candle']);

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
        // It is listed where it is now, for techniques that wake probes near receivers.
        expect(receiver.changes.receivers).toHaveLength(1);
        expect(receiver.changes.receivers[0].min[1]).toBeGreaterThan(8);
        expect(receiver.changes.receivers[0].max[1]).toBeLessThan(10);

        model.updateLight(basic.lights[4], { position: [0, 0, 0] });
        const light = encodeAndSubmit(gpu, scene);
        expect(light.changes.regions.map((region) => region.reason)).toEqual(['light', 'light']);
        expect(light.changes.lights).toHaveLength(1);
        expect(light.changes.lights[0].before).toMatchObject({ type: 'point', position: [-2, 1, 3] });
        expect(light.changes.lights[0].after).toMatchObject({ type: 'point', position: [0, 0, 0] });
        expect(light.changes.lights[0].index).toBe(scene.lightIndex(basic.lights[4]));
        // The candle's range (5 m) caps its radius: next to the sun it would otherwise reach sqrt(5 / (3 / 1000)) = 41 m.
        expect(light.changes.lights[0].influenceRadius).toEqual({ before: 5, after: 5 });
        expect(light.changes.regions[1].bounds).toEqual({ min: [-5, -5, -5], max: [5, 5, 5] });
    });

    it('lists each changed light once a frame, from where the frame began, and leaves out lights that end where they began', () => {
        const { gpu, scene, model } = setup();
        const basic = buildBasicScene(model);
        encodeAndSubmit(gpu, scene);
        const candle = basic.lights[4];
        model.updateLight(candle, { intensity: 10 });
        model.updateLight(candle, { intensity: 20 });
        // The spot goes there and back again in one frame: no change.
        model.updateLight(basic.lights[2], { intensity: 1 });
        model.updateLight(basic.lights[2], { intensity: 50 });
        const edited = encodeAndSubmit(gpu, scene).changes;
        expect(edited.lights).toHaveLength(1);
        expect(edited.lights[0].before?.intensity).toBe(5);
        expect(edited.lights[0].after?.intensity).toBe(20);

        // Updating a light to the state it already has changes nothing.
        model.updateLight(candle, { intensity: 20 });
        expect(encodeAndSubmit(gpu, scene).changes.lights).toEqual([]);

        model.destroyLight(candle);
        const destroyed = encodeAndSubmit(gpu, scene).changes;
        expect(destroyed.lights).toHaveLength(1);
        expect(destroyed.lights[0]).toMatchObject({ index: -1, after: null, influenceRadius: { before: 5, after: 0 } });

        // Created and destroyed in one frame: nothing happened.
        model.destroyLight(model.createLight({ type: 'point', color: [1, 1, 1], intensity: 1, position: [0, 0, 0], range: 1 }));
        expect(encodeAndSubmit(gpu, scene).changes.lights).toEqual([]);
    });

    it('round-trips every LightChanges field of each light type through SceneChanges.lights (T2.1 check 2)', () => {
        // One full state and one changed value per field. The mapped types over each descriptor's
        // own keys make this fail to compile when the contract gains a field the test doesn't cover.
        type Fields<Descriptor> = { [Key in Exclude<keyof Descriptor, 'type'>]-?: Descriptor[Key] };
        const common = { color: [1, 0.9, 0.8] as ColorRGB, intensity: 3, castsShadows: true, group: 1, label: 'first' };
        const changedCommon = { color: [0.2, 0.4, 1] as ColorRGB, intensity: 7, castsShadows: false, group: 3, label: 'second' };
        const directional: Fields<DirectionalLightDescriptor> = { ...common, direction: [0, -1, 0], angularRadius: 0.01 };
        const directionalChanged: Fields<DirectionalLightDescriptor> = { ...changedCommon, direction: [0.6, -0.8, 0], angularRadius: 0.05 };
        const point: Fields<PointLightDescriptor> = { ...common, position: [1, 2, 3], range: 0, radius: 0.1 };
        const pointChanged: Fields<PointLightDescriptor> = { ...changedCommon, position: [-4, 0.5, 2], range: 6, radius: 0.3 };
        const spot: Fields<SpotLightDescriptor> = { ...common, position: [1, 2, 3], direction: [0, -1, 0], range: 8, innerConeAngle: 0.2, outerConeAngle: 0.5, radius: 0.05 };
        const spotChanged: Fields<SpotLightDescriptor> = { ...changedCommon, position: [2, 3, 4], direction: [0, 0, -1], range: 0, innerConeAngle: 0.4, outerConeAngle: 0.9, radius: 0.2 };
        const cases: Array<{ type: LightDescriptor['type']; start: Record<string, unknown>; changed: Record<string, unknown> }> = [
            { type: 'directional', start: directional, changed: directionalChanged },
            { type: 'point', start: point, changed: pointChanged },
            { type: 'spot', start: spot, changed: spotChanged },
        ];

        for (const { type, start, changed } of cases) {
            const { gpu, scene, model } = setup();
            let state = { type, ...start } as LightDescriptor;
            const light = model.createLight(state);
            const created = encodeAndSubmit(gpu, scene).changes.lights;
            expect(created).toEqual([expect.objectContaining({ light, before: null, after: state, index: scene.lightIndex(light) })]);

            // One field at a time: before is the whole old state, after the whole new one, and nothing else moved.
            for (const field of Object.keys(changed)) {
                const next = { ...state, [field]: changed[field] } as LightDescriptor;
                model.updateLight(light, { [field]: changed[field] } as LightChanges);
                const listed = encodeAndSubmit(gpu, scene).changes.lights;
                expect(listed, `${type}.${field}`).toHaveLength(1);
                expect(listed[0].light).toBe(light);
                expect(listed[0].index).toBe(scene.lightIndex(light));
                expect(listed[0].before, `${type}.${field} before`).toEqual(state);
                expect(listed[0].after, `${type}.${field} after`).toEqual(next);
                state = next;
            }

            // Every field back at once, in one change.
            const original = { type, ...start } as LightDescriptor;
            model.updateLight(light, start as LightChanges);
            const together = encodeAndSubmit(gpu, scene).changes.lights;
            expect(together).toHaveLength(1);
            expect(together[0].before).toEqual(state);
            expect(together[0].after).toEqual(original);

            model.destroyLight(light);
            const destroyed = encodeAndSubmit(gpu, scene).changes.lights;
            expect(destroyed).toEqual([expect.objectContaining({ light, before: original, after: null, index: -1 })]);
        }
    });

    it('keeps the old state even when the host edits its arrays in place', () => {
        const { gpu, scene, model } = setup();
        const position: [number, number, number] = [1, 2, 3];
        const lamp = model.createLight({ type: 'point', color: [1, 1, 1], intensity: 4, position, range: 0 });
        encodeAndSubmit(gpu, scene);
        position[0] = 7;
        model.updateLight(lamp, { position });
        const change = encodeAndSubmit(gpu, scene).changes.lights[0];
        expect(change.before).toMatchObject({ position: [1, 2, 3] });
        expect(change.after).toMatchObject({ position: [7, 2, 3] });
    });

    it('flags a group change as global, and a frame with more light changes than are listed as global too', () => {
        const { gpu, scene, model } = setup();
        const lamps = Array.from({ length: MAX_LISTED_LIGHT_CHANGES + 1 }, (_, index) =>
            model.createLight({ type: 'point', color: [1, 1, 1], intensity: 1, position: [index, 0, 0], range: 2 }),
        );
        const created = encodeAndSubmit(gpu, scene).changes;
        expect(created.lightsOverflowed).toBe(true);
        expect(created.lights).toEqual([]);
        expect(created.global).toContain('lights');

        model.updateLight(lamps[0], { group: 2 });
        const regrouped = encodeAndSubmit(gpu, scene).changes;
        expect(regrouped.lightsOverflowed).toBe(false);
        expect(regrouped.lights).toHaveLength(1);
        expect(regrouped.global).toEqual(['lights']);
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

describe('moves (T2.M1)', () => {
    it('works out the world box and the move distance from the frame as the eight corners would', () => {
        const random = createRandom(21);
        const meshBounds = new Float32Array([-0.3, 0.1, -2, 1.7, 0.4, 0.5]);
        const frame = new Float64Array(12);
        const bounds = new Float32Array(6);
        const reference = new Float32Array(6);
        const corners = (matrix: number[]) =>
            Array.from({ length: 8 }, (_, corner) => {
                const point = [corner & 1 ? meshBounds[3] : meshBounds[0], corner & 2 ? meshBounds[4] : meshBounds[1], corner & 4 ? meshBounds[5] : meshBounds[2]];
                return [0, 1, 2].map((axis) => matrix[axis] * point[0] + matrix[4 + axis] * point[1] + matrix[8 + axis] * point[2] + matrix[12 + axis]);
            });
        let previous: number[] | null = null;
        for (let trial = 0; trial < 50; trial++) {
            const matrix = composeTransform([random() * 20 - 10, random() * 5, random() * 20 - 10], [random() - 0.5, random() - 0.5, random() - 0.5], random() * 6, [0.2 + random() * 2, -1 + random() * 2.5, 0.5 + random()]);
            const moved = transformFrame(meshBounds, 0, matrix, 0, frame, 0, bounds, 0);
            transformBounds(meshBounds, matrix, reference, 0);
            for (let index = 0; index < 6; index++) expect(bounds[index]).toBeCloseTo(reference[index], 4);
            if (previous) {
                const before = corners(previous);
                const after = corners(matrix);
                const expected = Math.max(...after.map((corner, index) => Math.hypot(corner[0] - before[index][0], corner[1] - before[index][1], corner[2] - before[index][2])));
                expect(moved).toBeCloseTo(expected, 6);
            }
            previous = matrix;
        }
    });

    /** A cube mesh of known bounds (-0.5 to 0.5) and one material, and a placement of it. */
    function unitCube(model: SceneModel) {
        const half = 0.5;
        const positions = new Float32Array([-half, -half, -half, half, -half, -half, half, half, -half, -half, half, -half, -half, -half, half, half, -half, half, half, half, half, -half, half, half]);
        const indices = new Uint32Array([0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2]);
        return { mesh: model.createMesh({ positions, indices }), material: model.createMaterial({}) };
    }

    it('says how far a moved instance went, as its farthest corner, in both regions, with its slot', () => {
        const { gpu, scene, model } = setup();
        const { mesh, material } = unitCube(model);
        const instance = model.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        const created = encodeAndSubmit(gpu, scene).changes.regions;
        // Created: a whole change there, no move distance.
        expect(created).toHaveLength(1);
        expect(created[0].moved).toBeUndefined();
        expect(created[0].instance).toBe(scene.instanceSlot(instance));

        model.setTransform(instance, composeTransform([1, 0, 0], [0, 1, 0], 0, [1, 1, 1]));
        const slid = encodeAndSubmit(gpu, scene).changes.regions;
        expect(slid.map((region) => region.moved)).toEqual([1, 1]);
        expect(slid.map((region) => region.instance)).toEqual([scene.instanceSlot(instance), scene.instanceSlot(instance)]);

        // A quarter turn in place: the world box is the same, but its corners moved by the chord sqrt(2) x 0.707 = 1.
        model.setTransform(instance, composeTransform([1, 0, 0], [0, 1, 0], Math.PI / 2, [1, 1, 1]));
        const turned = encodeAndSubmit(gpu, scene).changes.regions;
        expect(turned[0].bounds.min[0]).toBeCloseTo(turned[1].bounds.min[0], 5);
        expect(turned[0].moved).toBeCloseTo(1, 5);
    });

    it('says when a moved instance gives off light, from its materials, also in a merged region (T3.4)', () => {
        const { gpu, scene, model } = setup();
        const { mesh, material } = unitCube(model);
        const glowing = model.createMaterial({ emissive: [1, 0.5, 0.1], emissiveIntensity: 3 });
        const switchedOff = model.createMaterial({ emissive: [1, 0.5, 0.1], emissiveIntensity: 0 });
        const plain = model.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        const lamp = model.createInstance({ mesh, materials: [material, glowing], transform: composeTransform([3, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        const dark = model.createInstance({ mesh, materials: [switchedOff], transform: composeTransform([6, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        const created = encodeAndSubmit(gpu, scene).changes.regions;
        expect(created.map((region) => region.emits === true)).toEqual([false, true, false]);

        // setTransform (moveInstance) and setTransforms' slide both carry it, on where it was and where it is.
        model.setTransform(lamp, composeTransform([3, 1, 0], [0, 1, 0], 0, [1, 1, 1]));
        model.setTransform(plain, composeTransform([0, 1, 0], [0, 1, 0], 0, [1, 1, 1]));
        expect(encodeAndSubmit(gpu, scene).changes.regions.map((region) => [region.instance, region.emits === true])).toEqual([
            [scene.instanceSlot(lamp), true],
            [scene.instanceSlot(lamp), true],
            [scene.instanceSlot(plain), false],
            [scene.instanceSlot(plain), false],
        ]);
        scene.setTransforms(Uint32Array.of(lamp, dark), new Float32Array([...composeTransform([3, 2, 0], [0, 1, 0], 0, [1, 1, 1]), ...composeTransform([6, 2, 0], [0, 1, 0], 0, [1, 1, 1])]));
        expect(encodeAndSubmit(gpu, scene).changes.regions.map((region) => region.emits === true)).toEqual([true, true, false, false]);

        // Past MAX_DIRTY_REGIONS the one merged region emits when any of its regions did.
        const crowd = Array.from({ length: 300 }, (_, index) => model.createInstance({ mesh, materials: [material], transform: composeTransform([index, 5, 0], [0, 1, 0], 0, [1, 1, 1]) }));
        encodeAndSubmit(gpu, scene);
        const matrices = new Float32Array((crowd.length + 1) * 16);
        crowd.forEach((_, index) => matrices.set(composeTransform([index, 6, 0], [0, 1, 0], 0, [1, 1, 1]), index * 16));
        matrices.set(composeTransform([3, 3, 0], [0, 1, 0], 0, [1, 1, 1]), crowd.length * 16);
        scene.setTransforms(Uint32Array.from([...crowd, lamp]), matrices);
        const merged = encodeAndSubmit(gpu, scene).changes.regions;
        expect(merged).toHaveLength(1);
        expect(merged[0].emits).toBe(true);
    });

    it('setTransforms moves instances exactly as setTransform does, with the same regions', () => {
        const one = setup();
        const batch = setup();
        const random = createRandom(9);
        const transforms: number[][] = [];
        const handles: { one: InstanceHandle[]; batch: InstanceHandle[] } = { one: [], batch: [] };
        for (const [which, { model }] of [['one', one], ['batch', batch]] as const) {
            const { mesh, material } = unitCube(model);
            for (let index = 0; index < 20; index++) handles[which].push(model.createInstance({ mesh, materials: [material], transform: composeTransform([index, 0, 0], [0, 1, 0], 0, [1, 1, 1]) }));
        }
        encodeAndSubmit(one.gpu, one.scene);
        encodeAndSubmit(batch.gpu, batch.scene);
        // Every other instance moves; f32 values, so both paths see the same numbers.
        const moving = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
        for (const _ of moving) transforms.push(Array.from(new Float32Array(composeTransform([random() * 10, random(), random() * 10], [0, 1, 0], random() * 3, [1 + random(), 1, 1]))));
        moving.forEach((index, entry) => one.model.setTransform(handles.one[index], transforms[entry]));
        batch.scene.setTransforms(
            Uint32Array.from(moving.map((index) => handles.batch[index])),
            Float32Array.from(transforms.flat()),
        );
        moving.forEach((index, entry) => (batch.model.instances.get(handles.batch[index])!.transform = transforms[entry]));
        const fromOne = encodeAndSubmit(one.gpu, one.scene);
        const fromBatch = encodeAndSubmit(batch.gpu, batch.scene);
        expect(fromBatch.changes.regions).toEqual(fromOne.changes.regions);
        expect(fromBatch.instancesChanged).toBe(true);
        expect(readBack(batch.scene, batch.model)).toEqual([]);
        expect(readBack(one.scene, one.model)).toEqual([]);
    });

    it('setTransforms changes nothing when an entry is bad', () => {
        const { gpu, scene, model } = setup();
        const { mesh, material } = unitCube(model);
        const instance = model.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        encodeAndSubmit(gpu, scene);
        const moved = new Float32Array(composeTransform([5, 0, 0], [0, 1, 0], 0, [1, 1, 1]));
        const broken = new Float32Array(32);
        broken.set(moved);
        broken[16 + 12] = Number.NaN;
        expect(() => scene.setTransforms(Uint32Array.of(instance, instance), broken)).toThrow(/finite/);
        expect(() => scene.setTransforms(Uint32Array.of(instance, 12345), new Float32Array(32))).toThrow(/unknown or destroyed instance/);
        expect(() => scene.setTransforms(Uint32Array.of(instance), new Float32Array(8))).toThrow(/16 numbers/);
        expect(encodeAndSubmit(gpu, scene).changes.regions).toEqual([]);
        expect(scene.bounds.min[0]).toBeCloseTo(-0.5, 4);
    });

    it('merges a frame of more than MAX_DIRTY_REGIONS moves into one region with the largest move and no instance', () => {
        const { gpu, scene, model } = setup();
        const { mesh, material } = unitCube(model);
        const instances = Array.from({ length: 300 }, (_, index) => model.createInstance({ mesh, materials: [material], transform: composeTransform([index, 0, 0], [0, 1, 0], 0, [1, 1, 1]) }));
        encodeAndSubmit(gpu, scene);
        const matrices = new Float32Array(instances.length * 16);
        instances.forEach((_, index) => matrices.set(composeTransform([index, index === 7 ? 3 : 0.5, 0], [0, 1, 0], 0, [1, 1, 1]), index * 16));
        scene.setTransforms(Uint32Array.from(instances), matrices);
        const regions = encodeAndSubmit(gpu, scene).changes.regions;
        expect(regions).toHaveLength(1);
        expect(regions[0].reason).toBe('geometry');
        expect(regions[0].moved).toBeCloseTo(3, 5);
        expect(regions[0].instance).toBeUndefined();
        expect(regions[0].bounds.max[0]).toBeCloseTo(299.5, 3);
    });
});

describe('deforming meshes and receivers (T3.1)', () => {
    const half = 0.5;
    const CUBE_POSITIONS = new Float32Array([-half, -half, -half, half, -half, -half, half, half, -half, -half, half, -half, -half, -half, half, half, -half, half, half, half, half, -half, half, half]);
    const CUBE_INDICES = new Uint32Array([0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2]);
    const scaled = (factor: number) => CUBE_POSITIONS.map((value) => value * factor);

    /** gi/pose-copy on the CPU: xyz of each vertex from the source (offset and stride in floats) into the vertex buffer. */
    const poseCopy: FakeKernel = (bindings) => {
        const source = new Float32Array(bindings.get(0)!.buffer.bytes.buffer);
        const vertices = new Float32Array(bindings.get(1)!.buffer.bytes.buffer);
        const params = bindings.get(2)!;
        const [vertexCount, offset, stride, destination] = new Uint32Array(params.buffer.bytes.buffer, params.offset, 4);
        for (let vertex = 0; vertex < vertexCount; vertex++) {
            for (let axis = 0; axis < 3; axis++) vertices[destination + vertex * 3 + axis] = source[offset + vertex * stride + axis];
        }
    };

    function setupDeforming(deformBounds?: Aabb, options: { posesTraced?: boolean } = {}) {
        const warnings: string[] = [];
        const gpu = createFakeGpu({}, { kernels: { poseCopy } });
        const scene = new PlatformScene(gpu.device, { warn: (message) => warnings.push(message), posesTraced: options.posesTraced });
        const model = new SceneModel(scene);
        const descriptor = { positions: CUBE_POSITIONS, indices: CUBE_INDICES, deforming: true, deformBounds, label: 'jelly' };
        const mesh = model.createMesh(descriptor);
        const material = model.createMaterial({});
        return { gpu, scene, model, mesh, material, descriptor, warnings };
    }

    it('applies a pose: new positions in the scene copy, the mesh reported posed (not uploaded) for its refit, and its box with how far it moved', () => {
        const { gpu, scene, model, mesh, material, descriptor } = setupDeforming();
        const instance = model.createInstance({ mesh, materials: [material], transform: composeTransform([3, 0, 0], [0, 1, 0], 0, [2, 2, 2]) });
        encodeAndSubmit(gpu, scene);

        // A pose that grows the cube by half: it leaves the rest box, so the box grows with it.
        const grown = scaled(1.5);
        scene.setDeformedPositions(mesh, grown);
        model.meshes.set(mesh, { ...descriptor, positions: grown });
        const first = encodeAndSubmit(gpu, scene);
        expect(first.meshesUploaded).toEqual([]);
        expect(first.meshesPosed.map((posed) => posed.slot)).toEqual([scene.meshSlot(mesh)]);
        expect(first.meshesPosed[0].layout.bounds[3]).toBeCloseTo(0.75, 5);
        expect(first.instancesChanged).toBe(true);
        // The box before (a whole change: its box changed), then the box now with the move:
        // each corner travelled 0.25 x sqrt(3) in mesh space, x 2 for the instance's scale.
        expect(first.changes.regions.map((region) => region.moved)).toEqual([undefined, expect.closeTo(0.5 * Math.sqrt(3), 5)]);
        expect(first.changes.regions.map((region) => region.instance)).toEqual([scene.instanceSlot(instance), scene.instanceSlot(instance)]);
        expect(first.changes.regions[1].bounds.min[0]).toBeCloseTo(3 - 1.5, 4);
        expect(readBack(scene, model)).toEqual([]);

        // Back to the rest pose: inside the grown box, which stays, so the top level doesn't change.
        scene.setDeformedPositions(mesh, CUBE_POSITIONS);
        const back = encodeAndSubmit(gpu, scene);
        expect(back.meshesPosed).toHaveLength(1);
        expect(back.instancesChanged).toBe(false);
        expect(back.changes.regions).toHaveLength(1);
        expect(back.changes.regions[0].moved).toBeCloseTo(0.5 * Math.sqrt(3), 5);
        expect(back.changes.regions[0].bounds.min[0]).toBeCloseTo(3 - 1.5, 4);

        // Nothing given: nothing posed.
        expect(encodeAndSubmit(gpu, scene).meshesPosed).toEqual([]);
    });

    it('drops a deforming mesh\'s normals: the scene copy has none, so hits use the posed triangle\'s own normal', () => {
        const { gpu, scene, model, descriptor } = setupDeforming();
        const normals = new Float32Array(CUBE_POSITIONS.length).fill(0).map((_, index) => (index % 3 === 1 ? 1 : 0));
        const withNormals = model.createMesh({ ...descriptor, normals });
        const rigid = model.createMesh({ positions: CUBE_POSITIONS, indices: CUBE_INDICES, normals });
        const uploaded = encodeAndSubmit(gpu, scene).meshesUploaded;
        const layoutOf = (handle: typeof rigid) => uploaded.find((mesh) => mesh.slot === scene.meshSlot(handle))!.layout;
        expect(layoutOf(withNormals).offsets.normals).toBe(NONE);
        expect(layoutOf(rigid).offsets.normals).not.toBe(NONE);
    });

    it('copies a pose given in a GPU buffer (any stride) into the scene copy, and reports the box\'s diagonal as its move', async () => {
        const { gpu, scene, model, mesh, material, descriptor } = setupDeforming({ min: [-1, -1, -1], max: [1, 1, 1] });
        model.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        encodeAndSubmit(gpu, scene);
        await scene.prepareGpuPoses();
        // Four floats a vertex, from the second vertex slot on: offset 16 bytes, stride 16.
        const pose = scaled(1.5);
        const source = gpu.device.createBuffer({ size: 16 + 8 * 16, usage: GPUBufferUsage.STORAGE });
        const floats = new Float32Array((source as unknown as FakeBuffer).bytes.buffer);
        for (let vertex = 0; vertex < 8; vertex++) floats.set(pose.subarray(vertex * 3, vertex * 3 + 3), 4 + vertex * 4);
        {
            scene.setDeformedPositions(mesh, { buffer: source, offset: 16, stride: 16 });
            const upload = encodeAndSubmit(gpu, scene);
            expect(upload.meshesPosed.map((posed) => posed.slot)).toEqual([scene.meshSlot(mesh)]);
            // The declared box stays (it can't grow from positions the CPU can't see), and the move is its diagonal.
            expect(upload.instancesChanged).toBe(false);
            expect(upload.changes.regions).toHaveLength(1);
            expect(upload.changes.regions[0].moved).toBeCloseTo(2 * Math.sqrt(3), 5);
            // The positions are the pose's; the instance's box is the declared one, not the pose's tight box the checker expects.
            model.meshes.set(mesh, { ...descriptor, positions: pose });
            const problems = readBack(scene, model);
            expect(problems.filter((problem) => !/bounds: got \[-1\.0+,-1\.0+,-1\.0+,1\.0+,1\.0+,1\.0+\]/.test(problem))).toEqual([]);
        }
    });

    it('drops a GPU pose it can\'t copy in the next encode, keeping no reference to the host\'s buffer', async () => {
        const { gpu, scene, mesh } = setupDeforming({ min: [-1, -1, -1], max: [1, 1, 1] });
        const source = gpu.device.createBuffer({ size: 96, usage: GPUBufferUsage.STORAGE });
        // Given before the mesh is on the GPU: that encode uploads the mesh, and the pose is dropped.
        scene.setDeformedPositions(mesh, { buffer: source });
        expect(encodeAndSubmit(gpu, scene).meshesPosed).toEqual([]);
        // Given while the copy kernel isn't compiled: dropped too, not applied once it is.
        scene.setDeformedPositions(mesh, { buffer: source });
        expect(encodeAndSubmit(gpu, scene).meshesPosed).toEqual([]);
        await scene.prepareGpuPoses();
        expect(encodeAndSubmit(gpu, scene).meshesPosed).toEqual([]);
        scene.setDeformedPositions(mesh, { buffer: source });
        expect(encodeAndSubmit(gpu, scene).meshesPosed).toHaveLength(1);
    });

    it('checks GPU poses at the call: deformBounds required, usage, size and texture shape', () => {
        const unbounded = setupDeforming();
        const storage = (size: number) => unbounded.gpu.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
        expect(() => unbounded.scene.setDeformedPositions(unbounded.mesh, { buffer: storage(96) })).toThrow(/needs deformBounds/);
        const { gpu, scene, mesh } = setupDeforming({ min: [-1, -1, -1], max: [1, 1, 1] });
        expect(() => scene.setDeformedPositions(mesh, { buffer: gpu.device.createBuffer({ size: 96, usage: GPUBufferUsage.COPY_SRC }) })).toThrow(/STORAGE/);
        expect(() => scene.setDeformedPositions(mesh, { buffer: storage(95) })).toThrow(/96/);
        expect(() => scene.setDeformedPositions(mesh, { buffer: storage(96), stride: 8 })).toThrow(/stride/);
        const texture = (format: string, width: number, height: number, usage = 0x1) => ({ texture: { format, width, height, usage } as unknown as GPUTexture });
        expect(() => scene.setDeformedPositions(mesh, texture('rgba16float', 16, 1))).toThrow(/rgba32float/);
        expect(() => scene.setDeformedPositions(mesh, texture('rgba32float', 8, 1))).toThrow(/multiple of 16/);
        expect(() => scene.setDeformedPositions(mesh, texture('rgba32float', 16, 1, 0x4))).toThrow(/COPY_SRC/);
        scene.setDeformedPositions(mesh, texture('rgba32float', 16, 1));
    });

    it('without subgroups (posesTraced false), keeps the rest pose and warns once', () => {
        const { gpu, scene, model, mesh, material, descriptor, warnings } = setupDeforming(undefined, { posesTraced: false });
        model.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        encodeAndSubmit(gpu, scene);
        scene.setDeformedPositions(mesh, scaled(2));
        scene.setDeformedPositions(mesh, scaled(3));
        const upload = encodeAndSubmit(gpu, scene);
        expect(upload.meshesPosed).toEqual([]);
        expect(upload.changes.regions).toEqual([]);
        expect(warnings).toHaveLength(1);
        expect(warnings[0]).toMatch(/no subgroups.*rest pose/);
        model.meshes.set(mesh, descriptor);
        expect(readBack(scene, model)).toEqual([]);
        // Bad calls still throw.
        expect(() => scene.setDeformedPositions(mesh, new Float32Array(9))).toThrow(/8 vertices/);
    });

    it('starts from deformBounds, and grows it with one warning when a pose leaves it', () => {
        const { gpu, scene, model, mesh, material, warnings } = setupDeforming({ min: [-1, -1, -1], max: [1, 2, 1] });
        model.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        const created = encodeAndSubmit(gpu, scene).changes.regions;
        expect(created[0].bounds.max[1]).toBeCloseTo(2, 4);
        expect(scene.bounds.max[1]).toBeCloseTo(2, 4);

        // Inside the declared box: one region, the box as declared, no warning.
        scene.setDeformedPositions(mesh, scaled(1.8));
        const inside = encodeAndSubmit(gpu, scene);
        expect(inside.changes.regions).toHaveLength(1);
        expect(inside.changes.regions[0].bounds.max[1]).toBeCloseTo(2, 4);
        expect(warnings).toEqual([]);

        // Outside it twice: the box grows, and the warning is shown once.
        scene.setDeformedPositions(mesh, scaled(5));
        expect(encodeAndSubmit(gpu, scene).changes.regions[1].bounds.max[0]).toBeCloseTo(2.5, 4);
        scene.setDeformedPositions(mesh, scaled(6));
        encodeAndSubmit(gpu, scene);
        expect(warnings).toHaveLength(1);
        expect(warnings[0]).toMatch(/jelly.*deformBounds/);
        expect(scene.bounds.max[0]).toBeCloseTo(3, 4);
    });

    it('keeps only the newest pose, and waits for a mesh not on the GPU yet', () => {
        const { gpu, scene, model, mesh, material, descriptor } = setupDeforming();
        model.createInstance({ mesh, materials: [material], transform: composeTransform([0, 0, 0], [0, 1, 0], 0, [1, 1, 1]) });
        // Given before the mesh's first upload: that encode uploads the rest pose, the next one applies the pose.
        scene.setDeformedPositions(mesh, scaled(2));
        const first = encodeAndSubmit(gpu, scene);
        expect(first.meshesUploaded).toHaveLength(1);
        expect(first.meshesPosed).toEqual([]);
        expect(encodeAndSubmit(gpu, scene).meshesPosed).toHaveLength(1);

        // Only the newest of two poses arrives, and the host's array was copied.
        const host = scaled(3);
        scene.setDeformedPositions(mesh, scaled(2.5));
        scene.setDeformedPositions(mesh, host);
        host.fill(0);
        const applied = encodeAndSubmit(gpu, scene);
        expect(applied.meshesPosed).toHaveLength(1);
        expect(applied.changes.regions.at(-1)!.moved).toBeCloseTo(0.5 * Math.sqrt(3), 5);
        model.meshes.set(mesh, { ...descriptor, positions: scaled(3) });
        expect(readBack(scene, model)).toEqual([]);

        // A new shape drops a waiting pose (it may not even have the same vertex count).
        scene.setDeformedPositions(mesh, scaled(4));
        model.updateMesh(mesh, descriptor);
        const reshaped = encodeAndSubmit(gpu, scene);
        expect(reshaped.meshesUploaded).toHaveLength(1);
        expect(reshaped.meshesPosed).toEqual([]);
        expect(encodeAndSubmit(gpu, scene).meshesPosed).toEqual([]);
    });

    it('refuses bad poses at the call and keeps nothing of them', () => {
        const { gpu, scene, model, mesh } = setupDeforming();
        encodeAndSubmit(gpu, scene);
        const rigid = model.createMesh({ positions: CUBE_POSITIONS, indices: CUBE_INDICES });
        expect(() => scene.setDeformedPositions(rigid, CUBE_POSITIONS)).toThrow(/deforming: true/);
        expect(() => scene.setDeformedPositions(mesh, new Float32Array(9))).toThrow(/8 vertices/);
        const broken = scaled(1);
        broken[7] = Number.NaN;
        expect(() => scene.setDeformedPositions(mesh, broken)).toThrow(/vertex 2 .*non-finite/);
        expect(() => scene.createMesh({ positions: CUBE_POSITIONS, indices: CUBE_INDICES, deforming: true, deformBounds: { min: [0, 2, 0], max: [1, 1, 1] } })).toThrow(/min above max/);
        // Only the new rigid mesh's first upload.
        expect(encodeAndSubmit(gpu, scene).meshesUploaded.map((uploaded) => uploaded.slot)).toEqual([scene.meshSlot(rigid)]);
    });

    it('lists receivers where they are when created, shown, moved or deformed, and not when hidden', () => {
        const { gpu, scene, model, mesh, material } = setupDeforming();
        const receiver = model.createInstance({ mesh, materials: [material], transform: composeTransform([5, 0, 0], [0, 1, 0], 0, [1, 1, 1]), contributesToGI: false });
        const created = encodeAndSubmit(gpu, scene).changes;
        expect(created.regions).toEqual([]);
        expect(created.receivers).toHaveLength(1);
        expect(created.receivers[0].min[0]).toBeCloseTo(4.5, 4);

        model.setVisible(receiver, false);
        model.setTransform(receiver, composeTransform([6, 0, 0], [0, 1, 0], 0, [1, 1, 1]));
        expect(encodeAndSubmit(gpu, scene).changes.receivers).toEqual([]);

        model.setVisible(receiver, true);
        const shown = encodeAndSubmit(gpu, scene).changes.receivers;
        expect(shown).toHaveLength(1);
        expect(shown[0].min[0]).toBeCloseTo(5.5, 4);

        // Moved with setTransforms, as the three adapter sends moves.
        scene.setTransforms(Uint32Array.of(receiver), new Float32Array(composeTransform([0, 7, 0], [0, 1, 0], 0, [1, 1, 1])));
        const moved = encodeAndSubmit(gpu, scene).changes;
        expect(moved.regions).toEqual([]);
        expect(moved.receivers).toHaveLength(1);
        expect(moved.receivers[0].min[1]).toBeCloseTo(6.5, 4);

        scene.setDeformedPositions(mesh, scaled(1.2));
        const deformed = encodeAndSubmit(gpu, scene).changes;
        expect(deformed.regions).toEqual([]);
        expect(deformed.receivers).toHaveLength(1);
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
