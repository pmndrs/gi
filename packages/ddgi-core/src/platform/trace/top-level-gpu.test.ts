/**
 * The GPU top level's CPU side (B1a): shape builds, what gets written where,
 * and when a new shape is asked for. The fake device runs no compute, so the
 * refit itself is checked on real GPUs (tools/gpu/pages/tlas-oracle).
 */

import { refitOnCpu, PRIMITIVE_BOX_FLOATS } from '@pmndrs/bvh/refit-cpu';
import { describe, expect, it } from 'vitest';
import { PlatformKernelBuilder } from '../kernel-builder.ts';
import { PlatformScene } from '../scene/platform-scene.ts';
import { createFakeGpu, type FakeBuffer, type FakeGpu } from '../scene/testing/fake-gpu.ts';
import { createRandom, randomAffine } from './oracle-fixtures.ts';
import { BVH_NODE_BYTES, BVH_NODE_WORDS, TOP_HEADER_WORD } from './layout.ts';
import { PlatformRayQueries } from './ray-queries.ts';
import { transformBounds } from './top-level.ts';
import { topLevelGatherWgsl } from './top-level-gpu.ts';
import { buildTopLevelShape, TopLevelShapeKeeper, type TopLevelBuildInput } from './top-level-shape.ts';
import { GILimitError } from '../contract/errors.ts';
import { BVH_REFIT_STAT, BVH_REFIT_STATS_BYTES } from '@pmndrs/bvh/gpu-refit';

/** A build input over `count` instances whose slot is their index. */
function randomInput(seed: number, count: number): TopLevelBuildInput {
    const random = createRandom(seed);
    const objectToWorld = new Float64Array(count * 16);
    const meshBounds = new Float32Array(count * 6);
    for (let index = 0; index < count; index++) {
        objectToWorld.set(randomAffine(random, { extent: 50, mirror: index % 3 === 0 }), index * 16);
        meshBounds.set([-1, -0.5, -0.2, 1, 0.5, 0.2], index * 6);
    }
    return { slots: Uint32Array.from({ length: count }, (_, index) => index), visible: new Uint8Array(count).fill(1), objectToWorld, meshBounds };
}

function copyInput(input: TopLevelBuildInput): TopLevelBuildInput {
    return { slots: input.slots.slice(), visible: input.visible.slice(), objectToWorld: input.objectToWorld.slice(), meshBounds: input.meshBounds.slice() };
}

/** The fake GPU plus what the tracer's top level needs: layouts, features, compute passes and a mappable buffer. */
function createGpu(features: string[], limits: Partial<GPUSupportedLimits> = {}): FakeGpu & { compute: { passes: number; mapAsyncCalls: number } } {
    const fake = createFakeGpu(limits);
    const state = { passes: 0, mapAsyncCalls: 0 };
    const device = fake.device as unknown as Record<string, unknown>;
    const createBuffer = device.createBuffer as (descriptor: GPUBufferDescriptor) => FakeBuffer;
    Object.assign(device, {
        features: new Set(features),
        createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ descriptor }),
        createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ descriptor }),
        // Readbacks are counted, and never resolve here: the fake device has no GPU to wait for.
        createBuffer: (descriptor: GPUBufferDescriptor) =>
            Object.assign(createBuffer(descriptor), {
                mapAsync: () => {
                    state.mapAsyncCalls++;
                    return new Promise(() => {});
                },
            }),
    });
    Object.assign(device.limits as object, { maxComputeWorkgroupsPerDimension: 65535, maxBindGroups: 4 });
    const pass = { setBindGroup() {}, setPipeline() {}, dispatchWorkgroups() {}, end() {} };
    const createEncoder = fake.createEncoder;
    return Object.assign(fake, {
        createEncoder: () =>
            Object.assign(createEncoder(), {
                beginComputePass: () => {
                    state.passes++;
                    return pass;
                },
            }),
        compute: state,
    });
}

function cube(): { positions: Float32Array; indices: Uint32Array } {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
    return { positions, indices: new Uint32Array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]) };
}

describe('top-level shapes', () => {
    it('refitting a shape with the boxes it was built from gives back the built boxes, bit for bit', () => {
        const count = 300;
        const { objectToWorld, meshBounds, ...rest } = randomInput(7, count);
        const shape = buildTopLevelShape(copyInput({ objectToWorld, meshBounds, ...rest }));
        expect(shape.nodeCount).toBe(count - 1);
        expect(shape.leafSlots.length).toBe(count);
        expect(shape.relativeCost).toBeGreaterThan(1);

        // The refit's input: each record's world box, in record order.
        const primitiveBoxes = new Float32Array(count * PRIMITIVE_BOX_FLOATS);
        const box = new Float32Array(6);
        for (let record = 0; record < count; record++) {
            const input = shape.recordSlots[record];   // slot = input index here
            transformBounds(meshBounds.subarray(input * 6, input * 6 + 6), objectToWorld.subarray(input * 16, input * 16 + 16), box, 0);
            primitiveBoxes.set(box.subarray(0, 3), record * PRIMITIVE_BOX_FLOATS);
            primitiveBoxes.set(box.subarray(3, 6), record * PRIMITIVE_BOX_FLOATS + 4);
        }
        const built = new Uint32Array(shape.nodes.slice(0, shape.nodeCount * BVH_NODE_BYTES));
        const refit = built.slice();
        const parents = Uint32Array.from(shape.parentSlots, (slot) => (slot === 0xffffffff ? slot : slot >> 1));
        refitOnCpu(new Float32Array(refit.buffer), refit, 0, shape.nodeCount, parents, primitiveBoxes);
        expect(refit).toEqual(built);
        // Every node but the root has a parent slot that points back at it.
        const words = built;
        for (let node = 1; node < shape.nodeCount; node++) {
            const slot = shape.parentSlots[node];
            const parentBase = (slot >> 1) * BVH_NODE_WORDS;
            expect(words[parentBase + ((slot & 1) === 0 ? 3 : 7)]).toBe(node);
        }
    });

    it('builds an empty shape for no instances', () => {
        const shape = buildTopLevelShape({ slots: new Uint32Array(0), visible: new Uint8Array(0), objectToWorld: new Float64Array(0), meshBounds: new Float32Array(0) });
        expect(shape.nodeCount).toBe(0);
        expect(shape.leafSlots.length).toBe(0);
    });

    it('keeps a shape whose quality holds, and builds a new one once it has decayed', () => {
        const keeper = new TopLevelShapeKeeper();
        const input = randomInput(11, 400);
        const built = keeper.build(copyInput(input), 1);
        // Small moves: the refit shape is about as good as built.
        const nudged = copyInput(input);
        for (let index = 0; index < 400; index++) nudged.objectToWorld[index * 16 + 12] += 0.05;
        const kept = keeper.check(nudged, 1.25, 1, 2);
        expect(kept.shape).toBeNull();
        expect(kept.builtCost).toBeCloseTo(built.relativeCost, 6);
        expect(kept.relativeCost).toBeLessThan(1.25 * kept.builtCost);
        // Every instance somewhere else: the old shape no longer suits them.
        const scattered = randomInput(12, 400);
        const rebuilt = keeper.check(scattered, 1.25, 1, 2);
        expect(rebuilt.relativeCost).toBeGreaterThan(1.25 * rebuilt.builtCost);
        expect(rebuilt.shape?.recordSlots.length).toBe(400);
    });

    it('judges the shape in use, named by generation, not the newest it built', () => {
        const keeper = new TopLevelShapeKeeper();
        const input = randomInput(21, 300);
        const first = keeper.build(copyInput(input), 1);
        // Shape 2 was built for scattered positions, but (say) it couldn't be swapped in.
        keeper.build(randomInput(22, 300), 2);
        // Checking against shape 1 at its own positions: nothing to rebuild, and the cost is shape 1's.
        const check = keeper.check(copyInput(input), 1.25, 1, 3);
        expect(check.shape).toBeNull();
        expect(check.builtCost).toBeCloseTo(first.relativeCost, 6);
        // A generation it doesn't know (never built, or long gone): it builds a fresh shape.
        expect(keeper.check(copyInput(input), 1.25, 99, 4).shape).not.toBeNull();
    });

    it('gathers through the scene copy and the bvh package’s instance box', () => {
        const wgsl = topLevelGatherWgsl();
        expect(wgsl).toContain('fn bvhInstanceWorldBox(');
        expect(wgsl).toContain('giInstanceNormalMatrix(slot)');
        expect(wgsl).toContain('fn giTopLevelGather(');
    });
});

describe('the rays backend’s top level', () => {
    async function setUp(features: string[], options: { diagnostics?: boolean; limits?: Partial<GPUSupportedLimits> } = {}) {
        const gpu = createGpu(features, options.limits);
        const scene = new PlatformScene(gpu.device);
        const rays = new PlatformRayQueries(gpu.device, scene, { buildIn: 'main', diagnostics: options.diagnostics });
        await rays.prepare(new PlatformKernelBuilder(gpu.device, 'test'), { batches: false });
        const frame = () => {
            const encoder = gpu.createEncoder();
            rays.update(encoder, scene.encode(encoder));
            gpu.submit(encoder);
        };
        return { gpu, scene, rays, frame };
    }

    it('takes the CPU fallback on a device without subgroups', async () => {
        const { rays } = await setUp([]);
        expect(rays.topLevelPath).toBe('cpu');
        expect(rays.gpuTopLevel).toBeNull();
        rays.dispose();
    });

    it('refits on the GPU when instances move, and builds a new shape only when one is created', async () => {
        const { gpu, scene, rays, frame } = await setUp(['subgroups']);
        expect(rays.topLevelPath).toBe('gpu-refit');
        const top = rays.gpuTopLevel;
        if (!top) throw new Error('no GPU top level');
        const mesh = scene.createMesh(cube());
        const material = scene.createMaterial({});
        const translation = (x: number) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1];
        const first = scene.createInstance({ mesh, materials: [material], transform: translation(0) });
        scene.createInstance({ mesh, materials: [material], transform: translation(3) });
        frame();
        expect(rays.building).toBe(true);
        await rays.whenBuilt();
        frame();
        expect(rays.building).toBe(false);
        expect(top.counters.shapesBuilt).toBe(1);
        expect(rays.topLevelInstanceCount).toBe(2);
        const header = new Uint32Array((top.buffer as unknown as FakeBuffer).bytes.buffer, 0, 4);
        expect(header[TOP_HEADER_WORD.instanceCount]).toBe(2);
        expect(header[TOP_HEADER_WORD.nodeCount]).toBe(1);

        // Moving: one refit pass, no new shape.
        const passesBefore = gpu.compute.passes;
        scene.setTransform(first, translation(-2));
        frame();
        expect(gpu.compute.passes).toBe(passesBefore + 1);
        expect(rays.building).toBe(false);
        expect(top.counters.shapesBuilt).toBe(1);

        // Hiding and showing: refits only, since the shape keeps hidden instances.
        scene.setVisible(first, false);
        frame();
        scene.setVisible(first, true);
        frame();
        expect(rays.building).toBe(false);

        // A new instance needs a new shape; it arrives off the frame.
        scene.createInstance({ mesh, materials: [material], transform: translation(6) });
        frame();
        expect(rays.building).toBe(true);
        await rays.whenBuilt();
        frame();
        expect(top.counters.shapesBuilt).toBe(2);
        expect(rays.topLevelInstanceCount).toBe(3);

        // Nothing changed: no pass at all.
        const idleBefore = gpu.compute.passes;
        frame();
        expect(gpu.compute.passes).toBe(idleBefore);
        // And no readback ever, without diagnostics: hosts are promised none.
        expect(gpu.compute.mapAsyncCalls).toBe(0);
        rays.dispose();
        expect(gpu.buffers.filter((buffer) => buffer.label.includes('top level')).every((buffer) => buffer.destroyed)).toBe(true);
    });

    it('reads the refit stats back only with diagnostics on', async () => {
        const { gpu, scene, rays, frame } = await setUp(['subgroups'], { diagnostics: true });
        const mesh = scene.createMesh(cube());
        const instance = scene.createInstance({ mesh, materials: [scene.createMaterial({})], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
        frame();
        await rays.whenBuilt();
        frame();
        scene.setTransform(instance, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1]);
        frame();
        expect(gpu.compute.mapAsyncCalls).toBeGreaterThan(0);
        rays.dispose();
    });

    it('keeps the old shape when a new one cannot be swapped in, reports it once, and goes on refitting', async () => {
        const { gpu, scene, rays, frame } = await setUp(['subgroups']);
        const top = rays.gpuTopLevel;
        if (!top) throw new Error('no GPU top level');
        const mesh = scene.createMesh(cube());
        const material = scene.createMaterial({});
        const at = (x: number) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1];
        const first = scene.createInstance({ mesh, materials: [material], transform: at(0) });
        frame();
        await rays.whenBuilt();
        frame();
        expect(top.instanceCount).toBe(1);

        // 300 more instances: the scene copy grows its buffers in this frame, under the normal limit...
        for (let index = 1; index <= 300; index++) scene.createInstance({ mesh, materials: [material], transform: at(index * 2) });
        frame();
        await rays.whenBuilt();
        // ...then the limit drops below what the new shape needs (about 38 KB), so swapping it in fails.
        Object.assign(gpu.device.limits as object, { maxStorageBufferBindingSize: 20_000 });
        expect(() => frame()).toThrow(GILimitError);
        expect(rays.building).toBe(false);
        expect(top.instanceCount).toBe(1);
        const header = new Uint32Array((top.buffer as unknown as FakeBuffer).bytes.buffer, 0, 4);
        expect(header[TOP_HEADER_WORD.instanceCount]).toBe(1);

        // Later frames don't throw, and moves are still refit on the old shape.
        const passesBefore = gpu.compute.passes;
        scene.setTransform(first, at(-3));
        expect(() => frame()).not.toThrow();
        expect(gpu.compute.passes).toBe(passesBefore + 1);
        expect(() => frame()).not.toThrow();

        // With room again, the next change of the instance set swaps in a shape that covers everything.
        Object.assign(gpu.device.limits as object, { maxStorageBufferBindingSize: 128 * 1024 * 1024 });
        scene.createInstance({ mesh, materials: [material], transform: at(-10) });
        frame();
        await rays.whenBuilt();
        frame();
        expect(top.instanceCount).toBe(302);
        rays.dispose();
    });

    it('warns once about refits that gave up, however often the stats say so, and counts every one', async () => {
        const gpu = createGpu(['subgroups']);
        // Every stats readback reports 3 reads that gave up and one repair.
        const device = gpu.device as unknown as Record<string, unknown>;
        const createBuffer = device.createBuffer as (descriptor: GPUBufferDescriptor) => FakeBuffer;
        device.createBuffer = (descriptor: GPUBufferDescriptor) => {
            const buffer = createBuffer(descriptor);
            if (!(descriptor.label ?? '').endsWith('stats readback')) return buffer;
            const words = new Uint32Array(BVH_REFIT_STATS_BYTES / 4);
            words[BVH_REFIT_STAT.incomplete] = 3;
            words[BVH_REFIT_STAT.repairs] = 1;
            return Object.assign(buffer, { mapAsync: () => Promise.resolve(), getMappedRange: () => words.buffer, unmap: () => {} });
        };
        const warnings: string[] = [];
        const scene = new PlatformScene(gpu.device);
        const rays = new PlatformRayQueries(gpu.device, scene, { buildIn: 'main', diagnostics: true, warn: (message) => warnings.push(message) });
        await rays.prepare(new PlatformKernelBuilder(gpu.device, 'test'), { batches: false });
        const frame = () => {
            const encoder = gpu.createEncoder();
            rays.update(encoder, scene.encode(encoder));
            gpu.submit(encoder);
        };
        const instance = scene.createInstance({ mesh: scene.createMesh(cube()), materials: [scene.createMaterial({})], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
        frame();
        await rays.whenBuilt();
        for (let step = 0; step < 8; step++) {
            scene.setTransform(instance, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, step, 0, 0, 1]);
            frame();
            await Promise.resolve();
        }
        expect(warnings).toHaveLength(1);
        expect(rays.gpuTopLevel?.counters.refitGaveUp).toBeGreaterThan(3);
        expect(rays.topLevelCounters?.['refit repairs (diagnostics)']).toBeGreaterThan(1);
        rays.dispose();
    });

    it('marks a removed mesh absent in the mesh table even when placing another tree fails that frame', async () => {
        // Room for small trees, but not for 48-byte triangle records of the big mesh.
        const { gpu, scene, rays, frame } = await setUp(['subgroups'], { limits: { maxStorageBufferBindingSize: 400_000 } });
        const material = scene.createMaterial({});
        const small = scene.createMesh(cube());
        const instance = scene.createInstance({ mesh: small, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
        frame();
        await rays.whenBuilt();
        frame();
        const meshTable = gpu.buffers.filter((buffer) => buffer.label.includes('top level mesh table') && !buffer.destroyed).at(-1);
        const smallSlot = scene.meshSlot(small);
        const rootWord = (table: FakeBuffer) => new Uint32Array(table.bytes.buffer)[smallSlot * 8 + 3];
        if (!meshTable) throw new Error('no mesh table');
        expect(rootWord(meshTable)).not.toBe(0xffffffff);

        // A big mesh whose tree can't be placed, uploaded and built first...
        const triangles = 9000;
        const positions = new Float32Array(triangles * 9).map((_, index) => (index % 97) / 10);
        const big = scene.createMesh({ positions, indices: Uint32Array.from({ length: triangles * 3 }, (_, index) => index) });
        scene.createInstance({ mesh: big, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1] });
        frame();
        await rays.whenBuilt();
        // ...then, in the frame where placing it fails, the small mesh goes away.
        scene.destroyInstance(instance);
        scene.destroyMesh(small);
        expect(() => frame()).toThrow(GILimitError);
        const tableAfter = gpu.buffers.filter((buffer) => buffer.label.includes('top level mesh table') && !buffer.destroyed).at(-1);
        if (!tableAfter) throw new Error('no mesh table');
        expect(rootWord(tableAfter)).toBe(0xffffffff);
        rays.dispose();
    });
});

describe('the scene copy’s instanceSetChanged flag', () => {
    it('is set when an instance that contributes to GI is created or destroyed, not when one moves or is hidden', () => {
        const gpu = createGpu(['subgroups']);
        const scene = new PlatformScene(gpu.device);
        const encode = () => scene.encode(gpu.createEncoder()).instanceSetChanged;
        const mesh = scene.createMesh(cube());
        const material = scene.createMaterial({});
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const instance = scene.createInstance({ mesh, materials: [material], transform: identity });
        expect(encode()).toBe(true);
        scene.setTransform(instance, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1]);
        expect(encode()).toBe(false);
        scene.setVisible(instance, false);
        expect(encode()).toBe(false);
        // An instance that never contributes to GI is never in a shape.
        scene.createInstance({ mesh, materials: [material], transform: identity, contributesToGI: false });
        expect(encode()).toBe(false);
        scene.destroyInstance(instance);
        expect(encode()).toBe(true);
        expect(scene.packGiInstances().slots.length).toBe(0);
        scene.dispose();
    });
});
