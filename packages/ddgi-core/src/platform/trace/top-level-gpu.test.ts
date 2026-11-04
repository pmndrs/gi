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
import { buildTopLevelShape } from './top-level-shape.ts';

/** The fake GPU plus what the tracer's top level needs: layouts, features, compute passes and a mappable buffer. */
function createGpu(features: string[]): FakeGpu & { compute: { passes: number } } {
    const fake = createFakeGpu();
    const state = { passes: 0 };
    const device = fake.device as unknown as Record<string, unknown>;
    const createBuffer = device.createBuffer as (descriptor: GPUBufferDescriptor) => FakeBuffer;
    Object.assign(device, {
        features: new Set(features),
        createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ descriptor }),
        createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ descriptor }),
        // Stats readbacks never resolve here: the fake device has no GPU to wait for.
        createBuffer: (descriptor: GPUBufferDescriptor) => Object.assign(createBuffer(descriptor), { mapAsync: () => new Promise(() => {}) }),
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
        const random = createRandom(7);
        const count = 300;
        const objectToWorld = new Float64Array(count * 16);
        const meshBounds = new Float32Array(count * 6);
        for (let index = 0; index < count; index++) {
            objectToWorld.set(randomAffine(random, { extent: 50, mirror: index % 3 === 0 }), index * 16);
            meshBounds.set([-1, -0.5, -0.2, 1, 0.5, 0.2], index * 6);
        }
        const shape = buildTopLevelShape({ objectToWorld: objectToWorld.slice(), meshBounds: meshBounds.slice() });
        expect(shape.nodeCount).toBe(count - 1);
        expect(shape.leafSlots.length).toBe(count);
        expect(shape.relativeCost).toBeGreaterThan(1);

        // The refit's input: each record's world box, in record order.
        const primitiveBoxes = new Float32Array(count * PRIMITIVE_BOX_FLOATS);
        const box = new Float32Array(6);
        for (let record = 0; record < count; record++) {
            const input = shape.order[record];
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
        const shape = buildTopLevelShape({ objectToWorld: new Float64Array(0), meshBounds: new Float32Array(0) });
        expect(shape.nodeCount).toBe(0);
        expect(shape.leafSlots.length).toBe(0);
    });

    it('gathers through the scene copy and the bvh package’s instance box', () => {
        const wgsl = topLevelGatherWgsl();
        expect(wgsl).toContain('fn bvhInstanceWorldBox(');
        expect(wgsl).toContain('giInstanceNormalMatrix(slot)');
        expect(wgsl).toContain('fn giTopLevelGather(');
    });
});

describe('the rays backend’s top level', () => {
    async function setUp(features: string[]) {
        const gpu = createGpu(features);
        const scene = new PlatformScene(gpu.device);
        const rays = new PlatformRayQueries(gpu.device, scene, { buildIn: 'main' });
        await rays.prepare(new PlatformKernelBuilder(gpu.device, 'test'));
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
        rays.dispose();
        expect(gpu.buffers.filter((buffer) => buffer.label.includes('top level')).every((buffer) => buffer.destroyed)).toBe(true);
    });
});
