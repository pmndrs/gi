import { describe, expect, it } from 'vitest';
import { GILimitError, GINotReadyError } from './contract/errors.ts';
import type { FrameInput } from './contract/frame.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame } from './contract/technique.ts';
import type { WgslModule } from './contract/wgsl.ts';
import { DEFAULT_UNITS_PER_FRAME, GIPlatformImpl } from './gi-platform.ts';
import { countStorageBuffers, orderModules, placeModules, PlatformKernelBuilder } from './kernel-builder.ts';
import { createFakeGpu, type FakeBuffer, type FakeGpu } from './scene/testing/fake-gpu.ts';
import { createCpuTracer } from './trace/cpu-traversal.ts';
import { TOP_HEADER_WORD } from './trace/layout.ts';
import type { TraceSceneData } from './trace/scene-assembly.ts';
import type { TopLevelBvh } from './trace/top-level.ts';

/** The fake GPU plus what the platform needs beyond the scene copy: layouts, `lost`, a few limits. */
function createPlatformGpu(limits: Partial<GPUSupportedLimits> = {}): FakeGpu & { loseDevice(): Promise<void> } {
    const fake = createFakeGpu(limits);
    let loseDevice: () => void = () => {};
    const lost = new Promise<GPUDeviceLostInfo>((resolve) => {
        loseDevice = () => resolve({ reason: 'unknown', message: 'test' } as GPUDeviceLostInfo);
    });
    const device = fake.device as unknown as Record<string, unknown>;
    Object.assign(device, {
        lost,
        createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ descriptor }),
        createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ descriptor }),
    });
    Object.assign(device.limits as object, { maxBindGroups: 4, maxComputeWorkgroupsPerDimension: 65535, ...limits });
    return {
        ...fake,
        loseDevice: async () => {
            loseDevice();
            await lost;
        },
    };
}

const FRAME: FrameInput = {
    deltaTime: 1 / 60,
    budget: {},
    camera: { position: [0, 0, 0], view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], projection: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], near: 0.1, far: 100 },
};

function encodeFrame(gpu: FakeGpu, platform: GIPlatformImpl, frame: FrameInput = FRAME): void {
    const encoder = gpu.createEncoder();
    platform.encode(encoder, frame);
    gpu.submit(encoder);
}

/** A unit quad in the z = 0 plane facing +z, two triangles. */
function quad(size = 1): { positions: Float32Array; indices: Uint32Array } {
    return {
        positions: new Float32Array([-size, -size, 0, size, -size, 0, size, size, 0, -size, size, 0]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
}

function translation(x: number, y: number, z: number): number[] {
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

/** The tracer's three buffers as the GPU holds them, read back from the fake device for the CPU walk. */
function traceSceneOnGpu(platform: GIPlatformImpl): TraceSceneData {
    const [nodes, triangles, topLevel] = platform.rays.module.resources().map((resource) => (resource as GPUBufferBinding).buffer as unknown as FakeBuffer);
    const topWords = new Uint32Array(topLevel.bytes.slice().buffer);
    return {
        nodes: nodes.bytes.slice().buffer,
        triangles: triangles.bytes.slice().buffer,
        topLevel: {
            buffer: topWords.buffer,
            instanceCount: topWords[TOP_HEADER_WORD.instanceCount],
            nodeCount: topWords[TOP_HEADER_WORD.nodeCount],
        } as TopLevelBvh,
        meshes: [],
        maxMeshDepth: 0,
        requiredStackSize: 0,
        byteLength: 0,
    };
}

function traceDown(platform: GIPlatformImpl, x: number, y: number) {
    return createCpuTracer(traceSceneOnGpu(platform)).closest({ origin: [x, y, 5], direction: [0, 0, -1], tMin: 0, tMax: 100 });
}

async function readyPlatform(gpu: FakeGpu, platform: GIPlatformImpl): Promise<void> {
    await platform.prepare();
    encodeFrame(gpu, platform);
    await platform.rays.whenBuilt();
    encodeFrame(gpu, platform);
}

/** A technique that records what the platform gives it. */
class RecordingTechnique implements GITechnique {
    readonly requirements = { queries: ['rays' as const] };
    readonly settings = {};
    readonly debugViews = [];
    readonly counters = { 'frames encoded': 0 };
    readonly memoryBytes = 1000;
    readonly msPerUnit = undefined;
    readonly frames: TechniqueFrame[] = [];
    context: TechniqueContext | null = null;
    prepared = 0;
    disposed = false;
    readonly id: string;
    readonly log: string[];

    constructor(id: string, log: string[]) {
        this.id = id;
        this.log = log;
    }
    get shading(): never {
        throw new Error('not needed here');
    }
    attach(context: TechniqueContext): void {
        this.context = context;
    }
    async prepare(): Promise<void> {
        this.prepared++;
    }
    encode(_encoder: GPUCommandEncoder, frame: TechniqueFrame): void {
        this.frames.push(frame);
        this.log.push(this.id);
    }
    resetHistory(): void {}
    dispose(): void {
        this.disposed = true;
    }
}

describe('kernel builder', () => {
    const module = (name: string, storage: number, dependencies: WgslModule[] = []): WgslModule => ({
        name,
        dependencies,
        code: (slot) => `// ${name} at ${slot.firstBinding}`,
        entries: Array.from({ length: storage }, () => ({ visibility: 4, buffer: { type: 'read-only-storage' as const } })),
        resources: () => [],
        version: 0,
        storageBufferCount: storage,
    });

    it('puts dependencies first, each module once, with consecutive bindings in group 0', () => {
        const scene = module('gi-scene', 2);
        const rays = module('gi-trace', 3);
        const shade = module('gi-shade', 0, [scene, rays]);
        const ordered = orderModules([shade, rays, scene]);
        expect(ordered.map((entry) => entry.name)).toEqual(['gi-scene', 'gi-trace', 'gi-shade']);
        expect(placeModules(ordered).map(({ slot }) => [slot.group, slot.firstBinding])).toEqual([
            [0, 0],
            [0, 2],
            [0, 5],
        ]);
        expect(countStorageBuffers(ordered)).toBe(5);
    });

    it('refuses two different modules with the same name, and a module that depends on itself', () => {
        expect(() => orderModules([module('gi-scene', 1), module('gi-scene', 1)])).toThrow(/two different/);
        const loop = module('gi-loop', 0);
        (loop.dependencies as WgslModule[]).push(loop);
        expect(() => orderModules([loop])).toThrow(/depends on itself/);
    });

    it('throws a GILimitError naming the limit before compiling an over-budget kernel', async () => {
        const gpu = createPlatformGpu();
        const builder = new PlatformKernelBuilder(gpu.device, 'test');
        const options = { label: 'too big', modules: [module('a', 5), module('b', 4)], code: '', entryPoint: 'main' };
        const error = await builder.computeKernel(options).catch((caught: unknown) => caught);
        expect(error).toBeInstanceOf(GILimitError);
        expect((error as GILimitError).limit).toBe('maxStorageBuffersPerShaderStage');
        expect((error as GILimitError).needed).toBe(9);
        expect(gpu.counters.pipelines).toBe(0);
    });
});

describe('the rays backend over the scene copy', () => {
    it('reports scene-copy slots and the mesh’s own triangle index, and front faces on the authored side', async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device, { buildIn: 'main' });
        const scene = platform.scene;
        const material = scene.createMaterial({ albedo: [1, 1, 1] });
        const mesh = scene.createMesh(quad());
        // A hidden instance first, so the traced ones don't sit in slots 0 and 1 by luck.
        scene.createInstance({ mesh, materials: [material], transform: translation(50, 0, 0), visible: false });
        const plain = scene.createInstance({ mesh, materials: [material], transform: translation(0, 0, 0) });
        // Mirrored in x: the authored +z side still faces up.
        const mirrored = scene.createInstance({ mesh, materials: [material], transform: [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 0, 0, 1] });
        await readyPlatform(gpu, platform);

        expect(platform.rays.building).toBe(false);
        expect(platform.rays.topLevelInstanceCount).toBe(2);
        const hit = traceDown(platform, 0.5, 0.2);
        expect(hit?.instanceIndex).toBe(platform.sceneCopy.instanceSlot(plain));
        // (0.5, 0.2) is inside the first triangle (0, 1, 2) of the quad.
        expect(hit?.triangleIndex).toBe(0);
        expect(hit?.frontFace).toBe(true);
        const mirroredHit = traceDown(platform, 4.3, -0.4);
        expect(mirroredHit?.instanceIndex).toBe(platform.sceneCopy.instanceSlot(mirrored));
        expect(mirroredHit?.frontFace).toBe(true);
        platform.dispose();
    });

    it('keeps an updated mesh’s old tree until the new one arrives, and drops removed meshes', async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device, { buildIn: 'main' });
        const scene = platform.scene;
        const material = scene.createMaterial({});
        const mesh = scene.createMesh(quad(1));
        const instance = scene.createInstance({ mesh, materials: [material], transform: translation(0, 0, 0) });
        await readyPlatform(gpu, platform);
        expect(traceDown(platform, 1.5, 0)).toBeNull();

        scene.updateMesh(mesh, quad(2));
        encodeFrame(gpu, platform);
        expect(platform.rays.building).toBe(true);
        // Meanwhile rays see the previous shape.
        expect(traceDown(platform, 1.5, 0)).toBeNull();
        expect(traceDown(platform, 0.5, 0)).not.toBeNull();
        await platform.rays.whenBuilt();
        encodeFrame(gpu, platform);
        expect(traceDown(platform, 1.5, 0)).not.toBeNull();

        scene.destroyInstance(instance);
        scene.destroyMesh(mesh);
        encodeFrame(gpu, platform);
        expect(platform.rays.topLevelInstanceCount).toBe(0);
        expect(traceDown(platform, 0.5, 0)).toBeNull();
        platform.dispose();
    });

    it('grows its buffers as meshes arrive, keeping the trees already placed, and bumps the module version', async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device, { buildIn: 'main' });
        const scene = platform.scene;
        const material = scene.createMaterial({});
        const first = scene.createMesh(quad());
        scene.createInstance({ mesh: first, materials: [material], transform: translation(0, 0, 0) });
        await readyPlatform(gpu, platform);
        const versionBefore = platform.rays.module.version;
        const memoryBefore = platform.rays.memory;

        // Enough triangles to outgrow the starting buffers.
        const big = new Float32Array(3 * 3 * 5000).map((_, index) => ((index * 7919) % 1000) / 100 + (index % 3 === 2 ? 20 : 0));
        const second = scene.createMesh({ positions: big, indices: Uint32Array.from({ length: 3 * 5000 }, (_, index) => index) });
        scene.createInstance({ mesh: second, materials: [material], transform: translation(100, 0, 0) });
        encodeFrame(gpu, platform);
        await platform.rays.whenBuilt();
        encodeFrame(gpu, platform);

        expect(platform.rays.memory.triangles).toBeGreaterThan(memoryBefore.triangles);
        expect(platform.rays.module.version).toBeGreaterThan(versionBefore);
        expect(platform.stats.memory.acceleration).toBe(platform.rays.memoryBytes);
        // The first mesh's tree was copied into the new buffers.
        expect(traceDown(platform, 0.5, 0.2)?.instanceIndex).toBe(0);
        platform.dispose();
    });

    it('throws a GILimitError naming maxStorageBufferBindingSize when the triangles outgrow one binding, every frame', async () => {
        // Room for the scene copy, but not for 48-byte triangle records of this mesh.
        const gpu = createPlatformGpu({ maxStorageBufferBindingSize: 400_000 });
        const platform = new GIPlatformImpl(gpu.device, { buildIn: 'main' });
        const scene = platform.scene;
        const triangles = 9000;
        const positions = new Float32Array(triangles * 9).map((_, index) => (index % 97) / 10);
        const mesh = scene.createMesh({ positions, indices: Uint32Array.from({ length: triangles * 3 }, (_, index) => index) });
        scene.createInstance({ mesh, materials: [scene.createMaterial({})], transform: translation(0, 0, 0) });
        await platform.prepare();
        encodeFrame(gpu, platform);
        await platform.rays.whenBuilt();
        for (let frame = 0; frame < 2; frame++) {
            const error = (() => {
                try {
                    encodeFrame(gpu, platform);
                } catch (caught) {
                    return caught;
                }
                return null;
            })();
            expect(error).toBeInstanceOf(GILimitError);
            expect((error as GILimitError).limit).toBe('maxStorageBufferBindingSize');
            expect((error as GILimitError).needed).toBeGreaterThanOrEqual(triangles * 48);
            expect((error as Error).message).toMatch(/requiredLimits/);
        }
        platform.dispose();
    });
});

describe('the platform object', () => {
    it('refuses to encode before prepare, and after the device is lost', async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device);
        expect(() => platform.encode(gpu.createEncoder(), FRAME)).toThrow(GINotReadyError);
        await platform.prepare();
        expect(platform.isReady).toBe(true);
        encodeFrame(gpu, platform);

        await gpu.loseDevice();
        expect(platform.isReady).toBe(false);
        const error = (() => {
            try {
                platform.encode(gpu.createEncoder(), FRAME);
            } catch (caught) {
                return caught;
            }
            return null;
        })();
        expect(error).toBeInstanceOf(GINotReadyError);
        expect((error as GINotReadyError).reason).toBe('device-lost');
        expect(() => platform.scene.createMaterial({})).toThrow(GINotReadyError);
    });

    it('encodes fallbacks first, passes units, skips paused frames, and reports memory per technique', async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device);
        const log: string[] = [];
        const top = platform.addTechnique(new RecordingTechnique('surfels', log));
        const below = platform.addTechnique(new RecordingTechnique('ddgi', log));
        platform.setFallback(top, below);
        expect(() => platform.setFallback(below, top)).toThrow(/loop/);
        platform.setOutput(top);
        expect(top.context?.fallback).toBeDefined();
        expect(top.context?.queries.has('rays')).toBe(true);
        expect(top.context?.scene).toBe(platform.sceneCopy);

        await platform.prepare();
        expect([top.prepared, below.prepared]).toEqual([1, 1]);
        encodeFrame(gpu, platform);
        expect(log).toEqual(['ddgi', 'surfels']);
        expect(top.frames[0].units).toBe(DEFAULT_UNITS_PER_FRAME);

        encodeFrame(gpu, platform, { ...FRAME, budget: { units: 64 } });
        expect(top.frames[1].units).toBe(64);
        expect(top.frames[1].frame).toBe(2);

        encodeFrame(gpu, platform, { ...FRAME, budget: { paused: true } });
        expect(top.frames.length).toBe(2);

        const memory = platform.stats.memory;
        expect(memory.techniques).toEqual({ surfels: 1000, ddgi: 1000 });
        expect(memory.total).toBe(memory.sceneCopy + memory.acceleration + 2000);

        expect(() => platform.removeTechnique(below)).toThrow(/fallback/);
        platform.setFallback(top, undefined);
        platform.removeTechnique(below);
        expect(below.disposed).toBe(true);
        platform.dispose();
        expect(top.disposed).toBe(true);
    });

    it('needs prepare again after a technique is added', async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device);
        await platform.prepare();
        platform.addTechnique(new RecordingTechnique('ddgi', []));
        expect(platform.isReady).toBe(false);
        expect(() => platform.encode(gpu.createEncoder(), FRAME)).toThrow(GINotReadyError);
        await platform.prepare();
        encodeFrame(gpu, platform);
        platform.dispose();
    });
});
