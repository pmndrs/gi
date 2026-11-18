import { describe, expect, it } from 'vitest';
import { GILimitError, GINotReadyError } from './contract/errors.ts';
import type { FrameInput } from './contract/frame.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame } from './contract/technique.ts';
import type { WgslModule } from './contract/wgsl.ts';
import { DEFAULT_UNITS_PER_FRAME, GIPlatformImpl } from './gi-platform.ts';
import { checkComputeLimits, isSupported, KERNEL_COMPUTE_LIMITS, optionalFeatures, requiredLimits } from './device-requirements.ts';
import { BLEND_WORKGROUP_SIZE } from '../techniques/ddgi/wgsl/blend.ts';
import { countGroupStorageBuffers, countStorageBuffers, orderModules, placeModules, PlatformKernelBuilder } from './kernel-builder.ts';
import { createFakeGpu, type FakeBuffer, type FakeGpu } from './scene/testing/fake-gpu.ts';
import { createCpuTracer } from './trace/cpu-traversal.ts';
import { TOP_HEADER_WORD } from './trace/layout.ts';
import type { TraceSceneData } from './trace/scene-assembly.ts';
import type { TopLevelBvh } from './trace/top-level.ts';

/** A texture stand-in for hit shading's sky (T1.4): it only needs to exist, give views and be destroyed. */
function fakeTexture(descriptor: GPUTextureDescriptor): unknown {
    const size = descriptor.size as GPUExtent3DDict;
    return {
        label: descriptor.label ?? '',
        width: size.width,
        height: size.height ?? 1,
        depthOrArrayLayers: size.depthOrArrayLayers ?? 1,
        format: descriptor.format,
        usage: descriptor.usage,
        dimension: '2d',
        destroyed: false,
        createView: () => ({}),
        destroy(this: { destroyed: boolean }) {
            this.destroyed = true;
        },
    };
}

/** No-op compute pass, for the sky projection the platform records on its first frame. */
const FAKE_COMPUTE_PASS = { setBindGroup() {}, setPipeline() {}, dispatchWorkgroups() {}, end() {} };

/** The fake GPU plus what the platform needs beyond the scene copy: layouts, `lost`, a few limits, and the sky's textures (T1.4). */
function createPlatformGpu(limits: Partial<GPUSupportedLimits> = {}): FakeGpu & { loseDevice(): Promise<void> } {
    const fake = createFakeGpu(limits);
    let loseDevice: () => void = () => {};
    const lost = new Promise<GPUDeviceLostInfo>((resolve) => {
        loseDevice = () => resolve({ reason: 'unknown', message: 'test' } as GPUDeviceLostInfo);
    });
    const device = fake.device as unknown as Record<string, unknown>;
    // A real device is an EventTarget (`uncapturederror`).
    const events = new EventTarget();
    Object.assign(device, {
        addEventListener: events.addEventListener.bind(events),
        removeEventListener: events.removeEventListener.bind(events),
        dispatchEvent: events.dispatchEvent.bind(events),
        lost,
        createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ descriptor }),
        createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ descriptor }),
        features: new Set<string>(),
        createTexture: fakeTexture,
        createSampler: () => ({}),
    });
    (device.queue as Record<string, unknown>).writeTexture = () => {};
    Object.assign(device.limits as object, { maxBindGroups: 4, maxComputeWorkgroupsPerDimension: 65535, ...limits });
    return {
        ...fake,
        createEncoder: () => Object.assign(fake.createEncoder(), { beginComputePass: () => FAKE_COMPUTE_PASS }),
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

    it("counts the technique's own storage buffers: DDGI's 8 of 8 compiles, a 9th throws a GILimitError", async () => {
        const gpu = createPlatformGpu();
        const builder = new PlatformKernelBuilder(gpu.device, 'test');
        const storage = (binding: number, type: GPUBufferBindingType = 'storage'): GPUBindGroupLayoutEntry => ({ binding, visibility: 4, buffer: { type } });
        const platformModules = [module('gi-trace', 3), module('gi-scene', 2)];
        const ddgiGroup = { entries: [storage(0, 'read-only-storage'), storage(1), storage(2)] };
        // A uniform buffer and a texture don't count against the storage-buffer limit.
        const extras = { entries: [{ binding: 0, visibility: 4, buffer: {} }, { binding: 1, visibility: 4, texture: {} }] };
        expect(countGroupStorageBuffers([ddgiGroup, extras])).toBe(3);

        const kernel = await builder.computeKernel({ label: 'ddgi-shaped', modules: platformModules, code: '', entryPoint: 'main', ownGroups: [ddgiGroup, extras] });
        expect(kernel.ownLayouts.length).toBe(2);

        const nine = { entries: [...ddgiGroup.entries, storage(3)] };
        const error = await builder
            .computeKernel({ label: 'nine', modules: platformModules, code: '', entryPoint: 'main', ownGroups: [nine] })
            .catch((caught: unknown) => caught);
        expect(error).toBeInstanceOf(GILimitError);
        expect((error as GILimitError).limit).toBe('maxStorageBuffersPerShaderStage');
        expect((error as GILimitError).needed).toBe(9);
    });
});

describe('device requirements', () => {
    const adapter = (limits: Partial<GPUSupportedLimits>, features: string[]): GPUAdapter =>
        ({ limits, features: new Set(features) }) as unknown as GPUAdapter;

    it("requiredLimits asks for the adapter's own maximums, never more", () => {
        // The texture store wants every layer the adapter allows (M4): one per texture.
        const storage = { maxStorageBufferBindingSize: 2147483644, maxBufferSize: 2147483648, maxStorageBuffersPerShaderStage: 10, maxTextureArrayLayers: 2048 };
        const compute = { maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupSizeX: 1024 };
        // Compute limits: what the kernels need, not the adapter's maximum.
        expect(requiredLimits(adapter({ ...storage, ...compute }, []))).toEqual({ ...storage, ...KERNEL_COMPUTE_LIMITS });
        // Capped at what the adapter has, so requestDevice can't fail over them.
        const small = { maxComputeInvocationsPerWorkgroup: 128, maxComputeWorkgroupSizeX: 128 };
        expect(requiredLimits(adapter({ ...storage, ...small }, []))).toEqual({ ...storage, ...small });
    });

    it('asks for workgroups as large as the largest kernel uses', () => {
        expect(KERNEL_COMPUTE_LIMITS.maxComputeWorkgroupSizeX).toBeGreaterThanOrEqual(BLEND_WORKGROUP_SIZE);
        expect(KERNEL_COMPUTE_LIMITS.maxComputeInvocationsPerWorkgroup).toBeGreaterThanOrEqual(BLEND_WORKGROUP_SIZE);
    });

    it('checkComputeLimits names the compute limit a device is below (compatibility mode defaults to 128)', () => {
        expect(() => checkComputeLimits({ maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256 })).not.toThrow();
        const error = (() => {
            try {
                checkComputeLimits({ maxComputeInvocationsPerWorkgroup: 128, maxComputeWorkgroupSizeX: 256 });
            } catch (caught) {
                return caught;
            }
            return null;
        })();
        expect(error).toBeInstanceOf(GILimitError);
        expect((error as GILimitError).limit).toBe('maxComputeInvocationsPerWorkgroup');
        expect((error as GILimitError).needed).toBe(256);
        expect((error as GILimitError).available).toBe(128);
    });

    it('prepare throws that GILimitError before compiling anything', async () => {
        const gpu = createPlatformGpu({ maxComputeWorkgroupSizeX: 128 });
        const platform = new GIPlatformImpl(gpu.device, { buildIn: 'main' });
        const error = await platform.prepare().catch((caught: unknown) => caught);
        expect(error).toBeInstanceOf(GILimitError);
        expect((error as GILimitError).limit).toBe('maxComputeWorkgroupSizeX');
        expect(gpu.counters.pipelines).toBe(0);
        platform.dispose();
    });

    it('optionalFeatures requests only what the adapter has', () => {
        expect(optionalFeatures(adapter({}, ['timestamp-query', 'subgroups', 'shader-f16']))).toEqual(['subgroups', 'timestamp-query']);
        // The compression families the texture store reads (CompressedPixelData), where present.
        expect(optionalFeatures(adapter({}, ['texture-compression-bc', 'texture-compression-astc']))).toEqual(['texture-compression-bc', 'texture-compression-astc']);
        expect(optionalFeatures(adapter({}, []))).toEqual([]);
    });

    it('isSupported needs the default 8 storage buffers per stage and 4 bind groups', () => {
        const device = (limits: Partial<GPUSupportedLimits>): GPUDevice => ({ limits }) as unknown as GPUDevice;
        expect(isSupported(device({ maxStorageBuffersPerShaderStage: 8, maxBindGroups: 4 }))).toBe(true);
        expect(isSupported(device({ maxStorageBuffersPerShaderStage: 4, maxBindGroups: 4 }))).toBe(false);
        expect(isSupported(device({ maxStorageBuffersPerShaderStage: 8, maxBindGroups: 3 }))).toBe(false);
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
    it("gives hosts hit shading's live sampleTextures setting, on by default (DECISIONS 2026-11-18)", () => {
        const platform = new GIPlatformImpl(createPlatformGpu().device);
        expect(platform.hitShadingSettings.sampleTextures).toBe(true);
        // The same object hit shading reads each frame, so the host's change is what the next encode writes.
        platform.hitShadingSettings.sampleTextures = false;
        expect(platform.shade.settings.sampleTextures).toBe(false);
        platform.dispose();
    });

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

        // Disposing after the loss frees what is left, and a second dispose does nothing.
        platform.dispose();
        platform.dispose();
        expect(platform.isReady).toBe(false);
    });

    it('compiles the ray-batch kernels only for a technique that asks for them, and never on first use (T2.14)', async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device);
        platform.addTechnique(new RecordingTechnique('inline tracer', []));
        const batch = { mode: 'closest' as const, rays: gpu.device.createBuffer({ size: 32, usage: 0 }), hits: gpu.device.createBuffer({ size: 24, usage: 0 }), count: 1 };
        // Before any prepare: not ready, whatever the techniques asked for.
        expect(() => platform.rays.encodeBatch(gpu.createEncoder(), batch)).toThrow(GINotReadyError);
        await platform.prepare();
        expect(platform.rays.batchesReady).toBe(false);
        const pipelinesWithout = gpu.counters.pipelines;
        expect(() => platform.rays.encodeBatch(gpu.createEncoder(), batch)).toThrow(/rayBatches/);
        // Refusing compiled nothing.
        expect(gpu.counters.pipelines).toBe(pipelinesWithout);
        expect(platform.stats.recompiles).toBe(0);

        // A technique that traces in batches says so; the next prepare compiles the two batch kernels.
        const batching = new RecordingTechnique('batch tracer', []);
        (batching as { requirements: GITechnique['requirements'] }).requirements = { queries: ['rays'], rayBatches: true };
        platform.addTechnique(batching);
        await platform.prepare();
        expect(platform.rays.batchesReady).toBe(true);
        expect(gpu.counters.pipelines - pipelinesWithout).toBe(2);
        expect(() => platform.rays.encodeBatch(gpu.createEncoder(), batch)).not.toThrow();
        expect(platform.stats.recompiles).toBe(0);

        // Tools that call encodeBatch themselves ask with the platform option.
        const tool = new GIPlatformImpl(createPlatformGpu().device, { rayBatches: true });
        await tool.prepare();
        expect(tool.rays.batchesReady).toBe(true);
    });

    it('disposes once: buffers destroyed, techniques disposed, later calls refused', async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device, { buildIn: 'main' });
        const technique = platform.addTechnique(new RecordingTechnique('ddgi', []));
        const mesh = platform.scene.createMesh(quad());
        platform.scene.createInstance({ mesh, materials: [platform.scene.createMaterial({})], transform: translation(0, 0, 0) });
        await readyPlatform(gpu, platform);

        platform.dispose();
        expect(technique.disposed).toBe(true);
        expect(gpu.buffers.every((buffer) => buffer.destroyed)).toBe(true);
        expect(() => platform.dispose()).not.toThrow();
        expect(platform.isReady).toBe(false);
        expect(() => platform.encode(gpu.createEncoder(), FRAME)).toThrow(/disposed/);
        expect(() => platform.scene.createMaterial({})).toThrow(/disposed/);
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

        // A light changed while paused reaches the techniques with the next frame they see (T2.4a).
        const lamp = platform.scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 5, position: [0, 1, 0], range: 0 });
        encodeFrame(gpu, platform, { ...FRAME, budget: { paused: true } });
        platform.scene.updateLight(lamp, { intensity: 7 });
        encodeFrame(gpu, platform, { ...FRAME, budget: { paused: true } });
        encodeFrame(gpu, platform);
        const seen = top.frames[2].changes.lights;
        expect(seen).toHaveLength(1);
        expect(seen[0].before).toBeNull();
        expect(seen[0].after?.intensity).toBe(7);
        encodeFrame(gpu, platform);
        expect(top.frames[3].changes.lights).toEqual([]);

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

    it("counts hit shading's sky in the scene copy and the total", async () => {
        const gpu = createPlatformGpu();
        const platform = new GIPlatformImpl(gpu.device);
        platform.addTechnique(new RecordingTechnique('ddgi', []));
        await platform.prepare();
        encodeFrame(gpu, platform);

        // The radiance map alone is hundreds of kilobytes; leaving it out once made the total too small.
        const shadeBytes = platform.shade.memoryBytes;
        expect(platform.shade.sky.memoryBytes).toBeGreaterThan(0);
        expect(shadeBytes).toBeGreaterThan(platform.shade.sky.memoryBytes);
        const memory = platform.stats.memory;
        expect(memory.sceneCopy).toBe(platform.sceneCopy.memoryBytes + shadeBytes);
        expect(memory.total).toBe(platform.sceneCopy.memoryBytes + shadeBytes + memory.acceleration + 1000);
        platform.dispose();
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

describe('uncaptured GPU errors', () => {
    /** An `uncapturederror` event with this message, as a device would dispatch it. */
    function errorEvent(message: string): Event {
        return Object.assign(new Event('uncapturederror'), { error: { message } });
    }

    it('forwards the platform own errors as warnings, listening without taking the slot, until disposed', () => {
        const gpu = createPlatformGpu();
        const device = gpu.device as unknown as EventTarget & { onuncapturederror?: unknown };
        const warnings: string[] = [];
        const originalWarn = console.warn;
        console.warn = (message: string) => warnings.push(message);
        try {
            const platform = new GIPlatformImpl(gpu.device, { label: 'mine' });
            expect(device.onuncapturederror).toBeUndefined();
            device.dispatchEvent(errorEvent('Invalid ComputePipeline "mine group 0" is invalid'));
            device.dispatchEvent(errorEvent('Invalid RenderPipeline "renderPipeline_cornell-white_21"'));
            expect(warnings).toHaveLength(1);
            expect(warnings[0]).toContain('mine group 0');
            platform.dispose();
            device.dispatchEvent(errorEvent('Invalid Buffer "mine scene table"'));
            expect(warnings).toHaveLength(1);
        } finally {
            console.warn = originalWarn;
        }
    });
});
