import { describe, expect, it, vi } from 'vitest';
import { HashGridCache } from './world-cache/hash-grid-cache.ts';
import { GILimitError } from '../../platform/contract/errors.ts';
import type { KernelBuilder, ModuleBindingEntry, WgslModule } from '../../platform/contract/wgsl.ts';
import { INDIRECT_ARGS_MODULE } from '../../platform/compute/wgsl.ts';
import { MIN_STORAGE_BUFFERS_PER_STAGE, MIN_STORAGE_TEXTURES_PER_STAGE } from '../../platform/device-requirements.ts';
import { countGroupStorageBuffers, countStorageBuffers, orderModules, PlatformKernelBuilder } from '../../platform/kernel-builder.ts';
import { SCENE_MODULE_ENTRIES } from '../../platform/scene/wgsl.ts';
import { SHADE_MODULE_ENTRIES } from '../../platform/shade/wgsl.ts';
import { TRACE_MODULE_ENTRIES } from '../../platform/trace/wgsl.ts';
import type { TechniqueContext } from '../../platform/contract/technique.ts';
import { cameraRelativeInverseViewProjection, invertMat4, multiplyMat4 } from './camera.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PROBE_CELLS, PROBE_SIZE, PROBE_SPAWN_TILE_SIZE, screenProbeLayout } from './settings.ts';
import { RadianceCacheTechnique } from './radiance-cache-technique.ts';
import { COUNTER_NAMES, COUNTERS_BYTES } from './wgsl/screen-probes.ts';
import { CAPSAICIN_FORMATS, CORE_FORMATS, denoiserKernels, reprojectionCellSize } from '@pmndrs/denoiser-gi1';
import { denoiserCamera, GI_DENOISER_LABEL } from './gi-denoiser.ts';
import { screenProbeCellSize } from './screen-irradiance.ts';

describe('settings (gi1.h)', () => {
    it("are Capsaicin's defaults", () => {
        expect(PROBE_SIZE).toBe(8);
        expect(PROBE_CELLS).toBe(64);
        expect(PROBE_SPAWN_TILE_SIZE).toBe(16);
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS).toMatchObject({
            useDirectLighting: true,
            useTemporalFeedback: false,
            useMultibounce: true,
            useResampling: false,
            hashGridCacheCellSize: 32,
            hashGridCacheMinCellSize: 0.1,
            hashGridCacheTileCellRatio: 8,
            hashGridCacheNumBuckets: 14,
            hashGridCacheNumTilesPerBucket: 4,
            hashGridCacheMaxSampleCount: 16,
        });
        const technique = new RadianceCacheTechnique({ useDirectLighting: false });
        expect(technique.settings.useDirectLighting).toBe(false);
        expect(technique.settings.useMultibounce).toBe(true);
        // Reprojection reads the host's velocity, as Capsaicin's does (gi1.comp:326).
        expect(technique.requirements).toEqual({ queries: ['rays'], screen: { velocity: true } });
    });

    it('size the probes as Capsaicin does (gi1.cpp:113-129)', () => {
        const layout = screenProbeLayout(1280, 720);
        expect(layout.probeCount).toEqual([160, 90]);
        expect(layout.atlasSize).toEqual([1280, 720]);
        expect(layout.maxProbeSpawnCount).toBe(80 * 45);
        expect(layout.maxRayCount).toBe(3600 * 64);
        expect(layout.probeMaskMipCount).toBe(8);
        // 1080p, the source map's figures: 8,160 spawn tiles, 522,240 rays.
        const fullHd = screenProbeLayout(1920, 1080);
        expect(fullHd.maxProbeSpawnCount).toBe(8160);
        expect(fullHd.maxRayCount).toBe(522240);
        expect(fullHd.atlasSize).toEqual([1920, 1080]);
        // A screen that is not whole tiles rounds the atlas up to them.
        expect(screenProbeLayout(1000, 700).atlasSize).toEqual([1000, 704]);
        expect(() => screenProbeLayout(0, 10)).toThrow(RangeError);
    });

    it('has room for every counter ClearCounters resets', () => {
        expect(COUNTER_NAMES.length * 4).toBeLessThanOrEqual(COUNTERS_BYTES);
    });
});

describe('the camera-relative inverse view-projection', () => {
    const perspective = (fovY: number, aspect: number, near: number, far: number): number[] => {
        // WebGPU's 0..1 depth, column-major.
        const f = 1 / Math.tan(fovY / 2);
        return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
    };
    const lookAtView = (eye: [number, number, number]): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -eye[0], -eye[1], -eye[2], 1];

    it('inverts a matrix', () => {
        const projection = perspective(1, 16 / 9, 0.1, 100);
        const product = multiplyMat4(projection, invertMat4(projection) ?? []);
        product.forEach((value, index) => expect(value).toBeCloseTo(index % 5 === 0 ? 1 : 0, 9));
        expect(invertMat4(new Array(16).fill(0))).toBeNull();
    });

    it('reconstructs a world point from its depth, relative to the eye', () => {
        const eye: [number, number, number] = [1000, 20, -500];
        const camera = { position: eye, view: lookAtView(eye), projection: perspective(1, 1, 0.1, 100) };
        const point = [1003, 21, -510];
        const clip = multiplyMat4(multiplyMat4(camera.projection, camera.view), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...point, 1]).slice(12, 16);
        const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1];
        const inverse = cameraRelativeInverseViewProjection(camera);
        const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row] * ndc[3]);
        [0, 1, 2].forEach((axis) => expect(eye[axis] + h[axis] / h[3]).toBeCloseTo(point[axis], 6));
    });
});

// ---- The binding budget: every kernel of the technique through the platform's kernel builder ----

function stubModule(name: string, entries: readonly ModuleBindingEntry[], dependencies: readonly WgslModule[] = []): WgslModule {
    const storageBufferCount = entries.filter((entry) => entry.buffer?.type === 'storage' || entry.buffer?.type === 'read-only-storage').length;
    return { name, dependencies, code: () => '', entries, resources: () => [], version: 0, storageBufferCount };
}

/** Just enough of a GPUDevice for `RadianceCacheTechnique.prepare` through the real kernel builder (no GPU, no bytes). */
function budgetDevice(maxStorageBuffersPerShaderStage: number): GPUDevice {
    const texture = (descriptor: GPUTextureDescriptor) => ({ label: descriptor.label, createView: () => ({}), destroy() {} });
    return {
        limits: {
            maxStorageBuffersPerShaderStage,
            maxBindGroups: 4,
            maxStorageBufferBindingSize: 2 ** 31,
            maxBufferSize: 2 ** 31,
            maxTextureDimension2D: 8192,
            maxComputeWorkgroupsPerDimension: 65535,
        },
        createBuffer: (descriptor: GPUBufferDescriptor) => ({ size: descriptor.size, label: descriptor.label, destroy() {} }),
        createTexture: texture,
        pushErrorScope() {},
        popErrorScope: () => Promise.resolve(null),
        createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => descriptor,
        createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => descriptor,
        createShaderModule: (descriptor: GPUShaderModuleDescriptor) => descriptor,
        createComputePipelineAsync: (descriptor: GPUComputePipelineDescriptor) => Promise.resolve({ label: descriptor.label }),
        createBindGroup: (descriptor: GPUBindGroupDescriptor) => descriptor,
        queue: { writeBuffer() {}, writeTexture() {} },
        features: new Set<string>(),
    } as unknown as GPUDevice;
}

/** The technique's context with the real kernel builder; records each kernel's storage buffers and whether the builder refused it. */
/** Storage textures a kernel binds in the compute stage: its own groups' and its modules'. */
function countStorageTextures(modules: readonly WgslModule[], groups: readonly GPUBindGroupLayoutDescriptor[]): number {
    const COMPUTE = 0x4;
    const own = groups.flatMap((group) => [...group.entries]).filter((entry) => entry.storageTexture !== undefined && (entry.visibility & COMPUTE) !== 0).length;
    const fromModules = orderModules(modules)
        .flatMap((module) => [...module.entries])
        .filter((entry) => entry.storageTexture !== undefined).length;
    return own + fromModules;
}

function budgetContext(device: GPUDevice): {
    context: TechniqueContext;
    storageBuffers: Map<string, number>;
    storageTextures: Map<string, number>;
    refused: Map<string, Promise<boolean>>;
} {
    const scene = stubModule('gi-scene', SCENE_MODULE_ENTRIES);
    const rays = stubModule('gi-rays', TRACE_MODULE_ENTRIES);
    const shade = stubModule('gi-shade', SHADE_MODULE_ENTRIES, [scene, rays]);
    const builder = new PlatformKernelBuilder(device, 'test');
    const storageBuffers = new Map<string, number>();
    const storageTextures = new Map<string, number>();
    const refused = new Map<string, Promise<boolean>>();
    const kernels: KernelBuilder = {
        computeKernel: (options) => {
            storageBuffers.set(options.label, countStorageBuffers(orderModules(options.modules)) + countGroupStorageBuffers(options.ownGroups ?? []));
            storageTextures.set(options.label, countStorageTextures(options.modules, options.ownGroups ?? []));
            const kernel = builder.computeKernel(options);
            refused.set(
                options.label,
                kernel.then(
                    () => false,
                    (error: unknown) => error instanceof GILimitError,
                ),
            );
            return kernel;
        },
    };
    const word = { buffer: {}, offset: 0 };
    const utility = { buffer: {}, indices: {}, count: word, total: word, memoryBytes: 16, capacity: 1, encode() {}, dispose() {} };
    const compute = {
        indirectModule: INDIRECT_ARGS_MODULE,
        createIndirectArgs: () => Promise.resolve(utility),
        createCompaction: () => Promise.resolve(utility),
        createPrefixScan: () => Promise.resolve(utility),
    };
    const direct = { prepare: () => Promise.resolve(), defaultWorkList: 'ray-order', traceWgsl: () => '' };
    const context = { device, kernels, compute, shade: { module: shade, direct }, timer: { computePass: () => undefined } } as unknown as TechniqueContext;
    return { context, storageBuffers, storageTextures, refused };
}

describe("the technique's binding budget (register D6): the platform's 8 storage buffers per stage", () => {
    /** The screen probes' kernels and their storage buffers (the world cache's are in hash-grid.test.ts). */
    const SCREEN_KERNELS: Record<string, number> = {
        'radiance-cache/ClearCounters': 1,
        'radiance-cache/ClearProbeMask': 0,
        'radiance-cache/InitCachedTileLRU': 1,
        'radiance-cache/ReprojectScreenProbes': 2,
        'radiance-cache/CountScreenProbes': 4,
        'radiance-cache/ScatterScreenProbes': 3,
        'radiance-cache/SpawnScreenProbes': 2,
        'radiance-cache/CompactScreenProbes': 6,
        'radiance-cache/PatchScreenProbes': 4,
        'radiance-cache/SampleScreenProbes': 7,
        'radiance-cache/PopulateScreenProbes (ray-order)': 8,
        'radiance-cache/BlendScreenProbes': 6,
        'radiance-cache/ReorderScreenProbes': 3,
        'radiance-cache/CopyPreviousScreen': 0,
    };

    it('holds for every kernel of the technique at 8, the history kernels (RC4.1) included', async () => {
        const { context, storageBuffers } = budgetContext(budgetDevice(MIN_STORAGE_BUFFERS_PER_STAGE));
        const technique = new RadianceCacheTechnique();
        technique.attach(context);
        await technique.prepare();
        for (const [label, count] of Object.entries(SCREEN_KERNELS)) expect(storageBuffers.get(label), label).toBe(count);
        for (const [label, count] of storageBuffers) expect(count, label).toBeLessThanOrEqual(MIN_STORAGE_BUFFERS_PER_STAGE);
    });

    it("holds WebGPU's default 4 storage textures per stage for every kernel; BlendScreenProbes uses all 4", async () => {
        expect(MIN_STORAGE_TEXTURES_PER_STAGE).toBe(4);
        const { context, storageTextures } = budgetContext(budgetDevice(MIN_STORAGE_BUFFERS_PER_STAGE));
        const technique = new RadianceCacheTechnique();
        technique.attach(context);
        await technique.prepare();
        expect(storageTextures.get('radiance-cache/BlendScreenProbes')).toBe(4);
        expect(storageTextures.get('radiance-cache/ReprojectScreenProbes')).toBe(3);
        expect(storageTextures.get('radiance-cache/SampleScreenProbes')).toBe(2);
        for (const [label, count] of storageTextures) expect(count, label).toBeLessThanOrEqual(MIN_STORAGE_TEXTURES_PER_STAGE);
    });

    it('is what the kernel builder checks: at 7, prepare throws GILimitError and refuses exactly the kernels that bind 8', async () => {
        const { context, storageBuffers, refused } = budgetContext(budgetDevice(MIN_STORAGE_BUFFERS_PER_STAGE - 1));
        // The world cache prepares first and would refuse its own kernels at 7 (hash-grid.test.ts covers
        // those); held aside here so the screen probes' kernels are built too.
        const worldCache = vi.spyOn(HashGridCache.prototype, 'prepare').mockResolvedValue(undefined);
        const technique = new RadianceCacheTechnique();
        technique.attach(context);
        await expect(technique.prepare()).rejects.toThrow(GILimitError);
        worldCache.mockRestore();
        const answers = await Promise.all([...refused].map(async ([label, answer]) => [label, await answer] as const));
        const refusedLabels = answers.filter(([, wasRefused]) => wasRefused).map(([label]) => label);
        const atEight = [...storageBuffers].filter(([, count]) => count === MIN_STORAGE_BUFFERS_PER_STAGE).map(([label]) => label);
        expect(refusedLabels.sort()).toEqual(atEight.sort());
        expect(refusedLabels).toContain('radiance-cache/PopulateScreenProbes (ray-order)');
        for (const label of Object.keys(SCREEN_KERNELS)) expect(storageBuffers.has(label), label).toBe(true);
    });
});

// ---- The GI denoiser (RC4.3): a package of its own, built outside the kernel builder ----

describe('the GI denoiser in the technique (RC4.3)', () => {
    const storage = (layout: GPUBindGroupLayoutDescriptor, kind: 'buffer' | 'storageTexture'): number =>
        [...layout.entries].filter((entry) => (kind === 'buffer' ? entry.buffer?.type === 'storage' || entry.buffer?.type === 'read-only-storage' : entry.storageTexture !== undefined)).length;

    it("fits the platform's 8 storage buffers and WebGPU's 4 storage textures per stage, and prepare compiles it", async () => {
        for (const kernel of [...denoiserKernels(CAPSAICIN_FORMATS), ...denoiserKernels(CORE_FORMATS)]) {
            expect(storage(kernel.layout, 'buffer'), kernel.entryPoint).toBeLessThanOrEqual(MIN_STORAGE_BUFFERS_PER_STAGE);
            expect(storage(kernel.layout, 'storageTexture'), kernel.entryPoint).toBeLessThanOrEqual(MIN_STORAGE_TEXTURES_PER_STAGE);
        }
        const device = budgetDevice(MIN_STORAGE_BUFFERS_PER_STAGE);
        const compiled: string[] = [];
        const createPipeline = device.createComputePipelineAsync.bind(device);
        device.createComputePipelineAsync = (descriptor: GPUComputePipelineDescriptor) => {
            if (descriptor.label?.startsWith(GI_DENOISER_LABEL)) compiled.push(descriptor.compute.entryPoint ?? '');
            return createPipeline(descriptor);
        };
        const { context } = budgetContext(device);
        const technique = new RadianceCacheTechnique();
        technique.attach(context);
        await technique.prepare();
        expect(technique.settings.useGIDenoiser).toBe(true);
        expect(technique.denoiser).not.toBeNull();
        expect(compiled.sort()).toEqual(denoiserKernels(CORE_FORMATS).map((kernel) => kernel.entryPoint).sort());
    });

    it("sizes its reprojection cell as the screen probes' (Capsaicin's one cell_size, gi1.cpp:1839-1843)", () => {
        const f = 1 / Math.tan(0.6);
        const projection = [f / (16 / 9), 0, 0, 0, 0, f, 0, 0, 0, 0, -1, -1, 0, 0, -0.1, 0];
        expect(reprojectionCellSize(projection, 1280, 720)).toBeCloseTo(screenProbeCellSize({ projection }, 1280, 720), 12);
    });

    it("hands the denoiser the jitter in pixels, y down (FrameCamera.jitter is NDC, y up)", () => {
        const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const camera = { position: [0, 0, 0] as [number, number, number], view: identity, projection: identity, near: 0.1, far: 100, jitter: [2 / 1280, -2 / 720] as [number, number] };
        const converted = denoiserCamera(camera, camera, { width: 1280, height: 720 });
        expect(converted.jitter?.[0]).toBeCloseTo(1, 12);
        expect(converted.jitter?.[1]).toBeCloseTo(1, 12);
        // And back, as the denoiser turns it into NDC (its jitterToNdc): the same offset.
        const ndc = [(2 * (converted.jitter?.[0] ?? 0)) / 1280, (-2 * (converted.jitter?.[1] ?? 0)) / 720];
        expect(ndc[0]).toBeCloseTo(camera.jitter[0], 12);
        expect(ndc[1]).toBeCloseTo(camera.jitter[1], 12);
    });
});
