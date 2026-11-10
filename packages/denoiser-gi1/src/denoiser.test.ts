import { describe, expect, it } from 'vitest';
import { frameConstants, reprojectionCellSize } from './camera.ts';
import { MAX_BLUR_MASK } from './constants.ts';
import { GI1Denoiser } from './gi1-denoiser.ts';
import { blurRadius, filterGI, linearDepths, removeNaNs, reprojectGI, type ReprojectInputs } from './reference.ts';
import { CAPSAICIN_FORMATS, CORE_FORMATS, denoiserKernels, ENTRY_POINT, formatsFor } from './wgsl.ts';

/** WebGPU's default (and guaranteed) limits per shader stage: what any host's device allows. */
const DEFAULT_STORAGE_BUFFERS_PER_STAGE = 8;
const DEFAULT_STORAGE_TEXTURES_PER_STAGE = 4;
const DEFAULT_SAMPLED_TEXTURES_PER_STAGE = 16;

describe("the denoiser's binding budget", () => {
    const count = (layout: GPUBindGroupLayoutDescriptor, kind: 'buffer' | 'storageTexture' | 'texture'): number =>
        [...layout.entries].filter((entry) => (kind === 'buffer' ? entry.buffer?.type === 'storage' || entry.buffer?.type === 'read-only-storage' : entry[kind] !== undefined)).length;

    for (const [name, formats] of [['Capsaicin\'s', CAPSAICIN_FORMATS], ['the fallback', CORE_FORMATS]] as const) {
        const kernels = denoiserKernels(formats);

        it(`fits WebGPU's default limits per stage with ${name} formats: no storage buffers, at most 3 storage textures`, () => {
            const storageTextures = Object.fromEntries(kernels.map((kernel) => [kernel.entryPoint, count(kernel.layout, 'storageTexture')]));
            expect(storageTextures).toEqual({ [ENTRY_POINT.reproject]: 3, [ENTRY_POINT.filter]: 2, [ENTRY_POINT.keepGuides]: 2 });
            for (const kernel of kernels) {
                expect(count(kernel.layout, 'buffer'), kernel.entryPoint).toBe(0);
                expect(count(kernel.layout, 'buffer')).toBeLessThanOrEqual(DEFAULT_STORAGE_BUFFERS_PER_STAGE);
                expect(count(kernel.layout, 'storageTexture')).toBeLessThanOrEqual(DEFAULT_STORAGE_TEXTURES_PER_STAGE);
                expect(count(kernel.layout, 'texture')).toBeLessThanOrEqual(DEFAULT_SAMPLED_TEXTURES_PER_STAGE);
            }
        });

        it(`declares in WGSL exactly the bindings its layout has, with ${name} formats`, () => {
            for (const kernel of kernels) {
                const declared = [...kernel.code.matchAll(/@group\(0\) @binding\((\d+)\)/g)].map((match) => Number(match[1])).sort((a, b) => a - b);
                expect(declared, kernel.entryPoint).toEqual([...kernel.layout.entries].map((entry) => entry.binding));
                expect(kernel.code).toContain(`fn ${kernel.entryPoint}(`);
                expect(kernel.code).not.toContain('${');
            }
            const reproject = kernels.find((kernel) => kernel.entryPoint === ENTRY_POINT.reproject);
            expect(reproject?.code).toContain(`texture_storage_2d<${formats.blurMask}, write>`);
            expect(reproject?.code).toContain(`texture_storage_2d<${formats.colorDelta}, write>`);
        });
    }

    it("takes Capsaicin's formats only where the device has texture-formats-tier1 (register D65)", () => {
        expect(formatsFor(new Set(['texture-formats-tier1']))).toEqual({ blurMask: 'r8snorm', colorDelta: 'r16float' });
        expect(formatsFor(new Set())).toEqual({ blurMask: 'rgba8snorm', colorDelta: 'r32float' });
    });
});

describe('the CPU reference', () => {
    it("removeNaNs is the identity on ordinary values and caps the large ones near 1e4 (gi_denoiser.hlsl:64-92)", () => {
        for (const value of [0, 1e-3, 0.5, 1, 7.5, 64, 1000]) expect(removeNaNs(value)).toBeCloseTo(value, 9);
        // Between -1 and 0 the saturate takes it to 0 (below -1 it would read 1e4; no input is negative).
        expect(removeNaNs(-0.5)).toBe(0);
        expect(removeNaNs(1e6)).toBeLessThan(1e4);
        // As on D3D, whose saturate takes NaN to 0 (register D71).
        expect(removeNaNs(Number.NaN)).toBe(0);
        expect(removeNaNs(Number.POSITIVE_INFINITY)).toBe(0);
    });

    it('turns the mask into Capsaicin\'s radius: at least 1 above 0, rounded, 0 at or below 0', () => {
        expect(blurRadius(1)).toBe(MAX_BLUR_MASK);
        expect(blurRadius(0)).toBe(0);
        expect(blurRadius(-1 / 8)).toBe(0);
        expect(blurRadius(0.01 / 8)).toBe(1);
        expect(blurRadius(3.49 / 8)).toBe(3);
        expect(blurRadius(3.5 / 8)).toBe(4);
    });

    // A still camera looking straight down at a plane: every pixel the same surface, frame after frame.
    const width = 32;
    const height = 24;
    const projection = (near: number, far: number, fovY: number, aspect: number): number[] => {
        const f = 1 / Math.tan(fovY / 2);
        return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
    };
    // Eye at (0, 2, 0) looking down -y: view maps world -y to view -z.
    const view = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, -2, 1];
    const camera = { view, projection: projection(0.1, 100, 1, width / height), prevView: view, prevProjection: projection(0.1, 100, 1, width / height), near: 0.1, far: 100, position: [0, 2, 0] as [number, number, number] };

    function planeInputs(radiance: (pixel: number) => [number, number, number, number], previous: { color: Float32Array; delta: Float32Array }): ReprojectInputs {
        const pixels = width * height;
        // The plane y = 0 seen from 2 m above: every pixel's depth from the projection of its point.
        const depth = new Float32Array(pixels);
        // View depth 2 everywhere (straight down at a plane 2 m away): NDC z = (p10 × z + p14) / -z at z = -2.
        const p = camera.projection;
        depth.fill((p[10] * -2 + p[14]) / 2);
        const normal = new Float32Array(pixels * 4);
        for (let pixel = 0; pixel < pixels; pixel++) normal.set([0, 1, 0, 0], pixel * 4);
        return {
            width,
            height,
            radiance: Float32Array.from({ length: pixels * 4 }, (_, index) => radiance(Math.floor(index / 4))[index % 4]),
            depth,
            normal,
            motion: new Float32Array(pixels * 4),
            previousColor: previous.color,
            previousColorDelta: previous.delta,
            previousDepth: depth,
            previousNormal: normal,
        };
    }

    function runFrame(inputs: ReprojectInputs): { color: Float32Array; delta: Float32Array; irradiance: Float32Array; mask: Float32Array } {
        const frame = frameConstants(camera, [0, 0], width, height);
        const reprojected = reprojectGI(inputs, frame);
        const radius = Int32Array.from(reprojected.blurMask, blurRadius);
        const linear = linearDepths(inputs.depth, width, height, frame);
        const first = filterGI(reprojected.color, radius, linear, inputs.normal, width, height, [1, 0]);
        const second = filterGI(first.color, radius, linear, inputs.normal, width, height, [0, 1]);
        if (!second.irradiance) throw new Error('the second pass writes the irradiance');
        return { color: second.color, delta: reprojected.colorDelta, irradiance: second.irradiance, mask: reprojected.blurMask };
    }

    it('passes a still constant image through unchanged, and the blur mask falls to 0 as the history grows', () => {
        const pixels = width * height;
        let previous: { color: Float32Array; delta: Float32Array } = { color: new Float32Array(pixels * 4), delta: new Float32Array(pixels) };
        const masks: number[] = [];
        for (let frame = 0; frame < 20; frame++) {
            const result = runFrame(planeInputs(() => [0.3, 0.6, 0.9, 1], previous));
            for (let pixel = 0; pixel < pixels; pixel++) {
                expect(result.irradiance[pixel * 4]).toBeCloseTo(0.3, 6);
                expect(result.irradiance[pixel * 4 + 1]).toBeCloseTo(0.6, 6);
                expect(result.irradiance[pixel * 4 + 2]).toBeCloseTo(0.9, 6);
            }
            const centre = (height / 2) * width + width / 2;
            masks.push(result.mask[centre] * MAX_BLUR_MASK);
            previous = { color: result.color, delta: result.delta };
        }
        // No history, then one more sample a frame: 8, 7, ... (the 3 × 3 tent adds weight-averaged counts).
        expect(masks[0]).toBe(MAX_BLUR_MASK);
        expect(masks[1]).toBeCloseTo(MAX_BLUR_MASK - 1, 9);
        expect(masks.at(-1)).toBe(0);
    });

    it('blurs a pixel without history and leaves a converged one alone', () => {
        const pixels = width * height;
        // Converge on a constant, then a frame of noise: converged pixels keep their history mean
        // almost unchanged; with no history the same noise is blurred to near its mean.
        let previous: { color: Float32Array; delta: Float32Array } = { color: new Float32Array(pixels * 4), delta: new Float32Array(pixels) };
        for (let frame = 0; frame < 20; frame++) {
            const result = runFrame(planeInputs(() => [1, 1, 1, 1], previous));
            previous = { color: result.color, delta: result.delta };
        }
        const noise = (pixel: number): [number, number, number, number] => {
            const value = pixel % 2 === 0 ? 0.5 : 1.5;
            return [value, value, value, 1];
        };
        const converged = runFrame(planeInputs(noise, previous));
        const fresh = runFrame(planeInputs(noise, { color: new Float32Array(pixels * 4), delta: new Float32Array(pixels) }));
        const centre = (height / 2) * width + width / 2;
        expect(converged.mask[centre]).toBe(0);
        expect(fresh.mask[centre]).toBe(1);
        // A blurred checker of 0.5 / 1.5 is near 1; the converged pixel moved little from 1.
        expect(Math.abs(fresh.irradiance[centre * 4] - 1)).toBeLessThan(0.1);
        expect(Math.abs(converged.irradiance[centre * 4] - 1)).toBeLessThan(0.05);
    });

    it("sizes the reprojection's cell as Capsaicin's cell_size (gi1.cpp:1839-1843)", () => {
        const fovY = 1;
        const p = projection(0.1, 100, fovY, 16 / 9);
        expect(reprojectionCellSize(p, 1280, 720)).toBeCloseTo(Math.tan(fovY * 8 * Math.max(1 / 720, 720 / 1280 ** 2)), 12);
    });
});

describe('the GI1Denoiser object', () => {
    it('compiles its three kernels on a device and refuses encode before configure', async () => {
        const created: string[] = [];
        const device = {
            createBuffer: (descriptor: GPUBufferDescriptor) => ({ size: descriptor.size, destroy() {} }),
            createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => descriptor,
            createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => descriptor,
            createShaderModule: (descriptor: GPUShaderModuleDescriptor) => descriptor,
            createComputePipelineAsync: (descriptor: GPUComputePipelineDescriptor) => {
                created.push(descriptor.compute.entryPoint ?? '');
                return Promise.resolve({});
            },
            queue: { writeBuffer() {} },
            features: new Set<string>(),
        } as unknown as GPUDevice;
        const denoiser = await GI1Denoiser.create(device);
        expect(created.sort()).toEqual([ENTRY_POINT.filter, ENTRY_POINT.keepGuides, ENTRY_POINT.reproject].sort());
        expect(denoiser.output).toBeNull();
        expect(() => denoiser.encode({} as GPUCommandEncoder, { radiance: {} as GPUTexture }, { depth: {} as GPUTexture, normal: {} as GPUTexture, motion: {} as GPUTexture }, {} as never)).toThrow(/configure/);
        expect(() => denoiser.configure({ width: 0, height: 4 })).toThrow(RangeError);
    });
});
