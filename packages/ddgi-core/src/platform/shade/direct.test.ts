import { describe, expect, it } from 'vitest';
import { GILimitError } from '../contract/errors.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { ComputeKernel, ComputeKernelOptions, KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import { hoistDirectives } from '../kernel-builder.ts';
import {
    DIRECT_KERNELS_WGSL,
    PlatformDeferredDirect,
    RAY_RESULT_BYTES,
    RAY_SCRATCH_HEADER_BYTES,
    RAY_SCRATCH_WORD,
    rayScratchBytes,
    rayScratchTraceWgsl,
    WORK_ITEM_BYTES,
} from './direct.ts';

/** WGSL sizes of the types the header uses. */
const TYPE_BYTES: Record<string, number> = { 'u32': 4, 'atomic<u32>': 4 };

/** The fields of `struct GiRayScratch` before its runtime-sized cells, with their sizes. */
function headerFields(code: string): Array<{ name: string; bytes: number }> {
    const body = /struct GiRayScratch \{([^}]*)\}/.exec(code)?.[1] ?? '';
    return body
        .split('\n')
        .map((line) => line.trim().replace(/,$/, ''))
        .filter((line) => line.includes(':') && !line.startsWith('//') && !line.startsWith('cells'))
        .map((line) => {
            const [name, type] = line.split(':').map((part) => part.trim());
            const array = /^array<vec4u, (\d+)>$/.exec(type);
            return { name, bytes: array ? Number(array[1]) * 16 : TYPE_BYTES[type] };
        });
}

/** Just enough of a device for the scratch and the encode, recording what it was asked to do. */
function fakeDevice(features: string[] = []) {
    const writes: Array<{ offset: number; words: number[] }> = [];
    const buffers: Array<{ descriptor: GPUBufferDescriptor; mapped: ArrayBuffer }> = [];
    const device = {
        features: new Set(features),
        limits: { maxStorageBufferBindingSize: 1 << 20, maxComputeWorkgroupsPerDimension: 65535 },
        createBuffer: (descriptor: GPUBufferDescriptor) => {
            const mapped = new ArrayBuffer(descriptor.size);
            buffers.push({ descriptor, mapped });
            return { ...descriptor, getMappedRange: () => mapped, unmap() {}, destroy() {} };
        },
        createBindGroup: (descriptor: GPUBindGroupDescriptor) => ({ ...descriptor }),
        queue: {
            writeBuffer: (_buffer: GPUBuffer, offset: number, data: Uint32Array) => writes.push({ offset, words: [...data] }),
        },
    } as unknown as GPUDevice;
    return { device, writes, buffers };
}

/** A kernel builder that "compiles" at once and remembers what it was asked for. */
function fakeKernels(): KernelBuilder & { readonly built: ComputeKernelOptions[] } {
    const built: ComputeKernelOptions[] = [];
    return {
        built,
        computeKernel: (options: ComputeKernelOptions): Promise<ComputeKernel> => {
            built.push(options);
            return Promise.resolve({ pipeline: {} as GPUComputePipeline, ownLayouts: [{} as GPUBindGroupLayout], bind() {} });
        },
    };
}

const SHADE_MODULE: WgslModule = { name: 'gi-shade', dependencies: [], code: () => '', entries: [], resources: () => [], version: 0, storageBufferCount: 0 };
const NO_TIMER: GpuTimerScope = { computePass: () => undefined };

describe('the ray scratch', () => {
    it('has a 256-byte header whose fields sit at RAY_SCRATCH_WORD', () => {
        const fields = headerFields(DIRECT_KERNELS_WGSL.direct);
        expect(fields.reduce((sum, field) => sum + field.bytes, 0)).toBe(RAY_SCRATCH_HEADER_BYTES);
        let offset = 0;
        const offsets = new Map<string, number>();
        for (const field of fields) {
            offsets.set(field.name, offset / 4);
            offset += field.bytes;
        }
        for (const [name, word] of Object.entries(RAY_SCRATCH_WORD)) expect(offsets.get(name), name).toBe(word);
    });

    it('is laid out header, ray results, then work items of 32 bytes', () => {
        expect(RAY_RESULT_BYTES).toBe(8);
        expect(WORK_ITEM_BYTES).toBe(32);
        expect(rayScratchBytes(1000, 300)).toBe(256 + 8000 + 9600);
    });

    it('declares every platform global with the gi prefix', () => {
        const code = [rayScratchTraceWgsl('ddgiRayScratch', 'subgroups'), DIRECT_KERNELS_WGSL.direct, DIRECT_KERNELS_WGSL.args].join('\n');
        for (const match of code.matchAll(/^(?:fn|struct|const|override|var<[^>]*>|var)\s+([A-Za-z_]\w*)/gm)) {
            expect(match[1].toLowerCase().startsWith('gi'), match[1]).toBe(true);
        }
    });

    it('gives a tracing kernel the subgroup list first and the fallback without subgroup operations', () => {
        const subgroups = rayScratchTraceWgsl('scratch', 'subgroups');
        expect(subgroups.trimStart().startsWith('enable subgroups;')).toBe(true);
        expect(subgroups).toContain('subgroupExclusiveAdd(');
        expect(subgroups).toContain('atomicAdd(&scratch.workCount, subgroupTotal)');
        const fallback = rayScratchTraceWgsl('scratch', 'atomics');
        expect(fallback).not.toMatch(/enable subgroups|subgroup[A-Z]\w*\(/);
        expect(fallback).toContain('atomicAdd(&scratch.workCount, 1u)');
        // The directive still comes first once the kernel builder has put modules before it.
        expect(hoistDirectives(`// ---- module gi-shade ----\nfn giSurfaceAt() {}\n${subgroups}`).startsWith('enable subgroups;')).toBe(true);
        for (const name of ['fn giStoreRayResult(', 'fn giNeedsDirect(', 'fn giSurfaceRecord(', 'fn giDeferDirect(']) expect(fallback).toContain(name);
    });
});

describe('deferred direct light', () => {
    it('chooses subgroups when the device has them, and refuses them when it has not', () => {
        expect(new PlatformDeferredDirect(fakeDevice(['subgroups']).device, fakeKernels(), SHADE_MODULE, 'gi').defaultWorkList).toBe('subgroups');
        const plain = new PlatformDeferredDirect(fakeDevice().device, fakeKernels(), SHADE_MODULE, 'gi');
        expect(plain.defaultWorkList).toBe('atomics');
        expect(() => plain.traceWgsl('scratch', 'subgroups')).toThrow(/subgroups/);
    });

    it('compiles gi/direct once, with the hit-shading module, and its args kernel without it', async () => {
        const kernels = fakeKernels();
        const direct = new PlatformDeferredDirect(fakeDevice().device, kernels, SHADE_MODULE, 'gi');
        expect(() => direct.createScratch({ label: 'rays', rays: 8, workItems: 8 })).toThrow(/prepare/);
        await Promise.all([direct.prepare(), direct.prepare()]);
        expect(kernels.built.map((kernel) => kernel.label)).toEqual(['gi/direct-args', 'gi/direct']);
        expect(kernels.built[0].modules).toEqual([]);
        expect(kernels.built[1].modules).toEqual([SHADE_MODULE]);
        // gi/direct binds one storage buffer of its own (the scratch); with the tracer's 3 and the scene copy's 2 that is 6.
        expect(kernels.built[1].ownGroups?.[0].entries).toHaveLength(1);
    });

    it('makes a scratch with its header written and its ray results bindable on their own', async () => {
        const { device, buffers } = fakeDevice();
        const direct = new PlatformDeferredDirect(device, fakeKernels(), SHADE_MODULE, 'gi');
        await direct.prepare();
        const scratch = direct.createScratch({ label: 'rays', rays: 1000, workItems: 300 });
        expect(scratch.bytes).toBe(rayScratchBytes(1000, 300));
        expect(scratch.rayResults).toMatchObject({ offset: RAY_SCRATCH_HEADER_BYTES, size: 8000 });
        const header = new Uint32Array(buffers[buffers.length - 1].mapped, 0, RAY_SCRATCH_HEADER_BYTES / 4);
        expect(header[RAY_SCRATCH_WORD.itemsStart]).toBe(1000);
        expect(header[RAY_SCRATCH_WORD.itemCapacity]).toBe(300);
        expect(header[RAY_SCRATCH_WORD.workCount]).toBe(0);
        expect(scratch.countsOffset).toBe(RAY_SCRATCH_WORD.itemsThisFrame * 4);
        expect(() => direct.createScratch({ label: 'too big', rays: 1 << 20, workItems: 0 })).toThrow(GILimitError);
    });

    it('records one pass named after its owner, writing the frame and the groups once a frame', async () => {
        const { device, writes } = fakeDevice();
        const direct = new PlatformDeferredDirect(device, fakeKernels(), SHADE_MODULE, 'gi');
        await direct.prepare();
        const scratch = direct.createScratch({ label: 'rays', rays: 64, workItems: 64 });
        const passes: Array<{ label: string; commands: string[] }> = [];
        const encoder = {
            beginComputePass: (descriptor: GPUComputePassDescriptor) => {
                const pass = { label: descriptor.label ?? '', commands: [] as string[] };
                passes.push(pass);
                return {
                    setPipeline: () => pass.commands.push('pipeline'),
                    setBindGroup: () => pass.commands.push('group'),
                    dispatchWorkgroups: () => pass.commands.push('dispatch'),
                    dispatchWorkgroupsIndirect: () => pass.commands.push('indirect'),
                    end: () => pass.commands.push('end'),
                };
            },
        } as unknown as GPUCommandEncoder;
        const frame = { frame: 12, groupMask: 0xf, owner: 'ddgi', timer: NO_TIMER };
        direct.encode(encoder, scratch, frame);
        direct.encode(encoder, scratch, frame);
        direct.encode(encoder, scratch, { ...frame, frame: 13 });
        expect(passes.map((pass) => pass.label)).toEqual(['ddgi/gi-direct', 'ddgi/gi-direct', 'ddgi/gi-direct']);
        expect(passes[0].commands.filter((command) => command === 'dispatch' || command === 'indirect')).toEqual(['dispatch', 'indirect']);
        // The second round of a frame writes nothing new.
        expect(writes).toEqual([
            { offset: RAY_SCRATCH_WORD.frame * 4, words: [12, 0xf] },
            { offset: RAY_SCRATCH_WORD.frame * 4, words: [13, 0xf] },
        ]);
    });
});
