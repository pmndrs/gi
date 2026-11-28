import { describe, expect, it } from 'vitest';
import { wgslNameProblems } from '../testing/wgsl-names.ts';
import { GILimitError } from '../contract/errors.ts';
import { compactOnCpu, exclusiveScanOnCpu, indirectArgsOnCpu } from './cpu-reference.ts';
import { PlatformGpuCompute } from './gpu-compute.ts';
import { scanWgsl, SCAN_TILE_ITEMS } from './scan.ts';
import { COMPACTION_IO, INDIRECT_ARGS_MODULE, INDIRECT_WRITER_WGSL, SCAN_IN_PLACE_IO, SCAN_TO_OUTPUT_IO } from './wgsl.ts';

describe('the CPU references', () => {
    it('scan exclusively and wrap at 2^32 like u32 adds', () => {
        const { prefix, total } = exclusiveScanOnCpu(new Uint32Array([3, 0, 5, 1]));
        expect([...prefix]).toEqual([0, 3, 3, 8]);
        expect(total).toBe(9);
        const wrapped = exclusiveScanOnCpu(new Uint32Array([0xffffffff, 2, 1]));
        expect([...wrapped.prefix]).toEqual([0, 0xffffffff, 1]);
        expect(wrapped.total).toBe(2);
        expect(exclusiveScanOnCpu(new Uint32Array(0)).total).toBe(0);
    });

    it('compact to the indices of non-zero flags, in order', () => {
        expect([...compactOnCpu(new Uint32Array([0, 1, 7, 0, 1]))]).toEqual([1, 2, 4]);
        expect(compactOnCpu(new Uint32Array([0, 0])).length).toBe(0);
    });

    it('size indirect dispatches, splitting over y past the per-dimension limit', () => {
        expect(indirectArgsOnCpu(0, 64, 65535)).toEqual([0, 0, 1]);
        expect(indirectArgsOnCpu(1, 64, 65535)).toEqual([1, 1, 1]);
        expect(indirectArgsOnCpu(64, 64, 65535)).toEqual([1, 1, 1]);
        expect(indirectArgsOnCpu(65, 64, 65535)).toEqual([2, 1, 1]);
        expect(indirectArgsOnCpu(65535 * 64, 64, 65535)).toEqual([65535, 1, 1]);
        // 10 M items in workgroups of 64: 156,250 workgroups in three even rows, about 156k launched, not 196k.
        expect(indirectArgsOnCpu(10_000_000, 64, 65535)).toEqual([52084, 3, 1]);
        const [x, y] = indirectArgsOnCpu(10_000_000, 64, 65535);
        expect(x * y * 64).toBeGreaterThanOrEqual(10_000_000);
        expect(x * y * 64 - 10_000_000).toBeLessThan(y * 64);
    });

    it('never pass the per-dimension limit, whatever the u32 count', () => {
        // 2^32 - 1 items, one per workgroup: 65,537 rows would be needed; both dimensions stop at the limit.
        expect(indirectArgsOnCpu(0xffffffff, 1, 65535)).toEqual([65535, 65535, 1]);
        for (const count of [65535 ** 2, 65535 ** 2 + 1, 0xffffffff]) {
            for (const perWorkgroup of [1, 32, 64, 256]) {
                const [x, y] = indirectArgsOnCpu(count, perWorkgroup, 65535);
                expect(x).toBeLessThanOrEqual(65535);
                expect(y).toBeLessThanOrEqual(65535);
            }
        }
    });
});

describe('the GPU utilities WGSL', () => {
    const shaders: Record<string, string> = {
        indirectWriter: INDIRECT_WRITER_WGSL,
        indirectModule: INDIRECT_ARGS_MODULE.code({ group: 0, firstBinding: 0 }),
    };
    for (const io of [SCAN_TO_OUTPUT_IO, SCAN_IN_PLACE_IO, COMPACTION_IO]) {
        for (const reduction of ['subgroups', 'workgroup'] as const) shaders[`${io.name} (${reduction})`] = scanWgsl({ reduction, io });
    }

    it('use no reserved word as a name, and no function shares a name with a variable', () => {
        for (const [name, code] of Object.entries(shaders)) expect(wgslNameProblems(code), name).toEqual([]);
    });

    it('enable subgroups only on the subgroup path', () => {
        for (const [name, code] of Object.entries(shaders)) {
            if (name.endsWith('(subgroups)')) expect(code.startsWith('enable subgroups;'), name).toBe(true);
            // No subgroup operation or builtin (comments may still say "subgroups").
            else expect(code, name).not.toMatch(/enable subgroups|subgroup[A-Z]\w*\(|subgroup_/);
        }
    });
});

/** Just enough of a device to make the utilities, recording what they were asked to do. */
function fakeDevice(options: { features?: string[]; maxStorageBufferBindingSize?: number } = {}) {
    const writes: Array<{ buffer: string; offset: number; words: number[] }> = [];
    const dispatches: Array<{ pipeline: string; workgroups: number[]; offsets?: number[] }> = [];
    let offsets: number[] = [];
    let pipeline = '';
    const buffer = (descriptor: GPUBufferDescriptor) => {
        const mapped = new ArrayBuffer(descriptor.size);
        return { ...descriptor, mapped, getMappedRange: () => mapped, unmap() {}, destroy() {} };
    };
    const device = {
        features: new Set(options.features ?? []),
        limits: { maxStorageBufferBindingSize: options.maxStorageBufferBindingSize ?? 1 << 27, maxComputeWorkgroupsPerDimension: 65535, minUniformBufferOffsetAlignment: 256 },
        createBuffer: buffer,
        createBindGroup: (descriptor: GPUBindGroupDescriptor) => ({ ...descriptor }),
        createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ ...descriptor }),
        createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ ...descriptor }),
        createShaderModule: (descriptor: GPUShaderModuleDescriptor) => ({ ...descriptor }),
        createComputePipelineAsync: async (descriptor: GPUComputePipelineDescriptor) => ({ label: descriptor.compute.entryPoint }),
        queue: {
            writeBuffer: (target: GPUBuffer, offset: number, data: Uint32Array) => writes.push({ buffer: target.label, offset, words: [...data] }),
        },
    } as unknown as GPUDevice;
    const pass = {
        setPipeline: (chosen: GPUComputePipeline) => (pipeline = chosen.label),
        setBindGroup: (_index: number, _group: GPUBindGroup, dynamicOffsets: number[] = []) => (offsets = [...dynamicOffsets]),
        dispatchWorkgroups: (...workgroups: number[]) => dispatches.push({ pipeline, workgroups, ...(offsets.length > 0 ? { offsets } : {}) }),
    } as unknown as GPUComputePassEncoder;
    const storage = (size: number) => buffer({ label: 'caller buffer', size, usage: 0x80 }) as unknown as GPUBuffer;
    return { device, pass, writes, dispatches, storage };
}

describe('PlatformGpuCompute', () => {
    it('takes subgroups where the device has them, and the fallback otherwise', () => {
        expect(new PlatformGpuCompute(fakeDevice({ features: ['subgroups'] }).device).reduction).toBe('subgroups');
        expect(new PlatformGpuCompute(fakeDevice().device).reduction).toBe('workgroup');
        expect(new PlatformGpuCompute(fakeDevice({ features: ['subgroups'] }).device, { reduction: 'workgroup' }).reduction).toBe('workgroup');
        expect(() => new PlatformGpuCompute(fakeDevice().device, { reduction: 'subgroups' })).toThrow(/subgroups/);
    });

    it('refuses a capacity larger than one storage binding, naming the limit', async () => {
        const gpu = fakeDevice({ maxStorageBufferBindingSize: 1024 });
        const compute = new PlatformGpuCompute(gpu.device);
        await expect(compute.createPrefixScan({ label: 'test', input: gpu.storage(4096), capacity: 1024 })).rejects.toBeInstanceOf(GILimitError);
        await expect(compute.createPrefixScan({ label: 'test', input: gpu.storage(4096), capacity: 0 })).rejects.toThrow(/capacity/);
        await expect(compute.createPrefixScan({ label: 'test', input: gpu.storage(16), capacity: 256 })).rejects.toThrow(/holds 16 bytes/);
    });

    it('scans in three dispatches, and an empty scan still writes its total', async () => {
        const gpu = fakeDevice({ features: ['subgroups'] });
        const compute = new PlatformGpuCompute(gpu.device);
        const scan = await compute.createPrefixScan({ label: 'test', input: gpu.storage(4 * 10_000), capacity: 10_000 });
        scan.encode(gpu.pass, 10_000);
        const tiles = Math.ceil(10_000 / SCAN_TILE_ITEMS);
        expect(gpu.dispatches).toEqual([
            { pipeline: 'scanReduce', workgroups: [tiles, 1], offsets: [0] },
            { pipeline: 'scanAcrossTiles', workgroups: [1], offsets: [0] },
            { pipeline: 'scanDownsweep', workgroups: [tiles, 1], offsets: [0] },
        ]);
        expect(gpu.writes).toEqual([{ buffer: 'test scan params', offset: 0, words: [10_000, tiles, 0, 0] }]);
        // The same count again reuses its slot; zero items get a slot of their own and run only the kernel that writes the total.
        gpu.dispatches.length = 0;
        scan.encode(gpu.pass, 10_000);
        expect(gpu.writes).toHaveLength(1);
        gpu.dispatches.length = 0;
        scan.encode(gpu.pass, 0);
        expect(gpu.dispatches).toEqual([{ pipeline: 'scanAcrossTiles', workgroups: [1], offsets: [256] }]);
        expect(gpu.writes[1]).toEqual({ buffer: 'test scan params', offset: 256, words: [0, 0, 0, 0] });
        expect(() => scan.encode(gpu.pass, 10_001)).toThrow(RangeError);
        expect(scan.total.offset).toBe(0);
    });

    it('gives every count its own slot, written once, so encodes in one submit keep their counts', async () => {
        const gpu = fakeDevice();
        const compute = new PlatformGpuCompute(gpu.device);
        const scan = await compute.createPrefixScan({ label: 'test', input: gpu.storage(4 * 1000), capacity: 1000 });
        // 70 different counts (more than the 64 slots of 4 pages), then the first ones again.
        for (let count = 1; count <= 70; count++) scan.encode(gpu.pass, count);
        const firstPageWrites = gpu.writes.filter((write) => write.words[0] <= 16);
        expect(firstPageWrites.map((write) => write.offset)).toEqual(Array.from({ length: 16 }, (_, slot) => slot * 256));
        // One write per count: a slot, once written, is never written again.
        expect(gpu.writes).toHaveLength(70);
        expect(new Set(gpu.writes.map((write) => write.words[0])).size).toBe(70);
        // At most 4 pages are kept: count 1's page was dropped, so it gets a new slot.
        scan.encode(gpu.pass, 1);
        expect(gpu.writes).toHaveLength(71);
        // Count 70 is still kept, and reuses its slot.
        scan.encode(gpu.pass, 70);
        expect(gpu.writes).toHaveLength(71);
        expect(scan.memoryBytes).toBe((Math.ceil(1000 / SCAN_TILE_ITEMS) + 1) * 4 + 4 + 4 * 16 * 256);
    });

    it('reports the memory it allocated, the kept indices included when it made them', async () => {
        const gpu = fakeDevice();
        const compute = new PlatformGpuCompute(gpu.device);
        const own = await compute.createCompaction({ label: 'test', flags: gpu.storage(4 * 5000), capacity: 5000 });
        const given = await compute.createCompaction({ label: 'test', flags: gpu.storage(4 * 5000), output: gpu.storage(4 * 5000), capacity: 5000 });
        // Tile sums, the count word, and one page of 16 parameter slots of 256 bytes.
        const scratch = (Math.ceil(5000 / SCAN_TILE_ITEMS) + 1) * 4 + 4 + 16 * 256;
        expect(given.memoryBytes).toBe(scratch);
        expect(own.memoryBytes).toBe(scratch + 5000 * 4);
    });

    it('writes the indirect writer its count word, items per workgroup and the device limit once, at creation', async () => {
        const gpu = fakeDevice();
        const compute = new PlatformGpuCompute(gpu.device);
        const args = await compute.createIndirectArgs({ label: 'test', count: { buffer: gpu.storage(64), offset: 8 }, itemsPerWorkgroup: 64 });
        const params = (args as unknown as { params: { mapped: ArrayBuffer } }).params;
        expect([...new Uint32Array(params.mapped)]).toEqual([2, 64, 65535, 0]);
        args.encode(gpu.pass);
        expect(gpu.dispatches).toEqual([{ pipeline: 'giWriteIndirectArgs', workgroups: [1] }]);
        await expect(compute.createIndirectArgs({ label: 'test', count: { buffer: gpu.storage(64), offset: 2 }, itemsPerWorkgroup: 64 })).rejects.toThrow(/multiple of 4/);
        await expect(compute.createIndirectArgs({ label: 'test', count: { buffer: gpu.storage(64), offset: 0 }, itemsPerWorkgroup: 0 })).rejects.toThrow(/itemsPerWorkgroup/);
    });
});
