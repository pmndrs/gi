/**
 * The platform's GPU utilities (RC1.3; the contract is
 * platform/contract/compute.ts): prefix scan, stream compaction and indirect
 * dispatch arguments.
 *
 * **One scan.** The scan and the compaction run the platform's three scan
 * kernels (reduce, across tiles, downsweep; ./scan.ts), with an io from
 * ./wgsl.ts. Subgroups first; on a
 * device without `subgroups` the same kernels are compiled with the declared
 * fallback in workgroup memory (./workgroup-scan.ts), which assumes nothing
 * about subgroup lanes. Both give the same results (the
 * `platform-compute` GPU page checks it on the RTX 3060 and the Intel UHD).
 *
 * **Compiled once per platform.** Each kind of kernel compiles when the first
 * utility of that kind is made and is shared by every later one; a utility
 * itself only allocates its scratch and a bind group.
 */

import { createScanPipelines, encodeScan, SCAN_BINDING, SCAN_PARAMS_BYTES, scanParams, scanTileSumWords, type ScanIo, type ScanPipelines } from './scan.ts';
import { GILimitError } from '../contract/errors.ts';
import type {
    CompactionOptions,
    ComputeReduction,
    GpuCompaction,
    GpuComputeUtilities,
    GpuIndirectArgs,
    GpuPrefixScan,
    GpuWord,
    IndirectArgsOptions,
    PrefixScanOptions,
    GpuScreenHistory,
    GpuScreenSpaceGI,
    ScreenHistoryOptions,
    ScreenSpaceGIOptions,
} from '../contract/compute.ts';
import { compileScreenHistoryKernel, ScreenHistory, type ScreenHistoryKernel } from '../screen/screen-history.ts';
import { compileScreenSpaceGIKernel, ScreenSpaceGI, type ScreenSpaceGIKernel } from '../screen/ssgi.ts';
import type { WgslModule } from '../contract/wgsl.ts';
import {
    COMPACTION_IO,
    COMPUTE_IO_BINDING,
    INDIRECT_ARGS_BYTES,
    INDIRECT_ARGS_MODULE,
    INDIRECT_WRITER_BINDING,
    INDIRECT_WRITER_ENTRY_POINT,
    INDIRECT_WRITER_PARAMS_BYTES,
    INDIRECT_WRITER_WGSL,
    RESULT_BYTES,
    SCAN_IN_PLACE_IO,
    SCAN_TO_OUTPUT_IO,
} from './wgsl.ts';

/** `GPUBufferUsage` bits; the globals don't exist in Node, where tests import this file. */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80, indirect: 0x100 } as const;
/** `GPUShaderStage.COMPUTE`. */
const SHADER_STAGE_COMPUTE = 0x4;
const WORD_BYTES = 4;

/** The three uses of the scan kernels, each with its own io and bind group layout. */
type ScanVariant = 'to-output' | 'in-place' | 'compaction';

const SCAN_VARIANT_IO: Readonly<Record<ScanVariant, ScanIo>> = {
    'to-output': SCAN_TO_OUTPUT_IO,
    'in-place': SCAN_IN_PLACE_IO,
    compaction: COMPACTION_IO,
};

/** One variant's compiled kernels and the layout their bind groups use. */
interface ScanKernels {
    readonly layout: GPUBindGroupLayout;
    readonly pipelines: ScanPipelines;
}

interface WriterKernel {
    readonly layout: GPUBindGroupLayout;
    readonly pipeline: GPUComputePipeline;
}

export interface PlatformGpuComputeOptions {
    /** Prefix for pipeline labels (the platform's label). */
    readonly label?: string;
    /**
     * The path to take. Defaults to `subgroups` when the device has them and
     * the fallback otherwise; a test or a timing page may ask for the fallback
     * on a device with subgroups.
     */
    readonly reduction?: ComputeReduction;
}

/** One layout entry of a compute-only buffer binding. */
function bufferEntry(binding: number, type: GPUBufferBindingType, hasDynamicOffset = false): GPUBindGroupLayoutEntry {
    return { binding, visibility: SHADER_STAGE_COMPUTE, buffer: { type, hasDynamicOffset } };
}

/** Group 0 of each scan variant (the table in ./wgsl.ts). One layout for all three kernels, so one bind group serves them. */
function scanLayoutEntries(variant: ScanVariant): GPUBindGroupLayoutEntry[] {
    const entries = [
        bufferEntry(SCAN_BINDING.tileSums, 'storage'),
        // Each encode picks its own parameter slot (ScanRun).
        bufferEntry(SCAN_BINDING.params, 'uniform', true),
        bufferEntry(COMPUTE_IO_BINDING.source, variant === 'in-place' ? 'storage' : 'read-only-storage'),
        bufferEntry(COMPUTE_IO_BINDING.result, 'storage'),
    ];
    if (variant !== 'in-place') entries.push(bufferEntry(COMPUTE_IO_BINDING.destination, 'storage'));
    return entries;
}

function checkStorageBuffer(buffer: GPUBuffer, bytes: number, what: string, label: string): void {
    if ((buffer.usage & BUFFER_USAGE.storage) === 0) throw new Error(`@pmndrs/gi: ${label}: ${what} needs STORAGE usage`);
    if (buffer.size < bytes) throw new RangeError(`@pmndrs/gi: ${label}: ${what} holds ${buffer.size} bytes, the capacity needs ${bytes}`);
}

export class PlatformGpuCompute implements GpuComputeUtilities {
    readonly reduction: ComputeReduction;
    readonly indirectModule: WgslModule = INDIRECT_ARGS_MODULE;

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly scanKernels = new Map<ScanVariant, Promise<ScanKernels>>();
    private writerKernel: Promise<WriterKernel> | null = null;
    private screenHistoryKernel: Promise<ScreenHistoryKernel> | null = null;
    private screenSpaceGIKernel: Promise<ScreenSpaceGIKernel> | null = null;

    constructor(device: GPUDevice, options: PlatformGpuComputeOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'gi';
        const hasSubgroups = device.features.has('subgroups');
        this.reduction = options.reduction ?? (hasSubgroups ? 'subgroups' : 'workgroup');
        if (this.reduction === 'subgroups' && !hasSubgroups) throw new Error('@pmndrs/gi: the GPU utilities were asked for subgroups on a device without the "subgroups" feature');
    }

    async createPrefixScan(options: PrefixScanOptions): Promise<GpuPrefixScan> {
        const inPlace = options.output === undefined || options.output === options.input;
        const variant: ScanVariant = inPlace ? 'in-place' : 'to-output';
        this.checkCapacity(options.capacity, options.label);
        const bytes = options.capacity * WORD_BYTES;
        checkStorageBuffer(options.input, bytes, 'the input', options.label);
        if (!inPlace && options.output) checkStorageBuffer(options.output, bytes, 'the output', options.label);
        const kernels = await this.scanKernelsFor(variant);
        const label = options.label;
        let run = new ScanRun(this.device, kernels, { label, capacity: options.capacity, source: options.input, destination: inPlace ? undefined : options.output });
        return {
            get capacity() {
                return run.capacity;
            },
            get total() {
                return run.result;
            },
            get memoryBytes() {
                return run.memoryBytes;
            },
            encode: (pass, count) => run.encode(pass, count),
            rebind: (next) => {
                const nextInPlace = next.output === undefined || next.output === next.input;
                if (nextInPlace !== inPlace) throw new Error(`@pmndrs/gi: ${label}: rebind keeps the scan ${inPlace ? 'in place' : 'to an output'}`);
                this.checkCapacity(next.capacity, label);
                checkStorageBuffer(next.input, next.capacity * WORD_BYTES, 'the input', label);
                if (!nextInPlace && next.output) checkStorageBuffer(next.output, next.capacity * WORD_BYTES, 'the output', label);
                // Destroying the old scratch is safe for command buffers already submitted (WebGPU waits for them).
                run.dispose();
                run = new ScanRun(this.device, kernels, { label, capacity: next.capacity, source: next.input, destination: nextInPlace ? undefined : next.output });
            },
            dispose: () => run.dispose(),
        };
    }

    async createCompaction(options: CompactionOptions): Promise<GpuCompaction> {
        this.checkCapacity(options.capacity, options.label);
        const bytes = options.capacity * WORD_BYTES;
        checkStorageBuffer(options.flags, bytes, 'the flags', options.label);
        if (options.output) checkStorageBuffer(options.output, bytes, 'the output', options.label);
        const kernels = await this.scanKernelsFor('compaction');
        const indices =
            options.output ??
            this.device.createBuffer({ label: `${options.label} kept indices`, size: bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        const ownsIndices = options.output === undefined;
        const run = new ScanRun(this.device, kernels, { label: options.label, capacity: options.capacity, source: options.flags, destination: indices });
        return {
            capacity: options.capacity,
            indices,
            count: run.result,
            get memoryBytes() {
                return run.memoryBytes + (ownsIndices ? indices.size : 0);
            },
            encode: (pass, count) => run.encode(pass, count),
            dispose: () => {
                run.dispose();
                if (ownsIndices) indices.destroy();
            },
        };
    }

    async createIndirectArgs(options: IndirectArgsOptions): Promise<GpuIndirectArgs> {
        const { count, itemsPerWorkgroup, label } = options;
        if (!Number.isInteger(itemsPerWorkgroup) || itemsPerWorkgroup < 1) throw new RangeError(`@pmndrs/gi: ${label}: itemsPerWorkgroup must be a whole number of at least 1, got ${itemsPerWorkgroup}`);
        if (!Number.isInteger(count.offset) || count.offset % WORD_BYTES !== 0 || count.offset < 0) throw new RangeError(`@pmndrs/gi: ${label}: the count's offset must be a multiple of 4, got ${count.offset}`);
        checkStorageBuffer(count.buffer, count.offset + WORD_BYTES, 'the count', label);
        const kernel = await this.writerKernelFor();
        return new IndirectArgsWriter(this.device, kernel, options);
    }

    async createScreenHistory(options: ScreenHistoryOptions): Promise<GpuScreenHistory> {
        this.screenHistoryKernel ??= compileScreenHistoryKernel(this.device, this.label);
        return new ScreenHistory(this.device, await this.screenHistoryKernel, options);
    }

    async createScreenSpaceGI(options: ScreenSpaceGIOptions): Promise<GpuScreenSpaceGI> {
        this.screenSpaceGIKernel ??= compileScreenSpaceGIKernel(this.device, this.label);
        return new ScreenSpaceGI(this.device, await this.screenSpaceGIKernel, options);
    }

    /** At least one item, and no more than one storage binding holds. */
    private checkCapacity(capacity: number, label: string): void {
        if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError(`@pmndrs/gi: ${label}: capacity must be a whole number of at least 1, got ${capacity}`);
        const available = this.device.limits.maxStorageBufferBindingSize;
        if (capacity * WORD_BYTES > available) throw new GILimitError('maxStorageBufferBindingSize', capacity * WORD_BYTES, available);
    }

    private scanKernelsFor(variant: ScanVariant): Promise<ScanKernels> {
        let kernels = this.scanKernels.get(variant);
        if (!kernels) {
            kernels = this.compileScan(variant);
            this.scanKernels.set(variant, kernels);
        }
        return kernels;
    }

    private async compileScan(variant: ScanVariant): Promise<ScanKernels> {
        const label = `${this.label} gi/${variant === 'compaction' ? 'compaction' : `scan ${variant}`} (${this.reduction})`;
        const layout = this.device.createBindGroupLayout({ label, entries: scanLayoutEntries(variant) });
        const pipelineLayout = this.device.createPipelineLayout({ label, bindGroupLayouts: [layout] });
        const pipelines = await createScanPipelines(this.device, label, { reduction: this.reduction, io: SCAN_VARIANT_IO[variant], layout: pipelineLayout });
        return { layout, pipelines };
    }

    private writerKernelFor(): Promise<WriterKernel> {
        this.writerKernel ??= this.compileWriter();
        return this.writerKernel;
    }

    private async compileWriter(): Promise<WriterKernel> {
        const label = `${this.label} gi/indirect-args`;
        const layout = this.device.createBindGroupLayout({
            label,
            entries: [
                bufferEntry(INDIRECT_WRITER_BINDING.countSource, 'read-only-storage'),
                bufferEntry(INDIRECT_WRITER_BINDING.args, 'storage'),
                bufferEntry(INDIRECT_WRITER_BINDING.params, 'uniform'),
            ],
        });
        const module = this.device.createShaderModule({ label, code: INDIRECT_WRITER_WGSL });
        const pipeline = await this.device.createComputePipelineAsync({
            label,
            layout: this.device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
            compute: { module, entryPoint: INDIRECT_WRITER_ENTRY_POINT },
        });
        return { layout, pipeline };
    }
}

/** Parameter slots in one page (each `minUniformBufferOffsetAlignment` bytes, 256 on every device we know). */
const PARAM_SLOTS_PER_PAGE = 16;
/** Pages one scan keeps at most; when a new one is needed the oldest is dropped. */
const MAX_PARAM_PAGES = 4;

/** A uniform buffer of parameter slots, and the scan's bind group over it. */
interface ParamPage {
    readonly buffer: GPUBuffer;
    readonly bindGroup: GPUBindGroup;
    used: number;
}

/** Where one count's `ScanParams` live. */
interface ParamSlot {
    readonly page: ParamPage;
    readonly offset: number;
}

/**
 * One scan or compaction: its scratch, and its parameters.
 *
 * **Every encode has its own count.** The count reaches the GPU in a uniform
 * block written with `queue.writeBuffer`, which lands before the next submit
 * runs. One block rewritten per encode would make two encodes in one submit
 * both see the last count. So each count gets a slot of its own, chosen per
 * encode with a dynamic offset, and a slot is written once and never changed:
 * whenever a submit runs, every slot it uses holds the count it was encoded
 * with. Slots are kept per count (a count seen again reuses its slot), in
 * pages of {@link PARAM_SLOTS_PER_PAGE}; past {@link MAX_PARAM_PAGES} pages the
 * oldest page is dropped from the table, but not destroyed, since a command
 * buffer not yet submitted may still use it: it is freed once nothing refers
 * to it.
 */
class ScanRun {
    readonly result: GpuWord;

    private readonly device: GPUDevice;
    private readonly kernels: ScanKernels;
    private readonly label: string;
    private readonly tileSums: GPUBuffer;
    private readonly resultBuffer: GPUBuffer;
    /** The bind group entries except the parameters, which each page adds. */
    private readonly entries: readonly GPUBindGroupEntry[];
    readonly capacity: number;
    private readonly maxPerDimension: number;
    private readonly slotBytes: number;
    private readonly pages: ParamPage[] = [];
    private readonly slots = new Map<number, ParamSlot>();

    constructor(
        device: GPUDevice,
        kernels: ScanKernels,
        options: { readonly label: string; readonly capacity: number; readonly source: GPUBuffer; readonly destination: GPUBuffer | undefined },
    ) {
        this.device = device;
        this.kernels = kernels;
        this.label = options.label;
        this.capacity = options.capacity;
        this.maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
        this.slotBytes = Math.max(SCAN_PARAMS_BYTES, device.limits.minUniformBufferOffsetAlignment);
        const { label } = options;
        this.tileSums = device.createBuffer({ label: `${label} scan tile sums`, size: scanTileSumWords(options.capacity) * WORD_BYTES, usage: BUFFER_USAGE.storage });
        this.resultBuffer = device.createBuffer({ label: `${label} scan total`, size: RESULT_BYTES, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        this.result = { buffer: this.resultBuffer, offset: 0 };
        const itemBytes = options.capacity * WORD_BYTES;
        const entries: GPUBindGroupEntry[] = [
            { binding: SCAN_BINDING.tileSums, resource: { buffer: this.tileSums } },
            { binding: COMPUTE_IO_BINDING.source, resource: { buffer: options.source, size: itemBytes } },
            { binding: COMPUTE_IO_BINDING.result, resource: { buffer: this.resultBuffer } },
        ];
        if (options.destination) entries.push({ binding: COMPUTE_IO_BINDING.destination, resource: { buffer: options.destination, size: itemBytes } });
        this.entries = entries;
        // The first page now, so memoryBytes means something before the first encode.
        this.addPage();
    }

    /** Scratch, the total and the parameter pages held now (at most {@link MAX_PARAM_PAGES}). */
    get memoryBytes(): number {
        return this.tileSums.size + this.resultBuffer.size + this.pages.reduce((bytes, page) => bytes + page.buffer.size, 0);
    }

    encode(pass: GPUComputePassEncoder, count: number): void {
        if (!Number.isInteger(count) || count < 0 || count > this.capacity) throw new RangeError(`@pmndrs/gi: a scan of capacity ${this.capacity} was asked for ${count} items`);
        const slot = this.slotFor(count);
        const { pipelines } = this.kernels;
        if (count === 0) {
            // Nothing to scan, but the total still has to say 0 for the passes that read it:
            // the across-tiles kernel with no tiles writes exactly that.
            pass.setPipeline(pipelines.tileSums);
            pass.setBindGroup(0, slot.page.bindGroup, [slot.offset]);
            pass.dispatchWorkgroups(1);
            return;
        }
        // One layout for all three kernels, so the same group serves each.
        const group = slot.page.bindGroup;
        encodeScan(pass, pipelines, { reduce: group, tileSums: group, downsweep: group, count }, this.maxPerDimension, [slot.offset]);
    }

    dispose(): void {
        this.tileSums.destroy();
        this.resultBuffer.destroy();
        for (const page of this.pages) page.buffer.destroy();
        this.pages.length = 0;
        this.slots.clear();
    }

    /** The slot holding `count`'s parameters, written now if the count is new. */
    private slotFor(count: number): ParamSlot {
        const known = this.slots.get(count);
        if (known) return known;
        let page = this.pages[this.pages.length - 1];
        if (page.used === PARAM_SLOTS_PER_PAGE) page = this.addPage();
        const slot: ParamSlot = { page, offset: page.used * this.slotBytes };
        page.used += 1;
        this.device.queue.writeBuffer(page.buffer, slot.offset, scanParams(count));
        this.slots.set(count, slot);
        return slot;
    }

    private addPage(): ParamPage {
        if (this.pages.length === MAX_PARAM_PAGES) {
            // Not destroyed: an encoded command buffer may still use it (see the class comment).
            const oldest = this.pages.shift();
            for (const [count, slot] of this.slots) if (slot.page === oldest) this.slots.delete(count);
        }
        const buffer = this.device.createBuffer({
            label: `${this.label} scan params`,
            size: PARAM_SLOTS_PER_PAGE * this.slotBytes,
            usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst,
        });
        const bindGroup = this.device.createBindGroup({
            label: `${this.label} scan`,
            layout: this.kernels.layout,
            entries: [...this.entries, { binding: SCAN_BINDING.params, resource: { buffer, size: SCAN_PARAMS_BYTES } }],
        });
        const page: ParamPage = { buffer, bindGroup, used: 0 };
        this.pages.push(page);
        return page;
    }
}

class IndirectArgsWriter implements GpuIndirectArgs {
    readonly buffer: GPUBuffer;
    readonly memoryBytes: number;

    private readonly params: GPUBuffer;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroup: GPUBindGroup;

    constructor(device: GPUDevice, kernel: WriterKernel, options: IndirectArgsOptions) {
        const { label, count } = options;
        this.pipeline = kernel.pipeline;
        this.buffer = device.createBuffer({
            label: `${label} indirect args`,
            size: INDIRECT_ARGS_BYTES,
            usage: BUFFER_USAGE.indirect | BUFFER_USAGE.storage | BUFFER_USAGE.copySrc,
        });
        // Fixed for the writer's life, so written once at creation (no queue write to wait for).
        this.params = device.createBuffer({ label: `${label} indirect args params`, size: INDIRECT_WRITER_PARAMS_BYTES, usage: BUFFER_USAGE.uniform, mappedAtCreation: true });
        new Uint32Array(this.params.getMappedRange()).set([count.offset / WORD_BYTES, options.itemsPerWorkgroup, device.limits.maxComputeWorkgroupsPerDimension, 0]);
        this.params.unmap();
        this.memoryBytes = this.buffer.size + this.params.size;
        this.bindGroup = device.createBindGroup({
            label: `${label} indirect args`,
            layout: kernel.layout,
            entries: [
                // Storage bindings start at 256-byte aligned offsets, so the binding starts at 0 and the word is found by index.
                { binding: INDIRECT_WRITER_BINDING.countSource, resource: { buffer: count.buffer, offset: 0, size: count.offset + WORD_BYTES } },
                { binding: INDIRECT_WRITER_BINDING.args, resource: { buffer: this.buffer } },
                { binding: INDIRECT_WRITER_BINDING.params, resource: { buffer: this.params } },
            ],
        });
    }

    encode(pass: GPUComputePassEncoder): void {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(1);
    }

    dispose(): void {
        this.buffer.destroy();
        this.params.destroy();
    }
}
