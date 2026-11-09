/**
 * GPU utilities a technique builds its passes from (RC1.3): an exclusive
 * prefix scan, stream compaction, and the arguments for an indirect dispatch
 * sized by a count the GPU computed. The radiance cache needs them for work
 * the CPU never sees the size of (spawned screen probes, cells to update).
 *
 * Every utility records into the caller's compute pass, owns no device (the
 * host owns it), allocates its scratch once, for the capacity it was created
 * with, and reports those bytes in `memoryBytes` (a technique adds them to its
 * own). Nothing is read back to the CPU: counts and totals stay on the GPU as
 * {@link GpuWord}s that later passes read.
 *
 * Inside a workgroup they use subgroup operations where the device has
 * `subgroups`, and the declared fallback in workgroup memory where it hasn't
 * (DECISIONS 2026-11-04). Both give the same results; the scan is the one
 * `@pmndrs/bvh` builds its trees with (packages/bvh/src/gpu-scan.ts).
 *
 * **Counts reach the GPU with the next submit.** `encode(pass, count)` writes
 * the count with `queue.writeBuffer`, which lands before the next submitted
 * command buffer runs. So one utility object handles one count per submit;
 * to scan two different counts in one submit, make two objects.
 */

import type { WgslModule } from './wgsl';

/** How a workgroup combines its threads' values: subgroup operations, or the fallback in workgroup memory. */
export type ComputeReduction = 'subgroups' | 'workgroup';

/** One `u32` a pass wrote on the GPU (a total, a count) for later passes to read. */
export interface GpuWord {
    /** Has `STORAGE` usage (and `COPY_SRC`, for tests that read it back). */
    readonly buffer: GPUBuffer;
    /** Byte offset of the word in `buffer`: a multiple of 4. */
    readonly offset: number;
}

export interface PrefixScanOptions {
    /** Shown in GPU errors and captures. */
    readonly label: string;
    /** The `u32` values to scan (`STORAGE` usage). Overwritten with the result when there is no `output`. */
    readonly input: GPUBuffer;
    /** Where the exclusive prefix sums go (`STORAGE` usage). Omit it, or pass `input`, to scan in place. */
    readonly output?: GPUBuffer;
    /** The most items one encode may scan. Sizes the scratch; at most the device's largest storage binding in `u32`s. */
    readonly capacity: number;
}

/** An exclusive prefix sum of `u32`s: item i becomes the sum of items 0..i-1 (wrapping at 2^32). */
export interface GpuPrefixScan {
    readonly capacity: number;
    /** The sum of all the items of the latest encode, written by its pass (0 for an empty scan). */
    readonly total: GpuWord;
    /** GPU bytes this scan allocated (scratch and its uniform block, not the caller's buffers). */
    readonly memoryBytes: number;
    /** Records the scan of the first `count` items (at most `capacity`) into `pass`. */
    encode(pass: GPUComputePassEncoder, count: number): void;
    dispose(): void;
}

export interface CompactionOptions {
    readonly label: string;
    /** One `u32` per item (`STORAGE` usage): non-zero keeps the item. Left unchanged. */
    readonly flags: GPUBuffer;
    /** Where the kept items' indices go (`STORAGE` usage, at least `capacity` words). Omitted, the compaction allocates its own. */
    readonly output?: GPUBuffer;
    /** The most items one encode may look at. */
    readonly capacity: number;
}

/**
 * Stream compaction: the indices of the kept items, in increasing order
 * (stable), packed at the start of `indices`, and how many there are.
 */
export interface GpuCompaction {
    readonly capacity: number;
    /** The kept indices: the first `count` words are valid after the pass. */
    readonly indices: GPUBuffer;
    /** How many items the latest encode kept. */
    readonly count: GpuWord;
    /** GPU bytes this compaction allocated (scratch, and `indices` when it made them). */
    readonly memoryBytes: number;
    /** Records the compaction of the first `count` flags (at most `capacity`) into `pass`. */
    encode(pass: GPUComputePassEncoder, count: number): void;
    dispose(): void;
}

export interface IndirectArgsOptions {
    readonly label: string;
    /** The count to size the dispatch for, written by an earlier pass (a compaction's `count`, a scan's `total`). */
    readonly count: GpuWord;
    /** Items one workgroup of the consuming kernel covers: its workgroup size, times items per thread if it loops. */
    readonly itemsPerWorkgroup: number;
}

/**
 * Indirect dispatch arguments from a GPU count: `ceil(count / itemsPerWorkgroup)`
 * workgroups, laid out as `[x, y, 1]` with x at most the device's
 * `maxComputeWorkgroupsPerDimension` and the rest in rows along y. The last
 * row can run past the count, so the consuming kernel checks each item's
 * index against the count, which the buffer also holds.
 *
 * The buffer is a `GiIndirectArgs` ({@link GpuComputeUtilities.indirectModule}):
 * the workgroups, then the count. Pass it to
 * `pass.dispatchWorkgroupsIndirect(args.buffer, 0)` and, in the same dispatch,
 * bind it read-only to read `count` (both uses are read-only, which WebGPU
 * allows together).
 */
export interface GpuIndirectArgs {
    /** 16 bytes, `INDIRECT | STORAGE | COPY_SRC` usage. */
    readonly buffer: GPUBuffer;
    readonly memoryBytes: number;
    /** Records the one-thread dispatch that writes the arguments into `pass`. Place it after the pass that writes the count. */
    encode(pass: GPUComputePassEncoder): void;
    dispose(): void;
}

/** The platform's GPU utilities, on `TechniqueContext.compute`. */
export interface GpuComputeUtilities {
    /** The path the utilities take on this device: `subgroups` where the device has them. */
    readonly reduction: ComputeReduction;
    /**
     * Code only, nothing bound, for kernels that consume indirect arguments:
     *
     * ```wgsl
     * struct GiIndirectArgs {
     *     workgroups: vec3u,   // what dispatchWorkgroupsIndirect reads
     *     count: u32,          // the count the workgroups were sized for
     * }
     * // The item this invocation handles, in a dispatch sized by the writer
     * // (workgroups fill x first, then rows along y).
     * fn giIndirectItemIndex(workgroup: vec3u, groups: vec3u, localIndex: u32, workgroupSize: u32) -> u32
     * ```
     */
    readonly indirectModule: WgslModule;
    /** A scan's kernels compile on the first scan made (asynchronously); later ones reuse them. */
    createPrefixScan(options: PrefixScanOptions): Promise<GpuPrefixScan>;
    createCompaction(options: CompactionOptions): Promise<GpuCompaction>;
    createIndirectArgs(options: IndirectArgsOptions): Promise<GpuIndirectArgs>;
}
