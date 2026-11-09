/**
 * WGSL of the platform's GPU utilities (platform/contract/compute.ts).
 *
 * The scan and the compaction are `@pmndrs/bvh`'s scan kernels
 * (packages/bvh/src/gpu-scan.ts) with an io of their own: the io says what
 * the kernels read and what they do with each prefix. Their bindings in group
 * 0 are the scan's tile sums and parameters, then the io's:
 *
 * | binding | scan into `output` | scan in place | compaction        |
 * | ------- | ------------------ | ------------- | ----------------- |
 * | 2       | input (read)       | data          | flags (read)      |
 * | 3       | output             | (none)        | kept indices      |
 * | 4       | result             | result        | result            |
 *
 * `result` word 0 is the total (the kept count for a compaction), at a fixed
 * place whatever the count, so later passes can read it as a `GpuWord`.
 *
 * The indirect-arguments writer is a one-thread kernel of its own.
 */

import { SCAN_IO_FIRST_BINDING, type ScanIo } from '@pmndrs/bvh/gpu-scan';
import type { WgslModule } from '../contract/wgsl.ts';

/** The io bindings (see the table above). */
export const COMPUTE_IO_BINDING = {
    /** The scan's input, the in-place scan's data, or the compaction's flags. */
    source: SCAN_IO_FIRST_BINDING,
    /** The scan's output or the compaction's kept indices. */
    destination: SCAN_IO_FIRST_BINDING + 1,
    /** One word: the total, or the kept count. */
    result: SCAN_IO_FIRST_BINDING + 2,
} as const;

/** Bytes of the result buffer: one word, the total. */
export const RESULT_BYTES = 4;

const RESULT_WGSL = /* wgsl */ `
@group(0) @binding(${COMPUTE_IO_BINDING.result}) var<storage, read_write> giComputeResult: array<u32, 1>;

fn scanStoreTotal(total: u32) {
    giComputeResult[0] = total;
}
`;

/** Scan `input` into `output`; the total into the result word. */
export const SCAN_TO_OUTPUT_IO: ScanIo = {
    name: 'gi/scan into output',
    wgsl: /* wgsl */ `
@group(0) @binding(${COMPUTE_IO_BINDING.source}) var<storage, read> giScanInput: array<u32>;
@group(0) @binding(${COMPUTE_IO_BINDING.destination}) var<storage, read_write> giScanOutput: array<u32>;
${RESULT_WGSL}
fn scanLoad(at: u32) -> u32 {
    return giScanInput[at];
}

fn scanStore(at: u32, value: u32, prefix: u32) {
    giScanOutput[at] = prefix;
}
`,
};

/**
 * Scan `data` in place; the total into the result word. A separate io, not
 * the one above with the same buffer twice: one buffer can't be bound both
 * read-only and writable in one dispatch.
 */
export const SCAN_IN_PLACE_IO: ScanIo = {
    name: 'gi/scan in place',
    wgsl: /* wgsl */ `
@group(0) @binding(${COMPUTE_IO_BINDING.source}) var<storage, read_write> giScanData: array<u32>;
${RESULT_WGSL}
fn scanLoad(at: u32) -> u32 {
    return giScanData[at];
}

// Every load of this thread's items came before its stores, and no thread
// stores another's items, so scanning in place is safe.
fn scanStore(at: u32, value: u32, prefix: u32) {
    giScanData[at] = prefix;
}
`,
};

/**
 * Stream compaction: a kept item counts 1, so its exclusive prefix is its
 * place in the packed list, and items keep their order (stable). The total
 * is the number kept.
 */
export const COMPACTION_IO: ScanIo = {
    name: 'gi/compaction',
    wgsl: /* wgsl */ `
@group(0) @binding(${COMPUTE_IO_BINDING.source}) var<storage, read> giCompactFlags: array<u32>;
@group(0) @binding(${COMPUTE_IO_BINDING.destination}) var<storage, read_write> giCompactIndices: array<u32>;
${RESULT_WGSL}
// Any non-zero flag keeps the item.
fn scanLoad(at: u32) -> u32 {
    return select(0u, 1u, giCompactFlags[at] != 0u);
}

fn scanStore(at: u32, value: u32, prefix: u32) {
    if (value != 0u) {
        giCompactIndices[prefix] = at;
    }
}
`,
};

/** Bytes of a `GiIndirectArgs`: three workgroup counts and the count. */
export const INDIRECT_ARGS_BYTES = 16;

/** Word of `GiIndirectArgs.count`. */
export const INDIRECT_ARGS_COUNT_WORD = 3;

/** Bytes of the writer's uniform block (`GiIndirectWriterParams`). */
export const INDIRECT_WRITER_PARAMS_BYTES = 16;

const INDIRECT_ARGS_WGSL = /* wgsl */ `
// ---- Indirect dispatch arguments (platform/compute) ----

// What the arguments writer leaves: the workgroups dispatchWorkgroupsIndirect
// reads, then the count they were sized for. The last row of workgroups runs
// past the count, so a consuming kernel binds this read-only and checks
// each item's index against 'count'.
struct GiIndirectArgs {
    workgroups: vec3u,
    count: u32,
}

// The item this invocation handles in a dispatch the writer sized:
// workgroups fill x first, then rows along y.
fn giIndirectItemIndex(workgroup: vec3u, groups: vec3u, localIndex: u32, workgroupSize: u32) -> u32 {
    return (workgroup.y * groups.x + workgroup.x) * workgroupSize + localIndex;
}
`;

/** `GiIndirectArgs` and `giIndirectItemIndex` for consuming kernels: code only, nothing bound. */
export const INDIRECT_ARGS_MODULE: WgslModule = {
    name: 'gi-indirect-args',
    dependencies: [],
    code: () => INDIRECT_ARGS_WGSL,
    entries: [],
    resources: () => [],
    version: 0,
    storageBufferCount: 0,
};

/** Entry point of the writer. */
export const INDIRECT_WRITER_ENTRY_POINT = 'giWriteIndirectArgs';

/** The writer's bindings in group 0. */
export const INDIRECT_WRITER_BINDING = { countSource: 0, args: 1, params: 2 } as const;

/** The writer: one thread turns the count into `[x, y, 1]` workgroups (the CPU twin is `indirectArgsOnCpu`). */
export const INDIRECT_WRITER_WGSL = /* wgsl */ `${INDIRECT_ARGS_WGSL}
struct GiIndirectWriterParams {
    countWord: u32,                  // where the count is in giIndirectCountSource
    itemsPerWorkgroup: u32,
    maxWorkgroupsPerDimension: u32,  // the device's maxComputeWorkgroupsPerDimension
    unused: u32,
}

@group(0) @binding(${INDIRECT_WRITER_BINDING.countSource}) var<storage, read> giIndirectCountSource: array<u32>;
@group(0) @binding(${INDIRECT_WRITER_BINDING.args}) var<storage, read_write> giIndirectArgs: GiIndirectArgs;
@group(0) @binding(${INDIRECT_WRITER_BINDING.params}) var<uniform> giIndirectWriter: GiIndirectWriterParams;

// Rounds up without adding divisor - 1 first, which could wrap past 2^32.
fn giDivideRoundingUp(value: u32, divisor: u32) -> u32 {
    return value / divisor + select(0u, 1u, value % divisor != 0u);
}

@compute @workgroup_size(1)
fn ${INDIRECT_WRITER_ENTRY_POINT}() {
    let count = giIndirectCountSource[giIndirectWriter.countWord];
    let workgroups = giDivideRoundingUp(count, giIndirectWriter.itemsPerWorkgroup);
    // As many as fit along x, the rest in rows along y. y stays under the
    // limit too: items fit one storage binding (under 2^32 bytes, so under
    // 2^30 words), and 2^30 / 65535 (the smallest limit) is about 16,400 rows.
    let x = min(workgroups, giIndirectWriter.maxWorkgroupsPerDimension);
    var y = 0u;
    if (x > 0u) {
        y = giDivideRoundingUp(workgroups, x);
    }
    giIndirectArgs.workgroups = vec3u(x, y, 1u);
    giIndirectArgs.count = count;
}
`;
