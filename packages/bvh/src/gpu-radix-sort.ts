/**
 * Stable key-value radix sort of `u32` pairs on the GPU, least significant
 * digit first, reduce-then-scan (docs/design/bvh-framework.md, "Sort and
 * scan"): no decoupled look-back, so no workgroup waits on another and nothing
 * depends on forward progress between workgroups. The LBVH builder sorts its
 * Morton codes with it.
 *
 * **One pass per {@link RADIX_DIGIT_BITS}-bit digit**, each three steps:
 *
 * 1. `sortCount`, one workgroup per tile of {@link SORT_TILE_KEYS} keys: how
 *    many keys of the tile have each digit, stored digit-major
 *    (`digit * tileCount + tile`).
 * 2. An exclusive scan of those counts (gpu-scan.ts). Digit-major order makes
 *    the scanned value exactly where the tile's first key with that digit goes.
 * 3. `sortScatter`, one workgroup per tile: each thread takes
 *    {@link KEYS_PER_THREAD} neighbouring keys, counts its digits, and a
 *    workgroup scan over the threads (in subgroup order, gpu-workgroup-scan.ts)
 *    gives each key its rank among the tile's keys with the same digit. Keys
 *    keep their order within a digit, so the sort is stable, which the LBVH
 *    relies on: equal Morton codes stay in triangle order.
 *
 * A tile's counts fit in 16 bits (a tile has 2048 keys), so a thread keeps its
 * {@link RADIX_BINS} counts packed two to a word and the workgroup scans the
 * 16 words at once.
 *
 * A key set that fits one tile skips steps 1 and 2: the scatter takes the
 * digit offsets from its own counts (`singleTile`). Most meshes of a real scene
 * are that small, and it saves two thirds of the dispatches.
 */

import { createScanBindGroups, encodeScan, SCAN_PARAMS_BYTES, scanParams, type ScanBindGroups, type ScanPipelines } from './gpu-scan.ts';
import { dispatchWorkgroups2d, SCAN_WORKGROUP_SIZE, WORKGROUP_INDEX_WGSL, workgroupScanWgsl } from './gpu-workgroup-scan.ts';

/** Bits sorted per pass. 5 bits sort a 30-bit Morton code in 6 passes. */
export const RADIX_DIGIT_BITS = 5;
export const RADIX_BINS = 1 << RADIX_DIGIT_BITS;

/** Counts packed per word (16 bits each). */
const BIN_WORDS = RADIX_BINS / 2;

/** Keys one thread ranks, one after another. */
const KEYS_PER_THREAD = 8;

/** Keys one workgroup handles. Must stay below 65536, the largest packed count. */
export const SORT_TILE_KEYS = SCAN_WORKGROUP_SIZE * KEYS_PER_THREAD;

/** Bytes of one pass's uniform block (`SortParams`). */
export const SORT_PARAMS_BYTES = 16;

/** Passes needed to sort keys of `keyBits` bits. */
export function radixPassCount(keyBits: number): number {
    return Math.ceil(keyBits / RADIX_DIGIT_BITS);
}

/** Words of digit counts (later offsets) a sort of `count` keys needs. */
export function sortOffsetWords(count: number): number {
    return RADIX_BINS * Math.ceil(count / SORT_TILE_KEYS);
}

/** `SortParams` of pass `pass` over `count` keys. */
export function sortParams(count: number, pass: number): Uint32Array<ArrayBuffer> {
    const tileCount = Math.ceil(count / SORT_TILE_KEYS);
    return new Uint32Array([count, tileCount, pass * RADIX_DIGIT_BITS, tileCount <= 1 ? 1 : 0]);
}

/** WGSL of the count and scatter kernels. Needs `subgroups`. */
export function radixSortWgsl(): string {
    return /* wgsl */ `enable subgroups;

struct SortParams {
    count: u32,
    tileCount: u32,
    shift: u32,        // the digit's lowest bit
    singleTile: u32,   // 1: one tile, so no counts were scanned and the scatter makes its own offsets
}

@group(0) @binding(0) var<storage, read> sortKeysIn: array<u32>;
@group(0) @binding(1) var<storage, read> sortValuesIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> sortKeysOut: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortValuesOut: array<u32>;
// Per digit and tile, digit-major: the count, then (after the scan) where those keys go.
@group(0) @binding(4) var<storage, read_write> sortOffsets: array<u32>;
@group(0) @binding(5) var<uniform> sortParams: SortParams;

const SORT_BINS: u32 = ${RADIX_BINS}u;
const SORT_DIGIT_MASK: u32 = ${RADIX_BINS - 1}u;
const SORT_KEYS_PER_THREAD: u32 = ${KEYS_PER_THREAD}u;
const SORT_TILE: u32 = ${SORT_TILE_KEYS}u;
${WORKGROUP_INDEX_WGSL}
${workgroupScanWgsl(BIN_WORDS)}

fn sortDigit(key: u32) -> u32 {
    return (key >> sortParams.shift) & SORT_DIGIT_MASK;
}

// Adds one to digit 'digit' of packed counts: two 16-bit counts per word.
fn sortCountOne(counts: ptr<function, array<u32, SCAN_WORDS>>, digit: u32) {
    (*counts)[digit >> 1u] += 1u << ((digit & 1u) * 16u);
}

fn sortUnpack(counts: array<u32, SCAN_WORDS>, digit: u32) -> u32 {
    return (counts[digit >> 1u] >> ((digit & 1u) * 16u)) & 0xffffu;
}

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
fn sortCount(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let tile = flatWorkgroupIndex(workgroup, groups);
    if (tile >= sortParams.tileCount) {
        return;   // the whole workgroup: tile is uniform
    }
    let order = scanTakeOrder();
    let first = tile * SORT_TILE + order.thread * SORT_KEYS_PER_THREAD;
    var counts: array<u32, SCAN_WORDS>;
    for (var key = 0u; key < SORT_KEYS_PER_THREAD; key += 1u) {
        if (first + key < sortParams.count) {
            sortCountOne(&counts, sortDigit(sortKeysIn[first + key]));
        }
    }
    _ = scanExclusive(counts, order, localIndex);
    if (localIndex < SORT_BINS) {
        sortOffsets[localIndex * sortParams.tileCount + tile] = sortUnpack(scanTotals, localIndex);
    }
}

// Where this tile's first key of each digit goes.
var<workgroup> sortTileOffsets: array<u32, ${RADIX_BINS}>;

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
fn sortScatter(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let tile = flatWorkgroupIndex(workgroup, groups);
    if (tile >= sortParams.tileCount) {
        return;
    }
    let order = scanTakeOrder();
    let single = sortParams.singleTile == 1u;
    if (!single && localIndex < SORT_BINS) {
        sortTileOffsets[localIndex] = sortOffsets[localIndex * sortParams.tileCount + tile];
    }

    let first = tile * SORT_TILE + order.thread * SORT_KEYS_PER_THREAD;
    var keys: array<u32, ${KEYS_PER_THREAD}>;
    var values: array<u32, ${KEYS_PER_THREAD}>;
    var counts: array<u32, SCAN_WORDS>;
    for (var key = 0u; key < SORT_KEYS_PER_THREAD; key += 1u) {
        if (first + key < sortParams.count) {
            keys[key] = sortKeysIn[first + key];
            values[key] = sortValuesIn[first + key];
            sortCountOne(&counts, sortDigit(keys[key]));
        }
    }
    // The scan's barriers also make sortTileOffsets visible.
    var rank = scanExclusive(counts, order, localIndex);

    if (single) {
        // One tile: the offsets are the exclusive prefix of the tile's own digit totals.
        if (localIndex == 0u) {
            var running = 0u;
            for (var digit = 0u; digit < SORT_BINS; digit += 1u) {
                sortTileOffsets[digit] = running;
                running += sortUnpack(scanTotals, digit);
            }
        }
        workgroupBarrier();
    }

    for (var key = 0u; key < SORT_KEYS_PER_THREAD; key += 1u) {
        if (first + key < sortParams.count) {
            let digit = sortDigit(keys[key]);
            let destination = sortTileOffsets[digit] + sortUnpack(rank, digit);
            sortCountOne(&rank, digit);
            sortKeysOut[destination] = keys[key];
            sortValuesOut[destination] = values[key];
        }
    }
}
`;
}

export interface RadixSortPipelines {
    readonly count: GPUComputePipeline;
    readonly scatter: GPUComputePipeline;
    readonly scan: ScanPipelines;
}

export async function createRadixSortPipelines(device: GPUDevice, scan: ScanPipelines, label: string): Promise<RadixSortPipelines> {
    const module = device.createShaderModule({ label: `${label} radix sort`, code: radixSortWgsl() });
    const pipeline = (entryPoint: string) =>
        device.createComputePipelineAsync({ label: `${label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } });
    const [count, scatter] = await Promise.all([pipeline('sortCount'), pipeline('sortScatter')]);
    return { count, scatter, scan };
}

/** The buffers one sort uses. Keys and values ping-pong between `a` and `b`; with an even pass count the result ends in `a`. */
export interface RadixSortBuffers {
    readonly keysA: GPUBuffer;
    readonly valuesA: GPUBuffer;
    readonly keysB: GPUBuffer;
    readonly valuesB: GPUBuffer;
    /** At least {@link sortOffsetWords} words. */
    readonly offsets: GPUBuffer;
    /** At least `scanTileSumWords(sortOffsetWords(count))` words. */
    readonly tileSums: GPUBuffer;
    /** Uniforms: pass `p`'s `SortParams` at `paramsOffset + p * paramsStride`, then the offsets scan's `ScanParams`. */
    readonly params: GPUBuffer;
    readonly paramsOffset: number;
    readonly paramsStride: number;
}

/** A sort of a fixed number of keys over fixed buffers, with its bind groups made once. */
export interface RadixSort {
    readonly count: number;
    readonly passes: number;
    /** Writes the uniforms; call before each submit that encodes the sort (the values never change, so once is enough). */
    writeParams(queue: GPUQueue): void;
    encode(pass: GPUComputePassEncoder): void;
}

export function createRadixSort(
    device: GPUDevice,
    pipelines: RadixSortPipelines,
    buffers: RadixSortBuffers,
    count: number,
    keyBits: number,
    label: string,
): RadixSort {
    const passes = radixPassCount(keyBits);
    if (passes % 2 !== 0) throw new RangeError(`the sort needs an even number of passes to end in buffer a; ${keyBits}-bit keys take ${passes}`);
    const tileCount = Math.ceil(count / SORT_TILE_KEYS);
    const singleTile = tileCount <= 1;
    const offsetWords = sortOffsetWords(count);
    const scanParamsOffset = buffers.paramsOffset + passes * buffers.paramsStride;
    const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;

    const passGroups = Array.from({ length: passes }, (_, pass) => {
        const forward = pass % 2 === 0;
        const entries = (pipeline: GPUComputePipeline, withValues: boolean): GPUBindGroup => {
            const list: GPUBindGroupEntry[] = [
                { binding: 0, resource: { buffer: forward ? buffers.keysA : buffers.keysB } },
                { binding: 4, resource: { buffer: buffers.offsets, size: Math.max(4, offsetWords * 4) } },
                { binding: 5, resource: { buffer: buffers.params, offset: buffers.paramsOffset + pass * buffers.paramsStride, size: SORT_PARAMS_BYTES } },
            ];
            if (withValues) {
                list.push(
                    { binding: 1, resource: { buffer: forward ? buffers.valuesA : buffers.valuesB } },
                    { binding: 2, resource: { buffer: forward ? buffers.keysB : buffers.keysA } },
                    { binding: 3, resource: { buffer: forward ? buffers.valuesB : buffers.valuesA } },
                );
            }
            return device.createBindGroup({ label: `${label} pass ${pass}`, layout: pipeline.getBindGroupLayout(0), entries: list });
        };
        return { count: entries(pipelines.count, false), scatter: entries(pipelines.scatter, true) };
    });
    const scanGroups: ScanBindGroups | null = singleTile
        ? null
        : createScanBindGroups(
              device,
              pipelines.scan,
              {
                  data: { buffer: buffers.offsets, size: offsetWords * 4 },
                  tileSums: { buffer: buffers.tileSums },
                  params: { buffer: buffers.params, offset: scanParamsOffset, size: SCAN_PARAMS_BYTES },
                  count: offsetWords,
              },
              `${label} offsets scan`,
          );

    return {
        count,
        passes,
        writeParams(queue) {
            for (let pass = 0; pass < passes; pass++) {
                queue.writeBuffer(buffers.params, buffers.paramsOffset + pass * buffers.paramsStride, sortParams(count, pass));
            }
            queue.writeBuffer(buffers.params, scanParamsOffset, scanParams(offsetWords));
        },
        encode(pass) {
            if (count === 0) return;
            for (const groups of passGroups) {
                if (scanGroups) {
                    pass.setPipeline(pipelines.count);
                    pass.setBindGroup(0, groups.count);
                    dispatchWorkgroups2d(pass, tileCount, maxPerDimension);
                    encodeScan(pass, pipelines.scan, scanGroups, maxPerDimension);
                }
                pass.setPipeline(pipelines.scatter);
                pass.setBindGroup(0, groups.scatter);
                dispatchWorkgroups2d(pass, tileCount, maxPerDimension);
            }
        },
    };
}
