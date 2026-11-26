/**
 * Stable, segmented key-value radix sort of `u32` pairs on the GPU, least
 * significant digit first, reduce-then-scan (docs/design/bvh-framework.md,
 * "Sort and scan"): no decoupled look-back, so no workgroup waits on another
 * and nothing depends on forward progress between workgroups. The LBVH
 * builder sorts every mesh's Morton codes with it at once: each mesh is a
 * segment, sorted within its own range, and all segments share every dispatch.
 *
 * **One pass per {@link RADIX_DIGIT_BITS}-bit digit**, each three steps:
 *
 * 1. `sortCount`, one workgroup per tile of up to {@link SORT_TILE_KEYS} keys
 *    (a tile never crosses a segment): how many keys of the tile have each
 *    digit. Counts are stored segment by segment, digit-major within a segment
 *    (`segment block, then digit * tiles + tile`).
 * 2. An exclusive scan of all the counts (gpu-scan.ts). That order makes each
 *    scanned value exactly where the tile's first key with that digit goes:
 *    everything in earlier segments, then the segment's smaller digits, then
 *    the same digit in the segment's earlier tiles.
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

/** Words per tile in the tile table: first key, end key, offsets base, offsets stride. */
export const SORT_TILE_WORDS = 4;

/** Bytes of one pass's uniform block (`SortParams`). */
export const SORT_PARAMS_BYTES = 16;

/** Passes needed to sort keys of `keyBits` bits. */
export function radixPassCount(keyBits: number): number {
    return Math.ceil(keyBits / RADIX_DIGIT_BITS);
}

/** Tiles a segment of `count` keys takes. */
export function sortTilesOf(count: number): number {
    return Math.ceil(count / SORT_TILE_KEYS);
}

/**
 * The tile table for segments of the given sizes, laid out one after another
 * from key 0: per tile its first key, its end (exclusive), and where its digit
 * counts go (`base + digit * stride`). Also the number of count words.
 */
export function sortTileTable(segmentCounts: readonly number[]): { tiles: Uint32Array<ArrayBuffer>; tileCount: number; offsetWords: number } {
    const tileCount = segmentCounts.reduce((total, count) => total + sortTilesOf(count), 0);
    const tiles = new Uint32Array(Math.max(1, tileCount) * SORT_TILE_WORDS);
    let key = 0;
    let tile = 0;
    let offsets = 0;
    for (const count of segmentCounts) {
        const segmentTiles = sortTilesOf(count);
        for (let inSegment = 0; inSegment < segmentTiles; inSegment++, tile++) {
            const first = key + inSegment * SORT_TILE_KEYS;
            tiles.set([first, Math.min(first + SORT_TILE_KEYS, key + count), offsets + inSegment, segmentTiles], tile * SORT_TILE_WORDS);
        }
        key += count;
        offsets += segmentTiles * RADIX_BINS;
    }
    return { tiles, tileCount, offsetWords: offsets };
}

/** `SortParams` of pass `pass`. The tile table starts at word `tilesOffset` of the tables buffer. */
export function sortParams(pass: number, tileCount: number, tilesOffset: number): Uint32Array<ArrayBuffer> {
    return new Uint32Array([pass * RADIX_DIGIT_BITS, tileCount, tilesOffset, 0]);
}

/** WGSL of the count and scatter kernels. Needs `subgroups`. */
export function radixSortWgsl(): string {
    return /* wgsl */ `enable subgroups;

struct SortParams {
    shift: u32,         // the digit's lowest bit
    tileCount: u32,
    tilesOffset: u32,   // where the tile table starts in sortTables, in words
    unused: u32,
}

@group(0) @binding(0) var<storage, read> sortKeysIn: array<u32>;
@group(0) @binding(1) var<storage, read> sortValuesIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> sortKeysOut: array<u32>;
@group(0) @binding(3) var<storage, read_write> sortValuesOut: array<u32>;
// Per segment, digit and tile: the count, then (after the scan) where those keys go.
@group(0) @binding(4) var<storage, read_write> sortOffsets: array<u32>;
// Per tile: first key, end key (exclusive), offsets base, offsets stride (the segment's tile count).
@group(0) @binding(5) var<storage, read> sortTables: array<u32>;
@group(0) @binding(6) var<uniform> sortParams: SortParams;

const SORT_BINS: u32 = ${RADIX_BINS}u;
const SORT_DIGIT_MASK: u32 = ${RADIX_BINS - 1}u;
const SORT_KEYS_PER_THREAD: u32 = ${KEYS_PER_THREAD}u;
const SORT_TILE_WORDS: u32 = ${SORT_TILE_WORDS}u;
const SORT_WORKGROUP_SIZE: u32 = ${SCAN_WORKGROUP_SIZE}u;
${WORKGROUP_INDEX_WGSL}
${workgroupScanWgsl(BIN_WORDS)}

struct SortTile {
    first: u32,
    end: u32,
    offsetsBase: u32,
    offsetsStride: u32,
}

fn sortTile(tile: u32) -> SortTile {
    let at = sortParams.tilesOffset + tile * SORT_TILE_WORDS;
    return SortTile(sortTables[at], sortTables[at + 1u], sortTables[at + 2u], sortTables[at + 3u]);
}

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
    let tileIndex = flatWorkgroupIndex(workgroup, groups);
    if (tileIndex >= sortParams.tileCount) {
        return;   // the whole workgroup: the index is uniform
    }
    let tile = sortTile(tileIndex);
    let order = scanTakeOrder(localIndex);
    // Counting doesn't care which thread has which key, so the loads go across the tile, side by side.
    var counts: array<u32, SCAN_WORDS>;
    for (var key = 0u; key < SORT_KEYS_PER_THREAD; key += 1u) {
        let at = tile.first + key * SORT_WORKGROUP_SIZE + localIndex;
        if (at < tile.end) {
            sortCountOne(&counts, sortDigit(sortKeysIn[at]));
        }
    }
    _ = scanExclusive(counts, order, localIndex);
    if (localIndex < SORT_BINS) {
        sortOffsets[tile.offsetsBase + localIndex * tile.offsetsStride] = sortUnpack(scanTotals, localIndex);
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
    let tileIndex = flatWorkgroupIndex(workgroup, groups);
    if (tileIndex >= sortParams.tileCount) {
        return;
    }
    let tile = sortTile(tileIndex);
    let order = scanTakeOrder(localIndex);
    if (localIndex < SORT_BINS) {
        sortTileOffsets[localIndex] = sortOffsets[tile.offsetsBase + localIndex * tile.offsetsStride];
    }

    // Ranking does care: each thread owns 8 neighbouring keys, in the scan's thread order.
    let first = tile.first + order.thread * SORT_KEYS_PER_THREAD;
    var keys: array<u32, ${KEYS_PER_THREAD}>;
    var values: array<u32, ${KEYS_PER_THREAD}>;
    var counts: array<u32, SCAN_WORDS>;
    for (var key = 0u; key < SORT_KEYS_PER_THREAD; key += 1u) {
        if (first + key < tile.end) {
            keys[key] = sortKeysIn[first + key];
            values[key] = sortValuesIn[first + key];
            sortCountOne(&counts, sortDigit(keys[key]));
        }
    }
    // The scan's barriers also make sortTileOffsets visible.
    var rank = scanExclusive(counts, order, localIndex);
    for (var key = 0u; key < SORT_KEYS_PER_THREAD; key += 1u) {
        if (first + key < tile.end) {
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

export async function createRadixSortPipelines(device: GPUDevice, scan: ScanPipelines | Promise<ScanPipelines>, label: string): Promise<RadixSortPipelines> {
    const module = device.createShaderModule({ label: `${label} radix sort`, code: radixSortWgsl() });
    const pipeline = (entryPoint: string) =>
        device.createComputePipelineAsync({ label: `${label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } });
    // The scan may still be compiling (a promise): the sort's own compiles are asked for first, so they run alongside it.
    const [count, scatter, scanPipelines] = await Promise.all([pipeline('sortCount'), pipeline('sortScatter'), scan]);
    return { count, scatter, scan: scanPipelines };
}

/** The buffers one sort uses. Keys and values ping-pong between `a` and `b`: the result ends in `a` after an even number of passes, in `b` after an odd one. */
export interface RadixSortBuffers {
    readonly keysA: GPUBuffer;
    readonly valuesA: GPUBuffer;
    readonly keysB: GPUBuffer;
    readonly valuesB: GPUBuffer;
    /** At least `offsetWords` words (from {@link sortTileTable}). */
    readonly offsets: GPUBuffer;
    /** At least `scanTileSumWords(offsetWords)` words. */
    readonly tileSums: GPUBuffer;
    /** Holds the tile table at word `tilesOffset`. */
    readonly tables: GPUBuffer;
    readonly tilesOffset: number;
    /** Uniforms: pass `p`'s `SortParams` at `paramsOffset + p * paramsStride`, then the offsets scan's `ScanParams`. */
    readonly params: GPUBuffer;
    readonly paramsOffset: number;
    readonly paramsStride: number;
}

/** A sort over fixed buffers and segments, with its bind groups made once. */
export interface RadixSort {
    readonly passes: number;
    /** Whether the sorted keys and values end in buffers `a` (an even pass count) or `b`. */
    readonly resultInA: boolean;
    /** Writes the uniforms. They never change, so once is enough. */
    writeParams(queue: GPUQueue): void;
    encode(pass: GPUComputePassEncoder): void;
}

/** A sort of segments whose tile table (`sortTileTable`) is already in `buffers.tables`. */
export function createRadixSort(
    device: GPUDevice,
    pipelines: RadixSortPipelines,
    buffers: RadixSortBuffers,
    table: { tileCount: number; offsetWords: number },
    keyBits: number,
    label: string,
): RadixSort {
    const passes = radixPassCount(keyBits);
    const { tileCount, offsetWords } = table;
    const scanParamsOffset = buffers.paramsOffset + passes * buffers.paramsStride;
    const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;

    const passGroups = Array.from({ length: passes }, (_, pass) => {
        const forward = pass % 2 === 0;
        const group = (pipeline: GPUComputePipeline, withValues: boolean): GPUBindGroup => {
            const entries: GPUBindGroupEntry[] = [
                { binding: 0, resource: { buffer: forward ? buffers.keysA : buffers.keysB } },
                { binding: 4, resource: { buffer: buffers.offsets, size: Math.max(4, offsetWords * 4) } },
                { binding: 5, resource: { buffer: buffers.tables } },
                { binding: 6, resource: { buffer: buffers.params, offset: buffers.paramsOffset + pass * buffers.paramsStride, size: SORT_PARAMS_BYTES } },
            ];
            if (withValues) {
                entries.push(
                    { binding: 1, resource: { buffer: forward ? buffers.valuesA : buffers.valuesB } },
                    { binding: 2, resource: { buffer: forward ? buffers.keysB : buffers.keysA } },
                    { binding: 3, resource: { buffer: forward ? buffers.valuesB : buffers.valuesA } },
                );
            }
            return device.createBindGroup({ label: `${label} pass ${pass}`, layout: pipeline.getBindGroupLayout(0), entries });
        };
        return { count: group(pipelines.count, false), scatter: group(pipelines.scatter, true) };
    });
    const scanGroups: ScanBindGroups = createScanBindGroups(
        device,
        pipelines.scan,
        {
            data: { buffer: buffers.offsets, size: Math.max(4, offsetWords * 4) },
            tileSums: { buffer: buffers.tileSums },
            params: { buffer: buffers.params, offset: scanParamsOffset, size: SCAN_PARAMS_BYTES },
            count: offsetWords,
        },
        `${label} offsets scan`,
    );

    return {
        passes,
        resultInA: passes % 2 === 0,
        writeParams(queue) {
            for (let pass = 0; pass < passes; pass++) {
                queue.writeBuffer(buffers.params, buffers.paramsOffset + pass * buffers.paramsStride, sortParams(pass, tileCount, buffers.tilesOffset));
            }
            queue.writeBuffer(buffers.params, scanParamsOffset, scanParams(offsetWords));
        },
        encode(pass) {
            if (tileCount === 0) return;
            for (const groups of passGroups) {
                pass.setPipeline(pipelines.count);
                pass.setBindGroup(0, groups.count);
                dispatchWorkgroups2d(pass, tileCount, maxPerDimension);
                encodeScan(pass, pipelines.scan, scanGroups, maxPerDimension);
                pass.setPipeline(pipelines.scatter);
                pass.setBindGroup(0, groups.scatter);
                dispatchWorkgroups2d(pass, tileCount, maxPerDimension);
            }
        },
    };
}
