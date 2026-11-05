/**
 * Exclusive prefix sum of a `u32` array on the GPU, in place: reduce, then
 * scan (docs/design/bvh-framework.md, "Sort and scan"). Three dispatches:
 *
 * 1. `scanReduce`, one workgroup per tile of {@link SCAN_TILE_ITEMS}: the
 *    tile's sum.
 * 2. `scanAcrossTiles`, one workgroup: the exclusive prefix of the tile sums, a
 *    tile's worth at a time, carrying the running total; the grand total goes
 *    after the last tile sum.
 * 3. `scanDownsweep`, one workgroup per tile: the tile's own exclusive prefix
 *    plus the tile's offset.
 *
 * No workgroup ever waits on another (WebGPU promises no forward progress
 * between workgroups, so a decoupled look-back could hang); the dispatch
 * boundaries do the synchronising. Inside a workgroup the work is done with
 * subgroup operations (gpu-workgroup-scan.ts), so it needs the `subgroups`
 * feature. The LBVH builder uses it to number the nodes it keeps, and the
 * radix sort for its digit offsets.
 */

import { dispatchWorkgroups2d, SCAN_WORKGROUP_SIZE, WORKGROUP_INDEX_WGSL, workgroupScanWgsl } from './gpu-workgroup-scan.ts';

/** Items each thread sums or scans, one after another, in a tile. */
const ITEMS_PER_THREAD = 8;

/** Items one workgroup handles. */
export const SCAN_TILE_ITEMS = SCAN_WORKGROUP_SIZE * ITEMS_PER_THREAD;

/** Bytes of the scan's uniform block (`ScanParams`). */
export const SCAN_PARAMS_BYTES = 16;

/** Tile sums a scan of `count` items needs, plus one word for the grand total. */
export function scanTileSumWords(count: number): number {
    return Math.ceil(count / SCAN_TILE_ITEMS) + 1;
}

/** `ScanParams` for `count` items. */
export function scanParams(count: number): Uint32Array<ArrayBuffer> {
    return new Uint32Array([count, Math.ceil(count / SCAN_TILE_ITEMS), 0, 0]);
}

/** WGSL of the three scan kernels. Needs `subgroups`. */
export function scanWgsl(): string {
    return /* wgsl */ `enable subgroups;

struct ScanParams {
    count: u32,
    tileCount: u32,
    unused0: u32,
    unused1: u32,
}

@group(0) @binding(0) var<storage, read_write> scanData: array<u32>;
// One sum per tile, then the grand total.
@group(0) @binding(1) var<storage, read_write> scanTileSums: array<u32>;
@group(0) @binding(2) var<uniform> scanParams: ScanParams;

const SCAN_ITEMS_PER_THREAD: u32 = ${ITEMS_PER_THREAD}u;
const SCAN_TILE: u32 = ${SCAN_TILE_ITEMS}u;
${WORKGROUP_INDEX_WGSL}
${workgroupScanWgsl(1)}

// The exclusive prefix of one tile, written back in place and shifted by
// 'offset': each thread owns 8 neighbouring items, sums them one after another
// and gets the sum of everything before its run from the workgroup scan.
// Works on scanData, or on scanTileSums when 'inTileSums' (WGSL can't pass a
// storage array to a function). Afterwards scanTotals[0] holds the tile's total.
fn scanTileInPlace(order: ScanOrder, localIndex: u32, tileStart: u32, end: u32, offset: u32, inTileSums: bool) {
    let first = tileStart + order.thread * SCAN_ITEMS_PER_THREAD;
    var before: array<u32, ${ITEMS_PER_THREAD}>;
    var sum = 0u;
    for (var item = 0u; item < SCAN_ITEMS_PER_THREAD; item += 1u) {
        let at = first + item;
        var value = 0u;
        if (at < end) {
            if (inTileSums) {
                value = scanTileSums[at];
            } else {
                value = scanData[at];
            }
        }
        before[item] = sum;
        sum += value;
    }
    let runStart = scanExclusive(array<u32, 1>(sum), order, localIndex)[0] + offset;
    for (var item = 0u; item < SCAN_ITEMS_PER_THREAD; item += 1u) {
        let at = first + item;
        if (at < end) {
            if (inTileSums) {
                scanTileSums[at] = runStart + before[item];
            } else {
                scanData[at] = runStart + before[item];
            }
        }
    }
}

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
fn scanReduce(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let tile = flatWorkgroupIndex(workgroup, groups);
    if (tile >= scanParams.tileCount) {
        return;   // the whole workgroup: tile is uniform
    }
    let order = scanTakeOrder();
    let first = tile * SCAN_TILE + order.thread * SCAN_ITEMS_PER_THREAD;
    var sum = 0u;
    for (var item = 0u; item < SCAN_ITEMS_PER_THREAD; item += 1u) {
        let at = first + item;
        if (at < scanParams.count) {
            sum += scanData[at];
        }
    }
    _ = scanExclusive(array<u32, 1>(sum), order, localIndex);
    if (localIndex == 0u) {
        scanTileSums[tile] = scanTotals[0];
    }
}

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
fn scanAcrossTiles(@builtin(local_invocation_index) localIndex: u32) {
    let order = scanTakeOrder();
    let tileCount = scanParams.tileCount;
    var carry = 0u;
    for (var start = 0u; start < tileCount; start += SCAN_TILE) {
        scanTileInPlace(order, localIndex, start, tileCount, carry, true);
        carry += scanTotals[0];
        // The next round reuses the slot totals.
        workgroupBarrier();
    }
    if (localIndex == 0u) {
        scanTileSums[tileCount] = carry;
    }
}

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
fn scanDownsweep(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let tile = flatWorkgroupIndex(workgroup, groups);
    if (tile >= scanParams.tileCount) {
        return;
    }
    let order = scanTakeOrder();
    scanTileInPlace(order, localIndex, tile * SCAN_TILE, scanParams.count, scanTileSums[tile], false);
}
`;
}

/** The scan's three pipelines, compiled once and shared by every scan on a device. */
export interface ScanPipelines {
    readonly reduce: GPUComputePipeline;
    readonly tileSums: GPUComputePipeline;
    readonly downsweep: GPUComputePipeline;
}

export async function createScanPipelines(device: GPUDevice, label: string): Promise<ScanPipelines> {
    const module = device.createShaderModule({ label: `${label} scan`, code: scanWgsl() });
    const pipeline = (entryPoint: string) =>
        device.createComputePipelineAsync({ label: `${label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } });
    const [reduce, tileSums, downsweep] = await Promise.all([pipeline('scanReduce'), pipeline('scanAcrossTiles'), pipeline('scanDownsweep')]);
    return { reduce, tileSums, downsweep };
}

/** Where one scan reads and writes: the data, its tile sums, and its `ScanParams` (at a 256-byte aligned offset). */
export interface ScanBinding {
    readonly data: GPUBufferBinding;
    readonly tileSums: GPUBufferBinding;
    readonly params: GPUBufferBinding;
    /** Items to scan; the same number as in the params. */
    readonly count: number;
}

/** Bind groups for one scan, made once per buffer set and reused. */
export interface ScanBindGroups {
    readonly reduce: GPUBindGroup;
    readonly tileSums: GPUBindGroup;
    readonly downsweep: GPUBindGroup;
    readonly count: number;
}

export function createScanBindGroups(device: GPUDevice, pipelines: ScanPipelines, binding: ScanBinding, label: string): ScanBindGroups {
    // 'auto' layouts are per pipeline, so each kernel gets its own bind group over the same buffers.
    const group = (pipeline: GPUComputePipeline) =>
        device.createBindGroup({
            label,
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: binding.data },
                { binding: 1, resource: binding.tileSums },
                { binding: 2, resource: binding.params },
            ],
        });
    return {
        reduce: group(pipelines.reduce),
        tileSums: group(pipelines.tileSums),
        downsweep: group(pipelines.downsweep),
        count: binding.count,
    };
}

/** Records the three dispatches of one scan into `pass`. A scan of nothing records nothing. */
export function encodeScan(pass: GPUComputePassEncoder, pipelines: ScanPipelines, groups: ScanBindGroups, maxPerDimension: number): void {
    if (groups.count === 0) return;
    const tiles = Math.ceil(groups.count / SCAN_TILE_ITEMS);
    pass.setPipeline(pipelines.reduce);
    pass.setBindGroup(0, groups.reduce);
    dispatchWorkgroups2d(pass, tiles, maxPerDimension);
    pass.setPipeline(pipelines.tileSums);
    pass.setBindGroup(0, groups.tileSums);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(pipelines.downsweep);
    pass.setBindGroup(0, groups.downsweep);
    dispatchWorkgroups2d(pass, tiles, maxPerDimension);
}
