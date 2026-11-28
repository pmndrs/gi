/**
 * Exclusive prefix sum of a `u32` array on the GPU: reduce, then scan. Three
 * dispatches:
 *
 * 1. `scanReduce`, one workgroup per tile of {@link SCAN_TILE_ITEMS}: the
 *    tile's sum.
 * 2. `scanAcrossTiles`, one workgroup: the exclusive prefix of the tile sums, a
 *    tile's worth at a time, carrying the running total; the grand total is
 *    handed to the io's `scanStoreTotal`.
 * 3. `scanDownsweep`, one workgroup per tile: the tile's own exclusive prefix
 *    plus the tile's offset, handed item by item to the io's `scanStore`.
 *
 * No workgroup ever waits on another (WebGPU promises no forward progress
 * between workgroups, so a decoupled look-back could hang); the dispatch
 * boundaries do the synchronising. Inside a workgroup the work is done with
 * subgroup operations, or with the declared fallback in workgroup memory on
 * devices without `subgroups` (workgroup-scan.ts).
 *
 * **One scan, several uses.** What the scan reads and what it does with each
 * prefix is a {@link ScanIo}: a few lines of WGSL with the data bindings and
 * three functions. The platform's utilities (gpu-compute.ts) give it three
 * (./wgsl.ts): a scan into another buffer, a scan in place, and stream
 * compaction, whose store writes each kept item's index at its prefix.
 *
 * **Where it came from.** The frozen BVH copy's `gpu-scan.ts`, with the
 * radiance cache's `ScanIo` changes (RC1.3), moved here when gi moved onto the
 * published `@pmndrs/bvh` (pmndrs/gi#63, docs/plans/sibling-migration.md §3,
 * item 3). The BVH package keeps its own scan, internal and subgroups-only, for
 * its builders. This one is gi's.
 */

import { dispatchWorkgroups2d, SCAN_WORKGROUP_SIZE, WORKGROUP_INDEX_WGSL, workgroupScanWgsl, type ScanReduction } from './workgroup-scan.ts';

export type { ScanReduction } from './workgroup-scan.ts';

/** Items each thread sums or scans, one after another, in a tile. */
const ITEMS_PER_THREAD = 8;

/** Items one workgroup handles. */
export const SCAN_TILE_ITEMS = SCAN_WORKGROUP_SIZE * ITEMS_PER_THREAD;

/** Bytes of the scan's uniform block (`ScanParams`). */
export const SCAN_PARAMS_BYTES = 16;

/** The scan's own bindings in group 0. An io's bindings start at {@link SCAN_IO_FIRST_BINDING}. */
export const SCAN_BINDING = {
    /** `scanTileSums`: storage, read-write, {@link scanTileSumWords} words. */
    tileSums: 0,
    /** `scanParams`: uniform, {@link SCAN_PARAMS_BYTES} bytes ({@link scanParams}). */
    params: 1,
} as const;

/** The first binding number an io may use. */
export const SCAN_IO_FIRST_BINDING = 2;

/** Entry points of the three scan kernels, in the order they run. */
export const SCAN_ENTRY_POINTS = {
    reduce: 'scanReduce',
    acrossTiles: 'scanAcrossTiles',
    downsweep: 'scanDownsweep',
} as const;

/** Tile sums a scan of `count` items needs, plus one word for the grand total. */
export function scanTileSumWords(count: number): number {
    return Math.ceil(count / SCAN_TILE_ITEMS) + 1;
}

/** `ScanParams` for `count` items. */
export function scanParams(count: number): Uint32Array<ArrayBuffer> {
    return new Uint32Array([count, Math.ceil(count / SCAN_TILE_ITEMS), 0, 0]);
}

/**
 * What a scan reads and what it does with the results: WGSL declaring the
 * io's bindings (group 0, from {@link SCAN_IO_FIRST_BINDING}) and three
 * functions, which may use `scanTileSums` and `scanParams`:
 *
 * ```wgsl
 * // The value of item 'at' (at < scanParams.count).
 * fn scanLoad(at: u32) -> u32
 * // Called by the downsweep once per item, with its value and its exclusive prefix.
 * fn scanStore(at: u32, value: u32, prefix: u32)
 * // Called once, by one thread of scanAcrossTiles, with the sum of all items.
 * fn scanStoreTotal(total: u32)
 * ```
 *
 * `scanStore` runs after every `scanLoad` of the same thread's items, so an
 * io may overwrite its input in place. The io's names must not collide with
 * the scan's (`scan...`; compute.test.ts checks the platform's ios).
 */
export interface ScanIo {
    /** Shown in pipeline labels. */
    readonly name: string;
    readonly wgsl: string;
}

/** Which path the kernels take (default subgroups) and what they read and write. */
export interface ScanKernelOptions {
    readonly reduction?: ScanReduction;
    readonly io: ScanIo;
}

/** WGSL of the three scan kernels. With `subgroups` (the default) the device needs that feature. */
export function scanWgsl(options: ScanKernelOptions): string {
    const reduction = options.reduction ?? 'subgroups';
    const io = options.io;
    const enable = reduction === 'subgroups' ? 'enable subgroups;\n' : '';
    return /* wgsl */ `${enable}
struct ScanParams {
    count: u32,
    tileCount: u32,
    unused0: u32,
    unused1: u32,
}

// One sum per tile (after scanAcrossTiles: each tile's offset), then whatever the io keeps there.
@group(0) @binding(${SCAN_BINDING.tileSums}) var<storage, read_write> scanTileSums: array<u32>;
@group(0) @binding(${SCAN_BINDING.params}) var<uniform> scanParams: ScanParams;

const SCAN_ITEMS_PER_THREAD: u32 = ${ITEMS_PER_THREAD}u;
const SCAN_TILE: u32 = ${SCAN_TILE_ITEMS}u;
${WORKGROUP_INDEX_WGSL}
${workgroupScanWgsl(1, reduction)}

// ---- The io: ${io.name} ----
${io.wgsl}
// ---- The scan ----

// The exclusive prefix of one tile, shifted by 'offset': each thread owns 8
// neighbouring items, sums them one after another and gets the sum of
// everything before its run from the workgroup scan. Works on the io's items,
// or on scanTileSums in place when 'inTileSums' (WGSL can't pass a storage
// array to a function). Afterwards scanTotals[0] holds the tile's total.
fn scanTile(order: ScanOrder, localIndex: u32, tileStart: u32, end: u32, offset: u32, inTileSums: bool) {
    let first = tileStart + order.thread * SCAN_ITEMS_PER_THREAD;
    var values: array<u32, ${ITEMS_PER_THREAD}>;
    var before: array<u32, ${ITEMS_PER_THREAD}>;
    var sum = 0u;
    for (var item = 0u; item < SCAN_ITEMS_PER_THREAD; item += 1u) {
        let at = first + item;
        var value = 0u;
        if (at < end) {
            if (inTileSums) {
                value = scanTileSums[at];
            } else {
                value = scanLoad(at);
            }
        }
        values[item] = value;
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
                scanStore(at, values[item], runStart + before[item]);
            }
        }
    }
}

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
fn ${SCAN_ENTRY_POINTS.reduce}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let tile = flatWorkgroupIndex(workgroup, groups);
    if (tile >= scanParams.tileCount) {
        return;   // the whole workgroup: tile is uniform
    }
    let order = scanTakeOrder(localIndex);
    let first = tile * SCAN_TILE + order.thread * SCAN_ITEMS_PER_THREAD;
    var sum = 0u;
    for (var item = 0u; item < SCAN_ITEMS_PER_THREAD; item += 1u) {
        let at = first + item;
        if (at < scanParams.count) {
            sum += scanLoad(at);
        }
    }
    _ = scanExclusive(array<u32, 1>(sum), order, localIndex);
    if (localIndex == 0u) {
        scanTileSums[tile] = scanTotals[0];
    }
}

// One workgroup. With no tiles (an empty scan) the loop doesn't run and the total is 0.
@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
fn ${SCAN_ENTRY_POINTS.acrossTiles}(@builtin(local_invocation_index) localIndex: u32) {
    let order = scanTakeOrder(localIndex);
    let tileCount = scanParams.tileCount;
    var carry = 0u;
    for (var start = 0u; start < tileCount; start += SCAN_TILE) {
        scanTile(order, localIndex, start, tileCount, carry, true);
        carry += scanTotals[0];
        // The next round reuses the workgroup scan's memory.
        workgroupBarrier();
    }
    if (localIndex == 0u) {
        scanStoreTotal(carry);
    }
}

@compute @workgroup_size(${SCAN_WORKGROUP_SIZE})
fn ${SCAN_ENTRY_POINTS.downsweep}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let tile = flatWorkgroupIndex(workgroup, groups);
    if (tile >= scanParams.tileCount) {
        return;
    }
    let order = scanTakeOrder(localIndex);
    scanTile(order, localIndex, tile * SCAN_TILE, scanParams.count, scanTileSums[tile], false);
}
`;
}

/** The scan's three pipelines, compiled once and shared by every scan with the same io on a device. */
export interface ScanPipelines {
    readonly reduce: GPUComputePipeline;
    readonly tileSums: GPUComputePipeline;
    readonly downsweep: GPUComputePipeline;
}

/** Compiles the three kernels with one explicit layout, so one bind group serves all three. */
export async function createScanPipelines(device: GPUDevice, label: string, options: ScanKernelOptions & { readonly layout: GPUPipelineLayout }): Promise<ScanPipelines> {
    const module = device.createShaderModule({ label: `${label} scan`, code: scanWgsl(options) });
    const pipeline = (entryPoint: string) => device.createComputePipelineAsync({ label: `${label} ${entryPoint}`, layout: options.layout, compute: { module, entryPoint } });
    const [reduce, tileSums, downsweep] = await Promise.all([pipeline(SCAN_ENTRY_POINTS.reduce), pipeline(SCAN_ENTRY_POINTS.acrossTiles), pipeline(SCAN_ENTRY_POINTS.downsweep)]);
    return { reduce, tileSums, downsweep };
}

/** Bind groups for one scan, one per kernel (they may be the same group), and the item count. */
export interface ScanBindGroups {
    readonly reduce: GPUBindGroup;
    readonly tileSums: GPUBindGroup;
    readonly downsweep: GPUBindGroup;
    readonly count: number;
}

/**
 * Records the three dispatches of one scan into `pass`. A scan of nothing
 * records nothing. `dynamicOffsets` go with every bind group, for layouts with
 * dynamic offsets (the platform's utilities pick each encode's parameter slot
 * that way).
 */
export function encodeScan(pass: GPUComputePassEncoder, pipelines: ScanPipelines, groups: ScanBindGroups, maxPerDimension: number, dynamicOffsets: readonly number[] = []): void {
    if (groups.count === 0) return;
    const tiles = Math.ceil(groups.count / SCAN_TILE_ITEMS);
    pass.setPipeline(pipelines.reduce);
    pass.setBindGroup(0, groups.reduce, dynamicOffsets);
    dispatchWorkgroups2d(pass, tiles, maxPerDimension);
    pass.setPipeline(pipelines.tileSums);
    pass.setBindGroup(0, groups.tileSums, dynamicOffsets);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(pipelines.downsweep);
    pass.setBindGroup(0, groups.downsweep, dynamicOffsets);
    dispatchWorkgroups2d(pass, tiles, maxPerDimension);
}
