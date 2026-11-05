/**
 * GPU H-PLOC builder (stage B2 of docs/design/bvh-framework.md, the target
 * builder): mesh trees built on the GPU in the tracer's layout, behind the
 * same contract as the LBVH (B1, gpu-lbvh.ts). It reuses B1's pieces: the
 * batches (mesh and tile tables, one dispatch for every mesh), triangle boxes,
 * Morton codes, the segmented radix sort, the scans, and the readback. H-PLOC
 * (Benthin et al., HPG 2024) climbs the same radix tree as the LBVH and
 * merges clusters with PLOC on the way up (gpu-hploc-wgsl.ts), which gives
 * trees close to a full SAH build's.
 *
 * **Steps**, all in one compute pass:
 *
 * 1. Triangle boxes, the centroid box and the mesh box (B1's kernel).
 * 2. Morton codes: B1's 30-bit cubic codes (`'cubic-30'`, the default, one
 *    sort word), or tinybvh's 63-bit codes (`'tinybvh-63'`, two sort words),
 *    which make the trees tinybvh's `BuildHPLOC` makes.
 * 3. The stable segmented radix sort, once per code word.
 * 4. The split metric between neighbours, then the clear of the handed-over words.
 * 5. The climb with the merges.
 * 6. Which made nodes stay (the SAH leaf collapse was decided as each node was
 *    made), a scan to number them and one to place their leaves' triangles,
 *    then the nodes, the triangles' tree positions, the triangles and each
 *    mesh's stats.
 *
 * **Quality** (CPU twin, hploc-cpu.ts, on Sponza): SAH cost 1.03 times
 * tinybvh's default build with the default codes and leaf collapse, against
 * 1.22 for the LBVH. With tinybvh's codes and one-triangle leaves the twin's
 * trees are exactly tinybvh's `BuildHPLOC`'s.
 *
 * **Limits.** A batch's largest buffer holds 64 bytes per triangle (the output
 * nodes); {@link planGpuHplocBatches} splits a scene to fit the device. A mesh
 * may have at most {@link HPLOC_MAX_MESH_TRIANGLES} triangles.
 *
 * **Checks.** As the LBVH: a mesh's tree is refused (`problem` set) on a bad
 * index, a handed-over word still unwritten after the bounded wait, a merge
 * that found no pair (NaN boxes), or a tree too deep; callers then use the
 * CPU builder (mesh-builder.ts).
 *
 * Needs the `subgroups` feature; without it the declared fallback is tinybvh
 * (B0). The merge's shape depends on the device's smallest subgroup size
 * (`device.adapterInfo.subgroupMinSize`), never on a fixed warp width.
 */

import { GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES, type GpuLayoutMeshBvh } from './gpu-layout.ts';
import {
    bufferLimit,
    buildTables,
    checkGeometry,
    checkOptions,
    copyBatchGeometry,
    indexBytes,
    MAP_MODE_READ,
    meshResult,
    placeMeshes,
    planMeshBatches,
    positionBytes,
    readTreeLayouts,
    UNIFORM_STRIDE,
    uploadBatchGeometry,
    USAGE,
    align4,
    type GpuLbvhGeometry,
    type GpuLbvhTrees,
    type MeshPlacement,
} from './gpu-lbvh.ts';
import { LBVH_FIRST_ROUND_TRIANGLES, LBVH_PARAMS_BYTES, LBVH_STATS_BYTES, LBVH_STATS_WORDS, LBVH_WORKGROUP_SIZE } from './gpu-lbvh-wgsl.ts';
import {
    HPLOC_CLIMB_WORDS,
    HPLOC_MAX_MESH_TRIANGLES,
    HPLOC_MORTON,
    HPLOC_PARAMS_BYTES,
    HPLOC_RECORD_WORDS,
    HPLOC_STAT,
    HPLOC_TIE_BREAK,
    hplocSlotsFor,
    hplocWgsl,
} from './gpu-hploc-wgsl.ts';
import type { HplocMortonCodes, HplocOptions } from './hploc-cpu.ts';
import { createRadixSort, createRadixSortPipelines, radixPassCount, type RadixSort, type RadixSortPipelines } from './gpu-radix-sort.ts';
import { createScanBindGroups, createScanPipelines, encodeScan, SCAN_PARAMS_BYTES, scanParams, scanTileSumWords, type ScanBindGroups } from './gpu-scan.ts';
import { dispatchWorkgroups2d } from './gpu-workgroup-scan.ts';

export type { HplocMortonCodes, HplocOptions, HplocTieBreak } from './hploc-cpu.ts';
export { HPLOC_MAX_MESH_TRIANGLES } from './gpu-hploc-wgsl.ts';

/** The smallest subgroup WebGPU allows, assumed when the device doesn't say. */
const MIN_SUBGROUP_SIZE = 4;

/** The biggest per-triangle share of any one batch buffer: the output nodes, 64 bytes. */
const LARGEST_BYTES_PER_TRIANGLE = Math.max(GPU_LAYOUT_NODE_BYTES, GPU_LAYOUT_TRIANGLE_BYTES, (HPLOC_CLIMB_WORDS + HPLOC_RECORD_WORDS + 1) * 4);

/** Bits of each code word the sorts sort. */
const SORT_BITS = { 'cubic-30': [30], 'tinybvh-63': [32, 31] } as const satisfies Record<HplocMortonCodes, readonly number[]>;

/**
 * Splits meshes into batches whose buffers this device can bind, in order. A
 * mesh too big for any batch, or for H-PLOC, comes back in `tooBig`.
 */
export function planGpuHplocBatches(device: GPUDevice, geometries: readonly GpuLbvhGeometry[]): { batches: number[][]; tooBig: number[] } {
    return planMeshBatches(device, geometries, LARGEST_BYTES_PER_TRIANGLE, HPLOC_MAX_MESH_TRIANGLES);
}

/** The steps of a batch's build, in order, as `GpuHplocBatch.encode` can time them. */
export const HPLOC_STAGES = ['boxes', 'morton', 'sort', 'deltas', 'clear', 'climb', 'mark', 'rank', 'emit', 'triangles'] as const;
export type HplocStage = (typeof HPLOC_STAGES)[number];

const KERNELS = [
    'lbvhStart',
    'lbvhBoxes',
    'lbvhMorton',
    'hplocMortonTinybvh',
    'hplocGatherHigh',
    'hplocDeltasKernel',
    'hplocClear',
    'hplocClimb',
    'hplocMark',
    'hplocEmit',
    'hplocTrianglesKernel',
    'hplocFinish',
] as const;
type HplocKernel = (typeof KERNELS)[number];

/** The bindings (gpu-lbvh-wgsl.ts and gpu-hploc-wgsl.ts) each kernel uses. Pipelines have 'auto' layouts, which hold exactly these. */
const BINDING = {
    positions: 0,
    indices: 1,
    flags: 2,
    primitiveBoxes: 3,
    keys: 4,
    values: 5,
    rank: 9,
    outNodes: 10,
    outTriangles: 11,
    outParents: 12,
    stats: 13,
    params: 14,
    tables: 15,
    statsView: 17,
    deltas: 22,
    codes: 23,
    scratch: 24,
    scratchView: 25,
    scratchFill: 26,
    links: 27,
    hplocParams: 28,
    valuesIn: 29,
} as const;
type BindingName = keyof typeof BINDING;

const KERNEL_BINDINGS: Record<HplocKernel, readonly BindingName[]> = {
    lbvhStart: ['rank', 'stats', 'params'],
    lbvhBoxes: ['positions', 'indices', 'primitiveBoxes', 'stats', 'params', 'tables'],
    lbvhMorton: ['primitiveBoxes', 'keys', 'values', 'params', 'tables', 'statsView'],
    hplocMortonTinybvh: ['primitiveBoxes', 'keys', 'values', 'codes', 'params', 'tables', 'statsView'],
    hplocGatherHigh: ['keys', 'values', 'codes', 'valuesIn', 'params', 'tables'],
    hplocDeltasKernel: ['keys', 'values', 'codes', 'deltas', 'params', 'tables', 'hplocParams'],
    hplocClear: ['scratchFill', 'hplocParams'],
    hplocClimb: ['primitiveBoxes', 'values', 'deltas', 'scratch', 'links', 'stats', 'params', 'tables', 'hplocParams'],
    hplocMark: ['scratchView', 'links', 'statsView', 'params', 'tables', 'hplocParams'],
    hplocEmit: ['primitiveBoxes', 'values', 'scratchView', 'links', 'outNodes', 'outParents', 'statsView', 'params', 'tables', 'hplocParams'],
    hplocTrianglesKernel: ['positions', 'indices', 'flags', 'values', 'links', 'outTriangles', 'params', 'tables', 'hplocParams'],
    hplocFinish: ['primitiveBoxes', 'values', 'scratchView', 'links', 'stats', 'params', 'tables', 'hplocParams'],
};

/** Kernels that read the sorted keys and values, which end in buffers a or b depending on the codes. */
const READS_SORTED = new Set<HplocKernel>(['hplocDeltasKernel', 'hplocClimb', 'hplocEmit', 'hplocTrianglesKernel', 'hplocFinish']);

/** The device's smallest subgroup, as it reports it (Chrome: `device.adapterInfo.subgroupMinSize`). */
function minSubgroupSize(device: GPUDevice): number {
    const info = (device as GPUDevice & { adapterInfo?: { subgroupMinSize?: number } }).adapterInfo;
    const size = info?.subgroupMinSize;
    return typeof size === 'number' && size >= MIN_SUBGROUP_SIZE ? size : MIN_SUBGROUP_SIZE;
}

function checkHplocOptions(options: HplocOptions): Required<HplocOptions> {
    const mortonCodes = options.mortonCodes ?? 'cubic-30';
    const tieBreak = options.tieBreak ?? 'pairs';
    if (!(mortonCodes in HPLOC_MORTON)) throw new RangeError(`unknown Morton codes: ${String(mortonCodes)}`);
    if (!(tieBreak in HPLOC_TIE_BREAK)) throw new RangeError(`unknown tie break: ${String(tieBreak)}`);
    return { ...checkOptions(options), mortonCodes, tieBreak };
}

/** Compiles the builder's kernels once per device and builds any number of batches with them. */
export class GpuHplocBuilder {
    /** The builder needs subgroups; without them, callers use the declared fallback (tinybvh, B0). */
    static isSupported(device: GPUDevice): boolean {
        return device.features.has('subgroups');
    }

    /** Cluster slots per lane in the merge on this device (1 when subgroups are at least 32 wide). */
    readonly slots: number;
    private readonly device: GPUDevice;
    private readonly label: string;
    private pipelines: Record<HplocKernel, GPUComputePipeline> | null = null;
    private sortPipelines: RadixSortPipelines | null = null;

    constructor(device: GPUDevice, options: { label?: string } = {}) {
        if (!GpuHplocBuilder.isSupported(device)) throw new Error('@pmndrs/bvh: the GPU H-PLOC builder needs a device with the "subgroups" feature');
        this.device = device;
        this.label = options.label ?? 'gpu hploc';
        this.slots = hplocSlotsFor(minSubgroupSize(device));
    }

    /** Compiles the kernels (asynchronously, so the main thread doesn't stall). */
    async prepare(): Promise<void> {
        if (this.pipelines) return;
        const device = this.device;
        const module = device.createShaderModule({ label: this.label, code: hplocWgsl(this.slots) });
        const scan = await createScanPipelines(device, this.label);
        const [sort, ...kernels] = await Promise.all([
            createRadixSortPipelines(device, scan, this.label),
            ...KERNELS.map((entryPoint) =>
                device.createComputePipelineAsync({ label: `${this.label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } }),
            ),
        ]);
        this.sortPipelines = sort;
        this.pipelines = Object.fromEntries(KERNELS.map((name, index) => [name, kernels[index]])) as Record<HplocKernel, GPUComputePipeline>;
    }

    get ready(): boolean {
        return this.pipelines !== null;
    }

    /** Builds every mesh's tree in one batch and one submit. A refused mesh has `problem` set; the others are fine. */
    async buildMany(geometries: readonly GpuLbvhGeometry[], options: HplocOptions = {}): Promise<GpuLbvhTrees> {
        const batch = this.createBatch(geometries, options);
        try {
            const encoder = this.device.createCommandEncoder({ label: this.label });
            batch.encode(encoder);
            this.device.queue.submit([encoder.finish()]);
            return await batch.finish();
        } finally {
            batch.destroy();
        }
    }

    /** Builds one mesh's tree. Throws if the build was refused. */
    async build(geometry: GpuLbvhGeometry, options: HplocOptions = {}): Promise<GpuLbvhTrees> {
        const trees = await this.buildMany([geometry], options);
        const problem = trees.meshes[0].problem;
        if (problem) {
            trees.destroy();
            throw new Error(`@pmndrs/bvh: GPU H-PLOC build refused: ${problem}`);
        }
        return trees;
    }

    /**
     * Uploads the inputs and sets up every buffer and bind group of a batch, so
     * it can be recorded with `encode` (any number of times: each recording
     * rebuilds the same trees) and read with `finish`. Throws if the batch is
     * too big for the device (see {@link planGpuHplocBatches}).
     */
    createBatch(geometries: readonly GpuLbvhGeometry[], options: HplocOptions = {}): GpuHplocBatch {
        if (!this.pipelines || !this.sortPipelines) throw new Error('@pmndrs/bvh: call prepare() before building');
        if (geometries.length === 0) throw new RangeError('a batch needs at least one mesh');
        const settings = checkHplocOptions(options);
        geometries.forEach(checkGeometry);
        const plan = planGpuHplocBatches(this.device, geometries);
        if (plan.batches.length !== 1 || plan.tooBig.length > 0) {
            throw new RangeError(
                `these meshes don't fit one batch on this device (buffers of at most ${bufferLimit(this.device)} bytes, ` +
                    `meshes of at most ${HPLOC_MAX_MESH_TRIANGLES} triangles); split them with planGpuHplocBatches`,
            );
        }
        return new GpuHplocBatch(this.device, this.label, this.pipelines, this.sortPipelines, geometries, settings);
    }

    /** Copies every usable tree of a batch back into CPU memory (one readback). */
    readLayouts(trees: GpuLbvhTrees): Promise<(GpuLayoutMeshBvh | null)[]> {
        return readTreeLayouts(this.device, trees, this.label);
    }
}

type BatchBuffer = 'positions' | 'indices' | 'flags' | 'stats' | 'outNodes' | 'outTriangles' | 'outParents';

/** One batch of mesh builds: inputs, scratch and outputs, recorded into one compute pass. */
export class GpuHplocBatch {
    readonly meshCount: number;
    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly mortonCodes: HplocMortonCodes;
    private readonly pipelines: Record<HplocKernel, GPUComputePipeline>;
    private readonly sortPipelines: RadixSortPipelines;
    private readonly meshes: MeshPlacement[];
    private readonly tileCount: number;
    /** Workgroups of the clear: one vec4 of the climb's scratch per thread. */
    private readonly clearWorkgroups: number;
    /** Buffers freed with the batch; the outputs are handed to `finish`'s caller instead. */
    private readonly owned: GPUBuffer[] = [];
    private readonly buffers: Record<BatchBuffer, GPUBuffer>;
    private readonly groups: Record<HplocKernel, GPUBindGroup>;
    /** One sort per code word, in order: the low word's first. */
    private readonly sorts: RadixSort[];
    private readonly rankScan: ScanBindGroups;
    private readonly leafRunScan: ScanBindGroups;
    private finished = false;
    private destroyed = false;

    constructor(
        device: GPUDevice,
        label: string,
        pipelines: Record<HplocKernel, GPUComputePipeline>,
        sortPipelines: RadixSortPipelines,
        geometries: readonly GpuLbvhGeometry[],
        options: Required<HplocOptions>,
    ) {
        this.device = device;
        this.label = label;
        this.mortonCodes = options.mortonCodes;
        this.pipelines = pipelines;
        this.sortPipelines = sortPipelines;
        this.meshCount = geometries.length;
        this.meshes = placeMeshes(geometries);
        const last = this.meshes[this.meshes.length - 1];
        const triangleTotal = last.triangleStart + last.triangleCount;
        const innerTotal = last.innerStart + last.innerCount;
        const { tables, tilesOffset, tileCount, sortTilesOffset, sortTable } = buildTables(this.meshes);
        this.tileCount = tileCount;
        const tinybvhCodes = options.mortonCodes === 'tinybvh-63';

        const storage = USAGE.STORAGE | USAGE.COPY_DST;
        const output = USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST;
        const make = (name: string, bytes: number, usage: number, owned = true): GPUBuffer => {
            const buffer = device.createBuffer({ label: `${label} ${name}`, size: Math.max(16, align4(bytes)), usage });
            if (owned) this.owned.push(buffer);
            return buffer;
        };
        const lastPositionEnd = last.positionByteStart + align4(positionBytes(last.geometry));
        const lastIndexEnd = this.meshes.reduce((end, mesh) => Math.max(end, mesh.indexByteStart + align4(indexBytes(mesh))), 4);
        const lastFlagEnd = this.meshes.reduce((end, mesh) => Math.max(end, mesh.flagsWordStart + (mesh.hasFlags ? mesh.triangleCount : 0)), 1);
        this.buffers = {
            positions: make('positions', lastPositionEnd, storage),
            indices: make('indices', lastIndexEnd, storage),
            flags: make('flags', lastFlagEnd * 4, storage),
            // Every mesh's stats, then the word lbvhStart clears for the LBVH's climb queue (unused here).
            stats: make('stats', this.meshCount * LBVH_STATS_BYTES + 16, USAGE.STORAGE | USAGE.COPY_SRC),
            outNodes: make('nodes', innerTotal * GPU_LAYOUT_NODE_BYTES, output, false),
            outTriangles: make('triangles', triangleTotal * GPU_LAYOUT_TRIANGLE_BYTES, output, false),
            outParents: make('parents', innerTotal * 4, output, false),
        };
        // hplocScratch: [climb words per radix-tree node][a record per made node][a stored cluster per sorted position].
        const recordsOffset = innerTotal * HPLOC_CLIMB_WORDS;
        const listsOffset = recordsOffset + innerTotal * HPLOC_RECORD_WORDS;
        // hplocLinks: [the node scan, one more at the end][the leaf-run scan, one more][2 children per made node]
        // [parents: leaves, made nodes][tree positions]. The scans are bound at offsets, which must be 256-byte aligned.
        const leafRunsOffset = Math.ceil((innerTotal + 1) / 64) * 64;
        const childrenOffset = leafRunsOffset + innerTotal + 1;
        const parentsOffset = childrenOffset + innerTotal * 2;
        const treePositionsOffset = parentsOffset + triangleTotal + innerTotal;
        // The scratch is cleared in whole vec4s.
        const scratchQuads = Math.ceil((listsOffset + triangleTotal) / 4);
        this.clearWorkgroups = Math.ceil(scratchQuads / LBVH_WORKGROUP_SIZE);
        const scratch = {
            tables: make('tables', tables.byteLength, storage),
            primitiveBoxes: make('primitive boxes', triangleTotal * 32, storage),
            keysA: make('keys a', triangleTotal * 4, storage),
            valuesA: make('values a', triangleTotal * 4, storage),
            keysB: make('keys b', triangleTotal * 4, storage),
            valuesB: make('values b', triangleTotal * 4, storage),
            sortOffsets: make('sort offsets', sortTable.offsetWords * 4, storage),
            tileSums: make('scan tile sums', Math.max(scanTileSumWords(sortTable.offsetWords), scanTileSumWords(innerTotal + 1)) * 4, storage),
            codes: make('codes', tinybvhCodes ? triangleTotal * 8 : 16, storage),
            deltas: make('deltas', triangleTotal * 8, storage),
            scratch: make('climb scratch', scratchQuads * 16, storage),
            links: make('links', (treePositionsOffset + triangleTotal) * 4, storage),
        };
        device.queue.writeBuffer(scratch.tables, 0, tables);
        uploadBatchGeometry(device.queue, this.meshes, this.buffers);

        // Uniform records, 256 bytes apart: LbvhParams, each sort's passes and scan, the rank scan, HplocParams.
        const sortBits = SORT_BITS[options.mortonCodes];
        // A sort's uniforms: one record per pass, then its scan's.
        const sortRecords = sortBits.map((bits) => radixPassCount(bits) + 1);
        const rankScanRecord = 1 + sortRecords.reduce((total, records) => total + records, 0);
        const leafRunScanRecord = rankScanRecord + 1;
        const hplocParamsRecord = leafRunScanRecord + 1;
        const params = make('params', (hplocParamsRecord + 1) * UNIFORM_STRIDE, USAGE.UNIFORM | USAGE.COPY_DST);
        const lbvhParams = new ArrayBuffer(LBVH_PARAMS_BYTES);
        // LbvhParams, field by field (gpu-lbvh-wgsl.ts).
        const words = new Uint32Array(lbvhParams);
        const floats = new Float32Array(lbvhParams);
        words[0] = this.meshCount; // meshCount
        words[1] = tileCount; // tileCount
        words[2] = tilesOffset; // tilesOffset
        words[3] = innerTotal; // innerTotal
        words[4] = options.maxLeafPrimitives; // maxLeaf
        floats[5] = options.traversalCost; // traversalCost
        floats[6] = options.intersectionCost; // intersectionCost
        words[7] = LBVH_FIRST_ROUND_TRIANGLES; // climbLimit (the LBVH's; unused here)
        // HplocParams, field by field (gpu-hploc-wgsl.ts).
        const hplocParams = new Uint32Array([
            triangleTotal, // triangleTotal
            innerTotal, // innerTotal
            HPLOC_MORTON[options.mortonCodes], // mortonCodes
            recordsOffset, // recordsOffset
            listsOffset, // listsOffset
            childrenOffset, // childrenOffset
            parentsOffset, // parentsOffset
            treePositionsOffset, // treePositionsOffset
            HPLOC_TIE_BREAK[options.tieBreak], // tieBreak
            leafRunsOffset, // leafRunsOffset
            scratchQuads, // scratchQuads
            0, // unused
        ]);
        if (hplocParams.byteLength !== HPLOC_PARAMS_BYTES) throw new Error('HplocParams changed size');

        // The sorted keys and values end in a after an even number of passes, in b after an odd one.
        const sortBuffers = {
            keysA: scratch.keysA,
            valuesA: scratch.valuesA,
            keysB: scratch.keysB,
            valuesB: scratch.valuesB,
            offsets: scratch.sortOffsets,
            tileSums: scratch.tileSums,
            tables: scratch.tables,
            tilesOffset: sortTilesOffset,
            params,
            paramsStride: UNIFORM_STRIDE,
        };
        let record = 1;
        this.sorts = sortBits.map((bits, index) => {
            const sort = createRadixSort(device, sortPipelines, { ...sortBuffers, paramsOffset: record * UNIFORM_STRIDE }, sortTable, bits, `${label} sort ${index}`);
            record += sortRecords[index];
            return sort;
        });
        // 'tinybvh-63': the high-word sort starts from buffers a (hplocGatherHigh writes them from the first sort's b).
        if (tinybvhCodes && this.sorts[0].resultInA) throw new Error('the low-word sort must end in buffers b');
        const sortedInA = this.sorts[this.sorts.length - 1].resultInA;
        const sortedKeys = sortedInA ? scratch.keysA : scratch.keysB;
        const sortedValues = sortedInA ? scratch.valuesA : scratch.valuesB;

        const resources: Record<BindingName, GPUBindingResource> = {
            positions: { buffer: this.buffers.positions },
            indices: { buffer: this.buffers.indices },
            flags: { buffer: this.buffers.flags },
            primitiveBoxes: { buffer: scratch.primitiveBoxes },
            keys: { buffer: scratch.keysA },
            values: { buffer: scratch.valuesA },
            rank: { buffer: scratch.links },
            outNodes: { buffer: this.buffers.outNodes },
            outTriangles: { buffer: this.buffers.outTriangles },
            outParents: { buffer: this.buffers.outParents },
            stats: { buffer: this.buffers.stats },
            params: { buffer: params, offset: 0, size: LBVH_PARAMS_BYTES },
            tables: { buffer: scratch.tables },
            statsView: { buffer: this.buffers.stats },
            deltas: { buffer: scratch.deltas },
            codes: { buffer: scratch.codes },
            scratch: { buffer: scratch.scratch },
            scratchView: { buffer: scratch.scratch },
            scratchFill: { buffer: scratch.scratch },
            links: { buffer: scratch.links },
            hplocParams: { buffer: params, offset: hplocParamsRecord * UNIFORM_STRIDE, size: HPLOC_PARAMS_BYTES },
            valuesIn: { buffer: scratch.valuesB },
        };
        const resourceFor = (kernel: HplocKernel, name: BindingName): GPUBindingResource => {
            if (READS_SORTED.has(kernel) && name === 'keys') return { buffer: sortedKeys };
            if (READS_SORTED.has(kernel) && name === 'values') return { buffer: sortedValues };
            return resources[name];
        };
        this.groups = Object.fromEntries(
            KERNELS.map((kernel) => [
                kernel,
                device.createBindGroup({
                    label: `${label} ${kernel}`,
                    layout: pipelines[kernel].getBindGroupLayout(0),
                    entries: KERNEL_BINDINGS[kernel].map((name) => ({ binding: BINDING[name], resource: resourceFor(kernel, name) })),
                }),
            ]),
        ) as Record<HplocKernel, GPUBindGroup>;

        this.rankScan = createScanBindGroups(
            device,
            sortPipelines.scan,
            {
                data: { buffer: scratch.links, offset: 0, size: (innerTotal + 1) * 4 },
                tileSums: { buffer: scratch.tileSums },
                params: { buffer: params, offset: rankScanRecord * UNIFORM_STRIDE, size: SCAN_PARAMS_BYTES },
                count: innerTotal + 1,
            },
            `${label} rank scan`,
        );
        this.leafRunScan = createScanBindGroups(
            device,
            sortPipelines.scan,
            {
                data: { buffer: scratch.links, offset: leafRunsOffset * 4, size: (innerTotal + 1) * 4 },
                tileSums: { buffer: scratch.tileSums },
                params: { buffer: params, offset: leafRunScanRecord * UNIFORM_STRIDE, size: SCAN_PARAMS_BYTES },
                count: innerTotal + 1,
            },
            `${label} leaf-run scan`,
        );
        // Every uniform is written once, here: they never change, so a batch can be recorded again and again.
        device.queue.writeBuffer(params, 0, lbvhParams);
        for (const sort of this.sorts) sort.writeParams(device.queue);
        device.queue.writeBuffer(params, rankScanRecord * UNIFORM_STRIDE, scanParams(innerTotal + 1));
        device.queue.writeBuffer(params, leafRunScanRecord * UNIFORM_STRIDE, scanParams(innerTotal + 1));
        device.queue.writeBuffer(params, hplocParamsRecord * UNIFORM_STRIDE, hplocParams);
    }

    /**
     * Records every mesh's build into `encoder`: copies of any GPU-buffer
     * inputs, then one compute pass. `timestampWrites` (optional) times the pass.
     *
     * For a breakdown, `stageTimestamps` instead puts each step in a pass of
     * its own and times it with the writes it returns. The extra passes cost a
     * little time themselves, so use it to compare steps, not to time the build.
     */
    encode(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites, stageTimestamps?: (stage: HplocStage) => GPUComputePassTimestampWrites): void {
        if (this.destroyed) throw new Error('@pmndrs/bvh: the batch was destroyed');
        copyBatchGeometry(encoder, this.meshes, this.buffers);
        const maxPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        const shared: GPUComputePassEncoder | null = stageTimestamps ? null : encoder.beginComputePass({ label: this.label, timestampWrites });
        const stage = (name: HplocStage, recordStage: (pass: GPUComputePassEncoder) => void): void => {
            if (shared) {
                recordStage(shared);
                return;
            }
            const pass = encoder.beginComputePass({ label: `${this.label} ${name}`, timestampWrites: stageTimestamps?.(name) });
            recordStage(pass);
            pass.end();
        };
        const run = (pass: GPUComputePassEncoder, kernel: HplocKernel, workgroups: number): void => {
            pass.setPipeline(this.pipelines[kernel]);
            pass.setBindGroup(0, this.groups[kernel]);
            dispatchWorkgroups2d(pass, workgroups, maxPerDimension);
        };
        const perMesh = Math.ceil(this.meshCount / LBVH_WORKGROUP_SIZE);
        const tinybvhCodes = this.mortonCodes === 'tinybvh-63';
        stage('boxes', (pass) => {
            run(pass, 'lbvhStart', perMesh);
            run(pass, 'lbvhBoxes', this.tileCount);
        });
        stage('morton', (pass) => run(pass, tinybvhCodes ? 'hplocMortonTinybvh' : 'lbvhMorton', this.tileCount));
        stage('sort', (pass) => {
            this.sorts[0].encode(pass);
            if (tinybvhCodes) {
                run(pass, 'hplocGatherHigh', this.tileCount);
                this.sorts[1].encode(pass);
            }
        });
        stage('deltas', (pass) => run(pass, 'hplocDeltasKernel', this.tileCount));
        stage('clear', (pass) => run(pass, 'hplocClear', this.clearWorkgroups));
        stage('climb', (pass) => run(pass, 'hplocClimb', this.tileCount));
        stage('mark', (pass) => run(pass, 'hplocMark', this.tileCount));
        stage('rank', (pass) => {
            encodeScan(pass, this.sortPipelines.scan, this.rankScan, maxPerDimension);
            encodeScan(pass, this.sortPipelines.scan, this.leafRunScan, maxPerDimension);
        });
        // The tree positions come from the emit pass, so the triangles go after it.
        stage('emit', (pass) => run(pass, 'hplocEmit', this.tileCount));
        stage('triangles', (pass) => {
            run(pass, 'hplocTrianglesKernel', this.tileCount);
            run(pass, 'hplocFinish', perMesh);
        });
        shared?.end();
    }

    /**
     * Reads every mesh's stats back (one copy and one map), checks them, and
     * returns the trees. Call after submitting an `encode`. Frees the batch's
     * scratch; the trees stay until the caller destroys them.
     */
    async finish(): Promise<GpuLbvhTrees> {
        if (this.finished || this.destroyed) throw new Error('@pmndrs/bvh: the batch was already finished');
        const device = this.device;
        const bytes = this.meshCount * LBVH_STATS_BYTES;
        const readback = device.createBuffer({ label: `${this.label} stats readback`, size: bytes, usage: USAGE.MAP_READ | USAGE.COPY_DST });
        try {
            const encoder = device.createCommandEncoder({ label: `${this.label} stats` });
            encoder.copyBufferToBuffer(this.buffers.stats, 0, readback, 0, bytes);
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(MAP_MODE_READ);
            const words = new Uint32Array(readback.getMappedRange().slice(0));
            // From here the outputs are the caller's; if the read above failed, destroy() frees them with the rest.
            this.finished = true;
            const { outNodes, outTriangles, outParents } = this.buffers;
            return {
                nodes: outNodes,
                triangles: outTriangles,
                parents: outParents,
                meshes: this.meshes.map((mesh, index) => {
                    const stats = words.subarray(index * LBVH_STATS_WORDS, (index + 1) * LBVH_STATS_WORDS);
                    return meshResult(mesh, index, stats, hplocProblems(stats));
                }),
                destroy() {
                    outNodes.destroy();
                    outTriangles.destroy();
                    outParents.destroy();
                },
            };
        } finally {
            readback.destroy();
            this.destroy();
        }
    }

    /** Frees the scratch, the inputs and the uniforms. Trees handed out by `finish` are not touched. */
    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        for (const buffer of this.owned) buffer.destroy();
        // The outputs are only handed out by finish(); before that they are the batch's.
        if (!this.finished) {
            this.buffers.outNodes.destroy();
            this.buffers.outTriangles.destroy();
            this.buffers.outParents.destroy();
        }
    }
}

/** What H-PLOC's own stats words say is wrong with a mesh's tree. */
function hplocProblems(stats: Uint32Array): string[] {
    const problems: string[] = [];
    if (stats[HPLOC_STAT.failedMerges] > 0) problems.push(`${stats[HPLOC_STAT.failedMerges]} merges found no mutual pair (NaN boxes?)`);
    if (stats[HPLOC_STAT.narrowSubgroups] > 0) problems.push(`${stats[HPLOC_STAT.narrowSubgroups]} subgroups were narrower than the device reported`);
    if (stats[HPLOC_STAT.tooDeep] > 0) problems.push('the tree is deeper than H-PLOC allows');
    return problems;
}
