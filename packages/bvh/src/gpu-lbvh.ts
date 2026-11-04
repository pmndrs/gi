/**
 * GPU LBVH builder (stage B1 of docs/design/bvh-framework.md): one tree per
 * mesh, built on the GPU in the tracer's layout (gpu-layout.ts), so the
 * tracer and everything that traces (`giTraceClosest`, `giTraceAny`, DDGI)
 * read its trees exactly as they read tinybvh's (B0). It is the stepping
 * stone to H-PLOC (B2), which reuses the Morton codes, the sort and the
 * bottom-up pass.
 *
 * **Steps**, all in one compute pass (WGSL in gpu-lbvh-wgsl.ts):
 *
 * 1. Triangle boxes, and the box around their centroids (subgroup, then
 *    workgroup, then one global atomic per workgroup).
 * 2. 30-bit Morton codes over each mesh's own centroid box, 10 bits per axis,
 *    in cubic cells (see `lbvhMorton` for why not stretched per axis).
 * 3. A stable radix sort of (code, triangle) pairs (gpu-radix-sort.ts), so equal
 *    codes stay in triangle order; the climb's comparison also breaks ties by
 *    sorted position, the "index" tie-break HIPRT uses.
 * 4. Apetrei's bottom-up build: one thread per sorted triangle climbs, the
 *    second thread to reach a node finishes it. It computes each node's box,
 *    SAH cost and height on the way, and marks a node whose triangles are
 *    cheaper as one leaf (SAH leaf collapse, at most `maxLeafPrimitives`).
 * 5. The inner nodes that stay are numbered by a scan (gpu-scan.ts), and
 *    written in the tracer's layout with their parents; triangles are written
 *    in sorted order, which is tree order.
 *
 * **Batches.** A scene has many meshes (Sponza 115), so a batch records every
 * mesh's build into one compute pass, reusing one set of scratch buffers sized
 * for the largest mesh (the dispatches run in order). One readback of a few
 * words per mesh at the end gives the counts and the checks.
 *
 * **Checks.** A build is refused (`problem` set) if a corner names a vertex
 * past the end, or if a handed-over word was still unwritten after the
 * climb's bounded wait (see gpu-lbvh-wgsl.ts). Callers then use the CPU
 * builder (mesh-builder.ts does that).
 *
 * Needs the `subgroups` feature ({@link GpuLbvhBuilder.isSupported}). Without
 * it, the declared fallback is tinybvh (B0) in a worker (DECISIONS 2026-11-04:
 * subgroups first, one fallback).
 */

import {
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_MAX_TRIANGLES,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_TRIANGLE_BYTES,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';
import {
    LBVH_INDEX_FORMAT,
    LBVH_NODE_INFO_WORDS,
    LBVH_PARAMS_BYTES,
    LBVH_SCRATCH_NODE_WORDS,
    LBVH_STAT,
    LBVH_STATS_BYTES,
    LBVH_STATS_WORDS,
    LBVH_WORKGROUP_SIZE,
    lbvhWgsl,
} from './gpu-lbvh-wgsl.ts';
import { createRadixSort, createRadixSortPipelines, sortOffsetWords, type RadixSort, type RadixSortPipelines } from './gpu-radix-sort.ts';
import { createScanBindGroups, createScanPipelines, encodeScan, SCAN_PARAMS_BYTES, scanParams, scanTileSumWords, type ScanBindGroups } from './gpu-scan.ts';
import { dispatchWorkgroups2d } from './gpu-workgroup-scan.ts';
import type { TinyBvhInput } from './tinybvh.ts';

/** Bits of the Morton codes: 10 per axis. */
export const LBVH_MORTON_BITS = 30;

/** Buffer usages and the map mode; the globals don't exist in Node, where tests import this file. */
const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;
const MAP_MODE_READ = 0x1;

/** Uniform blocks sit 256 bytes apart: the default `minUniformBufferOffsetAlignment`. */
const UNIFORM_STRIDE = 256;

/** Uniform records per mesh: the LBVH's, one per sort pass, the sort's offset scan, the node rank scan. */
const SORT_PARAMS_RECORD = 1;
function uniformRecords(sortPasses: number): { sortScan: number; rankScan: number; total: number } {
    return { sortScan: SORT_PARAMS_RECORD + sortPasses, rankScan: SORT_PARAMS_RECORD + sortPasses + 1, total: SORT_PARAMS_RECORD + sortPasses + 2 };
}

/** Geometry already on the GPU, for example a deformed vertex buffer an adapter wrote. Buffers need `STORAGE` usage. */
export interface GpuLbvhBufferGeometry {
    /** f32 vertex positions. */
    readonly positionBuffer: GPUBuffer;
    readonly vertexCount: number;
    /** Floats from one vertex to the next. Default 3 (tightly packed). */
    readonly positionStrideFloats?: number;
    /** Float index of vertex 0's x. Default 0. */
    readonly positionOffsetFloats?: number;
    /** Three indices per triangle; omit for a plain triangle list. */
    readonly indexBuffer?: GPUBuffer | null;
    readonly indexFormat?: 'uint16' | 'uint32';
    /** Index of the mesh's first index in `indexBuffer` (not bytes). Default 0. */
    readonly indexOffset?: number;
    readonly triangleCount: number;
    /** One `u32` per triangle, by original triangle number. Default 0. */
    readonly triangleFlagsBuffer?: GPUBuffer | null;
}

/** CPU arrays (uploaded for the build) or buffers already on the GPU. */
export type GpuLbvhGeometry = TinyBvhInput | GpuLbvhBufferGeometry;

export interface GpuLbvhOptions {
    /** Largest leaf the SAH collapse may make. 1 to 8; default 8, the tracer's largest leaf. */
    readonly maxLeafPrimitives?: number;
    /** SAH cost of visiting a node, relative to {@link intersectionCost}. Default 1. */
    readonly traversalCost?: number;
    /** SAH cost of testing one triangle. Default 1. */
    readonly intersectionCost?: number;
}

/**
 * A mesh's tree on the GPU, in the tracer's layout. The buffers are sized for
 * the most nodes the mesh could need (one fewer than its triangles); the first
 * `nodeCount` are the tree. They belong to the caller: `destroy()` frees them.
 */
export interface GpuLbvhTree {
    /** `nodeCount` nodes of 64 bytes from offset 0. Node 0 is the root. */
    readonly nodes: GPUBuffer;
    /** `triangleCount` triangles of 48 bytes, in tree order. */
    readonly triangles: GPUBuffer;
    /** One `u32` per node: its parent, or 0xffffffff for the root. */
    readonly parents: GPUBuffer;
    readonly nodeCount: number;
    readonly triangleCount: number;
    readonly leafCount: number;
    /** Levels of nodes on the longest path. */
    readonly maxDepth: number;
    /** Object-space box around the mesh: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly bounds: Float32Array;
    destroy(): void;
}

export interface GpuLbvhBuild {
    readonly tree: GpuLbvhTree;
    /** Why the tree must not be used, or null. A refused tree's buffers are already freed. */
    readonly problem: string | null;
}

interface MeshJob {
    readonly label: string;
    readonly triangleCount: number;
    readonly vertexCount: number;
    readonly params: GPUBuffer;
    readonly stats: GPUBuffer;
    readonly nodes: GPUBuffer;
    readonly triangles: GPUBuffer;
    readonly parents: GPUBuffer;
    /** Input buffers this batch uploaded (CPU geometry), freed when the batch is. */
    readonly uploaded: GPUBuffer[];
    readonly groups: Record<LbvhKernel, GPUBindGroup>;
    readonly sort: RadixSort;
    readonly rankScan: ScanBindGroups;
}

const KERNELS = ['lbvhStart', 'lbvhBoxes', 'lbvhMorton', 'lbvhClear', 'lbvhClimb', 'lbvhMark', 'lbvhEmit', 'lbvhTriangles'] as const;
type LbvhKernel = (typeof KERNELS)[number];

/** The bindings (gpu-lbvh-wgsl.ts) each kernel uses. Pipelines have 'auto' layouts, which hold exactly these. */
const BINDING = {
    positions: 0,
    indices: 1,
    flags: 2,
    primitiveBoxes: 3,
    keys: 4,
    values: 5,
    scratch: 6,
    state: 7,
    nodeInfo: 8,
    rank: 9,
    outNodes: 10,
    outTriangles: 11,
    outParents: 12,
    stats: 13,
    params: 14,
} as const;
type BindingName = keyof typeof BINDING;

const KERNEL_BINDINGS: Record<LbvhKernel, readonly BindingName[]> = {
    lbvhStart: ['stats'],
    lbvhBoxes: ['positions', 'indices', 'primitiveBoxes', 'stats', 'params'],
    lbvhMorton: ['primitiveBoxes', 'keys', 'values', 'stats', 'params'],
    lbvhClear: ['scratch', 'state', 'params'],
    lbvhClimb: ['primitiveBoxes', 'keys', 'values', 'scratch', 'state', 'nodeInfo', 'stats', 'params'],
    lbvhMark: ['nodeInfo', 'rank', 'params'],
    lbvhEmit: ['primitiveBoxes', 'scratch', 'nodeInfo', 'rank', 'outNodes', 'outParents', 'stats', 'params'],
    lbvhTriangles: ['positions', 'indices', 'flags', 'values', 'outTriangles', 'params'],
};

/** Scratch shared by every mesh of a batch, sized for the largest. */
interface BatchScratch {
    readonly buffers: Record<'primitiveBoxes' | 'keysA' | 'valuesA' | 'keysB' | 'valuesB' | 'sortOffsets' | 'tileSums' | 'scratch' | 'state' | 'nodeInfo' | 'rank' | 'dummy', GPUBuffer>;
}

function isBufferGeometry(geometry: GpuLbvhGeometry): geometry is GpuLbvhBufferGeometry {
    return 'positionBuffer' in geometry;
}

function triangleCountOf(geometry: GpuLbvhGeometry): number {
    if (isBufferGeometry(geometry)) return geometry.triangleCount;
    const entries = geometry.indices ? geometry.indices.length : geometry.positions.length / 3;
    return entries / 3;
}

function checkOptions(options: GpuLbvhOptions): Required<GpuLbvhOptions> {
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    if (!Number.isInteger(maxLeaf) || maxLeaf < 1 || maxLeaf > GPU_LAYOUT_MAX_LEAF_PRIMITIVES) {
        throw new RangeError(`maxLeafPrimitives must be an integer from 1 to ${GPU_LAYOUT_MAX_LEAF_PRIMITIVES}, got ${maxLeaf}`);
    }
    const traversalCost = options.traversalCost ?? 1;
    const intersectionCost = options.intersectionCost ?? 1;
    if (!(traversalCost > 0) || !(intersectionCost > 0)) throw new RangeError('the SAH costs must be positive');
    return { maxLeafPrimitives: maxLeaf, traversalCost, intersectionCost };
}

function checkGeometry(geometry: GpuLbvhGeometry, index: number): void {
    const where = `mesh ${index}`;
    if (isBufferGeometry(geometry)) {
        if (!Number.isInteger(geometry.triangleCount) || geometry.triangleCount < 1) throw new RangeError(`${where}: needs at least one triangle`);
        if (!Number.isInteger(geometry.vertexCount) || geometry.vertexCount < 1) throw new RangeError(`${where}: needs at least one vertex`);
    } else {
        const { positions, indices, triangleFlags } = geometry;
        const triangles = triangleCountOf(geometry);
        if (positions.length % 3 !== 0 || !Number.isInteger(triangles) || triangles < 1) {
            throw new RangeError(`${where}: needs at least one triangle, with three floats per vertex and three indices per triangle`);
        }
        if (triangleFlags && triangleFlags.length < triangles) throw new RangeError(`${where}: triangleFlags must have one entry per triangle`);
        if (indices && positions.length === 0) throw new RangeError(`${where}: has indices but no vertices`);
    }
    if (triangleCountOf(geometry) > GPU_LAYOUT_MAX_TRIANGLES) throw new RangeError(`${where}: too many triangles for one tree: ${triangleCountOf(geometry)}`);
}

/** Bytes of each buffer a mesh of `triangles` triangles needs, so a batch can refuse sizes the device can't bind. */
export function gpuLbvhBufferBytes(triangles: number): Record<string, number> {
    const inner = Math.max(1, triangles - 1);
    return {
        'primitive boxes': triangles * 32,
        'sort keys': triangles * 4,
        'inner-node scratch': inner * LBVH_SCRATCH_NODE_WORDS * 4,
        'inner-node state': inner * 3 * 4,
        'inner-node info': inner * LBVH_NODE_INFO_WORDS * 4,
        'output nodes': inner * GPU_LAYOUT_NODE_BYTES,
        'output triangles': triangles * GPU_LAYOUT_TRIANGLE_BYTES,
    };
}

/** Throws if the device can't hold a mesh this big; callers fall back to the CPU builder. */
export function checkGpuLbvhFits(device: GPUDevice, triangles: number): void {
    const limit = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    for (const [what, bytes] of Object.entries(gpuLbvhBufferBytes(triangles))) {
        if (bytes > limit) {
            throw new RangeError(`a GPU LBVH of ${triangles} triangles needs ${what} of ${bytes} bytes; this device binds at most ${limit}`);
        }
    }
}

/** Compiles the builder's kernels once per device and builds any number of batches with them. */
export class GpuLbvhBuilder {
    /** The builder needs subgroups; without them, callers use the declared fallback (tinybvh, B0). */
    static isSupported(device: GPUDevice): boolean {
        return device.features.has('subgroups');
    }

    private readonly device: GPUDevice;
    private readonly label: string;
    private pipelines: Record<LbvhKernel, GPUComputePipeline> | null = null;
    private sortPipelines: RadixSortPipelines | null = null;

    constructor(device: GPUDevice, options: { label?: string } = {}) {
        if (!GpuLbvhBuilder.isSupported(device)) throw new Error('@pmndrs/bvh: the GPU LBVH builder needs a device with the "subgroups" feature');
        this.device = device;
        this.label = options.label ?? 'gpu lbvh';
    }

    /** Compiles the kernels (asynchronously, so the main thread doesn't stall). */
    async prepare(): Promise<void> {
        if (this.pipelines) return;
        const device = this.device;
        const module = device.createShaderModule({ label: this.label, code: lbvhWgsl() });
        const scan = await createScanPipelines(device, this.label);
        const [sort, ...kernels] = await Promise.all([
            createRadixSortPipelines(device, scan, this.label),
            ...KERNELS.map((entryPoint) =>
                device.createComputePipelineAsync({ label: `${this.label} ${entryPoint}`, layout: 'auto', compute: { module, entryPoint } }),
            ),
        ]);
        this.sortPipelines = sort;
        this.pipelines = Object.fromEntries(KERNELS.map((name, index) => [name, kernels[index]])) as Record<LbvhKernel, GPUComputePipeline>;
    }

    get ready(): boolean {
        return this.pipelines !== null;
    }

    /** Builds one mesh's tree. Throws if the build was refused. */
    async build(geometry: GpuLbvhGeometry, options: GpuLbvhOptions = {}): Promise<GpuLbvhTree> {
        const [result] = await this.buildMany([geometry], options);
        if (result.problem) throw new Error(`@pmndrs/bvh: GPU LBVH build refused: ${result.problem}`);
        return result.tree;
    }

    /** Builds every mesh's tree in one submit. A refused mesh has `problem` set; the others are fine. */
    async buildMany(geometries: readonly GpuLbvhGeometry[], options: GpuLbvhOptions = {}): Promise<GpuLbvhBuild[]> {
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

    /**
     * Uploads the inputs and sets up every buffer and bind group of a batch, so
     * it can be recorded with `encode` (any number of times: each recording
     * rebuilds the same trees) and read with `finish`.
     */
    createBatch(geometries: readonly GpuLbvhGeometry[], options: GpuLbvhOptions = {}): GpuLbvhBatch {
        if (!this.pipelines || !this.sortPipelines) throw new Error('@pmndrs/bvh: call prepare() before building');
        const settings = checkOptions(options);
        geometries.forEach((geometry, index) => {
            checkGeometry(geometry, index);
            checkGpuLbvhFits(this.device, triangleCountOf(geometry));
        });
        return new GpuLbvhBatch(this.device, this.label, this.pipelines, this.sortPipelines, geometries, settings);
    }

    /** Copies a tree back into CPU memory, in exactly the shape tinybvh's `buildLayout` returns. */
    async readLayout(tree: GpuLbvhTree): Promise<GpuLayoutMeshBvh> {
        const device = this.device;
        const nodeBytes = tree.nodeCount * GPU_LAYOUT_NODE_BYTES;
        const triangleBytes = tree.triangleCount * GPU_LAYOUT_TRIANGLE_BYTES;
        const parentBytes = tree.nodeCount * 4;
        const readback = device.createBuffer({ label: `${this.label} readback`, size: nodeBytes + triangleBytes + parentBytes, usage: USAGE.MAP_READ | USAGE.COPY_DST });
        try {
            const encoder = device.createCommandEncoder({ label: `${this.label} readback` });
            encoder.copyBufferToBuffer(tree.nodes, 0, readback, 0, nodeBytes);
            encoder.copyBufferToBuffer(tree.triangles, 0, readback, nodeBytes, triangleBytes);
            encoder.copyBufferToBuffer(tree.parents, 0, readback, nodeBytes + triangleBytes, parentBytes);
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(MAP_MODE_READ);
            const bytes = readback.getMappedRange();
            return {
                nodes: bytes.slice(0, nodeBytes),
                triangles: bytes.slice(nodeBytes, nodeBytes + triangleBytes),
                parents: new Uint32Array(bytes.slice(nodeBytes + triangleBytes)),
                nodeCount: tree.nodeCount,
                triangleCount: tree.triangleCount,
                maxDepth: tree.maxDepth,
                leafCount: tree.leafCount,
                bounds: tree.bounds.slice(),
            };
        } finally {
            readback.destroy();
        }
    }
}

/** One batch of mesh builds: inputs, scratch and outputs, recorded into one compute pass. */
export class GpuLbvhBatch {
    readonly triangleCounts: readonly number[];
    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly pipelines: Record<LbvhKernel, GPUComputePipeline>;
    private readonly sortPipelines: RadixSortPipelines;
    private readonly scratch: BatchScratch;
    private readonly jobs: MeshJob[];
    private finished = false;
    private destroyed = false;

    constructor(
        device: GPUDevice,
        label: string,
        pipelines: Record<LbvhKernel, GPUComputePipeline>,
        sortPipelines: RadixSortPipelines,
        geometries: readonly GpuLbvhGeometry[],
        options: Required<GpuLbvhOptions>,
    ) {
        this.device = device;
        this.label = label;
        this.pipelines = pipelines;
        this.sortPipelines = sortPipelines;
        this.triangleCounts = geometries.map(triangleCountOf);
        const largest = Math.max(1, ...this.triangleCounts);
        this.scratch = this.createScratch(largest);
        this.jobs = geometries.map((geometry, index) => this.createJob(geometry, index, options));
    }

    /**
     * Records every mesh's build into one compute pass on `encoder`.
     * `timestampWrites` (optional) times the whole pass.
     */
    encode(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void {
        if (this.destroyed) throw new Error('@pmndrs/bvh: the batch was destroyed');
        const maxPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        const pass = encoder.beginComputePass({ label: this.label, timestampWrites });
        const run = (job: MeshJob, kernel: LbvhKernel, threads: number): void => {
            if (threads <= 0) return;
            pass.setPipeline(this.pipelines[kernel]);
            pass.setBindGroup(0, job.groups[kernel]);
            dispatchWorkgroups2d(pass, Math.ceil(threads / LBVH_WORKGROUP_SIZE), maxPerDimension);
        };
        for (const job of this.jobs) {
            const triangles = job.triangleCount;
            const inner = triangles - 1;
            pass.setPipeline(this.pipelines.lbvhStart);
            pass.setBindGroup(0, job.groups.lbvhStart);
            pass.dispatchWorkgroups(1);
            run(job, 'lbvhBoxes', triangles);
            run(job, 'lbvhMorton', triangles);
            job.sort.encode(pass);
            run(job, 'lbvhClear', inner);
            run(job, 'lbvhClimb', triangles);
            run(job, 'lbvhMark', inner);
            encodeScan(pass, this.sortPipelines.scan, job.rankScan, maxPerDimension);
            run(job, 'lbvhEmit', Math.max(1, inner));
            run(job, 'lbvhTriangles', triangles);
        }
        pass.end();
    }

    /**
     * Reads every mesh's stats back (one copy and one map), checks them, and
     * returns the trees. Call after submitting an `encode`. Frees the batch's
     * scratch; the trees stay until the caller destroys them.
     */
    async finish(): Promise<GpuLbvhBuild[]> {
        if (this.finished || this.destroyed) throw new Error('@pmndrs/bvh: the batch was already finished');
        this.finished = true;
        const device = this.device;
        const readback = device.createBuffer({ label: `${this.label} stats readback`, size: this.jobs.length * LBVH_STATS_BYTES, usage: USAGE.MAP_READ | USAGE.COPY_DST });
        try {
            const encoder = device.createCommandEncoder({ label: `${this.label} stats` });
            this.jobs.forEach((job, index) => encoder.copyBufferToBuffer(job.stats, 0, readback, index * LBVH_STATS_BYTES, LBVH_STATS_BYTES));
            device.queue.submit([encoder.finish()]);
            await readback.mapAsync(MAP_MODE_READ);
            const words = new Uint32Array(readback.getMappedRange().slice(0));
            return this.jobs.map((job, index) => this.result(job, words.subarray(index * LBVH_STATS_WORDS, (index + 1) * LBVH_STATS_WORDS)));
        } finally {
            readback.destroy();
            this.destroy();
        }
    }

    /** Frees the scratch, the uploaded inputs and the uniforms. Trees handed out by `finish` are not touched. */
    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        for (const buffer of Object.values(this.scratch.buffers)) buffer.destroy();
        for (const job of this.jobs) {
            job.params.destroy();
            job.stats.destroy();
            for (const buffer of job.uploaded) buffer.destroy();
            // Trees are only handed out by finish(); before that they are the batch's.
            if (!this.finished) {
                job.nodes.destroy();
                job.triangles.destroy();
                job.parents.destroy();
            }
        }
    }

    private result(job: MeshJob, stats: Uint32Array): GpuLbvhBuild {
        const floats = new Float32Array(stats.buffer, stats.byteOffset, stats.length);
        const bounds = floats.slice(LBVH_STAT.bounds, LBVH_STAT.bounds + 6);
        const nodeCount = stats[LBVH_STAT.nodeCount];
        const problems: string[] = [];
        if (stats[LBVH_STAT.badIndices] > 0) problems.push(`${stats[LBVH_STAT.badIndices]} triangle corners name a vertex past the last (${job.vertexCount})`);
        if (stats[LBVH_STAT.incomplete] > 0) problems.push(`${stats[LBVH_STAT.incomplete]} handed-over words were still unwritten after the wait`);
        if (stats[LBVH_STAT.rootsFinished] !== 1) problems.push(`${stats[LBVH_STAT.rootsFinished]} threads finished the root, not 1`);
        if (nodeCount < 1 || nodeCount > Math.max(1, job.triangleCount - 1)) problems.push(`the tree has ${nodeCount} nodes`);
        const tree: GpuLbvhTree = {
            nodes: job.nodes,
            triangles: job.triangles,
            parents: job.parents,
            nodeCount,
            triangleCount: job.triangleCount,
            leafCount: stats[LBVH_STAT.leafCount],
            maxDepth: stats[LBVH_STAT.maxDepth],
            bounds,
            destroy() {
                job.nodes.destroy();
                job.triangles.destroy();
                job.parents.destroy();
            },
        };
        const problem = problems.length > 0 ? `${job.label}: ${problems.join('; ')}` : null;
        if (problem) tree.destroy();
        return { tree, problem };
    }

    private createBuffer(name: string, bytes: number, usage: number): GPUBuffer {
        // Bindings must be at least 4 bytes and a multiple of 4.
        const size = Math.max(16, Math.ceil(bytes / 4) * 4);
        return this.device.createBuffer({ label: `${this.label} ${name}`, size, usage });
    }

    private createScratch(largest: number): BatchScratch {
        const storage = USAGE.STORAGE;
        const inner = Math.max(1, largest - 1);
        const tileSumWords = Math.max(scanTileSumWords(sortOffsetWords(largest)), scanTileSumWords(inner));
        return {
            buffers: {
                primitiveBoxes: this.createBuffer('primitive boxes', largest * 32, storage),
                keysA: this.createBuffer('keys a', largest * 4, storage),
                valuesA: this.createBuffer('values a', largest * 4, storage),
                keysB: this.createBuffer('keys b', largest * 4, storage),
                valuesB: this.createBuffer('values b', largest * 4, storage),
                sortOffsets: this.createBuffer('sort offsets', sortOffsetWords(largest) * 4, storage),
                tileSums: this.createBuffer('scan tile sums', tileSumWords * 4, storage),
                scratch: this.createBuffer('inner-node scratch', inner * LBVH_SCRATCH_NODE_WORDS * 4, storage),
                state: this.createBuffer('inner-node state', inner * 3 * 4, storage),
                nodeInfo: this.createBuffer('inner-node info', inner * LBVH_NODE_INFO_WORDS * 4, storage),
                rank: this.createBuffer('inner-node rank', inner * 4, storage),
                dummy: this.createBuffer('unused input', 16, storage),
            },
        };
    }

    private upload(job: { uploaded: GPUBuffer[] }, name: string, data: ArrayBufferView): GPUBuffer {
        const buffer = this.createBuffer(name, data.byteLength, USAGE.STORAGE | USAGE.COPY_DST);
        // writeBuffer needs a multiple of 4 bytes; a Uint16Array of odd length is padded by the caller.
        this.device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
        job.uploaded.push(buffer);
        return buffer;
    }

    private createJob(geometry: GpuLbvhGeometry, index: number, options: Required<GpuLbvhOptions>): MeshJob {
        const device = this.device;
        const label = `${this.label} mesh ${index}`;
        const triangleCount = triangleCountOf(geometry);
        const inner = Math.max(1, triangleCount - 1);
        const uploaded: GPUBuffer[] = [];
        const dummy = this.scratch.buffers.dummy;

        let positions: GPUBuffer;
        let indices: GPUBuffer = dummy;
        let flags: GPUBuffer = dummy;
        let vertexCount: number;
        let positionStride = 3;
        let positionOffset = 0;
        let indexFormat: number = LBVH_INDEX_FORMAT.none;
        let indexOffset = 0;
        if (isBufferGeometry(geometry)) {
            positions = geometry.positionBuffer;
            vertexCount = geometry.vertexCount;
            positionStride = geometry.positionStrideFloats ?? 3;
            positionOffset = geometry.positionOffsetFloats ?? 0;
            if (geometry.indexBuffer) {
                indices = geometry.indexBuffer;
                indexFormat = geometry.indexFormat === 'uint16' ? LBVH_INDEX_FORMAT.uint16 : LBVH_INDEX_FORMAT.uint32;
                indexOffset = geometry.indexOffset ?? 0;
            }
            if (geometry.triangleFlagsBuffer) flags = geometry.triangleFlagsBuffer;
        } else {
            const job = { uploaded };
            positions = this.upload(job, 'positions', geometry.positions);
            vertexCount = geometry.positions.length / 3;
            if (geometry.indices) {
                let data: Uint16Array | Uint32Array = geometry.indices;
                if (data instanceof Uint16Array && data.length % 2 === 1) {
                    const padded = new Uint16Array(data.length + 1);
                    padded.set(data);
                    data = padded;
                }
                indices = this.upload(job, 'indices', data);
                indexFormat = data instanceof Uint16Array ? LBVH_INDEX_FORMAT.uint16 : LBVH_INDEX_FORMAT.uint32;
            }
            if (geometry.triangleFlags) flags = this.upload(job, 'triangle flags', geometry.triangleFlags.subarray(0, triangleCount));
        }

        const sortPasses = 6;
        const records = uniformRecords(sortPasses);
        const params = this.createBuffer(`${label} params`, records.total * UNIFORM_STRIDE, USAGE.UNIFORM | USAGE.COPY_DST);
        const paramsData = new ArrayBuffer(LBVH_PARAMS_BYTES);
        const paramWords = new Uint32Array(paramsData);
        const paramFloats = new Float32Array(paramsData);
        paramWords.set([triangleCount, vertexCount, positionStride, positionOffset, indexFormat, indexOffset, flags === dummy ? 0 : 1, options.maxLeafPrimitives]);
        paramFloats[8] = options.traversalCost;
        paramFloats[9] = options.intersectionCost;

        const stats = this.createBuffer(`${label} stats`, LBVH_STATS_BYTES, USAGE.STORAGE | USAGE.COPY_SRC);
        const nodes = this.createBuffer(`${label} nodes`, inner * GPU_LAYOUT_NODE_BYTES, USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST);
        const triangles = this.createBuffer(`${label} triangles`, triangleCount * GPU_LAYOUT_TRIANGLE_BYTES, USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST);
        const parents = this.createBuffer(`${label} parents`, inner * 4, USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST);

        const scratch = this.scratch.buffers;
        const resources: Record<BindingName, GPUBindingResource> = {
            positions: { buffer: positions },
            indices: { buffer: indices },
            flags: { buffer: flags },
            primitiveBoxes: { buffer: scratch.primitiveBoxes },
            keys: { buffer: scratch.keysA },
            values: { buffer: scratch.valuesA },
            scratch: { buffer: scratch.scratch },
            state: { buffer: scratch.state },
            nodeInfo: { buffer: scratch.nodeInfo },
            rank: { buffer: scratch.rank },
            outNodes: { buffer: nodes },
            outTriangles: { buffer: triangles },
            outParents: { buffer: parents },
            stats: { buffer: stats },
            params: { buffer: params, offset: 0, size: LBVH_PARAMS_BYTES },
        };
        const groups = Object.fromEntries(
            KERNELS.map((kernel) => [
                kernel,
                device.createBindGroup({
                    label: `${label} ${kernel}`,
                    layout: this.pipelines[kernel].getBindGroupLayout(0),
                    entries: KERNEL_BINDINGS[kernel].map((name) => ({ binding: BINDING[name], resource: resources[name] })),
                }),
            ]),
        ) as Record<LbvhKernel, GPUBindGroup>;

        const sort = createRadixSort(
            device,
            this.sortPipelines,
            {
                keysA: scratch.keysA,
                valuesA: scratch.valuesA,
                keysB: scratch.keysB,
                valuesB: scratch.valuesB,
                offsets: scratch.sortOffsets,
                tileSums: scratch.tileSums,
                params,
                paramsOffset: SORT_PARAMS_RECORD * UNIFORM_STRIDE,
                paramsStride: UNIFORM_STRIDE,
            },
            triangleCount,
            LBVH_MORTON_BITS,
            `${label} sort`,
        );
        if (sort.passes !== sortPasses) throw new Error(`the sort takes ${sort.passes} passes, the uniforms have room for ${sortPasses}`);
        if (records.sortScan * UNIFORM_STRIDE !== SORT_PARAMS_RECORD * UNIFORM_STRIDE + sortPasses * UNIFORM_STRIDE) {
            throw new Error('the sort keeps its offsets scan right after its passes');
        }

        // Every uniform is written once, here: they never change, so a batch can be recorded again and again.
        const rankOffset = records.rankScan * UNIFORM_STRIDE;
        const rankCount = triangleCount - 1;
        device.queue.writeBuffer(params, 0, paramsData);
        sort.writeParams(device.queue);
        device.queue.writeBuffer(params, rankOffset, scanParams(rankCount));
        const rankScan = createScanBindGroups(
            device,
            this.sortPipelines.scan,
            {
                data: { buffer: scratch.rank },
                tileSums: { buffer: scratch.tileSums },
                params: { buffer: params, offset: rankOffset, size: SCAN_PARAMS_BYTES },
                count: rankCount,
            },
            `${label} rank scan`,
        );

        return { label, triangleCount, vertexCount, params, stats, nodes, triangles, parents, uploaded, groups, sort, rankScan };
    }
}
