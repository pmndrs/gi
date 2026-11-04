/**
 * tinybvh's default triangle builder (SIMD Bonsai), compiled to WebAssembly (stage B0 of
 * docs/design/bvh-framework.md).
 *
 * The .wasm is built from native/tinybvh-wasm.cpp by native/build-wasm.ts and
 * stored base64-encoded in tinybvh-wasm.ts, so it loads the same way in a
 * worker, on a page, in Node and in any bundler. It is single-threaded: run it
 * in a worker for big meshes.
 *
 * Two outputs from the same tree:
 * - `build`: the tree's shape (`MeshTreeTopology`, the Builder contract's plain description);
 * - `buildLayout` (stage B0b): the tracer's GPU layout, written inside the WASM.
 *   It replaces a JavaScript pass over every node and triangle, which was
 *   memory-bound and slow on big meshes (timings: PROGRESS.md, B0b).
 *
 * tinybvh (https://github.com/jbikker/tinybvh, MIT, Jacco Bikker). Commit and
 * hashes: native/build-wasm.ts and THIRD_PARTY.md. Licence texts (tinybvh's and
 * the Emscripten runtime's): licenses/, also embedded at the top of tinybvh-wasm.ts.
 */

import {
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_MAX_TRIANGLES,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_TRIANGLE_BYTES,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';
import { singleLeafTopology, TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology } from './topology.ts';
import { TINYBVH_WASM_BASE64 } from './tinybvh-wasm.ts';

/** tinybvh's node: aabbMin (3 floats), leftFirst, aabbMax (3 floats), triCount. 32 bytes. */
const TINYBVH_NODE_WORDS = 8;
const NODE_LEFT_FIRST = 3;
const NODE_TRIANGLE_COUNT = 7;
/** Floats in the mesh box the WASM writes: min x, y, z, then max x, y, z. */
const BOUNDS_FLOATS = 6;

/** The `Status` values of native/tinybvh-wasm.cpp. */
const BUILD_STATUS = {
    ok: 0,
    noTriangles: 1,
    outOfMemory: 2,
    indexOutOfRange: 3,
    nothingBuilt: 4,
    badLeafLimit: 5,
    badOrder: 6,
    badTree: 7,
    tooLarge: 8,
} as const;

/**
 * The largest byte size `tbvh_alloc` takes: its argument is a 32-bit WASM
 * integer, so a bigger size would wrap to a small one.
 */
const MAX_ALLOCATION_BYTES = 0xffffffff;

/** What `badTree` found (`TreeProblem` in native/tinybvh-wasm.cpp), read from `tbvh_error_detail()`. */
const TREE_PROBLEMS: Record<number, string> = {
    1: 'a child points past the node pool',
    2: 'a node is reached twice',
    3: 'there are more nodes than the triangles allow',
    4: 'a leaf covers positions past the last triangle',
    5: 'a position is in more than one leaf',
    6: 'a position is in no leaf',
};

interface TinyBvhExports {
    memory: WebAssembly.Memory;
    _initialize(): void;
    tbvh_alloc(bytes: number): number;
    tbvh_free(pointer: number): void;
    tbvh_build(positions: number, vertexCount: number, indices: number, triangleCount: number): number;
    tbvh_error_detail(): number;
    tbvh_node_slots(): number;
    tbvh_nodes(): number;
    tbvh_triangle_order(): number;
    tbvh_triangle_order_length(): number;
    tbvh_write_layout(triangleFlags: number, maxLeafPrimitives: number): number;
    tbvh_layout_nodes(): number;
    tbvh_layout_node_count(): number;
    tbvh_layout_triangles(): number;
    tbvh_layout_parents(): number;
    tbvh_layout_bounds(): number;
    tbvh_layout_max_depth(): number;
    tbvh_layout_leaf_count(): number;
    tbvh_release(): void;
}

/** Triangle geometry in object space; the same shape as the tracer's `MeshGeometry`. */
export interface TinyBvhInput {
    /** `x, y, z` per vertex. */
    readonly positions: Float32Array;
    /** Three vertex indices per triangle; omit for a plain triangle list. */
    readonly indices?: Uint16Array | Uint32Array | null;
    /**
     * Per-triangle bit flags, indexed by the original triangle number. Only the
     * GPU layout stores them (`buildLayout`); they default to 0.
     */
    readonly triangleFlags?: Uint32Array | null;
}

export interface TinyBvhBuild {
    readonly topology: MeshTreeTopology;
    /** Everything: the copies in and out of WASM memory, tinybvh's `Build`, and reading its tree, in ms. */
    readonly buildMs: number;
    /** tinybvh's `Build` alone, in ms. */
    readonly wasmBuildMs: number;
}

export interface TinyBvhLayoutOptions {
    /** Leaves with more triangles than this are cut in half until they fit. 1 to 8; default 8. */
    readonly maxLeafPrimitives?: number;
}

export interface TinyBvhLayoutBuild {
    readonly layout: GpuLayoutMeshBvh;
    /** Everything: the copies in and out of WASM memory, tinybvh's `Build`, and writing the layout, in ms. */
    readonly buildMs: number;
    /** tinybvh's `Build` alone, in ms. */
    readonly wasmBuildMs: number;
    /** Writing the GPU layout inside the WASM, in ms. */
    readonly layoutMs: number;
}

export interface TinyBvhBuilder {
    /** Builds one mesh's tree and returns its shape. Synchronous; one at a time. */
    build(geometry: TinyBvhInput): TinyBvhBuild;
    /**
     * Builds one mesh's tree and returns it in the tracer's GPU layout, written
     * inside the WASM. Byte for byte what ddgi-core's `buildMeshBvhFromTopology`
     * makes from `build`'s topology of the same mesh: the boxes are recomputed
     * from the triangles, not taken from tinybvh. Synchronous; one at a time.
     */
    buildLayout(geometry: TinyBvhInput, options?: TinyBvhLayoutOptions): TinyBvhLayoutBuild;
}

let compiled: Promise<WebAssembly.Module> | null = null;

function decodeBase64(text: string): Uint8Array<ArrayBuffer> {
    const binary = atob(text);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
}

/**
 * The imports a standalone Emscripten module asks for. tinybvh only writes to
 * stdout/stderr on a fatal error and then exits, so writes are collected and
 * turned into the error message.
 */
function makeImports(getMemory: () => WebAssembly.Memory, output: string[]): WebAssembly.Imports {
    const decoder = new TextDecoder();
    return {
        env: {
            emscripten_notify_memory_growth: () => {},
        },
        wasi_snapshot_preview1: {
            fd_write: (_fd: number, iovs: number, iovsLength: number, writtenPointer: number): number => {
                const view = new DataView(getMemory().buffer);
                let written = 0;
                for (let i = 0; i < iovsLength; i++) {
                    const pointer = view.getUint32(iovs + i * 8, true);
                    const length = view.getUint32(iovs + i * 8 + 4, true);
                    output.push(decoder.decode(new Uint8Array(getMemory().buffer, pointer, length)));
                    written += length;
                }
                view.setUint32(writtenPointer, written, true);
                return 0;
            },
            fd_close: () => 0,
            fd_seek: () => 70, // ESPIPE: stdout cannot seek
            proc_exit: (code: number): never => {
                throw new Error(`tinybvh stopped (exit ${code}): ${output.join('').trim() || 'no message'}`);
            },
        },
    };
}

/** Compiles and starts tinybvh. Each builder has its own WASM memory. */
export async function createTinyBvhBuilder(): Promise<TinyBvhBuilder> {
    compiled ??= WebAssembly.compile(decodeBase64(TINYBVH_WASM_BASE64));
    const output: string[] = [];
    let memory: WebAssembly.Memory | null = null;
    const instance = await WebAssembly.instantiate(await compiled, makeImports(() => memory as WebAssembly.Memory, output));
    const wasm = instance.exports as unknown as TinyBvhExports;
    memory = wasm.memory;
    wasm._initialize();
    return {
        build: (geometry) => buildTopology(wasm, geometry),
        buildLayout: (geometry, options = {}) => buildGpuLayout(wasm, geometry, options),
    };
}

/** Counts, after checking that the arrays describe whole vertices and triangles. */
function countInput(geometry: TinyBvhInput): { vertexCount: number; triangleCount: number } {
    const { positions, indices } = geometry;
    const vertexCount = positions.length / 3;
    const triangleCount = indices ? indices.length / 3 : vertexCount / 3;
    if (!Number.isInteger(vertexCount) || !Number.isInteger(triangleCount) || triangleCount < 1) {
        throw new RangeError('tinybvh needs at least one triangle, with three floats per vertex and three indices per triangle');
    }
    if (triangleCount > GPU_LAYOUT_MAX_TRIANGLES) throw new RangeError(`too many triangles for one tree: ${triangleCount}`);
    return { vertexCount, triangleCount };
}

/** Turns a failed WASM status into an exception. */
function throwOnStatus(wasm: TinyBvhExports, status: number, geometry: TinyBvhInput, vertexCount: number): void {
    if (status === BUILD_STATUS.ok) return;
    const detail = wasm.tbvh_error_detail();
    switch (status) {
        case BUILD_STATUS.indexOutOfRange:
            throw new RangeError(`index ${detail} is ${geometry.indices?.[detail] ?? detail}, but there are ${vertexCount} vertices`);
        case BUILD_STATUS.outOfMemory:
            throw new Error('tinybvh: out of WASM memory');
        case BUILD_STATUS.tooLarge:
            throw new RangeError('tinybvh: the mesh is too big for 32-bit WebAssembly (one of its arrays would be 4 GB or more)');
        case BUILD_STATUS.badOrder:
            throw new Error('tinybvh ordered its triangles wrongly: not every triangle exactly once');
        case BUILD_STATUS.badTree:
            throw new Error(`tinybvh's nodes do not form a tree over every triangle: ${TREE_PROBLEMS[detail] ?? `problem ${detail}`}`);
        default:
            throw new Error(`tinybvh failed with status ${status}`);
    }
}

/** Allocates WASM memory for an input array, refusing sizes the 32-bit allocator would wrap. */
function allocateInput(wasm: TinyBvhExports, bytes: number, what: string): number {
    if (bytes > MAX_ALLOCATION_BYTES) throw new RangeError(`tinybvh: the ${what} are too big for 32-bit WebAssembly (${bytes} bytes)`);
    const pointer = wasm.tbvh_alloc(bytes);
    if (pointer === 0) throw new Error(`tinybvh: out of WASM memory for the ${what}`);
    return pointer;
}

/**
 * Copies the input into WASM memory, has tinybvh build, runs `read` while the
 * tree is still in WASM memory, and frees everything. The copies are plain
 * typed-array `set`s; the WASM checks the indices and widens the vertices to
 * the 16 bytes tinybvh reads.
 */
function withTinyBvhTree<T>(wasm: TinyBvhExports, geometry: TinyBvhInput, read: (wasmBuildMs: number) => T): T {
    const { positions, indices } = geometry;
    const { vertexCount, triangleCount } = countInput(geometry);
    const positionPointer = allocateInput(wasm, positions.length * 4, 'positions');
    let indexPointer = 0;
    try {
        if (indices) indexPointer = allocateInput(wasm, indices.length * 4, 'indices');
        // Views are made after every allocation: growing the memory replaces its buffer.
        new Float32Array(wasm.memory.buffer, positionPointer, positions.length).set(positions);
        if (indices) new Uint32Array(wasm.memory.buffer, indexPointer, indices.length).set(indices);

        const wasmStart = performance.now();
        const status = wasm.tbvh_build(positionPointer, vertexCount, indexPointer, triangleCount);
        const wasmBuildMs = performance.now() - wasmStart;
        throwOnStatus(wasm, status, geometry, vertexCount);
        return read(wasmBuildMs);
    } finally {
        wasm.tbvh_release();
        wasm.tbvh_free(positionPointer);
        if (indexPointer) wasm.tbvh_free(indexPointer);
    }
}

function buildTopology(wasm: TinyBvhExports, geometry: TinyBvhInput): TinyBvhBuild {
    const start = performance.now();
    const { triangleCount } = countInput(geometry);
    return withTinyBvhTree(wasm, geometry, (wasmBuildMs) => {
        const nodeSlots = wasm.tbvh_node_slots();
        if (nodeSlots === 0) throw new Error('tinybvh built no tree');
        const orderLength = wasm.tbvh_triangle_order_length();
        if (orderLength !== triangleCount) throw new Error(`tinybvh ordered ${orderLength} triangles, expected ${triangleCount}`);
        const nodes = new Uint32Array(wasm.memory.buffer, wasm.tbvh_nodes(), nodeSlots * TINYBVH_NODE_WORDS);
        const order = new Uint32Array(wasm.memory.buffer, wasm.tbvh_triangle_order(), triangleCount).slice();
        const topology = topologyFromTinyBvhNodes(nodes, order);
        return { topology, buildMs: performance.now() - start, wasmBuildMs };
    });
}

function buildGpuLayout(wasm: TinyBvhExports, geometry: TinyBvhInput, options: TinyBvhLayoutOptions): TinyBvhLayoutBuild {
    const start = performance.now();
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    if (!Number.isInteger(maxLeaf) || maxLeaf < 1 || maxLeaf > GPU_LAYOUT_MAX_LEAF_PRIMITIVES) {
        throw new RangeError(`maxLeafPrimitives must be an integer from 1 to ${GPU_LAYOUT_MAX_LEAF_PRIMITIVES}, got ${maxLeaf}`);
    }
    const { triangleFlags } = geometry;
    const { vertexCount, triangleCount } = countInput(geometry);
    if (triangleFlags && triangleFlags.length < triangleCount) throw new RangeError('triangleFlags must have one entry per triangle');

    return withTinyBvhTree(wasm, geometry, (wasmBuildMs) => {
        const flagsPointer = triangleFlags ? allocateInput(wasm, triangleCount * 4, 'triangle flags') : 0;
        try {
            if (triangleFlags) new Uint32Array(wasm.memory.buffer, flagsPointer, triangleCount).set(triangleFlags.subarray(0, triangleCount));

            const layoutStart = performance.now();
            throwOnStatus(wasm, wasm.tbvh_write_layout(flagsPointer, maxLeaf), geometry, vertexCount);
            const layoutMs = performance.now() - layoutStart;

            // Copied out (`slice` copies): the WASM memory is reused by the next build.
            const memory = wasm.memory.buffer;
            const nodeCount = wasm.tbvh_layout_node_count();
            const nodesAt = wasm.tbvh_layout_nodes();
            const trianglesAt = wasm.tbvh_layout_triangles();
            const layout: GpuLayoutMeshBvh = {
                nodes: memory.slice(nodesAt, nodesAt + nodeCount * GPU_LAYOUT_NODE_BYTES),
                triangles: memory.slice(trianglesAt, trianglesAt + triangleCount * GPU_LAYOUT_TRIANGLE_BYTES),
                parents: new Uint32Array(memory, wasm.tbvh_layout_parents(), nodeCount).slice(),
                nodeCount,
                triangleCount,
                maxDepth: wasm.tbvh_layout_max_depth(),
                leafCount: wasm.tbvh_layout_leaf_count(),
                bounds: new Float32Array(memory, wasm.tbvh_layout_bounds(), BOUNDS_FLOATS).slice(),
            };
            return { layout, buildMs: performance.now() - start, wasmBuildMs, layoutMs };
        } finally {
            if (flagsPointer) wasm.tbvh_free(flagsPointer);
        }
    });
}

/**
 * tinybvh's node pool to a {@link MeshTreeTopology}. Its root is node 0 and an
 * inner node's children are `leftFirst` and `leftFirst + 1` (slot 1 is unused).
 * Inner nodes are numbered in the order a depth-first walk reaches them.
 */
export function topologyFromTinyBvhNodes(nodes: Uint32Array, order: Uint32Array): MeshTreeTopology {
    const isLeaf = (node: number) => nodes[node * TINYBVH_NODE_WORDS + NODE_TRIANGLE_COUNT] > 0;
    if (isLeaf(0)) {
        return singleLeafTopology(order, nodes[NODE_LEFT_FIRST], nodes[NODE_TRIANGLE_COUNT]);
    }
    const slots = nodes.length / TINYBVH_NODE_WORDS;
    // Inner nodes are at most half the slots; grown to the real count at the end.
    const children = new Uint32Array(Math.ceil(slots / 2) * TOPOLOGY_WORDS_PER_NODE);
    let innerCount = 1;
    // Pairs of (tinybvh node, our index), flat, with an explicit end: a big mesh has
    // millions of nodes, and a tuple per node (or shrinking the array on every pop)
    // costs more than the walk itself.
    const stack: number[] = [0, 0];
    let stackEnd = 2;
    while (stackEnd > 0) {
        stackEnd -= 2;
        const node = stack[stackEnd];
        const ours = stack[stackEnd + 1];
        const left = nodes[node * TINYBVH_NODE_WORDS + NODE_LEFT_FIRST];
        for (let side = 0; side < 2; side++) {
            const child = left + side;
            if (child >= slots) throw new RangeError(`tinybvh node ${node} points past the node pool`);
            const target = ours * TOPOLOGY_WORDS_PER_NODE + side * 2;
            if (isLeaf(child)) {
                children[target] = nodes[child * TINYBVH_NODE_WORDS + NODE_LEFT_FIRST];
                children[target + 1] = nodes[child * TINYBVH_NODE_WORDS + NODE_TRIANGLE_COUNT];
            } else {
                if (innerCount * TOPOLOGY_WORDS_PER_NODE >= children.length) throw new RangeError('tinybvh nodes do not form a tree');
                const index = innerCount++;
                children[target] = index;
                children[target + 1] = 0;
                stack[stackEnd] = child;
                stack[stackEnd + 1] = index;
                stackEnd += 2;
            }
        }
    }
    return { order, children: children.slice(0, innerCount * TOPOLOGY_WORDS_PER_NODE) };
}
