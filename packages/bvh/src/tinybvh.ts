/**
 * tinybvh's default triangle builder (SIMD Bonsai), compiled to WebAssembly (stage B0 of
 * docs/design/bvh-framework.md).
 *
 * The .wasm is built from native/tinybvh-wasm.cpp by native/build-wasm.ts and
 * stored base64-encoded in tinybvh-wasm.ts, so it loads the same way in a
 * worker, on a page, in Node and in any bundler. It is single-threaded: run it
 * in a worker for big meshes.
 *
 * tinybvh (https://github.com/jbikker/tinybvh, MIT, Jacco Bikker). Commit and
 * hashes: native/build-wasm.ts and THIRD_PARTY.md. Licence texts (tinybvh's and
 * the Emscripten runtime's): licenses/, also embedded at the top of tinybvh-wasm.ts.
 */

import { singleLeafTopology, TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology } from './topology.ts';
import { TINYBVH_WASM_BASE64 } from './tinybvh-wasm.ts';

/** tinybvh's node: aabbMin (3 floats), leftFirst, aabbMax (3 floats), triCount. 32 bytes. */
const TINYBVH_NODE_WORDS = 8;
const NODE_LEFT_FIRST = 3;
const NODE_TRIANGLE_COUNT = 7;
/** tinybvh reads vertices as 16-byte vectors. */
const FLOATS_PER_INPUT_VERTEX = 4;

interface TinyBvhExports {
    memory: WebAssembly.Memory;
    _initialize(): void;
    tbvh_build(vertices: number, vertexCount: number, indices: number, triangleCount: number): number;
    tbvh_nodes(): number;
    tbvh_triangle_order(): number;
    tbvh_triangle_order_length(): number;
    tbvh_release(): void;
    tbvh_alloc(bytes: number): number;
    tbvh_free(pointer: number): void;
}

/** Triangle geometry in object space; the same shape as the tracer's `MeshGeometry`. */
export interface TinyBvhInput {
    /** `x, y, z` per vertex. */
    readonly positions: Float32Array;
    /** Three vertex indices per triangle; omit for a plain triangle list. */
    readonly indices?: Uint16Array | Uint32Array | null;
}

export interface TinyBvhBuild {
    readonly topology: MeshTreeTopology;
    /** Everything: the copies in and out of WASM memory, tinybvh's `Build`, and reading its tree, in ms. */
    readonly buildMs: number;
    /** tinybvh's `Build` alone, in ms. */
    readonly wasmBuildMs: number;
}

export interface TinyBvhBuilder {
    /** Builds one mesh's tree. Synchronous; one at a time. */
    build(geometry: TinyBvhInput): TinyBvhBuild;
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
    return { build: (geometry) => buildWithTinyBvh(wasm, geometry) };
}

function buildWithTinyBvh(wasm: TinyBvhExports, geometry: TinyBvhInput): TinyBvhBuild {
    const { positions, indices } = geometry;
    const vertexCount = positions.length / 3;
    const triangleCount = indices ? indices.length / 3 : vertexCount / 3;
    if (!Number.isInteger(vertexCount) || !Number.isInteger(triangleCount) || triangleCount < 1) {
        throw new RangeError('tinybvh needs at least one triangle, with three floats per vertex and three indices per triangle');
    }
    const start = performance.now();
    const vertexPointer = wasm.tbvh_alloc(vertexCount * FLOATS_PER_INPUT_VERTEX * 4);
    const indexPointer = indices ? wasm.tbvh_alloc(indices.length * 4) : 0;
    try {
        if (vertexPointer === 0 || (indices && indexPointer === 0)) throw new Error('tinybvh: out of WASM memory for the input');
        // Views are made after every allocation: growing the memory replaces its buffer.
        const vertices = new Float32Array(wasm.memory.buffer, vertexPointer, vertexCount * FLOATS_PER_INPUT_VERTEX);
        for (let vertex = 0; vertex < vertexCount; vertex++) {
            vertices[vertex * 4] = positions[vertex * 3];
            vertices[vertex * 4 + 1] = positions[vertex * 3 + 1];
            vertices[vertex * 4 + 2] = positions[vertex * 3 + 2];
            vertices[vertex * 4 + 3] = 0;
        }
        if (indices) {
            for (let i = 0; i < indices.length; i++) {
                if (indices[i] >= vertexCount) throw new RangeError(`index ${i} is ${indices[i]}, but there are ${vertexCount} vertices`);
            }
            new Uint32Array(wasm.memory.buffer, indexPointer, indices.length).set(indices);
        }

        const wasmStart = performance.now();
        const nodeSlots = wasm.tbvh_build(vertexPointer, vertexCount, indexPointer, triangleCount);
        const wasmBuildMs = performance.now() - wasmStart;
        if (nodeSlots === 0) throw new Error('tinybvh built no tree');
        const orderLength = wasm.tbvh_triangle_order_length();
        if (orderLength !== triangleCount) throw new Error(`tinybvh ordered ${orderLength} triangles, expected ${triangleCount}`);
        const nodes = new Uint32Array(wasm.memory.buffer, wasm.tbvh_nodes(), nodeSlots * TINYBVH_NODE_WORDS);
        const order = new Uint32Array(wasm.memory.buffer, wasm.tbvh_triangle_order(), triangleCount).slice();
        const topology = topologyFromTinyBvhNodes(nodes, order);
        return { topology, buildMs: performance.now() - start, wasmBuildMs };
    } finally {
        wasm.tbvh_release();
        wasm.tbvh_free(vertexPointer);
        if (indexPointer) wasm.tbvh_free(indexPointer);
    }
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
