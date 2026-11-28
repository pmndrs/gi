/**
 * Per-mesh trees built by three-mesh-bvh, converted to our tracer's layout
 * (T0.10 part C experiment: "build with three-mesh-bvh, trace with ours").
 *
 * This is adapter-side code: the core never imports three or three-mesh-bvh.
 * three-mesh-bvh builds the tree; this file reads the shape of its packed tree
 * and hands it to the core as plain arrays (`buildMeshBvhFromTopology`), which
 * writes our 64-byte nodes and 48-byte triangles and recomputes
 * every box from the triangles.
 *
 * three-mesh-bvh v0.9.18 packs a tree as 32-byte nodes, depth first
 * (src/core/build/buildTree.js, MIT):
 *
 *     float32[0..5]  the node's own box (min xyz, max xyz)
 *     uint32[6]      leaf: first triangle position; inner: right child, as a node offset from this node
 *     uint32[7]      inner: split axis
 *     uint16[14]     leaf: triangle count
 *     uint16[15]     0xffff marks a leaf
 *
 * The left child of an inner node is the next node. With `indirect: true` the
 * library leaves the geometry's index alone and keeps `_indirectBuffer`, where
 * position `i` holds the original triangle, which is exactly our `order`.
 */

import { BufferAttribute, BufferGeometry } from 'three';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import {
    buildMeshBvhFromTopology,
    countTriangles,
    EMPTY_CHILD_INDEX,
    TOPOLOGY_WORDS_PER_NODE,
    type MeshBvh,
    type MeshGeometry,
    type MeshTreeTopology,
} from '../../../../packages/core/src/platform/trace/index.ts';

/** Size of one three-mesh-bvh node, in 32-bit words and 16-bit halves. */
const TMBVH_WORDS_PER_NODE = 8;
const TMBVH_HALVES_PER_NODE = 16;
const TMBVH_LEAF_FLAG = 0xffff;

/** The parts of a built `MeshBVH` the conversion reads. They are underscored (internal) in the library, so their shape is checked. */
interface PackedMeshBvh {
    _roots: ArrayBuffer[];
    _indirectBuffer: Uint32Array | Uint16Array | null;
}

/** Reads the shape of a three-mesh-bvh tree (built with `indirect: true`) as a {@link MeshTreeTopology}. */
export function topologyFromMeshBvh(bvh: MeshBVH, triangleCount: number): MeshTreeTopology {
    const packed = bvh as unknown as PackedMeshBvh;
    if (!Array.isArray(packed._roots) || packed._roots.length !== 1) {
        throw new Error(`expected one three-mesh-bvh root (a geometry without groups), got ${packed._roots?.length}`);
    }
    if (!packed._indirectBuffer || packed._indirectBuffer.length !== triangleCount) {
        throw new Error('the three-mesh-bvh tree must be built with indirect: true');
    }
    const root = packed._roots[0];
    const words = new Uint32Array(root);
    const halves = new Uint16Array(root);
    const nodeCount = root.byteLength / (TMBVH_WORDS_PER_NODE * 4);
    const isLeaf = (node: number): boolean => halves[node * TMBVH_HALVES_PER_NODE + 15] === TMBVH_LEAF_FLAG;
    const leafStart = (node: number): number => words[node * TMBVH_WORDS_PER_NODE + 6];
    const leafCount = (node: number): number => halves[node * TMBVH_HALVES_PER_NODE + 14];

    // Our numbering keeps only inner nodes; leaves become child entries of their parent.
    const innerIndex = new Int32Array(nodeCount).fill(-1);
    let innerCount = 0;
    for (let node = 0; node < nodeCount; node++) {
        if (!isLeaf(node)) innerIndex[node] = innerCount++;
    }
    const order = Uint32Array.from(packed._indirectBuffer);

    if (isLeaf(0)) {
        // The whole mesh is one leaf: our root must still be a node with two children.
        const start = leafStart(0);
        const count = leafCount(0);
        const half = Math.ceil(count / 2);
        const children =
            count === 1
                ? Uint32Array.of(start, 1, EMPTY_CHILD_INDEX, 0)
                : Uint32Array.of(start, half, start + half, count - half);
        return { order, children };
    }

    const children = new Uint32Array(innerCount * TOPOLOGY_WORDS_PER_NODE);
    const writeChild = (target: number, child: number): void => {
        if (isLeaf(child)) {
            children[target] = leafStart(child);
            children[target + 1] = leafCount(child);
        } else {
            children[target] = innerIndex[child];
            children[target + 1] = 0;
        }
    };
    for (let node = 0; node < nodeCount; node++) {
        if (isLeaf(node)) continue;
        const base = innerIndex[node] * TOPOLOGY_WORDS_PER_NODE;
        writeChild(base, node + 1);
        writeChild(base + 2, node + words[node * TMBVH_WORDS_PER_NODE + 6]);
    }
    return { order, children };
}

export interface TmbvhConvertedBuild {
    bvh: MeshBvh;
    /** `new MeshBVH(...)` alone. */
    tmbvhMs: number;
    /** Reading its tree and writing ours (nodes, boxes, triangles). */
    convertMs: number;
}

/**
 * Builds `geometry`'s tree with three-mesh-bvh (SAH, `targetLeafSize`) and
 * converts it. Leaves above our limit of 8 triangles are cut in half by the core.
 */
export function buildMeshBvhWithTmbvh(geometry: MeshGeometry, targetLeafSize: number): TmbvhConvertedBuild {
    const threeGeometry = new BufferGeometry();
    threeGeometry.setAttribute('position', new BufferAttribute(geometry.positions, 3));
    // Indirect mode leaves the index as it is, so the shared index array is safe to pass.
    if (geometry.indices) threeGeometry.setIndex(new BufferAttribute(geometry.indices, 1));

    const start = performance.now();
    const meshBvh = new MeshBVH(threeGeometry, { strategy: SAH, targetLeafSize, indirect: true, verbose: false });
    const built = performance.now();
    const topology = topologyFromMeshBvh(meshBvh, countTriangles(geometry));
    const bvh = buildMeshBvhFromTopology(geometry, topology);
    const converted = performance.now();
    return { bvh, tmbvhMs: built - start, convertMs: converted - built };
}
