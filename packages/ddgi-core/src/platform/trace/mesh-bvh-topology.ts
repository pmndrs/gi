/**
 * Mesh BVHs from a tree someone else built.
 *
 * An engine adapter may already have a good per-mesh tree (three-mesh-bvh's,
 * for example). This takes the *shape* of such a tree as plain arrays (which
 * triangle is stored where, and which node is whose child) and produces the
 * tracer's own `MeshBvh`: 64-byte two-child nodes and 48-byte triangles in tree
 * order. Nothing here knows about any engine.
 *
 * The boxes are not taken from the other tree. They are recomputed from the
 * triangles' own f32 corners, bottom-up, so every box contains its triangles
 * exactly; a box rounded a hair too small would let rays slip past. Leaves
 * larger than the tracer accepts are cut in half until they fit.
 */

import { BVH_NODE_BYTES, BVH_NODE_WORDS, DEFAULT_TRIANGLE_TEST, EMPTY_CHILD_INDEX, MAX_LEAF_PRIMITIVES, NODE_WORD, NO_PARENT, type TriangleTest } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { checkGeometry, growBoxByTriangle, packTriangles, type CheckedGeometry } from './triangle-packing.ts';
import { TOPOLOGY_EMPTY_CHILD, TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology, type TinyBvhBuilder } from '@pmndrs/bvh';

// The tree description is the BVH package's Builder contract, so any builder there fits.
export { TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology } from '@pmndrs/bvh';

export interface MeshTreeTopologyOptions {
    /** Defaults to `moller-trumbore`, like `buildMeshBvh`. */
    readonly triangleTest?: TriangleTest;
    /** Leaves with more triangles than this are cut in half until they fit. At most {@link MAX_LEAF_PRIMITIVES} (the default). */
    readonly maxLeafPrimitives?: number;
}

/** A child to emit: an inner node of the input tree, or a run of positions that becomes a leaf or gets cut. */
interface PendingChild {
    readonly outputNode: number;
    readonly depth: number;
    /** Input node index, or -1 for a run of positions. */
    readonly inputNode: number;
    readonly start: number;
    readonly count: number;
}

export function buildMeshBvhFromTopology(geometry: MeshGeometry, topology: MeshTreeTopology, options: MeshTreeTopologyOptions = {}): MeshBvh {
    const triangleTest = options.triangleTest ?? DEFAULT_TRIANGLE_TEST;
    const maxLeaf = options.maxLeafPrimitives ?? MAX_LEAF_PRIMITIVES;
    if (!Number.isInteger(maxLeaf) || maxLeaf < 1 || maxLeaf > MAX_LEAF_PRIMITIVES) {
        throw new RangeError(`maxLeafPrimitives must be an integer from 1 to ${MAX_LEAF_PRIMITIVES}, got ${maxLeaf}`);
    }
    const checked = checkGeometry(geometry);
    const triangleCount = checked.triangleCount;
    const { order, children } = topology;
    checkOrder(order, triangleCount);
    const inputNodeCount = children.length / TOPOLOGY_WORDS_PER_NODE;
    if (!Number.isInteger(inputNodeCount) || inputNodeCount < 1) {
        throw new RangeError(`children must hold ${TOPOLOGY_WORDS_PER_NODE} words per node and at least a root`);
    }

    // A binary tree whose leaves hold at least one triangle has at most (triangles - 1) inner nodes.
    const maxNodes = Math.max(1, triangleCount - 1);
    const nodeBuffer = new ArrayBuffer(maxNodes * BVH_NODE_BYTES);
    const nodeWords = new Uint32Array(nodeBuffer);
    const parents = new Uint32Array(maxNodes);
    parents[0] = NO_PARENT;
    let nodeCount = 1;
    let maxDepth = 1;
    let leafCount = 0;
    const seenInputNodes = new Uint8Array(inputNodeCount);
    const coveredPositions = new Uint8Array(triangleCount);

    const writeLeaf = (outputNode: number, side: 'left' | 'right', start: number, count: number): void => {
        if (start + count > triangleCount) throw new RangeError(`a leaf covers positions ${start} to ${start + count - 1}, but there are ${triangleCount} triangles`);
        for (let position = start; position < start + count; position++) {
            if (coveredPositions[position]) throw new RangeError(`position ${position} is in more than one leaf`);
            coveredPositions[position] = 1;
        }
        const base = outputNode * BVH_NODE_WORDS;
        nodeWords[base + (side === 'left' ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)] = start;
        nodeWords[base + (side === 'left' ? NODE_WORD.leftCount : NODE_WORD.rightCount)] = count;
        leafCount++;
    };

    // Nodes are numbered in the order they are reached (depth first, left first), so a
    // child always has a larger number than its parent; the box pass below relies on that.
    const pending: PendingChild[] = [];
    const placeChild = (outputNode: number, side: 'left' | 'right', depth: number, index: number, count: number, fromInput: boolean): void => {
        const isRun = !fromInput || count > 0;
        if (isRun && count <= maxLeaf) {
            writeLeaf(outputNode, side, index, count);
            return;
        }
        if (nodeCount >= maxNodes) throw new RangeError('the tree has more nodes than its triangles allow (a cycle, or empty leaves)');
        const child = nodeCount++;
        parents[child] = outputNode;
        const base = outputNode * BVH_NODE_WORDS;
        nodeWords[base + (side === 'left' ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)] = child;
        nodeWords[base + (side === 'left' ? NODE_WORD.leftCount : NODE_WORD.rightCount)] = 0;
        pending.push(isRun ? { outputNode: child, depth: depth + 1, inputNode: -1, start: index, count } : { outputNode: child, depth: depth + 1, inputNode: index, start: 0, count: 0 });
    };

    pending.push({ outputNode: 0, depth: 1, inputNode: 0, start: 0, count: 0 });
    while (pending.length > 0) {
        const task = pending.pop() as PendingChild;
        maxDepth = Math.max(maxDepth, task.depth);
        let left: [number, number, boolean];
        let right: [number, number, boolean] | null;
        if (task.inputNode >= 0) {
            const node = task.inputNode;
            if (node >= inputNodeCount) throw new RangeError(`node ${node} does not exist`);
            if (seenInputNodes[node]) throw new RangeError(`node ${node} is reached twice`);
            seenInputNodes[node] = 1;
            const words = node * TOPOLOGY_WORDS_PER_NODE;
            left = [children[words], children[words + 1], true];
            const rightIndex = children[words + 2];
            right = rightIndex === TOPOLOGY_EMPTY_CHILD ? null : [rightIndex, children[words + 3], true];
            if (left[0] === TOPOLOGY_EMPTY_CHILD) throw new RangeError(`node ${node} has an empty left child`);
        } else {
            // A leaf too big for the tracer: cut the run in half.
            const half = Math.floor(task.count / 2);
            left = [task.start, half, false];
            right = [task.start + half, task.count - half, false];
        }
        // Push right before left, so the left subtree is numbered first.
        const pendingBefore = pending.length;
        placeChild(task.outputNode, 'left', task.depth, left[0], left[1], left[2]);
        if (right) {
            placeChild(task.outputNode, 'right', task.depth, right[0], right[1], right[2]);
        } else {
            if (task.outputNode !== 0 || triangleCount !== 1) throw new RangeError('only the root of a one-triangle mesh may have an empty child');
            const base = task.outputNode * BVH_NODE_WORDS;
            nodeWords[base + NODE_WORD.rightIndex] = EMPTY_CHILD_INDEX;
            nodeWords[base + NODE_WORD.rightCount] = 0;
        }
        // placeChild pushed left then right; swap so left pops first.
        if (pending.length - pendingBefore === 2) {
            const last = pending.length - 1;
            [pending[last - 1], pending[last]] = [pending[last], pending[last - 1]];
        }
    }
    for (let position = 0; position < triangleCount; position++) {
        if (!coveredPositions[position]) throw new RangeError(`position ${position} is in no leaf`);
    }

    const nodeFloats = new Float32Array(nodeBuffer);
    const bounds = computeBoxes(nodeFloats, nodeWords, nodeCount, geometry.positions, checked, order);

    return {
        nodes: nodeBuffer.slice(0, nodeCount * BVH_NODE_BYTES),
        triangles: packTriangles(geometry, checked, order, triangleTest),
        parents: parents.slice(0, nodeCount),
        nodeCount,
        triangleCount,
        triangleTest,
        maxDepth,
        leafCount,
        bounds,
    };
}

function checkOrder(order: Uint32Array, triangleCount: number): void {
    if (order.length !== triangleCount) throw new RangeError(`order has ${order.length} entries for ${triangleCount} triangles`);
    const seen = new Uint8Array(triangleCount);
    for (const triangle of order) {
        if (triangle >= triangleCount || seen[triangle]) throw new RangeError('order must list every triangle exactly once');
        seen[triangle] = 1;
    }
}

/**
 * Writes every child box, last node first: a leaf's box is around its
 * triangles' corners, an inner child's box around that child's two boxes
 * (already done, since children come after their parents). Returns the root box.
 */
function computeBoxes(
    nodeFloats: Float32Array,
    nodeWords: Uint32Array,
    nodeCount: number,
    positions: Float32Array,
    checked: CheckedGeometry,
    order: Uint32Array,
): Float32Array {
    const box = new Float32Array(6);
    const resetBox = (): void => {
        box[0] = box[1] = box[2] = Infinity;
        box[3] = box[4] = box[5] = -Infinity;
    };
    const growByChild = (node: number, minWord: number, maxWord: number): void => {
        const base = node * BVH_NODE_WORDS;
        for (let axis = 0; axis < 3; axis++) {
            const low = nodeFloats[base + minWord + axis];
            const high = nodeFloats[base + maxWord + axis];
            if (low < box[axis]) box[axis] = low;
            if (high > box[3 + axis]) box[3 + axis] = high;
        }
    };
    // Per side: [index word, count word, min word, max word]. Plain numbers, so the loop allocates nothing.
    const sides = [
        [NODE_WORD.leftIndex, NODE_WORD.leftCount, NODE_WORD.leftMin, NODE_WORD.leftMax],
        [NODE_WORD.rightIndex, NODE_WORD.rightCount, NODE_WORD.rightMin, NODE_WORD.rightMax],
    ];
    for (let node = nodeCount - 1; node >= 0; node--) {
        const base = node * BVH_NODE_WORDS;
        for (let side = 0; side < 2; side++) {
            const [indexWord, countWord, minWord, maxWord] = sides[side];
            const index = nodeWords[base + indexWord];
            if (index === EMPTY_CHILD_INDEX) continue;
            const count = nodeWords[base + countWord];
            resetBox();
            if (count > 0) {
                for (let position = index; position < index + count; position++) growBoxByTriangle(box, 0, positions, checked, order[position]);
            } else {
                growByChild(index, NODE_WORD.leftMin, NODE_WORD.leftMax);
                if (nodeWords[index * BVH_NODE_WORDS + NODE_WORD.rightIndex] !== EMPTY_CHILD_INDEX) {
                    growByChild(index, NODE_WORD.rightMin, NODE_WORD.rightMax);
                }
            }
            for (let axis = 0; axis < 3; axis++) {
                nodeFloats[base + minWord + axis] = box[axis];
                nodeFloats[base + maxWord + axis] = box[3 + axis];
            }
        }
    }
    resetBox();
    growByChild(0, NODE_WORD.leftMin, NODE_WORD.leftMax);
    if (nodeWords[NODE_WORD.rightIndex] !== EMPTY_CHILD_INDEX) growByChild(0, NODE_WORD.rightMin, NODE_WORD.rightMax);
    return Float32Array.from(box);
}

/**
 * Builds a mesh BVH with tinybvh's binned SAH builder (WASM, from @pmndrs/bvh):
 * tinybvh decides the tree's shape, and {@link buildMeshBvhFromTopology} writes
 * our layout. `tinyBvh` comes from `createTinyBvhBuilder()`; one build at a time.
 */
export function buildMeshBvhWithTinyBvh(geometry: MeshGeometry, tinyBvh: TinyBvhBuilder, options: MeshTreeTopologyOptions = {}): MeshBvh {
    const { topology } = tinyBvh.build(geometry);
    return buildMeshBvhFromTopology(geometry, topology, options);
}
