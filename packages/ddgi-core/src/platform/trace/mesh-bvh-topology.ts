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
 *
 * Big meshes have millions of nodes, so both passes avoid allocating per node:
 * on Sponza, objects and tuples per node cost more than the work itself.
 */

import { BVH_NODE_BYTES, BVH_NODE_WORDS, EMPTY_CHILD_INDEX, MAX_LEAF_PRIMITIVES, NODE_WORD, NO_PARENT, TRIANGLE_WORD, TRIANGLE_WORDS } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { checkGeometry, packTriangles } from './triangle-packing.ts';
import { TOPOLOGY_EMPTY_CHILD, TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology, type TinyBvhBuilder } from '@pmndrs/bvh';

// The tree description is the BVH package's Builder contract, so any builder there fits.
export { TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology } from '@pmndrs/bvh';

export interface MeshTreeTopologyOptions {
    /** Leaves with more triangles than this are cut in half until they fit. At most {@link MAX_LEAF_PRIMITIVES} (the default). */
    readonly maxLeafPrimitives?: number;
}

/**
 * The walk's to-do list holds one entry per child still to emit, as this many
 * numbers in a flat array: output node, depth, input node, start, count. The
 * input node is an inner node of the input tree, or NOT_A_NODE for a run of
 * positions (start, count) that becomes a leaf or gets cut.
 */
const PENDING_FIELDS = 5;
const NOT_A_NODE = -1;

/** A triangle record keeps its three corners evenly spaced, this many words apart. */
const CORNER_STRIDE = TRIANGLE_WORD.vertex1 - TRIANGLE_WORD.vertex0;
if (TRIANGLE_WORD.vertex2 - TRIANGLE_WORD.vertex1 !== CORNER_STRIDE) throw new Error('triangle corners must be evenly spaced');

export function buildMeshBvhFromTopology(geometry: MeshGeometry, topology: MeshTreeTopology, options: MeshTreeTopologyOptions = {}): MeshBvh {
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

    /**
     * Emits one child of `outputNode`, on the side whose index and count words
     * are given. A run that fits becomes a leaf, and NOT_A_NODE is returned.
     * Anything else gets the next node number, which is returned for the caller to queue.
     */
    const placeChild = (outputNode: number, indexWord: number, countWord: number, index: number, count: number, fromInput: boolean): number => {
        const base = outputNode * BVH_NODE_WORDS;
        const isRun = !fromInput || count > 0;
        if (isRun && count <= maxLeaf) {
            if (index + count > triangleCount) throw new RangeError(`a leaf covers positions ${index} to ${index + count - 1}, but there are ${triangleCount} triangles`);
            for (let position = index; position < index + count; position++) {
                if (coveredPositions[position]) throw new RangeError(`position ${position} is in more than one leaf`);
                coveredPositions[position] = 1;
            }
            nodeWords[base + indexWord] = index;
            nodeWords[base + countWord] = count;
            leafCount++;
            return NOT_A_NODE;
        }
        if (nodeCount >= maxNodes) throw new RangeError('the tree has more nodes than its triangles allow (a cycle, or empty leaves)');
        const child = nodeCount++;
        parents[child] = outputNode;
        nodeWords[base + indexWord] = child;
        nodeWords[base + countWord] = 0;
        return child;
    };

    // Nodes are numbered in the order they are reached (depth first, left first), so a
    // child always has a larger number than its parent; the box pass below relies on that.
    // `pendingEnd` marks the end of the live entries; the array itself only grows
    // (shrinking it on every pop would make the engine resize it over and over).
    const pending: number[] = [0, 1, 0, 0, 0];
    let pendingEnd = PENDING_FIELDS;
    const queue = (outputNode: number, depth: number, index: number, count: number, fromInput: boolean): void => {
        const isInputInnerNode = fromInput && count === 0;
        pending[pendingEnd] = outputNode;
        pending[pendingEnd + 1] = depth;
        pending[pendingEnd + 2] = isInputInnerNode ? index : NOT_A_NODE;
        pending[pendingEnd + 3] = index;
        pending[pendingEnd + 4] = count;
        pendingEnd += PENDING_FIELDS;
    };
    while (pendingEnd > 0) {
        pendingEnd -= PENDING_FIELDS;
        const outputNode = pending[pendingEnd];
        const depth = pending[pendingEnd + 1];
        const inputNode = pending[pendingEnd + 2];
        const start = pending[pendingEnd + 3];
        const count = pending[pendingEnd + 4];
        if (depth > maxDepth) maxDepth = depth;

        let leftIndex: number;
        let leftCount: number;
        let rightIndex: number;
        let rightCount: number;
        const fromInput = inputNode !== NOT_A_NODE;
        if (fromInput) {
            if (inputNode >= inputNodeCount) throw new RangeError(`node ${inputNode} does not exist`);
            if (seenInputNodes[inputNode]) throw new RangeError(`node ${inputNode} is reached twice`);
            seenInputNodes[inputNode] = 1;
            const words = inputNode * TOPOLOGY_WORDS_PER_NODE;
            leftIndex = children[words];
            leftCount = children[words + 1];
            rightIndex = children[words + 2];
            rightCount = children[words + 3];
            if (leftIndex === TOPOLOGY_EMPTY_CHILD) throw new RangeError(`node ${inputNode} has an empty left child`);
        } else {
            // A leaf too big for the tracer: cut the run in half.
            const half = Math.floor(count / 2);
            leftIndex = start;
            leftCount = half;
            rightIndex = start + half;
            rightCount = count - half;
        }

        const leftChild = placeChild(outputNode, NODE_WORD.leftIndex, NODE_WORD.leftCount, leftIndex, leftCount, fromInput);
        let rightChild = NOT_A_NODE;
        if (fromInput && rightIndex === TOPOLOGY_EMPTY_CHILD) {
            if (outputNode !== 0 || triangleCount !== 1) throw new RangeError('only the root of a one-triangle mesh may have an empty child');
            nodeWords[outputNode * BVH_NODE_WORDS + NODE_WORD.rightIndex] = EMPTY_CHILD_INDEX;
            nodeWords[outputNode * BVH_NODE_WORDS + NODE_WORD.rightCount] = 0;
        } else {
            rightChild = placeChild(outputNode, NODE_WORD.rightIndex, NODE_WORD.rightCount, rightIndex, rightCount, fromInput);
        }
        // Right is queued first, so the left subtree is walked (and numbered) first.
        if (rightChild !== NOT_A_NODE) queue(rightChild, depth + 1, rightIndex, rightCount, fromInput);
        if (leftChild !== NOT_A_NODE) queue(leftChild, depth + 1, leftIndex, leftCount, fromInput);
    }
    for (let position = 0; position < triangleCount; position++) {
        if (!coveredPositions[position]) throw new RangeError(`position ${position} is in no leaf`);
    }

    const triangles = packTriangles(geometry, checked, order);
    const bounds = computeBoxes(new Float32Array(nodeBuffer), nodeWords, nodeCount, new Float32Array(triangles));

    return {
        nodes: nodeBuffer.slice(0, nodeCount * BVH_NODE_BYTES),
        triangles,
        parents: parents.slice(0, nodeCount),
        nodeCount,
        triangleCount,
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
 *
 * Leaf corners are read from the packed triangles (`triangleFloats`, in tree
 * order), which hold the input's f32 corners unchanged. A leaf's triangles sit
 * side by side there, which is much faster than looking each one up through the
 * order and the index buffer. Taking the min and max of f32 values is exact, so
 * the boxes come out the same either way.
 */
function computeBoxes(nodeFloats: Float32Array, nodeWords: Uint32Array, nodeCount: number, triangleFloats: Float32Array): Float32Array {
    for (let node = nodeCount - 1; node >= 0; node--) {
        const base = node * BVH_NODE_WORDS;
        writeChildBox(nodeFloats, nodeWords, triangleFloats, base, NODE_WORD.leftIndex, NODE_WORD.leftCount, NODE_WORD.leftMin, NODE_WORD.leftMax);
        if (nodeWords[base + NODE_WORD.rightIndex] !== EMPTY_CHILD_INDEX) {
            writeChildBox(nodeFloats, nodeWords, triangleFloats, base, NODE_WORD.rightIndex, NODE_WORD.rightCount, NODE_WORD.rightMin, NODE_WORD.rightMax);
        }
    }
    const root = new Float32Array(6);
    writeUnionOfChildBoxes(nodeFloats, nodeWords, 0, root, 0, 3);
    return root;
}

/**
 * Writes the box around `node`'s child boxes into `target` (min at `minAt`, max
 * at `maxAt`). Compares the same way as the leaf loop, starting from an empty
 * box, so the result is bit-for-bit what growing a box child by child gives.
 */
function writeUnionOfChildBoxes(nodeFloats: Float32Array, nodeWords: Uint32Array, node: number, target: Float32Array, minAt: number, maxAt: number): void {
    const base = node * BVH_NODE_WORDS;
    // Only the root of a one-triangle mesh has no right child.
    const hasRight = nodeWords[base + NODE_WORD.rightIndex] !== EMPTY_CHILD_INDEX;
    for (let axis = 0; axis < 3; axis++) {
        let low = Infinity;
        let high = -Infinity;
        const leftLow = nodeFloats[base + NODE_WORD.leftMin + axis];
        const leftHigh = nodeFloats[base + NODE_WORD.leftMax + axis];
        if (leftLow < low) low = leftLow;
        if (leftHigh > high) high = leftHigh;
        if (hasRight) {
            const rightLow = nodeFloats[base + NODE_WORD.rightMin + axis];
            const rightHigh = nodeFloats[base + NODE_WORD.rightMax + axis];
            if (rightLow < low) low = rightLow;
            if (rightHigh > high) high = rightHigh;
        }
        target[minAt + axis] = low;
        target[maxAt + axis] = high;
    }
}

/**
 * Computes and stores the box of one child of the node at word `base`, on the
 * side whose words are given. A leaf's box uses plain local variables, not a
 * scratch array: this runs millions of times on a big mesh.
 */
function writeChildBox(
    nodeFloats: Float32Array,
    nodeWords: Uint32Array,
    triangleFloats: Float32Array,
    base: number,
    indexWord: number,
    countWord: number,
    minWord: number,
    maxWord: number,
): void {
    const index = nodeWords[base + indexWord];
    const count = nodeWords[base + countWord];
    if (count === 0) {
        // An inner child: the box around its own two child boxes (written already).
        writeUnionOfChildBoxes(nodeFloats, nodeWords, index, nodeFloats, base + minWord, base + maxWord);
        return;
    }
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    const end = (index + count) * TRIANGLE_WORDS;
    for (let triangle = index * TRIANGLE_WORDS; triangle < end; triangle += TRIANGLE_WORDS) {
        for (let at = triangle + TRIANGLE_WORD.vertex0; at <= triangle + TRIANGLE_WORD.vertex2; at += CORNER_STRIDE) {
            const x = triangleFloats[at];
            const y = triangleFloats[at + 1];
            const z = triangleFloats[at + 2];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            if (z < minZ) minZ = z;
            if (z > maxZ) maxZ = z;
        }
    }
    nodeFloats[base + minWord] = minX;
    nodeFloats[base + minWord + 1] = minY;
    nodeFloats[base + minWord + 2] = minZ;
    nodeFloats[base + maxWord] = maxX;
    nodeFloats[base + maxWord + 1] = maxY;
    nodeFloats[base + maxWord + 2] = maxZ;
}

/**
 * Builds a mesh BVH with tinybvh's builder (WASM, from @pmndrs/bvh). The WASM
 * also writes our layout (stage B0b), byte for byte what
 * {@link buildMeshBvhFromTopology} makes from tinybvh's tree
 * (`mesh-bvh-tinybvh-layout.test.ts`), without the JavaScript pass over every
 * node and triangle (about 0.65 s on Sponza). `tinyBvh` comes from
 * `createTinyBvhBuilder()`; one build at a time.
 */
export function buildMeshBvhWithTinyBvh(geometry: MeshGeometry, tinyBvh: TinyBvhBuilder, options: MeshTreeTopologyOptions = {}): MeshBvh {
    return tinyBvh.buildLayout(geometry, options).layout;
}
