/**
 * CPU twins of the GPU refit (gpu-refit.ts), for trees in the tracer's GPU
 * layout (gpu-layout.ts):
 *
 * - {@link leafSlotsOf}: which child slots are leaves, in primitive order. The
 *   GPU refit starts one thread per leaf slot.
 * - {@link relativeSahCost}: the tree-quality signal a refit is judged by.
 * - {@link refitOnCpu}: the same bottom-up refit the GPU does, written plainly.
 *   Box unions are exact (min and max of f32 values), so a correct GPU refit
 *   matches it bit for bit; the GPU test pages compare the two.
 *
 * A child slot is `node * 2 + side` (side 0 left, 1 right).
 */

import {
    GPU_LAYOUT_CHILD_STRIDE_WORDS,
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
} from './gpu-layout.ts';

/** Largest finite f32. An empty box is stored as a point here, which no ray can reach (see gpu-refit.ts). */
export const BVH_FLOAT_MAX = 3.4028234663852886e38;

/** Floats per primitive box in the refit's input: `(minX, minY, minZ, unused, maxX, maxY, maxZ, unused)`. */
export const PRIMITIVE_BOX_FLOATS = 8;

/** The child slot's `index` word. */
function indexWord(side: number): number {
    return side === 0 ? GPU_LAYOUT_NODE_WORD.leftIndex : GPU_LAYOUT_NODE_WORD.rightIndex;
}

/** The child slot's `count` word. */
function countWord(side: number): number {
    return side === 0 ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount;
}

/**
 * Every child slot that is a leaf, as `node * 2 + side`, ordered by the
 * leaf's first primitive. Builders store primitives in tree order, so
 * neighbouring entries are often siblings, which the GPU refit's subgroup
 * step pairs up.
 */
export function leafSlotsOf(words: Uint32Array, nodeWordOffset: number, nodeCount: number): Uint32Array {
    const leaves: { first: number; slot: number }[] = [];
    for (let node = 0; node < nodeCount; node++) {
        const base = nodeWordOffset + node * GPU_LAYOUT_NODE_WORDS;
        for (let side = 0; side < 2; side++) {
            const index = words[base + indexWord(side)];
            if (index === GPU_LAYOUT_EMPTY_CHILD) continue;
            if (words[base + countWord(side)] > 0) leaves.push({ first: index, slot: node * 2 + side });
        }
    }
    leaves.sort((a, b) => a.first - b.first);
    return Uint32Array.from(leaves, (leaf) => leaf.slot);
}

/**
 * Converts a parent array (parent node per node, as builders write it) into
 * parent slots, which say which side of the parent each node hangs on.
 */
export function parentSlotsOf(words: Uint32Array, nodeWordOffset: number, nodeCount: number, parents: Uint32Array): Uint32Array {
    const slots = new Uint32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) {
        const parent = parents[node];
        if (parent === GPU_LAYOUT_NO_PARENT) {
            slots[node] = GPU_LAYOUT_NO_PARENT;
            continue;
        }
        const base = nodeWordOffset + parent * GPU_LAYOUT_NODE_WORDS;
        const isLeft = words[base + GPU_LAYOUT_NODE_WORD.leftIndex] === node && words[base + GPU_LAYOUT_NODE_WORD.leftCount] === 0;
        slots[node] = parent * 2 + (isLeft ? 0 : 1);
    }
    return slots;
}

/**
 * Nodes grouped by depth, deepest level first, for a refit that goes level by
 * level (the GPU refit's repair pass): every node comes after all its
 * descendants. `levelStarts[level]` to `levelStarts[level + 1]` index into
 * `levelNodes`; there are `levelStarts.length - 1` levels.
 */
export function levelOrderOf(parentSlots: Uint32Array, nodeCount: number): { levelStarts: Uint32Array; levelNodes: Uint32Array } {
    const depth = new Int32Array(nodeCount).fill(-1);
    const depthOf = (node: number): number => {
        // Walk up to a node of known depth, then fill the path back down (no recursion: trees can be deep).
        const path: number[] = [];
        let at = node;
        while (at !== GPU_LAYOUT_NO_PARENT && depth[at] < 0) {
            path.push(at);
            const parentSlot = parentSlots[at];
            at = parentSlot === GPU_LAYOUT_NO_PARENT ? GPU_LAYOUT_NO_PARENT : parentSlot >> 1;
        }
        let level = at === GPU_LAYOUT_NO_PARENT ? -1 : depth[at];
        for (let index = path.length - 1; index >= 0; index--) depth[path[index]] = ++level;
        return depth[node];
    };
    let deepest = -1;
    for (let node = 0; node < nodeCount; node++) deepest = Math.max(deepest, depthOf(node));
    const levelCount = deepest + 1;
    // Level 0 of the output is the deepest depth.
    const counts = new Uint32Array(levelCount);
    for (let node = 0; node < nodeCount; node++) counts[deepest - depth[node]]++;
    const levelStarts = new Uint32Array(levelCount + 1);
    for (let level = 0; level < levelCount; level++) levelStarts[level + 1] = levelStarts[level] + counts[level];
    const fill = levelStarts.slice(0, levelCount);
    const levelNodes = new Uint32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) levelNodes[fill[deepest - depth[node]]++] = node;
    return { levelStarts, levelNodes };
}

/** `[minX, minY, minZ, maxX, maxY, maxZ]`, with an empty box as `+max, -max`. */
type Box = [number, number, number, number, number, number];

function emptyBox(): Box {
    return [BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX];
}

function isEmpty(box: Box): boolean {
    return box[0] > box[3];
}

function unionInto(target: Box, other: Box): void {
    for (let axis = 0; axis < 3; axis++) {
        target[axis] = Math.min(target[axis], other[axis]);
        target[axis + 3] = Math.max(target[axis + 3], other[axis + 3]);
    }
}

/** Half the surface area: proportional to the chance that a random ray hits the box. 0 for an empty box. */
export function halfArea(box: ArrayLike<number>): number {
    if (box[0] > box[3]) return 0;
    const x = box[3] - box[0];
    const y = box[4] - box[1];
    const z = box[5] - box[2];
    return x * y + y * z + z * x;
}

function readSlot(floats: Float32Array, nodeWordOffset: number, slot: number): Box {
    const base = nodeWordOffset + (slot >> 1) * GPU_LAYOUT_NODE_WORDS + (slot & 1) * GPU_LAYOUT_CHILD_STRIDE_WORDS;
    const box: Box = [floats[base], floats[base + 1], floats[base + 2], floats[base + 4], floats[base + 5], floats[base + 6]];
    // The stored form of an empty box (a point at the largest float) reads back as empty.
    return box[0] === Math.fround(BVH_FLOAT_MAX) ? emptyBox() : box;
}

function writeSlot(floats: Float32Array, nodeWordOffset: number, slot: number, box: Box): void {
    const base = nodeWordOffset + (slot >> 1) * GPU_LAYOUT_NODE_WORDS + (slot & 1) * GPU_LAYOUT_CHILD_STRIDE_WORDS;
    const stored = isEmpty(box) ? [BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX] : box;
    floats[base] = stored[0];
    floats[base + 1] = stored[1];
    floats[base + 2] = stored[2];
    floats[base + 4] = stored[3];
    floats[base + 5] = stored[4];
    floats[base + 6] = stored[5];
}

/**
 * The tree-quality signal: the summed half-area of every inner node (the
 * root included) over the root's. It is the expected number of inner nodes a
 * random ray through the root's box visits, the traversal half of the SAH
 * cost. A refit keeps the tree's shape while things move, so this grows as
 * the shape stops suiting the new positions; a rebuild resets it.
 *
 * Reads the boxes stored in the nodes. 1 for a one-node tree, 0 for an empty one.
 */
export function relativeSahCost(floats: Float32Array, words: Uint32Array, nodeWordOffset: number, nodeCount: number): number {
    if (nodeCount === 0) return 0;
    let summed = 0;
    let rootArea = 0;
    for (let node = 0; node < nodeCount; node++) {
        const box = readSlot(floats, nodeWordOffset, node * 2);
        if (words[nodeWordOffset + node * GPU_LAYOUT_NODE_WORDS + GPU_LAYOUT_NODE_WORD.rightIndex] !== GPU_LAYOUT_EMPTY_CHILD) {
            unionInto(box, readSlot(floats, nodeWordOffset, node * 2 + 1));
        }
        const area = halfArea(box);
        summed += area;
        if (node === 0) rootArea = area;
    }
    return rootArea > 0 ? summed / rootArea : 0;
}

export interface CpuRefitResult {
    /** `[minX, minY, minZ, maxX, maxY, maxZ]` of the root. */
    readonly rootBox: Float64Array;
    /** Summed half-area of every inner node, the root included (the numerator of {@link relativeSahCost}). */
    readonly summedHalfArea: number;
}

/**
 * Refits the tree in `floats`/`words` (one buffer, two views) in place from
 * `primitiveBoxes` ({@link PRIMITIVE_BOX_FLOATS} per primitive), keeping its
 * shape: each leaf slot gets the union of its primitives' boxes, each inner
 * slot the union of its node's two slots. Empty primitive boxes (`min > max`)
 * stay empty, and an empty slot is stored as a point at the largest float.
 */
export function refitOnCpu(
    floats: Float32Array,
    words: Uint32Array,
    nodeWordOffset: number,
    nodeCount: number,
    parents: Uint32Array,
    primitiveBoxes: Float32Array,
): CpuRefitResult {
    if (nodeCount === 0) return { rootBox: new Float64Array(emptyBox()), summedHalfArea: 0 };
    // Children before parents: a node's index is not ordered, so visit by depth, deepest first.
    const depth = new Uint32Array(nodeCount);
    for (let node = 0; node < nodeCount; node++) {
        let level = 0;
        for (let up = parents[node]; up !== GPU_LAYOUT_NO_PARENT; up = parents[up]) level++;
        depth[node] = level;
    }
    const order = Array.from({ length: nodeCount }, (_, node) => node).sort((a, b) => depth[b] - depth[a]);
    const nodeBoxes: Box[] = new Array<Box>(nodeCount);
    let summedHalfArea = 0;
    for (const node of order) {
        const base = nodeWordOffset + node * GPU_LAYOUT_NODE_WORDS;
        const box = emptyBox();
        for (let side = 0; side < 2; side++) {
            const index = words[base + indexWord(side)];
            if (index === GPU_LAYOUT_EMPTY_CHILD) continue;
            const count = words[base + countWord(side)];
            let child: Box;
            if (count > 0) {
                child = emptyBox();
                for (let primitive = index; primitive < index + count; primitive++) {
                    const at = primitive * PRIMITIVE_BOX_FLOATS;
                    unionInto(child, [
                        primitiveBoxes[at], primitiveBoxes[at + 1], primitiveBoxes[at + 2],
                        primitiveBoxes[at + 4], primitiveBoxes[at + 5], primitiveBoxes[at + 6],
                    ]);
                }
            } else {
                child = nodeBoxes[index];
            }
            writeSlot(floats, nodeWordOffset, node * 2 + side, child);
            unionInto(box, child);
        }
        nodeBoxes[node] = box;
        summedHalfArea += halfArea(box);
    }
    return { rootBox: new Float64Array(nodeBoxes[0]), summedHalfArea };
}
