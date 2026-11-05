/**
 * The GPU LBVH (gpu-lbvh.ts) written plainly on the CPU: the same Morton codes,
 * stable sort, Apetrei climb, SAH collapse, node numbering and layout, one
 * step after another. It exists to test the algorithm without a GPU (the unit
 * tests) and to cross-check the GPU's trees (the bvh-lbvh page). It is not a
 * builder anyone should ship: tinybvh (B0) is the CPU builder.
 *
 * Arithmetic is rounded to f32 at every step (`Math.fround`), so it computes
 * what the kernels compute, except where the GPU's division (WGSL allows 2.5
 * units in the last place) or a fused multiply-add rounds differently. Then a
 * Morton code or a near-tie collapse decision can differ, which is why the GPU
 * page compares tree cost and size, not bytes.
 */

import {
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
    GPU_LAYOUT_TRIANGLE_BYTES,
    GPU_LAYOUT_TRIANGLE_WORD,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';
import type { GpuLbvhOptions } from './gpu-lbvh.ts';
import type { TinyBvhInput } from './tinybvh.ts';

const f = Math.fround;
const FLOAT_MAX = 3.4028234663852886e38;

/** HIPRT's expandBits3D: the low 10 bits of `value`, two zero bits after each. */
export function expandMortonBits(value: number): number {
    let v = Math.imul(value, 0x00010001) & 0xff0000ff;
    v = Math.imul(v, 0x00000101) & 0x0f00f00f;
    v = Math.imul(v, 0x00000011) & 0xc30c30c3;
    v = Math.imul(v, 0x00000005) & 0x49249249;
    return v >>> 0;
}

/** The 30-bit Morton code of a cell (10 bits per axis), x in the highest of each triple. */
export function mortonCode(x: number, y: number, z: number): number {
    return (expandMortonBits(x) * 4 + expandMortonBits(y) * 2 + expandMortonBits(z)) >>> 0;
}

/** Each triangle's box (min xyz, max xyz) and the box around their centroids. */
export function triangleBoxes(geometry: TinyBvhInput): { boxes: Float32Array; centroidBounds: Float32Array } {
    const { positions, indices } = geometry;
    const count = indices ? indices.length / 3 : positions.length / 9;
    const boxes = new Float32Array(count * 6);
    const centroidBounds = new Float32Array([FLOAT_MAX, FLOAT_MAX, FLOAT_MAX, -FLOAT_MAX, -FLOAT_MAX, -FLOAT_MAX]);
    for (let triangle = 0; triangle < count; triangle++) {
        const box = boxes.subarray(triangle * 6, triangle * 6 + 6);
        box.set([FLOAT_MAX, FLOAT_MAX, FLOAT_MAX, -FLOAT_MAX, -FLOAT_MAX, -FLOAT_MAX]);
        for (let corner = 0; corner < 3; corner++) {
            const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
            for (let axis = 0; axis < 3; axis++) {
                const value = positions[vertex * 3 + axis];
                box[axis] = Math.min(box[axis], value);
                box[3 + axis] = Math.max(box[3 + axis], value);
            }
        }
        for (let axis = 0; axis < 3; axis++) {
            const centroid = f(f(box[axis] + box[3 + axis]) * 0.5);
            centroidBounds[axis] = Math.min(centroidBounds[axis], centroid);
            centroidBounds[3 + axis] = Math.max(centroidBounds[3 + axis], centroid);
        }
    }
    return { boxes, centroidBounds };
}

/** Morton codes in cubic cells over the centroid box (1024 along its longest side), as the `lbvhMorton` kernel computes them. */
export function mortonCodes(boxes: Float32Array, centroidBounds: Float32Array): Uint32Array {
    const count = boxes.length / 6;
    const keys = new Uint32Array(count);
    const longest = Math.max(...[0, 1, 2].map((axis) => f(centroidBounds[3 + axis] - centroidBounds[axis])));
    const scale = longest > 0 ? f(1024 / longest) : 0;
    const cell = [0, 0, 0];
    for (let triangle = 0; triangle < count; triangle++) {
        for (let axis = 0; axis < 3; axis++) {
            const centroid = f(f(boxes[triangle * 6 + axis] + boxes[triangle * 6 + 3 + axis]) * 0.5);
            const scaled = f(f(centroid - centroidBounds[axis]) * scale);
            cell[axis] = Math.trunc(Math.min(Math.max(scaled, 0), 1023));
        }
        keys[triangle] = mortonCode(cell[0], cell[1], cell[2]);
    }
    return keys;
}

/** Triangle numbers sorted by key, equal keys in triangle order (what the stable radix sort gives). */
export function stableSortByKey(keys: Uint32Array): { keys: Uint32Array; values: Uint32Array } {
    const values = Uint32Array.from({ length: keys.length }, (_, index) => index);
    values.sort((a, b) => keys[a] - keys[b] || a - b);
    return { keys: Uint32Array.from(values, (value) => keys[value]), values };
}

/** One inner node of the radix tree, split between sorted positions `split` and `split + 1`. */
interface RadixNode {
    first: number;
    count: number;
    collapse: boolean;
    parent: number;
    /** Per side: an inner node's split, or a leaf as `-1 - position`. */
    child: [number, number];
    box: [Float32Array, Float32Array];
}

/** Builds the tree from sorted keys and the triangles at each sorted position, as the GPU does after its sort. */
export function buildLbvhFromSorted(
    geometry: TinyBvhInput,
    boxes: Float32Array,
    sorted: { keys: Uint32Array; values: Uint32Array },
    options: GpuLbvhOptions = {},
): GpuLayoutMeshBvh & { rootSplit: number } {
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    const traversalCost = f(options.traversalCost ?? 1);
    const intersectionCost = f(options.intersectionCost ?? 1);
    const { keys, values } = sorted;
    const count = keys.length;
    const halfArea = (lower: ArrayLike<number>, upper: ArrayLike<number>): number => {
        const x = f(upper[0] - lower[0]);
        const y = f(upper[1] - lower[1]);
        const z = f(upper[2] - lower[2]);
        return f(f(f(x * y) + f(y * z)) + f(z * x));
    };
    const triangleBox = (position: number): Float32Array => boxes.subarray(values[position] * 6, values[position] * 6 + 6);

    if (count === 1) return { ...singleTriangleLayout(geometry, triangleBox(0)), rootSplit: 0 };

    // HIPRT's delta with the position tie-break, compared as (key xor, position xor).
    const less = (a: number, b: number): boolean => {
        const keyA = (keys[a] ^ keys[a + 1]) >>> 0;
        const keyB = (keys[b] ^ keys[b + 1]) >>> 0;
        return keyA < keyB || (keyA === keyB && ((a ^ (a + 1)) >>> 0) < ((b ^ (b + 1)) >>> 0));
    };
    const findParent = (first: number, end: number): { node: number; side: number } =>
        first === 0 || (end !== count && less(end - 1, first - 1)) ? { node: end - 1, side: 0 } : { node: first - 1, side: 1 };

    const nodes: RadixNode[] = Array.from({ length: count - 1 }, () => ({
        first: 0,
        count: 0,
        collapse: false,
        parent: GPU_LAYOUT_NO_PARENT,
        child: [0, 0],
        box: [new Float32Array(6), new Float32Array(6)],
    }));
    const exchange = new Int32Array(count - 1).fill(-1);
    const slotCost = new Float32Array((count - 1) * 2);
    const slotHeight = new Uint32Array((count - 1) * 2);
    let rootSplit = -1;
    let maxDepth = 0;

    // Leaves climb in order. Which thread arrives second never changes the result, only who computes it.
    for (let position = 0; position < count; position++) {
        let first = position;
        let end = position + 1;
        let reference = -1 - position;
        let box: Float32Array = triangleBox(position);
        let cost = f(intersectionCost * halfArea(box, box.subarray(3)));
        let height = 0;
        let parent = findParent(first, end);
        for (;;) {
            const node = nodes[parent.node];
            node.child[parent.side] = reference;
            node.box[parent.side].set(box);
            slotCost[parent.node * 2 + parent.side] = cost;
            slotHeight[parent.node * 2 + parent.side] = height;
            const other = exchange[parent.node];
            exchange[parent.node] = parent.side === 1 ? end : first;
            if (other < 0) break;
            if (parent.side === 1) first = other;
            else end = other;
            const sibling = 1 - parent.side;
            const merged = new Float32Array(6);
            for (let axis = 0; axis < 3; axis++) {
                merged[axis] = Math.min(node.box[0][axis], node.box[1][axis]);
                merged[3 + axis] = Math.max(node.box[0][3 + axis], node.box[1][3 + axis]);
            }
            const area = halfArea(merged, merged.subarray(3));
            const nodeCount = end - first;
            const innerCost = f(f(f(traversalCost * area) + cost) + slotCost[parent.node * 2 + sibling]);
            const leafCost = nodeCount <= maxLeaf ? f(f(intersectionCost * nodeCount) * area) : FLOAT_MAX;
            const isRoot = first === 0 && end === count;
            node.first = first;
            node.count = nodeCount;
            node.collapse = !isRoot && leafCost <= innerCost;
            const nodeHeight = 1 + Math.max(height, slotHeight[parent.node * 2 + sibling]);
            reference = parent.node;
            box = merged;
            cost = Math.min(innerCost, leafCost);
            height = node.collapse ? 0 : nodeHeight;
            if (isRoot) {
                rootSplit = parent.node;
                maxDepth = nodeHeight;
                node.parent = GPU_LAYOUT_NO_PARENT;
                break;
            }
            const next = findParent(first, end);
            node.parent = next.node;
            parent = next;
        }
    }
    if (rootSplit < 0) throw new Error('the climb never reached the root');

    // Which inner nodes stay, and their numbers: split order, with the root swapped to 0.
    const kept = (split: number): boolean => {
        if (nodes[split].collapse) return false;
        for (let above = nodes[split].parent; above !== GPU_LAYOUT_NO_PARENT; above = nodes[above].parent) {
            if (nodes[above].count > maxLeaf) return true;
            if (nodes[above].collapse) return false;
        }
        return true;
    };
    const rank = new Uint32Array(count - 1);
    let keptCount = 0;
    for (let split = 0; split < count - 1; split++) {
        rank[split] = keptCount;
        if (kept(split)) keptCount++;
    }
    const rootRank = rank[rootSplit];
    const outputIndex = (split: number): number => (rank[split] === rootRank ? 0 : rank[split] === 0 ? rootRank : rank[split]);

    const nodeBuffer = new ArrayBuffer(keptCount * GPU_LAYOUT_NODE_BYTES);
    const words = new Uint32Array(nodeBuffer);
    const floats = new Float32Array(nodeBuffer);
    const parents = new Uint32Array(keptCount);
    const word = GPU_LAYOUT_NODE_WORD;
    const indexWord = [word.leftIndex, word.rightIndex];
    const countWord = [word.leftCount, word.rightCount];
    const minWord = [word.leftMin, word.rightMin];
    const maxOffset = word.leftMax - word.leftMin;
    let leafCount = 0;
    for (let split = 0; split < count - 1; split++) {
        if (!kept(split)) continue;
        const node = nodes[split];
        const out = outputIndex(split);
        for (let side = 0; side < 2; side++) {
            const child = node.child[side];
            let index: number;
            let childCount: number;
            if (child < 0) {
                index = -1 - child;
                childCount = 1;
            } else if (nodes[child].collapse) {
                index = nodes[child].first;
                childCount = nodes[child].count;
            } else {
                index = outputIndex(child);
                childCount = 0;
            }
            if (childCount > 0) leafCount++;
            const base = out * GPU_LAYOUT_NODE_WORDS;
            words[base + indexWord[side]] = index;
            words[base + countWord[side]] = childCount;
            for (let axis = 0; axis < 3; axis++) {
                floats[base + minWord[side] + axis] = node.box[side][axis];
                floats[base + minWord[side] + maxOffset + axis] = node.box[side][3 + axis];
            }
        }
        parents[out] = split === rootSplit ? GPU_LAYOUT_NO_PARENT : outputIndex(node.parent);
    }
    const root = nodes[rootSplit];
    const bounds = new Float32Array(6);
    for (let axis = 0; axis < 3; axis++) {
        bounds[axis] = Math.min(root.box[0][axis], root.box[1][axis]);
        bounds[3 + axis] = Math.max(root.box[0][3 + axis], root.box[1][3 + axis]);
    }
    return {
        nodes: nodeBuffer,
        triangles: packTrianglesInOrder(geometry, values),
        parents,
        nodeCount: keptCount,
        triangleCount: count,
        maxDepth,
        leafCount,
        bounds,
        rootSplit,
    };
}

/** The whole CPU twin: boxes, Morton codes, sort, build. */
export function buildLbvhOnCpu(geometry: TinyBvhInput, options: GpuLbvhOptions = {}): GpuLayoutMeshBvh {
    const { boxes, centroidBounds } = triangleBoxes(geometry);
    const sorted = stableSortByKey(mortonCodes(boxes, centroidBounds));
    return buildLbvhFromSorted(geometry, boxes, sorted, options);
}

function singleTriangleLayout(geometry: TinyBvhInput, box: Float32Array): GpuLayoutMeshBvh {
    const nodes = new ArrayBuffer(GPU_LAYOUT_NODE_BYTES);
    const words = new Uint32Array(nodes);
    const floats = new Float32Array(nodes);
    const word = GPU_LAYOUT_NODE_WORD;
    for (let axis = 0; axis < 3; axis++) {
        floats[word.leftMin + axis] = box[axis];
        floats[word.leftMax + axis] = box[3 + axis];
    }
    words[word.leftIndex] = 0;
    words[word.leftCount] = 1;
    words[word.rightIndex] = GPU_LAYOUT_EMPTY_CHILD;
    words[word.rightCount] = 0;
    return {
        nodes,
        triangles: packTrianglesInOrder(geometry, Uint32Array.of(0)),
        parents: Uint32Array.of(GPU_LAYOUT_NO_PARENT),
        nodeCount: 1,
        triangleCount: 1,
        maxDepth: 1,
        leafCount: 1,
        bounds: box.slice(),
    };
}

/** Triangles in tree order (`order[position]` is the triangle there), 48 bytes each, corners as they are. */
function packTrianglesInOrder(geometry: TinyBvhInput, order: Uint32Array): ArrayBuffer {
    const { positions, indices, triangleFlags } = geometry;
    const buffer = new ArrayBuffer(order.length * GPU_LAYOUT_TRIANGLE_BYTES);
    const words = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);
    const triangleWords = GPU_LAYOUT_TRIANGLE_BYTES / 4;
    const t = GPU_LAYOUT_TRIANGLE_WORD;
    const corners = [t.vertex0, t.vertex1, t.vertex2];
    order.forEach((triangle, position) => {
        const base = position * triangleWords;
        for (let corner = 0; corner < 3; corner++) {
            const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
            for (let axis = 0; axis < 3; axis++) floats[base + corners[corner] + axis] = positions[vertex * 3 + axis];
        }
        words[base + t.originalIndex] = triangle;
        words[base + t.flags] = triangleFlags ? triangleFlags[triangle] : 0;
    });
    return buffer;
}
