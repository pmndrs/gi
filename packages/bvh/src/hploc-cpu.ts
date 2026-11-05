/**
 * H-PLOC (Benthin et al., HPG 2024, "H-PLOC: Hierarchical Parallel
 * Locally-Ordered Clustering for Bounding Volume Hierarchy Construction") on
 * the CPU, one leaf's climb after another: tinybvh's `BuildHPLOC` (MIT, Jacco
 * Bikker; licenses/tinybvh.txt) ported step for step, with the Morton codes as
 * a choice, and one change, on exact ties (below). It is the oracle for the
 * GPU H-PLOC (gpu-hploc.ts):
 *
 * - with `mortonCodes: 'tinybvh-63'` and `tieBreak: 'tinybvh'` it builds the
 *   tree tinybvh's `BuildHPLOC` builds (tinybvh says its trees are identical
 *   to NexusBVH's), which the unit tests check against tinybvh itself
 *   (`builder: 'hploc'`);
 * - with any options it builds what the GPU builds with the same options.
 *
 * **Ties.** tinybvh takes the first of equally good neighbours, in the order
 * +1, -1, +2, -2, ... On exactly equal areas (identical triangles, or evenly
 * spaced equal ones) every cluster then picks the one after it, only the last
 * pair is mutual, and each round merges one pair: the tree becomes a chain.
 * 3000 identical triangles give depth 128 that way, twice the tracer's
 * 64-entry stack. So by default (`tieBreak: 'pairs'`) an exact tie goes to
 * the nearer neighbour, then the pair that starts on an even place, then the
 * lower place: ties pair up as (0, 1), (2, 3), ... and the tree stays
 * balanced. Only exact ties change. They are common on regular grids, where
 * the pairs make slightly cheaper trees than tinybvh's (1.8% on the test
 * grid); `tieBreak: 'tinybvh'` keeps tinybvh's rule, for comparisons.
 *
 * Arithmetic is rounded to f32 at every step (`Math.fround`), as in the LBVH
 * twin (lbvh-cpu.ts), so it computes what the kernels compute, except where
 * the GPU's division or a fused multiply-add rounds differently.
 *
 * **The algorithm.** Triangles are sorted by Morton code. Each one climbs
 * Apetrei's radix tree (as in the LBVH), and the climb that finishes a node
 * holds the node's clusters: the left child's then the right child's, in
 * Morton order. While a node holds more than {@link HPLOC_MERGE_THRESHOLD}
 * (the root: more than one), PLOC merges them: each cluster finds the cluster
 * within {@link HPLOC_SEARCH_RADIUS} places whose merged box has the smallest
 * surface area, and mutual nearest neighbours become one cluster, a new inner
 * node. So a node never holds more than twice the threshold, which is what
 * lets the GPU merge one node's clusters in one subgroup.
 *
 * **The layout.** H-PLOC makes leaves of one triangle. The layout step
 * (`hplocLayout`, the same rules as the GPU's) then collapses subtrees that
 * are cheaper as one leaf (the SAH leaf collapse of B1, at most
 * `maxLeafPrimitives` triangles), and writes the tracer's layout.
 */

import {
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
    type GpuLayoutMeshBvh,
} from './gpu-layout.ts';
import type { GpuLbvhOptions } from './gpu-lbvh.ts';
import { mortonCodes, packTrianglesInOrder, singleTriangleLayout, triangleBoxes } from './lbvh-cpu.ts';
import type { TinyBvhInput } from './tinybvh.ts';

const f = Math.fround;
const FLOAT_MAX = 3.4028234663852886e38;

/** How far, in Morton order, a cluster looks for its nearest neighbour (tinybvh, NexusBVH: 8). */
export const HPLOC_SEARCH_RADIUS = 8;

/** A node keeps merging while it holds more clusters than this: half the 32-wide wave of the paper. */
export const HPLOC_MERGE_THRESHOLD = 16;

/** Most clusters one node holds while it merges: both children's, each at most the threshold. */
export const HPLOC_MAX_CLUSTERS = 2 * HPLOC_MERGE_THRESHOLD;

/**
 * Which Morton codes order the triangles:
 * - `'cubic-30'`: B1's, 30 bits in cubic cells over the mesh's centroid box
 *   (one sort word);
 * - `'tinybvh-63'`: tinybvh's `BuildHPLOC` codes, 63 bits (21 per axis), each
 *   axis stretched over the mesh's box (two sort words). Needed to compare
 *   trees with tinybvh's.
 */
export type HplocMortonCodes = 'cubic-30' | 'tinybvh-63';

/**
 * How the nearest-neighbour search breaks exact ties: `'pairs'` (the
 * default; see the top of this file), or `'tinybvh'` exactly as tinybvh does,
 * to compare trees with tinybvh's.
 */
export type HplocTieBreak = 'pairs' | 'tinybvh';

export interface HplocOptions extends GpuLbvhOptions {
    /** Default `'cubic-30'`. */
    readonly mortonCodes?: HplocMortonCodes;
    /** Default `'pairs'`. */
    readonly tieBreak?: HplocTieBreak;
}

/**
 * The order among pairs at an equal merged area (`'pairs'`): nearer first,
 * then a pair starting on an even place, then the lower one. It belongs to the
 * pair, not to either cluster, so both see the same order (the GPU's hplocPairRank).
 */
export function hplocPairRank(a: number, b: number): number {
    const low = Math.min(a, b);
    return (Math.max(a, b) - low) * 64 + (low & 1) * 32 + low;
}

/** Each triangle's Morton code as two words: the high word, then the low word (0 for 30-bit codes). */
export function hplocMortonKeys(boxes: Float32Array, centroidBounds: Float32Array, codes: HplocMortonCodes): { high: Uint32Array; low: Uint32Array } {
    const count = boxes.length / 6;
    if (codes === 'cubic-30') return { high: mortonCodes(boxes, centroidBounds), low: new Uint32Array(count) };
    // tinybvh: the box around every triangle's box, 21 bits per axis over each axis' extent.
    const rootMin = [FLOAT_MAX, FLOAT_MAX, FLOAT_MAX];
    const rootMax = [-FLOAT_MAX, -FLOAT_MAX, -FLOAT_MAX];
    for (let triangle = 0; triangle < count; triangle++) {
        for (let axis = 0; axis < 3; axis++) {
            rootMin[axis] = Math.min(rootMin[axis], boxes[triangle * 6 + axis]);
            rootMax[axis] = Math.max(rootMax[axis], boxes[triangle * 6 + 3 + axis]);
        }
    }
    const extent = [0, 1, 2].map((axis) => f(rootMax[axis] - rootMin[axis]));
    const cellsPerAxis = 0x1fffff;
    const high = new Uint32Array(count);
    const low = new Uint32Array(count);
    const cell = [0, 0, 0];
    for (let triangle = 0; triangle < count; triangle++) {
        for (let axis = 0; axis < 3; axis++) {
            const centroid = f(f(f(boxes[triangle * 6 + axis] + boxes[triangle * 6 + 3 + axis]) * 0.5) - rootMin[axis]);
            cell[axis] = extent[axis] > 0 ? Math.trunc(Math.min(Math.max(f(f(centroid / extent[axis]) * cellsPerAxis), 0), cellsPerAxis)) : 0;
        }
        // Bit b of axis a lands at bit 3b + a (x lowest, as tinybvh interleaves).
        let highWord = 0;
        let lowWord = 0;
        for (let bit = 0; bit < 21; bit++) {
            for (let axis = 0; axis < 3; axis++) {
                if (((cell[axis] >>> bit) & 1) === 0) continue;
                const at = 3 * bit + axis;
                if (at < 32) lowWord |= 1 << at;
                else highWord |= 1 << (at - 32);
            }
        }
        high[triangle] = highWord >>> 0;
        low[triangle] = lowWord >>> 0;
    }
    return { high, low };
}

/** Triangle numbers sorted by (high, low) key, equal keys in triangle order (two stable sorts give the same). */
export function hplocSortedOrder(high: Uint32Array, low: Uint32Array): Uint32Array {
    const order = Uint32Array.from({ length: high.length }, (_, index) => index);
    order.sort((a, b) => high[a] - high[b] || low[a] - low[b] || a - b);
    return order;
}

/**
 * The clusters H-PLOC made. Cluster `p < triangleCount` is the leaf at sorted
 * position `p` (the triangle `order[p]`); cluster `triangleCount + k` is inner
 * node `k`, with children `left[k]` and `right[k]` and box `innerBoxes[6k..]`.
 */
export interface HplocClusterTree {
    readonly triangleCount: number;
    readonly order: Uint32Array;
    readonly leafBoxes: Float32Array;
    readonly left: Uint32Array;
    readonly right: Uint32Array;
    readonly innerBoxes: Float32Array;
    readonly root: number;
}

/** The tree tinybvh's `BuildHPLOC` builds, from the triangles' boxes and sorted order (ties broken as `tieBreak` says). */
export function buildHplocClusters(
    boxes: Float32Array,
    order: Uint32Array,
    high: Uint32Array,
    low: Uint32Array,
    tieBreak: HplocTieBreak = 'pairs',
): HplocClusterTree {
    const count = order.length;
    const leafBoxes = new Float32Array(count * 6);
    order.forEach((triangle, position) => leafBoxes.set(boxes.subarray(triangle * 6, triangle * 6 + 6), position * 6));
    const innerCount = Math.max(0, count - 1);
    const left = new Uint32Array(innerCount);
    const right = new Uint32Array(innerCount);
    const innerBoxes = new Float32Array(innerCount * 6);
    if (count === 1) return { triangleCount: 1, order, leafBoxes, left, right, innerBoxes, root: 0 };
    const boxOf = (cluster: number): Float32Array =>
        cluster < count ? leafBoxes.subarray(cluster * 6, cluster * 6 + 6) : innerBoxes.subarray((cluster - count) * 6, (cluster - count) * 6 + 6);

    // The LBVH split metric between neighbours: the keys' xor, or the positions' when the keys are equal.
    const deltaHigh = new Uint32Array(count - 1);
    const deltaLow = new Uint32Array(count - 1);
    for (let position = 0; position + 1 < count; position++) {
        const a = order[position];
        const b = order[position + 1];
        deltaHigh[position] = (high[a] ^ high[b]) >>> 0;
        deltaLow[position] = (low[a] ^ low[b]) >>> 0;
        if (deltaHigh[position] === 0 && deltaLow[position] === 0) deltaLow[position] = (position ^ (position + 1)) >>> 0;
    }
    const less = (a: number, b: number): boolean => deltaHigh[a] < deltaHigh[b] || (deltaHigh[a] === deltaHigh[b] && deltaLow[a] < deltaLow[b]);

    const mergedArea = (a: number, b: number): number => {
        const boxA = boxOf(a);
        const boxB = boxOf(b);
        const x = f(Math.max(boxA[3], boxB[3]) - Math.min(boxA[0], boxB[0]));
        const y = f(Math.max(boxA[4], boxB[4]) - Math.min(boxA[1], boxB[1]));
        const z = f(Math.max(boxA[5], boxB[5]) - Math.min(boxA[2], boxB[2]));
        return f(f(f(x * y) + f(y * z)) + f(z * x));
    };

    // Each node's clusters live at the start of its range, as in tinybvh.
    const clusters = Uint32Array.from({ length: count }, (_, position) => position);
    const listSize = new Uint32Array(count).fill(1);
    // What the first climb to reach a node left there: the far end of its range (tinybvh's `parent`).
    const farEnd = new Int32Array(count - 1).fill(-1);
    const nearest = new Int32Array(HPLOC_MAX_CLUSTERS);
    const best = new Float32Array(HPLOC_MAX_CLUSTERS);
    let innerNext = 0;

    // Whether partner `other` at `area` is better for cluster `j` than its best so far.
    const closer = (j: number, other: number, area: number): boolean =>
        area < best[j] || (tieBreak === 'pairs' && area === best[j] && hplocPairRank(j, other) < hplocPairRank(j, nearest[j]));

    // One PLOC round over the `size` clusters at `start`: returns how many are left.
    const mergeRound = (start: number, size: number): number => {
        for (let j = 0; j < size; j++) {
            nearest[j] = j + 1 < size ? j + 1 : j - 1;
            best[j] = 1e30;
        }
        for (let r = 1; r <= HPLOC_SEARCH_RADIUS && r < size; r++) {
            for (let j = size - r - 1; j >= 0; j--) {
                const area = mergedArea(clusters[start + j], clusters[start + j + r]);
                if (closer(j, j + r, area)) {
                    best[j] = area;
                    nearest[j] = j + r;
                }
                if (closer(j + r, j, area)) {
                    best[j + r] = area;
                    nearest[j + r] = j;
                }
            }
        }
        // Mutual nearest neighbours merge; the new cluster takes the first one's place.
        let kept = 0;
        for (let j = 0; j < size; j++) {
            const k = nearest[j];
            if (nearest[k] !== j) {
                clusters[start + kept++] = clusters[start + j];
            } else if (j < k) {
                const a = clusters[start + j];
                const b = clusters[start + k];
                const boxA = boxOf(a);
                const boxB = boxOf(b);
                const box = innerBoxes.subarray(innerNext * 6, innerNext * 6 + 6);
                for (let axis = 0; axis < 3; axis++) {
                    box[axis] = Math.min(boxA[axis], boxB[axis]);
                    box[3 + axis] = Math.max(boxA[3 + axis], boxB[3 + axis]);
                }
                left[innerNext] = a;
                right[innerNext] = b;
                clusters[start + kept++] = count + innerNext++;
            }
        }
        if (kept === size) throw new Error('H-PLOC: no mutual nearest neighbours (NaN boxes?)');
        return kept;
    };

    for (let position = 0; position < count; position++) {
        let first = position;
        let last = position;
        while (first > 0 || last < count - 1) {
            let split: number;
            if (first === 0 || (last < count - 1 && less(last, first - 1))) {
                // The range is the left child of the node split between last and last + 1.
                const other = farEnd[last];
                farEnd[last] = first;
                split = last + 1;
                if (other < 0) break;
                last = other;
            } else {
                const other = farEnd[first - 1];
                farEnd[first - 1] = last;
                split = first;
                if (other < 0) break;
                first = other;
            }
            const leftSize = listSize[first];
            const rightSize = listSize[split];
            clusters.copyWithin(first + leftSize, split, split + rightSize);
            let size = leftSize + rightSize;
            const threshold = first === 0 && last === count - 1 ? 1 : HPLOC_MERGE_THRESHOLD;
            while (size > threshold) size = mergeRound(first, size);
            listSize[first] = size;
        }
    }
    if (innerNext !== count - 1) throw new Error(`H-PLOC made ${innerNext} inner nodes, expected ${count - 1}`);
    return { triangleCount: count, order, leafBoxes, left, right, innerBoxes, root: clusters[0] };
}

/**
 * The tracer's layout of an H-PLOC tree, with the SAH leaf collapse: an inner
 * node of at most `maxLeafPrimitives` triangles becomes one leaf when that is
 * no more expensive (B1's rule; the root stays an inner node). Each leaf's
 * triangles are neighbours in the triangle order. Nodes are numbered depth
 * first; the GPU numbers them differently, so compare costs and counts, not bytes.
 */
export function hplocLayout(geometry: TinyBvhInput, tree: HplocClusterTree, options: GpuLbvhOptions = {}): GpuLayoutMeshBvh {
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    const traversalCost = f(options.traversalCost ?? 1);
    const intersectionCost = f(options.intersectionCost ?? 1);
    const { triangleCount: count, order, leafBoxes, left, right, innerBoxes, root } = tree;
    if (count === 1) return singleTriangleLayout(geometry, leafBoxes.subarray(0, 6));
    const halfArea = (box: Float32Array): number => {
        const x = f(box[3] - box[0]);
        const y = f(box[4] - box[1]);
        const z = f(box[5] - box[2]);
        return f(f(f(x * y) + f(y * z)) + f(z * x));
    };
    const boxOf = (cluster: number): Float32Array =>
        cluster < count ? leafBoxes.subarray(cluster * 6, cluster * 6 + 6) : innerBoxes.subarray((cluster - count) * 6, (cluster - count) * 6 + 6);

    // Bottom up (an inner node is always made after its children): triangles, cost, collapse and height.
    const innerCount = count - 1;
    const triangles = new Uint32Array(innerCount);
    const cost = new Float32Array(innerCount);
    const collapse = new Uint8Array(innerCount);
    const height = new Uint32Array(innerCount);
    const childTriangles = (cluster: number): number => (cluster < count ? 1 : triangles[cluster - count]);
    const childCost = (cluster: number): number => (cluster < count ? f(intersectionCost * halfArea(boxOf(cluster))) : cost[cluster - count]);
    const childHeight = (cluster: number): number => (cluster < count ? 0 : height[cluster - count]);
    let maxDepth = 0;
    for (let k = 0; k < innerCount; k++) {
        const area = halfArea(innerBoxes.subarray(k * 6, k * 6 + 6));
        const nodeTriangles = childTriangles(left[k]) + childTriangles(right[k]);
        const innerCost = f(f(f(traversalCost * area) + childCost(left[k])) + childCost(right[k]));
        const leafCost = nodeTriangles <= maxLeaf ? f(f(intersectionCost * nodeTriangles) * area) : FLOAT_MAX;
        const isRoot = count + k === root;
        triangles[k] = nodeTriangles;
        collapse[k] = !isRoot && leafCost <= innerCost ? 1 : 0;
        cost[k] = Math.min(innerCost, leafCost);
        const nodeHeight = 1 + Math.max(childHeight(left[k]), childHeight(right[k]));
        height[k] = collapse[k] ? 0 : nodeHeight;
        if (isRoot) maxDepth = nodeHeight;
    }

    // Depth first from the root. A leaf (one triangle, or a collapsed subtree) takes the next run of triangle positions.
    const treeOrder = new Uint32Array(count);
    let placed = 0;
    const placeTriangles = (cluster: number): void => {
        const stack = [cluster];
        while (stack.length > 0) {
            const next = stack.pop()!;
            if (next < count) {
                treeOrder[placed++] = order[next];
            } else {
                stack.push(right[next - count], left[next - count]);
            }
        }
    };
    // Kept nodes: every inner node but those inside collapsed subtrees (a subtree of t triangles has t - 1).
    let keptCount = innerCount;
    for (const pending = [root]; pending.length > 0; ) {
        const cluster = pending.pop()!;
        if (cluster < count) continue;
        const k = cluster - count;
        if (collapse[k]) keptCount -= triangles[k] - 1;
        else pending.push(left[k], right[k]);
    }
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
    let numbered = 1;
    const stack: { cluster: number; out: number; parent: number }[] = [{ cluster: root, out: 0, parent: GPU_LAYOUT_NO_PARENT }];
    while (stack.length > 0) {
        const { cluster, out, parent } = stack.pop()!;
        const k = cluster - count;
        parents[out] = parent;
        const children = [left[k], right[k]];
        for (let side = 0; side < 2; side++) {
            const child = children[side];
            const base = out * GPU_LAYOUT_NODE_WORDS;
            const box = boxOf(child);
            for (let axis = 0; axis < 3; axis++) {
                floats[base + minWord[side] + axis] = box[axis];
                floats[base + minWord[side] + maxOffset + axis] = box[3 + axis];
            }
            if (child < count || collapse[child - count]) {
                words[base + indexWord[side]] = placed;
                words[base + countWord[side]] = childTriangles(child);
                placeTriangles(child);
                leafCount++;
            } else {
                const childOut = numbered++;
                words[base + indexWord[side]] = childOut;
                words[base + countWord[side]] = 0;
                stack.push({ cluster: child, out: childOut, parent: out });
            }
        }
    }
    const rootBox = boxOf(root);
    return {
        nodes: nodeBuffer,
        triangles: packTrianglesInOrder(geometry, treeOrder),
        parents,
        nodeCount: keptCount,
        triangleCount: count,
        maxDepth,
        leafCount,
        bounds: rootBox.slice(),
    };
}

/** The whole CPU H-PLOC: boxes, Morton codes, sort, clusters, layout. */
export function buildHplocOnCpu(geometry: TinyBvhInput, options: HplocOptions = {}): GpuLayoutMeshBvh {
    const { boxes, centroidBounds } = triangleBoxes(geometry);
    const { high, low } = hplocMortonKeys(boxes, centroidBounds, options.mortonCodes ?? 'cubic-30');
    const order = hplocSortedOrder(high, low);
    return hplocLayout(geometry, buildHplocClusters(boxes, order, high, low, options.tieBreak), options);
}
