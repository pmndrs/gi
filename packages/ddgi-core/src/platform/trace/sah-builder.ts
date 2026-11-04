/**
 * Binned surface-area-heuristic (SAH) BVH builder.
 *
 * One builder serves both levels: the per-mesh trees (primitives are
 * triangles) and the top level (primitives are instance boxes). It takes one
 * axis-aligned box per primitive and produces nodes in the 64-byte two-child
 * layout from `layout.ts`, plus the order the primitives must be stored in so
 * that every leaf points at a contiguous run.
 *
 * How it chooses a split, in plain terms. For a set of primitives it tries
 * all three axes. On each axis it drops the primitive centroids into 16 bins,
 * then scores the 15 bin boundaries by the SAH cost:
 *
 *     cost = traversalCost + intersectionCost * (A_left * N_left + A_right * N_right) / A_parent
 *
 * A is a box's surface area, N a primitive count. A random ray is more likely
 * to hit a big box, so the cost counts how many primitives a ray is expected
 * to test after taking one more step down the tree. The cheapest boundary
 * wins. A node with few enough primitives becomes a leaf when that is cheaper
 * than the best split.
 *
 * The builder never uses recursion (a degenerate scene could be deeper than
 * the JS call stack allows) and does not touch the DOM, so it runs unchanged
 * in a worker, in Node tests, or on the main thread.
 */

import { BVH_NODE_BYTES, BVH_NODE_WORDS, EMPTY_CHILD_INDEX, MAX_LEAF_PRIMITIVES, NODE_WORD, NO_PARENT } from './layout.ts';

export interface SahBuildOptions {
    /** Bins per axis. 16 is the usual sweet spot between build time and tree quality. */
    readonly binCount: number;
    /** Most primitives a leaf may hold. The design uses 4 for triangles and 1 for instances. */
    readonly maxLeafPrimitives: number;
    /** Relative cost of visiting a node (two box tests). Only the ratio to `intersectionCost` matters. */
    readonly traversalCost: number;
    /** Relative cost of testing one primitive. */
    readonly intersectionCost: number;
}

export const DEFAULT_SAH_OPTIONS: SahBuildOptions = {
    binCount: 16,
    maxLeafPrimitives: 4,
    traversalCost: 1,
    intersectionCost: 1,
};

export interface SahTree {
    /** `nodeCount` nodes in the layout from `layout.ts`. Child indices are relative to this tree. */
    readonly nodes: ArrayBuffer;
    readonly nodeCount: number;
    /**
     * `order[i]` is the input primitive that must be stored at position `i`.
     * A leaf's `index` and `count` refer to positions in this order.
     */
    readonly order: Uint32Array;
    /** Parent node of every node; the root has {@link NO_PARENT}. Only the refit pass reads this. */
    readonly parents: Uint32Array;
    /** Levels of nodes on the longest root-to-leaf path (a lone root counts as 1). Bounds the traversal stack use. */
    readonly maxDepth: number;
    readonly leafCount: number;
    /** `[minX, minY, minZ, maxX, maxY, maxZ]` around everything. */
    readonly bounds: Float32Array;
}

/** `[minX, minY, minZ, maxX, maxY, maxZ]` */
type Box = Float32Array;

interface SplitPlan {
    readonly kind: 'bins' | 'median';
    readonly axis: number;
    /** First bin that goes to the right child. */
    readonly boundaryBin: number;
    /** Smallest centroid on `axis` over the range being split. */
    readonly centroidMin: number;
    /** Bins per unit of centroid distance on `axis`. */
    readonly binScale: number;
    readonly cost: number;
}

/** Used when every centroid coincides, so no plane can separate the primitives. */
const MEDIAN_PLAN: SplitPlan = { kind: 'median', axis: 0, boundaryBin: 0, centroidMin: 0, binScale: 0, cost: Infinity };

type ChildPlan = { readonly kind: 'leaf' } | { readonly kind: 'inner'; readonly split: SplitPlan };

interface PendingNode {
    readonly nodeIndex: number;
    readonly start: number;
    readonly end: number;
    readonly depth: number;
    readonly split: SplitPlan;
}

function resetBox(box: Box): void {
    box[0] = box[1] = box[2] = Infinity;
    box[3] = box[4] = box[5] = -Infinity;
}

function growBox(box: Box, source: Float32Array, offset: number): void {
    for (let axis = 0; axis < 3; axis++) {
        if (source[offset + axis] < box[axis]) box[axis] = source[offset + axis];
        if (source[offset + 3 + axis] > box[3 + axis]) box[3 + axis] = source[offset + 3 + axis];
    }
}

function surfaceArea(box: Box): number {
    const dx = box[3] - box[0];
    const dy = box[4] - box[1];
    const dz = box[5] - box[2];
    // An empty box (no primitives yet) has negative extents.
    if (dx < 0 || dy < 0 || dz < 0) return 0;
    return 2 * (dx * dy + dy * dz + dz * dx);
}

function validateOptions(options: SahBuildOptions): void {
    if (!Number.isInteger(options.binCount) || options.binCount < 2 || options.binCount > 256) {
        throw new RangeError(`binCount must be an integer from 2 to 256, got ${options.binCount}`);
    }
    if (
        !Number.isInteger(options.maxLeafPrimitives) ||
        options.maxLeafPrimitives < 1 ||
        options.maxLeafPrimitives > MAX_LEAF_PRIMITIVES
    ) {
        throw new RangeError(`maxLeafPrimitives must be an integer from 1 to ${MAX_LEAF_PRIMITIVES}, got ${options.maxLeafPrimitives}`);
    }
    if (!(options.traversalCost > 0) || !(options.intersectionCost > 0)) {
        throw new RangeError('traversalCost and intersectionCost must be positive');
    }
}

/**
 * Builds a BVH over `primitiveBounds` (six floats per primitive).
 * Throws on non-finite bounds rather than producing a tree that silently misses.
 */
export function buildSahTree(primitiveBounds: Float32Array, options: SahBuildOptions = DEFAULT_SAH_OPTIONS): SahTree {
    validateOptions(options);
    if (primitiveBounds.length % 6 !== 0) {
        throw new RangeError('primitiveBounds must hold six floats per primitive');
    }
    return new SahTreeBuilder(primitiveBounds, options).build();
}

class SahTreeBuilder {
    private readonly primitiveCount: number;
    private readonly centroids: Float32Array;
    private readonly order: Uint32Array;

    private readonly nodeBuffer: ArrayBuffer;
    private readonly nodeFloats: Float32Array;
    private readonly nodeWords: Uint32Array;
    private readonly parents: Uint32Array;
    private nodeCount = 0;
    private maxDepth = 0;
    private leafCount = 0;

    // Scratch space reused by every split search.
    private readonly binCounts: Uint32Array;
    private readonly binBoxes: Float32Array;
    private readonly rightAreas: Float64Array;
    private readonly rightCounts: Uint32Array;
    private readonly runningBox: Box = new Float32Array(6);
    private readonly centroidMin = new Float64Array(3);
    private readonly centroidMax = new Float64Array(3);

    constructor(
        private readonly primitiveBounds: Float32Array,
        private readonly options: SahBuildOptions,
    ) {
        this.primitiveCount = primitiveBounds.length / 6;
        this.centroids = new Float32Array(this.primitiveCount * 3);
        this.order = new Uint32Array(this.primitiveCount);

        // A binary tree whose leaves hold at least one primitive has at most
        // (primitives - 1) inner nodes; a lone primitive still needs a root.
        const maxNodes = Math.max(1, this.primitiveCount - 1);
        this.nodeBuffer = new ArrayBuffer(maxNodes * BVH_NODE_BYTES);
        this.nodeFloats = new Float32Array(this.nodeBuffer);
        this.nodeWords = new Uint32Array(this.nodeBuffer);
        this.parents = new Uint32Array(maxNodes);

        this.binCounts = new Uint32Array(options.binCount);
        this.binBoxes = new Float32Array(options.binCount * 6);
        this.rightAreas = new Float64Array(options.binCount);
        this.rightCounts = new Uint32Array(options.binCount);
    }

    build(): SahTree {
        this.prepareCentroids();
        const rootBox = new Float32Array(6);
        resetBox(rootBox);
        for (let primitive = 0; primitive < this.primitiveCount; primitive++) {
            growBox(rootBox, this.primitiveBounds, primitive * 6);
        }

        if (this.primitiveCount === 0) {
            return this.finish(rootBox);
        }
        if (this.primitiveCount === 1) {
            this.buildSinglePrimitiveTree(rootBox);
            return this.finish(rootBox);
        }

        // The root is always an inner node, even when a single leaf would be
        // cheaper, because a node holds two children.
        const rootChoice = this.planChild(0, this.primitiveCount, rootBox, true);
        if (rootChoice.kind !== 'inner') throw new Error('unreachable: the root must split');
        this.parents[0] = NO_PARENT;
        this.nodeCount = 1;
        const pending: PendingNode[] = [
            { nodeIndex: 0, start: 0, end: this.primitiveCount, depth: 1, split: rootChoice.split },
        ];
        this.processPending(pending);
        return this.finish(rootBox);
    }

    private prepareCentroids(): void {
        for (let primitive = 0; primitive < this.primitiveCount; primitive++) {
            this.order[primitive] = primitive;
            for (let axis = 0; axis < 3; axis++) {
                const low = this.primitiveBounds[primitive * 6 + axis];
                const high = this.primitiveBounds[primitive * 6 + 3 + axis];
                const centroid = 0.5 * (low + high);
                if (!Number.isFinite(centroid) || !Number.isFinite(low) || !Number.isFinite(high)) {
                    throw new RangeError(`primitive ${primitive} has non-finite bounds`);
                }
                this.centroids[primitive * 3 + axis] = centroid;
            }
        }
    }

    private buildSinglePrimitiveTree(rootBox: Box): void {
        this.nodeCount = 1;
        this.parents[0] = NO_PARENT;
        this.maxDepth = 1;
        this.writeChild(0, 'left', rootBox, 0, 1);
        this.leafCount = 1;
        this.writeEmptyChild(0, 'right');
    }

    /** Splits pending nodes until every primitive sits in a leaf. */
    private processPending(pending: PendingNode[]): void {
        const leftBox = new Float32Array(6);
        const rightBox = new Float32Array(6);
        while (pending.length > 0) {
            const task = pending.pop() as PendingNode;
            this.maxDepth = Math.max(this.maxDepth, task.depth);

            const middle = this.partition(task.start, task.end, task.split);
            this.computeBounds(task.start, middle, leftBox);
            this.computeBounds(middle, task.end, rightBox);

            const sides = [
                { name: 'left' as const, start: task.start, end: middle, box: leftBox },
                { name: 'right' as const, start: middle, end: task.end, box: rightBox },
            ];
            const newChildren: PendingNode[] = [];
            for (const side of sides) {
                const choice = this.planChild(side.start, side.end, side.box, false);
                if (choice.kind === 'leaf') {
                    this.writeChild(task.nodeIndex, side.name, side.box, side.start, side.end - side.start);
                    this.leafCount++;
                } else {
                    const childIndex = this.nodeCount++;
                    this.parents[childIndex] = task.nodeIndex;
                    this.writeChild(task.nodeIndex, side.name, side.box, childIndex, 0);
                    newChildren.push({
                        nodeIndex: childIndex,
                        start: side.start,
                        end: side.end,
                        depth: task.depth + 1,
                        split: choice.split,
                    });
                }
            }
            // Child nodes were numbered left first, so two inner children sit
            // next to each other in memory. Push in reverse so the left one is
            // processed next (the stack pops the last pushed).
            for (let child = newChildren.length - 1; child >= 0; child--) pending.push(newChildren[child]);
        }
    }

    private finish(bounds: Box): SahTree {
        const nodeCount = this.nodeCount;
        return {
            nodes: this.nodeBuffer.slice(0, nodeCount * BVH_NODE_BYTES),
            nodeCount,
            order: this.order,
            parents: this.parents.slice(0, nodeCount),
            maxDepth: this.maxDepth,
            leafCount: this.leafCount,
            bounds: Float32Array.from(bounds),
        };
    }

    /** Leaf or inner node for the primitives `[start, end)`? Inner nodes come with the split to use. */
    private planChild(start: number, end: number, box: Box, forceSplit: boolean): ChildPlan {
        const count = end - start;
        if (count === 1 && !forceSplit) return { kind: 'leaf' };

        const best = this.findBestSplit(start, end, box);
        if (count <= this.options.maxLeafPrimitives && !forceSplit) {
            const leafCost = this.options.intersectionCost * count;
            if (best === null || leafCost <= best.cost) return { kind: 'leaf' };
        }
        // Too many primitives for a leaf, or splitting is cheaper. If no plane
        // separates the centroids, split the list in half so the tree still terminates.
        return { kind: 'inner', split: best ?? MEDIAN_PLAN };
    }

    private findBestSplit(start: number, end: number, parentBox: Box): SplitPlan | null {
        const { binCount, traversalCost, intersectionCost } = this.options;
        const total = end - start;
        const parentArea = surfaceArea(parentBox);

        this.centroidMin.fill(Infinity);
        this.centroidMax.fill(-Infinity);
        for (let position = start; position < end; position++) {
            const base = this.order[position] * 3;
            for (let axis = 0; axis < 3; axis++) {
                const value = this.centroids[base + axis];
                if (value < this.centroidMin[axis]) this.centroidMin[axis] = value;
                if (value > this.centroidMax[axis]) this.centroidMax[axis] = value;
            }
        }

        let best: SplitPlan | null = null;
        let bestCost = Infinity;
        const leftBox = this.runningBox;

        for (let axis = 0; axis < 3; axis++) {
            const extent = this.centroidMax[axis] - this.centroidMin[axis];
            if (!(extent > 0)) continue;
            const binScale = binCount / extent;
            if (!Number.isFinite(binScale)) continue;
            const centroidMin = this.centroidMin[axis];

            this.binCounts.fill(0);
            for (let bin = 0; bin < binCount; bin++) resetBox(this.binBoxes.subarray(bin * 6, bin * 6 + 6));
            for (let position = start; position < end; position++) {
                const primitive = this.order[position];
                const bin = this.binOf(primitive, axis, centroidMin, binScale);
                this.binCounts[bin]++;
                growBox(this.binBoxes.subarray(bin * 6, bin * 6 + 6), this.primitiveBounds, primitive * 6);
            }

            // Sweep from the right so each boundary knows its right side...
            resetBox(leftBox);
            let runningCount = 0;
            for (let bin = binCount - 1; bin >= 1; bin--) {
                runningCount += this.binCounts[bin];
                growBox(leftBox, this.binBoxes, bin * 6);
                this.rightAreas[bin] = surfaceArea(leftBox);
                this.rightCounts[bin] = runningCount;
            }
            // ...then from the left, scoring each boundary.
            resetBox(leftBox);
            runningCount = 0;
            for (let bin = 0; bin < binCount - 1; bin++) {
                runningCount += this.binCounts[bin];
                growBox(leftBox, this.binBoxes, bin * 6);
                const leftCount = runningCount;
                const rightCount = this.rightCounts[bin + 1];
                if (leftCount === 0 || rightCount === 0) continue;

                const leftArea = surfaceArea(leftBox);
                const rightArea = this.rightAreas[bin + 1];
                // Flat or point-like parents have no area to compare, so fall
                // back to the share of primitives each side holds.
                const expectedTests =
                    parentArea > 0
                        ? (leftArea * leftCount + rightArea * rightCount) / parentArea
                        : (leftCount * leftCount + rightCount * rightCount) / total;
                const cost = traversalCost + intersectionCost * expectedTests;
                if (cost < bestCost) {
                    bestCost = cost;
                    best = { kind: 'bins', axis, boundaryBin: bin + 1, centroidMin, binScale, cost };
                }
            }
        }
        return best;
    }

    private binOf(primitive: number, axis: number, centroidMin: number, binScale: number): number {
        const centroid = this.centroids[primitive * 3 + axis];
        const bin = Math.floor((centroid - centroidMin) * binScale);
        return Math.min(this.options.binCount - 1, Math.max(0, bin));
    }

    /** Reorders `order[start, end)` so the left child's primitives come first. Returns the first right index. */
    private partition(start: number, end: number, split: SplitPlan): number {
        if (split.kind === 'bins') {
            let left = start;
            let right = end - 1;
            while (left <= right) {
                const primitive = this.order[left];
                if (this.binOf(primitive, split.axis, split.centroidMin, split.binScale) < split.boundaryBin) {
                    left++;
                } else {
                    this.order[left] = this.order[right];
                    this.order[right] = primitive;
                    right--;
                }
            }
            // The cost search skipped empty sides, so both are non-empty here.
            if (left > start && left < end) return left;
        }
        return (start + end) >>> 1;
    }

    private computeBounds(start: number, end: number, out: Box): void {
        resetBox(out);
        for (let position = start; position < end; position++) {
            growBox(out, this.primitiveBounds, this.order[position] * 6);
        }
    }

    private writeChild(nodeIndex: number, side: 'left' | 'right', box: Box, index: number, count: number): void {
        const base = nodeIndex * BVH_NODE_WORDS;
        const left = side === 'left';
        this.nodeFloats.set(box.subarray(0, 3), base + (left ? NODE_WORD.leftMin : NODE_WORD.rightMin));
        this.nodeFloats.set(box.subarray(3, 6), base + (left ? NODE_WORD.leftMax : NODE_WORD.rightMax));
        this.nodeWords[base + (left ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)] = index;
        this.nodeWords[base + (left ? NODE_WORD.leftCount : NODE_WORD.rightCount)] = count;
    }

    private writeEmptyChild(nodeIndex: number, side: 'left' | 'right'): void {
        const base = nodeIndex * BVH_NODE_WORDS;
        const left = side === 'left';
        // The box words stay zero; traversal checks the index before it looks at the box.
        this.nodeWords[base + (left ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)] = EMPTY_CHILD_INDEX;
        this.nodeWords[base + (left ? NODE_WORD.leftCount : NODE_WORD.rightCount)] = 0;
    }
}
