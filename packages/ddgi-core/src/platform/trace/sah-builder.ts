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
 * How it stays fast (T0.10 part C; Sponza's 3.7 M triangles went from about
 * 36 s to a few seconds in Node). None of this changes which tree comes out:
 * - Primitive boxes and centroids are copied into flat arrays that are kept in
 *   the same order as `order`, and partitioning swaps all three in place. The
 *   inner loops then read memory front to back instead of jumping through
 *   `order`, and nothing is allocated per node.
 * - One pass over a node's primitives fills the bins of all three axes.
 * - The children's boxes are the unions of the bins on each side of the
 *   chosen boundary, kept from the sweep, so they are not recomputed. The
 *   children's centroid bounds are gathered while partitioning.
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

/** Floats in one box: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
const BOX_FLOATS = 6;

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

// ---- Small helpers on flat box arrays ------------------------------------------------------
//
// Every box lives in a Float32Array at some offset. Keeping one array type
// everywhere matters for speed: JS engines optimise a helper much better when
// it only ever sees one kind of typed array.

function resetBox(boxes: Float32Array, offset: number): void {
    boxes[offset] = boxes[offset + 1] = boxes[offset + 2] = Infinity;
    boxes[offset + 3] = boxes[offset + 4] = boxes[offset + 5] = -Infinity;
}

/** Grows the box at `target[targetOffset]` to contain the box at `source[sourceOffset]`. */
function growBox(target: Float32Array, targetOffset: number, source: Float32Array, sourceOffset: number): void {
    for (let axis = 0; axis < 3; axis++) {
        const low = source[sourceOffset + axis];
        const high = source[sourceOffset + 3 + axis];
        if (low < target[targetOffset + axis]) target[targetOffset + axis] = low;
        if (high > target[targetOffset + 3 + axis]) target[targetOffset + 3 + axis] = high;
    }
}

/** Grows the box at `target[targetOffset]` to contain the point at `point[pointOffset]`. */
function growBoxByPoint(target: Float32Array, targetOffset: number, point: Float32Array, pointOffset: number): void {
    for (let axis = 0; axis < 3; axis++) {
        const value = point[pointOffset + axis];
        if (value < target[targetOffset + axis]) target[targetOffset + axis] = value;
        if (value > target[targetOffset + 3 + axis]) target[targetOffset + 3 + axis] = value;
    }
}

/** `growBox` with the source box already in registers: binning reads each primitive's box once for all three axes. */
function growBinBox(
    target: Float32Array,
    offset: number,
    lowX: number,
    lowY: number,
    lowZ: number,
    highX: number,
    highY: number,
    highZ: number,
): void {
    if (lowX < target[offset]) target[offset] = lowX;
    if (lowY < target[offset + 1]) target[offset + 1] = lowY;
    if (lowZ < target[offset + 2]) target[offset + 2] = lowZ;
    if (highX > target[offset + 3]) target[offset + 3] = highX;
    if (highY > target[offset + 4]) target[offset + 4] = highY;
    if (highZ > target[offset + 5]) target[offset + 5] = highZ;
}

function copyBox(target: Float32Array, targetOffset: number, source: Float32Array, sourceOffset: number): void {
    for (let i = 0; i < BOX_FLOATS; i++) target[targetOffset + i] = source[sourceOffset + i];
}

/** Surface area from a box's extents. An empty box (no primitives yet) has negative extents and area 0. */
function areaOfExtents(dx: number, dy: number, dz: number): number {
    if (dx < 0 || dy < 0 || dz < 0) return 0;
    return 2 * (dx * dy + dy * dz + dz * dx);
}

function surfaceArea(boxes: Float32Array, offset: number): number {
    return areaOfExtents(boxes[offset + 3] - boxes[offset], boxes[offset + 4] - boxes[offset + 1], boxes[offset + 5] - boxes[offset + 2]);
}

// ---- Split plans ---------------------------------------------------------------------------

/** How a node's primitive range is divided between its two children. */
const SPLIT_KIND = {
    /** Centroids in bins below `boundaryBin` on `axis` go left. */
    bins: 0,
    /** No plane separates the centroids: split the list in half so the tree still terminates. */
    median: 1,
} as const;

/** The plan for one node: where to split, what it costs, and the two child boxes it gives. */
class SplitResult {
    /** False when no plane separates the centroids; the split is then a median split. */
    found = false;
    axis = 0;
    /** First bin that goes to the right child. */
    boundaryBin = 0;
    /** Smallest centroid on `axis` over the range being split. */
    centroidMin = 0;
    /** Bins per unit of centroid distance on `axis`. */
    binScale = 0;
    cost = Infinity;
    /** `[left box, right box]`. For a binned split, the unions of the bins on each side of the boundary. */
    readonly childBoxes = new Float32Array(2 * BOX_FLOATS);
}

/**
 * Nodes waiting to be split, as a stack in flat arrays so the build
 * allocates nothing per node. An entry is a node index, its primitive range,
 * its depth and the split to apply, with both child boxes.
 */
class PendingStack {
    /** nodeIndex, start, end, depth, kind, axis, boundaryBin */
    static readonly INTEGERS = 7;
    /** centroidMin, binScale (kept in double precision, as the split search computed them) */
    static readonly SCALARS = 2;
    /** left box, right box */
    static readonly BOXES = 2 * BOX_FLOATS;

    /** Entries currently on the stack. */
    size = 0;
    private capacity = 64;
    private integers = new Int32Array(this.capacity * PendingStack.INTEGERS);
    private scalars = new Float64Array(this.capacity * PendingStack.SCALARS);
    private boxes = new Float32Array(this.capacity * PendingStack.BOXES);

    push(nodeIndex: number, start: number, end: number, depth: number, split: SplitResult): void {
        if (this.size === this.capacity) this.grow();
        const slot = this.size++;
        const i = slot * PendingStack.INTEGERS;
        this.integers[i] = nodeIndex;
        this.integers[i + 1] = start;
        this.integers[i + 2] = end;
        this.integers[i + 3] = depth;
        this.integers[i + 4] = split.found ? SPLIT_KIND.bins : SPLIT_KIND.median;
        this.integers[i + 5] = split.axis;
        this.integers[i + 6] = split.boundaryBin;
        this.scalars[slot * PendingStack.SCALARS] = split.centroidMin;
        this.scalars[slot * PendingStack.SCALARS + 1] = split.binScale;
        this.boxes.set(split.childBoxes, slot * PendingStack.BOXES);
    }

    /** Removes the top entry and returns its slot. The slot stays readable until the next push. */
    pop(): number {
        return --this.size;
    }

    private grow(): void {
        this.capacity *= 2;
        const integers = new Int32Array(this.capacity * PendingStack.INTEGERS);
        integers.set(this.integers);
        this.integers = integers;
        const scalars = new Float64Array(this.capacity * PendingStack.SCALARS);
        scalars.set(this.scalars);
        this.scalars = scalars;
        const boxes = new Float32Array(this.capacity * PendingStack.BOXES);
        boxes.set(this.boxes);
        this.boxes = boxes;
    }

    nodeIndex(slot: number): number {
        return this.integers[slot * PendingStack.INTEGERS];
    }
    start(slot: number): number {
        return this.integers[slot * PendingStack.INTEGERS + 1];
    }
    end(slot: number): number {
        return this.integers[slot * PendingStack.INTEGERS + 2];
    }
    depth(slot: number): number {
        return this.integers[slot * PendingStack.INTEGERS + 3];
    }
    kind(slot: number): number {
        return this.integers[slot * PendingStack.INTEGERS + 4];
    }
    axis(slot: number): number {
        return this.integers[slot * PendingStack.INTEGERS + 5];
    }
    boundaryBin(slot: number): number {
        return this.integers[slot * PendingStack.INTEGERS + 6];
    }
    centroidMin(slot: number): number {
        return this.scalars[slot * PendingStack.SCALARS];
    }
    binScale(slot: number): number {
        return this.scalars[slot * PendingStack.SCALARS + 1];
    }
    /** Copies the entry's `[left box, right box]` into `target`. */
    copyChildBoxes(slot: number, target: Float32Array): void {
        const first = slot * PendingStack.BOXES;
        for (let i = 0; i < PendingStack.BOXES; i++) target[i] = this.boxes[first + i];
    }
}

// ---- The builder ---------------------------------------------------------------------------

class SahTreeBuilder {
    private readonly primitiveCount: number;
    private readonly binCount: number;
    /** Primitive boxes, kept in the same order as `order` (six floats each). */
    private readonly boxes: Float32Array;
    /** Primitive centroids, kept in the same order as `order` (three floats each). */
    private readonly centroids: Float32Array;
    private readonly order: Uint32Array;

    private readonly nodeBuffer: ArrayBuffer;
    private readonly nodeFloats: Float32Array;
    private readonly nodeWords: Uint32Array;
    private readonly parents: Uint32Array;
    private nodeCount = 0;
    private maxDepth = 0;
    private leafCount = 0;

    // Scratch space reused by every split search. Bins are per axis: x's bins, then y's, then z's.
    private readonly binCounts: Uint32Array;
    private readonly binBoxes: Float32Array;
    /** `binBoxes` with every box empty, copied over it to reset all bins in one call. */
    private readonly emptyBinBoxes: Float32Array;
    private readonly rightBoxes: Float32Array;
    private readonly rightAreas: Float64Array;
    private readonly rightCounts: Uint32Array;
    /** The bins of one axis that hold primitives, in increasing order. */
    private readonly usedBins: Uint32Array;
    /** Centroid bounds of the left and right part after a partition, `[left, right]`. */
    private readonly sideCentroidBounds = new Float32Array(2 * BOX_FLOATS);

    constructor(
        primitiveBounds: Float32Array,
        private readonly options: SahBuildOptions,
    ) {
        this.primitiveCount = primitiveBounds.length / 6;
        this.binCount = options.binCount;
        this.boxes = Float32Array.from(primitiveBounds);
        this.centroids = new Float32Array(this.primitiveCount * 3);
        this.order = new Uint32Array(this.primitiveCount);

        // A binary tree whose leaves hold at least one primitive has at most
        // (primitives - 1) inner nodes; a lone primitive still needs a root.
        const maxNodes = Math.max(1, this.primitiveCount - 1);
        this.nodeBuffer = new ArrayBuffer(maxNodes * BVH_NODE_BYTES);
        this.nodeFloats = new Float32Array(this.nodeBuffer);
        this.nodeWords = new Uint32Array(this.nodeBuffer);
        this.parents = new Uint32Array(maxNodes);

        this.binCounts = new Uint32Array(3 * this.binCount);
        this.binBoxes = new Float32Array(3 * this.binCount * BOX_FLOATS);
        this.emptyBinBoxes = new Float32Array(3 * this.binCount * BOX_FLOATS);
        for (let bin = 0; bin < 3 * this.binCount; bin++) resetBox(this.emptyBinBoxes, bin * BOX_FLOATS);
        this.rightBoxes = new Float32Array(this.binCount * BOX_FLOATS);
        this.rightAreas = new Float64Array(this.binCount);
        this.rightCounts = new Uint32Array(this.binCount);
        this.usedBins = new Uint32Array(this.binCount);
    }

    build(): SahTree {
        const rootCentroidBounds = new Float32Array(BOX_FLOATS);
        this.prepareCentroids(rootCentroidBounds);
        const rootBox = new Float32Array(BOX_FLOATS);
        resetBox(rootBox, 0);
        for (let position = 0; position < this.primitiveCount; position++) {
            growBox(rootBox, 0, this.boxes, position * BOX_FLOATS);
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
        const rootSplit = new SplitResult();
        this.findBestSplit(0, this.primitiveCount, rootBox, 0, rootCentroidBounds, 0, rootSplit);
        if (!rootSplit.found) this.planMedianSplit(0, this.primitiveCount, rootSplit);
        this.parents[0] = NO_PARENT;
        this.nodeCount = 1;
        const pending = new PendingStack();
        pending.push(0, 0, this.primitiveCount, 1, rootSplit);
        this.processPending(pending);
        return this.finish(rootBox);
    }

    private prepareCentroids(bounds: Float32Array): void {
        resetBox(bounds, 0);
        for (let primitive = 0; primitive < this.primitiveCount; primitive++) {
            this.order[primitive] = primitive;
            for (let axis = 0; axis < 3; axis++) {
                const low = this.boxes[primitive * 6 + axis];
                const high = this.boxes[primitive * 6 + 3 + axis];
                const centroid = 0.5 * (low + high);
                if (!Number.isFinite(centroid) || !Number.isFinite(low) || !Number.isFinite(high)) {
                    throw new RangeError(`primitive ${primitive} has non-finite bounds`);
                }
                this.centroids[primitive * 3 + axis] = centroid;
            }
            growBoxByPoint(bounds, 0, this.centroids, primitive * 3);
        }
    }

    private buildSinglePrimitiveTree(rootBox: Float32Array): void {
        this.nodeCount = 1;
        this.parents[0] = NO_PARENT;
        this.maxDepth = 1;
        this.writeChild(0, 'left', rootBox, 0, 0, 1);
        this.leafCount = 1;
        this.writeEmptyChild(0, 'right');
    }

    /** Splits pending nodes until every primitive sits in a leaf. */
    private processPending(pending: PendingStack): void {
        const childBoxes = new Float32Array(2 * BOX_FLOATS);
        const leftSplit = new SplitResult();
        const rightSplit = new SplitResult();
        const centroidBounds = this.sideCentroidBounds;
        while (pending.size > 0) {
            const slot = pending.pop();
            const nodeIndex = pending.nodeIndex(slot);
            const start = pending.start(slot);
            const end = pending.end(slot);
            const depth = pending.depth(slot);
            this.maxDepth = Math.max(this.maxDepth, depth);

            const middle = this.partition(pending, slot, start, end);
            pending.copyChildBoxes(slot, childBoxes);

            const leftIsLeaf = this.planChild(start, middle, childBoxes, 0, centroidBounds, 0, leftSplit);
            const rightIsLeaf = this.planChild(middle, end, childBoxes, BOX_FLOATS, centroidBounds, BOX_FLOATS, rightSplit);

            // Inner children are numbered left first, so two inner children sit
            // next to each other in memory.
            let leftChild = -1;
            let rightChild = -1;
            if (leftIsLeaf) {
                this.writeChild(nodeIndex, 'left', childBoxes, 0, start, middle - start);
                this.leafCount++;
            } else {
                leftChild = this.addInnerChild(nodeIndex, 'left', childBoxes, 0);
            }
            if (rightIsLeaf) {
                this.writeChild(nodeIndex, 'right', childBoxes, BOX_FLOATS, middle, end - middle);
                this.leafCount++;
            } else {
                rightChild = this.addInnerChild(nodeIndex, 'right', childBoxes, BOX_FLOATS);
            }
            // Right first, so the left child is split next (the stack pops the last pushed).
            if (rightChild >= 0) pending.push(rightChild, middle, end, depth + 1, rightSplit);
            if (leftChild >= 0) pending.push(leftChild, start, middle, depth + 1, leftSplit);
        }
    }

    private addInnerChild(nodeIndex: number, side: 'left' | 'right', boxes: Float32Array, boxOffset: number): number {
        const childIndex = this.nodeCount++;
        this.parents[childIndex] = nodeIndex;
        this.writeChild(nodeIndex, side, boxes, boxOffset, childIndex, 0);
        return childIndex;
    }

    private finish(bounds: Float32Array): SahTree {
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

    /**
     * Leaf or inner node for the primitives `[start, end)`? Returns true for a
     * leaf. For an inner node, `split` receives the split to use.
     */
    private planChild(
        start: number,
        end: number,
        boxes: Float32Array,
        boxOffset: number,
        centroidBounds: Float32Array,
        centroidOffset: number,
        split: SplitResult,
    ): boolean {
        const count = end - start;
        if (count === 1) return true;
        this.findBestSplit(start, end, boxes, boxOffset, centroidBounds, centroidOffset, split);
        if (count <= this.options.maxLeafPrimitives) {
            const leafCost = this.options.intersectionCost * count;
            if (!split.found || leafCost <= split.cost) return true;
        }
        // Too many primitives for a leaf, or splitting is cheaper.
        if (!split.found) this.planMedianSplit(start, end, split);
        return false;
    }

    /** No plane separates the centroids: cut the list in half so the tree still terminates. */
    private planMedianSplit(start: number, end: number, split: SplitResult): void {
        const middle = (start + end) >>> 1;
        this.computeBounds(start, middle, split.childBoxes, 0);
        this.computeBounds(middle, end, split.childBoxes, BOX_FLOATS);
    }

    /**
     * The cheapest binned split of `[start, end)`, written to `split`
     * (`split.found` is false when no plane separates the centroids).
     */
    private findBestSplit(
        start: number,
        end: number,
        parentBoxes: Float32Array,
        parentOffset: number,
        centroidBounds: Float32Array,
        centroidOffset: number,
        split: SplitResult,
    ): void {
        const { traversalCost, intersectionCost } = this.options;
        const binCount = this.binCount;
        const lastBin = binCount - 1;
        const total = end - start;
        const parentArea = surfaceArea(parentBoxes, parentOffset);
        split.found = false;
        split.cost = Infinity;

        // How centroids map to bins on each axis. 0 marks an axis with no extent, which cannot be split.
        const scaleX = this.binScaleOf(centroidBounds, centroidOffset, 0);
        const scaleY = this.binScaleOf(centroidBounds, centroidOffset, 1);
        const scaleZ = this.binScaleOf(centroidBounds, centroidOffset, 2);
        if (scaleX === 0 && scaleY === 0 && scaleZ === 0) return;
        const originX = centroidBounds[centroidOffset];
        const originY = centroidBounds[centroidOffset + 1];
        const originZ = centroidBounds[centroidOffset + 2];

        // Fill the bins of every splittable axis in one pass over the primitives.
        const binCounts = this.binCounts;
        const binBoxes = this.binBoxes;
        binCounts.fill(0);
        binBoxes.set(this.emptyBinBoxes);
        const centroids = this.centroids;
        const boxes = this.boxes;
        for (let position = start; position < end; position++) {
            const c = position * 3;
            const b = position * BOX_FLOATS;
            const lowX = boxes[b];
            const lowY = boxes[b + 1];
            const lowZ = boxes[b + 2];
            const highX = boxes[b + 3];
            const highY = boxes[b + 4];
            const highZ = boxes[b + 5];
            // The bin formula is the one in `binOf`, so binning and partitioning always agree.
            if (scaleX > 0) {
                const bin = Math.min(lastBin, Math.max(0, Math.floor((centroids[c] - originX) * scaleX)));
                binCounts[bin]++;
                growBinBox(binBoxes, bin * BOX_FLOATS, lowX, lowY, lowZ, highX, highY, highZ);
            }
            if (scaleY > 0) {
                const bin = binCount + Math.min(lastBin, Math.max(0, Math.floor((centroids[c + 1] - originY) * scaleY)));
                binCounts[bin]++;
                growBinBox(binBoxes, bin * BOX_FLOATS, lowX, lowY, lowZ, highX, highY, highZ);
            }
            if (scaleZ > 0) {
                const bin = 2 * binCount + Math.min(lastBin, Math.max(0, Math.floor((centroids[c + 2] - originZ) * scaleZ)));
                binCounts[bin]++;
                growBinBox(binBoxes, bin * BOX_FLOATS, lowX, lowY, lowZ, highX, highY, highZ);
            }
        }

        // Only bins that hold primitives are swept. An empty bin changes neither
        // side, so every boundary next to it scores the same as the boundary
        // just after the last non-empty bin below it, and the search keeps the
        // first of equal costs. Skipping them therefore picks the same split as
        // sweeping all 16 bins, and most nodes are small enough that most bins are empty.
        const usedBins = this.usedBins;
        const rightBoxes = this.rightBoxes;
        const rightAreas = this.rightAreas;
        const rightCounts = this.rightCounts;
        for (let axis = 0; axis < 3; axis++) {
            const binScale = axis === 0 ? scaleX : axis === 1 ? scaleY : scaleZ;
            if (binScale === 0) continue;
            const firstBin = axis * binCount;
            let usedCount = 0;
            for (let bin = 0; bin < binCount; bin++) {
                if (binCounts[firstBin + bin] > 0) usedBins[usedCount++] = bin;
            }
            if (usedCount < 2) continue;
            const lastUsed = usedCount - 1;

            // Sweep from the right so each boundary knows its right side...
            copyBox(rightBoxes, lastUsed * BOX_FLOATS, binBoxes, (firstBin + usedBins[lastUsed]) * BOX_FLOATS);
            rightCounts[lastUsed] = binCounts[firstBin + usedBins[lastUsed]];
            rightAreas[lastUsed] = surfaceArea(rightBoxes, lastUsed * BOX_FLOATS);
            for (let used = lastUsed - 1; used >= 1; used--) {
                copyBox(rightBoxes, used * BOX_FLOATS, rightBoxes, (used + 1) * BOX_FLOATS);
                growBox(rightBoxes, used * BOX_FLOATS, binBoxes, (firstBin + usedBins[used]) * BOX_FLOATS);
                rightCounts[used] = rightCounts[used + 1] + binCounts[firstBin + usedBins[used]];
                rightAreas[used] = surfaceArea(rightBoxes, used * BOX_FLOATS);
            }

            // ...then from the left, scoring each boundary. The left box lives in local variables.
            let minX = Infinity;
            let minY = Infinity;
            let minZ = Infinity;
            let maxX = -Infinity;
            let maxY = -Infinity;
            let maxZ = -Infinity;
            let leftCount = 0;
            for (let used = 0; used < lastUsed; used++) {
                const binBox = (firstBin + usedBins[used]) * BOX_FLOATS;
                leftCount += binCounts[firstBin + usedBins[used]];
                if (binBoxes[binBox] < minX) minX = binBoxes[binBox];
                if (binBoxes[binBox + 1] < minY) minY = binBoxes[binBox + 1];
                if (binBoxes[binBox + 2] < minZ) minZ = binBoxes[binBox + 2];
                if (binBoxes[binBox + 3] > maxX) maxX = binBoxes[binBox + 3];
                if (binBoxes[binBox + 4] > maxY) maxY = binBoxes[binBox + 4];
                if (binBoxes[binBox + 5] > maxZ) maxZ = binBoxes[binBox + 5];
                const rightCount = rightCounts[used + 1];

                const leftArea = areaOfExtents(maxX - minX, maxY - minY, maxZ - minZ);
                const rightArea = rightAreas[used + 1];
                // Flat or point-like parents have no area to compare, so fall
                // back to the share of primitives each side holds.
                const expectedTests =
                    parentArea > 0
                        ? (leftArea * leftCount + rightArea * rightCount) / parentArea
                        : (leftCount * leftCount + rightCount * rightCount) / total;
                const cost = traversalCost + intersectionCost * expectedTests;
                if (cost < split.cost) {
                    split.found = true;
                    split.cost = cost;
                    split.axis = axis;
                    split.boundaryBin = usedBins[used] + 1;
                    split.centroidMin = centroidBounds[centroidOffset + axis];
                    split.binScale = binScale;
                    const childBoxes = split.childBoxes;
                    childBoxes[0] = minX;
                    childBoxes[1] = minY;
                    childBoxes[2] = minZ;
                    childBoxes[3] = maxX;
                    childBoxes[4] = maxY;
                    childBoxes[5] = maxZ;
                    copyBox(childBoxes, BOX_FLOATS, rightBoxes, (used + 1) * BOX_FLOATS);
                }
            }
        }
    }

    /** Bins per unit of centroid distance on `axis`, or 0 when the centroids have no extent there. */
    private binScaleOf(centroidBounds: Float32Array, offset: number, axis: number): number {
        const extent = centroidBounds[offset + 3 + axis] - centroidBounds[offset + axis];
        if (!(extent > 0)) return 0;
        const binScale = this.binCount / extent;
        return Number.isFinite(binScale) ? binScale : 0;
    }

    private binOf(position: number, axis: number, centroidMin: number, binScale: number): number {
        const centroid = this.centroids[position * 3 + axis];
        const bin = Math.floor((centroid - centroidMin) * binScale);
        return Math.min(this.binCount - 1, Math.max(0, bin));
    }

    /**
     * Reorders `[start, end)` (order, boxes and centroids together) so the left
     * child's primitives come first, and returns the first right position.
     * Also writes the two sides' centroid bounds to `this.sideCentroidBounds`.
     */
    private partition(pending: PendingStack, slot: number, start: number, end: number): number {
        const sides = this.sideCentroidBounds;
        resetBox(sides, 0);
        resetBox(sides, BOX_FLOATS);

        if (pending.kind(slot) === SPLIT_KIND.bins) {
            const axis = pending.axis(slot);
            const boundaryBin = pending.boundaryBin(slot);
            const centroidMin = pending.centroidMin(slot);
            const binScale = pending.binScale(slot);
            let left = start;
            let right = end - 1;
            while (left <= right) {
                if (this.binOf(left, axis, centroidMin, binScale) < boundaryBin) {
                    growBoxByPoint(sides, 0, this.centroids, left * 3);
                    left++;
                } else {
                    this.swap(left, right);
                    growBoxByPoint(sides, BOX_FLOATS, this.centroids, right * 3);
                    right--;
                }
            }
            // The cost search skipped boundaries with an empty side, so both are non-empty here.
            if (left > start && left < end) return left;
            throw new Error('unreachable: a binned split left one side empty');
        }

        // Median split: the order stays as it is; only the centroid bounds are needed.
        const middle = (start + end) >>> 1;
        for (let position = start; position < middle; position++) growBoxByPoint(sides, 0, this.centroids, position * 3);
        for (let position = middle; position < end; position++) growBoxByPoint(sides, BOX_FLOATS, this.centroids, position * 3);
        return middle;
    }

    private swap(a: number, b: number): void {
        const order = this.order;
        const primitive = order[a];
        order[a] = order[b];
        order[b] = primitive;
        const centroids = this.centroids;
        for (let i = 0; i < 3; i++) {
            const value = centroids[a * 3 + i];
            centroids[a * 3 + i] = centroids[b * 3 + i];
            centroids[b * 3 + i] = value;
        }
        const boxes = this.boxes;
        for (let i = 0; i < BOX_FLOATS; i++) {
            const value = boxes[a * BOX_FLOATS + i];
            boxes[a * BOX_FLOATS + i] = boxes[b * BOX_FLOATS + i];
            boxes[b * BOX_FLOATS + i] = value;
        }
    }

    private computeBounds(start: number, end: number, out: Float32Array, offset: number): void {
        resetBox(out, offset);
        for (let position = start; position < end; position++) {
            growBox(out, offset, this.boxes, position * BOX_FLOATS);
        }
    }

    private writeChild(
        nodeIndex: number,
        side: 'left' | 'right',
        boxes: Float32Array,
        boxOffset: number,
        index: number,
        count: number,
    ): void {
        const base = nodeIndex * BVH_NODE_WORDS;
        const left = side === 'left';
        const minWord = base + (left ? NODE_WORD.leftMin : NODE_WORD.rightMin);
        const maxWord = base + (left ? NODE_WORD.leftMax : NODE_WORD.rightMax);
        for (let axis = 0; axis < 3; axis++) {
            this.nodeFloats[minWord + axis] = boxes[boxOffset + axis];
            this.nodeFloats[maxWord + axis] = boxes[boxOffset + 3 + axis];
        }
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
