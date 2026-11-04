/**
 * The *shape* of the top-level tree, built on the CPU (off the main thread,
 * see top-level-async.ts) for the GPU refit (top-level-gpu.ts, stage B1a).
 *
 * The shape is the tree's nodes (child indices, plus boxes as of the build),
 * which record holds which instance, and what the GPU refit needs to climb
 * the tree. Its boxes are only a starting point: the refit recomputes every
 * box from the instances' current transforms in the same frame the shape is
 * swapped in, so the build may use boxes that are a little stale.
 */

import { leafSlotsOf, parentSlotsOf, relativeSahCost } from '@pmndrs/bvh/refit-cpu';
import { BVH_NODE_BYTES } from './layout.ts';
import { buildSahTree, DEFAULT_SAH_OPTIONS } from './sah-builder.ts';
import { transformBounds } from './top-level.ts';

/** What a top-level build needs per instance, as flat arrays that can move to a worker. */
export interface TopLevelBuildInput {
    /** Object-to-world, 16 numbers per instance (column-major 4x4). */
    readonly objectToWorld: Float64Array;
    /** The instance's mesh box in object space, six floats per instance (`[min, max]`). */
    readonly meshBounds: Float32Array;
}

export interface TopLevelShape {
    /** `nodeCount` nodes in the tracer's layout. A leaf's `index` is a record index. */
    readonly nodes: ArrayBuffer;
    readonly nodeCount: number;
    /** Levels of nodes on the longest path (bounds the traversal stack). */
    readonly maxDepth: number;
    /** `order[record]` is the position in the build's input of the instance that record holds. */
    readonly order: Uint32Array;
    /** Parent of each node as a child slot (`parent * 2 + side`); see `@pmndrs/bvh` gpu-refit.ts. */
    readonly parentSlots: Uint32Array;
    /** Every leaf child slot, in record order. */
    readonly leafSlots: Uint32Array;
    /** `relativeSahCost` of the tree as built: the baseline the refit tree's quality is compared with. */
    readonly relativeCost: number;
}

/** Builds the shape over the instances' world boxes. One instance per leaf. */
export function buildTopLevelShape(input: TopLevelBuildInput): TopLevelShape {
    const instanceCount = input.meshBounds.length / 6;
    const boxes = new Float32Array(instanceCount * 6);
    for (let instance = 0; instance < instanceCount; instance++) {
        transformBounds(
            input.meshBounds.subarray(instance * 6, instance * 6 + 6),
            input.objectToWorld.subarray(instance * 16, instance * 16 + 16),
            boxes,
            instance * 6,
        );
    }
    // One instance per leaf, as the CPU top level: the instance's own tree is where the real work is.
    const tree = buildSahTree(boxes, { ...DEFAULT_SAH_OPTIONS, maxLeafPrimitives: 1 });
    const words = new Uint32Array(tree.nodes, 0, (tree.nodeCount * BVH_NODE_BYTES) / 4);
    const floats = new Float32Array(tree.nodes, 0, words.length);
    return {
        nodes: tree.nodes,
        nodeCount: tree.nodeCount,
        maxDepth: tree.maxDepth,
        order: tree.order,
        parentSlots: parentSlotsOf(words, 0, tree.nodeCount, tree.parents),
        leafSlots: leafSlotsOf(words, 0, tree.nodeCount),
        relativeCost: relativeSahCost(floats, words, 0, tree.nodeCount),
    };
}

/** The input buffers, which move to the worker with the request. */
export function topLevelInputTransferables(input: TopLevelBuildInput): Transferable[] {
    return [input.objectToWorld.buffer, input.meshBounds.buffer];
}

/** The buffers to move (not copy) when a shape crosses a worker boundary. */
export function topLevelShapeTransferables(shape: TopLevelShape): Transferable[] {
    return [shape.nodes, shape.order.buffer, shape.parentSlots.buffer, shape.leafSlots.buffer];
}
