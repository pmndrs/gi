/**
 * The *shape* of the top-level tree, built on the CPU (off the main thread,
 * see top-level-async.ts) for the GPU refit (top-level-gpu.ts, stage B1a).
 *
 * The shape is the tree's nodes (child indices, plus boxes as of the build),
 * which record holds which instance, and what the GPU refit needs to climb
 * the tree. Its boxes are only a starting point: the refit recomputes every
 * box from the instances' current transforms in the same frame the shape is
 * swapped in, so the build may use boxes that are a little stale.
 *
 * **When to build a new one, without reading anything back from the GPU.**
 * The platform promises hosts no readbacks frame after frame, so the refit's own quality
 * numbers stay on the GPU. Instead {@link TopLevelShapeKeeper} (one per
 * worker) remembers the last shape it built, and a *check* request refits
 * that shape on the CPU from the current transforms (`refitOnCpu`, the same
 * unions as the GPU) and builds a new shape only when its relative SAH cost
 * has passed the given ratio of its as-built cost.
 */

import { halfArea, leafSlotsOf, parentSlotsOf, PRIMITIVE_BOX_FLOATS, refitOnCpu, relativeSahCost } from '@pmndrs/bvh/refit-cpu';
import { BVH_NODE_BYTES } from './layout.ts';
import { buildSahTree, DEFAULT_SAH_OPTIONS } from './sah-builder.ts';
import { transformBounds } from './top-level.ts';

/** What a top-level build needs per instance, as flat arrays that can move to a worker. */
export interface TopLevelBuildInput {
    /** The scene's instance slot of each instance. */
    readonly slots: Uint32Array;
    /** 1 when the instance is visible. Hidden instances are in the shape too, with empty boxes while hidden. */
    readonly visible: Uint8Array;
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
    /** The scene instance slot each record holds. */
    readonly recordSlots: Uint32Array;
    /** Parent of each node as a child slot (`parent * 2 + side`); see `@pmndrs/bvh` gpu-refit.ts. */
    readonly parentSlots: Uint32Array;
    /** Every leaf child slot, in record order. */
    readonly leafSlots: Uint32Array;
    /** `relativeSahCost` of the tree as built: the baseline its quality is compared with. */
    readonly relativeCost: number;
}

/** The answer to a quality check. */
export interface TopLevelCheck {
    /** A new shape when the old one's cost passed the ratio (or there was none); null to keep the old one. */
    readonly shape: TopLevelShape | null;
    /** Relative SAH cost of the old shape refit to the current transforms (0 without an old shape). */
    readonly relativeCost: number;
    /** The old shape's cost as built. */
    readonly builtCost: number;
}

/** The world box of each input instance, six floats each. */
function worldBoxes(input: TopLevelBuildInput): Float32Array {
    const count = input.slots.length;
    const boxes = new Float32Array(count * 6);
    for (let instance = 0; instance < count; instance++) {
        transformBounds(input.meshBounds.subarray(instance * 6, instance * 6 + 6), input.objectToWorld.subarray(instance * 16, instance * 16 + 16), boxes, instance * 6);
    }
    return boxes;
}

/** Builds the shape over the instances' world boxes. One instance per leaf. */
export function buildTopLevelShape(input: TopLevelBuildInput): TopLevelShape {
    return buildShapeKeepingParents(input).shape;
}

function buildShapeKeepingParents(input: TopLevelBuildInput): { shape: TopLevelShape; parents: Uint32Array } {
    // One instance per leaf, as the CPU top level: the instance's own tree is where the real work is.
    const tree = buildSahTree(worldBoxes(input), { ...DEFAULT_SAH_OPTIONS, maxLeafPrimitives: 1 });
    const words = new Uint32Array(tree.nodes, 0, (tree.nodeCount * BVH_NODE_BYTES) / 4);
    const floats = new Float32Array(tree.nodes, 0, words.length);
    const recordSlots = Uint32Array.from(tree.order, (inputIndex) => input.slots[inputIndex]);
    const shape: TopLevelShape = {
        nodes: tree.nodes,
        nodeCount: tree.nodeCount,
        maxDepth: tree.maxDepth,
        recordSlots,
        parentSlots: parentSlotsOf(words, 0, tree.nodeCount, tree.parents),
        leafSlots: leafSlotsOf(words, 0, tree.nodeCount),
        relativeCost: relativeSahCost(floats, words, 0, tree.nodeCount),
    };
    return { shape, parents: tree.parents };
}

/** Shapes a keeper remembers. Old ones go first; the main thread only ever asks about recent ones. */
const KEPT_SHAPES = 4;

interface KeptShape {
    readonly nodes: ArrayBuffer;
    readonly parents: Uint32Array;
    readonly nodeCount: number;
    readonly recordSlots: Uint32Array;
    readonly builtCost: number;
}

/**
 * Builds shapes and remembers the last few, by generation, so it can judge
 * the one the GPU actually uses against new transforms. A shape the main
 * thread could not swap in (it threw) is never the one judged, because a
 * check names the generation in use. Lives in the shape worker (or on the
 * main thread when there is none); requests are handled one at a time.
 */
export class TopLevelShapeKeeper {
    private readonly kept = new Map<number, KeptShape>();

    /** Builds a shape and remembers it as `generation`. */
    build(input: TopLevelBuildInput, generation: number): TopLevelShape {
        const { shape, parents } = buildShapeKeepingParents(input);
        // A copy: the shape's own buffers move to the main thread.
        this.kept.set(generation, {
            nodes: shape.nodes.slice(0, shape.nodeCount * BVH_NODE_BYTES),
            parents,
            nodeCount: shape.nodeCount,
            recordSlots: shape.recordSlots.slice(),
            builtCost: shape.relativeCost,
        });
        for (const old of this.kept.keys()) {
            if (this.kept.size <= KEPT_SHAPES) break;
            this.kept.delete(old);
        }
        return shape;
    }

    /**
     * Refits shape `currentGeneration` (the one in use) to `input`'s
     * transforms, and builds a new one as `generation` if its cost passed
     * `ratio` times its as-built cost, or if that shape isn't known.
     */
    check(input: TopLevelBuildInput, ratio: number, currentGeneration: number, generation: number): TopLevelCheck {
        const kept = this.kept.get(currentGeneration);
        if (!kept || kept.nodeCount === 0) return { shape: this.build(input, generation), relativeCost: 0, builtCost: 0 };

        // Each record's current world box, as the GPU gather makes it: empty when hidden or gone.
        let largestSlot = 0;
        for (const slot of input.slots) largestSlot = Math.max(largestSlot, slot);
        const inputBySlot = new Int32Array(largestSlot + 1).fill(-1);
        input.slots.forEach((slot, index) => (inputBySlot[slot] = index));
        const recordCount = kept.recordSlots.length;
        const primitiveBoxes = new Float32Array(recordCount * PRIMITIVE_BOX_FLOATS);
        const box = new Float32Array(6);
        for (let record = 0; record < recordCount; record++) {
            const slot = kept.recordSlots[record];
            const index = slot < inputBySlot.length ? inputBySlot[slot] : -1;
            const at = record * PRIMITIVE_BOX_FLOATS;
            if (index < 0 || input.visible[index] === 0) {
                primitiveBoxes.set([Infinity, Infinity, Infinity], at);
                primitiveBoxes.set([-Infinity, -Infinity, -Infinity], at + 4);
                continue;
            }
            transformBounds(input.meshBounds.subarray(index * 6, index * 6 + 6), input.objectToWorld.subarray(index * 16, index * 16 + 16), box, 0);
            primitiveBoxes.set(box.subarray(0, 3), at);
            primitiveBoxes.set(box.subarray(3, 6), at + 4);
        }
        const nodes = kept.nodes.slice(0);
        const refit = refitOnCpu(new Float32Array(nodes), new Uint32Array(nodes), 0, kept.nodeCount, kept.parents, primitiveBoxes);
        const rootArea = halfArea(refit.rootBox);
        const relativeCost = rootArea > 0 ? refit.summedHalfArea / rootArea : 0;
        const decayed = kept.builtCost > 0 && relativeCost > ratio * kept.builtCost;
        return { shape: decayed ? this.build(input, generation) : null, relativeCost, builtCost: kept.builtCost };
    }
}

/** The input buffers, which move to the worker with the request. */
export function topLevelInputTransferables(input: TopLevelBuildInput): Transferable[] {
    return [input.slots.buffer, input.visible.buffer, input.objectToWorld.buffer, input.meshBounds.buffer];
}

/** The buffers to move (not copy) when a shape crosses a worker boundary. */
export function topLevelShapeTransferables(shape: TopLevelShape): Transferable[] {
    return [shape.nodes, shape.recordSlots.buffer, shape.parentSlots.buffer, shape.leafSlots.buffer];
}
