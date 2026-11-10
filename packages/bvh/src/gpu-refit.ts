/**
 * Bottom-up refit of a BVH in the tracer's GPU layout (gpu-layout.ts), on the
 * GPU: the tree keeps its shape and every child box is recomputed from the
 * primitives' current boxes. Used for the top-level tree (stage B1a), whose
 * primitives are instances that move every frame; the same kernels refit a
 * mesh tree's triangles later (B1).
 *
 * **How it works.** Three dispatches in the caller's compute pass:
 *
 * 1. `bvhRefitClear`, one thread per node: zeroes the node's arrival counter
 *    and marks both child boxes "unwritten".
 * 2. `bvhRefitClimb`, one thread per leaf slot: the thread computes its
 *    leaf's box (the union of the leaf's primitives) and climbs. At each node
 *    it stores its box in its slot, then bumps the node's arrival counter. The
 *    first thread to arrive stops; the second knows both children are done,
 *    reads its sibling's box, takes the union and carries it to the parent.
 *    So every node is finished by exactly one thread, with no float atomics
 *    (WGSL has none) and no thread ever waiting on a workgroup that might not
 *    have started (WebGPU promises no such progress).
 * 3. `bvhRefitRepair`, one workgroup: returns at once, unless a read in the
 *    climb gave up (below). Then it refits the whole tree again, level by
 *    level from the deepest, with barriers between levels: slower, but it
 *    needs no cross-workgroup visibility at all. So a box built from an
 *    unwritten word is never kept, not even for one frame, and no readback
 *    is needed to find out.
 *
 * **Memory visibility.** WGSL atomics are relaxed, and plain stores from one
 * workgroup are not guaranteed visible to another within a dispatch. So the
 * boxes themselves are written and read with 32-bit `atomicStore` /
 * `atomicLoad` (no float maths on them, only bit copies), which every backend
 * performs at the coherence point. The relaxed order means the second thread
 * may in theory see the counter before its sibling's box; the clear pass
 * marks boxes "unwritten" so the reader can tell, and it re-reads (a bounded
 * spin: the box store was issued before the counter's, so it only has to
 * land). A spin that runs out is counted in the stats and triggers the repair
 * pass. The stats can be read back for diagnostics (`diagnostics: true`,
 * off by default: the platform promises hosts no readbacks frame after frame), which is how a
 * give-up reaches the platform's warnings and counters.
 *
 * **Subgroups** (DECISIONS 2026-11-04: written for subgroups first, with one
 * declared fallback). Builders store primitives in tree order, so two leaf
 * slots of the same node are often neighbouring threads. Before climbing,
 * each thread looks at its subgroup neighbours with `subgroupShuffleUp/Down`;
 * when the neighbour holds its sibling leaf, the pair finishes their node in
 * registers: no counter, no box round trip through memory, and half the
 * threads stop at once. The check compares node and side, never lane numbers,
 * so it is right for any subgroup size and any lane layout (it just pairs
 * less often). Devices without subgroups don't use this refit at all: the
 * tracer falls back to rebuilding the top level on the CPU.
 *
 * **Quality signal.** The climb also sums the half-areas of all inner nodes
 * (each node's area is added by the thread that finishes it, and partial sums
 * travel up with the boxes, so no float atomics here either). The root's sum
 * over its own area is `relativeSahCost` (refit-cpu.ts), reported in the
 * diagnostics stats. A caller that decides when to rebuild without
 * readbacks computes the same number on the CPU (`refitOnCpu`).
 *
 * **Empty boxes.** A primitive that must not be hit (a hidden instance, say)
 * has an empty box (`min > max`). Unions skip it, and an empty slot is stored
 * as a point at the largest f32: every ray's slab test puts that point at
 * `t >= 3.4e38` or behind it, which the tracer treats as a miss.
 */

import {
    GPU_LAYOUT_CHILD_STRIDE_WORDS,
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
} from './gpu-layout.ts';
import { BVH_FLOAT_MAX, levelOrderOf, PRIMITIVE_BOX_FLOATS } from './refit-cpu.ts';

/** Threads per workgroup of the clear and climb kernels. A multiple of every subgroup size up to 64. */
export const BVH_REFIT_WORKGROUP_SIZE = 64;

/** Threads of the repair pass's single workgroup: the default device's largest workgroup. */
const REPAIR_WORKGROUP_SIZE = 256;

/** Words of the uniform parameters (see `BvhRefitParams` in the WGSL). */
const PARAM_WORDS = 8;

/** Re-reads of a box word before a spin gives up and is counted (see the top of this file). */
const MAX_SPINS = 4096;

/** Written over a child box by the clear pass. A NaN bit pattern, so never a real box value. */
const UNWRITTEN = 0xffffffff;

/** Word positions in the stats buffer. */
export const BVH_REFIT_STAT = {
    rootLower: 0,
    summedHalfArea: 3,
    rootUpper: 4,
    /** The `generation` of the tree that was refit, echoed so a late readback can be matched to its tree. */
    generation: 7,
    /** Box reads that were still unwritten after the bounded spin, or climbs that didn't end. Must be 0. */
    incomplete: 8,
    /** Threads that finished the root: 1 per refit of a non-empty tree, 2 when the repair pass ran too. */
    rootsFinished: 9,
    /** 1 when the repair pass refit the tree again (because `incomplete` > 0). */
    repairs: 10,
} as const;
export const BVH_REFIT_STATS_BYTES = 64;

/** `GPUShaderStage.COMPUTE` and buffer usages; the globals don't exist in Node, where tests import this file. */
const SHADER_STAGE_COMPUTE = 0x4;
const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;
const MAP_MODE_READ = 0x1;

const hex = (value: number): string => `0x${value.toString(16)}u`;

/** WGSL for both kernels. Needs a device with the `subgroups` feature. */
export function bvhRefitWgsl(): string {
    const word = GPU_LAYOUT_NODE_WORD;
    const stat = BVH_REFIT_STAT;
    return /* wgsl */ `enable subgroups;

// ---- Bindings ----------------------------------------------------------------

struct BvhRefitParams {
    nodeWordOffset: u32,     // where node 0 starts in bvhTree, in 32-bit words
    nodeCount: u32,
    leafCount: u32,
    generation: u32,
    levelCount: u32,         // depth levels, for the repair pass
    levelStartsOffset: u32,  // where the level starts begin in bvhTopology
    levelNodesOffset: u32,   // where the nodes in level order begin in bvhTopology
    unused: u32,
}

// The tree, read and written as 32-bit words: boxes are copied as bits, never as floats.
@group(0) @binding(0) var<storage, read_write> bvhTree: array<atomic<u32>>;
// Two vec4s per primitive: (min.xyz, unused), (max.xyz, unused). Empty when min > max.
@group(0) @binding(1) var<storage, read> bvhPrimitiveBoxes: array<vec4f>;
// [parent slot of each node (node * 2 + side), or BVH_NO_PARENT], [each leaf slot],
// then for the repair pass [first index of each level, plus the end] and [nodes, deepest level first].
@group(0) @binding(2) var<storage, read> bvhTopology: array<u32>;
// [arrival counter per node] then [summed inner-node half-area below each child slot].
@group(0) @binding(3) var<storage, read_write> bvhScratch: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> bvhStats: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> bvhParams: BvhRefitParams;

// ---- Constants (generated from gpu-layout.ts) -----------------------------------

const BVH_WORKGROUP_SIZE: u32 = ${BVH_REFIT_WORKGROUP_SIZE}u;
const BVH_NODE_WORDS: u32 = ${GPU_LAYOUT_NODE_WORDS}u;
const BVH_CHILD_STRIDE_WORDS: u32 = ${GPU_LAYOUT_CHILD_STRIDE_WORDS}u;
const BVH_MAX_OFFSET_WORDS: u32 = ${word.leftMax - word.leftMin}u;   // from a child's min to its max
const BVH_LEFT_INDEX: u32 = ${word.leftIndex}u;
const BVH_RIGHT_INDEX: u32 = ${word.rightIndex}u;
const BVH_LEFT_COUNT: u32 = ${word.leftCount}u;
const BVH_RIGHT_COUNT: u32 = ${word.rightCount}u;
const BVH_EMPTY_CHILD: u32 = ${hex(GPU_LAYOUT_EMPTY_CHILD)};
const BVH_NO_PARENT: u32 = ${hex(GPU_LAYOUT_NO_PARENT)};
const BVH_UNWRITTEN: u32 = ${hex(UNWRITTEN)};
const BVH_MAX_SPINS: u32 = ${MAX_SPINS}u;
const BVH_PRIMITIVE_VEC4S: u32 = ${PRIMITIVE_BOX_FLOATS / 4}u;

// The largest finite f32 (${BVH_FLOAT_MAX}). An empty box is stored as a point here (see refit-cpu.ts).
const BVH_FLOAT_MAX: f32 = 0x1.fffffep+127f;

const BVH_STAT_ROOT_LOWER: u32 = ${stat.rootLower}u;
const BVH_STAT_SUMMED_AREA: u32 = ${stat.summedHalfArea}u;
const BVH_STAT_ROOT_UPPER: u32 = ${stat.rootUpper}u;
const BVH_STAT_GENERATION: u32 = ${stat.generation}u;
const BVH_STAT_INCOMPLETE: u32 = ${stat.incomplete}u;
const BVH_STAT_ROOTS_FINISHED: u32 = ${stat.rootsFinished}u;
const BVH_STAT_REPAIRS: u32 = ${stat.repairs}u;
const BVH_REPAIR_WORKGROUP_SIZE: u32 = ${REPAIR_WORKGROUP_SIZE}u;

// ---- Boxes --------------------------------------------------------------------

struct BvhBox {
    lower: vec3f,
    upper: vec3f,
}

fn bvhEmptyBox() -> BvhBox {
    return BvhBox(vec3f(BVH_FLOAT_MAX), vec3f(-BVH_FLOAT_MAX));
}

fn bvhUnion(a: BvhBox, b: BvhBox) -> BvhBox {
    return BvhBox(min(a.lower, b.lower), max(a.upper, b.upper));
}

fn bvhHalfArea(box: BvhBox) -> f32 {
    if (box.lower.x > box.upper.x) {
        return 0.0;
    }
    let size = box.upper - box.lower;
    return size.x * size.y + size.y * size.z + size.z * size.x;
}

fn bvhNodeWord(node: u32) -> u32 {
    return bvhParams.nodeWordOffset + node * BVH_NODE_WORDS;
}

// The first word of a child slot's box (its min.x).
fn bvhSlotWord(slot: u32) -> u32 {
    return bvhNodeWord(slot >> 1u) + (slot & 1u) * BVH_CHILD_STRIDE_WORDS;
}

fn bvhCostWord(slot: u32) -> u32 {
    return bvhParams.nodeCount + slot;
}

// ---- Reading and writing slots ---------------------------------------------------

fn bvhStoreSlot(slot: u32, box: BvhBox, belowArea: f32) {
    var lower = box.lower;
    var upper = box.upper;
    if (box.lower.x > box.upper.x) {
        // Empty: a point at the largest float, which no ray reaches.
        lower = vec3f(BVH_FLOAT_MAX);
        upper = vec3f(BVH_FLOAT_MAX);
    }
    let base = bvhSlotWord(slot);
    atomicStore(&bvhTree[base], bitcast<u32>(lower.x));
    atomicStore(&bvhTree[base + 1u], bitcast<u32>(lower.y));
    atomicStore(&bvhTree[base + 2u], bitcast<u32>(lower.z));
    atomicStore(&bvhTree[base + BVH_MAX_OFFSET_WORDS], bitcast<u32>(upper.x));
    atomicStore(&bvhTree[base + BVH_MAX_OFFSET_WORDS + 1u], bitcast<u32>(upper.y));
    atomicStore(&bvhTree[base + BVH_MAX_OFFSET_WORDS + 2u], bitcast<u32>(upper.z));
    atomicStore(&bvhScratch[bvhCostWord(slot)], bitcast<u32>(belowArea));
}

// One word of the tree, waiting (bounded) until the clear pass's mark is gone.
fn bvhWaitTreeWord(index: u32) -> u32 {
    var value = atomicLoad(&bvhTree[index]);
    for (var spins = 0u; value == BVH_UNWRITTEN; spins += 1u) {
        if (spins == BVH_MAX_SPINS) {
            atomicAdd(&bvhStats[BVH_STAT_INCOMPLETE], 1u);
            break;
        }
        value = atomicLoad(&bvhTree[index]);
    }
    return value;
}

fn bvhWaitCost(slot: u32) -> f32 {
    var value = atomicLoad(&bvhScratch[bvhCostWord(slot)]);
    for (var spins = 0u; value == BVH_UNWRITTEN; spins += 1u) {
        if (spins == BVH_MAX_SPINS) {
            atomicAdd(&bvhStats[BVH_STAT_INCOMPLETE], 1u);
            break;
        }
        value = atomicLoad(&bvhScratch[bvhCostWord(slot)]);
    }
    return bitcast<f32>(value);
}

// A stored box back as a box: the stored form of an empty box is a point at the largest float.
fn bvhStoredBox(lower: vec3f, upper: vec3f) -> BvhBox {
    if (lower.x == BVH_FLOAT_MAX) {
        return bvhEmptyBox();
    }
    return BvhBox(lower, upper);
}

// A sibling's box during the climb, waiting (bounded) for each word to be written.
fn bvhLoadSlot(slot: u32) -> BvhBox {
    let base = bvhSlotWord(slot);
    let lower = vec3f(
        bitcast<f32>(bvhWaitTreeWord(base)),
        bitcast<f32>(bvhWaitTreeWord(base + 1u)),
        bitcast<f32>(bvhWaitTreeWord(base + 2u)),
    );
    let upper = vec3f(
        bitcast<f32>(bvhWaitTreeWord(base + BVH_MAX_OFFSET_WORDS)),
        bitcast<f32>(bvhWaitTreeWord(base + BVH_MAX_OFFSET_WORDS + 1u)),
        bitcast<f32>(bvhWaitTreeWord(base + BVH_MAX_OFFSET_WORDS + 2u)),
    );
    return bvhStoredBox(lower, upper);
}

// A box the repair pass wrote in an earlier level (visible after the level's barrier).
fn bvhReadSlot(slot: u32) -> BvhBox {
    let base = bvhSlotWord(slot);
    let lower = vec3f(
        bitcast<f32>(atomicLoad(&bvhTree[base])),
        bitcast<f32>(atomicLoad(&bvhTree[base + 1u])),
        bitcast<f32>(atomicLoad(&bvhTree[base + 2u])),
    );
    let upper = vec3f(
        bitcast<f32>(atomicLoad(&bvhTree[base + BVH_MAX_OFFSET_WORDS])),
        bitcast<f32>(atomicLoad(&bvhTree[base + BVH_MAX_OFFSET_WORDS + 1u])),
        bitcast<f32>(atomicLoad(&bvhTree[base + BVH_MAX_OFFSET_WORDS + 2u])),
    );
    return bvhStoredBox(lower, upper);
}

// The union of a leaf slot's primitives.
fn bvhLeafBox(slot: u32) -> BvhBox {
    let base = bvhNodeWord(slot >> 1u);
    let isRight = (slot & 1u) == 1u;
    let first = atomicLoad(&bvhTree[base + select(BVH_LEFT_INDEX, BVH_RIGHT_INDEX, isRight)]);
    let count = atomicLoad(&bvhTree[base + select(BVH_LEFT_COUNT, BVH_RIGHT_COUNT, isRight)]);
    var box = bvhEmptyBox();
    for (var primitive = first; primitive < first + count; primitive += 1u) {
        let at = primitive * BVH_PRIMITIVE_VEC4S;
        box = bvhUnion(box, BvhBox(bvhPrimitiveBoxes[at].xyz, bvhPrimitiveBoxes[at + 1u].xyz));
    }
    return box;
}

fn bvhFinishRoot(box: BvhBox, summedArea: f32) {
    atomicStore(&bvhStats[BVH_STAT_ROOT_LOWER], bitcast<u32>(box.lower.x));
    atomicStore(&bvhStats[BVH_STAT_ROOT_LOWER + 1u], bitcast<u32>(box.lower.y));
    atomicStore(&bvhStats[BVH_STAT_ROOT_LOWER + 2u], bitcast<u32>(box.lower.z));
    atomicStore(&bvhStats[BVH_STAT_SUMMED_AREA], bitcast<u32>(summedArea));
    atomicStore(&bvhStats[BVH_STAT_ROOT_UPPER], bitcast<u32>(box.upper.x));
    atomicStore(&bvhStats[BVH_STAT_ROOT_UPPER + 1u], bitcast<u32>(box.upper.y));
    atomicStore(&bvhStats[BVH_STAT_ROOT_UPPER + 2u], bitcast<u32>(box.upper.z));
    atomicStore(&bvhStats[BVH_STAT_GENERATION], bvhParams.generation);
    atomicAdd(&bvhStats[BVH_STAT_ROOTS_FINISHED], 1u);
}

// Dispatches are 2D so very large trees stay under the per-dimension workgroup limit.
fn bvhThreadIndex(id: vec3u, groups: vec3u) -> u32 {
    return id.y * groups.x * BVH_WORKGROUP_SIZE + id.x;
}

// Whether lane 'lane' is running, from a subgroupBallot(true) mask: bit (lane % 32)
// of the mask's word (lane / 32). Indexing the vector by a runtime value is fine
// here: it happens once per thread, not in a loop.
fn bvhLaneRunning(running: vec4u, lane: u32) -> bool {
    let word = running[lane / 32u];
    return ((word >> (lane % 32u)) & 1u) == 1u;
}

// ---- Kernels -------------------------------------------------------------------------

@compute @workgroup_size(${BVH_REFIT_WORKGROUP_SIZE})
fn bvhRefitClear(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let node = bvhThreadIndex(id, groups);
    if (node == 0u) {
        atomicStore(&bvhStats[BVH_STAT_INCOMPLETE], 0u);
        atomicStore(&bvhStats[BVH_STAT_ROOTS_FINISHED], 0u);
        atomicStore(&bvhStats[BVH_STAT_REPAIRS], 0u);
    }
    if (node >= bvhParams.nodeCount) {
        return;
    }
    atomicStore(&bvhScratch[node], 0u);
    let base = bvhNodeWord(node);
    for (var side = 0u; side < 2u; side += 1u) {
        let slot = node * 2u + side;
        atomicStore(&bvhScratch[bvhCostWord(slot)], BVH_UNWRITTEN);
        // The missing right child of a one-primitive tree is never written, so it isn't marked either.
        if (atomicLoad(&bvhTree[base + select(BVH_LEFT_INDEX, BVH_RIGHT_INDEX, side == 1u)]) == BVH_EMPTY_CHILD) {
            continue;
        }
        let box = bvhSlotWord(slot);
        for (var offset = 0u; offset < 3u; offset += 1u) {
            atomicStore(&bvhTree[box + offset], BVH_UNWRITTEN);
            atomicStore(&bvhTree[box + BVH_MAX_OFFSET_WORDS + offset], BVH_UNWRITTEN);
        }
    }
}

@compute @workgroup_size(${BVH_REFIT_WORKGROUP_SIZE})
fn bvhRefitClimb(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) laneCount: u32,
) {
    let leaf = bvhThreadIndex(id, groups);
    let hasLeaf = leaf < bvhParams.leafCount;
    var slot = BVH_NO_PARENT;   // no slot: never matches a neighbour's
    var box = bvhEmptyBox();
    if (hasLeaf) {
        slot = bvhTopology[bvhParams.nodeCount + leaf];
        box = bvhLeafBox(slot);
    }

    // Subgroup step. Every thread of the workgroup reaches these calls (no return
    // above), so they run in uniform control flow as WGSL requires. A neighbour
    // lane's value counts only if that lane is running and holds the other side
    // of the same node; nothing here depends on the subgroup size.
    let running = subgroupBallot(true);
    let slotAbove = subgroupShuffleDown(slot, 1u);
    let lowerAbove = subgroupShuffleDown(box.lower, 1u);
    let upperAbove = subgroupShuffleDown(box.upper, 1u);
    let slotBelow = subgroupShuffleUp(slot, 1u);
    if (!hasLeaf) {
        return;
    }
    let isLeft = (slot & 1u) == 0u;
    let abovePaired = isLeft && lane + 1u < laneCount && bvhLaneRunning(running, lane + 1u) && slotAbove == slot + 1u;
    let belowPaired = !isLeft && lane > 0u && bvhLaneRunning(running, lane - 1u) && slotBelow == slot - 1u;

    if (belowPaired) {
        // The lane below holds the left sibling and finishes this node; this lane only stores its box.
        bvhStoreSlot(slot, box, 0.0);
        return;
    }

    var belowArea = 0.0;   // summed inner-node half-area under the current slot
    if (abovePaired) {
        // Finish this node in registers: the right sibling's box came from the lane above.
        bvhStoreSlot(slot, box, 0.0);
        box = bvhUnion(box, BvhBox(lowerAbove, upperAbove));
        belowArea = bvhHalfArea(box);
        let parent = bvhTopology[slot >> 1u];
        if (parent == BVH_NO_PARENT) {
            bvhFinishRoot(box, belowArea);
            return;
        }
        slot = parent;
    }

    // Climb. The bound only guards against a malformed tree: a real climb ends at the root.
    for (var step = 0u; step <= bvhParams.nodeCount; step += 1u) {
        let node = slot >> 1u;
        bvhStoreSlot(slot, box, belowArea);
        if (bvhParams.leafCount == 1u) {
            // A one-primitive tree: the root has only this child.
            bvhFinishRoot(box, bvhHalfArea(box));
            return;
        }
        if (atomicAdd(&bvhScratch[node], 1u) == 0u) {
            return;   // the sibling isn't finished yet; its thread carries on from here
        }
        let sibling = slot ^ 1u;
        box = bvhUnion(box, bvhLoadSlot(sibling));
        belowArea = belowArea + bvhWaitCost(sibling) + bvhHalfArea(box);
        let parent = bvhTopology[node];
        if (parent == BVH_NO_PARENT) {
            bvhFinishRoot(box, belowArea);
            return;
        }
        slot = parent;
    }
    atomicAdd(&bvhStats[BVH_STAT_INCOMPLETE], 1u);
}

// Set by the repair workgroup's first thread, so every thread sees the same answer.
var<workgroup> bvhRepairNeeded: u32;

// Runs as one workgroup after the climb. Normally it returns at once; when a
// climb read gave up, it refits the whole tree level by level, deepest first.
// Within one workgroup, the barrier after each level makes that level's boxes
// visible to the next, so this path needs no cross-workgroup visibility.
@compute @workgroup_size(${REPAIR_WORKGROUP_SIZE})
fn bvhRefitRepair(@builtin(local_invocation_index) thread: u32) {
    if (thread == 0u) {
        bvhRepairNeeded = atomicLoad(&bvhStats[BVH_STAT_INCOMPLETE]);
    }
    if (workgroupUniformLoad(&bvhRepairNeeded) == 0u) {
        return;
    }
    if (thread == 0u) {
        atomicAdd(&bvhStats[BVH_STAT_REPAIRS], 1u);
    }
    for (var level = 0u; level < bvhParams.levelCount; level += 1u) {
        let first = bvhTopology[bvhParams.levelStartsOffset + level];
        let end = bvhTopology[bvhParams.levelStartsOffset + level + 1u];
        for (var at = first + thread; at < end; at += BVH_REPAIR_WORKGROUP_SIZE) {
            let node = bvhTopology[bvhParams.levelNodesOffset + at];
            let base = bvhNodeWord(node);
            var nodeBox = bvhEmptyBox();
            var nodeBelowArea = 0.0;
            for (var side = 0u; side < 2u; side += 1u) {
                let slot = node * 2u + side;
                let child = atomicLoad(&bvhTree[base + select(BVH_LEFT_INDEX, BVH_RIGHT_INDEX, side == 1u)]);
                if (child == BVH_EMPTY_CHILD) {
                    continue;
                }
                let count = atomicLoad(&bvhTree[base + select(BVH_LEFT_COUNT, BVH_RIGHT_COUNT, side == 1u)]);
                var box: BvhBox;
                var belowArea = 0.0;
                if (count > 0u) {
                    box = bvhLeafBox(slot);
                } else {
                    // An inner child, finished in an earlier (deeper) level.
                    box = bvhReadSlot(child * 2u);
                    belowArea = bitcast<f32>(atomicLoad(&bvhScratch[bvhCostWord(child * 2u)]));
                    if (atomicLoad(&bvhTree[bvhNodeWord(child) + BVH_RIGHT_INDEX]) != BVH_EMPTY_CHILD) {
                        box = bvhUnion(box, bvhReadSlot(child * 2u + 1u));
                        belowArea += bitcast<f32>(atomicLoad(&bvhScratch[bvhCostWord(child * 2u + 1u)]));
                    }
                    belowArea += bvhHalfArea(box);
                }
                bvhStoreSlot(slot, box, belowArea);
                nodeBox = bvhUnion(nodeBox, box);
                nodeBelowArea += belowArea;
            }
            if (bvhTopology[node] == BVH_NO_PARENT) {
                bvhFinishRoot(nodeBox, nodeBelowArea + bvhHalfArea(nodeBox));
            }
        }
        storageBarrier();
        workgroupBarrier();
    }
}
`;
}

/** The tree shape the refit needs, from whoever built the tree. */
export interface BvhRefitShape {
    /** Where node 0 starts in the tree buffer, in 32-bit words. */
    readonly nodeWordOffset: number;
    readonly nodeCount: number;
    /** Parent of each node as a child slot (`parent * 2 + side`), or GPU_LAYOUT_NO_PARENT for the root. */
    readonly parentSlots: Uint32Array;
    /** Every leaf child slot, in primitive order (`leafSlotsOf`). */
    readonly leafSlots: Uint32Array;
    /** Any number that tells trees apart; echoed in the stats. */
    readonly generation: number;
}

/** One refit's summary, read back a frame or two later. */
export interface BvhRefitStats {
    readonly generation: number;
    readonly rootBox: Float32Array;
    /** Summed half-area of every inner node, the root included. */
    readonly summedHalfArea: number;
    /** `summedHalfArea` over the root's half-area: `relativeSahCost` of the refit tree. */
    readonly relativeCost: number;
    /** Climb reads that gave up waiting; see {@link BVH_REFIT_STAT}. Non-zero means the repair pass ran. */
    readonly incomplete: number;
    readonly rootsFinished: number;
    /** 1 when the repair pass refit the tree again. */
    readonly repairs: number;
}

export interface GpuBvhRefitOptions {
    readonly label?: string;
    /**
     * Read the stats back (a frame or two late) into `latestStats`. Off by
     * default: the platform promises hosts no readbacks frame after frame, so only its stats or
     * debug mode turns this on. Correctness never depends on it: a refit that
     * gives up is repaired on the GPU in the same frame either way.
     */
    readonly diagnostics?: boolean;
}

function dispatch2d(pass: GPUComputePassEncoder, threads: number, maxPerDimension: number): void {
    const workgroups = Math.ceil(threads / BVH_REFIT_WORKGROUP_SIZE);
    if (workgroups === 0) return;
    const x = Math.min(workgroups, maxPerDimension);
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
}

/**
 * Owns the refit's pipelines and its own buffers (topology, scratch, stats,
 * parameters). The tree and the primitive boxes belong to the caller.
 */
export class GpuBvhRefit {
    /** The refit needs subgroups; without them a caller uses its declared fallback (a CPU build). */
    static isSupported(device: GPUDevice): boolean {
        return device.features.has('subgroups');
    }

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly layout: GPUBindGroupLayout;
    private clearPipeline: GPUComputePipeline | null = null;
    private climbPipeline: GPUComputePipeline | null = null;
    private repairPipeline: GPUComputePipeline | null = null;
    private readonly diagnostics: boolean;
    private topology: GPUBuffer;
    private scratch: GPUBuffer;
    private readonly stats: GPUBuffer;
    private readonly params: GPUBuffer;
    private readonly statsReadback: GPUBuffer;
    private statsState: 'idle' | 'copied' | 'mapping' = 'idle';
    private shape: { nodeCount: number; leafCount: number } = { nodeCount: 0, leafCount: 0 };
    private cachedGroup: { group: GPUBindGroup; tree: GPUBuffer; boxes: GPUBuffer; topology: GPUBuffer; scratch: GPUBuffer } | null = null;
    private disposed = false;

    /** The newest stats read back, or null before the first. */
    latestStats: BvhRefitStats | null = null;

    constructor(device: GPUDevice, options: GpuBvhRefitOptions = {}) {
        if (!GpuBvhRefit.isSupported(device)) throw new Error('@pmndrs/bvh: the GPU refit needs a device with the "subgroups" feature');
        this.device = device;
        this.label = options.label ?? 'bvh refit';
        this.diagnostics = options.diagnostics ?? false;
        const storage = (type: GPUBufferBindingType): GPUBindGroupLayoutEntry['buffer'] => ({ type });
        this.layout = device.createBindGroupLayout({
            label: this.label,
            entries: [
                { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: storage('storage') },
                { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: storage('read-only-storage') },
                { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: storage('read-only-storage') },
                { binding: 3, visibility: SHADER_STAGE_COMPUTE, buffer: storage('storage') },
                { binding: 4, visibility: SHADER_STAGE_COMPUTE, buffer: storage('storage') },
                { binding: 5, visibility: SHADER_STAGE_COMPUTE, buffer: storage('uniform') },
            ],
        });
        this.topology = this.createStorage('topology', 64);
        this.scratch = this.createStorage('scratch', 64);
        this.stats = device.createBuffer({ label: `${this.label} stats`, size: BVH_REFIT_STATS_BYTES, usage: USAGE.STORAGE | USAGE.COPY_SRC });
        this.params = device.createBuffer({ label: `${this.label} params`, size: PARAM_WORDS * 4, usage: USAGE.UNIFORM | USAGE.COPY_DST });
        this.statsReadback = device.createBuffer({ label: `${this.label} stats readback`, size: BVH_REFIT_STATS_BYTES, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    }

    /** Compiles the kernels (asynchronously, so the main thread doesn't stall). */
    async prepare(): Promise<void> {
        if (this.ready) return;
        const module = this.device.createShaderModule({ label: this.label, code: bvhRefitWgsl() });
        const layout = this.device.createPipelineLayout({ label: this.label, bindGroupLayouts: [this.layout] });
        const pipeline = (entryPoint: string) =>
            this.device.createComputePipelineAsync({ label: `${this.label} ${entryPoint}`, layout, compute: { module, entryPoint } });
        [this.clearPipeline, this.climbPipeline, this.repairPipeline] = await Promise.all([
            pipeline('bvhRefitClear'),
            pipeline('bvhRefitClimb'),
            pipeline('bvhRefitRepair'),
        ]);
    }

    get ready(): boolean {
        return this.clearPipeline !== null && this.climbPipeline !== null && this.repairPipeline !== null;
    }

    /** Takes a new tree shape. Its buffers are written with `queue.writeBuffer`, so they are in place for the next submit. */
    setShape(shape: BvhRefitShape): void {
        const nodeCount = shape.nodeCount;
        const leafCount = shape.leafSlots.length;
        if (shape.parentSlots.length < nodeCount) throw new RangeError('parentSlots must hold one entry per node');
        const levels = levelOrderOf(shape.parentSlots, nodeCount);
        const levelCount = levels.levelStarts.length - 1;
        const levelStartsOffset = nodeCount + leafCount;
        const levelNodesOffset = levelStartsOffset + levels.levelStarts.length;
        const topologyWords = Math.max(1, levelNodesOffset + nodeCount);
        if (this.topology.size < topologyWords * 4) {
            this.topology.destroy();
            this.topology = this.createStorage('topology', topologyWords * 8);
        }
        const scratchWords = Math.max(1, nodeCount * 3);
        if (this.scratch.size < scratchWords * 4) {
            this.scratch.destroy();
            this.scratch = this.createStorage('scratch', scratchWords * 8);
        }
        const topology = new Uint32Array(topologyWords);
        topology.set(shape.parentSlots.subarray(0, nodeCount), 0);
        topology.set(shape.leafSlots, nodeCount);
        topology.set(levels.levelStarts, levelStartsOffset);
        topology.set(levels.levelNodes, levelNodesOffset);
        this.device.queue.writeBuffer(this.topology, 0, topology);
        const params = new Uint32Array(PARAM_WORDS);
        params.set([shape.nodeWordOffset, nodeCount, leafCount, shape.generation >>> 0, levelCount, levelStartsOffset, levelNodesOffset]);
        this.device.queue.writeBuffer(this.params, 0, params);
        this.shape = { nodeCount, leafCount };
    }

    /**
     * Records the refit into `pass`: the clear, the climb and the repair
     * (which returns at once unless the climb gave up). `tree` holds
     * the nodes at the shape's `nodeWordOffset`; `primitiveBoxes` holds
     * {@link PRIMITIVE_BOX_FLOATS} floats per primitive and must be written
     * by an earlier dispatch or command.
     */
    encode(pass: GPUComputePassEncoder, tree: GPUBuffer, primitiveBoxes: GPUBuffer): void {
        if (!this.clearPipeline || !this.climbPipeline || !this.repairPipeline) throw new Error('@pmndrs/bvh: call prepare() before encode()');
        if (this.shape.nodeCount === 0) return;
        const maxPerDimension = this.device.limits.maxComputeWorkgroupsPerDimension;
        pass.setBindGroup(0, this.bindGroup(tree, primitiveBoxes));
        pass.setPipeline(this.clearPipeline);
        dispatch2d(pass, this.shape.nodeCount, maxPerDimension);
        pass.setPipeline(this.climbPipeline);
        dispatch2d(pass, this.shape.leafCount, maxPerDimension);
        pass.setPipeline(this.repairPipeline);
        pass.dispatchWorkgroups(1);
    }

    /**
     * Stats readback, part 1, without ever waiting: call at the start of a
     * frame. It maps the copy {@link copyStats} recorded in an earlier frame
     * (the host has submitted that frame by now); `latestStats` updates when
     * the mapping resolves.
     */
    pollStats(): void {
        if (!this.diagnostics) return;
        if (this.statsState === 'copied') {
            this.statsState = 'mapping';
            this.statsReadback.mapAsync(MAP_MODE_READ).then(
                () => {
                    if (this.disposed) return;
                    const words = new Uint32Array(this.statsReadback.getMappedRange().slice(0));
                    this.statsReadback.unmap();
                    this.latestStats = parseStats(words);
                    this.statsState = 'idle';
                },
                () => {
                    // Mapping fails when the device is lost or the buffer destroyed; nothing to read then.
                    if (!this.disposed) this.statsState = 'idle';
                },
            );
        }
    }

    /**
     * Stats readback, part 2: after {@link encode}, copies this refit's stats
     * out unless an earlier copy is still being read. Does nothing unless
     * `diagnostics` is on.
     */
    copyStats(encoder: GPUCommandEncoder): void {
        if (this.diagnostics && this.statsState === 'idle' && this.shape.nodeCount > 0) {
            encoder.copyBufferToBuffer(this.stats, 0, this.statsReadback, 0, BVH_REFIT_STATS_BYTES);
            this.statsState = 'copied';
        }
    }

    get memoryBytes(): number {
        return this.topology.size + this.scratch.size + this.stats.size + this.params.size + this.statsReadback.size;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.topology.destroy();
        this.scratch.destroy();
        this.stats.destroy();
        this.params.destroy();
        this.statsReadback.destroy();
    }

    private createStorage(name: string, size: number): GPUBuffer {
        return this.device.createBuffer({ label: `${this.label} ${name}`, size, usage: USAGE.STORAGE | USAGE.COPY_DST });
    }

    private bindGroup(tree: GPUBuffer, boxes: GPUBuffer): GPUBindGroup {
        const cached = this.cachedGroup;
        if (cached && cached.tree === tree && cached.boxes === boxes && cached.topology === this.topology && cached.scratch === this.scratch) {
            return cached.group;
        }
        const group = this.device.createBindGroup({
            label: this.label,
            layout: this.layout,
            entries: [
                { binding: 0, resource: { buffer: tree } },
                { binding: 1, resource: { buffer: boxes } },
                { binding: 2, resource: { buffer: this.topology } },
                { binding: 3, resource: { buffer: this.scratch } },
                { binding: 4, resource: { buffer: this.stats } },
                { binding: 5, resource: { buffer: this.params } },
            ],
        });
        this.cachedGroup = { group, tree, boxes, topology: this.topology, scratch: this.scratch };
        return group;
    }
}

/** Decodes the stats buffer's words. */
export function parseStats(words: Uint32Array): BvhRefitStats {
    const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
    const rootBox = new Float32Array([
        floats[BVH_REFIT_STAT.rootLower], floats[BVH_REFIT_STAT.rootLower + 1], floats[BVH_REFIT_STAT.rootLower + 2],
        floats[BVH_REFIT_STAT.rootUpper], floats[BVH_REFIT_STAT.rootUpper + 1], floats[BVH_REFIT_STAT.rootUpper + 2],
    ]);
    const summedHalfArea = floats[BVH_REFIT_STAT.summedHalfArea];
    const size = [rootBox[3] - rootBox[0], rootBox[4] - rootBox[1], rootBox[5] - rootBox[2]];
    const rootArea = rootBox[0] > rootBox[3] ? 0 : size[0] * size[1] + size[1] * size[2] + size[2] * size[0];
    return {
        generation: words[BVH_REFIT_STAT.generation],
        rootBox,
        summedHalfArea,
        relativeCost: rootArea > 0 ? summedHalfArea / rootArea : 0,
        incomplete: words[BVH_REFIT_STAT.incomplete],
        rootsFinished: words[BVH_REFIT_STAT.rootsFinished],
        repairs: words[BVH_REFIT_STAT.repairs],
    };
}
