/**
 * WGSL of the GPU LBVH builder (stage B1, gpu-lbvh.ts). Kept apart from the
 * host code so the kernels read top to bottom in build order.
 *
 * Ported from HIPRT's `LbvhBuilderKernels.h` and `MortonCode.h` (MIT,
 * Advanced Micro Devices; licenses/hiprt.txt): the Morton code, the
 * 64-bit "highest differing bit" comparison with the index as tie-break, and
 * `findParent` / `EmitTopologyAndFitBounds`, Apetrei's bottom-up build
 * (Apetrei 2014, "Fast and Simply Scalable Bounding Volume Hierarchy
 * Construction"). What differs from HIPRT, and why:
 *
 * - **No 64-bit integers in WGSL:** the key-plus-index comparison is done as
 *   a pair of `u32`s, compared high word first.
 * - **Memory visibility:** HIPRT relies on `__threadfence()` to make the first
 *   child's plain stores visible to the second. WGSL has no such fence across
 *   workgroups, so everything one climbing thread hands to another goes
 *   through 32-bit atomic stores and loads (bit copies only), and the reader
 *   waits, bounded, for the clear pass's "unwritten" mark to go, as the B1a
 *   refit does (gpu-refit.ts, "Memory visibility"). A wait that runs out is
 *   counted; the builder then refuses the tree.
 * - **Children's data travel in the parent's slots** (HIPRT reads the sibling
 *   through its scratch node): each thread stores its subtree's box, reference,
 *   SAH cost and height into its side of the parent before the exchange, so the
 *   second thread reads one place.
 * - **SAH leaf collapse in the same climb:** the thread that finishes a node
 *   knows both children's costs, so it decides then whether the node would be
 *   cheaper as one leaf (at most `maxLeaf` triangles; the subtree's triangles
 *   are already neighbours in sorted order). Compaction and the final layout
 *   come in later dispatches.
 * - **Subgroups:** sibling leaves held by neighbouring lanes finish their node
 *   in registers, as in the B1a refit: the check compares positions and sides,
 *   never lane numbers, so any subgroup size or lane layout is right.
 */

import {
    GPU_LAYOUT_EMPTY_CHILD,
    GPU_LAYOUT_NO_PARENT,
    GPU_LAYOUT_NODE_WORD,
    GPU_LAYOUT_NODE_WORDS,
    GPU_LAYOUT_TRIANGLE_BYTES,
    GPU_LAYOUT_TRIANGLE_WORD,
} from './gpu-layout.ts';
import { BVH_FLOAT_MAX } from './refit-cpu.ts';
import { SCAN_WORKGROUP_SIZE, WORKGROUP_INDEX_WGSL } from './gpu-workgroup-scan.ts';

/** Threads per workgroup of every LBVH kernel. */
export const LBVH_WORKGROUP_SIZE = SCAN_WORKGROUP_SIZE;

/** Words of scratch per inner node: two child slots of {@link LBVH_SLOT_WORDS}. */
export const LBVH_SCRATCH_NODE_WORDS = 16;
const LBVH_SLOT_WORDS = 8;

/** Words of `lbvhNodeInfo` per inner node: first sorted position, count and collapse flag, parent. */
export const LBVH_NODE_INFO_WORDS = 3;

/** Re-reads of a handed-over word before the wait gives up (see the top of this file). */
const MAX_SPINS = 4096;

/** The clear pass's mark on every handed-over word, and "no exchange yet". A NaN pattern, never a real value. */
const UNWRITTEN = 0xffffffff;

/** A child reference with this bit is a leaf: a single triangle at that sorted position. */
const LEAF_BIT = 0x80000000;

/** In `lbvhNodeInfo`'s second word: the node is cheaper as one leaf. The low bits are its triangle count. */
const COLLAPSE_BIT = 0x80000000;

/** Word positions in the stats buffer. */
export const LBVH_STAT = {
    /** Centroid bounds as order-preserving integers: min x, y, z, then max x, y, z. */
    centroidMin: 0,
    centroidMax: 3,
    /** Split index of the root in the radix tree. */
    rootNode: 6,
    nodeCount: 7,
    leafCount: 8,
    maxDepth: 9,
    /** The mesh's box as f32 bits: min x, y, z, then max x, y, z. */
    bounds: 10,
    /** Handed-over words still unwritten after the bounded wait, or climbs that didn't end. Must be 0. */
    incomplete: 16,
    /** Threads that finished the root: must be 1. */
    rootsFinished: 17,
    /** Triangle corners that named a vertex past the end. Must be 0. */
    badIndices: 18,
} as const;
export const LBVH_STATS_WORDS = 32;
export const LBVH_STATS_BYTES = LBVH_STATS_WORDS * 4;

/** Bytes of the uniform block (`LbvhParams`). */
export const LBVH_PARAMS_BYTES = 48;

/** `LbvhParams.indexFormat` values. */
export const LBVH_INDEX_FORMAT = { none: 0, uint16: 16, uint32: 32 } as const;

const hex = (value: number): string => `0x${value.toString(16)}u`;

export function lbvhWgsl(): string {
    const word = GPU_LAYOUT_NODE_WORD;
    const triangle = GPU_LAYOUT_TRIANGLE_WORD;
    const stat = LBVH_STAT;
    return /* wgsl */ `enable subgroups;

struct LbvhParams {
    count: u32,              // triangles
    vertexCount: u32,
    positionStride: u32,     // floats from one vertex to the next
    positionOffset: u32,     // float index of vertex 0's x
    indexFormat: u32,        // 0: a plain triangle list, 16 or 32: index bits
    indexOffset: u32,        // in indices, not bytes
    hasFlags: u32,
    maxLeaf: u32,            // largest leaf the collapse may make
    traversalCost: f32,      // SAH cost of visiting a node
    intersectionCost: f32,   // SAH cost of testing a triangle
    unused0: u32,
    unused1: u32,
}

// ---- Bindings ----------------------------------------------------------------------
// Pipelines use 'auto' layouts, so each kernel binds only what it uses (at most 7 storage buffers).

@group(0) @binding(0) var<storage, read> lbvhPositions: array<f32>;
@group(0) @binding(1) var<storage, read> lbvhIndices: array<u32>;
@group(0) @binding(2) var<storage, read> lbvhFlags: array<u32>;
// Two per triangle, in the input's order: (min.xyz, 0), (max.xyz, 0).
@group(0) @binding(3) var<storage, read_write> lbvhPrimitiveBoxes: array<vec4f>;
// Morton codes, then (after the sort) in sorted order.
@group(0) @binding(4) var<storage, read_write> lbvhKeys: array<u32>;
// Triangle numbers, then (after the sort) the triangle at each sorted position.
@group(0) @binding(5) var<storage, read_write> lbvhValues: array<u32>;
// Per inner node, two child slots: (min.xyz, reference), (max.xyz, cost). Handed between climbing threads.
@group(0) @binding(6) var<storage, read_write> lbvhScratch: array<atomic<u32>>;
// [the range end each inner node's first child left, for its second] then [a height per child slot].
@group(0) @binding(7) var<storage, read_write> lbvhState: array<atomic<u32>>;
// Per inner node: first sorted position, count | collapse bit, parent. Read only in later dispatches.
@group(0) @binding(8) var<storage, read_write> lbvhNodeInfo: array<u32>;
// Per inner node: 1 if kept, then (after the scan) its number among the kept nodes.
@group(0) @binding(9) var<storage, read_write> lbvhRank: array<u32>;
@group(0) @binding(10) var<storage, read_write> lbvhOutNodes: array<u32>;
@group(0) @binding(11) var<storage, read_write> lbvhOutTriangles: array<u32>;
@group(0) @binding(12) var<storage, read_write> lbvhOutParents: array<u32>;
@group(0) @binding(13) var<storage, read_write> lbvhStats: array<atomic<u32>>;
@group(0) @binding(14) var<uniform> lbvhParams: LbvhParams;

// ---- Constants (generated from gpu-layout.ts and gpu-lbvh-wgsl.ts) ------------------

const LBVH_WORKGROUP_SIZE: u32 = ${LBVH_WORKGROUP_SIZE}u;
const LBVH_SCRATCH_NODE_WORDS: u32 = ${LBVH_SCRATCH_NODE_WORDS}u;
const LBVH_SLOT_WORDS: u32 = ${LBVH_SLOT_WORDS}u;
const LBVH_NODE_INFO_WORDS: u32 = ${LBVH_NODE_INFO_WORDS}u;
const LBVH_UNWRITTEN: u32 = ${hex(UNWRITTEN)};
const LBVH_MAX_SPINS: u32 = ${MAX_SPINS}u;
const LBVH_LEAF_BIT: u32 = ${hex(LEAF_BIT)};
const LBVH_COLLAPSE_BIT: u32 = ${hex(COLLAPSE_BIT)};
const LBVH_COUNT_MASK: u32 = ${hex(COLLAPSE_BIT - 1)};
const LBVH_NO_PARENT: u32 = ${hex(GPU_LAYOUT_NO_PARENT)};
const LBVH_EMPTY_CHILD: u32 = ${hex(GPU_LAYOUT_EMPTY_CHILD)};
const LBVH_FLOAT_MAX: f32 = 0x1.fffffep+127f;   // ${BVH_FLOAT_MAX}

const LBVH_NODE_WORDS: u32 = ${GPU_LAYOUT_NODE_WORDS}u;
const LBVH_LEFT_INDEX: u32 = ${word.leftIndex}u;
const LBVH_RIGHT_INDEX: u32 = ${word.rightIndex}u;
const LBVH_LEFT_COUNT: u32 = ${word.leftCount}u;
const LBVH_RIGHT_COUNT: u32 = ${word.rightCount}u;
const LBVH_CHILD_MAX_OFFSET: u32 = ${word.leftMax - word.leftMin}u;   // from a child's min to its max
const LBVH_CHILD_STRIDE: u32 = ${word.rightMin - word.leftMin}u;      // from the left child's box to the right's
const LBVH_TRIANGLE_WORDS: u32 = ${GPU_LAYOUT_TRIANGLE_BYTES / 4}u;
const LBVH_TRIANGLE_CORNER_STRIDE: u32 = ${triangle.vertex1 - triangle.vertex0}u;
const LBVH_TRIANGLE_ORIGINAL_INDEX: u32 = ${triangle.originalIndex}u;
const LBVH_TRIANGLE_FLAGS: u32 = ${triangle.flags}u;
const LBVH_TRIANGLE_PADDING: u32 = ${triangle.padding}u;

const LBVH_STAT_CENTROID_MIN: u32 = ${stat.centroidMin}u;
const LBVH_STAT_CENTROID_MAX: u32 = ${stat.centroidMax}u;
const LBVH_STAT_ROOT_NODE: u32 = ${stat.rootNode}u;
const LBVH_STAT_NODE_COUNT: u32 = ${stat.nodeCount}u;
const LBVH_STAT_LEAF_COUNT: u32 = ${stat.leafCount}u;
const LBVH_STAT_MAX_DEPTH: u32 = ${stat.maxDepth}u;
const LBVH_STAT_BOUNDS: u32 = ${stat.bounds}u;
const LBVH_STAT_INCOMPLETE: u32 = ${stat.incomplete}u;
const LBVH_STAT_ROOTS_FINISHED: u32 = ${stat.rootsFinished}u;
const LBVH_STAT_BAD_INDICES: u32 = ${stat.badIndices}u;
const LBVH_STATS_WORDS: u32 = ${LBVH_STATS_WORDS}u;
${WORKGROUP_INDEX_WGSL}

fn lbvhThreadIndex(workgroup: vec3u, groups: vec3u, localIndex: u32) -> u32 {
    return flatWorkgroupIndex(workgroup, groups) * LBVH_WORKGROUP_SIZE + localIndex;
}

// Inner nodes of the radix tree: one fewer than triangles.
fn lbvhInnerCount() -> u32 {
    return max(lbvhParams.count, 1u) - 1u;
}

// ---- Geometry -------------------------------------------------------------------------

fn lbvhVertexOf(triangle: u32, corner: u32) -> u32 {
    let entry = lbvhParams.indexOffset + triangle * 3u + corner;
    var vertex = triangle * 3u + corner;
    if (lbvhParams.indexFormat == 32u) {
        vertex = lbvhIndices[entry];
    } else if (lbvhParams.indexFormat == 16u) {
        vertex = (lbvhIndices[entry >> 1u] >> ((entry & 1u) * 16u)) & 0xffffu;
    }
    return vertex;
}

fn lbvhCorner(vertex: u32) -> vec3f {
    let at = lbvhParams.positionOffset + vertex * lbvhParams.positionStride;
    return vec3f(lbvhPositions[at], lbvhPositions[at + 1u], lbvhPositions[at + 2u]);
}

struct LbvhBox {
    lower: vec3f,
    upper: vec3f,
}

fn lbvhPrimitiveBox(triangle: u32) -> LbvhBox {
    return LbvhBox(lbvhPrimitiveBoxes[triangle * 2u].xyz, lbvhPrimitiveBoxes[triangle * 2u + 1u].xyz);
}

fn lbvhHalfArea(lower: vec3f, upper: vec3f) -> f32 {
    let size = upper - lower;
    return size.x * size.y + size.y * size.z + size.z * size.x;
}

// Float to an integer with the same order, so atomicMin and atomicMax sort floats.
fn lbvhOrderedFromFloat(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn lbvhFloatFromOrdered(ordered: u32) -> f32 {
    return bitcast<f32>(select(~ordered, ordered ^ 0x80000000u, (ordered & 0x80000000u) != 0u));
}

// ---- 1. Start: the stats ----------------------------------------------------------------

// One thread. Every value the later kernels accumulate into starts here, so a
// recorded build can be submitted again and gives the same tree.
@compute @workgroup_size(1)
fn lbvhStart() {
    for (var at = 0u; at < LBVH_STATS_WORDS; at += 1u) {
        atomicStore(&lbvhStats[at], 0u);
    }
    for (var axis = 0u; axis < 3u; axis += 1u) {
        atomicStore(&lbvhStats[LBVH_STAT_CENTROID_MIN + axis], 0xffffffffu);
    }
}

// ---- 2. Triangle boxes and the centroid bounds -------------------------------------------

var<workgroup> lbvhGroupMin: array<atomic<u32>, 3>;
var<workgroup> lbvhGroupMax: array<atomic<u32>, 3>;

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhBoxes(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (localIndex < 3u) {
        atomicStore(&lbvhGroupMin[localIndex], 0xffffffffu);
    }
    workgroupBarrier();

    let triangle = lbvhThreadIndex(workgroup, groups, localIndex);
    let hasTriangle = triangle < lbvhParams.count;
    var centroid = vec3f(0.0);
    if (hasTriangle) {
        var lower = vec3f(LBVH_FLOAT_MAX);
        var upper = vec3f(-LBVH_FLOAT_MAX);
        for (var corner = 0u; corner < 3u; corner += 1u) {
            var vertex = lbvhVertexOf(triangle, corner);
            if (vertex >= lbvhParams.vertexCount) {
                atomicAdd(&lbvhStats[LBVH_STAT_BAD_INDICES], 1u);
                vertex = 0u;
            }
            let position = lbvhCorner(vertex);
            lower = min(lower, position);
            upper = max(upper, position);
        }
        lbvhPrimitiveBoxes[triangle * 2u] = vec4f(lower, 0.0);
        lbvhPrimitiveBoxes[triangle * 2u + 1u] = vec4f(upper, 0.0);
        centroid = (lower + upper) * 0.5;
    }

    // Reduce in the subgroup, then the workgroup, then one global atomic per axis and workgroup.
    let lowest = subgroupMin(select(vec3f(LBVH_FLOAT_MAX), centroid, hasTriangle));
    let highest = subgroupMax(select(vec3f(-LBVH_FLOAT_MAX), centroid, hasTriangle));
    if (subgroupElect()) {
        for (var axis = 0u; axis < 3u; axis += 1u) {
            atomicMin(&lbvhGroupMin[axis], lbvhOrderedFromFloat(lowest[axis]));
            atomicMax(&lbvhGroupMax[axis], lbvhOrderedFromFloat(highest[axis]));
        }
    }
    workgroupBarrier();
    if (localIndex < 3u) {
        atomicMin(&lbvhStats[LBVH_STAT_CENTROID_MIN + localIndex], atomicLoad(&lbvhGroupMin[localIndex]));
        atomicMax(&lbvhStats[LBVH_STAT_CENTROID_MAX + localIndex], atomicLoad(&lbvhGroupMax[localIndex]));
    }
}

// ---- 3. Morton codes (HIPRT MortonCode.h) -----------------------------------------------

// Spreads the low 10 bits of v so two zero bits follow each one.
fn lbvhExpandBits(value: u32) -> u32 {
    var v = (value * 0x00010001u) & 0xff0000ffu;
    v = (v * 0x00000101u) & 0x0f00f00fu;
    v = (v * 0x00000011u) & 0xc30c30c3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhMorton(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let triangle = lbvhThreadIndex(workgroup, groups, localIndex);
    if (triangle >= lbvhParams.count) {
        return;
    }
    var lower: vec3f;
    var upper: vec3f;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        lower[axis] = lbvhFloatFromOrdered(atomicLoad(&lbvhStats[LBVH_STAT_CENTROID_MIN + axis]));
        upper[axis] = lbvhFloatFromOrdered(atomicLoad(&lbvhStats[LBVH_STAT_CENTROID_MAX + axis]));
    }
    // Cubic cells over each mesh's own centroid box: 1024 along its longest
    // side, the same cell size on the others. HIPRT's plain code stretches every
    // axis to 1024 cells, which splits a flat or long mesh across its thin side
    // as often as along its long ones: SAH cost 2.4-2.6 times tinybvh's on slabs,
    // rods and flat grids, against 1.02-1.22 with cubic cells (NOTES, B1).
    let extent = upper - lower;
    let longest = max(extent.x, max(extent.y, extent.z));
    let scale = select(0.0, 1024.0 / longest, longest > 0.0);
    let box = lbvhPrimitiveBox(triangle);
    let centroid = (box.lower + box.upper) * 0.5;
    let cell = vec3u(clamp((centroid - lower) * scale, vec3f(0.0), vec3f(1023.0)));
    lbvhKeys[triangle] = lbvhExpandBits(cell.x) * 4u + lbvhExpandBits(cell.y) * 2u + lbvhExpandBits(cell.z);
    lbvhValues[triangle] = triangle;
}

// ---- 4. The climb (HIPRT EmitTopologyAndFitBounds; Apetrei 2014) -------------------------

// One child of an inner node, as the climbing thread carries it.
struct LbvhChild {
    lower: vec3f,
    upper: vec3f,
    reference: u32,   // LBVH_LEAF_BIT | sorted position, or an inner node's split index
    cost: f32,        // SAH cost of the subtree: the cheaper of one leaf and an inner node
    height: u32,      // levels of inner nodes below, 0 if the child ends up a leaf
}

fn lbvhHeightWord(node: u32, side: u32) -> u32 {
    return lbvhInnerCount() + node * 2u + side;
}

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhClear(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let node = lbvhThreadIndex(workgroup, groups, localIndex);
    if (node >= lbvhInnerCount()) {
        return;
    }
    let base = node * LBVH_SCRATCH_NODE_WORDS;
    for (var at = 0u; at < LBVH_SCRATCH_NODE_WORDS; at += 1u) {
        atomicStore(&lbvhScratch[base + at], LBVH_UNWRITTEN);
    }
    atomicStore(&lbvhState[node], LBVH_UNWRITTEN);
    atomicStore(&lbvhState[lbvhHeightWord(node, 0u)], LBVH_UNWRITTEN);
    atomicStore(&lbvhState[lbvhHeightWord(node, 1u)], LBVH_UNWRITTEN);
}

fn lbvhStoreChild(node: u32, side: u32, child: LbvhChild) {
    let base = node * LBVH_SCRATCH_NODE_WORDS + side * LBVH_SLOT_WORDS;
    atomicStore(&lbvhScratch[base], bitcast<u32>(child.lower.x));
    atomicStore(&lbvhScratch[base + 1u], bitcast<u32>(child.lower.y));
    atomicStore(&lbvhScratch[base + 2u], bitcast<u32>(child.lower.z));
    atomicStore(&lbvhScratch[base + 3u], child.reference);
    atomicStore(&lbvhScratch[base + 4u], bitcast<u32>(child.upper.x));
    atomicStore(&lbvhScratch[base + 5u], bitcast<u32>(child.upper.y));
    atomicStore(&lbvhScratch[base + 6u], bitcast<u32>(child.upper.z));
    atomicStore(&lbvhScratch[base + 7u], bitcast<u32>(child.cost));
    atomicStore(&lbvhState[lbvhHeightWord(node, side)], child.height);
}

// One handed-over word, waiting (bounded) until the clear pass's mark is gone.
fn lbvhWaitScratch(at: u32) -> u32 {
    var value = atomicLoad(&lbvhScratch[at]);
    for (var spins = 0u; value == LBVH_UNWRITTEN; spins += 1u) {
        if (spins == LBVH_MAX_SPINS) {
            atomicAdd(&lbvhStats[LBVH_STAT_INCOMPLETE], 1u);
            break;
        }
        value = atomicLoad(&lbvhScratch[at]);
    }
    return value;
}

fn lbvhWaitHeight(at: u32) -> u32 {
    var value = atomicLoad(&lbvhState[at]);
    for (var spins = 0u; value == LBVH_UNWRITTEN; spins += 1u) {
        if (spins == LBVH_MAX_SPINS) {
            atomicAdd(&lbvhStats[LBVH_STAT_INCOMPLETE], 1u);
            break;
        }
        value = atomicLoad(&lbvhState[at]);
    }
    return value;
}

// The sibling's slot during the climb: its thread stored it before its exchange, which this thread saw.
fn lbvhWaitChild(node: u32, side: u32) -> LbvhChild {
    let base = node * LBVH_SCRATCH_NODE_WORDS + side * LBVH_SLOT_WORDS;
    return LbvhChild(
        vec3f(bitcast<f32>(lbvhWaitScratch(base)), bitcast<f32>(lbvhWaitScratch(base + 1u)), bitcast<f32>(lbvhWaitScratch(base + 2u))),
        vec3f(bitcast<f32>(lbvhWaitScratch(base + 4u)), bitcast<f32>(lbvhWaitScratch(base + 5u)), bitcast<f32>(lbvhWaitScratch(base + 6u))),
        lbvhWaitScratch(base + 3u),
        bitcast<f32>(lbvhWaitScratch(base + 7u)),
        lbvhWaitHeight(lbvhHeightWord(node, side)),
    );
}

// A slot in a later dispatch, when every store has landed.
fn lbvhReadChild(node: u32, side: u32) -> LbvhChild {
    let base = node * LBVH_SCRATCH_NODE_WORDS + side * LBVH_SLOT_WORDS;
    return LbvhChild(
        vec3f(bitcast<f32>(atomicLoad(&lbvhScratch[base])), bitcast<f32>(atomicLoad(&lbvhScratch[base + 1u])), bitcast<f32>(atomicLoad(&lbvhScratch[base + 2u]))),
        vec3f(bitcast<f32>(atomicLoad(&lbvhScratch[base + 4u])), bitcast<f32>(atomicLoad(&lbvhScratch[base + 5u])), bitcast<f32>(atomicLoad(&lbvhScratch[base + 6u]))),
        atomicLoad(&lbvhScratch[base + 3u]),
        bitcast<f32>(atomicLoad(&lbvhScratch[base + 7u])),
        0u,
    );
}

// HIPRT findHighestDifferentBit, as a pair: (keys differing, positions differing).
// Bigger means the two neighbours differ in a higher bit. Position k and k + 1 only.
fn lbvhDelta(position: u32) -> vec2u {
    return vec2u(lbvhKeys[position] ^ lbvhKeys[position + 1u], position ^ (position + 1u));
}

fn lbvhLess(a: vec2u, b: vec2u) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

struct LbvhParent {
    node: u32,
    side: u32,   // 0: the range is the parent's left child
}

// HIPRT findParent for the range [first, end) (end exclusive), which is not the
// whole mesh: the parent splits next to whichever end has the closer neighbour.
fn lbvhFindParent(first: u32, end: u32) -> LbvhParent {
    if (first == 0u || (end != lbvhParams.count && lbvhLess(lbvhDelta(end - 1u), lbvhDelta(first - 1u)))) {
        return LbvhParent(end - 1u, 0u);
    }
    return LbvhParent(first - 1u, 1u);
}

// Both children are known: the node's box, its SAH cost, and whether it is cheaper as one leaf.
fn lbvhFinishNode(node: u32, first: u32, end: u32, a: LbvhChild, b: LbvhChild) -> LbvhChild {
    let lower = min(a.lower, b.lower);
    let upper = max(a.upper, b.upper);
    let count = end - first;
    let area = lbvhHalfArea(lower, upper);
    let innerCost = lbvhParams.traversalCost * area + a.cost + b.cost;
    var leafCost = LBVH_FLOAT_MAX;
    if (count <= lbvhParams.maxLeaf) {
        leafCost = lbvhParams.intersectionCost * f32(count) * area;
    }
    // The root stays an inner node: the layout's root always has two children.
    let isRoot = first == 0u && end == lbvhParams.count;
    let collapse = !isRoot && leafCost <= innerCost;
    lbvhNodeInfo[node * LBVH_NODE_INFO_WORDS] = first;
    lbvhNodeInfo[node * LBVH_NODE_INFO_WORDS + 1u] = count | select(0u, LBVH_COLLAPSE_BIT, collapse);
    let height = 1u + max(a.height, b.height);
    return LbvhChild(lower, upper, node, min(innerCost, leafCost), select(height, 0u, collapse));
}

fn lbvhFinishRoot(node: u32, root: LbvhChild) {
    lbvhNodeInfo[node * LBVH_NODE_INFO_WORDS + 2u] = LBVH_NO_PARENT;
    atomicStore(&lbvhStats[LBVH_STAT_ROOT_NODE], node);
    atomicStore(&lbvhStats[LBVH_STAT_MAX_DEPTH], root.height);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        atomicStore(&lbvhStats[LBVH_STAT_BOUNDS + axis], bitcast<u32>(root.lower[axis]));
        atomicStore(&lbvhStats[LBVH_STAT_BOUNDS + 3u + axis], bitcast<u32>(root.upper[axis]));
    }
    atomicAdd(&lbvhStats[LBVH_STAT_ROOTS_FINISHED], 1u);
}

// Whether lane 'lane' is running, from a subgroupBallot mask (bit lane % 32 of word lane / 32).
fn lbvhLaneRunning(running: vec4u, lane: u32) -> bool {
    return ((running[lane / 32u] >> (lane % 32u)) & 1u) == 1u;
}

// One thread per sorted position. Leaves climb; at each node the first thread
// to arrive stops, and the second (which then knows the node's whole range)
// finishes the node and carries on to the parent.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhClimb(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) laneCount: u32,
) {
    let count = lbvhParams.count;
    let position = lbvhThreadIndex(workgroup, groups, localIndex);
    let climbs = position < count && count > 1u;
    var child = LbvhChild(vec3f(0.0), vec3f(0.0), LBVH_LEAF_BIT | position, 0.0, 0u);
    var first = position;
    var end = position + 1u;
    var parent = LbvhParent(LBVH_NO_PARENT, 0u);
    if (climbs) {
        let box = lbvhPrimitiveBox(lbvhValues[position]);
        child.lower = box.lower;
        child.upper = box.upper;
        child.cost = lbvhParams.intersectionCost * lbvhHalfArea(box.lower, box.upper);
        parent = lbvhFindParent(first, end);
    }

    // Subgroup step, in uniform control flow (no return above). Sibling leaves
    // are neighbouring positions; when neighbouring lanes hold them, the lower
    // lane finishes their node in registers. Only positions and sides are
    // compared, so this is right for any subgroup size and lane layout.
    let running = subgroupBallot(climbs);
    let positionAbove = subgroupShuffleDown(position, 1u);
    let parentAbove = subgroupShuffleDown(parent.node, 1u);
    let sideAbove = subgroupShuffleDown(parent.side, 1u);
    let lowerAbove = subgroupShuffleDown(child.lower, 1u);
    let upperAbove = subgroupShuffleDown(child.upper, 1u);
    let costAbove = subgroupShuffleDown(child.cost, 1u);
    let positionBelow = subgroupShuffleUp(position, 1u);
    let parentBelow = subgroupShuffleUp(parent.node, 1u);
    let sideBelow = subgroupShuffleUp(parent.side, 1u);
    if (!climbs) {
        return;
    }
    let pairedAbove = parent.side == 0u && lane + 1u < laneCount && lbvhLaneRunning(running, lane + 1u) &&
        positionAbove == position + 1u && parentAbove == parent.node && sideAbove == 1u;
    let pairedBelow = parent.side == 1u && lane > 0u && lbvhLaneRunning(running, lane - 1u) &&
        positionBelow + 1u == position && parentBelow == parent.node && sideBelow == 0u;
    if (pairedBelow) {
        return;   // the lane below holds the left sibling and finishes the node
    }
    if (pairedAbove) {
        let node = parent.node;
        let right = LbvhChild(lowerAbove, upperAbove, LBVH_LEAF_BIT | (position + 1u), costAbove, 0u);
        // Stored for the emit pass; no other thread reads them during the climb.
        lbvhStoreChild(node, 0u, child);
        lbvhStoreChild(node, 1u, right);
        end = position + 2u;
        child = lbvhFinishNode(node, first, end, child, right);
        if (first == 0u && end == count) {
            lbvhFinishRoot(node, child);
            return;
        }
        parent = lbvhFindParent(first, end);
        lbvhNodeInfo[node * LBVH_NODE_INFO_WORDS + 2u] = parent.node;
    }

    // The bound only guards against a broken climb: a real one ends at the root.
    for (var step = 0u; step < count; step += 1u) {
        let node = parent.node;
        lbvhStoreChild(node, parent.side, child);
        // Leave this side's far end for the sibling's thread; the first to arrive gets "unwritten" back and stops.
        let other = atomicExchange(&lbvhState[node], select(first, end, parent.side == 1u));
        if (other == LBVH_UNWRITTEN) {
            return;
        }
        if (parent.side == 1u) {
            first = other;
        } else {
            end = other;
        }
        child = lbvhFinishNode(node, first, end, child, lbvhWaitChild(node, 1u - parent.side));
        if (first == 0u && end == count) {
            lbvhFinishRoot(node, child);
            return;
        }
        parent = lbvhFindParent(first, end);
        lbvhNodeInfo[node * LBVH_NODE_INFO_WORDS + 2u] = parent.node;
    }
    atomicAdd(&lbvhStats[LBVH_STAT_INCOMPLETE], 1u);
}

// ---- 5. Which inner nodes stay ----------------------------------------------------------

// A node stays unless it or an ancestor became a leaf. Only ancestors with at
// most maxLeaf triangles can have, and counts grow on the way up, so the walk
// is short.
fn lbvhKept(node: u32) -> bool {
    if ((lbvhNodeInfo[node * LBVH_NODE_INFO_WORDS + 1u] & LBVH_COLLAPSE_BIT) != 0u) {
        return false;
    }
    var above = lbvhNodeInfo[node * LBVH_NODE_INFO_WORDS + 2u];
    for (var guard = 0u; guard <= lbvhParams.maxLeaf && above != LBVH_NO_PARENT; guard += 1u) {
        let info = lbvhNodeInfo[above * LBVH_NODE_INFO_WORDS + 1u];
        if ((info & LBVH_COUNT_MASK) > lbvhParams.maxLeaf) {
            return true;
        }
        if ((info & LBVH_COLLAPSE_BIT) != 0u) {
            return false;
        }
        above = lbvhNodeInfo[above * LBVH_NODE_INFO_WORDS + 2u];
    }
    return true;
}

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhMark(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let node = lbvhThreadIndex(workgroup, groups, localIndex);
    if (node >= lbvhInnerCount()) {
        return;
    }
    lbvhRank[node] = select(0u, 1u, lbvhKept(node));
}

// ---- 6. The tracer's layout ----------------------------------------------------------------

// Kept nodes are numbered by the scan in split order; the root swaps with
// whichever node got 0, because the layout's root is node 0.
fn lbvhOutputIndex(rank: u32, rootRank: u32) -> u32 {
    if (rank == rootRank) {
        return 0u;
    }
    return select(rank, rootRank, rank == 0u);
}

fn lbvhWriteChild(out: u32, side: u32, lower: vec3f, upper: vec3f, index: u32, count: u32) {
    let box = out + side * LBVH_CHILD_STRIDE;
    lbvhOutNodes[box] = bitcast<u32>(lower.x);
    lbvhOutNodes[box + 1u] = bitcast<u32>(lower.y);
    lbvhOutNodes[box + 2u] = bitcast<u32>(lower.z);
    lbvhOutNodes[box + LBVH_CHILD_MAX_OFFSET] = bitcast<u32>(upper.x);
    lbvhOutNodes[box + LBVH_CHILD_MAX_OFFSET + 1u] = bitcast<u32>(upper.y);
    lbvhOutNodes[box + LBVH_CHILD_MAX_OFFSET + 2u] = bitcast<u32>(upper.z);
    lbvhOutNodes[out + select(LBVH_LEFT_INDEX, LBVH_RIGHT_INDEX, side == 1u)] = index;
    lbvhOutNodes[out + select(LBVH_LEFT_COUNT, LBVH_RIGHT_COUNT, side == 1u)] = count;
}

// A one-triangle mesh: the root holds the triangle on the left and nothing on the right.
fn lbvhEmitSingleTriangle() {
    let box = lbvhPrimitiveBox(0u);
    lbvhWriteChild(0u, 0u, box.lower, box.upper, 0u, 1u);
    lbvhWriteChild(0u, 1u, vec3f(0.0), vec3f(0.0), LBVH_EMPTY_CHILD, 0u);
    lbvhOutParents[0] = LBVH_NO_PARENT;
    atomicStore(&lbvhStats[LBVH_STAT_NODE_COUNT], 1u);
    atomicStore(&lbvhStats[LBVH_STAT_LEAF_COUNT], 1u);
    atomicStore(&lbvhStats[LBVH_STAT_MAX_DEPTH], 1u);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        atomicStore(&lbvhStats[LBVH_STAT_BOUNDS + axis], bitcast<u32>(box.lower[axis]));
        atomicStore(&lbvhStats[LBVH_STAT_BOUNDS + 3u + axis], bitcast<u32>(box.upper[axis]));
    }
    atomicStore(&lbvhStats[LBVH_STAT_ROOTS_FINISHED], 1u);
}

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhEmit(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let node = lbvhThreadIndex(workgroup, groups, localIndex);
    if (lbvhParams.count == 1u) {
        if (node == 0u) {
            lbvhEmitSingleTriangle();
        }
        return;
    }
    if (node >= lbvhInnerCount() || !lbvhKept(node)) {
        return;
    }
    let root = atomicLoad(&lbvhStats[LBVH_STAT_ROOT_NODE]);
    let rootRank = lbvhRank[root];
    let out = lbvhOutputIndex(lbvhRank[node], rootRank);
    var leaves = 0u;
    for (var side = 0u; side < 2u; side += 1u) {
        let child = lbvhReadChild(node, side);
        var index: u32;
        var count: u32;
        if ((child.reference & LBVH_LEAF_BIT) != 0u) {
            index = child.reference & ~LBVH_LEAF_BIT;
            count = 1u;
        } else {
            let info = lbvhNodeInfo[child.reference * LBVH_NODE_INFO_WORDS + 1u];
            if ((info & LBVH_COLLAPSE_BIT) != 0u) {
                // A subtree's triangles are neighbours in sorted order: the leaf is that run.
                index = lbvhNodeInfo[child.reference * LBVH_NODE_INFO_WORDS];
                count = info & LBVH_COUNT_MASK;
            } else {
                index = lbvhOutputIndex(lbvhRank[child.reference], rootRank);
                count = 0u;
            }
        }
        if (count > 0u) {
            leaves += 1u;
        }
        lbvhWriteChild(out * LBVH_NODE_WORDS, side, child.lower, child.upper, index, count);
    }
    var parent = LBVH_NO_PARENT;
    if (node != root) {
        parent = lbvhOutputIndex(lbvhRank[lbvhNodeInfo[node * LBVH_NODE_INFO_WORDS + 2u]], rootRank);
    }
    lbvhOutParents[out] = parent;
    atomicAdd(&lbvhStats[LBVH_STAT_NODE_COUNT], 1u);
    atomicAdd(&lbvhStats[LBVH_STAT_LEAF_COUNT], leaves);
}

// Triangles in tree order (sorted order): three corners as they are, the original index and the flags.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhTriangles(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let position = lbvhThreadIndex(workgroup, groups, localIndex);
    if (position >= lbvhParams.count) {
        return;
    }
    let triangle = lbvhValues[position];
    let base = position * LBVH_TRIANGLE_WORDS;
    for (var corner = 0u; corner < 3u; corner += 1u) {
        // A bad index was counted by lbvhBoxes (and the build refused); clamping only keeps the read in range.
        let vertex = min(lbvhVertexOf(triangle, corner), max(lbvhParams.vertexCount, 1u) - 1u);
        let position = lbvhCorner(vertex);
        let at = base + corner * LBVH_TRIANGLE_CORNER_STRIDE;
        lbvhOutTriangles[at] = bitcast<u32>(position.x);
        lbvhOutTriangles[at + 1u] = bitcast<u32>(position.y);
        lbvhOutTriangles[at + 2u] = bitcast<u32>(position.z);
    }
    var flags = 0u;
    if (lbvhParams.hasFlags == 1u) {
        flags = lbvhFlags[triangle];
    }
    lbvhOutTriangles[base + LBVH_TRIANGLE_ORIGINAL_INDEX] = triangle;
    lbvhOutTriangles[base + LBVH_TRIANGLE_FLAGS] = flags;
    lbvhOutTriangles[base + LBVH_TRIANGLE_PADDING] = 0u;
}
`;
}
