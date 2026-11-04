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
 *   through its scratch node): a thread carrying an inner node stores its box,
 *   reference, SAH cost and height into its side of the parent before the
 *   exchange, so the second thread reads one place. A leaf (one triangle) is
 *   never stored: the second thread knows from the range it got back whether
 *   its sibling is a single triangle, and at which sorted position, and reads
 *   its box itself; the emit pass does the same. That halves the handed-over
 *   words (B1 profile: the climb went from 12 to TODO ms on Sponza).
 * - **SAH leaf collapse in the same climb:** the thread that finishes a node
 *   knows both children's costs, so it decides then whether the node would be
 *   cheaper as one leaf (at most `maxLeaf` triangles; the subtree's triangles
 *   are already neighbours in sorted order). Compaction and the final layout
 *   come in later dispatches.
 * - **Subgroups:** sibling leaves held by neighbouring lanes finish their node
 *   in registers, as in the B1a refit: the check compares positions and sides,
 *   never lane numbers, so any subgroup size or lane layout is right.
 * - **Batches** (HIPRT has a separate batch builder): every mesh of a batch
 *   shares each dispatch. A mesh table says where each mesh's data sits in the
 *   batch's buffers, and a tile table gives each workgroup its mesh and first
 *   triangle, so a workgroup never spans two meshes.
 *
 * Words more than one thread reads in one dispatch (a mesh's stats) are read
 * through plain read-only views of the same buffers wherever the dispatch
 * doesn't write them: an atomic load is a read-modify-write on some backends,
 * and millions of them on one word serialise (B1 profile: 23 ms of Sponza's
 * build went to the Morton kernel's six).
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

/** Threads per workgroup of every LBVH kernel, and so triangles per tile of the tile table. */
export const LBVH_WORKGROUP_SIZE = SCAN_WORKGROUP_SIZE;

/** Words of scratch per inner node: two child slots of {@link LBVH_SLOT_WORDS}. */
export const LBVH_SCRATCH_NODE_WORDS = 16;
const LBVH_SLOT_WORDS = 8;

/** Words of `lbvhNodeInfo` per inner node: first sorted position, count and collapse flag, parent. */
export const LBVH_NODE_INFO_WORDS = 3;

/**
 * Words of scratch per inner node besides its slots, in regions after all the
 * slots: the exchange word, then a height per child slot.
 */
export const LBVH_STATE_WORDS = 3;

/**
 * The first round of the climb stops a thread once the node it finished has
 * this many triangles, and queues its arrival at the parent for the second
 * round. Why: a climb is a chain of dependent atomic round trips, and in a
 * subgroup one lane keeps climbing long after the others stopped, so the
 * subgroup stays resident; a single round made the climb Sponza's slowest
 * step (13 ms) whatever its atomics cost. The second round has few threads,
 * each with a long climb, which fits the GPU in one or two waves.
 */
export const LBVH_FIRST_ROUND_TRIANGLES = 64;

/** Re-reads of a handed-over word before the wait gives up (see the top of this file). */
const MAX_SPINS = 4096;

/** The clear pass's mark on every handed-over word, and "no exchange yet". A NaN pattern, never a real value. */
const UNWRITTEN = 0xffffffff;

/** A child reference with this bit is a leaf: a single triangle at that sorted position. */
const LEAF_BIT = 0x80000000;

/** In `lbvhNodeInfo`'s second word: the node is cheaper as one leaf. The low bits are its triangle count. */
const COLLAPSE_BIT = 0x80000000;

/** Word positions in a mesh's stats (one block of {@link LBVH_STATS_WORDS} per mesh). */
export const LBVH_STAT = {
    /** Centroid bounds as order-preserving integers: min x, y, z, then max x, y, z. */
    centroidMin: 0,
    centroidMax: 3,
    /** Split index of the root in the mesh's radix tree. */
    rootNode: 6,
    /** Where the mesh's first node is in the output nodes and parents, in nodes. */
    nodeOffset: 7,
    nodeCount: 8,
    leafCount: 9,
    maxDepth: 10,
    /** The mesh's box as f32 bits: min x, y, z, then max x, y, z. */
    bounds: 11,
    /** Handed-over words still unwritten after the bounded wait, or climbs that didn't end. Must be 0. */
    incomplete: 17,
    /** Threads that finished the root: must be 1. */
    rootsFinished: 18,
    /** Triangle corners that named a vertex past the end. Must be 0. */
    badIndices: 19,
} as const;
export const LBVH_STATS_WORDS = 32;
export const LBVH_STATS_BYTES = LBVH_STATS_WORDS * 4;

/** Word positions in a mesh's entry of the mesh table (one block of {@link LBVH_MESH_WORDS} per mesh). */
export const LBVH_MESH = {
    /** The mesh's first triangle in the batch's triangle-indexed buffers (boxes, keys, values, output triangles). */
    triangleStart: 0,
    triangleCount: 1,
    /** The mesh's first inner node in the batch's node-indexed scratch (and its node count there: at least 1). */
    innerStart: 2,
    innerCount: 3,
    vertexCount: 4,
    /** Float index of vertex 0's x in the batch's positions, and floats from one vertex to the next. */
    positionOffset: 5,
    positionStride: 6,
    /** 0: a plain triangle list, 16 or 32: index bits. */
    indexFormat: 7,
    /** The mesh's first index in the batch's indices, counted in indices of its format. */
    indexOffset: 8,
    /** The mesh's first flag word in the batch's flags. */
    flagsOffset: 9,
    hasFlags: 10,
} as const;
export const LBVH_MESH_WORDS = 16;

/** Words per tile in the tile table: the mesh, and the mesh's triangle the tile starts at. */
export const LBVH_TILE_WORDS = 2;

/** Bytes of the uniform block (`LbvhParams`). */
export const LBVH_PARAMS_BYTES = 32;

/** `LbvhMesh.indexFormat` values. */
export const LBVH_INDEX_FORMAT = { none: 0, uint16: 16, uint32: 32 } as const;

const hex = (value: number): string => `0x${value.toString(16)}u`;

export function lbvhWgsl(): string {
    const word = GPU_LAYOUT_NODE_WORD;
    const triangle = GPU_LAYOUT_TRIANGLE_WORD;
    const stat = LBVH_STAT;
    const mesh = LBVH_MESH;
    return /* wgsl */ `enable subgroups;

struct LbvhParams {
    meshCount: u32,
    tileCount: u32,          // workgroups that have work: entries in the tile table
    tilesOffset: u32,        // where the tile table starts in lbvhTables, in words
    innerTotal: u32,         // inner nodes of every mesh together (where lbvhScratch's regions start)
    maxLeaf: u32,            // largest leaf the collapse may make
    traversalCost: f32,      // SAH cost of visiting a node
    intersectionCost: f32,   // SAH cost of testing a triangle
    climbLimit: u32,         // the climb's first round stops at nodes this big (LBVH_FIRST_ROUND_TRIANGLES)
}

// ---- Bindings ----------------------------------------------------------------------
// Pipelines use 'auto' layouts, so each kernel binds only what it uses (at most 8 storage buffers).

@group(0) @binding(0) var<storage, read> lbvhPositions: array<f32>;
@group(0) @binding(1) var<storage, read> lbvhIndices: array<u32>;
@group(0) @binding(2) var<storage, read> lbvhFlags: array<u32>;
// Two per triangle: (min.xyz, 0), (max.xyz, 0).
@group(0) @binding(3) var<storage, read_write> lbvhPrimitiveBoxes: array<vec4f>;
// Morton codes, then (after the sort) in sorted order.
@group(0) @binding(4) var<storage, read_write> lbvhKeys: array<u32>;
// The mesh's own triangle numbers, then (after the sort) the triangle at each sorted position.
@group(0) @binding(5) var<storage, read_write> lbvhValues: array<u32>;
// Handed between climbing threads. [Per inner node, two child slots: (min.xyz, reference), (max.xyz, cost)],
// then [the range end each inner node's first child left, for its second], then [a height per child slot].
@group(0) @binding(6) var<storage, read_write> lbvhScratch: array<atomic<u32>>;
// Per inner node: first sorted position, count | collapse bit, parent. Read only in later dispatches.
@group(0) @binding(8) var<storage, read_write> lbvhNodeInfo: array<u32>;
// Per inner node: 1 if kept, then (after the scan) its place among every mesh's kept nodes. One more at the end.
@group(0) @binding(9) var<storage, read_write> lbvhRank: array<u32>;
@group(0) @binding(10) var<storage, read_write> lbvhOutNodes: array<u32>;
@group(0) @binding(11) var<storage, read_write> lbvhOutTriangles: array<u32>;
@group(0) @binding(12) var<storage, read_write> lbvhOutParents: array<u32>;
@group(0) @binding(13) var<storage, read_write> lbvhStats: array<atomic<u32>>;
@group(0) @binding(14) var<uniform> lbvhParams: LbvhParams;
// [the mesh table] then, at lbvhParams.tilesOffset, [the tile table].
@group(0) @binding(15) var<storage, read> lbvhTables: array<u32>;
// Plain views of lbvhScratch and lbvhStats for dispatches that don't hand words between threads.
@group(0) @binding(16) var<storage, read> lbvhScratchView: array<u32>;
@group(0) @binding(17) var<storage, read> lbvhStatsView: array<u32>;
@group(0) @binding(18) var<storage, read_write> lbvhScratchFill: array<u32>;
// The climb's second round: [how many arrivals], then [an inner node * 2 + side per arrival].
@group(0) @binding(20) var<storage, read_write> lbvhQueue: array<u32>;
// The second round's indirect dispatch: x, y, z workgroups.
@group(0) @binding(21) var<storage, read_write> lbvhContinueArgs: array<u32>;

// ---- Constants (generated from gpu-layout.ts and gpu-lbvh-wgsl.ts) ------------------

const LBVH_WORKGROUP_SIZE: u32 = ${LBVH_WORKGROUP_SIZE}u;
const LBVH_SCRATCH_NODE_WORDS: u32 = ${LBVH_SCRATCH_NODE_WORDS}u;
const LBVH_SLOT_WORDS: u32 = ${LBVH_SLOT_WORDS}u;
const LBVH_NODE_INFO_WORDS: u32 = ${LBVH_NODE_INFO_WORDS}u;
const LBVH_MESH_WORDS: u32 = ${LBVH_MESH_WORDS}u;
const LBVH_TILE_WORDS: u32 = ${LBVH_TILE_WORDS}u;
const LBVH_STATS_WORDS: u32 = ${LBVH_STATS_WORDS}u;
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
const LBVH_STAT_NODE_OFFSET: u32 = ${stat.nodeOffset}u;
const LBVH_STAT_NODE_COUNT: u32 = ${stat.nodeCount}u;
const LBVH_STAT_LEAF_COUNT: u32 = ${stat.leafCount}u;
const LBVH_STAT_MAX_DEPTH: u32 = ${stat.maxDepth}u;
const LBVH_STAT_BOUNDS: u32 = ${stat.bounds}u;
const LBVH_STAT_INCOMPLETE: u32 = ${stat.incomplete}u;
const LBVH_STAT_ROOTS_FINISHED: u32 = ${stat.rootsFinished}u;
const LBVH_STAT_BAD_INDICES: u32 = ${stat.badIndices}u;
// The largest dispatch dimension every device allows.
const LBVH_MAX_WORKGROUPS_PER_DIMENSION: u32 = 65535u;
${WORKGROUP_INDEX_WGSL}

// ---- Meshes and tiles -------------------------------------------------------------------

struct LbvhMesh {
    index: u32,
    triangleStart: u32,
    triangleCount: u32,
    innerStart: u32,
    innerCount: u32,
    vertexCount: u32,
    positionOffset: u32,
    positionStride: u32,
    indexFormat: u32,
    indexOffset: u32,
    flagsOffset: u32,
    hasFlags: u32,
}

fn lbvhMesh(index: u32) -> LbvhMesh {
    let at = index * LBVH_MESH_WORDS;
    return LbvhMesh(
        index,
        lbvhTables[at + ${mesh.triangleStart}u],
        lbvhTables[at + ${mesh.triangleCount}u],
        lbvhTables[at + ${mesh.innerStart}u],
        lbvhTables[at + ${mesh.innerCount}u],
        lbvhTables[at + ${mesh.vertexCount}u],
        lbvhTables[at + ${mesh.positionOffset}u],
        lbvhTables[at + ${mesh.positionStride}u],
        lbvhTables[at + ${mesh.indexFormat}u],
        lbvhTables[at + ${mesh.indexOffset}u],
        lbvhTables[at + ${mesh.flagsOffset}u],
        lbvhTables[at + ${mesh.hasFlags}u],
    );
}

// A workgroup's work: its mesh, and the first of the mesh's triangles (or inner nodes) it covers.
struct LbvhTile {
    mesh: LbvhMesh,
    first: u32,
}

fn lbvhTile(workgroup: vec3u, groups: vec3u) -> LbvhTile {
    let at = lbvhParams.tilesOffset + flatWorkgroupIndex(workgroup, groups) * LBVH_TILE_WORDS;
    return LbvhTile(lbvhMesh(lbvhTables[at]), lbvhTables[at + 1u]);
}

fn lbvhHasTile(workgroup: vec3u, groups: vec3u) -> bool {
    return flatWorkgroupIndex(workgroup, groups) < lbvhParams.tileCount;
}

fn lbvhStatWord(mesh: u32, stat: u32) -> u32 {
    return mesh * LBVH_STATS_WORDS + stat;
}

// The climb's queue of arrivals for its second round, counted after every mesh's stats.
fn lbvhQueueCountWord() -> u32 {
    return lbvhParams.meshCount * LBVH_STATS_WORDS;
}

// The mesh an inner node belongs to: the last whose first inner node is not after it.
fn lbvhMeshOfInner(node: u32) -> LbvhMesh {
    var low = 0u;
    var high = lbvhParams.meshCount;
    while (high - low > 1u) {
        let middle = (low + high) / 2u;
        if (lbvhTables[middle * LBVH_MESH_WORDS + ${mesh.innerStart}u] <= node) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return lbvhMesh(low);
}

// ---- Geometry -------------------------------------------------------------------------

fn lbvhVertexOf(mesh: LbvhMesh, triangle: u32, corner: u32) -> u32 {
    let entry = mesh.indexOffset + triangle * 3u + corner;
    var vertex = triangle * 3u + corner;
    if (mesh.indexFormat == 32u) {
        vertex = lbvhIndices[entry];
    } else if (mesh.indexFormat == 16u) {
        vertex = (lbvhIndices[entry >> 1u] >> ((entry & 1u) * 16u)) & 0xffffu;
    }
    return vertex;
}

fn lbvhCorner(mesh: LbvhMesh, vertex: u32) -> vec3f {
    let at = mesh.positionOffset + vertex * mesh.positionStride;
    return vec3f(lbvhPositions[at], lbvhPositions[at + 1u], lbvhPositions[at + 2u]);
}

struct LbvhBox {
    lower: vec3f,
    upper: vec3f,
}

// The box of a mesh's triangle, by the mesh's own triangle number.
fn lbvhPrimitiveBox(mesh: LbvhMesh, triangle: u32) -> LbvhBox {
    let at = (mesh.triangleStart + triangle) * 2u;
    return LbvhBox(lbvhPrimitiveBoxes[at].xyz, lbvhPrimitiveBoxes[at + 1u].xyz);
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

// One thread per mesh. Every value the later kernels accumulate into starts
// here, so a recorded build can be submitted again and gives the same trees.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhStart(@builtin(global_invocation_id) id: vec3u) {
    let mesh = id.x;
    if (mesh == 0u) {
        lbvhRank[lbvhParams.innerTotal] = 0u;   // the end marker of the node scan
        atomicStore(&lbvhStats[lbvhQueueCountWord()], 0u);
    }
    if (mesh >= lbvhParams.meshCount) {
        return;
    }
    for (var at = 0u; at < LBVH_STATS_WORDS; at += 1u) {
        atomicStore(&lbvhStats[lbvhStatWord(mesh, at)], 0u);
    }
    for (var axis = 0u; axis < 3u; axis += 1u) {
        atomicStore(&lbvhStats[lbvhStatWord(mesh, LBVH_STAT_CENTROID_MIN + axis)], 0xffffffffu);
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
    if (!lbvhHasTile(workgroup, groups)) {
        return;   // the whole workgroup: the index is uniform
    }
    if (localIndex < 3u) {
        atomicStore(&lbvhGroupMin[localIndex], 0xffffffffu);
    }
    workgroupBarrier();

    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let triangle = tile.first + localIndex;
    let hasTriangle = triangle < mesh.triangleCount;
    var centroid = vec3f(0.0);
    if (hasTriangle) {
        var lower = vec3f(LBVH_FLOAT_MAX);
        var upper = vec3f(-LBVH_FLOAT_MAX);
        for (var corner = 0u; corner < 3u; corner += 1u) {
            var vertex = lbvhVertexOf(mesh, triangle, corner);
            if (vertex >= mesh.vertexCount) {
                atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_BAD_INDICES)], 1u);
                vertex = 0u;
            }
            let position = lbvhCorner(mesh, vertex);
            lower = min(lower, position);
            upper = max(upper, position);
        }
        let at = (mesh.triangleStart + triangle) * 2u;
        lbvhPrimitiveBoxes[at] = vec4f(lower, 0.0);
        lbvhPrimitiveBoxes[at + 1u] = vec4f(upper, 0.0);
        centroid = (lower + upper) * 0.5;
    }

    // Reduce in the subgroup, then the workgroup (one mesh), then one global atomic per axis and workgroup.
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
        atomicMin(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_CENTROID_MIN + localIndex)], atomicLoad(&lbvhGroupMin[localIndex]));
        atomicMax(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_CENTROID_MAX + localIndex)], atomicLoad(&lbvhGroupMax[localIndex]));
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
    if (!lbvhHasTile(workgroup, groups)) {
        return;
    }
    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let triangle = tile.first + localIndex;
    if (triangle >= mesh.triangleCount) {
        return;
    }
    var lower: vec3f;
    var upper: vec3f;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        lower[axis] = lbvhFloatFromOrdered(lbvhStatsView[lbvhStatWord(mesh.index, LBVH_STAT_CENTROID_MIN + axis)]);
        upper[axis] = lbvhFloatFromOrdered(lbvhStatsView[lbvhStatWord(mesh.index, LBVH_STAT_CENTROID_MAX + axis)]);
    }
    // Cubic cells over each mesh's own centroid box: 1024 along its longest
    // side, the same cell size on the others. HIPRT's plain code stretches every
    // axis to 1024 cells, which splits a flat or long mesh across its thin side
    // as often as along its long ones: SAH cost 2.4-2.6 times tinybvh's on slabs,
    // rods and flat grids, against 1.02-1.22 with cubic cells (NOTES, B1).
    let extent = upper - lower;
    let longest = max(extent.x, max(extent.y, extent.z));
    let scale = select(0.0, 1024.0 / longest, longest > 0.0);
    let box = lbvhPrimitiveBox(mesh, triangle);
    let centroid = (box.lower + box.upper) * 0.5;
    let cell = vec3u(clamp((centroid - lower) * scale, vec3f(0.0), vec3f(1023.0)));
    lbvhKeys[mesh.triangleStart + triangle] = lbvhExpandBits(cell.x) * 4u + lbvhExpandBits(cell.y) * 2u + lbvhExpandBits(cell.z);
    lbvhValues[mesh.triangleStart + triangle] = triangle;
}

// ---- 4. The climb (HIPRT EmitTopologyAndFitBounds; Apetrei 2014) -------------------------

// One child of an inner node, as the climbing thread carries it. Positions and
// split indices are the mesh's own; the batch's buffers are reached through
// the mesh's starts.
struct LbvhChild {
    lower: vec3f,
    upper: vec3f,
    reference: u32,   // LBVH_LEAF_BIT | sorted position, or an inner node's split index
    cost: f32,        // SAH cost of the subtree: the cheaper of one leaf and an inner node
    height: u32,      // levels of inner nodes below, 0 if the child ends up a leaf
}

fn lbvhExchangeWord(innerNode: u32) -> u32 {
    return lbvhParams.innerTotal * LBVH_SCRATCH_NODE_WORDS + innerNode;
}

fn lbvhHeightWord(innerNode: u32, side: u32) -> u32 {
    return lbvhParams.innerTotal * (LBVH_SCRATCH_NODE_WORDS + 1u) + innerNode * 2u + side;
}

// One thread per inner node: marks every handed-over word unwritten, with
// plain stores through the fill views (the climb's atomics see them: it is a
// later dispatch).
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhClear(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (!lbvhHasTile(workgroup, groups)) {
        return;
    }
    let tile = lbvhTile(workgroup, groups);
    let local = tile.first + localIndex;
    if (local >= tile.mesh.innerCount) {
        return;
    }
    let node = tile.mesh.innerStart + local;
    let base = node * LBVH_SCRATCH_NODE_WORDS;
    for (var at = 0u; at < LBVH_SCRATCH_NODE_WORDS; at += 1u) {
        lbvhScratchFill[base + at] = LBVH_UNWRITTEN;
    }
    lbvhScratchFill[lbvhExchangeWord(node)] = LBVH_UNWRITTEN;
    lbvhScratchFill[lbvhHeightWord(node, 0u)] = LBVH_UNWRITTEN;
    lbvhScratchFill[lbvhHeightWord(node, 1u)] = LBVH_UNWRITTEN;
}

fn lbvhStoreChild(mesh: LbvhMesh, split: u32, side: u32, child: LbvhChild) {
    let node = mesh.innerStart + split;
    let base = node * LBVH_SCRATCH_NODE_WORDS + side * LBVH_SLOT_WORDS;
    atomicStore(&lbvhScratch[base], bitcast<u32>(child.lower.x));
    atomicStore(&lbvhScratch[base + 1u], bitcast<u32>(child.lower.y));
    atomicStore(&lbvhScratch[base + 2u], bitcast<u32>(child.lower.z));
    atomicStore(&lbvhScratch[base + 3u], child.reference);
    atomicStore(&lbvhScratch[base + 4u], bitcast<u32>(child.upper.x));
    atomicStore(&lbvhScratch[base + 5u], bitcast<u32>(child.upper.y));
    atomicStore(&lbvhScratch[base + 6u], bitcast<u32>(child.upper.z));
    atomicStore(&lbvhScratch[base + 7u], bitcast<u32>(child.cost));
    atomicStore(&lbvhScratch[lbvhHeightWord(node, side)], child.height);
}

// One handed-over word, waiting (bounded) until the clear pass's mark is gone.
fn lbvhWaitScratch(mesh: u32, at: u32) -> u32 {
    var value = atomicLoad(&lbvhScratch[at]);
    for (var spins = 0u; value == LBVH_UNWRITTEN; spins += 1u) {
        if (spins == LBVH_MAX_SPINS) {
            atomicAdd(&lbvhStats[lbvhStatWord(mesh, LBVH_STAT_INCOMPLETE)], 1u);
            break;
        }
        value = atomicLoad(&lbvhScratch[at]);
    }
    return value;
}

// The sibling's slot during the climb: its thread stored it before its exchange, which this thread saw.
fn lbvhWaitChild(mesh: LbvhMesh, split: u32, side: u32) -> LbvhChild {
    let node = mesh.innerStart + split;
    let base = node * LBVH_SCRATCH_NODE_WORDS + side * LBVH_SLOT_WORDS;
    let m = mesh.index;
    return LbvhChild(
        vec3f(bitcast<f32>(lbvhWaitScratch(m, base)), bitcast<f32>(lbvhWaitScratch(m, base + 1u)), bitcast<f32>(lbvhWaitScratch(m, base + 2u))),
        vec3f(bitcast<f32>(lbvhWaitScratch(m, base + 4u)), bitcast<f32>(lbvhWaitScratch(m, base + 5u)), bitcast<f32>(lbvhWaitScratch(m, base + 6u))),
        lbvhWaitScratch(m, base + 3u),
        bitcast<f32>(lbvhWaitScratch(m, base + 7u)),
        lbvhWaitScratch(m, lbvhHeightWord(node, side)),
    );
}

// A single triangle as a child, from its sorted position: what its thread would have handed over.
fn lbvhLeafChild(mesh: LbvhMesh, position: u32) -> LbvhChild {
    let box = lbvhPrimitiveBox(mesh, lbvhValues[mesh.triangleStart + position]);
    return LbvhChild(box.lower, box.upper, LBVH_LEAF_BIT | position, lbvhParams.intersectionCost * lbvhHalfArea(box.lower, box.upper), 0u);
}

// HIPRT findHighestDifferentBit, as a pair: (keys differing, positions differing).
// Bigger means the two neighbours differ in a higher bit. Positions k and k + 1 only.
fn lbvhDelta(mesh: LbvhMesh, position: u32) -> vec2u {
    let at = mesh.triangleStart + position;
    return vec2u(lbvhKeys[at] ^ lbvhKeys[at + 1u], position ^ (position + 1u));
}

fn lbvhLess(a: vec2u, b: vec2u) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

struct LbvhParent {
    split: u32,
    side: u32,   // 0: the range is the parent's left child
}

// HIPRT findParent for the range [first, end) (end exclusive), which is not the
// whole mesh: the parent splits next to whichever end has the closer neighbour.
fn lbvhFindParent(mesh: LbvhMesh, first: u32, end: u32) -> LbvhParent {
    if (first == 0u || (end != mesh.triangleCount && lbvhLess(lbvhDelta(mesh, end - 1u), lbvhDelta(mesh, first - 1u)))) {
        return LbvhParent(end - 1u, 0u);
    }
    return LbvhParent(first - 1u, 1u);
}

fn lbvhInfoWord(mesh: LbvhMesh, split: u32, word: u32) -> u32 {
    return (mesh.innerStart + split) * LBVH_NODE_INFO_WORDS + word;
}

// Both children are known: the node's box, its SAH cost, and whether it is cheaper as one leaf.
fn lbvhFinishNode(mesh: LbvhMesh, split: u32, first: u32, end: u32, a: LbvhChild, b: LbvhChild) -> LbvhChild {
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
    let isRoot = first == 0u && end == mesh.triangleCount;
    let collapse = !isRoot && leafCost <= innerCost;
    lbvhNodeInfo[lbvhInfoWord(mesh, split, 0u)] = first;
    lbvhNodeInfo[lbvhInfoWord(mesh, split, 1u)] = count | select(0u, LBVH_COLLAPSE_BIT, collapse);
    let height = 1u + max(a.height, b.height);
    return LbvhChild(lower, upper, split, min(innerCost, leafCost), select(height, 0u, collapse));
}

fn lbvhFinishRoot(mesh: LbvhMesh, split: u32, root: LbvhChild) {
    lbvhNodeInfo[lbvhInfoWord(mesh, split, 2u)] = LBVH_NO_PARENT;
    atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_ROOT_NODE)], split);
    atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_MAX_DEPTH)], root.height);
    for (var axis = 0u; axis < 3u; axis += 1u) {
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_BOUNDS + axis)], bitcast<u32>(root.lower[axis]));
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_BOUNDS + 3u + axis)], bitcast<u32>(root.upper[axis]));
    }
    atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_ROOTS_FINISHED)], 1u);
}

// Whether lane 'lane' is running, from a subgroupBallot mask (bit lane % 32 of word lane / 32).
fn lbvhLaneRunning(running: vec4u, lane: u32) -> bool {
    return ((running[lane / 32u] >> (lane % 32u)) & 1u) == 1u;
}

// Queues a finished node's arrival at its parent for the climb's second round.
// Its data are already in its side of the parent.
fn lbvhQueueArrival(mesh: LbvhMesh, parent: LbvhParent) {
    let at = atomicAdd(&lbvhStats[lbvhQueueCountWord()], 1u);
    lbvhQueue[1u + at] = (mesh.innerStart + parent.split) * 2u + parent.side;
}

// Carries 'child' (the range [first, end)) up from its arrival at 'parent':
// at each node the first thread to arrive stops, and the second (which then
// knows the node's whole range) finishes the node and carries on. Stops early,
// queueing the arrival, once a finished node has 'limit' triangles or more.
// An inner child's data must already be in its side of 'parent'.
fn lbvhClimbFrom(mesh: LbvhMesh, start: LbvhChild, startFirst: u32, startEnd: u32, startParent: LbvhParent, limit: u32) {
    var child = start;
    var first = startFirst;
    var end = startEnd;
    var parent = startParent;
    // The bound only guards against a broken climb: a real one ends at the root.
    for (var step = 0u; step < mesh.triangleCount; step += 1u) {
        let split = parent.split;
        // Leave this side's far end for the sibling's thread; the first to arrive gets "unwritten" back and stops.
        let other = atomicExchange(&lbvhScratch[lbvhExchangeWord(mesh.innerStart + split)], select(first, end, parent.side == 1u));
        if (other == LBVH_UNWRITTEN) {
            return;
        }
        // The sibling's range is [other, split + 1) on the left or [split + 1, other) on the right; one position is a leaf.
        var sibling: LbvhChild;
        if (parent.side == 1u) {
            first = other;
            if (other == split) {
                sibling = lbvhLeafChild(mesh, split);
            } else {
                sibling = lbvhWaitChild(mesh, split, 0u);
            }
        } else {
            end = other;
            if (other == split + 2u) {
                sibling = lbvhLeafChild(mesh, split + 1u);
            } else {
                sibling = lbvhWaitChild(mesh, split, 1u);
            }
        }
        child = lbvhFinishNode(mesh, split, first, end, child, sibling);
        if (first == 0u && end == mesh.triangleCount) {
            lbvhFinishRoot(mesh, split, child);
            return;
        }
        parent = lbvhFindParent(mesh, first, end);
        lbvhNodeInfo[lbvhInfoWord(mesh, split, 2u)] = parent.split;
        // The node's data go into its side of the parent, for the sibling's thread (if it comes second) and the emit pass.
        lbvhStoreChild(mesh, parent.split, parent.side, child);
        if (end - first >= limit) {
            lbvhQueueArrival(mesh, parent);
            return;
        }
    }
    atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_INCOMPLETE)], 1u);
}

// The climb's first round: one thread per sorted position, up to nodes of
// lbvhParams.climbLimit triangles (see LBVH_FIRST_ROUND_TRIANGLES).
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhClimb(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) laneCount: u32,
) {
    if (!lbvhHasTile(workgroup, groups)) {
        return;   // the whole workgroup: the index is uniform
    }
    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let count = mesh.triangleCount;
    let position = tile.first + localIndex;
    let climbs = position < count && count > 1u;
    var child = LbvhChild(vec3f(0.0), vec3f(0.0), LBVH_LEAF_BIT | position, 0.0, 0u);
    var first = position;
    var end = position + 1u;
    var parent = LbvhParent(LBVH_NO_PARENT, 0u);
    if (climbs) {
        child = lbvhLeafChild(mesh, position);
        parent = lbvhFindParent(mesh, first, end);
    }

    // Subgroup step, in uniform control flow (no return above but the whole
    // workgroup's). Sibling leaves are neighbouring positions; when
    // neighbouring lanes hold them, the lower lane finishes their node in
    // registers. Only positions and sides are compared (a workgroup holds one
    // mesh), so this is right for any subgroup size and lane layout.
    let running = subgroupBallot(climbs);
    let positionAbove = subgroupShuffleDown(position, 1u);
    let parentAbove = subgroupShuffleDown(parent.split, 1u);
    let sideAbove = subgroupShuffleDown(parent.side, 1u);
    let lowerAbove = subgroupShuffleDown(child.lower, 1u);
    let upperAbove = subgroupShuffleDown(child.upper, 1u);
    let costAbove = subgroupShuffleDown(child.cost, 1u);
    let positionBelow = subgroupShuffleUp(position, 1u);
    let parentBelow = subgroupShuffleUp(parent.split, 1u);
    let sideBelow = subgroupShuffleUp(parent.side, 1u);
    if (!climbs) {
        return;
    }
    let pairedAbove = parent.side == 0u && lane + 1u < laneCount && lbvhLaneRunning(running, lane + 1u) &&
        positionAbove == position + 1u && parentAbove == parent.split && sideAbove == 1u;
    let pairedBelow = parent.side == 1u && lane > 0u && lbvhLaneRunning(running, lane - 1u) &&
        positionBelow + 1u == position && parentBelow == parent.split && sideBelow == 0u;
    if (pairedBelow) {
        return;   // the lane below holds the left sibling and finishes the node
    }
    if (pairedAbove) {
        // Two leaves: nothing was stored for them, the emit pass reads leaves from their positions.
        let split = parent.split;
        let right = LbvhChild(lowerAbove, upperAbove, LBVH_LEAF_BIT | (position + 1u), costAbove, 0u);
        end = position + 2u;
        child = lbvhFinishNode(mesh, split, first, end, child, right);
        if (first == 0u && end == count) {
            lbvhFinishRoot(mesh, split, child);
            return;
        }
        parent = lbvhFindParent(mesh, first, end);
        lbvhNodeInfo[lbvhInfoWord(mesh, split, 2u)] = parent.split;
        lbvhStoreChild(mesh, parent.split, parent.side, child);
        if (end - first >= lbvhParams.climbLimit) {
            lbvhQueueArrival(mesh, parent);
            return;
        }
    }
    lbvhClimbFrom(mesh, child, first, end, parent, lbvhParams.climbLimit);
}

// One thread: the second round's dispatch size, from the queue's count.
@compute @workgroup_size(1)
fn lbvhContinueArgs() {
    let count = atomicLoad(&lbvhStats[lbvhQueueCountWord()]);
    lbvhQueue[0] = count;
    let workgroups = (count + LBVH_WORKGROUP_SIZE - 1u) / LBVH_WORKGROUP_SIZE;
    let x = min(workgroups, LBVH_MAX_WORKGROUPS_PER_DIMENSION);
    lbvhContinueArgs[0] = x;
    lbvhContinueArgs[1] = select(1u, (workgroups + x - 1u) / x, x > 0u);
    lbvhContinueArgs[2] = 1u;
}

// The climb's second round: one thread per queued arrival, to the root.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhClimbOn(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let at = flatWorkgroupIndex(workgroup, groups) * LBVH_WORKGROUP_SIZE + localIndex;
    if (at >= lbvhQueue[0]) {
        return;
    }
    let arrival = lbvhQueue[1u + at];
    let node = arrival >> 1u;
    let side = arrival & 1u;
    let mesh = lbvhMeshOfInner(node);
    let split = node - mesh.innerStart;
    // Stored by the first round, a dispatch ago.
    let child = lbvhWaitChild(mesh, split, side);
    let first = lbvhNodeInfo[lbvhInfoWord(mesh, child.reference, 0u)];
    let end = first + (lbvhNodeInfo[lbvhInfoWord(mesh, child.reference, 1u)] & LBVH_COUNT_MASK);
    lbvhClimbFrom(mesh, child, first, end, LbvhParent(split, side), 0xffffffffu);
}

// ---- 5. Which inner nodes stay ----------------------------------------------------------

// A node stays unless it or an ancestor became a leaf. Only ancestors with at
// most maxLeaf triangles can have, and counts grow on the way up, so the walk
// is short.
fn lbvhKept(mesh: LbvhMesh, split: u32) -> bool {
    if ((lbvhNodeInfo[lbvhInfoWord(mesh, split, 1u)] & LBVH_COLLAPSE_BIT) != 0u) {
        return false;
    }
    var above = lbvhNodeInfo[lbvhInfoWord(mesh, split, 2u)];
    for (var guard = 0u; guard <= lbvhParams.maxLeaf && above != LBVH_NO_PARENT; guard += 1u) {
        let info = lbvhNodeInfo[lbvhInfoWord(mesh, above, 1u)];
        if ((info & LBVH_COUNT_MASK) > lbvhParams.maxLeaf) {
            return true;
        }
        if ((info & LBVH_COLLAPSE_BIT) != 0u) {
            return false;
        }
        above = lbvhNodeInfo[lbvhInfoWord(mesh, above, 2u)];
    }
    return true;
}

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhMark(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (!lbvhHasTile(workgroup, groups)) {
        return;
    }
    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let split = tile.first + localIndex;
    if (split >= mesh.innerCount) {
        return;
    }
    // A one-triangle mesh has no inner node but still needs one node, its root.
    lbvhRank[mesh.innerStart + split] = select(select(0u, 1u, lbvhKept(mesh, split)), 1u, mesh.triangleCount == 1u);
}

// ---- 6. The tracer's layout ----------------------------------------------------------------

// Kept nodes are numbered by the scan in split order across the whole batch; a
// mesh's nodes are its run of numbers. Within the run the root swaps with
// whichever node got the run's first number, because a mesh's root is its node 0.
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

// A leaf child in the emit pass: the box of the triangle record at its sorted
// position (written by lbvhTriangles, which runs first), the same corners the
// climb's boxes came from.
fn lbvhTriangleBox(mesh: LbvhMesh, position: u32) -> LbvhBox {
    let base = (mesh.triangleStart + position) * LBVH_TRIANGLE_WORDS;
    var lower = vec3f(LBVH_FLOAT_MAX);
    var upper = vec3f(-LBVH_FLOAT_MAX);
    for (var corner = 0u; corner < 3u; corner += 1u) {
        let at = base + corner * LBVH_TRIANGLE_CORNER_STRIDE;
        let point = vec3f(bitcast<f32>(lbvhOutTriangles[at]), bitcast<f32>(lbvhOutTriangles[at + 1u]), bitcast<f32>(lbvhOutTriangles[at + 2u]));
        lower = min(lower, point);
        upper = max(upper, point);
    }
    return LbvhBox(lower, upper);
}

// A slot in a later dispatch, when every store has landed: read through the plain view.
fn lbvhReadChild(mesh: LbvhMesh, split: u32, side: u32) -> LbvhChild {
    let base = (mesh.innerStart + split) * LBVH_SCRATCH_NODE_WORDS + side * LBVH_SLOT_WORDS;
    return LbvhChild(
        vec3f(bitcast<f32>(lbvhScratchView[base]), bitcast<f32>(lbvhScratchView[base + 1u]), bitcast<f32>(lbvhScratchView[base + 2u])),
        vec3f(bitcast<f32>(lbvhScratchView[base + 4u]), bitcast<f32>(lbvhScratchView[base + 5u]), bitcast<f32>(lbvhScratchView[base + 6u])),
        lbvhScratchView[base + 3u],
        bitcast<f32>(lbvhScratchView[base + 7u]),
        0u,
    );
}

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhEmit(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (!lbvhHasTile(workgroup, groups)) {
        return;
    }
    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let split = tile.first + localIndex;
    if (split >= mesh.innerCount) {
        return;
    }
    // The mesh's run of node numbers starts at the scanned value of its first inner node.
    let nodeOffset = lbvhRank[mesh.innerStart];
    if (mesh.triangleCount == 1u) {
        // One triangle: the root holds it on the left and nothing on the right.
        let box = lbvhTriangleBox(mesh, 0u);
        let out = nodeOffset * LBVH_NODE_WORDS;
        lbvhWriteChild(out, 0u, box.lower, box.upper, 0u, 1u);
        lbvhWriteChild(out, 1u, vec3f(0.0), vec3f(0.0), LBVH_EMPTY_CHILD, 0u);
        lbvhOutParents[nodeOffset] = LBVH_NO_PARENT;
        return;
    }
    if (!lbvhKept(mesh, split)) {
        return;
    }
    let root = lbvhStatsView[lbvhStatWord(mesh.index, LBVH_STAT_ROOT_NODE)];
    let rootRank = lbvhRank[mesh.innerStart + root] - nodeOffset;
    let out = lbvhOutputIndex(lbvhRank[mesh.innerStart + split] - nodeOffset, rootRank);
    // The left child covers [first, split + 1), the right [split + 1, first + count).
    let first = lbvhNodeInfo[lbvhInfoWord(mesh, split, 0u)];
    let end = first + (lbvhNodeInfo[lbvhInfoWord(mesh, split, 1u)] & LBVH_COUNT_MASK);
    for (var side = 0u; side < 2u; side += 1u) {
        let leafPosition = select(first, split + 1u, side == 1u);
        let isLeaf = select(split + 1u - first, end - split - 1u, side == 1u) == 1u;
        var child: LbvhChild;
        var index: u32;
        var count: u32;
        if (isLeaf) {
            let box = lbvhTriangleBox(mesh, leafPosition);
            child = LbvhChild(box.lower, box.upper, LBVH_LEAF_BIT | leafPosition, 0.0, 0u);
            index = leafPosition;
            count = 1u;
        } else {
            child = lbvhReadChild(mesh, split, side);
            let info = lbvhNodeInfo[lbvhInfoWord(mesh, child.reference, 1u)];
            if ((info & LBVH_COLLAPSE_BIT) != 0u) {
                // A subtree's triangles are neighbours in sorted order: the leaf is that run.
                index = lbvhNodeInfo[lbvhInfoWord(mesh, child.reference, 0u)];
                count = info & LBVH_COUNT_MASK;
            } else {
                index = lbvhOutputIndex(lbvhRank[mesh.innerStart + child.reference] - nodeOffset, rootRank);
                count = 0u;
            }
        }
        lbvhWriteChild((nodeOffset + out) * LBVH_NODE_WORDS, side, child.lower, child.upper, index, count);
    }
    var parent = LBVH_NO_PARENT;
    if (split != root) {
        let above = lbvhNodeInfo[lbvhInfoWord(mesh, split, 2u)];
        parent = lbvhOutputIndex(lbvhRank[mesh.innerStart + above] - nodeOffset, rootRank);
    }
    lbvhOutParents[nodeOffset + out] = parent;
}

// One thread per mesh: where its nodes landed and how many there are. A binary
// tree has one leaf more than inner nodes; a one-triangle mesh has one of each.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhFinish(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= lbvhParams.meshCount) {
        return;
    }
    let mesh = lbvhMesh(id.x);
    let nodeOffset = lbvhRank[mesh.innerStart];
    // The scanned value after the mesh's last inner node: the next mesh's offset, or the end marker's.
    let nodeCount = lbvhRank[mesh.innerStart + mesh.innerCount] - nodeOffset;
    atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_NODE_OFFSET)], nodeOffset);
    atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_NODE_COUNT)], nodeCount);
    if (mesh.triangleCount == 1u) {
        let box = lbvhPrimitiveBox(mesh, 0u);
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_LEAF_COUNT)], 1u);
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_MAX_DEPTH)], 1u);
        for (var axis = 0u; axis < 3u; axis += 1u) {
            atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_BOUNDS + axis)], bitcast<u32>(box.lower[axis]));
            atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_BOUNDS + 3u + axis)], bitcast<u32>(box.upper[axis]));
        }
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_ROOTS_FINISHED)], 1u);
    } else {
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_LEAF_COUNT)], nodeCount + 1u);
    }
}

// Triangles in tree order (sorted order): three corners as they are, the original index and the flags.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn lbvhTriangles(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (!lbvhHasTile(workgroup, groups)) {
        return;
    }
    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let position = tile.first + localIndex;
    if (position >= mesh.triangleCount) {
        return;
    }
    let triangle = lbvhValues[mesh.triangleStart + position];
    let base = (mesh.triangleStart + position) * LBVH_TRIANGLE_WORDS;
    for (var corner = 0u; corner < 3u; corner += 1u) {
        // A bad index was counted by lbvhBoxes (and the build refused); clamping only keeps the read in range.
        let vertex = min(lbvhVertexOf(mesh, triangle, corner), max(mesh.vertexCount, 1u) - 1u);
        let position3 = lbvhCorner(mesh, vertex);
        let at = base + corner * LBVH_TRIANGLE_CORNER_STRIDE;
        lbvhOutTriangles[at] = bitcast<u32>(position3.x);
        lbvhOutTriangles[at + 1u] = bitcast<u32>(position3.y);
        lbvhOutTriangles[at + 2u] = bitcast<u32>(position3.z);
    }
    var flags = 0u;
    if (mesh.hasFlags == 1u) {
        flags = lbvhFlags[mesh.flagsOffset + triangle];
    }
    lbvhOutTriangles[base + LBVH_TRIANGLE_ORIGINAL_INDEX] = triangle;
    lbvhOutTriangles[base + LBVH_TRIANGLE_FLAGS] = flags;
    lbvhOutTriangles[base + LBVH_TRIANGLE_PADDING] = 0u;
}
`;
}
