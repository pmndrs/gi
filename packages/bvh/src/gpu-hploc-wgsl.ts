/*!
 * The H-PLOC merge below is ported from tinybvh's BuildHPLOC (tiny_bvh_base.h);
 * its licence (packages/bvh/licenses/tinybvh.txt):
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2024-2026, Jacco Bikker / Breda University of Applied Sciences.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * WGSL of the GPU H-PLOC builder (stage B2, gpu-hploc.ts). It builds on the
 * LBVH module (gpu-lbvh-wgsl.ts, ported from HIPRT): the same mesh and tile
 * tables, triangle boxes, Morton codes and Apetrei climb, with H-PLOC's
 * merging (Benthin et al., HPG 2024) on the way up. The merge is tinybvh's
 * `BuildHPLOC` (MIT) step for step, so the trees are tinybvh's when the Morton
 * codes are (`'tinybvh-63'`), except on exact area ties, which pair up
 * instead of chaining the tree (hploc-cpu.ts, "Ties"). The CPU twin is hploc-cpu.ts.
 *
 * **The climb.** One thread per sorted triangle climbs the radix tree. At
 * each node the first thread to arrive stops; the second, which then knows
 * the node's whole range, carries on. A node holds clusters: those of its
 * left child, then those of its right child. A node of at most
 * {@link HPLOC_MERGE_THRESHOLD} triangles holds its triangles as they are,
 * which needs no storage: its range says what they are. A bigger node (and
 * the root) is merged: PLOC brings its clusters down to the threshold (the
 * root's to one), and its clusters are stored for its parent.
 *
 * **Merging in a subgroup.** A node holds at most {@link HPLOC_MAX_CLUSTERS}
 * clusters while it is merged (two children of at most the threshold each),
 * and the whole subgroup merges it together, one node at a time: cluster `c`
 * sits in lane `c % size`, slot `c / size`, where each lane has
 * `HPLOC_SLOTS = ceil(32 / smallest subgroup size)` slots (one on a 32-wide
 * GPU, four on Intel's UHD at 8 wide). The nearest-neighbour search, the
 * mutual check and the compaction move clusters between lanes with shuffles
 * by cluster number, and the compaction's order is the clusters' order,
 * never a lane number. So the merge is the same for every subgroup size and
 * lane layout, and so is the tree. The paper and NexusBVH use one cluster per
 * lane of a 32-wide warp, which is the one-slot case here.
 *
 * The subgroup stays together around the merges (lanes that stopped climbing
 * help), so the subgroup operations run in control flow that is uniform in
 * the subgroup but not in the workgroup. WGSL's uniformity analysis only knows
 * workgroups, so the functions that do it switch its `subgroup_uniformity`
 * check off.
 *
 * **Memory visibility** (as in the LBVH, gpu-lbvh-wgsl.ts): WGSL has no fence
 * across workgroups, so every word one thread hands to another in the climb
 * is a 32-bit atomic, written once after a clear pass marked it unwritten, and
 * the reader waits, bounded, for the mark to go:
 *
 * - the exchange word of each radix-tree node, and a word per side with that
 *   child's cluster count and level (height in the radix tree);
 * - the record of each node the merges make: its box, SAH cost, triangle
 *   count, collapse flag and height;
 * - a stored node's clusters, at the start of its range of sorted positions,
 *   as tinybvh keeps them. These positions are written again on the way up
 *   (a parent's clusters go where its left child's were), so each entry
 *   carries the level of the node that wrote it and the reader waits for the
 *   level it expects. Nodes that contain a position form a chain with a
 *   different level each, so a stale entry never passes for the right one.
 *
 * A wait that runs out is counted, and the builder refuses the tree.
 *
 * **The layout** comes after the climb, in later dispatches: the SAH leaf
 * collapse was decided as each node was made (B1's rule), a scan numbers the
 * nodes that stay, and each triangle goes to its place in depth-first leaf
 * order, which keeps every leaf's triangles together.
 */

import { HPLOC_MAX_CLUSTERS, HPLOC_MERGE_THRESHOLD, HPLOC_SEARCH_RADIUS } from './hploc-cpu.ts';
import { LBVH_STAT, LBVH_STATS_WORDS, LBVH_WORKGROUP_SIZE, lbvhKernelsWgsl } from './gpu-lbvh-wgsl.ts';

export { HPLOC_MAX_CLUSTERS, HPLOC_MERGE_THRESHOLD, HPLOC_SEARCH_RADIUS };

/** Words per radix-tree node in the climb's scratch: the exchange word, then one per side. */
export const HPLOC_CLIMB_WORDS = 3;

/** Words per node the merges make: its box (min xyz, max xyz), SAH cost, info. */
export const HPLOC_RECORD_WORDS = 8;

/** Bits of a stored cluster entry that hold the writer's level; the cluster number is above them. */
const LEVEL_BITS = 7;

/** Highest level a radix-tree node may have: one below the "unwritten" pattern's. */
export const HPLOC_MAX_LEVEL = (1 << LEVEL_BITS) - 2;

/** Most triangles one mesh may have: cluster numbers (twice the triangles) must fit above the level bits. */
export const HPLOC_MAX_MESH_TRIANGLES = 2 ** (32 - LEVEL_BITS - 1);

/** A record's info word: triangles above bit 8, the collapse flag at bit 7, the height (0 if collapsed) below. */
const INFO_TRIANGLES_SHIFT = 8;
const INFO_COLLAPSE_BIT = 0x80;
const INFO_HEIGHT_MASK = 0x7f;

/** H-PLOC's words in a mesh's stats, after the LBVH's (gpu-lbvh-wgsl.ts, LBVH_STAT). */
export const HPLOC_STAT = {
    /** The root's cluster number, written when the root is merged. */
    root: 26,
    /** Nodes the merges made: must end at triangles - 1. */
    nodesMade: 27,
    /** Merge rounds that found no mutual pair (NaN boxes). Must be 0. */
    failedMerges: 28,
    /** Subgroups too narrow for HPLOC_SLOTS slots. Must be 0. */
    narrowSubgroups: 29,
    /** Nodes deeper than HPLOC_MAX_LEVEL, or subtrees taller than the info word holds. Must be 0. */
    tooDeep: 30,
} as const;
if (Math.min(...Object.values(HPLOC_STAT)) <= Math.max(...Object.values(LBVH_STAT)) || Math.max(...Object.values(HPLOC_STAT)) >= LBVH_STATS_WORDS) {
    throw new Error('HPLOC_STAT must sit after LBVH_STAT, inside a mesh\'s stats block');
}

/** `HplocParams.mortonCodes` values. */
export const HPLOC_MORTON = { 'cubic-30': 0, 'tinybvh-63': 1 } as const;

/** `HplocParams.tieBreak` values (hploc-cpu.ts, "Ties"). */
export const HPLOC_TIE_BREAK = { pairs: 0, tinybvh: 1 } as const;

/** Bytes of the uniform block (`HplocParams`). */
export const HPLOC_PARAMS_BYTES = 48;

/** Slots per lane for a device whose subgroups are at least `minSubgroupSize` wide. */
export function hplocSlotsFor(minSubgroupSize: number): number {
    return Math.max(1, Math.ceil(HPLOC_MAX_CLUSTERS / minSubgroupSize));
}

const hex = (value: number): string => `0x${value.toString(16)}u`;

/** The whole H-PLOC module: the LBVH kernels it builds on, then its own, for `slots` slots per lane. */
export function hplocWgsl(slots: number): string {
    if (!Number.isInteger(slots) || slots < 1 || slots > HPLOC_MAX_CLUSTERS) throw new RangeError(`H-PLOC needs 1 to ${HPLOC_MAX_CLUSTERS} slots per lane, got ${slots}`);
    const stat = HPLOC_STAT;
    return /* wgsl */ `enable subgroups;
${lbvhKernelsWgsl()}

// ==== H-PLOC (gpu-hploc-wgsl.ts) ======================================================

struct HplocParams {
    triangleTotal: u32,        // triangles of every mesh together
    innerTotal: u32,           // radix-tree nodes of every mesh together (lbvhParams.innerTotal)
    mortonCodes: u32,          // HPLOC_MORTON
    recordsOffset: u32,        // where the merges' records start in hplocScratch, in words
    listsOffset: u32,          // where the stored clusters start in hplocScratch
    childrenOffset: u32,       // where the merges' children start in hplocLinks
    parentsOffset: u32,        // where every cluster's parent starts in hplocLinks
    treePositionsOffset: u32,  // where each sorted triangle's tree position starts in hplocLinks
    tieBreak: u32,             // HPLOC_TIE_BREAK
    unused0: u32,
    unused1: u32,
    unused2: u32,
}

// Per sorted position: the split metric between it and the next, (high, low).
@group(0) @binding(22) var<storage, read_write> hplocDeltas: array<u32>;
// 'tinybvh-63' only: per triangle, its Morton code's high word, then its low word.
@group(0) @binding(23) var<storage, read_write> hplocCodes: array<u32>;
// Handed between climbing threads: [3 words per radix-tree node][a record per made node][a stored cluster per sorted position].
@group(0) @binding(24) var<storage, read_write> hplocScratch: array<atomic<u32>>;
// Plain views of hplocScratch, for the clear and for the dispatches after the climb.
@group(0) @binding(25) var<storage, read> hplocScratchView: array<u32>;
@group(0) @binding(26) var<storage, read_write> hplocScratchFill: array<u32>;
// Read only in later dispatches: [the node scan (lbvhRank), one more at the end][2 children per made node]
// [every cluster's parent: leaves, then made nodes][each sorted triangle's tree position].
@group(0) @binding(27) var<storage, read_write> hplocLinks: array<u32>;
@group(0) @binding(28) var<uniform> hplocParams: HplocParams;
// 'tinybvh-63' only: the triangles in order of the low code words (the first sort's result).
@group(0) @binding(29) var<storage, read> hplocValuesIn: array<u32>;

const HPLOC_SLOTS: u32 = ${slots}u;
const HPLOC_RADIUS: u32 = ${HPLOC_SEARCH_RADIUS}u;
const HPLOC_THRESHOLD: u32 = ${HPLOC_MERGE_THRESHOLD}u;
const HPLOC_MAX_CLUSTERS: u32 = ${HPLOC_MAX_CLUSTERS}u;
const HPLOC_CLIMB_WORDS: u32 = ${HPLOC_CLIMB_WORDS}u;
const HPLOC_RECORD_WORDS: u32 = ${HPLOC_RECORD_WORDS}u;
const HPLOC_LEVEL_BITS: u32 = ${LEVEL_BITS}u;
const HPLOC_LEVEL_MASK: u32 = ${hex((1 << LEVEL_BITS) - 1)};
const HPLOC_MAX_LEVEL: u32 = ${HPLOC_MAX_LEVEL}u;
const HPLOC_INFO_TRIANGLES_SHIFT: u32 = ${INFO_TRIANGLES_SHIFT}u;
const HPLOC_INFO_COLLAPSE_BIT: u32 = ${hex(INFO_COLLAPSE_BIT)};
const HPLOC_INFO_HEIGHT_MASK: u32 = ${hex(INFO_HEIGHT_MASK)};
// A side word: the child's cluster count above bit 8, its level below.
const HPLOC_SIDE_COUNT_SHIFT: u32 = 8u;
const HPLOC_SIDE_LEVEL_MASK: u32 = 0xffu;
// tinybvh's "far" area (BVH_FAR), where every nearest-neighbour search starts.
const HPLOC_FAR: f32 = 1e30f;
const HPLOC_TINYBVH_63: u32 = ${HPLOC_MORTON['tinybvh-63']}u;
const HPLOC_TIES_IN_PAIRS: u32 = ${HPLOC_TIE_BREAK.pairs}u;
const HPLOC_NO_LANE: u32 = 0xffffffffu;
// What a merge returns when it could not finish.
const HPLOC_FAILED: u32 = 0xffffffffu;
const HPLOC_STAT_ROOT: u32 = ${stat.root}u;
const HPLOC_STAT_NODES_MADE: u32 = ${stat.nodesMade}u;
const HPLOC_STAT_FAILED_MERGES: u32 = ${stat.failedMerges}u;
const HPLOC_STAT_NARROW_SUBGROUPS: u32 = ${stat.narrowSubgroups}u;
const HPLOC_STAT_TOO_DEEP: u32 = ${stat.tooDeep}u;

// ---- Where things are -------------------------------------------------------------------

fn hplocExchangeWord(mesh: LbvhMesh, split: u32) -> u32 {
    return (mesh.innerStart + split) * HPLOC_CLIMB_WORDS;
}

fn hplocSideWord(mesh: LbvhMesh, split: u32, side: u32) -> u32 {
    return (mesh.innerStart + split) * HPLOC_CLIMB_WORDS + 1u + side;
}

// Word 'word' of the record of made node 'made' (cluster triangleCount + made).
fn hplocRecordWord(mesh: LbvhMesh, made: u32, word: u32) -> u32 {
    return hplocParams.recordsOffset + (mesh.innerStart + made) * HPLOC_RECORD_WORDS + word;
}

fn hplocListWord(mesh: LbvhMesh, position: u32) -> u32 {
    return hplocParams.listsOffset + mesh.triangleStart + position;
}

fn hplocChildWord(mesh: LbvhMesh, made: u32, side: u32) -> u32 {
    return hplocParams.childrenOffset + (mesh.innerStart + made) * 2u + side;
}

fn hplocParentWord(mesh: LbvhMesh, cluster: u32) -> u32 {
    if (cluster < mesh.triangleCount) {
        return hplocParams.parentsOffset + mesh.triangleStart + cluster;
    }
    return hplocParams.parentsOffset + hplocParams.triangleTotal + mesh.innerStart + cluster - mesh.triangleCount;
}

fn hplocTreePositionWord(mesh: LbvhMesh, position: u32) -> u32 {
    return hplocParams.treePositionsOffset + mesh.triangleStart + position;
}

// ---- 3b. tinybvh's Morton codes ('tinybvh-63') -------------------------------------------

// 21 bits per axis over the mesh's box, each axis stretched to its own extent,
// bit b of axis a at bit 3b + a of a 63-bit code (as tinybvh's BuildHPLOC).
// The low word is sorted first (lbvhKeys), the high word after (hplocGatherHigh).
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn hplocMortonTinybvh(
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
        lower[axis] = lbvhFloatFromOrdered(lbvhStatsView[lbvhStatWord(mesh.index, LBVH_STAT_MESH_MIN + axis)]);
        upper[axis] = lbvhFloatFromOrdered(lbvhStatsView[lbvhStatWord(mesh.index, LBVH_STAT_MESH_MAX + axis)]);
    }
    let extent = upper - lower;
    let box = lbvhPrimitiveBox(mesh, triangle);
    let centroid = (box.lower + box.upper) * 0.5 - lower;
    let cellsPerAxis = 2097151.0;   // 0x1fffff
    let scaled = clamp(centroid / extent * cellsPerAxis, vec3f(0.0), vec3f(cellsPerAxis));
    let cell = select(vec3u(0u), vec3u(scaled), extent > vec3f(0.0));
    var low = 0u;
    var high = 0u;
    for (var bit = 0u; bit < 21u; bit += 1u) {
        let triple = ((cell.x >> bit) & 1u) | (((cell.y >> bit) & 1u) << 1u) | (((cell.z >> bit) & 1u) << 2u);
        let at = 3u * bit;
        if (at >= 32u) {
            high |= triple << (at - 32u);
        } else {
            low |= triple << at;            // bits past 31 fall off here...
            if (at + 2u >= 32u) {
                high |= triple >> (32u - at);   // ...and land in the high word
            }
        }
    }
    let at = mesh.triangleStart + triangle;
    lbvhKeys[at] = low;
    lbvhValues[at] = triangle;
    hplocCodes[at * 2u] = high;
    hplocCodes[at * 2u + 1u] = low;
}

// Between the two sorts: the high code words in the order the first sort left,
// so the second (stable) sort orders by high word, then low word.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn hplocGatherHigh(
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
    let at = mesh.triangleStart + position;
    let triangle = hplocValuesIn[at];
    lbvhKeys[at] = hplocCodes[(mesh.triangleStart + triangle) * 2u];
    lbvhValues[at] = triangle;
}

// ---- 3c. The split metric -------------------------------------------------------------

// tinybvh's: the xor of neighbouring codes as a 64-bit number (high, low), or
// the xor of their positions when the codes are equal. 30-bit codes have no
// low word, so any difference in code beats any difference in position.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn hplocDeltasKernel(
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
    if (position + 1u >= mesh.triangleCount) {
        return;
    }
    let at = mesh.triangleStart + position;
    let high = lbvhKeys[at] ^ lbvhKeys[at + 1u];
    var low = 0u;
    if (hplocParams.mortonCodes == HPLOC_TINYBVH_63) {
        let lowA = hplocCodes[(mesh.triangleStart + lbvhValues[at]) * 2u + 1u];
        let lowB = hplocCodes[(mesh.triangleStart + lbvhValues[at + 1u]) * 2u + 1u];
        low = lowA ^ lowB;
    }
    if (high == 0u && low == 0u) {
        low = position ^ (position + 1u);
    }
    hplocDeltas[at * 2u] = high;
    hplocDeltas[at * 2u + 1u] = low;
}

// ---- 4. Clear ---------------------------------------------------------------------------

// Marks every handed-over word unwritten (plain stores; the climb is a later dispatch).
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn hplocClear(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (!lbvhHasTile(workgroup, groups)) {
        return;
    }
    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let local = tile.first + localIndex;
    if (local < mesh.innerCount) {
        for (var word = 0u; word < HPLOC_CLIMB_WORDS; word += 1u) {
            hplocScratchFill[hplocExchangeWord(mesh, local) + word] = LBVH_UNWRITTEN;
        }
        for (var word = 0u; word < HPLOC_RECORD_WORDS; word += 1u) {
            hplocScratchFill[hplocRecordWord(mesh, local, word)] = LBVH_UNWRITTEN;
        }
    }
    if (local < mesh.triangleCount) {
        hplocScratchFill[hplocListWord(mesh, local)] = LBVH_UNWRITTEN;
    }
    if (local == 0u) {
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_ROOT)], LBVH_UNWRITTEN);
    }
}

// ---- 5. The climb, with the merges ------------------------------------------------------

// One cluster as the merge holds it.
struct HplocCluster {
    lower: vec3f,
    upper: vec3f,
    id: u32,     // a sorted position (a leaf) below the mesh's triangle count, a made node at or above it
    cost: f32,   // SAH cost of its subtree: the cheaper of one leaf and an inner node
    info: u32,   // triangles << 8 | collapse bit | height
}

alias HplocSlots = array<HplocCluster, HPLOC_SLOTS>;
alias HplocSlotWords = array<u32, HPLOC_SLOTS>;
alias HplocSlotFloats = array<f32, HPLOC_SLOTS>;
alias HplocSlotBoxes = array<LbvhBox, HPLOC_SLOTS>;

// A node being merged. Its range is [first, end); the left child is [first, split + 1), the right [split + 1, end).
struct HplocNode {
    first: u32,
    split: u32,
    end: u32,
    leftClusters: u32,
    rightClusters: u32,
    leftLevel: u32,
    rightLevel: u32,
    level: u32,   // the node's own: 1 + the higher child's
}

fn hplocDelta(mesh: LbvhMesh, position: u32) -> vec2u {
    let at = (mesh.triangleStart + position) * 2u;
    return vec2u(hplocDeltas[at], hplocDeltas[at + 1u]);
}

// Apetrei's choice of parent for the range [first, end) (as lbvhFindParent, on tinybvh's metric).
fn hplocFindParent(mesh: LbvhMesh, first: u32, end: u32) -> LbvhParent {
    if (first == 0u || (end != mesh.triangleCount && lbvhLess(hplocDelta(mesh, end - 1u), hplocDelta(mesh, first - 1u)))) {
        return LbvhParent(end - 1u, 0u);
    }
    return LbvhParent(first - 1u, 1u);
}

// One handed-over word, waiting (bounded) until the clear pass's mark is gone.
fn hplocWait(mesh: u32, at: u32) -> u32 {
    var value = atomicLoad(&hplocScratch[at]);
    for (var spins = 0u; value == LBVH_UNWRITTEN; spins += 1u) {
        if (spins == LBVH_MAX_SPINS) {
            atomicAdd(&lbvhStats[lbvhStatWord(mesh, LBVH_STAT_INCOMPLETE)], 1u);
            break;
        }
        value = atomicLoad(&hplocScratch[at]);
    }
    return value;
}

// A stored cluster entry, waiting (bounded) until the one the node at 'level' wrote is there.
fn hplocWaitEntry(mesh: u32, at: u32, level: u32) -> u32 {
    var value = atomicLoad(&hplocScratch[at]);
    for (var spins = 0u; (value & HPLOC_LEVEL_MASK) != level; spins += 1u) {
        if (spins == LBVH_MAX_SPINS) {
            atomicAdd(&lbvhStats[lbvhStatWord(mesh, LBVH_STAT_INCOMPLETE)], 1u);
            break;
        }
        value = atomicLoad(&hplocScratch[at]);
    }
    return value;
}

fn hplocLeafCluster(mesh: LbvhMesh, position: u32) -> HplocCluster {
    let box = lbvhPrimitiveBox(mesh, lbvhValues[mesh.triangleStart + position]);
    let cost = lbvhParams.intersectionCost * lbvhHalfArea(box.lower, box.upper);
    return HplocCluster(box.lower, box.upper, position, cost, 1u << HPLOC_INFO_TRIANGLES_SHIFT);
}

// A made node, from its record: its maker stored it before the node could be handed on.
fn hplocMadeCluster(mesh: LbvhMesh, cluster: u32) -> HplocCluster {
    let made = cluster - mesh.triangleCount;
    let m = mesh.index;
    var lower: vec3f;
    var upper: vec3f;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        lower[axis] = bitcast<f32>(hplocWait(m, hplocRecordWord(mesh, made, axis)));
        upper[axis] = bitcast<f32>(hplocWait(m, hplocRecordWord(mesh, made, 3u + axis)));
    }
    let cost = bitcast<f32>(hplocWait(m, hplocRecordWord(mesh, made, 6u)));
    return HplocCluster(lower, upper, cluster, cost, hplocWait(m, hplocRecordWord(mesh, made, 7u)));
}

// Cluster 'index' of a node being merged: the left child's clusters, then the right child's.
fn hplocLoadCluster(mesh: LbvhMesh, node: HplocNode, index: u32) -> HplocCluster {
    let fromLeft = index < node.leftClusters;
    let childFirst = select(node.split + 1u, node.first, fromLeft);
    let childEnd = select(node.end, node.split + 1u, fromLeft);
    let inChild = select(index - node.leftClusters, index, fromLeft);
    // A child of at most the threshold's triangles holds them as they are, in sorted order.
    var cluster = childFirst + inChild;
    if (childEnd - childFirst > HPLOC_THRESHOLD) {
        let level = select(node.rightLevel, node.leftLevel, fromLeft);
        cluster = hplocWaitEntry(mesh.index, hplocListWord(mesh, childFirst + inChild), level) >> HPLOC_LEVEL_BITS;
    }
    if (cluster < mesh.triangleCount) {
        return hplocLeafCluster(mesh, cluster);
    }
    return hplocMadeCluster(mesh, cluster);
}

fn hplocMergedArea(lowerA: vec3f, upperA: vec3f, lowerB: vec3f, upperB: vec3f) -> f32 {
    return lbvhHalfArea(min(lowerA, lowerB), max(upperA, upperB));
}

// The order among pairs at an exactly equal area ('pairs'): nearer first, then a
// pair starting on an even place, then the lower one (hploc-cpu.ts, "Ties":
// without it, equal areas chain the tree). A property of the pair, so both
// clusters agree. With 'tinybvh' the first candidate wins, as in tinybvh.
fn hplocPairRank(a: u32, b: u32) -> u32 {
    let low = min(a, b);
    return (max(a, b) - low) * 64u + (low & 1u) * 32u + low;
}

// Whether 'other', at merged area 'area', is a better neighbour for 'index' than 'nearest' at 'best'.
fn hplocCloser(index: u32, other: u32, area: f32, nearest: u32, best: f32) -> bool {
    return area < best || (hplocParams.tieBreak == HPLOC_TIES_IN_PAIRS && area == best && hplocPairRank(index, other) < hplocPairRank(index, nearest));
}

// Makes node 'made' from clusters a (the first in order) and b: its record (handed
// on, through atomics), children and parents (read in later dispatches only), and
// the SAH collapse decision, B1's rule. Both children are finished subtrees.
fn hplocMakeNode(mesh: LbvhMesh, made: u32, a: HplocCluster, b: HplocCluster, isRoot: bool) -> HplocCluster {
    let lower = min(a.lower, b.lower);
    let upper = max(a.upper, b.upper);
    let area = lbvhHalfArea(lower, upper);
    let triangles = (a.info >> HPLOC_INFO_TRIANGLES_SHIFT) + (b.info >> HPLOC_INFO_TRIANGLES_SHIFT);
    let innerCost = lbvhParams.traversalCost * area + a.cost + b.cost;
    var leafCost = LBVH_FLOAT_MAX;
    if (triangles <= lbvhParams.maxLeaf) {
        leafCost = lbvhParams.intersectionCost * f32(triangles) * area;
    }
    // The root stays an inner node: the layout's root always has two children.
    let collapse = !isRoot && leafCost <= innerCost;
    var height = 1u + max(a.info & HPLOC_INFO_HEIGHT_MASK, b.info & HPLOC_INFO_HEIGHT_MASK);
    if (height > HPLOC_INFO_HEIGHT_MASK) {
        atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_TOO_DEEP)], 1u);
        height = HPLOC_INFO_HEIGHT_MASK;
    }
    let info = (triangles << HPLOC_INFO_TRIANGLES_SHIFT) | select(height, HPLOC_INFO_COLLAPSE_BIT, collapse);
    let cost = min(innerCost, leafCost);
    let id = mesh.triangleCount + made;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        atomicStore(&hplocScratch[hplocRecordWord(mesh, made, axis)], bitcast<u32>(lower[axis]));
        atomicStore(&hplocScratch[hplocRecordWord(mesh, made, 3u + axis)], bitcast<u32>(upper[axis]));
    }
    atomicStore(&hplocScratch[hplocRecordWord(mesh, made, 6u)], bitcast<u32>(cost));
    atomicStore(&hplocScratch[hplocRecordWord(mesh, made, 7u)], info);
    hplocLinks[hplocChildWord(mesh, made, 0u)] = a.id;
    hplocLinks[hplocChildWord(mesh, made, 1u)] = b.id;
    hplocLinks[hplocParentWord(mesh, a.id)] = id;
    hplocLinks[hplocParentWord(mesh, b.id)] = id;
    return HplocCluster(lower, upper, id, cost, info);
}

// ---- The subgroup's merge: shuffles by cluster number ----

// For each of this lane's slots, the cluster at index 'wanted[slot]', wherever it is.
@diagnostic(off, subgroup_uniformity)
fn hplocGather(clusters: HplocSlots, wanted: HplocSlotWords, laneCount: u32) -> HplocSlots {
    var result: HplocSlots;
    for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
        let sourceLane = wanted[slot] % laneCount;
        let sourceSlot = wanted[slot] / laneCount;
        for (var offered = 0u; offered < HPLOC_SLOTS; offered += 1u) {
            let lower = subgroupShuffle(clusters[offered].lower, sourceLane);
            let upper = subgroupShuffle(clusters[offered].upper, sourceLane);
            let id = subgroupShuffle(clusters[offered].id, sourceLane);
            let cost = subgroupShuffle(clusters[offered].cost, sourceLane);
            let info = subgroupShuffle(clusters[offered].info, sourceLane);
            if (offered == sourceSlot) {
                result[slot] = HplocCluster(lower, upper, id, cost, info);
            }
        }
    }
    return result;
}

// The boxes only, for the nearest-neighbour search.
@diagnostic(off, subgroup_uniformity)
fn hplocGatherBoxes(clusters: HplocSlots, wanted: HplocSlotWords, laneCount: u32) -> HplocSlotBoxes {
    var result: HplocSlotBoxes;
    for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
        let sourceLane = wanted[slot] % laneCount;
        let sourceSlot = wanted[slot] / laneCount;
        for (var offered = 0u; offered < HPLOC_SLOTS; offered += 1u) {
            let lower = subgroupShuffle(clusters[offered].lower, sourceLane);
            let upper = subgroupShuffle(clusters[offered].upper, sourceLane);
            if (offered == sourceSlot) {
                result[slot] = LbvhBox(lower, upper);
            }
        }
    }
    return result;
}

@diagnostic(off, subgroup_uniformity)
fn hplocGatherWords(values: HplocSlotWords, wanted: HplocSlotWords, laneCount: u32) -> HplocSlotWords {
    var result: HplocSlotWords;
    for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
        let sourceLane = wanted[slot] % laneCount;
        let sourceSlot = wanted[slot] / laneCount;
        for (var offered = 0u; offered < HPLOC_SLOTS; offered += 1u) {
            let value = subgroupShuffle(values[offered], sourceLane);
            if (offered == sourceSlot) {
                result[slot] = value;
            }
        }
    }
    return result;
}

// Exclusive prefix sums of per-slot flags in cluster order: every lane's slot 0, then slot 1, ...
@diagnostic(off, subgroup_uniformity)
fn hplocPrefix(flags: HplocSlotWords) -> HplocSlotWords {
    var prefix: HplocSlotWords;
    var before = 0u;
    for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
        prefix[slot] = before + subgroupExclusiveAdd(flags[slot]);
        before += subgroupAdd(flags[slot]);
    }
    return prefix;
}

@diagnostic(off, subgroup_uniformity)
fn hplocTotal(flags: HplocSlotWords) -> u32 {
    var total = 0u;
    for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
        total += subgroupAdd(flags[slot]);
    }
    return total;
}

// The flags as one mask over cluster numbers (at most 32 clusters: bit c is cluster c).
@diagnostic(off, subgroup_uniformity)
fn hplocClusterMask(flags: HplocSlotWords, laneCount: u32) -> u32 {
    var mask = 0u;
    for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
        // Clusters of slot 'slot' start at slot * laneCount; their lanes are below 32, so they are in the ballot's first word.
        let lanes = subgroupBallot(flags[slot] != 0u).x;
        if (slot * laneCount < 32u) {
            mask |= lanes << (slot * laneCount);
        }
    }
    return mask;
}

// The position of the set bit with 'rank' set bits below it.
fn hplocNthSetBit(mask: u32, rank: u32) -> u32 {
    var remaining = rank;
    var bits = mask;
    var position = 0u;
    for (var width = 16u; width > 0u; width /= 2u) {
        let below = countOneBits(bits & ((1u << width) - 1u));
        if (remaining >= below) {
            remaining -= below;
            bits = bits >> width;
            position += width;
        }
    }
    return position;
}

// The whole subgroup merges one node (tinybvh's BuildHPLOC, the loop over 'n'):
// loads its children's clusters, merges mutual nearest neighbours until it holds
// at most HPLOC_THRESHOLD clusters (the root: one), and stores what is left for
// its parent. Returns the clusters left, or HPLOC_FAILED.
@diagnostic(off, subgroup_uniformity)
fn hplocMerge(mesh: LbvhMesh, node: HplocNode, lane: u32, laneCount: u32) -> u32 {
    let isRoot = node.first == 0u && node.end == mesh.triangleCount;
    var count = node.leftClusters + node.rightClusters;
    var clusters: HplocSlots;
    for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
        let index = slot * laneCount + lane;
        if (index < count) {
            clusters[slot] = hplocLoadCluster(mesh, node, index);
        }
    }
    let threshold = select(HPLOC_THRESHOLD, 1u, isRoot);
    // Every round merges at least one pair, so HPLOC_MAX_CLUSTERS rounds always suffice.
    for (var mergeRound = 0u; count > threshold && mergeRound < HPLOC_MAX_CLUSTERS; mergeRound += 1u) {
        // 1. Each cluster's nearest neighbour within the radius, by merged area, exact ties
        //    as hplocCloser says. Candidates in tinybvh's order (+1, -1, +2, -2, ...). The
        //    pair (j, j + r) is always measured in that order, so both clusters see the same area.
        var nearest: HplocSlotWords;
        var best: HplocSlotFloats;
        for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
            let index = slot * laneCount + lane;
            nearest[slot] = select(index - 1u, index + 1u, index + 1u < count);
            best[slot] = HPLOC_FAR;
        }
        for (var radius = 1u; radius <= HPLOC_RADIUS && radius < count; radius += 1u) {
            var upIndex: HplocSlotWords;
            var downIndex: HplocSlotWords;
            for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
                let index = slot * laneCount + lane;
                upIndex[slot] = index + radius;
                downIndex[slot] = index - radius;   // wraps below 0, where it is not used
            }
            let above = hplocGatherBoxes(clusters, upIndex, laneCount);
            let below = hplocGatherBoxes(clusters, downIndex, laneCount);
            for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
                let index = slot * laneCount + lane;
                if (index >= count) {
                    continue;
                }
                let own = clusters[slot];
                if (index + radius < count) {
                    let area = hplocMergedArea(own.lower, own.upper, above[slot].lower, above[slot].upper);
                    if (hplocCloser(index, index + radius, area, nearest[slot], best[slot])) {
                        best[slot] = area;
                        nearest[slot] = index + radius;
                    }
                }
                if (index >= radius) {
                    let area = hplocMergedArea(below[slot].lower, below[slot].upper, own.lower, own.upper);
                    if (hplocCloser(index, index - radius, area, nearest[slot], best[slot])) {
                        best[slot] = area;
                        nearest[slot] = index - radius;
                    }
                }
            }
        }

        // 2. Mutual nearest neighbours merge; the new node takes the first one's place.
        let nearestOfNearest = hplocGatherWords(nearest, nearest, laneCount);
        let partner = hplocGather(clusters, nearest, laneCount);
        var keep: HplocSlotWords;
        var makes: HplocSlotWords;
        for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
            let index = slot * laneCount + lane;
            let mutual = index < count && nearestOfNearest[slot] == index;
            keep[slot] = select(0u, 1u, index < count && !(mutual && index > nearest[slot]));
            makes[slot] = select(0u, 1u, mutual && index < nearest[slot]);
        }
        let kept = hplocTotal(keep);
        let made = hplocTotal(makes);
        if (made == 0u) {
            if (lane == 0u) {
                atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_FAILED_MERGES)], 1u);
            }
            return HPLOC_FAILED;
        }
        var base = 0u;
        if (subgroupElect()) {
            base = atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_NODES_MADE)], made);
        }
        // The elected lane is the lowest active one, which is where broadcastFirst reads.
        base = subgroupBroadcastFirst(base);
        let makeRank = hplocPrefix(makes);
        for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
            if (makes[slot] == 1u) {
                let madeIndex = base + makeRank[slot];
                if (madeIndex < mesh.innerCount) {
                    // The last pair of the root's merge makes the root.
                    clusters[slot] = hplocMakeNode(mesh, madeIndex, clusters[slot], partner[slot], isRoot && kept == 1u);
                } else {
                    atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_INCOMPLETE)], 1u);
                }
            }
        }

        // 3. Close the gaps: kept cluster number k moves to place k, in order.
        let keptMask = hplocClusterMask(keep, laneCount);
        var sources: HplocSlotWords;
        for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
            let index = slot * laneCount + lane;
            sources[slot] = hplocNthSetBit(keptMask, min(index, kept - 1u));
        }
        clusters = hplocGather(clusters, sources, laneCount);
        count = kept;
    }
    if (count > threshold) {
        if (lane == 0u) {
            atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_FAILED_MERGES)], 1u);
        }
        return HPLOC_FAILED;
    }

    // 4. What is left, at the start of the node's range, tagged with its level.
    for (var slot = 0u; slot < HPLOC_SLOTS; slot += 1u) {
        let index = slot * laneCount + lane;
        if (index < count) {
            atomicStore(&hplocScratch[hplocListWord(mesh, node.first + index)], (clusters[slot].id << HPLOC_LEVEL_BITS) | node.level);
        }
    }
    if (isRoot && lane == 0u) {
        // Cluster 0 is lane 0's slot 0.
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_ROOT)], clusters[0].id);
        atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_ROOTS_FINISHED)], 1u);
    }
    return count;
}

// The lowest lane set in a subgroupBallot mask, or HPLOC_NO_LANE.
fn hplocLowestLane(lanes: vec4u) -> u32 {
    for (var word = 0u; word < 4u; word += 1u) {
        if (lanes[word] != 0u) {
            return word * 32u + firstTrailingBit(lanes[word]);
        }
    }
    return HPLOC_NO_LANE;
}

fn hplocWithoutLane(lanes: vec4u, lane: u32) -> vec4u {
    var result = lanes;
    result[lane / 32u] &= ~(1u << (lane % 32u));
    return result;
}

@diagnostic(off, subgroup_uniformity)
fn hplocShuffleNode(node: HplocNode, sourceLane: u32) -> HplocNode {
    return HplocNode(
        subgroupShuffle(node.first, sourceLane),
        subgroupShuffle(node.split, sourceLane),
        subgroupShuffle(node.end, sourceLane),
        subgroupShuffle(node.leftClusters, sourceLane),
        subgroupShuffle(node.rightClusters, sourceLane),
        subgroupShuffle(node.leftLevel, sourceLane),
        subgroupShuffle(node.rightLevel, sourceLane),
        subgroupShuffle(node.level, sourceLane),
    );
}

// One thread per sorted position. Every step, each climbing thread goes up one
// level; then the subgroup merges, one after another, every node that needs it.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
@diagnostic(off, subgroup_uniformity)
fn hplocClimb(
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
    if (laneCount * HPLOC_SLOTS < HPLOC_MAX_CLUSTERS) {
        // The device reported a smaller subgroup than this one has; the tree is refused.
        if (subgroupElect()) {
            atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_NARROW_SUBGROUPS)], 1u);
        }
        return;
    }
    let position = tile.first + localIndex;
    var climbing = position < count && count > 1u;
    // The node this thread carries: its range, how many clusters it holds, and its level.
    var first = position;
    var end = position + 1u;
    var clusters = 1u;
    var level = 0u;
    // Lanes without a triangle, and lanes that stopped, stay in the loop to help merge.
    for (var step = 0u; step <= HPLOC_MAX_LEVEL; step += 1u) {
        var node = HplocNode(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        var merges = false;
        if (climbing) {
            let parent = hplocFindParent(mesh, first, end);
            // This side's count and level first; then leave this side's far end for the sibling's thread.
            atomicStore(&hplocScratch[hplocSideWord(mesh, parent.split, parent.side)], (clusters << HPLOC_SIDE_COUNT_SHIFT) | level);
            let other = atomicExchange(&hplocScratch[hplocExchangeWord(mesh, parent.split)], select(first, end, parent.side == 1u));
            if (other == LBVH_UNWRITTEN) {
                climbing = false;   // the first to arrive: the sibling's thread carries on
            } else {
                let sibling = hplocWait(mesh.index, hplocSideWord(mesh, parent.split, 1u - parent.side));
                let siblingClusters = sibling >> HPLOC_SIDE_COUNT_SHIFT;
                let siblingLevel = sibling & HPLOC_SIDE_LEVEL_MASK;
                if (parent.side == 1u) {
                    first = other;
                    node.leftClusters = siblingClusters;
                    node.leftLevel = siblingLevel;
                    node.rightClusters = clusters;
                    node.rightLevel = level;
                } else {
                    end = other;
                    node.leftClusters = clusters;
                    node.leftLevel = level;
                    node.rightClusters = siblingClusters;
                    node.rightLevel = siblingLevel;
                }
                node.first = first;
                node.split = parent.split;
                node.end = end;
                level = 1u + max(node.leftLevel, node.rightLevel);
                node.level = level;
                if (level > HPLOC_MAX_LEVEL) {
                    atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_TOO_DEEP)], 1u);
                    climbing = false;
                } else if (end - first > HPLOC_THRESHOLD || (first == 0u && end == count)) {
                    merges = true;
                } else {
                    clusters = end - first;   // a small node holds its triangles as they are
                }
            }
        }
        var waiting = subgroupBallot(merges);
        loop {
            let eventLane = hplocLowestLane(waiting);
            if (eventLane == HPLOC_NO_LANE) {
                break;
            }
            waiting = hplocWithoutLane(waiting, eventLane);
            let merged = hplocShuffleNode(node, eventLane);
            let left = hplocMerge(mesh, merged, lane, laneCount);
            if (lane == eventLane) {
                clusters = left;
                // The root is done; a failed merge stops the climb (the tree is refused).
                if (left == HPLOC_FAILED || (merged.first == 0u && merged.end == count)) {
                    climbing = false;
                }
            }
        }
        if (!subgroupAny(climbing)) {
            break;
        }
    }
}

// ---- 6. Which made nodes stay -----------------------------------------------------------

fn hplocRootOf(mesh: LbvhMesh) -> u32 {
    return lbvhStatsView[lbvhStatWord(mesh.index, HPLOC_STAT_ROOT)];
}

// A cluster's info word in the dispatches after the climb.
fn hplocInfo(mesh: LbvhMesh, cluster: u32) -> u32 {
    if (cluster < mesh.triangleCount) {
        return 1u << HPLOC_INFO_TRIANGLES_SHIFT;
    }
    return hplocScratchView[hplocRecordWord(mesh, cluster - mesh.triangleCount, 7u)];
}

fn hplocTriangles(info: u32) -> u32 {
    return info >> HPLOC_INFO_TRIANGLES_SHIFT;
}

fn hplocCollapsed(info: u32) -> bool {
    return (info & HPLOC_INFO_COLLAPSE_BIT) != 0u;
}

// A made node stays unless it or an ancestor became a leaf. Only ancestors with
// at most maxLeaf triangles can have, and counts grow on the way up.
fn hplocKept(mesh: LbvhMesh, cluster: u32, root: u32) -> bool {
    if (hplocCollapsed(hplocInfo(mesh, cluster))) {
        return false;
    }
    var current = cluster;
    for (var guard = 0u; guard <= lbvhParams.maxLeaf && current != root; guard += 1u) {
        let above = hplocLinks[hplocParentWord(mesh, current)];
        let info = hplocInfo(mesh, above);
        if (hplocTriangles(info) > lbvhParams.maxLeaf) {
            return true;
        }
        if (hplocCollapsed(info)) {
            return false;
        }
        current = above;
    }
    return true;
}

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn hplocMark(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (!lbvhHasTile(workgroup, groups)) {
        return;
    }
    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let made = tile.first + localIndex;
    if (made >= mesh.innerCount) {
        return;
    }
    // A one-triangle mesh made no node but still needs one, its root.
    var kept = mesh.triangleCount == 1u;
    if (!kept) {
        kept = hplocKept(mesh, mesh.triangleCount + made, hplocRootOf(mesh));
    }
    hplocLinks[mesh.innerStart + made] = select(0u, 1u, kept);
}

// ---- 7. The tracer's layout -----------------------------------------------------------

fn hplocBoxOf(mesh: LbvhMesh, cluster: u32) -> LbvhBox {
    if (cluster < mesh.triangleCount) {
        return lbvhPrimitiveBox(mesh, lbvhValues[mesh.triangleStart + cluster]);
    }
    let made = cluster - mesh.triangleCount;
    var lower: vec3f;
    var upper: vec3f;
    for (var axis = 0u; axis < 3u; axis += 1u) {
        lower[axis] = bitcast<f32>(hplocScratchView[hplocRecordWord(mesh, made, axis)]);
        upper[axis] = bitcast<f32>(hplocScratchView[hplocRecordWord(mesh, made, 3u + axis)]);
    }
    return LbvhBox(lower, upper);
}

// Where a cluster's first triangle goes in depth-first leaf order: the
// triangles of every left sibling on the way to the root come before it.
fn hplocTreeOffset(mesh: LbvhMesh, cluster: u32, root: u32) -> u32 {
    var offset = 0u;
    var current = cluster;
    // Every node on the way is a kept node, whose height fits HPLOC_INFO_HEIGHT_MASK.
    for (var guard = 0u; guard <= HPLOC_INFO_HEIGHT_MASK && current != root; guard += 1u) {
        let above = hplocLinks[hplocParentWord(mesh, current)];
        let left = hplocLinks[hplocChildWord(mesh, above - mesh.triangleCount, 0u)];
        if (left != current) {
            offset += hplocTriangles(hplocInfo(mesh, left));
        }
        current = above;
    }
    return offset;
}

// A leaf of the layout (one triangle, or a collapsed subtree of at most maxLeaf):
// its triangles' tree positions from 'start', depth first.
fn hplocPlaceLeaves(mesh: LbvhMesh, cluster: u32, start: u32) {
    // A subtree of at most 8 triangles needs at most 8 entries.
    var stack: array<u32, ${2 * HPLOC_MERGE_THRESHOLD}>;
    stack[0] = cluster;
    var depth = 1u;
    var placed = start;
    while (depth > 0u) {
        depth -= 1u;
        let current = stack[depth];
        if (current < mesh.triangleCount) {
            hplocLinks[hplocTreePositionWord(mesh, current)] = placed;
            placed += 1u;
        } else if (depth + 2u <= ${2 * HPLOC_MERGE_THRESHOLD}u) {
            let made = current - mesh.triangleCount;
            stack[depth] = hplocLinks[hplocChildWord(mesh, made, 1u)];
            stack[depth + 1u] = hplocLinks[hplocChildWord(mesh, made, 0u)];
            depth += 2u;
        }
    }
}

// The output number of a kept made node: its scanned rank within the mesh's run, the root swapped to 0.
fn hplocOutputIndex(mesh: LbvhMesh, cluster: u32, nodeOffset: u32, rootRank: u32) -> u32 {
    return lbvhOutputIndex(hplocLinks[mesh.innerStart + cluster - mesh.triangleCount] - nodeOffset, rootRank);
}

@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn hplocEmit(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (!lbvhHasTile(workgroup, groups)) {
        return;
    }
    let tile = lbvhTile(workgroup, groups);
    let mesh = tile.mesh;
    let made = tile.first + localIndex;
    if (made >= mesh.innerCount) {
        return;
    }
    // The mesh's run of node numbers starts at the scanned value of its first made node.
    let nodeOffset = hplocLinks[mesh.innerStart];
    if (mesh.triangleCount == 1u) {
        // One triangle: the root holds it on the left and nothing on the right.
        let box = lbvhPrimitiveBox(mesh, 0u);
        let out = nodeOffset * LBVH_NODE_WORDS;
        lbvhWriteChild(out, 0u, box.lower, box.upper, 0u, 1u);
        lbvhWriteChild(out, 1u, vec3f(0.0), vec3f(0.0), LBVH_EMPTY_CHILD, 0u);
        lbvhOutParents[nodeOffset] = LBVH_NO_PARENT;
        hplocLinks[hplocTreePositionWord(mesh, 0u)] = 0u;
        return;
    }
    let cluster = mesh.triangleCount + made;
    let root = hplocRootOf(mesh);
    if (!hplocKept(mesh, cluster, root)) {
        return;
    }
    let rootRank = hplocLinks[mesh.innerStart + root - mesh.triangleCount] - nodeOffset;
    let out = hplocOutputIndex(mesh, cluster, nodeOffset, rootRank);
    let offset = hplocTreeOffset(mesh, cluster, root);
    let left = hplocLinks[hplocChildWord(mesh, made, 0u)];
    for (var side = 0u; side < 2u; side += 1u) {
        let child = hplocLinks[hplocChildWord(mesh, made, side)];
        let info = hplocInfo(mesh, child);
        let box = hplocBoxOf(mesh, child);
        var index: u32;
        var count = 0u;
        if (child < mesh.triangleCount || hplocCollapsed(info)) {
            index = offset;
            if (side == 1u) {
                index += hplocTriangles(hplocInfo(mesh, left));
            }
            count = hplocTriangles(info);
            hplocPlaceLeaves(mesh, child, index);
        } else {
            index = hplocOutputIndex(mesh, child, nodeOffset, rootRank);
        }
        lbvhWriteChild((nodeOffset + out) * LBVH_NODE_WORDS, side, box.lower, box.upper, index, count);
    }
    var parent = LBVH_NO_PARENT;
    if (cluster != root) {
        parent = hplocOutputIndex(mesh, hplocLinks[hplocParentWord(mesh, cluster)], nodeOffset, rootRank);
    }
    lbvhOutParents[nodeOffset + out] = parent;
}

// Each triangle to its tree position (written by hplocEmit, which runs first).
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn hplocTrianglesKernel(
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
    lbvhWriteTriangle(mesh, lbvhValues[mesh.triangleStart + position], hplocLinks[hplocTreePositionWord(mesh, position)]);
}

// One thread per mesh: where its nodes landed, how many, its depth and box, and the checks.
@compute @workgroup_size(${LBVH_WORKGROUP_SIZE})
fn hplocFinish(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= lbvhParams.meshCount) {
        return;
    }
    let mesh = lbvhMesh(id.x);
    let nodeOffset = hplocLinks[mesh.innerStart];
    // The scanned value after the mesh's last made node: the next mesh's offset, or the end marker's.
    let nodeCount = hplocLinks[mesh.innerStart + mesh.innerCount] - nodeOffset;
    atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_NODE_OFFSET)], nodeOffset);
    atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_NODE_COUNT)], nodeCount);
    var lower: vec3f;
    var upper: vec3f;
    if (mesh.triangleCount == 1u) {
        let box = lbvhPrimitiveBox(mesh, 0u);
        lower = box.lower;
        upper = box.upper;
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_LEAF_COUNT)], 1u);
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_MAX_DEPTH)], 1u);
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_ROOTS_FINISHED)], 1u);
    } else {
        let root = atomicLoad(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_ROOT)]);
        if (root == LBVH_UNWRITTEN || root < mesh.triangleCount) {
            return;   // no root: rootsFinished stays 0 and the tree is refused
        }
        let box = hplocBoxOf(mesh, root);
        lower = box.lower;
        upper = box.upper;
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_LEAF_COUNT)], nodeCount + 1u);
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_MAX_DEPTH)], hplocInfo(mesh, root) & HPLOC_INFO_HEIGHT_MASK);
        if (atomicLoad(&lbvhStats[lbvhStatWord(mesh.index, HPLOC_STAT_NODES_MADE)]) != mesh.triangleCount - 1u) {
            atomicAdd(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_INCOMPLETE)], 1u);
        }
    }
    for (var axis = 0u; axis < 3u; axis += 1u) {
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_BOUNDS + axis)], bitcast<u32>(lower[axis]));
        atomicStore(&lbvhStats[lbvhStatWord(mesh.index, LBVH_STAT_BOUNDS + 3u + axis)], bitcast<u32>(upper[axis]));
    }
}
`;
}
