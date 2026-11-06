/**
 * Checks a tree in the tracer's GPU layout (gpu-layout.ts) against the mesh
 * it was built from, whoever built it. It is what every builder behind the
 * Builder contract must satisfy before the ray oracle is worth running:
 *
 * - the triangles are the mesh's, each exactly once, corners bit for bit,
 *   with the right original index and flags;
 * - the nodes form one tree from node 0: every node reached once, every
 *   triangle position in exactly one leaf, leaves of 1 to `maxLeafPrimitives`;
 * - every child box is exactly the box of the triangles below it (boxes are
 *   min and max of f32 values, so "exactly" is bit equality);
 * - the parents, the depth, the leaf count and the mesh box agree with the nodes.
 *
 * Plain arrays and an explicit stack, so it handles meshes of millions of
 * triangles. Returns the problems found (at most `maxProblems`), empty if none.
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
import type { TinyBvhInput } from './tinybvh.ts';

const TRIANGLE_WORDS = GPU_LAYOUT_TRIANGLE_BYTES / 4;
const CHILD_INDEX = [GPU_LAYOUT_NODE_WORD.leftIndex, GPU_LAYOUT_NODE_WORD.rightIndex] as const;
const CHILD_COUNT = [GPU_LAYOUT_NODE_WORD.leftCount, GPU_LAYOUT_NODE_WORD.rightCount] as const;
const CHILD_MIN = [GPU_LAYOUT_NODE_WORD.leftMin, GPU_LAYOUT_NODE_WORD.rightMin] as const;
const MAX_OFFSET = GPU_LAYOUT_NODE_WORD.leftMax - GPU_LAYOUT_NODE_WORD.leftMin;

export interface LayoutCheckOptions {
    readonly maxLeafPrimitives?: number;
    readonly maxProblems?: number;
}

export function checkGpuLayout(layout: GpuLayoutMeshBvh, geometry: TinyBvhInput, options: LayoutCheckOptions = {}): string[] {
    const maxLeaf = options.maxLeafPrimitives ?? GPU_LAYOUT_MAX_LEAF_PRIMITIVES;
    const maxProblems = options.maxProblems ?? 20;
    const problems: string[] = [];
    const report = (problem: string): boolean => {
        if (problems.length < maxProblems) problems.push(problem);
        return problems.length >= maxProblems;
    };

    const { positions, indices, triangleFlags } = geometry;
    const triangleCount = indices ? indices.length / 3 : positions.length / 3 / 3;
    const { nodeCount } = layout;
    if (layout.triangleCount !== triangleCount) report(`the layout has ${layout.triangleCount} triangles, the mesh ${triangleCount}`);
    if (nodeCount < 1 || layout.nodes.byteLength < nodeCount * GPU_LAYOUT_NODE_BYTES) report(`${nodeCount} nodes in ${layout.nodes.byteLength} bytes`);
    if (layout.triangles.byteLength !== triangleCount * GPU_LAYOUT_TRIANGLE_BYTES) report(`triangles take ${layout.triangles.byteLength} bytes`);
    if (layout.parents.length < nodeCount) report(`${layout.parents.length} parents for ${nodeCount} nodes`);
    if (problems.length > 0) return problems;

    // ---- Triangles: the mesh's, each once, corners as they are.
    const triangleWords = new Uint32Array(layout.triangles);
    const triangleFloats = new Float32Array(layout.triangles);
    const positionBits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
    const seen = new Uint8Array(triangleCount);
    const corners = [GPU_LAYOUT_TRIANGLE_WORD.vertex0, GPU_LAYOUT_TRIANGLE_WORD.vertex1, GPU_LAYOUT_TRIANGLE_WORD.vertex2];
    for (let slot = 0; slot < triangleCount; slot++) {
        const base = slot * TRIANGLE_WORDS;
        const original = triangleWords[base + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
        if (original >= triangleCount || seen[original]) {
            if (report(`triangle slot ${slot} holds triangle ${original}, which is out of range or already placed`)) return problems;
            continue;
        }
        seen[original] = 1;
        for (let corner = 0; corner < 3; corner++) {
            const vertex = indices ? indices[original * 3 + corner] : original * 3 + corner;
            for (let axis = 0; axis < 3; axis++) {
                if (triangleWords[base + corners[corner] + axis] !== positionBits[vertex * 3 + axis]) {
                    if (report(`triangle slot ${slot} (triangle ${original}) corner ${corner} differs from the mesh`)) return problems;
                }
            }
        }
        const flags = triangleFlags ? triangleFlags[original] : 0;
        if (triangleWords[base + GPU_LAYOUT_TRIANGLE_WORD.flags] !== flags) report(`triangle slot ${slot} has flags ${triangleWords[base + GPU_LAYOUT_TRIANGLE_WORD.flags]}, not ${flags}`);
    }
    if (problems.length > 0) return problems;

    // ---- Topology: a preorder walk from the root, every node once, every position in one leaf.
    const nodeWords = new Uint32Array(layout.nodes, 0, nodeCount * GPU_LAYOUT_NODE_WORDS);
    const nodeFloats = new Float32Array(layout.nodes, 0, nodeCount * GPU_LAYOUT_NODE_WORDS);
    const order = new Uint32Array(nodeCount);
    const depth = new Uint32Array(nodeCount);
    const reached = new Uint8Array(nodeCount);
    const covered = new Uint8Array(triangleCount);
    let ordered = 0;
    let leaves = 0;
    let deepest = 0;
    const stack: number[] = [0];
    reached[0] = 1;
    depth[0] = 1;
    if (layout.parents[0] !== GPU_LAYOUT_NO_PARENT) report(`the root's parent is ${layout.parents[0]}`);
    while (stack.length > 0) {
        const node = stack.pop() as number;
        order[ordered++] = node;
        deepest = Math.max(deepest, depth[node]);
        for (let side = 0; side < 2; side++) {
            const index = nodeWords[node * GPU_LAYOUT_NODE_WORDS + CHILD_INDEX[side]];
            const count = nodeWords[node * GPU_LAYOUT_NODE_WORDS + CHILD_COUNT[side]];
            if (index === GPU_LAYOUT_EMPTY_CHILD) {
                if (side === 0 || node !== 0 || triangleCount !== 1) report(`node ${node} side ${side} is empty`);
                continue;
            }
            if (count > 0) {
                leaves++;
                if (count > maxLeaf) report(`node ${node} side ${side} is a leaf of ${count} triangles (most ${maxLeaf})`);
                if (index + count > triangleCount) {
                    report(`node ${node} side ${side} covers positions past the last triangle`);
                    continue;
                }
                for (let position = index; position < index + count; position++) {
                    if (covered[position]) report(`position ${position} is in more than one leaf`);
                    covered[position] = 1;
                }
            } else {
                if (index >= nodeCount || reached[index]) {
                    if (report(`node ${node} side ${side} points at node ${index}, which is out of range or already reached`)) return problems;
                    continue;
                }
                reached[index] = 1;
                depth[index] = depth[node] + 1;
                if (layout.parents[index] !== node) report(`node ${index}'s parent is ${layout.parents[index]}, not ${node}`);
                stack.push(index);
            }
        }
    }
    if (ordered !== nodeCount) report(`${nodeCount - ordered} nodes are not reached from the root`);
    for (let position = 0; position < triangleCount; position++) {
        if (!covered[position]) {
            report(`position ${position} is in no leaf`);
            break;
        }
    }
    if (leaves !== layout.leafCount) report(`${leaves} leaves, the layout says ${layout.leafCount}`);
    if (deepest !== layout.maxDepth) report(`the deepest node is at level ${deepest}, the layout says ${layout.maxDepth}`);
    if (problems.length > 0) return problems;

    // ---- Boxes: children before parents (reverse preorder), each child box the exact box below it.
    const boxes = new Float32Array(nodeCount * 6);
    const childBox = new Float32Array(6);
    for (let at = nodeCount - 1; at >= 0; at--) {
        const node = order[at];
        const nodeBox = boxes.subarray(node * 6, node * 6 + 6);
        nodeBox.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
        for (let side = 0; side < 2; side++) {
            const index = nodeWords[node * GPU_LAYOUT_NODE_WORDS + CHILD_INDEX[side]];
            const count = nodeWords[node * GPU_LAYOUT_NODE_WORDS + CHILD_COUNT[side]];
            if (index === GPU_LAYOUT_EMPTY_CHILD) continue;
            if (count > 0) {
                childBox.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
                for (let slot = index; slot < index + count; slot++) {
                    for (const corner of corners) {
                        for (let axis = 0; axis < 3; axis++) {
                            const value = triangleFloats[slot * TRIANGLE_WORDS + corner + axis];
                            childBox[axis] = Math.min(childBox[axis], value);
                            childBox[3 + axis] = Math.max(childBox[3 + axis], value);
                        }
                    }
                }
            } else {
                childBox.set(boxes.subarray(index * 6, index * 6 + 6));
            }
            const stored = node * GPU_LAYOUT_NODE_WORDS + CHILD_MIN[side];
            for (let axis = 0; axis < 3; axis++) {
                if (nodeFloats[stored + axis] !== childBox[axis] || nodeFloats[stored + MAX_OFFSET + axis] !== childBox[3 + axis]) {
                    if (report(`node ${node} side ${side}'s box is not the box of what is below it (axis ${axis})`)) return problems;
                    break;
                }
            }
            for (let axis = 0; axis < 3; axis++) {
                nodeBox[axis] = Math.min(nodeBox[axis], childBox[axis]);
                nodeBox[3 + axis] = Math.max(nodeBox[3 + axis], childBox[3 + axis]);
            }
        }
    }
    for (let axis = 0; axis < 6; axis++) {
        if (layout.bounds[axis] !== boxes[axis]) {
            report(`the mesh box differs from the root's box at ${axis}: ${layout.bounds[axis]} vs ${boxes[axis]}`);
            break;
        }
    }
    return problems;
}

/**
 * The SAH cost of a tree in the layout, relative to the root's box: the
 * expected cost of a random ray through the root's box, counting
 * `traversalCost` per inner node visited and `intersectionCost` per triangle
 * tested. What GPU builders are judged by against tinybvh (bvh-framework.md, B2).
 */
export function treeSahCost(layout: GpuLayoutMeshBvh, traversalCost = 1, intersectionCost = 1): number {
    const words = new Uint32Array(layout.nodes, 0, layout.nodeCount * GPU_LAYOUT_NODE_WORDS);
    const floats = new Float32Array(layout.nodes, 0, layout.nodeCount * GPU_LAYOUT_NODE_WORDS);
    const area = (base: number): number => {
        const x = floats[base + MAX_OFFSET] - floats[base];
        const y = floats[base + MAX_OFFSET + 1] - floats[base + 1];
        const z = floats[base + MAX_OFFSET + 2] - floats[base + 2];
        return x * y + y * z + z * x;
    };
    let total = 0;
    let rootArea = 0;
    for (let node = 0; node < layout.nodeCount; node++) {
        const base = node * GPU_LAYOUT_NODE_WORDS;
        let lower = [Infinity, Infinity, Infinity];
        let upper = [-Infinity, -Infinity, -Infinity];
        for (let side = 0; side < 2; side++) {
            if (words[base + CHILD_INDEX[side]] === GPU_LAYOUT_EMPTY_CHILD) continue;
            const box = base + CHILD_MIN[side];
            lower = lower.map((value, axis) => Math.min(value, floats[box + axis]));
            upper = upper.map((value, axis) => Math.max(value, floats[box + MAX_OFFSET + axis]));
            const count = words[base + CHILD_COUNT[side]];
            if (count > 0) total += intersectionCost * count * area(box);
        }
        const size = [upper[0] - lower[0], upper[1] - lower[1], upper[2] - lower[2]];
        const nodeArea = size[0] * size[1] + size[1] * size[2] + size[2] * size[0];
        total += traversalCost * nodeArea;
        if (node === 0) rootArea = nodeArea;
    }
    return rootArea > 0 ? total / rootArea : 0;
}
