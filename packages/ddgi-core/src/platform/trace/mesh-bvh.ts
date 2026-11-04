/**
 * Per-mesh BVH (the "bottom level"): turns triangle geometry into the node
 * array and the tree-ordered triangle array the GPU traverses.
 *
 * Runs anywhere (main thread, worker, Node). `mesh-bvh-async.ts` wraps it in a
 * worker for large meshes.
 */

import {
    DEFAULT_TRIANGLE_TEST,
    MAX_REFERENCE_INDEX,
    TRIANGLE_BYTES,
    TRIANGLE_WORD,
    TRIANGLE_WORDS,
    type TriangleTest,
} from './layout.ts';
import { buildSahTree, DEFAULT_SAH_OPTIONS, type SahBuildOptions } from './sah-builder.ts';

/** Triangle geometry in object space. */
export interface MeshGeometry {
    /** `x, y, z` per vertex, tightly packed. */
    readonly positions: Float32Array;
    /** Three vertex indices per triangle; omit for an unindexed triangle list. */
    readonly indices?: Uint16Array | Uint32Array | null;
    /**
     * Optional per-triangle bit flags, indexed by the original triangle number
     * (see `TRIANGLE_FLAG_ALPHA_TEST`). Defaults to 0.
     */
    readonly triangleFlags?: Uint32Array | null;
}

/** The result of building a mesh BVH. All indices are relative to this mesh. */
export interface MeshBvh {
    /** `nodeCount` nodes, 64 bytes each. Node 0 is the root. */
    readonly nodes: ArrayBuffer;
    /** `triangleCount` triangles, 48 bytes each, in tree order. */
    readonly triangles: ArrayBuffer;
    /** Parent of each node, for the bottom-up refit pass (M3). */
    readonly parents: Uint32Array;
    readonly nodeCount: number;
    readonly triangleCount: number;
    /** What the triangles store; the shader must be generated for the same test. */
    readonly triangleTest: TriangleTest;
    /** Levels of nodes on the longest path. The traversal stack needs at most this many entries for this mesh. */
    readonly maxDepth: number;
    readonly leafCount: number;
    /** Object-space box around the mesh: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly bounds: Float32Array;
}

/** Buffers to hand to `postMessage` so a built tree moves out of a worker without a copy. */
export function meshBvhTransferables(bvh: MeshBvh): ArrayBuffer[] {
    return [bvh.nodes, bvh.triangles, bvh.parents.buffer as ArrayBuffer, bvh.bounds.buffer as ArrayBuffer];
}

export function countTriangles(geometry: MeshGeometry): number {
    const entries = geometry.indices ? geometry.indices.length : geometry.positions.length / 3;
    if (!Number.isInteger(entries) || entries % 3 !== 0) {
        throw new RangeError('triangle geometry must have a multiple of three vertices or indices');
    }
    return entries / 3;
}

export interface MeshBvhBuildOptions extends Partial<SahBuildOptions> {
    /** Defaults to `moller-trumbore`. See {@link TriangleTest}. */
    readonly triangleTest?: TriangleTest;
}

export function buildMeshBvh(geometry: MeshGeometry, options: MeshBvhBuildOptions = {}): MeshBvh {
    const { triangleTest = DEFAULT_TRIANGLE_TEST, ...sahOptions } = options;
    const { positions, indices, triangleFlags } = geometry;
    if (positions.length % 3 !== 0) throw new RangeError('positions must hold three floats per vertex');
    const vertexCount = positions.length / 3;
    const triangleCount = countTriangles(geometry);
    if (triangleCount === 0) throw new RangeError('a mesh BVH needs at least one triangle');
    if (triangleCount > MAX_REFERENCE_INDEX) throw new RangeError(`too many triangles for one BVH: ${triangleCount}`);
    if (triangleFlags && triangleFlags.length < triangleCount) {
        throw new RangeError('triangleFlags must have one entry per triangle');
    }

    const vertexOf = (triangle: number, corner: number): number => {
        const vertex = indices ? indices[triangle * 3 + corner] : triangle * 3 + corner;
        if (vertex >= vertexCount) throw new RangeError(`triangle ${triangle} uses vertex ${vertex}, but there are ${vertexCount}`);
        return vertex;
    };

    // One box per triangle. These are exact: the positions are already f32.
    const triangleBounds = new Float32Array(triangleCount * 6);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
        const slot = triangle * 6;
        for (let axis = 0; axis < 3; axis++) {
            const a = positions[vertexOf(triangle, 0) * 3 + axis];
            const b = positions[vertexOf(triangle, 1) * 3 + axis];
            const c = positions[vertexOf(triangle, 2) * 3 + axis];
            triangleBounds[slot + axis] = Math.min(a, b, c);
            triangleBounds[slot + 3 + axis] = Math.max(a, b, c);
        }
    }

    const tree = buildSahTree(triangleBounds, { ...DEFAULT_SAH_OPTIONS, ...sahOptions });

    // Store triangles in tree order so a leaf is just "start index + count".
    const trianglesBuffer = new ArrayBuffer(triangleCount * TRIANGLE_BYTES);
    const floats = new Float32Array(trianglesBuffer);
    const words = new Uint32Array(trianglesBuffer);
    for (let slot = 0; slot < triangleCount; slot++) {
        const original = tree.order[slot];
        const base = slot * TRIANGLE_WORDS;
        const a = vertexOf(original, 0) * 3;
        const b = vertexOf(original, 1) * 3;
        const c = vertexOf(original, 2) * 3;
        for (let axis = 0; axis < 3; axis++) {
            floats[base + TRIANGLE_WORD.vertex0 + axis] = positions[a + axis];
            if (triangleTest === 'watertight') {
                // Corners are stored as they are, so a vertex shared by two triangles has identical bits in both.
                floats[base + TRIANGLE_WORD.secondVector + axis] = positions[b + axis];
                floats[base + TRIANGLE_WORD.thirdVector + axis] = positions[c + axis];
            } else {
                floats[base + TRIANGLE_WORD.secondVector + axis] = positions[b + axis] - positions[a + axis];
                floats[base + TRIANGLE_WORD.thirdVector + axis] = positions[c + axis] - positions[a + axis];
            }
        }
        words[base + TRIANGLE_WORD.originalIndex] = original;
        words[base + TRIANGLE_WORD.flags] = triangleFlags ? triangleFlags[original] : 0;
        words[base + TRIANGLE_WORD.padding] = 0;
    }

    return {
        nodes: tree.nodes,
        triangles: trianglesBuffer,
        parents: tree.parents,
        nodeCount: tree.nodeCount,
        triangleCount,
        triangleTest,
        maxDepth: tree.maxDepth,
        leafCount: tree.leafCount,
        bounds: tree.bounds,
    };
}
