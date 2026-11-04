/**
 * Per-mesh BVH (the "bottom level"): turns triangle geometry into the node
 * array and the tree-ordered triangle array the GPU traverses.
 *
 * Runs anywhere (main thread, worker, Node). `mesh-bvh-async.ts` wraps it in a
 * worker for large meshes.
 */

import { buildSahTree, DEFAULT_SAH_OPTIONS, type SahBuildOptions } from './sah-builder.ts';
import { checkGeometry, packTriangles } from './triangle-packing.ts';

export { countTriangles } from './triangle-packing.ts';

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

/**
 * Who decides the tree's shape:
 * - `sah`: our binned SAH builder in this file (synchronous);
 * - `tinybvh`: tinybvh's builder in WASM (@pmndrs/bvh). It needs an asynchronous start,
 *   so it is only available through `buildMeshBvhAsync` and `createMeshBvhBuilder`.
 */
export type MeshBvhBuilderKind = 'sah' | 'tinybvh';

export interface MeshBvhBuildOptions extends Partial<SahBuildOptions> {
    /** Defaults to `sah`. See {@link MeshBvhBuilderKind}. */
    readonly builder?: MeshBvhBuilderKind;
}

export function buildMeshBvh(geometry: MeshGeometry, options: MeshBvhBuildOptions = {}): MeshBvh {
    const { builder = 'sah', ...sahOptions } = options;
    if (builder !== 'sah') throw new RangeError(`buildMeshBvh only runs the 'sah' builder; use buildMeshBvhAsync for '${builder}'`);
    const { positions } = geometry;
    const checked = checkGeometry(geometry);
    const { triangleCount, vertexOf } = checked;

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
    const trianglesBuffer = packTriangles(geometry, checked, tree.order);

    return {
        nodes: tree.nodes,
        triangles: trianglesBuffer,
        parents: tree.parents,
        nodeCount: tree.nodeCount,
        triangleCount,
        maxDepth: tree.maxDepth,
        leafCount: tree.leafCount,
        bounds: tree.bounds,
    };
}
