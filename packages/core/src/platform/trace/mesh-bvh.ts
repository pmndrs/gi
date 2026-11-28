/**
 * Per-mesh BVH (the "bottom level"): the shapes a mesh tree has on its way to
 * the GPU, the node array and the tree-ordered triangle array the walk reads.
 *
 * tinybvh (WASM, `@pmndrs/bvh`) builds every mesh tree and writes this layout
 * itself (`mesh-bvh-build.ts`); `mesh-bvh-async.ts` runs it in a worker. Trees
 * built elsewhere (three-mesh-bvh's) come in through `mesh-bvh-topology.ts`.
 */

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

/** Options for building a mesh tree. */
export interface MeshBvhBuildOptions {
    /** Leaves with more triangles than this are cut in half until they fit. 1 to 8; default 8. */
    readonly maxLeafPrimitives?: number;
}
