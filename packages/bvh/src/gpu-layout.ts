/**
 * The tracer's GPU layout, as tinybvh's WASM writes it (stage B0b): what a
 * builder hands to the tracer when it writes the layout itself instead of a
 * `MeshTreeTopology` (topology.ts). The layout is defined, word by word, in
 * `packages/ddgi-core/src/platform/trace/layout.ts` (and the WGSL generated
 * from it); `native/tinybvh-wasm.cpp` repeats it as C structs.
 *
 * The sizes and limits below are the one definition: layout.ts imports them
 * (this package can't import the core). A test in ddgi-core checks that the
 * WASM's bytes equal the JavaScript conversion's.
 */

/** One node: two child boxes (Aila-Laine layout), 64 bytes. */
export const GPU_LAYOUT_NODE_BYTES = 64;

/** One triangle in tree order: three corners, the original index and flags, 48 bytes. */
export const GPU_LAYOUT_TRIANGLE_BYTES = 48;

/** The tracer's largest leaf; larger leaves are cut in half until they fit. */
export const GPU_LAYOUT_MAX_LEAF_PRIMITIVES = 8;

/** Most triangles one mesh may have: triangle positions must fit the tracer's 28-bit references. */
export const GPU_LAYOUT_MAX_TRIANGLES = 0x0fffffff;

/** A mesh's tree in the GPU layout. The same shape as the tracer's `MeshBvh`. */
export interface GpuLayoutMeshBvh {
    /** `nodeCount` nodes, 64 bytes each. Node 0 is the root. */
    readonly nodes: ArrayBuffer;
    /** `triangleCount` triangles, 48 bytes each, in tree order. */
    readonly triangles: ArrayBuffer;
    /** Parent of each node (0xffffffff for the root), for a bottom-up refit. */
    readonly parents: Uint32Array;
    readonly nodeCount: number;
    readonly triangleCount: number;
    /** Levels of nodes on the longest path. */
    readonly maxDepth: number;
    readonly leafCount: number;
    /** Object-space box around the mesh: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly bounds: Float32Array;
}
