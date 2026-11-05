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
export const GPU_LAYOUT_NODE_WORDS = GPU_LAYOUT_NODE_BYTES / 4;

/**
 * Word positions inside a node, matching the tracer's WGSL `GiBvhNode`:
 *
 *     leftMin: vec3f,  leftIndex: u32,
 *     leftMax: vec3f,  rightIndex: u32,
 *     rightMin: vec3f, leftCount: u32,
 *     rightMax: vec3f, rightCount: u32,
 *
 * A child with `count > 0` is a leaf holding `count` primitives starting at
 * `index`; `count == 0` is an inner node at `index`. Child `side` (0 left,
 * 1 right) keeps its box at words `8 * side` (min) and `8 * side + 4` (max).
 */
export const GPU_LAYOUT_NODE_WORD = {
    leftMin: 0,
    leftIndex: 3,
    leftMax: 4,
    rightIndex: 7,
    rightMin: 8,
    leftCount: 11,
    rightMax: 12,
    rightCount: 15,
} as const;

/** Words between the left child's box and the right child's box (see {@link GPU_LAYOUT_NODE_WORD}). */
export const GPU_LAYOUT_CHILD_STRIDE_WORDS = 8;

/**
 * `index` of a child that does not exist. Only a tree with a single primitive
 * has one: its root holds that primitive on the left and nothing on the right.
 */
export const GPU_LAYOUT_EMPTY_CHILD = 0xffffffff;

/** Parent of the root node in a parent array. */
export const GPU_LAYOUT_NO_PARENT = 0xffffffff;

/** One triangle in tree order: three corners, the original index and flags, 48 bytes. */
export const GPU_LAYOUT_TRIANGLE_BYTES = 48;

/**
 * Word positions inside a triangle, matching the tracer's WGSL `GiBvhTriangle`
 * (and ddgi-core's `TRIANGLE_WORD`): each corner is a `vec3f` followed by one
 * word, the original triangle index after the first, the flags after the second.
 */
export const GPU_LAYOUT_TRIANGLE_WORD = {
    vertex0: 0,
    originalIndex: 3,
    vertex1: 4,
    flags: 7,
    vertex2: 8,
    padding: 11,
} as const;

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
