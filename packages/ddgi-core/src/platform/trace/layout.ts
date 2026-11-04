/**
 * Memory layout of the ray-tracing acceleration structures, in one place.
 *
 * The CPU builders write these layouts and the WGSL reads them, so every
 * number the two sides must agree on lives here, and the WGSL source is
 * generated from these constants rather than repeating them.
 *
 * Everything is plain 32-bit words in a few buffers so it survives a worker
 * hand-off (transferable `ArrayBuffer`s) and a GPU upload without conversion.
 * See docs/design/trace.md for the reasoning.
 */

/** One BVH node: two child boxes (Aila-Laine layout), 16 words. */
export const BVH_NODE_BYTES = 64;
export const BVH_NODE_WORDS = BVH_NODE_BYTES / 4;

/**
 * Word positions inside a node. Matches the WGSL `GiBvhNode`:
 *
 *     leftMin: vec3f,  leftIndex: u32,
 *     leftMax: vec3f,  rightIndex: u32,
 *     rightMin: vec3f, leftCount: u32,
 *     rightMax: vec3f, rightCount: u32,
 *
 * A child with `count > 0` is a leaf holding `count` primitives starting at
 * `index`. A child with `count == 0` is an inner node at `index`.
 */
export const NODE_WORD = {
    leftMin: 0,
    leftIndex: 3,
    leftMax: 4,
    rightIndex: 7,
    rightMin: 8,
    leftCount: 11,
    rightMax: 12,
    rightCount: 15,
} as const;

/**
 * `index` value of a child that does not exist. Only a tree with a single
 * primitive has one: its root node holds that primitive on the left and
 * nothing on the right.
 */
export const EMPTY_CHILD_INDEX = 0xffffffff;

/** Parent index of the root node in the parent array. */
export const NO_PARENT = 0xffffffff;

/**
 * How the ray-triangle test is done. Changes what a triangle stores, so the
 * builder and the shader must be given the same value.
 *
 * - `moller-trumbore`: the design's default. Triangles store a corner and two
 *   edges (`v0`, `v1 - v0`, `v2 - v0`), which is all the test needs.
 * - `watertight`: Woop, Benthin and Wald 2013. Triangles store three corners,
 *   because the test must see the *same bits* for a vertex shared by two
 *   triangles, and `v0 + edge` does not round the same way twice.
 */
export type TriangleTest = 'moller-trumbore' | 'watertight';

export const DEFAULT_TRIANGLE_TEST: TriangleTest = 'moller-trumbore';

/** One triangle in tree order: 48 bytes either way, so memory does not change with the test. */
export const TRIANGLE_BYTES = 48;
export const TRIANGLE_WORDS = TRIANGLE_BYTES / 4;

/**
 * Word positions inside a triangle. Matches the WGSL `GiBvhTriangle`.
 * `secondVector` is `v1 - v0` (Moller-Trumbore) or `v1` (watertight);
 * `thirdVector` is `v2 - v0` or `v2`.
 */
export const TRIANGLE_WORD = {
    vertex0: 0,
    originalIndex: 3,
    secondVector: 4,
    flags: 7,
    thirdVector: 8,
    padding: 11,
} as const;

/** Triangle `flags` bit 0: the material needs an alpha test (used from M4). */
export const TRIANGLE_FLAG_ALPHA_TEST = 1;

/**
 * An instance record: the world-to-object transform as three rows of
 * `vec4f` (the 3x4 matrix), then `(rootNodeOffset, triangleOffset,
 * instanceIndex, flags)` as four u32.
 */
export const INSTANCE_RECORD_BYTES = 64;
export const INSTANCE_RECORD_WORDS = INSTANCE_RECORD_BYTES / 4;

/** Word positions inside an instance record. */
export const INSTANCE_WORD = {
    worldToObjectRow0: 0,
    worldToObjectRow1: 4,
    worldToObjectRow2: 8,
    rootNodeOffset: 12,
    triangleOffset: 13,
    instanceIndex: 14,
    flags: 15,
} as const;

/** Instance `flags` bits. */
export const INSTANCE_FLAG = {
    /** Rays that hit this instance count (an instance can be hidden from GI only). */
    contributesToGi: 1,
    /** The instance is drawn at all. */
    visible: 2,
    /**
     * The object-to-world transform has a negative determinant (a mirror). It
     * flips triangle winding, so `frontFace` must be flipped to match.
     */
    mirrored: 4,
} as const;

/**
 * The top-level buffer is `[header][instance records][top-level nodes]`.
 * The header is 64 bytes so records and nodes stay 64-byte aligned.
 */
export const TOP_LEVEL_HEADER_BYTES = 64;
export const TOP_LEVEL_HEADER_WORDS = TOP_LEVEL_HEADER_BYTES / 4;

/** Word positions inside the top-level header. */
export const TOP_HEADER_WORD = {
    instanceCount: 0,
    nodeCount: 1,
    /** Informational: how many levels the top-level tree has. */
    maxDepth: 2,
} as const;

/**
 * The traversal stack is shared by both levels (see the WGSL). Stack entries
 * are 32-bit "child references": the index in the low bits and the leaf
 * primitive count in the top bits (0 = inner node).
 */
export const TRACE_STACK_SIZE = 64;
export const REF_COUNT_SHIFT = 28;
export const REF_INDEX_MASK = 0x0fffffff;
/** Pushed when entering an instance; popping it switches back to the world-space ray. */
export const REF_INSTANCE_MARKER = 0xffffffff;
/** Largest leaf the reference encoding can hold without colliding with {@link REF_INSTANCE_MARKER}. */
export const MAX_LEAF_PRIMITIVES = 8;
/** Node and triangle indices must fit under the reference index mask. */
export const MAX_REFERENCE_INDEX = REF_INDEX_MASK;

export function makeReference(index: number, count: number): number {
    return ((count << REF_COUNT_SHIFT) | index) >>> 0;
}

export function referenceIndex(reference: number): number {
    return reference & REF_INDEX_MASK;
}

export function referenceCount(reference: number): number {
    return reference >>> REF_COUNT_SHIFT;
}

/**
 * Read-only typed views over a node array, so builders, tests and the CPU
 * traversal share one way to touch the words.
 */
export interface NodeView {
    readonly floats: Float32Array;
    readonly words: Uint32Array;
}

export function nodeView(buffer: ArrayBuffer, byteOffset = 0, byteLength = buffer.byteLength - byteOffset): NodeView {
    return {
        floats: new Float32Array(buffer, byteOffset, byteLength / 4),
        words: new Uint32Array(buffer, byteOffset, byteLength / 4),
    };
}
