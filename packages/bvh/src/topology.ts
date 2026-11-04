/**
 * The Builder contract's output, CPU side: the *shape* of a binary tree over a
 * mesh's triangles, as plain arrays. Any builder (tinybvh in WASM, an engine's
 * own BVH, later a GPU build read back) produces this; the tracer turns it into
 * its GPU layout and computes the boxes itself, so a builder only has to get
 * the tree right, not the bytes.
 */

/** Words per node in {@link MeshTreeTopology.children}: leftIndex, leftCount, rightIndex, rightCount. */
export const TOPOLOGY_WORDS_PER_NODE = 4;

/** `index` of a right child that does not exist. Only a one-triangle mesh has one. */
export const TOPOLOGY_EMPTY_CHILD = 0xffffffff;

/**
 * The shape of a binary tree over a mesh's triangles.
 *
 * - `order[i]` is the original triangle stored at position `i`: a permutation
 *   of `0 .. triangleCount - 1`.
 * - `children` holds four words per node, node 0 being the root:
 *   `leftIndex, leftCount, rightIndex, rightCount`. A child with `count > 0` is
 *   a leaf of the triangles at positions `[index, index + count)` of `order`;
 *   `count == 0` means the inner node `index`. Only a one-triangle mesh may
 *   have an empty right child (`index` = {@link TOPOLOGY_EMPTY_CHILD}).
 */
export interface MeshTreeTopology {
    readonly order: Uint32Array;
    readonly children: Uint32Array;
}

/**
 * The topology of a tree that is a single leaf: the root still needs two
 * children, so the run is cut in two (or, for one triangle, the right child is empty).
 */
export function singleLeafTopology(order: Uint32Array, start: number, count: number): MeshTreeTopology {
    const half = Math.ceil(count / 2);
    const children =
        count === 1 ? Uint32Array.of(start, 1, TOPOLOGY_EMPTY_CHILD, 0) : Uint32Array.of(start, half, start + half, count - half);
    return { order, children };
}
