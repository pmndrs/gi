/**
 * Turns watertight triangle records into Moller-Trumbore ones, in place.
 *
 * The two triangle tests store the same 48 bytes differently: watertight keeps
 * the three corners, Moller-Trumbore keeps one corner and two edges. The tree
 * is identical, so the benchmark builds each scene once (watertight) and
 * derives the Moller-Trumbore triangles here instead of building twice, which
 * would double the memory and the page's run time on Bistro.
 *
 * It is exact: the builder computes an edge as `float32(corner - v0)` from the
 * same float32 corners, and so does this. A test checks that against a real
 * Moller-Trumbore build.
 */

import { TRIANGLE_WORD, TRIANGLE_WORDS } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';

export function convertWatertightToMollerTrumbore(triangles: ArrayBuffer): void {
    const floats = new Float32Array(triangles);
    for (let base = 0; base + TRIANGLE_WORDS <= floats.length; base += TRIANGLE_WORDS) {
        for (let axis = 0; axis < 3; axis++) {
            const corner0 = floats[base + TRIANGLE_WORD.vertex0 + axis];
            floats[base + TRIANGLE_WORD.secondVector + axis] -= corner0;
            floats[base + TRIANGLE_WORD.thirdVector + axis] -= corner0;
        }
    }
}
