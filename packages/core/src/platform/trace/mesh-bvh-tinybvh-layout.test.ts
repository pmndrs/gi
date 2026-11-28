/**
 * Stage B0b: tinybvh's WASM writes the tracer's GPU layout itself. Its output
 * must be byte for byte what the JavaScript conversion (`buildMeshBvhFromTopology`)
 * makes from tinybvh's tree of the same mesh. tinybvh is single-threaded and
 * deterministic, so building the same mesh twice gives the same tree; the first
 * test checks that too.
 */

import {
    createTinyBvhBuilder,
    GPU_LAYOUT_MAX_LEAF_PRIMITIVES,
    GPU_LAYOUT_MAX_TRIANGLES,
    GPU_LAYOUT_NODE_BYTES,
    GPU_LAYOUT_TRIANGLE_BYTES,
    TOPOLOGY_EMPTY_CHILD,
    type GpuLayoutMeshBvh,
    type MeshTreeTopology,
    type TinyBvhBuilder,
} from '@pmndrs/bvh';
import { beforeAll, describe, expect, it } from 'vitest';
import { BVH_NODE_BYTES, MAX_LEAF_PRIMITIVES, MAX_REFERENCE_INDEX, TRIANGLE_BYTES } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhFromTopology, TOPOLOGY_WORDS_PER_NODE } from './mesh-bvh-topology.ts';
import { createRandom, randomTriangleSoup, tessellatedQuad, type Random } from './oracle-fixtures.ts';

let tinyBvh: TinyBvhBuilder;
beforeAll(async () => {
    tinyBvh = await createTinyBvhBuilder();
});

function bytesOf(data: ArrayBuffer | ArrayBufferView): Uint8Array {
    return data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

/**
 * The first byte at which two arrays differ, or -1 if they are the same.
 * (A deep `toEqual` on millions of bytes takes seconds.) Comparing bytes also
 * tells -0 from +0.
 */
function firstDifference(a: ArrayBuffer | ArrayBufferView, b: ArrayBuffer | ArrayBufferView): number {
    const left = bytesOf(a);
    const right = bytesOf(b);
    const length = Math.min(left.length, right.length);
    for (let i = 0; i < length; i++) if (left[i] !== right[i]) return i;
    return left.length === right.length ? -1 : length;
}

function expectSameBytes(fromWasm: GpuLayoutMeshBvh, fromJavaScript: MeshBvh): void {
    const counts = (bvh: GpuLayoutMeshBvh | MeshBvh) => [bvh.nodeCount, bvh.triangleCount, bvh.maxDepth, bvh.leafCount];
    expect(counts(fromWasm)).toEqual(counts(fromJavaScript));
    expect({
        nodes: firstDifference(fromWasm.nodes, fromJavaScript.nodes),
        triangles: firstDifference(fromWasm.triangles, fromJavaScript.triangles),
        parents: firstDifference(fromWasm.parents, fromJavaScript.parents),
        bounds: firstDifference(fromWasm.bounds, fromJavaScript.bounds),
    }).toEqual({ nodes: -1, triangles: -1, parents: -1, bounds: -1 });
}

/** The largest leaf tinybvh made, so the tests can show that cutting leaves was exercised. */
function largestLeaf(topology: MeshTreeTopology): number {
    const { children } = topology;
    let largest = 0;
    for (let word = 0; word < children.length; word += 2) {
        if (children[word] !== TOPOLOGY_EMPTY_CHILD) largest = Math.max(largest, children[word + 1]);
    }
    return largest;
}

/** Builds `geometry` both ways, checks the bytes match, and returns tinybvh's tree shape. */
function compareBothWays(geometry: MeshGeometry, maxLeafPrimitives?: number): MeshTreeTopology {
    const { topology } = tinyBvh.build(geometry);
    const fromJavaScript = buildMeshBvhFromTopology(geometry, topology, { maxLeafPrimitives });
    const fromWasm = tinyBvh.buildLayout(geometry, { maxLeafPrimitives }).layout;
    expectSameBytes(fromWasm, fromJavaScript);
    return topology;
}

/** The same triangle many times over: no split can separate them, so tinybvh makes one big leaf. */
function stackedTriangles(count: number): MeshGeometry {
    const positions = new Float32Array(count * 9);
    for (let triangle = 0; triangle < count; triangle++) positions.set([0, 0, 0, 1, 0, 0, 0, 1, 0], triangle * 9);
    return { positions };
}

/**
 * A grid of shared vertices whose coordinates are often zero, half of them
 * negative zero. Box comparisons between -0 and +0 are ties, so the bytes show
 * whether both sides compare in the same order.
 */
function signedZeroGrid(random: Random): MeshGeometry {
    const quad = tessellatedQuad(24, 1);
    const positions = Float32Array.from(quad.positions);
    for (let i = 0; i < positions.length; i++) {
        if (random() < 0.4) positions[i] = random() < 0.5 ? -0 : 0;
    }
    return { positions, indices: quad.indices };
}

/**
 * A triangle soup with NaN written into the coordinates `isNaN` picks. Box code
 * must skip a NaN the same way on both paths (`if (value < low)` is false for a
 * NaN). The NaN is the default quiet one: copying through JavaScript numbers
 * could quiet a signalling NaN, which the WASM copies bit for bit.
 */
function soupWithNaN(count: number, seed: number, isNaN: (triangle: number, corner: number, axis: number) => boolean): MeshGeometry {
    const positions = Float32Array.from(randomTriangleSoup(createRandom(seed), count, 10, 1).positions);
    for (let triangle = 0; triangle < count; triangle++) {
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) if (isNaN(triangle, corner, axis)) positions[triangle * 9 + corner * 3 + axis] = NaN;
        }
    }
    return { positions };
}

describe('tinybvh GPU layout written in WASM (B0b)', () => {
    it('agrees with layout.ts on sizes and limits', () => {
        expect(GPU_LAYOUT_NODE_BYTES).toBe(BVH_NODE_BYTES);
        expect(GPU_LAYOUT_TRIANGLE_BYTES).toBe(TRIANGLE_BYTES);
        expect(GPU_LAYOUT_MAX_LEAF_PRIMITIVES).toBe(MAX_LEAF_PRIMITIVES);
        expect(GPU_LAYOUT_MAX_TRIANGLES).toBe(MAX_REFERENCE_INDEX);
    });

    it('builds the same tree twice from the same mesh', () => {
        const geometry = randomTriangleSoup(createRandom(1), 5000, 10, 1);
        const first = tinyBvh.build(geometry).topology;
        const second = tinyBvh.build(geometry).topology;
        expect(second.order).toEqual(first.order);
        expect(second.children).toEqual(first.children);
    });

    it('matches the JavaScript conversion byte for byte on triangle soups of every size', () => {
        for (const count of [1, 2, 3, 5, 8, 9, 17, 64, 1000, 20000]) compareBothWays(randomTriangleSoup(createRandom(count), count, 10, 1));
    });

    it('matches on single-leaf roots, with and without cutting the leaf', () => {
        // One triangle: the root's right child is empty.
        const single = compareBothWays(randomTriangleSoup(createRandom(100), 1, 10, 1));
        expect(Array.from(single.children)).toEqual([0, 1, TOPOLOGY_EMPTY_CHILD, 0]);
        // Identical triangles can't be split, so tinybvh's root is one leaf; the root
        // cuts it in two (left half rounded up), then in half again for small leaf limits.
        const three = stackedTriangles(3);
        expect(Array.from(compareBothWays(three).children)).toEqual([0, 2, 2, 1]);
        const forty = stackedTriangles(40);
        expect(Array.from(compareBothWays(forty).children)).toEqual([0, 20, 20, 20]);
        for (let maxLeaf = 1; maxLeaf <= GPU_LAYOUT_MAX_LEAF_PRIMITIVES; maxLeaf++) {
            compareBothWays(three, maxLeaf);
            compareBothWays(forty, maxLeaf);
        }
    });

    it('matches when inner leaves are cut to fit, for every leaf limit', () => {
        // Clusters of identical triangles inside a soup give leaves bigger than the tracer's.
        const soup = randomTriangleSoup(createRandom(200), 600, 10, 1);
        const positions = Float32Array.from(soup.positions);
        for (let triangle = 0; triangle < 600; triangle += 30) {
            for (let copy = 1; copy < 20; copy++) positions.copyWithin((triangle + copy) * 9, triangle * 9, triangle * 9 + 9);
        }
        const clustered = { positions };
        const topology = compareBothWays(clustered);
        expect(topology.children.length).toBeGreaterThan(TOPOLOGY_WORDS_PER_NODE);
        expect(largestLeaf(topology)).toBeGreaterThan(GPU_LAYOUT_MAX_LEAF_PRIMITIVES);
        for (let maxLeaf = 1; maxLeaf < GPU_LAYOUT_MAX_LEAF_PRIMITIVES; maxLeaf++) compareBothWays(clustered, maxLeaf);
    });

    it('matches on shared vertices with 16- and 32-bit indices, signed zeros and triangle flags', () => {
        const grid = signedZeroGrid(createRandom(300));
        compareBothWays(grid);
        const indices = grid.indices as Uint32Array;
        expect(indices.every((index) => index < 0x10000)).toBe(true);
        compareBothWays({ positions: grid.positions, indices: Uint16Array.from(indices) });
        const triangleFlags = Uint32Array.from({ length: indices.length / 3 }, (_, triangle) => (triangle * 2654435761) >>> 0);
        compareBothWays({ ...grid, triangleFlags });
    });

    it('matches when some coordinates are NaN, and both refuse the tree when whole triangles are', () => {
        // One coordinate of some corners is NaN: tinybvh still builds a valid tree, and
        // the boxes skip the NaN on that axis only.
        for (const count of [5, 300, 3000]) {
            compareBothWays(soupWithNaN(count, count, (triangle, corner, axis) => triangle % 7 === 0 && corner === 1 && axis === 0));
        }
        // Every coordinate of every 11th triangle is NaN. With 300 triangles tinybvh's tree is
        // still whole; with 3000 it leaves positions out of every leaf, and both paths say so.
        compareBothWays(soupWithNaN(300, 300, (triangle) => triangle % 11 === 0));
        const broken = soupWithNaN(3000, 3000, (triangle) => triangle % 11 === 0);
        expect(() => buildMeshBvhFromTopology(broken, tinyBvh.build(broken).topology)).toThrow(/in no leaf/);
        expect(() => tinyBvh.buildLayout(broken)).toThrow(/in no leaf/);
    });

    it('refuses bad input the way the JavaScript path does', () => {
        const geometry = randomTriangleSoup(createRandom(400), 4, 10, 1);
        expect(() => tinyBvh.buildLayout({ positions: geometry.positions, indices: Uint32Array.of(0, 1, 2, 0, 1, 99) })).toThrow(/index 5 is 99/);
        expect(() => tinyBvh.buildLayout(geometry, { maxLeafPrimitives: 0 })).toThrow(/maxLeafPrimitives/);
        expect(() => tinyBvh.buildLayout(geometry, { maxLeafPrimitives: 9 })).toThrow(/maxLeafPrimitives/);
        expect(() => tinyBvh.buildLayout({ ...geometry, triangleFlags: new Uint32Array(3) })).toThrow(/triangleFlags/);
        // A failed build leaves the builder usable.
        expect(tinyBvh.buildLayout(geometry).layout.triangleCount).toBe(4);
    });
});
