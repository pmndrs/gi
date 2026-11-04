import { describe, expect, it } from 'vitest';
import { createCpuTracer } from './cpu-traversal.ts';
import { BVH_NODE_WORDS, EMPTY_CHILD_INDEX, NODE_WORD, TRIANGLE_WORD, TRIANGLE_WORDS } from './layout.ts';
import { buildMeshBvh, type MeshBvh, type MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhAsync } from './mesh-bvh-build.ts';
import { buildMeshBvhFromTopology, TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { createRandom, randomRays, randomTriangleSoup } from './oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene } from './scene-assembly.ts';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** The shape of one of our own trees, as the topology input describes it. */
function topologyOf(bvh: MeshBvh): MeshTreeTopology {
    const words = new Uint32Array(bvh.nodes);
    const triangleWords = new Uint32Array(bvh.triangles);
    const children = new Uint32Array(bvh.nodeCount * TOPOLOGY_WORDS_PER_NODE);
    for (let node = 0; node < bvh.nodeCount; node++) {
        const base = node * BVH_NODE_WORDS;
        children.set(
            [words[base + NODE_WORD.leftIndex], words[base + NODE_WORD.leftCount], words[base + NODE_WORD.rightIndex], words[base + NODE_WORD.rightCount]],
            node * TOPOLOGY_WORDS_PER_NODE,
        );
    }
    const order = new Uint32Array(bvh.triangleCount);
    for (let slot = 0; slot < bvh.triangleCount; slot++) order[slot] = triangleWords[slot * TRIANGLE_WORDS + TRIANGLE_WORD.originalIndex];
    return { order, children };
}

/** Closest and any-hit mismatches of the CPU walk over `bvh` against the brute-force reference. */
function mismatchesAgainstReference(geometry: MeshGeometry, bvh: MeshBvh, seed: number): number {
    const instances = [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY }];
    const tracer = createCpuTracer(assembleTraceScene([bvh], instances));
    const reference = buildReferenceScene([geometry], instances);
    const closest = newTally();
    const any = newTally();
    randomRays(createRandom(seed), 1500, [-10, -10, -10, 10, 10, 10], 4).forEach((ray, index) => {
        const hit = tracer.closest(ray);
        addToTally(closest, compareClosest(hit && { hit: true, ...hit }, traceClosestReference(reference, ray), ray), `ray ${index}`);
        addToTally(any, compareAny(tracer.any(ray), traceAnyReference(reference, ray)), `ray ${index}`);
    });
    return closest.mismatches + any.mismatches;
}

describe('buildMeshBvhFromTopology', () => {
    it('rebuilds one of our own trees byte for byte from its shape alone', () => {
        for (const count of [1, 2, 9, 700]) {
            const geometry = randomTriangleSoup(createRandom(count), count, 10, 1);
            const original = buildMeshBvh(geometry, { triangleTest: 'watertight' });
            const rebuilt = buildMeshBvhFromTopology(geometry, topologyOf(original), { triangleTest: 'watertight' });
            expect(new Uint8Array(rebuilt.nodes)).toEqual(new Uint8Array(original.nodes));
            expect(new Uint8Array(rebuilt.triangles)).toEqual(new Uint8Array(original.triangles));
            expect(rebuilt.parents).toEqual(original.parents);
            expect([rebuilt.maxDepth, rebuilt.leafCount]).toEqual([original.maxDepth, original.leafCount]);
            expect(rebuilt.bounds).toEqual(original.bounds);
        }
    });

    it('cuts leaves that are too big, recomputes boxes, and still traces like the reference', () => {
        const geometry = randomTriangleSoup(createRandom(40), 300, 10, 1);
        // A deliberately poor tree: the root holds two leaves of 150 triangles in input order.
        const topology = { order: Uint32Array.from({ length: 300 }, (_, i) => i), children: Uint32Array.of(0, 150, 150, 150) };
        const bvh = buildMeshBvhFromTopology(geometry, topology, { triangleTest: 'watertight', maxLeafPrimitives: 4 });
        const words = new Uint32Array(bvh.nodes);
        for (let node = 0; node < bvh.nodeCount; node++) {
            expect(words[node * BVH_NODE_WORDS + NODE_WORD.leftCount]).toBeLessThanOrEqual(4);
            expect(words[node * BVH_NODE_WORDS + NODE_WORD.rightCount]).toBeLessThanOrEqual(4);
        }
        expect(bvh.leafCount).toBeGreaterThanOrEqual(75);
        expect(mismatchesAgainstReference(geometry, bvh, 41)).toBe(0);
    });

    it('refuses shapes that do not describe a tree over every triangle', () => {
        const geometry = randomTriangleSoup(createRandom(50), 4, 10, 1);
        const order = Uint32Array.of(0, 1, 2, 3);
        expect(() => buildMeshBvhFromTopology(geometry, { order: Uint32Array.of(0, 1, 1, 3), children: Uint32Array.of(0, 2, 2, 2) })).toThrow(/every triangle/);
        expect(() => buildMeshBvhFromTopology(geometry, { order, children: Uint32Array.of(0, 2, 1, 2) })).toThrow(/more than one leaf/);
        expect(() => buildMeshBvhFromTopology(geometry, { order, children: Uint32Array.of(0, 2, 2, 1) })).toThrow(/no leaf/);
        expect(() => buildMeshBvhFromTopology(geometry, { order, children: Uint32Array.of(0, 0, 0, 4) })).toThrow(/reached twice|more nodes/);
        expect(() => buildMeshBvhFromTopology(geometry, { order, children: Uint32Array.of(0, 4, EMPTY_CHILD_INDEX, 0) })).toThrow(/empty child/);
    });
});

describe('tinybvh-built mesh BVHs', () => {
    it('trace like the brute-force reference', async () => {
        const geometry = randomTriangleSoup(createRandom(60), 2000, 10, 1);
        const bvh = await buildMeshBvhAsync(geometry, { builder: 'tinybvh', triangleTest: 'watertight' });
        expect(bvh.triangleCount).toBe(2000);
        expect(bvh.triangleTest).toBe('watertight');
        expect(mismatchesAgainstReference(geometry, bvh, 61)).toBe(0);
    });

    it('is refused by the synchronous builder, which cannot start WASM', () => {
        expect(() => buildMeshBvh(randomTriangleSoup(createRandom(62), 4, 10, 1), { builder: 'tinybvh' })).toThrow(/buildMeshBvhAsync/);
    });
});
