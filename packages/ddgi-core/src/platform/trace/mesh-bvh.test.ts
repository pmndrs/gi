import { describe, expect, it } from 'vitest';
import { BVH_NODE_BYTES, NO_PARENT, NODE_WORD, EMPTY_CHILD_INDEX, TRIANGLE_BYTES, TRIANGLE_WORD, nodeView } from './layout.ts';
import { buildMeshBvh, type MeshBvh, type MeshGeometry } from './mesh-bvh.ts';
import { createMeshBvhBuilder } from './mesh-bvh-async.ts';
import { createRandom, randomTriangleSoup, tessellatedQuad } from './oracle-fixtures.ts';

/** Vertex coordinates of an original triangle, straight from the input geometry. */
function originalVertices(geometry: MeshGeometry, triangle: number): number[][] {
    return [0, 1, 2].map((corner) => {
        const vertex = geometry.indices ? geometry.indices[triangle * 3 + corner] : triangle * 3 + corner;
        return [geometry.positions[vertex * 3], geometry.positions[vertex * 3 + 1], geometry.positions[vertex * 3 + 2]];
    });
}

interface TreeFacts {
    leafSizes: number[];
    seenTriangles: number[];
    depth: number;
    emptyChildren: number;
}

/**
 * Walks the tree and asserts the structural rules: boxes contain what they
 * point at, parents match, every triangle is in exactly one leaf.
 *
 * Per-element rules are collected as problem strings and asserted once at the
 * end: a vitest `expect` per number made the 4000-triangle case take seconds
 * (and time out under load), while a plain comparison costs nothing.
 */
function checkTree(bvh: MeshBvh, geometry: MeshGeometry): TreeFacts {
    const { floats, words } = nodeView(bvh.nodes);
    const triangleWords = new Uint32Array(bvh.triangles);
    const triangleFloats = new Float32Array(bvh.triangles);
    const facts: TreeFacts = { leafSizes: [], seenTriangles: [], depth: 0, emptyChildren: 0 };

    expect(bvh.nodes.byteLength).toBe(bvh.nodeCount * BVH_NODE_BYTES);
    expect(bvh.triangles.byteLength).toBe(bvh.triangleCount * TRIANGLE_BYTES);
    expect(bvh.parents.length).toBe(bvh.nodeCount);
    expect(bvh.parents[0]).toBe(NO_PARENT);

    const problems: string[] = [];
    const check = (ok: boolean, describe: () => string): void => {
        if (!ok && problems.length < 20) problems.push(describe());
    };

    const visitedNodes = new Set<number>();
    const stack: { node: number; depth: number }[] = [{ node: 0, depth: 1 }];
    while (stack.length > 0) {
        const { node, depth } = stack.pop() as { node: number; depth: number };
        check(!visitedNodes.has(node), () => `node ${node} is reached twice`);
        visitedNodes.add(node);
        facts.depth = Math.max(facts.depth, depth);
        const base = node * 16;

        for (const side of ['left', 'right'] as const) {
            const minWord = base + (side === 'left' ? NODE_WORD.leftMin : NODE_WORD.rightMin);
            const maxWord = base + (side === 'left' ? NODE_WORD.leftMax : NODE_WORD.rightMax);
            const index = words[base + (side === 'left' ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)];
            const count = words[base + (side === 'left' ? NODE_WORD.leftCount : NODE_WORD.rightCount)];
            if (index === EMPTY_CHILD_INDEX) {
                check(side === 'right' && bvh.triangleCount === 1, () => `node ${node} has an empty ${side} child in a tree of ${bvh.triangleCount} triangles`);
                facts.emptyChildren++;
                continue;
            }
            const boxMin = [floats[minWord], floats[minWord + 1], floats[minWord + 2]];
            const boxMax = [floats[maxWord], floats[maxWord + 1], floats[maxWord + 2]];

            if (count === 0) {
                // Inner child: its parent entry is this node, and its own boxes sit inside this one.
                check(bvh.parents[index] === node, () => `node ${index} names parent ${bvh.parents[index]}, not ${node}`);
                const childBase = index * 16;
                for (const childSide of [NODE_WORD.leftMin, NODE_WORD.rightMin]) {
                    for (let axis = 0; axis < 3; axis++) {
                        check(
                            floats[childBase + childSide + axis] >= boxMin[axis] && floats[childBase + childSide + 4 + axis] <= boxMax[axis],
                            () => `node ${index}'s boxes stick out of its parent's box on axis ${axis}`,
                        );
                    }
                }
                stack.push({ node: index, depth: depth + 1 });
            } else {
                facts.leafSizes.push(count);
                for (let slot = index; slot < index + count; slot++) {
                    const original = triangleWords[slot * 12 + TRIANGLE_WORD.originalIndex];
                    facts.seenTriangles.push(original);
                    // The leaf's box must contain the triangle's real vertices.
                    for (const vertex of originalVertices(geometry, original)) {
                        for (let axis = 0; axis < 3; axis++) {
                            check(vertex[axis] >= boxMin[axis] && vertex[axis] <= boxMax[axis], () => `triangle ${original} sticks out of its leaf box on axis ${axis}`);
                        }
                    }
                    // The stored corner and edges rebuild the original vertices.
                    const [v0, v1, v2] = originalVertices(geometry, original);
                    for (let axis = 0; axis < 3; axis++) {
                        check(triangleFloats[slot * 12 + TRIANGLE_WORD.vertex0 + axis] === v0[axis], () => `triangle ${original}'s first corner is not stored exactly`);
                        const rebuiltV1 = triangleFloats[slot * 12 + TRIANGLE_WORD.vertex0 + axis] + triangleFloats[slot * 12 + TRIANGLE_WORD.secondVector + axis];
                        const rebuiltV2 = triangleFloats[slot * 12 + TRIANGLE_WORD.vertex0 + axis] + triangleFloats[slot * 12 + TRIANGLE_WORD.thirdVector + axis];
                        check(
                            Math.abs(rebuiltV1 - v1[axis]) < 5e-5 && Math.abs(rebuiltV2 - v2[axis]) < 5e-5,
                            () => `triangle ${original}'s edges do not rebuild its corners`,
                        );
                    }
                }
            }
        }
    }
    expect(problems).toEqual([]);
    expect(visitedNodes.size).toBe(bvh.nodeCount);
    return facts;
}

function expectEveryTriangleOnce(facts: TreeFacts, triangleCount: number): void {
    const sorted = [...facts.seenTriangles].sort((a, b) => a - b);
    expect(sorted.length).toBe(triangleCount);
    expect(sorted.every((value, position) => value === position)).toBe(true);
}

describe('buildMeshBvh', () => {
    for (const triangleCount of [1, 2, 3, 4, 5, 17, 300, 4000]) {
        it(`builds a valid tree for ${triangleCount} random triangles`, () => {
            const geometry = randomTriangleSoup(createRandom(triangleCount), triangleCount, 20, 1.5);
            const bvh = buildMeshBvh(geometry);
            const facts = checkTree(bvh, geometry);
            expectEveryTriangleOnce(facts, triangleCount);
            expect(Math.max(...facts.leafSizes)).toBeLessThanOrEqual(4);
            expect(facts.depth).toBe(bvh.maxDepth);
            expect(bvh.leafCount).toBe(facts.leafSizes.length);
            if (triangleCount === 1) expect(facts.emptyChildren).toBe(1);
        });
    }

    it('reports a depth that grows like log2 of the triangle count for well-spread triangles', () => {
        const geometry = randomTriangleSoup(createRandom(7), 8192, 50, 0.5);
        const bvh = buildMeshBvh(geometry);
        // Perfectly balanced with 4-triangle leaves would be 11 levels; SAH trees run deeper but not wildly.
        expect(bvh.maxDepth).toBeGreaterThanOrEqual(11);
        expect(bvh.maxDepth).toBeLessThan(40);
    });

    it('respects the leaf size option', () => {
        const geometry = randomTriangleSoup(createRandom(11), 500, 10, 1);
        for (const maxLeafPrimitives of [1, 2, 8]) {
            const bvh = buildMeshBvh(geometry, { maxLeafPrimitives });
            const facts = checkTree(bvh, geometry);
            expectEveryTriangleOnce(facts, 500);
            expect(Math.max(...facts.leafSizes)).toBeLessThanOrEqual(maxLeafPrimitives);
        }
        expect(() => buildMeshBvh(geometry, { maxLeafPrimitives: 9 })).toThrow(/maxLeafPrimitives/);
    });

    it('handles a flat tessellated quad (zero-thickness boxes)', () => {
        const geometry = tessellatedQuad(16, 2);
        const bvh = buildMeshBvh(geometry);
        expectEveryTriangleOnce(checkTree(bvh, geometry), 16 * 16 * 2);
    });

    it('terminates on identical triangles, where no plane separates the centroids', () => {
        const one = [0, 0, 0, 1, 0, 0, 0, 1, 0];
        const geometry: MeshGeometry = { positions: new Float32Array(Array.from({ length: 64 }, () => one).flat()) };
        const bvh = buildMeshBvh(geometry);
        expectEveryTriangleOnce(checkTree(bvh, geometry), 64);
        // Halving 64 identical triangles down to leaves of 4 needs about 5 levels.
        expect(bvh.maxDepth).toBeLessThanOrEqual(8);
    });

    it('keeps the original triangle index of indexed geometry and carries triangle flags', () => {
        const base = tessellatedQuad(4, 1);
        const flags = new Uint32Array(base.indices!.length / 3).map((_, triangle) => (triangle % 3 === 0 ? 1 : 0));
        const geometry: MeshGeometry = { ...base, triangleFlags: flags };
        const bvh = buildMeshBvh(geometry);
        const words = new Uint32Array(bvh.triangles);
        for (let slot = 0; slot < bvh.triangleCount; slot++) {
            const original = words[slot * 12 + TRIANGLE_WORD.originalIndex];
            expect(words[slot * 12 + TRIANGLE_WORD.flags]).toBe(flags[original]);
        }
    });

    it('rejects bad input instead of building a tree that misses', () => {
        expect(() => buildMeshBvh({ positions: new Float32Array(9).fill(NaN) })).toThrow(/non-finite/);
        expect(() => buildMeshBvh({ positions: new Float32Array(0) })).toThrow(/at least one triangle/);
        expect(() => buildMeshBvh({ positions: new Float32Array(10) })).toThrow(/three floats/);
        expect(() => buildMeshBvh({ positions: new Float32Array(9), indices: new Uint32Array([0, 1, 5]) })).toThrow(/vertex 5/);
    });

    it('stores a parent for every node so a refit can walk upward', () => {
        const geometry = randomTriangleSoup(createRandom(3), 1000, 10, 1);
        const bvh = buildMeshBvh(geometry);
        for (let node = 1; node < bvh.nodeCount; node++) {
            expect(bvh.parents[node]).toBeLessThan(node);
        }
    });
});

describe('createMeshBvhBuilder', () => {
    it('builds inline when no worker is available (Node) and matches a direct build', async () => {
        const geometry = randomTriangleSoup(createRandom(5), 200, 10, 1);
        const builder = createMeshBvhBuilder();
        expect(builder.usesWorker).toBe(false);
        const built = await builder.build(geometry);
        const direct = buildMeshBvh(geometry);
        expect(built.nodeCount).toBe(direct.nodeCount);
        expect(built.maxDepth).toBe(direct.maxDepth);
        builder.dispose();
    });

    it('refuses to fall back when a worker is required', () => {
        expect(() => createMeshBvhBuilder({ worker: 'always' })).toThrow(/worker/);
    });

    it('rejects when the geometry is invalid', async () => {
        const builder = createMeshBvhBuilder({ worker: 'never' });
        await expect(builder.build({ positions: new Float32Array(0) })).rejects.toThrow(/at least one triangle/);
    });
});
