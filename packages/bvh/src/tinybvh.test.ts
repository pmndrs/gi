import { describe, expect, it } from 'vitest';
import { createTinyBvhBuilder } from './tinybvh.ts';
import { TOPOLOGY_EMPTY_CHILD, TOPOLOGY_WORDS_PER_NODE, type MeshTreeTopology } from './topology.ts';

/** A random triangle soup, `x, y, z` per vertex, three vertices per triangle (deterministic). */
function randomSoup(count: number, seed: number): Float32Array {
    let state = seed >>> 0;
    const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    const positions = new Float32Array(count * 9);
    for (let triangle = 0; triangle < count; triangle++) {
        const center = [random() * 20 - 10, random() * 20 - 10, random() * 20 - 10];
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) positions[triangle * 9 + corner * 3 + axis] = center[axis] + random() - 0.5;
        }
    }
    return positions;
}

/** Walks the topology: every node reached once, every position in exactly one leaf. Returns the leaf sizes. */
function checkTopology(topology: MeshTreeTopology, triangleCount: number): number[] {
    const { order, children } = topology;
    expect([...order].sort((a, b) => a - b)).toEqual(Array.from({ length: triangleCount }, (_, i) => i));
    const covered = new Uint8Array(triangleCount);
    const seen = new Set<number>();
    const leafSizes: number[] = [];
    const stack = [0];
    while (stack.length > 0) {
        const node = stack.pop() as number;
        expect(seen.has(node)).toBe(false);
        seen.add(node);
        for (const side of [0, 1]) {
            const index = children[node * TOPOLOGY_WORDS_PER_NODE + side * 2];
            const count = children[node * TOPOLOGY_WORDS_PER_NODE + side * 2 + 1];
            if (index === TOPOLOGY_EMPTY_CHILD) continue;
            if (count === 0) {
                stack.push(index);
                continue;
            }
            leafSizes.push(count);
            for (let position = index; position < index + count; position++) {
                expect(covered[position]).toBe(0);
                covered[position] = 1;
            }
        }
    }
    expect(seen.size).toBe(children.length / TOPOLOGY_WORDS_PER_NODE);
    expect(covered.every((value) => value === 1)).toBe(true);
    return leafSizes;
}

describe('tinybvh in WASM', () => {
    it('builds a tree over every triangle, indexed or not, and can build again', async () => {
        const builder = await createTinyBvhBuilder();
        for (const count of [1, 2, 7, 3000]) {
            const positions = randomSoup(count, count);
            const plain = builder.build({ positions });
            checkTopology(plain.topology, count);
            // The same soup through an index buffer gives the same tree.
            const indices = Uint32Array.from({ length: count * 3 }, (_, i) => i);
            const indexed = builder.build({ positions, indices });
            expect(indexed.topology.children).toEqual(plain.topology.children);
            expect(indexed.topology.order).toEqual(plain.topology.order);
        }
    });

    it('refuses an index that points past the vertices', async () => {
        const builder = await createTinyBvhBuilder();
        expect(() => builder.build({ positions: randomSoup(1, 1), indices: Uint32Array.of(0, 1, 9) })).toThrow(/index 2/);
    });

    it("builds with its H-PLOC builder when asked (one triangle per leaf), and refuses a builder it doesn't have", async () => {
        const builder = await createTinyBvhBuilder();
        for (const count of [1, 2, 7, 3000]) {
            const leafSizes = checkTopology(builder.build({ positions: randomSoup(count, count) }, { builder: 'hploc' }).topology, count);
            expect(leafSizes.every((size) => size === 1)).toBe(true);
        }
        // The default builder still runs after an H-PLOC build: the choice is per build.
        expect(checkTopology(builder.build({ positions: randomSoup(3000, 1) }).topology, 3000).some((size) => size > 1)).toBe(true);
        expect(() => builder.build({ positions: randomSoup(1, 1) }, { builder: 'sweep' as 'hploc' })).toThrow(/unknown tinybvh builder/);
    });
});
