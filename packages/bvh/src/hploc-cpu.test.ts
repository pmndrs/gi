import { describe, expect, it } from 'vitest';
import { hplocSlotsFor, hplocWgsl, HPLOC_MAX_LEVEL, HPLOC_MAX_MESH_TRIANGLES } from './gpu-hploc-wgsl.ts';
import { buildHplocOnCpu, hplocMortonKeys, hplocPairRank, hplocSortedOrder, HPLOC_MAX_CLUSTERS } from './hploc-cpu.ts';
import { checkGpuLayout, treeSahCost } from './layout-check.ts';
import { triangleBoxes } from './lbvh-cpu.ts';
import { createTinyBvhBuilder, type TinyBvhInput } from './tinybvh.ts';
import { wgslNameProblems } from './wgsl-names.ts';

/** A random triangle soup, `x, y, z` per vertex, three vertices per triangle (deterministic). */
function randomSoup(count: number, seed: number, spread = 20): Float32Array {
    let state = seed >>> 0;
    const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    const positions = new Float32Array(count * 9);
    for (let triangle = 0; triangle < count; triangle++) {
        const center = [random() * spread - spread / 2, random() * spread - spread / 2, random() * spread - spread / 2];
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) positions[triangle * 9 + corner * 3 + axis] = center[axis] + random() - 0.5;
        }
    }
    return positions;
}

/** A wavy grid of `size` x `size` quads, indexed (shared vertices). */
function grid(size: number): TinyBvhInput {
    const positions = new Float32Array((size + 1) * (size + 1) * 3);
    for (let z = 0; z <= size; z++) {
        for (let x = 0; x <= size; x++) positions.set([x * 0.1, Math.sin(x * 0.3) * Math.cos(z * 0.2), z * 0.1], (z * (size + 1) + x) * 3);
    }
    const indices = new Uint32Array(size * size * 6);
    for (let z = 0; z < size; z++) {
        for (let x = 0; x < size; x++) {
            const corner = z * (size + 1) + x;
            indices.set([corner, corner + 1, corner + size + 1, corner + 1, corner + size + 2, corner + size + 1], (z * size + x) * 6);
        }
    }
    return { positions, indices };
}

const meshes: Record<string, TinyBvhInput> = {
    'one triangle': { positions: randomSoup(1, 1) },
    'two triangles': { positions: randomSoup(2, 2) },
    'seventeen triangles': { positions: randomSoup(17, 3) },
    'random soup': { positions: randomSoup(5000, 4) },
    'long soup': { positions: randomSoup(3000, 5).map((value, index) => (index % 3 === 0 ? value * 40 : value)) },
    'flat soup': { positions: randomSoup(3000, 6).map((value, index) => (index % 3 === 1 ? 0 : value)) },
    grid: grid(60),
};

describe('H-PLOC on the CPU (the oracle for the GPU H-PLOC)', () => {
    it("with tinybvh's codes and ties, builds tinybvh's BuildHPLOC tree", async () => {
        const tinyBvh = await createTinyBvhBuilder();
        for (const [name, geometry] of Object.entries(meshes)) {
            const theirs = tinyBvh.buildLayout(geometry, { builder: 'hploc', maxLeafPrimitives: 1 }).layout;
            const ours = buildHplocOnCpu(geometry, { mortonCodes: 'tinybvh-63', tieBreak: 'tinybvh', maxLeafPrimitives: 1 });
            expect(checkGpuLayout(ours, geometry, { maxLeafPrimitives: 1 }), name).toEqual([]);
            // The same tree, numbered differently: equal node counts, depth and cost.
            expect(ours.nodeCount, name).toBe(theirs.nodeCount);
            expect(ours.maxDepth, name).toBe(theirs.maxDepth);
            expect(treeSahCost(ours), name).toBeCloseTo(treeSahCost(theirs), 4);
        }
    });

    it('makes valid trees with every option, collapsed leaves included', () => {
        for (const [name, geometry] of Object.entries(meshes)) {
            for (const mortonCodes of ['cubic-30', 'tinybvh-63'] as const) {
                for (const maxLeafPrimitives of [1, 4, 8]) {
                    const layout = buildHplocOnCpu(geometry, { mortonCodes, maxLeafPrimitives });
                    expect(checkGpuLayout(layout, geometry, { maxLeafPrimitives }), `${name}, ${mortonCodes}, leaves of ${maxLeafPrimitives}`).toEqual([]);
                }
            }
        }
    });

    it('builds trees within a few percent of tinybvh\'s default build, and better than its own leaves of one', async () => {
        const tinyBvh = await createTinyBvhBuilder();
        for (const name of ['random soup', 'flat soup', 'grid']) {
            const geometry = meshes[name];
            const best = treeSahCost(tinyBvh.buildLayout(geometry).layout);
            const collapsed = treeSahCost(buildHplocOnCpu(geometry));
            const single = treeSahCost(buildHplocOnCpu(geometry, { maxLeafPrimitives: 1 }));
            expect(collapsed / best, name).toBeLessThan(1.15);
            expect(collapsed, name).toBeLessThan(single);
        }
    });

    it('pairs up exact ties instead of chaining the tree', async () => {
        // 3000 identical triangles: every merged area ties.
        const positions = Float32Array.from({ length: 3000 * 9 }, (_, index) => [0, 0, 0, 1, 0, 0, 0, 1, 0][index % 9]);
        const tinyBvh = await createTinyBvhBuilder();
        const chained = tinyBvh.buildLayout({ positions }, { builder: 'hploc', maxLeafPrimitives: 1 }).layout;
        const paired = buildHplocOnCpu({ positions }, { mortonCodes: 'tinybvh-63', maxLeafPrimitives: 1 });
        expect(chained.maxDepth).toBeGreaterThan(64);
        expect(paired.maxDepth).toBeLessThanOrEqual(Math.ceil(Math.log2(3000)) + 1);
        // Identical boxes cost the same in any shape.
        expect(treeSahCost(paired)).toBeCloseTo(treeSahCost(chained), 6);
    });

    it('orders tied pairs the same from both sides, nearer and even-aligned first', () => {
        expect(hplocPairRank(3, 4)).toBe(hplocPairRank(4, 3));
        expect(hplocPairRank(4, 5)).toBeLessThan(hplocPairRank(3, 4));
        expect(hplocPairRank(4, 5)).toBeLessThan(hplocPairRank(4, 6));
        const ranks = new Set<number>();
        for (let a = 0; a < HPLOC_MAX_CLUSTERS; a++) for (let b = a + 1; b < Math.min(a + 9, HPLOC_MAX_CLUSTERS); b++) ranks.add(hplocPairRank(a, b));
        // Distinct for every pair the search can see, so every tie is broken.
        let pairs = 0;
        for (let a = 0; a < HPLOC_MAX_CLUSTERS; a++) pairs += Math.min(8, HPLOC_MAX_CLUSTERS - 1 - a);
        expect(ranks.size).toBe(pairs);
    });

    it("computes tinybvh's 63-bit codes and sorts them by both words", () => {
        // Two triangles at opposite corners of the mesh box, one in the middle.
        const positions = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5]);
        const { boxes, centroidBounds } = triangleBoxes({ positions });
        const { high, low } = hplocMortonKeys(boxes, centroidBounds, 'tinybvh-63');
        expect([high[0], low[0]]).toEqual([0, 0]);
        // Every one of the 63 bits set at the far corner.
        expect([high[1], low[1]]).toEqual([0x7fffffff, 0xffffffff]);
        expect([...hplocSortedOrder(high, low)]).toEqual([0, 2, 1]);
        // Equal high words: the low word decides; equal codes: the triangle order.
        expect([...hplocSortedOrder(Uint32Array.of(1, 1, 0, 1), Uint32Array.of(5, 2, 9, 2))]).toEqual([2, 1, 3, 0]);
    });
});

describe('the H-PLOC shader', () => {
    it('uses no reserved word as a name, and no function shares a name with a variable', () => {
        for (const slots of [1, 2, 4, 8]) expect(wgslNameProblems(hplocWgsl(slots)), `${slots} slots`).toEqual([]);
    });

    it('enables subgroups and holds 32 clusters per subgroup for any subgroup size', () => {
        expect(hplocWgsl(1).startsWith('enable subgroups;')).toBe(true);
        for (const size of [4, 8, 16, 32, 64, 128]) expect(size * hplocSlotsFor(size), `subgroups of ${size}`).toBeGreaterThanOrEqual(HPLOC_MAX_CLUSTERS);
        expect(hplocSlotsFor(32)).toBe(1);
        expect(hplocSlotsFor(8)).toBe(4);
    });

    it('packs cluster numbers and levels into one word', () => {
        // Cluster numbers go up to twice the triangles, above the level bits; levels stay below the "unwritten" pattern.
        expect(2 * HPLOC_MAX_MESH_TRIANGLES * 2 ** 7).toBeLessThanOrEqual(2 ** 32);
        expect(HPLOC_MAX_LEVEL).toBeLessThan(0x7f);
    });
});
