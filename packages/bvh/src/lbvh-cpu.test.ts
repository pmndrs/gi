import { describe, expect, it } from 'vitest';
import { checkGpuLayout, treeSahCost } from './layout-check.ts';
import { buildLbvhOnCpu, mortonCode, mortonCodes, stableSortByKey, triangleBoxes } from './lbvh-cpu.ts';
import { createTinyBvhBuilder } from './tinybvh.ts';
import type { TinyBvhInput } from './tinybvh.ts';

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

/** A flat grid of `size` x `size` quads in the xz plane, indexed (shared vertices). */
function grid(size: number): TinyBvhInput {
    const positions = new Float32Array((size + 1) * (size + 1) * 3);
    for (let z = 0; z <= size; z++) {
        for (let x = 0; x <= size; x++) positions.set([x, 0, z], (z * (size + 1) + x) * 3);
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

describe('LBVH on the CPU (the GPU builder written plainly)', () => {
    it('spreads Morton bits as HIPRT does', () => {
        expect(mortonCode(0, 0, 0)).toBe(0);
        expect(mortonCode(1, 0, 0)).toBe(4);
        expect(mortonCode(0, 1, 0)).toBe(2);
        expect(mortonCode(0, 0, 1)).toBe(1);
        expect(mortonCode(1023, 1023, 1023)).toBe(2 ** 30 - 1);
    });

    it('keeps equal codes in triangle order', () => {
        const sorted = stableSortByKey(Uint32Array.of(5, 3, 5, 3, 0));
        expect([...sorted.values]).toEqual([4, 1, 3, 0, 2]);
        expect([...sorted.keys]).toEqual([0, 3, 3, 5, 5]);
    });

    it('builds valid trees: edge cases, soups, a grid, duplicates, flat meshes', () => {
        const cases: [string, TinyBvhInput][] = [
            ['one triangle', { positions: randomSoup(1, 1) }],
            ['two triangles', { positions: randomSoup(2, 2) }],
            ['three triangles', { positions: randomSoup(3, 3) }],
            ['random soup', { positions: randomSoup(3000, 4) }],
            ['grid', grid(40)],
            // Every triangle the same: every Morton code is equal, so the tree comes from the position tie-break alone.
            ['duplicates', { positions: Float32Array.from({ length: 500 * 9 }, (_, index) => [0, 0, 0, 1, 0, 0, 0, 1, 0][index % 9]) }],
            ['with flags', { positions: randomSoup(100, 5), triangleFlags: Uint32Array.from({ length: 100 }, (_, index) => index % 3) }],
        ];
        for (const [name, geometry] of cases) {
            for (const maxLeafPrimitives of [1, 4, 8]) {
                const layout = buildLbvhOnCpu(geometry, { maxLeafPrimitives });
                expect(checkGpuLayout(layout, geometry, { maxLeafPrimitives }), `${name}, leaves of ${maxLeafPrimitives}`).toEqual([]);
            }
        }
    });

    it('makes one-triangle leaves when leaves of one are asked for', () => {
        const geometry = { positions: randomSoup(200, 6) };
        const layout = buildLbvhOnCpu(geometry, { maxLeafPrimitives: 1 });
        expect(layout.leafCount).toBe(200);
        expect(layout.nodeCount).toBe(199);
    });

    it('collapses: fewer nodes and a lower SAH cost than one-triangle leaves', () => {
        const geometry = { positions: randomSoup(5000, 7) };
        const single = buildLbvhOnCpu(geometry, { maxLeafPrimitives: 1 });
        const collapsed = buildLbvhOnCpu(geometry);
        expect(collapsed.nodeCount).toBeLessThan(single.nodeCount);
        expect(treeSahCost(collapsed)).toBeLessThan(treeSahCost(single));
    });

    it('lands near tinybvh in SAH cost (LBVH trees are a little worse)', async () => {
        const tinyBvh = await createTinyBvhBuilder();
        for (const geometry of [{ positions: randomSoup(20000, 8) }, grid(100)]) {
            const ours = treeSahCost(buildLbvhOnCpu(geometry));
            const reference = treeSahCost(tinyBvh.buildLayout(geometry).layout);
            expect(ours / reference).toBeGreaterThan(0.95);
            expect(ours / reference).toBeLessThan(1.3);
        }
    });

    it('computes Morton codes over the centroid box', () => {
        const { boxes, centroidBounds } = triangleBoxes({ positions: Float32Array.of(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2) });
        expect([...centroidBounds]).toEqual([0, 0, 0, 2, 2, 2]);
        expect([...mortonCodes(boxes, centroidBounds)]).toEqual([0, 2 ** 30 - 1]);
        // Cubic cells: a box twice as long as it is high gets half the cells across its height.
        const flat = triangleBoxes({ positions: Float32Array.of(0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0) });
        expect([...mortonCodes(flat.boxes, flat.centroidBounds)]).toEqual([0, mortonCode(1023, 512, 0)]);
    });
});
