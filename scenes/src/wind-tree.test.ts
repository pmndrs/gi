import { describe, expect, it } from 'vitest';
import { generateWindTree, triangleCount, writeWindTreeGlb } from '../../tools/assets/convert/wind-tree.ts';

describe('the wind tree generator', () => {
    const tree = generateWindTree(20261109);

    it('makes 20,000 to 100,000 triangles in three meshes: trunk, branches, leaves', () => {
        expect(triangleCount(tree)).toBeGreaterThanOrEqual(20_000);
        expect(triangleCount(tree)).toBeLessThanOrEqual(100_000);
        expect(tree.meshes.map((mesh) => mesh.name)).toEqual(['trunk', 'branches', 'leaves']);
    });

    it('stands on the ground, 8 to 12 m tall', () => {
        const heights = tree.meshes.flatMap((mesh) => mesh.positions.filter((_, i) => i % 3 === 1));
        expect(Math.min(...heights)).toBeGreaterThan(-0.4);
        expect(Math.max(...heights)).toBeGreaterThan(8);
        expect(Math.max(...heights)).toBeLessThan(12);
    });

    it('has wind weights from 0 at the foot, up to 1 on leaves, with leaves above every tube', () => {
        const weights = (name: string) => tree.meshes.find((mesh) => mesh.name === name)!.colors.filter((_, i) => i % 4 === 3);
        const tubes = [...weights('trunk'), ...weights('branches')];
        const leaves = weights('leaves');
        expect(Math.min(...tubes)).toBe(0);
        expect(Math.max(...tubes)).toBeLessThanOrEqual(0.6);
        expect(Math.min(...leaves)).toBeGreaterThanOrEqual(0.7);
        expect(Math.max(...leaves)).toBeLessThanOrEqual(1);
    });

    it('writes the same bytes for the same seed, and different bytes for another', () => {
        const same = Buffer.from(writeWindTreeGlb(generateWindTree(20261109)));
        expect(same.equals(Buffer.from(writeWindTreeGlb(tree)))).toBe(true);
        expect(same.equals(Buffer.from(writeWindTreeGlb(generateWindTree(1))))).toBe(false);
    });
});
