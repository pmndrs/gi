import { describe, expect, it } from 'vitest';
import { mortonCode, selectionOrder } from './sponza-candles.ts';

/** A seeded cloud of points, denser on one side (as Intel's candles are denser in some parts of Sponza). */
function cloud(count: number, seed = 5): Float64Array {
    let state = seed;
    const random = () => (state = (state * 16807) % 2147483647) / 2147483647;
    const points = new Float64Array(count * 3);
    for (let index = 0; index < count; index++) {
        points[index * 3] = -15 + 26 * random() ** 1.5;
        points[index * 3 + 1] = 0.5 + 10.5 * random();
        points[index * 3 + 2] = -5.7 + 11.4 * random();
    }
    return points;
}

describe("sponza-candles' selection order (T2.1)", () => {
    const points = cloud(10_000);
    const order = selectionOrder(points);

    it('lists every candle exactly once, the same way every time', () => {
        expect(order).toHaveLength(10_000);
        expect(new Set(order).size).toBe(10_000);
        expect(selectionOrder(points)).toEqual(order);
    });

    it('spreads any first N through the scene in proportion to where the candles are', () => {
        // 4 x 2 x 2 cells over the box: each holds about its share of the first N, for several N.
        const cellOf = (index: number) => {
            const x = Math.min(3, Math.floor(((points[index * 3] + 15) / 26) * 4));
            const y = Math.min(1, Math.floor(((points[index * 3 + 1] - 0.5) / 10.5) * 2));
            const z = Math.min(1, Math.floor(((points[index * 3 + 2] + 5.7) / 11.4) * 2));
            return x + 4 * (y + 2 * z);
        };
        const all = new Array(16).fill(0);
        for (let index = 0; index < 10_000; index++) all[cellOf(index)]++;
        for (const taken of [256, 1024, 4096]) {
            const counts = new Array(16).fill(0);
            for (const index of order.slice(0, taken)) counts[cellOf(index)]++;
            for (let cell = 0; cell < 16; cell++) {
                const expected = (all[cell] / 10_000) * taken;
                // Within 25% of the share, or 4 candles for the emptiest cells: far tighter than a random pick.
                expect(Math.abs(counts[cell] - expected), `${taken} candles, cell ${cell}`).toBeLessThanOrEqual(Math.max(4, 0.25 * expected));
            }
        }
    });

    it('codes points along a Z-order curve: x in the lowest bit, then y, then z', () => {
        const min = [0, 0, 0];
        const max = [1, 1, 1];
        expect(mortonCode([0, 0, 0], min, max)).toBe(0);
        expect(mortonCode([0.5, 0, 0], min, max)).toBe(1 << 27);
        expect(mortonCode([0, 0.5, 0], min, max)).toBe(1 << 28);
        expect(mortonCode([0, 0, 0.5], min, max)).toBe(1 << 29);
        expect(mortonCode([1, 1, 1], min, max)).toBe(2 ** 30 - 1);
    });
});
