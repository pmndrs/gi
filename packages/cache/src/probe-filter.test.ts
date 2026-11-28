/**
 * The CPU port of `FilterScreenProbes` (capsaicin-math.ts, RC4.2) on
 * synthetic screens: a uniform field comes through unchanged, a depth edge
 * is not blurred across, a backup fill stays a fill, and NaN neighbours add
 * nothing. The GPU kernel is checked against this port by the
 * `radiance-cache-near-field` page.
 */

import { describe, expect, it } from 'vitest';
import { filterScreenProbesCell, INVALID_ID, packSeed, type Direction, type ProbeFilterInputs } from './capsaicin-math.ts';
import { PROBE_SIZE } from './settings.ts';

const SIZE = 128;
const TILES = SIZE / PROBE_SIZE;
/** Metres per pixel on the planes, seen straight on from the origin. */
const PIXEL = 0.01;

/**
 * A screen of planes facing the camera (normal +z, the camera looking down
 * −z): depth −5 m, or `nearDepth` right of `edgeX`; a probe at every tile's
 * centre; the atlas from `atlas`.
 */
function screen(atlas: ProbeFilterInputs['atlas'], edgeX = SIZE, nearDepth = -3): ProbeFilterInputs {
    return {
        width: SIZE,
        height: SIZE,
        cellSize: 0.02,
        eye: [0, 0, 0],
        viewForward: [0, 0, -1],
        probeMaskMipCount: 1,
        worldPosition: (x, y): Direction => [(x - SIZE / 2) * PIXEL, (SIZE / 2 - y) * PIXEL, x >= edgeX ? nearDepth : -5],
        normal: (): Direction => [0, 0, 1],
        probeMask: (level, x, y) => (level === 0 && x < TILES && y < TILES ? packSeed(x * PROBE_SIZE + 4, y * PROBE_SIZE + 4) : INVALID_ID),
        atlas,
    };
}

describe('FilterScreenProbes on the CPU (gi1.comp:1356-1442)', () => {
    it('leaves a uniform field unchanged, in both directions', () => {
        const inputs = screen(() => [2, 2, 2, 3]);
        for (const direction of [
            [1, 0],
            [0, 1],
        ] as const) {
            for (const cell of [0, 9, 27, 36, 63]) {
                const { value, taps } = filterScreenProbesCell(inputs, [60, 60], cell, direction);
                expect(taps).toBeGreaterThan(0);
                for (let channel = 0; channel < 3; channel++) expect(value[channel]).toBeCloseTo(2, 6);
            }
        }
    });

    it('averages neighbours on the same plane', () => {
        // Brightness by tile column, squared (so the sides' mean is not the centre's): a probe takes
        // its neighbours across, but not down.
        const inputs = screen((x) => [Math.floor(x / PROBE_SIZE) ** 2, 0, 0, 3]);
        const across = filterScreenProbesCell(inputs, [60, 60], 36, [1, 0]);
        const down = filterScreenProbesCell(inputs, [60, 60], 36, [0, 1]);
        expect(across.taps).toBeGreaterThan(0);
        expect(across.value[0]).toBeGreaterThan(49.5);
        expect(down.value[0]).toBeCloseTo(49, 6);
    });

    it("doesn't blur across a depth edge", () => {
        // Left of pixel 64 at 5 m and dark, right of it at 3 m and bright.
        const inputs = screen((x) => (x >= 64 ? [10, 10, 10, 3] : [1, 1, 1, 3]), 64);
        const result = filterScreenProbesCell(inputs, [60, 60], 36, [1, 0]);
        expect(result.planeRefused).toBeGreaterThan(0);
        expect(result.value[0]).toBeCloseTo(1, 6);
        // The same without the edge mixes the two.
        const flat = filterScreenProbesCell(screen((x) => (x >= 64 ? [10, 10, 10, 3] : [1, 1, 1, 3])), [60, 60], 36, [1, 0]);
        expect(flat.value[0]).toBeGreaterThan(1.5);
    });

    it('keeps a backup fill a fill, and skips neighbour fills and NaNs (D49, D50)', () => {
        const fillCentre = screen((x) => (Math.floor(x / PROBE_SIZE) === 7 ? [4, 4, 4, -1] : [2, 2, 2, 3]));
        const centre = filterScreenProbesCell(fillCentre, [60, 60], 36, [1, 0]);
        expect(centre.value[3]).toBe(-1);
        expect(centre.taps).toBeGreaterThan(0);
        expect(centre.value[0]).toBeLessThan(4);
        const fillNeighbours = screen((x) => (Math.floor(x / PROBE_SIZE) === 7 ? [2, 2, 2, 3] : [50, 50, 50, -1]));
        expect(filterScreenProbesCell(fillNeighbours, [60, 60], 36, [1, 0]).value[0]).toBeCloseTo(2, 6);
        const nan = screen((x) => (Math.floor(x / PROBE_SIZE) === 7 ? [2, 2, 2, 3] : [Number.NaN, Infinity, -1, 3]));
        const cleaned = filterScreenProbesCell(nan, [60, 60], 36, [1, 0]);
        expect(cleaned.taps).toBeGreaterThan(0);
        expect(cleaned.value.every(Number.isFinite)).toBe(true);
        expect(cleaned.value[0]).toBeLessThan(2);
    });
});
