import { describe, expect, it } from 'vitest';
import { cornerSkyVisibility, polygonViewFactor } from './sky-view.ts';

describe('the view factor behind the RC6.1 sky check', () => {
    it('is 1 for a huge square right above a point, facing it', () => {
        const side = 1e6;
        const square: [number, number, number][] = [[-side, 1, -side], [side, 1, -side], [side, 1, side], [-side, 1, side]];
        expect(polygonViewFactor([0, 0, 0], [0, 1, 0], square)).toBeCloseTo(1, 5);
    });

    it('matches the closed form for a long vertical wall seen from the ground', () => {
        // An infinitely long wall of height H at distance d: F = (1 - d / sqrt(d² + H²)) / 2.
        const height = 10;
        const halfLength = 1e6;
        const wall: [number, number, number][] = [[-halfLength, 0, 0], [halfLength, 0, 0], [halfLength, height, 0], [-halfLength, height, 0]];
        for (const distance of [0.5, 3, 10, 40]) {
            const expected = (1 - distance / Math.hypot(distance, height)) / 2;
            expect(polygonViewFactor([0, 0, distance], [0, 1, 0], wall)).toBeCloseTo(expected, 6);
        }
    });

    it('gives the corner the shares expected far from the wall and on it', () => {
        // Far out on the floor the 10 m wall hides little of the sky.
        expect(cornerSkyVisibility([0, 0, 100], 'floor')).toBeGreaterThan(0.995);
        // At the wall's foot it hides nearly half (its 40 m width leaves the sides open).
        expect(cornerSkyVisibility([0, 0, 0.01], 'floor')).toBeCloseTo(0.5, 2);
        // The floor hides most of the lower half of the wall's hemisphere; seen from 5 m up, its
        // edge 200 m out leaves 1.4% more of the sky (a 2-million-ray Monte Carlo gave 0.5141).
        expect(cornerSkyVisibility([0, 5, 0], 'wall')).toBeCloseTo(0.5142, 3);
        expect(cornerSkyVisibility([0, 0.5, 0], 'wall')).toBeCloseTo(0.5014, 3);
    });
});
