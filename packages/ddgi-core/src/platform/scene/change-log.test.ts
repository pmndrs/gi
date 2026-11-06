import { describe, expect, it } from 'vitest';
import type { LightDescriptor, PointLightDescriptor } from '../contract/scene.ts';
import { influenceRadius, referenceIrradiance, sameLight } from './change-log.ts';

const lamp = (intensity: number, range = 0, color: [number, number, number] = [1, 1, 1]): PointLightDescriptor => ({ type: 'point', color, intensity, position: [0, 0, 0], range });
const sun = (intensity: number): LightDescriptor => ({ type: 'directional', color: [1, 0.9, 0.8], intensity, direction: [0, -1, 0] });

describe('the influence radius (m2-lights.md §1)', () => {
    it('uses the sun as the reference, or the brightest lamp at 1 m without one', () => {
        expect(referenceIrradiance([lamp(12), lamp(3)])).toBe(12);
        expect(referenceIrradiance([lamp(12), sun(2)])).toBe(2);
        // A black sun is no reference.
        expect(referenceIrradiance([lamp(12), sun(0)])).toBe(12);
        expect(referenceIrradiance([])).toBe(0);
        // The strongest colour channel counts.
        expect(referenceIrradiance([lamp(10, 0, [0.2, 0.5, 0.1])])).toBe(5);
    });

    it('ends where the unwindowed irradiance falls to 1/1000 of the reference, capped by the range', () => {
        // The only lamp: I / d^2 = I / 1000 at d = sqrt(1000).
        expect(influenceRadius(lamp(12), 12)).toBeCloseTo(Math.sqrt(1000), 10);
        // A lamp a quarter as bright reaches half as far.
        expect(influenceRadius(lamp(3), 12)).toBeCloseTo(Math.sqrt(1000) / 2, 10);
        // Two-room's lamp has range 14, under its 31.6 m.
        expect(influenceRadius(lamp(12, 14), 12)).toBe(14);
        expect(influenceRadius(lamp(0), 12)).toBe(0);
        expect(influenceRadius(null, 12)).toBe(0);
        expect(influenceRadius(sun(3), 3)).toBe(Infinity);
    });

    it('compares light states field by field', () => {
        expect(sameLight(lamp(1), lamp(1))).toBe(true);
        expect(sameLight(lamp(1), lamp(2))).toBe(false);
        expect(sameLight(lamp(1), { ...lamp(1), castsShadows: false })).toBe(false);
        expect(sameLight(lamp(1), { ...lamp(1), position: [0, 0, 1e-7] })).toBe(false);
    });
});
