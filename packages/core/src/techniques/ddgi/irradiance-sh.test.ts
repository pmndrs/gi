import { describe, expect, it } from 'vitest';
import { fillInactiveProbes, L2_SH_FLOATS_PER_PROBE } from './irradiance-sh.ts';
import { PROBE_STATE } from './layout.ts';

describe('fillInactiveProbes', () => {
    it('gives inactive probes their nearest active neighbours’ mean, ring by ring', () => {
        // A row of 5 probes: active at 0 (value 1) and 4 (value 3); 1-3 off.
        const counts: [number, number, number] = [5, 1, 1];
        const coefficients = new Float32Array(5 * L2_SH_FLOATS_PER_PROBE);
        const states = new Uint8Array([PROBE_STATE.active, PROBE_STATE.offInsideGeometry, PROBE_STATE.notUpdatedYet, PROBE_STATE.offNoSurfaceNearby, PROBE_STATE.active]);
        coefficients.fill(1, 0, L2_SH_FLOATS_PER_PROBE);
        coefficients.fill(99, L2_SH_FLOATS_PER_PROBE, 4 * L2_SH_FLOATS_PER_PROBE);
        coefficients.fill(3, 4 * L2_SH_FLOATS_PER_PROBE);
        expect(fillInactiveProbes(coefficients, states, counts)).toBe(3);
        const value = (probe: number): number => coefficients[probe * L2_SH_FLOATS_PER_PROBE + 5];
        // First ring: 1 from probe 0, 3 from probe 4; second ring: probe 2 from both (not from its own old 99).
        expect([value(0), value(1), value(2), value(3), value(4)]).toEqual([1, 1, 2, 3, 3]);
        // The states are left as they were (they say what was filled).
        expect(states[2]).toBe(PROBE_STATE.notUpdatedYet);
    });

    it('fills nothing when no probe is active', () => {
        const coefficients = new Float32Array(8 * L2_SH_FLOATS_PER_PROBE).fill(5);
        expect(fillInactiveProbes(coefficients, new Uint8Array(8).fill(PROBE_STATE.offInsideGeometry), [2, 2, 2])).toBe(0);
        expect(coefficients[0]).toBe(5);
    });
});
