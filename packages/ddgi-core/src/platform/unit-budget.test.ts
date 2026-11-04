import { describe, expect, it } from 'vitest';
import { DEFAULT_UNITS_PER_FRAME, MS_PER_UNIT_SMOOTHING, UnitCostMeter, unitsForFrame } from './gi-platform.ts';

describe('turning a ms budget into units (T1.5b)', () => {
    it('divides gpuMs by the cost per unit', () => {
        // DDGI on the RTX 3060, Cornell: about 4.85 µs per probe.
        expect(unitsForFrame({ gpuMs: 2 }, [0.00485])).toEqual([412]);
    });

    it('shares the time evenly between techniques', () => {
        expect(unitsForFrame({ gpuMs: 2 }, [0.01, 0.001])).toEqual([100, 1000]);
    });

    it('caps by units when both are given', () => {
        expect(unitsForFrame({ gpuMs: 2, units: 300 }, [0.00485])).toEqual([300]);
    });

    it('falls back to units, then the default, without a cost for every technique', () => {
        expect(unitsForFrame({ gpuMs: 2 }, [undefined])).toEqual([DEFAULT_UNITS_PER_FRAME]);
        expect(unitsForFrame({ gpuMs: 2, units: 64 }, [0.01, undefined])).toEqual([64, 64]);
        expect(unitsForFrame({ units: 64 }, [0.01])).toEqual([64]);
        expect(unitsForFrame({}, [0.01])).toEqual([DEFAULT_UNITS_PER_FRAME]);
    });

    it('gives nothing for a zero budget', () => {
        expect(unitsForFrame({ gpuMs: 0 }, [0.01])).toEqual([0]);
    });
});

describe('the unit cost meter', () => {
    it("divides a frame's technique passes by the units handed out that frame", () => {
        const meter = new UnitCostMeter();
        meter.handOut(7, 'ddgi', 512);
        meter.measure(7, [
            { name: 'ddgi/trace', gpuMs: 1.0 },
            { name: 'ddgi/blend', gpuMs: 0.5 },
            { name: 'ddgi/write', gpuMs: 0.036 },
            { name: 'trace/tlas-refit', gpuMs: 0.3 },
        ]);
        expect(meter.msPerUnit('ddgi')).toBeCloseTo(1.536 / 512, 12);
        expect(meter.msPerUnit('voxel')).toBeUndefined();
    });

    it('smooths over frames and ignores frames it handed nothing out for', () => {
        const meter = new UnitCostMeter();
        meter.handOut(1, 'ddgi', 100);
        meter.measure(1, [{ name: 'ddgi/trace', gpuMs: 1 }]);
        meter.handOut(2, 'ddgi', 100);
        meter.measure(2, [{ name: 'ddgi/trace', gpuMs: 2 }]);
        meter.measure(3, [{ name: 'ddgi/trace', gpuMs: 100 }]);
        expect(meter.msPerUnit('ddgi')).toBeCloseTo(0.01 + (0.02 - 0.01) * MS_PER_UNIT_SMOOTHING, 12);
    });

    it('skips frames with no units or no timed passes', () => {
        const meter = new UnitCostMeter();
        meter.handOut(1, 'ddgi', 0);
        meter.measure(1, [{ name: 'ddgi/trace', gpuMs: 1 }]);
        meter.handOut(2, 'ddgi', 10);
        meter.measure(2, []);
        expect(meter.msPerUnit('ddgi')).toBeUndefined();
    });
});

describe('a budget bigger than the work', () => {
    /**
     * A technique with 100 units of work at 0.01 ms each, given 10 ms: it can
     * only use 100 units (1 ms). Measured against the units used, the cost
     * stays 0.01 ms and the hand-out stays 1000. Measured against the units
     * handed out (the old way) the cost would fall every frame and the
     * hand-out grow without limit.
     */
    function simulate(measureAgainst: 'used' | 'handed out'): { msPerUnit: number | undefined; lastHandOut: number } {
        const meter = new UnitCostMeter();
        const capacity = 100;
        const trueCost = 0.01;
        let lastHandOut = 0;
        for (let frame = 1; frame <= 200; frame++) {
            const [units] = unitsForFrame({ gpuMs: 10, units: frame === 1 ? 512 : undefined }, [meter.msPerUnit('t')]);
            lastHandOut = units;
            const used = Math.min(units, capacity);
            meter.handOut(frame, 't', measureAgainst === 'used' ? used : units);
            meter.measure(frame, [{ name: 't/work', gpuMs: used * trueCost }]);
        }
        return { msPerUnit: meter.msPerUnit('t'), lastHandOut };
    }

    it('keeps the cost per unit true and the hand-out bounded', () => {
        const result = simulate('used');
        expect(result.msPerUnit).toBeCloseTo(0.01, 12);
        expect(result.lastHandOut).toBe(1000);
    });

    it('would run away if measured against the units handed out', () => {
        expect(simulate('handed out').lastHandOut).toBeGreaterThan(100_000);
    });

    it('forgets everything on reset', () => {
        const meter = new UnitCostMeter();
        meter.handOut(1, 't', 10);
        meter.measure(1, [{ name: 't/work', gpuMs: 1 }]);
        meter.reset();
        expect(meter.msPerUnit('t')).toBeUndefined();
    });
});
