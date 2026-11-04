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
