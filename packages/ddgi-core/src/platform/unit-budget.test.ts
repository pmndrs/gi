import { describe, expect, it } from 'vitest';
import { DEFAULT_UNITS_PER_FRAME, limitUnitsDrop, MAX_UNITS_DROP_PER_FRAME, RECENT_FRAMES, UnitCostMeter, unitsForFrame, type UnitCostModel } from './gi-platform.ts';

const proportional = (msPerUnit: number): UnitCostModel => ({ fixedMs: 0, msPerUnit });

describe('turning a ms budget into units (T1.5b, fixed part T1.9a)', () => {
    it('divides gpuMs by the cost per unit when there is no fixed part', () => {
        // DDGI on the RTX 3060, Cornell: about 4.85 µs per probe.
        expect(unitsForFrame({ gpuMs: 2 }, [proportional(0.00485)])).toEqual([412]);
    });

    it('takes the fixed part off the budget first', () => {
        // Sponza-like: 1.2 ms of latency plus 5 µs per probe leaves 0.8 ms for 160 probes.
        expect(unitsForFrame({ gpuMs: 2 }, [{ fixedMs: 1.2, msPerUnit: 0.005 }])).toEqual([160]);
    });

    it('gives one unit when the fixed part alone is over the budget, so the technique keeps working and being measured', () => {
        expect(unitsForFrame({ gpuMs: 1 }, [{ fixedMs: 1.5, msPerUnit: 0.005 }])).toEqual([1]);
    });

    it('shares the time evenly between techniques', () => {
        expect(unitsForFrame({ gpuMs: 2 }, [proportional(0.01), proportional(0.001)])).toEqual([100, 1000]);
    });

    it('caps by units when both are given', () => {
        expect(unitsForFrame({ gpuMs: 2, units: 300 }, [proportional(0.00485)])).toEqual([300]);
    });

    it('falls back to units, then the default, without a cost for every technique', () => {
        expect(unitsForFrame({ gpuMs: 2 }, [undefined])).toEqual([DEFAULT_UNITS_PER_FRAME]);
        expect(unitsForFrame({ gpuMs: 2, units: 64 }, [proportional(0.01), undefined])).toEqual([64, 64]);
        expect(unitsForFrame({ units: 64 }, [proportional(0.01)])).toEqual([64]);
        expect(unitsForFrame({}, [proportional(0.01)])).toEqual([DEFAULT_UNITS_PER_FRAME]);
    });

    it('gives nothing for a zero budget', () => {
        expect(unitsForFrame({ gpuMs: 0 }, [proportional(0.01)])).toEqual([0]);
    });
});

describe('the drop limit (T1.9a)', () => {
    it('lets a millisecond budget shrink the units by at most half a frame', () => {
        expect(MAX_UNITS_DROP_PER_FRAME).toBe(0.5);
        expect(limitUnitsDrop(1, 64)).toBe(32);
        expect(limitUnitsDrop(80, 64)).toBe(80);
    });

    it('does not apply after a frame without a millisecond budget, or to a zero budget', () => {
        // The warm-up's big batches are budgeted in units: the first millisecond frame may drop at once.
        expect(limitUnitsDrop(200, undefined)).toBe(200);
        expect(limitUnitsDrop(0, 64)).toBe(0);
    });
});

describe('the unit cost meter', () => {
    it("measures a frame's technique passes against the units used that frame", () => {
        const meter = new UnitCostMeter();
        meter.handOut(7, 'ddgi', 512);
        meter.measure(7, [
            { name: 'ddgi/trace', gpuMs: 1.0 },
            { name: 'ddgi/blend', gpuMs: 0.5 },
            { name: 'ddgi/write', gpuMs: 0.036 },
            { name: 'trace/tlas-refit', gpuMs: 0.3 },
        ]);
        // One batch size only: no spread to fit a slope from, so the model is proportional.
        expect(meter.model('ddgi')).toEqual({ fixedMs: 0, msPerUnit: 1.536 / 512 });
        expect(meter.model('voxel')).toBeUndefined();
    });

    it('takes the median of recent frames and ignores frames it handed nothing out for', () => {
        const meter = new UnitCostMeter();
        let frame = 0;
        for (const ms of [1, 1.2, 0.9]) {
            meter.handOut(++frame, 'ddgi', 100);
            meter.measure(frame, [{ name: 'ddgi/trace', gpuMs: ms }]);
        }
        meter.measure(99, [{ name: 'ddgi/trace', gpuMs: 100 }]);
        expect(meter.model('ddgi')?.msPerUnit).toBeCloseTo(0.01, 12);
    });

    it('skips frames with no units or no timed passes', () => {
        const meter = new UnitCostMeter();
        meter.handOut(1, 'ddgi', 0);
        meter.measure(1, [{ name: 'ddgi/trace', gpuMs: 1 }]);
        meter.handOut(2, 'ddgi', 10);
        meter.measure(2, []);
        expect(meter.model('ddgi')).toBeUndefined();
    });

    it('fits a fixed part once batch sizes are spread out', () => {
        const meter = new UnitCostMeter();
        const cost = (units: number) => 1.2 + 0.005 * units;
        let frame = 0;
        // A warm-up of big batches, then small ones, as the viewer runs it.
        for (const units of [4096, 4096, 4096, 4096, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]) {
            meter.handOut(++frame, 't', units);
            meter.measure(frame, [{ name: 't/work', gpuMs: cost(units) }]);
        }
        const model = meter.model('t') as UnitCostModel;
        expect(model.msPerUnit).toBeCloseTo(0.005, 9);
        expect(model.fixedMs).toBeCloseTo(1.2, 6);
    });

    it('stays proportional at the batch in use when big batches cost more per unit (a convex curve)', () => {
        const meter = new UnitCostMeter();
        let frame = 0;
        // Intel UHD on Cornell: 1,728-probe warm-up batches at about 21 µs per probe, then ~96-probe batches at about 15.
        for (const [units, ms] of [[1728, 36], [1728, 36], [96, 1.4], [96, 1.4], [96, 1.4]] as const) {
            meter.handOut(++frame, 't', units);
            meter.measure(frame, [{ name: 't/work', gpuMs: ms }]);
        }
        const model = meter.model('t') as UnitCostModel;
        expect(model.fixedMs + model.msPerUnit * 96).toBeCloseTo(1.4, 6);
        expect(unitsForFrame({ gpuMs: 2 }, [model])).toEqual([137]);
    });

    it('ignores frames slowed by outside work, up to nearly half of them, but follows a lasting change', () => {
        const meter = new UnitCostMeter();
        let frame = 0;
        const add = (ms: number) => {
            meter.handOut(++frame, 't', 100);
            meter.measure(frame, [{ name: 't/work', gpuMs: ms }]);
        };
        // Every third frame 2.5 ms slower, as the RTX 3060 showed.
        for (let index = 0; index < 60; index++) add(index % 3 === 2 ? 3.5 : 1);
        expect(meter.model('t')?.msPerUnit).toBeCloseTo(0.01, 12);
        // A lasting change is followed once it fills more than half the window.
        for (let index = 0; index < Math.ceil(RECENT_FRAMES / 2); index++) add(4);
        expect(meter.model('t')?.msPerUnit).toBeCloseTo(0.04, 12);
    });

    it('forgets everything on reset', () => {
        const meter = new UnitCostMeter();
        meter.handOut(1, 't', 10);
        meter.measure(1, [{ name: 't/work', gpuMs: 1 }]);
        meter.reset();
        expect(meter.model('t')).toBeUndefined();
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
    function simulate(measureAgainst: 'used' | 'handed out'): { model: UnitCostModel | undefined; lastHandOut: number; largestHandOut: number } {
        const meter = new UnitCostMeter();
        const capacity = 100;
        const trueCost = 0.01;
        let lastHandOut = 0;
        let largestHandOut = 0;
        for (let frame = 1; frame <= 200; frame++) {
            const [units] = unitsForFrame({ gpuMs: 10, units: frame === 1 ? 512 : undefined }, [meter.model('t')]);
            lastHandOut = units;
            largestHandOut = Math.max(largestHandOut, units);
            const used = Math.min(units, capacity);
            meter.handOut(frame, 't', measureAgainst === 'used' ? used : units);
            meter.measure(frame, [{ name: 't/work', gpuMs: used * trueCost }]);
        }
        return { model: meter.model('t'), lastHandOut, largestHandOut };
    }

    it('keeps the cost per unit true and the hand-out bounded', () => {
        const result = simulate('used');
        expect(result.model?.msPerUnit).toBeCloseTo(0.01, 12);
        expect(result.lastHandOut).toBe(1000);
    });

    it('would run away if measured against the units handed out', () => {
        expect(simulate('handed out').largestHandOut).toBeGreaterThan(100_000);
    });
});

/**
 * The budget loop as the viewer runs it (T1.9a): a warm-up of big batches,
 * then `gpuMs: 2`, with timings arriving 3 frames late, a GPU that is
 * latency-bound below a full wave (flat, then linear), ±10% noise per frame,
 * and outside GPU work adding 2.5 ms to every third frame (what the RTX 3060
 * showed). Shaped like Sponza on the RTX: about 1.5 ms up to 128 probes,
 * then about 5.5 µs per probe.
 */
function simulateBudget(options: { spikes: boolean; seed?: number }): { units: number[]; ms: number[] } {
    const meter = new UnitCostMeter();
    const lag = 3;
    const truth = (units: number) => Math.max(1.5, 0.8 + 0.0055 * units);
    let state = options.seed ?? 12345;
    const random = () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
    const pending: Array<{ frame: number; ms: number }> = [];
    const units: number[] = [];
    const ms: number[] = [];
    for (let frame = 1; frame <= 600; frame++) {
        const budget = frame <= 20 ? { units: 4096 } : { gpuMs: 2 };
        const [handed] = unitsForFrame(budget, [meter.model('ddgi')]);
        meter.handOut(frame, 'ddgi', handed);
        const spike = options.spikes && frame % 3 === 0 ? 2.5 : 0;
        const measured = truth(handed) * (0.9 + 0.2 * random()) + spike;
        pending.push({ frame, ms: measured });
        while (pending.length > 0 && pending[0].frame <= frame - lag) {
            const arrived = pending.shift()!;
            meter.measure(arrived.frame, [{ name: 'ddgi/trace', gpuMs: arrived.ms }]);
        }
        if (frame > 20) {
            units.push(handed);
            ms.push(measured - spike);
        }
    }
    return { units, ms };
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
}

describe('the budget loop with a fixed cost, late timings and outside spikes (T1.9a)', () => {
    it('settles on the batch whose own time is the budget', () => {
        for (const spikes of [false, true]) {
            const { units, ms } = simulateBudget({ spikes });
            const settled = { units: units.slice(200), ms: ms.slice(200) };
            // truth(n) = 2 at n = 218.
            expect(median(settled.units)).toBeGreaterThan(190);
            expect(median(settled.units)).toBeLessThan(240);
            expect(Math.abs(median(settled.ms) - 2) / 2).toBeLessThan(0.1);
        }
    });

    it('comes down from the warm-up within a few frames, without ringing', () => {
        const { units } = simulateBudget({ spikes: true });
        // After the first 10 budget frames, never above twice the settled batch.
        expect(Math.max(...units.slice(10))).toBeLessThan(2 * 218);
    });
});
