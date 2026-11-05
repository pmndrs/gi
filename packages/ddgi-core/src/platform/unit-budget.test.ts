import { describe, expect, it } from 'vitest';
import { BUCKET_MAX_AGE_FRAMES, DEFAULT_UNITS_PER_FRAME, RECENT_FRAMES, steerUnits, UNIT_STEERING, UnitCostMeter, unitsForFrame, type UnitCostModel } from './gi-platform.ts';

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

describe('steering the batch from frame to frame (T1.9a review)', () => {
    it('takes the model target on the first millisecond frame (after a warm-up in units)', () => {
        expect(steerUnits(133, undefined, 2)).toBe(133);
    });

    it('waits at the previous batch until enough frames near it are measured', () => {
        expect(steerUnits(133, { units: 60, nearMs: undefined }, 2)).toBe(60);
    });

    it('keeps the batch inside the dead band', () => {
        expect(steerUnits(133, { units: 60, nearMs: 2.08 }, 2)).toBe(60);
    });

    it('moves by the square root of budget / measured, within the shrink and growth limits', () => {
        // 2 / 2.5 = 0.8, its square root 0.894: 60 → 54.
        expect(steerUnits(1, { units: 60, nearMs: 2.5 }, 2)).toBe(54);
        // Far over the budget: at most maxShrink.
        expect(steerUnits(1, { units: 60, nearMs: 8 }, 2)).toBe(Math.floor(60 * UNIT_STEERING.maxShrink));
        // Far under: at most maxGrowth (+ 1, so small batches can grow).
        expect(steerUnits(1000, { units: 60, nearMs: 0.5 }, 2)).toBe(Math.ceil(60 * UNIT_STEERING.maxGrowth) + 1);
    });

    it('ignores the model target once steering (the line is wrong below the flat part)', () => {
        expect(steerUnits(1, { units: 25, nearMs: 2.0 }, 2)).toBe(25);
    });

    it('keeps a zero budget at zero', () => {
        expect(steerUnits(0, { units: 100, nearMs: 2 }, 0)).toBe(0);
    });

    it('applies the units cap last (FrameBudget: a hard cap when gpuMs is set)', () => {
        const model: UnitCostModel = { fixedMs: 1.2, msPerUnit: 0.006 };
        // After a 1,000-probe frame, steering alone would allow 800; the cap of 10 wins.
        expect(unitsForFrame({ gpuMs: 2, units: 10 }, [model], [{ units: 1000, nearMs: 9 }])).toEqual([10]);
    });

    it('measures near a batch from recent frames within the near share, once there are enough', () => {
        const meter = new UnitCostMeter();
        let frame = 0;
        for (const [units, ms] of [[60, 2], [62, 2.2], [400, 9], [58, 1.8], [61, 2.1]] as const) {
            meter.handOut(++frame, 't', units);
            meter.measure(frame, [{ name: 't/work', gpuMs: ms }]);
        }
        expect(meter.msNear('t', 60)).toBeUndefined();
        meter.handOut(++frame, 't', 59);
        meter.measure(frame, [{ name: 't/work', gpuMs: 3.5 }]);
        // Five frames between 48 and 75 probes: 1.8, 2, 2.1, 2.2, 3.5.
        expect(meter.msNear('t', 60)).toBe(2.1);
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

describe('aging out batch sizes (T1.9a review)', () => {
    it('drops a batch size not seen for BUCKET_MAX_AGE_FRAMES; the last fitted slope stays', () => {
        const meter = new UnitCostMeter();
        let frame = 0;
        const add = (units: number, ms: number) => {
            meter.handOut(++frame, 't', units);
            meter.measure(frame, [{ name: 't/work', gpuMs: ms }]);
        };
        for (let index = 0; index < 5; index++) add(4096, 1.2 + 0.005 * 4096);
        for (let index = 0; index < 40; index++) add(64, 1.2 + 0.005 * 64);
        expect(meter.model('t')?.fixedMs).toBeCloseTo(1.2, 6);
        expect(meter.batchSizesKnown('t')).toBe(2);
        frame += BUCKET_MAX_AGE_FRAMES;
        // The 4,096 bucket ages out. The new frames say the fixed part grew; the old slope is kept.
        for (let index = 0; index < 40; index++) add(64, 1.5 + 0.005 * 64);
        expect(meter.batchSizesKnown('t')).toBe(1);
        const model = meter.model('t') as UnitCostModel;
        expect(model.msPerUnit).toBeCloseTo(0.005, 9);
        expect(model.fixedMs).toBeCloseTo(1.5, 6);
    });

    it('drops every batch size when told the geometry changed, keeping the recent frames', () => {
        const meter = new UnitCostMeter();
        let frame = 0;
        const add = (units: number, ms: number) => {
            meter.handOut(++frame, 't', units);
            meter.measure(frame, [{ name: 't/work', gpuMs: ms }]);
        };
        add(4096, 21.68);
        add(64, 1.52);
        expect(meter.model('t')?.fixedMs).toBeGreaterThan(0);
        meter.forgetBatchSizes();
        expect(meter.batchSizesKnown('t')).toBe(0);
        add(64, 1.52);
        // No spread and no old slope: proportional, until new batch sizes give a line again.
        expect(meter.model('t')?.fixedMs).toBe(0);
    });
});

/**
 * The budget loop as the platform runs it (T1.9a and its review): a warm-up
 * of big batches in units, then `gpuMs: 2` with the previous batch fed back
 * for steering, timings arriving 3 frames late, ±10% noise per frame, outside
 * GPU work adding 2.5 ms to every third frame, and bursts in which it adds
 * 0.8 ms to the other frames too (what the RTX 3060 showed on Sponza).
 */
interface BudgetRun {
    units: number[];
    /** The technique's own time per frame (the outside work taken out). */
    ms: number[];
}

function simulateBudget(truth: (units: number) => number, options: { spikes: boolean; bursts?: boolean; frames?: number; seed?: number }): BudgetRun {
    const meter = new UnitCostMeter();
    const lag = 3;
    const warmupFrames = 20;
    let state = options.seed ?? 12345;
    const random = () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
    const pending: Array<{ frame: number; ms: number }> = [];
    const run: BudgetRun = { units: [], ms: [] };
    let previous: number | undefined;
    for (let frame = 1; frame <= (options.frames ?? 600); frame++) {
        const budget = frame <= warmupFrames ? { units: 4096 } : { gpuMs: 2 };
        // As the platform does it (gi-platform.ts, encode).
        const last = previous === undefined ? undefined : { units: previous, nearMs: meter.msNear('ddgi', previous) };
        const [handed] = unitsForFrame(budget, [meter.model('ddgi')], [last]);
        previous = budget.gpuMs === undefined ? undefined : handed;
        meter.handOut(frame, 'ddgi', handed);
        // Bursts of 1 s: the RTX showed them about once in 15 s.
        const inBurst = options.bursts === true && frame % 900 >= 300 && frame % 900 < 360;
        const outside = !options.spikes ? 0 : frame % 3 === 0 ? 2.5 : inBurst ? 0.8 : 0;
        const own = truth(handed) * (0.9 + 0.2 * random());
        pending.push({ frame, ms: own + outside });
        while (pending.length > 0 && pending[0].frame <= frame - lag) {
            const arrived = pending.shift()!;
            meter.measure(arrived.frame, [{ name: 'ddgi/trace', gpuMs: arrived.ms }]);
        }
        if (frame > warmupFrames) {
            run.units.push(handed);
            run.ms.push(own);
        }
    }
    return run;
}

function quantile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
}

const median = (values: readonly number[]) => quantile(values, 0.5);

/** Flat, then linear: about 1.5 ms up to 128 probes, then 5.5 µs per probe. truth(n) = 2 at n = 218. */
const flatThenLinear = (units: number) => Math.max(1.5, 0.8 + 0.0055 * units);
/**
 * Sponza at 60 Hz on the RTX: a single probe costs 0.2-0.3 ms, the time
 * climbs to about 1.7 ms by 32 probes, then about 6 µs per probe.
 * truth(n) = 2 at about 60 probes.
 */
const sponzaLike = (units: number) => 0.2 + 1.4 * (1 - Math.exp(-units / 12)) + 0.006 * units;

describe('the budget loop with a fixed cost, late timings and outside spikes (T1.9a)', () => {
    it('settles on the batch whose own time is the budget', () => {
        for (const spikes of [false, true]) {
            const { units, ms } = simulateBudget(flatThenLinear, { spikes });
            const settled = { units: units.slice(200), ms: ms.slice(200) };
            // The dead band (±5% of the budget) leaves it anywhere from about 180 to 240.
            expect(median(settled.units)).toBeGreaterThan(180);
            expect(median(settled.units)).toBeLessThan(240);
            expect(Math.abs(median(settled.ms) - 2) / 2).toBeLessThan(0.1);
        }
    });

    it('comes down from the warm-up within a few frames, without ringing', () => {
        const { units } = simulateBudget(flatThenLinear, { spikes: true });
        expect(Math.max(...units.slice(10))).toBeLessThan(2 * 218);
    });

    it('holds a steady batch on a steady Sponza-like scene, past the buckets aging out', () => {
        const { units, ms } = simulateBudget(sponzaLike, { spikes: true, frames: 3000 });
        const settled = { units: units.slice(200), ms: ms.slice(200) };
        const typical = median(settled.units);
        expect(Math.abs(median(settled.ms) - 2) / 2).toBeLessThan(0.1);
        expect(quantile(settled.units, 0.05)).toBeGreaterThanOrEqual(typical / 2);
        expect(quantile(settled.units, 0.95)).toBeLessThanOrEqual(typical * 2);
        expect(Math.min(...settled.units)).toBeGreaterThanOrEqual(typical / 2);
    });

    it('shrinks during a 1 s burst of outside work and recovers within 3 s', () => {
        const { units, ms } = simulateBudget(sponzaLike, { spikes: true, bursts: true, frames: 3000 });
        const settled = { units: units.slice(200), ms: ms.slice(200) };
        const typical = median(settled.units);
        expect(Math.abs(median(settled.ms) - 2) / 2).toBeLessThan(0.1);
        // Bursts cover simulated frames 300-359 of every 900 (run index = frame - 21).
        for (const burstStart of [300, 1200, 2100]) {
            const afterRecovery = units.slice(burstStart + 60 + 180 - 21, burstStart + 900 - 21 - 300);
            expect(Math.min(...afterRecovery)).toBeGreaterThanOrEqual(typical / 2);
        }
        expect(Math.min(...settled.units)).toBeGreaterThanOrEqual(1);
    });
});
