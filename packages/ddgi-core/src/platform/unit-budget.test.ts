import { describe, expect, it } from 'vitest';
import type { FrameBudget } from './contract/frame.ts';
import {
    CALIBRATION,
    CalibrationBurst,
    DEFAULT_UNITS_PER_FRAME,
    GEOMETRY_RECALIBRATION,
    SATURATION,
    SaturationGuard,
    UnitCostMeter,
    WorkBudget,
    workFloorOf,
    type BudgetReport,
    type TechniqueWorkFloor,
} from './work-budget.ts';
import { FLOOR_MAX_CALIBRATED_SHARE, FLOOR_SWEEP_SECONDS, minProbesPerFrame } from '../techniques/ddgi/settings.ts';

/** Sponza on the RTX 3060 back to back: latency-bound below about 30 probes, then about 6 µs a probe. */
const sponzaLike = (units: number) => 0.2 + 1.0 * (1 - Math.exp(-units / 12)) + 0.006 * units;
/** A straight line, for checking the fit: 1.2 ms + 5 µs a probe. */
const line = (units: number) => 1.2 + 0.005 * units;
/** Cornell on the RTX: 0.15 ms + 1.4 µs a probe. */
const cornellLike = (units: number) => 0.15 + 0.0014 * units;
/** The Intel UHD on Sponza, back to back (design doc): 16 probes 2.1 ms, 40 3.0, 81 4.8, 160 6.7; then about 30 µs a probe. */
const INTEL_SPONZA_POINTS = [
    [0, 1.2],
    [16, 2.1],
    [40, 3.0],
    [81, 4.8],
    [160, 6.7],
] as const;
function intelSponzaLike(units: number): number {
    for (let index = 1; index < INTEL_SPONZA_POINTS.length; index++) {
        const [x0, y0] = INTEL_SPONZA_POINTS[index - 1];
        const [x1, y1] = INTEL_SPONZA_POINTS[index];
        if (units <= x1) return y0 + ((y1 - y0) * (units - x0)) / (x1 - x0);
    }
    return 6.7 + 0.03 * (units - 160);
}

/** DDGI's floor on a volume of `probeCount` probes, the way the technique gives it to the platform. */
function ddgiFloor(probeCount: number, setting: number | 'auto' = 'auto', maxProbesPerFrame = 4096): TechniqueWorkFloor {
    return { minUnitsPerFrame: (rate) => minProbesPerFrame(setting, probeCount, maxProbesPerFrame, rate) };
}

interface SimulationOptions {
    readonly frames: number;
    readonly budget: (frame: number) => FrameBudget;
    /** The technique's true GPU ms for a batch, at full clocks. */
    readonly cost: (units: number) => number;
    /** Seconds between frames as the host reports them (0: back to back). */
    readonly deltaTime?: (frame: number) => number;
    /** Multiplies the true cost: 2 for a GPU in its low-power state at a light load. */
    readonly clockFactor?: (frame: number) => number;
    /** The most units the technique can use (all its probes, or `maxProbesPerFrame`). */
    readonly capacity?: number;
    readonly floor?: TechniqueWorkFloor;
    /** Frames whose scene upload had a geometry change. */
    readonly geometryChange?: (frame: number) => boolean;
    readonly timed?: boolean;
    /** Frames until a frame's timings arrive (the platform's readback lag). */
    readonly lag?: number;
    readonly seed?: number;
    /** Called before each frame, for requests such as `recalibrate`. */
    readonly beforeFrame?: (frame: number, budget: WorkBudget) => void;
}

interface Simulation {
    readonly handed: number[];
    readonly used: number[];
    readonly reports: BudgetReport[];
    readonly budget: WorkBudget;
}

/**
 * The platform's loop for one technique (gi-platform.ts, encode): the work
 * budget's hand-out, the technique using at most `capacity`, the meter
 * pairing timings with the units used, timings arriving `lag` frames late
 * with ±5% noise.
 */
function simulate(options: SimulationOptions): Simulation {
    const budget = new WorkBudget(options.floor);
    const meter = new UnitCostMeter();
    const lag = options.lag ?? 3;
    let state = options.seed ?? 12345;
    const noise = () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return 0.95 + 0.1 * (state / 2 ** 32);
    };
    const pending: Array<{ frame: number; ms: number }> = [];
    const run: Simulation = { handed: [], used: [], reports: [], budget };
    for (let frame = 1; frame <= options.frames; frame++) {
        options.beforeFrame?.(frame, budget);
        if (options.geometryChange?.(frame)) budget.geometryChanged();
        const frameBudget = options.budget(frame);
        const shareMs = frameBudget.gpuMs;
        const handed = budget.unitsFor({ frame, deltaTime: options.deltaTime?.(frame) ?? 1 / 60, budget: frameBudget, shareMs, timed: options.timed ?? true });
        const used = Math.min(handed, options.capacity ?? Number.POSITIVE_INFINITY);
        meter.handOut(frame, 'ddgi', used);
        budget.record(handed, used);
        const ms = used === 0 ? 0 : options.cost(used) * (options.clockFactor?.(frame) ?? 1) * noise();
        pending.push({ frame, ms });
        while (pending.length > 0 && pending[0].frame <= frame - lag) {
            const arrived = pending.shift() as { frame: number; ms: number };
            for (const { sample } of meter.measure(arrived.frame, [{ name: 'ddgi/trace', gpuMs: arrived.ms }])) budget.addSample(sample);
        }
        run.handed.push(handed);
        run.used.push(used);
        run.reports.push(budget.report);
    }
    return run;
}

/** The first frame (1-based) whose report is calibrated, or -1. */
function firstCalibratedFrame(run: Simulation): number {
    return run.reports.findIndex((report) => report.state === 'calibrated') + 1 || -1;
}

const unique = (values: readonly number[]) => [...new Set(values)];

describe('the unit cost meter', () => {
    it("measures a frame's technique passes against the units used that frame", () => {
        const meter = new UnitCostMeter();
        meter.handOut(7, 'ddgi', 512);
        const measured = meter.measure(7, [
            { name: 'ddgi/trace', gpuMs: 1.0 },
            { name: 'ddgi/blend', gpuMs: 0.5 },
            { name: 'ddgi/write', gpuMs: 0.036 },
            { name: 'trace/tlas-refit', gpuMs: 0.3 },
        ]);
        expect(measured).toEqual([{ techniqueId: 'ddgi', sample: { frame: 7, units: 512, ms: 1.536 } }]);
    });

    it('skips frames it handed nothing out for, frames with no units and frames with no timed passes', () => {
        const meter = new UnitCostMeter();
        expect(meter.measure(99, [{ name: 'ddgi/trace', gpuMs: 100 }])).toEqual([]);
        meter.handOut(1, 'ddgi', 0);
        expect(meter.measure(1, [{ name: 'ddgi/trace', gpuMs: 1 }])).toEqual([]);
        meter.handOut(2, 'ddgi', 10);
        expect(meter.measure(2, [])).toEqual([]);
    });
});

describe('calibration from back-to-back data (budget v2)', () => {
    it('finds the batch whose cost is the budget on a straight line, and reports the line', () => {
        // 2 ms on 1.2 ms + 5 µs a probe: 160 probes.
        const run = simulate({ frames: 200, budget: () => ({ gpuMs: 2 }), cost: line, deltaTime: () => 0 });
        const calibration = run.budget.report.calibration;
        expect(calibration?.measured).toBe(true);
        expect(Math.abs((calibration?.units ?? 0) - 160)).toBeLessThanOrEqual(8);
        // The reported line, fitted through medians of timings with ±5% noise.
        expect(Math.abs((calibration?.model?.fixedMs ?? 0) - 1.2)).toBeLessThan(0.15);
        expect(Math.abs((calibration?.model?.msPerUnit ?? 0) * 1000 - 5)).toBeLessThan(0.5);
        // Back to back: the host gave no time between frames.
        expect(calibration?.frameSeconds).toBe(0);
    });

    it('lands within 10% of the budget on a curve that bends (Sponza on the RTX), in about a second', () => {
        for (const gpuMs of [1, 2, 4]) {
            const run = simulate({ frames: 300, budget: () => ({ gpuMs }), cost: sponzaLike, deltaTime: () => 1 / 60 });
            const units = run.budget.report.calibratedUnits ?? 0;
            expect(sponzaLike(units)).toBeLessThanOrEqual(gpuMs * 1.1);
            // Not far under either: the next probes up would have fit.
            expect(sponzaLike(units * 1.15 + 1)).toBeGreaterThan(gpuMs * 0.95);
            expect(firstCalibratedFrame(run)).toBeLessThan(90);
        }
    });

    it('lands within 10% on the Intel-like curve, where big batches cost more per probe', () => {
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 2 }), cost: intelSponzaLike, deltaTime: () => 1 / 60 });
        const units = run.budget.report.calibratedUnits ?? 0;
        expect(units).toBeGreaterThan(10);
        expect(intelSponzaLike(units)).toBeLessThanOrEqual(2.2);
    });

    it('gives every probe when all of them fit (Cornell at 8 ms)', () => {
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 8 }), cost: cornellLike, capacity: 1000 });
        expect(run.budget.report.calibratedUnits).toBe(1000);
        // Never hands out far past what the technique could use.
        expect(Math.max(...run.handed)).toBeLessThanOrEqual(CALIBRATION.firstUnits * CALIBRATION.ladderStep);
    });

    it("doesn't count the ramp frames, which ran at low clocks", () => {
        // The first frames of the burst take 3× as long (the GPU waking up); after them, full clocks.
        const rampedFrames = CALIBRATION.rampFrames;
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 2 }), cost: line, clockFactor: (frame) => (frame <= rampedFrames ? 3 : 1) });
        expect(Math.abs((run.budget.report.calibratedUnits ?? 0) - 160)).toBeLessThanOrEqual(8);
    });

    it('keeps the GPU loaded between small batches with a filler of about heavyMs', () => {
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 2 }), cost: sponzaLike, deltaTime: () => 1 / 60 });
        const burst = run.handed.slice(CALIBRATION.rampFrames, firstCalibratedFrame(run) - 1);
        const filler = Math.max(...burst.slice(CALIBRATION.samplesPerSize + 4));
        expect(sponzaLike(filler)).toBeGreaterThan(CALIBRATION.heavyMs * 0.7);
        // Small batches alternate with the filler: no two small ones in a row.
        const small = burst.map((units) => sponzaLike(units) < CALIBRATION.heavyMs / 2);
        expect(small.some((isSmall, index) => isSmall && small[index + 1])).toBe(false);
    });

    it('gives up after maxFrames without timings, and falls back to units or the default', () => {
        const burst = new CalibrationBurst(2, undefined);
        let units = 0;
        for (let frame = 1; frame <= CALIBRATION.maxFrames + 1; frame++) units = burst.next(frame, 1 / 60);
        expect(burst.result?.measured).toBe(false);
        expect(units).toBe(DEFAULT_UNITS_PER_FRAME);
    });

    it('a seeded burst (same work, new budget) only times the new answer', () => {
        const first = simulate({ frames: 300, budget: () => ({ gpuMs: 2 }), cost: sponzaLike });
        const seed = first.budget.report.calibration?.samples;
        const burst = new CalibrationBurst(4, undefined, seed);
        expect(burst.result).toBeUndefined();
        // Frame by frame, each frame's timing arriving 3 frames later.
        const handed: number[] = [];
        let frame = 1;
        for (; frame <= 80 && !burst.result; frame++) {
            handed[frame] = burst.next(frame, 0);
            const arrived = frame - 3;
            if (arrived >= 1) burst.add({ frame: arrived, units: handed[arrived], ms: sponzaLike(handed[arrived]) });
        }
        expect(burst.result?.measured).toBe(true);
        expect(sponzaLike(burst.result?.units ?? 0)).toBeLessThanOrEqual(4 * 1.1);
        expect(frame).toBeLessThan(first.reports.findIndex((report) => report.state === 'calibrated'));
    });
});

describe('each frame hands out the calibrated batch (budget v2)', () => {
    it('keeps the batch while frames are on time, even when the GPU clocks down and the same work takes twice as long', () => {
        // Calibrated with the clocks up; from frame 200 on a light load halves the clock.
        const run = simulate({ frames: 1200, budget: () => ({ gpuMs: 2 }), cost: sponzaLike, clockFactor: (frame) => (frame >= 200 ? 2 : 1) });
        const calibrated = run.budget.report.calibratedUnits as number;
        expect(unique(run.handed.slice(200))).toEqual([calibrated]);
        expect(run.budget.report.steppedDown).toBe(false);
    });

    it('applies budget.units as a hard cap, in the burst too', () => {
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 2, units: 50 }), cost: line, floor: ddgiFloor(19_425) });
        expect(Math.max(...run.handed)).toBeLessThanOrEqual(50);
        expect(unique(run.handed.slice(200))).toEqual([50]);
    });

    it('hands out units, or the default, without a millisecond budget or without timings', () => {
        expect(simulate({ frames: 5, budget: () => ({ units: 64 }), cost: line }).handed).toEqual([64, 64, 64, 64, 64]);
        expect(simulate({ frames: 2, budget: () => ({}), cost: line }).handed).toEqual([DEFAULT_UNITS_PER_FRAME, DEFAULT_UNITS_PER_FRAME]);
        const untimed = simulate({ frames: 3, budget: () => ({ gpuMs: 2, units: 100 }), cost: line, timed: false });
        expect(untimed.handed).toEqual([100, 100, 100]);
        expect(untimed.budget.report.state).toBe('not calibrated');
    });

    it('gives nothing for a zero budget', () => {
        expect(simulate({ frames: 3, budget: () => ({ gpuMs: 0 }), cost: line, floor: ddgiFloor(1000) }).handed).toEqual([0, 0, 0]);
    });

    it("uses a technique's own cost per unit instead of a burst", () => {
        const budget = new WorkBudget(undefined, () => 0.01);
        expect(budget.unitsFor({ frame: 1, deltaTime: 1 / 60, budget: { gpuMs: 2 }, shareMs: 2, timed: true })).toBe(200);
        expect(budget.report.state).toBe('calibrated');
    });
});

describe('re-calibration', () => {
    it('calibrates again, from the timings it has, when gpuMs changes', () => {
        const run = simulate({ frames: 900, budget: (frame) => ({ gpuMs: frame < 400 ? 2 : 4 }), cost: sponzaLike });
        const before = run.reports[390].calibratedUnits as number;
        const after = run.budget.report.calibratedUnits as number;
        expect(sponzaLike(before)).toBeLessThanOrEqual(2.2);
        expect(after).toBeGreaterThan(before * 2);
        expect(sponzaLike(after)).toBeLessThanOrEqual(4.4);
        // The second burst is short: it starts from the curve measured the first time.
        const secondBurst = run.reports.slice(399).findIndex((report) => report.state === 'calibrated');
        expect(secondBurst).toBeLessThan(firstCalibratedFrame(run));
    });

    it('calibrates again once the geometry has been still for a second, at most every 10 s', () => {
        // The scene gets twice as expensive at frame 1000 (a mesh added), and is still after it.
        const changeFrame = 1000;
        let frameNow = 0;
        const run = simulate({
            frames: 2000,
            budget: () => ({ gpuMs: 2 }),
            cost: (units) => line(units) * (frameNow >= changeFrame ? 2 : 1),
            geometryChange: (frame) => {
                frameNow = frame;
                return frame === changeFrame;
            },
        });
        const stillFrames = GEOMETRY_RECALIBRATION.stillSeconds * 60;
        const burstStart = run.reports.slice(changeFrame).findIndex((report) => report.state === 'calibrating');
        expect(Math.abs(burstStart - stillFrames)).toBeLessThanOrEqual(2);
        // 2 ms on 2.4 ms + 10 µs a probe: the fixed part alone is over, so 1 probe.
        expect(run.budget.report.calibratedUnits).toBeLessThan(10);
    });

    it('the first geometry re-calibration runs at once (trees finished after a host started early); the next waits 10 s after it', () => {
        // Changes at 5 s (trees arriving after the first calibration) and again at 8 s.
        const run = simulate({ frames: 1600, budget: () => ({ gpuMs: 2 }), cost: line, geometryChange: (frame) => frame === 300 || frame === 480 });
        const bursts = run.reports.flatMap((report, index) => (report.state === 'calibrating' && run.reports[index - 1]?.state !== 'calibrating' ? [index + 1] : []));
        expect(bursts[0]).toBe(1);
        expect(Math.abs(bursts[1] - 360)).toBeLessThanOrEqual(2);
        // The second change is still for a second at 9 s, but waits until 10 s after the 6 s burst.
        expect(Math.abs(bursts[2] - (bursts[1] + GEOMETRY_RECALIBRATION.minSpacingSeconds * 60))).toBeLessThanOrEqual(2);
        expect(bursts).toHaveLength(3);
    });

    it("doesn't calibrate every frame while something keeps moving", () => {
        const run = simulate({ frames: 1500, budget: () => ({ gpuMs: 2 }), cost: line, geometryChange: (frame) => frame > 400 });
        expect(run.reports.slice(400).every((report) => report.state === 'calibrated')).toBe(true);
    });

    it('calibrates again on request', () => {
        const run = simulate({
            frames: 600,
            budget: () => ({ gpuMs: 2 }),
            cost: line,
            beforeFrame: (frame, budget) => {
                if (frame === 400) budget.recalibrate();
            },
        });
        expect(run.reports[398].state).toBe('calibrated');
        expect(run.reports[399].state).toBe('calibrating');
        expect(run.budget.report.state).toBe('calibrated');
    });

    it('a new platform budget (prepare, a technique added) starts with a burst', () => {
        const budget = new WorkBudget(undefined);
        budget.unitsFor({ frame: 1, deltaTime: 1 / 60, budget: { gpuMs: 2 }, shareMs: 2, timed: true });
        expect(budget.report.state).toBe('calibrating');
    });
});

describe('saturation: the frame rate says when to step down (budget v2)', () => {
    const late = (frame: number) => (frame >= 400 && frame < 1000 ? 1 / 40 : 1 / 60);

    it('steps down ×0.8 a second while frames are late and the work is over its share, then recovers ×1.1 a second', () => {
        // The GPU is saturated from frame 400 to 1000: frames come at 40 Hz against 60, and the
        // work itself is 30% slower (the GPU shared with a heavier frame).
        const run = simulate({
            frames: 2200,
            budget: () => ({ gpuMs: 2, targetFps: 60 }),
            cost: sponzaLike,
            deltaTime: late,
            clockFactor: (frame) => (frame >= 400 && frame < 1000 ? 1.3 : 1),
        });
        const calibrated = run.budget.report.calibratedUnits as number;
        expect(run.handed[399]).toBe(calibrated);
        // After about 4 late seconds (160 frames at 40 Hz each): 0.8^n of the batch.
        const steppedDown = run.reports[960];
        expect(steppedDown.steppedDown).toBe(true);
        expect(steppedDown.framesLate).toBe(true);
        expect(run.handed[960]).toBeLessThan(calibrated * 0.8);
        // Back on time: back to the calibrated batch, never above it.
        expect(run.handed.at(-1)).toBe(calibrated);
        expect(Math.max(...run.handed.slice(1000))).toBe(calibrated);
        // ×1.1 a second at most.
        const recovered = run.handed.slice(1000).findIndex((units) => units === calibrated);
        const steps = Math.ceil(Math.log(calibrated / run.handed[999]) / Math.log(SATURATION.recovery));
        expect(recovered).toBeGreaterThanOrEqual((steps - 2) * 60);
    });

    it('never steps below the floor', () => {
        const run = simulate({
            frames: 1400,
            budget: () => ({ gpuMs: 2, targetFps: 60 }),
            cost: sponzaLike,
            deltaTime: (frame) => (frame >= 400 ? 1 / 30 : 1 / 60),
            clockFactor: (frame) => (frame >= 400 ? 3 : 1),
            floor: ddgiFloor(19_425),
        });
        // At 60 Hz Sponza's floor is 81 probes (but at most twice the calibrated batch).
        const floor = run.budget.report.floorUnits;
        expect(floor).toBeGreaterThan(0);
        expect(Math.min(...run.handed.slice(400))).toBeGreaterThanOrEqual(floor);
    });

    it('holds when frames are late but the work is within its share (the frame is slow for other reasons)', () => {
        const run = simulate({ frames: 1400, budget: () => ({ gpuMs: 2, targetFps: 60 }), cost: line, deltaTime: (frame) => (frame >= 400 ? 1 / 30 : 1 / 60) });
        const calibrated = run.budget.report.calibratedUnits as number;
        expect(unique(run.handed.slice(400))).toEqual([calibrated]);
    });

    it('finds the target from the observed frame rate without targetFps, remembering the on-time rate', () => {
        const guard = new SaturationGuard();
        let verdict;
        for (let frame = 0; frame < 120; frame++) verdict = guard.addFrame(1 / 60, undefined, 2) ?? verdict;
        expect(verdict).toBe('recover');
        expect(guard.framesPerSecond(undefined)).toBeCloseTo(60, 6);
        guard.addUpdateMs(3);
        for (let frame = 0; frame < 40; frame++) verdict = guard.addFrame(1 / 40, undefined, 2) ?? verdict;
        expect(verdict).toBe('step down');
        // The on-time rate is remembered while frames are late.
        expect(guard.framesPerSecond(undefined)).toBeCloseTo(60, 6);
        expect(guard.framesPerSecond(144)).toBe(144);
    });
});

describe('the floor (DDGI minProbesPerFrame)', () => {
    it("'auto' asks for a full sweep within 4 s at the host's frame rate", () => {
        // The design's numbers: Sponza 81 at 60 Hz, Cornell 8.
        expect(minProbesPerFrame('auto', 19_425, 4096, { framesPerSecond: 60, calibratedUnits: 1000 })).toBe(81);
        expect(minProbesPerFrame('auto', 1_728, 4096, { framesPerSecond: 60, calibratedUnits: 1000 })).toBe(8);
        expect(minProbesPerFrame('auto', 19_425, 4096, { framesPerSecond: 30, calibratedUnits: 1000 })).toBe(Math.ceil(19_425 / (FLOOR_SWEEP_SECONDS * 30)));
        expect(minProbesPerFrame('auto', 19_425, 4096, { framesPerSecond: 120, calibratedUnits: 1000 })).toBe(41);
    });

    it("'auto' never asks for more than twice the calibrated batch (the Intel UHD on Sponza)", () => {
        expect(minProbesPerFrame('auto', 19_425, 4096, { framesPerSecond: 60, calibratedUnits: 16 })).toBe(FLOOR_MAX_CALIBRATED_SHARE * 16);
    });

    it('a number replaces auto (a host can lower it), 0 turns it off, and maxProbesPerFrame caps it', () => {
        expect(minProbesPerFrame(10, 19_425, 4096, { framesPerSecond: 60, calibratedUnits: 1 })).toBe(10);
        expect(minProbesPerFrame(0, 19_425, 4096, { framesPerSecond: 60, calibratedUnits: 100 })).toBe(0);
        expect(minProbesPerFrame(5000, 19_425, 300, { framesPerSecond: 60, calibratedUnits: 100 })).toBe(300);
        expect(minProbesPerFrame('auto', 19_425, 50, { framesPerSecond: 60, calibratedUnits: 100 })).toBe(50);
    });

    it('wins over the millisecond budget, under budget.units', () => {
        // The Intel-like device: 2 ms buys about 15 probes, the 4 s sweep wants 81, the cap allows twice the batch.
        const intel = simulate({ frames: 600, budget: () => ({ gpuMs: 2 }), cost: intelSponzaLike, floor: ddgiFloor(19_425) });
        const calibrated = intel.budget.report.calibratedUnits as number;
        expect(unique(intel.handed.slice(400))).toEqual([Math.min(81, FLOOR_MAX_CALIBRATED_SHARE * calibrated)]);
        const capped = simulate({ frames: 600, budget: () => ({ gpuMs: 2, units: 20 }), cost: intelSponzaLike, floor: ddgiFloor(19_425) });
        expect(unique(capped.handed.slice(400))).toEqual([20]);
    });

    it('is found on the technique by name', () => {
        expect(workFloorOf(ddgiFloor(10))).toBeDefined();
        expect(workFloorOf({})).toBeUndefined();
    });
});

describe('behaviour at 30, 60 and 120 Hz (budget v2)', () => {
    for (const hz of [30, 60, 120]) {
        it(`${hz} Hz: one stable batch, the floor respected (Sponza on the RTX and on the Intel)`, () => {
            for (const cost of [sponzaLike, intelSponzaLike]) {
                const run = simulate({ frames: hz * 20, budget: () => ({ gpuMs: 2 }), cost, deltaTime: () => 1 / hz, floor: ddgiFloor(19_425) });
                const settled = run.handed.slice(hz * 5);
                // The floor follows the observed frame rate once a second of frames is in.
                expect(unique(settled)).toHaveLength(1);
                const report = run.budget.report;
                expect(report.framesPerSecond).toBeCloseTo(hz, 6);
                const floor = minProbesPerFrame('auto', 19_425, 4096, { framesPerSecond: hz, calibratedUnits: report.calibratedUnits as number });
                expect(report.floorUnits).toBe(floor);
                expect(settled[0]).toBe(Math.max(floor, report.calibratedUnits as number));
                expect(report.steppedDown).toBe(false);
            }
        });
    }
});
