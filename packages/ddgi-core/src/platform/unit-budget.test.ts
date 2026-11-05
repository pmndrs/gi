import { describe, expect, it } from 'vitest';
import type { FrameBudget } from './contract/frame.ts';
import type { GITechnique } from './contract/technique.ts';
import {
    CALIBRATION,
    CalibrationBurst,
    costOnCurve,
    DEFAULT_UNITS_PER_FRAME,
    SATURATION,
    SaturationGuard,
    UnitCostMeter,
    unitsOnCurve,
    WorkBudget,
    type BudgetReport,
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

/** A technique with DDGI's floor for a volume of `probeCount` probes. */
function ddgiLike(probeCount: number, setting: number | 'auto' = 'auto', maxProbesPerFrame = 4096): Pick<GITechnique, 'msPerUnit' | 'minUnitsPerFrame'> {
    return { msPerUnit: undefined, minUnitsPerFrame: (rate) => minProbesPerFrame(setting, probeCount, maxProbesPerFrame, rate) };
}

const VSYNC_60 = 1 / 60;
/** A vsync'd 60 Hz host whose GPU does `otherMs` of its own work: a frame that doesn't fit waits for the next refresh. */
const gpuBound60 = (otherMs: number) => (ddgiMs: number) => Math.ceil((otherMs + ddgiMs) / (1000 * VSYNC_60) - 1e-9) * VSYNC_60;

interface SimulationOptions {
    readonly frames: number;
    readonly budget: (frame: number) => FrameBudget;
    /** The technique's true GPU ms for a batch, at full clocks. */
    readonly cost: (units: number) => number;
    /** Seconds since the previous frame as the host reports them, given this frame's real DDGI ms (0: back to back). */
    readonly deltaTime?: (frame: number, ddgiMs: number) => number;
    /** Multiplies the true cost: above 1 while the GPU's clocks are down. */
    readonly clockFactor?: (frame: number) => number;
    /** The most units the technique can use (all its probes, or `maxProbesPerFrame`). */
    readonly capacity?: number;
    readonly technique?: Pick<GITechnique, 'msPerUnit' | 'minUnitsPerFrame'>;
    readonly geometryChange?: (frame: number) => boolean;
    readonly sceneSettling?: (frame: number) => boolean;
    readonly timed?: boolean;
    /** Frames until a frame's timings arrive (the platform's readback lag). */
    readonly lag?: number;
    /** Relative noise on each timing (uniform ±). */
    readonly noise?: number;
    readonly seed?: number;
    /** Called before each frame, for requests such as `recalibrate`. */
    readonly beforeFrame?: (frame: number, budget: WorkBudget) => void;
}

interface Simulation {
    readonly handed: number[];
    readonly used: number[];
    readonly reports: BudgetReport[];
    /** The host's frame time, seconds. */
    readonly frameSeconds: number[];
    readonly budget: WorkBudget;
}

/**
 * The platform's loop for one technique (gi-platform.ts, encode): the work
 * budget's hand-out, the technique using at most `capacity`, the meter
 * pairing timings with the units used, timings arriving `lag` frames late
 * with noise. The host's frame time can depend on this frame's DDGI work.
 */
function simulate(options: SimulationOptions): Simulation {
    const budget = new WorkBudget(options.technique);
    const meter = new UnitCostMeter();
    const lag = options.lag ?? 3;
    const noiseShare = options.noise ?? 0.03;
    let state = options.seed ?? 12345;
    const noise = () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return 1 - noiseShare + 2 * noiseShare * (state / 2 ** 32);
    };
    const pending: Array<{ frame: number; ms: number }> = [];
    const run: Simulation = { handed: [], used: [], reports: [], frameSeconds: [], budget };
    let previousDdgiMs = 0;
    for (let frame = 1; frame <= options.frames; frame++) {
        options.beforeFrame?.(frame, budget);
        if (options.geometryChange?.(frame)) budget.geometryChanged();
        const frameBudget = options.budget(frame);
        // The interval ending at this frame came from the previous frame's work.
        const deltaTime = options.deltaTime ? options.deltaTime(frame, previousDdgiMs) : VSYNC_60;
        const handed = budget.unitsFor({ frame, deltaTime, budget: frameBudget, shareMs: frameBudget.gpuMs, timed: options.timed ?? true, sceneSettling: options.sceneSettling?.(frame) });
        const used = Math.min(handed, options.capacity ?? Number.POSITIVE_INFINITY);
        meter.handOut(frame, 'ddgi', used);
        budget.record(handed, used);
        const ms = used === 0 ? 0 : options.cost(used) * (options.clockFactor?.(frame) ?? 1) * noise();
        previousDdgiMs = ms;
        pending.push({ frame, ms });
        while (pending.length > 0 && pending[0].frame <= frame - lag) {
            const arrived = pending.shift() as { frame: number; ms: number };
            for (const { sample } of meter.measure(arrived.frame, [{ name: 'ddgi/trace', gpuMs: arrived.ms }])) budget.addSample(sample);
        }
        run.handed.push(handed);
        run.used.push(used);
        run.reports.push(budget.report);
        run.frameSeconds.push(deltaTime);
    }
    return run;
}

/** Frames (1-based) at which a burst started. */
function burstStarts(run: Simulation): number[] {
    return run.reports.flatMap((report, index) => (report.state === 'calibrating' && run.reports[index - 1]?.state !== 'calibrating' ? [index + 1] : []));
}

const unique = (values: readonly number[]) => [...new Set(values)];
const within = (value: number, target: number, share: number) => Math.abs(value - target) <= share * target;

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

describe('the cost curve', () => {
    const curve = [
        { units: 64, ms: 1.4, spread: 0 },
        { units: 256, ms: 2.6, spread: 0 },
    ];
    it('reads costs along straight lines between sizes and beyond them', () => {
        expect(costOnCurve(curve, 160)).toBeCloseTo(2.0, 9);
        expect(costOnCurve(curve, 448)).toBeCloseTo(3.8, 9);
        expect(costOnCurve([{ units: 100, ms: 1, spread: 0 }], 50)).toBeCloseTo(0.5, 9);
    });
    it('solves the largest batch that fits a share, within a limit', () => {
        expect(unitsOnCurve(curve, 2, 10_000)).toBe(160);
        // 448 on paper; floating point may land a hair under.
        expect(unitsOnCurve(curve, 3.8, 10_000)).toBeGreaterThanOrEqual(447);
        expect(unitsOnCurve(curve, 3.8, 300)).toBe(300);
        expect(unitsOnCurve(curve, 0.1, 10_000)).toBe(1);
    });
});

describe('calibration on sustained frames (budget v2, after the review)', () => {
    it('finds the batch whose cost is the budget on a straight line, and reports the line and its convergence', () => {
        // 2 ms on 1.2 ms + 5 µs a probe: 160 probes.
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 2 }), cost: line, deltaTime: () => 0 });
        const calibration = run.budget.report.calibration;
        expect(calibration?.measured).toBe(true);
        expect(within(line(calibration?.units ?? 0), 2, 0.05)).toBe(true);
        expect(Math.abs((calibration?.model?.fixedMs ?? 0) - 1.2)).toBeLessThan(0.15);
        expect(Math.abs((calibration?.model?.msPerUnit ?? 0) * 1000 - 5)).toBeLessThan(0.6);
        // The spread of the six timings it settled on.
        expect(calibration?.spread).toBeLessThanOrEqual(CALIBRATION.agreeShare + CALIBRATION.driftShare);
        expect(calibration?.frames).toBeLessThan(80);
    });

    for (const [name, cost] of [
        ['Sponza on the RTX', sponzaLike],
        ['Sponza on the Intel', intelSponzaLike],
        ['a straight line', line],
    ] as const) {
        it(`lands within ±10% of the budget, both ways (${name})`, () => {
            for (const gpuMs of [1, 2, 4]) {
                const run = simulate({ frames: 400, budget: () => ({ gpuMs }), cost, deltaTime: () => 0 });
                const units = run.budget.report.calibratedUnits ?? 0;
                if (units > 1) expect(within(cost(units), gpuMs, 0.1)).toBe(true);
            }
        });
    }

    it('waits for the clocks: timings still moving are not used (the review: a burst at 1.7× the sustained cost)', () => {
        // The GPU starts at 1.8× the sustained cost and settles over the first 40 frames.
        const clockFactor = (frame: number) => (frame < 40 ? 1.8 - (0.8 * frame) / 40 : 1);
        for (const cost of [cornellLike, sponzaLike, line]) {
            const run = simulate({ frames: 400, budget: () => ({ gpuMs: 2 }), cost, clockFactor, capacity: 1728, deltaTime: () => 0, noise: 0.02 });
            const units = run.budget.report.calibratedUnits ?? 0;
            expect(within(cost(units), 2, 0.1)).toBe(true);
        }
    });

    it('confirms the batch over back-to-back frames after the burst, and re-solves when the GPU settled elsewhere', () => {
        // The burst reads the GPU 20% slow (right after the warm-up); a few seconds on, the true cost.
        const burstEnds = { frame: Number.POSITIVE_INFINITY };
        const run = simulate({
            frames: 900,
            budget: () => ({ gpuMs: 2 }),
            cost: sponzaLike,
            deltaTime: () => 0,
            clockFactor: (frame) => (frame <= burstEnds.frame + 100 ? 1.2 : 1),
            beforeFrame: (_frame, budget) => {
                if (budget.report.state === 'calibrated' && burstEnds.frame === Number.POSITIVE_INFINITY) burstEnds.frame = _frame;
            },
        });
        const report = run.budget.report;
        expect(report.confirming).toBe(false);
        expect(report.calibration?.confirmFactor).toBeCloseTo(1 / 1.2, 1);
        expect(within(sponzaLike(report.calibratedUnits as number), 2, 0.1)).toBe(true);
    });

    it('keeps the batch when the confirmation agrees, and waits for back-to-back frames to confirm at all', () => {
        const agreed = simulate({ frames: 700, budget: () => ({ gpuMs: 2 }), cost: line, deltaTime: () => 0 });
        expect(agreed.budget.report.calibration?.confirmFactor).toBe(1);
        // A 60 Hz host after the burst: nothing to confirm with, the batch stands.
        const live = simulate({ frames: 700, budget: () => ({ gpuMs: 2 }), cost: line, deltaTime: (frame) => (frame < 120 ? 0 : VSYNC_60) });
        expect(live.budget.report.confirming).toBe(true);
        expect(unique(live.handed.slice(200))).toEqual([live.budget.report.calibratedUnits]);
    });

    it('gives every probe when all of them fit (Cornell at 8 ms)', () => {
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 8 }), cost: cornellLike, capacity: 1000, deltaTime: () => 0 });
        expect(run.budget.report.calibratedUnits).toBe(1000);
        expect(Math.max(...run.handed)).toBeLessThanOrEqual(CALIBRATION.firstUnits * CALIBRATION.ladderStep);
    });

    it('runs no filler for a host running back to back', () => {
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 2 }), cost: sponzaLike, deltaTime: () => 0 });
        expect(run.budget.report.calibration?.fillerFrames).toBe(0);
    });

    it('a 60 Hz host gets filler in its first calibration only: each at most half a frame, at most fillerMaxMs in all', () => {
        const run = simulate({
            frames: 900,
            budget: () => ({ gpuMs: 2 }),
            cost: sponzaLike,
            beforeFrame: (frame, budget) => {
                if (frame === 500) budget.recalibrate();
            },
        });
        const first = run.reports[450].calibration;
        expect(first?.fillerFrames).toBeGreaterThan(0);
        expect(first?.fillerMs).toBeLessThanOrEqual(CALIBRATION.fillerMaxMs + CALIBRATION.fillerFrameShare * 1000 * VSYNC_60);
        const burst = run.handed.slice(0, (first?.frames ?? 0) - 1);
        expect(Math.max(...burst.map(sponzaLike))).toBeLessThanOrEqual(CALIBRATION.fillerFrameShare * 1000 * VSYNC_60 * 1.1 + 0.0001 + sponzaLike(CALIBRATION.firstUnits));
        expect(Math.max(...burst.slice(20).map(sponzaLike))).toBeLessThanOrEqual(CALIBRATION.fillerFrameShare * 1000 * VSYNC_60 * 1.1);
        expect(run.budget.report.calibration?.fillerFrames).toBe(0);
    });

    it('gives up after maxFrames without timings, and falls back to units or the default', () => {
        const burst = new CalibrationBurst(2, undefined, false);
        let units = 0;
        for (let frame = 1; frame <= CALIBRATION.maxFrames + 1; frame++) units = burst.next(frame, VSYNC_60);
        expect(burst.result?.measured).toBe(false);
        expect(units).toBe(DEFAULT_UNITS_PER_FRAME);
    });

    it('the first calibration waits while the mesh trees are built', () => {
        const run = simulate({ frames: 300, budget: () => ({ gpuMs: 2 }), cost: line, sceneSettling: (frame) => frame <= 100 });
        expect(run.reports.slice(0, 100).every((report) => report.state === 'not calibrated')).toBe(true);
        expect(burstStarts(run)).toEqual([101]);
    });
});

describe('each frame hands out the calibrated batch (budget v2)', () => {
    it('keeps the batch while frames are on time, even when the GPU clocks down and the same work takes twice as long', () => {
        const run = simulate({ frames: 1200, budget: () => ({ gpuMs: 2 }), cost: sponzaLike, deltaTime: (frame) => (frame < 200 ? 0 : VSYNC_60), clockFactor: (frame) => (frame >= 200 ? 2 : 1) });
        const calibrated = run.budget.report.calibratedUnits as number;
        expect(unique(run.handed.slice(200))).toEqual([calibrated]);
        expect(run.budget.report.steppedDown).toBe(false);
    });

    it('applies budget.units as a hard cap, in the burst too; lifting the cap re-solves without a burst', () => {
        const run = simulate({ frames: 600, budget: (frame) => (frame < 300 ? { gpuMs: 2, units: 50 } : { gpuMs: 2 }), cost: line, technique: ddgiLike(19_425), deltaTime: () => 0 });
        expect(Math.max(...run.handed.slice(0, 299))).toBeLessThanOrEqual(50);
        expect(unique(run.handed.slice(200, 299))).toEqual([50]);
        expect(burstStarts(run)).toEqual([1]);
        // Only sizes up to the old cap were measured: the new batch is extrapolated from them,
        // at least the old cap and never over the budget (+10%).
        const lifted = run.handed.at(-1) as number;
        expect(lifted).toBeGreaterThanOrEqual(50);
        expect(line(lifted)).toBeLessThanOrEqual(2.2);
    });

    it('hands out units, or the default, without a millisecond budget or without timings', () => {
        expect(simulate({ frames: 5, budget: () => ({ units: 64 }), cost: line }).handed).toEqual([64, 64, 64, 64, 64]);
        expect(simulate({ frames: 2, budget: () => ({}), cost: line }).handed).toEqual([DEFAULT_UNITS_PER_FRAME, DEFAULT_UNITS_PER_FRAME]);
        const untimed = simulate({ frames: 3, budget: () => ({ gpuMs: 2, units: 100 }), cost: line, timed: false });
        expect(untimed.handed).toEqual([100, 100, 100]);
        expect(untimed.budget.report.state).toBe('not calibrated');
    });

    it('gives nothing for a zero budget', () => {
        expect(simulate({ frames: 3, budget: () => ({ gpuMs: 0 }), cost: line, technique: ddgiLike(1000) }).handed).toEqual([0, 0, 0]);
    });

    it("uses a technique's own cost per unit instead of a burst", () => {
        const budget = new WorkBudget({ msPerUnit: 0.01 });
        expect(budget.unitsFor({ frame: 1, deltaTime: VSYNC_60, budget: { gpuMs: 2 }, shareMs: 2, timed: true })).toBe(200);
        expect(budget.report.state).toBe('calibrated');
    });
});

describe('no bursts during live use (budget v2, after the review)', () => {
    it('a new gpuMs is solved from the measured curve, with no burst', () => {
        const run = simulate({ frames: 900, budget: (frame) => ({ gpuMs: frame < 400 ? 2 : 4 }), cost: sponzaLike, deltaTime: (frame) => (frame < 200 ? 0 : VSYNC_60) });
        expect(burstStarts(run)).toEqual([1]);
        const after = run.budget.report.calibratedUnits as number;
        expect(within(sponzaLike(after), 4, 0.1)).toBe(true);
        expect(run.handed.at(-1)).toBe(after);
    });

    it('a geometry change marks the calibration stale and keeps the batch; recalibrate() runs a burst and clears it', () => {
        const run = simulate({
            frames: 1200,
            budget: () => ({ gpuMs: 2 }),
            cost: line,
            geometryChange: (frame) => frame === 400,
            beforeFrame: (frame, budget) => {
                if (frame === 800) budget.recalibrate();
            },
        });
        const calibrated = run.reports[390].calibratedUnits;
        expect(run.reports[700].stale).toBe(true);
        expect(unique(run.handed.slice(400, 799))).toEqual([calibrated]);
        expect(burstStarts(run)).toEqual([1, 800]);
        expect(run.budget.report.stale).toBe(false);
    });

    it("doesn't calibrate while something keeps moving", () => {
        const run = simulate({ frames: 1500, budget: () => ({ gpuMs: 2 }), cost: line, geometryChange: (frame) => frame > 400 });
        expect(burstStarts(run)).toEqual([1]);
    });
});

describe('saturation: step down only when frames are late and the GPU is busy (budget v2, after the review)', () => {
    const calibrateFirst = (frame: number, ddgiMs: number, then: (ddgiMs: number) => number) => (frame < 200 ? 0 : then(ddgiMs));

    it('settles under sustained saturation: a 60 Hz GPU-bound host whose own work leaves 1.2 ms', () => {
        // 15.5 ms of the host's GPU work: DDGI's calibrated 2 ms makes every frame miss the refresh.
        // No floor here (DDGI's would itself not fit in 1.2 ms on Sponza).
        const host = gpuBound60(15.5);
        // targetFps: a host late from its first frame would otherwise look like an on-time 30 Hz host.
        const run = simulate({ frames: 60 * 90, budget: () => ({ gpuMs: 2, targetFps: 60 }), cost: sponzaLike, deltaTime: (frame, ms) => calibrateFirst(frame, ms, host) });
        const calibrated = run.budget.report.calibratedUnits as number;
        const last30s = run.reports.slice(-60 * 30);
        // Mostly on time, and the batch moves rarely.
        const late = run.frameSeconds.slice(-60 * 30).filter((seconds) => seconds > VSYNC_60 * 1.5).length / (60 * 30);
        expect(late).toBeLessThan(0.15);
        const steps = last30s.filter((report, index) => index > 0 && report.units < last30s[index - 1].units).length;
        expect(steps).toBeLessThanOrEqual(3);
        expect(Math.max(...run.handed.slice(-60 * 30))).toBeLessThan(calibrated);
        expect(Math.min(...run.handed.slice(200))).toBeGreaterThanOrEqual(run.budget.report.floorUnits);
    });

    it("doesn't step down for a CPU-bound host: frames late, but DDGI at low clocks", () => {
        // The host makes 30 frames a second against a 60 Hz target; its GPU idles, so DDGI reads 2× slower.
        const run = simulate({ frames: 60 * 30, budget: () => ({ gpuMs: 2, targetFps: 60 }), cost: sponzaLike, deltaTime: (frame) => (frame < 200 ? 0 : 1 / 30), clockFactor: (frame) => (frame < 200 ? 1 : 2) });
        const calibrated = run.budget.report.calibratedUnits as number;
        expect(unique(run.handed.slice(200))).toEqual([calibrated]);
        expect(run.budget.report.framesLate).toBe(true);
    });

    it('an uncapped host with no targetFps has no ceiling: no step-down however its frame time moves', () => {
        // About 200 Hz, swinging between 4 and 7 ms with the view, GPU busy.
        const run = simulate({ frames: 60 * 60, budget: () => ({ gpuMs: 2 }), cost: sponzaLike, deltaTime: (frame, ms) => calibrateFirst(frame, ms, () => (frame % 600 < 300 ? 0.004 : 0.007)) });
        const calibrated = run.budget.report.calibratedUnits as number;
        expect(unique(run.handed.slice(200))).toEqual([calibrated]);
    });

    it('with targetFps, an uncapped host is held to it', () => {
        const run = simulate({ frames: 60 * 30, budget: () => ({ gpuMs: 2, targetFps: 250 }), cost: sponzaLike, deltaTime: (frame) => (frame < 200 ? 0 : 0.006) });
        expect(run.budget.report.steppedDown).toBe(true);
    });

    it('a host changing frame rate (120 Hz to 60 Hz, vsync): stable at each, back at the calibrated batch', () => {
        const run = simulate({ frames: 60 * 110, budget: () => ({ gpuMs: 2 }), cost: sponzaLike, deltaTime: (frame) => (frame < 200 ? 0 : frame < 1800 ? 1 / 120 : VSYNC_60), technique: ddgiLike(19_425) });
        const calibrated = run.budget.report.calibratedUnits as number;
        expect(unique(run.handed.slice(1000, 1800))).toEqual([calibrated]);
        // The slower rate reads as late for up to targetMemorySeconds (a GPU busy at full clocks
        // can't be told from saturation), then it is the new rate: recovery below the batch that was
        // late, then probing up every probeEverySeconds. Back at the calibrated batch within a minute.
        expect(unique(run.handed.slice(1800 + 60 * 60))).toEqual([calibrated]);
        expect(run.reports.slice(200).every((report) => report.units >= report.floorUnits)).toBe(true);
    });

    it('recovers only after recoverAfterSeconds on time, up to below the batch that was late', () => {
        // Late (GPU busy) from 4 s to 6 s, then on time.
        const run = simulate({ frames: 60 * 40, budget: () => ({ gpuMs: 2, targetFps: 60 }), cost: sponzaLike, deltaTime: (frame) => (frame < 200 ? 0 : frame >= 240 && frame < 360 ? 1 / 40 : VSYNC_60) });
        const calibrated = run.budget.report.calibratedUnits as number;
        const lowest = Math.min(...run.handed);
        expect(lowest).toBeLessThan(calibrated * 0.8);
        // No growth in the first recoverAfterSeconds after the last late window.
        const lateEnd = run.reports.length - 1 - [...run.reports].reverse().findIndex((report) => report.framesLate);
        const heldUntil = lateEnd + Math.round(SATURATION.recoverAfterSeconds * 60) - 70;
        expect(unique(run.handed.slice(lateEnd + 1, heldUntil))).toHaveLength(1);
        expect(run.reports[lateEnd + 300].ceilingUnits).toBeDefined();
        expect(Math.max(...run.handed.slice(lateEnd, lateEnd + 600))).toBeLessThan(calibrated);
    });

    it('treats an observed rate as a target only when it is a display refresh rate', () => {
        const guard = new SaturationGuard();
        for (let frame = 0; frame < 70; frame++) guard.addFrame(VSYNC_60, undefined);
        expect(guard.targetSeconds(undefined)).toBeCloseTo(VSYNC_60, 9);
        const uncapped = new SaturationGuard();
        for (let frame = 0; frame < 300; frame++) uncapped.addFrame(1 / 230, undefined);
        expect(uncapped.targetSeconds(undefined)).toBeUndefined();
        expect(uncapped.targetSeconds(144)).toBeCloseTo(1 / 144, 9);
        expect(uncapped.framesPerSecond(undefined)).toBeCloseTo(230, 6);
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
        const intel = simulate({ frames: 900, budget: () => ({ gpuMs: 2 }), cost: intelSponzaLike, technique: ddgiLike(19_425), deltaTime: (frame) => (frame < 300 ? 0 : VSYNC_60) });
        const calibrated = intel.budget.report.calibratedUnits as number;
        expect(unique(intel.handed.slice(600))).toEqual([Math.min(81, FLOOR_MAX_CALIBRATED_SHARE * calibrated)]);
        const capped = simulate({ frames: 900, budget: () => ({ gpuMs: 2, units: 20 }), cost: intelSponzaLike, technique: ddgiLike(19_425), deltaTime: (frame) => (frame < 300 ? 0 : VSYNC_60) });
        expect(unique(capped.handed.slice(600))).toEqual([20]);
    });
});

describe('behaviour at 30, 60 and 120 Hz (budget v2)', () => {
    for (const hz of [30, 60, 120]) {
        it(`${hz} Hz: one stable batch, the floor respected (Sponza on the RTX and on the Intel)`, () => {
            for (const cost of [sponzaLike, intelSponzaLike]) {
                const run = simulate({ frames: 300 + hz * 20, budget: () => ({ gpuMs: 2 }), cost, deltaTime: (frame) => (frame < 300 ? 0 : 1 / hz), technique: ddgiLike(19_425) });
                const settled = run.handed.slice(300 + hz * 2);
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
