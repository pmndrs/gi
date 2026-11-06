/**
 * T2.4c check 5's cost (docs/design/m2-lights.md §4.5, "Its cost"): Sponza
 * through the three adapter, GPU-bound (two frames in flight, as Budget v2
 * is measured), at the batch the platform calibrates for `gpuMs` 2 on this
 * code. A timing page: run it alone under the bench lock, which judges the
 * clock (VALID / INVALID).
 *
 * 1. A short warm-up (WARM_UP_PASSES passes, so the probes are classified),
 *    then the platform's calibration (budget v2), then the calibrated batch
 *    as a fixed count for the rest.
 * 2. **still**: STILL_FRAMES frames with nothing changing. `ddgi/propagate`
 *    finds nothing (its census count stays 0); its pass time is reported.
 * 3. **step**: the sun turned 30° lower; for STEP_FRAMES frames, the frames
 *    in which keys pass their change on. `ddgi/propagate` must cost at most
 *    0.1 ms (its median over those frames; the p90 and largest reported,
 *    since other programs' GPU work lands in some frames, NOTES).
 * 4. **day**: a 10-second day, DAY_FRAMES frames (60 Hz): the sun's
 *    elevation rises from DAY_LOW_DEGREES to DAY_HIGH_DEGREES and sets again
 *    (a sine over the day), while its azimuth moves steadily through
 *    DAY_AZIMUTH_DEGREES around the scene's own. An explicit path: until
 *    T2.6a this was built from repeated `turnedSun` calls, which took the sun
 *    below the horizon after about 77 frames (and, with the corrected
 *    helper, would hold it near the zenith for most of the day). The update, every `ddgi/` pass the platform
 *    counts against the budget (mark, schedule, trace, blend, write,
 *    propagate, census), must stay within Budget v2's margin, at most
 *    gpuMs × 1.1, as a median over the day's timed frames; its p95, and the
 *    trace + blend + write part Budget v2 calibrates on, are reported.
 *
 * Manual (needs the Sponza download), under the bench lock:
 *   node tools/reference/bench-lock.ts run --what "T2.4c cost" -- pnpm test:gpu --filter ddgi-key-probes-cost --gpu nvidia --json <file>
 */

import { assert } from '../../client.ts';
import { aimSun, startSponza, sunAngles, sunDirection, sunFromAngles, turnedSun, type SponzaUnderTest } from '../ddgi-shared/sponza-response.ts';
import { yieldToCallbacks } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const BUDGET_MS = 2;
/** Budget v2's margin (docs/design/ddgi.md, "Defined measurements": ≤ gpuMs +10%). */
const BUDGET_MARGIN = 0.1;
/** Check 5's bound on `ddgi/propagate` in a frame with a global change. */
const PROPAGATE_MS = 0.1;
const WARM_UP_PASSES = 4;
const CALIBRATION_MAX_FRAMES = 600;
const STILL_FRAMES = 240;
const STEP_FRAMES = 90;
const SUN_TURN_DEGREES = 30;
const DAY_FRAMES = 600;
/** The day's path: elevation from low to high and back, azimuth moving steadily (degrees). */
const DAY_LOW_DEGREES = 5;
const DAY_HIGH_DEGREES = 75;
const DAY_AZIMUTH_DEGREES = 160;
const FRAMES_IN_FLIGHT = 2;
const PROPAGATE_PASS = 'ddgi/propagate';
const MARK_PASS = 'ddgi/mark';
const CALIBRATED_PASSES = ['ddgi/trace', 'ddgi/blend', 'ddgi/write'];

/** Keeps up to FRAMES_IN_FLIGHT frames queued (GPU-bound), as ddgi-batch-cost does. */
class GpuBoundPacing {
    private readonly pending: Promise<void>[] = [];
    constructor(private readonly device: GPUDevice) {}
    async after(): Promise<void> {
        this.pending.push(this.device.queue.onSubmittedWorkDone());
        if (this.pending.length >= FRAMES_IN_FLIGHT) await this.pending.shift();
        await yieldToCallbacks();
    }
    async drain(): Promise<void> {
        await Promise.all(this.pending.splice(0));
    }
}

/** Every pass timing seen, by platform frame (the platform keeps only the newest timed frame's). */
class FrameTimings {
    readonly byFrame = new Map<number, Map<string, number>>();
    note(test: SponzaUnderTest): void {
        for (const pass of test.scene.platform.stats.passes) {
            if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
            let passes = this.byFrame.get(pass.frame);
            if (!passes) {
                passes = new Map();
                this.byFrame.set(pass.frame, passes);
            }
            passes.set(pass.name, pass.gpuMs);
        }
    }
    /** Frames in [first, last] with timings: every ddgi/ pass summed, the calibrated passes, and named passes. */
    frames(first: number, last: number): Array<{ frame: number; total: number; calibrated: number; passes: Map<string, number> }> {
        const result: Array<{ frame: number; total: number; calibrated: number; passes: Map<string, number> }> = [];
        for (let frame = first; frame <= last; frame++) {
            const passes = this.byFrame.get(frame);
            if (!passes || !CALIBRATED_PASSES.every((name) => passes.has(name))) continue;
            let total = 0;
            for (const [name, ms] of passes) if (name.startsWith('ddgi/')) total += ms;
            result.push({ frame, total, calibrated: CALIBRATED_PASSES.reduce((sum, name) => sum + (passes.get(name) ?? 0), 0), passes });
        }
        return result;
    }
}

/** Each pass's median over the frames (every pass seen in any of them; 0 in a frame without it), in ms. */
function passMedians(frames: ReadonlyArray<{ passes: Map<string, number> }>): Record<string, number> {
    const names = new Set(frames.flatMap((entry) => [...entry.passes.keys()]));
    const medians: Record<string, number> = {};
    for (const name of [...names].sort()) medians[name] = round(quantile(frames.map((entry) => entry.passes.get(name) ?? 0), 0.5));
    return medians;
}

/** The census counters that say what a frame's update list held, sampled once a frame (they arrive a few frames late). */
const LIST_COUNTERS = ['rays traced', 'hits lit by gi/direct', 'probes updated at priority', 'probes owed updates', 'probes propagated'] as const;

/** Each list counter's median over the samples. */
function counterMedians(samples: ReadonlyArray<Readonly<Record<string, number>>>): Record<string, number> {
    const medians: Record<string, number> = {};
    for (const name of LIST_COUNTERS) medians[name] = quantile(samples.map((counters) => counters[name] ?? Number.NaN).filter((value) => value >= 0), 0.5);
    return medians;
}

function quantile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))] ?? Number.NaN;
}
const round = (value: number, digits = 4) => Number(value.toFixed(digits));

await runThreePage(async ({ device, data, log: logLine }) => {
    const log = (line: string): void => {
        logLine(line);
        console.log(line);
    };
    const results = data;
    const failures: string[] = [];
    const test = await startSponza(device, 'sponza cost');
    const { scene } = test;
    const timings = new FrameTimings();
    const pacing = new GpuBoundPacing(device);
    const run = async (frames: number, budget: Parameters<typeof scene.frame>[0], deltaTime: number, before?: (index: number) => void): Promise<{ first: number; last: number; counters: Array<Readonly<Record<string, number>>> }> => {
        const first = scene.platform.stats.frame + 1;
        const counters: Array<Readonly<Record<string, number>>> = [];
        for (let index = 0; index < frames; index++) {
            before?.(index);
            scene.frame(budget, deltaTime);
            timings.note(test);
            counters.push(scene.ddgi.counters);
            await pacing.after();
        }
        await pacing.drain();
        // Timings arrive a few frames late: frames with no work let the last ones land.
        for (let extra = 0; extra < 6; extra++) {
            scene.frame({ paused: true }, 0);
            timings.note(test);
            await device.queue.onSubmittedWorkDone();
            await yieldToCallbacks();
        }
        return { first, last: first + frames - 1, counters };
    };

    // 1. Warm-up and calibration.
    scene.ddgi.settings.warmUpPasses = WARM_UP_PASSES;
    scene.ddgi.resetHistory();
    while (scene.ddgi.warmingUp) {
        scene.frame({ paused: false }, 0);
        await pacing.after();
    }
    await pacing.drain();
    scene.ddgi.settings.warmUpPasses = 0;
    let calibrationFrames = 0;
    for (;;) {
        scene.frame({ gpuMs: BUDGET_MS }, 0);
        calibrationFrames++;
        const report = scene.platform.budgetReport(scene.ddgi);
        if (!(report?.state === 'calibrating' || report?.confirming === true) || calibrationFrames >= CALIBRATION_MAX_FRAMES) break;
        await pacing.after();
    }
    await pacing.drain();
    const calibration = scene.platform.budgetReport(scene.ddgi)?.calibration;
    assert(calibration?.measured, `the platform did not calibrate within ${calibrationFrames} frames`);
    const units = calibration.units;
    results.calibratedUnits = units;
    log(`calibrated at ${BUDGET_MS} ms: ${units} probes a frame (${calibrationFrames} frames)`);

    // 2. Still.
    const stillRange = await run(STILL_FRAMES, { units }, 1 / 60);
    const still = timings.frames(stillRange.first, stillRange.last);
    const stillPropagate = still.map((entry) => entry.passes.get(PROPAGATE_PASS) ?? NaN).filter((ms) => ms >= 0);
    const stillCounters = scene.ddgi.counters;
    results.still = {
        frames: still.length,
        updateMedian: round(quantile(still.map((entry) => entry.total), 0.5)),
        calibratedMedian: round(quantile(still.map((entry) => entry.calibrated), 0.5)),
        propagateMedian: round(quantile(stillPropagate, 0.5)),
        probesPropagatedTotal: stillCounters['probes propagated (total)'],
        passes: passMedians(still),
        list: counterMedians(stillRange.counters),
    };
    log(`still, ${still.length} timed frames: update ${round(quantile(still.map((entry) => entry.total), 0.5))} ms (median, every ddgi/ pass), ddgi/propagate ${round(quantile(stillPropagate, 0.5))} ms with nothing to do; probes propagated ${stillCounters['probes propagated (total)']}`);
    if ((stillCounters['probes propagated (total)'] ?? 0) !== 0) failures.push(`still: propagation ran (${stillCounters['probes propagated (total)']} probes)`);

    // 3. The step: the frames in which keys pass their change on are the first few dozen.
    const startDirection = sunDirection(test.sun);
    const step = await run(STEP_FRAMES, { units }, 1 / 60, (index) => {
        if (index === 0) aimSun(test, turnedSun(startDirection, SUN_TURN_DEGREES));
    });
    const stepCounters = scene.ddgi.counters;
    // Keys pass their change while they are owed updates: all of them within about a second on Sponza (PROGRESS, T2.4c).
    const busy = timings.frames(step.first, step.last).filter((entry) => entry.frame - step.first < 45);
    const propagateBusy = busy.map((entry) => entry.passes.get(PROPAGATE_PASS) ?? NaN).filter((ms) => ms >= 0);
    const markMs = timings.frames(step.first, step.first).map((entry) => entry.passes.get(MARK_PASS) ?? NaN);
    results.step = {
        frames: busy.length,
        propagateMedian: round(quantile(propagateBusy, 0.5)),
        propagateP90: round(quantile(propagateBusy, 0.9)),
        propagateLargest: round(Math.max(...propagateBusy)),
        markMs: markMs[0] ?? null,
        keyChangesPassed: stepCounters['key changes passed (total)'],
        probesPropagated: stepCounters['probes propagated (total)'],
        updateMedian: round(quantile(busy.map((entry) => entry.total), 0.5)),
        passes: passMedians(busy),
        list: counterMedians(step.counters.slice(0, 45)),
    };
    log(
        `the sun turned ${SUN_TURN_DEGREES} degrees: ddgi/propagate over ${propagateBusy.length} timed frames of the first 45 median ${round(quantile(propagateBusy, 0.5))} ms, p90 ${round(quantile(propagateBusy, 0.9))}, largest ${round(Math.max(...propagateBusy))} (bar ${PROPAGATE_MS}); ` +
            `ddgi/mark in the step's frame ${markMs[0] ?? 'not timed'} ms; update median ${round(quantile(busy.map((entry) => entry.total), 0.5))} ms; ${stepCounters['key changes passed (total)']} key changes passed, ${stepCounters['probes propagated (total)']} probes propagated`,
    );
    if (!(quantile(propagateBusy, 0.5) <= PROPAGATE_MS)) failures.push(`step: ddgi/propagate median ${round(quantile(propagateBusy, 0.5))} ms (bar ${PROPAGATE_MS})`);

    // 4. A 10-second day.
    const startAzimuth = sunAngles(startDirection).azimuth;
    const sunAtDayFrame = (frame: number) => {
        const t = frame / DAY_FRAMES;
        return sunFromAngles(DAY_LOW_DEGREES + (DAY_HIGH_DEGREES - DAY_LOW_DEGREES) * Math.sin(Math.PI * t), startAzimuth + DAY_AZIMUTH_DEGREES * (t - 0.5));
    };
    aimSun(test, sunAtDayFrame(0));
    await run(30, { units }, 1 / 60);
    let dayFrame = 0;
    const elevations: number[] = [];
    const day = await run(DAY_FRAMES, { units }, 1 / 60, () => {
        dayFrame++;
        const direction = sunAtDayFrame(dayFrame);
        elevations.push(sunAngles(direction).elevation);
        aimSun(test, direction);
    });
    const dayFrames = timings.frames(day.first, day.last);
    const dayTotals = dayFrames.map((entry) => entry.total);
    const dayPropagate = dayFrames.map((entry) => entry.passes.get(PROPAGATE_PASS) ?? NaN).filter((ms) => ms >= 0);
    const dayMark = dayFrames.map((entry) => entry.passes.get(MARK_PASS) ?? NaN).filter((ms) => ms >= 0);
    const bar = BUDGET_MS * (1 + BUDGET_MARGIN);
    results.day = {
        frames: dayFrames.length,
        path: { lowDegrees: DAY_LOW_DEGREES, highDegrees: DAY_HIGH_DEGREES, azimuthDegrees: DAY_AZIMUTH_DEGREES, lowestElevation: round(Math.min(...elevations)), highestElevation: round(Math.max(...elevations)) },
        updateMedian: round(quantile(dayTotals, 0.5)),
        updateP95: round(quantile(dayTotals, 0.95)),
        calibratedMedian: round(quantile(dayFrames.map((entry) => entry.calibrated), 0.5)),
        propagateMedian: round(quantile(dayPropagate, 0.5)),
        markMedian: round(quantile(dayMark, 0.5)),
        bar,
        passes: passMedians(dayFrames),
        list: counterMedians(day.counters),
        counters: scene.ddgi.counters,
    };
    // Where the time goes (T2.4d): each pass, and what the update list held, still against the step and the day.
    const describeSection = (name: string, section: { passes: Record<string, number>; list: Record<string, number> }) =>
        `${name}: ${Object.entries(section.passes).map(([pass, ms]) => `${pass} ${ms}`).join(', ')}; list (medians): ${Object.entries(section.list).map(([counter, value]) => `${counter} ${value}`).join(', ')}`;
    log(describeSection('still', results.still as never));
    log(describeSection('after the step (first 45 frames)', results.step as never));
    log(describeSection('day', results.day as never));
    log(
        `a 10-second day (elevation ${DAY_LOW_DEGREES}-${DAY_HIGH_DEGREES}-${DAY_LOW_DEGREES} degrees, azimuth through ${DAY_AZIMUTH_DEGREES}, in ${DAY_FRAMES} frames) at ${units} probes, ${dayFrames.length} timed frames: update (every ddgi/ pass) median ${round(quantile(dayTotals, 0.5))} ms, p95 ${round(quantile(dayTotals, 0.95))} (bar ${bar}); ` +
            `trace + blend + write ${round(quantile(dayFrames.map((entry) => entry.calibrated), 0.5))}; ddgi/propagate ${round(quantile(dayPropagate, 0.5))}; ddgi/mark ${round(quantile(dayMark, 0.5))}`,
    );
    if (!(quantile(dayTotals, 0.5) <= bar)) failures.push(`day: the update's median ${round(quantile(dayTotals, 0.5))} ms is over Budget v2's margin (${bar} ms)`);
    scene.dispose();
    assert(failures.length === 0, failures.join('; '));
});
