/**
 * Budget v2: how many work units (DDGI: probes) each technique gets per frame
 * (docs/design/ddgi.md, "Budget v2"; accepted by Dennis, DECISIONS
 * 2026-11-05). It replaces T1.9a's frame-by-frame millisecond steering.
 *
 * Why work, not milliseconds, is the control: a GPU-millisecond budget and
 * GPU power management feed each other. Less work lets the GPU idle, it
 * lowers its clocks, the same work takes longer, and a millisecond controller
 * cuts the work again. On a quiet laptop at 60 Hz the RTX 3060 sat in its
 * low-power state and 2 ms bought 2-3 Sponza probes a frame (a full sweep
 * every 162 s). The budget exists to protect the frame time, and a GPU that
 * idles has time to spare.
 *
 * Per technique ({@link WorkBudget}):
 * 1. **Calibration** ({@link CalibrationBurst}): after `prepare` (or on
 *    `recalibrate()`), with the GPU loaded, measure batch sizes until their
 *    timings agree (the clocks have settled), and take the largest batch
 *    whose cost fits the technique's share of `budget.gpuMs`. The measured
 *    curve is kept: a new `gpuMs` is solved from it with no new burst.
 * 2. **Each frame hands out that batch.** Nothing is steered on GPU
 *    milliseconds while frames arrive on time: a slow-looking pass on an
 *    idle GPU is low clocks, not a real cost. A geometry change marks the
 *    calibration stale (reported) but keeps the batch; the host re-runs it
 *    at its next `prepare` or with `recalibrate()`.
 * 3. **The floor** (`GITechnique.minUnitsPerFrame`): the technique may ask
 *    for a minimum (DDGI: a full sweep within 4 s), which wins over the
 *    millisecond ceiling, so a weak device runs slower, never frozen.
 * 3b. **The calibration guard** (T2.4b, after M1 once handed Sponza 26 probes
 *    a frame instead of about 290): a calibration under half of the
 *    previous one on the same device and scene is measured once more before
 *    it is used ({@link CALIBRATION} `guardShare`). Every burst keeps its
 *    timings (`Calibration.samples`), so a bad one can be read afterwards.
 * 4. **Saturation** ({@link SaturationGuard}): the batch steps down only when
 *    frames are late *and* the technique's timings show the GPU at full
 *    clocks (close to the calibrated cost), which is when the GPU is really
 *    the bottleneck. Recovery waits for frames on time and stops below the
 *    level that was late, then probes up slowly (no oscillation).
 *
 * `budget.units` caps everything, the burst included (FrameBudget).
 */

import type { FrameBudget } from './contract/frame.ts';
import type { GITechnique } from './contract/technique.ts';

/**
 * Work units a technique gets when `budget.gpuMs` can't be used (no GPU
 * timings: `gpuTiming` off, or no `timestamp-query`) and the budget names no
 * units: DDGI's default probe count per frame on Tier A (docs/design/ddgi.md).
 */
export const DEFAULT_UNITS_PER_FRAME = 512;

/**
 * A technique's GPU cost for a batch of `units`: `fixedMs + msPerUnit × units`.
 *
 * Why a fixed part (T1.9a): a small batch doesn't fill the GPU, so its time
 * is mostly the latency of the slowest workgroup, not the sum of the work.
 * DDGI on Sponza (RTX 3060) costs about the same for 32 probes as for 128,
 * then about 6 µs per probe more. Budget v2 reports the line around the
 * calibrated batch ({@link Calibration.model}).
 */
export interface UnitCostModel {
    readonly fixedMs: number;
    readonly msPerUnit: number;
}

/** One timed frame of a technique: the units it used and the GPU ms its passes took. */
export interface CostSample {
    /** The platform frame the work was recorded in. */
    readonly frame: number;
    readonly units: number;
    readonly ms: number;
}

/** Frames of handed-out units remembered while their timings are on the way (they arrive a few frames late). */
const REMEMBERED_FRAMES = 32;

/**
 * Pairs each technique's GPU time in a frame with the units it used that
 * frame. The time is the sum of the passes named `<technique id>/...` (DDGI:
 * trace, blend, write, and the census while timing is on); the units are
 * what its `encode` returned, so a hand-out bigger than the work can't make
 * units look cheaper than they are.
 */
export class UnitCostMeter {
    private readonly handedOut = new Map<number, Map<string, number>>();

    /** Records the units a technique used in a frame. */
    handOut(frame: number, techniqueId: string, units: number): void {
        let frameUnits = this.handedOut.get(frame);
        if (!frameUnits) {
            frameUnits = new Map();
            this.handedOut.set(frame, frameUnits);
            for (const old of this.handedOut.keys()) if (old <= frame - REMEMBERED_FRAMES) this.handedOut.delete(old);
        }
        frameUnits.set(techniqueId, units);
    }

    /** One frame's pass timings: a sample for each technique that did work and was timed. */
    measure(frame: number, passes: readonly { readonly name: string; readonly gpuMs: number }[]): Array<{ readonly techniqueId: string; readonly sample: CostSample }> {
        const frameUnits = this.handedOut.get(frame);
        if (!frameUnits) return [];
        const measured: Array<{ techniqueId: string; sample: CostSample }> = [];
        for (const [techniqueId, units] of frameUnits) {
            if (!(units > 0)) continue;
            const prefix = `${techniqueId}/`;
            const timed = passes.filter((pass) => pass.name.startsWith(prefix) && pass.gpuMs >= 0);
            if (timed.length === 0) continue;
            const ms = timed.reduce((sum, pass) => sum + pass.gpuMs, 0);
            measured.push({ techniqueId, sample: { frame, units, ms } });
        }
        this.handedOut.delete(frame);
        return measured;
    }

    reset(): void {
        this.handedOut.clear();
    }
}

// ---------------------------------------------------------------------------
// The cost curve
// ---------------------------------------------------------------------------

/** One batch size's settled cost: the median of timings that agreed. */
export interface CurvePoint {
    readonly units: number;
    readonly ms: number;
    /** (largest − smallest) / smallest of the timings it was taken from. */
    readonly spread: number;
}

/**
 * The cost of `units` on the measured curve: straight lines between measured
 * sizes, the end segments extended beyond them, proportional with one size.
 */
export function costOnCurve(curve: readonly CurvePoint[], units: number): number | undefined {
    if (curve.length === 0) return undefined;
    if (curve.length === 1) return (curve[0].ms / curve[0].units) * units;
    let upper = curve.findIndex((point) => point.units >= units);
    if (upper <= 0) upper = upper === 0 ? 1 : curve.length - 1;
    const a = curve[upper - 1];
    const b = curve[upper];
    return a.ms + ((b.ms - a.ms) * (units - a.units)) / (b.units - a.units);
}

/**
 * The largest batch, at most `limit`, whose cost on the curve fits `shareMs`
 * (at least 1 while there is a share, so the technique keeps working and
 * being measured).
 */
export function unitsOnCurve(curve: readonly CurvePoint[], shareMs: number, limit: number): number {
    if (curve.length === 0 || !(shareMs > 0)) return 0;
    const fitting = curve.filter((point) => point.ms <= shareMs);
    const under = fitting.at(-1);
    const over = curve.find((point) => point.ms > shareMs && (under === undefined || point.units > under.units));
    let units: number;
    if (under && over) {
        units = under.units + Math.floor(((shareMs - under.ms) * (over.units - under.units)) / (over.ms - under.ms));
        units = Math.min(units, over.units - 1);
    } else {
        // Outside the measured sizes: along the end segment (or proportionally with one size).
        const ends = under ? curve.slice(-2) : curve.slice(0, 2);
        const [a, b] = ends.length === 2 ? ends : [{ units: 0, ms: 0 }, ends[0]];
        const msPerUnit = (b.ms - a.ms) / (b.units - a.units);
        units = msPerUnit > 0 ? Math.floor(a.units + (shareMs - a.ms) / msPerUnit) : (under?.units ?? 1);
        // Beyond the largest size measured: no further than one ladder step, which the
        // technique may not even have (its capacity is known only once it used fewer).
        if (under) units = Math.min(Math.max(units, under.units), under.units * CALIBRATION.ladderStep);
    }
    return Math.max(1, Math.min(limit, units));
}

/** The line through the measured sizes on either side of `units` (the two nearest when outside them). */
function modelAround(curve: readonly CurvePoint[], units: number): UnitCostModel | undefined {
    if (curve.length === 0) return undefined;
    if (curve.length === 1) return { fixedMs: 0, msPerUnit: curve[0].ms / curve[0].units };
    let upper = curve.findIndex((point) => point.units > units);
    if (upper <= 0) upper = upper === 0 ? 1 : curve.length - 1;
    const a = curve[upper - 1];
    const b = curve[upper];
    const msPerUnit = (b.ms - a.ms) / (b.units - a.units);
    if (!(msPerUnit > 0)) return { fixedMs: 0, msPerUnit: b.ms / b.units };
    return { fixedMs: a.ms - msPerUnit * a.units, msPerUnit };
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/**
 * How a calibration burst measures (see {@link CalibrationBurst}).
 *
 * - `firstUnits`: the first batch size tried.
 * - `agreeSamples`, `agreeShare`, `driftShare`: a size's cost counts once
 *   its latest `agreeSamples` timings agree within `agreeShare` (largest /
 *   smallest) and their median is within `driftShare` of the median of the
 *   `agreeSamples` before them: settled, and not drifting. Timings that still move are the
 *   GPU changing its clocks (the review of the first version: a fixed
 *   6-frame ramp let the burst measure some runs at 1.7× the sustained cost,
 *   so Cornell got 695 probes for 1.14 ms). Clocks rising 2% a frame pass
 *   three timings within 5%, but not two trios 2% apart.
 * - `maxSamplesPerSize`: a size whose timings never agree (a noisy machine)
 *   takes the median of its last `agreeSamples + 2` timings after this many,
 *   with the spread reported.
 * - `ladderStep`, `smallestUnits`: sizes step by this factor while looking
 *   for the two on either side of the budget, down to `smallestUnits`.
 * - `verifyTolerance`, `verifyRounds`: the batch solved between the two
 *   sizes is measured itself and accepted within this share of the budget,
 *   otherwise solved again with the new point (the curve isn't a straight
 *   line between sizes four times apart). The page then requires ±10%.
 * - `fillerFrameShare`, `fillerMaxMs`: only in the first calibration after
 *   `prepare`, and only when the host leaves time between frames (it isn't
 *   running them back to back), small batches alternate with a filler batch
 *   that keeps the GPU loaded so its clocks stay up. A filler is at most this
 *   share of the observed frame interval, and all of them together at most
 *   this many GPU ms. Counted in the stats.
 * - `maxFrames`: a burst still running after this many frames gives up
 *   (timings aren't arriving, or never agree) and reports itself not
 *   calibrated; the hand-out is then the best estimate from the sizes it did
 *   settle, or the technique's floor when there are none (WorkBudget).
 * - `confirmFrames`, `confirmJudgedFrames`, `confirmTolerance`: after the
 *   burst, while the host still runs frames back to back (`deltaTime` 0,
 *   its loading screen), the chosen batch is timed for `confirmFrames`
 *   frames; if the median of the last `confirmJudgedFrames` is off the
 *   curve by more than `confirmTolerance`, the curve is scaled by that
 *   factor and the batch solved again. Why: on the RTX 3060 a calibration
 *   right after the warm-up read the GPU 15-20% off, either way, while one
 *   a few seconds later matched the sustained cost within 5% (budget v2
 *   rework). The GPU settles over seconds, which agreeing timings a few
 *   frames apart can't see.
 * - `guardShare`: a burst whose batch comes out under this share of the
 *   previous calibration's on the same device and scene (solved for the
 *   same share and cap) is measured once more, and the second result is
 *   used whatever it is. Why (M1, the flicker check): one burst handed
 *   Sponza 26 probes a frame instead of about 290, and the next run didn't
 *   repeat it, so it was a burst that measured a passing state of the GPU,
 *   not the scene. Half leaves room for real changes (a new budget is solved
 *   from the curve anyway; a heavier scene really is slower) and catches a
 *   10× miss. A scene that really got that much heavier costs one more burst.
 */
export const CALIBRATION = {
    firstUnits: 1024,
    agreeSamples: 3,
    agreeShare: 0.05,
    driftShare: 0.02,
    maxSamplesPerSize: 24,
    ladderStep: 4,
    smallestUnits: 4,
    verifyTolerance: 0.05,
    verifyRounds: 3,
    fillerFrameShare: 0.5,
    fillerMaxMs: 1000,
    maxFrames: 400,
    confirmFrames: 300,
    confirmJudgedFrames: 100,
    confirmTolerance: 0.1,
    guardShare: 0.5,
} as const;

/** What a calibration found. */
export interface Calibration {
    /** The batch the technique's share buys with the clocks up. */
    readonly units: number;
    /** The technique's share of `budget.gpuMs` it was solved for. */
    readonly shareMs: number;
    /** The line around the batch (between the measured sizes on either side of it). Undefined when nothing was measured. */
    readonly model: UnitCostModel | undefined;
    /** `budget.units` it was solved under. */
    readonly cap: number | undefined;
    /** Every size measured, smallest first: a new `gpuMs` or cap is solved from it without a burst. */
    readonly curve: readonly CurvePoint[];
    /** The most units the technique can use (all its probes, `maxProbesPerFrame`), as far as the burst found. */
    readonly capacity: number;
    /** Burst frames it took. */
    readonly frames: number;
    /** The spread of the timings the chosen batch's cost was taken from (convergence). */
    readonly spread: number;
    /** Filler frames and their estimated GPU ms. */
    readonly fillerFrames: number;
    readonly fillerMs: number;
    /** False when the burst gave up (CALIBRATION `maxFrames`); `units` is then its best estimate, 0 with none. */
    readonly measured: boolean;
    /**
     * Sustained timings after the burst against the curve (CALIBRATION
     * `confirmFrames`): the factor the curve was scaled by, 1 when it was
     * within the tolerance; undefined until confirmed.
     */
    readonly confirmFactor?: number;
    /** Every timing the burst took, in the order they arrived (the calibration's log). */
    readonly samples: readonly CostSample[];
    /**
     * The calibration guard (CALIBRATION `guardShare`): the batch a first
     * burst found, when it came out under half of the previous calibration
     * and this one is its re-measurement; undefined otherwise.
     */
    readonly remeasuredFrom?: number;
}

/**
 * One calibration: a run of frames ("burst") whose batch sizes the platform
 * picks, timed until each size's timings agree.
 *
 * Frame by frame (what {@link next} hands out):
 * 1. `firstUnits`, until its timings agree.
 * 2. Down (or up) by `ladderStep` until two settled sizes sit on either side
 *    of the share.
 * 3. The batch on the line between them is measured itself and accepted, or
 *    solved again (`verifyRounds`).
 *
 * The host runs it best GPU-bound behind a loading screen, two frames in
 * flight so the GPU never waits: the viewer and the batch-cost page do
 * (about 20-170 frames). A host at 60 Hz gets filler batches in the
 * platform's first calibration instead (see CALIBRATION).
 */
export class CalibrationBurst {
    private readonly shareMs: number;
    private readonly cap: number | undefined;
    private readonly fillerAllowed: boolean;
    /** Timings of the sizes not settled yet, in arrival order. */
    private readonly pending = new Map<number, number[]>();
    /** Settled sizes. */
    private readonly settled = new Map<number, CurvePoint>();
    /** The most units the technique can use, once it used fewer than it was handed. */
    private capacity = Number.POSITIVE_INFINITY;
    private probe: number;
    private verifying = false;
    private verifyRoundsUsed = 0;
    private frames = 0;
    /** Timings of frames before this one are from before the burst. */
    private firstFrame: number | undefined;
    private readonly frameSeconds: number[] = [];
    private fillerThisFrame = false;
    private fillerFrames = 0;
    private fillerMs = 0;
    /** Every timing taken, for the calibration's log. */
    private readonly samples: CostSample[] = [];
    private finished: Calibration | undefined;

    /** `cap`: `budget.units`. `fillerAllowed`: the platform's first calibration, behind its start-up (see CALIBRATION). */
    constructor(shareMs: number, cap: number | undefined, fillerAllowed: boolean) {
        this.shareMs = shareMs;
        this.cap = cap;
        this.fillerAllowed = fillerAllowed;
        this.probe = Math.min(CALIBRATION.firstUnits, this.limit);
    }

    /** The most units a burst frame may hand out: the host's cap and the technique's capacity. */
    private get limit(): number {
        return Math.min(this.cap ?? Number.POSITIVE_INFINITY, this.capacity);
    }

    /** The calibration, once the burst is done. */
    get result(): Calibration | undefined {
        return this.finished;
    }

    /** Units to hand out in burst frame `frame` (the platform's frame number). */
    next(frame: number, deltaTime: number): number {
        this.firstFrame ??= frame;
        this.frames++;
        if (deltaTime > 0 && Number.isFinite(deltaTime)) this.frameSeconds.push(deltaTime);
        if (this.frames > CALIBRATION.maxFrames && !this.finished) {
            // Gave up: the best estimate from the sizes that did settle (0 with none; the
            // budget then hands out the floor), reported as not calibrated.
            const curve = this.curve();
            this.finished = this.calibration(curve.length > 0 ? unitsOnCurve(curve, this.shareMs, this.limit) : 0, false);
            return this.finished.units;
        }
        const filler = this.fillerUnits();
        this.fillerThisFrame = filler !== undefined && !this.fillerThisFrame;
        if (this.fillerThisFrame && filler !== undefined) {
            this.fillerFrames++;
            this.fillerMs += filler.ms;
            return filler.units;
        }
        return this.probe;
    }

    /** What the technique used of a burst hand-out: fewer means that is all it can use. */
    used(handed: number, used: number): void {
        if (!(used < handed)) return;
        this.capacity = Math.min(this.capacity, Math.max(0, used));
        if (this.probe > this.limit) this.probe = this.limit;
        if (this.limit === 0 && !this.finished) this.finished = this.calibration(0, true);
    }

    /** A timed frame. Frames from before the burst don't count. */
    add(sample: CostSample): void {
        if (this.finished || this.firstFrame === undefined || sample.frame < this.firstFrame) return;
        if (!(sample.units > 0) || !(sample.ms >= 0)) return;
        this.samples.push(sample);
        if (this.settled.has(sample.units)) return;
        let list = this.pending.get(sample.units);
        if (!list) {
            list = [];
            this.pending.set(sample.units, list);
        }
        list.push(sample.ms);
        const point = settle(sample.units, list);
        if (point) {
            this.settled.set(sample.units, point);
            this.pending.delete(sample.units);
            this.decide();
        }
    }

    /**
     * A filler batch for this frame, or undefined: only in the first
     * calibration, when the host leaves time between frames, while the size
     * being measured is small next to the filler, and within `fillerMaxMs`.
     */
    private fillerUnits(): { units: number; ms: number } | undefined {
        if (!this.fillerAllowed || this.frameSeconds.length === 0 || this.fillerMs >= CALIBRATION.fillerMaxMs) return undefined;
        const curve = this.curve();
        const largest = curve.at(-1);
        if (!largest) return undefined;
        const frameMs = median(this.frameSeconds) * 1000;
        const targetMs = CALIBRATION.fillerFrameShare * frameMs;
        const units = Math.min(this.limit, Math.floor((largest.units * targetMs) / largest.ms));
        const ms = costOnCurve(curve, units) ?? targetMs;
        // A size this close to the filler loads the GPU by itself.
        if (!(units > 1) || (costOnCurve(curve, this.probe) ?? 0) >= targetMs / 2) return undefined;
        return { units, ms };
    }

    /** After a size settled: the next size, or the answer. */
    private decide(): void {
        const share = this.shareMs;
        // Bounded: each pass either waits for timings, finishes, or moves to a new size.
        for (let guard = 0; guard < 64 && !this.finished; guard++) {
            const probePoint = this.settled.get(this.probe);
            if (!probePoint) return;

            if (this.verifying) {
                this.verifying = false;
                if (Math.abs(probePoint.ms - share) <= CALIBRATION.verifyTolerance * share) {
                    this.finish(this.probe);
                    return;
                }
                // Off by more than the tolerance: the line between the two sizes around it was
                // wrong, most likely because the size on the other side was measured while the
                // clocks still moved. Drop that size from the curve; the ladder below then moves
                // on from the sizes left (and measures that size again if it is the next step).
                const curve = this.curve();
                const suspect =
                    probePoint.ms <= share
                        ? curve.find((point) => point.units > this.probe && point.ms > share)
                        : curve.filter((point) => point.units < this.probe && point.ms <= share).at(-1);
                if (suspect) this.settled.delete(suspect.units);
            }

            const curve = this.curve();
            const under = curve.filter((point) => point.ms <= share).at(-1);
            const over = curve.find((point) => point.ms > share && (under === undefined || point.units > under.units));
            if (under === undefined) {
                // Every size measured costs more than the share: go smaller, down to `smallestUnits`.
                if (curve[0].units <= CALIBRATION.smallestUnits) {
                    this.finish(1);
                    return;
                }
                this.probe = Math.max(CALIBRATION.smallestUnits, Math.floor(curve[0].units / CALIBRATION.ladderStep));
                continue;
            }
            if (over === undefined) {
                // Every size measured fits: go bigger, up to all the technique can use.
                if (under.units >= this.limit) {
                    this.finish(this.limit);
                    return;
                }
                this.probe = Math.min(this.limit, under.units * CALIBRATION.ladderStep);
                continue;
            }
            const candidate = unitsOnCurve(curve, share, this.limit);
            if (candidate <= under.units || this.verifyRoundsUsed >= CALIBRATION.verifyRounds) {
                this.finish(Math.max(under.units, candidate));
                return;
            }
            this.verifyRoundsUsed++;
            this.verifying = true;
            this.probe = candidate;
        }
    }

    private finish(units: number): void {
        this.finished = this.calibration(units, true);
    }

    private calibration(units: number, measured: boolean): Calibration {
        const limited = Math.max(0, Math.min(Math.floor(units), this.limit));
        const curve = this.curve();
        return {
            units: limited,
            shareMs: this.shareMs,
            cap: this.cap,
            model: measured ? modelAround(curve, limited) : undefined,
            curve,
            capacity: this.capacity,
            frames: this.frames,
            spread: this.settled.get(limited)?.spread ?? Math.max(0, ...curve.map((point) => point.spread)),
            fillerFrames: this.fillerFrames,
            fillerMs: this.fillerMs,
            measured,
            samples: [...this.samples],
        };
    }

    /** Every settled size within the limit, smallest first. */
    private curve(): CurvePoint[] {
        return [...this.settled.values()].filter((point) => point.units <= this.limit).sort((a, b) => a.units - b.units);
    }
}

/**
 * A size's cost once its timings have settled (see CALIBRATION:
 * `agreeSamples`), or after `maxSamplesPerSize` the median of the latest few
 * whatever their spread. Undefined while still moving.
 */
function settle(units: number, timings: readonly number[]): CurvePoint | undefined {
    const spreadOf = (values: readonly number[]) => (Math.min(...values) > 0 ? Math.max(...values) / Math.min(...values) - 1 : Number.POSITIVE_INFINITY);
    const count = CALIBRATION.agreeSamples;
    if (timings.length >= 2 * count) {
        const recent = timings.slice(-count);
        const before = timings.slice(-2 * count, -count);
        if (spreadOf(recent) <= CALIBRATION.agreeShare && spreadOf([median(recent), median(before)]) <= CALIBRATION.driftShare) {
            return { units, ms: median(recent), spread: spreadOf(timings.slice(-2 * count)) };
        }
    }
    if (timings.length >= CALIBRATION.maxSamplesPerSize) {
        const latest = timings.slice(-(CALIBRATION.agreeSamples + 2));
        return { units, ms: median(latest), spread: spreadOf(latest) };
    }
    return undefined;
}

// ---------------------------------------------------------------------------
// Saturation
// ---------------------------------------------------------------------------

/**
 * When the batch steps down, and how it comes back.
 *
 * Frames late means a step down, whatever the technique's timings read
 * (DECISIONS 2026-11-05, after the budget v2 re-review). The first version
 * stepped down only when the timings were near their calibrated cost (the
 * GPU "busy"), but a thermally throttled RTX 3060 reads about 3× its
 * calibrated cost, so that gate blocked the step-down in exactly the hot
 * state it is for. "Late" already needs the frame to miss its interval, so
 * an idle GPU at low clocks is never late. The cost: a CPU-bound host that
 * misses its interval steps DDGI down too, which doesn't help it; the floor
 * bounds how far.
 *
 * - `windowSeconds`: the frame time is the median over about this long; at
 *   most one step per window.
 * - `lateShare`: frames are late when that median is over the target
 *   interval by more than this share.
 * - `stepDown`: the batch's factor per late window, never below the floor.
 * - `ceilingShare`: after a late episode, recovery stops at this share of
 *   the batch that was late first, below the level that caused it. An
 *   episode runs from the first late window until frames have been on time
 *   for `recoverAfterSeconds`: a load that leaves a window on time now and
 *   then is still one episode. Why (the RTX run of the page's saturation
 *   test): ending the episode at every on-time window let each new late
 *   window set the ceiling again from the batch already stepped down, so
 *   the ceiling ratcheted down (0.9 × 1,569, then 0.9 × 1,255) and the batch
 *   stopped there, ×1.05 per 10 s from full recovery.
 * - `recoverAfterSeconds`, `recovery`: after this long on time, the batch
 *   grows by this factor per window up to that ceiling.
 * - `probeEverySeconds`, `probeStep`: once at the ceiling and on time this
 *   long, the ceiling rises by this factor, back to the calibrated batch.
 * - Without `budget.targetFps`, the target interval is **learned from the
 *   display, and kept while frames are late** (the learned-target rule): it
 *   is the interval of a vsync-like window, and only a later window that is
 *   itself vsync-like *and on time* can change it, to a faster rate. Why
 *   (budget v2 re-review): a target that followed the recent windows forgot
 *   60 Hz a few seconds into a slowdown, counted the 30 Hz frames as on time
 *   and climbed back to the full batch while every frame stayed late. A
 *   vsync host whose frames slow from 60 Hz to 30 Hz can't be told from one
 *   that switched to a 30 Hz display, so at first the slowdown wins. The
 *   rate the floor uses follows the learned target too, so a slowdown
 *   doesn't raise the floor.
 * - `adoptSlowerRateSeconds` (the slower-rate rule): a display rate can drop
 *   for good (Chrome's Energy Saver caps at 30 fps; the window moves to a
 *   60 Hz screen), and the viewer and ddgi-three don't pass `targetFps`.
 *   When the batch has been at the floor and the frames steady and
 *   vsync-like at one slower refresh rate for this long, that rate is
 *   adopted. At the floor DDGI's work is as small as it gets, so it isn't
 *   what keeps the frames at the slower rate: the display rate changed.
 *   10 s is several times what reaching the floor takes, so a short
 *   overload that stops on its own is never mistaken for a new display,
 *   and short enough that DDGI doesn't sit at the floor for long. A real
 *   GPU overload makes unsteady late frames, which aren't vsync-like, so it
 *   never adopts a rate. `GIPlatform.recalibrate()` forgets the learned
 *   rate too. Hosts may still pass `targetFps`; nothing depends on it.
 * - `refreshRates`, `refreshTolerance`, `vsyncJitter`: vsync-like means
 *   within `refreshTolerance` of a common display refresh rate *and* steady:
 *   the window's frame times from the 10th to the 90th percentile span at
 *   most `vsyncJitter` of its median. The refresh list alone matched about
 *   40% of all rates (re-review); a display paces frames to within a
 *   fraction of a millisecond, while an uncapped frame rate wanders with the
 *   view and the work. Without a vsync-like window the host is taken to be
 *   uncapped and has no ceiling (the batch never steps down). A host can
 *   always say `targetFps`, and must when it is late from the start: a vsync
 *   host that never makes 60 Hz looks like an on-time 30 Hz host.
 * - A frame that misses now and then leaves the window's median at the
 *   display interval: not late, no step-down. That is meant: one late frame
 *   in many isn't saturation.
 */
export const SATURATION = {
    windowSeconds: 1,
    lateShare: 0.1,
    stepDown: 0.8,
    ceilingShare: 0.9,
    recoverAfterSeconds: 3,
    recovery: 1.1,
    probeEverySeconds: 10,
    probeStep: 1.05,
    refreshRates: [24, 25, 30, 48, 50, 60, 72, 75, 90, 100, 120, 144, 165],
    refreshTolerance: 0.03,
    vsyncJitter: 0.1,
    adoptSlowerRateSeconds: 10,
} as const;

/** What a full window of frames says. */
export interface SaturationWindow {
    /** Frames were late against the target interval (never, with no target). */
    readonly late: boolean;
    readonly seconds: number;
}

/** One window's frame time: its median and how steady it was. */
interface FrameWindow {
    readonly seconds: number;
    /** (p90 − p10) / median of the window's frame times. */
    readonly jitter: number;
}

/** Paced by a display: steady, at a common refresh rate (SATURATION). */
function isVsyncLike(window: FrameWindow): boolean {
    return window.jitter <= SATURATION.vsyncJitter && SATURATION.refreshRates.some((rate) => Math.abs(window.seconds * rate - 1) <= SATURATION.refreshTolerance);
}

/** Watches the frame time (`FrameInput.deltaTime`) over windows of about a second. */
export class SaturationGuard {
    private frameTimes: number[] = [];
    private windowSeconds = 0;
    /** The display's interval as learned (the learned-target rule, SATURATION); undefined until a vsync-like window. */
    private learnedSeconds: number | undefined;
    /** The fastest window seen, for the floor's frame rate before a display interval is learned. */
    private fastestSeconds: number | undefined;
    /** A slower display rate seen while at the floor, and for how long (the slower-rate rule, SATURATION). */
    private slowerRate: { seconds: number; heldSeconds: number } | undefined;
    /** The last full window, for the stats. */
    framesLate = false;

    /**
     * One frame. Returns the window's verdict when a window is complete.
     * Frames with no time (back to back, `deltaTime` 0) don't count.
     * `atFloor`: the technique's batch is at its floor (for the slower-rate rule).
     */
    addFrame(deltaTime: number, targetFps: number | undefined, atFloor = false): SaturationWindow | undefined {
        if (!(deltaTime > 0) || !Number.isFinite(deltaTime)) return undefined;
        this.frameTimes.push(deltaTime);
        this.windowSeconds += deltaTime;
        if (this.windowSeconds < SATURATION.windowSeconds) return undefined;

        const frameSeconds = median(this.frameTimes);
        const jitter = (quantile(this.frameTimes, 0.9) - quantile(this.frameTimes, 0.1)) / frameSeconds;
        const window: FrameWindow = { seconds: frameSeconds, jitter };
        this.fastestSeconds = Math.min(this.fastestSeconds ?? Number.POSITIVE_INFINITY, frameSeconds);
        // The learned-target rule: a vsync-like window teaches the display's interval when there
        // is none yet, or a faster one when it is on time; a late window never replaces it.
        const onTimeForLearned = this.learnedSeconds === undefined || frameSeconds <= this.learnedSeconds * (1 + SATURATION.lateShare);
        if (isVsyncLike(window) && onTimeForLearned) this.learnedSeconds = Math.min(this.learnedSeconds ?? Number.POSITIVE_INFINITY, frameSeconds);
        // The slower-rate rule: at the floor, steady vsync-like frames at one slower rate for
        // long enough mean the display rate changed; adopt it.
        if (!onTimeForLearned && atFloor && isVsyncLike(window)) {
            const sameRate = this.slowerRate !== undefined && Math.abs(this.slowerRate.seconds / frameSeconds - 1) <= SATURATION.refreshTolerance;
            this.slowerRate = sameRate && this.slowerRate ? { seconds: this.slowerRate.seconds, heldSeconds: this.slowerRate.heldSeconds + this.windowSeconds } : { seconds: frameSeconds, heldSeconds: this.windowSeconds };
            if (this.slowerRate.heldSeconds >= SATURATION.adoptSlowerRateSeconds) {
                this.learnedSeconds = this.slowerRate.seconds;
                this.slowerRate = undefined;
            }
        } else {
            this.slowerRate = undefined;
        }
        const target = this.targetSeconds(targetFps);
        const late = target !== undefined && frameSeconds > target * (1 + SATURATION.lateShare);
        const verdict = { late, seconds: this.windowSeconds };
        this.framesLate = late;
        this.frameTimes = [];
        this.windowSeconds = 0;
        return verdict;
    }

    /**
     * The interval frames should come at: `1 / targetFps`, or the learned
     * display interval (see SATURATION); undefined for an uncapped host (no
     * ceiling).
     */
    targetSeconds(targetFps: number | undefined): number | undefined {
        if (targetFps !== undefined && targetFps > 0) return 1 / targetFps;
        return this.learnedSeconds;
    }

    /**
     * The host's frame rate as far as it is known (for floors given in
     * seconds): `targetFps`, else the learned display rate, else the fastest
     * window seen. Never the rate of a slowdown.
     */
    framesPerSecond(targetFps: number | undefined): number | undefined {
        if (targetFps !== undefined && targetFps > 0) return targetFps;
        const seconds = this.learnedSeconds ?? this.fastestSeconds;
        return seconds === undefined ? undefined : 1 / seconds;
    }

    /** A new calibration: the open window goes; the remembered frame rate stays. */
    startOver(): void {
        this.frameTimes = [];
        this.windowSeconds = 0;
        this.framesLate = false;
    }

    /** `recalibrate()`: forget the learned display rate too; the next vsync-like window teaches it again. */
    forgetRate(): void {
        this.learnedSeconds = undefined;
        this.fastestSeconds = undefined;
        this.slowerRate = undefined;
    }
}

// ---------------------------------------------------------------------------
// One technique's budget
// ---------------------------------------------------------------------------

/** The frame rate the floor assumes before one is known. */
const ASSUMED_FRAMES_PER_SECOND = 60;

/** What a technique's budget needs each frame. */
export interface BudgetFrameInput {
    /** The platform's frame number (timings come back tagged with it). */
    readonly frame: number;
    readonly deltaTime: number;
    readonly budget: FrameBudget;
    /** The technique's share of `budget.gpuMs`; undefined without one. */
    readonly shareMs: number | undefined;
    /** GPU timings will arrive (`gpuTiming` on a device with `timestamp-query`). */
    readonly timed: boolean;
    /** The mesh trees are still being built: the first calibration waits (rays don't see those meshes yet). */
    readonly sceneSettling?: boolean;
}

/**
 * - `units`: no millisecond budget; the hand-out is `budget.units` (or the default).
 * - `not calibrated`: a millisecond budget but no timings, or the trees are still being built (`budget.units` or the default), or the burst gave up (its best estimate, or the floor).
 * - `calibrating`: a burst is running.
 * - `calibrated`: the calibrated batch, the floor, the saturation guard.
 */
export type BudgetState = 'units' | 'not calibrated' | 'calibrating' | 'calibrated';

/** What the budget did, for `stats` and the viewer. */
export interface BudgetReport {
    readonly state: BudgetState;
    /** Units handed out in the latest frame. */
    readonly units: number;
    /** The calibrated batch, before the floor and the saturation step. */
    readonly calibratedUnits: number | undefined;
    /** The floor in the latest frame (0 without one). */
    readonly floorUnits: number;
    /** The batch is below the calibrated number after frames were late. */
    readonly steppedDown: boolean;
    /** Where recovery stops for now (below the batch that was late); undefined when not limited. */
    readonly ceilingUnits: number | undefined;
    /** The latest full window's frames were late. */
    readonly framesLate: boolean;
    /** The geometry changed since the calibration; the host may `recalibrate()`. */
    readonly stale: boolean;
    /** The burst is done, and its batch is being timed over back-to-back frames to confirm it (CALIBRATION). */
    readonly confirming: boolean;
    /** The frame rate the floor uses; undefined until known. */
    readonly framesPerSecond: number | undefined;
    readonly calibration: Calibration | undefined;
}

/** What the calibration guard compares with (CALIBRATION `guardShare`): the previous calibration on this device and scene. */
export interface CalibrationMemory {
    /** The previous calibration, if any. */
    previous(): Calibration | undefined;
    /** A calibration was accepted: it becomes the previous one. */
    remember(calibration: Calibration): void;
}

/**
 * One technique's units per frame under budget v2 (see the file comment).
 * The platform keeps one per attached technique and makes a new one on
 * `prepare` and when techniques are added or removed.
 */
export class WorkBudget {
    private readonly technique: Pick<GITechnique, 'msPerUnit' | 'minUnitsPerFrame'>;
    private readonly guard = new SaturationGuard();
    private calibration: Calibration | undefined;
    private burst: CalibrationBurst | undefined;
    /** Filler batches are allowed: the platform's first calibration, and none has finished yet. */
    private fillerAllowed: boolean;
    private recalibrateRequested = false;
    private stale = false;
    /** The geometry changed while a burst ran: its calibration is stale from the start. */
    private geometryChangedDuringBurst = false;
    /** The batch now: the calibrated one, or less while stepped down. */
    private current = 0;
    /** Recovery stops here after a step-down (AIMD); undefined when not limited. */
    private ceiling: number | undefined;
    private onTimeSeconds = 0;
    private atCeilingSeconds = 0;
    /** Inside a late episode (see SATURATION `ceilingShare`): the ceiling was set at its start. */
    private lateEpisode = false;
    /** Timings of frames up to this one belong to the last burst. */
    private calibratedAfterFrame = 0;
    /** The platform frame of the latest hand-out. */
    private latestFrame = 0;
    /** After a burst: back-to-back frames whose timings confirm it, and their measured / calibrated ratios. */
    private confirming = false;
    private readonly confirmFrames = new Set<number>();
    private confirmRatios: number[] = [];
    /** Frames handed out while confirming: it ends after twice `confirmFrames` even if timings went missing. */
    private confirmHandedFrames = 0;
    /** The calibration guard: the previous calibration on this device and scene, and the batch of a first burst being re-measured. */
    private readonly memory: CalibrationMemory | undefined;
    private remeasuring: number | undefined;
    private readonly warn: ((message: string) => void) | undefined;
    private latest: BudgetReport = {
        state: 'units',
        units: 0,
        calibratedUnits: undefined,
        floorUnits: 0,
        steppedDown: false,
        ceilingUnits: undefined,
        framesLate: false,
        stale: false,
        confirming: false,
        framesPerSecond: undefined,
        calibration: undefined,
    };

    /**
     * `technique`: its own cost estimate (`msPerUnit`, which replaces the burst) and its floor.
     * `fillerAllowed`: false for a technique added while the host runs live (no filler batches
     * in its frames); the platform allows them only before its first calibration.
     */
    constructor(
        technique: Pick<GITechnique, 'msPerUnit' | 'minUnitsPerFrame'> = { msPerUnit: undefined },
        options: { readonly fillerAllowed?: boolean; readonly memory?: CalibrationMemory; readonly warn?: (message: string) => void } = {},
    ) {
        this.technique = technique;
        this.fillerAllowed = options.fillerAllowed ?? true;
        this.memory = options.memory;
        this.warn = options.warn;
    }

    get report(): BudgetReport {
        return this.latest;
    }

    /** The cost line of the current calibration, if one was measured. */
    get model(): UnitCostModel | undefined {
        return this.calibration?.model;
    }

    /** Units the technique may spend this frame. */
    unitsFor(input: BudgetFrameInput): number {
        const { budget, shareMs } = input;
        this.latestFrame = input.frame;
        const cap = budget.units;
        const capped = (units: number) => (cap === undefined ? units : Math.min(units, cap));
        if (shareMs === undefined) return this.handOut('units', cap ?? DEFAULT_UNITS_PER_FRAME, 0);
        // A zero (or nonsense) share asks for no work, floor or not.
        if (!(shareMs > 0)) return this.handOut('units', 0, 0);

        const ownMsPerUnit = this.technique.msPerUnit;
        if (ownMsPerUnit !== undefined && ownMsPerUnit > 0) {
            if (!this.calibration || this.calibration.shareMs !== shareMs || this.calibration.cap !== cap) this.adopt(proportionalCalibration(shareMs, cap, ownMsPerUnit), false);
        } else if (!input.timed) {
            return this.handOut('not calibrated', cap ?? DEFAULT_UNITS_PER_FRAME, 0);
        } else {
            if (this.recalibrateRequested || (!this.calibration && !this.burst)) {
                // The first calibration waits for the mesh trees: rays wouldn't see those meshes yet.
                if (!this.calibration && input.sceneSettling) return this.handOut('not calibrated', cap ?? DEFAULT_UNITS_PER_FRAME, 0);
                this.recalibrateRequested = false;
                this.geometryChangedDuringBurst = false;
                this.burst = new CalibrationBurst(shareMs, cap, this.fillerAllowed);
                this.guard.startOver();
            }
            const burst = this.burst;
            if (burst) {
                const units = burst.next(input.frame, input.deltaTime);
                if (!burst.result) return this.handOut('calibrating', capped(units), 0);
                // A burst that gave up finishes inside `next`: on to the hand-out below.
                if (!this.finishBurst(burst.result)) return this.handOut('calibrating', capped(units), 0);
            }
        }

        let calibration = this.calibration as Calibration;
        if (!calibration.measured) {
            // The burst gave up: its best estimate, at least the floor. The floor is capped at twice
            // that estimate, as after a calibration (DDGI's 'auto'); only when nothing was measured
            // at all is it uncapped (calibratedUnits Infinity). With neither, the default.
            const framesPerSecond = this.guard.framesPerSecond(budget.targetFps) ?? ASSUMED_FRAMES_PER_SECOND;
            const floorUnits = this.floorUnits(framesPerSecond, calibration.units > 0 ? calibration.units : Number.POSITIVE_INFINITY);
            const units = Math.max(calibration.units, floorUnits);
            return this.handOut('not calibrated', capped(units > 0 ? units : DEFAULT_UNITS_PER_FRAME), floorUnits);
        }
        // A new share or cap: solved from the measured curve, no burst.
        if (calibration.shareMs !== shareMs || calibration.cap !== cap) {
            this.adopt(resolve(calibration, shareMs, cap), false);
            calibration = this.calibration as Calibration;
        }

        // Back-to-back frames after a burst confirm it (CALIBRATION `confirmFrames`). The first
        // frame with time between frames (the host left its loading screen), or twice as many
        // frames as it needs, ends the confirmation unconfirmed: it never stays open.
        if (this.confirming) {
            this.confirmHandedFrames++;
            if (input.deltaTime > 0 || this.confirmHandedFrames > 2 * CALIBRATION.confirmFrames) this.endConfirmation();
            else this.confirmFrames.add(input.frame);
        }
        const framesPerSecond = this.guard.framesPerSecond(budget.targetFps);
        const floorUnits = this.floorUnits(framesPerSecond ?? ASSUMED_FRAMES_PER_SECOND, calibration.units);
        const window = this.guard.addFrame(input.deltaTime, budget.targetFps, this.current <= floorUnits);
        if (window) this.applyWindow(window, calibration.units, floorUnits);
        const units = Math.max(floorUnits, this.current);
        return this.handOut('calibrated', capped(units), floorUnits, framesPerSecond);
    }

    /** After the technique's encode: what it was handed and what it used. */
    record(handed: number, used: number): void {
        const burst = this.burst;
        if (!burst) return;
        burst.used(handed, used);
        if (burst.result) this.finishBurst(burst.result);
    }

    /** A timed frame of this technique. */
    addSample(sample: CostSample): void {
        const burst = this.burst;
        if (burst) {
            burst.add(sample);
            if (burst.result) this.finishBurst(burst.result);
            return;
        }
        const calibration = this.calibration;
        if (!calibration?.measured || sample.frame <= this.calibratedAfterFrame) return;
        const calibratedMs = costOnCurve(calibration.curve, sample.units);
        if (calibratedMs !== undefined && this.confirming && this.confirmFrames.delete(sample.frame)) this.confirm(calibration, sample.ms / calibratedMs);
    }

    private endConfirmation(): void {
        this.confirming = false;
        this.confirmFrames.clear();
        this.confirmRatios = [];
    }

    /** One back-to-back timing after the burst; once there are enough, scale the curve if it is off. */
    private confirm(calibration: Calibration, ratio: number): void {
        this.confirmRatios.push(ratio);
        if (this.confirmRatios.length < CALIBRATION.confirmFrames) return;
        const factor = median(this.confirmRatios.slice(-CALIBRATION.confirmJudgedFrames));
        this.endConfirmation();
        if (Math.abs(factor - 1) <= CALIBRATION.confirmTolerance) {
            this.calibration = { ...calibration, confirmFactor: 1 };
            return;
        }
        const curve = calibration.curve.map((point) => ({ ...point, ms: point.ms * factor }));
        const scaled = resolve({ ...calibration, curve }, calibration.shareMs, calibration.cap);
        this.adopt({ ...scaled, confirmFactor: factor }, false);
    }

    /** The scene's geometry changed: the calibration is stale (reported). The batch stays until the host re-calibrates. */
    geometryChanged(): void {
        if (this.calibration) this.stale = true;
        // A burst running now measures a scene that is changing: stale once it ends.
        if (this.burst) this.geometryChangedDuringBurst = true;
    }

    /** Calibrate again at the next millisecond frame (`GIPlatform.recalibrate`). */
    recalibrate(): void {
        this.recalibrateRequested = true;
        // The display rate may have changed too (the slower-rate rule, SATURATION): learn it again.
        this.guard.forgetRate();
    }

    /** One window of the saturation guard: step down, hold, or recover (AIMD, see SATURATION). */
    private applyWindow(window: SaturationWindow, calibratedUnits: number, floorUnits: number): void {
        if (window.late) {
            // Late: step down, whatever the technique's timings read (see SATURATION for why
            // there is no clock test), never below the floor.
            this.onTimeSeconds = 0;
            this.atCeilingSeconds = 0;
            // The ceiling comes from the batch that was late first, not from the steps after it.
            if (!this.lateEpisode) this.ceiling = Math.max(floorUnits, Math.floor(this.current * SATURATION.ceilingShare));
            this.lateEpisode = true;
            this.current = Math.max(floorUnits, Math.floor(this.current * SATURATION.stepDown));
            return;
        }
        this.onTimeSeconds += window.seconds;
        if (this.onTimeSeconds < SATURATION.recoverAfterSeconds) return;
        // On time long enough: the episode is over, and recovery starts.
        this.lateEpisode = false;
        if (this.current >= calibratedUnits) return;
        const limit = Math.min(calibratedUnits, this.ceiling ?? calibratedUnits);
        if (this.current < limit) {
            // + 1 so a batch of a few units still grows.
            this.current = Math.min(limit, Math.max(this.current + 1, Math.ceil(this.current * SATURATION.recovery)));
            return;
        }
        // At the ceiling and on time: probe a little higher now and then.
        this.atCeilingSeconds += window.seconds;
        if (this.ceiling !== undefined && this.atCeilingSeconds >= SATURATION.probeEverySeconds) {
            this.atCeilingSeconds = 0;
            const raised = Math.max(this.ceiling + 1, Math.ceil(this.ceiling * SATURATION.probeStep));
            this.ceiling = raised >= calibratedUnits ? undefined : raised;
            this.current = Math.min(calibratedUnits, raised);
        }
    }

    /**
     * A burst ended: the calibration guard (CALIBRATION `guardShare`), then
     * the result is taken. Returns false when it starts a re-measurement
     * instead: a measured batch under half of the previous calibration's on
     * this device and scene, the first time. The second result is taken
     * whatever it is, and both are reported (`remeasuredFrom`).
     */
    private finishBurst(result: Calibration): boolean {
        const previous = result.measured ? this.memory?.previous() : undefined;
        if (previous?.measured && this.remeasuring === undefined) {
            const expected = unitsOnCurve(previous.curve, result.shareMs, Math.min(result.capacity, result.cap ?? Number.POSITIVE_INFINITY));
            if (result.units < CALIBRATION.guardShare * expected) {
                this.remeasuring = result.units;
                this.warn?.(
                    `@pmndrs/gi: the GPU budget's calibration came out at ${result.units} units a frame, under half of the previous calibration on this device and scene (${expected}); measuring once more. Its timings: ${describeSamples(result.samples)}`,
                );
                this.burst = new CalibrationBurst(result.shareMs, result.cap, false);
                return false;
            }
        }
        const taken = this.remeasuring === undefined ? result : { ...result, remeasuredFrom: this.remeasuring };
        this.remeasuring = undefined;
        this.adopt(taken, true);
        if (taken.measured) this.memory?.remember(taken);
        return true;
    }

    /**
     * Takes a calibration: `measuredNow` from a burst that just ended (every frame
     * handed out so far was a burst frame, whose timings may still arrive), or
     * else solved again from one.
     */
    private adopt(calibration: Calibration, measuredNow: boolean): void {
        this.calibration = calibration;
        this.burst = undefined;
        this.current = calibration.units;
        this.ceiling = undefined;
        this.onTimeSeconds = 0;
        this.atCeilingSeconds = 0;
        this.lateEpisode = false;
        if (measuredNow) {
            this.calibratedAfterFrame = this.latestFrame;
            this.fillerAllowed = false;
            this.stale = this.geometryChangedDuringBurst;
            this.geometryChangedDuringBurst = false;
            // A measured burst is confirmed over back-to-back frames, if the host runs any.
            this.endConfirmation();
            this.confirming = calibration.measured && calibration.frames > 0;
            this.confirmHandedFrames = 0;
        }
        this.guard.startOver();
    }

    private floorUnits(framesPerSecond: number, calibratedUnits: number): number {
        const units = this.technique.minUnitsPerFrame?.({ framesPerSecond, calibratedUnits });
        return units !== undefined && Number.isFinite(units) && units > 0 ? Math.floor(units) : 0;
    }

    private handOut(state: BudgetState, units: number, floorUnits: number, framesPerSecond?: number): number {
        const calibration = this.calibration;
        this.latest = {
            state,
            units,
            calibratedUnits: calibration?.measured ? calibration.units : undefined,
            floorUnits,
            steppedDown: state === 'calibrated' && calibration !== undefined && this.current < calibration.units,
            ceilingUnits: this.ceiling,
            framesLate: this.guard.framesLate,
            stale: this.stale,
            confirming: this.confirming,
            framesPerSecond: framesPerSecond ?? this.latest.framesPerSecond,
            calibration,
        };
        return units;
    }
}

/** A burst's timings in a line, for the calibration guard's warning: `units: ms` in arrival order. */
function describeSamples(samples: readonly CostSample[]): string {
    return samples.map((sample) => `${sample.units}: ${sample.ms.toFixed(3)} ms`).join(', ');
}

/** A calibration from the technique's own cost per unit (no burst). */
function proportionalCalibration(shareMs: number, cap: number | undefined, msPerUnit: number): Calibration {
    const point: CurvePoint = { units: 1, ms: msPerUnit, spread: 0 };
    const units = Math.max(1, Math.floor(shareMs / msPerUnit));
    return {
        units: cap === undefined ? units : Math.min(units, cap),
        shareMs,
        cap,
        model: { fixedMs: 0, msPerUnit },
        curve: [point],
        capacity: Number.POSITIVE_INFINITY,
        frames: 0,
        spread: 0,
        fillerFrames: 0,
        fillerMs: 0,
        measured: true,
        samples: [],
    };
}

/** The same measurements solved for a new share (a new `gpuMs`) or cap: no burst. */
function resolve(calibration: Calibration, shareMs: number, cap: number | undefined): Calibration {
    const units = unitsOnCurve(calibration.curve, shareMs, Math.min(calibration.capacity, cap ?? Number.POSITIVE_INFINITY));
    return { ...calibration, units, shareMs, cap, model: modelAround(calibration.curve, units) };
}

/** Nearest-rank quantile. */
function quantile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(share * sorted.length) - 1))];
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
