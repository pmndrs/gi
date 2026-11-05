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
 * 1. **Calibration** ({@link CalibrationBurst}): with the GPU loaded (its
 *    clocks up), measure what batches cost and take the largest batch whose
 *    cost fits the technique's share of `budget.gpuMs`. Done once, and again
 *    after a new `prepare`, a geometry change, a new `gpuMs` or on request.
 * 2. **Each frame hands out that batch.** Nothing is steered on GPU
 *    milliseconds while frames arrive on time: a slow-looking pass on an
 *    idle GPU is low clocks, not a real cost.
 * 3. **The floor** ({@link TechniqueWorkFloor}): the technique may ask for a
 *    minimum (DDGI: a full sweep within 4 s), which wins over the millisecond
 *    ceiling so a weak device runs slower, never frozen.
 * 4. **Saturation** ({@link SaturationGuard}): only when frames are late
 *    *and* the technique's measured time is over its share does the batch
 *    step down (×0.8 a second, never below the floor); on-time frames bring
 *    it back (×1.1 a second) to the calibrated number. When frames are late
 *    the GPU is busy and its clocks are up, so the milliseconds then mean
 *    what they say.
 *
 * `budget.units` caps everything, the burst included (FrameBudget).
 */

import type { FrameBudget } from './contract/frame.ts';

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
 * then about 6 µs per probe more. Budget v2 reports the line it calibrated
 * from (see {@link Calibration.model}).
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
// The floor
// ---------------------------------------------------------------------------

/** What a technique's floor depends on. */
export interface WorkFloorRate {
    /** The frame rate the host runs at: `budget.targetFps`, or the rate observed from `deltaTime` (60 until known). */
    readonly framesPerSecond: number;
    /** The batch the budget buys with the clocks up (the calibration). */
    readonly calibratedUnits: number;
}

/**
 * Optional for a technique: the fewest units it should get per frame under a
 * millisecond budget (budget v2's floor). The floor wins over the millisecond
 * ceiling; `budget.units` still caps it. DDGI implements it from its
 * `minProbesPerFrame` setting.
 *
 * Not in the technique contract (contract/technique.ts) yet: the platform
 * looks for the method by name ({@link workFloorOf}). Proposed for the
 * contract in PROGRESS.
 */
export interface TechniqueWorkFloor {
    minUnitsPerFrame(rate: WorkFloorRate): number;
}

/** The technique's floor, if it has one. */
export function workFloorOf(technique: object): TechniqueWorkFloor | undefined {
    const candidate = technique as Partial<TechniqueWorkFloor>;
    return typeof candidate.minUnitsPerFrame === 'function' ? (technique as TechniqueWorkFloor) : undefined;
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/**
 * How a calibration burst measures (see {@link CalibrationBurst}).
 *
 * - `firstUnits`: the first batch size tried.
 * - `rampFrames`: burst frames at a large batch before any timing counts.
 *   The GPU raises its clocks under load; the first frames of a burst on an
 *   idle GPU would be timed at low clocks (T1.9 quiet window: Sponza took
 *   about 2× as long at 60 Hz with a light load as back to back).
 * - `samplesPerSize`: timed frames per batch size; its cost is their median,
 *   so one frame slowed by another program's GPU work changes nothing.
 * - `ladderStep`: batch sizes step by this factor while looking for the two
 *   sizes on either side of the budget; `smallestUnits` is where it stops.
 * - `heavyMs`: GPU time of the filler batch run between measured batches
 *   smaller than half of it, so a host at 60 Hz with a light load still
 *   keeps the GPU busy and its clocks up while small batches are timed.
 *   The filler frames are real updates.
 * - `verifyRounds`, `verifyTolerance`: the batch solved from the two sizes
 *   around the budget is measured itself, up to `verifyRounds` times; it is
 *   accepted within `verifyTolerance` of the budget, otherwise solved again
 *   with the new point. Why: between two sizes four times apart the cost
 *   isn't a straight line (latency-bound small batches, T1.9a), so a
 *   straight line through them can be off by 10% or more.
 * - `maxFrames`: a burst still running after this many frames gives up
 *   (timings aren't arriving) and the hand-out falls back to `budget.units`,
 *   or {@link DEFAULT_UNITS_PER_FRAME}.
 */
export const CALIBRATION = {
    firstUnits: 1024,
    rampFrames: 6,
    samplesPerSize: 3,
    ladderStep: 4,
    smallestUnits: 4,
    heavyMs: 12,
    verifyRounds: 3,
    verifyTolerance: 0.05,
    maxFrames: 400,
} as const;

/** What a calibration found. */
export interface Calibration {
    /** The batch the technique's share buys with the clocks up: `floor((shareMs − fixedMs) / msPerUnit)` on the line around it. */
    readonly units: number;
    /** The technique's share of `budget.gpuMs` it was calibrated for. */
    readonly shareMs: number;
    /** `budget.units` when it was calibrated: a cap changes which sizes can be measured. */
    readonly cap: number | undefined;
    /**
     * The cost line the batch came from: through the two measured sizes on
     * either side of it (a local fit; the curve bends below about 100 probes
     * on large scenes). Undefined when nothing was measured.
     */
    readonly model: UnitCostModel | undefined;
    /** The median time between burst frames the host reported, seconds; 0 when it gave none (frames run back to back). */
    readonly frameSeconds: number;
    /** Frames the burst took. */
    readonly frames: number;
    /** Every batch size measured with the clocks up and its timings, so a new `gpuMs` can be solved again with a short burst. */
    readonly samples: ReadonlyMap<number, readonly number[]>;
    /** False when the burst gave up without timings; `units` is then the fallback. */
    readonly measured: boolean;
}

interface CurvePoint {
    readonly units: number;
    readonly ms: number;
}

/**
 * One calibration: a short run of frames ("burst") whose batch sizes the
 * platform picks, timed with the GPU loaded.
 *
 * How it runs, frame by frame (what {@link next} hands out):
 * 1. `rampFrames` frames of a large batch, untimed: the clocks come up.
 * 2. `firstUnits`, until `samplesPerSize` timings are in. Its cost sets the
 *    filler batch (about `heavyMs`).
 * 3. Down (or up) by `ladderStep` until two measured sizes sit on either
 *    side of the share. Batches smaller than half the filler alternate with
 *    the filler, so the GPU stays loaded whatever the host's frame rate.
 * 4. The batch on the line between those two is measured itself and
 *    accepted, or solved again (`verifyRounds`).
 *
 * With a host that runs its warm-up back to back (the viewer, the batch-cost
 * page), the burst is part of the warm-up: about 30-60 frames. At 60 Hz it
 * takes about a second, with the filler frames heavier than usual.
 */
export class CalibrationBurst {
    /** The technique's share of `budget.gpuMs` this burst solves for. */
    readonly shareMs: number;
    /** `budget.units` when it started. */
    readonly cap: number | undefined;
    /** Timings by the units the technique used (a re-started burst takes them over). */
    private readonly timings: Map<number, number[]>;
    /** The most units the technique can use: the cap, or less once it used fewer than handed. */
    private limit: number;
    /** The batch size being measured. */
    private probe: number;
    /** The filler batch, once a cost is known. */
    private heavy: number | undefined;
    private verifying = false;
    private verifyRoundsUsed = 0;
    private frames = 0;
    /** Timings of frames up to this platform frame are ramp frames (or older) and don't count. */
    private rampUntilFrame: number | undefined;
    private fillerThisFrame = false;
    private readonly frameSeconds: number[] = [];
    private finished: Calibration | undefined;

    /** `seed`: timings from an earlier calibration of the same work (a new `gpuMs` re-solves from them). */
    constructor(shareMs: number, cap: number | undefined, seed?: ReadonlyMap<number, readonly number[]>) {
        this.shareMs = shareMs;
        this.cap = cap;
        this.limit = cap ?? Number.POSITIVE_INFINITY;
        this.timings = new Map([...(seed ?? new Map<number, readonly number[]>())].map(([units, list]) => [units, [...list]]));
        this.probe = Math.min(CALIBRATION.firstUnits, this.limit);
        const largest = this.curve().at(-1);
        if (largest) {
            this.heavy = this.fillerFrom(largest);
            // The seeded sizes are measured already: solve from them straight away,
            // so the burst only has to time the new answer.
            this.probe = largest.units;
            this.decide();
        }
    }

    /** The calibration, once the burst is done. */
    get result(): Calibration | undefined {
        return this.finished;
    }

    /** Timings so far, by the units the technique used. */
    get samples(): ReadonlyMap<number, readonly number[]> {
        return this.timings;
    }

    /** Units to hand out in burst frame `frame` (the platform's frame number). */
    next(frame: number, deltaTime: number): number {
        this.frames++;
        if (deltaTime >= 0 && Number.isFinite(deltaTime)) this.frameSeconds.push(deltaTime);
        if (this.frames > CALIBRATION.maxFrames && !this.finished) {
            this.finished = this.calibration(this.cap ?? DEFAULT_UNITS_PER_FRAME, false);
            return this.finished.units;
        }
        if (this.frames <= CALIBRATION.rampFrames) {
            this.rampUntilFrame = frame;
            return Math.min(this.heavy ?? CALIBRATION.firstUnits, this.limit);
        }
        const filler = this.heavy !== undefined && this.probe < this.heavy / 2;
        this.fillerThisFrame = filler && !this.fillerThisFrame;
        return this.fillerThisFrame ? (this.heavy as number) : this.probe;
    }

    /** What the technique used of a burst hand-out: fewer means that is all it can use. */
    used(handed: number, used: number): void {
        if (!(used < handed)) return;
        this.limit = Math.min(this.limit, Math.max(0, used));
        if (this.probe > this.limit) this.probe = this.limit;
        if (this.heavy !== undefined && this.heavy > this.limit) this.heavy = this.limit;
        if (this.limit === 0 && !this.finished) this.finished = this.calibration(0, true);
    }

    /** A timed frame. Frames from before the ramp ended don't count. */
    add(sample: CostSample): void {
        if (this.finished || this.rampUntilFrame === undefined || sample.frame <= this.rampUntilFrame) return;
        if (!(sample.units > 0) || !(sample.ms >= 0)) return;
        let list = this.timings.get(sample.units);
        if (!list) {
            list = [];
            this.timings.set(sample.units, list);
        }
        list.push(sample.ms);
        this.decide();
    }

    /** After the batch being measured has enough timings: the next size, or the answer. */
    private decide(): void {
        const share = this.shareMs;
        // Bounded: each pass either waits for timings, finishes, or moves to a new size.
        for (let guard = 0; guard < 64 && !this.finished; guard++) {
            const probeMs = this.medianAt(this.probe);
            if (probeMs === undefined) return;
            this.heavy ??= this.fillerFrom({ units: this.probe, ms: probeMs });

            if (this.verifying) {
                this.verifying = false;
                const tolerance = CALIBRATION.verifyTolerance;
                if (probeMs <= share * (1 + tolerance) && probeMs >= share * (1 - tolerance)) {
                    this.finish(this.probe);
                    return;
                }
                // Off by more than the tolerance: solve again with this point in the curve.
            }

            const curve = this.curve();
            const fitting = curve.filter((point) => point.ms <= share);
            const under = fitting.at(-1);
            const over = curve.find((point) => point.ms > share && (under === undefined || point.units > under.units));
            if (under === undefined) {
                // Every size measured costs more than the share: go smaller, down to `smallestUnits`.
                const smallest = curve[0];
                if (smallest.units <= CALIBRATION.smallestUnits) {
                    this.finish(1);
                    return;
                }
                this.probe = Math.max(CALIBRATION.smallestUnits, Math.floor(smallest.units / CALIBRATION.ladderStep));
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
            const candidate = solveBetween(under, over, share);
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
        return {
            units: limited,
            shareMs: this.shareMs,
            cap: this.cap,
            model: measured ? modelAround(this.curve(), limited) : undefined,
            frameSeconds: this.frameSeconds.length > 0 ? median(this.frameSeconds) : 0,
            frames: this.frames,
            samples: this.timings,
            measured,
        };
    }

    /** About `heavyMs` of work, judged proportionally from one measured size. */
    private fillerFrom(point: CurvePoint): number {
        const units = point.ms > 0 ? Math.round((point.units * CALIBRATION.heavyMs) / point.ms) : point.units;
        return Math.max(1, Math.min(this.limit, units));
    }

    /** The cost at `units`: the median of its timings, once there are `samplesPerSize`. */
    private medianAt(units: number): number | undefined {
        const list = this.timings.get(units);
        return list && list.length >= CALIBRATION.samplesPerSize ? median(list) : undefined;
    }

    /** Every size with enough timings (and within the limit), smallest first. */
    private curve(): CurvePoint[] {
        const points: CurvePoint[] = [];
        for (const units of this.timings.keys()) {
            const ms = this.medianAt(units);
            if (ms !== undefined && units <= this.limit) points.push({ units, ms });
        }
        return points.sort((a, b) => a.units - b.units);
    }
}

/** The largest batch on the line through `under` and `over` that costs at most `share`, strictly below `over`. */
function solveBetween(under: CurvePoint, over: CurvePoint, share: number): number {
    const msPerUnit = (over.ms - under.ms) / (over.units - under.units);
    const units = under.units + Math.floor((share - under.ms) / msPerUnit);
    return Math.max(under.units, Math.min(over.units - 1, units));
}

/**
 * The cost line reported for a batch of `units` (fixed + per unit): a
 * least-squares line through the measured sizes within a ladder step of it
 * (median timings), or the two nearest when fewer are that close.
 * Proportional (no fixed part) with one size, or when noise makes the line
 * fall. Only reported: the batch itself comes from {@link solveBetween} and
 * its own measurement.
 */
function modelAround(curve: readonly CurvePoint[], units: number): UnitCostModel | undefined {
    if (curve.length === 0) return undefined;
    const byDistance = [...curve].sort((a, b) => Math.abs(Math.log(a.units / units)) - Math.abs(Math.log(b.units / units)));
    const near = curve.filter((point) => point.units * CALIBRATION.ladderStep >= units && point.units <= units * CALIBRATION.ladderStep);
    const points = near.length >= 2 ? near : byDistance.slice(0, 2);
    const meanUnits = points.reduce((sum, point) => sum + point.units, 0) / points.length;
    const meanMs = points.reduce((sum, point) => sum + point.ms, 0) / points.length;
    let covariance = 0;
    let variance = 0;
    for (const point of points) {
        covariance += (point.units - meanUnits) * (point.ms - meanMs);
        variance += (point.units - meanUnits) ** 2;
    }
    const msPerUnit = variance > 0 ? covariance / variance : 0;
    if (!(msPerUnit > 0)) {
        const nearest = byDistance[0];
        return { fixedMs: 0, msPerUnit: nearest.ms / nearest.units };
    }
    return { fixedMs: meanMs - msPerUnit * meanUnits, msPerUnit };
}

// ---------------------------------------------------------------------------
// Saturation
// ---------------------------------------------------------------------------

/**
 * When the batch steps down: the frame rate, not GPU milliseconds, says the
 * GPU is saturated.
 *
 * - `windowSeconds`: the frame time is the median over about this long, and
 *   a step is taken at most once per window;
 * - `lateShare`: frames are late when that median is over the target
 *   interval by more than this share;
 * - `overShare`: the technique is over its share when the median of its
 *   timings in the window is over the share by more than this. The
 *   calibrated batch costs the share give or take a few percent, so without
 *   a margin noise alone would decide; 10% is the tolerance "the budget
 *   holds" already allows (docs/design/ddgi.md, "Defined measurements");
 * - `stepDown` / `recovery`: the batch's factor per window while saturated
 *   (late and over the share) / while on time, back up to the calibrated
 *   batch;
 * - `targetMemorySeconds`: without `budget.targetFps`, the target interval is
 *   the shortest per-window median of this long. It remembers the rate the
 *   host reached when it was on time (vsync), so a few late seconds don't
 *   become the new target.
 */
export const SATURATION = {
    windowSeconds: 1,
    lateShare: 0.1,
    overShare: 0.1,
    stepDown: 0.8,
    recovery: 1.1,
    targetMemorySeconds: 10,
} as const;

/** What a full window of frames says. */
export type SaturationVerdict = 'step down' | 'recover' | 'hold';

/**
 * Watches the frame time (`FrameInput.deltaTime`) and the technique's
 * measured GPU time over windows of about a second.
 */
export class SaturationGuard {
    private frameTimes: number[] = [];
    private windowSeconds = 0;
    private updateMs: number[] = [];
    /** Per-window median frame times, the latest last, for the observed target. */
    private windowMedians: number[] = [];
    /** The last full window's verdict inputs, for the stats. */
    framesLate = false;
    frameSeconds: number | undefined;
    targetSeconds: number | undefined;

    /** A timing of the technique's work in a normal (not burst) frame. */
    addUpdateMs(ms: number): void {
        if (ms >= 0) this.updateMs.push(ms);
    }

    /**
     * One frame. Returns a verdict when a window is complete. Frames with no
     * time (back to back, `deltaTime` 0) don't count.
     */
    addFrame(deltaTime: number, targetFps: number | undefined, shareMs: number): SaturationVerdict | undefined {
        if (!(deltaTime > 0) || !Number.isFinite(deltaTime)) return undefined;
        this.frameTimes.push(deltaTime);
        this.windowSeconds += deltaTime;
        if (this.windowSeconds < SATURATION.windowSeconds) return undefined;

        const frameSeconds = median(this.frameTimes);
        this.windowMedians.push(frameSeconds);
        const remembered = Math.max(1, Math.round(SATURATION.targetMemorySeconds / SATURATION.windowSeconds));
        if (this.windowMedians.length > remembered) this.windowMedians.shift();
        const target = targetFps !== undefined && targetFps > 0 ? 1 / targetFps : Math.min(...this.windowMedians);
        const late = frameSeconds > target * (1 + SATURATION.lateShare);
        const over = this.updateMs.length > 0 && median(this.updateMs) > shareMs * (1 + SATURATION.overShare);

        this.frameSeconds = frameSeconds;
        this.targetSeconds = target;
        this.framesLate = late;
        this.frameTimes = [];
        this.updateMs = [];
        this.windowSeconds = 0;
        if (late && over) return 'step down';
        if (!late) return 'recover';
        return 'hold';
    }

    /** The host's frame rate as far as it is known: `targetFps`, else the observed target. */
    framesPerSecond(targetFps: number | undefined): number | undefined {
        if (targetFps !== undefined && targetFps > 0) return targetFps;
        return this.windowMedians.length > 0 ? 1 / Math.min(...this.windowMedians) : undefined;
    }

    /** A new calibration: the open window and its timings go; the remembered frame rate stays. */
    startOver(): void {
        this.frameTimes = [];
        this.updateMs = [];
        this.windowSeconds = 0;
        this.framesLate = false;
    }
}

// ---------------------------------------------------------------------------
// One technique's budget
// ---------------------------------------------------------------------------

/**
 * When a geometry change re-calibrates: once the geometry has been still for
 * `stillSeconds`, and at most once per `minSpacingSeconds` (the first one at
 * any time). Objects that move now and then would otherwise re-run the burst,
 * with its heavier frames, every few seconds; one that moves every frame
 * never lets it run. Mesh trees still being built count as a change (the
 * platform reports them every frame until they are in): a host that starts
 * before they are built is calibrated again a second after.
 */
export const GEOMETRY_RECALIBRATION = {
    stillSeconds: 1,
    minSpacingSeconds: 10,
} as const;

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
}

/**
 * - `units`: no millisecond budget; the hand-out is `budget.units` (or the default).
 * - `not calibrated`: a millisecond budget but no timings (or the burst gave up); the fallback.
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
    /** The batch is below the calibrated number because frames were late with the technique over its share. */
    readonly steppedDown: boolean;
    /** The latest full window's frames were late. */
    readonly framesLate: boolean;
    /** The frame rate the floor and the guard use; undefined until known. */
    readonly framesPerSecond: number | undefined;
    readonly calibration: Calibration | undefined;
}

/**
 * One technique's units per frame under budget v2 (see the file comment).
 * The platform keeps one per attached technique and drops it on `prepare`
 * and when techniques are added or removed.
 */
export class WorkBudget {
    private readonly floor: TechniqueWorkFloor | undefined;
    private readonly ownMsPerUnit: () => number | undefined;
    private readonly guard = new SaturationGuard();
    private calibration: Calibration | undefined;
    private burst: CalibrationBurst | undefined;
    /** The batch now: the calibrated one, or less while stepped down. */
    private current = 0;
    /** Timings of frames up to this one belong to the last burst. */
    private calibratedAfterFrame = 0;
    /** The platform frame of the latest hand-out. */
    private latestFrame = 0;
    private recalibrateRequested = false;
    /** Seconds the geometry has been still since a change that is still to be calibrated for; undefined when none is due. */
    private geometryStillSeconds: number | undefined;
    /** Seconds since the last re-calibration a geometry change started (none yet: any time). */
    private secondsSinceGeometryCalibration = Number.POSITIVE_INFINITY;
    private latest: BudgetReport = {
        state: 'units',
        units: 0,
        calibratedUnits: undefined,
        floorUnits: 0,
        steppedDown: false,
        framesLate: false,
        framesPerSecond: undefined,
        calibration: undefined,
    };

    /** `ownMsPerUnit`: the technique's own cost estimate (`GITechnique.msPerUnit`), which replaces the burst when given. */
    constructor(floor: TechniqueWorkFloor | undefined, ownMsPerUnit: () => number | undefined = () => undefined) {
        this.floor = floor;
        this.ownMsPerUnit = ownMsPerUnit;
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
        if (shareMs === undefined) return this.handOut('units', cap ?? DEFAULT_UNITS_PER_FRAME, 0);
        // A zero (or nonsense) share asks for no work, floor or not.
        if (!(shareMs > 0)) return this.handOut('units', 0, 0);

        const ownMsPerUnit = this.ownMsPerUnit();
        if (ownMsPerUnit !== undefined && ownMsPerUnit > 0) {
            if (!this.calibration || this.calibration.shareMs !== shareMs || this.calibration.cap !== cap) {
                const units = Math.max(1, Math.floor(shareMs / ownMsPerUnit));
                this.adopt({
                    units: cap === undefined ? units : Math.min(units, cap),
                    shareMs,
                    cap,
                    model: { fixedMs: 0, msPerUnit: ownMsPerUnit },
                    frameSeconds: 0,
                    frames: 0,
                    samples: new Map(),
                    measured: true,
                });
            }
        } else if (!input.timed) {
            return this.handOut('not calibrated', cap ?? DEFAULT_UNITS_PER_FRAME, 0);
        } else {
            this.trackTime(input.deltaTime);
            this.startBurstIfDue(shareMs, cap);
            const burst = this.burst;
            if (burst) {
                const units = burst.next(input.frame, input.deltaTime);
                // A burst that gave up (no timings) finishes inside `next`.
                if (burst.result) this.adopt(burst.result);
                return this.handOut('calibrating', cap === undefined ? units : Math.min(units, cap), 0);
            }
        }

        const calibration = this.calibration as Calibration;
        if (!calibration.measured) return this.handOut('not calibrated', calibration.units, 0);
        const verdict = this.guard.addFrame(input.deltaTime, budget.targetFps, shareMs);
        const framesPerSecond = this.guard.framesPerSecond(budget.targetFps);
        const floorUnits = this.floorUnits(framesPerSecond ?? ASSUMED_FRAMES_PER_SECOND, calibration.units);
        if (verdict === 'step down') {
            this.current = Math.max(floorUnits, Math.floor(this.current * SATURATION.stepDown));
        } else if (verdict === 'recover' && this.current < calibration.units) {
            // + 1 so a batch of a few units still grows.
            this.current = Math.min(calibration.units, Math.max(this.current + 1, Math.ceil(this.current * SATURATION.recovery)));
        }
        const units = Math.max(floorUnits, this.current);
        return this.handOut('calibrated', cap === undefined ? units : Math.min(units, cap), floorUnits, framesPerSecond);
    }

    /** After the technique's encode: what it was handed and what it used. */
    record(handed: number, used: number): void {
        this.burst?.used(handed, used);
        if (this.burst?.result) this.adopt(this.burst.result);
    }

    /** A timed frame of this technique. */
    addSample(sample: CostSample): void {
        const burst = this.burst;
        if (burst) {
            burst.add(sample);
            if (burst.result) this.adopt(burst.result);
            return;
        }
        if (this.calibration && sample.frame > this.calibratedAfterFrame) this.guard.addUpdateMs(sample.ms);
    }

    /** The scene's geometry changed this frame: calibrate again once it is still. */
    geometryChanged(): void {
        this.geometryStillSeconds = 0;
    }

    /** Calibrate again at the next millisecond frame (`GIPlatformImpl.recalibrate`). */
    recalibrate(): void {
        this.recalibrateRequested = true;
    }

    private trackTime(deltaTime: number): void {
        const seconds = deltaTime > 0 && Number.isFinite(deltaTime) ? deltaTime : 0;
        this.secondsSinceGeometryCalibration += seconds;
        if (this.geometryStillSeconds !== undefined) this.geometryStillSeconds += seconds;
    }

    private startBurstIfDue(shareMs: number, cap: number | undefined): void {
        const calibration = this.calibration;
        const inProgress = this.burst;
        const geometryDue =
            this.geometryStillSeconds !== undefined &&
            this.geometryStillSeconds >= GEOMETRY_RECALIBRATION.stillSeconds &&
            this.secondsSinceGeometryCalibration >= GEOMETRY_RECALIBRATION.minSpacingSeconds;
        if (this.recalibrateRequested || geometryDue || (!calibration && !inProgress)) {
            if (geometryDue) this.secondsSinceGeometryCalibration = 0;
            this.recalibrateRequested = false;
            this.geometryStillSeconds = undefined;
            this.beginBurst(new CalibrationBurst(shareMs, cap));
            return;
        }
        // A new share or cap: the same work, so the timings measured so far still hold
        // and seed the new burst, which then needs only a few frames.
        const aimedAt = inProgress ?? calibration;
        if (aimedAt && (aimedAt.shareMs !== shareMs || aimedAt.cap !== cap)) {
            const seed = inProgress ? inProgress.samples : calibration?.measured ? calibration.samples : undefined;
            this.beginBurst(new CalibrationBurst(shareMs, cap, seed));
        }
    }

    private beginBurst(burst: CalibrationBurst): void {
        this.burst = burst;
        this.guard.startOver();
        // A seeded burst may already know the answer.
        if (burst.result) this.adopt(burst.result);
    }

    /** Takes a finished calibration. Every frame handed out so far was a burst frame, whose timings may still arrive. */
    private adopt(calibration: Calibration): void {
        this.calibration = calibration;
        this.burst = undefined;
        this.current = calibration.units;
        this.calibratedAfterFrame = this.latestFrame;
        this.guard.startOver();
    }

    private floorUnits(framesPerSecond: number, calibratedUnits: number): number {
        if (!this.floor) return 0;
        const units = this.floor.minUnitsPerFrame({ framesPerSecond, calibratedUnits });
        return Number.isFinite(units) && units > 0 ? Math.floor(units) : 0;
    }

    private handOut(state: BudgetState, units: number, floorUnits: number, framesPerSecond?: number): number {
        const calibration = this.calibration;
        this.latest = {
            state,
            units,
            calibratedUnits: calibration?.measured ? calibration.units : undefined,
            floorUnits,
            steppedDown: state === 'calibrated' && calibration !== undefined && this.current < calibration.units,
            framesLate: this.guard.framesLate,
            framesPerSecond: framesPerSecond ?? this.latest.framesPerSecond,
            calibration,
        };
        return units;
    }
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
