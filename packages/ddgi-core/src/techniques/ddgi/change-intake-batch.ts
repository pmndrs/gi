/**
 * The batch while a change is taken in (docs/design/m2-lights.md §4.5, "Its
 * cost"; T2.4d, extended to single steps in T2.18).
 *
 * Budget v2 calibrates DDGI's probes a frame on a still scene, where the
 * round robin's list holds the volume's mix of active and off probes. After
 * a change, the priority share lists only active probes, which trace all
 * their rays (256) where an off probe traces its 32 fixed ones, and
 * `ddgi/propagate` passes the keys' change on to their fine probes, so the
 * same batch costs more:
 * - through a day on Sponza at the calibrated 586 the update went from
 *   2.04 ms still to 3.29 ms (VALID, PROGRESS 2026-11-06, T2.4d): the rays
 *   traced a frame rose 1.58× and the update 1.56×, almost all of it in
 *   `ddgi/trace`;
 * - after a single 30° sun turn at the calibrated 136 (about 800 MHz) the 60
 *   frames after it ran at 2.44 ms median, p95 3.12, against 2.04 still
 *   (VALID, PROGRESS 2026-11-08, T2.10): the rays traced 1.48×, and
 *   `ddgi/propagate` +0.05 ms.
 *
 * So from the frame a change is marked until the census says nothing is
 * owed any more, the batch is the hand-out times what a listed probe cost
 * on still frames over what it costs now: the design's "second calibrated
 * count", worked out from the census instead of a second calibration burst,
 * which would make frames heavier during play. A probe's cost is counted in
 * rays: the rays it traced, plus the propagation the frame did, at
 * PROPAGATED_PROBE_RAYS a propagated probe.
 *
 * The census arrives a few frames late. Until a reading from the change's
 * own frames arrives, the list is assumed to cost the most it can: active
 * probes first, each tracing its full rays, and off probes (their fixed rays
 * only) after them only once the volume's active probes are all listed. So
 * the first frames after a step are not the heaviest, and a hand-out that
 * lists the whole volume (a test mode updating every probe every frame)
 * costs the same as on still frames and is not cut. A change that keeps going
 * (a day passing, a lamp swinging, a host animating at 30 Hz on a 60 Hz
 * display) is taken in for as long as it is marked: a census reading from
 * after its latest mark still shows probes owed.
 *
 * This reads only the census DDGI already copies back for its counters
 * (when the platform times passes); nothing new is read back. Without it,
 * the batch is the hand-out.
 */

import { CHANGE_BATCH_MIN_SHARE, PROPAGATED_PROBE_RAYS } from './settings.ts';

/**
 * How fast the still frames' cost a probe follows new readings: a running
 * mean over about this many census readings, so one odd frame can't move it.
 */
const STILL_READINGS = 16;

/** One census reading, from the frame it was copied in. */
export interface CensusReading {
    /** The platform frame the census was recorded in. */
    readonly frame: number;
    /** Rays `ddgi/trace` traced in that frame. */
    readonly raysTraced: number;
    /** Probes in that frame's update list. */
    readonly listedProbes: number;
    /** Probes owed updates and not waiting for their keys, as the scheduler counted them. */
    readonly owedProbes: number;
    /** Fine probes `ddgi/propagate` passed a key's change to. */
    readonly probesPropagated: number;
    /** Probes that trace their full rays (active, or not updated yet), from the census's per-state counts. */
    readonly activeProbes: number;
}

/** The rays a listed probe traces: all of them when active, the fixed ones only when off. */
export interface RaysPerProbe {
    readonly full: number;
    readonly fixed: number;
}

export class ChangeIntakeBatch {
    /** The frame of the first mark of the change being taken in now; null when none is. */
    private intakeStartFrame: number | null = null;
    /** The frame of the latest mark. */
    private lastMarkFrame = Number.NEGATIVE_INFINITY;
    /** The frame of the newest census reading seen. */
    private latestReadingFrame = Number.NEGATIVE_INFINITY;
    /** What a listed probe costs in rays on still frames (a running mean). */
    private stillCostPerProbe: number | null = null;
    /** What a listed probe cost in rays in the newest reading. */
    private latestCostPerProbe: number | null = null;
    /** Probes that trace their full rays, in the newest reading. */
    private activeProbes: number | null = null;

    /** Call once a frame, before the batch is chosen: the frame, and whether `ddgi/mark` runs in it. */
    noteFrame(frame: number, marked: boolean): void {
        if (!marked) return;
        this.lastMarkFrame = frame;
        if (this.intakeStartFrame === null) this.intakeStartFrame = frame;
    }

    /**
     * A census reading. Readings with nothing owed are still frames (the
     * round robin's list alone); the first one recorded at or after the
     * latest mark ends the intake.
     */
    noteCensus(reading: CensusReading): void {
        const { frame, raysTraced, listedProbes, owedProbes, probesPropagated, activeProbes } = reading;
        if (!(listedProbes > 0) || !(raysTraced > 0) || frame < this.latestReadingFrame) return;
        this.latestReadingFrame = frame;
        this.activeProbes = activeProbes;
        const costPerProbe = (raysTraced + Math.max(0, probesPropagated) * PROPAGATED_PROBE_RAYS) / listedProbes;
        this.latestCostPerProbe = costPerProbe;
        if (owedProbes !== 0) return;
        this.stillCostPerProbe = this.stillCostPerProbe === null ? costPerProbe : this.stillCostPerProbe + (costPerProbe - this.stillCostPerProbe) / STILL_READINGS;
        // Nothing owed in a frame at or after the latest mark: the change is taken in.
        if (this.intakeStartFrame !== null && frame >= this.lastMarkFrame) this.intakeStartFrame = null;
    }

    /** Whether a change is being taken in: marked, and no census since its latest mark has shown nothing owed. */
    get takingInChange(): boolean {
        return this.intakeStartFrame !== null;
    }

    /**
     * The share of a hand-out of `units` probes (at most the volume's) to
     * list this frame: 1 on still frames or before any still reading; while
     * a change is taken in, the still cost a probe over the current one (the
     * most the list can cost until a reading from the change's frames
     * arrives), at most 1 and at least CHANGE_BATCH_MIN_SHARE.
     */
    share(units: number, rays: RaysPerProbe): number {
        if (this.intakeStartFrame === null || this.stillCostPerProbe === null || !(units > 0)) return 1;
        const readingSinceChange = this.latestCostPerProbe !== null && this.latestReadingFrame >= this.intakeStartFrame;
        const currentCost = readingSinceChange && this.latestCostPerProbe !== null ? this.latestCostPerProbe : this.mostCostPerProbe(units, rays);
        if (!(currentCost > 0)) return 1;
        return Math.min(1, Math.max(CHANGE_BATCH_MIN_SHARE, this.stillCostPerProbe / currentCost));
    }

    /** The batch for a hand-out of `units` probes, at most the volume's. */
    batchFor(units: number, rays: RaysPerProbe): number {
        return Math.floor(units * this.share(units, rays));
    }

    /** The most a list of `units` probes can cost a probe in rays: every active probe it can hold at full rays, the rest off. */
    private mostCostPerProbe(units: number, rays: RaysPerProbe): number {
        const activeListed = Math.min(units, this.activeProbes ?? units);
        return (activeListed * rays.full + (units - activeListed) * rays.fixed) / units;
    }

    /** Forget everything (a new field: `prepare`, `resetHistory`). */
    reset(): void {
        this.intakeStartFrame = null;
        this.lastMarkFrame = Number.NEGATIVE_INFINITY;
        this.latestReadingFrame = Number.NEGATIVE_INFINITY;
        this.stillCostPerProbe = null;
        this.latestCostPerProbe = null;
        this.activeProbes = null;
    }
}
