/**
 * The batch while changes continue (docs/design/m2-lights.md §4.5, "Its
 * cost"; T2.4d): Budget v2 calibrates DDGI's probes a frame on a still scene,
 * where the round robin's list holds the volume's mix of active and off
 * probes. While a change keeps going (a sun moving through a day), the
 * priority share lists only active probes, which trace all their rays (256)
 * where an off probe traces its 32 fixed ones, so the same batch costs more.
 * On Sponza at the calibrated 586 probes a frame the update went from
 * 2.04 ms still to 3.29 ms through a day (VALID, PROGRESS 2026-11-06,
 * T2.4d): the rays traced a frame rose 1.58× and the update 1.56×, almost
 * all of it in `ddgi/trace` (+0.96 ms), with `gi/direct` (+0.09 ms),
 * `ddgi/blend` (+0.12 ms) and propagation and the mark (+0.10 ms) after.
 *
 * So while changes continue, the batch is the hand-out times the rays a
 * listed probe traced on still frames over the rays it traces now: the
 * design's "second calibrated count", worked out from the census instead of
 * a second calibration burst, which would make frames heavier during play.
 * A single step is not a continuing change (its response is judged at the
 * calibrated count, and its few busy frames are T2.10's to judge).
 *
 * The census arrives a few frames late, so the batch follows the list's mix
 * a few frames behind; during a long change that is what matters.
 */

import { CONTINUING_CHANGE_FRAMES, CONTINUING_CHANGE_GAP_FRAMES, CONTINUING_CHANGE_MIN_SHARE } from './settings.ts';

/**
 * How fast the still frames' rays a probe follow new readings: a running
 * mean over about this many census readings, so one odd frame can't move it.
 */
const STILL_RAYS_READINGS = 16;

export class ContinuingChangeBatch {
    /**
     * Frames the current run of marks spans, from its first mark to its
     * latest, counting the unmarked frames between them (0: no run).
     */
    private runFrames = 0;
    /** Unmarked frames since the latest mark (infinite before the first). */
    private framesSinceMark = Number.POSITIVE_INFINITY;
    private stillRaysPerProbe: number | null = null;
    private latestRaysPerProbe: number | null = null;

    /**
     * Call once a frame, before the batch is chosen: whether `ddgi/mark` runs
     * this frame. A host animating slower than it draws (30 Hz on a 60 Hz
     * display) marks only some frames; gaps of up to
     * CONTINUING_CHANGE_GAP_FRAMES unmarked frames keep the run going, so its
     * change still counts as continuing (T2.8).
     */
    noteFrame(marked: boolean): void {
        if (marked) {
            const joinsRun = this.runFrames > 0 && this.framesSinceMark <= CONTINUING_CHANGE_GAP_FRAMES;
            this.runFrames = joinsRun ? this.runFrames + this.framesSinceMark + 1 : 1;
            this.framesSinceMark = 0;
            return;
        }
        this.framesSinceMark++;
        if (this.framesSinceMark > CONTINUING_CHANGE_GAP_FRAMES) this.runFrames = 0;
    }

    /**
     * A census reading: the rays traced in that frame, the probes listed in
     * it, and how many probes were owed updates. Frames with nothing owed
     * are still frames: the round robin's list alone.
     */
    noteCensus(raysTraced: number, listedProbes: number, owedProbes: number): void {
        if (!(listedProbes > 0) || !(raysTraced > 0)) return;
        const perProbe = raysTraced / listedProbes;
        this.latestRaysPerProbe = perProbe;
        if (owedProbes !== 0) return;
        this.stillRaysPerProbe = this.stillRaysPerProbe === null ? perProbe : this.stillRaysPerProbe + (perProbe - this.stillRaysPerProbe) / STILL_RAYS_READINGS;
    }

    /**
     * Whether changes continue: a run of marks, with no gap longer than
     * CONTINUING_CHANGE_GAP_FRAMES, has spanned CONTINUING_CHANGE_FRAMES
     * frames, and its latest mark is at most that gap ago.
     */
    get continuing(): boolean {
        return this.runFrames >= CONTINUING_CHANGE_FRAMES;
    }

    /**
     * The share of the hand-out to list this frame: 1 unless changes
     * continue and both readings are known; then the still rays a probe over
     * the latest, at most 1 and at least CONTINUING_CHANGE_MIN_SHARE.
     */
    get share(): number {
        if (!this.continuing || this.stillRaysPerProbe === null || this.latestRaysPerProbe === null) return 1;
        return Math.min(1, Math.max(CONTINUING_CHANGE_MIN_SHARE, this.stillRaysPerProbe / this.latestRaysPerProbe));
    }

    /** The batch for a hand-out of `units` probes. */
    batchFor(units: number): number {
        return Math.floor(units * this.share);
    }

    /** Forget everything (a new field: `prepare`, `resetHistory`). */
    reset(): void {
        this.runFrames = 0;
        this.framesSinceMark = Number.POSITIVE_INFINITY;
        this.stillRaysPerProbe = null;
        this.latestRaysPerProbe = null;
    }
}
