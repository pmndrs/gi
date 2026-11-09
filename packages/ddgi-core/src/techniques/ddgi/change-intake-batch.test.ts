import { describe, expect, it } from 'vitest';
import { ChangeIntakeBatch, type CensusReading } from './change-intake-batch.ts';
import { changeServing } from './ddgi-technique.ts';
import { CHANGE_BATCH_MIN_SHARE, CHANGE_INTAKE_MAX_FRAMES, CONTINUING_CHANGE_FRAMES, CONTINUING_CHANGE_GAP_FRAMES, PROPAGATED_PROBE_RAYS, probeGrid, withDefaults } from './settings.ts';

/** Sponza's census in T2.10's VALID run: 136 probes listed, still and after a 30° sun turn (medians). */
const LISTED = 136;
const STILL_RAYS = 19_584;
const STEP_RAYS = 28_992;
const STEP_PROPAGATED = 926;
const RAYS = { full: 256, fixed: 32 };
/** Sponza takes a change in over more than FAR_RULE_FRAMES (its keys don't all owe their full count). */
const OUTLASTS = true;
/** No Budget v2 floor, so the share alone decides. */
const NO_FLOOR = 0;
/** Sponza's active probes (about half its 19,425 are off). */
const ACTIVE = 9_000;

function still(frame: number): CensusReading {
    return { frame, raysTraced: STILL_RAYS, listedProbes: LISTED, owedProbes: 0, probesPropagated: 0, activeProbes: ACTIVE };
}

function busy(frame: number): CensusReading {
    return { frame, raysTraced: STEP_RAYS, listedProbes: LISTED, owedProbes: 2_613, probesPropagated: STEP_PROPAGATED, activeProbes: ACTIVE };
}

describe('the batch while a change is taken in (T2.4d, T2.18)', () => {
    it('lists the whole hand-out on still frames', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(5, false);
        expect(batch.takingInChange).toBe(false);
        expect(batch.batchFor(LISTED, RAYS, OUTLASTS, NO_FLOOR)).toBe(LISTED);
    });

    it('lists fewer from the step itself: every listed probe counted at its full rays until a reading from the change arrives', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        expect(batch.takingInChange).toBe(true);
        expect(batch.share(LISTED, RAYS, OUTLASTS)).toBeCloseTo(STILL_RAYS / LISTED / RAYS.full, 6);
        // A still reading from before the step arrives late: still the full-rays guess.
        batch.noteCensus(still(8));
        batch.noteFrame(11, false);
        expect(batch.share(LISTED, RAYS, OUTLASTS)).toBeCloseTo(STILL_RAYS / LISTED / RAYS.full, 6);
    });

    it('does not cut a hand-out that lists the whole volume (a test mode updating every probe every frame)', () => {
        // two-room: 1,071 probes, say 700 active; the whole volume listed still and after the step.
        const probes = 1_071;
        const active = 700;
        const wholeVolume = active * RAYS.full + (probes - active) * RAYS.fixed;
        const batch = new ChangeIntakeBatch();
        batch.noteCensus({ frame: 0, raysTraced: wholeVolume, listedProbes: probes, owedProbes: 0, probesPropagated: 0, activeProbes: active });
        batch.noteFrame(10, true);
        expect(batch.batchFor(probes, RAYS, OUTLASTS, NO_FLOOR)).toBe(probes);
    });

    it('then follows the census: still cost a probe over the cost now, propagation counted in rays', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        batch.noteCensus(busy(10));
        const busyCost = (STEP_RAYS + STEP_PROPAGATED * PROPAGATED_PROBE_RAYS) / LISTED;
        expect(batch.share(LISTED, RAYS, OUTLASTS)).toBeCloseTo(STILL_RAYS / LISTED / busyCost, 6);
        expect(batch.batchFor(LISTED, RAYS, OUTLASTS, NO_FLOOR)).toBe(Math.floor((LISTED * STILL_RAYS) / LISTED / busyCost));
    });

    it('ends when a reading from at or after the latest mark shows nothing owed, not before', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        batch.noteCensus(busy(10));
        // The change marked again at frame 20: a reading from frame 15 with nothing owed doesn't end it.
        batch.noteFrame(20, true);
        batch.noteCensus(still(15));
        expect(batch.takingInChange).toBe(true);
        batch.noteCensus(busy(21));
        expect(batch.takingInChange).toBe(true);
        batch.noteCensus(still(40));
        expect(batch.takingInChange).toBe(false);
        expect(batch.batchFor(LISTED, RAYS, OUTLASTS, NO_FLOOR)).toBe(LISTED);
    });

    it('takes in a change marked every other frame (a host animating at 30 Hz on a 60 Hz display) for as long as it goes on', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        for (let frame = 10; frame < 200; frame++) {
            batch.noteFrame(frame, frame % 2 === 0);
            // Readings arrive three frames late and show the change's work still owed.
            if (frame >= 13) batch.noteCensus(busy(frame - 3));
            expect(batch.takingInChange).toBe(true);
        }
    });

    it('ignores readings older than the newest one', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        batch.noteCensus(busy(12));
        const share = batch.share(LISTED, RAYS, OUTLASTS);
        batch.noteCensus({ ...busy(11), raysTraced: STEP_RAYS * 4 });
        expect(batch.share(LISTED, RAYS, OUTLASTS)).toBe(share);
    });

    it('never goes below its floor, never above the hand-out, and needs a still reading first', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteFrame(1, true);
        batch.noteCensus(busy(1));
        expect(batch.share(LISTED, RAYS, OUTLASTS)).toBe(1);
        batch.noteFrame(2, true);
        batch.noteCensus(still(0));
        // Frame 0 is older than the newest reading: ignored, still nothing to compare with.
        expect(batch.share(LISTED, RAYS, OUTLASTS)).toBe(1);
        const fresh = new ChangeIntakeBatch();
        fresh.noteCensus(still(0));
        fresh.noteFrame(1, true);
        fresh.noteCensus({ ...busy(1), raysTraced: STILL_RAYS * 10 });
        expect(fresh.share(LISTED, RAYS, OUTLASTS)).toBe(CHANGE_BATCH_MIN_SHARE);
        fresh.noteCensus({ ...busy(2), raysTraced: STILL_RAYS / 2, probesPropagated: 0 });
        expect(fresh.share(LISTED, RAYS, OUTLASTS)).toBe(1);
        fresh.reset();
        expect(fresh.takingInChange).toBe(false);
        expect(fresh.share(LISTED, RAYS, OUTLASTS)).toBe(1);
    });

    it('serves a single step at the hand-out where a change fits the window (the furnished room, two-room, Cornell)', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        batch.noteCensus(busy(10));
        expect(batch.takingInChange).toBe(true);
        expect(batch.batchFor(LISTED, RAYS, false, NO_FLOOR)).toBe(LISTED);
    });

    it('cuts a change that continues there too, once its marks span CONTINUING_CHANGE_FRAMES (T2.4d), gaps included (T2.8)', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        let continuingFrom: number | null = null;
        for (let frame = 0; frame < 3 * CONTINUING_CHANGE_FRAMES; frame++) {
            // Marked every other frame: a host animating at 30 Hz on a 60 Hz display.
            batch.noteFrame(100 + frame, frame % 2 === 0);
            if (frame >= 3) batch.noteCensus(busy(100 + frame - 3));
            if (continuingFrom === null && batch.continuing) continuingFrom = frame;
            if (continuingFrom !== null) expect(batch.continuing).toBe(true);
            else expect(batch.batchFor(LISTED, RAYS, false, NO_FLOOR)).toBe(LISTED);
        }
        // From the first mark at which the run spans CONTINUING_CHANGE_FRAMES frames (frame 30: 0 to 30 is 31 frames).
        expect(continuingFrom).toBe(CONTINUING_CHANGE_FRAMES);
        expect(batch.batchFor(LISTED, RAYS, false, NO_FLOOR)).toBeLessThan(LISTED);
        // A few frames without a mark are the host's update rate; one more ends the run. The loop
        // ended on an unmarked frame, so CONTINUING_CHANGE_GAP_FRAMES - 1 more keep it going.
        const last = 100 + 3 * CONTINUING_CHANGE_FRAMES;
        for (let gap = 1; gap < CONTINUING_CHANGE_GAP_FRAMES; gap++) batch.noteFrame(last + gap, false);
        expect(batch.continuing).toBe(true);
        batch.noteFrame(last + CONTINUING_CHANGE_GAP_FRAMES, false);
        expect(batch.continuing).toBe(false);
    });

    it("never cuts below Budget v2's floor, and the floor never lifts the batch above the hand-out", () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        // The full-rays guess cuts 136 to 76; Sponza's floor at 60 Hz is 81.
        expect(batch.batchFor(LISTED, RAYS, OUTLASTS, NO_FLOOR)).toBe(76);
        expect(batch.batchFor(LISTED, RAYS, OUTLASTS, 81)).toBe(81);
        expect(batch.batchFor(LISTED, RAYS, OUTLASTS, 500)).toBe(LISTED);
    });

    it('ends the intake CHANGE_INTAKE_MAX_FRAMES after the latest mark even while the census still shows work owed', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        for (let frame = 11; frame <= 10 + CHANGE_INTAKE_MAX_FRAMES; frame++) {
            batch.noteFrame(frame, false);
            batch.noteCensus(busy(frame - 3));
            expect(batch.takingInChange).toBe(true);
        }
        batch.noteFrame(11 + CHANGE_INTAKE_MAX_FRAMES, false);
        expect(batch.takingInChange).toBe(false);
        expect(batch.batchFor(LISTED, RAYS, OUTLASTS, NO_FLOOR)).toBe(LISTED);
    });

    it('ends the intake on the cap with no census at all (GPU timings off), and lists the hand-out meanwhile', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteFrame(10, true);
        expect(batch.takingInChange).toBe(true);
        expect(batch.batchFor(LISTED, RAYS, OUTLASTS, NO_FLOOR)).toBe(LISTED);
        for (let frame = 11; frame <= 11 + CHANGE_INTAKE_MAX_FRAMES; frame++) batch.noteFrame(frame, false);
        expect(batch.takingInChange).toBe(false);
    });
});

describe('changeServing: where a single change gets the smaller batch (T2.18)', () => {
    /** A volume of `counts` probes at 1 m spacing, and its grid. */
    const volumeOf = (counts: readonly [number, number, number]) => {
        const volume = { center: [0, 0, 0] as [number, number, number], size: [counts[0] - 1, counts[1] - 1, counts[2] - 1] as [number, number, number], spacing: 1 };
        return { volume, grid: probeGrid(volume) };
    };

    it("Sponza at its calibrated batch: the key path, keys away from the view owing 4, so a change outlasts the window", () => {
        const { volume, grid: sponza } = volumeOf([37, 21, 25]);
        expect(sponza.probeCount).toBe(19_425);
        const serving = changeServing(sponza, 134, withDefaults({ volume }));
        expect(serving).toEqual({ bigVolume: true, keysFitWindow: false, changeOutlastsWindow: true });
        // With key probes off, "far probes owe 2" spreads a change over many frames too.
        expect(changeServing(sponza, 134, withDefaults({ volume, keyProbes: false })).changeOutlastsWindow).toBe(true);
    });

    it('the furnished room: the key path, but every key owes its full count, so a step is served at the hand-out', () => {
        const { volume, grid: room } = volumeOf([17, 14, 21]);
        expect(room.probeCount).toBe(4_998);
        expect(changeServing(room, 571, withDefaults({ volume }))).toEqual({ bigVolume: true, keysFitWindow: true, changeOutlastsWindow: false });
    });

    it('two-room: every probe owes its full count within the window, no key path', () => {
        const { volume, grid: twoRoom } = volumeOf([17, 7, 9]);
        expect(changeServing(twoRoom, 880, withDefaults({ volume })).changeOutlastsWindow).toBe(false);
    });
});
