import { describe, expect, it } from 'vitest';
import { ChangeIntakeBatch, type CensusReading } from './change-intake-batch.ts';
import { CHANGE_BATCH_MIN_SHARE, PROPAGATED_PROBE_RAYS } from './settings.ts';

/** Sponza's census in T2.10's VALID run: 136 probes listed, still and after a 30° sun turn (medians). */
const LISTED = 136;
const STILL_RAYS = 19_584;
const STEP_RAYS = 28_992;
const STEP_PROPAGATED = 926;
const RAYS = { full: 256, fixed: 32 };
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
        expect(batch.batchFor(LISTED, RAYS)).toBe(LISTED);
    });

    it('lists fewer from the step itself: every listed probe counted at its full rays until a reading from the change arrives', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        expect(batch.takingInChange).toBe(true);
        expect(batch.share(LISTED, RAYS)).toBeCloseTo(STILL_RAYS / LISTED / RAYS.full, 6);
        // A still reading from before the step arrives late: still the full-rays guess.
        batch.noteCensus(still(8));
        batch.noteFrame(11, false);
        expect(batch.share(LISTED, RAYS)).toBeCloseTo(STILL_RAYS / LISTED / RAYS.full, 6);
    });

    it('does not cut a hand-out that lists the whole volume (a test mode updating every probe every frame)', () => {
        // two-room: 1,071 probes, say 700 active; the whole volume listed still and after the step.
        const probes = 1_071;
        const active = 700;
        const wholeVolume = active * RAYS.full + (probes - active) * RAYS.fixed;
        const batch = new ChangeIntakeBatch();
        batch.noteCensus({ frame: 0, raysTraced: wholeVolume, listedProbes: probes, owedProbes: 0, probesPropagated: 0, activeProbes: active });
        batch.noteFrame(10, true);
        expect(batch.batchFor(probes, RAYS)).toBe(probes);
    });

    it('then follows the census: still cost a probe over the cost now, propagation counted in rays', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteCensus(still(0));
        batch.noteFrame(10, true);
        batch.noteCensus(busy(10));
        const busyCost = (STEP_RAYS + STEP_PROPAGATED * PROPAGATED_PROBE_RAYS) / LISTED;
        expect(batch.share(LISTED, RAYS)).toBeCloseTo(STILL_RAYS / LISTED / busyCost, 6);
        expect(batch.batchFor(LISTED, RAYS)).toBe(Math.floor((LISTED * STILL_RAYS) / LISTED / busyCost));
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
        expect(batch.batchFor(LISTED, RAYS)).toBe(LISTED);
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
        const share = batch.share(LISTED, RAYS);
        batch.noteCensus({ ...busy(11), raysTraced: STEP_RAYS * 4 });
        expect(batch.share(LISTED, RAYS)).toBe(share);
    });

    it('never goes below its floor, never above the hand-out, and needs a still reading first', () => {
        const batch = new ChangeIntakeBatch();
        batch.noteFrame(1, true);
        batch.noteCensus(busy(1));
        expect(batch.share(LISTED, RAYS)).toBe(1);
        batch.noteFrame(2, true);
        batch.noteCensus(still(0));
        // Frame 0 is older than the newest reading: ignored, still nothing to compare with.
        expect(batch.share(LISTED, RAYS)).toBe(1);
        const fresh = new ChangeIntakeBatch();
        fresh.noteCensus(still(0));
        fresh.noteFrame(1, true);
        fresh.noteCensus({ ...busy(1), raysTraced: STILL_RAYS * 10 });
        expect(fresh.share(LISTED, RAYS)).toBe(CHANGE_BATCH_MIN_SHARE);
        fresh.noteCensus({ ...busy(2), raysTraced: STILL_RAYS / 2, probesPropagated: 0 });
        expect(fresh.share(LISTED, RAYS)).toBe(1);
        fresh.reset();
        expect(fresh.takingInChange).toBe(false);
        expect(fresh.share(LISTED, RAYS)).toBe(1);
    });
});
