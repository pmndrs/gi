import { describe, expect, it } from 'vitest';
import { ContinuingChangeBatch } from './continuing-change.ts';
import { CONTINUING_CHANGE_FRAMES, CONTINUING_CHANGE_GAP_FRAMES, CONTINUING_CHANGE_MIN_SHARE } from './settings.ts';

/** Sponza's census in T2.4d's VALID run: 586 probes listed, still and through a day. */
const LISTED = 586;
const STILL_RAYS = 83_936;
const DAY_RAYS = 132_544;

describe('the batch while changes continue (T2.4d)', () => {
    it('lists the whole hand-out on still frames and after a single step', () => {
        const batch = new ContinuingChangeBatch();
        batch.noteCensus(STILL_RAYS, LISTED, 0);
        batch.noteFrame(true);
        batch.noteCensus(DAY_RAYS, LISTED, 4_000);
        expect(batch.continuing).toBe(false);
        expect(batch.batchFor(LISTED)).toBe(LISTED);
    });

    it('scales the hand-out by the still rays a probe over the latest once changes continue', () => {
        const batch = new ContinuingChangeBatch();
        batch.noteCensus(STILL_RAYS, LISTED, 0);
        for (let frame = 0; frame < CONTINUING_CHANGE_FRAMES; frame++) batch.noteFrame(true);
        batch.noteCensus(DAY_RAYS, LISTED, 4_800);
        expect(batch.continuing).toBe(true);
        expect(batch.share).toBeCloseTo(STILL_RAYS / DAY_RAYS, 6);
        expect(batch.batchFor(LISTED)).toBe(Math.floor((LISTED * STILL_RAYS) / DAY_RAYS));
        // A few frames without a mark are a host's update rate, not the end of the change.
        for (let frame = 0; frame < CONTINUING_CHANGE_GAP_FRAMES; frame++) batch.noteFrame(false);
        expect(batch.continuing).toBe(true);
        // One more ends it.
        batch.noteFrame(false);
        expect(batch.continuing).toBe(false);
        expect(batch.batchFor(LISTED)).toBe(LISTED);
    });

    it('counts a change marked every other frame (a host animating at 30 Hz on a 60 Hz display) as continuing', () => {
        const batch = new ContinuingChangeBatch();
        batch.noteCensus(STILL_RAYS, LISTED, 0);
        let continuingFrom: number | null = null;
        for (let frame = 0; frame < 3 * CONTINUING_CHANGE_FRAMES; frame++) {
            batch.noteFrame(frame % 2 === 0);
            if (continuingFrom === null && batch.continuing) continuingFrom = frame;
            // Once on, it stays on through the unmarked frames.
            if (continuingFrom !== null) expect(batch.continuing).toBe(true);
        }
        // From the first mark at which the run spans CONTINUING_CHANGE_FRAMES frames (frame 30: 0 to 30 is 31 frames).
        expect(continuingFrom).toBe(CONTINUING_CHANGE_FRAMES);
        batch.noteCensus(DAY_RAYS, LISTED, 4_800);
        expect(batch.batchFor(LISTED)).toBe(Math.floor((LISTED * STILL_RAYS) / DAY_RAYS));
    });

    it('starts a new run after a gap longer than CONTINUING_CHANGE_GAP_FRAMES', () => {
        const batch = new ContinuingChangeBatch();
        for (let frame = 0; frame < CONTINUING_CHANGE_FRAMES - 1; frame++) batch.noteFrame(true);
        for (let frame = 0; frame <= CONTINUING_CHANGE_GAP_FRAMES; frame++) batch.noteFrame(false);
        batch.noteFrame(true);
        expect(batch.continuing).toBe(false);
        for (let frame = 1; frame < CONTINUING_CHANGE_FRAMES - 1; frame++) batch.noteFrame(true);
        expect(batch.continuing).toBe(false);
        batch.noteFrame(true);
        expect(batch.continuing).toBe(true);
    });

    it('never goes below its floor, never above the hand-out, and needs a still reading first', () => {
        const batch = new ContinuingChangeBatch();
        for (let frame = 0; frame < CONTINUING_CHANGE_FRAMES; frame++) batch.noteFrame(true);
        batch.noteCensus(DAY_RAYS, LISTED, 100);
        expect(batch.share).toBe(1);
        batch.noteCensus(STILL_RAYS, LISTED, 0);
        batch.noteCensus(STILL_RAYS * 10, LISTED, 100);
        expect(batch.share).toBe(CONTINUING_CHANGE_MIN_SHARE);
        batch.noteCensus(STILL_RAYS / 2, LISTED, 100);
        expect(batch.share).toBe(1);
        batch.reset();
        expect(batch.continuing).toBe(false);
    });
});
