import { describe, expect, it } from 'vitest';
import { ContinuingChangeBatch } from './continuing-change.ts';
import { CONTINUING_CHANGE_FRAMES, CONTINUING_CHANGE_MIN_SHARE } from './settings.ts';

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
        // One frame without a mark ends it.
        batch.noteFrame(false);
        expect(batch.batchFor(LISTED)).toBe(LISTED);
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
