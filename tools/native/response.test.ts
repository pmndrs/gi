import { describe, expect, it } from 'vitest';
import { bestShift, sweepResponse } from './response.ts';

/** A settled view along an out-and-back path of `frames` frames: bright at the ends (lamp at the start), dark mid-way. */
function settledCurve(frame: number, frames: number): number {
    const last = frames - 1;
    const along = Math.min(Math.max(frame, 0), last) / last; // 0 → 1 over the pass
    return 1 - 0.8 * Math.sin(Math.PI * along) ** 2;
}

describe('response measures (T2.10)', () => {
    it('finds the shift between a curve and a delayed copy, to a tenth of a frame', () => {
        const base = Array.from({ length: 400 }, (_, frame) => settledCurve(frame, 400));
        const delayed = Array.from({ length: 400 }, (_, frame) => settledCurve(frame - 7, 400));
        expect(bestShift(delayed, base).shift).toBeCloseTo(7, 1);
        expect(bestShift(base, delayed).shift).toBeCloseTo(-7, 1);
    });

    it('reads no lag from a view that keeps up with the lamp', () => {
        const values = Array.from({ length: 625 }, (_, frame) => settledCurve(frame, 625));
        const sweep = sweepResponse(values);
        expect(sweep.lagFrames).toBeCloseTo(0, 1);
        expect(sweep.outBackGap).toBeLessThan(1e-9);
        expect(sweep.lowestShare).toBeCloseTo(0.2, 3);
    });

    it('reads a view that runs d frames behind the lamp as a lag of d', () => {
        // Held at the start before frame 0, as the 600 warm-up frames hold it
        const values = Array.from({ length: 625 }, (_, frame) => settledCurve(frame - 12, 625));
        const sweep = sweepResponse(values);
        expect(sweep.lagFrames).toBeGreaterThan(11);
        expect(sweep.lagFrames).toBeLessThan(13);
        expect(sweep.outBackGap).toBeGreaterThan(0.01);
    });
});
