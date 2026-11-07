import { describe, expect, it } from 'vitest';
import { parseResponseCsv, responseCsv, stepMarks, type ResponseSample } from './response-curve';

function curve(means: readonly number[]): ResponseSample[] {
    return means.map((mean, index) => ({ frame: index + 1, timeMs: index * 16.7, mean, probesUpdated: 880 }));
}

describe('the response meter', () => {
    it("marks the first change and the first frame within 5% as the T2.4b page's stepResponse does", () => {
        // A step from 1 to 2: the first frame barely moves, then the curve climbs, overshoots and settles.
        const samples = curve([1.005, 1.3, 1.7, 1.92, 2.15, 2.06, 2.02, 2.0]);
        const marks = stepMarks(1, 2, samples);
        expect(marks.firstChangedFrame).toBe(2);
        expect(marks.enteredFrame).toBe(4);
        // 2.15 is 7.5% over: it left the band, and is back for good from frame 6.
        expect(marks.settledFrame).toBe(6);
        expect(marks.leftBandAgain).toBe(true);
        expect(marks.overshootShareOfStep).toBeCloseTo(0.15, 9);
        expect(marks.stepShare).toBeCloseTo(0.5, 9);
    });

    it('scales the first change by the larger side, so a lamp switched on from black has one', () => {
        const marks = stepMarks(0, 1, curve([0.005, 0.02, 0.5, 0.97]));
        expect(marks.firstChangedFrame).toBe(2);
        expect(marks.enteredFrame).toBe(4);
    });

    it('says never when the curve never settles', () => {
        const marks = stepMarks(1, 2, curve([1.1, 1.2, 1.3]));
        expect(marks.enteredFrame).toBeNull();
        expect(marks.settledFrame).toBeNull();
        expect(marks.leftBandAgain).toBe(false);
    });

    it('writes a CSV that reads back with the same marks', () => {
        const samples = curve([1.3, 1.95, 2.0]);
        const marks = stepMarks(1, 2, samples);
        const text = responseCsv(
            { scene: 'two-room', camera: 'lit-room', motion: 'light-step', region: [0, 0, 640, 360], budgetMs: 2, adapter: 'nvidia ampere', settleFrames: 240, settleBudgetMs: 16 },
            marks,
            samples,
        );
        expect(text.split('\n')[4]).toBe('frame,ms,mean,change_from_before,error_from_settled,probes_updated');
        const back = parseResponseCsv(text);
        expect(back.samples).toHaveLength(3);
        expect(back.samples[1].mean).toBeCloseTo(1.95, 6);
        expect(back.firstChangedFrame).toBe(marks.firstChangedFrame);
        expect(back.enteredFrame).toBe(marks.enteredFrame);
        expect(back.settledFrame).toBe(marks.settledFrame);
    });
});
