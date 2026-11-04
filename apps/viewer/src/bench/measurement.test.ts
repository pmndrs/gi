import { describe, expect, it } from 'vitest';
import { median, NOISY_CV_THRESHOLD, standardDeviation, summarize, summarizeDifference, t95 } from './measurement.ts';

describe('median and spread', () => {
    it('takes the middle value, or the mean of the middle two', () => {
        expect(median([3, 1, 2])).toBe(2);
        expect(median([4, 1, 3, 2])).toBe(2.5);
    });

    it('uses the sample standard deviation', () => {
        expect(standardDeviation([2, 4, 4, 4, 5, 5, 7, 9])).toBeCloseTo(2.138, 3);
        expect(standardDeviation([5])).toBe(0);
    });
});

describe('summarize', () => {
    it('reports the median with its coefficient of variation', () => {
        const result = summarize('frame_gpu_ms', [10, 10.2, 9.8]);
        expect(result.value).toBe(10);
        expect(result.cv).toBeCloseTo(0.02, 3);
        expect(result.noisy).toBe(false);
        expect(result.problem).toBeNull();
        expect(result.repeats).toBe(3);
    });

    it(`marks a spread above ${NOISY_CV_THRESHOLD * 100}% as noisy, and keeps the value`, () => {
        const result = summarize('gi_gpu_ms', [0.5, 1.5, 1]);
        expect(result.value).toBe(1);
        expect(result.cv).toBeCloseTo(0.5, 3);
        expect(result.noisy).toBe(true);
    });

    it('has no spread for a single measurement', () => {
        const result = summarize('first_frame_wall_ms', [120]);
        expect(result.cv).toBeNull();
        expect(result.noisy).toBe(false);
    });

    it('never hands out a negative number: it is a failed measurement with the reason', () => {
        const result = summarize('gi_gpu_ms', [-0.2, 0.1, -0.05], 'GI on was faster than GI off');
        expect(result.value).toBeNull();
        expect(result.problem).toContain('negative');
        expect(result.problem).toContain('GI on was faster than GI off');
    });

    it('never hands out NaN or infinity', () => {
        expect(summarize('bake_wall_ms', [1, Number.NaN]).value).toBeNull();
        expect(summarize('bake_wall_ms', [Number.POSITIVE_INFINITY]).value).toBeNull();
        expect(summarize('bake_wall_ms', []).problem).toBe('no valid samples');
    });

    it('calls repeats spread around a mean of zero noisy, without a ratio', () => {
        const result = summarize('gi_gpu_ms', [-0.1, 0, 0.1]);
        expect(result.value).toBe(0);
        expect(result.cv).toBeNull();
        expect(result.noisy).toBe(true);
    });
});

describe('summarizeDifference (GI cost = GI on minus GI off, per round)', () => {
    it('uses the t quantile for the round count', () => {
        expect(t95(4)).toBe(2.776);
        expect(t95(100)).toBe(1.96);
    });

    it('reports the mean with a 95% interval; clearly above 0 is a resolved value', () => {
        // mean 2.5, s = 0.1581, n = 5: half width 2.776 * 0.1581 / sqrt(5) = 0.1963
        const result = summarizeDifference('gi_gpu_ms', [2.3, 2.4, 2.5, 2.6, 2.7]);
        expect(result.value).toBeCloseTo(2.5, 6);
        expect(result.confidenceInterval!.low).toBeCloseTo(2.3037, 3);
        expect(result.confidenceInterval!.high).toBeCloseTo(2.6963, 3);
        expect(result.belowNoise).toBe(false);
        expect(result.problem).toBeNull();
    });

    it('writes a negative value as measured when the interval contains 0, marked below noise', () => {
        const result = summarizeDifference('gi_gpu_ms', [-0.3, 0.2, -0.1, 0.1, -0.2]);
        expect(result.value).toBeCloseTo(-0.06, 6);
        expect(result.confidenceInterval!.low).toBeLessThan(0);
        expect(result.confidenceInterval!.high).toBeGreaterThan(0);
        expect(result.belowNoise).toBe(true);
        expect(result.problem).toBeNull();
    });

    it('fails only when the whole interval is below 0', () => {
        const result = summarizeDifference('gi_gpu_ms', [-1.1, -0.9, -1, -1.05, -0.95]);
        expect(result.value).toBeNull();
        expect(result.problem).toContain('reliably negative');
        expect(result.confidenceInterval!.high).toBeLessThan(0);
    });

    it('needs at least two finite rounds', () => {
        expect(summarizeDifference('gi_gpu_ms', [0.5]).value).toBeNull();
        expect(summarizeDifference('gi_gpu_ms', [0.5, Number.NaN]).value).toBeNull();
        expect(summarizeDifference('gi_gpu_ms', []).problem).toBe('no valid rounds');
    });
});
