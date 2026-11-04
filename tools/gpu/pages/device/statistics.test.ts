import { describe, expect, it } from 'vitest';
import { median, roundOrNull, summarize } from './statistics.ts';

describe('median', () => {
    it('takes the middle value of an odd count', () => {
        expect(median([9, 1, 5])).toBe(5);
    });
    it('averages the two middle values of an even count', () => {
        expect(median([4, 1, 3, 2])).toBe(2.5);
    });
    it('is NaN for no samples', () => {
        expect(median([])).toBeNaN();
    });
    it('is not moved by one slow frame', () => {
        expect(median([10, 10.1, 9.9, 10.2, 400])).toBe(10.1);
    });
});

describe('summarize', () => {
    it('reports count, median, mean, min and max', () => {
        expect(summarize([1, 2, 6])).toEqual({ samples: 3, median: 2, mean: 3, min: 1, max: 6 });
    });
    it('copes with no samples', () => {
        expect(summarize([]).samples).toBe(0);
    });
});

describe('roundOrNull', () => {
    it('rounds and maps non-finite numbers to null', () => {
        expect(roundOrNull(1.23456, 2)).toBe(1.23);
        expect(roundOrNull(Number.NaN)).toBeNull();
        expect(roundOrNull(Infinity)).toBeNull();
    });
});
