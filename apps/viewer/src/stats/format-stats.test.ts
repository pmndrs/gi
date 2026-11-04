import { describe, expect, it } from 'vitest';
import { formatStats, SmoothedValue } from './format-stats.ts';

describe('SmoothedValue', () => {
    it('starts at the first sample and then moves toward new ones', () => {
        const smoothed = new SmoothedValue(0.5);
        expect(smoothed.value).toBeUndefined();
        smoothed.add(10);
        expect(smoothed.value).toBe(10);
        smoothed.add(20);
        expect(smoothed.value).toBe(15);
        smoothed.reset();
        expect(smoothed.value).toBeUndefined();
    });
});

describe('formatStats', () => {
    it('shows the frame time with fps', () => {
        expect(formatStats(20, new Map()).frameLine).toBe('frame: 20.0 ms (50 fps)');
    });

    it('shows a dash before the first frame', () => {
        expect(formatStats(undefined, new Map()).frameLine).toBe('frame: -');
    });

    it('has no pass lines when timestamps are unavailable', () => {
        const text = formatStats(16, new Map());
        expect(text.passLines).toEqual([]);
        expect(text.totalLine).toBe('');
    });

    it('lists the slowest pass first, aligned, with a total', () => {
        const text = formatStats(
            16,
            new Map([
                ['blend', 0.5],
                ['ddgi/trace', 3.25],
            ]),
        );
        expect(text.passLines).toEqual(['ddgi/trace     3.25 ms', 'blend          0.50 ms']);
        expect(text.totalLine).toBe('GPU total      3.75 ms');
    });
});
