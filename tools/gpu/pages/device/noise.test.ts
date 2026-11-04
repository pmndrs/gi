import { describe, expect, it } from 'vitest';
import { assessOverhead } from './noise.ts';

describe('assessOverhead', () => {
    it('calls a small difference inside a wide scatter "within noise" and bounds it', () => {
        const result = assessOverhead([-0.3, 0.1, 0.2, 0.9], 90, 0.1);
        expect(result.overheadMs).toBeCloseTo(0.15, 9);
        expect(result.withinNoise).toBe(true);
        expect(result.boundMs).toBe(0.9);
        expect(result.boundPercent).toBeCloseTo(1, 9);
        expect(result.text).toContain('within noise, under ~1.0%');
    });

    it('reports a consistent difference as real', () => {
        const result = assessOverhead([1.0, 1.1, 1.1, 1.0], 10, 0.1);
        expect(result.withinNoise).toBe(false);
        expect(result.text).toContain('outside the round-to-round range');
        expect(result.text).toContain('1.05 ms');
    });

    it('treats a difference below the timer step as noise even if the rounds agree', () => {
        const result = assessOverhead([0.05, 0.05, 0.05], 5, 0.1);
        expect(result.withinNoise).toBe(true);
        expect(result.boundMs).toBe(0.1);
    });
});
