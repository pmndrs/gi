import { describe, expect, it } from 'vitest';
import { chooseProbeGridLayout } from './probe-grid-layout';

describe('chooseProbeGridLayout', () => {
    it('keeps the manifest spacing when it fits the budget', () => {
        const layout = chooseProbeGridLayout([5.8, 5.6, 5.8], 0.5, 2000);
        expect(layout.counts).toEqual([12, 12, 12]);
        expect(layout.total).toBe(1728);
        expect(layout.spacing).toBe(0.5);
    });

    it('widens the spacing until the probe count fits the budget', () => {
        const layout = chooseProbeGridLayout([36.5, 20, 24], 1, 600);
        expect(layout.total).toBeLessThanOrEqual(600);
        expect(layout.total).toBeGreaterThan(300);
        expect(layout.spacing).toBeGreaterThan(1);
    });

    it('never goes below 2 probes per axis', () => {
        const layout = chooseProbeGridLayout([0.1, 100, 100], 1, 8);
        expect(layout.counts[0]).toBe(2);
    });

    it('rejects nonsense input', () => {
        expect(() => chooseProbeGridLayout([1, 1, 1], 0, 100)).toThrow();
        expect(() => chooseProbeGridLayout([1, 1, 1], 1, 2)).toThrow();
    });
});
