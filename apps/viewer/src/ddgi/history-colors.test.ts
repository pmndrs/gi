import { describe, expect, it } from 'vitest';
import { halfwaySamples, historyColor, HISTORY_COLORS, settledShare } from './history-colors';

describe('history colours', () => {
    it('run from red at no history through yellow halfway to green at the full count', () => {
        expect(historyColor(0, 200)).toEqual(HISTORY_COLORS.fewUpdates);
        historyColor(halfwaySamples(200), 200).forEach((channel, index) => expect(channel).toBeCloseTo(HISTORY_COLORS.halfway[index], 10));
        historyColor(200, 200).forEach((channel, index) => expect(channel).toBeCloseTo(HISTORY_COLORS.settled[index], 10));
    });

    it('use a logarithmic scale, halfway at about 13 of 200, and stay within 0 to 1', () => {
        expect(halfwaySamples(200)).toBeCloseTo(13.18, 2);
        expect(settledShare(-5, 200)).toBe(0);
        expect(settledShare(1000, 200)).toBe(1);
        expect(settledShare(5, 0)).toBe(0);
    });
});
