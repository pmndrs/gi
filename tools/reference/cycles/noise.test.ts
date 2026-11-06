import { describe, expect, it } from 'vitest';
import { noiseFromIndependentRender, noiseFromParts } from './noise.ts';

describe('noise.ts: a reference noise from two independent renders', () => {
    it('splits the difference evenly between two renders of the same sample count', () => {
        expect(noiseFromIndependentRender(0.02, 1024, 1024)).toBeCloseTo(0.02 / Math.SQRT2, 12);
    });

    it('gives a short second render most of the difference', () => {
        // 1024 against 128 samples: the second render is sqrt(8) times noisier.
        const reference = noiseFromIndependentRender(0.03, 1024, 128);
        const other = reference * Math.sqrt(1024 / 128);
        expect(Math.hypot(reference, other)).toBeCloseTo(0.03, 12);
    });

    it('halves the difference of two halves for their average', () => {
        expect(noiseFromParts(0.01, 2)).toBeCloseTo(0.005, 12);
        expect(noiseFromParts(0.01, 4)).toBeCloseTo(0.01 / Math.SQRT2 / 2, 12);
    });
});
