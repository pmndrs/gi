import { describe, expect, it } from 'vitest';
import { timestampsToPasses, totalByName } from './timestamp-results';

describe('timestampsToPasses', () => {
    it('converts nanosecond differences to milliseconds', () => {
        // Pass 0 runs 1.5 ms, pass 1 runs 0.25 ms; the start values are arbitrary clock readings.
        const timestamps = new BigUint64Array([
            1_000_000_000n,
            1_001_500_000n,
            2_000_000_000n,
            2_000_250_000n,
        ]);
        const passes = timestampsToPasses(['trace', 'blend'], timestamps);
        expect(passes).toEqual([
            { name: 'trace', gpuMs: 1.5 },
            { name: 'blend', gpuMs: 0.25 },
        ]);
    });

    it('keeps full precision for large 64-bit clock values', () => {
        // Above 2^53 a plain Number subtraction would lose the low digits.
        const start = 2n ** 60n;
        const timestamps = new BigUint64Array([start, start + 123_000n]);
        expect(timestampsToPasses(['p'], timestamps)[0].gpuMs).toBeCloseTo(0.123, 9);
    });

    it('drops a pass whose end is before its start', () => {
        const timestamps = new BigUint64Array([500n, 100n, 100n, 1_100n]);
        const passes = timestampsToPasses(['broken', 'fine'], timestamps);
        expect(passes.map((pass) => pass.name)).toEqual(['fine']);
    });

    it('accepts a zero-length pass', () => {
        const passes = timestampsToPasses(['empty'], new BigUint64Array([7n, 7n]));
        expect(passes).toEqual([{ name: 'empty', gpuMs: 0 }]);
    });

    it('ignores passes with missing timestamps', () => {
        expect(timestampsToPasses(['a', 'b'], new BigUint64Array([0n, 1_000_000n]))).toEqual([
            { name: 'a', gpuMs: 1 },
        ]);
        expect(timestampsToPasses([], new BigUint64Array(0))).toEqual([]);
    });
});

describe('totalByName', () => {
    it('adds up passes that share a name', () => {
        const totals = totalByName([
            { name: 'cascade', gpuMs: 0.5 },
            { name: 'other', gpuMs: 2 },
            { name: 'cascade', gpuMs: 0.25 },
        ]);
        expect(totals.get('cascade')).toBe(0.75);
        expect(totals.get('other')).toBe(2);
        expect(totals.size).toBe(2);
    });
});
