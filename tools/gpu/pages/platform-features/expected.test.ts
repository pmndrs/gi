import { describe, expect, it } from 'vitest';
import { F16_EXPECTED, halfToFloat, splitDispatch, subgroupInput, workgroupsRun, wrongSubgroupSums } from './expected.ts';

describe('splitDispatch', () => {
    it('keeps a small count in x', () => {
        expect(splitDispatch(37, 65535)).toEqual({ x: 37, y: 1, z: 1 });
    });

    it('gives [0, 1, 1] for nothing', () => {
        expect(splitDispatch(0, 65535)).toEqual({ x: 0, y: 1, z: 1 });
    });

    it('splits past the per-dimension limit and rounds y up', () => {
        expect(splitDispatch(5, 2)).toEqual({ x: 2, y: 3, z: 1 });
        expect(splitDispatch(65535 * 2 + 5, 65535)).toEqual({ x: 65535, y: 3, z: 1 });
    });
});

describe('workgroupsRun', () => {
    it('multiplies the dimensions', () => {
        expect(workgroupsRun({ x: 2, y: 3, z: 1 }, 65535)).toBe(6);
    });

    it('runs nothing when a dimension is over the limit', () => {
        expect(workgroupsRun({ x: 65536, y: 1, z: 1 }, 65535)).toBe(0);
    });
});

describe('halfToFloat', () => {
    it('decodes normal, subnormal and special values', () => {
        expect(halfToFloat(0x3c00)).toBe(1);
        expect(halfToFloat(0xc000)).toBe(-2);
        expect(halfToFloat(0x7bff)).toBe(65504);
        expect(halfToFloat(0x0001)).toBe(2 ** -24);
        expect(halfToFloat(0x7c00)).toBe(Infinity);
        expect(halfToFloat(0x7e00)).toBeNaN();
    });

    it('every expected f16 result is exact in f16', () => {
        // Encode by search: the value must appear among the 65,536 half patterns.
        const exact = new Set<number>();
        for (let bits = 0; bits < 0x10000; bits++) exact.add(halfToFloat(bits));
        for (const { value } of F16_EXPECTED) expect(exact.has(value)).toBe(true);
    });
});

describe('wrongSubgroupSums', () => {
    it('accepts sums grouped by leader in any lane order', () => {
        const inputs = [1, 2, 3, 4, 5, 6].map((_, index) => subgroupInput(index));
        // Two subgroups with interleaved lanes: leaders 0 and 1.
        const leaders = [0, 1, 0, 1, 0, 1];
        const even = inputs[0] + inputs[2] + inputs[4];
        const odd = inputs[1] + inputs[3] + inputs[5];
        expect(wrongSubgroupSums(inputs, [even, odd, even, odd, even, odd], leaders)).toBe(0);
        expect(wrongSubgroupSums(inputs, [even, odd, even, odd, even, even], leaders)).toBe(1);
    });
});
