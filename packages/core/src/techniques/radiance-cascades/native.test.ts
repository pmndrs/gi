import { describe, expect, it } from 'vitest';
import { nativeRcLayout } from './native.ts';

describe('native Split RC storage layout', () => {
    it('packs non-power-of-two world grids without dropping the last cell', () => {
        const { levels, bins } = nativeRcLayout([3, 2, 5]);
        expect(levels.map(level => level.dims)).toEqual([[3,2,5], [2,1,3], [1,1,2]]);
        expect(levels.map(level => level.directions)).toEqual([24,96,384]);
        expect(levels.map(level => level.base)).toEqual([0,720,1296]);
        expect(bins).toBe(2064);
    });
    it('rejects invalid grid dimensions and overflowing GPU indices', () => {
        expect(() => nativeRcLayout([0,1,1])).toThrow(RangeError);
        expect(() => nativeRcLayout([2.5,1,1])).toThrow(RangeError);
        expect(() => nativeRcLayout([2048,2048,2048])).toThrow(RangeError);
    });
});
