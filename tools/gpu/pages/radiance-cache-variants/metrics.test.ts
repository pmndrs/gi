import { describe, expect, it } from 'vitest';
import { flicker, settleFrame, stepResponse } from './metrics.ts';

/** Block means per frame: `blocks` blocks following `value(frame, block)`. */
function frames(count: number, blocks: number, value: (frame: number, block: number) => number): Float32Array[] {
    return Array.from({ length: count }, (_, frame) => Float32Array.from({ length: blocks }, (_, block) => value(frame, block)));
}

describe('P6.5 step response', () => {
    it('reads an instant step as followed in one frame, with nothing left over', () => {
        const before = new Float32Array([1, 1, 1, 1]);
        const response = stepResponse(before, frames(80, 4, () => 2))!;
        expect(response.changedBlocks).toBe(4);
        expect(response.endOverBefore).toBeCloseTo(2);
        expect(response.framesTo50).toBe(1);
        expect(response.framesTo90).toBe(1);
        expect(response.residualAt1).toBe(0);
        expect(response.overshoot).toBe(0);
    });

    it('reads an exponential approach: half way in about ln 2 / rate frames, 90% in ln 10 / rate', () => {
        const rate = 0.1;
        const response = stepResponse(new Float32Array([1, 1]), frames(200, 2, (frame) => 2 - Math.exp(-rate * (frame + 1))))!;
        expect(response.framesTo50).toBe(Math.ceil(Math.log(2) / rate));
        expect(response.framesTo90).toBe(Math.ceil(Math.log(10) / rate));
        expect(response.residualAt4).toBeCloseTo(Math.exp(-0.4), 2);
    });

    it('measures only the blocks that changed, and an overshoot past settled', () => {
        const before = new Float32Array([1, 1, 1]);
        // Block 0 jumps to 3 then settles at 2; the others don't change.
        const response = stepResponse(before, frames(80, 3, (frame, block) => (block === 0 ? (frame < 2 ? 3 : 2) : 1)))!;
        expect(response.changedBlocks).toBe(1);
        expect(response.overshoot).toBeCloseTo(1);
        expect(response.framesTo90).toBe(3);
    });

    it('gives null when nothing changed, and when the run never settles', () => {
        expect(stepResponse(new Float32Array([1, 1]), frames(60, 2, () => 1))).toBeNull();
        const slow = stepResponse(new Float32Array([1]), frames(60, 1, (frame) => 1 + frame / 59))!;
        expect(slow.framesTo90).toBeNull();
    });
});

describe('P6.5 warm-up and flicker', () => {
    it('finds the frame a series stays near its target from', () => {
        expect(settleFrame([0, 0.5, 0.97, 1.2, 0.98, 1.0], 1, 0.05)).toBe(5);
        expect(settleFrame([0, 0.5], 1, 0.05)).toBeNull();
    });

    it('measures the change per pixel against the pixel, or the floor where dark, and the frame change', () => {
        const surface = new Uint8Array([1, 1, 1, 1, 1, 1, 1, 1, 0]);
        const a = new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 100]);
        const b = new Float32Array([1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 0]);
        const result = flicker([[a, b]], surface);
        expect(result.meanPerPixel).toBeCloseTo(0.1);
        expect(result.meanOfFrame).toBeCloseTo(0.1);
        expect(flicker([[a, a]], surface).meanPerPixel).toBe(0);
    });
});
