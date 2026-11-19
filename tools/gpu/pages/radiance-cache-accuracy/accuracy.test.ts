import { describe, expect, it } from 'vitest';
import { accuracyNumbers, addImages, giTerm, rgbOfRgba, type RgbImage } from './accuracy.ts';

function constant(width: number, height: number, value: number): RgbImage {
    return { width, height, rgb: new Float32Array(width * height * 3).fill(value) };
}

describe('radiance-cache accuracy numbers', () => {
    it('reads a uniformly 10% bright image as +10% bias and 10% error', () => {
        const numbers = accuracyNumbers(constant(32, 16, 2), constant(32, 16, 2.2));
        expect(numbers.bias).toBeCloseTo(0.1, 6);
        expect(numbers.meanError8).toBeCloseTo(0.1, 6);
        expect(numbers.p95Error8).toBeCloseTo(0.1 / 1.05, 4);
        expect(numbers.pixels).toBe(512);
    });

    it('takes noise out with the 8 x 8 blocks but keeps it in the bias only as its mean', () => {
        const reference = constant(64, 64, 1);
        const test = constant(64, 64, 1);
        // A checkerboard of +-50%: no bias, and every 8 x 8 block averages to the reference.
        for (let pixel = 0; pixel < 64 * 64; pixel++) test.rgb.fill(((pixel % 64) + Math.floor(pixel / 64)) % 2 === 0 ? 1.5 : 0.5, pixel * 3, pixel * 3 + 3);
        const numbers = accuracyNumbers(reference, test);
        expect(numbers.bias).toBeCloseTo(0, 6);
        expect(numbers.meanError8).toBeCloseTo(0, 6);
    });

    it('compares only the masked pixels', () => {
        const reference = constant(16, 8, 1);
        const test = constant(16, 8, 1);
        const mask = new Uint8Array(16 * 8);
        for (let pixel = 0; pixel < mask.length; pixel++) {
            const left = pixel % 16 < 8;
            mask[pixel] = left ? 1 : 0;
            if (!left) test.rgb.fill(5, pixel * 3, pixel * 3 + 3);
        }
        const numbers = accuracyNumbers(reference, test, mask);
        expect(numbers.bias).toBe(0);
        expect(numbers.pixels).toBe(64);
    });

    it('skips non-finite values', () => {
        const test = constant(8, 8, 1);
        test.rgb[0] = Number.NaN;
        expect(accuracyNumbers(constant(8, 8, 1), test).pixels).toBe(63);
    });

    it('forms the GI term as albedo x E / pi and adds images', () => {
        const albedo = new Float32Array([0.5, 0.25, 1, 1]);
        const irradiance = new Float32Array([Math.PI, 2 * Math.PI, 0, 1]);
        expect(Array.from(giTerm(albedo, irradiance, 1, 1).rgb)).toEqual([0.5, 0.5, 0]);
        expect(Array.from(addImages(constant(1, 1, 1), constant(1, 1, 2)).rgb)).toEqual([3, 3, 3]);
    });

    it('drops alpha and flips rows on request', () => {
        const rgba = new Float32Array([1, 1, 1, 9, 2, 2, 2, 9]); // two rows of one pixel, bottom row first
        expect(Array.from(rgbOfRgba(rgba, 1, 2).rgb)).toEqual([1, 1, 1, 2, 2, 2]);
        expect(Array.from(rgbOfRgba(rgba, 1, 2, true).rgb)).toEqual([2, 2, 2, 1, 1, 1]);
    });
});
