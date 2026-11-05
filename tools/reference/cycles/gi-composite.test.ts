import { describe, expect, it } from 'vitest';
import type { FloatImage } from '../../compare/exr.ts';
import { checkLightsOnlyImage, composeGiImage, LIGHTS_ONLY_TOLERANCE } from './gi-composite.ts';

/** A lit-looking image: a smooth gradient, every value above 0. */
function directImage(width = 32, height = 16): FloatImage {
    const rgb = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = 0.05 + (0.5 * x) / width + (0.2 * y) / height;
            rgb.set([value, value * 0.8, value * 0.6], (y * width + x) * 3);
        }
    }
    return { width, height, rgb };
}

function scaled(image: FloatImage, factor: number): FloatImage {
    return { ...image, rgb: image.rgb.map((value) => value * factor) };
}

/** Deterministic per-pixel noise with mean 0 (a fixed pattern of +-amplitude), so block averages cancel most of it. */
function withNoise(image: FloatImage, amplitude: number): FloatImage {
    return { ...image, rgb: image.rgb.map((value, index) => value * (1 + (index % 2 === 0 ? amplitude : -amplitude))) };
}

describe('checkLightsOnlyImage', () => {
    it('passes a lights-only image equal to the direct image (a scene lit by its lights alone)', () => {
        const direct = directImage();
        const check = checkLightsOnlyImage('equals-direct', direct, direct);
        expect(check.passed).toBe(true);
        expect(check.bias).toBe(0);
    });

    it('passes per-pixel noise that averages out', () => {
        const direct = directImage();
        expect(checkLightsOnlyImage('equals-direct', direct, withNoise(direct, 0.2)).passed).toBe(true);
    });

    it('fails a lights-only image that lost part of the light from the lights', () => {
        const direct = directImage();
        const check = checkLightsOnlyImage('equals-direct', direct, scaled(direct, 1 - 2 * LIGHTS_ONLY_TOLERANCE.bias));
        expect(check.passed).toBe(false);
        expect(check.summary).toContain('bias');
    });

    it('fails a lights-only image whose light moved, even when the total is the same', () => {
        const direct = directImage();
        // Mirror the image left to right: same sum, different place.
        const mirrored = new Float32Array(direct.rgb.length);
        for (let y = 0; y < direct.height; y++) {
            for (let x = 0; x < direct.width; x++) {
                for (let channel = 0; channel < 3; channel++) {
                    mirrored[(y * direct.width + x) * 3 + channel] = direct.rgb[(y * direct.width + (direct.width - 1 - x)) * 3 + channel];
                }
            }
        }
        const check = checkLightsOnlyImage('equals-direct', direct, { ...direct, rgb: mirrored });
        expect(Math.abs(check.bias ?? 1)).toBeLessThan(1e-6);
        expect(check.passed).toBe(false);
    });

    it('passes a black lights-only image in a scene without manifest lights, and fails one with light left', () => {
        const direct = directImage();
        const black = scaled(direct, 0);
        expect(checkLightsOnlyImage('black', direct, black).passed).toBe(true);
        expect(checkLightsOnlyImage('black', direct, scaled(direct, 0.01)).passed).toBe(false);
    });

    it('does not judge a scene with both kinds of source', () => {
        const direct = directImage();
        const check = checkLightsOnlyImage(null, direct, scaled(direct, 0.5));
        expect(check.passed).toBe(true);
        expect(check.summary).toContain('not checked');
    });

    it('refuses images of different sizes', () => {
        expect(() => checkLightsOnlyImage('equals-direct', directImage(32, 16), directImage(16, 16))).toThrow(/differ in size/);
    });
});

describe('composeGiImage', () => {
    it('adds indirect and direct and takes the lights-only direct away', () => {
        const indirect = directImage();
        const direct = scaled(indirect, 3);
        const lightsDirect = scaled(indirect, 2);
        const gi = composeGiImage([indirect, direct], lightsDirect);
        for (let index = 0; index < gi.rgb.length; index++) expect(gi.rgb[index]).toBeCloseTo(indirect.rgb[index] * 2, 6);
    });

    it('gives the indirect pass when the lights are the only source', () => {
        const indirect = directImage();
        const direct = scaled(indirect, 4);
        const gi = composeGiImage([indirect, direct], direct);
        for (let index = 0; index < gi.rgb.length; index++) expect(gi.rgb[index]).toBeCloseTo(indirect.rgb[index], 6);
    });
});
