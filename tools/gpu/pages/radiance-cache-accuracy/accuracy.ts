/**
 * The numbers the radiance-cache accuracy page (main.ts) reports, as pure
 * functions on interleaved RGB float images, so they run in the page and in
 * vitest alike. They are tools/compare's definitions (tools/compare/metrics.ts),
 * taken the way every DDGI accuracy number is taken (M1's HD accuracy,
 * DECISIONS 2026-11-06; `m4-textured-accuracy/run.ts`):
 *
 * - **bias**: sum(test − reference) / sum(reference), over every pixel and
 *   channel. Positive is brighter than the reference. SPEC's bar is ±10%.
 * - **mean error (8 × 8)**: sum |test − reference| / sum reference on 8 × 8
 *   block means (`tools/compare --downsample 8`). Blocks take most of the
 *   Monte Carlo noise out of both images while a lighting difference
 *   survives.
 * - **p95 error (8 × 8)**: the 95th percentile of the per-block relative
 *   luminance error |T − R| / (R + 5% of the mean R) (`relativeErrorMap`):
 *   the bad corners the mean hides.
 *
 * A mask (one byte per pixel, 1 = compared) restricts all three to some
 * pixels, e.g. those where the world cache has a cell; pixels outside it are
 * left out of both images before the blocks are averaged.
 */

import { compareImages, downsample, percentile, relativeErrorMap } from '../../../compare/metrics.ts';

/** tools/compare's block size for the accuracy numbers. */
export const BLOCK = 8;

export interface AccuracyNumbers {
    /** sum(T − R) / sum(R) over the compared pixels. */
    readonly bias: number;
    /** sum |T − R| / sum R over 8 × 8 block means. */
    readonly meanError8: number;
    /** 95th percentile of the 8 × 8 blocks' relative luminance error. */
    readonly p95Error8: number;
    /** Mean of the reference over the compared pixels and channels (radiance). */
    readonly meanReference: number;
    /** Pixels compared. */
    readonly pixels: number;
}

/** An image of interleaved RGB floats, row 0 at the top. */
export interface RgbImage {
    readonly width: number;
    readonly height: number;
    readonly rgb: Float32Array;
}

/**
 * The accuracy numbers of `test` against `reference` (same size), over the
 * pixels `mask` marks (all when absent). Non-finite values are left out in
 * both images, as tools/compare does.
 */
export function accuracyNumbers(reference: RgbImage, test: RgbImage, mask?: Uint8Array): AccuracyNumbers {
    const { width, height } = reference;
    if (test.width !== width || test.height !== height) throw new Error(`images differ in size: ${width} x ${height} and ${test.width} x ${test.height}`);
    const pixelCount = width * height;
    if (mask && mask.length !== pixelCount) throw new Error(`the mask has ${mask.length} entries for ${pixelCount} pixels`);
    const usable = new Uint8Array(pixelCount);
    let sumReference = 0;
    let sumDifference = 0;
    let pixels = 0;
    for (let pixel = 0; pixel < pixelCount; pixel++) {
        if (mask && !mask[pixel]) continue;
        let finite = true;
        for (let channel = 0; channel < 3; channel++) {
            if (!Number.isFinite(reference.rgb[pixel * 3 + channel]) || !Number.isFinite(test.rgb[pixel * 3 + channel])) finite = false;
        }
        if (!finite) continue;
        usable[pixel] = 1;
        pixels++;
        for (let channel = 0; channel < 3; channel++) {
            const r = reference.rgb[pixel * 3 + channel];
            sumReference += r;
            sumDifference += test.rgb[pixel * 3 + channel] - r;
        }
    }
    const referenceBlocks = downsample(reference.rgb, width, height, BLOCK, usable);
    const testBlocks = downsample(test.rgb, width, height, BLOCK, usable);
    const blocks = compareImages(referenceBlocks.rgb, testBlocks.rgb);
    return {
        bias: sumReference > 0 ? sumDifference / sumReference : 0,
        meanError8: blocks.meanRelativeError,
        p95Error8: percentile(relativeErrorMap(referenceBlocks.rgb, testBlocks.rgb), 95),
        meanReference: pixels > 0 ? sumReference / (3 * pixels) : 0,
        pixels,
    };
}

/** The sum of two images of the same size (e.g. Cycles' lights-direct and indirect parts). */
export function addImages(a: RgbImage, b: RgbImage): RgbImage {
    if (a.width !== b.width || a.height !== b.height) throw new Error('images differ in size');
    const rgb = new Float32Array(a.rgb.length);
    for (let index = 0; index < rgb.length; index++) rgb[index] = a.rgb[index] + b.rgb[index];
    return { width: a.width, height: a.height, rgb };
}

/**
 * The radiance cache's GI term per pixel, as the composite adds it
 * (`giAlbedo × E / π`; the `radiance-cache-three` page checks the composite
 * against this): from rgba images of the albedo and the irradiance E.
 */
export function giTerm(albedoRgba: Float32Array, irradianceRgba: Float32Array, width: number, height: number): RgbImage {
    const rgb = new Float32Array(width * height * 3);
    for (let pixel = 0; pixel < width * height; pixel++) {
        for (let channel = 0; channel < 3; channel++) rgb[pixel * 3 + channel] = (albedoRgba[pixel * 4 + channel] * irradianceRgba[pixel * 4 + channel]) / Math.PI;
    }
    return { width, height, rgb };
}

/** RGB of an RGBA float image, rows reversed when `flipRows` (three's EXRLoader keeps the bottom row first). */
export function rgbOfRgba(rgba: ArrayLike<number>, width: number, height: number, flipRows = false): RgbImage {
    const channels = rgba.length / (width * height);
    if (channels !== 4 && channels !== 3) throw new Error(`expected 3 or 4 channels, got ${channels}`);
    const rgb = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
        const sourceRow = flipRows ? height - 1 - y : y;
        for (let x = 0; x < width; x++) {
            const source = (sourceRow * width + x) * channels;
            const target = (y * width + x) * 3;
            rgb[target] = rgba[source];
            rgb[target + 1] = rgba[source + 1];
            rgb[target + 2] = rgba[source + 2];
        }
    }
    return { width, height, rgb };
}
