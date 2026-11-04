/**
 * Image difference numbers. Pure functions on interleaved RGB float arrays,
 * with no I/O, so the maths can be unit-tested.
 *
 * Definitions (also printed by `tools/compare --help`). R is the reference
 * (the first image) and T the test (the second), summed over every pixel and
 * every colour channel; non-finite values are skipped in both and counted
 * (tools/compare masks whole pixels first, see findFinitePixels):
 *
 *   mean relative error  =  sum |T - R| / sum R
 *       The average absolute error divided by the average reference value.
 *       A test image that is 10 percent too bright everywhere scores 0.100.
 *       It is a ratio of sums on purpose: a per-pixel mean of |T - R| / R
 *       explodes in the near-black pixels of a dark room and says nothing
 *       about the lit parts.
 *   bias                 =  sum (T - R) / sum R
 *       The same without the absolute value. Positive means the test is
 *       brighter overall, negative darker.
 *   rms                  =  sqrt( mean over values of (T - R)^2 )
 *       In image units (radiance).
 *   relative rms         =  rms / mean R
 */

export interface ImageMetrics {
    pixelCount: number;
    meanReference: number;
    meanTest: number;
    meanRelativeError: number;
    bias: number;
    rms: number;
    relativeRms: number;
    /** Values skipped because either image had a NaN or infinity there. */
    nonFiniteValues: number;
}

export function compareImages(reference: Float32Array, test: Float32Array): ImageMetrics {
    if (reference.length !== test.length) {
        throw new Error(`images differ in size: ${reference.length / 3} pixels vs ${test.length / 3}`);
    }
    let sumReference = 0;
    let sumTest = 0;
    let sumAbsolute = 0;
    let sumSigned = 0;
    let sumSquared = 0;
    let used = 0;
    let nonFinite = 0;
    for (let index = 0; index < reference.length; index++) {
        const r = reference[index];
        const t = test[index];
        if (!Number.isFinite(r) || !Number.isFinite(t)) {
            nonFinite++;
            continue;
        }
        const difference = t - r;
        sumReference += r;
        sumTest += t;
        sumAbsolute += Math.abs(difference);
        sumSigned += difference;
        sumSquared += difference * difference;
        used++;
    }
    const meanReference = used > 0 ? sumReference / used : 0;
    const rms = used > 0 ? Math.sqrt(sumSquared / used) : 0;
    return {
        pixelCount: reference.length / 3,
        meanReference,
        meanTest: used > 0 ? sumTest / used : 0,
        meanRelativeError: sumReference > 0 ? sumAbsolute / sumReference : sumAbsolute > 0 ? Infinity : 0,
        bias: sumReference > 0 ? sumSigned / sumReference : sumSigned > 0 ? Infinity : 0,
        rms,
        relativeRms: meanReference > 0 ? rms / meanReference : rms > 0 ? Infinity : 0,
        nonFiniteValues: nonFinite,
    };
}

/** Rec. 709 luminance of an interleaved RGB array, one value per pixel. */
export function luminance(rgb: Float32Array): Float32Array {
    const out = new Float32Array(rgb.length / 3);
    for (let pixel = 0; pixel < out.length; pixel++) {
        out[pixel] = 0.2126 * rgb[pixel * 3] + 0.7152 * rgb[pixel * 3 + 1] + 0.0722 * rgb[pixel * 3 + 2];
    }
    return out;
}

/**
 * Per-pixel relative error of luminance, for the heatmap only:
 * |T - R| / (R + floor), where floor is `floorFraction` of the mean reference
 * luminance. The floor stops a black reference pixel from reading as infinite
 * error. The summary numbers above do not use this. A pixel where either image
 * is not finite gets NaN.
 */
export function relativeErrorMap(reference: Float32Array, test: Float32Array, floorFraction = 0.05): Float32Array {
    const referenceLuminance = luminance(reference);
    const testLuminance = luminance(test);
    let sum = 0;
    let count = 0;
    for (const value of referenceLuminance) {
        if (Number.isFinite(value)) {
            sum += value;
            count++;
        }
    }
    const floor = (count > 0 ? sum / count : 0) * floorFraction;
    const out = new Float32Array(referenceLuminance.length);
    for (let pixel = 0; pixel < out.length; pixel++) {
        const r = referenceLuminance[pixel];
        const t = testLuminance[pixel];
        // NaN marks "no data" (a NaN/Inf pixel, or a block with no finite pixel):
        // percentile() skips it, and the heatmap paints it at full error so it shows.
        out[pixel] = Number.isFinite(r) && Number.isFinite(t) ? Math.abs(t - r) / (r + floor + 1e-12) : NaN;
    }
    return out;
}

/** Which pixels can be compared: every channel finite in BOTH images. */
export interface FinitePixels {
    /** One entry per pixel: 1 usable, 0 has a NaN or infinity in either image. */
    usable: Uint8Array;
    /** Pixels with at least one non-finite channel in the reference, in the test, and in either. */
    referenceBad: number;
    testBad: number;
    eitherBad: number;
}

function pixelIsFinite(rgb: Float32Array, pixel: number): boolean {
    return Number.isFinite(rgb[pixel * 3]) && Number.isFinite(rgb[pixel * 3 + 1]) && Number.isFinite(rgb[pixel * 3 + 2]);
}

/**
 * Finds the NaN/infinite pixels of a pair. A pixel bad in one image is left
 * out of BOTH, so the two block averages are always over the same pixels.
 */
export function findFinitePixels(reference: Float32Array, test: Float32Array): FinitePixels {
    if (reference.length !== test.length) {
        throw new Error(`images differ in size: ${reference.length / 3} pixels vs ${test.length / 3}`);
    }
    const pixelCount = reference.length / 3;
    const usable = new Uint8Array(pixelCount);
    let referenceBad = 0;
    let testBad = 0;
    let eitherBad = 0;
    for (let pixel = 0; pixel < pixelCount; pixel++) {
        const referenceOk = pixelIsFinite(reference, pixel);
        const testOk = pixelIsFinite(test, pixel);
        if (!referenceOk) referenceBad++;
        if (!testOk) testBad++;
        if (referenceOk && testOk) usable[pixel] = 1;
        else eitherBad++;
    }
    return { usable, referenceBad, testBad, eitherBad };
}

/** A copy with every unusable pixel set to NaN in all channels, so every sum skips it. */
export function maskImage(rgb: Float32Array, usable: Uint8Array): Float32Array {
    const out = Float32Array.from(rgb);
    for (let pixel = 0; pixel < usable.length; pixel++) {
        if (!usable[pixel]) out.fill(NaN, pixel * 3, pixel * 3 + 3);
    }
    return out;
}

/**
 * Averages each `factor` x `factor` block of pixels into one (edge blocks may be
 * smaller). Both images of a pair are downsampled the same way before
 * comparing, to take most of the Monte Carlo noise out: noise falls by about
 * `factor`, while a real difference in the lighting survives.
 *
 * Only usable pixels are averaged: those marked in `usable` (from
 * findFinitePixels, so both images of a pair skip the same pixels), or, without
 * it, those with all channels finite. One NaN would otherwise poison its whole
 * block. A block with no usable pixel is NaN, and counted in `emptyBlocks`.
 */
export function downsample(
    rgb: Float32Array,
    width: number,
    height: number,
    factor: number,
    usable?: Uint8Array,
): { width: number; height: number; rgb: Float32Array; emptyBlocks: number } {
    if (factor <= 1) return { width, height, rgb, emptyBlocks: 0 };
    const outWidth = Math.ceil(width / factor);
    const outHeight = Math.ceil(height / factor);
    const out = new Float32Array(outWidth * outHeight * 3);
    let emptyBlocks = 0;
    for (let blockY = 0; blockY < outHeight; blockY++) {
        for (let blockX = 0; blockX < outWidth; blockX++) {
            const sums = [0, 0, 0];
            let count = 0;
            for (let y = blockY * factor; y < Math.min(height, (blockY + 1) * factor); y++) {
                for (let x = blockX * factor; x < Math.min(width, (blockX + 1) * factor); x++) {
                    const pixel = y * width + x;
                    const ok = usable ? usable[pixel] === 1 : pixelIsFinite(rgb, pixel);
                    if (!ok) continue;
                    for (let channel = 0; channel < 3; channel++) sums[channel] += rgb[pixel * 3 + channel];
                    count++;
                }
            }
            if (count === 0) emptyBlocks++;
            const target = (blockY * outWidth + blockX) * 3;
            for (let channel = 0; channel < 3; channel++) out[target + channel] = count > 0 ? sums[channel] / count : NaN;
        }
    }
    return { width: outWidth, height: outHeight, rgb: out, emptyBlocks };
}

/**
 * For FLIP, which must never see a NaN or infinity: copies of both images with
 * every pixel that is non-finite in EITHER set to 0 in both, and how many.
 */
export function zeroNonFinitePixels(reference: Float32Array, test: Float32Array): { reference: Float32Array; test: Float32Array; zeroedPixels: number } {
    const { usable, eitherBad } = findFinitePixels(reference, test);
    if (eitherBad === 0) return { reference, test, zeroedPixels: 0 };
    const zero = (rgb: Float32Array): Float32Array => {
        const out = Float32Array.from(rgb);
        for (let pixel = 0; pixel < usable.length; pixel++) {
            if (!usable[pixel]) out.fill(0, pixel * 3, pixel * 3 + 3);
        }
        return out;
    };
    return { reference: zero(reference), test: zero(test), zeroedPixels: eitherBad };
}

/** The p-th percentile (0 to 100) by linear interpolation between the two nearest ranks. Ignores non-finite values. */
export function percentile(values: ArrayLike<number>, p: number): number {
    const sorted = Float64Array.from(Array.from(values).filter(Number.isFinite)).sort();
    if (sorted.length === 0) return 0;
    const position = (Math.min(Math.max(p, 0), 100) / 100) * (sorted.length - 1);
    const lower = Math.floor(position);
    const upper = Math.min(lower + 1, sorted.length - 1);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

/**
 * Limits every finite value to at most `limit`. NaN and infinities are left as
 * they are, so they are still found and skipped like everywhere else (an
 * infinity clamped to the limit would pass as a real, bright pixel).
 */
export function clampImage(rgb: Float32Array, limit: number): Float32Array {
    return rgb.map((value) => (Number.isFinite(value) && value > limit ? limit : value));
}

/** Multiplies every value; used by tests. */
export function scaleImage(rgb: Float32Array, factor: number): Float32Array {
    return rgb.map((value) => value * factor);
}
