/**
 * Image difference numbers. Pure functions on interleaved RGB float arrays,
 * with no I/O, so the maths can be unit-tested.
 *
 * Definitions (also printed by `tools/compare --help`). R is the reference
 * (the first image) and T the test (the second), summed over every pixel and
 * every colour channel; non-finite values are skipped in both and counted:
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
 * error. The summary numbers above do not use this.
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
        out[pixel] = Number.isFinite(r) && Number.isFinite(t) ? Math.abs(t - r) / (r + floor + 1e-12) : 1;
    }
    return out;
}

/**
 * Averages each `factor` x `factor` block of pixels into one (edge blocks may be
 * smaller). Both images of a pair are downsampled the same way before
 * comparing, to take most of the Monte Carlo noise out: noise falls by about
 * `factor`, while a real difference in the lighting survives.
 */
export function downsample(rgb: Float32Array, width: number, height: number, factor: number): { width: number; height: number; rgb: Float32Array } {
    if (factor <= 1) return { width, height, rgb };
    const outWidth = Math.ceil(width / factor);
    const outHeight = Math.ceil(height / factor);
    const out = new Float32Array(outWidth * outHeight * 3);
    for (let blockY = 0; blockY < outHeight; blockY++) {
        for (let blockX = 0; blockX < outWidth; blockX++) {
            const sums = [0, 0, 0];
            let count = 0;
            for (let y = blockY * factor; y < Math.min(height, (blockY + 1) * factor); y++) {
                for (let x = blockX * factor; x < Math.min(width, (blockX + 1) * factor); x++) {
                    for (let channel = 0; channel < 3; channel++) sums[channel] += rgb[(y * width + x) * 3 + channel];
                    count++;
                }
            }
            for (let channel = 0; channel < 3; channel++) out[(blockY * outWidth + blockX) * 3 + channel] = sums[channel] / count;
        }
    }
    return { width: outWidth, height: outHeight, rgb: out };
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

/** Limits every value to at most `limit`. */
export function clampImage(rgb: Float32Array, limit: number): Float32Array {
    return rgb.map((value) => (value > limit ? limit : value));
}

/** Multiplies every value; used by tests. */
export function scaleImage(rgb: Float32Array, factor: number): Float32Array {
    return rgb.map((value) => value * factor);
}
