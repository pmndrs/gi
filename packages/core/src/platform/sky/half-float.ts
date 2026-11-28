/**
 * IEEE 754 binary16 ("half float") conversion, and the RGB half image the sky
 * code passes around. Engine-free and Node-free.
 *
 * Moved here from `scenes/src/sky/exr-half.ts` (T1.4) so the platform's
 * `procedural` environment can build the very same half floats as the shared
 * sky files; the EXR codec stays in `scenes` and re-exports these.
 */

/**
 * An RGB image of half floats in image order: row 0 is the top row (the zenith,
 * for an equirectangular sky). GPU textures of the platform use the opposite
 * row order (see `equirect.ts`), so uploads flip the rows.
 */
export interface HalfRgbImage {
    width: number;
    height: number;
    /** IEEE 754 binary16 bit patterns, interleaved R, G, B; length width * height * 3. */
    halves: Uint16Array;
}

const float32Scratch = new Float32Array(1);
const uint32Scratch = new Uint32Array(float32Scratch.buffer);

/**
 * float32 to binary16 with round-to-nearest-even, the IEEE default, so a
 * value is never biased up or down on average. Values above the largest half
 * (65504) are an error rather than silently becoming infinity: a sky that
 * bright means a units mistake upstream.
 */
export function floatToHalf(value: number): number {
    if (Number.isNaN(value)) throw new Error('cannot store NaN in the sky file');
    float32Scratch[0] = value;
    const bits = uint32Scratch[0];
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    const mantissa = bits & 0x7fffff;

    if (exponent === 0xff) throw new Error(`cannot store ${value} in the sky file`);
    // Re-bias the exponent from float32 (127) to half (15).
    const halfExponent = exponent - 127 + 15;
    if (halfExponent >= 0x1f) throw new Error(`${value} is larger than the largest half float (65504)`);

    if (halfExponent <= 0) {
        // Subnormal half (or zero): shift the mantissa, with its implicit 1, into place.
        if (halfExponent < -10) return sign;
        const fullMantissa = mantissa | 0x800000;
        const shift = 14 - halfExponent;
        const halfMantissa = fullMantissa >>> shift;
        const remainder = fullMantissa & ((1 << shift) - 1);
        const halfway = 1 << (shift - 1);
        const roundUp = remainder > halfway || (remainder === halfway && (halfMantissa & 1) === 1);
        return sign | (halfMantissa + (roundUp ? 1 : 0));
    }

    const halfMantissa = mantissa >>> 13;
    const remainder = mantissa & 0x1fff;
    const roundUp = remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) === 1);
    // A carry out of the mantissa correctly bumps the exponent.
    const rounded = ((halfExponent << 10) | halfMantissa) + (roundUp ? 1 : 0);
    if (rounded >= 0x7c00) throw new Error(`${value} is larger than the largest half float (65504)`);
    return sign | rounded;
}

export function halfToFloat(half: number): number {
    const sign = (half & 0x8000) !== 0 ? -1 : 1;
    const exponent = (half >> 10) & 0x1f;
    const fraction = half & 0x3ff;
    if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
    if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
    return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** binary16 1.0, the alpha of every texel. */
const HALF_ONE = 0x3c00;

/**
 * An image-order RGB half image to the RGBA halves of an `rgba16float`
 * texture in the platform's equirect texture order: rows flipped, so the
 * first row in memory is the nadir and the last the zenith (`equirect.ts`),
 * and alpha 1. The values are copied bit for bit.
 */
export function halfImageToTextureRgba(image: HalfRgbImage): Uint16Array {
    const { width, height, halves } = image;
    const rgba = new Uint16Array(width * height * 4);
    for (let row = 0; row < height; row++) {
        const sourceRow = height - 1 - row;
        for (let column = 0; column < width; column++) {
            const source = (sourceRow * width + column) * 3;
            const target = (row * width + column) * 4;
            rgba[target] = halves[source];
            rgba[target + 1] = halves[source + 1];
            rgba[target + 2] = halves[source + 2];
            rgba[target + 3] = HALF_ONE;
        }
    }
    return rgba;
}
