/**
 * Real spherical harmonics for the sky's irradiance: the basis, the cosine
 * convolution, and CPU versions of what the GPU projection computes (for
 * tests and reference numbers).
 *
 * Why bands 0, 1, 2 and 4 (18 coefficients), not the usual 0-2 (9):
 * irradiance is radiance convolved with the clamped cosine, and in the SH
 * domain that convolution multiplies band l by a constant A_l (Ramamoorthi and
 * Hanrahan, "An Efficient Representation for Irradiance Environment Maps",
 * SIGGRAPH 2001):
 *
 *   A_0 = pi, A_1 = 2 pi / 3, A_2 = pi / 4, A_3 = 0, A_4 = -pi / 24, ...
 *
 * Bands 0-2 are the classic choice, but they leave out A_4, and on our skies
 * (bright horizon, sun glow, black below the horizon) that costs up to 0.9%
 * on an upward-facing ground, beyond the ±0.5% reference tolerance. Measured
 * on the committed sky files (open-ground, R/G/B): bands 0-2 +0.74 / -0.33 /
 * -0.87%; adding band 4: +0.30 / -0.11 / -0.26% (sponza and furnished-room
 * alike, worst 0.35%). Band 3 is multiplied by A_3 = 0, so it would add
 * nothing to irradiance and is skipped. DECISIONS 2026-11-05.
 *
 * Basis convention: the standard real SH in Cartesian form, with z as the
 * polar axis. Which axis is polar doesn't matter for a complete set of bands
 * (each band is closed under rotation), so world +Y up needs no rotation.
 * The order below is the order of the coefficients everywhere (CPU, the GPU
 * uniform, the WGSL).
 */

/** Bands present, in coefficient order. */
export const SH_BANDS = [0, 1, 2, 4] as const;

/** 1 + 3 + 5 + 9. */
export const SH_COEFFICIENT_COUNT = 18;

/** The band of each coefficient, in order. */
export const SH_COEFFICIENT_BANDS: readonly number[] = SH_BANDS.flatMap((band) => new Array<number>(2 * band + 1).fill(band));

/**
 * The clamped-cosine convolution constant A_l of each band present
 * (Ramamoorthi and Hanrahan 2001, equation 8; A_4 = -pi / 24).
 */
export function cosineLobeBandWeight(band: number): number {
    switch (band) {
        case 0:
            return Math.PI;
        case 1:
            return (2 * Math.PI) / 3;
        case 2:
            return Math.PI / 4;
        case 4:
            return -Math.PI / 24;
        default:
            throw new Error(`band ${band} is not stored`);
    }
}

/**
 * The 18 basis functions at a unit direction, in coefficient order. The
 * normalisation constants are the standard ones (for example
 * Y_4^0 = (3 / 16) sqrt(1 / pi) (35 z^4 - 30 z^2 + 3)); the unit test checks
 * that the set is orthonormal.
 */
export function shBasis(x: number, y: number, z: number, out: Float64Array = new Float64Array(SH_COEFFICIENT_COUNT)): Float64Array {
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    // Band 0
    out[0] = 0.28209479177387814;
    // Band 1
    out[1] = 0.4886025119029199 * y;
    out[2] = 0.4886025119029199 * z;
    out[3] = 0.4886025119029199 * x;
    // Band 2
    out[4] = 1.0925484305920792 * x * y;
    out[5] = 1.0925484305920792 * y * z;
    out[6] = 0.31539156525252005 * (3 * zz - 1);
    out[7] = 1.0925484305920792 * x * z;
    out[8] = 0.5462742152960396 * (xx - yy);
    // Band 4
    out[9] = 2.5033429417967046 * x * y * (xx - yy);
    out[10] = 1.7701307697799304 * y * z * (3 * xx - yy);
    out[11] = 0.9461746957575601 * x * y * (7 * zz - 1);
    out[12] = 0.6690465435572892 * y * z * (7 * zz - 3);
    out[13] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3);
    out[14] = 0.6690465435572892 * x * z * (7 * zz - 3);
    out[15] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1);
    out[16] = 1.7701307697799304 * x * z * (xx - 3 * yy);
    out[17] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return out;
}

/**
 * The same basis in WGSL, `giSkyShBasis(d) -> array<f32, 18>`. Written out
 * rather than generated so it reads like the formulas above; the GPU tests
 * compare its irradiance with the CPU's.
 */
export const SH_BASIS_WGSL = /* wgsl */ `
// Real SH bands 0, 1, 2 and 4 at a unit direction, z as the polar axis
// (platform/sky/spherical-harmonics.ts has the same functions on the CPU).
fn giSkyShBasis(d: vec3f) -> array<f32, 18> {
    let xx = d.x * d.x;
    let yy = d.y * d.y;
    let zz = d.z * d.z;
    return array<f32, 18>(
        0.28209479177387814,
        0.4886025119029199 * d.y,
        0.4886025119029199 * d.z,
        0.4886025119029199 * d.x,
        1.0925484305920792 * d.x * d.y,
        1.0925484305920792 * d.y * d.z,
        0.31539156525252005 * (3.0 * zz - 1.0),
        1.0925484305920792 * d.x * d.z,
        0.5462742152960396 * (xx - yy),
        2.5033429417967046 * d.x * d.y * (xx - yy),
        1.7701307697799304 * d.y * d.z * (3.0 * xx - yy),
        0.9461746957575601 * d.x * d.y * (7.0 * zz - 1.0),
        0.6690465435572892 * d.y * d.z * (7.0 * zz - 3.0),
        0.10578554691520431 * (35.0 * zz * zz - 30.0 * zz + 3.0),
        0.6690465435572892 * d.x * d.z * (7.0 * zz - 3.0),
        0.47308734787878004 * (xx - yy) * (7.0 * zz - 1.0),
        1.7701307697799304 * d.x * d.z * (xx - 3.0 * yy),
        0.6258357354491761 * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy)),
    );
}
`;

/** Radiance SH (18 coefficients x RGB, coefficient-major: c0.r, c0.g, c0.b, c1.r, ...). */
export type ShCoefficients = Float64Array;

/**
 * Projects an equirect radiance image (image order, row 0 the zenith, RGB
 * floats) onto the basis, each texel taken as constant over its exact solid
 * angle, with the basis at the texel centre: the same sum the GPU projection
 * makes. Returns radiance coefficients (not yet convolved).
 */
export function projectEquirectRadiance(rgb: ArrayLike<number>, width: number, height: number): ShCoefficients {
    const coefficients = new Float64Array(SH_COEFFICIENT_COUNT * 3);
    const basis = new Float64Array(SH_COEFFICIENT_COUNT);
    for (let row = 0; row < height; row++) {
        const theta0 = (row / height) * Math.PI;
        const theta1 = ((row + 1) / height) * Math.PI;
        const solidAngle = ((2 * Math.PI) / width) * (Math.cos(theta0) - Math.cos(theta1));
        const polar = ((row + 0.5) / height) * Math.PI;
        for (let column = 0; column < width; column++) {
            const azimuth = ((column + 0.5) / width - 0.5) * 2 * Math.PI;
            const horizontal = Math.sin(polar);
            shBasis(horizontal * Math.cos(azimuth), Math.cos(polar), horizontal * Math.sin(azimuth), basis);
            const base = (row * width + column) * 3;
            for (let index = 0; index < SH_COEFFICIENT_COUNT; index++) {
                const weight = basis[index] * solidAngle;
                coefficients[index * 3] += rgb[base] * weight;
                coefficients[index * 3 + 1] += rgb[base + 1] * weight;
                coefficients[index * 3 + 2] += rgb[base + 2] * weight;
            }
        }
    }
    return coefficients;
}

/** Radiance coefficients to irradiance coefficients: band l times A_l. */
export function convolveWithCosineLobe(radiance: ShCoefficients): ShCoefficients {
    const irradiance = new Float64Array(radiance.length);
    for (let index = 0; index < SH_COEFFICIENT_COUNT; index++) {
        const weight = cosineLobeBandWeight(SH_COEFFICIENT_BANDS[index]);
        for (let channel = 0; channel < 3; channel++) irradiance[index * 3 + channel] = radiance[index * 3 + channel] * weight;
    }
    return irradiance;
}

/** Irradiance at a unit normal from irradiance coefficients, clamped at 0 as the GPU does. */
export function evaluateIrradiance(irradiance: ShCoefficients, normal: readonly [number, number, number]): [number, number, number] {
    const basis = shBasis(normal[0], normal[1], normal[2]);
    const result: [number, number, number] = [0, 0, 0];
    for (let index = 0; index < SH_COEFFICIENT_COUNT; index++) {
        for (let channel = 0; channel < 3; channel++) result[channel] += irradiance[index * 3 + channel] * basis[index];
    }
    return [Math.max(0, result[0]), Math.max(0, result[1]), Math.max(0, result[2])];
}

/**
 * Brute-force irradiance at a unit normal from an equirect image (image
 * order), each texel's radiance times cos+ at its centre times its solid
 * angle. The reference the SH answer is compared with.
 */
export function bruteForceIrradiance(rgb: ArrayLike<number>, width: number, height: number, normal: readonly [number, number, number]): [number, number, number] {
    const result: [number, number, number] = [0, 0, 0];
    for (let row = 0; row < height; row++) {
        const theta0 = (row / height) * Math.PI;
        const theta1 = ((row + 1) / height) * Math.PI;
        const solidAngle = ((2 * Math.PI) / width) * (Math.cos(theta0) - Math.cos(theta1));
        const polar = ((row + 0.5) / height) * Math.PI;
        for (let column = 0; column < width; column++) {
            const azimuth = ((column + 0.5) / width - 0.5) * 2 * Math.PI;
            const horizontal = Math.sin(polar);
            const cosine = normal[0] * horizontal * Math.cos(azimuth) + normal[1] * Math.cos(polar) + normal[2] * horizontal * Math.sin(azimuth);
            if (cosine <= 0) continue;
            const base = (row * width + column) * 3;
            const weight = cosine * solidAngle;
            result[0] += rgb[base] * weight;
            result[1] += rgb[base + 1] * weight;
            result[2] += rgb[base + 2] * weight;
        }
    }
    return result;
}
