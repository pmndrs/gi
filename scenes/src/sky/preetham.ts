/**
 * The Preetham clear-sky model: sky luminance and colour for a view direction,
 * given the sun's position and the atmosphere's turbidity.
 *
 * Source: A. J. Preetham, P. Shirley, B. Smits, "A Practical Analytic Model for
 * Daylight", SIGGRAPH 1999, section 3 and Appendix A.2. Implemented from the
 * paper's equations and tables, not from any existing code.
 *
 * The model has three parts:
 *  - Perez et al.'s five-parameter sky distribution F(theta, gamma), which
 *    shapes the sky relative to its zenith: brighter near the horizon and in a
 *    glow around the sun (the "circumsolar" region);
 *  - zenith values (luminance Yz, chromaticity xz, yz) as functions of the
 *    turbidity T and the sun's zenith angle;
 *  - for each of Y, x and y: value(theta, gamma) = zenith * F(theta, gamma) / F(0, thetaSun).
 *
 * Angles: theta is the view direction's angle from the zenith, gamma the angle
 * between the view direction and the sun, thetaSun the sun's angle from the
 * zenith. Everything is in radians.
 *
 * Turbidity T is the ratio of the optical thickness of the hazy atmosphere to
 * a molecules-only one. The paper fits the model for roughly T = 2 to 10; below
 * about 1.7 the luminance coefficients go wrong (negative values), so
 * {@link checkTurbidity} refuses them.
 *
 * The absolute luminance Yz is in kcd/m^2, but this project rescales the
 * whole sky to a calibrated brightness (`sky-image.ts`), so only the sky's
 * shape and colour from this file are used.
 */

/** The five Perez coefficients A..E for one quantity (Y, x or y). */
export interface PerezCoefficients {
    /** Darkening or brightening towards the horizon. */
    a: number;
    /** Luminance gradient near the horizon. */
    b: number;
    /** Relative intensity of the circumsolar region. */
    c: number;
    /** Width of the circumsolar region. */
    d: number;
    /** Relative backscattered light. */
    e: number;
}

/** Luminance (any unit) and CIE xy chromaticity. */
export interface SkyXyY {
    x: number;
    y: number;
    luminance: number;
}

export const MIN_TURBIDITY = 1.7;
export const MAX_TURBIDITY = 20;

export function checkTurbidity(turbidity: number): void {
    if (!Number.isFinite(turbidity) || turbidity < MIN_TURBIDITY || turbidity > MAX_TURBIDITY) {
        throw new Error(`turbidity ${turbidity} is outside the Preetham model's range (${MIN_TURBIDITY} to ${MAX_TURBIDITY})`);
    }
}

/** Paper, Appendix A.2: distribution coefficients for luminance Y, linear in T. */
export function luminanceCoefficients(turbidity: number): PerezCoefficients {
    const t = turbidity;
    return {
        a: 0.1787 * t - 1.463,
        b: -0.3554 * t + 0.4275,
        c: -0.0227 * t + 5.3251,
        d: 0.1206 * t - 2.5771,
        e: -0.067 * t + 0.3703,
    };
}

/** Paper, Appendix A.2: distribution coefficients for chromaticity x. */
export function chromaticityXCoefficients(turbidity: number): PerezCoefficients {
    const t = turbidity;
    return {
        a: -0.0193 * t - 0.2592,
        b: -0.0665 * t + 0.0008,
        c: -0.0004 * t + 0.2125,
        d: -0.0641 * t - 0.8989,
        e: -0.0033 * t + 0.0452,
    };
}

/** Paper, Appendix A.2: distribution coefficients for chromaticity y. */
export function chromaticityYCoefficients(turbidity: number): PerezCoefficients {
    const t = turbidity;
    return {
        a: -0.0167 * t - 0.2608,
        b: -0.095 * t + 0.0092,
        c: -0.0079 * t + 0.2102,
        d: -0.0441 * t - 1.6537,
        e: -0.0109 * t + 0.0529,
    };
}

/**
 * Perez et al.'s sky distribution:
 *   F(theta, gamma) = (1 + A exp(B / cos theta)) (1 + C exp(D gamma) + E cos^2 gamma)
 *
 * At the horizon cos(theta) goes to 0 and B is negative, so exp(B / cos theta)
 * goes to 0; a tiny floor on cos(theta) gives that limit without dividing by 0.
 */
export function perez(coefficients: PerezCoefficients, theta: number, gamma: number): number {
    const cosTheta = Math.max(Math.cos(theta), 1e-6);
    const cosGamma = Math.cos(gamma);
    const horizonTerm = 1 + coefficients.a * Math.exp(coefficients.b / cosTheta);
    const sunTerm = 1 + coefficients.c * Math.exp(coefficients.d * gamma) + coefficients.e * cosGamma * cosGamma;
    return horizonTerm * sunTerm;
}

/**
 * Zenith luminance in kcd/m^2 (paper, Appendix A.2):
 *   chi = (4/9 - T/120) (pi - 2 thetaSun)
 *   Yz  = (4.0453 T - 4.9710) tan(chi) - 0.2155 T + 2.4192
 */
export function zenithLuminance(turbidity: number, sunZenith: number): number {
    const chi = (4 / 9 - turbidity / 120) * (Math.PI - 2 * sunZenith);
    return (4.0453 * turbidity - 4.971) * Math.tan(chi) - 0.2155 * turbidity + 2.4192;
}

/**
 * Zenith chromaticity: [T^2, T, 1] * M * [thetaSun^3, thetaSun^2, thetaSun, 1]
 * with the paper's 3x4 matrices for x and y (Appendix A.2).
 */
const ZENITH_X_MATRIX: readonly (readonly number[])[] = [
    [0.00166, -0.00375, 0.00209, 0],
    [-0.02903, 0.06377, -0.03202, 0.00394],
    [0.11693, -0.21196, 0.06052, 0.25886],
];

const ZENITH_Y_MATRIX: readonly (readonly number[])[] = [
    [0.00275, -0.0061, 0.00317, 0],
    [-0.04214, 0.0897, -0.04153, 0.00516],
    [0.15346, -0.26756, 0.0667, 0.26688],
];

function zenithChromaticity(matrix: readonly (readonly number[])[], turbidity: number, sunZenith: number): number {
    const turbidityPowers = [turbidity * turbidity, turbidity, 1];
    const anglePowers = [sunZenith ** 3, sunZenith ** 2, sunZenith, 1];
    let sum = 0;
    for (let row = 0; row < 3; row++) {
        for (let column = 0; column < 4; column++) {
            sum += turbidityPowers[row] * matrix[row][column] * anglePowers[column];
        }
    }
    return sum;
}

/**
 * Precomputed per-sky values, so evaluating many directions doesn't redo the
 * zenith terms. Build with {@link createPreethamSky}.
 */
export interface PreethamSky {
    turbidity: number;
    sunZenith: number;
    /** Unit vector towards the sun. */
    sunDirection: readonly [number, number, number];
    luminance: PerezCoefficients;
    chromaticityX: PerezCoefficients;
    chromaticityY: PerezCoefficients;
    zenith: SkyXyY;
    /** F(0, thetaSun) for Y, x and y: the normalisers that make the zenith come out at the zenith values. */
    zenithPerez: { luminance: number; x: number; y: number };
}

/**
 * @param sunDirection direction towards the sun, +Y up (it is normalised here)
 * @param turbidity see {@link checkTurbidity}
 */
export function createPreethamSky(sunDirection: readonly [number, number, number], turbidity: number): PreethamSky {
    checkTurbidity(turbidity);
    const length = Math.hypot(sunDirection[0], sunDirection[1], sunDirection[2]);
    if (!(length > 0)) throw new Error('sunDirection must not be a zero vector');
    const sun: [number, number, number] = [sunDirection[0] / length, sunDirection[1] / length, sunDirection[2] / length];
    if (sun[1] <= 0) throw new Error('the Preetham model needs the sun above the horizon (sunDirection y > 0)');
    const sunZenith = Math.acos(Math.min(1, sun[1]));

    const luminance = luminanceCoefficients(turbidity);
    const chromaticityX = chromaticityXCoefficients(turbidity);
    const chromaticityY = chromaticityYCoefficients(turbidity);
    return {
        turbidity,
        sunZenith,
        sunDirection: sun,
        luminance,
        chromaticityX,
        chromaticityY,
        zenith: {
            luminance: zenithLuminance(turbidity, sunZenith),
            x: zenithChromaticity(ZENITH_X_MATRIX, turbidity, sunZenith),
            y: zenithChromaticity(ZENITH_Y_MATRIX, turbidity, sunZenith),
        },
        // The zenith direction has theta = 0 and gamma = thetaSun.
        zenithPerez: {
            luminance: perez(luminance, 0, sunZenith),
            x: perez(chromaticityX, 0, sunZenith),
            y: perez(chromaticityY, 0, sunZenith),
        },
    };
}

/**
 * Sky luminance (kcd/m^2) and chromaticity seen along a unit direction.
 * Only defined above the horizon (direction y > 0); callers decide what lies below.
 */
export function preethamSkyXyY(sky: PreethamSky, direction: readonly [number, number, number]): SkyXyY {
    const theta = Math.acos(Math.min(1, Math.max(-1, direction[1])));
    const cosGamma = direction[0] * sky.sunDirection[0] + direction[1] * sky.sunDirection[1] + direction[2] * sky.sunDirection[2];
    const gamma = Math.acos(Math.min(1, Math.max(-1, cosGamma)));
    return {
        luminance: (sky.zenith.luminance * perez(sky.luminance, theta, gamma)) / sky.zenithPerez.luminance,
        x: (sky.zenith.x * perez(sky.chromaticityX, theta, gamma)) / sky.zenithPerez.x,
        y: (sky.zenith.y * perez(sky.chromaticityY, theta, gamma)) / sky.zenithPerez.y,
    };
}

/**
 * CIE xyY to linear Rec.709 / sRGB primaries (D65 white), which is what three.js
 * and Blender's "Linear Rec.709" treat as linear RGB. The matrix is the
 * standard XYZ-to-linear-sRGB one (IEC 61966-2-1). Negative channels, which
 * only happen for colours outside the sRGB gamut, are clamped to 0.
 */
export function xyYToLinearRgb(colour: SkyXyY): [number, number, number] {
    if (colour.y <= 0) return [0, 0, 0];
    const bigY = colour.luminance;
    const bigX = (colour.x / colour.y) * bigY;
    const bigZ = ((1 - colour.x - colour.y) / colour.y) * bigY;
    const r = 3.2404542 * bigX - 1.5371385 * bigY - 0.4985314 * bigZ;
    const g = -0.969266 * bigX + 1.8760108 * bigY + 0.041556 * bigZ;
    const b = 0.0556434 * bigX - 0.2040259 * bigY + 1.0572252 * bigZ;
    return [Math.max(0, r), Math.max(0, g), Math.max(0, b)];
}

/** Rec.709 luminance weights: the Y row of the linear-sRGB-to-XYZ matrix. */
export function rec709Luminance(r: number, g: number, b: number): number {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}
