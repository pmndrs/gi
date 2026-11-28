/**
 * The calibrated Preetham sky as an equirectangular half-float image: the
 * shared code behind the sky files (`tools/sky/generate.ts`, through
 * `scenes/src/sky/sky-image.ts`) and the platform's `procedural` environment.
 * One function, so both make the same pixels (T1.4; DECISIONS 2026-11-04).
 *
 * What is in the image:
 *  - Above the horizon: the Preetham sky (`preetham.ts`) for the sun direction
 *    and turbidity, in linear Rec.709 RGB.
 *  - No sun disc. The sun is a directional light, which every renderer
 *    already has; drawing it into the sky too would count it twice.
 *  - Below the horizon: black. The ground is scene geometry.
 *
 * Brightness (the calibration): the image is scaled so that a flat, open,
 * upward-facing ground gets sky irradiance equal to a realistic share of the
 * sun's irradiance on that same ground:
 *
 *   E_sky,horizontal = ratio(elevation, turbidity) * E_sun * sin(elevation) * intensity
 *
 * where `ratio` is the Ineichen-Perez clear-sky diffuse-to-direct horizontal
 * ratio (`clear-sky-ratio.ts`) and E_sun the sun's luminance-weighted
 * illuminance at normal incidence. Luminance is the Rec.709 Y of the stored
 * RGB. The irradiance is measured on the half floats actually written, so what
 * a reader integrates is what the calibration promised, to half-float rounding.
 */

import { ineichenPerezClearSky, type ClearSkyIrradiance } from './clear-sky-ratio.ts';
import { rowHorizontalIrradianceWeight, texelCentreDirection } from './equirect.ts';
import { floatToHalf, halfToFloat, type HalfRgbImage } from './half-float.ts';
import { createPreethamSky, preethamSkyXyY, rec709Luminance, xyYToLinearRgb } from './preetham.ts';

/** 512 x 256 texels, 0.7 degrees each: the sky has no sun disc and no sharp detail, so more would only make the file bigger. */
export const DEFAULT_SKY_WIDTH = 512;
export const DEFAULT_SKY_HEIGHT = 256;

/**
 * Preetham turbidity when none is given: the sky files' default (the viewer's
 * old `SkyMesh` default), and the platform's `procedural` environment's.
 * The one default for both (T1.4).
 */
export const DEFAULT_TURBIDITY = 2;

export interface SkyCalibration {
    model: string;
    clearSky: ClearSkyIrradiance;
    /** A plain multiplier on top of the calibrated sky (1 = as calibrated). */
    intensity: number;
    /** E_sun * sin(elevation): the sun's luminance-weighted irradiance on the ground. */
    sunHorizontalIlluminance: number;
    /** What the sky's luminance-weighted horizontal irradiance was scaled to. */
    targetSkyHorizontalIlluminance: number;
}

/** Horizontal (upward-facing ground) irradiance delivered by an image, per channel and as luminance. */
export interface HorizontalIrradiance {
    rgb: [number, number, number];
    luminance: number;
}

export interface CalibratedSkyOptions {
    /** Direction towards the sun, +Y up; need not be normalised. The sun must be at least 5 degrees up (the clear-sky model's range). */
    sunDirection: readonly [number, number, number];
    /** Preetham turbidity, 1.7 to 20. Defaults to {@link DEFAULT_TURBIDITY}. */
    turbidity?: number;
    /** E_sun: the luminance-weighted illuminance, at normal incidence, of the sun this sky goes with. */
    sunIlluminance: number;
    /** Multiplier on top of the calibration. Defaults to 1. */
    intensity?: number;
    width?: number;
    height?: number;
}

export interface CalibratedSky {
    /** Image order: row 0 is the zenith (see `equirect.ts`). */
    image: HalfRgbImage;
    /** Unit vector towards the sun. */
    sunDirection: [number, number, number];
    sunElevationRad: number;
    turbidity: number;
    calibration: SkyCalibration;
    /** Measured on the stored half floats. */
    measured: HorizontalIrradiance;
}

/** Sums radiance times each texel's horizontal-irradiance weight over the upper hemisphere. */
export function measureHorizontalIrradiance(width: number, height: number, radianceAt: (index: number) => number): HorizontalIrradiance {
    const rgb: [number, number, number] = [0, 0, 0];
    for (let row = 0; row < height; row++) {
        const weight = rowHorizontalIrradianceWeight(row, width, height);
        if (weight === 0) continue;
        for (let column = 0; column < width; column++) {
            const base = (row * width + column) * 3;
            rgb[0] += radianceAt(base) * weight;
            rgb[1] += radianceAt(base + 1) * weight;
            rgb[2] += radianceAt(base + 2) * weight;
        }
    }
    return { rgb, luminance: rec709Luminance(rgb[0], rgb[1], rgb[2]) };
}

export function measureHalfImage(image: HalfRgbImage): HorizontalIrradiance {
    return measureHorizontalIrradiance(image.width, image.height, (index) => halfToFloat(image.halves[index]));
}

/** The unit vector towards the sun and its elevation, computed the one way both callers use. */
export function sunDirectionAndElevation(sunDirection: readonly [number, number, number]): { direction: [number, number, number]; elevationRad: number } {
    const length = Math.hypot(sunDirection[0], sunDirection[1], sunDirection[2]);
    const direction: [number, number, number] = [sunDirection[0] / length, sunDirection[1] / length, sunDirection[2] / length];
    return { direction, elevationRad: Math.asin(Math.min(1, Math.max(-1, direction[1]))) };
}

/**
 * Builds the calibrated sky. About 25 ms for the default 512 x 256 on the
 * CPU (Node 24, the RTX laptop), most of it the Preetham evaluations.
 */
export function buildCalibratedSky(options: CalibratedSkyOptions): CalibratedSky {
    const width = options.width ?? DEFAULT_SKY_WIDTH;
    const height = options.height ?? DEFAULT_SKY_HEIGHT;
    if (height % 2 !== 0) throw new Error('the sky height must be even, so the horizon falls on a row boundary');

    const sun = sunDirectionAndElevation(options.sunDirection);
    const turbidity = options.turbidity ?? DEFAULT_TURBIDITY;
    const preetham = createPreethamSky(sun.direction, turbidity);

    // Unscaled sky in linear RGB (Preetham's kcd/m^2, which the calibration replaces).
    const radiance = new Float64Array(width * height * 3);
    for (let row = 0; row < height / 2; row++) {
        for (let column = 0; column < width; column++) {
            const direction = texelCentreDirection(column, row, width, height);
            const rgb = xyYToLinearRgb(preethamSkyXyY(preetham, direction));
            radiance.set(rgb, (row * width + column) * 3);
        }
    }

    const clearSky = ineichenPerezClearSky(sun.elevationRad, turbidity);
    const intensity = options.intensity ?? 1;
    const sunHorizontalIlluminance = options.sunIlluminance * Math.sin(sun.elevationRad);
    const targetSkyHorizontalIlluminance = clearSky.skyToSunHorizontalRatio * sunHorizontalIlluminance * intensity;

    const unscaled = measureHorizontalIrradiance(width, height, (index) => radiance[index]);
    const scale = unscaled.luminance > 0 ? targetSkyHorizontalIlluminance / unscaled.luminance : 0;
    const halves = new Uint16Array(width * height * 3);
    for (let i = 0; i < halves.length; i++) halves[i] = floatToHalf(radiance[i] * scale);
    const image: HalfRgbImage = { width, height, halves };

    return {
        image,
        sunDirection: sun.direction,
        sunElevationRad: sun.elevationRad,
        turbidity,
        calibration: {
            model: 'Ineichen and Perez 2002 clear sky (sea level), diffuse-to-direct horizontal ratio',
            clearSky,
            intensity,
            sunHorizontalIlluminance,
            targetSkyHorizontalIlluminance,
        },
        measured: measureHalfImage(image),
    };
}
