/**
 * One sky definition (a manifest's procedural environment) becomes one
 * calibrated, equirectangular HDR image: the shared sky that the viewer,
 * Cycles, the native harness and the platform all read.
 *
 * What is in the image:
 *  - Above the horizon: the Preetham sky (`preetham.ts`) for the manifest's
 *    sun direction and turbidity, in linear Rec.709 RGB.
 *  - No sun disc. The sun is the manifest's directional light, which every
 *    renderer already has; drawing it into the sky too would count it twice.
 *  - Below the horizon: black. The ground is the scene's geometry, so each
 *    renderer gets its ground light from the same place (the scene), and an
 *    open scene's edge doesn't get light from below that no renderer can agree on.
 *
 * Brightness (the calibration): the image is scaled so that a flat, open,
 * upward-facing ground gets sky irradiance equal to a realistic share of the
 * sun's irradiance on that same ground:
 *
 *   E_sky,horizontal = ratio(elevation, turbidity) * E_sun * sin(elevation) * environment.intensity
 *
 * where `ratio` is the clear-sky diffuse-to-direct horizontal ratio of the
 * Ineichen-Perez model (`clear-sky-ratio.ts`) and E_sun is the sun's
 * luminance-weighted illuminance at normal incidence (manifest `intensity`
 * times the Rec.709 luminance of its `color`). Luminance is the Rec.709 Y of
 * the stored RGB, the same weights renderers use. The irradiance is measured
 * on the half floats actually written, so what a reader integrates is what
 * the calibration promised (to half-float rounding).
 */

import type { ManifestEnvironment, ManifestLight, SceneManifest, Vec3 } from '../manifest.ts';
import { ineichenPerezClearSky, type ClearSkyIrradiance } from './clear-sky-ratio.ts';
import { rowHorizontalIrradianceWeight, texelCentreDirection } from './equirect.ts';
import { floatToHalf, halfToFloat, type HalfRgbImage } from './exr-half.ts';
import { createPreethamSky, preethamSkyXyY, rec709Luminance, xyYToLinearRgb } from './preetham.ts';

/** 512 x 256 texels, 0.7 degrees each: the sky has no sun disc and no sharp detail, so more would only make the file bigger. */
export const DEFAULT_SKY_WIDTH = 512;
export const DEFAULT_SKY_HEIGHT = 256;

/** Preetham turbidity when the manifest leaves it out (the viewer's old `SkyMesh` default). */
export const DEFAULT_TURBIDITY = 2;

/** A directional light counts as the sky's sun when it travels within this angle of -sunDirection. */
const SUN_MATCH_DEGREES = 1;

export type ProceduralEnvironment = Extract<ManifestEnvironment, { type: 'procedural' }>;

export interface SkySun {
    /** Unit vector towards the sun. */
    direction: [number, number, number];
    elevationRad: number;
    /** Luminance-weighted illuminance at normal incidence. */
    illuminance: number;
    /** Where the illuminance came from, in words (for the sidecar and messages). */
    source: string;
}

export interface SkyCalibration {
    model: string;
    clearSky: ClearSkyIrradiance;
    /** environment.intensity: a plain multiplier on top of the calibrated sky (1 = as calibrated). */
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

export interface SkyImage {
    image: HalfRgbImage;
    sun: SkySun;
    turbidity: number;
    calibration: SkyCalibration;
    /** Measured on the stored half floats. */
    measured: HorizontalIrradiance;
}

function normalize(vector: Vec3): [number, number, number] {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    return [vector[0] / length, vector[1] / length, vector[2] / length];
}

/**
 * Finds the sun the sky belongs to: the manifest's directional light that
 * travels opposite `sunDirection`, or, for a sky-only scene with no such light,
 * `environment.sunIlluminance`. Exactly one of the two must be there, so the
 * calibration never silently picks one.
 */
export function resolveSkySun(environment: ProceduralEnvironment, lights: readonly ManifestLight[]): SkySun {
    const direction = normalize(environment.sunDirection);
    const elevationRad = Math.asin(Math.min(1, Math.max(-1, direction[1])));
    const cosLimit = Math.cos((SUN_MATCH_DEGREES * Math.PI) / 180);
    const matching = lights.flatMap((light, index) => {
        if (light.type !== 'directional') return [];
        const travel = normalize(light.direction);
        const cosAngle = -(travel[0] * direction[0] + travel[1] * direction[1] + travel[2] * direction[2]);
        return cosAngle >= cosLimit ? [{ light, index }] : [];
    });

    if (matching.length > 1) throw new Error(`more than one directional light points along environment.sunDirection (lights ${matching.map((m) => m.index).join(', ')})`);
    const given = environment.sunIlluminance;
    if (matching.length === 1 && given !== undefined) {
        throw new Error(`environment.sunIlluminance is only for scenes without a sun light; lights[${matching[0].index}] is the sun here`);
    }
    if (matching.length === 1) {
        const { light, index } = matching[0];
        const illuminance = light.intensity * rec709Luminance(light.color[0], light.color[1], light.color[2]);
        return { direction, elevationRad, illuminance, source: `lights[${index}]${light.label ? ` "${light.label}"` : ''}: intensity ${light.intensity} x luminance of its colour` };
    }
    if (given === undefined) {
        throw new Error(
            'no directional light travels along -environment.sunDirection (within 1 degree), and environment.sunIlluminance is not set; the sky needs one of them to know how bright the sun is',
        );
    }
    return { direction, elevationRad, illuminance: given, source: 'environment.sunIlluminance (no sun light in this scene)' };
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

export interface BuildSkyOptions {
    width?: number;
    height?: number;
}

/** Builds the calibrated sky for a manifest. Throws if its environment is not procedural. */
export function buildSkyImage(manifest: Pick<SceneManifest, 'environment' | 'lights'>, options: BuildSkyOptions = {}): SkyImage {
    const environment = manifest.environment;
    if (environment.type !== 'procedural') throw new Error(`only a procedural environment becomes a sky file (this one is "${environment.type}")`);
    const width = options.width ?? DEFAULT_SKY_WIDTH;
    const height = options.height ?? DEFAULT_SKY_HEIGHT;
    if (height % 2 !== 0) throw new Error('the sky height must be even, so the horizon falls on a row boundary');

    const sun = resolveSkySun(environment, manifest.lights);
    const turbidity = environment.turbidity ?? DEFAULT_TURBIDITY;
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
    const intensity = environment.intensity ?? 1;
    const sunHorizontalIlluminance = sun.illuminance * Math.sin(sun.elevationRad);
    const targetSkyHorizontalIlluminance = clearSky.skyToSunHorizontalRatio * sunHorizontalIlluminance * intensity;

    const unscaled = measureHorizontalIrradiance(width, height, (index) => radiance[index]);
    const scale = unscaled.luminance > 0 ? targetSkyHorizontalIlluminance / unscaled.luminance : 0;
    const halves = new Uint16Array(width * height * 3);
    for (let i = 0; i < halves.length; i++) halves[i] = floatToHalf(radiance[i] * scale);
    const image: HalfRgbImage = { width, height, halves };

    return {
        image,
        sun,
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
