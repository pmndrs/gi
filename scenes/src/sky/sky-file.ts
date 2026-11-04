/**
 * Where the shared sky files live and how a host loads one, with no engine.
 *
 * A scene whose manifest environment is `procedural` has two committed files
 * in `scenes/sky/`, written by `node tools/sky/generate.ts`:
 *  - `<scene id>.exr`: the sky (equirectangular, half-float B/G/R, ZIP), see
 *    `equirect.ts` for the layout and `sky-image.ts` for what is in it;
 *  - `<scene id>.json`: the sidecar ({@link SkySidecar}): how it was made,
 *    the calibration numbers, and the flat-colour equivalent for renderers
 *    that can only take one sky colour (the native harness).
 *
 * The platform takes the sky as `setEnvironment({ type: 'texture', layout:
 * 'equirect', texture })`. A host makes that texture with
 * {@link loadSkyFile} and `createSkyTexture` (`sky-texture.ts`).
 */

import { decodeHalfExr, type HalfRgbImage, type Inflate } from './exr-half.ts';

/** Repo-relative folder of the sky files. */
export const SKY_DIRECTORY = 'scenes/sky';

export function skyFileName(sceneId: string): string {
    return `${sceneId}.exr`;
}

export function skySidecarName(sceneId: string): string {
    return `${sceneId}.json`;
}

/** The sidecar written next to each sky file. Numbers are in the manifest's light units (lux, and lux / sr for radiance). */
export interface SkySidecar {
    scene: string;
    file: string;
    generator: string;
    skyModel: string;
    width: number;
    height: number;
    layout: string;
    sunDirection: [number, number, number];
    sunElevationDeg: number;
    turbidity: number;
    /** Luminance-weighted sun illuminance at normal incidence, and where it came from. */
    sunIlluminance: number;
    sunSource: string;
    calibration: {
        model: string;
        /** Ineichen-Perez diffuse / direct on a horizontal surface, for this elevation and turbidity. */
        skyToSunHorizontalRatio: number;
        /** The same sky as a share of all daylight on the ground (diffuse / global). */
        diffuseFraction: number;
        airMass: number;
        intensity: number;
        sunHorizontalIlluminance: number;
        targetSkyHorizontalIlluminance: number;
    };
    /** Integrated from the stored half floats. */
    measured: {
        skyHorizontalIrradianceRgb: [number, number, number];
        skyHorizontalIlluminance: number;
        /** measured / target - 1: what half-float rounding left over. */
        calibrationError: number;
    };
    /**
     * A uniform upper-hemisphere radiance with the same horizontal irradiance,
     * per channel (E / pi). Only for renderers that take a single sky colour.
     */
    flatEquivalentRadianceRgb: [number, number, number];
    /** SHA-256 of the decoded half floats (R, G, B interleaved, little-endian), so a check doesn't depend on zlib's output bytes. */
    pixelSha256: string;
}

/** Decodes sky file bytes. `inflate` defaults to the platform's `DecompressionStream`. */
export function decodeSkyFile(bytes: Uint8Array, inflate?: Inflate): Promise<HalfRgbImage> {
    return decodeHalfExr(bytes, inflate);
}

/** Fetches and decodes a sky file. Works in browsers and in Node 18+. */
export async function loadSkyFile(url: string | URL): Promise<HalfRgbImage> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not load the sky file ${String(url)}: HTTP ${response.status}`);
    return decodeSkyFile(new Uint8Array(await response.arrayBuffer()));
}
