/**
 * One sky definition (a manifest's procedural environment) becomes one
 * calibrated, equirectangular HDR image: the shared sky that the viewer,
 * Cycles, the native harness and the platform all read.
 *
 * The sky itself (Preetham, no sun disc, black below the horizon) and its
 * calibration (the Ineichen-Perez share of the sun's light on open ground)
 * are the core's `buildCalibratedSky`
 * (`packages/ddgi-core/src/platform/sky/calibrated-sky.ts`), the same code
 * the platform runs for a `procedural` environment (T1.4). This file adds
 * what is manifest-specific: which light is the sky's sun, and how bright it is.
 */

import type { EnvironmentDescriptor } from '../../../packages/ddgi-core/src/platform/contract/scene.ts';
import {
    buildCalibratedSky,
    DEFAULT_SKY_HEIGHT,
    DEFAULT_SKY_WIDTH,
    DEFAULT_TURBIDITY,
    measureHalfImage,
    measureHorizontalIrradiance,
    type HorizontalIrradiance,
    type SkyCalibration,
} from '../../../packages/ddgi-core/src/platform/sky/calibrated-sky.ts';
import type { HalfRgbImage } from '../../../packages/ddgi-core/src/platform/sky/half-float.ts';
import { rec709Luminance } from '../../../packages/ddgi-core/src/platform/sky/preetham.ts';
import type { ManifestEnvironment, ManifestLight, SceneManifest, Vec3 } from '../manifest.ts';

export { DEFAULT_SKY_HEIGHT, DEFAULT_SKY_WIDTH, DEFAULT_TURBIDITY, measureHalfImage, measureHorizontalIrradiance, type HorizontalIrradiance, type SkyCalibration };

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

export interface BuildSkyOptions {
    width?: number;
    height?: number;
}

/** Builds the calibrated sky for a manifest. Throws if its environment is not procedural. */
export function buildSkyImage(manifest: Pick<SceneManifest, 'environment' | 'lights'>, options: BuildSkyOptions = {}): SkyImage {
    const environment = manifest.environment;
    if (environment.type !== 'procedural') throw new Error(`only a procedural environment becomes a sky file (this one is "${environment.type}")`);
    const sun = resolveSkySun(environment, manifest.lights);
    const sky = buildCalibratedSky({
        sunDirection: environment.sunDirection,
        turbidity: environment.turbidity,
        sunIlluminance: sun.illuminance,
        intensity: environment.intensity,
        width: options.width,
        height: options.height,
    });
    return { image: sky.image, sun, turbidity: sky.turbidity, calibration: sky.calibration, measured: sky.measured };
}

/**
 * The platform environment that gives the same sky as the manifest's sky
 * file: `procedural`, with `intensity` the illuminance of the sun the sky goes
 * with (the platform's meaning of `intensity` for a procedural sky, see
 * `contract/scene.ts`) times the manifest's own `intensity`. The platform then
 * builds exactly the file's pixels.
 */
export function platformSkyEnvironment(manifest: Pick<SceneManifest, 'environment' | 'lights'>): EnvironmentDescriptor {
    const environment = manifest.environment;
    if (environment.type !== 'procedural') throw new Error(`only a procedural environment has a platform sky (this one is "${environment.type}")`);
    const sun = resolveSkySun(environment, manifest.lights);
    return {
        type: 'procedural',
        sunDirection: environment.sunDirection,
        turbidity: environment.turbidity ?? DEFAULT_TURBIDITY,
        intensity: sun.illuminance * (environment.intensity ?? 1),
    };
}
