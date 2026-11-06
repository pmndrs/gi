/**
 * The numbers the GPU needs to make a `procedural` sky itself (T2.6a,
 * docs/design/m2-lights.md §6.2), and when a new environment needs a new
 * bake. The GPU kernels (`projection-wgsl.ts`) evaluate the same Preetham
 * model as `preetham.ts` and are calibrated the same way as
 * `calibrated-sky.ts`:
 *
 * - **the shape**, per view direction: Preetham's Perez distribution for Y, x
 *   and y, from coefficients that depend only on the turbidity, times each
 *   quantity's zenith value over F(0, thetaSun). Those few numbers per sky
 *   are worked out here on the CPU, and the kernel does the per-direction
 *   part;
 * - **the brightness**: the sky is scaled so that open, flat ground gets the
 *   Ineichen-Perez share of the sun's light. The target is a few scalar
 *   formulas (here); what the unscaled sky delivers to that ground is a sum
 *   over the sky, which the GPU makes in the same pass as the SH, over the
 *   same 512 x 256 grid of texel centres the CPU's sky files are built on.
 *
 * So a re-bake costs the CPU a few dozen arithmetic operations and one small
 * buffer write, and never builds an image. The CPU sky (`buildCalibratedSky`)
 * stays: it writes the shared sky files for Cycles and native, and the GPU
 * test holds the GPU's map to it.
 */

import type { EnvironmentDescriptor } from '../contract/scene.ts';
import { DEFAULT_SKY_HEIGHT, DEFAULT_SKY_WIDTH, DEFAULT_TURBIDITY, sunDirectionAndElevation } from './calibrated-sky.ts';
import { ineichenPerezClearSky, MIN_SUN_ELEVATION_DEG } from './clear-sky-ratio.ts';
import { createPreethamSky, MAX_TURBIDITY, MIN_TURBIDITY, type PerezCoefficients } from './preetham.ts';

/**
 * The grid the GPU integrates a procedural sky over (for its SH and its
 * calibration): the CPU sky's own texel centres, so the GPU's brightness
 * comes out as the CPU's to float rounding.
 */
export const PROCEDURAL_GRID_WIDTH = DEFAULT_SKY_WIDTH;
export const PROCEDURAL_GRID_HEIGHT = DEFAULT_SKY_HEIGHT;

/**
 * The smallest turn of the sun that re-bakes the sky (m2-lights.md §6.2).
 * 0.05 degrees is under a tenth of the sun's own width, and about 30 frames
 * of a day animated an hour every 10 s (1.5 degrees a second); smaller
 * turns would cost a bake and change nothing anyone could see.
 */
export const REBAKE_SUN_DEGREES = 0.05;

/** One Perez distribution as the GPU takes it: A to E, and the quantity's zenith value over F(0, thetaSun). */
export interface GpuPerez extends PerezCoefficients {
    readonly zenithScale: number;
}

export interface ProceduralSkyParameters {
    /** Unit vector toward the sun, raised to the clear-sky model's lowest elevation if it was below. */
    readonly sunDirection: readonly [number, number, number];
    /** Clamped to the model's range. */
    readonly turbidity: number;
    readonly sunIlluminance: number;
    readonly intensity: number;
    readonly luminance: GpuPerez;
    readonly chromaticityX: GpuPerez;
    readonly chromaticityY: GpuPerez;
    /** What the sky's luminance-weighted irradiance on open, flat ground is scaled to (`calibrated-sky.ts`). */
    readonly targetHorizontalLuminance: number;
}

type ProceduralEnvironment = Extract<EnvironmentDescriptor, { type: 'procedural' }>;

/**
 * The sky's numbers for a `procedural` environment, with the contract's
 * defaults and clamps: turbidity to 1.7..20, and the sun to at least 5
 * degrees above the horizon (same azimuth), as the CPU sky does.
 */
export function proceduralSkyParameters(environment: ProceduralEnvironment): ProceduralSkyParameters {
    const turbidity = Math.min(MAX_TURBIDITY, Math.max(MIN_TURBIDITY, environment.turbidity ?? DEFAULT_TURBIDITY));
    const sun = sunDirectionAndElevation(liftSunAboveMinimum(environment.sunDirection));
    const sunIlluminance = environment.sunIlluminance ?? 1;
    const intensity = environment.intensity ?? 1;
    const preetham = createPreethamSky(sun.direction, turbidity);
    const clearSky = ineichenPerezClearSky(sun.elevationRad, turbidity);
    return {
        sunDirection: sun.direction,
        turbidity,
        sunIlluminance,
        intensity,
        luminance: { ...preetham.luminance, zenithScale: preetham.zenith.luminance / preetham.zenithPerez.luminance },
        chromaticityX: { ...preetham.chromaticityX, zenithScale: preetham.zenith.x / preetham.zenithPerez.x },
        chromaticityY: { ...preetham.chromaticityY, zenithScale: preetham.zenith.y / preetham.zenithPerez.y },
        // The same formula as buildCalibratedSky's targetSkyHorizontalIlluminance.
        targetHorizontalLuminance: clearSky.skyToSunHorizontalRatio * sunIlluminance * Math.sin(sun.elevationRad) * intensity,
    };
}

/**
 * Whether a procedural sky with `next`'s numbers needs a new bake after one
 * made with `baked`'s (m2-lights.md §6.2): the sun turned more than
 * {@link REBAKE_SUN_DEGREES}, or the turbidity, `sunIlluminance` or
 * `intensity` changed at all. Compared with the last *bake*, not the last
 * call, so a sun creeping 0.01 degrees a frame still re-bakes every few
 * frames.
 */
export function proceduralSkyNeedsRebake(baked: ProceduralSkyParameters, next: ProceduralSkyParameters): boolean {
    if (baked.turbidity !== next.turbidity || baked.sunIlluminance !== next.sunIlluminance || baked.intensity !== next.intensity) return true;
    const cosine = baked.sunDirection[0] * next.sunDirection[0] + baked.sunDirection[1] * next.sunDirection[1] + baked.sunDirection[2] * next.sunDirection[2];
    return cosine < Math.cos((REBAKE_SUN_DEGREES * Math.PI) / 180);
}

/** The sun direction, raised to the clear-sky model's lowest elevation if it is below it (same azimuth). */
export function liftSunAboveMinimum(sunDirection: readonly [number, number, number]): [number, number, number] {
    const length = Math.hypot(sunDirection[0], sunDirection[1], sunDirection[2]);
    // A thousandth of a degree above the model's limit, so rounding can't put it just under.
    const minimumElevation = ((MIN_SUN_ELEVATION_DEG + 0.001) * Math.PI) / 180;
    if (sunDirection[1] / length >= Math.sin(minimumElevation)) return [sunDirection[0], sunDirection[1], sunDirection[2]];
    const horizontal = Math.hypot(sunDirection[0], sunDirection[2]);
    // Straight down has no azimuth: put the sun towards +X.
    const [x, z] = horizontal > 0 ? [sunDirection[0] / horizontal, sunDirection[2] / horizontal] : [1, 0];
    return [x * Math.cos(minimumElevation), Math.sin(minimumElevation), z * Math.cos(minimumElevation)];
}
