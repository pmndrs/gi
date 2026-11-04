/**
 * How bright the sky should be next to the sun: the ratio of diffuse (sky)
 * irradiance to direct (sun) irradiance on a horizontal surface under a clear
 * sky, for a given sun elevation and turbidity.
 *
 * Source: P. Ineichen and R. Perez, "A new airmass independent formulation for
 * the Linke turbidity coefficient", Solar Energy 73(3), 151-157, 2002. This is
 * the clear-sky model behind measured-data tools like pvlib (`ineichen`) and
 * the SoDa service, fitted to ground stations. Implemented from the paper's
 * equations, at sea level, the way pvlib's `ineichen` applies them by default
 * (checked against its documentation, not copied from its code):
 *
 *   air mass       AM  = 1 / (cos z + 0.50572 (96.07995 - z_deg)^-1.6364)   (Kasten and Young 1989)
 *   global (GHI)   G   = cg1 I0 cos z exp(-cg2 AM (fh1 + fh2 (TL - 1)))
 *   beam (DNI)     B   = b I0 exp(-0.09 AM (TL - 1))
 *   beam limit     B  <= G (1 - (0.1 - 0.2 exp(-TL)) / (0.1 + 0.882 / fh1)) / cos z
 *   diffuse (DHI)  D   = G - B cos z
 *
 * with, at altitude 0: cg1 = 0.868, cg2 = 0.0387, fh1 = fh2 = 1, b = 0.664 + 0.163 / fh1.
 * I0 (the solar constant) cancels in the ratio we use.
 *
 * Left out, as in pvlib's default: the "Perez enhancement" factor
 * exp(0.01 AM^1.8) on G from Perez et al. 2002 (Solar Energy 73, 307-317). It
 * would raise G by about 1% at a high sun and so the sky share by about 9%.
 *
 * Two assumptions, stated so they can be revisited:
 *  - The manifest's turbidity is Preetham's T, used here as Linke turbidity TL.
 *    Both are "how many clean, dry atmospheres would scatter this much"; Linke's
 *    also counts water vapour. For the clear skies we use (T = 2 to 4) the
 *    difference is within the spread of real clear days.
 *  - The ratio is broadband radiometric (W/m^2). Daylight illuminance ratios
 *    are a little higher, because sky light is bluer and the eye's luminous
 *    efficacy for it is higher. Renderers here work in RGB with "lux" as a
 *    label, so the radiometric ratio is the one that keeps the scene's
 *    balance; the difference is about 10-20% of the sky share, not a factor.
 */

export interface ClearSkyIrradiance {
    /** Sun elevation above the horizon, radians. */
    sunElevation: number;
    turbidity: number;
    /** Relative optical air mass. */
    airMass: number;
    /** Global horizontal irradiance, as a fraction of the solar constant I0. */
    globalHorizontal: number;
    /** Direct normal irradiance (the sun's beam), fraction of I0. */
    directNormal: number;
    /** Direct horizontal irradiance: directNormal * sin(elevation). */
    directHorizontal: number;
    /** Diffuse horizontal irradiance: the sky's light on a flat open ground. */
    diffuseHorizontal: number;
    /** diffuseHorizontal / directHorizontal: the sky's share next to the sun's, on the ground. */
    skyToSunHorizontalRatio: number;
    /** diffuseHorizontal / globalHorizontal: the measured "diffuse fraction". */
    diffuseFraction: number;
}

/** Below this elevation the air-mass formula and the clear-sky fit are unreliable. */
export const MIN_SUN_ELEVATION_DEG = 5;

const RADIANS_TO_DEGREES = 180 / Math.PI;

/** Kasten and Young (1989) relative optical air mass; zenith angle in radians. */
export function kastenYoungAirMass(sunZenith: number): number {
    const zenithDegrees = sunZenith * RADIANS_TO_DEGREES;
    return 1 / (Math.cos(sunZenith) + 0.50572 * (96.07995 - zenithDegrees) ** -1.6364);
}

/**
 * @param sunElevation radians above the horizon, at least {@link MIN_SUN_ELEVATION_DEG}
 * @param turbidity Linke turbidity (the manifest's Preetham turbidity, see above)
 */
export function ineichenPerezClearSky(sunElevation: number, turbidity: number): ClearSkyIrradiance {
    if (!(sunElevation * RADIANS_TO_DEGREES >= MIN_SUN_ELEVATION_DEG) || sunElevation > Math.PI / 2) {
        throw new Error(
            `sun elevation ${(sunElevation * RADIANS_TO_DEGREES).toFixed(1)} degrees is outside the clear-sky model's range (${MIN_SUN_ELEVATION_DEG} to 90)`,
        );
    }
    if (!(turbidity >= 1)) throw new Error(`Linke turbidity ${turbidity} must be at least 1`);

    const sunZenith = Math.PI / 2 - sunElevation;
    const cosZenith = Math.cos(sunZenith);
    const airMass = kastenYoungAirMass(sunZenith);

    // Sea level (altitude 0 m): the altitude terms reduce to these constants.
    const fh1 = 1;
    const fh2 = 1;
    const cg1 = 0.868;
    const cg2 = 0.0387;
    const beamCoefficient = 0.664 + 0.163 / fh1;

    const globalHorizontal = cg1 * cosZenith * Math.exp(-cg2 * airMass * (fh1 + fh2 * (turbidity - 1)));
    const beamFromTurbidity = beamCoefficient * Math.exp(-0.09 * airMass * (turbidity - 1));
    // The paper's empirical limit keeps the beam from exceeding what the global
    // irradiance allows on very clear skies (it bites at TL below about 2.5).
    const beamLimit = (globalHorizontal * (1 - (0.1 - 0.2 * Math.exp(-turbidity)) / (0.1 + 0.882 / fh1))) / cosZenith;
    const directNormal = Math.max(0, Math.min(beamFromTurbidity, beamLimit));
    const directHorizontal = directNormal * cosZenith;
    const diffuseHorizontal = globalHorizontal - directHorizontal;

    return {
        sunElevation,
        turbidity,
        airMass,
        globalHorizontal,
        directNormal,
        directHorizontal,
        diffuseHorizontal,
        skyToSunHorizontalRatio: diffuseHorizontal / directHorizontal,
        diffuseFraction: diffuseHorizontal / globalHorizontal,
    };
}
