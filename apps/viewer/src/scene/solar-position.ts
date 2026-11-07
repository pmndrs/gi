/**
 * Where the sun is for a time of day and a day of the year: the simple
 * textbook formula (declination from the day, hour angle from the time),
 * good to about a degree, which is all a viewer's time-of-day slider needs
 * (docs/design/m2-lights.md §7: "a simple solar-position formula").
 *
 * The scene frame: +y is up. A scene has no north of its own, so the viewer
 * turns the compass to match the manifest's sun (`fitSunToManifest`): at the
 * time it opens with, the sun is exactly where the manifest put it, and the
 * slider moves it from there along a real sun path.
 *
 * Below MIN_SKY_ELEVATION_DEG the calibrated sky model stops (the platform
 * raises its sun to it), so `timeOfDaySun` keeps the sun's direction at that
 * elevation and fades the sun and sky towards night instead
 * (`daylight`), reaching black at NIGHT_ELEVATION_DEG.
 *
 * No engine imports, so it is unit-tested.
 */

type Vec3 = [number, number, number];

const DEGREES = Math.PI / 180;

/** The latitude the viewer's sun moves at, degrees north: Dubrovnik's, whose Sponza palace Sponza is named after. */
export const DEFAULT_LATITUDE_DEG = 42.6;
/** The longest day of the year (northern summer), where the viewer starts. */
export const SUMMER_SOLSTICE_DAY = 172;
/** The calibrated sky's lowest sun (the platform's MIN_SUN_ELEVATION_DEG, clear-sky-ratio.ts). */
export const MIN_SKY_ELEVATION_DEG = 5;
/** Where the viewer's fade to night ends: the sun and the sky are off from here down. */
export const NIGHT_ELEVATION_DEG = -4;

/** The sun's place in the sky, degrees: elevation above the horizon, azimuth clockwise from north (90 is east). */
export interface SolarAngles {
    elevationDeg: number;
    azimuthDeg: number;
}

/**
 * The sun's elevation and azimuth.
 * @param hour local solar time, 0 to 24 (12 is noon)
 * @param dayOfYear 1 to 365
 */
export function solarAngles(hour: number, dayOfYear: number, latitudeDeg = DEFAULT_LATITUDE_DEG): SolarAngles {
    // Declination: the sun's angle above the equator, +23.44° at the June solstice.
    const declination = 23.44 * DEGREES * Math.sin((2 * Math.PI * (284 + dayOfYear)) / 365);
    // Hour angle: 15° an hour from noon, negative in the morning.
    const hourAngle = (hour - 12) * 15 * DEGREES;
    const latitude = latitudeDeg * DEGREES;
    const sinElevation = Math.sin(latitude) * Math.sin(declination) + Math.cos(latitude) * Math.cos(declination) * Math.cos(hourAngle);
    const elevation = Math.asin(Math.min(1, Math.max(-1, sinElevation)));
    // Azimuth from south, positive towards west, then turned to "clockwise from north".
    const fromSouth = Math.atan2(Math.sin(hourAngle), Math.cos(hourAngle) * Math.sin(latitude) - Math.tan(declination) * Math.cos(latitude));
    const azimuth = fromSouth + Math.PI;
    return { elevationDeg: elevation / DEGREES, azimuthDeg: ((azimuth / DEGREES) % 360 + 360) % 360 };
}

/**
 * The unit vector towards the sun in the scene frame (+y up), with the
 * scene's north `northDeg` degrees clockwise (seen from above) from −z.
 */
export function sunVector(angles: SolarAngles, northDeg: number): Vec3 {
    const elevation = angles.elevationDeg * DEGREES;
    // Clockwise from above is from −z towards +x.
    const heading = (angles.azimuthDeg + northDeg) * DEGREES;
    const horizontal = Math.cos(elevation);
    return [horizontal * Math.sin(heading), Math.sin(elevation), -horizontal * Math.cos(heading)];
}

/** Elevation (degrees) and heading (degrees clockwise from −z, seen from above) of a direction towards the sun. */
export function vectorAngles(towardSun: readonly [number, number, number]): { elevationDeg: number; headingDeg: number } {
    const length = Math.hypot(towardSun[0], towardSun[1], towardSun[2]);
    const elevationDeg = Math.asin(Math.min(1, Math.max(-1, towardSun[1] / length))) / DEGREES;
    const headingDeg = ((Math.atan2(towardSun[0], -towardSun[2]) / DEGREES) % 360 + 360) % 360;
    return { elevationDeg, headingDeg };
}

/** How the viewer's time of day starts on a scene: its time, its day, and the compass turn that puts the sun where the manifest has it. */
export interface TimeOfDayFit {
    hour: number;
    dayOfYear: number;
    northDeg: number;
}

/**
 * The afternoon time on the longest day whose sun is as high as the
 * manifest's, and the compass turn that then puts it at the manifest's
 * heading. A manifest sun higher than that day's noon starts at noon (the
 * highest it gets; Sponza's 70.6° against 70.8° at 42.6° north).
 * @param towardSun the manifest's direction towards the sun (minus its light's direction)
 */
export function fitSunToManifest(towardSun: readonly [number, number, number], latitudeDeg = DEFAULT_LATITUDE_DEG): TimeOfDayFit {
    const target = vectorAngles(towardSun);
    const day = SUMMER_SOLSTICE_DAY;
    // Elevation falls monotonically from noon to midnight: bisect the afternoon.
    let hour = 12;
    if (solarAngles(12, day, latitudeDeg).elevationDeg > target.elevationDeg) {
        let low = 12;
        let high = 24;
        for (let step = 0; step < 40; step++) {
            const middle = (low + high) / 2;
            if (solarAngles(middle, day, latitudeDeg).elevationDeg > target.elevationDeg) low = middle;
            else high = middle;
        }
        hour = (low + high) / 2;
    }
    const angles = solarAngles(hour, day, latitudeDeg);
    const northDeg = (((target.headingDeg - angles.azimuthDeg) % 360) + 360) % 360;
    return { hour, dayOfYear: day, northDeg };
}

/**
 * How much of the day's light is up: 1 with the sun at or above
 * MIN_SKY_ELEVATION_DEG, 0 at NIGHT_ELEVATION_DEG and below, a smooth fade
 * between. The calibrated sky can't follow the sun below 5° (the model's
 * limit), so this stands in for dusk.
 */
export function daylight(elevationDeg: number): number {
    const t = Math.min(1, Math.max(0, (elevationDeg - NIGHT_ELEVATION_DEG) / (MIN_SKY_ELEVATION_DEG - NIGHT_ELEVATION_DEG)));
    return t * t * (3 - 2 * t);
}

/** What the time of day sets: where the sun is (never below the sky model's limit) and how much of its light is up. */
export interface TimeOfDaySun {
    /** Unit vector towards the sun, raised to MIN_SKY_ELEVATION_DEG when lower (same heading). */
    towardSun: Vec3;
    /** The sun's true elevation, degrees (may be below the horizon). */
    elevationDeg: number;
    /** 0 to 1: the sun light's and the sky's share of their full brightness (`daylight`). */
    daylight: number;
}

export function timeOfDaySun(hour: number, dayOfYear: number, northDeg: number, latitudeDeg = DEFAULT_LATITUDE_DEG): TimeOfDaySun {
    const angles = solarAngles(hour, dayOfYear, latitudeDeg);
    // A thousandth of a degree above the model's limit, as the platform lifts it, so rounding can't put it under.
    const lifted = { ...angles, elevationDeg: Math.max(angles.elevationDeg, MIN_SKY_ELEVATION_DEG + 0.001) };
    return { towardSun: sunVector(lifted, northDeg), elevationDeg: angles.elevationDeg, daylight: daylight(angles.elevationDeg) };
}

/** The hours (0 to 24) the sun crosses `elevationDeg` on a day, morning and evening; null when it never does. */
export function hoursAtElevation(elevationDeg: number, dayOfYear: number, latitudeDeg = DEFAULT_LATITUDE_DEG): { morning: number; evening: number } | null {
    const noon = solarAngles(12, dayOfYear, latitudeDeg).elevationDeg;
    const midnight = solarAngles(0, dayOfYear, latitudeDeg).elevationDeg;
    if (noon < elevationDeg || midnight > elevationDeg) return null;
    let low = 12;
    let high = 24;
    for (let step = 0; step < 40; step++) {
        const middle = (low + high) / 2;
        if (solarAngles(middle, dayOfYear, latitudeDeg).elevationDeg > elevationDeg) low = middle;
        else high = middle;
    }
    const evening = (low + high) / 2;
    // The path is symmetric about solar noon.
    return { morning: 24 - evening, evening };
}
