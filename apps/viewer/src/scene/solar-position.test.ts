import { describe, expect, it } from 'vitest';
import { daylight, DEFAULT_LATITUDE_DEG, fitSunToManifest, hoursAtElevation, MIN_SKY_ELEVATION_DEG, solarAngles, sunVector, timeOfDaySun, vectorAngles } from './solar-position';

describe('the sun for a time of day', () => {
    it('stands highest at noon, in the south, at 90 − latitude + declination', () => {
        const noon = solarAngles(12, 172);
        expect(noon.elevationDeg).toBeCloseTo(90 - DEFAULT_LATITUDE_DEG + 23.44, 1);
        expect(noon.azimuthDeg).toBeCloseTo(180, 3);
        // Equinox: 90 − latitude.
        expect(solarAngles(12, 81).elevationDeg).toBeCloseTo(90 - DEFAULT_LATITUDE_DEG, 0);
    });

    it('rises in the east and sets in the west, symmetric about noon', () => {
        const morning = solarAngles(9, 172);
        const evening = solarAngles(15, 172);
        expect(morning.azimuthDeg).toBeGreaterThan(45);
        expect(morning.azimuthDeg).toBeLessThan(180);
        expect(evening.azimuthDeg).toBeGreaterThan(180);
        expect(morning.elevationDeg).toBeCloseTo(evening.elevationDeg, 6);
        expect(morning.azimuthDeg + evening.azimuthDeg).toBeCloseTo(360, 6);
        expect(solarAngles(0, 172).elevationDeg).toBeLessThan(0);
    });

    it('turns angles into a direction and back', () => {
        for (const [elevationDeg, azimuthDeg, northDeg] of [
            [30, 90, 0],
            [70, 200, 45],
            [10, 300, 270],
        ]) {
            const vector = sunVector({ elevationDeg, azimuthDeg }, northDeg);
            expect(Math.hypot(...vector)).toBeCloseTo(1, 9);
            const back = vectorAngles(vector);
            expect(back.elevationDeg).toBeCloseTo(elevationDeg, 9);
            expect(back.headingDeg).toBeCloseTo((azimuthDeg + northDeg) % 360, 9);
        }
        // North is −z, east +x.
        const east = sunVector({ elevationDeg: 0, azimuthDeg: 90 }, 0);
        expect(east[0]).toBeCloseTo(1, 9);
        expect(east[2]).toBeCloseTo(0, 9);
    });

    it("starts where the manifest's sun is", () => {
        // A sun 40° up in the south-west of the scene's frame.
        const toward: [number, number, number] = [Math.cos((40 * Math.PI) / 180) * 0.6, Math.sin((40 * Math.PI) / 180), Math.cos((40 * Math.PI) / 180) * 0.8];
        const length = Math.hypot(...toward);
        const fit = fitSunToManifest(toward);
        const sun = timeOfDaySun(fit.hour, fit.dayOfYear, fit.northDeg);
        expect(fit.hour).toBeGreaterThan(12);
        for (let axis = 0; axis < 3; axis++) expect(sun.towardSun[axis]).toBeCloseTo(toward[axis] / length, 6);
        expect(sun.daylight).toBe(1);
        // Sponza's sun, 70.6° up, is just under the solstice noon's 70.8° here: it starts shortly after noon.
        const sponza = fitSunToManifest([-0.3, 0.9, -0.1]);
        const sponzaSun = timeOfDaySun(sponza.hour, sponza.dayOfYear, sponza.northDeg);
        const length2 = Math.hypot(0.3, 0.9, 0.1);
        expect(sponzaSun.towardSun[1]).toBeCloseTo(0.9 / length2, 4);
    });

    it('keeps the sun at the sky model limit and fades it out towards night', () => {
        expect(daylight(MIN_SKY_ELEVATION_DEG)).toBe(1);
        expect(daylight(30)).toBe(1);
        expect(daylight(-4)).toBe(0);
        expect(daylight(-20)).toBe(0);
        expect(daylight(0.5)).toBeGreaterThan(0);
        expect(daylight(0.5)).toBeLessThan(1);
        const dusk = timeOfDaySun(21.5, 172, 0);
        expect(dusk.elevationDeg).toBeLessThan(MIN_SKY_ELEVATION_DEG);
        expect(vectorAngles(dusk.towardSun).elevationDeg).toBeGreaterThanOrEqual(MIN_SKY_ELEVATION_DEG);
        const crossing = hoursAtElevation(0, 172);
        expect(crossing).not.toBeNull();
        expect(solarAngles(crossing!.evening, 172).elevationDeg).toBeCloseTo(0, 4);
        expect(solarAngles(crossing!.morning, 172).elevationDeg).toBeCloseTo(0, 4);
    });
});
