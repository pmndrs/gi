/// <reference types="node" />
import { readFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { deflateSync, inflateSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { readExr } from '../../../tools/compare/exr.ts';
import { compareWithCommitted, loadSkyManifests, skyDirectory } from '../../../tools/sky/generate.ts';
import type { ManifestLight, SceneManifest } from '../manifest.ts';
import { OPEN_GROUND_ALBEDO } from '../procedural/open-ground.ts';
import { ineichenPerezClearSky, kastenYoungAirMass } from './clear-sky-ratio.ts';
import { directionToEquirectUv, equirectUvToDirection, rowHorizontalIrradianceWeight, rowSolidAngle } from './equirect.ts';
import { decodeHalfExr, encodeHalfExr, floatToHalf, halfImageToFloats, halfToFloat, type HalfRgbImage } from './exr-half.ts';
import { createPreethamSky, perez, preethamSkyXyY } from './preetham.ts';
import { decodeSkyFile, skyFileName, type SkySidecar } from './sky-file.ts';
import { buildSkyImage, measureHalfImage, resolveSkySun, type ProceduralEnvironment } from './sky-image.ts';
import { skyImageToRgbaHalves } from './sky-texture.ts';

const repoRoot = resolve(__dirname, '..', '..', '..');
const DEG = Math.PI / 180;

const sun: ManifestLight = { type: 'directional', label: 'sun', color: [1, 0.95, 0.85], intensity: 3, direction: [0.3, -0.9, 0.1] };
const sponzaLikeSky: ProceduralEnvironment = { type: 'procedural', sunDirection: [-0.3, 0.9, -0.1], turbidity: 3, intensity: 1 };
const smallSky = (environment: ProceduralEnvironment = sponzaLikeSky, lights: ManifestLight[] = [sun]) =>
    buildSkyImage({ environment, lights }, { width: 128, height: 64 });

function texel(image: HalfRgbImage, column: number, row: number): [number, number, number] {
    const base = (row * image.width + column) * 3;
    return [halfToFloat(image.halves[base]), halfToFloat(image.halves[base + 1]), halfToFloat(image.halves[base + 2])];
}

describe('half floats', () => {
    it('round-trips exactly representable values and rounds to nearest even', () => {
        for (const value of [0, 1, -2, 0.5, 65504, 2 ** -14, 2 ** -24, 1 + 2 ** -10]) {
            expect(halfToFloat(floatToHalf(value))).toBe(value);
        }
        // Halfway between 1 and 1 + 2^-10 rounds to the even mantissa (1); just above it rounds up.
        expect(halfToFloat(floatToHalf(1 + 2 ** -11))).toBe(1);
        expect(halfToFloat(floatToHalf(1 + 2 ** -11 + 2 ** -20))).toBe(1 + 2 ** -10);
        expect(halfToFloat(floatToHalf(1 + 3 * 2 ** -11))).toBe(1 + 2 * 2 ** -10);
    });

    it('keeps every normal value within half a step (2^-11 relative) and refuses overflow and NaN', () => {
        // From 1.2e-4 (above the smallest normal half, 6.1e-5) to 22000.
        for (let i = 0; i < 2000; i++) {
            const value = Math.exp((i / 2000) * 19 - 9);
            const relative = Math.abs(halfToFloat(floatToHalf(value)) - value) / value;
            expect(relative).toBeLessThanOrEqual(2 ** -11 + 1e-12);
        }
        expect(() => floatToHalf(70000)).toThrow(/largest half/);
        expect(() => floatToHalf(NaN)).toThrow(/NaN/);
    });
});

describe('the sky file format (half-float OpenEXR)', () => {
    const image = smallSky().image;

    it('round-trips through ZIP and uncompressed files, bit for bit', async () => {
        for (const deflate of [undefined, (bytes: Uint8Array) => deflateSync(bytes)]) {
            const file = encodeHalfExr(image, deflate);
            const decoded = await decodeHalfExr(file, (bytes) => inflateSync(bytes));
            expect(decoded.width).toBe(image.width);
            expect(decoded.height).toBe(image.height);
            expect(Buffer.from(decoded.halves.buffer).equals(Buffer.from(image.halves.buffer))).toBe(true);
        }
    });

    it('decodes with the browser-style DecompressionStream inflater too', async () => {
        const decoded = await decodeSkyFile(encodeHalfExr(image, (bytes) => deflateSync(bytes)));
        expect(Buffer.from(decoded.halves.buffer).equals(Buffer.from(image.halves.buffer))).toBe(true);
    });

    it('is read the same by an independent decoder (tools/compare)', () => {
        const fromCompare = readExr(encodeHalfExr(image, (bytes) => deflateSync(bytes)));
        expect(Array.from(fromCompare.rgb)).toEqual(Array.from(halfImageToFloats(image)));
    });

    it('expands to RGBA halves with alpha 1 for the GPU texture', () => {
        const rgba = skyImageToRgbaHalves(image);
        expect(rgba.length).toBe(image.width * image.height * 4);
        expect(halfToFloat(rgba[3])).toBe(1);
        expect(rgba[4 * 10 + 2]).toBe(image.halves[3 * 10 + 2]);
    });
});

describe('the equirect layout', () => {
    it('puts directions where three.js equirectUV does (u), with the zenith in the first row (v)', () => {
        const at = (direction: [number, number, number]) => directionToEquirectUv(direction);
        expect(at([1, 0, 0])).toEqual({ u: 0.5, v: 0.5 });
        expect(at([0, 0, 1]).u).toBeCloseTo(0.75, 12);
        expect(at([0, 0, -1]).u).toBeCloseTo(0.25, 12);
        expect(at([0, 1, 0]).v).toBe(0);
        expect(at([0, -1, 0]).v).toBeCloseTo(1, 12);
    });

    it('inverts', () => {
        for (const direction of [[0.3, 0.5, -0.8], [-0.9, -0.1, 0.2], [0.01, 0.99, 0.1]] as [number, number, number][]) {
            const length = Math.hypot(...direction);
            const back = equirectUvToDirection(directionToEquirectUv(direction));
            back.forEach((component, axis) => expect(component).toBeCloseTo(direction[axis] / length, 12));
        }
    });

    it('row weights integrate to 4 pi (solid angle) and pi (horizontal irradiance of a uniform sky)', () => {
        let solidAngle = 0;
        let irradianceWeight = 0;
        for (let row = 0; row < 64; row++) {
            solidAngle += rowSolidAngle(row, 128, 64) * 128;
            irradianceWeight += rowHorizontalIrradianceWeight(row, 128, 64) * 128;
        }
        expect(solidAngle).toBeCloseTo(4 * Math.PI, 10);
        expect(irradianceWeight).toBeCloseTo(Math.PI, 10);
    });
});

describe('the Preetham sky', () => {
    const sky = createPreethamSky([-0.3, 0.9, -0.1], 3);

    it('equals its zenith values at the zenith', () => {
        const zenith = preethamSkyXyY(sky, [0, 1, 0]);
        expect(zenith.luminance).toBeCloseTo(sky.zenith.luminance, 12);
        expect(zenith.x).toBeCloseTo(sky.zenith.x, 12);
        expect(zenith.y).toBeCloseTo(sky.zenith.y, 12);
        // A clear zenith is blue: x and y well below the white point (0.3127, 0.3290).
        expect(zenith.x).toBeLessThan(0.3);
        expect(zenith.y).toBeLessThan(0.31);
    });

    it('is mirror-symmetric about the vertical plane through the sun', () => {
        // Reflect across the plane spanned by the up axis and the sun's horizontal direction.
        const horizontal = Math.hypot(sky.sunDirection[0], sky.sunDirection[2]);
        const sx = sky.sunDirection[0] / horizontal;
        const sz = sky.sunDirection[2] / horizontal;
        for (const direction of [[0.6, 0.3, 0.74], [-0.2, 0.1, 0.97], [0.9, 0.42, -0.1]] as [number, number, number][]) {
            const along = direction[0] * sx + direction[2] * sz;
            const mirrored: [number, number, number] = [2 * along * sx - direction[0], direction[1], 2 * along * sz - direction[2]];
            const a = preethamSkyXyY(sky, direction);
            const b = preethamSkyXyY(sky, mirrored);
            expect(b.luminance).toBeCloseTo(a.luminance, 10);
            expect(b.x).toBeCloseTo(a.x, 10);
        }
    });

    it('brightens towards the sun and towards the horizon', () => {
        expect(perez(sky.luminance, 80 * DEG, 10 * DEG)).toBeGreaterThan(perez(sky.luminance, 80 * DEG, 120 * DEG));
        expect(perez(sky.luminance, 85 * DEG, 90 * DEG)).toBeGreaterThan(perez(sky.luminance, 30 * DEG, 90 * DEG));
    });

    it('refuses turbidities the model was not fitted for', () => {
        expect(() => createPreethamSky([0, 1, 0], 1)).toThrow(/turbidity/);
        expect(() => createPreethamSky([1, -0.1, 0], 3)).toThrow(/above the horizon/);
    });
});

describe('the clear-sky calibration ratio (Ineichen and Perez 2002)', () => {
    it('matches a hand calculation at a 70.64 degree sun and turbidity 3', () => {
        const elevation = 70.64 * DEG;
        const zenith = Math.PI / 2 - elevation;
        const airMass = 1 / (Math.cos(zenith) + 0.50572 * (96.07995 - 19.36) ** -1.6364);
        expect(kastenYoungAirMass(zenith)).toBeCloseTo(airMass, 12);
        const global = 0.868 * Math.cos(zenith) * Math.exp(-0.0387 * airMass * 3);
        const beam = 0.827 * Math.exp(-0.09 * airMass * 2) * Math.cos(zenith);
        const result = ineichenPerezClearSky(elevation, 3);
        expect(result.globalHorizontal).toBeCloseTo(global, 12);
        expect(result.directHorizontal).toBeCloseTo(beam, 12);
        expect(result.skyToSunHorizontalRatio).toBeCloseTo((global - beam) / beam, 12);
    });

    it('gives realistic clear-sky diffuse fractions (6-30% with the sun above 30 degrees), growing with haze and a lower sun', () => {
        for (const turbidity of [2, 3, 4]) {
            let previous = Infinity;
            for (const elevationDeg of [10, 20, 30, 50, 70, 90]) {
                const { diffuseFraction } = ineichenPerezClearSky(elevationDeg * DEG, turbidity);
                if (elevationDeg >= 30) {
                    expect(diffuseFraction).toBeGreaterThan(0.06);
                    expect(diffuseFraction).toBeLessThan(0.3);
                }
                expect(diffuseFraction).toBeLessThanOrEqual(previous + 1e-12);
                previous = diffuseFraction;
            }
        }
        expect(ineichenPerezClearSky(50 * DEG, 4).diffuseFraction).toBeGreaterThan(ineichenPerezClearSky(50 * DEG, 2.5).diffuseFraction);
    });

    it('refuses a sun too low for the model', () => {
        expect(() => ineichenPerezClearSky(2 * DEG, 3)).toThrow(/elevation/);
    });
});

describe('building a calibrated sky', () => {
    it('lights a flat open ground with the stated share of the sun', () => {
        const built = smallSky();
        const sunHorizontal = 3 * (0.2126 + 0.7152 * 0.95 + 0.0722 * 0.85) * Math.sin(built.sun.elevationRad);
        const ratio = ineichenPerezClearSky(built.sun.elevationRad, 3).skyToSunHorizontalRatio;
        expect(measureHalfImage(built.image).luminance / sunHorizontal).toBeCloseTo(ratio, 4);
    });

    it('scales with environment.intensity', () => {
        const base = smallSky().measured.luminance;
        const doubled = smallSky({ ...sponzaLikeSky, intensity: 2 }).measured.luminance;
        expect(doubled / base).toBeCloseTo(2, 4);
    });

    it('is black below the horizon and has no sun disc', () => {
        const { image, sun: skySun } = smallSky();
        for (let row = image.height / 2; row < image.height; row++) {
            for (let column = 0; column < image.width; column += 7) expect(texel(image, column, row)).toEqual([0, 0, 0]);
        }
        // The texel nearest the sun is the glow, not a disc: within a few times its neighbours.
        const { u, v } = directionToEquirectUv(skySun.direction);
        const column = Math.floor(u * image.width);
        const row = Math.floor(v * image.height);
        const centre = texel(image, column, row)[1];
        const neighbour = texel(image, (column + 3) % image.width, row)[1];
        expect(centre / neighbour).toBeLessThan(2);
    });

    it('puts the sun glow at the sun azimuth (orientation)', () => {
        const environment: ProceduralEnvironment = { type: 'procedural', sunDirection: [-0.5, 0.6, -0.62], turbidity: 3, sunIlluminance: 3 };
        const { image, sun: skySun } = buildSkyImage({ environment, lights: [] }, { width: 256, height: 128 });
        // Search the ring 10 degrees above the horizon for its brightest texel.
        const row = Math.floor((80 / 180) * image.height);
        let brightest = 0;
        for (let column = 1; column < image.width; column++) {
            if (texel(image, column, row)[1] > texel(image, brightest, row)[1]) brightest = column;
        }
        const azimuth = Math.atan2(skySun.direction[2], skySun.direction[0]);
        const brightestAzimuth = ((brightest + 0.5) / image.width - 0.5) * 2 * Math.PI;
        expect(Math.abs(brightestAzimuth - azimuth)).toBeLessThan((2 * Math.PI) / image.width + 1e-9);
    });

    it('finds the sun light, or needs environment.sunIlluminance, never both', () => {
        expect(resolveSkySun(sponzaLikeSky, [sun]).illuminance).toBeCloseTo(3 * (0.2126 + 0.7152 * 0.95 + 0.0722 * 0.85), 10);
        expect(() => resolveSkySun(sponzaLikeSky, [])).toThrow(/sunIlluminance is not set/);
        expect(() => resolveSkySun({ ...sponzaLikeSky, sunIlluminance: 3 }, [sun])).toThrow(/only for scenes without a sun light/);
        expect(resolveSkySun({ ...sponzaLikeSky, sunIlluminance: 5 }, []).illuminance).toBe(5);
        const otherWay: ManifestLight = { ...sun, direction: [1, -1, 0] };
        expect(() => resolveSkySun(sponzaLikeSky, [otherWay])).toThrow(/no directional light/);
    });
});

describe('the committed sky files in scenes/sky', async () => {
    const manifests: SceneManifest[] = await loadSkyManifests();

    it('exist for every scene with a procedural sky and match a fresh generation', async () => {
        expect(manifests.length).toBeGreaterThan(0);
        for (const manifest of manifests) {
            expect(await compareWithCommitted(manifest, buildSkyImage(manifest)), manifest.id).toEqual([]);
        }
    });

    it('stay small enough to commit (under 1 MB each)', async () => {
        for (const manifest of manifests) {
            const bytes = await readFile(join(skyDirectory, skyFileName(manifest.id)));
            expect(bytes.length, manifest.id).toBeLessThan(1024 * 1024);
        }
    });

    it('give the open-ground scene its known answer: ground radiance = albedo x E / pi', async () => {
        const sidecar = JSON.parse(await readFile(join(repoRoot, 'scenes', 'sky', 'open-ground.json'), 'utf8')) as SkySidecar;
        const image = await decodeSkyFile(new Uint8Array(await readFile(join(repoRoot, 'scenes', 'sky', 'open-ground.exr'))), (bytes) => inflateSync(bytes));
        const measured = measureHalfImage(image);
        expect(measured.luminance).toBeCloseTo(sidecar.measured.skyHorizontalIlluminance, 5);
        // The number the viewer and Cycles renders are checked against (tools/sky/README.md).
        const groundRadiance = (OPEN_GROUND_ALBEDO * measured.luminance) / Math.PI;
        expect(groundRadiance).toBeGreaterThan(0.04);
        expect(groundRadiance).toBeLessThan(0.06);
    });
});
