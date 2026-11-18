import { describe, expect, it } from 'vitest';
import { DataUtils, SphericalHarmonics3, Vector3 } from 'three/webgpu';
import { L2_SH_FLOATS_PER_PROBE, projectIrradianceToSH, shProjectionDirections, type IrradianceSHGrid } from '@pmndrs/ddgi-core';
import { LIGHT_PROBE_GRID_JSON_FORMAT_VERSION, lightProbeGridFromJSON, lightProbeGridToJSON, packCoefficientAtlas, toLightProbeGrid } from './light-probe-grid.ts';

type Rgb = readonly [number, number, number];

/** three's own evaluation of radiance SH: SphericalHarmonics3.getIrradianceAt (the same constants as TSL's getShIrradianceAt). */
function threeIrradianceAt(coefficients: ArrayLike<number>, normal: Vector3): Rgb {
    const sh = new SphericalHarmonics3();
    sh.fromArray(Array.from(coefficients));
    const result = sh.getIrradianceAt(normal, new Vector3());
    return [result.x, result.y, result.z];
}

/** Normals spread over the sphere, other than the projection's own directions. */
function testNormals(count: number): Vector3[] {
    const normals: Vector3[] = [];
    for (let index = 0; index < count; index++) {
        const z = 1 - (2 * index + 0.37) / count;
        const radius = Math.sqrt(Math.max(0, 1 - z * z));
        const azimuth = index * 2.1;
        normals.push(new Vector3(radius * Math.cos(azimuth), radius * Math.sin(azimuth), z).normalize());
    }
    return [new Vector3(0, 1, 0), new Vector3(0, -1, 0), new Vector3(1, 0, 0), new Vector3(0, 0, -1), ...normals];
}

function expectRgbClose(actual: Rgb, expected: Rgb, tolerance: number): void {
    for (let channel = 0; channel < 3; channel++) expect(Math.abs(actual[channel] - expected[channel])).toBeLessThan(tolerance);
}

describe('L2 SH export: the convention three samples with', () => {
    it('gives back a uniform irradiance through three’s formula', () => {
        const coefficients = projectIrradianceToSH(() => [1, 2, 3]);
        // Radiance SH: a uniform E = 1 is radiance 1 / π, so c0 = 1 / (π Y00).
        expect(coefficients[0]).toBeCloseTo(1 / (Math.PI * 0.28209479177387814), 5);
        // The higher bands are 0 up to the quadrature's error (a few parts in 10,000 of c0, per unit of E).
        for (let float = 3; float < L2_SH_FLOATS_PER_PROBE; float++) expect(Math.abs(coefficients[float])).toBeLessThan(2e-3);
        for (const normal of testNormals(32)) expectRgbClose(threeIrradianceAt(coefficients, normal), [1, 2, 3], 1e-3);
    });

    it('gives back the irradiance of a sky lit from above only (radiance 1 over y > 0)', () => {
        // The irradiance of that radiance is exactly π (1 + n_y) / 2: linear in n, so L2 holds it.
        const irradiance = (y: number): number => (Math.PI * (1 + y)) / 2;
        const coefficients = projectIrradianceToSH((_x, y) => [irradiance(y), irradiance(y), irradiance(y)]);
        for (const normal of testNormals(64)) {
            const expected = irradiance(normal.y);
            expectRgbClose(threeIrradianceAt(coefficients, normal), [expected, expected, expected], 2e-3);
        }
        // And the coefficients are that radiance's own: c0 = ∫ 1 · Y00 over the upper hemisphere = 2π Y00.
        expect(coefficients[0]).toBeCloseTo(2 * Math.PI * 0.28209479177387814, 3);
    });

    it('gives back any L2 irradiance, every band and colour, through three’s formula', () => {
        const irradianceAt = (x: number, y: number, z: number): Rgb => [
            1 + 0.5 * x - 0.2 * y,
            2 + 0.3 * y * z + 0.4 * x * y,
            0.5 + 0.2 * (x * x - y * y) + 0.1 * z + 0.15 * (3 * z * z - 1) - 0.1 * x * z,
        ];
        const coefficients = projectIrradianceToSH(irradianceAt);
        for (const normal of testNormals(64)) expectRgbClose(threeIrradianceAt(coefficients, normal), irradianceAt(normal.x, normal.y, normal.z), 2e-3);
    });

    it('spreads the projection directions evenly over the whole sphere', () => {
        const directions = shProjectionDirections(1024);
        let weight = 0;
        const mean = [0, 0, 0];
        for (let index = 0; index < 1024; index++) {
            const length = Math.hypot(directions[index * 4], directions[index * 4 + 1], directions[index * 4 + 2]);
            expect(length).toBeCloseTo(1, 5);
            weight += directions[index * 4 + 3];
            for (let axis = 0; axis < 3; axis++) mean[axis] += directions[index * 4 + axis] / 1024;
        }
        expect(weight).toBeCloseTo(4 * Math.PI, 4);
        for (const axis of mean) expect(Math.abs(axis)).toBeLessThan(2e-3);
    });
});

function sampleGrid(counts: readonly [number, number, number]): IrradianceSHGrid {
    const probeCount = counts[0] * counts[1] * counts[2];
    const coefficients = new Float32Array(probeCount * L2_SH_FLOATS_PER_PROBE);
    // Values with no short decimal form, so an inexact round trip would show.
    for (let float = 0; float < coefficients.length; float++) coefficients[float] = Math.sin(float * 12.9898 + 0.1) * 3.7 + 1 / 3;
    return {
        counts,
        origin: [-1.3, 0.25, 2.1],
        spacing: [0.4, 0.7, 0.35],
        coefficients,
        probeStates: new Uint8Array(probeCount),
        filledProbes: 0,
        composedLayers: 1,
    };
}

describe('toLightProbeGrid', () => {
    it('puts three’s probes where ours are', () => {
        const data = sampleGrid([3, 2, 4]);
        const grid = toLightProbeGrid(data);
        const position = new Vector3();
        for (const [x, y, z] of [[0, 0, 0], [2, 1, 3], [1, 0, 2]] as const) {
            grid.getProbePosition(x, y, z, position);
            expect(position.x).toBeCloseTo(data.origin[0] + x * data.spacing[0], 6);
            expect(position.y).toBeCloseTo(data.origin[1] + y * data.spacing[1], 6);
            expect(position.z).toBeCloseTo(data.origin[2] + z * data.spacing[2], 6);
        }
        expect(grid.boundingBox.min.x).toBeCloseTo(data.origin[0] - data.spacing[0] / 2, 6);
        expect(grid.boundingBox.max.z).toBeCloseTo(data.origin[2] + 3.5 * data.spacing[2], 6);
        expect(grid.texture).toBe(grid.coefficientTexture);
    });

    it('packs the atlas so LightProbeGridNode’s unpacking reads each probe’s 27 floats', () => {
        const counts: [number, number, number] = [3, 2, 4];
        const data = sampleGrid(counts);
        const atlas = packCoefficientAtlas(data.coefficients, counts);
        const [nx, ny, nz] = counts;
        const paddedSlices = nz + 2;
        expect([atlas.image.width, atlas.image.height, atlas.image.depth]).toEqual([nx, ny, 7 * paddedSlices]);
        const texels = atlas.image.data as Uint16Array;
        const texel = (slice: number, x: number, y: number): number[] =>
            Array.from(texels.subarray(((slice * ny + y) * nx + x) * 4, ((slice * ny + y) * nx + x) * 4 + 4), (half) => DataUtils.fromHalfFloat(half));
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const probe = x + y * nx + z * nx * ny;
                    // LightProbeGridNode: s0.xyz, (s0.w, s1.xy), (s1.zw, s2.x), s2.yzw, s3.xyz, ... s6.xyz.
                    const read = Array.from({ length: 7 }, (_, subVolume) => texel(subVolume * paddedSlices + 1 + z, x, y)).flat();
                    for (let float = 0; float < L2_SH_FLOATS_PER_PROBE; float++) {
                        const expected = DataUtils.fromHalfFloat(DataUtils.toHalfFloat(data.coefficients[probe * L2_SH_FLOATS_PER_PROBE + float]));
                        expect(read[float]).toBe(expected);
                    }
                    expect(read[27]).toBe(0);
                    // The padding slices copy the edge slices.
                    for (let subVolume = 0; subVolume < 7; subVolume++) {
                        if (z === 0) expect(texel(subVolume * paddedSlices, x, y)).toEqual(texel(subVolume * paddedSlices + 1, x, y));
                        if (z === nz - 1) expect(texel(subVolume * paddedSlices + nz + 1, x, y)).toEqual(texel(subVolume * paddedSlices + nz, x, y));
                    }
                }
            }
        }
    });
});

describe('LightProbeGrid JSON', () => {
    it('round-trips exactly: every coefficient bit for bit, the placement and the atlas', () => {
        const grid = toLightProbeGrid(sampleGrid([4, 3, 5]), { intensity: 0.8, priority: 2 });
        const text = JSON.stringify(lightProbeGridToJSON(grid));
        const back = lightProbeGridFromJSON(text);
        expect(new Uint32Array(back.coefficients.buffer)).toEqual(new Uint32Array(grid.coefficients.buffer));
        expect([back.width, back.height, back.depth]).toEqual([grid.width, grid.height, grid.depth]);
        expect(back.resolution.toArray()).toEqual(grid.resolution.toArray());
        expect(back.position.toArray()).toEqual(grid.position.toArray());
        expect(back.boundingBox.min.toArray()).toEqual(grid.boundingBox.min.toArray());
        expect([back.intensity, back.priority]).toEqual([0.8, 2]);
        expect(back.coefficientTexture.image.data).toEqual(grid.coefficientTexture.image.data);
        // And once more: the text itself is stable.
        expect(JSON.stringify(lightProbeGridToJSON(back))).toBe(text);
    });

    it('writes its format version and refuses others by name', () => {
        const json = lightProbeGridToJSON(toLightProbeGrid(sampleGrid([2, 2, 2])));
        expect(json.formatVersion).toBe(LIGHT_PROBE_GRID_JSON_FORMAT_VERSION);
        expect(json.type).toBe('LightProbeGrid');
        expect(() => lightProbeGridFromJSON({ ...json, formatVersion: 2 as 1 })).toThrow(/format version 2/);
        expect(() => lightProbeGridFromJSON({ ...json, type: 'Other' as 'LightProbeGrid' })).toThrow(/not LightProbeGrid JSON/);
        expect(() => lightProbeGridFromJSON({ ...json, coefficients: json.coefficients.slice(1) })).toThrow(/needs 216 coefficients/);
    });
});
