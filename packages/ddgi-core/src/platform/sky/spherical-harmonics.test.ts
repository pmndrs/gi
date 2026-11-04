import { describe, expect, it } from 'vitest';
import { buildCalibratedSky } from './calibrated-sky.ts';
import { halfToFloat } from './half-float.ts';
import {
    bruteForceIrradiance,
    convolveWithCosineLobe,
    evaluateIrradiance,
    projectEquirectRadiance,
    SH_COEFFICIENT_BANDS,
    SH_COEFFICIENT_COUNT,
    shBasis,
} from './spherical-harmonics.ts';

/** Gauss-Legendre nodes and weights on [-1, 1] (Newton on the Legendre recurrence). */
function gaussLegendre(count: number): { nodes: number[]; weights: number[] } {
    const nodes: number[] = [];
    const weights: number[] = [];
    for (let i = 1; i <= count; i++) {
        let x = Math.cos((Math.PI * (i - 0.25)) / (count + 0.5));
        let derivative = 0;
        for (let iteration = 0; iteration < 100; iteration++) {
            let p0 = 1;
            let p1 = x;
            for (let k = 2; k <= count; k++) {
                const p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k;
                p0 = p1;
                p1 = p2;
            }
            derivative = (count * (x * p1 - p0)) / (x * x - 1);
            const step = p1 / derivative;
            x -= step;
            if (Math.abs(step) < 1e-15) break;
        }
        nodes.push(x);
        weights.push(2 / ((1 - x * x) * derivative * derivative));
    }
    return { nodes, weights };
}

describe('the sky SH basis (bands 0, 1, 2, 4)', () => {
    it('has 18 coefficients in band order', () => {
        expect(SH_COEFFICIENT_BANDS).toEqual([0, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4]);
        expect(SH_COEFFICIENT_BANDS.length).toBe(SH_COEFFICIENT_COUNT);
    });

    it('is orthonormal over the sphere (exact quadrature: Gauss-Legendre in z, uniform in azimuth)', () => {
        // Products of two band-4 functions are degree-8 polynomials: 16 nodes in z and 32 in azimuth integrate them exactly.
        const { nodes, weights } = gaussLegendre(16);
        const azimuthCount = 32;
        const gram = new Float64Array(SH_COEFFICIENT_COUNT * SH_COEFFICIENT_COUNT);
        const basis = new Float64Array(SH_COEFFICIENT_COUNT);
        nodes.forEach((z, nodeIndex) => {
            const radius = Math.sqrt(1 - z * z);
            for (let a = 0; a < azimuthCount; a++) {
                const azimuth = ((a + 0.5) / azimuthCount) * 2 * Math.PI;
                shBasis(radius * Math.cos(azimuth), radius * Math.sin(azimuth), z, basis);
                const weight = (weights[nodeIndex] * 2 * Math.PI) / azimuthCount;
                for (let i = 0; i < SH_COEFFICIENT_COUNT; i++) {
                    for (let j = 0; j < SH_COEFFICIENT_COUNT; j++) gram[i * SH_COEFFICIENT_COUNT + j] += basis[i] * basis[j] * weight;
                }
            }
        });
        for (let i = 0; i < SH_COEFFICIENT_COUNT; i++) {
            for (let j = 0; j < SH_COEFFICIENT_COUNT; j++) expect(gram[i * SH_COEFFICIENT_COUNT + j]).toBeCloseTo(i === j ? 1 : 0, 10);
        }
    });

    it('gives pi L for every normal under a uniform sky, and the exact pi L on flat ground under a uniform upper hemisphere', () => {
        // The basis is taken at texel centres (the midpoint rule, as on the GPU), which leaves bands 2 and 4
        // slightly off zero for a constant sky: about 2e-5 of pi L at 256 x 128, 3.5e-4 at 64 x 32.
        const width = 256;
        const height = 128;
        const uniform = new Float64Array(width * height * 3).fill(0.5);
        const irradiance = convolveWithCosineLobe(projectEquirectRadiance(uniform, width, height));
        for (const normal of [[0, 1, 0], [1, 0, 0], [0, -1, 0], [0.6, 0, -0.8]] as const) {
            expect(Math.abs(evaluateIrradiance(irradiance, normal)[1] / (Math.PI * 0.5) - 1)).toBeLessThan(1e-4);
        }
        const upperOnly = new Float64Array(width * height * 3);
        upperOnly.fill(0.5, 0, (width * height * 3) / 2);
        const upper = convolveWithCosineLobe(projectEquirectRadiance(upperOnly, width, height));
        // Bands 0 and 1 already give exactly pi L at the zenith; band 2 and 4 integrate to 0 over a hemisphere (up to the midpoint rule).
        expect(Math.abs(evaluateIrradiance(upper, [0, 1, 0])[0] / (Math.PI * 0.5) - 1)).toBeLessThan(1e-4);
    });

    it('meets the ±0.5% reference tolerance on an upward-facing ground under the open-ground sky (worst channel within 0.35%)', () => {
        // The open-ground sky file's settings (scenes/manifests/open-ground.json): sun of 3 lux, turbidity 3.
        const sky = buildCalibratedSky({ sunDirection: [-0.5, 0.6, -0.62], turbidity: 3, sunIlluminance: 3 });
        const rgb = Float64Array.from(sky.image.halves, (half) => halfToFloat(half));
        const { width, height } = sky.image;
        const irradiance = convolveWithCosineLobe(projectEquirectRadiance(rgb, width, height));
        const up = evaluateIrradiance(irradiance, [0, 1, 0]);
        for (let channel = 0; channel < 3; channel++) {
            // `measured` integrates the same texels with exact per-row cosine weights.
            expect(Math.abs(up[channel] / sky.measured.rgb[channel] - 1)).toBeLessThan(0.0035);
        }
        // Tilted and vertical surfaces: the truncation error is larger, but stays within 2% of the
        // brute-force integral for these (luminance-weighted), and is reported by the GPU page.
        for (const normal of [[0.7071, 0.7071, 0], [1, 0, 0], [-0.5, 0, 0.866], [0, -1, 0.0001]] as const) {
            const length = Math.hypot(...normal);
            const unit: [number, number, number] = [normal[0] / length, normal[1] / length, normal[2] / length];
            const sh = evaluateIrradiance(irradiance, unit);
            const reference = bruteForceIrradiance(rgb, width, height, unit);
            const luminance = (value: readonly number[]) => 0.2126 * value[0] + 0.7152 * value[1] + 0.0722 * value[2];
            const scale = luminance(sky.measured.rgb);
            // Relative to the ground's irradiance, so a dark downward-facing surface isn't judged by a tiny denominator.
            expect(Math.abs(luminance(sh) - luminance(reference)) / scale).toBeLessThan(0.02);
        }
    });
});
