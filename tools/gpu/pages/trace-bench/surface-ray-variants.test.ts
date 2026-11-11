/**
 * Tests of the surface-ray variants (RC5.4): the baseline is RC1.5's probe
 * set exactly, each variant changes only what it says, the orders are
 * permutations, and the count summary's SIMD efficiency is what it claims.
 * No GPU.
 */

import { describe, expect, it } from 'vitest';
import { RAY_FLOATS } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { createRandom } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { COUNTED_WORD, COUNTED_WORDS, NO_ORIGIN } from './counted-walk-wgsl.ts';
import { makeProbeRays, offsetRayOrigin } from './rc-rays.ts';
import {
    cellMajorOrder,
    directionBin,
    makeVariantProbeRays,
    meanInstanceBoxesAtOrigins,
    mortonCode,
    probeOrigins,
    rayOriginIds,
    reorder,
    shuffledOrder,
    sortedOrder,
    summariseCounts,
    transplantOrigins,
    withInterval,
    type SurfaceProbes,
} from './surface-ray-variants.ts';

/** Three probes on a floor, a wall and a slope. */
function sampleProbes(): SurfaceProbes {
    const slope = Math.SQRT1_2;
    return {
        positions: Float64Array.from([1, 0, 2, 5, 1.5, -3, -2, 0.5, 0.25]),
        normals: Float64Array.from([0, 1, 0, -1, 0, 0, 0, slope, slope]),
        originIds: Uint32Array.from([0, 10, 1, 20, 2, 30]),
        count: 3,
    };
}

function isPermutation(order: Uint32Array): boolean {
    const seen = new Uint8Array(order.length);
    for (const index of order) {
        if (index >= order.length || seen[index]) return false;
        seen[index] = 1;
    }
    return true;
}

describe('surface-ray variants', () => {
    it("makes RC1.5's probe set exactly for the stratified cosine kind", () => {
        const probes = sampleProbes();
        const origins = probeOrigins(probes, 'platform');
        const ours = makeVariantProbeRays({ origins, normals: probes.normals, count: 3, raysPerProbe: 64, kind: 'cosine-stratified', random: createRandom(7) });
        const rc = makeProbeRays({ origins, normals: probes.normals, count: 3 }, 64, createRandom(7));
        expect(Array.from(ours)).toEqual(Array.from(rc));
    });

    it("puts the platform origin where rc-rays' offset puts it, and a pushed one that far along the normal", () => {
        const probes = sampleProbes();
        const platform = probeOrigins(probes, 'platform');
        expect(Array.from(platform.subarray(3, 6))).toEqual(offsetRayOrigin([5, 1.5, -3], [-1, 0, 0]));
        const pushed = probeOrigins(probes, 0.5);
        expect(pushed[3]).toBeCloseTo(4.5, 12);
        expect(pushed[1]).toBeCloseTo(0.5, 12);
    });

    it('keeps hemisphere kinds on the normal side and spreads uniform-sphere over both', () => {
        const probes = sampleProbes();
        const origins = probeOrigins(probes, 'platform');
        for (const kind of ['cosine-random', 'uniform-hemisphere', 'uniform-sphere'] as const) {
            const rays = makeVariantProbeRays({ origins, normals: probes.normals, count: 3, raysPerProbe: 256, kind, random: createRandom(3) });
            let below = 0;
            for (let ray = 0; ray < rays.length / RAY_FLOATS; ray++) {
                const probe = Math.floor(ray / 256);
                const base = ray * RAY_FLOATS;
                const cosine =
                    rays[base + 4] * probes.normals[probe * 3] + rays[base + 5] * probes.normals[probe * 3 + 1] + rays[base + 6] * probes.normals[probe * 3 + 2];
                if (cosine < 0) below++;
            }
            if (kind === 'uniform-sphere') expect(below / (3 * 256)).toBeGreaterThan(0.4);
            else expect(below).toBe(0);
        }
    });

    it('gives every probe the same shared directions', () => {
        const probes = sampleProbes();
        const shared = Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, -1, 0.6, 0.8, 0]);
        const rays = makeVariantProbeRays({ origins: probeOrigins(probes, 'platform'), normals: probes.normals, count: 3, raysPerProbe: 4, kind: 'shared-sphere', random: createRandom(1), sharedDirections: shared });
        for (let probe = 0; probe < 3; probe++) {
            for (let cell = 0; cell < 4; cell++) {
                const base = (probe * 4 + cell) * RAY_FLOATS;
                expect(rays[base + 4]).toBeCloseTo(shared[cell * 3], 6);
                expect(rays[base + 6]).toBeCloseTo(shared[cell * 3 + 2], 6);
            }
        }
    });

    it('changes only the interval or only the origins', () => {
        const probes = sampleProbes();
        const rays = makeVariantProbeRays({ origins: probeOrigins(probes, 'platform'), normals: probes.normals, count: 3, raysPerProbe: 4, kind: 'cosine-random', random: createRandom(5) });
        const cut = withInterval(rays, { tMax: 2 });
        const moved = transplantOrigins(rays, 4, Float64Array.from([9, 9, 9, 8, 8, 8, 7, 7, 7]));
        for (let ray = 0; ray < 12; ray++) {
            const base = ray * RAY_FLOATS;
            expect(cut[base + 7]).toBe(2);
            expect(Array.from(cut.subarray(base, base + 7))).toEqual(Array.from(rays.subarray(base, base + 7)));
            expect(moved[base]).toBe(9 - Math.floor(ray / 4));
            expect(Array.from(moved.subarray(base + 3, base + 8))).toEqual(Array.from(rays.subarray(base + 3, base + 8)));
        }
    });

    it('reorders by permutations, rays and their origin records together', () => {
        const probes = sampleProbes();
        const rays = makeVariantProbeRays({ origins: probeOrigins(probes, 'platform'), normals: probes.normals, count: 3, raysPerProbe: 64, kind: 'cosine-stratified', random: createRandom(2) });
        const ids = rayOriginIds(probes.originIds, 3, 64);
        const box = { min: [-5, -5, -5] as [number, number, number], max: [5, 5, 5] as [number, number, number] };
        for (const order of [shuffledOrder(192, createRandom(9)), cellMajorOrder(3, 64), sortedOrder(rays, box, 'direction-then-origin'), sortedOrder(rays, box, 'origin-then-direction')]) {
            expect(isPermutation(order)).toBe(true);
            const movedRays = reorder(rays, order, RAY_FLOATS);
            const movedIds = reorder(ids, order, 2);
            for (let slot = 0; slot < 192; slot++) {
                expect(movedRays[slot * RAY_FLOATS + 4]).toBe(rays[order[slot] * RAY_FLOATS + 4]);
                expect(movedIds[slot * 2 + 1]).toBe(ids[order[slot] * 2 + 1]);
            }
        }
        expect(Array.from(cellMajorOrder(2, 3))).toEqual([0, 3, 1, 4, 2, 5]);
        expect(Array.from(rayOriginIds(null, 1, 2))).toEqual([NO_ORIGIN, NO_ORIGIN, NO_ORIGIN, NO_ORIGIN]);
    });

    it('bins directions and Morton-codes points into their ranges', () => {
        for (const [x, y, z] of [[1, 0, 0], [0, -1, 0], [0, 0, -1], [0.5, 0.5, -0.7071]]) {
            const bin = directionBin(x, y, z, 8);
            expect(bin).toBeGreaterThanOrEqual(0);
            expect(bin).toBeLessThan(64);
        }
        expect(directionBin(0, 0, 1, 8)).not.toBe(directionBin(0, 0, -1, 8));
        const box = { min: [0, 0, 0] as [number, number, number], max: [1, 1, 1] as [number, number, number] };
        expect(mortonCode([0, 0, 0], box)).toBe(0);
        expect(mortonCode([1, 1, 1], box)).toBe(2 ** 30 - 1);
        expect(mortonCode([0.6, 0, 0], box)).toBeGreaterThan(mortonCode([0.4, 0, 0], box));
    });

    it("summarises counts: means, self-hits and the SIMD group's share of busy lanes", () => {
        const rays = 64;
        const buffer = new ArrayBuffer(rays * COUNTED_WORDS * 4);
        const words = new Uint32Array(buffer);
        const floats = new Float32Array(buffer);
        for (let ray = 0; ray < rays; ray++) {
            const base = ray * COUNTED_WORDS;
            floats[base + COUNTED_WORD.t] = ray % 2 === 0 ? 2 : -1;
            // In each group of 32 one ray takes 32 steps and the rest 1: 63 of 1024 lane-steps busy.
            words[base + COUNTED_WORD.steps] = ray % 32 === 0 ? 32 : 1;
            words[base + COUNTED_WORD.instanceEntries] = 3;
            words[base + COUNTED_WORD.instancesWithHit] = 1;
            words[base + COUNTED_WORD.firstHitIsOrigin] = ray === 5 ? 1 : 0;
        }
        const summary = summariseCounts(buffer, rays);
        expect(summary.hitFraction).toBe(0.5);
        expect(summary.medianHitT).toBe(2);
        expect(summary.meanSteps).toBeCloseTo(63 / 32, 12);
        expect(summary.simd32Efficiency).toBeCloseTo(63 / 1024, 12);
        expect(summary.simd32StepsPaidPerRay).toBe(32);
        expect(summary.meanInstanceEntriesWithoutHit).toBe(2);
        expect(summary.selfHitFraction).toBe(1 / 64);
    });

    it('counts the instance boxes around sampled origins', () => {
        const rays = new Float32Array(2 * RAY_FLOATS);
        rays.set([0.5, 0.5, 0.5], 0);
        rays.set([5, 5, 5], RAY_FLOATS);
        const boxes = [
            { min: [0, 0, 0] as [number, number, number], max: [1, 1, 1] as [number, number, number] },
            { min: [0, 0, 0] as [number, number, number], max: [10, 10, 10] as [number, number, number] },
        ];
        expect(meanInstanceBoxesAtOrigins(rays, boxes, 1)).toBe(1.5);
    });
});
