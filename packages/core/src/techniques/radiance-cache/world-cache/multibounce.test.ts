import { describe, expect, it } from 'vitest';
import type { Vec3 } from '../../../platform/contract/math.ts';
import { DEFAULT_HASH_GRID_CACHE_OPTIONS } from './capacity.ts';
import { HASH_GRID_CONSTANTS_BYTES, HASH_GRID_CONSTANTS_WORD, hashGridCacheWgsl } from './hash-grid-wgsl.ts';
import { hashGridCacheBufferLayout, hashGridCacheCapacity } from './capacity.ts';
import { pcgHash, updateTilesAccumulate, updateTilesAccumulateIndirect, type CellValue } from './hash-grid-cpu.ts';
import {
    makeRandom,
    mapToCosineHemisphere,
    multibounceBypassAddition,
    multibounceDraw,
    multibounceSample,
    multibounceSettledIndirect,
    quaternionInverse,
    quaternionRotationZ,
    quaternionTransform,
    rand,
    randInt,
    sampleBRDFAndEvaluateDiffuse,
} from './multibounce-cpu.ts';
import {
    generateMultibounceReservoirsWgsl,
    generateReservoirsWgsl,
    hashGridCacheRayListLayout,
    MULTIBOUNCE_DIRECT_WORDS,
    MULTIBOUNCE_INFO_WORDS,
    MULTIBOUNCE_RAY_WORDS,
    POPULATION_ENTRY_POINT,
    populateMultibounceCellsWgsl,
    RESERVOIR_WORDS,
    updateMultibounceCellsWgsl,
} from './hash-grid-population-wgsl.ts';

// ---- Independent reference: Capsaicin's PCG in BigInt, mod 2^32 at every step ----

const MASK = 0xffffffffn;

function pcgOutput(state: bigint): number {
    const word = (((state >> ((state >> 28n) + 4n)) ^ state) * 277803737n) & MASK;
    return Number((word >> 22n) ^ word);
}

function pcgHashBigInt(value: number): bigint {
    return BigInt(pcgOutput((BigInt(value) * 747796405n + 2891336453n) & MASK));
}

/** MakeRandom(seed, index) then `count` randInt calls (random_number_generator.hlsl:38-46, 108-116). */
function randIntsBigInt(seed: number, index: number, count: number): number[] {
    const increment = ((BigInt(index) << 1n) | 1n) & MASK;
    let state = ((pcgHashBigInt(seed) + increment) * 747796405n + increment) & MASK;
    const out: number[] = [];
    for (let i = 0; i < count; i++) {
        out.push(pcgOutput(state));
        state = (state * 747796405n + 2891336453n) & MASK;
    }
    return out;
}

function length(v: readonly number[]): number {
    return Math.hypot(...v);
}

function dot(a: readonly number[], b: readonly number[]): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

describe("Capsaicin's random numbers (CPU twin of rcMakeRandom)", () => {
    it('gives the BigInt reference bits for MakeRandom(seed, index) and randInt', () => {
        for (const [seed, index] of [
            [0, 0],
            [1, 0],
            [12345, 77],
            [0xffffffff, 0xfffffffe],
            [230399, 4294967295],
        ]) {
            const random = makeRandom(seed, index);
            expect([randInt(random), randInt(random), randInt(random)]).toEqual(randIntsBigInt(seed, index, 3));
        }
    });

    it('rand is in [0, 1) and its draws are uniform: about 70% at or below 0.7', () => {
        let below = 0;
        let outside = 0;
        const draws = 50_000;
        for (let did = 0; did < draws; did++) {
            const value = rand(makeRandom(did, 1234));
            if (!(value >= 0 && value < 1)) outside++;
            if (value <= Math.fround(0.7)) below++;
        }
        expect(outside).toBe(0);
        // Binomial standard deviation sqrt(0.21 / 50k) = 0.002: five of them.
        expect(Math.abs(below / draws - 0.7)).toBeLessThan(0.01);
    });
});

describe('the second-bounce direction (sampleBRDFAndEvaluateDiffuse, Lambert)', () => {
    const normals: Vec3[] = [
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1], // QuaternionRotationZ's special case
        [1, 0, 0],
        [0.48, -0.6, 0.64],
    ];

    it('QuaternionRotationZ takes the normal to +z, and its inverse takes +z back', () => {
        for (const normal of normals) {
            const rotation = quaternionRotationZ(normal);
            const toZ = quaternionTransform(rotation, normal);
            const back = quaternionTransform(quaternionInverse(rotation), [0, 0, 1]);
            // The special case is a half turn about x: it takes -z to +z exactly.
            expect(toZ[2]).toBeCloseTo(1, 5);
            back.forEach((value, axis) => expect(value).toBeCloseTo(normal[axis], 5));
        }
    });

    it('mapToCosineHemisphere gives unit directions with z = sqrt(1 - u)', () => {
        for (const samples of [
            [0, 0],
            [0.25, 0.5],
            [0.999, 0.1],
            [1 - 2 ** -24, 0.75],
        ] as [number, number][]) {
            const direction = mapToCosineHemisphere(samples);
            expect(length(direction)).toBeCloseTo(1, 6);
            expect(direction[2]).toBeCloseTo(Math.sqrt(1 - samples[0]), 6);
        }
    });

    it('samples the hemisphere about the normal, cosine-weighted, with BRDF / pdf equal to the albedo', () => {
        const albedo: Vec3 = [0.5, 0.25, 0.8];
        for (const normal of normals) {
            let cosineSum = 0;
            let worst = 0;
            let belowSurface = 0;
            const count = 5_000;
            for (let did = 0; did < count; did++) {
                const random = makeRandom(did, 7);
                const sample = sampleBRDFAndEvaluateDiffuse(albedo, [rand(random), rand(random)], normal);
                const cosine = dot(sample.direction, normal);
                if (!(cosine > 0)) belowSurface++;
                worst = Math.max(
                    worst,
                    Math.abs(length(sample.direction) - 1),
                    Math.abs(sample.pdf - cosine / Math.PI) / (cosine / Math.PI),
                    ...sample.brdf.map((value, channel) => Math.abs(value / sample.pdf - albedo[channel]) / albedo[channel]),
                );
                cosineSum += cosine;
            }
            expect(belowSurface).toBe(0);
            // f32 steps: well within 1e-4 (relative) except where the cosine is tiny.
            expect(worst).toBeLessThan(1e-3);
            // A cosine-weighted hemisphere has a mean cosine of 2/3 (standard deviation of the mean about 0.0033 here).
            expect(Math.abs(cosineSum / count - 2 / 3)).toBeLessThan(0.015);
        }
    });
});

describe('the multibounce sample and its settled value', () => {
    it('discards with probability 0.7 and draws the direction from the next two numbers', () => {
        let kept = 0;
        let mismatches = 0;
        const draws = 50_000;
        for (let did = 0; did < draws; did++) {
            const draw = multibounceDraw(did, 99, 0.7);
            const random = makeRandom(did, 99);
            const first = rand(random);
            if (draw.discarded !== first <= Math.fround(0.7)) mismatches++;
            if (draw.discarded) continue;
            kept++;
            const samples = [rand(random), rand(random)];
            if (draw.samples[0] !== samples[0] || draw.samples[1] !== samples[1]) mismatches++;
        }
        expect(mismatches).toBe(0);
        expect(Math.abs(kept / draws - 0.3)).toBeLessThan(0.01);
    });

    it("weighs the second hit's light by BRDF / pdf, with no 1 / (1 - p); a miss is a sample of no light (register D37)", () => {
        const sample = sampleBRDFAndEvaluateDiffuse([0.5, 0.5, 0.5], [0.3, 0.6], [0, -1, 0]);
        const value = multibounceSample([2, 1, 0.5], sample.brdf, sample.pdf);
        [2, 1, 0.5].forEach((direct, channel) => expect(value[channel]).toBeCloseTo(0.5 * direct, 4));
        expect(multibounceSample(null, sample.brdf, sample.pdf)).toEqual([0, 0, 0]);
        // A bypassing first bounce's probe ray takes the draw over 1 - p.
        multibounceBypassAddition(value, 0.7).forEach((added, channel) => expect(added).toBeCloseTo(value[channel] / 0.3, 4));
    });

    it('settles at the second bounce, albedo x L x P(hit), whatever the discard probability (register D37)', () => {
        // One first-bounce cell over many draws: a kept ray hits (a surface of light L) or misses
        // with probability 1/2, and every kept ray counts. The probe ray of a bypassing first
        // bounce averages its own draws, discarded ones giving 0.
        const albedo = 0.5;
        const secondDirect = 0.6366;
        const draws = 40_000;
        for (const discard of [0.7, 0.3]) {
            let sum = 0;
            let count = 0;
            let bypassSum = 0;
            for (let did = 0; did < draws; did++) {
                const draw = multibounceDraw(did, 5, discard);
                if (draw.discarded) continue;
                const hits = (pcgHash(did ^ 0x5bd1e995) & 1) === 0;
                const sample = sampleBRDFAndEvaluateDiffuse([albedo, albedo, albedo], draw.samples, [0, -1, 0]);
                const value = multibounceSample(hits ? [secondDirect, secondDirect, secondDirect] : null, sample.brdf, sample.pdf);
                sum += value[0];
                count++;
                bypassSum += multibounceBypassAddition(value, discard)[0];
            }
            const expected = multibounceSettledIndirect(albedo, secondDirect, 0.5);
            // Binomial on the hit share: standard deviation of the mean about 0.5 % here.
            expect(Math.abs(sum / count / expected - 1)).toBeLessThan(0.03);
            expect(Math.abs(bypassSum / draws / expected - 1)).toBeLessThan(0.05);
        }
        expect(multibounceSettledIndirect(albedo, secondDirect)).toBeCloseTo(albedo * secondDirect, 10);
    });
});

describe("a sparse cell's indirect value over frames (UpdateTiles, register D37)", () => {
    /**
     * One cell with `entries` first-bounce entries a frame, each kept with probability 0.3 and
     * adding a sample of `mu` (one entry a frame is a cell its probes rarely reach). The reading
     * each frame is the cell's mean, 0 at a count of 0, as ResolveCells divides it; averaged
     * over frames 100 to 4,100.
     */
    function meanReading(entries: number, blend: (stored: CellValue, samples: CellValue, cap: number) => CellValue): number {
        const mu = 0.5;
        let value: CellValue = [0, 0, 0, 0];
        let sum = 0;
        for (let frame = 0; frame < 4100; frame++) {
            let kept = 0;
            for (let entry = 0; entry < entries; entry++) if (!multibounceDraw(frame * 16 + entry, 3, 0.7).discarded) kept++;
            value = blend(value, [mu * kept, mu * kept, mu * kept, kept], 16);
            if (frame >= 100) sum += value[3] > 0 ? value[0] / value[3] : 0;
        }
        return sum / 4000 / mu;
    }

    it("reads the samples' mean at one, two and three entries a frame, where Capsaicin's step-down reads low at one", () => {
        for (const entries of [1, 2, 3]) expect(meanReading(entries, updateTilesAccumulateIndirect)).toBeCloseTo(1, 3);
        // Capsaicin's blend: a frame with no sample drops the count by one, and a count of 0 reads 0.
        expect(meanReading(1, updateTilesAccumulate)).toBeLessThan(0.6);
        expect(meanReading(3, updateTilesAccumulate)).toBeGreaterThan(0.97);
    });
});

describe('the multibounce kernels and lists', () => {
    const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS), true);

    it("keep room for a second-bounce entry's BRDF, pdf and ray per probe ray", () => {
        const maxRayCount = 230_400;
        const lists = hashGridCacheRayListLayout(maxRayCount, 22282240);
        expect(lists.multibounceInfoBuffer).toBe(lists.reservoirBuffer + RESERVOIR_WORDS * 2 * maxRayCount);
        expect(lists.multibounceRayBuffer).toBe(lists.multibounceInfoBuffer + MULTIBOUNCE_INFO_WORDS * maxRayCount);
        expect(lists.firstBounceBypassBuffer).toBe(lists.multibounceRayBuffer + MULTIBOUNCE_RAY_WORDS * maxRayCount);
        expect(lists.multibounceDirectBuffer).toBe(lists.firstBounceBypassBuffer + maxRayCount);
        expect(lists.bytes).toBe((lists.multibounceDirectBuffer + MULTIBOUNCE_DIRECT_WORDS * maxRayCount) * 4);
        // 32 words a probe ray: 29,491,200 bytes at 1280 x 720.
        expect(lists.bytes).toBe(32 * maxRayCount * 4);
    });

    it('declare their entry points, read the multibounce constants and the counters Capsaicin dispatches on', () => {
        const populate = populateMultibounceCellsWgsl(layout);
        expect(populate).toContain(`fn ${POPULATION_ENTRY_POINT.populateMultibounceCells}(`);
        expect(populate).toContain('discard_multibounce_ray_probability');
        expect(populate).toContain('HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1');
        expect(populate).toContain('giTraceClosest');
        const reservoirs = generateMultibounceReservoirsWgsl(layout);
        expect(reservoirs).toContain(`fn ${POPULATION_ENTRY_POINT.generateMultibounceReservoirs}(`);
        expect(reservoirs).toContain('giDirectChooseLight(surface, HashGridCache_LightMask(), rcRand(&random))');
        expect(generateReservoirsWgsl(layout)).toContain('giDirectChooseLight(surface, HashGridCache_LightMask(), rcRand(&random))');
        expect(reservoirs).toContain('select(0u, GI_DIRECT_SKY_LIGHT_BIT, g_HashGridCacheConstants.environment_light_enable != 0u)');
        const update = updateMultibounceCellsWgsl(layout);
        expect(update).toContain(`fn ${POPULATION_ENTRY_POINT.updateMultibounceCells}(`);
        expect(update).toContain('HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER');
        expect(update).toContain('1.0 - g_HashGridCacheConstants.discard_multibounce_ray_probability');
    });

    it('put use_multibounce in the constant block at the word the TypeScript writes', () => {
        const code = hashGridCacheWgsl(1, layout);
        const body = /struct HashGridCacheConstants \{([^}]*)\}/.exec(code)?.[1] ?? '';
        const fields = body
            .split('\n')
            .map((line) => line.replace(/\/\/.*$/, '').trim())
            .filter((line) => line.length > 0)
            .map((line) => line.split(':')[0]);
        // eye_position is a vec3f (three words); every other field one word.
        expect(fields.indexOf('use_multibounce') + 2).toBe(HASH_GRID_CONSTANTS_WORD.useMultibounce);
        // RC3.4: the sky among a hit's lights (word 31); P6.4's use_neural_tail after it, then padding to 144 bytes.
        expect(fields.indexOf('environment_light_enable') + 2).toBe(HASH_GRID_CONSTANTS_WORD.environmentLightEnable);
        expect(fields.indexOf('use_neural_tail') + 2).toBe(HASH_GRID_CONSTANTS_WORD.useNeuralTail);
        expect(fields.length + 2).toBe(HASH_GRID_CONSTANTS_BYTES / 4);
    });
});
