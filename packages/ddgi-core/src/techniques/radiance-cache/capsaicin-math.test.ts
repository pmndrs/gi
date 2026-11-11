import { describe, expect, it } from 'vitest';
import {
    calculateHaltonNumber,
    calculateHaltonSequence,
    floatToHalfBits,
    getOrthoVectors,
    halfBitsToFloat,
    INVALID_ID,
    makeRandom,
    mapToHemiOctahedron,
    mapToHemiOctahedronInverse,
    nearestHalf,
    offsetPosition,
    packHalf3,
    packSeed,
    probeCellOf,
    quantizeRadiance,
    rand,
    rand2,
    sampleBRDFAndEvaluateDiffuse,
    spawnJitter,
    spawnScreenProbes,
    spawnTileCounts,
    toProbeFrame,
    unpackHalf3,
    unpackSeed,
    type Direction,
} from './capsaicin-math.ts';
import { PROBE_SIZE, PROBE_SPAWN_TILE_SIZE } from './settings.ts';

const dot = (a: Direction, b: Direction): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const length = (a: Direction): number => Math.sqrt(dot(a, a));

/** A spread of unit normals: the axes (both signs), diagonals, and points of a Fibonacci sphere. */
function testNormals(): Direction[] {
    const normals: Direction[] = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [Math.SQRT1_2, Math.SQRT1_2, 0],
        [0, Math.SQRT1_2, -Math.SQRT1_2],
    ];
    const count = 64;
    for (let index = 0; index < count; index++) {
        const z = 1 - (2 * (index + 0.5)) / count;
        const radius = Math.sqrt(1 - z * z);
        const angle = index * 2.399963229728653;
        normals.push([radius * Math.cos(angle), radius * Math.sin(angle), z]);
    }
    return normals;
}

describe('Halton (gi1.hlsl)', () => {
    it('gives the radical inverse in bases 2 and 3', () => {
        expect([1, 2, 3, 4, 5].map((index) => calculateHaltonNumber(index, 2))).toEqual([0.5, 0.25, 0.75, 0.125, 0.625]);
        const base3 = [1, 2, 3, 4].map((index) => calculateHaltonNumber(index, 3));
        [1 / 3, 2 / 3, 1 / 9, 4 / 9].forEach((expected, index) => expect(base3[index]).toBeCloseTo(expected, 6));
        expect(calculateHaltonNumber(0, 2)).toBe(0);
    });

    it('starts at index 1 and repeats every 256 frames', () => {
        expect(calculateHaltonSequence(0)).toEqual([0.5, calculateHaltonNumber(1, 3)]);
        expect(calculateHaltonSequence(256)).toEqual(calculateHaltonSequence(0));
        expect(calculateHaltonSequence(255 + 256 * 7)).toEqual([calculateHaltonNumber(256, 2), calculateHaltonNumber(256, 3)]);
    });

    it('keeps the jitter inside the 16-pixel spawn tile for every frame of the cycle', () => {
        const seen = new Set<string>();
        for (let frame = 0; frame < 256; frame++) {
            const [x, y] = spawnJitter(frame);
            expect(Number.isInteger(x) && Number.isInteger(y)).toBe(true);
            expect(x >= 0 && x < PROBE_SPAWN_TILE_SIZE && y >= 0 && y < PROBE_SPAWN_TILE_SIZE).toBe(true);
            seen.add(`${x},${y}`);
        }
        // A low-discrepancy sequence visits most of the tile's 256 pixels in 256 frames.
        expect(seen.size).toBeGreaterThan(200);
    });
});

describe('the hemi-octahedral map (sampling.hlsl)', () => {
    it('maps the square onto the +z hemisphere as unit directions', () => {
        for (let u = 0; u <= 1; u += 1 / 32) {
            for (let v = 0; v <= 1; v += 1 / 32) {
                const direction = mapToHemiOctahedron(u, v);
                expect(length(direction)).toBeCloseTo(1, 6);
                expect(direction[2]).toBeGreaterThanOrEqual(-1e-9);
            }
        }
        const centre = mapToHemiOctahedron(0.5, 0.5);
        expect(centre[2]).toBeCloseTo(1, 9);
    });

    it('is inverted by mapToHemiOctahedronInverse', () => {
        for (let u = 0.01; u < 1; u += 0.07) {
            for (let v = 0.01; v < 1; v += 0.07) {
                const [backU, backV] = mapToHemiOctahedronInverse(mapToHemiOctahedron(u, v));
                expect(backU).toBeCloseTo(u, 6);
                expect(backV).toBeCloseTo(v, 6);
            }
        }
    });

    it('maps a direction just longer than 1 near the pole to the pole, not NaN (register D34)', () => {
        // Two probe rays read back on the Apple M5 (radiance-cache-probes, 2026-11-10), whose
        // pack2x16float rounds to nearest: the halves are about 1e-4 longer than 1, so the
        // probe-frame z is past 1. Capsaicin's sqrt(1 - |z|) is NaN there.
        const normalize = (a: Direction): Direction => [a[0] / length(a), a[1] / length(a), a[2] / length(a)];
        const sponza = { words: [3002382593, 47615] as const, normal: normalize([-0.63134765625, -0.22412109375, -0.74169921875]) };
        const cornell = { words: [2618307801, 15264] as const, normal: normalize([0.292236328125, 0, 0.9560546875]) };
        for (const ray of [sponza, cornell]) {
            const direction = unpackHalf3(ray.words);
            expect(length(direction)).toBeGreaterThan(1.0001);
            expect(toProbeFrame(direction, ray.normal)[2]).toBeGreaterThan(1);
            expect(mapToHemiOctahedronInverse(toProbeFrame(direction, ray.normal))).toEqual([0.5, 0.5]);
            // Sampled in cells 36 and 28; the pole is the corner cells 27, 28, 35 and 36 share, and cell 36 holds it.
            expect(probeCellOf(direction, ray.normal)).toBe(36);
        }
        expect(mapToHemiOctahedronInverse([0, 0, 1.0002])).toEqual([0.5, 0.5]);
    });

    it('gives every cell the same solid angle (equal area): uniform points have mean cos 1/2', () => {
        // Uniform over the hemisphere's solid angle: the mean of z is 1/2, of z^2 is 1/3.
        const steps = 400;
        let sumZ = 0;
        let sumZ2 = 0;
        const perCell = new Array<number>(PROBE_SIZE * PROBE_SIZE).fill(0);
        for (let i = 0; i < steps; i++) {
            for (let j = 0; j < steps; j++) {
                const [x, y, z] = mapToHemiOctahedron((i + 0.5) / steps, (j + 0.5) / steps);
                sumZ += z;
                sumZ2 += z * z;
                perCell[probeCellOf([x, y, z], [0, 0, 1])]++;
            }
        }
        expect(sumZ / steps ** 2).toBeCloseTo(0.5, 3);
        expect(sumZ2 / steps ** 2).toBeCloseTo(1 / 3, 3);
        // Each cell of the square gets the points that map back into it.
        for (const count of perCell) expect(count).toBe(steps ** 2 / perCell.length);
    });
});

describe('GetOrthoVectors and the probe frame (sampling.hlsl)', () => {
    it('completes any normal to an orthonormal basis', () => {
        for (const normal of testNormals()) {
            const { b1, b2 } = getOrthoVectors(normal);
            expect(length(b1)).toBeCloseTo(1, 9);
            expect(length(b2)).toBeCloseTo(1, 9);
            expect(dot(b1, b2)).toBeCloseTo(0, 9);
            expect(dot(b1, normal)).toBeCloseTo(0, 9);
            expect(dot(b2, normal)).toBeCloseTo(0, 9);
        }
    });

    it('puts each cell centre in its own cell, on the normal side, for any normal', () => {
        for (const normal of testNormals()) {
            const { b1, b2 } = getOrthoVectors(normal);
            for (let cell = 0; cell < PROBE_SIZE * PROBE_SIZE; cell++) {
                const local = mapToHemiOctahedron(((cell % PROBE_SIZE) + 0.5) / PROBE_SIZE, (Math.floor(cell / PROBE_SIZE) + 0.5) / PROBE_SIZE);
                const world: Direction = [0, 1, 2].map((axis) => local[0] * b1[axis] + local[1] * b2[axis] + local[2] * normal[axis]) as unknown as Direction;
                expect(dot(world, normal)).toBeGreaterThan(0);
                expect(probeCellOf(world, normal)).toBe(cell);
                const back = toProbeFrame(world, normal);
                back.forEach((value, axis) => expect(value).toBeCloseTo(local[axis], 9));
            }
        }
    });
});

describe('spawning (SpawnScreenProbes, gi1.comp:555-579)', () => {
    it('places one candidate per 16 x 16 tile at the frame jitter, clamped to the screen', () => {
        const width = 1000;
        const height = 700;
        const frame = 5;
        const tiles = spawnScreenProbes(width, height, frame, () => false);
        const [tilesX, tilesY] = spawnTileCounts(width, height);
        expect([tilesX, tilesY]).toEqual([63, 44]);
        expect(tiles.length).toBe(tilesX * tilesY);
        const [jx, jy] = spawnJitter(frame);
        tiles.forEach((tile, index) => {
            const tileX = index % tilesX;
            const tileY = Math.floor(index / tilesX);
            expect(tile.seed).toEqual([Math.min(tileX * 16 + jx, width - 1), Math.min(tileY * 16 + jy, height - 1)]);
            expect(tile.spawned).toBe(true);
            // The seed stays in its own tile, also at the clamped edge.
            expect(Math.floor(tile.seed[0] / 16)).toBe(tileX);
            expect(Math.floor(tile.seed[1] / 16)).toBe(tileY);
        });
    });

    it('spawns nothing on sky pixels', () => {
        // Sky above row 300.
        const tiles = spawnScreenProbes(1280, 720, 0, (_x, y) => y < 300);
        for (const tile of tiles) expect(tile.spawned).toBe(tile.seed[1] >= 300);
        expect(tiles.filter((tile) => tile.spawned).length).toBeGreaterThan(0);
        expect(tiles.filter((tile) => !tile.spawned).length).toBeGreaterThan(0);
    });
});

describe('packing (screen_probes.hlsl, pack.hlsl)', () => {
    it('packs a seed as x << 16 | y', () => {
        expect(packSeed(3, 5)).toBe(0x00030005);
        expect(unpackSeed(packSeed(65535, 0))).toEqual([65535, 0]);
        expect(unpackSeed(packSeed(1279, 719))).toEqual([1279, 719]);
        // The largest seed is still distinct from "no probe".
        expect(packSeed(65535, 65534)).not.toBe(INVALID_ID);
    });

    it('converts halves as f32tof16 does: round to nearest even, exact for halves', () => {
        expect(floatToHalfBits(1)).toBe(0x3c00);
        expect(floatToHalfBits(-2)).toBe(0xc000);
        expect(floatToHalfBits(65504)).toBe(0x7bff);
        expect(floatToHalfBits(1e6)).toBe(0x7c00);
        expect(floatToHalfBits(2 ** -24)).toBe(0x0001);
        // 1 + 2^-11 lies halfway between 1 and the next half: rounds to even (1).
        expect(floatToHalfBits(1 + 2 ** -11)).toBe(0x3c00);
        expect(floatToHalfBits(1 + 3 * 2 ** -11)).toBe(0x3c02);
        for (const bits of [0x0001, 0x03ff, 0x0400, 0x3555, 0x3c00, 0x7bff, 0x8001, 0xbc00]) {
            expect(floatToHalfBits(halfBitsToFloat(bits))).toBe(bits);
        }
    });

    it('rounds to the nearest half before a store, which a store that rounds to nearest leaves alone (register D59)', () => {
        const towardZero = (value: number): number => {
            const nearest = floatToHalfBits(value);
            return Math.abs(halfBitsToFloat(nearest)) > Math.abs(Math.fround(value)) ? nearest - 1 : nearest;
        };
        const values: number[] = [0, 1, -2, 40, 39.99999, 65504, 65519, -65519, 2 ** -24, 2 ** -25, 3 * 2 ** -25, 2 ** -14 * (1 - 2 ** -11), 1 + 2 ** -11, 1 + 3 * 2 ** -11, 1e-30];
        let state = 7;
        for (let index = 0; index < 20000; index++) {
            state = (Math.imul(state, 1103515245) + 12345) >>> 0;
            const exponent = (state % 40) - 25;
            state = (Math.imul(state, 1103515245) + 12345) >>> 0;
            // Every fourth a tie between two halves; the sign alternates.
            const mantissa = index % 4 === 0 ? (Math.floor((state / 2 ** 32) * 1024) + 0.5) / 1024 : state / 2 ** 32;
            values.push((index % 2 === 0 ? 1 : -1) * (1 + mantissa) * 2 ** exponent);
        }
        for (const value of values) {
            const rounded = nearestHalf(value);
            // It is the nearest half (ties to even), so it is exactly a half...
            expect(rounded).toBe(halfBitsToFloat(floatToHalfBits(value)));
            // ...which a nearest-rounding store keeps, giving what it gave without the rounding...
            expect(floatToHalfBits(rounded)).toBe(floatToHalfBits(value));
            // ...and a store that rounds toward zero keeps too.
            expect(towardZero(rounded)).toBe(floatToHalfBits(value));
        }
    });

    it('pushes a point off its surface as offsetPosition does (geometry.hlsl:35-47, register D39)', () => {
        const bits = (value: number): number => new Int32Array(new Float32Array([value]).buffer)[0];
        // Along +z at z = 3: z steps 256 ulps up on its bits; x and y stay (their normal component truncates to 0).
        const up = offsetPosition([1, 2, 3], [0, 0, 1]);
        expect(up[0]).toBe(1);
        expect(up[1]).toBe(2);
        expect(bits(up[2]) - bits(3)).toBe(256);
        // A negative coordinate steps further from zero for a normal pointing away from zero.
        const down = offsetPosition([1, 2, -3], [0, 0, -1]);
        expect(bits(down[2]) - bits(-3)).toBe(256);
        expect(down[2]).toBeLessThan(-3);
        // Within 1/32 of the origin: a plain n / 65536 push.
        const nearOrigin = offsetPosition([0.01, 0, 5], [0.6, 0, 0.8]);
        expect(nearOrigin[0]).toBe(Math.fround(Math.fround(0.01) + Math.fround(0.6) / 65536));
        expect(nearOrigin[1]).toBe(0);
        // 0.8 x 256 = 204.8 truncates to 204 steps.
        expect(bits(nearOrigin[2]) - bits(5)).toBe(204);
    });

    it('packs a direction as three halves', () => {
        const direction: Direction = [0.6, -0.48, 0.64];
        const packed = packHalf3(direction);
        expect(packed[1] >>> 16).toBe(0);
        unpackHalf3(packed).forEach((value, axis) => expect(value).toBeCloseTo(direction[axis], 3));
    });

    it('quantizes radiance by 10^4, to the nearest integer', () => {
        expect(quantizeRadiance(1)).toBe(10000);
        expect(quantizeRadiance(0.00005)).toBe(0);
        expect(quantizeRadiance(0.00006)).toBe(1);
        expect(quantizeRadiance(0.12344)).toBe(1234);
    });
});

describe('MakeRandom and rand (random_number_generator.hlsl:38-116, seeded as register D18)', () => {
    /** The WGSL's rcMakeRandom and rcRand written out in BigInt, mod 2^32 at every step. */
    function referenceRandoms(seed: number, index: number, count: number): number[] {
        const mask = 0xffffffffn;
        const pcg = (value: bigint): bigint => {
            const state = (value * 747796405n + 2891336453n) & mask;
            const word = (((state >> ((state >> 28n) + 4n)) ^ state) * 277803737n) & mask;
            return (word >> 22n) ^ word;
        };
        const increment = ((BigInt(index) << 1n) | 1n) & mask;
        let state = (((pcg(BigInt(seed)) + increment) & mask) * 747796405n + increment) & mask;
        const out: number[] = [];
        for (let n = 0; n < count; n++) {
            const word = (((state >> ((state >> 28n) + 4n)) ^ state) * 277803737n) & mask;
            state = (state * 747796405n + 2891336453n) & mask;
            out.push(Number(((word >> 22n) ^ word) >> 8n) / 16777216);
        }
        return out;
    }

    it('gives the BigInt reference, three draws deep, for several seeds and frames', () => {
        for (const [seed, frame] of [[0, 0], [1, 7], [123456, 4000000000], [0xffffffff, 1]]) {
            const random = makeRandom(seed, frame);
            const first = rand(random);
            const [second, third] = rand2(random);
            expect([first, second, third]).toEqual(referenceRandoms(seed, frame, 3));
            for (const value of [first, second, third]) expect(value >= 0 && value < 1).toBe(true);
        }
    });
});

describe('sampleBRDFAndEvaluateDiffuse (material_sampling.hlsl:741-773), Lambert', () => {
    const albedo: Direction = [0.5, 0.25, 0.75];

    it('samples about +z unturned: the cosine-hemisphere point itself', () => {
        const sample = sampleBRDFAndEvaluateDiffuse(albedo, [0, 0, 1], [0.36, 0.25]);
        expect(sample.direction[0]).toBeCloseTo(0, 12);
        expect(sample.direction[1]).toBeCloseTo(0.6, 12);
        expect(sample.direction[2]).toBeCloseTo(0.8, 12);
        expect(sample.pdf).toBeCloseTo(0.8 / Math.PI, 12);
    });

    it('keeps the cosine to the normal at sqrt(1 - u) for every normal, -z (the special case) included', () => {
        for (const normal of testNormals()) {
            for (const samples of [[0, 0], [0.5, 0.3], [0.99, 0.9], [0.2, 0.75]] as const) {
                const sample = sampleBRDFAndEvaluateDiffuse(albedo, normal, samples);
                expect(length(sample.direction)).toBeCloseTo(1, 12);
                expect(dot(sample.direction, normal)).toBeCloseTo(Math.sqrt(1 - samples[0]), 4);
            }
        }
    });

    it('has brdf / pdf = albedo, the estimator UpdateMultibounceCells takes', () => {
        const sample = sampleBRDFAndEvaluateDiffuse(albedo, [0, 1, 0], [0.7, 0.1]);
        sample.brdf.forEach((value, channel) => expect(value / sample.pdf).toBeCloseTo(albedo[channel], 12));
    });
});
