import { describe, expect, it } from 'vitest';
import { RadianceCacheTechnique } from '../../radiance-cache-technique.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS } from '../../settings.ts';
import { SharcCache } from './sharc-cache.ts';
import {
    sharcCellSize,
    sharcCellSizePerMetre,
    sharcFurnaceCeiling,
    sharcFurnaceFloor,
    sharcKey,
    sharcLevel,
    sharcMix,
    sharcMix2,
    sharcNormalBin,
    sharcResolveStep,
    sharcSettledWeight,
} from './sharc-cpu.ts';
import { DEFAULT_SHARC_SETTINGS, resolveSharcSettings } from './sharc-settings.ts';
import { SHARC_BUCKET, SHARC_COUNTER, SHARC_COUNTER_WORDS, SHARC_ENTRY_POINT, sharcLookupFunctionWgsl, sharcLookupWgsl, sharcResolveWgsl, sharcUpdateWgsl } from './sharc-wgsl.ts';

// ---- Independent references: the hashes in BigInt arithmetic, mod 2^32 at every step ----

const MASK = 0xffffffffn;

function pcgBigInt(value: number): number {
    const state = (BigInt(value) * 747796405n + 2891336453n) & MASK;
    const word = (((state >> ((state >> 28n) + 4n)) ^ state) * 277803737n) & MASK;
    return Number((word >> 22n) ^ word);
}

function murmurFinaliserBigInt(value: number): number {
    let h = (BigInt(value) * 0x85ebca6bn) & MASK;
    h ^= h >> 13n;
    h = (h * 0xc2b2ae35n) & MASK;
    h ^= h >> 16n;
    return Number(h);
}

/** Balanced braces and brackets: a cheap check that template interpolation didn't break the WGSL. */
function balanced(code: string): boolean {
    let depth = 0;
    for (const character of code) {
        if (character === '{' || character === '(' || character === '[') depth++;
        if (character === '}' || character === ')' || character === ']') depth--;
        if (depth < 0) return false;
    }
    return depth === 0;
}

describe('SHaRC hash cache: the cell (CPU twins)', () => {
    it('hashes as the references do, across the u32 range', () => {
        for (const value of [0, 1, 7, 0x9e3779b9, 0x7fffffff, 0x80000000, 0xffffffff, 123456789]) {
            expect(sharcMix(value)).toBe(pcgBigInt(value));
            expect(sharcMix2(value)).toBe(murmurFinaliserBigInt(value));
        }
    });

    it('sizes cells by distance: about cellPixels pixels, never below minCell, one level per doubling', () => {
        const perMetre = sharcCellSizePerMetre(Math.PI / 3, 720, 16);
        expect(perMetre).toBeCloseTo(((2 * Math.tan(Math.PI / 6)) / 720) * 16, 12);
        const constants = { eye: [0, 0, 0] as const, cellSizePerMetre: perMetre, minCell: 0.02 };
        // Close up the smallest cell wins: level 0.
        expect(sharcLevel([0, 0, 0.5], constants)).toBe(0);
        // Far away: a cell of about d x perMetre, rounded down to a power of two of minCell.
        for (const distance of [5, 20, 80]) {
            const level = sharcLevel([0, 0, distance], constants);
            const size = sharcCellSize(level, 0.02);
            expect(size).toBeLessThanOrEqual(distance * perMetre * 1.0001);
            expect(size * 2).toBeGreaterThan(distance * perMetre);
        }
        expect(sharcLevel([0, 0, 40], constants)).toBe(sharcLevel([0, 0, 20], constants) + 1);
    });

    it('bins normals by dominant axis and sign, or not at all', () => {
        expect([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].map((n) => sharcNormalBin(n as [number, number, number]))).toEqual([1, 2, 3, 4, 5, 6]);
        expect(sharcNormalBin([0.6, 0.8, 0], true)).toBe(3);
        expect(sharcNormalBin([0, -1, 0], false)).toBe(0);
    });

    it('keys a cell to a bucket start and a non-zero checksum that change with level, bin and position', () => {
        const key = sharcKey([1.234, 0.5, -3.2], 3, 2, 0.02, 20);
        expect(key[0] % SHARC_BUCKET).toBe(0);
        expect(key[0]).toBeLessThan(2 ** 20);
        expect(key[1]).not.toBe(0);
        expect(sharcKey([1.234, 0.5, -3.2], 3, 3, 0.02, 20)).not.toEqual(key);
        expect(sharcKey([1.234, 0.5, -3.2], 4, 2, 0.02, 20)).not.toEqual(key);
        // Same cell (size 0.08 at level 2): same key; the next cell over: another.
        expect(sharcKey([1.21, 0.49, -3.19], 3, 2, 0.02, 20)).toEqual(key);
        expect(sharcKey([1.234 + 0.08, 0.5, -3.2], 3, 2, 0.02, 20)).not.toEqual(key);
    });
});

describe('SHaRC hash cache: the resolve (CPU twin)', () => {
    it('keeps the exact mean of equal samples and settles at n x historyFrames of weight', () => {
        let cell = { radiance: 0, weight: 0 };
        for (let frame = 0; frame < 1000; frame++) cell = sharcResolveStep(cell, 4 * 0.75, 4, 32);
        expect(cell.radiance).toBeCloseTo(0.75, 12);
        expect(cell.weight).toBeCloseTo(sharcSettledWeight(4, 32), 6);
    });

    it('moves toward a new value with a time constant of historyFrames frames, and ages only with samples', () => {
        let cell = { radiance: 1, weight: sharcSettledWeight(1, 32) };
        cell = sharcResolveStep(cell, 0, 0, 32);
        expect(cell).toEqual({ radiance: 1, weight: 32 });
        for (let frame = 0; frame < 32; frame++) cell = sharcResolveStep(cell, 0, 1, 32);
        // (1 - 1/32)^32 = 0.362: about 1/e left after one time constant.
        expect(cell.radiance).toBeCloseTo((1 - 1 / 32) ** 32, 6);
    });
});

describe('SHaRC hash cache: the furnace answers', () => {
    const albedo = 0.5;
    const sun = 4;
    it('reads GI-1.2s second bounce with one bounce ray and no cache tail, and every bounce with it', () => {
        const floorDirect = (albedo * sun) / Math.PI;
        expect(sharcFurnaceCeiling(albedo, sun, 'direct')).toBe(0);
        expect(sharcFurnaceCeiling(albedo, sun, 'one-bounce')).toBeCloseTo(albedo * floorDirect, 12);
        expect(sharcFurnaceCeiling(albedo, sun, 'one-bounce', 0.5)).toBeCloseTo(0.5 * albedo * floorDirect, 12);
        // L_c = rho L_f, L_f = rho E / pi + rho L_c.
        const ceiling = sharcFurnaceCeiling(albedo, sun, 'every-bounce');
        const floor = sharcFurnaceFloor(albedo, sun);
        expect(ceiling).toBeCloseTo(albedo * floor, 12);
        expect(floor).toBeCloseTo(floorDirect + albedo * ceiling, 12);
        expect(ceiling).toBeCloseTo(0.42441, 5);
        // Over the strips half of the ceiling's rays find the floor.
        const stripsCeiling = sharcFurnaceCeiling(albedo, sun, 'every-bounce', 0.5);
        expect(stripsCeiling).toBeCloseTo(0.5 * albedo * sharcFurnaceFloor(albedo, sun, 0.5), 12);
    });
});

describe('SHaRC hash cache: settings', () => {
    it('defaults to the pathtracer tuned cell and table, and the two-bounce cache-tail path', () => {
        expect(resolveSharcSettings()).toEqual({ ...DEFAULT_SHARC_SETTINGS });
        expect(DEFAULT_SHARC_SETTINGS).toMatchObject({ log2Entries: 20, cellPixels: 16, minSamples: 64, historyFrames: 32, staleFrames: 64, fallbackLevels: 3, pyramid: true, bounceRays: 2, cacheTail: true, updateFraction: 0.25 });
    });

    it('names the setting out of range', () => {
        expect(() => resolveSharcSettings({ bounceRays: 3 })).toThrow(/sharc.bounceRays/);
        expect(() => resolveSharcSettings({ log2Entries: 30 })).toThrow(/sharc.log2Entries/);
        expect(() => resolveSharcSettings({ minCell: 0 })).toThrow(/sharc.minCell/);
        expect(() => resolveSharcSettings({ updateFraction: 0 })).toThrow(/sharc.updateFraction/);
    });

    it('keeps the GI-1.2 hash grid the technique default, and copies the SHaRC overrides', () => {
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS.worldCache).toBe('gi1-hash-grid');
        const overrides = { bounceRays: 1 };
        const technique = new RadianceCacheTechnique({ worldCache: 'sharc', sharc: overrides });
        expect(technique.settings.worldCache).toBe('sharc');
        technique.settings.sharc.bounceRays = 0;
        expect(overrides.bounceRays).toBe(1);
        expect(new RadianceCacheTechnique().settings.sharc).toEqual({});
        expect(new SharcCache().id).toBe('sharc');
    });
});

describe('SHaRC hash cache: WGSL', () => {
    const options = { fallbackLevels: 3, pyramid: true };
    it('generates balanced code with each entry point and the counters past the header', () => {
        const kernels = { update: sharcUpdateWgsl(options), resolve: sharcResolveWgsl(options), lookup: sharcLookupWgsl(options), reader: sharcLookupFunctionWgsl(3, options) };
        for (const [name, code] of Object.entries(kernels)) expect(balanced(code), name).toBe(true);
        expect(kernels.update).toContain(`fn ${SHARC_ENTRY_POINT.update}(`);
        expect(kernels.resolve).toContain(`fn ${SHARC_ENTRY_POINT.resolve}(`);
        expect(kernels.lookup).toContain(`fn ${SHARC_ENTRY_POINT.lookup}(`);
        expect(kernels.reader).toContain('fn worldCacheLookup(position: vec3f, normal: vec3f, direction: vec3f) -> vec4f');
        expect(kernels.reader).toContain('@group(3)');
        expect(kernels.reader).not.toMatch(/atomic<|atomicLoad/);
        expect(Math.max(...Object.values(SHARC_COUNTER))).toBeLessThan(SHARC_COUNTER_WORDS);
    });

    it('bakes the fallback and pyramid levels and the bucket loops in as constants (Tint loop counters on D3D12)', () => {
        expect(sharcUpdateWgsl(options)).toContain('const SHARC_FALLBACK_LEVELS: u32 = 3u;');
        expect(sharcUpdateWgsl(options)).toContain('const SHARC_PYRAMID_LEVELS: u32 = 3u;');
        expect(sharcUpdateWgsl({ fallbackLevels: 2, pyramid: false })).toContain('const SHARC_PYRAMID_LEVELS: u32 = 0u;');
        expect(sharcUpdateWgsl(options)).toContain(`const SHARC_BUCKET: u32 = ${SHARC_BUCKET}u;`);
    });
});
