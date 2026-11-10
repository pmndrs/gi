import { describe, expect, it } from 'vitest';
import { quantizeRadiance } from './capsaicin-math.ts';
import {
    CACHED_TILE_UNALLOCATED,
    cdfCellProbabilities,
    exclusiveScan,
    findCellIndex,
    initialCachedTileLRU,
    listCachedTiles,
    probeCacheLayout,
    reorderCachedTiles,
    screenProbeCdf,
    temporalBlend,
} from './probe-cache.ts';
import { PROBE_CELLS, screenProbeLayout } from './settings.ts';
import { LARGEST_QUANTIZED_HIT_DISTANCE } from './wgsl/capsaicin.ts';

describe('the probe cache layout (probe-cache.ts)', () => {
    it("packs Capsaicin's arrays without overlap, sized as gi1.cpp:182-335", () => {
        const layout = screenProbeLayout(1280, 720);
        const cache = probeCacheLayout(layout);
        expect(cache.slots).toBe(160 * 90);
        expect(cache.spawnTiles).toBe(80 * 45);
        const runs: Array<[number, number]> = [
            [cache.lru[0], cache.slots],
            [cache.lru[1], cache.slots],
            [cache.mru, cache.slots],
            [cache.list, cache.slots],
            [cache.listElement, 4 * cache.slots],
            [cache.emptyTile, cache.slots],
            [cache.overrideTile, cache.spawnTiles],
            [cache.spawnProbe, 2 * cache.spawnTiles],
        ];
        runs.sort((a, b) => a[0] - b[0]);
        for (let index = 1; index < runs.length; index++) expect(runs[index][0]).toBe(runs[index - 1][0] + runs[index - 1][1]);
        expect(cache.words).toBe(runs[runs.length - 1][0] + runs[runs.length - 1][1]);
    });
});

describe('the LRU (CountScreenProbes, ScatterScreenProbes, ReorderScreenProbes)', () => {
    it('starts with every slot holding its own tile, unallocated (InitCachedTileLRU)', () => {
        const lru = initialCachedTileLRU(4);
        expect(Array.from(lru)).toEqual([0, 1, 2, 3].map((slot) => (slot | CACHED_TILE_UNALLOCATED) >>> 0));
    });

    it('lists only allocated slots that project inside the screen and the depth range, by tile', () => {
        // Four slots: 0 and 2 allocated on screen, 1 allocated behind the far plane, 3 unallocated.
        const lru = new Uint32Array([5, 6, 7, (8 | CACHED_TILE_UNALLOCATED) >>> 0]);
        const positions: Record<number, [number, number, number]> = { 5: [0.1, 0.1, 0.5], 6: [0, 0, 2], 7: [-0.6, -0.6, 0.5], 8: [0, 0, 0.5] };
        // A "projection" that is the identity, so positions are already NDC.
        const lists = listCachedTiles(lru, (tile) => positions[tile], (world) => world, [4, 4]);
        expect(lists.elementCount).toBe(2);
        // (0.1, 0.1) -> uv (0.55, 0.45) -> tile (2, 1); (-0.6, -0.6) -> uv (0.2, 0.8) -> tile (0, 3).
        expect(lists.slotsByTile.get(2 + 1 * 4)).toEqual([0]);
        expect(lists.slotsByTile.get(0 + 3 * 4)).toEqual([2]);
        expect(lists.counts.reduce((sum, count) => sum + count, 0)).toBe(2);
        expect(Array.from(lists.starts)).toEqual(Array.from(exclusiveScan(lists.counts)));
    });

    it('keeps untouched slots first in their order, then the used ones by MRU (the least recently used are reused first)', () => {
        const lru = new Uint32Array([10, 11, 12, 13, 14]);
        // Slots 1 and 3 were used this frame: 3 first (MRU 0), then 1 (MRU 1).
        const flags = new Uint32Array([1, 0, 1, 0, 1]);
        const mru = new Uint32Array([0, 1, 0, 0, 0]);
        expect(Array.from(reorderCachedTiles(lru, flags, mru, 5))).toEqual([10, 12, 14, 13, 11]);
    });

    it('is a permutation whatever the flags', () => {
        const slots = 64;
        const lru = Uint32Array.from({ length: slots }, (_, slot) => slot * 3);
        const flags = Uint32Array.from({ length: slots }, (_, slot) => (slot % 5 === 0 ? 0 : 1));
        let next = 0;
        const mru = Uint32Array.from({ length: slots }, (_, slot) => (slot % 5 === 0 ? next++ : 0));
        const reordered = reorderCachedTiles(lru, flags, mru, slots);
        expect(Array.from(reordered).sort((a, b) => a - b)).toEqual(Array.from(lru));
    });
});

describe('ray guiding (ScreenProbes_ScanRadiance, ScreenProbes_FindCellIndex)', () => {
    it('draws each cell in proportion to its weight', () => {
        const weights = [0, 1, 0, 3, 0, 0];
        const cdf = screenProbeCdf(weights);
        expect(cdf).toEqual([0, 0, 0.25, 0.25, 1, 1]);
        const counts = new Array<number>(weights.length).fill(0);
        const draws = 4000;
        for (let draw = 0; draw < draws; draw++) counts[findCellIndex(cdf, (draw + 0.5) / draws)]++;
        expect(counts).toEqual([0, 1000, 0, 3000, 0, 0]);
        expect(cdfCellProbabilities(weights)).toEqual([0, 0.25, 0, 0.75, 0, 0]);
    });

    it('sends every draw to the last cell when the history is black (Capsaicin quirk, kept)', () => {
        const weights = new Array<number>(PROBE_CELLS).fill(0);
        const cdf = screenProbeCdf(weights);
        expect(findCellIndex(cdf, 0)).toBe(PROBE_CELLS - 1);
        expect(findCellIndex(cdf, 0.99)).toBe(PROBE_CELLS - 1);
        expect(cdfCellProbabilities(weights)[PROBE_CELLS - 1]).toBe(1);
    });
});

describe('the biased hysteresis (BlendScreenProbes, gi1.comp:1262-1268)', () => {
    it('takes a darker or similar value at once and keeps most of the history against a jump up', () => {
        expect(temporalBlend(0.5, 1)).toBe(0);
        expect(temporalBlend(1, 1)).toBe(0);
        expect(temporalBlend(2, 1)).toBe(0);
        expect(temporalBlend(4, 1)).toBeCloseTo(0.25, 12);
        expect(temporalBlend(1, 0)).toBeCloseTo(0.95 * 0.95, 12);
    });
});

describe('the quantized hit distance (register D48)', () => {
    it('lets a whole probe of missed rays share one cell without wrapping the u32 sum', () => {
        const missed = quantizeRadiance(65504);
        // Unclamped, seven misses in one cell already wrap.
        expect(7 * missed).toBeGreaterThan(2 ** 32 - 1);
        expect(LARGEST_QUANTIZED_HIT_DISTANCE * PROBE_CELLS).toBeLessThanOrEqual(2 ** 32 - 1);
        expect(LARGEST_QUANTIZED_HIT_DISTANCE / 1e4).toBeCloseTo(6710.8863, 4);
    });
});
