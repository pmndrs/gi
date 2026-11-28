import { describe, expect, it } from 'vitest';
import { CELL_SHADING_BINS, CELL_SHADING_SUB_BINS, DEFAULT_HASH_GRID_CACHE_OPTIONS, hashGridCacheBufferLayout, hashGridCacheCapacity } from './capacity.ts';
import { cellShadingListLayout } from './cell-shading-common.ts';
import { cellPriorityClass, keepOneSample, prioritizeCells, scatterCells, simulateCell } from './cell-shading-cpu.ts';
import { CELL_PRIORITY_CLASS, CELL_SHADING_ENTRY_POINT, CELL_SHADING_UNIFORM_BYTES, claimCellsWgsl, prioritizeCellsWgsl, scatterCellsWgsl, shadeCellsWgsl } from './cell-shading-wgsl.ts';
import { cellShadingWords } from './hash-grid-cache.ts';
import { generateReservoirsWgsl, updateTilesWgsl } from './hash-grid-population-wgsl.ts';
import { makeRandom, rand } from './multibounce-cpu.ts';

const MAX_SAMPLE_COUNT = DEFAULT_HASH_GRID_CACHE_OPTIONS.hashGridCacheMaxSampleCount;

describe("lever E8's priority (cells lit on a budget)", () => {
    it('puts new and changed cells first, then converging ones, then those whose turn it is', () => {
        expect(cellPriorityClass({ count: 0, changed: false, turn: false }, MAX_SAMPLE_COUNT)).toBe(CELL_PRIORITY_CLASS.newOrChanged);
        expect(cellPriorityClass({ count: 16, changed: true, turn: false }, MAX_SAMPLE_COUNT)).toBe(CELL_PRIORITY_CLASS.newOrChanged);
        expect(cellPriorityClass({ count: 15, changed: false, turn: true }, MAX_SAMPLE_COUNT)).toBe(CELL_PRIORITY_CLASS.converging);
        expect(cellPriorityClass({ count: 16, changed: false, turn: true }, MAX_SAMPLE_COUNT)).toBe(CELL_PRIORITY_CLASS.turn);
        expect(cellPriorityClass({ count: 16, changed: false, turn: false }, MAX_SAMPLE_COUNT)).toBe(CELL_PRIORITY_CLASS.rest);
    });

    it('sorts by counting: each bin starts where the bins before it end, and the budget caps the lit count', () => {
        const counts = Array.from({ length: CELL_SHADING_BINS }, (_, bin) => (bin % 3) + 1);
        const total = counts.reduce((sum, count) => sum + count, 0);
        const prioritized = prioritizeCells(counts, 20);
        let expected = 0;
        prioritized.firstSlots.forEach((slot, bin) => {
            expect(slot).toBe(expected);
            expected += counts[bin];
        });
        expect(prioritized.shaded).toBe(20);
        expect(prioritizeCells(counts, total + 5).shaded).toBe(total);
        expect(prioritizeCells(counts, 0).shaded).toBe(0);
    });

    it('lights every candidate of a better bin before any of a worse one, whatever order they arrive in', () => {
        const random = makeRandom(7, 0);
        const bins = Array.from({ length: 5000 }, () => Math.floor(rand(random) * CELL_SHADING_BINS));
        const counts = new Array<number>(CELL_SHADING_BINS).fill(0);
        for (const bin of bins) counts[bin]++;
        const prioritized = prioritizeCells(counts, 1234);
        const slots = scatterCells(bins, prioritized);
        expect(slots.filter((slot) => slot.lit).length).toBe(1234);
        expect(new Set(slots.map((slot) => slot.slot)).size).toBe(bins.length);
        const worstLit = Math.max(...bins.filter((_, index) => slots[index].lit));
        const bestUnlit = Math.min(...bins.filter((_, index) => !slots[index].lit));
        expect(worstLit).toBeLessThanOrEqual(bestUnlit);
    });

    it("gives settled cells a turn period of what's left of the budget after the urgent ones", () => {
        const counts = new Array<number>(CELL_SHADING_BINS).fill(0);
        counts[0] = 1000; // new
        counts[CELL_PRIORITY_CLASS.converging * CELL_SHADING_SUB_BINS] = 1000;
        counts[CELL_PRIORITY_CLASS.rest * CELL_SHADING_SUB_BINS] = 30000;
        // 20,000 - 2,000 urgent leaves 18,000 for 30,000 settled: a turn every 2 frames.
        expect(prioritizeCells(counts, 20000).period).toBe(2);
        // Nothing left: a turn once in 'settled' frames; no settled cells: every frame.
        expect(prioritizeCells(counts, 1000).period).toBe(30000);
        expect(prioritizeCells([5, ...new Array<number>(CELL_SHADING_BINS - 1).fill(0)], 1).period).toBe(1);
    });

    it("cuts a changed cell's history to one sample and keeps its mean", () => {
        expect(keepOneSample([8, 4, 2, 16])).toEqual([0.5, 0.25, 0.125, 1]);
        expect(keepOneSample([0.5, 0, 0, 1])).toEqual([0.5, 0, 0, 1]);
        expect(keepOneSample([0, 0, 0, 0])).toEqual([0, 0, 0, 0]);
    });
});

describe("lever E8's estimate (a CPU model of one cell)", () => {
    // A cell across a shadow's edge: 6 hits a frame, each lit (light 1, exponential noise of mean 1)
    // with probability 0.4, so a hit's expected light is 0.4.
    const hitsAtEdge = (seed: number) => {
        const random = makeRandom(seed, 0);
        return () => Array.from({ length: 6 }, () => (rand(random) < 0.4 ? -Math.log(1 - rand(random)) : 0));
    };
    const meanAfter = (values: readonly number[], first: number) => values.slice(first).reduce((sum, value) => sum + value, 0) / (values.length - first);

    it('converges to the mean per-hit lighting converges to, lit on a quarter of the frames from one hit drawn uniformly', () => {
        const random = makeRandom(11, 0);
        const history = simulateCell({ frames: 40000, hits: hitsAtEdge(3), lit: () => rand(random) < 0.25, pick: () => rand(random), maxCount: MAX_SAMPLE_COUNT });
        expect(meanAfter(history.perHit, 1000)).toBeCloseTo(0.4, 2);
        expect(Math.abs(meanAfter(history.budgeted, 1000) / 0.4 - 1)).toBeLessThan(0.03);
    });

    it('would be biased if the sample were not drawn uniformly (the check can tell)', () => {
        // The hits ordered lit first: always taking the first reads the share of frames with any lit hit.
        const ordered = hitsAtEdge(5);
        const history = simulateCell({ frames: 20000, hits: () => [...ordered()].sort((a, b) => b - a), lit: () => true, pick: () => 0, maxCount: MAX_SAMPLE_COUNT });
        expect(meanAfter(history.budgeted, 1000)).toBeGreaterThan(0.8);
    });

    it('follows a light that doubles in 36 frames per hit (the 16-sample cap); a changed cell lit once reads the new light', () => {
        // Per hit: 2 - (15/16)^k within 5% of 2 needs k >= ln(10) / ln(16/15) = 35.7 frames.
        const history = simulateCell({ frames: 200, hits: (frame) => [frame < 100 ? 1 : 2, frame < 100 ? 1 : 2], lit: () => true, pick: () => 0, maxCount: MAX_SAMPLE_COUNT });
        const within = history.perHit.findIndex((value, frame) => frame >= 100 && Math.abs(value / 2 - 1) < 0.05);
        expect(within - 100 + 1).toBe(36);
        // Budgeted, a changed cell keeps one sample and, lit, takes the new sample whole (ShadeCells empties it).
        expect(keepOneSample([16, 0, 0, 16])[0]).toBe(1);
    });
});

describe("lever E8's kernels and lists", () => {
    const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS), true);

    it('declare their entry points, and the population passes branch on the setting', () => {
        const claim = claimCellsWgsl(layout);
        expect(claim).toContain(`fn ${CELL_SHADING_ENTRY_POINT.claimFirstBounceCells}(`);
        expect(claim).toContain(`fn ${CELL_SHADING_ENTRY_POINT.claimSecondBounceCells}(`);
        expect(prioritizeCellsWgsl(layout)).toContain(`fn ${CELL_SHADING_ENTRY_POINT.prioritizeCells}(`);
        expect(scatterCellsWgsl(layout)).toContain(`fn ${CELL_SHADING_ENTRY_POINT.scatterCells}(`);
        expect(shadeCellsWgsl(layout)).toContain('giTraceAny');
        expect(generateReservoirsWgsl(layout)).toContain('HashGridCache_OfferCellSample(did');
        expect(updateTilesWgsl(layout)).toContain('HashGridCache_CellShadingBudgeted()');
    });

    it('light a cell from the lights GenerateReservoirs draws from, the sky with environment_light_enable (register D62)', () => {
        const shade = shadeCellsWgsl(layout);
        expect(shade).toContain('giDirectChooseLight(surface, HashGridCache_LightMask(), rcRand(&random))');
        expect(shade).toContain('select(0u, GI_DIRECT_SKY_LIGHT_BIT, g_HashGridCacheConstants.environment_light_enable != 0u)');
    });

    it('keep two candidate words and a lit word per visibility entry, then the bins, in a buffer of their own', () => {
        const lists = cellShadingListLayout(2000);
        expect(lists.lit).toBe(lists.candidates + 2 * 2000);
        expect(lists.bins).toBe(lists.lit + 2000);
        expect(lists.bytes).toBe((lists.bins + 3 * CELL_SHADING_BINS) * 4);
    });

    it('pack the budget and the change events: each box with its epoch plus one, a change with no bounds as the whole world', () => {
        const box = { min: [0, 1, 2] as const, max: [3, 4, 5] as const };
        const words = cellShadingWords(5000, { events: [{ box, epoch: 41 }, { box: null, epoch: 7 }], everything: true, boxes: [] });
        expect(words.byteLength).toBe(CELL_SHADING_UNIFORM_BYTES);
        expect([...new Uint32Array(words, 0, 2)]).toEqual([5000, 2]);
        expect([...new Float32Array(words, 16, 3)]).toEqual([0, 1, 2]);
        expect(new Uint32Array(words, 28, 1)[0]).toBe(42);
        expect([...new Float32Array(words, 32, 3)]).toEqual([3, 4, 5]);
        expect(new Float32Array(words, 48, 1)[0]).toBeLessThan(-1e38);
        expect(new Uint32Array(words, 60, 1)[0]).toBe(8);
        expect([...new Uint32Array(cellShadingWords(1, undefined), 0, 2)]).toEqual([1, 0]);
        expect(() => cellShadingWords(1, { events: new Array(17).fill({ box, epoch: 1 }), everything: false, boxes: [] })).toThrow(RangeError);
    });
});
