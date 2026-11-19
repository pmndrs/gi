import { describe, expect, it } from 'vitest';
import { cameraRelativeViewProjection } from './camera.ts';
import { historyChangesFromEvents, historyChangesWords, NO_HISTORY_CHANGES } from './screen-probe-history.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PROBE_CELLS } from './settings.ts';
import {
    blendProbeCells,
    blendRunningMean,
    HISTORY_SAMPLE_CAP,
    CACHED_TILE_NEW,
    cellCdf,
    findCellIndex,
    guidedCell,
    luminance,
    packNormal,
    RAY_GUIDING_UNIFORM_SHARE,
    reorderCachedTiles,
    simulateConstantSky,
    temporalBlendFactor,
    unpackNormal,
} from './screen-probe-history-cpu.ts';
import { CACHED_TILE_NEW as WGSL_CACHED_TILE_NEW, HISTORY_CHANGES_BYTES, MAX_HISTORY_CHANGE_BOXES, SAMPLE_WGSL } from './wgsl/screen-probes.ts';

describe("BlendScreenProbes' biased hysteresis (gi1.comp:1262-1268)", () => {
    it('keeps nothing of the past when the new value is darker or under twice as bright', () => {
        expect(temporalBlendFactor(1, 1)).toBe(0);
        expect(temporalBlendFactor(0, 1)).toBe(0);
        expect(temporalBlendFactor(1.9, 1)).toBe(0);
    });

    it('holds back a sudden brightening, at most 0.95 squared', () => {
        // (3 - 1 - 1) / 3 = 1/3, squared.
        expect(temporalBlendFactor(3, 1)).toBeCloseTo(1 / 9, 12);
        expect(temporalBlendFactor(100, 0)).toBeCloseTo(0.9025, 12);
    });

    it('uses Rec. 709 luminance', () => {
        expect(luminance([1, 1, 1])).toBeCloseTo(1, 12);
        expect(luminance([0, 1, 0])).toBeCloseTo(0.7152, 12);
    });
});

describe('the empty-cell rule (register D41)', () => {
    // Every reached cell holds the sky (1); cells 0..19 got no ray this frame.
    const counts = Array.from({ length: PROBE_CELLS }, (_, cell) => (cell < 20 ? 0 : 1 + (cell % 3)));
    const sums = counts.map((count) => count * 1);
    const history = new Array<number>(PROBE_CELLS).fill(1);

    it("Capsaicin's backup darkens an unreached cell by the number of empty cells", () => {
        const cells = blendProbeCells(sums, counts, history, 'capsaicin');
        // The summed samples (each ray counted) over the 44 reached cells, then over the 20 empty ones;
        // the hysteresis takes the darker value at once.
        const samples = counts.reduce((sum, count) => sum + count, 0);
        expect(cells[0]).toBeCloseTo(samples / 44 / 20, 12);
        expect(cells[63]).toBe(1);
    });

    it('ours keeps the previous radiance, and without history the mean of the reached cells', () => {
        expect(blendProbeCells(sums, counts, history, 'fixed').every((value) => value === 1)).toBe(true);
        const noHistory = blendProbeCells(sums, counts, new Array<number | null>(PROBE_CELLS).fill(null), 'fixed');
        expect(noHistory.every((value) => Math.abs(value - 1) < 1e-12)).toBe(true);
    });
});

describe('ray guiding (SampleScreenProbes, gi1.comp:916-978, register D42)', () => {
    it('builds the exclusive CDF and finds the last cell at or below a draw, skipping empty cells', () => {
        const { cdf, total } = cellCdf([1, 0, 1, 2]);
        expect(total).toBe(4);
        expect(cdf).toEqual([0, 0.25, 0.25, 0.5]);
        expect(findCellIndex(cdf, 0.1)).toBe(0);
        expect(findCellIndex(cdf, 0.3)).toBe(2);
        expect(findCellIndex(cdf, 0.9)).toBe(3);
    });

    it("Capsaicin's empty CDF sends every draw to the last cell; ours samples the ray's own cell", () => {
        const empty = cellCdf(new Array<number>(PROBE_CELLS).fill(0));
        expect(findCellIndex(empty.cdf, 0.3)).toBe(PROBE_CELLS - 1);
        expect(guidedCell(5, true, empty, 0.9)).toBe(5);
    });

    it('keeps a uniform share of rays on their own cells, and rescales the rest of the draw', () => {
        const weights = new Array<number>(PROBE_CELLS).fill(0);
        weights[40] = 1;
        const cdf = cellCdf(weights);
        expect(guidedCell(3, true, cdf, RAY_GUIDING_UNIFORM_SHARE * 0.99)).toBe(3);
        expect(guidedCell(3, true, cdf, RAY_GUIDING_UNIFORM_SHARE)).toBe(40);
        expect(guidedCell(3, false, cdf, 0.9)).toBe(3);
        // The CPU twin's share is the technique's default, which the GPU reads from RcFrame.
        expect(RAY_GUIDING_UNIFORM_SHARE).toBe(DEFAULT_RADIANCE_CACHE_SETTINGS.rayGuidingUniformShare);
        expect(SAMPLE_WGSL).toContain('rcFrame.rayGuidingUniformShare');
    });

    it("with a share of 0 follows Capsaicin's rule: every ray guided, an all-black CDF sends all to the last cell", () => {
        const weights = new Array<number>(PROBE_CELLS).fill(0);
        weights[40] = 1;
        expect(guidedCell(3, true, cellCdf(weights), 0.01, 0)).toBe(40);
        const empty = cellCdf(new Array<number>(PROBE_CELLS).fill(0));
        expect(guidedCell(5, true, empty, 0.9, 0)).toBe(PROBE_CELLS - 1);
        expect(guidedCell(5, false, empty, 0.9, 0)).toBe(5);
    });
});

describe('the off-screen probe cache', () => {
    it('packs normals as three 10-bit snorms (pack.hlsl)', () => {
        expect(packNormal([0, 0, 1])).toBe((511 << 20) >>> 0);
        expect(unpackNormal(packNormal([0, 0, -1]))).toEqual([0, 0, -1]);
        for (const normal of [[0.6, 0, -0.8], [-0.267, 0.534, 0.802], [1, 0, 0]] as const) {
            unpackNormal(packNormal(normal)).forEach((value, axis) => expect(Math.abs(value - normal[axis])).toBeLessThanOrEqual(0.5 / 511 + 1e-12));
        }
    });

    it('reorders the LRU list: unused entries first, in order, then the used ones by their MRU order (gi1.comp:1333-1354)', () => {
        const lru = [10, 11, 12, 13, 14];
        const flags = [1, 0, 1, 0, 1];
        const mru = [0, 1, 0, 0, 0];
        expect(reorderCachedTiles(lru, flags, mru)).toEqual([10, 12, 14, 13, 11]);
        expect(CACHED_TILE_NEW).toBe(WGSL_CACHED_TILE_NEW);
    });

    it('projects a cached position relative to the eye, as CountScreenProbes does', () => {
        const eye: [number, number, number] = [500, 3, -200];
        const view = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -eye[0], -eye[1], -eye[2], 1];
        const f = 1 / Math.tan(0.5);
        const projection = [f, 0, 0, 0, 0, f, 0, 0, 0, 0, 100 / (0.1 - 100), -1, 0, 0, (0.1 * 100) / (0.1 - 100), 0];
        const matrix = cameraRelativeViewProjection({ position: eye, view, projection });
        // A point straight ahead projects to the centre.
        const relative = [0, 0, -10];
        const clip = [0, 1, 2, 3].map((row) => matrix[row] * relative[0] + matrix[4 + row] * relative[1] + matrix[8 + row] * relative[2] + matrix[12 + row]);
        expect(clip[0] / clip[3]).toBeCloseTo(0, 12);
        expect(clip[1] / clip[3]).toBeCloseTo(0, 12);
        expect(clip[3]).toBeGreaterThan(0);
    });
});

describe('a constant sky with history (register D41): the cause of GI-1.2 drifting dark', () => {
    const report = [1, 2, 4, 8, 16, 32, 64, 128, 200, 300];

    it("reproduces Capsaicin's 1.30 at frame 1 falling to about 0.87 (native: 1.294 and 0.874)", () => {
        const results = simulateConstantSky({ rule: 'capsaicin', guiding: 'capsaicin', frames: 300, report });
        expect(results[0].capsaicinProjection).toBeCloseTo(1.3, 3);
        const steady = results.filter((result) => result.frame >= 64);
        for (const result of steady) {
            // The model settles at 0.876-0.880 from frame 16 on.
            expect(result.capsaicinProjection).toBeGreaterThan(0.85);
            expect(result.capsaicinProjection).toBeLessThan(0.9);
            // Darkened cells are rarely drawn again: about half of the cells get no ray each frame.
            expect(result.unreachedShare).toBeGreaterThan(0.45);
            expect(result.unreachedShare).toBeLessThan(0.6);
        }
    }, 60_000);

    it('the empty-cell rule is the cause: with it fixed the sky stays at 1 under either guiding', () => {
        for (const guiding of ['capsaicin', 'fixed'] as const) {
            for (const result of simulateConstantSky({ rule: 'fixed', guiding, frames: 300, report })) {
                expect(Math.abs(result.ourProjection - 1)).toBeLessThan(0.002);
                expect(result.capsaicinProjection).toBeCloseTo(1.3, 3);
                // Sampling with replacement still leaves about 40% of the cells without a ray each frame.
                if (result.frame > 1) expect(result.unreachedShare).toBeGreaterThan(0.38);
            }
        }
    }, 60_000);

    it("our guiding alone does not fix it: Capsaicin's rule still drifts", () => {
        const results = simulateConstantSky({ rule: 'capsaicin', guiding: 'fixed', frames: 128, report: [128] });
        expect(results[0].ourProjection).toBeLessThan(0.8);
    }, 60_000);
});

describe('noisy rays with history (register D46): the running mean against the hysteresis', () => {
    // E / (pi L) averaged over frames 100 to 200, so one frame's probe noise doesn't decide.
    const settled = (rule: 'fixed' | 'running', noise: 'exponential' | 'sparse'): number => {
        const report = Array.from({ length: 26 }, (_, index) => 100 + 4 * index);
        const results = simulateConstantSky({ rule, guiding: 'fixed', frames: 200, report, noise });
        return results.reduce((sum, result) => sum + result.ourProjection, 0) / results.length;
    };

    it('the hysteresis darkens noisy cells (6% at coefficient of variation 1, over half on sparse rays)', () => {
        expect(settled('fixed', 'exponential')).toBeLessThan(0.95);
        expect(settled('fixed', 'sparse')).toBeLessThan(0.6);
    }, 120_000);

    it('the running mean keeps the mean within 1% at coefficient of variation 1, and within 3% on sparse rays', () => {
        expect(Math.abs(settled('running', 'exponential') - 1)).toBeLessThan(0.01);
        // What is left on sparse rays is ray guiding by the cell's own estimate (NOTES).
        expect(Math.abs(settled('running', 'sparse') - 1)).toBeLessThan(0.03);
    }, 120_000);

    it('the running mean keeps the noiseless sky exact', () => {
        for (const result of simulateConstantSky({ rule: 'running', guiding: 'fixed', frames: 100, report: [1, 50, 100] })) {
            expect(Math.abs(result.ourProjection - 1)).toBeLessThan(0.002);
        }
    }, 60_000);

    it('takes k rays into a cell of n samples with weight k / min(n + k, cap), and fills without history', () => {
        const sums = new Array<number>(64).fill(0);
        const counts = new Array<number>(64).fill(0);
        sums[0] = 4;
        counts[0] = 2; // two rays, mean 2
        sums[1] = 3;
        counts[1] = 1;
        const previous: Array<number | null> = new Array<number | null>(64).fill(null);
        const previousCounts = new Array<number>(64).fill(0);
        previous[0] = 1;
        previousCounts[0] = 7;
        previous[2] = 5;
        previousCounts[2] = 3;
        const { values, counts: next } = blendRunningMean(sums, counts, previous, previousCounts);
        expect(values[0]).toBeCloseTo(1 + (2 - 1) * (2 / 8), 12);
        expect(next[0]).toBe(HISTORY_SAMPLE_CAP);
        expect(values[1]).toBe(3);
        expect(next[1]).toBe(1);
        expect(values[2]).toBe(5);
        expect(next[2]).toBe(3);
        // Neither rays nor history: the reached cells' mean, as a fill that counts nothing.
        expect(values[3]).toBeCloseTo((2 + 3) / 2, 12);
        expect(next[3]).toBe(0);
        expect(HISTORY_SAMPLE_CAP).toBe(DEFAULT_RADIANCE_CACHE_SETTINGS.historySampleCap);
    });
});

describe('scene changes for the running means (D46)', () => {
    const box = (x: number) => ({ min: [x, 0, 0] as const, max: [x + 1, 1, 1] as const });

    it("forgets at the events armed this frame: their boxes, everything for one with no bounds or too many", () => {
        expect(historyChangesFromEvents([{ box: box(0) }, { box: box(1) }])).toEqual({ everything: false, boxes: [box(0), box(1)] });
        expect(historyChangesFromEvents([{ box: box(0) }, { box: null }]).everything).toBe(true);
        expect(historyChangesFromEvents(Array.from({ length: MAX_HISTORY_CHANGE_BOXES + 1 }, (_, index) => ({ box: box(index) }))).everything).toBe(true);
        expect(historyChangesFromEvents([])).toBe(NO_HISTORY_CHANGES);
    });

    it('writes RcHistoryChanges: the count, the flag, then each box as min and max', () => {
        const words = historyChangesWords({ everything: false, boxes: [box(3)] });
        expect(words.byteLength).toBe(HISTORY_CHANGES_BYTES);
        expect(Array.from(new Uint32Array(words, 0, 2))).toEqual([1, 0]);
        expect(Array.from(new Float32Array(words, 16, 8))).toEqual([3, 0, 0, 0, 4, 1, 1, 0]);
        expect(Array.from(new Uint32Array(historyChangesWords({ everything: true, boxes: [] }), 0, 2))).toEqual([0, 1]);
    });
});
