import { describe, expect, it } from 'vitest';
import { GILimitError } from '../../platform/contract/errors.ts';
import { withDefaults } from './settings.ts';
import { DEFAULT_MAX_PASSES_PER_FRAME } from '../../platform/timing/gpu-timer.ts';
import { MAX_VOLUMES, planVolumes, TIMED_PASSES_PER_VOLUME, UnitSharer, VOLUME_TABLE_ROW_FLOATS, VOLUME_TABLE_TEXEL, volumeEdgeCoverage, volumeFloorUnits, volumeSettingsList, writeVolumeParameters, writeVolumeWindow } from './volumes.ts';

// Two-room's volume (17 x 7 x 9 probes at 0.5 m) and a dense one inside it (13 x 12 x 13 at 0.25 m).
const COARSE = { center: [4.1, 1.4, 2.0], size: [8.4, 3.0, 4.2], spacing: 0.5 } as const;
const DENSE = { center: [2.0, 1.4, 2.0], size: [3.0, 2.75, 3.0], spacing: 0.25, priority: 1 } as const;

describe('several DDGI volumes (m3-moving.md §4.1)', () => {
    it('reads one volume as the one-element case, and a list as the volumes', () => {
        const one = withDefaults({ volume: COARSE });
        expect(volumeSettingsList(one)).toEqual([COARSE]);
        const two = withDefaults({ volumes: [COARSE, DENSE] });
        expect(volumeSettingsList(two)).toEqual([COARSE, DENSE]);
        // Hosts that read one volume see the first.
        expect(two.volume).toBe(COARSE);
        expect(() => withDefaults({ volumes: [] })).toThrow(/at least|non-empty/);
    });

    it('puts higher priority first, gives each volume its own layers and probes, and sizes the atlas for the largest footprint', () => {
        const plan = planVolumes([COARSE, DENSE], 256);
        expect(plan.volumes.map((volume) => volume.settingsIndex)).toEqual([1, 0]);
        const [dense, coarse] = plan.volumes;
        expect(dense.grid.counts).toEqual([13, 12, 13]);
        expect(coarse.grid.counts).toEqual([17, 7, 9]);
        expect(dense.layerBase).toBe(0);
        expect(coarse.layerBase).toBe(12);
        expect(dense.probeBase).toBe(0);
        expect(coarse.probeBase).toBe(13 * 12 * 13);
        expect(plan.probeCount).toBe(13 * 12 * 13 + 17 * 7 * 9);
        // Tiles across: the largest nx; layers: the sum of ny; tiles down: the largest nz.
        expect(plan.atlasCounts).toEqual([17, 19, 13]);
    });

    it('keeps the settings order between equal priorities', () => {
        const plan = planVolumes([COARSE, { ...COARSE, center: [0, 0, 0] }, DENSE], 256);
        expect(plan.volumes.map((volume) => volume.settingsIndex)).toEqual([2, 0, 1]);
    });

    it('lays one volume out exactly as before volumes: layer base 0, the atlas its own grid', () => {
        const plan = planVolumes([COARSE], 256);
        expect(plan.volumes[0].layerBase).toBe(0);
        expect(plan.atlasCounts).toEqual(plan.volumes[0].grid.counts);
    });

    it('refuses more than 8 volumes, and more array layers than the device has', () => {
        expect(() => planVolumes(Array.from({ length: MAX_VOLUMES + 1 }, () => COARSE), 256)).toThrow(/at most 8/);
        expect(() => planVolumes([COARSE, DENSE], 18)).toThrow(GILimitError);
        expect(() => planVolumes([COARSE, DENSE], 18)).toThrow(/maxTextureArrayLayers = 19/);
    });

    it('writes a volume-table row as the four sampling vec4s, with the layer base and the shared atlas size', () => {
        const plan = planVolumes([COARSE, DENSE], 256);
        const row = new Float32Array(VOLUME_TABLE_ROW_FLOATS);
        writeVolumeParameters(row, 0, plan.volumes[1], plan, { irradiance: 6, distance: 14 }, { normal: 0.05, view: 0.04 });
        const coarse = plan.volumes[1].grid;
        expect([...row.subarray(0, 4)]).toEqual([...coarse.origin, 0.05].map(Math.fround));
        expect([...row.subarray(4, 8)]).toEqual([...coarse.spacing, 0.04].map(Math.fround));
        expect([...row.subarray(8, 12)]).toEqual([17, 7, 9, 12]);
        expect([...row.subarray(12, 16)]).toEqual([6, 14, 17, 13]);
    });

    it('writes the window, the fade with the weight-floors flag (DV-06) in its w, and the valid extent', () => {
        const row = new Float32Array(VOLUME_TABLE_ROW_FLOATS);
        const extent = { min: [0, 1, 2], max: [16, 6, 8] } as const;
        writeVolumeWindow(row, 0, [3, 0, -2], [0.5, 0.5, 0.5], extent, true);
        const texel = (index: number): number[] => [...row.subarray(index * 4, index * 4 + 4)];
        expect(texel(VOLUME_TABLE_TEXEL.window)).toEqual([3, 0, -2, 0]);
        expect(texel(VOLUME_TABLE_TEXEL.fade)).toEqual([0.5, 0.5, 0.5, 1]);
        expect(texel(VOLUME_TABLE_TEXEL.validMin)).toEqual([0, 1, 2, 0]);
        expect(texel(VOLUME_TABLE_TEXEL.validMax)).toEqual([16, 6, 8, 0]);
        writeVolumeWindow(row, 0, [3, 0, -2], [0.5, 0.5, 0.5], extent, false);
        expect(texel(VOLUME_TABLE_TEXEL.fade)).toEqual([0.5, 0.5, 0.5, 0]);
    });

    it('shares a frame’s probe updates in proportion to the volumes’ probes, each frame adding up to the hand-out', () => {
        expect(new UnitSharer([3000, 1000]).share(100)).toEqual([75, 25]);
        const shares = new UnitSharer([2028, 1071]).share(101);
        expect(shares.reduce((sum, share) => sum + share, 0)).toBe(101);
        expect(shares[0]).toBeGreaterThan(shares[1]);
        expect(new UnitSharer([10, 10]).share(0)).toEqual([0, 0]);
        expect(new UnitSharer([10, 10]).share(Number.POSITIVE_INFINITY)).toEqual([Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]);
    });

    it('never starves a small volume beside a big one: its fraction carries over frames (the T3.5 review cases)', () => {
        for (const [units, counts] of [
            [154, [20000, 125]],
            [30, [4840, 125]],
        ] as const) {
            const sharer = new UnitSharer(counts);
            const frames = 1000;
            const given = [0, 0];
            for (let frame = 0; frame < frames; frame++) {
                const shares = sharer.share(units);
                expect(shares[0] + shares[1]).toBe(units);
                given[0] += shares[0];
                given[1] += shares[1];
            }
            // Over the frames each volume gets its exact share, within one probe.
            const exactSmall = (frames * units * counts[1]) / (counts[0] + counts[1]);
            expect(given[1]).toBeGreaterThan(0);
            expect(Math.abs(given[1] - exactSmall)).toBeLessThanOrEqual(1);
            expect(given[0] + given[1]).toBe(frames * units);
        }
    });

    it('adds up to the hand-out over a long run with changing hand-outs and three volumes', () => {
        const counts = [19425, 2700, 64];
        const sharer = new UnitSharer(counts);
        const total = counts.reduce((sum, count) => sum + count, 0);
        const given = [0, 0, 0];
        let handedOut = 0;
        for (let frame = 0; frame < 5000; frame++) {
            const units = 40 + ((frame * 37) % 300);
            const shares = sharer.share(units);
            expect(shares.reduce((sum, share) => sum + share, 0)).toBe(units);
            shares.forEach((share, index) => (given[index] += share));
            handedOut += units;
        }
        expect(given.reduce((sum, share) => sum + share, 0)).toBe(handedOut);
        counts.forEach((count, index) => expect(Math.abs(given[index] - (handedOut * count) / total)).toBeLessThanOrEqual(1));
    });

    it('gives every volume its own floor: a sweep within 4 s and never 0 probes a frame', () => {
        const rate = { framesPerSecond: 60, calibratedUnits: 154 };
        const small = volumeFloorUnits('auto', { probeCount: 125, maxProbesPerFrame: 125 }, 20125, rate);
        expect(small).toBeGreaterThanOrEqual(1);
        // 125 probes at 1 a frame: swept in 125 frames, about 2 s at 60 Hz.
        expect(Math.ceil(125 / small) / 60).toBeLessThanOrEqual(4);
        const big = volumeFloorUnits('auto', { probeCount: 20000, maxProbesPerFrame: 4096 }, 20125, rate);
        expect(Math.ceil(20000 / big) / 60).toBeLessThanOrEqual(4);
        // An explicit floor is shared by probe count (rounded up, never 0); 0 switches it off.
        expect(volumeFloorUnits(30, { probeCount: 125, maxProbesPerFrame: 125 }, 4965, rate)).toBe(1);
        expect(volumeFloorUnits(0, { probeCount: 125, maxProbesPerFrame: 125 }, 4965, rate)).toBe(0);
    });

    it('fades a volume out over the half spacing past its margin', () => {
        const [dense] = planVolumes([DENSE], 256).volumes;
        const lastX = dense.grid.origin[0] + (dense.grid.counts[0] - 1) * dense.grid.spacing[0];
        const at = (x: number): number => volumeEdgeCoverage(dense.grid, [x, 1.4, 2.0]);
        expect(at(2.0)).toBe(1);
        // Full out to half a spacing past the last probe, then a straight fade over the next half spacing.
        expect(at(lastX + 0.5 * 0.25)).toBe(1);
        expect(at(lastX + 0.75 * 0.25)).toBeCloseTo(0.5, 6);
        expect(at(lastX + 0.25)).toBe(0);
        expect(at(lastX + 1)).toBe(0);
    });

    it('fits every volume’s passes in the GPU timer’s frame, with room for the platform’s and extra direct rounds', () => {
        // The platform's own timed passes in a frame: top-level refit, light grid, sky bake, mesh packing; 8 is generous.
        const platformPasses = 8;
        expect(MAX_VOLUMES * TIMED_PASSES_PER_VOLUME + platformPasses).toBeLessThanOrEqual(DEFAULT_MAX_PASSES_PER_FRAME);
    });

    it('hands one volume the hand-out exactly as it came', () => {
        const sharer = new UnitSharer([1071]);
        expect(sharer.share(100.7)).toEqual([100.7]);
        expect(sharer.share(Number.POSITIVE_INFINITY)).toEqual([Number.POSITIVE_INFINITY]);
    });
});
