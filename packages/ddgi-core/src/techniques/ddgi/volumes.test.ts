import { describe, expect, it } from 'vitest';
import { GILimitError } from '../../platform/contract/errors.ts';
import { withDefaults } from './settings.ts';
import { MAX_VOLUMES, planVolumes, shareUnits, VOLUME_TABLE_ROW_FLOATS, volumeEdgeCoverage, volumeSettingsList, writeVolumeParameters } from './volumes.ts';

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

    it('shares a frame’s probe updates in proportion to the volumes’ probes, summing to the hand-out', () => {
        expect(shareUnits(100, [3000, 1000])).toEqual([75, 25]);
        const shares = shareUnits(101, [2028, 1071]);
        expect(shares.reduce((sum, share) => sum + share, 0)).toBe(101);
        expect(shares[0]).toBeGreaterThan(shares[1]);
        expect(shareUnits(0, [10, 10])).toEqual([0, 0]);
        expect(shareUnits(Number.POSITIVE_INFINITY, [10, 10])).toEqual([Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY]);
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

    it('hands one volume the hand-out exactly as it came', () => {
        expect(shareUnits(100.7, [1071])).toEqual([100.7]);
        expect(shareUnits(Number.POSITIVE_INFINITY, [1071])).toEqual([Number.POSITIVE_INFINITY]);
    });
});
