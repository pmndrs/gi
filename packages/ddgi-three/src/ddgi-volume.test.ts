import { describe, expect, it } from 'vitest';
import { probeGrid } from '../../ddgi-core/src/techniques/ddgi/settings.ts';
import { DDGIVolume } from './ddgi-volume.ts';

describe('DDGIVolume (LightProbeGrid-shaped)', () => {
    it('puts a DDGI probe at the centre of each cell', () => {
        const volume = new DDGIVolume(4, 2, 6, 4, 2, 3);
        volume.position.set(1, 2, 3);
        volume.updateMatrixWorld();
        const settings = volume.ddgiVolume();
        expect(settings.center).toEqual([1, 2, 3]);
        expect(settings.spacing).toEqual([1, 1, 2]);
        expect(settings.size).toEqual([3, 1, 4]);
        const grid = probeGrid(settings);
        expect(grid.counts).toEqual([4, 2, 3]);
        // The first probe sits half a cell inside the volume's corner.
        expect(grid.origin).toEqual([1 - 2 + 0.5, 2 - 1 + 0.5, 3 - 3 + 1]);
    });

    it('keeps the probe count when the spacing is not a whole number', () => {
        const volume = new DDGIVolume(3.4, 1, 1, 3, 1, 1);
        volume.updateMatrixWorld();
        const settings = volume.ddgiVolume();
        expect((settings.spacing as readonly number[])[0]).toBeCloseTo(3.4 / 3);
        expect(settings.size[0]).toBeCloseTo(3.4 - 3.4 / 3);
        const grid = probeGrid(settings);
        expect(grid.counts).toEqual([3, 1, 1]);
        // Probes at the cell centres: -1.7 + 3.4 / 6, 0, and +1.7 - 3.4 / 6.
        expect(grid.origin[0]).toBeCloseTo(-1.7 + 3.4 / 6);
    });

    it('defaults to one probe per unit of length, like LightProbeGrid', () => {
        const volume = new DDGIVolume(3.4, 0.2, 5);
        expect(volume.resolution.toArray()).toEqual([3, 1, 5]);
    });

    it('takes a preset\'s spacing only where the counts were left to the default', () => {
        const defaulted = new DDGIVolume(8, 4, 6);
        defaulted.useDefaultSpacing(2);
        expect(defaulted.resolution.toArray()).toEqual([4, 2, 3]);
        // Counts the caller gave are kept, axis by axis.
        const partly = new DDGIVolume(8, 4, 6, 16);
        partly.useDefaultSpacing(2);
        expect(partly.resolution.toArray()).toEqual([16, 2, 3]);
        // So are counts set after the constructor.
        const changed = new DDGIVolume(8, 4, 6);
        changed.resolution.set(5, 5, 5);
        changed.useDefaultSpacing(2);
        expect(changed.resolution.toArray()).toEqual([5, 5, 5]);
        // At least one probe, and nonsense spacings change nothing.
        const small = new DDGIVolume(1, 1, 1);
        small.useDefaultSpacing(4);
        expect(small.resolution.toArray()).toEqual([1, 1, 1]);
        small.useDefaultSpacing(Number.NaN);
        expect(small.resolution.toArray()).toEqual([1, 1, 1]);
    });

    it('keeps a bounding box around its world position', () => {
        const volume = new DDGIVolume(2, 4, 6);
        volume.position.set(10, 0, 0);
        volume.updateMatrixWorld();
        const box = volume.updateBoundingBox();
        expect(box.min.toArray()).toEqual([9, -2, -3]);
        expect(box.max.toArray()).toEqual([11, 2, 3]);
    });

    it('has priority and intensity like LightProbeGrid, above its default priority', () => {
        const volume = new DDGIVolume();
        expect(volume.intensity).toBe(1);
        expect(volume.priority).toBeGreaterThan(0);
    });
});
