import { describe, expect, it } from 'vitest';
import { probeGrid } from '../../ddgi-core/src/techniques/ddgi/settings.ts';
import { GIVolume } from './gi-volume.ts';

describe('GIVolume (LightProbeGrid-shaped)', () => {
    it('puts a DDGI probe at the centre of each cell', () => {
        const volume = new GIVolume(4, 2, 6, 4, 2, 3);
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

    it('defaults to one probe per unit of length, like LightProbeGrid', () => {
        const volume = new GIVolume(3.4, 0.2, 5);
        expect(volume.resolution.toArray()).toEqual([3, 1, 5]);
    });

    it('keeps a bounding box around its world position', () => {
        const volume = new GIVolume(2, 4, 6);
        volume.position.set(10, 0, 0);
        volume.updateMatrixWorld();
        const box = volume.updateBoundingBox();
        expect(box.min.toArray()).toEqual([9, -2, -3]);
        expect(box.max.toArray()).toEqual([11, 2, 3]);
    });

    it('has priority and intensity like LightProbeGrid, above its default priority', () => {
        const volume = new GIVolume();
        expect(volume.intensity).toBe(1);
        expect(volume.priority).toBeGreaterThan(0);
    });
});
