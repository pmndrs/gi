import { describe, expect, it } from 'vitest';
import { cameraRelativeInverseViewProjection, invertMat4, multiplyMat4 } from './camera.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PROBE_CELLS, PROBE_SIZE, PROBE_SPAWN_TILE_SIZE, screenProbeLayout } from './settings.ts';
import { RadianceCacheTechnique } from './radiance-cache-technique.ts';
import { COUNTER_NAMES, COUNTERS_BYTES } from './wgsl/screen-probes.ts';

describe('settings (gi1.h)', () => {
    it("are Capsaicin's defaults", () => {
        expect(PROBE_SIZE).toBe(8);
        expect(PROBE_CELLS).toBe(64);
        expect(PROBE_SPAWN_TILE_SIZE).toBe(16);
        expect(DEFAULT_RADIANCE_CACHE_SETTINGS).toMatchObject({
            useDirectLighting: true,
            useTemporalFeedback: false,
            useMultibounce: true,
            useResampling: false,
            hashGridCacheCellSize: 32,
            hashGridCacheMinCellSize: 0.1,
            hashGridCacheTileCellRatio: 8,
            hashGridCacheNumBuckets: 14,
            hashGridCacheNumTilesPerBucket: 4,
            hashGridCacheMaxSampleCount: 16,
        });
        const technique = new RadianceCacheTechnique({ useDirectLighting: false });
        expect(technique.settings.useDirectLighting).toBe(false);
        expect(technique.settings.useMultibounce).toBe(true);
        // SSGI (off by default, D61) reads the host's lit image and albedo when on (RC4.2, register D51).
        expect(technique.requirements).toEqual({ queries: ['rays'], screen: { velocity: true, albedo: false, lighting: false } });
        expect(new RadianceCacheTechnique({ ssgi: true }).requirements).toEqual({ queries: ['rays'], screen: { velocity: true, albedo: true, lighting: true } });
    });

    it('size the probes as Capsaicin does (gi1.cpp:113-129)', () => {
        const layout = screenProbeLayout(1280, 720);
        expect(layout.probeCount).toEqual([160, 90]);
        expect(layout.atlasSize).toEqual([1280, 720]);
        expect(layout.maxProbeSpawnCount).toBe(80 * 45);
        expect(layout.maxRayCount).toBe(3600 * 64);
        expect(layout.probeMaskMipCount).toBe(8);
        // 1080p, the source map's figures: 8,160 spawn tiles, 522,240 rays.
        const fullHd = screenProbeLayout(1920, 1080);
        expect(fullHd.maxProbeSpawnCount).toBe(8160);
        expect(fullHd.maxRayCount).toBe(522240);
        expect(fullHd.atlasSize).toEqual([1920, 1080]);
        // A screen that is not whole tiles rounds the atlas up to them.
        expect(screenProbeLayout(1000, 700).atlasSize).toEqual([1000, 704]);
        expect(() => screenProbeLayout(0, 10)).toThrow(RangeError);
    });

    it('has room for every counter ClearCounters resets', () => {
        expect(COUNTER_NAMES.length * 4).toBeLessThanOrEqual(COUNTERS_BYTES);
    });
});

describe('the camera-relative inverse view-projection', () => {
    const perspective = (fovY: number, aspect: number, near: number, far: number): number[] => {
        // WebGPU's 0..1 depth, column-major.
        const f = 1 / Math.tan(fovY / 2);
        return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
    };
    const lookAtView = (eye: [number, number, number]): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -eye[0], -eye[1], -eye[2], 1];

    it('inverts a matrix', () => {
        const projection = perspective(1, 16 / 9, 0.1, 100);
        const product = multiplyMat4(projection, invertMat4(projection) ?? []);
        product.forEach((value, index) => expect(value).toBeCloseTo(index % 5 === 0 ? 1 : 0, 9));
        expect(invertMat4(new Array(16).fill(0))).toBeNull();
    });

    it('reconstructs a world point from its depth, relative to the eye', () => {
        const eye: [number, number, number] = [1000, 20, -500];
        const camera = { position: eye, view: lookAtView(eye), projection: perspective(1, 1, 0.1, 100) };
        const point = [1003, 21, -510];
        const clip = multiplyMat4(multiplyMat4(camera.projection, camera.view), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...point, 1]).slice(12, 16);
        const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1];
        const inverse = cameraRelativeInverseViewProjection(camera);
        const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row] * ndc[3]);
        [0, 1, 2].forEach((axis) => expect(eye[axis] + h[axis] / h[3]).toBeCloseTo(point[axis], 6));
    });
});
