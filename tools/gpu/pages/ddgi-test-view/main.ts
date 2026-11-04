/**
 * T1.5a done-when 2: the engine-free test view of DDGI's indirect light.
 *
 * Converges DDGI on a manifest scene (every probe every frame), then renders
 * the test view of one camera bookmark (ddgi-shared/harness.ts, `TestView`):
 * `albedo / π × ddgiSampleIrradiance(...)` per pixel, plus the full image.
 *
 * Two ways to run it:
 * - `pnpm test:gpu --filter ddgi-test-view`: the point-light Cornell box at
 *   bookmark 1, with the light's range window off as in Cycles, on the volume
 *   with every probe in the air (`CORNELL_POINT_IN_AIR_VOLUME`: T1.5a has no
 *   relocation, and the manifest's volume buries two probe layers in the
 *   floor and ceiling slabs). Fails if the image's mean is not within ±10%
 *   of the Cycles indirect reference's mean (the M1 accuracy target), or has
 *   non-finite pixels.
 * - `node tools/gpu/pages/ddgi-test-view/render.ts`: any scene and camera; the
 *   script injects `window.ddgiTestViewJob`, takes the images back, writes
 *   EXRs and runs `tools/compare` on them.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { DDGISettingsInput } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { CORNELL_POINT_IN_AIR_VOLUME, createDdgiScene, floatsToBase64, imageMean, runFrames, TestView } from '../ddgi-shared/harness.ts';

export interface TestViewJob {
    readonly scene: string;
    /** 1-based bookmark index. */
    readonly camera: number;
    readonly width: number;
    readonly height: number;
    readonly frames: number;
    readonly ignoreLightRange: boolean;
    /** Return the images as base64 floats (for the render script). */
    readonly exportImages: boolean;
    /** Mean of the reference's indirect image, to check the test view against; null to skip. */
    readonly referenceIndirectMean: number | null;
    /** DDGI settings on top of the defaults and the manifest's volume (experiments). */
    readonly settings?: Partial<DDGISettingsInput>;
}

/**
 * Mean of `bench/references/cornell-box-point/front/cycles-indirect.exr`
 * (Cycles 4.5, 2048 samples, 512 × 512, T1.5 prep), all pixels and channels.
 */
const CORNELL_POINT_CYCLES_INDIRECT_MEAN = 0.083968;
/** SPEC's M1 accuracy target: indirect light's mean within ±10%. */
const MEAN_TOLERANCE = 0.1;

const DEFAULT_JOB: TestViewJob = {
    scene: 'cornell-box-point',
    camera: 1,
    width: 512,
    height: 512,
    frames: 600,
    ignoreLightRange: true,
    exportImages: false,
    referenceIndirectMean: CORNELL_POINT_CYCLES_INDIRECT_MEAN,
    settings: { volume: CORNELL_POINT_IN_AIR_VOLUME },
};

declare global {
    interface Window {
        ddgiTestViewJob?: TestViewJob;
    }
}

const job: TestViewJob = window.ddgiTestViewJob ?? DEFAULT_JOB;

await runGpuTest(
    async ({ device, log }) => {
        const scene = await createDdgiScene(device, job.scene, job.settings ?? {}, { ignoreLightRange: job.ignoreLightRange });
        const field = scene.ddgi.field;
        assert(field, 'DDGI has no field after prepare');
        const camera = scene.manifest.cameras[job.camera - 1];
        assert(camera, `${job.scene} has no camera ${job.camera}`);

        const start = performance.now();
        await runFrames(device, scene.platform, job.frames, field.grid.probeCount);
        const convergeMs = performance.now() - start;
        const view = await TestView.create(device, scene.platform);
        const images = await view.render(field, camera, job.width, job.height);
        scene.platform.dispose();

        let nonFinite = 0;
        for (const image of [images.indirect, images.full]) for (const value of image) if (!Number.isFinite(value)) nonFinite++;
        const indirectMean = imageMean(images.indirect);
        const fullMean = imageMean(images.full);
        log(`${job.scene} / ${camera.name}: ${job.frames} frames of ${field.grid.probeCount} probes; indirect mean ${indirectMean.toFixed(6)}, full mean ${fullMean.toFixed(6)}`);
        assert(nonFinite === 0, `${nonFinite} non-finite pixel values`);
        assert(indirectMean > 0, 'the test view is black');

        const result: Record<string, unknown> = {
            scene: job.scene,
            camera: camera.name,
            width: job.width,
            height: job.height,
            frames: job.frames,
            probes: field.grid.counts,
            probesPerFrame: field.grid.probeCount,
            ignoreLightRange: job.ignoreLightRange,
            settings: job.settings ?? {},
            distanceFormat: field.layout.distanceFormat,
            indirectMean,
            // DDGI's field counts light seen straight from an emitter as GI; Cycles' indirect pass doesn't.
            emissiveSurfaces: scene.data.materials.some((material) => material.emissive.some((channel) => channel > 0)),
            fullMean,
            convergeMs: Math.round(convergeMs),
        };
        if (job.referenceIndirectMean !== null) {
            const bias = indirectMean / job.referenceIndirectMean - 1;
            result.referenceIndirectMean = job.referenceIndirectMean;
            result.indirectBias = bias;
            assert(Math.abs(bias) <= MEAN_TOLERANCE, `indirect mean ${indirectMean.toFixed(6)} is ${(bias * 100).toFixed(2)}% from the reference's ${job.referenceIndirectMean}, outside ±${MEAN_TOLERANCE * 100}%`);
        }
        if (job.exportImages) {
            result.indirectBase64 = floatsToBase64(images.indirect);
            result.fullBase64 = floatsToBase64(images.full);
        }
        return result;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
