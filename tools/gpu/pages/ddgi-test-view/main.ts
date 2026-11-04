/**
 * The engine-free test view of DDGI's indirect light (T1.5a done-when 2,
 * gated on the manifest's volume since T1.5b).
 *
 * Converges DDGI on a manifest scene (every probe every frame), then renders
 * the test view of one camera bookmark (ddgi-shared/harness.ts, `TestView`):
 * `albedo / π × ddgiSampleIrradiance(...)` per pixel, plus the full image.
 *
 * Two ways to run it:
 * - `pnpm test:gpu --filter ddgi-test-view`: the point-light Cornell box at
 *   bookmark 1, with the light's range window off as in Cycles, on the
 *   manifest's own volume (DECISIONS 2026-11-05). Fails if the image's mean
 *   is not within ±10% of the Cycles indirect reference's mean (the M1
 *   accuracy target), or has non-finite pixels. It also reports, without
 *   gating, the volume with every probe in the air near the surfaces
 *   (`CORNELL_POINT_IN_AIR_VOLUME`, T1.5a's measurement).
 * - `node tools/gpu/pages/ddgi-test-view/render.ts`: any scene and camera; the
 *   script injects `window.ddgiTestViewJob`, takes the images back, writes
 *   EXRs and runs `tools/compare` on them.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { DDGISettingsInput, DDGIVolumeSettings } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { createDdgiScene, floatsToBase64, imageMean, runFrames, TestView, type TestViewImages, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { CORNELL_POINT_IN_AIR_VOLUME } from '../ddgi-shared/volumes.ts';

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
    /** More volumes to render and report against the reference, without gating. */
    readonly extraVolumes?: readonly { readonly label: string; readonly volume: DDGIVolumeSettings }[];
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
    extraVolumes: [{ label: 'in air', volume: CORNELL_POINT_IN_AIR_VOLUME }],
};

declare global {
    interface Window {
        ddgiTestViewJob?: TestViewJob;
    }
}

const job: TestViewJob = window.ddgiTestViewJob ?? DEFAULT_JOB;

interface Rendered {
    readonly images: TestViewImages;
    readonly indirectMean: number;
    readonly fullMean: number;
    readonly probes: readonly number[];
    readonly distanceFormat: string;
    readonly emissiveSurfaces: boolean;
    readonly convergeMs: number;
    readonly camera: string;
}

/** Converges DDGI with these settings and renders the job's camera. */
async function render(device: GPUDevice, settings: Partial<DDGISettingsInput>): Promise<Rendered> {
    const scene = await createDdgiScene(device, job.scene, settings, { ignoreLightRange: job.ignoreLightRange });
    try {
        const field = scene.ddgi.field;
        assert(field, 'DDGI has no field after prepare');
        const camera = scene.manifest.cameras[job.camera - 1];
        assert(camera, `${job.scene} has no camera ${job.camera}`);
        const start = performance.now();
        await runFrames(device, scene.platform, job.frames, field.grid.probeCount);
        const convergeMs = Math.round(performance.now() - start);
        const images = await (await TestView.create(device, scene.platform)).render(field, camera, job.width, job.height);
        let nonFinite = 0;
        for (const image of [images.indirect, images.full]) for (const value of image) if (!Number.isFinite(value)) nonFinite++;
        assert(nonFinite === 0, `${nonFinite} non-finite pixel values`);
        const indirectMean = imageMean(images.indirect);
        assert(indirectMean > 0, 'the test view is black');
        return {
            images,
            indirectMean,
            fullMean: imageMean(images.full),
            probes: field.grid.counts,
            distanceFormat: field.layout.distanceFormat,
            // DDGI's field counts light seen straight from an emitter as GI; Cycles' indirect pass doesn't.
            emissiveSurfaces: scene.data.materials.some((material) => material.emissive.some((channel) => channel > 0)),
            convergeMs,
            camera: camera.name,
        };
    } finally {
        scene.platform.dispose();
    }
}

function percent(value: number | null): string {
    return value === null ? 'no reference' : `${(value * 100).toFixed(2)}%`;
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const bias = (mean: number): number | null => (job.referenceIndirectMean === null ? null : mean / job.referenceIndirectMean - 1);
        const primary = await render(device, job.settings ?? {});
        log(`${job.scene} / ${primary.camera}: ${job.frames} frames; indirect mean ${primary.indirectMean.toFixed(6)} (${percent(bias(primary.indirectMean))}), full mean ${primary.fullMean.toFixed(6)}`);
        const result: Record<string, unknown> = {
            scene: job.scene,
            camera: primary.camera,
            width: job.width,
            height: job.height,
            frames: job.frames,
            probes: primary.probes,
            ignoreLightRange: job.ignoreLightRange,
            settings: job.settings ?? {},
            distanceFormat: primary.distanceFormat,
            indirectMean: primary.indirectMean,
            emissiveSurfaces: primary.emissiveSurfaces,
            fullMean: primary.fullMean,
            convergeMs: primary.convergeMs,
            referenceIndirectMean: job.referenceIndirectMean,
            indirectBias: bias(primary.indirectMean),
        };

        // Extra volumes: reported, never gating.
        const extras: Record<string, unknown> = {};
        for (const extra of job.extraVolumes ?? []) {
            const rendered = await render(device, { ...job.settings, volume: extra.volume });
            extras[extra.label] = { indirectMean: rendered.indirectMean, indirectBias: bias(rendered.indirectMean), fullMean: rendered.fullMean };
            log(`${extra.label} (not gating): indirect mean ${rendered.indirectMean.toFixed(6)} (${percent(bias(rendered.indirectMean))})`);
        }
        result.extraVolumes = extras;

        if (job.exportImages) {
            result.indirectBase64 = floatsToBase64(primary.images.indirect);
            result.fullBase64 = floatsToBase64(primary.images.full);
        }
        const primaryBias = bias(primary.indirectMean);
        if (primaryBias !== null) {
            assert(
                Math.abs(primaryBias) <= MEAN_TOLERANCE,
                `indirect mean ${primary.indirectMean.toFixed(6)} is ${percent(primaryBias)} from the reference's ${job.referenceIndirectMean}, outside ±${MEAN_TOLERANCE * 100}%`,
            );
        }
        return result;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
