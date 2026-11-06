/**
 * T2.4a done-whens 1 and 3: on a still scene, change-driven history must
 * change nothing.
 *
 * - **Check 1 (the field):** after FRAMES frames the field matches M1's
 *   (T1.12's rule, commit db431f6) within 0.5% mean. Two numbers per scene:
 *   the mean of the irradiance atlas over every probe and texel (the whole
 *   field, whatever the camera sees), and the mean of the engine-free test
 *   view's indirect image at the scene's first bookmark (what a viewer sees).
 *   The M1 numbers were measured with this page at db431f6 on each GPU
 *   (M1_REFERENCE below; PROGRESS 2026-11-06, T2.4a). DDGI seeds its rays by
 *   frame number, so a still scene that marks nothing should come out
 *   practically identical, not just within 0.5%.
 * - **Check 3 (the counter):** the technique's "probes marked (total)"
 *   counter stays 0 over the run. A still scene has no light changes, so
 *   `ddgi/mark` never runs; the counter is the GPU's own count of probes it
 *   marked, so a mark that ran on a still frame would show here too.
 *
 * Each scene: DDGI on the manifest's volume, default settings, FRAMES frames
 * of UNITS[scene] probes (the RTX's calibrated batch at 2 ms, T1.9b, as the
 * T1.12 flicker check used), deltaTime 1/60 s (a 60 Hz host), census on
 * (`gpuTiming`), so the counters are read back.
 *
 * Runs on M1's code too (no counter there): the counter is then reported as
 * absent and not checked. That is how M1_REFERENCE was measured.
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, halfToFloat, imageMean, readTextureBytes, releaseDeviceOnPageHide, runFrames, TestView } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

/** Frames per scene: the done-when's 600. */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 600;
/** Probes a frame per scene: the RTX's calibrated batch at 2 ms (T1.9b medians), the T1.12 flicker check's fixed counts. */
const UNITS: Readonly<Record<string, number>> = { 'cornell-box': 1574, 'two-room': 707 };
const SCENES = Object.keys(UNITS);
const WIDTH = 640;
const HEIGHT = 360;
/** Check 1's bar: the field's mean within 0.5% of M1's. */
const MEAN_TOLERANCE = 0.005;
const MARKED_TOTAL = 'probes marked (total)';

/**
 * This page's numbers at M1 (commit db431f6, T1.12's rule), per GPU vendor:
 * the irradiance atlas's mean and the test view's indirect mean after FRAMES
 * frames. Measured 2026-11-06 (PROGRESS, T2.4a): RTX 3060 Laptop and Intel
 * UHD, Chrome 156, headless.
 */
const M1_REFERENCE: Readonly<Record<string, Readonly<Record<string, { atlas: number; view: number }>>>> = {
    // Two runs on the RTX gave the same digits (the seeds are the frame number).
    nvidia: {
        'cornell-box': { atlas: 0.33536604567456785, view: 0.0332847894315802 },
        'two-room': { atlas: 1.6783929207532382, view: 0.7109995752508513 },
    },
    intel: {
        'cornell-box': { atlas: 0.3350773758127751, view: 0.03325704779517551 },
        'two-room': { atlas: 1.688233890406929, view: 0.7090814875086249 },
    },
};

/** Mean of rgb over every texel of an rgba16float atlas (borders included: they copy interior texels, the same in both runs). */
async function atlasMean(device: GPUDevice, texture: GPUTexture): Promise<number> {
    const halves = new Uint16Array((await readTextureBytes(device, texture, 8)).buffer);
    let sum = 0;
    for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
    return sum / ((halves.length / 4) * 3);
}

await runGpuTest(
    async ({ adapter, device, log }) => {
        releaseDeviceOnPageHide(device);
        const vendor = /nvidia/i.test(adapter.info.vendor) ? 'nvidia' : /intel/i.test(adapter.info.vendor) ? 'intel' : adapter.info.vendor;
        const references = M1_REFERENCE[vendor] ?? {};
        const results: Record<string, unknown> = { frames: FRAMES, vendor };
        const failures: string[] = [];
        for (const sceneId of SCENES) {
            const scene = await createDdgiScene(device, sceneId, {}, { gpuTiming: true });
            const field = scene.ddgi.field;
            assert(field, 'no field after prepare');
            await runFrames(device, scene.platform, FRAMES, UNITS[sceneId]);
            // The census arrives a few frames late: run a few frames with no probe updates so the last counts land.
            await runFrames(device, scene.platform, 4, 0);
            const view = await TestView.create(device, scene.platform);
            const viewMean = imageMean((await view.render(field, scene.manifest.cameras[0], WIDTH, HEIGHT)).indirect);
            const fieldMean = await atlasMean(device, field.irradiance);
            const counters = scene.ddgi.counters;
            scene.platform.dispose();

            const marked = counters[MARKED_TOTAL];
            const reference = references[sceneId];
            const atlasChange = reference ? fieldMean / reference.atlas - 1 : null;
            const viewChange = reference ? viewMean / reference.view - 1 : null;
            results[sceneId] = { units: UNITS[sceneId], atlasMean: fieldMean, viewMean, m1: reference ?? null, atlasChange, viewChange, probesMarkedTotal: marked ?? null, counters };
            log(
                `${sceneId}: atlas mean ${fieldMean.toPrecision(7)}, view mean ${viewMean.toPrecision(7)}` +
                    (reference ? ` (M1 ${reference.atlas.toPrecision(7)} / ${reference.view.toPrecision(7)}: ${(atlasChange! * 100).toFixed(4)}% / ${(viewChange! * 100).toFixed(4)}%)` : ' (no M1 number for this GPU)') +
                    `; probes marked ${marked ?? 'not counted (M1 code)'}`,
            );
            if (marked !== undefined && marked !== 0) failures.push(`${sceneId}: a still scene marked ${marked} probes`);
            if (atlasChange !== null && Math.abs(atlasChange) > MEAN_TOLERANCE) failures.push(`${sceneId}: the field's mean moved ${(atlasChange * 100).toFixed(3)}% from M1's`);
            if (viewChange !== null && Math.abs(viewChange) > MEAN_TOLERANCE) failures.push(`${sceneId}: the view's mean moved ${(viewChange * 100).toFixed(3)}% from M1's`);
        }
        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
