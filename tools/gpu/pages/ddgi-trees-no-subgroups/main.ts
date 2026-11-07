/**
 * T2.12 check 4: a device without subgroups still gets tinybvh trees and GI.
 *
 * The page asks for no optional features, so the device has no `subgroups`
 * (the adapter may have them; the device doesn't), as on a browser or GPU
 * that lacks them. There the GPU's first trees (H-PLOC) can't run, so:
 *
 * - the platform builds no GPU trees (`rays.gpuFirstTrees` false, no GPU
 *   tree built) and every mesh gets tinybvh's tree, as in M1;
 * - the top level is the CPU fallback (B1a's declared fallback);
 * - DDGI runs: Cornell, 600 frames at the RTX's calibrated batch as
 *   `ddgi-change-still` runs them, and the irradiance atlas's mean is
 *   within 1% of M1's on the same GPU vendor (that page's M1 numbers, made
 *   with subgroups; the kernels' fallbacks differ by float rounding only,
 *   NOTES) and the test view's indirect light isn't black.
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, halfToFloat, imageMean, readTextureBytes, releaseDeviceOnPageHide, runFrames, TestView } from '../ddgi-shared/harness.ts';

const FRAMES = 600;
/** Cornell at the RTX's calibrated batch at 2 ms (T1.9b), as `ddgi-change-still`. */
const UNITS = 1574;
/** `ddgi-change-still`'s M1 atlas means for Cornell after 600 frames (PROGRESS 2026-11-06, T2.4a). */
const M1_ATLAS_MEAN: Readonly<Record<string, number>> = { nvidia: 0.33536604567456785, intel: 0.3350773758127751 };
const ATLAS_TOLERANCE = 0.01;

async function atlasMean(device: GPUDevice, texture: GPUTexture): Promise<number> {
    const halves = new Uint16Array((await readTextureBytes(device, texture, 8)).buffer);
    let sum = 0;
    for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
    return sum / ((halves.length / 4) * 3);
}

await runGpuTest(async ({ adapter, device, log }) => {
    releaseDeviceOnPageHide(device);
    assert(!device.features.has('subgroups'), 'the device was made with subgroups; this page needs a device without');
    const vendor = /nvidia/i.test(adapter.info.vendor) ? 'nvidia' : /intel/i.test(adapter.info.vendor) ? 'intel' : adapter.info.vendor;

    const scene = await createDdgiScene(device, 'cornell-box', {}, { gpuTiming: true });
    const rays = scene.platform.rays;
    const counts = rays.meshTreeCounts;
    const meshCount = counts.onGpuTrees + counts.onTinybvhTrees;
    log(`trees: ${counts.onTinybvhTrees} of ${meshCount} meshes on tinybvh trees, ${counts.gpuTreesBuilt} GPU trees built; top level ${rays.topLevelPath}`);
    assert(!rays.gpuFirstTrees, 'the platform says it builds first trees on the GPU without subgroups');
    assert(counts.gpuTreesBuilt === 0 && counts.onGpuTrees === 0, `${counts.gpuTreesBuilt} GPU trees were built without subgroups`);
    assert(meshCount > 0 && counts.onTinybvhTrees === meshCount, `only ${counts.onTinybvhTrees} of ${meshCount} meshes have tinybvh trees`);
    assert(rays.topLevelPath === 'cpu', `the top level took the ${rays.topLevelPath} path without subgroups`);

    const field = scene.ddgi.field;
    assert(field, 'no field after prepare');
    await runFrames(device, scene.platform, FRAMES, UNITS);
    const fieldMean = await atlasMean(device, field.irradiance);
    const view = await TestView.create(device, scene.platform);
    const viewMean = imageMean((await view.render(field, scene.manifest.cameras[0], 320, 180)).indirect);
    const recompiles = scene.platform.stats.recompiles;
    scene.platform.dispose();

    const reference = M1_ATLAS_MEAN[vendor];
    const change = reference === undefined ? null : fieldMean / reference - 1;
    log(`after ${FRAMES} frames: atlas mean ${fieldMean.toPrecision(7)}${change === null ? '' : ` (M1 with subgroups ${reference.toPrecision(7)}: ${(change * 100).toFixed(3)}%)`}, view mean ${viewMean.toPrecision(5)}; recompiles ${recompiles}`);
    assert(fieldMean > 0 && viewMean > 0, 'no GI: the field or the view is black');
    if (change !== null) assert(Math.abs(change) <= ATLAS_TOLERANCE, `the field's mean is ${(change * 100).toFixed(3)}% from M1's (bar 1%)`);
    assert(recompiles === 0, `${recompiles} compute pipelines were made while ready`);
    return { vendor, meshes: meshCount, trees: counts, topLevelPath: rays.topLevelPath, atlasMean: fieldMean, m1AtlasMean: reference ?? null, change, viewMean, recompiles };
});
