/**
 * T1.6: DDGI and the hello renderer on a compatibility-mode device
 * (`featureLevel: 'compatibility'`, no `core-features-and-limits`), where
 * fragment shaders may bind no storage buffers and textures bind only with
 * the view dimension they were created for.
 *
 * Asks for a compatibility adapter, then a device with `requiredLimits`
 * alone, as a host would: it raises the storage buffers per stage
 * (compatibility defaults allow 4; DDGI's trace kernel needs 7) and the
 * compute workgroup limits (compatibility defaults allow 128 threads; the
 * platform's largest kernels use 256). The fragment-stage limits stay at the
 * compatibility defaults, which is what this page is about. Converges the point-light Cornell box and
 * checks the hello renderer's GI-only image against the engine-free test
 * view, as ddgi-shading-consistency does, with the same tolerances.
 *
 * Fails, saying so, when the browser hands out a core device instead (then
 * compatibility mode is untested here).
 */

import { assert, drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { ForwardRenderer } from '../../../../examples/webgpu/hello/src/renderer.ts';
import { imageMean, releaseDeviceOnPageHide, runFrames, TestView } from '../ddgi-shared/harness.ts';
import { createHelloScene, readRgba32FloatAsRgb } from '../ddgi-shading-shared/hello-scene.ts';

const SCENE = 'cornell-box-point';
const CAMERA = 1;
const SIZE = 256;
const FRAMES = 300;
/** As ddgi-shading-consistency: means within 0.2%, at most 1% of pixels off by more than 1% (or 1% of the mean). */
const MEAN_TOLERANCE = 0.002;
const PIXEL_TOLERANCE = 0.01;
const PIXEL_FLOOR_OF_MEAN = 0.01;
const MAX_DIFFERING_SHARE = 0.01;

const startTime = performance.now();
const logLines: string[] = [];
let deviceErrors: string[] = [];
let adapterSummary = null;
try {
    assert(navigator.gpu, 'WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ featureLevel: 'compatibility' });
    assert(adapter, 'requestAdapter({ featureLevel: "compatibility" }) returned null');
    const features = [...optionalFeatures(adapter)];
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: features });
    deviceErrors = watchDevice(device);
    releaseDeviceOnPageHide(device);
    adapterSummary = summarizeAdapterInfo(device.adapterInfo, device.features);
    assert(!device.features.has('core-features-and-limits'), 'the browser gave a core device, not a compatibility-mode one; compatibility mode is untested here');
    logLines.push(
        `compatibility device: ${device.adapterInfo.vendor} ${device.adapterInfo.architecture}; fragment storage buffers allowed: ${device.limits.maxStorageBuffersInFragmentStage ?? 'n/a'}; ` +
            `storage buffers per stage ${device.limits.maxStorageBuffersPerShaderStage}; workgroup ${device.limits.maxComputeInvocationsPerWorkgroup} threads, ${device.limits.maxComputeWorkgroupStorageSize} bytes`,
    );

    const { platform, ddgi, scene } = await createHelloScene(device, SCENE, { ignoreLightRange: true });
    const field = ddgi.field;
    assert(field, 'DDGI has no field after prepare');
    await runFrames(device, platform, FRAMES, field.grid.probeCount);
    const camera = scene.manifest.cameras[CAMERA - 1];
    const testView = await (await TestView.create(device, platform)).render(field, camera, SIZE, SIZE);
    const renderer = new ForwardRenderer(device, scene, { shading: platform.shading, hdrFormat: 'rgba32float' });
    renderer.resize(SIZE, SIZE);
    const encoder = device.createCommandEncoder({ label: 'hello render' });
    renderer.render(encoder, renderer.camera(camera), { mode: 'gi' });
    device.queue.submit([encoder.finish()]);
    const raster = await readRgba32FloatAsRgb(device, renderer.hdrTexture!);

    const referenceMean = imageMean(testView.indirect);
    const floor = PIXEL_FLOOR_OF_MEAN * referenceMean;
    let differing = 0;
    for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
        for (let channel = 0; channel < 3; channel++) {
            const a = raster[pixel * 3 + channel];
            const b = testView.indirect[pixel * 3 + channel];
            assert(Number.isFinite(a), 'non-finite pixel in the raster image');
            if (Math.abs(a - b) > Math.max(PIXEL_TOLERANCE * Math.abs(b), floor)) {
                differing++;
                break;
            }
        }
    }
    const meanDifference = imageMean(raster) / referenceMean - 1;
    logLines.push(`raster against rays: mean ${(meanDifference * 100).toFixed(4)}%, ${differing} pixels differ`);
    assert(referenceMean > 0, 'the test view is black');
    assert(Math.abs(meanDifference) <= MEAN_TOLERANCE, `means differ by ${(meanDifference * 100).toFixed(3)}%`);
    assert(differing / (SIZE * SIZE) <= MAX_DIFFERING_SHARE, `${differing} pixels differ`);
    renderer.destroy();
    platform.dispose();
    await drainDevice(device);
    reportGpuTest({
        status: 'pass',
        data: { scene: SCENE, frames: FRAMES, size: SIZE, rasterMeanDifference: meanDifference, differingPixels: differing, features },
        adapter: adapterSummary,
        deviceErrors,
        startTime,
        logLines,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), adapter: adapterSummary, deviceErrors, startTime, logLines });
}
