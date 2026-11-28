/**
 * RC4.2 review: what SSGI (the platform's `gi/ssgi`) costs on the GPU at one, two and four slices,
 * on two-room's screen at 1280 × 720 and 1920 × 1080 (its work is per pixel, so the scene matters
 * little). The platform resolves its pass timestamps at the end of `encode`, before the screen
 * stage, so screen-stage passes aren't in `stats.passes` yet; this page times SSGI by itself with
 * its own query set instead: each frame draws the screen, then runs SSGI `RUNS` times between two
 * timestamps (with its running means, as the radiance cache runs it), and reports the median and
 * 90th percentile per run over `MEASURE_FRAMES` frames. A measurement: only a device error fails.
 *
 * Time it cool, alone: `node tools/reference/bench-lock.ts run -- pnpm test:gpu --filter radiance-cache-ssgi-cost --gpu nvidia`.
 */

import { DEFAULT_SCREEN_SPACE_GI_SETTINGS } from '../../../../packages/core/src/platform/contract/compute.ts';
import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const SIZES = [
    [1280, 720],
    [1920, 1080],
] as const;
const SLICES = [1, 2, 4] as const;
const WARM_FRAMES = 10;
const MEASURE_FRAMES = 30;
/** SSGI runs between the two timestamps of a frame. */
const RUNS = 4;

function percentile(values: readonly number[], fraction: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))] ?? NaN;
}

async function measure(device: GPUDevice, width: number, height: number): Promise<Record<string, unknown>> {
    const manifest = loadManifest('two-room');
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `ssgi cost ${width}` });
    await platform.prepare();
    describeScene(platform, manifest);
    const screen = await PageScreen.create(platform, device, width, height);
    const ssgi = await platform.compute.createScreenSpaceGI({ label: `ssgi cost ${width}` });
    const history = await platform.compute.createScreenHistory({ label: `ssgi cost ${width}` });
    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
    const resolveBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let frameNumber = 0;
    const frame = async (slices: number, timed: boolean): Promise<number | null> => {
        const encoder = device.createCommandEncoder({ label: `ssgi cost ${width}` });
        const frameCam = frameCamera(camera, width, height);
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCam };
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        for (let run = 0; run < RUNS; run++) {
            let timestampWrites: GPUComputePassTimestampWrites | undefined;
            if (timed && run === 0) timestampWrites = { querySet, beginningOfPassWriteIndex: 0 };
            if (timed && run === RUNS - 1) timestampWrites = { querySet, endOfPassWriteIndex: 1 };
            ssgi.encode(encoder, {
                screen: screen.screen,
                camera: frameCam,
                frame: frameNumber,
                lighting: screen.screen.lighting ?? null,
                settings: { ...DEFAULT_SCREEN_SPACE_GI_SETTINGS, sliceCount: slices },
                history: history.texturesFor(width, height),
                timestampWrites,
            });
        }
        history.encode(encoder, screen.screen);
        if (timed) {
            encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
            encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
        }
        device.queue.submit([encoder.finish()]);
        frameNumber++;
        if (!timed) return null;
        await readBuffer.mapAsync(GPUMapMode.READ);
        const [begin, end] = new BigUint64Array(readBuffer.getMappedRange().slice(0));
        readBuffer.unmap();
        return Number(end - begin) / 1e6 / RUNS;
    };
    try {
        await frame(1, false);
        await platform.rays.whenBuilt();
        const results: Record<string, unknown> = {};
        for (const slices of SLICES) {
            for (let index = 0; index < WARM_FRAMES; index++) await frame(slices, false);
            await device.queue.onSubmittedWorkDone();
            const times: number[] = [];
            for (let index = 0; index < MEASURE_FRAMES; index++) {
                const ms = await frame(slices, true);
                if (ms !== null && ms >= 0) times.push(ms);
            }
            results[`${slices} slice${slices > 1 ? 's' : ''}`] = { frames: times.length, medianMs: Number(percentile(times, 0.5).toFixed(4)), p90Ms: Number(percentile(times, 0.9).toFixed(4)) };
        }
        return results;
    } finally {
        ssgi.dispose();
        history.dispose();
        querySet.destroy();
        resolveBuffer.destroy();
        readBuffer.destroy();
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    if (!device.features.has('timestamp-query')) throw new Error('this device has no timestamp-query');
    const sizes: Record<string, unknown> = {};
    for (const [width, height] of SIZES) sizes[`${width}x${height}`] = await measure(device, width, height);
    await drainDevice(device);
    reportGpuTest({
        status: deviceErrors.length === 0 ? 'pass' : 'fail',
        data: { scene: 'two-room', runsPerFrame: RUNS, warmFrames: WARM_FRAMES, measureFrames: MEASURE_FRAMES, steps: DEFAULT_SCREEN_SPACE_GI_SETTINGS.stepCount, sizes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
