/**
 * RC4.2 review: what SSGI (`gi/ssgi`, the radiance cache's `radiance-cache/ssgi` pass) costs on the
 * GPU at one, two and four slices, from the platform's pass timestamps, on two-room at 1280 × 720
 * and 1920 × 1080 (SSGI's work is per pixel, so the scene matters little). A measurement: only a
 * missing timestamp or a device error fails. For each size and slice count, after `WARM_FRAMES`
 * frames, the median and 90th percentile of `MEASURE_FRAMES` frames' SSGI pass, beside the
 * irradiance pass (the probe filter, projection and interpolation) for scale.
 *
 * Time it cool, alone: `node tools/reference/bench-lock.ts run -- pnpm test:gpu --filter radiance-cache-ssgi-cost --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const SIZES = [
    [1280, 720],
    [1920, 1080],
] as const;
const SLICES = [1, 2, 4] as const;
const WARM_FRAMES = 30;
const MEASURE_FRAMES = 90;
const PASSES = { ssgi: 'radiance-cache/ssgi', irradiance: 'radiance-cache/irradiance' } as const;

function percentile(values: readonly number[], fraction: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))] ?? NaN;
}

async function measure(device: GPUDevice, width: number, height: number): Promise<Record<string, unknown>> {
    const manifest = loadManifest('two-room');
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `ssgi cost ${width}`, gpuTiming: true });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    describeScene(platform, manifest);
    const screen = await PageScreen.create(platform, device, width, height);
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `ssgi cost ${width}` });
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, width, height) };
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    try {
        frame();
        await platform.rays.whenBuilt();
        const results: Record<string, unknown> = {};
        for (const slices of SLICES) {
            technique.settings.ssgiSliceCount = slices;
            for (let index = 0; index < WARM_FRAMES; index++) frame();
            await device.queue.onSubmittedWorkDone();
            const times: Record<keyof typeof PASSES, Map<number, number>> = { ssgi: new Map(), irradiance: new Map() };
            for (let index = 0; index < MEASURE_FRAMES; index++) {
                frame();
                await device.queue.onSubmittedWorkDone();
                for (const pass of platform.stats.passes) {
                    for (const [key, name] of Object.entries(PASSES) as Array<[keyof typeof PASSES, string]>) {
                        if (pass.name === name && pass.gpuMs !== undefined && pass.gpuMs >= 0) times[key].set(pass.frame, pass.gpuMs);
                    }
                }
            }
            const summary = (values: Map<number, number>) => ({ frames: values.size, medianMs: Number(percentile([...values.values()], 0.5).toFixed(4)), p90Ms: Number(percentile([...values.values()], 0.9).toFixed(4)) });
            results[`${slices} slice${slices > 1 ? 's' : ''}`] = { ssgi: summary(times.ssgi), irradiance: summary(times.irradiance) };
        }
        return results;
    } finally {
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
    const failures: string[] = [];
    for (const [width, height] of SIZES) {
        const results = await measure(device, width, height);
        sizes[`${width}x${height}`] = results;
        for (const [name, value] of Object.entries(results)) {
            const frames = (value as { ssgi: { frames: number } }).ssgi.frames;
            if (frames === 0) failures.push(`${width} × ${height}, ${name}: no SSGI timestamps`);
        }
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { scene: 'two-room', warmFrames: WARM_FRAMES, measureFrames: MEASURE_FRAMES, sizes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
