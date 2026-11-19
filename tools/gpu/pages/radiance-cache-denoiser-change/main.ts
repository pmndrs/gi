/**
 * RC4.3 review: the GI denoiser (`Gi1Denoiser`, gi-denoiser.ts) through a real scene change.
 * Two-room (from its manifest) at 640 × 360 from its first bookmark, camera still; its lamp
 * moves once from the manifest position to the other room's side (`SceneApi.updateLight`,
 * which reports the light's reach box as a changed region, the box D46 and the denoiser's
 * soft reset take).
 *
 * For each setting of `denoiserHistoryClamp` × `denoiserChangeReset`:
 *
 * - converge `CONVERGE_FRAMES` with the lamp at A, move it to B, then run `AFTER_FRAMES`;
 * - the reference is the raw interpolated irradiance averaged over the last `REFERENCE_FRAMES`
 *   (the new lighting, converged and noise-averaged);
 * - per frame after the move: the mean relative error of the denoised E against that reference
 *   (|E − ref| / max(ref, 5% of its mean)), and the frame-to-frame change of the denoised E;
 * - settled: the first frame after which the error stays within 1.25× its final level (the
 *   mean over the last `REFERENCE_FRAMES`).
 *
 * The frame-to-frame change over the first 8 frames after the move shows whether the history
 * is kept (a dropped history shows the raw input's noise). Gate (defaults: clamp on, reset
 * on): settled within `SETTLE_GATE` frames, no NaN or negative values, no device errors.
 *
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-denoiser-change --gpu nvidia`
 */

import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 640;
const HEIGHT = 360;
const WARM_FRAMES = 60;
const CONVERGE_FRAMES = 120;
const AFTER_FRAMES = 160;
const REFERENCE_FRAMES = 40;
const SETTLE_GATE = 16;
const DARK_FLOOR = 0.05;
/** Where the lamp goes: the manifest's `light-step` key in the same room (it starts at [2, 2.45, 2]). */
const LAMP_B: [number, number, number] = [1.0, 1.8, 2.0];

const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));
async function readE(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
    const bytes = await readTexture(device, texture, 8, WIDTH, HEIGHT);
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const e = new Float32Array(WIDTH * HEIGHT);
    for (let pixel = 0; pixel < e.length; pixel++) e[pixel] = (HALF_TABLE[words[pixel * 4]] + HALF_TABLE[words[pixel * 4 + 1]] + HALF_TABLE[words[pixel * 4 + 2]]) / 3;
    return e;
}

interface Run {
    readonly settledFrame: number | null;
    readonly finalError: number;
    readonly errorAtFrame: Record<number, number>;
    readonly meanChangeFirst8: number;
    readonly bad: number;
}

async function measure(device: GPUDevice, clamp: 'neighbourhood' | 'none', reset: boolean): Promise<Run> {
    const manifest = loadManifest('two-room');
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: 'radiance-cache-denoiser-change' });
    const technique = platform.addTechnique(new RadianceCacheTechnique({ denoiserHistoryClamp: clamp, denoiserChangeReset: reset }));
    platform.setOutput(technique);
    await platform.prepare();
    const lights: LightHandle[] = [];
    describeScene(platform, manifest, {}, lights);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: 'radiance-cache-denoiser-change' });
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    try {
        frame();
        await platform.rays.whenBuilt();
        technique.resetHistory();
        for (let index = 0; index < WARM_FRAMES + CONVERGE_FRAMES; index++) {
            frame();
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();
        const depth = new Float32Array((await readTexture(device, screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
        const surface = (pixel: number): boolean => depth[pixel] !== 1;
        let previous = await readE(device, technique.probes!.denoisedIrradiance!);

        platform.scene.updateLight(lights[0], { position: LAMP_B });
        const denoisedFrames: Float32Array[] = [];
        const rawSum = new Float64Array(WIDTH * HEIGHT);
        const changes: number[] = [];
        let bad = 0;
        for (let index = 0; index < AFTER_FRAMES; index++) {
            frame();
            await device.queue.onSubmittedWorkDone();
            const probes = technique.probes!;
            const denoised = await readE(device, probes.denoisedIrradiance!);
            denoisedFrames.push(denoised);
            if (index >= AFTER_FRAMES - REFERENCE_FRAMES) {
                const raw = await readE(device, probes.irradiance);
                for (let pixel = 0; pixel < raw.length; pixel++) rawSum[pixel] += raw[pixel];
            }
            if (index < 8) {
                let sum = 0;
                let count = 0;
                for (let pixel = 0; pixel < denoised.length; pixel++) {
                    if (!surface(pixel)) continue;
                    sum += Math.abs(denoised[pixel] - previous[pixel]) / Math.max(previous[pixel], 1e-6);
                    count++;
                }
                changes.push(sum / Math.max(count, 1));
            }
            for (let pixel = 0; pixel < denoised.length; pixel++) if (surface(pixel) && !(Number.isFinite(denoised[pixel]) && denoised[pixel] >= 0)) bad++;
            previous = denoised;
        }
        const reference = Float64Array.from(rawSum, (sum) => sum / REFERENCE_FRAMES);
        let referenceMean = 0;
        let surfaceCount = 0;
        for (let pixel = 0; pixel < reference.length; pixel++) {
            if (!surface(pixel)) continue;
            referenceMean += reference[pixel];
            surfaceCount++;
        }
        const floor = (DARK_FLOOR * referenceMean) / Math.max(surfaceCount, 1);
        const errors = denoisedFrames.map((e) => {
            let sum = 0;
            for (let pixel = 0; pixel < e.length; pixel++) if (surface(pixel)) sum += Math.abs(e[pixel] - reference[pixel]) / Math.max(reference[pixel], floor);
            return sum / Math.max(surfaceCount, 1);
        });
        const finalError = errors.slice(-REFERENCE_FRAMES).reduce((a, b) => a + b, 0) / REFERENCE_FRAMES;
        // The first frame from which the error stays within 1.25x its final level.
        let lastAbove = -1;
        errors.forEach((error, index) => {
            if (error > 1.25 * finalError) lastAbove = index;
        });
        const settledFrame = lastAbove + 2 <= errors.length - REFERENCE_FRAMES ? lastAbove + 2 : null;
        const errorAtFrame: Record<number, number> = {};
        for (const at of [1, 2, 4, 8, 16, 32, 64, 120]) errorAtFrame[at] = Number(errors[at - 1].toFixed(4));
        return {
            settledFrame,
            finalError: Number(finalError.toFixed(4)),
            errorAtFrame,
            meanChangeFirst8: Number((changes.reduce((a, b) => a + b, 0) / changes.length).toFixed(4)),
            bad,
        };
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
    const runs: Record<string, Run> = {};
    for (const [clamp, reset] of [['neighbourhood', true], ['neighbourhood', false], ['none', true], ['none', false]] as const) {
        runs[`clamp ${clamp}, change reset ${reset ? 'on' : 'off'}`] = await measure(device, clamp, reset);
    }
    const failures: string[] = [];
    for (const [name, run] of Object.entries(runs)) if (run.bad > 0) failures.push(`${name}: ${run.bad} bad values`);
    const defaults = runs['clamp neighbourhood, change reset on'];
    if (defaults.settledFrame === null || defaults.settledFrame > SETTLE_GATE) failures.push(`defaults settle at frame ${defaults.settledFrame}, gate ${SETTLE_GATE}`);
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, lampTo: LAMP_B, runs },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
