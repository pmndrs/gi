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
 *   mean over the last `REFERENCE_FRAMES`), for the denoised E and for the raw E it reads (the
 *   radiance cache itself takes many frames to follow a moved light); the denoiser's lag is the
 *   difference.
 *
 * The frame-to-frame change over the first 8 frames after the move shows whether the history
 * is kept (a dropped history shows the raw input's noise; the raw change is reported beside
 * it). Gate (on the run at the defaults: clamp on, reset on): 64 frames after the move the
 * denoised error is at most `ERROR_RATIO_GATE` times the raw input's (the radiance cache itself
 * takes about 60 frames to follow), no NaN or negative values, no device errors.
 *
 * `VITE_RC_COMPARE` (RC4.5, radiance-cache-probes/compare-env.ts) sets every run's other
 * settings, and the gate moves to the run whose clamp and reset it names, so a Capsaicin value
 * can be run against the gate written for it: `VITE_RC_COMPARE=denoiserHistoryClamp=none` gates
 * the run with the clamp off and the reset on.
 *
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-denoiser-change --gpu nvidia`
 */

import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { setCompareValues } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/compare-toggles.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { PAGE_COMPARE_VALUES, pageRadianceCache } from '../radiance-cache-probes/compare-env.ts';
import { readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 640;
const HEIGHT = 360;
const WARM_FRAMES = 60;
const CONVERGE_FRAMES = 120;
const AFTER_FRAMES = 160;
const REFERENCE_FRAMES = 40;
const ERROR_RATIO_GATE = 1.25;
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

interface Curve {
    /** The first frame from which the error stays within 1.25x its final level (null: not before the reference frames). */
    readonly settledFrame: number | null;
    readonly finalError: number;
    readonly errorAtFrame: Record<number, number>;
    /** Mean per-pixel frame-to-frame change over the first 8 frames after the move. */
    readonly meanChangeFirst8: number;
}

interface Run {
    readonly denoised: Curve;
    /** The interpolated irradiance the denoiser reads: how fast the radiance cache itself follows. */
    readonly raw: Curve;
    /**
     * Denoised error over raw error at frame 64 after the move: the radiance cache itself takes
     * about 60 frames to follow the light, and the denoiser can't be faster than its input; this
     * says how much it adds. Settle frames don't compare (the denoised curve keeps falling below
     * the raw one's noise floor).
     */
    readonly errorRatioAt64: number;
    readonly bad: number;
}

async function measure(device: GPUDevice, clamp: 'neighbourhood' | 'none', reset: boolean): Promise<Run> {
    const manifest = loadManifest('two-room');
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: 'radiance-cache-denoiser-change' });
    // The run's compare values (if any) over the defaults, then this run's clamp and reset.
    const technique = platform.addTechnique(pageRadianceCache());
    technique.settings.denoiserHistoryClamp = clamp;
    technique.settings.denoiserChangeReset = reset;
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
        let previousRaw = await readE(device, technique.probes!.irradiance);

        platform.scene.updateLight(lights[0], { position: LAMP_B });
        const denoisedFrames: Float32Array[] = [];
        const rawFrames: Float32Array[] = [];
        const rawSum = new Float64Array(WIDTH * HEIGHT);
        const changes: number[] = [];
        const rawChanges: number[] = [];
        const change = (now: Float32Array, before: Float32Array): number => {
            let sum = 0;
            let count = 0;
            for (let pixel = 0; pixel < now.length; pixel++) {
                if (!surface(pixel)) continue;
                sum += Math.abs(now[pixel] - before[pixel]) / Math.max(before[pixel], 1e-6);
                count++;
            }
            return sum / Math.max(count, 1);
        };
        let bad = 0;
        for (let index = 0; index < AFTER_FRAMES; index++) {
            frame();
            await device.queue.onSubmittedWorkDone();
            const probes = technique.probes!;
            const denoised = await readE(device, probes.denoisedIrradiance!);
            denoisedFrames.push(denoised);
            const raw = await readE(device, probes.irradiance);
            rawFrames.push(raw);
            if (index >= AFTER_FRAMES - REFERENCE_FRAMES) for (let pixel = 0; pixel < raw.length; pixel++) rawSum[pixel] += raw[pixel];
            if (index < 8) {
                changes.push(change(denoised, previous));
                rawChanges.push(change(raw, previousRaw));
            }
            previousRaw = raw;
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
        const curve = (frames: Float32Array[], frameChanges: number[]): Curve => {
            const errors = frames.map((e) => {
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
            const errorAtFrame: Record<number, number> = {};
            for (const at of [1, 2, 4, 8, 16, 32, 64, 120]) errorAtFrame[at] = Number(errors[at - 1].toFixed(4));
            return {
                settledFrame: lastAbove + 2 <= errors.length - REFERENCE_FRAMES ? lastAbove + 2 : null,
                finalError: Number(finalError.toFixed(4)),
                errorAtFrame,
                meanChangeFirst8: Number((frameChanges.reduce((a, b) => a + b, 0) / frameChanges.length).toFixed(4)),
            };
        };
        const denoisedCurve = curve(denoisedFrames, changes);
        const rawCurve = curve(rawFrames, rawChanges);
        return {
            denoised: denoisedCurve,
            raw: rawCurve,
            errorRatioAt64: Number((denoisedCurve.errorAtFrame[64] / rawCurve.errorAtFrame[64]).toFixed(3)),
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
    // The gated run: the defaults, or the clamp and reset VITE_RC_COMPARE names.
    const gatedSettings = { ...DEFAULT_RADIANCE_CACHE_SETTINGS };
    setCompareValues(gatedSettings, PAGE_COMPARE_VALUES);
    const gatedName = `clamp ${gatedSettings.denoiserHistoryClamp}, change reset ${gatedSettings.denoiserChangeReset ? 'on' : 'off'}`;
    const gated = runs[gatedName];
    if (!(gated.errorRatioAt64 <= ERROR_RATIO_GATE)) failures.push(`${gatedName}: denoised error ${gated.errorRatioAt64}x the raw input's at frame 64, gate ${ERROR_RATIO_GATE}`);
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, lampTo: LAMP_B, compare: PAGE_COMPARE_VALUES, gated: gatedName, runs },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
