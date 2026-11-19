/**
 * P6.2b, experiment E9 (`cellChangeReset`): how noisy the radiance cache's
 * irradiance is in the frames after a light change, with E9 off, on (a
 * marked cell's history dropped), and on keeping 1 or 3 samples of history
 * (`cellChangeResetHistory`). A measurement (not a gate): only NaN, infinite
 * or negative values fail.
 *
 * Scenes as radiance-cache-stability loads them, at 1280 × 720 from their
 * first bookmark, the camera still, screen-probe history on, the
 * interpolation jitter off: Sponza (with curtains, through the viewer's
 * loader) with its sun doubled and, separately, its point light doubled;
 * two-room with its lamp doubled.
 *
 * Per case and setting: `SETTLE_FRAMES` frames at the light's own intensity
 * (past E9's 16-frame window), `BASELINE_FRAMES` frames read, the light
 * doubled, `AFTER_FRAMES` frames read, the light put back. Per surface pixel
 * of the interpolated irradiance E (the image the composite reads; there is
 * no denoiser yet):
 *
 * - **noise:** |E_t − (E_t−1 + E_t+1) / 2| / max(E_t, 5% of the frame's mean),
 *   the second difference, which a steady rise doesn't move, so it measures
 *   the jumps, not the lag; its mean and 95th percentile before the change
 *   and over the 16 frames after it (E9's window);
 * - **step:** |E_t − E_t−1| / max(E_t−1, 5% of the mean), over the same
 *   frames (the rise included);
 * - the frame's mean E over the mean before, at frames 1, 4 and 16 after.
 *
 * Through the lock: `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-change-noise --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const WARM_FRAMES = 60;
const SETTLE_FRAMES = 60;
const BASELINE_FRAMES = 10;
/** E9's window (16) and one more, for the last frame's second difference. */
const AFTER_FRAMES = 18;
const WINDOW = 16;
const DARK_FLOOR = 0.05;

const SETTINGS: readonly { readonly name: string; readonly settings: Partial<RadianceCacheSettings> }[] = [
    { name: 'E9 off', settings: { cellChangeReset: false, cellChangeResetHistory: 0 } },
    { name: 'E9 on', settings: { cellChangeReset: true, cellChangeResetHistory: 0 } },
    { name: 'E9 on, 1 kept', settings: { cellChangeReset: true, cellChangeResetHistory: 1 } },
    { name: 'E9 on, 3 kept', settings: { cellChangeReset: true, cellChangeResetHistory: 3 } },
];

const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));

/** Relative values pooled over pixels and frames: their sum and a histogram (steps of 0.0005 up to 5) for the percentile. */
class Tally {
    static readonly STEP = 0.0005;
    readonly bins = new Uint32Array(10001);
    sum = 0;
    count = 0;
    bad = 0;

    add(value: number): void {
        if (!Number.isFinite(value)) {
            this.bad++;
            return;
        }
        this.sum += value;
        this.count++;
        this.bins[Math.min(Math.floor(value / Tally.STEP), this.bins.length - 1)]++;
    }

    get mean(): number {
        return this.sum / Math.max(this.count, 1);
    }

    percentile(fraction: number): number {
        const target = fraction * this.count;
        let running = 0;
        for (let bin = 0; bin < this.bins.length; bin++) {
            running += this.bins[bin];
            if (running >= target) return (bin + 1) * Tally.STEP;
        }
        return Infinity;
    }
}

interface Frame {
    readonly e: Float32Array;
    readonly mean: number;
}

/** The scene's light whose intensity the case doubles: set its intensity (and the frame picks it up). */
type LightControl = (scale: number) => void;

interface Loaded {
    readonly platform: GIPlatformImpl;
    readonly technique: RadianceCacheTechnique;
    readonly screen: PageScreen;
    readonly lights: Record<string, LightControl>;
    frame(): void;
    read(): Promise<Frame>;
    dispose(): void;
}

async function load(device: GPUDevice, id: string): Promise<Loaded> {
    const manifest: SceneManifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-change-noise ${id}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique({ interpolationJitter: false }));
    platform.setOutput(technique);
    await platform.prepare();
    let beforeFrame = (): void => {};
    const lights: Record<string, LightControl> = {};
    if (id === 'sponza') {
        // Through the viewer's loader and the three adapter's scene sync; the lights are three's.
        const loaded = await loadScene(manifest, () => {});
        const content = new Set<object>();
        loaded.content.traverse((object) => content.add(object));
        loaded.scene.traverse((object) => {
            if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
        });
        loaded.scene.updateMatrixWorld();
        const sync = new SceneSync(platform.scene, { device, warn: () => {} });
        sync.environment = platformSkyEnvironment(manifest);
        beforeFrame = () => sync.update(loaded.scene);
        loaded.scene.traverse((object) => {
            const light = object as unknown as { isDirectionalLight?: boolean; isPointLight?: boolean; intensity: number };
            const kind = light.isDirectionalLight ? 'sun' : light.isPointLight ? 'point' : null;
            if (!kind || lights[kind]) return;
            const base = light.intensity;
            lights[kind] = (scale) => {
                light.intensity = base * scale;
            };
        });
    } else {
        const handles: LightHandle[] = [];
        describeScene(platform, manifest, {}, handles);
        const base = manifest.lights[0];
        lights.point = (scale) => platform.scene.updateLight(handles[0], { intensity: base.intensity * scale });
    }
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        beforeFrame();
        const encoder = device.createCommandEncoder({ label: `radiance-cache-change-noise ${id}` });
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    const depth = { values: null as Float32Array | null };
    const read = async (): Promise<Frame> => {
        await device.queue.onSubmittedWorkDone();
        const probes = technique.probes;
        if (!probes) throw new Error('no probes after encodeScreen');
        const bytes = await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT);
        const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        depth.values ??= new Float32Array((await readTexture(device, screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
        const e = new Float32Array(WIDTH * HEIGHT);
        let sum = 0;
        let count = 0;
        for (let pixel = 0; pixel < e.length; pixel++) {
            if (depth.values[pixel] === 1) {
                e[pixel] = NaN;
                continue;
            }
            e[pixel] = (HALF_TABLE[halves[pixel * 4]] + HALF_TABLE[halves[pixel * 4 + 1]] + HALF_TABLE[halves[pixel * 4 + 2]]) / 3;
            sum += e[pixel];
            count++;
        }
        return { e, mean: sum / Math.max(count, 1) };
    };
    frame();
    await platform.rays.whenBuilt();
    technique.resetHistory();
    for (let index = 0; index < WARM_FRAMES; index++) {
        frame();
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }
    return {
        platform,
        technique,
        screen,
        lights,
        frame,
        read,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

/** The second difference at frame index `t` of `frames` into `tally`, and the step from t-1 into `steps`. */
function tallyFrame(frames: readonly Frame[], t: number, noise: Tally, steps: Tally | null): number {
    const [before, now, after] = [frames[t - 1], frames[t], frames[t + 1]];
    const floor = DARK_FLOOR * now.mean;
    const stepFloor = DARK_FLOOR * before.mean;
    let bad = 0;
    for (let pixel = 0; pixel < now.e.length; pixel++) {
        const value = now.e[pixel];
        if (Number.isNaN(value)) continue;
        if (!(Number.isFinite(value) && value >= 0)) {
            bad++;
            continue;
        }
        noise.add(Math.abs(value - (before.e[pixel] + after.e[pixel]) / 2) / Math.max(value, floor));
        steps?.add(Math.abs(value - before.e[pixel]) / Math.max(before.e[pixel], stepFloor));
    }
    return bad;
}

async function measureCase(scene: Loaded, light: string, settings: Partial<RadianceCacheSettings>): Promise<Record<string, number>> {
    const control = scene.lights[light];
    if (!control) throw new Error(`no ${light} light in the scene`);
    Object.assign(scene.technique.settings, settings);
    control(1);
    for (let index = 0; index < SETTLE_FRAMES; index++) scene.frame();
    const baseline: Frame[] = [];
    for (let index = 0; index < BASELINE_FRAMES; index++) {
        scene.frame();
        baseline.push(await scene.read());
    }
    // The last baseline frame is the "before" (t = 0) of the frames after.
    control(2);
    const after: Frame[] = [baseline[baseline.length - 1]];
    for (let index = 0; index < AFTER_FRAMES; index++) {
        scene.frame();
        after.push(await scene.read());
    }
    control(1);
    const baselineNoise = new Tally();
    let bad = 0;
    for (let t = 1; t < baseline.length - 1; t++) bad += tallyFrame(baseline, t, baselineNoise, null);
    const noise = new Tally();
    const steps = new Tally();
    for (let t = 1; t <= WINDOW; t++) bad += tallyFrame(after, t, noise, steps);
    const before = baseline.reduce((sum, frame) => sum + frame.mean, 0) / baseline.length;
    const round = (value: number, digits = 4): number => Number(value.toFixed(digits));
    return {
        baselineNoiseMean: round(baselineNoise.mean),
        baselineNoiseP95: round(baselineNoise.percentile(0.95)),
        noiseMean: round(noise.mean),
        noiseP95: round(noise.percentile(0.95)),
        stepMean: round(steps.mean),
        stepP95: round(steps.percentile(0.95)),
        meanE1: round(after[1].mean / before),
        meanE4: round(after[4].mean / before),
        meanE16: round(after[16].mean / before),
        bad,
    };
}

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    const results: Record<string, Record<string, Record<string, number>>> = {};
    const failures: string[] = [];
    for (const [id, cases] of [['sponza', ['sun', 'point']], ['two-room', ['point']]] as const) {
        const scene = await load(device, id);
        try {
            for (const light of cases) {
                const name = `${id} ${light} doubled`;
                results[name] = {};
                for (const { name: setting, settings } of SETTINGS) {
                    const measured = await measureCase(scene, light, settings);
                    results[name][setting] = measured;
                    if (measured.bad > 0) failures.push(`${name}, ${setting}: ${measured.bad} pixels' E is NaN, infinite or negative`);
                }
            }
        } finally {
            scene.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, settleFrames: SETTLE_FRAMES, window: WINDOW, results },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
