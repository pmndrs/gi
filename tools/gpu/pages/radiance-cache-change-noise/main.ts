/**
 * P6.2b, experiment E9 (`cellChangeReset`): what the change reset costs in
 * noise, and how fast the image follows, on real scenes, with E9 off, on (a
 * marked cell's history dropped) and on keeping 3 samples of history
 * (`cellChangeResetHistory: 3`). Since the review (2026-11-19), cells and
 * probes forget their history once per epoch of a change event
 * (`ChangeEventTracker`, platform contract): at the start, every 8 frames
 * while it goes on, and once when it stops.
 *
 * Scenes as radiance-cache-stability loads them, at 1280 × 720 from their
 * first bookmark, the camera still, screen-probe history on, the
 * interpolation jitter off. Per surface pixel of the interpolated irradiance
 * E (the image the composite reads; no denoiser yet):
 *
 * - **noise:** |E_t − (E_t−1 + E_t+1) / 2| / max(E_t, 5% of the frame's mean),
 *   the second difference, which a steady rise doesn't move;
 * - **lag:** the frames until the frame's mean E stays within 5% of the
 *   settled image (the mean of frames 100-119 after the change).
 *
 * 1. **A light doubled once** (Sponza's sun; two-room's lamp; and two-room
 *    with multibounce off, to tell the second bounce's pace from the rest):
 *    noise before and over the 16 frames after, the lag, and the mean E 16
 *    frames after over the settled. Gates, on E9's default (history dropped):
 *    the noise over the 16 frames after is at most 1.5 times E9 off's (two-room
 *    measured 1.34), and the image follows no slower (the lag no longer,
 *    frame 16 no further from settled).
 * 2. **A change that goes on** for 60 frames (Sponza's sun turning 0.5° a
 *    frame; a 0.6 m box crossing two-room's lit room at 2 cm a frame), then
 *    held for 120: the noise over frames 10-59 of the motion, and the
 *    per-pixel error of the motion's last frame against the held image's
 *    settled E (frames 110-119). Gates: with E9's default the noise is at most
 *    1.25 times E9 off's (before the events, E9 reset every cell every frame
 *    of such a change), and the image follows: its error at most 1.05 times
 *    E9 off's.
 *
 * Only NaN, infinite or negative values and the gates fail.
 *
 * Through the lock: `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-change-noise --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { InstanceHandle, LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
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
/** Frames read every frame after a change (lag and noise), then the settled window. */
const FOLLOW_FRAMES = 40;
const SETTLED_FROM = 100;
const SETTLED_TO = 120;
const WINDOW = 16;
const MOTION_FRAMES = 60;
const HOLD_FRAMES = 120;
const DARK_FLOOR = 0.05;
const LAG_WITHIN = 0.05;

const SETTINGS: readonly { readonly name: string; readonly settings: Partial<RadianceCacheSettings> }[] = [
    { name: 'E9 off', settings: { cellChangeReset: false, cellChangeResetHistory: 0 } },
    { name: 'E9 on', settings: { cellChangeReset: true, cellChangeResetHistory: 0 } },
    { name: 'E9 on, 3 kept', settings: { cellChangeReset: true, cellChangeResetHistory: 3 } },
];
/** The gated setting: E9's default (lead, 2026-11-20: history dropped, for fast reaction and no ghosting). */
const KEPT = 'E9 on';
const OFF = 'E9 off';

const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));

/** Relative values pooled over pixels and frames: their sum and a histogram (steps of 0.0005 up to 5) for the percentile. */
class Tally {
    static readonly STEP = 0.0005;
    readonly bins = new Uint32Array(10001);
    sum = 0;
    count = 0;

    add(value: number): void {
        if (!Number.isFinite(value)) return;
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
    readonly bad: number;
}

/** A scene's changeable parts: a light's intensity scale, the sun's turn, the box's place along its path. */
interface Controls {
    lightScale?: (scale: number) => void;
    sunTurn?: (degrees: number) => void;
    boxAt?: (step: number) => void;
}

interface Loaded {
    readonly technique: RadianceCacheTechnique;
    readonly controls: Controls;
    frame(): void;
    read(): Promise<Frame>;
    dispose(): void;
}

/** A box of `size` metres at the origin: six quads facing out. */
function boxMesh(platform: GIPlatformImpl, size: number): InstanceHandle {
    const h = size / 2;
    const faces: [number[], number[]][] = [
        [[h, -h, -h, h, h, -h, h, h, h, h, -h, h], [1, 0, 0]],
        [[-h, -h, h, -h, h, h, -h, h, -h, -h, -h, -h], [-1, 0, 0]],
        [[-h, h, -h, -h, h, h, h, h, h, h, h, -h], [0, 1, 0]],
        [[-h, -h, h, -h, -h, -h, h, -h, -h, h, -h, h], [0, -1, 0]],
        [[-h, -h, h, h, -h, h, h, h, h, -h, h, h], [0, 0, 1]],
        [[h, -h, -h, -h, -h, -h, -h, h, -h, h, h, -h], [0, 0, -1]],
    ];
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    faces.forEach(([corners, normal], face) => {
        positions.push(...corners);
        for (let corner = 0; corner < 4; corner++) normals.push(...normal);
        const base = face * 4;
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    });
    const scene = platform.scene;
    const mesh = scene.createMesh({ positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices), label: 'moving box' });
    const material = scene.createMaterial({ albedo: [0.7, 0.7, 0.7], label: 'moving box' });
    return scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -100, 0, 1], label: 'moving box' });
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
    const controls: Controls = {};
    if (id === 'sponza') {
        // Through the viewer's loader and the three adapter's scene sync; the sun is three's.
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
            const sun = object as unknown as { isDirectionalLight?: boolean; intensity: number; position: { x: number; y: number; z: number }; target: { position: { x: number; z: number } }; updateMatrixWorld(): void };
            if (!sun.isDirectionalLight || controls.lightScale) return;
            const base = sun.intensity;
            const start = { x: sun.position.x, z: sun.position.z };
            controls.lightScale = (scale) => {
                sun.intensity = base * scale;
            };
            // Turns the sun about the vertical through its target.
            controls.sunTurn = (degrees) => {
                const angle = (degrees * Math.PI) / 180;
                const cx = sun.target.position.x;
                const cz = sun.target.position.z;
                const dx = start.x - cx;
                const dz = start.z - cz;
                sun.position.x = cx + dx * Math.cos(angle) - dz * Math.sin(angle);
                sun.position.z = cz + dx * Math.sin(angle) + dz * Math.cos(angle);
                sun.updateMatrixWorld();
            };
        });
    } else {
        const handles: LightHandle[] = [];
        describeScene(platform, manifest, {}, handles);
        const base = manifest.lights[0];
        controls.lightScale = (scale) => platform.scene.updateLight(handles[0], { intensity: base.intensity * scale });
        const box = boxMesh(platform, 0.6);
        // Across the lit room: from (0.6, 0.8, 0.6) toward (3.4, 0.8, 3.4), 2 cm a frame.
        controls.boxAt = (step) => {
            const t = Math.min(step * 0.02, 2.8);
            platform.scene.setTransform(box, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.6 + t, 0.8, 0.6 + t, 1]);
        };
        controls.boxAt(0);
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
    let depth: Float32Array | null = null;
    const read = async (): Promise<Frame> => {
        await device.queue.onSubmittedWorkDone();
        const probes = technique.probes;
        if (!probes) throw new Error('no probes after encodeScreen');
        const bytes = await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT);
        const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
        depth ??= new Float32Array((await readTexture(device, screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
        const e = new Float32Array(WIDTH * HEIGHT);
        let sum = 0;
        let count = 0;
        let bad = 0;
        for (let pixel = 0; pixel < e.length; pixel++) {
            if (depth[pixel] === 1) {
                e[pixel] = NaN;
                continue;
            }
            const value = (HALF_TABLE[halves[pixel * 4]] + HALF_TABLE[halves[pixel * 4 + 1]] + HALF_TABLE[halves[pixel * 4 + 2]]) / 3;
            if (!(Number.isFinite(value) && value >= 0)) bad++;
            e[pixel] = value;
            sum += value;
            count++;
        }
        return { e, mean: sum / Math.max(count, 1), bad };
    };
    frame();
    await platform.rays.whenBuilt();
    technique.resetHistory();
    for (let index = 0; index < WARM_FRAMES; index++) {
        frame();
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }
    return {
        technique,
        controls,
        frame,
        read,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

/** The second difference at index `t` of `frames` into `noise`. */
function tallyNoise(frames: readonly Frame[], t: number, noise: Tally): void {
    const [before, now, after] = [frames[t - 1], frames[t], frames[t + 1]];
    const floor = DARK_FLOOR * now.mean;
    for (let pixel = 0; pixel < now.e.length; pixel++) {
        const value = now.e[pixel];
        if (Number.isNaN(value)) continue;
        noise.add(Math.abs(value - (before.e[pixel] + after.e[pixel]) / 2) / Math.max(value, floor));
    }
}

/** Per pixel, |E − reference| / max(reference, 5% of its mean), averaged. */
function imageError(frame: Frame, reference: Frame): number {
    const floor = DARK_FLOOR * reference.mean;
    let sum = 0;
    let count = 0;
    for (let pixel = 0; pixel < frame.e.length; pixel++) {
        const value = reference.e[pixel];
        if (Number.isNaN(value)) continue;
        sum += Math.abs(frame.e[pixel] - value) / Math.max(value, floor);
        count++;
    }
    return sum / Math.max(count, 1);
}

/** The per-pixel mean of some frames. */
function meanFrame(frames: readonly Frame[]): Frame {
    const e = new Float32Array(frames[0].e.length);
    for (const frame of frames) for (let pixel = 0; pixel < e.length; pixel++) e[pixel] += frame.e[pixel] / frames.length;
    return { e, mean: frames.reduce((sum, frame) => sum + frame.mean, 0) / frames.length, bad: 0 };
}

const round = (value: number, digits = 4): number => Number(value.toFixed(digits));

async function stepCase(scene: Loaded, settings: Partial<RadianceCacheSettings>): Promise<Record<string, number>> {
    const scale = scene.controls.lightScale;
    if (!scale) throw new Error('no light to scale');
    Object.assign(scene.technique.settings, settings);
    scale(1);
    for (let index = 0; index < SETTLE_FRAMES; index++) scene.frame();
    const baseline: Frame[] = [];
    for (let index = 0; index < BASELINE_FRAMES; index++) {
        scene.frame();
        baseline.push(await scene.read());
    }
    scale(2);
    const after: Frame[] = [baseline[baseline.length - 1]];
    const settled: Frame[] = [];
    for (let index = 1; index <= SETTLED_TO; index++) {
        scene.frame();
        if (index <= FOLLOW_FRAMES) after.push(await scene.read());
        else if (index > SETTLED_FROM) settled.push(await scene.read());
    }
    scale(1);
    const baselineNoise = new Tally();
    for (let t = 1; t < baseline.length - 1; t++) tallyNoise(baseline, t, baselineNoise);
    const noise = new Tally();
    for (let t = 1; t <= WINDOW; t++) tallyNoise(after, t, noise);
    const reference = settled.reduce((sum, frame) => sum + frame.mean, 0) / settled.length;
    let lag = FOLLOW_FRAMES + 1;
    for (let t = FOLLOW_FRAMES; t >= 1; t--) {
        if (Math.abs(after[t].mean / reference - 1) > LAG_WITHIN) break;
        lag = t;
    }
    const before = baseline.reduce((sum, frame) => sum + frame.mean, 0) / baseline.length;
    return {
        baselineNoiseMean: round(baselineNoise.mean),
        noiseMean: round(noise.mean),
        noiseP95: round(noise.percentile(0.95)),
        lagFrames: lag,
        settledOverBefore: round(reference / before),
        meanE1OverSettled: round(after[1].mean / reference),
        meanE16OverSettled: round(after[16].mean / reference),
        bad: [...baseline, ...after, ...settled].reduce((sum, frame) => sum + frame.bad, 0),
    };
}

async function motionCase(scene: Loaded, kind: 'sun' | 'box', settings: Partial<RadianceCacheSettings>): Promise<Record<string, number>> {
    const move = kind === 'sun' ? scene.controls.sunTurn : scene.controls.boxAt;
    if (!move) throw new Error(`no ${kind} to move`);
    Object.assign(scene.technique.settings, settings);
    move(0);
    for (let index = 0; index < SETTLE_FRAMES; index++) scene.frame();
    const motion: Frame[] = [];
    for (let step = 1; step <= MOTION_FRAMES; step++) {
        move(kind === 'sun' ? 0.5 * step : step);
        scene.frame();
        if (step >= 10) motion.push(await scene.read());
    }
    const held: Frame[] = [];
    for (let index = 1; index <= HOLD_FRAMES; index++) {
        scene.frame();
        if (index > HOLD_FRAMES - 10) held.push(await scene.read());
    }
    move(0);
    const noise = new Tally();
    for (let t = 1; t < motion.length - 1; t++) tallyNoise(motion, t, noise);
    const reference = meanFrame(held);
    return {
        noiseMean: round(noise.mean),
        noiseP95: round(noise.percentile(0.95)),
        endError: round(imageError(motion[motion.length - 1], reference)),
        bad: [...motion, ...held].reduce((sum, frame) => sum + frame.bad, 0),
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
    const steps: Record<string, Record<string, Record<string, number>>> = {};
    const motions: Record<string, Record<string, Record<string, number>>> = {};
    const failures: string[] = [];
    for (const id of ['sponza', 'two-room'] as const) {
        const scene = await load(device, id);
        try {
            const lightName = id === 'sponza' ? 'sponza sun doubled' : 'two-room lamp doubled';
            steps[lightName] = {};
            for (const { name, settings } of SETTINGS) steps[lightName][name] = await stepCase(scene, settings);
            if (id === 'two-room') {
                // The second bounce's pace: the same change with multibounce off.
                steps['two-room lamp doubled, multibounce off'] = {};
                for (const { name, settings } of SETTINGS) steps['two-room lamp doubled, multibounce off'][name] = await stepCase(scene, { ...settings, useMultibounce: false });
                Object.assign(scene.technique.settings, { useMultibounce: true });
            }
            const motionName = id === 'sponza' ? 'sponza sun turning' : 'two-room box moving';
            motions[motionName] = {};
            for (const { name, settings } of SETTINGS) motions[motionName][name] = await motionCase(scene, id === 'sponza' ? 'sun' : 'box', settings);
        } finally {
            scene.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }
    for (const [name, results] of [...Object.entries(steps), ...Object.entries(motions)]) {
        for (const [setting, measured] of Object.entries(results)) if (measured.bad > 0) failures.push(`${name}, ${setting}: ${measured.bad} pixels' E is NaN, infinite or negative`);
    }
    for (const [name, results] of Object.entries(steps)) {
        if (results[KEPT].noiseMean > 1.5 * results[OFF].noiseMean) failures.push(`${name}: noise after the change ${results[KEPT].noiseMean} with ${KEPT}, over 1.5 times ${OFF}'s ${results[OFF].noiseMean}`);
        if (results[KEPT].lagFrames > results[OFF].lagFrames || results[KEPT].meanE16OverSettled < results[OFF].meanE16OverSettled) {
            failures.push(`${name}: with ${KEPT} the lag is ${results[KEPT].lagFrames} frames and frame 16 at ${results[KEPT].meanE16OverSettled} of settled, with ${OFF} ${results[OFF].lagFrames} and ${results[OFF].meanE16OverSettled}`);
        }
    }
    for (const [name, results] of Object.entries(motions)) {
        if (results[KEPT].noiseMean > 1.25 * results[OFF].noiseMean) failures.push(`${name}: noise during the change ${results[KEPT].noiseMean} with ${KEPT}, over 1.25 times ${OFF}'s ${results[OFF].noiseMean}`);
        if (results[KEPT].endError > 1.05 * results[OFF].endError) failures.push(`${name}: error at the motion's end ${results[KEPT].endError} with ${KEPT}, over 1.05 times ${OFF}'s ${results[OFF].endError}`);
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, steps, motions },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
