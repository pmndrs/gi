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
 * E, the raw image before the GI denoiser (RC4.3; the composite reads the
 * denoiser's output, which this page doesn't measure, and the denoiser feeds
 * nothing back into E):
 *
 * - **noise** (what a viewer sees as noise; gated): the second difference with
 *   the whole image's change taken out, each neighbour frame scaled to frame
 *   t's mean: |E_t − (E_t−1 · m_t / m_t−1 + E_t+1 · m_t / m_t+1) / 2| /
 *   max(E_t, 5% of m_t), m the frame's mean E. A light's change scales the
 *   image, so a fast brightening of the whole image isn't counted;
 * - **sharpness** (recorded, not gated): the plain second difference
 *   |E_t − (E_t−1 + E_t+1) / 2| / max(E_t, 5% of m_t), which also counts a
 *   rise that is fast and then slows. Until 2026-11-24 this was the gated
 *   "noise"; on two-room most of E9's share of it was its mean E following
 *   faster (the frame mean's own second difference 1.01% with E9, 0.32%
 *   without), so it says how sharply E responds, not how noisy it is;
 * - **lag:** the frames until the frame's mean E stays within 5% of the
 *   settled image (the mean of frames 381-400 after the change; read every
 *   frame up to 300).
 *
 * The windows (2026-11-24, from a 1,000-frame trace of the mean after each
 * step): before a step the scene settles for 400 frames, and the settled image
 * is frames 381-400. With E9 off two-room's mean reaches 95% of its final value
 * at about frame 120 and 99.5% near frame 300; Sponza's 95% near frame 130 and
 * 99.5% near 400. The old 60 settle frames left the last step still fading
 * (each later setting's "settled over before" read lower: 1.90, 1.83, 1.73),
 * and the old window (frames 101-120, the lag read to 40) sat at 0.94-0.99 of
 * final, so two-room's lag was 41 (never within 5% inside 40) with E9 on and
 * off, and the lag gate passed whatever E9 did.
 *
 * 1. **A light doubled once** (Sponza's sun; two-room's lamp; and two-room
 *    with multibounce off, to tell the second bounce's pace from the rest):
 *    noise before and over the 16 frames after, the lag, and the mean E 16
 *    frames after over the settled. Gates, on E9's default (history dropped):
 *    the noise E9 adds over the 16 frames after (E9 on's minus E9 off's, both
 *    at the defaults) is at most half of E9 off's noise with the probe filter
 *    and the ray-guiding correction off (`rcfilter=false,rcguidefix=false`,
 *    the configuration P6.2b's 1.5× gate was set on: the same +50%), and the
 *    image follows no slower (the lag no longer, frame 16 no further from
 *    settled). DECISIONS 2026-11-24: RC4.2's probe filter and D64 halved E9
 *    off's noise but not the rise E9 adds, so the 1.5× ratio of the defaults
 *    (1.34 at P6.2b, 1.79 after RC4.2, both on what is now `sharpness`)
 *    stopped measuring E9; it is recorded (`ratio`), not gated.
 *
 *    What the noise gate can see (mutations, 2026-11-24, two-room, none
 *    kept): noise in the world cache's cells is averaged out of E by the
 *    screen probes' running mean and filter (marked cells' values scaled by
 *    a random 0.5-1.5: E9's added noise 0.28 → 0.30 points; every cell, every
 *    frame: E9 off's 0.70 → 0.78% on the old measure), so the gate bounds
 *    visible noise only. And it bounds it loosely: probe cells near the
 *    change skipping their running mean for the event's 17 frames with E9 on
 *    raised E9's added noise from 0.28 to 0.71 points (E9 on 0.84 → 1.26%),
 *    inside the 0.80 allowed, since half the calibration run's noise is more
 *    than E9 off's whole visible noise at the defaults (0.56%).
 * 2. **A change that goes on** for 60 frames (Sponza's sun turning 0.5° a
 *    frame; a 0.6 m box crossing two-room's lit room at 2 cm a frame), then
 *    held for 400: the noise over frames 10-59 of the motion, and the
 *    per-pixel error of the motion's last frame against the held image's
 *    settled E (frames 391-400; 110-119 until 2026-11-24, not yet settled). Gates: with E9's default the noise is at most
 *    1.25 times E9 off's (before the events, E9 reset every cell every frame
 *    of such a change), and the image follows: its error at most 1.05 times
 *    E9 off's. The box runs twice per setting and the gate takes the two
 *    runs' mean error (one run's moved by about 6%, as much as the gate).
 *
 * Only NaN, infinite or negative values and the gates fail.
 *
 * `VITE_RC_COMPARE` runs it at compare toggles (RC4.5, radiance-cache-probes/compare-env.ts);
 * the E9 settings of each case are set over them, and the calibration run's filter and D64 over
 * those.
 *
 * Through the lock: `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-change-noise --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { InstanceHandle, LightHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { halfBitsToFloat } from '../../../../packages/core/src/techniques/radiance-cache/capsaicin-math.ts';
import type { RadianceCacheTechnique } from '../../../../packages/core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/core/src/techniques/radiance-cache/settings.ts';
import { SceneSync } from '../../../../packages/three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { compareValuesFromText } from '../../../../packages/core/src/techniques/radiance-cache/compare-toggles.ts';
import { PAGE_COMPARE_VALUES, pageRadianceCache } from '../radiance-cache-probes/compare-env.ts';
import { readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const WARM_FRAMES = 60;
/** Frames before a step or a motion: the last case's change has faded (see the doc comment). */
const SETTLE_FRAMES = 400;
const BASELINE_FRAMES = 10;
/** Frames whose mean is read after a change (the lag), then the settled window (frames 381-400). */
const FOLLOW_FRAMES = 300;
const SETTLED_FROM = 380;
const SETTLED_TO = 400;
const WINDOW = 16;
const MOTION_FRAMES = 60;
/** Frames the motion's end is held: its last 10 are the settled image (120 left it short, as the steps' windows did). */
const HOLD_FRAMES = 400;
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
/**
 * E9 off in the configuration P6.2b's gate was set on: RC4.2's probe filter and the ray-guiding
 * correction (D64) off, by their compare toggles (DECISIONS 2026-11-24). Half its noise is what E9
 * may add at the defaults.
 */
const CALIBRATION = 'E9 off, filter and D64 off';
const CALIBRATION_SETTINGS: Partial<RadianceCacheSettings> = { cellChangeReset: false, cellChangeResetHistory: 0, ...(compareValuesFromText('rcfilter=false,rcguidefix=false') as Partial<RadianceCacheSettings>) };
/** P6.2b's 1.5× of E9 off: the noise E9 may add, as a share of the calibration run's. */
const ADDED_NOISE_ALLOWANCE = 0.5;
const STEP_SETTINGS = [...SETTINGS, { name: CALIBRATION, settings: CALIBRATION_SETTINGS }];

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
    /** The page's settings (its own and the run's compare values), which each case starts from. */
    readonly base: Readonly<RadianceCacheSettings>;
    readonly controls: Controls;
    frame(): void;
    /** Waits for the frames submitted so far (so long runs don't queue hundreds). */
    drain(): Promise<void>;
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
    const technique = platform.addTechnique(pageRadianceCache({ interpolationJitter: false }));
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
        base: { ...technique.settings },
        controls,
        frame,
        drain: () => device.queue.onSubmittedWorkDone(),
        read,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

/**
 * At index `t` of `frames`: the second difference with the whole image's change taken out (the
 * neighbours scaled to frame t's mean) into `noise`, and the plain one into `sharpness`.
 */
function tallyNoise(frames: readonly Frame[], t: number, noise: Tally, sharpness: Tally): void {
    const [before, now, after] = [frames[t - 1], frames[t], frames[t + 1]];
    const floor = DARK_FLOOR * now.mean;
    const beforeScale = now.mean / Math.max(before.mean, 1e-12);
    const afterScale = now.mean / Math.max(after.mean, 1e-12);
    for (let pixel = 0; pixel < now.e.length; pixel++) {
        const value = now.e[pixel];
        if (Number.isNaN(value)) continue;
        const denominator = Math.max(value, floor);
        noise.add(Math.abs(value - (before.e[pixel] * beforeScale + after.e[pixel] * afterScale) / 2) / denominator);
        sharpness.add(Math.abs(value - (before.e[pixel] + after.e[pixel]) / 2) / denominator);
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

/** A frame's mean E only, for the frames past the noise window. */
async function readMean(scene: Loaded): Promise<{ readonly mean: number; readonly bad: number }> {
    const { mean, bad } = await scene.read();
    return { mean, bad };
}

async function stepCase(scene: Loaded, settings: Partial<RadianceCacheSettings>): Promise<Record<string, number>> {
    const scale = scene.controls.lightScale;
    if (!scale) throw new Error('no light to scale');
    Object.assign(scene.technique.settings, scene.base, settings);
    scale(1);
    for (let index = 0; index < SETTLE_FRAMES; index++) {
        scene.frame();
        if (index % 8 === 7) await scene.drain();
    }
    const baseline: Frame[] = [];
    for (let index = 0; index < BASELINE_FRAMES; index++) {
        scene.frame();
        baseline.push(await scene.read());
    }
    scale(2);
    // Whole frames for the noise window (and one past it), means after that.
    const after: Frame[] = [baseline[baseline.length - 1]];
    const means: number[] = [baseline[baseline.length - 1].mean];
    const settled: number[] = [];
    let bad = 0;
    for (let index = 1; index <= SETTLED_TO; index++) {
        scene.frame();
        if (index <= WINDOW + 1) {
            const frame = await scene.read();
            after.push(frame);
            means.push(frame.mean);
            bad += frame.bad;
        } else if (index <= FOLLOW_FRAMES || index > SETTLED_FROM) {
            const frame = await readMean(scene);
            bad += frame.bad;
            if (index <= FOLLOW_FRAMES) means.push(frame.mean);
            else settled.push(frame.mean);
        } else if (index % 8 === 0) await scene.drain();
    }
    scale(1);
    const baselineNoise = new Tally();
    for (let t = 1; t < baseline.length - 1; t++) tallyNoise(baseline, t, baselineNoise, new Tally());
    const noise = new Tally();
    const sharpness = new Tally();
    for (let t = 1; t <= WINDOW; t++) tallyNoise(after, t, noise, sharpness);
    const reference = settled.reduce((sum, mean) => sum + mean, 0) / settled.length;
    let lag = FOLLOW_FRAMES + 1;
    for (let t = FOLLOW_FRAMES; t >= 1; t--) {
        if (Math.abs(means[t] / reference - 1) > LAG_WITHIN) break;
        lag = t;
    }
    const before = baseline.reduce((sum, frame) => sum + frame.mean, 0) / baseline.length;
    return {
        baselineNoiseMean: round(baselineNoise.mean, 5),
        noiseMean: round(noise.mean, 5),
        noiseP95: round(noise.percentile(0.95)),
        sharpnessMean: round(sharpness.mean, 5),
        lagFrames: lag,
        settledOverBefore: round(reference / before),
        meanE1OverSettled: round(means[1] / reference),
        meanE16OverSettled: round(means[16] / reference),
        meanE40OverSettled: round(means[40] / reference),
        bad: bad + baseline.reduce((sum, frame) => sum + frame.bad, 0),
    };
}

async function motionCase(scene: Loaded, kind: 'sun' | 'box', settings: Partial<RadianceCacheSettings>): Promise<Record<string, number>> {
    const move = kind === 'sun' ? scene.controls.sunTurn : scene.controls.boxAt;
    if (!move) throw new Error(`no ${kind} to move`);
    Object.assign(scene.technique.settings, scene.base, settings);
    move(0);
    for (let index = 0; index < SETTLE_FRAMES; index++) {
        scene.frame();
        if (index % 8 === 7) await scene.drain();
    }
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
        else if (index % 8 === 0) await scene.drain();
    }
    move(0);
    const noise = new Tally();
    const sharpness = new Tally();
    for (let t = 1; t < motion.length - 1; t++) tallyNoise(motion, t, noise, sharpness);
    const reference = meanFrame(held);
    return {
        noiseMean: round(noise.mean, 5),
        noiseP95: round(noise.percentile(0.95)),
        sharpnessMean: round(sharpness.mean, 5),
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
            for (const { name, settings } of STEP_SETTINGS) steps[lightName][name] = await stepCase(scene, settings);
            if (id === 'two-room') {
                // The second bounce's pace: the same change with multibounce off.
                steps['two-room lamp doubled, multibounce off'] = {};
                for (const { name, settings } of STEP_SETTINGS) steps['two-room lamp doubled, multibounce off'][name] = await stepCase(scene, { ...settings, useMultibounce: false });
            }
            const motionName = id === 'sponza' ? 'sponza sun turning' : 'two-room box moving';
            motions[motionName] = {};
            for (const { name, settings } of SETTINGS) {
                if (id === 'sponza') {
                    motions[motionName][name] = await motionCase(scene, 'sun', settings);
                    continue;
                }
                // The box's end error moves by about 6% from run to run, as much as its gate (1.05×):
                // two runs (each on its own frames, so its own random numbers), the gate on their mean.
                const first = await motionCase(scene, 'box', settings);
                const second = await motionCase(scene, 'box', settings);
                motions[motionName][name] = {
                    ...first,
                    noiseMean: round((first.noiseMean + second.noiseMean) / 2, 5),
                    sharpnessMean: round((first.sharpnessMean + second.sharpnessMean) / 2, 5),
                    endError: round((first.endError + second.endError) / 2),
                    endErrorRun1: first.endError,
                    endErrorRun2: second.endError,
                    bad: first.bad + second.bad,
                };
            }
        } finally {
            scene.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }
    for (const [name, results] of [...Object.entries(steps), ...Object.entries(motions)]) {
        for (const [setting, measured] of Object.entries(results)) if (measured.bad > 0) failures.push(`${name}, ${setting}: ${measured.bad} pixels' E is NaN, infinite or negative`);
    }
    const noiseGates: Record<string, Record<string, number>> = {};
    for (const [name, results] of Object.entries(steps)) {
        // The noise E9 adds, against half the calibration run's (DECISIONS 2026-11-24); the ratio is recorded only.
        const added = results[KEPT].noiseMean - results[OFF].noiseMean;
        const allowed = ADDED_NOISE_ALLOWANCE * results[CALIBRATION].noiseMean;
        noiseGates[name] = { added: round(added, 5), allowed: round(allowed, 5), ratio: round(results[KEPT].noiseMean / results[OFF].noiseMean, 3) };
        if (added > allowed) failures.push(`${name}: E9 adds ${round(added, 5)} noise after the change (${KEPT} ${results[KEPT].noiseMean}, ${OFF} ${results[OFF].noiseMean}), over half of ${CALIBRATION}'s ${results[CALIBRATION].noiseMean}`);
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
        data: { width: WIDTH, height: HEIGHT, compare: PAGE_COMPARE_VALUES, noiseGates, steps, motions },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
