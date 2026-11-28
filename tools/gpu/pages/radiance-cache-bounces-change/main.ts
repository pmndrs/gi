/**
 * RC3.6: the ways of getting the bounces past the second, compared over time
 * (a measurement, not a gate: DECISIONS 2026-11-23 keeps the alternatives as
 * variants until P6.5; timing is never gated). Accuracy against Cycles is
 * `radiance-cache-accuracy`; the analytic gates are `radiance-cache-bounces`.
 *
 * **Modes** (each made fresh, its own platform and technique):
 * - `gi1-rays`: the cache alone, GI-1.2's two bounces (the baseline);
 * - `cache feedback` (register E12): GI-1.2's rays, each second hit also
 *   bringing its cell's indirect value;
 * - `temporal feedback` (register D68): GI-1.2's rays, probe rays whose hit was
 *   on last frame's screen reading last frame's lit image;
 * - `cache + temporal feedback`: both;
 * - `DDGI (P6.3)`: a DDGI volume (the manifest's, DDGI at its defaults,
 *   256 probes a frame, as the viewer ships, RC4.6) stacked under the cache,
 *   `secondBounce: 'auto'` reading it;
 * - `DDGI + temporal feedback`;
 * - `SHaRC (P6.1)`: `worldCache: 'sharc'` (every bounce in its own cache);
 * - `neural tail (P6.4)`: GI-1.2's rays with `neuralCache: 'tail'` (needs the
 *   optional `@pathtracer/mlp` sibling; without it the mode is skipped).
 * Cache feedback with DDGI stacked is the DDGI mode (no second-bounce rays to
 * feed: E10), and SHaRC doesn't read a stacked volume, so neither is run.
 *
 * **Scenes**, as radiance-cache-change-noise loads them (1280 × 720, the first
 * bookmark, the camera still, screen-probe history on, interpolation jitter
 * off, the raw interpolated irradiance E before the GI denoiser): two-room
 * (the lamp; a 0.6 m box crossing the lit room at 2 cm a frame) and Sponza
 * (the sun; through the viewer's loader and the three adapter's scene sync).
 *
 * **What each mode reports**, per scene:
 * - **flicker** (still, after `SETTLE_FRAMES`): over `STILL_FRAMES` frames, the
 *   mean of radiance-cache-change-noise's noise (each pixel's second difference
 *   with the whole image's change taken out, over max(E, 5% of the mean)), and
 *   the frame-to-frame change |E_t − E_t−1| / max(E_t−1, 5% of the mean), mean
 *   and p95;
 * - **brightness**: the settled image's mean E (more bounces read brighter);
 * - **lag** after the light doubles (two-room's lamp, Sponza's sun) and after
 *   two-room's lamp is switched on (from off, settled): the frames until the
 *   frame's mean E stays within 5% of the settled image (frames 381-400 after
 *   the change), and the mean E 1, 16 and 40 frames after over settled;
 * - **ghosting** (two-room's box crossing for 60 frames, then held): the
 *   per-pixel error against the held image's settled E (frames 391-400) on the
 *   motion's last frame, and 16 and 60 frames after it stopped: what is left of
 *   the box's old light and shadow. Two runs, their mean;
 * - **memory**: the radiance cache's GPU bytes (and the stacked DDGI's).
 *
 * Fails only on NaN, infinite or negative E, or device errors.
 * `VITE_RC_BOUNCES_MODES` and `VITE_RC_BOUNCES_SCENES` pick subsets (names as above,
 * comma-separated).
 *
 * Through the lock: `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-bounces-change --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { InstanceHandle, LightHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { halfBitsToFloat } from '../../../../packages/cache/src/capsaicin-math.ts';
import type { RadianceCacheTechnique } from '../../../../packages/cache/src/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/cache/src/settings.ts';
import { SceneSync } from '../../../../packages/three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { PAGE_COMPARE_VALUES, pageRadianceCache } from '../radiance-cache-probes/compare-env.ts';
import { readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const WARM_FRAMES = 60;
const SETTLE_FRAMES = 400;
const STILL_FRAMES = 30;
const FOLLOW_FRAMES = 300;
const SETTLED_FROM = 380;
const SETTLED_TO = 400;
const MOTION_FRAMES = 60;
const HOLD_FRAMES = 400;
const DARK_FLOOR = 0.05;
const LAG_WITHIN = 0.05;
/** DDGI's probes a frame when stacked: what the viewer ships (RC4.6). */
const DDGI_UNITS = 256;

interface Mode {
    readonly name: string;
    readonly settings: Partial<RadianceCacheSettings>;
    readonly ddgi?: boolean;
}

const ALL_MODES: readonly Mode[] = [
    { name: 'gi1-rays', settings: { secondBounce: 'gi1-rays' } },
    { name: 'cache feedback', settings: { secondBounce: 'gi1-rays', cacheFeedback: true } },
    { name: 'temporal feedback', settings: { secondBounce: 'gi1-rays', useTemporalFeedback: true } },
    { name: 'cache + temporal feedback', settings: { secondBounce: 'gi1-rays', cacheFeedback: true, useTemporalFeedback: true } },
    { name: 'DDGI (P6.3)', settings: { secondBounce: 'auto' }, ddgi: true },
    { name: 'DDGI + temporal feedback', settings: { secondBounce: 'auto', useTemporalFeedback: true }, ddgi: true },
    { name: 'SHaRC (P6.1)', settings: { worldCache: 'sharc' } },
    { name: 'neural tail (P6.4)', settings: { secondBounce: 'gi1-rays', neuralCache: 'tail' } },
];
const env = import.meta.env as Record<string, unknown>;
const listFromEnv = (name: string): string[] | null => (typeof env[name] === 'string' && env[name] ? (env[name] as string).split(',').map((item) => item.trim()).filter(Boolean) : null);
const modeNames = listFromEnv('VITE_RC_BOUNCES_MODES');
const MODES = modeNames ? ALL_MODES.filter((mode) => modeNames.includes(mode.name)) : ALL_MODES;
const SCENES = listFromEnv('VITE_RC_BOUNCES_SCENES') ?? ['two-room', 'sponza'];

const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));
const round = (value: number, digits = 4): number => Number(value.toFixed(digits));

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

interface Controls {
    lightScale?: (scale: number) => void;
    boxAt?: (step: number) => void;
}

interface Loaded {
    readonly technique: RadianceCacheTechnique;
    readonly ddgi: DDGITechnique | null;
    readonly controls: Controls;
    frame(): void;
    drain(): Promise<void>;
    read(): Promise<Frame>;
    dispose(): void;
}

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

async function load(device: GPUDevice, id: string, mode: Mode): Promise<Loaded> {
    const manifest: SceneManifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-bounces-change ${id} ${mode.name}` });
    const technique = platform.addTechnique(pageRadianceCache({ interpolationJitter: false, ...mode.settings }));
    Object.assign(technique.settings, mode.settings);
    platform.setOutput(technique);
    const ddgi = mode.ddgi ? platform.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing } })) : null;
    if (ddgi) platform.setFallback(technique, ddgi);
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
            const sun = object as unknown as { isDirectionalLight?: boolean; intensity: number };
            if (!sun.isDirectionalLight || controls.lightScale) return;
            const base = sun.intensity;
            controls.lightScale = (scale) => {
                sun.intensity = base * scale;
            };
        });
    } else {
        const handles: LightHandle[] = [];
        describeScene(platform, manifest, {}, handles);
        const base = manifest.lights[0];
        controls.lightScale = (scale) => platform.scene.updateLight(handles[0], { intensity: base.intensity * scale });
        const box = boxMesh(platform, 0.6);
        // Across the lit room: from (0.6, 0.8, 0.6) toward (3.4, 0.8, 3.4), 2 cm a frame (radiance-cache-change-noise's path).
        controls.boxAt = (step) => {
            const t = Math.min(step * 0.02, 2.8);
            platform.scene.setTransform(box, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.6 + t, 0.8, 0.6 + t, 1]);
        };
        controls.boxAt(0);
    }
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        beforeFrame();
        const encoder = device.createCommandEncoder({ label: `radiance-cache-bounces-change ${id}` });
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: ddgi ? DDGI_UNITS : 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
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
        ddgi,
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

async function settle(scene: Loaded, frames: number): Promise<void> {
    for (let index = 0; index < frames; index++) {
        scene.frame();
        if (index % 8 === 7) await scene.drain();
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

function meanFrame(frames: readonly Frame[]): Frame {
    const e = new Float32Array(frames[0].e.length);
    for (const frame of frames) for (let pixel = 0; pixel < e.length; pixel++) e[pixel] += frame.e[pixel] / frames.length;
    return { e, mean: frames.reduce((sum, frame) => sum + frame.mean, 0) / frames.length, bad: 0 };
}

/** Flicker when still: the noise (second difference, the image's own change taken out) and the frame-to-frame change. */
async function stillCase(scene: Loaded): Promise<Record<string, number>> {
    await settle(scene, SETTLE_FRAMES);
    const frames: Frame[] = [];
    for (let index = 0; index < STILL_FRAMES; index++) {
        scene.frame();
        frames.push(await scene.read());
    }
    const noise = new Tally();
    const change = new Tally();
    for (let t = 1; t < frames.length; t++) {
        const [before, now] = [frames[t - 1], frames[t]];
        const floor = DARK_FLOOR * now.mean;
        for (let pixel = 0; pixel < now.e.length; pixel++) {
            const value = now.e[pixel];
            if (Number.isNaN(value)) continue;
            change.add(Math.abs(value - before.e[pixel]) / Math.max(before.e[pixel], DARK_FLOOR * before.mean));
            if (t + 1 < frames.length) {
                const after = frames[t + 1];
                const beforeScale = now.mean / Math.max(before.mean, 1e-12);
                const afterScale = now.mean / Math.max(after.mean, 1e-12);
                noise.add(Math.abs(value - (before.e[pixel] * beforeScale + after.e[pixel] * afterScale) / 2) / Math.max(value, floor));
            }
        }
    }
    return {
        meanE: round(frames.reduce((sum, frame) => sum + frame.mean, 0) / frames.length, 5),
        noiseMean: round(noise.mean, 5),
        changeMean: round(change.mean, 5),
        changeP95: round(change.percentile(0.95)),
        bad: frames.reduce((sum, frame) => sum + frame.bad, 0),
    };
}

/** The light from `from` to `to` × its intensity, after settling at `from`: the lag to the settled image. */
async function stepCase(scene: Loaded, from: number, to: number): Promise<Record<string, number>> {
    const scale = scene.controls.lightScale;
    if (!scale) throw new Error('no light to scale');
    scale(from);
    await settle(scene, SETTLE_FRAMES);
    const before: number[] = [];
    let bad = 0;
    for (let index = 0; index < 10; index++) {
        scene.frame();
        const frame = await scene.read();
        before.push(frame.mean);
        bad += frame.bad;
    }
    scale(to);
    const means: number[] = [];
    const settled: number[] = [];
    for (let index = 1; index <= SETTLED_TO; index++) {
        scene.frame();
        if (index <= FOLLOW_FRAMES || index > SETTLED_FROM) {
            const frame = await scene.read();
            bad += frame.bad;
            if (index <= FOLLOW_FRAMES) means.push(frame.mean);
            else settled.push(frame.mean);
        } else if (index % 8 === 0) await scene.drain();
    }
    scale(1);
    const reference = settled.reduce((sum, mean) => sum + mean, 0) / settled.length;
    let lag = FOLLOW_FRAMES + 1;
    for (let t = FOLLOW_FRAMES; t >= 1; t--) {
        if (Math.abs(means[t - 1] / reference - 1) > LAG_WITHIN) break;
        lag = t;
    }
    const beforeMean = before.reduce((sum, mean) => sum + mean, 0) / before.length;
    return {
        lagFrames: lag,
        settledMeanE: round(reference, 5),
        settledOverBefore: round(reference / Math.max(beforeMean, 1e-12), 3),
        meanE1OverSettled: round(means[0] / reference),
        meanE16OverSettled: round(means[15] / reference),
        meanE40OverSettled: round(means[39] / reference),
        bad,
    };
}

/** The box crossing, then held: the error against the held image at the motion's end and 16 and 60 frames after. */
async function boxCase(scene: Loaded): Promise<Record<string, number>> {
    const move = scene.controls.boxAt;
    if (!move) throw new Error('no box to move');
    move(0);
    await settle(scene, SETTLE_FRAMES);
    for (let step = 1; step < MOTION_FRAMES; step++) {
        move(step);
        scene.frame();
        if (step % 8 === 0) await scene.drain();
    }
    move(MOTION_FRAMES);
    scene.frame();
    const end = await scene.read();
    const after: Record<number, Frame> = {};
    const held: Frame[] = [];
    for (let index = 1; index <= HOLD_FRAMES; index++) {
        scene.frame();
        if (index === 16 || index === 60) after[index] = await scene.read();
        else if (index > HOLD_FRAMES - 10) held.push(await scene.read());
        else if (index % 8 === 0) await scene.drain();
    }
    move(0);
    const reference = meanFrame(held);
    return {
        endError: imageError(end, reference),
        error16: imageError(after[16], reference),
        error60: imageError(after[60], reference),
        bad: [end, after[16], after[60], ...held].reduce((sum, frame) => sum + frame.bad, 0),
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
    const results: Record<string, Record<string, Record<string, unknown>>> = {};
    const skipped: string[] = [];
    const failures: string[] = [];
    for (const id of SCENES) {
        results[id] = {};
        for (const mode of MODES) {
            const scene = await load(device, id, mode);
            try {
                if (mode.settings.neuralCache === 'tail' && !scene.technique.neuralTail) {
                    skipped.push(`${id}/${mode.name}: the neural tail is off (the optional @pathtracer/mlp sibling is missing or off its pin)`);
                    continue;
                }
                const row: Record<string, unknown> = {
                    memoryBytes: scene.technique.memoryBytes,
                    worldCacheBytes: scene.technique.worldCacheBackend?.memoryBytes ?? 0,
                    ddgiBytes: scene.ddgi?.memoryBytes ?? 0,
                    secondBounceInUse: scene.technique.worldCache?.secondBounceInUse ?? null,
                };
                row.still = await stillCase(scene);
                row.doubled = await stepCase(scene, 1, 2);
                if (id === 'two-room') {
                    row.switchedOn = await stepCase(scene, 0, 1);
                    const first = await boxCase(scene);
                    const second = await boxCase(scene);
                    row.box = {
                        endError: round((first.endError + second.endError) / 2),
                        error16: round((first.error16 + second.error16) / 2),
                        error60: round((first.error60 + second.error60) / 2),
                        endErrorRuns: [round(first.endError), round(second.endError)],
                        bad: first.bad + second.bad,
                    };
                }
                for (const [name, value] of Object.entries(row)) {
                    const bad = (value as { bad?: number } | null)?.bad ?? 0;
                    if (bad > 0) failures.push(`${id}/${mode.name}/${name}: ${bad} pixels' E is NaN, infinite or negative`);
                }
                results[id][mode.name] = row;
                console.log(id, mode.name, JSON.stringify(row));
            } finally {
                scene.dispose();
                await device.queue.onSubmittedWorkDone();
            }
        }
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, compare: PAGE_COMPARE_VALUES, ddgiUnits: DDGI_UNITS, skipped, results },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
