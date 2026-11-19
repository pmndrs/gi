/**
 * RC4.1: how steady the radiance cache's irradiance is from frame to frame, a
 * measurement (recorded, not a gate: DECISIONS 2026-11-19 asks for temporal
 * stability; the target is Dennis's to set). Only NaN, infinite or negative
 * values and device errors fail the page.
 *
 * On `cornell-box` and `two-room` (from their manifests) and `sponza` (with curtains, through the
 * viewer's loader and the three adapter's scene sync, as a loader only), each
 * at 1280 × 720 from its first bookmark, the `FrameScreen` drawn by primary
 * rays (radiance-cache-probes/screen.ts):
 *
 * - **still:** the camera at the bookmark;
 * - **moving:** the camera sliding sideways at 0.5 m/s (60 frames a second)
 *   and turning 3 degrees a second.
 *
 * Each with the interpolation's jitter on (the product) and off (RC4.1 review: the jitter is
 * meant for the GI denoiser, RC4.3), and with the screen probes' history on (RC4.1) and off (the history
 * dropped before every frame, so each frame runs as the first does; the world
 * cache keeps its cells in both). After `CONVERGE_FRAMES` frames, over
 * `MEASURE_FRAMES` frames, per surface pixel: the relative change of the
 * interpolated irradiance E from the previous frame, |E_t − E_t−1| /
 * max(E_t−1, 5% of the frame's mean E), as the mean and the 95th percentile
 * over every pixel and frame, and the mean E itself. Moving, the previous
 * frame's E is read at the pixel the velocity points back to; a pixel whose
 * previous pixel shows another surface (a disocclusion) is left out.
 *
 * Through the lock, one page at a time:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc4.1 stability" -- pnpm test:gpu --filter radiance-cache-stability --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { cameraRelativeInverseViewProjection } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/camera.ts';
import { halfBitsToFloat } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
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
/** Frames before the warm world cache is measured on, then frames per configuration to settle, then to measure. */
const WARM_FRAMES = 60;
const CONVERGE_FRAMES = 30;
const MEASURE_FRAMES = 40;
/** The moving camera: metres and degrees per frame (0.5 m/s and 3 degrees/s at 60 frames a second). */
const SLIDE_PER_FRAME = 0.5 / 60;
const TURN_PER_FRAME = 3 / 60;
/** Camera motion and screen-probe history, each measured with the interpolation jitter on and off. */
const CONFIGURATIONS = [
    ['still', 'on'],
    ['still', 'off'],
    ['moving', 'on'],
    ['moving', 'off'],
] as const;
/**
 * RC4.2: the probe filter and SSGI on (the product), the filter alone, and both off (RC4.1's
 * image), each measured with history; history off only for the first.
 */
const VARIANTS = [
    { name: 'filter and SSGI', probeFilter: true, ssgi: true },
    { name: 'filter only', probeFilter: true, ssgi: false },
    { name: 'neither (RC4.1)', probeFilter: false, ssgi: false },
] as const;
/** A pixel's change is relative to the larger of its previous E and this share of the frame's mean E. */
const DARK_FLOOR = 0.05;

type Vector = [number, number, number];

const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));
function halvesFast(bytes: Uint8Array): Float32Array {
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const out = new Float32Array(words.length);
    for (let index = 0; index < words.length; index++) out[index] = HALF_TABLE[words[index]];
    return out;
}

/** Relative changes pooled over pixels and frames: their sum and a histogram (steps of 0.0005 up to 5) for the percentile. */
class ChangeTally {
    static readonly STEP = 0.0005;
    readonly bins = new Uint32Array(10001);
    sum = 0;
    count = 0;

    add(value: number): void {
        if (!Number.isFinite(value)) return;
        this.sum += value;
        this.count++;
        this.bins[Math.min(Math.floor(value / ChangeTally.STEP), this.bins.length - 1)]++;
    }

    /** The upper edge of the bin holding the fraction's value. */
    percentile(fraction: number): number {
        const target = fraction * this.count;
        let running = 0;
        for (let bin = 0; bin < this.bins.length; bin++) {
            running += this.bins[bin];
            if (running >= target) return (bin + 1) * ChangeTally.STEP;
        }
        return Infinity;
    }
}

/** Sponza through the viewer's loader, copied into the platform by the three adapter's scene sync (a loader here). */
async function describeLoaded(platform: GIPlatformImpl, device: GPUDevice, manifest: SceneManifest): Promise<() => void> {
    const loaded = await loadScene(manifest, () => {});
    const content = new Set<object>();
    loaded.content.traverse((object) => content.add(object));
    loaded.scene.traverse((object) => {
        if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    loaded.scene.updateMatrixWorld();
    const sync = new SceneSync(platform.scene, { device, warn: () => {} });
    sync.environment = platformSkyEnvironment(manifest);
    return () => sync.update(loaded.scene);
}

/** The bookmark camera after `step` frames of the slow move: sideways along the view's right, turning about y. */
function movingCamera(base: PageCamera, step: number): PageCamera {
    const forward = [base.target[0] - base.position[0], base.target[1] - base.position[1], base.target[2] - base.position[2]];
    const flat = Math.hypot(forward[0], forward[2]) || 1;
    const right = [-forward[2] / flat, 0, forward[0] / flat];
    const position: Vector = [base.position[0] + right[0] * SLIDE_PER_FRAME * step, base.position[1], base.position[2] + right[2] * SLIDE_PER_FRAME * step];
    const angle = (TURN_PER_FRAME * step * Math.PI) / 180;
    const target: Vector = [position[0] + forward[0] * Math.cos(angle) + forward[2] * Math.sin(angle), position[1] + forward[1], position[2] - forward[0] * Math.sin(angle) + forward[2] * Math.cos(angle)];
    return { ...base, position, target };
}

interface FrameImage {
    readonly e: Float32Array;
    readonly depth: Float32Array;
    readonly velocity: Float32Array | null;
    readonly camera: PageCamera;
}

interface Measurement {
    readonly meanChange: number;
    readonly p95Change: number;
    readonly meanE: number;
    readonly pixels: number;
    readonly leftOut: number;
    readonly bad: number;
}

/** World position of a pixel from a depth readback and the camera it was drawn with. */
function worldOf(depth: Float32Array, camera: PageCamera): (x: number, y: number) => Vector {
    const frame = frameCamera(camera, WIDTH, HEIGHT);
    const inverse = cameraRelativeInverseViewProjection(frame);
    const eye = frame.position;
    return (x, y) => {
        const ndc = [(2 * (x + 0.5)) / WIDTH - 1, 2 * (1 - (y + 0.5) / HEIGHT) - 1, depth[y * WIDTH + x], 1];
        const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row] * ndc[3]);
        return [eye[0] + h[0] / h[3], eye[1] + h[1] / h[3], eye[2] + h[2] / h[3]];
    };
}

/** Relative changes of E between two frames, at the pixel the velocity points back to (still: the same pixel). */
function changes(previous: FrameImage, current: FrameImage, tally: ChangeTally): { leftOut: number } {
    let leftOut = 0;
    let previousSum = 0;
    let previousCount = 0;
    for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
        if (previous.depth[pixel] === 1) continue;
        previousSum += previous.e[pixel];
        previousCount++;
    }
    const floor = DARK_FLOOR * (previousSum / Math.max(previousCount, 1));
    const worldNow = current.velocity ? worldOf(current.depth, current.camera) : null;
    const worldBefore = current.velocity ? worldOf(previous.depth, previous.camera) : null;
    const pixelSize = (2 * Math.tan((current.camera.fovYDeg * Math.PI) / 360)) / HEIGHT;
    for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
            const pixel = y * WIDTH + x;
            if (current.depth[pixel] === 1) continue;
            let source = pixel;
            if (current.velocity && worldNow && worldBefore) {
                const vx = current.velocity[pixel * 4];
                const vy = current.velocity[pixel * 4 + 1];
                const px = Math.trunc(((x + 0.5) / WIDTH - 0.5 * vx) * WIDTH);
                const py = Math.trunc(((y + 0.5) / HEIGHT + 0.5 * vy) * HEIGHT);
                if (px < 0 || py < 0 || px >= WIDTH || py >= HEIGHT || previous.depth[py * WIDTH + px] === 1) {
                    leftOut++;
                    continue;
                }
                const now = worldNow(x, y);
                const before = worldBefore(px, py);
                const distance = Math.hypot(now[0] - current.camera.position[0], now[1] - current.camera.position[1], now[2] - current.camera.position[2]);
                if (Math.hypot(now[0] - before[0], now[1] - before[1], now[2] - before[2]) > 3 * pixelSize * distance) {
                    leftOut++;
                    continue;
                }
                source = py * WIDTH + px;
            }
            const before = previous.e[source];
            tally.add(Math.abs(current.e[pixel] - before) / Math.max(before, floor));
        }
    }
    return { leftOut };
}

async function measureScene(device: GPUDevice, id: string, loader: 'manifest' | 'viewer'): Promise<Record<string, Measurement>> {
    const manifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const base: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-stability ${id}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    let beforeFrame = (): void => {};
    if (loader === 'viewer') beforeFrame = await describeLoaded(platform, device, manifest);
    else describeScene(platform, manifest);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    let camera = base;
    const frame = (): void => {
        beforeFrame();
        const encoder = device.createCommandEncoder({ label: `radiance-cache-stability ${id}` });
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    const readFrame = async (withVelocity: boolean): Promise<FrameImage> => {
        const probes = technique.probes;
        if (!probes) throw new Error('no probes after encodeScreen');
        const irradiance = halvesFast(await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT));
        const e = new Float32Array(WIDTH * HEIGHT);
        for (let pixel = 0; pixel < e.length; pixel++) e[pixel] = (irradiance[pixel * 4] + irradiance[pixel * 4 + 1] + irradiance[pixel * 4 + 2]) / 3;
        const depth = new Float32Array((await readTexture(device, screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
        const velocityTexture = screen.screen.velocity;
        const velocity = withVelocity && velocityTexture ? halvesFast(await readTexture(device, velocityTexture, 8, WIDTH, HEIGHT)) : null;
        return { e, depth, velocity, camera };
    };
    try {
        frame();
        await platform.rays.whenBuilt();
        technique.resetHistory();
        for (let index = 0; index < WARM_FRAMES; index++) {
            frame();
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        const results: Record<string, Measurement> = {};
        for (const variant of VARIANTS) {
            technique.settings.probeFilter = variant.probeFilter;
            technique.settings.ssgi = variant.ssgi;
            for (const jitter of [true, false]) {
                for (const [motion, history] of CONFIGURATIONS) {
                    if (history === 'off' && variant !== VARIANTS[0]) continue;
                    technique.settings.interpolationJitter = jitter;
                    technique.resetScreenHistory();
                    let step = 0;
                    const advance = (): void => {
                        if (history === 'off') technique.resetScreenHistory();
                        if (motion === 'moving') camera = movingCamera(base, step++);
                        frame();
                    };
                    camera = base;
                    for (let index = 0; index < CONVERGE_FRAMES; index++) {
                        advance();
                        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
                    }
                    await device.queue.onSubmittedWorkDone();
                    let previous = await readFrame(motion === 'moving');
                    const tally = new ChangeTally();
                    let leftOut = 0;
                    let eSum = 0;
                    let eCount = 0;
                    let bad = 0;
                    for (let index = 0; index < MEASURE_FRAMES; index++) {
                        advance();
                        await device.queue.onSubmittedWorkDone();
                        const current = await readFrame(motion === 'moving');
                        for (let pixel = 0; pixel < current.e.length; pixel++) {
                            if (current.depth[pixel] === 1) continue;
                            const value = current.e[pixel];
                            if (!(Number.isFinite(value) && value >= 0)) bad++;
                            else {
                                eSum += value;
                                eCount++;
                            }
                        }
                        leftOut += changes(previous, current, tally).leftOut;
                        previous = current;
                    }
                    results[`${variant.name}: ${motion}, history ${history}, jitter ${jitter ? 'on' : 'off'}`] = {
                        meanChange: Number((tally.sum / Math.max(tally.count, 1)).toFixed(5)),
                        p95Change: Number(tally.percentile(0.95).toFixed(4)),
                        meanE: Number((eSum / Math.max(eCount, 1)).toPrecision(5)),
                        pixels: tally.count,
                        leftOut,
                        bad,
                    };
                }
            }
        }
        technique.settings.interpolationJitter = true;
        technique.settings.probeFilter = true;
        technique.settings.ssgi = true;
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
    const scenes: Record<string, Record<string, Measurement>> = {};
    const failures: string[] = [];
    for (const [id, loader] of [['cornell-box', 'manifest'], ['two-room', 'manifest'], ['sponza', 'viewer']] as const) {
        scenes[id] = await measureScene(device, id, loader);
        for (const [name, measurement] of Object.entries(scenes[id])) {
            if (measurement.bad > 0) failures.push(`${id}, ${name}: ${measurement.bad} pixels' E is NaN, infinite or negative`);
            if (measurement.pixels === 0) failures.push(`${id}, ${name}: no pixel measured`);
        }
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, convergeFrames: CONVERGE_FRAMES, measureFrames: MEASURE_FRAMES, scenes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
