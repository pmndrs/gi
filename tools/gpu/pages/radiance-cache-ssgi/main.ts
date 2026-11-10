/**
 * RC4.2: Capsaicin's SSGI technique on the GPU (packages/ddgi-core/src/
 * techniques/radiance-cache/wgsl/ssgi.ts), checked against its CPU twin
 * (ssgi-cpu.ts) and against what a corner must give. 1280 × 720, the
 * engine-free page screen (radiance-cache-probes/screen.ts), on a device
 * held at the platform's 8 storage buffers per stage.
 *
 * The corner: a 400 m floor (y = 0, albedo 0, so its last-frame lighting is
 * black) and a 40 × 10 m wall (z = 0, facing +z, albedo 0.75) under a black
 * sky, no lights, seen from (0, 2, 6); the page screen's last-frame lighting
 * is albedo × 2 (0 on the floor, 1.5 on the wall), so any near-field light
 * on the floor is the wall's.
 *
 * 1. **The twin:** every pixel's occlusion, bent normal and near-field light
 *    against the twin run on the depth, normals and lighting the GPU read,
 *    with the same random numbers. The twin marks a pixel sensitive where a
 *    decision sits at its threshold, or where its own result moves past half
 *    the tolerance when its positions are rounded to f32 as the GPU holds
 *    them (the file comment of ssgi-cpu.ts); those are counted, the rest
 *    must match (occlusion within 2e-3, each bent-normal component as stored,
 *    0.5 n + 0.5, within 2e-3, the near-field light within 2e-3 relative plus
 *    1e-4) but for at most 0.1% of them: the GPU's f32 arithmetic is not
 *    only its positions, and Capsaicin's kernel amplifies it (the truncation's
 *    1e-5 clamp; samples within a pixel), so a few pixels the f32 test passes
 *    still move. Over all non-sky pixels, sensitive ones included, the
 *    means of the occlusion and the bent normal's z agree within 0.1%, and
 *    the mean near-field light on the floor near the wall (the wall's light,
 *    check 2's pixels) within 1%. The whole image's near-field mean is
 *    reported, not checked: off the floor near the wall it is mostly a
 *    plane's light on itself through a raised horizon a hair high, which
 *    the truncation's world-origin intersection amplifies most (first run:
 *    1.5% apart on a mean of 0.002).
 * 2. **The corner:** floor pixels within 5 cm of the wall: the bent normal
 *    leans away from the wall (mean z > 0.1), the occlusion is below 1 (mean
 *    under 0.9) and the wall's light reaches them (mean near-field above 0);
 *    floor pixels 1 m or more from the wall see none of it (near-field 0).
 * 3. **The near-field light's scale (information, the design doc's question
 *    for AMD):** near the wall, the near-field light over (1 − occlusion) ×
 *    the wall's lit colour: 1 if it is the wall's radiance times the share of
 *    the cosine-weighted hemisphere it covers, i.e. irradiance / π.
 * 4. **An open floor (information):** occlusion on floor pixels far from the
 *    wall, whose samples all lie on the floor: below 1 where samples fall
 *    within a pixel or two (Capsaicin rebuilds a sample at its own uv with
 *    its texel's depth, ssgi.comp:194-196), worst at a grazing view.
 *
 * Also: no NaN or infinite value in either output. Correctness only, no
 * timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-ssgi`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { cameraRelativeInverseViewProjection } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/camera.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { ssgiConstants } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/ssgi.ts';
import { ssgiTwinPixel, type SSGIScreen, type SSGITolerance } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/ssgi-cpu.ts';
import { RADIANCE_CACHE_DEBUG_VIEW } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/debug.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';
import { quad } from '../radiance-cache-world-cache/corner.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const CAMERA: PageCamera = { position: [0, 2, 6], target: [0, 1, 0], fovYDeg: 60, near: 0.05, far: 1000 };
const FLOOR_ALBEDO = 0;
const WALL_ALBEDO = 0.75;
/** The page screen's lighting: albedo × this (radiance-cache-probes/screen.ts). */
const LIGHTING_SCALE = 2;
const WALL_LIGHTING = WALL_ALBEDO * LIGHTING_SCALE;
/** Frames before the check: the trees, then a few with them. */
const WARM_FRAMES = 3;
/** Check 1's tolerances (the file comment). */
const AO_TOLERANCE = 2e-3;
const BENT_TOLERANCE = 2e-3;
const NEAR_RELATIVE_TOLERANCE = 2e-3;
const NEAR_ABSOLUTE_TOLERANCE = 1e-4;
const TOLERANCE: SSGITolerance = { occlusion: AO_TOLERANCE, bentNormal: BENT_TOLERANCE, nearFieldRelative: NEAR_RELATIVE_TOLERANCE, nearFieldAbsolute: NEAR_ABSOLUTE_TOLERANCE };
/** Check 1: the share of compared pixels that may still differ, and how close the image means must be. */
const MISMATCH_SHARE = 1e-3;
const MEAN_TOLERANCE = { occlusion: 1e-3, bentZ: 1e-3, nearWallNearField: 1e-2 } as const;
/** Check 2's bands on the floor: within this of the wall, and at least this far. */
const NEAR_WALL = 0.05;
const FAR_FROM_WALL = 1;

type Vector = [number, number, number];

function frameInput(): FrameInput {
    return { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(CAMERA, WIDTH, HEIGHT) };
}

interface Readback {
    readonly screen: SSGIScreen;
    /** rgba per pixel: bent normal as 0.5 n + 0.5, occlusion. */
    readonly occlusion: Float32Array;
    /** rgba per pixel: near-field light. */
    readonly nearField: Float32Array;
    readonly frameIndex: number;
}

async function readFrame(device: GPUDevice, technique: RadianceCacheTechnique, screen: PageScreen): Promise<Readback> {
    const probes = technique.probes;
    const previousLighting = screen.screen.previousLighting;
    if (!probes || !previousLighting) throw new Error('no SSGI outputs after encodeScreen');
    const depth = new Float32Array((await readTexture(device, screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
    const normal = halves(await readTexture(device, screen.screen.normal, 8, WIDTH, HEIGHT));
    const lighting = halves(await readTexture(device, previousLighting, 8, WIDTH, HEIGHT));
    const camera = frameCamera(CAMERA, WIDTH, HEIGHT);
    return {
        screen: {
            width: WIDTH,
            height: HEIGHT,
            depth,
            normal,
            lighting,
            lightingWidth: WIDTH,
            lightingHeight: HEIGHT,
            farDepth: 1,
            inverseViewProjection: cameraRelativeInverseViewProjection(camera),
            eye: camera.position,
            frameIndex: probes.frameIndex,
        },
        occlusion: halves(await readTexture(device, probes.occlusionAndBentNormal, 8, WIDTH, HEIGHT)),
        nearField: halves(await readTexture(device, probes.nearField, 8, WIDTH, HEIGHT)),
        frameIndex: probes.frameIndex,
    };
}

/** The world point under a pixel (f64), for telling floor from wall and the distance to the wall. */
function worldAt(screen: SSGIScreen, x: number, y: number): Vector {
    const m = screen.inverseViewProjection;
    const ndc = [(2 * (x + 0.5)) / WIDTH - 1, 1 - (2 * (y + 0.5)) / HEIGHT, screen.depth[x + y * WIDTH], 1];
    const h = [0, 1, 2, 3].map((row) => m[row] * ndc[0] + m[4 + row] * ndc[1] + m[8 + row] * ndc[2] + m[12 + row] * ndc[3]);
    return [screen.eye[0] + h[0] / h[3], screen.eye[1] + h[1] / h[3], screen.eye[2] + h[2] / h[3]];
}

const mean = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

/** Draws one of the technique's debug views into a canvas on the page, for the runner's screenshot. */
async function showDebugView(device: GPUDevice, technique: RadianceCacheTechnique, view: number, caption: string, column: HTMLElement): Promise<void> {
    const target = device.createTexture({ size: [WIDTH, HEIGHT], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    technique.encodeDebugView(encoder, view as Parameters<RadianceCacheTechnique['encodeDebugView']>[1], target);
    device.queue.submit([encoder.finish()]);
    const pixels = await readTexture(device, target, 4, WIDTH, HEIGHT);
    target.destroy();
    const canvas = document.createElement('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = '480px';
    canvas.style.display = 'block';
    canvas.getContext('2d')?.putImageData(new ImageData(Uint8ClampedArray.from(pixels), WIDTH, HEIGHT), 0, 0);
    const label = document.createElement('div');
    label.textContent = caption;
    label.style.font = '11px sans-serif';
    label.style.color = '#ddd';
    column.append(label, canvas);
}

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const { device, adapterStorageBuffersPerStage } = await requestBudgetDevice(adapter);
    window.addEventListener('pagehide', () => device.destroy(), { once: true });
    deviceErrors = watchDevice(device);
    document.body.style.background = '#222';
    document.body.style.margin = '0';
    const gallery = document.createElement('div');
    gallery.style.display = 'flex';
    gallery.style.gap = '8px';
    document.body.append(gallery);
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 30) failures.push(message);
    };

    const platform = new GIPlatformImpl(device, { label: 'radiance-cache-ssgi' });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    const half = 200;
    quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], FLOOR_ALBEDO);
    quad(platform, 'wall', [[-20, 0, 0], [20, 0, 0], [20, 10, 0], [-20, 10, 0]], [0, 0, 1], WALL_ALBEDO);
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    screen.lightingScale = LIGHTING_SCALE;
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: 'radiance-cache-ssgi' });
        platform.encode(encoder, frameInput());
        screen.encode(encoder, CAMERA);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    frame();
    await platform.rays.whenBuilt();
    for (let index = 0; index < WARM_FRAMES; index++) frame();
    await device.queue.onSubmittedWorkDone();

    const readback = await readFrame(device, technique, screen);
    const constants = ssgiConstants(frameCamera(CAMERA, WIDTH, HEIGHT), DEFAULT_RADIANCE_CACHE_SETTINGS);

    // 1. The twin, and 2-4 from the same pixels.
    let compared = 0;
    let sensitive = 0;
    let sensitiveMatching = 0;
    const sums = { gpuOcclusion: 0, twinOcclusion: 0, gpuBentZ: 0, twinBentZ: 0, gpuNear: 0, twinNear: 0, pixels: 0 };
    let sky = 0;
    let mismatches = 0;
    let nonFinite = 0;
    let zeroLengthSamples = 0;
    const worst = { occlusion: 0, bentNormal: 0, nearField: 0 };
    const examples: string[] = [];
    const nearWall = { bentZ: [] as number[], occlusion: [] as number[], nearField: [] as number[], twinNearField: [] as number[], scale: [] as number[] };
    const farFloor = { occlusion: [] as number[], nearField: [] as number[] };
    for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
            const pixel = x + y * WIDTH;
            const gpuOcclusion = readback.occlusion.subarray(pixel * 4, pixel * 4 + 4);
            const gpuNear = readback.nearField.subarray(pixel * 4, pixel * 4 + 4);
            if (![...gpuOcclusion, ...gpuNear].every(Number.isFinite)) {
                nonFinite++;
                continue;
            }
            const twin = ssgiTwinPixel(readback.screen, constants, x, y, TOLERANCE);
            if (!twin) {
                sky++;
                if (gpuOcclusion.some((value) => value !== 0) || gpuNear.some((value) => value !== 0)) {
                    mismatches++;
                    fail(`pixel ${x}, ${y}: a sky pixel's outputs are not zero`);
                }
                continue;
            }
            zeroLengthSamples += twin.zeroLengthSamples;
            const occlusionError = Math.abs(gpuOcclusion[3] - twin.ambientOcclusion);
            const bentError = Math.max(...[0, 1, 2].map((axis) => Math.abs(gpuOcclusion[axis] - (0.5 * twin.bentNormal[axis] + 0.5))));
            const nearError = Math.max(...[0, 1, 2].map((axis) => Math.abs(gpuNear[axis] - twin.nearField[axis]) / (NEAR_RELATIVE_TOLERANCE * Math.abs(twin.nearField[axis]) + NEAR_ABSOLUTE_TOLERANCE)));
            sums.pixels++;
            sums.gpuOcclusion += gpuOcclusion[3];
            sums.twinOcclusion += twin.ambientOcclusion;
            sums.gpuBentZ += 2 * gpuOcclusion[2] - 1;
            sums.twinBentZ += twin.bentNormal[2];
            sums.gpuNear += gpuNear[0];
            sums.twinNear += twin.nearField[0];
            const differs = occlusionError > AO_TOLERANCE || bentError > BENT_TOLERANCE || nearError > 1;
            if (twin.sensitive) {
                sensitive++;
                if (!differs) sensitiveMatching++;
            } else {
                compared++;
                worst.occlusion = Math.max(worst.occlusion, occlusionError);
                worst.bentNormal = Math.max(worst.bentNormal, bentError);
                worst.nearField = Math.max(worst.nearField, nearError);
                if (differs) {
                    mismatches++;
                    if (mismatches <= 3) {
                        examples.push(`pixel ${x}, ${y}: GPU ao ${gpuOcclusion[3].toFixed(4)} bent ${[0, 1, 2].map((a) => gpuOcclusion[a].toFixed(3))} near ${gpuNear[0].toPrecision(4)}; twin ${twin.ambientOcclusion.toFixed(4)} ${twin.bentNormal.map((v) => (0.5 * v + 0.5).toFixed(3))} ${twin.nearField[0].toPrecision(4)}`);
                    }
                }
            }
            // The floor by its stored normal (the page writes the geometric normal).
            const isFloor = readback.screen.normal[pixel * 4 + 1] > 0.99;
            if (!isFloor) continue;
            const z = worldAt(readback.screen, x, y)[2];
            const bentZ = 2 * gpuOcclusion[2] - 1;
            if (z <= NEAR_WALL) {
                nearWall.bentZ.push(bentZ);
                nearWall.occlusion.push(gpuOcclusion[3]);
                nearWall.nearField.push(gpuNear[0]);
                nearWall.twinNearField.push(twin.nearField[0]);
                if (gpuOcclusion[3] < 0.95) nearWall.scale.push(gpuNear[0] / ((1 - gpuOcclusion[3]) * WALL_LIGHTING));
            } else if (z >= FAR_FROM_WALL) {
                farFloor.occlusion.push(gpuOcclusion[3]);
                farFloor.nearField.push(gpuNear[0]);
            }
        }
    }
    if (mismatches > MISMATCH_SHARE * compared) fail(`twin: ${mismatches} of ${compared} compared pixels differ (at most ${MISMATCH_SHARE * 100}%): ${examples.join('; ')}`);
    if (nonFinite > 0) fail(`${nonFinite} pixels have a NaN or infinite output`);
    if (compared < (WIDTH * HEIGHT) / 4) fail(`twin: only ${compared} pixels compared (${sensitive} sensitive)`);
    const means = {
        gpuOcclusion: sums.gpuOcclusion / sums.pixels,
        twinOcclusion: sums.twinOcclusion / sums.pixels,
        gpuBentZ: sums.gpuBentZ / sums.pixels,
        twinBentZ: sums.twinBentZ / sums.pixels,
        gpuNearField: sums.gpuNear / sums.pixels,
        twinNearField: sums.twinNear / sums.pixels,
    };
    const relative = (a: number, b: number): number => Math.abs(a - b) / Math.max(Math.abs(b), 1e-9);
    if (relative(means.gpuOcclusion, means.twinOcclusion) > MEAN_TOLERANCE.occlusion) fail(`twin: mean occlusion ${means.gpuOcclusion} against ${means.twinOcclusion}`);
    if (Math.abs(means.gpuBentZ - means.twinBentZ) > MEAN_TOLERANCE.bentZ) fail(`twin: mean bent-normal z ${means.gpuBentZ} against ${means.twinBentZ}`);
    const nearWallMeans = { gpu: mean(nearWall.nearField), twin: mean(nearWall.twinNearField) };
    if (relative(nearWallMeans.gpu, nearWallMeans.twin) > MEAN_TOLERANCE.nearWallNearField) fail(`twin: mean near-field light near the wall ${nearWallMeans.gpu} against ${nearWallMeans.twin}`);

    // 2. The corner.
    const corner = {
        nearWallPixels: nearWall.bentZ.length,
        meanBentZ: mean(nearWall.bentZ),
        meanOcclusion: mean(nearWall.occlusion),
        meanNearField: mean(nearWall.nearField),
        farFloorPixels: farFloor.occlusion.length,
        farFloorMaxNearField: farFloor.nearField.reduce((largest, value) => Math.max(largest, value), 0),
    };
    if (corner.nearWallPixels < 1000) fail(`corner: only ${corner.nearWallPixels} floor pixels within ${NEAR_WALL} m of the wall`);
    if (!(corner.meanBentZ > 0.1)) fail(`corner: the bent normal near the wall leans ${corner.meanBentZ.toFixed(3)} away from it (expected above 0.1)`);
    if (!(corner.meanOcclusion < 0.9)) fail(`corner: the occlusion near the wall is ${corner.meanOcclusion.toFixed(3)} (expected below 0.9)`);
    if (!(corner.meanNearField > 0)) fail('corner: no near-field light from the wall');
    const farLit = farFloor.nearField.filter((value) => value !== 0).length;
    if (farLit > 0) fail(`corner: ${farLit} floor pixels 1 m or more from the wall have near-field light`);

    // 3, 4: information.
    const sortedScale = [...nearWall.scale].sort((a, b) => a - b);
    const sortedFar = [...farFloor.occlusion].sort((a, b) => a - b);

    for (const [view, caption] of [
        [RADIANCE_CACHE_DEBUG_VIEW.occlusion, 'SSGI occlusion'],
        [RADIANCE_CACHE_DEBUG_VIEW.bentNormal, 'SSGI bent normal'],
    ] as const) {
        const column = document.createElement('div');
        gallery.append(column);
        await showDebugView(device, technique, view, caption, column);
    }

    screen.dispose();
    platform.dispose();
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: {
            width: WIDTH,
            height: HEIGHT,
            adapterStorageBuffersPerStage,
            frameIndex: readback.frameIndex,
            twin: {
                compared,
                mismatches,
                sensitive,
                sensitiveMatching,
                sky,
                nonFinite,
                zeroLengthSamples,
                worstOcclusionError: worst.occlusion,
                worstBentNormalError: worst.bentNormal,
                worstNearFieldErrorOverTolerance: worst.nearField,
                means,
                nearWallNearFieldMeans: nearWallMeans,
                examples,
            },
            corner,
            nearFieldScale: { pixels: sortedScale.length, median: sortedScale[Math.floor(sortedScale.length / 2)], p10: sortedScale[Math.floor(sortedScale.length / 10)], p90: sortedScale[Math.floor((sortedScale.length * 9) / 10)] },
            openFloor: { pixels: sortedFar.length, meanOcclusion: mean(farFloor.occlusion), p5: sortedFar[Math.floor(sortedFar.length / 20)], belowHalf: sortedFar.filter((value) => value < 0.5).length },
        },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
