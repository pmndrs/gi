/**
 * RC2.2: the radiance cache's image passes on the GPU (`FilterProbeMask`,
 * `ProjectScreenProbes`, `InterpolateScreenProbes`), checked against
 * analytic answers. Engine-free: the `FrameScreen` is drawn by primary rays
 * through the platform's tracer (radiance-cache-probes/screen.ts), on a
 * 400 m floor seen from 3 m up at 1280 × 720, so every probe's 64 rays see
 * only the sky.
 *
 * 1. **Lambert** (RC2.2 done-when): under a constant sky of radiance L, the
 *    interpolated irradiance of every floor pixel is E = πL within 1%, as the
 *    mean and the 95th percentile of the per-pixel error.
 * 2. **A direction-varying sky**, L(d) = 1 + 0.6 d.x − 0.3 d.z (an
 *    equirectangular image the platform bakes): each probe's nine SH
 *    coefficients against the hemisphere's SH integrated finely on the CPU,
 *    and the interpolated E against the CPU's E from those coefficients. The
 *    sky changes along both tangents of the floor's probe frame, so an atlas
 *    written transposed in its tile, or read so by the projection, moves the
 *    z-band coefficients by about 30% of the largest (the CPU's own figure
 *    for a transposed probe is reported).
 * 3. On both: the mask's mip chain holds, at every level, a probe of its
 *    2 × 2 children when any has one (`FilterProbeMask`); the irradiance has
 *    no negative, infinite or NaN value.
 *
 * The page draws the probe atlas, the mask and the SH view of the varying
 * sky (the technique's engine-free debug views) for the runner's screenshot.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc2.2 image" -- pnpm test:gpu --filter radiance-cache-image --gpu nvidia` (or intel).
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { equirectUvToDirection } from '../../../../packages/ddgi-core/src/platform/sky/equirect.ts';
import { INVALID_ID, integrateHemisphereSH, projectProbe, probeCellDirection, shIrradiance, unpackSeed, type Direction } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { PROBE_SIZE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { RADIANCE_CACHE_DEBUG_VIEW, type RadianceCacheDebugViewId } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/debug.ts';
import { PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-probes.ts';
import { SH_BLOCK_SIZE, SH_COEFFICIENTS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/spherical-harmonics.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { readBuffer } from '../ddgi-shared/harness.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { PageScreen, frameCamera, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
/** The Lambert check's tolerance on E / (π L), for the mean and the 95th percentile (RC2.2 done-when). */
const LAMBERT_TOLERANCE = 0.01;
/** The constant sky's radiance: not 1, so a lost or doubled factor can't hide. */
const CONSTANT_SKY = 1.5;
/** The varying sky: L(d) = 1 + 0.6 d.x − 0.3 d.z, at least 0.33 everywhere. */
const varyingSky = (direction: Direction): number => 1 + 0.6 * direction[0] - 0.3 * direction[2];
/**
 * The varying sky's SH, against the fine integral, as a share of the largest
 * coefficient. Averaged over every probe: 2.5% (the CPU's 64-cell projection
 * of the same sky is within 1.7%; screen-irradiance.test.ts). Per probe the
 * one random ray per cell adds noise, so the 95th percentile of each probe's
 * worst coefficient gets 6%. A transposed probe is off by about 30%.
 */
const SH_MEAN_TOLERANCE = 0.025;
const SH_PROBE_P95_TOLERANCE = 0.06;
/** The varying sky's interpolated E against the CPU's E from the reference SH: mean and 95th percentile. */
const VARYING_E_TOLERANCE = 0.02;
/** Equirect image of the varying sky the page hands the platform (it bakes its own 256 × 128 map from it). */
const SKY_IMAGE = { width: 512, height: 256 } as const;
/** The floor's normal: every probe's frame. */
const UP: Direction = [0, 1, 0];
const CAMERA: PageCamera = { position: [0, 3, 0], target: [0, 1, -10], fovYDeg: 60, near: 0.1, far: 1000 };

interface ImageRun {
    readonly technique: RadianceCacheTechnique;
    readonly platform: GIPlatformImpl;
    readonly screen: PageScreen;
    dispose(): void;
}

function frameInput(): FrameInput {
    return { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(CAMERA, WIDTH, HEIGHT) };
}

/** A 400 m floor at y = 0 under `environment`, no lights; frames until the trees are built, then one more. */
async function runFloor(device: GPUDevice, name: string, environment: Parameters<GIPlatformImpl['scene']['setEnvironment']>[0]): Promise<ImageRun> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-image ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    const half = 200;
    const scene = platform.scene;
    const mesh = scene.createMesh({
        positions: new Float32Array([-half, 0, -half, half, 0, -half, half, 0, half, -half, 0, half]),
        normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
        indices: new Uint32Array([0, 2, 1, 0, 3, 2]),
        label: 'floor',
    });
    const material = scene.createMaterial({ albedo: [0.5, 0.5, 0.5], label: 'floor' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'floor' });
    scene.setEnvironment(environment);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `radiance-cache-image ${name}` });
        platform.encode(encoder, frameInput());
        screen.encode(encoder, CAMERA);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    frame();
    await platform.rays.whenBuilt();
    frame();
    frame();
    await device.queue.onSubmittedWorkDone();
    return {
        technique,
        platform,
        screen,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

/**
 * The varying sky as an equirect texture in the platform's GPU order: memory
 * row 0 is the nadir, the mirror of the image order `equirectUvToDirection`
 * takes (equirect.ts).
 */
function varyingSkyTexture(device: GPUDevice): GPUTexture {
    const texture = device.createTexture({ label: 'varying sky', size: [SKY_IMAGE.width, SKY_IMAGE.height], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const texels = new Float32Array(SKY_IMAGE.width * SKY_IMAGE.height * 4);
    for (let row = 0; row < SKY_IMAGE.height; row++) {
        for (let column = 0; column < SKY_IMAGE.width; column++) {
            const imageRow = SKY_IMAGE.height - 1 - row;
            const direction = equirectUvToDirection({ u: (column + 0.5) / SKY_IMAGE.width, v: (imageRow + 0.5) / SKY_IMAGE.height });
            const radiance = varyingSky(direction);
            texels.set([radiance, radiance, radiance, 1], (row * SKY_IMAGE.width + column) * 4);
        }
    }
    device.queue.writeTexture({ texture }, texels, { bytesPerRow: SKY_IMAGE.width * 16 }, [SKY_IMAGE.width, SKY_IMAGE.height]);
    return texture;
}

/** The largest value, by a loop: a spread of a screen's pixels would overflow the call stack. */
function largest(values: readonly number[]): number {
    let result = -Infinity;
    for (const value of values) result = Math.max(result, value);
    return result;
}

function percentile(values: number[], fraction: number): number {
    if (values.length === 0) return NaN;
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

interface Readback {
    readonly isSky: (pixel: number) => boolean;
    /** rgba per pixel: E in rgb, the denoiser hint in alpha. */
    readonly irradiance: Float32Array;
    readonly seeds: number[];
    readonly probeSH: Uint32Array;
    readonly shWidth: number;
}

async function readRun(device: GPUDevice, run: ImageRun): Promise<Readback> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const depth = new Float32Array((await readTexture(device, run.screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
    const irradiance = halves(await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT));
    const spawnWords = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + probes.layout.maxProbeSpawnCount * 4));
    const seeds = Array.from(spawnWords.subarray(1, 1 + spawnWords[0]));
    const shWidth = probes.layout.probeCount[0] * SH_BLOCK_SIZE;
    const probeSH = new Uint32Array((await readTexture(device, probes.probeSH, 8, shWidth, probes.layout.probeCount[1] * SH_BLOCK_SIZE)).buffer);
    return { isSky: (pixel) => depth[pixel] === 1, irradiance, seeds, probeSH, shWidth };
}

/** A probe tile's nine coefficients (rgb averaged) from the SH texture's 3 × 3 block (register D26). */
function probeCoefficients(readback: Readback, seed: number): number[] {
    const [x, y] = unpackSeed(seed);
    const tile = [Math.floor(x / PROBE_SIZE), Math.floor(y / PROBE_SIZE)];
    const coefficients: number[] = [];
    for (let index = 0; index < SH_COEFFICIENTS; index++) {
        const texel = (tile[1] * SH_BLOCK_SIZE + Math.floor(index / SH_BLOCK_SIZE)) * readback.shWidth + tile[0] * SH_BLOCK_SIZE + (index % SH_BLOCK_SIZE);
        const words = new Uint32Array([readback.probeSH[texel * 2], readback.probeSH[texel * 2 + 1]]);
        const rgbw = halves(new Uint8Array(words.buffer));
        coefficients.push((rgbw[0] + rgbw[1] + rgbw[2]) / 3);
    }
    return coefficients;
}

/** Floor pixels' E as a ratio to `expected` (rgb averaged), with counts of bad values. */
function irradianceRatios(readback: Readback, expected: number): { ratios: number[]; nonFinite: number; negative: number; backupHints: number } {
    const ratios: number[] = [];
    let nonFinite = 0;
    let negative = 0;
    let backupHints = 0;
    for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
        const rgba = readback.irradiance.subarray(pixel * 4, pixel * 4 + 4);
        if (!Array.from(rgba).every(Number.isFinite)) {
            nonFinite++;
            continue;
        }
        if (rgba[0] < 0 || rgba[1] < 0 || rgba[2] < 0) negative++;
        if (readback.isSky(pixel)) continue;
        if (rgba[3] === 0) backupHints++;
        ratios.push((rgba[0] + rgba[1] + rgba[2]) / 3 / expected);
    }
    return { ratios, nonFinite, negative, backupHints };
}

/** FilterProbeMask: every texel of level i holds a probe of its 2 × 2 children at i − 1 when any has one, else none. */
async function checkMaskMips(device: GPUDevice, run: ImageRun, fail: (message: string) => void): Promise<{ levels: number; mismatches: number }> {
    const probes = run.technique.probes;
    if (!probes) return { levels: 0, mismatches: 1 };
    const { probeCount, probeMaskMipCount } = probes.layout;
    let previous = new Uint32Array((await readTexture(device, probes.probeMask, 4, probeCount[0], probeCount[1], 0)).buffer);
    let previousSize = [probeCount[0], probeCount[1]];
    let mismatches = 0;
    for (let level = 1; level < probeMaskMipCount; level++) {
        const size = [Math.max(1, probeCount[0] >> level), Math.max(1, probeCount[1] >> level)];
        const current = new Uint32Array((await readTexture(device, probes.probeMask, 4, size[0], size[1], level)).buffer);
        for (let y = 0; y < size[1]; y++) {
            for (let x = 0; x < size[0]; x++) {
                // Capsaicin's rule (gi1.comp:258-281): the first child, in row order, that holds a probe.
                let expected = INVALID_ID;
                for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
                    const cx = 2 * x + dx;
                    const cy = 2 * y + dy;
                    if (cx < previousSize[0] && cy < previousSize[1] && previous[cy * previousSize[0] + cx] !== INVALID_ID) {
                        expected = previous[cy * previousSize[0] + cx];
                        break;
                    }
                }
                if (current[y * size[0] + x] !== expected) {
                    mismatches++;
                    if (mismatches <= 5) fail(`mask level ${level} texel ${x}, ${y}: GPU ${current[y * size[0] + x]}, expected ${expected}`);
                }
            }
        }
        previous = current;
        previousSize = size;
    }
    return { levels: probeMaskMipCount, mismatches };
}

/** Draws one of the technique's debug views into a canvas on the page, for the runner's screenshot. */
async function showDebugView(device: GPUDevice, run: ImageRun, view: RadianceCacheDebugViewId, exposure: number, caption: string, column: HTMLElement): Promise<void> {
    const target = device.createTexture({ size: [WIDTH, HEIGHT], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    run.technique.encodeDebugView(encoder, view, target, exposure);
    device.queue.submit([encoder.finish()]);
    const pixels = await readTexture(device, target, 4, WIDTH, HEIGHT);
    target.destroy();
    const canvas = document.createElement('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = '400px';
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
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
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
    const data: Record<string, unknown> = { width: WIDTH, height: HEIGHT };

    // 1. Lambert: a constant sky of radiance L reads E = pi L.
    const lambert = await runFloor(device, 'lambert', { type: 'color', color: [CONSTANT_SKY, CONSTANT_SKY, CONSTANT_SKY], intensity: 1 });
    try {
        const readback = await readRun(device, lambert);
        const { ratios, nonFinite, negative, backupHints } = irradianceRatios(readback, Math.PI * CONSTANT_SKY);
        const errors = ratios.map((ratio) => Math.abs(ratio - 1));
        const meanError = errors.reduce((sum, value) => sum + value, 0) / Math.max(errors.length, 1);
        const p95Error = percentile(errors, 0.95);
        if (ratios.length < (WIDTH * HEIGHT) / 4) fail(`lambert: only ${ratios.length} floor pixels`);
        if (!(meanError <= LAMBERT_TOLERANCE)) fail(`lambert: mean |E / (pi L) - 1| is ${(meanError * 100).toFixed(3)}%`);
        if (!(p95Error <= LAMBERT_TOLERANCE)) fail(`lambert: 95th percentile |E / (pi L) - 1| is ${(p95Error * 100).toFixed(3)}%`);
        if (nonFinite > 0 || negative > 0) fail(`lambert: ${nonFinite} pixels not finite, ${negative} negative`);
        const mask = await checkMaskMips(device, lambert, fail);
        data.lambert = {
            skyRadiance: CONSTANT_SKY,
            floorPixels: ratios.length,
            probes: readback.seeds.length,
            meanRatio: ratios.reduce((sum, value) => sum + value, 0) / Math.max(ratios.length, 1),
            meanError,
            p95Error,
            maxError: largest(errors),
            minRatio: -largest(ratios.map((ratio) => -ratio)),
            maxRatio: largest(ratios),
            backupHints,
            nonFinite,
            negative,
            maskLevels: mask.levels,
            maskMismatches: mask.mismatches,
        };
    } finally {
        lambert.dispose();
        await device.queue.onSubmittedWorkDone();
    }

    // 2. A direction-varying sky: the SH per probe against the fine integral, and E at the pixels.
    const skyTexture = varyingSkyTexture(device);
    const varying = await runFloor(device, 'varying', { type: 'texture', texture: skyTexture, layout: 'equirect' });
    try {
        const readback = await readRun(device, varying);
        const reference = integrateHemisphereSH(varyingSky, UP, 256);
        const scale = Math.max(...reference.map(Math.abs));
        const sums = new Array<number>(SH_COEFFICIENTS).fill(0);
        const worstPerProbe: number[] = [];
        for (const seed of readback.seeds) {
            const coefficients = probeCoefficients(readback, seed);
            coefficients.forEach((value, index) => (sums[index] += value));
            worstPerProbe.push(Math.max(...coefficients.map((value, index) => Math.abs(value - reference[index]) / scale)));
        }
        const means = sums.map((sum) => sum / Math.max(readback.seeds.length, 1));
        const meanErrors = means.map((value, index) => Math.abs(value - reference[index]) / scale);
        const worstMean = Math.max(...meanErrors);
        const probeP95 = percentile(worstPerProbe, 0.95);
        if (!(worstMean <= SH_MEAN_TOLERANCE)) fail(`varying sky: the probes' mean SH is ${(worstMean * 100).toFixed(2)}% off the integral (coefficient ${meanErrors.indexOf(worstMean)})`);
        if (!(probeP95 <= SH_PROBE_P95_TOLERANCE)) fail(`varying sky: the 95th percentile of a probe's worst SH coefficient is ${(probeP95 * 100).toFixed(2)}% off`);
        // What a probe transposed in its tile would read (each cell's value from its transposed cell), on the CPU.
        const transposed = projectProbe((cell) => varyingSky(probeCellDirection((cell % PROBE_SIZE) * PROBE_SIZE + Math.floor(cell / PROBE_SIZE), UP)), UP);
        const transposedError = Math.max(...transposed.map((value, index) => Math.abs(value - reference[index]) / scale));

        const expectedE = shIrradiance(reference, UP);
        const { ratios, nonFinite, negative, backupHints } = irradianceRatios(readback, expectedE);
        const errors = ratios.map((ratio) => Math.abs(ratio - 1));
        const meanError = errors.reduce((sum, value) => sum + value, 0) / Math.max(errors.length, 1);
        const p95Error = percentile(errors, 0.95);
        if (!(meanError <= VARYING_E_TOLERANCE)) fail(`varying sky: mean |E / E_cpu - 1| is ${(meanError * 100).toFixed(2)}%`);
        if (!(p95Error <= VARYING_E_TOLERANCE)) fail(`varying sky: 95th percentile |E / E_cpu - 1| is ${(p95Error * 100).toFixed(2)}%`);
        if (nonFinite > 0 || negative > 0) fail(`varying sky: ${nonFinite} pixels not finite, ${negative} negative`);
        const mask = await checkMaskMips(device, varying, fail);
        data.varyingSky = {
            sky: 'L(d) = 1 + 0.6 d.x - 0.3 d.z',
            probes: readback.seeds.length,
            referenceSH: reference,
            meanSH: means,
            worstMeanCoefficientError: worstMean,
            probeWorstCoefficientP95: probeP95,
            transposedProbeErrorOnTheCpu: transposedError,
            expectedE,
            meanError,
            p95Error,
            backupHints,
            nonFinite,
            negative,
            maskMismatches: mask.mismatches,
        };
        // One column per view, side by side, so the runner's screenshot holds all three.
        const views: Array<[RadianceCacheDebugViewId, string]> = [
            [RADIANCE_CACHE_DEBUG_VIEW.probeAtlas, 'varying sky: probe atlas'],
            [RADIANCE_CACHE_DEBUG_VIEW.probeMask, 'varying sky: probe mask'],
            [RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics, 'varying sky: probe SH (E / pi along the normal)'],
        ];
        for (const [view, caption] of views) {
            const column = document.createElement('div');
            gallery.append(column);
            await showDebugView(device, varying, view, 1, caption, column);
        }
    } finally {
        varying.dispose();
        skyTexture.destroy();
        await device.queue.onSubmittedWorkDone();
    }

    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data,
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
