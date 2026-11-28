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
 * 3. **The interpolation's mix** (RC2.2 review): a 1 m step and a wall at
 *    the back under the varying sky, so neighbouring probes hold different
 *    SH (by normal, by the sky the step and wall hide, and by each probe's
 *    one random ray per cell). Every pixel's interpolated E against the CPU
 *    port of `InterpolateScreenProbes` (capsaicin-math.ts: the same jitter,
 *    neighbour search, plane, depth and normal weights, backup) run on the
 *    GPU's own depth, normals, mask mips and SH; the denoiser hint must agree
 *    too. The CPU's nearest-probe-only figure is reported, to show a wrong
 *    mix would fail.
 * 4. On all: the mask's mip chain holds, at every level, a probe of its
 *    2 × 2 children when any has one (`FilterProbeMask`); the irradiance has
 *    no negative, infinite or NaN value.
 *
 * The page draws the probe atlas, the mask and the SH view of the varying
 * sky (the technique's engine-free debug views) for the runner's screenshot.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc2.2 image" -- pnpm test:gpu --filter radiance-cache-image --gpu nvidia` (or intel).
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { equirectUvToDirection } from '../../../../packages/core/src/platform/sky/equirect.ts';
import { cameraRelativeInverseViewProjection } from '../../../../packages/cache/src/camera.ts';
import {
    INVALID_ID,
    integrateHemisphereSH,
    interpolateScreenProbes,
    projectProbe,
    probeCellDirection,
    shIrradiance,
    unpackSeed,
    type Direction,
    type InterpolationInputs,
} from '../../../../packages/cache/src/capsaicin-math.ts';
import { screenProbeCellSize, viewForward } from '../../../../packages/cache/src/screen-irradiance.ts';
import { RadianceCacheTechnique } from '../../../../packages/cache/src/radiance-cache-technique.ts';
import { PROBE_SIZE } from '../../../../packages/cache/src/settings.ts';
import { RADIANCE_CACHE_DEBUG_VIEW, type RadianceCacheDebugViewId } from '../../../../packages/cache/src/wgsl/debug.ts';
import { PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/cache/src/wgsl/screen-probes.ts';
import { SH_BLOCK_SIZE, SH_COEFFICIENTS } from '../../../../packages/cache/src/wgsl/spherical-harmonics.ts';
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
/** Section 3: a pixel's E against the CPU's, relative to the larger of the CPU's E and 1% of the mean. */
const MIX_TOLERANCE = 0.01;
/**
 * Section 3: pixels allowed past {@link MIX_TOLERANCE}. The GPU's f32
 * positions and the CPU's f64 can decide the jitter's plane test or a probe's
 * plane test differently right at the threshold.
 */
const MIX_OUTLIER_FRACTION = 0.005;
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

/** Quads as triangle lists: four corners each, in order around the quad, and the quad's normal. */
function quadsMesh(quads: ReadonlyArray<{ corners: readonly Direction[]; normal: Direction }>): { positions: Float32Array; normals: Float32Array; indices: Uint32Array } {
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    for (const quad of quads) {
        const base = positions.length / 3;
        for (const corner of quad.corners) {
            positions.push(...corner);
            normals.push(...quad.normal);
        }
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    return { positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices) };
}

/**
 * Section 3's step and corner: a box 1 m high on the floor in front of the
 * camera and a 6 m wall 25 m away, so probes differ by normal and by what
 * they see (the box and wall hide part of the sky), and the pixel plane,
 * depth and normal weights all decide something.
 */
const STEP_AND_WALL = quadsMesh([
    { corners: [[-2, 1, -6], [1, 1, -6], [1, 1, -9], [-2, 1, -9]], normal: [0, 1, 0] },
    { corners: [[-2, 0, -6], [1, 0, -6], [1, 1, -6], [-2, 1, -6]], normal: [0, 0, 1] },
    { corners: [[1, 0, -6], [1, 0, -9], [1, 1, -9], [1, 1, -6]], normal: [1, 0, 0] },
    { corners: [[-2, 0, -9], [-2, 0, -6], [-2, 1, -6], [-2, 1, -9]], normal: [-1, 0, 0] },
    { corners: [[1, 0, -9], [-2, 0, -9], [-2, 1, -9], [1, 1, -9]], normal: [0, 0, -1] },
    { corners: [[-200, 0, -25], [200, 0, -25], [200, 6, -25], [-200, 6, -25]], normal: [0, 0, 1] },
]);

/** A 400 m floor at y = 0 under `environment`, no lights (with `withStep`, also section 3's box and wall); frames until the trees are built, then one more. */
async function runFloor(device: GPUDevice, name: string, environment: Parameters<GIPlatformImpl['scene']['setEnvironment']>[0], withStep = false): Promise<ImageRun> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-image ${name}` });
    // The step and wall run with SSGI (opt-in, D61), so section 3 checks its HAS_OCCLUSION branch too.
    const technique = platform.addTechnique(new RadianceCacheTechnique({ ssgi: withStep }));
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
    if (withStep) {
        const stepMesh = scene.createMesh({ ...STEP_AND_WALL, label: 'step and wall' });
        scene.createInstance({ mesh: stepMesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'step and wall' });
    }
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

/**
 * Section 3: the GPU's interpolated E at every surface pixel against the CPU
 * port of `InterpolateScreenProbes` on the GPU's own inputs.
 */
async function checkInterpolation(device: GPUDevice, run: ImageRun, fail: (message: string) => void): Promise<Record<string, unknown>> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const { probeCount, probeMaskMipCount } = probes.layout;
    const readback = await readRun(device, run);
    const depth = new Float32Array((await readTexture(device, run.screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
    const normals = halves(await readTexture(device, run.screen.screen.normal, 8, WIDTH, HEIGHT));
    const levels: Array<{ width: number; texels: Uint32Array }> = [];
    for (let level = 0; level < probeMaskMipCount; level++) {
        const size = [Math.max(1, probeCount[0] >> level), Math.max(1, probeCount[1] >> level)];
        levels.push({ width: size[0], texels: new Uint32Array((await readTexture(device, probes.probeMask, 4, size[0], size[1], level)).buffer) });
    }
    const camera = frameCamera(CAMERA, WIDTH, HEIGHT);
    const inverse = cameraRelativeInverseViewProjection(camera);
    const eye = camera.position;
    const worldPosition = (x: number, y: number): Direction => {
        const ndc = [(2 * (x + 0.5)) / WIDTH - 1, 2 * (1 - (y + 0.5) / HEIGHT) - 1, depth[y * WIDTH + x], 1];
        const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row] * ndc[3]);
        return [eye[0] + h[0] / h[3], eye[1] + h[1] / h[3], eye[2] + h[2] / h[3]];
    };
    const normal = (x: number, y: number): Direction => {
        const index = (y * WIDTH + x) * 4;
        const length = Math.hypot(normals[index], normals[index + 1], normals[index + 2]);
        return [normals[index] / length, normals[index + 1] / length, normals[index + 2] / length];
    };
    const occlusionImage = probes.occlusion ? halves(await readTexture(device, probes.occlusion.occlusion, 8, WIDTH, HEIGHT)) : null;
    const nearFieldImage = probes.occlusion ? halves(await readTexture(device, probes.occlusion.nearField, 8, WIDTH, HEIGHT)) : null;
    const coefficientsByTile = new Map<number, number[]>();
    const inputs: InterpolationInputs = {
        width: WIDTH,
        height: HEIGHT,
        frameIndex: probes.frameIndex,
        cellSize: screenProbeCellSize(camera, WIDTH, HEIGHT),
        eye: [eye[0], eye[1], eye[2]],
        viewForward: viewForward(camera),
        probeMaskMipCount,
        isSky: (x, y) => readback.isSky(y * WIDTH + x),
        worldPosition,
        normal,
        probeMask: (level, x, y) => levels[level].texels[y * levels[level].width + x],
        probeIrradiance: (tileX, tileY, direction, ao) => {
            const key = tileY * probeCount[0] + tileX;
            let coefficients = coefficientsByTile.get(key);
            if (!coefficients) {
                coefficients = probeCoefficients(readback, ((tileX * PROBE_SIZE) << 16) | (tileY * PROBE_SIZE));
                coefficientsByTile.set(key, coefficients);
            }
            return shIrradiance(coefficients, direction, ao);
        },
        // SSGI's images as the GPU read them (RC4.2): the CPU port checks the HAS_OCCLUSION branch on them.
        occlusion: occlusionImage
            ? (x, y) => {
                  const index = (y * WIDTH + x) * 4;
                  const near = nearFieldImage ?? occlusionImage;
                  return {
                      ao: occlusionImage[index + 3],
                      bentNormal: [occlusionImage[index], occlusionImage[index + 1], occlusionImage[index + 2]],
                      nearField: nearFieldImage ? (near[index] + near[index + 1] + near[index + 2]) / 3 : 0,
                  };
              }
            : undefined,
    };
    let surfacePixels = 0;
    let gpuSum = 0;
    let hintMismatches = 0;
    let backupPixels = 0;
    let mixedPixels = 0;
    const cpuValues: Array<{ pixel: number; cpu: number; nearest: number }> = [];
    for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
            const pixel = y * WIDTH + x;
            const result = interpolateScreenProbes(inputs, x, y);
            if (!result) continue;
            surfacePixels++;
            const gpu = readback.irradiance.subarray(pixel * 4, pixel * 4 + 4);
            gpuSum += (gpu[0] + gpu[1] + gpu[2]) / 3;
            if (gpu[3] !== result.hint) hintMismatches++;
            if (result.hint === 0) backupPixels++;
            if (result.weights.filter((w) => w > 0).length > 1) mixedPixels++;
            const [sx, sy] = unpackSeed(result.probes[0]);
            const nearest = result.probes[0] === INVALID_ID ? 0 : inputs.probeIrradiance(Math.floor(sx / PROBE_SIZE), Math.floor(sy / PROBE_SIZE), normal(x, y), 1);
            cpuValues.push({ pixel, cpu: result.irradiance, nearest });
        }
    }
    const meanE = gpuSum / Math.max(surfacePixels, 1);
    const errors: number[] = [];
    const nearestOnly: number[] = [];
    for (const { pixel, cpu, nearest } of cpuValues) {
        const gpu = (readback.irradiance[pixel * 4] + readback.irradiance[pixel * 4 + 1] + readback.irradiance[pixel * 4 + 2]) / 3;
        const scale = Math.max(cpu, 0.01 * meanE);
        errors.push(Math.abs(gpu - cpu) / scale);
        nearestOnly.push(Math.abs(nearest - cpu) / scale);
    }
    const outliers = errors.filter((error) => error > MIX_TOLERANCE).length;
    const outlierFraction = outliers / Math.max(surfacePixels, 1);
    const nearestOnlyFraction = nearestOnly.filter((error) => error > MIX_TOLERANCE).length / Math.max(surfacePixels, 1);
    if (surfacePixels < (WIDTH * HEIGHT) / 2) fail(`interpolation: only ${surfacePixels} surface pixels`);
    if (!(outlierFraction <= MIX_OUTLIER_FRACTION)) {
        fail(`interpolation: ${outliers} of ${surfacePixels} pixels (${(outlierFraction * 100).toFixed(2)}%) differ from the CPU's InterpolateScreenProbes by more than ${MIX_TOLERANCE * 100}%`);
    }
    if (!(hintMismatches <= MIX_OUTLIER_FRACTION * surfacePixels)) fail(`interpolation: the denoiser hint differs from the CPU's on ${hintMismatches} pixels`);
    // The check must be able to fail: the nearest probe alone has to miss on many pixels.
    if (!(nearestOnlyFraction >= 20 * MIX_OUTLIER_FRACTION)) {
        fail(`interpolation: the scene does not test the mix (the nearest probe alone is off on only ${(nearestOnlyFraction * 100).toFixed(2)}% of pixels)`);
    }
    return {
        surfacePixels,
        meanE,
        meanError: errors.reduce((sum, value) => sum + value, 0) / Math.max(errors.length, 1),
        p99Error: percentile(errors, 0.99),
        outliers,
        hintMismatches,
        backupPixels,
        mixedPixels,
        nearestOnlyOffFraction: nearestOnlyFraction,
    };
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

    // 3. The interpolation's mix on a step and a wall: every pixel against the CPU port.
    const stepSky = varyingSkyTexture(device);
    const step = await runFloor(device, 'step', { type: 'texture', texture: stepSky, layout: 'equirect' }, true);
    try {
        data.interpolation = await checkInterpolation(device, step, fail);
    } finally {
        step.dispose();
        stepSky.destroy();
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
