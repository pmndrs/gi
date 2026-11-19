/**
 * RC4.1: the radiance cache's screen probes with history on the GPU
 * (reprojection, the off-screen probe cache, the patch, ray guiding, the
 * blend), checked against analytic answers and CPU ports
 * (packages/ddgi-core/src/techniques/radiance-cache/screen-probe-history-cpu.ts).
 * Engine-free: the `FrameScreen` (depth, normals, velocity) is drawn by
 * primary rays through the platform's tracer (radiance-cache-probes/screen.ts).
 *
 * 1. **The constant sky stays at πL** (register D41): a 400 m floor under a
 *    constant sky of radiance L, 1280 × 720, history on. Every frame from 1
 *    to `STILL_FRAMES`, then `MOVING_FRAMES` with the camera moving, the
 *    floor's interpolated irradiance is E = πL within 1% (mean and 95th
 *    percentile of |E / πL − 1| per pixel). Native GPU-1.2 reads 1.29 at
 *    frame 1 and 0.87 at steady state; with Capsaicin's empty-cell rule this
 *    page reads 0.68 by frame 30 and fails. The share of probe cells no ray
 *    reached is reported (about 40% with history, from sampling with
 *    replacement).
 * 2. **The sums can't wrap** (register D40; the RC4.1 done-when from RC2.1's
 *    review): a sky of 60,000 above 65° elevation and black below, so ray
 *    guiding piles a probe's rays into the few bright cells. Every cell's hit
 *    distance reads ≥ 60,000 (every ray misses), every cell wholly inside the
 *    cap reads 60,000 and every cell wholly outside reads 0. A 32-bit sum
 *    wraps at 7 misses (or 7 bright rays) in a cell; the page checks that
 *    probes do get 7 or more rays in a cell. On the same frame, **ray
 *    guiding**: the rays per cell, pooled over every probe, against what each
 *    probe's CDF of previous luminance × cos θ (and the uniform share, D42)
 *    predicts, by chi-square at the 0.1% level.
 * 3. **`ReprojectScreenProbes` against the CPU** on a step and a wall under a
 *    direction-varying sky. First still: every probe tile the next frame did
 *    not spawn into keeps its seed and cells. Then the camera moving each
 *    frame: every probe tile the
 *    frame did not spawn into holds the seed and the 64 cells the CPU port
 *    computes from the GPU's own last-frame and this-frame buffers; the empty
 *    tiles are the CPU's. Before that, the screen's velocity is checked: the
 *    world point a pixel shows now is the one its previous pixel showed.
 * 4. **The off-screen cache and the patch** on the same scene, the camera
 *    sweeping past the step (parallax: probes evicted and found again): after
 *    every frame the LRU list is a permutation of the slots, the entries used
 *    this frame are exactly those with an MRU number, numbered 0..n−1; the
 *    probe list is the spawn's (CPU rule) except probes moved by the patch,
 *    each moved from the override list into an empty tile at the CPU's seed.
 *    The scene must evict, list and patch something.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc4.1 temporal" -- pnpm test:gpu --filter radiance-cache-temporal --gpu nvidia` (or intel).
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { equirectUvToDirection } from '../../../../packages/ddgi-core/src/platform/sky/equirect.ts';
import { cameraRelativeInverseViewProjection } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/camera.ts';
import {
    calculateHaltonSequence,
    floatToHalfBits,
    halfBitsToFloat,
    INVALID_ID,
    mapToHemiOctahedron,
    packSeed,
    probeCellOf,
    spawnScreenProbes,
    unpackHalf3,
    unpackSeed,
    type Direction,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { screenProbeCellSize } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/screen-irradiance.ts';
import { CACHED_TILE_NEW, luminance, RAY_GUIDING_UNIFORM_SHARE, reprojectProbeTile, type ReprojectionInputs } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/screen-probe-history-cpu.ts';
import { PROBE_CELLS, PROBE_SIZE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { COUNTER, PROBE_RAY_BYTES, PROBE_RAYS_HEADER_BYTES, PROBE_SPAWN_PROBE_BYTES, PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-probes.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { readBuffer } from '../ddgi-shared/harness.ts';
import { readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
/** Section 1: frames with the camera still, then moving; |E / πL − 1| allowed per frame (mean and p95). */
const STILL_FRAMES = 320;
const MOVING_FRAMES = 120;
const LAMBERT_TOLERANCE = 0.01;
const CONSTANT_SKY = 1.5;
/** Section 2: the bright cap's radiance (under the half range's 65,504) and its lower edge's elevation. */
const CAP_RADIANCE = 60000;
const CAP_ELEVATION_DEG = 65;
const CAP_FRAMES = 16;
/** Section 2: a cell counts as wholly inside or outside the cap when all its points are this far from the edge. */
const CAP_MARGIN_DEG = 3;
/** Section 3: frames of motion before the checked frame, and the share of tiles allowed to disagree with the CPU (f32 against f64 at a test's threshold). */
const REPROJECTION_WARM_FRAMES = 6;
const REPROJECTION_OUTLIER_FRACTION = 0.005;
/** Section 4: frames of the sweep, each checked. */
const CACHE_FRAMES = 40;
const SKY_IMAGE = { width: 512, height: 256 } as const;
const varyingSky = (direction: Direction): number => 1 + 0.6 * direction[0] - 0.3 * direction[2];

type Vector = [number, number, number];

/** Half floats by table: a screen's worth per frame, many frames. */
const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));
function halvesFast(bytes: Uint8Array): Float32Array {
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const out = new Float32Array(words.length);
    for (let index = 0; index < words.length; index++) out[index] = HALF_TABLE[words[index]];
    return out;
}

function percentile(values: Float64Array | number[], fraction: number): number {
    if (values.length === 0) return NaN;
    const sorted = Float64Array.from(values).sort();
    return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

/** Quads as triangle lists: four corners each, in order around the quad, and the quad's normal. */
function quadsMesh(quads: ReadonlyArray<{ corners: readonly Vector[]; normal: Vector }>): { positions: Float32Array; normals: Float32Array; indices: Uint32Array } {
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

/** A 1 m step and a 6 m wall 25 m away (radiance-cache-image's section 3). */
const STEP_AND_WALL = quadsMesh([
    { corners: [[-2, 1, -6], [1, 1, -6], [1, 1, -9], [-2, 1, -9]], normal: [0, 1, 0] },
    { corners: [[-2, 0, -6], [1, 0, -6], [1, 1, -6], [-2, 1, -6]], normal: [0, 0, 1] },
    { corners: [[1, 0, -6], [1, 0, -9], [1, 1, -9], [1, 1, -6]], normal: [1, 0, 0] },
    { corners: [[-2, 0, -9], [-2, 0, -6], [-2, 1, -6], [-2, 1, -9]], normal: [-1, 0, 0] },
    { corners: [[1, 0, -9], [-2, 0, -9], [-2, 1, -9], [1, 1, -9]], normal: [0, 0, -1] },
    { corners: [[-200, 0, -25], [200, 0, -25], [200, 6, -25], [-200, 6, -25]], normal: [0, 0, 1] },
]);

/** An equirect sky texture from a radiance function, in the platform's GPU order (memory row 0 is the nadir). */
function skyTexture(device: GPUDevice, radiance: (direction: Direction) => number): GPUTexture {
    const texture = device.createTexture({ label: 'sky', size: [SKY_IMAGE.width, SKY_IMAGE.height], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const texels = new Float32Array(SKY_IMAGE.width * SKY_IMAGE.height * 4);
    for (let row = 0; row < SKY_IMAGE.height; row++) {
        for (let column = 0; column < SKY_IMAGE.width; column++) {
            const imageRow = SKY_IMAGE.height - 1 - row;
            const value = radiance(equirectUvToDirection({ u: (column + 0.5) / SKY_IMAGE.width, v: (imageRow + 0.5) / SKY_IMAGE.height }));
            texels.set([value, value, value, 1], (row * SKY_IMAGE.width + column) * 4);
        }
    }
    device.queue.writeTexture({ texture }, texels, { bytesPerRow: SKY_IMAGE.width * 16 }, [SKY_IMAGE.width, SKY_IMAGE.height]);
    return texture;
}

interface Run {
    readonly technique: RadianceCacheTechnique;
    readonly platform: GIPlatformImpl;
    readonly screen: PageScreen;
    camera: PageCamera;
    /** One frame at `camera`: the host's encode, the page's screen, the screen stage. */
    frame(): void;
    dispose(): void;
}

/** A floor (and with `withStep`, the step and wall) under `environment`; frames until the trees are built. */
async function startRun(device: GPUDevice, name: string, environment: Parameters<GIPlatformImpl['scene']['setEnvironment']>[0], camera: PageCamera, withStep: boolean): Promise<Run> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-temporal ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    const scene = platform.scene;
    const half = 200;
    const material = scene.createMaterial({ albedo: [0.5, 0.5, 0.5], label: 'grey' });
    const floor = scene.createMesh({
        positions: new Float32Array([-half, 0, -half, half, 0, -half, half, 0, half, -half, 0, half]),
        normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
        indices: new Uint32Array([0, 2, 1, 0, 3, 2]),
        label: 'floor',
    });
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    scene.createInstance({ mesh: floor, materials: [material], transform: identity, label: 'floor' });
    if (withStep) scene.createInstance({ mesh: scene.createMesh({ ...STEP_AND_WALL, label: 'step and wall' }), materials: [material], transform: identity, label: 'step and wall' });
    scene.setEnvironment(environment);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const run: Run = {
        technique,
        platform,
        screen,
        camera,
        frame() {
            const encoder = device.createCommandEncoder({ label: `radiance-cache-temporal ${name}` });
            const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(run.camera, WIDTH, HEIGHT) };
            platform.encode(encoder, input);
            screen.encode(encoder, run.camera);
            platform.encodeScreen(encoder, screen.screen);
            device.queue.submit([encoder.finish()]);
        },
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
    // A frame places the scene; once the trees are built, history starts clean.
    run.frame();
    await platform.rays.whenBuilt();
    await device.queue.onSubmittedWorkDone();
    technique.resetHistory();
    return run;
}

/** The camera after `step` frames of a slow move: along x, and turning a little. */
function movedCamera(base: PageCamera, step: number, metresPerFrame: number, degreesPerFrame: number): PageCamera {
    const angle = (step * degreesPerFrame * Math.PI) / 180;
    const position: Vector = [base.position[0] + step * metresPerFrame, base.position[1], base.position[2] + step * metresPerFrame * 0.3];
    const offset = [base.target[0] - base.position[0], base.target[1] - base.position[1], base.target[2] - base.position[2]];
    const target: Vector = [position[0] + offset[0] * Math.cos(angle) + offset[2] * Math.sin(angle), position[1] + offset[1], position[2] - offset[0] * Math.sin(angle) + offset[2] * Math.cos(angle)];
    return { ...base, position, target };
}

interface ScreenReadback {
    readonly depth: Float32Array;
    readonly normal: Float32Array;
    readonly velocity: Float32Array;
}

async function readScreen(device: GPUDevice, run: Run): Promise<ScreenReadback> {
    const screen = run.screen.screen;
    const velocityTexture = screen.velocity;
    if (!velocityTexture) throw new Error('the page screen has no velocity');
    return {
        depth: new Float32Array((await readTexture(device, screen.depth, 4, WIDTH, HEIGHT)).buffer),
        normal: halvesFast(await readTexture(device, screen.normal, 8, WIDTH, HEIGHT)),
        velocity: halvesFast(await readTexture(device, velocityTexture, 8, WIDTH, HEIGHT)),
    };
}

/** World position and unit normal of a pixel from a screen readback and the camera it was drawn with. */
function screenGeometry(readback: ScreenReadback, camera: PageCamera): { world: (x: number, y: number) => Direction; normal: (x: number, y: number) => Direction; isSky: (x: number, y: number) => boolean } {
    const frame = frameCamera(camera, WIDTH, HEIGHT);
    const inverse = cameraRelativeInverseViewProjection(frame);
    const eye = frame.position;
    return {
        world: (x, y) => {
            const ndc = [(2 * (x + 0.5)) / WIDTH - 1, 2 * (1 - (y + 0.5) / HEIGHT) - 1, readback.depth[y * WIDTH + x], 1];
            const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row] * ndc[3]);
            return [eye[0] + h[0] / h[3], eye[1] + h[1] / h[3], eye[2] + h[2] / h[3]];
        },
        normal: (x, y) => {
            const index = (y * WIDTH + x) * 4;
            const length = Math.hypot(readback.normal[index], readback.normal[index + 1], readback.normal[index + 2]);
            return [readback.normal[index] / length, readback.normal[index + 1] / length, readback.normal[index + 2] / length];
        },
        isSky: (x, y) => readback.depth[y * WIDTH + x] === 1,
    };
}

/** The floor pixels' E / (πL): the mean, the 95th percentile of |ratio − 1|, and bad values. */
async function floorRatios(device: GPUDevice, run: Run, isSky: (pixel: number) => boolean, expected: number): Promise<{ mean: number; p95: number; worst: number; bad: number }> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const irradiance = halvesFast(await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT));
    const errors: number[] = [];
    let sum = 0;
    let bad = 0;
    for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
        if (isSky(pixel)) continue;
        const e = (irradiance[pixel * 4] + irradiance[pixel * 4 + 1] + irradiance[pixel * 4 + 2]) / 3;
        if (!Number.isFinite(e) || e < 0) {
            bad++;
            continue;
        }
        const ratio = e / expected;
        sum += ratio;
        errors.push(Math.abs(ratio - 1));
    }
    let worst = 0;
    for (const error of errors) worst = Math.max(worst, error);
    return { mean: sum / Math.max(errors.length, 1), p95: percentile(errors, 0.95), worst, bad };
}

/** The probes of the latest frame: their seeds and each ray's cell, counted per probe. */
async function rayCells(device: GPUDevice, run: Run, normalAt: (x: number, y: number) => Direction): Promise<{ seeds: number[]; countsPerProbe: number[][] }> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const spawnWords = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + probes.layout.maxProbeSpawnCount * 4));
    const seeds = Array.from(spawnWords.subarray(1, 1 + spawnWords[0]));
    const rayWords = new Uint32Array(await readBuffer(device, probes.probeRays, PROBE_RAYS_HEADER_BYTES + Math.max(seeds.length, 1) * PROBE_CELLS * PROBE_RAY_BYTES));
    const countsPerProbe = seeds.map((seed, probe) => {
        const [x, y] = unpackSeed(seed);
        const normal = normalAt(x, y);
        const counts = new Array<number>(PROBE_CELLS).fill(0);
        for (let ray = 0; ray < PROBE_CELLS; ray++) {
            const base = (PROBE_RAYS_HEADER_BYTES + (probe * PROBE_CELLS + ray) * PROBE_RAY_BYTES) / 4;
            counts[probeCellOf(unpackHalf3([rayWords[base + 4], rayWords[base + 5]]), normal)]++;
        }
        return counts;
    });
    return { seeds, countsPerProbe };
}

// ---- 1. The constant sky ----------------------------------------------------------------------

async function checkConstantSky(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, unknown>> {
    const base: PageCamera = { position: [0, 3, 0], target: [0, 1, -10], fovYDeg: 60, near: 0.1, far: 1000 };
    const run = await startRun(device, 'constant sky', { type: 'color', color: [CONSTANT_SKY, CONSTANT_SKY, CONSTANT_SKY], intensity: 1 }, base, false);
    try {
        const expected = Math.PI * CONSTANT_SKY;
        const up: Direction = [0, 1, 0];
        const perFrame: Array<{ frame: number; moving: boolean; mean: number; p95: number }> = [];
        const unreached: Array<{ frame: number; share: number; mostRaysInACell: number }> = [];
        let worstMean = 0;
        let worstP95 = 0;
        let bad = 0;
        const reportFrames = new Set([1, 2, 3, 4, 8, 16, 32, 48, 64, 96, 128, 200, 300, STILL_FRAMES, STILL_FRAMES + 1, STILL_FRAMES + 30, STILL_FRAMES + MOVING_FRAMES]);
        const coverageFrames = new Set([1, 2, 10, 100, STILL_FRAMES]);
        for (let frame = 1; frame <= STILL_FRAMES + MOVING_FRAMES; frame++) {
            const moving = frame > STILL_FRAMES;
            if (moving) run.camera = movedCamera(base, frame - STILL_FRAMES, 0.02, 0.15);
            run.frame();
            const depth = new Float32Array((await readTexture(device, run.screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
            const ratios = await floorRatios(device, run, (pixel) => depth[pixel] === 1, expected);
            bad += ratios.bad;
            worstMean = Math.max(worstMean, Math.abs(ratios.mean - 1));
            worstP95 = Math.max(worstP95, ratios.p95);
            if (!(Math.abs(ratios.mean - 1) <= LAMBERT_TOLERANCE)) fail(`constant sky, frame ${frame}${moving ? ' (moving)' : ''}: mean E / (pi L) is ${ratios.mean.toFixed(4)}`);
            if (!(ratios.p95 <= LAMBERT_TOLERANCE)) fail(`constant sky, frame ${frame}${moving ? ' (moving)' : ''}: 95th percentile |E / (pi L) - 1| is ${(ratios.p95 * 100).toFixed(3)}%`);
            if (reportFrames.has(frame)) perFrame.push({ frame, moving, mean: Number(ratios.mean.toFixed(5)), p95: Number(ratios.p95.toFixed(5)) });
            if (coverageFrames.has(frame)) {
                const cells = await rayCells(device, run, () => up);
                let empty = 0;
                let most = 0;
                for (const counts of cells.countsPerProbe) {
                    empty += counts.filter((count) => count === 0).length;
                    most = Math.max(most, ...counts);
                }
                unreached.push({ frame, share: Number((empty / Math.max(cells.seeds.length * PROBE_CELLS, 1)).toFixed(4)), mostRaysInACell: most });
            }
        }
        if (bad > 0) fail(`constant sky: ${bad} floor pixels not finite or negative`);
        return { skyRadiance: CONSTANT_SKY, stillFrames: STILL_FRAMES, movingFrames: MOVING_FRAMES, worstMeanError: worstMean, worstP95Error: worstP95, perFrame, unreachedCells: unreached };
    } finally {
        run.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- 2. Sums that can't wrap, and ray guiding -------------------------------------------------

/**
 * The chi-square of the rays per cell, pooled over this frame's probes, against
 * the CPU's guiding rule (`guidedCell`) on each probe's previous radiance: a ray
 * samples its own cell with the uniform share, or when its cell has no history
 * or the CDF is empty, and otherwise a cell in proportion to luminance × cos θ.
 */
async function checkGuiding(device: GPUDevice, run: Run, cells: { seeds: number[]; countsPerProbe: number[][] }, fail: (message: string) => void): Promise<Record<string, unknown>> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const [atlasWidth, atlasHeight] = probes.layout.atlasSize;
    const previous = halvesFast(await readTexture(device, probes.history.previousProbeAtlas, 8, atlasWidth, atlasHeight));
    const cosine = Array.from({ length: PROBE_CELLS }, (_, cell) => mapToHemiOctahedron(((cell % PROBE_SIZE) + 0.5) / PROBE_SIZE, (Math.floor(cell / PROBE_SIZE) + 0.5) / PROBE_SIZE)[2]);
    const expected = new Float64Array(PROBE_CELLS);
    const observed = new Float64Array(PROBE_CELLS);
    cells.seeds.forEach((seed, probe) => {
        const [x, y] = unpackSeed(seed);
        const tileX = Math.floor(x / PROBE_SIZE) * PROBE_SIZE;
        const tileY = Math.floor(y / PROBE_SIZE) * PROBE_SIZE;
        const weights: number[] = [];
        const history: boolean[] = [];
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const texel = ((tileY + Math.floor(cell / PROBE_SIZE)) * atlasWidth + tileX + (cell % PROBE_SIZE)) * 4;
            history.push(previous[texel + 3] > 0);
            weights.push(luminance([previous[texel], previous[texel + 1], previous[texel + 2]]) * cosine[cell]);
        }
        const total = weights.reduce((sum, weight) => sum + weight, 0);
        for (let thread = 0; thread < PROBE_CELLS; thread++) {
            if (!history[thread] || !(total > 0)) {
                expected[thread] += 1;
                continue;
            }
            expected[thread] += RAY_GUIDING_UNIFORM_SHARE;
            for (let cell = 0; cell < PROBE_CELLS; cell++) expected[cell] += ((1 - RAY_GUIDING_UNIFORM_SHARE) * weights[cell]) / total;
        }
        cells.countsPerProbe[probe].forEach((count, cell) => (observed[cell] += count));
    });
    let chiSquare = 0;
    let bins = 0;
    for (let cell = 0; cell < PROBE_CELLS; cell++) {
        if (expected[cell] < 5) continue;
        chiSquare += (observed[cell] - expected[cell]) ** 2 / expected[cell];
        bins++;
    }
    // The 0.1% point of chi-square with bins - 1 degrees of freedom, by the normal approximation.
    const dof = Math.max(bins - 1, 1);
    const limit = dof + 3.09 * Math.sqrt(2 * dof);
    if (!(chiSquare <= limit)) fail(`guiding: rays per cell against the CDF give chi-square ${chiSquare.toFixed(1)} over ${dof} degrees of freedom (limit ${limit.toFixed(1)})`);
    const brightest = Array.from(expected).indexOf(Math.max(...expected));
    return { chiSquare: Number(chiSquare.toFixed(2)), degreesOfFreedom: dof, limit: Number(limit.toFixed(1)), brightestCell: brightest, expectedThere: Number(expected[brightest].toFixed(1)), observedThere: observed[brightest] };
}

async function checkOverflow(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, unknown>> {
    const capSine = Math.sin((CAP_ELEVATION_DEG * Math.PI) / 180);
    const sky = skyTexture(device, (direction) => (direction[1] > capSine ? CAP_RADIANCE : 0));
    const camera: PageCamera = { position: [0, 3, 0], target: [0, 1, -10], fovYDeg: 60, near: 0.1, far: 1000 };
    const run = await startRun(device, 'bright cap', { type: 'texture', texture: sky, layout: 'equirect' }, camera, false);
    try {
        for (let frame = 0; frame < CAP_FRAMES; frame++) run.frame();
        await device.queue.onSubmittedWorkDone();
        const probes = run.technique.probes;
        if (!probes) throw new Error('no probes after encodeScreen');
        const up: Direction = [0, 1, 0];
        const cells = await rayCells(device, run, () => up);
        const mostPerProbe = cells.countsPerProbe.map((counts) => Math.max(...counts));
        const probesPastWrap = mostPerProbe.filter((most) => most >= 7).length;
        if (!(probesPastWrap >= cells.seeds.length / 2)) fail(`overflow: only ${probesPastWrap} of ${cells.seeds.length} probes got 7 or more rays in one cell, so the check does not reach a 32-bit wrap`);

        // Ray guiding (gi1.comp:901-978 with D42): the rays per cell, pooled over the probes, against what each
        // probe's CDF of previous luminance x cos theta predicts (the previous radiance SampleScreenProbes wrote).
        const guiding = await checkGuiding(device, run, cells, fail);

        // Each cell wholly inside or outside the cap, by 8 x 8 points of the cell (the probe frame is the floor's).
        const margin = (CAP_MARGIN_DEG * Math.PI) / 180;
        const capElevation = (CAP_ELEVATION_DEG * Math.PI) / 180;
        const placeOf = new Array<'inside' | 'outside' | 'edge'>(PROBE_CELLS);
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            let lowest = Infinity;
            let highest = -Infinity;
            for (let i = 0; i <= 8; i++) {
                for (let j = 0; j <= 8; j++) {
                    // Probe frame z is the floor's normal (up): elevation is asin(z).
                    const z = mapToHemiOctahedron(((cell % PROBE_SIZE) + i / 8) / PROBE_SIZE, (Math.floor(cell / PROBE_SIZE) + j / 8) / PROBE_SIZE)[2];
                    const elevation = Math.asin(Math.min(Math.max(z, -1), 1));
                    lowest = Math.min(lowest, elevation);
                    highest = Math.max(highest, elevation);
                }
            }
            placeOf[cell] = lowest > capElevation + margin ? 'inside' : highest < capElevation - margin ? 'outside' : 'edge';
        }
        const [atlasWidth, atlasHeight] = probes.layout.atlasSize;
        const atlas = halvesFast(await readTexture(device, probes.probeAtlas, 8, atlasWidth, atlasHeight));
        const [maskWidth, maskHeight] = probes.layout.probeCount;
        const mask = new Uint32Array((await readTexture(device, probes.probeMask, 4, maskWidth, maskHeight)).buffer);
        // The rays themselves: the sky they read on either side of the cap's edge (the platform's bake of the step).
        const results = probes.rayScratch.rayResults;
        const resultBytes = new Uint8Array(await readBuffer(device, results.buffer, (results.offset ?? 0) + cells.seeds.length * PROBE_CELLS * 8));
        const rayRadiance = halvesFast(resultBytes.subarray(results.offset ?? 0));
        const rayWords = new Uint32Array(await readBuffer(device, probes.probeRays, PROBE_RAYS_HEADER_BYTES + cells.seeds.length * PROBE_CELLS * PROBE_RAY_BYTES));
        let brightestRayOutside = 0;
        let brightestRayOutsideElevation = 0;
        for (let ray = 0; ray < cells.seeds.length * PROBE_CELLS; ray++) {
            const base = (PROBE_RAYS_HEADER_BYTES + ray * PROBE_RAY_BYTES) / 4;
            const elevation = Math.asin(Math.min(Math.max(unpackHalf3([rayWords[base + 4], rayWords[base + 5]])[1], -1), 1));
            const value = rayRadiance[ray * 4];
            if (elevation < capElevation - margin && value > brightestRayOutside) {
                brightestRayOutside = value;
                brightestRayOutsideElevation = (elevation * 180) / Math.PI;
            }
        }

        // Rays this frame per cell, by probe tile (the spawned probes).
        const raysByTile = new Map<number, number[]>();
        cells.seeds.forEach((seed, probe) => {
            const [x, y] = unpackSeed(seed);
            raysByTile.set(Math.floor(y / PROBE_SIZE) * maskWidth + Math.floor(x / PROBE_SIZE), cells.countsPerProbe[probe]);
        });
        let tiles = 0;
        let nearHitDistance = 0;
        let worstInside = 0;
        let worstOutside = 0;
        let insideCells = 0;
        let outsideCells = 0;
        // A dark cell no ray reached keeps what it held, so the frame-1 backup (the hemisphere's mean) can
        // linger in it, spread by the 3 x 3 taps' average: counted, not failed.
        let staleOutsideCells = 0;
        for (let tile = 0; tile < mask.length; tile++) {
            if (mask[tile] === INVALID_ID) continue;
            tiles++;
            const tileX = (tile % maskWidth) * PROBE_SIZE;
            const tileY = Math.floor(tile / maskWidth) * PROBE_SIZE;
            const rays = raysByTile.get(tile);
            for (let cell = 0; cell < PROBE_CELLS; cell++) {
                const texel = ((tileY + Math.floor(cell / PROBE_SIZE)) * atlasWidth + tileX + (cell % PROBE_SIZE)) * 4;
                const radiance = (atlas[texel] + atlas[texel + 1] + atlas[texel + 2]) / 3;
                if (!(atlas[texel + 3] >= 60000)) nearHitDistance++;
                if (placeOf[cell] === 'inside') {
                    insideCells++;
                    worstInside = Math.max(worstInside, Math.abs(radiance - CAP_RADIANCE) / CAP_RADIANCE);
                } else if (placeOf[cell] === 'outside') {
                    if (rays && rays[cell] > 0) {
                        outsideCells++;
                        worstOutside = Math.max(worstOutside, Math.abs(radiance) / CAP_RADIANCE);
                    } else if (Math.abs(radiance) > 0.01 * CAP_RADIANCE) {
                        staleOutsideCells++;
                    }
                }
            }
        }
        if (nearHitDistance > 0) fail(`overflow: ${nearHitDistance} cells read a hit distance under 60,000 though every ray misses`);
        if (!(worstInside <= 0.01)) fail(`overflow: a cell inside the cap reads ${(worstInside * 100).toFixed(2)}% off ${CAP_RADIANCE}`);
        if (!(worstOutside <= 1e-3)) fail(`overflow: a cell outside the cap that rays reached reads ${(worstOutside * CAP_RADIANCE).toFixed(1)}`);
        if (brightestRayOutside > 0) fail(`overflow: a ray below the cap read ${brightestRayOutside} at ${brightestRayOutsideElevation.toFixed(1)} degrees: the sky is not the step the check assumes`);
        if (insideCells === 0 || outsideCells === 0) fail(`overflow: ${insideCells} cells inside and ${outsideCells} reached cells outside the cap: the check needs both`);
        return {
            capRadiance: CAP_RADIANCE,
            capElevationDeg: CAP_ELEVATION_DEG,
            guiding,
            probes: cells.seeds.length,
            probesWithSevenOrMoreRaysInACell: probesPastWrap,
            mostRaysInACell: Math.max(...mostPerProbe),
            tilesChecked: tiles,
            cellsInside: insideCells,
            cellsOutside: outsideCells,
            worstInsideError: worstInside,
            worstOutsideValue: worstOutside * CAP_RADIANCE,
            staleOutsideCellsOver1Percent: staleOutsideCells,
            brightestRayOutside,
            brightestRayOutsideElevation,
            cellsWithANearHitDistance: nearHitDistance,
        };
    } finally {
        run.dispose();
        sky.destroy();
        await device.queue.onSubmittedWorkDone();
    }
}

/**
 * A still camera reprojects every probe to itself: after a few still frames,
 * every probe tile the next frame did not spawn into holds last frame's seed
 * and cells (within the 1e4 quantisation and half precision).
 */
async function checkStillReprojection(device: GPUDevice, run: Run, fail: (message: string) => void): Promise<Record<string, unknown>> {
    for (let frame = 0; frame < 4; frame++) run.frame();
    await device.queue.onSubmittedWorkDone();
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const [maskWidth, maskHeight] = probes.layout.probeCount;
    const [atlasWidth, atlasHeight] = probes.layout.atlasSize;
    const before = new Uint32Array((await readTexture(device, probes.probeMask, 4, maskWidth, maskHeight)).buffer);
    const beforeAtlas = halvesFast(await readTexture(device, probes.probeAtlas, 8, atlasWidth, atlasHeight));
    run.frame();
    await device.queue.onSubmittedWorkDone();
    const after = new Uint32Array((await readTexture(device, probes.probeMask, 4, maskWidth, maskHeight)).buffer);
    const afterAtlas = halvesFast(await readTexture(device, probes.probeAtlas, 8, atlasWidth, atlasHeight));
    const spawnWords = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + probes.layout.maxProbeSpawnCount * 4));
    const spawned = new Set(Array.from(spawnWords.subarray(1, 1 + spawnWords[0]), (seed) => {
        const [x, y] = unpackSeed(seed);
        return Math.floor(y / PROBE_SIZE) * maskWidth + Math.floor(x / PROBE_SIZE);
    }));
    let kept = 0;
    let moved = 0;
    let cellsChanged = 0;
    for (let tile = 0; tile < before.length; tile++) {
        if (before[tile] === INVALID_ID || spawned.has(tile)) continue;
        if (after[tile] !== before[tile]) {
            moved++;
            continue;
        }
        kept++;
        const tileX = (tile % maskWidth) * PROBE_SIZE;
        const tileY = Math.floor(tile / maskWidth) * PROBE_SIZE;
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const texel = ((tileY + Math.floor(cell / PROBE_SIZE)) * atlasWidth + tileX + (cell % PROBE_SIZE)) * 4;
            for (let channel = 0; channel < 4; channel++) {
                const a = beforeAtlas[texel + channel];
                const b = afterAtlas[texel + channel];
                if (Math.abs(a - b) > 1e-3 * Math.abs(a) + 1e-4) {
                    cellsChanged++;
                    break;
                }
            }
        }
    }
    if (moved > 0) fail(`still camera: ${moved} probes moved to another pixel`);
    if (cellsChanged > 0) fail(`still camera: ${cellsChanged} cells of kept probes changed`);
    if (kept < 1000) fail(`still camera: only ${kept} probes kept, too few to check`);
    return { kept, moved, cellsChanged };
}

// ---- 3. ReprojectScreenProbes against the CPU ------------------------------------------------------

async function checkReprojection(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, unknown>> {
    const sky = skyTexture(device, varyingSky);
    const base: PageCamera = { position: [-1, 2.5, 1], target: [0, 0.5, -8], fovYDeg: 60, near: 0.1, far: 1000 };
    const run = await startRun(device, 'reprojection', { type: 'texture', texture: sky, layout: 'equirect' }, base, true);
    try {
        // First the camera still: every probe tile the frame did not spawn into keeps its probe, seed and cells.
        const still = await checkStillReprojection(device, run, fail);
        // Then moving, fast enough that the screen's edge brings in whole new probe tiles (empty tiles) each frame.
        let step = 0;
        const advance = (): void => {
            step++;
            run.camera = movedCamera(base, step, 0.08, 1.2);
            run.frame();
        };
        for (let frame = 0; frame < REPROJECTION_WARM_FRAMES; frame++) advance();
        await device.queue.onSubmittedWorkDone();
        const probes = run.technique.probes;
        if (!probes) throw new Error('no probes after encodeScreen');
        const { layout } = probes;
        const [maskWidth, maskHeight] = layout.probeCount;
        const [atlasWidth, atlasHeight] = layout.atlasSize;

        // Last frame: its camera, screen, mask and atlas.
        const previousCamera = run.camera;
        const previousScreen = await readScreen(device, run);
        const previousGeometry = screenGeometry(previousScreen, previousCamera);
        const previousMask = new Uint32Array((await readTexture(device, probes.probeMask, 4, maskWidth, maskHeight)).buffer);
        const previousAtlas = halvesFast(await readTexture(device, probes.probeAtlas, 8, atlasWidth, atlasHeight));

        advance();
        await device.queue.onSubmittedWorkDone();
        const screen = await readScreen(device, run);
        const geometry = screenGeometry(screen, run.camera);
        const mask = new Uint32Array((await readTexture(device, probes.probeMask, 4, maskWidth, maskHeight)).buffer);
        const atlas = halvesFast(await readTexture(device, probes.probeAtlas, 8, atlasWidth, atlasHeight));
        const spawnWords = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + layout.maxProbeSpawnCount * 4));
        const blendedTiles = new Set(Array.from(spawnWords.subarray(1, 1 + spawnWords[0]), (seed) => {
            const [x, y] = unpackSeed(seed);
            return Math.floor(y / PROBE_SIZE) * maskWidth + Math.floor(x / PROBE_SIZE);
        }));
        const counters = new Uint32Array(await readBuffer(device, probes.counters, 64));
        const emptyCount = counters[COUNTER.emptyTileCount];
        const emptyList = new Set(new Uint32Array(await readBuffer(device, probes.history.emptyTiles, Math.max(emptyCount, 1) * 4)).subarray(0, emptyCount));

        // The velocity: the point a pixel shows now is the one its previous pixel showed.
        let velocityPixels = 0;
        let velocityMismatches = 0;
        for (let y = 4; y < HEIGHT; y += 23) {
            for (let x = 4; x < WIDTH; x += 23) {
                if (geometry.isSky(x, y)) continue;
                const v = [screen.velocity[(y * WIDTH + x) * 4], screen.velocity[(y * WIDTH + x) * 4 + 1]];
                const px = Math.trunc(((x + 0.5) / WIDTH - 0.5 * v[0]) * WIDTH);
                const py = Math.trunc(((y + 0.5) / HEIGHT + 0.5 * v[1]) * HEIGHT);
                if (px < 0 || py < 0 || px >= WIDTH || py >= HEIGHT || previousGeometry.isSky(px, py)) continue;
                const now = geometry.world(x, y);
                const before = previousGeometry.world(px, py);
                const distance = Math.hypot(now[0] - before[0], now[1] - before[1], now[2] - before[2]);
                const footprint = Math.hypot(now[0] - run.camera.position[0], now[1] - run.camera.position[1], now[2] - run.camera.position[2]) * (2 * Math.tan((run.camera.fovYDeg * Math.PI) / 360)) / HEIGHT;
                velocityPixels++;
                // Within a couple of pixels (truncation and a depth edge's neighbour).
                if (distance > 3 * footprint) velocityMismatches++;
            }
        }
        if (!(velocityMismatches <= 0.02 * velocityPixels)) fail(`velocity: ${velocityMismatches} of ${velocityPixels} pixels' previous pixel shows another point`);

        const frame = frameCamera(run.camera, WIDTH, HEIGHT);
        // The probes' frame index is a snapshot: this frame's comes from a fresh read.
        const frameIndex = run.technique.probes?.frameIndex ?? -1;
        const inputs: ReprojectionInputs = {
            width: WIDTH,
            height: HEIGHT,
            frameIndex,
            cellSize: screenProbeCellSize(frame, WIDTH, HEIGHT),
            eye: [frame.position[0], frame.position[1], frame.position[2]],
            isSky: geometry.isSky,
            worldPosition: geometry.world,
            normal: geometry.normal,
            velocity: (x, y) => [screen.velocity[(y * WIDTH + x) * 4], screen.velocity[(y * WIDTH + x) * 4 + 1]],
            previousProbeMask: (tileX, tileY) => previousMask[tileY * maskWidth + tileX],
            previousWorldPosition: previousGeometry.world,
            previousNormal: previousGeometry.normal,
            previousAtlas: (x, y) => {
                const texel = (y * atlasWidth + x) * 4;
                return [previousAtlas[texel], previousAtlas[texel + 1], previousAtlas[texel + 2], previousAtlas[texel + 3]];
            },
        };
        let checkedTiles = 0;
        let foundTiles = 0;
        let scoreTies = 0;
        let seedMismatches = 0;
        let cellMismatches = 0;
        let emptyMismatches = 0;
        let worstCellError = 0;
        const examples: string[] = [];
        const emptyExamples: string[] = [];
        for (let tileY = 0; tileY < maskHeight; tileY++) {
            for (let tileX = 0; tileX < maskWidth; tileX++) {
                const tile = tileY * maskWidth + tileX;
                let cpu = reprojectProbeTile(inputs, tileX, tileY, floatToHalfBits);
                if (cpu.empty !== emptyList.has(tile)) {
                    emptyMismatches++;
                    if (emptyExamples.length < 6) emptyExamples.push(`tile ${tileX},${tileY}: GPU ${emptyList.has(tile) ? 'listed' : 'not listed'}, CPU ${cpu.empty ? 'empty' : 'not empty'} (CPU seed ${cpu.seed === INVALID_ID ? 'none' : unpackSeed(cpu.seed).join(',')}, GPU mask ${mask[tile] === INVALID_ID ? 'none' : unpackSeed(mask[tile]).join(',')})`);
                }
                if (blendedTiles.has(tile)) continue; // the blend wrote this tile after the reprojection
                checkedTiles++;
                if (cpu.seed !== mask[tile]) {
                    // The score is a half: where the GPU's f32 and the CPU's f64 distance round to
                    // neighbouring halves, two pixels swap places. Such a tie is counted, and the cells
                    // compared from the GPU's pixel.
                    const gpuLane = mask[tile] === INVALID_ID ? -1 : ((unpackSeed(mask[tile])[1] % PROBE_SIZE) << 3) | (unpackSeed(mask[tile])[0] % PROBE_SIZE);
                    const best = Math.min(...cpu.scores.filter((score) => score >= 0));
                    const gpuScore = gpuLane >= 0 ? cpu.scores[gpuLane] : -1;
                    if (cpu.seed !== INVALID_ID && gpuScore >= 0 && (gpuScore >>> 16) - (best >>> 16) <= 1) {
                        scoreTies++;
                        cpu = reprojectProbeTile(inputs, tileX, tileY, floatToHalfBits, gpuLane);
                    } else {
                        seedMismatches++;
                        if (examples.length < 4) examples.push(`tile ${tileX},${tileY}: GPU seed ${mask[tile] === INVALID_ID ? 'none' : unpackSeed(mask[tile]).join(',')}, CPU ${cpu.seed === INVALID_ID ? 'none' : unpackSeed(cpu.seed).join(',')}`);
                        continue;
                    }
                }
                if (cpu.seed === INVALID_ID) continue;
                foundTiles++;
                let tileWrong = false;
                for (let cell = 0; cell < PROBE_CELLS; cell++) {
                    const texel = ((tileY * PROBE_SIZE + Math.floor(cell / PROBE_SIZE)) * atlasWidth + tileX * PROBE_SIZE + (cell % PROBE_SIZE)) * 4;
                    for (let channel = 0; channel < 4; channel++) {
                        const expected = cpu.cells[cell][channel];
                        const error = Math.abs(atlas[texel + channel] - expected) / Math.max(Math.abs(expected), 1);
                        worstCellError = Math.max(worstCellError, error);
                        if (error > 2e-3) tileWrong = true;
                    }
                }
                if (tileWrong) {
                    cellMismatches++;
                    if (examples.length < 4) examples.push(`tile ${tileX},${tileY}: cells differ from the CPU`);
                }
            }
        }
        const outliers = seedMismatches + cellMismatches;
        if (!(outliers <= REPROJECTION_OUTLIER_FRACTION * checkedTiles)) fail(`reprojection: ${seedMismatches} seeds and ${cellMismatches} tiles' cells of ${checkedTiles} differ from the CPU (${examples.join('; ')})`);
        if (!(emptyMismatches <= REPROJECTION_OUTLIER_FRACTION * maskWidth * maskHeight)) fail(`reprojection: ${emptyMismatches} tiles' empty-tile listing differs from the CPU's`);
        if (!(foundTiles >= 0.5 * checkedTiles)) fail(`reprojection: only ${foundTiles} of ${checkedTiles} tiles found last frame's probe`);
        if (emptyCount === 0) fail('reprojection: the motion made no empty tile, so the empty-tile check checks nothing');
        return { still, velocityPixels, velocityMismatches, checkedTiles, foundTiles, scoreTies, seedMismatches, cellMismatches, worstCellError, emptyTiles: emptyCount, emptyMismatches, examples, emptyExamples };
    } finally {
        run.dispose();
        sky.destroy();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- 4. The off-screen cache and the patch -----------------------------------------------------

async function checkCacheAndPatch(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, unknown>> {
    const sky = skyTexture(device, varyingSky);
    // Low, looking across the step at the wall; the sweep sideways hides and shows the floor behind the step.
    const base: PageCamera = { position: [-4, 1.6, 0], target: [0, 0.6, -8], fovYDeg: 60, near: 0.1, far: 1000 };
    const run = await startRun(device, 'cache', { type: 'texture', texture: sky, layout: 'equirect' }, base, true);
    try {
        const totals = { evictedNew: 0, evictedFound: 0, updated: 0, listElements: 0, patched: 0, emptyTiles: 0, overrideTiles: 0 };
        let lruFailures = 0;
        let mruFailures = 0;
        let patchFailures = 0;
        for (let frame = 1; frame <= CACHE_FRAMES; frame++) {
            // Sideways and back (a triangle wave), so tiles leave and come back.
            const phase = frame % 20;
            const offset = phase < 10 ? phase : 20 - phase;
            run.camera = { ...base, position: [base.position[0] + 0.5 * offset, base.position[1], base.position[2]], target: [base.target[0] + 0.5 * offset, base.target[1], base.target[2]] };
            run.frame();
            await device.queue.onSubmittedWorkDone();
            const probes = run.technique.probes;
            if (!probes) throw new Error('no probes after encodeScreen');
            const { history, layout } = probes;
            const tiles = history.probeTileCount;
            const counters = new Uint32Array(await readBuffer(device, probes.counters, 64));
            const lruCount = counters[COUNTER.cachedTileLRUCount];
            const mruCount = counters[COUNTER.cachedTileMRUCount];
            totals.listElements += counters[COUNTER.cachedTileListElementCount];
            totals.emptyTiles += counters[COUNTER.emptyTileCount];
            totals.overrideTiles += counters[COUNTER.overrideTileCount];

            // The LRU list after the reorder: a permutation of the slots.
            const lru = new Uint32Array(await readBuffer(device, history.cachedTileLRU, tiles * 4));
            const seen = new Uint8Array(tiles);
            let permutation = true;
            for (const entry of lru) {
                const slot = entry & ~CACHED_TILE_NEW;
                if (slot >= tiles || seen[slot]) permutation = false;
                else seen[slot] = 1;
            }
            if (!permutation) {
                lruFailures++;
                fail(`cache, frame ${frame}: the LRU list is not a permutation of the ${tiles} slots`);
            }
            // This frame's flags: the used entries (0) are exactly those with an MRU number, numbered 0..n-1.
            const flags = new Uint32Array(await readBuffer(device, probes.cachedTileLRUFlags, tiles * 4));
            const mru = new Uint32Array(await readBuffer(device, history.cachedTileMRU, tiles * 4));
            const used = Array.from(flags.keys()).filter((entry) => flags[entry] === 0);
            const numbers = new Set(used.map((entry) => mru[entry]));
            if (used.length !== mruCount || numbers.size !== mruCount || used.some((entry) => mru[entry] >= mruCount)) {
                mruFailures++;
                fail(`cache, frame ${frame}: ${used.length} entries used, ${mruCount} MRU numbers handed out (${numbers.size} distinct)`);
            }
            // Evictions and refreshes this frame (RcProbeSpawnProbe).
            const spawnWords = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + layout.maxProbeSpawnCount * 4));
            const probeCount = spawnWords[0];
            const spawnProbe = new Uint32Array(await readBuffer(device, history.spawnProbe, Math.max(probeCount, 1) * PROBE_SPAWN_PROBE_BYTES));
            for (let probe = 0; probe < probeCount; probe++) {
                const evicted = spawnProbe[probe * 4];
                if (evicted !== INVALID_ID) {
                    if ((evicted & CACHED_TILE_NEW) !== 0) totals.evictedNew++;
                    else totals.evictedFound++;
                }
                if (spawnProbe[probe * 4 + 1] !== INVALID_ID) totals.updated++;
            }
            if (lruCount > probeCount) fail(`cache, frame ${frame}: ${lruCount} LRU entries allocated for ${probeCount} probes`);

            // The patch: every probe is the spawn's, or a probe moved from the override list into an empty tile.
            const depth = new Float32Array((await readTexture(device, run.screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
            const isSky = (x: number, y: number): boolean => depth[y * WIDTH + x] === 1;
            const spawned = spawnScreenProbes(WIDTH, HEIGHT, probes.frameIndex, isSky).filter((tile) => tile.spawned);
            const emptyCount = counters[COUNTER.emptyTileCount];
            const overrideCount = counters[COUNTER.overrideTileCount];
            const emptyTiles = new Uint32Array(await readBuffer(device, history.emptyTiles, Math.max(emptyCount, 1) * 4)).subarray(0, emptyCount);
            const overrideTiles = new Set(new Uint32Array(await readBuffer(device, history.overrideTiles, Math.max(overrideCount, 1) * 4)).subarray(0, overrideCount));
            const [hx, hy] = calculateHaltonSequence(probes.frameIndex);
            const jitter = [Math.trunc(Math.min(hx * PROBE_SIZE, PROBE_SIZE - 1)), Math.trunc(Math.min(hy * PROBE_SIZE, PROBE_SIZE - 1))];
            const patchSeeds = new Set<number>();
            for (const tile of emptyTiles) {
                const x = Math.min((tile % layout.probeCount[0]) * PROBE_SIZE + jitter[0], WIDTH - 1);
                const y = Math.min(Math.floor(tile / layout.probeCount[0]) * PROBE_SIZE + jitter[1], HEIGHT - 1);
                if (!isSky(x, y)) patchSeeds.add(packSeed(x, y));
            }
            if (probeCount !== spawned.length) fail(`patch, frame ${frame}: ${probeCount} probes, the spawn has ${spawned.length}`);
            for (let probe = 0; probe < Math.min(probeCount, spawned.length); probe++) {
                const seed = spawnWords[1 + probe];
                if (seed === packSeed(spawned[probe].seed[0], spawned[probe].seed[1])) continue;
                totals.patched++;
                if (!overrideTiles.has(probe) || !patchSeeds.has(seed)) {
                    patchFailures++;
                    if (patchFailures <= 3) fail(`patch, frame ${frame}: probe ${probe} moved to ${unpackSeed(seed).join(',')}, ${overrideTiles.has(probe) ? 'not an empty tile the CPU patches' : 'but it was not on the override list'}`);
                }
            }
        }
        if (totals.evictedNew + totals.evictedFound === 0) fail('cache: the sweep evicted no probe into the cache');
        if (totals.listElements === 0) fail('cache: no cached probe was ever listed on the screen');
        if (totals.patched === 0) fail('patch: the sweep patched no probe');
        return { frames: CACHE_FRAMES, ...totals, lruFailures, mruFailures, patchFailures };
    } finally {
        run.dispose();
        sky.destroy();
        await device.queue.onSubmittedWorkDone();
    }
}

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    window.addEventListener('pagehide', () => device.destroy(), { once: true });
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 30) failures.push(message);
    };
    const data: Record<string, unknown> = { width: WIDTH, height: HEIGHT };
    const only = new URLSearchParams(location.search).get('only') ?? (import.meta.env.VITE_RC_TEMPORAL_ONLY as string | undefined);
    const sections: Array<[string, (device: GPUDevice, fail: (message: string) => void) => Promise<Record<string, unknown>>]> = [
        ['constantSky', checkConstantSky],
        ['overflow', checkOverflow],
        ['reprojection', checkReprojection],
        ['cacheAndPatch', checkCacheAndPatch],
    ];
    for (const [name, section] of sections) {
        if (only && only !== name) continue;
        data[name] = await section(device, fail);
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
