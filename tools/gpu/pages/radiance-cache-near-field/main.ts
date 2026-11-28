/**
 * RC4.2: the probe filter (`FilterScreenProbes`) and SSGI (the platform's
 * `gi/ssgi`, Capsaicin's SSGI technique) feeding `InterpolateScreenProbes`,
 * on the GPU, against analytic answers and CPU ports. Engine-free: the
 * `FrameScreen` is drawn by primary rays (radiance-cache-probes/screen.ts),
 * which also give `albedo` and `lighting` (emission plus one sample of direct
 * light).
 *
 * 1. **A uniform field through the filter**: a floor under a constant sky.
 *    Every cell of every probe reads the sky, before the filter (its first
 *    pass's input is copied, the scratch) and after; the filter must not move
 *    any of them.
 * 2. **The filter against its CPU port** on a 1 m step and a 3 m wall (depth
 *    edges both ways) under the constant sky: the second (down) pass's output,
 *    every listed probe's 64 cells, against `filterScreenProbesCell` run on
 *    the GPU's own scratch, depth, normals and mask. The number of
 *    neighbours the plane test refused is reported, so the edges are known to
 *    be exercised (a filter that blurs across them fails the comparison).
 * 3. **SSGI on an open plane**: the floor under the constant sky: the
 *    visibility is 1, the bent normal is the normal and the near-field light
 *    is 0 (it would be a share of the floor's own light otherwise).
 * 4. **SSGI in a corner**: a floor and a wall meeting at x = 0. Per floor pixel
 *    at distance d from the wall, over several frames (one random slice a
 *    pixel a frame), the mean visibility against the analytic cosine-weighted
 *    visibility of an infinite wall seen within SSGI's radius R (directions
 *    with ω_x < −d / R are blocked), and the bent normal's tilt away from the
 *    wall (analytic at the wall's foot: atan(2 / π) = 32.5°).
 * 5. **A near-field furnace**: the same corner with every surface emitting L
 *    and reflecting nothing, under a sky of L. Every direction carries L, so
 *    the probes' cone (far field) plus SSGI's near-field light must read πL.
 *    What Capsaicin's units would read (near-field / π, register D52) is
 *    reported beside it.
 *
 * All sections check the outputs for NaN, infinite and negative values.
 *
 * `VITE_RC_COMPARE` runs it at compare toggles (RC4.5, radiance-cache-probes/compare-env.ts),
 * for example SSGI's single frame: `VITE_RC_COMPARE=ssgiHistoryFrames=1`.
 *
 * Correctness only. Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-near-field --gpu nvidia`.
 */

import type { EnvironmentDescriptor, MaterialDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { cameraRelativeInverseViewProjection } from '../../../../packages/cache/src/camera.ts';
import { filterScreenProbesCell, unpackSeed, type Direction, type ProbeFilterInputs } from '../../../../packages/cache/src/capsaicin-math.ts';
import { screenProbeCellSize, viewForward } from '../../../../packages/cache/src/screen-irradiance.ts';
import type { RadianceCacheTechnique } from '../../../../packages/cache/src/radiance-cache-technique.ts';
import { PROBE_SIZE } from '../../../../packages/cache/src/settings.ts';
import { PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/cache/src/wgsl/screen-probes.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { readBuffer } from '../ddgi-shared/harness.ts';
import { PAGE_COMPARE_VALUES, pageRadianceCache } from '../radiance-cache-probes/compare-env.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { PageScreen, frameCamera, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 960;
const HEIGHT = 540;
const SKY = 1.5;
/** Section 1: a probe cell's change through the filter, relative (halves hold 1.5 exactly). */
const UNIFORM_TOLERANCE = 1e-3;
/** Section 2: a cell's GPU value against the CPU port, relative to the larger of the two and 1% of the sky; outliers allowed (f32 vs f64 at a test's threshold). */
const FILTER_TOLERANCE = 0.01;
const FILTER_OUTLIER_FRACTION = 0.001;
/** Section 3: the open plane's visibility (mean, 5th percentile), the bent normal's mean tilt and the near-field light against πL. */
const PLANE_MEAN_AO = 0.99;
const PLANE_P5_AO = 0.97;
const PLANE_BENT_DEGREES = 2;
const PLANE_NEAR_FIELD = 0.005;
/** Section 3b: a normal map leaning the normal up to 28.6 degrees each way along x, and the visibility's tolerance per lean. */
const NORMAL_MAP = { tilt: 0.5, wavelength: 0.6 } as const;
const NORMAL_MAP_TOLERANCE = 0.03;
/**
 * Section 4, at the wall's foot (d < R / 10): with many slices and steps (`DENSE_SSGI`) the visibility
 * against the analytic and the bent normal's tilt about the analytic 32.5 degrees; with Capsaicin's
 * defaults (two steps a side, which can miss the wall's highest point within R) the visibility between
 * the analytic and `CORNER_DEFAULT_MAX_AO`, and the bent normal leaning away from the wall.
 */
const CORNER_AO_TOLERANCE = 0.05;
const CORNER_TILT_DEGREES: readonly [number, number] = [24, 41];
const CORNER_DEFAULT_MAX_AO = 0.75;
const CORNER_DEFAULT_MIN_TILT = 10;
const DENSE_SSGI = { sliceCount: 4, stepCount: 16 } as const;
/** Section 5: E / πL near the wall (d < R) and away from it (d > 2R). */
const FURNACE_NEAR_TOLERANCE = 0.05;
const FURNACE_FAR_TOLERANCE = 0.02;
/** Frames before reading (the history and the lit image SSGI reads settle), and frames averaged in sections 4 and 5. */
const SETTLE_FRAMES = 12;
const AVERAGE_FRAMES = 8;
/** SSGI's default radius (ssgi.h:49). */
const RADIUS = 0.2;

const OPEN_CAMERA: PageCamera = { position: [0, 3, 0], target: [0, 1, -10], fovYDeg: 60, near: 0.1, far: 1000 };
const STEP_CAMERA: PageCamera = { position: [0, 3, 0], target: [0, 1, -10], fovYDeg: 60, near: 0.1, far: 1000 };
/** Looking at the corner line from 1.4 m away, slightly from above. */
const CORNER_CAMERA: PageCamera = { position: [1.0, 0.9, 1.0], target: [0, 0, -0.6], fovYDeg: 60, near: 0.05, far: 1000 };

type Quad = { corners: readonly Direction[]; normal: Direction };

function quadsMesh(quads: readonly Quad[]): { positions: Float32Array; normals: Float32Array; indices: Uint32Array } {
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

const FLOOR: Quad = { corners: [[-200, 0, 200], [200, 0, 200], [200, 0, -200], [-200, 0, -200]], normal: [0, 1, 0] };
/** A 1 m box and a wall 25 m away (radiance-cache-image's step), for depth edges. */
const STEP: readonly Quad[] = [
    { corners: [[-2, 1, -6], [1, 1, -6], [1, 1, -9], [-2, 1, -9]], normal: [0, 1, 0] },
    { corners: [[-2, 0, -6], [1, 0, -6], [1, 1, -6], [-2, 1, -6]], normal: [0, 0, 1] },
    { corners: [[1, 0, -6], [1, 0, -9], [1, 1, -9], [1, 1, -6]], normal: [1, 0, 0] },
    { corners: [[-2, 0, -9], [-2, 0, -6], [-2, 1, -6], [-2, 1, -9]], normal: [-1, 0, 0] },
    { corners: [[-200, 0, -25], [200, 0, -25], [200, 3, -25], [-200, 3, -25]], normal: [0, 0, 1] },
];
/**
 * A 0.3 m slab on z < 0, seen from 10 m straight above (its edge runs across the screen, so the
 * second, downward pass crosses it), and an emissive strip 0.25 m high 8 m away at z = 8: the floor's
 * probes by the slab see the strip on their horizon, the slab's (0.3 m up) don't. Across the slab's
 * edge only the plane test keeps them apart.
 */
const SLAB: readonly Quad[] = [
    { corners: [[-20, 0.3, 0], [20, 0.3, 0], [20, 0.3, -20], [-20, 0.3, -20]], normal: [0, 1, 0] },
    { corners: [[-20, 0, 0], [20, 0, 0], [20, 0.3, 0], [-20, 0.3, 0]], normal: [0, 0, 1] },
];
const EMISSIVE_EDGE: Quad = { corners: [[-20, 0, 8], [-20, 0.25, 8], [20, 0.25, 8], [20, 0, 8]], normal: [0, 0, -1] };
const EMISSIVE_EDGE_RADIANCE = 100;
const SLAB_CAMERA: PageCamera = { position: [0, 10, 0], target: [0, 0, -0.001], fovYDeg: 60, near: 0.1, far: 1000 };
/** The corner: the floor on x > 0 and a 6 m wall at x = 0 facing +x. */
const CORNER: readonly Quad[] = [
    { corners: [[0, 0, 100], [100, 0, 100], [100, 0, -100], [0, 0, -100]], normal: [0, 1, 0] },
    { corners: [[0, 0, -100], [0, 6, -100], [0, 6, 100], [0, 0, 100]], normal: [1, 0, 0] },
];

interface Run {
    readonly technique: RadianceCacheTechnique;
    readonly screen: PageScreen;
    readonly camera: PageCamera;
    frame(): void;
    dispose(): void;
}

/** Extra surfaces with their own material, and a normal map on every surface (PageScreen.normalBump). */
interface RunExtras {
    readonly also?: ReadonlyArray<{ readonly quads: readonly Quad[]; readonly material: MaterialDescriptor }>;
    readonly normalBump?: PageScreen['normalBump'];
}

/** The filter's output: the filtered atlas, or the atlas itself with Capsaicin's filtered history (`probeFilterHistory`, D59). */
function filterOutput(technique: RadianceCacheTechnique): GPUTexture {
    const probes = technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    return technique.settings.probeFilterHistory ? probes.probeAtlas : probes.filteredAtlas;
}

async function makeRun(device: GPUDevice, name: string, quads: readonly Quad[], material: MaterialDescriptor, environment: EnvironmentDescriptor, camera: PageCamera, extras: RunExtras = {}): Promise<Run> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-near-field ${name}` });
    // SSGI is opt-in (D61); VITE_RC_COMPARE sets compare toggles over it (RC4.5).
    const technique = platform.addTechnique(pageRadianceCache({ ssgi: true }));
    platform.setOutput(technique);
    await platform.prepare();
    const scene = platform.scene;
    const mesh = scene.createMesh({ ...quadsMesh(quads), label: name });
    const surface = scene.createMaterial({ ...material, label: name });
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    scene.createInstance({ mesh, materials: [surface], transform: identity, label: name });
    for (const [index, extra] of (extras.also ?? []).entries()) {
        const extraMesh = scene.createMesh({ ...quadsMesh(extra.quads), label: `${name} ${index}` });
        scene.createInstance({ mesh: extraMesh, materials: [scene.createMaterial({ ...extra.material, label: `${name} ${index}` })], transform: identity, label: `${name} ${index}` });
    }
    scene.setEnvironment(environment);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    if (extras.normalBump) screen.normalBump = extras.normalBump;
    const input = (): FrameInput => ({ deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) });
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `radiance-cache-near-field ${name}` });
        platform.encode(encoder, input());
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    frame();
    await platform.rays.whenBuilt();
    for (let index = 0; index < SETTLE_FRAMES; index++) frame();
    await device.queue.onSubmittedWorkDone();
    return {
        technique,
        screen,
        camera,
        frame,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

const constantSky = (radiance: number): EnvironmentDescriptor => ({ type: 'color', color: [radiance, radiance, radiance], intensity: 1 });

/** The run's screen as the CPU sees it: positions from depth (the technique's own reconstruction), normals, sky. */
async function readScreen(device: GPUDevice, run: Run) {
    const depth = new Float32Array((await readTexture(device, run.screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
    const normals = halves(await readTexture(device, run.screen.screen.normal, 8, WIDTH, HEIGHT));
    const camera = frameCamera(run.camera, WIDTH, HEIGHT);
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
    return { camera, eye: [eye[0], eye[1], eye[2]] as Direction, isSky: (x: number, y: number) => depth[y * WIDTH + x] === 1, worldPosition, normal };
}

async function readSeeds(device: GPUDevice, run: Run): Promise<number[]> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const words = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + probes.layout.maxProbeSpawnCount * 4));
    return Array.from(words.subarray(1, 1 + words[0]));
}

function percentile(values: number[], fraction: number): number {
    if (values.length === 0) return NaN;
    const sorted = Float64Array.from(values).sort();
    return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

const mean = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);

/** Counts of NaN or infinite values in the first `channels` of rgba halves, and of negative ones in the `nonNegative` channels. */
function badValues(values: Float32Array, channels = 3, nonNegative: readonly number[] = [0, 1, 2]): { nonFinite: number; negative: number } {
    let nonFinite = 0;
    let negative = 0;
    for (let index = 0; index < values.length; index += 4) {
        for (let channel = 0; channel < channels; channel++) {
            const value = values[index + channel];
            if (!Number.isFinite(value)) nonFinite++;
            else if (value < 0 && nonNegative.includes(channel)) negative++;
        }
    }
    return { nonFinite, negative };
}

/**
 * The cosine-weighted visibility of a floor point d from an infinite wall,
 * occluders counted within `radius`: directions with ω_x < −d / radius are
 * blocked. Integrated over the hemisphere on a fine grid.
 */
function analyticCornerVisibility(d: number, radius: number): number {
    const steps = 2000;
    let blocked = 0;
    for (let i = 0; i < steps; i++) {
        // cos θ uniform in [0, 1) is the cosine-weighted measure's elevation.
        const cosineSquared = (i + 0.5) / steps;
        const sinTheta = Math.sqrt(1 - cosineSquared);
        for (let j = 0; j < steps; j++) {
            const phi = ((j + 0.5) / steps) * 2 * Math.PI;
            if (sinTheta * Math.cos(phi) < -d / radius) blocked++;
        }
    }
    return 1 - blocked / (steps * steps);
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
    const data: Record<string, unknown> = { width: WIDTH, height: HEIGHT, compare: PAGE_COMPARE_VALUES };
    const grey: MaterialDescriptor = { albedo: [0.5, 0.5, 0.5] };

    // 1 and 3: the open floor under a constant sky.
    const open = await makeRun(device, 'open', [FLOOR], grey, constantSky(SKY), OPEN_CAMERA);
    try {
        const probes = open.technique.probes;
        if (!probes?.occlusion) throw new Error('no probes or no SSGI after encodeScreen');
        const [atlasWidth, atlasHeight] = probes.layout.atlasSize;
        const atlas = halves(await readTexture(device, filterOutput(open.technique), 8, atlasWidth, atlasHeight));
        const scratch = halves(await readTexture(device, probes.filterScratch, 8, atlasWidth, atlasHeight));
        const seeds = await readSeeds(device, open);
        let worst = 0;
        for (const seed of seeds) {
            const [sx, sy] = unpackSeed(seed);
            const tile = [Math.floor(sx / PROBE_SIZE) * PROBE_SIZE, Math.floor(sy / PROBE_SIZE) * PROBE_SIZE];
            for (let cell = 0; cell < 64; cell++) {
                const texel = ((tile[1] + Math.floor(cell / PROBE_SIZE)) * atlasWidth + tile[0] + (cell % PROBE_SIZE)) * 4;
                for (let channel = 0; channel < 3; channel++) {
                    worst = Math.max(worst, Math.abs(atlas[texel + channel] / SKY - 1), Math.abs(scratch[texel + channel] / SKY - 1));
                }
            }
        }
        if (seeds.length < 100) fail(`uniform field: only ${seeds.length} probes`);
        if (!(worst <= UNIFORM_TOLERANCE)) fail(`uniform field: a filtered cell is ${(worst * 100).toFixed(3)}% off the sky`);
        data.uniformField = { probes: seeds.length, worstCellError: worst };

        // 3: SSGI on the open plane.
        const screen = await readScreen(device, open);
        const occlusion = halves(await readTexture(device, probes.occlusion.occlusion, 8, WIDTH, HEIGHT));
        const nearField = halves(await readTexture(device, probes.occlusion.nearField, 8, WIDTH, HEIGHT));
        const aos: number[] = [];
        const tilts: number[] = [];
        const nears: number[] = [];
        // The visibility by distance from the camera, and the five lowest pixels, to show where any loss is.
        const bands = [0, 5, 10, 20, 50, 100, 1e9];
        const bandSums = bands.slice(1).map(() => ({ sum: 0, low: 0, count: 0 }));
        const lowest: Array<{ x: number; y: number; ao: number; distance: number }> = [];
        for (let y = 0; y < HEIGHT; y++) {
            for (let x = 0; x < WIDTH; x++) {
                if (screen.isSky(x, y)) continue;
                const index = (y * WIDTH + x) * 4;
                aos.push(occlusion[index + 3]);
                const distance = Math.hypot(...screen.worldPosition(x, y).map((value, axis) => value - screen.eye[axis]));
                const band = bandSums[bands.findIndex((edge, at) => at < bands.length - 1 && distance >= edge && distance < bands[at + 1])];
                band.sum += occlusion[index + 3];
                band.count++;
                if (occlusion[index + 3] < 0.97) band.low++;
                lowest.push({ x, y, ao: occlusion[index + 3], distance });
                if (lowest.length > 5) lowest.sort((a, b) => a.ao - b.ao).pop();
                const length = Math.hypot(occlusion[index], occlusion[index + 1], occlusion[index + 2]);
                tilts.push((Math.acos(Math.min(occlusion[index + 1] / length, 1)) * 180) / Math.PI);
                nears.push((nearField[index] + nearField[index + 1] + nearField[index + 2]) / 3 / (Math.PI * SKY));
            }
        }
        const plane = { pixels: aos.length, meanAO: mean(aos), p5AO: percentile(aos, 0.05), minAO: percentile(aos, 0), meanBentTiltDegrees: mean(tilts), p99BentTiltDegrees: percentile(tilts, 0.99), meanNearFieldOverPiL: mean(nears), p99NearFieldOverPiL: percentile(nears, 0.99) };
        const byDistance = bandSums.map((band, at) => ({ fromMetres: bands[at], meanAO: band.sum / Math.max(band.count, 1), below97: band.low, pixels: band.count }));
        data.openPlane = { ...plane, byDistance, lowest, occlusion: badValues(occlusion, 4, [3]), nearField: badValues(nearField) };
        if (!(plane.meanAO >= PLANE_MEAN_AO)) fail(`open plane: mean visibility ${plane.meanAO.toFixed(4)}`);
        if (!(plane.p5AO >= PLANE_P5_AO)) fail(`open plane: 5th percentile visibility ${plane.p5AO.toFixed(4)}`);
        if (!(plane.meanBentTiltDegrees <= PLANE_BENT_DEGREES)) fail(`open plane: the bent normal tilts ${plane.meanBentTiltDegrees.toFixed(2)} degrees on average`);
        if (!(Math.abs(plane.meanNearFieldOverPiL) <= PLANE_NEAR_FIELD)) fail(`open plane: near-field light ${(plane.meanNearFieldOverPiL * 100).toFixed(3)}% of pi L`);
        for (const [what, counts] of [['occlusion', badValues(occlusion, 4, [3])], ['near field', badValues(nearField)]] as const) {
            if (counts.nonFinite > 0 || counts.negative > 0) fail(`open plane ${what}: ${counts.nonFinite} not finite, ${counts.negative} negative`);
        }
    } finally {
        open.dispose();
        await device.queue.onSubmittedWorkDone();
    }

    // 3b: the open floor with a normal map: the samples must stay on the floor (D56 takes their plane
    // from the depth), so a pixel whose normal leans θ off the floor's sees the floor cut its
    // cosine lobe: visibility (1 + cos θ) / 2.
    const bumped = await makeRun(device, 'normal map', [FLOOR], grey, constantSky(SKY), OPEN_CAMERA, { normalBump: NORMAL_MAP });
    try {
        const probes = bumped.technique.probes;
        if (!probes?.occlusion) throw new Error('no SSGI after encodeScreen');
        const screen = await readScreen(device, bumped);
        const edges = [0, 5, 10, 15, 20, 25, 30];
        const sums = edges.slice(1).map(() => ({ ao: 0, expected: 0, count: 0 }));
        for (let frame = 0; frame < AVERAGE_FRAMES; frame++) {
            bumped.frame();
            const occlusion = halves(await readTexture(device, probes.occlusion.occlusion, 8, WIDTH, HEIGHT));
            for (let y = 0; y < HEIGHT; y++) {
                for (let x = 0; x < WIDTH; x++) {
                    if (screen.isSky(x, y)) continue;
                    // Within 20 m, where the map's 0.6 m wavelength spans several pixels.
                    const position = screen.worldPosition(x, y);
                    if (Math.hypot(...position.map((value, axis) => value - screen.eye[axis])) > 20) continue;
                    const lean = Math.acos(Math.min(screen.normal(x, y)[1], 1));
                    const degrees = (lean * 180) / Math.PI;
                    const bin = edges.findIndex((edge, at) => at < edges.length - 1 && degrees >= edge && degrees < edges[at + 1]);
                    if (bin < 0) continue;
                    sums[bin].ao += occlusion[(y * WIDTH + x) * 4 + 3];
                    sums[bin].expected += (1 + Math.cos(lean)) / 2;
                    sums[bin].count++;
                }
            }
        }
        const table = sums.map((sum, bin) => ({ leanFrom: edges[bin], leanTo: edges[bin + 1], samples: sum.count, meanAO: sum.ao / Math.max(sum.count, 1), expectedAO: sum.expected / Math.max(sum.count, 1) }));
        data.normalMappedPlane = { tiltDegrees: (NORMAL_MAP.tilt * 180) / Math.PI, wavelength: NORMAL_MAP.wavelength, bins: table };
        for (const row of table.filter((entry) => entry.samples >= 1000)) {
            if (!(Math.abs(row.meanAO - row.expectedAO) <= NORMAL_MAP_TOLERANCE)) fail(`normal-mapped plane: visibility ${row.meanAO.toFixed(3)} where the normal leans ${row.leanFrom}-${row.leanTo} degrees, expected ${row.expectedAO.toFixed(3)}`);
        }
    } finally {
        bumped.dispose();
        await device.queue.onSubmittedWorkDone();
    }

    // 2: the filter's second pass against the CPU port, on depth edges: the step, and the slab whose
    // edge only the plane test refuses (the depth weight is 0.78 there and the low emissive wall's
    // parallax between slab and floor is under the 3.6 degree angle test).
    const filterScenes = [
        { name: 'step', quads: [FLOOR, ...STEP], camera: STEP_CAMERA, extras: {} },
        { name: 'slab', quads: [FLOOR, ...SLAB], camera: SLAB_CAMERA, extras: { also: [{ quads: [EMISSIVE_EDGE], material: { albedo: [0, 0, 0], emissive: [1, 1, 1], emissiveIntensity: EMISSIVE_EDGE_RADIANCE } }] } },
    ] as const;
    for (const filterScene of filterScenes) {
        const step = await makeRun(device, filterScene.name, filterScene.quads, grey, constantSky(SKY), filterScene.camera, filterScene.extras);
        try {
            const probes = step.technique.probes;
            if (!probes) throw new Error('no probes after encodeScreen');
            const { layout } = probes;
            const [atlasWidth, atlasHeight] = layout.atlasSize;
            const atlas = halves(await readTexture(device, filterOutput(step.technique), 8, atlasWidth, atlasHeight));
            const scratch = halves(await readTexture(device, probes.filterScratch, 8, atlasWidth, atlasHeight));
            const levels: Array<{ width: number; texels: Uint32Array }> = [];
            for (let level = 0; level < layout.probeMaskMipCount; level++) {
                const size = [Math.max(1, layout.probeCount[0] >> level), Math.max(1, layout.probeCount[1] >> level)];
                levels.push({ width: size[0], texels: new Uint32Array((await readTexture(device, probes.probeMask, 4, size[0], size[1], level)).buffer) });
            }
            const seeds = await readSeeds(device, step);
            const screen = await readScreen(device, step);
            const inputs: ProbeFilterInputs = {
                width: WIDTH,
                height: HEIGHT,
                cellSize: screenProbeCellSize(screen.camera, WIDTH, HEIGHT),
                eye: screen.eye,
                viewForward: viewForward(screen.camera),
                probeMaskMipCount: layout.probeMaskMipCount,
                worldPosition: screen.worldPosition,
                normal: screen.normal,
                probeMask: (level, x, y) => levels[level].texels[y * levels[level].width + x],
                atlas: (x, y) => {
                    const index = (y * atlasWidth + x) * 4;
                    return [scratch[index], scratch[index + 1], scratch[index + 2], scratch[index + 3]];
                },
            };
            let cells = 0;
            let outliers = 0;
            let worst = 0;
            let planeRefused = 0;
            let angleRefused = 0;
            let taps = 0;
            let changed = 0;
            for (const seed of seeds) {
                const probeSeed = unpackSeed(seed);
                const tile = [Math.floor(probeSeed[0] / PROBE_SIZE) * PROBE_SIZE, Math.floor(probeSeed[1] / PROBE_SIZE) * PROBE_SIZE];
                for (let cell = 0; cell < 64; cell++) {
                    const result = filterScreenProbesCell(inputs, probeSeed, cell, [0, 1]);
                    const texel = ((tile[1] + Math.floor(cell / PROBE_SIZE)) * atlasWidth + tile[0] + (cell % PROBE_SIZE)) * 4;
                    cells++;
                    taps += result.taps;
                    planeRefused += result.planeRefused;
                    angleRefused += result.angleRefused;
                    let error = 0;
                    for (let channel = 0; channel < 3; channel++) {
                        const gpu = atlas[texel + channel];
                        const cpu = result.value[channel];
                        error = Math.max(error, Math.abs(gpu - cpu) / Math.max(Math.abs(cpu), Math.abs(gpu), 0.01 * SKY));
                        if (Math.abs(scratch[texel + channel] - gpu) > 1e-3 * SKY) changed++;
                    }
                    if (error > FILTER_TOLERANCE) outliers++;
                    worst = Math.max(worst, error);
                }
            }
            const outlierFraction = outliers / Math.max(cells, 1);
            data[`filterAgainstCpu ${filterScene.name}`] = { probes: seeds.length, cells, taps, planeRefused, angleRefused, cellChannelsChangedByThePass: changed, outliers, outlierFraction, worstError: worst };
            if (!(outlierFraction <= FILTER_OUTLIER_FRACTION)) fail(`filter (${filterScene.name}): ${outliers} of ${cells} cells (${(outlierFraction * 100).toFixed(2)}%) differ from the CPU port by more than ${FILTER_TOLERANCE * 100}%`);
            if (planeRefused === 0) fail('filter: no neighbour was refused by the plane test, so the depth edges were not exercised');
            if (changed === 0) fail('filter: the second pass changed nothing, so the comparison shows nothing');
            const bad = badValues(atlas);
            if (bad.nonFinite > 0 || bad.negative > 0) fail(`filter: the atlas has ${bad.nonFinite} non-finite and ${bad.negative} negative values`);
        } finally {
            step.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }

    // 4 and 5: the corner, lit by the sky (4), then as a furnace (5).
    const bins = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5];
    const binOf = (d: number): number => bins.findIndex((edge, index) => index < bins.length - 1 && d >= edge && d < bins[index + 1]);
    const cornerPixels = (screen: Awaited<ReturnType<typeof readScreen>>): Array<{ index: number; d: number }> => {
        const pixels: Array<{ index: number; d: number }> = [];
        for (let y = 0; y < HEIGHT; y++) {
            for (let x = 0; x < WIDTH; x++) {
                if (screen.isSky(x, y)) continue;
                const normal = screen.normal(x, y);
                if (normal[1] < 0.99) continue; // floor pixels only
                const d = screen.worldPosition(x, y)[0];
                if (d >= 0 && d < bins[bins.length - 1]) pixels.push({ index: y * WIDTH + x, d });
            }
        }
        return pixels;
    };

    const corner = await makeRun(device, 'corner', CORNER, grey, constantSky(SKY), CORNER_CAMERA);
    try {
        const probes = corner.technique.probes;
        if (!probes?.occlusion) throw new Error('no SSGI after encodeScreen');
        const occlusionTexture = probes.occlusion.occlusion;
        const pixels = cornerPixels(await readScreen(device, corner));
        // The visibility and bent normal per distance from the wall, averaged over frames.
        const measure = async () => {
            const sums = bins.slice(1).map(() => ({ ao: 0, bentX: 0, bentY: 0, count: 0 }));
            let bad = { nonFinite: 0, negative: 0 };
            for (let frame = 0; frame < AVERAGE_FRAMES; frame++) {
                corner.frame();
                const occlusion = halves(await readTexture(device, occlusionTexture, 8, WIDTH, HEIGHT));
                const counts = badValues(occlusion, 4, [3]);
                bad = { nonFinite: bad.nonFinite + counts.nonFinite, negative: bad.negative + counts.negative };
                for (const { index, d } of pixels) {
                    const sum = sums[binOf(d)];
                    sum.ao += occlusion[index * 4 + 3];
                    sum.bentX += occlusion[index * 4];
                    sum.bentY += occlusion[index * 4 + 1];
                    sum.count++;
                }
            }
            const table = sums.map((sum, bin) => {
                const middle = (bins[bin] + bins[bin + 1]) / 2;
                const tilt = (Math.atan2(sum.bentX, sum.bentY) * 180) / Math.PI;
                return { dFrom: bins[bin], dTo: bins[bin + 1], samples: sum.count, meanAO: sum.ao / Math.max(sum.count, 1), analyticAO: analyticCornerVisibility(middle, RADIUS), bentTiltAwayDegrees: tilt };
            });
            return { table, bad };
        };
        // Capsaicin's defaults (one slice, two steps a side), then many slices and steps, which
        // find the wall's true horizon: the integrals are checked on the second, and the first
        // must occlude and lean away from the wall.
        const defaults = await measure();
        corner.technique.settings.ssgiSliceCount = DENSE_SSGI.sliceCount;
        corner.technique.settings.ssgiStepCount = DENSE_SSGI.stepCount;
        const dense = await measure();
        data.corner = { radius: RADIUS, frames: AVERAGE_FRAMES, analyticTiltAtTheFootDegrees: (Math.atan(2 / Math.PI) * 180) / Math.PI, defaults, dense: { ...DENSE_SSGI, ...dense } };
        const foot = (table: typeof defaults.table) => table.filter((row) => row.dTo <= RADIUS / 10 && row.samples > 0);
        if (foot(dense.table).length === 0) fail('corner: no floor pixels within R / 10 of the wall');
        for (const row of foot(dense.table)) {
            if (!(Math.abs(row.meanAO - row.analyticAO) <= CORNER_AO_TOLERANCE)) fail(`corner (dense): visibility ${row.meanAO.toFixed(3)} at d ${row.dFrom}-${row.dTo} m, analytic ${row.analyticAO.toFixed(3)}`);
            if (!(row.bentTiltAwayDegrees >= CORNER_TILT_DEGREES[0] && row.bentTiltAwayDegrees <= CORNER_TILT_DEGREES[1])) fail(`corner (dense): the bent normal tilts ${row.bentTiltAwayDegrees.toFixed(1)} degrees away from the wall at d ${row.dFrom}-${row.dTo} m`);
        }
        for (const row of foot(defaults.table)) {
            if (!(row.meanAO >= row.analyticAO - CORNER_AO_TOLERANCE && row.meanAO <= CORNER_DEFAULT_MAX_AO)) fail(`corner (defaults): visibility ${row.meanAO.toFixed(3)} at d ${row.dFrom}-${row.dTo} m, analytic ${row.analyticAO.toFixed(3)}`);
            if (!(row.bentTiltAwayDegrees >= CORNER_DEFAULT_MIN_TILT)) fail(`corner (defaults): the bent normal tilts ${row.bentTiltAwayDegrees.toFixed(1)} degrees away from the wall at d ${row.dFrom}-${row.dTo} m`);
        }
        for (const { table, bad } of [defaults, dense]) {
            for (const row of table.filter((entry) => entry.dFrom >= 1.5 * RADIUS && entry.samples > 0)) {
                if (!(row.meanAO >= PLANE_MEAN_AO)) fail(`corner: visibility ${row.meanAO.toFixed(3)} beyond the radius (d ${row.dFrom}-${row.dTo} m)`);
            }
            if (bad.nonFinite > 0 || bad.negative > 0) fail(`corner: occlusion has ${bad.nonFinite} non-finite and ${bad.negative} negative values`);
        }
    } finally {
        corner.dispose();
        await device.queue.onSubmittedWorkDone();
    }

    const emitter = 1;
    const furnace = await makeRun(device, 'furnace', CORNER, { albedo: [0, 0, 0], emissive: [emitter, emitter, emitter], emissiveIntensity: 1 }, constantSky(emitter), CORNER_CAMERA);
    try {
        const probes = furnace.technique.probes;
        if (!probes?.occlusion) throw new Error('no SSGI after encodeScreen');
        const pixels = cornerPixels(await readScreen(device, furnace));
        const sums = bins.slice(1).map(() => ({ e: 0, capsaicin: 0, near: 0, count: 0 }));
        let bad = { nonFinite: 0, negative: 0 };
        for (let frame = 0; frame < AVERAGE_FRAMES; frame++) {
            furnace.frame();
            const irradiance = halves(await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT));
            const nearField = halves(await readTexture(device, probes.occlusion.nearField, 8, WIDTH, HEIGHT));
            const counts = badValues(irradiance);
            bad = { nonFinite: bad.nonFinite + counts.nonFinite, negative: bad.negative + counts.negative };
            for (const { index, d } of pixels) {
                const sum = sums[binOf(d)];
                const e = (irradiance[index * 4] + irradiance[index * 4 + 1] + irradiance[index * 4 + 2]) / 3;
                const near = (nearField[index * 4] + nearField[index * 4 + 1] + nearField[index * 4 + 2]) / 3;
                sum.e += e;
                sum.near += near;
                // Capsaicin adds the near-field light as E / pi (register D52).
                sum.capsaicin += e - near + near / Math.PI;
                sum.count++;
            }
        }
        const piL = Math.PI * emitter;
        const table = sums.map((sum, bin) => ({
            dFrom: bins[bin],
            dTo: bins[bin + 1],
            samples: sum.count,
            eOverPiL: sum.e / Math.max(sum.count, 1) / piL,
            nearFieldOverPiL: sum.near / Math.max(sum.count, 1) / piL,
            capsaicinUnitsOverPiL: sum.capsaicin / Math.max(sum.count, 1) / piL,
        }));
        data.furnace = { radiance: emitter, frames: AVERAGE_FRAMES, bins: table, ...bad };
        for (const row of table) {
            if (row.samples === 0) continue;
            const tolerance = row.dTo <= RADIUS ? FURNACE_NEAR_TOLERANCE : row.dFrom >= 2 * RADIUS ? FURNACE_FAR_TOLERANCE : null;
            if (tolerance !== null && !(Math.abs(row.eOverPiL - 1) <= tolerance)) fail(`furnace: E / pi L ${row.eOverPiL.toFixed(4)} at d ${row.dFrom}-${row.dTo} m`);
        }
        if (bad.nonFinite > 0 || bad.negative > 0) fail(`furnace: irradiance has ${bad.nonFinite} non-finite and ${bad.negative} negative values`);
    } finally {
        furnace.dispose();
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
