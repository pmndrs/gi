/**
 * RC4.2: `FilterScreenProbes` on the GPU (packages/ddgi-core/src/techniques/
 * radiance-cache/wgsl/screen-irradiance.ts), checked against its CPU twin
 * (probe-filter.ts), and the filtered atlas kept as the history the next
 * frame reprojects. 1280 × 720, the engine-free page screen
 * (radiance-cache-probes/screen.ts), on a device held at the platform's 8
 * storage buffers per stage.
 *
 * The scene is the pillar (radiance-cache-probes/scenes.ts): a floor under a
 * constant sky and a 2.5 m box on it, seen past the box's edge, so probes on
 * the floor have probes on the box beside them in the screen with a jump in
 * depth.
 *
 * 1. **A captured atlas against the twin:** after a few frames with history,
 *    the frame's (filtered) atlas is written back and filtered again on its
 *    own (`encodeProbeFilter`: Capsaicin's copy, then both passes); every
 *    listed probe's 64 cells against the twin's two passes on the same atlas,
 *    probes, mask mips and screen: within three half steps of the value
 *    (the first pass's results are stored as halves, rounded either way,
 *    before the second reads them). Cells where a test sits at its threshold
 *    are counted, not compared.
 * 2. **A constant atlas stays constant:** every cell (0.75, 0.5, 0.25) at a
 *    hit distance of 40 m, filtered: every listed cell's radiance reads the
 *    same within a half step (the weights are normalised), its hit distance
 *    within two. The hit distance can only drift down: each neighbour takes
 *    the min of its own and the running mean, so an f32 rounding below 40 is
 *    kept and never one above, and the M5's `textureStore` to `rgba16float`
 *    rounds toward zero (NOTES), once per pass (first run: 8,137 of 156,160
 *    cells at 39.9375, two steps down; the twin, rounding to nearest, keeps 40).
 * 3. **No blur across a depth edge:** the box's probes at 10, the floor's at
 *    1, filtered: against the twin as in 1, and the floor's cells with a box
 *    probe among their six neighbours each way keep 1 within a half step
 *    unless the twin let a box probe in (counted).
 * 4. **The history is the filtered atlas:** after a frame the previous atlas
 *    (what the next frame reprojects) equals the atlas the filter left, bit
 *    for bit; with the camera still, the next frame's reprojected probes (in
 *    the mask, not re-spawned) keep those filtered cells within the
 *    reprojection's quantization (×1e4) and two half steps.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-filter`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { cameraRelativeInverseViewProjection } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/camera.ts';
import { floatToHalfBits, INVALID_ID, unpackSeed } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { filterScreenProbes, toHalf, type ProbeAtlas, type ProbeFilterInputs } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/probe-filter.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { screenProbeCellSize, viewForward } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/screen-irradiance.ts';
import { PROBE_CELLS, PROBE_SIZE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-probes.ts';
import { LARGEST_QUANTIZED_HIT_DISTANCE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/capsaicin.ts';
import { FILTER_RADIUS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-irradiance.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { readBuffer } from '../ddgi-shared/harness.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { describePillar } from '../radiance-cache-probes/scenes.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const TILES_X = Math.ceil(WIDTH / PROBE_SIZE);
const TILES_Y = Math.ceil(HEIGHT / PROBE_SIZE);
const CAMERA: PageCamera = { position: [-1.2, 1.6, 0], target: [-0.2, 0.6, -5], fovYDeg: 60, near: 0.1, far: 1000 };
/** Frames with history before the checks: past the Halton cycle (every tile has had a probe by frame 8). */
const WARM_FRAMES = 10;
/** Check 1's tolerance: three half steps of the value (2^-10 each), and a floor for values near 0. */
const HALF_STEPS = 3;
const FLOOR_TOLERANCE = 1e-6;
/** Check 2's constant atlas, and the hit distance's allowance in half steps (the file comment). */
const CONSTANT_HIT_DISTANCE_STEPS = 2;
const CONSTANT_RADIANCE = [0.75, 0.5, 0.25] as const;
const CONSTANT_HIT_DISTANCE = 40;
/** Check 3's values. */
const BOX_VALUE = 10;
const FLOOR_VALUE = 1;
/** Check 4: the capped hit distance (register D48) and the half step there. */
const CAPPED_HIT_DISTANCE = LARGEST_QUANTIZED_HIT_DISTANCE / 1e4;
const CAPPED_HIT_DISTANCE_TOLERANCE = 4.01;

const ATLAS_WIDTH = TILES_X * PROBE_SIZE;
const ATLAS_HEIGHT = TILES_Y * PROBE_SIZE;

interface Run {
    readonly platform: GIPlatformImpl;
    readonly technique: RadianceCacheTechnique;
    readonly screen: PageScreen;
    frame(): Promise<void>;
}

function frameInput(): FrameInput {
    return { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(CAMERA, WIDTH, HEIGHT) };
}

async function readAtlas(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
    return halves(await readTexture(device, texture, 8, ATLAS_WIDTH, ATLAS_HEIGHT));
}

/** Writes an atlas (four values a texel) into the technique's atlas as halves. */
function writeAtlas(device: GPUDevice, texture: GPUTexture, texels: Float32Array): void {
    const bits = new Uint16Array(texels.length);
    for (let index = 0; index < texels.length; index++) bits[index] = floatToHalfBits(texels[index]);
    device.queue.writeTexture({ texture }, bits, { bytesPerRow: ATLAS_WIDTH * 8 }, [ATLAS_WIDTH, ATLAS_HEIGHT]);
}

/** What the twin needs, read back after a frame. */
async function readInputs(device: GPUDevice, run: Run): Promise<ProbeFilterInputs> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const { probeCount, probeMaskMipCount, maxProbeSpawnCount } = probes.layout;
    const maskLevels: { width: number; height: number; texels: Uint32Array }[] = [];
    for (let level = 0; level < probeMaskMipCount; level++) {
        const width = Math.max(1, probeCount[0] >> level);
        const height = Math.max(1, probeCount[1] >> level);
        const bytes = await readTexture(device, probes.probeMask, 4, width, height, level);
        maskLevels.push({ width, height, texels: new Uint32Array(bytes.buffer, bytes.byteOffset, width * height) });
    }
    const spawn = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + maxProbeSpawnCount * 4));
    const depthBytes = await readTexture(device, run.screen.screen.depth, 4, WIDTH, HEIGHT);
    const camera = frameCamera(CAMERA, WIDTH, HEIGHT);
    return {
        width: WIDTH,
        height: HEIGHT,
        depth: new Float32Array(depthBytes.buffer, depthBytes.byteOffset, WIDTH * HEIGHT),
        normal: halves(await readTexture(device, run.screen.screen.normal, 8, WIDTH, HEIGHT)),
        maskLevels,
        seeds: Array.from(spawn.subarray(1, 1 + spawn[0])),
        inverseViewProjection: cameraRelativeInverseViewProjection(camera),
        eye: camera.position,
        viewForward: viewForward(camera),
        cellSize: screenProbeCellSize(camera, WIDTH, HEIGHT),
    };
}

/** The atlas texel of cell `cell` of the probe whose seed is `seed`. */
function cellTexel(seed: number, cell: number): number {
    const [x, y] = unpackSeed(seed);
    const px = Math.floor(x / PROBE_SIZE) * PROBE_SIZE + (cell % PROBE_SIZE);
    const py = Math.floor(y / PROBE_SIZE) * PROBE_SIZE + Math.floor(cell / PROBE_SIZE);
    return px + py * ATLAS_WIDTH;
}

const halfStep = (value: number): number => Math.abs(value) * 2 ** -10;

interface Comparison {
    readonly cells: number;
    readonly undecided: number;
    readonly mismatches: number;
    readonly worstRelative: number;
    readonly accepted: readonly [number, number];
    readonly examples: string[];
}

/** Filters `atlas` on the GPU (on the latest frame's inputs) and against the twin; returns the GPU's result and the comparison. */
async function filterAndCompare(device: GPUDevice, run: Run, inputs: ProbeFilterInputs, atlas: Float32Array): Promise<{ gpu: Float32Array; twin: ProbeAtlas; comparison: Comparison }> {
    const probes = run.technique.probes;
    if (!probes) throw new Error('no probes');
    writeAtlas(device, probes.probeAtlas, atlas);
    // The twin reads what the GPU reads: the halves written.
    const written = Float32Array.from(atlas, (value) => toHalf(value));
    const encoder = device.createCommandEncoder({ label: 'radiance-cache-filter filter' });
    run.technique.encodeProbeFilter(encoder);
    device.queue.submit([encoder.finish()]);
    const gpu = await readAtlas(device, probes.probeAtlas);
    const result = filterScreenProbes(inputs, { width: ATLAS_WIDTH, height: ATLAS_HEIGHT, texels: written });
    let cells = 0;
    let undecided = 0;
    let mismatches = 0;
    let worstRelative = 0;
    const examples: string[] = [];
    for (const seed of inputs.seeds) {
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const texel = cellTexel(seed, cell);
            if (result.undecided[texel] === 1) {
                undecided++;
                continue;
            }
            cells++;
            for (let channel = 0; channel < 4; channel++) {
                const expected = result.atlas.texels[texel * 4 + channel];
                const actual = gpu[texel * 4 + channel];
                const error = Math.abs(actual - expected);
                worstRelative = Math.max(worstRelative, error / Math.max(Math.abs(expected), 1e-6));
                if (!(error <= HALF_STEPS * halfStep(expected) + FLOOR_TOLERANCE)) {
                    mismatches++;
                    if (examples.length < 4) examples.push(`seed ${unpackSeed(seed).join(',')} cell ${cell} channel ${channel}: GPU ${actual}, twin ${expected}`);
                }
            }
        }
    }
    return { gpu, twin: result.atlas, comparison: { cells, undecided, mismatches, worstRelative, accepted: result.accepted, examples } };
}


const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const { device, adapterStorageBuffersPerStage } = await requestBudgetDevice(adapter);
    window.addEventListener('pagehide', () => device.destroy(), { once: true });
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 30) failures.push(message);
    };

    const platform = new GIPlatformImpl(device, { label: 'radiance-cache-filter' });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    await describePillar(platform);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const run: Run = {
        platform,
        technique,
        screen,
        async frame() {
            const encoder = device.createCommandEncoder({ label: 'radiance-cache-filter' });
            platform.encode(encoder, frameInput());
            screen.encode(encoder, CAMERA);
            platform.encodeScreen(encoder, screen.screen);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        },
    };
    await run.frame();
    await platform.rays.whenBuilt();
    technique.resetHistory();
    for (let frame = 0; frame < WARM_FRAMES; frame++) await run.frame();

    // 4 first, on the frame just run: the history is the filtered atlas.
    const probes = technique.probes;
    const history = technique.probeHistory;
    if (!probes || !history) throw new Error('no probes after the warm-up');
    const filtered = await readAtlas(device, probes.probeAtlas);
    const previous = await readAtlas(device, history.previousProbeAtlas);
    let historyDifferences = 0;
    for (let index = 0; index < filtered.length; index++) if (filtered[index] !== previous[index] && !(Number.isNaN(filtered[index]) && Number.isNaN(previous[index]))) historyDifferences++;
    if (historyDifferences > 0) fail(`history: the previous atlas differs from the filtered atlas in ${historyDifferences} values`);
    const maskBefore = new Uint32Array((await readTexture(device, probes.probeMask, 4, TILES_X, TILES_Y)).buffer);
    await run.frame();
    const nextAtlas = await readAtlas(device, probes.probeAtlas);
    const nextMask = new Uint32Array((await readTexture(device, probes.probeMask, 4, TILES_X, TILES_Y)).buffer);
    const nextSpawn = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + probes.layout.maxProbeSpawnCount * 4));
    const listedTiles = new Set<number>();
    for (let probe = 0; probe < nextSpawn[0]; probe++) {
        const [x, y] = unpackSeed(nextSpawn[1 + probe]);
        listedTiles.add(Math.floor(y / PROBE_SIZE) * TILES_X + Math.floor(x / PROBE_SIZE));
    }
    let reprojectedTiles = 0;
    let reprojectedMismatches = 0;
    for (let tile = 0; tile < TILES_X * TILES_Y; tile++) {
        if (listedTiles.has(tile) || nextMask[tile] === INVALID_ID || nextMask[tile] !== maskBefore[tile]) continue;
        reprojectedTiles++;
        const tx = tile % TILES_X;
        const ty = Math.floor(tile / TILES_X);
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const texel = (ty * PROBE_SIZE + Math.floor(cell / PROBE_SIZE)) * ATLAS_WIDTH + tx * PROBE_SIZE + (cell % PROBE_SIZE);
            for (let channel = 0; channel < 4; channel++) {
                const a = channel === 3 ? Math.min(filtered[texel * 4 + channel], CAPPED_HIT_DISTANCE) : filtered[texel * 4 + channel];
                const b = nextAtlas[texel * 4 + channel];
                const allowed = 1e-4 + Math.abs(a) * 2 ** -9 + (channel === 3 ? CAPPED_HIT_DISTANCE_TOLERANCE : 0);
                if (Math.abs(a - b) > allowed) reprojectedMismatches++;
            }
        }
    }
    if (reprojectedTiles === 0) fail('history: no probe was reprojected');
    if (reprojectedMismatches > 0) fail(`history: ${reprojectedMismatches} values of reprojected probes are not the filtered atlas's`);

    // The inputs of the frame just run, for checks 1-3 (each filters on them again).
    const inputs = await readInputs(device, run);

    // 1. The frame's own (filtered) atlas, filtered again, against the twin.
    const captured = await readAtlas(device, probes.probeAtlas);
    const capturedCheck = await filterAndCompare(device, run, inputs, captured);
    if (capturedCheck.comparison.mismatches > 0) fail(`captured atlas: ${capturedCheck.comparison.mismatches} values differ from the twin: ${capturedCheck.comparison.examples.join('; ')}`);
    if (capturedCheck.comparison.accepted[0] === 0 || capturedCheck.comparison.accepted[1] === 0) fail('captured atlas: the twin accepted no neighbour in a pass');
    if (capturedCheck.comparison.cells < inputs.seeds.length * PROBE_CELLS * 0.9) fail(`captured atlas: only ${capturedCheck.comparison.cells} cells compared (${capturedCheck.comparison.undecided} undecided)`);

    // 2. A constant atlas.
    const constant = new Float32Array(ATLAS_WIDTH * ATLAS_HEIGHT * 4);
    for (let texel = 0; texel < ATLAS_WIDTH * ATLAS_HEIGHT; texel++) constant.set([...CONSTANT_RADIANCE, CONSTANT_HIT_DISTANCE], texel * 4);
    const constantCheck = await filterAndCompare(device, run, inputs, constant);
    let constantChanged = 0;
    let constantCells = 0;
    const constantExamples: string[] = [];
    let constantHitDistancesDown = 0;
    for (const seed of inputs.seeds) {
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const texel = cellTexel(seed, cell);
            constantCells++;
            if (constantCheck.gpu[texel * 4 + 3] < CONSTANT_HIT_DISTANCE) constantHitDistancesDown++;
            const expected = [...CONSTANT_RADIANCE, CONSTANT_HIT_DISTANCE];
            if (expected.some((value, channel) => Math.abs(constantCheck.gpu[texel * 4 + channel] - value) > (channel === 3 ? CONSTANT_HIT_DISTANCE_STEPS : 1) * halfStep(value))) {
                constantChanged++;
                if (constantExamples.length < 4) constantExamples.push(`seed ${unpackSeed(seed).join(',')} cell ${cell}: ${Array.from(constantCheck.gpu.subarray(texel * 4, texel * 4 + 4)).join(', ')} (twin ${Array.from(constantCheck.twin.texels.subarray(texel * 4, texel * 4 + 4)).join(', ')})`);
            }
        }
    }
    if (constantChanged > 0) fail(`constant atlas: ${constantChanged} of ${constantCells} cells changed by more than a half step (hit distance: ${CONSTANT_HIT_DISTANCE_STEPS}): ${constantExamples.join('; ')}`);
    if (constantCheck.comparison.mismatches > 0) fail(`constant atlas: ${constantCheck.comparison.mismatches} values differ from the twin`);

    // 3. The box against the floor: a probe is the box's if its seed's normal isn't straight up.
    const onBox = (seed: number): boolean => {
        const [x, y] = unpackSeed(seed);
        return inputs.normal[(x + y * WIDTH) * 4 + 1] < 0.99;
    };
    const boxTiles = new Set<number>();
    for (const seed of inputs.seeds) if (onBox(seed)) boxTiles.add(Math.floor(unpackSeed(seed)[1] / PROBE_SIZE) * TILES_X + Math.floor(unpackSeed(seed)[0] / PROBE_SIZE));
    const edgeAtlas = new Float32Array(ATLAS_WIDTH * ATLAS_HEIGHT * 4);
    for (let texel = 0; texel < ATLAS_WIDTH * ATLAS_HEIGHT; texel++) {
        const tile = Math.floor(Math.floor(texel / ATLAS_WIDTH) / PROBE_SIZE) * TILES_X + Math.floor((texel % ATLAS_WIDTH) / PROBE_SIZE);
        const value = boxTiles.has(tile) ? BOX_VALUE : FLOOR_VALUE;
        edgeAtlas.set([value, value, value, CONSTANT_HIT_DISTANCE], texel * 4);
    }
    const edgeCheck = await filterAndCompare(device, run, inputs, edgeAtlas);
    if (edgeCheck.comparison.mismatches > 0) fail(`depth edge: ${edgeCheck.comparison.mismatches} values differ from the twin: ${edgeCheck.comparison.examples.join('; ')}`);
    let besideBox = 0;
    let keptByGpu = 0;
    let keptByTwin = 0;
    let keptDisagree = 0;
    for (const seed of inputs.seeds) {
        if (onBox(seed)) continue;
        const [x, y] = unpackSeed(seed);
        const tx = Math.floor(x / PROBE_SIZE);
        const ty = Math.floor(y / PROBE_SIZE);
        let nearBox = false;
        for (let step = -FILTER_RADIUS; step <= FILTER_RADIUS; step++) {
            if (boxTiles.has(ty * TILES_X + tx + step) || boxTiles.has((ty + step) * TILES_X + tx)) nearBox = true;
        }
        if (!nearBox) continue;
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const texel = cellTexel(seed, cell);
            besideBox++;
            const gpuKept = Math.abs(edgeCheck.gpu[texel * 4] - FLOOR_VALUE) <= halfStep(FLOOR_VALUE);
            const twinKept = Math.abs(edgeCheck.twin.texels[texel * 4] - FLOOR_VALUE) <= halfStep(FLOOR_VALUE);
            if (gpuKept) keptByGpu++;
            if (twinKept) keptByTwin++;
            if (gpuKept !== twinKept) keptDisagree++;
        }
    }
    if (besideBox === 0) fail('depth edge: no floor probe beside the box');
    if (keptDisagree > 0) fail(`depth edge: ${keptDisagree} floor cells beside the box kept their value on one side only`);
    if (keptByGpu < besideBox / 2) fail(`depth edge: only ${keptByGpu} of ${besideBox} floor cells beside the box kept their value`);

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
            probes: inputs.seeds.length,
            history: { historyDifferences, reprojectedTiles, reprojectedMismatches },
            captured: capturedCheck.comparison,
            constant: { cells: constantCells, changed: constantChanged, hitDistancesBelow: constantHitDistancesDown, examples: constantExamples, twin: constantCheck.comparison },
            depthEdge: { boxProbes: boxTiles.size, floorCellsBesideBox: besideBox, keptByGpu, keptByTwin, keptDisagree, twin: edgeCheck.comparison },
        },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
