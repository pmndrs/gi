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
 * Checks 1-3 and 6 run the technique's own filter class
 * (`ScreenProbeFilter`, screen-probe-filter.ts) on atlases of the page's own,
 * with the latest frame's probes, mask and screen and uniforms the page
 * writes by their named offsets: nothing in the technique is there for tests.
 *
 * 1. **A captured atlas against the twin:** after a few frames with history,
 *    the frame's (filtered) atlas is copied and filtered again on its own
 *    (Capsaicin's copy into the second atlas, then both passes); every
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
 *    kept and never one above. Before register D59's fix the M5's
 *    `textureStore` to `rgba16float` rounded that toward zero (NOTES), once per
 *    pass (first run: 8,137 of 156,160 cells at 39.9375, two steps down); the
 *    kernel now rounds to the nearest half first, as the twin does.
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
 * 5. **How `textureStore` rounds to `rgba16float`** (register D59): 4,096
 *    f32 values that aren't halves, stored by a small kernel and read back,
 *    against round-to-nearest-even and round-toward-zero (reported: the M5
 *    rounds toward zero). The same values stored through the atlas kernels'
 *    `rcNearestHalf` (wgsl/capsaicin.ts, the production function) must all
 *    come back as round-to-nearest-even, ties included.
 * 6. **Does the hit distance drift?** (register D59): the constant atlas
 *    filtered 300 times over (each pass's store feeding the next, with no new
 *    rays: the worst case), and the frame's own loop over 300 still frames,
 *    where the blend brings new rays in: the sky cells' hit distance at
 *    frames 10, 100 and 300. Checked: the iterated filter's 40 m stays within
 *    1 m (every listed cell, at rounds 1, 10, 100 and 300; RC6.2's
 *    done-when; before the fix it fell to 29.7 m), and in the frame's loop
 *    the sky cells fall no more than three half steps (12 m) from frame 100
 *    to frame 300. (Before the fix the loop fell 27 m by frame 100, 0.4%,
 *    then 2.8 m more; the fall is reported.)
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-filter`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { cameraRelativeInverseViewProjection } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/camera.ts';
import { halfBitsToFloat } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { floatToHalfBits, INVALID_ID, unpackSeed } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { filterScreenProbes, toHalf, type ProbeAtlas, type ProbeFilterInputs } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/probe-filter.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { screenProbeCellSize, viewForward } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/screen-irradiance.ts';
import { PROBE_CELLS, PROBE_SIZE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { FRAME_UNIFORM_BYTES, FRAME_UNIFORM_OFFSETS, PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-probes.ts';
import { SCREEN_PROBES_CONSTANTS_BYTES, SCREEN_PROBES_CONSTANTS_OFFSETS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-irradiance.ts';
import { ScreenProbeFilter } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/screen-probe-filter.ts';
import { CAPSAICIN_WGSL, LARGEST_QUANTIZED_HIT_DISTANCE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/capsaicin.ts';
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

/** Check 6: frames (and filter rounds) of the drift measurement, and how far the frame loop's sky hit distance may still fall after frame 100: three half steps at 6.7 km. */
const DRIFT_FRAMES = 300;
const DRIFT_TOLERANCE = 12.01;
/** Check 6: how far the iterated filter's constant 40 m may fall in any listed cell (RC6.2's done-when). */
const ITERATED_DRIFT_TOLERANCE = 1;
const SKY_HIT_RANGE = [6000, 7000] as const;
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

/**
 * The technique's filter class on the page's own atlases (the file comment):
 * the latest frame's probes, mask and screen, uniforms written by name.
 */
class PageFilter {
    readonly atlas: GPUTexture;
    private readonly secondAtlas: GPUTexture;
    private readonly frameUniforms: GPUBuffer;
    private readonly screenConstants: GPUBuffer;

    private constructor(
        private readonly device: GPUDevice,
        private readonly filter: ScreenProbeFilter,
        private readonly run: Run,
    ) {
        const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
        this.atlas = device.createTexture({ label: 'page atlas', size: [ATLAS_WIDTH, ATLAS_HEIGHT], format: 'rgba16float', usage });
        this.secondAtlas = device.createTexture({ label: 'page second atlas', size: [ATLAS_WIDTH, ATLAS_HEIGHT], format: 'rgba16float', usage });
        this.frameUniforms = device.createBuffer({ label: 'page RcFrame', size: FRAME_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.screenConstants = device.createBuffer({ label: 'page RcScreenProbesConstants', size: SCREEN_PROBES_CONSTANTS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const camera = frameCamera(CAMERA, WIDTH, HEIGHT);
        // RcFrame: the fields the filter reads (the screen is the page's: signed normals, depth 0 near).
        const frame = new ArrayBuffer(FRAME_UNIFORM_BYTES);
        new Float32Array(frame).set(cameraRelativeInverseViewProjection(camera), FRAME_UNIFORM_OFFSETS.inverseViewProjection / 4);
        new Float32Array(frame).set(camera.position, FRAME_UNIFORM_OFFSETS.eye / 4);
        new Uint32Array(frame).set([WIDTH, HEIGHT], FRAME_UNIFORM_OFFSETS.bufferDimensions / 4);
        device.queue.writeBuffer(this.frameUniforms, 0, frame);
        const constants = new ArrayBuffer(SCREEN_PROBES_CONSTANTS_BYTES);
        new Float32Array(constants).set(viewForward(camera), SCREEN_PROBES_CONSTANTS_OFFSETS.viewForward / 4);
        new Float32Array(constants)[SCREEN_PROBES_CONSTANTS_OFFSETS.cellSize / 4] = screenProbeCellSize(camera, WIDTH, HEIGHT);
        const layout = run.technique.probes?.layout;
        if (!layout) throw new Error('no probes before the page filter');
        new Uint32Array(constants)[SCREEN_PROBES_CONSTANTS_OFFSETS.probeMaskMipCount / 4] = layout.probeMaskMipCount;
        device.queue.writeBuffer(this.screenConstants, 0, constants);
    }

    static async create(device: GPUDevice, run: Run): Promise<PageFilter> {
        return new PageFilter(device, await ScreenProbeFilter.create({ device, kernels: run.platform.kernels }), run);
    }

    /** Records `times` rounds of Capsaicin's copy and both passes on the page's atlas. */
    encode(times = 1): void {
        const probes = this.run.technique.probes;
        if (!probes) throw new Error('no probes');
        const encoder = this.device.createCommandEncoder({ label: 'radiance-cache-filter page filter' });
        for (let round = 0; round < times; round++) {
            encoder.copyTextureToTexture({ texture: this.atlas }, { texture: this.secondAtlas }, [ATLAS_WIDTH, ATLAS_HEIGHT]);
            const pass = encoder.beginComputePass({ label: 'radiance-cache-filter page filter' });
            this.filter.record(pass, {
                layout: probes.layout,
                frameUniforms: this.frameUniforms,
                screenConstants: this.screenConstants,
                probeSpawn: probes.probeSpawn,
                probeMask: probes.probeMask,
                depth: this.run.screen.screen.depth,
                normal: this.run.screen.screen.normal,
                atlas: this.atlas,
                secondAtlas: this.secondAtlas,
            });
            pass.end();
        }
        this.device.queue.submit([encoder.finish()]);
    }

    dispose(): void {
        this.filter.dispose();
        for (const resource of [this.atlas, this.secondAtlas, this.frameUniforms, this.screenConstants]) resource.destroy();
    }
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
async function filterAndCompare(device: GPUDevice, filter: PageFilter, inputs: ProbeFilterInputs, atlas: Float32Array): Promise<{ gpu: Float32Array; twin: ProbeAtlas; comparison: Comparison }> {
    writeAtlas(device, filter.atlas, atlas);
    // The twin reads what the GPU reads: the halves written.
    const written = Float32Array.from(atlas, (value) => toHalf(value));
    filter.encode();
    const gpu = await readAtlas(device, filter.atlas);
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


// ---- Checks 5 and 6: the half-float stores and the hit distance's drift ----------------------

// Each value stored as it is (the store's own rounding) and through the atlas kernels' rcNearestHalf.
const STORE_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var stored: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var storedNearest: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let width = textureDimensions(stored).x;
    if (id.x >= arrayLength(&values)) {
        return;
    }
    let texel = vec2u(id.x % width, id.x / width);
    let value = vec4f(values[id.x], 0.0, 0.0, 1.0);
    textureStore(stored, texel, value);
    textureStore(storedNearest, texel, rcNearestHalf(value));
}
`;

/** Values that aren't halves, spread over the half range's exponents, ties included. */
function storeValues(count: number): Float32Array<ArrayBuffer> {
    const values = new Float32Array(count);
    let state = 12345;
    const random = (): number => {
        state = (Math.imul(state, 1103515245) + 12345) >>> 0;
        return state / 2 ** 32;
    };
    for (let index = 0; index < count; index++) {
        const exponent = Math.floor(random() * 30) - 14;
        // Every eighth a tie: halfway between two halves.
        const mantissa = index % 8 === 0 ? (Math.floor(random() * 1024) + 0.5) / 1024 : random();
        values[index] = (1 + mantissa) * 2 ** exponent;
    }
    return values;
}

/** Round-toward-zero half bits of a positive f32 (the round-to-nearest-even bits, stepped down when they rounded up). */
function towardZeroHalfBits(value: number): number {
    const nearest = floatToHalfBits(value);
    return halfBitsToFloat(nearest) > value ? nearest - 1 : nearest;
}

interface StoreRounding {
    readonly values: number;
    readonly nearestEven: number;
    readonly towardZero: number;
    readonly both: number;
    readonly neither: number;
}

/** How each stored value came back: the nearest-even half, the one toward zero, both (no difference) or neither. */
function classifyStores(values: Float32Array, bits: Uint16Array): StoreRounding {
    let nearestEven = 0;
    let towardZero = 0;
    let both = 0;
    let neither = 0;
    values.forEach((value, index) => {
        const got = bits[index * 4];
        const isNearest = got === floatToHalfBits(value);
        const isTowardZero = got === towardZeroHalfBits(value);
        if (isNearest && isTowardZero) both++;
        else if (isNearest) nearestEven++;
        else if (isTowardZero) towardZero++;
        else neither++;
    });
    return { values: values.length, nearestEven, towardZero, both, neither };
}

async function checkStoreRounding(device: GPUDevice): Promise<{ plain: StoreRounding; throughNearestHalf: StoreRounding }> {
    const side = 64;
    const values = storeValues(side * side);
    const input = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(input, 0, values);
    const makeTexture = (): GPUTexture => device.createTexture({ size: [side, side], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const plainTexture = makeTexture();
    const nearestTexture = makeTexture();
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: STORE_WGSL }), entryPoint: 'main' } });
    const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: input } },
            { binding: 1, resource: plainTexture.createView() },
            { binding: 2, resource: nearestTexture.createView() },
        ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(values.length / 64);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const readBits = async (texture: GPUTexture): Promise<Uint16Array> => {
        const bytes = await readTexture(device, texture, 8, side, side);
        return new Uint16Array(bytes.buffer, bytes.byteOffset, side * side * 4);
    };
    const plain = classifyStores(values, await readBits(plainTexture));
    const throughNearestHalf = classifyStores(values, await readBits(nearestTexture));
    input.destroy();
    plainTexture.destroy();
    nearestTexture.destroy();
    return { plain, throughNearestHalf };
}

/**
 * The listed probes' cells holding a sky hit distance (6,000 to 7,000 m: the capped 6,710.9 m less
 * any drift, leaving out the 65,504 m an empty cell holds): their mean and lowest hit distance, and
 * mean radiance.
 */
function skyCells(atlas: Float32Array, seeds: readonly number[]): { cells: number; meanHitDistance: number; lowestHitDistance: number; meanRadiance: number } {
    let cells = 0;
    let distance = 0;
    let lowest = Infinity;
    let radiance = 0;
    for (const seed of seeds) {
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const texel = cellTexel(seed, cell);
            const hit = atlas[texel * 4 + 3];
            if (!(hit > SKY_HIT_RANGE[0] && hit < SKY_HIT_RANGE[1])) continue;
            cells++;
            distance += hit;
            lowest = Math.min(lowest, hit);
            radiance += atlas[texel * 4];
        }
    }
    return { cells, meanHitDistance: distance / Math.max(cells, 1), lowestHitDistance: lowest, meanRadiance: radiance / Math.max(cells, 1) };
}

/** The listed probes' cells: mean and lowest hit distance, mean radiance. */
function listedCells(atlas: Float32Array, seeds: readonly number[]): { meanHitDistance: number; lowestHitDistance: number; meanRadiance: number } {
    let distance = 0;
    let lowest = Infinity;
    let radiance = 0;
    let cells = 0;
    for (const seed of seeds) {
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const texel = cellTexel(seed, cell);
            distance += atlas[texel * 4 + 3];
            lowest = Math.min(lowest, atlas[texel * 4 + 3]);
            radiance += atlas[texel * 4];
            cells++;
        }
    }
    return { meanHitDistance: distance / cells, lowestHitDistance: lowest, meanRadiance: radiance / cells };
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
    const pageFilter = await PageFilter.create(device, run);

    // 1. The frame's own (filtered) atlas, filtered again, against the twin.
    const captured = await readAtlas(device, probes.probeAtlas);
    const capturedCheck = await filterAndCompare(device, pageFilter, inputs, captured);
    if (capturedCheck.comparison.mismatches > 0) fail(`captured atlas: ${capturedCheck.comparison.mismatches} values differ from the twin: ${capturedCheck.comparison.examples.join('; ')}`);
    if (capturedCheck.comparison.accepted[0] === 0 || capturedCheck.comparison.accepted[1] === 0) fail('captured atlas: the twin accepted no neighbour in a pass');
    if (capturedCheck.comparison.cells < inputs.seeds.length * PROBE_CELLS * 0.9) fail(`captured atlas: only ${capturedCheck.comparison.cells} cells compared (${capturedCheck.comparison.undecided} undecided)`);

    // 2. A constant atlas.
    const constant = new Float32Array(ATLAS_WIDTH * ATLAS_HEIGHT * 4);
    for (let texel = 0; texel < ATLAS_WIDTH * ATLAS_HEIGHT; texel++) constant.set([...CONSTANT_RADIANCE, CONSTANT_HIT_DISTANCE], texel * 4);
    const constantCheck = await filterAndCompare(device, pageFilter, inputs, constant);
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
    const edgeCheck = await filterAndCompare(device, pageFilter, inputs, edgeAtlas);
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

    // 6a. The constant atlas filtered 300 times over (the worst case: no new rays).
    writeAtlas(device, pageFilter.atlas, constant);
    const iterated: Record<string, unknown>[] = [];
    let rounds = 0;
    for (const target of [1, 10, 100, DRIFT_FRAMES]) {
        pageFilter.encode(target - rounds);
        rounds = target;
        const cells = listedCells(await readAtlas(device, pageFilter.atlas), inputs.seeds);
        iterated.push({ rounds, ...cells });
        if (!(cells.lowestHitDistance >= CONSTANT_HIT_DISTANCE - ITERATED_DRIFT_TOLERANCE)) {
            fail(`drift: after ${rounds} filter rounds a listed cell's hit distance is ${cells.lowestHitDistance} (constant ${CONSTANT_HIT_DISTANCE})`);
        }
    }
    pageFilter.dispose();

    // 5. How the M5 stores halves to a texture.
    const storeRounding = await checkStoreRounding(device);
    const nearestStores = storeRounding.throughNearestHalf;
    if (nearestStores.nearestEven + nearestStores.both !== nearestStores.values) {
        fail(`stores: ${nearestStores.values - nearestStores.nearestEven - nearestStores.both} of ${nearestStores.values} values stored through rcNearestHalf aren't the nearest-even half (${nearestStores.towardZero} toward zero)`);
    }

    // 6b. The frame's own loop: the sky cells' hit distance over 300 still frames from a reset.
    technique.resetHistory();
    const loop: Record<string, unknown>[] = [];
    let skyAtTen = 0;
    let skyAtHundred = 0;
    let fallFromTen = 0;
    let lateFall = 0;
    for (let frame = 1; frame <= DRIFT_FRAMES; frame++) {
        await run.frame();
        if (frame !== 10 && frame !== 100 && frame !== DRIFT_FRAMES) continue;
        const now = technique.probes;
        if (!now) throw new Error('no probes in the drift loop');
        const spawn = new Uint32Array(await readBuffer(device, now.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + now.layout.maxProbeSpawnCount * 4));
        const sky = skyCells(await readAtlas(device, now.probeAtlas), Array.from(spawn.subarray(1, 1 + spawn[0])));
        loop.push({ frame, ...sky });
        if (frame === 10) skyAtTen = sky.meanHitDistance;
        if (frame === 100) skyAtHundred = sky.meanHitDistance;
        if (frame === DRIFT_FRAMES) {
            fallFromTen = skyAtTen - sky.meanHitDistance;
            lateFall = skyAtHundred - sky.meanHitDistance;
        }
    }
    if (!(lateFall <= DRIFT_TOLERANCE)) fail(`drift: the sky cells' mean hit distance still fell ${lateFall.toFixed(2)} m from frame 100 to frame ${DRIFT_FRAMES}`);

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
            storeRounding,
            drift: { iteratedFilter: iterated, frameLoop: loop, fallFromFrame10: fallFromTen, fallFromFrame100: lateFall },
            depthEdge: { boxProbes: boxTiles.size, floorCellsBesideBox: besideBox, keptByGpu, keptByTwin, keptDisagree, twin: edgeCheck.comparison },
        },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
