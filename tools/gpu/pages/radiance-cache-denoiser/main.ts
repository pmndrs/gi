/**
 * RC4.3: GI-1's image-space GI denoiser (`@pmndrs/denoiser-gi1`: Capsaicin's
 * `ReprojectGI` and `FilterGI` ×2) on the GPU, against its CPU reference
 * (packages/denoiser-gi1/src/reference.ts), then inside the radiance cache.
 *
 * **The module alone**, at 480 × 270 on a CPU-ray-cast scene (synthetic.ts:
 * floor, back wall, pillar, sky), the guides uploaded exactly:
 *
 * 0. **Facts:** how this GPU rounds an f32 stored into an `rgba16float`
 *    storage texture (nearest or toward zero), which the CPU reference then
 *    copies for every half store; whether core WebGPU (no
 *    `texture-formats-tier1`) refuses Capsaicin's `r8snorm` and `r16float`
 *    as storage formats (register D65).
 * 1. **A still constant input stays constant:** E = (0.3, 0.6, 0.9) with
 *    hint 1 for 40 frames; sky pixels read 0, every surface pixel's output is
 *    the CPU reference's within 2 half steps every frame (the reference run
 *    from the GPU's own previous state), it settles (frame-to-frame change
 *    within one half step over the last 10 frames), and the reference with
 *    round-to-nearest stores, run alone for the 40 frames, keeps E within
 *    0.1%. The GPU's distance from E is reported (information: the half
 *    store's rounding, fed back through up to 64 samples).
 * 2. **The filter's weights against the CPU reference on captured frames:** a
 *    noisy checker (10% noise, 3% of pixels with hint 0), still camera, at
 *    frames 1 (no history: everything blurred at radius 8), 3 and 12; the
 *    GPU's previous state read back, the frame run on both: blur mask (within
 *    the snorm step), colour delta, `ReprojectGI`'s history where the radius
 *    is 0 (the first filter pass leaves it as it is), the first pass, the
 *    second pass from the GPU's first, and the irradiance, within 2 half steps.
 *    Pixels where a test sits within 1e-3 of its threshold are counted, not
 *    failed (and the first pass skips pixels whose footprint holds one).
 * 3. **Reprojection under a known camera move:** converged at camera A, then
 *    a pan and dolly to B: the same comparison for B's frame (velocity from
 *    the exact motion), and with the input's hint set to 0 at B (so the
 *    output is the reprojected history alone), every pixel whose
 *    neighbourhood at A lies in one checker cell reads that cell's value
 *    within 1% (a wrong motion would land on the wrong cell). The pixels B
 *    sees that A didn't (the CPU reference finds no history) have the full
 *    blur mask, the converged ones mask 0, and the noise in the disoccluded
 *    region falls by more than half (it is blurred) while the converged
 *    pixels keep their history.
 * 4. **A NaN and an infinity in the input (information):** what the
 *    output reads there and whether either reaches a neighbour (register D71).
 *
 * **In the radiance cache** (1280 × 720, the page screen, the 8-buffer device):
 *
 * 5. **Flicker and NaN on Cornell and Sponza, still camera** (T1.12's
 *    measure, information, as RC4.1's): the denoised irradiance beside the
 *    interpolated E of the same frames (RC4.1 measured E at Cornell 21.6% /
 *    81.7%, Sponza 12.6% / 43.1%). Stated before the run: the denoised
 *    output's mean and p95 flicker are below E's on both scenes. Gate: no NaN,
 *    infinity or negative value in either image on any frame.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Run through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-denoiser`.
 */

import { GI1Denoiser, frameConstants, type FrameConstants } from '../../../../packages/denoiser-gi1/src/index.ts';
import { blurRadius, filterGI, linearDepths, reprojectGI, type ReprojectResult } from '../../../../packages/denoiser-gi1/src/reference.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest } from '../ddgi-shared/harness.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { bookmarkCamera, describeSponza } from '../radiance-cache-probes/scenes.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';
import { castRay, checkerCell, denoiserCameraOf, renderGuides, type Guides } from './synthetic.ts';

const WIDTH = 480;
const HEIGHT = 270;
const PIXELS = WIDTH * HEIGHT;
const TECHNIQUE_WIDTH = 1280;
const TECHNIQUE_HEIGHT = 720;

/** Check 1: frames, and the last ones over which the output must have settled. */
const CONSTANT_FRAMES = 40;
const SETTLED_FRAMES = 10;
const CONSTANT_E: [number, number, number] = [0.3, 0.6, 0.9];
/** Check 2: the frames compared (frame 1 is the first after a reset). */
const TWIN_FRAMES = [1, 3, 12];
const NOISE = 0.1;
const WEAK_HINT_SHARE = 0.03;
/** Check 3: frames at A before the move. */
const MOVE_WARM_FRAMES = 24;
/** How near a threshold a CPU test may be and still be counted rather than failed. */
const THRESHOLD_MARGIN = 1e-3;
/** Check 5: warm-up and measured frames (as RC4.1's check 7). */
const CONVERGE_WARM_FRAMES = 32;
const CONVERGE_MEASURE_FRAMES = 32;

type Vector = [number, number, number];
type RoundingMode = 'nearest' | 'truncate';

// ---- Halves on the CPU, rounded as the GPU stores them ------------------------------------------

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** The half an f32 becomes, as bits, rounded to nearest-even or toward zero (finite, normal-range values). */
function halfBits(value: number, mode: RoundingMode): number {
    f32[0] = value;
    const bits = u32[0];
    const sign = (bits >>> 16) & 0x8000;
    const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
    const mantissa = bits & 0x7fffff;
    if (exponent <= 0) {
        // Subnormal halves: shift the full mantissa into place.
        if (exponent < -10) return sign;
        const full = mantissa | 0x800000;
        const shift = 14 - exponent;
        let half = full >>> shift;
        if (mode === 'nearest') {
            const rest = full & ((1 << shift) - 1);
            const halfway = 1 << (shift - 1);
            if (rest > halfway || (rest === halfway && (half & 1) === 1)) half += 1;
        }
        return sign | half;
    }
    if (exponent >= 0x1f) return sign | 0x7c00;
    let half = (exponent << 10) | (mantissa >>> 13);
    if (mode === 'nearest') {
        const rest = mantissa & 0x1fff;
        if (rest > 0x1000 || (rest === 0x1000 && (half & 1) === 1)) half += 1;
    }
    return sign | half;
}

function halfToFloat(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) return sign * mantissa * 2 ** -24;
    if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
    return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function quantize(values: Float32Array, mode: RoundingMode): Float32Array {
    return Float32Array.from(values, (value) => halfToFloat(halfBits(value, mode)));
}

/** A half step at `value`: the spacing of halves around it. */
function halfStep(value: number): number {
    const magnitude = Math.abs(value);
    if (magnitude < 2 ** -14) return 2 ** -24;
    return 2 ** (Math.floor(Math.log2(magnitude)) - 10);
}

/** a and b agree within `steps` half steps (of the larger), or both are tiny. */
function withinHalfSteps(a: number, b: number, steps: number): boolean {
    return Math.abs(a - b) <= steps * halfStep(Math.max(Math.abs(a), Math.abs(b))) + 2 ** -24;
}

// ---- GPU helpers --------------------------------------------------------------------------------

const TEXTURE_USAGE = { copySrc: 0x01, copyDst: 0x02, textureBinding: 0x04, storageBinding: 0x08 } as const;

function uploadTexture(device: GPUDevice, label: string, format: 'r32float' | 'rgba32float', data: Float32Array): GPUTexture {
    const texture = device.createTexture({ label, size: [WIDTH, HEIGHT], format, usage: TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copyDst | TEXTURE_USAGE.copySrc });
    writeTexture(device, texture, format, data);
    return texture;
}

function writeTexture(device: GPUDevice, texture: GPUTexture, format: 'r32float' | 'rgba32float', data: Float32Array): void {
    const channels = format === 'r32float' ? 1 : 4;
    // Every image here is a plain Float32Array over its own ArrayBuffer.
    device.queue.writeTexture({ texture }, data as Float32Array<ArrayBuffer>, { bytesPerRow: WIDTH * 4 * channels }, [WIDTH, HEIGHT]);
}

async function readF32(device: GPUDevice, texture: GPUTexture, width = WIDTH, height = HEIGHT): Promise<Float32Array> {
    const bytes = await readTexture(device, texture, 4, width, height);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

async function readHalf4(device: GPUDevice, texture: GPUTexture, width = WIDTH, height = HEIGHT): Promise<Float32Array> {
    return halves(await readTexture(device, texture, 8, width, height));
}

/** The blur mask (`rgba8snorm`, r) as the kernels read it: max(byte / 127, −1). */
async function readMask(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
    const bytes = await readTexture(device, texture, 4, WIDTH, HEIGHT);
    const signed = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return Float32Array.from({ length: PIXELS }, (_, pixel) => Math.max(signed[pixel * 4] / 127, -1));
}

/** The denoiser's state after a frame, read back. */
interface State {
    readonly color: [Float32Array, Float32Array];
    readonly colorDelta: [Float32Array, Float32Array];
    readonly mask: Float32Array;
    readonly previousDepth: Float32Array;
    readonly previousNormal: Float32Array;
    readonly irradiance: Float32Array;
    readonly latest: 0 | 1 | undefined;
}

async function readState(device: GPUDevice, denoiser: GI1Denoiser): Promise<State> {
    const history = denoiser.history;
    const output = denoiser.output;
    if (!history || !output) throw new Error('the denoiser is not configured');
    return {
        color: [await readHalf4(device, history.color[0]), await readHalf4(device, history.color[1])],
        colorDelta: [await readF32(device, history.colorDelta[0]), await readF32(device, history.colorDelta[1])],
        mask: await readMask(device, history.blurMask),
        previousDepth: await readF32(device, history.previousDepth),
        previousNormal: await readHalf4(device, history.previousNormal),
        irradiance: await readHalf4(device, output),
        latest: history.latest,
    };
}

// ---- Check 0: facts -----------------------------------------------------------------------------

const ROUNDING_PROBE_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var stored: texture_storage_2d<rgba16float, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= vec2u(64u))) {
        return;
    }
    let index = (id.y * 64u + id.x) * 4u;
    textureStore(stored, id.xy, vec4f(values[index], values[index + 1u], values[index + 2u], values[index + 3u]));
}
`;

/** How this GPU rounds f32 into an `rgba16float` storage texture: the share of 16,384 values that came out nearest-even, and toward zero. */
async function measureStoreRounding(device: GPUDevice): Promise<{ mode: RoundingMode; nearest: number; truncate: number; inexact: number; values: number }> {
    const count = 64 * 64 * 4;
    // Values that are not halves, from 1e-3 to 1e3.
    const values = Float32Array.from({ length: count }, (_, index) => Math.fround(10 ** (-3 + (6 * ((index * 2654435761) % 4093)) / 4093) * (1 + 1e-4 * (index % 7))));
    const buffer = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, values);
    const texture = device.createTexture({ size: [64, 64], format: 'rgba16float', usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.copySrc });
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: ROUNDING_PROBE_WGSL }), entryPoint: 'main' } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }, { binding: 1, resource: texture.createView() }] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(8, 8);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const bytes = await readTexture(device, texture, 8, 64, 64);
    const bits = new Uint16Array(bytes.buffer, bytes.byteOffset, count);
    let nearest = 0;
    let truncate = 0;
    let inexact = 0;
    values.forEach((value, index) => {
        const toNearest = halfBits(value, 'nearest');
        const towardZero = halfBits(value, 'truncate');
        if (toNearest === towardZero) return;
        inexact++;
        if (bits[index] === toNearest) nearest++;
        else if (bits[index] === towardZero) truncate++;
    });
    buffer.destroy();
    texture.destroy();
    return { mode: truncate > nearest ? 'truncate' : 'nearest', nearest, truncate, inexact, values: count };
}

/** Whether a device without `texture-formats-tier1` (a fresh adapter's, no features asked) refuses a storage texture of `format`. */
async function storageFormatRefused(formats: readonly GPUTextureFormat[]): Promise<Record<string, boolean>> {
    // An adapter gives one device; this one is asked for nothing.
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('no second adapter for the format check');
    const plain = await adapter.requestDevice();
    const refused: Record<string, boolean> = {};
    for (const format of formats) {
        plain.pushErrorScope('validation');
        let thrown = false;
        try {
            plain.createTexture({ size: [4, 4], format, usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding }).destroy();
        } catch {
            thrown = true;
        }
        const error = await plain.popErrorScope();
        refused[format] = thrown || error !== null;
    }
    plain.destroy();
    return refused;
}

// ---- The module harness -------------------------------------------------------------------------

/** One frame's inputs on the CPU. */
interface FrameData {
    readonly guides: Guides;
    readonly radiance: Float32Array;
    readonly camera: ReturnType<typeof denoiserCameraOf>;
}

class Harness {
    private readonly depth: GPUTexture;
    private readonly normal: GPUTexture;
    private readonly motion: GPUTexture;
    private readonly radiance: GPUTexture;

    constructor(
        private readonly device: GPUDevice,
        readonly denoiser: GI1Denoiser,
    ) {
        const empty4 = new Float32Array(PIXELS * 4);
        this.depth = uploadTexture(device, 'denoiser page depth', 'r32float', new Float32Array(PIXELS));
        this.normal = uploadTexture(device, 'denoiser page normal', 'rgba32float', empty4);
        this.motion = uploadTexture(device, 'denoiser page motion', 'rgba32float', empty4);
        this.radiance = uploadTexture(device, 'denoiser page radiance', 'rgba32float', empty4);
        denoiser.configure({ width: WIDTH, height: HEIGHT });
    }

    async frame(data: FrameData): Promise<void> {
        writeTexture(this.device, this.depth, 'r32float', data.guides.depth);
        writeTexture(this.device, this.normal, 'rgba32float', data.guides.normal);
        writeTexture(this.device, this.motion, 'rgba32float', data.guides.motion);
        writeTexture(this.device, this.radiance, 'rgba32float', data.radiance);
        this.denoiser.dispatch({ radiance: this.radiance }, { depth: this.depth, normal: this.normal, motion: this.motion }, data.camera);
        await this.device.queue.onSubmittedWorkDone();
    }

    dispose(): void {
        for (const texture of [this.depth, this.normal, this.motion, this.radiance]) texture.destroy();
    }
}

/** A deterministic hash in [0, 1). */
function hash(a: number, b: number): number {
    let h = (Math.imul(a, 0x9e3779b1) ^ Math.imul(b + 0x7f4a7c15, 0x85ebca6b)) >>> 0;
    h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
    h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/** The input a GI-1 interpolation would hand over: (0, 0, 0, 1) on sky, else E with its hint. */
function radianceImage(guides: Guides, surface: (pixel: number) => [number, number, number, number]): Float32Array {
    const out = new Float32Array(PIXELS * 4);
    for (let pixel = 0; pixel < PIXELS; pixel++) out.set(guides.hits[pixel] ? surface(pixel) : [0, 0, 0, 1], pixel * 4);
    return out;
}

/** The noisy checker: each cell's value × (1 ± noise), a few pixels with hint 0. */
function noisyChecker(guides: Guides, frame: number): Float32Array {
    return radianceImage(guides, (pixel) => {
        const hit = guides.hits[pixel];
        const value = hit ? checkerCell(hit).value : 0;
        const noisy = value * (1 + NOISE * (2 * hash(pixel, frame) - 1));
        const hint = hash(pixel, frame + 7919) < WEAK_HINT_SHARE ? 0 : 1;
        return [noisy, noisy, noisy, hint];
    });
}

// ---- The comparison of one frame against the CPU reference -------------------------------------

interface FrameComparison {
    readonly failures: string[];
    readonly summary: Record<string, number>;
    readonly reference: ReprojectResult;
}

function compareFrame(label: string, before: State, after: State, data: FrameData, frame: FrameConstants, mode: RoundingMode): FrameComparison {
    const failures: string[] = [];
    const read = before.latest;
    const written = after.latest;
    if (written === undefined) throw new Error('no frame was written');
    const zeros4 = new Float32Array(PIXELS * 4);
    const zeros1 = new Float32Array(PIXELS);
    // The frame reads the history the previous frame wrote (none just after a reset).
    const reference = reprojectGI(
        {
            width: WIDTH,
            height: HEIGHT,
            radiance: data.radiance,
            depth: data.guides.depth,
            normal: data.guides.normal,
            motion: data.guides.motion,
            previousColor: read === undefined ? zeros4 : before.color[read],
            previousColorDelta: read === undefined ? zeros1 : before.colorDelta[read],
            previousDepth: before.previousDepth,
            previousNormal: before.previousNormal,
        },
        frame,
        THRESHOLD_MARGIN,
    );
    const firstPass = after.color[written === 0 ? 1 : 0];
    const secondPass = after.color[written];
    const gpuRadius = Int32Array.from(after.mask, blurRadius);
    const linear = linearDepths(data.guides.depth, WIDTH, HEIGHT, frame);
    const undecided = reference.undecided;

    let maskOff = 0;
    let radiusOff = 0;
    let radiusSnormRounded = 0;
    let deltaOff = 0;
    let reprojectOff = 0;
    let reprojectChecked = 0;
    for (let pixel = 0; pixel < PIXELS; pixel++) {
        if (undecided[pixel]) continue;
        if (Math.abs(after.mask[pixel] - reference.blurMask[pixel]) > 1.01 / 127) maskOff++;
        const cpuRadius = blurRadius(reference.blurMask[pixel]);
        if (cpuRadius !== gpuRadius[pixel]) {
            // The snorm byte moves the mask by up to 1/127 of 8 pixels: near a rounding edge it decides the radius.
            const radius = reference.blurMask[pixel] * 8;
            const nearEdge = Math.abs(radius - Math.round(radius - 0.5) - 0.5) < 8 / 127 || Math.abs(radius) < 8 / 127;
            if (nearEdge) radiusSnormRounded++;
            else radiusOff++;
        }
        const delta = reference.colorDelta[pixel];
        const gpuDelta = after.colorDelta[written][pixel];
        if (Math.abs(gpuDelta - delta) > 1e-6 + 1e-3 * Math.abs(delta)) deltaOff++;
        if (gpuRadius[pixel] === 0) {
            reprojectChecked++;
            for (let channel = 0; channel < 4; channel++) {
                const expected = halfToFloat(halfBits(reference.color[pixel * 4 + channel], mode));
                if (!withinHalfSteps(firstPass[pixel * 4 + channel], expected, 2)) {
                    reprojectOff++;
                    break;
                }
            }
        }
    }

    // The first pass from the reference's (half-rounded) history, skipping pixels whose footprint holds an undecided one.
    const firstReference = filterGI(quantize(reference.color, mode), gpuRadius, linear, data.guides.normal, WIDTH, HEIGHT, [1, 0]);
    let firstOff = 0;
    let firstChecked = 0;
    for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
            const pixel = y * WIDTH + x;
            const radius = gpuRadius[pixel];
            let clean = true;
            for (let offset = -radius; offset <= radius && clean; offset++) {
                const tap = y * WIDTH + Math.min(Math.max(x + offset, 0), WIDTH - 1);
                if (undecided[tap]) clean = false;
            }
            if (!clean) continue;
            firstChecked++;
            for (let channel = 0; channel < 4; channel++) {
                if (!withinHalfSteps(firstPass[pixel * 4 + channel], halfToFloat(halfBits(firstReference.color[pixel * 4 + channel], mode)), 2)) {
                    firstOff++;
                    break;
                }
            }
        }
    }

    // The second pass from the GPU's own first pass: FilterGI's weights alone.
    const secondReference = filterGI(firstPass, gpuRadius, linear, data.guides.normal, WIDTH, HEIGHT, [0, 1]);
    let secondOff = 0;
    let irradianceOff = 0;
    for (let pixel = 0; pixel < PIXELS; pixel++) {
        for (let channel = 0; channel < 4; channel++) {
            if (!withinHalfSteps(secondPass[pixel * 4 + channel], halfToFloat(halfBits(secondReference.color[pixel * 4 + channel], mode)), 2)) {
                secondOff++;
                break;
            }
        }
        const irradiance = secondReference.irradiance;
        if (!irradiance) throw new Error('the second pass writes the irradiance');
        for (let channel = 0; channel < 4; channel++) {
            if (!withinHalfSteps(after.irradiance[pixel * 4 + channel], halfToFloat(halfBits(irradiance[pixel * 4 + channel], mode)), 2)) {
                irradianceOff++;
                break;
            }
        }
    }
    let undecidedCount = 0;
    for (const flag of undecided) undecidedCount += flag;
    const summary = { undecided: undecidedCount, maskOff, radiusOff, radiusSnormRounded, deltaOff, reprojectChecked, reprojectOff, firstChecked, firstOff, secondOff, irradianceOff };
    for (const [name, count] of Object.entries({ maskOff, radiusOff, deltaOff, reprojectOff, firstOff, secondOff, irradianceOff })) {
        if (count > 0) failures.push(`${label}: ${count} pixels ${name}`);
    }
    return { failures, summary, reference };
}

// ---- Check 1: a still constant input ------------------------------------------------------------

const STILL_CAMERA: PageCamera = { position: [-1.2, 1.6, 0], target: [-0.2, 0.6, -5], fovYDeg: 60, near: 0.1, far: 100 };

async function checkStillConstant(device: GPUDevice, denoiser: GI1Denoiser, mode: RoundingMode): Promise<{ failures: string[]; summary: Record<string, unknown> }> {
    const failures: string[] = [];
    const harness = new Harness(device, denoiser);
    denoiser.resetHistory();
    const guides = renderGuides(STILL_CAMERA, STILL_CAMERA, WIDTH, HEIGHT);
    const data: FrameData = { guides, radiance: radianceImage(guides, () => [...CONSTANT_E, 1]), camera: denoiserCameraOf(STILL_CAMERA, STILL_CAMERA, WIDTH, HEIGHT) };
    const frame = frameConstants(data.camera, [0, 0], WIDTH, HEIGHT);
    const surface = Uint32Array.from({ length: PIXELS }, (_, pixel) => pixel).filter((pixel) => guides.hits[pixel] !== null);
    let before = await readState(device, denoiser);
    before = { ...before, latest: undefined };
    let worstOverE = 0;
    let meanOverE = 0;
    let skyNonZero = 0;
    let twinOff = 0;
    const settledChange: number[] = [];
    let previousOutput: Float32Array | null = null;
    for (let index = 0; index < CONSTANT_FRAMES; index++) {
        await harness.frame(data);
        const after = await readState(device, denoiser);
        const comparison = compareFrame(`still constant frame ${index + 1}`, before, after, data, frame, mode);
        twinOff += comparison.failures.length;
        if (comparison.failures.length > 0 && failures.length < 4) failures.push(...comparison.failures);
        let sum = 0;
        for (const pixel of surface) {
            for (let channel = 0; channel < 3; channel++) {
                const ratio = after.irradiance[pixel * 4 + channel] / CONSTANT_E[channel];
                worstOverE = Math.max(worstOverE, Math.abs(ratio - 1));
                sum += ratio;
            }
        }
        meanOverE = sum / (3 * surface.length);
        for (let pixel = 0; pixel < PIXELS; pixel++) {
            if (guides.hits[pixel]) continue;
            if (after.irradiance[pixel * 4] !== 0 || after.irradiance[pixel * 4 + 1] !== 0 || after.irradiance[pixel * 4 + 2] !== 0) skyNonZero++;
        }
        if (previousOutput && index >= CONSTANT_FRAMES - SETTLED_FRAMES) {
            let worstSteps = 0;
            for (const pixel of surface) {
                for (let channel = 0; channel < 3; channel++) {
                    const a = after.irradiance[pixel * 4 + channel];
                    const b = previousOutput[pixel * 4 + channel];
                    worstSteps = Math.max(worstSteps, Math.abs(a - b) / halfStep(Math.max(a, b)));
                }
            }
            settledChange.push(worstSteps);
        }
        previousOutput = after.irradiance;
        before = after;
    }
    harness.dispose();
    if (skyNonZero > 0) failures.push(`still constant: ${skyNonZero} sky pixel-frames read other than 0`);
    const settled = Math.max(...settledChange);
    if (settled > 1) failures.push(`still constant: the output still moves by ${settled} half steps a frame in the last ${SETTLED_FRAMES} frames`);

    // The reference alone with round-to-nearest stores: the port's own arithmetic on a constant.
    let state: { color: Float32Array; delta: Float32Array } = { color: new Float32Array(PIXELS * 4), delta: new Float32Array(PIXELS) };
    let referenceWorst = 0;
    const linear = linearDepths(guides.depth, WIDTH, HEIGHT, frame);
    const keptNormal = Float32Array.from(guides.normal);
    for (let index = 0; index < CONSTANT_FRAMES; index++) {
        const reprojected = reprojectGI(
            { width: WIDTH, height: HEIGHT, radiance: data.radiance, depth: guides.depth, normal: guides.normal, motion: guides.motion, previousColor: state.color, previousColorDelta: state.delta, previousDepth: guides.depth, previousNormal: keptNormal },
            frame,
        );
        const radius = Int32Array.from(reprojected.blurMask, (mask) => blurRadius(Math.max(Math.round(mask * 127) / 127, -1)));
        const first = filterGI(quantize(reprojected.color, 'nearest'), radius, linear, guides.normal, WIDTH, HEIGHT, [1, 0]);
        const second = filterGI(quantize(first.color, 'nearest'), radius, linear, guides.normal, WIDTH, HEIGHT, [0, 1]);
        state = { color: quantize(second.color, 'nearest'), delta: reprojected.colorDelta };
        const irradiance = second.irradiance;
        if (!irradiance) throw new Error('the second pass writes the irradiance');
        for (const pixel of surface) {
            for (let channel = 0; channel < 3; channel++) referenceWorst = Math.max(referenceWorst, Math.abs(irradiance[pixel * 4 + channel] / CONSTANT_E[channel] - 1));
        }
    }
    if (referenceWorst > 1e-3) failures.push(`still constant: the CPU reference with round-to-nearest halves moves E by ${(referenceWorst * 100).toFixed(3)}%`);
    return {
        failures,
        summary: {
            frames: CONSTANT_FRAMES,
            surfacePixels: surface.length,
            gpuWorstOverE: worstOverE,
            gpuMeanOverELastFrame: meanOverE,
            referenceNearestWorstOverE: referenceWorst,
            settledWorstHalfSteps: settled,
            framesOffTheReference: twinOff,
        },
    };
}

// ---- Check 2: the reference on captured frames --------------------------------------------------

async function checkTwinStill(device: GPUDevice, denoiser: GI1Denoiser, mode: RoundingMode): Promise<{ failures: string[]; summary: Record<string, unknown> }> {
    const failures: string[] = [];
    const harness = new Harness(device, denoiser);
    denoiser.resetHistory();
    const guides = renderGuides(STILL_CAMERA, STILL_CAMERA, WIDTH, HEIGHT);
    const camera = denoiserCameraOf(STILL_CAMERA, STILL_CAMERA, WIDTH, HEIGHT);
    const frame = frameConstants(camera, [0, 0], WIDTH, HEIGHT);
    const summary: Record<string, unknown> = {};
    let before: State = { ...(await readState(device, denoiser)), latest: undefined };
    const lastFrame = Math.max(...TWIN_FRAMES);
    for (let index = 1; index <= lastFrame; index++) {
        const data: FrameData = { guides, radiance: noisyChecker(guides, index), camera };
        await harness.frame(data);
        if (TWIN_FRAMES.includes(index)) {
            const after = await readState(device, denoiser);
            const comparison = compareFrame(`still noise frame ${index}`, before, after, data, frame, mode);
            failures.push(...comparison.failures);
            if (index === lastFrame && comparison.summary.reprojectChecked === 0) failures.push(`still noise frame ${index}: no converged pixel (radius 0) to check the reprojection on`);
            const masks = { full: 0, none: 0, partial: 0 };
            for (let pixel = 0; pixel < PIXELS; pixel++) {
                if (!guides.hits[pixel]) continue;
                const radius = blurRadius(after.mask[pixel]);
                if (radius === 8) masks.full++;
                else if (radius === 0) masks.none++;
                else masks.partial++;
            }
            summary[`frame${index}`] = { ...comparison.summary, radius8: masks.full, radius0: masks.none, radiusBetween: masks.partial };
            before = after;
        } else {
            before = await readState(device, denoiser);
        }
    }
    harness.dispose();
    return { failures, summary };
}

// ---- Check 3: a camera move ---------------------------------------------------------------------

const CAMERA_A: PageCamera = STILL_CAMERA;
/** A pan to the right and a step forward: the pillar uncovers the wall behind its left edge. */
const CAMERA_B: PageCamera = { position: [-0.9, 1.6, -0.35], target: [0.2, 0.6, -5], fovYDeg: 60, near: 0.1, far: 100 };

async function checkCameraMove(device: GPUDevice, denoiser: GI1Denoiser, mode: RoundingMode): Promise<{ failures: string[]; summary: Record<string, unknown> }> {
    const failures: string[] = [];
    const harness = new Harness(device, denoiser);
    const guidesA = renderGuides(CAMERA_A, CAMERA_A, WIDTH, HEIGHT);
    const cameraA = denoiserCameraOf(CAMERA_A, CAMERA_A, WIDTH, HEIGHT);
    const guidesB = renderGuides(CAMERA_B, CAMERA_A, WIDTH, HEIGHT);
    const cameraB = denoiserCameraOf(CAMERA_B, CAMERA_A, WIDTH, HEIGHT);
    const frameB = frameConstants(cameraB, [0, 0], WIDTH, HEIGHT);

    // (a) The noisy checker: converge at A, then B's frame against the reference.
    denoiser.resetHistory();
    for (let index = 1; index <= MOVE_WARM_FRAMES; index++) await harness.frame({ guides: guidesA, radiance: noisyChecker(guidesA, index), camera: cameraA });
    const beforeB = await readState(device, denoiser);
    const dataB: FrameData = { guides: guidesB, radiance: noisyChecker(guidesB, MOVE_WARM_FRAMES + 1), camera: cameraB };
    await harness.frame(dataB);
    const afterB = await readState(device, denoiser);
    const comparison = compareFrame('camera move', beforeB, afterB, dataB, frameB, mode);
    failures.push(...comparison.failures);

    // Disoccluded (no history found) and converged pixels: their masks, and the blur.
    const reference = comparison.reference;
    let disoccluded = 0;
    let disoccludedMaskOff = 0;
    let converged = 0;
    let convergedMaskOff = 0;
    let inputNoise = 0;
    let outputNoise = 0;
    for (let pixel = 0; pixel < PIXELS; pixel++) {
        const hit = guidesB.hits[pixel];
        if (!hit || reference.undecided[pixel]) continue;
        const truth = checkerCell(hit).value;
        if (!reference.reprojected[pixel]) {
            disoccluded++;
            if (afterB.mask[pixel] !== 1) disoccludedMaskOff++;
            inputNoise += (dataB.radiance[pixel * 4] - truth) ** 2;
            outputNoise += (afterB.irradiance[pixel * 4] - truth) ** 2;
        } else if (reference.blurMask[pixel] === 0) {
            converged++;
            if (afterB.mask[pixel] !== 0) convergedMaskOff++;
        }
    }
    if (disoccluded < 100) failures.push(`camera move: only ${disoccluded} disoccluded pixels; the move doesn't test what it should`);
    if (disoccludedMaskOff > 0) failures.push(`camera move: ${disoccludedMaskOff} disoccluded pixels without the full blur mask`);
    if (convergedMaskOff > 0) failures.push(`camera move: ${convergedMaskOff} converged pixels with a blur mask`);
    // The blurred error includes the checker's own edges inside the footprint, so the bound is on the noise falling, not vanishing.
    const noiseRatio = Math.sqrt(outputNoise / Math.max(inputNoise, 1e-12));
    if (!(noiseRatio < 0.5)) failures.push(`camera move: the disoccluded region's error fell only to ${noiseRatio.toFixed(3)} of the input's`);

    // (b) Where the history lands: converge at A on the clean checker, then B with every hint 0.
    denoiser.resetHistory();
    const clean = (guides: Guides) => radianceImage(guides, (pixel) => {
        const hit = guides.hits[pixel];
        const value = hit ? checkerCell(hit).value : 0;
        return [value, value, value, 1];
    });
    for (let index = 1; index <= MOVE_WARM_FRAMES; index++) await harness.frame({ guides: guidesA, radiance: clean(guidesA), camera: cameraA });
    // Black with hint 0 everywhere: a pixel with history keeps it as it is (gi1.comp:4085).
    await harness.frame({ guides: guidesB, radiance: new Float32Array(PIXELS * 4), camera: cameraB });
    const landed = await readState(device, denoiser);
    const landing = checkLanding(guidesA, guidesB, landed);
    if (landing.checked < 1000) failures.push(`camera move: only ${landing.checked} pixels to check where the history lands`);
    if (landing.off > 0) failures.push(`camera move: ${landing.off} of ${landing.checked} pixels read another checker cell's value (worst ${landing.worst.toFixed(4)})`);
    harness.dispose();
    return {
        failures,
        summary: { ...comparison.summary, disoccluded, converged, noiseRatio, landing },
    };
}

/**
 * Pixels of B whose reprojected position at A has a 5 × 5 neighbourhood all
 * in B's pixel's own checker cell (so the 3 × 3 taps, wherever they land
 * within a pixel, read that cell): the output must be the cell's value.
 */
function checkLanding(guidesA: Guides, guidesB: Guides, state: State): { checked: number; off: number; worst: number } {
    let checked = 0;
    let off = 0;
    let worst = 0;
    for (let y = 0; y < HEIGHT; y++) {
        for (let x = 0; x < WIDTH; x++) {
            const pixel = y * WIDTH + x;
            const hit = guidesB.hits[pixel];
            if (!hit) continue;
            const cell = checkerCell(hit);
            const previousX = ((x + 0.5) / WIDTH - guidesB.motion[pixel * 4] * 0.5) * WIDTH;
            const previousY = ((y + 0.5) / HEIGHT + guidesB.motion[pixel * 4 + 1] * 0.5) * HEIGHT;
            const cx = Math.floor(previousX);
            const cy = Math.floor(previousY);
            let inside = cx >= 3 && cy >= 3 && cx < WIDTH - 3 && cy < HEIGHT - 3;
            for (let dy = -2; dy <= 2 && inside; dy++) {
                for (let dx = -2; dx <= 2 && inside; dx++) {
                    const neighbour = guidesA.hits[(cy + dy) * WIDTH + (cx + dx)];
                    if (!neighbour || checkerCell(neighbour).id !== cell.id) inside = false;
                }
            }
            // The same surface point must be what A saw there (not hidden behind the pillar at A).
            if (inside) {
                const origin = CAMERA_A.position;
                const toPoint: Vector = [hit.point[0] - origin[0], hit.point[1] - origin[1], hit.point[2] - origin[2]];
                const distance = Math.hypot(...toPoint);
                const seen = castRay(origin, [toPoint[0] / distance, toPoint[1] / distance, toPoint[2] / distance]);
                if (!seen || Math.abs(seen.t - distance) > 1e-6 * distance) inside = false;
            }
            if (!inside) continue;
            checked++;
            const error = Math.abs(state.irradiance[pixel * 4] / cell.value - 1);
            worst = Math.max(worst, error);
            if (error > 0.01) off++;
        }
    }
    return { checked, off, worst };
}

// ---- Check 4: a NaN and an infinity in the input ------------------------------------------------

async function checkNonFinite(device: GPUDevice, denoiser: GI1Denoiser): Promise<Record<string, unknown>> {
    const harness = new Harness(device, denoiser);
    denoiser.resetHistory();
    const guides = renderGuides(STILL_CAMERA, STILL_CAMERA, WIDTH, HEIGHT);
    const camera = denoiserCameraOf(STILL_CAMERA, STILL_CAMERA, WIDTH, HEIGHT);
    for (let index = 1; index <= 12; index++) await harness.frame({ guides, radiance: radianceImage(guides, () => [...CONSTANT_E, 1]), camera });
    // Two surface pixels in the lower middle of the image (the floor).
    const nanPixel = Math.floor(HEIGHT * 0.85) * WIDTH + Math.floor(WIDTH * 0.4);
    const infinityPixel = Math.floor(HEIGHT * 0.85) * WIDTH + Math.floor(WIDTH * 0.6);
    const radiance = radianceImage(guides, () => [...CONSTANT_E, 1]);
    radiance.set([NaN, NaN, NaN, 1], nanPixel * 4);
    radiance.set([Infinity, Infinity, Infinity, 1], infinityPixel * 4);
    const results: Record<string, unknown> = {};
    for (let index = 1; index <= 3; index++) {
        await harness.frame({ guides, radiance, camera });
        const state = await readState(device, denoiser);
        let nonFinite = 0;
        for (const value of state.irradiance) if (!Number.isFinite(value)) nonFinite++;
        results[`frame${index}`] = {
            atNaN: Array.from(state.irradiance.subarray(nanPixel * 4, nanPixel * 4 + 3)),
            atInfinity: Array.from(state.irradiance.subarray(infinityPixel * 4, infinityPixel * 4 + 3)),
            nextToNaN: Array.from(state.irradiance.subarray((nanPixel + 1) * 4, (nanPixel + 1) * 4 + 3)),
            nonFiniteValues: nonFinite,
        };
    }
    harness.dispose();
    return results;
}

// ---- Check 5: in the radiance cache -------------------------------------------------------------

interface Run {
    readonly technique: RadianceCacheTechnique;
    readonly screen: PageScreen;
    frame(): Promise<void>;
    dispose(): void;
}

async function startRun(device: GPUDevice, name: string, camera: PageCamera, describe: (platform: GIPlatformImpl) => Promise<(() => void) | void>): Promise<Run> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-denoiser ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    const beforeFrame = (await describe(platform)) ?? (() => {});
    const screen = await PageScreen.create(platform, device, TECHNIQUE_WIDTH, TECHNIQUE_HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, TECHNIQUE_WIDTH, TECHNIQUE_HEIGHT) };
    const run: Run = {
        technique,
        screen,
        async frame() {
            beforeFrame();
            const encoder = device.createCommandEncoder({ label: `radiance-cache-denoiser ${name}` });
            platform.encode(encoder, input);
            screen.encode(encoder, camera);
            platform.encodeScreen(encoder, screen.screen);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        },
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
    await run.frame();
    await platform.rays.whenBuilt();
    return run;
}

function percentile(sorted: Float32Array, share: number): number {
    if (sorted.length === 0) return 0;
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
}

/** T1.12's measure between two frames, over surface pixels: mean and p95 of |ΔE| (rgb mean) over the image's mean. */
function frameChange(previous: Float32Array, current: Float32Array, surface: Uint32Array): { mean: number; p95: number } {
    let sum = 0;
    for (const pixel of surface) sum += (current[pixel * 4] + current[pixel * 4 + 1] + current[pixel * 4 + 2]) / 3;
    const imageMean = sum / Math.max(surface.length, 1);
    const changes = new Float32Array(surface.length);
    let total = 0;
    surface.forEach((pixel, index) => {
        let change = 0;
        for (let channel = 0; channel < 3; channel++) change += Math.abs(current[pixel * 4 + channel] - previous[pixel * 4 + channel]) / 3;
        changes[index] = change / imageMean;
        total += changes[index];
    });
    changes.sort();
    return { mean: total / Math.max(surface.length, 1), p95: percentile(changes, 0.95) };
}

function badValues(image: Float32Array, surface: Uint32Array): number {
    let bad = 0;
    for (const pixel of surface) {
        for (let channel = 0; channel < 3; channel++) {
            const value = image[pixel * 4 + channel];
            if (!Number.isFinite(value) || value < 0) bad++;
        }
    }
    return bad;
}

async function checkFlicker(device: GPUDevice, name: string, run: Run): Promise<{ failures: string[]; summary: Record<string, unknown> }> {
    const failures: string[] = [];
    run.technique.resetHistory();
    await run.frame();
    const depth = await readF32(device, run.screen.screen.depth, TECHNIQUE_WIDTH, TECHNIQUE_HEIGHT);
    const surface = Uint32Array.from({ length: TECHNIQUE_WIDTH * TECHNIQUE_HEIGHT }, (_, pixel) => pixel).filter((pixel) => depth[pixel] !== 1);
    const read = async (): Promise<{ interpolated: Float32Array; denoised: Float32Array }> => {
        const probes = run.technique.probes;
        const output = run.technique.denoiser?.output;
        if (!probes || !output) throw new Error('no images');
        return { interpolated: await readHalf4(device, probes.irradiance, TECHNIQUE_WIDTH, TECHNIQUE_HEIGHT), denoised: await readHalf4(device, output, TECHNIQUE_WIDTH, TECHNIQUE_HEIGHT) };
    };
    let previous = await read();
    const interpolated: { mean: number; p95: number }[] = [];
    const denoised: { mean: number; p95: number }[] = [];
    let bad = 0;
    let means = { interpolated: 0, denoised: 0 };
    for (let frame = 1; frame < CONVERGE_WARM_FRAMES + CONVERGE_MEASURE_FRAMES; frame++) {
        await run.frame();
        const current = await read();
        bad += badValues(current.interpolated, surface) + badValues(current.denoised, surface);
        if (frame >= CONVERGE_WARM_FRAMES) {
            interpolated.push(frameChange(previous.interpolated, current.interpolated, surface));
            denoised.push(frameChange(previous.denoised, current.denoised, surface));
            const mean = (image: Float32Array) => surface.reduce((sum, pixel) => sum + (image[pixel * 4] + image[pixel * 4 + 1] + image[pixel * 4 + 2]) / 3, 0) / surface.length;
            means = { interpolated: mean(current.interpolated), denoised: mean(current.denoised) };
        }
        previous = current;
    }
    if (bad > 0) failures.push(`${name}: ${bad} NaN, infinite or negative values in the interpolated or denoised irradiance`);
    const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
    return {
        failures,
        summary: {
            interpolatedFlickerMean: average(interpolated.map((change) => change.mean)),
            interpolatedFlickerP95: average(interpolated.map((change) => change.p95)),
            denoisedFlickerMean: average(denoised.map((change) => change.mean)),
            denoisedFlickerP95: average(denoised.map((change) => change.p95)),
            lastFrameMeanInterpolated: means.interpolated,
            lastFrameMeanDenoised: means.denoised,
            surfacePixels: surface.length,
            ddgiThresholds: { mean: 0.001, p95: 0.005 },
        },
    };
}

// ---- The page -----------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const { device, adapterStorageBuffersPerStage } = await requestBudgetDevice(adapter);
    window.addEventListener('pagehide', () => device.destroy(), { once: true });
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    const results: Record<string, unknown> = {};

    const rounding = await measureStoreRounding(device);
    const refusedWithoutTier1 = await storageFormatRefused(['r8snorm', 'r16float', 'rgba8snorm', 'r32float', 'rgba16float']);
    results.facts = { rgba16floatStoreRounding: rounding, refusedAsStorageWithoutTier1: refusedWithoutTier1, adapterHasTier1: adapter.features.has('texture-formats-tier1') };

    const denoiser = await GI1Denoiser.create(device, { label: 'radiance-cache-denoiser page' });
    try {
        const still = await checkStillConstant(device, denoiser, rounding.mode);
        failures.push(...still.failures);
        results.stillConstant = still.summary;
        const twin = await checkTwinStill(device, denoiser, rounding.mode);
        failures.push(...twin.failures);
        results.capturedFrames = twin.summary;
        const move = await checkCameraMove(device, denoiser, rounding.mode);
        failures.push(...move.failures);
        results.cameraMove = move.summary;
        results.nonFinite = await checkNonFinite(device, denoiser);
    } finally {
        denoiser.dispose();
        await device.queue.onSubmittedWorkDone();
    }

    const cornell = loadManifest('cornell-box');
    const sponza = loadManifest('sponza');
    const scenes: { name: string; camera: PageCamera; describe: (platform: GIPlatformImpl) => Promise<(() => void) | void> }[] = [
        { name: 'cornell-box', camera: bookmarkCamera(cornell, 100), describe: async (platform) => void describeScene(platform, cornell) },
        { name: 'sponza', camera: bookmarkCamera(sponza, 1000), describe: (platform) => describeSponza(platform, device, sponza) },
    ];
    for (const scene of scenes) {
        const run = await startRun(device, scene.name, scene.camera, scene.describe);
        try {
            const flicker = await checkFlicker(device, scene.name, run);
            failures.push(...flicker.failures);
            results[scene.name] = flicker.summary;
        } finally {
            run.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }

    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.slice(0, 20).join(' | '),
        data: { width: WIDTH, height: HEIGHT, adapterStorageBuffersPerStage, results },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
