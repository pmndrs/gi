/**
 * Shared by the T2.4b pages (the GPU scheduler's response checks): frames
 * with a real camera (the scheduler orders probes by their camera weight),
 * the probe history read back, converging a field, and the response of a
 * view after a step, counted in frames.
 *
 * Not a page itself (no index.html).
 */

import type { FrameBudget, FrameCamera, FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { DDGIField } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { PROBE_HISTORY_FLAGS_MASK, PROBE_HISTORY_UPDATES, PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS } from '../../../../packages/ddgi-core/src/techniques/ddgi/layout.ts';
import type { DDGISettings } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { halfToFloat, readBuffer } from './harness.ts';
import { applyHysteresisMode, FIXED_097, parseHysteresisMode } from './hysteresis-modes.ts';

type Vec3 = [number, number, number];

/** The view of a manifest camera bookmark as the platform takes it: column-major matrices, WebGPU depth 0..1. */
export function bookmarkCamera(bookmark: CameraBookmark, aspect: number, near = 0.05, far = 500): FrameCamera {
    const eye = bookmark.position as Vec3;
    const target = bookmark.target as Vec3;
    const forward = normalize(subtract(target, eye));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    // World to view: the camera looks down -z.
    const view = [right[0], up[0], -forward[0], 0, right[1], up[1], -forward[1], 0, right[2], up[2], -forward[2], 0, -dot(right, eye), -dot(up, eye), dot(forward, eye), 1];
    const f = 1 / Math.tan((bookmark.fovYDeg * Math.PI) / 360);
    const projection = [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
    return { position: eye, view, projection, near, far };
}

export function cameraFrameInput(budget: number | FrameBudget, camera: FrameCamera, deltaTime = 1 / 60): FrameInput {
    return { deltaTime, budget: typeof budget === 'number' ? { units: budget } : budget, camera };
}

/** One frame, submitted. */
export function runCameraFrame(device: GPUDevice, platform: GIPlatformImpl, budget: number | FrameBudget, camera: FrameCamera, deltaTime = 1 / 60): void {
    const encoder = device.createCommandEncoder({ label: 'ddgi test frame' });
    platform.encode(encoder, cameraFrameInput(budget, camera, deltaTime));
    device.queue.submit([encoder.finish()]);
}

/** Frames, waiting for the GPU every few so the page never sits in one long task. */
export async function runCameraFrames(device: GPUDevice, platform: GIPlatformImpl, frames: number, budget: number | FrameBudget, camera: FrameCamera, deltaTime = 1 / 60): Promise<void> {
    for (let index = 0; index < frames; index++) {
        runCameraFrame(device, platform, budget, camera, deltaTime);
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();
}

/** One probe's history record, decoded (PROBE_HISTORY_WORD). */
export interface ProbeHistoryRecord {
    readonly samples: number;
    readonly updates: number;
    readonly owed: number;
    readonly pendingChange: number;
    readonly urgency: number;
    readonly flags: number;
    /** The frame (24 bits) it was last put in the update list. */
    readonly scheduledFrame: number;
    readonly mean: Vec3;
    readonly noise: number;
}

export async function readHistory(device: GPUDevice, field: DDGIField): Promise<ProbeHistoryRecord[]> {
    const probeCount = field.grid.probeCount;
    const words = new Uint32Array(await readBuffer(device, field.probeHistory, probeCount * PROBE_HISTORY_WORDS * 4));
    const floats = new Float32Array(words.buffer);
    const records: ProbeHistoryRecord[] = [];
    for (let probe = 0; probe < probeCount; probe++) {
        const base = probe * PROBE_HISTORY_WORDS;
        const updates = words[base + PROBE_HISTORY_WORD.updates];
        const schedule = words[base + PROBE_HISTORY_WORD.scheduleAndFlags];
        const redGreen = words[base + PROBE_HISTORY_WORD.meanRedGreen];
        const blueNoise = words[base + PROBE_HISTORY_WORD.meanBlueNoise];
        records.push({
            samples: floats[base + PROBE_HISTORY_WORD.samples],
            updates: updates & PROBE_HISTORY_UPDATES.countMask,
            owed: (updates >>> PROBE_HISTORY_UPDATES.owedShift) & PROBE_HISTORY_UPDATES.owedMask,
            pendingChange: floats[base + PROBE_HISTORY_WORD.pendingChange],
            urgency: floats[base + PROBE_HISTORY_WORD.urgency],
            flags: schedule & PROBE_HISTORY_FLAGS_MASK,
            scheduledFrame: schedule >>> 8,
            mean: [halfToFloat(redGreen & 0xffff), halfToFloat(redGreen >>> 16), halfToFloat(blueNoise & 0xffff)],
            noise: halfToFloat(blueNoise >>> 16),
        });
    }
    return records;
}

/** A probe's grid coordinates from its index (x + nx (z + nz y)). */
export function probeCoordinates(probe: number, counts: readonly [number, number, number]): Vec3 {
    const [nx, , nz] = counts;
    return [probe % nx, Math.floor(probe / (nx * nz)), Math.floor(probe / nx) % nz];
}

/** How a view follows a step, from its mean each frame after it (frame 1 is the step's own frame). */
export interface StepResponse {
    /** The view's change in frame 1 against before the step, as a share of before. */
    readonly firstFrameChange: number;
    /** The step: settled against before, as a share of before. */
    readonly stepShare: number;
    /** First frame within the tolerance of settled. */
    readonly enteredFrame: number | null;
    /** First frame from which every measured frame stays within the tolerance; null if the last one is out. */
    readonly settledFrame: number | null;
    /** It entered the band and left it again (counts as not settled until it is back for good). */
    readonly leftBandAgain: boolean;
    /** The furthest the view went past settled, as a share of the step (0 when it never passed it). */
    readonly overshootShareOfStep: number;
    /** The same as a share of settled. */
    readonly overshoot: number;
    /** The error against settled each frame (signed share of settled). */
    readonly errors: number[];
}

export function stepResponse(before: number, settled: number, means: readonly number[], tolerance: number): StepResponse {
    const errors = means.map((mean) => mean / settled - 1);
    const inBand = errors.map((error) => Math.abs(error) <= tolerance);
    const enteredIndex = inBand.indexOf(true);
    let settledIndex: number | null = null;
    for (let index = inBand.length - 1; index >= 0 && inBand[index]; index--) settledIndex = index;
    const leftBandAgain = enteredIndex >= 0 && (settledIndex === null || settledIndex > enteredIndex);
    // Past settled in the step's direction: above it for a step up, below for a step down.
    const direction = Math.sign(settled - before);
    const past = Math.max(0, ...means.map((mean) => direction * (mean - settled)));
    const step = Math.abs(settled - before);
    return {
        firstFrameChange: means.length > 0 ? means[0] / before - 1 : 0,
        stepShare: settled / before - 1,
        enteredFrame: enteredIndex >= 0 ? enteredIndex + 1 : null,
        settledFrame: settledIndex === null ? null : settledIndex + 1,
        leftBandAgain,
        overshootShareOfStep: step > 0 ? past / step : 0,
        overshoot: past / settled,
        errors,
    };
}

/** The curve as text: the error by frame, frames 1-12 then every 6th. */
export function describeErrors(errors: readonly number[]): string {
    return errors
        .map((error, index) => ({ frame: index + 1, error }))
        .filter(({ frame }) => frame <= 12 || frame % 6 === 0)
        .map(({ frame, error }) => `${frame}: ${(error * 100).toFixed(1)}%`)
        .join(', ');
}

/** How a field is converged before a step (`convergeField`). */
export interface ConvergeOptions {
    /** Frames at a fixed fast hysteresis first (bounce light settles fast, noisily). */
    readonly fastFrames: number;
    readonly fastHysteresis: number;
    /** Then blocks at the fixed 0.97 until the view's mean moves less than `convergedTolerance` between blocks. */
    readonly blockFrames: number;
    readonly convergedTolerance: number;
    readonly maxBlocks: number;
    /** Then this many frames of the default rule at `settleUnits`, so every probe's history reaches its steady count. */
    readonly settleFrames: number;
    /** Probes a frame while converging (every probe, or as many as fit). */
    readonly convergeUnits: number;
    readonly settleUnits: number;
    /**
     * The frame time of the settle frames, seconds (default 1/60): DDGI's
     * history clock advances by it, so a comparison with a host running at
     * another frame rate settles at that host's (T2.8, `viewer-lighting`).
     */
    readonly settleDeltaTime?: number;
}

/**
 * Converges a field (T2.4a's ddgi-change-step procedure): a field that isn't
 * converged reads low, and one whose cut histories converge fast then looks
 * as if it overshoots. Returns the view's mean, and throws when the view
 * didn't settle.
 */
export async function convergeField(
    device: GPUDevice,
    platform: GIPlatformImpl,
    settings: DDGISettings,
    camera: FrameCamera,
    viewMean: () => Promise<number>,
    options: ConvergeOptions,
    label: string,
    log: (line: string) => void,
): Promise<number> {
    const defaults = parseHysteresisMode('default');
    applyHysteresisMode(settings, { name: 'fast', settings: { ...FIXED_097.settings, hysteresis: options.fastHysteresis } });
    await runCameraFrames(device, platform, options.fastFrames, options.convergeUnits, camera);
    applyHysteresisMode(settings, FIXED_097);
    let mean = await viewMean();
    let settled = false;
    for (let block = 1; block <= options.maxBlocks; block++) {
        await runCameraFrames(device, platform, options.blockFrames, options.convergeUnits, camera);
        const next = await viewMean();
        const change = next / mean - 1;
        mean = next;
        if (block >= 2 && Math.abs(change) < options.convergedTolerance) {
            settled = true;
            log(`${label}: converged after ${block} blocks, view mean ${next.toPrecision(6)}`);
            break;
        }
    }
    if (!settled) throw new Error(`${label}: the view's mean did not settle within ${options.maxBlocks} blocks`);
    applyHysteresisMode(settings, defaults);
    await runCameraFrames(device, platform, options.settleFrames, options.settleUnits, camera, options.settleDeltaTime);
    return viewMean();
}

function subtract(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(a: Vec3): Vec3 {
    const length = Math.hypot(...a);
    return [a[0] / length, a[1] / length, a[2] / length];
}

/**
 * Check 3's census (T2.4b): the longest any probe went without being put in
 * the update list. A test kernel reads the history's scheduled-frame stamps
 * (24 bits, PROBE_HISTORY_WORD.scheduleAndFlags) after each frame and keeps
 * the largest `frame − stamp` over every probe, on the GPU, so a long run on
 * Sponza needs no read-back per frame. Start it once every probe has been
 * listed at least once (a stamp of 0 would read as a huge gap).
 */
export class ScheduleGapCensus {
    private constructor(
        private readonly device: GPUDevice,
        private readonly kernel: Awaited<ReturnType<GIPlatformImpl['kernels']['computeKernel']>>,
        private readonly uniforms: GPUBuffer,
        private readonly result: GPUBuffer,
        private readonly group: GPUBindGroup,
        private readonly probeCount: number,
    ) {}

    static async create(device: GPUDevice, platform: GIPlatformImpl, field: DDGIField): Promise<ScheduleGapCensus> {
        const layout: GPUBindGroupLayoutDescriptor = {
            label: 'ddgi test schedule gap',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        };
        const kernel = await platform.kernels.computeKernel({
            label: 'ddgi test schedule gap',
            modules: [],
            code: /* wgsl */ `
struct GapUniforms { frameStamp: u32, probeCount: u32, spare0: u32, spare1: u32 }
@group(1) @binding(0) var<uniform> gap: GapUniforms;
@group(1) @binding(1) var<storage, read> gapHistory: array<u32>;
// The largest gap, then the probe it was at.
@group(1) @binding(2) var<storage, read_write> gapResult: array<atomic<u32>, 2>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= gap.probeCount) {
        return;
    }
    let stamp = gapHistory[id.x * ${PROBE_HISTORY_WORDS}u + ${PROBE_HISTORY_WORD.scheduleAndFlags}u] >> 8u;
    let frames = (gap.frameStamp - stamp) & 0xffffffu;
    let previous = atomicMax(&gapResult[0], frames);
    if (frames > previous) {
        atomicStore(&gapResult[1], id.x);
    }
}
`,
            entryPoint: 'main',
            ownGroups: [layout],
        });
        const uniforms = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const result = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const group = device.createBindGroup({
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: field.probeHistory } },
                { binding: 2, resource: { buffer: result } },
            ],
        });
        return new ScheduleGapCensus(device, kernel, uniforms, result, group, field.grid.probeCount);
    }

    /** After the platform's frame `frame` (its `stats.frame`): every probe's gap since it was last listed. */
    measure(frame: number): void {
        this.device.queue.writeBuffer(this.uniforms, 0, new Uint32Array([frame & 0xffffff, this.probeCount, 0, 0]));
        const encoder = this.device.createCommandEncoder({ label: 'ddgi test schedule gap' });
        const pass = encoder.beginComputePass();
        this.kernel.bind(pass);
        pass.setBindGroup(1, this.group);
        pass.dispatchWorkgroups(Math.ceil(this.probeCount / 64));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    /** The largest gap measured, in frames, and the probe. */
    async read(): Promise<{ frames: number; probe: number }> {
        const values = new Uint32Array(await readBuffer(this.device, this.result, 8));
        return { frames: values[0], probe: values[1] };
    }

    dispose(): void {
        this.uniforms.destroy();
        this.result.destroy();
    }
}
