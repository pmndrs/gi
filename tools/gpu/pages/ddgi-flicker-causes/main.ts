/**
 * T1.12: why DDGI flickers on a still scene (Dennis: Cornell "is constantly
 * moving ... the lighting cant settle"). The `ddgi-flicker` page measures how
 * much; this page changes one thing at a time to find out why.
 *
 * Steps (one scene, one camera):
 * 1. Start the scene as the viewer does, warm up, then converge with full
 *    passes (the flicker page's rule).
 * 2. For each experiment: apply its settings, settle with full passes at
 *    those settings (`settlePasses` updates of every probe), switch to its
 *    probes-per-frame, then measure MEASURE_FRAMES frames. Each frame: one
 *    platform update, one test-view render, the irradiance atlas and the
 *    probe-data texture read back. Then the base settings come back.
 *
 * Measured per experiment:
 * - **image, frame to frame** (the flicker page's metric): per surface pixel
 *   d = mean over rgb |GI(t) - GI(t-1)| / mean(GI(t)); the mean and p95 over
 *   pixels, averaged over frame pairs. Also per material class (red wall,
 *   green wall, white), each relative to that class's own mean GI.
 * - **image, over time** (the "can't settle" part, which a frame-to-frame
 *   number undercounts when hysteresis makes the wobble slow): each pixel's
 *   standard deviation over the measured frames / mean GI; mean and p95.
 * - **atlas texels**, interior irradiance texels of probes active throughout,
 *   value = mean of rgb: the change per frame and the standard deviation over
 *   time, both / the atlas mean; split into probes against the red wall
 *   (x = 0), against the green wall (x = nx - 1) and the rest.
 * - **probe states**: probes whose state or offset changed between frames.
 *
 * Experiments that need more than settings, done from this page only by
 * rewriting the technique's uniform upload (no core edit):
 * - `rotation: 'fixed'`: every update uses the same ray rotation instead of a
 *   new random one per frame;
 * - `freezeFrame`: the frame number the kernels see stays constant, which
 *   freezes the random numbers of the stochastic half rounding (and of the
 *   hit shading's light sampling). With a fixed rotation too, an update is
 *   fully deterministic, so what still changes is the floor.
 *
 * The per-update texel noise (sigma) and the hysteresis theory: right after
 * the base experiment, with the base's per-texel means mu from its measured
 * frames, one update of every probe runs with hysteresis 0 (and the firefly
 * clamp off), so each texel holds one update's estimate e. sigma = rms over
 * texels of (e - mu) / atlas mean, repeated SIGMA_REPEATS times with a settle
 * in between. For a texel updated every frame with hysteresis h,
 * x(t) = h x(t-1) + (1-h) e(t) gives
 *   standard deviation over time      = sigma * sqrt((1-h) / (1+h)),
 *   frame-to-frame change (rms)       = sigma * (1-h) * sqrt(2 / (1+h)),
 * and with a share f of probes updated per frame the rms change is sqrt(f)
 * times that (the over-time deviation doesn't depend on f).
 *
 *   VITE_DDGI_SCENES=cornell-box        (one scene; cornell-box-point for the point light)
 *   VITE_DDGI_CAMERA=1
 *   VITE_DDGI_CAUSE_EXPERIMENTS=base,units-32,...   (default: all, in the order below)
 *   VITE_DDGI_FLICKER_FRAMES=120
 *
 * Correctness only (no timings), so no bench lock.
 */

import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { halfToFloat, imageMean, readTextureBytes, TestView } from '../ddgi-shared/harness.ts';
import { startViewerScene, warmUp, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';
import { MAT3_COLUMN_WORDS, UNIFORM_WORD } from '../../../../packages/ddgi-core/src/techniques/ddgi/layout.ts';
import { frameRayRotation } from '../../../../packages/ddgi-core/src/techniques/ddgi/ray-rotation.ts';
import type { DDGISettings } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}
function countFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' ? Number(value) : NaN;
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

const SCENE = listFromEnv(env.VITE_DDGI_SCENES, ['cornell-box'])[0];
const CAMERA = countFromEnv(env.VITE_DDGI_CAMERA, 1);
const MEASURE_FRAMES = countFromEnv(env.VITE_DDGI_FLICKER_FRAMES, 120);
const WIDTH = countFromEnv(env.VITE_DDGI_WIDTH, 640);
const HEIGHT = countFromEnv(env.VITE_DDGI_HEIGHT, 360);

/** The flicker page's convergence rule. */
const BLOCK_PASSES = 32;
const CONVERGED_TOLERANCE = 0.005;
const MIN_BLOCKS = 2;
const MAX_BLOCKS = 12;
const PROBES_UPDATED = 'probes updated';
/** Full passes at an experiment's settings before measuring: five time constants at hysteresis 0.97. */
const DEFAULT_SETTLE_PASSES = 150;
const SIGMA_REPEATS = 3;
/** The frame number whose rotation stands in for "fixed", and the frozen frame number. Any value works. */
const FIXED_ROTATION_FRAME = 7;
const FROZEN_FRAME = 7;
/** A firefly clamp that never trips. */
const CLAMP_OFF = 1e30;
/** Bytes per texel: irradiance and probe data are rgba16float. */
const RGBA16_BYTES = 8;
/** The probe data's w channel: 0 is active. */
const STATE_ACTIVE = 0;

interface Experiment {
    readonly name: string;
    /** Probes updated per frame while measuring; 'all' = every probe. */
    readonly units: number | 'all';
    readonly settings?: Partial<Pick<DDGISettings, 'hysteresis' | 'raysPerProbe' | 'maxBrightnessChange' | 'bounceEnergy'>>;
    readonly rotation?: 'random' | 'fixed';
    readonly freezeFrame?: boolean;
    readonly settlePasses?: number;
    /** Measure sigma right after this experiment (base only). */
    readonly sigma?: boolean;
}

const EXPERIMENTS: readonly Experiment[] = [
    { name: 'base', units: 'all', sigma: true },
    { name: 'units-32', units: 32 },
    { name: 'units-128', units: 128 },
    { name: 'units-512', units: 512 },
    { name: 'units-876', units: 876 },
    { name: 'hysteresis-0.98', units: 'all', settings: { hysteresis: 0.98 }, settlePasses: 250 },
    { name: 'hysteresis-0.99', units: 'all', settings: { hysteresis: 0.99 }, settlePasses: 500 },
    { name: 'rays-128', units: 'all', settings: { raysPerProbe: 128 } },
    { name: 'rays-512', units: 'all', settings: { raysPerProbe: 512 } },
    { name: 'rotation-fixed', units: 'all', rotation: 'fixed' },
    { name: 'frame-frozen', units: 'all', freezeFrame: true },
    { name: 'rotation-fixed-frame-frozen', units: 'all', rotation: 'fixed', freezeFrame: true },
    { name: 'clamp-off', units: 'all', settings: { maxBrightnessChange: CLAMP_OFF } },
    { name: 'clamp-0.05', units: 'all', settings: { maxBrightnessChange: 0.05 } },
    { name: 'bounce-off', units: 'all', settings: { bounceEnergy: 0 } },
];

const SELECTED = listFromEnv(
    env.VITE_DDGI_CAUSE_EXPERIMENTS,
    EXPERIMENTS.map((experiment) => experiment.name),
);

// ---- Statistics ----

function percentile(sorted: Float32Array | Float64Array, share: number): number {
    return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(share * sorted.length) - 1))];
}
function mean(values: ArrayLike<number>): number {
    let sum = 0;
    for (let index = 0; index < values.length; index++) sum += values[index];
    return sum / values.length;
}
function rootMeanSquare(values: ArrayLike<number>): number {
    let sum = 0;
    for (let index = 0; index < values.length; index++) sum += values[index] * values[index];
    return Math.sqrt(sum / values.length);
}
function round(value: number, significant = 4): number {
    return Number.isFinite(value) ? Number(value.toPrecision(significant)) : value;
}
function sortedCopy(values: ArrayLike<number>): Float64Array {
    return Float64Array.from(values).sort();
}

/** Per-element running mean and variance over frames (Welford). */
class RunningMoments {
    private count = 0;
    readonly mean: Float64Array;
    private readonly squares: Float64Array;
    constructor(size: number) {
        this.mean = new Float64Array(size);
        this.squares = new Float64Array(size);
    }
    add(values: ArrayLike<number>): void {
        this.count++;
        for (let index = 0; index < this.mean.length; index++) {
            const delta = values[index] - this.mean[index];
            this.mean[index] += delta / this.count;
            this.squares[index] += delta * (values[index] - this.mean[index]);
        }
    }
    standardDeviation(index: number): number {
        return this.count > 1 ? Math.sqrt(this.squares[index] / (this.count - 1)) : 0;
    }
}

// ---- The uniform rewrite (rotation and frame number) ----

interface UniformOverride {
    rotation: 'random' | 'fixed';
    freezeFrame: boolean;
}

/**
 * Wraps `queue.writeBuffer` so the DDGI uniform upload (buffer label
 * 'ddgi uniforms', written whole from word 0 each frame) can carry a fixed
 * rotation or frame number. Every other write passes through untouched.
 */
function installUniformOverride(device: GPUDevice): UniformOverride {
    const override: UniformOverride = { rotation: 'random', freezeFrame: false };
    const queue = device.queue;
    const originalWrite = queue.writeBuffer.bind(queue);
    const fixedRotation = frameRayRotation(FIXED_ROTATION_FRAME);
    queue.writeBuffer = ((buffer: GPUBuffer, bufferOffset: number, data: BufferSource, dataOffset?: number, size?: number) => {
        const changes = override.rotation === 'fixed' || override.freezeFrame;
        if (!changes || buffer.label !== 'ddgi uniforms' || bufferOffset !== 0 || !(data instanceof Float32Array) || dataOffset !== undefined) {
            return originalWrite(buffer, bufferOffset, data, dataOffset, size);
        }
        const floats = new Float32Array(data);
        const words = new Uint32Array(floats.buffer);
        if (override.rotation === 'fixed') {
            for (let column = 0; column < 3; column++) floats.set(fixedRotation.slice(column * 3, column * 3 + 3), UNIFORM_WORD.rayRotation + column * MAT3_COLUMN_WORDS);
        }
        if (override.freezeFrame) words[UNIFORM_WORD.frame] = FROZEN_FRAME;
        return originalWrite(buffer, bufferOffset, floats, dataOffset, size);
    }) as GPUQueue['writeBuffer'];
    return override;
}

// ---- Reading the field ----

interface FieldSnapshot {
    /** Per interior irradiance texel (probe-major, then texel), the mean of rgb. */
    readonly texels: Float32Array;
    /** Per probe: state, and offset xyz. */
    readonly states: Uint8Array;
    readonly offsets: Float32Array;
}

async function readField(scene: ViewerScene): Promise<FieldSnapshot> {
    const field = scene.ddgi.field;
    assert(field, 'DDGI has no field');
    const [nx, ny, nz] = field.grid.counts;
    const texelsPerSide = scene.ddgi.settings.irradianceTexels;
    const tile = texelsPerSide + 2;
    const irradiance = new Uint16Array((await readTextureBytes(scene.device, field.irradiance, RGBA16_BYTES)).buffer);
    const probeData = new Uint16Array((await readTextureBytes(scene.device, field.probeData, RGBA16_BYTES)).buffer);
    const probeCount = nx * ny * nz;
    const texelsPerProbe = texelsPerSide * texelsPerSide;
    const texels = new Float32Array(probeCount * texelsPerProbe);
    const states = new Uint8Array(probeCount);
    const offsets = new Float32Array(probeCount * 3);
    const width = field.irradiance.width;
    const height = field.irradiance.height;
    for (let y = 0; y < ny; y++) {
        for (let z = 0; z < nz; z++) {
            for (let x = 0; x < nx; x++) {
                const probe = x + nx * (z + nz * y);
                const dataBase = ((y * nz + z) * nx + x) * 4;
                for (let axis = 0; axis < 3; axis++) offsets[probe * 3 + axis] = halfToFloat(probeData[dataBase + axis]);
                states[probe] = Math.round(halfToFloat(probeData[dataBase + 3]));
                for (let row = 0; row < texelsPerSide; row++) {
                    for (let column = 0; column < texelsPerSide; column++) {
                        // Layer y; tile (x, z); 1-texel border.
                        const atlasX = x * tile + 1 + column;
                        const atlasY = z * tile + 1 + row;
                        const base = ((y * height + atlasY) * width + atlasX) * 4;
                        const value = (halfToFloat(irradiance[base]) + halfToFloat(irradiance[base + 1]) + halfToFloat(irradiance[base + 2])) / 3;
                        texels[probe * texelsPerProbe + row * texelsPerSide + column] = value;
                    }
                }
            }
        }
    }
    return { texels, states, offsets };
}

// ---- Pixel classes ----

const CLASSES = ['red', 'green', 'white', 'other'] as const;
type PixelClass = (typeof CLASSES)[number];

/** Cornell's materials by GI albedo (red 0.63/0.065/0.05, green 0.14/0.45/0.091, white about 0.7). */
function classify(albedo: Float32Array, pixel: number): PixelClass {
    const [r, g, b] = [albedo[pixel * 3], albedo[pixel * 3 + 1], albedo[pixel * 3 + 2]];
    if (r > 0.5 && g < 0.2 && b < 0.2) return 'red';
    if (g > 0.3 && r < 0.3 && b < 0.2) return 'green';
    if (r > 0.6 && g > 0.6 && b > 0.6) return 'white';
    return 'other';
}

// ---- One experiment ----

interface ExperimentContext {
    readonly scene: ViewerScene;
    readonly view: TestView;
    readonly bookmark: ViewerScene['manifest']['cameras'][number];
    readonly override: UniformOverride;
    readonly base: Pick<DDGISettings, 'hysteresis' | 'raysPerProbe' | 'maxBrightnessChange' | 'bounceEnergy'>;
    readonly log: (line: string) => void;
}

function applyExperiment(context: ExperimentContext, experiment: Experiment | null): void {
    const settings = context.scene.ddgi.settings;
    Object.assign(settings, context.base, experiment?.settings ?? {});
    context.override.rotation = experiment?.rotation ?? 'random';
    context.override.freezeFrame = experiment?.freezeFrame ?? false;
}

async function fullPasses(scene: ViewerScene, passes: number): Promise<void> {
    for (let pass = 0; pass < passes; pass++) {
        scene.frame({ units: scene.probeCount });
        if (pass % 8 === 7) await scene.device.queue.onSubmittedWorkDone();
    }
    await scene.device.queue.onSubmittedWorkDone();
}

async function renderIndirect(context: ExperimentContext): Promise<{ indirect: Float32Array; albedo: Float32Array | undefined }> {
    const field = context.scene.ddgi.field;
    assert(field, 'DDGI has no field');
    const images = await context.view.render(field, context.bookmark, WIDTH, HEIGHT);
    // `albedo` arrived with T1.13's test-view change; without it the classes are skipped.
    return { indirect: images.indirect, albedo: (images as { albedo?: Float32Array }).albedo };
}

interface ExperimentResult {
    readonly row: Record<string, unknown>;
    /** Per texel mean over the measured frames, for sigma. */
    readonly texelMeans: Float64Array;
    readonly activeTexels: Uint8Array;
    readonly atlasMean: number;
    readonly atlasChangeRms: number;
    readonly atlasDeviationRms: number;
}

async function runExperiment(context: ExperimentContext, experiment: Experiment): Promise<ExperimentResult> {
    const { scene, log } = context;
    applyExperiment(context, experiment);
    await fullPasses(scene, experiment.settlePasses ?? DEFAULT_SETTLE_PASSES);
    const units = experiment.units === 'all' ? scene.probeCount : experiment.units;

    const field = scene.ddgi.field;
    assert(field, 'DDGI has no field');
    const [nx] = field.grid.counts;
    const texelsPerSide = scene.ddgi.settings.irradianceTexels;
    const texelsPerProbe = texelsPerSide * texelsPerSide;

    let previousImage = (await renderIndirect(context)).indirect;
    let previousField = await readField(scene);
    const startStates = previousField.states;
    const pixelCount = WIDTH * HEIGHT;
    const surface = new Uint8Array(pixelCount);
    const pixelClass: PixelClass[] = new Array<PixelClass>(pixelCount).fill('other');
    let haveClasses = false;

    const pixelMoments = new RunningMoments(pixelCount);
    const texelMoments = new RunningMoments(previousField.texels.length);
    const imageChangeMeans: number[] = [];
    const imageChangeP95s: number[] = [];
    const classChangeSums = new Map<PixelClass, number>(CLASSES.map((name) => [name, 0]));
    const texelChangeMeans: number[] = [];
    const texelChangeP95s: number[] = [];
    const texelChangeSquares: number[] = [];
    const regionChangeSums = { red: 0, green: 0, rest: 0 };
    const stateChanges: number[] = [];
    const offsetChanges: number[] = [];
    const probesUpdated: number[] = [];
    const imageMeans: number[] = [];
    let surfaceMean = 0;
    let atlasMean = 0;
    const activeTexels = new Uint8Array(previousField.texels.length);

    for (let frame = 0; frame < MEASURE_FRAMES; frame++) {
        scene.frame({ units });
        probesUpdated.push(scene.ddgi.counters[PROBES_UPDATED] ?? 0);
        const { indirect, albedo } = await renderIndirect(context);
        const fieldNow = await readField(scene);

        if (frame === 0) {
            let sum = 0;
            let count = 0;
            for (let pixel = 0; pixel < pixelCount; pixel++) {
                const value = indirect[pixel * 3] + indirect[pixel * 3 + 1] + indirect[pixel * 3 + 2];
                surface[pixel] = value > 0 ? 1 : 0;
                if (value > 0) {
                    sum += value / 3;
                    count++;
                }
                if (albedo) pixelClass[pixel] = classify(albedo, pixel);
            }
            haveClasses = albedo !== undefined;
            surfaceMean = sum / count;
            assert(count > 0, 'the test view is black');
        }
        if (frame === 0) {
            // Texels of probes active at the start and now; the atlas mean over them.
            let sum = 0;
            let count = 0;
            for (let probe = 0; probe < startStates.length; probe++) {
                if (startStates[probe] !== STATE_ACTIVE || fieldNow.states[probe] !== STATE_ACTIVE) continue;
                for (let texel = 0; texel < texelsPerProbe; texel++) {
                    activeTexels[probe * texelsPerProbe + texel] = 1;
                    sum += fieldNow.texels[probe * texelsPerProbe + texel];
                    count++;
                }
            }
            atlasMean = sum / count;
        }

        // Image, frame to frame.
        const changes: number[] = [];
        const classSums = new Map<PixelClass, { change: number; value: number }>(CLASSES.map((name) => [name, { change: 0, value: 0 }]));
        const luminance = new Float32Array(pixelCount);
        for (let pixel = 0; pixel < pixelCount; pixel++) {
            luminance[pixel] = (indirect[pixel * 3] + indirect[pixel * 3 + 1] + indirect[pixel * 3 + 2]) / 3;
            if (!surface[pixel]) continue;
            let change = 0;
            for (let channel = 0; channel < 3; channel++) change += Math.abs(indirect[pixel * 3 + channel] - previousImage[pixel * 3 + channel]);
            change /= 3;
            changes.push(change / surfaceMean);
            const sums = classSums.get(pixelClass[pixel])!;
            sums.change += change;
            sums.value += luminance[pixel];
        }
        const sortedChanges = sortedCopy(changes);
        imageChangeMeans.push(mean(changes));
        imageChangeP95s.push(percentile(sortedChanges, 0.95));
        for (const name of CLASSES) {
            const sums = classSums.get(name)!;
            classChangeSums.set(name, classChangeSums.get(name)! + (sums.value > 0 ? sums.change / sums.value : 0));
        }
        pixelMoments.add(luminance);
        imageMeans.push(imageMean(indirect));

        // Atlas, frame to frame.
        const texelChanges: number[] = [];
        const regionFrame = { red: { sum: 0, count: 0 }, green: { sum: 0, count: 0 }, rest: { sum: 0, count: 0 } };
        for (let index = 0; index < fieldNow.texels.length; index++) {
            if (!activeTexels[index]) continue;
            const change = Math.abs(fieldNow.texels[index] - previousField.texels[index]) / atlasMean;
            texelChanges.push(change);
            const probe = Math.floor(index / texelsPerProbe);
            const x = probe % nx;
            const region = x === 0 ? regionFrame.red : x === nx - 1 ? regionFrame.green : regionFrame.rest;
            region.sum += change;
            region.count++;
        }
        texelChangeMeans.push(mean(texelChanges));
        texelChangeP95s.push(percentile(sortedCopy(texelChanges), 0.95));
        texelChangeSquares.push(rootMeanSquare(texelChanges) ** 2);
        regionChangeSums.red += regionFrame.red.sum / Math.max(1, regionFrame.red.count);
        regionChangeSums.green += regionFrame.green.sum / Math.max(1, regionFrame.green.count);
        regionChangeSums.rest += regionFrame.rest.sum / Math.max(1, regionFrame.rest.count);
        texelMoments.add(fieldNow.texels);

        // Probe states and offsets.
        let changedStates = 0;
        let movedOffsets = 0;
        for (let probe = 0; probe < fieldNow.states.length; probe++) {
            if (fieldNow.states[probe] !== previousField.states[probe]) changedStates++;
            for (let axis = 0; axis < 3; axis++) {
                if (fieldNow.offsets[probe * 3 + axis] !== previousField.offsets[probe * 3 + axis]) {
                    movedOffsets++;
                    break;
                }
            }
        }
        stateChanges.push(changedStates);
        offsetChanges.push(movedOffsets);

        previousImage = indirect;
        previousField = fieldNow;
    }

    // Over time: per-pixel and per-texel standard deviation.
    const pixelDeviations: number[] = [];
    const classDeviation = new Map<PixelClass, { sum: number; value: number }>(CLASSES.map((name) => [name, { sum: 0, value: 0 }]));
    for (let pixel = 0; pixel < pixelCount; pixel++) {
        if (!surface[pixel]) continue;
        const deviation = pixelMoments.standardDeviation(pixel);
        pixelDeviations.push(deviation / surfaceMean);
        const sums = classDeviation.get(pixelClass[pixel])!;
        sums.sum += deviation;
        sums.value += pixelMoments.mean[pixel];
    }
    const texelDeviations: number[] = [];
    const regionDeviation = { red: [] as number[], green: [] as number[], rest: [] as number[] };
    for (let index = 0; index < activeTexels.length; index++) {
        if (!activeTexels[index]) continue;
        const deviation = texelMoments.standardDeviation(index) / atlasMean;
        texelDeviations.push(deviation);
        const x = Math.floor(index / texelsPerProbe) % nx;
        (x === 0 ? regionDeviation.red : x === nx - 1 ? regionDeviation.green : regionDeviation.rest).push(deviation);
    }
    const frames = MEASURE_FRAMES;
    const firstMeans = mean(imageMeans.slice(0, 10));
    const lastMeans = mean(imageMeans.slice(-10));
    const percent = (value: number): number => round(value * 100);
    const row: Record<string, unknown> = {
        experiment: experiment.name,
        units_per_frame: round(mean(probesUpdated)),
        share_per_frame: round(mean(probesUpdated) / scene.probeCount),
        image_change_mean_pct: percent(mean(imageChangeMeans)),
        image_change_p95_pct: percent(mean(imageChangeP95s)),
        image_deviation_mean_pct: percent(mean(pixelDeviations)),
        image_deviation_p95_pct: percent(percentile(sortedCopy(pixelDeviations), 0.95)),
        atlas_change_mean_pct: percent(mean(texelChangeMeans)),
        atlas_change_p95_pct: percent(mean(texelChangeP95s)),
        atlas_change_rms_pct: percent(Math.sqrt(mean(texelChangeSquares))),
        atlas_deviation_rms_pct: percent(rootMeanSquare(texelDeviations)),
        atlas_deviation_p95_pct: percent(percentile(sortedCopy(texelDeviations), 0.95)),
        atlas_change_red_wall_probes_pct: percent(regionChangeSums.red / frames),
        atlas_change_green_wall_probes_pct: percent(regionChangeSums.green / frames),
        atlas_change_other_probes_pct: percent(regionChangeSums.rest / frames),
        atlas_deviation_red_wall_probes_rms_pct: percent(rootMeanSquare(regionDeviation.red)),
        atlas_deviation_green_wall_probes_rms_pct: percent(rootMeanSquare(regionDeviation.green)),
        atlas_deviation_other_probes_rms_pct: percent(rootMeanSquare(regionDeviation.rest)),
        state_changes_per_frame: round(mean(stateChanges)),
        offset_changes_per_frame: round(mean(offsetChanges)),
        image_mean: round(mean(imageMeans), 6),
        image_mean_drift_pct: percent(lastMeans / firstMeans - 1),
        atlas_mean: round(atlasMean, 6),
        active_probes: round(mean(Array.from(startStates, (state) => (state === STATE_ACTIVE ? 1 : 0))) * scene.probeCount),
    };
    if (haveClasses) {
        for (const name of CLASSES) {
            if (name === 'other') continue;
            row[`image_change_${name}_pct`] = percent(classChangeSums.get(name)! / frames);
            const deviation = classDeviation.get(name)!;
            row[`image_deviation_${name}_pct`] = percent(deviation.value > 0 ? deviation.sum / deviation.value : 0);
        }
    }
    log(
        `${experiment.name}: image change ${row.image_change_mean_pct}% / p95 ${row.image_change_p95_pct}%, over time ${row.image_deviation_mean_pct}%; ` +
            `atlas change ${row.atlas_change_mean_pct}% (rms ${row.atlas_change_rms_pct}%), over time rms ${row.atlas_deviation_rms_pct}%; ` +
            `walls red ${row.image_change_red_pct ?? '-'}% green ${row.image_change_green_pct ?? '-'}% white ${row.image_change_white_pct ?? '-'}%; ` +
            `state changes ${row.state_changes_per_frame}/frame, offsets ${row.offset_changes_per_frame}/frame; mean ${row.image_mean}`,
    );
    return {
        row,
        texelMeans: texelMoments.mean,
        activeTexels,
        atlasMean,
        atlasChangeRms: Math.sqrt(mean(texelChangeSquares)),
        atlasDeviationRms: rootMeanSquare(texelDeviations),
    };
}

/** One update of every probe at hysteresis 0 shows each texel's single-update estimate. */
async function measureSigma(context: ExperimentContext, base: ExperimentResult): Promise<Record<string, unknown>> {
    const { scene, log } = context;
    const hysteresis = context.base.hysteresis;
    const sigmas: number[] = [];
    const relativeSigmas: number[] = [];
    for (let repeat = 0; repeat < SIGMA_REPEATS; repeat++) {
        if (repeat > 0) {
            applyExperiment(context, null);
            await fullPasses(scene, DEFAULT_SETTLE_PASSES);
        }
        Object.assign(scene.ddgi.settings, { hysteresis: 0, maxBrightnessChange: CLAMP_OFF });
        scene.frame({ units: scene.probeCount });
        const estimate = await readField(scene);
        const differences: number[] = [];
        const relative: number[] = [];
        for (let index = 0; index < estimate.texels.length; index++) {
            if (!base.activeTexels[index]) continue;
            const difference = estimate.texels[index] - base.texelMeans[index];
            differences.push(difference / base.atlasMean);
            if (base.texelMeans[index] > 0.05 * base.atlasMean) relative.push(difference / base.texelMeans[index]);
        }
        sigmas.push(rootMeanSquare(differences));
        relativeSigmas.push(rootMeanSquare(relative));
    }
    applyExperiment(context, null);
    await fullPasses(scene, DEFAULT_SETTLE_PASSES);
    const sigma = Math.sqrt(mean(sigmas.map((value) => value * value)));
    const predictedDeviation = sigma * Math.sqrt((1 - hysteresis) / (1 + hysteresis));
    const predictedChange = sigma * (1 - hysteresis) * Math.sqrt(2 / (1 + hysteresis));
    const result = {
        hysteresis,
        sigma_rms_pct: round(sigma * 100),
        sigma_per_repeat_pct: sigmas.map((value) => round(value * 100)),
        sigma_relative_to_own_texel_rms_pct: round(Math.sqrt(mean(relativeSigmas.map((value) => value * value))) * 100),
        predicted_deviation_rms_pct: round(predictedDeviation * 100),
        measured_deviation_rms_pct: round(base.atlasDeviationRms * 100),
        predicted_change_rms_pct: round(predictedChange * 100),
        measured_change_rms_pct: round(base.atlasChangeRms * 100),
    };
    log(
        `sigma: one update's texel noise ${result.sigma_rms_pct}% of the atlas mean (${result.sigma_relative_to_own_texel_rms_pct}% of its own texel); ` +
            `over time predicted ${result.predicted_deviation_rms_pct}% against ${result.measured_deviation_rms_pct}%; per frame predicted ${result.predicted_change_rms_pct}% against ${result.measured_change_rms_pct}%`,
    );
    return result;
}

await runThreePage(async ({ device, data, log }) => {
    data.userAgent = navigator.userAgent;
    data.browserVersion = await fullBrowserVersion();
    const manifest = loadAllManifests().find((candidate) => candidate.id === SCENE);
    assert(manifest, `no manifest '${SCENE}'`);
    const bookmark = manifest.cameras[CAMERA - 1];
    assert(bookmark, `${SCENE} has no camera ${CAMERA}`);
    data.scene = SCENE;
    data.camera = bookmark.name;
    const override = installUniformOverride(device);

    const scene = await startViewerScene(device, manifest, { gpuTiming: false, label: `flicker causes ${SCENE}` });
    try {
        const field = scene.ddgi.field;
        assert(field, 'DDGI has no field');
        data.probes = scene.probeCount;
        const view = await TestView.create(device, scene.platform);
        await warmUp(scene);

        let blockMean = imageMean((await view.render(field, bookmark, WIDTH, HEIGHT)).indirect);
        let settled = false;
        for (let block = 1; block <= MAX_BLOCKS && !settled; block++) {
            await fullPasses(scene, BLOCK_PASSES);
            const next = imageMean((await view.render(field, bookmark, WIDTH, HEIGHT)).indirect);
            settled = block >= MIN_BLOCKS && Math.abs(next / blockMean - 1) < CONVERGED_TOLERANCE;
            blockMean = next;
        }
        assert(settled, 'the GI image mean did not settle');

        const settings = scene.ddgi.settings;
        const context: ExperimentContext = {
            scene,
            view,
            bookmark,
            override,
            base: { hysteresis: settings.hysteresis, raysPerProbe: settings.raysPerProbe, maxBrightnessChange: settings.maxBrightnessChange, bounceEnergy: settings.bounceEnergy },
            log,
        };
        const rows: Array<Record<string, unknown>> = [];
        data.experiments = rows;
        for (const experiment of EXPERIMENTS.filter((candidate) => SELECTED.includes(candidate.name))) {
            const result = await runExperiment(context, experiment);
            rows.push(result.row);
            applyExperiment(context, null);
            if (experiment.sigma) data.sigma = await measureSigma(context, result);
        }
    } finally {
        scene.dispose();
    }
});
