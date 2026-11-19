/**
 * P6.4 (register E11): GPU time and memory of the neural tail beside the
 * GI-1.2 hash grid it enhances, on `sponza` (with curtains) and
 * `bistro-exterior` at 1280 × 720 from each scene's first bookmark, the
 * camera still: GI-1.2's multibounce (`neuralCache: 'off'`), the tail at its
 * defaults (the compact network, 64 → 3 × 32 → 3, f16, 2 steps a frame), the
 * tail in f32, and the tail with the pathtracer's default network (64 → 5 ×
 * 64 → 3). The device is the platform's, which asks for `shader-f16` and the
 * adapter's workgroup memory (32 KiB on the RTX 3060; DECISIONS 2026-11-20).
 *
 * Each configuration warms for `WARM_FRAMES` frames from an empty cache and a
 * fresh network (the gate opens at frame 17), then times `TIMED_FRAMES`
 * frames with a `GpuTimer` of this page's (the platform's own resolves before
 * the screen stage; NOTES). Per frame it sums the hash grid's passes, the
 * tail's inference (`NeuralTailInference`), its training steps
 * (`NeuralTailTraining`, every step's pass) and its gathering
 * (`NeuralTailGather`: cell records and queries; `NeuralTailScreenRecords`),
 * and every pass of the technique; it reports their medians and the GPU
 * memory of the grid, the tail (network, optimizer and scratch, training set,
 * queries) and the technique.
 *
 * Recorded, not a gate (DECISIONS 2026-11-06); it fails only on errors or
 * frames without timings. Timings need the GPU lock and its VALID verdict,
 * with the cool wait:
 * `node tools/reference/bench-lock.ts run --what "p6.4 neural timing" -- pnpm test:gpu --filter radiance-cache-neural-timing --gpu nvidia`,
 * with `VITE_DDGI_MAX_TEXTURE_SIZE=1024` on a 6 GB GPU (Bistro's textures).
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { TechniqueContext } from '../../../../packages/ddgi-core/src/platform/contract/technique.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { GpuTimer } from '../../../../packages/ddgi-core/src/platform/timing/gpu-timer.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { loadManifest, readBuffer, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const WARM_FRAMES = 60;
const TIMED_FRAMES = 120;
const env = import.meta.env as Record<string, unknown>;
/** T2.17: scale model textures down to this side at load; undefined keeps them. */
const MAX_TEXTURE_SIZE = typeof env.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' ? Number(env.VITE_DDGI_MAX_TEXTURE_SIZE) : undefined;

const ALL_CONFIGURATIONS: readonly (readonly [string, Partial<RadianceCacheSettings>])[] = [
    ['gi1 multibounce', { useMultibounce: true, neuralCache: 'off' }],
    ['tail', { useMultibounce: true, neuralCache: 'tail' }],
    ['tail, f32', { useMultibounce: true, neuralCache: 'tail', neural: { precision: 'f32' } }],
    ['tail, default network', { useMultibounce: true, neuralCache: 'tail', neural: { network: 'default' } }],
];
/** `VITE_NEURAL_TIMING_CONFIGS`: a semicolon-separated subset of the names above (all by default). */
const CONFIG_FILTER = typeof env.VITE_NEURAL_TIMING_CONFIGS === 'string' ? env.VITE_NEURAL_TIMING_CONFIGS.split(';').map((name) => name.trim()) : null;
const CONFIGURATIONS = CONFIG_FILTER ? ALL_CONFIGURATIONS.filter(([name]) => CONFIG_FILTER.includes(name)) : ALL_CONFIGURATIONS;

/** The hash grid's timed passes (`radiance-cache/<section>`). */
const GRID_SECTIONS = new Set([
    'PurgeRadianceCache',
    'PopulateScreenProbesCells',
    'PopulateMultibounceCells',
    'GenerateReservoirs',
    'GenerateMultibounceReservoirs',
    'PopulateRadianceCache',
    'UpdateRadianceCache',
    'UpdateMultibounceCells',
    'ResolveCells',
]);
/** The tail's timed passes. */
const TAIL_SECTIONS = new Set(['NeuralTailGather', 'NeuralTailInference', 'NeuralTailScreenRecords', 'NeuralTailTraining']);

/** The technique with this page's timer in its context (the platform's own timer resolves before the screen stage). */
class TimedRadianceCache extends RadianceCacheTechnique {
    private readonly pageTimer: GpuTimer;

    constructor(settings: Partial<RadianceCacheSettings>, pageTimer: GpuTimer) {
        super(settings);
        this.pageTimer = pageTimer;
    }

    override attach(context: TechniqueContext): void {
        super.attach({ ...context, timer: this.pageTimer });
    }
}

function median(values: readonly number[]): number {
    if (values.length === 0) return NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

const round = (value: number, digits = 3): number => Number(value.toFixed(digits));
const mib = (bytes: number): number => round(bytes / 2 ** 20, 1);

interface Loaded {
    readonly manifest: SceneManifest;
    readonly scene: Awaited<ReturnType<typeof loadScene>>;
}

async function load(id: string): Promise<Loaded> {
    const manifest = loadManifest(id);
    const scene = await loadScene(manifest, () => {}, MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE });
    const content = new Set<object>();
    scene.content.traverse((object) => content.add(object));
    // The model only is GI geometry; the sky dome is the viewer's background (GI gets the sky as its environment).
    scene.scene.traverse((object) => {
        if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    scene.scene.updateMatrixWorld();
    return { manifest, scene };
}

async function timeConfiguration(device: GPUDevice, loaded: Loaded, name: string, settings: Partial<RadianceCacheSettings>): Promise<Record<string, unknown>> {
    const { manifest } = loaded;
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 2000 };
    const timer = new GpuTimer(device, { warn: (message) => console.warn(message) });
    await timer.ready;
    const platform = new GIPlatformImpl(device, { label: `neural timing ${manifest.id} ${name}` });
    const technique = platform.addTechnique(new TimedRadianceCache(settings, timer));
    platform.setOutput(technique);
    await platform.prepare();
    const sync = new SceneSync(platform.scene, { device, warn: () => {} });
    sync.environment = platformSkyEnvironment(manifest);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    let frameNumber = 0;
    const frame = async (timed: boolean): Promise<void> => {
        sync.update(loaded.scene.scene);
        if (timed) {
            await timer.waitForFreeSlot();
            timer.beginFrame(++frameNumber);
        }
        const encoder = device.createCommandEncoder({ label: `neural timing ${manifest.id}` });
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        if (timed) timer.resolve(encoder);
        device.queue.submit([encoder.finish()]);
        if (timed) timer.readback();
    };
    try {
        await frame(false);
        await platform.rays.whenBuilt();
        technique.resetHistory();
        for (let index = 0; index < WARM_FRAMES; index++) {
            await frame(false);
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();
        timer.takeSamples();
        for (let index = 0; index < TIMED_FRAMES; index++) await frame(true);
        await timer.drain();
        const samples = timer.takeSamples();
        const series = { grid: [] as number[], tail: [] as number[], inference: [] as number[], training: [] as number[], gather: [] as number[], technique: [] as number[] };
        const perPass = new Map<string, number[]>();
        for (const sample of samples) {
            const sums = { grid: 0, tail: 0, inference: 0, training: 0, gather: 0, technique: 0 };
            const passSums = new Map<string, number>();
            for (const pass of sample.passes) {
                sums.technique += pass.gpuMs;
                const section = pass.name.startsWith('radiance-cache/') ? pass.name.slice('radiance-cache/'.length) : pass.name;
                if (GRID_SECTIONS.has(section)) sums.grid += pass.gpuMs;
                if (TAIL_SECTIONS.has(section)) sums.tail += pass.gpuMs;
                if (section === 'NeuralTailInference') sums.inference += pass.gpuMs;
                if (section === 'NeuralTailTraining') sums.training += pass.gpuMs;
                if (section === 'NeuralTailGather' || section === 'NeuralTailScreenRecords') sums.gather += pass.gpuMs;
                if (GRID_SECTIONS.has(section) || TAIL_SECTIONS.has(section)) passSums.set(section, (passSums.get(section) ?? 0) + pass.gpuMs);
            }
            for (const key of Object.keys(series) as (keyof typeof series)[]) series[key].push(sums[key]);
            for (const [section, ms] of passSums) {
                const list = perPass.get(section) ?? [];
                list.push(ms);
                perPass.set(section, list);
            }
        }
        const grid = technique.worldCache;
        const tail = technique.neuralTail;
        const counts = grid ? new Uint32Array(await readBuffer(device, grid.buffers.tileWords, 8 * 4)) : null;
        return {
            timedFrames: samples.length,
            gridMsMedian: round(median(series.grid)),
            tailMsMedian: round(median(series.tail)),
            inferenceMsMedian: round(median(series.inference)),
            trainingMsMedian: round(median(series.training)),
            gatherMsMedian: round(median(series.gather)),
            techniqueMsMedian: round(median(series.technique)),
            perPassMedianMs: Object.fromEntries([...perPass].map(([section, values]) => [section, round(median(values))])),
            gridMemoryMiB: mib(grid?.memoryBytes ?? 0),
            tailMemoryMiB: mib(tail?.memoryBytes ?? 0),
            networkMemoryMiB: mib(tail?.network?.bufferBytes ?? 0),
            techniqueMemoryMiB: mib(technique.memoryBytes),
            workgroupStorageBytes: device.limits.maxComputeWorkgroupStorageSize,
            tail: tail ? { steps: tail.steps, gateOpen: tail.active, network: `${tail.network?.inferBackend} ${tail.network?.precision}`, weights: tail.network?.weightCount } : null,
            firstBounceEntries: counts?.[HASH_GRID_COUNTER.visibilityCount0] ?? null,
            secondBounceEntries: counts?.[HASH_GRID_COUNTER.visibilityCount1] ?? null,
            timerCounters: { ...timer.counters },
        };
    } finally {
        screen.dispose();
        platform.dispose();
        timer.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    // The platform's device: shader-f16 and the adapter's workgroup memory included (DECISIONS 2026-11-20).
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    if (!device.features.has('timestamp-query')) throw new Error('this page needs timestamp-query');
    const failures: string[] = [];
    const scenes: Record<string, Record<string, unknown>> = {};
    for (const id of ['sponza', 'bistro-exterior']) {
        const loaded = await load(id);
        scenes[id] = {};
        for (const [name, settings] of CONFIGURATIONS) {
            const result = await timeConfiguration(device, loaded, name, settings);
            scenes[id][name] = result;
            if (!(Number(result.timedFrames) > 0)) failures.push(`${id}, ${name}: no timed frames`);
        }
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { configurations: CONFIGURATIONS.map(([name]) => name), width: WIDTH, height: HEIGHT, warmFrames: WARM_FRAMES, timedFrames: TIMED_FRAMES, maxTextureSize: MAX_TEXTURE_SIZE ?? null, scenes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
