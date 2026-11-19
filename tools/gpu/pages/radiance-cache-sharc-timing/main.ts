/**
 * P6.1: GPU time and memory of the radiance cache's world-cache backends,
 * Capsaicin's GI-1.2 hash grid (multibounce on, its defaults) against the
 * SHaRC-style hash cache (its defaults: two bounce rays and the cache tail;
 * and one bounce ray with the cache tail, the cheaper setting), on `sponza`
 * (with curtains) and `bistro-exterior` at 1280 × 720 from each scene's first
 * bookmark, the camera still.
 *
 * Each configuration warms for `WARM_FRAMES` frames from an empty cache, then
 * times `TIMED_FRAMES` frames with the platform's `GpuTimer` (timestamp
 * queries per pass; the technique's context gets this page's timer, since the
 * platform's own resolves before the screen stage). Per frame it sums the
 * world cache's passes (GI-1.2: the purge, population, multibounce, update and
 * resolve sections; SHaRC: `SharcUpdate`, `SharcResolve`, `SharcLookup`) and
 * every pass of the technique; it reports their medians, each cache pass's
 * median, and the cache's and the technique's GPU memory.
 *
 * Recorded, not a gate (DECISIONS 2026-11-06: timing gates are deferred); it
 * fails only on errors or frames without timings. Timings need the GPU lock
 * and its VALID verdict (cool start, no throttling):
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "p6.1 sharc timing" -- pnpm test:gpu --filter radiance-cache-sharc-timing --gpu nvidia`,
 * with `VITE_DDGI_MAX_TEXTURE_SIZE=1024` on a 6 GB GPU (Bistro's textures,
 * as `radiance-cache-overflow`).
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { TechniqueContext } from '../../../../packages/ddgi-core/src/platform/contract/technique.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { GpuTimer } from '../../../../packages/ddgi-core/src/platform/timing/gpu-timer.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import type { SharcCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/sharc/sharc-cache.ts';
import { SHARC_COUNTER, SHARC_COUNTER_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/sharc/sharc-wgsl.ts';
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

const CONFIGURATIONS: readonly (readonly [string, Partial<RadianceCacheSettings>])[] = [
    ['gi1-hash-grid', { worldCache: 'gi1-hash-grid' }],
    ['sharc', { worldCache: 'sharc' }],
    ['sharc, one bounce ray', { worldCache: 'sharc', sharc: { bounceRays: 1 } }],
    ['sharc, every probe updating', { worldCache: 'sharc', sharc: { updateFraction: 1 } }],
];

/** The world cache's timed passes, by backend (`radiance-cache/<section>`). */
const CACHE_SECTIONS = new Set([
    'PurgeRadianceCache',
    'PopulateScreenProbesCells',
    'PopulateMultibounceCells',
    'GenerateReservoirs',
    'GenerateMultibounceReservoirs',
    'PopulateRadianceCache',
    'UpdateRadianceCache',
    'UpdateMultibounceCells',
    'ResolveCells',
    'SharcUpdate',
    'SharcResolve',
    'SharcLookup',
]);

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
    const platform = new GIPlatformImpl(device, { label: `sharc timing ${manifest.id} ${name}` });
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
        const encoder = device.createCommandEncoder({ label: `sharc timing ${manifest.id}` });
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
        const cacheMs: number[] = [];
        const techniqueMs: number[] = [];
        const perPass = new Map<string, number[]>();
        for (const sample of samples) {
            let cache = 0;
            let total = 0;
            for (const pass of sample.passes) {
                total += pass.gpuMs;
                const section = pass.name.startsWith('radiance-cache/') ? pass.name.slice('radiance-cache/'.length) : pass.name;
                if (CACHE_SECTIONS.has(section)) {
                    cache += pass.gpuMs;
                    const list = perPass.get(section) ?? [];
                    list.push(pass.gpuMs);
                    perPass.set(section, list);
                }
            }
            cacheMs.push(cache);
            techniqueMs.push(total);
        }
        const backend = technique.worldCacheBackend;
        const result: Record<string, unknown> = {
            timedFrames: samples.length,
            cacheMsMedian: round(median(cacheMs)),
            cacheMsMean: round(cacheMs.reduce((sum, value) => sum + value, 0) / Math.max(cacheMs.length, 1)),
            techniqueMsMedian: round(median(techniqueMs)),
            perPassMedianMs: Object.fromEntries([...perPass].map(([section, values]) => [section, round(median(values))])),
            cacheMemoryMiB: round((backend?.memoryBytes ?? 0) / 2 ** 20, 1),
            techniqueMemoryMiB: round(technique.memoryBytes / 2 ** 20, 1),
            timerCounters: { ...timer.counters },
        };
        if (backend?.id === 'sharc') {
            const sharc = backend as SharcCache;
            const counters = new Uint32Array(await readBuffer(device, sharc.buffers.accum, SHARC_COUNTER_WORDS * 4));
            const keys = new Uint32Array(await readBuffer(device, sharc.buffers.keys, sharc.entries * 4));
            let cellsInUse = 0;
            for (const key of keys) if (key !== 0) cellsInUse++;
            result.sharc = { cellsInUse, slots: sharc.entries, ...Object.fromEntries(Object.entries(SHARC_COUNTER).map(([counter, word]) => [counter, counters[word]])) };
        }
        return result;
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
        data: { width: WIDTH, height: HEIGHT, warmFrames: WARM_FRAMES, timedFrames: TIMED_FRAMES, maxTextureSize: MAX_TEXTURE_SIZE ?? null, scenes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
