/**
 * P6.5: GPU time and memory of the radiance cache's world-cache variants (the
 * compare panel's presets, `RADIANCE_CACHE_VARIANT_PRESETS`), beside
 * radiance-cache-variants (accuracy, lag, flicker). Recorded, not a gate
 * (DECISIONS 2026-11-06/09: no timing gates); it fails only on device errors
 * or a variant with no timed frame.
 *
 * Per scene (its first bookmark, the camera still, 1280 × 720 as the other
 * radiance-cache timing pages), per variant: the platform, the radiance cache
 * at the preset's values and, unless the preset runs the cache alone, the
 * manifest's DDGI volume stacked under it at the viewer's 256 probes a frame
 * (`STACKED_DDGI_PROBES_PER_FRAME`). `WARM_FRAMES` frames from nothing (DDGI's
 * field fills at that rate too), then `TIMED_FRAMES` timed frames with the
 * page's `GpuTimer` in both techniques' contexts (radiance-cache-ddgi-bounce-cost's
 * way: the platform's own timer resolves before the screen stage). Reported:
 * per frame, the radiance cache's passes (all of them, and the world cache's
 * own), DDGI's passes, and their sum, as medians and means; each technique's
 * GPU memory.
 *
 * The scenes come through the viewer's loader and the three adapter's scene
 * sync (no three renderer); the `FrameScreen` is drawn by primary rays. The
 * numbers count only with the lock's VALID verdict (cool start, no
 * throttling), so run it with the cool wait:
 * `node tools/reference/bench-lock.ts run --what "p6.5 variants timing" --clock-file bench/results/rc-variants-timing.clock.json -- pnpm test:gpu --filter radiance-cache-variants-timing --gpu nvidia --json bench/results/rc-variants-timing.json`,
 * then `node tools/reference/rc-variants-table.ts` reads the clock file for the verdict.
 *
 * `VITE_RC_VARIANTS` and `VITE_RC_VARIANT_SCENES` pick a subset (comma-separated).
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { GITechnique, TechniqueContext } from '../../../../packages/core/src/platform/contract/technique.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { GpuTimer } from '../../../../packages/core/src/platform/timing/gpu-timer.ts';
import { DDGITechnique } from '../../../../packages/ddgi/src/ddgi-technique.ts';
import { hostCompareValue, RADIANCE_CACHE_VARIANT_PRESETS, setCompareValues, type RadianceCacheVariantPreset } from '../../../../packages/cache/src/compare-toggles.ts';
import { RadianceCacheTechnique } from '../../../../packages/cache/src/radiance-cache-technique.ts';
import { SceneSync } from '../../../../packages/three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { STACKED_DDGI_PROBES_PER_FRAME } from '../../../../apps/viewer/src/techniques/radiance-cache-technique.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { PAGE_COMPARE_VALUES } from '../radiance-cache-probes/compare-env.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
/** As the variants page: DDGI under the cache needs about 100 frames at 256 probes a frame to fill Bistro exterior's field once. */
const env = import.meta.env as Record<string, unknown>;
const numberFromEnv = (name: string, fallback: number): number => (typeof env[name] === 'string' && env[name] ? Number(env[name]) : fallback);
/**
 * Frames before timing, and timed frames: `VITE_RC_VARIANT_TIMING_WARM_FRAMES` and `_TIMED_FRAMES`.
 * The split runs (tools/reference/rc-variants-timing-split.ts) shorten both so a run's GPU load
 * stays inside the clock rule's 10-15 s; the whole set at these defaults throttles the RTX 3060
 * Laptop (2026-11-28, INVALID).
 */
const WARM_FRAMES = numberFromEnv('VITE_RC_VARIANT_TIMING_WARM_FRAMES', 240);
const TIMED_FRAMES = numberFromEnv('VITE_RC_VARIANT_TIMING_TIMED_FRAMES', 120);
/** `VITE_RC_VARIANT_TIMING_PAUSE_MS`: idle this long before the timed frames (0 by default). */
const PAUSE_BEFORE_TIMED_MS = numberFromEnv('VITE_RC_VARIANT_TIMING_PAUSE_MS', 0);
const listFromEnv = (name: string): string[] | null => (typeof env[name] === 'string' && env[name] ? (env[name] as string).split(',').map((item) => item.trim()).filter(Boolean) : null);
const SCENES = listFromEnv('VITE_RC_VARIANT_SCENES') ?? ['cornell-box-point', 'cornell-box', 'two-room', 'sponza', 'bistro-interior', 'bistro-exterior', 'furnished-room'];
const variantIds = listFromEnv('VITE_RC_VARIANTS');
const VARIANTS = variantIds ? RADIANCE_CACHE_VARIANT_PRESETS.filter((preset) => variantIds.includes(preset.id)) : RADIANCE_CACHE_VARIANT_PRESETS;
/** T2.17: Bistro's textures capped where they don't fit the 6 GB RTX (as the accuracy pages). */
const TEXTURE_CAPS: Record<string, number> = { 'bistro-exterior': 1024, 'bistro-interior': 1024 };

/** The world cache's timed sections, both backends (hash-grid-cache.ts, sharc-cache.ts). */
const WORLD_CACHE_SECTIONS = new Set([
    'PurgeRadianceCache',
    'PopulateScreenProbesCells',
    'PopulateMultibounceCells',
    'GenerateReservoirs',
    'GenerateMultibounceReservoirs',
    'PopulateRadianceCache',
    'SampleBounceSourceCells',
    'UpdateRadianceCache',
    'UpdateMultibounceCells',
    'ResolveCells',
    'ClaimCells',
    'PrioritizeCells',
    'ShadeCells',
    'SharcUpdate',
    'SharcBegin',
    'SharcBounce1',
    'SharcBounce2',
    'SharcLight',
    'SharcCombine',
    'SharcResolve',
    'SharcLookup',
]);

const round = (value: number, digits = 3): number => Number(value.toFixed(digits));

function median(values: readonly number[]): number {
    if (values.length === 0) return Number.NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

const mean = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);

/** Hands the technique a context whose `timer` is the page's (the context's other members, getters included, as they are). */
function withTimer(technique: GITechnique, timer: GpuTimer): void {
    const attach = technique.attach.bind(technique);
    technique.attach = (context: TechniqueContext) => attach(Object.create(context, { timer: { value: timer } }) as TechniqueContext);
}

interface Loaded {
    readonly manifest: SceneManifest;
    readonly scene: Awaited<ReturnType<typeof loadScene>>['scene'];
}

async function load(id: string): Promise<Loaded> {
    const manifest = loadManifest(id);
    const maxTextureSize = TEXTURE_CAPS[id];
    const loaded = await loadScene(manifest, () => {}, maxTextureSize === undefined ? {} : { maxTextureSize });
    const content = new Set<object>();
    loaded.content.traverse((object) => content.add(object));
    // The model only is GI geometry; the sky dome is the viewer's background (GI gets the sky as its environment).
    loaded.scene.traverse((object) => {
        if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
        const light = object as { isPointLight?: boolean; isSpotLight?: boolean; distance?: number };
        if (light.isPointLight || light.isSpotLight) light.distance = 0;
    });
    loaded.scene.updateMatrixWorld();
    return { manifest, scene: loaded.scene };
}

async function timeVariant(device: GPUDevice, loaded: Loaded, preset: RadianceCacheVariantPreset): Promise<Record<string, unknown>> {
    const { manifest } = loaded;
    const values = { ...preset.values, ...PAGE_COMPARE_VALUES };
    const stacked = !hostCompareValue(values, 'host.cacheAlone');
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 2000 };
    const timer = new GpuTimer(device, { slotCount: 8, maxPassesPerFrame: 128 });
    await timer.ready;
    if (!timer.enabled) throw new Error(`the GPU timer is ${timer.status}: timestamp-query is needed`);
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-variants-timing ${manifest.id} ${preset.id}` });
    const technique = new RadianceCacheTechnique({ useTemporalFeedback: values.useTemporalFeedback === true });
    setCompareValues(technique.settings, values);
    withTimer(technique, timer);
    platform.addTechnique(technique);
    platform.setOutput(technique);
    let ddgi: DDGITechnique | null = null;
    if (stacked) {
        ddgi = new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing } });
        withTimer(ddgi, timer);
        platform.addTechnique(ddgi);
        platform.setFallback(technique, ddgi);
    }
    await platform.prepare();
    const sync = new SceneSync(platform.scene, { device, warn: () => {} });
    sync.environment = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: stacked ? STACKED_DDGI_PROBES_PER_FRAME : 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    let frameNumber = 0;
    const frame = async (timed: boolean): Promise<void> => {
        sync.update(loaded.scene);
        if (timed) await timer.waitForFreeSlot();
        timer.beginFrame(++frameNumber);
        const encoder = device.createCommandEncoder({ label: `radiance-cache-variants-timing ${manifest.id}` });
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        timer.resolve(encoder);
        device.queue.submit([encoder.finish()]);
        timer.readback();
    };
    try {
        await frame(false);
        await platform.rays.whenBuilt();
        technique.resetHistory();
        for (let index = 0; index < WARM_FRAMES; index++) {
            await frame(false);
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        await timer.drain();
        timer.takeSamples();
        // The split runs idle here so the card cools between the warm-up and the timed frames
        // (it reaches its 87 C target about 8 s into a GPU-bound load from 50 C).
        if (PAUSE_BEFORE_TIMED_MS > 0) await new Promise((resolve) => setTimeout(resolve, PAUSE_BEFORE_TIMED_MS));
        for (let index = 0; index < TIMED_FRAMES; index++) await frame(true);
        await timer.drain();
        const samples = timer.takeSamples();
        const radianceCacheMs: number[] = [];
        const worldCacheMs: number[] = [];
        const ddgiMs: number[] = [];
        const neuralMs: number[] = [];
        const perPass = new Map<string, number[]>();
        for (const sample of samples) {
            let radianceCache = 0;
            let worldCache = 0;
            let ddgiTotal = 0;
            let neural = 0;
            for (const pass of sample.passes) {
                if (pass.name.startsWith('ddgi/')) ddgiTotal += pass.gpuMs;
                else if (pass.name.startsWith('radiance-cache/')) {
                    radianceCache += pass.gpuMs;
                    const section = pass.name.slice('radiance-cache/'.length);
                    if (WORLD_CACHE_SECTIONS.has(section)) worldCache += pass.gpuMs;
                    if (section.startsWith('NeuralTail')) neural += pass.gpuMs;
                }
                const list = perPass.get(pass.name) ?? [];
                list.push(pass.gpuMs);
                perPass.set(pass.name, list);
            }
            radianceCacheMs.push(radianceCache);
            worldCacheMs.push(worldCache);
            ddgiMs.push(ddgiTotal);
            neuralMs.push(neural);
        }
        const totalMs = radianceCacheMs.map((value, index) => value + ddgiMs[index]);
        return {
            timedFrames: samples.length,
            ddgiStacked: stacked,
            secondBounceInUse: technique.worldCache?.secondBounceInUse ?? null,
            neuralTailRan: technique.neuralTail !== null,
            totalMsMedian: round(median(totalMs)),
            totalMsMean: round(mean(totalMs)),
            radianceCacheMsMedian: round(median(radianceCacheMs)),
            worldCacheMsMedian: round(median(worldCacheMs)),
            ddgiMsMedian: round(median(ddgiMs)),
            neuralTailMsMedian: round(median(neuralMs)),
            perPassMedianMs: Object.fromEntries([...perPass].map(([name, list]) => [name, round(median(list))])),
            memoryMiB: {
                radianceCache: round(technique.memoryBytes / 2 ** 20, 1),
                worldCache: round((technique.worldCacheBackend?.memoryBytes ?? 0) / 2 ** 20, 1),
                ddgi: round((ddgi?.memoryBytes ?? 0) / 2 ** 20, 1),
            },
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
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    const scenes: Record<string, Record<string, unknown>> = {};
    for (const id of SCENES) {
        const loaded = await load(id);
        scenes[id] = {};
        for (const preset of VARIANTS) {
            const result = await timeVariant(device, loaded, preset);
            scenes[id][preset.id] = result;
            if (!(Number(result.timedFrames) > 0)) failures.push(`${id}, ${preset.id}: no timed frames`);
        }
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: {
            variants: VARIANTS.map((preset) => preset.id),
            compare: PAGE_COMPARE_VALUES,
            width: WIDTH,
            height: HEIGHT,
            warmFrames: WARM_FRAMES,
            timedFrames: TIMED_FRAMES,
            stackedDdgiProbesPerFrame: STACKED_DDGI_PROBES_PER_FRAME,
            shaderF16: device.features.has('shader-f16'),
            scenes,
        },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
