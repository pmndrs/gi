/**
 * P6.3, experiment E10: the GPU time of the radiance cache's world cache per
 * frame with GI-1.2's second-bounce rays (`secondBounce: 'gi1-rays'`) against
 * the second bounce read from a DDGI volume stacked under it
 * (`'ddgi-volume'`), on Sponza (with curtains) and Bistro exterior at
 * 1280 × 720 from each scene's first bookmark, the camera still. DDGI runs in
 * both (on the manifest's volume, `DDGI_UNITS` probes a frame), so its own
 * update cost is the same on both sides and is listed separately. A
 * measurement (DECISIONS 2026-11-06/09: no timing gates); the page fails only
 * on device errors or when a setting records no timed frame.
 *
 * The scenes come through the viewer's loader and the three adapter's scene
 * sync; the `FrameScreen` is drawn by primary rays. Both techniques get the
 * page's `GpuTimer` in their context (radiance-cache-cell-cost's way: the
 * platform's timer resolves before the screen stage). Per scene: `WARM_FRAMES`
 * frames, then per setting `SETTLE_FRAMES` frames and `TIMED_FRAMES` timed
 * frames: the mean of each world-cache section, their sum, the probe trace,
 * the radiance cache's total, DDGI's total and its passes, and the frame's
 * counts; and each technique's GPU memory.
 *
 * Through the lock (the runner prints its VALID / INVALID timing verdict):
 * `node tools/reference/bench-lock.ts run -- pnpm test:gpu --filter radiance-cache-ddgi-bounce-cost --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { GITechnique, TechniqueContext } from '../../../../packages/ddgi-core/src/platform/contract/technique.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { GpuTimer } from '../../../../packages/ddgi-core/src/platform/timing/gpu-timer.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const WARM_FRAMES = 60;
const SETTLE_FRAMES = 30;
const TIMED_FRAMES = 60;
/** DDGI's probes updated a frame (its `maxProbesPerFrame` default is 4,096). */
const DDGI_UNITS = 2048;
const SETTINGS: readonly Pick<RadianceCacheSettings, 'secondBounce'>[] = [{ secondBounce: 'gi1-rays' }, { secondBounce: 'ddgi-volume' }];

/** The world cache's timed sections (hash-grid-cache.ts), in their order; E10's in its slot. */
const WORLD_CACHE_SECTIONS = [
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
] as const;

async function describeLoaded(platform: GIPlatformImpl, device: GPUDevice, manifest: SceneManifest): Promise<() => void> {
    const loaded = await loadScene(manifest, () => {});
    const content = new Set<object>();
    loaded.content.traverse((object) => content.add(object));
    loaded.scene.traverse((object) => {
        if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    loaded.scene.updateMatrixWorld();
    const sync = new SceneSync(platform.scene, { device, warn: () => {} });
    sync.environment = platformSkyEnvironment(manifest);
    return () => sync.update(loaded.scene);
}

async function readCounters(device: GPUDevice, buffer: GPUBuffer): Promise<Uint32Array> {
    const readback = device.createBuffer({ label: 'ddgi-bounce-cost counters', size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'ddgi-bounce-cost counters' });
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, 256);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    return words;
}

const round = (value: number, digits = 3): number => Number(value.toFixed(digits));

/** Hands the technique a context whose `timer` is the page's (the context's other members, getters included, as they are). */
function withTimer(technique: GITechnique, timer: GpuTimer): void {
    const attach = technique.attach.bind(technique);
    technique.attach = (context: TechniqueContext) => attach(Object.create(context, { timer: { value: timer } }) as TechniqueContext);
}

async function measureScene(device: GPUDevice, id: string): Promise<Record<string, unknown>> {
    const manifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const timer = new GpuTimer(device, { slotCount: 8, maxPassesPerFrame: 96 });
    await timer.ready;
    if (!timer.enabled) throw new Error(`the GPU timer is ${timer.status}: timestamp-query is needed`);
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-ddgi-bounce-cost ${id}` });
    const technique = new RadianceCacheTechnique();
    const ddgi = new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing } });
    withTimer(technique, timer);
    withTimer(ddgi, timer);
    platform.addTechnique(technique);
    platform.addTechnique(ddgi);
    platform.setOutput(technique);
    // E10's host arrangement: DDGI stacked under the radiance cache (encoded first, its shading module the cache's bounce source).
    platform.setFallback(technique, ddgi);
    await platform.prepare();
    const beforeFrame = await describeLoaded(platform, device, manifest);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    let frameNumber = 0;
    const frame = (): void => {
        beforeFrame();
        timer.beginFrame(++frameNumber);
        const encoder = device.createCommandEncoder({ label: `radiance-cache-ddgi-bounce-cost ${id}` });
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: DDGI_UNITS }, camera: frameCamera(camera, WIDTH, HEIGHT) };
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        timer.resolve(encoder);
        device.queue.submit([encoder.finish()]);
        timer.readback();
    };
    try {
        frame();
        await platform.rays.whenBuilt();
        technique.resetHistory();
        for (let index = 0; index < WARM_FRAMES; index++) {
            frame();
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        const cache = technique.worldCache;
        if (!cache) throw new Error('no world cache');
        const results: Record<string, unknown> = {};
        for (const setting of SETTINGS) {
            Object.assign(technique.settings, setting);
            for (let index = 0; index < SETTLE_FRAMES; index++) {
                frame();
                if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
            }
            await timer.drain();
            timer.takeSamples();
            const sums = new Map<string, number>();
            let timedFrames = 0;
            for (let index = 0; index < TIMED_FRAMES; index++) {
                await timer.waitForFreeSlot();
                frame();
            }
            await timer.drain();
            for (const sample of timer.takeSamples()) {
                timedFrames++;
                for (const pass of sample.passes) sums.set(pass.name, (sums.get(pass.name) ?? 0) + pass.gpuMs);
            }
            if (timedFrames === 0) throw new Error(`${id}, ${setting.secondBounce}: no timed frame`);
            if (cache.secondBounceInUse !== setting.secondBounce) throw new Error(`${id}: asked for ${setting.secondBounce}, the cache ran ${cache.secondBounceInUse}`);
            const counters = await readCounters(device, cache.buffers.tileWords);
            const mean = (name: string): number => (sums.get(name) ?? 0) / timedFrames;
            const prefixed = (prefix: string): string[] => [...sums.keys()].filter((name) => name.startsWith(prefix));
            const sections = Object.fromEntries(WORLD_CACHE_SECTIONS.map((section) => [section, round(mean(`radiance-cache/${section}`))]));
            const worldCacheMs = WORLD_CACHE_SECTIONS.reduce((sum, section) => sum + mean(`radiance-cache/${section}`), 0);
            const radianceCacheMs = prefixed('radiance-cache/').reduce((sum, name) => sum + mean(name), 0);
            const ddgiPasses = prefixed('ddgi/');
            results[setting.secondBounce] = {
                worldCacheMs: round(worldCacheMs),
                probeTraceMs: round(mean('radiance-cache/sample-populate')),
                radianceCacheMs: round(radianceCacheMs),
                ddgiMs: round(ddgiPasses.reduce((sum, name) => sum + mean(name), 0)),
                ddgiPasses: Object.fromEntries(ddgiPasses.map((name) => [name, round(mean(name))])),
                timedFrames,
                sections,
                counts: {
                    firstBounceEntries: counters[HASH_GRID_COUNTER.visibilityCount0],
                    secondBounceEntries: counters[HASH_GRID_COUNTER.visibilityCount1],
                    shadowRays: counters[HASH_GRID_COUNTER.visibilityRayCount],
                },
            };
        }
        results.memoryMiB = {
            radianceCache: round(technique.memoryBytes / 2 ** 20, 2),
            worldCache: round(cache.memoryBytes / 2 ** 20, 2),
            ddgi: round(ddgi.memoryBytes / 2 ** 20, 2),
            ddgiProbes: ddgi.field?.volumes.reduce((sum, volume) => sum + volume.grid.probeCount, 0) ?? null,
        };
        return results;
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
    const scenes: Record<string, unknown> = {};
    for (const id of ['sponza', 'bistro-exterior']) scenes[id] = await measureScene(device, id);
    await drainDevice(device);
    reportGpuTest({
        status: deviceErrors.length === 0 ? 'pass' : 'fail',
        data: { width: WIDTH, height: HEIGHT, ddgiUnits: DDGI_UNITS, warmFrames: WARM_FRAMES, settleFrames: SETTLE_FRAMES, timedFrames: TIMED_FRAMES, scenes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
