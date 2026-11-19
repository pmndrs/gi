/**
 * P6.2, lever E8: the GPU time of the radiance cache's world-cache passes per
 * frame, cells lit per hit (GI-1.2) against on a budget (`cellShading:
 * 'budgeted'`, 5k, 20k, 80k cells), on Sponza (with curtains) and Bistro
 * exterior at 1280 × 720 from each scene's first bookmark, the camera still.
 * A measurement (DECISIONS 2026-11-06/09: no timing gates); the page fails
 * only on device errors or when a setting records no timed frame.
 *
 * The scenes come through the viewer's loader and the three adapter's scene
 * sync (as radiance-cache-stability loads Sponza); the `FrameScreen` is drawn
 * by primary rays (radiance-cache-probes/screen.ts). Each pass is timed with
 * the platform's `GpuTimer` (timestamp queries around each compute pass): the
 * page hands the technique a context whose `timer` is the page's own, since
 * the platform's timer resolves its queries at the end of `encode`, before
 * the screen stage where the radiance cache works.
 *
 * Per scene: `WARM_FRAMES` frames per hit, then for each setting
 * `SETTLE_FRAMES` frames, then `TIMED_FRAMES` timed frames: the mean time of
 * each world-cache section (`radiance-cache/<section>`, Capsaicin's
 * `TimedSection` names and the lever's `ClaimCells`, `PrioritizeCells`,
 * `ShadeCells`) and their sum; the probe trace (`radiance-cache/sample-populate`)
 * and the screen passes beside them; and the last frame's counts: probe-ray
 * entries, second-bounce entries, shadow rays, cells hit and cells lit.
 *
 * Through the lock (the runner prints its VALID / INVALID timing verdict):
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-cell-cost --gpu nvidia`.
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
import { loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const WARM_FRAMES = 60;
const SETTLE_FRAMES = 30;
const TIMED_FRAMES = 60;
/** Per hit, then each budget (cells lit a frame). */
const SETTINGS: readonly Pick<RadianceCacheSettings, 'cellShading' | 'cellShadingBudget'>[] = [
    { cellShading: 'per-hit', cellShadingBudget: 0 },
    ...[5000, 20000, 80000].map((budget) => ({ cellShading: 'budgeted' as const, cellShadingBudget: budget })),
];

/** The world cache's timed sections (hash-grid-cache.ts), in their order. */
const WORLD_CACHE_SECTIONS = [
    'PurgeRadianceCache',
    'PopulateScreenProbesCells',
    'PopulateMultibounceCells',
    'GenerateReservoirs',
    'GenerateMultibounceReservoirs',
    'PopulateRadianceCache',
    'ClaimCells',
    'PrioritizeCells',
    'ShadeCells',
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
    const readback = device.createBuffer({ label: 'cell-cost counters', size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'cell-cost counters' });
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, 256);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    return words;
}

const round = (value: number, digits = 3): number => Number(value.toFixed(digits));

async function measureScene(device: GPUDevice, id: string): Promise<Record<string, unknown>> {
    const manifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const timer = new GpuTimer(device, { slotCount: 8, maxPassesPerFrame: 64 });
    await timer.ready;
    if (!timer.enabled) throw new Error(`the GPU timer is ${timer.status}: timestamp-query is needed`);
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-cell-cost ${id}` });
    const technique = new RadianceCacheTechnique();
    // The technique's context with the page's timer (the context's other members, getters included, as they are).
    const attach = technique.attach.bind(technique);
    technique.attach = (context: TechniqueContext) => attach(Object.create(context, { timer: { value: timer } }) as TechniqueContext);
    platform.addTechnique(technique);
    platform.setOutput(technique);
    await platform.prepare();
    const beforeFrame = await describeLoaded(platform, device, manifest);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    let frameNumber = 0;
    const frame = (): void => {
        beforeFrame();
        timer.beginFrame(++frameNumber);
        const encoder = device.createCommandEncoder({ label: `radiance-cache-cell-cost ${id}` });
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
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
            if (timedFrames === 0) throw new Error(`${id}, ${JSON.stringify(setting)}: no timed frame`);
            const counters = await readCounters(device, cache.buffers.tileWords);
            const mean = (name: string): number => (sums.get(name) ?? 0) / timedFrames;
            const sections = Object.fromEntries(WORLD_CACHE_SECTIONS.map((section) => [section, round(mean(`radiance-cache/${section}`))]));
            const worldCacheMs = WORLD_CACHE_SECTIONS.reduce((sum, section) => sum + mean(`radiance-cache/${section}`), 0);
            const allMs = [...sums.keys()].reduce((sum, name) => sum + mean(name), 0);
            const name = setting.cellShading === 'per-hit' ? 'per-hit' : `budgeted ${setting.cellShadingBudget / 1000}k`;
            results[name] = {
                worldCacheMs: round(worldCacheMs),
                probeTraceMs: round(mean('radiance-cache/sample-populate')),
                techniqueMs: round(allMs),
                timedFrames,
                sections,
                counts: {
                    firstBounceEntries: counters[HASH_GRID_COUNTER.visibilityCount0],
                    secondBounceEntries: counters[HASH_GRID_COUNTER.visibilityCount1],
                    shadowRays: counters[HASH_GRID_COUNTER.visibilityRayCount],
                    cellsHit: setting.cellShading === 'per-hit' ? null : counters[HASH_GRID_COUNTER.cellCandidateCount],
                    cellsLit: setting.cellShading === 'per-hit' ? null : counters[HASH_GRID_COUNTER.cellShadedCount],
                },
            };
        }
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
        data: { width: WIDTH, height: HEIGHT, warmFrames: WARM_FRAMES, settleFrames: SETTLE_FRAMES, timedFrames: TIMED_FRAMES, scenes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
