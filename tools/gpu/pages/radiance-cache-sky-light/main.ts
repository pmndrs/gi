/**
 * RC6.1: what the sky as a light at the world cache's hits (register D33)
 * does to an outdoor scene. On `sponza` (with curtains) and
 * `bistro-exterior`, from each scene's first bookmark at 1280 × 720, the
 * radiance cache at Capsaicin's defaults (multibounce on) runs 120 still
 * frames from an empty cache twice: with `environmentLightEnable` off (the
 * port before RC6.1) and on (Capsaicin's default), each on a fresh platform
 * over the same loaded scene. After each run the page reads back:
 *
 * - the interpolated irradiance image (`probes.irradiance`, before the
 *   denoiser): the mean E (rgb mean) over the image, background included
 *   (it is 0 in both runs), and over the pixels with any E;
 * - the probe rays' results: the mean light of the rays that hit (what the
 *   world cache gives them) and of those that miss (the sky, unchanged);
 * - the hash grid's lists: the hits that drew a light this frame, and how
 *   many of them drew the sky.
 *
 * Information for PROGRESS, not a gate: it fails only on errors (a device
 * error, a scene that doesn't load) or a value that isn't finite. Scenes load
 * as `radiance-cache-overflow` loads them. Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc6.1 sky light" -- pnpm test:gpu --filter radiance-cache-sky-light`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { GI_DIRECT_SKY_LIGHT } from '../../../../packages/ddgi-core/src/platform/contract/shade.ts';
import { RESERVOIR_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { PROBE_RAY_BYTES, PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-probes.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { assert, drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { loadManifest, readBuffer, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const FRAMES = 120;
const SCENES = ['sponza', 'bistro-exterior'] as const;
const env = import.meta.env as Record<string, unknown>;
/** T2.17: scale model textures down to this side at load; undefined keeps them. */
const MAX_TEXTURE_SIZE = typeof env.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' ? Number(env.VITE_DDGI_MAX_TEXTURE_SIZE) : undefined;

interface RunResult {
    /** Mean of E's rgb over the whole image (the background is 0). */
    readonly meanE: number;
    /** The same over the pixels with any E. */
    readonly meanEOverLitPixels: number;
    readonly litPixels: number;
    /** Mean light (rgb mean) of the probe rays that hit, and of those that miss. */
    readonly meanHitRayRadiance: number;
    readonly meanMissRayRadiance: number;
    readonly hitRays: number;
    readonly missRays: number;
    /** First-bounce hits this frame, the ones that drew a light, and those that drew the sky. */
    readonly hits: number;
    readonly hitsWithLight: number;
    readonly skyDraws: number;
}

type LoadedScene = Awaited<ReturnType<typeof loadScene>>;

async function runOnce(device: GPUDevice, manifest: SceneManifest, loaded: LoadedScene, environmentLightEnable: boolean): Promise<RunResult> {
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 2000 };
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-sky-light ${manifest.id}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique({ environmentLightEnable }));
    platform.setOutput(technique);
    await platform.prepare();
    const sync = new SceneSync(platform.scene, { device, warn: () => {} });
    sync.environment = platformSkyEnvironment(manifest);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frameInput: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    try {
        const frame = (): void => {
            sync.update(loaded.scene);
            const encoder = device.createCommandEncoder({ label: 'radiance-cache-sky-light frame' });
            platform.encode(encoder, frameInput);
            screen.encode(encoder, camera);
            platform.encodeScreen(encoder, screen.screen);
            device.queue.submit([encoder.finish()]);
        };
        // A frame places the scene; then the mesh trees, and the measured frames with them, from an empty cache.
        frame();
        await platform.rays.whenBuilt();
        technique.resetHistory();
        for (let index = 0; index < FRAMES; index++) {
            frame();
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();

        const probes = technique.probes;
        const cache = technique.worldCache;
        assert(probes && cache && cache.rayLists, `${manifest.id}: no probes or world cache after the frames`);

        const irradiance = halves(await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT));
        let sumE = 0;
        let litPixels = 0;
        let notFinite = 0;
        for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
            const e = (irradiance[pixel * 4] + irradiance[pixel * 4 + 1] + irradiance[pixel * 4 + 2]) / 3;
            if (!Number.isFinite(e)) {
                notFinite++;
                continue;
            }
            sumE += e;
            if (e > 0) litPixels++;
        }
        assert(notFinite === 0, `${manifest.id}: ${notFinite} irradiance pixels aren't finite`);

        const rayCount = new Uint32Array(await readBuffer(device, probes.probeRays, 4))[0];
        const rays = new DataView(await readBuffer(device, probes.probeRays, PROBE_RAYS_HEADER_BYTES + rayCount * PROBE_RAY_BYTES));
        const results = probes.rayScratch.rayResults;
        const resultHalves = halves(new Uint8Array(await readBuffer(device, results.buffer, (results.offset ?? 0) + rayCount * 8)), results.offset ?? 0);
        let hitSum = 0;
        let missSum = 0;
        let hitRays = 0;
        let missRays = 0;
        for (let ray = 0; ray < rayCount; ray++) {
            const hitT = rays.getFloat32(PROBE_RAYS_HEADER_BYTES + ray * PROBE_RAY_BYTES + 24, true);
            const light = (resultHalves[ray * 4] + resultHalves[ray * 4 + 1] + resultHalves[ray * 4 + 2]) / 3;
            assert(Number.isFinite(light), `${manifest.id}: ray ${ray}'s light isn't finite`);
            if (hitT >= 0) {
                hitSum += light;
                hitRays++;
            } else {
                missSum += light;
                missRays++;
            }
        }

        const counters = new Uint32Array(await readBuffer(device, cache.buffers.words, 64 * 4));
        const shadowRays = counters[HASH_GRID_COUNTER.visibilityRayCount];
        const layout = cache.rayLists.layout;
        const reservoirs = new Uint32Array(await readBuffer(device, cache.rayLists.buffer, (layout.reservoirBuffer + RESERVOIR_WORDS * shadowRays) * 4));
        let skyDraws = 0;
        for (let ray = 0; ray < shadowRays; ray++) {
            if (reservoirs[layout.reservoirBuffer + ray * RESERVOIR_WORDS] === GI_DIRECT_SKY_LIGHT) skyDraws++;
        }
        return {
            meanE: sumE / (WIDTH * HEIGHT),
            meanEOverLitPixels: sumE / Math.max(litPixels, 1),
            litPixels,
            meanHitRayRadiance: hitSum / Math.max(hitRays, 1),
            meanMissRayRadiance: missSum / Math.max(missRays, 1),
            hitRays,
            missRays,
            // The shadow-ray list holds first- and second-bounce entries (multibounce on).
            hits: counters[HASH_GRID_COUNTER.visibilityCount0] + counters[HASH_GRID_COUNTER.visibilityCount1],
            hitsWithLight: shadowRays,
            skyDraws,
        };
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

async function measureScene(device: GPUDevice, id: string): Promise<Record<string, unknown>> {
    const manifest = loadManifest(id);
    // The model only is GI geometry; the sky dome is the viewer's background (GI gets the sky as its environment).
    const loaded = await loadScene(manifest, () => {}, MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE });
    const content = new Set<object>();
    loaded.content.traverse((object) => content.add(object));
    loaded.scene.traverse((object) => {
        if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    loaded.scene.updateMatrixWorld();
    const before = await runOnce(device, manifest, loaded, false);
    const after = await runOnce(device, manifest, loaded, true);
    return {
        before,
        after,
        meanEAfterOverBefore: after.meanE / before.meanE,
        hitRayRadianceAfterOverBefore: after.meanHitRayRadiance / before.meanHitRayRadiance,
        skyDrawShare: after.skyDraws / Math.max(after.hitsWithLight, 1),
    };
}

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    // The platform's 8 storage buffers per stage, whatever the adapter allows (budget-device.ts).
    const { device } = await requestBudgetDevice(adapter);
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    const scenes: Record<string, unknown> = {};
    for (const id of SCENES) scenes[id] = await measureScene(device, id);
    await drainDevice(device);
    reportGpuTest({
        status: 'pass',
        data: { width: WIDTH, height: HEIGHT, frames: FRAMES, maxTextureSize: MAX_TEXTURE_SIZE ?? null, scenes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
