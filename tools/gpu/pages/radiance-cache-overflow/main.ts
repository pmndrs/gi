/**
 * RC3.2: the world cache's bucket overflow, counted (not timed), on `sponza`
 * (with curtains) and `bistro-exterior` at 1280 × 720 from each scene's first
 * bookmark, the camera still, for 300 frames of the radiance cache.
 *
 * Each frame the hash grid counts the inserts it refused because every slot
 * of their bucket held another tile (register D23), and the page copies that
 * count, the live tiles and the frame's hits (the visibility list) into a
 * buffer it reads once at the end. The page reports the totals; it fails only
 * on errors (a device error, a scene that doesn't load), as the numbers are
 * information for PROGRESS.
 *
 * Correctness and counting only, no timing (DECISIONS 2026-11-09). Scenes
 * load through the viewer's loader and the three adapter's scene sync (as a
 * loader only), as `radiance-cache-probes` loads Sponza; its `PageScreen`
 * draws the screen. Bistro's textures fit the 6 GB card beside the 1 GiB cache
 * with `VITE_DDGI_MAX_TEXTURE_SIZE=1024` (T2.17's cap; GI reads only their
 * averages). Through the lock, on the RTX:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc3.2 overflow" -- pnpm test:gpu --filter radiance-cache-overflow --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { RadianceCacheTechnique } from '../../../../packages/core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { SceneSync } from '../../../../packages/three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { loadManifest, readBuffer, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const FRAMES = 300;
const env = import.meta.env as Record<string, unknown>;
/** T2.17: scale model textures down to this side at load; undefined keeps them. */
const MAX_TEXTURE_SIZE = typeof env.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' ? Number(env.VITE_DDGI_MAX_TEXTURE_SIZE) : undefined;

/** Words copied per frame: refused inserts, live tiles, hits inserted. */
const WORDS_PER_FRAME = 3;

interface SceneCounts {
    readonly frames: number;
    readonly refusedInserts: number;
    readonly framesWithOverflow: number;
    readonly mostRefusedInAFrame: number;
    readonly firstFrameWithOverflow: number | null;
    readonly liveTilesAtEnd: number;
    readonly mostLiveTiles: number;
    readonly tileSlots: number;
    readonly meanHitsInserted: number;
    /**
     * The last frame's probe rays that hit a back face within 2 cm (RC3.5 review): the sign of a
     * ray whose origin, rebuilt from the screen's depth, sits under thin geometry at a grazing
     * view (register D17). Some can be real (a thin double-sided leaf); information only.
     */
    readonly probeRays: number;
    readonly backFaceHitsWithin2cm: number;
    readonly backFaceHitsWithin2cmShare: number;
}

/** The last frame's probe rays, and those that hit a back face within 2 cm (`WorldCacheProbeRay`: hitT word 6, front face word 9). */
async function countBackFaceHits(device: GPUDevice, technique: RadianceCacheTechnique): Promise<{ probeRays: number; backFaceHitsWithin2cm: number; backFaceHitsWithin2cmShare: number }> {
    const probes = technique.probes;
    if (!probes) return { probeRays: 0, backFaceHitsWithin2cm: 0, backFaceHitsWithin2cmShare: 0 };
    const count = new Uint32Array(await readBuffer(device, probes.probeRays, 4))[0];
    const words = new Uint32Array(await readBuffer(device, probes.probeRays, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + count * WORLD_CACHE_PROBE_RAY_BYTES));
    const floats = new Float32Array(words.buffer);
    let backFace = 0;
    for (let ray = 0; ray < count; ray++) {
        const base = (WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES) / 4;
        const hitT = floats[base + 6];
        if (hitT >= 0 && hitT < 0.02 && words[base + 9] === 0) backFace++;
    }
    return { probeRays: count, backFaceHitsWithin2cm: backFace, backFaceHitsWithin2cmShare: backFace / Math.max(count, 1) };
}

async function countScene(device: GPUDevice, manifest: SceneManifest): Promise<SceneCounts> {
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 2000 };
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-overflow ${manifest.id}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();

    // The model only is GI geometry; the sky dome is the viewer's background (GI gets the sky as its environment).
    const loaded = await loadScene(manifest, () => {}, MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE });
    const content = new Set<object>();
    loaded.content.traverse((object) => content.add(object));
    loaded.scene.traverse((object) => {
        if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    loaded.scene.updateMatrixWorld();
    const sync = new SceneSync(platform.scene, { device, warn: () => {} });
    sync.environment = platformSkyEnvironment(manifest);

    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frameInput: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    const counts = device.createBuffer({ label: 'radiance-cache-overflow counts', size: FRAMES * WORDS_PER_FRAME * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    try {
        const frame = (record: number | null): void => {
            sync.update(loaded.scene);
            const encoder = device.createCommandEncoder({ label: 'radiance-cache-overflow frame' });
            platform.encode(encoder, frameInput);
            screen.encode(encoder, camera);
            platform.encodeScreen(encoder, screen.screen);
            const cache = technique.worldCache;
            if (record !== null && cache) {
                const base = record * WORDS_PER_FRAME * 4;
                const tileWords = cache.buffers.tileWords;
                encoder.copyBufferToBuffer(tileWords, HASH_GRID_COUNTER.bucketOverflowCount * 4, counts, base, 4);
                encoder.copyBufferToBuffer(cache.packedTiles.count.buffer, cache.packedTiles.count.offset, counts, base + 4, 4);
                encoder.copyBufferToBuffer(tileWords, HASH_GRID_COUNTER.visibilityCount0 * 4, counts, base + 8, 4);
            }
            device.queue.submit([encoder.finish()]);
        };
        // A frame places the scene; then the mesh trees, and the counted frames with them, from an empty cache.
        frame(null);
        await platform.rays.whenBuilt();
        technique.resetHistory();
        for (let index = 0; index < FRAMES; index++) {
            frame(index);
            // Keep the queue short: a few frames in flight at most.
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();
        const words = new Uint32Array(await readBuffer(device, counts, FRAMES * WORDS_PER_FRAME * 4));
        const grazing = await countBackFaceHits(device, technique);
        let refusedInserts = 0;
        let framesWithOverflow = 0;
        let mostRefusedInAFrame = 0;
        let firstFrameWithOverflow: number | null = null;
        let mostLiveTiles = 0;
        let hits = 0;
        for (let index = 0; index < FRAMES; index++) {
            const refused = words[index * WORDS_PER_FRAME];
            refusedInserts += refused;
            if (refused > 0) {
                framesWithOverflow++;
                firstFrameWithOverflow ??= index;
            }
            mostRefusedInAFrame = Math.max(mostRefusedInAFrame, refused);
            mostLiveTiles = Math.max(mostLiveTiles, words[index * WORDS_PER_FRAME + 1]);
            hits += words[index * WORDS_PER_FRAME + 2];
        }
        return {
            frames: FRAMES,
            refusedInserts,
            framesWithOverflow,
            mostRefusedInAFrame,
            firstFrameWithOverflow,
            liveTilesAtEnd: words[(FRAMES - 1) * WORDS_PER_FRAME + 1],
            mostLiveTiles,
            tileSlots: technique.worldCache?.capacity.numTiles ?? 0,
            meanHitsInserted: hits / FRAMES,
            ...grazing,
        };
    } finally {
        counts.destroy();
        screen.dispose();
        platform.dispose();
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
    const scenes: Record<string, SceneCounts> = {};
    for (const id of ['sponza', 'bistro-exterior']) scenes[id] = await countScene(device, loadManifest(id));
    await drainDevice(device);
    reportGpuTest({
        status: 'pass',
        data: { width: WIDTH, height: HEIGHT, maxTextureSize: MAX_TEXTURE_SIZE ?? null, scenes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
