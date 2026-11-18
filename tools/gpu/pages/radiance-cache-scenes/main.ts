/**
 * RC3.2: no NaN, infinite or negative values in the radiance cache's output
 * on `cornell-box`, `two-room` and `sponza` (with curtains), each at
 * 1280 × 720 from its first bookmark, the camera still, for 60 frames, with
 * Capsaicin's default GI-1.2 multibounce on. RC3.3 adds the cells' indirect
 * values to the scan, and runs `cornell-box` and `two-room` again with
 * multibounce off: the mean probe-ray radiance after frame 60 must be higher
 * with it on in `two-room` (its closed room is lit by the second bounce; the
 * done-when's contact sheet needs the viewer, RC2.2), and is reported for
 * `cornell-box`.
 *
 * On this branch the technique's output is the probe atlas (the screen
 * cache's image passes, interpolation included, and the viewer's technique
 * entry are on RC2.2's branch), so the scan reads what the world cache
 * resolves and what the screen cache makes of it, after frames 1, 30 and 60:
 *
 * - every probe ray's radiance (the world cache's `ResolveCells`, and the
 *   bypassed rays' own samples, in `rays.radiance`);
 * - every texel of the probe atlas (`BlendScreenProbes`' output);
 * - every cell of every live tile of the world cache, all four levels
 *   (rgb × count and the count), direct and indirect values.
 *
 * Each must be finite and at least 0. Device errors fail the page too. The
 * mean probe radiance is reported, as information.
 *
 * Scenes load as `radiance-cache-probes` loads them (Cornell and two-room from
 * their manifests; Sponza through the viewer's loader and the three adapter's
 * scene sync, as a loader only); its `PageScreen` draws the screen.
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc3.2 scenes" -- pnpm test:gpu --filter radiance-cache-scenes --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
/** The frames after which the scan runs (counted from an empty cache). */
const SCAN_AFTER = [1, 30, 60] as const;

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const readback = device.createBuffer({ label: 'radiance-cache-scenes readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'radiance-cache-scenes readback' });
    encoder.copyBufferToBuffer(source, offset, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0, byteLength);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** An `rgba16float` texture's level 0 as half bits, rows packed. */
async function readHalfTexture(device: GPUDevice, texture: GPUTexture): Promise<Uint16Array> {
    const rowBytes = texture.width * 8;
    const paddedRow = Math.ceil(rowBytes / 256) * 256;
    const buffer = device.createBuffer({ label: 'radiance-cache-scenes atlas', size: paddedRow * texture.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: paddedRow }, [texture.width, texture.height]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(buffer.getMappedRange());
    const out = new Uint8Array(rowBytes * texture.height);
    for (let row = 0; row < texture.height; row++) out.set(mapped.subarray(row * paddedRow, row * paddedRow + rowBytes), row * rowBytes);
    buffer.unmap();
    buffer.destroy();
    return new Uint16Array(out.buffer);
}

/** Values that aren't finite and at least 0, among the half bits picked by `use` (by index). */
function badHalves(bits: Uint16Array, use: (index: number) => boolean): { checked: number; bad: number; first: string | null } {
    let checked = 0;
    let bad = 0;
    let first: string | null = null;
    for (let index = 0; index < bits.length; index++) {
        if (!use(index)) continue;
        checked++;
        const value = halfBitsToFloat(bits[index]);
        if (!(Number.isFinite(value) && value >= 0)) {
            bad++;
            first ??= `${value} at ${index}`;
        }
    }
    return { checked, bad, first };
}

interface Scan {
    readonly frame: number;
    readonly probeRays: number;
    readonly rayValuesBad: number;
    readonly atlasValuesBad: number;
    readonly liveTiles: number;
    readonly cellValuesBad: number;
    readonly meanRayRadiance: number;
    readonly firstBad: string | null;
}

async function scan(device: GPUDevice, technique: RadianceCacheTechnique, frame: number): Promise<Scan> {
    const probes = technique.probes;
    const cache = technique.worldCache;
    if (!probes || !cache) throw new Error('no probes or world cache after encodeScreen');

    // Each probe ray: rgb radiance and the hit distance, as halves (the distance isn't radiance).
    const rayCount = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
    const results = probes.rayScratch.rayResults;
    const rayBits = new Uint16Array(await readRange(device, results.buffer, results.offset ?? 0, rayCount * 8));
    const rays = badHalves(rayBits, (index) => index % 4 !== 3);
    let radianceSum = 0;
    for (let index = 0; index < rayBits.length; index++) if (index % 4 !== 3) radianceSum += halfBitsToFloat(rayBits[index]);

    // The atlas: radiance and hit distance per texel.
    const atlas = badHalves(await readHalfTexture(device, probes.probeAtlas), (index) => index % 4 !== 3);

    // Every live tile's cells, all levels: four halves per cell.
    const packed = cache.packedTiles;
    const liveTiles = new Uint32Array(await readRange(device, packed.count.buffer, packed.count.offset, 4))[0];
    const tiles = new Uint32Array(await readRange(device, cache.buffers.tileWords, packed.firstWord * 4, liveTiles * 4));
    const tileBytes = cache.capacity.numCellsPerTile * 8;
    let cellBad = 0;
    let firstCell: string | null = null;
    // The direct values, then the indirect ones (GI-1.2's second bounce, RC3.3).
    for (const valueBase of [cache.layout.cellValues.valueBuffer * 8, cache.layout.cellValues.valueIndirectBuffer * 8]) {
        // In batches, so the staging buffer stays small on a scene with many tiles.
        const batch = 4096;
        for (let start = 0; start < tiles.length; start += batch) {
            const slice = tiles.subarray(start, Math.min(tiles.length, start + batch));
            const staging = device.createBuffer({ label: 'radiance-cache-scenes tiles', size: Math.max(4, slice.length * tileBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
            const encoder = device.createCommandEncoder();
            slice.forEach((tile, index) => encoder.copyBufferToBuffer(cache.buffers.cellValues, valueBase + tile * tileBytes, staging, index * tileBytes, tileBytes));
            device.queue.submit([encoder.finish()]);
            const cells = badHalves(new Uint16Array(await readRange(device, staging, 0, slice.length * tileBytes)), () => true);
            staging.destroy();
            cellBad += cells.bad;
            firstCell ??= cells.first;
        }
    }

    return {
        frame,
        probeRays: rayCount,
        rayValuesBad: rays.bad,
        atlasValuesBad: atlas.bad,
        liveTiles,
        cellValuesBad: cellBad,
        meanRayRadiance: rays.checked > 0 ? radianceSum / rays.checked : 0,
        firstBad: rays.first ?? atlas.first ?? firstCell,
    };
}

/** Sponza through the viewer's loader, copied into the platform by the three adapter's scene sync (a loader here). */
async function describeLoaded(platform: GIPlatformImpl, device: GPUDevice, manifest: SceneManifest): Promise<() => void> {
    const loaded = await loadScene(manifest, () => {});
    // Only the model is GI geometry; the sky dome is the viewer's background (GI gets the sky as its environment).
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

async function scanScene(device: GPUDevice, id: string, loader: 'manifest' | 'viewer', settings: Partial<RadianceCacheSettings> = {}): Promise<Scan[]> {
    const manifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-scenes ${id}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    let beforeFrame = (): void => {};
    if (loader === 'viewer') beforeFrame = await describeLoaded(platform, device, manifest);
    else describeScene(platform, manifest);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frameInput: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    const frame = (): void => {
        beforeFrame();
        const encoder = device.createCommandEncoder({ label: `radiance-cache-scenes ${id}` });
        platform.encode(encoder, frameInput);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    try {
        // A frame places the scene; then the mesh trees, and the scanned frames with them, from an empty cache.
        frame();
        await platform.rays.whenBuilt();
        technique.resetHistory();
        const scans: Scan[] = [];
        for (let index = 1; index <= SCAN_AFTER[SCAN_AFTER.length - 1]; index++) {
            frame();
            if (index % 8 === 0) await device.queue.onSubmittedWorkDone();
            if ((SCAN_AFTER as readonly number[]).includes(index)) {
                await device.queue.onSubmittedWorkDone();
                scans.push(await scan(device, technique, index));
            }
        }
        return scans;
    } finally {
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
    const scenes: Record<string, Scan[]> = {};
    const failures: string[] = [];
    // Capsaicin's defaults (multibounce on), then cornell-box and two-room with multibounce off.
    const runs = [
        ['cornell-box', 'manifest', true],
        ['two-room', 'manifest', true],
        ['sponza', 'viewer', true],
        ['cornell-box', 'manifest', false],
        ['two-room', 'manifest', false],
    ] as const;
    for (const [scene, loader, useMultibounce] of runs) {
        const id = useMultibounce ? scene : `${scene} (multibounce off)`;
        const scans = await scanScene(device, scene, loader, { useMultibounce });
        scenes[id] = scans;
        for (const result of scans) {
            const bad = result.rayValuesBad + result.atlasValuesBad + result.cellValuesBad;
            if (bad > 0) failures.push(`${id} after frame ${result.frame}: ${result.rayValuesBad} ray, ${result.atlasValuesBad} atlas and ${result.cellValuesBad} cell values are NaN, infinite or negative (first: ${result.firstBad})`);
            if (result.probeRays === 0 || result.liveTiles === 0) failures.push(`${id} after frame ${result.frame}: ${result.probeRays} probe rays and ${result.liveTiles} live tiles; nothing to scan`);
        }
    }
    // The second bounce lights two-room's closed room: the mean probe radiance after frame 60 rises with it.
    const lastMean = (id: string): number => scenes[id]?.[scenes[id].length - 1]?.meanRayRadiance ?? 0;
    const multibounceGain = {
        'two-room': lastMean('two-room') / Math.max(lastMean('two-room (multibounce off)'), 1e-9),
        'cornell-box': lastMean('cornell-box') / Math.max(lastMean('cornell-box (multibounce off)'), 1e-9),
    };
    if (!(lastMean('two-room') > lastMean('two-room (multibounce off)'))) {
        failures.push(`two-room's mean probe radiance after frame 60 is ${lastMean('two-room')} with multibounce and ${lastMean('two-room (multibounce off)')} without: the second bounce should raise it`);
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, multibounceGain, scenes },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
