/**
 * RC2.1: the radiance cache's screen probes on the GPU, checked against the
 * CPU ports of Capsaicin's rules (packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts).
 *
 * Three scenes, each at 1280 × 720 with one frame of the technique, its
 * `FrameScreen` drawn engine-free by primary rays through the platform's
 * tracer (screen.ts; the three adapter's screen inputs come with RC2.2):
 *
 * - `sky-plane`: a 400 m floor under a constant white sky (radiance 1), no
 *   lights, the camera 3 m up looking a little down, so the top of the image
 *   is sky;
 * - `cornell-box` from its manifest and first bookmark;
 * - `sponza` (with curtains) through the viewer's loader and the three
 *   adapter's scene sync (as a scene loader only), its first bookmark.
 *
 * Checks on every scene:
 * 1. **Spawn:** each spawn tile's flag and seed equal the CPU's spawn rule
 *    (`SpawnScreenProbes`: the Halton jitter of the frame the technique ran,
 *    no probe on a sky pixel, sky read from the page's depth).
 * 2. **Compaction:** the probe list holds exactly the non-sky tiles' seeds,
 *    in tile order, and its count is theirs; the probe mask holds each
 *    probe's seed in its 8 × 8 tile and "no probe" everywhere else.
 * 3. **Ray directions:** the ray count on the GPU is 64 × the probes; every
 *    ray of every probe points to the normal's side and lies in its own
 *    hemi-octahedral cell (a ray that half precision moved across a cell
 *    edge, within 0.02 of a cell, is counted and reported, as Capsaicin's
 *    `BlendScreenProbes` would put it in the neighbour too); every ray carries
 *    the bypass flag (phase 2 lights every ray directly).
 *
 * And on `sky-plane`:
 * 4. **Sky:** every ray of every probe misses, and every cell of every probe
 *    in the atlas reads the sky's radiance (1) within 1%.
 *
 * The page draws the probe atlas and the probe mask of each scene (the
 * technique's engine-free debug views) for the runner's screenshot.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Run through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc2.1 probes" -- pnpm test:gpu --filter radiance-cache-probes --gpu nvidia` (or intel).
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import {
    halfBitsToFloat,
    INVALID_ID,
    mapToHemiOctahedronInverse,
    packSeed,
    probeCellOf,
    spawnScreenProbes,
    toProbeFrame,
    unpackHalf3,
    type Direction,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { PROBE_CELLS, PROBE_SIZE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { RADIANCE_CACHE_DEBUG_VIEW, type RadianceCacheDebugViewId } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/debug.ts';
import { PROBE_RAY_BYPASS, PROBE_RAY_BYTES, PROBE_RAYS_HEADER_BYTES, PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-probes.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, readBuffer } from '../ddgi-shared/harness.ts';
import { PageScreen, frameCamera, type PageCamera } from './screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
/** How far a ray may sit past its cell's edge, in cells, and still count as moved by half precision. */
const HALF_PRECISION_EDGE = 0.02;
/** The sky check's tolerance (RC2.1 done-when). */
const SKY_TOLERANCE = 0.01;
/** The sky-plane scene's sky radiance. */
const SKY_RADIANCE = 1;

interface SceneRun {
    readonly name: string;
    readonly camera: PageCamera;
    readonly technique: RadianceCacheTechnique;
    readonly platform: GIPlatformImpl;
    readonly screen: PageScreen;
    /** Runs one frame: the host's encode, the page's screen, the technique's screen stage. */
    frame(): void;
    dispose(): void;
}

function frameInput(camera: PageCamera): FrameInput {
    return { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
}

async function startScene(device: GPUDevice, name: string, camera: PageCamera, describe: (platform: GIPlatformImpl) => Promise<(() => void) | void>): Promise<SceneRun> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-probes ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    const beforeFrame = (await describe(platform)) ?? (() => {});
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        beforeFrame();
        const encoder = device.createCommandEncoder({ label: `radiance-cache-probes ${name}` });
        platform.encode(encoder, frameInput(camera));
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    // A frame places the scene; then the mesh trees, and a frame with them.
    frame();
    await platform.rays.whenBuilt();
    frame();
    await device.queue.onSubmittedWorkDone();
    if (platform.rays.building) throw new Error(`${name}: the platform still reports trees building`);
    return {
        name,
        camera,
        technique,
        platform,
        screen,
        frame,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

/** A 400 m floor at y = 0 under a constant white sky, no lights. */
async function describeSkyPlane(platform: GIPlatformImpl): Promise<void> {
    const half = 200;
    const scene = platform.scene;
    const mesh = scene.createMesh({
        positions: new Float32Array([-half, 0, -half, half, 0, -half, half, 0, half, -half, 0, half]),
        normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
        indices: new Uint32Array([0, 2, 1, 0, 3, 2]),
        label: 'floor',
    });
    const material = scene.createMaterial({ albedo: [0.5, 0.5, 0.5], label: 'floor' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'floor' });
    scene.setEnvironment({ type: 'color', color: [SKY_RADIANCE, SKY_RADIANCE, SKY_RADIANCE], intensity: 1 });
}

function bookmarkCamera(manifest: SceneManifest, far: number): PageCamera {
    const bookmark = manifest.cameras[0];
    return { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far };
}

/** Sponza through the viewer's loader, copied into the platform by the three adapter's scene sync (a loader here, nothing more). */
async function describeSponza(platform: GIPlatformImpl, device: GPUDevice, manifest: SceneManifest): Promise<() => void> {
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
    return () => {
        sync.update(loaded.scene);
    };
}

/** A texture's level as bytes, rows packed (the copy pads them to 256 bytes). */
async function readTexture(device: GPUDevice, texture: GPUTexture, bytesPerTexel: number, width: number, height: number): Promise<Uint8Array> {
    const rowBytes = width * bytesPerTexel;
    const paddedRow = Math.ceil(rowBytes / 256) * 256;
    const buffer = device.createBuffer({ size: paddedRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture, mipLevel: 0 }, { buffer, bytesPerRow: paddedRow }, [width, height]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(buffer.getMappedRange());
    const out = new Uint8Array(rowBytes * height);
    for (let row = 0; row < height; row++) out.set(mapped.subarray(row * paddedRow, row * paddedRow + rowBytes), row * rowBytes);
    buffer.unmap();
    buffer.destroy();
    return out;
}

function halves(bytes: Uint8Array): Float32Array {
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    return Float32Array.from(words, (bits) => halfBitsToFloat(bits));
}

interface SceneChecks {
    readonly failures: string[];
    readonly summary: Record<string, unknown>;
}

async function checkScene(device: GPUDevice, run: SceneRun, skyCheck: boolean): Promise<SceneChecks> {
    const failures: string[] = [];
    const probes = run.technique.probes;
    if (!probes) return { failures: [`${run.name}: no probes after encodeScreen`], summary: {} };
    const { layout, frameIndex } = probes;
    const fail = (message: string): void => {
        if (failures.length < 20) failures.push(`${run.name}: ${message}`);
    };

    // The page's screen, as the technique read it.
    const depth = new Float32Array((await readTexture(device, run.screen.screen.depth, 4, WIDTH, HEIGHT)).buffer);
    const normals = halves(await readTexture(device, run.screen.screen.normal, 8, WIDTH, HEIGHT));
    const isSky = (x: number, y: number): boolean => depth[y * WIDTH + x] === 1;
    const normalAt = (x: number, y: number): Direction => {
        const base = (y * WIDTH + x) * 4;
        const length = Math.hypot(normals[base], normals[base + 1], normals[base + 2]);
        return [normals[base] / length, normals[base + 1] / length, normals[base + 2] / length];
    };

    // 1. Spawn.
    const expected = spawnScreenProbes(WIDTH, HEIGHT, frameIndex, isSky);
    const tileCount = layout.maxProbeSpawnCount;
    if (expected.length !== tileCount) fail(`the CPU makes ${expected.length} spawn tiles, the layout ${tileCount}`);
    const flags = new Uint32Array(await readBuffer(device, probes.spawnFlags, tileCount * 4));
    const seeds = new Uint32Array(await readBuffer(device, probes.spawnSeeds, tileCount * 4));
    let spawnMismatches = 0;
    expected.forEach((tile, index) => {
        const flagWrong = flags[index] !== (tile.spawned ? 1 : 0);
        const seedWrong = tile.spawned && seeds[index] !== packSeed(tile.seed[0], tile.seed[1]);
        if (flagWrong || seedWrong) {
            spawnMismatches++;
            fail(`spawn tile ${index}: GPU flag ${flags[index]} seed ${seeds[index]}, CPU ${tile.spawned ? 1 : 0} at (${tile.seed.join(', ')})`);
        }
    });
    const spawned = expected.filter((tile) => tile.spawned);

    // 2. Compaction and the mask.
    const probeSpawn = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + tileCount * 4));
    const probeCount = probeSpawn[0];
    if (probeCount !== spawned.length) fail(`the probe list counts ${probeCount}, the non-sky tiles are ${spawned.length}`);
    let listMismatches = 0;
    spawned.forEach((tile, index) => {
        if (probeSpawn[1 + index] !== packSeed(tile.seed[0], tile.seed[1])) {
            listMismatches++;
            fail(`probe ${index}: GPU seed ${probeSpawn[1 + index]}, CPU (${tile.seed.join(', ')})`);
        }
    });
    const [maskWidth, maskHeight] = layout.probeCount;
    const mask = new Uint32Array((await readTexture(device, probes.probeMask, 4, maskWidth, maskHeight)).buffer);
    const expectedMask = new Uint32Array(maskWidth * maskHeight).fill(INVALID_ID);
    for (const tile of spawned) expectedMask[Math.floor(tile.seed[1] / PROBE_SIZE) * maskWidth + Math.floor(tile.seed[0] / PROBE_SIZE)] = packSeed(tile.seed[0], tile.seed[1]);
    let maskMismatches = 0;
    for (let texel = 0; texel < mask.length; texel++) {
        if (mask[texel] !== expectedMask[texel]) {
            maskMismatches++;
            fail(`mask tile ${texel % maskWidth}, ${Math.floor(texel / maskWidth)}: GPU ${mask[texel]}, CPU ${expectedMask[texel]}`);
        }
    }

    // 3. Ray directions.
    const rayBytes = PROBE_RAYS_HEADER_BYTES + spawned.length * PROBE_CELLS * PROBE_RAY_BYTES;
    const rayWords = new Uint32Array(await readBuffer(device, probes.probeRays, Math.max(rayBytes, 16)));
    if (rayWords[0] !== probeCount * PROBE_CELLS) fail(`the GPU's ray count is ${rayWords[0]}, 64 x the probes is ${probeCount * PROBE_CELLS}`);
    const rayFloats = new Float32Array(rayWords.buffer);
    let wrongSide = 0;
    let wrongCell = 0;
    let movedByHalfPrecision = 0;
    let withoutBypass = 0;
    let hits = 0;
    let minNormalDot = Infinity;
    spawned.forEach((tile, probe) => {
        const normal = normalAt(tile.seed[0], tile.seed[1]);
        for (let cell = 0; cell < PROBE_CELLS; cell++) {
            const base = (PROBE_RAYS_HEADER_BYTES + (probe * PROBE_CELLS + cell) * PROBE_RAY_BYTES) / 4;
            const direction = unpackHalf3([rayWords[base + 4], rayWords[base + 5]]);
            if ((rayWords[base + 3] & PROBE_RAY_BYPASS) === 0) withoutBypass++;
            if (rayFloats[base + 6] >= 0) hits++;
            const facing = direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2];
            minNormalDot = Math.min(minNormalDot, facing);
            if (!(facing > 0)) {
                wrongSide++;
                fail(`probe ${probe} cell ${cell}: direction (${direction.map((value) => value.toFixed(4)).join(', ')}) is not on the normal's side`);
                continue;
            }
            if (probeCellOf(direction, normal) === cell) continue;
            // How far past the cell's edges the ray lies, in cells.
            const [u, v] = mapToHemiOctahedronInverse(toProbeFrame(direction, normal)).map((value) => value * PROBE_SIZE);
            const cellX = cell % PROBE_SIZE;
            const cellY = Math.floor(cell / PROBE_SIZE);
            const outside = Math.max(cellX - u, u - (cellX + 1), cellY - v, v - (cellY + 1), 0);
            if (outside <= HALF_PRECISION_EDGE) {
                movedByHalfPrecision++;
            } else {
                wrongCell++;
                fail(`probe ${probe} cell ${cell}: the ray lies in cell ${probeCellOf(direction, normal)}, ${outside.toFixed(3)} cells outside its own`);
            }
        }
    });
    if (withoutBypass > 0) fail(`${withoutBypass} rays lack the bypass flag (phase 2 lights every ray directly)`);

    // 4. The sky.
    const sky: Record<string, number> = {};
    if (skyCheck) {
        if (hits > 0) fail(`${hits} rays hit something; nothing is above the floor`);
        const [atlasWidth, atlasHeight] = layout.atlasSize;
        const atlas = halves(await readTexture(device, probes.probeAtlas, 8, atlasWidth, atlasHeight));
        let cells = 0;
        let worst = 0;
        let minimum = Infinity;
        let maximum = -Infinity;
        for (const tile of spawned) {
            const tileX = Math.floor(tile.seed[0] / PROBE_SIZE) * PROBE_SIZE;
            const tileY = Math.floor(tile.seed[1] / PROBE_SIZE) * PROBE_SIZE;
            for (let cell = 0; cell < PROBE_CELLS; cell++) {
                const texel = ((tileY + Math.floor(cell / PROBE_SIZE)) * atlasWidth + tileX + (cell % PROBE_SIZE)) * 4;
                for (let channel = 0; channel < 3; channel++) {
                    const value = atlas[texel + channel];
                    minimum = Math.min(minimum, value);
                    maximum = Math.max(maximum, value);
                    worst = Math.max(worst, Math.abs(value - SKY_RADIANCE) / SKY_RADIANCE);
                }
                cells++;
            }
        }
        if (!(worst <= SKY_TOLERANCE)) fail(`a probe cell reads the sky ${worst * 100}% off (min ${minimum}, max ${maximum}, sky ${SKY_RADIANCE})`);
        Object.assign(sky, { cells, worstRelativeError: worst, minimum, maximum });
    }

    return {
        failures,
        summary: {
            frameIndex,
            spawnTiles: tileCount,
            skyTiles: tileCount - spawned.length,
            probes: probeCount,
            spawnMismatches,
            listMismatches,
            maskMismatches,
            rays: rayWords[0],
            rayHits: hits,
            wrongSide,
            wrongCell,
            movedByHalfPrecision,
            minNormalDot,
            ...(skyCheck ? { sky } : {}),
            memoryMB: Number((run.technique.memoryBytes / 1e6).toFixed(2)),
            counters: run.technique.counters,
        },
    };
}

/** Draws a debug view into a canvas on the page, with a caption, for the runner's screenshot. */
async function showDebugView(device: GPUDevice, run: SceneRun, view: RadianceCacheDebugViewId, caption: string, column: HTMLElement): Promise<void> {
    const target = device.createTexture({ size: [WIDTH, HEIGHT], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const encoder = device.createCommandEncoder();
    run.technique.encodeDebugView(encoder, view, target);
    device.queue.submit([encoder.finish()]);
    const pixels = await readTexture(device, target, 4, WIDTH, HEIGHT);
    target.destroy();
    const canvas = document.createElement('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = '400px';
    canvas.style.height = '225px';
    canvas.style.display = 'block';
    canvas.getContext('2d')?.putImageData(new ImageData(Uint8ClampedArray.from(pixels), WIDTH, HEIGHT), 0, 0);
    const label = document.createElement('div');
    label.textContent = caption;
    label.style.font = '12px sans-serif';
    column.append(label, canvas);
}

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    // As a host would ask (requestGIDevice): the adapter's storage limits, and subgroups and timestamps when it has them.
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    window.addEventListener('pagehide', () => device.destroy(), { once: true });
    deviceErrors = watchDevice(device);
    const gallery = document.createElement('div');
    gallery.style.display = 'flex';
    gallery.style.gap = '8px';
    document.body.append(gallery);

    const cornell = loadManifest('cornell-box');
    const sponza = loadManifest('sponza');
    const scenes: { name: string; camera: PageCamera; sky: boolean; describe: (platform: GIPlatformImpl) => Promise<(() => void) | void> }[] = [
        { name: 'sky-plane', camera: { position: [0, 3, 0], target: [0, 1, -10], fovYDeg: 60, near: 0.1, far: 1000 }, sky: true, describe: describeSkyPlane },
        {
            name: 'cornell-box',
            camera: bookmarkCamera(cornell, 100),
            sky: false,
            describe: async (platform) => {
                describeScene(platform, cornell);
            },
        },
        { name: 'sponza', camera: bookmarkCamera(sponza, 1000), sky: false, describe: (platform) => describeSponza(platform, device, sponza) },
    ];

    const results: Record<string, unknown> = {};
    const failures: string[] = [];
    for (const scene of scenes) {
        const run = await startScene(device, scene.name, scene.camera, scene.describe);
        try {
            const checks = await checkScene(device, run, scene.sky);
            results[scene.name] = checks.summary;
            failures.push(...checks.failures);
            const column = document.createElement('div');
            gallery.append(column);
            await showDebugView(device, run, RADIANCE_CACHE_DEBUG_VIEW.probeAtlas, `${scene.name}: probe atlas`, column);
            await showDebugView(device, run, RADIANCE_CACHE_DEBUG_VIEW.probeMask, `${scene.name}: probe mask`, column);
        } finally {
            run.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }
    await drainDevice(device);
    const summary = summarizeAdapterInfo(device.adapterInfo, device.features);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, scenes: results },
        adapter: summary,
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
