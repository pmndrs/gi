/**
 * P6.3, experiment E10: the world cache's second bounce read from a DDGI
 * volume stacked under the radiance cache (`secondBounce: 'ddgi-volume'`,
 * `gi.setFallback(radianceCache, ddgi)`), against GI-1.2's second-bounce rays
 * (`'gi1-rays'`, the default). Correctness checks with expected values
 * written before the run, and the lag after a light change as a measurement.
 *
 * **The furnace** (radiance-cache-world-cache's sections 8-13): a floor
 * (y = 0, facing up) and a ceiling (y = 3, facing down), 400 m wide, albedo
 * ρ = 0.5, a black sky, a sun straight down (E = 4) that casts no shadow, so it
 * lights the floor and never the ceiling's underside. The camera at (0, 2, 0)
 * looks down at (0, 0, -2): every probe is on the floor, and its rays that hit,
 * hit the ceiling. A ceiling cell has no direct light, so a ceiling probe ray
 * reads its cell's indirect value (or, shorter than its cell, its own sample:
 * register D38). The rays judged are those whose hit is within 8 m of the
 * camera (tiles under 2 m there, so no tile spans floor and ceiling). DDGI's
 * volume: 24 × 2 × 24 m at 1 m spacing around them (probes at y = 0.5, 1.5,
 * 2.5), every probe updated every frame.
 *
 * Expected, before the run (L_f1 = ρ E / π = 0.6366, the floor's direct light):
 *
 * 1. **gi1-rays:** the second bounce ρ L_f1 = ρ² E / π = 0.3183 (RC3.5).
 * 2. **ddgi-volume, DDGI with no bounce of its own** (`bounceEnergy` 0): DDGI's
 *    field is then the floor's direct light only, E = π L_f1 under the ceiling,
 *    so albedo / π × E is the same second bounce, 0.3183: the lookup itself is
 *    right (position, normal side, albedo / π).
 * 3. **ddgi-volume, DDGI as it ships** (`bounceEnergy` 1): DDGI's field holds
 *    every bounce, so the floor's light is L_f = L_f1 / (1 − ρ²) = 0.8488 and
 *    a ceiling cell's indirect value is every later bounce, ρ L_f = 0.4244 (the
 *    analytic every-bounce value). GI-1.2's estimate stops at the second.
 *
 * Sections 1-3 run DDGI at a fixed hysteresis of 0.97 for 600 frames (DDGI's
 * own convergence is its business; its furnace page shows the adaptive rule
 * needs 1,800), and require the near ceiling rays' mean within 2% (1) or 3%
 * (2, 3) of the expected value.
 *
 * 4-6. **Strips** (the furnace's floor as strips 5 cm wide every 10 cm, so
 *    half the rays from the ceiling miss into the black below): gi1-rays and
 *    ddgi-volume without DDGI's bounce read the second bounce ρ L_f1 × 1/2 =
 *    0.1592; with DDGI's bounce, every bounce: the strips' light is
 *    L_f1 / (1 − ρ² / 2) = 0.7276 and the ceiling's ρ / 2 × that = 0.1819.
 *    Same tolerances.
 *
 * 7. **What runs:** with 'ddgi-volume' the frame has no second-bounce entry
 *    and no second-bounce shadow ray, and the cache reports 'ddgi-volume' in
 *    use; asked for 'ddgi-volume' with nothing stacked under it, the cache
 *    warns and traces GI-1.2's rays.
 *
 * 8. **Lag after a light change** (a measurement; DDGI at its defaults: the
 *    adaptive hysteresis and its change response, warm-up on): after 300
 *    frames, the light is doubled, and each frame after it the page reads the
 *    near-ceiling rays' mean (the furnace, sun doubled) or the mean of every
 *    probe ray that hits (two-room, its lamp doubled), and the mean indirect
 *    value of the live finest cells. Everything is linear in the one light, so
 *    the settled value after the change is twice the one before (the mean of
 *    the last 10 frames before it); the count is the first frame within 5% of
 *    that. Each scene with gi1-rays and with ddgi-volume, and ddgi-volume with
 *    the cells' indirect cap at 4 samples instead of 16
 *    (`hashGridCacheMaxMultibounceSampleCount`): DDGI's value carries no ray
 *    noise, so a shorter running mean may do.
 *
 * `VITE_E10_SECTIONS=accuracy` runs sections 1-7 only and `lag` section 8
 * only (for mutation runs); unset, all of them.
 *
 * Correctness only, no timing (radiance-cache-ddgi-bounce-cost times it).
 * Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-ddgi-bounce --gpu nvidia`.
 */

import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import type { DDGISettings } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { halfBitsToFloat, unpackHalf3 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import type { HashGridCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cache.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;

const FURNACE = {
    camera: { position: [0, 2, 0], target: [0, 0, -2], fovYDeg: 60, near: 0.05, far: 1000 } as PageCamera,
    halfWidth: 200,
    ceilingY: 3,
    albedo: 0.5,
    sunIntensity: 4,
} as const;
/** The furnace's DDGI volume: probes at y = 0.5, 1.5 and 2.5, 1 m apart, around the near rays' hits. */
const FURNACE_VOLUME = { center: [0, 1.5, -4] as Vec3, size: [24, 2, 24] as Vec3, spacing: 1 };
/** Rays judged: their hit within this distance of the camera (tiles under 2 m). */
const NEAR_HIT_METRES = 8;
const STRIPS = { width: 0.05, period: 0.1, halfLength: 100, fill: 0.5 } as const;

/** The floor's direct light, ρ E / π. */
const FLOOR_DIRECT = (FURNACE.albedo * FURNACE.sunIntensity) / Math.PI;
const EXPECTED = {
    /** ρ L_f1 = ρ² E / π. */
    secondBounce: FURNACE.albedo * FLOOR_DIRECT,
    /** ρ L_f1 / (1 − ρ²): every bounce after the first. */
    everyBounce: (FURNACE.albedo * FLOOR_DIRECT) / (1 - FURNACE.albedo ** 2),
    /** ρ f L_f1. */
    stripsSecondBounce: FURNACE.albedo * STRIPS.fill * FLOOR_DIRECT,
    /** ρ f L_f1 / (1 − ρ² f). */
    stripsEveryBounce: (FURNACE.albedo * STRIPS.fill * FLOOR_DIRECT) / (1 - FURNACE.albedo ** 2 * STRIPS.fill),
} as const;

/** DDGI for the accuracy sections: a fixed hysteresis so the field settles in 600 frames, every probe each frame. */
const ACCURACY_FRAMES = 600;
const DDGI_ACCURACY: Partial<Omit<DDGISettings, 'volume' | 'volumes'>> = { adaptiveHysteresis: false, hysteresis: 0.97, warmUpPasses: 0 };
/** Probes updated a frame: every probe of the furnace's volume (1,875) and two-room's (1,134). */
const DDGI_UNITS = 2048;

const LAG_SETTLE_FRAMES = 300;
const LAG_MAX_FRAMES = 240;
const LAG_TOLERANCE = 0.05;
/** Which sections run: `accuracy` (1-7), `lag` (8) or all. */
const SECTIONS = String((import.meta.env as Record<string, unknown>).VITE_E10_SECTIONS ?? 'all');

const FURNACE_SUN: LightDescriptor = { type: 'directional', direction: [0, -1, 0], color: [1, 1, 1], intensity: FURNACE.sunIntensity, castsShadows: false, label: 'furnace sun' };

type Mode = 'gi1-rays' | 'ddgi-volume';

interface Scenario {
    readonly platform: GIPlatformImpl;
    readonly technique: RadianceCacheTechnique;
    readonly ddgi: DDGITechnique | null;
    readonly cache: HashGridCache;
    readonly lights: readonly LightHandle[];
    readonly camera: PageCamera;
    frame(): void;
    dispose(): void;
}

interface ScenarioOptions {
    readonly scene: { readonly kind: 'furnace'; readonly strips: boolean } | { readonly kind: 'manifest'; readonly id: string };
    readonly settings: Partial<RadianceCacheSettings>;
    /** DDGI stacked under the radiance cache, with these settings; null for none. */
    readonly ddgi: Partial<Omit<DDGISettings, 'volume' | 'volumes'>> | null;
}

// ---- The scenes ---------------------------------------------------------------------------------

function quad(platform: GIPlatformImpl, label: string, corners: readonly Vec3[], normal: Vec3, albedo: number): void {
    const scene = platform.scene;
    const mesh = scene.createMesh({
        positions: new Float32Array(corners.flat()),
        normals: new Float32Array([...normal, ...normal, ...normal, ...normal]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        label,
    });
    const material = scene.createMaterial({ albedo: [albedo, albedo, albedo], label });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label });
}

function stripFloor(platform: GIPlatformImpl): void {
    const { width, period, halfLength } = STRIPS;
    const count = Math.round((2 * halfLength) / period);
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    for (let strip = 0; strip < count; strip++) {
        const z0 = -halfLength + strip * period;
        const z1 = z0 + width;
        const base = strip * 4;
        positions.push(-halfLength, 0, z1, halfLength, 0, z1, halfLength, 0, z0, -halfLength, 0, z0);
        normals.push(0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0);
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    const scene = platform.scene;
    const mesh = scene.createMesh({ positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices), label: 'strip floor' });
    const material = scene.createMaterial({ albedo: [FURNACE.albedo, FURNACE.albedo, FURNACE.albedo], label: 'strip floor' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'strip floor' });
}

function describeFurnace(platform: GIPlatformImpl, strips: boolean): LightHandle[] {
    const half = FURNACE.halfWidth;
    const y = FURNACE.ceilingY;
    if (strips) stripFloor(platform);
    else quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], FURNACE.albedo);
    quad(platform, 'ceiling', [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]], [0, -1, 0], FURNACE.albedo);
    const handles = [platform.scene.createLight(FURNACE_SUN)];
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    return handles;
}

async function startScenario(device: GPUDevice, name: string, options: ScenarioOptions): Promise<Scenario> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-ddgi-bounce ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(options.settings));
    platform.setOutput(technique);
    let camera: PageCamera = FURNACE.camera;
    let volume = { center: FURNACE_VOLUME.center, size: FURNACE_VOLUME.size, spacing: FURNACE_VOLUME.spacing };
    const manifest = options.scene.kind === 'manifest' ? loadManifest(options.scene.id) : null;
    if (manifest) {
        const bookmark = manifest.cameras[0];
        camera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
        volume = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
    }
    // The host-level arrangement E10 needs: DDGI attached beside the radiance cache and stacked
    // under it, so the platform encodes it first and hands its shading module to the cache.
    const ddgi = options.ddgi ? platform.addTechnique(new DDGITechnique({ ...options.ddgi, volume })) : null;
    if (ddgi) platform.setFallback(technique, ddgi);
    await platform.prepare();
    const lights: LightHandle[] = [];
    if (manifest) describeScene(platform, manifest, {}, lights);
    else lights.push(...describeFurnace(platform, options.scene.kind === 'furnace' && options.scene.strips));
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `radiance-cache-ddgi-bounce ${name}` });
        platform.encode(encoder, { deltaTime: 1 / 60, budget: { units: DDGI_UNITS }, camera: frameCamera(camera, WIDTH, HEIGHT) });
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    frame();
    await platform.rays.whenBuilt();
    frame();
    await device.queue.onSubmittedWorkDone();
    technique.resetHistory();
    const cache = technique.worldCache;
    if (!cache) throw new Error(`${name}: no world cache after prepare`);
    return {
        platform,
        technique,
        ddgi,
        cache,
        lights,
        camera,
        frame,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

async function runFrames(device: GPUDevice, scenario: Scenario, frames: number): Promise<void> {
    for (let index = 0; index < frames; index++) {
        scenario.frame();
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();
}

// ---- Reading back -------------------------------------------------------------------------------

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const readback = device.createBuffer({ label: 'ddgi-bounce readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'ddgi-bounce readback' });
    encoder.copyBufferToBuffer(source, offset, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0, byteLength);
    readback.unmap();
    readback.destroy();
    return copy;
}

const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));

interface RayReading {
    /** Mean radiance (channel mean) of the rays that hit within NEAR_HIT_METRES of the camera, on the ceiling (furnace) or anywhere (manifest scenes). */
    readonly nearMean: number;
    readonly nearRays: number;
    /** Mean over every ray that hits. */
    readonly hitMean: number;
    readonly hitRays: number;
    /** Rays whose radiance is NaN, infinite or negative. */
    readonly bad: number;
}

async function readRays(device: GPUDevice, scenario: Scenario, furnace: boolean): Promise<RayReading> {
    const probes = scenario.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const rayCount = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
    const rayWords = new Uint32Array(await readRange(device, probes.probeRays, 0, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + rayCount * WORLD_CACHE_PROBE_RAY_BYTES));
    const rayFloats = new Float32Array(rayWords.buffer);
    const results = probes.rayScratch.rayResults;
    const radianceWords = new Uint32Array(await readRange(device, results.buffer, results.offset ?? 0, rayCount * 8));
    const eye = scenario.camera.position;
    let nearSum = 0;
    let nearRays = 0;
    let hitSum = 0;
    let hitRays = 0;
    let bad = 0;
    for (let ray = 0; ray < rayCount; ray++) {
        const base = (WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES) / 4;
        const hitT = rayFloats[base + 6];
        if (!(hitT >= 0)) continue;
        const direction = unpackHalf3([rayWords[base + 4], rayWords[base + 5]]);
        const hit: Vec3 = [rayFloats[base] + hitT * direction[0], rayFloats[base + 1] + hitT * direction[1], rayFloats[base + 2] + hitT * direction[2]];
        const r = HALF_TABLE[radianceWords[ray * 2] & 0xffff];
        const g = HALF_TABLE[radianceWords[ray * 2] >>> 16];
        const b = HALF_TABLE[radianceWords[ray * 2 + 1] & 0xffff];
        const value = (r + g + b) / 3;
        if (!(Number.isFinite(value) && value >= 0)) {
            bad++;
            continue;
        }
        hitSum += value;
        hitRays++;
        const onCeiling = Math.abs(hit[1] - FURNACE.ceilingY) < 1e-3;
        if (furnace && !onCeiling) continue;
        if (Math.hypot(hit[0] - eye[0], hit[2] - eye[2]) > NEAR_HIT_METRES) continue;
        nearSum += value;
        nearRays++;
    }
    return { nearMean: nearSum / Math.max(nearRays, 1), nearRays, hitMean: hitSum / Math.max(hitRays, 1), hitRays, bad };
}

/** The mean indirect value (channel mean of rgb / count) of the live tiles' finest cells that have one. */
async function meanIndirect(device: GPUDevice, cache: HashGridCache): Promise<{ mean: number; cells: number }> {
    const packed = cache.packedTiles;
    const count = new Uint32Array(await readRange(device, packed.count.buffer, packed.count.offset, 4))[0];
    if (count === 0) return { mean: 0, cells: 0 };
    const tiles = new Uint32Array(await readRange(device, cache.buffers.tileWords, packed.firstWord * 4, count * 4));
    const capacity = cache.capacity;
    const tileBytes = capacity.numCellsPerTile * 8;
    const staging = device.createBuffer({ label: 'ddgi-bounce tiles', size: count * tileBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'ddgi-bounce tiles' });
    const base = cache.layout.cellValues.valueIndirectBuffer * 8;
    tiles.forEach((tile, index) => encoder.copyBufferToBuffer(cache.buffers.cellValues, base + tile * tileBytes, staging, index * tileBytes, tileBytes));
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readRange(device, staging, 0, count * tileBytes));
    staging.destroy();
    const finest = capacity.numCellsPerTileMip[0];
    let sum = 0;
    let cells = 0;
    for (let tile = 0; tile < count; tile++) {
        for (let cell = 0; cell < finest; cell++) {
            const word = (tile * capacity.numCellsPerTile + cell) * 2;
            const w = HALF_TABLE[words[word + 1] >>> 16];
            if (!(w > 0)) continue;
            const rgb = HALF_TABLE[words[word] & 0xffff] + HALF_TABLE[words[word] >>> 16] + HALF_TABLE[words[word + 1] & 0xffff];
            sum += rgb / 3 / w;
            cells++;
        }
    }
    return { mean: sum / Math.max(cells, 1), cells };
}

async function readCounters(device: GPUDevice, cache: HashGridCache): Promise<Uint32Array> {
    return new Uint32Array(await readRange(device, cache.buffers.tileWords, 0, 64 * 4));
}

const round = (value: number, digits = 4): number => Number(value.toFixed(digits));

// ---- Sections 1-6: the furnace and the strips ---------------------------------------------------

interface AccuracyCase {
    readonly name: string;
    readonly strips: boolean;
    readonly mode: Mode;
    /** DDGI's bounceEnergy (ddgi-volume only). */
    readonly ddgiBounce: number;
    readonly expected: number;
    readonly tolerance: number;
}

const ACCURACY_CASES: readonly AccuracyCase[] = [
    { name: '1 furnace, gi1-rays (second bounce)', strips: false, mode: 'gi1-rays', ddgiBounce: 1, expected: EXPECTED.secondBounce, tolerance: 0.02 },
    { name: '2 furnace, ddgi-volume, DDGI bounce off (second bounce)', strips: false, mode: 'ddgi-volume', ddgiBounce: 0, expected: EXPECTED.secondBounce, tolerance: 0.03 },
    { name: '3 furnace, ddgi-volume, DDGI bounce on (every bounce)', strips: false, mode: 'ddgi-volume', ddgiBounce: 1, expected: EXPECTED.everyBounce, tolerance: 0.03 },
    { name: '4 strips, gi1-rays (second bounce)', strips: true, mode: 'gi1-rays', ddgiBounce: 1, expected: EXPECTED.stripsSecondBounce, tolerance: 0.02 },
    { name: '5 strips, ddgi-volume, DDGI bounce off (second bounce)', strips: true, mode: 'ddgi-volume', ddgiBounce: 0, expected: EXPECTED.stripsSecondBounce, tolerance: 0.03 },
    { name: '6 strips, ddgi-volume, DDGI bounce on (every bounce)', strips: true, mode: 'ddgi-volume', ddgiBounce: 1, expected: EXPECTED.stripsEveryBounce, tolerance: 0.03 },
];

async function runAccuracy(device: GPUDevice, testCase: AccuracyCase, failures: string[]): Promise<Record<string, unknown>> {
    const scenario = await startScenario(device, testCase.name, {
        scene: { kind: 'furnace', strips: testCase.strips },
        settings: { secondBounce: testCase.mode },
        // gi1-rays runs DDGI too (stacked but not read), so both sides carry the same scene and passes.
        ddgi: { ...DDGI_ACCURACY, bounceEnergy: testCase.ddgiBounce },
    });
    try {
        await runFrames(device, scenario, ACCURACY_FRAMES);
        const rays = await readRays(device, scenario, true);
        const indirect = await meanIndirect(device, scenario.cache);
        const counters = await readCounters(device, scenario.cache);
        const ratio = rays.nearMean / testCase.expected;
        if (rays.nearRays < 1000) failures.push(`${testCase.name}: only ${rays.nearRays} near ceiling rays`);
        if (Math.abs(ratio - 1) > testCase.tolerance) {
            failures.push(`${testCase.name}: near ceiling rays read ${rays.nearMean.toFixed(4)}, expected ${testCase.expected.toFixed(4)} within ${testCase.tolerance * 100}% (ratio ${ratio.toFixed(4)})`);
        }
        if (rays.bad > 0) failures.push(`${testCase.name}: ${rays.bad} rays NaN, infinite or negative`);
        if (scenario.cache.secondBounceInUse !== testCase.mode) failures.push(`${testCase.name}: the cache ran ${scenario.cache.secondBounceInUse}, not ${testCase.mode}`);
        const secondBounceEntries = counters[HASH_GRID_COUNTER.visibilityCount1];
        if (testCase.mode === 'ddgi-volume' && secondBounceEntries !== 0) failures.push(`${testCase.name}: ${secondBounceEntries} second-bounce entries; 'ddgi-volume' traces none`);
        if (testCase.mode === 'gi1-rays' && secondBounceEntries === 0) failures.push(`${testCase.name}: no second-bounce entries with 'gi1-rays'`);
        return {
            expected: round(testCase.expected),
            nearCeilingMean: round(rays.nearMean),
            ratio: round(ratio),
            nearRays: rays.nearRays,
            cellIndirectMean: round(indirect.mean),
            cellsWithIndirect: indirect.cells,
            firstBounceEntries: counters[HASH_GRID_COUNTER.visibilityCount0],
            secondBounceEntries,
            shadowRays: counters[HASH_GRID_COUNTER.visibilityRayCount],
        };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- Section 7: nothing stacked ------------------------------------------------------------------

async function runNoFallback(device: GPUDevice, failures: string[]): Promise<Record<string, unknown>> {
    const warnings: string[] = [];
    const consoleWarn = console.warn;
    console.warn = (...args: unknown[]) => {
        warnings.push(args.map(String).join(' '));
        consoleWarn(...args);
    };
    const scenario = await startScenario(device, '7 no fallback', { scene: { kind: 'furnace', strips: false }, settings: { secondBounce: 'ddgi-volume' }, ddgi: null });
    try {
        await runFrames(device, scenario, 4);
        const counters = await readCounters(device, scenario.cache);
        const warned = warnings.some((message) => message.includes("'ddgi-volume' needs a technique stacked under it"));
        if (scenario.cache.secondBounceInUse !== 'gi1-rays') failures.push(`7: with nothing stacked the cache ran ${scenario.cache.secondBounceInUse}, not gi1-rays`);
        if (counters[HASH_GRID_COUNTER.visibilityCount1] === 0) failures.push('7: with nothing stacked no second-bounce ray was traced');
        if (!warned) failures.push('7: no warning that ddgi-volume needs a stacked technique');
        return { inUse: scenario.cache.secondBounceInUse, secondBounceEntries: counters[HASH_GRID_COUNTER.visibilityCount1], warned };
    } finally {
        console.warn = consoleWarn;
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- Section 8: lag after a light change ---------------------------------------------------------

interface LagCase {
    readonly name: string;
    readonly scene: ScenarioOptions['scene'];
    readonly settings: Partial<RadianceCacheSettings>;
}

const LAG_SCENES: readonly { readonly label: string; readonly scene: ScenarioOptions['scene'] }[] = [
    { label: 'furnace, sun doubled', scene: { kind: 'furnace', strips: false } },
    { label: 'two-room, lamp doubled', scene: { kind: 'manifest', id: 'two-room' } },
];
const LAG_SETTINGS: readonly { readonly label: string; readonly settings: Partial<RadianceCacheSettings> }[] = [
    { label: 'gi1-rays', settings: { secondBounce: 'gi1-rays' } },
    { label: 'ddgi-volume', settings: { secondBounce: 'ddgi-volume' } },
    { label: 'ddgi-volume, indirect cap 4', settings: { secondBounce: 'ddgi-volume', hashGridCacheMaxMultibounceSampleCount: 4 } },
];

function framesToSettle(series: readonly number[], target: number): number | null {
    const index = series.findIndex((value) => Math.abs(value / target - 1) <= LAG_TOLERANCE);
    return index < 0 ? null : index + 1;
}

async function runLag(device: GPUDevice, lagCase: LagCase, failures: string[]): Promise<Record<string, unknown>> {
    const furnace = lagCase.scene.kind === 'furnace';
    // DDGI at its defaults (the adaptive hysteresis, the change response, the warm-up).
    const scenario = await startScenario(device, lagCase.name, { scene: lagCase.scene, settings: lagCase.settings, ddgi: {} });
    try {
        await runFrames(device, scenario, LAG_SETTLE_FRAMES - 10);
        const before: { rays: number; cells: number }[] = [];
        for (let index = 0; index < 10; index++) {
            scenario.frame();
            await device.queue.onSubmittedWorkDone();
            const rays = await readRays(device, scenario, furnace);
            before.push({ rays: furnace ? rays.nearMean : rays.hitMean, cells: (await meanIndirect(device, scenario.cache)).mean });
        }
        const baseRays = before.reduce((sum, value) => sum + value.rays, 0) / before.length;
        const baseCells = before.reduce((sum, value) => sum + value.cells, 0) / before.length;
        const light = scenario.lights[0];
        const intensity = furnace ? FURNACE.sunIntensity : loadManifest('two-room').lights[0].intensity;
        scenario.platform.scene.updateLight(light, { intensity: 2 * intensity });
        const raySeries: number[] = [];
        const cellSeries: number[] = [];
        let bad = 0;
        for (let index = 0; index < LAG_MAX_FRAMES; index++) {
            scenario.frame();
            await device.queue.onSubmittedWorkDone();
            const rays = await readRays(device, scenario, furnace);
            bad += rays.bad;
            raySeries.push(furnace ? rays.nearMean : rays.hitMean);
            cellSeries.push((await meanIndirect(device, scenario.cache)).mean);
        }
        if (bad > 0) failures.push(`${lagCase.name}: ${bad} rays NaN, infinite or negative after the change`);
        const pick = (series: readonly number[]) => [1, 2, 5, 10, 20, 40, 80, 160, 240].map((frame) => `${frame}: ${round(series[frame - 1] / (2 * (series === raySeries ? baseRays : baseCells)), 3)}`).join(', ');
        return {
            baseRays: round(baseRays),
            baseCellIndirect: round(baseCells),
            raysFramesToWithin5: framesToSettle(raySeries, 2 * baseRays),
            cellIndirectFramesToWithin5: framesToSettle(cellSeries, 2 * baseCells),
            raysOverTarget: pick(raySeries),
            cellIndirectOverTarget: pick(cellSeries),
        };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- The run ------------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    const accuracy: Record<string, unknown> = {};
    let noFallback: Record<string, unknown> = {};
    if (SECTIONS !== 'lag') {
        for (const testCase of ACCURACY_CASES) accuracy[testCase.name] = await runAccuracy(device, testCase, failures);
        noFallback = await runNoFallback(device, failures);
    }
    const lag: Record<string, unknown> = {};
    for (const { label, scene } of SECTIONS === 'accuracy' ? [] : LAG_SCENES) {
        for (const { label: settingsLabel, settings } of LAG_SETTINGS) {
            const name = `8 ${label}, ${settingsLabel}`;
            lag[name] = await runLag(device, { name, scene, settings }, failures);
        }
    }
    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data: { width: WIDTH, height: HEIGHT, expected: Object.fromEntries(Object.entries(EXPECTED).map(([key, value]) => [key, round(value)])), accuracy, noFallback, lag },
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
