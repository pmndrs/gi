/**
 * P6.1: the SHaRC-style hash cache (ported from pmndrs/pathtracer) in the
 * radiance cache's world-cache slot, against analytic answers, with
 * Capsaicin's GI-1.2 hash grid run through the same checks beside it (A/B).
 * The GI-1.2 grid's own gates are `radiance-cache-world-cache`'s; here its
 * numbers are recorded, and only the SHaRC backend's fail the page.
 *
 * Each check reads the probe rays the screen cache traced and the radiance the
 * world cache wrote into them (`rays.radiance`), every frame from an empty
 * cache, so it also gives the warm-up: the first frame from which the judged
 * rays' mean stays within 2% of the answer ("warm frames"), and the first from
 * which 95% of the judged rays each stay within 2% ("rays warm frames"). The
 * scenes are `radiance-cache-world-cache`'s (black sky, albedo 0.5, 1280 × 720,
 * the screen drawn by primary rays through the platform's tracer):
 *
 * 1. **Lit plane** (the corner: a floor and a wall, the sun along (0, -1, -1),
 *    intensity 8; GI-1.2 with multibounce off, SHaRC with `bounceRays` 0, 40
 *    frames): every hit reads albedo / π × E (E = 8 cos 45°). SHaRC: every
 *    hit ray within 2% from frame 40, its rays both read cells and their own
 *    sample (shorter than their cell), and inserts refused by a full bucket
 *    under 0.1% of the rays (a refused ray reads its own estimate).
 * 2. **Shadow** (the corner and a level slab at y = 1.5 between it and the sun,
 *    40 frames): rays whose hit is in the slab's shadow (the floor for z in
 *    [0, 2], the wall below 0.5 m, under x in [-1, 2]) or under the slab read
 *    0, the others albedo / π × E. Judged only where the hit is further from
 *    the shadow's edges and from the slab than the coarsest cell a lookup can
 *    reach (8 cells of its own level for SHaRC, two tiles for GI-1.2, which
 *    leaves GI-1.2 no ray under 1 m of reach: its own page judges its shadow
 *    cell by cell): the shadowed mean below 2% of the lit value, the lit mean
 *    within 2%. A shadow ray that never hit would light the shadowed rays and
 *    fail.
 * 3. **Furnace, second bounce** (a floor facing up and a ceiling facing down
 *    3 m above, 400 m wide; a sun straight down, intensity 4, that casts no
 *    shadow: it lights the floor, never the ceiling; GI-1.2 with multibounce
 *    on, SHaRC with one bounce ray and no cache tail, 40 frames): the ceiling
 *    rays read the second bounce ρ² E / π = 0.3183. Judged: ceiling hits whose
 *    GI-1.2 tile is under 2 m (beyond, its tiles hold floor and ceiling hits
 *    together; NOTES, "a hash-grid cell is a column through its tile"), the
 *    same rays for both. SHaRC: mean within 1%, 95% of rays within 2%.
 * 4. **Furnace, every bounce** (SHaRC's defaults, two bounce rays and the
 *    cache tail, 200 frames): the ceiling settles at ρ (ρE/π) / (1 − ρ²) =
 *    0.4244 (`sharcFurnaceCeiling`), the light every bounce between the planes
 *    brings. Mean within 2% from frame 200. GI-1.2's two bounces read 25% low
 *    against it; recorded.
 * 5. **Strips, half the rays miss** (the furnace's floor as 5 cm strips every
 *    10 cm): a ray from the ceiling finds the floor half the time, a miss is a
 *    sample of no light. One bounce: ρ² E / π / 2 = 0.1592 (GI-1.2 with
 *    multibounce, SHaRC one bounce ray; a mean over the hits alone would read
 *    twice that); every bounce: `sharcFurnaceCeiling(..., 0.5)` = 0.1819
 *    (SHaRC's defaults, 200 frames). Judged on the near ceiling rays' mean,
 *    within 2% (a ray's value is a noisy mix of hits and misses).
 * 7. **Shadow edge** (section 2's slab, 40 frames, the last 10 pooled): the
 *    floor rays across the shadow's edge at z = 2, binned by distance from it
 *    (4 cm bins); the width from 10% to 90% of the lit value. SHaRC's must be
 *    at most 4 of its finest cells there, and the 50% point within one cell of
 *    the edge. Section 2 leaves rays within 8 cells of the edge out; this is
 *    where coarse reads would show.
 * 8. **Lag and flicker** (the corner, multibounce, the sun doubled after 150
 *    frames, 200 frames after): frames until the mean irradiance E over the
 *    screen, and the mean hit-ray radiance, stay within 5% and 1% of twice
 *    their value before the change; and the flicker of E over the 30 still
 *    frames before it (radiance-cache-stability's measure: per pixel
 *    |ΔE| / max(E, 5% of the mean), mean and p95). For GI-1.2 single and
 *    multibounce, SHaRC's defaults and SHaRC without the change reset, with a
 *    64-sample threshold and with a 16-frame history, so flicker isn't bought
 *    with lag. SHaRC's defaults must reach 5% (E) no later than GI-1.2's
 *    multibounce.
 * 9. **A sky, not black** (section 5's strips and ceiling, no sun, a sky of
 *    radiance 1; SHaRC with one bounce ray and no tail): a bounce ray from the
 *    ceiling that misses through a gap brings the sky (no sky light in the
 *    light choice yet, register E4), one that hits a strip brings its light
 *    (none), so the ceiling reads ρ × 1/2 × 1 = 0.25 within 2%. A sky-on-miss
 *    rule turned the wrong way reads 0. TODO(rc3-4 merge): with
 *    `environmentLightEnable` on, misses bring nothing and the strips' sky
 *    light sample is blocked by the ceiling, so run this with it off.
 * 6. **NaN scan** (SHaRC's defaults on `cornell-box`, `two-room` and `sponza`
 *    with curtains, from the first bookmark, 60 frames): every probe ray's
 *    radiance and every resolved slot finite and at least 0 after frames 1,
 *    30 and 60; cells in use and refused inserts recorded.
 *
 * Correctness only; no timing here (`radiance-cache-sharc-timing` records it).
 * Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "p6.1 sharc" -- pnpm test:gpu --filter radiance-cache-world-cache-sharc --gpu nvidia`.
 */

import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat, unpackHalf3 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { hashGridCacheCellSizeConstant, hashGridCacheGetCellSize, verticalFieldOfView } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import type { SharcCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/sharc/sharc-cache.ts';
import { sharcCellSize, sharcCellSizePerMetre, sharcFurnaceCeiling, sharcLevel } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/sharc/sharc-cpu.ts';
import { DEFAULT_SHARC_SETTINGS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/sharc/sharc-settings.ts';
import { SHARC_COUNTER, SHARC_COUNTER_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/sharc/sharc-wgsl.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const TOLERANCE = 0.02;
const ALBEDO = 0.5;

const CORNER_CAMERA: PageCamera = { position: [0, 2, 6], target: [0, 1, 0], fovYDeg: 60, near: 0.05, far: 1000 };
const SUN_INTENSITY = 8;
const SUN_TRAVEL: Vec3 = [0, -1, -1];
/** albedo / π × E, E = 8 cos 45° on the floor and the wall. */
const LIT_RADIANCE = (ALBEDO / Math.PI) * SUN_INTENSITY * Math.SQRT1_2;
const SLAB = { minX: -1, maxX: 2, minZ: 1, maxZ: 3.5, y: 1.5 } as const;

const FURNACE_CAMERA: PageCamera = { position: [0, 2, 0], target: [0, 0, -2], fovYDeg: 60, near: 0.05, far: 1000 };
const CEILING_Y = 3;
const FURNACE_SUN_IRRADIANCE = 4;
const FURNACE_SUN: LightDescriptor = { type: 'directional', direction: [0, -1, 0], color: [1, 1, 1], intensity: FURNACE_SUN_IRRADIANCE, castsShadows: false, label: 'furnace sun' };
const SECOND_BOUNCE = sharcFurnaceCeiling(ALBEDO, FURNACE_SUN_IRRADIANCE, 'one-bounce');
const EVERY_BOUNCE = sharcFurnaceCeiling(ALBEDO, FURNACE_SUN_IRRADIANCE, 'every-bounce');
const STRIPS = { width: 0.05, period: 0.1, halfLength: 100, fill: 0.5 } as const;
const STRIPS_SECOND_BOUNCE = sharcFurnaceCeiling(ALBEDO, FURNACE_SUN_IRRADIANCE, 'one-bounce', STRIPS.fill);
const STRIPS_EVERY_BOUNCE = sharcFurnaceCeiling(ALBEDO, FURNACE_SUN_IRRADIANCE, 'every-bounce', STRIPS.fill);

/** The GI-1.2 grid's cell-size constant at Capsaicin's defaults (32 px, min 0.1 m, tiles of 8 cells), for the judged-ray filters. */
const GI1_CELL_CONSTANTS = { cellSize: hashGridCacheCellSizeConstant(verticalFieldOfView(frameCamera(CORNER_CAMERA, WIDTH, HEIGHT).projection), WIDTH, HEIGHT, 32), minCellSize: 0.1 };
const GI1_TILE_CELLS = 8;

const GI1_SINGLE: Partial<RadianceCacheSettings> = { worldCache: 'gi1-hash-grid', useMultibounce: false };
const GI1_MULTI: Partial<RadianceCacheSettings> = { worldCache: 'gi1-hash-grid', useMultibounce: true };
const SHARC_DIRECT: Partial<RadianceCacheSettings> = { worldCache: 'sharc', sharc: { bounceRays: 0 } };
const SHARC_ONE_BOUNCE: Partial<RadianceCacheSettings> = { worldCache: 'sharc', sharc: { bounceRays: 1, cacheTail: false } };
const SHARC_DEFAULT: Partial<RadianceCacheSettings> = { worldCache: 'sharc' };

type SceneKind = 'corner' | 'slab' | 'furnace' | 'strips' | 'skyStrips';
/** Section 9's sky: a constant radiance. */
const SKY_RADIANCE = 1;
const SKY_STRIPS_CEILING = ALBEDO * (1 - STRIPS.fill) * SKY_RADIANCE;
type Surface = 'floor' | 'wall' | 'slab' | 'ceiling' | 'other';

interface ProbeRay {
    readonly origin: Vec3;
    readonly direction: Vec3;
    readonly hitT: number;
    readonly hit: Vec3;
    readonly surface: Surface;
}

// ---- Scenes ---------------------------------------------------------------------------------------

function quad(platform: GIPlatformImpl, label: string, corners: readonly Vec3[], normal: Vec3): void {
    const scene = platform.scene;
    const mesh = scene.createMesh({ positions: new Float32Array(corners.flat()), normals: new Float32Array([...normal, ...normal, ...normal, ...normal]), indices: new Uint32Array([0, 1, 2, 0, 2, 3]), label });
    const material = scene.createMaterial({ albedo: [ALBEDO, ALBEDO, ALBEDO], label });
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
    const material = scene.createMaterial({ albedo: [ALBEDO, ALBEDO, ALBEDO], label: 'strip floor' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'strip floor' });
}

function describe(platform: GIPlatformImpl, kind: SceneKind): PageCamera {
    const half = 200;
    if (kind === 'furnace' || kind === 'strips' || kind === 'skyStrips') {
        if (kind === 'furnace') quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0]);
        else stripFloor(platform);
        quad(platform, 'ceiling', [[-half, CEILING_Y, -half], [half, CEILING_Y, -half], [half, CEILING_Y, half], [-half, CEILING_Y, half]], [0, -1, 0]);
        if (kind === 'skyStrips') {
            platform.scene.setEnvironment({ type: 'color', color: [SKY_RADIANCE, SKY_RADIANCE, SKY_RADIANCE], intensity: 1 });
        } else {
            platform.scene.createLight(FURNACE_SUN);
            platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
        }
        return FURNACE_CAMERA;
    }
    quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0]);
    quad(platform, 'wall', [[-20, 0, 0], [20, 0, 0], [20, 10, 0], [-20, 10, 0]], [0, 0, 1]);
    if (kind === 'slab') {
        const { minX, maxX, minZ, maxZ, y } = SLAB;
        quad(platform, 'slab', [[minX, y, maxZ], [maxX, y, maxZ], [maxX, y, minZ], [minX, y, minZ]], [0, 1, 0]);
    }
    platform.scene.createLight({ type: 'directional', direction: SUN_TRAVEL, color: [1, 1, 1], intensity: SUN_INTENSITY, label: 'sun' });
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    return CORNER_CAMERA;
}

function classify(hit: Vec3, kind: SceneKind): Surface {
    if (kind === 'furnace' || kind === 'strips' || kind === 'skyStrips') return Math.abs(hit[1] - CEILING_Y) < 1e-3 ? 'ceiling' : Math.abs(hit[1]) < 1e-3 ? 'floor' : 'other';
    if (kind === 'slab' && Math.abs(hit[1] - SLAB.y) < 1e-3 && hit[0] >= SLAB.minX - 1e-3 && hit[0] <= SLAB.maxX + 1e-3 && hit[2] >= SLAB.minZ - 1e-3 && hit[2] <= SLAB.maxZ + 1e-3) return 'slab';
    if (Math.abs(hit[1]) < 1e-3) return 'floor';
    if (Math.abs(hit[2]) < 1e-3) return 'wall';
    return 'other';
}

// ---- Running a scenario ---------------------------------------------------------------------------

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const readback = device.createBuffer({ label: 'sharc page readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'sharc page readback' });
    encoder.copyBufferToBuffer(source, offset, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0, byteLength);
    readback.unmap();
    readback.destroy();
    return copy;
}

interface FrameState {
    readonly rays: ProbeRay[];
    /** Per ray: rgb and the hit distance, as halves unpacked. */
    readonly radiance: Float32Array;
}

async function readFrame(device: GPUDevice, technique: RadianceCacheTechnique, kind: SceneKind): Promise<FrameState> {
    const probes = technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const count = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
    const words = new Uint32Array(await readRange(device, probes.probeRays, 0, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + count * WORLD_CACHE_PROBE_RAY_BYTES));
    const floats = new Float32Array(words.buffer);
    const rays: ProbeRay[] = [];
    for (let ray = 0; ray < count; ray++) {
        const base = (WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES) / 4;
        const origin: Vec3 = [floats[base], floats[base + 1], floats[base + 2]];
        const direction = unpackHalf3([words[base + 4], words[base + 5]]);
        const hitT = floats[base + 6];
        const hit: Vec3 = [origin[0] + hitT * direction[0], origin[1] + hitT * direction[1], origin[2] + hitT * direction[2]];
        rays.push({ origin, direction, hitT, hit, surface: hitT >= 0 ? classify(hit, kind) : 'other' });
    }
    const results = probes.rayScratch.rayResults;
    const packed = new Uint32Array(await readRange(device, results.buffer, results.offset ?? 0, count * 8));
    const radiance = new Float32Array(count * 4);
    for (let index = 0; index < packed.length; index++) {
        radiance[index * 2] = halfBitsToFloat(packed[index] & 0xffff);
        radiance[index * 2 + 1] = halfBitsToFloat(packed[index] >>> 16);
    }
    return { rays, radiance };
}

/** A judged ray's expected value, or null to leave it out. */
type Judge = (ray: ProbeRay, backend: 'gi1' | 'sharc') => number | null;

interface Run {
    readonly backend: 'gi1' | 'sharc';
    readonly frames: number;
    /** Per frame: the judged rays' mean relative error (signed) and the share of them within 2%. */
    readonly meanError: number[];
    readonly withinShare: number[];
    readonly judged: number;
    readonly last: FrameState;
    readonly counters: Record<string, number>;
}

/** The first frame (1-based) from which every later frame's value passes; null if the last doesn't. */
function warmFrame(passes: readonly boolean[]): number | null {
    let first: number | null = null;
    for (let index = 0; index < passes.length; index++) {
        if (passes[index]) first ??= index + 1;
        else first = null;
    }
    return first;
}

async function runScenario(device: GPUDevice, kind: SceneKind, settings: Partial<RadianceCacheSettings>, frames: number, judge: Judge, observe?: (state: FrameState, frame: number) => void): Promise<Run> {
    const backend = settings.worldCache === 'sharc' ? 'sharc' : 'gi1';
    const platform = new GIPlatformImpl(device, { label: `sharc page ${kind} ${backend}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    const camera = describe(platform, kind);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `sharc page ${kind}` });
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    try {
        frame();
        await platform.rays.whenBuilt();
        frame();
        await device.queue.onSubmittedWorkDone();
        technique.resetHistory();
        const meanError: number[] = [];
        const withinShare: number[] = [];
        let state: FrameState | null = null;
        let judged = 0;
        for (let index = 0; index < frames; index++) {
            frame();
            await device.queue.onSubmittedWorkDone();
            state = await readFrame(device, technique, kind);
            observe?.(state, index);
            let sum = 0;
            let expectedSum = 0;
            let within = 0;
            judged = 0;
            state.rays.forEach((ray, rayIndex) => {
                const expected = judge(ray, backend);
                if (expected === null) return;
                judged++;
                const value = (state!.radiance[rayIndex * 4] + state!.radiance[rayIndex * 4 + 1] + state!.radiance[rayIndex * 4 + 2]) / 3;
                sum += value;
                expectedSum += expected;
                if (Math.abs(value - expected) <= TOLERANCE * Math.max(expected, LIT_RADIANCE * 0.05)) within++;
            });
            meanError.push(expectedSum > 0 ? (sum - expectedSum) / expectedSum : sum / Math.max(judged, 1));
            withinShare.push(within / Math.max(judged, 1));
        }
        if (!state) throw new Error('no frame ran');
        const counters: Record<string, number> = {};
        const sharc = technique.worldCacheBackend;
        if (backend === 'sharc' && sharc) {
            const words = new Uint32Array(await readRange(device, (sharc as SharcCache).buffers.accum, 0, SHARC_COUNTER_WORDS * 4));
            for (const [name, word] of Object.entries(SHARC_COUNTER)) counters[name] = words[word];
        }
        return { backend, frames, meanError, withinShare, judged, last: state, counters };
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

function summary(run: Run, meanTolerance = TOLERANCE): Record<string, unknown> {
    return {
        judgedRays: run.judged,
        finalMeanError: Number(run.meanError[run.meanError.length - 1].toFixed(5)),
        finalWithinShare: Number(run.withinShare[run.withinShare.length - 1].toFixed(4)),
        warmFrames: warmFrame(run.meanError.map((error) => Math.abs(error) <= meanTolerance)),
        raysWarmFrames: warmFrame(run.withinShare.map((share) => share >= 0.95)),
        counters: run.counters,
    };
}

// ---- Judges ----------------------------------------------------------------------------------------

function gi1TileSize(hit: Vec3, camera: PageCamera): number {
    return hashGridCacheGetCellSize(hit, camera.position, GI1_CELL_CONSTANTS) * GI1_TILE_CELLS;
}

const SHARC_CONSTANTS = (camera: PageCamera) => ({ eye: camera.position, cellSizePerMetre: sharcCellSizePerMetre((camera.fovYDeg * Math.PI) / 180, HEIGHT, DEFAULT_SHARC_SETTINGS.cellPixels), minCell: DEFAULT_SHARC_SETTINGS.minCell });

/** Every hit on the floor or the wall reads the lit value. */
const litPlaneJudge: Judge = (ray) => (ray.hitT >= 0 && (ray.surface === 'floor' || ray.surface === 'wall') ? LIT_RADIANCE : null);

/** Section 2: 0 or the lit value where the hit is clear of the shadow's edges and the slab by the reach of a lookup. */
const shadowJudge: Judge = (ray, backend) => {
    if (ray.hitT < 0) return null;
    const [x, y, z] = ray.hit;
    const reach = backend === 'sharc' ? 8 * sharcCellSize(sharcLevel(ray.hit, SHARC_CONSTANTS(CORNER_CAMERA)), DEFAULT_SHARC_SETTINGS.minCell) : 2 * gi1TileSize(ray.hit, CORNER_CAMERA);
    if (reach > 1) return null;
    if (ray.surface === 'slab') return ray.direction[1] < 0 ? LIT_RADIANCE : 0;
    const xInside = x > SLAB.minX + reach && x < SLAB.maxX - reach;
    const xOutside = x < SLAB.minX - reach || x > SLAB.maxX + reach;
    if (ray.surface === 'floor') {
        // Clear of the slab above too (its underside shares the floor's cells only within reach).
        if (Math.abs(SLAB.y - y) < reach) return null;
        if (xOutside || z > 2 + reach) return LIT_RADIANCE;
        if (xInside && z < 2 - reach) return 0;
        return null;
    }
    if (ray.surface === 'wall') {
        if (xOutside || y > 0.5 + reach) return y > SLAB.y - reach && y < SLAB.y + reach ? null : LIT_RADIANCE;
        if (xInside && y < 0.5 - reach) return 0;
        return null;
    }
    return null;
};

/** The furnace's ceiling rays whose GI-1.2 tile is under 2 m. */
function ceilingJudge(expected: number): Judge {
    return (ray) => (ray.hitT >= 0 && ray.surface === 'ceiling' && ray.direction[1] > 0 && gi1TileSize(ray.hit, FURNACE_CAMERA) < 2 ? expected : null);
}

// ---- Section 7: the shadow's edge -----------------------------------------------------------------

/** Floor rays across the slab shadow's edge at z = 2, away from its ends, binned by z - 2. */
class EdgeProfile {
    static readonly BIN = 0.04;
    static readonly REACH = 0.6;
    readonly sums = new Float64Array(Math.round((2 * EdgeProfile.REACH) / EdgeProfile.BIN));
    readonly counts = new Uint32Array(this.sums.length);

    add(state: FrameState): void {
        state.rays.forEach((ray, index) => {
            if (ray.hitT < 0 || ray.surface !== 'floor') return;
            const [x, , z] = ray.hit;
            if (x < SLAB.minX + 0.3 || x > SLAB.maxX - 0.3 || Math.abs(z - 2) >= EdgeProfile.REACH) return;
            const bin = Math.floor((z - 2 + EdgeProfile.REACH) / EdgeProfile.BIN);
            this.sums[bin] += (state.radiance[index * 4] + state.radiance[index * 4 + 1] + state.radiance[index * 4 + 2]) / 3;
            this.counts[bin]++;
        });
    }

    /** The 10%, 50% and 90% crossings of the lit value, from the shadow side (negative) to the lit side, in metres from the edge. */
    result(): { width: number; middle: number; profile: number[] } {
        const profile = Array.from(this.sums, (sum, bin) => (this.counts[bin] > 0 ? sum / this.counts[bin] / LIT_RADIANCE : NaN));
        const crossing = (level: number): number => {
            for (let bin = 1; bin < profile.length; bin++) {
                const a = profile[bin - 1];
                const b = profile[bin];
                if (!(a < level && b >= level)) continue;
                const t = (level - a) / (b - a);
                return -EdgeProfile.REACH + (bin - 0.5 + t) * EdgeProfile.BIN;
            }
            return NaN;
        };
        return { width: crossing(0.9) - crossing(0.1), middle: crossing(0.5), profile: profile.map((value) => Number(value.toFixed(3))) };
    }
}

// ---- Section 8: lag and flicker ---------------------------------------------------------------------

const HALF_TABLE = Float32Array.from({ length: 65536 }, (_, bits) => halfBitsToFloat(bits));

interface LagResult {
    readonly flickerMean: number;
    readonly flickerP95: number;
    readonly irradianceFramesTo5: number | null;
    readonly irradianceFramesTo1: number | null;
    readonly rayFramesTo5: number | null;
    readonly rayFramesTo1: number | null;
    /** E's mean over the last frame, over twice its mean before the change. */
    readonly finalIrradianceRatio: number;
    readonly counters: Record<string, number>;
}

/** The first frame after the change (1-based) from which every later frame is within `tolerance` of the target. */
function settledAfter(values: readonly number[], target: number, tolerance: number): number | null {
    return warmFrame(values.map((value) => Math.abs(value / target - 1) <= tolerance));
}

async function runLag(device: GPUDevice, settings: Partial<RadianceCacheSettings>): Promise<LagResult> {
    const BEFORE = 150;
    const AFTER = 200;
    const STILL = 30;
    const platform = new GIPlatformImpl(device, { label: 'sharc page lag' });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    const camera = describe(platform, 'corner');
    const sun = platform.scene.createLight({ type: 'directional', direction: SUN_TRAVEL, color: [1, 1, 1], intensity: SUN_INTENSITY, label: 'lag sun' });
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: 'sharc page lag' });
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    // describe() makes the corner's sun; a second one along the same direction, dark until the change and
    // then as bright, doubles the light (a change the platform reports as global, like a sun change).
    platform.scene.updateLight(sun, { intensity: 0 });
    try {
        frame();
        await platform.rays.whenBuilt();
        frame();
        await device.queue.onSubmittedWorkDone();
        technique.resetHistory();
        const depth = new Float32Array(WIDTH * HEIGHT);
        const readE = async (): Promise<Float32Array> => {
            const probes = technique.probes;
            if (!probes) throw new Error('lag: no probes');
            const bytes = await readTexture(device, probes.irradiance, 8, WIDTH, HEIGHT);
            const halves = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
            const e = new Float32Array(WIDTH * HEIGHT);
            for (let pixel = 0; pixel < e.length; pixel++) e[pixel] = (HALF_TABLE[halves[pixel * 4]] + HALF_TABLE[halves[pixel * 4 + 1]] + HALF_TABLE[halves[pixel * 4 + 2]]) / 3;
            return e;
        };
        const surfaceMean = (e: Float32Array): number => {
            let sum = 0;
            let count = 0;
            for (let pixel = 0; pixel < e.length; pixel++) {
                if (depth[pixel] === 1) continue;
                sum += e[pixel];
                count++;
            }
            return sum / Math.max(count, 1);
        };
        const rayMean = async (): Promise<number> => {
            const state = await readFrame(device, technique, 'corner');
            let sum = 0;
            let count = 0;
            state.rays.forEach((ray, index) => {
                if (ray.hitT < 0) return;
                sum += (state.radiance[index * 4] + state.radiance[index * 4 + 1] + state.radiance[index * 4 + 2]) / 3;
                count++;
            });
            return sum / Math.max(count, 1);
        };
        const changes: number[] = [];
        let previous: Float32Array | null = null;
        let beforeE = 0;
        let beforeRay = 0;
        for (let index = 0; index < BEFORE; index++) {
            frame();
            if (index < BEFORE - STILL) {
                if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
                continue;
            }
            await device.queue.onSubmittedWorkDone();
            if (index === BEFORE - STILL) depth.set(new Float32Array((await readTexture(device, screen.screen.depth, 4, WIDTH, HEIGHT)).buffer));
            const e = await readE();
            const mean = surfaceMean(e);
            beforeE += mean / STILL;
            beforeRay += (await rayMean()) / STILL;
            if (previous) {
                const floor = 0.05 * mean;
                for (let pixel = 0; pixel < e.length; pixel++) {
                    if (depth[pixel] === 1) continue;
                    changes.push(Math.abs(e[pixel] - previous[pixel]) / Math.max(previous[pixel], floor));
                }
            }
            previous = e;
        }
        platform.scene.updateLight(sun, { intensity: SUN_INTENSITY });
        const eAfter: number[] = [];
        const rayAfter: number[] = [];
        for (let index = 0; index < AFTER; index++) {
            frame();
            await device.queue.onSubmittedWorkDone();
            eAfter.push(surfaceMean(await readE()));
            rayAfter.push(await rayMean());
        }
        changes.sort((a, b) => a - b);
        const counters: Record<string, number> = {};
        const backend = technique.worldCacheBackend;
        if (backend?.id === 'sharc') {
            const words = new Uint32Array(await readRange(device, (backend as SharcCache).buffers.accum, 0, SHARC_COUNTER_WORDS * 4));
            for (const [counter, word] of Object.entries(SHARC_COUNTER)) counters[counter] = words[word];
        }
        return {
            flickerMean: Number((changes.reduce((sum, value) => sum + value, 0) / Math.max(changes.length, 1)).toFixed(5)),
            flickerP95: Number((changes[Math.floor(0.95 * (changes.length - 1))] ?? NaN).toFixed(4)),
            irradianceFramesTo5: settledAfter(eAfter, 2 * beforeE, 0.05),
            irradianceFramesTo1: settledAfter(eAfter, 2 * beforeE, 0.01),
            rayFramesTo5: settledAfter(rayAfter, 2 * beforeRay, 0.05),
            rayFramesTo1: settledAfter(rayAfter, 2 * beforeRay, 0.01),
            finalIrradianceRatio: Number((eAfter[eAfter.length - 1] / (2 * beforeE)).toFixed(4)),
            counters,
        };
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- Section 6: the NaN scan ---------------------------------------------------------------------

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

interface Scan {
    readonly frame: number;
    readonly probeRays: number;
    readonly rayValuesBad: number;
    readonly cellsInUse: number;
    readonly cellValuesBad: number;
    readonly droppedInserts: number;
    readonly meanRayRadiance: number;
}

async function scanScene(device: GPUDevice, id: string, loader: 'manifest' | 'viewer'): Promise<Scan[]> {
    const manifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `sharc page scan ${id}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(SHARC_DEFAULT));
    platform.setOutput(technique);
    await platform.prepare();
    let beforeFrame = (): void => {};
    if (loader === 'viewer') beforeFrame = await describeLoaded(platform, device, manifest);
    else describeScene(platform, manifest);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    const frame = (): void => {
        beforeFrame();
        const encoder = device.createCommandEncoder({ label: `sharc page scan ${id}` });
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    try {
        frame();
        await platform.rays.whenBuilt();
        technique.resetHistory();
        const scans: Scan[] = [];
        for (let index = 1; index <= 60; index++) {
            frame();
            if (index % 8 === 0) await device.queue.onSubmittedWorkDone();
            if (index !== 1 && index !== 30 && index !== 60) continue;
            await device.queue.onSubmittedWorkDone();
            const probes = technique.probes;
            const cache = technique.worldCacheBackend as SharcCache | null;
            if (!probes || !cache) throw new Error(`${id}: no probes or SHaRC cache`);
            const count = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
            const packed = new Uint32Array(await readRange(device, probes.rayScratch.rayResults.buffer, probes.rayScratch.rayResults.offset ?? 0, count * 8));
            let rayValuesBad = 0;
            let raySum = 0;
            for (let ray = 0; ray < count; ray++) {
                const rgb = [halfBitsToFloat(packed[ray * 2] & 0xffff), halfBitsToFloat(packed[ray * 2] >>> 16), halfBitsToFloat(packed[ray * 2 + 1] & 0xffff)];
                if (rgb.some((value) => !(Number.isFinite(value) && value >= 0))) rayValuesBad++;
                else raySum += (rgb[0] + rgb[1] + rgb[2]) / 3;
            }
            const keys = new Uint32Array(await readRange(device, cache.buffers.keys, 0, cache.entries * 4));
            const resolved = new Float32Array(await readRange(device, cache.buffers.resolved, 0, cache.entries * 16));
            const counters = new Uint32Array(await readRange(device, cache.buffers.accum, 0, SHARC_COUNTER_WORDS * 4));
            let cellsInUse = 0;
            let cellValuesBad = 0;
            for (let slot = 0; slot < cache.entries; slot++) {
                if (keys[slot] === 0) continue;
                cellsInUse++;
                for (let channel = 0; channel < 4; channel++) {
                    const value = resolved[slot * 4 + channel];
                    if (!(Number.isFinite(value) && value >= 0)) {
                        cellValuesBad++;
                        break;
                    }
                }
            }
            scans.push({ frame: index, probeRays: count, rayValuesBad, cellsInUse, cellValuesBad, droppedInserts: counters[SHARC_COUNTER.droppedInserts], meanRayRadiance: Number((raySum / Math.max(count, 1)).toPrecision(5)) });
        }
        return scans;
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- The page ----------------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    const fail = (message: string): void => {
        failures.push(message);
    };
    const data: Record<string, unknown> = {};

    // 1. Lit plane.
    {
        const gi1 = await runScenario(device, 'corner', GI1_SINGLE, 40, litPlaneJudge);
        const sharc = await runScenario(device, 'corner', SHARC_DIRECT, 40, litPlaneJudge);
        data.litPlane = { expected: LIT_RADIANCE, gi1: summary(gi1), sharc: summary(sharc) };
        if (sharc.withinShare[sharc.withinShare.length - 1] < 1) fail(`lit plane: ${((1 - sharc.withinShare[sharc.withinShare.length - 1]) * 100).toFixed(2)}% of SHaRC's hit rays are more than 2% off ${LIT_RADIANCE.toFixed(4)}`);
        if (!(sharc.counters.ownReads > 0 && sharc.counters.cacheReads > 0)) fail(`lit plane: SHaRC's rays should both read cells and their own samples (${sharc.counters.cacheReads} cell reads, ${sharc.counters.ownReads} own)`);
        // A full bucket refuses an insert (open addressing in buckets of 8, as the pathtracer's): at a
        // third of the slots in use a few dozen buckets of 131,072 are full, and a ray whose cell was
        // refused reads its own estimate. Judged as a share of the rays, not as zero.
        if (sharc.counters.droppedInserts > 1e-3 * sharc.counters.rayCount) fail(`lit plane: ${sharc.counters.droppedInserts} SHaRC inserts refused (bucket full) for ${sharc.counters.rayCount} rays, over 0.1%`);
    }

    // 2. Shadow.
    {
        const lit = (ray: ProbeRay, backend: 'gi1' | 'sharc'): number | null => (shadowJudge(ray, backend) === LIT_RADIANCE ? LIT_RADIANCE : null);
        const dark = (ray: ProbeRay, backend: 'gi1' | 'sharc'): number | null => (shadowJudge(ray, backend) === 0 ? 0 : null);
        const results: Record<string, unknown> = {};
        for (const [name, settings] of [['gi1', GI1_SINGLE], ['sharc', SHARC_DIRECT]] as const) {
            const edge = new EdgeProfile();
            const litRun = await runScenario(device, 'slab', settings, 40, lit, (state, frame) => {
                if (frame >= 30) edge.add(state);
            });
            const profile = edge.result();
            const finest = sharcCellSize(sharcLevel([0.5, 0, 2], SHARC_CONSTANTS(CORNER_CAMERA)), DEFAULT_SHARC_SETTINGS.minCell);
            results[`${name}Edge`] = { ...profile, finestCell: finest };
            if (name === 'sharc') {
                if (!(profile.width <= 4 * finest)) fail(`shadow edge: SHaRC's 10-90% width is ${profile.width.toFixed(3)} m, over 4 finest cells (${(4 * finest).toFixed(3)} m)`);
                if (!(Math.abs(profile.middle) <= finest)) fail(`shadow edge: SHaRC's 50% point is ${profile.middle.toFixed(3)} m from the edge, over one finest cell (${finest.toFixed(3)} m)`);
            }
            const darkRun = await runScenario(device, 'slab', settings, 40, dark);
            const darkMean = darkRun.meanError[darkRun.meanError.length - 1];
            results[name] = { lit: summary(litRun), darkMeanOverLit: Number((darkMean / LIT_RADIANCE).toFixed(5)), darkRays: darkRun.judged };
            if (name === 'sharc') {
                if (Math.abs(litRun.meanError[litRun.meanError.length - 1]) > TOLERANCE) fail(`shadow: SHaRC's lit rays' mean is ${(litRun.meanError[litRun.meanError.length - 1] * 100).toFixed(2)}% off`);
                if (darkMean > TOLERANCE * LIT_RADIANCE) fail(`shadow: SHaRC's shadowed rays read ${darkMean.toFixed(4)} on average, over 2% of the lit ${LIT_RADIANCE.toFixed(4)}`);
                if (litRun.judged < 100 || darkRun.judged < 100) fail(`shadow: too few judged rays (${litRun.judged} lit, ${darkRun.judged} shadowed)`);
            }
        }
        data.shadow = results;
    }

    // 3-4. Furnace: GI-1.2's two bounces, SHaRC's one bounce ray, SHaRC's every bounce.
    {
        const gi1 = await runScenario(device, 'furnace', GI1_MULTI, 40, ceilingJudge(SECOND_BOUNCE));
        const oneBounce = await runScenario(device, 'furnace', SHARC_ONE_BOUNCE, 40, ceilingJudge(SECOND_BOUNCE));
        const everyBounce = await runScenario(device, 'furnace', SHARC_DEFAULT, 200, ceilingJudge(EVERY_BOUNCE));
        const gi1Final = gi1.meanError[gi1.meanError.length - 1];
        data.furnace = {
            secondBounce: SECOND_BOUNCE,
            everyBounce: EVERY_BOUNCE,
            gi1: { ...summary(gi1), meanErrorAgainstEveryBounce: Number(((1 + gi1Final) * (SECOND_BOUNCE / EVERY_BOUNCE) - 1).toFixed(5)) },
            sharcOneBounce: summary(oneBounce, 0.01),
            sharcEveryBounce: summary(everyBounce),
        };
        const oneFinal = oneBounce.meanError[oneBounce.meanError.length - 1];
        if (Math.abs(oneFinal) > 0.01) fail(`furnace, one bounce: SHaRC's ceiling rays average ${(oneFinal * 100).toFixed(2)}% off ${SECOND_BOUNCE.toFixed(4)}`);
        if (oneBounce.withinShare[oneBounce.withinShare.length - 1] < 0.95) fail(`furnace, one bounce: only ${(oneBounce.withinShare[oneBounce.withinShare.length - 1] * 100).toFixed(1)}% of SHaRC's ceiling rays within 2%`);
        const everyFinal = everyBounce.meanError[everyBounce.meanError.length - 1];
        if (Math.abs(everyFinal) > TOLERANCE) fail(`furnace, every bounce: SHaRC's ceiling rays average ${(everyFinal * 100).toFixed(2)}% off ${EVERY_BOUNCE.toFixed(4)} after 200 frames`);
        if (oneBounce.judged < 1000) fail(`furnace: only ${oneBounce.judged} judged ceiling rays`);
    }

    // 5. Strips: half the bounce rays miss.
    {
        const gi1 = await runScenario(device, 'strips', GI1_MULTI, 40, ceilingJudge(STRIPS_SECOND_BOUNCE));
        const oneBounce = await runScenario(device, 'strips', SHARC_ONE_BOUNCE, 40, ceilingJudge(STRIPS_SECOND_BOUNCE));
        const everyBounce = await runScenario(device, 'strips', SHARC_DEFAULT, 200, ceilingJudge(STRIPS_EVERY_BOUNCE));
        data.strips = { secondBounce: STRIPS_SECOND_BOUNCE, everyBounce: STRIPS_EVERY_BOUNCE, gi1: summary(gi1), sharcOneBounce: summary(oneBounce), sharcEveryBounce: summary(everyBounce) };
        for (const [name, run, expected] of [['one bounce', oneBounce, STRIPS_SECOND_BOUNCE], ['every bounce', everyBounce, STRIPS_EVERY_BOUNCE]] as const) {
            const final = run.meanError[run.meanError.length - 1];
            if (Math.abs(final) > TOLERANCE) fail(`strips, ${name}: SHaRC's ceiling rays average ${(final * 100).toFixed(2)}% off ${expected.toFixed(4)}`);
        }
    }

    // 8. Lag and flicker.
    {
        const configurations: readonly (readonly [string, Partial<RadianceCacheSettings>])[] = [
            ['gi1 single bounce', GI1_SINGLE],
            ['gi1 multibounce', GI1_MULTI],
            ['sharc', SHARC_DEFAULT],
            ['sharc, no change reset', { worldCache: 'sharc', sharc: { changeReset: false } }],
            ['sharc, 64-sample threshold', { worldCache: 'sharc', sharc: { minSamples: 64 } }],
            ['sharc, 16-frame history, no change reset', { worldCache: 'sharc', sharc: { historyFrames: 16, changeReset: false } }],
        ];
        const lag: Record<string, LagResult> = {};
        for (const [name, settings] of configurations) lag[name] = await runLag(device, settings);
        data.lag = lag;
        const sharcFrames = lag.sharc.irradianceFramesTo5;
        const gi1Frames = lag['gi1 multibounce'].irradianceFramesTo5;
        if (sharcFrames === null || (gi1Frames !== null && sharcFrames > gi1Frames)) fail(`lag: SHaRC's E reaches 5% after ${sharcFrames ?? 'never'} frames, GI-1.2 multibounce's after ${gi1Frames ?? 'never'}`);
    }

    // 9. A sky, not black.
    {
        const run = await runScenario(device, 'skyStrips', SHARC_ONE_BOUNCE, 40, ceilingJudge(SKY_STRIPS_CEILING));
        data.sky = { expected: SKY_STRIPS_CEILING, sharcOneBounce: summary(run) };
        const final = run.meanError[run.meanError.length - 1];
        if (!(Math.abs(final) <= TOLERANCE)) fail(`sky: SHaRC's ceiling rays average ${(final * 100).toFixed(2)}% off ${SKY_STRIPS_CEILING.toFixed(4)} (the sky through bounce rays that miss)`);
    }

    // 6. NaN scan.
    {
        const scans: Record<string, Scan[]> = {};
        for (const [id, loader] of [['cornell-box', 'manifest'], ['two-room', 'manifest'], ['sponza', 'viewer']] as const) {
            scans[id] = await scanScene(device, id, loader);
            for (const scan of scans[id]) {
                if (scan.rayValuesBad + scan.cellValuesBad > 0) fail(`${id} after frame ${scan.frame}: ${scan.rayValuesBad} ray and ${scan.cellValuesBad} cell values are NaN, infinite or negative`);
                if (scan.probeRays === 0 || scan.cellsInUse === 0) fail(`${id} after frame ${scan.frame}: ${scan.probeRays} probe rays and ${scan.cellsInUse} cells; nothing to scan`);
            }
        }
        data.nanScan = scans;
    }

    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 && deviceErrors.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data,
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
