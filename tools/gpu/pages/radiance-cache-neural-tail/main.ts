/**
 * P6.4 (register E11): the neural tail, pmndrs/pathtracer's network
 * (`@pathtracer/mlp`) as the second bounce's tail behind the GI-1.2 hash grid
 * (`neuralCache: 'tail'`), against analytic answers, with GI-1.2's
 * multibounce (`neuralCache: 'off'`, the default) run through the same checks
 * beside it (A/B). GI-1.2's own gates are `radiance-cache-world-cache`'s; here
 * its numbers are recorded and only the tail's fail the page.
 *
 * Each check reads the probe rays the screen cache traced and the radiance the
 * world cache wrote into them, every frame from an empty cache and a fresh
 * network, so it also gives the warm-up: the first frame from which the
 * judged rays' mean stays within 5% and within 2% of the answer. The tail's
 * gate opens after `warmupSteps` (256) training steps at 16 a frame, so from
 * frame 17; until then the frame is GI-1.2's. Scenes as
 * `radiance-cache-world-cache`'s (black sky, albedo 0.5, 1280 × 720, the
 * screen drawn by primary rays through the platform's tracer):
 *
 * 1. **Lit plane** (a 400 m floor, albedo 0.5, and a black wall, albedo 0,
 *    the sun along (0, -1, -1), intensity 8; multibounce on, 300 frames): the
 *    floor's light is albedo / π × E (E = 8 cos 45°) and nothing else, since
 *    its bounce rays find only the black wall or the black sky; the wall
 *    reflects nothing. So with the tail on, the floor hits' mean is within 2%
 *    of the lit value and 95% of them within 2% each, and the wall hits read
 *    under 1% of it: the network must answer 0 where nothing is lit, and the
 *    direct light must come through as it is.
 * 2. **Furnace, every bounce** (a floor facing up and a ceiling facing down
 *    3 m above, 400 m wide, a sun straight down of irradiance 4 that casts no
 *    shadow; 300 frames): GI-1.2 reads the second bounce ρ² E / π = 0.3183;
 *    with the tail the ceiling settles at every bounce, ρ (ρE/π) / (1 − ρ²) =
 *    0.4244 (`furnaceCeiling`). Judged: ceiling hits whose GI-1.2 tile is
 *    under 2 m (beyond, its tiles hold floor and ceiling hits together; NOTES),
 *    their mean at the last frame: within 2% with 8 px cells, within 6% at
 *    Capsaicin's 32 px. At 32 px the far tiles' columns put the floor's direct
 *    light into ceiling rays at grazing angles, so the floor's screen
 *    irradiance reads about 13% over π × the ceiling, the screen records teach
 *    the network that, and the fixed point settles 4-5% over; at 8 px the
 *    columns are out of reach and the floor's records read ρ × the ceiling.
 *    Also recorded: the tail without screen records (`screenStride: 0`),
 *    which can't learn the floor here (every probe is on it, every probe ray
 *    goes up, so no cell record is a floor's), and the tail's answers and
 *    training targets by surface.
 * 3. **Strips** (the furnace's floor as 5 cm strips every 10 cm: half of a
 *    ceiling bounce ray misses; 300 frames): GI-1.2 0.1592, every bounce
 *    `furnaceCeiling(..., 1/2)` = 0.1819; the tail's near ceiling rays'
 *    mean within 2% at 32 px cells, within 6% at 8 px (the strips' own
 *    screen irradiance reads about 20% low at 8 px: probes start under thin
 *    strips, register D17). Also recorded: GI-1.2 at 8 px for both scenes
 *    (its second bounce doesn't read the floor's irradiance).
 * 4. **Lag and flicker** (the corner, floor and wall albedo 0.5, multibounce;
 *    the sun doubled after 150 frames, 200 frames after): frames until the
 *    mean irradiance E over the screen, and the mean hit-ray radiance, stay
 *    within 5% and 1% of twice their value before the change, and the
 *    flicker of E over the 30 still frames before it (per pixel
 *    |ΔE| / max(E, 5% of the mean), mean and p95; radiance-cache-stability's
 *    measure). GI-1.2, the tail's defaults (no EMA; 16 frames of warm-up
 *    training after the change, which the platform reports as global), the
 *    tail without that boost, and the pathtracer's settings (the warm-up EMA,
 *    no boost). Recorded, and the tail's E must settle within
 *    1% within the 200 frames (some frame count, not a race with GI-1.2's;
 *    1%, since the tail is a few per cent of E: a network that stopped
 *    learning would still settle within 5%).
 * 5. **NaN scan** (the tail's defaults on `cornell-box`, `two-room` and
 *    `sponza` with curtains, from the first bookmark, 100 frames): every probe
 *    ray's radiance, every tail answer and every training target finite and
 *    at least 0 after frames 1, 50 and 100, with the gate open by frame 50.
 * 6. **With DDGI stacked** (the furnace, a DDGI volume under the cache with
 *    `setFallback`, 40 frames): with `secondBounce: 'auto'` the second bounce
 *    comes from DDGI (P6.3, register E10), whose field already carries the
 *    later bounces, so the tail stays off: no training step, the gate shut,
 *    one warning naming `neuralCache`; with `secondBounce: 'gi1-rays'` the
 *    tail runs as without DDGI (its gate opens, no warning).
 *
 * 7. **The f32 fallback** (a device without `shader-f16` and with WebGPU's
 *    default 16 KiB of workgroup memory, as a host gets it without the
 *    platform's advice): the compact network runs in f32 with smaller tiles
 *    and the furnace at 8 px cells still reads every bounce within 2%; the
 *    default 5 × 64 network can't fit, so the tail is off with a warning
 *    naming `maxComputeWorkgroupStorageSize`.
 * 8. **Memory** at 1280 × 720, 1920 × 1080 and 3840 × 2160 (Cornell box, 3
 *    frames; recorded): the tail's buffers, its answers in the grid's lists,
 *    the training set and query capacities.
 * 9. **Training targets on Bistro exterior** (100 frames; recorded): the cell
 *    and screen records' mean targets and the share of screen records with a
 *    zero target (rays that missed: no sky in the targets).
 *
 * No glossy case: the platform's materials are diffuse (register D8), so
 * GI-1.1's reflections, where the pathtracer found the network halves the
 * converged error, can't be checked here yet.
 *
 * Correctness only; GPU time and memory are `radiance-cache-neural-timing`'s.
 * Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "p6.4 neural tail" -- pnpm test:gpu --filter radiance-cache-neural-tail --gpu nvidia`.
 */

import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat, unpackHalf3 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { furnaceCeiling } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/neural-cache/neural-tail-cpu.ts';
import { NEURAL_TAIL_COUNT_WORD, NEURAL_TAIL_RECORD_KIND, NEURAL_TAIL_RECORD_VEC4 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/neural-cache/neural-tail-wgsl.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { hashGridCacheCellSizeConstant, hashGridCacheGetCellSize, verticalFieldOfView } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import { NEURAL_TAIL_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { describeScene, loadManifest, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { readTexture } from '../radiance-cache-probes/readback.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const PAGE_ENV = import.meta.env as Record<string, unknown>;
/** T2.17: scale model textures down to this side at load (Bistro on a 6 GB GPU); undefined keeps them. */
const MAX_TEXTURE_SIZE = typeof PAGE_ENV.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' ? Number(PAGE_ENV.VITE_DDGI_MAX_TEXTURE_SIZE) : undefined;
const WIDTH = 1280;
const HEIGHT = 720;
const TOLERANCE = 0.02;
/**
 * The every-bounce checks take the floor's light from the screen's irradiance (screen records),
 * so they inherit its errors, each judged within 2% where the floor's irradiance is clean and
 * within this where it isn't (P6.4): the furnace at Capsaicin's 32 px cells reads 4-5% over
 * (far ceiling tiles are columns that also hold floor hits, NOTES), clean at 8 px; the strips at
 * 8 px read 4-5% under (the strips' screen irradiance reads about 20% under ρ × the ceiling
 * there; probes start under thin strips, register D17), clean at 32 px.
 */
const COLUMN_TOLERANCE = 0.06;
const ALBEDO = 0.5;
/** Frames of each analytic check: the gate opens at frame 17, the fixed point needs a few hundred more. */
const CHECK_FRAMES = Number((import.meta.env as Record<string, unknown>).VITE_NEURAL_TAIL_FRAMES ?? 300);

const CORNER_CAMERA: PageCamera = { position: [0, 2, 6], target: [0, 1, 0], fovYDeg: 60, near: 0.05, far: 1000 };
const SUN_INTENSITY = 8;
const SUN_TRAVEL: Vec3 = [0, -1, -1];
/** albedo / π × E, E = 8 cos 45° on the floor. */
const LIT_RADIANCE = (ALBEDO / Math.PI) * SUN_INTENSITY * Math.SQRT1_2;

const FURNACE_CAMERA: PageCamera = { position: [0, 2, 0], target: [0, 0, -2], fovYDeg: 60, near: 0.05, far: 1000 };
const CEILING_Y = 3;
const FURNACE_SUN_IRRADIANCE = 4;
const FURNACE_SUN: LightDescriptor = { type: 'directional', direction: [0, -1, 0], color: [1, 1, 1], intensity: FURNACE_SUN_IRRADIANCE, castsShadows: false, label: 'furnace sun' };
const SECOND_BOUNCE = furnaceCeiling(ALBEDO, FURNACE_SUN_IRRADIANCE, 'second-bounce');
const EVERY_BOUNCE = furnaceCeiling(ALBEDO, FURNACE_SUN_IRRADIANCE, 'every-bounce');
const STRIPS = { width: 0.05, period: 0.1, halfLength: 100, fill: 0.5 } as const;
const STRIPS_SECOND_BOUNCE = furnaceCeiling(ALBEDO, FURNACE_SUN_IRRADIANCE, 'second-bounce', STRIPS.fill);
const STRIPS_EVERY_BOUNCE = furnaceCeiling(ALBEDO, FURNACE_SUN_IRRADIANCE, 'every-bounce', STRIPS.fill);

/** The GI-1.2 grid's cell-size constant at Capsaicin's defaults (32 px, min 0.1 m, tiles of 8 cells), for the judged-ray filters. */
const GI1_CELL_CONSTANTS = { cellSize: hashGridCacheCellSizeConstant(verticalFieldOfView(frameCamera(CORNER_CAMERA, WIDTH, HEIGHT).projection), WIDTH, HEIGHT, 32), minCellSize: 0.1 };
const GI1_TILE_CELLS = 8;

const GI1: Partial<RadianceCacheSettings> = { useMultibounce: true, neuralCache: 'off' };
const TAIL: Partial<RadianceCacheSettings> = { useMultibounce: true, neuralCache: 'tail' };
const TAIL_NO_SCREEN: Partial<RadianceCacheSettings> = { useMultibounce: true, neuralCache: 'tail', neural: { screenStride: 0 } };

type SceneKind = 'litPlane' | 'corner' | 'furnace' | 'strips';
type Surface = 'floor' | 'wall' | 'ceiling' | 'other';

interface ProbeRay {
    readonly direction: Vec3;
    readonly hitT: number;
    readonly hit: Vec3;
    readonly surface: Surface;
}

// ---- Scenes ---------------------------------------------------------------------------------------

function quad(platform: GIPlatformImpl, label: string, corners: readonly Vec3[], normal: Vec3, albedo = ALBEDO): void {
    const scene = platform.scene;
    const mesh = scene.createMesh({ positions: new Float32Array(corners.flat()), normals: new Float32Array([...normal, ...normal, ...normal, ...normal]), indices: new Uint32Array([0, 1, 2, 0, 2, 3]), label });
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
    const material = scene.createMaterial({ albedo: [ALBEDO, ALBEDO, ALBEDO], label: 'strip floor' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'strip floor' });
}

function describe(platform: GIPlatformImpl, kind: SceneKind): PageCamera {
    const half = 200;
    if (kind === 'furnace' || kind === 'strips') {
        if (kind === 'furnace') quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0]);
        else stripFloor(platform);
        quad(platform, 'ceiling', [[-half, CEILING_Y, -half], [half, CEILING_Y, -half], [half, CEILING_Y, half], [-half, CEILING_Y, half]], [0, -1, 0]);
        platform.scene.createLight(FURNACE_SUN);
        platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
        return FURNACE_CAMERA;
    }
    quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0]);
    quad(platform, 'wall', [[-20, 0, 0], [20, 0, 0], [20, 10, 0], [-20, 10, 0]], [0, 0, 1], kind === 'litPlane' ? 0 : ALBEDO);
    platform.scene.createLight({ type: 'directional', direction: SUN_TRAVEL, color: [1, 1, 1], intensity: SUN_INTENSITY, label: 'sun' });
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    return CORNER_CAMERA;
}

function classify(hit: Vec3, kind: SceneKind): Surface {
    if (kind === 'furnace' || kind === 'strips') return Math.abs(hit[1] - CEILING_Y) < 1e-3 ? 'ceiling' : Math.abs(hit[1]) < 1e-3 ? 'floor' : 'other';
    if (Math.abs(hit[1]) < 1e-3) return 'floor';
    if (Math.abs(hit[2]) < 1e-3) return 'wall';
    return 'other';
}

// ---- Running a scenario ---------------------------------------------------------------------------

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const readback = device.createBuffer({ label: 'neural tail page readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'neural tail page readback' });
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
        rays.push({ direction, hitT, hit, surface: hitT >= 0 ? classify(hit, kind) : 'other' });
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
type Judge = (ray: ProbeRay) => number | null;

interface Run {
    /** Per frame: the judged rays' mean relative error (signed; against the lit value where the expectation is 0) and the share within 2%. */
    readonly meanError: number[];
    readonly withinShare: number[];
    readonly judged: number;
    readonly tailSteps: number;
    readonly tailGateOpen: boolean;
    /** The tail's state after the last frame, by surface: its answers' mean (the second hits') and its training targets' mean. */
    readonly tailBySurface: Record<string, TailDiagnostics> | null;
}

/** The tail's answers and training records at one surface (or the whole scene). */
interface TailDiagnostics {
    answers: number;
    meanAnswer: number;
    cellRecords: number;
    meanCellTarget: number;
    screenRecords: number;
    meanScreenTarget: number;
    /** Screen records whose target is 0 (their ray missed, or found black): the sky is no longer in them. */
    screenZeroTargets: number;
}

/** The tail's last answers and training records (by kind), grouped by `place` (the surface their hit is on). */
async function tailDiagnostics(device: GPUDevice, technique: RadianceCacheTechnique, place: (position: Vec3) => string): Promise<Record<string, TailDiagnostics> | null> {
    const tail = technique.neuralTail;
    const cache = technique.worldCache;
    const lists = cache?.rayLists;
    const buffers = tail?.buffers;
    if (!tail || !cache || !lists || !buffers?.queries || !tail.active) return null;
    const sums: Record<string, { answers: number; answerSum: number; cell: number; cellSum: number; screen: number; screenSum: number; zero: number }> = {};
    const group = (key: string) => (sums[key] ??= { answers: 0, answerSum: 0, cell: 0, cellSum: 0, screen: 0, screenSum: 0, zero: 0 });
    const secondBounce = Math.min(new Uint32Array(await readRange(device, cache.buffers.tileWords, HASH_GRID_COUNTER.visibilityCount1 * 4, 4))[0], buffers.queryCapacity);
    if (secondBounce > 0) {
        const answers = new Uint32Array(await readRange(device, lists.buffer, lists.layout.neuralTailBuffer * 4, secondBounce * NEURAL_TAIL_WORDS * 4));
        const queries = new Float32Array(await readRange(device, buffers.queries, 0, secondBounce * 3 * 16));
        for (let entry = 0; entry < secondBounce; entry++) {
            if (queries[entry * 12 + 3] === 0) continue;
            const value = (halfBitsToFloat(answers[entry * 2] & 0xffff) + halfBitsToFloat(answers[entry * 2] >>> 16) + halfBitsToFloat(answers[entry * 2 + 1] & 0xffff)) / 3;
            const entryGroup = group(place([queries[entry * 12], queries[entry * 12 + 1], queries[entry * 12 + 2]]));
            entryGroup.answers++;
            entryGroup.answerSum += value;
        }
    }
    const records = Math.min(new Uint32Array(await readRange(device, buffers.counts, NEURAL_TAIL_COUNT_WORD.records * 4, 4))[0], buffers.setCapacity);
    const set = new Float32Array(await readRange(device, buffers.set, 0, records * NEURAL_TAIL_RECORD_VEC4 * 16));
    for (let record = 0; record < records; record++) {
        const base = record * NEURAL_TAIL_RECORD_VEC4 * 4;
        const recordGroup = group(place([set[base], set[base + 1], set[base + 2]]));
        const target = (set[base + 12] + set[base + 13] + set[base + 14]) / 3;
        if (set[base + 3] === NEURAL_TAIL_RECORD_KIND.screen) {
            recordGroup.screen++;
            recordGroup.screenSum += target;
            if (target === 0) recordGroup.zero++;
        } else {
            recordGroup.cell++;
            recordGroup.cellSum += target;
        }
    }
    const mean = (sum: number, count: number): number => Number((sum / Math.max(count, 1)).toPrecision(5));
    return Object.fromEntries(
        Object.entries(sums).map(([key, g]) => [
            key,
            { answers: g.answers, meanAnswer: mean(g.answerSum, g.answers), cellRecords: g.cell, meanCellTarget: mean(g.cellSum, g.cell), screenRecords: g.screen, meanScreenTarget: mean(g.screenSum, g.screen), screenZeroTargets: g.zero },
        ]),
    );
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

function rayValue(state: FrameState, index: number): number {
    return (state.radiance[index * 4] + state.radiance[index * 4 + 1] + state.radiance[index * 4 + 2]) / 3;
}

/** One scenario, each frame judged by every judge (one `Run` each). */
async function runScenario(device: GPUDevice, kind: SceneKind, settings: Partial<RadianceCacheSettings>, frames: number, judges: readonly Judge[]): Promise<Run[]> {
    const platform = new GIPlatformImpl(device, { label: `neural tail page ${kind}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    const camera = describe(platform, kind);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `neural tail page ${kind}` });
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
        const meanError = judges.map((): number[] => []);
        const withinShare = judges.map((): number[] => []);
        const judged = judges.map(() => 0);
        for (let index = 0; index < frames; index++) {
            frame();
            await device.queue.onSubmittedWorkDone();
            const state = await readFrame(device, technique, kind);
            judges.forEach((judge, judgeIndex) => {
                let sum = 0;
                let expectedSum = 0;
                let within = 0;
                let count = 0;
                state.rays.forEach((ray, rayIndex) => {
                    const expected = judge(ray);
                    if (expected === null) return;
                    count++;
                    const value = rayValue(state, rayIndex);
                    sum += value;
                    expectedSum += expected;
                    if (Math.abs(value - expected) <= TOLERANCE * Math.max(expected, LIT_RADIANCE * 0.05)) within++;
                });
                judged[judgeIndex] = count;
                meanError[judgeIndex].push(expectedSum > 0 ? (sum - expectedSum) / expectedSum : sum / Math.max(count, 1) / LIT_RADIANCE);
                withinShare[judgeIndex].push(within / Math.max(count, 1));
            });
        }
        const tail = technique.neuralTail;
        const bySurface = await tailDiagnostics(device, technique, (position) => classify(position, kind));
        return judges.map((_, judgeIndex) => ({ meanError: meanError[judgeIndex], withinShare: withinShare[judgeIndex], judged: judged[judgeIndex], tailSteps: tail?.steps ?? 0, tailGateOpen: tail?.active ?? false, tailBySurface: bySurface }));
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

function summary(run: Run): Record<string, unknown> {
    return {
        judgedRays: run.judged,
        finalMeanError: Number(run.meanError[run.meanError.length - 1].toFixed(5)),
        finalWithinShare: Number(run.withinShare[run.withinShare.length - 1].toFixed(4)),
        framesTo5: warmFrame(run.meanError.map((error) => Math.abs(error) <= 0.05)),
        framesTo2: warmFrame(run.meanError.map((error) => Math.abs(error) <= TOLERANCE)),
        raysWarmFrames: warmFrame(run.withinShare.map((share) => share >= 0.95)),
        tailSteps: run.tailSteps,
        tailGateOpen: run.tailGateOpen,
        meanErrorAtFrames: Object.fromEntries([25, 50, 100, 150, 200, 250, 300].filter((frame) => frame <= run.meanError.length).map((frame) => [frame, Number(run.meanError[frame - 1].toFixed(4))])),
        tailBySurface: run.tailBySurface,
    };
}

const final = (run: Run): number => run.meanError[run.meanError.length - 1];
const finalShare = (run: Run): number => run.withinShare[run.withinShare.length - 1];

// ---- Judges ----------------------------------------------------------------------------------------

function gi1TileSize(hit: Vec3, camera: PageCamera): number {
    return hashGridCacheGetCellSize(hit, camera.position, GI1_CELL_CONSTANTS) * GI1_TILE_CELLS;
}

/** The furnace's ceiling rays whose GI-1.2 tile is under 2 m. */
function ceilingJudge(expected: number): Judge {
    return (ray) => (ray.hitT >= 0 && ray.surface === 'ceiling' && ray.direction[1] > 0 && gi1TileSize(ray.hit, FURNACE_CAMERA) < 2 ? expected : null);
}

/** Section 1: floor hits under 2 m tiles read the lit value (a far tile can hold wall hits too). */
const litFloorJudge: Judge = (ray) => (ray.hitT >= 0 && ray.surface === 'floor' && gi1TileSize(ray.hit, CORNER_CAMERA) < 2 ? LIT_RADIANCE : null);
/** Section 1: wall hits above 3 m (their tiles, under 3 m across there, can't reach the floor) read 0. */
const blackWallJudge: Judge = (ray) => (ray.hitT >= 0 && ray.surface === 'wall' && ray.hit[1] > 3 && gi1TileSize(ray.hit, CORNER_CAMERA) < 3 ? 0 : null);

// ---- Section 4: lag and flicker ---------------------------------------------------------------------

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
    /** E's mean before the change, over the screen. */
    readonly irradianceBefore: number;
}

/** The first frame after the change (1-based) from which every later frame is within `tolerance` of the target. */
function settledAfter(values: readonly number[], target: number, tolerance: number): number | null {
    return warmFrame(values.map((value) => Math.abs(value / target - 1) <= tolerance));
}

async function runLag(device: GPUDevice, settings: Partial<RadianceCacheSettings>): Promise<LagResult> {
    const BEFORE = 150;
    const AFTER = 200;
    const STILL = 30;
    const platform = new GIPlatformImpl(device, { label: 'neural tail page lag' });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    const camera = describe(platform, 'corner');
    const sun = platform.scene.createLight({ type: 'directional', direction: SUN_TRAVEL, color: [1, 1, 1], intensity: SUN_INTENSITY, label: 'lag sun' });
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: 'neural tail page lag' });
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
                sum += rayValue(state, index);
                count++;
            });
            return sum / Math.max(count, 1);
        };
        // Relative changes pooled over pixels and frames as a histogram (steps of 0.0005 up to 5), as
        // radiance-cache-stability tallies them: millions of values, so no array to sort.
        const bins = new Uint32Array(10001);
        let changeSum = 0;
        let changeCount = 0;
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
                    const change = Math.abs(e[pixel] - previous[pixel]) / Math.max(previous[pixel], floor);
                    if (!Number.isFinite(change)) continue;
                    changeSum += change;
                    changeCount++;
                    bins[Math.min(Math.floor(change / 0.0005), bins.length - 1)]++;
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
        let p95 = NaN;
        for (let bin = 0, running = 0; bin < bins.length; bin++) {
            running += bins[bin];
            if (running >= 0.95 * changeCount) {
                p95 = (bin + 1) * 0.0005;
                break;
            }
        }
        return {
            flickerMean: Number((changeSum / Math.max(changeCount, 1)).toFixed(5)),
            flickerP95: Number(p95.toFixed(4)),
            irradianceFramesTo5: settledAfter(eAfter, 2 * beforeE, 0.05),
            irradianceFramesTo1: settledAfter(eAfter, 2 * beforeE, 0.01),
            rayFramesTo5: settledAfter(rayAfter, 2 * beforeRay, 0.05),
            rayFramesTo1: settledAfter(rayAfter, 2 * beforeRay, 0.01),
            finalIrradianceRatio: Number((eAfter[eAfter.length - 1] / (2 * beforeE)).toFixed(4)),
            irradianceBefore: Number(beforeE.toPrecision(5)),
        };
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- Section 5: the NaN scan ---------------------------------------------------------------------

async function describeLoaded(platform: GIPlatformImpl, device: GPUDevice, manifest: SceneManifest): Promise<() => void> {
    const loaded = await loadScene(manifest, () => {}, MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE });
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
    readonly tailAnswers: number;
    readonly tailAnswersBad: number;
    readonly meanTailAnswer: number;
    readonly trainingRecords: number;
    readonly trainingTargetsBad: number;
    readonly gateOpen: boolean;
    readonly meanRayRadiance: number;
}

const isGood = (value: number): boolean => Number.isFinite(value) && value >= 0;

async function scanScene(device: GPUDevice, id: string, loader: 'manifest' | 'viewer'): Promise<Scan[]> {
    const manifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `neural tail page scan ${id}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(TAIL));
    platform.setOutput(technique);
    await platform.prepare();
    let beforeFrame = (): void => {};
    if (loader === 'viewer') beforeFrame = await describeLoaded(platform, device, manifest);
    else describeScene(platform, manifest);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
    const frame = (): void => {
        beforeFrame();
        const encoder = device.createCommandEncoder({ label: `neural tail page scan ${id}` });
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
        for (let index = 1; index <= 100; index++) {
            frame();
            if (index % 8 === 0) await device.queue.onSubmittedWorkDone();
            if (index !== 1 && index !== 50 && index !== 100) continue;
            await device.queue.onSubmittedWorkDone();
            const probes = technique.probes;
            const cache = technique.worldCache;
            const tail = technique.neuralTail;
            const tailBuffers = tail?.buffers;
            const lists = cache?.rayLists;
            if (!probes || !cache || !tail || !tailBuffers || !lists) throw new Error(`${id}: no probes, cache or neural tail`);
            const count = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
            const packed = new Uint32Array(await readRange(device, probes.rayScratch.rayResults.buffer, probes.rayScratch.rayResults.offset ?? 0, count * 8));
            let rayValuesBad = 0;
            let raySum = 0;
            for (let ray = 0; ray < count; ray++) {
                const rgb = [halfBitsToFloat(packed[ray * 2] & 0xffff), halfBitsToFloat(packed[ray * 2] >>> 16), halfBitsToFloat(packed[ray * 2 + 1] & 0xffff)];
                if (!rgb.every(isGood)) rayValuesBad++;
                else raySum += (rgb[0] + rgb[1] + rgb[2]) / 3;
            }
            // The tail's answers this frame (the second-bounce entries'), written once the gate is open.
            const secondBounce = new Uint32Array(await readRange(device, cache.buffers.tileWords, HASH_GRID_COUNTER.visibilityCount1 * 4, 4))[0];
            let tailAnswersBad = 0;
            let tailAnswers = 0;
            let tailSum = 0;
            if (tail.active && secondBounce > 0) {
                const answers = new Uint32Array(await readRange(device, lists.buffer, lists.layout.neuralTailBuffer * 4, secondBounce * NEURAL_TAIL_WORDS * 4));
                for (let entry = 0; entry < secondBounce; entry++) {
                    const rgbw = [halfBitsToFloat(answers[entry * 2] & 0xffff), halfBitsToFloat(answers[entry * 2] >>> 16), halfBitsToFloat(answers[entry * 2 + 1] & 0xffff), halfBitsToFloat(answers[entry * 2 + 1] >>> 16)];
                    if (rgbw[3] === 0) continue;
                    tailAnswers++;
                    if (!rgbw.slice(0, 3).every(isGood)) tailAnswersBad++;
                    else tailSum += (rgbw[0] + rgbw[1] + rgbw[2]) / 3;
                }
            }
            const records = Math.min(new Uint32Array(await readRange(device, tailBuffers.counts, NEURAL_TAIL_COUNT_WORD.records * 4, 4))[0], tailBuffers.setCapacity);
            const set = new Float32Array(await readRange(device, tailBuffers.set, 0, records * NEURAL_TAIL_RECORD_VEC4 * 16));
            let trainingTargetsBad = 0;
            for (let record = 0; record < records; record++) {
                const base = (record * NEURAL_TAIL_RECORD_VEC4 + 3) * 4;
                if (![set[base], set[base + 1], set[base + 2]].every(isGood)) trainingTargetsBad++;
            }
            scans.push({
                frame: index,
                probeRays: count,
                rayValuesBad,
                tailAnswers,
                tailAnswersBad,
                meanTailAnswer: Number((tailSum / Math.max(tailAnswers, 1)).toPrecision(5)),
                trainingRecords: records,
                trainingTargetsBad,
                gateOpen: tail.active,
                meanRayRadiance: Number((raySum / Math.max(count, 1)).toPrecision(5)),
            });
        }
        return scans;
    } finally {
        screen.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- Section 6: with DDGI stacked ------------------------------------------------------------------

/** The furnace with a DDGI volume stacked under the cache (P6.3's arrangement), `frames` frames: what the tail did and what it said. */
async function runWithDdgi(device: GPUDevice, settings: Partial<RadianceCacheSettings>, frames: number): Promise<{ inUse: string | null; steps: number; gateOpen: boolean; warnings: number }> {
    const warnings: string[] = [];
    const consoleWarn = console.warn;
    console.warn = (...args: unknown[]) => {
        warnings.push(args.map(String).join(' '));
        consoleWarn(...args);
    };
    const platform = new GIPlatformImpl(device, { label: 'neural tail page ddgi' });
    try {
        const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
        platform.setOutput(technique);
        const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: [0, 1.5, -4], size: [24, 2, 24], spacing: 1 } }));
        platform.setFallback(technique, ddgi);
        await platform.prepare();
        const camera = describe(platform, 'furnace');
        const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
        try {
            const frame = (): void => {
                const encoder = device.createCommandEncoder({ label: 'neural tail page ddgi' });
                platform.encode(encoder, { deltaTime: 1 / 60, budget: { units: 2048 }, camera: frameCamera(camera, WIDTH, HEIGHT) });
                screen.encode(encoder, camera);
                platform.encodeScreen(encoder, screen.screen);
                device.queue.submit([encoder.finish()]);
            };
            frame();
            await platform.rays.whenBuilt();
            for (let index = 0; index < frames; index++) {
                frame();
                if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
            }
            await device.queue.onSubmittedWorkDone();
            const tail = technique.neuralTail;
            return { inUse: technique.worldCache?.secondBounceInUse ?? null, steps: tail?.steps ?? -1, gateOpen: tail?.active ?? false, warnings: warnings.filter((message) => message.includes('neuralCache')).length };
        } finally {
            screen.dispose();
        }
    } finally {
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
        console.warn = consoleWarn;
    }
}

// ---- Section 7: the f32 fallback on a default-limits device ----------------------------------------

/** A device as a host gets it without the platform's advice on workgroup memory and f16: WebGPU's 16 KiB, no shader-f16. */
async function defaultLimitsDevice(): Promise<GPUDevice> {
    // A fresh adapter: one adapter makes one device.
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('f32 fallback: no second adapter');
    return adapter.requestDevice({
        requiredLimits: { ...requiredLimits(adapter), maxComputeWorkgroupStorageSize: 16384 },
        requiredFeatures: optionalFeatures(adapter).filter((feature) => feature !== 'shader-f16'),
    });
}

/** Prepares a technique on `device` with `settings` and reports whether it made a tail, its precision, and the warnings that named `neuralCache`. */
async function prepareOnly(device: GPUDevice, settings: Partial<RadianceCacheSettings>): Promise<{ tail: boolean; precision: string | null; warnings: string[] }> {
    const warnings: string[] = [];
    const consoleWarn = console.warn;
    console.warn = (...args: unknown[]) => {
        warnings.push(args.map(String).join(' '));
        consoleWarn(...args);
    };
    const platform = new GIPlatformImpl(device, { label: 'neural tail page prepare' });
    try {
        const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
        platform.setOutput(technique);
        await platform.prepare();
        return { tail: technique.neuralTail !== null, precision: technique.neuralTail?.network?.precision ?? null, warnings: warnings.filter((message) => message.includes('neuralCache')) };
    } finally {
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
        console.warn = consoleWarn;
    }
}

// ---- Section 8: memory by screen size ---------------------------------------------------------------

async function memoryAt(device: GPUDevice, width: number, height: number): Promise<Record<string, number>> {
    const manifest = loadManifest('cornell-box');
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 1000 };
    const platform = new GIPlatformImpl(device, { label: `neural tail page memory ${width}x${height}` });
    try {
        const technique = platform.addTechnique(new RadianceCacheTechnique(TAIL));
        platform.setOutput(technique);
        await platform.prepare();
        describeScene(platform, manifest);
        const screen = await PageScreen.create(platform, device, width, height);
        try {
            for (let index = 0; index < 3; index++) {
                const encoder = device.createCommandEncoder({ label: 'neural tail page memory' });
                platform.encode(encoder, { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, width, height) });
                screen.encode(encoder, camera);
                platform.encodeScreen(encoder, screen.screen);
                device.queue.submit([encoder.finish()]);
                await device.queue.onSubmittedWorkDone();
            }
            const tail = technique.neuralTail;
            const lists = technique.worldCache?.rayLists;
            const mib = (bytes: number): number => Number((bytes / 2 ** 20).toFixed(2));
            // The tail's answers live in the grid's per-ray lists, last (2 words a second-bounce entry).
            const answerBytes = lists && lists.layout.neuralTailBuffer > 0 ? lists.layout.bytes - lists.layout.neuralTailBuffer * 4 : 0;
            return {
                tailMiB: mib(tail?.memoryBytes ?? 0),
                networkMiB: mib(tail?.network?.bufferBytes ?? 0),
                answersInTheGridListsMiB: mib(answerBytes),
                tailWithAnswersMiB: mib((tail?.memoryBytes ?? 0) + answerBytes),
                queryCapacity: tail?.buffers?.queryCapacity ?? 0,
                setCapacity: tail?.buffers?.setCapacity ?? 0,
                gridListsMiB: mib(lists?.layout.bytes ?? 0),
                techniqueMiB: mib(technique.memoryBytes),
            };
        } finally {
            screen.dispose();
        }
    } finally {
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- Section 9: the training targets on a real scene ------------------------------------------------

async function targetsOn(device: GPUDevice, id: string): Promise<{ frames: number; diagnostics: Record<string, TailDiagnostics> | null }> {
    const manifest = loadManifest(id);
    const bookmark = manifest.cameras[0];
    const camera: PageCamera = { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far: 2000 };
    const platform = new GIPlatformImpl(device, { label: `neural tail page targets ${id}` });
    try {
        const technique = platform.addTechnique(new RadianceCacheTechnique(TAIL));
        platform.setOutput(technique);
        await platform.prepare();
        const beforeFrame = await describeLoaded(platform, device, manifest);
        const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
        try {
            const frames = 100;
            for (let index = 0; index < frames; index++) {
                beforeFrame();
                const encoder = device.createCommandEncoder({ label: `neural tail page targets ${id}` });
                platform.encode(encoder, { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) });
                screen.encode(encoder, camera);
                platform.encodeScreen(encoder, screen.screen);
                device.queue.submit([encoder.finish()]);
                if (index === 0) await platform.rays.whenBuilt();
                if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
            }
            await device.queue.onSubmittedWorkDone();
            return { frames, diagnostics: await tailDiagnostics(device, technique, () => 'scene') };
        } finally {
            screen.dispose();
        }
    } finally {
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
    const data: Record<string, unknown> = { glossy: 'no glossy case: the platform has diffuse materials only (register D8)' };
    const env = import.meta.env as Record<string, unknown>;
    /** `VITE_NEURAL_TAIL_SECTIONS`: a comma-separated subset of the section numbers (all by default), for mutation runs. */
    const sections = typeof env.VITE_NEURAL_TAIL_SECTIONS === 'string' ? new Set(env.VITE_NEURAL_TAIL_SECTIONS.split(',').map((section) => Number(section.trim()))) : null;
    const runs = (section: number): boolean => sections === null || sections.has(section);

    // 1. Lit plane with a black wall: the direct light as it is, nothing added.
    if (runs(1)) {
        const results: Record<string, unknown> = { expected: LIT_RADIANCE };
        for (const [name, settings] of [['gi1', GI1], ['tail', TAIL]] as const) {
            const [floor, wall] = await runScenario(device, 'litPlane', settings, CHECK_FRAMES, [litFloorJudge, blackWallJudge]);
            results[name] = { floor: summary(floor), wallMeanOverLit: Number(final(wall).toFixed(5)), wallRays: wall.judged };
            if (name !== 'tail') continue;
            if (Math.abs(final(floor)) > TOLERANCE) fail(`lit plane: the tail's floor rays average ${(final(floor) * 100).toFixed(2)}% off ${LIT_RADIANCE.toFixed(4)}`);
            if (finalShare(floor) < 0.95) fail(`lit plane: only ${(finalShare(floor) * 100).toFixed(1)}% of the tail's floor rays within 2%`);
            if (!(final(wall) < 0.01)) fail(`lit plane: the tail's black-wall rays read ${(final(wall) * 100).toFixed(2)}% of the lit value (expected 0)`);
            if (floor.judged < 1000 || wall.judged < 1000) fail(`lit plane: too few judged rays (${floor.judged} floor, ${wall.judged} wall)`);
            if (!floor.tailGateOpen) fail('lit plane: the tail\'s gate never opened');
        }
        data.litPlane = results;
    }

    // 2. Furnace: GI-1.2's two bounces against the tail's every bounce.
    if (runs(2)) {
        const [gi1] = await runScenario(device, 'furnace', GI1, CHECK_FRAMES, [ceilingJudge(SECOND_BOUNCE)]);
        const [tail] = await runScenario(device, 'furnace', TAIL, CHECK_FRAMES, [ceilingJudge(EVERY_BOUNCE)]);
        const [noScreen] = await runScenario(device, 'furnace', TAIL_NO_SCREEN, CHECK_FRAMES, [ceilingJudge(EVERY_BOUNCE)]);
        const [fineCells] = await runScenario(device, 'furnace', { ...TAIL, hashGridCacheCellSize: 8 }, CHECK_FRAMES, [ceilingJudge(EVERY_BOUNCE)]);
        const [gi1FineCells] = await runScenario(device, 'furnace', { ...GI1, hashGridCacheCellSize: 8 }, 60, [ceilingJudge(SECOND_BOUNCE)]);
        data.furnace = {
            secondBounce: SECOND_BOUNCE,
            everyBounce: EVERY_BOUNCE,
            gi1: { ...summary(gi1), meanErrorAgainstEveryBounce: Number(((1 + final(gi1)) * (SECOND_BOUNCE / EVERY_BOUNCE) - 1).toFixed(5)) },
            tail: summary(tail),
            tailWithoutScreenRecords: summary(noScreen),
            tailWithCellsOf8Pixels: summary(fineCells),
            gi1WithCellsOf8Pixels: summary(gi1FineCells),
        };
        if (Math.abs(final(fineCells)) > TOLERANCE) fail(`furnace, every bounce (8 px cells): the tail's ceiling rays average ${(final(fineCells) * 100).toFixed(2)}% off ${EVERY_BOUNCE.toFixed(4)} after ${CHECK_FRAMES} frames`);
        if (Math.abs(final(tail)) > COLUMN_TOLERANCE) fail(`furnace, every bounce (Capsaicin's 32 px cells): the tail's ceiling rays average ${(final(tail) * 100).toFixed(2)}% off ${EVERY_BOUNCE.toFixed(4)} after ${CHECK_FRAMES} frames (over ${COLUMN_TOLERANCE * 100}%)`);
        if (tail.judged < 1000) fail(`furnace: only ${tail.judged} judged ceiling rays`);
    }

    // 3. Strips: half the bounce rays miss.
    if (runs(3)) {
        const [gi1] = await runScenario(device, 'strips', GI1, CHECK_FRAMES, [ceilingJudge(STRIPS_SECOND_BOUNCE)]);
        const [tail] = await runScenario(device, 'strips', TAIL, CHECK_FRAMES, [ceilingJudge(STRIPS_EVERY_BOUNCE)]);
        const [fineCells] = await runScenario(device, 'strips', { ...TAIL, hashGridCacheCellSize: 8 }, CHECK_FRAMES, [ceilingJudge(STRIPS_EVERY_BOUNCE)]);
        const [gi1FineCells] = await runScenario(device, 'strips', { ...GI1, hashGridCacheCellSize: 8 }, 60, [ceilingJudge(STRIPS_SECOND_BOUNCE)]);
        data.strips = { secondBounce: STRIPS_SECOND_BOUNCE, everyBounce: STRIPS_EVERY_BOUNCE, gi1: summary(gi1), tail: summary(tail), tailWithCellsOf8Pixels: summary(fineCells), gi1WithCellsOf8Pixels: summary(gi1FineCells) };
        if (Math.abs(final(tail)) > TOLERANCE) fail(`strips, every bounce (32 px cells): the tail's ceiling rays average ${(final(tail) * 100).toFixed(2)}% off ${STRIPS_EVERY_BOUNCE.toFixed(4)}`);
        if (Math.abs(final(fineCells)) > COLUMN_TOLERANCE) fail(`strips, every bounce (8 px cells): the tail's ceiling rays average ${(final(fineCells) * 100).toFixed(2)}% off ${STRIPS_EVERY_BOUNCE.toFixed(4)} (over ${COLUMN_TOLERANCE * 100}%)`);
    }

    // 4. Lag and flicker.
    if (runs(4)) {
        const configurations: readonly (readonly [string, Partial<RadianceCacheSettings>])[] = [
            ['gi1 multibounce', GI1],
            ['tail', TAIL],
            ['tail, no change boost', { ...TAIL, neural: { changeBoostFrames: 0 } }],
            ['tail, warm-up EMA, no boost (the pathtracer defaults)', { ...TAIL, neural: { ema: 'warmup', changeBoostFrames: 0 } }],
        ];
        const lag: Record<string, LagResult> = {};
        for (const [name, settings] of configurations) lag[name] = await runLag(device, settings);
        data.lag = lag;
        // The tail's share of E is a few per cent, so a network that stopped learning would still settle within 5%: judged at 1%.
        if (lag.tail.irradianceFramesTo1 === null) fail(`lag: the tail's E never settles within 1% of twice its value after the sun doubles (last frame ${lag.tail.finalIrradianceRatio} of it)`);
    }

    // 5. NaN scan.
    if (runs(5)) {
        const scans: Record<string, Scan[]> = {};
        for (const [id, loader] of [['cornell-box', 'manifest'], ['two-room', 'manifest'], ['sponza', 'viewer']] as const) {
            scans[id] = await scanScene(device, id, loader);
            for (const scan of scans[id]) {
                if (scan.rayValuesBad + scan.tailAnswersBad + scan.trainingTargetsBad > 0) {
                    fail(`${id} after frame ${scan.frame}: ${scan.rayValuesBad} ray values, ${scan.tailAnswersBad} tail answers and ${scan.trainingTargetsBad} training targets are NaN, infinite or negative`);
                }
                if (scan.probeRays === 0 || scan.trainingRecords === 0) fail(`${id} after frame ${scan.frame}: ${scan.probeRays} probe rays and ${scan.trainingRecords} training records; nothing to scan`);
                if (scan.frame >= 50 && (!scan.gateOpen || scan.tailAnswers === 0)) fail(`${id} after frame ${scan.frame}: the gate is ${scan.gateOpen ? 'open' : 'closed'} with ${scan.tailAnswers} tail answers`);
            }
        }
        data.nanScan = scans;
    }

    // 6. With DDGI stacked: the tail only extends GI-1.2's own second-bounce rays.
    if (runs(6)) {
        const auto = await runWithDdgi(device, { ...TAIL, secondBounce: 'auto' }, 40);
        const rays = await runWithDdgi(device, { ...TAIL, secondBounce: 'gi1-rays' }, 40);
        data.withDdgi = { auto, gi1Rays: rays };
        if (auto.inUse !== 'fallback' || auto.steps !== 0 || auto.gateOpen || auto.warnings !== 1) {
            fail(`with DDGI stacked, 'auto': the second bounce from ${auto.inUse}, ${auto.steps} training steps, gate ${auto.gateOpen ? 'open' : 'shut'}, ${auto.warnings} warnings (expected the fallback, 0, shut, 1)`);
        }
        if (rays.inUse !== 'gi1-rays' || !rays.gateOpen || rays.warnings !== 0) {
            fail(`with DDGI stacked, 'gi1-rays': the second bounce from ${rays.inUse}, gate ${rays.gateOpen ? 'open' : 'shut'}, ${rays.warnings} warnings (expected gi1-rays, open, 0)`);
        }
    }

    // 7. The f32 fallback on a default-limits device (16 KiB of workgroup memory, no shader-f16).
    if (runs(7)) {
        const small = await defaultLimitsDevice();
        try {
            const [fallback] = await runScenario(small, 'furnace', { ...TAIL, hashGridCacheCellSize: 8 }, CHECK_FRAMES, [ceilingJudge(EVERY_BOUNCE)]);
            const defaultNetwork = await prepareOnly(small, { ...TAIL, neural: { network: 'default' } });
            const compact = await prepareOnly(small, TAIL);
            data.f32Fallback = { device: { maxComputeWorkgroupStorageSize: small.limits.maxComputeWorkgroupStorageSize, shaderF16: small.features.has('shader-f16') }, compact, furnace: summary(fallback), defaultNetwork };
            if (!compact.tail || compact.precision !== 'f32') fail(`f32 fallback: the compact network on a default-limits device made ${compact.tail ? `a ${compact.precision} tail` : 'no tail'} (expected f32)`);
            if (Math.abs(final(fallback)) > TOLERANCE) fail(`f32 fallback: the furnace (8 px cells) reads ${(final(fallback) * 100).toFixed(2)}% off ${EVERY_BOUNCE.toFixed(4)}`);
            if (defaultNetwork.tail || !defaultNetwork.warnings.some((message) => message.includes('maxComputeWorkgroupStorageSize'))) {
                fail(`f32 fallback: the default network on a default-limits device should be off with a warning naming maxComputeWorkgroupStorageSize (tail ${defaultNetwork.tail}, warnings ${JSON.stringify(defaultNetwork.warnings)})`);
            }
        } finally {
            small.destroy();
        }
    }

    // 8. Memory by screen size (recorded).
    if (runs(8)) {
        data.memory = { '1280x720': await memoryAt(device, 1280, 720), '1920x1080': await memoryAt(device, 1920, 1080), '3840x2160': await memoryAt(device, 3840, 2160) };
    }

    // 9. The training targets on Bistro exterior (recorded): cell and screen records' means, the screen's zero targets.
    if (runs(9)) {
        data.targets = { 'bistro-exterior': await targetsOn(device, 'bistro-exterior') };
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
