/**
 * P6.2, lever E8: the world cache's cells lit on a budget (`cellShading:
 * 'budgeted'`, cell-shading-wgsl.ts) against GI-1.2's per-hit lighting, on
 * the world-cache page's analytic scenes (radiance-cache-world-cache), at
 * 1280 × 720, through the technique.
 *
 * The scenes, as there: the **corner** (a 400 m floor and a 40 × 10 m wall,
 * albedo 0.5, black sky, camera at (0, 2, 6) looking at the wall's foot), lit
 * by a sun along (0, -1, -1) of intensity 8, so floor and wall both leave
 * albedo / π × 8 cos 45° toward any probe; the **furnace** (floor and ceiling
 * 3 m apart, albedo 0.5, a shadowless sun of 4 straight down), whose ceiling
 * reads only the second bounce ρ² E / π; and the furnace over **strips**
 * (half the floor gone), whose ceiling reads half that.
 *
 * 1. **The lit plane converges** (single bounce), for each budget (5k, 20k,
 *    80k) and per hit, from an empty cache: the frame from which every probe
 *    ray that hit reads albedo / π × E within 2% (a single sun: every sample
 *    is exact, so a ray is off only while no level of its cell has a sample),
 *    then every lit finest cell's mean within 2%. Every frame, the lit count
 *    is min(candidates, budget). Gate: converged by `LIT_PLANE_FRAMES`.
 * 2. **The queue** (the lit plane at 5k, 12 frames from an empty cache, read
 *    back every frame): the candidates are the distinct cells of the
 *    frame's visibility entries, each listed once; the lit cells are distinct
 *    candidates, as many as min(candidates, budget); every lit candidate's
 *    priority bin is no worse than any unlit one's (the counting sort); new
 *    cells come first (frame 1: every candidate is new); and the sample is
 *    drawn uniformly from the cell's entries: over every lit cell with two or
 *    more entries, the winner's rank among them (0 the first, 1 the last)
 *    averages 1/2 within 0.03 (about 3 standard errors).
 * 3. **The furnace** (multibounce, 20k and per hit): the frame from which the
 *    near ceiling rays (tiles under 2 m, which can't hold floor hits) average
 *    ρ² E / π within 1%; gate by `FURNACE_FRAMES`, and then each near
 *    ceiling ray within 2% (0.5% of them may hold a rare miss).
 * 4. **Strips** (multibounce, 20k): after `FURNACE_FRAMES`, the near ceiling
 *    rays average ρ² E / π × 1/2 within 2%.
 * 5. **Lag after a light change** (single bounce, per hit and each budget):
 *    converged for 60 frames, then (a) the sun's intensity doubled (a change
 *    with no bounds, `global: ['sun']`) or (b) the corner lit by a point
 *    light at (0, 1.5, 2) whose intensity doubles (a dirty region, its
 *    reach); per frame the mean radiance of the rays that hit, against the
 *    mean the same setting settles at 150 to 200 frames after the change:
 *    the frames until it stays within 5%. Gate: the budgets at 20k and 80k
 *    follow the sun no slower than per hit (their changed cells start over).
 *    Recorded: every lag.
 *
 * Correctness and measurement, no timing (the cost is
 * radiance-cache-cell-shading-cost). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-cell-shading --gpu nvidia`.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat, unpackHalf3 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { CELL_SHADING_BINS, CELL_SHADING_SUB_BINS, HASH_GRID_COUNTER } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { CELL_SHADING_SALT } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/cell-shading-common.ts';
import { CELL_CANDIDATE_BIN_SHIFT, CELL_CANDIDATE_CHANGED_BIT, CELL_PRIORITY_CLASS, CELL_SHADING_CHANGE_FRAMES, CELL_SHADING_MAX_ENTRIES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/cell-shading-wgsl.ts';
import type { HashGridCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cache.ts';
import { hashGridCacheCellSizeConstant, hashGridCacheGetCellSize, pcgHash, verticalFieldOfView } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import { MULTIBOUNCE_MISS_CELL } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { multibounceSettledIndirect } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/multibounce-cpu.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const CORNER_CAMERA: PageCamera = { position: [0, 2, 6], target: [0, 1, 0], fovYDeg: 60, near: 0.05, far: 1000 };
const ALBEDO = 0.5;
const SUN_INTENSITY = 8;
const SUN_TRAVEL: Vec3 = [0, -1, -1];
const LIT_RADIANCE = (ALBEDO / Math.PI) * SUN_INTENSITY * Math.SQRT1_2;
const TOLERANCE = 0.02;
const FURNACE = {
    camera: { position: [0, 2, 0], target: [0, 0, -2], fovYDeg: 60, near: 0.05, far: 1000 } as PageCamera,
    halfWidth: 200,
    ceilingY: 3,
    albedo: 0.5,
    sunIntensity: 4,
} as const;
const FLOOR_RADIANCE = (FURNACE.albedo * FURNACE.sunIntensity) / Math.PI;
const CEILING_INDIRECT = multibounceSettledIndirect(FURNACE.albedo, FLOOR_RADIANCE);
const STRIPS = { width: 0.05, period: 0.1, halfLength: 100, fill: 0.5 } as const;
const STRIPS_INDIRECT = multibounceSettledIndirect(FURNACE.albedo, FLOOR_RADIANCE, STRIPS.fill);
/** Tiles under this size can't reach from the ceiling to the floor (the world-cache page's NEAR_TILE_SIZE). */
const NEAR_TILE_SIZE = 2;
const TILE_CELLS = 8;

/** The budgets measured, and per hit (null). */
const BUDGETS = [5000, 20000, 80000] as const;
/** Gates: frames to converge on the lit plane and in the furnace, at any budget measured. */
const LIT_PLANE_FRAMES = 150;
const FURNACE_FRAMES = 200;
const QUEUE_FRAMES = 12;
const LAG_CONVERGE_FRAMES = 60;
const LAG_FRAMES = 200;
/** The settled mean after a change: frames LAG_SETTLED_FROM to LAG_FRAMES after it. */
const LAG_SETTLED_FROM = 150;
const LAG_WITHIN = 0.05;
/**
 * The furnace's near ceiling rays more than 2% off allowed with cells lit on the budget: 2%, where
 * per hit allows 0.5% (the rare misses). A far floor tile also holds ceiling hits (a cell is a
 * column through its tile), so a far floor cell lit from one drawn hit a frame is noisier than
 * per hit's mean over all its hits, and the second-bounce samples that read it are too (first run:
 * 0.94% of the rays, the mean within 0.05%).
 */
const BUDGETED_RAYS_OFF = 0.02;

type Mode = 'per-hit' | number;

function settingsFor(mode: Mode, multibounce: boolean): Partial<RadianceCacheSettings> {
    return mode === 'per-hit' ? { useMultibounce: multibounce, cellShading: 'per-hit' } : { useMultibounce: multibounce, cellShading: 'budgeted', cellShadingBudget: mode };
}

function modeName(mode: Mode): string {
    return mode === 'per-hit' ? 'per-hit' : `budgeted ${mode / 1000}k`;
}

// ---- Scenes (as radiance-cache-world-cache builds them) ----------------------------------------

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

type SceneKind = 'corner' | 'furnace' | 'strips';

function describe(platform: GIPlatformImpl, kind: SceneKind): void {
    if (kind === 'corner') {
        quad(platform, 'floor', [[-200, 0, 200], [200, 0, 200], [200, 0, -200], [-200, 0, -200]], [0, 1, 0], ALBEDO);
        quad(platform, 'wall', [[-20, 0, 0], [20, 0, 0], [20, 10, 0], [-20, 10, 0]], [0, 0, 1], ALBEDO);
    } else {
        const half = FURNACE.halfWidth;
        const y = FURNACE.ceilingY;
        if (kind === 'strips') stripFloor(platform);
        else quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], FURNACE.albedo);
        quad(platform, 'ceiling', [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]], [0, -1, 0], FURNACE.albedo);
    }
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
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

const CORNER_SUN: LightDescriptor = { type: 'directional', direction: SUN_TRAVEL, color: [1, 1, 1], intensity: SUN_INTENSITY, label: 'sun' };
const FURNACE_SUN: LightDescriptor = { type: 'directional', direction: [0, -1, 0], color: [1, 1, 1], intensity: FURNACE.sunIntensity, castsShadows: false, label: 'furnace sun' };
const CORNER_POINT: LightDescriptor = { type: 'point', position: [0, 1.5, 2], range: 0, color: [1, 1, 1], intensity: 10, label: 'point' };

interface Scenario {
    readonly platform: GIPlatformImpl;
    readonly technique: RadianceCacheTechnique;
    readonly cache: HashGridCache;
    readonly lights: readonly LightHandle[];
    readonly camera: PageCamera;
    frame(): void;
    dispose(): void;
}

async function startScenario(device: GPUDevice, name: string, kind: SceneKind, lights: readonly LightDescriptor[], settings: Partial<RadianceCacheSettings>): Promise<Scenario> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-cell-shading ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    describe(platform, kind);
    const handles = lights.map((light) => platform.scene.createLight(light));
    const camera = kind === 'corner' ? CORNER_CAMERA : FURNACE.camera;
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `radiance-cache-cell-shading ${name}` });
        const input: FrameInput = { deltaTime: 1 / 60, budget: { units: 0 }, camera: frameCamera(camera, WIDTH, HEIGHT) };
        platform.encode(encoder, input);
        screen.encode(encoder, camera);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    frame();
    await platform.rays.whenBuilt();
    // Past the window lever E8 remembers scene changes for (the scene's creation is one), so the
    // checks start from an empty cache with no change pending.
    for (let index = 0; index <= CELL_SHADING_CHANGE_FRAMES; index++) frame();
    await device.queue.onSubmittedWorkDone();
    technique.resetHistory();
    const cache = technique.worldCache;
    if (!cache) throw new Error(`${name}: no world cache after prepare`);
    return {
        platform,
        technique,
        cache,
        lights: handles,
        camera,
        frame,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

// ---- Reading back ---------------------------------------------------------------------------------

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const readback = device.createBuffer({ label: 'cell-shading readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'cell-shading readback' });
    encoder.copyBufferToBuffer(source, offset, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0, byteLength);
    readback.unmap();
    readback.destroy();
    return copy;
}

interface RayState {
    /** Per ray: hit point, or null for a miss. */
    readonly hits: (Vec3 | null)[];
    /** Per ray: rgb radiance and the hit distance, as halves. */
    readonly radiance: Float32Array;
}

async function readRays(device: GPUDevice, scenario: Scenario): Promise<RayState> {
    const probes = scenario.technique.probes;
    if (!probes) throw new Error('no probes after encodeScreen');
    const count = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
    const words = new Uint32Array(await readRange(device, probes.probeRays, 0, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + count * WORLD_CACHE_PROBE_RAY_BYTES));
    const floats = new Float32Array(words.buffer);
    const hits: (Vec3 | null)[] = [];
    for (let ray = 0; ray < count; ray++) {
        const base = (WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES) / 4;
        const hitT = floats[base + 6];
        if (hitT < 0) {
            hits.push(null);
            continue;
        }
        const direction = unpackHalf3([words[base + 4], words[base + 5]]);
        hits.push([floats[base] + hitT * direction[0], floats[base + 1] + hitT * direction[1], floats[base + 2] + hitT * direction[2]]);
    }
    const results = probes.rayScratch.rayResults;
    const packed = new Uint32Array(await readRange(device, results.buffer, results.offset ?? 0, count * 8));
    const radiance = new Float32Array(count * 4);
    for (let index = 0; index < packed.length; index++) {
        radiance[index * 2] = halfBitsToFloat(packed[index] & 0xffff);
        radiance[index * 2 + 1] = halfBitsToFloat(packed[index] >>> 16);
    }
    return { hits, radiance };
}

async function readCounters(device: GPUDevice, cache: HashGridCache): Promise<Uint32Array> {
    return new Uint32Array(await readRange(device, cache.buffers.tileWords, 0, 64 * 4));
}

/** The mean radiance (rgb averaged) of the rays that hit, and how many are within `TOLERANCE` of `expected`. */
function hitRayStats(state: RayState, expected?: number): { mean: number; hits: number; within: number } {
    let sum = 0;
    let hits = 0;
    let within = 0;
    state.hits.forEach((hit, ray) => {
        if (!hit) return;
        const value = (state.radiance[ray * 4] + state.radiance[ray * 4 + 1] + state.radiance[ray * 4 + 2]) / 3;
        sum += value;
        hits++;
        if (expected !== undefined && Math.abs(value / expected - 1) <= TOLERANCE) within++;
    });
    return { mean: sum / Math.max(hits, 1), hits, within };
}

// ---- 1. The lit plane converges ---------------------------------------------------------------------

async function litPlane(device: GPUDevice, mode: Mode, fail: (message: string) => void): Promise<Record<string, number>> {
    const scenario = await startScenario(device, `lit plane ${modeName(mode)}`, 'corner', [CORNER_SUN], settingsFor(mode, false));
    try {
        let convergedAt = -1;
        let candidatesMax = 0;
        let shadowRaysMean = 0;
        for (let frame = 1; frame <= LIT_PLANE_FRAMES; frame++) {
            scenario.frame();
            const counters = await readCounters(device, scenario.cache);
            const state = await readRays(device, scenario);
            const stats = hitRayStats(state, LIT_RADIANCE);
            shadowRaysMean += counters[HASH_GRID_COUNTER.visibilityRayCount] / LIT_PLANE_FRAMES;
            if (mode !== 'per-hit') {
                const candidates = counters[HASH_GRID_COUNTER.cellCandidateCount];
                candidatesMax = Math.max(candidatesMax, candidates);
                if (counters[HASH_GRID_COUNTER.cellShadedCount] !== Math.min(candidates, mode)) fail(`lit plane ${modeName(mode)} frame ${frame}: ${counters[HASH_GRID_COUNTER.cellShadedCount]} cells lit of ${candidates} candidates`);
            }
            if (stats.within === stats.hits) {
                if (convergedAt < 0) convergedAt = frame;
            } else convergedAt = -1;
        }
        if (convergedAt < 0) fail(`lit plane ${modeName(mode)}: not every hit ray within 2% of ${LIT_RADIANCE.toFixed(4)} after ${LIT_PLANE_FRAMES} frames`);
        const final = hitRayStats(await readRays(device, scenario), LIT_RADIANCE);
        return { convergedAtFrame: convergedAt, meanOverExpected: Number((final.mean / LIT_RADIANCE).toFixed(5)), hitRays: final.hits, candidatesMax, shadowRaysPerFrame: Math.round(shadowRaysMean) };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- 2. The queue ---------------------------------------------------------------------------------------

/** `HashGridCache_CellSampleKey` on the CPU. */
function sampleKey(visibilityIndex: number, frameIndex: number): number {
    return (pcgHash((pcgHash((visibilityIndex ^ CELL_SHADING_SALT.select) >>> 0) + pcgHash(frameIndex >>> 0)) >>> 0) | 1) >>> 0;
}

async function queue(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, number>> {
    const budget = BUDGETS[0];
    const scenario = await startScenario(device, 'queue', 'corner', [CORNER_SUN], settingsFor(budget, false));
    const tally = { frames: 0, candidates: 0, lit: 0, rankSum: 0, rankCells: 0, keyMismatches: 0, frameOneNotNew: 0, orderViolations: 0 };
    try {
        for (let frame = 1; frame <= QUEUE_FRAMES; frame++) {
            scenario.frame();
            // The frame number the cache's constants carried (g_FrameIndex), which the keys are drawn with.
            const frameIndex = scenario.platform.stats.frame;
            const counters = await readCounters(device, scenario.cache);
            const lists = scenario.cache.rayLists;
            if (!lists) throw new Error('no ray lists');
            const words = new Uint32Array(await readRange(device, lists.buffer, 0, lists.buffer.size));
            const layout = lists.layout;
            const entries = counters[HASH_GRID_COUNTER.visibilityCount0] + counters[HASH_GRID_COUNTER.visibilityCount1];
            const candidates = counters[HASH_GRID_COUNTER.cellCandidateCount];
            const lit = counters[HASH_GRID_COUNTER.cellShadedCount];
            // The cells' entries, in entry order.
            const entriesOf = new Map<number, number[]>();
            for (let entry = 0; entry < entries; entry++) {
                const cell = words[layout.visibilityCellBuffer + entry];
                if (cell === MULTIBOUNCE_MISS_CELL) continue;
                const list = entriesOf.get(cell);
                if (list) list.push(entry);
                else entriesOf.set(cell, [entry]);
            }
            if (candidates !== entriesOf.size) fail(`queue frame ${frame}: ${candidates} candidates for ${entriesOf.size} distinct cells hit`);
            if (lit !== Math.min(candidates, budget)) fail(`queue frame ${frame}: ${lit} lit of ${candidates} candidates at a budget of ${budget}`);
            const entryMask = CELL_SHADING_MAX_ENTRIES - 1;
            const candidateCells = new Map<number, { entry: number; bin: number }>();
            for (let index = 0; index < candidates; index++) {
                const word = words[layout.cellCandidateBuffer + index];
                const entry = word & entryMask;
                const cell = words[layout.visibilityCellBuffer + entry];
                if (candidateCells.has(cell)) fail(`queue frame ${frame}: cell ${cell} listed twice`);
                candidateCells.set(cell, { entry, bin: word >>> CELL_CANDIDATE_BIN_SHIFT });
                // The claimer is the entry with the cell's largest key (a tie, one in billions, is skipped).
                const keys = (entriesOf.get(cell) ?? []).map((other) => sampleKey(other, frameIndex));
                if (keys.length > 0 && sampleKey(entry, frameIndex) !== Math.max(...keys)) tally.keyMismatches++;
                if (frame === 1 && word >>> CELL_CANDIDATE_BIN_SHIFT >= CELL_SHADING_SUB_BINS * (CELL_PRIORITY_CLASS.newOrChanged + 1)) tally.frameOneNotNew++;
            }
            const litCells = new Set<number>();
            let worstLitBin = 0;
            for (let slot = 0; slot < lit; slot++) {
                const word = words[layout.cellShadeBuffer + slot];
                const entry = word & entryMask;
                if (word & CELL_CANDIDATE_CHANGED_BIT) fail(`queue frame ${frame}: a lit cell marked changed with no change`);
                const cell = words[layout.visibilityCellBuffer + entry];
                const candidate = candidateCells.get(cell);
                if (!candidate || candidate.entry !== entry) fail(`queue frame ${frame}: lit slot ${slot} is not a candidate's entry`);
                if (litCells.has(cell)) fail(`queue frame ${frame}: cell ${cell} lit twice`);
                litCells.add(cell);
                worstLitBin = Math.max(worstLitBin, candidate?.bin ?? 0);
                const cellEntries = entriesOf.get(cell) ?? [];
                if (cellEntries.length >= 2) {
                    tally.rankSum += cellEntries.indexOf(entry) / (cellEntries.length - 1);
                    tally.rankCells++;
                }
            }
            for (const [cell, candidate] of candidateCells) if (!litCells.has(cell) && candidate.bin < worstLitBin) tally.orderViolations++;
            tally.frames++;
            tally.candidates += candidates;
            tally.lit += lit;
        }
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
    const meanRank = tally.rankSum / Math.max(tally.rankCells, 1);
    if (tally.rankCells < 2000) fail(`queue: only ${tally.rankCells} lit cells with two or more entries; the rank check needs 2,000`);
    if (Math.abs(meanRank - 0.5) > 0.03) fail(`queue: the sample's rank among its cell's entries averages ${meanRank.toFixed(4)}, not 1/2: the draw isn't uniform`);
    if (tally.keyMismatches > 0) fail(`queue: ${tally.keyMismatches} candidates are not their cell's largest key`);
    if (tally.frameOneNotNew > 0) fail(`queue: ${tally.frameOneNotNew} candidates of the first frame are not in the new class`);
    if (tally.orderViolations > 0) fail(`queue: ${tally.orderViolations} unlit candidates have a better bin than a lit one`);
    return { ...tally, meanRank: Number(meanRank.toFixed(4)), bins: CELL_SHADING_BINS };
}

// ---- 3 and 4. The furnace and the strips --------------------------------------------------------------

function nearCeilingStats(state: RayState, expected: number): { mean: number; rays: number; off: number } {
    const cellSize = hashGridCacheCellSizeConstant(verticalFieldOfView(frameCamera(FURNACE.camera, WIDTH, HEIGHT).projection), WIDTH, HEIGHT, 32);
    let sum = 0;
    let rays = 0;
    const values: number[] = [];
    state.hits.forEach((hit, ray) => {
        if (!hit || Math.abs(hit[1] - FURNACE.ceilingY) > 1e-3) return;
        if (hashGridCacheGetCellSize(hit, FURNACE.camera.position, { cellSize, minCellSize: 0.1 }) * TILE_CELLS >= NEAR_TILE_SIZE) return;
        const value = (state.radiance[ray * 4] + state.radiance[ray * 4 + 1] + state.radiance[ray * 4 + 2]) / 3;
        sum += value;
        rays++;
        values.push(value);
    });
    const mean = sum / Math.max(rays, 1);
    return { mean, rays, off: values.filter((value) => Math.abs(value / expected - 1) > TOLERANCE).length };
}

async function furnace(device: GPUDevice, mode: Mode, kind: 'furnace' | 'strips', fail: (message: string) => void): Promise<Record<string, number>> {
    const expected = kind === 'furnace' ? CEILING_INDIRECT : STRIPS_INDIRECT;
    const scenario = await startScenario(device, `${kind} ${modeName(mode)}`, kind, [FURNACE_SUN], settingsFor(mode, true));
    try {
        let convergedAt = -1;
        for (let frame = 1; frame <= FURNACE_FRAMES; frame++) {
            scenario.frame();
            // Read every 5th frame (and the last): the furnace converges over tens of frames.
            if (frame % 5 !== 0 && frame !== FURNACE_FRAMES) continue;
            const stats = nearCeilingStats(await readRays(device, scenario), expected);
            if (Math.abs(stats.mean / expected - 1) <= 0.01) {
                if (convergedAt < 0) convergedAt = frame;
            } else convergedAt = -1;
        }
        const final = nearCeilingStats(await readRays(device, scenario), expected);
        const name = `${kind} ${modeName(mode)}`;
        if (final.rays === 0) fail(`${name}: no near ceiling ray`);
        if (kind === 'furnace') {
            if (convergedAt < 0) fail(`${name}: the near ceiling rays average ${final.mean.toFixed(4)}, not ${expected.toFixed(4)} within 1%, after ${FURNACE_FRAMES} frames`);
            if (final.off > Math.ceil((mode === 'per-hit' ? 0.005 : BUDGETED_RAYS_OFF) * final.rays)) fail(`${name}: ${final.off} of ${final.rays} near ceiling rays off ${expected.toFixed(4)} by more than 2%`);
        } else if (Math.abs(final.mean / expected - 1) > TOLERANCE) fail(`${name}: the near ceiling rays average ${final.mean.toFixed(4)}, expected ${expected.toFixed(4)} within 2%`);
        return { convergedWithin1PercentAtFrame: convergedAt, meanOverExpected: Number((final.mean / expected).toFixed(5)), nearRays: final.rays, nearRaysOff: final.off };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- 5. Lag after a light change -----------------------------------------------------------------------

async function lag(device: GPUDevice, mode: Mode, light: 'sun' | 'point'): Promise<Record<string, number>> {
    const descriptor = light === 'sun' ? CORNER_SUN : CORNER_POINT;
    const scenario = await startScenario(device, `lag ${light} ${modeName(mode)}`, 'corner', [descriptor], settingsFor(mode, false));
    try {
        for (let frame = 0; frame < LAG_CONVERGE_FRAMES; frame++) scenario.frame();
        const before = hitRayStats(await readRays(device, scenario)).mean;
        scenario.platform.scene.updateLight(scenario.lights[0], { intensity: 2 * descriptor.intensity });
        const means: number[] = [];
        for (let frame = 0; frame < LAG_FRAMES; frame++) {
            scenario.frame();
            means.push(hitRayStats(await readRays(device, scenario)).mean);
        }
        const settled = means.slice(LAG_SETTLED_FROM).reduce((sum, value) => sum + value, 0) / (LAG_FRAMES - LAG_SETTLED_FROM);
        // The first frame after the change from which every later frame is within 5% of the settled mean.
        let lagFrames = LAG_FRAMES;
        for (let frame = LAG_FRAMES - 1; frame >= 0; frame--) {
            if (Math.abs(means[frame] / settled - 1) > LAG_WITHIN) break;
            lagFrames = frame + 1;
        }
        return {
            framesToWithin5Percent: lagFrames,
            settledOverBefore: Number((settled / before).toFixed(4)),
            firstFrameOverSettled: Number((means[0] / settled).toFixed(4)),
            ...(light === 'sun' ? { settledOverAnalytic: Number((settled / (2 * LIT_RADIANCE)).toFixed(4)) } : {}),
        };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- The page ---------------------------------------------------------------------------------------------

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
        if (failures.length < 24) failures.push(message);
    };
    const modes: Mode[] = ['per-hit', ...BUDGETS];
    const data: Record<string, unknown> = { litRadiance: LIT_RADIANCE, ceilingIndirect: CEILING_INDIRECT, stripsIndirect: STRIPS_INDIRECT };

    const lit: Record<string, unknown> = {};
    for (const mode of modes) lit[modeName(mode)] = await litPlane(device, mode, fail);
    data.litPlane = lit;

    data.queue = await queue(device, fail);

    const furnaces: Record<string, unknown> = {};
    for (const mode of ['per-hit', 20000] as const) furnaces[modeName(mode)] = await furnace(device, mode, 'furnace', fail);
    furnaces[`strips ${modeName(20000)}`] = await furnace(device, 20000, 'strips', fail);
    data.furnace = furnaces;

    const lags: Record<string, Record<string, Record<string, number>>> = { sun: {}, point: {} };
    for (const light of ['sun', 'point'] as const) for (const mode of modes) lags[light][modeName(mode)] = await lag(device, mode, light);
    data.lag = lags;
    const perHitSunLag = lags.sun['per-hit'].framesToWithin5Percent;
    for (const budget of [20000, 80000]) {
        const budgetedLag = lags.sun[modeName(budget)].framesToWithin5Percent;
        if (budgetedLag > perHitSunLag) fail(`lag: ${modeName(budget)} follows the sun in ${budgetedLag} frames, per hit in ${perHitSunLag}`);
    }

    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data,
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
