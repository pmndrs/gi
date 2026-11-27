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
 *    ρ² E / π within 1%; gate by `FURNACE_FRAMES`. Then the analytic gate
 *    (RC3.4, the world-cache page's): the second bounces from near ceiling
 *    cells that land on the floor in tiles under 2 m read ρ L_f = ρ² E / π
 *    within 1%, in every mode (`second-bounce-landings.ts`). How many near
 *    ceiling rays are more than 2% off is recorded, not gated: since register
 *    D63 a ray reads its cell's mean over landings in floor cells of
 *    different floor shares (a far floor cell is a column holding ceiling
 *    hits too), so single rays aren't ρ² E / π.
 * 4. **Strips** (multibounce, 20k): after `FURNACE_FRAMES`, the near ceiling
 *    rays average ρ² E / π × 1/2 within 2%, and the landings on strips in
 *    floor tiles under 2 m read ρ L_f within 1%.
 * 5. **Lag after a light change** (single bounce, per hit and each budget):
 *    converged for 60 frames, then (a) the sun's intensity doubled (a change
 *    with no bounds, `global: ['sun']`) or (b) the corner lit by a point
 *    light at (0, 1.5, 2) whose intensity doubles (a dirty region, its
 *    reach); per frame the mean radiance of the rays that hit, against the
 *    mean the same setting settles at 150 to 200 frames after the change:
 *    the frames until it stays within 5%. Gate: the budgets at 20k and 80k
 *    follow the sun no slower than per hit (their changed cells start over).
 *    Recorded: every lag.
 * 6. **A light switched off** (multibounce, the corner's sun to 0): the frames
 *    until the hit rays' mean is under 5% of what it was; E9 and the budget
 *    at 20k (its hits with no light are samples of 0, register D63) must get
 *    there faster than per hit. And the furnace's second bounce after its sun
 *    doubles (E9 faster than per hit).
 *
 * P6.2b (experiment E9, `cellChangeReset`): every section also runs per hit
 * with E9 on, which must converge as per hit does and follow both changes
 * in fewer frames than per hit without it.
 *
 * 7. **The shared-cell bounce** (pmndrs/gi#54, `secondBounce: 'shared-cell'`,
 *    `cellShading` left per hit, which the mode overrides; multibounce on),
 *    run first. Known answers:
 *    a. cold cache: the furnace's first frame from an empty cache, every lit
 *       (ceiling) cell's one indirect sample is exact: ρ² E / π, one sample;
 *    b. indirect miss: the ceiling alone under a white sky (the world-cache
 *       page's scene), every lit cell's indirect value is exactly 0 with a
 *       count (no sky counted twice, misses counted);
 *    c. short-ray bypass: cells from 8 m at a budget of 0, the near ceiling rays
 *       shorter than their cell average ρ² E / π within 1.5% (ShortRayBounce's
 *       roulette and its 1 / (1 - p)); rays through their cell read 0;
 *    d. at 20k, the furnace (ρ² E / π within 1% by `FURNACE_FRAMES`) and the
 *       strips (half that within 2%: misses count as samples), seen from above:
 *       the probes are all on the floor, so no floor cell is lit and every
 *       continuation lands cold; and the furnace seen level from halfway up,
 *       where probes on the ceiling light the floor cells, so continuations read
 *       warm landing cells (ρ² E / π within 1%);
 *    e. zero budget: nothing lit, every ray through its cell reads exactly 0;
 *    and the furnace's lag after its sun doubles (recorded).
 *    `VITE_RC_CELL_SHADING_ONLY=shared` runs section 7 alone.
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
import { CELL_CANDIDATE_CHANGED_BIT, CELL_CANDIDATE_ENTRY_MASK, CELL_PRIORITY_CLASS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/cell-shading-wgsl.ts';
import type { HashGridCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cache.ts';
import { hashGridCacheCellSizeConstant, hashGridCacheGetCellSize, pcgHash, verticalFieldOfView } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import { MULTIBOUNCE_MISS_CELL } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { multibounceSettledIndirect } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/multibounce-cpu.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';
import { readSecondBounceLandings } from '../radiance-cache-world-cache/second-bounce-landings.ts';

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
/** Frames past the scene's creation before a check starts: past every change event's epochs (the tracker forgets an event 16 frames after it stops; it re-arms every 8). */
const PAST_CHANGES_FRAMES = 32;
/** E9's lag gate under a single light: a change is followed within this many frames. */
const E9_LAG_FRAMES = 3;

/** The lit count against the frame's target, min(candidates, budget): the thinned bin can fall a little short. */
function litCountOk(lit: number, target: number): boolean {
    return lit <= target && lit >= target - 4 * Math.sqrt(target) - 1;
}
/**
 * The analytic gate on the second bounce (RC3.4, the lead's ruling of 2026-11-23; the world-cache
 * page's): the landings in near floor tiles read ρ L_f within 1%. It replaced a per-ray gate (at
 * most 0.5% of the near ceiling rays more than 2% off ρ² E / π per hit, 2% budgeted), which the
 * page still records: a far floor tile also holds ceiling hits (a cell is a column through its
 * tile), and since register D63 those count as samples of no light, so a near ceiling ray reads a
 * mean over landings of different floor shares (per hit: 15,599 of 127,454 rays off, mean 0.9919;
 * with D63's samples removed 275 of 110,467). Budgeted cells have D63 since P6.2's fix 3 (a drawn
 * hit with no light is a sample of 0 in ShadeCells), so the same holds for them.
 */
const LANDING_TOLERANCE = 0.01;

/**
 * pmndrs/gi#54: `secondBounce: 'shared-cell'` at a budget of `shared` cells. `cellShading` is left
 * per hit: the mode lights cells on the budget itself. `minCellSize` raises the grid's smallest
 * cell (the bypass check makes every near ray shorter than its cell).
 */
interface SharedMode {
    readonly shared: number;
    readonly minCellSize?: number;
}

/** Per hit (GI-1.2), per hit with experiment E9's change reset (history dropped, or 3 samples kept), budgeted at that many cells (E8), or the shared-cell bounce. */
type Mode = 'per-hit' | 'per-hit+E9' | 'per-hit+E9 keep 3' | number | SharedMode;

function settingsFor(mode: Mode, multibounce: boolean): Partial<RadianceCacheSettings> {
    if (mode === 'per-hit') return { useMultibounce: multibounce, cellShading: 'per-hit', cellChangeReset: false };
    if (mode === 'per-hit+E9') return { useMultibounce: multibounce, cellShading: 'per-hit', cellChangeReset: true, cellChangeResetHistory: 0 };
    if (mode === 'per-hit+E9 keep 3') return { useMultibounce: multibounce, cellShading: 'per-hit', cellChangeReset: true, cellChangeResetHistory: 3 };
    if (typeof mode === 'object') {
        return {
            useMultibounce: multibounce,
            cellShading: 'per-hit',
            secondBounce: 'shared-cell',
            cellShadingBudget: mode.shared,
            cellChangeReset: false,
            ...(mode.minCellSize === undefined ? {} : { hashGridCacheMinCellSize: mode.minCellSize }),
        };
    }
    return { useMultibounce: multibounce, cellShading: 'budgeted', cellShadingBudget: mode, cellChangeReset: false };
}

function modeName(mode: Mode): string {
    if (typeof mode === 'object') return `shared-cell ${mode.shared / 1000}k${mode.minCellSize === undefined ? '' : `, cells from ${mode.minCellSize} m`}`;
    return typeof mode === 'number' ? `budgeted ${mode / 1000}k` : mode;
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

/**
 * `'ceiling only'` (the world-cache page's): the furnace's ceiling under a white sky, and a black
 * 1 × 1 m patch on the ground for the probes, the camera 0.3 m over it looking down. The ceiling's
 * continuations go down: nearly all miss, the rest land on the black patch.
 */
type SceneKind = 'corner' | 'furnace' | 'strips' | 'ceiling only' | 'furnace level';
/** The furnace seen level, from halfway up: probes on floor and ceiling, so floor cells are lit too (the shared bounce's warm landings). */
const FURNACE_LEVEL_CAMERA: PageCamera = { position: [0, 1.5, 0], target: [0, 1.5, -2], fovYDeg: 60, near: 0.05, far: 1000 };
const CEILING_ONLY_CAMERA: PageCamera = { position: [0, 0.3, 0], target: [0, 0, 0], fovYDeg: 60, near: 0.05, far: 1000 };

function describe(platform: GIPlatformImpl, kind: SceneKind): void {
    if (kind === 'ceiling only') {
        const half = FURNACE.halfWidth;
        const y = FURNACE.ceilingY;
        quad(platform, 'ceiling', [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]], [0, -1, 0], FURNACE.albedo);
        quad(platform, 'black probe patch', [[-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0, -0.5], [-0.5, 0, -0.5]], [0, 1, 0], 0);
        platform.scene.setEnvironment({ type: 'color', color: [1, 1, 1], intensity: 1 });
        return;
    }
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
    const camera = kind === 'corner' ? CORNER_CAMERA : kind === 'ceiling only' ? CEILING_ONLY_CAMERA : kind === 'furnace level' ? FURNACE_LEVEL_CAMERA : FURNACE.camera;
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
    for (let index = 0; index < PAST_CHANGES_FRAMES; index++) frame();
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
    /** Per ray: its origin. */
    readonly origins: Vec3[];
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
    const origins: Vec3[] = [];
    for (let ray = 0; ray < count; ray++) {
        const base = (WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES) / 4;
        origins.push([floats[base], floats[base + 1], floats[base + 2]]);
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
    return { hits, origins, radiance };
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
            if (typeof mode === 'number') {
                const candidates = counters[HASH_GRID_COUNTER.cellCandidateCount];
                candidatesMax = Math.max(candidatesMax, candidates);
                const target = counters[HASH_GRID_COUNTER.cellShadedTarget];
                const lit = Math.min(counters[HASH_GRID_COUNTER.cellShadedCount], target);
                if (target !== Math.min(candidates, mode) || !litCountOk(lit, target)) fail(`lit plane ${modeName(mode)} frame ${frame}: ${lit} cells lit of ${candidates} candidates (target ${target})`);
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
            const cellLists = scenario.cache.cellShadingLists;
            if (!lists || !cellLists) throw new Error('no ray lists or cell lists');
            const words = new Uint32Array(await readRange(device, lists.buffer, 0, lists.buffer.size));
            const cellWords = new Uint32Array(await readRange(device, cellLists.buffer, 0, cellLists.buffer.size));
            const layout = lists.layout;
            const cellLayout = cellLists.layout;
            const entries = counters[HASH_GRID_COUNTER.visibilityCount0] + counters[HASH_GRID_COUNTER.visibilityCount1];
            const candidates = counters[HASH_GRID_COUNTER.cellCandidateCount];
            const target = counters[HASH_GRID_COUNTER.cellShadedTarget];
            const lit = Math.min(counters[HASH_GRID_COUNTER.cellShadedCount], target);
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
            if (target !== Math.min(candidates, budget) || !litCountOk(lit, target)) fail(`queue frame ${frame}: ${lit} lit of ${candidates} candidates at a budget of ${budget} (target ${target})`);
            const entryMask = CELL_CANDIDATE_ENTRY_MASK;
            const candidateCells = new Map<number, { entry: number; bin: number }>();
            for (let index = 0; index < candidates; index++) {
                const word = cellWords[cellLayout.candidates + 2 * index];
                const bin = cellWords[cellLayout.candidates + 2 * index + 1];
                const entry = word & entryMask;
                const cell = words[layout.visibilityCellBuffer + entry];
                if (candidateCells.has(cell)) fail(`queue frame ${frame}: cell ${cell} listed twice`);
                candidateCells.set(cell, { entry, bin });
                // The claimer is the entry with the cell's largest key (a tie, one in billions, is skipped).
                const keys = (entriesOf.get(cell) ?? []).map((other) => sampleKey(other, frameIndex));
                if (keys.length > 0 && sampleKey(entry, frameIndex) !== Math.max(...keys)) tally.keyMismatches++;
                if (frame === 1 && bin >= CELL_SHADING_SUB_BINS * (CELL_PRIORITY_CLASS.newOrChanged + 1)) tally.frameOneNotNew++;
            }
            const litCells = new Set<number>();
            let worstLitBin = 0;
            for (let slot = 0; slot < lit; slot++) {
                const word = cellWords[cellLayout.lit + slot];
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
    if (Math.abs(meanRank - 0.5) > 0.01) fail(`queue: the sample's rank among its cell's entries averages ${meanRank.toFixed(4)}, not 1/2: the draw isn't uniform`);
    if (tally.keyMismatches > 0) fail(`queue: ${tally.keyMismatches} candidates are not their cell's largest key`);
    if (tally.frameOneNotNew > 0) fail(`queue: ${tally.frameOneNotNew} candidates of the first frame are not in the new class`);
    if (tally.orderViolations > 0) fail(`queue: ${tally.orderViolations} unlit candidates have a better bin than a lit one`);
    return { ...tally, meanRank: Number(meanRank.toFixed(4)), bins: CELL_SHADING_BINS };
}

// ---- 3 and 4. The furnace and the strips --------------------------------------------------------------

function nearCeilingStats(state: RayState, expected: number, camera: PageCamera = FURNACE.camera): { mean: number; rays: number; off: number } {
    const cellSize = hashGridCacheCellSizeConstant(verticalFieldOfView(frameCamera(camera, WIDTH, HEIGHT).projection), WIDTH, HEIGHT, 32);
    let sum = 0;
    let rays = 0;
    const values: number[] = [];
    state.hits.forEach((hit, ray) => {
        if (!hit || Math.abs(hit[1] - FURNACE.ceilingY) > 1e-3) return;
        if (hashGridCacheGetCellSize(hit, camera.position, { cellSize, minCellSize: 0.1 }) * TILE_CELLS >= NEAR_TILE_SIZE) return;
        const value = (state.radiance[ray * 4] + state.radiance[ray * 4 + 1] + state.radiance[ray * 4 + 2]) / 3;
        sum += value;
        rays++;
        values.push(value);
    });
    const mean = sum / Math.max(rays, 1);
    return { mean, rays, off: values.filter((value) => Math.abs(value / expected - 1) > TOLERANCE).length };
}

async function furnace(device: GPUDevice, mode: Mode, kind: 'furnace' | 'strips' | 'furnace level', fail: (message: string) => void): Promise<Record<string, number>> {
    const expected = kind === 'strips' ? STRIPS_INDIRECT : CEILING_INDIRECT;
    const scenario = await startScenario(device, `${kind} ${modeName(mode)}`, kind, [FURNACE_SUN], settingsFor(mode, true));
    try {
        let convergedAt = -1;
        for (let frame = 1; frame <= FURNACE_FRAMES; frame++) {
            scenario.frame();
            // Read every 5th frame (and the last): the furnace converges over tens of frames.
            if (frame % 5 !== 0 && frame !== FURNACE_FRAMES) continue;
            const stats = nearCeilingStats(await readRays(device, scenario), expected, scenario.camera);
            if (Math.abs(stats.mean / expected - 1) <= 0.01) {
                if (convergedAt < 0) convergedAt = frame;
            } else convergedAt = -1;
        }
        const final = nearCeilingStats(await readRays(device, scenario), expected, scenario.camera);
        const name = `${kind} ${modeName(mode)}`;
        if (final.rays === 0) fail(`${name}: no near ceiling ray`);
        if (kind !== 'strips') {
            if (convergedAt < 0) fail(`${name}: the near ceiling rays average ${final.mean.toFixed(4)}, not ${expected.toFixed(4)} within 1%, after ${FURNACE_FRAMES} frames`);
        } else if (Math.abs(final.mean / expected - 1) > TOLERANCE) fail(`${name}: the near ceiling rays average ${final.mean.toFixed(4)}, expected ${expected.toFixed(4)} within 2%`);
        // The shared bounce (#54) keeps no second-bounce entries, so there are no landings to read: its
        // known answers are the near ceiling mean above and the cold cache's exact samples (section 7).
        if (typeof mode === 'object') {
            if (scenario.cache.secondBounceInUse !== 'shared-cell') fail(`${name}: the second bounce in use is ${scenario.cache.secondBounceInUse}, not shared-cell`);
            return { convergedWithin1PercentAtFrame: convergedAt, meanOverExpected: Number((final.mean / expected).toFixed(5)), nearRays: final.rays, nearRaysOff: final.off };
        }
        const landings = await readSecondBounceLandings(device, scenario.technique, scenario.cache, { camera: FURNACE.camera, width: WIDTH, height: HEIGHT, ceilingY: FURNACE.ceilingY });
        // The analytic gate (LANDING_TOLERANCE); `nearRaysOff` is recorded only (see there).
        const nearLandings = landings.nearFloorLandings;
        if (nearLandings.entries < 1000) fail(`${name}: only ${nearLandings.entries} second bounces land on the floor in tiles under ${NEAR_TILE_SIZE} m`);
        else if (Math.abs(nearLandings.mean / CEILING_INDIRECT - 1) > LANDING_TOLERANCE) {
            fail(`${name}: the second bounces landing on the floor in tiles under ${NEAR_TILE_SIZE} m read ${nearLandings.mean.toFixed(4)}, rho L_f = ${CEILING_INDIRECT.toFixed(4)} within ${LANDING_TOLERANCE * 100}%`);
        }
        return {
            convergedWithin1PercentAtFrame: convergedAt,
            meanOverExpected: Number((final.mean / expected).toFixed(5)),
            nearRays: final.rays,
            nearRaysOff: final.off,
            nearFloorLandingsOverRhoLf: Number((nearLandings.mean / CEILING_INDIRECT).toFixed(5)),
            nearFloorLandings: nearLandings.entries,
            throughCacheOverExpected: Number((landings.throughCache.mean / expected).toFixed(5)),
        };
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

// ---- 6. A light switched off, and the second bounce after a change ----------------------------------------

/**
 * The sun switched off (intensity 0): frames until the hit rays' mean is under 5% of what it was.
 * With multibounce on: with it off, a hit with no light isn't resolved at all (register D35), so
 * its ray reads 0 at once whatever the cells hold.
 */
async function lagToZero(device: GPUDevice, mode: Mode): Promise<Record<string, number>> {
    const scenario = await startScenario(device, `off ${modeName(mode)}`, 'corner', [CORNER_SUN], settingsFor(mode, true));
    try {
        for (let frame = 0; frame < LAG_CONVERGE_FRAMES; frame++) scenario.frame();
        const before = hitRayStats(await readRays(device, scenario)).mean;
        scenario.platform.scene.updateLight(scenario.lights[0], { intensity: 0 });
        let frames = LAG_FRAMES;
        const means: number[] = [];
        for (let frame = 0; frame < 60; frame++) {
            scenario.frame();
            means.push(hitRayStats(await readRays(device, scenario)).mean / before);
        }
        for (let frame = means.length - 1; frame >= 0; frame--) {
            if (means[frame] > LAG_WITHIN) break;
            frames = frame + 1;
        }
        return { framesToUnder5Percent: frames, firstFrameOverBefore: Number(means[0].toFixed(4)), lastFrameOverBefore: Number(means[means.length - 1].toFixed(4)) };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

/** The furnace's sun doubled: frames until the near ceiling rays (the second bounce alone) stay within 5% of where they settle. */
async function multibounceLag(device: GPUDevice, mode: Mode): Promise<Record<string, number>> {
    const scenario = await startScenario(device, `multibounce lag ${modeName(mode)}`, 'furnace', [FURNACE_SUN], settingsFor(mode, true));
    try {
        for (let frame = 0; frame < LAG_CONVERGE_FRAMES; frame++) scenario.frame();
        const before = nearCeilingStats(await readRays(device, scenario), CEILING_INDIRECT).mean;
        scenario.platform.scene.updateLight(scenario.lights[0], { intensity: 2 * FURNACE.sunIntensity });
        const means: number[] = [];
        for (let frame = 0; frame < LAG_FRAMES; frame++) {
            scenario.frame();
            if (frame < 60 || frame >= LAG_SETTLED_FROM) means.push(nearCeilingStats(await readRays(device, scenario), 2 * CEILING_INDIRECT).mean);
            else means.push(NaN);
        }
        const settledFrames = means.slice(LAG_SETTLED_FROM);
        const settled = settledFrames.reduce((sum, value) => sum + value, 0) / settledFrames.length;
        let frames = LAG_FRAMES;
        for (let frame = 59; frame >= 0; frame--) {
            if (Math.abs(means[frame] / settled - 1) > LAG_WITHIN) break;
            frames = frame + 1;
        }
        return { framesToWithin5Percent: frames, settledOverBefore: Number((settled / before).toFixed(4)), settledOverAnalytic: Number((settled / (2 * CEILING_INDIRECT)).toFixed(4)) };
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

// ---- 7. The shared-cell bounce (pmndrs/gi#54) ------------------------------------------------------------

const SHARED_BUDGET = 20000;
/** Exact samples (a shadowless sun, one light) differ only by the radiance's quantisation and halves. */
const EXACT_TOLERANCE = 0.005;
/** Section 7's bypass scene: cells from 8 m, so the furnace's 3 m segments are all shorter than their cell. */
const BYPASS_MIN_CELL_SIZE = 8;
const BYPASS_FRAMES = 12;

interface LitCellValue {
    /** The lit cell's sample hit: its probe ray's hit point. */
    readonly hit: Vec3;
    readonly direct: [number, number, number, number];
    readonly indirect: [number, number, number, number];
}

/** A cell value (`vec2u`: four halves), its rgb and count. */
function cellValue(words: Uint32Array, element: number): [number, number, number, number] {
    const xy = words[2 * element];
    const zw = words[2 * element + 1];
    return [halfBitsToFloat(xy & 0xffff), halfBitsToFloat(xy >>> 16), halfBitsToFloat(zw & 0xffff), halfBitsToFloat(zw >>> 16)];
}

/** This frame's lit cells (the shade list), each with its sample hit and its finest direct and indirect values. */
async function readLitCells(device: GPUDevice, scenario: Scenario, state: RayState): Promise<{ lit: number; target: number; cells: LitCellValue[] }> {
    const cache = scenario.cache;
    const counters = await readCounters(device, cache);
    const lists = cache.rayLists;
    const cellLists = cache.cellShadingLists;
    if (!lists || !cellLists) throw new Error('no ray lists or cell lists');
    const words = new Uint32Array(await readRange(device, lists.buffer, 0, lists.buffer.size));
    const cellWords = new Uint32Array(await readRange(device, cellLists.buffer, 0, cellLists.buffer.size));
    const values = new Uint32Array(await readRange(device, cache.buffers.cellValues, 0, cache.buffers.cellValues.size));
    const target = counters[HASH_GRID_COUNTER.cellShadedTarget];
    const lit = Math.min(counters[HASH_GRID_COUNTER.cellShadedCount], target);
    const { valueBuffer, valueIndirectBuffer } = cache.layout.cellValues;
    const cells: LitCellValue[] = [];
    for (let slot = 0; slot < lit; slot++) {
        const entry = cellWords[cellLists.layout.lit + slot] & CELL_CANDIDATE_ENTRY_MASK;
        const cell = words[lists.layout.visibilityCellBuffer + entry];
        const hit = state.hits[words[lists.layout.visibilityQueryBuffer + entry]];
        if (!hit) continue;
        cells.push({ hit, direct: cellValue(values, valueBuffer + cell), indirect: cellValue(values, valueIndirectBuffer + cell) });
    }
    return { lit, target, cells };
}

function onCeiling(hit: Vec3): boolean {
    return Math.abs(hit[1] - FURNACE.ceilingY) <= 1e-3;
}

/**
 * 7a. **Cold cache** (the furnace, the first frame from an empty cache): the probes are on the
 * floor, so every lit cell is a ceiling cell, and every destination is cold (no cell has a direct
 * value before ShadeCells), so each lit cell's one continuation lights its landing on the floor
 * with the sun, as ShadeCells would. Under a shadowless sun that sample is exact: every lit cell
 * holds ρ · ρ E / π, one sample.
 */
async function sharedColdCache(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, number>> {
    const mode: SharedMode = { shared: SHARED_BUDGET };
    const scenario = await startScenario(device, `cold ${modeName(mode)}`, 'furnace', [FURNACE_SUN], settingsFor(mode, true));
    try {
        scenario.frame();
        if (scenario.cache.secondBounceInUse !== 'shared-cell') fail(`shared cold cache: the second bounce in use is ${scenario.cache.secondBounceInUse}`);
        const { lit, target, cells } = await readLitCells(device, scenario, await readRays(device, scenario));
        const tally = { lit, target, ceilingCells: 0, ceilingOff: 0, notOneSample: 0, ceilingMean: 0 };
        for (const cell of cells) {
            const [r, g, b, count] = cell.indirect;
            if (count !== 1) tally.notOneSample++;
            if (!onCeiling(cell.hit)) continue;
            const value = (r + g + b) / 3;
            tally.ceilingCells++;
            tally.ceilingMean += value;
            if (Math.abs(value / CEILING_INDIRECT - 1) > EXACT_TOLERANCE) tally.ceilingOff++;
        }
        tally.ceilingMean = Number((tally.ceilingMean / Math.max(tally.ceilingCells, 1) / CEILING_INDIRECT).toFixed(5));
        if (tally.ceilingCells < 1000) fail(`shared cold cache: only ${tally.ceilingCells} lit ceiling cells`);
        if (tally.notOneSample > 0) fail(`shared cold cache: ${tally.notOneSample} lit cells don't hold exactly one indirect sample after the first frame`);
        // A continuation leaving the 400 m floor (about 1 in 5,000 at 3 m) misses: allow 0.5%.
        if (tally.ceilingOff > 0.005 * tally.ceilingCells) fail(`shared cold cache: ${tally.ceilingOff} of ${tally.ceilingCells} lit ceiling cells are not rho^2 E / pi = ${CEILING_INDIRECT.toFixed(4)} within ${EXACT_TOLERANCE * 100}% (mean ${tally.ceilingMean})`);
        return tally;
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

/**
 * 7b. **Indirect miss** (the ceiling only, under a white sky): the lit cells are ceiling cells,
 * whose continuations go down and miss (or land on the black patch). The sky reaches the ceiling
 * through its own light sample (its direct value), so a miss is a sample of no light: every lit
 * cell's indirect value is exactly 0, with a count (a miss is counted, not skipped, which the
 * strips' half in 7d needs too), and its direct value is lit.
 */
async function sharedMiss(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, number>> {
    const mode: SharedMode = { shared: SHARED_BUDGET };
    const scenario = await startScenario(device, `miss ${modeName(mode)}`, 'ceiling only', [], settingsFor(mode, true));
    const tally = { ceilingCells: 0, lit: 0, indirectNotZero: 0, uncounted: 0, directLit: 0 };
    try {
        for (let frame = 1; frame <= 5; frame++) {
            scenario.frame();
            const { lit, cells } = await readLitCells(device, scenario, await readRays(device, scenario));
            tally.lit += lit;
            for (const cell of cells) {
                if (!onCeiling(cell.hit)) continue;
                tally.ceilingCells++;
                const [r, g, b, count] = cell.indirect;
                if (r !== 0 || g !== 0 || b !== 0) tally.indirectNotZero++;
                if (!(count >= 1)) tally.uncounted++;
                if (cell.direct[0] > 0) tally.directLit++;
            }
        }
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
    if (tally.ceilingCells < 1000) fail(`shared miss: only ${tally.ceilingCells} lit ceiling cells`);
    if (tally.indirectNotZero > 0) fail(`shared miss: ${tally.indirectNotZero} of ${tally.ceilingCells} lit ceiling cells have indirect light though their continuations miss (the sky counted twice?)`);
    if (tally.uncounted > 0) fail(`shared miss: ${tally.uncounted} lit ceiling cells have no indirect sample: a miss must count as a sample of no light`);
    // The sky lights the ceiling (its direct value): the scene is lit, so a zero indirect means something.
    if (tally.directLit < 0.9 * tally.ceilingCells) fail(`shared miss: only ${tally.directLit} of ${tally.ceilingCells} lit ceiling cells have direct light from the sky`);
    return tally;
}

/**
 * 7c. **Short-ray bypass** (the furnace with cells from 8 m, at a budget of 0): a probe ray shorter
 * than its cell reads no cell, so ShortRayBounce gives it its own continuation, kept with
 * probability 1 - p and weighted 1 / (1 - p). Its 3 m segment to the floor is shorter than its cell
 * too, so it is lit at the landing (exact, as in 7a). Over the frames the near ceiling rays that
 * bypass average ρ² E / π within 1.5%; with no cell lit, every ray that doesn't bypass reads 0.
 */
async function sharedBypass(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, number>> {
    const mode: SharedMode = { shared: 0, minCellSize: BYPASS_MIN_CELL_SIZE };
    const scenario = await startScenario(device, `bypass ${modeName(mode)}`, 'furnace', [FURNACE_SUN], settingsFor(mode, true));
    const cellSize = hashGridCacheCellSizeConstant(verticalFieldOfView(frameCamera(FURNACE.camera, WIDTH, HEIGHT).projection), WIDTH, HEIGHT, 32);
    const tally = { bypassRays: 0, bypassSum: 0, kept: 0, throughCellRays: 0, throughCellLit: 0, nonFinite: 0 };
    try {
        for (let frame = 1; frame <= BYPASS_FRAMES; frame++) {
            scenario.frame();
            const state = await readRays(device, scenario);
            state.hits.forEach((hit, ray) => {
                if (!hit || !onCeiling(hit)) return;
                const value = (state.radiance[ray * 4] + state.radiance[ray * 4 + 1] + state.radiance[ray * 4 + 2]) / 3;
                if (!Number.isFinite(value)) {
                    tally.nonFinite++;
                    return;
                }
                const length = Math.hypot(hit[0] - state.origins[ray][0], hit[1] - state.origins[ray][1], hit[2] - state.origins[ray][2]);
                const cell = hashGridCacheGetCellSize(hit, FURNACE.camera.position, { cellSize, minCellSize: BYPASS_MIN_CELL_SIZE });
                // Clear of the threshold either way (the GPU's distance can round across it).
                if (length < 0.99 * cell) {
                    tally.bypassRays++;
                    tally.bypassSum += value;
                    if (value > 0) tally.kept++;
                } else if (length > 1.01 * cell) {
                    tally.throughCellRays++;
                    if (value !== 0) tally.throughCellLit++;
                }
            });
        }
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
    const mean = tally.bypassSum / Math.max(tally.bypassRays, 1);
    if (tally.bypassRays < 10000) fail(`shared bypass: only ${tally.bypassRays} ceiling rays shorter than their cell`);
    else if (Math.abs(mean / CEILING_INDIRECT - 1) > 0.015) fail(`shared bypass: the ceiling rays shorter than their cell average ${mean.toFixed(4)}, rho^2 E / pi = ${CEILING_INDIRECT.toFixed(4)} within 1.5% (kept ${tally.kept} of ${tally.bypassRays})`);
    if (tally.throughCellLit > 0) fail(`shared bypass: ${tally.throughCellLit} of ${tally.throughCellRays} ceiling rays through their cell have light at a budget of 0`);
    if (tally.nonFinite > 0) fail(`shared bypass: ${tally.nonFinite} ray radiances are not finite`);
    return { ...tally, bypassMeanOverExpected: Number((mean / CEILING_INDIRECT).toFixed(5)), keptShare: Number((tally.kept / Math.max(tally.bypassRays, 1)).toFixed(4)) };
}

/**
 * 7e. **Zero budget** (the furnace, budget 0, from an empty cache): nothing is lit (target and lit
 * count 0 every frame), no indirect sample reaches a cell, and so every probe ray (none is shorter
 * than its cell here) reads exactly 0, never NaN.
 */
async function sharedZeroBudget(device: GPUDevice, fail: (message: string) => void): Promise<Record<string, number>> {
    const mode: SharedMode = { shared: 0 };
    const scenario = await startScenario(device, `zero ${modeName(mode)}`, 'furnace', [FURNACE_SUN], settingsFor(mode, true));
    const cellSize = hashGridCacheCellSizeConstant(verticalFieldOfView(frameCamera(FURNACE.camera, WIDTH, HEIGHT).projection), WIDTH, HEIGHT, 32);
    const tally = { frames: 0, hitRays: 0, litRays: 0, nonFinite: 0, litCounts: 0, shortRays: 0 };
    try {
        for (let frame = 1; frame <= 20; frame++) {
            scenario.frame();
            const counters = await readCounters(device, scenario.cache);
            if (counters[HASH_GRID_COUNTER.cellShadedTarget] !== 0 || counters[HASH_GRID_COUNTER.cellShadedCount] !== 0) tally.litCounts++;
            const state = await readRays(device, scenario);
            state.hits.forEach((hit, ray) => {
                if (!hit) return;
                // A ray about as short as its cell or shorter may bypass, and has its own bounce (7c).
                const length = Math.hypot(hit[0] - state.origins[ray][0], hit[1] - state.origins[ray][1], hit[2] - state.origins[ray][2]);
                if (length <= 1.01 * hashGridCacheGetCellSize(hit, FURNACE.camera.position, { cellSize, minCellSize: 0.1 })) {
                    tally.shortRays++;
                    return;
                }
                tally.hitRays++;
                const value = state.radiance[ray * 4] + state.radiance[ray * 4 + 1] + state.radiance[ray * 4 + 2];
                if (!Number.isFinite(value)) tally.nonFinite++;
                else if (value !== 0) tally.litRays++;
            });
            tally.frames++;
        }
    } finally {
        scenario.dispose();
        await device.queue.onSubmittedWorkDone();
    }
    if (tally.litCounts > 0) fail(`shared zero budget: ${tally.litCounts} frames lit cells at a budget of 0`);
    if (tally.hitRays === 0) fail('shared zero budget: no ray hit');
    if (tally.litRays > 0) fail(`shared zero budget: ${tally.litRays} of ${tally.hitRays} hit rays have light with no cell lit`);
    if (tally.nonFinite > 0) fail(`shared zero budget: ${tally.nonFinite} ray radiances are not finite`);
    return tally;
}

/** Sections 1 to 6 (lever E8 and experiment E9). */
async function sectionsOneToSix(device: GPUDevice, modes: readonly Mode[], data: Record<string, unknown>, fail: (message: string) => void): Promise<void> {
    const lit: Record<string, unknown> = {};
    for (const mode of modes) lit[modeName(mode)] = await litPlane(device, mode, fail);
    data.litPlane = lit;

    data.queue = await queue(device, fail);

    const furnaces: Record<string, unknown> = {};
    for (const mode of ['per-hit', 'per-hit+E9', 'per-hit+E9 keep 3', 20000] as const) furnaces[modeName(mode)] = await furnace(device, mode, 'furnace', fail);
    furnaces[`strips ${modeName(20000)}`] = await furnace(device, 20000, 'strips', fail);
    data.furnace = furnaces;

    const lags: Record<string, Record<string, Record<string, number>>> = { sun: {}, point: {} };
    for (const light of ['sun', 'point'] as const) for (const mode of modes) lags[light][modeName(mode)] = await lag(device, mode, light);
    data.lag = lags;
    const perHitSunLag = lags.sun['per-hit'].framesToWithin5Percent;
    // E9 (the change reset alone, per hit) follows a single light's change within E9_LAG_FRAMES (dropping the
    // history; keeping 3 samples, faster than per hit without it), and a light switched off as quickly.
    for (const light of ['sun', 'point'] as const) {
        const resetLag = lags[light]['per-hit+E9'].framesToWithin5Percent;
        if (light === 'sun' && resetLag > E9_LAG_FRAMES) fail(`lag: per hit with E9 follows the sun in ${resetLag} frames, more than ${E9_LAG_FRAMES}`);
        if (resetLag >= lags[light]['per-hit'].framesToWithin5Percent) fail(`lag: per hit with E9 follows the ${light} in ${resetLag} frames, per hit without it in ${lags[light]['per-hit'].framesToWithin5Percent}`);
        const keepLag = lags[light]['per-hit+E9 keep 3'].framesToWithin5Percent;
        if (keepLag >= lags[light]['per-hit'].framesToWithin5Percent) fail(`lag: per hit with E9 keeping 3 follows the ${light} in ${keepLag} frames, per hit without it in ${lags[light]['per-hit'].framesToWithin5Percent}`);
    }
    const offLags: Record<string, Record<string, number>> = {};
    for (const mode of ['per-hit', 'per-hit+E9', 'per-hit+E9 keep 3', 20000] as const) offLags[modeName(mode)] = await lagToZero(device, mode);
    data.lagToZero = offLags;
    // E9 follows a light switched off faster than per hit (Capsaicin's step-down, 16 frames), and so
    // does E8 at 20k: its changed cells start over from their next sample, which, with no light to
    // draw, is a sample of 0 (register D63, P6.2's fix 3; a cell marked dark instead, as before the
    // fix, held its tile's coarser levels and never got there).
    for (const mode of ['per-hit+E9', 'per-hit+E9 keep 3', 20000] as const) {
        const name = modeName(mode);
        if (offLags[name].framesToUnder5Percent >= offLags['per-hit'].framesToUnder5Percent) fail(`lag: ${name} follows the sun switched off in ${offLags[name].framesToUnder5Percent} frames, per hit in ${offLags['per-hit'].framesToUnder5Percent}`);
    }
    const bounceLags: Record<string, Record<string, number>> = {};
    for (const mode of ['per-hit', 'per-hit+E9', 'per-hit+E9 keep 3'] as const) bounceLags[modeName(mode)] = await multibounceLag(device, mode);
    data.multibounceLag = bounceLags;
    for (const mode of ['per-hit+E9', 'per-hit+E9 keep 3'] as const) {
        if (bounceLags[mode].framesToWithin5Percent >= bounceLags['per-hit'].framesToWithin5Percent) fail(`lag: the second bounce follows the furnace's sun in ${bounceLags[mode].framesToWithin5Percent} frames with ${mode}, ${bounceLags['per-hit'].framesToWithin5Percent} per hit`);
    }
    for (const budget of [20000, 80000]) {
        const budgetedLag = lags.sun[modeName(budget)].framesToWithin5Percent;
        if (budgetedLag > perHitSunLag) fail(`lag: ${modeName(budget)} follows the sun in ${budgetedLag} frames, per hit in ${perHitSunLag}`);
    }
}

// ---- The page ---------------------------------------------------------------------------------------------

const startTime = performance.now();
/** `VITE_RC_CELL_SHADING_ONLY=shared`: section 7 alone (the shared-cell bounce, #54). */
const onlyShared = (import.meta.env.VITE_RC_CELL_SHADING_ONLY as string | undefined) === 'shared';
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
    const modes: Mode[] = ['per-hit', 'per-hit+E9', 'per-hit+E9 keep 3', ...BUDGETS];
    const data: Record<string, unknown> = { litRadiance: LIT_RADIANCE, ceilingIndirect: CEILING_INDIRECT, stripsIndirect: STRIPS_INDIRECT };

    // 7. The shared-cell bounce (#54), first: VITE_RC_CELL_SHADING_ONLY=shared runs it alone.
    const shared: Record<string, unknown> = {};
    const sharedMode: SharedMode = { shared: SHARED_BUDGET };
    shared.coldCache = await sharedColdCache(device, fail);
    shared.miss = await sharedMiss(device, fail);
    shared.bypass = await sharedBypass(device, fail);
    shared.furnace = await furnace(device, sharedMode, 'furnace', fail);
    shared.strips = await furnace(device, sharedMode, 'strips', fail);
    shared.furnaceLevel = await furnace(device, sharedMode, 'furnace level', fail);
    shared.zeroBudget = await sharedZeroBudget(device, fail);
    shared.multibounceLag = await multibounceLag(device, sharedMode);
    data.sharedCell = shared;
    if (!onlyShared) await sectionsOneToSix(device, modes, data, fail);

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
