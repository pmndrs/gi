/**
 * RC3.2: the radiance cache's world cache (Capsaicin's hash grid) populated
 * and resolved through the technique, on the GPU.
 *
 * The scene is a corner: a 400 m floor (y = 0, facing up) and a 40 × 10 m
 * wall (z = 0, facing +z), both grey (albedo 0.5), under a black sky. The
 * camera at (0, 2, 6) looks at the wall's foot, so probes sit on the floor
 * and the wall: the floor's probe rays hit the wall, the wall's hit the floor,
 * and those near the corner are shorter than their cell. The screen comes
 * from primary rays through the platform's tracer (radiance-cache-probes'
 * `PageScreen`).
 *
 * 1. **A lit plane converges** (sun along (0, -1, -1), intensity 8, after
 *    40 frames from an empty cache): floor and wall both get E = 8 cos 45°,
 *    so the light leaving any hit toward any probe is albedo / π × E. Every
 *    probe ray that hit reads it within 2%, and so does every finest-level
 *    cell with samples (rgb / count). The GPU's bypass flags match the
 *    CPU's "ray shorter than its cell" (rays within 0.1% of a cell's size
 *    are counted, not checked), some rays take each path, and every hit was
 *    inserted, resolved and given a shadow ray (no bucket overflowed); the
 *    update list holds every touched tile once (register D32).
 * 2. **UpdateTiles** (same frame): no finest-level count above 16 and some
 *    at 16; every coarser cell of every live tile is the sum of its 2 × 2
 *    finer cells (rgb and count, within half precision).
 * 3. **ResolveCells** (a point light at (0, 1.5, 2), one frame on an empty
 *    cache, so most cells have fewer than 16 samples): every resolved ray
 *    reads what Capsaicin's filtered readback picks from the cells as read
 *    back (the CPU's `hashGridCacheFilteredRadiance`); some rays read a
 *    coarser level, and some of those differ from their own cell's mean by
 *    more than 5% (so the check can tell the levels apart).
 * 4. **Emissive hits skip the cache** (the sun and an emissive patch on the
 *    floor, emission (3, 2, 1)): every ray that hit the patch's front reads
 *    its emission alone (a back-face hit is lit as any surface, as Capsaicin's); each was inserted (it is on the visibility list) and
 *    none was resolved or given a shadow ray. (Short rays are section 1.)
 * 5. **The light's weight** (two suns, red along (0, -1, -1) and blue along
 *    (0, -1, -0.5), intensity 8 each, 40 frames): one light is drawn per hit
 *    in proportion to its brightness, so each sample lights one channel; the
 *    mean over every hit ray of red and of blue is within 2% of albedo / π ×
 *    the light's E on that surface, and green stays 0.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc3.2 world cache" -- pnpm test:gpu --filter radiance-cache-world-cache --gpu nvidia` (or intel).
 */

import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { requestBudgetDevice } from '../radiance-cache-probes/budget-device.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat, unpackHalf3 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { HASH_GRID_COUNTER, type HashGridCacheCapacity } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import type { HashGridCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cache.ts';
import {
    hashGridCacheCellIndex,
    hashGridCacheCellOffsetMip0,
    hashGridCacheCellSizeConstant,
    hashGridCacheFilteredRadiance,
    hashGridCacheGetCellSize,
    verticalFieldOfView,
    type CellValue,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import { BYPASS_CACHE_BIT, RESERVOIR_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { assert, drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const CAMERA: PageCamera = { position: [0, 2, 6], target: [0, 1, 0], fovYDeg: 60, near: 0.05, far: 1000 };
const FRAME_CAMERA = frameCamera(CAMERA, WIDTH, HEIGHT);

const ALBEDO = 0.5;
const SUN_INTENSITY = 8;
/** The direction the sun's light travels: down and toward the wall, so floor and wall both see it at 45°. */
const SUN_TRAVEL: Vec3 = [0, -1, -1];
/** The second sun of section 5: down and a little toward the wall. */
const BLUE_SUN_TRAVEL: Vec3 = [0, -1, -0.5];
/** albedo / π × E with E = 8 cos 45° on the floor and on the wall. */
const LIT_RADIANCE = (ALBEDO / Math.PI) * SUN_INTENSITY * Math.SQRT1_2;
/** The done-when's tolerance for the converged radiance. */
const TOLERANCE = 0.02;
const CONVERGENCE_FRAMES = 40;
/** `max_sample_count` (gi1.h:63). */
const MAX_SAMPLE_COUNT = 16;
/** Half precision: one step is 2^-10 of the value; the packing rounds to the nearest (half a step). */
const HALF_TOLERANCE = 2 ** -10;

const EMISSION: Vec3 = [3, 2, 1];
const PATCH = { minX: -1, maxX: 1, minZ: 1.5, maxZ: 2.5, y: 0.002 } as const;

/** Cells per tile at each level, and where each level starts (Capsaicin's defaults: 64, 16, 4, 1). */
const LEVEL_SIDES = [8, 4, 2, 1] as const;

type Surface = 'floor' | 'wall' | 'patch' | 'other';

interface ProbeRay {
    readonly origin: Vec3;
    readonly direction: Vec3;
    readonly hitT: number;
    readonly hit: Vec3;
    /** The hit was on the side the mesh was authored as front. */
    readonly frontFace: boolean;
    readonly surface: Surface;
}

interface Scenario {
    readonly platform: GIPlatformImpl;
    readonly technique: RadianceCacheTechnique;
    readonly cache: HashGridCache;
    readonly screen: PageScreen;
    /** Records and submits one frame: the host's encode, the screen, the technique's screen stage. */
    frame(): void;
    dispose(): void;
}

// ---- The scene ---------------------------------------------------------------------------------

function quad(platform: GIPlatformImpl, label: string, corners: readonly Vec3[], normal: Vec3, albedo: number, emissive?: Vec3): void {
    const scene = platform.scene;
    const mesh = scene.createMesh({
        positions: new Float32Array(corners.flat()),
        normals: new Float32Array([...normal, ...normal, ...normal, ...normal]),
        // Counter-clockwise seen from the normal's side: the front face.
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        label,
    });
    const material = scene.createMaterial({ albedo: [albedo, albedo, albedo], emissive, label });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label });
}

function describeCorner(platform: GIPlatformImpl, lights: readonly LightDescriptor[], withPatch: boolean): void {
    const half = 200;
    quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], ALBEDO);
    quad(platform, 'wall', [[-20, 0, 0], [20, 0, 0], [20, 10, 0], [-20, 10, 0]], [0, 0, 1], ALBEDO);
    if (withPatch) {
        const { minX, maxX, minZ, maxZ, y } = PATCH;
        quad(platform, 'emissive patch', [[minX, y, maxZ], [maxX, y, maxZ], [maxX, y, minZ], [minX, y, minZ]], [0, 1, 0], ALBEDO, EMISSION);
    }
    for (const light of lights) platform.scene.createLight(light);
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
}

function sun(travel: Vec3, color: Vec3 = [1, 1, 1]): LightDescriptor {
    return { type: 'directional', direction: travel, color, intensity: SUN_INTENSITY, label: 'sun' };
}

function frameInput(): FrameInput {
    return { deltaTime: 1 / 60, budget: { units: 0 }, camera: FRAME_CAMERA };
}

async function startScenario(device: GPUDevice, name: string, lights: readonly LightDescriptor[], withPatch: boolean): Promise<Scenario> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-world-cache ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    platform.setOutput(technique);
    await platform.prepare();
    describeCorner(platform, lights, withPatch);
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `radiance-cache-world-cache ${name}` });
        platform.encode(encoder, frameInput());
        screen.encode(encoder, CAMERA);
        platform.encodeScreen(encoder, screen.screen);
        device.queue.submit([encoder.finish()]);
    };
    // A frame places the scene; then the mesh trees, and a frame with them.
    frame();
    await platform.rays.whenBuilt();
    frame();
    await device.queue.onSubmittedWorkDone();
    if (platform.rays.building) throw new Error(`${name}: the platform still reports trees building`);
    // Start the checks from an empty cache, with the trees in place.
    technique.resetHistory();
    const cache = technique.worldCache;
    if (!cache) throw new Error(`${name}: the technique has no world cache after prepare`);
    return {
        platform,
        technique,
        cache,
        screen,
        frame,
        dispose() {
            screen.dispose();
            platform.dispose();
        },
    };
}

async function runFrames(device: GPUDevice, scenario: Scenario, frames: number): Promise<void> {
    for (let frame = 0; frame < frames; frame++) scenario.frame();
    await device.queue.onSubmittedWorkDone();
}

// ---- Reading the GPU's state back ---------------------------------------------------------------

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const readback = device.createBuffer({ label: 'world-cache readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'world-cache readback' });
    encoder.copyBufferToBuffer(source, offset, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0, byteLength);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** Half-float pairs to floats: each `u32` holds two halves, the first in its low bits. */
function unpackHalves(words: Uint32Array): Float32Array {
    const out = new Float32Array(words.length * 2);
    for (let index = 0; index < words.length; index++) {
        out[index * 2] = halfBitsToFloat(words[index] & 0xffff);
        out[index * 2 + 1] = halfBitsToFloat(words[index] >>> 16);
    }
    return out;
}

interface FrameState {
    readonly rays: ProbeRay[];
    /** Per ray: rgb radiance and the hit distance, as halves. */
    readonly radiance: Float32Array;
    readonly counters: Uint32Array;
    readonly visibilityCell: Uint32Array;
    readonly visibilityQuery: Uint32Array;
    readonly visibilityRay: Uint32Array;
    readonly resolve: Uint32Array;
    readonly reservoirs: Uint32Array;
    readonly updateTiles: Uint32Array;
    readonly packedTiles: Uint32Array;
}

function classify(hit: Vec3): Surface {
    if (hit[1] > 0.001 && hit[1] < 0.004 && hit[0] >= PATCH.minX - 1e-3 && hit[0] <= PATCH.maxX + 1e-3 && hit[2] >= PATCH.minZ - 1e-3 && hit[2] <= PATCH.maxZ + 1e-3) return 'patch';
    if (Math.abs(hit[1]) < 1e-3) return 'floor';
    if (Math.abs(hit[2]) < 1e-3) return 'wall';
    return 'other';
}

async function readFrameState(device: GPUDevice, scenario: Scenario): Promise<FrameState> {
    const probes = scenario.technique.probes;
    const lists = scenario.cache.rayLists;
    assert(probes && lists, 'no probes or ray lists after encodeScreen');
    const rayCount = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
    const rayBytes = WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + rayCount * WORLD_CACHE_PROBE_RAY_BYTES;
    const rayWords = new Uint32Array(await readRange(device, probes.probeRays, 0, rayBytes));
    const rayFloats = new Float32Array(rayWords.buffer);
    const rays: ProbeRay[] = [];
    for (let ray = 0; ray < rayCount; ray++) {
        const base = (WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES) / 4;
        const origin: Vec3 = [rayFloats[base], rayFloats[base + 1], rayFloats[base + 2]];
        const direction = unpackHalf3([rayWords[base + 4], rayWords[base + 5]]);
        const hitT = rayFloats[base + 6];
        const hit: Vec3 = [origin[0] + hitT * direction[0], origin[1] + hitT * direction[1], origin[2] + hitT * direction[2]];
        rays.push({ origin, direction, hitT, hit, frontFace: rayWords[base + 9] !== 0, surface: hitT >= 0 ? classify(hit) : 'other' });
    }
    const results = probes.rayScratch.rayResults;
    const radiance = unpackHalves(new Uint32Array(await readRange(device, results.buffer, results.offset ?? 0, rayCount * 8)));
    const counters = new Uint32Array(await readRange(device, scenario.cache.buffers.words, 0, 64 * 4));
    const words = new Uint32Array(await readRange(device, lists.buffer, 0, lists.buffer.size));
    const layout = lists.layout;
    const count0 = counters[HASH_GRID_COUNTER.visibilityCount0];
    const rayListCount = counters[HASH_GRID_COUNTER.visibilityRayCount];
    const resolveCount = counters[HASH_GRID_COUNTER.resolveCount];
    const packed = scenario.cache.packedTiles;
    const packedCount = new Uint32Array(await readRange(device, packed.count.buffer, packed.count.offset, 4))[0];
    const packedTiles = new Uint32Array(await readRange(device, scenario.cache.buffers.words, packed.firstWord * 4, packedCount * 4));
    return {
        rays,
        radiance,
        counters,
        visibilityCell: words.slice(layout.visibilityCellBuffer, layout.visibilityCellBuffer + count0),
        visibilityQuery: words.slice(layout.visibilityQueryBuffer, layout.visibilityQueryBuffer + count0),
        visibilityRay: words.slice(layout.visibilityRayBuffer, layout.visibilityRayBuffer + rayListCount),
        resolve: words.slice(layout.resolveBuffer, layout.resolveBuffer + resolveCount),
        reservoirs: words.slice(layout.reservoirBuffer, layout.reservoirBuffer + RESERVOIR_WORDS * rayListCount),
        updateTiles: words.slice(layout.updateTileBuffer, layout.updateTileBuffer + counters[HASH_GRID_COUNTER.updateTileCount]),
        packedTiles,
    };
}

/** The direct values of whole tiles (all four levels), unpacked: `numCellsPerTile` × 4 floats per tile, in `tiles` order. */
async function readTiles(device: GPUDevice, cache: HashGridCache, tiles: readonly number[]): Promise<Float32Array> {
    const capacity = cache.capacity;
    const tileBytes = capacity.numCellsPerTile * 8;
    const staging = device.createBuffer({ label: 'world-cache tiles', size: Math.max(4, tiles.length * tileBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'world-cache tiles' });
    const valueBase = cache.layout.cellValues.valueBuffer * 8;
    tiles.forEach((tile, index) => encoder.copyBufferToBuffer(cache.buffers.cellValues, valueBase + tile * tileBytes, staging, index * tileBytes, tileBytes));
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readRange(device, staging, 0, tiles.length * tileBytes));
    staging.destroy();
    return unpackHalves(words);
}

/** One cell's value in `readTiles`' output: tile number `slot` (its place in the list), cell `cellInTile`. */
function cellValue(values: Float32Array, capacity: HashGridCacheCapacity, slot: number, cellInTile: number): CellValue {
    const base = (slot * capacity.numCellsPerTile + cellInTile) * 4;
    return [values[base], values[base + 1], values[base + 2], values[base + 3]];
}

function relativeError(value: number, expected: number): number {
    return Math.abs(value - expected) / Math.abs(expected);
}

// ---- Section 1 and 2: the lit plane and UpdateTiles ----------------------------------------------

interface Checks {
    readonly failures: string[];
    readonly data: Record<string, unknown>;
}

function cellSizeConstants(): { cellSize: number; minCellSize: number } {
    return { cellSize: hashGridCacheCellSizeConstant(verticalFieldOfView(FRAME_CAMERA.projection), WIDTH, HEIGHT, 32), minCellSize: 0.1 };
}

async function checkLitPlane(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`lit plane: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const hits = state.rays.filter((ray) => ray.hitT >= 0);
    const counters = state.counters;

    // Every hit inserted, resolved and lit (the sun reaches both surfaces), nothing refused.
    const overflow = counters[HASH_GRID_COUNTER.bucketOverflowCount];
    if (overflow !== 0) fail(`${overflow} inserts refused (bucket overflow)`);
    if (counters[HASH_GRID_COUNTER.visibilityCount0] !== hits.length) fail(`visibility list holds ${counters[HASH_GRID_COUNTER.visibilityCount0]} hits, the rays ${hits.length}`);
    if (state.resolve.length !== hits.length) fail(`resolve list holds ${state.resolve.length}, hits ${hits.length}`);
    if (state.visibilityRay.length !== hits.length) fail(`${state.visibilityRay.length} shadow rays for ${hits.length} hits`);
    // The update list: each tile touched this frame, once (register D32).
    const touchedTiles = new Set(Array.from(state.visibilityCell, (cell) => Math.floor(cell / scenario.cache.capacity.numCellsPerTile)));
    const listedTiles = new Set(state.updateTiles);
    if (listedTiles.size !== state.updateTiles.length) fail(`the update list holds ${state.updateTiles.length} entries for ${listedTiles.size} tiles`);
    if (listedTiles.size !== touchedTiles.size || [...touchedTiles].some((tile) => !listedTiles.has(tile))) fail(`the update list holds ${listedTiles.size} tiles, the hits touched ${touchedTiles.size}`);
    const others = hits.filter((ray) => ray.surface !== 'floor' && ray.surface !== 'wall').length;
    if (others > 0) fail(`${others} hits on neither the floor nor the wall`);

    // Each hit ray reads albedo / π × E.
    let worstRay = 0;
    let raysOff = 0;
    state.rays.forEach((ray, index) => {
        if (ray.hitT < 0) return;
        const error = Math.max(...[0, 1, 2].map((channel) => relativeError(state.radiance[index * 4 + channel], LIT_RADIANCE)));
        worstRay = Math.max(worstRay, error);
        if (error > TOLERANCE) {
            raysOff++;
            if (raysOff <= 3) fail(`ray ${index} (${ray.surface}, ${ray.hitT.toFixed(3)} m) reads ${[0, 1, 2].map((channel) => state.radiance[index * 4 + channel].toFixed(4)).join(', ')}, expected ${LIT_RADIANCE.toFixed(4)}`);
        }
    });
    if (raysOff > 0) fail(`${raysOff} of ${hits.length} hit rays are more than 2% off`);

    // The bypass: the GPU's flag on each resolve entry against the CPU's "shorter than its cell".
    const constants = cellSizeConstants();
    let bypassed = 0;
    let resolved = 0;
    let undecided = 0;
    let flagMismatches = 0;
    for (const entry of state.resolve) {
        const visibility = entry & ~BYPASS_CACHE_BIT;
        const ray = state.rays[state.visibilityQuery[visibility]];
        const gpuBypass = (entry & BYPASS_CACHE_BIT) !== 0;
        const cellSize = hashGridCacheGetCellSize(ray.hit, CAMERA.position, constants);
        const rayLength = Math.hypot(ray.hit[0] - ray.origin[0], ray.hit[1] - ray.origin[1], ray.hit[2] - ray.origin[2]);
        if (gpuBypass) bypassed++;
        else resolved++;
        if (Math.abs(rayLength - cellSize) < 1e-3 * cellSize) {
            undecided++;
            continue;
        }
        if (gpuBypass !== rayLength < cellSize) {
            flagMismatches++;
            if (flagMismatches <= 3) fail(`ray ${state.visibilityQuery[visibility]}: ${rayLength.toFixed(4)} m long in a ${cellSize} m cell, GPU bypass ${gpuBypass}`);
        }
    }
    if (bypassed === 0 || resolved === 0) fail(`the rays should take both paths: ${bypassed} bypassed, ${resolved} resolved`);
    // A shadow ray's bypass flag is its resolve entry's.
    const resolveFlag = new Map<number, boolean>();
    for (const entry of state.resolve) resolveFlag.set(entry & ~BYPASS_CACHE_BIT, (entry & BYPASS_CACHE_BIT) !== 0);
    const shadowFlagMismatches = Array.from(state.visibilityRay).filter((entry) => resolveFlag.get(entry & ~BYPASS_CACHE_BIT) !== ((entry & BYPASS_CACHE_BIT) !== 0)).length;
    if (shadowFlagMismatches > 0) fail(`${shadowFlagMismatches} shadow rays carry another bypass flag than their resolve entry`);

    // Section 2 and the cells: every live tile, all four levels.
    const capacity = scenario.cache.capacity;
    const tiles = Array.from(state.packedTiles);
    const values = await readTiles(device, scenario.cache, tiles);
    let cellsWithSamples = 0;
    let worstCell = 0;
    let cellsOff = 0;
    let largestCount = 0;
    let countsAtCap = 0;
    let levelChecks = 0;
    let levelMismatches = 0;
    let worstLevelError = 0;
    tiles.forEach((tile, slot) => {
        for (let cell = 0; cell < capacity.numCellsPerTileMip[0]; cell++) {
            const value = cellValue(values, capacity, slot, cell);
            largestCount = Math.max(largestCount, value[3]);
            if (value[3] === MAX_SAMPLE_COUNT) countsAtCap++;
            if (!(value[3] > 0)) continue;
            cellsWithSamples++;
            const error = Math.max(...[0, 1, 2].map((channel) => relativeError(value[channel] / value[3], LIT_RADIANCE)));
            worstCell = Math.max(worstCell, error);
            if (error > TOLERANCE) cellsOff++;
        }
        // Each coarser cell against the sum of its 2 x 2 finer cells.
        for (let level = 1; level < LEVEL_SIDES.length; level++) {
            const side = LEVEL_SIDES[level];
            for (let y = 0; y < side; y++) {
                for (let x = 0; x < side; x++) {
                    const offset = (cellX: number, cellY: number, cellLevel: number): number =>
                        hashGridCacheCellIndex([cellX << cellLevel, cellY << cellLevel], tile, cellLevel, capacity) - tile * capacity.numCellsPerTile;
                    const coarse = cellValue(values, capacity, slot, offset(x, y, level));
                    const sum = [0, 0, 0, 0];
                    for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
                        const fine = cellValue(values, capacity, slot, offset(2 * x + dx, 2 * y + dy, level - 1));
                        for (let channel = 0; channel < 4; channel++) sum[channel] += fine[channel];
                    }
                    levelChecks++;
                    const error = Math.max(...[0, 1, 2, 3].map((channel) => Math.abs(coarse[channel] - sum[channel]) / Math.max(Math.abs(sum[channel]), 1e-3)));
                    worstLevelError = Math.max(worstLevelError, error);
                    if (error > HALF_TOLERANCE) {
                        levelMismatches++;
                        if (levelMismatches <= 3) fail(`tile ${tile} level ${level} cell (${x}, ${y}) holds ${coarse.map((v) => v.toFixed(4)).join(', ')}, its 2x2 sum ${sum.map((v) => v.toFixed(4)).join(', ')}`);
                    }
                }
            }
        }
    });
    if (cellsOff > 0) fail(`${cellsOff} of ${cellsWithSamples} cells are more than 2% off albedo / pi x E`);
    if (largestCount > MAX_SAMPLE_COUNT) fail(`a finest-level cell counts ${largestCount} samples, above the cap of ${MAX_SAMPLE_COUNT}`);
    if (countsAtCap === 0) fail(`no cell reached the cap of ${MAX_SAMPLE_COUNT} samples after ${CONVERGENCE_FRAMES} frames`);
    if (levelMismatches > 0) fail(`${levelMismatches} of ${levelChecks} coarser cells aren't the sum of their 2x2 cells`);

    return {
        failures,
        data: {
            expectedRadiance: LIT_RADIANCE,
            rays: state.rays.length,
            hits: hits.length,
            worstRayError: worstRay,
            bypassed,
            resolved,
            undecidedBypass: undecided,
            bypassFlagMismatches: flagMismatches,
            bucketOverflow: overflow,
            updateListEntries: state.updateTiles.length,
            updateListTiles: listedTiles.size,
            liveTiles: tiles.length,
            cellsWithSamples,
            worstCellError: worstCell,
            largestCount,
            cellsAtCap: countsAtCap,
            coarserCellsChecked: levelChecks,
            coarserCellMismatches: levelMismatches,
            worstCoarserCellError: worstLevelError,
        },
    };
}

// ---- Section 3: ResolveCells ----------------------------------------------------------------------

async function checkResolve(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`resolve: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const capacity = scenario.cache.capacity;
    const entries = Array.from(state.resolve).filter((entry) => (entry & BYPASS_CACHE_BIT) === 0);
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);
    const tiles = [...new Set(entries.map((entry) => tileOf(state.visibilityCell[entry])))];
    const slotOfTile = new Map(tiles.map((tile, slot) => [tile, slot]));
    const values = await readTiles(device, scenario.cache, tiles);

    const levelCounts = [0, 0, 0, 0];
    let mismatches = 0;
    let worst = 0;
    let coarserAndDifferent = 0;
    for (const entry of entries) {
        const cell = state.visibilityCell[entry];
        const query = state.visibilityQuery[entry];
        const { tileIndex, cellOffset } = hashGridCacheCellOffsetMip0(cell, capacity);
        const slot = slotOfTile.get(tileIndex) ?? 0;
        const levels = [0, 1, 2, 3].map((level) =>
            capacity.sizeTileMip[level] > 0 ? cellValue(values, capacity, slot, hashGridCacheCellIndex(cellOffset, tileIndex, level, capacity) - tileIndex * capacity.numCellsPerTile) : null,
        );
        const picked = hashGridCacheFilteredRadiance(levels, MAX_SAMPLE_COUNT);
        levelCounts[picked.level]++;
        const expected = [0, 1, 2].map((channel) => picked.value[channel] / Math.max(picked.value[3], 1));
        const got = [0, 1, 2].map((channel) => state.radiance[query * 4 + channel]);
        const error = Math.max(...[0, 1, 2].map((channel) => Math.abs(got[channel] - expected[channel]) / Math.max(Math.abs(expected[channel]), 1e-4)));
        worst = Math.max(worst, error);
        if (error > 2 * HALF_TOLERANCE) {
            mismatches++;
            if (mismatches <= 3) fail(`ray ${query} reads ${got.map((v) => v.toFixed(5)).join(', ')}, the level-${picked.level} pick gives ${expected.map((v) => v.toFixed(5)).join(', ')}`);
        }
        const own = levels[0];
        if (picked.level > 0 && own && own[3] > 0) {
            const ownMean = own[0] / own[3];
            if (Math.abs(ownMean - expected[0]) > 0.05 * Math.max(ownMean, expected[0])) coarserAndDifferent++;
        }
    }
    const coarser = levelCounts[1] + levelCounts[2] + levelCounts[3];
    if (entries.length === 0) fail('no resolved rays');
    if (mismatches > 0) fail(`${mismatches} of ${entries.length} resolved rays don't read Capsaicin's pick`);
    if (coarser === 0) fail('no ray read a coarser level: the check can\'t see the pick');
    if (coarserAndDifferent === 0) fail('no coarser pick differs from its own cell by 5%: the check can\'t tell the levels apart');
    return { failures, data: { resolvedRays: entries.length, tiles: tiles.length, picksByLevel: levelCounts, coarserPicksThatDiffer: coarserAndDifferent, worstError: worst } };
}

// ---- Section 4: emissive hits --------------------------------------------------------------------

async function checkEmissive(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`emissive: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    // Capsaicin bypasses the cache on an emissive front face only (gi1.comp:2444): the floor's
    // probes beside the patch can catch its edge from below, and those hits are lit as any other.
    const patchRays = new Set<number>();
    let patchBackFaces = 0;
    state.rays.forEach((ray, index) => {
        if (ray.hitT < 0 || ray.surface !== 'patch') return;
        if (ray.frontFace) patchRays.add(index);
        else patchBackFaces++;
    });
    let worst = 0;
    for (const index of patchRays) {
        const error = Math.max(...[0, 1, 2].map((channel) => relativeError(state.radiance[index * 4 + channel], EMISSION[channel])));
        worst = Math.max(worst, error);
    }
    if (patchRays.size === 0) fail('no probe ray hit the emissive patch');
    if (worst > 2 * HALF_TOLERANCE) fail(`a ray on the patch reads ${(worst * 100).toFixed(3)}% off its emission alone`);
    // Inserted (on the visibility list), but neither resolved nor given a shadow ray.
    const visibilityOfPatch = new Set<number>();
    state.visibilityQuery.forEach((query, visibility) => {
        if (patchRays.has(query)) visibilityOfPatch.add(visibility);
    });
    if (visibilityOfPatch.size !== patchRays.size) fail(`${visibilityOfPatch.size} of ${patchRays.size} patch hits are on the visibility list`);
    const resolvedPatch = Array.from(state.resolve).filter((entry) => visibilityOfPatch.has(entry & ~BYPASS_CACHE_BIT)).length;
    const shadowPatch = Array.from(state.visibilityRay).filter((entry) => visibilityOfPatch.has(entry & ~BYPASS_CACHE_BIT)).length;
    if (resolvedPatch > 0) fail(`${resolvedPatch} patch hits were resolved from the cache`);
    if (shadowPatch > 0) fail(`${shadowPatch} patch hits got a shadow ray`);
    return { failures, data: { patchRays: patchRays.size, patchBackFaces, worstEmissionError: worst, inserted: visibilityOfPatch.size, resolved: resolvedPatch, shadowRays: shadowPatch } };
}

// ---- Section 5: two coloured suns -----------------------------------------------------------------

function cosineTo(travel: Vec3, normal: Vec3): number {
    const length = Math.hypot(...travel);
    return Math.max(0, -(travel[0] * normal[0] + travel[1] * normal[1] + travel[2] * normal[2]) / length);
}

async function checkTwoSuns(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`two suns: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const sums = { red: 0, redExpected: 0, green: 0, blue: 0, blueExpected: 0 };
    let hits = 0;
    state.rays.forEach((ray, index) => {
        if (ray.hitT < 0 || (ray.surface !== 'floor' && ray.surface !== 'wall')) return;
        hits++;
        const normal: Vec3 = ray.surface === 'floor' ? [0, 1, 0] : [0, 0, 1];
        sums.red += state.radiance[index * 4];
        sums.green += state.radiance[index * 4 + 1];
        sums.blue += state.radiance[index * 4 + 2];
        sums.redExpected += (ALBEDO / Math.PI) * SUN_INTENSITY * cosineTo(SUN_TRAVEL, normal);
        sums.blueExpected += (ALBEDO / Math.PI) * SUN_INTENSITY * cosineTo(BLUE_SUN_TRAVEL, normal);
    });
    const redRatio = sums.red / sums.redExpected;
    const blueRatio = sums.blue / sums.blueExpected;
    const greenMean = sums.green / Math.max(hits, 1);
    if (hits === 0) fail('no hits');
    if (Math.abs(redRatio - 1) > TOLERANCE) fail(`the mean red is ${redRatio.toFixed(4)} x albedo / pi x E`);
    if (Math.abs(blueRatio - 1) > TOLERANCE) fail(`the mean blue is ${blueRatio.toFixed(4)} x albedo / pi x E`);
    if (greenMean > 1e-3) fail(`green reads ${greenMean} on average; neither sun has any`);
    return { failures, data: { hits, meanRedOverExpected: redRatio, meanBlueOverExpected: blueRatio, meanGreen: greenMean } };
}

// ---- The page --------------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    // As a host would ask (requestGIDevice): the adapter's storage limits, and its optional features;
    // but the platform's 8 storage buffers per stage, whatever the adapter allows (budget-device.ts).
    const { device, adapterStorageBuffersPerStage } = await requestBudgetDevice(adapter);
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);

    const failures: string[] = [];
    const data: Record<string, unknown> = { adapterStorageBuffersPerStage };
    const run = async (name: string, lights: readonly LightDescriptor[], withPatch: boolean, frames: number, check: (device: GPUDevice, scenario: Scenario) => Promise<Checks>): Promise<void> => {
        const scenario = await startScenario(device, name, lights, withPatch);
        try {
            await runFrames(device, scenario, frames);
            const checks = await check(device, scenario);
            failures.push(...checks.failures);
            data[name] = checks.data;
        } finally {
            scenario.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    };

    await run('litPlane', [sun(SUN_TRAVEL)], false, CONVERGENCE_FRAMES, checkLitPlane);
    await run('resolve', [{ type: 'point', position: [0, 1.5, 2], range: 0, color: [1, 1, 1], intensity: 10, label: 'point' }], false, 1, checkResolve);
    await run('emissive', [sun(SUN_TRAVEL)], true, 1, checkEmissive);
    await run('twoSuns', [sun(SUN_TRAVEL, [1, 0, 0]), sun(BLUE_SUN_TRAVEL, [0, 0, 1])], false, CONVERGENCE_FRAMES, checkTwoSuns);

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
