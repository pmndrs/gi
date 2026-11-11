/**
 * The sunlit corner the world-cache pages share (RC3.2's
 * `radiance-cache-world-cache`, RC3.3's `radiance-cache-multibounce`): a
 * 400 m floor (y = 0, facing up) and a 40 x 10 m wall (z = 0, facing +z),
 * both grey (albedo 0.5), under a black sky, seen from (0, 2, 6) at
 * 1280 x 720 through primary rays (radiance-cache-probes' `PageScreen`); and
 * the readbacks of the technique's probe rays, the hash grid's counters,
 * per-ray lists and cell values. RC6.1 adds a constant sky for the corner,
 * and a closed box around the camera (`ScenarioWorld`).
 */
import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { halfBitsToFloat, unpackHalf3 } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { HASH_GRID_COUNTER, type HashGridCacheCapacity } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import type { HashGridCache } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cache.ts';
import { type CellValue } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import { MULTIBOUNCE_INFO_WORDS, RESERVOIR_WORDS, VISIBILITY_WORD, VISIBILITY_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { assert } from '../../client.ts';
import { frameCamera, PageScreen, type PageCamera } from '../radiance-cache-probes/screen.ts';

export const WIDTH = 1280;
export const HEIGHT = 720;
export const CAMERA: PageCamera = { position: [0, 2, 6], target: [0, 1, 0], fovYDeg: 60, near: 0.05, far: 1000 };
export const FRAME_CAMERA = frameCamera(CAMERA, WIDTH, HEIGHT);

export const ALBEDO = 0.5;
export const SUN_INTENSITY = 8;
/** The direction the sun's light travels: down and toward the wall, so floor and wall both see it at 45°. */
export const SUN_TRAVEL: Vec3 = [0, -1, -1];
/** The second sun of section 5: down and a little toward the wall. */
export const BLUE_SUN_TRAVEL: Vec3 = [0, -1, -0.5];
/** albedo / π × E with E = 8 cos 45° on the floor and on the wall. */
export const LIT_RADIANCE = (ALBEDO / Math.PI) * SUN_INTENSITY * Math.SQRT1_2;
/** The done-when's tolerance for the converged radiance. */
export const TOLERANCE = 0.02;
export const CONVERGENCE_FRAMES = 40;
/** `max_sample_count` (gi1.h:63). */
export const MAX_SAMPLE_COUNT = 16;
/** Half precision: one step is 2^-10 of the value; the packing rounds to the nearest (half a step). */
export const HALF_TOLERANCE = 2 ** -10;

export const EMISSION: Vec3 = [3, 2, 1];
export const PATCH = { minX: -1, maxX: 1, minZ: 1.5, maxZ: 2.5, y: 0.002 } as const;

/** Cells per tile at each level, and where each level starts (Capsaicin's defaults: 64, 16, 4, 1). */
export const LEVEL_SIDES = [8, 4, 2, 1] as const;

export type Surface = 'floor' | 'wall' | 'patch' | 'other';

export interface ProbeRay {
    readonly origin: Vec3;
    readonly direction: Vec3;
    readonly hitT: number;
    readonly hit: Vec3;
    /** The hit was on the side the mesh was authored as front. */
    readonly frontFace: boolean;
    readonly surface: Surface;
}

export interface Scenario {
    readonly platform: GIPlatformImpl;
    readonly technique: RadianceCacheTechnique;
    readonly cache: HashGridCache;
    readonly screen: PageScreen;
    /** Records and submits one frame: the host's encode, the screen, the technique's screen stage. */
    frame(): void;
    dispose(): void;
}

// ---- The scene ---------------------------------------------------------------------------------

export function quad(platform: GIPlatformImpl, label: string, corners: readonly Vec3[], normal: Vec3, albedo: number, emissive?: Vec3): void {
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

/** The floor's half side and the wall's corners, for the CPU's view of the corner (RC6.1's sky check). */
export const FLOOR_HALF_SIDE = 200;
export const WALL = { minX: -20, maxX: 20, height: 10 } as const;

/** What a scenario's scene holds besides its lights: the corner (default) or the closed box, and the sky. */
export interface ScenarioWorld {
    /** The constant sky's radiance (RC6.1); black by default. */
    readonly sky?: Vec3;
    /** The closed box around the camera instead of the corner (RC6.1). */
    readonly closedBox?: boolean;
}

/** The closed box (RC6.1): 8 x 4 x 10 m around the camera at (0, 2, 6), every side grey and facing in. */
export const CLOSED_BOX = { minX: -4, maxX: 4, minY: 0, maxY: 4, minZ: -1, maxZ: 9 } as const;

export function describeCorner(platform: GIPlatformImpl, lights: readonly LightDescriptor[], withPatch: boolean, world: ScenarioWorld = {}): void {
    if (world.closedBox) {
        describeClosedBox(platform);
    } else {
        const half = FLOOR_HALF_SIDE;
        quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], ALBEDO);
        quad(platform, 'wall', [[WALL.minX, 0, 0], [WALL.maxX, 0, 0], [WALL.maxX, WALL.height, 0], [WALL.minX, WALL.height, 0]], [0, 0, 1], ALBEDO);
    }
    if (withPatch) {
        const { minX, maxX, minZ, maxZ, y } = PATCH;
        quad(platform, 'emissive patch', [[minX, y, maxZ], [maxX, y, maxZ], [maxX, y, minZ], [minX, y, minZ]], [0, 1, 0], ALBEDO, EMISSION);
    }
    for (const light of lights) platform.scene.createLight(light);
    platform.scene.setEnvironment({ type: 'color', color: [...(world.sky ?? [0, 0, 0])], intensity: 1 });
}

/** Six quads sharing their corners exactly, so the box has no gap a shadow ray could leave through. */
function describeClosedBox(platform: GIPlatformImpl): void {
    const { minX, maxX, minY, maxY, minZ, maxZ } = CLOSED_BOX;
    const sides: { label: string; corners: Vec3[]; normal: Vec3 }[] = [
        { label: 'box floor', corners: [[minX, minY, minZ], [minX, minY, maxZ], [maxX, minY, maxZ], [maxX, minY, minZ]], normal: [0, 1, 0] },
        { label: 'box ceiling', corners: [[minX, maxY, minZ], [maxX, maxY, minZ], [maxX, maxY, maxZ], [minX, maxY, maxZ]], normal: [0, -1, 0] },
        { label: 'box back', corners: [[minX, minY, minZ], [maxX, minY, minZ], [maxX, maxY, minZ], [minX, maxY, minZ]], normal: [0, 0, 1] },
        { label: 'box front', corners: [[minX, minY, maxZ], [minX, maxY, maxZ], [maxX, maxY, maxZ], [maxX, minY, maxZ]], normal: [0, 0, -1] },
        { label: 'box left', corners: [[minX, minY, minZ], [minX, maxY, minZ], [minX, maxY, maxZ], [minX, minY, maxZ]], normal: [1, 0, 0] },
        { label: 'box right', corners: [[maxX, minY, minZ], [maxX, minY, maxZ], [maxX, maxY, maxZ], [maxX, maxY, minZ]], normal: [-1, 0, 0] },
    ];
    for (const side of sides) {
        // quad() makes the corners' winding the front face: check it faces into the box.
        const [a, b, c] = side.corners;
        const edge1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        const edge2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const cross = [edge1[1] * edge2[2] - edge1[2] * edge2[1], edge1[2] * edge2[0] - edge1[0] * edge2[2], edge1[0] * edge2[1] - edge1[1] * edge2[0]];
        const facing = cross[0] * side.normal[0] + cross[1] * side.normal[1] + cross[2] * side.normal[2];
        assert(facing > 0, `${side.label}: its corners wind away from its normal`);
        quad(platform, side.label, side.corners, side.normal, ALBEDO);
    }
}

export function sun(travel: Vec3, color: Vec3 = [1, 1, 1]): LightDescriptor {
    return { type: 'directional', direction: travel, color, intensity: SUN_INTENSITY, label: 'sun' };
}

export function frameInput(): FrameInput {
    return { deltaTime: 1 / 60, budget: { units: 0 }, camera: FRAME_CAMERA };
}

export async function startScenario(
    device: GPUDevice,
    name: string,
    lights: readonly LightDescriptor[],
    withPatch: boolean,
    settings: Partial<RadianceCacheSettings> = {},
    world: ScenarioWorld = {},
): Promise<Scenario> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-world-cache ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    describeCorner(platform, lights, withPatch, world);
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

export async function runFrames(device: GPUDevice, scenario: Scenario, frames: number): Promise<void> {
    for (let frame = 0; frame < frames; frame++) scenario.frame();
    await device.queue.onSubmittedWorkDone();
}

// ---- Reading the GPU's state back ---------------------------------------------------------------

export async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
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
export function unpackHalves(words: Uint32Array): Float32Array {
    const out = new Float32Array(words.length * 2);
    for (let index = 0; index < words.length; index++) {
        out[index * 2] = halfBitsToFloat(words[index] & 0xffff);
        out[index * 2 + 1] = halfBitsToFloat(words[index] >>> 16);
    }
    return out;
}

export interface FrameState {
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
    /** Per first-bounce entry: 1 when its probe ray bypasses the cache (register D40). */
    readonly visibilityBypass: Uint32Array;
    /** The second-bounce half of the visibility lists (RC3.3), entry `i` at visibility index `count0 + i`. */
    readonly secondBounce: SecondBounceEntries;
}

/** The second-bounce entries `PopulateMultibounceCells` wrote this frame. */
export interface SecondBounceEntries {
    readonly count: number;
    /** Each entry's cell. */
    readonly cell: Uint32Array;
    /** Each entry's query: the first-bounce entry it was traced from. */
    readonly query: Uint32Array;
    /** Each entry's ray origin, direction (from halves) and hit distance. */
    readonly origin: Vec3[];
    readonly direction: Vec3[];
    readonly hitT: Float32Array;
    /** Each entry's brdf (rgb) and pdf, from halves. */
    readonly info: Float32Array;
}

export function classify(hit: Vec3): Surface {
    if (hit[1] > 0.001 && hit[1] < 0.004 && hit[0] >= PATCH.minX - 1e-3 && hit[0] <= PATCH.maxX + 1e-3 && hit[2] >= PATCH.minZ - 1e-3 && hit[2] <= PATCH.maxZ + 1e-3) return 'patch';
    if (Math.abs(hit[1]) < 1e-3) return 'floor';
    if (Math.abs(hit[2]) < 1e-3) return 'wall';
    return 'other';
}

export async function readFrameState(device: GPUDevice, scenario: Scenario): Promise<FrameState> {
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
    const count1 = counters[HASH_GRID_COUNTER.visibilityCount1];
    const floats = new Float32Array(words.buffer);
    const origin: Vec3[] = [];
    const direction: Vec3[] = [];
    const hitT = new Float32Array(count1);
    for (let entry = 0; entry < count1; entry++) {
        const base = layout.visibilityBuffer + VISIBILITY_WORDS * (count0 + entry);
        origin.push([floats[base + VISIBILITY_WORD.origin], floats[base + VISIBILITY_WORD.origin + 1], floats[base + VISIBILITY_WORD.origin + 2]]);
        direction.push(unpackHalf3([words[base + VISIBILITY_WORD.direction], words[base + VISIBILITY_WORD.direction + 1]]));
        hitT[entry] = floats[base + VISIBILITY_WORD.hitDistance];
    }
    const secondBounce: SecondBounceEntries = {
        count: count1,
        cell: words.slice(layout.visibilityCellBuffer + count0, layout.visibilityCellBuffer + count0 + count1),
        query: words.slice(layout.visibilityQueryBuffer + count0, layout.visibilityQueryBuffer + count0 + count1),
        origin,
        direction,
        hitT,
        info: unpackHalves(words.slice(layout.multibounceInfoBuffer, layout.multibounceInfoBuffer + MULTIBOUNCE_INFO_WORDS * count1)),
    };
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
        visibilityBypass: words.slice(layout.visibilityBypassBuffer, layout.visibilityBypassBuffer + count0),
        secondBounce,
    };
}

/** Each cell's indirect update scratch (`UpdateCellValueIndirectBuffer`): four words, quantised rgb and a sample count. */
export async function readIndirectScratch(device: GPUDevice, cache: HashGridCache, cells: readonly number[]): Promise<Map<number, Uint32Array>> {
    const scratch = new Map<number, Uint32Array>();
    if (cells.length === 0) return scratch;
    const cellBytes = 16;
    const staging = device.createBuffer({ label: 'world-cache indirect scratch', size: cells.length * cellBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'world-cache indirect scratch' });
    const base = cache.layout.words.updateCellValueIndirectBuffer * 4;
    cells.forEach((cell, index) => encoder.copyBufferToBuffer(cache.buffers.words, base + cell * cellBytes, staging, index * cellBytes, cellBytes));
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readRange(device, staging, 0, cells.length * cellBytes));
    staging.destroy();
    cells.forEach((cell, index) => scratch.set(cell, words.slice(index * 4, index * 4 + 4)));
    return scratch;
}

/** The direct (or indirect) values of whole tiles (all four levels), unpacked: `numCellsPerTile` × 4 floats per tile, in `tiles` order. */
export async function readTiles(device: GPUDevice, cache: HashGridCache, tiles: readonly number[], which: 'direct' | 'indirect' = 'direct'): Promise<Float32Array> {
    const capacity = cache.capacity;
    const tileBytes = capacity.numCellsPerTile * 8;
    const staging = device.createBuffer({ label: 'world-cache tiles', size: Math.max(4, tiles.length * tileBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'world-cache tiles' });
    const valueBase = (which === 'direct' ? cache.layout.cellValues.valueBuffer : cache.layout.cellValues.valueIndirectBuffer) * 8;
    tiles.forEach((tile, index) => encoder.copyBufferToBuffer(cache.buffers.cellValues, valueBase + tile * tileBytes, staging, index * tileBytes, tileBytes));
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readRange(device, staging, 0, tiles.length * tileBytes));
    staging.destroy();
    return unpackHalves(words);
}

/** One cell's value in `readTiles`' output: tile number `slot` (its place in the list), cell `cellInTile`. */
export function cellValue(values: Float32Array, capacity: HashGridCacheCapacity, slot: number, cellInTile: number): CellValue {
    const base = (slot * capacity.numCellsPerTile + cellInTile) * 4;
    return [values[base], values[base + 1], values[base + 2], values[base + 3]];
}

export function relativeError(value: number, expected: number): number {
    return Math.abs(value - expected) / Math.abs(expected);
}

