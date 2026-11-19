/**
 * RC3.2 and RC3.3: the radiance cache's world cache (Capsaicin's hash grid)
 * populated and resolved through the technique, on the GPU; and GI-1.2's
 * multibounce.
 *
 * Sections 1 to 7 are GI-1.0's single bounce: they run with multibounce off
 * (`useMultibounce: false`), and pass as they did in RC3.2 (the single-bounce
 * path is unchanged by RC3.3). Sections 8 to 10 are the multibounce.
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
 * 6. **The shadow ray, blocked** (the sun of section 1 and a level slab at
 *    y = 1.5 between it and the corner, 40 frames): the slab shades the
 *    floor for z in [0, 2] and the wall below 0.5 m. A cell covers a box
 *    (a square of its tile across the ray's dominant axis, through the whole
 *    tile along it; the tile's key bins the direction, so each axis its rays
 *    can have as dominant gives a box), so a cell is judged only when every
 *    floor and wall point in its boxes is on one side of the shadow's edge
 *    (1 cm clear) and no part of the slab is in them: its mean is 0 in shadow and
 *    albedo / π × E within 2% in the light, and so is every resolved ray
 *    whose picked level's box is (the pick as section 3 checks it). A
 *    bypassed ray (shorter than its cell) reads its own sample, judged by
 *    its hit's side 1 cm from the edge. Every kind has cells or rays, so a
 *    `giTraceAny` that never hit would light the shadowed ones and fail.
 * 7. **UpdateTiles' blend weight** (section 1's sun for 40 frames, then one
 *    frame at four times its intensity): every finest cell of every tile on
 *    the update list holds what the CPU port of the blend
 *    (`updateTilesAccumulate`) makes of its value before the frame and the
 *    frame's samples (one per shadow ray into it, each the new radiance):
 *    at the cap the mean moves 1/16 of the way, below it 1/count; a listed
 *    cell with no sample blends in a zero, as Capsaicin's does. Within two
 *    half steps; capped and uncapped cells must both be there.
 *
 * The multibounce sections use a second scene, the **furnace**: two facing
 * planes 400 m wide, a floor (y = 0, facing up) and a ceiling (y = 3, facing
 * down), albedo ρ = 0.5, under a black sky, lit by a sun straight down
 * (intensity E = 4) that casts no shadow, so it reaches the floor through the
 * ceiling and never the ceiling's underside. The camera at (0, 2, 0) looks
 * down at (0, 0, -2), so every probe is on the floor and every probe ray
 * that hits, hits the ceiling. Expected, before the run:
 *
 * - The floor's light leaving any point: L_f = ρ E / π = 0.6366.
 * - The ceiling gets no direct light: its cells' direct values have no
 *   samples, and with multibounce off its probe rays read 0.
 * - With multibounce on, 30% of the ceiling hits trace a cosine-weighted
 *   ray down to the floor; each sample is L_f × BRDF / pdf / (1 - 0.7) =
 *   ρ L_f / 0.3, and a cell averages the samples it gets, so a ceiling cell's
 *   indirect value, and every probe ray reading it, is ρ² E / (π × 0.3) =
 *   1.0610: up from 0 by that much, and 1 / 0.3 = 3.33 times the second
 *   bounce's expected value ρ² E / π = 0.3183, because only kept rays add a
 *   sample (Capsaicin's arithmetic, kept for now: DECISIONS 2026-11-19 fixes it by default behind a switch; NOTES, RC3.3).
 *
 * Far from the camera a tile is taller than the 3 m gap, and nearly level
 * rays up and down share a direction bin, so a far tile holds floor and
 * ceiling hits in the same cells (NOTES, "a hash-grid cell is a column
 * through its tile"): there a ceiling ray also reads the floor's direct
 * light. The checks allow for it where tiles reach 2 m.
 *
 * Two of Capsaicin's index uses are ported as they are (NOTES, RC3.3), and
 * the checks expect what they do: a second-bounce shadow ray that bypasses
 * the cache (far from the camera a floor cell is wider than the 3 m ray)
 * adds the floor's light, L_f, to the probe ray numbered as its first-bounce
 * entry (`PopulateCells`), and `UpdateMultibounceCells` reads the shadow-ray
 * list at a second-bounce entry's visibility index and, where that word has
 * the bypass bit, adds its sample (about 1.0610) to the probe ray numbered
 * as the first-bounce entry. A probe ray's expected reading is its cell's
 * levels plus those additions.
 *
 * 8. **Multibounce, converged** (40 frames): every probe ray that hit the
 *    ceiling reads what its cell's levels hold (direct plus indirect, as
 *    `ResolveCells` picks them), and where its tile is under 2 m (it can't
 *    reach the floor) that is no direct light and an indirect 1.0610 within
 *    2%; every cell with indirect samples has an indirect mean of 1.0610
 *    within 2%, and every cell with direct samples (the floor's, from the
 *    second-bounce hits) a direct mean of L_f within 2%. The last
 *    frame's draws match the CPU twin (`multibounce-cpu.ts`): the first-bounce
 *    entries whose `rand()` is above 0.7 are exactly those with a
 *    second-bounce entry (less the rays the CPU sees leave the planes), each
 *    second-bounce ray's direction is the CPU's within half precision, its
 *    BRDF / pdf is ρ and its pdf cos / π. The update list holds each tile
 *    once (register D32).
 * 9. **The one-frame lag** (multibounce turned on after two frames with it
 *    off, which empties the cache and its indirect scratch; then one frame,
 *    then a second): after the first frame no ceiling cell has an indirect
 *    sample and the ceiling rays under 2 m tiles read 0, while each ceiling
 *    cell's indirect scratch holds one count per second-bounce entry whose
 *    first bounce is in it, and the quantised sum of their samples; after the
 *    second frame those rays that read an indirect level with samples read
 *    1.0610 within 2%.
 * 10. **Multibounce off** (40 frames): no second-bounce entry, no shadow ray
 *    and no resolve entry (Capsaicin resolves its shadow-ray list without
 *    multibounce, register D35), and every probe ray reads 0.
 * 11. **Both misdirected additions, predicted** (the sunlit corner with
 *    multibounce on, 40 frames; the furnace never fires the second one):
 *    every probe ray not bypassed itself reads its cell's levels, plus
 *    `UpdateMultibounceCells`' addition where predicted (its exact sample,
 *    from the second cell's levels and the entry's BRDF and pdf), plus 0 or
 *    albedo / π × E where `PopulateCells`' is (the wall shades some second
 *    hits); both must fire. Rays that bypass the cache themselves are left
 *    out: their own addition and a misdirected one in the same dispatch can
 *    race (`ScreenProbes_AccumulateRadiance` is not atomic; NOTES).
 * 12. **Emptied when a cell size changes** (the corner, single bounce): after
 *    a frame with `hashGridCacheCellSize`, then `hashGridCacheMinCellSize`,
 *    changed, the live tiles are exactly the frame's (gi1.cpp:1740-1745);
 *    a frame with nothing changed keeps more.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc3.2 world cache" -- pnpm test:gpu --filter radiance-cache-world-cache --gpu nvidia` (or intel).
 */

import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
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
    updateTilesAccumulate,
    verticalFieldOfView,
    type CellValue,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import { BYPASS_CACHE_BIT, MULTIBOUNCE_INFO_WORDS, MULTIBOUNCE_RAY_WORDS, RESERVOIR_WORDS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import { multibounceDraw, multibounceSettledIndirect, sampleBRDFAndEvaluateDiffuse } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/multibounce-cpu.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
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

/**
 * Section 6's blocker: a level slab (one quad, facing up; the tracer hits
 * both sides) between the corner and the sun. Its shadow along (0, -1, -1)
 * covers the floor for z in [0, 2] and the wall below y = 0.5, for x in
 * [-1, 2]; the camera sees all of it past the slab's near edge.
 */
const SLAB = { minX: -1, maxX: 2, minZ: 1, maxZ: 3.5, y: 1.5 } as const;

/** Section 7: the sun's intensity for the one frame after convergence at SUN_INTENSITY. */
const CHANGED_SUN_INTENSITY = 4 * SUN_INTENSITY;

/** Cells per tile at each level, and where each level starts (Capsaicin's defaults: 64, 16, 4, 1). */
const LEVEL_SIDES = [8, 4, 2, 1] as const;

/** Sections 8-10: the furnace (see the file's comment). */
const FURNACE = {
    camera: { position: [0, 2, 0], target: [0, 0, -2], fovYDeg: 60, near: 0.05, far: 1000 } as PageCamera,
    halfWidth: 200,
    ceilingY: 3,
    albedo: 0.5,
    sunIntensity: 4,
} as const;
/** `gi1_hash_grid_cache_discard_multibounce_ray_probability` (gi1.h:64). */
const DISCARD_PROBABILITY = 0.7;
/** The floor's light: ρ E / π. */
const FLOOR_RADIANCE = (FURNACE.albedo * FURNACE.sunIntensity) / Math.PI;
/** The second bounce a ceiling cell stands for: ρ L_f = ρ² E / π, what an unbiased estimate would settle at. */
const TRUE_SECOND_BOUNCE = FURNACE.albedo * FLOOR_RADIANCE;
/**
 * Capsaicin's factor on it: 1 / (1 - p), since only kept rays add a sample (NOTES, RC3.3). A
 * switch that removes the bias would test against TRUE_SECOND_BOUNCE alone.
 */
const CAPSAICIN_FACTOR = 1 / (1 - DISCARD_PROBABILITY);
/** A ceiling cell's settled indirect value, as Capsaicin computes it (`multibounceSettledIndirect`). */
const CEILING_INDIRECT = multibounceSettledIndirect(FURNACE.albedo, FLOOR_RADIANCE, DISCARD_PROBABILITY);
/** The ceiling's normal on the side the probe rays hit: down. */
const CEILING_NORMAL: Vec3 = [0, -1, 0];

type Surface = 'floor' | 'wall' | 'patch' | 'slab' | 'ceiling' | 'other';

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
    readonly camera: PageCamera;
    /** The furnace (sections 8-10) rather than the corner. */
    readonly furnace: boolean;
    readonly platform: GIPlatformImpl;
    readonly lights: readonly LightHandle[];
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

interface CornerExtras {
    /** Section 4's emissive patch on the floor. */
    readonly patch?: boolean;
    /** Section 6's slab between the corner and the sun. */
    readonly slab?: boolean;
    /** Sections 8-10: the furnace (floor and ceiling) instead of the corner, seen from its own camera. */
    readonly furnace?: boolean;
}

function describeCorner(platform: GIPlatformImpl, lights: readonly LightDescriptor[], extras: CornerExtras): LightHandle[] {
    if (extras.furnace) return describeFurnace(platform, lights);
    const half = 200;
    quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], ALBEDO);
    quad(platform, 'wall', [[-20, 0, 0], [20, 0, 0], [20, 10, 0], [-20, 10, 0]], [0, 0, 1], ALBEDO);
    if (extras.patch) {
        const { minX, maxX, minZ, maxZ, y } = PATCH;
        quad(platform, 'emissive patch', [[minX, y, maxZ], [maxX, y, maxZ], [maxX, y, minZ], [minX, y, minZ]], [0, 1, 0], ALBEDO, EMISSION);
    }
    if (extras.slab) {
        const { minX, maxX, minZ, maxZ, y } = SLAB;
        quad(platform, 'slab', [[minX, y, maxZ], [maxX, y, maxZ], [maxX, y, minZ], [minX, y, minZ]], [0, 1, 0], ALBEDO);
    }
    const handles = lights.map((light) => platform.scene.createLight(light));
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    return handles;
}

/** The furnace: a floor facing up and a ceiling facing down, 400 m wide, under a black sky. */
function describeFurnace(platform: GIPlatformImpl, lights: readonly LightDescriptor[]): LightHandle[] {
    const half = FURNACE.halfWidth;
    const y = FURNACE.ceilingY;
    quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], FURNACE.albedo);
    quad(platform, 'ceiling', [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]], [0, -1, 0], FURNACE.albedo);
    const handles = lights.map((light) => platform.scene.createLight(light));
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    return handles;
}

/** The furnace's sun: straight down, through the ceiling (it casts no shadow). */
const FURNACE_SUN: LightDescriptor = { type: 'directional', direction: [0, -1, 0], color: [1, 1, 1], intensity: FURNACE.sunIntensity, castsShadows: false, label: 'furnace sun' };

function sun(travel: Vec3, color: Vec3 = [1, 1, 1]): LightDescriptor {
    return { type: 'directional', direction: travel, color, intensity: SUN_INTENSITY, label: 'sun' };
}

function frameInput(camera: PageCamera): FrameInput {
    return { deltaTime: 1 / 60, budget: { units: 0 }, camera: camera === CAMERA ? FRAME_CAMERA : frameCamera(camera, WIDTH, HEIGHT) };
}

async function startScenario(device: GPUDevice, name: string, lights: readonly LightDescriptor[], extras: CornerExtras, settings: Partial<RadianceCacheSettings>): Promise<Scenario> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-world-cache ${name}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique(settings));
    platform.setOutput(technique);
    await platform.prepare();
    const lightHandles = describeCorner(platform, lights, extras);
    const camera = extras.furnace ? FURNACE.camera : CAMERA;
    const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
    const frame = (): void => {
        const encoder = device.createCommandEncoder({ label: `radiance-cache-world-cache ${name}` });
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
    // Start the checks from an empty cache, with the trees in place.
    technique.resetHistory();
    const cache = technique.worldCache;
    if (!cache) throw new Error(`${name}: the technique has no world cache after prepare`);
    return {
        camera,
        furnace: extras.furnace ?? false,
        platform,
        lights: lightHandles,
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
    /** The first-bounce visibility entries' cells and queries (probe rays). */
    readonly visibilityCell: Uint32Array;
    readonly visibilityQuery: Uint32Array;
    /** The second-bounce entries' (RC3.3): cells, queries (first-bounce entries), BRDF and pdf (four halves each), and rays (`MULTIBOUNCE_RAY_WORDS` each). */
    readonly multibounceCell: Uint32Array;
    readonly multibounceQuery: Uint32Array;
    readonly multibounceInfo: Float32Array;
    readonly multibounceRays: Uint32Array;
    /** The shadow-ray list's words, all of them (`UpdateMultibounceCells` reads it past the frame's count: NOTES, RC3.3). */
    readonly visibilityRayWords: Uint32Array;
    readonly visibilityRay: Uint32Array;
    readonly resolve: Uint32Array;
    readonly reservoirs: Uint32Array;
    readonly updateTiles: Uint32Array;
    readonly packedTiles: Uint32Array;
}

function classify(hit: Vec3, furnace: boolean): Surface {
    if (furnace) return Math.abs(hit[1] - FURNACE.ceilingY) < 1e-3 ? 'ceiling' : Math.abs(hit[1]) < 1e-3 ? 'floor' : 'other';
    if (Math.abs(hit[1] - SLAB.y) < 1e-3 && hit[0] >= SLAB.minX - 1e-3 && hit[0] <= SLAB.maxX + 1e-3 && hit[2] >= SLAB.minZ - 1e-3 && hit[2] <= SLAB.maxZ + 1e-3) return 'slab';
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
        rays.push({ origin, direction, hitT, hit, frontFace: rayWords[base + 9] !== 0, surface: hitT >= 0 ? classify(hit, scenario.furnace) : 'other' });
    }
    const results = probes.rayScratch.rayResults;
    const radiance = unpackHalves(new Uint32Array(await readRange(device, results.buffer, results.offset ?? 0, rayCount * 8)));
    const counters = new Uint32Array(await readRange(device, scenario.cache.buffers.tileWords, 0, 64 * 4));
    const words = new Uint32Array(await readRange(device, lists.buffer, 0, lists.buffer.size));
    const layout = lists.layout;
    const count0 = counters[HASH_GRID_COUNTER.visibilityCount0];
    const count1 = counters[HASH_GRID_COUNTER.visibilityCount1];
    const rayListCount = counters[HASH_GRID_COUNTER.visibilityRayCount];
    const resolveCount = counters[HASH_GRID_COUNTER.resolveCount];
    const packed = scenario.cache.packedTiles;
    const packedCount = new Uint32Array(await readRange(device, packed.count.buffer, packed.count.offset, 4))[0];
    const packedTiles = new Uint32Array(await readRange(device, scenario.cache.buffers.tileWords, packed.firstWord * 4, packedCount * 4));
    return {
        rays,
        radiance,
        counters,
        visibilityCell: words.slice(layout.visibilityCellBuffer, layout.visibilityCellBuffer + count0),
        visibilityQuery: words.slice(layout.visibilityQueryBuffer, layout.visibilityQueryBuffer + count0),
        multibounceCell: words.slice(layout.visibilityCellBuffer + count0, layout.visibilityCellBuffer + count0 + count1),
        multibounceQuery: words.slice(layout.visibilityQueryBuffer + count0, layout.visibilityQueryBuffer + count0 + count1),
        multibounceInfo: unpackHalves(words.slice(layout.multibounceInfoBuffer, layout.multibounceInfoBuffer + MULTIBOUNCE_INFO_WORDS * count1)),
        multibounceRays: words.slice(layout.multibounceRayBuffer, layout.multibounceRayBuffer + MULTIBOUNCE_RAY_WORDS * count1),
        visibilityRayWords: words.slice(layout.visibilityRayBuffer, layout.visibilityRayBuffer + layout.maxCombinedRayCount),
        visibilityRay: words.slice(layout.visibilityRayBuffer, layout.visibilityRayBuffer + rayListCount),
        resolve: words.slice(layout.resolveBuffer, layout.resolveBuffer + resolveCount),
        reservoirs: words.slice(layout.reservoirBuffer, layout.reservoirBuffer + RESERVOIR_WORDS * rayListCount),
        updateTiles: words.slice(layout.updateTileBuffer, layout.updateTileBuffer + counters[HASH_GRID_COUNTER.updateTileCount]),
        packedTiles,
    };
}

/**
 * The direct values (or with `indirect`, the indirect values) of whole tiles (all four levels),
 * unpacked: `numCellsPerTile` × 4 floats per tile, in `tiles` order.
 */
async function readTiles(device: GPUDevice, cache: HashGridCache, tiles: readonly number[], indirect = false): Promise<Float32Array> {
    const capacity = cache.capacity;
    const tileBytes = capacity.numCellsPerTile * 8;
    const staging = device.createBuffer({ label: 'world-cache tiles', size: Math.max(4, tiles.length * tileBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'world-cache tiles' });
    const valueBase = (indirect ? cache.layout.cellValues.valueIndirectBuffer : cache.layout.cellValues.valueBuffer) * 8;
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

// ---- Section 6: the shadow ray, blocked ---------------------------------------------------------

/**
 * Where a floor or wall point's ray toward the sun (along -SUN_TRAVEL, (0, 1, 1)) crosses the
 * slab's plane, and how far inside the slab's rectangle that is (negative outside). The
 * crossing moves with the point one for one along x and along the surface's other axis, so the
 * distance is also how far the point is from its shadow's edge on its own surface.
 */
function slabShadowMargin(point: Vec3): number {
    const rise = SLAB.y - point[1];
    if (rise <= 0) return -Infinity; // at or above the slab: nothing between it and the sun
    const x = point[0];
    const z = point[2] + rise;
    const insideX = Math.min(x - SLAB.minX, SLAB.maxX - x);
    const insideZ = Math.min(z - SLAB.minZ, SLAB.maxZ - z);
    if (insideX > 0 && insideZ > 0) return Math.min(insideX, insideZ);
    return -Math.hypot(Math.min(insideX, 0), Math.min(insideZ, 0));
}

interface Box {
    readonly lo: Vec3;
    readonly hi: Vec3;
}

/** The two axes across a dominant axis, in `HashGridCache_GetDesc`'s order. */
const ACROSS: readonly (readonly [number, number])[] = [
    [1, 2],
    [0, 2],
    [0, 1],
];

/** A cell as `HashGridCache_GetDesc` keys it: its tile's corner and cell size, its 2D offset (the GPU's), and the dominant axes the tile's rays can have. */
interface CellKey {
    /** The tile's corner; two along an axis where the hit is on a tile's edge (the GPU's f32 hit may be on either side). */
    readonly tileLos: readonly Vec3[];
    readonly cellSize: number;
    readonly offset: readonly [number, number];
    readonly axes: readonly number[];
}

/**
 * The hit's cell. The tile is a cube of 8 cells a side at the hit's cell size; the cell is a
 * square of it across the ray's dominant axis, running through the whole tile along that axis.
 * The tile's key holds the direction only binned (5 bins a component), so other rays of the
 * same tile may have another dominant axis and put the same 2D offset across that one: every
 * axis a direction in the bins can have as its largest is kept. The offset is the GPU's
 * (`offset`, from the cell index).
 */
function cellKey(hit: Vec3, direction: Vec3, offset: readonly [number, number], constants: { cellSize: number; minCellSize: number }): CellKey {
    const cellSize = hashGridCacheGetCellSize(hit, CAMERA.position, constants);
    const tileSize = cellSize * LEVEL_SIDES[0];
    const corners = hit.map((value) => {
        const scaled = value / tileSize;
        const floor = Math.floor(scaled);
        const options = scaled - floor < 1e-4 ? [floor - 1, floor] : floor + 1 - scaled < 1e-4 ? [floor, floor + 1] : [floor];
        return options.map((option) => option * tileSize);
    });
    const tileLos: Vec3[] = [];
    for (const x of corners[0]) for (const y of corners[1]) for (const z of corners[2]) tileLos.push([x, y, z]);
    // The bin of each component (floor(0.5 + (0.5 c + 0.5) x 4)) and the |c| it allows.
    const ranges = direction.map((component) => {
        const bin = Math.floor(0.5 + (0.5 * component + 0.5) * 4);
        const low = Math.max(-1, (bin - 2.5) / 2);
        const high = Math.min(1, (bin - 1.5) / 2);
        return low >= 0 ? [low, high] : high <= 0 ? [-high, -low] : [0, Math.max(-low, high)];
    });
    const axes = [0, 1, 2].filter((axis) => [0, 1, 2].every((other) => other === axis || ranges[axis][1] >= ranges[other][0]));
    return { tileLos, cellSize, offset, axes };
}

/** The boxes a cell (or its coarser cell at `level`) can cover, one per possible dominant axis. */
function cellBoxes(key: CellKey, level: number): Box[] {
    const tileSize = key.cellSize * LEVEL_SIDES[0];
    const side = 2 ** level;
    return key.tileLos.flatMap((tileLo) =>
        key.axes.map((dominant) => {
            const lo = [...tileLo] as [number, number, number];
            const hi = lo.map((value) => value + tileSize) as [number, number, number];
            ACROSS[dominant].forEach((axis, index) => {
                lo[axis] += Math.floor(key.offset[index] / side) * side * key.cellSize;
                hi[axis] = lo[axis] + side * key.cellSize;
            });
            return { lo, hi };
        }),
    );
}

type BoxLight = 'shadowed' | 'lit' | 'edge' | 'slab' | 'empty';

/** One verdict for several boxes: shadowed or lit only when every box that holds a surface is. */
function boxesLight(boxes: readonly Box[]): BoxLight {
    const lights = boxes.map(boxLight).filter((light) => light !== 'empty');
    if (lights.length === 0) return 'empty';
    if (lights.includes('slab')) return 'slab';
    if (lights.every((light) => light === 'shadowed')) return 'shadowed';
    if (lights.every((light) => light === 'lit')) return 'lit';
    return 'edge';
}

/**
 * Whether every floor and wall point a probe ray can hit inside the box is in the slab's shadow
 * (1 cm in at least) or out of it (1 cm out), sampled on a 17 × 17 grid of each piece; a box
 * that holds part of the slab is left out (which side its rays hit isn't known here).
 */
function boxLight(box: Box): BoxLight {
    const { lo, hi } = box;
    const near = (low: number, high: number, value: number): boolean => low - 1e-3 <= value && value <= high + 1e-3;
    if (near(lo[1], hi[1], SLAB.y) && hi[0] > SLAB.minX && lo[0] < SLAB.maxX && hi[2] > SLAB.minZ && lo[2] < SLAB.maxZ) return 'slab';
    const pieces: ((u: number, v: number) => Vec3)[] = [];
    const span = (low: number, high: number, min: number, max: number): [number, number] | null => {
        const from = Math.max(low, min);
        const to = Math.min(high, max);
        return from <= to ? [from, to] : null;
    };
    if (near(lo[1], hi[1], 0)) {
        const x = span(lo[0], hi[0], -200, 200);
        const z = span(lo[2], hi[2], 0, 200); // the floor in front of the wall
        if (x && z) pieces.push((u, v) => [x[0] + u * (x[1] - x[0]), 0, z[0] + v * (z[1] - z[0])]);
    }
    if (near(lo[2], hi[2], 0)) {
        const x = span(lo[0], hi[0], -20, 20);
        const y = span(lo[1], hi[1], 0, 10);
        if (x && y) pieces.push((u, v) => [x[0] + u * (x[1] - x[0]), y[0] + v * (y[1] - y[0]), 0]);
    }
    if (pieces.length === 0) return 'empty';
    let least = Infinity;
    let most = -Infinity;
    const steps = 16;
    for (const piece of pieces) {
        for (let i = 0; i <= steps; i++) {
            for (let j = 0; j <= steps; j++) {
                const margin = slabShadowMargin(piece(i / steps, j / steps));
                least = Math.min(least, margin);
                most = Math.max(most, margin);
            }
        }
    }
    if (least > 0.01) return 'shadowed';
    if (most < -0.01) return 'lit';
    return 'edge';
}

interface Tally {
    rays: number;
    off: number;
    worst: number;
}

function newTally(): Tally {
    return { rays: 0, off: 0, worst: 0 };
}

async function checkShadow(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`shadow: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const capacity = scenario.cache.capacity;
    const constants = cellSizeConstants();
    const counts: Record<BoxLight, number> = { shadowed: 0, lit: 0, edge: 0, slab: 0, empty: 0 };
    // Expected readings: 0 in shadow (every sample is 0), albedo / pi x E in the light.
    const judge = (tally: Tally, values: readonly number[], shadowed: boolean, describe: () => string): void => {
        tally.rays++;
        const error = shadowed ? Math.max(...values) / LIT_RADIANCE : Math.max(...values.map((value) => relativeError(value, LIT_RADIANCE)));
        tally.worst = Math.max(tally.worst, error);
        if (shadowed ? error > 1e-3 : error > TOLERANCE) {
            tally.off++;
            if (tally.off <= 2) fail(`${describe()} reads ${values.map((value) => value.toFixed(4)).join(', ')}, expected ${shadowed ? 0 : LIT_RADIANCE.toFixed(4)}`);
        }
    };

    // The cells: each finest cell hit this frame, whose whole box is on one side of the edge.
    const hitsByCell = new Map<number, number[]>();
    state.visibilityCell.forEach((cell, visibility) => {
        const list = hitsByCell.get(cell) ?? [];
        list.push(state.visibilityQuery[visibility]);
        hitsByCell.set(cell, list);
    });
    const tiles = [...new Set(Array.from(hitsByCell.keys(), (cell) => Math.floor(cell / capacity.numCellsPerTile)))];
    const slotOfTile = new Map(tiles.map((tile, slot) => [tile, slot]));
    const values = await readTiles(device, scenario.cache, tiles);
    const cells = { shadowed: newTally(), lit: newTally() };
    let cellSizeDisagreements = 0;
    for (const [cell, rays] of hitsByCell) {
        const { tileIndex, cellOffset } = hashGridCacheCellOffsetMip0(cell, capacity);
        const keys = rays.map((ray) => cellKey(state.rays[ray].hit, state.rays[ray].direction, [cellOffset[0], cellOffset[1]], constants));
        // Every hit's boxes: the cell holds samples from all of them. A hit whose cell size the
        // CPU makes another (a distance on a level's edge) can't place the GPU's offset: left out.
        if (keys.some((key) => key.cellSize !== keys[0].cellSize)) {
            cellSizeDisagreements++;
            continue;
        }
        const light = boxesLight(keys.flatMap((key) => cellBoxes(key, 0)));
        counts[light]++;
        if (light !== 'shadowed' && light !== 'lit') continue;
        const value = cellValue(values, capacity, slotOfTile.get(tileIndex) ?? 0, hashGridCacheCellIndex(cellOffset, tileIndex, 0, capacity) - tileIndex * capacity.numCellsPerTile);
        if (!(value[3] > 0)) continue;
        judge(cells[light], [0, 1, 2].map((channel) => value[channel] / value[3]), light === 'shadowed', () => `cell ${cell} (${light}, ${value[3]} samples)`);
    }

    // The rays: a resolved ray reads the level ResolveCells picks (the CPU's pick from the cells
    // as read back, as section 3 checks), so its box is that level's; a bypassed ray reads its
    // own sample, so only its hit's own side counts.
    const rays = { cachedShadowed: newTally(), cachedLit: newTally(), bypassedShadowed: newTally(), bypassedLit: newTally() };
    let raysOnTheEdge = 0;
    let slabHits = 0;
    for (const entry of state.resolve) {
        const visibility = entry & ~BYPASS_CACHE_BIT;
        const index = state.visibilityQuery[visibility];
        const ray = state.rays[index];
        if (ray.surface === 'slab') {
            slabHits++;
            continue;
        }
        const rgb = [0, 1, 2].map((channel) => state.radiance[index * 4 + channel]);
        const describe = (side: string): (() => string) => () => `ray ${index} (${ray.surface}, ${side})`;
        if ((entry & BYPASS_CACHE_BIT) !== 0) {
            const margin = slabShadowMargin(ray.hit);
            if (Math.abs(margin) < 0.01) {
                raysOnTheEdge++;
                continue;
            }
            judge(margin > 0 ? rays.bypassedShadowed : rays.bypassedLit, rgb, margin > 0, describe(`bypassed, ${margin > 0 ? 'in shadow' : 'lit'}`));
            continue;
        }
        const cell = state.visibilityCell[visibility];
        const { tileIndex, cellOffset } = hashGridCacheCellOffsetMip0(cell, capacity);
        const slot = slotOfTile.get(tileIndex) ?? 0;
        const levels = [0, 1, 2, 3].map((level) =>
            capacity.sizeTileMip[level] > 0 ? cellValue(values, capacity, slot, hashGridCacheCellIndex(cellOffset, tileIndex, level, capacity) - tileIndex * capacity.numCellsPerTile) : null,
        );
        const picked = hashGridCacheFilteredRadiance(levels, MAX_SAMPLE_COUNT);
        const light = boxesLight(cellBoxes(cellKey(ray.hit, ray.direction, [cellOffset[0], cellOffset[1]], constants), picked.level));
        if (light !== 'shadowed' && light !== 'lit') {
            raysOnTheEdge++;
            continue;
        }
        judge(light === 'shadowed' ? rays.cachedShadowed : rays.cachedLit, rgb, light === 'shadowed', describe(`resolved at level ${picked.level}, ${light}`));
    }

    const tallies = { shadowedCells: cells.shadowed, litCells: cells.lit, ...rays };
    for (const [name, tally] of Object.entries(tallies)) {
        if (tally.rays === 0) fail(`nothing to check for ${name}: move the slab`);
        if (tally.off > 0) fail(`${tally.off} of ${tally.rays} ${name} are off`);
    }
    const overflow = state.counters[HASH_GRID_COUNTER.bucketOverflowCount];
    if (overflow !== 0) fail(`${overflow} inserts refused`);
    return { failures, data: { ...tallies, cellBoxes: counts, cellSizeDisagreements, raysOnTheEdge, slabHits, slab: SLAB } };
}

// ---- Section 7: UpdateTiles' blend weight ---------------------------------------------------------

/** The tiles' finest-level cells as `[rgb × count, count]`, by tile. */
function finestCells(values: Float32Array, capacity: HashGridCacheCapacity, tiles: readonly number[]): Map<number, CellValue[]> {
    const cells = new Map<number, CellValue[]>();
    tiles.forEach((tile, slot) => {
        cells.set(
            tile,
            Array.from({ length: capacity.numCellsPerTileMip[0] }, (_, cell) => cellValue(values, capacity, slot, cell)),
        );
    });
    return cells;
}

async function checkBlend(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`blend: ${message}`);
    };
    const capacity = scenario.cache.capacity;
    // The cells converged at SUN_INTENSITY: every sample of a lit cell is the same, so its mean is L1.
    const before = await readFrameState(device, scenario);
    const tiles = Array.from(before.packedTiles);
    const previous = finestCells(await readTiles(device, scenario.cache, tiles), capacity, tiles);

    // One frame with the sun four times as bright: each new sample is L2 = 4 L1.
    scenario.platform.scene.updateLight(scenario.lights[0], { intensity: CHANGED_SUN_INTENSITY });
    await runFrames(device, scenario, 1);
    const state = await readFrameState(device, scenario);
    const after = finestCells(await readTiles(device, scenario.cache, tiles), capacity, tiles);

    // This frame's samples per cell: one per shadow ray (PopulateCells adds a count of 1 for
    // each, lit or not), each L2 quantised to 1/1000 as HashGridCache_QuantizeRadiance does.
    const samples = new Map<number, number>();
    for (const entry of state.visibilityRay) {
        const cell = state.visibilityCell[entry & ~BYPASS_CACHE_BIT];
        samples.set(cell, (samples.get(cell) ?? 0) + 1);
    }
    const changedRadiance = (ALBEDO / Math.PI) * CHANGED_SUN_INTENSITY * Math.SQRT1_2;
    const quantizedSample = Math.round(1000 * changedRadiance) / 1000;

    const kinds: Record<'capped' | 'uncapped' | 'noSample', Tally> = {
        // At the cap before and after: the mean moves (L2 - L1) / 16.
        capped: { rays: 0, off: 0, worst: 0 },
        // Below the cap after: the weight is 1 / (count before + new samples).
        uncapped: { rays: 0, off: 0, worst: 0 },
        // On the update list with no sample this frame: Capsaicin blends in a zero sample.
        noSample: { rays: 0, off: 0, worst: 0 },
    };
    const updated = new Set(state.updateTiles);
    let tilesChecked = 0;
    for (const tile of tiles) {
        const was = previous.get(tile);
        const now = after.get(tile);
        if (!updated.has(tile) || !was || !now) continue;
        tilesChecked++;
        for (let cell = 0; cell < capacity.numCellsPerTileMip[0]; cell++) {
            const old = was[cell];
            if (!(old[3] > 0)) continue;
            const count = samples.get(tile * capacity.numCellsPerTile + cell) ?? 0;
            const sum = quantizedSample * count;
            const expected = updateTilesAccumulate(old, [sum, sum, sum, count], MAX_SAMPLE_COUNT);
            const got = now[cell];
            const kind = count === 0 ? kinds.noSample : old[3] >= MAX_SAMPLE_COUNT ? kinds.capped : kinds.uncapped;
            kind.rays++;
            // rgb x count and the count, each stored as a half: within a half step or two.
            const error = Math.max(...[0, 1, 2, 3].map((channel) => Math.abs(got[channel] - expected[channel]) / Math.max(Math.abs(expected[channel]), 1e-3)));
            kind.worst = Math.max(kind.worst, error);
            if (error > 2 * HALF_TOLERANCE) {
                kind.off++;
                if (kind.off <= 2) {
                    const shown = (value: readonly number[]): string => value.map((channel) => channel.toFixed(4)).join(', ');
                    fail(`tile ${tile} cell ${cell} (${old[3]} samples before, ${count} new) holds ${shown(got)}, UpdateTiles gives ${shown(expected)}`);
                }
            }
        }
    }
    for (const [name, kind] of Object.entries(kinds)) {
        if (kind.off > 0) fail(`${kind.off} of ${kind.rays} ${name} cells differ from UpdateTiles' blend`);
    }
    if (kinds.capped.rays === 0) fail('no capped cell got a sample: the 1/16 weight is unchecked');
    if (kinds.uncapped.rays === 0) fail('no cell below the cap got a sample: the 1/count weight is unchecked');
    return {
        failures,
        data: {
            convergedRadiance: LIT_RADIANCE,
            newSampleRadiance: quantizedSample,
            cappedMeanExpected: LIT_RADIANCE + (quantizedSample - LIT_RADIANCE) / MAX_SAMPLE_COUNT,
            tilesChecked,
            cellsByKind: kinds,
        },
    };
}

// ---- Sections 8-10: GI-1.2's multibounce in the furnace -------------------------------------------

/** Sections 1-7 run GI-1.0's single bounce; sections 8 and 10 set multibounce as they need it. */
const SINGLE_BOUNCE: Partial<RadianceCacheSettings> = { useMultibounce: false };
const MULTIBOUNCE: Partial<RadianceCacheSettings> = { useMultibounce: true };

/** The frame index the last frame ran with: the decay stamp `HashGridCache_TouchTile` left on a tile it touched. */
async function lastFrameIndex(device: GPUDevice, cache: HashGridCache, cell: number): Promise<number> {
    const tile = Math.floor(cell / cache.capacity.numCellsPerTile);
    const word = cache.layout.tileWords.decayTileBuffer + tile;
    return new Uint32Array(await readRange(device, cache.buffers.tileWords, word * 4, 4))[0];
}

/** A second-bounce entry's ray direction, from its record's halves (`WorldCacheProbeRay.direction`, words 4 and 5). */
function multibounceDirection(state: FrameState, entry: number): Vec3 {
    const base = entry * MULTIBOUNCE_RAY_WORDS;
    return unpackHalf3([state.multibounceRays[base + 4], state.multibounceRays[base + 5]]);
}

/** Every finest-level cell of `tiles` (direct or indirect values), by cell index. */
async function finestValues(device: GPUDevice, cache: HashGridCache, tiles: readonly number[], indirect: boolean): Promise<Map<number, CellValue>> {
    const capacity = cache.capacity;
    const values = await readTiles(device, cache, tiles, indirect);
    const cells = new Map<number, CellValue>();
    tiles.forEach((tile, slot) => {
        for (let cell = 0; cell < capacity.numCellsPerTileMip[0]; cell++) cells.set(tile * capacity.numCellsPerTile + cell, cellValue(values, capacity, slot, cell));
    });
    return cells;
}

/** The level `ResolveCells` reads for a finest cell, from `readTiles`' output of its tile, and its value. */
function pickLevel(values: Float32Array, cache: HashGridCache, slot: number, cell: number, maxCount: number): { readonly level: number; readonly value: CellValue } {
    const capacity = cache.capacity;
    const { tileIndex, cellOffset } = hashGridCacheCellOffsetMip0(cell, capacity);
    const levels = [0, 1, 2, 3].map((level) =>
        capacity.sizeTileMip[level] > 0 ? cellValue(values, capacity, slot, hashGridCacheCellIndex(cellOffset, tileIndex, level, capacity) - tileIndex * capacity.numCellsPerTile) : null,
    );
    return hashGridCacheFilteredRadiance(levels, maxCount);
}

/** A tile under this size (8 cells at the hit's cell size) can't reach from the ceiling to the floor 3 m below. */
const NEAR_TILE_SIZE = 2;

/**
 * What Capsaicin's two kept index uses add to probe rays this frame (see the file's comment),
 * by probe ray: `PopulateCells` adds L_f for each second-bounce shadow ray with the bypass bit,
 * `UpdateMultibounceCells` about CEILING_INDIRECT for each second-bounce entry whose word of the
 * shadow-ray list (read at its visibility index) has the bypass bit. Both land on the probe ray
 * numbered as the first-bounce entry.
 */
function misdirectedAdditions(state: FrameState): { readonly added: Map<number, number>; readonly fromPopulateCells: number; readonly fromUpdateMultibounceCells: number } {
    const added = new Map<number, number>();
    const add = (ray: number, value: number): void => {
        added.set(ray, (added.get(ray) ?? 0) + value);
    };
    const firstBounceCount = state.visibilityCell.length;
    let fromPopulateCells = 0;
    for (const entry of state.visibilityRay) {
        const visibility = entry & ~BYPASS_CACHE_BIT;
        if ((entry & BYPASS_CACHE_BIT) === 0 || visibility < firstBounceCount) continue;
        add(state.multibounceQuery[visibility - firstBounceCount], FLOOR_RADIANCE);
        fromPopulateCells++;
    }
    let fromUpdateMultibounceCells = 0;
    state.multibounceQuery.forEach((query, entry) => {
        if ((state.visibilityRayWords[firstBounceCount + entry] & BYPASS_CACHE_BIT) === 0) return;
        add(query, CEILING_INDIRECT);
        fromUpdateMultibounceCells++;
    });
    return { added, fromPopulateCells, fromUpdateMultibounceCells };
}

/**
 * Every probe ray, judged against what its ceiling cell's levels hold. A resolved ray reads its
 * picked direct level's mean plus its picked indirect level's (`ResolveCells`; within four half
 * steps), a miss reads 0 (the black sky), and either reads Capsaicin's misdirected additions on
 * top (within 1%: their samples are known to about 0.3%). Where a ray's tile is under 2 m its
 * cell holds ceiling hits only: the direct level has no sample, and less the additions the ray
 * reads CEILING_INDIRECT within 2% if its indirect level has samples, 0 if not.
 */
async function judgeCeilingRays(device: GPUDevice, scenario: Scenario, state: FrameState, fail: (message: string) => void): Promise<Record<string, number>> {
    const cache = scenario.cache;
    const capacity = cache.capacity;
    const constants = cellSizeConstants();
    const misdirected = misdirectedAdditions(state);
    const entries = Array.from(state.resolve).filter((entry) => (entry & BYPASS_CACHE_BIT) === 0);
    const bypassed = state.resolve.length - entries.length;
    const resolvedCellOf = new Map(entries.map((entry) => [state.visibilityQuery[entry], state.visibilityCell[entry]]));
    const tiles = [...new Set(entries.map((entry) => Math.floor(state.visibilityCell[entry] / capacity.numCellsPerTile)))];
    const slotOfTile = new Map(tiles.map((tile, slot) => [tile, slot]));
    const direct = await readTiles(device, cache, tiles, false);
    const indirect = await readTiles(device, cache, tiles, true);
    const tally = { picksOff: 0, worstPick: 0, near: 0, nearWithIndirect: 0, nearWithoutIndirect: 0, nearWithDirect: 0, nearOff: 0, worstNear: 0, farWithDirect: 0 };
    state.rays.forEach((ray, index) => {
        const got = [0, 1, 2].map((channel) => state.radiance[index * 4 + channel]);
        const extra = misdirected.added.get(index) ?? 0;
        const cell = resolvedCellOf.get(index);
        let resolved = [0, 0, 0];
        let directCount = 0;
        let indirectCount = 0;
        if (cell !== undefined) {
            const slot = slotOfTile.get(Math.floor(cell / capacity.numCellsPerTile)) ?? 0;
            const directPick = pickLevel(direct, cache, slot, cell, MAX_SAMPLE_COUNT).value;
            const indirectPick = pickLevel(indirect, cache, slot, cell, MAX_SAMPLE_COUNT).value;
            resolved = [0, 1, 2].map((channel) => directPick[channel] / Math.max(directPick[3], 1) + indirectPick[channel] / Math.max(indirectPick[3], 1));
            directCount = directPick[3];
            indirectCount = indirectPick[3];
        }
        const expected = resolved.map((value) => value + extra);
        const pickError = Math.max(...[0, 1, 2].map((channel) => Math.abs(got[channel] - expected[channel]) / Math.max(expected[channel], 1e-2)));
        tally.worstPick = Math.max(tally.worstPick, pickError);
        if (pickError > (extra > 0 ? 0.01 : 4 * HALF_TOLERANCE)) {
            tally.picksOff++;
            if (tally.picksOff <= 3) fail(`ray ${index} reads ${got.map((value) => value.toFixed(4)).join(', ')}, its cell's levels give ${resolved.map((value) => value.toFixed(4)).join(', ')} and the misdirected additions ${extra.toFixed(4)}`);
        }
        if (cell === undefined) return;
        const tileSize = hashGridCacheGetCellSize(ray.hit, FURNACE.camera.position, constants) * LEVEL_SIDES[0];
        if (tileSize >= NEAR_TILE_SIZE) {
            if (directCount > 0) tally.farWithDirect++;
            return;
        }
        tally.near++;
        if (directCount > 0) tally.nearWithDirect++;
        const hasIndirect = indirectCount > 0;
        if (hasIndirect) tally.nearWithIndirect++;
        else tally.nearWithoutIndirect++;
        const own = got.map((value) => value - extra);
        const error = hasIndirect ? Math.max(...own.map((value) => relativeError(value, CEILING_INDIRECT))) : Math.max(...own.map(Math.abs)) / CEILING_INDIRECT;
        tally.worstNear = Math.max(tally.worstNear, error);
        if (error > (hasIndirect ? TOLERANCE : extra > 0 ? 0.01 : 1e-3)) {
            tally.nearOff++;
            if (tally.nearOff <= 3) fail(`ray ${index} (tile ${tileSize.toFixed(3)} m) reads ${got.map((value) => value.toFixed(4)).join(', ')} with ${extra.toFixed(4)} misdirected, expected ${hasIndirect ? CEILING_INDIRECT.toFixed(4) : 0}`);
        }
    });
    if (entries.length === 0) fail('no resolved ceiling rays');
    if (tally.picksOff > 0) fail(`${tally.picksOff} of ${state.rays.length} probe rays don't read their cell's levels and the misdirected additions`);
    if (tally.near === 0) fail(`no ceiling ray in a tile under ${NEAR_TILE_SIZE} m: move the camera`);
    if (tally.nearWithDirect > 0) fail(`${tally.nearWithDirect} ceiling rays in tiles under ${NEAR_TILE_SIZE} m read direct light: the ceiling has none`);
    if (tally.nearOff > 0) fail(`${tally.nearOff} of ${tally.near} ceiling rays in tiles under ${NEAR_TILE_SIZE} m are off`);
    return {
        resolvedRays: entries.length,
        bypassedRays: bypassed,
        misdirectedFromPopulateCells: misdirected.fromPopulateCells,
        misdirectedFromUpdateMultibounceCells: misdirected.fromUpdateMultibounceCells,
        ...tally,
    };
}

async function checkMultibounce(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 16) failures.push(`multibounce: ${message}`);
    };
    const cache = scenario.cache;
    const capacity = cache.capacity;
    const state = await readFrameState(device, scenario);
    const counters = state.counters;
    const hits = state.rays.filter((ray) => ray.hitT >= 0);
    const notCeiling = hits.filter((ray) => ray.surface !== 'ceiling').length;
    if (hits.length === 0) fail('no probe ray hit anything');
    if (notCeiling > 0) fail(`${notCeiling} probe rays hit something other than the ceiling`);
    const overflow = counters[HASH_GRID_COUNTER.bucketOverflowCount];
    if (overflow !== 0) fail(`${overflow} inserts refused (bucket overflow)`);
    // GI-1.2 resolves every non-emissive hit, lit or not.
    if (state.visibilityCell.length !== hits.length) fail(`${state.visibilityCell.length} first-bounce entries for ${hits.length} hits`);
    if (state.resolve.length !== hits.length) fail(`the resolve list holds ${state.resolve.length}, the hits ${hits.length}`);
    // Every shadow ray is a second-bounce hit's: the ceiling has no light.
    const firstBounceShadowRays = Array.from(state.visibilityRay).filter((entry) => (entry & ~BYPASS_CACHE_BIT) < state.visibilityCell.length).length;
    if (firstBounceShadowRays > 0) fail(`${firstBounceShadowRays} shadow rays from the unlit ceiling`);
    if (state.visibilityRay.length !== state.multibounceCell.length) fail(`${state.visibilityRay.length} shadow rays for ${state.multibounceCell.length} second-bounce hits on the sunlit floor`);
    const listed = new Set(state.updateTiles);
    if (listed.size !== state.updateTiles.length) fail(`the update list holds ${state.updateTiles.length} entries for ${listed.size} tiles (register D32)`);

    // The probe rays.
    const rays = await judgeCeilingRays(device, scenario, state, fail);

    // The cells of every tile the frame touched (the ceiling's, the floor's and the far tiles
    // that hold both): an indirect mean of CEILING_INDIRECT wherever there are indirect samples
    // (only the ceiling's first-bounce hits get them), a direct mean of L_f wherever there are
    // direct samples (only the floor's second-bounce hits get them).
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);
    const touchedTiles = [...new Set([...Array.from(state.visibilityCell, tileOf), ...Array.from(state.multibounceCell, tileOf)])];
    const directCells = await finestValues(device, cache, touchedTiles, false);
    const indirectCells = await finestValues(device, cache, touchedTiles, true);
    const judgeCells = (cells: Map<number, CellValue>, expected: number, name: string): { cells: number; off: number; worst: number } => {
        const result = { cells: 0, off: 0, worst: 0 };
        for (const [cell, value] of cells) {
            if (!(value[3] > 0)) continue;
            result.cells++;
            const error = Math.max(...[0, 1, 2].map((channel) => relativeError(value[channel] / value[3], expected)));
            result.worst = Math.max(result.worst, error);
            if (error > TOLERANCE) {
                result.off++;
                if (result.off <= 3) fail(`cell ${cell}'s ${name} value is a mean of ${(value[0] / value[3]).toFixed(4)} over ${value[3]} samples, expected ${expected.toFixed(4)}`);
            }
        }
        if (result.cells === 0) fail(`no cell has ${name} samples`);
        if (result.off > 0) fail(`${result.off} of ${result.cells} cells' ${name} means are more than 2% off`);
        return result;
    };
    const indirectTally = judgeCells(indirectCells, CEILING_INDIRECT, 'indirect');
    // The measured indirect value over the true second bounce: Capsaicin's factor, 1 / (1 - p).
    let indirectMeanSum = 0;
    for (const value of indirectCells.values()) if (value[3] > 0) indirectMeanSum += value[0] / value[3];
    const measuredOverTrueSecondBounce = indirectMeanSum / Math.max(indirectTally.cells, 1) / TRUE_SECOND_BOUNCE;
    const directTally = judgeCells(directCells, FLOOR_RADIANCE, 'direct');
    // The first-bounce cells with direct samples: those of far tiles that hold the floor too.
    const ceilingCellsWithDirect = new Set(Array.from(state.visibilityCell).filter((cell) => (directCells.get(cell)?.[3] ?? 0) > 0)).size;

    // The draws, against the CPU twin: which first-bounce entries trace a second ray, and where to.
    const frameIndex = await lastFrameIndex(device, cache, state.visibilityCell[0]);
    const entryOf = new Map<number, number>();
    let duplicateQueries = 0;
    state.multibounceQuery.forEach((query, entry) => {
        if (entryOf.has(query)) duplicateQueries++;
        entryOf.set(query, entry);
    });
    if (duplicateQueries > 0) fail(`${duplicateQueries} first-bounce entries have two second-bounce entries`);
    let kept = 0;
    let discardedWithEntry = 0;
    let keptMisses = 0;
    let keptMissesWithEntry = 0;
    let keptHitsWithoutEntry = 0;
    let undecided = 0;
    let directionsOff = 0;
    let worstDirection = 0;
    let brdfOff = 0;
    let worstBrdf = 0;
    for (let did = 0; did < state.visibilityCell.length; did++) {
        const draw = multibounceDraw(did, frameIndex, DISCARD_PROBABILITY);
        const entry = entryOf.get(did);
        if (draw.discarded) {
            if (entry !== undefined) discardedWithEntry++;
            continue;
        }
        kept++;
        const sample = sampleBRDFAndEvaluateDiffuse([FURNACE.albedo, FURNACE.albedo, FURNACE.albedo], draw.samples, CEILING_NORMAL);
        const start = state.rays[state.visibilityQuery[did]].hit;
        // Down to the floor, unless it leaves the planes first (or runs along them).
        const t = sample.direction[1] < 0 ? FURNACE.ceilingY / -sample.direction[1] : Infinity;
        const landing = [start[0] + t * sample.direction[0], start[2] + t * sample.direction[2]];
        const margin = FURNACE.halfWidth - Math.max(Math.abs(landing[0]), Math.abs(landing[1]));
        if (Math.abs(margin) < 0.5) {
            undecided++;
            continue;
        }
        if (margin < 0) {
            keptMisses++;
            if (entry !== undefined) keptMissesWithEntry++;
            continue;
        }
        if (entry === undefined) {
            keptHitsWithoutEntry++;
            continue;
        }
        const direction = multibounceDirection(state, entry);
        const directionError = Math.max(...[0, 1, 2].map((axis) => Math.abs(direction[axis] - sample.direction[axis])));
        worstDirection = Math.max(worstDirection, directionError);
        if (directionError > 2e-3) {
            directionsOff++;
            if (directionsOff <= 3) fail(`entry ${did}: the GPU traced (${direction.map((value) => value.toFixed(4)).join(', ')}), the CPU (${sample.direction.map((value) => value.toFixed(4)).join(', ')})`);
        }
        // BRDF / pdf is the albedo, the pdf cos / pi (each a half).
        const info = state.multibounceInfo.slice(entry * 4, entry * 4 + 4);
        const cosine = -direction[1];
        const brdfError = Math.max(
            ...[0, 1, 2].map((channel) => relativeError(info[channel] / info[3], FURNACE.albedo)),
            Math.abs(info[3] - cosine / Math.PI) / Math.max(cosine / Math.PI, 1e-3),
        );
        worstBrdf = Math.max(worstBrdf, brdfError);
        if (brdfError > 4e-3) {
            brdfOff++;
            if (brdfOff <= 3) fail(`entry ${did}: BRDF ${Array.from(info.slice(0, 3), (value) => value.toFixed(5)).join(', ')} and pdf ${info[3].toFixed(5)} for a cosine of ${cosine.toFixed(4)}`);
        }
    }
    if (discardedWithEntry > 0) fail(`${discardedWithEntry} discarded first-bounce entries traced a second ray`);
    if (keptMissesWithEntry > 0) fail(`${keptMissesWithEntry} second rays the CPU sees leave the planes hit something`);
    if (keptHitsWithoutEntry > 0) fail(`${keptHitsWithoutEntry} kept first-bounce entries have no second-bounce entry`);
    if (directionsOff > 0) fail(`${directionsOff} second-bounce directions differ from the CPU's`);
    if (brdfOff > 0) fail(`${brdfOff} second-bounce entries' BRDF or pdf is off`);
    const keptShare = kept / Math.max(state.visibilityCell.length, 1);
    // Binomial: at 30% of tens of thousands of entries the share is within a fraction of a percent.
    if (Math.abs(keptShare - (1 - DISCARD_PROBABILITY)) > 0.02) fail(`${(keptShare * 100).toFixed(2)}% of the first-bounce entries were kept, expected about 30%`);

    return {
        failures,
        data: {
            trueSecondBounce: TRUE_SECOND_BOUNCE,
            capsaicinFactor: CAPSAICIN_FACTOR,
            expectedCeilingIndirect: CEILING_INDIRECT,
            measuredOverTrueSecondBounce,
            expectedFloorRadiance: FLOOR_RADIANCE,
            frameIndex,
            hits: hits.length,
            secondBounceEntries: state.multibounceCell.length,
            keptShare,
            keptMisses,
            undecidedLandings: undecided,
            worstDirectionError: worstDirection,
            worstBrdfError: worstBrdf,
            rays,
            cellsWithIndirect: indirectTally,
            cellsWithDirect: directTally,
            ceilingCellsWithDirect,
            bucketOverflow: overflow,
        },
    };
}

async function checkLag(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 16) failures.push(`lag: ${message}`);
    };
    const cache = scenario.cache;
    const capacity = cache.capacity;
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);

    // Started with multibounce off: turning it on empties the cache and its indirect scratch.
    scenario.technique.settings.useMultibounce = true;
    await runFrames(device, scenario, 1);
    const first = await readFrameState(device, scenario);
    if (first.multibounceCell.length === 0) fail('no second-bounce entry in the first frame');

    // UpdateTiles ran before UpdateMultibounceCells: no indirect value yet, so the rays read 0.
    const ceilingTiles = [...new Set(Array.from(first.visibilityCell, tileOf))];
    const indirectValues = await readTiles(device, cache, ceilingTiles, true);
    let indirectSamples = 0;
    for (let index = 3; index < indirectValues.length; index += 4) if (indirectValues[index] > 0) indirectSamples++;
    if (indirectSamples > 0) fail(`${indirectSamples} ceiling cells (any level) have an indirect sample after the first frame`);
    // The rays read what the cells hold: in tiles under 2 m, nothing yet.
    const firstRays = await judgeCeilingRays(device, scenario, first, fail);
    if (firstRays.nearWithIndirect > 0) fail(`${firstRays.nearWithIndirect} rays read an indirect sample after the first frame`);

    // ...but the scratch holds this frame's samples for next frame: one count per second-bounce
    // entry whose first bounce is in the cell, and their quantised sum.
    const expectedCounts = new Map<number, number>();
    for (const query of first.multibounceQuery) {
        const cell = first.visibilityCell[query];
        expectedCounts.set(cell, (expectedCounts.get(cell) ?? 0) + 1);
    }
    const cells = [...expectedCounts.keys()];
    const staging = device.createBuffer({ label: 'world-cache indirect scratch', size: Math.max(16, cells.length * 16), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'world-cache indirect scratch' });
    const scratchBase = cache.layout.updateScratch.updateCellValueIndirectBuffer * 4;
    cells.forEach((cell, index) => encoder.copyBufferToBuffer(cache.buffers.updateScratch, scratchBase + cell * 16, staging, index * 16, 16));
    device.queue.submit([encoder.finish()]);
    const scratch = new Uint32Array(await readRange(device, staging, 0, cells.length * 16));
    staging.destroy();
    let countsOff = 0;
    let meansOff = 0;
    let worstMean = 0;
    cells.forEach((cell, index) => {
        const count = scratch[index * 4 + 3];
        const expected = expectedCounts.get(cell) ?? 0;
        if (count !== expected) {
            countsOff++;
            if (countsOff <= 3) fail(`cell ${cell}'s indirect scratch counts ${count}, its second-bounce entries ${expected}`);
        }
        // Quantised at 1/1000 (kHashGridCache_FloatQuantize).
        const mean = scratch[index * 4] / 1000 / Math.max(count, 1);
        const error = relativeError(mean, CEILING_INDIRECT);
        worstMean = Math.max(worstMean, error);
        if (error > TOLERANCE) {
            meansOff++;
            if (meansOff <= 3) fail(`cell ${cell}'s indirect scratch averages ${mean.toFixed(4)}, expected ${CEILING_INDIRECT.toFixed(4)}`);
        }
    });
    if (countsOff > 0) fail(`${countsOff} of ${cells.length} cells' scratch counts differ from their second-bounce entries`);
    if (meansOff > 0) fail(`${meansOff} of ${cells.length} cells' scratch averages are more than 2% off`);

    // The second frame: UpdateTiles blends the first frame's samples in, and the rays read them.
    await runFrames(device, scenario, 1);
    const second = await readFrameState(device, scenario);
    const rays = await judgeCeilingRays(device, scenario, second, fail);
    if (rays.nearWithIndirect === 0) fail('no ray reads an indirect sample after the second frame');

    return {
        failures,
        data: {
            firstFrame: { secondBounceEntries: first.multibounceCell.length, ceilingCellsWithIndirect: indirectSamples, rays: firstRays, scratchCells: cells.length, worstScratchMeanError: worstMean },
            secondFrame: rays,
        },
    };
}

async function checkMultibounceOff(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`multibounce off: ${message}`);
    };
    const cache = scenario.cache;
    const state = await readFrameState(device, scenario);
    const hits = state.rays.filter((ray) => ray.hitT >= 0);
    if (hits.length === 0) fail('no probe ray hit anything');
    if (state.visibilityCell.length !== hits.length) fail(`${state.visibilityCell.length} first-bounce entries for ${hits.length} hits`);
    if (state.multibounceCell.length > 0) fail(`${state.multibounceCell.length} second-bounce entries`);
    // Capsaicin resolves its shadow-ray list without multibounce: the unlit ceiling has neither.
    if (state.visibilityRay.length > 0) fail(`${state.visibilityRay.length} shadow rays from the unlit ceiling`);
    if (state.resolve.length > 0) fail(`${state.resolve.length} resolve entries without a light (register D35)`);
    let raysLit = 0;
    state.rays.forEach((_, index) => {
        if (Math.max(state.radiance[index * 4], state.radiance[index * 4 + 1], state.radiance[index * 4 + 2]) > 0) raysLit++;
    });
    if (raysLit > 0) fail(`${raysLit} probe rays read light: there is none without the second bounce`);
    const tiles = [...new Set(Array.from(state.visibilityCell, (cell) => Math.floor(cell / cache.capacity.numCellsPerTile)))];
    const indirectValues = await readTiles(device, cache, tiles, true);
    let indirectSamples = 0;
    for (let index = 3; index < indirectValues.length; index += 4) if (indirectValues[index] > 0) indirectSamples++;
    if (indirectSamples > 0) fail(`${indirectSamples} ceiling cells have indirect samples`);
    return { failures, data: { hits: hits.length, secondBounceEntries: state.multibounceCell.length, shadowRays: state.visibilityRay.length, resolveEntries: state.resolve.length, raysLit, indirectSamples } };
}

// ---- Section 11: Capsaicin's misdirected bypass additions where both paths fire -------------------

/**
 * GIDenoiser_RemoveNaNs on one finite non-negative channel, as the WGSL computes it: c / (1 + c),
 * clamped, mapped back. Close to the identity below about 1e4.
 */
function removeNaNs(value: number): number {
    const mapped = Math.min(Math.max(Math.fround(value / (1 + value)), 0), 1);
    return Math.fround(mapped / Math.max(1 - mapped, 1e-4));
}

async function checkCornerMisdirected(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`corner misdirected: ${message}`);
    };
    const cache = scenario.cache;
    const capacity = cache.capacity;
    const state = await readFrameState(device, scenario);
    const firstCount = state.visibilityCell.length;
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);

    // PopulateCells: a second-bounce shadow ray with the bypass bit adds its light, which is
    // albedo / pi x E where the sun reaches the hit and nothing (a zero isn't added) where the
    // wall shades it, to the probe ray numbered as its first-bounce entry.
    const populateCellsTargets = new Set<number>();
    for (const entry of state.visibilityRay) {
        const visibility = entry & ~BYPASS_CACHE_BIT;
        if ((entry & BYPASS_CACHE_BIT) !== 0 && visibility >= firstCount) populateCellsTargets.add(state.multibounceQuery[visibility - firstCount]);
    }
    // UpdateMultibounceCells: where the shadow-ray list's word at the entry's visibility index has
    // the bypass bit, the entry's sample goes to that probe ray too. Its value is known exactly:
    // the second cell's filtered direct (the levels as read back: nothing writes them after) x
    // BRDF / pdf / (1 - p).
    const updateEntries: number[] = [];
    state.multibounceQuery.forEach((_, entry) => {
        if ((state.visibilityRayWords[firstCount + entry] & BYPASS_CACHE_BIT) !== 0) updateEntries.push(entry);
    });

    const resolved = Array.from(state.resolve).filter((entry) => (entry & BYPASS_CACHE_BIT) === 0);
    const bypassedRays = new Set(Array.from(state.resolve).filter((entry) => (entry & BYPASS_CACHE_BIT) !== 0).map((entry) => state.visibilityQuery[entry & ~BYPASS_CACHE_BIT]));
    const resolvedCellOf = new Map(resolved.map((entry) => [state.visibilityQuery[entry], state.visibilityCell[entry]]));
    const tiles = [...new Set([...resolved.map((entry) => tileOf(state.visibilityCell[entry])), ...updateEntries.map((entry) => tileOf(state.multibounceCell[entry]))])];
    const slotOfTile = new Map(tiles.map((tile, slot) => [tile, slot]));
    const direct = await readTiles(device, cache, tiles, false);
    const indirect = await readTiles(device, cache, tiles, true);
    const levelsOf = (cell: number, values: Float32Array): CellValue => pickLevel(values, cache, slotOfTile.get(tileOf(cell)) ?? 0, cell, MAX_SAMPLE_COUNT).value;

    const updateAdds = new Map<number, number[]>();
    for (const entry of updateEntries) {
        const value = levelsOf(state.multibounceCell[entry], direct);
        const info = state.multibounceInfo.slice(entry * 4, entry * 4 + 4);
        const sample = [0, 1, 2].map((channel) => removeNaNs(((value[channel] / Math.max(value[3], 1)) * info[channel]) / info[3] / (1 - DISCARD_PROBABILITY)));
        if (Math.max(...sample) > 0) updateAdds.set(state.multibounceQuery[entry], sample);
    }

    const tally = { judged: 0, skippedBypassed: 0, off: 0, worst: 0, populateCellsLit: 0, populateCellsNothing: 0, updateMultibounceCellsAdds: 0 };
    state.rays.forEach((ray, index) => {
        if (bypassedRays.has(index)) {
            tally.skippedBypassed++;
            return;
        }
        const cell = resolvedCellOf.get(index);
        let base = [0, 0, 0];
        if (cell !== undefined) {
            const d = levelsOf(cell, direct);
            const i = levelsOf(cell, indirect);
            base = [0, 1, 2].map((channel) => d[channel] / Math.max(d[3], 1) + i[channel] / Math.max(i[3], 1));
        }
        const update = updateAdds.get(index) ?? [0, 0, 0];
        if (updateAdds.has(index)) tally.updateMultibounceCellsAdds++;
        const got = [0, 1, 2].map((channel) => state.radiance[index * 4 + channel]);
        const residual = [0, 1, 2].map((channel) => got[channel] - base[channel] - update[channel]);
        const scale = Math.max(...got, 1e-2);
        const tolerance = updateAdds.has(index) || populateCellsTargets.has(index) ? 0.01 : 4 * HALF_TOLERANCE;
        const nothing = Math.max(...residual.map(Math.abs)) / scale;
        const lit = Math.max(...residual.map((value) => Math.abs(value - LIT_RADIANCE))) / scale;
        let error = nothing;
        if (populateCellsTargets.has(index)) {
            error = Math.min(nothing, lit);
            if (lit < nothing) tally.populateCellsLit++;
            else tally.populateCellsNothing++;
        }
        tally.judged++;
        tally.worst = Math.max(tally.worst, error);
        if (error > tolerance) {
            tally.off++;
            if (tally.off <= 3) fail(`ray ${index} (${ray.surface}) reads ${got.map((value) => value.toFixed(4)).join(', ')}: its levels give ${base.map((value) => value.toFixed(4)).join(', ')}, UpdateMultibounceCells' addition ${update[0].toFixed(4)}${populateCellsTargets.has(index) ? `, PopulateCells' 0 or ${LIT_RADIANCE.toFixed(4)}` : ''}`);
        }
    });
    if (tally.off > 0) fail(`${tally.off} of ${tally.judged} probe rays don't read their levels and the predicted additions`);
    if (tally.populateCellsLit === 0) fail("PopulateCells' misdirected addition never fired with light: the check can't see it");
    if (tally.updateMultibounceCellsAdds === 0) fail("UpdateMultibounceCells' misdirected addition never fired: the check can't see it");
    return { failures, data: { firstBounceEntries: firstCount, secondBounceEntries: state.multibounceCell.length, populateCellsTargets: populateCellsTargets.size, ...tally } };
}

// ---- Section 12: emptied when a cell size changes ---------------------------------------------------

async function checkClearOnChange(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`cleared on change: ${message}`);
    };
    const capacity = scenario.cache.capacity;
    const settings = scenario.technique.settings;
    const data: Record<string, unknown> = {};
    const frameTiles = async (): Promise<{ live: number[]; touched: Set<number> }> => {
        const state = await readFrameState(device, scenario);
        return { live: Array.from(state.packedTiles), touched: new Set(Array.from(state.visibilityCell, (cell) => Math.floor(cell / capacity.numCellsPerTile))) };
    };
    // A frame with nothing changed: tiles touched in earlier frames but not this one stay live.
    await runFrames(device, scenario, 1);
    const steady = await frameTiles();
    data.unchanged = { liveTiles: steady.live.length, touchedTiles: steady.touched.size };
    if (!(steady.live.length > steady.touched.size)) fail(`with nothing changed ${steady.live.length} tiles are live and ${steady.touched.size} touched: the check can't tell an emptied cache`);
    const changes: [string, () => void][] = [
        ['hashGridCacheCellSize', () => (settings.hashGridCacheCellSize = 16)],
        ['hashGridCacheMinCellSize', () => (settings.hashGridCacheMinCellSize = 0.2)],
    ];
    for (const [name, change] of changes) {
        // A few frames at the setting, then the change and one frame: only that frame's tiles live.
        await runFrames(device, scenario, 3);
        change();
        await runFrames(device, scenario, 1);
        const after = await frameTiles();
        const stale = after.live.filter((tile) => !after.touched.has(tile)).length;
        data[name] = { liveTiles: after.live.length, touchedTiles: after.touched.size, staleTiles: stale };
        if (stale > 0 || after.live.length !== after.touched.size) fail(`after ${name} changed, ${after.live.length} tiles are live, ${after.touched.size} touched and ${stale} left from before (Capsaicin empties the cache, gi1.cpp:1740-1745)`);
    }
    return { failures, data };
}

// ---- The page --------------------------------------------------------------------------------------

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    // As a host would ask (requestGIDevice): the adapter's storage limits, and its optional features.
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    releaseDeviceOnPageHide(device);
    deviceErrors = watchDevice(device);

    const failures: string[] = [];
    const data: Record<string, unknown> = {};
    const run = async (
        name: string,
        lights: readonly LightDescriptor[],
        extras: CornerExtras,
        frames: number,
        check: (device: GPUDevice, scenario: Scenario) => Promise<Checks>,
        settings: Partial<RadianceCacheSettings> = SINGLE_BOUNCE,
    ): Promise<void> => {
        const scenario = await startScenario(device, name, lights, extras, settings);
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

    await run('litPlane', [sun(SUN_TRAVEL)], {}, CONVERGENCE_FRAMES, checkLitPlane);
    await run('resolve', [{ type: 'point', position: [0, 1.5, 2], range: 0, color: [1, 1, 1], intensity: 10, label: 'point' }], {}, 1, checkResolve);
    await run('emissive', [sun(SUN_TRAVEL)], { patch: true }, 1, checkEmissive);
    await run('twoSuns', [sun(SUN_TRAVEL, [1, 0, 0]), sun(BLUE_SUN_TRAVEL, [0, 0, 1])], {}, CONVERGENCE_FRAMES, checkTwoSuns);
    await run('shadow', [sun(SUN_TRAVEL)], { slab: true }, CONVERGENCE_FRAMES, checkShadow);
    // Converged at SUN_INTENSITY by the run; checkBlend changes the sun and runs one more frame.
    await run('blend', [sun(SUN_TRAVEL)], {}, CONVERGENCE_FRAMES, checkBlend);

    // GI-1.2's multibounce (RC3.3) in the furnace.
    await run('multibounce', [FURNACE_SUN], { furnace: true }, CONVERGENCE_FRAMES, checkMultibounce, MULTIBOUNCE);
    // Started with multibounce off; checkLag turns it on and runs its two frames.
    await run('multibounceLag', [FURNACE_SUN], { furnace: true }, 0, checkLag);
    await run('multibounceOff', [FURNACE_SUN], { furnace: true }, CONVERGENCE_FRAMES, checkMultibounceOff);
    await run('cornerMisdirected', [sun(SUN_TRAVEL)], {}, CONVERGENCE_FRAMES, checkCornerMisdirected, MULTIBOUNCE);
    await run('clearOnChange', [sun(SUN_TRAVEL)], {}, 10, checkClearOnChange);

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
