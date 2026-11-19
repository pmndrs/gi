/**
 * RC3.2, RC3.3, RC3.4 and RC3.5: the radiance cache's world cache (Capsaicin's
 * hash grid) populated and resolved through the technique, on the GPU; GI-1.2's
 * multibounce, with Capsaicin's errors in it fixed (RC3.5); and the sky as one
 * of a hit's lights (RC3.4).
 *
 * Sections 1 to 7 are GI-1.0's single bounce: they run with multibounce off
 * (`useMultibounce: false`), and pass as they did in RC3.2 (the single-bounce
 * path is unchanged by RC3.3). Sections 8 to 11 and 13 are the multibounce.
 * Sections 1 to 14 have a black sky, so the sky light (RC3.4) is never drawn
 * there (its weight is 0); sections 15 to 18 are under a sky.
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
 *   ray down to the floor; each kept ray adds L_f × BRDF / pdf = ρ L_f
 *   (a miss, about 1 in 10,000 here, adds a sample of no light), and a cell
 *   averages the samples it gets, so a ceiling cell's indirect value, and
 *   every probe ray reading it, is the second bounce ρ² E / π = 0.3183
 *   (RC3.5, register D37; Capsaicin's estimate read 1 / 0.3 = 3.33 times
 *   that, 1.0610).
 *
 * Far from the camera a tile is taller than the 3 m gap, and nearly level
 * rays up and down share a direction bin, so a far tile holds floor and
 * ceiling hits in the same cells (NOTES, "a hash-grid cell is a column
 * through its tile"): there a ceiling ray also reads the floor's direct
 * light. The checks allow for it where tiles reach 2 m.
 *
 * A probe ray's expected reading (`predictProbeRays`) is its cell's levels,
 * or its own sample where it bypasses the cache, plus the one addition
 * `UpdateMultibounceCells` makes since RC3.5 (register D38): a bypassing
 * first bounce's own probe ray gets its second-bounce draw over 1 - p.
 * Capsaicin's two misdirected additions are gone, so a ray that read one
 * would be off.
 *
 * 8. **Multibounce, converged** (40 frames): every probe ray reads its
 *    prediction, and where its tile is under 2 m (it can't reach the floor)
 *    that is no direct light and an indirect 0.3183 within 2% (up to 1% of
 *    the rays and cells may hold one of the rare misses), averaging 0.3183
 *    within 1%; the cells' indirect means average the true second bounce
 *    within 1%, and every cell with direct samples (the floor's, from the
 *    second-bounce hits) has a direct mean of L_f within 2%. The last
 *    frame's draws match the CPU twin (`multibounce-cpu.ts`): the
 *    first-bounce entries whose `rand()` is above 0.7 are exactly those with
 *    a second-bounce entry, a miss where the CPU sees the ray leave the
 *    planes, each direction is the CPU's within half precision, its BRDF /
 *    pdf is ρ and its pdf cos / π. The update list holds each tile once
 *    (register D32).
 * 9. **The one-frame lag** (multibounce turned on after two frames with it
 *    off, which empties the cache and its indirect scratch; then one frame,
 *    then a second): after the first frame no ceiling cell has an indirect
 *    sample and the ceiling rays under 2 m tiles read 0, while each ceiling
 *    cell's indirect scratch holds one count per second-bounce entry whose
 *    first bounce is in it, and the quantised sum of their samples (ρ L_f a
 *    hit, 0 a miss); after the second frame those rays read 0.3183.
 * 10. **Multibounce off** (40 frames): no second-bounce entry, no shadow ray
 *    and no resolve entry (Capsaicin resolves its shadow-ray list without
 *    multibounce, register D35), and every probe ray reads 0.
 * 11. **The bypass additions, predicted** (section 6's corner with the
 *    slab, multibounce on, 40 frames): every probe ray reads its prediction,
 *    a self-bypassed ray its own sun sample (0 in the slab's shade) plus its
 *    first bounce's draw over 1 - p, that draw from the second hit's own light
 *    where its segment bypasses (near the shade's edge that differs from its
 *    cell's); the first-bounce bypass flags match the resolve list, every
 *    second hit's own light is 0 or albedo / π × E, and both kinds of
 *    addition fire.
 * 12. **Emptied when a cell size changes** (the corner, single bounce): after
 *    a frame with `hashGridCacheCellSize`, then `hashGridCacheMinCellSize`,
 *    changed, the live tiles are exactly the frame's (gi1.cpp:1740-1745);
 *    a frame with nothing changed keeps more.
 * 13. **An open scene** (the furnace with its floor as strips 5 cm wide every
 *    10 cm, 40 frames): half of a near ceiling cell's kept second-bounce rays
 *    hit (the CPU lands the recorded rays and agrees which), so its indirect
 *    value is the second bounce ρ L_f × 1/2 = 0.1592: the near ceiling cells
 *    and rays average it within 2% (one cell is a noisy mean of hits and
 *    misses, so only the averages are judged); a mean over the hits alone, as
 *    Capsaicin's, would read 0.3183. A few probes on the strips start just
 *    below them (the depth's precision at a grazing view, register D17) and
 *    see their undersides; they are left out of the ceiling averages.
 *
 * Sections 15 to 18 (RC3.4) are the sky as one of a hit's lights (register
 * D49): no sun unless named, multibounce on, the sky the only light, so every
 * reservoir (first and second bounce) must draw the sky with weight 1. Each
 * cell's direct value is judged against the mean, over this frame's entries
 * in it, of ρ L (1 − F), F the share of the point's cosine-weighted sky its
 * occluder hides (Lambert's contour integral over the occluding polygon, exact;
 * checked against the infinite strip's closed form), within 2% plus five
 * standard deviations of its samples' mean (a sample is ρ L or 0 by its
 * shadow ray), 0.1% of cells allowed past it; the means are held to 0.5-1%.
 * ρ = 0.5, L = 1.5. Expected, before the run:
 *
 * 15. **The open floor** (`SkyScene` openFloor: the 400 m floor, a 4 m probe
 *    patch 50 m up): every floor cell and the floor's probe rays read ρ L =
 *    0.75 (F ≤ 0.2%); the second bounce goes up into the sky, and its misses
 *    add no light, so the floor's indirect values stay under 1% of ρ L (a miss
 *    reading the sky would add ρ L again).
 * 16. **The coded sky and a sun** (the same scene; the sky (1 + d) / 2 per
 *    channel × 2, the sun of section 1 casting no shadow): on the floor the sky
 *    weighs the luminance of its irradiance, S (π/2, 5π/6, π/2), the sun its
 *    E = 8 cos 45°, so the sky is drawn for 45.06% of hits, each weighed 1 / q;
 *    the floor cells average ρ / π (E_sky + E_sun) per channel within 1% (the
 *    sky read along the normal instead of the drawn direction would put
 *    green's sky part 20% high; a wrong weight shows in the share).
 * 17. **The ceiling only** (the done-when): the 400 m ceiling 3 m up, a black
 *    1 m probe patch on the ground: the ceiling's cells read ρ L (1 − F), F up
 *    to 3.5% over the patch; its second-bounce rays go down and nearly all
 *    miss (the rest hit the black patch), so every ceiling cell's indirect
 *    value is exactly 0 with samples counted. The sky reaches the ceiling
 *    through its own light sample; a miss reading it would count it twice.
 * 18. **The corner** (sections 1-7's floor and 40 × 10 m wall): floor cells
 *    ρ L (1 − F_wall), wall cells ρ L (1 − F_floor), the cells that see half
 *    their sky hidden within 2% on average (so the shadow ray is traced); the
 *    second bounce near the corner (each kept ray brings the other surface's
 *    direct value × ρ, a CPU quadrature over cosine-weighted directions,
 *    tabulated) summed over every near cell within 3% of its sum, a surface.
 *
 * Correctness only, no timing (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc3.2 world cache" -- pnpm test:gpu --filter radiance-cache-world-cache --gpu nvidia` (or intel).
 */

import type { LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { EnvironmentDescriptor, LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/ddgi-core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DIRECT_SKY_LIGHT } from '../../../../packages/ddgi-core/src/platform/shade/wgsl.ts';
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
import {
    BYPASS_CACHE_BIT,
    MULTIBOUNCE_DIRECT_WORDS,
    MULTIBOUNCE_INFO_WORDS,
    MULTIBOUNCE_MISS_CELL,
    MULTIBOUNCE_RAY_WORDS,
    RESERVOIR_WORDS,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/hash-grid-population-wgsl.ts';
import {
    multibounceBypassAddition,
    multibounceDraw,
    multibounceSample,
    multibounceSettledIndirect,
    sampleBRDFAndEvaluateDiffuse,
} from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/multibounce-cpu.ts';
import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { WORLD_CACHE_PROBE_RAY_BYPASS, WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';
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

/**
 * Section 11's second slab, 25 cm over the floor in front of the wall: rays from the floor under
 * it reach its underside (a back face the sun never lights) in less than a cell, so a bypassing
 * second hit with no light is there. Its shadow along (0, -1, -1) covers the floor for z in
 * [0.25, 2.25].
 */
const LOW_SLAB = { minX: -2, maxX: 2, minZ: 0.5, maxZ: 2.5, y: 0.25 } as const;

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
 * Capsaicin's factor on it, 1 / (1 - p) (its estimate averaged kept rays and still divided by
 * 1 - p), fixed in RC3.5 (register D37); reported beside the measured ratio, now 1.
 */
const CAPSAICIN_FACTOR = 1 / (1 - DISCARD_PROBABILITY);
/** A ceiling cell's settled indirect value: the second bounce itself (`multibounceSettledIndirect`, every kept ray hits). */
const CEILING_INDIRECT = multibounceSettledIndirect(FURNACE.albedo, FLOOR_RADIANCE);
/**
 * Section 13's floor: strips 5 cm wide every 10 cm (along x, 200 m long, over 200 m), so half
 * the floor is there. A ray from the ceiling 3 m above lands spread over metres, many periods,
 * so it hits a strip with probability 1/2 wherever it starts.
 */
const STRIPS = { width: 0.05, period: 0.1, halfLength: 100, fill: 0.5 } as const;
/** A ceiling cell's settled indirect value over the strips: the second bounce, ρ L_f × 1/2. */
const STRIPS_INDIRECT = multibounceSettledIndirect(FURNACE.albedo, FLOOR_RADIANCE, STRIPS.fill);
/** The ceiling's normal on the side the probe rays hit: down. */
const CEILING_NORMAL: Vec3 = [0, -1, 0];

type Surface = 'floor' | 'wall' | 'patch' | 'slab' | 'ceiling' | 'other';

interface ProbeRay {
    readonly origin: Vec3;
    readonly direction: Vec3;
    readonly hitT: number;
    readonly hit: Vec3;
    /** The hit triangle in its instance's mesh (`WorldCacheProbeRay.hitTriangle`). */
    readonly triangle: number;
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
    /** Section 11's low slab. */
    readonly lowSlab?: boolean;
    /** Sections 8-10: the furnace (floor and ceiling) instead of the corner, seen from its own camera. */
    readonly furnace?: boolean;
    /** Section 13: the furnace's floor as strips covering half of it. */
    readonly strips?: boolean;
    /** Sections 15-18 (RC3.4): a scene under a sky (`SkyScene`), instead of the corner or the furnace. */
    readonly sky?: SkyScene;
    /** Section 19: the corrugated floor under its probe patch, instead of the corner. */
    readonly corrugated?: boolean;
}

function describeCorner(platform: GIPlatformImpl, lights: readonly LightDescriptor[], extras: CornerExtras): LightHandle[] {
    if (extras.sky) return describeSkyScene(platform, lights, extras.sky);
    if (extras.corrugated) {
        corrugatedFloor(platform);
        quad(platform, 'probe patch', horizontalSquare(CORRUGATION_PATCH.half, CORRUGATION_PATCH.y, DOWN), DOWN, ALBEDO);
        const handles = lights.map((light) => platform.scene.createLight(light));
        platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
        return handles;
    }
    if (extras.furnace) return describeFurnace(platform, lights, extras.strips ?? false);
    const half = 200;
    quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], ALBEDO);
    quad(platform, 'wall', [[-20, 0, 0], [20, 0, 0], [20, 10, 0], [-20, 10, 0]], [0, 0, 1], ALBEDO);
    if (extras.patch) {
        const { minX, maxX, minZ, maxZ, y } = PATCH;
        quad(platform, 'emissive patch', [[minX, y, maxZ], [maxX, y, maxZ], [maxX, y, minZ], [minX, y, minZ]], [0, 1, 0], ALBEDO, EMISSION);
    }
    for (const [present, slab, label] of [[extras.slab, SLAB, 'slab'], [extras.lowSlab, LOW_SLAB, 'low slab']] as const) {
        if (!present) continue;
        const { minX, maxX, minZ, maxZ, y } = slab;
        quad(platform, label, [[minX, y, maxZ], [maxX, y, maxZ], [maxX, y, minZ], [minX, y, minZ]], [0, 1, 0], ALBEDO);
    }
    const handles = lights.map((light) => platform.scene.createLight(light));
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    return handles;
}

/** The furnace: a floor facing up and a ceiling facing down, 400 m wide, under a black sky. */
function describeFurnace(platform: GIPlatformImpl, lights: readonly LightDescriptor[], strips: boolean): LightHandle[] {
    const half = FURNACE.halfWidth;
    const y = FURNACE.ceilingY;
    if (strips) stripFloor(platform);
    else quad(platform, 'floor', [[-half, 0, half], [half, 0, half], [half, 0, -half], [-half, 0, -half]], [0, 1, 0], FURNACE.albedo);
    quad(platform, 'ceiling', [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]], [0, -1, 0], FURNACE.albedo);
    const handles = lights.map((light) => platform.scene.createLight(light));
    platform.scene.setEnvironment({ type: 'color', color: [0, 0, 0], intensity: 1 });
    return handles;
}

/** Section 13's floor: one mesh of strips (STRIPS), facing up. */
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
    const camera = extras.corrugated ? CORRUGATION_CAMERA : extras.sky ? SKY_LAYOUT_CAMERA[extras.sky.layout] : extras.furnace ? FURNACE.camera : CAMERA;
    // Sections 15-18: the sky, after describeCorner's black one.
    const skyTexture = extras.sky?.environment === 'coded' ? codedSkyTexture(device) : null;
    if (extras.sky) platform.scene.setEnvironment(skyEnvironment(extras.sky, skyTexture));
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
        // The open floor and the ceiling-only scene are horizontal planes too: classified as the furnace's.
        furnace: (extras.furnace ?? false) || (extras.sky !== undefined && extras.sky.layout !== 'corner'),
        platform,
        lights: lightHandles,
        technique,
        cache,
        screen,
        frame,
        dispose() {
            screen.dispose();
            platform.dispose();
            skyTexture?.destroy();
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
    /** Per first-bounce entry, 1 where its probe ray bypasses the cache (register D38). */
    readonly firstBounceBypass: Uint32Array;
    /** Per second-bounce entry, its own direct light where its segment bypasses (four floats: rgb, 0). */
    readonly multibounceDirect: Float32Array;
    readonly visibilityRay: Uint32Array;
    readonly resolve: Uint32Array;
    readonly reservoirs: Uint32Array;
    readonly updateTiles: Uint32Array;
    readonly packedTiles: Uint32Array;
}

function classify(hit: Vec3, furnace: boolean): Surface {
    if (furnace) return Math.abs(hit[1] - FURNACE.ceilingY) < 1e-3 ? 'ceiling' : Math.abs(hit[1]) < 1e-3 ? 'floor' : 'other';
    if (Math.abs(hit[1] - LOW_SLAB.y) < 1e-3 && hit[0] >= LOW_SLAB.minX - 1e-3 && hit[0] <= LOW_SLAB.maxX + 1e-3 && hit[2] >= LOW_SLAB.minZ - 1e-3 && hit[2] <= LOW_SLAB.maxZ + 1e-3) return 'slab';
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
        rays.push({ origin, direction, hitT, hit, triangle: rayWords[base + 8], frontFace: rayWords[base + 9] !== 0, surface: hitT >= 0 ? classify(hit, scenario.furnace) : 'other' });
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
        firstBounceBypass: words.slice(layout.firstBounceBypassBuffer, layout.firstBounceBypassBuffer + count0),
        multibounceDirect: unpackHalves(words.slice(layout.multibounceDirectBuffer, layout.multibounceDirectBuffer + MULTIBOUNCE_DIRECT_WORDS * count1)),
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
function slabShadowMargin(point: Vec3, slab: { minX: number; maxX: number; minZ: number; maxZ: number; y: number } = SLAB): number {
    const rise = slab.y - point[1];
    if (rise <= 0) return -Infinity; // at or above the slab: nothing between it and the sun
    const x = point[0];
    const z = point[2] + rise;
    const insideX = Math.min(x - slab.minX, slab.maxX - x);
    const insideZ = Math.min(z - slab.minZ, slab.maxZ - z);
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
 * GIDenoiser_RemoveNaNs on one finite non-negative channel, as the WGSL computes it: c / (1 + c),
 * clamped, mapped back. Close to the identity below about 1e4.
 */
function removeNaNs(value: number): number {
    const mapped = Math.min(Math.max(Math.fround(value / (1 + value)), 0), 1);
    return Math.fround(mapped / Math.max(1 - mapped, 1e-4));
}

/** What every probe ray should read this frame, from the cells as read back and the lists. */
interface ProbeRayPrediction {
    /** rgb; null where the page can't say (a self-bypassed ray whose own light isn't known). */
    readonly expected: (number[] | null)[];
    /** The cell a resolved ray reads, by probe ray. */
    readonly resolvedCellOf: Map<number, number>;
    /** Sample counts of the levels a resolved ray reads, by probe ray: [direct, indirect]. */
    readonly counts: Map<number, [number, number]>;
    /** The brightest channel of the direct level's mean a resolved ray reads, by probe ray. */
    readonly directMeans: Map<number, number>;
    /** The bypass addition `UpdateMultibounceCells` makes to a probe ray (register D38), by probe ray. */
    readonly additions: Map<number, number[]>;
    readonly tally: Record<string, number>;
}

/**
 * Every probe ray's reading, predicted: a resolved ray reads its cell's picked direct level's
 * mean plus its picked indirect level's (`ResolveCells`); a ray that bypasses the cache reads
 * its own direct sample (`ownDirect`, the PopulateCells sample, which the page knows per
 * scene); a miss reads 0 (the black sky). On top, `UpdateMultibounceCells` adds a bypassing
 * first bounce's one draw over 1 - p to that first bounce's own probe ray (register D38): the
 * second hit's light (its cell's filtered direct, or its own light where its segment bypasses)
 * x BRDF / pdf. Nothing else adds to a probe ray: Capsaicin's two misdirected additions are
 * gone, so a ray that read any would be off.
 */
async function predictProbeRays(device: GPUDevice, scenario: Scenario, state: FrameState, ownDirect: (ray: number) => number | null): Promise<ProbeRayPrediction> {
    const cache = scenario.cache;
    const capacity = cache.capacity;
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);
    const resolved = Array.from(state.resolve).filter((entry) => (entry & BYPASS_CACHE_BIT) === 0);
    const selfBypassed = new Set(Array.from(state.resolve).filter((entry) => (entry & BYPASS_CACHE_BIT) !== 0).map((entry) => state.visibilityQuery[entry & ~BYPASS_CACHE_BIT]));
    const resolvedCellOf = new Map(resolved.map((entry) => [state.visibilityQuery[entry], state.visibilityCell[entry]]));
    // The second-bounce entries whose first bounce bypasses: their samples go to its probe ray.
    const additionEntries: number[] = [];
    state.multibounceQuery.forEach((query, entry) => {
        if (state.firstBounceBypass[query] !== 0) additionEntries.push(entry);
    });
    const secondCells = additionEntries.map((entry) => state.multibounceCell[entry]).filter((cell) => cell !== MULTIBOUNCE_MISS_CELL);
    const tiles = [...new Set([...resolved.map((entry) => tileOf(state.visibilityCell[entry])), ...secondCells.map(tileOf)])];
    const slotOfTile = new Map(tiles.map((tile, slot) => [tile, slot]));
    const direct = await readTiles(device, cache, tiles, false);
    const indirect = await readTiles(device, cache, tiles, true);
    const levelsOf = (cell: number, values: Float32Array): CellValue => pickLevel(values, cache, slotOfTile.get(tileOf(cell)) ?? 0, cell, MAX_SAMPLE_COUNT).value;

    const tally = { selfBypassedRays: selfBypassed.size, additions: 0, additionsFromOwnLight: 0, additionsFromMisses: 0, firstBounceBypassFlags: 0, firstBounceBypassFlagsOff: 0 };
    // The first bounce's bypass flag by visibility index must match its resolve entry's.
    const resolveFlag = new Map(Array.from(state.resolve, (entry) => [entry & ~BYPASS_CACHE_BIT, (entry & BYPASS_CACHE_BIT) !== 0]));
    state.firstBounceBypass.forEach((flag, visibility) => {
        if (flag !== 0) tally.firstBounceBypassFlags++;
        if ((resolveFlag.get(visibility) ?? false) !== (flag !== 0)) tally.firstBounceBypassFlagsOff++;
    });
    const additions = new Map<number, number[]>();
    for (const entry of additionEntries) {
        const cell = state.multibounceCell[entry];
        if (cell === MULTIBOUNCE_MISS_CELL) {
            tally.additionsFromMisses++;
            continue;
        }
        let second: Vec3;
        if ((state.multibounceRays[entry * MULTIBOUNCE_RAY_WORDS + 3] & WORLD_CACHE_PROBE_RAY_BYPASS) !== 0) {
            second = [state.multibounceDirect[entry * 4], state.multibounceDirect[entry * 4 + 1], state.multibounceDirect[entry * 4 + 2]];
            tally.additionsFromOwnLight++;
        } else {
            const value = levelsOf(cell, direct);
            second = [value[0] / Math.max(value[3], 1), value[1] / Math.max(value[3], 1), value[2] / Math.max(value[3], 1)];
        }
        const info = state.multibounceInfo.slice(entry * 4, entry * 4 + 4);
        const sample = multibounceSample(second, [info[0], info[1], info[2]], info[3]);
        const added = multibounceBypassAddition(sample, DISCARD_PROBABILITY).map(removeNaNs);
        if (Math.max(...added) <= 0) continue;
        additions.set(state.visibilityQuery[state.multibounceQuery[entry]], added);
        tally.additions++;
    }

    const counts = new Map<number, [number, number]>();
    const directMeans = new Map<number, number>();
    const expected = state.rays.map((_, index) => {
        const extra = additions.get(index) ?? [0, 0, 0];
        if (selfBypassed.has(index)) {
            const own = ownDirect(index);
            return own === null ? null : extra.map((value) => value + own);
        }
        const cell = resolvedCellOf.get(index);
        if (cell === undefined) return extra;
        const d = levelsOf(cell, direct);
        const i = levelsOf(cell, indirect);
        counts.set(index, [d[3], i[3]]);
        directMeans.set(index, Math.max(d[0], d[1], d[2]) / Math.max(d[3], 1));
        return [0, 1, 2].map((channel) => d[channel] / Math.max(d[3], 1) + i[channel] / Math.max(i[3], 1) + extra[channel]);
    });
    return { expected, resolvedCellOf, counts, directMeans, additions, tally };
}

/**
 * Every second-bounce entry's sample as `UpdateMultibounceCells` makes it this frame: the light
 * at the second hit (its cell's filtered direct mean as read back, or its own light where its
 * segment bypasses) × BRDF / pdf; a miss 0. Brightest channel (the furnace is grey).
 *
 * Why the page needs it: a hash-grid cell is a column through its tile (NOTES), so a far floor
 * cell can hold ceiling hits too, which have no light. Since register D50 those count as
 * samples of no light, so such a cell reads the floor's light times its floor share, and a
 * second bounce landing there brings that, not L_f: the second bounce the cache can give is
 * this mean, a little under ρ L_f (Capsaicin's rule, which counted only lit hits, hid it).
 */
async function secondBounceSamples(device: GPUDevice, scenario: Scenario, state: FrameState): Promise<Float64Array> {
    const cache = scenario.cache;
    const capacity = cache.capacity;
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);
    const landingTiles = [...new Set(Array.from(state.multibounceCell).filter((cell) => cell !== MULTIBOUNCE_MISS_CELL).map(tileOf))];
    const slotOfTile = new Map(landingTiles.map((tile, slot) => [tile, slot]));
    const direct = await readTiles(device, cache, landingTiles, false);
    const samples = new Float64Array(state.multibounceCell.length);
    state.multibounceCell.forEach((cell, entry) => {
        if (cell === MULTIBOUNCE_MISS_CELL) return;
        let second: Vec3;
        if ((state.multibounceRays[entry * MULTIBOUNCE_RAY_WORDS + 3] & WORLD_CACHE_PROBE_RAY_BYPASS) !== 0) {
            second = [state.multibounceDirect[entry * 4], state.multibounceDirect[entry * 4 + 1], state.multibounceDirect[entry * 4 + 2]];
        } else {
            const value = pickLevel(direct, cache, slotOfTile.get(tileOf(cell)) ?? 0, cell, MAX_SAMPLE_COUNT).value;
            second = [value[0] / Math.max(value[3], 1), value[1] / Math.max(value[3], 1), value[2] / Math.max(value[3], 1)];
        }
        const info = state.multibounceInfo.slice(entry * 4, entry * 4 + 4);
        samples[entry] = Math.max(...multibounceSample(second, [info[0], info[1], info[2]], info[3]));
    });
    return samples;
}

/** The mean of `secondBounceSamples` over the entries whose first hit is in one of `cells`. */
function meanSampleOver(state: FrameState, samples: Float64Array, cells: Set<number>): { mean: number; entries: number } {
    let sum = 0;
    let entries = 0;
    state.multibounceQuery.forEach((query, entry) => {
        if (!cells.has(state.visibilityCell[query])) return;
        sum += samples[entry];
        entries++;
    });
    return { mean: sum / Math.max(entries, 1), entries };
}

/** The furnace's first-bounce cells on the ceiling in tiles under NEAR_TILE_SIZE (they can't reach the floor). */
function nearCeilingCells(state: FrameState): Set<number> {
    const constants = cellSizeConstants();
    const cells = new Set<number>();
    state.visibilityCell.forEach((cell, visibility) => {
        const ray = state.rays[state.visibilityQuery[visibility]];
        if (ray.surface === 'ceiling' && hashGridCacheGetCellSize(ray.hit, FURNACE.camera.position, constants) * LEVEL_SIDES[0] < NEAR_TILE_SIZE) cells.add(cell);
    });
    return cells;
}

/**
 * Every probe ray against its prediction (`predictProbeRays`): within four half steps, or 1%
 * where a bypass addition (known to about 0.3%) is in it.
 */
function judgePredictedRays(state: FrameState, prediction: ProbeRayPrediction, fail: (message: string) => void): { judged: number; skipped: number; off: number; worst: number } {
    const result = { judged: 0, skipped: 0, off: 0, worst: 0 };
    state.rays.forEach((_, index) => {
        const expected = prediction.expected[index];
        if (!expected) {
            result.skipped++;
            return;
        }
        const got = [0, 1, 2].map((channel) => state.radiance[index * 4 + channel]);
        const error = Math.max(...[0, 1, 2].map((channel) => Math.abs(got[channel] - expected[channel]) / Math.max(expected[channel], 1e-2)));
        result.judged++;
        result.worst = Math.max(result.worst, error);
        if (error > (prediction.additions.has(index) ? 0.01 : 4 * HALF_TOLERANCE)) {
            result.off++;
            if (result.off <= 3) fail(`ray ${index} reads ${got.map((value) => value.toFixed(4)).join(', ')}, predicted ${expected.map((value) => value.toFixed(4)).join(', ')} (bypass addition ${(prediction.additions.get(index) ?? [0])[0].toFixed(4)})`);
        }
    });
    if (result.off > 0) fail(`${result.off} of ${result.judged} probe rays don't read their prediction (levels, own sample, bypass addition)`);
    return result;
}

/**
 * The furnace's probe rays: each reads its prediction, and where its tile is under 2 m its cell
 * holds ceiling hits only, so its direct level reads no light (since register D50 with samples
 * counted: the ceiling's hits have no light to draw), and the near rays reading an indirect
 * level average `expectedIndirect` within `meanTolerance`. One ray's indirect value is its
 * cell's mean of second bounces that land in floor cells of different floor shares (see
 * `secondBounceSamples`), so single rays are judged only against their prediction; how many
 * are more than 2% off `expectedIndirect` is recorded.
 */
async function judgeCeilingRays(device: GPUDevice, scenario: Scenario, state: FrameState, fail: (message: string) => void, expectedIndirect = CEILING_INDIRECT, meanTolerance = 0.01) {
    const constants = cellSizeConstants();
    // The ceiling has no light: a self-bypassed ceiling ray's own sample is 0.
    const prediction = await predictProbeRays(device, scenario, state, () => 0);
    const rays = judgePredictedRays(state, prediction, fail);
    const tally = { near: 0, nearWithIndirect: 0, nearWithoutIndirect: 0, nearWithDirect: 0, nearWithDirectLight: 0, nearOff: 0, worstNear: 0, nearIndirectMean: 0, nearMeanWithZeros: 0, farWithDirect: 0 };
    let nearIndirectSum = 0;
    let nearSum = 0;
    for (const index of prediction.resolvedCellOf.keys()) {
        if (state.rays[index].surface !== 'ceiling') continue;
        const [directCount, indirectCount] = prediction.counts.get(index) ?? [0, 0];
        const tileSize = hashGridCacheGetCellSize(state.rays[index].hit, FURNACE.camera.position, constants) * LEVEL_SIDES[0];
        if (tileSize >= NEAR_TILE_SIZE) {
            if (directCount > 0) tally.farWithDirect++;
            continue;
        }
        tally.near++;
        if (directCount > 0) tally.nearWithDirect++;
        if ((prediction.directMeans.get(index) ?? 0) > 0) tally.nearWithDirectLight++;
        const extra = prediction.additions.get(index) ?? [0, 0, 0];
        const own = [0, 1, 2].map((channel) => state.radiance[index * 4 + channel] - extra[channel]);
        nearSum += own[0];
        if (indirectCount > 0) {
            tally.nearWithIndirect++;
            nearIndirectSum += own[0];
        } else tally.nearWithoutIndirect++;
        const error = indirectCount > 0 ? Math.max(...own.map((value) => relativeError(value, expectedIndirect))) : Math.max(...own.map(Math.abs)) / expectedIndirect;
        tally.worstNear = Math.max(tally.worstNear, error);
        if (error > (indirectCount > 0 ? TOLERANCE : 1e-3)) tally.nearOff++;
    }
    tally.nearIndirectMean = nearIndirectSum / Math.max(tally.nearWithIndirect, 1);
    tally.nearMeanWithZeros = nearSum / Math.max(tally.near, 1);
    if (tally.near === 0) fail(`no ceiling ray in a tile under ${NEAR_TILE_SIZE} m: move the camera`);
    if (tally.nearWithDirectLight > 0) fail(`${tally.nearWithDirectLight} ceiling rays in tiles under ${NEAR_TILE_SIZE} m read direct light: the ceiling has none`);
    if (tally.nearWithIndirect > 0 && relativeError(tally.nearIndirectMean, expectedIndirect) > meanTolerance) fail(`the near ceiling rays read ${tally.nearIndirectMean.toFixed(4)} on average, expected ${expectedIndirect.toFixed(4)}`);
    return { ...rays, ...prediction.tally, ...tally };
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
    const secondHits = Array.from(state.multibounceCell).filter((cell) => cell !== MULTIBOUNCE_MISS_CELL);
    if (state.visibilityRay.length !== secondHits.length) fail(`${state.visibilityRay.length} shadow rays for ${secondHits.length} second-bounce hits on the sunlit floor`);
    const listed = new Set(state.updateTiles);
    if (listed.size !== state.updateTiles.length) fail(`the update list holds ${state.updateTiles.length} entries for ${listed.size} tiles (register D32)`);

    // The second bounce the cache gives the near ceiling this frame (`secondBounceSamples`): within
    // 3% of the true second bounce (far floor cells that share a column with the unlit ceiling
    // read less than L_f since register D50; measured about 1%).
    const samples = await secondBounceSamples(device, scenario, state);
    const throughCache = meanSampleOver(state, samples, nearCeilingCells(state));
    if (relativeError(throughCache.mean, CEILING_INDIRECT) > 0.03) fail(`the near ceiling's second bounce through the cache is ${throughCache.mean.toFixed(4)}, the true one ${CEILING_INDIRECT.toFixed(4)}`);

    // The probe rays.
    const rays = await judgeCeilingRays(device, scenario, state, fail, throughCache.mean);

    // The cells of every tile the frame touched (the ceiling's, the floor's and the far tiles
    // that hold both): indirect values (only the ceiling's first-bounce hits get them) and
    // direct values (the floor's second-bounce hits, lit, and since register D50 the ceiling's
    // first-bounce hits as samples of no light).
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);
    const touchedTiles = [...new Set([...Array.from(state.visibilityCell, tileOf), ...secondHits.map(tileOf)])];
    const directCells = await finestValues(device, cache, touchedTiles, false);
    const indirectCells = await finestValues(device, cache, touchedTiles, true);
    // The cells' indirect values, on average: this frame's second bounce through the cache over
    // every entry within 1%, and the true second bounce within 3% (1 since RC3.5 but for the
    // columns above; Capsaicin's was 1 / (1 - p)). One cell's value is its landings' mean, so
    // cells are counted against CEILING_INDIRECT, not judged one by one.
    const indirectTally = { cells: 0, off: 0, worst: 0 };
    let indirectMeanSum = 0;
    for (const value of indirectCells.values()) {
        if (!(value[3] > 0)) continue;
        indirectTally.cells++;
        indirectMeanSum += value[0] / value[3];
        const error = relativeError(value[0] / value[3], CEILING_INDIRECT);
        indirectTally.worst = Math.max(indirectTally.worst, error);
        if (error > TOLERANCE) indirectTally.off++;
    }
    if (indirectTally.cells === 0) fail('no cell has indirect samples');
    const indirectCellMean = indirectMeanSum / Math.max(indirectTally.cells, 1);
    const allEntries = meanSampleOver(state, samples, new Set(state.visibilityCell));
    const measuredOverTrueSecondBounce = indirectCellMean / TRUE_SECOND_BOUNCE;
    if (relativeError(indirectCellMean, allEntries.mean) > 0.01) fail(`the cells' indirect means average ${indirectCellMean.toFixed(4)}, this frame's second bounce through the cache ${allEntries.mean.toFixed(4)}`);
    if (Math.abs(measuredOverTrueSecondBounce - 1) > 0.03) fail(`the cells' indirect means average ${measuredOverTrueSecondBounce.toFixed(4)} x the true second bounce`);

    // Register D50, the direct values, in tiles under 2 m (which can't hold both planes): the
    // ceiling's cells hold its first-bounce hits, which have no light to draw, as samples of no
    // light: exactly 0, with samples counted (Capsaicin counted none); the floor's hold its lit
    // second-bounce hits: L_f within 2%. Cells of far tiles holding both are recorded; section 19
    // judges a known mixture.
    const directTally = { pureFloor: 0, pureFloorOff: 0, pureCeiling: 0, pureCeilingSampled: 0, pureCeilingLit: 0, mixed: 0, mixedMeasured: 0, mixedExpected: 0, mixedLitOnly: 0 };
    const nearConstants = cellSizeConstants();
    for (const [cell, { points }] of cellHits(state, true)) {
        const surfaces = points.map((point) => classify(point, true));
        if (surfaces.some((surface) => surface !== 'floor' && surface !== 'ceiling')) continue;
        const floorShare = surfaces.filter((surface) => surface === 'floor').length / surfaces.length;
        const near = hashGridCacheGetCellSize(points[0], FURNACE.camera.position, nearConstants) * LEVEL_SIDES[0] < NEAR_TILE_SIZE;
        if (!near && (floorShare === 0 || floorShare === 1)) continue;
        const value = directCells.get(cell);
        const count = value?.[3] ?? 0;
        const mean = value && count > 0 ? Math.max(value[0], value[1], value[2]) / count : 0;
        if (floorShare === 0) {
            directTally.pureCeiling++;
            if (count > 0) directTally.pureCeilingSampled++;
            if (mean > 0) directTally.pureCeilingLit++;
        } else if (count > 0 && floorShare === 1) {
            directTally.pureFloor++;
            if (relativeError(mean, FLOOR_RADIANCE) > TOLERANCE) directTally.pureFloorOff++;
        } else if (count > 0) {
            directTally.mixed++;
            directTally.mixedMeasured += mean;
            directTally.mixedExpected += FLOOR_RADIANCE * floorShare;
            directTally.mixedLitOnly += FLOOR_RADIANCE;
        }
    }
    const mixedRatio = directTally.mixedMeasured / Math.max(directTally.mixedExpected, 1e-30);
    if (directTally.pureCeilingLit > 0) fail(`${directTally.pureCeilingLit} ceiling-only cells hold direct light: the ceiling has none`);
    if (directTally.pureCeilingSampled < 0.99 * directTally.pureCeiling) fail(`only ${directTally.pureCeilingSampled} of ${directTally.pureCeiling} ceiling-only cells counted their hits as samples of no light (register D50)`);
    if (directTally.pureFloor === 0 || directTally.pureFloorOff > Math.ceil(0.005 * directTally.pureFloor)) fail(`${directTally.pureFloorOff} of ${directTally.pureFloor} floor-only cells are more than 2% off L_f`);
    if (directTally.pureCeiling < 1000) fail(`only ${directTally.pureCeiling} near ceiling cells`);

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
    let keptMissesWithCell = 0;
    let keptHitsWithoutCell = 0;
    let keptWithoutEntry = 0;
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
        // Every kept ray has an entry (register D37): with a cell if it hits, with none if it misses.
        if (entry === undefined) {
            keptWithoutEntry++;
            continue;
        }
        const isMiss = state.multibounceCell[entry] === MULTIBOUNCE_MISS_CELL;
        if (margin < 0) keptMisses++;
        if (Math.abs(margin) < 0.5) undecided++;
        else if (margin < 0 && !isMiss) keptMissesWithCell++;
        else if (margin >= 0 && isMiss) keptHitsWithoutCell++;
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
    if (keptWithoutEntry > 0) fail(`${keptWithoutEntry} kept first-bounce entries have no second-bounce entry`);
    if (keptMissesWithCell > 0) fail(`${keptMissesWithCell} second rays the CPU sees leave the planes have a cell`);
    if (keptHitsWithoutCell > 0) fail(`${keptHitsWithoutCell} second rays the CPU sees land on the floor are entered as misses`);
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
            nearSecondBounceThroughCache: throughCache,
            allSecondBounceThroughCache: allEntries,
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
            directValues: { ...directTally, mixedRatio },
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
    // entry whose first bounce is in the cell, and their quantised sum (each sample as
    // `secondBounceSamples` rebuilds it from the landing cells as read back; a miss 0).
    const firstSamples = await secondBounceSamples(device, scenario, first);
    const expectedCounts = new Map<number, number>();
    const expectedSums = new Map<number, number>();
    first.multibounceQuery.forEach((query, entry) => {
        const cell = first.visibilityCell[query];
        expectedCounts.set(cell, (expectedCounts.get(cell) ?? 0) + 1);
        expectedSums.set(cell, (expectedSums.get(cell) ?? 0) + firstSamples[entry]);
    });
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
        // Quantised at 1/1000 (kHashGridCache_FloatQuantize); a miss is a sample of no light.
        const mean = scratch[index * 4] / 1000 / Math.max(count, 1);
        const expectedMean = (expectedSums.get(cell) ?? 0) / Math.max(expected, 1);
        // Each sample is rounded to 1/1000 before it is summed, so a cell of small samples (its
        // landings in floor cells that share a column with the unlit ceiling) may be off by that.
        const error = Math.abs(mean - expectedMean) / Math.max(expectedMean, QUANTISATION_STEP / TOLERANCE);
        worstMean = Math.max(worstMean, error);
        if (error > TOLERANCE) {
            meansOff++;
            if (meansOff <= 3) fail(`cell ${cell}'s indirect scratch averages ${mean.toFixed(4)}, expected ${expectedMean.toFixed(4)}`);
        }
    });
    if (countsOff > 0) fail(`${countsOff} of ${cells.length} cells' scratch counts differ from their second-bounce entries`);
    if (meansOff > 0) fail(`${meansOff} of ${cells.length} cells' scratch averages are more than 2% off`);

    // The second frame: UpdateTiles blends the first frame's samples in, and the rays read them:
    // the near ones average the first frame's second bounce over the near cells (2%: a ray reads
    // its cell's few samples, or a coarser level's).
    const firstThroughCache = meanSampleOver(first, firstSamples, nearCeilingCells(first));
    await runFrames(device, scenario, 1);
    const second = await readFrameState(device, scenario);
    const rays = await judgeCeilingRays(device, scenario, second, fail, firstThroughCache.mean, TOLERANCE);
    if (rays.nearWithIndirect === 0) fail('no ray reads an indirect sample after the second frame');

    return {
        failures,
        data: {
            firstFrame: { secondBounceEntries: first.multibounceCell.length, ceilingCellsWithIndirect: indirectSamples, rays: firstRays, scratchCells: cells.length, worstScratchMeanError: worstMean },
            secondFrame: { ...rays, expectedFromFirstFrame: firstThroughCache },
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

// ---- Section 11: the bypass additions, predicted where they fire -----------------------------------

async function checkCornerBypass(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`corner bypass: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const firstCount = state.visibilityCell.length;
    // Section 6's slab shades part of the corner, so a cell near the shadow's edge holds lit and
    // shadowed samples and a second hit's own light can differ from its cell's. A self-bypassed
    // ray's own sample is albedo / pi x E where the sun reaches its hit and 0 in the slab's
    // shadow (1 cm from the edge either way; the slab's own hits and the edge are left out).
    const prediction = await predictProbeRays(device, scenario, state, (index) => {
        const ray = state.rays[index];
        if (ray.surface !== 'floor' && ray.surface !== 'wall') return null;
        const margin = Math.max(slabShadowMargin(ray.hit, SLAB), slabShadowMargin(ray.hit, LOW_SLAB));
        if (Math.abs(margin) < 0.01) return null;
        return margin > 0 ? 0 : LIT_RADIANCE;
    });
    const rays = judgePredictedRays(state, prediction, fail);
    const tally = prediction.tally;
    if (tally.firstBounceBypassFlagsOff > 0) fail(`${tally.firstBounceBypassFlagsOff} first-bounce bypass flags differ from their resolve entries`);
    if (tally.firstBounceBypassFlags === 0) fail('no first bounce bypasses the cache: the addition is unchecked');
    if (tally.additions === 0) fail("UpdateMultibounceCells' bypass addition never fired: the check can't see it");
    if (tally.additionsFromOwnLight === 0) fail("no addition used a second hit's own light: that path is unchecked");
    // A segment's own light is one sun sample: albedo / pi x E, or 0 where the wall shades the hit.
    let ownLights = 0;
    let ownLightsOff = 0;
    for (let entry = 0; entry < state.multibounceCell.length; entry++) {
        if ((state.multibounceRays[entry * MULTIBOUNCE_RAY_WORDS + 3] & WORLD_CACHE_PROBE_RAY_BYPASS) === 0) continue;
        ownLights++;
        const value = state.multibounceDirect[entry * 4];
        if (Math.abs(value) > 1e-3 && relativeError(value, LIT_RADIANCE) > 2 * HALF_TOLERANCE) ownLightsOff++;
    }
    if (ownLightsOff > 0) fail(`${ownLightsOff} of ${ownLights} second hits' own light is neither 0 nor albedo / pi x E`);
    // A bypassing second hit on a back face (the wall's or the slab's underside, which the sun
    // never reaches) has no light drawn and no shadow ray: its own light must be the 0 that
    // GenerateMultibounceReservoirs wrote, not an older frame's word.
    let unlitOwnLights = 0;
    let unlitOwnLightsOff = 0;
    for (let entry = 0; entry < state.multibounceCell.length; entry++) {
        const base = entry * MULTIBOUNCE_RAY_WORDS;
        if ((state.multibounceRays[base + 3] & WORLD_CACHE_PROBE_RAY_BYPASS) === 0 || state.multibounceRays[base + 9] !== 0) continue;
        unlitOwnLights++;
        if (Math.max(...state.multibounceDirect.slice(entry * 4, entry * 4 + 3)) !== 0) unlitOwnLightsOff++;
    }
    if (unlitOwnLights === 0) fail('no bypassing second hit on a back face: the unlit own light is unchecked');
    if (unlitOwnLightsOff > 0) fail(`${unlitOwnLightsOff} of ${unlitOwnLights} bypassing second hits on a back face hold an own light`);
    // Where Capsaicin's two misdirected additions would have landed this frame: those rays now read their prediction.
    const capsaicinTargets = new Set<number>();
    for (const entry of state.visibilityRay) {
        const visibility = entry & ~BYPASS_CACHE_BIT;
        if ((entry & BYPASS_CACHE_BIT) !== 0 && visibility >= firstCount) capsaicinTargets.add(state.multibounceQuery[visibility - firstCount]);
    }
    if (rays.skipped > tally.selfBypassedRays) fail(`${rays.skipped} rays had no prediction`);
    if (rays.skipped > 0.5 * tally.selfBypassedRays) fail(`${rays.skipped} of ${tally.selfBypassedRays} self-bypassed rays had no prediction: move the slab`);
    return { failures, data: { firstBounceEntries: firstCount, secondBounceEntries: state.multibounceCell.length, ...tally, segmentsWithOwnLight: ownLights, unlitSegmentsWithOwnLight: unlitOwnLights, capsaicinMisdirectedTargets: capsaicinTargets.size, rays } };
}

// ---- Section 13: an open scene, half the second-bounce rays missing --------------------------------

async function checkStrips(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`strips: ${message}`);
    };
    const cache = scenario.cache;
    const capacity = cache.capacity;
    const constants = cellSizeConstants();
    const state = await readFrameState(device, scenario);
    const entries = state.multibounceCell.length;
    const hitEntries = Array.from(state.multibounceCell).filter((cell) => cell !== MULTIBOUNCE_MISS_CELL).length;
    const hitShare = hitEntries / Math.max(entries, 1);
    // The second bounce the cache gives the near ceiling this frame: the analytic STRIPS_INDIRECT
    // within 5% (far floor cells that share a column with the unlit ceiling read less than L_f
    // since register D50; a hit-only mean would read twice it), and the rays and cells against it.
    const samples = await secondBounceSamples(device, scenario, state);
    const throughCache = meanSampleOver(state, samples, nearCeilingCells(state));
    if (relativeError(throughCache.mean, STRIPS_INDIRECT) > 0.05) fail(`the near ceiling's second bounce through the cache is ${throughCache.mean.toFixed(4)}, the analytic ${STRIPS_INDIRECT.toFixed(4)}`);
    // The rays: each reads its prediction; the near ones average the second bounce (one ray is a
    // cell's noisy mean of hits and misses, so only the average is judged).
    const rays = await judgeCeilingRays(device, scenario, state, fail, throughCache.mean, TOLERANCE);
    // The near cells' indirect means, averaged.
    const nearCells = new Set<number>();
    state.visibilityCell.forEach((cell, visibility) => {
        const ray = state.rays[state.visibilityQuery[visibility]];
        if (ray.surface !== 'ceiling') return;
        const hit = ray.hit;
        if (hashGridCacheGetCellSize(hit, FURNACE.camera.position, constants) * LEVEL_SIDES[0] < NEAR_TILE_SIZE) nearCells.add(cell);
    });
    const tiles = [...new Set([...nearCells].map((cell) => Math.floor(cell / capacity.numCellsPerTile)))];
    const indirectCells = await finestValues(device, cache, tiles, true);
    let sum = 0;
    let cells = 0;
    // The cells with samples. Those without include the cells first reached this frame (the
    // one-frame lag), so they are counted, not averaged in; the rays' average above counts the
    // rays reading no indirect value as 0, which is what Capsaicin's step-down (register D37)
    // pulled down: about 2% of the near rays read 0 with it, 7 of 50,785 without.
    let cellsWithoutSamples = 0;
    for (const cell of nearCells) {
        const value = indirectCells.get(cell);
        if (!value || !(value[3] > 0)) {
            cellsWithoutSamples++;
            continue;
        }
        sum += value[0] / value[3];
        cells++;
    }
    const nearCellMean = sum / Math.max(cells, 1);
    const firstHitsOnTheFloor = state.rays.filter((ray) => ray.hitT >= 0 && ray.surface === 'floor').length;
    // By sample count (each bucket should average the same), and the near entries' hit share:
    // the strips cover half the floor, finely enough that a ray's landing point spreads over
    // many periods, so half the kept rays hit, and the CPU agrees which (it lands the recorded
    // ray on the strips' plane; rays longer than 5 m, whose half-precision direction blurs the
    // landing, and landings within 2 mm of an edge are left out).
    const byCount: Record<string, [number, number]> = {};
    for (const cell of nearCells) {
        const value = indirectCells.get(cell);
        if (!value || !(value[3] > 0)) continue;
        const bucket = value[3] >= 16 ? '16' : value[3] >= 8 ? '8-15' : value[3] >= 4 ? '4-7' : value[3] >= 2 ? '2-3' : value[3] >= 1 ? '1' : '<1';
        const entry = byCount[bucket] ?? [0, 0];
        entry[0] += value[0] / value[3];
        entry[1]++;
        byCount[bucket] = entry;
    }
    const byCountMean = Object.fromEntries(Object.entries(byCount).map(([key, [total, n]]) => [key, [total / n / STRIPS_INDIRECT, n]]));
    let nearEntries = 0;
    let nearHits = 0;
    let cpuHits = 0;
    let cpuJudged = 0;
    let disagreements = 0;
    state.multibounceQuery.forEach((query, entry) => {
        if (!nearCells.has(state.visibilityCell[query])) return;
        nearEntries++;
        const gpuHit = state.multibounceCell[entry] !== MULTIBOUNCE_MISS_CELL;
        if (gpuHit) nearHits++;
        const base = entry * MULTIBOUNCE_RAY_WORDS;
        const floats = new Float32Array(state.multibounceRays.buffer, state.multibounceRays.byteOffset + base * 4, 3);
        const direction = multibounceDirection(state, entry);
        if (direction[1] < 0) {
            const t = -floats[1] / direction[1];
            const x = floats[0] + t * direction[0];
            const z = floats[2] + t * direction[2];
            const phase = (((z + STRIPS.halfLength) % STRIPS.period) + STRIPS.period) % STRIPS.period;
            const inside = Math.abs(x) <= STRIPS.halfLength && Math.abs(z) <= STRIPS.halfLength && phase <= STRIPS.width;
            const margin = Math.min(Math.abs(phase), Math.abs(phase - STRIPS.width), Math.abs(STRIPS.period - phase));
            if (margin > 2e-3 && t < 5) {
                if (inside) cpuHits++;
                cpuJudged++;
                if (inside !== gpuHit) disagreements++;
            }
        }
    });
    if (cells === 0) fail('no near ceiling cell has indirect samples');
    const nearHitShare = nearHits / Math.max(nearEntries, 1);
    if (Math.abs(nearHitShare - STRIPS.fill) > 0.02) fail(`${(nearHitShare * 100).toFixed(2)}% of the near ceiling cells' kept second-bounce rays hit, expected ${STRIPS.fill * 100}%`);
    if (disagreements > Math.ceil(0.005 * cpuJudged)) fail(`the GPU and the CPU disagree on ${disagreements} of ${cpuJudged} landings`);
    // 2%: half the kept rays miss, so this frame's mean of about 20,000 samples is itself known to 0.7%.
    if (relativeError(rays.nearMeanWithZeros, throughCache.mean) > TOLERANCE) fail(`the near ceiling rays, those reading no indirect value as 0, average ${rays.nearMeanWithZeros.toFixed(4)}, expected ${throughCache.mean.toFixed(4)}`);
    if (relativeError(nearCellMean, throughCache.mean) > TOLERANCE) fail(`the near ceiling cells' indirect means average ${nearCellMean.toFixed(4)}, expected ${throughCache.mean.toFixed(4)} (a hit-only mean would read about ${(STRIPS_INDIRECT / STRIPS.fill).toFixed(4)})`);
    return { failures, data: { expectedIndirect: STRIPS_INDIRECT, nearSecondBounceThroughCache: throughCache, hitOnlyMeanWouldRead: STRIPS_INDIRECT / STRIPS.fill, entries, hitShare, nearCells: cells, nearCellsWithoutSamples: cellsWithoutSamples, nearCellMean, nearCellMeanOverExpected: nearCellMean / STRIPS_INDIRECT, byCountMean, nearEntries, nearHitShare, cpuHitShare: cpuHits / Math.max(cpuJudged, 1), cpuJudged, disagreements, firstHitsOnTheFloor, rays } };
}

// ---- Section 14: every first bounce bypasses; its probe rays average the second bounce -------------

/**
 * Cells of 900 pixels (Capsaicin's default is 32): tan(60° × 900 / 720) = 3.7 m a metre from the
 * camera, wider than the furnace's 3 m gap, so nearly every probe ray bypasses the cache.
 */
const BYPASS_ALL: Partial<RadianceCacheSettings> = { useMultibounce: true, hashGridCacheCellSize: 900 };

async function checkBypassAverage(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`bypass average: ${message}`);
    };
    // Over four frames: a self-bypassed ceiling ray reads its own sample (0, the ceiling has no
    // light) plus its one draw over 1 - p (register D38), so a single ray reads 0 or about
    // rho L_f / 0.3, and their average is the true second bounce. Independent of the
    // per-ray prediction, which uses the same formula as the kernel.
    let sum = 0;
    let rays = 0;
    let predictedOff = 0;
    for (let frame = 0; frame < 4; frame++) {
        await runFrames(device, scenario, 1);
        const state = await readFrameState(device, scenario);
        const prediction = await predictProbeRays(device, scenario, state, () => 0);
        predictedOff += judgePredictedRays(state, prediction, fail).off;
        for (const entry of state.resolve) {
            if ((entry & BYPASS_CACHE_BIT) === 0) continue;
            const index = state.visibilityQuery[entry & ~BYPASS_CACHE_BIT];
            if (state.rays[index].surface !== 'ceiling') continue;
            sum += state.radiance[index * 4];
            rays++;
        }
    }
    const mean = sum / Math.max(rays, 1);
    if (rays === 0) fail('no ceiling ray bypassed the cache');
    if (relativeError(mean, TRUE_SECOND_BOUNCE) > TOLERANCE) fail(`the self-bypassed ceiling rays average ${mean.toFixed(4)}, the second bounce is ${TRUE_SECOND_BOUNCE.toFixed(4)}`);
    return { failures, data: { selfBypassedCeilingRays: rays, mean, meanOverTrueSecondBounce: mean / TRUE_SECOND_BOUNCE, predictedOff } };
}

// ---- Sections 15-18: the sky as a light (RC3.4) ---------------------------------------------------

/** The constant sky's radiance in sections 15, 17 and 18. */
const SKY_RADIANCE = 1.5;
/** Section 16's coded sky: per channel (1 + d) / 2 times this, d the direction's x, y or z (`codedSkyTexture`). */
const CODED_SKY_SCALE = 2;
const SKY_ALBEDO = 0.5;
/** ρ L: what a surface open to the constant sky reflects, the done-when's "cell radiance = ρL". */
const OPEN_SKY_RADIANCE = SKY_ALBEDO * SKY_RADIANCE;
const UP: Vec3 = [0, 1, 0];
const DOWN: Vec3 = [0, -1, 0];

/**
 * A scene under a sky (sections 15-18):
 * - `openFloor`: the furnace's 400 m floor (y = 0, facing up) and nothing over it but a 4 × 4 m
 *   patch 50 m up, facing down, for the probes; the camera 1 m under the patch looks straight up
 *   at it, so every probe is on the patch and every probe ray that hits, hits the floor. The patch
 *   hides at most 16 / (π 50²) = 0.2% of the floor's sky (counted, `viewFactor`).
 * - `ceilingOnly`: the furnace's ceiling (y = 3, facing down) and a black 1 × 1 m patch on the
 *   ground (facing up, albedo 0) for the probes; the camera 0.3 m over the patch looks straight
 *   down at it. The ceiling sees the sky below it but for the patch (at most 3.5%, counted), and
 *   its second-bounce rays go down: nearly all miss, the rest hit the black patch.
 * - `corner`: the corner of sections 1-7 (floor and 40 × 10 m wall), no slab.
 */
interface SkyScene {
    readonly layout: 'openFloor' | 'ceilingOnly' | 'corner';
    /** `constant`: SKY_RADIANCE in every direction. `coded`: (1 + d) / 2 per channel × CODED_SKY_SCALE. */
    readonly environment: 'constant' | 'coded';
}

const OPEN_FLOOR = { patchY: 50, patchHalf: 2 } as const;
const CEILING_ONLY = { patchHalf: 0.5 } as const;
const SKY_LAYOUT_CAMERA: Record<SkyScene['layout'], PageCamera> = {
    openFloor: { position: [0, OPEN_FLOOR.patchY - 1, 0], target: [0, OPEN_FLOOR.patchY, 0], fovYDeg: 60, near: 0.05, far: 1000 },
    ceilingOnly: { position: [0, 0.3, 0], target: [0, 0, 0], fovYDeg: 60, near: 0.05, far: 1000 },
    corner: CAMERA,
};

/** A horizontal square, half-width `half` at height `y`, as `quad` takes it: counter-clockwise seen from `normal`'s side. */
function horizontalSquare(half: number, y: number, normal: Vec3): Vec3[] {
    return normal[1] > 0
        ? [[-half, y, half], [half, y, half], [half, y, -half], [-half, y, -half]]
        : [[-half, y, -half], [half, y, -half], [half, y, half], [-half, y, half]];
}

/** The corner's wall (sections 1-7), and the part of its floor in front of it (z ≥ 0): the occluders of sections 18's floor and wall. */
const CORNER_WALL: readonly Vec3[] = [[-20, 0, 0], [20, 0, 0], [20, 10, 0], [-20, 10, 0]];
const CORNER_FLOOR_IN_FRONT: readonly Vec3[] = [[-200, 0, 0], [200, 0, 0], [200, 0, 200], [-200, 0, 200]];

function describeSkyScene(platform: GIPlatformImpl, lights: readonly LightDescriptor[], sky: SkyScene): LightHandle[] {
    const half = FURNACE.halfWidth;
    if (sky.layout === 'openFloor') {
        quad(platform, 'floor', horizontalSquare(half, 0, UP), UP, SKY_ALBEDO);
        quad(platform, 'probe patch', horizontalSquare(OPEN_FLOOR.patchHalf, OPEN_FLOOR.patchY, DOWN), DOWN, SKY_ALBEDO);
    } else if (sky.layout === 'ceilingOnly') {
        quad(platform, 'ceiling', horizontalSquare(half, FURNACE.ceilingY, DOWN), DOWN, SKY_ALBEDO);
        quad(platform, 'black probe patch', horizontalSquare(CEILING_ONLY.patchHalf, 0, UP), UP, 0);
    } else {
        quad(platform, 'floor', horizontalSquare(half, 0, UP), UP, SKY_ALBEDO);
        quad(platform, 'wall', [...CORNER_WALL], [0, 0, 1], SKY_ALBEDO);
    }
    return lights.map((light) => platform.scene.createLight(light));
}

/** three.js's equirect direction for texture coordinates (v = 0 at the first row in memory, the nadir), as hit-shading-sky writes it. */
function equirectTexelDirection(u: number, v: number): Vec3 {
    const azimuth = (u - 0.5) * 2 * Math.PI;
    const latitude = (v - 0.5) * Math.PI;
    return [Math.cos(latitude) * Math.cos(azimuth), Math.sin(latitude), Math.cos(latitude) * Math.sin(azimuth)];
}

/**
 * Section 16's sky: each texel holds (1 + d) / 2 for its own direction d (hit-shading-sky's
 * coded texture), so red follows x, green y and blue z. Linear in d, so its irradiance on a
 * normal n is exact: π/2 + (π/3) n per channel (`codedSkyIrradiance`), and so are its SH.
 */
function codedSkyTexture(device: GPUDevice): GPUTexture {
    const width = 512;
    const height = 256;
    const texture = device.createTexture({ label: 'coded sky', size: { width, height }, format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const values = new Float32Array(width * height * 4);
    for (let row = 0; row < height; row++) {
        for (let column = 0; column < width; column++) {
            const d = equirectTexelDirection((column + 0.5) / width, (row + 0.5) / height);
            values.set([(1 + d[0]) / 2, (1 + d[1]) / 2, (1 + d[2]) / 2, 1], (row * width + column) * 4);
        }
    }
    device.queue.writeTexture({ texture }, values, { bytesPerRow: width * 16 }, { width, height });
    return texture;
}

function skyEnvironment(sky: SkyScene, texture: GPUTexture | null): EnvironmentDescriptor {
    if (sky.environment === 'coded' && texture) return { type: 'texture', texture, layout: 'equirect', intensity: CODED_SKY_SCALE };
    return { type: 'color', color: [1, 1, 1], intensity: SKY_RADIANCE };
}

/** The coded sky's radiance along d. */
function codedSkyRadiance(d: Vec3): Vec3 {
    return [0, 1, 2].map((channel) => (CODED_SKY_SCALE * (1 + d[channel])) / 2) as unknown as Vec3;
}

/** The coded sky's irradiance on a surface facing n, with nothing in the way: S (π/2 + (π/3) n) per channel. */
function codedSkyIrradiance(n: Vec3): Vec3 {
    return [0, 1, 2].map((channel) => CODED_SKY_SCALE * (Math.PI / 2 + (Math.PI / 3) * n[channel])) as unknown as Vec3;
}

function luminance(rgb: Vec3): number {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

// Small vector helpers for the view factors.
function vSub(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function vDot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function vCross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function vNormalize(a: Vec3): Vec3 {
    const length = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / length, a[1] / length, a[2] / length];
}

/**
 * The share of a surface point's cosine-weighted hemisphere (unit normal `normal`) that a
 * polygon covers, the polygon wholly on the normal's side (an edge in the point's plane is
 * fine): Lambert's contour integral, F = |Σ θ_i n · (v_i × v_{i+1}) / |v_i × v_{i+1}|| / 2π, v_i
 * the unit directions to the corners, θ_i the angle between neighbours. Exact; independent of
 * the GPU's estimate, which draws directions and traces them.
 */
function viewFactor(point: Vec3, normal: Vec3, polygon: readonly Vec3[]): number {
    let sum = 0;
    for (let index = 0; index < polygon.length; index++) {
        const a = vNormalize(vSub(polygon[index], point));
        const b = vNormalize(vSub(polygon[(index + 1) % polygon.length], point));
        const across = vCross(a, b);
        const length = Math.hypot(across[0], across[1], across[2]);
        if (length < 1e-12) continue;
        sum += Math.acos(Math.min(Math.max(vDot(a, b), -1), 1)) * (vDot(normal, across) / length);
    }
    return Math.abs(sum) / (2 * Math.PI);
}

/** A surface of a sky scene: its normal on the lit side, the polygon that hides part of its sky, and what a hit on it should reflect. */
interface SkySurface {
    readonly normal: Vec3;
    readonly occluder: readonly Vec3[];
    /**
     * A hit point put back on the surface's plane, and for the corner a hair (0.1 mm) off the
     * other surface's: a hit read back sits a rounding error off its plane, and a point a hair
     * behind the corner's other surface would see its occluder edge-on (Lambert's contour needs
     * the polygon on the normal's side). Both surfaces see half their sky hidden at the corner.
     */
    snap(point: Vec3): Vec3;
}

/** The surfaces whose cells sections 15, 17 and 18 judge, by the page's classification. */
function skySurface(layout: SkyScene['layout'], surface: Surface): SkySurface | null {
    if (layout === 'openFloor' && surface === 'floor') return { normal: UP, occluder: horizontalSquare(OPEN_FLOOR.patchHalf, OPEN_FLOOR.patchY, DOWN), snap: (p) => [p[0], 0, p[2]] };
    if (layout === 'ceilingOnly' && surface === 'ceiling') return { normal: DOWN, occluder: horizontalSquare(CEILING_ONLY.patchHalf, 0, UP), snap: (p) => [p[0], FURNACE.ceilingY, p[2]] };
    if (layout === 'corner' && surface === 'floor') return CORNER_FLOOR;
    if (layout === 'corner' && surface === 'wall') return CORNER_WALL_SURFACE;
    return null;
}

const CORNER_OFFSET = 1e-4;
const CORNER_FLOOR: SkySurface = { normal: UP, occluder: CORNER_WALL, snap: (p) => [p[0], 0, Math.max(p[2], CORNER_OFFSET)] };
const CORNER_WALL_SURFACE: SkySurface = { normal: [0, 0, 1], occluder: CORNER_FLOOR_IN_FRONT, snap: (p) => [p[0], Math.max(p[1], CORNER_OFFSET), 0] };

/** F, the share of a surface point's sky its occluder hides (sections 15, 17, 18). */
function hiddenSky(point: Vec3, surface: SkySurface): number {
    return viewFactor(surface.snap(point), surface.normal, surface.occluder);
}

/** ρ L (1 − F): a surface's direct value under the constant sky. */
function constantSkyDirect(point: Vec3, surface: SkySurface): number {
    return OPEN_SKY_RADIANCE * (1 - hiddenSky(point, surface));
}

/** A second-bounce entry's hit point, from its record (origin words 0-2, direction halves 4-5, hit distance 6). */
function multibounceHit(state: FrameState, entry: number): Vec3 {
    const base = entry * MULTIBOUNCE_RAY_WORDS;
    const floats = new Float32Array(state.multibounceRays.buffer, state.multibounceRays.byteOffset + base * 4, 7);
    const direction = multibounceDirection(state, entry);
    return [floats[0] + floats[6] * direction[0], floats[1] + floats[6] * direction[1], floats[2] + floats[6] * direction[2]];
}

/**
 * Per cell: the hit points of this frame's entries in it (first and second bounce), each a
 * sample its direct value took this frame, with their surfaces. A cell whose entries are on
 * a surface the section doesn't judge is left out.
 */
function cellHits(state: FrameState, furnace: boolean): Map<number, { points: Vec3[]; surfaces: Set<Surface> }> {
    const cells = new Map<number, { points: Vec3[]; surfaces: Set<Surface> }>();
    const add = (cell: number, point: Vec3, surface: Surface): void => {
        const entry = cells.get(cell) ?? { points: [], surfaces: new Set<Surface>() };
        entry.points.push(point);
        entry.surfaces.add(surface);
        cells.set(cell, entry);
    };
    state.visibilityCell.forEach((cell, visibility) => {
        const ray = state.rays[state.visibilityQuery[visibility]];
        add(cell, ray.hit, ray.surface);
    });
    state.multibounceCell.forEach((cell, entry) => {
        if (cell === MULTIBOUNCE_MISS_CELL) return;
        const hit = multibounceHit(state, entry);
        add(cell, hit, classify(hit, furnace));
    });
    return cells;
}

/** How a set of cells reads against what they should: per-cell ratios and their spread. */
interface CellJudgement {
    cells: number;
    meanRatio: number;
    worst: number;
    p99: number;
    /** Cells further from their expected value than their bound (2% plus five standard deviations of their samples' mean). */
    off: number;
    /** The worst |ratio − 1| over the cell's bound. */
    worstOverBound: number;
    /** The three cells furthest past their bound, for the record. */
    worstCells: Omit<CellRatio, 'bound'>[];
    /** By the expected value's share of the open sky, 0.1 wide: mean ratio and cells. */
    byOpenShare: Record<string, [number, number]>;
}

/** A judged cell: measured over expected, the expected value, and how far off (relative) its samples' spread allows. */
interface CellRatio {
    readonly ratio: number;
    readonly expected: number;
    readonly bound: number;
    /** What the record shows of the cell: its samples' count, this frame's entries and their mean point. */
    readonly count?: number;
    readonly entries?: number;
    readonly at?: Vec3;
}

function judgeCells(ratios: readonly CellRatio[]): CellJudgement {
    const errors = ratios.map(({ ratio }) => Math.abs(ratio - 1)).sort((a, b) => a - b);
    const byOpenShare: Record<string, [number, number]> = {};
    for (const { ratio, expected } of ratios) {
        const bucket = (Math.floor((expected / OPEN_SKY_RADIANCE) * 10) / 10).toFixed(1);
        const entry = byOpenShare[bucket] ?? [0, 0];
        entry[0] += ratio;
        entry[1]++;
        byOpenShare[bucket] = entry;
    }
    for (const entry of Object.values(byOpenShare)) entry[0] /= entry[1];
    return {
        cells: ratios.length,
        meanRatio: ratios.reduce((sum, { ratio }) => sum + ratio, 0) / Math.max(ratios.length, 1),
        worst: errors[errors.length - 1] ?? 0,
        p99: errors[Math.floor(0.99 * (errors.length - 1))] ?? 0,
        off: ratios.filter(({ ratio, bound }) => Math.abs(ratio - 1) > bound).length,
        worstOverBound: Math.max(0, ...ratios.map(({ ratio, bound }) => Math.abs(ratio - 1) / bound)),
        worstCells: [...ratios]
            .sort((a, b) => Math.abs(b.ratio - 1) / b.bound - Math.abs(a.ratio - 1) / a.bound)
            .slice(0, 3)
            .map(({ bound, ...rest }) => ({ ...rest, overBound: Math.abs(rest.ratio - 1) / bound })),
        byOpenShare,
    };
}

/**
 * The finest-level direct values of every cell this frame's entries reached on the judged
 * surfaces, against the mean of ρ L (1 − F) over the entries' hit points (the constant sky).
 *
 * Each cell's bound: 2%, plus five standard deviations of its mean. A sample is ρ L or 0 by
 * its shadow ray (a point whose sky is hidden by F: variance (ρ L)² F (1 − F)); `UpdateTiles`
 * averages a frame's n samples and, at the cap of 16, moves 1/16 of the way to that mean each
 * frame, an average over about 31 frames (2 / α − 1), so 31 n samples; below the cap, its
 * count of samples.
 */
async function judgeConstantSkyCells(device: GPUDevice, scenario: Scenario, state: FrameState, layout: SkyScene['layout']): Promise<{ judgement: CellJudgement; bySurface: Record<string, CellJudgement> }> {
    const capacity = scenario.cache.capacity;
    const hits = cellHits(state, scenario.furnace);
    const judged = [...hits].filter(([, { surfaces }]) => [...surfaces].every((surface) => skySurface(layout, surface) !== null));
    const tiles = [...new Set(judged.map(([cell]) => Math.floor(cell / capacity.numCellsPerTile)))];
    const values = await finestValues(device, scenario.cache, tiles, false);
    const all: CellRatio[] = [];
    const bySurface = new Map<string, CellRatio[]>();
    for (const [cell, { points, surfaces }] of judged) {
        const value = values.get(cell);
        if (!value || !(value[3] > 0)) continue;
        // The cell's entries' points, each on its own surface (a cell is a column through its tile).
        let expected = 0;
        let variance = 0;
        for (const point of points) {
            const surface = skySurface(layout, classify(point, scenario.furnace));
            if (!surface) continue;
            const hidden = hiddenSky(point, surface);
            expected += OPEN_SKY_RADIANCE * (1 - hidden);
            variance += OPEN_SKY_RADIANCE * OPEN_SKY_RADIANCE * hidden * (1 - hidden);
        }
        expected /= points.length;
        variance /= points.length;
        const samples = value[3] >= MAX_SAMPLE_COUNT ? 31 * points.length : value[3];
        const bound = TOLERANCE + (5 * Math.sqrt(variance / samples)) / expected;
        const measured = (value[0] + value[1] + value[2]) / (3 * value[3]);
        const at: Vec3 = [0, 1, 2].map((axis) => points.reduce((sum, point) => sum + point[axis], 0) / points.length) as unknown as Vec3;
        const sample = { ratio: measured / expected, expected, bound, count: value[3], entries: points.length, at };
        all.push(sample);
        const key = [...surfaces].sort().join('+');
        bySurface.set(key, [...(bySurface.get(key) ?? []), sample]);
    }
    return { judgement: judgeCells(all), bySurface: Object.fromEntries([...bySurface].map(([key, ratios]) => [key, judgeCells(ratios)])) };
}

/** The shadow-ray list's reservoirs: how many drew the sky, with what weights, and how many of them are second-bounce entries. */
function skyReservoirs(state: FrameState): { total: number; sky: number; secondBounce: number; secondBounceSky: number; weights: Map<number, number> } {
    const count0 = state.counters[HASH_GRID_COUNTER.visibilityCount0];
    const words = state.reservoirs;
    const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
    const result = { total: state.visibilityRay.length, sky: 0, secondBounce: 0, secondBounceSky: 0, weights: new Map<number, number>() };
    state.visibilityRay.forEach((entry, index) => {
        const isSky = words[RESERVOIR_WORDS * index] === DIRECT_SKY_LIGHT;
        const isSecond = (entry & ~BYPASS_CACHE_BIT) >= count0;
        if (isSky) result.sky++;
        if (isSecond) result.secondBounce++;
        if (isSky && isSecond) result.secondBounceSky++;
        const weight = floats[RESERVOIR_WORDS * index + 1];
        result.weights.set(weight, (result.weights.get(weight) ?? 0) + 1);
    });
    return result;
}

/** The weights in a sky-only scene: each drawn light is the sky, its weight 1 (the only candidate). */
function checkSkyOnlyReservoirs(state: FrameState, fail: (message: string) => void, secondBounceExpected: boolean): Record<string, number> {
    const reservoirs = skyReservoirs(state);
    if (reservoirs.total === 0) fail('no shadow rays: the sky was never drawn');
    if (reservoirs.sky !== reservoirs.total) fail(`${reservoirs.total - reservoirs.sky} of ${reservoirs.total} reservoirs drew something other than the sky, the only light`);
    // weightSum / skyWeight with the two equal: 1 within the GPU's division (WGSL allows 2.5 ulp; the RTX reads 1 - 2^-24).
    const otherWeights = [...reservoirs.weights].filter(([weight]) => Math.abs(weight - 1) > 1e-6).reduce((sum, [, count]) => sum + count, 0);
    if (otherWeights > 0) fail(`${otherWeights} reservoirs weigh the sky by other than 1, the only candidate`);
    if (secondBounceExpected && reservoirs.secondBounceSky === 0) fail('no second-bounce hit drew the sky (GenerateMultibounceReservoirs)');
    const weights = [...reservoirs.weights.keys()];
    return { shadowRays: reservoirs.total, skyDrawn: reservoirs.sky, secondBounceShadowRays: reservoirs.secondBounce, secondBounceSkyDrawn: reservoirs.secondBounceSky, smallestWeight: Math.min(...weights), largestWeight: Math.max(...weights) };
}

/** Probe rays that hit `surface`: their mean reading (rgb averaged), each the resolve of its cell or its own sample, plus the second bounce. */
function meanRayReading(state: FrameState, surface: Surface): { rays: number; mean: number } {
    let sum = 0;
    let rays = 0;
    state.rays.forEach((ray, index) => {
        if (ray.hitT < 0 || ray.surface !== surface) return;
        sum += (state.radiance[index * 4] + state.radiance[index * 4 + 1] + state.radiance[index * 4 + 2]) / 3;
        rays++;
    });
    return { rays, mean: sum / Math.max(rays, 1) };
}

/** The second-bounce entries: how many missed, and the indirect values of the cells the judged surface's first hits are in. */
async function indirectOfSurface(device: GPUDevice, scenario: Scenario, state: FrameState, surface: Surface): Promise<{ cells: Map<number, CellValue>; entries: number; misses: number }> {
    const capacity = scenario.cache.capacity;
    const surfaceCells = new Set<number>();
    state.visibilityCell.forEach((cell, visibility) => {
        if (state.rays[state.visibilityQuery[visibility]].surface === surface) surfaceCells.add(cell);
    });
    const tiles = [...new Set([...surfaceCells].map((cell) => Math.floor(cell / capacity.numCellsPerTile)))];
    const values = await finestValues(device, scenario.cache, tiles, true);
    const cells = new Map<number, CellValue>();
    for (const cell of surfaceCells) {
        const value = values.get(cell);
        if (value) cells.set(cell, value);
    }
    const misses = Array.from(state.multibounceCell).filter((cell) => cell === MULTIBOUNCE_MISS_CELL).length;
    return { cells, entries: state.multibounceCell.length, misses };
}

// ---- Section 15: the open floor under a constant sky ---------------------------------------------

async function checkSkyOpenFloor(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`sky, open floor: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const reservoirs = checkSkyOnlyReservoirs(state, fail, true);
    // Every floor cell reads ρ L (1 − F), F ≤ 0.2% from the patch, within 2% (one sample of the
    // sky is π L exactly wherever its ray escapes).
    const { judgement } = await judgeConstantSkyCells(device, scenario, state, 'openFloor');
    if (judgement.cells < 100) fail(`only ${judgement.cells} floor cells to judge`);
    if (judgement.off > Math.ceil(SKY_CELLS_ALLOWED_OFF * judgement.cells)) fail(`${judgement.off} of ${judgement.cells} floor cells are past their bound off rho L (1 - F) (worst ${(judgement.worst * 100).toFixed(2)}%)`);
    if (Math.abs(judgement.meanRatio - 1) > 0.005) fail(`the floor cells average ${judgement.meanRatio.toFixed(4)} x rho L (1 - F)`);
    // Second bounce: the floor's kept rays go up into the sky; a miss adds no light (register D37),
    // the sky having lit the floor through its own sample. Only the rare hit on the patch's
    // underside (which sees the floor, not the sky) adds a little.
    const indirect = await indirectOfSurface(device, scenario, state, 'floor');
    let indirectSum = 0;
    let indirectCells = 0;
    for (const value of indirect.cells.values()) {
        if (!(value[3] > 0)) continue;
        indirectSum += (value[0] + value[1] + value[2]) / (3 * value[3]);
        indirectCells++;
    }
    const indirectMean = indirectSum / Math.max(indirectCells, 1);
    if (indirectCells === 0) fail('no floor cell has indirect samples');
    if (indirect.misses === 0) fail('no second-bounce ray missed');
    if (indirectMean > 0.01 * OPEN_SKY_RADIANCE) fail(`the floor cells' indirect values average ${indirectMean.toFixed(5)}, expected about 0 (a miss would add rho L = ${OPEN_SKY_RADIANCE} if it read the sky)`);
    // The probe rays that hit the floor read ρ L (direct, plus the near-0 indirect).
    const rays = meanRayReading(state, 'floor');
    if (Math.abs(rays.mean / OPEN_SKY_RADIANCE - 1) > 0.01) fail(`the floor's probe rays read ${rays.mean.toFixed(4)} on average, expected rho L = ${OPEN_SKY_RADIANCE}`);
    return { failures, data: { expected: OPEN_SKY_RADIANCE, reservoirs, cells: judgement, indirect: { cells: indirectCells, mean: indirectMean, secondBounceEntries: indirect.entries, misses: indirect.misses }, rays } };
}

// ---- Section 16: the open floor under the coded sky and a sun --------------------------------------

/** Section 16's sun: along (0, -1, -1), casting no shadow (the patch would shade a spot of the floor). */
const SKY_SUN: LightDescriptor = { type: 'directional', direction: SUN_TRAVEL, color: [1, 1, 1], intensity: SUN_INTENSITY, castsShadows: false, label: 'sun' };
const SKY_SUN_IRRADIANCE = SUN_INTENSITY * Math.SQRT1_2;

async function checkSkyAndSun(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`coded sky and sun: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    // The light choice: on the floor the sky weighs the luminance of its irradiance (exact for
    // the coded sky's SH), the sun its E, so the sky is drawn with q = w_sky / (w_sky + w_sun)
    // and weighed 1 / q, the sun 1 / (1 - q).
    const skyIrradiance = codedSkyIrradiance(UP);
    const skyWeight = luminance(skyIrradiance);
    const q = skyWeight / (skyWeight + SKY_SUN_IRRADIANCE);
    const count0 = state.counters[HASH_GRID_COUNTER.visibilityCount0];
    const floats = new Float32Array(state.reservoirs.buffer, state.reservoirs.byteOffset, state.reservoirs.length);
    let firstBounce = 0;
    let skyDrawn = 0;
    let weightsOff = 0;
    state.visibilityRay.forEach((entry, index) => {
        const visibility = entry & ~BYPASS_CACHE_BIT;
        if (visibility >= count0 || state.rays[state.visibilityQuery[visibility]].surface !== 'floor') return;
        firstBounce++;
        const isSky = state.reservoirs[RESERVOIR_WORDS * index] === DIRECT_SKY_LIGHT;
        if (isSky) skyDrawn++;
        const expectedWeight = isSky ? 1 / q : 1 / (1 - q);
        if (relativeError(floats[RESERVOIR_WORDS * index + 1], expectedWeight) > 2e-3) weightsOff++;
    });
    const skyShare = skyDrawn / Math.max(firstBounce, 1);
    if (firstBounce < 1000) fail(`only ${firstBounce} floor reservoirs`);
    if (Math.abs(skyShare - q) > 0.01) fail(`the sky was drawn for ${(skyShare * 100).toFixed(2)}% of the floor's hits, expected ${(q * 100).toFixed(2)}%`);
    if (weightsOff > 0) fail(`${weightsOff} of ${firstBounce} reservoirs aren't weighed 1 / q (sky ${(1 / q).toFixed(4)}, sun ${(1 / (1 - q)).toFixed(4)})`);
    // The floor cells, per channel: ρ / π × (the sky's irradiance less the patch's share + the
    // sun's). Green carries the sky's direction (its irradiance on the floor is 5π/6 S, 2π/3 S
    // more than on a ceiling); red and blue π/2 S. A sample read along the normal instead of its
    // direction would put green's sky part at π S, 20% high. One cell mixes sun and sky draws, so
    // only the mean over cells is judged (within 1%).
    const capacity = scenario.cache.capacity;
    const hits = cellHits(state, scenario.furnace);
    const floorCells = [...hits].filter(([, { surfaces }]) => surfaces.size === 1 && surfaces.has('floor'));
    const tiles = [...new Set(floorCells.map(([cell]) => Math.floor(cell / capacity.numCellsPerTile)))];
    const values = await finestValues(device, scenario.cache, tiles, false);
    const patch = horizontalSquare(OPEN_FLOOR.patchHalf, OPEN_FLOOR.patchY, DOWN);
    const ratioSums = [0, 0, 0];
    let cells = 0;
    for (const [cell, { points }] of floorCells) {
        const value = values.get(cell);
        if (!value || !(value[3] > 0)) continue;
        const expected = [0, 0, 0];
        for (const point of points) {
            // The patch hides F of the floor's sky, about the sky's radiance toward its centre.
            const hidden = viewFactor(point, UP, patch);
            const towardPatch = codedSkyRadiance(vNormalize(vSub([0, OPEN_FLOOR.patchY, 0], point)));
            for (let channel = 0; channel < 3; channel++) expected[channel] += (SKY_ALBEDO / Math.PI) * (skyIrradiance[channel] - Math.PI * hidden * towardPatch[channel] + SKY_SUN_IRRADIANCE);
        }
        for (let channel = 0; channel < 3; channel++) ratioSums[channel] += value[channel] / value[3] / (expected[channel] / points.length);
        cells++;
    }
    const meanRatios = ratioSums.map((sum) => sum / Math.max(cells, 1));
    if (cells < 100) fail(`only ${cells} floor cells to judge`);
    meanRatios.forEach((ratio, channel) => {
        if (Math.abs(ratio - 1) > 0.01) fail(`the floor cells' ${['red', 'green', 'blue'][channel]} averages ${ratio.toFixed(4)} x its expected value`);
    });
    const expectedOpen = [0, 1, 2].map((channel) => (SKY_ALBEDO / Math.PI) * (skyIrradiance[channel] + SKY_SUN_IRRADIANCE));
    return { failures, data: { skyWeight, sunWeight: SKY_SUN_IRRADIANCE, expectedSkyShare: q, skyShare, firstBounceReservoirs: firstBounce, weightsOff, expectedOpenFloor: expectedOpen, cells, meanRatios } };
}

// ---- Section 17: the ceiling only, under a constant sky ---------------------------------------------

async function checkSkyCeilingOnly(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`sky, ceiling only: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const reservoirs = checkSkyOnlyReservoirs(state, fail, false);
    // The ceiling's cells: ρ L (1 − F), F up to 3.5% over the patch. A sample is 0 or ρ L by its
    // ray's visibility, so a cell over the patch is a noisy mean: each cell within its bound
    // (`judgeConstantSkyCells`), and the mean over cells within 0.5%.
    const { judgement, bySurface } = await judgeConstantSkyCells(device, scenario, state, 'ceilingOnly');
    if (judgement.cells < 100) fail(`only ${judgement.cells} ceiling cells to judge`);
    if (judgement.off > Math.ceil(SKY_CELLS_ALLOWED_OFF * judgement.cells)) fail(`${judgement.off} of ${judgement.cells} ceiling cells are past their bound off rho L (1 - F) (worst ${(judgement.worst * 100).toFixed(2)}%)`);
    if (Math.abs(judgement.meanRatio - 1) > 0.005) fail(`the ceiling cells average ${judgement.meanRatio.toFixed(4)} x rho L (1 - F)`);
    // The done-when: second-bounce misses stay 0. The ceiling's kept rays go down; nearly all
    // miss, the rest hit the black patch (no light). The sky reaches the ceiling through its own
    // light sample, so a miss must add a sample of no light (register D37): every ceiling cell's
    // indirect value is exactly 0, with samples counted.
    const indirect = await indirectOfSurface(device, scenario, state, 'ceiling');
    let withSamples = 0;
    let lit = 0;
    let largest = 0;
    for (const value of indirect.cells.values()) {
        if (!(value[3] > 0)) continue;
        withSamples++;
        const mean = Math.max(value[0], value[1], value[2]) / value[3];
        largest = Math.max(largest, mean);
        if (mean !== 0) lit++;
    }
    if (indirect.misses === 0) fail('no second-bounce ray missed');
    if (withSamples < 100) fail(`only ${withSamples} ceiling cells have indirect samples`);
    if (lit > 0) fail(`${lit} of ${withSamples} ceiling cells hold indirect light (largest ${largest.toFixed(5)}): a miss read the sky, which the ceiling already gets as its light`);
    const rays = meanRayReading(state, 'ceiling');
    return { failures, data: { expectedOpen: OPEN_SKY_RADIANCE, reservoirs, cells: judgement, bySurface, secondBounce: { entries: indirect.entries, misses: indirect.misses, cellsWithIndirectSamples: withSamples, cellsWithIndirectLight: lit, largestIndirect: largest }, rays } };
}

// ---- Section 18: the corner under a constant sky ------------------------------------------------------

/**
 * The share of cells allowed past their bound. The bound is five standard deviations of the
 * shadow rays' spread, but a cell of few samples where F is small counts its blocked rays as a
 * Poisson count, whose tail is heavier than the normal's; and a cell whose tile is on the update
 * list but which got no sample this frame blends in a zero (Capsaicin's `UpdateTiles`, kept for
 * the direct values, register D37), so a sparse cell far out reads a few percent low (seen to
 * 7% on the open floor, where every sample is exactly ρ L). The means are held tight.
 */
const SKY_CELLS_ALLOWED_OFF = 1e-3;

/**
 * The second bounce on the corner's floor and wall: each kept ray lands on the other surface
 * (or misses, 0) and brings its direct value ρ L (1 − F) there, × BRDF / pdf = ρ. Over the
 * floor point's cosine-weighted directions: ρ × the mean of that (`rings` × `sectors`
 * stratified directions, whose sector edges fall on the corner's plane, where the other
 * surface begins).
 * The wall's 40 m width and the floor's 400 m are the polygons'.
 */
function cornerIndirect(hit: Vec3, surface: 'floor' | 'wall', rings = 16, sectors = 32): number {
    const point = (surface === 'floor' ? CORNER_FLOOR : CORNER_WALL_SURFACE).snap(hit);
    const normal: Vec3 = surface === 'floor' ? UP : [0, 0, 1];
    const tangent: Vec3 = surface === 'floor' ? [1, 0, 0] : [1, 0, 0];
    const bitangent = vCross(normal, tangent);
    let sum = 0;
    for (let ring = 0; ring < rings; ring++) {
        const u = (ring + 0.5) / rings;
        const radius = Math.sqrt(u);
        const up = Math.sqrt(1 - u);
        for (let sector = 0; sector < sectors; sector++) {
            const angle = (2 * Math.PI * (sector + 0.5)) / sectors;
            const d: Vec3 = [0, 1, 2].map((axis) => tangent[axis] * radius * Math.cos(angle) + bitangent[axis] * radius * Math.sin(angle) + normal[axis] * up) as unknown as Vec3;
            if (surface === 'floor') {
                // Toward the wall's front (z = 0, x in [-20, 20], y in [0, 10]).
                if (!(d[2] < 0)) continue;
                const t = -point[2] / d[2];
                const q: Vec3 = [point[0] + t * d[0], point[1] + t * d[1], 0];
                if (Math.abs(q[0]) > 20 || q[1] < 0 || q[1] > 10) continue;
                sum += constantSkyDirect(q, CORNER_WALL_SURFACE);
            } else {
                // Toward the floor in front of the wall (y = 0, z in [0, 200]).
                if (!(d[1] < 0)) continue;
                const t = -point[1] / d[1];
                const q: Vec3 = [point[0] + t * d[0], 0, point[2] + t * d[2]];
                if (Math.abs(q[0]) > 200 || q[2] > 200) continue;
                sum += constantSkyDirect(q, CORNER_FLOOR);
            }
        }
    }
    return (SKY_ALBEDO * sum) / (rings * sectors);
}

/**
 * `cornerIndirect` tabulated near the corner (|x| ≤ 4 m, up to 4 m from the corner line, every
 * 0.5 m across and 5 cm out, 32 × 64 directions a point), read bilinearly: every near cell's
 * every entry can then be judged. Made once a surface, when first read.
 */
const CORNER_TABLE = { halfX: 4, reach: 4, xStep: 0.5, distanceStep: 0.05 } as const;
const cornerTables = new Map<'floor' | 'wall', Float64Array>();

function cornerIndirectAt(hit: Vec3, surface: 'floor' | 'wall'): number {
    const columns = Math.round((2 * CORNER_TABLE.halfX) / CORNER_TABLE.xStep) + 1;
    const rows = Math.round(CORNER_TABLE.reach / CORNER_TABLE.distanceStep) + 1;
    let table = cornerTables.get(surface);
    if (!table) {
        table = new Float64Array(columns * rows);
        for (let row = 0; row < rows; row++) {
            for (let column = 0; column < columns; column++) {
                const x = -CORNER_TABLE.halfX + column * CORNER_TABLE.xStep;
                const distance = row * CORNER_TABLE.distanceStep;
                table[row * columns + column] = cornerIndirect(surface === 'floor' ? [x, 0, distance] : [x, distance, 0], surface, 32, 64);
            }
        }
        cornerTables.set(surface, table);
    }
    const fx = Math.min(Math.max((hit[0] + CORNER_TABLE.halfX) / CORNER_TABLE.xStep, 0), columns - 1);
    const fd = Math.min(Math.max((surface === 'floor' ? hit[2] : hit[1]) / CORNER_TABLE.distanceStep, 0), rows - 1);
    const x0 = Math.min(Math.floor(fx), columns - 2);
    const d0 = Math.min(Math.floor(fd), rows - 2);
    const tx = fx - x0;
    const td = fd - d0;
    const at = (row: number, column: number): number => table[row * columns + column];
    return (1 - td) * ((1 - tx) * at(d0, x0) + tx * at(d0, x0 + 1)) + td * ((1 - tx) * at(d0 + 1, x0) + tx * at(d0 + 1, x0 + 1));
}

async function checkSkyCorner(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`sky, corner: ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const reservoirs = checkSkyOnlyReservoirs(state, fail, true);
    // The view factors: the infinite-strip answer for a floor point in front of an endless wall
    // of height H, F = (1 - x / sqrt(x² + H²)) / 2, against Lambert's contour on a very wide wall.
    const strip = (x: number, height: number): number => (1 - x / Math.hypot(x, height)) / 2;
    const contourCheck = Math.max(...[0.1, 1, 3].map((x) => Math.abs(viewFactor([0, 0, x], UP, [[-1e5, 0, 0], [1e5, 0, 0], [1e5, 10, 0], [-1e5, 10, 0]]) - strip(x, 10))));
    if (contourCheck > 1e-4) fail(`the page's view factor is ${contourCheck} off the infinite strip's`);
    // The direct values: floor ρ L (1 − F_wall), wall ρ L (1 − F_floor).
    const { judgement, bySurface } = await judgeConstantSkyCells(device, scenario, state, 'corner');
    if (judgement.cells < 100) fail(`only ${judgement.cells} cells to judge`);
    if (judgement.off > Math.ceil(SKY_CELLS_ALLOWED_OFF * judgement.cells)) fail(`${judgement.off} of ${judgement.cells} cells are past their bound off rho L (1 - F) (worst ${(judgement.worst * 100).toFixed(2)}%)`);
    if (Math.abs(judgement.meanRatio - 1) > 0.01) fail(`the cells average ${judgement.meanRatio.toFixed(4)} x rho L (1 - F)`);
    // The shadow ray matters here: the cells that see the wall over half their sky must read so.
    const halfShaded = judgement.byOpenShare['0.5'];
    if (!halfShaded || halfShaded[1] < 50) fail(`only ${halfShaded?.[1] ?? 0} cells see half their sky hidden`);
    else if (Math.abs(halfShaded[0] - 1) > 0.02) fail(`the cells with half their sky hidden average ${halfShaded[0].toFixed(4)} x rho L (1 - F)`);
    // The second bounce, near the corner (within 4 m of it, |x| ≤ 4): each cell's indirect value
    // against the quadrature averaged over this frame's entries in it, the cells whose entries
    // are all on one surface. One cell's kept rays are few and land on light or miss, so a cell
    // is a noisy mean (its spread is recorded): only the ratio of the sums over every such cell
    // is judged, within 3%, its standard error (recorded) under 1%.
    const capacity = scenario.cache.capacity;
    const near = new Map<number, { surface: Surface; points: Vec3[]; mixed: boolean }>();
    state.visibilityCell.forEach((cell, visibility) => {
        const ray = state.rays[state.visibilityQuery[visibility]];
        const entry = near.get(cell) ?? { surface: ray.surface, points: [], mixed: false };
        if (entry.surface !== ray.surface) entry.mixed = true;
        entry.points.push(ray.hit);
        near.set(cell, entry);
    });
    const judgedCells = [...near].filter(([, { surface, points, mixed }]) => !mixed && (surface === 'floor' || surface === 'wall') && points.every((point) => Math.abs(point[0]) <= CORNER_TABLE.halfX && point[1] <= CORNER_TABLE.reach && point[2] <= CORNER_TABLE.reach));
    const tiles = [...new Set(judgedCells.map(([cell]) => Math.floor(cell / capacity.numCellsPerTile)))];
    const indirectValues = await finestValues(device, scenario.cache, tiles, true);
    // Per surface, the cells' measured and expected values.
    const pairs: Record<string, [number, number][]> = { floor: [], wall: [] };
    const byDistance: Record<string, [number, number]> = {};
    for (const [cell, { surface, points }] of judgedCells) {
        const value = indirectValues.get(cell);
        if (!value || !(value[3] > 0)) continue;
        const side = surface as 'floor' | 'wall';
        const expected = points.reduce((sum, point) => sum + cornerIndirectAt(point, side), 0) / points.length;
        const measured = (value[0] + value[1] + value[2]) / (3 * value[3]);
        pairs[side].push([measured, expected]);
        // By distance from the corner line, for the record.
        const distance = side === 'floor' ? points[0][2] : points[0][1];
        const bucket = `${side} ${distance < 0.25 ? '<0.25' : distance < 1 ? '0.25-1' : '1-4'} m`;
        const entry = byDistance[bucket] ?? [0, 0];
        entry[0] += measured;
        entry[1] += expected;
        byDistance[bucket] = entry;
    }
    const indirectRatios = Object.fromEntries(
        Object.entries(pairs).map(([surface, cells]) => {
            // The ratio of the sums (cells where little light is expected count less), and its
            // standard error from the cells' residuals (the delta method); the cells' own spread.
            const measuredSum = cells.reduce((total, [measured]) => total + measured, 0);
            const expectedSum = cells.reduce((total, [, expected]) => total + expected, 0);
            const sumRatio = measuredSum / Math.max(expectedSum, 1e-30);
            const residual = Math.sqrt(cells.reduce((total, [measured, expected]) => total + (measured - sumRatio * expected) ** 2, 0) / Math.max(cells.length - 1, 1));
            const sumRatioError = (residual * Math.sqrt(cells.length)) / Math.max(expectedSum, 1e-30);
            const ratios = cells.map(([measured, expected]) => measured / expected);
            const meanRatio = ratios.reduce((total, ratio) => total + ratio, 0) / Math.max(ratios.length, 1);
            const cellSpread = Math.sqrt(ratios.reduce((total, ratio) => total + (ratio - meanRatio) ** 2, 0) / Math.max(ratios.length - 1, 1));
            return [surface, { cells: cells.length, sumRatio, sumRatioError, meanRatio, cellSpread }];
        }),
    );
    const indirectByDistance = Object.fromEntries(Object.entries(byDistance).map(([bucket, [measured, expected]]) => [bucket, measured / expected]));
    for (const [surface, { cells, sumRatio, sumRatioError }] of Object.entries(indirectRatios)) {
        if (cells < 500) fail(`only ${cells} ${surface} cells near the corner have indirect samples`);
        else if (Math.abs(sumRatio - 1) > 0.03) fail(`the ${surface}'s indirect values near the corner sum to ${sumRatio.toFixed(4)} x the second bounce's (standard error ${sumRatioError.toFixed(4)})`);
        else if (sumRatioError > 0.01) fail(`the ${surface}'s indirect sum is too noisy to judge (standard error ${sumRatioError.toFixed(4)})`);
    }
    // Where the second bounce loses its percent, for the record: this frame's second-bounce
    // samples of the near first hits, as the cache made them (the landing cell's filtered direct
    // value, `secondBounceSamples`) against ρ × the landing point's own ρ L (1 − F) (a miss 0).
    const samples = await secondBounceSamples(device, scenario, state);
    const landing: Record<string, { throughCache: number; atPoint: number; entries: number }> = {};
    state.multibounceQuery.forEach((query, entry) => {
        const ray = state.rays[state.visibilityQuery[query]];
        if ((ray.surface !== 'floor' && ray.surface !== 'wall') || Math.abs(ray.hit[0]) > CORNER_TABLE.halfX || ray.hit[1] > CORNER_TABLE.reach || ray.hit[2] > CORNER_TABLE.reach) return;
        const record = landing[ray.surface] ?? { throughCache: 0, atPoint: 0, entries: 0 };
        record.entries++;
        record.throughCache += samples[entry];
        if (state.multibounceCell[entry] !== MULTIBOUNCE_MISS_CELL) {
            const hit = multibounceHit(state, entry);
            const surface = ray.surface === 'floor' ? CORNER_WALL_SURFACE : CORNER_FLOOR;
            record.atPoint += SKY_ALBEDO * constantSkyDirect(hit, surface);
        }
        landing[ray.surface] = record;
    });
    const secondBounceLanding = Object.fromEntries(Object.entries(landing).map(([surface, { throughCache, atPoint, entries }]) => [surface, { entries, throughCacheOverAtPoint: throughCache / Math.max(atPoint, 1e-30) }]));
    return { failures, data: { contourCheck, reservoirs, cells: judgement, bySurface, indirect: indirectRatios, indirectByDistance, secondBounceLanding, cornerIndirectExamples: { floorAt0_5m: cornerIndirect([0, 0, 0.5], 'floor'), wallAt0_5m: cornerIndirect([0, 0.5, 0], 'wall') } } };
}

// ---- Section 19: hits with no light count (register D50) --------------------------------------------

/** One kept quantisation step of `HashGridCache_QuantizeRadiance` (1/1000). */
const QUANTISATION_STEP = 1e-3;

/**
 * Section 19's floor: corrugated, ridges along z every 2 cm, faces sloping 45° (1 cm up, 1 cm
 * across), so a cell (about 9 cm a side 2 m from the camera) holds both slopes. Built face by
 * face with flat normals: face k is triangles 2k and 2k + 1, rising (normal (−1, 1, 0) / √2)
 * for even k, falling ((1, 1, 0) / √2) for odd k.
 */
const CORRUGATION = { width: 0.01, height: 0.01, halfLength: 10 } as const;
/** Its probes: a 4 m patch 2 m up, facing down, the camera 1 m under it looking up. */
const CORRUGATION_PATCH = { y: 2, half: 2 } as const;
const CORRUGATION_CAMERA: PageCamera = { position: [0, 1, 0], target: [0, 2, 0], fovYDeg: 60, near: 0.05, far: 1000 };
/**
 * The sun, toward-light direction 60° from the zenith on the −x side, casting no shadow: the
 * rising faces see it at cos 15° = 0.966, the falling faces face away (cos 105° < 0), so their
 * hits have no light to draw (a black sky).
 */
const CORRUGATION_SUN: LightDescriptor = { type: 'directional', direction: [Math.sin(Math.PI / 3), -Math.cos(Math.PI / 3), 0], color: [1, 1, 1], intensity: SUN_INTENSITY, castsShadows: false, label: 'low sun' };
const RISING_RADIANCE = (ALBEDO / Math.PI) * SUN_INTENSITY * Math.cos(Math.PI / 12);

function corrugatedFloor(platform: GIPlatformImpl): void {
    const { width, height, halfLength } = CORRUGATION;
    const faces = Math.round((2 * halfLength) / width);
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    for (let face = 0; face < faces; face++) {
        const x0 = -halfLength + face * width;
        const rising = face % 2 === 0;
        const y0 = rising ? 0 : height;
        const y1 = rising ? height : 0;
        const normal = rising ? [-Math.SQRT1_2, Math.SQRT1_2, 0] : [Math.SQRT1_2, Math.SQRT1_2, 0];
        const base = face * 4;
        // Counter-clockwise seen from the normal's side.
        positions.push(x0, y0, halfLength, x0 + width, y1, halfLength, x0 + width, y1, -halfLength, x0, y0, -halfLength);
        for (let corner = 0; corner < 4; corner++) normals.push(...normal);
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
    const scene = platform.scene;
    const mesh = scene.createMesh({ positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices), label: 'corrugated floor' });
    const material = scene.createMaterial({ albedo: [ALBEDO, ALBEDO, ALBEDO], label: 'corrugated floor' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'corrugated floor' });
}

/**
 * A cell's direct value is the mean of its samples, every hit one: the rising faces' hits lit
 * (albedo / π × E cos 15°), the falling faces' hits samples of no light. So each cell reads
 * that times its rising share, from this frame's first-bounce entries in it (only hits on a
 * face's front and below the patch are counted; cells with other hits are left out). Summed
 * over the cells with at least 4 entries, within 2% (one frame's share stands for the cell's
 * history, which averages about 31 frames). Counting only the lit hits, as Capsaicin's
 * `GenerateReservoirs` does, every cell would read the rising faces' radiance, about twice.
 */
async function checkUnlitHits(device: GPUDevice, scenario: Scenario): Promise<Checks> {
    const failures: string[] = [];
    const fail = (message: string): void => {
        if (failures.length < 12) failures.push(`unlit hits (${scenario.technique.settings.useMultibounce ? 'multibounce' : 'single bounce'}): ${message}`);
    };
    const state = await readFrameState(device, scenario);
    const capacity = scenario.cache.capacity;
    const byCell = new Map<number, { rising: number; falling: number; other: number }>();
    let risingHits = 0;
    let fallingHits = 0;
    state.visibilityCell.forEach((cell, visibility) => {
        const ray = state.rays[state.visibilityQuery[visibility]];
        const entry = byCell.get(cell) ?? { rising: 0, falling: 0, other: 0 };
        const onFloor = ray.hit[1] < CORRUGATION.height + 0.01;
        if (!onFloor || !ray.frontFace) entry.other++;
        else if (Math.floor(ray.triangle / 2) % 2 === 0) {
            entry.rising++;
            risingHits++;
        } else {
            entry.falling++;
            fallingHits++;
        }
        byCell.set(cell, entry);
    });
    const judged = [...byCell].filter(([, entry]) => entry.other === 0 && entry.rising + entry.falling >= 4);
    const mixedCells = judged.filter(([, entry]) => entry.rising > 0 && entry.falling > 0).length;
    const tiles = [...new Set(judged.map(([cell]) => Math.floor(cell / capacity.numCellsPerTile)))];
    const values = await finestValues(device, scenario.cache, tiles, false);
    let measuredSum = 0;
    let expectedSum = 0;
    let litOnlySum = 0;
    let cells = 0;
    let unsampled = 0;
    for (const [cell, entry] of judged) {
        const value = values.get(cell);
        if (!value || !(value[3] > 0)) {
            unsampled++;
            continue;
        }
        cells++;
        measuredSum += (value[0] + value[1] + value[2]) / (3 * value[3]);
        expectedSum += (RISING_RADIANCE * entry.rising) / (entry.rising + entry.falling);
        litOnlySum += entry.rising > 0 ? RISING_RADIANCE : 0;
    }
    const ratio = measuredSum / Math.max(expectedSum, 1e-30);
    if (risingHits === 0 || fallingHits === 0) fail(`${risingHits} hits on rising faces and ${fallingHits} on falling ones: both are needed`);
    if (mixedCells < 500) fail(`only ${mixedCells} cells hold both slopes`);
    if (unsampled > 0) fail(`${unsampled} cells with entries this frame have no samples`);
    if (Math.abs(ratio - 1) > TOLERANCE) fail(`the cells sum to ${ratio.toFixed(4)} x albedo / pi x E cos 15° x their rising share (the lit hits alone would read ${(litOnlySum / expectedSum).toFixed(4)} x)`);
    return { failures, data: { risingRadiance: RISING_RADIANCE, risingHits, fallingHits, cells, mixedCells, unsampled, ratio, litOnlyWouldRead: litOnlySum / Math.max(expectedSum, 1e-30) } };
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
    await run('cornerBypass', [sun(SUN_TRAVEL)], { slab: true, lowSlab: true }, CONVERGENCE_FRAMES, checkCornerBypass, MULTIBOUNCE);
    await run('clearOnChange', [sun(SUN_TRAVEL)], {}, 10, checkClearOnChange);
    await run('strips', [FURNACE_SUN], { furnace: true, strips: true }, CONVERGENCE_FRAMES, checkStrips, MULTIBOUNCE);
    await run('bypassAverage', [FURNACE_SUN], { furnace: true }, CONVERGENCE_FRAMES, checkBypassAverage, BYPASS_ALL);

    // The sky as a light (RC3.4), GI-1.2's multibounce on (Capsaicin's default).
    await run('skyOpenFloor', [], { sky: { layout: 'openFloor', environment: 'constant' } }, CONVERGENCE_FRAMES, checkSkyOpenFloor, MULTIBOUNCE);
    await run('skyAndSun', [SKY_SUN], { sky: { layout: 'openFloor', environment: 'coded' } }, CONVERGENCE_FRAMES, checkSkyAndSun, MULTIBOUNCE);
    await run('skyCeilingOnly', [], { sky: { layout: 'ceilingOnly', environment: 'constant' } }, CONVERGENCE_FRAMES, checkSkyCeilingOnly, MULTIBOUNCE);
    await run('skyCorner', [], { sky: { layout: 'corner', environment: 'constant' } }, CONVERGENCE_FRAMES, checkSkyCorner, MULTIBOUNCE);

    // Hits with no light to draw count as samples of no light (register D50), with multibounce
    // (GenerateReservoirs after the resolve list) and without (its early return, register D35).
    await run('unlitHits', [CORRUGATION_SUN], { corrugated: true }, CONVERGENCE_FRAMES, checkUnlitHits, MULTIBOUNCE);
    await run('unlitHitsSingleBounce', [CORRUGATION_SUN], { corrugated: true }, CONVERGENCE_FRAMES, checkUnlitHits, SINGLE_BOUNCE);

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
