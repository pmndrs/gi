/**
 * CPU twins of the SHaRC-style hash cache's WGSL (`sharc-wgsl.ts`), for the
 * unit tests and the GPU page's expectations: the cell (LOD level, normal
 * bin, both hashes), the resolve's running mean, and the furnace's analytic
 * answers for each update-path setting. Ported with the WGSL from the
 * pmndrs/pathtracer project's `bench/nrc/hash-grid.ts` (dd771c8, MIT).
 */

type Vec3 = readonly [number, number, number];

const f32 = Math.fround;

/** u32 multiply, wrapping. */
function mul(a: number, b: number): number {
    return Math.imul(a, b) >>> 0;
}

/** `sharcMix`: the PCG hash. */
export function sharcMix(x: number): number {
    const s = (mul(x >>> 0, 747796405) + 2891336453) >>> 0;
    const w = mul(((s >>> ((s >>> 28) + 4)) ^ s) >>> 0, 277803737);
    return ((w >>> 22) ^ w) >>> 0;
}

/** `sharcMix2`: murmur3's finaliser. */
export function sharcMix2(x: number): number {
    let h = mul(x >>> 0, 0x85ebca6b);
    h = (h ^ (h >>> 13)) >>> 0;
    h = mul(h, 0xc2b2ae35);
    h = (h ^ (h >>> 16)) >>> 0;
    return h;
}

/** The LOD rule's inputs (`SharcParams`): the eye, a cell's size per metre of distance, the smallest cell. */
export interface SharcCellConstants {
    readonly eye: Vec3;
    readonly cellSizePerMetre: number;
    readonly minCell: number;
}

/** A cell's size per metre of distance: one pixel's footprint (2 tan(fovY / 2) / height) times `cellPixels`. */
export function sharcCellSizePerMetre(fovY: number, height: number, cellPixels: number): number {
    return ((2 * Math.tan(fovY / 2)) / height) * cellPixels;
}

/** `sharcLevel` (in f32 steps, as the GPU computes it). */
export function sharcLevel(position: Vec3, constants: SharcCellConstants, maxLevel = 24): number {
    const distance = f32(Math.hypot(position[0] - constants.eye[0], position[1] - constants.eye[1], position[2] - constants.eye[2]));
    const size = Math.max(f32(distance * f32(constants.cellSizePerMetre)), f32(constants.minCell));
    return Math.min(Math.max(Math.floor(Math.log2(f32(size / f32(constants.minCell)))), 0), maxLevel);
}

export function sharcCellSize(level: number, minCell: number): number {
    return f32(f32(minCell) * 2 ** level);
}

/** `sharcNormalBin`: 1-6 by the dominant axis and sign, 0 with one bin. */
export function sharcNormalBin(normal: Vec3, normalBins = true): number {
    if (!normalBins) return 0;
    const a = normal.map(Math.abs);
    if (a[0] >= a[1] && a[0] >= a[2]) return normal[0] < 0 ? 2 : 1;
    if (a[1] >= a[2]) return normal[1] < 0 ? 4 : 3;
    return normal[2] < 0 ? 6 : 5;
}

/** `sharcKey`: the first slot of the cell's bucket and its checksum (never 0). */
export function sharcKey(position: Vec3, bin: number, level: number, minCell: number, log2Entries: number, bucket = 8): readonly [number, number] {
    const size = sharcCellSize(level, minCell);
    const cell = position.map((value) => Math.floor(f32(f32(value) / size)) >>> 0);
    const levelAndBin = (level | (bin << 8)) >>> 0;
    const h = sharcMix((sharcMix((sharcMix((sharcMix((cell[0] + 0x9e3779b9) >>> 0) ^ cell[1]) >>> 0) ^ cell[2]) >>> 0) ^ levelAndBin) >>> 0);
    const g = sharcMix2(
        (sharcMix2((sharcMix2((sharcMix2((cell[2] ^ 0x27d4eb2f) >>> 0) + cell[0]) >>> 0) + cell[1]) >>> 0) + mul(levelAndBin, 0x165667b1)) >>> 0,
    );
    const mask = (2 ** log2Entries - 1) >>> 0;
    return [(h & mask & ~(bucket - 1)) >>> 0, Math.max(g, 1)];
}

/** One frame of `SharcResolve` for a slot with `count` new samples summing to `sum`: the running mean and its weight. */
export function sharcResolveStep(previous: { readonly radiance: number; readonly weight: number }, sum: number, count: number, historyFrames: number): { radiance: number; weight: number } {
    if (count === 0) return { ...previous };
    const history = previous.weight * (1 - 1 / Math.max(1, historyFrames));
    const weight = history + count;
    return { radiance: (previous.radiance * history + sum) / weight, weight };
}

/** The settled weight of a cell that gets `samplesPerFrame` samples every frame: n / (1 − decay) = n · historyFrames. */
export function sharcSettledWeight(samplesPerFrame: number, historyFrames: number): number {
    return samplesPerFrame * Math.max(1, historyFrames);
}

/**
 * The furnace (two facing planes, the floor lit by a sun E straight down that
 * casts no shadow, the ceiling unlit, albedo ρ, a black sky): the light
 * leaving the ceiling, which the probe rays from the floor read, for each
 * update-path setting. `floorFill` is the share of the floor a ray from the
 * ceiling finds (the strips: 1/2); a ray from the floor always finds the
 * ceiling.
 *
 * - `bounceRays` 0: 0 (the ceiling has no direct light).
 * - 1 without the cache tail: ρ × floorFill × ρE/π, GI-1.2's second bounce.
 * - with the cache tail (and floor cells in the cache): every bounce. The
 *   floor leaves L_f = ρE/π + ρ L_c and the ceiling L_c = ρ floorFill L_f,
 *   so L_c = ρ floorFill (ρE/π) / (1 − ρ² floorFill).
 */
export function sharcFurnaceCeiling(albedo: number, sunIrradiance: number, mode: 'direct' | 'one-bounce' | 'every-bounce', floorFill = 1): number {
    const floorDirect = (albedo * sunIrradiance) / Math.PI;
    if (mode === 'direct') return 0;
    if (mode === 'one-bounce') return albedo * floorFill * floorDirect;
    return (albedo * floorFill * floorDirect) / (1 - albedo * albedo * floorFill);
}

/** The furnace's floor for the every-bounce setting: L_f = ρE/π + ρ L_c. */
export function sharcFurnaceFloor(albedo: number, sunIrradiance: number, floorFill = 1): number {
    return (albedo * sunIrradiance) / Math.PI + albedo * sharcFurnaceCeiling(albedo, sunIrradiance, 'every-bounce', floorFill);
}
