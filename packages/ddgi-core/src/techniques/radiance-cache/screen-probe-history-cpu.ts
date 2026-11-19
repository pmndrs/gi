/**
 * CPU twins of the screen probes' history arithmetic (RC4.1): the ray
 * guiding's CDF, the blend's cell rule and biased hysteresis, the cache's
 * normal packing and LRU reorder, and `ReprojectScreenProbes` for one probe
 * tile. Unit tests check them against hand-worked values; the GPU pages run
 * them on the GPU's own inputs.
 *
 * Also the derivation of register D41: a model of `SampleScreenProbes` and
 * `BlendScreenProbes` with history under a constant sky
 * ({@link simulateConstantSky}), with Capsaicin's empty-cell rule and with
 * ours.
 */

import {
    CpuRandom,
    INVALID_ID,
    mapToHemiOctahedron,
    mapToHemiOctahedronInverse,
    projectProbe,
    shIrradiance,
    toProbeFrame,
    getOrthoVectors,
    unpackSeed,
    packSeed,
    calculateHaltonSequence,
    type Direction,
} from './capsaicin-math.ts';
import { PROBE_CELLS, PROBE_SIZE, PROBE_SPAWN_TILE_SIZE } from './settings.ts';

/** `luminance` (math/color.hlsl): Rec. 709 weights. */
export function luminance(rgb: readonly [number, number, number]): number {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

/**
 * `BlendScreenProbes`' biased temporal hysteresis (gi1.comp:1262-1268): the
 * share of the previous radiance a cell keeps, from the new luminance `a` and
 * the previous `b`. 0 when `a < 2b`; up to 0.95² against a sudden brightening.
 */
export function temporalBlendFactor(a: number, b: number): number {
    const t = Math.min(Math.max(Math.max(a - b - Math.min(a, b), 0) / Math.max(Math.max(a, b), 1e-4), 0), 0.95);
    return t * t;
}

/**
 * The blend's rule: Capsaicin's (the backup for every unreached cell, then the
 * hysteresis; gi1.comp:1212-1269); `fixed`, D41 alone (an unreached cell keeps
 * its value, then Capsaicin's hysteresis); `running`, ours (D41 and the
 * running mean of D46, with the fill of D47).
 */
export type EmptyCellRule = 'capsaicin' | 'fixed' | 'running';

/** The most samples a cell's running mean counts (D46): the default of `RadianceCacheSettings.historySampleCap`. */
export const HISTORY_SAMPLE_CAP = 8;

/**
 * One probe's blend with the running mean (D46): `sums` and `counts` this
 * frame's rays per cell, `previous` the previous radiance and
 * `previousCounts` the samples it stands for (0: no history). A cell with
 * rays moves to their mean by k / min(n + k, cap); a cell without keeps its
 * value and count; a cell with neither is a fill (the reached cells' mean),
 * count 0, which is never history (D47). With `expectedRays` (the rays each
 * cell expected under ray guiding, register D62), a cell's step is k over its
 * expected rays times n, n its reached frames up to the cap, so its expected
 * step does not depend on what it reads; n counts the frames since its history
 * began, reached or not, so it doesn't either.
 */
export function blendRunningMean(
    sums: readonly number[],
    counts: readonly number[],
    previous: ReadonlyArray<number | null>,
    previousCounts: readonly number[],
    cap = HISTORY_SAMPLE_CAP,
    expectedRays?: readonly number[],
): { values: number[]; counts: number[] } {
    let reached = 0;
    let meanSum = 0;
    counts.forEach((count, cell) => {
        if (count === 0) return;
        reached++;
        meanSum += sums[cell] / count;
    });
    const fill = meanSum / Math.max(reached, 1);
    const values: number[] = [];
    const nextCounts: number[] = [];
    for (let cell = 0; cell < sums.length; cell++) {
        const before = previous[cell];
        const history = before === null ? 0 : previousCounts[cell];
        const k = counts[cell];
        if (k > 0) {
            const mean = sums[cell] / k;
            const corrected = expectedRays !== undefined && cap > 1;
            const n = corrected ? Math.min(history + 1, cap) : Math.min(history + k, cap);
            const weight = corrected ? Math.min(k / (Math.max(expectedRays[cell], 1e-6) * n), 1) : Math.min(k / n, 1);
            values.push(history === 0 || before === null ? mean : before + (mean - before) * weight);
            nextCounts.push(n);
        } else if (before !== null && history > 0) {
            values.push(before);
            // Corrected (D62), the count is frames since the history began, reached or not: it may not
            // depend on whether guiding sent a ray here, or a dark cell would take bigger steps.
            nextCounts.push(expectedRays !== undefined && cap > 1 ? Math.min(history + 1, cap) : history);
        } else {
            values.push(fill);
            nextCounts.push(0);
        }
    }
    return { values, counts: nextCounts };
}

/**
 * One probe's blend (gi1.comp:1193-1269) on per-cell sums: `sums[cell]` and
 * `counts[cell]` this frame's samples, `previous[cell]` the previous radiance
 * (`null` where the cell has no history; one channel here). Capsaicin gives
 * an unreached cell the backup, the reached cells' summed samples over their
 * number, divided again by the number of empty cells; we give it its previous
 * value, and with no history the mean of the reached cells' means.
 */
export function blendProbeCells(sums: readonly number[], counts: readonly number[], previous: ReadonlyArray<number | null>, rule: EmptyCellRule): number[] {
    let reached = 0;
    let backupSum = 0;
    for (let cell = 0; cell < PROBE_CELLS; cell++) {
        if (counts[cell] === 0) continue;
        reached++;
        backupSum += rule === 'capsaicin' ? sums[cell] : sums[cell] / counts[cell];
    }
    const mean = backupSum / Math.max(reached, 1);
    const backup = rule === 'capsaicin' ? mean / Math.max(PROBE_CELLS - reached, 1) : mean;
    return sums.map((sum, cell) => {
        const before = previous[cell];
        let value: number;
        if (counts[cell] > 0) value = sum / counts[cell];
        else if (rule === 'fixed' && before !== null) value = before;
        else value = backup;
        if (before !== null) value += (before - value) * temporalBlendFactor(value, before);
        return value;
    });
}

/**
 * `ScreenProbes_ScanRadiance` (screen_probes.hlsl:282-338): each cell's
 * exclusive prefix sum over the total, and the total. A total of 0 leaves the
 * CDF at 0 (register D42).
 */
export function cellCdf(weights: readonly number[]): { cdf: number[]; total: number } {
    const total = weights.reduce((sum, value) => sum + value, 0);
    let running = 0;
    const cdf = weights.map((weight) => {
        const value = total > 0 ? running / total : 0;
        running += weight;
        return value;
    });
    return { cdf, total };
}

/** `ScreenProbes_FindCellIndex` (screen_probes.hlsl:257-280): the last cell whose CDF is at most `s`. */
export function findCellIndex(cdf: readonly number[], s: number): number {
    let index = 0;
    let count = cdf.length;
    while (count > 0) {
        const half = count >> 1;
        const mid = index + half;
        if (cdf[mid] > s) {
            count = half;
        } else {
            index = mid + 1;
            count -= half + 1;
        }
    }
    return Math.max(index, 1) - 1;
}

/** The share of rays that sample their own cell under guiding (register D42): the default of `RadianceCacheSettings.rayGuidingUniformShare`. */
export const RAY_GUIDING_UNIFORM_SHARE = 0.125;

/**
 * The cell a ray samples (`SampleScreenProbes`, gi1.comp:969-972, with D42):
 * its own cell when its cell has no history, when the CDF is empty, or for a
 * `guide` draw below the uniform share; otherwise the CDF's cell for the draw
 * rescaled past the share. A `uniformShare` of 0 is Capsaicin's rule, as the
 * WGSL takes it: no test of the total, so an all-black CDF (all zeros) gives
 * the last cell.
 */
export function guidedCell(ownCell: number, ownHasHistory: boolean, cdf: { cdf: readonly number[]; total: number }, guide: number, uniformShare = RAY_GUIDING_UNIFORM_SHARE): number {
    if (!ownHasHistory) return ownCell;
    if (uniformShare <= 0) return findCellIndex(cdf.cdf, guide);
    if (!(cdf.total > 0) || guide < uniformShare || uniformShare >= 1) return ownCell;
    return findCellIndex(cdf.cdf, (guide - uniformShare) / (1 - uniformShare));
}

/** `packNormal` (pack.hlsl): three 10-bit snorms in one word. */
export function packNormal(value: Direction): number {
    const component = (x: number): number => Math.trunc(Math.min(Math.max(x, -1), 1) * 511 + 0.5 * Math.sign(x)) & 0x3ff;
    return (component(value[0]) | (component(value[1]) << 10) | (component(value[2]) << 20)) >>> 0;
}

/** `unpackNormal` (pack.hlsl). */
export function unpackNormal(packed: number): [number, number, number] {
    const signed = (shift: number): number => ((packed << shift) >> 22) / 511;
    return [signed(22), signed(12), signed(2)];
}

/** The off-screen cache's flag for an LRU entry never used, or newly allocated (gi1.comp:289, 871). */
export const CACHED_TILE_NEW = 0x80000000;

/**
 * `ReorderScreenProbes` (gi1.comp:1333-1354) with its scan: the entries not
 * used this frame (flag 1) keep their order at the front, the used ones
 * (flag 0) follow in their MRU order. Returns the next LRU list.
 */
export function reorderCachedTiles(lru: readonly number[], flags: readonly number[], mru: readonly number[]): number[] {
    const next = new Array<number>(lru.length).fill(-1);
    const unused = flags.slice(0, lru.length).reduce((sum, flag) => sum + flag, 0);
    let offset = 0;
    for (let entry = 0; entry < lru.length; entry++) {
        const target = flags[entry] === 0 ? mru[entry] + unused : offset;
        if (flags[entry] !== 0) offset += flags[entry];
        if (target >= 0 && target < next.length) next[target] = lru[entry];
    }
    return next;
}

// ---- ReprojectScreenProbes on the CPU ------------------------------------------------------------

/** What `ReprojectScreenProbes` reads, as functions of pixels (the GPU's own buffers in the pages). */
export interface ReprojectionInputs {
    readonly width: number;
    readonly height: number;
    readonly frameIndex: number;
    /** `ScreenProbesConstants.cell_size`, per unit distance. */
    readonly cellSize: number;
    readonly eye: Direction;
    /** This frame: sky, world position and normal at a pixel; the velocity (NDC, current − previous, y up). */
    readonly isSky: (x: number, y: number) => boolean;
    readonly worldPosition: (x: number, y: number) => Direction;
    readonly normal: (x: number, y: number) => Direction;
    readonly velocity: (x: number, y: number) => readonly [number, number];
    /** Last frame: its probe mask (level 0, by probe tile), its probe's world position and normal at a seed, its atlas texel. */
    readonly previousProbeMask: (tileX: number, tileY: number) => number;
    readonly previousWorldPosition: (x: number, y: number) => Direction;
    readonly previousNormal: (x: number, y: number) => Direction;
    readonly previousAtlas: (x: number, y: number) => readonly [number, number, number, number];
}

/** One probe tile's reprojection: the seed it found (or `INVALID_ID`), its 64 cells, and whether it was listed as an empty tile. */
export interface ReprojectedTile {
    readonly seed: number;
    /** Each lane's score (`(half(distance / cell size) << 16) | lane`), or -1 where the lane found no probe. */
    readonly scores: readonly number[];
    readonly cells: Array<readonly [number, number, number, number]>;
    readonly empty: boolean;
}

/**
 * `ReprojectScreenProbes` (gi1.comp:292-499) for probe tile (`tileX`,
 * `tileY`), with D41's backup and D47's fills. Sums are exact here (the GPU's are 64-bit,
 * D40), and quantised as the GPU quantises. `lane`, when given, takes that
 * pixel as the seed instead of the best one (a page comparing with the GPU
 * where f32 and f64 round a score to different halves).
 */
export function reprojectProbeTile(inputs: ReprojectionInputs, tileX: number, tileY: number, halfBits: (value: number) => number, forcedLane?: number): ReprojectedTile {
    const { width, height } = inputs;
    const dot = (a: Direction, b: Direction): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const distance = (a: Direction, b: Direction): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    const previousMaskAt = (x: number, y: number): number => {
        const [vx, vy] = inputs.velocity(x, y);
        const uv = [(x + 0.5) / width, (y + 0.5) / height];
        const previousUv = [uv[0] - 0.5 * vx, uv[1] + 0.5 * vy];
        const px = Math.trunc(previousUv[0] * width);
        const py = Math.trunc(previousUv[1] * height);
        if (px < 0 || py < 0 || px >= width || py >= height) return INVALID_ID;
        return inputs.previousProbeMask(Math.floor(px / PROBE_SIZE), Math.floor(py / PROBE_SIZE));
    };
    const surface = (x: number, y: number): boolean => x < width && y < height && !inputs.isSky(x, y);

    // The best pixel's score: (half(distance / cell size) << 16) | lane.
    let best = (halfBits(65504) << 16) | 0xffff;
    const scores = new Array<number>(PROBE_CELLS).fill(-1);
    for (let lane = 0; lane < PROBE_CELLS; lane++) {
        const x = tileX * PROBE_SIZE + (lane % PROBE_SIZE);
        const y = tileY * PROBE_SIZE + Math.floor(lane / PROBE_SIZE);
        if (!surface(x, y)) continue;
        const mask = previousMaskAt(x, y);
        if (mask === INVALID_ID) continue;
        const [sx, sy] = unpackSeed(mask);
        const world = inputs.worldPosition(x, y);
        const normal = inputs.normal(x, y);
        const cellSize = distance(inputs.eye, world) * inputs.cellSize;
        const probePosition = inputs.previousWorldPosition(sx, sy);
        const probeNormal = inputs.previousNormal(sx, sy);
        const offset: Direction = [probePosition[0] - world[0], probePosition[1] - world[1], probePosition[2] - world[2]];
        if (Math.abs(dot(offset, normal)) < cellSize && dot(normal, probeNormal) > 0.95) {
            const score = ((halfBits(Math.min(distance(probePosition, world) / cellSize, 65504)) << 16) | lane) >>> 0;
            scores[lane] = score;
            best = Math.min(best, score);
        }
    }
    const lane = forcedLane ?? best & 0xffff;
    const tileTopLeft = [tileX * PROBE_SIZE, tileY * PROBE_SIZE];
    if (lane === 0xffff) {
        // A disocclusion: listed as empty when the spawn's jitter is in another 8 x 8 tile of its 16 x 16.
        let empty = false;
        if (surface(tileTopLeft[0], tileTopLeft[1])) {
            const [hx, hy] = calculateHaltonSequence(inputs.frameIndex);
            const jitter = [Math.trunc(Math.min(hx * PROBE_SPAWN_TILE_SIZE, PROBE_SPAWN_TILE_SIZE - 1)), Math.trunc(Math.min(hy * PROBE_SPAWN_TILE_SIZE, PROBE_SPAWN_TILE_SIZE - 1))];
            const ratio = PROBE_SPAWN_TILE_SIZE / PROBE_SIZE;
            empty = Math.floor(jitter[0] / PROBE_SIZE) !== tileX % ratio || Math.floor(jitter[1] / PROBE_SIZE) !== tileY % ratio;
        }
        return { seed: INVALID_ID, scores, cells: [], empty };
    }

    const seedX = tileTopLeft[0] + (lane & 7);
    const seedY = tileTopLeft[1] + (lane >> 3);
    const mask = previousMaskAt(seedX, seedY);
    const [sx, sy] = unpackSeed(mask);
    const seedWorld = inputs.worldPosition(seedX, seedY);
    const seedNormal = inputs.normal(seedX, seedY);
    const probePosition = inputs.previousWorldPosition(sx, sy);
    const probeNormal = inputs.previousNormal(sx, sy);
    const { b1, b2 } = getOrthoVectors(probeNormal);
    const sums = Array.from({ length: PROBE_CELLS }, () => [0, 0, 0, 0]);
    const counts = new Array<number>(PROBE_CELLS).fill(0);
    for (let cell = 0; cell < PROBE_CELLS; cell++) {
        const radiance = inputs.previousAtlas(Math.floor(sx / PROBE_SIZE) * PROBE_SIZE + (cell % PROBE_SIZE), Math.floor(sy / PROBE_SIZE) * PROBE_SIZE + Math.floor(cell / PROBE_SIZE));
        const [lx, ly, lz] = mapToHemiOctahedron(((cell % PROBE_SIZE) + 0.5) / PROBE_SIZE, (Math.floor(cell / PROBE_SIZE) + 0.5) / PROBE_SIZE);
        const direction: Direction = [lx * b1[0] + ly * b2[0] + lz * probeNormal[0], lx * b1[1] + ly * b2[1] + lz * probeNormal[1], lx * b1[2] + ly * b2[2] + lz * probeNormal[2]];
        const hit: Direction = [probePosition[0] + direction[0] * radiance[3], probePosition[1] + direction[1] * radiance[3], probePosition[2] + direction[2] * radiance[3]];
        const reprojected: Direction = [hit[0] - seedWorld[0], hit[1] - seedWorld[1], hit[2] - seedWorld[2]];
        const length = Math.hypot(...reprojected);
        const unit: Direction = [reprojected[0] / length, reprojected[1] / length, reprojected[2] / length];
        // A fill (negative hit distance) is no history and isn't moved (D47).
        if (radiance[3] < 0 || !(dot(seedNormal, unit) > 0)) continue;
        const [u, v] = mapToHemiOctahedronInverse(toProbeFrame(unit, seedNormal));
        const remap = Math.min(Math.trunc(u * PROBE_SIZE), PROBE_SIZE - 1) + Math.min(Math.trunc(v * PROBE_SIZE), PROBE_SIZE - 1) * PROBE_SIZE;
        const values = [radiance[0], radiance[1], radiance[2], Math.min(length, 65504)];
        values.forEach((value, channel) => (sums[remap][channel] += Math.round(1e4 * value)));
        counts[remap]++;
    }
    const reachedMeans = sums.map((sum, cell) => (counts[cell] > 0 ? sum.map((value) => value / 1e4 / counts[cell]) : null));
    const reached = reachedMeans.filter((mean) => mean !== null) as number[][];
    const backup: [number, number, number, number] = [0, 1, 2].map((channel) => reached.reduce((sum, mean) => sum + mean[channel], 0) / Math.max(reached.length, 1)) as unknown as [number, number, number, number];
    // The fill's hit distance (D47).
    backup[3] = -1;
    const cells = reachedMeans.map((mean) => (mean ? ([mean[0], mean[1], mean[2], Math.min(mean[3], 65504)] as const) : backup));
    return { seed: packSeed(seedX, seedY), scores, cells, empty: false };
}

// ---- The constant-sky model (register D41) -------------------------------------------------------

export interface ConstantSkyOptions {
    readonly rule: EmptyCellRule;
    /** Capsaicin's guiding (no uniform share, the empty CDF's last cell) or ours (D42). */
    readonly guiding: 'capsaicin' | 'fixed';
    readonly frames: number;
    /** Probe tiles across a square torus (every tile has a probe; one in each 2 × 2 is spawned per frame). */
    readonly tiles?: number;
    /** Frames (1-based) at which to report E / (π L). */
    readonly report?: readonly number[];
    readonly seed?: number;
    /**
     * Each ray's radiance: the sky's 1 (`none`), or noisy with mean 1:
     * exponential (coefficient of variation 1), or 5 with probability 0.2
     * and 0 otherwise (`sparse`, a bright source few rays find).
     */
    readonly noise?: 'none' | 'exponential' | 'sparse';
    /** With the running mean: weight each step by the rays the cell expected (register D62). */
    readonly guidingCorrection?: boolean;
}

/**
 * `SampleScreenProbes` and `BlendScreenProbes` with history, on a plane under
 * a constant sky of radiance 1 seen by a still camera: every probe tile holds
 * a probe, and each frame one tile of every 2 × 2 (the 16 × 16 spawn tile) is
 * spawned at the Halton cycle's tile, guided by the 3 × 3 tiles' previous
 * radiance (all on the plane with the same frame, so a cell moves into the
 * same cell), with 64 rays drawn with replacement, every one reading the sky.
 * With `noise`, each ray's radiance is random with mean 1, so the model also
 * shows the bias of the blend's rule on noisy cells (D46).
 * Returns E / (π L) over all probes (the probe's SH with Capsaicin's
 * projection, `1/8` at the cell centre, and with ours, the solid angle over
 * the cell's mean basis, D25 and D34), and the share of cells no ray reached.
 *
 * The model leaves out the parts that cannot change a constant sky's value:
 * the reprojection (a still camera keeps every probe), the world cache (no
 * hit), the probe filter and the GI denoiser (averages of equal probes).
 */
/**
 * The rays each cell of a probe can expect from `guidedCell`'s draws (register
 * D62): its own ray where that isn't guided (always, or the uniform share),
 * plus every guided ray's chance of landing in it.
 */
export function expectedRaysPerCell(ownHasHistory: readonly boolean[], cdf: { cdf: readonly number[]; total: number }, uniformShare = RAY_GUIDING_UNIFORM_SHARE): number[] {
    const guided = ownHasHistory.filter(Boolean).length;
    return ownHasHistory.map((history, cell) => {
        const step = Math.max((cell + 1 < cdf.cdf.length ? cdf.cdf[cell + 1] : 1) - cdf.cdf[cell], 0);
        if (uniformShare <= 0) return (history ? 0 : 1) + guided * step;
        if (!(cdf.total > 0) || uniformShare >= 1) return 1;
        return (history ? uniformShare : 1) + guided * (1 - uniformShare) * step;
    });
}

export function simulateConstantSky(options: ConstantSkyOptions): Array<{ frame: number; capsaicinProjection: number; ourProjection: number; unreachedShare: number }> {
    const tiles = options.tiles ?? 16;
    const up: Direction = [0, 0, 1];
    const cosine = Array.from({ length: PROBE_CELLS }, (_, cell) => mapToHemiOctahedron(((cell % PROBE_SIZE) + 0.5) / PROBE_SIZE, (Math.floor(cell / PROBE_SIZE) + 0.5) / PROBE_SIZE)[2]);
    let probes: Array<number[] | null> = new Array(tiles * tiles).fill(null);
    // The running mean's counts per probe cell (D46): 0 is no history, a fill (D47).
    let probeCounts: number[][] = Array.from({ length: tiles * tiles }, () => new Array<number>(PROBE_CELLS).fill(0));
    const noise = options.noise ?? 'none';
    const random = new CpuRandom(options.seed ?? 1, 0);
    const report = new Set(options.report ?? [1, options.frames]);
    const results: Array<{ frame: number; capsaicinProjection: number; ourProjection: number; unreachedShare: number }> = [];
    let unreachedCells = 0;
    let spawnedCells = 0;
    for (let frame = 0; frame < options.frames; frame++) {
        const next = probes.map((probe) => probe?.slice() ?? null);
        const nextCounts = probeCounts.map((counts) => counts.slice());
        const jitter = [frame % 2, Math.floor(frame / 2) % 2];
        for (let ty = jitter[1]; ty < tiles; ty += 2) {
            for (let tx = jitter[0]; tx < tiles; tx += 2) {
                // The 3 x 3 taps' previous radiance, per cell.
                const tapSums = new Array<number>(PROBE_CELLS).fill(0);
                const tapCounts = new Array<number>(PROBE_CELLS).fill(0);
                const tapHistory = new Array<number>(PROBE_CELLS).fill(0);
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const index = ((ty + dy + tiles) % tiles) * tiles + ((tx + dx + tiles) % tiles);
                        const tap = probes[index];
                        if (!tap) continue;
                        tap.forEach((value, cell) => {
                            // Under D47 a fill is no history: the taps skip it.
                            if (options.rule === 'running' && probeCounts[index][cell] === 0) return;
                            tapSums[cell] += value;
                            tapCounts[cell]++;
                            tapHistory[cell] += probeCounts[index][cell];
                        });
                    }
                }
                const previous = tapSums.map((sum, cell) => (tapCounts[cell] > 0 ? sum / tapCounts[cell] : null));
                const previousCounts = tapHistory.map((sum, cell) => (tapCounts[cell] > 0 ? Math.floor(sum / tapCounts[cell]) : 0));
                const cdf = cellCdf(previous.map((value, cell) => (value ?? 0) * cosine[cell]));
                const sums = new Array<number>(PROBE_CELLS).fill(0);
                const counts = new Array<number>(PROBE_CELLS).fill(0);
                if (options.rule !== 'running') previousCounts.fill(1);
                for (let thread = 0; thread < PROBE_CELLS; thread++) {
                    const guide = random.rand();
                    const ownHasHistory = previous[thread] !== null;
                    const cell =
                        options.guiding === 'fixed'
                            ? guidedCell(thread, ownHasHistory, cdf, guide)
                            : ownHasHistory
                              ? findCellIndex(cdf.total > 0 ? cdf.cdf : new Array<number>(PROBE_CELLS).fill(0), guide)
                              : thread;
                    const u = random.rand();
                    sums[cell] += noise === 'exponential' ? -Math.log(1 - u) : noise === 'sparse' ? (u < 0.2 ? 5 : 0) : 1;
                    counts[cell]++;
                }
                unreachedCells += counts.filter((count) => count === 0).length;
                spawnedCells += PROBE_CELLS;
                if (options.rule === 'running') {
                    const expected = options.guidingCorrection ? expectedRaysPerCell(previous.map((value) => value !== null), cdf, options.guiding === 'fixed' ? RAY_GUIDING_UNIFORM_SHARE : 0) : undefined;
                    const blended = blendRunningMean(sums, counts, previous, previousCounts, HISTORY_SAMPLE_CAP, expected);
                    next[ty * tiles + tx] = blended.values;
                    nextCounts[ty * tiles + tx] = blended.counts;
                } else {
                    next[ty * tiles + tx] = blendProbeCells(sums, counts, previous, options.rule);
                    nextCounts[ty * tiles + tx] = new Array<number>(PROBE_CELLS).fill(1);
                }
            }
        }
        probes = next;
        probeCounts = nextCounts;
        if (report.has(frame + 1)) {
            const live = probes.filter((probe): probe is number[] => probe !== null);
            const mean = (weight: number, subsamples: number): number =>
                live.reduce((sum, probe) => sum + shIrradiance(projectProbe((cell) => probe[cell], up, weight, subsamples), up), 0) / live.length / Math.PI;
            results.push({ frame: frame + 1, capsaicinProjection: mean(1 / PROBE_SIZE, 1), ourProjection: mean((2 * Math.PI) / PROBE_CELLS, 4), unreachedShare: unreachedCells / spawnedCells });
            unreachedCells = 0;
            spawnedCells = 0;
        }
    }
    return results;
}
