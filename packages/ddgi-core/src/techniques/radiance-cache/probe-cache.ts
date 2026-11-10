/**
 * The screen probes' history and off-screen probe cache (RC4.1): where
 * Capsaicin's buffers live in ours, and CPU twins of the cache's bookkeeping
 * for the tests.
 *
 * **The packed words** (register D6). Capsaicin keeps the cache in a dozen
 * structured buffers (`GI1::ScreenProbes`, gi1.h:205-231, made in
 * gi1.cpp:182-335). A WebGPU stage binds at most 8 storage buffers, and
 * `SampleScreenProbes` reads seven of them, so the ones no platform scan
 * touches share one `u32` buffer at fixed word offsets, given to the kernels
 * in `RcFrame`:
 *
 * | Array | Capsaicin's buffer | Words |
 * | --- | --- | --- |
 * | LRU, this frame's half | `ProbeCachedTileLRUBuffer` (`probe_cached_tile_lru_buffers_[index]`) | N |
 * | LRU, the next frame's half | `PreviousProbeCachedTileLRUBuffer` (`[1 - index]`) | N |
 * | MRU | `ProbeCachedTileMRUBuffer` | N |
 * | list | `ProbeCachedTileListBuffer` | N |
 * | list elements | `ProbeCachedTileListElementBuffer` (`uint4` each) | 4 N |
 * | empty tiles | `EmptyTileBuffer` | N |
 * | override tiles | `OverrideTileBuffer` | S |
 * | spawn probe | `ProbeSpawnProbeBuffer` (`uint2` each) | 2 S |
 *
 * N is the probe tiles (`max_probe_count`), S the spawn tiles
 * (`max_probe_spawn_count`). The four arrays the platform's scan reads or
 * writes (the LRU flags and their scan, the list counts and theirs) are
 * buffers of their own, since a scan takes whole buffers.
 */

import { INVALID_ID } from './capsaicin-math.ts';
import { PROBE_SIZE, type ScreenProbeLayout } from './settings.ts';

const WORD_BYTES = 4;

/** High bit of an LRU entry: the slot holds no cached probe yet (gi1.comp:289). */
export const CACHED_TILE_UNALLOCATED = 0x80000000;

/** Word offsets of the probe cache's arrays in the packed buffer. */
export interface ProbeCacheLayout {
    /** Probe tiles: the cache's slots (`max_probe_count`). */
    readonly slots: number;
    /** Spawn tiles (`max_probe_spawn_count`). */
    readonly spawnTiles: number;
    /** The two LRU halves: `lru[half]`. */
    readonly lru: readonly [number, number];
    readonly mru: number;
    readonly list: number;
    readonly listElement: number;
    readonly emptyTile: number;
    readonly overrideTile: number;
    readonly spawnProbe: number;
    /** Words in the packed buffer. */
    readonly words: number;
    readonly bytes: number;
}

export function probeCacheLayout(layout: Pick<ScreenProbeLayout, 'probeCount' | 'maxProbeSpawnCount'>): ProbeCacheLayout {
    const slots = layout.probeCount[0] * layout.probeCount[1];
    const spawnTiles = layout.maxProbeSpawnCount;
    let next = 0;
    const take = (words: number): number => {
        const start = next;
        next += words;
        return start;
    };
    const lru: [number, number] = [take(slots), take(slots)];
    const mru = take(slots);
    const list = take(slots);
    const listElement = take(4 * slots);
    const emptyTile = take(slots);
    const overrideTile = take(spawnTiles);
    const spawnProbe = take(2 * spawnTiles);
    return { slots, spawnTiles, lru, mru, list, listElement, emptyTile, overrideTile, spawnProbe, words: next, bytes: next * WORD_BYTES };
}

/** The probe tiles of the largest screen the device can draw: what the scans' buffers are made for, in `prepare`. */
export function largestProbeTileCount(maxTextureDimension2D: number): number {
    return Math.ceil(maxTextureDimension2D / PROBE_SIZE) ** 2;
}

/** `InitCachedTileLRU` (gi1.comp:286-290) on the CPU. */
export function initialCachedTileLRU(slots: number): Uint32Array {
    return Uint32Array.from({ length: slots }, (_, slot) => (slot | CACHED_TILE_UNALLOCATED) >>> 0);
}

/** What `CountScreenProbes` and `ScatterScreenProbes` make: the cached slots on screen, by probe tile. */
export interface CachedTileLists {
    /** `ProbeCachedTileListCountBuffer`: slots listed in each probe tile. */
    readonly counts: Uint32Array;
    /** `ProbeCachedTileListIndexBuffer`: the exclusive scan of `counts`. */
    readonly starts: Uint32Array;
    /** Each probe tile's slots, sorted (the GPU's order within a tile follows its atomics, so tests compare sets). */
    readonly slotsByTile: ReadonlyMap<number, readonly number[]>;
    /** `ProbeCachedTileListElementCountBuffer`. */
    readonly elementCount: number;
    /** Slots whose projection lies within `edge` of the screen's, the depth range's or a tile's bounds: f32 and f64 may disagree on them. */
    readonly undecided: readonly number[];
}

/**
 * `CountScreenProbes` (gi1.comp:501-539) then the scan and
 * `ScatterScreenProbes` (gi1.comp:541-553), in f64. `project` is
 * `transformPointProjection(world_pos, g_ViewProjection)`: clip space over w.
 */
export function listCachedTiles(
    lru: Uint32Array,
    cachedPosition: (cachedTile: number) => readonly [number, number, number],
    project: (world: readonly [number, number, number]) => readonly [number, number, number],
    probeCount: readonly [number, number],
    edge = 1e-5,
): CachedTileLists {
    const slots = probeCount[0] * probeCount[1];
    const counts = new Uint32Array(slots);
    const byTile = new Map<number, number[]>();
    const undecided: number[] = [];
    let elementCount = 0;
    for (let slot = 0; slot < slots; slot++) {
        const entry = lru[slot];
        if ((entry & CACHED_TILE_UNALLOCATED) !== 0) continue;
        const [x, y, z] = project(cachedPosition(entry));
        const homogeneous = [0.5 * x + 0.5, -0.5 * y + 0.5, z];
        // Near the screen's or the depth range's bounds, or a tile's edge, f32 and f64 may disagree.
        const nearTileEdge = [0, 1].some((axis) => {
            const scaled = homogeneous[axis] * probeCount[axis];
            return Math.abs(scaled - Math.round(scaled)) < edge * probeCount[axis];
        });
        if (nearTileEdge || homogeneous.some((value) => Math.abs(value) < edge || Math.abs(value - 1) < edge)) undecided.push(slot);
        if (!homogeneous.every((value) => value > 0 && value < 1)) continue;
        const tile = Math.floor(homogeneous[0] * probeCount[0]) + Math.floor(homogeneous[1] * probeCount[1]) * probeCount[0];
        counts[tile]++;
        elementCount++;
        const list = byTile.get(tile) ?? [];
        list.push(slot);
        byTile.set(tile, list);
    }
    const starts = exclusiveScan(counts);
    for (const list of byTile.values()) list.sort((a, b) => a - b);
    return { counts, starts, slotsByTile: byTile, elementCount, undecided };
}

/** An exclusive prefix sum, wrapping at 2^32 as the GPU's scan does. */
export function exclusiveScan(values: ArrayLike<number>): Uint32Array {
    const out = new Uint32Array(values.length);
    let sum = 0;
    for (let index = 0; index < values.length; index++) {
        out[index] = sum;
        sum = (sum + values[index]) >>> 0;
    }
    return out;
}

/**
 * The scan of the LRU flags and `ReorderScreenProbes` (gi1.comp:1333-1354):
 * the next frame's LRU. Slots with flag 1 (untouched) keep their order at the
 * front; slots with flag 0 (used this frame) follow, ordered by their MRU index.
 */
export function reorderCachedTiles(lru: Uint32Array, flags: Uint32Array, mru: Uint32Array, slots: number): Uint32Array {
    // The GPU scans slots + 1 flags (gi1.cpp:2553-2554); the last is never written, so 0.
    const scan = exclusiveScan(Array.from({ length: slots + 1 }, (_, index) => (index < slots ? flags[index] : 0)));
    const untouched = scan[slots];
    const next = new Uint32Array(slots).fill(INVALID_ID);
    for (let slot = 0; slot < slots; slot++) {
        const offset = flags[slot] === 0 ? mru[slot] + untouched : scan[slot];
        next[offset] = lru[slot];
    }
    return next;
}

/**
 * `ScreenProbes_ScanRadiance` (screen_probes.hlsl:309-351): each cell's start
 * in the probe's sampling CDF, the exclusive sum of the weights before it over
 * the total (at least 1e-5).
 */
export function screenProbeCdf(weights: readonly number[]): number[] {
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    const starts: number[] = [];
    let sum = 0;
    for (const weight of weights) {
        starts.push(sum / Math.max(total, 1e-5));
        sum += weight;
    }
    return starts;
}

/** `ScreenProbes_FindCellIndex` (screen_probes.hlsl:283-306): the last cell whose start is at most `s`. */
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

/**
 * Each cell's chance of being drawn from the CDF: its weight over the total,
 * or, when every weight is 0, all on the last cell (FindCellIndex's answer then).
 */
export function cdfCellProbabilities(weights: readonly number[]): number[] {
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    if (!(total > 0)) return weights.map((_, cell) => (cell === weights.length - 1 ? 1 : 0));
    return weights.map((weight) => weight / total);
}

/** Rec. 709 luminance (color.hlsl:34-37). */
export function luminance(rgb: readonly [number, number, number]): number {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

/**
 * `BlendScreenProbes`' shadow-preserving biased hysteresis (gi1.comp:1262-1268):
 * the history's share in the blend for new luminance `a` against history `b`.
 */
export function temporalBlend(a: number, b: number): number {
    const share = Math.min(Math.max(Math.max(a - b - Math.min(a, b), 0) / Math.max(Math.max(a, b), 1e-4), 0), 0.95);
    return share * share;
}
