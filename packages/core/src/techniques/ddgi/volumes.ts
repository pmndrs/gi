/**
 * Several DDGI volumes in one technique (docs/design/m3-moving.md §4.1):
 * where each volume's probes live in the shared atlases, the volume table
 * renderers read, the order volumes blend in, and how a frame's probe
 * updates are shared between them. Plain numbers, unit-tested without a GPU.
 *
 * - **One atlas set for all volumes.** The irradiance, distance and
 *   probe-data textures stay three 2D arrays, and volume v owns the array
 *   layers `[layerBase_v, layerBase_v + ny_v)`, probe (x, z) owning tile
 *   (x, z) in each. Every layer is as large as the largest footprint (the
 *   largest nx and nz over the volumes). *Why shared:* a shading module may
 *   bind at most 6 textures (docs/api/platform.md, gap 1); a set per volume
 *   would break that at the second volume.
 * - **The volume table** is a small `rgba32float` texture, one row per
 *   volume in priority order: the volume's four sampling `vec4f`s (origin
 *   and normal bias, spacing and view bias, counts and layer base, texel
 *   sizes and the atlas's tiles), exactly what the update kernels get as
 *   their first four uniforms; then where its window is on the lattice, and
 *   its valid extent (the camera-following volume, scroll.ts; a fixed
 *   volume's window is 0 and its extent every probe). Renderers and the
 *   update's bounce light read every volume through it
 *   (`ddgiSampleVolumes`, wgsl/sample.ts).
 * - **Priority:** higher first; volumes of equal priority keep the order the
 *   settings list them in. A dense volume inside a coarse one sets a higher
 *   priority, so inside it only it is read, and in its fade band the coarse
 *   one fills in what it leaves.
 * - **Each volume is updated by the same kernels** with its own uniforms,
 *   update list, history and scheduler: the kernels see one volume at a time
 *   (`ddgiUniformVolume()`), so nothing in them changes but where a probe's
 *   tiles are. A frame's probe updates are shared in proportion to the
 *   volumes' probe counts (`UnitSharer`), so every volume is swept at the
 *   same rate.
 */

import { GILimitError } from '../../platform/contract/errors.ts';
import { minProbesPerFrame, probeGrid, type DDGIVolumeSettings, type ProbeGrid } from './settings.ts';

/** At most this many volumes in one technique (m3-moving.md §4.1). */
export const MAX_VOLUMES = 8;

/**
 * Timed passes one volume adds to a frame with one `gi/direct` round (up to
 * 1,024 probes): scroll (a camera-following volume only), mark, schedule,
 * trace, `gi/direct`, blend, write, propagate, compose, census. The
 * platform's GPU timer must have room for MAX_VOLUMES of these
 * (gpu-timer.ts, DEFAULT_MAX_PASSES_PER_FRAME; volumes.test.ts).
 */
export const TIMED_PASSES_PER_VOLUME = 10;

/**
 * Coverage: how far past the outermost probes the field still counts as
 * fully covering a point, and over how much farther it then fades to 0, both
 * in probe spacings.
 *
 * Each probe answers for the half cell around it, so the volume the grid
 * covers is the probe box grown by half a spacing; the settings box always
 * lies inside it (probe counts are `floor(size / spacing) + 1`, centred).
 * Past that, sampling extrapolates the outermost cell, so coverage fades out
 * over the next half cell. A room whose walls sit just outside the outermost
 * probes (Cornell: 3 cm) keeps full coverage on its walls. With several
 * volumes this fade is also the band in which a higher-priority volume hands
 * over to the next one.
 */
export const COVERAGE_FULL_MARGIN_SPACINGS = 0.5;
export const COVERAGE_FADE_SPACINGS = 0.5;

/**
 * Coverage: the summed probe weight (trilinear × direction × visibility,
 * after the crush) at which a point counts as fully seen.
 *
 * Why this value: a probe that sees the point keeps at least the direction
 * floor (0.2) times its trilinear share, while an occluded one keeps at most
 * 1.2 × 0.05 = 0.06 before the crush and about 0.005 after it. So a sum of
 * 0.02 needs a visible probe with at least a 10% trilinear share, and 8
 * occluded probes alone stay below it.
 */
export const COVERAGE_FULL_WEIGHT = 0.02;

/**
 * The volume blend stops once the weight gathered over the volumes read so
 * far reaches this (m3-moving.md §4.1): what the rest could add would change
 * the answer by under 1%.
 */
export const BLEND_COMPLETE_WEIGHT = 0.99;

/** ...or after this many volumes, so no point costs more than two volumes' sampling (m3-moving.md §4.1). */
export const MAX_BLENDED_VOLUMES = 2;

/**
 * The texels of one volume-table row (`ddgiTableVolume` in wgsl/sample.ts):
 * - `sampling` (4 texels): the four sampling `vec4f`s, the update kernels'
 *   first four uniforms. For a camera-following volume the origin is its
 *   window's first probe, so everything that works in window probes is as
 *   for a fixed volume;
 * - `window`: xyz the lattice coordinates of the window's first probe
 *   (whole numbers; 0 for a fixed volume), w unused. Slots and key probes
 *   follow from it;
 * - `fade`: xyz how wide the fade band is on each axis, in spacings
 *   (COVERAGE_FADE_SPACINGS for a fixed volume, scroll.ts's
 *   `cameraFadeSpacings` for a camera-following one); w 1 when the sampling
 *   weights have RTXGI's floors (`samplingWeightFloors`, DV-06), else 0;
 * - `validMin`, `validMax`: xyz the valid extent's low and high sides in
 *   lattice units, where coverage fades (scroll.ts), w unused. For a fixed
 *   volume its first and last probes. **Written by the GPU** for a
 *   camera-following volume (`ddgi/scroll` copies them in every frame), so
 *   the CPU writes them only when the table is new.
 */
export const VOLUME_TABLE_TEXEL = { sampling: 0, window: 4, fade: 5, validMin: 6, validMax: 7 } as const;
/** Texels in one row of the volume table. */
export const VOLUME_TABLE_TEXELS = 8;
/** The texels at the start of a row that the CPU keeps up to date every frame: the sampling `vec4f`s, the window and the fade. The rest are the valid extent's. */
export const VOLUME_TABLE_CPU_TEXELS = VOLUME_TABLE_TEXEL.validMin;
export const VOLUME_TABLE_FORMAT: GPUTextureFormat = 'rgba32float';
export const VOLUME_TABLE_TEXEL_BYTES = 16;
/** Floats in one row of the volume table. */
export const VOLUME_TABLE_ROW_FLOATS = VOLUME_TABLE_TEXELS * 4;
/** Floats of the sampling `vec4f`s at the start of a row, the same as the update kernels' first four uniforms. */
export const VOLUME_SAMPLING_FLOATS = 16;

/** One volume of a technique, placed in the shared atlases. */
export interface PlacedVolume {
    /** Its place in the settings' list. */
    readonly settingsIndex: number;
    /** Its probes as the settings place them; for a camera-following volume, its window's first place, whose first probe is lattice 0 (scroll.ts). */
    readonly grid: ProbeGrid;
    readonly priority: number;
    /** It follows the camera (`follow: 'camera'`, scroll.ts). */
    readonly followsCamera: boolean;
    /** The atlas layer of its probe row y = 0. */
    readonly layerBase: number;
    /**
     * Its first probe in the technique-wide numbering: the `points` debug
     * view lists every volume's probes in table order, this volume's from
     * here on.
     */
    readonly probeBase: number;
}

/** Every volume of a technique, in priority order (its volume-table rows), and the atlas they share. */
export interface VolumePlan {
    /** Row i of the volume table is `volumes[i]`. */
    readonly volumes: readonly PlacedVolume[];
    /**
     * The shared atlas in probe tiles, as `ProbeGrid.counts` orders axes:
     * [tiles across a layer (the largest nx), layers (the sum of ny), tiles
     * down a layer (the largest nz)]. `fieldLayout` (layout.ts) sizes the
     * textures from it.
     */
    readonly atlasCounts: readonly [number, number, number];
    /** Probes over every volume. */
    readonly probeCount: number;
}

/** The volumes a technique's settings describe: `volumes` when given, else the one `volume`. */
export function volumeSettingsList(settings: { readonly volume: DDGIVolumeSettings; readonly volumes?: readonly DDGIVolumeSettings[] }): readonly DDGIVolumeSettings[] {
    return settings.volumes && settings.volumes.length > 0 ? settings.volumes : [settings.volume];
}

/**
 * Orders the volumes by priority (higher first; equal priorities keep the
 * settings' order), gives each its layers in the shared atlas and its first
 * probe, and sizes the atlas. Throws for more than {@link MAX_VOLUMES}
 * volumes, and `GILimitError` past the device's array layers.
 */
export function planVolumes(settings: readonly DDGIVolumeSettings[], maxTextureArrayLayers: number): VolumePlan {
    if (settings.length === 0) throw new Error('@pmndrs/gi: a DDGI technique needs at least one volume');
    if (settings.length > MAX_VOLUMES) throw new Error(`@pmndrs/gi: a DDGI technique takes at most ${MAX_VOLUMES} volumes, got ${settings.length}`);
    const ordered = settings
        .map((volume, settingsIndex) => ({ settingsIndex, grid: probeGrid(volume), priority: volumePriority(volume), followsCamera: volume.follow === 'camera' }))
        .sort((a, b) => b.priority - a.priority || a.settingsIndex - b.settingsIndex);
    let layerBase = 0;
    let probeBase = 0;
    const volumes = ordered.map((volume): PlacedVolume => {
        const placed = { ...volume, layerBase, probeBase };
        layerBase += volume.grid.counts[1];
        probeBase += volume.grid.probeCount;
        return placed;
    });
    // Each volume's probe rows are layers of their own, so the layers add up (m3-moving.md §4.1: past 256, prepare throws).
    if (layerBase > maxTextureArrayLayers) throw new GILimitError('maxTextureArrayLayers', layerBase, maxTextureArrayLayers);
    const tilesX = Math.max(...volumes.map((volume) => volume.grid.counts[0]));
    const tilesZ = Math.max(...volumes.map((volume) => volume.grid.counts[2]));
    return { volumes, atlasCounts: [tilesX, layerBase, tilesZ], probeCount: probeBase };
}

/** A valid extent in lattice units (scroll.ts): its low and high sides per axis. */
export interface ValidExtent {
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
}

/** A fixed volume's fade on every axis, in spacings. */
export const FIXED_VOLUME_FADE: readonly [number, number, number] = [COVERAGE_FADE_SPACINGS, COVERAGE_FADE_SPACINGS, COVERAGE_FADE_SPACINGS];

/**
 * A volume's edge coverage at a point (the CPU twin of
 * `ddgiVolumeEdgeCoverage`, wgsl/sample.ts): 1 out to
 * {@link COVERAGE_FULL_MARGIN_SPACINGS} past its valid extent, falling to 0
 * over the next `fade` spacings on each axis (half a spacing for a fixed
 * volume), the lowest over the axes. Between 0 and 1 is the volume's fade
 * band, where the next volume in priority order fills in. The valid extent
 * is in lattice units from `grid.origin` (lattice 0); without one, it is the
 * grid's own probes, as for a fixed volume.
 */
export function volumeEdgeCoverage(
    grid: ProbeGrid,
    position: readonly [number, number, number],
    extent: ValidExtent = wholeGridExtent(grid),
    fade: readonly [number, number, number] = FIXED_VOLUME_FADE,
): number {
    let coverage = 1;
    for (let axis = 0; axis < 3; axis++) {
        const first = grid.origin[axis] + extent.min[axis] * grid.spacing[axis];
        const last = grid.origin[axis] + extent.max[axis] * grid.spacing[axis];
        const outside = Math.max(first - position[axis], position[axis] - last, 0) / grid.spacing[axis];
        coverage = Math.min(coverage, Math.min(1, Math.max(0, 1 - (outside - COVERAGE_FULL_MARGIN_SPACINGS) / fade[axis])));
    }
    return coverage;
}

/** A volume's priority: its setting, or 0. Not a number counts as 0. */
export function volumePriority(volume: DDGIVolumeSettings): number {
    return Number.isFinite(volume.priority) ? (volume.priority as number) : 0;
}

/**
 * One volume's four sampling `vec4f`s (`DdgiVolume` in wgsl/sample.ts), into
 * `target` from `offset`: origin and normal bias, spacing and view bias,
 * counts and layer base, texel sizes and the atlas's tiles across and down a
 * layer. The update kernels' first four uniforms and the start of a
 * volume-table row. `origin` is the world position of the first probe (of
 * the window's first, for a camera-following volume); the grid's own by
 * default.
 */
export function writeVolumeParameters(
    target: Float32Array,
    offset: number,
    volume: PlacedVolume,
    plan: VolumePlan,
    texels: { readonly irradiance: number; readonly distance: number },
    bias: { readonly normal: number; readonly view: number },
    origin: readonly [number, number, number] = volume.grid.origin,
): void {
    const { grid, layerBase } = volume;
    target.set([...origin, bias.normal], offset);
    target.set([...grid.spacing, bias.view], offset + 4);
    target.set([...grid.counts, layerBase], offset + 8);
    target.set([texels.irradiance, texels.distance, plan.atlasCounts[0], plan.atlasCounts[2]], offset + 12);
}

/**
 * The rest of a volume-table row after the sampling `vec4f`s, into `target`
 * from `offset` (the row's start): the window's first probe in lattice units,
 * the fade's width per axis, whether the sampling weights have RTXGI's
 * floors, and the valid extent (VOLUME_TABLE_TEXEL). A fixed volume: window
 * 0, fade FIXED_VOLUME_FADE, the extent its first and last probes.
 */
export function writeVolumeWindow(
    target: Float32Array,
    offset: number,
    window: readonly [number, number, number],
    fade: readonly [number, number, number],
    extent: ValidExtent,
    samplingWeightFloors: boolean,
): void {
    target.set([...window, 0], offset + VOLUME_TABLE_TEXEL.window * 4);
    target.set([...fade, samplingWeightFloors ? 1 : 0], offset + VOLUME_TABLE_TEXEL.fade * 4);
    target.set([...extent.min, 0], offset + VOLUME_TABLE_TEXEL.validMin * 4);
    target.set([...extent.max, 0], offset + VOLUME_TABLE_TEXEL.validMax * 4);
}

/** A fixed volume's valid extent: its first and last probes, in lattice units. */
export function wholeGridExtent(grid: ProbeGrid): ValidExtent {
    return { min: [0, 0, 0], max: [grid.counts[0] - 1, grid.counts[1] - 1, grid.counts[2] - 1] };
}

/**
 * Shares a frame's probe updates between the volumes in proportion to their
 * probe counts, so each is swept at the same rate. A volume's exact share is
 * rarely whole (Sponza-sized coarse volume of 20,000 probes and a dense one
 * of 125 at 154 probes a frame: 0.96 a frame for the small one), so each
 * volume carries the fraction it didn't get to the next frame; over frames
 * every volume gets exactly its share, however small. Each frame the whole
 * probes handed out add up to the hand-out (rounded down), the leftovers
 * going to the volumes owed the most (largest remainder).
 *
 * One volume gets the hand-out exactly as it came (fractions and Infinity
 * included), as the technique always took it. Infinity gives every volume
 * as many as it can take.
 */
export class UnitSharer {
    private readonly probeCounts: readonly number[];
    private readonly totalProbes: number;
    /** What each volume was owed but didn't get, in probes; carried to the next frame (each within ±1). */
    private readonly carried: number[];

    constructor(probeCounts: readonly number[]) {
        this.probeCounts = probeCounts;
        this.totalProbes = probeCounts.reduce((sum, count) => sum + count, 0);
        this.carried = probeCounts.map(() => 0);
    }

    share(units: number): number[] {
        const counts = this.probeCounts;
        if (counts.length === 1) return [units];
        if (units === Number.POSITIVE_INFINITY) return counts.map(() => Number.POSITIVE_INFINITY);
        if (!(units > 0) || !Number.isFinite(units) || this.totalProbes <= 0) return counts.map(() => 0);
        // What each volume is owed this frame: its exact share plus what it was owed before.
        const owed = counts.map((count, index) => (units * count) / this.totalProbes + this.carried[index]);
        const given = owed.map((amount) => Math.max(0, Math.floor(amount)));
        // The probes left over, one each to the volumes owed the largest fraction (ties: list order).
        let left = Math.floor(units) - given.reduce((sum, share) => sum + share, 0);
        const byRemainder = owed.map((amount, index) => ({ remainder: amount - given[index], index })).sort((a, b) => b.remainder - a.remainder || a.index - b.index);
        for (let next = 0; left > 0; next = (next + 1) % byRemainder.length, left--) given[byRemainder[next].index]++;
        // Rounding the owed amounts can give out one more than the hand-out; take it back from the least owed.
        for (let next = byRemainder.length - 1; left < 0; next = (next - 1 + byRemainder.length) % byRemainder.length) {
            const index = byRemainder[next].index;
            if (given[index] > 0) {
                given[index]--;
                left++;
            }
        }
        // Clamped, so a volume that can't take its share (none here; shares never exceed it) can't bank a debt for ever.
        owed.forEach((amount, index) => (this.carried[index] = Math.min(1, Math.max(-1, amount - given[index]))));
        return given;
    }
}

/**
 * Budget v2's floor for one volume of several (settings.ts,
 * `minProbesPerFrame`): every volume gets its own, so each is swept within
 * FLOOR_SWEEP_SECONDS whatever its size, and never 0 probes a frame unless
 * the floor is switched off (`minProbesPerFrame: 0`). An explicit number is
 * shared by probe count, rounded up. The device limit (twice the calibrated
 * batch) is the technique's, as with one volume.
 */
export function volumeFloorUnits(
    setting: number | 'auto',
    volume: { readonly probeCount: number; readonly maxProbesPerFrame: number },
    totalProbes: number,
    rate: { readonly framesPerSecond: number; readonly calibratedUnits: number },
): number {
    if (setting !== 'auto' && !(Number.isFinite(setting) && setting > 0)) return 0;
    const volumeSetting = setting === 'auto' ? 'auto' : Math.ceil((setting * volume.probeCount) / Math.max(1, totalProbes));
    const floor = minProbesPerFrame(volumeSetting, volume.probeCount, volume.maxProbesPerFrame, rate);
    return Math.min(volume.maxProbesPerFrame, Math.max(1, floor));
}
