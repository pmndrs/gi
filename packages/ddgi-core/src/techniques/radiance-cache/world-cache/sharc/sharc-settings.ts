/**
 * Settings of the SHaRC-style hash cache (`SharcCache`, phase 6 experiment
 * P6.1), the radiance cache's `worldCache: 'sharc'` backend. The cell and
 * table defaults are the pmndrs/pathtracer project's tuned ones
 * (`bench/nrc/hash-grid.ts`, `DEFAULT_HASH_GRID`, at dd771c8: its M7 sweeps on
 * sponza-white, nrc-hash-cache.md §6); the update-path settings are this
 * slot's own (register E2, E3).
 */

export interface SharcSettings {
    /** Table size: 2^log2Entries slots (20: 1M slots, 40 MiB). Read at `prepare`. */
    log2Entries: number;
    /** The smallest cell, in scene units (metres). */
    minCell: number;
    /** A cell's size target in pixels at its distance from the camera (SHaRC's LOD rule). */
    cellPixels: number;
    /** Separate cells per dominant normal axis and sign (6 bins), or one cell for all normals. */
    normalBins: boolean;
    /** Time constant of a cell's running mean, in frames (its history ages only in frames with new samples). */
    historyFrames: number;
    /** A cell with no new samples for this many frames is evicted. */
    staleFrames: number;
    /** Coarser levels a lookup tries when a cell has too few samples. Read at `prepare`. */
    fallbackLevels: number;
    /** Each sample also adds into its `fallbackLevels` coarser cells (a multi-resolution grid). Read at `prepare`. */
    pyramid: boolean;
    /** A level with less sample weight than this is passed over by the lookup. */
    minSamples: number;
    /**
     * Bounce rays per update path, 0 to 2 (register E2). 0: a probe ray's
     * hit gets its direct light only (GI-1.0's single bounce). 1: plus one
     * cosine-weighted bounce whose hit is lit by the cache (or, without
     * `cacheTail`, by its own light sample: GI-1.2's two bounces). 2: the
     * first bounce's hit is a vertex too (lit, and inserted into its own
     * cell), so surfaces no probe ray reaches fill as well.
     */
    bounceRays: number;
    /**
     * The last bounce's hit reads the cache (the cache as the path's tail:
     * every bounce, converging over frames), where it has `minSamples` and the
     * segment is longer than the cell; otherwise its own light sample.
     * Off: the path stops at the tail hit's direct light.
     */
    cacheTail: boolean;
    /**
     * A bounce ray that misses brings the sky's radiance (giSkyRadiance). The
     * platform's light choice has no sky light (register D33), so this is how
     * the sky reaches cached surfaces. Turn it off when the sky becomes a
     * light in `giDirectChooseLight` (RC3.4), or both would count it.
     */
    skyOnMiss: boolean;
    /**
     * The share of probes whose rays run the update path this frame (SHaRC
     * updates from a few per cent of its paths). Chosen per probe (a
     * workgroup), at random each frame, so a probe's 64 rays update together
     * and the skipped ones cost no tracing. The other probes' rays only read
     * the cache: a ray shorter than its cell reads its cell too, and a cell
     * with no samples yet reads black (register E7).
     */
    updateFraction: number;
}

export const DEFAULT_SHARC_SETTINGS: Readonly<SharcSettings> = Object.freeze({
    log2Entries: 20,
    minCell: 0.02,
    // The pathtracer's M7 sweeps: 16 px cells, the pyramid and a 64-sample threshold removed the
    // single-sample blotches after camera changes at no bias; 3 fallback levels cut misses after a
    // cut to 2% (nrc-hash-cache.md §6).
    cellPixels: 16,
    normalBins: true,
    historyFrames: 32,
    staleFrames: 64,
    fallbackLevels: 3,
    pyramid: true,
    minSamples: 64,
    bounceRays: 2,
    cacheTail: true,
    skyOnMiss: true,
    // P6.1 timing (RTX 3060, Sponza 1280 x 720): every probe updating costs 92 ms with two bounce
    // rays, eight times GI-1.2's cache; a quarter keeps the accuracy checks and costs about a quarter.
    updateFraction: 0.25,
});

/** `settings` over the defaults, with the ranges checked (a RangeError names the setting). */
export function resolveSharcSettings(settings: Partial<SharcSettings> = {}): SharcSettings {
    const resolved: SharcSettings = { ...DEFAULT_SHARC_SETTINGS, ...settings };
    const check = (name: keyof SharcSettings, ok: boolean, rule: string): void => {
        if (!ok) throw new RangeError(`@pmndrs/gi: sharc.${name} must be ${rule}, got ${String(resolved[name])}`);
    };
    check('log2Entries', Number.isInteger(resolved.log2Entries) && resolved.log2Entries >= 10 && resolved.log2Entries <= 24, 'a whole number from 10 to 24');
    check('minCell', resolved.minCell > 0, 'above 0');
    check('cellPixels', resolved.cellPixels > 0, 'above 0');
    check('historyFrames', resolved.historyFrames >= 1, 'at least 1');
    check('staleFrames', Number.isInteger(resolved.staleFrames) && resolved.staleFrames >= 1, 'a whole number of at least 1');
    check('fallbackLevels', Number.isInteger(resolved.fallbackLevels) && resolved.fallbackLevels >= 0 && resolved.fallbackLevels <= 8, 'a whole number from 0 to 8');
    check('minSamples', resolved.minSamples >= 0, 'at least 0');
    check('updateFraction', resolved.updateFraction > 0 && resolved.updateFraction <= 1, 'above 0 and at most 1');
    check('bounceRays', Number.isInteger(resolved.bounceRays) && resolved.bounceRays >= 0 && resolved.bounceRays <= 2, '0, 1 or 2');
    return resolved;
}
