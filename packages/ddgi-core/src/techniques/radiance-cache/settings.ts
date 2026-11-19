/**
 * The radiance cache's settings: Capsaicin v1.3's defaults for GI-1
 * (`GI1::RenderOptions`, `src/core/src/render_techniques/gi1/gi1.h:48-89`),
 * named after its options, plus the screen probes' compile-time constants
 * (`GI1::ScreenProbes`, gi1.h:191-198), which Capsaicin can't change at run
 * time either.
 *
 * Phase 2 (RC2.1) reads `useDirectLighting` only; the rest are listed now so
 * the defaults are in one place, with the phase that reads each. Glossy
 * reflections' options are left out until the platform has roughness and
 * metalness (register D8); `gi1_disable_specular_materials` and
 * `gi1_disable_albedo_textures` are left out because the platform has neither
 * specular materials (D8, D16) nor textures at hits (D2); the debug and DXR 1.0
 * options are left out for good (docs/design/radiance-cache.md, "The baseline is Capsaicin's default
 * configuration").
 */

/** One probe is `PROBE_SIZE` × `PROBE_SIZE` cells and covers a tile of that many pixels (`probe_size_`, gi1.h:191). */
export const PROBE_SIZE = 8;

/** Rays per probe: one per cell. */
export const PROBE_CELLS = PROBE_SIZE * PROBE_SIZE;

/**
 * Capsaicin's spawn rates (`ScreenProbes::SamplingMode`, gi1.h:177-185). Its
 * default and only compiled mode is `QuarterSpp` (`sampling_mode_`,
 * gi1.h:192): one probe per 16 × 16 pixel tile per frame, a quarter of a ray
 * per pixel.
 */
export const SAMPLING_MODE = 'QuarterSpp';

/** Pixels across one spawn tile: `probe_size_ << 1` in `QuarterSpp` (`probe_spawn_tile_size_`, gi1.h:195-198). */
export const PROBE_SPAWN_TILE_SIZE = PROBE_SIZE << 1;

/** GI-1's options, as Capsaicin names them without the `gi1_` prefix. */
export interface RadianceCacheSettings {
    /**
     * `gi1_use_direct_lighting` (gi1.h:53, default true). Probe rays that miss
     * read the sky, and rays that hit an emissive front face take its emission
     * and stop there (`PopulateScreenProbesHandleMiss`, gi1.comp:1078-1084;
     * `GenerateReservoirs`, gi1.comp:2439-2452). Read since phase 2.
     */
    useDirectLighting: boolean;
    /** `gi1_use_temporal_feedback` (gi1.h:54, default false): last frame's lit image fed back at hits. Phase 3. */
    useTemporalFeedback: boolean;
    /** `gi1_use_multibounce` (gi1.h:55, default true): GI-1.2's second-bounce cells. Phase 3. */
    useMultibounce: boolean;
    /** `gi1_use_resampling` (gi1.h:51, default false): world-space ReSTIR of the light samples. Off in the baseline. */
    useResampling: boolean;
    /** `gi1_disable_alpha_testing` (gi1.h:52, default false). The platform's tracer honours cutouts (register D1). */
    disableAlphaTesting: boolean;
    /** `gi1_hash_grid_cache_cell_size` (gi1.h:58, default 32): a world-cache cell's size in projected pixels. Phase 3. */
    hashGridCacheCellSize: number;
    /** `gi1_hash_grid_cache_min_cell_size` (gi1.h:59, default 0.1): the smallest cell, in scene units. Phase 3. */
    hashGridCacheMinCellSize: number;
    /** `gi1_hash_grid_cache_tile_cell_ratio` (gi1.h:60, default 8): cells across a tile (8 × 8). Phase 3. */
    hashGridCacheTileCellRatio: number;
    /** `gi1_hash_grid_cache_num_buckets` (gi1.h:61, default 14): log2 of the bucket count (16,384). Phase 3. */
    hashGridCacheNumBuckets: number;
    /** `gi1_hash_grid_cache_num_tiles_per_bucket` (gi1.h:62, default 4): log2 of the tiles per bucket (16). Phase 3. */
    hashGridCacheNumTilesPerBucket: number;
    /** `gi1_hash_grid_cache_max_sample_count` (gi1.h:63, default 16): a cell's temporal sample cap. Phase 3. */
    hashGridCacheMaxSampleCount: number;
    /** `gi1_hash_grid_cache_discard_multibounce_ray_probability` (gi1.h:64, default 0.7). Phase 3. */
    hashGridCacheDiscardMultibounceRayProbability: number;
    /** `gi1_hash_grid_cache_max_multibounce_sample_count` (gi1.h:65, default 16). Phase 3. */
    hashGridCacheMaxMultibounceSampleCount: number;
    /** `gi1_reservoir_cache_cell_size` (gi1.h:71, default 16): world-space ReSTIR's cell, with `useResampling`. */
    reservoirCacheCellSize: number;
    /**
     * `InterpolateScreenProbes` jitters each pixel's lookup position by up to a
     * spawn tile (gi1.comp:1540-1551; default true, as Capsaicin). The jitter is
     * meant for the GI denoiser (RC4.3) to average away; until it exists it is
     * part of the frame-to-frame change. Off only to measure that (RC4.1 review).
     * Read every frame.
     */
    interpolationJitter: boolean;

    // ---- The screen probes' temporal rules (RC4.1's changes to Capsaicin, register D41-D47) ----
    // Each has a value that brings Capsaicin's behaviour back (RC4.5, compare-toggles.ts), read
    // every frame: no page reload. Changing one listed in SCREEN_HISTORY_SETTINGS starts the
    // screen probes' history over at the next frame, so the old rule's cells don't linger.

    /**
     * How a probe cell carries its past (register D46). `running-mean` (ours): a cell
     * with history moves to this frame's mean of its rays by k / n, n capped at
     * `historySampleCap`. `capsaicin-hysteresis`: this frame's mean, then Capsaicin's
     * biased hysteresis, which keeps the past only against a brightening of over 2×
     * (gi1.comp:1258-1269).
     */
    probeBlend: 'running-mean' | 'capsaicin-hysteresis';
    /** The most samples a cell's running mean counts (D46; 8). 1 makes every cell a one-frame estimate. */
    historySampleCap: number;
    /**
     * A probe cell no ray reached this frame (register D41). `keep-previous` (ours): it
     * keeps its previous value; only a cell with no history takes the backup, the mean of
     * the reached cells' means. `capsaicin`: every such cell takes Capsaicin's backup, the
     * reached cells' summed samples over their number divided again by the number of empty
     * cells (gi1.comp:407-429, 1212-1256), which darkens the probes over time.
     */
    emptyCellFill: 'keep-previous' | 'capsaicin';
    /**
     * Whether a backup fill counts as history (register D47). False (ours): a fill is
     * stored with a negative hit distance and the reprojection and the taps skip it.
     * True (Capsaicin): it is stored with the largest hit distance (65,504, D19) and read
     * later as if measured (gi1.comp:427, 1232).
     */
    backupFillsAsHistory: boolean;
    /**
     * The share of a probe's rays that sample their own cell under ray guiding
     * (register D42; 1/8). 0 is Capsaicin's rule: every ray of a cell with history is
     * guided by the luminance CDF, and a probe whose cells all read black sends every
     * ray to its last cell (gi1.comp:969-973; the CDF over max(total, 1e-5)).
     */
    rayGuidingUniformShare: number;
    /**
     * Where the platform reports a scene change, a probe cell whose last ray crosses a
     * changed box forgets its sample count, so its next ray replaces it (D46; ours).
     * Capsaicin has no such reset (false).
     */
    probeChangeReset: boolean;
    /**
     * The screen probes keep their history from frame to frame (true, as Capsaicin).
     * False drops it before every frame, so each frame runs as the first does: for
     * comparing, not a product setting. The world cache keeps its cells either way.
     */
    probeHistory: boolean;
}

/**
 * Settings whose change starts the screen probes' history over at the next frame:
 * cells made under the old rule (darkened backups, fills kept as history) would
 * otherwise linger under the new one, and the picture would show a mix of both.
 */
export const SCREEN_HISTORY_SETTINGS: readonly (keyof RadianceCacheSettings)[] = ['probeBlend', 'emptyCellFill', 'backupFillsAsHistory'];

/** Capsaicin's defaults (gi1.h:48-71). */
export const DEFAULT_RADIANCE_CACHE_SETTINGS: Readonly<RadianceCacheSettings> = Object.freeze({
    useDirectLighting: true,
    useTemporalFeedback: false,
    useMultibounce: true,
    useResampling: false,
    disableAlphaTesting: false,
    hashGridCacheCellSize: 32,
    hashGridCacheMinCellSize: 0.1,
    hashGridCacheTileCellRatio: 8,
    hashGridCacheNumBuckets: 14,
    hashGridCacheNumTilesPerBucket: 4,
    hashGridCacheMaxSampleCount: 16,
    hashGridCacheDiscardMultibounceRayProbability: 0.7,
    hashGridCacheMaxMultibounceSampleCount: 16,
    reservoirCacheCellSize: 16,
    interpolationJitter: true,
    // Ours (RC4.1, decided: DECISIONS 2026-11-19); compare-toggles.ts lists Capsaicin's values.
    probeBlend: 'running-mean',
    historySampleCap: 8,
    emptyCellFill: 'keep-previous',
    backupFillsAsHistory: false,
    rayGuidingUniformShare: 0.125,
    probeChangeReset: true,
    probeHistory: true,
});

/** The sizes Capsaicin derives from the render size (`GI1::ScreenProbes::ensureMemoryIsAllocated`, gi1.cpp:113-129). */
export interface ScreenProbeLayout {
    readonly width: number;
    readonly height: number;
    /** Probe tiles (8 × 8 pixels) across and down: the probe mask's size (`probe_count`, gi1.cpp:117-118). */
    readonly probeCount: readonly [number, number];
    /** The probe atlas: `probeCount` × 8 texels each way (`probe_buffer_width/height`, gi1.cpp:120-121). */
    readonly atlasSize: readonly [number, number];
    /** Levels of the probe mask's mip chain (`probe_mask_mip_count`, gi1.cpp:123; filled by RC2.2's `FilterProbeMask`). */
    readonly probeMaskMipCount: number;
    /** Spawn tiles (16 × 16 pixels): the most probes a frame can spawn (`max_probe_spawn_count`, gi1.cpp:126-127). */
    readonly maxProbeSpawnCount: number;
    /** One ray per cell of every spawn tile (`max_ray_count`, gi1.cpp:129). */
    readonly maxRayCount: number;
}

/** The layout for a `width` × `height` screen. */
export function screenProbeLayout(width: number, height: number): ScreenProbeLayout {
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
        throw new RangeError(`@pmndrs/gi: the radiance cache needs a screen of at least 1 × 1 whole pixels, got ${width} × ${height}`);
    }
    const probeCount: [number, number] = [Math.ceil(width / PROBE_SIZE), Math.ceil(height / PROBE_SIZE)];
    const maxProbeSpawnCount = Math.ceil(width / PROBE_SPAWN_TILE_SIZE) * Math.ceil(height / PROBE_SPAWN_TILE_SIZE);
    return {
        width,
        height,
        probeCount,
        atlasSize: [probeCount[0] * PROBE_SIZE, probeCount[1] * PROBE_SIZE],
        // gfxCalculateMipCount: down to 1 × 1 along the longer side.
        probeMaskMipCount: Math.floor(Math.log2(Math.max(probeCount[0], probeCount[1]))) + 1,
        maxProbeSpawnCount,
        maxRayCount: maxProbeSpawnCount * PROBE_CELLS,
    };
}
