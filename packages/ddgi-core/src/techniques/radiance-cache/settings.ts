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
 * configuration"). Capsaicin's SSGI technique's options are here too (RC4.2, `ssgi*`), but for
 * `ssgi_unroll_kernel_` (default off), a variant with the counts fixed at 1 and 2 (register D57).
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
    /**
     * `gi1_use_multibounce` (gi1.h:55, default true): GI-1.2's second-bounce
     * cells. Phase 3. Like Capsaicin's `USE_MULTI_BOUNCE` define it sizes the
     * world cache at `prepare` (register D42): off, the 535 MB of indirect
     * values and scratch aren't made. Turning it on takes effect at the next
     * `prepare` (call `GIPlatform.prepare()` again); turning it off stops the
     * second bounce at once and frees the memory at the next `prepare`.
     */
    useMultibounce: boolean;
    /**
     * `environment_light_enable` (Capsaicin's `LightBuilder`, light_builder.h:63,
     * default true): the sky is one of the lights the world cache's hits
     * sample, so a cache cell gets the sky light that reaches its surface
     * (RC6.1, register D33). Off, the cells get the scene's lights alone and
     * the sky reaches the probes only through rays that miss. Read every
     * frame.
     */
    environmentLightEnable: boolean;
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
     * Capsaicin's SSGI technique (`SSGI::RenderOptions`, ssgi.h:46-53), which
     * the GI-1.2 renderer runs before GI-1 (gi1_renderer.cpp:60-61) and whose
     * occlusion, bent normal and near-field light `InterpolateScreenProbes`
     * reads (`HAS_OCCLUSION`, gi1.comp:1621-1626). RC4.2.
     *
     * `ssgi_slice_count_` (default 1): directions around the pixel, each a
     * slice with a horizon on both sides.
     */
    ssgiSliceCount: number;
    /** `ssgi_step_count_` (default 2): depth samples per side of a slice. */
    ssgiStepCount: number;
    /** `ssgi_view_radius_` (default 0.2): how far the samples reach, in scene units. */
    ssgiViewRadius: number;
    /** `ssgi_falloff_range_` (default 0.05): the outer share of the radius over which a sample's weight falls to 0. */
    ssgiFalloffRange: number;
    /**
     * Ours, not Capsaicin's (default true): run GI-1's image-space denoiser
     * (`ReprojectGI`, `FilterGI` ×2; RC4.3, gi-denoiser.ts), which Capsaicin
     * always runs. Off, the shading module reads `InterpolateScreenProbes`'
     * output directly, for comparisons. Read every frame.
     */
    useGIDenoiser: boolean;
}

/** Capsaicin's defaults (gi1.h:48-71; SSGI's, ssgi.h:48-51; the light builder's, light_builder.h:61-63). */
export const DEFAULT_RADIANCE_CACHE_SETTINGS: Readonly<RadianceCacheSettings> = Object.freeze({
    useDirectLighting: true,
    useTemporalFeedback: false,
    useMultibounce: true,
    environmentLightEnable: true,
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
    ssgiSliceCount: 1,
    ssgiStepCount: 2,
    ssgiViewRadius: 0.2,
    ssgiFalloffRange: 0.05,
    useGIDenoiser: true,
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
        throw new RangeError(`@pmndrs/ddgi: the radiance cache needs a screen of at least 1 × 1 whole pixels, got ${width} × ${height}`);
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
