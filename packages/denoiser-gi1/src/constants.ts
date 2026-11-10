/**
 * The GI denoiser's constants, Capsaicin v1.3's (914b915), each with the line
 * it comes from. The WGSL kernels (wgsl.ts) and the CPU reference
 * (reference.ts) both read them from here, so the two can't drift apart.
 *
 * Paths are below `src/core/src/render_techniques/gi1/` in Capsaicin.
 */

/** `kGIDenoiser_MaxBlurMask` (gi_denoiser.hlsl:26): the widest blur radius, and the history length below which a pixel is blurred. */
export const MAX_BLUR_MASK = 8;

/** The normal test for reusing a previous pixel (`dot(normal, previous_normal) > 0.95f`, gi1.comp:4051). */
export const REPROJECTION_NORMAL_COSINE = 0.95;

/**
 * The reprojection's distance test grows at grazing angles: the cell size is
 * multiplied by `lerp(1, 5, pow(1 - cos, 6))` (gi1.comp:4026).
 */
export const GRAZING_CELL_SCALE = 5;
export const GRAZING_CELL_EXPONENT = 6;

/** `kOneOverSqrtOfTwo` (gi1.comp:4031): a tap's weight falls to 0 at √2 pixels from the reprojected position. */
export const ONE_OVER_SQRT_OF_TWO = 0.707107;

/** The luminance-change estimate's smoothing: `lerp(color_delta, lumaA - lumaB, 1.0f / 8.0f)` (gi1.comp:4077). */
export const COLOR_DELTA_SMOOTHING = 1 / 8;

/** The smallest history luminance the change is divided by (`max(lumaB, 1e-4f)`, gi1.comp:4078). */
export const MIN_HISTORY_LUMINANCE = 1e-4;

/**
 * The history's length cap: `lerp(4, 8 × min(|count|, 8), alpha_blend) ×
 * vignette`, at least 1 (gi1.comp:4092). A changing pixel keeps 4 samples;
 * a steady one up to 64.
 */
export const MIN_HISTORY_SAMPLES = 4;
export const HISTORY_SAMPLES_PER_COUNT = 8;

/** The vignette that shortens the history at the screen's edges: `pow(15 × u(1 - v) × v(1 - u), 0.25)` (gi1.comp:4090-4091). */
export const VIGNETTE_SCALE = 15;
export const VIGNETTE_EXPONENT = 0.25;

/**
 * FilterGI's depth weight `exp2(-k × |1 - depth / centre depth|)`: k = 200
 * for a pixel with history samples, 20 for one without (gi1.comp:4134).
 */
export const FILTER_DEPTH_SHARPNESS_WITH_HISTORY = 2e2;
export const FILTER_DEPTH_SHARPNESS_WITHOUT_HISTORY = 2e1;

/** FilterGI's normal weight, `max(dot(n, centre n), 0)` squared twice (gi1.comp:4135-4136). */
export const FILTER_NORMAL_POWER = 4;

/** `GIDenoiser_RemoveNaNs`' floor on `1 - x / (1 + x)` (gi_denoiser.hlsl:69), which also caps a value at about 1e4. */
export const REMOVE_NANS_FLOOR = 1e-4;

/** Rec. 709 luminance (`luminance`, src/core/src/math/color.hlsl:34-37). */
export const LUMINANCE_WEIGHTS: readonly [number, number, number] = [0.2126, 0.7152, 0.0722];

/**
 * GI-1's screen-probe size in pixels (`ScreenProbes::probe_size_`, gi1.h:191),
 * which sets the reprojection's distance tolerance through Capsaicin's
 * `cell_size` (gi1.cpp:1839-1843). The denoiser's default; a host without
 * screen probes keeps it.
 */
export const DEFAULT_CELL_PIXELS = 8;

/** Threads per workgroup side: `[numthreads(8, 8, 1)]` on both kernels (gi1.comp:3996, 4104). */
export const WORKGROUP_SIZE = 8;
