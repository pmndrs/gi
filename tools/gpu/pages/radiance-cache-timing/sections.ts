/**
 * The radiance cache's timed passes as native Capsaicin's per-pass CSV names
 * and groups them (`ddgi-research/research/gi-1.0/native-capsaicin.md`,
 * "GI-1.2 per-pass timings": the groups `summarize.py` sums). Our passes are
 * named `radiance-cache/<Capsaicin section>` (timed-pass.ts, RC5.1), so the
 * section is the name without its prefix, but for two cases:
 *
 * - Capsaicin's `PopulateScreenProbesCells` traces the probe rays and inserts
 *   their hits' cells in one kernel; the port times the trace
 *   (`PopulateScreenProbesCells (trace)`) and the insert (the world cache's
 *   `PopulateScreenProbesCells`) apart. The section is their sum; both parts
 *   are reported.
 * - The GI denoiser's passes carry its own label first
 *   (`radiance-cache/gi-denoiser/ReprojectGI`).
 *
 * Native sections the port has no pass for, and why:
 * - `ResolveGI1` (denoise and resolve): the host's composite applies the GI
 *   (three's `DDGICompositeNode` for this page), in three's own passes, which
 *   the platform doesn't time;
 * - `ClearReservoirs`, `CompactReservoirs`, `ResampleReservoirs`,
 *   `CalculateLightSamplerBounds`, `BuildLightSampler` (probe tracing and cell
 *   population): the port lights hits with the platform's hit shading, not
 *   Capsaicin's light sampler (its light grid, when built, is a platform pass
 *   in `encode`);
 * - `MarkFireflies`, `CleanupFireflies`, the reflection passes: reflections
 *   are not ported.
 */

/** Native Capsaicin's groups, in its table's order, then SSGI, then what only the port has. */
export const CAPSAICIN_GROUPS = [
    'probe tracing and cell population',
    'cache update and resolve',
    'screen probes',
    'denoise and resolve',
    'ssgi',
    'ours only',
] as const;

export type CapsaicinGroup = (typeof CAPSAICIN_GROUPS)[number];

/** The groups native Capsaicin's "diffuse subtotal" adds up (SSGI is outside it). */
export const DIFFUSE_GROUPS: readonly CapsaicinGroup[] = ['probe tracing and cell population', 'cache update and resolve', 'screen probes', 'denoise and resolve'];

/** Each section the port times, in GI-1's order (gi1.cpp:2238-2936), with its group. */
export const SECTIONS: readonly { readonly section: string; readonly group: CapsaicinGroup; readonly multibounceOnly?: true }[] = [
    { section: 'SSGI', group: 'ssgi' },
    { section: 'PurgeRadianceCache', group: 'cache update and resolve' },
    { section: 'ReprojectScreenProbes', group: 'screen probes' },
    { section: 'LookupScreenProbes', group: 'screen probes' },
    { section: 'SpawnScreenProbes', group: 'screen probes' },
    { section: 'PatchScreenProbes', group: 'screen probes' },
    { section: 'SampleScreenProbes', group: 'probe tracing and cell population' },
    { section: 'PopulateScreenProbesCells', group: 'probe tracing and cell population' },
    { section: 'PopulateMultibounceCells', group: 'probe tracing and cell population', multibounceOnly: true },
    { section: 'GenerateReservoirs', group: 'probe tracing and cell population' },
    { section: 'GenerateMultibounceReservoirs', group: 'probe tracing and cell population', multibounceOnly: true },
    { section: 'PopulateRadianceCache', group: 'probe tracing and cell population' },
    { section: 'UpdateRadianceCache', group: 'cache update and resolve' },
    { section: 'UpdateMultibounceCells', group: 'cache update and resolve', multibounceOnly: true },
    { section: 'ResolveCells', group: 'cache update and resolve' },
    { section: 'BlendScreenProbes', group: 'screen probes' },
    { section: 'ReorderScreenProbes', group: 'screen probes' },
    { section: 'FilterProbeMask', group: 'screen probes' },
    { section: 'FilterScreenProbes', group: 'screen probes' },
    { section: 'ProjectScreenProbes', group: 'screen probes' },
    { section: 'InterpolateScreenProbes', group: 'screen probes' },
    { section: 'ReprojectGI', group: 'denoise and resolve' },
    { section: 'FilterGI', group: 'denoise and resolve' },
    // The screen and denoiser history copies (Capsaicin's framework copies depth and normals, untimed).
    { section: 'keep-history', group: 'ours only' },
];

const RADIANCE_CACHE_PREFIX = 'radiance-cache/';
const DENOISER_PREFIX = 'gi-denoiser/';
const TRACE_SUFFIX = ' (trace)';

/**
 * The Capsaicin section a pass name in `stats.passes` belongs to, or null for
 * a pass that isn't the radiance cache's (the platform's own, such as
 * `platform/top-level refit`).
 */
export function sectionOf(passName: string): string | null {
    if (!passName.startsWith(RADIANCE_CACHE_PREFIX)) return null;
    let section = passName.slice(RADIANCE_CACHE_PREFIX.length);
    if (section.startsWith(DENOISER_PREFIX)) section = section.slice(DENOISER_PREFIX.length);
    if (section.endsWith(TRACE_SUFFIX)) section = section.slice(0, -TRACE_SUFFIX.length);
    return section;
}

export function groupOf(section: string): CapsaicinGroup | undefined {
    return SECTIONS.find((entry) => entry.section === section)?.group;
}

/** What the page reports for one scene and setting (main.ts; to-csv.ts reads it). */
export interface SettingTiming {
    readonly useMultibounce: boolean;
    readonly framesTimed: number;
    /** Median GPU ms of each timed pass, by its `stats.passes` name. */
    readonly passes: Record<string, number>;
    /** Median GPU ms of each Capsaicin section (its passes summed per frame), in GI-1's order. */
    readonly sections: Record<string, number>;
    /** Sum of the section medians per native Capsaicin group. */
    readonly groups: Record<CapsaicinGroup, number>;
    /** Native's "diffuse subtotal": the four GI-1 groups, without SSGI and what only the port has. */
    readonly diffuseSubtotalMs: number;
    /** Median over frames of every radiance-cache pass summed. */
    readonly radianceCacheFrameMs: number;
    /** The platform's own passes (`platform/...`), medians. */
    readonly platformPasses: Record<string, number>;
    readonly counters: Record<string, number>;
}

export interface SceneTiming {
    readonly triangles: number;
    readonly maxTextureSize: number;
    readonly settings: readonly SettingTiming[];
}
