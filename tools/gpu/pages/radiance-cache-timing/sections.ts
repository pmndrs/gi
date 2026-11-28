/**
 * The radiance cache's timed passes, named and grouped as native Capsaicin's
 * per-pass CSV (`research/gi-1.0/native-capsaicin.md`, "GI-1.2 per-pass
 * timings": the groups its `summarize.py` sums). The technique names its
 * passes `radiance-cache/<Capsaicin section>` (`techniques/radiance-cache/
 * timed-pass.ts`, P6.6), so the section is the name without its prefix, but
 * for one case: Capsaicin's `PopulateScreenProbesCells` traces the probe rays
 * and inserts their hits' cells in one kernel; the port times the trace
 * (`PopulateScreenProbesCells (trace)`) and the insert (the world cache's
 * `PopulateScreenProbesCells`) apart. The section is their sum; both parts
 * are reported.
 *
 * Native sections the port has no pass for, and why:
 * - `ReprojectGI`, `FilterGI` (the GI denoiser): `Gi1Denoiser` from
 *   `denoiser/ffx` records its passes without timestamp writes when it
 *   encodes into the host's encoder (its own timer works only in its
 *   `dispatch`), so the platform can't time them. Not timed until the
 *   denoiser takes timestamp writes from its caller;
 * - `ResolveGI1`: the host's composite applies the GI (three's
 *   `DDGICompositeNode` on this page), in three's own passes, which the
 *   platform doesn't time;
 * - `ClearReservoirs`, `CompactReservoirs`, `ResampleReservoirs`,
 *   `CalculateLightSamplerBounds`, `BuildLightSampler`: the port lights hits
 *   with the platform's hit shading, not Capsaicin's light sampler (its light
 *   grid, when built, is a platform pass in `encode`);
 * - `MarkFireflies`, `CleanupFireflies`, the reflection passes: not ported.
 *
 * Not timed either: the technique's buffer and texture copies between passes
 * and the platform's screen-history copy (`gi/screen-history`, the screen's
 * depth and normals kept for the next frame).
 */

import type { RadianceCacheSettings } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { POPULATE_TRACE_SECTION, RADIANCE_CACHE_TIMER_PREFIX } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/timed-pass.ts';

/** Native Capsaicin's groups, in its table's order, then SSGI, then what only the port has. */
export const CAPSAICIN_GROUPS = ['probe tracing and cell population', 'cache update and resolve', 'screen probes', 'denoise and resolve', 'ssgi', 'ours only'] as const;

export type CapsaicinGroup = (typeof CAPSAICIN_GROUPS)[number];

/** The groups native Capsaicin's "diffuse subtotal" adds up (SSGI is outside it). */
export const DIFFUSE_GROUPS: readonly CapsaicinGroup[] = ['probe tracing and cell population', 'cache update and resolve', 'screen probes', 'denoise and resolve'];

/** The settings that decide which sections run: what the page varies, plus what it leaves at the defaults. */
export type TimedSettings = Pick<RadianceCacheSettings, 'useMultibounce' | 'cellShading' | 'worldCache' | 'probeFilter' | 'ssgi' | 'neuralCache' | 'secondBounce'>;

interface SectionEntry {
    readonly section: string;
    readonly group: CapsaicinGroup;
    /** Whether the section runs under these settings; every frame when absent. */
    readonly runsWhen?: (settings: TimedSettings) => boolean;
}

const hashGrid = (settings: TimedSettings): boolean => settings.worldCache === 'gi1-hash-grid';
const sharc = (settings: TimedSettings): boolean => settings.worldCache === 'sharc';
/**
 * GI-1.2's second-bounce rays. The page stacks nothing under the cache, so
 * `secondBounce: 'auto'` (the default) resolves to GI-1.2's rays and
 * `SampleBounceSourceCells` (experiment E10) never runs here.
 */
const secondBounceRays = (settings: TimedSettings): boolean => hashGrid(settings) && settings.useMultibounce;
const neuralTail = (settings: TimedSettings): boolean => secondBounceRays(settings) && settings.neuralCache === 'tail';

/** Each section the port times, in the frame's order, with its group and when it runs. */
export const SECTIONS: readonly SectionEntry[] = [
    { section: 'ClearCounters', group: 'screen probes' },
    { section: 'ReprojectScreenProbes', group: 'screen probes' },
    { section: 'LookupScreenProbes', group: 'screen probes' },
    { section: 'SpawnScreenProbes', group: 'screen probes' },
    { section: 'PatchScreenProbes', group: 'screen probes' },
    { section: 'SampleScreenProbes', group: 'probe tracing and cell population' },
    // The trace here, the cells' insert in the hash grid; with SHaRC the trace alone.
    { section: 'PopulateScreenProbesCells', group: 'probe tracing and cell population' },
    // GI-1.2's hash grid (world-cache/hash-grid-cache.ts).
    { section: 'PurgeRadianceCache', group: 'cache update and resolve', runsWhen: hashGrid },
    { section: 'PopulateMultibounceCells', group: 'probe tracing and cell population', runsWhen: secondBounceRays },
    { section: 'GenerateReservoirs', group: 'probe tracing and cell population', runsWhen: hashGrid },
    { section: 'GenerateMultibounceReservoirs', group: 'probe tracing and cell population', runsWhen: secondBounceRays },
    { section: 'PopulateRadianceCache', group: 'probe tracing and cell population', runsWhen: hashGrid },
    // Lever E8's cells lit on a budget (P6.2): they light the cells, so they sit in the population group.
    { section: 'ClaimCells', group: 'probe tracing and cell population', runsWhen: (settings) => hashGrid(settings) && settings.cellShading === 'budgeted' },
    { section: 'PrioritizeCells', group: 'probe tracing and cell population', runsWhen: (settings) => hashGrid(settings) && settings.cellShading === 'budgeted' },
    { section: 'ShadeCells', group: 'probe tracing and cell population', runsWhen: (settings) => hashGrid(settings) && settings.cellShading === 'budgeted' },
    // pmndrs/gi#54: one traced bounce from each lit cell's sample hit, before ShadeCells.
    { section: 'SharedCellBounce', group: 'probe tracing and cell population', runsWhen: (settings) => secondBounceRays(settings) && settings.secondBounce === 'shared-cell' },
    // Experiment E10 (P6.3): the second bounce read from a technique stacked under the cache. The
    // page stacks none, so it never runs here; listed so a run with one maps its pass.
    { section: 'SampleBounceSourceCells', group: 'cache update and resolve', runsWhen: () => false },
    { section: 'UpdateRadianceCache', group: 'cache update and resolve', runsWhen: hashGrid },
    // The neural tail (P6.4), with GI-1.2's second-bounce rays only.
    { section: 'NeuralTailGather', group: 'ours only', runsWhen: neuralTail },
    { section: 'NeuralTailTraining', group: 'ours only', runsWhen: neuralTail },
    { section: 'NeuralTailInference', group: 'ours only', runsWhen: neuralTail },
    { section: 'UpdateMultibounceCells', group: 'cache update and resolve', runsWhen: secondBounceRays },
    { section: 'ResolveCells', group: 'cache update and resolve', runsWhen: hashGrid },
    // The SHaRC-style cache in the world-cache slot (P6.1): its update path, resolve and lookup.
    { section: 'SharcBegin', group: 'ours only', runsWhen: sharc },
    { section: 'SharcBounce1', group: 'ours only', runsWhen: sharc },
    { section: 'SharcBounce2', group: 'ours only', runsWhen: sharc },
    { section: 'SharcLight', group: 'ours only', runsWhen: sharc },
    { section: 'SharcCombine', group: 'ours only', runsWhen: sharc },
    { section: 'SharcResolve', group: 'ours only', runsWhen: sharc },
    { section: 'SharcLookup', group: 'ours only', runsWhen: sharc },
    { section: 'BlendScreenProbes', group: 'screen probes' },
    { section: 'SSGI', group: 'ssgi', runsWhen: (settings) => settings.ssgi },
    { section: 'FilterProbeMask', group: 'screen probes' },
    { section: 'FilterScreenProbes', group: 'screen probes', runsWhen: (settings) => settings.probeFilter },
    { section: 'ProjectScreenProbes', group: 'screen probes' },
    { section: 'InterpolateScreenProbes', group: 'screen probes' },
    { section: 'ReorderScreenProbes', group: 'screen probes' },
];

const PREFIX = `${RADIANCE_CACHE_TIMER_PREFIX}/`;
const TRACE_SUFFIX = POPULATE_TRACE_SECTION.slice('PopulateScreenProbesCells'.length);

/**
 * The section a pass name in `stats.passes` belongs to, or null for a pass
 * that isn't the radiance cache's (the platform's own, such as
 * `platform/top-level refit`).
 */
export function sectionOf(passName: string): string | null {
    if (!passName.startsWith(PREFIX)) return null;
    const section = passName.slice(PREFIX.length);
    return section.endsWith(TRACE_SUFFIX) ? section.slice(0, -TRACE_SUFFIX.length) : section;
}

export function groupOf(section: string): CapsaicinGroup | undefined {
    return SECTIONS.find((entry) => entry.section === section)?.group;
}

/** Whether a section runs every frame under these settings. */
export function sectionRuns(section: string, settings: TimedSettings): boolean {
    const entry = SECTIONS.find((candidate) => candidate.section === section);
    return entry !== undefined && (entry.runsWhen?.(settings) ?? true);
}

/** What the page reports for one scene and setting (main.ts; to-csv.ts reads it). */
export interface SettingTiming {
    /** A short name for the setting, used in the CSV's metric names: `mb_on`, `mb_off`, `mb_on_budgeted`, `mb_on_sharc`... */
    readonly name: string;
    readonly settings: TimedSettings;
    readonly framesTimed: number;
    /** Median GPU ms of each timed pass, by its `stats.passes` name. */
    readonly passes: Record<string, number>;
    /** Median GPU ms of each section (its passes summed per frame), in the frame's order. */
    readonly sections: Record<string, number>;
    /** Sum of the section medians per native Capsaicin group. */
    readonly groups: Record<CapsaicinGroup, number>;
    /** Native's "diffuse subtotal": the four GI-1 groups, without SSGI and what only the port has. */
    readonly diffuseSubtotalMs: number;
    /** Median over frames of every radiance-cache pass summed. */
    readonly radianceCacheFrameMs: number;
    /** The platform's own passes (`platform/...`), medians. */
    readonly platformPasses: Record<string, number>;
    /** The technique's memory as the platform reports it, in bytes. */
    readonly memoryBytes: number;
}

export interface SceneTiming {
    readonly triangles: number;
    readonly maxTextureSize: number;
    readonly settings: readonly SettingTiming[];
}
