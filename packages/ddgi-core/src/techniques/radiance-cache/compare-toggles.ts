/**
 * The radiance cache's compare toggles (RC4.5): every setting that changes how
 * the technique behaves over time, each with the value that brings
 * Capsaicin's behaviour back, so the two can be compared live (Dennis,
 * 2026-11-19: "does your speed improvement/history have a toggle? that way we
 * can see"). The viewer's "Radiance cache · compare" section is drawn from
 * this list, and its links store each toggle under its `urlName`.
 *
 * Adding one (a later branch's setting, docs/design/radiance-cache.md,
 * "Compare toggles"):
 *
 * 1. add the setting to `RadianceCacheSettings` and its default to
 *    `DEFAULT_RADIANCE_CACHE_SETTINGS` (settings.ts), read every frame, or
 *    listed in `SCREEN_HISTORY_SETTINGS` if a change must start the history
 *    over;
 * 2. add an entry below with `toggle(...)`: its group, a short label, a link
 *    name no other entry uses, Capsaicin's value and one plain line of help
 *    saying what it does and which value is Capsaicin's.
 *
 * The viewer, its links and the unit tests pick it up from here. Engine-free
 * and DOM-free, so the viewer's link code (url-state.ts) can import it.
 */

import { DEFAULT_RADIANCE_CACHE_SETTINGS, PREPARE_SETTINGS, SCREEN_HISTORY_SETTINGS, type RadianceCacheSettings } from './settings.ts';
import { DEFAULT_SHARC_SETTINGS, type SharcSettings } from './world-cache/sharc/sharc-settings.ts';

/** The panel's groups, in the order it shows them. */
export const COMPARE_GROUPS = ['screen-probes', 'world-cache', 'denoiser', 'experiments'] as const;
export type CompareGroup = (typeof COMPARE_GROUPS)[number];

export const COMPARE_GROUP_LABELS: Readonly<Record<CompareGroup, string>> = {
    'screen-probes': 'Screen probes',
    'world-cache': 'World cache',
    denoiser: 'Denoiser',
    experiments: 'Experiments',
};

/** Settings whose values are booleans, strings of a few choices, or numbers: the kinds a toggle can show. */
type KeysOfType<S, T> = { [K in keyof S]: S[K] extends T ? K : never }[keyof S];
/** A SHaRC setting, inside `RadianceCacheSettings.sharc` (P6.1), named `sharc.<key>`. */
type SharcSetting<T> = `sharc.${KeysOfType<SharcSettings, T> & string}`;
/**
 * Choices the host makes about how it sets the technique up, not settings of the technique
 * (`host.<key>`): the host reads them from the compare values (`hostCompareValue`), and the
 * technique's settings never see them. Each is read when the host makes the technique.
 */
export const HOST_COMPARE_DEFAULTS = {
    /** RC4.6: the viewer stacks a DDGI volume under the radiance cache for its second bounce (E10); true runs the cache alone. */
    'host.cacheAlone': false,
} as const;
type HostSetting = keyof typeof HOST_COMPARE_DEFAULTS;
export type CompareBooleanSetting = KeysOfType<RadianceCacheSettings, boolean> | SharcSetting<boolean> | HostSetting;
export type CompareNumberSetting = KeysOfType<RadianceCacheSettings, number> | SharcSetting<number>;
export type CompareChoiceSetting = KeysOfType<RadianceCacheSettings, string>;
export type CompareSetting = CompareBooleanSetting | CompareNumberSetting | CompareChoiceSetting;
type CompareValue = boolean | number | string;

/** What every toggle says about itself. */
interface ToggleCommon<K extends CompareSetting, V extends CompareValue> {
    /** The technique's setting it drives (`sharc.<key>` for one of the SHaRC backend's). */
    readonly setting: K;
    readonly group: CompareGroup;
    /** A few words for the panel. */
    readonly label: string;
    /** Its name in the viewer's links (lowercase letters and digits, unique). */
    readonly urlName: string;
    /** One plain line: what it does, and which value is Capsaicin's (or that Capsaicin has none). */
    readonly help: string;
    /** The divergence-register row it undoes (docs/design/radiance-cache.md), if any. */
    readonly register?: string;
    /** The decided value (`DEFAULT_RADIANCE_CACHE_SETTINGS`, or `DEFAULT_SHARC_SETTINGS`). */
    readonly defaultValue: V;
    /** The value that behaves as Capsaicin v1.3 does; absent where Capsaicin has nothing like it. */
    readonly capsaicin?: V;
    /** Changing it starts the screen probes' history over (`SCREEN_HISTORY_SETTINGS`). */
    readonly resetsHistory: boolean;
    /** Read at `prepare`: changing it makes the technique again (`PREPARE_SETTINGS`; the viewer restarts it). */
    readonly restartsTechnique: boolean;
}

export interface CompareBooleanToggle extends ToggleCommon<CompareBooleanSetting, boolean> {
    readonly kind: 'boolean';
}

export interface CompareChoiceToggle extends ToggleCommon<CompareChoiceSetting, string> {
    readonly kind: 'choice';
    /** Every value it can take, with the panel's words for it. */
    readonly choices: readonly { readonly value: string; readonly label: string }[];
}

export interface CompareNumberToggle extends ToggleCommon<CompareNumberSetting, number> {
    readonly kind: 'number';
    readonly min: number;
    readonly max: number;
    readonly step: number;
}

export type CompareToggle = CompareBooleanToggle | CompareChoiceToggle | CompareNumberToggle;

/** The compare settings that differ from the defaults, by setting name (what a link stores). */
export type CompareValues = { [K in CompareSetting]?: CompareValue };

type ToggleInput<T extends CompareToggle> = Omit<T, 'defaultValue' | 'resetsHistory' | 'restartsTechnique'>;

const SHARC_PREFIX = 'sharc.';
const HOST_PREFIX = 'host.';

/** A host choice's value in `values`, or its default (`HOST_COMPARE_DEFAULTS`). */
export function hostCompareValue<K extends HostSetting>(values: CompareValues, setting: K): (typeof HOST_COMPARE_DEFAULTS)[K] {
    const value = (values as Record<string, CompareValue | undefined>)[setting];
    return (value === undefined ? HOST_COMPARE_DEFAULTS[setting] : value) as (typeof HOST_COMPARE_DEFAULTS)[K];
}

/** Whether the toggle is a host choice (`host.<key>`), which the technique's settings don't hold. */
export function isHostToggle(entry: { readonly setting: CompareSetting }): boolean {
    return entry.setting.startsWith(HOST_PREFIX);
}

/**
 * A toggle's current value in `settings` (a `sharc.<key>` read through `settings.sharc` over
 * SHaRC's defaults). A host choice isn't in the settings: this gives its default.
 */
export function compareSettingValue(settings: Readonly<RadianceCacheSettings>, setting: CompareSetting): CompareValue {
    if (setting.startsWith(HOST_PREFIX)) return HOST_COMPARE_DEFAULTS[setting as HostSetting];
    if (setting.startsWith(SHARC_PREFIX)) {
        const key = setting.slice(SHARC_PREFIX.length) as keyof SharcSettings;
        return (settings.sharc[key] ?? DEFAULT_SHARC_SETTINGS[key]) as CompareValue;
    }
    return settings[setting as keyof RadianceCacheSettings] as CompareValue;
}

/** Sets a toggle's value in `settings` (a SHaRC one in a new `settings.sharc`, which the cache reads every frame). */
function setCompareSetting(settings: RadianceCacheSettings, setting: CompareSetting, value: CompareValue): void {
    if (setting.startsWith(HOST_PREFIX)) return; // the host's, not the technique's
    if (setting.startsWith(SHARC_PREFIX)) {
        settings.sharc = { ...settings.sharc, [setting.slice(SHARC_PREFIX.length)]: value };
        return;
    }
    (settings as unknown as Record<string, CompareValue>)[setting] = value;
}

/** An entry, with its default, history reset and restart taken from settings.ts so they can't drift. */
function toggle<T extends CompareToggle>(input: ToggleInput<T>): T {
    const host = input.setting.startsWith(HOST_PREFIX);
    const top = host || input.setting.startsWith(SHARC_PREFIX) ? null : (input.setting as keyof RadianceCacheSettings);
    return {
        ...input,
        defaultValue: compareSettingValue(DEFAULT_RADIANCE_CACHE_SETTINGS, input.setting),
        resetsHistory: top !== null && SCREEN_HISTORY_SETTINGS.includes(top),
        // A host choice shapes how the technique is made.
        restartsTechnique: host || (top !== null && PREPARE_SETTINGS.includes(top)),
    } as T;
}

/**
 * Every compare toggle, in the panel's order within each group. Later branches
 * add theirs here: RC4.2 (SSGI on/off), RC4.3 (the denoiser on/off, its
 * history clamp and clip, its change reset), P6.1 (the world-cache backend),
 * P6.2/2b (cell shading, budget, cell change reset, history), P6.3, P6.4.
 */
export const RADIANCE_CACHE_COMPARE_TOGGLES: readonly CompareToggle[] = [
    // ---- Screen probes (RC4.1's changes, register D41-D47) ----
    toggle<CompareChoiceToggle>({
        kind: 'choice',
        setting: 'probeBlend',
        group: 'screen-probes',
        label: 'Probe blend',
        urlName: 'rcblend',
        register: 'D46',
        choices: [
            { value: 'running-mean', label: 'Running mean (ours)' },
            { value: 'capsaicin-hysteresis', label: "Biased hysteresis (Capsaicin's)" },
        ],
        capsaicin: 'capsaicin-hysteresis',
        help: "How a probe cell keeps its past. Ours averages a cell's rays over up to the history cap; Capsaicin's takes each frame's rays as they come and keeps the past only against a sudden brightening, which reads darker and noisier.",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'historySampleCap',
        group: 'screen-probes',
        label: 'History cap (samples)',
        urlName: 'rccap',
        register: 'D46',
        min: 1,
        max: 32,
        step: 1,
        capsaicin: 1,
        help: 'The most samples the running mean averages a cell over: more is steadier but slower to follow a change; 1 makes every cell this frame alone (Capsaicin keeps no count; 1 is closest).',
    }),
    toggle<CompareChoiceToggle>({
        kind: 'choice',
        setting: 'emptyCellFill',
        group: 'screen-probes',
        label: 'Cells no ray reached',
        urlName: 'rcfill',
        register: 'D41',
        choices: [
            { value: 'keep-previous', label: 'Keep their last value (ours)' },
            { value: 'capsaicin', label: "Darkened backup (Capsaicin's)" },
        ],
        capsaicin: 'capsaicin',
        help: "What a probe cell gets in a frame no ray reached it. Ours keeps its last value; Capsaicin's fills it with the probe's average divided by the number of empty cells, which darkens GI-1.2 over time (to about 0.87 of the light under a constant sky).",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'backupFillsAsHistory',
        group: 'screen-probes',
        label: 'Fills count as history',
        urlName: 'rcfillhist',
        register: 'D47',
        capsaicin: true,
        help: "Whether a cell filled in for lack of rays is later reused as if it had been measured. Off (ours): a fill is made again each frame; on (Capsaicin's): it is kept and moved like a real sample.",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'rayGuidingUniformShare',
        group: 'screen-probes',
        label: 'Rays kept unguided (share)',
        urlName: 'rcguide',
        register: 'D42',
        min: 0,
        max: 1,
        step: 0.025,
        capsaicin: 0,
        help: "The share of each probe's rays that look in their own direction instead of following the bright cells, so a dark direction is still checked now and then (ours 1/8). 0 is Capsaicin's: every ray follows the bright cells.",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'probeChangeReset',
        group: 'screen-probes',
        label: 'Forget history where the scene changed',
        urlName: 'rcreset',
        register: 'D46',
        capsaicin: false,
        help: "Where a light or object moved, probe cells looking through that spot start their average over, so the change shows at once (ours). Capsaicin has no such reset (off).",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'interpolationJitter',
        group: 'screen-probes',
        label: 'Jitter the probe lookup',
        urlName: 'rcjitter',
        capsaicin: true,
        help: "Each pixel reads the probes from a slightly shifted spot each frame, for the denoiser to average away (on, as Capsaicin). Off shows how much of the frame-to-frame change is this jitter.",
    }),

    // ---- World cache (Capsaicin's own temporal settings; ours are the same) ----
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'hashGridCacheMaxSampleCount',
        group: 'world-cache',
        label: 'Cell history (samples)',
        urlName: 'rcwcap',
        min: 1,
        max: 64,
        step: 1,
        capsaicin: 16,
        help: "The most samples a world-cache cell averages its light over: more is steadier but slower to follow a change. Capsaicin's 16 is the default.",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'hashGridCacheMaxMultibounceSampleCount',
        group: 'world-cache',
        label: 'Second-bounce history (samples)',
        urlName: 'rcwbounce',
        min: 1,
        max: 64,
        step: 1,
        capsaicin: 16,
        help: "The same for the cells' second bounce (GI-1.2's multibounce). Capsaicin's 16 is the default.",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'environmentLightEnable',
        group: 'world-cache',
        label: 'Sky lights the cells',
        urlName: 'rcskylight',
        register: 'D62',
        capsaicin: true,
        help: "On, a world-cache cell draws the sky as one of its lights (with a shadow ray), as Capsaicin's environment light does (on, the default). Off, the cells get no skylight and the sky reaches the probes only through rays that miss, as before RC3.4.",
    }),

    toggle<CompareChoiceToggle>({
        kind: 'choice',
        setting: 'worldCache',
        group: 'world-cache',
        label: 'World cache',
        urlName: 'rcworld',
        register: 'E1',
        choices: [
            { value: 'gi1-hash-grid', label: "GI-1.2 hash grid (Capsaicin's)" },
            { value: 'sharc', label: 'SHaRC-style cache (experiment)' },
        ],
        capsaicin: 'gi1-hash-grid',
        help: "Which cache lights the probe rays' hits. Capsaicin's GI-1.2 hash grid (the default), or a SHaRC-style hash cache from pmndrs/pathtracer (smaller, more bounces). Changing it starts the radiance cache over.",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'host.cacheAlone',
        group: 'world-cache',
        label: 'Run the cache alone (no DDGI under it)',
        urlName: 'rcalone',
        register: 'E10',
        capsaicin: true,
        help: "Off (ours, what ships): a DDGI volume runs under the radiance cache, updating 256 probes a frame, and gives the cache its second bounce and every bounce after. On: the cache alone, its second bounce from GI-1.2's own rays, as Capsaicin. Changing it starts the radiance cache over.",
    }),
    toggle<CompareChoiceToggle>({
        kind: 'choice',
        setting: 'secondBounce',
        group: 'world-cache',
        label: 'Second bounce from',
        urlName: 'rcsecond',
        register: 'E10',
        choices: [
            { value: 'auto', label: 'Auto (ours)' },
            { value: 'gi1-rays', label: "GI-1.2's own rays (Capsaicin's)" },
            { value: 'fallback', label: 'The technique stacked under it' },
        ],
        capsaicin: 'gi1-rays',
        help: "Where the hash grid's second bounce comes from. Capsaicin traces its own second-bounce rays; ours (auto) reads it from the DDGI volume under the radiance cache when one is there, else the same rays. 'Stacked' asks for DDGI and warns without it. Changing what runs empties the world cache.",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'fallbackIndirectSampleCount',
        group: 'world-cache',
        label: 'Stacked second bounce history (samples)',
        urlName: 'rcfallbackcap',
        register: 'E10',
        min: 1,
        max: 32,
        step: 1,
        help: "With the second bounce from a stacked technique: the most samples a cell averages it over (ours 4). Capsaicin has no stacked technique, so no value of its own.",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'sharc.changeReset',
        group: 'world-cache',
        label: 'SHaRC: forget history where the scene changed',
        urlName: 'rcsharcreset',
        register: 'E7a',
        help: "With the SHaRC cache: where a light or object changed, a cell keeps only a few samples, so the new light shows at once (ours, on). Off: its 32-frame average alone, about 80 frames to follow the sun doubling. Not in Capsaicin.",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'sharc.minSamples',
        group: 'world-cache',
        label: 'SHaRC: samples before a cell is read',
        urlName: 'rcsharcmin',
        min: 1,
        max: 128,
        step: 1,
        help: "With the SHaRC cache: a cell with fewer samples than this is passed over for a coarser one (ours 16). Lower reads finer cells sooner but blotchier. Not in Capsaicin.",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'sharc.historyFrames',
        group: 'world-cache',
        label: 'SHaRC: cell history (frames)',
        urlName: 'rcsharchist',
        min: 1,
        max: 128,
        step: 1,
        help: "With the SHaRC cache: how many frames a cell's average reaches back (ours 32): more is steadier but slower to follow a change. Not in Capsaicin.",
    }),

    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'rayGuidingCorrection',
        group: 'screen-probes',
        label: 'Ray-guiding correction',
        urlName: 'rcguidefix',
        register: 'D64',
        help: "Guided rays favour cells that read bright, so a cell that read low is sampled less and stays low. On (ours) weights each cell's update by the rays it got over the rays it expected, which keeps the mean right; off is RC4.1's weighting. Capsaicin has no running mean.",
    }),

    // ---- The probe filter and SSGI (RC4.2, register D59-D61) ----
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'probeFilter',
        group: 'screen-probes',
        label: 'Probe filter',
        urlName: 'rcfilter',
        capsaicin: true,
        help: "Each probe's cells averaged with up to three neighbour probes on each side before the irradiance is read from them, which takes out much of the probes' noise. On, as Capsaicin always runs it; off only to measure what it does (E9's noise gate is calibrated with it off).",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'probeFilterHistory',
        group: 'screen-probes',
        label: 'Filtered probes as history',
        urlName: 'rcfilterhistory',
        register: 'D59',
        capsaicin: true,
        help: "Whether next frame starts from the filtered probes. Off (ours) keeps the unfiltered ones as history, so the filter doesn't compound frame after frame; on is Capsaicin's.",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'ssgi',
        group: 'screen-probes',
        label: 'SSGI (near-field)',
        urlName: 'rcssgi',
        register: 'D61',
        capsaicin: true,
        help: "Screen-space occlusion and near-field light from the depth buffer, narrowing each pixel's GI to what nearby geometry leaves open. Off by default (noise); Capsaicin always runs it.",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'ssgiSliceCount',
        group: 'screen-probes',
        label: 'SSGI slices',
        urlName: 'rcssgislices',
        min: 1,
        max: 8,
        step: 1,
        capsaicin: 1,
        help: 'Directions SSGI searches around each pixel per frame: more is less noisy and costs more (about 0.8 ms a slice at 1280 × 720 on an RTX 3060). Capsaicin uses 1.',
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'ssgiHistoryFrames',
        group: 'screen-probes',
        label: 'SSGI history (frames)',
        urlName: 'rcssgihistory',
        register: 'D60',
        min: 1,
        max: 32,
        step: 1,
        capsaicin: 1,
        help: "Frames SSGI's results are averaged over, following the surfaces as the camera moves: more is steadier, slower to show a moved object. 1 is Capsaicin's single frame.",
    }),
    toggle<CompareChoiceToggle>({
        kind: 'choice',
        setting: 'cellShading',
        group: 'world-cache',
        label: 'Cell lighting',
        urlName: 'rccellshade',
        register: 'E8',
        choices: [
            { value: 'per-hit', label: "Every hit (Capsaicin's)" },
            { value: 'budgeted', label: 'A budget of cells (id Tech 8)' },
        ],
        capsaicin: 'per-hit',
        help: "How the hash grid lights its cells. Capsaicin draws a light and traces a shadow ray for every hit; 'budgeted' lights at most the budget's cells a frame, one sample each, new and changed cells first, and the rest keep their last value (fewer shadow rays, older cells).",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'cellShadingBudget',
        group: 'world-cache',
        label: 'Cells lit a frame (budget)',
        urlName: 'rccellbudget',
        register: 'E8',
        min: 0,
        max: 200000,
        step: 1000,
        help: "With cell lighting on a budget: how many cells are lit a frame (ours 20,000; id Tech 8 lights about that many). Fewer is cheaper and slower to settle. Not in Capsaicin.",
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'cellChangeReset',
        group: 'world-cache',
        label: 'Cells forget history where the scene changed',
        urlName: 'rccellreset',
        register: 'E9',
        capsaicin: false,
        help: "Where a light or object changed, a world-cache cell forgets its 16-sample average once per change (and again every 8 frames while the change goes on), so a doubled sun shows in about 2 frames instead of about 37 (ours, on). Capsaicin has no such reset (off).",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'cellChangeResetHistory',
        group: 'world-cache',
        label: 'Samples kept when a cell forgets',
        urlName: 'rccellkeep',
        register: 'E9',
        min: 0,
        max: 15,
        step: 1,
        help: "When a cell forgets its history after a change: how many samples of the old average it keeps beside the new frame's (ours 0, follows at once; 3 is a little calmer and takes about 19 frames for a doubled sun). Not in Capsaicin.",
    }),
    toggle<CompareChoiceToggle>({
        kind: 'choice',
        setting: 'neuralCache',
        group: 'world-cache',
        label: 'Neural tail (every bounce)',
        urlName: 'rcneural',
        register: 'E11',
        choices: [
            { value: 'off', label: "Off: two bounces (Capsaicin's)" },
            { value: 'tail', label: 'A small network answers the later bounces' },
        ],
        capsaicin: 'off',
        help: "With GI-1.2's own second-bounce rays only (the cache alone, or the second bounce from GI-1.2's rays): a small network (pmndrs/pathtracer's, from its private package) learns the cells' indirect light and answers the bounces past the second, so the cache carries every bounce. Off (the default, Capsaicin's) is two bounces. With DDGI giving the second bounce, or without the package, it stays off and warns once.",
    }),

    // ---- The GI denoiser (RC4.3, register D65-D66) ----
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'denoiser',
        group: 'denoiser',
        label: 'GI denoiser',
        urlName: 'rcdenoise',
        register: 'D65',
        capsaicin: true,
        help: "GI-1.0's image-space denoiser (ReprojectGI and FilterGI, from pmndrs/denoiser) on the interpolated irradiance, which the picture then shows. On, as Capsaicin always runs it; off shows the raw interpolation.",
    }),
    toggle<CompareChoiceToggle>({
        kind: 'choice',
        setting: 'denoiserHistoryClamp',
        group: 'denoiser',
        label: 'Denoiser history clamp',
        urlName: 'rcdnclamp',
        register: 'D66',
        choices: [
            { value: 'neighbourhood', label: 'Clamp to a short history (ours)' },
            { value: 'none', label: "No clamp (Capsaicin's)" },
        ],
        capsaicin: 'none',
        help: "Whether the denoiser's long history is held near a short one of about 4 frames, so a lighting change shows within a few frames (ours). Capsaicin uses its history as it is, which takes 45-160 frames to follow a change (none).",
    }),
    toggle<CompareNumberToggle>({
        kind: 'number',
        setting: 'denoiserHistoryClip',
        group: 'denoiser',
        label: 'Clamp width (standard errors)',
        urlName: 'rcdnclip',
        register: 'D66',
        min: 0.5,
        max: 8,
        step: 0.25,
        help: 'With the clamp on: how far the long history may sit from the short one, in standard errors (ours 2). Smaller follows changes sooner and flickers more. Capsaicin has no clamp, so no value of its own.',
    }),
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'denoiserChangeReset',
        group: 'denoiser',
        label: 'Denoiser: short history where the scene changed',
        urlName: 'rcdnreset',
        register: 'D66',
        capsaicin: false,
        help: "Where a light or object changed, the denoiser keeps at most a few frames of history there for 8 frames, so the change shows sooner (ours, on). Capsaicin's denoiser hears of no change (off).",
    }),

    // ---- Experiments (not Capsaicin's changes; for seeing what history does) ----
    toggle<CompareBooleanToggle>({
        kind: 'boolean',
        setting: 'probeHistory',
        group: 'experiments',
        label: 'Screen probes keep history',
        urlName: 'rchistory',
        capsaicin: true,
        help: 'Off drops the screen probes\' history before every frame, so each frame starts from scratch: the noisy picture all the history work improves on. On, as Capsaicin.',
    }),
];

/** What a variant needs from the host besides the radiance cache itself (P6.5). */
export interface VariantNeeds {
    /** A DDGI volume stacked under the radiance cache (`setFallback`), updating probes every frame. */
    readonly ddgiVolume: boolean;
    /** The device's `shader-f16` feature. */
    readonly shaderF16: boolean;
    /** An optional sibling package the variant runs from, or null. */
    readonly sibling: string | null;
}

/**
 * A named variant of the radiance cache's world cache (P6.5): the compare values that make it,
 * so the viewer's compare panel can switch whole variants at once and the GPU pages can run
 * them by name (`VITE_RC_COMPARE=variant=sharc`). Values not named stay at ours.
 */
export interface RadianceCacheVariantPreset {
    /** Its name in links and on GPU pages (lowercase letters, digits and dashes). */
    readonly id: string;
    /** A few words for the panel. */
    readonly label: string;
    /** One plain line: what it is, and what it costs or needs. */
    readonly help: string;
    /** The register rows it is built from (docs/design/radiance-cache.md). */
    readonly register: readonly string[];
    /** The compare values that differ from ours. */
    readonly values: CompareValues;
    readonly needs: VariantNeeds;
}

/**
 * The world-cache variants P6.5 compares, in the comparison's order. DECISIONS 2026-11-23: they
 * stay variants until Dennis picks the default from the comparison (docs/design/radiance-cache-variants.md).
 * The first two differ only in whether a DDGI volume runs under the cache; `gi12-ddgi` is what
 * ships today (all values ours).
 */
export const RADIANCE_CACHE_VARIANT_PRESETS: readonly RadianceCacheVariantPreset[] = [
    {
        id: 'gi12-rays',
        label: "GI-1.2 grid, its own second-bounce rays (Capsaicin's)",
        help: "The hash grid alone, its second bounce from GI-1.2's own rays, as Capsaicin runs it. Needs nothing from the host; two bounces, so closed rooms read dark.",
        register: ['E10'],
        values: { 'host.cacheAlone': true, secondBounce: 'gi1-rays' },
        needs: { ddgiVolume: false, shaderF16: false, sibling: null },
    },
    {
        id: 'gi12-ddgi',
        label: 'GI-1.2 grid over DDGI (ours, what ships)',
        help: 'The hash grid with a DDGI volume under it that gives the second bounce and every bounce after (256 probes a frame). Needs a volume around the scene; DDGI costs its own GPU time and memory.',
        register: ['E10'],
        values: {},
        needs: { ddgiVolume: true, shaderF16: false, sibling: null },
    },
    {
        id: 'sharc',
        label: 'SHaRC cache alone',
        help: "pmndrs/pathtracer's SHaRC-style hash cache in place of the GI-1.2 grid, with no DDGI. Every bounce through its own cache; smaller, and needs nothing from the host.",
        register: ['E1'],
        values: { worldCache: 'sharc', 'host.cacheAlone': true },
        needs: { ddgiVolume: false, shaderF16: false, sibling: null },
    },
    {
        id: 'sharc-ddgi',
        label: 'SHaRC cache over DDGI',
        help: "The SHaRC cache with the viewer's DDGI volume still stacked under it. SHaRC doesn't read the volume (only the GI-1.2 grid takes its second bounce from it), so this shows SHaRC's picture at the volume's cost.",
        register: ['E1', 'E10'],
        values: { worldCache: 'sharc' },
        needs: { ddgiVolume: true, shaderF16: false, sibling: null },
    },
    {
        id: 'budgeted',
        label: 'Budgeted cells (id Tech 8), over DDGI',
        help: 'The GI-1.2 grid over DDGI, its cells lit on a budget (20,000 a frame, one light sample each, new and changed cells first) instead of at every hit: fewer shadow rays, older cells.',
        register: ['E8', 'E10'],
        values: { cellShading: 'budgeted' },
        needs: { ddgiVolume: true, shaderF16: false, sibling: null },
    },
    {
        id: 'neural-tail',
        label: 'GI-1.2 grid with the neural tail',
        help: "The hash grid alone with GI-1.2's rays, and a small network that learns the cells' indirect light and answers the bounces past the second. Needs shader-f16 and the optional @pathtracer/mlp package; without either it runs as the GI-1.2 grid alone.",
        register: ['E11'],
        values: { 'host.cacheAlone': true, secondBounce: 'gi1-rays', neuralCache: 'tail' },
        needs: { ddgiVolume: false, shaderF16: true, sibling: '@pathtracer/mlp' },
    },
];

/** The variant preset of this id, or undefined. */
export function variantPreset(id: string): RadianceCacheVariantPreset | undefined {
    return RADIANCE_CACHE_VARIANT_PRESETS.find((preset) => preset.id === id);
}

/**
 * The variant preset `values` equal (every toggle at the preset's value or, where it names none,
 * at ours), or undefined for a mix no preset makes.
 */
export function variantOfValues(values: CompareValues): RadianceCacheVariantPreset | undefined {
    const given = values as Record<string, CompareValue | undefined>;
    return RADIANCE_CACHE_VARIANT_PRESETS.find((preset) => {
        const wanted = preset.values as Record<string, CompareValue | undefined>;
        return RADIANCE_CACHE_COMPARE_TOGGLES.every((entry) => (given[entry.setting] ?? entry.defaultValue) === (wanted[entry.setting] ?? entry.defaultValue));
    });
}

/** The toggle for a setting, or undefined. */
export function compareToggle(setting: string): CompareToggle | undefined {
    return RADIANCE_CACHE_COMPARE_TOGGLES.find((entry) => entry.setting === setting);
}

/** The toggles of one group, in order. */
export function compareTogglesIn(group: CompareGroup): CompareToggle[] {
    return RADIANCE_CACHE_COMPARE_TOGGLES.filter((entry) => entry.group === group);
}

/** Whether `value` equals the toggle's Capsaicin value. */
export function isCapsaicinValue(entry: CompareToggle, value: unknown): boolean {
    return value === entry.capsaicin;
}

/** Every toggle Capsaicin has at its Capsaicin value (the panel's "All Capsaicin"; the others at ours), as values that differ from the defaults. */
export function capsaicinCompareValues(): CompareValues {
    const values: Record<string, unknown> = {};
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) if (entry.capsaicin !== undefined && entry.capsaicin !== entry.defaultValue) values[entry.setting] = entry.capsaicin;
    return values as CompareValues;
}

/**
 * A value read from text (a link), or undefined when the text isn't one the
 * toggle takes. Booleans are `1` and `0`; numbers are clamped to the range and
 * put on the step; choices must be one of theirs.
 */
export function parseCompareValue(entry: CompareToggle, text: string | null): boolean | number | string | undefined {
    if (text === null) return undefined;
    switch (entry.kind) {
        case 'boolean':
            return text === '1' ? true : text === '0' ? false : undefined;
        case 'choice':
            return entry.choices.some((choice) => choice.value === text) ? text : undefined;
        case 'number': {
            if (text.trim() === '') return undefined;
            const value = Number(text);
            if (!Number.isFinite(value)) return undefined;
            const stepped = Math.round((value - entry.min) / entry.step) * entry.step + entry.min;
            // Rounded to the step's decimals, so 0.1 + 0.025 doesn't print as 0.12500000000000003.
            const decimals = (String(entry.step).split('.')[1] ?? '').length;
            return Number(Math.min(entry.max, Math.max(entry.min, stepped)).toFixed(decimals));
        }
    }
}

/** A value as a link writes it (the inverse of `parseCompareValue`). */
export function formatCompareValue(entry: CompareToggle, value: boolean | number | string): string {
    if (entry.kind === 'boolean') return value ? '1' : '0';
    return String(value);
}

/** The compare values stored in a link's parameters (only those that differ from the defaults). */
export function readCompareValues(params: { get(name: string): string | null }): CompareValues {
    const values: Record<string, unknown> = {};
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
        const value = parseCompareValue(entry, params.get(entry.urlName));
        if (value !== undefined && value !== entry.defaultValue) values[entry.setting] = value;
    }
    return values as CompareValues;
}

/** Writes the values that differ from the defaults into a link's parameters. */
export function writeCompareValues(values: CompareValues, params: { set(name: string, value: string): void }): void {
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
        const value = (values as Record<string, boolean | number | string | undefined>)[entry.setting];
        if (value !== undefined && value !== entry.defaultValue) params.set(entry.urlName, formatCompareValue(entry, value));
    }
}

/** The link names the toggles use, so a link reader can tell them from scene settings. */
export const COMPARE_URL_NAMES: readonly string[] = RADIANCE_CACHE_COMPARE_TOGGLES.map((entry) => entry.urlName);

/**
 * Puts the compare values into a technique's settings: each toggle's value,
 * or its default where `values` has none. Applied live (the technique reads
 * them every frame and resets its history where a rule changed).
 */
export function applyCompareValues(settings: RadianceCacheSettings, values: CompareValues): void {
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
        const value = (values as Record<string, CompareValue | undefined>)[entry.setting];
        setCompareSetting(settings, entry.setting, value === undefined ? entry.defaultValue : value);
    }
}

/** Sets only the toggles `values` names (a GPU page's configuration over its own settings). */
export function setCompareValues(settings: RadianceCacheSettings, values: CompareValues): void {
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
        const value = (values as Record<string, CompareValue | undefined>)[entry.setting];
        if (value !== undefined) setCompareSetting(settings, entry.setting, value);
    }
}

/**
 * Compare values from text such as `probeBlend=capsaicin-hysteresis,rccap=1`
 * (setting names or link names, comma- or ampersand-separated): how the GPU
 * pages take a configuration from `VITE_RC_COMPARE`. Throws on a name or value
 * it doesn't know, so a typo can't run the defaults unnoticed.
 */
export function compareValuesFromText(text: string | undefined): CompareValues {
    const values: Record<string, unknown> = {};
    if (!text || text.trim() === '') return values as CompareValues;
    for (const part of text.split(/[,&]/)) {
        const trimmed = part.trim();
        if (trimmed === '') continue;
        const [name, raw = ''] = trimmed.split('=');
        if (name === 'capsaicin' || name === 'all-capsaicin') {
            Object.assign(values, capsaicinCompareValues());
            continue;
        }
        if (name === 'variant') {
            // P6.5: a whole variant (its values; later names in the text still apply over it).
            const preset = variantPreset(raw);
            if (!preset) throw new Error(`@pmndrs/gi: no radiance-cache variant named "${raw}" (${RADIANCE_CACHE_VARIANT_PRESETS.map((entry) => entry.id).join(', ')})`);
            Object.assign(values, preset.values);
            continue;
        }
        const entry = RADIANCE_CACHE_COMPARE_TOGGLES.find((candidate) => candidate.setting === name || candidate.urlName === name);
        if (!entry) throw new Error(`@pmndrs/gi: no compare toggle named "${name}"`);
        const text = entry.kind === 'boolean' && (raw === 'true' || raw === 'false') ? (raw === 'true' ? '1' : '0') : raw;
        const value = parseCompareValue(entry, text);
        if (value === undefined) throw new Error(`@pmndrs/gi: "${raw}" is not a value of the compare toggle ${entry.setting}`);
        values[entry.setting] = value;
    }
    return values as CompareValues;
}
