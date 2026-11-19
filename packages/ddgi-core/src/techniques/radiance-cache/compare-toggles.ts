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

import { DEFAULT_RADIANCE_CACHE_SETTINGS, SCREEN_HISTORY_SETTINGS, type RadianceCacheSettings } from './settings.ts';

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
type KeysOfType<T> = { [K in keyof RadianceCacheSettings]: RadianceCacheSettings[K] extends T ? K : never }[keyof RadianceCacheSettings];
export type CompareBooleanSetting = KeysOfType<boolean>;
export type CompareNumberSetting = KeysOfType<number>;
export type CompareChoiceSetting = KeysOfType<string>;
export type CompareSetting = CompareBooleanSetting | CompareNumberSetting | CompareChoiceSetting;

/** What every toggle says about itself. */
interface ToggleCommon<K extends CompareSetting> {
    /** The technique's setting it drives. */
    readonly setting: K;
    readonly group: CompareGroup;
    /** A few words for the panel. */
    readonly label: string;
    /** Its name in the viewer's links (lowercase letters and digits, unique). */
    readonly urlName: string;
    /** One plain line: what it does, and which value is Capsaicin's. */
    readonly help: string;
    /** The divergence-register row it undoes (docs/design/radiance-cache.md), if any. */
    readonly register?: string;
    /** The decided value (`DEFAULT_RADIANCE_CACHE_SETTINGS`). */
    readonly defaultValue: RadianceCacheSettings[K];
    /** The value that behaves as Capsaicin v1.3 does. */
    readonly capsaicin: RadianceCacheSettings[K];
    /** Changing it starts the screen probes' history over (`SCREEN_HISTORY_SETTINGS`). */
    readonly resetsHistory: boolean;
}

export interface CompareBooleanToggle extends ToggleCommon<CompareBooleanSetting> {
    readonly kind: 'boolean';
}

export interface CompareChoiceToggle extends ToggleCommon<CompareChoiceSetting> {
    readonly kind: 'choice';
    /** Every value it can take, with the panel's words for it. */
    readonly choices: readonly { readonly value: string; readonly label: string }[];
}

export interface CompareNumberToggle extends ToggleCommon<CompareNumberSetting> {
    readonly kind: 'number';
    readonly min: number;
    readonly max: number;
    readonly step: number;
}

export type CompareToggle = CompareBooleanToggle | CompareChoiceToggle | CompareNumberToggle;

/** The compare settings that differ from the defaults, by setting name (what a link stores). */
export type CompareValues = { -readonly [K in CompareSetting]?: RadianceCacheSettings[K] };

type ToggleInput<T extends CompareToggle> = Omit<T, 'defaultValue' | 'resetsHistory'>;

/** An entry, with its default and history reset taken from settings.ts so they can't drift. */
function toggle<T extends CompareToggle>(input: ToggleInput<T>): T {
    return {
        ...input,
        defaultValue: DEFAULT_RADIANCE_CACHE_SETTINGS[input.setting],
        resetsHistory: SCREEN_HISTORY_SETTINGS.includes(input.setting),
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

/** Every toggle at its Capsaicin value (the panel's "All Capsaicin"), as values that differ from the defaults. */
export function capsaicinCompareValues(): CompareValues {
    const values: Record<string, unknown> = {};
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) if (entry.capsaicin !== entry.defaultValue) values[entry.setting] = entry.capsaicin;
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
    const target = settings as unknown as Record<string, unknown>;
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
        const value = (values as Record<string, unknown>)[entry.setting];
        target[entry.setting] = value === undefined ? entry.defaultValue : value;
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
        const entry = RADIANCE_CACHE_COMPARE_TOGGLES.find((candidate) => candidate.setting === name || candidate.urlName === name);
        if (!entry) throw new Error(`@pmndrs/gi: no compare toggle named "${name}"`);
        const text = entry.kind === 'boolean' && (raw === 'true' || raw === 'false') ? (raw === 'true' ? '1' : '0') : raw;
        const value = parseCompareValue(entry, text);
        if (value === undefined) throw new Error(`@pmndrs/gi: "${raw}" is not a value of the compare toggle ${entry.setting}`);
        values[entry.setting] = value;
    }
    return values as CompareValues;
}
