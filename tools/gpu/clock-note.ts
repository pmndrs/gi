/**
 * The NVIDIA GPU's clock during a run, and how it becomes CSV rows.
 *
 * Why: the RTX 3060's clock was found locked at 1200 MHz (max 2100) after a day of timings
 * (docs/NOTES.md). `node tools/reference/bench-lock.ts run` now samples `nvidia-smi` about
 * once a second while its command runs and keeps a summary in a small JSON file (rewritten
 * after every sample, so a tool running inside the command can already read it). The
 * to-csv scripts and tools/native/bench.ps1 read that file and add `note/sm_clock_mhz` rows.
 *
 * The file is named by `--clock-file <path>` on a to-csv script, or by the DDGI_CLOCK_FILE
 * environment variable, which `bench-lock run` sets for its command. Without either, no
 * note is written: a leftover file from an earlier run is never picked up by accident.
 */

import { existsSync, readFileSync } from 'node:fs';

/** `bench-lock run` gives its command this variable: the file its clock summary is written to. */
export const CLOCK_FILE_ENV = 'DDGI_CLOCK_FILE';

/**
 * The query that `bench-lock run` samples; one CSV line per GPU. Temperature and the active
 * clock-throttle reasons (a hex bit mask) say WHY a clock is low; the three clock fields alone
 * (older callers, tests) still parse.
 */
export const NVIDIA_SMI_CLOCK_ARGS = [
    '--query-gpu=clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,clocks_event_reasons.active',
    '--format=csv,noheader,nounits',
];

/** Bits of `clocks_event_reasons.active` (nvidia-smi -q -d PERFORMANCE shows the same names). */
const REASON_SW_POWER_CAP = 0x4;
const REASON_HW_POWER_BRAKE = 0x80;
const REASON_SW_THERMAL = 0x20;
const REASON_HW_THERMAL = 0x40;

/** A sample counts as "under load" at this GPU utilization or more; idle samples sit at the desktop's clock. */
export const LOADED_UTILIZATION_PERCENT = 70;

/** Fewer loaded samples than this and the summary falls back to all samples. */
const MIN_LOADED_SAMPLES = 3;

/**
 * The RTX 3060 Laptop's clock below which a run counts as clocked down (`note/clock_low,1`).
 * Its maximum is 2100 MHz and a cool card under load runs at 1400 to 2000 MHz, so a MEDIAN under
 * 1500 MHz over the loaded samples means the clock was held down (heat, the power limit, a low
 * power state or a clock lock) and the milliseconds are not a full-speed card's. Measured: a
 * sustained GPU-bound run on this laptop reaches 2000 MHz in seconds, then heats to the GPU's
 * 87 C target and settles at 450 to 600 MHz. Only the RTX is judged: callers write the note on
 * NVIDIA rows only.
 */
export const LOW_CLOCK_MHZ = 1500;

/**
 * A pass/fail timing run must start with the RTX below this temperature (DECISIONS, 2026-11-05: "how
 * pass/fail timing is taken on the laptop"): a sustained load heats it to its 87 C target in about
 * 20 s, and a card that starts warm gets there sooner. `bench-lock run` waits for it by default
 * (`--cool-below`), and a run that started at or above it gets `note/start_hot,1`.
 */
export const START_TEMP_LIMIT_C = 70;

/**
 * Whether the run's clock was low. Judged only from at least MIN_LOADED_SAMPLES samples under
 * load: the fallback to all samples (short pages, latency-bound pages, the minimized native
 * harness) includes idle seconds and would read low for no reason, so those runs get no verdict.
 */
export function isClockLow(summary: ClockSummary): boolean {
    if (summary.smMhzLoaded === null || summary.loadedSamples < MIN_LOADED_SAMPLES) return false;
    return summary.smMhzLoaded.median < LOW_CLOCK_MHZ;
}

/** A plain-language warning for a low-clock run, or null. Printed at the end of `bench-lock run` and by the to-csv scripts. */
export function clockLowWarning(summary: ClockSummary): string | null {
    if (!isClockLow(summary) || summary.smMhzLoaded === null) return null;
    return (
        `WARNING: the GPU clock was LOW under load: SM median ${summary.smMhzLoaded.median} MHz over ${summary.loadedSamples} loaded samples ` +
        `(below ${LOW_CLOCK_MHZ} MHz; a cool card reaches 1400 to 2000). ${describeClockCause(summary)}` +
        'These milliseconds are slow-clock numbers: CSV rows get note/clock_low,1. ' +
        'Let the GPU cool (under about 70 C) before timing, keep runs short, and close other GPU programs ' +
        '(nvidia-smi --query-compute-apps=pid,name --format=csv; nvidia-smi -q -d PERFORMANCE).'
    );
}

/** What nvidia-smi said held the loaded samples down, or an empty string when it did not say. */
function describeClockCause(summary: ClockSummary): string {
    if (summary.thermalLimitedShare === undefined || summary.powerCappedShare === undefined) return '';
    const percent = (share: number) => `${Math.round(share * 100)}%`;
    const temperature = summary.temperatureCLoaded ? `, GPU temperature ${summary.temperatureCLoaded.median} C (max ${summary.temperatureCLoaded.max})` : '';
    return `nvidia-smi: thermal throttling in ${percent(summary.thermalLimitedShare)} and the power limit in ${percent(summary.powerCappedShare)} of the loaded samples${temperature}. `;
}

export interface ClockSample {
    smMhz: number;
    memMhz: number;
    utilizationPercent: number;
    /** Absent when the line had only the three clock fields. */
    temperatureC?: number;
    /** The `clocks_event_reasons.active` bit mask; absent like temperatureC. */
    throttleReasons?: number;
}

/** How the command under `bench-lock run` started: the GPU temperature, and the cool-down wait before it. */
export interface StartConditions {
    /** The GPU temperature when the command started (after any cool-down wait). */
    temperatureC: number;
    /** The temperature the wait aimed to get under; null when there was no wait (`--no-cool-wait`, the CPU lock). */
    coolBelowC: number | null;
    /** How long the wait took. */
    waitedSeconds: number;
    /** The wait hit its time cap and the command started anyway, still at or above `coolBelowC`. */
    capped: boolean;
}

export interface ClockStats {
    min: number;
    median: number;
    max: number;
}

export interface ClockSummary {
    /** ISO time of the last update. */
    updatedAt: string;
    samples: number;
    /** Samples at or above LOADED_UTILIZATION_PERCENT. */
    loadedSamples: number;
    loadedThresholdPercent: number;
    /** Over every sample (idle ones included). */
    smMhz: ClockStats;
    memMhz: ClockStats;
    /** Over loaded samples only; null if there were none. */
    smMhzLoaded: ClockStats | null;
    memMhzLoaded: ClockStats | null;
    utilizationPercent: ClockStats;
    /** Over loaded samples that carry a temperature; absent in summaries from older runs. */
    temperatureCLoaded?: ClockStats | null;
    /** Share (0 to 1) of loaded samples where nvidia-smi named a thermal limit as an active reason. */
    thermalLimitedShare?: number;
    /** The same for the power limit (software cap or hardware power brake). */
    powerCappedShare?: number;
    /** How the command started; absent in summaries from older runs, and when nvidia-smi gave no temperature. */
    start?: StartConditions;
}

/** The first GPU's line of the `NVIDIA_SMI_CLOCK_ARGS` query, or null if it is not three or five numbers. */
export function parseClockLine(text: string): ClockSample | null {
    const line = text.split(/\r?\n/).find((candidate) => candidate.trim() !== '');
    if (!line) return null;
    const fields = line.split(',').map((field) => Number(field.trim()));
    // Number() reads the reasons mask ("0x0000000000000020") as hex.
    if ((fields.length !== 3 && fields.length !== 5) || !fields.every((value) => Number.isFinite(value))) return null;
    const sample: ClockSample = { smMhz: fields[0], memMhz: fields[1], utilizationPercent: fields[2] };
    if (fields.length === 5) {
        sample.temperatureC = fields[3];
        sample.throttleReasons = fields[4];
    }
    return sample;
}

function stats(values: readonly number[]): ClockStats {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    const median = sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
    return { min: sorted[0], median, max: sorted[sorted.length - 1] };
}

export function summarizeClockSamples(samples: readonly ClockSample[], now: Date = new Date()): ClockSummary | null {
    if (samples.length === 0) return null;
    const loaded = samples.filter((sample) => sample.utilizationPercent >= LOADED_UTILIZATION_PERCENT);
    const withReasons = loaded.filter((sample) => sample.throttleReasons !== undefined);
    const shareWith = (mask: number) => (withReasons.length === 0 ? 0 : withReasons.filter((sample) => ((sample.throttleReasons ?? 0) & mask) !== 0).length / withReasons.length);
    const temperatures = loaded.flatMap((sample) => (sample.temperatureC === undefined ? [] : [sample.temperatureC]));
    return {
        updatedAt: now.toISOString(),
        samples: samples.length,
        loadedSamples: loaded.length,
        loadedThresholdPercent: LOADED_UTILIZATION_PERCENT,
        smMhz: stats(samples.map((sample) => sample.smMhz)),
        memMhz: stats(samples.map((sample) => sample.memMhz)),
        smMhzLoaded: loaded.length > 0 ? stats(loaded.map((sample) => sample.smMhz)) : null,
        memMhzLoaded: loaded.length > 0 ? stats(loaded.map((sample) => sample.memMhz)) : null,
        utilizationPercent: stats(samples.map((sample) => sample.utilizationPercent)),
        temperatureCLoaded: temperatures.length > 0 ? stats(temperatures) : null,
        ...(withReasons.length > 0
            ? { thermalLimitedShare: shareWith(REASON_SW_THERMAL | REASON_HW_THERMAL), powerCappedShare: shareWith(REASON_SW_POWER_CAP | REASON_HW_POWER_BRAKE) }
            : {}),
    };
}

/** One plain-language line for the end of a run. */
export function describeClockSummary(summary: ClockSummary): string {
    const range = (value: ClockStats) => `min ${value.min} / median ${value.median} / max ${value.max}`;
    const loaded = summary.smMhzLoaded && summary.memMhzLoaded
        ? `; under load (${summary.loadedSamples} samples at ${summary.loadedThresholdPercent}%+ use): SM ${range(summary.smMhzLoaded)} MHz, memory ${range(summary.memMhzLoaded)} MHz`
        : '; no sample was under load';
    return `GPU clock over ${summary.samples} samples: SM ${range(summary.smMhz)} MHz, memory ${range(summary.memMhz)} MHz${loaded}`;
}

/** Reads a clock summary file; null if it does not exist or is not a summary. */
export function readClockSummary(path: string): ClockSummary | null {
    if (!existsSync(path)) return null;
    try {
        const value = JSON.parse(readFileSync(path, 'utf8')) as Partial<ClockSummary>;
        return value.smMhz && value.memMhz && typeof value.samples === 'number' ? (value as ClockSummary) : null;
    } catch {
        return null;
    }
}

/**
 * The note rows for a summary. The headline `note/sm_clock_mhz` is the median over the samples
 * under load (all samples if fewer than three were); min and max are over the same set.
 * `note/clock_low,1` is added when the loaded median is under LOW_CLOCK_MHZ (see isClockLow).
 * `note/start_temp_c` is the GPU temperature when the command started, and `note/start_hot,1` marks
 * a start at or above START_TEMP_LIMIT_C (see startedHot).
 */
export function clockNoteRows(summary: ClockSummary): Array<{ scene: string; metric: string; value: number }> {
    const useLoaded = summary.smMhzLoaded !== null && summary.memMhzLoaded !== null && summary.loadedSamples >= MIN_LOADED_SAMPLES;
    const sm = (useLoaded ? summary.smMhzLoaded : summary.smMhz) as ClockStats;
    const mem = (useLoaded ? summary.memMhzLoaded : summary.memMhz) as ClockStats;
    return [
        { scene: 'all', metric: 'note/sm_clock_mhz', value: sm.median },
        { scene: 'all', metric: 'note/sm_clock_min_mhz', value: sm.min },
        { scene: 'all', metric: 'note/sm_clock_max_mhz', value: sm.max },
        { scene: 'all', metric: 'note/mem_clock_mhz', value: mem.median },
        ...(isClockLow(summary) ? [{ scene: 'all', metric: 'note/clock_low', value: 1 }] : []),
        ...(summary.start ? [{ scene: 'all', metric: 'note/start_temp_c', value: summary.start.temperatureC }] : []),
        ...(startedHot(summary) ? [{ scene: 'all', metric: 'note/start_hot', value: 1 }] : []),
    ];
}

/** Whether the run started at or above START_TEMP_LIMIT_C, so it can't pass or fail a timing target. */
export function startedHot(summary: ClockSummary): boolean {
    return summary.start !== undefined && summary.start.temperatureC >= START_TEMP_LIMIT_C;
}

/** A plain-language warning for a run that started hot, or null. */
export function startHotWarning(summary: ClockSummary): string | null {
    if (!startedHot(summary) || summary.start === undefined) return null;
    const why = summary.start.capped
        ? `the cool-down wait gave up after ${Math.round(summary.start.waitedSeconds / 60)} min`
        : summary.start.coolBelowC === null
            ? 'there was no cool-down wait'
            : `the wait only aimed for ${summary.start.coolBelowC} C`;
    return (
        `WARNING: the GPU was at ${summary.start.temperatureC} C when the run started (${why}). ` +
        `Pass/fail timing starts below ${START_TEMP_LIMIT_C} C, so CSV rows get note/start_hot,1.`
    );
}

/** Only the NVIDIA GPU is sampled, so the note belongs on NVIDIA rows only. */
export function isNvidiaDeviceName(device: string): boolean {
    return /nvidia|geforce|rtx|gtx/i.test(device);
}

/**
 * Note rows for a to-csv script: from `--clock-file`, else DDGI_CLOCK_FILE, else none.
 * Warns (and returns none) if a named file is missing.
 */
export function clockNoteRowsFromOption(clockFile: string | undefined, env: NodeJS.ProcessEnv = process.env): Array<{ scene: string; metric: string; value: number }> {
    const path = clockFile ?? env[CLOCK_FILE_ENV];
    if (!path) return [];
    const summary = readClockSummary(path);
    if (!summary) {
        console.error(`no clock summary in ${path}; no note/sm_clock_mhz rows written`);
        return [];
    }
    const warning = clockLowWarning(summary);
    if (warning) console.error(warning);
    const hot = startHotWarning(summary);
    if (hot) console.error(hot);
    return clockNoteRows(summary);
}
