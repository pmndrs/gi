/**
 * The NVIDIA GPU's clock during a run, whether the run counts as valid timing, and how both
 * become CSV rows.
 *
 * Why: the RTX 3060 Laptop throttles. A sustained GPU-bound load heats it to its 87 C target in
 * about 20 s, and then it holds about 500 MHz (docs/NOTES.md). `node tools/reference/bench-lock.ts
 * run` waits for the GPU to cool below 70 C, then samples `nvidia-smi` every 200 ms while its
 * command runs and keeps a summary in a small JSON file (rewritten after every sample, so a tool
 * running inside the command can already read it). The to-csv scripts and tools/native/bench.ps1
 * read that file and add `note/...` rows.
 *
 * The verdict is "the clock rule for valid timing" (docs/DECISIONS.md, 2026-11-05): see
 * judgeTiming. It replaced the `note/clock_low` (median under 1,500 MHz) flag the same day.
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

/** Fewer loaded samples than this and the `note/sm_clock_mhz` headline falls back to all samples. */
const MIN_LOADED_SAMPLES_FOR_HEADLINE = 3;

/**
 * A timing run must start with the RTX below this temperature (DECISIONS, 2026-11-05): a card that
 * starts warm reaches its thermal limit sooner. `bench-lock run` waits for it by default
 * (`--cool-below`).
 */
export const START_TEMP_LIMIT_C = 70;

/**
 * A run is valid only if fewer than this share of its loaded samples show a thermal-throttle
 * reason. The state to exclude is the thermal collapse to about 500 MHz; from a cool start the card
 * is power-limited instead, which is its normal state and doesn't count against a run.
 */
export const THERMAL_SHARE_LIMIT = 0.05;

/**
 * Fewer loaded samples than this: no verdict. At 200 ms sampling that is a second under load; with
 * fewer, one throttled sample is already 20% or more and the share means nothing.
 */
export const MIN_LOADED_SAMPLES_FOR_VERDICT = 5;

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
    /** How often nvidia-smi was sampled; absent in summaries from before 200 ms sampling (those were 1 s). */
    sampleIntervalMs?: number;
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

export type TimingVerdict =
    | { kind: 'valid'; reason: string }
    | { kind: 'invalid'; reason: string }
    | { kind: 'no-verdict'; reason: string };

/**
 * The clock rule for valid timing (DECISIONS, 2026-11-05):
 *
 *   invalid     the run started at 70 C or more (whatever else happened, it doesn't count), or
 *               5% or more of its loaded samples show a thermal-throttle reason;
 *   no verdict  no start temperature or no throttle reasons were recorded, or fewer than
 *               MIN_LOADED_SAMPLES_FOR_VERDICT samples were under load;
 *   valid       otherwise: a cool start, and under 5% thermal samples.
 *
 * There is no MHz bar: a cool card is power-limited and its median lands anywhere from about
 * 1,000 to 1,700 MHz depending on the load, which is the card's real speed, not a fault.
 */
export function judgeTiming(summary: ClockSummary): TimingVerdict {
    const { start } = summary;
    if (start === undefined) return { kind: 'no-verdict', reason: 'no start temperature was recorded' };
    if (start.temperatureC >= START_TEMP_LIMIT_C) {
        return { kind: 'invalid', reason: `hot start: the GPU was at ${start.temperatureC} C (${START_TEMP_LIMIT_C} C or more)` };
    }
    if (summary.loadedSamples < MIN_LOADED_SAMPLES_FOR_VERDICT) {
        return {
            kind: 'no-verdict',
            reason: `only ${summary.loadedSamples} samples under load (${MIN_LOADED_SAMPLES_FOR_VERDICT} needed)`,
        };
    }
    if (summary.thermalLimitedShare === undefined) return { kind: 'no-verdict', reason: 'nvidia-smi gave no throttle reasons' };
    const thermal = `thermal throttling in ${formatPercent(summary.thermalLimitedShare)} of ${summary.loadedSamples} loaded samples`;
    if (summary.thermalLimitedShare >= THERMAL_SHARE_LIMIT) {
        return { kind: 'invalid', reason: `${thermal} (${formatPercent(THERMAL_SHARE_LIMIT)} or more)` };
    }
    return { kind: 'valid', reason: `started at ${start.temperatureC} C, ${thermal}` };
}

function formatPercent(share: number): string {
    return `${Math.round(share * 1000) / 10}%`;
}

/** The plain-language verdict line printed at the end of `bench-lock run` and by the to-csv scripts. */
export function describeTimingVerdict(summary: ClockSummary): string {
    const verdict = judgeTiming(summary);
    const label = verdict.kind === 'valid' ? 'VALID' : verdict.kind === 'invalid' ? 'INVALID' : 'NO VERDICT';
    const clock = summary.smMhzLoaded ? ` SM clock median under load ${summary.smMhzLoaded.median} MHz (information only).` : '';
    const power = summary.powerCappedShare === undefined ? '' : ` Power limit in ${formatPercent(summary.powerCappedShare)} of loaded samples.`;
    const advice =
        verdict.kind === 'invalid'
            ? ' These milliseconds can\'t pass or fail a target: let the GPU cool, keep the timed section to 10-15 s, and time again.'
            : '';
    return `Timing ${label}: ${verdict.reason}.${clock}${power}${advice}`;
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
 * The note rows for a summary:
 *
 *   note/sm_clock_mhz (+ min, max, mem)  the median over the samples under load (all samples if
 *                                        fewer than three were); information only.
 *   note/start_temp_c                    the GPU temperature when the command started.
 *   note/thermal_share                   the share (0 to 1) of loaded samples with a thermal-throttle reason.
 *   note/timing_valid,1|0                the clock rule's verdict (judgeTiming); 0 also for no verdict,
 *   note/timing_no_verdict,1             which then gets this row as well.
 */
export function clockNoteRows(summary: ClockSummary): Array<{ scene: string; metric: string; value: number }> {
    const useLoaded = summary.smMhzLoaded !== null && summary.memMhzLoaded !== null && summary.loadedSamples >= MIN_LOADED_SAMPLES_FOR_HEADLINE;
    const sm = (useLoaded ? summary.smMhzLoaded : summary.smMhz) as ClockStats;
    const mem = (useLoaded ? summary.memMhzLoaded : summary.memMhz) as ClockStats;
    const verdict = judgeTiming(summary);
    return [
        { scene: 'all', metric: 'note/sm_clock_mhz', value: sm.median },
        { scene: 'all', metric: 'note/sm_clock_min_mhz', value: sm.min },
        { scene: 'all', metric: 'note/sm_clock_max_mhz', value: sm.max },
        { scene: 'all', metric: 'note/mem_clock_mhz', value: mem.median },
        ...(summary.start ? [{ scene: 'all', metric: 'note/start_temp_c', value: summary.start.temperatureC }] : []),
        ...(summary.thermalLimitedShare === undefined ? [] : [{ scene: 'all', metric: 'note/thermal_share', value: Math.round(summary.thermalLimitedShare * 1000) / 1000 }]),
        { scene: 'all', metric: 'note/timing_valid', value: verdict.kind === 'valid' ? 1 : 0 },
        ...(verdict.kind === 'no-verdict' ? [{ scene: 'all', metric: 'note/timing_no_verdict', value: 1 }] : []),
    ];
}

/** Only the NVIDIA GPU is sampled, so the note belongs on NVIDIA rows only. */
export function isNvidiaDeviceName(device: string): boolean {
    return /nvidia|geforce|rtx|gtx/i.test(device);
}

/**
 * Note rows for a to-csv script: from `--clock-file`, else DDGI_CLOCK_FILE, else none. Prints the
 * timing verdict. Warns (and returns none) if a named file is missing.
 */
export function clockNoteRowsFromOption(clockFile: string | undefined, env: NodeJS.ProcessEnv = process.env): Array<{ scene: string; metric: string; value: number }> {
    const path = clockFile ?? env[CLOCK_FILE_ENV];
    if (!path) return [];
    const summary = readClockSummary(path);
    if (!summary) {
        console.error(`no clock summary in ${path}; no note/sm_clock_mhz rows written`);
        return [];
    }
    console.error(describeTimingVerdict(summary));
    return clockNoteRows(summary);
}
