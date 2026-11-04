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

/** The query that `bench-lock run` samples; one CSV line per GPU. */
export const NVIDIA_SMI_CLOCK_ARGS = ['--query-gpu=clocks.sm,clocks.mem,utilization.gpu', '--format=csv,noheader,nounits'];

/** A sample counts as "under load" at this GPU utilization or more; idle samples sit at the desktop's clock. */
export const LOADED_UTILIZATION_PERCENT = 70;

/** Fewer loaded samples than this and the summary falls back to all samples. */
const MIN_LOADED_SAMPLES = 3;

export interface ClockSample {
    smMhz: number;
    memMhz: number;
    utilizationPercent: number;
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
}

/** The first GPU's line of the `NVIDIA_SMI_CLOCK_ARGS` query, or null if it is not three numbers. */
export function parseClockLine(text: string): ClockSample | null {
    const line = text.split(/\r?\n/).find((candidate) => candidate.trim() !== '');
    if (!line) return null;
    const fields = line.split(',').map((field) => Number(field.trim()));
    if (fields.length !== 3 || !fields.every((value) => Number.isFinite(value))) return null;
    return { smMhz: fields[0], memMhz: fields[1], utilizationPercent: fields[2] };
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
    ];
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
    return clockNoteRows(summary);
}
