/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the `ddgi-batch-cost` page into rows
 * for `bench/results/timing.csv` (columns as in bench/results/README.md).
 *
 *   node tools/gpu/pages/ddgi-batch-cost/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--browser <text>] [--clock-file <file>] [--set <n>] [--note <metric>=<n>] [--include-failed]
 *
 * Several files of the same GPU are repeated runs: `run1/`, `run2/`, ... plus
 * `median/` rows. Metric names start with `batch_cost/`, or with
 * `batch_cost/<inline|deferred>/` for runs that name their direct-light path
 * (T2.2 on; the rest of the name is the same):
 *
 * - `batch_cost/batch<n>/...`: fixed batches of n probes, each frame waited
 *   for (the GPU kept busy): median GPU ms of `ddgi/trace`, `ddgi/blend`,
 *   `ddgi/write`, `ddgi/census`, their update sum, its p95, and µs per probe;
 * - `batch_cost/fit_all/...` and `batch_cost/fit_upto512/...`: the update
 *   fitted as fixed ms + µs per probe, over all batches or those up to 512;
 * - `batch_cost/budget<ms>ms/...`: T1.9a's millisecond controller, frames
 *   at `budget.gpuMs` and 60 Hz (median, p05, p95, mean update ms; share of
 *   frames within ±10%; probes per frame). Old runs only;
 * - `batch_cost/v2_budget<ms>ms/...`: budget v2, GPU-bound (back to back):
 *   the calibrated batch's update ms (median, p05, p95), probes per frame,
 *   the floor and the full-sweep time at 60 Hz;
 * - `batch_cost/v2_rate<hz>hz/...`: budget v2 at 30, 60 and 120 Hz: probes
 *   per frame (median, min, max), the floor, frames late, stepped down or
 *   below the floor, the sweep time, and the update ms (light load,
 *   information only);
 * - `batch_cost/startup/...`: milliseconds to each start-up stage, and the
 *   slowest pipeline compiles;
 * - `batch_cost/breakdown/batch<n>/<variant>_ms`: `ddgi/trace` with parts of
 *   its shading taken out (main.ts, TRACE_VARIANTS);
 * - `batch_cost/steady/batch<n>/<raf|waited>/...`: a fixed batch at 60 Hz or
 *   back to back: update ms, spike count and share (frames over 1.5 × the
 *   median + 0.2 ms), the spikes' extra ms (per pass), and their spacing in
 *   wall-clock ms (median, quartiles). Spikes spaced by another program's
 *   clock, not ours, are that program's GPU work.
 *
 * The browser column comes from the page (user agent plus client hints);
 * `--browser` overrides it (runs from before the page reported the full
 * version read "156.0.0.0").
 * `--set <n>` adds a `note/set,<n>` row (scene `all`) when a measurement was
 * repeated as a new `bench-lock run` set (bench/results/README.md).
 * `--clock-file` (or DDGI_CLOCK_FILE, which `bench-lock run` sets) adds
 * `note/sm_clock_mhz` and friends to the NVIDIA rows. Without --append the
 * rows are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from '../../clock-note.ts';
import { gitCommit } from '../../measured-commit.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from '../trace-bench/csv-rows.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';
const DEFAULT_CSV = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'timing.csv');
const BATCH_COST = 'batch_cost';

/** What the page reports per scene (main.ts); every section is optional. */
interface SceneResult {
    /** The scene's id and the direct-light path (T2.2 runs; older runs key results by scene alone). */
    scene?: string;
    directLight?: 'inline' | 'deferred';
    probes: number;
    startup?: Record<string, number>;
    /** T1.9a's millisecond controller (old runs). */
    budget?: Record<string, number | string>;
    /** Budget v2, GPU-bound. */
    budgetV2?: Record<string, number>;
    /** Budget v2 at host frame rates, by `<hz>hz`. */
    rates?: Record<string, Record<string, number>>;
    curve?: { sizes: Record<string, Record<string, number>> } & Record<string, unknown>;
    breakdown?: Record<string, Record<string, number>>;
    /** Fixed batches: per batch size, per cadence (raf, waited), the spike numbers. */
    steady?: Record<string, Record<string, unknown>>;
}

interface PageData {
    userAgent?: string;
    /** From the client hints; the user agent freezes it at 156.0.0.0. */
    browserVersion?: string | null;
    scenes: Record<string, SceneResult>;
}

/** Numeric fields only (traces and labels are for reading the JSON, not for the CSV). */
function numericRows(scene: string, prefix: string, record: Record<string, unknown>): MetricRow[] {
    return Object.entries(record).flatMap(([name, value]) => (typeof value === 'number' && Number.isFinite(value) ? [{ scene, metric: `${prefix}/${name}`, value }] : []));
}

export function batchCostRows(data: PageData): MetricRow[] {
    const rows: MetricRow[] = [];
    for (const [key, result] of Object.entries(data.scenes)) {
        const scene = result.scene ?? key;
        // A run that measured a direct-light path names it in every metric: batch_cost/<path>/...
        const metricPrefix = result.directLight ? `${BATCH_COST}/${result.directLight}` : BATCH_COST;
        rows.push({ scene, metric: `${metricPrefix}/probes_in_volume`, value: result.probes });
        if (result.startup) {
            // Pipeline labels carry the page's platform label ("batch cost <scene> ddgi/trace"); keep the kernel's own name.
            const startup = Object.fromEntries(Object.entries(result.startup).map(([name, value]) => [name.replace(`compile_ms/batch cost ${scene} `, 'compile_ms/'), value]));
            rows.push(...numericRows(scene, `${metricPrefix}/startup`, startup));
        }
        if (result.budget) {
            const budgetMs = result.budget.budget_ms;
            rows.push(...numericRows(scene, `${metricPrefix}/budget${budgetMs}ms`, result.budget));
        }
        if (result.budgetV2) {
            rows.push(...numericRows(scene, `${metricPrefix}/v2_budget${result.budgetV2.budget_ms}ms`, result.budgetV2));
        }
        if (result.rates) {
            for (const [rate, values] of Object.entries(result.rates)) rows.push(...numericRows(scene, `${metricPrefix}/v2_rate${rate}`, values));
        }
        if (result.curve) {
            for (const [size, values] of Object.entries(result.curve.sizes)) rows.push(...numericRows(scene, `${metricPrefix}/batch${size}`, values));
            const fit = result.curve as Record<string, unknown>;
            rows.push(
                ...numericRows(scene, `${metricPrefix}/fit_all`, { fixed_ms: fit.fit_all_fixed_ms, us_per_probe: fit.fit_all_us_per_probe }),
                ...numericRows(scene, `${metricPrefix}/fit_upto512`, { fixed_ms: fit.fit_small_fixed_ms, us_per_probe: fit.fit_small_us_per_probe }),
            );
        }
        if (result.steady) {
            for (const [size, cadences] of Object.entries(result.steady)) {
                for (const [cadence, values] of Object.entries(cadences)) {
                    if (typeof values === 'object' && values !== null) rows.push(...numericRows(scene, `${metricPrefix}/steady/batch${size}/${cadence}`, values as Record<string, unknown>));
                }
            }
        }
        if (result.breakdown) {
            for (const [size, values] of Object.entries(result.breakdown)) rows.push(...numericRows(scene, `${metricPrefix}/breakdown/batch${size}`, values));
        }
    }
    return rows;
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, commit: { type: 'string' }, browser: { type: 'string' }, 'clock-file': { type: 'string' }, set: { type: 'string' }, note: { type: 'string', multiple: true }, 'include-failed': { type: 'boolean' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tools/gpu/pages/ddgi-batch-cost/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--browser <text>] [--clock-file <file>] [--set <n>]');
        process.exit(2);
    }
    const clockNotes = clockNoteRowsFromOption(values['clock-file']);
    // `--note quiet=1` becomes the row `note/quiet,1` (scene `all`).
    const extraNotes: MetricRow[] = (values.note ?? []).map((note) => {
        const [metric, value] = note.split('=');
        if (!metric || value === undefined || !Number.isFinite(Number(value))) throw new Error(`--note needs <metric>=<number>, got "${note}"`);
        return { scene: 'all', metric: `note/${metric}`, value: Number(value) };
    });
    const setNote: MetricRow[] = [...(values.set === undefined ? [] : [{ scene: 'all', metric: 'note/set', value: Number(values.set) }]), ...extraNotes];
    let failedRuns = 0;
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][] }>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            if (page.name !== 'ddgi-batch-cost') continue;
            // The page fails when a budget isn't held, but its numbers are still the measurement: --include-failed keeps them
            // and marks the group with note/page_failed (the number of runs that failed).
            if (page.outcome !== 'pass' && values['include-failed']) failedRuns += 1;
            else if (page.outcome !== 'pass') {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const data = page.data as unknown as PageData;
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const browser = values.browser ?? (data.userAgent ? browserFromUserAgent(data.userAgent, data.browserVersion ?? null) : 'Chrome headless');
            const prefix = [summary.startedAt.slice(0, 10), commit, device, browser];
            const group = groups.get(device) ?? { prefix, runs: [] };
            group.runs.push(batchCostRows(data));
            groups.set(device, group);
        }
    }
    const lines: string[] = [];
    for (const { prefix, runs } of groups.values()) {
        const rows = [...(runs.length === 1 ? runs[0] : repeatedRunRows(runs)), ...setNote, ...(failedRuns > 0 ? [{ scene: 'all', metric: 'note/page_failed', value: failedRuns }] : []), ...(isNvidiaDeviceName(prefix[2]) ? clockNotes : [])];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(',')));
    }
    if (lines.length === 0) {
        console.error('no passing ddgi-batch-cost pages in those files');
        process.exit(1);
    }
    if (!values.append) {
        console.log([CSV_HEADER, ...lines].join('\n'));
        return;
    }
    if (!existsSync(DEFAULT_CSV)) await writeFile(DEFAULT_CSV, `${CSV_HEADER}\n`);
    await appendFile(DEFAULT_CSV, `${lines.join('\n')}\n`);
    console.error(`added ${lines.length} rows to ${DEFAULT_CSV}`);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
