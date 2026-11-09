/// <reference types="node" />
/**
 * Turns a test-runner `--json` file from the `platform-compute` page, run with
 * VITE_PLATFORM_COMPUTE_TIMING=1, into rows of `bench/results/compute.csv`
 * (RC1.3). Columns, the gpu-tester's set:
 *
 *   date, commit, device, browser, adapter, scene, camera, setting, metric, value, unit
 *
 * - `scene` is `synthetic` (seeded random items, no scene), `camera` `none`;
 * - `setting` is `<reduction>/<items>`: `subgroups/items1M`, `workgroup/items10M`
 *   (the fallback, forced on the same device);
 * - `metric` is `<utility>_ms_median`, `<utility>_ms_p95` or `<utility>_samples`
 *   for `scan` (into an output buffer), `compaction` and `args` (the
 *   indirect-arguments writer), each its own timed compute pass;
 * - the clock rule's notes (`--clock-file`, or DDGI_CLOCK_FILE, which
 *   `bench-lock run` sets) go on NVIDIA rows with setting `note`.
 *
 *   node tools/gpu/pages/platform-compute/to-csv.ts <runner.json> [--append] [--commit <hash>] [--clock-file <file>]
 *
 * Without --append the rows are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from '../../clock-note.ts';
import { gitCommit } from '../../measured-commit.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue } from '../trace-bench/csv-rows.ts';

const CSV_HEADER = 'date,commit,device,browser,adapter,scene,camera,setting,metric,value,unit';
const DEFAULT_CSV = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'compute.csv');

interface TimingStats {
    readonly medianMs: number;
    readonly p95Ms: number;
    readonly samples: number;
}

interface PageData {
    readonly timing?: Record<string, Record<string, Record<string, TimingStats>>>;
    readonly userAgent?: string;
    readonly browserVersion?: string | null;
}

export interface ComputeRow {
    readonly setting: string;
    readonly metric: string;
    readonly value: number;
    readonly unit: string;
}

/** The timing section's rows: per reduction, item count and utility, the median, p95 and sample count. */
export function computeTimingRows(timing: NonNullable<PageData['timing']>): ComputeRow[] {
    const rows: ComputeRow[] = [];
    for (const [reduction, sizes] of Object.entries(timing)) {
        for (const [items, utilities] of Object.entries(sizes)) {
            for (const [utility, stats] of Object.entries(utilities)) {
                const setting = `${reduction}/${items}`;
                rows.push({ setting, metric: `${utility}_ms_median`, value: stats.medianMs, unit: 'ms' });
                rows.push({ setting, metric: `${utility}_ms_p95`, value: stats.p95Ms, unit: 'ms' });
                rows.push({ setting, metric: `${utility}_samples`, value: stats.samples, unit: 'count' });
            }
        }
    }
    return rows;
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, commit: { type: 'string' }, 'clock-file': { type: 'string' } },
    });
    if (positionals.length !== 1) {
        console.error('Usage: node tools/gpu/pages/platform-compute/to-csv.ts <runner.json> [--append] [--commit <hash>] [--clock-file <file>]');
        process.exit(2);
    }
    const summary = JSON.parse(await readFile(positionals[0], 'utf8')) as RunSummary & { commit?: string };
    const page = summary.pages.find((candidate) => candidate.name === 'platform-compute');
    if (!page || page.outcome !== 'pass') throw new Error(`no passing platform-compute page in ${positionals[0]}`);
    const data = page.data as PageData;
    if (!data.timing) throw new Error('the run has no timing section: run the page with VITE_PLATFORM_COMPUTE_TIMING=1');
    const device = deviceNameFromAdapter(summary.adapter);
    const adapter = summary.adapter ? `${summary.adapter.vendor}/${summary.adapter.architecture}` : 'unknown';
    const browser = data.userAgent ? browserFromUserAgent(data.userAgent, data.browserVersion ?? null) : 'Chrome headless';
    const commit = values.commit ?? summary.commit ?? gitCommit();
    const rows = computeTimingRows(data.timing);
    if (isNvidiaDeviceName(device)) {
        for (const note of clockNoteRowsFromOption(values['clock-file'])) rows.push({ setting: 'note', metric: note.metric, value: note.value, unit: '' });
    }
    const prefix = [summary.startedAt.slice(0, 10), commit, device, browser, adapter, 'synthetic', 'none'];
    const lines = rows.map((row) => [...prefix, row.setting, row.metric, formatValue(row.value), row.unit].join(','));
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
