/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the `ddgi-timing` page into rows for
 * `bench/results/timing.csv` (columns as in bench/results/README.md).
 *
 *   node tools/gpu/pages/ddgi-timing/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--browser <text>] [--clock-file <file>]
 *
 * Several files of the same GPU are repeated runs: `run1/`, `run2/`, ... plus `median/` rows, as
 * for trace.csv. The scene column is the DDGI scene (`cornell-box-point`, `two-room`). The page
 * reports no user agent, so name the browser with --browser (default "Chrome headless").
 * `--clock-file` (or DDGI_CLOCK_FILE, which `bench-lock run` sets) adds `note/sm_clock_mhz` and
 * friends to the NVIDIA rows. Without --append the rows are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from '../../clock-note.ts';
import { gitCommit } from '../../measured-commit.ts';
import { deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from '../trace-bench/csv-rows.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';
const DEFAULT_CSV = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'timing.csv');

/** What the page returns per scene (main.ts). */
interface SceneTiming {
    probes: number;
    raysPerProbe: number;
    units: Record<string, { probesUpdated: number; traceMs: number; blendMs: number; writeMs: number; msPerProbe: number; probesIn2Ms: number }>;
    budgetCheck: { gpuMs: number; medianProbesUpdated: number; medianDdgiMs: number; msPerUnit: number };
}

export function ddgiTimingRows(data: Record<string, SceneTiming>): MetricRow[] {
    const rows: MetricRow[] = [];
    for (const [scene, timing] of Object.entries(data)) {
        rows.push({ scene, metric: 'ddgi/probes_in_volume', value: timing.probes }, { scene, metric: 'ddgi/rays_per_probe', value: timing.raysPerProbe });
        for (const [units, unit] of Object.entries(timing.units)) {
            const prefix = `ddgi/units${units}`;
            rows.push(
                { scene, metric: `${prefix}/probes_updated`, value: unit.probesUpdated },
                { scene, metric: `${prefix}/trace_gpu_ms_median`, value: unit.traceMs },
                { scene, metric: `${prefix}/blend_gpu_ms_median`, value: unit.blendMs },
                { scene, metric: `${prefix}/write_gpu_ms_median`, value: unit.writeMs },
                { scene, metric: `${prefix}/us_per_probe`, value: unit.msPerProbe * 1000 },
                { scene, metric: `${prefix}/probes_in_2ms`, value: unit.probesIn2Ms },
            );
        }
        const budget = timing.budgetCheck;
        rows.push(
            { scene, metric: 'ddgi/budget_2ms/probes_updated_median', value: budget.medianProbesUpdated },
            { scene, metric: 'ddgi/budget_2ms/ddgi_gpu_ms_median', value: budget.medianDdgiMs },
        );
    }
    return rows;
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, commit: { type: 'string' }, browser: { type: 'string' }, 'clock-file': { type: 'string' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tools/gpu/pages/ddgi-timing/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--browser <text>] [--clock-file <file>]');
        process.exit(2);
    }
    const clockNotes = clockNoteRowsFromOption(values['clock-file']);
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][] }>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            if (page.name !== 'ddgi-timing') continue;
            if (page.outcome !== 'pass') {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const prefix = [summary.startedAt.slice(0, 10), commit, device, values.browser ?? 'Chrome headless'];
            const group = groups.get(device) ?? { prefix, runs: [] };
            group.runs.push(ddgiTimingRows(page.data as unknown as Record<string, SceneTiming>));
            groups.set(device, group);
        }
    }
    const lines: string[] = [];
    for (const { prefix, runs } of groups.values()) {
        const rows = [...(runs.length === 1 ? runs[0] : repeatedRunRows(runs)), ...(isNvidiaDeviceName(prefix[2]) ? clockNotes : [])];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(',')));
    }
    if (lines.length === 0) {
        console.error('no passing ddgi-timing pages in those files');
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
