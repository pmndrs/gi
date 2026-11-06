/// <reference types="node" />
/**
 * Turns a test-runner `--json` file from the `ddgi-memory` page into rows for
 * `bench/results/memory.csv` (columns as in bench/results/README.md).
 *
 *   node tools/gpu/pages/ddgi-memory/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--note <metric>=<value>] [--include-failed]
 *
 * Metrics are `memory/<name>`, bytes unless the name ends in `_mb` (decimal
 * megabytes, 1 MB = 1,000,000 bytes, as the viewer's stats show them).
 * `memory/total_*` is `gi.stats.memory.total`; `memory/everything_*` adds the
 * shading module's sky, which the total leaves out (see main.ts). Several
 * files are separate GPUs or repeated runs: repeated runs of one GPU get
 * `run<n>/` and `median/` rows like the other CSVs. Without --append the rows
 * are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { gitCommit } from '../../measured-commit.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from '../trace-bench/csv-rows.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';
const DEFAULT_CSV = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'memory.csv');

interface PageData {
    userAgent?: string;
    browserVersion?: string | null;
    scenes: Record<string, Record<string, number | string>>;
}

export function memoryRows(data: PageData): MetricRow[] {
    return Object.entries(data.scenes).flatMap(([scene, values]) =>
        Object.entries(values).flatMap(([name, value]) => (typeof value === 'number' && Number.isFinite(value) ? [{ scene, metric: `memory/${name}`, value }] : [])),
    );
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, commit: { type: 'string' }, note: { type: 'string', multiple: true }, 'include-failed': { type: 'boolean' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tools/gpu/pages/ddgi-memory/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--note <metric>=<value>] [--include-failed]');
        process.exit(2);
    }
    const notes: MetricRow[] = (values.note ?? []).map((note) => {
        const [metric, value] = note.split('=');
        if (!metric || value === undefined || !Number.isFinite(Number(value))) throw new Error(`--note needs <metric>=<number>, got "${note}"`);
        return { scene: 'all', metric: `note/${metric}`, value: Number(value) };
    });
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][]; failed: number }>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            if (page.name !== 'ddgi-memory') continue;
            const failed = page.outcome !== 'pass';
            if (failed && !values['include-failed']) {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const data = page.data as unknown as PageData;
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const prefix = [summary.startedAt.slice(0, 10), commit, device, browserFromUserAgent(data.userAgent ?? '', data.browserVersion ?? null)];
            const group = groups.get(device) ?? { prefix, runs: [], failed: 0 };
            group.runs.push(memoryRows(data));
            group.failed += failed ? 1 : 0;
            groups.set(device, group);
        }
    }
    const lines: string[] = [];
    for (const { prefix, runs, failed } of groups.values()) {
        const rows = [...(runs.length === 1 ? runs[0] : repeatedRunRows(runs)), ...(failed > 0 ? [{ scene: 'all', metric: 'note/page_failed', value: failed }] : []), ...notes];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(',')));
    }
    if (lines.length === 0) {
        console.error('no ddgi-memory pages in those files');
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
