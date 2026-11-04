/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the trace-bench pages into rows for
 * `bench/results/trace.csv`.
 *
 *   node tools/gpu/pages/trace-bench/to-csv.ts <runner.json> [more runner.json ...] [options]
 *
 * One file gives one set of rows, named as in csv-rows.ts. Several files are
 * repeated runs of the same pages on the same GPU (see repeat.ts): every run's
 * rows come out prefixed `run1/`, `run2/`, ... and a `median/` row per metric
 * follows, plus `median/runs`.
 *
 *   --append            add the rows to bench/results/trace.csv (written with its header if new)
 *   --out <file>        add them to this file instead (implies --append)
 *   --commit <hash>     default: the commit the runner recorded when the pages ran (RunSummary.commit:
 *                       HEAD, plus "-dirty" if the measured code differed, see MEASURED_PATHS); for runner
 *                       files from before that field, HEAD now, with a warning
 *   --device <name>     default: derived from the adapter, for example "RTX 3060 Laptop"
 *   --browser <text>    default: derived from the page's user agent, for example "Chrome 156.0.8078.74 headless"
 *   --date <YYYY-MM-DD> default: the day the run started
 *   --include-failed    also take pages that did not pass (a disagreement fails a page; its speeds are suspect)
 *   --clock-file <file> add `note/sm_clock_mhz` (and min, max, memory) rows to the NVIDIA runs, from the summary
 *                       `bench-lock run` writes; default: the file named by DDGI_CLOCK_FILE, which it sets
 *
 * Without --append or --out the rows are printed. Columns are those of
 * bench/results/README.md: date, commit, device, browser, scene, metric, value.
 */

import { existsSync } from 'node:fs';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { PageSummary, RunSummary } from '../../contract.ts';
import { gitCommit } from '../../measured-commit.ts';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from '../../clock-note.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, traceBenchRows, type MetricRow } from './csv-rows.ts';
import type { TraceBenchData } from './results.ts';

// Kept here for older callers (repeat.ts); the definition moved next to the runner.
export { gitCommit, MEASURED_PATHS } from '../../measured-commit.ts';

export const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '..', '..', '..', '..');
const DEFAULT_CSV = resolve(REPOSITORY_ROOT, 'bench', 'results', 'trace.csv');

export interface CsvContext {
    date: string;
    commit: string;
    device: string;
    browser: string;
}

export function isTraceBenchPage(page: PageSummary): boolean {
    return (page.data as { benchmark?: unknown }).benchmark === 'trace-bench';
}

/** CSV lines (no header) for one page. */
export function pageToCsvLines(page: PageSummary, context: CsvContext): string[] {
    const data = page.data as unknown as TraceBenchData;
    const prefix = [context.date, context.commit, context.device, context.browser];
    return traceBenchRows(data).map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(','));
}

interface PageRun {
    key: string;
    context: CsvContext;
    rows: MetricRow[];
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: {
            append: { type: 'boolean' },
            out: { type: 'string' },
            commit: { type: 'string' },
            device: { type: 'string' },
            browser: { type: 'string' },
            date: { type: 'string' },
            'include-failed': { type: 'boolean' },
            'clock-file': { type: 'string' },
        },
    });
    if (positionals.length < 1) {
        console.error(
            'Usage: node tools/gpu/pages/trace-bench/to-csv.ts <runner.json> [more.json ...] [--append] [--out file] [--commit h] [--device n] [--browser b] [--date d] [--include-failed] [--clock-file f]',
        );
        process.exit(2);
    }

    // Every passing trace-bench page of every file, grouped by page and device.
    const groups = new Map<string, PageRun[]>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const date = values.date ?? summary.startedAt.slice(0, 10);
        // The commit belongs to the moment the pages ran, not to now: the tree may have moved since.
        let commit = values.commit ?? summary.commit;
        if (commit === undefined) {
            commit = gitCommit();
            console.error(`${file} has no recorded commit (runner from before T1.3); using the current one, ${commit}`);
        }
        const device = values.device ?? deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages.filter(isTraceBenchPage)) {
            if (page.outcome !== 'pass' && !values['include-failed']) {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome} (${page.reasons.join('; ') || 'no reason given'}). Use --include-failed to take it anyway.`);
                continue;
            }
            const pageData = page.data as unknown as TraceBenchData;
            const browser = values.browser ?? browserFromUserAgent(pageData.userAgent ?? '', pageData.browserVersion ?? null);
            const key = `${page.name}|${device}`;
            const run: PageRun = { key, context: { date, commit, device, browser }, rows: traceBenchRows(pageData) };
            groups.set(key, [...(groups.get(key) ?? []), run]);
        }
    }

    const clockNotes = clockNoteRowsFromOption(values['clock-file']);
    const lines: string[] = [];
    for (const runs of groups.values()) {
        // Repeated runs share the first run's date and browser; the rows say how many there were.
        const context = runs[0].context;
        const rows = [...(runs.length === 1 ? runs[0].rows : repeatedRunRows(runs.map((run) => run.rows))), ...(isNvidiaDeviceName(context.device) ? clockNotes : [])];
        const prefix = [context.date, context.commit, context.device, context.browser];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(',')));
    }
    if (lines.length === 0) {
        console.error('no trace-bench rows found in that file');
        process.exit(1);
    }

    if (!values.append && !values.out) {
        console.log([CSV_HEADER, ...lines].join('\n'));
        return;
    }
    const target = values.out ? resolve(values.out) : DEFAULT_CSV;
    await mkdir(dirname(target), { recursive: true });
    if (!existsSync(target)) await writeFile(target, `${CSV_HEADER}\n`);
    await appendFile(target, `${lines.join('\n')}\n`);
    console.error(`added ${lines.length} rows to ${target}`);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
