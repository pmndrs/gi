/// <reference types="node" />
/**
 * Turns a test-runner `--json` file from the trace-bench pages into rows for
 * `bench/results/trace.csv`.
 *
 *   node tools/gpu/pages/trace-bench/to-csv.ts <runner.json> [options]
 *
 *   --append            add the rows to bench/results/trace.csv (written with its header if new)
 *   --out <file>        add them to this file instead (implies --append)
 *   --commit <hash>     default: `git rev-parse --short HEAD`, plus "-dirty" if tracked files changed
 *   --device <name>     default: derived from the adapter, for example "RTX 3060 Laptop"
 *   --browser <text>    default: derived from the page's user agent, for example "Chrome 156.0.8078.74 headless"
 *   --date <YYYY-MM-DD> default: the day the run started
 *   --include-failed    also take pages that did not pass (a disagreement fails a page; its speeds are suspect)
 *
 * Without --append or --out the rows are printed. Columns are those of
 * bench/results/README.md: date, commit, device, browser, scene, metric, value.
 */

import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { PageSummary, RunSummary } from '../../contract.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, traceBenchRows } from './csv-rows.ts';
import type { TraceBenchData } from './results.ts';

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

function gitCommit(): string {
    const git = (...args: string[]) => execFileSync('git', args, { cwd: REPOSITORY_ROOT, encoding: 'utf8' }).trim();
    const hash = git('rev-parse', '--short', 'HEAD');
    return git('status', '--porcelain', '--untracked-files=no') === '' ? hash : `${hash}-dirty`;
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
        },
    });
    if (positionals.length !== 1) {
        console.error('Usage: node tools/gpu/pages/trace-bench/to-csv.ts <runner.json> [--append] [--out file] [--commit h] [--device n] [--browser b] [--date d] [--include-failed]');
        process.exit(2);
    }

    const summary = JSON.parse(await readFile(positionals[0], 'utf8')) as RunSummary;
    const date = values.date ?? summary.startedAt.slice(0, 10);
    const commit = values.commit ?? gitCommit();
    const device = values.device ?? deviceNameFromAdapter(summary.adapter);

    const lines: string[] = [];
    for (const page of summary.pages.filter(isTraceBenchPage)) {
        if (page.outcome !== 'pass' && !values['include-failed']) {
            console.error(`skipping ${page.name}: ${page.outcome} (${page.reasons.join('; ') || 'no reason given'}). Use --include-failed to take it anyway.`);
            continue;
        }
        const pageData = page.data as unknown as TraceBenchData;
        const browser = values.browser ?? browserFromUserAgent(pageData.userAgent ?? '', pageData.browserVersion ?? null);
        lines.push(...pageToCsvLines(page, { date, commit, device, browser }));
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
