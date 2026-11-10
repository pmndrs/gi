/// <reference types="node" />
/**
 * Turns one page's `data` from a test-runner `--json` file into CSV rows, for
 * pages that have no to-csv script of their own (T2.10). Every number in the
 * page's data becomes a row, its name the path to it under `--prefix`
 * (`<prefix>/<key>/<key>...`); booleans become 1 or 0; strings are left out.
 * Arrays of numbers become `<name>/<index>` rows, up to `--max-array` long
 * (default 130, so per-frame series of a few seconds fit).
 *
 *   node tools/gpu/flatten-to-csv.ts <runner.json> --page <name> --prefix <metric prefix> --csv <file in bench/results>
 *       [--scene <id>] [--scene-key <data key>] [--exclude <regexp>] [--max-array <n>] [--clock-file <file>]
 *       [--note <metric>=<n>] [--include-failed] [--append]
 *
 * The scene column: `--scene`, or with `--scene-key` the first path segment
 * under that key names it (`--scene-key scenes`: `scenes/two-room/...` rows go
 * to scene `two-room`), otherwise `all`. Columns and notes are those of
 * bench/results/README.md; `--clock-file` (or DDGI_CLOCK_FILE) adds the clock
 * rule's `note/...` rows to RTX rows. Without `--append` the rows are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from './contract.ts';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from './clock-note.ts';
import { gitCommit } from './measured-commit.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, type MetricRow } from './pages/trace-bench/csv-rows.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';
const RESULTS = resolve(import.meta.dirname, '..', '..', 'bench', 'results');

/** Every number under `value`, as rows named by their path. */
export function flattenRows(value: unknown, path: string[], options: { maxArray: number; exclude: RegExp | null }): Array<{ path: string[]; value: number }> {
    const name = path.join('/');
    if (options.exclude && path.length > 0 && options.exclude.test(name)) return [];
    if (typeof value === 'number') return Number.isFinite(value) ? [{ path, value }] : [];
    if (typeof value === 'boolean') return [{ path, value: value ? 1 : 0 }];
    if (Array.isArray(value)) {
        if (value.length > options.maxArray) return [];
        return value.flatMap((item, index) => flattenRows(item, [...path, String(index)], options));
    }
    if (typeof value === 'object' && value !== null) {
        return Object.entries(value as Record<string, unknown>).flatMap(([key, item]) => flattenRows(item, [...path, key], options));
    }
    return [];
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: {
            page: { type: 'string' },
            prefix: { type: 'string' },
            csv: { type: 'string' },
            scene: { type: 'string' },
            'scene-key': { type: 'string' },
            exclude: { type: 'string' },
            'max-array': { type: 'string' },
            'clock-file': { type: 'string' },
            note: { type: 'string', multiple: true },
            'include-failed': { type: 'boolean' },
            append: { type: 'boolean' },
        },
    });
    if (positionals.length !== 1 || !values.page || !values.prefix || !values.csv) {
        console.error('Usage: node tools/gpu/flatten-to-csv.ts <runner.json> --page <name> --prefix <prefix> --csv <file> [options; see the file comment]');
        process.exit(2);
    }
    const summary = JSON.parse(await readFile(positionals[0], 'utf8')) as RunSummary & { commit?: string };
    const page = summary.pages.find((candidate) => candidate.name === values.page);
    if (!page) throw new Error(`no page ${values.page} in ${positionals[0]}`);
    if (page.outcome !== 'pass' && !values['include-failed']) throw new Error(`${values.page} did not pass (${page.outcome}); --include-failed keeps its numbers with note/page_failed`);
    const data = page.data as Record<string, unknown>;
    const device = deviceNameFromAdapter(summary.adapter);
    const userAgent = typeof data.userAgent === 'string' ? data.userAgent : null;
    const browser = userAgent ? browserFromUserAgent(userAgent, typeof data.browserVersion === 'string' ? data.browserVersion : null) : 'Chrome headless';
    const commit = summary.commit ?? gitCommit();
    const prefix = [summary.startedAt.slice(0, 10), commit, device, browser];
    const exclude = values.exclude ? new RegExp(values.exclude) : null;
    const maxArray = values['max-array'] ? Number(values['max-array']) : 130;
    const sceneKey = values['scene-key'];

    const rows: MetricRow[] = [];
    for (const row of flattenRows(data, [], { maxArray, exclude })) {
        let scene = values.scene ?? 'all';
        let path = row.path;
        if (sceneKey && path[0] === sceneKey && path.length > 2) {
            scene = path[1];
            path = path.slice(2);
        }
        rows.push({ scene, metric: `${values.prefix}/${path.join('/')}`, value: row.value });
    }
    for (const note of values.note ?? []) {
        const [metric, value] = note.split('=');
        if (!metric || value === undefined || !Number.isFinite(Number(value))) throw new Error(`--note needs <metric>=<number>, got "${note}"`);
        rows.push({ scene: 'all', metric: `note/${metric}`, value: Number(value) });
    }
    if (page.outcome !== 'pass') rows.push({ scene: 'all', metric: 'note/page_failed', value: 1 });
    if (isNvidiaDeviceName(device)) rows.push(...clockNoteRowsFromOption(values['clock-file']));

    const lines = rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(','));
    if (!values.append) {
        console.log([CSV_HEADER, ...lines].join('\n'));
        return;
    }
    const file = resolve(RESULTS, values.csv);
    if (!existsSync(file)) await writeFile(file, `${CSV_HEADER}\n`);
    await appendFile(file, `${lines.join('\n')}\n`);
    console.error(`added ${lines.length} rows to ${file}`);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
