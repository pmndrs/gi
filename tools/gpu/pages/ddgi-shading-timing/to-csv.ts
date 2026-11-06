/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the `ddgi-shading-timing` page into
 * rows for `bench/results/shading.csv` (columns as in bench/results/README.md).
 *
 *   node tools/gpu/pages/ddgi-shading-timing/to-csv.ts <runner.json> [more runner.json ...] [--append] [--commit <hash>] [--note <metric>=<value>] [--clock-file <file>]
 *
 * Several files of the same GPU are repeated runs: rows come out as `run1/`,
 * `run2/`, ... plus a `median/` row per metric, as for trace.csv. The scene
 * column is `<scene>/<camera bookmark name>`. Without --append the rows are
 * printed. The commit defaults to the one the runner recorded. `--note`
 * adds a `note/<metric>` row per GPU (for example `gpu_busy=1`). `--clock-file` (or DDGI_CLOCK_FILE, which
 * `bench-lock run` sets) adds `note/sm_clock_mhz` and friends to the NVIDIA rows.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { gitCommit } from '../../measured-commit.ts';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from '../../clock-note.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from '../trace-bench/csv-rows.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';
const DEFAULT_CSV = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'shading.csv');

/** What the page reports (main.ts); only the fields that become rows. */
interface CameraTiming {
    moduleMsMedian: number;
    flatMsMedian: number;
    giSamplingMs: number;
    passes: number;
}

interface ShadingTimingData {
    benchmark: 'ddgi-shading-timing';
    scene: string;
    width: number;
    height: number;
    cameras: Record<string, CameraTiming>;
    userAgent: string;
    browserVersion: string | null;
}

export function shadingTimingRows(data: ShadingTimingData): MetricRow[] {
    const rows: MetricRow[] = [];
    for (const [camera, timing] of Object.entries(data.cameras)) {
        // The done-when size (2560x1440) keeps the plain name; another size is part of the scene, so the sizes never mix in the medians.
        const sizeSuffix = data.width === 2560 && data.height === 1440 ? '' : `@${data.width}x${data.height}`;
        const scene = `${data.scene}/${camera}${sizeSuffix}`;
        rows.push(
            { scene, metric: 'shading/width_px', value: data.width },
            { scene, metric: 'shading/height_px', value: data.height },
            // The hello forward pass with DDGI's giIrradiance, and with a constant in its place.
            { scene, metric: 'shading/forward_gpu_ms_with_gi_median', value: timing.moduleMsMedian },
            { scene, metric: 'shading/forward_gpu_ms_constant_median', value: timing.flatMsMedian },
            // The done-when number: the difference of the two medians.
            { scene, metric: 'shading/gi_sampling_gpu_ms', value: timing.giSamplingMs },
            { scene, metric: 'shading/passes_timed', value: timing.passes },
        );
    }
    return rows;
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, commit: { type: 'string' }, note: { type: 'string', multiple: true }, 'clock-file': { type: 'string' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tools/gpu/pages/ddgi-shading-timing/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--note <metric>=<value>] [--clock-file <file>]');
        process.exit(2);
    }
    const notes: MetricRow[] = (values.note ?? []).map((note) => {
        const [metric, value] = note.split('=');
        if (!metric || value === undefined || !Number.isFinite(Number(value))) throw new Error(`--note needs <metric>=<number>, got "${note}"`);
        return { scene: 'all', metric: `note/${metric}`, value: Number(value) };
    });
    const clockNotes = clockNoteRowsFromOption(values['clock-file']);
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][] }>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            const data = page.data as unknown as ShadingTimingData;
            if (data.benchmark !== 'ddgi-shading-timing') continue;
            if (page.outcome !== 'pass') {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const prefix = [summary.startedAt.slice(0, 10), commit, device, browserFromUserAgent(data.userAgent, data.browserVersion)];
            const group = groups.get(device) ?? { prefix, runs: [] };
            group.runs.push(shadingTimingRows(data));
            groups.set(device, group);
        }
    }
    const lines: string[] = [];
    for (const { prefix, runs } of groups.values()) {
        const rows = [...(runs.length === 1 ? runs[0] : repeatedRunRows(runs)), ...notes, ...(isNvidiaDeviceName(prefix[2]) ? clockNotes : [])];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(',')));
    }
    if (lines.length === 0) {
        console.error('no passing ddgi-shading-timing pages in those files');
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
