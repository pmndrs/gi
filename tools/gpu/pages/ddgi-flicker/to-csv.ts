/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the `ddgi-flicker` page into rows for
 * `bench/results/flicker.csv`, and writes the page's PNG frame sequences.
 *
 *   node tools/gpu/pages/ddgi-flicker/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--images <dir>] [--note <metric>=<value>] [--clock-file <file>]
 *
 * Several files of the same GPU and scene are repeated runs: `run1/`, `run2/`,
 * ... plus `median/` rows. The scene column is `<scene>/<camera bookmark name>`;
 * metrics are `flicker/<budget>/<name>` with the budget written `gpuMs2` or
 * `units256`, or `flicker/<rule>/<budget>/<name>` when the run named
 * hysteresis rules (`tau5`, `fixed0.97`; definitions at the top of main.ts). `--images <dir>` saves the
 * PNGs of the FIRST file with images under `<dir>/<gpu>/<scene>/` (the page
 * only makes them when asked, so give it files from a run with
 * VITE_DDGI_FLICKER_IMAGES=1). Without --append the rows are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from '../../clock-note.ts';
import { gitCommit } from '../../measured-commit.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from '../trace-bench/csv-rows.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';
const DEFAULT_CSV = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'flicker.csv');

interface PageData {
    userAgent?: string;
    browserVersion?: string | null;
    scene: string;
    camera: string;
    width: number;
    height: number;
    probes: number;
    warmup_passes: number;
    converge_passes: number;
    converged_image_mean: number;
    settings: Record<string, Record<string, number | string>>;
    images?: Array<{ name: string; base64: string }>;
}

export function flickerRows(data: PageData): MetricRow[] {
    const scene = `${data.scene}/${data.camera}`;
    const rows: MetricRow[] = [
        { scene, metric: 'flicker/width_px', value: data.width },
        { scene, metric: 'flicker/height_px', value: data.height },
        { scene, metric: 'flicker/probes_in_volume', value: data.probes },
        { scene, metric: 'flicker/converge_passes', value: data.converge_passes },
        { scene, metric: 'flicker/converged_image_mean', value: data.converged_image_mean },
    ];
    for (const [budget, values] of Object.entries(data.settings)) {
        // '<budget>', or '<hysteresis rule>/<budget>' when the run named rules (tau:5/gpuMs:2 becomes tau5/gpuMs2).
        const name = budget.replaceAll(':', '');
        for (const [metric, value] of Object.entries(values)) {
            if (typeof value === 'number' && Number.isFinite(value)) rows.push({ scene, metric: `flicker/${name}/${metric}`, value });
        }
    }
    return rows;
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, commit: { type: 'string' }, images: { type: 'string' }, note: { type: 'string', multiple: true }, 'clock-file': { type: 'string' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tools/gpu/pages/ddgi-flicker/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>] [--images <dir>] [--note <metric>=<value>] [--clock-file <file>]');
        process.exit(2);
    }
    const notes: MetricRow[] = (values.note ?? []).map((note) => {
        const [metric, value] = note.split('=');
        if (!metric || value === undefined || !Number.isFinite(Number(value))) throw new Error(`--note needs <metric>=<number>, got "${note}"`);
        return { scene: 'all', metric: `note/${metric}`, value: Number(value) };
    });
    const clockNotes = clockNoteRowsFromOption(values['clock-file']);
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][] }>();
    let imagesWritten = false;
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            if (page.name !== 'ddgi-flicker') continue;
            if (page.outcome !== 'pass') {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const data = page.data as unknown as PageData;
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const prefix = [summary.startedAt.slice(0, 10), commit, device, browserFromUserAgent(data.userAgent ?? '', data.browserVersion ?? null)];
            const key = `${device}|${data.scene}`;
            const group = groups.get(key) ?? { prefix, runs: [] };
            group.runs.push(flickerRows(data));
            groups.set(key, group);
            if (values.images && data.images && data.images.length > 0 && !imagesWritten) {
                const directory = join(values.images, device.toLowerCase().includes('rtx') ? 'rtx3060' : 'intel-uhd', data.scene);
                await mkdir(directory, { recursive: true });
                for (const image of data.images) await writeFile(join(directory, `${image.name}.png`), Buffer.from(image.base64, 'base64'));
                console.error(`wrote ${data.images.length} PNGs to ${directory}`);
                imagesWritten = true;
            }
        }
    }
    const lines: string[] = [];
    for (const { prefix, runs } of groups.values()) {
        const rows = [...(runs.length === 1 ? runs[0] : repeatedRunRows(runs)), ...notes, ...(isNvidiaDeviceName(prefix[2]) ? clockNotes : [])];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(',')));
    }
    if (lines.length === 0) {
        console.error('no passing ddgi-flicker pages in those files');
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
