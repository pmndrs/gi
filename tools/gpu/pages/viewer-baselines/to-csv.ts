/// <reference types="node" />
/**
 * Turns the runner's `--json` output of the `viewer-baselines` page into
 * rows for `bench/results/baselines.csv` and prints a table.
 *
 *   pnpm test:gpu --filter viewer-baselines --gpu nvidia --json run.json
 *   node tools/gpu/pages/viewer-baselines/to-csv.ts run1.json [run2.json ...] [--device "RTX 3060 Laptop"] [--append] [--note quiet=1] [--clock-file <file>]
 *
 * Several files are repeated runs: rows come out as `run1/`, `run2/`, ... plus
 * `median/` rows, as for the other CSVs. `--clock-file` (or DDGI_CLOCK_FILE,
 * which `bench-lock run` sets) adds the `note/sm_clock_mhz` rows on NVIDIA.
 *
 * Without `--append` it only prints. With it, rows are added to
 * `bench/results/baselines.csv` (columns as in bench/results/README.md).
 * The technique is part of the metric name, for example `lpg_bake_wall_ms`.
 *
 * What gets written per measurement (definitions in apps/viewer/src/bench/baselines.ts):
 *  - `<technique>_<metric>`: the value (a median when repeated; a mean for a
 *    difference such as `gi_gpu_ms`). Always finite. Never negative, except a
 *    difference whose confidence interval contains 0.
 *  - For a difference: `_ci_low` and `_ci_high` (the 95% confidence
 *    interval) and `_below_noise` (1 when the interval contains 0).
 *  - `<technique>_<metric>_cv` and `<technique>_<metric>_noisy` (1 or 0) for a
 *    repeated measurement: its spread, and whether that is above the noise
 *    threshold of the run.
 *  - `<technique>_<metric>_problem`: instead of the value, when the
 *    measurement failed, with the reason.
 */

import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from '../../clock-note.ts';
import { formatValue, repeatedRunRows, type MetricRow } from '../trace-bench/csv-rows.ts';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..', '..');
const csvPath = join(repoRoot, 'bench', 'results', 'baselines.csv');
const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';

interface Measurement {
    name: string;
    value: number | null;
    repeats: number;
    cv: number | null;
    noisy: boolean;
    confidenceInterval: { low: number; high: number } | null;
    belowNoise: boolean | null;
    problem: string | null;
}

interface Row {
    scene: string;
    technique: string;
    triangles: number;
    measurements: Measurement[];
    details: Record<string, number | string>;
    error: string | null;
}

interface PageData {
    rows: Row[];
    adapter: { vendor: string; architecture: string; description: string };
    hasTimestamps: boolean;
    userAgent: string;
    noisyCvThreshold: number;
    options: { sampleFrames: number; frameRounds: number; bakeRuns: number; width: number; height: number; maxTextureSize?: number | null };
}

interface RunJson {
    gpuRequested: string | null;
    pages: Array<{ name: string; outcome: string; data: PageData | null }>;
}

interface Options {
    files: string[];
    device: string | null;
    append: boolean;
    notes: MetricRow[];
    clockFile: string | undefined;
}

function parseArguments(argv: string[]): Options {
    const { values, positionals } = parseArgs({
        args: argv,
        allowPositionals: true,
        options: { device: { type: 'string' }, append: { type: 'boolean' }, note: { type: 'string', multiple: true }, 'clock-file': { type: 'string' } },
    });
    if (positionals.length === 0) throw new Error('usage: to-csv.ts <runner-json> [more.json ...] [--device "<name>"] [--append] [--note <metric>=<number>] [--clock-file <file>]');
    // `--note quiet=1` becomes the row `note/quiet,1` (scene `all`), as in the other CSVs.
    const notes: MetricRow[] = (values.note ?? []).map((note) => {
        const [metric, value] = note.split('=');
        if (!metric || value === undefined || !Number.isFinite(Number(value))) throw new Error(`--note needs <metric>=<number>, got "${note}"`);
        return { scene: 'all', metric: `note/${metric}`, value: Number(value) };
    });
    return { files: positionals, device: values.device ?? null, append: values.append ?? false, notes, clockFile: values['clock-file'] };
}

function defaultDeviceName(vendor: string): string {
    if (vendor === 'nvidia') return 'RTX 3060 Laptop';
    if (vendor === 'intel') return 'Intel UHD';
    return vendor;
}

/** "...HeadlessChrome/156.0.8078.74 ..." becomes "Chrome 156.0.8078.74 headless". */
function browserName(userAgent: string): string {
    const match = /(Headless)?Chrome\/([\d.]+)/.exec(userAgent);
    return match ? `Chrome ${match[2]}${match[1] ? ' headless' : ''}` : userAgent;
}

function gitCommit(): string {
    try {
        const hash = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim();
        const dirty = execFileSync('git', ['status', '--porcelain'], { cwd: repoRoot, encoding: 'utf8' }).trim() !== '';
        return dirty ? `${hash}-dirty` : hash;
    } catch {
        return 'unknown';
    }
}

/** Commas and line breaks would break the CSV; a problem text is one cell. */
function cellText(text: string): string {
    return text.replaceAll(',', ';').replaceAll('\n', ' ');
}

/**
 * The page's numbers as rows (scene, metric, value) for one run. Failed
 * measurements come back as text in `problems`: they are not numbers, so
 * they stay out of the run1/median arithmetic.
 */
function metricRows(data: PageData): { rows: MetricRow[]; problems: Array<{ scene: string; metric: string; text: string }> } {
    const rows: MetricRow[] = [];
    const problems: Array<{ scene: string; metric: string; text: string }> = [];
    const add = (scene: string, metric: string, value: number, mayBeNegative = false) => {
        if (!Number.isFinite(value) || (value < 0 && !mayBeNegative)) {
            // baselines.ts never hands these out; refuse rather than write a number nobody should trust.
            throw new Error(`${scene} ${metric}: refusing to write ${value}`);
        }
        rows.push({ scene, metric, value });
    };
    // The view size goes into the file, so a 1440p run can be told from a 720p one.
    add('all', 'view/width_px', data.options.width);
    add('all', 'view/height_px', data.options.height);
    // Model textures scaled down at load (T2.17); absent when they were kept as the files have them.
    if (typeof data.options.maxTextureSize === 'number') add('all', 'view/max_texture_px', data.options.maxTextureSize);
    for (const row of data.rows) {
        const name = row.technique;
        // A failed measurement is recorded with its reason, never as a blank or a made-up number.
        if (row.error) problems.push({ scene: row.scene, metric: `${name}_problem`, text: row.error });
        for (const measurement of row.measurements) {
            const metric = `${name}_${measurement.name}`;
            if (measurement.value === null) {
                problems.push({ scene: row.scene, metric: `${metric}_problem`, text: measurement.problem ?? 'failed without a reason' });
                continue;
            }
            // Only a difference whose interval contains 0 may be negative (see baselines.ts).
            add(row.scene, metric, measurement.value, measurement.belowNoise === true);
            if (measurement.confidenceInterval) {
                add(row.scene, `${metric}_ci_low`, measurement.confidenceInterval.low, true);
                add(row.scene, `${metric}_ci_high`, measurement.confidenceInterval.high, true);
            }
            if (measurement.belowNoise !== null) add(row.scene, `${metric}_below_noise`, measurement.belowNoise ? 1 : 0);
            if (measurement.repeats > 1) {
                if (measurement.cv !== null) add(row.scene, `${metric}_cv`, measurement.cv);
                add(row.scene, `${metric}_noisy`, measurement.noisy ? 1 : 0);
            }
        }
        for (const [key, value] of Object.entries(row.details)) {
            if (typeof value === 'number' && Number.isFinite(value) && value >= 0) add(row.scene, `${name}_${key}`, value);
        }
    }
    return { rows, problems };
}

/** A cell of the PROGRESS table: the value, marked when noisy, or "failed". */
function cell(row: Row, name: string, digits: number): string {
    const measurement = row.measurements.find((candidate) => candidate.name === name);
    if (!measurement) return '-';
    if (measurement.value === null) return 'failed';
    const interval = measurement.confidenceInterval
        ? ` [${measurement.confidenceInterval.low.toFixed(digits)}, ${measurement.confidenceInterval.high.toFixed(digits)}]`
        : '';
    const flags = [measurement.noisy ? 'noisy' : '', measurement.belowNoise ? 'below noise' : ''].filter(Boolean).join(', ');
    return `${measurement.value.toFixed(digits)}${interval}${flags ? ` (${flags})` : ''}`;
}

/** The table that goes in PROGRESS: scene x technique, bake ms, frame ms, GI ms. */
function markdownTable(data: PageData, device: string): string {
    const { options } = data;
    const lines = [
        `Device: ${device} (${data.adapter.architecture}), ${options.width}x${options.height}, ${options.frameRounds} rounds of ${options.sampleFrames} frames, ${options.bakeRuns} bake runs, timestamps ${data.hasTimestamps ? 'yes' : 'no'}, noisy above CV ${data.noisyCvThreshold}`,
        '',
        '| scene | technique | bake ms (wall) | bake ms (GPU pass sum) | frame ms (GPU) | GI-off frame ms (GPU) | GI ms [95% interval] | note |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ];
    for (const row of data.rows) {
        const problems = row.measurements.filter((measurement) => measurement.problem).map((measurement) => `${measurement.name}: ${measurement.problem}`);
        const size =
            row.technique === 'lpg'
                ? `${row.details.probes} probes`
                : row.technique === 'vxgi'
                  ? `${row.details.voxel_resolution} voxels on the longest axis`
                  : row.technique === 'ddgi'
                    ? `${row.details.probes} probes, budget ${row.details.budget_ms} ms; first GI ${cell(row, 'first_gi_wall_ms', 0)} ms, warmed ${cell(row, 'warmed_wall_ms', 0)} ms; frame p95 ${cell(row, 'frame_gpu_p95_ms', 2)} (GI off ${cell(row, 'gi_off_frame_gpu_p95_ms', 2)}) ms`
                    : '';
        const note = [row.error, ...problems, size].filter(Boolean).join('; ');
        lines.push(
            `| ${row.scene} | ${row.technique} | ${cell(row, 'bake_wall_ms', 0)} | ${cell(row, 'bake_gpu_pass_sum_ms', 0)} | ${cell(row, 'frame_gpu_ms', 2)} | ${cell(row, 'gi_off_frame_gpu_ms', 2)} | ${cell(row, 'gi_gpu_ms', 2)} | ${note} |`,
        );
    }
    return lines.join('\n');
}

const options = parseArguments(process.argv.slice(2));
const pages: PageData[] = [];
for (const file of options.files) {
    const run = JSON.parse(readFileSync(file, 'utf8')) as RunJson;
    const page = run.pages.find((candidate) => candidate.name === 'viewer-baselines');
    if (!page?.data) throw new Error(`${file} has no data for the viewer-baselines page (outcome: ${page?.outcome ?? 'page missing'}).`);
    if (page.outcome !== 'pass') console.log(`Note: the page did not pass in ${file} (${page.outcome}); failed measurements are written as _problem rows.\n`);
    pages.push(page.data);
}
const device = options.device ?? defaultDeviceName(pages[0].adapter.vendor);

pages.forEach((data, index) => {
    if (pages.length > 1) console.log(`\nrun ${index + 1}`);
    console.log(markdownTable(data, device));
});
if (options.append) {
    if (!existsSync(csvPath)) writeFileSync(csvPath, `${CSV_HEADER}\n`);
    const perRun = pages.map(metricRows);
    const rows: MetricRow[] = [
        ...(perRun.length === 1 ? perRun[0].rows : repeatedRunRows(perRun.map((run) => run.rows))),
        ...options.notes,
        ...(isNvidiaDeviceName(device) ? clockNoteRowsFromOption(options.clockFile) : []),
    ];
    const date = new Date().toISOString().slice(0, 10);
    const prefix = `${date},${gitCommit()},${device},${browserName(pages[0].userAgent)}`;
    const lines = rows.map((row) => `${prefix},${row.scene},${row.metric},${formatValue(row.value)}`);
    perRun.forEach((run, index) => {
        for (const problem of run.problems) lines.push(`${prefix},${problem.scene},run${index + 1}/${problem.metric},${cellText(problem.text)}`);
    });
    appendFileSync(csvPath, `${lines.join('\n')}\n`);
    console.log(`\nappended ${lines.length} rows to ${csvPath}`);
}
