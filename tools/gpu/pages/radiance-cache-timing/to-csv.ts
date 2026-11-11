/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the `radiance-cache-timing` page into
 * CSV rows: `bench/results/radiance-cache-m5-passes.csv` for the Apple M5
 * (every row's note says a Mac's numbers are a hint), and
 * `bench/results/radiance-cache-passes.csv` for any other GPU (the RTX 3060's
 * rows are the record; NVIDIA rows get the clock notes).
 *
 *   node tools/gpu/pages/radiance-cache-timing/to-csv.ts <runner.json> [more.json ...] [--append] [--csv <file>] [--device <text>] [--commit <hash>] [--browser <text>] [--clock-file <file>]
 *
 * Chrome reports no description for Apple GPUs, so name the Mac's with
 * --device (the M5's rows say `Apple M5 Pro (apple/metal-3)`).
 *
 * Columns: date, commit, device, browser, scene, metric, value, note. Several
 * files of the same GPU are repeated runs: `run1/`, `run2/`, ... plus
 * `median/` rows. Metrics, per multibounce setting (`mb_on`, `mb_off`):
 * `rc/<setting>/section/<Capsaicin section>/gpu_ms_median`,
 * `rc/<setting>/group/<native group>/gpu_ms` (the sum of its section
 * medians), with lever 1's budget as `mb_on_budget` and an alternation's
 * second round as `..._round2` (RC-L1), `rc/<setting>/diffuse_subtotal_gpu_ms` (native's subtotal),
 * `rc/<setting>/all_passes_gpu_ms_median` (the median frame of every
 * radiance-cache pass), `rc/<setting>/pass/<pass>/gpu_ms_median` (only for
 * a section that is several passes: the trace and the cells of
 * `PopulateScreenProbesCells`),
 * `rc/<setting>/platform/<pass>/gpu_ms_median` and `rc/<setting>/frames_timed`.
 * Without --append the rows are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { clockNoteRowsFromOption, isNvidiaDeviceName } from '../../clock-note.ts';
import { gitCommit } from '../../measured-commit.ts';
import { deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from '../trace-bench/csv-rows.ts';
import { CAPSAICIN_GROUPS, sectionOf, type SceneTiming } from './sections.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value,note';
const RESULTS = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results');
const M5_CSV = resolve(RESULTS, 'radiance-cache-m5-passes.csv');
const RECORD_CSV = resolve(RESULTS, 'radiance-cache-passes.csv');
/** Every Apple row says what it is: the Mac's timing orders work, the RTX 3060 under the bench lock is the record (DECISIONS 2026-11-11). */
const APPLE_NOTE = 'Mac GPU timing: a hint not the record; radiance-cache-timing page; 1280x720 first bookmark; medians of 120 frames after 300 warm-up';
const PAGE = 'radiance-cache-timing';

/** A metric name part: no commas (the CSV has no quoting for metrics) and no spaces. */
const metricPart = (text: string): string => text.replace(/[ ,]+/g, '_');

export function radianceCacheTimingRows(data: Record<string, SceneTiming>): MetricRow[] {
    const rows: MetricRow[] = [];
    for (const [scene, timing] of Object.entries(data)) {
        rows.push({ scene, metric: 'rc/max_texture_size', value: timing.maxTextureSize }, { scene, metric: 'rc/triangles', value: timing.triangles });
        for (const setting of timing.settings) {
            // Lever 1 (RC-L1): budget rows and later alternation rounds get their own names; a
            // per-hit first round keeps the names it always had.
            const cellShading = setting.cellShading === 'budget' ? '_budget' : '';
            const round = (setting.round ?? 1) > 1 ? `_round${setting.round}` : '';
            const prefix = `rc/${setting.useMultibounce ? 'mb_on' : 'mb_off'}${cellShading}${round}`;
            rows.push({ scene, metric: `${prefix}/frames_timed`, value: setting.framesTimed });
            for (const [section, ms] of Object.entries(setting.sections)) rows.push({ scene, metric: `${prefix}/section/${metricPart(section)}/gpu_ms_median`, value: ms });
            for (const group of CAPSAICIN_GROUPS) rows.push({ scene, metric: `${prefix}/group/${metricPart(group)}/gpu_ms`, value: setting.groups[group] });
            rows.push(
                { scene, metric: `${prefix}/diffuse_subtotal_gpu_ms`, value: setting.diffuseSubtotalMs },
                { scene, metric: `${prefix}/all_passes_gpu_ms_median`, value: setting.radianceCacheFrameMs },
            );
            // Pass rows only where a section is more than one pass (the trace and the cells of
            // PopulateScreenProbesCells); elsewhere the pass is its section's row.
            const passesPerSection = new Map<string, number>();
            for (const name of Object.keys(setting.passes)) {
                const section = sectionOf(name);
                if (section !== null) passesPerSection.set(section, (passesPerSection.get(section) ?? 0) + 1);
            }
            for (const [name, ms] of Object.entries(setting.passes)) {
                const section = sectionOf(name);
                if (section !== null && (passesPerSection.get(section) ?? 0) > 1) rows.push({ scene, metric: `${prefix}/pass/${metricPart(name)}/gpu_ms_median`, value: ms });
            }
            for (const [name, ms] of Object.entries(setting.platformPasses)) rows.push({ scene, metric: `${prefix}/platform/${metricPart(name)}/gpu_ms_median`, value: ms });
        }
    }
    return rows;
}

const isAppleDevice = (device: string): boolean => /apple/i.test(device);

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, csv: { type: 'string' }, device: { type: 'string' }, commit: { type: 'string' }, browser: { type: 'string' }, 'clock-file': { type: 'string' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tools/gpu/pages/radiance-cache-timing/to-csv.ts <runner.json> [more.json ...] [--append] [--csv <file>] [--device <text>] [--commit <hash>] [--browser <text>] [--clock-file <file>]');
        process.exit(2);
    }
    const clockNotes = clockNoteRowsFromOption(values['clock-file']);
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][] }>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = values.device ?? deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            if (page.name !== PAGE) continue;
            if (page.outcome !== 'pass') {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const prefix = [summary.startedAt.slice(0, 10), commit, device, values.browser ?? 'Chrome headless'];
            const group = groups.get(device) ?? { prefix, runs: [] };
            group.runs.push(radianceCacheTimingRows(page.data as unknown as Record<string, SceneTiming>));
            groups.set(device, group);
        }
    }
    const linesByFile = new Map<string, string[]>();
    for (const { prefix, runs } of groups.values()) {
        const device = prefix[2];
        const rows = [...(runs.length === 1 ? runs[0] : repeatedRunRows(runs)), ...(isNvidiaDeviceName(device) ? clockNotes : [])];
        const note = isAppleDevice(device) ? APPLE_NOTE : '';
        const file = values.csv ? resolve(values.csv) : isAppleDevice(device) ? M5_CSV : RECORD_CSV;
        const lines = linesByFile.get(file) ?? [];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value), note].join(',')));
        linesByFile.set(file, lines);
    }
    if (linesByFile.size === 0) {
        console.error(`no passing ${PAGE} pages in those files`);
        process.exit(1);
    }
    for (const [file, lines] of linesByFile) {
        if (!values.append) {
            console.log([`# ${file}`, CSV_HEADER, ...lines].join('\n'));
            continue;
        }
        if (!existsSync(file)) await writeFile(file, `${CSV_HEADER}\n`);
        await appendFile(file, `${lines.join('\n')}\n`);
        console.error(`added ${lines.length} rows to ${file}`);
    }
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
