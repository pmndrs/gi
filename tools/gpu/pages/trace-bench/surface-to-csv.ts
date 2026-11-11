/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the surface-ray pages
 * (`trace-bench-surface-*`, RC5.4) into CSV rows:
 * `bench/results/surface-rays-m5.csv` for the Apple M5 (every row's note says
 * a Mac's timing is a hint), `bench/results/surface-rays.csv` for any other GPU.
 *
 *   node tools/gpu/pages/trace-bench/surface-to-csv.ts <runner.json> [more.json ...] [--append] [--csv <file>] [--device <text>] [--commit <hash>]
 *
 * Columns: date, commit, device, browser, scene, metric, value, note. Several
 * files of one scene and GPU are repeated runs: the timed rows come out as
 * `run1/`, `run2/`, ... plus `median/` rows; the counted walk's rows, which
 * are the same in every run, come out once, unprefixed. Metrics, per tree
 * layout and set: `surface/<tree>/<set>/loaded_mrays_per_s` and
 * `single_mrays_per_s` (timed), and the counted walk's means:
 * `steps`, `mesh_nodes`, `top_level_nodes`, `instance_entries`,
 * `instance_entries_without_hit`, `triangle_tests`, `origin_boxes`,
 * `steps_before_first_hit`, `instance_boxes_at_origin`, `simd32_efficiency`,
 * `workgroup64_efficiency`, `hit_fraction`, `median_hit_t_m`,
 * `self_hit_fraction`, `origin_triangle_tested_fraction`, `counted_agreement`
 * (the counted copy against the production kernel), `agreement_with_instanced`
 * (merged and split trees). The paired section (instanced and merged trees,
 * and the tie-break walks, timed in turn) gives
 * `surface/paired/<tree>/<set>/loaded_mrays_per_s_{median,min,max}_of_rounds`
 * and `surface/paired/ratio/<name>/{median,min,max}_of_rounds`.
 * Without --append the rows are printed.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { gitCommit } from '../../measured-commit.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from './csv-rows.ts';
import type { SurfaceRaysData, SurfaceSetResult } from './surface-rays-bench.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value,note';
const RESULTS = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results');
const M5_CSV = resolve(RESULTS, 'surface-rays-m5.csv');
const OTHER_CSV = resolve(RESULTS, 'surface-rays.csv');
const APPLE_NOTE = 'Mac GPU timing: a hint not the record; trace-bench-surface page (RC5.4); ours-tinybvh batch kernel; loaded = back to back for 0.8 s';

/** The counted walk's per-set values written as rows, by metric name. */
const COUNT_METRICS: [string, (set: SurfaceSetResult) => number][] = [
    ['steps', (set) => set.meanSteps],
    ['mesh_nodes', (set) => set.meanMeshNodes],
    ['top_level_nodes', (set) => set.meanTopLevelNodes],
    ['instance_entries', (set) => set.meanInstanceEntries],
    ['instance_entries_without_hit', (set) => set.meanInstanceEntriesWithoutHit],
    ['triangle_tests', (set) => set.meanTriangleTests],
    ['origin_boxes', (set) => set.meanOriginBoxes],
    ['steps_before_first_hit', (set) => set.meanStepsBeforeFirstHit],
    ['instance_boxes_at_origin', (set) => set.meanInstanceBoxesAtOrigin],
    ['simd32_efficiency', (set) => set.simd32Efficiency],
    ['workgroup64_efficiency', (set) => set.workgroup64Efficiency],
    ['hit_fraction', (set) => set.hitFraction],
    ['median_hit_t_m', (set) => set.medianHitT],
    ['self_hit_fraction', (set) => set.selfHitFraction],
    ['origin_triangle_tested_fraction', (set) => set.originTriangleTestedFraction],
    ['counted_agreement', (set) => set.countedAgreement],
    ['agreement_with_instanced', (set) => set.agreementWithInstanced ?? Number.NaN],
];

/** A run's rows: `timed` changes from run to run, `counted` doesn't (the walk's work depends only on the rays and trees). */
export function surfaceRaysRows(data: SurfaceRaysData): { timed: MetricRow[]; counted: MetricRow[] } {
    const scene = data.scene;
    const timed: MetricRow[] = [];
    const counted: MetricRow[] = [
        { scene, metric: 'surface/probes_placed', value: data.probesPlaced },
        { scene, metric: 'surface/grid_probes', value: data.gridProbes },
        { scene, metric: 'surface/region_grid_probes', value: data.regionGridProbes },
        { scene, metric: 'surface/instances', value: data.instanceCount },
    ];
    for (const set of data.sets) {
        const prefix = `surface/${set.tree}/${set.name}`;
        // The tie-break sets' rate is the paired section's median, so it is written there, not here.
        if (set.group !== 'tie-break') {
            timed.push({ scene, metric: `${prefix}/loaded_mrays_per_s`, value: set.loadedMrays }, { scene, metric: `${prefix}/single_mrays_per_s`, value: set.singleMrays });
        }
        for (const [name, read] of COUNT_METRICS) {
            const value = read(set);
            // No value (an instanced set has no agreement with itself; a tie-break set has no sampled boxes): no row.
            if (Number.isFinite(value)) counted.push({ scene, metric: `${prefix}/${name}`, value });
        }
    }
    for (const entry of data.paired?.entries ?? []) {
        const rates = [...entry.mraysPerRound].sort((a, b) => a - b);
        const prefix = `surface/paired/${entry.tree}/${entry.name}`;
        timed.push(
            { scene, metric: `${prefix}/loaded_mrays_per_s_median_of_rounds`, value: rates[rates.length >> 1] },
            { scene, metric: `${prefix}/loaded_mrays_per_s_min_of_rounds`, value: rates[0] },
            { scene, metric: `${prefix}/loaded_mrays_per_s_max_of_rounds`, value: rates[rates.length - 1] },
        );
        counted.push({ scene, metric: `${prefix}/agreement_with_production`, value: entry.agreementWithProduction });
    }
    for (const ratio of data.paired?.ratios ?? []) {
        timed.push(
            { scene, metric: `surface/paired/ratio/${ratio.name}/median_of_rounds`, value: ratio.median },
            { scene, metric: `surface/paired/ratio/${ratio.name}/min_of_rounds`, value: ratio.min },
            { scene, metric: `surface/paired/ratio/${ratio.name}/max_of_rounds`, value: ratio.max },
        );
    }
    if (data.paired) counted.push({ scene, metric: 'surface/paired/rounds', value: data.paired.rounds });
    return { timed, counted };
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, csv: { type: 'string' }, device: { type: 'string' }, commit: { type: 'string' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tools/gpu/pages/trace-bench/surface-to-csv.ts <runner.json> [more.json ...] [--append] [--csv <file>] [--device <text>] [--commit <hash>]');
        process.exit(2);
    }
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][]; counted: MetricRow[] }>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = values.device ?? deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            const data = page.data as unknown as SurfaceRaysData;
            if (data.benchmark !== 'surface-rays') continue;
            if (page.outcome !== 'pass') {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const prefix = [summary.startedAt.slice(0, 10), commit, device, browserFromUserAgent(data.userAgent, data.browserVersion)];
            const key = `${device}|${data.scene}`;
            const rows = surfaceRaysRows(data);
            const group = groups.get(key) ?? { prefix, runs: [], counted: rows.counted };
            group.runs.push(rows.timed);
            groups.set(key, group);
        }
    }
    const linesByFile = new Map<string, string[]>();
    for (const { prefix, runs, counted } of groups.values()) {
        const device = prefix[2];
        const apple = /apple/i.test(device);
        const rows = [...counted, ...(runs.length === 1 ? runs[0] : repeatedRunRows(runs))];
        const file = values.csv ? resolve(values.csv) : apple ? M5_CSV : OTHER_CSV;
        const lines = linesByFile.get(file) ?? [];
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value), apple ? APPLE_NOTE : ''].join(',')));
        linesByFile.set(file, lines);
    }
    if (linesByFile.size === 0) {
        console.error('no passing surface-ray pages in those files');
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
