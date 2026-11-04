/// <reference types="node" />
/**
 * Turns test-runner `--json` files from the `tlas-bench` page into rows for
 * `bench/results/bvh.csv` (columns as in bench/results/README.md).
 *
 *   node tools/gpu/pages/tlas-bench/to-csv.ts <runner.json> [more runner.json ...] [--append] [--commit <hash>]
 *
 * Several files of the same GPU are repeated runs: rows come out as `run1/`,
 * `run2/`, ... plus a `median/` row per metric, as for trace.csv. Without
 * --append the rows are printed. The commit defaults to the one the runner
 * recorded when the pages ran.
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { RunSummary } from '../../contract.ts';
import { gitCommit } from '../../measured-commit.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue, repeatedRunRows, type MetricRow } from '../trace-bench/csv-rows.ts';

const CSV_HEADER = 'date,commit,device,browser,scene,metric,value';
const DEFAULT_CSV = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'bvh.csv');

/** What the page reports (main.ts); only the fields that become rows. */
interface TlasBenchData {
    benchmark: 'tlas-bench';
    scene: string;
    instances: number;
    refitGpuMsMedian: number;
    refitGpuMsMax: number;
    updateCpuMsMedian: number;
    sceneEncodeCpuMsMedian: number;
    setTransformCpuMsMedian: number;
    shapeBuildMsFirst: number;
    shapeSnapshotMsFirst: number;
    qualityRebuildsWhileMoving: number;
    topLevelBytes: number;
    userAgent: string;
    browserVersion: string | null;
}

export function tlasBenchRows(data: TlasBenchData): MetricRow[] {
    const row = (metric: string, value: number): MetricRow => ({ scene: data.scene, metric, value });
    return [
        row('scene/instances', data.instances),
        row('tlas/refit_gpu_ms_median', data.refitGpuMsMedian),
        row('tlas/refit_gpu_ms_max', data.refitGpuMsMax),
        row('tlas/update_cpu_ms_median', data.updateCpuMsMedian),
        row('tlas/scene_upload_cpu_ms_median', data.sceneEncodeCpuMsMedian),
        row('tlas/set_transform_all_cpu_ms_median', data.setTransformCpuMsMedian),
        row('tlas/shape_build_worker_ms', data.shapeBuildMsFirst),
        row('tlas/shape_snapshot_cpu_ms', data.shapeSnapshotMsFirst),
        row('tlas/quality_rebuilds_60_frames', data.qualityRebuildsWhileMoving),
        row('tlas/memory_bytes', data.topLevelBytes),
    ];
}

async function main(): Promise<void> {
    const { values, positionals } = parseArgs({
        args: process.argv.slice(2),
        allowPositionals: true,
        options: { append: { type: 'boolean' }, commit: { type: 'string' } },
    });
    if (positionals.length === 0) {
        console.error('Usage: node tools/gpu/pages/tlas-bench/to-csv.ts <runner.json> [more.json ...] [--append] [--commit <hash>]');
        process.exit(2);
    }
    const groups = new Map<string, { prefix: string[]; runs: MetricRow[][] }>();
    for (const file of positionals) {
        const summary = JSON.parse(await readFile(file, 'utf8')) as RunSummary;
        const device = deviceNameFromAdapter(summary.adapter);
        for (const page of summary.pages) {
            const data = page.data as unknown as TlasBenchData;
            if (data.benchmark !== 'tlas-bench') continue;
            if (page.outcome !== 'pass') {
                console.error(`skipping ${page.name} in ${file}: ${page.outcome}`);
                continue;
            }
            const commit = values.commit ?? summary.commit ?? gitCommit();
            const prefix = [summary.startedAt.slice(0, 10), commit, device, browserFromUserAgent(data.userAgent, data.browserVersion)];
            const group = groups.get(device) ?? { prefix, runs: [] };
            group.runs.push(tlasBenchRows(data));
            groups.set(device, group);
        }
    }
    const lines: string[] = [];
    for (const { prefix, runs } of groups.values()) {
        const rows = runs.length === 1 ? runs[0] : repeatedRunRows(runs);
        lines.push(...rows.map((row) => [...prefix, row.scene, row.metric, formatValue(row.value)].join(',')));
    }
    if (lines.length === 0) {
        console.error('no passing tlas-bench pages in those files');
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
