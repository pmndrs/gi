/// <reference types="node" />
/**
 * Times `three-screen-inputs-cost` and adds what the page can't see: the GPU
 * clock and use in each timed section (nvidia-smi every 200 ms) and the bench
 * lock's verdict. Run it under the bench lock:
 *
 *   node tools/reference/bench-lock.ts run --what "rc1.1 screen inputs cost" -- node tools/gpu/pages/three-screen-inputs-cost/run.ts --gpu nvidia --csv
 *
 * With `--csv` it appends the rows to bench/results/screen-inputs.csv, with
 * the columns `date, commit, device, browser, adapter, scene, camera,
 * setting, metric, value, unit` (one row per number; `setting` is the
 * section's size). Per section: each difference's median, p10 and p90 (ms),
 * each variant's GPU ms per frame and CPU ms to record, the section's GPU
 * use and loaded SM clock, and the clock rule's notes from the lock (NVIDIA
 * only). Only rows whose run the lock calls VALID count as timings.
 *
 * Options:
 *   --gpu nvidia|intel      default nvidia (Intel has no nvidia-smi: no clock, no verdict)
 *   --out <file>            the combined JSON (default: the OS temp folder)
 *   --env KEY=VALUE         passed to the page (repeatable), e.g. --env VITE_DDGI_SECONDS_720=2
 *   --csv                   append the rows to bench/results/screen-inputs.csv
 */

import { existsSync } from 'node:fs';
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { CLOCK_FILE_ENV, clockNoteRows, describeTimingVerdict, isNvidiaDeviceName, readClockSummary } from '../../clock-note.ts';
import type { RunSummary } from '../../contract.ts';
import { gitCommit } from '../../measured-commit.ts';
import type { InterleavedTiming } from '../three-composite-shared/composite-check.ts';
import { GPU_BOUND_USE_PERCENT, printSection, runPage, sectionReport, startSampler, type SectionReport } from '../three-composite-shared/section-clock.ts';
import { browserFromUserAgent, deviceNameFromAdapter, formatValue } from '../trace-bench/csv-rows.ts';

const PAGE = 'three-screen-inputs-cost';
const CSV_FILE = resolve(import.meta.dirname, '..', '..', '..', '..', 'bench', 'results', 'screen-inputs.csv');
const CSV_HEADER = 'date,commit,device,browser,adapter,scene,camera,setting,metric,value,unit';
const SCENE = 'sponza';
const CAMERA = 'bookmark1';
const HEADLINE = 'filling every channel (fill-all - composite-mrt)';

/** A short metric name from a difference's label: the part in brackets. */
function deltaMetric(name: string): string {
    const inBrackets = /\(([^)]+)\)/.exec(name)?.[1] ?? name;
    return inBrackets.replace(/\s+-\s+/g, '_minus_').replace(/[^a-z0-9_]+/gi, '_');
}

interface Row {
    readonly setting: string;
    readonly metric: string;
    readonly value: number;
    readonly unit: string;
}

function sectionRows(report: SectionReport): Row[] {
    const setting = `${report.width}x${report.height}`;
    const rows: Row[] = [];
    for (const [name, delta] of Object.entries(report.deltas)) {
        const metric = `gpu_ms/${deltaMetric(name)}`;
        rows.push({ setting, metric: `${metric}/median`, value: delta.median, unit: 'ms' });
        rows.push({ setting, metric: `${metric}/p10`, value: delta.p10, unit: 'ms' });
        rows.push({ setting, metric: `${metric}/p90`, value: delta.p90, unit: 'ms' });
    }
    for (const [id, variant] of Object.entries(report.variants)) {
        rows.push({ setting, metric: `gpu_ms_per_frame/${id}`, value: variant.totalMs, unit: 'ms' });
        rows.push({ setting, metric: `cpu_ms_to_record/${id}`, value: variant.cpuMs, unit: 'ms' });
    }
    rows.push({ setting, metric: 'note/cycles', value: report.cycles, unit: 'count' });
    if (report.gpuUseMedianPercent !== null) rows.push({ setting, metric: 'note/gpu_use_median', value: report.gpuUseMedianPercent, unit: 'percent' });
    if (report.loadedSmMhzMedian !== null) rows.push({ setting, metric: 'note/sm_clock_mhz', value: report.loadedSmMhzMedian, unit: 'MHz' });
    if (report.gpuUseMedianPercent !== null) rows.push({ setting, metric: 'note/gpu_bound', value: report.gpuUseMedianPercent >= GPU_BOUND_USE_PERCENT ? 1 : 0, unit: 'flag' });
    return rows;
}

function csvField(text: string): string {
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            gpu: { type: 'string', default: 'nvidia' },
            out: { type: 'string' },
            env: { type: 'string', multiple: true },
            csv: { type: 'boolean', default: false },
        },
    });
    const gpu = values.gpu ?? 'nvidia';
    if (gpu !== 'nvidia' && gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const out = values.out ?? join(tmpdir(), `${PAGE}-${gpu}-${stamp}.json`);
    const pageJson = `${out}.page.json`;
    const env: Record<string, string> = {};
    for (const pair of values.env ?? []) {
        const split = pair.indexOf('=');
        if (split > 0) env[pair.slice(0, split)] = pair.slice(split + 1);
    }

    const sampler = gpu === 'nvidia' ? startSampler() : null;
    const code = await runPage(PAGE, gpu, pageJson, env);
    sampler?.stop();

    const result = JSON.parse(await readFile(pageJson, 'utf8')) as RunSummary & { commit?: string };
    const page = result.pages[0];
    const data = (page?.data ?? {}) as { timing?: { sections: InterleavedTiming[] }; userAgent?: string };
    const sections = (data.timing?.sections ?? []).map((section) => sectionReport(section, sampler?.samples ?? []));
    for (const section of sections) printSection(section);
    const clockFile = process.env[CLOCK_FILE_ENV];
    const lockSummary = clockFile ? readClockSummary(clockFile) : null;
    const lockVerdict = lockSummary ? describeTimingVerdict(lockSummary) : 'no bench-lock clock file: run this under `bench-lock run` for a verdict';
    console.log(`\nBench lock: ${lockVerdict}`);
    for (const section of sections) {
        const headline = section.deltas[HEADLINE];
        if (headline) console.log(`Filling FrameScreen at ${section.size}: ${headline.median.toFixed(3)} ms over the composite's own MRT (counts only if the lock says VALID and the section was GPU-bound).`);
    }

    if (values.csv) {
        if (page?.outcome !== 'pass') throw new Error(`${PAGE} did not pass (${page?.outcome}); no rows written`);
        const device = deviceNameFromAdapter(result.adapter);
        const browser = data.userAgent ? browserFromUserAgent(data.userAgent) : 'Chrome headless';
        const adapter = result.adapter ? `${result.adapter.vendor}/${result.adapter.architecture}` : 'unknown';
        const prefix = [result.startedAt.slice(0, 10), result.commit ?? gitCommit(), device, browser, adapter, SCENE, CAMERA];
        const rows: Row[] = sections.flatMap(sectionRows);
        // The clock rule's notes for the whole run (one lock run), NVIDIA only.
        if (lockSummary && isNvidiaDeviceName(device)) for (const note of clockNoteRows(lockSummary)) rows.push({ setting: 'all', metric: note.metric, value: note.value, unit: 'note' });
        const lines = rows.filter((row) => Number.isFinite(row.value)).map((row) => [...prefix, row.setting, row.metric, formatValue(row.value), row.unit].map(csvField).join(','));
        if (!existsSync(CSV_FILE)) await writeFile(CSV_FILE, `${CSV_HEADER}\n`);
        await appendFile(CSV_FILE, `${lines.join('\n')}\n`);
        console.log(`appended ${lines.length} rows to ${CSV_FILE}`);
    }
    await writeFile(out, `${JSON.stringify({ gpu, commit: result.commit, adapter: result.adapter, env, pageOutcome: page?.outcome, benchLock: { verdict: lockVerdict, summary: lockSummary }, sections, clockSamples: sampler?.samples.length ?? 0 }, null, 2)}\n`);
    console.log(`wrote ${out}`);
    process.exitCode = code;
}

await main();
