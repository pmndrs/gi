/**
 * Times `three-composite-sponza` and adds what the page can't see: the GPU
 * clock during each timed section. Run it under the bench lock:
 *
 *   node tools/reference/bench-lock.ts run --what "three composite sponza" -- node tools/gpu/pages/three-composite-sponza/run.ts --gpu nvidia
 *
 * While the page runs, this samples `nvidia-smi` every 200 ms (the clock
 * rule's rate, DECISIONS 2026-11-05) and stamps each sample with the wall
 * clock. The page reports each timed section's wall-clock window; the
 * samples inside it give the section's GPU use (it must sit at 90% or more,
 * or the page wasn't GPU-bound and its milliseconds read long at a
 * low-power clock) and its SM clock median. Each difference is also given
 * clock-normalised: ms × the section's loaded SM clock median ÷ 1,500 MHz
 * (GPU time scales about inversely with the clock), so runs at different
 * clocks compare.
 *
 * Whether the timing counts is the bench lock's call alone (the clock rule:
 * a cool start and under 5% thermal samples, `judgeTiming`): this prints
 * its verdict from the clock file `bench-lock run` hands over
 * (`DDGI_CLOCK_FILE`), as it stands when the page has finished, and doesn't
 * judge on its own.
 *
 * Options:
 *   --gpu nvidia|intel      default nvidia (Intel has no nvidia-smi: no clock, no verdict)
 *   --out <file>            the combined JSON (default: the OS temp folder)
 *   --env KEY=VALUE         passed to the page (repeatable), e.g. --env VITE_DDGI_SECONDS=10
 *
 * By default the page runs with VITE_DDGI_SKIP=match and VITE_DDGI_GI_FRAMES=8
 * (a short warm-up: the shading cost doesn't depend on the field's values).
 */

import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { CLOCK_FILE_ENV, LOADED_UTILIZATION_PERCENT, NVIDIA_SMI_CLOCK_ARGS, describeTimingVerdict, parseClockLine, readClockSummary, summarizeClockSamples, type ClockSample, type ClockSummary } from '../../clock-note.ts';
import type { InterleavedTiming, Spread } from '../three-composite-shared/composite-check.ts';

/** The clock normalised to: ms × MHz ÷ this. */
const REFERENCE_MHZ = 1500;
const SAMPLE_INTERVAL_MS = 200;
/** The shading bar (DECISIONS 2026-11-05, T1.14), on the default path. */
const COST_BAR_MS = 1.0;
const DEFAULT_PATH_DELTA = 'showing GI, default path (composite + GTAO full) - off';

interface StampedSample {
    readonly atMs: number;
    readonly sample: ClockSample;
}

/** Samples nvidia-smi every 200 ms until stopped; empty when nvidia-smi isn't there. */
function startSampler(): { samples: StampedSample[]; stop(): void } {
    const samples: StampedSample[] = [];
    const child = spawn('nvidia-smi', [...NVIDIA_SMI_CLOCK_ARGS, `--loop-ms=${SAMPLE_INTERVAL_MS}`], { stdio: ['ignore', 'pipe', 'ignore'] });
    child.on('error', () => {});
    let pending = '';
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', (chunk: string) => {
        pending += chunk;
        const lines = pending.split(/\r?\n/);
        pending = lines.pop() ?? '';
        for (const line of lines) {
            const sample = parseClockLine(line);
            if (sample) samples.push({ atMs: Date.now(), sample });
        }
    });
    return { samples, stop: () => child.kill() };
}

function runPage(gpu: string, jsonFile: string, env: Record<string, string>): Promise<number> {
    return new Promise((resolve) => {
        const child = spawn('pnpm', ['test:gpu', '--gpu', gpu, '--filter', 'three-composite-sponza', '--json', jsonFile], {
            stdio: 'inherit',
            shell: true,
            env: { ...process.env, ...env },
        });
        child.on('exit', (code) => resolve(code ?? 1));
    });
}

/** GPU use the timed section must hold, or the page wasn't GPU-bound. */
const GPU_BOUND_USE_PERCENT = 90;

interface SectionReport {
    readonly label: string;
    readonly size: string;
    readonly seconds: number;
    readonly cycles: number;
    readonly clock: ClockSummary | null;
    /** nvidia-smi samples inside the section, and their GPU use. */
    readonly samples: number;
    readonly gpuUseMedianPercent: number | null;
    readonly gpuUseMinPercent: number | null;
    /** Samples at or above the bench lock's "loaded" use (70%). */
    readonly loadedSamples: number;
    readonly loadedSmMhzMedian: number | null;
    readonly deltas: Record<string, Spread & { readonly normalisedMedian: number | null; readonly normalisedP10: number | null; readonly normalisedP90: number | null }>;
    readonly variants: InterleavedTiming['variants'];
}

function sectionReport(section: InterleavedTiming, samples: readonly StampedSample[]): SectionReport {
    const inside = samples.filter((entry) => entry.atMs >= section.windowStartMs && entry.atMs <= section.windowEndMs).map((entry) => entry.sample);
    const clock = summarizeClockSamples(inside);
    const use = inside.map((sample) => sample.utilizationPercent).sort((a, b) => a - b);
    const mhz = clock?.smMhzLoaded?.median ?? null;
    const normalise = (ms: number) => (mhz === null ? null : (ms * mhz) / REFERENCE_MHZ);
    const deltas: SectionReport['deltas'] = {};
    for (const [name, spread] of Object.entries(section.deltas)) {
        deltas[name] = { ...spread, normalisedMedian: normalise(spread.median), normalisedP10: normalise(spread.p10), normalisedP90: normalise(spread.p90) };
    }
    return {
        label: section.label,
        size: `${section.width} x ${section.height}`,
        seconds: (section.windowEndMs - section.windowStartMs) / 1000,
        cycles: section.cycles,
        clock,
        samples: inside.length,
        gpuUseMedianPercent: use.length > 0 ? use[Math.floor(use.length / 2)] : null,
        gpuUseMinPercent: use.length > 0 ? use[0] : null,
        loadedSamples: use.filter((percent) => percent >= LOADED_UTILIZATION_PERCENT).length,
        loadedSmMhzMedian: mhz,
        deltas,
        variants: section.variants,
    };
}

function printSection(report: SectionReport): void {
    const use = report.gpuUseMedianPercent === null ? 'no GPU samples' : `GPU use median ${report.gpuUseMedianPercent}% (min ${report.gpuUseMinPercent}%) over ${report.samples} samples, ${report.loadedSamples} at 70% or more`;
    const bound = report.gpuUseMedianPercent === null ? '' : report.gpuUseMedianPercent >= GPU_BOUND_USE_PERCENT ? '; GPU-bound' : `; NOT GPU-bound (under ${GPU_BOUND_USE_PERCENT}%)`;
    console.log(`\n${report.label} (${report.size}): ${report.cycles} cycles in ${report.seconds.toFixed(1)} s; ${use}${bound}; SM clock under load ${report.loadedSmMhzMedian ?? 'n/a'} MHz`);
    for (const [id, variant] of Object.entries(report.variants)) console.log(`  ${id.padEnd(20)} ${variant.totalMs.toFixed(3)} ms per frame on the GPU, ${variant.cpuMs.toFixed(2)} ms to record`);
    for (const [name, delta] of Object.entries(report.deltas)) {
        const normalised = delta.normalisedMedian === null ? '' : `; at ${REFERENCE_MHZ} MHz ${delta.normalisedMedian.toFixed(3)} ms (p10 to p90 ${delta.normalisedP10!.toFixed(3)} to ${delta.normalisedP90!.toFixed(3)})`;
        console.log(`  ${name}: ${delta.median.toFixed(3)} ms (p10 to p90 ${delta.p10.toFixed(3)} to ${delta.p90.toFixed(3)}, MAD ${delta.mad.toFixed(3)}, thirds ${delta.thirds.map((ms) => ms.toFixed(3)).join('/')})${normalised}`);
    }
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            gpu: { type: 'string', default: 'nvidia' },
            out: { type: 'string' },
            env: { type: 'string', multiple: true },
        },
    });
    const gpu = values.gpu ?? 'nvidia';
    if (gpu !== 'nvidia' && gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const out = values.out ?? join(tmpdir(), `three-composite-sponza-${gpu}-${stamp}.json`);
    const pageJson = `${out}.page.json`;
    const env: Record<string, string> = { VITE_DDGI_SKIP: 'match', VITE_DDGI_GI_FRAMES: '8' };
    for (const pair of values.env ?? []) {
        const split = pair.indexOf('=');
        if (split > 0) env[pair.slice(0, split)] = pair.slice(split + 1);
    }

    const sampler = gpu === 'nvidia' ? startSampler() : null;
    const code = await runPage(gpu, pageJson, env);
    sampler?.stop();

    const result = JSON.parse(await readFile(pageJson, 'utf8')) as { pages: Array<{ outcome: string; data: { timing?: { sections: InterleavedTiming[]; targets: string } } }>; adapter: unknown; commit: string };
    const page = result.pages[0];
    const sections = (page?.data.timing?.sections ?? []).map((section) => sectionReport(section, sampler?.samples ?? []));
    for (const section of sections) printSection(section);
    // The bench lock's verdict on the run so far (it prints the final one itself when the lock is released).
    const clockFile = process.env[CLOCK_FILE_ENV];
    const lockSummary = clockFile ? readClockSummary(clockFile) : null;
    const lockVerdict = lockSummary ? describeTimingVerdict(lockSummary) : 'no bench-lock clock file: run this under `bench-lock run` for a verdict';
    console.log(`\nBench lock: ${lockVerdict}`);
    const main1440 = sections[0];
    if (main1440) {
        const cost = main1440.deltas[DEFAULT_PATH_DELTA];
        const gpuBound = main1440.gpuUseMedianPercent !== null && main1440.gpuUseMedianPercent >= GPU_BOUND_USE_PERCENT;
        console.log(
            `Showing GI on the default path, ${main1440.label}: ${cost.median.toFixed(3)} ms (bar ${COST_BAR_MS} ms; ${cost.median <= COST_BAR_MS ? 'within' : 'over'}). ` +
                `It counts only if the bench lock says VALID${gpuBound ? '' : ' and the section is GPU-bound, which it was not'}.`,
        );
    }
    await writeFile(
        out,
        `${JSON.stringify({ gpu, commit: result.commit, adapter: result.adapter, env, pageOutcome: page?.outcome, targets: page?.data.timing?.targets, benchLock: { verdict: lockVerdict, summary: lockSummary }, sections, clockSamples: sampler?.samples.length ?? 0 }, null, 2)}\n`,
    );
    console.log(`wrote ${out}`);
    process.exitCode = code;
}

await main();
