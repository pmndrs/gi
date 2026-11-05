/**
 * Times `three-composite-sponza` and adds what the page can't see: the GPU
 * clock during each timed section. Run it under the bench lock:
 *
 *   node tools/reference/bench-lock.ts run --what "three composite sponza" -- node tools/gpu/pages/three-composite-sponza/run.ts --gpu nvidia
 *
 * While the page runs, this samples `nvidia-smi` every 200 ms (the clock
 * rule's rate, DECISIONS 2026-11-05) and stamps each sample with the wall
 * clock. The page reports each timed section's wall-clock window; the
 * samples inside it give the section's SM clock median, its thermal-throttle
 * share and the clock rule's verdict (`judgeTiming`, with the temperature at
 * the section's start). Each difference is also given clock-normalised:
 * ms × the section's loaded SM clock median ÷ 1,500 MHz (GPU time scales
 * about inversely with the clock), so runs at different clocks compare.
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
import { NVIDIA_SMI_CLOCK_ARGS, judgeTiming, parseClockLine, summarizeClockSamples, type ClockSample, type ClockSummary } from '../../clock-note.ts';
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

interface SectionReport {
    readonly size: string;
    readonly seconds: number;
    readonly cycles: number;
    readonly clock: ClockSummary | null;
    readonly verdict: string;
    readonly loadedSmMhzMedian: number | null;
    readonly deltas: Record<string, Spread & { readonly normalisedMedian: number | null; readonly normalisedP10: number | null; readonly normalisedP90: number | null }>;
    readonly variants: InterleavedTiming['variants'];
}

function sectionReport(section: InterleavedTiming, samples: readonly StampedSample[]): SectionReport {
    const inside = samples.filter((entry) => entry.atMs >= section.windowStartMs && entry.atMs <= section.windowEndMs).map((entry) => entry.sample);
    const before = samples.filter((entry) => entry.atMs <= section.windowStartMs).at(-1)?.sample ?? inside[0];
    let clock = summarizeClockSamples(inside);
    if (clock && before?.temperatureC !== undefined) clock = { ...clock, start: { temperatureC: before.temperatureC, coolBelowC: null } as ClockSummary['start'] };
    const verdict = clock ? judgeTiming(clock) : { kind: 'no-verdict', reason: 'no clock samples (no nvidia-smi)' };
    const mhz = clock?.smMhzLoaded?.median ?? null;
    const normalise = (ms: number) => (mhz === null ? null : (ms * mhz) / REFERENCE_MHZ);
    const deltas: SectionReport['deltas'] = {};
    for (const [name, spread] of Object.entries(section.deltas)) {
        deltas[name] = { ...spread, normalisedMedian: normalise(spread.median), normalisedP10: normalise(spread.p10), normalisedP90: normalise(spread.p90) };
    }
    return {
        size: `${section.width} x ${section.height}`,
        seconds: (section.windowEndMs - section.windowStartMs) / 1000,
        cycles: section.cycles,
        clock,
        verdict: `${verdict.kind}: ${verdict.reason}`,
        loadedSmMhzMedian: mhz,
        deltas,
        variants: section.variants,
    };
}

function printSection(report: SectionReport): void {
    console.log(`\n${report.size}: ${report.cycles} cycles in ${report.seconds.toFixed(1)} s; SM clock under load ${report.loadedSmMhzMedian ?? 'n/a'} MHz; timing ${report.verdict}`);
    for (const [id, variant] of Object.entries(report.variants)) console.log(`  ${id.padEnd(20)} ${variant.totalMs.toFixed(3)} ms per frame`);
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
    const main1440 = sections[0];
    if (main1440) {
        const cost = main1440.deltas[DEFAULT_PATH_DELTA];
        const valid = main1440.verdict.startsWith('valid');
        console.log(
            `\nShowing GI on the default path at ${main1440.size}: ${cost.median.toFixed(3)} ms (bar ${COST_BAR_MS} ms): ${valid ? (cost.median <= COST_BAR_MS ? 'within the bar' : 'over the bar') : 'no judgement, the timing is not valid'}.`,
        );
    }
    await writeFile(out, `${JSON.stringify({ gpu, commit: result.commit, adapter: result.adapter, env, pageOutcome: page?.outcome, targets: page?.data.timing?.targets, sections, clockSamples: sampler?.samples.length ?? 0 }, null, 2)}\n`);
    console.log(`wrote ${out}`);
    process.exitCode = code;
}

await main();
