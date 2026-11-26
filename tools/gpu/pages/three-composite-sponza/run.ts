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

import { readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { CLOCK_FILE_ENV, describeTimingVerdict, readClockSummary } from '../../clock-note.ts';
import type { InterleavedTiming } from '../three-composite-shared/composite-check.ts';
import { GPU_BOUND_USE_PERCENT, printSection, runPage, sectionReport, startSampler } from '../three-composite-shared/section-clock.ts';

/** The shading bar (DECISIONS 2026-11-05, T1.14), on the default path. */
const COST_BAR_MS = 1.0;
const DEFAULT_PATH_DELTA = 'showing GI, default path (composite + GTAO full) - off';

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
    const code = await runPage('three-composite-sponza', gpu, pageJson, env);
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
    const cost = main1440?.deltas[DEFAULT_PATH_DELTA];
    if (main1440 && cost) {
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
