/// <reference types="node" />
/**
 * For the run scripts of interleaved-timing pages (`three-composite-sponza`,
 * `three-screen-inputs-cost`): nvidia-smi sampled every 200 ms while the page
 * runs, stamped with the wall clock, and each timed section's GPU use and
 * clock from the samples inside its window. Moved out of
 * three-composite-sponza/run.ts (RC1.1) so both scripts share it.
 *
 * Not a page itself (no index.html).
 */

import { spawn } from 'node:child_process';
import { LOADED_UTILIZATION_PERCENT, NVIDIA_SMI_CLOCK_ARGS, parseClockLine, summarizeClockSamples, type ClockSample, type ClockSummary } from '../../clock-note.ts';
import type { InterleavedTiming, Spread } from './composite-check.ts';

/** The clock normalised to: ms × MHz ÷ this. */
export const REFERENCE_MHZ = 1500;
const SAMPLE_INTERVAL_MS = 200;
/** GPU use the timed section must hold, or the page wasn't GPU-bound. */
export const GPU_BOUND_USE_PERCENT = 90;

export interface StampedSample {
    readonly atMs: number;
    readonly sample: ClockSample;
}

/** Samples nvidia-smi every 200 ms until stopped; empty when nvidia-smi isn't there. */
export function startSampler(): { samples: StampedSample[]; stop(): void } {
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

/** Runs `pnpm test:gpu` on one page with extra environment variables; resolves with its exit code. */
export function runPage(page: string, gpu: string, jsonFile: string, env: Record<string, string>): Promise<number> {
    return new Promise((resolve) => {
        const child = spawn('pnpm', ['test:gpu', '--gpu', gpu, '--filter', page, '--json', jsonFile], {
            stdio: 'inherit',
            shell: true,
            env: { ...process.env, ...env },
        });
        child.on('exit', (code) => resolve(code ?? 1));
    });
}

export interface SectionReport {
    readonly label: string;
    readonly size: string;
    readonly width: number;
    readonly height: number;
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

export function sectionReport(section: InterleavedTiming, samples: readonly StampedSample[]): SectionReport {
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
        width: section.width,
        height: section.height,
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

export function printSection(report: SectionReport): void {
    const use = report.gpuUseMedianPercent === null ? 'no GPU samples' : `GPU use median ${report.gpuUseMedianPercent}% (min ${report.gpuUseMinPercent}%) over ${report.samples} samples, ${report.loadedSamples} at 70% or more`;
    const bound = report.gpuUseMedianPercent === null ? '' : report.gpuUseMedianPercent >= GPU_BOUND_USE_PERCENT ? '; GPU-bound' : `; NOT GPU-bound (under ${GPU_BOUND_USE_PERCENT}%)`;
    console.log(`\n${report.label} (${report.size}): ${report.cycles} cycles in ${report.seconds.toFixed(1)} s; ${use}${bound}; SM clock under load ${report.loadedSmMhzMedian ?? 'n/a'} MHz`);
    for (const [id, variant] of Object.entries(report.variants)) {
        const upscaler = variant.extraGpuMs === null ? '' : ` + ${variant.extraGpuMs.toFixed(3)} ms upscaler compute`;
        console.log(`  ${id.padEnd(30)} ${variant.totalMs.toFixed(3)} ms per frame on the GPU${upscaler}, ${variant.cpuMs.toFixed(2)} ms to record`);
    }
    for (const [name, delta] of Object.entries(report.deltas)) {
        const normalised = delta.normalisedMedian === null ? '' : `; at ${REFERENCE_MHZ} MHz ${delta.normalisedMedian.toFixed(3)} ms (p10 to p90 ${delta.normalisedP10!.toFixed(3)} to ${delta.normalisedP90!.toFixed(3)})`;
        console.log(`  ${name}: ${delta.median.toFixed(3)} ms (p10 to p90 ${delta.p10.toFixed(3)} to ${delta.p90.toFixed(3)}, MAD ${delta.mad.toFixed(3)}, thirds ${delta.thirds.map((ms) => ms.toFixed(3)).join('/')})${normalised}`);
    }
}
