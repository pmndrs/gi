/**
 * Device page: what does this browser and GPU support, and what do GPU
 * timestamps cost?
 *
 * Under the test runner it reports numbers in `data`. Opened by hand (on this
 * PC or on a phone through `pnpm device:serve`) it shows a table and a "copy"
 * box, and sends the same results to the LAN server.
 *
 * URL options: `?label=iPad` names the device in the results; `?rounds=`,
 * `?frames=` (frames per block) and `?scale=` (workload loop length factor)
 * change the run size.
 */

import { GpuTimer } from '../../../../packages/core/src/platform/timing/index.ts';
import { QUERY_TOKEN } from '../../contract.ts';
import { runGpuTest } from '../../client.ts';
import { collectEnvironment } from './environment.ts';
import { DEFAULT_MEASURE_OPTIONS, measureTimingOverhead, type MeasureOptions } from './measure.ts';
import { postResults, renderResultsPage, type DeviceResultsPayload, type DeviceRunData, type RoundedSummary } from './results.ts';
import { roundOrNull, type Summary } from './statistics.ts';
import { createWorkload, DEFAULT_WORKLOAD, type WorkloadConfig } from './workload.ts';

function positiveNumber(params: URLSearchParams, name: string, fallback: number): number {
    const value = Number(params.get(name));
    return Number.isFinite(value) && value > 0 ? value : fallback;
}

function roundSummary(summary: Summary): RoundedSummary {
    return {
        samples: summary.samples,
        median: roundOrNull(summary.median),
        mean: roundOrNull(summary.mean),
        min: roundOrNull(summary.min),
        max: roundOrNull(summary.max),
    };
}

const params = new URLSearchParams(location.search);
const underRunner = params.has(QUERY_TOKEN);
const label = params.get('label') ?? '';

const scale = positiveNumber(params, 'scale', 1);
const workloadConfig: WorkloadConfig = {
    ...DEFAULT_WORKLOAD,
    computeIterations: Math.round(DEFAULT_WORKLOAD.computeIterations * scale),
    renderIterations: Math.round(DEFAULT_WORKLOAD.renderIterations * scale),
};
const measureOptions: MeasureOptions = {
    ...DEFAULT_MEASURE_OPTIONS,
    rounds: Math.round(positiveNumber(params, 'rounds', DEFAULT_MEASURE_OPTIONS.rounds)),
    framesPerBlock: Math.round(positiveNumber(params, 'frames', DEFAULT_MEASURE_OPTIONS.framesPerBlock)),
};

// Collected first and separately: a browser without WebGPU must still say so.
const environment = await collectEnvironment();
let runData: DeviceRunData | null = null;

const report = await runGpuTest(
    async ({ adapter, device, log }) => {
        const timer = new GpuTimer(device, { warn: log });
        const workload = createWorkload(device, workloadConfig);
        const result = await measureTimingOverhead(device, workload, timer, measureOptions);
        workload.destroy();
        const timerStatus = timer.status;
        timer.dispose();

        const passGpuMs: Record<string, RoundedSummary> = {};
        for (const [name, summary] of Object.entries(result.passGpuMs)) passGpuMs[name] = roundSummary(summary);
        runData = {
            timestampsOnAdapter: adapter.features.has('timestamp-query'),
            timestampsOnDevice: device.features.has('timestamp-query'),
            timerStatus,
            workload: { ...workloadConfig },
            untimedFrameMs: roundSummary(result.untimedFrameMs),
            timedFrameMs: roundSummary(result.timedFrameMs),
            passGpuMs,
            overheadMs: result.overhead ? roundOrNull(result.overhead.overheadMs) : null,
            overheadRoundMinMs: result.overhead ? roundOrNull(result.overhead.roundMinMs) : null,
            overheadRoundMaxMs: result.overhead ? roundOrNull(result.overhead.roundMaxMs) : null,
            roundOverheadsMs: result.roundOverheadsMs.map((value) => roundOrNull(value)),
            overheadWithinNoise: result.overhead ? result.overhead.withinNoise : null,
            overheadBoundPercent: result.overhead ? roundOrNull(result.overhead.boundPercent, 1) : null,
            overheadText: result.overhead ? result.overhead.text : 'n/a (no timestamps)',
            timerResolutionMs: roundOrNull(result.timerResolutionMs, 4),
            timerCounters: { ...timer.counters },
            timedFramesWithoutResult: result.timedFramesWithoutResult,
        };
        log(`timer status: ${timerStatus}`);
        return { ...runData };
    },
    { optionalFeatures: ['timestamp-query'] },
);

const payload: DeviceResultsPayload = {
    label,
    collectedAt: new Date().toISOString(),
    environment,
    status: report.status,
    message: report.message,
    run: runData,
    deviceErrors: report.deviceErrors,
};

// The runner's server has no results endpoint, and a 404 would be logged as a page error.
const sendStatus = underRunner ? 'Running under the test runner; results go to the runner.' : await postResults(payload);
renderResultsPage(payload, sendStatus);
