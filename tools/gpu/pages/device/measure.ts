/**
 * Runs the workload with and without GPU timestamps and compares.
 *
 * Each frame is encoded, submitted, and then waited for with
 * `queue.onSubmittedWorkDone()`. The wait makes the frame time cover the whole
 * GPU run, so extra GPU work or extra CPU work from timing shows up in it. It
 * also means the page does not depend on `requestAnimationFrame`, which
 * browsers pause in hidden tabs.
 *
 * Untimed and timed frames alternate in blocks, and the order flips every
 * round, so slow drift (clock speeds, thermals) hits both about equally.
 */

import { type GpuTimer } from '../../../../packages/ddgi-core/src/platform/timing/index.ts';
import { assessOverhead, measureTimerResolutionMs, type OverheadAssessment } from './noise.ts';
import { median, summarize, type Summary } from './statistics.ts';
import { type Workload } from './workload.ts';

export interface MeasureOptions {
    /** Frames to run before measuring, so shader compilation and clock ramp-up are not counted. */
    warmupFrames: number;
    /** Alternating untimed/timed blocks. */
    rounds: number;
    framesPerBlock: number;
}

export const DEFAULT_MEASURE_OPTIONS: MeasureOptions = { warmupFrames: 8, rounds: 6, framesPerBlock: 12 };

export interface MeasureResult {
    /** CPU-side time from encode to `onSubmittedWorkDone`, without timestamps. */
    untimedFrameMs: Summary;
    /** The same with timestamps, resolve and readback. Empty if timestamps are unavailable. */
    timedFrameMs: Summary;
    /** GPU milliseconds per pass name, over the timed frames. Empty if timestamps are unavailable. */
    passGpuMs: Record<string, Summary>;
    /** Timed minus untimed, with the scatter between rounds. Null if timestamps are unavailable. */
    overhead: OverheadAssessment | null;
    /** Per-round overheads the assessment is built from. */
    roundOverheadsMs: number[];
    timerResolutionMs: number;
    /** Timed frames whose results never arrived (all slots were busy). */
    timedFramesWithoutResult: number;
}

export async function measureTimingOverhead(
    device: GPUDevice,
    workload: Workload,
    timer: GpuTimer,
    options: MeasureOptions,
): Promise<MeasureResult> {
    await timer.ready;
    const timestampsWork = timer.status === 'ready';

    async function runFrame(timed: boolean): Promise<number> {
        // Waiting for a free slot happens outside the measured time, and means a timed frame is never dropped.
        if (timed) await timer.waitForFreeSlot();
        const start = performance.now();
        const encoder = device.createCommandEncoder({ label: timed ? 'timed frame' : 'untimed frame' });
        if (timed) timer.beginFrame();
        workload.encode(encoder, timed ? timer : null);
        if (timed) timer.resolve(encoder);
        device.queue.submit([encoder.finish()]);
        if (timed) timer.readback();
        await device.queue.onSubmittedWorkDone();
        return performance.now() - start;
    }

    for (let frame = 0; frame < options.warmupFrames; frame++) {
        await runFrame(false);
        if (timestampsWork) await runFrame(true);
    }
    await timer.drain();
    timer.takeSamples(); // discard warmup samples

    const untimedFrameTimes: number[] = [];
    const timedFrameTimes: number[] = [];
    const roundOverheadsMs: number[] = [];
    for (let round = 0; round < options.rounds; round++) {
        const untimedThisRound: number[] = [];
        const timedThisRound: number[] = [];
        const modes = timestampsWork ? [false, true] : [false];
        if (round % 2 === 1) modes.reverse();
        for (const timed of modes) {
            const times = timed ? timedThisRound : untimedThisRound;
            for (let frame = 0; frame < options.framesPerBlock; frame++) times.push(await runFrame(timed));
        }
        untimedFrameTimes.push(...untimedThisRound);
        timedFrameTimes.push(...timedThisRound);
        if (timestampsWork) roundOverheadsMs.push(median(timedThisRound) - median(untimedThisRound));
    }
    await timer.drain();

    const passTimes = new Map<string, number[]>();
    const samples = timer.takeSamples();
    for (const sample of samples) {
        for (const pass of sample.passes) {
            const times = passTimes.get(pass.name) ?? [];
            times.push(pass.gpuMs);
            passTimes.set(pass.name, times);
        }
    }
    const passGpuMs: Record<string, Summary> = {};
    for (const [name, times] of passTimes) passGpuMs[name] = summarize(times);

    const untimedFrameMs = summarize(untimedFrameTimes);
    const timedFrameMs = summarize(timedFrameTimes);
    const timerResolutionMs = measureTimerResolutionMs();
    return {
        untimedFrameMs,
        timedFrameMs,
        passGpuMs,
        overhead: timestampsWork ? assessOverhead(roundOverheadsMs, untimedFrameMs.median, timerResolutionMs) : null,
        roundOverheadsMs,
        timerResolutionMs,
        timedFramesWithoutResult: timedFrameTimes.length - samples.length,
    };
}
