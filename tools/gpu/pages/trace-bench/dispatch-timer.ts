/**
 * Times one compute dispatch on the GPU.
 *
 * Preferred: timestamp queries around the dispatch's own pass, through the
 * platform `GpuTimer` in strict mode (a missing or failed timer throws instead
 * of silently leaving a dispatch untimed). Without the `timestamp-query`
 * feature the fallback is wall time from submit to "queue drained", labelled
 * as such in the results.
 *
 * Every dispatch is timed alone: it is submitted, waited for, and read back
 * before the next starts, so no two timed passes overlap on the GPU.
 */

import { GpuTimer } from '../../../../packages/ddgi-core/src/platform/timing/index.ts';
import { median } from './statistics.ts';
import type { TimingSource, TimingSummary } from './results.ts';

export interface DispatchTimer {
    readonly source: TimingSource;
    /**
     * Records the dispatch with `encode` (which must pass `writes` to the
     * compute pass it begins), runs it, and returns its GPU milliseconds.
     */
    time(encode: (encoder: GPUCommandEncoder, writes: GPUComputePassTimestampWrites | undefined) => void): Promise<number>;
    dispose(): void;
}

const PASS_NAME = 'trace-bench/dispatch';

export async function createDispatchTimer(device: GPUDevice): Promise<DispatchTimer> {
    if (!device.features.has('timestamp-query')) return createWallClockTimer(device);

    const timer = new GpuTimer(device, { strict: true, slotCount: 2, maxPassesPerFrame: 1 });
    await timer.ready;
    if (timer.status !== 'ready') throw new Error(`GPU timer is ${timer.status}, so the benchmark cannot time on the GPU`);

    return {
        source: 'timestamp-query',
        async time(encode) {
            await timer.waitForFreeSlot();
            timer.beginFrame();
            const encoder = device.createCommandEncoder({ label: 'trace-bench dispatch' });
            encode(encoder, timer.computePass(PASS_NAME));
            timer.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            timer.readback();
            await timer.drain();
            const passes = timer.takeSamples().flatMap((sample) => sample.passes.filter((pass) => pass.name === PASS_NAME));
            if (passes.length !== 1) throw new Error(`expected one timed pass, the GPU timer returned ${passes.length}`);
            return passes[0].gpuMs;
        },
        dispose: () => timer.dispose(),
    };
}

function createWallClockTimer(device: GPUDevice): DispatchTimer {
    return {
        source: 'wall-clock',
        async time(encode) {
            const encoder = device.createCommandEncoder({ label: 'trace-bench dispatch' });
            encode(encoder, undefined);
            const start = performance.now();
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            return performance.now() - start;
        },
        dispose: () => {},
    };
}

/** Runs `timeOnce` for the warm-ups (discarded) and then the timed dispatches. */
export async function measureDispatches(
    timeOnce: () => Promise<number>,
    warmups: number,
    timed: number,
): Promise<TimingSummary> {
    for (let i = 0; i < warmups; i++) await timeOnce();
    const samplesMs: number[] = [];
    for (let i = 0; i < timed; i++) {
        const milliseconds = await timeOnce();
        if (!Number.isFinite(milliseconds) || milliseconds <= 0) throw new Error(`a dispatch reported ${milliseconds} ms`);
        samplesMs.push(milliseconds);
    }
    return {
        medianMs: median(samplesMs),
        minMs: Math.min(...samplesMs),
        maxMs: Math.max(...samplesMs),
        samplesMs,
    };
}
