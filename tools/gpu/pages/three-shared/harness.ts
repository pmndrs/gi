/**
 * Shared by the three-adapter GPU pages (T1.7): the device made the way the
 * adapter's host helper makes it (`requestGIDevice`), handed to three, with
 * uncaptured errors and pipeline creations counted, and the report written
 * for the runner. Like the T0.8 spike harness, but on the adapter's own code.
 *
 * Not a page itself (no index.html).
 */

import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import type { AdapterSummary } from '../../contract.ts';
import { createGIRenderer } from '../../../../packages/ddgi-three/src/host.ts';
import type { WebGPURenderer } from 'three/webgpu';

/** Render and compute pipelines created so far: a recompile shows up as a change. */
export interface PipelineCounts {
    render: number;
    compute: number;
}

export interface ThreePageContext {
    readonly device: GPUDevice;
    readonly renderer: WebGPURenderer;
    readonly adapterSummary: AdapterSummary;
    readonly pipelines: PipelineCounts;
    log(message: string): void;
    /** Numbers to report; filled as the page goes, so a later failure still reports them. */
    readonly data: Record<string, unknown>;
}

function countCalls(device: GPUDevice, methodName: keyof GPUDevice, increment: () => void): void {
    const original = device[methodName] as unknown as (...args: unknown[]) => unknown;
    (device as unknown as Record<string, unknown>)[methodName] = (...args: unknown[]) => {
        increment();
        return original.apply(device, args);
    };
}

/** Runs a three page: device and renderer by the adapter's helper, then `body`, a drain, and the report. */
export async function runThreePage(body: (context: ThreePageContext) => Promise<void>, size = 256): Promise<void> {
    const startTime = performance.now();
    const logLines: string[] = [];
    const data: Record<string, unknown> = {};
    let deviceErrors: string[] = [];
    let adapterSummary: AdapterSummary | null = null;
    let status: 'pass' | 'fail' = 'pass';
    let message: string | undefined;
    const asyncProblems: string[] = [];
    window.addEventListener('error', (event) => asyncProblems.push(`Uncaught error: ${event.message}`));
    window.addEventListener('unhandledrejection', (event) => asyncProblems.push(`Unhandled rejection: ${String(event.reason)}`));

    try {
        const { device, renderer } = await createGIRenderer({ renderer: { antialias: false } });
        // Free the GPU when the page goes (NOTES: release the device between pages).
        window.addEventListener('pagehide', () => device.destroy(), { once: true });
        deviceErrors = watchDevice(device);
        adapterSummary = summarizeAdapterInfo(device.adapterInfo, device.features);
        const pipelines: PipelineCounts = { render: 0, compute: 0 };
        countCalls(device, 'createRenderPipeline', () => pipelines.render++);
        countCalls(device, 'createRenderPipelineAsync', () => pipelines.render++);
        countCalls(device, 'createComputePipeline', () => pipelines.compute++);
        countCalls(device, 'createComputePipelineAsync', () => pipelines.compute++);
        renderer.setPixelRatio(1);
        renderer.setSize(size, size, false);
        document.body.append(renderer.domElement);
        await body({ device, renderer, adapterSummary, pipelines, log: (line) => logLines.push(line), data });
        await drainDevice(device);
    } catch (error) {
        status = 'fail';
        message = error instanceof Error ? error.message : String(error);
    }
    if (status === 'pass' && asyncProblems.length > 0) {
        status = 'fail';
        message = asyncProblems.join(' | ');
    }
    reportGpuTest({ status, message, data, adapter: adapterSummary, deviceErrors, startTime, logLines });
}

/** Runs frames of `step`, letting the GPU catch up every few so the page never sits in one long task. */
export async function runSteps(device: GPUDevice, count: number, step: (index: number) => void): Promise<void> {
    for (let index = 0; index < count; index++) {
        step(index);
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();
}
