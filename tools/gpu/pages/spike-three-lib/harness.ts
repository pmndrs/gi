/**
 * Shared setup for the T0.8 spike pages (three.js sharing).
 *
 * Three facts shape it:
 *
 * 1. Our code, not three, creates the GPUDevice, with raised limits. The
 *    renderer then receives it as `new WebGPURenderer({ device })`. That is
 *    the "host owns the device" rule the platform follows.
 * 2. three owns the device afterwards, so the page cannot use `runGpuTest`
 *    (which also creates a device). It uses the lower-level helpers instead:
 *    `watchDevice` (an `addEventListener('uncapturederror')`, never the
 *    `onuncapturederror` slot that three overwrites), `drainDevice`, and
 *    `reportGpuTest`.
 * 3. The pages count shader-module and pipeline creations by wrapping the
 *    device's create methods, because "how many recompiles?" is a question
 *    this spike has to answer with numbers.
 */

import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import type { AdapterSummary } from '../../contract.ts';

/**
 * Limits we ask for above the WebGPU defaults. Each is clamped to what the
 * adapter offers, so the request never fails; the page reports what it got.
 * The storage-buffer numbers matter for the Meshlet pipelines, which bind many
 * buffers per stage.
 */
export const RAISED_LIMIT_TARGETS: Readonly<Record<string, number>> = {
    maxStorageBuffersPerShaderStage: 16,
    maxSampledTexturesPerShaderStage: 32,
    maxSamplersPerShaderStage: 16,
    maxStorageBufferBindingSize: 1024 * 1024 * 1024,
    maxBufferSize: 1024 * 1024 * 1024,
    maxBindGroups: 8,
    maxColorAttachmentBytesPerSample: 64,
};

/** How many GPU objects the page created, so the spike can count recompiles. */
export interface CreationCounts {
    shaderModules: number;
    renderPipelines: number;
    computePipelines: number;
}

export interface SpikeDevice {
    adapter: GPUAdapter;
    device: GPUDevice;
    /** Live counters; read `snapshotCounts()` before and after an action for a delta. */
    counts: CreationCounts;
    /** The limits we raised, with the default and the granted value. */
    limits: Array<{ name: string; defaultValue: number; granted: number }>;
}

export function snapshotCounts(counts: CreationCounts): CreationCounts {
    return { ...counts };
}

export function diffCounts(after: CreationCounts, before: CreationCounts): CreationCounts {
    return {
        shaderModules: after.shaderModules - before.shaderModules,
        renderPipelines: after.renderPipelines - before.renderPipelines,
        computePipelines: after.computePipelines - before.computePipelines,
    };
}

/**
 * Wraps a method of the device so every call increments a counter. Async
 * variants count too, since three uses `createRenderPipelineAsync` when it
 * compiles ahead of time.
 */
function countCalls(device: GPUDevice, methodName: keyof GPUDevice, increment: () => void): void {
    const original = device[methodName] as unknown as (...args: unknown[]) => unknown;
    (device as unknown as Record<string, unknown>)[methodName] = (...args: unknown[]) => {
        increment();
        return original.apply(device, args);
    };
}

export async function createRaisedLimitsDevice(): Promise<SpikeDevice> {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('requestAdapter() returned null (no usable GPU).');

    // Ask for the largest limit the adapter allows, up to our target.
    const requiredLimits: Record<string, number> = {};
    const limits: SpikeDevice['limits'] = [];
    const adapterLimits = adapter.limits as unknown as Record<string, number>;
    for (const [name, target] of Object.entries(RAISED_LIMIT_TARGETS)) {
        const available = adapterLimits[name];
        if (typeof available !== 'number') continue;
        requiredLimits[name] = Math.min(target, available);
    }

    // three itself requests every feature the adapter has when it creates the
    // device. A device we create must offer the same, or three would run in a
    // reduced mode (it reads `device.features` at init).
    const device = await adapter.requestDevice({
        requiredFeatures: [...adapter.features] as GPUFeatureName[],
        requiredLimits,
    });

    const grantedLimits = device.limits as unknown as Record<string, number>;
    for (const name of Object.keys(requiredLimits)) {
        limits.push({
            name,
            // The default is whatever a plain requestDevice() would have granted; the spec values are listed in the WebGPU spec.
            defaultValue: DEFAULT_LIMITS[name] ?? Number.NaN,
            granted: grantedLimits[name],
        });
    }

    const counts: CreationCounts = { shaderModules: 0, renderPipelines: 0, computePipelines: 0 };
    countCalls(device, 'createShaderModule', () => counts.shaderModules++);
    countCalls(device, 'createRenderPipeline', () => counts.renderPipelines++);
    countCalls(device, 'createRenderPipelineAsync', () => counts.renderPipelines++);
    countCalls(device, 'createComputePipeline', () => counts.computePipelines++);
    countCalls(device, 'createComputePipelineAsync', () => counts.computePipelines++);

    return { adapter, device, counts, limits };
}

/** WebGPU spec defaults for the limits we raise. */
const DEFAULT_LIMITS: Readonly<Record<string, number>> = {
    maxStorageBuffersPerShaderStage: 8,
    maxSampledTexturesPerShaderStage: 16,
    maxSamplersPerShaderStage: 16,
    maxStorageBufferBindingSize: 128 * 1024 * 1024,
    maxBufferSize: 256 * 1024 * 1024,
    maxBindGroups: 4,
    maxColorAttachmentBytesPerSample: 32,
};

export interface SpikeContext extends SpikeDevice {
    adapterSummary: AdapterSummary;
    log(message: string): void;
    /**
     * Numbers and answers to print. Fill it as you go: if a later check throws,
     * what was recorded so far still reaches the report.
     */
    data: Record<string, unknown>;
}

/**
 * Runs a spike page: creates the device, runs `body`, waits for late
 * validation errors and publishes the report for the runner. `body` records
 * results in `context.data` and throws (or uses `assert`) to fail.
 */
export async function runSpike(body: (context: SpikeContext) => Promise<void>): Promise<void> {
    const startTime = performance.now();
    const logLines: string[] = [];
    let deviceErrors: string[] = [];
    let adapterSummary: AdapterSummary | null = null;
    let status: 'pass' | 'fail' = 'pass';
    let message: string | undefined;
    const data: Record<string, unknown> = {};

    // Anything thrown inside a callback (not awaited by `body`) must fail the page too.
    const asyncProblems: string[] = [];
    window.addEventListener('error', (event) => asyncProblems.push(`Uncaught error: ${event.message}`));
    window.addEventListener('unhandledrejection', (event) =>
        asyncProblems.push(`Unhandled rejection: ${String(event.reason)}`),
    );

    try {
        const spikeDevice = await createRaisedLimitsDevice();
        const { device } = spikeDevice;
        deviceErrors = watchDevice(device);
        adapterSummary = summarizeAdapterInfo(device.adapterInfo, device.features);
        data.limits = spikeDevice.limits;
        await body({
            ...spikeDevice,
            adapterSummary,
            log: (line) => logLines.push(line),
            data,
        });
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
