/**
 * Page-side helper for GPU test pages. Runs in the browser, not in Node.
 *
 *     import { runGpuTest, assert } from '../../client.ts';
 *
 *     await runGpuTest(async ({ device }) => {
 *         ...use the device, throw (or assert) when something is wrong...
 *         return { someNumber: 42 };   // optional: shown by the runner as data
 *     });
 *
 * It requests the adapter and device, makes sure uncaptured GPU errors are
 * recorded, and leaves a {@link GpuTestReport} on `window.__gpuTestReport` for the runner.
 * It also draws a one-screen summary, so opening the page by hand works too.
 */

import { QUERY_TOKEN, REPORT_PROPERTY, type AdapterSummary, type GpuTestReport } from './contract.ts';

export interface GpuTestContext {
    adapter: GPUAdapter;
    device: GPUDevice;
    adapterSummary: AdapterSummary;
    /** Shown in the summary and in the run output. Use for progress, not for pass/fail. */
    log(message: string): void;
}

export interface GpuTestOptions {
    /** The test fails if the device cannot be created with these. */
    requiredFeatures?: GPUFeatureName[];
    /** Requested only if the adapter has them; check `device.features` afterwards. */
    optionalFeatures?: GPUFeatureName[];
    requiredLimits?: Record<string, number>;
}

export function assert(condition: unknown, message: string): asserts condition {
    if (!condition) throw new Error(message);
}

/** GPUAdapterInfo fields are WebIDL getters; JSON.stringify(info) gives {}. */
export function summarizeAdapter(adapter: GPUAdapter): AdapterSummary {
    return summarizeAdapterInfo(adapter.info, adapter.features);
}

/** For a framework-owned device, which has `device.adapterInfo` but not the adapter. */
export function summarizeAdapterInfo(info: GPUAdapterInfo, features: Iterable<string> = []): AdapterSummary {
    return {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        isFallbackAdapter: info.isFallbackAdapter ?? false,
        features: [...features].sort(),
    };
}

/**
 * How long to wait after the queue drains for a late `uncapturederror` event.
 * Errors that arrive after this window are missed by the page (the runner's
 * own collection of the browser log may still see them).
 */
export const DRAIN_GRACE_MS = 50;

/**
 * Starts recording `uncapturederror` and device loss. Returns the live list of
 * messages: pass it to {@link reportGpuTest} as `deviceErrors`.
 * Uses addEventListener, never `device.onuncapturederror = ...`: a host
 * framework (three.js assigns that slot) may own it.
 */
export function watchDevice(device: GPUDevice): string[] {
    const errors: string[] = [];
    device.addEventListener('uncapturederror', (event) => {
        errors.push((event as GPUUncapturedErrorEvent).error.message);
    });
    void device.lost.then((info) => errors.push(`Device lost (${info.reason}): ${info.message}`));
    return errors;
}

/**
 * Waits until the queue has drained and error events have had a turn
 * ({@link DRAIN_GRACE_MS}). Validation errors surface asynchronously; without
 * this a failing shader could report a clean pass.
 */
export async function drainDevice(device: GPUDevice): Promise<void> {
    await device.queue.onSubmittedWorkDone();
    await new Promise((resolve) => setTimeout(resolve, DRAIN_GRACE_MS));
}

export interface GpuTestResult {
    status: 'pass' | 'fail';
    message?: string;
    /** Anything worth printing or recording. Typed arrays and bigints are converted for you. */
    data?: Record<string, unknown>;
    /** The adapter the test ran on. With a framework-owned device use `summarizeAdapterInfo(device.adapterInfo, device.features)`. */
    adapter?: GPUAdapter | AdapterSummary | null;
    deviceErrors?: string[];
    /** `performance.now()` at the start of the test, for the duration. */
    startTime?: number;
    /** Extra lines for the on-screen summary. */
    logLines?: string[];
}

/**
 * Low-level: publishes the report for the runner and draws the summary.
 * `runGpuTest` calls this. Use it directly when something else owns the
 * device (for example three's WebGPURenderer): do your own setup, call
 * {@link watchDevice} on its device, {@link drainDevice} at the end, then report.
 */
export function reportGpuTest(result: GpuTestResult): GpuTestReport {
    const adapter =
        result.adapter && 'info' in result.adapter ? summarizeAdapter(result.adapter) : (result.adapter ?? null);
    const deviceErrors = result.deviceErrors ?? [];
    let status = result.status;
    let message = result.message;
    if (status === 'pass' && deviceErrors.length > 0) {
        status = 'fail';
        message = deviceErrors.join(' | ');
    }
    const report: GpuTestReport = {
        token: new URLSearchParams(location.search).get(QUERY_TOKEN),
        status,
        message,
        data: toPlainData(result.data ?? {}),
        adapter,
        deviceErrors,
        durationMs: performance.now() - (result.startTime ?? performance.now()),
    };
    (window as unknown as Record<string, unknown>)[REPORT_PROPERTY] = report;
    renderSummary(report, result.logLines ?? []);
    return report;
}

/**
 * The runner reads the report with DevTools `returnByValue`, which turns a typed
 * array into a plain object keyed by index. Convert to real arrays first.
 */
function toPlainData(data: Record<string, unknown>): Record<string, unknown> {
    const plain = (_key: string, value: unknown): unknown => {
        if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
            return Array.from(value as unknown as ArrayLike<number | bigint>, (item) =>
                typeof item === 'bigint' ? item.toString() : item,
            );
        }
        return typeof value === 'bigint' ? value.toString() : value;
    };
    return JSON.parse(JSON.stringify(data, plain)) as Record<string, unknown>;
}

export async function runGpuTest(
    body: (context: GpuTestContext) => Promise<Record<string, unknown> | void>,
    options: GpuTestOptions = {},
): Promise<GpuTestReport> {
    const startTime = performance.now();
    const logLines: string[] = [];
    let deviceErrors: string[] = [];
    let adapterSummary: AdapterSummary | null = null;
    let status: GpuTestResult['status'] = 'pass';
    let message: string | undefined;
    let data: Record<string, unknown> = {};

    // An exception outside the body (an async callback, say) must fail the test too.
    const asyncProblems: string[] = [];
    window.addEventListener('error', (event) => asyncProblems.push(`Uncaught error: ${event.message}`));
    window.addEventListener('unhandledrejection', (event) =>
        asyncProblems.push(`Unhandled rejection: ${String(event.reason)}`),
    );

    try {
        const gpu = navigator.gpu;
        assert(gpu, 'WebGPU is not available in this browser (navigator.gpu is undefined).');
        const adapter = await gpu.requestAdapter();
        assert(adapter, 'WebGPU is available but requestAdapter() returned null (no usable GPU).');
        adapterSummary = summarizeAdapter(adapter);

        const requestedFeatures = [
            ...(options.requiredFeatures ?? []),
            ...(options.optionalFeatures ?? []).filter((feature) => adapter.features.has(feature)),
        ];
        const device = await adapter.requestDevice({
            requiredFeatures: requestedFeatures,
            requiredLimits: options.requiredLimits,
        });
        deviceErrors = watchDevice(device);

        const returned = await body({
            adapter,
            device,
            adapterSummary,
            log: (line) => logLines.push(line),
        });
        if (returned) data = returned;
        await drainDevice(device);
    } catch (error) {
        status = 'fail';
        message = error instanceof Error ? error.message : String(error);
    }

    if (status === 'pass' && asyncProblems.length > 0) {
        status = 'fail';
        message = asyncProblems.join(' | ');
    }

    return reportGpuTest({ status, message, data, adapter: adapterSummary, deviceErrors, startTime, logLines });
}

function renderSummary(report: GpuTestReport, logLines: string[]): void {
    const view = document.createElement('pre');
    view.style.cssText = 'font: 14px monospace; padding: 12px; color: #eee; background: #111; margin: 0;';
    const adapterText = report.adapter
        ? `${report.adapter.vendor} / ${report.adapter.architecture}`
        : 'no adapter';
    view.textContent = [
        `${document.title}: ${report.status.toUpperCase()}`,
        `adapter: ${adapterText}`,
        report.message ? `message: ${report.message}` : '',
        ...logLines,
        `data: ${JSON.stringify(report.data, null, 2)}`,
    ]
        .filter((line) => line !== '')
        .join('\n');
    document.body.style.cssText = `margin: 0; background: ${report.status === 'pass' ? '#0b3d17' : '#5a1111'};`;
    document.body.append(view);
}
