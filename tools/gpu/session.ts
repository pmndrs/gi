/**
 * One Chrome instance, one tab. Checks WebGPU and the adapter, then runs test
 * pages in it one after another, collecting browser messages and screenshots.
 */

import { mkdtempSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { CdpClient } from './cdp.ts';
import {
    discoverGpuLuid,
    findBrowser,
    removeProfile,
    startChrome,
} from './chrome.ts';
import { registerCleanup, shutdownSignal, throwIfShuttingDown } from './cleanup.ts';
import {
    QUERY_GPU,
    QUERY_TOKEN,
    REPORT_PROPERTY,
    type AdapterSummary,
    type GpuChoice,
    type GpuTestReport,
} from './contract.ts';
import { adapterMismatch, evaluatePage, type BrowserMessage, type PageVerdict } from './evaluate.ts';
import { type TestPage } from './pages.ts';

/** WebGPU cannot be used at all: no `navigator.gpu`, or no adapter. */
export class WebGpuUnavailableError extends Error {}

/** WebGPU works, but on a GPU other than the one asked for. */
export class WrongAdapterError extends Error {}

export interface SessionOptions {
    serverOrigin: string;
    chromePath: string | null;
    gpu: GpuChoice | null;
    /** Start Chrome with the GPU disabled (a self-check of the failure path). */
    withoutGpu: boolean;
}

export interface PageResult {
    page: TestPage;
    verdict: PageVerdict;
    report: GpuTestReport | null;
    messages: BrowserMessage[];
    durationMs: number;
    screenshotPath: string | null;
}

export interface PageRunOptions {
    timeoutMs: number;
    /** Judge the page as a self-test: it passes only if the runner catches a failure. */
    expectFailure: boolean;
    screenshotDir: string;
    /** Prefix for the screenshot file name, for example the requested GPU. */
    gpuLabel: string;
}

/** How often to look for the page's report. Short, so a fast test is not padded with waiting. */
const POLL_INTERVAL_MS = 50;

/**
 * After the report appears, wait this long before judging the page. The page's own
 * drain window is 50 ms, and Chrome forwards its log entries to us a little later.
 */
const LATE_MESSAGE_GRACE_MS = 100;

/** A DevTools HTTP request to a local Chrome should answer at once. */
const DEVTOOLS_REQUEST_TIMEOUT_MS = 10_000;

/** Page.captureScreenshot on a busy GPU can be slow, but not this slow. */
const SCREENSHOT_TIMEOUT_MS = 10_000;

export class BrowserSession {
    private readonly cdp: CdpClient;
    private readonly options: SessionOptions;
    private readonly closeResources: () => Promise<void>;
    private messages: BrowserMessage[] = [];
    private crashed = false;

    private constructor(cdp: CdpClient, options: SessionOptions, closeResources: () => Promise<void>) {
        this.cdp = cdp;
        this.options = options;
        this.closeResources = closeResources;
        this.listenToBrowser();
    }

    static async launch(options: SessionOptions): Promise<BrowserSession> {
        throwIfShuttingDown();
        const executable = findBrowser(options.chromePath);
        // Only Intel needs pinning: headless Chrome gives the discrete NVIDIA GPU by default
        // (and the adapter check in preflight() fails the run if it ever stops doing so).
        const luid =
            options.gpu === 'intel' ? await discoverGpuLuid('intel', shutdownSignal) : null;

        throwIfShuttingDown(); // Ctrl-C may have arrived during the GPU lookup above
        // Create the profile and start Chrome synchronously, and register the cleanup in the
        // same breath: no moment exists where Chrome runs and Ctrl-C could not stop it.
        const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-gpu-chrome-'));
        const chrome = startChrome(executable, { profileDir, luid, withoutGpu: options.withoutGpu });
        let cdp: CdpClient | null = null;
        const closeResources = registerCleanup(async () => {
            cdp?.close();
            await chrome.stop();
            await removeProfile(profileDir);
        });

        try {
            await chrome.ready();
            const targetsResponse = await fetch(`${chrome.devtoolsOrigin}/json/list`, {
                signal: AbortSignal.timeout(DEVTOOLS_REQUEST_TIMEOUT_MS),
            });
            const targets = (await targetsResponse.json()) as Array<{ type: string; webSocketDebuggerUrl: string }>;
            const pageTarget = targets.find((target) => target.type === 'page');
            if (!pageTarget) throw new Error('Chrome started without a page target.');
            cdp = await CdpClient.connect(pageTarget.webSocketDebuggerUrl);
            await Promise.all([
                cdp.send('Page.enable'),
                cdp.send('Runtime.enable'),
                cdp.send('Log.enable'),
                cdp.send('Inspector.enable'),
            ]);
            return new BrowserSession(cdp, options, closeResources);
        } catch (error) {
            await closeResources();
            throw error;
        }
    }

    /**
     * Opens an empty page and asks for an adapter, before any test page runs.
     * Throws a {@link WebGpuUnavailableError} or {@link WrongAdapterError} with a plain message.
     */
    async preflight(): Promise<AdapterSummary> {
        await this.navigate(`${this.options.serverOrigin}/preflight.html`);
        // This inline script cannot import client.ts, so its adapter summary duplicates summarizeAdapter().
        const probe = await this.cdp.evaluate<{ problem: 'no-gpu-object' | 'no-adapter' } | { adapter: AdapterSummary }>(`
            (async () => {
                if (!navigator.gpu) return { problem: 'no-gpu-object' };
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) return { problem: 'no-adapter' };
                const { vendor, architecture, device, description } = adapter.info;
                return { adapter: { vendor, architecture, device, description,
                    isFallbackAdapter: adapter.info.isFallbackAdapter ?? false,
                    features: [...adapter.features].sort() } };
            })()`);

        if ('problem' in probe) {
            const reason =
                probe.problem === 'no-gpu-object'
                    ? 'navigator.gpu is undefined (the browser has no WebGPU, or the page is not a secure context)'
                    : 'navigator.gpu.requestAdapter() returned null (no usable GPU, or the GPU is disabled)';
            throw new WebGpuUnavailableError(`WebGPU is not available: ${reason}.`);
        }
        const mismatch = adapterMismatch(this.options.gpu, probe.adapter);
        if (mismatch) throw new WrongAdapterError(mismatch);
        return probe.adapter;
    }

    async runPage(page: TestPage, run: PageRunOptions): Promise<PageResult> {
        const { timeoutMs, screenshotDir, gpuLabel } = run;
        const startTime = Date.now();
        const token = `${process.pid}-${startTime}-${page.name}`;
        // Start every page from a blank document, so the previous page's GPU objects are released.
        await this.navigate('about:blank');
        this.messages = [];
        this.crashed = false;

        const query = new URLSearchParams({ [QUERY_TOKEN]: token });
        if (this.options.gpu) query.set(QUERY_GPU, this.options.gpu);
        let runnerProblem: string | null = null;
        let report: GpuTestReport | null = null;
        try {
            await this.navigate(`${this.options.serverOrigin}${page.urlPath}?${query}`);
            const deadline = Date.now() + timeoutMs;
            while (report === null && runnerProblem === null) {
                if (this.crashed) runnerProblem = 'The page crashed.';
                else if (Date.now() > deadline) {
                    runnerProblem = `Timed out after ${timeoutMs} ms without a report (the page hung, or its script failed to load).`;
                } else {
                    report = await this.readReport(token);
                    if (report === null) await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
                }
            }
        } catch (error) {
            runnerProblem = error instanceof Error ? error.message : String(error);
        }

        // Let late console messages (device errors arrive after the report) land before judging.
        await new Promise((resolve) => setTimeout(resolve, LATE_MESSAGE_GRACE_MS));

        const screenshotPath = await this.screenshot(screenshotDir, `${gpuLabel}-${page.name}.png`);
        const messages = [...this.messages];
        const verdict = evaluatePage({
            report,
            runnerProblem,
            messages,
            expectFailure: run.expectFailure,
            expectMessage: page.expectMessage,
            wantedGpu: this.options.gpu,
        });
        return { page, verdict, report, messages, durationMs: Date.now() - startTime, screenshotPath };
    }

    async close(): Promise<void> {
        await this.closeResources();
    }

    private async navigate(url: string): Promise<void> {
        const result = await this.cdp.send<{ errorText?: string }>('Page.navigate', { url });
        if (result.errorText) throw new Error(`Navigation to ${url} failed: ${result.errorText}`);
    }

    private async readReport(token: string): Promise<GpuTestReport | null> {
        const report = await this.cdp.evaluate<GpuTestReport | null>(
            `window.${REPORT_PROPERTY} ?? null`,
        );
        return report && report.token === token ? report : null;
    }

    private async screenshot(directory: string, fileName: string): Promise<string | null> {
        try {
            const { data } = await this.cdp.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, SCREENSHOT_TIMEOUT_MS);
            await mkdir(directory, { recursive: true });
            const path = join(directory, fileName);
            await writeFile(path, Buffer.from(data, 'base64'));
            return path;
        } catch {
            // A page that crashed cannot be photographed; the failure is already reported.
            return null;
        }
    }

    private record(message: BrowserMessage): void {
        const duplicate = this.messages.some(
            (existing) => existing.level === message.level && existing.text === message.text,
        );
        if (!duplicate) this.messages.push(message);
    }

    private listenToBrowser(): void {
        // Log.entryAdded carries what Chrome itself reports, including Dawn's WGSL errors.
        this.cdp.on('Log.entryAdded', (params) => {
            const entry = params.entry as { level: string; text: string; url?: string; lineNumber?: number };
            this.record({
                origin: 'log',
                level: entry.level === 'verbose' ? 'debug' : (entry.level as BrowserMessage['level']),
                text: entry.text,
                where: entry.url ? `${entry.url}:${entry.lineNumber ?? 0}` : undefined,
            });
        });
        this.cdp.on('Runtime.consoleAPICalled', (params) => {
            const args = (params.args as Array<{ value?: unknown; description?: string }>) ?? [];
            this.record({
                origin: 'console',
                level: consoleLevel(String(params.type)),
                text: args.map((arg) => (arg.value !== undefined ? String(arg.value) : (arg.description ?? ''))).join(' '),
            });
        });
        this.cdp.on('Runtime.exceptionThrown', (params) => {
            const details = params.exceptionDetails as { text: string; exception?: { description?: string } };
            this.record({ origin: 'exception', level: 'error', text: details.exception?.description ?? details.text });
        });
        this.cdp.on('Inspector.targetCrashed', () => {
            this.crashed = true;
        });
    }
}

/** Maps console.<type> to a log level. */
function consoleLevel(type: string): BrowserMessage['level'] {
    switch (type) {
        case 'error':
        case 'assert':
            return 'error';
        case 'warning':
            return 'warning';
        case 'info':
            return 'info';
        case 'debug':
            return 'debug';
        default:
            return 'log';
    }
}
