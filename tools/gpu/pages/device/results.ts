/**
 * What the device page reports: the data structure, the plain-text version
 * people copy and paste, and the sending of it to the LAN server.
 */

import { type DeviceEnvironment } from './environment.ts';
import { type Summary } from './statistics.ts';

/** Summary numbers rounded for display; null where a number does not exist. */
export type RoundedSummary = { [K in keyof Summary]: number | null };

/** The `data` the page puts in the runner report. */
export interface DeviceRunData {
    timestampsOnAdapter: boolean;
    timestampsOnDevice: boolean;
    /** The timer's own status: ready, unsupported or failed. */
    timerStatus: string;
    workload: Record<string, number>;
    untimedFrameMs: RoundedSummary;
    timedFrameMs: RoundedSummary;
    passGpuMs: Record<string, RoundedSummary>;
    /** Median over rounds of timed minus untimed frame time. Null without timestamps. */
    overheadMs: number | null;
    /** Smallest and largest per-round overhead. */
    overheadRoundMinMs: number | null;
    overheadRoundMaxMs: number | null;
    roundOverheadsMs: Array<number | null>;
    /** True when the overhead is no bigger than the round-to-round scatter or the clock step. */
    overheadWithinNoise: boolean | null;
    /** "Under about this much of a frame", in percent. */
    overheadBoundPercent: number | null;
    /** The sentence to show people, e.g. "within noise, under ~1.0% of a frame". */
    overheadText: string;
    /** The finest step performance.now() showed in this browser. */
    timerResolutionMs: number | null;
    /** How often the timer left work untimed. Should be all zero. */
    timerCounters: { framesWithoutSlot: number; passesOverCapacity: number };
    timedFramesWithoutResult: number;
}

/** Everything one run of the page knows. This is what gets POSTed and copied. */
export interface DeviceResultsPayload {
    label: string;
    collectedAt: string;
    environment: DeviceEnvironment;
    status: 'pass' | 'fail';
    message?: string;
    /** Absent if the run failed before measuring. */
    run: DeviceRunData | null;
    deviceErrors: string[];
}

/** A short browser name and version from the user agent, good enough for a results table. */
export function describeBrowser(userAgent: string): string {
    const patterns: Array<[RegExp, string]> = [
        [/Edg\/([\d.]+)/, 'Edge'],
        [/CriOS\/([\d.]+)/, 'Chrome on iOS'],
        [/FxiOS\/([\d.]+)/, 'Firefox on iOS'],
        [/Firefox\/([\d.]+)/, 'Firefox'],
        [/Chrome\/([\d.]+)/, 'Chrome'],
        [/Version\/([\d.]+).*Safari/, 'Safari'],
    ];
    for (const [pattern, name] of patterns) {
        const match = pattern.exec(userAgent);
        if (match) return `${name} ${match[1]}`;
    }
    return userAgent;
}

function yesNo(value: boolean): string {
    return value ? 'yes' : 'no';
}

function formatMs(value: number | null): string {
    return value === null ? 'n/a' : `${value.toFixed(3)} ms`;
}

function formatSummary(summary: RoundedSummary): string {
    if (summary.samples === 0 || summary.median === null) return 'n/a';
    return `median ${formatMs(summary.median)} (min ${formatMs(summary.min)}, max ${formatMs(summary.max)}, ${summary.samples} frames)`;
}

function describeAdapter(environment: DeviceEnvironment): string {
    const adapter = environment.adapter;
    if (!adapter) return 'none';
    const parts = [adapter.vendor, adapter.architecture, adapter.device, adapter.description].filter((part) => part !== '');
    const text = parts.length > 0 ? parts.join(' / ') : 'adapter (browser hides the details)';
    return adapter.isFallbackAdapter ? `${text} [software fallback]` : text;
}

/** Plain text for pasting into a chat or a file. Ends with the same data as one line of JSON. */
export function formatResultsText(payload: DeviceResultsPayload): string {
    const { environment, run } = payload;
    const lines = [
        `ddgi device page: ${payload.status.toUpperCase()}${payload.message ? ` (${payload.message})` : ''}`,
        `label: ${payload.label || '(none)'}`,
        `when: ${payload.collectedAt}`,
        `browser: ${describeBrowser(environment.userAgent)}`,
        `platform: ${environment.platform}`,
        `WebGPU: ${yesNo(environment.adapterAvailable)}${environment.problem ? ` - ${environment.problem}` : ''}`,
        `adapter: ${describeAdapter(environment)}`,
    ];
    if (run) {
        lines.push(
            `timestamp-query: adapter ${yesNo(run.timestampsOnAdapter)}, device ${yesNo(run.timestampsOnDevice)}, timer ${run.timerStatus}`,
        );
        for (const [name, summary] of Object.entries(run.passGpuMs)) {
            lines.push(`GPU ${name}: ${formatSummary(summary)}`);
        }
        lines.push(`frame untimed: ${formatSummary(run.untimedFrameMs)}`);
        lines.push(`frame timed: ${formatSummary(run.timedFrameMs)}`);
        lines.push(`timing overhead: ${run.overheadText}`);
        lines.push(`  per round (ms): ${run.roundOverheadsMs.map((value) => value?.toFixed(2) ?? 'n/a').join(', ')}`);
        lines.push(`  clock step (performance.now): ${formatMs(run.timerResolutionMs)}`);
        if (run.timerCounters.framesWithoutSlot > 0 || run.timerCounters.passesOverCapacity > 0) {
            lines.push(`  timer left work untimed: ${JSON.stringify(run.timerCounters)}`);
        }
    } else {
        lines.push('timestamp-query: not measured');
    }
    if (payload.deviceErrors.length > 0) lines.push(`GPU errors: ${payload.deviceErrors.join(' | ')}`);
    lines.push('', `json: ${JSON.stringify(payload)}`);
    return lines.join('\n');
}

/** Rows for the big on-screen table. */
export function resultRows(payload: DeviceResultsPayload): Array<[string, string]> {
    const { environment, run } = payload;
    const rows: Array<[string, string]> = [
        ['Result', payload.status === 'pass' ? 'PASS' : `FAIL: ${payload.message ?? 'unknown'}`],
        ['Device', payload.label || '(add ?label=name to the URL)'],
        ['Browser', describeBrowser(environment.userAgent)],
        ['WebGPU', environment.adapterAvailable ? 'YES' : `NO - ${environment.problem ?? 'unknown reason'}`],
        ['Adapter', describeAdapter(environment)],
    ];
    if (run) {
        rows.push(['Timestamps', run.timestampsOnDevice ? 'YES' : 'NO']);
        for (const [name, summary] of Object.entries(run.passGpuMs)) {
            rows.push([`GPU ${name}`, summary.median === null ? 'n/a' : `${formatMs(summary.median)} (median)`]);
        }
        rows.push(['Frame, untimed', formatMs(run.untimedFrameMs.median)]);
        rows.push(['Frame, timed', formatMs(run.timedFrameMs.median)]);
        rows.push(['Timing overhead', run.overheadText]);
        rows.push(['Overhead per round', run.roundOverheadsMs.map((value) => value?.toFixed(2) ?? 'n/a').join(', ') + ' ms']);
        rows.push(['Clock step', formatMs(run.timerResolutionMs)]);
    }
    return rows;
}

/**
 * Sends the results to the LAN server (`pnpm device:serve`), which appends
 * them to a file. Returns a sentence for the screen. The test runner's own
 * server has no such endpoint, so this is skipped there.
 */
export async function postResults(payload: DeviceResultsPayload): Promise<string> {
    try {
        const response = await fetch('/__device-results', {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(payload),
        });
        return response.ok
            ? 'Results were sent to the server.'
            : `The server did not take the results (HTTP ${response.status}). Use Copy instead.`;
    } catch (error) {
        return `Could not reach the server (${error instanceof Error ? error.message : String(error)}). Use Copy instead.`;
    }
}

/** Draws the table, the copy box and the send status. */
export function renderResultsPage(payload: DeviceResultsPayload, sendStatus: string): void {
    const text = formatResultsText(payload);
    const page = document.createElement('div');
    page.style.cssText = 'font: 20px/1.4 system-ui, sans-serif; padding: 16px; color: #eee; background: #14181d;';

    const heading = document.createElement('h1');
    heading.textContent = `Device check: ${payload.status === 'pass' ? 'done' : 'problem'}`;
    heading.style.cssText = 'font-size: 28px; margin: 0 0 12px;';
    page.append(heading);

    const table = document.createElement('table');
    table.style.cssText = 'border-collapse: collapse; width: 100%; font-size: 20px;';
    for (const [name, value] of resultRows(payload)) {
        const row = table.insertRow();
        const nameCell = row.insertCell();
        const valueCell = row.insertCell();
        nameCell.textContent = name;
        valueCell.textContent = value;
        nameCell.style.cssText = 'padding: 6px 12px 6px 0; color: #9fb0c3; vertical-align: top; white-space: nowrap;';
        valueCell.style.cssText = 'padding: 6px 0; font-weight: 600; word-break: break-word;';
        row.style.borderBottom = '1px solid #2a323c';
    }
    page.append(table);

    const sendLine = document.createElement('p');
    sendLine.textContent = sendStatus;
    page.append(sendLine);

    const copyButton = document.createElement('button');
    copyButton.textContent = 'Copy results';
    copyButton.style.cssText = 'font-size: 22px; padding: 12px 20px; margin: 8px 0;';
    const box = document.createElement('textarea');
    box.value = text;
    box.readOnly = true;
    box.rows = 14;
    box.style.cssText = 'width: 100%; box-sizing: border-box; font: 14px monospace;';
    copyButton.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(text);
            copyButton.textContent = 'Copied';
        } catch {
            // Clipboard access can be refused; selecting the text lets the user copy it by hand.
            box.select();
            copyButton.textContent = 'Select the text below and copy it';
        }
    });
    page.append(copyButton, box);

    document.body.style.margin = '0';
    document.body.prepend(page);
}
