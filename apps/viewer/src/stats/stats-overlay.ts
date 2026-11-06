/**
 * A small corner panel with CPU frame time, per-pass GPU milliseconds, and
 * (when DDGI is on screen) DDGI's own numbers.
 *
 * The box never takes the mouse (`pointer-events: none`): dragging over it
 * must orbit the camera like anywhere else on the picture. What each line
 * means is explained in the panel instead ("What the stats mean", from
 * `ui/help-text.ts`).
 *
 * It only needs numbers, not the GPU timer itself, so the viewer decides
 * where they come from:
 *
 *     const stats = new StatsOverlay();
 *     // every frame:
 *     stats.recordFrame(frameMs);
 *     stats.setPassTimings(timer.timings);   // or an empty map when timing is off
 *     stats.setExtraLines(ddgiText.lines()); // or [] when DDGI isn't shown
 */

import type { StatsLine } from './ddgi-stats.ts';
import { formatStats, SmoothedValue } from './format-stats.ts';

/** Redraw the text a few times a second; faster than that is unreadable. */
const REDRAW_INTERVAL_MS = 250;

const NO_EXTRA_LINES = (): readonly StatsLine[] => [];

export class StatsOverlay {
    readonly element: HTMLDivElement;

    private readonly frameTime = new SmoothedValue();
    private readonly passTimes = new Map<string, SmoothedValue>();
    /** Smoothed pass times handed to `formatStats`, refilled on each redraw. */
    private readonly smoothedMs = new Map<string, number>();
    private extraLines: () => readonly StatsLine[] = NO_EXTRA_LINES;
    /** One element per line, kept and reused: only their text changes between redraws. */
    private readonly rows: HTMLDivElement[] = [];
    private lastRedrawAt = 0;

    /**
     * @param gpuCaption a line shown above the GPU pass lines, saying what the numbers are
     *   (for example "GPU, one frame, averaged:"); nothing when empty
     */
    constructor(
        parent: HTMLElement = document.body,
        private readonly gpuCaption = '',
    ) {
        this.element = document.createElement('div');
        this.element.className = 'stats-overlay';
        this.element.style.cssText = [
            'position: fixed',
            'top: 8px',
            'left: 8px',
            'margin: 0',
            'padding: 6px 8px',
            'background: rgba(0, 0, 0, 0.6)',
            'color: #cfd8e3',
            'font: 12px/1.4 ui-monospace, Menlo, Consolas, monospace',
            'white-space: pre',
            // The mouse goes straight through to the picture (camera controls, the wipe line).
            'pointer-events: none',
            'z-index: 10',
        ].join(';');
        parent.append(this.element);
    }

    /** Call once per frame with the time since the previous frame. */
    recordFrame(frameMs: number): void {
        this.frameTime.add(frameMs);
        this.redrawIfDue();
    }

    /**
     * Call whenever new timings are known (they arrive a few frames late).
     * Passes missing from `timings` drop out of the panel.
     */
    setPassTimings(timings: ReadonlyMap<string, number>): void {
        // Deleting the entry being visited is safe in a Map iteration.
        for (const name of this.passTimes.keys()) {
            if (!timings.has(name)) this.passTimes.delete(name);
        }
        for (const [name, ms] of timings) {
            let smoothed = this.passTimes.get(name);
            if (!smoothed) {
                smoothed = new SmoothedValue();
                this.passTimes.set(name, smoothed);
            }
            smoothed.add(ms);
        }
    }

    /** Starts the frame-time average again (after a set-up, whose frames are drawn only a few times a second). */
    resetFrameTime(): void {
        this.frameTime.reset();
    }

    /** Where the lines under the frame and GPU numbers (DDGI's) come from; read only when the box is redrawn. */
    setExtraLines(source: () => readonly StatsLine[]): void {
        this.extraLines = source;
    }

    dispose(): void {
        this.element.remove();
    }

    private redrawIfDue(): void {
        const now = performance.now();
        if (now - this.lastRedrawAt < REDRAW_INTERVAL_MS) return;
        this.lastRedrawAt = now;

        this.smoothedMs.clear();
        for (const [name, smoothed] of this.passTimes) {
            if (smoothed.value !== undefined) this.smoothedMs.set(name, smoothed.value);
        }
        const text = formatStats(this.frameTime.value, this.smoothedMs);
        let row = 0;
        this.setRow(row++, text.frameLine);
        if (text.passLines.length > 0) {
            if (this.gpuCaption) this.setRow(row++, this.gpuCaption);
            for (const line of text.passLines) this.setRow(row++, line);
            this.setRow(row++, text.totalLine);
        }
        for (const line of this.extraLines()) this.setRow(row++, line.text);
        for (let unused = row; unused < this.rows.length; unused++) this.rows[unused].hidden = true;
    }

    /** Shows `text` on line `index`, making the line the first time it is needed. */
    private setRow(index: number, text: string): void {
        let element = this.rows[index];
        if (!element) {
            element = document.createElement('div');
            this.rows.push(element);
            this.element.append(element);
        }
        element.hidden = false;
        if (element.textContent !== text) element.textContent = text;
    }
}
