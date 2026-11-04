/**
 * A small corner panel with CPU frame time, per-pass GPU milliseconds, and
 * (when DDGI is on screen) DDGI's own numbers. Every line explains itself
 * when the mouse is over it (`title`, from `ui/help-text.ts`).
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

import { HELP } from '../ui/help-text.ts';
import type { StatsLine } from './ddgi-stats.ts';
import { formatStats, SmoothedValue } from './format-stats.ts';

/** Redraw the text a few times a second; faster than that is unreadable. */
const REDRAW_INTERVAL_MS = 250;

interface OverlayLine {
    text: string;
    help: string;
}

export class StatsOverlay {
    readonly element: HTMLDivElement;

    private readonly frameTime = new SmoothedValue();
    private readonly passTimes = new Map<string, SmoothedValue>();
    private extraLines: readonly StatsLine[] = [];
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
            // Lines answer the mouse with their help text; the box is small, so it hides little of the picture.
            'cursor: help',
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
        for (const name of [...this.passTimes.keys()]) {
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

    /** Lines shown under the frame and GPU numbers (DDGI's); an empty list removes them. */
    setExtraLines(lines: readonly StatsLine[]): void {
        this.extraLines = lines;
    }

    dispose(): void {
        this.element.remove();
    }

    private redrawIfDue(): void {
        const now = performance.now();
        if (now - this.lastRedrawAt < REDRAW_INTERVAL_MS) return;
        this.lastRedrawAt = now;

        const smoothedMs = new Map<string, number>();
        for (const [name, smoothed] of this.passTimes) {
            if (smoothed.value !== undefined) smoothedMs.set(name, smoothed.value);
        }
        const text = formatStats(this.frameTime.value, smoothedMs);
        const lines: OverlayLine[] = [{ text: text.frameLine, help: HELP.statsFrame }];
        if (text.passLines.length > 0) {
            if (this.gpuCaption) lines.push({ text: this.gpuCaption, help: HELP.statsGpu });
            for (const line of [...text.passLines, text.totalLine]) lines.push({ text: line, help: HELP.statsGpu });
        }
        for (const line of this.extraLines) lines.push({ text: line.text, help: HELP[line.help] });

        this.element.replaceChildren(
            ...lines.map((line) => {
                const row = document.createElement('div');
                row.textContent = line.text;
                row.title = line.help;
                return row;
            }),
        );
    }
}
