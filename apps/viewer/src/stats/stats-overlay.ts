/**
 * A small corner panel with CPU frame time and per-pass GPU milliseconds.
 *
 * It only needs numbers, not the GPU timer itself, so the viewer decides
 * where they come from:
 *
 *     const stats = new StatsOverlay();
 *     // every frame:
 *     stats.recordFrame(frameMs);
 *     stats.setPassTimings(timer.timings);   // or an empty map when timing is off
 */

import { formatStats, SmoothedValue } from './format-stats.ts';

/** Redraw the text a few times a second; faster than that is unreadable. */
const REDRAW_INTERVAL_MS = 250;

export class StatsOverlay {
    readonly element: HTMLPreElement;

    private readonly frameTime = new SmoothedValue();
    private readonly passTimes = new Map<string, SmoothedValue>();
    private lastRedrawAt = 0;

    constructor(parent: HTMLElement = document.body) {
        this.element = document.createElement('pre');
        this.element.style.cssText = [
            'position: fixed',
            'top: 8px',
            'left: 8px',
            'margin: 0',
            'padding: 6px 8px',
            'background: rgba(0, 0, 0, 0.6)',
            'color: #cfd8e3',
            'font: 12px/1.4 ui-monospace, Menlo, Consolas, monospace',
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
        this.element.textContent = [text.frameLine, ...text.passLines, text.totalLine].filter((line) => line !== '').join('\n');
    }
}
