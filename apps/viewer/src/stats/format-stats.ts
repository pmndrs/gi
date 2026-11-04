/**
 * The text of the stats overlay, kept free of the DOM so it can be tested.
 */

/**
 * A running average that follows changes but doesn't flicker. Frame times
 * jump around from frame to frame; a number that rewrites itself 60 times a
 * second can't be read.
 */
export class SmoothedValue {
    private average: number | undefined;
    private readonly smoothing: number;

    /** @param smoothing how much of the old average is kept per sample, from 0 (none) to just under 1 (slow) */
    constructor(smoothing = 0.9) {
        this.smoothing = smoothing;
    }

    add(sample: number): void {
        this.average = this.average === undefined ? sample : this.average * this.smoothing + sample * (1 - this.smoothing);
    }

    get value(): number | undefined {
        return this.average;
    }

    reset(): void {
        this.average = undefined;
    }
}

export interface StatsText {
    /** For example `16.7 ms  (60 fps)`. */
    frameLine: string;
    /** One line per pass, longest name padded so the numbers line up. */
    passLines: string[];
    /** The sum of the pass lines, or an empty string when there are none. */
    totalLine: string;
}

/**
 * @param frameMs CPU frame time (time between frames), if known
 * @param passMs  GPU milliseconds per pass name. Empty when timestamps are unavailable.
 */
export function formatStats(frameMs: number | undefined, passMs: ReadonlyMap<string, number>): StatsText {
    const frameLine =
        frameMs === undefined || frameMs <= 0 ? 'frame: -' : `frame: ${frameMs.toFixed(1)} ms (${(1000 / frameMs).toFixed(0)} fps)`;

    const entries = [...passMs.entries()].sort((a, b) => b[1] - a[1]);
    if (entries.length === 0) return { frameLine, passLines: [], totalLine: '' };

    const nameWidth = Math.max(...entries.map(([name]) => name.length));
    const passLines = entries.map(([name, ms]) => `${name.padEnd(nameWidth)}  ${ms.toFixed(2).padStart(7)} ms`);
    const total = entries.reduce((sum, [, ms]) => sum + ms, 0);
    return { frameLine, passLines, totalLine: `${'GPU total'.padEnd(nameWidth)}  ${total.toFixed(2).padStart(7)} ms` };
}
