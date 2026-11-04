/**
 * Deciding whether a timing difference is real. The cost of timestamps is
 * tiny next to a frame, so the difference between timed and untimed frames
 * is often smaller than the measurement's own scatter. Reporting "+0.1 ms"
 * for that would claim more than we know.
 */

import { median } from './statistics.ts';

export interface OverheadAssessment {
    /** Median over rounds of (timed frame median minus untimed frame median). */
    overheadMs: number;
    /** Smallest and largest per-round overhead: how much the answer moves between rounds. */
    roundMinMs: number;
    roundMaxMs: number;
    /** The finest step `performance.now()` can show. A difference below it cannot be seen at all. */
    timerResolutionMs: number;
    /** True when the overhead is no bigger than the scatter or the timer step. */
    withinNoise: boolean;
    /** The biggest overhead any round showed, or the timer step if that is larger: "under about this much". */
    boundMs: number;
    boundPercent: number;
    /** One sentence for people. */
    text: string;
}

/**
 * @param roundOverheadsMs one overhead per alternating round
 * @param untimedFrameMs the median untimed frame time, for the percentages
 * @param timerResolutionMs see {@link measureTimerResolutionMs}
 */
export function assessOverhead(
    roundOverheadsMs: readonly number[],
    untimedFrameMs: number,
    timerResolutionMs: number,
): OverheadAssessment {
    const overheadMs = median(roundOverheadsMs);
    const roundMinMs = Math.min(...roundOverheadsMs);
    const roundMaxMs = Math.max(...roundOverheadsMs);
    const spreadMs = roundMaxMs - roundMinMs;
    // The overhead is "real" only if it is clearly bigger than both the
    // round-to-round scatter and what the clock can resolve.
    const withinNoise = Math.abs(overheadMs) <= Math.max(spreadMs, timerResolutionMs);
    const boundMs = Math.max(Math.abs(roundMinMs), Math.abs(roundMaxMs), timerResolutionMs);
    const boundPercent = (boundMs / untimedFrameMs) * 100;

    const rounded = (value: number, decimals: number): string => value.toFixed(decimals);
    const text = withinNoise
        ? `within noise, under ~${rounded(boundPercent, 1)}% of a frame (rounds ranged ${rounded(roundMinMs, 2)} to ${rounded(roundMaxMs, 2)} ms)`
        : `${rounded(overheadMs, 2)} ms (${rounded((overheadMs / untimedFrameMs) * 100, 1)}%), outside the round-to-round range ${rounded(roundMinMs, 2)} to ${rounded(roundMaxMs, 2)} ms`;
    return { overheadMs, roundMinMs, roundMaxMs, timerResolutionMs, withinNoise, boundMs, boundPercent, text };
}

/**
 * The smallest step `performance.now()` shows, found by spinning until the
 * clock has ticked a number of times. Browsers round it on purpose (to 100
 * microseconds in Chrome, coarser in some others), and a frame-time
 * difference smaller than that is invisible.
 */
export function measureTimerResolutionMs(ticksToObserve = 50, maxSpinMs = 250): number {
    let smallestStep = Number.POSITIVE_INFINITY;
    let previous = performance.now();
    const startedAt = previous;
    let ticks = 0;
    while (ticks < ticksToObserve && previous - startedAt < maxSpinMs) {
        const now = performance.now();
        if (now > previous) {
            smallestStep = Math.min(smallestStep, now - previous);
            previous = now;
            ticks++;
        }
    }
    return Number.isFinite(smallestStep) ? smallestStep : Number.NaN;
}
