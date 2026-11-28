/**
 * Colours for "how settled is this probe": each probe's history count n from
 * DDGI's adaptive hysteresis (the updates its light is averaged over, up to
 * `hysteresisMaxSamples`, 200 by default). Few updates means a noisy probe; the
 * full count means it has settled as far as DDGI lets it.
 *
 * The scale is logarithmic, because the noise of an average falls like 1/√n:
 * going from 1 to 14 updates quiets a probe about as much as going from 14 to
 * 200. Red at 0, yellow halfway (about √(n_max + 1) − 1 updates, 13 for 200),
 * green at the full count.
 *
 * Plain numbers, no engine, so the legend and the tests use the same function
 * the probe spheres' shader mirrors (`probe-spheres.ts`).
 */

import { DDGI_DEBUG_COLORS } from '../../../../packages/gi/src/ddgi';

export type Rgb = readonly [number, number, number];

/** DDGI's own debug red, yellow and green, so every probe view uses one palette. */
export const HISTORY_COLORS = {
    fewUpdates: DDGI_DEBUG_COLORS.noCoverage,
    halfway: DDGI_DEBUG_COLORS.stateNotUpdatedYet,
    settled: DDGI_DEBUG_COLORS.fullCoverage,
    /** Probes that are off or not updated yet keep no history worth showing. */
    notActive: [0.18, 0.18, 0.18],
} as const satisfies Record<string, Rgb>;

/** Where n sits between 0 and the full count, on the logarithmic scale (0 to 1). */
export function settledShare(samples: number, maxSamples: number): number {
    if (!(maxSamples > 0)) return 0;
    const share = Math.log(Math.max(0, samples) + 1) / Math.log(maxSamples + 1);
    return Math.min(1, Math.max(0, share));
}

function mixRgb(from: Rgb, to: Rgb, amount: number): Rgb {
    return [from[0] + (to[0] - from[0]) * amount, from[1] + (to[1] - from[1]) * amount, from[2] + (to[2] - from[2]) * amount];
}

/** The colour of an active probe whose history counts `samples` updates. */
export function historyColor(samples: number, maxSamples: number): Rgb {
    const share = settledShare(samples, maxSamples);
    return share < 0.5 ? mixRgb(HISTORY_COLORS.fewUpdates, HISTORY_COLORS.halfway, share * 2) : mixRgb(HISTORY_COLORS.halfway, HISTORY_COLORS.settled, share * 2 - 1);
}

/** The count that lands exactly halfway (yellow): √(n_max + 1) − 1. */
export function halfwaySamples(maxSamples: number): number {
    return Math.sqrt(maxSamples + 1) - 1;
}
