/**
 * Small numeric helpers for the benchmark: medians and the closest-hit
 * agreement check between two tracers. Pure functions, no GPU.
 */

import { AGREEMENT_ABSOLUTE_T, AGREEMENT_RELATIVE_T } from './config.ts';

export function median(values: readonly number[]): number {
    if (values.length === 0) return Number.NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = sorted.length >> 1;
    return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function maximum(values: readonly number[]): number {
    let best = -Infinity;
    for (const value of values) if (value > best) best = value;
    return best;
}

export function sum(values: readonly number[]): number {
    let total = 0;
    for (const value of values) total += value;
    return total;
}

/** A closest-hit record's distance: negative means the ray hit nothing. */
export function isHit(t: number): boolean {
    return t >= 0;
}

export interface Agreement {
    rays: number;
    /** Same answer: both miss, or both hit with t within tolerance. */
    agree: number;
    /** One tracer hit, the other missed. */
    hitMissMismatch: number;
    /** Both hit, but at different distances. */
    distanceMismatch: number;
    /** Both hit: how many rays that was. */
    bothHit: number;
    /** `agree / rays`, or 1 for an empty comparison. */
    rate: number;
}

/**
 * Compares two arrays of closest-hit distances ray by ray. A big disagreement
 * means a bug in one of the tracers, not a speed result.
 */
export function compareClosestDistances(
    a: ArrayLike<number>,
    b: ArrayLike<number>,
    relativeTolerance = AGREEMENT_RELATIVE_T,
    absoluteTolerance = AGREEMENT_ABSOLUTE_T,
): Agreement {
    if (a.length !== b.length) throw new RangeError(`cannot compare ${a.length} results with ${b.length}`);
    let agree = 0;
    let hitMissMismatch = 0;
    let distanceMismatch = 0;
    let bothHit = 0;
    for (let ray = 0; ray < a.length; ray++) {
        const hitA = isHit(a[ray]);
        const hitB = isHit(b[ray]);
        if (hitA !== hitB) {
            hitMissMismatch++;
        } else if (!hitA) {
            agree++;
        } else {
            bothHit++;
            const allowed = Math.max(Math.abs(a[ray]), Math.abs(b[ray])) * relativeTolerance + absoluteTolerance;
            if (Math.abs(a[ray] - b[ray]) <= allowed) agree++;
            else distanceMismatch++;
        }
    }
    return {
        rays: a.length,
        agree,
        hitMissMismatch,
        distanceMismatch,
        bothHit,
        rate: a.length === 0 ? 1 : agree / a.length,
    };
}

/** How often two any-hit answers (1 = blocked) are the same. */
export function compareBlocked(a: ArrayLike<number>, b: ArrayLike<number>): { rays: number; agree: number; rate: number } {
    if (a.length !== b.length) throw new RangeError(`cannot compare ${a.length} results with ${b.length}`);
    let agree = 0;
    for (let ray = 0; ray < a.length; ray++) if ((a[ray] !== 0) === (b[ray] !== 0)) agree++;
    return { rays: a.length, agree, rate: a.length === 0 ? 1 : agree / a.length };
}

/** Million rays per second for `rays` traced in `milliseconds`. */
export function megaRaysPerSecond(rays: number, milliseconds: number): number {
    return rays / (milliseconds * 1000);
}
