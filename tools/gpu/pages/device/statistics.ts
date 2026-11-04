/**
 * Small statistics helpers for the device page. Medians are the headline
 * number because one slow frame (a browser hiccup) must not move the result.
 */

export interface Summary {
    samples: number;
    median: number;
    mean: number;
    min: number;
    max: number;
}

export function median(values: readonly number[]): number {
    if (values.length === 0) return Number.NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function summarize(values: readonly number[]): Summary {
    if (values.length === 0) {
        return { samples: 0, median: Number.NaN, mean: Number.NaN, min: Number.NaN, max: Number.NaN };
    }
    const total = values.reduce((sum, value) => sum + value, 0);
    return {
        samples: values.length,
        median: median(values),
        mean: total / values.length,
        min: Math.min(...values),
        max: Math.max(...values),
    };
}

/** Rounds for display and JSON; NaN becomes null so it survives JSON.stringify. */
export function roundOrNull(value: number, decimals = 3): number | null {
    if (!Number.isFinite(value)) return null;
    const factor = 10 ** decimals;
    return Math.round(value * factor) / factor;
}
