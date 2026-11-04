/**
 * How the baseline benchmark turns repeated timings into one reported number.
 * No engine imports, so it is unit-tested.
 *
 * Two kinds of number:
 *  - A time (`summarize`): the median of the repeats. It can't be negative;
 *    a negative or non-finite result is a failed measurement.
 *  - A difference of two times (`summarizeDifference`, for example the GI cost
 *    = GI-on frame minus GI-off frame): the mean of the paired differences,
 *    with a 95% confidence interval. A small cost measured against noise can
 *    honestly come out a little negative, so:
 *      - interval entirely above 0: a resolved value;
 *      - interval contains 0: the value as measured, negative included, marked
 *        `belowNoise` (the cost is smaller than this run can resolve);
 *      - interval entirely below 0: a failed measurement, because the "with"
 *        side was reliably cheaper than the "without" side, which means the
 *        measurement itself is wrong.
 */

/** A repeated number whose coefficient of variation is above this is marked noisy. */
export const NOISY_CV_THRESHOLD = 0.1;

export interface Measurement {
    /** Metric name without the technique prefix, for example `bake_wall_ms`. */
    name: string;
    /**
     * Null when the measurement failed; `problem` says why. Never NaN or infinite. Negative
     * only for a difference whose confidence interval contains 0 (then `belowNoise` is true).
     */
    value: number | null;
    /** How many repeats the value is the median (time) or mean (difference) of. */
    repeats: number;
    /** Standard deviation over the repeats divided by the mean; null for a single measurement. */
    cv: number | null;
    /** True when `cv` is above {@link NOISY_CV_THRESHOLD}, or the repeats spread around a mean of zero. */
    noisy: boolean;
    /** For a difference: the 95% confidence interval of its mean. Null for a time. */
    confidenceInterval: { low: number; high: number } | null;
    /** For a difference: true when the interval contains 0. Null for a time. */
    belowNoise: boolean | null;
    problem: string | null;
}

/**
 * Two-sided 95% quantiles of Student's t distribution, by degrees of freedom (index 0 is
 * 1 degree). Standard table values; past 30 degrees of freedom 1.96 (the normal) is used.
 */
const T_95_BY_DEGREES_OF_FREEDOM = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131, 2.12, 2.11, 2.101, 2.093, 2.086,
    2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042,
];

export function t95(degreesOfFreedom: number): number {
    return T_95_BY_DEGREES_OF_FREEDOM[degreesOfFreedom - 1] ?? 1.96;
}

export function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Sample standard deviation (divides by n - 1); zero for fewer than two values. */
export function standardDeviation(values: readonly number[]): number {
    if (values.length < 2) return 0;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
    return Math.sqrt(variance);
}

export function failedMeasurement(name: string, problem: string, repeats = 0): Measurement {
    return { name, value: null, repeats, cv: null, noisy: false, confidenceInterval: null, belowNoise: null, problem };
}

/**
 * Turns the repeats of one number into a measurement: the median, its spread, and a
 * failure instead of a value when the result is not a usable time.
 * @param whyNegative what a negative result means for this metric, in plain words
 */
export function summarize(name: string, values: readonly number[], whyNegative = 'a time cannot be negative'): Measurement {
    if (values.length === 0) return failedMeasurement(name, 'no valid samples');
    if (values.some((value) => !Number.isFinite(value))) return failedMeasurement(name, 'a sample was not a finite number', values.length);
    const value = median(values);
    const { spread, cv, noisy } = spreadOf(values);
    if (value < 0) {
        const detail = `${value.toFixed(3)} ms; spread ${spread.toFixed(3)} ms over ${values.length} run${values.length === 1 ? '' : 's'}`;
        return failedMeasurement(name, `came out negative (${detail}): ${whyNegative}`, values.length);
    }
    return { name, value, repeats: values.length, cv, noisy, confidenceInterval: null, belowNoise: null, problem: null };
}

/** Coefficient of variation and the noisy flag, shared by both kinds of number. */
function spreadOf(values: readonly number[]): { mean: number; spread: number; cv: number | null; noisy: boolean } {
    const mean = values.reduce((sum, item) => sum + item, 0) / values.length;
    const spread = standardDeviation(values);
    const cv = values.length > 1 && mean !== 0 ? spread / Math.abs(mean) : null;
    const noisy = values.length > 1 && (cv === null ? spread > 0 : cv > NOISY_CV_THRESHOLD);
    return { mean, spread, cv, noisy };
}

/**
 * A difference of two paired times, one value per round (see the file comment for the rule).
 * The interval is mean +- t * s / sqrt(n), from the round-to-round spread of the differences.
 */
export function summarizeDifference(name: string, differences: readonly number[]): Measurement {
    if (differences.length === 0) return failedMeasurement(name, 'no valid rounds');
    if (differences.some((value) => !Number.isFinite(value))) return failedMeasurement(name, 'a round was not a finite number', differences.length);
    if (differences.length < 2) return failedMeasurement(name, 'a confidence interval needs at least 2 rounds', differences.length);
    const { mean, spread, cv, noisy } = spreadOf(differences);
    const halfWidth = (t95(differences.length - 1) * spread) / Math.sqrt(differences.length);
    const confidenceInterval = { low: mean - halfWidth, high: mean + halfWidth };
    if (confidenceInterval.high < 0) {
        const detail = `${mean.toFixed(3)} ms, 95% interval ${confidenceInterval.low.toFixed(3)} to ${confidenceInterval.high.toFixed(3)} ms over ${differences.length} rounds`;
        return {
            ...failedMeasurement(name, `reliably negative (${detail}): the side with the work was cheaper than the side without, so the measurement is wrong`, differences.length),
            confidenceInterval,
        };
    }
    const belowNoise = confidenceInterval.low <= 0;
    return { name, value: mean, repeats: differences.length, cv, noisy, confidenceInterval, belowNoise, problem: null };
}
