/**
 * Turning raw timestamp values into pass durations. Pure functions, no GPU.
 */

/** One timed pass, as read back from the GPU. */
export interface TimedPass {
    /** `<technique>/<pass>` or whatever the caller passed to the timer. */
    readonly name: string;
    readonly gpuMs: number;
}

const NANOSECONDS_PER_MILLISECOND = 1_000_000;

/**
 * `timestamps` holds the resolved query values: for pass n, index `2n` is the
 * start and `2n + 1` the end, in nanoseconds (that is the unit WebGPU defines).
 *
 * A pass whose end is before its start is left out. That happens when the
 * counter wraps or a driver reports garbage, and a negative or huge duration
 * would be worse than a missing one.
 */
export function timestampsToPasses(
    passNames: readonly string[],
    timestamps: ArrayLike<bigint>,
): TimedPass[] {
    const passes: TimedPass[] = [];
    for (let passNumber = 0; passNumber < passNames.length; passNumber++) {
        const start = timestamps[passNumber * 2];
        const end = timestamps[passNumber * 2 + 1];
        if (start === undefined || end === undefined || end < start) continue;
        passes.push({
            name: passNames[passNumber],
            gpuMs: Number(end - start) / NANOSECONDS_PER_MILLISECOND,
        });
    }
    return passes;
}

/**
 * Milliseconds per pass name. A name used more than once in a frame (a pass
 * that runs per cascade, say) is added up, because the reader wants to know
 * what that name cost the frame in total.
 */
export function totalByName(passes: readonly TimedPass[]): Map<string, number> {
    const totals = new Map<string, number>();
    for (const pass of passes) {
        totals.set(pass.name, (totals.get(pass.name) ?? 0) + pass.gpuMs);
    }
    return totals;
}
