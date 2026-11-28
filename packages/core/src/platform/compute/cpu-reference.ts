/**
 * CPU twins of the GPU utilities (RC1.3), for tests: what the GPU must
 * produce, computed the obvious way.
 */

/** Exclusive prefix sums of `values`, wrapping at 2^32 as `u32` adds do, and the total. */
export function exclusiveScanOnCpu(values: Uint32Array): { prefix: Uint32Array; total: number } {
    const prefix = new Uint32Array(values.length);
    let running = 0;
    for (let index = 0; index < values.length; index++) {
        prefix[index] = running;
        running = (running + values[index]) >>> 0;
    }
    return { prefix, total: running };
}

/** The indices of the non-zero flags, in increasing order. */
export function compactOnCpu(flags: Uint32Array): Uint32Array {
    let kept = 0;
    for (const flag of flags) if (flag !== 0) kept++;
    const indices = new Uint32Array(kept);
    let next = 0;
    for (let index = 0; index < flags.length; index++) {
        if (flags[index] !== 0) indices[next++] = index;
    }
    return indices;
}

/**
 * The workgroups `[x, y, z]` the indirect-arguments writer gives for `count`
 * items: `ceil(count / itemsPerWorkgroup)` workgroups in as few rows along y
 * as hold them, x as even as those rows allow, neither past
 * `maxWorkgroupsPerDimension` (beyond limit² workgroups the dispatch is
 * limit² and the kernel loops). No items: `[0, 0, 1]`, which runs nothing.
 */
export function indirectArgsOnCpu(count: number, itemsPerWorkgroup: number, maxWorkgroupsPerDimension: number): [number, number, number] {
    const workgroups = Math.ceil(count / itemsPerWorkgroup);
    if (workgroups === 0) return [0, 0, 1];
    const y = Math.min(Math.ceil(workgroups / maxWorkgroupsPerDimension), maxWorkgroupsPerDimension);
    const x = Math.min(Math.ceil(workgroups / y), maxWorkgroupsPerDimension);
    return [x, y, 1];
}
