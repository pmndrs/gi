/**
 * The radiance cache's timed compute passes (RC5.1). Each of Capsaicin's
 * timed sections (`TimedSection` in `GI1::render`, gi1.cpp:2229-2936, and
 * the SSGI technique's `Main`, ssgi.cpp:157) is a compute pass of its own,
 * named `radiance-cache/<section>`, so `GIStats.passes` lines up with native
 * Capsaicin's per-pass CSV section for section. The world cache names its
 * passes the same way (world-cache/hash-grid-cache.ts).
 *
 * Splitting a section's dispatches into their own pass changes no result:
 * WebGPU makes each dispatch its own usage scope inside a pass anyway, so
 * the order and visibility of their writes are the same either way.
 */

import type { GpuTimerScope } from '../../platform/contract/timing.ts';

/** The prefix of every radiance-cache pass name in `stats.passes`. */
export const RADIANCE_CACHE_TIMER_PREFIX = 'radiance-cache';

/**
 * Capsaicin's `PopulateScreenProbesCells` section traces the probe rays and
 * inserts their hits' cells in one kernel. The port traces in its own kernel
 * (this name) and inserts in the world cache's (the section's own name), so
 * the section's time is the sum of the two.
 */
export const POPULATE_TRACE_SECTION = 'PopulateScreenProbesCells (trace)';

/** Capsaicin's SSGI technique runs before GI-1 and times its kernel as its own `Main` (ssgi.cpp:157). */
export const SSGI_SECTION = 'SSGI';

/** Our own work with no Capsaicin section (its framework copies depth and normals untimed). */
export const KEEP_HISTORY_SECTION = 'keep-history';

/** Starts a compute pass labelled and timed as `radiance-cache/<section>`. */
export function beginTimedPass(encoder: GPUCommandEncoder, timer: GpuTimerScope, section: string): GPUComputePassEncoder {
    const name = `${RADIANCE_CACHE_TIMER_PREFIX}/${section}`;
    return encoder.beginComputePass({ label: name, timestampWrites: timer.computePass(name) });
}
