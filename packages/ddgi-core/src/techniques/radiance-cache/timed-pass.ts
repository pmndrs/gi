/**
 * The radiance cache's timed compute passes (P6.6, from the Mac track's
 * RC5.1). Each of Capsaicin's timed sections (`TimedSection` in
 * `GI1::render`, gi1.cpp:2229-2936, and the SSGI technique's `Main`,
 * ssgi.cpp:157) is a compute pass of its own, named
 * `radiance-cache/<section>`, so `GIStats.passes` lines up with native
 * Capsaicin's per-pass CSV section for section. The world cache, the SHaRC
 * backend and the neural tail name their passes the same way.
 *
 * Splitting a section's dispatches into their own pass changes no result:
 * WebGPU makes each dispatch its own usage scope inside a compute pass
 * anyway, so the order and visibility of their writes are the same either
 * way.
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

/** The full `stats.passes` name of a section. */
export function timedPassName(section: string): string {
    return `${RADIANCE_CACHE_TIMER_PREFIX}/${section}`;
}

/** Starts a compute pass labelled and timed as `radiance-cache/<section>`. */
export function beginTimedPass(encoder: GPUCommandEncoder, timer: GpuTimerScope, section: string): GPUComputePassEncoder {
    const name = timedPassName(section);
    return encoder.beginComputePass({ label: name, timestampWrites: timer.computePass(name) });
}

/**
 * The sections before the world cache that make and trace the probe rays:
 * one pass (`radiance-cache/sample-populate`) until P6.6. Pages that report
 * "the probe trace" add these up.
 */
export const PROBE_TRACE_SECTIONS: readonly string[] = ['PatchScreenProbes', 'SampleScreenProbes', POPULATE_TRACE_SECTION];
