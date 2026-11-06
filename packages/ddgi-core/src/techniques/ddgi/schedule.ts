/**
 * Which probes update each frame: the CPU side of the GPU scheduler
 * (docs/design/m2-lights.md §4.3; the kernels are wgsl/schedule.ts).
 *
 * - The round robin walks every probe in one fixed scrambled order,
 *   `order[i] = (i × stride) mod N`, with `stride` coprime with N and near
 *   0.618 N (the golden ratio's fraction). Consecutive entries then land far
 *   apart, so each frame's set is spread through the volume instead of being
 *   one slab of it, and every probe comes up exactly once per pass because a
 *   coprime stride visits every residue. (M1's CPU cursor walked the same
 *   order; the GPU keeps the cursor now.)
 * - `priorityBucket` and `scheduleOnCpu` are the CPU twins of the kernels'
 *   rules, for the unit tests: what the list holds for given owed probes,
 *   urgencies and camera weights.
 */

import { PRIORITY_BUCKET_OFFSET, PRIORITY_BUCKETS } from './layout.ts';

/** 1/φ: consecutive multiples of it mod 1 are as evenly spread as any sequence can be. */
const GOLDEN_FRACTION = (Math.sqrt(5) - 1) / 2;

function greatestCommonDivisor(a: number, b: number): number {
    while (b !== 0) [a, b] = [b, a % b];
    return a;
}

/** The stride coprime with `probeCount` nearest to 0.618 × `probeCount` (1 for tiny counts). */
export function scrambleStride(probeCount: number): number {
    if (probeCount <= 2) return 1;
    const target = Math.round(probeCount * GOLDEN_FRACTION);
    // Search outwards from the target; a coprime number is always close (1 is one, at worst).
    for (let distance = 0; distance < probeCount; distance++) {
        for (const candidate of [target - distance, target + distance]) {
            if (candidate >= 1 && candidate < probeCount && greatestCommonDivisor(candidate, probeCount) === 1) return candidate;
        }
    }
    return 1;
}

/** The priority-share count P for a list of `count` probes: `floor(share × count)`. */
export function priorityCount(count: number, share: number): number {
    return Math.max(0, Math.min(count, Math.floor(share * count)));
}

/** The bucket of a priority (`ddgiPriorityBucket`): bucket b holds [2^(b − offset), 2^(b + 1 − offset)). */
export function priorityBucket(priority: number): number {
    if (!(priority > 0)) return 0;
    const bucket = Math.floor(Math.log2(priority)) + PRIORITY_BUCKET_OFFSET;
    return Math.min(PRIORITY_BUCKETS - 1, Math.max(0, bucket));
}

/** One probe's part in a frame's scheduling (the CPU twin's input). */
export interface ScheduledProbe {
    /** Updates it is owed (0: none). */
    readonly owed: number;
    /** urgency × camera weight. */
    readonly priority: number;
}

export interface CpuSchedule {
    /** The probes listed at priority, bucket by bucket from the most urgent (inside a bucket: by index). */
    readonly priority: number[];
    /** The round robin's part, in order. */
    readonly roundRobin: number[];
    /** The probe at the cursor afterwards. */
    readonly cursorProbe: number;
    /** Probes per bucket. */
    readonly histogram: number[];
}

/**
 * The scheduler's rules on the CPU (the four kernels, wgsl/schedule.ts). On
 * the GPU the boundary bucket's members are whichever reserve a slot first;
 * here they are the lowest indices, so tests compare sets above the boundary
 * and counts in it.
 */
export function scheduleOnCpu(probes: readonly ScheduledProbe[], count: number, share: number, cursorProbe: number, stride: number): CpuSchedule {
    const probeCount = probes.length;
    const listCount = Math.min(count, probeCount);
    const histogram = new Array<number>(PRIORITY_BUCKETS).fill(0);
    const buckets = probes.map((probe) => (probe.owed > 0 ? priorityBucket(probe.priority) : -1));
    for (const bucket of buckets) if (bucket >= 0) histogram[bucket]++;
    const limit = priorityCount(listCount, share);
    const priority: number[] = [];
    for (let bucket = PRIORITY_BUCKETS - 1; bucket >= 0 && priority.length < limit; bucket--) {
        for (let probe = 0; probe < probeCount && priority.length < limit; probe++) if (buckets[probe] === bucket) priority.push(probe);
    }
    const stamped = new Set(priority);
    const roundRobin: number[] = [];
    let used = 0;
    for (let position = 0; position < listCount && priority.length + roundRobin.length < listCount; position++) {
        const probe = (cursorProbe + position * stride) % probeCount;
        if (stamped.has(probe)) continue;
        roundRobin.push(probe);
        used = position + 1;
    }
    return { priority, roundRobin, cursorProbe: (cursorProbe + used * stride) % probeCount, histogram };
}
