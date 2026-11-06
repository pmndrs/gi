/**
 * The GPU scheduler (docs/design/m2-lights.md §4.3): which probes this
 * frame's update list holds. It replaces M1's CPU cursor, because the
 * urgencies it orders by live on the GPU (`ddgi/mark` and the feedback rule
 * write them). The list is still one buffer filled with exactly
 * `updateCount` entries, so the trace, blend and write dispatches keep sizes
 * the CPU knows, and Budget v2 still sees the units it handed out.
 *
 * Four dispatches, every frame:
 * 1. `ddgi/schedule-count`, one thread per probe: takes the probe's feedback
 *    request (m2-lights.md §4.6) into its record, and a probe owed updates
 *    gets the priority `urgency × camera weight`, sorted into one of 32
 *    buckets, each twice as urgent as the one below. The buckets' histogram
 *    is counted **subgroups first** (one atomic per bucket present in a
 *    subgroup) or by the **declared fallback** (one atomic per thread).
 * 2. `ddgi/schedule-select`, one thread: the bucket where the priority share
 *    `P` fills up (the boundary). Probes above it all fit; from the boundary
 *    bucket, as many as there is room for.
 * 3. `ddgi/schedule-write`, one thread per probe: the chosen probes reserve
 *    list slots (subgroups first: one atomic per subgroup; fallback: one per
 *    thread) and are stamped with the frame.
 * 4. `ddgi/schedule-fill`, one workgroup: the rest of the list comes from the
 *    round robin, the scrambled order `order[i] = (i × stride) mod N` from
 *    the cursor on, skipping probes stamped this frame. A window of
 *    `updateCount` entries holds at most the listed ones stamped, so it
 *    always has enough unstamped ones. The cursor moves past the entries it
 *    used (stamped entries it passes were updated this frame anyway), so
 *    every probe still comes up once per pass of the round robin. The
 *    compaction is a workgroup scan: subgroups first, the declared fallback
 *    in workgroup memory.
 *
 * With nothing owed, the list is exactly M1's cursor order: the first
 * `updateCount` entries from the cursor.
 *
 * Bindings (group 1, one layout for all four): uniforms, the probe history
 * (plain words: each thread owns its probe's record in these dispatches),
 * the feedback requests, the schedule state (SCHEDULE_STATE_WORD) and the
 * update list: 4 storage buffers.
 */

import { PRIORITY_BUCKET_OFFSET, PRIORITY_BUCKETS, PROBE_HISTORY_UPDATES, PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS, SCHEDULE_STATE_WORD } from '../layout.ts';
import { DDGI_COMMON_WGSL } from './common.ts';
import type { PhaseBReduction } from './blend.ts';

export const SCHEDULE_WORKGROUP_SIZE = 256;

/** How the histogram and the compactions combine threads: subgroups first, or the declared fallback. */
export type ScheduleReduction = PhaseBReduction;

/** The four kernels' entry points, in the order they run. */
export const SCHEDULE_ENTRY_POINTS = {
    count: 'ddgiScheduleCount',
    select: 'ddgiScheduleSelect',
    write: 'ddgiScheduleWrite',
    fill: 'ddgiScheduleFill',
} as const;

/** The smallest subgroup WebGPU allows; sizes the fill's per-subgroup totals. */
const MIN_SUBGROUP_SIZE = 4;

function reductionWgsl(reduction: ScheduleReduction): string {
    if (reduction === 'subgroups') {
        return /* wgsl */ `
// ---- Subgroups first ----

// The histogram: each pass of the loop counts one bucket present in the
// subgroup (the smallest still pending) with one atomic, so a subgroup whose
// probes share a few buckets makes a few atomics instead of one per probe.
// The loop runs while any lane is pending, which is uniform in the subgroup
// but not, to WGSL's analysis, in the workgroup; the diagnostic allows that.
@diagnostic(off, subgroup_uniformity)
fn ddgiCountBucket(bucket: u32) {
    var pending = bucket != DDGI_NO_BUCKET;
    loop {
        if (!subgroupAny(pending)) {
            break;
        }
        let leader = subgroupMin(select(DDGI_NO_BUCKET, bucket, pending));
        let mine = pending && bucket == leader;
        let count = subgroupAdd(select(0u, 1u, mine));
        if (subgroupElect()) {
            atomicAdd(&ddgiScheduleState[DDGI_STATE_HISTOGRAM + leader], count);
        }
        pending = pending && !mine;
    }
}

// Reserves one list slot per taking thread from a counter: one atomic per
// subgroup, each lane its offset inside the subgroup's share. Called by every
// thread in uniform control flow.
fn ddgiReserveSlot(take: bool, counter: u32) -> u32 {
    let flag = select(0u, 1u, take);
    let count = subgroupAdd(flag);
    let offset = subgroupExclusiveAdd(flag);
    var base = 0u;
    if (subgroupElect() && count > 0u) {
        base = atomicAdd(&ddgiScheduleState[counter], count);
    }
    // subgroupElect's lane is the lowest active one, the lane subgroupBroadcastFirst reads.
    return subgroupBroadcastFirst(base) + offset;
}
`;
    }
    return /* wgsl */ `
// ---- The declared fallback without subgroups ----

fn ddgiCountBucket(bucket: u32) {
    if (bucket != DDGI_NO_BUCKET) {
        atomicAdd(&ddgiScheduleState[DDGI_STATE_HISTOGRAM + bucket], 1u);
    }
}

fn ddgiReserveSlot(take: bool, counter: u32) -> u32 {
    if (take) {
        return atomicAdd(&ddgiScheduleState[counter], 1u);
    }
    return 0u;
}
`;
}

/**
 * The fill's exclusive scan over one chunk of the window, subgroups first:
 * each subgroup scans itself, and one pass over the subgroups' totals joins
 * them. That needs subgroups to hold consecutive thread indices in order,
 * which a 1-D workgroup does on the GPUs we know, but WebGPU doesn't promise;
 * the kernel checks it on entry and takes the workgroup scan when it fails.
 */
function fillScanWgsl(reduction: ScheduleReduction): string {
    const workgroupScan = /* wgsl */ `
// The declared fallback: a Hillis-Steele scan in workgroup memory.
fn ddgiWorkgroupScan(flag: u32, thread: u32) -> u32 {
    ddgiFillScan[thread] = flag;
    workgroupBarrier();
    for (var step = 1u; step < DDGI_SCHEDULE_WORKGROUP_SIZE; step = step * 2u) {
        var before = 0u;
        if (thread >= step) {
            before = ddgiFillScan[thread - step];
        }
        workgroupBarrier();
        ddgiFillScan[thread] = ddgiFillScan[thread] + before;
        workgroupBarrier();
    }
    let inclusive = ddgiFillScan[thread];
    if (thread == DDGI_SCHEDULE_WORKGROUP_SIZE - 1u) {
        ddgiFillChunkTotal = inclusive;
    }
    workgroupBarrier();
    return inclusive - flag;
}
`;
    if (reduction !== 'subgroups') {
        return /* wgsl */ `
${workgroupScan}
fn ddgiChunkScan(flag: u32, thread: u32, lane: u32, laneCount: u32, ordered: bool) -> u32 {
    return ddgiWorkgroupScan(flag, thread);
}
`;
    }
    return /* wgsl */ `
${workgroupScan}
fn ddgiSubgroupScan(flag: u32, thread: u32, lane: u32, laneCount: u32) -> u32 {
    let inSubgroup = subgroupExclusiveAdd(flag);
    let subgroupTotal = subgroupAdd(flag);
    let subgroupIndex = thread / laneCount;
    if (lane == 0u) {
        ddgiFillSubgroupTotals[subgroupIndex] = subgroupTotal;
    }
    workgroupBarrier();
    if (thread == 0u) {
        var running = 0u;
        let subgroups = (DDGI_SCHEDULE_WORKGROUP_SIZE + laneCount - 1u) / laneCount;
        for (var index = 0u; index < subgroups; index++) {
            let total = ddgiFillSubgroupTotals[index];
            ddgiFillSubgroupTotals[index] = running;
            running += total;
        }
        ddgiFillChunkTotal = running;
    }
    workgroupBarrier();
    return ddgiFillSubgroupTotals[subgroupIndex] + inSubgroup;
}

fn ddgiChunkScan(flag: u32, thread: u32, lane: u32, laneCount: u32, ordered: bool) -> u32 {
    if (ordered) {
        return ddgiSubgroupScan(flag, thread, lane, laneCount);
    }
    return ddgiWorkgroupScan(flag, thread);
}
`;
}

export function scheduleKernelWgsl(reduction: ScheduleReduction): string {
    const enable = reduction === 'subgroups' ? 'enable subgroups;\n' : '';
    const subgroupBuiltins = reduction === 'subgroups' ? '@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) laneCount: u32,' : '';
    // The fallback has no subgroup builtins: the scan ignores them.
    const laneArguments = reduction === 'subgroups' ? 'lane, laneCount' : '0u, 1u';
    return /* wgsl */ `${enable}
${DDGI_COMMON_WGSL}

// ---- DDGI scheduler kernels (techniques/ddgi) ------------------------------------------

const DDGI_SCHEDULE_WORKGROUP_SIZE: u32 = ${SCHEDULE_WORKGROUP_SIZE}u;
const DDGI_PRIORITY_BUCKETS: u32 = ${PRIORITY_BUCKETS}u;
const DDGI_PRIORITY_BUCKET_OFFSET: i32 = ${PRIORITY_BUCKET_OFFSET};
// A probe owed nothing is in no bucket.
const DDGI_NO_BUCKET: u32 = 0xffffffffu;
// The probe history, read as words (PROBE_HISTORY_WORD in layout.ts).
const DDGI_HISTORY_WORDS: u32 = ${PROBE_HISTORY_WORDS}u;
const DDGI_HISTORY_UPDATES: u32 = ${PROBE_HISTORY_WORD.updates}u;
const DDGI_HISTORY_URGENCY: u32 = ${PROBE_HISTORY_WORD.urgency}u;
const DDGI_HISTORY_SCHEDULE_AND_FLAGS: u32 = ${PROBE_HISTORY_WORD.scheduleAndFlags}u;
const DDGI_OWED_SHIFT: u32 = ${PROBE_HISTORY_UPDATES.owedShift}u;
const DDGI_OWED_MASK: u32 = ${PROBE_HISTORY_UPDATES.owedMask}u;
// The scheduled frame sits above the flags' 8 bits; 24 bits only need to differ from this frame's.
const DDGI_SCHEDULED_SHIFT: u32 = 8u;
const DDGI_FLAGS_MASK: u32 = 0xffu;
// The schedule state's words (SCHEDULE_STATE_WORD in layout.ts).
const DDGI_STATE_CURSOR_PROBE: u32 = ${SCHEDULE_STATE_WORD.cursorProbe}u;
const DDGI_STATE_ABOVE_TOTAL: u32 = ${SCHEDULE_STATE_WORD.aboveTotal}u;
const DDGI_STATE_ABOVE_SLOTS: u32 = ${SCHEDULE_STATE_WORD.aboveSlots}u;
const DDGI_STATE_BOUNDARY_SLOTS: u32 = ${SCHEDULE_STATE_WORD.boundarySlots}u;
const DDGI_STATE_BOUNDARY_BUCKET: u32 = ${SCHEDULE_STATE_WORD.boundaryBucket}u;
const DDGI_STATE_BOUNDARY_ROOM: u32 = ${SCHEDULE_STATE_WORD.boundaryRoom}u;
const DDGI_STATE_LISTED: u32 = ${SCHEDULE_STATE_WORD.listed}u;
const DDGI_STATE_OWED: u32 = ${SCHEDULE_STATE_WORD.owed}u;
const DDGI_STATE_HISTOGRAM: u32 = ${SCHEDULE_STATE_WORD.histogram}u;

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read_write> ddgiHistoryWords: array<u32>;
@group(1) @binding(2) var<storage, read_write> ddgiFeedbackRequests: array<u32>;
@group(1) @binding(3) var<storage, read_write> ddgiScheduleState: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> ddgiUpdateList: array<u32>;

// The fill's scan (one workgroup).
var<workgroup> ddgiFillScan: array<u32, DDGI_SCHEDULE_WORKGROUP_SIZE>;
var<workgroup> ddgiFillSubgroupTotals: array<u32, ${SCHEDULE_WORKGROUP_SIZE / MIN_SUBGROUP_SIZE}>;
var<workgroup> ddgiFillChunkTotal: u32;
var<workgroup> ddgiFillPicked: u32;
var<workgroup> ddgiFillListed: u32;
var<workgroup> ddgiFillCursor: u32;
var<workgroup> ddgiFillUsed: u32;
var<workgroup> ddgiFillUnordered: atomic<u32>;

${reductionWgsl(reduction)}
${fillScanWgsl(reduction)}

fn ddgiTotalProbes() -> u32 {
    let counts = ddgiProbeCount();
    return counts.x * counts.y * counts.z;
}

// The frame as a probe's stamp stores it (24 bits).
fn ddgiFrameStamp() -> u32 {
    return ddgi.frame & (0xffffffffu >> DDGI_SCHEDULED_SHIFT);
}

fn ddgiStampOf(probeIndex: u32) -> u32 {
    return ddgiHistoryWords[probeIndex * DDGI_HISTORY_WORDS + DDGI_HISTORY_SCHEDULE_AND_FLAGS] >> DDGI_SCHEDULED_SHIFT;
}

// Marks the probe as listed this frame (keeps its flags).
fn ddgiStamp(probeIndex: u32) {
    let word = probeIndex * DDGI_HISTORY_WORDS + DDGI_HISTORY_SCHEDULE_AND_FLAGS;
    ddgiHistoryWords[word] = (ddgiHistoryWords[word] & DDGI_FLAGS_MASK) | (ddgiFrameStamp() << DDGI_SCHEDULED_SHIFT);
}

// Bucket b holds priorities in [2^(b - offset), 2^(b + 1 - offset)); the
// ends take everything beyond them.
fn ddgiPriorityBucket(priority: f32) -> u32 {
    if (!(priority > 0.0)) {
        return 0u;
    }
    let bucket = i32(floor(log2(priority))) + DDGI_PRIORITY_BUCKET_OFFSET;
    return u32(clamp(bucket, 0, i32(DDGI_PRIORITY_BUCKETS) - 1));
}

// The probe's bucket from its record, or DDGI_NO_BUCKET when it owes nothing.
fn ddgiOwedBucket(probeIndex: u32) -> u32 {
    let base = probeIndex * DDGI_HISTORY_WORDS;
    let owed = (ddgiHistoryWords[base + DDGI_HISTORY_UPDATES] >> DDGI_OWED_SHIFT) & DDGI_OWED_MASK;
    if (owed == 0u) {
        return DDGI_NO_BUCKET;
    }
    let urgency = bitcast<f32>(ddgiHistoryWords[base + DDGI_HISTORY_URGENCY]);
    return ddgiPriorityBucket(urgency * ddgiCameraWeight(ddgiProbeCoordinates(probeIndex, ddgiProbeCount())));
}

// A feedback request (written by last frame's ddgi/blend with atomicMax on
// the urgency's bits) becomes one owed update at that urgency, and is cleared.
fn ddgiTakeFeedbackRequest(probeIndex: u32) {
    let request = ddgiFeedbackRequests[probeIndex];
    if (request == 0u) {
        return;
    }
    ddgiFeedbackRequests[probeIndex] = 0u;
    let base = probeIndex * DDGI_HISTORY_WORDS;
    let updates = ddgiHistoryWords[base + DDGI_HISTORY_UPDATES];
    let owed = max((updates >> DDGI_OWED_SHIFT) & DDGI_OWED_MASK, 1u);
    ddgiHistoryWords[base + DDGI_HISTORY_UPDATES] = (updates & ~(DDGI_OWED_MASK << DDGI_OWED_SHIFT)) | (owed << DDGI_OWED_SHIFT);
    // Non-negative floats order like their bits.
    ddgiHistoryWords[base + DDGI_HISTORY_URGENCY] = max(ddgiHistoryWords[base + DDGI_HISTORY_URGENCY], request);
}

fn ddgiScheduleProbe(group: vec3u, groups: vec3u, thread: u32) -> u32 {
    // Dispatches past 65,535 workgroups use two dimensions.
    return (group.x + group.y * groups.x) * DDGI_SCHEDULE_WORKGROUP_SIZE + thread;
}

// (a * b) mod n without overflowing 32 bits, for n < 2^31: double and add
// over b's bits. b is a window position (at most the list's length), so this
// takes about a dozen steps.
fn ddgiAddMod(a: u32, b: u32, n: u32) -> u32 {
    let sum = a + b;
    return select(sum, sum - n, sum >= n);
}

fn ddgiMulMod(a: u32, b: u32, n: u32) -> u32 {
    var result = 0u;
    var addend = a % n;
    var bits = b;
    loop {
        if (bits == 0u) {
            break;
        }
        if ((bits & 1u) != 0u) {
            result = ddgiAddMod(result, addend, n);
        }
        addend = ddgiAddMod(addend, addend, n);
        bits = bits >> 1u;
    }
    return result;
}

// The round robin's entry 'position' after the cursor.
fn ddgiOrderAfterCursor(cursorProbe: u32, position: u32, probeCount: u32) -> u32 {
    return ddgiAddMod(cursorProbe, ddgiMulMod(ddgi.scrambleStride, position, probeCount), probeCount);
}

// ---- 1. count ----
@compute @workgroup_size(DDGI_SCHEDULE_WORKGROUP_SIZE)
fn ${SCHEDULE_ENTRY_POINTS.count}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let probeIndex = ddgiScheduleProbe(group, groups, thread);
    var bucket = DDGI_NO_BUCKET;
    if (probeIndex < ddgiTotalProbes()) {
        ddgiTakeFeedbackRequest(probeIndex);
        bucket = ddgiOwedBucket(probeIndex);
    }
    // Every thread, in uniform control flow (the subgroup form needs it).
    ddgiCountBucket(bucket);
}

// ---- 2. select ----
// One thread: from the most urgent bucket down, the bucket where the
// priority share fills up.
@compute @workgroup_size(1)
fn ${SCHEDULE_ENTRY_POINTS.select}() {
    let share = ddgi.priorityCount;
    var owed = 0u;
    for (var bucket = 0u; bucket < DDGI_PRIORITY_BUCKETS; bucket++) {
        owed += atomicLoad(&ddgiScheduleState[DDGI_STATE_HISTOGRAM + bucket]);
    }
    // Every owed probe fits: no boundary.
    var boundary = -1;
    var above = owed;
    var room = 0u;
    if (owed > share) {
        var cumulative = 0u;
        for (var bucket = i32(DDGI_PRIORITY_BUCKETS) - 1; bucket >= 0; bucket--) {
            let count = atomicLoad(&ddgiScheduleState[DDGI_STATE_HISTOGRAM + u32(bucket)]);
            if (cumulative + count >= share) {
                boundary = bucket;
                above = cumulative;
                room = share - cumulative;
                break;
            }
            cumulative += count;
        }
    }
    atomicStore(&ddgiScheduleState[DDGI_STATE_ABOVE_TOTAL], above);
    atomicStore(&ddgiScheduleState[DDGI_STATE_ABOVE_SLOTS], 0u);
    atomicStore(&ddgiScheduleState[DDGI_STATE_BOUNDARY_SLOTS], 0u);
    atomicStore(&ddgiScheduleState[DDGI_STATE_BOUNDARY_BUCKET], bitcast<u32>(boundary));
    atomicStore(&ddgiScheduleState[DDGI_STATE_BOUNDARY_ROOM], room);
    atomicStore(&ddgiScheduleState[DDGI_STATE_LISTED], above + room);
    atomicStore(&ddgiScheduleState[DDGI_STATE_OWED], owed);
}

// ---- 3. write ----
@compute @workgroup_size(DDGI_SCHEDULE_WORKGROUP_SIZE)
fn ${SCHEDULE_ENTRY_POINTS.write}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let probeIndex = ddgiScheduleProbe(group, groups, thread);
    let boundary = bitcast<i32>(atomicLoad(&ddgiScheduleState[DDGI_STATE_BOUNDARY_BUCKET]));
    var takeAbove = false;
    var takeBoundary = false;
    if (probeIndex < ddgiTotalProbes()) {
        let bucket = ddgiOwedBucket(probeIndex);
        if (bucket != DDGI_NO_BUCKET) {
            takeAbove = i32(bucket) > boundary;
            takeBoundary = i32(bucket) == boundary;
        }
    }
    // Every thread, in uniform control flow: slots above the boundary first
    // (all of them fit), then the boundary bucket's, up to its room.
    let aboveSlot = ddgiReserveSlot(takeAbove, DDGI_STATE_ABOVE_SLOTS);
    let boundaryRank = ddgiReserveSlot(takeBoundary, DDGI_STATE_BOUNDARY_SLOTS);
    if (takeAbove) {
        ddgiUpdateList[aboveSlot] = probeIndex;
        ddgiStamp(probeIndex);
    } else if (takeBoundary && boundaryRank < atomicLoad(&ddgiScheduleState[DDGI_STATE_BOUNDARY_ROOM])) {
        ddgiUpdateList[atomicLoad(&ddgiScheduleState[DDGI_STATE_ABOVE_TOTAL]) + boundaryRank] = probeIndex;
        ddgiStamp(probeIndex);
    }
}

// ---- 4. fill ----
@compute @workgroup_size(DDGI_SCHEDULE_WORKGROUP_SIZE)
fn ${SCHEDULE_ENTRY_POINTS.fill}(
    @builtin(local_invocation_index) thread: u32,
    ${subgroupBuiltins}
) {
    let probeCount = ddgiTotalProbes();
    if (thread == 0u) {
        ddgiFillListed = atomicLoad(&ddgiScheduleState[DDGI_STATE_LISTED]);
        ddgiFillCursor = atomicLoad(&ddgiScheduleState[DDGI_STATE_CURSOR_PROBE]);
        ddgiFillPicked = 0u;
        ddgiFillUsed = 0u;
        atomicStore(&ddgiFillUnordered, 0u);
    }
    // Read through workgroupUniformLoad, so the loop below stays in uniform control flow.
    let listed = workgroupUniformLoad(&ddgiFillListed);
    let cursorProbe = workgroupUniformLoad(&ddgiFillCursor);
    let need = ddgi.updateCount - min(listed, ddgi.updateCount);
    ${reduction === 'subgroups' ? '// Subgroups hold consecutive threads in order (see fillScanWgsl), or this dispatch takes the workgroup scan.\n    if (subgroupBroadcastFirst(thread) + lane != thread) {\n        atomicStore(&ddgiFillUnordered, 1u);\n    }' : ''}
    let ordered = workgroupUniformLoad(&ddgiFillUnordered) == 0u;
    for (var start = 0u; start < ddgi.updateCount; start += DDGI_SCHEDULE_WORKGROUP_SIZE) {
        let picked = workgroupUniformLoad(&ddgiFillPicked);
        if (picked >= need) {
            break;
        }
        let position = start + thread;
        var probeIndex = 0u;
        var free = false;
        if (position < ddgi.updateCount) {
            probeIndex = ddgiOrderAfterCursor(cursorProbe, position, probeCount);
            free = ddgiStampOf(probeIndex) != ddgiFrameStamp();
        }
        let flag = select(0u, 1u, free);
        let rank = ddgiChunkScan(flag, thread, ${laneArguments}, ordered);
        if (free && picked + rank < need) {
            ddgiUpdateList[listed + picked + rank] = probeIndex;
            ddgiStamp(probeIndex);
            if (picked + rank == need - 1u) {
                // The last entry taken: the cursor moves past it.
                ddgiFillUsed = position + 1u;
            }
        }
        if (thread == 0u) {
            ddgiFillPicked = picked + ddgiFillChunkTotal;
        }
    }
    workgroupBarrier();
    if (thread == 0u) {
        atomicStore(&ddgiScheduleState[DDGI_STATE_CURSOR_PROBE], ddgiOrderAfterCursor(cursorProbe, ddgiFillUsed, probeCount));
    }
}
`;
}
