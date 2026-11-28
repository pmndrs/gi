/**
 * `ddgi/scroll`: the camera-following volume's GPU half (docs/design/
 * m3-moving.md §4.2; the CPU half and the words used are in scroll.ts).
 * Two entry points, in one timed pass before the volume's other passes:
 *
 * 1. `ddgiScrollReset`, in frames where the window moved: one thread per
 *    probe. A probe in a plane that entered the window starts again: state 3
 *    (not updated yet, so sampling skips it) with relocation offset 0, its
 *    history record cleared (n = 0, k = 0, no flags, no change) and owed
 *    ENTERING_WARMING_UPDATES updates at ENTERING_URGENCY, so the scheduler
 *    updates it first, and its state ends after the second update (§5.2).
 *    Its slot's atlas tiles keep the plane that left until that first update,
 *    which replaces them outright (h = 0); state 3 keeps them from being read
 *    meanwhile. A teleport (or a reset) enters every slot.
 * 2. `ddgiScrollExtent`, every frame: six workgroups, one per side of the
 *    valid extent (low x, y, z, then high x, y, z). Each moves its side
 *    toward the CPU's target: at once when the target is inward (behind the
 *    camera), by at most VALID_EXTENT_STEP_SPACINGS when outward, and only
 *    over planes whose probes have finished warming (at most
 *    WARM_PLANE_COLD_SHARE of them still in state 3).
 *    The extent is kept in a small storage buffer (the GPU can't read and
 *    write the rgba32float table in one dispatch), which the technique then
 *    copies into the volume table's row.
 *
 * Why the extent's side must stay the coverage's reach R inside the warm
 * planes: coverage reaches R spacings past it (full for half a spacing, then
 * fading; R = 1 for a fixed volume's fade, 10.5 for the camera volume's on a
 * long axis), and the cell a covered point falls in must have warm probes at
 * both ends, or a probe finishing its warm-up would make the point's answer
 * jump. For the low side at v, the farthest cell reached has its lower plane
 * at floor(v − R); for the high side, its upper plane at ceil(v + R). A step
 * of under one plane reaches at most one plane further, so each side checks
 * that one plane.
 */

import { PROBE_HISTORY_FLAGS_MASK, PROBE_HISTORY_UPDATES, PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS, PROBE_STATE } from '../layout.ts';
import { ENTERING_URGENCY, ENTERING_WARMING_UPDATES, VALID_EXTENT_STEP_SPACINGS, WARM_PLANE_COLD_SHARE } from '../scroll.ts';

export const SCROLL_ENTRY_POINTS = { reset: 'ddgiScrollReset', extent: 'ddgiScrollExtent' } as const;

/** Threads per workgroup of both entry points. */
export const SCROLL_WORKGROUP_SIZE = 64;

/** The extent kernel's workgroups: one per side of the valid extent. */
export const SCROLL_EXTENT_SIDES = 6;

/**
 * Where each field of `DdgiScroll` sits, in 32-bit words (the WGSL struct
 * below; a unit test checks them under WGSL's layout rules).
 */
export const SCROLL_UNIFORM_WORD = {
    origin: 0,
    spacing: 4,
    counts: 8,
    window: 12,
    moved: 16,
    targetMin: 20,
    targetMax: 24,
    coverageReach: 28,
    layerBase: 32,
    probeBase: 33,
    everySlotEnters: 34,
    extentToTarget: 35,
} as const;
export const SCROLL_UNIFORM_WORDS = 36;
export const SCROLL_UNIFORM_BYTES = SCROLL_UNIFORM_WORDS * 4;

/** The valid extent's buffer: min xyz, a spare word, max xyz, a spare word; laid out as the table's two texels. */
export const SCROLL_EXTENT_WORDS = 8;
export const SCROLL_EXTENT_BYTES = SCROLL_EXTENT_WORDS * 4;

export const DDGI_SCROLL_WGSL = /* wgsl */ `
// ---- ddgi/scroll (packages/ddgi/src/wgsl/scroll.ts) -------------------------------------

struct DdgiScroll {
    origin: vec4f,            // xyz: world position of the window's first probe, after this frame's move
    spacing: vec4f,           // xyz: metres between probes
    counts: vec4i,            // xyz: probes per axis (the window's size)
    window: vec4i,            // xyz: lattice coordinates of the window's first probe, after this frame's move
    moved: vec4i,             // xyz: planes the window moved this frame per axis (the sign is the direction)
    targetMin: vec4f,         // xyz: where the valid extent's low sides head (lattice units)
    targetMax: vec4f,         // xyz: where its high sides head
    coverageReach: vec4f,     // xyz: how far coverage reaches past the extent per axis, in spacings (scroll.ts, coverageReach)
    layerBase: i32,           // the volume's first layer in the shared atlases
    probeBase: u32,           // its first point in the points debug view
    everySlotEnters: u32,     // 1: a teleport or a reset, every slot starts again
    extentToTarget: u32,      // 1: the extent takes its targets outright (with everySlotEnters, and after prepare)
}

const DDGI_SCROLL_WORKGROUP_SIZE: u32 = ${SCROLL_WORKGROUP_SIZE}u;
const DDGI_STATE_NOT_UPDATED_YET: f32 = ${PROBE_STATE.notUpdatedYet}.0;
const DDGI_HISTORY_WORDS: u32 = ${PROBE_HISTORY_WORDS}u;
const DDGI_HISTORY_UPDATES: u32 = ${PROBE_HISTORY_WORD.updates}u;
const DDGI_HISTORY_URGENCY: u32 = ${PROBE_HISTORY_WORD.urgency}u;
const DDGI_HISTORY_SCHEDULE_AND_FLAGS: u32 = ${PROBE_HISTORY_WORD.scheduleAndFlags}u;
const DDGI_HISTORY_FLAGS_MASK: u32 = ${PROBE_HISTORY_FLAGS_MASK}u;
const DDGI_OWED_SHIFT: u32 = ${PROBE_HISTORY_UPDATES.owedShift}u;
// An entering probe owes its two warming updates, at a whole change's urgency (scroll.ts says why).
const DDGI_ENTERING_WARMING_UPDATES: u32 = ${ENTERING_WARMING_UPDATES}u;
const DDGI_ENTERING_URGENCY: f32 = ${ENTERING_URGENCY.toFixed(1)};
// How far the valid extent moves out a frame, in spacings.
const DDGI_EXTENT_STEP: f32 = ${VALID_EXTENT_STEP_SPACINGS};
// A plane is warm with at most this share of its probes still in state 3 (WARM_PLANE_COLD_SHARE says why).
const DDGI_WARM_PLANE_COLD_SHARE: f32 = ${WARM_PLANE_COLD_SHARE};

@group(1) @binding(0) var<uniform> ddgiScroll: DdgiScroll;
// The reset: the volume's history records, feedback requests, probe data and points.
@group(1) @binding(1) var<storage, read_write> ddgiHistoryWords: array<u32>;
@group(1) @binding(2) var<storage, read_write> ddgiFeedbackRequests: array<u32>;
@group(1) @binding(3) var ddgiProbeDataOut: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(4) var ddgiPointsOut: texture_storage_2d<rgba32float, write>;
// The extent: the probe data (read), and the valid extent (min xyz, spare, max xyz, spare).
@group(1) @binding(5) var ddgiProbeData: texture_2d_array<f32>;
@group(1) @binding(6) var<storage, read_write> ddgiValidExtent: array<f32, ${SCROLL_EXTENT_WORDS}>;

// value mod divisor in 0..divisor - 1, also for negative values.
fn ddgiScrollModulo(value: vec3i, divisor: vec3i) -> vec3i {
    return ((value % divisor) + divisor) % divisor;
}

// The slot of window probe 'windowProbe' (the CPU twin is slotOf in scroll.ts).
fn ddgiScrollSlot(windowProbe: vec3i) -> vec3i {
    let counts = ddgiScroll.counts.xyz;
    return ddgiScrollModulo(windowProbe + ddgiScroll.window.xyz, counts);
}

// ---- 1. the reset ----

// Whether window probe 'windowProbe' is in a plane that entered this frame:
// after a move of m > 0 planes up an axis, the window's last m planes are
// new; after m < 0, its first -m.
fn ddgiScrollEntered(windowProbe: vec3i) -> bool {
    if (ddgiScroll.everySlotEnters != 0u) {
        return true;
    }
    let counts = ddgiScroll.counts.xyz;
    let moved = ddgiScroll.moved.xyz;
    let enteredAbove = (moved > vec3i(0)) & (windowProbe >= counts - moved);
    let enteredBelow = (moved < vec3i(0)) & (windowProbe < -moved);
    return any(enteredAbove | enteredBelow);
}

@compute @workgroup_size(DDGI_SCROLL_WORKGROUP_SIZE)
fn ${SCROLL_ENTRY_POINTS.reset}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let counts = ddgiScroll.counts.xyz;
    // Dispatches past 65,535 workgroups use two dimensions.
    let probeIndex = (group.x + group.y * groups.x) * DDGI_SCROLL_WORKGROUP_SIZE + thread;
    if (probeIndex >= u32(counts.x * counts.y * counts.z)) {
        return;
    }
    // Slot (x, y, z) of index x + nx * (z + nz * y), and the window probe it holds now.
    let index = i32(probeIndex);
    let slot = vec3i(index % counts.x, index / (counts.x * counts.z), (index / counts.x) % counts.z);
    let windowProbe = ddgiScrollModulo(slot - ddgiScroll.window.xyz, counts);
    if (!ddgiScrollEntered(windowProbe)) {
        return;
    }
    textureStore(ddgiProbeDataOut, slot.xz, ddgiScroll.layerBase + slot.y, vec4f(0.0, 0.0, 0.0, DDGI_STATE_NOT_UPDATED_YET));

    // The history record starts again, owing its warming updates. The
    // scheduler's stamp (the high bits of scheduleAndFlags) stays: it says
    // when the slot was last listed, so it isn't listed twice in a frame.
    let base = probeIndex * DDGI_HISTORY_WORDS;
    let stamp = ddgiHistoryWords[base + DDGI_HISTORY_SCHEDULE_AND_FLAGS] & ~DDGI_HISTORY_FLAGS_MASK;
    for (var word = 0u; word < DDGI_HISTORY_WORDS; word++) {
        ddgiHistoryWords[base + word] = 0u;
    }
    ddgiHistoryWords[base + DDGI_HISTORY_UPDATES] = DDGI_ENTERING_WARMING_UPDATES << DDGI_OWED_SHIFT;
    ddgiHistoryWords[base + DDGI_HISTORY_URGENCY] = bitcast<u32>(DDGI_ENTERING_URGENCY);
    ddgiHistoryWords[base + DDGI_HISTORY_SCHEDULE_AND_FLAGS] = stamp;
    ddgiFeedbackRequests[probeIndex] = 0u;

    // The points debug view: the probe at its new grid point, not updated yet.
    let pointIndex = ddgiScroll.probeBase + probeIndex;
    let pointsWidth = textureDimensions(ddgiPointsOut).x;
    let position = ddgiScroll.origin.xyz + vec3f(windowProbe) * ddgiScroll.spacing.xyz;
    textureStore(ddgiPointsOut, vec2u(pointIndex % pointsWidth, pointIndex / pointsWidth), vec4f(position, DDGI_STATE_NOT_UPDATED_YET));
}

// ---- 2. the valid extent ----

// Probes still warming (state 3) in the plane being checked.
var<workgroup> ddgiColdProbes: atomic<u32>;

@compute @workgroup_size(DDGI_SCROLL_WORKGROUP_SIZE)
fn ${SCROLL_ENTRY_POINTS.extent}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let side = group.x;
    let axis = side % 3u;
    let high = side >= 3u;
    let entry = select(axis, 4u + axis, high);
    let counts = ddgiScroll.counts.xyz;
    let current = ddgiValidExtent[entry];
    let reach = ddgiScroll.coverageReach[axis];
    // The farthest plane coverage reaches now (the file comment), in lattice units.
    let reachedNow = select(floor(current - reach), ceil(current + reach), high);

    // The one plane a step outward can bring into reach, and whether it is inside the window.
    let plane = i32(reachedNow) + select(-1, 1, high);
    let planeInWindow = plane >= ddgiScroll.window[axis] && plane < ddgiScroll.window[axis] + counts[axis];
    let windowPlane = plane - ddgiScroll.window[axis];
    // Count its probes still warming: every thread walks part of the plane
    // (the loop's bounds are uniform, so the barrier after it is too).
    let acrossAxis = (axis + 1u) % 3u;
    let downAxis = (axis + 2u) % 3u;
    let planeProbes = u32(counts[acrossAxis] * counts[downAxis]);
    for (var first = 0u; first < planeProbes; first += DDGI_SCROLL_WORKGROUP_SIZE) {
        let inPlane = first + thread;
        if (planeInWindow && inPlane < planeProbes) {
            var windowProbe = vec3i(0);
            windowProbe[axis] = windowPlane;
            windowProbe[acrossAxis] = i32(inPlane) % counts[acrossAxis];
            windowProbe[downAxis] = i32(inPlane) / counts[acrossAxis];
            let slot = ddgiScrollSlot(windowProbe);
            let state = textureLoad(ddgiProbeData, slot.xz, ddgiScroll.layerBase + slot.y, 0).a;
            if (round(state) == DDGI_STATE_NOT_UPDATED_YET) {
                atomicAdd(&ddgiColdProbes, 1u);
            }
        }
    }
    workgroupBarrier();
    if (thread != 0u) {
        return;
    }
    let planeWarm = planeInWindow && f32(atomicLoad(&ddgiColdProbes)) <= DDGI_WARM_PLANE_COLD_SHARE * f32(planeProbes);
    let goal = select(ddgiScroll.targetMin[axis], ddgiScroll.targetMax[axis], high);
    var next = goal;
    if (ddgiScroll.extentToTarget == 0u) {
        if (!high && goal < current) {
            // The low side moves out (down) by a step, past the next whole plane only when it is warm.
            next = max(current - DDGI_EXTENT_STEP, goal);
            if (floor(next - reach) < reachedNow && !planeWarm) {
                next = reachedNow + reach;
            }
        } else if (high && goal > current) {
            // The high side moves out (up) likewise.
            next = min(current + DDGI_EXTENT_STEP, goal);
            if (ceil(next + reach) > reachedNow && !planeWarm) {
                next = reachedNow - reach;
            }
        }
        // Otherwise the goal is inward (behind the camera): the side follows it at once.
    }
    ddgiValidExtent[entry] = next;
}
`;
