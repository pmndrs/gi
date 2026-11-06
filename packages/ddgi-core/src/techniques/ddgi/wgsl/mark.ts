/**
 * The `ddgi/mark` kernel: the change estimate (docs/design/m2-lights.md
 * §4.2). It runs only in frames whose `SceneChanges` report a change, before
 * the trace, and tells each probe how much its light changed, from the
 * lights' own numbers and the field's stored data, never from new rays. So
 * ray noise can't fake a change, and a still scene is never touched (T1.5b
 * measured every rule that read the change from the probes' own noisy
 * estimates biasing the field).
 *
 * One thread per (table entry, probe in the entry's box). The CPU writes the
 * table (mark-table.ts): one entry per changed point or spot light, its box
 * the light's influence radius around where it was and where it is, clipped
 * to the volume; and one entry per global change (a directional light, the
 * environment, more light changes than are listed), its box the whole
 * volume. Each entry's threads start at its `firstThread`, so a thread finds
 * its entry by binary search.
 *
 * Per probe, for an entry:
 * - **a local light:** its unshadowed irradiance vector per colour channel,
 *   `E_c × direction to the light`, before and after (three's falloff, range
 *   window and cone: the light-falloff module hit shading uses, without a
 *   surface cosine). `r_c = |after_c − before_c| / mean_c`, with `mean_c` the
 *   probe's stored mean irradiance in that channel, and
 *   `r = lightChangeScale × max_c r_c × max(visibility, 0.25)`;
 * - **a directional light:** `r` from the CPU (the change of its irradiance
 *   vector relative to the old one), times `0.25 + 0.75 × openness` toward
 *   the old and new direction;
 * - **the environment:** `r` from the CPU, times `0.25 + 0.75 × openness`
 *   over six directions;
 * - **everywhere:** `r` from the CPU as it is.
 *
 * Then the probe's record (PROBE_HISTORY_WORD in layout.ts):
 * `pendingChange += r`, `urgency = max(urgency, pendingChange)`, the marked
 * flag, and once `pendingChange` is over the tolerance, `owed = max(owed, K)`
 * updates at priority (T2.4b): K for a local change, and after a global
 * change on a big volume `farProbeUpdates` for probes away from the view
 * (m2-lights.md §6.4). Many entries can mark one probe, so these are atomics.
 * The blend cuts the history from `pendingChange` at the probe's next update,
 * for the updates it was promised.
 * Probes that are off or not updated yet are skipped: off probes owe
 * nothing, and a new probe has no history to cut.
 *
 * The marks are counted for the census (the "probes marked" counters):
 * **subgroups first**, one atomic per subgroup with `subgroupAdd`; the
 * declared fallback, one atomic per marking thread. Both count the same.
 *
 * Bindings (group 1): uniforms, the mark table, the probe history (as
 * atomic words), the census (3 storage buffers), the probe-data and
 * distance atlases (sampled) and the atlas sampler. Platform modules: the
 * light falloff only (no bindings).
 */

import {
    CENSUS_PROBES_MARKED,
    CENSUS_PROBES_MARKED_TOTAL,
    MARK_KIND,
    MARK_LIGHT_KIND,
    PROBE_HISTORY_FLAG,
    PROBE_HISTORY_UPDATES,
    PROBE_HISTORY_WORD,
    PROBE_HISTORY_WORDS,
} from '../layout.ts';
import { NEAR_CAMERA_WEIGHT } from '../settings.ts';
import { DDGI_COMMON_WGSL } from './common.ts';
import type { PhaseBReduction } from './blend.ts';

export const MARK_WORKGROUP_SIZE = 64;

/**
 * The visibility filter's floor (m2-lights.md §4.2): a probe whose distance
 * moments say the light is hidden still gets this share of the change,
 * because it may see lit surfaces through a door.
 */
export const MARK_VISIBILITY_FLOOR = 0.25;

/**
 * Below this mean irradiance a probe counts as dark: a light's change there
 * is measured against this instead of dividing by (almost) zero. Tiny on
 * purpose: a light switched on next to a dark probe really is a change of
 * 100% and more, and the probe should take the new light at once.
 */
export const MARK_DARK_IRRADIANCE = 1e-6;

/** How the kernel counts its marks for the census: subgroups first, or the declared fallback. */
export type MarkReduction = PhaseBReduction;

function countWgsl(reduction: MarkReduction): string {
    if (reduction === 'subgroups') {
        return /* wgsl */ `
// Subgroups first: one atomic per subgroup for the census. Called by every
// thread, in uniform control flow.
fn ddgiCountMarks(marked: bool) {
    let marks = subgroupAdd(select(0u, 1u, marked));
    if (subgroupElect() && marks > 0u) {
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_MARKED], marks);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_MARKED_TOTAL], marks);
    }
}
`;
    }
    return /* wgsl */ `
// The declared fallback without subgroups: each marking thread counts itself.
fn ddgiCountMarks(marked: bool) {
    if (marked) {
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_MARKED], 1u);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_MARKED_TOTAL], 1u);
    }
}
`;
}

export function markKernelWgsl(reduction: MarkReduction): string {
    const enable = reduction === 'subgroups' ? 'enable subgroups;\n' : '';
    return /* wgsl */ `${enable}
${DDGI_COMMON_WGSL}

// ---- DDGI mark kernel (techniques/ddgi) ------------------------------------------------

const DDGI_MARK_WORKGROUP_SIZE: u32 = ${MARK_WORKGROUP_SIZE}u;
const DDGI_CENSUS_PROBES_MARKED: u32 = ${CENSUS_PROBES_MARKED}u;
const DDGI_CENSUS_PROBES_MARKED_TOTAL: u32 = ${CENSUS_PROBES_MARKED_TOTAL}u;
const DDGI_MARK_LOCAL_LIGHT: u32 = ${MARK_KIND.localLight}u;
const DDGI_MARK_DIRECTIONAL_LIGHT: u32 = ${MARK_KIND.directionalLight}u;
const DDGI_MARK_ENVIRONMENT: u32 = ${MARK_KIND.environment}u;
const DDGI_MARK_LIGHT_SPOT: u32 = ${MARK_LIGHT_KIND.spot}u;
const DDGI_MARK_VISIBILITY_FLOOR: f32 = ${MARK_VISIBILITY_FLOOR};
const DDGI_MARK_DARK_IRRADIANCE: f32 = ${MARK_DARK_IRRADIANCE};
// The openness factor for global changes (m2-lights.md §4.2): a probe that
// sees open sky sees the change directly, one deep inside still gets a quarter.
const DDGI_OPENNESS_FLOOR: f32 = 0.25;
// The probe history, read as words (PROBE_HISTORY_WORD in layout.ts).
const DDGI_HISTORY_WORDS: u32 = ${PROBE_HISTORY_WORDS}u;
const DDGI_HISTORY_PENDING_CHANGE: u32 = ${PROBE_HISTORY_WORD.pendingChange}u;
const DDGI_HISTORY_URGENCY: u32 = ${PROBE_HISTORY_WORD.urgency}u;
const DDGI_HISTORY_SCHEDULE_AND_FLAGS: u32 = ${PROBE_HISTORY_WORD.scheduleAndFlags}u;
const DDGI_HISTORY_MEAN_RED_GREEN: u32 = ${PROBE_HISTORY_WORD.meanRedGreen}u;
const DDGI_HISTORY_MEAN_BLUE_NOISE: u32 = ${PROBE_HISTORY_WORD.meanBlueNoise}u;
const DDGI_FLAG_MARKED: u32 = ${PROBE_HISTORY_FLAG.marked}u;
const DDGI_FLAG_TAKING_CHANGE: u32 = ${PROBE_HISTORY_FLAG.takingChange}u;
const DDGI_HISTORY_UPDATES: u32 = ${PROBE_HISTORY_WORD.updates}u;
const DDGI_OWED_SHIFT: u32 = ${PROBE_HISTORY_UPDATES.owedShift}u;
const DDGI_OWED_MASK: u32 = ${PROBE_HISTORY_UPDATES.owedMask}u;
// After a global change, probes at least this near the camera's view owe the full K (m2-lights.md §6.4).
const DDGI_NEAR_CAMERA_WEIGHT: f32 = ${NEAR_CAMERA_WEIGHT};

// One light's state in the table (MARK_LIGHT_WORD in layout.ts). A side that
// doesn't exist (a light created or destroyed this frame) has intensity 0.
struct DdgiMarkLight {
    position: vec3f,
    kind: u32,              // MARK_LIGHT_KIND
    direction: vec3f,       // unit, the way the light travels (spot, directional)
    range: f32,             // three's window; 0 = none
    intensity: vec3f,       // colour x intensity, platform units
    cosInnerCone: f32,
    cosOuterCone: f32,
    spare0: f32,
    spare1: f32,
    spare2: f32,
}

// One entry (MARK_ENTRY_WORD in layout.ts): a box of probes and what changed there.
struct DdgiMarkEntry {
    boxMin: vec3u,
    kind: u32,              // MARK_KIND
    boxSize: vec3u,
    firstThread: u32,       // the threads before this entry's
    globalChange: f32,      // r for the global kinds, worked out on the CPU
    spare0: f32,
    spare1: f32,
    spare2: f32,
    before: DdgiMarkLight,
    after: DdgiMarkLight,
}

struct DdgiMarkTable {
    entryCount: u32,
    threadCount: u32,
    spare0: u32,
    spare1: u32,
    entries: array<DdgiMarkEntry>,
}

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read> ddgiMarkTable: DdgiMarkTable;
@group(1) @binding(2) var<storage, read_write> ddgiProbeHistoryWords: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> ddgiCensus: array<atomic<u32>>;
@group(1) @binding(4) var ddgiProbeData: texture_2d_array<f32>;
@group(1) @binding(5) var ddgiDistanceAtlas: texture_2d_array<f32>;
@group(1) @binding(6) var ddgiAtlasSampler: sampler;

${countWgsl(reduction)}

// The entry a thread belongs to: the last one whose firstThread is at most
// the thread's index (entries are in order of firstThread).
fn ddgiMarkEntryOf(thread: u32) -> u32 {
    var low = 0u;
    var high = ddgiMarkTable.entryCount;
    while (high - low > 1u) {
        let middle = (low + high) / 2u;
        if (ddgiMarkTable.entries[middle].firstThread <= thread) {
            low = middle;
        } else {
            high = middle;
        }
    }
    return low;
}

// A point or spot light's unshadowed irradiance vector at a point, per colour
// channel: column c is E_c times the unit direction to the light. No surface
// cosine: the vector is what a probe at the point gets from the light, before
// any surface turns it into a bounce.
fn ddgiMarkIrradianceVectors(light: DdgiMarkLight, point: vec3f) -> mat3x3f {
    let toLight = light.position - point;
    let lightDistance = length(toLight);
    let direction = toLight / max(lightDistance, 1.0e-20);
    var attenuation = giShadeDistanceAttenuation(lightDistance, light.range);
    if (light.kind == DDGI_MARK_LIGHT_SPOT) {
        attenuation *= giShadeSpotAttenuation(dot(-direction, light.direction), light.cosOuterCone, light.cosInnerCone);
    }
    let irradiance = light.intensity * attenuation;
    return mat3x3f(direction * irradiance.r, direction * irradiance.g, direction * irradiance.b);
}

// How visible a point at 'distance' along 'direction' is from the probe, by
// one Chebyshev test on the probe's distance moments, as sampling does
// (ddgiAccumulateField). The stored distances stop at distanceClamp, so the
// test can't see past it: beyond it, "nothing nearer" is all it can say,
// and the test asks only that much.
fn ddgiMarkVisibility(probe: vec3i, direction: vec3f, distance: f32) -> f32 {
    let volume = ddgiUniformVolume();
    let scale = ddgiAtlasScale(volume.distanceTexels, volume.counts);
    let uv = ddgiTileUv(ddgiTileTexel(direction, volume.distanceTexels), probe, scale);
    let moments = textureSampleLevel(ddgiDistanceAtlas, ddgiAtlasSampler, uv, probe.y, 0.0).xy;
    let tested = min(distance, ddgi.distanceClamp);
    if (tested <= moments.x) {
        return 1.0;
    }
    let variance = abs(moments.y - moments.x * moments.x);
    let excess = tested - moments.x;
    return variance / (variance + excess * excess);
}

// A local light's change at one probe (the file comment).
fn ddgiMarkLocalChange(entry: DdgiMarkEntry, probe: vec3i, position: vec3f, mean: vec3f) -> f32 {
    let before = ddgiMarkIrradianceVectors(entry.before, position);
    let after = ddgiMarkIrradianceVectors(entry.after, position);
    // Per channel, as vectors: a light that keeps its distance but moves to
    // the other side of the probe changes the vector and not its length, and
    // a colour change at equal brightness changes a channel and not the
    // luminance; a luminance difference would miss both.
    let change = vec3f(length(after[0] - before[0]), length(after[1] - before[1]), length(after[2] - before[2]));
    let relative = change / max(mean, vec3f(DDGI_MARK_DARK_IRRADIANCE));
    // Visibility toward wherever the light is lit: the larger of the two sides.
    let visibility = max(ddgiMarkLightVisibility(entry.before, probe, position), ddgiMarkLightVisibility(entry.after, probe, position));
    return ddgi.lightChangeScale * max(relative.r, max(relative.g, relative.b)) * max(visibility, DDGI_MARK_VISIBILITY_FLOOR);
}

// Visibility of a local light's position from the probe; 0 for a side that
// isn't lit (a light created or destroyed this frame).
fn ddgiMarkLightVisibility(light: DdgiMarkLight, probe: vec3i, position: vec3f) -> f32 {
    if (!any(light.intensity > vec3f(0.0))) {
        return 0.0;
    }
    let toLight = light.position - position;
    let lightDistance = length(toLight);
    return ddgiMarkVisibility(probe, toLight / max(lightDistance, 1.0e-20), lightDistance);
}

// How much open space the probe sees in a direction, 0..1: the same test at
// the distance clamp, the farthest the moments know about.
fn ddgiMarkOpenness(probe: vec3i, direction: vec3f) -> f32 {
    return ddgiMarkVisibility(probe, direction, ddgi.distanceClamp);
}

// A global change's share at one probe (the file comment).
fn ddgiMarkGlobalChange(entry: DdgiMarkEntry, probe: vec3i) -> f32 {
    if (entry.kind == DDGI_MARK_DIRECTIONAL_LIGHT) {
        // Toward the light: against the way it travels, old and new.
        let openness = max(ddgiMarkOpenness(probe, -entry.before.direction), ddgiMarkOpenness(probe, -entry.after.direction));
        return entry.globalChange * (DDGI_OPENNESS_FLOOR + (1.0 - DDGI_OPENNESS_FLOOR) * openness);
    }
    if (entry.kind == DDGI_MARK_ENVIRONMENT) {
        // The sky can come from anywhere: the mean over the six axis directions.
        var openness = 0.0;
        for (var axis = 0u; axis < 6u; axis++) {
            var direction = vec3f(0.0);
            direction[axis % 3u] = select(1.0, -1.0, axis >= 3u);
            openness += ddgiMarkOpenness(probe, direction);
        }
        return entry.globalChange * (DDGI_OPENNESS_FLOOR + (1.0 - DDGI_OPENNESS_FLOOR) * openness / 6.0);
    }
    return entry.globalChange;
}

// Adds r to the probe's record: pendingChange += r (a compare-and-swap loop,
// WGSL has no float atomics), urgency = max(urgency, pendingChange) (r and
// the sums are never negative, and non-negative floats order like their
// bits), and the marked flag. Once the change adds up to more than the
// tolerance, the probe is owed 'promised' updates at priority (the GPU
// scheduler keeps the promise; a smaller change cuts nothing, so it isn't
// worth the budget).
fn ddgiMarkProbe(probeIndex: u32, change: f32, promised: u32) {
    let base = probeIndex * DDGI_HISTORY_WORDS;
    var old = atomicLoad(&ddgiProbeHistoryWords[base + DDGI_HISTORY_PENDING_CHANGE]);
    var pending = bitcast<f32>(old) + change;
    loop {
        let swapped = atomicCompareExchangeWeak(&ddgiProbeHistoryWords[base + DDGI_HISTORY_PENDING_CHANGE], old, bitcast<u32>(pending));
        if (swapped.exchanged) {
            break;
        }
        old = swapped.old_value;
        pending = bitcast<f32>(old) + change;
    }
    atomicMax(&ddgiProbeHistoryWords[base + DDGI_HISTORY_URGENCY], bitcast<u32>(pending));
    atomicOr(&ddgiProbeHistoryWords[base + DDGI_HISTORY_SCHEDULE_AND_FLAGS], DDGI_FLAG_MARKED);
    if (pending > ddgi.changeTolerance) {
        ddgiOweUpdates(base, promised);
        atomicOr(&ddgiProbeHistoryWords[base + DDGI_HISTORY_SCHEDULE_AND_FLAGS], DDGI_FLAG_TAKING_CHANGE);
    }
}

// owed = max(owed, promised), in bits 16-23 of the updates word, whose low
// bits (k) other threads don't touch in this dispatch: a compare-and-swap
// loop, because WGSL has no atomic on part of a word.
fn ddgiOweUpdates(base: u32, promised: u32) {
    var old = atomicLoad(&ddgiProbeHistoryWords[base + DDGI_HISTORY_UPDATES]);
    loop {
        let owed = (old >> DDGI_OWED_SHIFT) & DDGI_OWED_MASK;
        if (owed >= promised) {
            break;
        }
        let next = (old & ~(DDGI_OWED_MASK << DDGI_OWED_SHIFT)) | (promised << DDGI_OWED_SHIFT);
        let swapped = atomicCompareExchangeWeak(&ddgiProbeHistoryWords[base + DDGI_HISTORY_UPDATES], old, next);
        if (swapped.exchanged) {
            break;
        }
        old = swapped.old_value;
    }
}

// Whole updates, 1..255 (the owed count's 8 bits).
fn ddgiWholeUpdates(updates: f32) -> u32 {
    return clamp(u32(round(updates)), 1u, DDGI_OWED_MASK);
}

// How many updates a change promises the probe (m2-lights.md §6.4): K for a
// local change, and for a global one too near the camera's view; far probes
// owe fewer after a global change on a big volume (farOwedUpdates is K when
// that rule is off this frame).
fn ddgiPromisedUpdates(entry: DdgiMarkEntry, probe: vec3i) -> u32 {
    let full = ddgiWholeUpdates(ddgi.changeUpdates);
    if (entry.kind == DDGI_MARK_LOCAL_LIGHT || ddgiCameraWeight(probe) >= DDGI_NEAR_CAMERA_WEIGHT) {
        return full;
    }
    return min(full, ddgiWholeUpdates(ddgi.farOwedUpdates));
}

// The probe's stored mean irradiance (written by ddgi/blend).
fn ddgiMarkStoredMean(probeIndex: u32) -> vec3f {
    let base = probeIndex * DDGI_HISTORY_WORDS;
    let redGreen = unpack2x16float(atomicLoad(&ddgiProbeHistoryWords[base + DDGI_HISTORY_MEAN_RED_GREEN]));
    let blue = unpack2x16float(atomicLoad(&ddgiProbeHistoryWords[base + DDGI_HISTORY_MEAN_BLUE_NOISE])).x;
    return vec3f(redGreen, blue);
}

// One thread's mark: the change at its probe for its entry, or 0 when it has
// none (past the end, a probe that is off or new, or no change there).
fn ddgiMarkThread(thread: u32) -> bool {
    if (thread >= ddgiMarkTable.threadCount) {
        return false;
    }
    let entry = ddgiMarkTable.entries[ddgiMarkEntryOf(thread)];
    let local = thread - entry.firstThread;
    let inBox = vec3u(local % entry.boxSize.x, local / (entry.boxSize.x * entry.boxSize.z), (local / entry.boxSize.x) % entry.boxSize.z);
    let probe = vec3i(entry.boxMin + inBox);
    let counts = ddgiProbeCount();
    let probeIndex = u32(probe.x) + counts.x * (u32(probe.z) + counts.z * u32(probe.y));
    let data = ddgiReadProbeData(ddgiProbeData, probe);
    if (data.state != DDGI_STATE_ACTIVE) {
        return false;
    }
    var change = 0.0;
    if (entry.kind == DDGI_MARK_LOCAL_LIGHT) {
        change = ddgiMarkLocalChange(entry, probe, ddgiProbePosition(probe, data.offset), ddgiMarkStoredMean(probeIndex));
    } else {
        change = ddgiMarkGlobalChange(entry, probe);
    }
    if (!(change > 0.0)) {
        return false;
    }
    ddgiMarkProbe(probeIndex, change, ddgiPromisedUpdates(entry, probe));
    return true;
}

@compute @workgroup_size(DDGI_MARK_WORKGROUP_SIZE)
fn ddgiMark(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localThread: u32,
) {
    // Dispatches past 65,535 workgroups use two dimensions.
    let thread = (group.x + group.y * groups.x) * DDGI_MARK_WORKGROUP_SIZE + localThread;
    let marked = ddgiMarkThread(thread);
    ddgiCountMarks(marked);
}
`;
}
