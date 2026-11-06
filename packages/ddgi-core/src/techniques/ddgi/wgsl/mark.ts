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
 *   surface cosine). `r_c = lightChangeScale × |after_c − before_c| / mean_c
 *   × max(visibility, 0.25)`, with `mean_c` the probe's stored mean
 *   irradiance in that channel; where the light at the probe got weaker,
 *   against what is left, `r_c / (1 − r_c)`; `r = max_c r_c`;
 * - **a directional light:** `r` from the CPU (the change of its irradiance
 *   vector relative to the old one), times `0.25 + 0.75 × openness` toward
 *   the old and new direction;
 * - **the environment:** `r` from the CPU, times `0.25 + 0.75 × openness`
 *   over six directions;
 * - **everywhere:** `r` from the CPU as it is;
 * - **geometry** (T2.M1, docs/design/m3-moving.md §5.1; one entry per old or
 *   new box of a moved object, or the box of one added, removed, shown or
 *   hidden): with `s = min(1, moved / spacing)` for a move and 1 for a whole
 *   change,
 *   - the **near set**, probes within NEAR_SET_SPACINGS of the box, off
 *     probes included (the change may free them): `r = s`. They are traced
 *     themselves, with the full count of updates, since their distances
 *     change too;
 *   - the **far set**, active probes within FAR_SET_SPACINGS that can see the
 *     box: their stored mean distance toward its centre reaches where that
 *     direction enters the box, less half a spacing (a test against the
 *     centre itself would fail for the old box, since the object hid its own
 *     centre); or their mover mask has the object's bit (they saw it at their
 *     last update). `r = min(1, Ω / Ω_ref) × s`, Ω about
 *     π × radius² / distance², the share of the view the object covers.
 *     They are served like a local light's probes (keys and propagation on
 *     the key-probe path).
 *
 * Then the probe's record (PROBE_HISTORY_WORD in layout.ts):
 * `pendingChange += r`, `urgency = max(urgency, pendingChange)`, the marked
 * flag, and once `pendingChange` is over the tolerance, `owed = max(owed, K)`
 * updates at priority (T2.4b): K for a local change, and after a global
 * change on a big volume `farProbeUpdates` for probes away from the view
 * (m2-lights.md §6.4). Many entries can mark one probe, so these are atomics.
 *
 * **Key probes** (T2.4c, m2-lights.md §4.5), when the frame's change takes
 * the key-probe path (`keyProbes`, on a volume too big for every probe's K
 * updates): a marked key probe's urgency is its change times
 * `keyProbeBoost`, and away from the view's core it owes `keyProbeUpdates`;
 * a fine probe away from the view's core (camera weight under 0.5) waits for
 * its keys: its urgency is its change times `keyResidualShare`, it owes
 * `fineOwedUpdates` (1), and it is flagged as awaiting a change, so
 * `ddgi/propagate` passes it its keys' change meanwhile and its own blend
 * cuts its history only for what propagation hasn't brought. Probes near and
 * in view keep their full urgency and owed updates, whatever their level.
 * Without the key-probe path, "far probes owe 2" is the fallback.
 *
 * **The change clock** (T2.4c): the largest change marked this frame goes
 * into the census (`atomicMax`), and a second, one-thread dispatch
 * (`ddgiAdvanceChangeClock`) adds it to the volume's change clock, which
 * every probe stamps at its update and propagation measures shares of a
 * key's change by.
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
 * light falloff only (no bindings). The clock dispatch uses the same group.
 */

import {
    CENSUS_CHANGE_CLOCK,
    CENSUS_LARGEST_CHANGE,
    CENSUS_PROBES_MARKED,
    CENSUS_PROBES_MARKED_TOTAL,
    CHANGE_CLOCK_UNITS,
    MARK_KIND,
    MARK_LIGHT_KIND,
    PROBE_HISTORY_FLAG,
    PROBE_HISTORY_UPDATES,
    PROBE_HISTORY_WORD,
    PROBE_HISTORY_WORDS,
} from '../layout.ts';
import { FAR_SET_REFERENCE_SOLID_ANGLE, FAR_SET_SPACINGS, MAX_GLOBAL_CHANGE, NEAR_SET_SPACINGS } from '../mark-table.ts';
import { NEAR_CAMERA_WEIGHT } from '../settings.ts';
import { DDGI_COMMON_WGSL } from './common.ts';
import type { PhaseBReduction } from './blend.ts';

export const MARK_WORKGROUP_SIZE = 64;

/** The mark module's entry points: the mark, then the clock (one thread). */
export const MARK_ENTRY_POINTS = { mark: 'ddgiMark', clock: 'ddgiAdvanceChangeClock' } as const;

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
// Subgroups first: one atomic per subgroup for the census, and the largest
// change for the change clock. Called by every thread, in uniform control
// flow; the change is 0 for a thread that marked nothing.
fn ddgiCountMarks(change: f32) {
    let marked = change > 0.0;
    let marks = subgroupAdd(select(0u, 1u, marked));
    let largest = subgroupMax(change);
    if (subgroupElect() && marks > 0u) {
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_MARKED], marks);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_MARKED_TOTAL], marks);
        // Non-negative floats order like their bits.
        atomicMax(&ddgiCensus[DDGI_CENSUS_LARGEST_CHANGE], bitcast<u32>(largest));
    }
}
`;
    }
    return /* wgsl */ `
// The declared fallback without subgroups: each marking thread counts itself.
fn ddgiCountMarks(change: f32) {
    if (change > 0.0) {
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_MARKED], 1u);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_MARKED_TOTAL], 1u);
        atomicMax(&ddgiCensus[DDGI_CENSUS_LARGEST_CHANGE], bitcast<u32>(change));
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
const DDGI_CENSUS_LARGEST_CHANGE: u32 = ${CENSUS_LARGEST_CHANGE}u;
const DDGI_CENSUS_CHANGE_CLOCK: u32 = ${CENSUS_CHANGE_CLOCK}u;
// The change clock's fixed point: units per 1.0 of relative change (CHANGE_CLOCK_UNITS in layout.ts).
const DDGI_CHANGE_CLOCK_UNITS: f32 = ${CHANGE_CLOCK_UNITS}.0;
const DDGI_MARK_LOCAL_LIGHT: u32 = ${MARK_KIND.localLight}u;
const DDGI_MARK_DIRECTIONAL_LIGHT: u32 = ${MARK_KIND.directionalLight}u;
const DDGI_MARK_ENVIRONMENT: u32 = ${MARK_KIND.environment}u;
const DDGI_MARK_LIGHT_SPOT: u32 = ${MARK_LIGHT_KIND.spot}u;
const DDGI_MARK_GEOMETRY: u32 = ${MARK_KIND.geometry}u;
// Geometry changes (m3-moving.md §5.1; NEAR_SET_SPACINGS etc. in mark-table.ts).
const DDGI_NEAR_SET_SPACINGS: f32 = ${NEAR_SET_SPACINGS};
const DDGI_FAR_SET_SPACINGS: f32 = ${FAR_SET_SPACINGS};
const DDGI_FAR_SET_REFERENCE_SOLID_ANGLE: f32 = ${FAR_SET_REFERENCE_SOLID_ANGLE};
// A far probe sees the box when its mean distance toward it reaches the box's near face, less this share of a spacing.
const DDGI_FAR_SET_DISTANCE_SLACK_SPACINGS: f32 = 0.5;
const DDGI_HISTORY_MOVER_MASK: u32 = ${PROBE_HISTORY_WORD.moverMask}u;
const DDGI_MARK_VISIBILITY_FLOOR: f32 = ${MARK_VISIBILITY_FLOOR};
const DDGI_MARK_DARK_IRRADIANCE: f32 = ${MARK_DARK_IRRADIANCE};
// A darkening leaves at least this share of the light, as far as its change is
// measured (so a light switched off counts as a change of 100x, MAX_GLOBAL_CHANGE).
const DDGI_MARK_SMALLEST_REMAINDER: f32 = ${1 / MAX_GLOBAL_CHANGE};
// Keeps ln(rho) finite in ddgiNeededUpdates.
const DDGI_MARK_SMALLEST_BOUNCE_SHARE: f32 = 0.001;
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
const DDGI_FLAG_AWAITING_CHANGE: u32 = ${PROBE_HISTORY_FLAG.awaitingChange}u;
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

// An entry's geometry block (MARK_GEOMETRY_WORD in layout.ts), for MARK_KIND.geometry.
struct DdgiMarkGeometry {
    regionMin: vec3f,       // the changed region's world box
    moved: f32,             // how far the object moved, metres; negative for a whole change (added, removed, shown, hidden)
    regionMax: vec3f,
    moverBit: u32,          // its bit in the probes' mover masks; 0 when it isn't a hot mover
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
    geometry: DdgiMarkGeometry,
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

// A change at one probe: r, which cuts its history, and the error it leaves
// against the new light, which sets how many updates it needs (ddgiNeededUpdates).
struct DdgiProbeChange {
    change: f32,
    againstNew: f32,
}

// A local light's change at one probe (the file comment).
fn ddgiMarkLocalChange(entry: DdgiMarkEntry, probe: vec3i, position: vec3f, mean: vec3f) -> DdgiProbeChange {
    let before = ddgiMarkIrradianceVectors(entry.before, position);
    let after = ddgiMarkIrradianceVectors(entry.after, position);
    // Per channel, as vectors: a light that keeps its distance but moves to
    // the other side of the probe changes the vector and not its length, and
    // a colour change at equal brightness changes a channel and not the
    // luminance; a luminance difference would miss both.
    let change = vec3f(length(after[0] - before[0]), length(after[1] - before[1]), length(after[2] - before[2]));
    // Visibility toward wherever the light is lit: the larger of the two sides.
    let visibility = max(ddgiMarkLightVisibility(entry.before, probe, position), ddgiMarkLightVisibility(entry.after, probe, position));
    let relative = ddgi.lightChangeScale * change / max(mean, vec3f(DDGI_MARK_DARK_IRRADIANCE)) * max(visibility, DDGI_MARK_VISIBILITY_FLOOR);
    // A darkening is measured against the light that is left: the tolerance
    // is a share of the new light, and a light halved leaves an error of 100%
    // of it, not 50% (T2.4b). A brightening stays against the old light, so
    // both directions are cut at least by the size of the change.
    let darker = vec3<bool>(length(after[0]) < length(before[0]), length(after[1]) < length(before[1]), length(after[2]) < length(before[2]));
    let cut = select(relative, relative / max(vec3f(1.0) - relative, vec3f(DDGI_MARK_SMALLEST_REMAINDER)), darker);
    // Against the new light a brightening by r is r / (1 + r): a lamp switched
    // on next to a dark probe is a 100% error, which the bounces carry in a
    // known number of updates, however dark the probe was.
    let againstNew = select(relative / (vec3f(1.0) + relative), cut, darker);
    return DdgiProbeChange(max(cut.r, max(cut.g, cut.b)), max(againstNew.r, max(againstNew.g, againstNew.b)));
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

// How a change is served at one probe (ddgiServing): the updates it is
// promised, the share of its change its urgency is, and whether it waits for
// its keys (a fine probe away from the view on the key-probe path).
struct DdgiServing {
    promised: u32,
    urgencyShare: f32,
    awaiting: bool,
}

// Adds r to the probe's record: pendingChange += r (a compare-and-swap loop,
// WGSL has no float atomics), urgency = max(urgency, pendingChange times its
// share) (r and the sums are never negative, and non-negative floats order
// like their bits), the marked flag, and the awaiting flag for a fine probe
// that waits for its keys. Once the change adds up to more than the
// tolerance, the probe is owed its promised updates at priority (the GPU
// scheduler keeps the promise; a smaller change cuts nothing, so it isn't
// worth the budget).
fn ddgiMarkProbe(probeIndex: u32, change: f32, serving: DdgiServing) {
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
    atomicMax(&ddgiProbeHistoryWords[base + DDGI_HISTORY_URGENCY], bitcast<u32>(pending * serving.urgencyShare));
    atomicOr(&ddgiProbeHistoryWords[base + DDGI_HISTORY_SCHEDULE_AND_FLAGS], DDGI_FLAG_MARKED | select(0u, DDGI_FLAG_AWAITING_CHANGE, serving.awaiting));
    if (pending > ddgi.changeTolerance) {
        ddgiOweUpdates(base, serving.promised);
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

// How a change is served at the probe (m2-lights.md §4.5, §6.4). Near the
// camera's view (camera weight 0.5 or more), at any level: K updates, or as
// many as the bounces need (ddgiNeededUpdates) when that is more, at full
// urgency (a key's boosted). Away from it:
// - on the key-probe path (keyProbes, this frame): a key owes keyOwedUpdates
//   at keyProbeBoost times the urgency, so the keys go first; a fine probe
//   owes fineOwedUpdates at keyResidualShare of it and waits for its keys,
//   whose updates bring it most of the way through ddgi/propagate;
// - otherwise, after a global change on a big volume, "far probes owe 2"
//   (farOwedUpdates is K when that rule is off this frame), left noisier
//   and later for it; after a local change, the full count.
fn ddgiServing(entry: DdgiMarkEntry, probe: vec3i, againstNew: f32) -> DdgiServing {
    let full = max(ddgiWholeUpdates(ddgi.changeUpdates), ddgiNeededUpdates(againstNew));
    let near = ddgiCameraWeight(probe) >= DDGI_NEAR_CAMERA_WEIGHT;
    if (ddgi.keyProbes != 0u) {
        let key = ddgiIsKeyProbe(probe);
        let share = select(1.0, ddgi.keyProbeBoost, key);
        if (near) {
            return DdgiServing(full, share, false);
        }
        if (key) {
            return DdgiServing(min(full, ddgiWholeUpdates(ddgi.keyOwedUpdates)), share, false);
        }
        return DdgiServing(ddgiWholeUpdates(ddgi.fineOwedUpdates), ddgi.keyResidualShare, true);
    }
    // A geometry change's far set is local too: the full count, like a light's.
    if (entry.kind == DDGI_MARK_LOCAL_LIGHT || entry.kind == DDGI_MARK_GEOMETRY || near) {
        return DdgiServing(full, 1.0, false);
    }
    return DdgiServing(min(full, ddgiWholeUpdates(ddgi.farOwedUpdates)), 1.0, false);
}

// The updates a change needs at best (T2.4b): even an update that keeps
// nothing of the old value (h = 0) bounces its estimate off the field, which
// still holds the old light, so the error against the new light shrinks by
// rho (changeBounceShare) an update at most. Within the tolerance after
// ln(epsilon / error) / ln(rho) of them. On two-room (rho about 0.8) a light
// halved needs 14, not K = 8.
fn ddgiNeededUpdates(againstNew: f32) -> u32 {
    let tolerance = ddgi.changeTolerance;
    if (!(againstNew > tolerance)) {
        return 0u;
    }
    let bounce = clamp(ddgi.changeBounceShare, DDGI_MARK_SMALLEST_BOUNCE_SHARE, 1.0 - DDGI_MARK_SMALLEST_BOUNCE_SHARE);
    return min(u32(ceil(log(tolerance / againstNew) / log(bounce))), DDGI_OWED_MASK);
}

// The updates a change of r promises a probe that is traced itself: K, or as many as the bounces need.
fn ddgiFullUpdates(againstNew: f32) -> u32 {
    return max(ddgiWholeUpdates(ddgi.changeUpdates), ddgiNeededUpdates(againstNew));
}

// Where a ray from 'origin' along 'direction' enters a box: 0 when it starts
// inside, a large distance when it misses (the slab test).
fn ddgiRayBoxEntry(origin: vec3f, direction: vec3f, boxMin: vec3f, boxMax: vec3f) -> f32 {
    let inverse = 1.0 / select(direction, vec3f(1.0e-12), abs(direction) < vec3f(1.0e-12));
    let toMin = (boxMin - origin) * inverse;
    let toMax = (boxMax - origin) * inverse;
    let near = max(max(min(toMin.x, toMax.x), min(toMin.y, toMax.y)), min(toMin.z, toMax.z));
    let far = min(min(max(toMin.x, toMax.x), max(toMin.y, toMax.y)), max(toMin.z, toMax.z));
    return select(3.0e38, max(near, 0.0), far >= max(near, 0.0));
}

// Whether the probe can see the region: its stored mean distance toward the
// box's centre reaches where that direction enters the box, less half a
// spacing. The stored distances stop at distanceClamp, so past it "nothing
// nearer than the clamp" is the most the moments can say, and that counts.
fn ddgiMarkSeesRegion(probe: vec3i, direction: vec3f, entryDistance: f32, smallestSpacing: f32) -> bool {
    let volume = ddgiUniformVolume();
    let scale = ddgiAtlasScale(volume.distanceTexels, volume.counts);
    let uv = ddgiTileUv(ddgiTileTexel(direction, volume.distanceTexels), probe, scale);
    let meanDistance = textureSampleLevel(ddgiDistanceAtlas, ddgiAtlasSampler, uv, probe.y, 0.0).x;
    return meanDistance >= min(entryDistance, ddgi.distanceClamp) - DDGI_FAR_SET_DISTANCE_SLACK_SPACINGS * smallestSpacing;
}

// Whether the probe's latest blended update saw the object (its mover mask has the object's bit).
fn ddgiMarkSawMover(probeIndex: u32, moverBit: u32) -> bool {
    let mask = atomicLoad(&ddgiProbeHistoryWords[probeIndex * DDGI_HISTORY_WORDS + DDGI_HISTORY_MOVER_MASK]);
    return (mask & moverBit) != 0u;
}

// A geometry entry's mark at one probe (the file comment): returns the change
// it marked, 0 for none. Off probes count in the near set (the change may
// free them); probes not updated yet never do (their first update starts
// from nothing anyway).
fn ddgiMarkGeometry(entry: DdgiMarkEntry, probe: vec3i, probeIndex: u32, data: DdgiProbeData) -> f32 {
    let geometry = entry.geometry;
    let spacing = ddgiUniformVolume().spacing;
    let smallestSpacing = min(spacing.x, min(spacing.y, spacing.z));
    // How much of a whole change the move is: a drag that moves a little every frame marks a little every frame.
    let moveShare = select(min(1.0, geometry.moved / smallestSpacing), 1.0, geometry.moved < 0.0);
    if (!(moveShare > 0.0) || data.state == DDGI_STATE_NOT_UPDATED_YET) {
        return 0.0;
    }
    let position = ddgiProbePosition(probe, data.offset);
    let nearMargin = DDGI_NEAR_SET_SPACINGS * spacing;
    if (all(position >= geometry.regionMin - nearMargin) && all(position <= geometry.regionMax + nearMargin)) {
        ddgiMarkProbe(probeIndex, moveShare, DdgiServing(ddgiFullUpdates(moveShare), 1.0, false));
        return moveShare;
    }
    if (data.state != DDGI_STATE_ACTIVE) {
        return 0.0;
    }
    // The probe box covers the reach by grid index; this checks the probe where relocation put it.
    let farMargin = DDGI_FAR_SET_SPACINGS * spacing;
    if (any(position < geometry.regionMin - farMargin) || any(position > geometry.regionMax + farMargin)) {
        return 0.0;
    }
    let toCentre = 0.5 * (geometry.regionMin + geometry.regionMax) - position;
    let distance = max(length(toCentre), 1.0e-6);
    let direction = toCentre / distance;
    let entryDistance = ddgiRayBoxEntry(position, direction, geometry.regionMin, geometry.regionMax);
    let seen = ddgiMarkSeesRegion(probe, direction, entryDistance, smallestSpacing) || (geometry.moverBit != 0u && ddgiMarkSawMover(probeIndex, geometry.moverBit));
    if (!seen) {
        return 0.0;
    }
    // The share of the probe's view the object covers, as a share of Omega_ref (capped at a whole change).
    let radius = 0.5 * length(geometry.regionMax - geometry.regionMin);
    let solidAngle = DDGI_PI * radius * radius / (distance * distance);
    let change = min(1.0, solidAngle / DDGI_FAR_SET_REFERENCE_SOLID_ANGLE) * moveShare;
    ddgiMarkProbe(probeIndex, change, ddgiServing(entry, probe, change));
    return change;
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
fn ddgiMarkThread(thread: u32) -> f32 {
    if (thread >= ddgiMarkTable.threadCount) {
        return 0.0;
    }
    let entry = ddgiMarkTable.entries[ddgiMarkEntryOf(thread)];
    let local = thread - entry.firstThread;
    let inBox = vec3u(local % entry.boxSize.x, local / (entry.boxSize.x * entry.boxSize.z), (local / entry.boxSize.x) % entry.boxSize.z);
    let probe = vec3i(entry.boxMin + inBox);
    let counts = ddgiProbeCount();
    let probeIndex = u32(probe.x) + counts.x * (u32(probe.z) + counts.z * u32(probe.y));
    let data = ddgiReadProbeData(ddgiProbeData, probe);
    if (entry.kind == DDGI_MARK_GEOMETRY) {
        return ddgiMarkGeometry(entry, probe, probeIndex, data);
    }
    if (data.state != DDGI_STATE_ACTIVE) {
        return 0.0;
    }

    var change = DdgiProbeChange(0.0, 0.0);
    if (entry.kind == DDGI_MARK_LOCAL_LIGHT) {
        change = ddgiMarkLocalChange(entry, probe, ddgiProbePosition(probe, data.offset), ddgiMarkStoredMean(probeIndex));
    } else {
        // The CPU's r is already against the new light where the light got weaker (mark-table.ts).
        let global = ddgiMarkGlobalChange(entry, probe);
        change = DdgiProbeChange(global, global);
    }
    if (!(change.change > 0.0)) {
        return 0.0;
    }
    ddgiMarkProbe(probeIndex, change.change, ddgiServing(entry, probe, change.againstNew));
    return change.change;
}

@compute @workgroup_size(DDGI_MARK_WORKGROUP_SIZE)
fn ddgiMark(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localThread: u32,
) {
    // Dispatches past 65,535 workgroups use two dimensions.
    let thread = (group.x + group.y * groups.x) * DDGI_MARK_WORKGROUP_SIZE + localThread;
    let change = ddgiMarkThread(thread);
    ddgiCountMarks(change);
}

// The change clock (T2.4c, m2-lights.md §4.5): adds this frame's largest
// change to the volume's clock, in fixed point, wrapping modulo 2^32; at
// least one unit for any change, so a frame with a change always moves it.
// One thread, after the mark.
@compute @workgroup_size(1)
fn ddgiAdvanceChangeClock() {
    let largest = bitcast<f32>(atomicLoad(&ddgiCensus[DDGI_CENSUS_LARGEST_CHANGE]));
    if (largest > 0.0) {
        atomicAdd(&ddgiCensus[DDGI_CENSUS_CHANGE_CLOCK], u32(ceil(largest * DDGI_CHANGE_CLOCK_UNITS)));
    }
}
`;
}
