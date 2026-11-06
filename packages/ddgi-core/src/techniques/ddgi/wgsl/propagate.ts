/**
 * Propagation (T2.4c, docs/design/m2-lights.md §4.5): a fine probe takes its
 * key neighbours' change until it is traced itself.
 *
 * After a change, key probes (every even grid index, about one in 8) are
 * served first. Each key update that takes in a reported change writes the
 * change it made, `Δ = new − old` per irradiance texel, to the key-delta
 * buffer (`ddgi/blend`). Then, after `ddgi/write`, in one pass
 * `ddgi/propagate` of four dispatches:
 *
 * 1. `ddgi/propagate-find`, one thread per probe: the fine probes that are
 *    awaiting a change (the flag `ddgi/mark` sets and the probe's own blend
 *    clears), active, with a key neighbour that passed its change this frame,
 *    go into a compacted list. **Subgroups first** (one atomic per subgroup
 *    reserves the slots) or the **declared fallback** (one per thread). It is
 *    dispatched indirectly, its size written at the start of the key-delta
 *    buffer by `ddgi/blend` when a key passes its change on, so in a frame
 *    where none does (a still scene) it runs no workgroup at all.
 * 2. `ddgi/propagate-args`, one thread: the indirect dispatch size for the
 *    list (two dimensions past the per-dimension limit).
 * 3. `ddgi/propagate`, one workgroup per listed fine probe: adds, per texel,
 *    `Σⱼ wⱼ · sⱼ · Δⱼ` over its key neighbours j that passed a change this
 *    frame, clamped so irradiance stays ≥ 0, into propagation's own scratch,
 *    and records which keys passed it theirs (the fine probe's blend cuts its
 *    history only for what is left). Its mean irradiance follows.
 *    - wⱼ: the fine probe's trilinear weight on the key lattice (equal for
 *      all its keys), times a visibility test from both ends, normalised over
 *      its active, visible keys whether or not they were updated this frame,
 *      so each key's share is fixed and the shares add up to 1. A key that
 *      fails the test counts 0: no light crosses a wall.
 *    - sⱼ: the part of the key's change that came after the fine probe's own
 *      last update, measured on the volume's change clock G (census):
 *      `(G − max(G_f, G_prev)) / (G − G_prev)`, 1 when `G = G_prev`. A key's
 *      Δ covers everything since its own last update; a fine probe traced in
 *      between already has the earlier part, and a change that keeps going
 *      would otherwise be counted twice.
 * 4. `ddgi/propagate-write` (write.ts's `ddgiWriteIrradianceTiles`, the same
 *    tile code as `ddgi/write`): copies the new tiles into the irradiance
 *    atlas, so the change reaches the image this frame.
 *
 * Distances aren't propagated: a light change doesn't move them.
 *
 * Bindings (group 1, one declaration for the module, a layout per kernel):
 * uniforms (0), the probe history as words (1), the census (2, propagate
 * only), the
 * propagation state written by find and args (3), the list (4), the key
 * deltas (5), propagation's scratch (6), the propagation state read-only
 * (7, beside its indirect use), and the irradiance, distance and probe-data
 * atlases (8-10, sampled) and the atlas sampler (11). `ddgi/propagate` binds
 * 6 storage buffers.
 */

import {
    CENSUS_CHANGE_CLOCK,
    CENSUS_PROBES_PROPAGATED,
    CENSUS_PROBES_PROPAGATED_TOTAL,
    KEY_DELTA_FIND_ARGS_WORDS,
    KEY_DELTA_HEADER,
    KEY_DELTA_HEADER_WORDS,
    KEY_NEIGHBOURS,
    keyDeltaStrideWords,
    PROBE_HISTORY_FLAG,
    PROBE_HISTORY_KEYS,
    PROBE_HISTORY_WORD,
    PROBE_HISTORY_WORDS,
    PROPAGATION_STATE_WORD,
    propagationScratchStrideWords,
} from '../layout.ts';
import { DDGI_COMMON_WGSL } from './common.ts';
import { DDGI_STOCHASTIC_HALF_WGSL } from './half.ts';
import type { PhaseBReduction } from './blend.ts';

export const PROPAGATE_FIND_WORKGROUP_SIZE = 256;
export const PROPAGATE_WORKGROUP_SIZE = 64;

/** The module's entry points, in the order they run (the write is write.ts's). */
export const PROPAGATE_ENTRY_POINTS = { find: 'ddgiPropagateFind', args: 'ddgiPropagateArgs', propagate: 'ddgiPropagate' } as const;

/**
 * A key counts for a fine probe when the two ends' Chebyshev tests, multiplied,
 * give at least this: as likely visible as not. A key across a wall reads
 * about 0.01; requiring a real share keeps the normalisation from
 * stretching a sliver of visibility into a full weight.
 */
export const PROPAGATION_VISIBLE_SHARE = 0.5;

/** How the find kernel reserves list slots: subgroups first, or the declared fallback. */
export type PropagateReduction = PhaseBReduction;

export interface PropagateKernelShape {
    readonly irradianceTexels: number;
    readonly reduction: PropagateReduction;
    /** How many fine probes the list holds (propagationListCapacity in layout.ts). */
    readonly listCapacity: number;
    /** The device's `maxComputeWorkgroupsPerDimension`, for the indirect dispatch's size. */
    readonly maxWorkgroupsPerDimension: number;
}

function reserveWgsl(reduction: PropagateReduction): string {
    if (reduction === 'subgroups') {
        return /* wgsl */ `
// Subgroups first: one atomic per subgroup reserves its taking lanes' slots,
// each lane its offset inside the subgroup's share. Called by every thread in
// uniform control flow.
fn ddgiReserveListSlot(take: bool) -> u32 {
    let flag = select(0u, 1u, take);
    let count = subgroupAdd(flag);
    let offset = subgroupExclusiveAdd(flag);
    var base = 0u;
    if (subgroupElect() && count > 0u) {
        base = atomicAdd(&ddgiPropagationState[DDGI_STATE_LISTED], count);
    }
    // subgroupElect's lane is the lowest active one, the lane subgroupBroadcastFirst reads.
    return subgroupBroadcastFirst(base) + offset;
}
`;
    }
    return /* wgsl */ `
// The declared fallback without subgroups: one atomic per taking thread.
fn ddgiReserveListSlot(take: bool) -> u32 {
    if (take) {
        return atomicAdd(&ddgiPropagationState[DDGI_STATE_LISTED], 1u);
    }
    return 0u;
}
`;
}

export function propagateKernelWgsl(shape: PropagateKernelShape): string {
    const { irradianceTexels, reduction, listCapacity, maxWorkgroupsPerDimension } = shape;
    const enable = reduction === 'subgroups' ? 'enable subgroups;\n' : '';
    return /* wgsl */ `${enable}
${DDGI_COMMON_WGSL}
${DDGI_STOCHASTIC_HALF_WGSL}

// ---- DDGI propagation kernels (techniques/ddgi) ----------------------------------------

const DDGI_FIND_WORKGROUP_SIZE: u32 = ${PROPAGATE_FIND_WORKGROUP_SIZE}u;
const DDGI_PROPAGATE_WORKGROUP_SIZE: u32 = ${PROPAGATE_WORKGROUP_SIZE}u;
const DDGI_IRRADIANCE_TEXELS: u32 = ${irradianceTexels}u;
const DDGI_IRRADIANCE_TEXEL_COUNT: u32 = DDGI_IRRADIANCE_TEXELS * DDGI_IRRADIANCE_TEXELS;
const DDGI_KEY_NEIGHBOURS: u32 = ${KEY_NEIGHBOURS}u;
const DDGI_LIST_CAPACITY: u32 = ${listCapacity}u;
const DDGI_MAX_WORKGROUPS_PER_DIMENSION: u32 = ${maxWorkgroupsPerDimension}u;
const DDGI_VISIBLE_SHARE: f32 = ${PROPAGATION_VISIBLE_SHARE};
// The probe history, as words (PROBE_HISTORY_WORD in layout.ts).
const DDGI_HISTORY_WORDS: u32 = ${PROBE_HISTORY_WORDS}u;
const DDGI_HISTORY_SCHEDULE_AND_FLAGS: u32 = ${PROBE_HISTORY_WORD.scheduleAndFlags}u;
const DDGI_HISTORY_MEAN_RED_GREEN: u32 = ${PROBE_HISTORY_WORD.meanRedGreen}u;
const DDGI_HISTORY_MEAN_BLUE_NOISE: u32 = ${PROBE_HISTORY_WORD.meanBlueNoise}u;
const DDGI_HISTORY_KEYS_RECEIVED: u32 = ${PROBE_HISTORY_WORD.keysReceived}u;
const DDGI_HISTORY_CHANGE_CLOCK: u32 = ${PROBE_HISTORY_WORD.changeClock}u;
const DDGI_HISTORY_KEY_DELTA_SLOT: u32 = ${PROBE_HISTORY_WORD.keyDeltaSlot}u;
const DDGI_FLAG_AWAITING_CHANGE: u32 = ${PROBE_HISTORY_FLAG.awaitingChange}u;
const DDGI_FLAG_PASSED_CHANGE: u32 = ${PROBE_HISTORY_FLAG.passedChange}u;
const DDGI_KEYS_BITS_MASK: u32 = ${PROBE_HISTORY_KEYS.bitsMask}u;
const DDGI_KEYS_WEIGHT_SHIFT: u32 = ${PROBE_HISTORY_KEYS.weightShift}u;
// The scheduled frame sits above the flags' 8 bits (wgsl/schedule.ts); 24 bits.
const DDGI_SCHEDULED_SHIFT: u32 = 8u;
// The census and the propagation state (layout.ts).
const DDGI_CENSUS_CHANGE_CLOCK: u32 = ${CENSUS_CHANGE_CLOCK}u;
const DDGI_CENSUS_PROBES_PROPAGATED: u32 = ${CENSUS_PROBES_PROPAGATED}u;
const DDGI_CENSUS_PROBES_PROPAGATED_TOTAL: u32 = ${CENSUS_PROBES_PROPAGATED_TOTAL}u;
const DDGI_STATE_GROUPS_X: u32 = ${PROPAGATION_STATE_WORD.groupsX}u;
const DDGI_STATE_GROUPS_Y: u32 = ${PROPAGATION_STATE_WORD.groupsY}u;
const DDGI_STATE_GROUPS_Z: u32 = ${PROPAGATION_STATE_WORD.groupsZ}u;
const DDGI_STATE_LISTED: u32 = ${PROPAGATION_STATE_WORD.listed}u;
// The key-delta buffer (KEY_DELTA_HEADER in layout.ts).
const DDGI_KEY_DELTA_STRIDE: u32 = ${keyDeltaStrideWords(irradianceTexels)}u;
const DDGI_KEY_DELTA_HEADER_WORDS: u32 = ${KEY_DELTA_HEADER_WORDS}u;
const DDGI_KEY_DELTA_PROBE_INDEX: u32 = ${KEY_DELTA_HEADER.probeIndex}u;
const DDGI_KEY_DELTA_CLOCK_BEFORE: u32 = ${KEY_DELTA_HEADER.clockBefore}u;
const DDGI_PROPAGATION_STRIDE: u32 = ${propagationScratchStrideWords(irradianceTexels)}u;

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read_write> ddgiHistoryWords: array<u32>;
@group(1) @binding(2) var<storage, read_write> ddgiCensus: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> ddgiPropagationState: array<atomic<u32>>;
@group(1) @binding(4) var<storage, read_write> ddgiPropagationList: array<u32>;
// The key deltas (KEY_DELTA_FIND_ARGS_WORDS of find's dispatch size, then the entries).
struct DdgiKeyDeltas {
    findGroups: array<u32, ${KEY_DELTA_FIND_ARGS_WORDS}>,
    words: array<u32>,
}
@group(1) @binding(5) var<storage, read> ddgiKeyDeltas: DdgiKeyDeltas;
@group(1) @binding(6) var<storage, read_write> ddgiPropagationScratch: array<u32>;
// The same state, read-only: ddgi/propagate is dispatched indirectly from it.
@group(1) @binding(7) var<storage, read> ddgiPropagationCount: array<u32>;
@group(1) @binding(8) var ddgiIrradianceAtlas: texture_2d_array<f32>;
@group(1) @binding(9) var ddgiDistanceAtlas: texture_2d_array<f32>;
@group(1) @binding(10) var ddgiProbeData: texture_2d_array<f32>;
@group(1) @binding(11) var ddgiAtlasSampler: sampler;

${reserveWgsl(reduction)}

fn ddgiProbeIndexOf(probe: vec3i) -> u32 {
    let counts = ddgiProbeCount();
    return u32(probe.x) + counts.x * (u32(probe.z) + counts.z * u32(probe.y));
}

// The frame as a probe's scheduled stamp stores it (24 bits).
fn ddgiFrameStamp() -> u32 {
    return ddgi.frame & (0xffffffffu >> DDGI_SCHEDULED_SHIFT);
}

// Whether a key probe passed its change on this frame: its update this frame
// (its scheduled stamp is this frame's) set the passedChange flag.
fn ddgiKeyPassedThisFrame(keyIndex: u32) -> bool {
    let word = ddgiHistoryWords[keyIndex * DDGI_HISTORY_WORDS + DDGI_HISTORY_SCHEDULE_AND_FLAGS];
    return (word >> DDGI_SCHEDULED_SHIFT) == ddgiFrameStamp() && (word & DDGI_FLAG_PASSED_CHANGE) != 0u;
}

// ---- 1. find ----
@compute @workgroup_size(DDGI_FIND_WORKGROUP_SIZE)
fn ${PROPAGATE_ENTRY_POINTS.find}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let counts = ddgiProbeCount();
    let probeIndex = (group.x + group.y * groups.x) * DDGI_FIND_WORKGROUP_SIZE + thread;
    var take = false;
    if (probeIndex < counts.x * counts.y * counts.z) {
        let probe = ddgiProbeCoordinates(probeIndex, counts);
        let flags = ddgiHistoryWords[probeIndex * DDGI_HISTORY_WORDS + DDGI_HISTORY_SCHEDULE_AND_FLAGS];
        if ((flags & DDGI_FLAG_AWAITING_CHANGE) != 0u && !ddgiIsKeyProbe(probe) && ddgiReadProbeData(ddgiProbeData, probe).state == DDGI_STATE_ACTIVE) {
            for (var neighbour = 0u; neighbour < DDGI_KEY_NEIGHBOURS; neighbour++) {
                let key = ddgiKeyNeighbourOffset(neighbour, probe);
                if (key.exists && ddgiKeyPassedThisFrame(ddgiProbeIndexOf(probe + key.offset))) {
                    take = true;
                    break;
                }
            }
        }
    }
    // Every thread, in uniform control flow (the subgroup form needs it).
    let slot = ddgiReserveListSlot(take);
    if (take && slot < DDGI_LIST_CAPACITY) {
        ddgiPropagationList[slot] = probeIndex;
    }
}

// ---- 2. args ----
// One thread: the list's length (no more than it holds) and the indirect
// dispatch size, one workgroup per entry, in two dimensions past the limit.
@compute @workgroup_size(1)
fn ${PROPAGATE_ENTRY_POINTS.args}() {
    let listed = min(atomicLoad(&ddgiPropagationState[DDGI_STATE_LISTED]), DDGI_LIST_CAPACITY);
    let groupsX = min(listed, DDGI_MAX_WORKGROUPS_PER_DIMENSION);
    let groupsY = select(0u, (listed + groupsX - 1u) / max(groupsX, 1u), groupsX > 0u);
    atomicStore(&ddgiPropagationState[DDGI_STATE_LISTED], listed);
    atomicStore(&ddgiPropagationState[DDGI_STATE_GROUPS_X], groupsX);
    atomicStore(&ddgiPropagationState[DDGI_STATE_GROUPS_Y], groupsY);
    atomicStore(&ddgiPropagationState[DDGI_STATE_GROUPS_Z], 1u);
}

// ---- 3. propagate ----

// How visible the point 'seen' is from the probe at 'origin' (grid index
// 'probe'), by one Chebyshev test on the probe's distance moments toward it,
// as sampling does: 1 when the mean distance in that direction reaches the
// point. The moments stop at distanceClamp, so the test vouches only for the
// probe's own end of the segment, up to that clamp.
fn ddgiEndVisibility(probe: vec3i, origin: vec3f, seen: vec3f) -> f32 {
    let volume = ddgiUniformVolume();
    let segment = seen - origin;
    let distance = length(segment);
    if (distance <= 0.0) {
        return 1.0;
    }
    let scale = ddgiAtlasScale(volume.distanceTexels, volume.counts);
    let uv = ddgiTileUv(ddgiTileTexel(segment / distance, volume.distanceTexels), probe, scale);
    let moments = textureSampleLevel(ddgiDistanceAtlas, ddgiAtlasSampler, uv, probe.y, 0.0).xy;
    let tested = min(distance, ddgi.distanceClamp);
    if (tested <= moments.x) {
        return 1.0;
    }
    let variance = abs(moments.y - moments.x * moments.x);
    let excess = tested - moments.x;
    return variance / (variance + excess * excess);
}

// The visibility between a fine probe and a key, from both ends (each end's
// moments vouch for its own part of the segment, up to its clamp, and the two
// clamps together must cover it). Within one grid a fine probe is at most
// about 2 spacings from its keys and each clamp is about 2.6, so either end
// could vouch for the whole of it; nested volumes (M3) can't, which is why
// both ends are asked.
fn ddgiPairVisibility(fine: vec3i, finePosition: vec3f, key: vec3i, keyPosition: vec3f) -> f32 {
    if (2.0 * ddgi.distanceClamp < distance(finePosition, keyPosition)) {
        return 0.0;
    }
    return ddgiEndVisibility(fine, finePosition, keyPosition) * ddgiEndVisibility(key, keyPosition, finePosition);
}

// The probe's keys, one thread each, then normalised by thread 0, for every thread.
var<workgroup> ddgiKeyWeight: array<f32, DDGI_KEY_NEIGHBOURS>;     // its visibility, then w_j normalised; 0 for a key that doesn't count
var<workgroup> ddgiKeyChange: array<f32, DDGI_KEY_NEIGHBOURS>;     // s_j for a key that passed its change this frame, else -1
var<workgroup> ddgiKeyShare: array<f32, DDGI_KEY_NEIGHBOURS>;      // w_j s_j for a key that passed its change this frame, else 0
var<workgroup> ddgiKeySlot: array<u32, DDGI_KEY_NEIGHBOURS>;       // its key-delta entry
var<workgroup> ddgiNewTexel: array<vec3f, DDGI_IRRADIANCE_TEXEL_COUNT>;
var<workgroup> ddgiReceived: u32;                                   // 1 when any key passed it a change this frame

// The share s_j of a key's change that came after the fine probe's own last
// update, on the change clock (differences modulo 2^32, as the clock wraps):
// (G - max(G_f, G_prev)) / (G - G_prev), and 1 when G = G_prev (a key's later
// owed updates after a step add no clock; the awaiting flag already keeps out
// fine probes traced since their mark).
fn ddgiChangeShare(clock: u32, fineStamp: u32, keyStampBefore: u32) -> f32 {
    let sinceKey = clock - keyStampBefore;
    if (sinceKey == 0u) {
        return 1.0;
    }
    let sinceFine = clock - fineStamp;
    return saturate(f32(min(sinceFine, sinceKey)) / f32(sinceKey));
}

// One thread per key neighbour j: whether the key counts (active, and
// visible from both ends), its visibility (the trilinear weight on the key
// lattice is the same for every key neighbour, so it cancels in the
// normalisation), and, when it passed its change this frame, its key-delta
// entry and s_j (the file comment). One thread each because each test reads
// two distance texels: done in turn by one thread they set the pass's time.
fn ddgiWeighKey(fine: vec3i, fineIndex: u32, neighbour: u32) {
    ddgiKeyWeight[neighbour] = 0.0;
    ddgiKeyChange[neighbour] = -1.0;
    let key = ddgiKeyNeighbourOffset(neighbour, fine);
    if (!key.exists) {
        return;
    }
    let keyProbe = fine + key.offset;
    let keyData = ddgiReadProbeData(ddgiProbeData, keyProbe);
    if (keyData.state != DDGI_STATE_ACTIVE) {
        return;
    }
    let finePosition = ddgiProbePosition(fine, ddgiReadProbeData(ddgiProbeData, fine).offset);
    let visibility = ddgiPairVisibility(fine, finePosition, keyProbe, ddgiProbePosition(keyProbe, keyData.offset));
    if (visibility < DDGI_VISIBLE_SHARE) {
        return;
    }
    ddgiKeyWeight[neighbour] = visibility;
    let keyIndex = ddgiProbeIndexOf(keyProbe);
    if (!ddgiKeyPassedThisFrame(keyIndex)) {
        return;
    }
    let slot = ddgiHistoryWords[keyIndex * DDGI_HISTORY_WORDS + DDGI_HISTORY_KEY_DELTA_SLOT];
    let entry = slot * DDGI_KEY_DELTA_STRIDE;
    // The entry is the key's own (a stale slot can't pass for it).
    if (ddgiKeyDeltas.words[entry + DDGI_KEY_DELTA_PROBE_INDEX] != keyIndex) {
        return;
    }
    let clock = atomicLoad(&ddgiCensus[DDGI_CENSUS_CHANGE_CLOCK]);
    let fineStamp = ddgiHistoryWords[fineIndex * DDGI_HISTORY_WORDS + DDGI_HISTORY_CHANGE_CLOCK];
    ddgiKeySlot[neighbour] = slot;
    ddgiKeyChange[neighbour] = ddgiChangeShare(clock, fineStamp, ddgiKeyDeltas.words[entry + DDGI_KEY_DELTA_CLOCK_BEFORE]);
}

// Thread 0, after ddgiWeighKey: the weights normalised over the keys that
// count, the shares w_j s_j, and what the probe has received.
fn ddgiNormaliseKeys(fineIndex: u32) {
    var weightSum = 0.0;
    for (var neighbour = 0u; neighbour < DDGI_KEY_NEIGHBOURS; neighbour++) {
        weightSum += ddgiKeyWeight[neighbour];
    }
    let base = fineIndex * DDGI_HISTORY_WORDS;
    var receivedBits = ddgiHistoryWords[base + DDGI_HISTORY_KEYS_RECEIVED] & DDGI_KEYS_BITS_MASK;
    var passed = false;
    for (var neighbour = 0u; neighbour < DDGI_KEY_NEIGHBOURS; neighbour++) {
        ddgiKeyShare[neighbour] = 0.0;
        if (!(ddgiKeyWeight[neighbour] > 0.0)) {
            continue;
        }
        let weight = ddgiKeyWeight[neighbour] / weightSum;
        ddgiKeyWeight[neighbour] = weight;
        if (ddgiKeyChange[neighbour] >= 0.0) {
            ddgiKeyShare[neighbour] = weight * ddgiKeyChange[neighbour];
            receivedBits = receivedBits | (1u << neighbour);
            passed = true;
        }
    }
    ddgiReceived = select(0u, 1u, passed);
    // What the probe has received since its own last update: each key's
    // weight once, however many of its updates passed it a change. Its blend
    // cuts its history for the rest (ddgiResidualChange in wgsl/blend.ts).
    var receivedWeight = 0.0;
    for (var neighbour = 0u; neighbour < DDGI_KEY_NEIGHBOURS; neighbour++) {
        if ((receivedBits & (1u << neighbour)) != 0u) {
            receivedWeight += ddgiKeyWeight[neighbour];
        }
    }
    ddgiHistoryWords[base + DDGI_HISTORY_KEYS_RECEIVED] = receivedBits | (pack2x16float(vec2f(0.0, saturate(receivedWeight))) & (0xffffu << DDGI_KEYS_WEIGHT_SHIFT));
}

// The atlas texel of interior texel 'texelIndex' of a probe's irradiance tile (the layer is probe.y).
fn ddgiIrradianceAtlasTexel(probe: vec3i, texelIndex: u32) -> vec2i {
    let border = i32(DDGI_TILE_BORDER);
    let texel = vec2i(vec2u(texelIndex % DDGI_IRRADIANCE_TEXELS, texelIndex / DDGI_IRRADIANCE_TEXELS));
    return probe.xz * (i32(DDGI_IRRADIANCE_TEXELS) + 2 * border) + border + texel;
}

fn ddgiKeyDeltaTexel(slot: u32, texelIndex: u32) -> vec3f {
    let word = slot * DDGI_KEY_DELTA_STRIDE + DDGI_KEY_DELTA_HEADER_WORDS + 2u * texelIndex;
    return vec3f(unpack2x16float(ddgiKeyDeltas.words[word]), unpack2x16float(ddgiKeyDeltas.words[word + 1u]).x);
}

@compute @workgroup_size(DDGI_PROPAGATE_WORKGROUP_SIZE)
fn ${PROPAGATE_ENTRY_POINTS.propagate}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let entry = group.x + group.y * groups.x;
    if (entry >= ddgiPropagationCount[DDGI_STATE_LISTED]) {
        return;
    }
    let fineIndex = ddgiPropagationList[entry];
    let fine = ddgiProbeCoordinates(fineIndex, ddgiProbeCount());
    if (thread < DDGI_KEY_NEIGHBOURS) {
        ddgiWeighKey(fine, fineIndex, thread);
    }
    workgroupBarrier();
    if (thread == 0u) {
        ddgiNormaliseKeys(fineIndex);
    }
    workgroupBarrier();

    // Every texel: old + the keys' shares of their changes, irradiance kept
    // at 0 or more (where a large darkening meets noise this biases a little
    // bright; the probe's own update corrects it). Alpha (the opt-in
    // big-change rule's verdict) is kept as it was.
    for (var texelIndex = thread; texelIndex < DDGI_IRRADIANCE_TEXEL_COUNT; texelIndex += DDGI_PROPAGATE_WORKGROUP_SIZE) {
        var change = vec3f(0.0);
        for (var neighbour = 0u; neighbour < DDGI_KEY_NEIGHBOURS; neighbour++) {
            let share = ddgiKeyShare[neighbour];
            if (share > 0.0) {
                change += share * ddgiKeyDeltaTexel(ddgiKeySlot[neighbour], texelIndex);
            }
        }
        let old = textureLoad(ddgiIrradianceAtlas, ddgiIrradianceAtlasTexel(fine, texelIndex), fine.y, 0);
        let value = max(old.rgb + change, vec3f(0.0));
        ddgiNewTexel[texelIndex] = value;
        let seed = (fineIndex * DDGI_IRRADIANCE_TEXEL_COUNT + texelIndex) ^ 0x2c1b3c6du;
        let word = entry * DDGI_PROPAGATION_STRIDE + 2u * texelIndex;
        ddgiPropagationScratch[word] = pack2x16float(vec2f(ddgiStochasticHalf(value.r, ddgiRandom(seed, 1u, ddgi.frame)), ddgiStochasticHalf(value.g, ddgiRandom(seed, 2u, ddgi.frame))));
        ddgiPropagationScratch[word + 1u] = pack2x16float(vec2f(ddgiStochasticHalf(value.b, ddgiRandom(seed, 3u, ddgi.frame)), old.a));
    }
    workgroupBarrier();

    // Thread 0: the probe's mean irradiance follows (ddgi/mark divides a
    // light's change by it); its noise word stays.
    if (thread == 0u && ddgiReceived != 0u) {
        var sum = vec3f(0.0);
        for (var texelIndex = 0u; texelIndex < DDGI_IRRADIANCE_TEXEL_COUNT; texelIndex++) {
            sum += ddgiNewTexel[texelIndex];
        }
        let mean = sum / f32(DDGI_IRRADIANCE_TEXEL_COUNT);
        let base = fineIndex * DDGI_HISTORY_WORDS;
        let noise = unpack2x16float(ddgiHistoryWords[base + DDGI_HISTORY_MEAN_BLUE_NOISE]).y;
        ddgiHistoryWords[base + DDGI_HISTORY_MEAN_RED_GREEN] = pack2x16float(mean.rg);
        ddgiHistoryWords[base + DDGI_HISTORY_MEAN_BLUE_NOISE] = pack2x16float(vec2f(mean.b, noise));
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_PROPAGATED], 1u);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_PROPAGATED_TOTAL], 1u);
    }
}
`;
}
