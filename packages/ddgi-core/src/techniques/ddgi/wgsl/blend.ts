/**
 * The `ddgi/blend` kernel (docs/design/ddgi.md): one workgroup per probe in
 * this frame's update list. It loads the probe's rays from the ray buffer
 * into workgroup memory, then
 *
 * - **phase B, classification and relocation**, over the 32 fixed rays
 *   (Majercik et al. 2021, section 5): a workgroup reduction finds how many
 *   fixed rays hit back faces, the closest back face, the closest and
 *   farthest front faces, and the front faces within the probe's reach, and
 *   thread 0 decides the probe's new offset and state;
 * - **phase C, blend**, for probes that traced all their rays and stay
 *   active: one thread per atlas texel loops over the blended rays and blends
 *   with the old value (hysteresis), writing the scratch buffer.
 *
 * The reduction is **subgroups first**: each subgroup reduces with
 * `subgroupAdd` / `subgroupMin` / `subgroupMax`, and one lane per subgroup
 * combines with workgroup atomics. The **declared fallback** (no `subgroups`
 * feature) has every thread do the workgroup atomics itself. The keys are
 * integers, so both give exactly the same result. Chosen when the kernel is
 * built; nothing assumes a subgroup size.
 *
 * Hysteresis knows the time between a probe's updates (T1.12): each probe
 * keeps how many updates it remembers (n), when it was last updated (t) and
 * how many blended updates it has had (k, for the warm-up ramp) in the
 * history buffer. Changes the scene reports cut n before that rule runs
 * (M2, docs/design/m2-lights.md §4.1): `ddgi/mark` adds each probe's change
 * to its record, and this kernel cuts the history for it, resets the change
 * and writes the probe's mean irradiance and noise back. Within a frame
 * `ddgi/mark` runs first, as its own dispatch, so the two never race.
 *
 * The GPU scheduler's promises (T2.4b, m2-lights.md §4.3): a probe owed
 * updates after a change has its history cut for the updates it was
 * promised (K, or fewer for far probes after a global change), and each
 * update takes one off; its urgency stays until the last.
 *
 * The feedback rule (T2.4b, m2-lights.md §4.6): a quiet update (no change
 * reported, none being taken in) whose estimate lands far from the stored
 * value against the probe's noise asks for one more update for itself and
 * its 6 neighbours, in a buffer of its own (`atomicMax` on the urgency's
 * bits), which `ddgi/schedule-count` takes in. It never touches hysteresis.
 *
 * Bindings (group 1; no platform module): uniforms, the update list, the ray
 * buffer, the scratch buffer, the census buffer, the probe history and the
 * feedback requests (6 storage buffers), and the irradiance, distance and
 * probe-data atlases (sampled, for the old values).
 */

import {
    CENSUS_FEEDBACK_FIRED,
    CENSUS_FEEDBACK_FIRED_TOTAL,
    CENSUS_PROBES_BLENDED_TOTAL,
    CENSUS_RAYS_TRACED,
    PROBE_HISTORY_FLAG,
    PROBE_HISTORY_FLAGS_MASK,
    PROBE_HISTORY_UPDATES,
    type ScratchLayout,
} from '../layout.ts';
import { CLOCK_WRAP_SECONDS, FIXED_RAYS } from '../settings.ts';

/**
 * The feedback rule's thresholds (m2-lights.md §4.6): an update whose
 * estimate lands more than this many times the probe's noise from its stored
 * value, and more than {@link FEEDBACK_MIN_SHARE} of it...
 */
export const FEEDBACK_NOISE_MULTIPLE = 4;
/** ...asks for one more update (2% of the stored mean: below that a change can't matter for the 5% target). */
export const FEEDBACK_MIN_SHARE = 0.02;
/** Its 6 face neighbours ask at this share of its urgency: a change nobody reported spreads to the probes around it. */
export const FEEDBACK_NEIGHBOUR_SHARE = 0.5;

/**
 * Bits of a reduction key that hold the fixed ray's index (the rest is the
 * hit distance's float bits). Derived from the fixed-ray count, and kept to
 * at most 8 of the 23 mantissa bits so the distance stays precise.
 */
export const KEY_INDEX_BITS = Math.ceil(Math.log2(FIXED_RAYS));
if (KEY_INDEX_BITS > 8) throw new Error(`@pmndrs/ddgi: ${FIXED_RAYS} fixed rays need ${KEY_INDEX_BITS} key bits; phase B's keys allow 8`);
const KEY_INDEX_MASK = (1 << KEY_INDEX_BITS) - 1;
import { DDGI_COMMON_WGSL } from './common.ts';
import { DDGI_STOCHASTIC_HALF_WGSL } from './half.ts';

/** The default `maxComputeInvocationsPerWorkgroup`. */
export const BLEND_WORKGROUP_SIZE = 256;

/** How phase B combines the fixed rays: subgroups first, or the declared fallback. */
export type PhaseBReduction = 'subgroups' | 'workgroup';

export interface BlendKernelShape {
    readonly irradianceTexels: number;
    readonly distanceTexels: number;
    readonly scratch: ScratchLayout;
    readonly reduction: PhaseBReduction;
}

/** Per-thread reduction into the workgroup atomics, in the two declared forms. */
function reductionWgsl(reduction: PhaseBReduction): string {
    if (reduction === 'subgroups') {
        return /* wgsl */ `
// Subgroups first: each subgroup reduces in hardware, then one lane per
// subgroup meets the others in the workgroup atomics. Called by every thread
// in uniform control flow (threads without a fixed ray pass neutral values).
fn ddgiReduceFixedRay(ray: DdgiFixedRayResult) {
    let backFaces = subgroupAdd(ray.backFace);
    let nearestBack = subgroupMin(ray.nearestBack);
    let closestBack = subgroupMin(ray.closestBack);
    let closestFront = subgroupMin(ray.closestFront);
    let farthestFront = subgroupMax(ray.farthestFront);
    let frontInReach = subgroupAdd(ray.frontInReach);
    if (subgroupElect()) {
        atomicAdd(&ddgiBackFaceCount, backFaces);
        atomicMin(&ddgiNearestBackFace, nearestBack);
        atomicMin(&ddgiClosestBackFace, closestBack);
        atomicMin(&ddgiClosestFrontFace, closestFront);
        atomicMax(&ddgiFarthestFrontFace, farthestFront);
        atomicAdd(&ddgiFrontFacesInReach, frontInReach);
    }
}
`;
    }
    return /* wgsl */ `
// The declared fallback without subgroups: every thread combines its own
// fixed ray in the workgroup atomics.
fn ddgiReduceFixedRay(ray: DdgiFixedRayResult) {
    atomicAdd(&ddgiBackFaceCount, ray.backFace);
    atomicMin(&ddgiNearestBackFace, ray.nearestBack);
    atomicMin(&ddgiClosestBackFace, ray.closestBack);
    atomicMin(&ddgiClosestFrontFace, ray.closestFront);
    atomicMax(&ddgiFarthestFrontFace, ray.farthestFront);
    atomicAdd(&ddgiFrontFacesInReach, ray.frontInReach);
}
`;
}

export function blendKernelWgsl(shape: BlendKernelShape): string {
    const { irradianceTexels, distanceTexels, scratch, reduction } = shape;
    // An enable directive must come before everything else in the module.
    const enable = reduction === 'subgroups' ? 'enable subgroups;\n' : '';
    return /* wgsl */ `${enable}
${DDGI_COMMON_WGSL}
${DDGI_STOCHASTIC_HALF_WGSL}

// ---- DDGI blend kernel (techniques/ddgi) -----------------------------------------------

const DDGI_BLEND_WORKGROUP_SIZE: u32 = ${BLEND_WORKGROUP_SIZE}u;
const DDGI_IRRADIANCE_TEXELS: u32 = ${irradianceTexels}u;
const DDGI_DISTANCE_TEXELS: u32 = ${distanceTexels}u;
const DDGI_IRRADIANCE_TEXEL_COUNT: u32 = DDGI_IRRADIANCE_TEXELS * DDGI_IRRADIANCE_TEXELS;
const DDGI_DISTANCE_TEXEL_COUNT: u32 = DDGI_DISTANCE_TEXELS * DDGI_DISTANCE_TEXELS;
const DDGI_SCRATCH_STRIDE: u32 = ${scratch.strideWords}u;
const DDGI_SCRATCH_IRRADIANCE: u32 = ${scratch.irradianceOffset}u;
const DDGI_SCRATCH_DISTANCE: u32 = ${scratch.distanceOffset}u;
const DDGI_SCRATCH_PROBE_DATA: u32 = ${scratch.probeDataOffset}u;
const DDGI_CENSUS_RAYS_TRACED: u32 = ${CENSUS_RAYS_TRACED}u;
// Must match the trace kernel: a back-face hit was stored at this share of its distance.
const DDGI_BACK_FACE_DISTANCE_SHARE: f32 = 0.2;
// When the probe's mean irradiance changed by more than the change threshold,
// its hysteresis is multiplied by this, so a light switch shows up quickly.
const DDGI_BIG_CHANGE_HYSTERESIS_SCALE: f32 = 0.75;
// The technique's clock wraps to 0 every this many seconds (CLOCK_WRAP_SECONDS in settings.ts).
const DDGI_CLOCK_WRAP_SECONDS: f32 = ${CLOCK_WRAP_SECONDS}.0;
// The probe history's packed fields (PROBE_HISTORY_UPDATES, PROBE_HISTORY_FLAG in layout.ts).
const DDGI_HISTORY_COUNT_MASK: u32 = ${PROBE_HISTORY_UPDATES.countMask}u;
const DDGI_HISTORY_FLAGS_MASK: u32 = ${PROBE_HISTORY_FLAGS_MASK}u;
const DDGI_FLAG_MARKED: u32 = ${PROBE_HISTORY_FLAG.marked}u;
const DDGI_FLAG_MARKED_BEFORE_LAST_UPDATE: u32 = ${PROBE_HISTORY_FLAG.markedBeforeLastUpdate}u;
const DDGI_FLAG_AWAITING_CHANGE: u32 = ${PROBE_HISTORY_FLAG.awaitingChange}u;
const DDGI_FLAG_TAKING_CHANGE: u32 = ${PROBE_HISTORY_FLAG.takingChange}u;
const DDGI_OWED_SHIFT: u32 = ${PROBE_HISTORY_UPDATES.owedShift}u;
const DDGI_OWED_MASK: u32 = ${PROBE_HISTORY_UPDATES.owedMask}u;
// The feedback rule (FEEDBACK_* above).
const DDGI_FEEDBACK_NOISE_MULTIPLE: f32 = ${FEEDBACK_NOISE_MULTIPLE}.0;
const DDGI_FEEDBACK_MIN_SHARE: f32 = ${FEEDBACK_MIN_SHARE};
const DDGI_FEEDBACK_NEIGHBOUR_SHARE: f32 = ${FEEDBACK_NEIGHBOUR_SHARE};
const DDGI_CENSUS_FEEDBACK_FIRED: u32 = ${CENSUS_FEEDBACK_FIRED}u;
const DDGI_CENSUS_FEEDBACK_FIRED_TOTAL: u32 = ${CENSUS_FEEDBACK_FIRED_TOTAL}u;
const DDGI_CENSUS_PROBES_BLENDED_TOTAL: u32 = ${CENSUS_PROBES_BLENDED_TOTAL}u;
// "No cut": larger than any history count.
const DDGI_NO_HISTORY_LIMIT: f32 = 3.0e38;
// The noise word is a running mean over about this many quiet updates (an
// exponential average with weight 1 / this). Long enough to average a few
// outliers away, short enough to follow a probe whose light changed.
const DDGI_NOISE_UPDATES: f32 = 64.0;
// What kind of update this is, for the noise word (ddgiUpdateKind).
const DDGI_UPDATE_FIRST: u32 = 0u;   // state 3: the history starts again, noise 0
const DDGI_UPDATE_QUIET: u32 = 1u;   // no change reported: feeds the noise, and may trip the feedback rule
const DDGI_UPDATE_MARKED: u32 = 2u;  // a reported change, or still taking one in: the noise is kept as it was

// Phase B rules (Majercik et al. 2021, section 5).
// More than this share of fixed rays hitting back faces means the probe is inside something.
const DDGI_INSIDE_BACK_FACE_SHARE: f32 = 0.25;
// Offsets stay within this share of the spacing on every axis, so a probe
// never leaves its own cell and the grid lookup stays valid.
const DDGI_MAX_OFFSET: f32 = 0.45;
// A probe moved out of geometry lands this share of minFrontFaceDistance past the surface.
const DDGI_PAST_SURFACE_SHARE: f32 = 0.5;
// Moving away from a close face along the most open direction is allowed
// unless that heads back towards the face (more than 60 degrees towards it,
// the paper's rule). A move that runs nearly along the face keeps the probe
// close to it, so it repeats each update and the probe slides along the face
// until its offset limit stops it (about 75 updates at most, T1.5b review).
// Requiring moves to head clearly away stopped the sliding, but raised the
// two-room leak from 0.80% to 0.97% (thin wall 0.59% to 0.70%, closed door
// 0.06% to 0.13%); moving straight away from the face raised it to 1.49%.
const DDGI_MAX_COS_TOWARDS_CLOSEST: f32 = 0.5;
// Moves shorter than this share of minFrontFaceDistance are dropped. A dead
// band, so the "too close" and "ease back" rules can't trade tiny moves for
// ever, and half-float rounding of the offset can't keep a probe jittering
// (123 Cornell probes were still creeping by fractions of a millimetre after
// 60 frames without it, T1.5b review).
const DDGI_MIN_MOVE_SHARE: f32 = 0.25;
// A probe takes part in interpolating surfaces up to one spacing away on each
// axis (its 8 cells). Its 32 fixed rays are about 40 degrees apart, so they
// can miss a sliver of surface poking into that box near its corners; such a
// surface nearly always reaches on past the box, where the rays do find it.
// So a front face within this many spacings counts as "in reach" (T1.5b: at 1,
// the Cornell box switched off 25 probes that a surface needed).
const DDGI_REACH_SPACINGS: f32 = 1.5;
// Share of its offset a probe in open space moves back towards its grid point per update.
const DDGI_EASE_BACK_SHARE: f32 = 0.1;
// Distances and ray indices share one 32-bit key: the distance's float bits
// (non-negative floats order like their bits) with the low 5 bits replaced by
// the fixed ray's index, so min and max return both. Costs ${KEY_INDEX_BITS} of 23 mantissa bits.
const DDGI_KEY_INDEX_MASK: u32 = ${KEY_INDEX_MASK}u;
const DDGI_KEY_NONE_MIN: u32 = 0xffffffffu;
const DDGI_KEY_NONE_MAX: u32 = 0u;
// Stands in for "no front face hit at all" when distances are compared.
const DDGI_NO_DISTANCE: f32 = 3.0e38;

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read> ddgiUpdateList: array<u32>;
@group(1) @binding(2) var<storage, read> ddgiRays: array<vec2u>;
@group(1) @binding(3) var<storage, read_write> ddgiScratch: array<u32>;
@group(1) @binding(4) var<storage, read_write> ddgiCensus: array<atomic<u32>>;
@group(1) @binding(5) var ddgiIrradianceAtlas: texture_2d_array<f32>;
@group(1) @binding(6) var ddgiDistanceAtlas: texture_2d_array<f32>;
@group(1) @binding(7) var ddgiProbeData: texture_2d_array<f32>;
// Per probe: its history for the hysteresis rule (DdgiProbeHistory). Zero until first written.
@group(1) @binding(8) var<storage, read_write> ddgiProbeHistory: array<DdgiProbeHistory>;
// Per probe: an update asked for by the feedback rule, as its urgency's bits (0: none). Read and cleared by ddgi/schedule-count.
@group(1) @binding(9) var<storage, read_write> ddgiFeedbackRequests: array<atomic<u32>>;

// Per ray: its direction (xyz) and signed hit distance (w, negative = back
// face). Decoded once here: the blend reads every ray once per texel.
var<workgroup> ddgiRayDirectionDistance: array<vec4f, DDGI_MAX_RAYS>;
// Per ray: rgb radiance as half floats (rg, then b and the distance), as the trace wrote it.
var<workgroup> ddgiRayRadiance: array<vec2u, DDGI_MAX_RAYS>;
// Phase B's reductions over the fixed rays.
var<workgroup> ddgiBackFaceCount: atomic<u32>;
var<workgroup> ddgiNearestBackFace: atomic<u32>;
var<workgroup> ddgiClosestBackFace: atomic<u32>;
var<workgroup> ddgiClosestFrontFace: atomic<u32>;
var<workgroup> ddgiFarthestFrontFace: atomic<u32>;
var<workgroup> ddgiFrontFacesInReach: atomic<u32>;
// Thread 0's verdicts, read by everyone with workgroupUniformLoad.
var<workgroup> ddgiBlendThisUpdate: u32;
var<workgroup> ddgiUpdateHysteresis: f32;
var<workgroup> ddgiFastHysteresis: u32;
var<workgroup> ddgiChangedNow: u32;
// The probe's history after this update, decided in phase B and written with
// its new mean and noise once phase C has them, and what kind of update it is
// (DDGI_UPDATE_*).
var<workgroup> ddgiNextHistory: DdgiProbeHistory;
var<workgroup> ddgiUpdateKind: u32;
// Phase C: each irradiance texel's new estimate and old value, for the
// big-change test on the probe's mean.
var<workgroup> ddgiTexelEstimate: array<vec4f, DDGI_IRRADIANCE_TEXEL_COUNT>;
var<workgroup> ddgiTexelOld: array<vec4f, DDGI_IRRADIANCE_TEXEL_COUNT>;

// One thread's fixed ray, as the reduction combines it.
struct DdgiFixedRayResult {
    backFace: u32,       // 1 for a back-face hit
    nearestBack: u32,    // key of any back-face hit, or DDGI_KEY_NONE_MIN
    closestBack: u32,    // key of a back-face hit the probe may leave through, or DDGI_KEY_NONE_MIN
    closestFront: u32,   // key of a front-face hit (or miss), or DDGI_KEY_NONE_MIN
    farthestFront: u32,  // the same key, or DDGI_KEY_NONE_MAX
    frontInReach: u32,   // 1 for a front-face hit inside the probe's reach
}

${reductionWgsl(reduction)}

fn ddgiDistanceKey(distance: f32, rayIndex: u32) -> u32 {
    return (bitcast<u32>(distance) & ~DDGI_KEY_INDEX_MASK) | rayIndex;
}

fn ddgiKeyDistance(key: u32) -> f32 {
    return bitcast<f32>(key & ~DDGI_KEY_INDEX_MASK);
}

fn ddgiKeyDirection(key: u32) -> vec3f {
    return ddgiRayDirectionDistance[key & DDGI_KEY_INDEX_MASK].xyz;
}

fn ddgiPackHalves(a: f32, b: f32, seed: u32) -> u32 {
    return pack2x16float(vec2f(ddgiStochasticHalf(a, ddgiRandom(seed, 1u, ddgi.frame)), ddgiStochasticHalf(b, ddgiRandom(seed, 2u, ddgi.frame))));
}

// Direction at the centre of interior texel 'texel' of a tile 'texels' wide.
fn ddgiTexelDirection(texel: vec2u, texels: u32) -> vec3f {
    return ddgiOctDecode((vec2f(texel) + 0.5) / f32(texels) * 2.0 - 1.0);
}

// Atlas texel of interior texel 'texel' of a probe's tile (the layer is probe.y).
fn ddgiAtlasTexel(probe: vec3i, texel: vec2u, texels: u32) -> vec2i {
    let border = i32(DDGI_TILE_BORDER);
    return probe.xz * (i32(texels) + 2 * border) + border + vec2i(texel);
}

// ---- Phase B: classification and relocation ----

// Whether a probe may sit at this offset (in spacings): inside its own cell
// (so the grid lookup stays valid) and inside the volume's box of probes.
// Why the box: a probe in an outer wall would otherwise often leave through
// the wall's outer face, the nearest one, and end up outside the room,
// where the room's own outside keeps it "near a surface" and active.
fn ddgiOffsetAllowed(probe: vec3i, offset: vec3f) -> bool {
    let position = vec3f(probe) + offset;
    let lastProbe = vec3f(ddgiProbeCount() - 1u);
    return all(abs(offset) <= vec3f(DDGI_MAX_OFFSET)) && all(position >= vec3f(0.0)) && all(position <= lastProbe);
}

// Each thread's share of the reduction: its fixed ray, or neutral values.
fn ddgiFixedRayResult(thread: u32, probe: vec3i, offset: vec3f) -> DdgiFixedRayResult {
    var result = DdgiFixedRayResult(0u, DDGI_KEY_NONE_MIN, DDGI_KEY_NONE_MIN, DDGI_KEY_NONE_MIN, DDGI_KEY_NONE_MAX, 0u);
    if (thread >= DDGI_FIXED_RAYS) {
        return result;
    }
    let ray = ddgiRayDirectionDistance[thread];
    let signedDistance = ray.w;
    if (signedDistance < 0.0) {
        result.backFace = 1u;
        // The true hit distance, from the stored shortened one.
        let distance = -signedDistance / DDGI_BACK_FACE_DISTANCE_SHARE;
        result.nearestBack = ddgiDistanceKey(distance, thread);
        // Only back faces the probe may leave through count for relocation.
        let exit = offset + ray.xyz * (distance + DDGI_PAST_SURFACE_SHARE * ddgi.minFrontFaceDistance) / ddgiUniformVolume().spacing;
        if (ddgiOffsetAllowed(probe, exit)) {
            result.closestBack = ddgiDistanceKey(distance, thread);
        }
    } else {
        // A miss counts as a front face very far away: the most open direction.
        result.closestFront = ddgiDistanceKey(signedDistance, thread);
        result.farthestFront = result.closestFront;
        // Within reach (see DDGI_REACH_SPACINGS).
        if (all(abs(ray.xyz * signedDistance) <= DDGI_REACH_SPACINGS * ddgiUniformVolume().spacing)) {
            result.frontInReach = 1u;
        }
    }
    return result;
}

struct DdgiProbeDecision {
    offset: vec3f,
    state: u32,
}

// Thread 0's verdict from the reductions (design: "Phase B").
fn ddgiDecide(probe: vec3i, offset: vec3f, state: u32) -> DdgiProbeDecision {
    let spacing = ddgiUniformVolume().spacing;
    let minFrontFace = ddgi.minFrontFaceDistance;
    let backFaces = atomicLoad(&ddgiBackFaceCount);
    let closestBack = atomicLoad(&ddgiClosestBackFace);  // the nearest one the probe may leave through
    let closestFront = atomicLoad(&ddgiClosestFrontFace);
    let farthestFront = atomicLoad(&ddgiFarthestFrontFace);
    let hasFront = closestFront != DDGI_KEY_NONE_MIN;
    let closestFrontDistance = select(DDGI_NO_DISTANCE, ddgiKeyDistance(closestFront), hasFront);
    let nearestBack = atomicLoad(&ddgiNearestBackFace);
    let nearestBackDistance = select(DDGI_NO_DISTANCE, ddgiKeyDistance(nearestBack), nearestBack != DDGI_KEY_NONE_MIN);
    // Inside something: the paper's rule (more than 25% back faces), or the
    // nearest thing the probe sees is a back face. The second catches probes
    // inside thin boards that overlap other solids: there many fixed rays hit
    // the other solids' faces, which count as front faces, and the share of
    // back faces stays under 25% (T1.5b, three probes in the cabinet row).
    let inside = f32(backFaces) > DDGI_INSIDE_BACK_FACE_SHARE * f32(DDGI_FIXED_RAYS) || nearestBackDistance < closestFrontDistance;

    var newOffset = offset;
    if (ddgi.relocation != 0u) {
        var step = vec3f(0.0);
        if (inside && closestBack != DDGI_KEY_NONE_MIN && state != DDGI_STATE_OFF_INSIDE_GEOMETRY) {
            // Inside something: through the closest back face it may leave by,
            // to just past it. Once: a probe already off for being inside had
            // its chance and stays put while it is still inside. Where solids
            // overlap (corners of wall and floor slabs), hopping on put it in
            // the other solid and back again for ever (T1.5b review).
            step = ddgiKeyDirection(closestBack) * (ddgiKeyDistance(closestBack) + DDGI_PAST_SURFACE_SHARE * minFrontFace);
        } else if (!inside && closestFrontDistance < minFrontFace) {
            // Too close to a surface: away along the most open direction, by
            // the shortfall, unless that heads back towards the close face.
            let away = ddgiKeyDirection(farthestFront);
            if (dot(away, ddgiKeyDirection(closestFront)) <= DDGI_MAX_COS_TOWARDS_CLOSEST) {
                step = away * (minFrontFace - closestFrontDistance);
            }
        } else if (!inside) {
            // In open space: ease back towards the grid point, unless that
            // brings the closest front face nearer than minFrontFaceDistance
            // (a probe moved out of a wall must not drift back into it).
            let back = -offset * spacing * DDGI_EASE_BACK_SHARE;
            let towardsClosest = select(0.0, dot(back, ddgiKeyDirection(closestFront)), hasFront);
            if (closestFrontDistance - towardsClosest >= minFrontFace) {
                step = back;
            }
        }
        let candidate = offset + step / spacing;
        // A new offset only where the probe may sit, and only for a real move;
        // otherwise the old one stays.
        if (length(step) >= DDGI_MIN_MOVE_SHARE * minFrontFace && ddgiOffsetAllowed(probe, candidate)) {
            newOffset = candidate;
        }
    }

    var newState = DDGI_STATE_ACTIVE;
    if (ddgi.classification != 0u && inside) {
        // Inside geometry, after relocation had its chance: if it moved, the
        // next update looks again from the new place.
        newState = DDGI_STATE_OFF_INSIDE_GEOMETRY;
    } else if (ddgi.classification != 0u && atomicLoad(&ddgiFrontFacesInReach) == 0u) {
        // No surface within reach: the probe helps no surface's interpolation.
        newState = DDGI_STATE_OFF_NO_SURFACE_NEARBY;
    } else if (ddgiIsOff(state)) {
        // Coming back on: it traced only its fixed rays this time, so it is
        // blended (from scratch) at its next update.
        newState = DDGI_STATE_NOT_UPDATED_YET;
    }
    return DdgiProbeDecision(newOffset, newState);
}

// ---- Phase C: blend ----

// One probe's history (PROBE_HISTORY_WORD in layout.ts; 48 bytes). Words
// later tasks fill are kept as they are.
struct DdgiProbeHistory {
    samples: f32,            // n: the updates its history counts as
    lastUpdateSeconds: f32,  // t: the technique's clock at its last blended update
    updates: u32,            // k in the low 16 bits (blended updates since new, reset or back on); updates owed at priority in bits 16-23; the sky transfer's count (T2.6b) above
    pendingChange: f32,      // r: the change ddgi/mark added up since its last update
    urgency: f32,            // the scheduler's urgency: the largest change since the mark, kept while updates are owed
    scheduleAndFlags: u32,   // flags in the low 8 bits (DDGI_FLAG_*); the frame it was last scheduled above
    meanRedGreen: u32,       // its mean irradiance, r and g as half floats
    meanBlueNoise: u32,      // b, and its noise, as half floats
    wakeUntil: u32,          // M3
    keysReceived: u32,       // T2.4c
    changeClock: u32,        // T2.4c
    spare: u32,
}

// This update's hysteresis for one probe, and the probe's history after it
// (its mean and noise are filled in once phase C has the estimate).
struct DdgiHysteresisStep {
    hysteresis: f32,
    history: DdgiProbeHistory,
    kind: u32,               // DDGI_UPDATE_*
}

// The ramp: the most a probe with k blended updates may remember. For its
// first K updates (rampUpdates, 16) that is k / a (updatesPerSample, 8); then
// one more per update, as a plain mean grows.
// Why: an update's bounce light reads the field the probe is still building,
// so its first estimates are too dark, and a plain mean (h = k / (k + 1))
// keeps them at full weight: its error from them shrinks only like
// k^-(1 - rho), with rho the share of a probe's light that comes back through
// the field. Cornell's 16-pass warm-up came out 25% low that way. Capped at
// k / a, every-frame updates get h = k / (k + a): a mean weighted towards
// later updates (weights grow like k^(a - 1)), whose error shrinks like
// k^-(a (1 - rho)), at a / sqrt(2a - 1) times the noise of a plain mean.
// Why only K updates: under the cap an every-frame probe would reach n_max
// only after a * n_max updates and stay noisy until then; after the ramp it
// gets there K / a + n_max updates in. A sparse probe settles on the clock at
// a few samples, so the ramp no longer binds once it has had a few updates.
fn ddgiRampCap(updates: u32) -> f32 {
    let k = f32(updates);
    let rampEnd = ddgi.rampUpdates;
    return select(rampEnd / ddgi.updatesPerSample + (k - rampEnd), k / ddgi.updatesPerSample, k <= rampEnd);
}

// The hysteresis rule (docs/design/ddgi.md, "Hysteresis that knows the time
// between updates"). A probe's history counts as n updates. Before this
// update, n fades with the time since the last one, n * exp(-elapsed / tau),
// so a probe updated rarely remembers less and follows changes within
// seconds, while one updated every frame builds up to n_max and goes quiet;
// it is also held to the ramp above. The new estimate then counts as one more
// update: h = n / (n + 1). After it, n + 1, capped at n_max, so h never
// exceeds n_max / (n_max + 1).
// A probe's first blended update (state 3: new, reset, or back on) starts
// from nothing (n = 0, k = 0), h = 0, so the field shows partial GI at once
// instead of fading in from black. The history is kept with the fixed rule
// too, so switching rules mid-run carries on from the right count.
//
// Before all that, a change the scene reported cuts n (ddgiChangeLimit), so
// the probe follows it within a few updates; with no change it does nothing.
fn ddgiHysteresisStep(history: DdgiProbeHistory, state: u32) -> DdgiHysteresisStep {
    let firstUpdate = state == DDGI_STATE_NOT_UPDATED_YET;
    let flags = history.scheduleAndFlags & DDGI_HISTORY_FLAGS_MASK;
    let marked = (flags & DDGI_FLAG_MARKED) != 0u;
    let owed = (history.updates >> DDGI_OWED_SHIFT) & DDGI_OWED_MASK;
    // The cut is for the updates the probe was promised (ddgi/mark: K, or
    // fewer for far probes after a global change); K when none are owed.
    let promised = select(ddgi.changeUpdates, f32(owed), owed > 0u);
    let limit = select(ddgiChangeLimit(history.pendingChange, flags, promised), DDGI_NO_HISTORY_LIMIT, firstUpdate);
    // After a cut the count grows back by one an update, as the design has it.
    // (Moving the probe back along the warm-up ramp as well was measured and
    // made it worse: on two-room's lamp step the view overshot further, 0.30 s
    // against 0.03 s at lightChangeScale 0.5; PROGRESS 2026-11-06, T2.4a.)
    let updates = select(history.updates & DDGI_HISTORY_COUNT_MASK, 0u, firstUpdate);
    // The clock wraps (CLOCK_WRAP_SECONDS): a negative difference means it
    // wrapped since this probe's last update, so add the period back.
    let sinceLastUpdate = ddgi.clockSeconds - history.lastUpdateSeconds;
    let elapsedSeconds = select(sinceLastUpdate, sinceLastUpdate + DDGI_CLOCK_WRAP_SECONDS, sinceLastUpdate < 0.0);
    // The cut comes first, then T1.12's fade and ramp as usual (m2-lights.md §4.1).
    let faded = select(min(history.samples, limit) * exp(-elapsedSeconds * ddgi.inverseMemorySeconds), 0.0, firstUpdate);
    let remembered = min(faded, ddgiRampCap(updates));
    let adaptive = remembered / (remembered + 1.0);
    // The fixed rule keeps its h, lowered to what the cut allows. Only when
    // there is a cut: GPUs divide through a reciprocal, and 1 / (3e38 + 1) is
    // a denormal they flush to 0, which made "no cut" read as h = 0 (caught by
    // the flicker page, whose converging runs the fixed rule).
    let cutHysteresis = select(1.0, limit / (limit + 1.0), limit < DDGI_NO_HISTORY_LIMIT);
    let fixed = select(min(ddgi.hysteresis, cutHysteresis), 0.0, firstUpdate);
    let hysteresis = select(fixed, adaptive, ddgi.adaptiveHysteresis != 0u);
    let samples = min(remembered + 1.0, ddgi.maxHistorySamples);
    // Saturating at the field's width: the count only matters while k / a is below n_max.
    let nextCount = select(updates + 1u, updates, updates == DDGI_HISTORY_COUNT_MASK);
    // One owed update is taken off; a first update starts with nothing owed
    // (a reset leaves old promises in the record).
    let nextOwed = select(owed - min(owed, 1u), 0u, firstUpdate);
    let takingChange = (flags & DDGI_FLAG_TAKING_CHANGE) != 0u;
    var next = history;
    next.samples = samples;
    next.lastUpdateSeconds = ddgi.clockSeconds;
    next.updates = (history.updates & ~(DDGI_HISTORY_COUNT_MASK | (DDGI_OWED_MASK << DDGI_OWED_SHIFT))) | nextCount | (nextOwed << DDGI_OWED_SHIFT);
    // This update takes the change in: it starts again from nothing, and the
    // steady-change rule remembers whether this update had been marked. The
    // awaiting flag (T2.4c) clears here too: the probe has been traced itself.
    // The urgency stays while updates are owed, so the scheduler keeps its
    // promise of all of them, not only the first.
    next.pendingChange = 0.0;
    next.urgency = select(0.0, history.urgency, nextOwed > 0u);
    let keptFlags = flags & ~(DDGI_FLAG_MARKED | DDGI_FLAG_MARKED_BEFORE_LAST_UPDATE | DDGI_FLAG_AWAITING_CHANGE | DDGI_FLAG_TAKING_CHANGE);
    let stillTaking = takingChange && nextOwed > 0u;
    let nextFlags = keptFlags | select(0u, DDGI_FLAG_MARKED_BEFORE_LAST_UPDATE, marked) | select(0u, DDGI_FLAG_TAKING_CHANGE, stillTaking);
    next.scheduleAndFlags = (history.scheduleAndFlags & ~DDGI_HISTORY_FLAGS_MASK) | select(nextFlags, 0u, firstUpdate);
    let kind = select(select(DDGI_UPDATE_QUIET, DDGI_UPDATE_MARKED, marked || takingChange), DDGI_UPDATE_FIRST, firstUpdate);
    return DdgiHysteresisStep(hysteresis, next, kind);
}

// The most samples a probe may keep after the changes reported since its
// last update (docs/design/m2-lights.md §4.1). r is the relative change of
// its light that ddgi/mark added up, epsilon the tolerance, K the updates it
// was promised (the scheduler serves them soon):
// - a step: the error left after K updates is r * n / (n + K); setting that
//   to epsilon gives n <= epsilon K / (r - epsilon) when r > epsilon, and
//   nothing for smaller changes;
// - a change that keeps going (marked before its previous update too): it
//   moves the probe only a little between two updates, so the step rule
//   alone would leave the history long and the probe trailing the change by
//   about tau of it; n <= epsilon / r keeps the lag near epsilon.
// Symmetric on purpose: brighter or darker, the cut is by the size of the
// change, so a lamp switched on arrives as fast as one switched off.
fn ddgiChangeLimit(pendingChange: f32, flags: u32, promised: f32) -> f32 {
    if (ddgi.changeDrivenHistory == 0u || pendingChange <= 0.0) {
        return DDGI_NO_HISTORY_LIMIT;
    }
    let tolerance = ddgi.changeTolerance;
    var limit = DDGI_NO_HISTORY_LIMIT;
    if (pendingChange > tolerance) {
        limit = tolerance * promised / (pendingChange - tolerance);
    }
    let steady = (flags & DDGI_FLAG_MARKED) != 0u && (flags & DDGI_FLAG_MARKED_BEFORE_LAST_UPDATE) != 0u;
    if (steady) {
        limit = min(limit, tolerance / pendingChange);
    }
    return limit;
}

// The cosine-weighted mean radiance of the blended rays around one
// irradiance texel's direction, times pi: the texel's new estimate of E.
fn ddgiEstimateIrradianceTexel(texelIndex: u32, blendedRays: u32) -> vec3f {
    let texel = vec2u(texelIndex % DDGI_IRRADIANCE_TEXELS, texelIndex / DDGI_IRRADIANCE_TEXELS);
    let normal = ddgiTexelDirection(texel, DDGI_IRRADIANCE_TEXELS);
    var radianceSum = vec3f(0.0);
    var weightSum = 0.0;
    for (var blended = 0u; blended < blendedRays; blended++) {
        let ray = DDGI_FIXED_RAYS + blended;
        let weight = max(0.0, dot(normal, ddgiRayDirectionDistance[ray].xyz));
        let packed = ddgiRayRadiance[ray];
        radianceSum += weight * vec3f(unpack2x16float(packed.x), unpack2x16float(packed.y).x);
        weightSum += weight;
    }
    // With rays spread evenly over the sphere, the cosine-weighted mean of the
    // radiance over the hemisphere is E / pi; the field stores E.
    return DDGI_PI * radianceSum / max(weightSum, 1.0e-6);
}

fn ddgiWriteIrradianceTexel(slot: u32, probeIndex: u32, state: u32, texelIndex: u32, probeHysteresis: f32, fastHysteresis: bool, changedNow: bool) {
    let estimate = ddgiTexelEstimate[texelIndex].rgb;
    let old = ddgiTexelOld[texelIndex].rgb;
    var hysteresis = probeHysteresis;
    if (fastHysteresis) {
        hysteresis *= DDGI_BIG_CHANGE_HYSTERESIS_SCALE;
    }
    var blendedValue = mix(estimate, old, hysteresis);
    if (state != DDGI_STATE_NOT_UPDATED_YET) {
        // Firefly clamp: one update may change a channel by at most
        // maxBrightnessChange times its old value, so one lucky ray can't
        // flash the probe.
        let limit = ddgi.maxBrightnessChange * max(old, vec3f(DDGI_SMALLEST_NORMAL_HALF));
        blendedValue = old + clamp(blendedValue - old, -limit, limit);
    }
    let base = slot * DDGI_SCRATCH_STRIDE + DDGI_SCRATCH_IRRADIANCE + 2u * texelIndex;
    let seed = probeIndex * DDGI_IRRADIANCE_TEXEL_COUNT + texelIndex;
    ddgiScratch[base] = ddgiPackHalves(blendedValue.r, blendedValue.g, seed);
    // Alpha carries this update's big-change verdict to the next update.
    ddgiScratch[base + 1u] = ddgiPackHalves(blendedValue.b, select(0.0, 1.0, changedNow), seed ^ 0x5bd1e995u);
}

fn ddgiBlendDistanceTexel(slot: u32, probeIndex: u32, probe: vec3i, texelIndex: u32, blendedRays: u32, hysteresis: f32) {
    let texel = vec2u(texelIndex % DDGI_DISTANCE_TEXELS, texelIndex / DDGI_DISTANCE_TEXELS);
    let normal = ddgiTexelDirection(texel, DDGI_DISTANCE_TEXELS);
    var moments = vec2f(0.0);
    var weightSum = 0.0;
    for (var blended = 0u; blended < blendedRays; blended++) {
        let directionDistance = ddgiRayDirectionDistance[DDGI_FIXED_RAYS + blended];
        // A sharp cosine power: distances are about one direction, not a hemisphere.
        let weight = pow(max(0.0, dot(normal, directionDistance.xyz)), ddgi.distanceSharpness);
        let distance = min(abs(directionDistance.w), ddgi.distanceClamp);
        moments += weight * vec2f(distance, distance * distance);
        weightSum += weight;
    }
    let old = textureLoad(ddgiDistanceAtlas, ddgiAtlasTexel(probe, texel, DDGI_DISTANCE_TEXELS), probe.y, 0).xy;
    // A texel no ray came near this update keeps its old value.
    let estimate = select(old, moments / weightSum, weightSum > 1.0e-6);
    let blendedValue = mix(estimate, old, hysteresis);

    let base = slot * DDGI_SCRATCH_STRIDE + DDGI_SCRATCH_DISTANCE + texelIndex;
    let seed = (probeIndex * DDGI_DISTANCE_TEXEL_COUNT + texelIndex) ^ 0x68e31da4u;
    ddgiScratch[base] = ddgiPackHalves(blendedValue.x, blendedValue.y, seed);
}

// Thread 0, once phase C has the estimate: the probe's history after this
// update, with its new mean irradiance and noise (m2-lights.md §4.1, words
// 6 and 7). One workgroup per probe and a probe is listed at most once a
// frame, so this write can't race.
// - The mean: the blend moves every texel by the same share 1 - h, so the
//   mean of the new texels is the same mix of the two means (the firefly
//   clamp aside, which only acts on outliers). ddgi/mark divides a light's
//   change by it.
// - The noise: how far an update's mean estimate lands from the stored mean,
//   |estimate - old| (the blend moves the mean by (1 - h) of that, so this
//   doesn't depend on h), as the root of a running mean of its square, the
//   channels averaged. Only quiet updates feed it: a reported change, or one
//   still being taken in, isn't noise.
// - The feedback rule compares a quiet update's change with that noise and
//   may ask for more updates (ddgiFeedbackFires).
fn ddgiWriteHistory(probeIndex: u32, probe: vec3i, estimateSum: vec3f, oldSum: vec3f, probeHysteresis: f32, fastHysteresis: bool) {
    let texelCount = f32(DDGI_IRRADIANCE_TEXEL_COUNT);
    let estimateMean = estimateSum / texelCount;
    let oldMean = oldSum / texelCount;
    let hysteresis = select(probeHysteresis, probeHysteresis * DDGI_BIG_CHANGE_HYSTERESIS_SCALE, fastHysteresis);
    let mean = max(mix(estimateMean, oldMean, hysteresis), vec3f(0.0));
    var history = ddgiNextHistory;
    let previousNoise = unpack2x16float(history.meanBlueNoise).y;
    // |estimate - stored| per channel, the channels' root mean square. The
    // blend moves the mean by (1 - h) of it, so this is the feedback rule's
    // |change of the mean| / (1 - h), the same whatever h is (after a cut, say).
    let deviation = abs(estimateMean - oldMean);
    let deviationSquared = dot(deviation, deviation) / 3.0;
    let deviationSize = sqrt(deviationSquared);
    let meanSize = sqrt(dot(oldMean, oldMean) / 3.0);
    // The updates blended before this one (the record already counts this one).
    let blendedBefore = (history.updates & DDGI_HISTORY_COUNT_MASK) - 1u;
    let fired = ddgiUpdateKind == DDGI_UPDATE_QUIET && ddgiFeedbackFires(deviationSize, previousNoise, meanSize, blendedBefore);
    var noise = previousNoise;
    if (ddgiUpdateKind == DDGI_UPDATE_FIRST) {
        noise = 0.0;
    } else if (ddgiUpdateKind == DDGI_UPDATE_QUIET) {
        // An update that tripped the feedback rule saw a change, not noise.
        // The first quiet update starts the average at its own value.
        let averaged = select(mix(previousNoise * previousNoise, deviationSquared, 1.0 / DDGI_NOISE_UPDATES), deviationSquared, previousNoise <= 0.0);
        noise = sqrt(averaged);
    }
    history.meanRedGreen = pack2x16float(mean.rg);
    history.meanBlueNoise = pack2x16float(vec2f(mean.b, noise));
    ddgiProbeHistory[probeIndex] = history;
    atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_BLENDED_TOTAL], 1u);
    if (fired) {
        ddgiRequestFeedback(probeIndex, probe, deviationSize / max(meanSize, DDGI_SMALLEST_NORMAL_HALF));
        atomicAdd(&ddgiCensus[DDGI_CENSUS_FEEDBACK_FIRED], 1u);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_FEEDBACK_FIRED_TOTAL], 1u);
    }
}

// The feedback rule's test (m2-lights.md §4.6) for a quiet update: the
// estimate lands more than 4 sigma from the stored value (sigma: the probe's
// noise word) and more than 2% of it. At 4 sigma a Gaussian noise trips about
// once in 16,000 updates. Not during a probe's warm-up ramp (its first
// rampUpdates updates are expected to move while bounce light fills in, and
// its noise isn't averaged yet), nor before it has a noise at all.
fn ddgiFeedbackFires(deviation: f32, noise: f32, mean: f32, blendedBefore: u32) -> bool {
    if (ddgi.feedback == 0u || !(noise > 0.0) || f32(blendedBefore) < ddgi.rampUpdates) {
        return false;
    }
    return deviation > DDGI_FEEDBACK_NOISE_MULTIPLE * noise && deviation > DDGI_FEEDBACK_MIN_SHARE * mean;
}

// One more update for the probe at this urgency, and for its active face
// neighbours at half of it. In a buffer of its own: the blend never writes
// another probe's history (each workgroup owns only its own probe).
// Non-negative floats order like their bits, so atomicMax keeps the largest.
fn ddgiRequestFeedback(probeIndex: u32, probe: vec3i, urgency: f32) {
    atomicMax(&ddgiFeedbackRequests[probeIndex], bitcast<u32>(urgency));
    let counts = vec3i(ddgiProbeCount());
    let neighbourUrgency = bitcast<u32>(urgency * DDGI_FEEDBACK_NEIGHBOUR_SHARE);
    for (var face = 0u; face < 6u; face++) {
        var neighbour = probe;
        neighbour[face % 3u] += select(1, -1, face >= 3u);
        if (any(neighbour < vec3i(0)) || any(neighbour >= counts)) {
            continue;
        }
        if (ddgiReadProbeData(ddgiProbeData, neighbour).state != DDGI_STATE_ACTIVE) {
            continue;
        }
        let neighbourIndex = u32(neighbour.x) + u32(counts.x) * (u32(neighbour.z) + u32(counts.z) * u32(neighbour.y));
        atomicMax(&ddgiFeedbackRequests[neighbourIndex], neighbourUrgency);
    }
}

// A listed probe that isn't blended this update (off, or switched off by it)
// owes nothing: off probes trace only their fixed rays and take no light,
// and a promise left in its record would keep it at priority for ever.
fn ddgiForgetPromises(probeIndex: u32) {
    var history = ddgiProbeHistory[probeIndex];
    history.updates = history.updates & ~(DDGI_OWED_MASK << DDGI_OWED_SHIFT);
    history.urgency = 0.0;
    history.scheduleAndFlags = history.scheduleAndFlags & ~DDGI_FLAG_TAKING_CHANGE;
    ddgiProbeHistory[probeIndex] = history;
}

@compute @workgroup_size(DDGI_BLEND_WORKGROUP_SIZE)
fn ddgiBlend(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    // Dispatches past 65,535 workgroups use two dimensions.
    let slot = group.x + group.y * groups.x;
    if (slot >= ddgi.updateCount) {
        return;
    }
    let probeIndex = ddgiUpdateList[slot];
    let probe = ddgiProbeCoordinates(probeIndex, ddgiProbeCount());
    let probeData = ddgiReadProbeData(ddgiProbeData, probe);
    let state = probeData.state;
    let tracedRays = ddgiTracedRays(state);
    let blendedRays = ddgi.raysPerProbe - DDGI_FIXED_RAYS;

    if (thread == 0u) {
        // Every reduction starts from its neutral value, set here rather
        // than relying on workgroup memory starting at zero.
        atomicStore(&ddgiBackFaceCount, 0u);
        atomicStore(&ddgiNearestBackFace, DDGI_KEY_NONE_MIN);
        atomicStore(&ddgiClosestBackFace, DDGI_KEY_NONE_MIN);
        atomicStore(&ddgiClosestFrontFace, DDGI_KEY_NONE_MIN);
        atomicStore(&ddgiFarthestFrontFace, DDGI_KEY_NONE_MAX);
        atomicStore(&ddgiFrontFacesInReach, 0u);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_RAYS_TRACED], tracedRays);
    }
    for (var rayIndex = thread; rayIndex < tracedRays; rayIndex += DDGI_BLEND_WORKGROUP_SIZE) {
        let record = ddgiRays[slot * ddgi.raysPerProbe + rayIndex];
        let signedDistance = unpack2x16float(record.y).y;
        ddgiRayDirectionDistance[rayIndex] = vec4f(ddgiRayDirection(rayIndex), signedDistance);
        ddgiRayRadiance[rayIndex] = record;
    }
    workgroupBarrier();

    // ---- Phase B ----
    ddgiReduceFixedRay(ddgiFixedRayResult(thread, probe, probeData.offset));
    workgroupBarrier();
    if (thread == 0u) {
        let decision = ddgiDecide(probe, probeData.offset, state);
        // Phase C runs for probes that traced all their rays and stay active.
        // (The write kernel copies the tiles exactly when the new state is active.)
        let blendThisUpdate = decision.state == DDGI_STATE_ACTIVE && !ddgiIsOff(state);
        ddgiBlendThisUpdate = select(0u, 1u, blendThisUpdate);
        if (blendThisUpdate) {
            // One workgroup per probe in the list, and a probe is listed at
            // most once a frame, so this read and write can't race.
            let step = ddgiHysteresisStep(ddgiProbeHistory[probeIndex], state);
            ddgiUpdateHysteresis = step.hysteresis;
            ddgiNextHistory = step.history;
            ddgiUpdateKind = step.kind;
        } else {
            ddgiForgetPromises(probeIndex);
        }
        // Offsets and states are small exact values in half floats; no rounding noise wanted.
        let base = slot * DDGI_SCRATCH_STRIDE + DDGI_SCRATCH_PROBE_DATA;
        ddgiScratch[base] = pack2x16float(decision.offset.xy);
        ddgiScratch[base + 1u] = pack2x16float(vec2f(decision.offset.z, f32(decision.state)));
    }
    if (workgroupUniformLoad(&ddgiBlendThisUpdate) == 0u) {
        return;
    }
    let hysteresis = workgroupUniformLoad(&ddgiUpdateHysteresis);

    // ---- Phase C ----
    // Step 1: each irradiance texel's estimate and old value; the distance texels in full.
    for (var texelIndex = thread; texelIndex < DDGI_IRRADIANCE_TEXEL_COUNT + DDGI_DISTANCE_TEXEL_COUNT; texelIndex += DDGI_BLEND_WORKGROUP_SIZE) {
        if (texelIndex < DDGI_IRRADIANCE_TEXEL_COUNT) {
            let texel = vec2u(texelIndex % DDGI_IRRADIANCE_TEXELS, texelIndex / DDGI_IRRADIANCE_TEXELS);
            ddgiTexelEstimate[texelIndex] = vec4f(ddgiEstimateIrradianceTexel(texelIndex, blendedRays), 0.0);
            ddgiTexelOld[texelIndex] = textureLoad(ddgiIrradianceAtlas, ddgiAtlasTexel(probe, texel, DDGI_IRRADIANCE_TEXELS), probe.y, 0);
        } else {
            ddgiBlendDistanceTexel(slot, probeIndex, probe, texelIndex - DDGI_IRRADIANCE_TEXEL_COUNT, blendedRays, hysteresis);
        }
    }
    workgroupBarrier();

    // Step 2: the big-change test, opt-in (irradianceChangeThreshold; a
    // negative value means off, the M1 default, DECISIONS 2026-11-05). It
    // looks at the probe's mean irradiance against its old mean, not per
    // texel, and its verdict sets the hysteresis of the probe's *next* update,
    // not this one.
    // Why not this one: a test on this update's estimate is tripped by its own
    // ray noise, and the noise is skewed (a few rays find a small bright spot),
    // so the updates that trip are mostly the bright ones, and they get the
    // lower hysteresis: about 9x the weight. That biased the field upwards
    // (per texel: in-air Cornell +8.3%; on the probe mean as well: +21.7%;
    // with no test: +3.3%, T1.5b). The next update's rays are independent of
    // this one's noise, so it blends without bias, and a real light switch
    // still trips the test again and gets the fast hysteresis one update late.
    // The verdict travels in the irradiance atlas's alpha, which nothing samples.
    if (thread == 0u) {
        var estimateSum = vec3f(0.0);
        var oldSum = vec3f(0.0);
        for (var texelIndex = 0u; texelIndex < DDGI_IRRADIANCE_TEXEL_COUNT; texelIndex++) {
            estimateSum += ddgiTexelEstimate[texelIndex].rgb;
            oldSum += ddgiTexelOld[texelIndex].rgb;
        }
        // The same share applies to the sums as to the means.
        let ruleOn = ddgi.irradianceChangeThreshold >= 0.0;
        ddgiChangedNow = select(0u, 1u, ruleOn && any(abs(estimateSum - oldSum) > ddgi.irradianceChangeThreshold * oldSum));
        ddgiFastHysteresis = select(0u, 1u, ddgiTexelOld[0].a > 0.5);
        ddgiWriteHistory(probeIndex, probe, estimateSum, oldSum, hysteresis, ddgiFastHysteresis != 0u);
    }
    let changedNow = workgroupUniformLoad(&ddgiChangedNow) != 0u;
    let fastHysteresis = workgroupUniformLoad(&ddgiFastHysteresis) != 0u;

    // Step 3: blend every irradiance texel with the probe's hysteresis.
    for (var texelIndex = thread; texelIndex < DDGI_IRRADIANCE_TEXEL_COUNT; texelIndex += DDGI_BLEND_WORKGROUP_SIZE) {
        ddgiWriteIrradianceTexel(slot, probeIndex, state, texelIndex, hysteresis, fastHysteresis, changedNow);
    }
}
`;
}
