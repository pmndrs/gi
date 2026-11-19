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
 * Key probes and propagation (T2.4c, m2-lights.md §4.5):
 * - every blended update stamps the volume's change clock (census) in the
 *   probe's record;
 * - a key probe whose update takes in a reported change writes the change
 *   it made, new − old per irradiance texel, to the key-delta buffer at its
 *   update-list slot, with its clock stamp from before this update, and
 *   flags itself, for `ddgi/propagate` later in the frame;
 * - a fine probe that was waiting for its keys is cut only for what
 *   propagation hasn't brought it: its change times
 *   max(keyResidualShare, 1 − 0.8 × the weight of the keys that passed it
 *   theirs), since a key's first update moves it about 80% of the way;
 * - the test mode `holdFineProbes`: a fine probe waiting for its keys is
 *   traced but not blended (its tiles and record stay as they are), so only
 *   propagation moves it.
 *
 * Probes near moved geometry (T2.M1, docs/design/m3-moving.md §5):
 * - the mover mask: each blended ray's hit point is tested against the boxes
 *   of the hot movers (the mover table, a uniform buffer the CPU fills from
 *   the scene's moves, mover-table.ts), and the probe's record keeps which
 *   of them its rays hit this update. `ddgi/mark` adds a probe to a moved
 *   object's far set when it has the object's bit. A ray that ends inside a
 *   mover's box but on something else (the floor under it) only over-marks;
 *   a ray that hits the mover always ends inside its box, so none is missed.
 *   *Why here and not in the trace:* the hit point is the probe's position
 *   plus the ray's direction times its stored distance, so no instance id
 *   has to travel from the trace, whose storage buffers are full;
 * - a probe in state 3 (new, reset, or back on) is used in shading only
 *   after its second blended update (§5.2): one noisy estimate is a pop. Its
 *   first blended update starts its history from nothing (h = 0) and keeps it
 *   in state 3 (the `warmedOnce` flag); its second blends as usual and makes
 *   it active. The scratch's tiles-written word tells `ddgi/write` that both
 *   have tiles to copy;
 * - a probe that comes back on keeps the updates it was owed, so its two
 *   warming updates come at priority;
 * - woken air probes (T3.4, §5.3): a probe `ddgi/mark` woke near a receiver
 *   or a moved or posed object (its `wakeUntil` is still ahead) isn't switched off
 *   for "no surface nearby"; one that was off comes back on through state 3
 *   like any other, so it too is used only after 2 blended updates. The
 *   first update after the wake ran out clears it, and classification works
 *   as before.
 *
 * Light groups (T2.5, light-groups.ts; m2-lights.md §5): with several
 * irradiance layers, each ray has a result per layer, and phase C blends
 * every layer's texels with the probe's one hysteresis (the history is per
 * probe, shared by its layers), one thread per (layer, texel), all in one
 * step beside the distance texels. The directions and distances are in
 * workgroup memory as before; with several layers the radiance is read
 * straight from the ray buffer instead (every layer's would not fit the
 * 16 KB of workgroup memory; the threads of a texel row read the same ray at
 * once, so the reads are shared). The design's first sketch blended the
 * layers one after the other through one radiance array, which left most of
 * the workgroup idle in every extra layer: on two-room the blend cost 2.6×
 * with 5 layers (T2.5). The probe's mean irradiance and noise, and the
 * opt-in big-change test, are over the layers' sum, the probe's whole light;
 * the big-change verdict lives in layer 0's alpha. With one layer (groups
 * off) this is the kernel as it was.
 *
 * RTXGI's back-face rule (TU.5, DV-01 in docs/divergence.md;
 * `settings.backFaceRayThreshold`): blended rays that hit a back face are
 * left out of the irradiance average instead of counting as black samples,
 * and an update whose blended rays hit back faces too often
 * (`ddgiTooManyBackFaces`) isn't blended at all: its irradiance tiles and its
 * history stay as they were (`DDGI_UPDATE_SKIPPED`), while its distances blend
 * as usual, as RTXGI's separate distance pass does. RTXGI's blend simply
 * returns there; a skipped update here also keeps a probe in state 3 and
 * takes one owed update off.
 *
 * Bindings (group 1; no platform module): uniforms, the update list, the ray
 * buffer, the scratch buffer, the census buffer, the probe history, the
 * feedback requests and the key deltas (7 storage buffers), the mover table
 * (a second uniform buffer), and the irradiance, distance and probe-data
 * atlases (sampled, for the old values).
 */

import {
    MOVER_TABLE_CAPACITY,
    SCRATCH_PROBE_DATA_WORD,
    CENSUS_BACK_FACE_SKIPS_TOTAL,
    CENSUS_CHANGE_CLOCK,
    CENSUS_FEEDBACK_FIRED,
    CENSUS_FEEDBACK_FIRED_TOTAL,
    CENSUS_KEY_CHANGES_PASSED,
    CENSUS_KEY_CHANGES_PASSED_TOTAL,
    CENSUS_PROBES_BLENDED_TOTAL,
    CENSUS_RAYS_TRACED,
    KEY_DELTA_HEADER,
    KEY_DELTA_HEADER_WORDS,
    keyDeltaStrideWords,
    PROBE_HISTORY_FLAG,
    PROBE_HISTORY_FLAGS_MASK,
    PROBE_HISTORY_KEYS,
    PROBE_HISTORY_UPDATES,
    type ScratchLayout,
} from '../layout.ts';
import { CLOCK_WRAP_SECONDS, FIXED_RAYS } from '../settings.ts';

/**
 * The feedback rule's thresholds (m2-lights.md §4.6): an update whose
 * estimate lands more than this many times the probe's noise from its stored
 * value, and more than {@link FEEDBACK_MIN_SHARE} of it...
 *
 * 6, not the design's 4 (T2.4b, DECISIONS 2026-11-06): DDGI's ray noise has
 * heavy tails (a few rays find a bright patch), so far more updates pass 4σ
 * than a Gaussian's 1 in 16,000. Measured on the RTX 3060 (`ddgi-feedback`,
 * still Cornell and two-room after the warm-up, frames 600-1,800): 4σ fired
 * on about 0.1-0.2% of updates, 5σ on 0.11-0.12%, 6σ on 0.07-0.09%, against
 * the bar of 0.1%. The unreported change still reached every probe within a
 * spacing of it within 0.5 s at 6.
 */
export const FEEDBACK_NOISE_MULTIPLE = 6;
/** ...asks for one more update (2% of the stored mean: below that a change can't matter for the 5% target). */
export const FEEDBACK_MIN_SHARE = 0.02;
/** Its 6 face neighbours ask at this share of its urgency: a change nobody reported spreads to the probes around it. */
export const FEEDBACK_NEIGHBOUR_SHARE = 0.5;

/**
 * How much of a change a key's first owed update is taken to pass on
 * (m2-lights.md §4.5): a fine probe whose keys have all passed theirs is
 * taken to have received this share of its change, and is cut for the rest
 * (at least keyResidualShare) when it is traced itself. The design's
 * estimate, from the cut a key gets for a large change.
 */
export const KEY_FIRST_UPDATE_SHARE = 0.8;

/**
 * Bits of a reduction key that hold the fixed ray's index (the rest is the
 * hit distance's float bits). Derived from the fixed-ray count, and kept to
 * at most 8 of the 23 mantissa bits so the distance stays precise.
 */
export const KEY_INDEX_BITS = Math.ceil(Math.log2(FIXED_RAYS));
if (KEY_INDEX_BITS > 8) throw new Error(`@pmndrs/gi: ${FIXED_RAYS} fixed rays need ${KEY_INDEX_BITS} key bits; phase B's keys allow 8`);
const KEY_INDEX_MASK = (1 << KEY_INDEX_BITS) - 1;
import { MAX_IRRADIANCE_LAYERS } from '../light-groups.ts';
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
    /**
     * The sun layer's index (T2.6b, `LayerPlan.sunLayer`), or -1 / omitted
     * without one. Its texels are at the sun's white reference, so the probe's
     * mean weighs them by the sun's light (`DdgiUniforms.sunLight`).
     */
    readonly sunLayer?: number;
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
    const sunLayer = shape.sunLayer ?? -1;
    if (sunLayer >= scratch.irradianceLayers) throw new RangeError(`@pmndrs/gi: the sun layer is ${sunLayer}, but the blend has ${scratch.irradianceLayers} layer(s)`);
    // The probe's mean over every layer: the sun layer weighed by the sun's light, the others as they are.
    const meanSum =
        sunLayer >= 0
            ? /* wgsl */ `
        // The sun layer holds the sun at a white reference: weighed
        // by the sun's colour x intensity, the mean is the probe's real light,
        // as without a sun layer (ddgi/mark divides a light's change by it).
        for (var item = 0u; item < layerTexels; item++) {
            let meanWeight = select(vec3f(1.0), ddgi.sunLight.rgb, item / DDGI_IRRADIANCE_TEXEL_COUNT == DDGI_SUN_LAYER);
            estimateSum += meanWeight * ddgiTexelEstimate[item].rgb;
            oldSum += meanWeight * ddgiTexelOld[item].rgb;
        }`
            : /* wgsl */ `
        for (var item = 0u; item < layerTexels; item++) {
            estimateSum += ddgiTexelEstimate[item].rgb;
            oldSum += ddgiTexelOld[item].rgb;
        }`;
    // Several irradiance layers (light groups): the radiance is read from the ray buffer (the file comment).
    const layered = scratch.irradianceLayers > 1;
    // Every layer's texel estimates and old values sit in workgroup memory beside the rays' directions (the file comment).
    if (scratch.irradianceLayers > MAX_IRRADIANCE_LAYERS) {
        throw new Error(
            `@pmndrs/gi: ddgi/blend keeps at most ${MAX_IRRADIANCE_LAYERS} irradiance layers in its 16 KB of workgroup memory, got ${scratch.irradianceLayers} (light-groups.ts, MAX_IRRADIANCE_LAYERS)`,
        );
    }
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
const DDGI_SCRATCH_OFFSET_XY: u32 = ${SCRATCH_PROBE_DATA_WORD.offsetXy}u;
const DDGI_SCRATCH_OFFSET_Z_AND_STATE: u32 = ${SCRATCH_PROBE_DATA_WORD.offsetZAndState}u;
const DDGI_SCRATCH_TILES_WRITTEN: u32 = ${SCRATCH_PROBE_DATA_WORD.tilesWritten}u;
// Irradiance layers (light groups, light-groups.ts): each ray has a result per
// layer, side by side; each layer's texels follow the one before in a scratch slot.
const DDGI_IRRADIANCE_LAYERS: u32 = ${scratch.irradianceLayers}u;
const DDGI_SCRATCH_IRRADIANCE_LAYER_WORDS: u32 = ${scratch.irradianceWords}u;
${sunLayer >= 0 ? `// The sun layer (T2.6b): at a white reference, weighed by ddgi.sunLight in the probe's mean.
const DDGI_SUN_LAYER: u32 = ${sunLayer}u;` : ''}
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
const DDGI_FLAG_PASSED_CHANGE: u32 = ${PROBE_HISTORY_FLAG.passedChange}u;
const DDGI_FLAG_WARMED_ONCE: u32 = ${PROBE_HISTORY_FLAG.warmedOnce}u;
const DDGI_FLAG_WAITING_FOR_KEYS: u32 = ${PROBE_HISTORY_FLAG.waitingForKeys}u;
// The mover mask (T2.M1). The trace stores a miss at 60,000 (wgsl/trace.ts);
// anything this far or farther is a miss and hits no mover.
const DDGI_MOVER_MISS_DISTANCE: f32 = 50000.0;
// A hit point is rebuilt from a half-float distance (about 1 part in 2,000)
// and an f32 direction, so a mover's box is grown by this much plus this
// share of the ray's length before the point is tested against it.
const DDGI_MOVER_BOX_MARGIN: f32 = 0.01;
const DDGI_MOVER_BOX_MARGIN_SHARE: f32 = 0.002;
const DDGI_KEYS_WEIGHT_SHIFT: u32 = ${PROBE_HISTORY_KEYS.weightShift}u;
const DDGI_KEY_FIRST_UPDATE_SHARE: f32 = ${KEY_FIRST_UPDATE_SHARE};
const DDGI_CENSUS_CHANGE_CLOCK: u32 = ${CENSUS_CHANGE_CLOCK}u;
const DDGI_CENSUS_KEY_CHANGES_PASSED: u32 = ${CENSUS_KEY_CHANGES_PASSED}u;
const DDGI_CENSUS_KEY_CHANGES_PASSED_TOTAL: u32 = ${CENSUS_KEY_CHANGES_PASSED_TOTAL}u;
// The key-delta buffer (KEY_DELTA_HEADER in layout.ts): per update-list slot,
// the key's probe index and clock stamp from before this update, then its
// change per irradiance texel.
const DDGI_KEY_DELTA_STRIDE: u32 = ${keyDeltaStrideWords(irradianceTexels, scratch.irradianceLayers)}u;
const DDGI_KEY_DELTA_HEADER_WORDS: u32 = ${KEY_DELTA_HEADER_WORDS}u;
const DDGI_KEY_DELTA_PROBE_INDEX: u32 = ${KEY_DELTA_HEADER.probeIndex}u;
const DDGI_KEY_DELTA_CLOCK_BEFORE: u32 = ${KEY_DELTA_HEADER.clockBefore}u;
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
// exponential average with weight 1 / this). Long, because the feedback rule
// compares single updates with it, and a noise read from few updates is often
// low by chance (at 16 the rule fired on 0.21% of still updates, at 64 on
// 0.18%, T2.4b). It is relative to the mean, so it needn't follow a change of
// the light's brightness.
const DDGI_NOISE_UPDATES: f32 = 64.0;
// Below this mean a probe counts as dark for the relative noise (MARK_DARK_IRRADIANCE in mark.ts).
const DDGI_DARK_MEAN: f32 = 1.0e-6;
// What kind of update this is, for the noise word (ddgiUpdateKind).
const DDGI_UPDATE_FIRST: u32 = 0u;   // state 3: the history starts again, noise 0
const DDGI_UPDATE_QUIET: u32 = 1u;   // no change reported: feeds the noise, and may trip the feedback rule
const DDGI_UPDATE_MARKED: u32 = 2u;  // a reported change, or still taking one in: the noise is kept as it was
const DDGI_UPDATE_HELD: u32 = 3u;    // the test mode holdFineProbes: a fine probe waiting for its keys keeps its tiles and its change
const DDGI_UPDATE_SKIPPED: u32 = 4u; // RTXGI's back-face rule: too many blended rays hit back faces, so the irradiance and history are kept
const DDGI_CENSUS_BACK_FACE_SKIPS_TOTAL: u32 = ${CENSUS_BACK_FACE_SKIPS_TOTAL}u;

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
// Per update-list slot: a key probe's change this update, for ddgi/propagate (DDGI_KEY_DELTA_*).
@group(1) @binding(10) var<storage, read_write> ddgiKeyDeltas: array<u32>;
// The hot movers' world boxes and mask bits (T2.M1, mover-table.ts; MOVER_TABLE_* in layout.ts).
@group(1) @binding(11) var<uniform> ddgiMovers: DdgiMoverTable;

struct DdgiMoverBox {
    min: vec3f,
    bit: u32,               // the entry's bit in the probes' mover masks
    max: vec3f,
    spare: u32,
}

struct DdgiMoverTable {
    count: u32,             // entries in use; 0 when nothing moved lately (then no ray is tested)
    spare0: u32,
    spare1: u32,
    spare2: u32,
    boxes: array<DdgiMoverBox, ${MOVER_TABLE_CAPACITY}>,
}

// Per ray: its direction (xyz) and signed hit distance (w, negative = back
// face). Decoded once here: the blend reads every ray once per texel.
var<workgroup> ddgiRayDirectionDistance: array<vec4f, DDGI_MAX_RAYS>;
${layered ? '' : `// Per ray: rgb radiance as half floats (rg, then b and the distance), as the
// trace wrote it. (With several layers the blend reads it from the ray buffer.)
var<workgroup> ddgiRayRadiance: array<vec2u, DDGI_MAX_RAYS>;`}
// Phase B's reductions over the fixed rays.
var<workgroup> ddgiBackFaceCount: atomic<u32>;
var<workgroup> ddgiNearestBackFace: atomic<u32>;
var<workgroup> ddgiClosestBackFace: atomic<u32>;
var<workgroup> ddgiClosestFrontFace: atomic<u32>;
var<workgroup> ddgiFarthestFrontFace: atomic<u32>;
var<workgroup> ddgiFrontFacesInReach: atomic<u32>;
// The mover bits this update's blended rays hit (ddgiMoverBits), for the probe's mover mask.
var<workgroup> ddgiMoverHits: atomic<u32>;
// How many of this update's blended rays hit back faces (RTXGI's back-face rule, ddgiTooManyBackFaces).
var<workgroup> ddgiBlendedBackFaces: atomic<u32>;
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
// 1 when this update is a key probe's that passes its change on (ddgiPassesChange).
var<workgroup> ddgiPassChange: u32;
// Phase C: each irradiance texel's new estimate and old value, every layer's
// (layer L's texel t at L × DDGI_IRRADIANCE_TEXEL_COUNT + t), for the
// big-change test and the mean over the probe.
var<workgroup> ddgiTexelEstimate: array<vec4f, DDGI_IRRADIANCE_TEXEL_COUNT * DDGI_IRRADIANCE_LAYERS>;
var<workgroup> ddgiTexelOld: array<vec4f, DDGI_IRRADIANCE_TEXEL_COUNT * DDGI_IRRADIANCE_LAYERS>;

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

// Atlas texel of interior texel 'texel' of a probe's tile (the array layer is ddgiProbeLayer(probe)).
fn ddgiAtlasTexel(probe: vec3i, texel: vec2u, texels: u32) -> vec2i {
    let border = i32(DDGI_TILE_BORDER);
    return probe.xz * (i32(texels) + 2 * border) + border + vec2i(texel);
}

// The same in irradiance layer 'layer': its tiles are 'layer' atlas widths of
// tiles to the right in the layered atlas (light-groups.ts).
fn ddgiIrradianceAtlasTexel(probe: vec3i, texel: vec2u, layer: u32) -> vec2i {
    let shifted = probe + vec3i(i32(layer) * ddgiAtlasTilesAcross(), 0, 0);
    return ddgiAtlasTexel(shifted, texel, DDGI_IRRADIANCE_TEXELS);
}

// A layer's own random numbers for the stochastic rounding: layer 0 keeps
// the seeds it had before there were layers.
fn ddgiLayerSeed(layer: u32) -> u32 {
    return layer * 0x9e3779b9u;
}

// Where a ray's result in layer 'layer' is in the ray buffer.
fn ddgiRayResult(slot: u32, rayIndex: u32, layer: u32) -> u32 {
    return (slot * ddgi.raysPerProbe + rayIndex) * DDGI_IRRADIANCE_LAYERS + layer;
}

// ---- Phase B: classification and relocation ----

// Whether a probe may sit at this offset (in spacings): inside its own cell
// (so the grid lookup stays valid) and inside the volume's box of probes
// (its window, for a camera-following volume: measured in window probes).
// Why the box: a probe in an outer wall would otherwise often leave through
// the wall's outer face, the nearest one, and end up outside the room,
// where the room's own outside keeps it "near a surface" and active.
fn ddgiOffsetAllowed(probe: vec3i, offset: vec3f) -> bool {
    let position = vec3f(ddgiWindowProbe(ddgiUniformVolume(), probe)) + offset;
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

// Whether a probe is awake (T3.4, m3-moving.md §5.3): ddgi/mark woke it near
// a receiver or a moved or posed object, and WAKE_FRAMES haven't passed since.
// Frames compare modulo 2^32, as the frame counter wraps; 0 is "not awake".
fn ddgiAwake(wakeUntil: u32) -> bool {
    return wakeUntil != 0u && bitcast<i32>(wakeUntil - ddgi.frame) > 0;
}

// Thread 0's verdict from the reductions (design: "Phase B"). 'awake': the
// probe was woken (ddgiAwake), so "no surface nearby" doesn't switch it off.
fn ddgiDecide(probe: vec3i, offset: vec3f, state: u32, awake: bool) -> DdgiProbeDecision {
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
    } else if (ddgi.classification != 0u && atomicLoad(&ddgiFrontFacesInReach) == 0u && !awake) {
        // No surface within reach: the probe helps no surface's interpolation.
        // Unless it is awake: a receiver or a moved object in mid-air needs it
        // (m3-moving.md §5.3). An awake probe that was off comes back on below.
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
    wakeUntil: u32,          // woken air probes (T3.4): awake until this frame; 0 when not woken
    keysReceived: u32,       // a fine probe awaiting a change: the key neighbours that passed it theirs (low 8 bits), their weight (high half)
    changeClock: u32,        // the volume's change clock at its last blended update
    keyDeltaSlot: u32,       // a key whose latest update passed its change: the update-list slot of its key-delta entry
    moverMask: u32,          // the hot movers its latest blended update's rays hit (T2.M1)
    awaitingSinceFrame: u32, // a fine probe awaiting a change: the frame ddgi/mark flagged it (T2.4d)
    spare1: u32,
    spare2: u32,
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
fn ddgiHysteresisStep(history: DdgiProbeHistory, state: u32, changeClock: u32) -> DdgiHysteresisStep {
    let flags = history.scheduleAndFlags & DDGI_HISTORY_FLAGS_MASK;
    // A probe in state 3 starts its history again at its first blended update
    // only; its second (warmedOnce) blends like any other (m3-moving.md §5.2).
    let firstUpdate = state == DDGI_STATE_NOT_UPDATED_YET && (flags & DDGI_FLAG_WARMED_ONCE) == 0u;
    let marked = (flags & DDGI_FLAG_MARKED) != 0u;
    let awaiting = (flags & DDGI_FLAG_AWAITING_CHANGE) != 0u;
    // The test mode: a fine probe waiting for its keys keeps its tiles
    // (ddgiBlend writes the old ones back), its change and its flags, so only
    // propagation moves it. It gives up its owed updates: held, it would stay
    // owed for ever and take priority slots from the keys whose propagation
    // is being measured.
    if (ddgi.holdFineProbes != 0u && awaiting && !firstUpdate) {
        var held = history;
        held.updates = history.updates & ~(DDGI_OWED_MASK << DDGI_OWED_SHIFT);
        held.urgency = 0.0;
        return DdgiHysteresisStep(1.0, held, DDGI_UPDATE_HELD);
    }
    let owed = (history.updates >> DDGI_OWED_SHIFT) & DDGI_OWED_MASK;
    // The cut is for the updates the probe was promised (ddgi/mark: K, or
    // fewer for far probes after a global change); K when none are owed.
    let promised = select(ddgi.changeUpdates, f32(owed), owed > 0u);
    // A fine probe that waited for its keys is cut only for what propagation
    // hasn't brought it (ddgiResidualChange).
    let change = select(history.pendingChange, ddgiResidualChange(history), awaiting);
    let limit = select(ddgiChangeLimit(change, flags, promised), DDGI_NO_HISTORY_LIMIT, firstUpdate);
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
    // Saturating at the field's width: the count only matters while k / a is below n_max.
    let nextCount = select(updates + 1u, updates, updates == DDGI_HISTORY_COUNT_MASK);
    // One owed update is taken off, a first update's too: a probe that came
    // back on keeps what it was owed (ddgiForgetPromises), so its second
    // warming update comes at priority as well. (A reset clears the record.)
    let nextOwed = owed - min(owed, 1u);
    let takingChange = (flags & DDGI_FLAG_TAKING_CHANGE) != 0u;
    // The cut holds for the updates a change promised (ddgiChangeLimit):
    // until the last of them the count doesn't grow back, so every one of
    // them replaces as much as the first. Then it grows by one an update,
    // as T1.12's rule has it.
    let holding = takingChange && nextOwed > 0u;
    let samples = select(min(remembered + 1.0, ddgi.maxHistorySamples), remembered, holding);
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
    // Traced itself: what its keys passed it is part of its history now. The
    // clock stamp says how much change it has seen (ddgi/propagate's share).
    next.keysReceived = 0u;
    next.changeClock = changeClock;
    // passedChange is set again below when this update passes its change on.
    // warmedOnce is cleared too: after its second update a state-3 probe is active.
    // waitingForKeys goes with the awaiting flag: traced itself, it waits for nothing.
    let keptFlags = flags & ~(DDGI_FLAG_MARKED | DDGI_FLAG_MARKED_BEFORE_LAST_UPDATE | DDGI_FLAG_AWAITING_CHANGE | DDGI_FLAG_TAKING_CHANGE | DDGI_FLAG_PASSED_CHANGE | DDGI_FLAG_WARMED_ONCE | DDGI_FLAG_WAITING_FOR_KEYS);
    let stillTaking = takingChange && nextOwed > 0u;
    let nextFlags = keptFlags | select(0u, DDGI_FLAG_MARKED_BEFORE_LAST_UPDATE, marked) | select(0u, DDGI_FLAG_TAKING_CHANGE, stillTaking);
    // A first update starts the flags again, with warmedOnce: one estimate so far.
    next.scheduleAndFlags = (history.scheduleAndFlags & ~DDGI_HISTORY_FLAGS_MASK) | select(nextFlags, DDGI_FLAG_WARMED_ONCE, firstUpdate);
    let kind = select(select(DDGI_UPDATE_QUIET, DDGI_UPDATE_MARKED, marked || takingChange), DDGI_UPDATE_FIRST, firstUpdate);
    return DdgiHysteresisStep(hysteresis, next, kind);
}

// The change a fine probe that waited for its keys is cut for at its own
// update (m2-lights.md §4.5): r × max(keyResidualShare, 1 − 0.8 × received),
// with 'received' the summed propagation weight of the keys that passed it
// their change since its last update (ddgi/propagate keeps it in the record).
// Why not the residual alone: a fine probe the round robin reaches before its
// keys are done has received little, and cutting it only for the residual
// would leave it lagging.
fn ddgiResidualChange(history: DdgiProbeHistory) -> f32 {
    let received = unpack2x16float(history.keysReceived >> DDGI_KEYS_WEIGHT_SHIFT).x;
    let share = max(ddgi.keyResidualShare, 1.0 - DDGI_KEY_FIRST_UPDATE_SHARE * saturate(received));
    return history.pendingChange * share;
}

// Whether this update of a probe passes its change on to its fine neighbours
// (m2-lights.md §4.5): a key probe, propagation on, its update taking in a
// reported change (owed updates since a mark), and not its first update
// (that starts from nothing, so new − old isn't a change of light).
fn ddgiPassesChange(probe: vec3i, history: DdgiProbeHistory, state: u32) -> bool {
    let takingChange = (history.scheduleAndFlags & DDGI_FLAG_TAKING_CHANGE) != 0u;
    return ddgi.propagation != 0u && takingChange && state != DDGI_STATE_NOT_UPDATED_YET && ddgiIsKeyProbe(probe);
}

// The most samples a probe may keep after the changes reported since its
// last update (docs/design/m2-lights.md §4.1, as revised in T2.4b). r is the
// relative change of its light that ddgi/mark added up (against the new
// light), epsilon the tolerance, K the updates it was promised (the
// scheduler serves them soon), rho the share of its light that comes back
// through the field (changeBounceShare):
// - a step: each held update (the count stays at this limit for the K
//   updates, ddgiHysteresisStep) leaves h of the error, plus the share of
//   the new estimate that bounces off the field's own error, (1 - h) rho:
//   the error shrinks by c = h + (1 - h) rho an update. Within epsilon after
//   K updates needs c^K r <= epsilon, so h = (c - rho) / (1 - rho) with
//   c = (epsilon / r)^(1 / K), and n = h / (1 - h); 0 when even h = 0 can't
//   do it (then the bounces set the pace). Nothing for r <= epsilon.
//   *Why not the design's n <= epsilon K / (r - epsilon):* that takes each
//   estimate as unbiased and lets the count grow back by one an update, so
//   the probe ends up a plain average of estimates that read the old light
//   through their bounces; their error shrinks only like k^-(1 - rho). On
//   two-room (every probe nearly every frame) the lamp halved was still 45%
//   of the step away after 30 frames (PROGRESS 2026-11-06, T2.4b). With
//   rho = 0 this rule gives h = (epsilon / r)^(1 / K), the unbiased case.
// - a change that keeps going (marked before its previous update too): it
//   moves the probe only a little between two updates, so the step rule
//   alone would leave the history long and the probe trailing the change by
//   about tau of it; n <= epsilon / r keeps the lag near epsilon.
// Symmetric on purpose: brighter or darker, the cut is by the size of the
// change against the new light, so a lamp switched on arrives as fast as one
// switched off.
fn ddgiChangeLimit(pendingChange: f32, flags: u32, promised: f32) -> f32 {
    if (ddgi.changeDrivenHistory == 0u || pendingChange <= 0.0) {
        return DDGI_NO_HISTORY_LIMIT;
    }
    let tolerance = ddgi.changeTolerance;
    var limit = DDGI_NO_HISTORY_LIMIT;
    if (pendingChange > tolerance) {
        let bounce = ddgi.changeBounceShare;
        let contraction = pow(tolerance / pendingChange, 1.0 / max(promised, 1.0));
        let hysteresis = max(0.0, (contraction - bounce) / (1.0 - bounce));
        limit = hysteresis / (1.0 - hysteresis);
    }
    let steady = (flags & DDGI_FLAG_MARKED) != 0u && (flags & DDGI_FLAG_MARKED_BEFORE_LAST_UPDATE) != 0u;
    if (steady) {
        limit = min(limit, tolerance / pendingChange);
    }
    return limit;
}

// Whether RTXGI's back-face rule is on (settings.backFaceRayThreshold; negative is off).
fn ddgiBackFaceRuleOn() -> bool {
    return ddgi.backFaceRayThreshold >= 0.0;
}

// RTXGI's test (ProbeBlendingCS.hlsl): with N blended rays and threshold t,
// its blend stops at the back face that brings the count to uint(N * t), so
// an update is skipped when at least that many of its blended rays hit back
// faces, and at the first back face when N * t is under 1. About 10% (t = 0.1)
// is more than a probe beside a surface sees: part of the probe's view is
// the inside of something.
fn ddgiTooManyBackFaces(backFaces: u32, blendedRays: u32) -> bool {
    if (!ddgiBackFaceRuleOn()) {
        return false;
    }
    let most = u32(f32(blendedRays) * ddgi.backFaceRayThreshold);
    return backFaces >= max(most, 1u);
}

// A probe's record after an update skipped for back faces: nothing was
// blended, so it is kept as it was (sample count, clock stamp, change, mean
// and noise), except that one owed update is taken off, with the urgency and
// the taking-a-change flag going with the last of them. Otherwise a probe that
// keeps skipping would stay at the front of the scheduler for ever.
fn ddgiSkippedHistory(history: DdgiProbeHistory) -> DdgiProbeHistory {
    let owed = (history.updates >> DDGI_OWED_SHIFT) & DDGI_OWED_MASK;
    let nextOwed = owed - min(owed, 1u);
    var next = history;
    next.updates = (history.updates & ~(DDGI_OWED_MASK << DDGI_OWED_SHIFT)) | (nextOwed << DDGI_OWED_SHIFT);
    next.urgency = select(0.0, history.urgency, nextOwed > 0u);
    if (nextOwed == 0u) {
        next.scheduleAndFlags = history.scheduleAndFlags & ~DDGI_FLAG_TAKING_CHANGE;
    }
    return next;
}

// The cosine-weighted mean radiance of the blended rays around one
// irradiance texel's direction in one layer, times pi: the texel's new
// estimate of E. With RTXGI's back-face rule on, rays that hit a back face
// are left out (neither radiance nor weight): they saw the inside of
// something, not light the probe should carry. With it off they count as
// black samples (the trace stores them black).
fn ddgiEstimateIrradianceTexel(slot: u32, layer: u32, texelIndex: u32, blendedRays: u32) -> vec3f {
    let texel = vec2u(texelIndex % DDGI_IRRADIANCE_TEXELS, texelIndex / DDGI_IRRADIANCE_TEXELS);
    let normal = ddgiTexelDirection(texel, DDGI_IRRADIANCE_TEXELS);
    let leaveOutBackFaces = ddgiBackFaceRuleOn();
    var radianceSum = vec3f(0.0);
    var weightSum = 0.0;
    for (var blended = 0u; blended < blendedRays; blended++) {
        let ray = DDGI_FIXED_RAYS + blended;
        if (leaveOutBackFaces && ddgiRayDirectionDistance[ray].w < 0.0) {
            continue;
        }
        let weight = max(0.0, dot(normal, ddgiRayDirectionDistance[ray].xyz));
        let packed = ${layered ? 'ddgiRays[ddgiRayResult(slot, ray, layer)]' : 'ddgiRayRadiance[ray]'};
        radianceSum += weight * vec3f(unpack2x16float(packed.x), unpack2x16float(packed.y).x);
        weightSum += weight;
    }
    // With rays spread evenly over the sphere, the cosine-weighted mean of the
    // radiance over the hemisphere is E / pi; the field stores E.
    return DDGI_PI * radianceSum / max(weightSum, 1.0e-6);
}

// One texel of irradiance layer 'layer' (its estimate and old value are in
// ddgiTexelEstimate and ddgiTexelOld). 'changedNow' is the big-change
// verdict, which only layer 0's alpha carries.
fn ddgiWriteIrradianceTexel(slot: u32, probeIndex: u32, state: u32, layer: u32, texelIndex: u32, probeHysteresis: f32, fastHysteresis: bool, changedNow: bool, passChange: bool) {
    let base = slot * DDGI_SCRATCH_STRIDE + DDGI_SCRATCH_IRRADIANCE + layer * DDGI_SCRATCH_IRRADIANCE_LAYER_WORDS + 2u * texelIndex;
    let item = layer * DDGI_IRRADIANCE_TEXEL_COUNT + texelIndex;
    if (ddgiUpdateKind == DDGI_UPDATE_HELD || ddgiUpdateKind == DDGI_UPDATE_SKIPPED) {
        // The test mode, or an update skipped for back faces: the old texel
        // as it is, alpha included (exact half floats pack exactly).
        let kept = ddgiTexelOld[item];
        ddgiScratch[base] = pack2x16float(kept.rg);
        ddgiScratch[base + 1u] = pack2x16float(kept.ba);
        return;
    }
    let estimate = ddgiTexelEstimate[item].rgb;
    let old = ddgiTexelOld[item].rgb;
    var hysteresis = probeHysteresis;
    if (fastHysteresis) {
        hysteresis *= DDGI_BIG_CHANGE_HYSTERESIS_SCALE;
    }
    var blendedValue = mix(estimate, old, hysteresis);
    // Not for a probe taking in a reported change (T2.4b): the change is
    // known and real, and the clamp would hold a probe that was dark when a
    // lamp came on near it to 3x a frame from almost nothing (two-room's lamp
    // switched on showed nothing for 7 frames).
    let takingChange = ddgiUpdateKind == DDGI_UPDATE_MARKED;
    if (state != DDGI_STATE_NOT_UPDATED_YET && !takingChange) {
        // Firefly clamp: one update may change a channel by at most
        // maxBrightnessChange times its old value, so one lucky ray can't
        // flash the probe.
        let limit = ddgi.maxBrightnessChange * max(old, vec3f(DDGI_SMALLEST_NORMAL_HALF));
        blendedValue = old + clamp(blendedValue - old, -limit, limit);
    }
    let seed = (probeIndex * DDGI_IRRADIANCE_TEXEL_COUNT + texelIndex) ^ ddgiLayerSeed(layer);
    ddgiScratch[base] = ddgiPackHalves(blendedValue.r, blendedValue.g, seed);
    // Alpha carries this update's big-change verdict to the next update (layer 0's).
    ddgiScratch[base + 1u] = ddgiPackHalves(blendedValue.b, select(0.0, 1.0, changedNow), seed ^ 0x5bd1e995u);
    if (passChange) {
        // The change this update made, for the key's fine neighbours (ddgi/propagate).
        let change = blendedValue - old;
        let deltaBase = slot * DDGI_KEY_DELTA_STRIDE + DDGI_KEY_DELTA_HEADER_WORDS + layer * DDGI_SCRATCH_IRRADIANCE_LAYER_WORDS + 2u * texelIndex;
        ddgiKeyDeltas[deltaBase] = pack2x16float(change.rg);
        ddgiKeyDeltas[deltaBase + 1u] = pack2x16float(vec2f(change.b, 0.0));
    }
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
    let old = textureLoad(ddgiDistanceAtlas, ddgiAtlasTexel(probe, texel, DDGI_DISTANCE_TEXELS), ddgiProbeLayer(probe), 0).xy;
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
//   doesn't depend on h), relative to the stored mean, as the root of a
//   running mean of its square, the channels' root mean square. Relative
//   because ray noise grows with the light: after a lamp makes a probe 10x
//   brighter, an absolute noise would read 10x too small until dozens of
//   quiet updates had passed, and the feedback rule would fire on every one
//   of them. Only quiet updates feed it: a reported change, or one still
//   being taken in, isn't noise. (Until T2.4b it was absolute, over 16
//   updates; only the feedback rule reads it.)
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
    // Relative to the stored mean (a dark probe's against DDGI_DARK_MEAN, so nothing divides by 0).
    let relativeDeviation = sqrt(dot(deviation, deviation) / 3.0) / max(sqrt(dot(oldMean, oldMean) / 3.0), DDGI_DARK_MEAN);
    // The updates blended before this one (the record already counts this one).
    let blendedBefore = (history.updates & DDGI_HISTORY_COUNT_MASK) - 1u;
    let fired = ddgiUpdateKind == DDGI_UPDATE_QUIET && ddgiFeedbackFires(relativeDeviation, previousNoise, blendedBefore);
    var noise = previousNoise;
    if (ddgiUpdateKind == DDGI_UPDATE_FIRST) {
        noise = 0.0;
    } else if (ddgiUpdateKind == DDGI_UPDATE_QUIET) {
        // Every quiet update feeds it, those that trip the rule too: left out,
        // the tail the noise has would be cut off and the noise read low.
        // The first quiet update starts the average at its own value.
        let squared = relativeDeviation * relativeDeviation;
        let averaged = select(mix(previousNoise * previousNoise, squared, 1.0 / DDGI_NOISE_UPDATES), squared, previousNoise <= 0.0);
        noise = sqrt(averaged);
    }
    history.meanRedGreen = pack2x16float(mean.rg);
    history.meanBlueNoise = pack2x16float(vec2f(mean.b, noise));
    ddgiProbeHistory[probeIndex] = history;
    atomicAdd(&ddgiCensus[DDGI_CENSUS_PROBES_BLENDED_TOTAL], 1u);
    if (fired) {
        ddgiRequestFeedback(probeIndex, probe, relativeDeviation);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_FEEDBACK_FIRED], 1u);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_FEEDBACK_FIRED_TOTAL], 1u);
    }
}

// The feedback rule's test (m2-lights.md §4.6) for a quiet update: the
// estimate lands more than FEEDBACK_NOISE_MULTIPLE times the probe's noise
// from the stored value, and more than 2% of it (both relative to the stored
// mean). Not during a probe's warm-up ramp (its first rampUpdates updates are
// expected to move while bounce light fills in, and its noise isn't averaged
// yet), nor before it has a noise at all.
fn ddgiFeedbackFires(relativeDeviation: f32, noise: f32, blendedBefore: u32) -> bool {
    if (ddgi.feedback == 0u || !(noise > 0.0) || f32(blendedBefore) < ddgi.rampUpdates) {
        return false;
    }
    return relativeDeviation > DDGI_FEEDBACK_NOISE_MULTIPLE * noise && relativeDeviation > DDGI_FEEDBACK_MIN_SHARE;
}

// One more update for the probe at this urgency, and for its active face
// neighbours at half of it. In a buffer of its own: the blend never writes
// another probe's history (each workgroup owns only its own probe).
// Non-negative floats order like their bits, so atomicMax keeps the largest.
// Neighbours are found in window probes (a camera-following volume's wrap
// around its slots) and must be inside the window.
fn ddgiRequestFeedback(probeIndex: u32, probe: vec3i, urgency: f32) {
    atomicMax(&ddgiFeedbackRequests[probeIndex], bitcast<u32>(urgency));
    let counts = vec3i(ddgiProbeCount());
    let windowProbe = ddgiWindowProbe(ddgiUniformVolume(), probe);
    let neighbourUrgency = bitcast<u32>(urgency * DDGI_FEEDBACK_NEIGHBOUR_SHARE);
    for (var face = 0u; face < 6u; face++) {
        var step = vec3i(0);
        step[face % 3u] = select(1, -1, face >= 3u);
        let windowNeighbour = windowProbe + step;
        if (any(windowNeighbour < vec3i(0)) || any(windowNeighbour >= counts)) {
            continue;
        }
        let neighbour = ddgiNeighbourSlot(probe, step);
        if (ddgiReadProbeData(ddgiProbeData, ddgiUniformVolume(), neighbour).state != DDGI_STATE_ACTIVE) {
            continue;
        }
        atomicMax(&ddgiFeedbackRequests[ddgiProbeIndex(neighbour)], neighbourUrgency);
    }
}

// A listed probe that isn't blended this update. Staying off (or switched
// off by it), it owes nothing: off probes trace only their fixed rays and
// take no light, and a promise left in its record would keep it at priority
// for ever; its change is dropped too, so marks near a dragged object can't
// pile up and owe it again every frame. Coming back on (new state 3, after
// a change freed it, m3-moving.md §5.2), it keeps what it was owed, so its
// two warming updates come at once. Either way it starts warming again.
// A wake that has run out is cleared ('awake', ddgiAwake).
fn ddgiForgetPromises(probeIndex: u32, newState: u32, awake: bool) {
    var history = ddgiProbeHistory[probeIndex];
    history.scheduleAndFlags = history.scheduleAndFlags & ~DDGI_FLAG_WARMED_ONCE;
    history.wakeUntil = select(0u, history.wakeUntil, awake);
    if (newState != DDGI_STATE_NOT_UPDATED_YET) {
        history.updates = history.updates & ~(DDGI_OWED_MASK << DDGI_OWED_SHIFT);
        history.urgency = 0.0;
        history.pendingChange = 0.0;
        history.scheduleAndFlags = history.scheduleAndFlags & ~DDGI_FLAG_TAKING_CHANGE;
    }
    ddgiProbeHistory[probeIndex] = history;
}

// The mover bits a blended ray's hit point falls in (the file comment): the
// point rebuilt from the probe's position, the ray's direction and its stored
// distance (a back face's true distance from its shortened one), against
// each hot mover's box, grown for the distance's rounding.
fn ddgiMoverBits(origin: vec3f, directionDistance: vec4f) -> u32 {
    let count = ddgiMovers.count;
    let signedDistance = directionDistance.w;
    let distance = select(signedDistance, -signedDistance / DDGI_BACK_FACE_DISTANCE_SHARE, signedDistance < 0.0);
    if (count == 0u || distance >= DDGI_MOVER_MISS_DISTANCE) {
        return 0u;
    }
    let point = origin + directionDistance.xyz * distance;
    let margin = vec3f(DDGI_MOVER_BOX_MARGIN + DDGI_MOVER_BOX_MARGIN_SHARE * distance);
    var bits = 0u;
    for (var mover = 0u; mover < count; mover++) {
        let box = ddgiMovers.boxes[mover];
        if (all(point >= box.min - margin) && all(point <= box.max + margin)) {
            bits |= box.bit;
        }
    }
    return bits;
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
    let probeData = ddgiReadProbeData(ddgiProbeData, ddgiUniformVolume(), probe);
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
        atomicStore(&ddgiMoverHits, 0u);
        atomicStore(&ddgiBlendedBackFaces, 0u);
        atomicAdd(&ddgiCensus[DDGI_CENSUS_RAYS_TRACED], tracedRays);
    }
    let origin = ddgiProbePosition(probe, probeData.offset);
    var moverBits = 0u;
    var blendedBackFaces = 0u;
    for (var rayIndex = thread; rayIndex < tracedRays; rayIndex += DDGI_BLEND_WORKGROUP_SIZE) {
        // Layer 0's result, which carries the hit distance (and is the only one with groups off).
        let record = ddgiRays[ddgiRayResult(slot, rayIndex, 0u)];
        let signedDistance = unpack2x16float(record.y).y;
        let directionDistance = vec4f(ddgiRayDirection(rayIndex), signedDistance);
        ddgiRayDirectionDistance[rayIndex] = directionDistance;
${layered ? '' : '        ddgiRayRadiance[rayIndex] = record;'}
        // Blended rays only: the mask says what the probe's light came from.
        if (rayIndex >= DDGI_FIXED_RAYS) {
            moverBits |= ddgiMoverBits(origin, directionDistance);
            blendedBackFaces += select(0u, 1u, signedDistance < 0.0);
        }
    }
    workgroupBarrier();

    // ---- Phase B ----
    ddgiReduceFixedRay(ddgiFixedRayResult(thread, probe, probeData.offset));
    // The mover hits join after the reduction (which needs every thread in step, for its subgroup
    // operations), and after the first barrier, so thread 0 has cleared them first.
    if (moverBits != 0u) {
        atomicOr(&ddgiMoverHits, moverBits);
    }
    // The blended back faces join the same way (thread 0 cleared them before the first barrier).
    if (blendedBackFaces != 0u) {
        atomicAdd(&ddgiBlendedBackFaces, blendedBackFaces);
    }
    workgroupBarrier();
    if (thread == 0u) {
        // Woken near a receiver or a moved object (ddgi/mark, T3.4): kept on whatever its rays find around it.
        let awake = ddgiAwake(ddgiProbeHistory[probeIndex].wakeUntil);
        let decision = ddgiDecide(probe, probeData.offset, state, awake);
        // Phase C runs for probes that traced all their rays and stay active.
        // (The write kernel copies the tiles exactly when the new state is active.)
        let blendThisUpdate = decision.state == DDGI_STATE_ACTIVE && !ddgiIsOff(state);
        ddgiBlendThisUpdate = select(0u, 1u, blendThisUpdate);
        ddgiPassChange = 0u;
        ddgiChangedNow = 0u;
        // The previous update's big-change verdict (layer 0's alpha), read once here:
        // phase C blends every layer's texels with it at once.
        ddgiFastHysteresis = select(0u, 1u, textureLoad(ddgiIrradianceAtlas, ddgiIrradianceAtlasTexel(probe, vec2u(0u), 0u), ddgiProbeLayer(probe), 0).a > 0.5);
        var newState = decision.state;
        if (blendThisUpdate) {
            // One workgroup per probe in the list, and a probe is listed at
            // most once a frame, so this read and write can't race.
            let history = ddgiProbeHistory[probeIndex];
            // Its first estimate since state 3: blended, but not yet used in shading (the file comment).
            if (state == DDGI_STATE_NOT_UPDATED_YET && (history.scheduleAndFlags & DDGI_FLAG_WARMED_ONCE) == 0u) {
                newState = DDGI_STATE_NOT_UPDATED_YET;
            }
            let changeClock = atomicLoad(&ddgiCensus[DDGI_CENSUS_CHANGE_CLOCK]);
            let step = ddgiHysteresisStep(history, state, changeClock);
            // RTXGI's back-face rule (the file comment): with too many
            // blended back faces the irradiance isn't blended. The distances
            // still blend at this update's hysteresis.
            let skipped = step.kind != DDGI_UPDATE_HELD && ddgiTooManyBackFaces(atomicLoad(&ddgiBlendedBackFaces), blendedRays);
            ddgiUpdateHysteresis = step.hysteresis;
            ddgiNextHistory = step.history;
            if (skipped) {
                ddgiNextHistory = ddgiSkippedHistory(history);
            }
            ddgiNextHistory.moverMask = atomicLoad(&ddgiMoverHits);
            // A wake that has run out is cleared, so the stored frame never comes round again as the counter wraps.
            ddgiNextHistory.wakeUntil = select(0u, history.wakeUntil, awake);
            ddgiUpdateKind = select(step.kind, DDGI_UPDATE_SKIPPED, skipped);
            if (step.kind == DDGI_UPDATE_HELD) {
                // Its record without the owed updates; nothing else of it changes (ddgiWriteHistory is skipped).
                ddgiProbeHistory[probeIndex] = step.history;
            }
            if (skipped) {
                // Nothing blended, so a probe in state 3 stays there: it is
                // used in shading only after two blended updates.
                if (state == DDGI_STATE_NOT_UPDATED_YET) {
                    newState = DDGI_STATE_NOT_UPDATED_YET;
                }
                // ddgiWriteHistory is skipped too, so the record is written here.
                ddgiProbeHistory[probeIndex] = ddgiNextHistory;
                atomicAdd(&ddgiCensus[DDGI_CENSUS_BACK_FACE_SKIPS_TOTAL], 1u);
            }
            if (step.kind != DDGI_UPDATE_HELD && !skipped && ddgiPassesChange(probe, history, state)) {
                // The header of this slot's key-delta entry, and the flag
                // ddgi/propagate finds it by (valid in this frame, the frame
                // of the probe's scheduled stamp).
                ddgiPassChange = 1u;
                let deltaBase = slot * DDGI_KEY_DELTA_STRIDE;
                ddgiKeyDeltas[deltaBase + DDGI_KEY_DELTA_PROBE_INDEX] = probeIndex;
                ddgiKeyDeltas[deltaBase + DDGI_KEY_DELTA_CLOCK_BEFORE] = history.changeClock;
                ddgiNextHistory.scheduleAndFlags = ddgiNextHistory.scheduleAndFlags | DDGI_FLAG_PASSED_CHANGE;
                ddgiNextHistory.keyDeltaSlot = slot;
                atomicAdd(&ddgiCensus[DDGI_CENSUS_KEY_CHANGES_PASSED], 1u);
                atomicAdd(&ddgiCensus[DDGI_CENSUS_KEY_CHANGES_PASSED_TOTAL], 1u);
            }
        } else {
            ddgiForgetPromises(probeIndex, decision.state, awake);
        }
        // Offsets and states are small exact values in half floats; no rounding noise wanted.
        let base = slot * DDGI_SCRATCH_STRIDE + DDGI_SCRATCH_PROBE_DATA;
        ddgiScratch[base + DDGI_SCRATCH_OFFSET_XY] = pack2x16float(decision.offset.xy);
        ddgiScratch[base + DDGI_SCRATCH_OFFSET_Z_AND_STATE] = pack2x16float(vec2f(decision.offset.z, f32(newState)));
        // ddgi/write copies the tiles exactly when this update wrote them.
        ddgiScratch[base + DDGI_SCRATCH_TILES_WRITTEN] = select(0u, 1u, blendThisUpdate);
    }

    if (workgroupUniformLoad(&ddgiBlendThisUpdate) == 0u) {
        return;
    }
    let hysteresis = workgroupUniformLoad(&ddgiUpdateHysteresis);
    let fastHysteresis = workgroupUniformLoad(&ddgiFastHysteresis) != 0u;
    let passChange = workgroupUniformLoad(&ddgiPassChange) != 0u;

    // ---- Phase C ----
    // Step 1: every layer's irradiance texels (one thread a layer and texel)
    // and the distance texels, side by side: each texel's estimate and old value.
    let layerTexels = DDGI_IRRADIANCE_TEXEL_COUNT * DDGI_IRRADIANCE_LAYERS;
    for (var item = thread; item < layerTexels + DDGI_DISTANCE_TEXEL_COUNT; item += DDGI_BLEND_WORKGROUP_SIZE) {
        if (item < layerTexels) {
            let layer = item / DDGI_IRRADIANCE_TEXEL_COUNT;
            let texelIndex = item % DDGI_IRRADIANCE_TEXEL_COUNT;
            let texel = vec2u(texelIndex % DDGI_IRRADIANCE_TEXELS, texelIndex / DDGI_IRRADIANCE_TEXELS);
            ddgiTexelEstimate[item] = vec4f(ddgiEstimateIrradianceTexel(slot, layer, texelIndex, blendedRays), 0.0);
            ddgiTexelOld[item] = textureLoad(ddgiIrradianceAtlas, ddgiIrradianceAtlasTexel(probe, texel, layer), ddgiProbeLayer(probe), 0);
        } else {
            ddgiBlendDistanceTexel(slot, probeIndex, probe, item - layerTexels, blendedRays, hysteresis);
        }
    }
    workgroupBarrier();

    // Step 2: the big-change test, opt-in (irradianceChangeThreshold; a
    // negative value means off, the M1 default, DECISIONS 2026-11-05). It
    // looks at the probe's mean irradiance against its old mean (over every
    // layer, the probe's whole light), not per texel, and its verdict sets the
    // hysteresis of the probe's *next* update, not this one.
    // Why not this one: a test on this update's estimate is tripped by its own
    // ray noise, and the noise is skewed (a few rays find a small bright spot),
    // so the updates that trip are mostly the bright ones, and they get the
    // lower hysteresis: about 9x the weight. That biased the field upwards
    // (per texel: in-air Cornell +8.3%; on the probe mean as well: +21.7%;
    // with no test: +3.3%, T1.5b). The next update's rays are independent of
    // this one's noise, so it blends without bias, and a real light switch
    // still trips the test again and gets the fast hysteresis one update late.
    // The verdict travels in layer 0's irradiance alpha, which nothing samples
    // (thread 0 read it in phase B).
    if (thread == 0u) {
        var estimateSum = vec3f(0.0);
        var oldSum = vec3f(0.0);${meanSum}
        // The same share applies to the sums as to the means.
        let ruleOn = ddgi.irradianceChangeThreshold >= 0.0;
        ddgiChangedNow = select(0u, 1u, ruleOn && any(abs(estimateSum - oldSum) > ddgi.irradianceChangeThreshold * oldSum));
        // A held probe (the test mode) keeps its record as it is; a skipped one's was written in phase B.
        if (ddgiUpdateKind != DDGI_UPDATE_HELD && ddgiUpdateKind != DDGI_UPDATE_SKIPPED) {
            ddgiWriteHistory(probeIndex, probe, estimateSum, oldSum, hysteresis, fastHysteresis);
        }
    }
    let changedNow = workgroupUniformLoad(&ddgiChangedNow) != 0u;

    // Step 3: blend every irradiance texel of every layer with the probe's hysteresis.
    for (var item = thread; item < layerTexels; item += DDGI_BLEND_WORKGROUP_SIZE) {
        let layer = item / DDGI_IRRADIANCE_TEXEL_COUNT;
        ddgiWriteIrradianceTexel(slot, probeIndex, state, layer, item % DDGI_IRRADIANCE_TEXEL_COUNT, hysteresis, fastHysteresis, changedNow && layer == 0u, passChange);
    }
}
`;
}
