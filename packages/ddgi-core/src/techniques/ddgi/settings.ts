/**
 * DDGI settings (docs/design/ddgi.md, "Settings"), their defaults, and the
 * volume they describe.
 *
 * Settings are a live object, as the technique contract asks: the host edits
 * fields and the next `encode` reads them. Seven of them need new GPU
 * resources and only take effect at the next `prepare`: `volume`, `volumes`,
 * `irradianceTexels`, `distanceTexels`, `maxProbesPerFrame`,
 * `lightGroups.count` (its weights are live) and `sunLayer`.
 */

import type { Vec3 } from '../../platform/contract/math.ts';
import type { DDGILightGroups } from './light-groups.ts';

export interface DDGIVolumeSettings {
    /** Centre of the box, metres. */
    readonly center: Vec3;
    /** Size of the box, metres. */
    readonly size: Vec3;
    /** Distance between neighbouring probes, metres: one number, or one per axis. */
    readonly spacing: number | Vec3;
    /**
     * With several volumes (`DDGISettings.volumes`): which one answers where
     * they overlap. Higher first, default 0; equal priorities go in the
     * list's order. A dense volume inside a coarse one gets the higher
     * priority, and the coarse one fills in across its fade band
     * (docs/design/m3-moving.md §4.1, volumes.ts).
     */
    readonly priority?: number;
    /**
     * `fixed` (the default): the probes stay where `center` and `size` put
     * them. `camera`: the box is a window of probes that follows the camera
     * (docs/design/m3-moving.md §4.2, scroll.ts). Its probes sit on a lattice
     * fixed in world space, through the box's probes at `spacing` steps; the
     * window starts where `center` and `size` put it, keeps its size, and
     * moves by whole planes of probes to stay centred on the camera, reusing
     * the planes it leaves behind for the planes entering ahead. Needs at
     * least 4 probes on every axis.
     */
    readonly follow?: 'fixed' | 'camera';
}

export interface DDGISettings {
    /**
     * The box of probes. New `prepare` when changed. With `volumes` given,
     * the first of them (the technique reads `volumes`).
     */
    volume: DDGIVolumeSettings;
    /**
     * Several boxes of probes in one technique (M3; at most 8, volumes.ts):
     * they share one set of atlases and blend by `priority` where they
     * overlap. When given, these are the volumes and `volume` is the first of
     * them; when not, `volume` is the one volume. New `prepare` when changed.
     */
    volumes?: readonly DDGIVolumeSettings[];
    /** Rays per probe and update: a multiple of 32, from 64 to 512. The first 32 are the fixed rays. */
    raysPerProbe: number;
    /** Interior texels per side of a probe's irradiance tile. New `prepare` when changed. */
    irradianceTexels: number;
    /** Interior texels per side of a probe's distance tile. New `prepare` when changed. */
    distanceTexels: number;
    /**
     * How much of a probe's old value each update keeps (docs/design/ddgi.md,
     * "Hysteresis that knows the time between updates"). On (the default):
     * per probe, from how many updates it remembers and how long ago they
     * were, so a probe updated often settles quietly and one updated rarely
     * still follows changes within seconds. Off: every update keeps
     * {@link DDGISettings.hysteresis} (the rule before T1.12).
     */
    adaptiveHysteresis: boolean;
    /** With `adaptiveHysteresis` off: the share of the old value every update keeps, 0..1. Not used when it is on. */
    hysteresis: number;
    /**
     * The field's memory time τ in seconds, on the technique's clock (the sum
     * of `deltaTime`): an update made this long ago counts e⁻¹ as much as a new
     * one. A probe updated every Δ seconds settles at a hysteresis of about
     * e^(−Δ/τ). `Infinity` never forgets (a plain running average up to
     * `hysteresisMaxSamples`).
     */
    hysteresisMemorySeconds: number;
    /**
     * The most updates a probe's history may count as, n_max: its hysteresis
     * never goes above n_max / (n_max + 1). 200 holds at most 0.995.
     */
    hysteresisMaxSamples: number;
    /**
     * The ramp, a: during a probe's first `hysteresisRampUpdates` blended
     * updates (since it was new, reset or back on), its history counts as at
     * most one update per `a`. Its first estimates read a field still filling
     * with bounce light and come out too dark; the ramp weights later updates
     * more, so they fade fast (docs/design/ddgi.md). 1 is a plain running
     * average.
     */
    hysteresisUpdatesPerSample: number;
    /**
     * How long the ramp lasts, K: a probe's first K blended updates. After
     * them its count grows by one per update again. `Infinity` keeps the
     * ramp until the count reaches `hysteresisMaxSamples`.
     */
    hysteresisRampUpdates: number;
    /**
     * Opt-in, off by default (Infinity): when a probe's mean irradiance moves
     * by more than this share of its old mean, its next update uses a lower
     * hysteresis. Off in M1 because any test on noisy rays selects by the
     * noise and biases the field (DECISIONS 2026-11-05); M2 lowers hysteresis
     * where SceneChanges says a light changed instead.
     */
    irradianceChangeThreshold: number;
    /**
     * Change-driven history (docs/design/m2-lights.md §4.1, §4.2). On (the
     * default): when the scene reports a light change, `ddgi/mark` works out
     * from the light's own numbers how much the light at each probe changed,
     * and the probe's next update cuts its history to match, so the change
     * shows within a few updates instead of fading in over the memory time.
     * A still scene is never touched. Off: the M1 rule alone (T1.12).
     */
    changeDrivenHistory: boolean;
    /**
     * ε, the tolerance a changed probe's history is cut for: after
     * {@link changeUpdates} updates it is within this share of its new light
     * (SPEC's "within 5% of settled"). Changes at or under it cut nothing.
     */
    changeTolerance: number;
    /** K, the updates a changed probe is expected to get soon after its change (the GPU scheduler, T2.4b, keeps the promise). */
    changeUpdates: number;
    /**
     * ρ, the share of a probe's light taken to come back through the field
     * (bounces), 0..0.95, for how deep a change cuts the history (T2.4b,
     * docs/design/ddgi.md "Change-driven history"). An update's estimate
     * bounces off a field that still holds the old light, so its error shrinks
     * only by h + (1 − h)ρ an update; the cut is solved with it so the probe
     * is within {@link changeTolerance} after its {@link changeUpdates}
     * promised updates. Higher: deeper cuts, noisier for a few updates; 0
     * takes estimates as unbiased (the design's first rule).
     */
    changeBounceShare: number;
    /**
     * How much of a light's change at a probe the probe is taken to see
     * (m2-lights.md §4.2): a probe sees a light only through the surfaces it
     * lights, so its own change is a share of the light's. Tuned in T2.4a
     * against probes that really changed ({@link DEFAULT_LIGHT_CHANGE_SCALE}).
     */
    lightChangeScale: number;
    /**
     * The share of each frame's probes the GPU scheduler gives to probes that
     * are owed updates after a change, most urgent and nearest the camera
     * first (docs/design/m2-lights.md §4.3). The rest always goes to the
     * round robin through every probe, so no probe is starved. 0..1.
     */
    priorityShare: number;
    /**
     * After a global change (the sun, the sky) on a volume too big to give
     * every probe {@link changeUpdates} updates within
     * {@link FAR_RULE_FRAMES} frames, probes away from the camera's view
     * (camera weight under {@link NEAR_CAMERA_WEIGHT}) are owed this many
     * updates instead, and their history is cut for that many (m2-lights.md
     * §6.4, "far probes owe 2"). They are noisier for a while, and their
     * history grows back with the round robin.
     */
    farProbeUpdates: number;
    /**
     * Key probes and propagation (T2.4c, m2-lights.md §4.5), the main path
     * for big changes. On (the default): on a volume too big to give every
     * changed probe {@link changeUpdates} updates within
     * {@link FAR_RULE_FRAMES} frames, after a change one probe in 8 (the key
     * probes, every even grid index) goes first, and each update of a key
     * passes the change it made on to the fine probes around it, which show
     * it at once and are traced themselves later at a lower priority. Off:
     * "far probes owe 2" ({@link farProbeUpdates}), the fallback.
     */
    keyProbes: boolean;
    /** A marked key probe's urgency is its change times this (m2-lights.md §4.5), so keys come before fine probes away from the view. */
    keyProbeBoost: number;
    /**
     * A marked fine probe away from the view's core (camera weight under
     * {@link NEAR_CAMERA_WEIGHT}) waits for its keys: its urgency is its
     * change times this, and its own update is cut for at least this share
     * of the change (what propagation is taken not to bring).
     */
    keyResidualShare: number;
    /** Updates a key probe away from the view's core owes after a change (near probes owe {@link changeUpdates}). */
    keyProbeUpdates: number;
    /**
     * A test mode, off by default (T2.4c's checks of propagation alone):
     * fine probes waiting for their keys' change are traced but not blended,
     * so only propagation moves them. Never on in an application: those
     * probes would keep their old light until it is switched off.
     */
    holdFineProbes: boolean;
    /**
     * The feedback rule (m2-lights.md §4.6). On (the default): a probe whose
     * own update lands further from its stored value than its noise allows,
     * with no change reported, asks for one more update soon, and so do its 6
     * neighbours. It only schedules; it never changes how an update blends.
     * Catches changes nobody reported (an emissive texture animated on the
     * GPU, a material edit).
     */
    feedback: boolean;
    /**
     * The warm-up (moved from the viewer into the technique, T2.4b): after
     * `prepare` and after `resetHistory`, once the ray-tracing trees are
     * built, this many passes over every probe, as many probes a frame as
     * {@link maxProbesPerFrame} allows, whatever the frame's budget, with no
     * time between them for the history. What a game does behind its loading
     * screen: at 2 ms a frame Sponza's 19,425 probes would take a minute to
     * fill from black. 0 switches it off; GI then fills in at the budget.
     */
    warmUpPasses: number;
    /** The warm-up stops early after this many seconds of warming... */
    warmUpMaxSeconds: number;
    /** ...but not before every probe was updated this many times, so no probe is left black. */
    warmUpMinPasses: number;
    /** Largest change one update may make, as a multiple of the old value (fireflies). */
    maxBrightnessChange: number;
    /** Exponent of the cosine weight when blending distances. */
    distanceSharpness: number;
    /** How far a sample point moves along its normal before probes are looked up, metres. `auto` = 0.1 × the smallest spacing. */
    normalBias: number | 'auto';
    /** How far a sample point moves back towards the viewer, metres. `auto` = 0.1 × the smallest spacing. */
    viewBias: number | 'auto';
    /** Rays stop here, metres. `Infinity` lets a miss see the sky. */
    maxRayDistance: number;
    /** Scale on bounce light. 1 is physical; native RTXGI uses 0.95, set it only to match native. */
    bounceEnergy: number;
    /** Move probes out of geometry (T1.5b). */
    relocation: boolean;
    /** Closest front face a probe may have before relocation moves it, metres. `auto` = 0.02 × the smallest spacing. */
    minFrontFaceDistance: number | 'auto';
    /** Switch off probes inside geometry or far from any surface (T1.5b). */
    classification: boolean;
    /** Largest number of probes updated in one frame: sizes the update list and the scratch buffer. New `prepare` when changed. */
    maxProbesPerFrame: number;
    /**
     * Light groups (T2.5, light-groups.ts; docs/design/m2-lights.md §5): one
     * irradiance layer per group plus one for the sky and emission, added up
     * with live weights, so dimming or recolouring a whole group shows in the
     * same frame with no update lag. Off (null, the default): one layer.
     * Opt-in because each layer costs memory and its share of every probe
     * update. Measured on the RTX 3060 (T2.5): Sponza's DDGI is 58 MB with
     * groups off, 119 MB with 2 groups, 170 MB with 4 (+112 MB: about 25.5 MB
     * a layer, mostly ray results and scratch, plus the composed atlas), and
     * an update per probe costs 1.05× (2 groups) and 1.10× (4) at 512 probes;
     * two-room 1.04× and 1.23×. `count` needs a new `prepare`; the weights are
     * read every frame.
     */
    lightGroups: DDGILightGroups | null;
    /**
     * The sun layer (T2.6b, light-groups.ts; docs/design/m2-lights.md §6.1):
     * the sun, the scene's first directional light, traced at a white
     * reference in an irradiance layer of its own and weighed by its colour ×
     * intensity when the layers are added up. A change of the sun's colour or
     * brightness then shows exactly in the frame it happens, with nothing
     * marked and nothing traced again; a change of its direction still
     * re-traces. Works with or without light groups (at most 4 groups, the
     * environment and the sun). Off by default: like a light group it costs a
     * layer's memory and its share of every probe update (PROGRESS, T2.6b).
     * Needs direct light deferred (the default). New `prepare` when changed.
     */
    sunLayer: boolean;
    /**
     * The fewest probes updated per frame under a millisecond budget, so the
     * field keeps converging on every device (budget v2). It wins over the
     * millisecond ceiling; `budget.units` still caps it. `auto` = enough for a
     * full sweep of every probe within {@link FLOOR_SWEEP_SECONDS} at the
     * host's frame rate, but never more than {@link FLOOR_MAX_CALIBRATED_SHARE}×
     * the batch the budget buys with the clocks up, so a weak GPU slows the
     * sweep rather than the frame. A number replaces it (0 turns it off).
     */
    minProbesPerFrame: number | 'auto';
}

/**
 * `hysteresisMemorySeconds` by default, τ = 5 s (T1.12, measured on the RTX
 * 3060 at 60 Hz; docs/design/ddgi.md has the table):
 * - Cornell, every probe each frame: any τ above about 3.4 s reaches the
 *   n_max cap, so 5 and 10 are equally quiet (per-pixel change 0.095%
 *   against 0.56% at the old fixed 0.97); 2 s stops short (0.155%).
 * - Sponza, 291 probes a frame (a sweep every 1.1 s): after its sun halves,
 *   the field is within 5% / 1% of the new answer in 13 / 22.5 s at 5 s,
 *   25.5 / 44 s at 10 s, 91 / 162.5 s at the old 0.97. The cost is
 *   steadiness on sparsely updated probes: Sponza's per-pixel change 0.027%
 *   at 5 s, 0.0145% at 10 s, 0.0038% at 0.97 (all far under Cornell's).
 * 5 s keeps Sponza responsive within seconds at half the noise of 2 s.
 */
export const ADAPTIVE_MEMORY_SECONDS = 5;

/**
 * `hysteresisMaxSamples` by default, n_max = 200: the hold the flicker
 * decision asked for, h ≤ 200/201 ≈ 0.995 (DECISIONS, Cornell's flicker).
 * 100 (h ≤ 0.99) was measured too: Cornell 0.185% instead of 0.095%, for
 * bounce light that settles in half the frames.
 */
export const ADAPTIVE_MAX_SAMPLES = 200;

/**
 * The ramp by default: a = 8 for a probe's first K = 16 updates (T1.12, RTX
 * 3060; docs/design/ddgi.md has the table). Cornell right after the 16-pass
 * warm-up: -26% with no ramp, -10.5% at a = 3 and -5 to -7% at a = 8, then
 * within 1% after about 1 s instead of 19 s. Kept to the first 16 updates
 * because an endless a = 8 reaches n_max only after 1600 updates: 400 frames
 * after the warm-up Cornell still changed 0.28% a frame against 0.089% with
 * K = 16 (the settled 0.094%). 16 is the warm-up's own pass count; a sparse
 * probe (Sponza) spends its first 16 updates, about 18 s after a restart,
 * slightly noisier.
 */
export const ADAPTIVE_UPDATES_PER_SAMPLE = 8;

/** `hysteresisRampUpdates` by default, K (see ADAPTIVE_UPDATES_PER_SAMPLE). */
export const ADAPTIVE_RAMP_UPDATES = 16;

/** `changeTolerance` by default: SPEC's "within 5% of settled" (m2-lights.md, ε). */
export const DEFAULT_CHANGE_TOLERANCE = 0.05;
/** `changeUpdates` by default, K = 8 (m2-lights.md §4.1: at r = 1 the first update keeps 0.30 of the old value, 5% is left after 8). */
export const DEFAULT_CHANGE_UPDATES = 8;
/**
 * `changeBounceShare` by default, ρ = 0.8 (T2.4b, DECISIONS 2026-11-06):
 * two-room's, measured: after its lamp was halved, with nearly every probe
 * updated at h ≈ 0 each frame (880 of 1,071), the view's error shrank by
 * 0.84 a frame, so ρ ≈ (0.84 − 0.18) / 0.82 = 0.80. Cornell's fit is 0.65
 * (T1.12); open scenes are lower, and get cuts somewhat deeper than they
 * need. With the design's rule (ρ taken as 0, the count growing back by one
 * an update) the halving was still 45% of the step away after 30 frames.
 */
export const DEFAULT_CHANGE_BOUNCE_SHARE = 0.8;
/**
 * `lightChangeScale` by default: 0.5, not the design's first guess of 0.25
 * (T2.4a, RTX 3060, `ddgi-change-step`: two-room's lamp moved 2 m, every
 * probe every frame; DECISIONS 2026-11-06). Measured against the two
 * converged fields, a probe's true change was 0.48 × the mark's unscaled
 * value at the median (quartiles 0.25-0.73), so 0.5 makes r about the true
 * change. The scale sets how far a probe's history is cut, not only whether:
 * at 0.25 every probe that changed by over 5% was still cut (lit-room recall
 * 99.8%), but by too little, and the view took 0.73 s to get within 5% (the
 * old lamp's bounce light lingered in probes cut too gently); at 0.5 it took
 * 2 frames; at 1 also 2 frames, with deeper cuts (more noise right after a
 * change, the design's risk 2; not measured).
 */
export const DEFAULT_LIGHT_CHANGE_SCALE = 0.5;

/**
 * `priorityShare` by default (m2-lights.md §4.3): three quarters of a frame's
 * probes go to probes owed updates, a quarter always to the round robin.
 */
export const DEFAULT_PRIORITY_SHARE = 0.75;

/** `farProbeUpdates` by default: two updates bring a far probe within 5% under the cut (m2-lights.md §6.4). */
export const DEFAULT_FAR_PROBE_UPDATES = 2;

/** `keyProbeBoost` by default (m2-lights.md §4.5): a key 10 spacings away scores about 0.03 × 4 = 0.14, a probe in view at least 0.5, so near probes still come first. */
export const DEFAULT_KEY_PROBE_BOOST = 4;
/** `keyResidualShare` by default (m2-lights.md §4.5). */
export const DEFAULT_KEY_RESIDUAL_SHARE = 0.3;
/**
 * `keyProbeUpdates` by default (m2-lights.md §4.5): every key out of the
 * view's core owes 4 after a change, on a volume whose keys couldn't all get
 * the full count within FAR_RULE_FRAMES (Sponza: 2,717 keys). Where they can
 * (the furnished room: 792 keys at 428 priority slots a frame), every key
 * owes the full count instead (T2.4d): the keys are the level that carries
 * the bounces, and a room's bounces need about 14 updates at ρ 0.8.
 */
export const DEFAULT_KEY_PROBE_UPDATES = 4;
/**
 * Updates a fine probe waiting for its keys owes (m2-lights.md §4.5's
 * table: every fine probe traced once since the change). Its keys have
 * already brought it most of the way, and its cut is for what is left.
 */
export const FINE_PROBE_UPDATES = 1;

/**
 * Key probes and propagation (T2.4c), or with `keyProbes` off "far probes
 * owe 2", apply when giving every probe K updates at this frame's batch
 * would take more than this many frames (`N × K / units`). Counted in
 * frames, so the rule doesn't depend on the host's frame rate.
 *
 * 30, the response target's 0.5 s at 60 Hz (T2.4d, DECISIONS 2026-11-06;
 * m2-lights.md §6.4 had 120, 2 s): where every probe's K updates don't fit in
 * the time the view has to respond, a global change has to reach the field
 * through the coarse level first. The furnished room (4,998 probes, 571 a
 * frame: 70 frames) takes the key path now, Sponza (about 250) as before;
 * two-room (10) and Cornell (9) don't. The same window decides whether every
 * key can owe its full count (DEFAULT_KEY_PROBE_UPDATES).
 */
export const FAR_RULE_FRAMES = 30;

/**
 * How long a fine probe waiting for its keys' change may go untraced, in
 * frames since `ddgi/mark` flagged it (T2.4d). Its keys usually take the
 * change in within about 15 frames; the bound is for changes that keep going
 * (a day passing re-marks the keys every frame, so they never finish) and for
 * keys the priority share doesn't reach. 30, the response target's 0.5 s at
 * 60 Hz: during a long change every fine probe is still traced about twice a
 * second, with its keys' change passed to it in between.
 */
export const KEY_WAIT_FRAMES = 30;

/**
 * Changes continue (§4.5's second count, T2.4d) once `ddgi/mark` has run in
 * this many frames in a row: a sun moving through a day, a lamp swinging, a
 * door swinging. A single step counts only on a volume that takes a change
 * in over more than FAR_RULE_FRAMES anyway (T2.18, change-intake-batch.ts).
 */
export const CONTINUING_CHANGE_FRAMES = 30;

/**
 * Frames without a mark that still count as part of a continuing change
 * (T2.8). A host that animates slower than it draws marks a change only on
 * some frames: at 30 Hz on a 60 Hz display, every other frame; at 30 Hz on
 * 144 Hz, about one frame in five. Those gaps are the host's update rate,
 * not the change stopping, so a run of marks survives up to this many
 * unmarked frames in a row. 4 covers an animation at a fifth of the frame
 * rate; after a real stop the smaller batch lasts at most 4 more frames.
 */
export const CONTINUING_CHANGE_GAP_FRAMES = 4;

/**
 * The longest a single change counts as being taken in, in frames after its
 * latest mark (change-intake-batch.ts, T2.18): 120, the frames T2.4c gives
 * the field to come within 5% (2 s at 60 Hz). Normally the census ends the
 * intake much sooner, once nothing is owed; the cap is for a frame without
 * a census (no GPU timings) and for owed work that never quite runs out.
 */
export const CHANGE_INTAKE_MAX_FRAMES = 120;

/**
 * While a change is taken in (§4.5's second count; change-intake-batch.ts),
 * the batch is at least this share of the hand-out, whatever the census
 * says: a guard against a census that reads oddly, not a tuned number (the
 * day on Sponza needs about 0.63, a sun turn's first frames about 0.56).
 */
export const CHANGE_BATCH_MIN_SHARE = 0.5;

/**
 * What one fine probe `ddgi/propagate` passes a key's change to costs, in
 * traced rays, for the batch while a change is taken in. Propagation reads
 * up to 8 key tiles per layer and traces nothing, but it runs in every frame
 * in which keys take a change in, on top of the list. Measured on Sponza
 * after a 30° sun turn (RTX 3060, about 800 MHz, T2.10's VALID run): the
 * pass rose 0.053 ms over a still frame for a median 926 probes propagated,
 * 5.7e-5 ms each, while each extra ray traced cost `ddgi/trace` 2.3e-5 ms
 * (0.22 ms for 9,408 more rays at the same batch): about 2.5 rays a probe.
 * Both scale with the GPU's clock, so the ratio holds as it moves.
 */
export const PROPAGATED_PROBE_RAYS = 2.5;

/**
 * The camera weight's distance falloff, in probe spacings (m2-lights.md
 * §4.3): `1 / (1 + (d / (4 × spacing))²)`. Within about 4 spacings of the
 * camera a probe counts fully; at 10 spacings about 0.14.
 */
export const CAMERA_FALLOFF_SPACINGS = 4;

/** A probe outside the view (frustum plus a cell's diagonal) counts this share as much (m2-lights.md §4.3). */
export const OUT_OF_VIEW_WEIGHT = 0.25;

/**
 * Probes with at least this camera weight are "near and in view": after a
 * global change they owe the full K updates, the rest {@link
 * DEFAULT_FAR_PROBE_UPDATES} (m2-lights.md §6.4). 0.5 is a probe in view
 * within 4 spacings of the camera.
 */
export const NEAR_CAMERA_WEIGHT = 0.5;

/** The warm-up by default: 16 passes, stopping after 8 s once 2 are done (the viewer's rule since T1.8; DECISIONS 2026-11-05). */
export const DEFAULT_WARM_UP_PASSES = 16;
export const DEFAULT_WARM_UP_MAX_SECONDS = 8;
export const DEFAULT_WARM_UP_MIN_PASSES = 2;

/** The kernel's stand-in for an endless ramp (no Infinity in a shader that may assume finite floats). */
const RAMP_UPDATES_ENDLESS = 1e9;

/** The largest ramp the kernel takes: beyond it a probe would stay noisy for thousands of updates. */
const MAX_UPDATES_PER_SAMPLE = 64;

/** What a host must give, a volume or a list of them; everything else has a default. */
export type DDGISettingsInput = Partial<DDGISettings> & ({ readonly volume: DDGIVolumeSettings } | { readonly volumes: readonly DDGIVolumeSettings[] });

export const DEFAULT_DDGI_SETTINGS: Readonly<Omit<DDGISettings, 'volume' | 'volumes'>> = {
    raysPerProbe: 256,
    irradianceTexels: 6,
    distanceTexels: 14,
    adaptiveHysteresis: true,
    hysteresis: 0.97,
    hysteresisMemorySeconds: ADAPTIVE_MEMORY_SECONDS,
    hysteresisMaxSamples: ADAPTIVE_MAX_SAMPLES,
    hysteresisUpdatesPerSample: ADAPTIVE_UPDATES_PER_SAMPLE,
    hysteresisRampUpdates: ADAPTIVE_RAMP_UPDATES,
    irradianceChangeThreshold: Infinity,
    changeDrivenHistory: true,
    changeTolerance: DEFAULT_CHANGE_TOLERANCE,
    changeUpdates: DEFAULT_CHANGE_UPDATES,
    changeBounceShare: DEFAULT_CHANGE_BOUNCE_SHARE,
    lightChangeScale: DEFAULT_LIGHT_CHANGE_SCALE,
    priorityShare: DEFAULT_PRIORITY_SHARE,
    farProbeUpdates: DEFAULT_FAR_PROBE_UPDATES,
    keyProbes: true,
    keyProbeBoost: DEFAULT_KEY_PROBE_BOOST,
    keyResidualShare: DEFAULT_KEY_RESIDUAL_SHARE,
    keyProbeUpdates: DEFAULT_KEY_PROBE_UPDATES,
    holdFineProbes: false,
    feedback: true,
    warmUpPasses: DEFAULT_WARM_UP_PASSES,
    warmUpMaxSeconds: DEFAULT_WARM_UP_MAX_SECONDS,
    warmUpMinPasses: DEFAULT_WARM_UP_MIN_PASSES,
    maxBrightnessChange: 2.0,
    distanceSharpness: 50,
    normalBias: 'auto',
    viewBias: 'auto',
    maxRayDistance: Infinity,
    bounceEnergy: 1.0,
    relocation: true,
    minFrontFaceDistance: 'auto',
    classification: true,
    maxProbesPerFrame: 4096,
    lightGroups: null,
    sunLayer: false,
    minProbesPerFrame: 'auto',
};

/** The first rays of every probe point in fixed directions and are never blended (design: "Rays"). */
export const FIXED_RAYS = 32;
export const MIN_RAYS_PER_PROBE = 64;
export const MAX_RAYS_PER_PROBE = 512;

/** "Unlimited" ray length, as the tracer's own unbounded shadow rays use. */
export const UNBOUNDED_RAY_DISTANCE = 1e27;

/** `auto` values, as shares of the smallest spacing. */
export const AUTO_NORMAL_BIAS_SHARE = 0.1;
export const AUTO_VIEW_BIAS_SHARE = 0.1;
/**
 * 0.02, not the design's first 0.1 (T1.5b, measured on the RTX 3060): probes
 * relocated nearer to surfaces read the steep irradiance near them better.
 * Two-room leak 1.08% at 0.1, 0.96% at 0.05, 0.86% at 0.02 (limit 1%);
 * Cornell +2.4% / +2.8% / +3.0%; thin wall 0.34% / 0.47% / 0.51% (reported).
 * Biases were swept too and left at 0.1: a larger normal bias raised the
 * two-room number (0.2: 1.25%), and no view bias let the thin wall leak 6.9%.
 */
export const AUTO_MIN_FRONT_FACE_SHARE = 0.02;

/**
 * `minProbesPerFrame: 'auto'` (budget v2, docs/design/ddgi.md): every probe
 * updated at least once in this many seconds. Sponza (19,425 probes) at 60 Hz
 * needs 81 a frame, Cornell 8.
 */
export const FLOOR_SWEEP_SECONDS = 4;

/**
 * The frame rate the floor assumes before frames carry a `deltaTime`, as the
 * platform's Budget v2 does (work-budget.ts).
 */
export const FLOOR_ASSUMED_FRAMES_PER_SECOND = 60;

/**
 * ...but `auto` never asks for more than this many times the calibrated batch
 * (the probes the budget buys with the clocks up). Measured on the Intel UHD,
 * Sponza: 16 probes cost 2.1 ms back to back, 40 about 3.0 ms, 81 about
 * 4.8 ms. So the full 4 s sweep would cost more than twice the budget there;
 * capped, Sponza runs about 32-40 probes (about 3 ms, a sweep every 8-10 s). On the
 * RTX 3060 the cap doesn't bind.
 */
export const FLOOR_MAX_CALIBRATED_SHARE = 2;

/**
 * The floor in probes per frame for a volume of `probeCount` probes, at
 * `framesPerSecond`, given the calibrated batch. At most `maxProbesPerFrame`
 * (the technique can't update more).
 */
export function minProbesPerFrame(
    setting: number | 'auto',
    probeCount: number,
    maxProbesPerFrame: number,
    rate: { readonly framesPerSecond: number; readonly calibratedUnits: number },
): number {
    if (setting !== 'auto') return Number.isFinite(setting) && setting > 0 ? Math.min(Math.floor(setting), maxProbesPerFrame) : 0;
    if (!(rate.framesPerSecond > 0) || probeCount <= 0) return 0;
    const sweep = Math.ceil(probeCount / (FLOOR_SWEEP_SECONDS * rate.framesPerSecond));
    const deviceLimit = FLOOR_MAX_CALIBRATED_SHARE * Math.max(0, Math.floor(rate.calibratedUnits));
    return Math.min(sweep, deviceLimit, maxProbesPerFrame);
}

/** Distances written to the distance atlas are clamped to this multiple of the cell diagonal. */
export const DISTANCE_CLAMP_SHARE_OF_DIAGONAL = 1.5;

/** How the uniform says "the big-change rule is off" (no Infinity in a shader that may assume finite floats). */
export const CHANGE_RULE_OFF = -1;

export function withDefaults(input: DDGISettingsInput): DDGISettings {
    // With a list of volumes, `volume` is the first of them, for hosts that read one.
    const volume = 'volume' in input && input.volume ? input.volume : input.volumes?.[0];
    if (!volume) throw new Error('@pmndrs/gi: DDGI settings need a volume (or a non-empty list of volumes)');
    return { ...DEFAULT_DDGI_SETTINGS, ...input, volume };
}

/** The probe grid a volume setting describes. */
export interface ProbeGrid {
    /** Probes per axis. */
    readonly counts: readonly [number, number, number];
    readonly probeCount: number;
    /** World position of probe (0, 0, 0). */
    readonly origin: readonly [number, number, number];
    readonly spacing: readonly [number, number, number];
}

/**
 * Probe counts are `floor(size / spacing) + 1` per axis and the grid is
 * centred in the box. A tiny tolerance keeps `3.0 / 0.5` from flooring to 5
 * through rounding.
 */
export function probeGrid(volume: DDGIVolumeSettings): ProbeGrid {
    const spacing: [number, number, number] =
        typeof volume.spacing === 'number' ? [volume.spacing, volume.spacing, volume.spacing] : [volume.spacing[0], volume.spacing[1], volume.spacing[2]];
    for (let axis = 0; axis < 3; axis++) {
        if (!(spacing[axis] > 0) || !Number.isFinite(spacing[axis])) throw new Error(`@pmndrs/gi: volume spacing must be a positive number, got ${spacing[axis]}`);
        if (!(volume.size[axis] >= 0) || !Number.isFinite(volume.size[axis])) throw new Error(`@pmndrs/gi: volume size must be zero or more, got ${volume.size[axis]}`);
    }
    const counts = spacing.map((axisSpacing, axis) => Math.floor(volume.size[axis] / axisSpacing + 1e-6) + 1) as [number, number, number];
    const origin = spacing.map((axisSpacing, axis) => volume.center[axis] - ((counts[axis] - 1) * axisSpacing) / 2) as [number, number, number];
    return { counts, probeCount: counts[0] * counts[1] * counts[2], origin, spacing };
}

/** `raysPerProbe` as the kernel uses it: a multiple of 32 inside 64..512. */
export function effectiveRaysPerProbe(raysPerProbe: number): number {
    const rounded = Math.round((Number.isFinite(raysPerProbe) ? raysPerProbe : DEFAULT_DDGI_SETTINGS.raysPerProbe) / FIXED_RAYS) * FIXED_RAYS;
    return Math.min(MAX_RAYS_PER_PROBE, Math.max(MIN_RAYS_PER_PROBE, rounded));
}

/** A distance setting in metres, resolving `auto` to `autoShare` of the smallest spacing. */
export function resolveBias(value: number | 'auto', grid: ProbeGrid, autoShare: number): number {
    if (value === 'auto' || !Number.isFinite(value)) return autoShare * Math.min(...grid.spacing);
    return Math.max(0, value);
}

/** Distances in the distance atlas never exceed this: 1.5 × the cell diagonal. */
export function distanceClamp(grid: ProbeGrid): number {
    return DISTANCE_CLAMP_SHARE_OF_DIAGONAL * Math.hypot(...grid.spacing);
}

/** `maxRayDistance` as the kernel uses it: Infinity (or anything not finite and positive) becomes the tracer's unbounded length. */
export function effectiveMaxRayDistance(maxRayDistance: number): number {
    return maxRayDistance > 0 && Number.isFinite(maxRayDistance) ? Math.min(maxRayDistance, UNBOUNDED_RAY_DISTANCE) : UNBOUNDED_RAY_DISTANCE;
}

function clamp(value: number, low: number, high: number, fallback: number): number {
    if (!Number.isFinite(value)) return fallback;
    return Math.min(high, Math.max(low, value));
}

/**
 * The technique's clock advances by at most this much per frame. A longer
 * frame is a stall (a hidden tab, a debugger, a loading hitch), not time in
 * which the probes could have seen the scene change; counted in full, a
 * minute in a background tab would make every probe forget its history at
 * once and the whole field flash noisy as it updates again.
 */
export const MAX_CLOCK_STEP_SECONDS = 0.25;

/**
 * The technique's clock wraps to 0 every this many seconds (2^12, about
 * 68 min). Why: the kernel keeps each probe's last update time as f32, whose
 * step is 0.49 ms below 4096 s but 7.8 ms after a day and 62.5 ms after a
 * week. Wrapped, every time stays under 4096 s, and the blend adds the period
 * back when `clock - t` comes out negative. Only a probe left unblended for
 * more than a whole period would be misread, and an active probe is blended
 * every sweep (seconds); an off probe starts again from state 3 when it comes
 * back on.
 */
export const CLOCK_WRAP_SECONDS = 4096;

/** How long the technique's clock advances for a frame with this `deltaTime`. */
export function clockStep(deltaTime: number): number {
    return Number.isFinite(deltaTime) && deltaTime > 0 ? Math.min(deltaTime, MAX_CLOCK_STEP_SECONDS) : 0;
}

/** The largest `hysteresisMaxSamples` the kernel takes (n / (n + 1) is 1 in f32 well before this). */
const MAX_HISTORY_SAMPLES = 1e6;

/** The live blend settings, each limited to a range that can't break the kernel. */
export function blendSettings(settings: DDGISettings): {
    adaptiveHysteresis: boolean;
    hysteresis: number;
    /** 1 / hysteresisMemorySeconds; 0 never forgets. */
    inverseMemorySeconds: number;
    maxSamples: number;
    updatesPerSample: number;
    rampUpdates: number;
    irradianceChangeThreshold: number;
    maxBrightnessChange: number;
    distanceSharpness: number;
    bounceEnergy: number;
    changeDrivenHistory: boolean;
    changeTolerance: number;
    changeUpdates: number;
    changeBounceShare: number;
    lightChangeScale: number;
    priorityShare: number;
    farProbeUpdates: number;
    keyProbes: boolean;
    keyProbeBoost: number;
    keyResidualShare: number;
    keyProbeUpdates: number;
    holdFineProbes: boolean;
    feedback: boolean;
} {
    const defaults = DEFAULT_DDGI_SETTINGS;
    const memorySeconds = settings.hysteresisMemorySeconds;
    // Infinity never forgets; zero, negative or NaN fall back to the default (no clock to divide by).
    const inverseMemorySeconds = memorySeconds === Infinity ? 0 : 1 / (memorySeconds > 0 ? memorySeconds : defaults.hysteresisMemorySeconds);
    return {
        adaptiveHysteresis: settings.adaptiveHysteresis !== false,
        hysteresis: clamp(settings.hysteresis, 0, 1, defaults.hysteresis),
        inverseMemorySeconds,
        maxSamples: clamp(settings.hysteresisMaxSamples, 0, MAX_HISTORY_SAMPLES, defaults.hysteresisMaxSamples),
        // At least 1 (a plain mean); the kernel divides by it.
        updatesPerSample: clamp(settings.hysteresisUpdatesPerSample, 1, MAX_UPDATES_PER_SAMPLE, defaults.hysteresisUpdatesPerSample),
        // Infinity is an endless ramp (a large finite number for the kernel); NaN or negative falls back to the default.
        rampUpdates: Math.min(settings.hysteresisRampUpdates >= 0 ? settings.hysteresisRampUpdates : defaults.hysteresisRampUpdates, RAMP_UPDATES_ENDLESS),
        // Infinity (or anything not a number) switches the rule off; the kernel reads a negative value as "off".
        irradianceChangeThreshold: settings.irradianceChangeThreshold >= 0 && Number.isFinite(settings.irradianceChangeThreshold) ? settings.irradianceChangeThreshold : CHANGE_RULE_OFF,
        maxBrightnessChange: clamp(settings.maxBrightnessChange, 0, Number.MAX_VALUE, defaults.maxBrightnessChange),
        distanceSharpness: clamp(settings.distanceSharpness, 0, 1000, defaults.distanceSharpness),
        bounceEnergy: clamp(settings.bounceEnergy, 0, Number.MAX_VALUE, defaults.bounceEnergy),
        changeDrivenHistory: settings.changeDrivenHistory !== false,
        // Above 0 and below 1: the cut divides by (r - ε) only when r > ε, and ε = 1 would never cut.
        changeTolerance: clamp(settings.changeTolerance, 1e-4, 0.99, defaults.changeTolerance),
        changeUpdates: clamp(settings.changeUpdates, 1, 1000, defaults.changeUpdates),
        // Below 1: the cut divides by 1 - rho.
        changeBounceShare: clamp(settings.changeBounceShare, 0, 0.95, defaults.changeBounceShare),
        lightChangeScale: clamp(settings.lightChangeScale, 0, 1000, defaults.lightChangeScale),
        priorityShare: clamp(settings.priorityShare, 0, 1, defaults.priorityShare),
        // Whole updates, 1..255: the owed count is 8 bits of the history record (PROBE_HISTORY_UPDATES).
        farProbeUpdates: Math.round(clamp(settings.farProbeUpdates, 1, MAX_OWED_UPDATES, defaults.farProbeUpdates)),
        keyProbes: settings.keyProbes !== false,
        keyProbeBoost: clamp(settings.keyProbeBoost, 0, 1000, defaults.keyProbeBoost),
        keyResidualShare: clamp(settings.keyResidualShare, 0, 1, defaults.keyResidualShare),
        keyProbeUpdates: Math.round(clamp(settings.keyProbeUpdates, 1, MAX_OWED_UPDATES, defaults.keyProbeUpdates)),
        holdFineProbes: settings.holdFineProbes === true,
        feedback: settings.feedback !== false,
    };
}

/** The most updates a probe can be owed: 8 bits in its history record (PROBE_HISTORY_UPDATES in layout.ts). */
export const MAX_OWED_UPDATES = 255;

/** K as whole owed updates (the owed count is an integer in the history record). */
export function owedUpdates(changeUpdates: number): number {
    return Math.max(1, Math.min(MAX_OWED_UPDATES, Math.round(changeUpdates)));
}

/** The live warm-up settings, limited to sensible values. */
export function warmUpSettings(settings: DDGISettings): { passes: number; maxSeconds: number; minPasses: number } {
    const defaults = DEFAULT_DDGI_SETTINGS;
    const passes = Number.isFinite(settings.warmUpPasses) && settings.warmUpPasses > 0 ? settings.warmUpPasses : 0;
    return {
        passes,
        maxSeconds: settings.warmUpMaxSeconds >= 0 ? settings.warmUpMaxSeconds : defaults.warmUpMaxSeconds,
        // Never more than the passes asked for.
        minPasses: Math.min(passes, settings.warmUpMinPasses >= 0 ? settings.warmUpMinPasses : defaults.warmUpMinPasses),
    };
}
