/**
 * DDGI settings (docs/design/ddgi.md, "Settings"), their defaults, and the
 * volume they describe.
 *
 * Settings are a live object, as the technique contract asks: the host edits
 * fields and the next `encode` reads them. Four of them need new GPU
 * resources and only take effect at the next `prepare`: `volume`,
 * `irradianceTexels`, `distanceTexels` and `maxProbesPerFrame`.
 */

import type { Vec3 } from '../../platform/contract/math.ts';

export interface DDGIVolumeSettings {
    /** Centre of the box, metres. */
    readonly center: Vec3;
    /** Size of the box, metres. */
    readonly size: Vec3;
    /** Distance between neighbouring probes, metres: one number, or one per axis. */
    readonly spacing: number | Vec3;
}

export interface DDGISettings {
    /** The box of probes. New `prepare` when changed. */
    volume: DDGIVolumeSettings;
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

/** The kernel's stand-in for an endless ramp (no Infinity in a shader that may assume finite floats). */
const RAMP_UPDATES_ENDLESS = 1e9;

/** The largest ramp the kernel takes: beyond it a probe would stay noisy for thousands of updates. */
const MAX_UPDATES_PER_SAMPLE = 64;

/** What a host must give; everything else has a default. */
export type DDGISettingsInput = Partial<DDGISettings> & { readonly volume: DDGIVolumeSettings };

export const DEFAULT_DDGI_SETTINGS: Readonly<Omit<DDGISettings, 'volume'>> = {
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
    return { ...DEFAULT_DDGI_SETTINGS, ...input };
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
        if (!(spacing[axis] > 0) || !Number.isFinite(spacing[axis])) throw new Error(`@pmndrs/ddgi: volume spacing must be a positive number, got ${spacing[axis]}`);
        if (!(volume.size[axis] >= 0) || !Number.isFinite(volume.size[axis])) throw new Error(`@pmndrs/ddgi: volume size must be zero or more, got ${volume.size[axis]}`);
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
    };
}
