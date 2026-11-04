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
    /** Share of the old value kept per update, 0..1. */
    hysteresis: number;
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
}

/** What a host must give; everything else has a default. */
export type DDGISettingsInput = Partial<DDGISettings> & { readonly volume: DDGIVolumeSettings };

export const DEFAULT_DDGI_SETTINGS: Readonly<Omit<DDGISettings, 'volume'>> = {
    raysPerProbe: 256,
    irradianceTexels: 6,
    distanceTexels: 14,
    hysteresis: 0.97,
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

/** The live blend settings, each limited to a range that can't break the kernel. */
export function blendSettings(settings: DDGISettings): {
    hysteresis: number;
    irradianceChangeThreshold: number;
    maxBrightnessChange: number;
    distanceSharpness: number;
    bounceEnergy: number;
} {
    const defaults = DEFAULT_DDGI_SETTINGS;
    return {
        hysteresis: clamp(settings.hysteresis, 0, 1, defaults.hysteresis),
        // Infinity (or anything not a number) switches the rule off; the kernel reads a negative value as "off".
        irradianceChangeThreshold: settings.irradianceChangeThreshold >= 0 && Number.isFinite(settings.irradianceChangeThreshold) ? settings.irradianceChangeThreshold : CHANGE_RULE_OFF,
        maxBrightnessChange: clamp(settings.maxBrightnessChange, 0, Number.MAX_VALUE, defaults.maxBrightnessChange),
        distanceSharpness: clamp(settings.distanceSharpness, 0, 1000, defaults.distanceSharpness),
        bounceEnergy: clamp(settings.bounceEnergy, 0, Number.MAX_VALUE, defaults.bounceEnergy),
    };
}
