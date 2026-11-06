/**
 * Where DDGI's data lives on the GPU (docs/design/ddgi.md, "The textures",
 * "The write kernel", "Bindings"): atlas sizes and formats, the scratch
 * buffer's layout, and the uniform buffer's layout. Plain numbers, so they are
 * unit-tested without a GPU and shared by the TypeScript and the WGSL.
 */

import { GILimitError } from '../../platform/contract/errors.ts';
import type { ProbeGrid } from './settings.ts';

/** One texel of border on every side of a tile, so bilinear filtering never reads a neighbour's tile. */
export const TILE_BORDER = 1;

/** Probe data: one texel per probe. xyz: relocation offset ÷ spacing; w: state. */
export const PROBE_DATA_FORMAT: GPUTextureFormat = 'rgba16float';
export const IRRADIANCE_FORMAT: GPUTextureFormat = 'rgba16float';

/**
 * The distance atlas is `rg16float` (mean, mean squared) when the device has
 * `texture-formats-tier1`, which makes `rg16float` a storage format; core
 * WebGPU doesn't. The declared fallback is `rgba16float`, twice the memory.
 * Both are filterable without `float32-filterable`.
 */
export type DistanceFormat = 'rg16float' | 'rgba16float';

export function distanceFormatFor(features: ReadonlySet<string>): DistanceFormat {
    return features.has('texture-formats-tier1') ? 'rg16float' : 'rgba16float';
}

export const BYTES_PER_TEXEL: Readonly<Record<DistanceFormat, number>> = { rg16float: 4, rgba16float: 8 };

/** Probe states (the probe-data texture's w channel). */
export const PROBE_STATE = {
    active: 0,
    offInsideGeometry: 1,
    offNoSurfaceNearby: 2,
    notUpdatedYet: 3,
} as const;

/** Legend for debug views and counters. */
export const PROBE_STATE_LABELS: Readonly<Record<number, string>> = {
    [PROBE_STATE.active]: 'active',
    [PROBE_STATE.offInsideGeometry]: 'off: inside geometry',
    [PROBE_STATE.offNoSurfaceNearby]: 'off: no surface nearby',
    [PROBE_STATE.notUpdatedYet]: 'not updated yet',
};

export interface AtlasSize {
    readonly width: number;
    readonly height: number;
    /** One layer per probe row y. */
    readonly layers: number;
    /** Tile side including the border. */
    readonly tile: number;
}

/** A layer per probe row `y`; inside a layer, probe `(x, z)` owns tile `(x, z)`. */
export function atlasSize(grid: ProbeGrid, interiorTexels: number): AtlasSize {
    const tile = interiorTexels + 2 * TILE_BORDER;
    return { width: grid.counts[0] * tile, height: grid.counts[2] * tile, layers: grid.counts[1], tile };
}

export interface FieldLayout {
    readonly irradiance: AtlasSize;
    readonly distance: AtlasSize;
    readonly probeData: AtlasSize;
    readonly distanceFormat: DistanceFormat;
}

export function fieldLayout(grid: ProbeGrid, irradianceTexels: number, distanceTexels: number, distanceFormat: DistanceFormat): FieldLayout {
    return {
        irradiance: atlasSize(grid, irradianceTexels),
        distance: atlasSize(grid, distanceTexels),
        // One texel per probe, no border.
        probeData: { width: grid.counts[0], height: grid.counts[2], layers: grid.counts[1], tile: 1 },
        distanceFormat,
    };
}

/** Throws `GILimitError` naming the first texture limit the layout is past. */
export function checkFieldLimits(layout: FieldLayout, limits: Pick<GPUSupportedLimits, 'maxTextureDimension2D' | 'maxTextureArrayLayers'>): void {
    for (const atlas of [layout.irradiance, layout.distance]) {
        const largest = Math.max(atlas.width, atlas.height);
        if (largest > limits.maxTextureDimension2D) throw new GILimitError('maxTextureDimension2D', largest, limits.maxTextureDimension2D);
    }
    if (layout.irradiance.layers > limits.maxTextureArrayLayers) {
        throw new GILimitError('maxTextureArrayLayers', layout.irradiance.layers, limits.maxTextureArrayLayers);
    }
}

/** Bytes of the three textures. */
export function fieldBytes(layout: FieldLayout): number {
    const bytes = (atlas: AtlasSize, bytesPerTexel: number): number => atlas.width * atlas.height * atlas.layers * bytesPerTexel;
    return bytes(layout.irradiance, 8) + bytes(layout.distance, BYTES_PER_TEXEL[layout.distanceFormat]) + bytes(layout.probeData, 8);
}

/**
 * One probe's slot in the scratch buffer, in 32-bit words, written by the
 * update kernel and copied into the atlases by the write kernel. Values are
 * packed with `pack2x16float`, the atlases' own precision.
 *
 * - irradiance: 2 words per interior texel (rg, then b with 0);
 * - distance: 1 word per interior texel (mean, mean squared);
 * - probe data: 2 words (offset xy, then offset z with the state).
 */
export interface ScratchLayout {
    readonly irradianceWords: number;
    readonly distanceWords: number;
    readonly probeDataWords: number;
    /** Words per probe slot. */
    readonly strideWords: number;
    readonly irradianceOffset: number;
    readonly distanceOffset: number;
    readonly probeDataOffset: number;
}

export const SCRATCH_WORDS_PER_IRRADIANCE_TEXEL = 2;
export const SCRATCH_WORDS_PER_DISTANCE_TEXEL = 1;
export const SCRATCH_PROBE_DATA_WORDS = 2;

export function scratchLayout(irradianceTexels: number, distanceTexels: number): ScratchLayout {
    const irradianceWords = irradianceTexels * irradianceTexels * SCRATCH_WORDS_PER_IRRADIANCE_TEXEL;
    const distanceWords = distanceTexels * distanceTexels * SCRATCH_WORDS_PER_DISTANCE_TEXEL;
    return {
        irradianceWords,
        distanceWords,
        probeDataWords: SCRATCH_PROBE_DATA_WORDS,
        strideWords: irradianceWords + distanceWords + SCRATCH_PROBE_DATA_WORDS,
        irradianceOffset: 0,
        distanceOffset: irradianceWords,
        probeDataOffset: irradianceWords + distanceWords,
    };
}

/**
 * Where each field of `DdgiUniforms` (wgsl/common.ts) sits, in 32-bit words.
 * The first four vec4s are exactly the vec4 parameters `ddgiSampleIrradiance`
 * takes, so the kernels and a renderer pass the same values. WGSL layout:
 * the mat3x3f has 16-byte columns, and the struct rounds up to 16 bytes.
 */
export const UNIFORM_WORD = {
    volumeOrigin: 0,
    volumeSpacing: 4,
    volumeCounts: 8,
    volumeTexels: 12,
    rayRotation: 16,
    hysteresis: 28,
    irradianceChangeThreshold: 29,
    maxBrightnessChange: 30,
    distanceSharpness: 31,
    maxRayDistance: 32,
    distanceClamp: 33,
    bounceEnergy: 34,
    minFrontFaceDistance: 35,
    relocation: 36,
    classification: 37,
    raysPerProbe: 38,
    updateCount: 39,
    frame: 40,
    adaptiveHysteresis: 41,
    clockSeconds: 42,
    inverseMemorySeconds: 43,
    maxHistorySamples: 44,
    updatesPerSample: 45,
    rampUpdates: 46,
    changeDrivenHistory: 47,
    changeTolerance: 48,
    changeUpdates: 49,
    lightChangeScale: 50,
    feedback: 51,
    priorityCount: 52,
    scrambleStride: 53,
    farOwedUpdates: 54,
    cameraFalloffDistance: 55,
    cameraPosition: 56,
    frustumLeft: 60,
    frustumRight: 64,
    frustumBottom: 68,
    frustumTop: 72,
    frustumNear: 76,
    frustumFar: 80,
    changeBounceShare: 84,
    keyProbes: 85,
    keyProbeBoost: 86,
    keyResidualShare: 87,
    keyOwedUpdates: 88,
    fineOwedUpdates: 89,
    propagation: 90,
    holdFineProbes: 91,
} as const;
/** 92 words, a whole number of the struct's 16-byte alignment. */
export const UNIFORM_WORDS = 92;
/** The six frustum planes in UNIFORM_WORD, in the order `frustumPlanes` (camera.ts) returns them. */
export const FRUSTUM_PLANE_WORDS = [
    UNIFORM_WORD.frustumLeft,
    UNIFORM_WORD.frustumRight,
    UNIFORM_WORD.frustumBottom,
    UNIFORM_WORD.frustumTop,
    UNIFORM_WORD.frustumNear,
    UNIFORM_WORD.frustumFar,
] as const;
export const UNIFORM_BYTES = UNIFORM_WORDS * 4;
/** Words per column of the rotation (vec3f columns padded to 16 bytes). */
export const MAT3_COLUMN_WORDS = 4;
/** The shading parameters: vec4s 0-3. */
export const SAMPLE_PARAMETER_VEC4_COUNT = 4;

/**
 * One ray in the ray buffer between `ddgi/trace` and `ddgi/blend`: two
 * words, rgb radiance as half floats (rg, then b) and the signed hit distance
 * as a half float (negative: a back face). Half floats are the atlases' own
 * precision. A probe's rays sit at `slot × raysPerProbe + ray`.
 */
export const RAY_RECORD_BYTES = 8;

/**
 * One probe's history (`DdgiProbeHistory` in wgsl/blend.ts): T1.12's sample
 * count, cut by changes the scene reports (docs/design/m2-lights.md §4.1).
 * 48 bytes, in 32-bit words. One record and one blend rule for every kind of
 * change; later tasks fill the words marked so, and until then they stay 0.
 *
 * - `samples` (f32): n, the updates the probe's light is averaged over after
 *   its last blended update (at most `hysteresisMaxSamples`);
 * - `lastUpdateSeconds` (f32): the technique's clock at that update;
 * - `updates`: k, its blended updates since it was new, reset or back on, in
 *   the low 16 bits (saturating); updates owed at priority in bits 16-23
 *   (set by `ddgi/mark` and feedback requests, one taken off by each
 *   blend); the sky transfer's own update count in bits 24-31
 *   (T2.6b). See {@link PROBE_HISTORY_UPDATES};
 * - `pendingChange` (f32): the relative change of the probe's light since its
 *   last update, added up over `ddgi/mark`'s marks and reset by the blend,
 *   which cuts the history for it. An f32, so many small marks don't round
 *   away;
 * - `urgency` (f32): the largest change since its last update, for the
 *   scheduler's priority (written by `ddgi/mark` and by feedback requests;
 *   kept while the probe is owed updates and reset with the last of them);
 * - `scheduleAndFlags`: flags in the low 8 bits ({@link PROBE_HISTORY_FLAG}),
 *   the frame it was last put in the update list in the high 24 (the
 *   scheduler's stamp, so no probe is listed twice in a frame);
 * - `meanRedGreen`, `meanBlueNoise`: the probe's mean irradiance over its
 *   texels (rgb as half floats, `pack2x16float`) and its noise: the root of a
 *   running mean, over quiet updates, of the squared distance between an
 *   update's mean estimate and the stored mean (a half float; T2.4b's
 *   feedback rule reads it). Written by every blend; `ddgi/mark` divides a
 *   light's change by the mean;
 * - `wakeUntil`: M3 (m3-moving.md 5.2);
 * - `keysReceived`: for a fine probe awaiting a change (T2.4c, m2-lights.md
 *   §4.5), which of its up to 8 key neighbours have passed it their change
 *   since its own last update, in the low 8 bits (bit j: key neighbour j,
 *   {@link keyNeighbourOffset}), and the sum of their propagation weights
 *   as a half float in the high 16 bits ({@link PROBE_HISTORY_KEYS});
 *   cleared by its own blend;
 * - `changeClock` (u32, fixed point, {@link CHANGE_CLOCK_UNITS}): the
 *   volume's change clock at its last blended update (T2.4c);
 * - `keyDeltaSlot`: for a key probe whose latest update passed its change on
 *   (the `passedChange` flag), the update-list slot of that update: where
 *   its entry in the key-delta buffer is (T2.4c).
 *
 * The history buffer (`DDGIField.probeHistory`) holds one per probe in
 * probe-index order, the order of the `points` debug view's texels. It starts
 * at zero, and a probe's first blended update (state 3) starts it again.
 */
export const PROBE_HISTORY_WORD = {
    samples: 0,
    lastUpdateSeconds: 1,
    updates: 2,
    pendingChange: 3,
    urgency: 4,
    scheduleAndFlags: 5,
    meanRedGreen: 6,
    meanBlueNoise: 7,
    wakeUntil: 8,
    keysReceived: 9,
    changeClock: 10,
    keyDeltaSlot: 11,
} as const;
export const PROBE_HISTORY_WORDS = 12;
export const PROBE_HISTORY_BYTES = PROBE_HISTORY_WORDS * 4;

/** The fields packed into the `updates` word. */
export const PROBE_HISTORY_UPDATES = {
    /** k, saturating here: the ramp only needs k while k / a is under n_max. */
    countMask: 0xffff,
    /** Updates owed at priority (T2.4b's scheduler): bits 16-23. */
    owedShift: 16,
    owedMask: 0xff,
    /** The sky transfer's update count (T2.6b): bits 24-31. */
    transferShift: 24,
} as const;

/** Flags in the low 8 bits of `scheduleAndFlags`. */
export const PROBE_HISTORY_FLAG = {
    /** `ddgi/mark` marked it since its last blended update. */
    marked: 1 << 0,
    /** It was marked before its last blended update too: a change that keeps going (the steady-change rule, m2-lights.md §4.1). */
    markedBeforeLastUpdate: 1 << 1,
    /** It waits for its key neighbours' change (T2.4c). */
    awaitingChange: 1 << 2,
    /**
     * It is taking in a reported change: owed updates since a mark (T2.4b).
     * Its updates are expected to move, so they neither feed its noise nor
     * trip the feedback rule (m2-lights.md §4.6). Cleared with the last owed update.
     */
    takingChange: 1 << 3,
    /**
     * A key probe's latest update passed its change to its fine neighbours
     * (T2.4c): its entry is in the key-delta buffer at `keyDeltaSlot`. Only
     * meaningful in the frame of its scheduled stamp: propagation looks at
     * keys updated this frame.
     */
    passedChange: 1 << 4,
} as const;
/** The packing of `keysReceived` (T2.4c). */
export const PROBE_HISTORY_KEYS = {
    /** Which key neighbours have passed their change: bit j for neighbour j. */
    bitsMask: 0xff,
    /** The sum of their weights, a half float in the high 16 bits. */
    weightShift: 16,
} as const;

/** The flags' share of `scheduleAndFlags`; the scheduled frame (T2.4b) is the rest. */
export const PROBE_HISTORY_FLAGS_MASK = 0xff;

/**
 * The census buffer (read back a few frames late, for the counters): probes
 * per state (indices are the states), rays traced this frame, the probe marks
 * `ddgi/mark` made this frame, the feedback tests that fired, the key probes
 * that passed their change on, the fine probes propagation reached and the
 * largest change marked (cleared every frame), then totals since `prepare`
 * (never cleared, so a test can read the total over a run even though not
 * every frame's census is read back), the volume's change clock, then the
 * scheduler's state.
 *
 * The change clock (T2.4c, m2-lights.md §4.5) is state rather than a count,
 * but it lives here because every kernel that reads or advances it
 * (`ddgi/mark`, `ddgi/blend`, `ddgi/propagate`) already binds the census.
 */
export const CENSUS_RAYS_TRACED = 4;
export const CENSUS_PROBES_MARKED = 5;
/** Probe updates whose feedback test fired this frame (`ddgi/blend`, m2-lights.md §4.6). */
export const CENSUS_FEEDBACK_FIRED = 6;
/** Key probes whose update this frame passed their change to their fine neighbours (`ddgi/blend`, T2.4c). */
export const CENSUS_KEY_CHANGES_PASSED = 7;
/** Fine probes that took in their keys' change this frame (`ddgi/propagate`, T2.4c). */
export const CENSUS_PROBES_PROPAGATED = 8;
/** The largest change `ddgi/mark` marked this frame, as its f32 bits (atomicMax; non-negative floats order like their bits). */
export const CENSUS_LARGEST_CHANGE = 9;
/** Words cleared at the start of every frame: those above. */
export const CENSUS_PER_FRAME_WORDS = 10;
export const CENSUS_PROBES_MARKED_TOTAL = 10;
/** Feedback tests that fired since `prepare`, and probe updates blended since `prepare` (never cleared): the share that fired. */
export const CENSUS_FEEDBACK_FIRED_TOTAL = 11;
export const CENSUS_PROBES_BLENDED_TOTAL = 12;
/** Fine probes propagation reached since `prepare` (a probe reached in two frames counts twice). */
export const CENSUS_PROBES_PROPAGATED_TOTAL = 13;
/** Key-probe updates that passed their change on since `prepare`. */
export const CENSUS_KEY_CHANGES_PASSED_TOTAL = 14;
/**
 * The volume's change clock G (T2.4c): every frame `ddgi/mark` adds the
 * largest change it marked, in fixed point ({@link CHANGE_CLOCK_UNITS}),
 * wrapping modulo 2^32. Never cleared.
 */
export const CENSUS_CHANGE_CLOCK = 15;
/**
 * Then a copy of the scheduler's state words 0..SCHEDULE_STATE_REPORTED_WORDS
 * (SCHEDULE_STATE_WORD), copied in before the census is read back, for the
 * counters ("probes owed", "probes updated at priority").
 */
export const CENSUS_SCHEDULE_STATE = 16;
export const SCHEDULE_STATE_REPORTED_WORDS = 8;
export const CENSUS_WORDS = CENSUS_SCHEDULE_STATE + SCHEDULE_STATE_REPORTED_WORDS;

/**
 * The change clock counts changes in units of 1/4096 of a relative change r
 * (m2-lights.md §4.5). A u32 in fixed point rather than an f32: an f32 that
 * grows for hours stops advancing once its step passes the size of small
 * marks. Differences are taken modulo 2^32, like T1.12's wrapping clock.
 */
export const CHANGE_CLOCK_UNITS = 4096;

/**
 * Key probes (T2.4c, m2-lights.md §4.5): a probe is a key when its grid
 * index on every axis is even or the last one, about one probe in 8. The
 * last one matters on even-sized grids, which would otherwise have no keys
 * on their far faces. Ordinary probes otherwise: the same atlas, traced and
 * blended by the same kernels. (The WGSL twin is `ddgiIsKeyProbe`.)
 */
export function isKeyProbe(coordinates: readonly [number, number, number], counts: readonly [number, number, number]): boolean {
    return coordinates.every((value, axis) => value % 2 === 0 || value === counts[axis] - 1);
}

/** Key probes in a grid: per axis, the even indices plus the last when it is odd. */
export function keyProbeCount(counts: readonly [number, number, number]): number {
    return counts.map((count) => Math.ceil(count / 2) + (count > 1 && count % 2 === 0 ? 1 : 0)).reduce((product, axisKeys) => product * axisKeys, 1);
}

/** Key neighbours a fine probe can have (2 per axis where its index isn't a key index). */
export const KEY_NEIGHBOURS = 8;

/**
 * Key neighbour j (0..7) of a fine probe, as an offset in grid steps (the
 * WGSL twin is `ddgiKeyNeighbourOffset`): on an axis where the probe's index
 * is a key index (even or last) the offset is 0 and that axis's bit of j
 * must be 0; otherwise the bit (x: 0, y: 1, z: 2) chooses the key one step
 * below (0) or above (1). Null when j isn't a neighbour of this probe. A fine
 * probe has 2, 4 or 8 key neighbours, all at equal trilinear weight: on the
 * key lattice it sits halfway between its keys on each such axis.
 */
export function keyNeighbourOffset(neighbour: number, coordinates: readonly [number, number, number], counts: readonly [number, number, number]): [number, number, number] | null {
    const offset: [number, number, number] = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
        const above = (neighbour >> axis) & 1;
        const keyAxis = coordinates[axis] % 2 === 0 || coordinates[axis] === counts[axis] - 1;
        if (keyAxis) {
            if (above) return null;
        } else {
            offset[axis] = above ? 1 : -1;
        }
    }
    return offset;
}

/**
 * The key-delta buffer (T2.4c): one entry per update-list slot, written by
 * `ddgi/blend` for a key probe whose update takes in a reported change, and
 * read by `ddgi/propagate` in the same frame. An entry is a header (the key's
 * probe index and its change-clock stamp from before this update, G_prev),
 * then the change this update made, new − old, per interior irradiance
 * texel, as half floats (rg, then b and 0): 288 bytes a layer at 6×6 texels.
 */
export const KEY_DELTA_HEADER = { probeIndex: 0, clockBefore: 1 } as const;
export const KEY_DELTA_HEADER_WORDS = 2;
export function keyDeltaStrideWords(irradianceTexels: number): number {
    return KEY_DELTA_HEADER_WORDS + irradianceTexels * irradianceTexels * SCRATCH_WORDS_PER_IRRADIANCE_TEXEL;
}

/**
 * Propagation's own scratch (T2.4c): per entry of its compacted list of fine
 * probes, the probe's new irradiance texels, packed as the update scratch
 * packs them (rg, then b and the alpha the atlas keeps), for its own write.
 */
export function propagationScratchStrideWords(irradianceTexels: number): number {
    return irradianceTexels * irradianceTexels * SCRATCH_WORDS_PER_IRRADIANCE_TEXEL;
}

/**
 * How many fine probes propagation can list in one frame: every fine probe,
 * but no more than the fine neighbours of a full update list of keys (each
 * key has at most 26).
 */
export function propagationListCapacity(counts: readonly [number, number, number], maxProbesPerFrame: number): number {
    const fineProbes = counts[0] * counts[1] * counts[2] - keyProbeCount(counts);
    return Math.max(1, Math.min(fineProbes, 26 * maxProbesPerFrame));
}

/**
 * The propagation state buffer (T2.4c): the indirect dispatch arguments for
 * `ddgi/propagate` and its write (x, y, z workgroups: one workgroup per
 * listed fine probe, in two dimensions past the per-dimension limit), then
 * how many fine probes `ddgi/propagate-find` listed this frame. Cleared every
 * frame that runs propagation.
 */
export const PROPAGATION_STATE_WORD = { groupsX: 0, groupsY: 1, groupsZ: 2, listed: 3 } as const;
export const PROPAGATION_STATE_WORDS = 4;

/**
 * The GPU scheduler's state buffer (wgsl/schedule.ts), one u32 per word:
 *
 * - `cursorProbe`: the probe at the round robin's cursor in the scrambled
 *   order (the order's next entry), kept from frame to frame;
 * - `aboveTotal`: probes listed this frame from buckets above the boundary
 *   bucket (all of them fit), written by `ddgi/schedule-select`;
 * - `aboveSlots`, `boundarySlots`: running counters `ddgi/schedule-write`
 *   reserves list slots with;
 * - `boundaryBucket`: the bucket where the priority share fills up, as an
 *   i32 (−1: every owed probe fits);
 * - `boundaryRoom`: how many probes of the boundary bucket fit;
 * - `listed`: probes listed at priority this frame (min(owed, P));
 * - `owed`: probes owed updates this frame;
 * - `histogram`: probes per priority bucket (PRIORITY_BUCKETS words).
 */
export const SCHEDULE_STATE_WORD = {
    cursorProbe: 0,
    aboveTotal: 1,
    aboveSlots: 2,
    boundarySlots: 3,
    boundaryBucket: 4,
    boundaryRoom: 5,
    listed: 6,
    owed: 7,
    histogram: 8,
} as const;
/** Priority buckets, each twice as urgent as the one below (m2-lights.md §4.3). */
export const PRIORITY_BUCKETS = 32;
/**
 * Bucket b holds priorities (urgency × camera weight) in
 * [2^(b − offset), 2^(b + 1 − offset)); bucket 0 also everything smaller and
 * the top bucket everything larger. With 20, the buckets span about 1e-6 to
 * 2,000: urgencies run from the 5% tolerance to MAX_GLOBAL_CHANGE (100), and
 * camera weights from 1 down to about 0.001 (far and out of view).
 */
export const PRIORITY_BUCKET_OFFSET = 20;
export const SCHEDULE_STATE_WORDS = SCHEDULE_STATE_WORD.histogram + PRIORITY_BUCKETS;

/**
 * `ddgi/mark`'s table (`DdgiMarkTable` in wgsl/mark.ts), written by the CPU
 * in frames with changes: a header (entry count, thread count, 2 spare
 * words), then one entry per changed light or global change. Each entry
 * covers a box of probes, one thread per probe; `firstThread` is the running
 * sum of the boxes before it, so a thread finds its entry by binary search.
 *
 * An entry: box min (3 u32) and kind ({@link MARK_KIND}), box size (3 u32)
 * and first thread, the global change r and 3 spare words, then the light
 * before and after, `MARK_LIGHT_WORDS` each ({@link MARK_LIGHT_WORD}).
 */
export const MARK_HEADER_WORDS = 4;
export const MARK_LIGHT_WORDS = 16;
export const MARK_ENTRY_WORD = {
    boxMin: 0,
    kind: 3,
    boxSize: 4,
    firstThread: 7,
    globalChange: 8,
    before: 12,
    after: 12 + MARK_LIGHT_WORDS,
} as const;
export const MARK_ENTRY_WORDS = 12 + 2 * MARK_LIGHT_WORDS;
/** Inside one light of an entry: position (3) and kind, direction (3) and range, colour × intensity (3) and cos inner cone, cos outer cone and 3 spare. */
export const MARK_LIGHT_WORD = {
    position: 0,
    kind: 3,
    direction: 4,
    range: 7,
    intensity: 8,
    cosInnerCone: 11,
    cosOuterCone: 12,
} as const;

/** What an entry marks. */
export const MARK_KIND = {
    /** A point or spot light changed: the change of its irradiance vector at each probe of its box. */
    localLight: 0,
    /** A directional light changed: every probe, r times the openness factor toward its old and new direction. */
    directionalLight: 1,
    /** The environment changed: every probe, r times the openness factor over six directions. */
    environment: 2,
    /** A change with no shape (more light changes than are listed): every probe, r as it is. */
    everywhere: 3,
} as const;

/** A light's kind inside an entry. */
export const MARK_LIGHT_KIND = { point: 0, spot: 1, directional: 2 } as const;

/** The points debug view: one rgba32float texel per probe, in probe-index order, rows as wide as the device allows. */
export function pointsSize(probeCount: number, maxTextureDimension2D: number): { width: number; height: number } {
    const width = Math.min(probeCount, maxTextureDimension2D);
    return { width, height: Math.ceil(probeCount / width) };
}
