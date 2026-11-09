/**
 * DDGI as a platform technique (docs/design/ddgi.md): a box of probes whose
 * irradiance and distance atlases are updated a budget of probes per frame.
 *
 * Each frame, into the host's encoder:
 * 0. `ddgi/mark`, only in frames whose `SceneChanges` report a light, sky or
 *    geometry change: tells each probe how much its light changed, from the
 *    lights' own numbers or how far an object moved and how much of the
 *    probe's view it covers (wgsl/mark.ts, mark-table.ts), so its next blend
 *    cuts its history to match, and promises it updates at priority
 *    (docs/design/m2-lights.md §4.1, §4.2; m3-moving.md §5.1). The hot movers
 *    (mover-table.ts) follow the frame's moves first: their mask bits go in
 *    the mark table, and their boxes to `ddgi/blend`'s mover table;
 * 0b. `ddgi/schedule`: the GPU scheduler's four dispatches fill the update
 *    list, probes owed updates first by urgency and nearness to the camera,
 *    the rest from the round robin (wgsl/schedule.ts, §4.3);
 * 1. `ddgi/trace`: one thread per ray traces inline, shades the hit and
 *    writes the ray results; with direct light deferred (the default), it
 *    puts hits a light can reach on the ray scratch's work list, and the
 *    platform's `gi/direct` adds their direct light (`<id>/gi-direct`, timed
 *    as DDGI's), one round of at most DIRECT_ROUND_PROBES probes at a time;
 * 2. `ddgi/blend`: one workgroup per probe classifies and relocates the
 *    probe (phase B) and blends its rays into the scratch buffer (phase C);
 * 3. `ddgi/write`: copies the scratch into the atlases, the probe data and
 *    the `points` debug view;
 * 3b. `ddgi/propagate`, with key probes on (T2.4c, §4.5): fine probes
 *    waiting for their key neighbours take the change those keys' updates
 *    made this frame, and their tiles are written (wgsl/propagate.ts). It
 *    does nothing in a frame where no key passed a change on;
 * 3c. `ddgi/compose`, with light groups on (T2.5, §5; light-groups.ts,
 *    wgsl/compose.ts): the composed atlas renderers sample, Σ weight ×
 *    layer, over the tiles written this frame, or over every tile in a
 *    frame where a weight changed (and in the first frame after `prepare`).
 *    A weight change marks nothing and cuts no history: it is a compose, not
 *    a light change;
 * 4. `ddgi/census`, only when the platform times passes (its `gpuTiming`
 *    option, the one setting that allows readbacks): counts probes per state
 *    for the counters, read back a few frames later.
 *
 * **The warm-up** (`warmUpPasses`, moved here from the viewer in T2.4b):
 * after `prepare` and `resetHistory`, the technique does no probe work while
 * the ray-tracing trees are still being built (rays would see through every
 * wall of a half-built scene to the sky), then makes full passes over every
 * probe, `maxProbesPerFrame` a frame whatever the frame's budget, with no
 * time between them for the history (the same running average the viewer's
 * warm-up gave). `warmingUp` says so, and the platform leaves those frames
 * out of the budget's calibration. A host's loading screen runs them, so
 * the first frame a user sees has converged GI.
 *
 * **Several volumes** (M3, docs/design/m3-moving.md §4.1, volumes.ts): the
 * volumes share the atlases, the kernels and the volume table, and each has
 * its own uniforms, buffers and bind groups (`PreparedResources`, one per
 * volume, with the shared parts the same objects in each). Each frame every
 * volume runs steps 0-4 above in turn, on its share of the frame's probes;
 * the clock, the hot movers, the layer weights and the warm-up are the
 * technique's, once a frame. One volume is the one-element case.
 *
 * **The camera-following volume** (M3, m3-moving.md §4.2, scroll.ts): a
 * volume with `follow: 'camera'` is a window of probes over a lattice fixed
 * in world space. Before any volume's passes, each such window follows the
 * frame's camera (`CameraWindow`, on the CPU), and `ddgi/scroll`
 * (wgsl/scroll.ts) resets the slots of the planes that entered and moves the
 * valid extent, which it copies into the volume table. The kernels then work
 * in slots, as for a fixed volume, whose window never moves.
 *
 * It sees only its `TechniqueContext`: the scene copy, the `rays` backend,
 * hit shading, the kernel builder and the timer. Never the host or an engine.
 * Renderers read the field through its shading module (`shading`,
 * shading.ts), which wraps the same `ddgiSampleVolumes` the update's bounce
 * light uses.
 */

import type { DebugView } from '../../platform/contract/stats.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame, TechniqueRequirements, WorkFloorRate } from '../../platform/contract/technique.ts';
import type { DirectWorkList, RayScratch } from '../../platform/contract/shade.ts';
import type { ColorRGB } from '../../platform/contract/math.ts';
import type { LightHandle } from '../../platform/contract/handles.ts';
import type { DirectionalLightDescriptor } from '../../platform/contract/scene.ts';
import type { ComputeKernel } from '../../platform/contract/wgsl.ts';
import { GILimitError } from '../../platform/contract/errors.ts';
import {
    CENSUS_CHANGE_CLOCK,
    CENSUS_DIRECT_DROPPED,
    CENSUS_DIRECT_HITS,
    CENSUS_FEEDBACK_FIRED,
    CENSUS_FEEDBACK_FIRED_TOTAL,
    CENSUS_KEY_CHANGES_PASSED,
    CENSUS_KEY_CHANGES_PASSED_TOTAL,
    CENSUS_PER_FRAME_WORDS,
    CENSUS_PROBES_BLENDED_TOTAL,
    CENSUS_PROBES_MARKED,
    CENSUS_PROBES_MARKED_TOTAL,
    CENSUS_PROBES_PROPAGATED,
    CENSUS_PROBES_PROPAGATED_TOTAL,
    CENSUS_RAYS_TRACED,
    CENSUS_SCHEDULE_STATE,
    CENSUS_WORDS,
    FRUSTUM_PLANE_WORDS,
    checkFieldLimits,
    distanceFormatFor,
    fieldBytes,
    fieldLayout,
    IRRADIANCE_FORMAT,
    keyDeltaStrideWords,
    keyProbeCount,
    PROBE_DATA_FORMAT,
    PROBE_HISTORY_BYTES,
    PROBE_STATE,
    PROBE_STATE_LABELS,
    pointsSize,
    PRIORITY_BUCKETS,
    PROPAGATE_WORKGROUPS,
    PROPAGATION_STATE_WORDS,
    propagationListCapacity,
    propagationScratchStrideWords,
    SCHEDULE_STATE_REPORTED_WORDS,
    SCHEDULE_STATE_WORD,
    SCHEDULE_STATE_WORDS,
    scratchLayout,
    MAT3_COLUMN_WORDS,
    MARK_ENTRY_WORDS,
    MARK_HEADER_WORDS,
    MOVER_TABLE_WORDS,
    UNIFORM_BYTES,
    UNIFORM_WORD,
    UNIFORM_WORDS,
    type DistanceFormat,
    type FieldLayout,
    type ScratchLayout,
} from './layout.ts';
import { frameRayRotation } from './ray-rotation.ts';
import {
    AUTO_MIN_FRONT_FACE_SHARE,
    AUTO_NORMAL_BIAS_SHARE,
    AUTO_VIEW_BIAS_SHARE,
    blendSettings,
    clockStep,
    CLOCK_WRAP_SECONDS,
    FIXED_RAYS,
    distanceClamp,
    effectiveMaxRayDistance,
    effectiveRaysPerProbe,
    FAR_RULE_FRAMES,
    FLOOR_ASSUMED_FRAMES_PER_SECOND,
    FINE_PROBE_UPDATES,
    MAX_OWED_UPDATES,
    minProbesPerFrame,
    owedUpdates,
    resolveBias,
    warmUpSettings,
    withDefaults,
    type DDGISettings,
    type DDGISettingsInput,
    type ProbeGrid,
} from './settings.ts';
import { cameraFalloffDistance, frustumPlanes, inViewMargin } from './camera.ts';
import {
    planVolumes,
    UnitSharer,
    volumeFloorUnits,
    volumeSettingsList,
    VOLUME_SAMPLING_FLOATS,
    VOLUME_TABLE_CPU_TEXELS,
    VOLUME_TABLE_FORMAT,
    VOLUME_TABLE_ROW_FLOATS,
    VOLUME_TABLE_TEXEL,
    VOLUME_TABLE_TEXEL_BYTES,
    VOLUME_TABLE_TEXELS,
    wholeGridExtent,
    writeVolumeParameters,
    writeVolumeWindow,
    type PlacedVolume,
    type VolumePlan,
} from './volumes.ts';
import { CameraWindow, windowProbeOf, type WindowMove } from './scroll.ts';
import {
    DDGI_SCROLL_WGSL,
    SCROLL_ENTRY_POINTS,
    SCROLL_EXTENT_BYTES,
    SCROLL_EXTENT_SIDES,
    SCROLL_UNIFORM_BYTES,
    SCROLL_UNIFORM_WORD,
    SCROLL_UNIFORM_WORDS,
    SCROLL_WORKGROUP_SIZE,
} from './wgsl/scroll.ts';
import { buildMarkTable, type MarkTable, type SunLayerMarks } from './mark-table.ts';
import { layerPlan, MAX_IRRADIANCE_LAYERS, MAX_LIGHT_GROUPS, sunLayerWeight, sunLightOf, writeLayerWeights, type LayerPlan } from './light-groups.ts';
import { COMPOSE_ENTRY_POINTS, COMPOSE_WEIGHTS_BYTES, composeKernelWgsl } from './wgsl/compose.ts';
import { MoverTable } from './mover-table.ts';
import { ChangeIntakeBatch } from './change-intake-batch.ts';
import { priorityCount, scrambleStride } from './schedule.ts';
import { DDGIShadingModule } from './shading.ts';
import { blendKernelWgsl, type PhaseBReduction } from './wgsl/blend.ts';
import { CENSUS_WORKGROUP_SIZE, DDGI_CENSUS_WGSL } from './wgsl/census.ts';
import { DDGI_LIGHT_GROUP_MASK, DDGI_TRACE_WGSL, deferredTraceKernelWgsl, DIRECT_ROUND_PROBES, TRACE_ROUND_BYTES, TRACE_WORKGROUP_SIZE } from './wgsl/trace.ts';
import { MARK_ENTRY_POINTS, MARK_WORKGROUP_SIZE, markKernelWgsl } from './wgsl/mark.ts';
import { PROPAGATE_ENTRY_POINTS, PROPAGATE_FIND_WORKGROUP_SIZE, propagateKernelWgsl } from './wgsl/propagate.ts';
import { SCHEDULE_ENTRY_POINTS, SCHEDULE_WORKGROUP_SIZE, scheduleKernelWgsl } from './wgsl/schedule.ts';
import { DDGI_DEBUG_VIEW } from './wgsl/shading.ts';
import { WRITE_ENTRY_POINTS, writeKernelWgsl } from './wgsl/write.ts';

/** WebGPU usage bits; the globals don't exist in Node, where tests import this file. */
const BUFFER_USAGE = { mapRead: 0x1, copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const TEXTURE_USAGE = { copySrc: 0x1, copyDst: 0x2, textureBinding: 0x4, storageBinding: 0x8 } as const;
const SHADER_STAGE_COMPUTE = 0x4;
const MAP_MODE_READ = 0x1;

/** Bytes of one probe-data texel (rgba16float) and one point (rgba32float). */
const PROBE_DATA_TEXEL_BYTES = 8;
const POINT_TEXEL_BYTES = 16;
/** The half-float bit pattern of 3.0, the "not updated yet" state. */
const HALF_THREE = 0x4200;
/** Census readbacks in flight at once; one is usually enough, three cover a slow driver. */
const CENSUS_STAGING_BUFFERS = 3;
/** The ray buffer's first size, in rays; it grows when a frame needs more. */
const INITIAL_RAY_CAPACITY = 512 * 256;
/** Frame times kept for the sweep counter (its median), about half a second at 60 Hz. */
const SWEEP_FRAME_TIMES = 31;
/** The mark table's first size in bytes (it grows when a frame needs more): room for 64 entries. */
const INITIAL_MARK_TABLE_BYTES = (MARK_HEADER_WORDS + 64 * MARK_ENTRY_WORDS) * 4;

/** Options fixed for the technique's life. */
export interface DDGITechniqueOptions {
    /**
     * How phase B reduces the fixed rays, how `ddgi/mark` counts its marks,
     * how the scheduler counts its histogram and compacts its list, and how
     * `ddgi/propagate-find` compacts its list.
     * Defaults to `subgroups` when the device has the feature, otherwise the
     * declared fallback `workgroup`. Both give the same states, offsets,
     * counts and histograms; tests run both.
     */
    readonly reduction?: PhaseBReduction;
    /**
     * Where a blended ray's direct light is computed:
     * - `deferred` (the default): `ddgi/trace` puts each hit a light can
     *   reach on a work list, and the platform's `gi/direct` traces the
     *   shadow rays over that list only (docs/design/m2-lights.md §2; 2.2×
     *   the probes at 2 ms on Sponza, T2.0);
     * - `inline`: M1's `ddgi/trace`, which traces the shadow rays itself.
     *   Kept for comparison.
     */
    readonly directLight?: DirectLightPath;
    /**
     * With `directLight: 'deferred'`, how `ddgi/trace` fills the work list:
     * `subgroups` (the default when the device has them) or the declared
     * fallback `atomics`. Both give the same field; tests run both.
     */
    readonly directWorkList?: DirectWorkList;
}

/** Where DDGI computes direct light (DDGITechniqueOptions.directLight). */
export type DirectLightPath = 'deferred' | 'inline';

/** One volume of the field, for test tools (DDGIField.volumes). */
export interface DDGIFieldVolume {
    readonly grid: ProbeGrid;
    /** Its priority (volumes.ts); the volumes are listed highest first, as the volume table's rows. */
    readonly priority: number;
    /** The atlas layer of its probe row y = 0. */
    readonly layerBase: number;
    /** Its first probe in the `points` debug view. */
    readonly probeBase: number;
    /**
     * Its four `vec4f` parameters of `ddgiSampleIrradiance` (the start of its
     * volume-table row): 16 floats, refreshed on every read of `field`. For a
     * camera-following volume the origin is its window's first probe, and its
     * data is in wrapping slots: read it through the volume table.
     */
    readonly sampleParameters: Float32Array;
    /** Its probes' hysteresis histories, as `DDGIField.probeHistory` (in slot order). */
    readonly probeHistory: GPUBuffer;
    /** It follows the camera (`follow: 'camera'`, scroll.ts). */
    readonly followsCamera: boolean;
    /** Lattice coordinates of its window's first probe (scroll.ts); [0, 0, 0] for a fixed volume. Read when `field` is read. */
    readonly window: readonly [number, number, number];
    /** A camera-following volume's valid extent, as `ddgi/scroll` keeps it (8 floats: min xyz, spare, max xyz, spare; lattice units); null for a fixed one. */
    readonly validExtent: GPUBuffer | null;
}

/**
 * What test tools read (renderers use `shading`): the textures, the volume
 * table, and the first volume's four `vec4f` parameters of
 * `ddgiSampleIrradiance`, in order. With several volumes, `grid`,
 * `sampleParameters` and `probeHistory` are the first (highest-priority)
 * volume's, and `volumes` lists every one. Textures are replaced only by
 * `prepare`.
 */
export interface DDGIField {
    readonly grid: ProbeGrid;
    /** Every volume, highest priority first (one with a single volume). */
    readonly volumes: readonly DDGIFieldVolume[];
    /** The volume table renderers read (`ddgiSampleVolumes`): a row per volume, in the order of `volumes`. */
    readonly volumeTable: GPUTexture;
    readonly layout: FieldLayout;
    /**
     * The irradiance renderers sample: the composed atlas with light groups
     * on (one block, Σ weight × layer), else the field's one layer.
     */
    readonly irradiance: GPUTexture;
    /**
     * The atlas the update writes, every irradiance layer side by side
     * (`layout.layeredIrradiance`, light-groups.ts); `irradiance` itself with
     * one layer.
     */
    readonly layeredIrradiance: GPUTexture;
    /** Which layer holds what (light-groups.ts). */
    readonly layers: LayerPlan;
    readonly distance: GPUTexture;
    readonly probeData: GPUTexture;
    /** volumeOrigin, volumeSpacing, volumeCounts, volumeTexels: 16 floats, refreshed from the live settings on every read. */
    readonly sampleParameters: Float32Array;
    /**
     * Each probe's hysteresis history, for debug tools (the viewer colours
     * probes by how many updates they average): `PROBE_HISTORY_WORDS` words
     * per probe in probe-index order, laid out as `PROBE_HISTORY_WORD` says.
     * A storage buffer, so compute only: copy what you need into a texture to
     * draw it. Written by `ddgi/blend` during `encode`.
     */
    readonly probeHistory: GPUBuffer;
}

interface CensusStaging {
    readonly buffer: GPUBuffer;
    state: 'free' | 'copied' | 'mapping';
    /** The probes the copied frame listed (its update count). */
    listedProbes: number;
    /** The platform frame the copy was recorded in. */
    frame: number;
}

/**
 * What every volume shares (volumes.ts): the atlases, the volume table, the
 * kernels and the field's shape. The same objects in every volume's
 * `PreparedResources`.
 */
interface SharedResources {
    readonly layout: FieldLayout;
    readonly scratch: ScratchLayout;
    readonly irradianceTexels: number;
    readonly distanceTexels: number;
    readonly reduction: PhaseBReduction;
    /** Where direct light is computed, and how the deferred work list is filled (null when inline). */
    readonly directLight: DirectLightPath;
    readonly directWorkList: DirectWorkList | null;
    /** Which irradiance layer holds what (light groups, light-groups.ts). */
    readonly plan: LayerPlan;
    /** The volumes in priority order, where each sits in the atlases, and the atlas's size. */
    readonly volumePlan: VolumePlan;
    /** The layered irradiance atlas the update reads and writes (one block per layer). */
    readonly irradiance: GPUTexture;
    /** With light groups on: the composed atlas renderers sample (null with one layer). */
    readonly composed: GPUTexture | null;
    /** With light groups on: the layers' weights (null with one layer). */
    readonly composeWeights: GPUBuffer | null;
    readonly distance: GPUTexture;
    readonly probeData: GPUTexture;
    /** Every volume's sampling parameters, a row each in priority order (volumes.ts): what renderers and bounce light read. */
    readonly volumeTable: GPUTexture;
    readonly points: GPUTexture;
    readonly sampler: GPUSampler;
    /** The hot movers' boxes and mask bits for `ddgi/blend` (a uniform buffer; mover-table.ts). */
    readonly movers: GPUBuffer;
    readonly traceKernel: ComputeKernel;
    readonly blendKernel: ComputeKernel;
    readonly writeKernel: ComputeKernel;
    readonly censusKernel: ComputeKernel;
    readonly markKernel: ComputeKernel;
    /** The change clock's one-thread dispatch after the mark (same bind group). */
    readonly markClockKernel: ComputeKernel;
    /** The scheduler's four kernels (wgsl/schedule.ts), one bind group for all. */
    readonly scheduleKernels: Readonly<Record<keyof typeof SCHEDULE_ENTRY_POINTS, ComputeKernel>>;
    /** `ddgi/scroll`'s two kernels (wgsl/scroll.ts); null when no volume follows the camera (they are not compiled then). */
    readonly scrollKernels: Readonly<Record<keyof typeof SCROLL_ENTRY_POINTS, ComputeKernel>> | null;
    /** Deferred only: the byte distance between two rounds' `DdgiTraceRound`s (dynamic offsets). */
    readonly traceRoundStride: number;
    /** Bytes of the shared textures and buffers. */
    readonly sharedBytes: number;
}

/** What the frame loop keeps for one volume between frames. */
interface VolumeFrameState {
    /** The latest census read back: probes per state, then rays traced. Null until the first arrives. */
    latestCensus: Uint32Array | null;
    /** The batch while a change is taken in (§4.5's second count; T2.4d, T2.18), from the census's rays a listed probe. */
    readonly changeIntake: ChangeIntakeBatch;
    /** Probes listed in the latest frame. */
    probesUpdated: number;
    /** The last frame's update count above 0: the batch "far probes owe 2" judges a frame with no updates by. */
    lastUpdateCount: number;
    /** Probe updates the current warm-up has made in this volume. */
    warmUpProbeUpdates: number;
    /** Its four sampling parameters (its volume-table row), refreshed in place from the live settings. */
    readonly sampleParameters: Float32Array;
}

/**
 * One volume's resources: its own buffers and bind groups, and the shared
 * ones. Most of the technique's methods take one of these, so they work on
 * one volume at a time, as the kernels do.
 */
interface PreparedResources extends SharedResources {
    /** This volume: its grid, priority, layer base and first probe (volumes.ts). */
    readonly volume: PlacedVolume;
    readonly grid: ProbeGrid;
    readonly maxProbesPerFrame: number;
    /** With light groups on: the compose kernels' bind groups for this volume (null with one layer). */
    readonly compose: ComposeResources | null;
    readonly uniforms: GPUBuffer;
    readonly updateList: GPUBuffer;
    readonly scratchBuffer: GPUBuffer;
    readonly census: GPUBuffer;
    /** Per probe: the updates its history counts as, the clock at its last update and its blended updates so far (adaptive hysteresis). */
    readonly probeHistory: GPUBuffer;
    readonly censusStaging: CensusStaging[];
    /** Propagation (T2.4c, wgsl/propagate.ts): find, args, propagate, and the irradiance-tile write. */
    readonly propagation: PropagationResources;
    readonly scheduleGroup: GPUBindGroup;
    /** The scheduler's state (SCHEDULE_STATE_WORD): the round robin's cursor and this frame's histogram and counts. */
    readonly scheduleState: GPUBuffer;
    /** One u32 per probe: an update the feedback rule asked for, as its urgency's bits. */
    readonly feedbackRequests: GPUBuffer;
    readonly writeGroup: GPUBindGroup;
    readonly censusGroup: GPUBindGroup;
    /** The ray scratch (ray results, and the work list when deferred) and the groups that bind it; replaced when it grows. */
    rays: RayBuffers;
    /** Deferred only: one `DdgiTraceRound` per round, `traceRoundStride` bytes apart (dynamic offsets). */
    readonly traceRounds: GPUBuffer | null;
    /** `ddgi/mark`'s table and the group that binds it; replaced when a frame's table outgrows it. */
    mark: { buffer: GPUBuffer; group: GPUBindGroup };
    /** The round robin's stride: order[i] = (i × stride) mod probe count (schedule.ts). */
    readonly stride: number;
    /** Bytes of this volume's own buffers (the ray scratch apart: it grows). */
    readonly ownBytes: number;
    readonly state: VolumeFrameState;
    /** A camera-following volume's window and `ddgi/scroll`'s buffers; null for a fixed volume. */
    readonly scroll: ScrollResources | null;
}

/** One camera-following volume's window and `ddgi/scroll`'s buffers and bind groups (scroll.ts, wgsl/scroll.ts). */
interface ScrollResources {
    readonly window: CameraWindow;
    /** Its row of the volume table. */
    readonly row: number;
    readonly uniforms: GPUBuffer;
    /** The valid extent (SCROLL_EXTENT_WORDS floats), copied into the table's row every frame. */
    readonly extent: GPUBuffer;
    readonly resetGroup: GPUBindGroup;
    readonly extentGroup: GPUBindGroup;
    /** The next scroll resets every slot (a teleport). */
    resetEverySlot: boolean;
    /** The next scroll sets the valid extent to its targets outright (after prepare, resetHistory or a teleport). */
    extentToTarget: boolean;
}

/** The ray scratch and the groups that bind it (`createRayBuffer`). */
interface RayBuffers {
    readonly scratch: RayScratch;
    /** Ray results it holds. */
    readonly capacity: number;
    readonly traceGroup: GPUBindGroup;
    readonly blendGroup: GPUBindGroup;
}

/** `ddgi/compose`'s kernels and one volume's bind groups (T2.5, wgsl/compose.ts); the texture and weights are `SharedResources`'. */
interface ComposeResources {
    readonly updatedKernel: ComputeKernel;
    readonly listedKernel: ComputeKernel;
    readonly allKernel: ComputeKernel;
    /** Binds the update list; the listed kernel's binds propagation's list and state. */
    readonly updatedGroup: GPUBindGroup;
    readonly listedGroup: GPUBindGroup;
}

/** The kernels each volume's propagation and compose groups are made for (`createVolumeResources`). */
interface VolumeKernels {
    readonly propagation: { readonly find: ComputeKernel; readonly args: ComputeKernel; readonly propagate: ComputeKernel; readonly write: ComputeKernel };
    /** With light groups on; null with one layer. */
    readonly compose: { readonly updated: ComputeKernel; readonly listed: ComputeKernel; readonly all: ComputeKernel } | null;
}

/** Propagation's kernels, buffers and bind groups (T2.4c, wgsl/propagate.ts). */
interface PropagationResources {
    readonly findKernel: ComputeKernel;
    readonly argsKernel: ComputeKernel;
    readonly propagateKernel: ComputeKernel;
    readonly writeKernel: ComputeKernel;
    /** Per update-list slot: a key's change this update (KEY_DELTA_HEADER); written by ddgi/blend. */
    readonly keyDeltas: GPUBuffer;
    /** The compacted list of fine probes and its scratch of new irradiance texels. */
    readonly list: GPUBuffer;
    readonly scratch: GPUBuffer;
    /** The list's length (PROPAGATION_STATE_WORD.listed). Read by the kernels directly, not as indirect dispatch arguments: each empty indirect dispatch cost about 0.022 ms (ddgi.md). */
    readonly state: GPUBuffer;
    readonly findGroup: GPUBindGroup;
    readonly argsGroup: GPUBindGroup;
    readonly propagateGroup: GPUBindGroup;
    readonly writeGroup: GPUBindGroup;
}

/** Where the warm-up is (see the file comment). */
type WarmUpPhase = 'waiting for trees' | 'warming' | 'done';

const COMPUTE = SHADER_STAGE_COMPUTE;
const SAMPLED_ARRAY = { sampleType: 'float', viewDimension: '2d-array' } as const;

/** `ddgi/trace` group 1 (wgsl/trace.ts). */
const TRACE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/trace group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 4, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 5, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 6, visibility: COMPUTE, sampler: { type: 'filtering' } },
        // The volume table, for bounce light from every volume (rgba32float, read with textureLoad).
        { binding: 8, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
    ],
};

/**
 * `ddgi/trace` group 1 with direct light deferred: binding 2 is the whole ray
 * scratch, binding 7 the round (a 16-byte slot of the rounds buffer, chosen
 * with a dynamic offset).
 */
const DEFERRED_TRACE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/trace (deferred direct light) group 1',
    entries: [...TRACE_GROUP.entries, { binding: 7, visibility: COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: TRACE_ROUND_BYTES } }],
};

/** `ddgi/blend` group 1 (wgsl/blend.ts). */
const BLEND_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/blend group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 6, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 7, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 8, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 9, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 10, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 11, visibility: COMPUTE, buffer: { type: 'uniform' } },
    ],
};

/** `ddgi/write`'s irradiance-tile entry point, propagation's write (wgsl/write.ts). */
const WRITE_TILES_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/propagate-write group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: COMPUTE, storageTexture: { access: 'write-only', format: IRRADIANCE_FORMAT, viewDimension: '2d-array' } },
        { binding: 7, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
    ],
};

/** `ddgi/propagate-find` group 1 (wgsl/propagate.ts). */
const PROPAGATE_FIND_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/propagate-find group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 10, visibility: COMPUTE, texture: SAMPLED_ARRAY },
    ],
};

/** `ddgi/propagate-args` group 1 (wgsl/propagate.ts). */
const PROPAGATE_ARGS_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/propagate-args group 1',
    entries: [{ binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } }],
};

/** `ddgi/propagate` group 1 (wgsl/propagate.ts): 6 storage buffers. */
const PROPAGATE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/propagate group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 8, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 9, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 10, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 11, visibility: COMPUTE, sampler: { type: 'filtering' } },
    ],
};

/** `ddgi/write` group 1 (wgsl/write.ts). */
function writeGroupLayout(distanceFormat: DistanceFormat): GPUBindGroupLayoutDescriptor {
    const storage = (format: GPUTextureFormat, viewDimension: GPUTextureViewDimension = '2d-array'): GPUStorageTextureBindingLayout => ({ access: 'write-only', format, viewDimension });
    return {
        label: 'ddgi/write group 1',
        entries: [
            { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: COMPUTE, storageTexture: storage(IRRADIANCE_FORMAT) },
            { binding: 4, visibility: COMPUTE, storageTexture: storage(distanceFormat) },
            { binding: 5, visibility: COMPUTE, storageTexture: storage(PROBE_DATA_FORMAT) },
            { binding: 6, visibility: COMPUTE, storageTexture: storage('rgba32float', '2d') },
        ],
    };
}

/** `ddgi/mark` group 1 (wgsl/mark.ts). */
const MARK_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/mark group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 5, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 6, visibility: COMPUTE, sampler: { type: 'filtering' } },
    ],
};

/** `ddgi/schedule-*` group 1 (wgsl/schedule.ts): the same for all four kernels. */
const SCHEDULE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/schedule group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: COMPUTE, buffer: { type: 'storage' } },
    ],
};

/** `ddgi/compose` group 1 (wgsl/compose.ts): the same for its three entry points. */
const COMPOSE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/compose group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 4, visibility: COMPUTE, storageTexture: { access: 'write-only', format: IRRADIANCE_FORMAT, viewDimension: '2d-array' } },
        { binding: 5, visibility: COMPUTE, buffer: { type: 'uniform' } },
    ],
};

/** `ddgi/scroll`'s reset (wgsl/scroll.ts): history, feedback requests, probe data and points, written. */
const SCROLL_RESET_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/scroll-reset group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: COMPUTE, storageTexture: { access: 'write-only', format: PROBE_DATA_FORMAT, viewDimension: '2d-array' } },
        { binding: 4, visibility: COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' } },
    ],
};

/** `ddgi/scroll`'s valid extent (wgsl/scroll.ts): the probe data, read, and the extent. */
const SCROLL_EXTENT_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/scroll-extent group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 5, visibility: COMPUTE, texture: SAMPLED_ARRAY },
        { binding: 6, visibility: COMPUTE, buffer: { type: 'storage' } },
    ],
};

/** `ddgi/census` group 1 (wgsl/census.ts). */
const CENSUS_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/census group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: COMPUTE, texture: SAMPLED_ARRAY },
    ],
};

export class DDGITechnique implements GITechnique<DDGISettings> {
    readonly id = 'ddgi';
    readonly requirements: TechniqueRequirements = { queries: ['rays'] };
    readonly settings: DDGISettings;
    readonly debugViews: readonly DebugView[];
    /** How renderers read the field (shading.ts). Its textures exist after `prepare`. */
    readonly shading: DDGIShadingModule;
    /**
     * Undefined: the platform measures DDGI's cost per probe itself, from the
     * `ddgi/*` pass timings and the probes it handed out (gi-platform.ts).
     */
    readonly msPerUnit: number | undefined = undefined;

    private readonly options: DDGITechniqueOptions;
    private context: TechniqueContext | null = null;
    /** Each volume's resources after `prepare`, highest priority first (volumes.ts); empty before. */
    private volumes: readonly PreparedResources[] = [];
    private readonly uniformData = new Float32Array(UNIFORM_WORDS);
    private readonly uniformWords = new Uint32Array(this.uniformData.buffer);
    /** `ddgi/scroll`'s uniforms (`DdgiScroll`), filled per camera-following volume. */
    private readonly scrollData = new Float32Array(SCROLL_UNIFORM_WORDS);
    private readonly scrollWords = new Int32Array(this.scrollData.buffer);
    /** The volume table as last written to the GPU (a row per volume), so it is written only when a bias changes. */
    private volumeTableRows = new Float32Array(0);
    /** The rate the platform last asked Budget v2's floor for (`minUnitsPerFrame`); null under a units budget. */
    private platformFloorRate: WorkFloorRate | null = null;
    /** Shares each frame's probes between the volumes, carrying fractions over frames (volumes.ts); new at `prepare`. */
    private unitSharer = new UnitSharer([]);
    /** Frames since `prepare` whose changes ran `ddgi/mark`, and the entries in the latest one (CPU-side counters). */
    private markFrames = 0;
    private latestMarkEntries = 0;
    /**
     * The technique's own clock in seconds: the sum of every frame's
     * `deltaTime` since `prepare` (each at most MAX_CLOCK_STEP_SECONDS),
     * wrapped to 0 every CLOCK_WRAP_SECONDS so the kernel's f32 times stay
     * precise. Adaptive hysteresis measures the time between a probe's
     * updates on it.
     */
    private clockSeconds = 0;
    /** Recent `deltaTime`s above 0, for the sweep counter. */
    private readonly frameTimes: number[] = [];
    /** The warm-up (see the file comment): its phase and when it started warming (wall clock, ms); each volume counts its own probe updates. */
    private warmUp: { phase: WarmUpPhase; startedAt: number | null } = { phase: 'done', startedAt: null };
    /** `resetHistory` asked for the history and the feedback requests to be cleared, in the next encode. */
    private historyClearPending = false;
    /** The objects that moved lately, for the mover mask (mover-table.ts). */
    private readonly movers = new MoverTable();
    /** The layers' weights as last written to the GPU (light groups), 4 floats a layer. */
    private readonly layerWeights = new Float32Array(MAX_IRRADIANCE_LAYERS * 4);
    /** Compose every tile in the next frame: after `prepare`, when the composed atlas starts empty. */
    private composeAllPending = false;
    /** Frames since `prepare` that composed every tile (a weight change, or the first frame). */
    private fullComposeFrames = 0;
    /**
     * With a sun layer: the light it held at the end of the last frame (the
     * scene's first directional light then), so `ddgi/mark` sees the sun
     * becoming another light (mark-table.ts, `SunLayerMarks`).
     */
    private sunInLayer: LightHandle | null = null;
    private disposed = false;

    constructor(settings: DDGISettingsInput, options: DDGITechniqueOptions = {}) {
        this.settings = withDefaults(settings);
        this.options = options;
        this.debugViews = [
            {
                kind: 'points',
                id: DDGI_DEBUG_VIEW.points,
                label: 'Probes',
                description: 'Each probe as a dot where it really sits (after relocation), coloured by its state.',
                texture: () => {
                    const first = this.firstVolume;
                    if (!first) throw new Error('@pmndrs/ddgi: the probes debug view exists after prepare');
                    return first.points;
                },
                // Every volume's probes, one volume after another (volumes.ts, probeBase).
                count: () => this.firstVolume?.volumePlan.probeCount ?? 0,
                states: PROBE_STATE_LABELS,
            },
            // Drawn through the shading module's giDebugColor (wgsl/shading.ts).
            {
                kind: 'screen',
                id: DDGI_DEBUG_VIEW.giOnly,
                label: 'GI only',
                description: 'The indirect light alone, as it would light a plain white surface.',
            },
            {
                kind: 'screen',
                id: DDGI_DEBUG_VIEW.coverage,
                label: 'Coverage',
                description: 'Green where DDGI answers fully; turning red near the edge of the probe volume and where no probe can see the surface, where the renderer uses its own environment light instead.',
            },
            {
                kind: 'screen',
                id: DDGI_DEBUG_VIEW.probeStates,
                label: 'Probes near this point',
                description: 'The 8 probes around each point, coloured by state and blended by distance: green active, red off (inside geometry), blue off (no surface nearby), yellow not updated yet.',
            },
        ];
        this.shading = new DDGIShadingModule({
            fieldTextures: () => {
                const first = this.firstVolume;
                if (!first) throw new Error('@pmndrs/ddgi: the DDGI shading module has textures after prepare (await gi.prepare())');
                // With light groups, renderers read the composed atlas: one block, as without them.
                return { irradiance: first.composed ?? first.irradiance, distance: first.distance, probeData: first.probeData, volumeTable: first.volumeTable };
            },
        });
    }

    attach(context: TechniqueContext): void {
        if (this.context) throw new Error('@pmndrs/ddgi: a DDGI technique can be attached to one platform only');
        this.context = context;
    }

    /**
     * The highest-priority volume's resources, null before `prepare`. The
     * shared textures, kernels and settings derived at `prepare` are read
     * from it (they are the same objects in every volume's).
     */
    private get firstVolume(): PreparedResources | null {
        return this.volumes[0] ?? null;
    }

    /** Which phase-B reduction the kernels use (after `prepare`). */
    get reduction(): PhaseBReduction | undefined {
        return this.firstVolume?.reduction;
    }

    /** Where direct light is computed (after `prepare`): `deferred` to the platform's `gi/direct`, or `inline`. */
    get directLight(): DirectLightPath | undefined {
        return this.firstVolume?.directLight;
    }

    /** With direct light deferred, how `ddgi/trace` fills the work list (after `prepare`; null when inline). */
    get directWorkList(): DirectWorkList | null | undefined {
        return this.firstVolume?.directWorkList;
    }

    async prepare(): Promise<void> {
        const context = this.context;
        if (!context) throw new Error('@pmndrs/ddgi: attach the DDGI technique (gi.addTechnique) before prepare');
        if (this.disposed) throw new Error('@pmndrs/ddgi: this DDGI technique was disposed');
        const device = context.device;

        // The volumes in priority order, each with its layers in the shared atlases (volumes.ts).
        const volumePlan = planVolumes(volumeSettingsList(this.settings), device.limits.maxTextureArrayLayers);
        const irradianceTexels = positiveInteger(this.settings.irradianceTexels, 'irradianceTexels');
        const distanceTexels = positiveInteger(this.settings.distanceTexels, 'distanceTexels');
        // Light groups (T2.5): one irradiance layer per group and one for the environment, or one layer with groups off;
        // and the sun's own layer last (T2.6b).
        const plan = layerPlan(this.settings.lightGroups, this.settings.sunLayer === true);
        const layerCount = plan.layers.length;
        const layout = fieldLayout({ counts: volumePlan.atlasCounts }, irradianceTexels, distanceTexels, distanceFormatFor(device.features as ReadonlySet<string>), layerCount);
        checkFieldLimits(layout, device.limits);
        const points = pointsSize(volumePlan.probeCount, device.limits.maxTextureDimension2D);
        if (points.height > device.limits.maxTextureDimension2D) throw new GILimitError('maxTextureDimension2D', points.height, device.limits.maxTextureDimension2D);
        const scratch = scratchLayout(irradianceTexels, distanceTexels, layerCount);
        // No probe updates twice in a frame, so more slots than a volume has probes would never be used.
        const maxProbesSetting = positiveInteger(this.settings.maxProbesPerFrame, 'maxProbesPerFrame');
        const maxProbesPerFrame = (grid: ProbeGrid): number => Math.min(maxProbesSetting, grid.probeCount);
        const hasSubgroups = device.features.has('subgroups');
        const reduction = this.options.reduction ?? (hasSubgroups ? 'subgroups' : 'workgroup');
        if (reduction === 'subgroups' && !hasSubgroups) throw new Error("@pmndrs/ddgi: the 'subgroups' reduction needs a device with the 'subgroups' feature");
        const directLight = this.options.directLight ?? 'deferred';
        const directWorkList = directLight === 'deferred' ? (this.options.directWorkList ?? context.shade.direct.defaultWorkList) : null;
        // The inline trace (kept for comparison) adds every light's direct light into one result.
        if (!directWorkList && layerCount > 1) throw new Error("@pmndrs/ddgi: light groups and the sun layer need direct light deferred to gi/direct (directLight: 'deferred', the default)");
        // Throws for a 'subgroups' work list on a device without them.
        const traceCode = directWorkList ? deferredTraceKernelWgsl(context.shade.direct.traceWgsl('ddgiRayScratch', directWorkList), plan) : DDGI_TRACE_WGSL;

        // Compile first: if a kernel fails, the old resources stay usable.
        const scheduleCode = scheduleKernelWgsl(reduction);
        const markCode = markKernelWgsl(reduction);
        const writeCode = writeKernelWgsl({ irradianceTexels, distanceTexels, scratch, distanceFormat: layout.distanceFormat });
        // The kernels take one list size: the largest volume's, which every volume's list is made to hold.
        const propagationCapacity = Math.max(...volumePlan.volumes.map((volume) => propagationListCapacity(volume.grid.counts, maxProbesPerFrame(volume.grid))));
        const propagateCode = propagateKernelWgsl({ irradianceTexels, irradianceLayers: layerCount, reduction, listCapacity: propagationCapacity, sunLayer: plan.sunLayer });
        const composeCode = plan.composed ? composeKernelWgsl({ irradianceTexels, irradianceLayers: layerCount }) : null;
        const composeKernel = (entryPoint: string, name: string): Promise<ComputeKernel | null> =>
            composeCode ? context.kernels.computeKernel({ label: `ddgi/compose-${name}`, modules: [], code: composeCode, entryPoint, ownGroups: [COMPOSE_GROUP] }) : Promise.resolve(null);
        const propagateKernel = (entryPoint: string, name: string, group: GPUBindGroupLayoutDescriptor): Promise<ComputeKernel> =>
            context.kernels.computeKernel({ label: `ddgi/${name} (${reduction})`, modules: [], code: propagateCode, entryPoint, ownGroups: [group] });
        const scheduleKernel = (entryPoint: string, name: string): Promise<ComputeKernel> =>
            context.kernels.computeKernel({ label: `ddgi/schedule-${name} (${reduction})`, modules: [], code: scheduleCode, entryPoint, ownGroups: [SCHEDULE_GROUP] });
        // ddgi/scroll only when a volume follows the camera, so fixed volumes compile what they did before.
        const scrollKernelsReady = volumePlan.volumes.some((volume) => volume.followsCamera)
            ? Promise.all([
                  context.kernels.computeKernel({ label: 'ddgi/scroll-reset', modules: [], code: DDGI_SCROLL_WGSL, entryPoint: SCROLL_ENTRY_POINTS.reset, ownGroups: [SCROLL_RESET_GROUP] }),
                  context.kernels.computeKernel({ label: 'ddgi/scroll-extent', modules: [], code: DDGI_SCROLL_WGSL, entryPoint: SCROLL_ENTRY_POINTS.extent, ownGroups: [SCROLL_EXTENT_GROUP] }),
              ]).then(([reset, extent]) => ({ reset, extent }))
            : Promise.resolve(null);
        const [
            traceKernel,
            blendKernel,
            writeKernel,
            censusKernel,
            markKernel,
            markClockKernel,
            scheduleCount,
            scheduleSelect,
            scheduleWrite,
            scheduleFill,
            propagateFind,
            propagateArgs,
            propagatePropagate,
            propagateWrite,
            composeUpdated,
            composeListed,
            composeAll,
        ] = await Promise.all([
            context.kernels.computeKernel({
                label: directWorkList ? `ddgi/trace (deferred direct light, ${directWorkList})` : 'ddgi/trace (inline direct light)',
                // Hit shading brings the scene copy and rays in as its dependencies.
                modules: [context.shade.module],
                code: traceCode,
                entryPoint: 'ddgiTrace',
                ownGroups: [directWorkList ? DEFERRED_TRACE_GROUP : TRACE_GROUP],
            }),
            context.kernels.computeKernel({
                label: `ddgi/blend (${reduction})`,
                modules: [],
                code: blendKernelWgsl({ irradianceTexels, distanceTexels, scratch, reduction, sunLayer: plan.sunLayer }),
                entryPoint: 'ddgiBlend',
                ownGroups: [BLEND_GROUP],
            }),
            context.kernels.computeKernel({
                label: 'ddgi/write',
                modules: [],
                code: writeCode,
                entryPoint: WRITE_ENTRY_POINTS.all,
                ownGroups: [writeGroupLayout(layout.distanceFormat)],
            }),
            context.kernels.computeKernel({
                label: 'ddgi/census',
                modules: [],
                code: DDGI_CENSUS_WGSL,
                entryPoint: 'ddgiCountStates',
                ownGroups: [CENSUS_GROUP],
            }),
            context.kernels.computeKernel({
                label: `ddgi/mark (${reduction})`,
                // The point and spot falloff hit shading uses, and nothing it binds.
                modules: [context.shade.lightFalloff, context.shade.skyChange],
                code: markCode,
                entryPoint: MARK_ENTRY_POINTS.mark,
                ownGroups: [MARK_GROUP],
            }),
            context.kernels.computeKernel({
                label: 'ddgi/mark-clock',
                modules: [context.shade.lightFalloff, context.shade.skyChange],
                code: markCode,
                entryPoint: MARK_ENTRY_POINTS.clock,
                ownGroups: [MARK_GROUP],
            }),
            scheduleKernel(SCHEDULE_ENTRY_POINTS.count, 'count'),
            scheduleKernel(SCHEDULE_ENTRY_POINTS.select, 'select'),
            scheduleKernel(SCHEDULE_ENTRY_POINTS.write, 'write'),
            scheduleKernel(SCHEDULE_ENTRY_POINTS.fill, 'fill'),
            propagateKernel(PROPAGATE_ENTRY_POINTS.find, 'propagate-find', PROPAGATE_FIND_GROUP),
            propagateKernel(PROPAGATE_ENTRY_POINTS.args, 'propagate-args', PROPAGATE_ARGS_GROUP),
            propagateKernel(PROPAGATE_ENTRY_POINTS.propagate, 'propagate', PROPAGATE_GROUP),
            context.kernels.computeKernel({
                label: 'ddgi/propagate-write',
                modules: [],
                code: writeCode,
                entryPoint: WRITE_ENTRY_POINTS.irradianceTiles,
                ownGroups: [WRITE_TILES_GROUP],
            }),
            composeKernel(COMPOSE_ENTRY_POINTS.updated, 'updated'),
            composeKernel(COMPOSE_ENTRY_POINTS.listed, 'listed'),
            composeKernel(COMPOSE_ENTRY_POINTS.all, 'all'),
            // gi/direct (also for the inline path: its ray scratch comes from the same place). Last, so it sits past the named kernels.
            context.shade.direct.prepare(),
        ]);
        const scrollKernels = await scrollKernelsReady;
        if (this.disposed) return;

        this.releaseResources();
        const atlasUsage = TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.copySrc;
        const arrayTexture = (label: string, size: FieldLayout['irradiance'], format: GPUTextureFormat, extraUsage = 0): GPUTexture =>
            device.createTexture({
                label: `ddgi ${label}`,
                size: { width: size.width, height: size.height, depthOrArrayLayers: size.layers },
                format,
                usage: atlasUsage | extraUsage,
                // Compatibility-mode devices fix a texture's binding dimension
                // when it is created, and a one-layer texture would default to
                // 2d. Every DDGI binding reads these as texture_2d_array.
                textureBindingViewDimension: '2d-array',
            });
        // Shared by every volume (volumes.ts). The layers side by side (one block with groups off), and the composed atlas
        // renderers sample when there are several.
        const irradiance = arrayTexture(layerCount > 1 ? 'irradiance layers' : 'irradiance', layout.layeredIrradiance, IRRADIANCE_FORMAT);
        const composed = plan.composed ? arrayTexture('composed irradiance', layout.irradiance, IRRADIANCE_FORMAT) : null;
        const distance = arrayTexture('distance', layout.distance, layout.distanceFormat);
        const probeData = arrayTexture('probe data', layout.probeData, PROBE_DATA_FORMAT, TEXTURE_USAGE.copyDst);
        const pointsTexture = device.createTexture({ label: 'ddgi points', size: points, format: 'rgba32float', usage: atlasUsage | TEXTURE_USAGE.copyDst });
        const movers = device.createBuffer({ label: 'ddgi movers', size: MOVER_TABLE_WORDS * 4, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        // A row per volume, in priority order. The CPU writes the sampling parameters and the window (when a bias changes
        // or a window moves); a camera-following volume's valid extent is copied in by ddgi/scroll (m3-moving.md §4.2).
        const volumeTable = device.createTexture({
            label: 'ddgi volume table',
            size: { width: VOLUME_TABLE_TEXELS, height: volumePlan.volumes.length },
            format: VOLUME_TABLE_FORMAT,
            usage: TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.copyDst | TEXTURE_USAGE.copySrc,
        });
        const composeWeights = composed ? device.createBuffer({ label: 'ddgi compose weights', size: COMPOSE_WEIGHTS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst }) : null;
        const sampler = device.createSampler({ label: 'ddgi atlases', magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
        const shared: SharedResources = {
            layout,
            scratch,
            irradianceTexels,
            distanceTexels,
            reduction,
            directLight,
            directWorkList,
            plan,
            volumePlan,
            irradiance,
            composed,
            composeWeights,
            distance,
            probeData,
            volumeTable,
            points: pointsTexture,
            sampler,
            movers,
            traceKernel,
            blendKernel,
            writeKernel,
            censusKernel,
            markKernel,
            markClockKernel,
            scheduleKernels: { count: scheduleCount, select: scheduleSelect, write: scheduleWrite, fill: scheduleFill },
            scrollKernels,
            // Deferred: a DdgiTraceRound per round, each at a uniform-offset boundary for its dynamic offset.
            traceRoundStride: Math.max(TRACE_ROUND_BYTES, device.limits.minUniformBufferOffsetAlignment),
            sharedBytes:
                fieldBytes(layout) +
                (composeWeights?.size ?? 0) +
                points.width * points.height * POINT_TEXEL_BYTES +
                movers.size +
                volumePlan.volumes.length * VOLUME_TABLE_TEXELS * VOLUME_TABLE_TEXEL_BYTES,
        };
        const kernels: VolumeKernels = {
            propagation: { find: propagateFind, args: propagateArgs, propagate: propagatePropagate, write: propagateWrite },
            compose: composeUpdated && composeListed && composeAll ? { updated: composeUpdated, listed: composeListed, all: composeAll } : null,
        };
        this.volumes = volumePlan.volumes.map((volume, index) =>
            this.createVolumeResources(context, shared, kernels, volume, index, { maxProbesPerFrame: maxProbesPerFrame(volume.grid), propagationCapacity }),
        );
        // The table is written from the settings at the first frame (refreshVolumeTable), and here, whole, so it is
        // complete before any: every row's valid extent too, which only ddgi/scroll changes after this.
        this.volumeTableRows = new Float32Array(0);
        this.unitSharer = new UnitSharer(this.volumes.map((volume) => volume.grid.probeCount));
        this.refreshVolumeTable(context, true);
        this.markFrames = 0;
        this.latestMarkEntries = 0;
        this.clockSeconds = 0;
        this.historyClearPending = false;
        this.movers.clear();
        // The composed atlas starts empty: the first frame composes every tile, with the weights as they are then.
        this.layerWeights.fill(Number.NaN);
        this.composeAllPending = composed !== null;
        this.fullComposeFrames = 0;
        // The field is new, so the sun layer holds nothing yet to be wrong about.
        this.sunInLayer = context.scene.firstDirectionalLight?.light ?? null;
        this.startWarmUp();
        this.markAllNotUpdated();
    }

    /**
     * One volume's own buffers and bind groups (volumes.ts): its uniforms,
     * update list, scratch, history, scheduler, census, propagation, ray
     * scratch and mark table, bound with the shared atlases and kernels.
     * Volume 0's labels are the single volume's; the others' say which.
     */
    private createVolumeResources(
        context: TechniqueContext,
        shared: SharedResources,
        kernels: VolumeKernels,
        volume: PlacedVolume,
        index: number,
        sizes: { readonly maxProbesPerFrame: number; readonly propagationCapacity: number },
    ): PreparedResources {
        const device = context.device;
        const { grid } = volume;
        const { maxProbesPerFrame, propagationCapacity } = sizes;
        const { irradianceTexels, scratch, irradiance, distance, probeData, sampler } = shared;
        const layerCount = shared.plan.layers.length;
        const label = (what: string): string => (index === 0 ? `ddgi ${what}` : `ddgi ${what} (volume ${index})`);

        const uniforms = device.createBuffer({ label: label('uniforms'), size: UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        // Written by the GPU scheduler.
        const updateList = device.createBuffer({ label: label('update list'), size: maxProbesPerFrame * 4, usage: BUFFER_USAGE.storage });
        const scratchBuffer = device.createBuffer({ label: label('scratch'), size: maxProbesPerFrame * scratch.strideWords * 4, usage: BUFFER_USAGE.storage });
        const census = device.createBuffer({ label: label('census'), size: CENSUS_WORDS * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        // Starts at zero (WebGPU clears new buffers): no probe remembers anything yet.
        // COPY_SRC so debug tools and tests can copy it out (DDGIField.probeHistory).
        const probeHistory = device.createBuffer({ label: label('probe history'), size: grid.probeCount * PROBE_HISTORY_BYTES, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        // The scheduler's cursor starts at the order's first entry (probe 0) and every count at 0: new buffers are zero.
        const scheduleState = device.createBuffer({ label: label('schedule state'), size: SCHEDULE_STATE_WORDS * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        const feedbackRequests = device.createBuffer({ label: label('feedback requests'), size: grid.probeCount * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copyDst });
        // Propagation (T2.4c): a key-delta entry per update-list slot, the list of fine probes, its scratch and its length.
        const keyDeltas = device.createBuffer({ label: label('key deltas'), size: maxProbesPerFrame * keyDeltaStrideWords(irradianceTexels, layerCount) * 4, usage: BUFFER_USAGE.storage });
        const propagationList = device.createBuffer({ label: label('propagation list'), size: propagationCapacity * 4, usage: BUFFER_USAGE.storage });
        const propagationScratch = device.createBuffer({
            label: label('propagation scratch'),
            size: propagationCapacity * propagationScratchStrideWords(irradianceTexels, layerCount) * 4,
            usage: BUFFER_USAGE.storage,
        });
        const propagationState = device.createBuffer({
            label: label('propagation state'),
            size: PROPAGATION_STATE_WORDS * 4,
            usage: BUFFER_USAGE.storage | BUFFER_USAGE.copyDst | BUFFER_USAGE.copySrc,
        });
        const censusStaging = Array.from({ length: CENSUS_STAGING_BUFFERS }, (_, stagingIndex) => ({
            buffer: device.createBuffer({ label: label(`census readback ${stagingIndex}`), size: CENSUS_WORDS * 4, usage: BUFFER_USAGE.mapRead | BUFFER_USAGE.copyDst }),
            state: 'free' as CensusStaging['state'],
            listedProbes: 0,
            frame: 0,
        }));
        // Deferred: a DdgiTraceRound per round. Each volume has its own, because the rounds are written with
        // queue.writeBuffer, which lands before the frame's commands run: a shared buffer would hold the last volume's.
        const traceRounds = shared.directWorkList
            ? device.createBuffer({ label: label('trace rounds'), size: Math.ceil(maxProbesPerFrame / DIRECT_ROUND_PROBES) * shared.traceRoundStride, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst })
            : null;

        // Array views explicitly: a one-layer texture's default view would be 2d, not 2d-array.
        const arrayView = (target: GPUTexture): GPUTextureView => target.createView({ dimension: '2d-array' });
        const writeGroup = device.createBindGroup({
            label: 'ddgi/write group 1',
            layout: shared.writeKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: updateList } },
                { binding: 2, resource: { buffer: scratchBuffer } },
                { binding: 3, resource: arrayView(irradiance) },
                { binding: 4, resource: arrayView(distance) },
                { binding: 5, resource: arrayView(probeData) },
                { binding: 6, resource: shared.points.createView() },
            ],
        });
        const censusGroup = device.createBindGroup({
            label: 'ddgi/census group 1',
            layout: shared.censusKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: census } },
                { binding: 2, resource: arrayView(probeData) },
            ],
        });
        // The four kernels' layouts are made from one descriptor, so one bind group fits them all.
        const scheduleGroup = device.createBindGroup({
            label: 'ddgi/schedule group 1',
            layout: shared.scheduleKernels.count.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: probeHistory } },
                { binding: 2, resource: { buffer: feedbackRequests } },
                { binding: 3, resource: { buffer: scheduleState } },
                { binding: 4, resource: { buffer: updateList } },
            ],
        });

        const propagationKernels = kernels.propagation;
        const propagation: PropagationResources = {
            findKernel: propagationKernels.find,
            argsKernel: propagationKernels.args,
            propagateKernel: propagationKernels.propagate,
            writeKernel: propagationKernels.write,
            keyDeltas,
            list: propagationList,
            scratch: propagationScratch,
            state: propagationState,
            findGroup: device.createBindGroup({
                label: 'ddgi/propagate-find group 1',
                layout: propagationKernels.find.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: uniforms } },
                    { binding: 1, resource: { buffer: probeHistory } },
                    { binding: 2, resource: { buffer: census } },
                    { binding: 3, resource: { buffer: propagationState } },
                    { binding: 4, resource: { buffer: propagationList } },
                    { binding: 10, resource: arrayView(probeData) },
                ],
            }),
            argsGroup: device.createBindGroup({
                label: 'ddgi/propagate-args group 1',
                layout: propagationKernels.args.ownLayouts[0],
                entries: [{ binding: 3, resource: { buffer: propagationState } }],
            }),
            propagateGroup: device.createBindGroup({
                label: 'ddgi/propagate group 1',
                layout: propagationKernels.propagate.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: uniforms } },
                    { binding: 1, resource: { buffer: probeHistory } },
                    { binding: 2, resource: { buffer: census } },
                    { binding: 4, resource: { buffer: propagationList } },
                    { binding: 5, resource: { buffer: keyDeltas } },
                    { binding: 6, resource: { buffer: propagationScratch } },
                    { binding: 7, resource: { buffer: propagationState } },
                    { binding: 8, resource: arrayView(irradiance) },
                    { binding: 9, resource: arrayView(distance) },
                    { binding: 10, resource: arrayView(probeData) },
                    { binding: 11, resource: sampler },
                ],
            }),
            writeGroup: device.createBindGroup({
                label: 'ddgi/propagate-write group 1',
                layout: propagationKernels.write.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: uniforms } },
                    { binding: 1, resource: { buffer: propagationList } },
                    { binding: 2, resource: { buffer: propagationScratch } },
                    { binding: 3, resource: arrayView(irradiance) },
                    { binding: 7, resource: { buffer: propagationState } },
                ],
            }),
        };

        let compose: ComposeResources | null = null;
        const composeKernels = kernels.compose;
        if (shared.composed && shared.composeWeights && composeKernels) {
            const { composed, composeWeights } = shared;
            const composeGroup = (groupLabel: string, list: GPUBuffer, listState: GPUBuffer): GPUBindGroup =>
                device.createBindGroup({
                    label: groupLabel,
                    layout: composeKernels.updated.ownLayouts[0],
                    entries: [
                        { binding: 0, resource: { buffer: uniforms } },
                        { binding: 1, resource: { buffer: list } },
                        { binding: 2, resource: { buffer: listState } },
                        { binding: 3, resource: arrayView(irradiance) },
                        { binding: 4, resource: arrayView(composed) },
                        { binding: 5, resource: { buffer: composeWeights } },
                    ],
                });
            compose = {
                updatedKernel: composeKernels.updated,
                listedKernel: composeKernels.listed,
                allKernel: composeKernels.all,
                // The update list's length is ddgi.updateCount; its state binding is unused (propagation's is bound to fill it).
                updatedGroup: composeGroup('ddgi/compose group 1 (update list)', updateList, propagationState),
                listedGroup: composeGroup('ddgi/compose group 1 (propagation list)', propagationList, propagationState),
            };
        }

        const scroll = volume.followsCamera ? this.createScrollResources(device, shared, volume, index, { probeHistory, feedbackRequests }) : null;
        const ownBytes =
            (scroll ? scroll.uniforms.size + scroll.extent.size : 0) +
            UNIFORM_BYTES +
            updateList.size +
            scratchBuffer.size +
            probeHistory.size +
            scheduleState.size +
            feedbackRequests.size +
            (traceRounds?.size ?? 0) +
            keyDeltas.size +
            propagationList.size +
            propagationScratch.size +
            propagationState.size +
            census.size * (1 + CENSUS_STAGING_BUFFERS);
        const partial = {
            ...shared,
            volume,
            grid,
            maxProbesPerFrame,
            compose,
            uniforms,
            updateList,
            scratchBuffer,
            census,
            probeHistory,
            censusStaging,
            propagation,
            scheduleGroup,
            scheduleState,
            feedbackRequests,
            writeGroup,
            censusGroup,
            traceRounds,
            stride: scrambleStride(grid.probeCount),
            ownBytes,
            scroll,
            state: {
                latestCensus: null,
                changeIntake: new ChangeIntakeBatch(),
                probesUpdated: 0,
                lastUpdateCount: 0,
                warmUpProbeUpdates: 0,
                sampleParameters: new Float32Array(VOLUME_SAMPLING_FLOATS),
            },
        };
        return {
            ...partial,
            rays: this.createRayBuffer(context, partial, Math.min(INITIAL_RAY_CAPACITY, maxProbesPerFrame * 256), effectiveRaysPerProbe(this.settings.raysPerProbe)),
            mark: this.createMarkTable(device, partial, INITIAL_MARK_TABLE_BYTES),
        };
    }

    /**
     * A camera-following volume's `ddgi/scroll` buffers and bind groups
     * (wgsl/scroll.ts). The valid extent starts as the window less a plane on
     * each side, and the first frame sets it from the camera.
     */
    private createScrollResources(
        device: GPUDevice,
        shared: SharedResources,
        volume: PlacedVolume,
        row: number,
        buffers: { readonly probeHistory: GPUBuffer; readonly feedbackRequests: GPUBuffer },
    ): ScrollResources {
        const kernels = shared.scrollKernels;
        if (!kernels) throw new Error('@pmndrs/ddgi: a camera-following volume needs the ddgi/scroll kernels');
        const window = new CameraWindow(volume.grid);
        const label = `ddgi scroll (volume ${row})`;
        const uniforms = device.createBuffer({ label: `${label} uniforms`, size: SCROLL_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const extent = device.createBuffer({ label: `${label} valid extent`, size: SCROLL_EXTENT_BYTES, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        const initial = window.initialExtent();
        device.queue.writeBuffer(extent, 0, new Float32Array([...initial.min, 0, ...initial.max, 0]));
        const arrayView = (target: GPUTexture): GPUTextureView => target.createView({ dimension: '2d-array' });
        const resetGroup = device.createBindGroup({
            label: 'ddgi/scroll-reset group 1',
            layout: kernels.reset.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: buffers.probeHistory } },
                { binding: 2, resource: { buffer: buffers.feedbackRequests } },
                { binding: 3, resource: arrayView(shared.probeData) },
                { binding: 4, resource: shared.points.createView() },
            ],
        });
        const extentGroup = device.createBindGroup({
            label: 'ddgi/scroll-extent group 1',
            layout: kernels.extent.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 5, resource: arrayView(shared.probeData) },
                { binding: 6, resource: { buffer: extent } },
            ],
        });
        return { window, row, uniforms, extent, resetGroup, extentGroup, resetEverySlot: false, extentToTarget: true };
    }

    /**
     * Each camera-following volume's window follows this frame's camera
     * (scroll.ts): a move enters planes, which `ddgi/scroll` resets this
     * frame. A teleport resets every slot and runs the warm-up again
     * (m3-moving.md §4.2), as after a scene load.
     */
    private followCamera(frame: TechniqueFrame): Map<PreparedResources, WindowMove> {
        const moves = new Map<PreparedResources, WindowMove>();
        for (const volume of this.volumes) {
            const scroll = volume.scroll;
            if (!scroll) continue;
            const move = scroll.window.follow(frame.camera.position);
            if (move.teleport) {
                scroll.resetEverySlot = true;
                scroll.extentToTarget = true;
                this.startWarmUp();
            }
            moves.set(volume, move);
        }
        return moves;
    }

    /**
     * `ddgi/scroll` for one camera-following volume (wgsl/scroll.ts), one
     * timed pass: the reset of the slots that entered (only in a frame where
     * some did), then the valid extent; then the extent is copied into the
     * volume's row of the table.
     */
    private encodeScroll(encoder: GPUCommandEncoder, context: TechniqueContext, resources: PreparedResources, move: WindowMove, frame: TechniqueFrame): void {
        const scroll = resources.scroll;
        const kernels = resources.scrollKernels;
        if (!scroll || !kernels) return;
        const { window } = scroll;
        const grid = window.grid;
        const targets = window.extentTargets(frame.camera.position);
        const everySlot = scroll.resetEverySlot;
        const data = this.scrollData;
        const words = this.scrollWords;
        data.fill(0);
        data.set(window.windowOrigin, SCROLL_UNIFORM_WORD.origin);
        data.set(grid.spacing, SCROLL_UNIFORM_WORD.spacing);
        words.set(grid.counts, SCROLL_UNIFORM_WORD.counts);
        words.set(window.window, SCROLL_UNIFORM_WORD.window);
        words.set(move.moved, SCROLL_UNIFORM_WORD.moved);
        data.set(targets.min, SCROLL_UNIFORM_WORD.targetMin);
        data.set(targets.max, SCROLL_UNIFORM_WORD.targetMax);
        words[SCROLL_UNIFORM_WORD.layerBase] = resources.volume.layerBase;
        words[SCROLL_UNIFORM_WORD.probeBase] = resources.volume.probeBase;
        words[SCROLL_UNIFORM_WORD.everySlotEnters] = everySlot ? 1 : 0;
        words[SCROLL_UNIFORM_WORD.extentToTarget] = scroll.extentToTarget ? 1 : 0;
        context.device.queue.writeBuffer(scroll.uniforms, 0, data);
        scroll.resetEverySlot = false;
        scroll.extentToTarget = false;

        const pass = encoder.beginComputePass({ label: 'ddgi/scroll', timestampWrites: context.timer.computePass('ddgi/scroll') });
        if (everySlot || move.probesEntering > 0) {
            const groups = Math.ceil(grid.probeCount / SCROLL_WORKGROUP_SIZE);
            const groupsX = Math.min(groups, context.device.limits.maxComputeWorkgroupsPerDimension);
            kernels.reset.bind(pass);
            pass.setBindGroup(1, scroll.resetGroup);
            pass.dispatchWorkgroups(groupsX, Math.ceil(groups / groupsX));
        }
        kernels.extent.bind(pass);
        pass.setBindGroup(1, scroll.extentGroup);
        pass.dispatchWorkgroups(SCROLL_EXTENT_SIDES);
        pass.end();
        // The two extent texels of the volume's row. One row, so bytesPerRow only has to be a valid pitch.
        encoder.copyBufferToTexture(
            { buffer: scroll.extent, bytesPerRow: 256 },
            { texture: resources.volumeTable, origin: { x: VOLUME_TABLE_TEXEL.validMin, y: scroll.row } },
            { width: VOLUME_TABLE_TEXELS - VOLUME_TABLE_TEXEL.validMin, height: 1 },
        );
    }

    /**
     * True from `prepare` (and `resetHistory`) until the warm-up is done: the
     * technique is waiting for the ray-tracing trees or making its warm-up
     * passes, and ignores the frame's budget. The platform leaves these
     * frames out of the budget's calibration. Always false with
     * `warmUpPasses` 0.
     */
    get warmingUp(): boolean {
        return this.firstVolume !== null && this.warmUp.phase !== 'done';
    }

    /**
     * Records this frame's updates; returns the probes it updated (the units
     * it used). What is the technique's comes first, once a frame: the
     * census read-backs, the warm-up, the clock, the hot movers, the sun
     * layer, the volume table and the layer weights. Then each volume, in
     * priority order, runs its own passes on its share of the frame's probes
     * (`encodeVolume`).
     */
    encode(encoder: GPUCommandEncoder, frame: TechniqueFrame): number {
        const volumes = this.volumes;
        const first = this.firstVolume;
        const context = this.context;
        if (!first || !context || this.disposed) return 0;
        for (const volume of volumes) this.collectCensus(volume);

        const warming = this.advanceWarmUp(context, first.volumePlan.probeCount);
        // Time passes whether or not this frame updates any probe; the warm-up's passes have no time between them.
        if (!warming) this.clockSeconds = (this.clockSeconds + clockStep(frame.deltaTime)) % CLOCK_WRAP_SECONDS;
        if (frame.deltaTime > 0 && Number.isFinite(frame.deltaTime)) {
            this.frameTimes.push(frame.deltaTime);
            if (this.frameTimes.length > SWEEP_FRAME_TIMES) this.frameTimes.shift();
        }
        // The hot movers follow this frame's moves first, so a mover's mask bit is known in the frame it moves.
        this.movers.update(frame.changes, frame.frame);
        if (this.movers.dirty) {
            context.device.queue.writeBuffer(first.movers, 0, this.movers.words);
            this.movers.dirty = false;
        }
        // The sun layer (T2.6b): the light it holds now, and the one it held last frame.
        const sun = context.scene.firstDirectionalLight;
        const sunMarks = first.plan.sunLayer >= 0 ? { sun: sun?.light ?? null, previousSun: this.sunInLayer } : null;
        this.sunInLayer = sun?.light ?? null;
        // Camera-following volumes move their windows with this frame's camera (scroll.ts): before the table is written,
        // and before any volume's passes, so this frame's kernels and renderers all see the moved windows.
        const windowMoves = this.followCamera(frame);
        // The biases are live settings, and renderers read them from the table; so are the windows.
        this.refreshVolumeTable(context);
        for (const [volume, move] of windowMoves) this.encodeScroll(encoder, context, volume, move, frame);
        // Light groups and the sun layer: a weight changed since the last frame (or the composed atlas is new), so every
        // tile is composed this frame. A change of the sun's colour or intensity is one, with a sun layer.
        const composeAll = this.updateLayerWeights(context, first, sun?.descriptor ?? null);

        // Each volume its floor and its share of the rest, so each is swept at the same rate (frameShares).
        const shares = this.frameShares(frame.units);
        const marks = { frames: 0, entries: 0 };
        let used = 0;
        volumes.forEach((volume, index) => {
            const { units, floor } = shares[index];
            used += this.encodeVolume(encoder, context, volume, frame, { units, floor, sunMarks, sun: sun?.descriptor ?? null, composeAll, marks });
        });
        this.latestMarkEntries = marks.entries;
        if (marks.frames > 0) this.markFrames++;
        // Every volume has cleared its history by now.
        this.historyClearPending = false;
        if (this.warmUp.phase === 'warming') this.finishWarmUpPass(volumes);
        return used;
    }

    /**
     * One volume's frame: marks, the scheduler, then trace, blend, write,
     * propagation, compose and census, on `units` probes of this volume (its
     * share of the frame's). Returns the probes it updated.
     */
    private encodeVolume(
        encoder: GPUCommandEncoder,
        context: TechniqueContext,
        resources: PreparedResources,
        frame: TechniqueFrame,
        shared: {
            readonly units: number;
            /** Budget v2's floor for this volume: the batch while a change is taken in never goes under it. */
            readonly floor: number;
            readonly sunMarks: SunLayerMarks | null;
            readonly sun: DirectionalLightDescriptor | null;
            readonly composeAll: boolean;
            /** Counted over the volumes for the counters: volumes that marked this frame, and their table entries. */
            readonly marks: { frames: number; entries: number };
        },
    ): number {
        const { state } = resources;
        const { composeAll } = shared;
        const probeCount = resources.grid.probeCount;
        // Marks land even in a frame that updates no probe: they wait in the
        // history for each probe's next update.
        // In window probes: the kernel finds each one's slot (wgsl/mark.ts).
        const markTable =
            this.settings.changeDrivenHistory !== false ? buildMarkTable(frame.changes, windowGrid(resources), this.movers, resources.plan.groupLayers, shared.sunMarks) : null;
        shared.marks.entries += markTable?.entryCount ?? 0;
        state.changeIntake.noteFrame(frame.frame, markTable !== null);
        const raysPerProbe = effectiveRaysPerProbe(this.settings.raysPerProbe);
        // No probe updates twice in a frame. While warming: a full batch, or none while the trees are built.
        // While a change is taken in, fewer than the hand-out, so the list costs what the calibrated batch does (change-intake-batch.ts).
        // Whether a single change counts is the volume's rule at the hand-out, not at the smaller batch.
        const offered = Math.min(shared.units, resources.maxProbesPerFrame, probeCount);
        const handOut =
            this.warmUp.phase === 'warming'
                ? resources.maxProbesPerFrame
                : this.warmUp.phase === 'waiting for trees'
                  ? 0
                  : state.changeIntake.batchFor(
                        offered,
                        { full: raysPerProbe, fixed: FIXED_RAYS },
                        changeServing(resources.grid, offered, this.settings).changeOutlastsWindow,
                        shared.floor,
                    );
        const count = Math.max(0, Math.min(Number.isNaN(handOut) ? 0 : Math.floor(handOut), resources.maxProbesPerFrame, probeCount));
        state.probesUpdated = count;
        if (count > 0) state.lastUpdateCount = count;
        if (count === 0 && !markTable && !this.historyClearPending && !composeAll) return 0;
        const device = context.device;
        const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;

        this.writeUniforms(resources, frame, count, raysPerProbe, shared.sun);
        device.queue.writeBuffer(resources.uniforms, 0, this.uniformData);
        if (this.historyClearPending) {
            // resetHistory: every probe starts again with nothing owed, cut or asked for.
            encoder.clearBuffer(resources.probeHistory);
            encoder.clearBuffer(resources.feedbackRequests);
        }
        // The census buffer gathers this frame's marks (ddgi/mark), rays traced and feedback (ddgi/blend); the running totals stay.
        encoder.clearBuffer(resources.census, 0, CENSUS_PER_FRAME_WORDS * 4);
        if (markTable) {
            this.encodeMark(encoder, context, resources, markTable, maxPerDimension);
            shared.marks.frames++;
        }
        if (count > 0) this.encodeSchedule(encoder, context, resources, maxPerDimension);
        if (this.warmUp.phase === 'warming') state.warmUpProbeUpdates += count;
        if (count === 0) {
            if (composeAll) this.encodeCompose(encoder, context, resources, 0, false, true, maxPerDimension);
            return 0;
        }

        const rays = resources.rays;
        const roundProbes = Math.min(count, DIRECT_ROUND_PROBES);
        // raysPerProbe is a live setting, so the work list is checked against this frame's blended rays.
        const listOutgrown = resources.directWorkList !== null && roundProbes * Math.max(0, raysPerProbe - FIXED_RAYS) > rays.scratch.itemCapacity;
        const outgrown = count * raysPerProbe > rays.capacity || listOutgrown;
        if (outgrown) {
            // Nothing recorded in this frame uses the old scratch yet, and earlier frames are submitted.
            resources.rays = this.createRayBuffer(context, resources, Math.max(count * raysPerProbe, rays.capacity * 2), raysPerProbe);
            rays.scratch.destroy();
        }

        const probesX = Math.min(count, maxPerDimension);
        const probesY = Math.ceil(count / probesX);

        // Separate passes so each is timed on its own. Passes are separate
        // usage scopes, so ddgi/write may store into the atlases the others sampled.
        if (resources.directWorkList) {
            this.encodeDeferredTrace(encoder, context, resources, count, raysPerProbe, frame.frame);
        } else {
            const trace = encoder.beginComputePass({ label: 'ddgi/trace', timestampWrites: context.timer.computePass('ddgi/trace') });
            resources.traceKernel.bind(trace);
            trace.setBindGroup(1, resources.rays.traceGroup);
            trace.dispatchWorkgroups(Math.ceil(raysPerProbe / TRACE_WORKGROUP_SIZE), probesX, probesY);
            trace.end();
        }

        const blend = encoder.beginComputePass({ label: 'ddgi/blend', timestampWrites: context.timer.computePass('ddgi/blend') });
        resources.blendKernel.bind(blend);
        blend.setBindGroup(1, resources.rays.blendGroup);
        blend.dispatchWorkgroups(probesX, probesY);
        blend.end();

        const write = encoder.beginComputePass({ label: 'ddgi/write', timestampWrites: context.timer.computePass('ddgi/write') });
        resources.writeKernel.bind(write);
        write.setBindGroup(1, resources.writeGroup);
        write.dispatchWorkgroups(probesX, probesY);
        write.end();

        const propagation = this.settings.keyProbes !== false;
        if (propagation) this.encodePropagation(encoder, context, resources, maxPerDimension);
        this.encodeCompose(encoder, context, resources, count, propagation, composeAll, maxPerDimension);
        this.encodeCensus(encoder, context, resources, maxPerDimension, frame.frame);
        return count;
    }

    /**
     * Light groups and the sun layer: writes the layers' weights when they
     * changed, and says whether this frame must compose every tile (a weight
     * changed, or the composed atlas is new). Nothing with one layer. The sun
     * layer's weight is the sun's colour × intensity now (light-groups.ts).
     */
    private updateLayerWeights(context: TechniqueContext, resources: PreparedResources, sun: DirectionalLightDescriptor | null): boolean {
        const weights = resources.composeWeights;
        if (!weights) return false;
        const sunWeight = sunLayerWeight(resources.plan, this.settings.lightGroups, sun);
        const changed = writeLayerWeights(resources.plan, this.settings.lightGroups, this.layerWeights, sunWeight);
        if (changed) context.device.queue.writeBuffer(weights, 0, this.layerWeights);
        const composeAll = changed || this.composeAllPending;
        this.composeAllPending = false;
        if (composeAll) this.fullComposeFrames++;
        return composeAll;
    }

    /**
     * `ddgi/compose` (wgsl/compose.ts), with light groups on: one pass, timed
     * as `ddgi/compose`. Every tile after a weight change; otherwise the
     * update list's tiles, and propagation's (its list's length is on the
     * GPU, so its dispatch is a fixed number of workgroups that loop, as
     * propagation's own write is). Each dispatch is its own usage scope, so
     * the compose reads what `ddgi/write` and propagation's write stored.
     */
    private encodeCompose(
        encoder: GPUCommandEncoder,
        context: TechniqueContext,
        resources: PreparedResources,
        count: number,
        propagation: boolean,
        composeAll: boolean,
        maxPerDimension: number,
    ): void {
        const compose = resources.compose;
        if (!compose) return;
        const pass = encoder.beginComputePass({ label: 'ddgi/compose', timestampWrites: context.timer.computePass('ddgi/compose') });
        if (composeAll) {
            const probes = resources.grid.probeCount;
            const groupsX = Math.min(probes, maxPerDimension);
            compose.allKernel.bind(pass);
            pass.setBindGroup(1, compose.updatedGroup);
            pass.dispatchWorkgroups(groupsX, Math.ceil(probes / groupsX));
        } else {
            if (count > 0) {
                const groupsX = Math.min(count, maxPerDimension);
                compose.updatedKernel.bind(pass);
                pass.setBindGroup(1, compose.updatedGroup);
                pass.dispatchWorkgroups(groupsX, Math.ceil(count / groupsX));
            }
            if (propagation) {
                compose.listedKernel.bind(pass);
                pass.setBindGroup(1, compose.listedGroup);
                pass.dispatchWorkgroups(PROPAGATE_WORKGROUPS);
            }
        }
        pass.end();
    }

    /**
     * Sets light group `group`'s weight (0..3, light-groups.ts): a linear RGB
     * multiplier on top of its lights' colour and intensity. Shows in the
     * next frame, composed; nothing is marked or traced again. Light groups
     * must be on (`settings.lightGroups`).
     */
    setGroupWeight(group: number, weight: ColorRGB): void {
        const lightGroups = this.settings.lightGroups;
        if (!lightGroups) throw new Error('@pmndrs/ddgi: light groups are off; set settings.lightGroups (and prepare) before setting a group weight');
        if (!Number.isInteger(group) || group < 0 || group >= MAX_LIGHT_GROUPS) throw new RangeError(`@pmndrs/ddgi: light groups are 0 to ${MAX_LIGHT_GROUPS - 1}, got ${group}`);
        while (lightGroups.weights.length <= group) lightGroups.weights.push([1, 1, 1]);
        lightGroups.weights[group] = [weight[0], weight[1], weight[2]];
    }

    /** Sets the environment layer's weight (the sky and emissive surfaces), as {@link setGroupWeight}. */
    setEnvironmentWeight(weight: ColorRGB): void {
        const lightGroups = this.settings.lightGroups;
        if (!lightGroups) throw new Error('@pmndrs/ddgi: light groups are off; set settings.lightGroups (and prepare) before setting the environment weight');
        lightGroups.environmentWeight = [weight[0], weight[1], weight[2]];
    }

    /**
     * `ddgi/trace` with direct light deferred: per round of at most
     * DIRECT_ROUND_PROBES probes, the trace (which fills the work list), then
     * the platform's `gi/direct` over that list, which also empties it for
     * the next round. The rounds' slots go in their own uniform slots, all
     * written before the host submits, each picked with a dynamic offset.
     */
    private encodeDeferredTrace(encoder: GPUCommandEncoder, context: TechniqueContext, resources: PreparedResources, count: number, raysPerProbe: number, frameNumber: number): void {
        const traceRounds = resources.traceRounds;
        if (!traceRounds) throw new Error('@pmndrs/ddgi: the deferred trace has no rounds buffer');
        const stride = resources.traceRoundStride;
        const rounds = Math.ceil(count / DIRECT_ROUND_PROBES);
        const words = new Uint32Array((rounds * stride) / 4);
        for (let round = 0; round < rounds; round++) {
            const firstSlot = round * DIRECT_ROUND_PROBES;
            words[(round * stride) / 4] = firstSlot;
            words[(round * stride) / 4 + 1] = Math.min(DIRECT_ROUND_PROBES, count - firstSlot);
        }
        context.device.queue.writeBuffer(traceRounds, 0, words);
        // Each light group's direct light into its own layer (all into the one layer with groups off), and the sun at
        // its white reference into the sun layer when there is one.
        const sunLayer = resources.plan.sunLayer >= 0 ? resources.plan.sunLayer : undefined;
        const direct = { frame: frameNumber, groupMask: DDGI_LIGHT_GROUP_MASK, groupLayers: resources.plan.groupLayers, sunLayer, owner: this.id, timer: context.timer };
        for (let round = 0; round < rounds; round++) {
            const slotCount = words[(round * stride) / 4 + 1];
            const trace = encoder.beginComputePass({ label: 'ddgi/trace', timestampWrites: context.timer.computePass('ddgi/trace') });
            resources.traceKernel.bind(trace);
            trace.setBindGroup(1, resources.rays.traceGroup, [round * stride]);
            trace.dispatchWorkgroups(Math.ceil(raysPerProbe / TRACE_WORKGROUP_SIZE), slotCount);
            trace.end();
            context.shade.direct.encode(encoder, resources.rays.scratch, direct);
        }
    }

    /**
     * Propagation (wgsl/propagate.ts): one pass of four dispatches, find
     * (one thread per probe), args (one thread), propagate and its write
     * (PROPAGATE_WORKGROUPS workgroups looping over the fine probes found).
     * A compute pass's dispatches are separate usage scopes, so the write may
     * store into the irradiance atlas propagate sampled. In a frame where no
     * key passed a change on, find's workgroups stop at once and the list is
     * empty.
     */
    private encodePropagation(encoder: GPUCommandEncoder, context: TechniqueContext, resources: PreparedResources, maxPerDimension: number): void {
        const propagation = resources.propagation;
        encoder.clearBuffer(propagation.state);
        const groups = Math.ceil(resources.grid.probeCount / PROPAGATE_FIND_WORKGROUP_SIZE);
        const groupsX = Math.min(groups, maxPerDimension);
        const pass = encoder.beginComputePass({ label: 'ddgi/propagate', timestampWrites: context.timer.computePass('ddgi/propagate') });
        propagation.findKernel.bind(pass);
        pass.setBindGroup(1, propagation.findGroup);
        pass.dispatchWorkgroups(groupsX, Math.ceil(groups / groupsX));
        propagation.argsKernel.bind(pass);
        pass.setBindGroup(1, propagation.argsGroup);
        pass.dispatchWorkgroups(1);
        // A fixed number of workgroups looping over the list (PROPAGATE_WORKGROUPS says why not indirect).
        propagation.propagateKernel.bind(pass);
        pass.setBindGroup(1, propagation.propagateGroup);
        pass.dispatchWorkgroups(PROPAGATE_WORKGROUPS);
        propagation.writeKernel.bind(pass);
        pass.setBindGroup(1, propagation.writeGroup);
        pass.dispatchWorkgroups(PROPAGATE_WORKGROUPS);
        pass.end();
    }

    /**
     * The GPU scheduler (wgsl/schedule.ts): this frame's update list, in one
     * pass of four dispatches (WebGPU orders the dispatches of a pass, and
     * one pass is one timing, `ddgi/schedule`). The histogram starts at zero.
     */
    private encodeSchedule(encoder: GPUCommandEncoder, context: TechniqueContext, resources: PreparedResources, maxPerDimension: number): void {
        encoder.clearBuffer(resources.scheduleState, SCHEDULE_STATE_WORD.histogram * 4, PRIORITY_BUCKETS * 4);
        const groups = Math.ceil(resources.grid.probeCount / SCHEDULE_WORKGROUP_SIZE);
        const groupsX = Math.min(groups, maxPerDimension);
        const groupsY = Math.ceil(groups / groupsX);
        const pass = encoder.beginComputePass({ label: 'ddgi/schedule', timestampWrites: context.timer.computePass('ddgi/schedule') });
        const { count, select, write, fill } = resources.scheduleKernels;
        count.bind(pass);
        pass.setBindGroup(1, resources.scheduleGroup);
        pass.dispatchWorkgroups(groupsX, groupsY);
        select.bind(pass);
        pass.setBindGroup(1, resources.scheduleGroup);
        pass.dispatchWorkgroups(1);
        write.bind(pass);
        pass.setBindGroup(1, resources.scheduleGroup);
        pass.dispatchWorkgroups(groupsX, groupsY);
        fill.bind(pass);
        pass.setBindGroup(1, resources.scheduleGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    /** A new warm-up (after `prepare` and `resetHistory`), or none with `warmUpPasses` 0. */
    private startWarmUp(): void {
        const passes = warmUpSettings(this.settings).passes;
        this.warmUp = { phase: passes > 0 ? 'waiting for trees' : 'done', startedAt: null };
        for (const volume of this.volumes) volume.state.warmUpProbeUpdates = 0;
    }

    /**
     * The warm-up's phase for this frame; true while it runs. It waits while
     * the scene's ray-tracing trees are being built, and stops at once when
     * `warmUpPasses` is set to 0.
     */
    private advanceWarmUp(context: TechniqueContext, probeCount: number): boolean {
        const warmUp = this.warmUp;
        if (warmUp.phase === 'done') return false;
        // Nothing to warm (no probes), or switched off meanwhile.
        if (warmUpSettings(this.settings).passes <= 0 || probeCount <= 0) {
            warmUp.phase = 'done';
            return false;
        }
        if (warmUp.phase === 'waiting for trees' && !context.queries.rays.building) {
            warmUp.phase = 'warming';
            warmUp.startedAt = performance.now();
        }
        return true;
    }

    /**
     * After a warm-up frame (each volume has counted its probes): ends the
     * warm-up once every volume has made its passes, or after its time once
     * every volume has made the fewest.
     */
    private finishWarmUpPass(volumes: readonly PreparedResources[]): void {
        const warmUp = this.warmUp;
        const settings = warmUpSettings(this.settings);
        const passes = warmUpPassesMade(volumes);
        const elapsedSeconds = warmUp.startedAt === null ? 0 : (performance.now() - warmUp.startedAt) / 1000;
        if (passes >= settings.passes || (elapsedSeconds > settings.maxSeconds && passes >= settings.minPasses)) warmUp.phase = 'done';
    }

    /** Full passes over every probe the warm-up has made (fractional while one is under way): the fewest over the volumes. */
    private get warmUpPasses(): number {
        return warmUpPassesMade(this.volumes);
    }

    /** The field's textures and sampling parameters; null before `prepare`. */
    get field(): DDGIField | null {
        const first = this.firstVolume;
        if (!first) return null;
        const volumes = this.volumes.map((volume): DDGIFieldVolume => ({
            grid: volume.grid,
            priority: volume.volume.priority,
            layerBase: volume.volume.layerBase,
            probeBase: volume.volume.probeBase,
            sampleParameters: this.volumeParameters(volume),
            probeHistory: volume.probeHistory,
            followsCamera: volume.scroll !== null,
            window: volume.scroll?.window.window ?? [0, 0, 0],
            validExtent: volume.scroll?.extent ?? null,
        }));
        return {
            grid: first.grid,
            volumes,
            volumeTable: first.volumeTable,
            layout: first.layout,
            irradiance: first.composed ?? first.irradiance,
            layeredIrradiance: first.irradiance,
            layers: first.plan,
            distance: first.distance,
            probeData: first.probeData,
            sampleParameters: volumes[0].sampleParameters,
            probeHistory: first.probeHistory,
        };
    }

    /**
     * Counters for the latest frame. Always there: "probes updated", "frames
     * with changes marked" (frames since `prepare` that ran `ddgi/mark`),
     * "changes marked this frame" (its table's entries: changed lights and
     * global changes), "warming up" (1 while `warmingUp`) and "warm-up
     * passes" (full passes the warm-up has made); "taking in a change" (1
     * from a mark until the census shows nothing owed: the batch may be
     * under the hand-out then); "seconds per full sweep"
     * (every probe updated once at this batch and the recent frame rate)
     * once frames carry a `deltaTime`. The per-state counts, rays traced,
     * "probes marked" (marks in that frame; a probe marked by two lights
     * counts twice), "probes marked (total)" (every mark since `prepare`),
     * "probes owed updates" and "probes updated at priority" (the
     * scheduler's), "feedback requests" (updates whose feedback test fired
     * that frame), "feedback requests (total)" and "probes blended (total)"
     * (since `prepare`), "key probes passing a change" and "probes
     * propagated" (that frame; T2.4c), their totals since `prepare`, and the
     * volume's "change clock" need the GPU census, which runs only when the
     * platform times passes (`gpuTiming`), and arrive a few frames late;
     * without them those counters are absent, not guessed.
     *
     * With several volumes, counts are summed over them ("volumes" says how
     * many), "taking in a change" is 1 while any volume is, and each volume's
     * change clock is reported as well ("change clock (volume i)", in
     * priority order) beside their sum ("change clock", modulo 2^32 as the
     * clocks wrap).
     */
    get counters(): Readonly<Record<string, number>> {
        const volumes = this.volumes;
        const first = this.firstVolume;
        const census = summedCensus(volumes);
        const sweep = this.sweepSeconds();
        const updates: Record<string, number> = {
            'volumes': volumes.length,
            'probes updated': volumes.reduce((sum, volume) => sum + volume.state.probesUpdated, 0),
            'frames with changes marked': this.markFrames,
            'changes marked this frame': this.latestMarkEntries,
            'warming up': this.warmingUp ? 1 : 0,
            'warm-up passes': this.warmUpPasses,
            'key probes': volumes.reduce((sum, volume) => sum + keyProbeCount(volume.grid.counts), 0),
            // Light groups (T2.5): the irradiance layers (1 with groups off), and the frames that composed every tile.
            'irradiance layers': first?.plan.layers.length ?? 0,
            // The sun layer (T2.6b): 1 when the sun has a layer of its own.
            'sun layer': (first?.plan.sunLayer ?? -1) >= 0 ? 1 : 0,
            'frames composed in full': this.fullComposeFrames,
            // §4.5's second count (change-intake-batch.ts): 1 while a change is taken in, when the batch may be under the hand-out.
            'taking in a change': volumes.some((volume) => volume.state.changeIntake.takingInChange) ? 1 : 0,
        };
        if (sweep !== undefined) updates['seconds per full sweep'] = sweep;
        // Camera-following volumes (scroll.ts), summed over them: frames their windows moved in, planes and probes
        // cleared, and teleports, since prepare.
        const windows = volumes.flatMap((volume) => (volume.scroll ? [volume.scroll.window] : []));
        if (windows.length > 0) {
            updates['window moves'] = windows.reduce((sum, window) => sum + window.moves, 0);
            updates['planes cleared (total)'] = windows.reduce((sum, window) => sum + window.planesCleared, 0);
            updates['probes cleared (total)'] = windows.reduce((sum, window) => sum + window.probesCleared, 0);
            updates['teleports'] = windows.reduce((sum, window) => sum + window.teleports, 0);
        }
        if (!census || !first?.state.latestCensus) return updates;
        return {
            ...updates,
            'probes marked': census[CENSUS_PROBES_MARKED],
            'probes marked (total)': census[CENSUS_PROBES_MARKED_TOTAL],
            'probes owed updates': census[CENSUS_SCHEDULE_STATE + SCHEDULE_STATE_WORD.owed],
            'probes updated at priority': census[CENSUS_SCHEDULE_STATE + SCHEDULE_STATE_WORD.listed],
            'feedback requests': census[CENSUS_FEEDBACK_FIRED],
            'feedback requests (total)': census[CENSUS_FEEDBACK_FIRED_TOTAL],
            'probes blended (total)': census[CENSUS_PROBES_BLENDED_TOTAL],
            'key probes passing a change': census[CENSUS_KEY_CHANGES_PASSED],
            'key changes passed (total)': census[CENSUS_KEY_CHANGES_PASSED_TOTAL],
            'probes propagated': census[CENSUS_PROBES_PROPAGATED],
            'probes propagated (total)': census[CENSUS_PROBES_PROPAGATED_TOTAL],
            // Each volume has a clock of its own: their sum (one volume's own with one), and each beside it with several.
            'change clock': census[CENSUS_CHANGE_CLOCK],
            ...(volumes.length > 1 ? Object.fromEntries(volumes.map((volume, index) => [`change clock (volume ${index})`, volume.state.latestCensus?.[CENSUS_CHANGE_CLOCK] ?? 0])) : {}),
            'rays traced': census[CENSUS_RAYS_TRACED],
            ...(first.directWorkList
                ? { 'hits lit by gi/direct': census[CENSUS_DIRECT_HITS], 'hits that did not fit gi/direct\'s list': census[CENSUS_DIRECT_DROPPED] }
                : {}),
            'probes active': census[PROBE_STATE.active],
            'probes off: inside geometry': census[PROBE_STATE.offInsideGeometry],
            'probes off: no surface nearby': census[PROBE_STATE.offNoSurfaceNearby],
            'probes not updated yet': census[PROBE_STATE.notUpdatedYet],
        };
    }

    /**
     * Budget v2's floor (`minProbesPerFrame`, settings.ts): the fewest probes
     * the platform hands out per frame under a millisecond budget. 0 before
     * `prepare`.
     */
    minUnitsPerFrame(rate: WorkFloorRate): number {
        // Kept for the batch while a change is taken in, which must not go under this floor either.
        this.platformFloorRate = rate;
        const volumes = this.volumes;
        if (volumes.length === 0) return 0;
        // With several volumes, each has its own floor (volumes.ts, volumeFloorUnits), and the technique's is their sum.
        return this.volumeFloors(rate).reduce((sum, floor) => sum + floor, 0);
    }

    /**
     * Budget v2's floor for each volume at `rate`. One volume: the
     * technique's floor, as before volumes. Several: each its own
     * (`volumeFloorUnits`), so a small volume beside a big one is still swept
     * within FLOOR_SWEEP_SECONDS and never gets 0 probes a frame.
     */
    private volumeFloors(rate: { readonly framesPerSecond: number; readonly calibratedUnits: number }): number[] {
        const volumes = this.volumes;
        const setting = this.settings.minProbesPerFrame;
        if (volumes.length === 1) return [minProbesPerFrame(setting, volumes[0].grid.probeCount, volumes[0].maxProbesPerFrame, rate)];
        const total = totalProbes(volumes);
        return volumes.map((volume) => volumeFloorUnits(setting, { probeCount: volume.grid.probeCount, maxProbesPerFrame: volume.maxProbesPerFrame }, total, rate));
    }

    /**
     * The rate Budget v2's floor is worked out at this frame: the one the
     * platform last asked about, or, when the host hands out units itself
     * (no millisecond budget, so the platform never asks), the observed frame
     * rate (60 until known) with the hand-out standing in for the calibrated
     * batch.
     */
    private floorRate(handOut: number): { readonly framesPerSecond: number; readonly calibratedUnits: number } {
        return this.platformFloorRate ?? { framesPerSecond: this.observedFramesPerSecond() ?? FLOOR_ASSUMED_FRAMES_PER_SECOND, calibratedUnits: handOut };
    }

    /**
     * This frame's probes for each volume, and each volume's floor (for the
     * batch while a change is taken in). One volume takes the hand-out as it
     * comes, as before volumes. Several: each gets its floor first, and the
     * rest of the hand-out is shared by probe count with the fractions
     * carried over frames (`UnitSharer`), so no volume is starved however
     * small its share; when the floors add up to more than the hand-out,
     * the floors win, as Budget v2's floor wins over the millisecond ceiling.
     */
    private frameShares(units: number): { readonly units: number; readonly floor: number }[] {
        const volumes = this.volumes;
        const maxTotal = totalMaxProbesPerFrame(volumes);
        const probeCount = totalProbes(volumes);
        if (volumes.length === 1) {
            const offered = Math.min(units, volumes[0].maxProbesPerFrame, probeCount);
            return [{ units, floor: this.volumeFloors(this.floorRate(offered))[0] }];
        }
        const floors = this.volumeFloors(this.floorRate(Math.min(units, maxTotal, probeCount)));
        if (units === Number.POSITIVE_INFINITY) return floors.map((floor) => ({ units, floor }));
        const rest = Math.max(0, (Number.isFinite(units) ? units : 0) - floors.reduce((sum, floor) => sum + floor, 0));
        const shares = this.unitSharer.share(rest);
        return floors.map((floor, index) => ({ units: floor + shares[index], floor }));
    }

    /** The frame rate from the recent frames' median `deltaTime`; undefined before any. */
    private observedFramesPerSecond(): number | undefined {
        if (this.frameTimes.length === 0) return undefined;
        const sorted = [...this.frameTimes].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        return median > 0 ? 1 / median : undefined;
    }

    /**
     * How long updating every probe once takes at the latest frame's batch and
     * the recent frame rate (the median `deltaTime`); undefined before frames
     * with a time, or with no probes updated.
     */
    private sweepSeconds(): number | undefined {
        const volumes = this.volumes;
        if (volumes.length === 0 || this.frameTimes.length === 0) return undefined;
        // The slowest volume's sweep (the batch is shared so they sweep at about the same rate).
        let frames = 0;
        for (const volume of volumes) {
            if (volume.state.probesUpdated <= 0 || volume.grid.probeCount <= 0) return undefined;
            frames = Math.max(frames, Math.ceil(volume.grid.probeCount / volume.state.probesUpdated));
        }
        const sorted = [...this.frameTimes].sort((a, b) => a - b);
        const frameSeconds = sorted[Math.floor(sorted.length / 2)];
        return frames * frameSeconds;
    }

    get memoryBytes(): number {
        const first = this.firstVolume;
        if (!first) return 0;
        return first.sharedBytes + this.volumes.reduce((sum, volume) => sum + volume.ownBytes + volume.rays.scratch.bytes, 0);
    }

    /**
     * Every probe goes back to "not updated yet" at its grid point: its next
     * update replaces it outright (hysteresis 0, and its adaptive-hysteresis
     * history starts again from that update), and sampling skips it until
     * then. Owed updates and feedback requests are forgotten, and the
     * warm-up runs again (unless `warmUpPasses` is 0). The round robin's
     * cursor carries on.
     */
    resetHistory(): void {
        if (!this.firstVolume) return;
        this.historyClearPending = true;
        this.movers.clear();
        for (const volume of this.volumes) {
            volume.state.changeIntake.reset();
            // Every probe is cold again: the valid extent takes its targets at the next frame (the warm-up covers it).
            if (volume.scroll) volume.scroll.extentToTarget = true;
        }
        this.startWarmUp();
        this.markAllNotUpdated();
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.releaseResources();
    }

    /** The mark table's buffer of at least `bytes`, and its bind group. */
    private createMarkTable(device: GPUDevice, resources: Omit<PreparedResources, 'rays' | 'mark'>, bytes: number): PreparedResources['mark'] {
        const buffer = device.createBuffer({ label: 'ddgi mark table', size: Math.ceil(bytes / 16) * 16, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copyDst });
        const arrayView = (target: GPUTexture): GPUTextureView => target.createView({ dimension: '2d-array' });
        const group = device.createBindGroup({
            label: 'ddgi/mark group 1',
            layout: resources.markKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: resources.uniforms } },
                { binding: 1, resource: { buffer } },
                { binding: 2, resource: { buffer: resources.probeHistory } },
                { binding: 3, resource: { buffer: resources.census } },
                { binding: 4, resource: arrayView(resources.probeData) },
                { binding: 5, resource: arrayView(resources.distance) },
                { binding: 6, resource: resources.sampler },
            ],
        });
        return { buffer, group };
    }

    /**
     * Writes this frame's mark table and dispatches `ddgi/mark`, one thread
     * per (entry, probe in its box). The table lands before the host submits
     * (queue.writeBuffer), so the dispatch reads it.
     */
    private encodeMark(encoder: GPUCommandEncoder, context: TechniqueContext, resources: PreparedResources, table: MarkTable, maxPerDimension: number): void {
        const device = context.device;
        if (table.words.byteLength > resources.mark.buffer.size) {
            // Nothing recorded this frame uses the old buffer yet (one mark a frame), and earlier frames are submitted.
            const old = resources.mark.buffer;
            resources.mark = this.createMarkTable(device, resources, Math.max(table.words.byteLength, old.size * 2));
            old.destroy();
        }
        device.queue.writeBuffer(resources.mark.buffer, 0, table.words);
        const groups = Math.ceil(table.threadCount / MARK_WORKGROUP_SIZE);
        if (groups === 0) return;
        const groupsX = Math.min(groups, maxPerDimension);
        const pass = encoder.beginComputePass({ label: 'ddgi/mark', timestampWrites: context.timer.computePass('ddgi/mark') });
        resources.markKernel.bind(pass);
        pass.setBindGroup(1, resources.mark.group);
        pass.dispatchWorkgroups(groupsX, Math.ceil(groups / groupsX));
        // The change clock takes this frame's largest change (one thread, the same group).
        resources.markClockKernel.bind(pass);
        pass.setBindGroup(1, resources.mark.group);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    /**
     * The ray scratch for `capacity` ray results (`RAY_RECORD_BYTES` each, layout.ts)
     * and, with direct light deferred, a work list with room for every
     * blended ray of one round at `raysPerProbe`; and the groups that bind it.
     */
    private createRayBuffer(context: TechniqueContext, resources: Omit<PreparedResources, 'rays' | 'mark'>, capacity: number, raysPerProbe: number): RayBuffers {
        const device = context.device;
        const deferred = resources.directWorkList !== null;
        const probesPerRound = deferred ? Math.min(DIRECT_ROUND_PROBES, Math.floor(capacity / raysPerProbe)) : 0;
        const scratch = context.shade.direct.createScratch({
            label: 'ddgi rays',
            rays: capacity,
            workItems: probesPerRound * Math.max(0, raysPerProbe - FIXED_RAYS),
            // A result per irradiance layer (light groups).
            layers: resources.plan.layers.length,
        });
        const arrayView = (target: GPUTexture): GPUTextureView => target.createView({ dimension: '2d-array' });
        const traceGroup = device.createBindGroup({
            label: 'ddgi/trace group 1',
            layout: resources.traceKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: resources.uniforms } },
                { binding: 1, resource: { buffer: resources.updateList } },
                // The deferred kernel binds the whole scratch (results and work list); M1's kernel the results.
                { binding: 2, resource: deferred ? { buffer: scratch.buffer } : scratch.rayResults },
                { binding: 3, resource: arrayView(resources.irradiance) },
                { binding: 4, resource: arrayView(resources.distance) },
                { binding: 5, resource: arrayView(resources.probeData) },
                { binding: 6, resource: resources.sampler },
                ...(deferred && resources.traceRounds ? [{ binding: 7, resource: { buffer: resources.traceRounds, size: TRACE_ROUND_BYTES } }] : []),
                { binding: 8, resource: resources.volumeTable.createView() },
            ],
        });
        const blendGroup = device.createBindGroup({
            label: 'ddgi/blend group 1',
            layout: resources.blendKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: resources.uniforms } },
                { binding: 1, resource: { buffer: resources.updateList } },
                { binding: 2, resource: scratch.rayResults },
                { binding: 3, resource: { buffer: resources.scratchBuffer } },
                { binding: 4, resource: { buffer: resources.census } },
                { binding: 5, resource: arrayView(resources.irradiance) },
                { binding: 6, resource: arrayView(resources.distance) },
                { binding: 7, resource: arrayView(resources.probeData) },
                { binding: 8, resource: { buffer: resources.probeHistory } },
                { binding: 9, resource: { buffer: resources.feedbackRequests } },
                { binding: 10, resource: { buffer: resources.propagation.keyDeltas } },
                { binding: 11, resource: { buffer: resources.movers } },
            ],
        });
        return { scratch, capacity, traceGroup, blendGroup };
    }

    /**
     * Counts probes per state into the census buffer and copies it out for
     * reading back. Only when the platform times passes: the platform makes
     * no readbacks unless its `gpuTiming` option is on, and the timer gives
     * this pass timestamp writes exactly then. Also skipped while every
     * readback buffer is still in flight.
     */
    private encodeCensus(encoder: GPUCommandEncoder, context: TechniqueContext, resources: PreparedResources, maxPerDimension: number, frame: number): void {
        const staging = resources.censusStaging.find((slot) => slot.state === 'free');
        if (!staging) return;
        const timestampWrites = context.timer.computePass('ddgi/census');
        if (!timestampWrites) return;
        const groups = Math.ceil(resources.grid.probeCount / CENSUS_WORKGROUP_SIZE);
        const groupsX = Math.min(groups, maxPerDimension);
        const pass = encoder.beginComputePass({ label: 'ddgi/census', timestampWrites });
        resources.censusKernel.bind(pass);
        pass.setBindGroup(1, resources.censusGroup);
        pass.dispatchWorkgroups(groupsX, Math.ceil(groups / groupsX));
        pass.end();
        // The scheduler's counts this frame, for the counters.
        encoder.copyBufferToBuffer(resources.scheduleState, 0, resources.census, CENSUS_SCHEDULE_STATE * 4, SCHEDULE_STATE_REPORTED_WORDS * 4);
        // gi/direct's two counts for this frame (the scratch's header), for the counters.
        if (resources.directWorkList) encoder.copyBufferToBuffer(resources.rays.scratch.buffer, resources.rays.scratch.countsOffset, resources.census, CENSUS_DIRECT_HITS * 4, 8);
        encoder.copyBufferToBuffer(resources.census, 0, staging.buffer, 0, CENSUS_WORDS * 4);
        staging.state = 'copied';
        staging.listedProbes = resources.state.probesUpdated;
        staging.frame = frame;
    }

    /**
     * Starts reading back census copies made in earlier frames. By the next
     * `encode` the host has submitted the frame that made them (the same
     * convention as the platform's GPU timer).
     */
    private collectCensus(resources: PreparedResources): void {
        for (const staging of resources.censusStaging) {
            if (staging.state !== 'copied') continue;
            staging.state = 'mapping';
            staging.buffer.mapAsync(MAP_MODE_READ).then(
                () => {
                    if (!this.volumes.includes(resources)) return;
                    const census = new Uint32Array(staging.buffer.getMappedRange().slice(0));
                    resources.state.latestCensus = census;
                    resources.state.changeIntake.noteCensus({
                        frame: staging.frame,
                        raysTraced: census[CENSUS_RAYS_TRACED],
                        listedProbes: staging.listedProbes,
                        owedProbes: census[CENSUS_SCHEDULE_STATE + SCHEDULE_STATE_WORD.owed],
                        probesPropagated: census[CENSUS_PROBES_PROPAGATED],
                        activeProbes: census[PROBE_STATE.active] + census[PROBE_STATE.notUpdatedYet],
                    });
                    staging.buffer.unmap();
                    staging.state = 'free';
                },
                () => {
                    // Destroyed while mapping (dispose or a new prepare): nothing to read.
                },
            );
        }
    }

    private releaseResources(): void {
        const volumes = this.volumes;
        const first = this.firstVolume;
        if (!first) return;
        this.volumes = [];
        // The shared textures and buffers once.
        for (const texture of [first.irradiance, first.distance, first.probeData, first.points, first.volumeTable]) texture.destroy();
        first.composed?.destroy();
        first.composeWeights?.destroy();
        first.movers.destroy();
        // Each volume's own.
        for (const resources of volumes) {
            for (const buffer of [
                resources.uniforms,
                resources.updateList,
                resources.scratchBuffer,
                resources.census,
                resources.probeHistory,
                resources.scheduleState,
                resources.feedbackRequests,
                resources.propagation.keyDeltas,
                resources.propagation.list,
                resources.propagation.scratch,
                resources.propagation.state,
                resources.mark.buffer,
            ]) {
                buffer.destroy();
            }
            resources.rays.scratch.destroy();
            resources.traceRounds?.destroy();
            resources.scroll?.uniforms.destroy();
            resources.scroll?.extent.destroy();
            for (const staging of resources.censusStaging) staging.buffer.destroy();
        }
    }

    /** Writes state 3 with zero offsets into every probe-data texel and point. Lands before the host's next submit. */
    private markAllNotUpdated(): void {
        const first = this.firstVolume;
        const context = this.context;
        if (!first || !context) return;
        const queue = context.device.queue;
        // Every layer of the shared texture, so every volume's probes (and the unused tiles of smaller volumes' layers).
        const { width, height, layers } = first.layout.probeData;
        const texels = new Uint16Array(width * height * layers * 4);
        for (let texel = 0; texel < width * height * layers; texel++) texels[texel * 4 + 3] = HALF_THREE;
        queue.writeTexture({ texture: first.probeData }, texels, { bytesPerRow: width * PROBE_DATA_TEXEL_BYTES, rowsPerImage: height }, { width, height, depthOrArrayLayers: layers });

        const points = first.points;
        const pointData = new Float32Array(points.width * points.height * 4);
        for (const resources of this.volumes) {
            const { grid, volume } = resources;
            const [nx, , nz] = grid.counts;
            // A camera-following volume's slots hold its window's probes (scroll.ts); a fixed volume's window is at 0.
            const window = resources.scroll?.window.window ?? [0, 0, 0];
            const origin = windowGrid(resources).origin;
            for (let probe = 0; probe < grid.probeCount; probe++) {
                const slot = [probe % nx, Math.floor(probe / (nx * nz)), Math.floor(probe / nx) % nz];
                const at = slot.map((index, axis) => origin[axis] + windowProbeOf(index, window[axis], grid.counts[axis]) * grid.spacing[axis]);
                pointData.set([...at, PROBE_STATE.notUpdatedYet], (volume.probeBase + probe) * 4);
            }
        }
        queue.writeTexture({ texture: points }, pointData, { bytesPerRow: points.width * POINT_TEXEL_BYTES }, { width: points.width, height: points.height });
    }

    /**
     * One volume's four sampling parameters from the live settings (its
     * volume-table row; `DdgiVolume` in wgsl/sample.ts names them), in its
     * own array, the same array every time.
     */
    private volumeParameters(resources: PreparedResources): Float32Array {
        const { grid } = resources;
        const bias = {
            normal: resolveBias(this.settings.normalBias, grid, AUTO_NORMAL_BIAS_SHARE),
            view: resolveBias(this.settings.viewBias, grid, AUTO_VIEW_BIAS_SHARE),
        };
        const texels = { irradiance: resources.irradianceTexels, distance: resources.distanceTexels };
        // A camera-following volume's origin is its window's first probe, where the window is this frame (scroll.ts).
        writeVolumeParameters(resources.state.sampleParameters, 0, resources.volume, resources.volumePlan, texels, bias, windowGrid(resources).origin);
        return resources.state.sampleParameters;
    }

    /**
     * Writes the volume table's CPU part (a row per volume, priority order:
     * the sampling parameters and the window, VOLUME_TABLE_CPU_TEXELS) when a
     * row changed since it was last written: after `prepare`, when a bias
     * setting changes, and when a camera-following volume's window moves.
     * `whole` (after `prepare`) writes the valid extents too: a fixed
     * volume's every probe, a camera-following volume's starting extent;
     * after that only `ddgi/scroll` writes them. Lands before the host's next
     * submit.
     */
    private refreshVolumeTable(context: TechniqueContext, whole = false): void {
        const volumes = this.volumes;
        const first = this.firstVolume;
        if (!first) return;
        const rows = new Float32Array(volumes.length * VOLUME_TABLE_ROW_FLOATS);
        volumes.forEach((volume, row) => {
            const offset = row * VOLUME_TABLE_ROW_FLOATS;
            rows.set(this.volumeParameters(volume), offset);
            const window = volume.scroll?.window;
            writeVolumeWindow(rows, offset, window?.window ?? [0, 0, 0], window ? window.initialExtent() : wholeGridExtent(volume.grid));
        });
        // Compare the CPU texels only: the extents in `rows` are the starting ones, not what the GPU holds.
        const cpuFloats = VOLUME_TABLE_CPU_TEXELS * 4;
        const same =
            !whole &&
            rows.length === this.volumeTableRows.length &&
            rows.every((value, index) => index % VOLUME_TABLE_ROW_FLOATS >= cpuFloats || Object.is(value, this.volumeTableRows[index]));
        if (same) return;
        const texels = whole ? VOLUME_TABLE_TEXELS : VOLUME_TABLE_CPU_TEXELS;
        // The rows' first `texels` texels, packed.
        const written = new Float32Array(volumes.length * texels * 4);
        for (let row = 0; row < volumes.length; row++) written.set(rows.subarray(row * VOLUME_TABLE_ROW_FLOATS, row * VOLUME_TABLE_ROW_FLOATS + texels * 4), row * texels * 4);
        context.device.queue.writeTexture(
            { texture: first.volumeTable },
            written,
            { bytesPerRow: texels * VOLUME_TABLE_TEXEL_BYTES, rowsPerImage: volumes.length },
            { width: texels, height: volumes.length },
        );
        this.volumeTableRows = rows;
    }

    private writeUniforms(resources: PreparedResources, frame: TechniqueFrame, count: number, raysPerProbe: number, sun: DirectionalLightDescriptor | null): void {
        const { grid } = resources;
        const data = this.uniformData;
        const words = this.uniformWords;
        // The first four vec4s: the volume, as its volume-table row; and where its window is on the lattice (scroll.ts).
        data.set(this.volumeParameters(resources), UNIFORM_WORD.volumeOrigin);
        data.set([...(resources.scroll?.window.window ?? [0, 0, 0]), 0], UNIFORM_WORD.volumeWindow);
        // Its probes' points from here on in the points debug view (volumes.ts).
        words[UNIFORM_WORD.probeBase] = resources.volume.probeBase;
        const rotation = frameRayRotation(frame.frame);
        for (let column = 0; column < 3; column++) {
            data.set(rotation.slice(column * 3, column * 3 + 3), UNIFORM_WORD.rayRotation + column * MAT3_COLUMN_WORDS);
        }
        const blend = blendSettings(this.settings);
        words[UNIFORM_WORD.adaptiveHysteresis] = blend.adaptiveHysteresis ? 1 : 0;
        data[UNIFORM_WORD.hysteresis] = blend.hysteresis;
        data[UNIFORM_WORD.clockSeconds] = this.clockSeconds;
        data[UNIFORM_WORD.inverseMemorySeconds] = blend.inverseMemorySeconds;
        data[UNIFORM_WORD.maxHistorySamples] = blend.maxSamples;
        data[UNIFORM_WORD.updatesPerSample] = blend.updatesPerSample;
        data[UNIFORM_WORD.rampUpdates] = blend.rampUpdates;
        data[UNIFORM_WORD.irradianceChangeThreshold] = blend.irradianceChangeThreshold;
        data[UNIFORM_WORD.maxBrightnessChange] = blend.maxBrightnessChange;
        data[UNIFORM_WORD.distanceSharpness] = blend.distanceSharpness;
        data[UNIFORM_WORD.maxRayDistance] = effectiveMaxRayDistance(this.settings.maxRayDistance);
        data[UNIFORM_WORD.distanceClamp] = distanceClamp(grid);
        data[UNIFORM_WORD.bounceEnergy] = blend.bounceEnergy;
        data[UNIFORM_WORD.minFrontFaceDistance] = resolveBias(this.settings.minFrontFaceDistance, grid, AUTO_MIN_FRONT_FACE_SHARE);
        words[UNIFORM_WORD.relocation] = this.settings.relocation ? 1 : 0;
        words[UNIFORM_WORD.classification] = this.settings.classification ? 1 : 0;
        words[UNIFORM_WORD.raysPerProbe] = raysPerProbe;
        words[UNIFORM_WORD.updateCount] = count;
        words[UNIFORM_WORD.frame] = frame.frame >>> 0;
        words[UNIFORM_WORD.changeDrivenHistory] = blend.changeDrivenHistory ? 1 : 0;
        data[UNIFORM_WORD.changeTolerance] = blend.changeTolerance;
        data[UNIFORM_WORD.changeUpdates] = blend.changeUpdates;
        data[UNIFORM_WORD.changeBounceShare] = blend.changeBounceShare;
        data[UNIFORM_WORD.lightChangeScale] = blend.lightChangeScale;

        // The scheduler (m2-lights.md §4.3, §6.4).
        words[UNIFORM_WORD.feedback] = blend.feedback ? 1 : 0;
        words[UNIFORM_WORD.priorityCount] = priorityCount(count, blend.priorityShare);
        words[UNIFORM_WORD.scrambleStride] = resources.stride;
        // Far probes owe fewer after a global change only where K for every probe
        // wouldn't fit in FAR_RULE_FRAMES frames at this batch (the last one, in a
        // frame that updates nothing).
        const fullUpdates = owedUpdates(blend.changeUpdates);
        const batch = count > 0 ? count : resources.state.lastUpdateCount;
        // On a big volume a big change takes the key-probe path (T2.4c,
        // m2-lights.md §4.5), or with key probes off, "far probes owe 2" (§6.4).
        const { bigVolume, keysFitWindow } = changeServing(grid, batch, this.settings);
        words[UNIFORM_WORD.keyProbes] = blend.keyProbes && bigVolume ? 1 : 0;
        data[UNIFORM_WORD.farOwedUpdates] = !blend.keyProbes && bigVolume ? Math.min(blend.farProbeUpdates, fullUpdates) : fullUpdates;
        data[UNIFORM_WORD.keyProbeBoost] = blend.keyProbeBoost;
        data[UNIFORM_WORD.keyResidualShare] = blend.keyResidualShare;
        // Every key owes its full count where all the keys' full counts fit
        // (T2.4d: the furnished room), else keyProbeUpdates away from the view (Sponza).
        data[UNIFORM_WORD.keyOwedUpdates] = keysFitWindow ? MAX_OWED_UPDATES : blend.keyProbeUpdates;
        data[UNIFORM_WORD.fineOwedUpdates] = FINE_PROBE_UPDATES;
        words[UNIFORM_WORD.propagation] = blend.keyProbes ? 1 : 0;
        words[UNIFORM_WORD.holdFineProbes] = blend.holdFineProbes ? 1 : 0;
        // The sun layer's unit (T2.6b): the blend weighs that layer by it in the probe's mean. Read only with a sun layer.
        data.set([...(sun ? sunLightOf(sun) : [0, 0, 0]), 0], UNIFORM_WORD.sunLight);
        data[UNIFORM_WORD.cameraFalloffDistance] = cameraFalloffDistance(grid);
        data.set([...frame.camera.position, inViewMargin(grid)], UNIFORM_WORD.cameraPosition);
        frustumPlanes(frame.camera).forEach((plane, index) => data.set(plane, FRUSTUM_PLANE_WORDS[index]));
    }
}

/**
 * How a big change is served on this volume at a batch of `batch` probes a
 * frame (m2-lights.md §4.5 and §6.4; T2.4c, T2.4d):
 * - `bigVolume`: K updates for every probe wouldn't fit in FAR_RULE_FRAMES
 *   frames, so a change takes the key-probe path (or, with key probes off,
 *   far probes owe 2);
 * - `keysFitWindow`: all the keys' full counts fit in FAR_RULE_FRAMES at the
 *   batch's priority share, so every key owes its full count (the furnished
 *   room), else keys away from the view owe `keyProbeUpdates` (Sponza);
 * - `changeOutlastsWindow`: either way, a change here is taken in over more
 *   than FAR_RULE_FRAMES frames, so the batch while it is taken in may be
 *   smaller than the hand-out (T2.18, change-intake-batch.ts). Not where the
 *   keys all fit: they converge a round a frame only while the priority share
 *   holds them all.
 */
export function changeServing(grid: ProbeGrid, batch: number, settings: DDGISettings): { bigVolume: boolean; keysFitWindow: boolean; changeOutlastsWindow: boolean } {
    const blend = blendSettings(settings);
    const fullUpdates = owedUpdates(blend.changeUpdates);
    const framesForEveryProbe = batch > 0 ? (grid.probeCount * fullUpdates) / batch : Number.POSITIVE_INFINITY;
    const bigVolume = framesForEveryProbe > FAR_RULE_FRAMES;
    const keysFitWindow = (keyProbeCount(grid.counts) * fullUpdates) / Math.max(1, priorityCount(batch, blend.priorityShare)) <= FAR_RULE_FRAMES;
    const changeOutlastsWindow = bigVolume && (!blend.keyProbes || !keysFitWindow);
    return { bigVolume, keysFitWindow, changeOutlastsWindow };
}

/**
 * A volume's grid where its window is this frame: a camera-following
 * volume's origin is its window's first probe (scroll.ts); a fixed volume's
 * grid as it is. What `ddgi/mark`'s table is built on (in window probes).
 */
function windowGrid(resources: Pick<PreparedResources, 'grid' | 'scroll'>): ProbeGrid {
    const window = resources.scroll?.window;
    return window ? { ...resources.grid, origin: window.windowOrigin } : resources.grid;
}

/** Probes over every volume. */
function totalProbes(volumes: readonly PreparedResources[]): number {
    return volumes.reduce((sum, volume) => sum + volume.grid.probeCount, 0);
}

/** The most probes the technique updates in a frame: every volume's most. */
function totalMaxProbesPerFrame(volumes: readonly PreparedResources[]): number {
    return volumes.reduce((sum, volume) => sum + volume.maxProbesPerFrame, 0);
}

/** Full passes the current warm-up has made over every volume's probes: the fewest of any volume (0 with none). */
function warmUpPassesMade(volumes: readonly PreparedResources[]): number {
    if (volumes.length === 0) return 0;
    return Math.min(...volumes.map((volume) => (volume.grid.probeCount > 0 ? volume.state.warmUpProbeUpdates / volume.grid.probeCount : 0)));
}

/**
 * The volumes' latest censuses added word by word, for the counters: counts
 * add up over volumes (the change clocks too, modulo 2^32; the counters list
 * each volume's beside the sum). The first volume's own census with one
 * volume; null until every volume has one.
 */
function summedCensus(volumes: readonly PreparedResources[]): Uint32Array | null {
    if (volumes.length === 0 || volumes.some((volume) => !volume.state.latestCensus)) return null;
    if (volumes.length === 1) return volumes[0].state.latestCensus;
    const sum = new Uint32Array(CENSUS_WORDS);
    for (const volume of volumes) {
        const census = volume.state.latestCensus as Uint32Array;
        for (let word = 0; word < CENSUS_WORDS; word++) sum[word] += census[word];
    }
    return sum;
}

function positiveInteger(value: number, name: string): number {
    if (!Number.isInteger(value) || value < 1) throw new Error(`@pmndrs/ddgi: ${name} must be a positive whole number, got ${value}`);
    return value;
}
