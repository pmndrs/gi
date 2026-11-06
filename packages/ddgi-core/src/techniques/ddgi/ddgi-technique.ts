/**
 * DDGI as a platform technique (docs/design/ddgi.md): a box of probes whose
 * irradiance and distance atlases are updated a budget of probes per frame.
 *
 * Each frame, into the host's encoder:
 * 0. `ddgi/mark`, only in frames whose `SceneChanges` report a light or sky
 *    change: tells each probe how much its light changed, from the lights'
 *    own numbers (wgsl/mark.ts, mark-table.ts), so its next blend cuts its
 *    history to match, and promises it updates at priority
 *    (docs/design/m2-lights.md §4.1, §4.2);
 * 0b. `ddgi/schedule`: the GPU scheduler's four dispatches fill the update
 *    list, probes owed updates first by urgency and nearness to the camera,
 *    the rest from the round robin (wgsl/schedule.ts, §4.3);
 * 1. `ddgi/trace`: one thread per ray traces inline, shades the hit and
 *    writes the ray buffer;
 * 2. `ddgi/blend`: one workgroup per probe classifies and relocates the
 *    probe (phase B) and blends its rays into the scratch buffer (phase C);
 * 3. `ddgi/write`: copies the scratch into the atlases, the probe data and
 *    the `points` debug view;
 * 3b. `ddgi/propagate`, with key probes on (T2.4c, §4.5): fine probes
 *    waiting for their key neighbours take the change those keys' updates
 *    made this frame, and their tiles are written (wgsl/propagate.ts). It
 *    does nothing in a frame where no key passed a change on;
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
 * It sees only its `TechniqueContext`: the scene copy, the `rays` backend,
 * hit shading, the kernel builder and the timer. Never the host or an engine.
 * Renderers read the field through its shading module (`shading`,
 * shading.ts), which wraps the same `ddgiSampleIrradiance` the update uses.
 */

import type { DebugView } from '../../platform/contract/stats.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame, TechniqueRequirements, WorkFloorRate } from '../../platform/contract/technique.ts';
import type { ComputeKernel } from '../../platform/contract/wgsl.ts';
import { GILimitError } from '../../platform/contract/errors.ts';
import {
    CENSUS_CHANGE_CLOCK,
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
    RAY_RECORD_BYTES,
    SAMPLE_PARAMETER_VEC4_COUNT,
    SCHEDULE_STATE_REPORTED_WORDS,
    SCHEDULE_STATE_WORD,
    SCHEDULE_STATE_WORDS,
    scratchLayout,
    MAT3_COLUMN_WORDS,
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
    distanceClamp,
    effectiveMaxRayDistance,
    effectiveRaysPerProbe,
    FAR_RULE_FRAMES,
    FINE_PROBE_UPDATES,
    minProbesPerFrame,
    owedUpdates,
    probeGrid,
    resolveBias,
    warmUpSettings,
    withDefaults,
    type DDGISettings,
    type DDGISettingsInput,
    type ProbeGrid,
} from './settings.ts';
import { cameraFalloffDistance, frustumPlanes, inViewMargin } from './camera.ts';
import { buildMarkTable, type MarkTable } from './mark-table.ts';
import { priorityCount, scrambleStride } from './schedule.ts';
import { DDGIShadingModule } from './shading.ts';
import { blendKernelWgsl, type PhaseBReduction } from './wgsl/blend.ts';
import { CENSUS_WORKGROUP_SIZE, DDGI_CENSUS_WGSL } from './wgsl/census.ts';
import { DDGI_TRACE_WGSL, TRACE_WORKGROUP_SIZE } from './wgsl/trace.ts';
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
const INITIAL_MARK_TABLE_BYTES = 64 * 176 + 16;

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
}

/**
 * What test tools read (renderers use `shading`): the three textures and the
 * four `vec4f` parameters of `ddgiSampleIrradiance`, in order. Textures are
 * replaced only by `prepare`.
 */
export interface DDGIField {
    readonly grid: ProbeGrid;
    readonly layout: FieldLayout;
    readonly irradiance: GPUTexture;
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
}

interface PreparedResources {
    readonly grid: ProbeGrid;
    readonly layout: FieldLayout;
    readonly scratch: ScratchLayout;
    readonly irradianceTexels: number;
    readonly distanceTexels: number;
    readonly maxProbesPerFrame: number;
    readonly reduction: PhaseBReduction;
    readonly irradiance: GPUTexture;
    readonly distance: GPUTexture;
    readonly probeData: GPUTexture;
    readonly points: GPUTexture;
    readonly sampler: GPUSampler;
    readonly uniforms: GPUBuffer;
    readonly updateList: GPUBuffer;
    readonly scratchBuffer: GPUBuffer;
    readonly census: GPUBuffer;
    /** Per probe: the updates its history counts as, the clock at its last update and its blended updates so far (adaptive hysteresis). */
    readonly probeHistory: GPUBuffer;
    readonly censusStaging: CensusStaging[];
    readonly traceKernel: ComputeKernel;
    readonly blendKernel: ComputeKernel;
    readonly writeKernel: ComputeKernel;
    readonly censusKernel: ComputeKernel;
    readonly markKernel: ComputeKernel;
    /** The change clock's one-thread dispatch after the mark (same bind group). */
    readonly markClockKernel: ComputeKernel;
    /** Propagation (T2.4c, wgsl/propagate.ts): find, args, propagate, and the irradiance-tile write. */
    readonly propagation: PropagationResources;
    /** The scheduler's four kernels (wgsl/schedule.ts), one bind group for all. */
    readonly scheduleKernels: Readonly<Record<keyof typeof SCHEDULE_ENTRY_POINTS, ComputeKernel>>;
    readonly scheduleGroup: GPUBindGroup;
    /** The scheduler's state (SCHEDULE_STATE_WORD): the round robin's cursor and this frame's histogram and counts. */
    readonly scheduleState: GPUBuffer;
    /** One u32 per probe: an update the feedback rule asked for, as its urgency's bits. */
    readonly feedbackRequests: GPUBuffer;
    readonly writeGroup: GPUBindGroup;
    readonly censusGroup: GPUBindGroup;
    /** The ray buffer and the groups that bind it; replaced when it grows. */
    rays: { buffer: GPUBuffer; capacity: number; traceGroup: GPUBindGroup; blendGroup: GPUBindGroup };
    /** `ddgi/mark`'s table and the group that binds it; replaced when a frame's table outgrows it. */
    mark: { buffer: GPUBuffer; group: GPUBindGroup };
    /** The round robin's stride: order[i] = (i × stride) mod probe count (schedule.ts). */
    readonly stride: number;
    readonly fixedBytes: number;
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
    ],
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
    private resources: PreparedResources | null = null;
    private readonly uniformData = new Float32Array(UNIFORM_WORDS);
    private readonly uniformWords = new Uint32Array(this.uniformData.buffer);
    private readonly parameterView = new Float32Array(SAMPLE_PARAMETER_VEC4_COUNT * 4);
    /** The latest census read back: probes per state, then rays traced. Null until the first arrives. */
    private latestCensus: Uint32Array | null = null;
    private probesUpdated = 0;
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
    /** The warm-up (see the file comment): its phase, its probe updates and when it started warming (wall clock, ms). */
    private warmUp: { phase: WarmUpPhase; probeUpdates: number; startedAt: number | null } = { phase: 'done', probeUpdates: 0, startedAt: null };
    /** The last frame's update count above 0: the batch "far probes owe 2" judges a frame with no updates by. */
    private lastUpdateCount = 0;
    /** `resetHistory` asked for the history and the feedback requests to be cleared, in the next encode. */
    private historyClearPending = false;
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
                    if (!this.resources) throw new Error('@pmndrs/ddgi: the probes debug view exists after prepare');
                    return this.resources.points;
                },
                count: () => this.resources?.grid.probeCount ?? 0,
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
                const resources = this.resources;
                if (!resources) throw new Error('@pmndrs/ddgi: the DDGI shading module has textures after prepare (await gi.prepare())');
                return { irradiance: resources.irradiance, distance: resources.distance, probeData: resources.probeData };
            },
            sampleParameters: () => {
                const resources = this.resources;
                if (!resources) throw new Error('@pmndrs/ddgi: the DDGI shading module has its volume after prepare (await gi.prepare())');
                return this.refreshSampleParameters(resources);
            },
        });
    }

    attach(context: TechniqueContext): void {
        if (this.context) throw new Error('@pmndrs/ddgi: a DDGI technique can be attached to one platform only');
        this.context = context;
    }

    /** Which phase-B reduction the kernels use (after `prepare`). */
    get reduction(): PhaseBReduction | undefined {
        return this.resources?.reduction;
    }

    async prepare(): Promise<void> {
        const context = this.context;
        if (!context) throw new Error('@pmndrs/ddgi: attach the DDGI technique (gi.addTechnique) before prepare');
        if (this.disposed) throw new Error('@pmndrs/ddgi: this DDGI technique was disposed');
        const device = context.device;

        const grid = probeGrid(this.settings.volume);
        const irradianceTexels = positiveInteger(this.settings.irradianceTexels, 'irradianceTexels');
        const distanceTexels = positiveInteger(this.settings.distanceTexels, 'distanceTexels');
        const layout = fieldLayout(grid, irradianceTexels, distanceTexels, distanceFormatFor(device.features as ReadonlySet<string>));
        checkFieldLimits(layout, device.limits);
        const points = pointsSize(grid.probeCount, device.limits.maxTextureDimension2D);
        if (points.height > device.limits.maxTextureDimension2D) throw new GILimitError('maxTextureDimension2D', points.height, device.limits.maxTextureDimension2D);
        const scratch = scratchLayout(irradianceTexels, distanceTexels);
        // No probe updates twice in a frame, so more slots than probes would never be used.
        const maxProbesPerFrame = Math.min(positiveInteger(this.settings.maxProbesPerFrame, 'maxProbesPerFrame'), grid.probeCount);
        const hasSubgroups = device.features.has('subgroups');
        const reduction = this.options.reduction ?? (hasSubgroups ? 'subgroups' : 'workgroup');
        if (reduction === 'subgroups' && !hasSubgroups) throw new Error("@pmndrs/ddgi: the 'subgroups' reduction needs a device with the 'subgroups' feature");

        // Compile first: if a kernel fails, the old resources stay usable.
        const scheduleCode = scheduleKernelWgsl(reduction);
        const markCode = markKernelWgsl(reduction);
        const writeCode = writeKernelWgsl({ irradianceTexels, distanceTexels, scratch, distanceFormat: layout.distanceFormat });
        const propagationCapacity = propagationListCapacity(grid.counts, maxProbesPerFrame);
        const propagateCode = propagateKernelWgsl({ irradianceTexels, reduction, listCapacity: propagationCapacity });
        const propagateKernel = (entryPoint: string, name: string, group: GPUBindGroupLayoutDescriptor): Promise<ComputeKernel> =>
            context.kernels.computeKernel({ label: `ddgi/${name} (${reduction})`, modules: [], code: propagateCode, entryPoint, ownGroups: [group] });
        const scheduleKernel = (entryPoint: string, name: string): Promise<ComputeKernel> =>
            context.kernels.computeKernel({ label: `ddgi/schedule-${name} (${reduction})`, modules: [], code: scheduleCode, entryPoint, ownGroups: [SCHEDULE_GROUP] });
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
        ] = await Promise.all([
            context.kernels.computeKernel({
                label: 'ddgi/trace',
                // Hit shading brings the scene copy and rays in as its dependencies.
                modules: [context.shade.module],
                code: DDGI_TRACE_WGSL,
                entryPoint: 'ddgiTrace',
                ownGroups: [TRACE_GROUP],
            }),
            context.kernels.computeKernel({
                label: `ddgi/blend (${reduction})`,
                modules: [],
                code: blendKernelWgsl({ irradianceTexels, distanceTexels, scratch, reduction }),
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
                modules: [context.shade.lightFalloff],
                code: markCode,
                entryPoint: MARK_ENTRY_POINTS.mark,
                ownGroups: [MARK_GROUP],
            }),
            context.kernels.computeKernel({
                label: 'ddgi/mark-clock',
                modules: [context.shade.lightFalloff],
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
        ]);
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
        const irradiance = arrayTexture('irradiance', layout.irradiance, IRRADIANCE_FORMAT);
        const distance = arrayTexture('distance', layout.distance, layout.distanceFormat);
        const probeData = arrayTexture('probe data', layout.probeData, PROBE_DATA_FORMAT, TEXTURE_USAGE.copyDst);
        const pointsTexture = device.createTexture({ label: 'ddgi points', size: points, format: 'rgba32float', usage: atlasUsage | TEXTURE_USAGE.copyDst });
        const uniforms = device.createBuffer({ label: 'ddgi uniforms', size: UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        // Written by the GPU scheduler.
        const updateList = device.createBuffer({ label: 'ddgi update list', size: maxProbesPerFrame * 4, usage: BUFFER_USAGE.storage });
        const scratchBuffer = device.createBuffer({ label: 'ddgi scratch', size: maxProbesPerFrame * scratch.strideWords * 4, usage: BUFFER_USAGE.storage });
        const census = device.createBuffer({ label: 'ddgi census', size: CENSUS_WORDS * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        // Starts at zero (WebGPU clears new buffers): no probe remembers anything yet.
        // COPY_SRC so debug tools and tests can copy it out (DDGIField.probeHistory).
        const probeHistory = device.createBuffer({ label: 'ddgi probe history', size: grid.probeCount * PROBE_HISTORY_BYTES, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        // The scheduler's cursor starts at the order's first entry (probe 0) and every count at 0: new buffers are zero.
        const scheduleState = device.createBuffer({ label: 'ddgi schedule state', size: SCHEDULE_STATE_WORDS * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        const feedbackRequests = device.createBuffer({ label: 'ddgi feedback requests', size: grid.probeCount * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copyDst });
        // Propagation (T2.4c): a key-delta entry per update-list slot, the list of fine probes, its scratch and its length.
        const keyDeltas = device.createBuffer({ label: 'ddgi key deltas', size: maxProbesPerFrame * keyDeltaStrideWords(irradianceTexels) * 4, usage: BUFFER_USAGE.storage });
        const propagationList = device.createBuffer({ label: 'ddgi propagation list', size: propagationCapacity * 4, usage: BUFFER_USAGE.storage });
        const propagationScratch = device.createBuffer({
            label: 'ddgi propagation scratch',
            size: propagationCapacity * propagationScratchStrideWords(irradianceTexels) * 4,
            usage: BUFFER_USAGE.storage,
        });
        const propagationState = device.createBuffer({
            label: 'ddgi propagation state',
            size: PROPAGATION_STATE_WORDS * 4,
            usage: BUFFER_USAGE.storage | BUFFER_USAGE.copyDst | BUFFER_USAGE.copySrc,
        });
        const censusStaging = Array.from({ length: CENSUS_STAGING_BUFFERS }, (_, index) => ({
            buffer: device.createBuffer({ label: `ddgi census readback ${index}`, size: CENSUS_WORDS * 4, usage: BUFFER_USAGE.mapRead | BUFFER_USAGE.copyDst }),
            state: 'free' as CensusStaging['state'],
        }));
        const sampler = device.createSampler({ label: 'ddgi atlases', magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });

        // Array views explicitly: a one-layer texture's default view would be 2d, not 2d-array.
        const arrayView = (target: GPUTexture): GPUTextureView => target.createView({ dimension: '2d-array' });
        const writeGroup = device.createBindGroup({
            label: 'ddgi/write group 1',
            layout: writeKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: updateList } },
                { binding: 2, resource: { buffer: scratchBuffer } },
                { binding: 3, resource: arrayView(irradiance) },
                { binding: 4, resource: arrayView(distance) },
                { binding: 5, resource: arrayView(probeData) },
                { binding: 6, resource: pointsTexture.createView() },
            ],
        });
        const censusGroup = device.createBindGroup({
            label: 'ddgi/census group 1',
            layout: censusKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: census } },
                { binding: 2, resource: arrayView(probeData) },
            ],
        });
        // The four kernels' layouts are made from one descriptor, so one bind group fits them all.
        const scheduleGroup = device.createBindGroup({
            label: 'ddgi/schedule group 1',
            layout: scheduleCount.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: probeHistory } },
                { binding: 2, resource: { buffer: feedbackRequests } },
                { binding: 3, resource: { buffer: scheduleState } },
                { binding: 4, resource: { buffer: updateList } },
            ],
        });

        const propagation: PropagationResources = {
            findKernel: propagateFind,
            argsKernel: propagateArgs,
            propagateKernel: propagatePropagate,
            writeKernel: propagateWrite,
            keyDeltas,
            list: propagationList,
            scratch: propagationScratch,
            state: propagationState,
            findGroup: device.createBindGroup({
                label: 'ddgi/propagate-find group 1',
                layout: propagateFind.ownLayouts[0],
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
                layout: propagateArgs.ownLayouts[0],
                entries: [{ binding: 3, resource: { buffer: propagationState } }],
            }),
            propagateGroup: device.createBindGroup({
                label: 'ddgi/propagate group 1',
                layout: propagatePropagate.ownLayouts[0],
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
                layout: propagateWrite.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: uniforms } },
                    { binding: 1, resource: { buffer: propagationList } },
                    { binding: 2, resource: { buffer: propagationScratch } },
                    { binding: 3, resource: arrayView(irradiance) },
                    { binding: 7, resource: { buffer: propagationState } },
                ],
            }),
        };

        const fixedBytes =
            fieldBytes(layout) +
            points.width * points.height * POINT_TEXEL_BYTES +
            UNIFORM_BYTES +
            updateList.size +
            scratchBuffer.size +
            probeHistory.size +
            scheduleState.size +
            feedbackRequests.size +
            keyDeltas.size +
            propagationList.size +
            propagationScratch.size +
            propagationState.size +
            census.size * (1 + CENSUS_STAGING_BUFFERS);
        const partial = {
            grid,
            layout,
            scratch,
            irradianceTexels,
            distanceTexels,
            maxProbesPerFrame,
            reduction,
            irradiance,
            distance,
            probeData,
            points: pointsTexture,
            sampler,
            uniforms,
            updateList,
            scratchBuffer,
            census,
            probeHistory,
            censusStaging,
            traceKernel,
            blendKernel,
            writeKernel,
            censusKernel,
            markKernel,
            markClockKernel,
            propagation,
            scheduleKernels: { count: scheduleCount, select: scheduleSelect, write: scheduleWrite, fill: scheduleFill },
            scheduleGroup,
            scheduleState,
            feedbackRequests,
            writeGroup,
            censusGroup,
            stride: scrambleStride(grid.probeCount),
            fixedBytes,
        };
        this.resources = {
            ...partial,
            rays: this.createRayBuffer(device, partial, Math.min(INITIAL_RAY_CAPACITY, maxProbesPerFrame * 256)),
            mark: this.createMarkTable(device, partial, INITIAL_MARK_TABLE_BYTES),
        };
        this.latestCensus = null;
        this.probesUpdated = 0;
        this.markFrames = 0;
        this.latestMarkEntries = 0;
        this.clockSeconds = 0;
        this.lastUpdateCount = 0;
        this.historyClearPending = false;
        this.startWarmUp();
        this.markAllNotUpdated();
    }

    /**
     * True from `prepare` (and `resetHistory`) until the warm-up is done: the
     * technique is waiting for the ray-tracing trees or making its warm-up
     * passes, and ignores the frame's budget. The platform leaves these
     * frames out of the budget's calibration. Always false with
     * `warmUpPasses` 0.
     */
    get warmingUp(): boolean {
        return this.resources !== null && this.warmUp.phase !== 'done';
    }

    /** Records this frame's updates; returns the probes it updated (the units it used). */
    encode(encoder: GPUCommandEncoder, frame: TechniqueFrame): number {
        const resources = this.resources;
        const context = this.context;
        if (!resources || !context || this.disposed) return 0;
        this.collectCensus(resources);

        const warming = this.advanceWarmUp(context, resources.grid.probeCount);
        // Time passes whether or not this frame updates any probe; the warm-up's passes have no time between them.
        if (!warming) this.clockSeconds = (this.clockSeconds + clockStep(frame.deltaTime)) % CLOCK_WRAP_SECONDS;
        const probeCount = resources.grid.probeCount;
        // No probe updates twice in a frame. While warming: a full batch, or none while the trees are built.
        const requested = this.warmUp.phase === 'warming' ? resources.maxProbesPerFrame : this.warmUp.phase === 'waiting for trees' ? 0 : frame.units;
        const count = Math.max(0, Math.min(Number.isNaN(requested) ? 0 : Math.floor(requested), resources.maxProbesPerFrame, probeCount));
        this.probesUpdated = count;
        if (count > 0) this.lastUpdateCount = count;
        if (frame.deltaTime > 0 && Number.isFinite(frame.deltaTime)) {
            this.frameTimes.push(frame.deltaTime);
            if (this.frameTimes.length > SWEEP_FRAME_TIMES) this.frameTimes.shift();
        }
        // Marks land even in a frame that updates no probe: they wait in the
        // history for each probe's next update.
        const markTable = this.settings.changeDrivenHistory !== false ? buildMarkTable(frame.changes, resources.grid) : null;
        this.latestMarkEntries = markTable?.entryCount ?? 0;
        if (count === 0 && !markTable && !this.historyClearPending) return 0;
        const raysPerProbe = effectiveRaysPerProbe(this.settings.raysPerProbe);
        const device = context.device;
        const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;

        this.writeUniforms(resources, frame, count, raysPerProbe);
        device.queue.writeBuffer(resources.uniforms, 0, this.uniformData);
        if (this.historyClearPending) {
            // resetHistory: every probe starts again with nothing owed, cut or asked for.
            encoder.clearBuffer(resources.probeHistory);
            encoder.clearBuffer(resources.feedbackRequests);
            this.historyClearPending = false;
        }
        // The census buffer gathers this frame's marks (ddgi/mark), rays traced and feedback (ddgi/blend); the running totals stay.
        encoder.clearBuffer(resources.census, 0, CENSUS_PER_FRAME_WORDS * 4);
        if (markTable) {
            this.encodeMark(encoder, context, resources, markTable, maxPerDimension);
            this.markFrames++;
        }
        if (count > 0) this.encodeSchedule(encoder, context, resources, maxPerDimension);
        if (this.warmUp.phase === 'warming') this.finishWarmUpPass(resources, count);
        if (count === 0) return 0;

        if (count * raysPerProbe > resources.rays.capacity) {
            const old = resources.rays.buffer;
            resources.rays = this.createRayBuffer(device, resources, Math.max(count * raysPerProbe, resources.rays.capacity * 2));
            old.destroy();
        }

        const probesX = Math.min(count, maxPerDimension);
        const probesY = Math.ceil(count / probesX);

        // Separate passes so each is timed on its own. Passes are separate
        // usage scopes, so ddgi/write may store into the atlases the others sampled.
        const trace = encoder.beginComputePass({ label: 'ddgi/trace', timestampWrites: context.timer.computePass('ddgi/trace') });
        resources.traceKernel.bind(trace);
        trace.setBindGroup(1, resources.rays.traceGroup);
        trace.dispatchWorkgroups(Math.ceil(raysPerProbe / TRACE_WORKGROUP_SIZE), probesX, probesY);
        trace.end();

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

        if (this.settings.keyProbes !== false) this.encodePropagation(encoder, context, resources, maxPerDimension);
        this.encodeCensus(encoder, context, resources, maxPerDimension);
        return count;
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
        this.warmUp = { phase: passes > 0 ? 'waiting for trees' : 'done', probeUpdates: 0, startedAt: null };
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

    /** Counts a warm-up frame's probes; ends the warm-up after its passes, or after its time once the fewest passes are made. */
    private finishWarmUpPass(resources: PreparedResources, count: number): void {
        const warmUp = this.warmUp;
        warmUp.probeUpdates += count;
        const settings = warmUpSettings(this.settings);
        const passes = warmUp.probeUpdates / resources.grid.probeCount;
        const elapsedSeconds = warmUp.startedAt === null ? 0 : (performance.now() - warmUp.startedAt) / 1000;
        if (passes >= settings.passes || (elapsedSeconds > settings.maxSeconds && passes >= settings.minPasses)) warmUp.phase = 'done';
    }

    /** Full passes over every probe the warm-up has made (fractional while one is under way). */
    private get warmUpPasses(): number {
        const probeCount = this.resources?.grid.probeCount ?? 0;
        return probeCount > 0 ? this.warmUp.probeUpdates / probeCount : 0;
    }

    /** The field's textures and sampling parameters; null before `prepare`. */
    get field(): DDGIField | null {
        const resources = this.resources;
        if (!resources) return null;
        this.refreshSampleParameters(resources);
        return {
            grid: resources.grid,
            layout: resources.layout,
            irradiance: resources.irradiance,
            distance: resources.distance,
            probeData: resources.probeData,
            sampleParameters: this.parameterView,
            probeHistory: resources.probeHistory,
        };
    }

    /**
     * Counters for the latest frame. Always there: "probes updated", "frames
     * with changes marked" (frames since `prepare` that ran `ddgi/mark`),
     * "changes marked this frame" (its table's entries: changed lights and
     * global changes), "warming up" (1 while `warmingUp`) and "warm-up
     * passes" (full passes the warm-up has made); "seconds per full sweep"
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
     */
    get counters(): Readonly<Record<string, number>> {
        const census = this.latestCensus;
        const sweep = this.sweepSeconds();
        const updates: Record<string, number> = {
            'probes updated': this.probesUpdated,
            'frames with changes marked': this.markFrames,
            'changes marked this frame': this.latestMarkEntries,
            'warming up': this.warmingUp ? 1 : 0,
            'warm-up passes': this.warmUpPasses,
            'key probes': this.resources ? keyProbeCount(this.resources.grid.counts) : 0,
        };
        if (sweep !== undefined) updates['seconds per full sweep'] = sweep;
        if (!census) return updates;
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
            'change clock': census[CENSUS_CHANGE_CLOCK],
            'rays traced': census[CENSUS_RAYS_TRACED],
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
        const resources = this.resources;
        if (!resources) return 0;
        return minProbesPerFrame(this.settings.minProbesPerFrame, resources.grid.probeCount, resources.maxProbesPerFrame, rate);
    }

    /**
     * How long updating every probe once takes at the latest frame's batch and
     * the recent frame rate (the median `deltaTime`); undefined before frames
     * with a time, or with no probes updated.
     */
    private sweepSeconds(): number | undefined {
        const probeCount = this.resources?.grid.probeCount ?? 0;
        if (this.probesUpdated <= 0 || probeCount <= 0 || this.frameTimes.length === 0) return undefined;
        const sorted = [...this.frameTimes].sort((a, b) => a - b);
        const frameSeconds = sorted[Math.floor(sorted.length / 2)];
        return Math.ceil(probeCount / this.probesUpdated) * frameSeconds;
    }

    get memoryBytes(): number {
        const resources = this.resources;
        return resources ? resources.fixedBytes + resources.rays.buffer.size : 0;
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
        if (!this.resources) return;
        this.historyClearPending = true;
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

    private createRayBuffer(device: GPUDevice, resources: Omit<PreparedResources, 'rays' | 'mark'>, capacity: number): PreparedResources['rays'] {
        const buffer = device.createBuffer({ label: 'ddgi rays', size: capacity * RAY_RECORD_BYTES, usage: BUFFER_USAGE.storage });
        const arrayView = (target: GPUTexture): GPUTextureView => target.createView({ dimension: '2d-array' });
        const traceGroup = device.createBindGroup({
            label: 'ddgi/trace group 1',
            layout: resources.traceKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: resources.uniforms } },
                { binding: 1, resource: { buffer: resources.updateList } },
                { binding: 2, resource: { buffer } },
                { binding: 3, resource: arrayView(resources.irradiance) },
                { binding: 4, resource: arrayView(resources.distance) },
                { binding: 5, resource: arrayView(resources.probeData) },
                { binding: 6, resource: resources.sampler },
            ],
        });
        const blendGroup = device.createBindGroup({
            label: 'ddgi/blend group 1',
            layout: resources.blendKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: resources.uniforms } },
                { binding: 1, resource: { buffer: resources.updateList } },
                { binding: 2, resource: { buffer } },
                { binding: 3, resource: { buffer: resources.scratchBuffer } },
                { binding: 4, resource: { buffer: resources.census } },
                { binding: 5, resource: arrayView(resources.irradiance) },
                { binding: 6, resource: arrayView(resources.distance) },
                { binding: 7, resource: arrayView(resources.probeData) },
                { binding: 8, resource: { buffer: resources.probeHistory } },
                { binding: 9, resource: { buffer: resources.feedbackRequests } },
                { binding: 10, resource: { buffer: resources.propagation.keyDeltas } },
            ],
        });
        return { buffer, capacity, traceGroup, blendGroup };
    }

    /**
     * Counts probes per state into the census buffer and copies it out for
     * reading back. Only when the platform times passes: the platform makes
     * no readbacks unless its `gpuTiming` option is on, and the timer gives
     * this pass timestamp writes exactly then. Also skipped while every
     * readback buffer is still in flight.
     */
    private encodeCensus(encoder: GPUCommandEncoder, context: TechniqueContext, resources: PreparedResources, maxPerDimension: number): void {
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
        encoder.copyBufferToBuffer(resources.census, 0, staging.buffer, 0, CENSUS_WORDS * 4);
        staging.state = 'copied';
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
                    if (this.resources !== resources) return;
                    this.latestCensus = new Uint32Array(staging.buffer.getMappedRange().slice(0));
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
        const resources = this.resources;
        if (!resources) return;
        this.resources = null;
        for (const texture of [resources.irradiance, resources.distance, resources.probeData, resources.points]) texture.destroy();
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
            resources.rays.buffer,
            resources.mark.buffer,
        ]) {
            buffer.destroy();
        }
        for (const staging of resources.censusStaging) staging.buffer.destroy();
    }

    /** Writes state 3 with zero offsets into every probe-data texel and point. Lands before the host's next submit. */
    private markAllNotUpdated(): void {
        const resources = this.resources;
        const context = this.context;
        if (!resources || !context) return;
        const queue = context.device.queue;
        const { width, height, layers } = resources.layout.probeData;
        const texels = new Uint16Array(width * height * layers * 4);
        for (let texel = 0; texel < width * height * layers; texel++) texels[texel * 4 + 3] = HALF_THREE;
        queue.writeTexture({ texture: resources.probeData }, texels, { bytesPerRow: width * PROBE_DATA_TEXEL_BYTES, rowsPerImage: height }, { width, height, depthOrArrayLayers: layers });

        const { grid, points } = resources;
        const pointData = new Float32Array(points.width * points.height * 4);
        const [nx, , nz] = grid.counts;
        for (let probe = 0; probe < grid.probeCount; probe++) {
            const x = probe % nx;
            const z = Math.floor(probe / nx) % nz;
            const y = Math.floor(probe / (nx * nz));
            pointData.set([grid.origin[0] + x * grid.spacing[0], grid.origin[1] + y * grid.spacing[1], grid.origin[2] + z * grid.spacing[2], PROBE_STATE.notUpdatedYet], probe * 4);
        }
        queue.writeTexture({ texture: points }, pointData, { bytesPerRow: points.width * POINT_TEXEL_BYTES }, { width: points.width, height: points.height });
    }

    /** The four sampling parameters from the live settings, in `parameterView` (the same array every time). */
    private refreshSampleParameters(resources: PreparedResources): Float32Array {
        this.fillSampleParameters(resources.grid, resources.irradianceTexels, resources.distanceTexels);
        this.parameterView.set(this.uniformData.subarray(0, SAMPLE_PARAMETER_VEC4_COUNT * 4));
        return this.parameterView;
    }

    /** The first four vec4s of the uniforms: the volume, with the live biases (`DdgiVolume` in wgsl/sample.ts names them). */
    private fillSampleParameters(grid: ProbeGrid, irradianceTexels: number, distanceTexels: number): void {
        const data = this.uniformData;
        data.set([...grid.origin, resolveBias(this.settings.normalBias, grid, AUTO_NORMAL_BIAS_SHARE)], UNIFORM_WORD.volumeOrigin);
        data.set([...grid.spacing, resolveBias(this.settings.viewBias, grid, AUTO_VIEW_BIAS_SHARE)], UNIFORM_WORD.volumeSpacing);
        data.set([...grid.counts, 0], UNIFORM_WORD.volumeCounts);
        data.set([irradianceTexels, distanceTexels, 0, 0], UNIFORM_WORD.volumeTexels);
    }

    private writeUniforms(resources: PreparedResources, frame: TechniqueFrame, count: number, raysPerProbe: number): void {
        const { grid } = resources;
        this.fillSampleParameters(grid, resources.irradianceTexels, resources.distanceTexels);
        const data = this.uniformData;
        const words = this.uniformWords;
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
        const batch = count > 0 ? count : this.lastUpdateCount;
        const framesForEveryProbe = batch > 0 ? (grid.probeCount * fullUpdates) / batch : Number.POSITIVE_INFINITY;
        // On such a volume a big change takes the key-probe path (T2.4c,
        // m2-lights.md §4.5), or with key probes off, "far probes owe 2" (§6.4).
        const bigVolume = framesForEveryProbe > FAR_RULE_FRAMES;
        words[UNIFORM_WORD.keyProbes] = blend.keyProbes && bigVolume ? 1 : 0;
        data[UNIFORM_WORD.farOwedUpdates] = !blend.keyProbes && bigVolume ? Math.min(blend.farProbeUpdates, fullUpdates) : fullUpdates;
        data[UNIFORM_WORD.keyProbeBoost] = blend.keyProbeBoost;
        data[UNIFORM_WORD.keyResidualShare] = blend.keyResidualShare;
        data[UNIFORM_WORD.keyOwedUpdates] = blend.keyProbeUpdates;
        data[UNIFORM_WORD.fineOwedUpdates] = FINE_PROBE_UPDATES;
        words[UNIFORM_WORD.propagation] = blend.keyProbes ? 1 : 0;
        words[UNIFORM_WORD.holdFineProbes] = blend.holdFineProbes ? 1 : 0;
        data[UNIFORM_WORD.cameraFalloffDistance] = cameraFalloffDistance(grid);
        data.set([...frame.camera.position, inViewMargin(grid)], UNIFORM_WORD.cameraPosition);
        frustumPlanes(frame.camera).forEach((plane, index) => data.set(plane, FRUSTUM_PLANE_WORDS[index]));
    }
}

function positiveInteger(value: number, name: string): number {
    if (!Number.isInteger(value) || value < 1) throw new Error(`@pmndrs/ddgi: ${name} must be a positive whole number, got ${value}`);
    return value;
}
