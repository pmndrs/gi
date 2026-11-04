/**
 * DDGI as a platform technique (docs/design/ddgi.md): a box of probes whose
 * irradiance and distance atlases are updated a budget of probes per frame.
 *
 * Each frame, into the host's encoder:
 * 1. `ddgi/trace`: one thread per ray traces inline, shades the hit and
 *    writes the ray buffer;
 * 2. `ddgi/blend`: one workgroup per probe classifies and relocates the
 *    probe (phase B) and blends its rays into the scratch buffer (phase C);
 * 3. `ddgi/write`: copies the scratch into the atlases, the probe data and
 *    the `points` debug view;
 * 4. `ddgi/census` (now and then): counts probes per state for the counters,
 *    read back a few frames later.
 *
 * It sees only its `TechniqueContext`: the scene copy, the `rays` backend,
 * hit shading, the kernel builder and the timer. Never the host or an engine.
 * The shading module that renderers read is T1.6, built on `field` and
 * `DDGI_SAMPLE_WGSL`.
 */

import type { ShadingModule } from '../../platform/contract/shading-module.ts';
import type { DebugView } from '../../platform/contract/stats.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame, TechniqueRequirements } from '../../platform/contract/technique.ts';
import type { ComputeKernel } from '../../platform/contract/wgsl.ts';
import { GILimitError } from '../../platform/contract/errors.ts';
import {
    CENSUS_RAYS_TRACED,
    CENSUS_WORDS,
    checkFieldLimits,
    distanceFormatFor,
    fieldBytes,
    fieldLayout,
    IRRADIANCE_FORMAT,
    PROBE_DATA_FORMAT,
    PROBE_STATE,
    PROBE_STATE_LABELS,
    pointsSize,
    RAY_RECORD_BYTES,
    SAMPLE_PARAMETER_VEC4_COUNT,
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
    distanceClamp,
    effectiveMaxRayDistance,
    effectiveRaysPerProbe,
    probeGrid,
    resolveBias,
    withDefaults,
    type DDGISettings,
    type DDGISettingsInput,
    type ProbeGrid,
} from './settings.ts';
import { UpdateCursor } from './update-order.ts';
import { blendKernelWgsl, type PhaseBReduction } from './wgsl/blend.ts';
import { CENSUS_WORKGROUP_SIZE, DDGI_CENSUS_WGSL } from './wgsl/census.ts';
import { DDGI_TRACE_WGSL, TRACE_WORKGROUP_SIZE } from './wgsl/trace.ts';
import { writeKernelWgsl } from './wgsl/write.ts';

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

/** Options fixed for the technique's life. */
export interface DDGITechniqueOptions {
    /**
     * How phase B reduces the fixed rays. Defaults to `subgroups` when the
     * device has the feature, otherwise the declared fallback `workgroup`.
     * Both give the same states and offsets; tests run both.
     */
    readonly reduction?: PhaseBReduction;
}

/**
 * What a renderer (T1.6's shading module) and test tools read: the three
 * textures and the four `vec4f` parameters of `ddgiSampleIrradiance`, in
 * order. Textures are replaced only by `prepare`.
 */
export interface DDGIField {
    readonly grid: ProbeGrid;
    readonly layout: FieldLayout;
    readonly irradiance: GPUTexture;
    readonly distance: GPUTexture;
    readonly probeData: GPUTexture;
    /** volumeOrigin, volumeSpacing, volumeCounts, volumeTexels: 16 floats, refreshed from the live settings on every read. */
    readonly sampleParameters: Float32Array;
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
    readonly censusStaging: CensusStaging[];
    readonly traceKernel: ComputeKernel;
    readonly blendKernel: ComputeKernel;
    readonly writeKernel: ComputeKernel;
    readonly censusKernel: ComputeKernel;
    readonly writeGroup: GPUBindGroup;
    readonly censusGroup: GPUBindGroup;
    /** The ray buffer and the groups that bind it; replaced when it grows. */
    rays: { buffer: GPUBuffer; capacity: number; traceGroup: GPUBindGroup; blendGroup: GPUBindGroup };
    readonly cursor: UpdateCursor;
    readonly list: Uint32Array<ArrayBuffer>;
    readonly fixedBytes: number;
}

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
    private disposed = false;

    constructor(settings: DDGISettingsInput, options: DDGITechniqueOptions = {}) {
        this.settings = withDefaults(settings);
        this.options = options;
        this.debugViews = [
            {
                kind: 'points',
                id: 0,
                label: 'Probes',
                description: 'Each probe as a dot where it really sits (after relocation), coloured by its state.',
                texture: () => {
                    if (!this.resources) throw new Error('@pmndrs/ddgi: the probes debug view exists after prepare');
                    return this.resources.points;
                },
                count: () => this.resources?.grid.probeCount ?? 0,
                states: PROBE_STATE_LABELS,
            },
        ];
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
        const [traceKernel, blendKernel, writeKernel, censusKernel] = await Promise.all([
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
                code: writeKernelWgsl({ irradianceTexels, distanceTexels, scratch, distanceFormat: layout.distanceFormat }),
                entryPoint: 'ddgiWrite',
                ownGroups: [writeGroupLayout(layout.distanceFormat)],
            }),
            context.kernels.computeKernel({
                label: 'ddgi/census',
                modules: [],
                code: DDGI_CENSUS_WGSL,
                entryPoint: 'ddgiCountStates',
                ownGroups: [CENSUS_GROUP],
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
            });
        const irradiance = arrayTexture('irradiance', layout.irradiance, IRRADIANCE_FORMAT);
        const distance = arrayTexture('distance', layout.distance, layout.distanceFormat);
        const probeData = arrayTexture('probe data', layout.probeData, PROBE_DATA_FORMAT, TEXTURE_USAGE.copyDst);
        const pointsTexture = device.createTexture({ label: 'ddgi points', size: points, format: 'rgba32float', usage: atlasUsage | TEXTURE_USAGE.copyDst });
        const uniforms = device.createBuffer({ label: 'ddgi uniforms', size: UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const updateList = device.createBuffer({ label: 'ddgi update list', size: maxProbesPerFrame * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copyDst });
        const scratchBuffer = device.createBuffer({ label: 'ddgi scratch', size: maxProbesPerFrame * scratch.strideWords * 4, usage: BUFFER_USAGE.storage });
        const census = device.createBuffer({ label: 'ddgi census', size: CENSUS_WORDS * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
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

        const fixedBytes =
            fieldBytes(layout) +
            points.width * points.height * POINT_TEXEL_BYTES +
            UNIFORM_BYTES +
            updateList.size +
            scratchBuffer.size +
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
            censusStaging,
            traceKernel,
            blendKernel,
            writeKernel,
            censusKernel,
            writeGroup,
            censusGroup,
            cursor: new UpdateCursor(grid.probeCount),
            list: new Uint32Array(maxProbesPerFrame),
            fixedBytes,
        };
        this.resources = { ...partial, rays: this.createRayBuffer(device, partial, Math.min(INITIAL_RAY_CAPACITY, maxProbesPerFrame * 256)) };
        this.latestCensus = null;
        this.probesUpdated = 0;
        this.markAllNotUpdated();
    }

    encode(encoder: GPUCommandEncoder, frame: TechniqueFrame): void {
        const resources = this.resources;
        const context = this.context;
        if (!resources || !context || this.disposed) return;
        this.collectCensus(resources);

        const count = resources.cursor.next(Math.min(frame.units, resources.maxProbesPerFrame), resources.list);
        this.probesUpdated = count;
        if (count === 0) return;
        const raysPerProbe = effectiveRaysPerProbe(this.settings.raysPerProbe);
        const device = context.device;

        this.writeUniforms(resources, frame.frame, count, raysPerProbe);
        device.queue.writeBuffer(resources.uniforms, 0, this.uniformData);
        device.queue.writeBuffer(resources.updateList, 0, resources.list, 0, count);
        if (count * raysPerProbe > resources.rays.capacity) {
            const old = resources.rays.buffer;
            resources.rays = this.createRayBuffer(device, resources, Math.max(count * raysPerProbe, resources.rays.capacity * 2));
            old.destroy();
        }
        // The census buffer also gathers this frame's rays traced (in ddgi/blend).
        encoder.clearBuffer(resources.census);

        const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
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

        this.encodeCensus(encoder, resources, maxPerDimension);
    }

    /** Renderers read the field through T1.6's shading module, built on {@link field}. */
    get shading(): ShadingModule {
        throw new Error('@pmndrs/ddgi: the DDGI shading module is not built yet (T1.6); read `field` with ddgiSampleIrradiance meanwhile');
    }

    /** The field's textures and sampling parameters; null before `prepare`. */
    get field(): DDGIField | null {
        const resources = this.resources;
        if (!resources) return null;
        this.fillSampleParameters(resources.grid, resources.irradianceTexels, resources.distanceTexels);
        this.parameterView.set(this.uniformData.subarray(0, SAMPLE_PARAMETER_VEC4_COUNT * 4));
        return {
            grid: resources.grid,
            layout: resources.layout,
            irradiance: resources.irradiance,
            distance: resources.distance,
            probeData: resources.probeData,
            sampleParameters: this.parameterView,
        };
    }

    /**
     * Counters for the latest frame. The per-state counts and rays traced come
     * from the GPU census, read back a few frames late; until the first one
     * arrives, every probe counts as not updated yet.
     */
    get counters(): Readonly<Record<string, number>> {
        const census = this.latestCensus;
        const probes = this.resources?.grid.probeCount ?? 0;
        return {
            'probes updated': this.probesUpdated,
            'rays traced': census ? census[CENSUS_RAYS_TRACED] : 0,
            'probes active': census ? census[PROBE_STATE.active] : 0,
            'probes off: inside geometry': census ? census[PROBE_STATE.offInsideGeometry] : 0,
            'probes off: no surface nearby': census ? census[PROBE_STATE.offNoSurfaceNearby] : 0,
            'probes not updated yet': census ? census[PROBE_STATE.notUpdatedYet] : probes,
        };
    }

    get memoryBytes(): number {
        const resources = this.resources;
        return resources ? resources.fixedBytes + resources.rays.buffer.size : 0;
    }

    /** Every probe goes back to "not updated yet" at its grid point: its next update replaces it outright, and sampling skips it until then. */
    resetHistory(): void {
        if (!this.resources) return;
        this.resources.cursor.resetHistory();
        this.markAllNotUpdated();
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.releaseResources();
    }

    private createRayBuffer(device: GPUDevice, resources: Omit<PreparedResources, 'rays'>, capacity: number): PreparedResources['rays'] {
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
            ],
        });
        return { buffer, capacity, traceGroup, blendGroup };
    }

    /** Counts probes per state into the census buffer and copies it out, when a readback buffer is free. */
    private encodeCensus(encoder: GPUCommandEncoder, resources: PreparedResources, maxPerDimension: number): void {
        const staging = resources.censusStaging.find((slot) => slot.state === 'free');
        if (!staging) return;
        const groups = Math.ceil(resources.grid.probeCount / CENSUS_WORKGROUP_SIZE);
        const groupsX = Math.min(groups, maxPerDimension);
        const pass = encoder.beginComputePass({ label: 'ddgi/census' });
        resources.censusKernel.bind(pass);
        pass.setBindGroup(1, resources.censusGroup);
        pass.dispatchWorkgroups(groupsX, Math.ceil(groups / groupsX));
        pass.end();
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
        for (const buffer of [resources.uniforms, resources.updateList, resources.scratchBuffer, resources.census, resources.rays.buffer]) buffer.destroy();
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

    /** The first four vec4s of the uniforms: the volume, with the live biases (`DdgiVolume` in wgsl/sample.ts names them). */
    private fillSampleParameters(grid: ProbeGrid, irradianceTexels: number, distanceTexels: number): void {
        const data = this.uniformData;
        data.set([...grid.origin, resolveBias(this.settings.normalBias, grid, AUTO_NORMAL_BIAS_SHARE)], UNIFORM_WORD.volumeOrigin);
        data.set([...grid.spacing, resolveBias(this.settings.viewBias, grid, AUTO_VIEW_BIAS_SHARE)], UNIFORM_WORD.volumeSpacing);
        data.set([...grid.counts, 0], UNIFORM_WORD.volumeCounts);
        data.set([irradianceTexels, distanceTexels, 0, 0], UNIFORM_WORD.volumeTexels);
    }

    private writeUniforms(resources: PreparedResources, frame: number, count: number, raysPerProbe: number): void {
        const { grid } = resources;
        this.fillSampleParameters(grid, resources.irradianceTexels, resources.distanceTexels);
        const data = this.uniformData;
        const words = this.uniformWords;
        const rotation = frameRayRotation(frame);
        for (let column = 0; column < 3; column++) {
            data.set(rotation.slice(column * 3, column * 3 + 3), UNIFORM_WORD.rayRotation + column * MAT3_COLUMN_WORDS);
        }
        const blend = blendSettings(this.settings);
        data[UNIFORM_WORD.hysteresis] = blend.hysteresis;
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
        words[UNIFORM_WORD.frame] = frame >>> 0;
    }
}

function positiveInteger(value: number, name: string): number {
    if (!Number.isInteger(value) || value < 1) throw new Error(`@pmndrs/ddgi: ${name} must be a positive whole number, got ${value}`);
    return value;
}
