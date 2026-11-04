/**
 * DDGI as a platform technique (docs/design/ddgi.md): a box of probes whose
 * irradiance and distance atlases are updated a budget of probes per frame
 * by the `ddgi/update` and `ddgi/write` kernels.
 *
 * It sees only its `TechniqueContext`: the scene copy, the `rays` backend,
 * hit shading, the kernel builder and the timer. Never the host or an engine.
 *
 * T1.5a builds the field and the update loop with every probe treated as
 * active. Classification and relocation (phase B) are T1.5b; the shading
 * module that renderers read is T1.6, built on `field` and `DDGI_SAMPLE_WGSL`.
 */

import type { ShadingModule } from '../../platform/contract/shading-module.ts';
import type { DebugView } from '../../platform/contract/stats.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame, TechniqueRequirements } from '../../platform/contract/technique.ts';
import type { ComputeKernel } from '../../platform/contract/wgsl.ts';
import {
    checkFieldLimits,
    distanceFormatFor,
    fieldBytes,
    fieldLayout,
    IRRADIANCE_FORMAT,
    PROBE_DATA_FORMAT,
    SAMPLE_PARAMETER_VEC4_COUNT,
    scratchLayout,
    UNIFORM_BYTES,
    UNIFORM_VEC4,
    UNIFORM_VEC4_COUNT,
    type DistanceFormat,
    type FieldLayout,
    type ScratchLayout,
} from './layout.ts';
import { frameRayRotation } from './ray-rotation.ts';
import {
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
import { updateKernelWgsl } from './wgsl/update.ts';
import { writeKernelWgsl } from './wgsl/write.ts';

/** WebGPU usage bits; the globals don't exist in Node, where tests import this file. */
const BUFFER_USAGE = { copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const TEXTURE_USAGE = { copySrc: 0x1, copyDst: 0x2, textureBinding: 0x4, storageBinding: 0x8 } as const;
const SHADER_STAGE_COMPUTE = 0x4;

/** Bytes of one probe-data texel (rgba16float). */
const PROBE_DATA_TEXEL_BYTES = 8;
/** The half-float bit pattern of 3.0, the "not updated yet" state. */
const HALF_THREE = 0x4200;

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

interface PreparedResources {
    readonly grid: ProbeGrid;
    readonly layout: FieldLayout;
    readonly scratch: ScratchLayout;
    readonly irradianceTexels: number;
    readonly distanceTexels: number;
    readonly maxProbesPerFrame: number;
    readonly irradiance: GPUTexture;
    readonly distance: GPUTexture;
    readonly probeData: GPUTexture;
    readonly uniforms: GPUBuffer;
    readonly updateList: GPUBuffer;
    readonly scratchBuffer: GPUBuffer;
    readonly updateKernel: ComputeKernel;
    readonly writeKernel: ComputeKernel;
    readonly updateGroup: GPUBindGroup;
    readonly writeGroup: GPUBindGroup;
    readonly cursor: UpdateCursor;
    readonly list: Uint32Array<ArrayBuffer>;
    readonly bytes: number;
}

/** The update kernel's group 1 (wgsl/update.ts). */
function updateGroupLayout(): GPUBindGroupLayoutDescriptor {
    const texture = { sampleType: 'float', viewDimension: '2d-array' } as const;
    return {
        label: 'ddgi/update group 1',
        entries: [
            { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: SHADER_STAGE_COMPUTE, texture },
            { binding: 4, visibility: SHADER_STAGE_COMPUTE, texture },
            { binding: 5, visibility: SHADER_STAGE_COMPUTE, texture },
            { binding: 6, visibility: SHADER_STAGE_COMPUTE, sampler: { type: 'filtering' } },
        ],
    };
}

/** The write kernel's group 1 (wgsl/write.ts). */
function writeGroupLayout(distanceFormat: DistanceFormat): GPUBindGroupLayoutDescriptor {
    const storage = (format: GPUTextureFormat): GPUStorageTextureBindingLayout => ({ access: 'write-only', format, viewDimension: '2d-array' });
    return {
        label: 'ddgi/write group 1',
        entries: [
            { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: SHADER_STAGE_COMPUTE, storageTexture: storage(IRRADIANCE_FORMAT) },
            { binding: 4, visibility: SHADER_STAGE_COMPUTE, storageTexture: storage(distanceFormat) },
            { binding: 5, visibility: SHADER_STAGE_COMPUTE, storageTexture: storage(PROBE_DATA_FORMAT) },
        ],
    };
}

export class DDGITechnique implements GITechnique<DDGISettings> {
    readonly id = 'ddgi';
    readonly requirements: TechniqueRequirements = { queries: ['rays'] };
    readonly settings: DDGISettings;
    /** The points view (probe positions and states) comes with classification, T1.5b. */
    readonly debugViews: readonly DebugView[] = [];
    /** Set from GPU timings once the platform converts ms budgets into units (T1.5b). */
    readonly msPerUnit: number | undefined = undefined;

    private context: TechniqueContext | null = null;
    private resources: PreparedResources | null = null;
    private readonly uniformData = new Float32Array(UNIFORM_VEC4_COUNT * 4);
    private readonly uniformWords = new Uint32Array(this.uniformData.buffer);
    private readonly parameterView = new Float32Array(SAMPLE_PARAMETER_VEC4_COUNT * 4);
    private latestCounters: Record<string, number> = {};
    private disposed = false;

    constructor(settings: DDGISettingsInput) {
        this.settings = withDefaults(settings);
    }

    attach(context: TechniqueContext): void {
        if (this.context) throw new Error('@pmndrs/ddgi: a DDGI technique can be attached to one platform only');
        this.context = context;
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
        const scratch = scratchLayout(irradianceTexels, distanceTexels);
        // No probe updates twice in a frame, so more slots than probes would never be used.
        const maxProbesPerFrame = Math.min(positiveInteger(this.settings.maxProbesPerFrame, 'maxProbesPerFrame'), grid.probeCount);

        // Compile first: if a kernel fails, the old resources stay usable.
        const [updateKernel, writeKernel] = await Promise.all([
            context.kernels.computeKernel({
                label: 'ddgi/update',
                // Hit shading brings the scene copy and rays in as its dependencies.
                modules: [context.shade.module],
                code: updateKernelWgsl({ irradianceTexels, distanceTexels, scratch }),
                entryPoint: 'ddgiUpdate',
                ownGroups: [updateGroupLayout()],
            }),
            context.kernels.computeKernel({
                label: 'ddgi/write',
                modules: [],
                code: writeKernelWgsl({ irradianceTexels, distanceTexels, scratch, distanceFormat: layout.distanceFormat }),
                entryPoint: 'ddgiWrite',
                ownGroups: [writeGroupLayout(layout.distanceFormat)],
            }),
        ]);
        if (this.disposed) return;

        this.releaseResources();
        const atlasUsage = TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.copySrc;
        const texture = (label: string, size: FieldLayout['irradiance'], format: GPUTextureFormat, extraUsage = 0): GPUTexture =>
            device.createTexture({
                label: `ddgi ${label}`,
                size: { width: size.width, height: size.height, depthOrArrayLayers: size.layers },
                format,
                usage: atlasUsage | extraUsage,
            });
        const irradiance = texture('irradiance', layout.irradiance, IRRADIANCE_FORMAT);
        const distance = texture('distance', layout.distance, layout.distanceFormat);
        const probeData = texture('probe data', layout.probeData, PROBE_DATA_FORMAT, TEXTURE_USAGE.copyDst);
        const uniforms = device.createBuffer({ label: 'ddgi uniforms', size: UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const updateList = device.createBuffer({ label: 'ddgi update list', size: maxProbesPerFrame * 4, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copyDst });
        const scratchBuffer = device.createBuffer({ label: 'ddgi scratch', size: maxProbesPerFrame * scratch.strideWords * 4, usage: BUFFER_USAGE.storage });
        const sampler = device.createSampler({ label: 'ddgi atlases', magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });

        // Array views explicitly: a one-layer texture's default view would be 2d, not 2d-array.
        const arrayView = (target: GPUTexture): GPUTextureView => target.createView({ dimension: '2d-array' });
        const updateGroup = device.createBindGroup({
            label: 'ddgi/update group 1',
            layout: updateKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: updateList } },
                { binding: 2, resource: { buffer: scratchBuffer } },
                { binding: 3, resource: arrayView(irradiance) },
                { binding: 4, resource: arrayView(distance) },
                { binding: 5, resource: arrayView(probeData) },
                { binding: 6, resource: sampler },
            ],
        });
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
            ],
        });

        const bytes = fieldBytes(layout) + UNIFORM_BYTES + updateList.size + scratchBuffer.size;
        this.resources = {
            grid,
            layout,
            scratch,
            irradianceTexels,
            distanceTexels,
            maxProbesPerFrame,
            irradiance,
            distance,
            probeData,
            uniforms,
            updateList,
            scratchBuffer,
            updateKernel,
            writeKernel,
            updateGroup,
            writeGroup,
            cursor: new UpdateCursor(grid.probeCount),
            list: new Uint32Array(maxProbesPerFrame),
            bytes,
        };
        this.markAllNotUpdated();
        this.latestCounters = this.countersFor(0, 0);
    }

    encode(encoder: GPUCommandEncoder, frame: TechniqueFrame): void {
        const resources = this.resources;
        const context = this.context;
        if (!resources || !context || this.disposed) return;

        const count = resources.cursor.next(Math.min(frame.units, resources.maxProbesPerFrame), resources.list);
        const raysPerProbe = effectiveRaysPerProbe(this.settings.raysPerProbe);
        this.latestCounters = this.countersFor(count, raysPerProbe);
        if (count === 0) return;

        this.writeUniforms(resources, frame.frame, count, raysPerProbe);
        const queue = context.device.queue;
        queue.writeBuffer(resources.uniforms, 0, this.uniformData);
        queue.writeBuffer(resources.updateList, 0, resources.list, 0, count);

        const maxPerDimension = context.device.limits.maxComputeWorkgroupsPerDimension;
        const groupsX = Math.min(count, maxPerDimension);
        const groupsY = Math.ceil(count / groupsX);

        // Two passes so each is timed on its own. The write pass stores into
        // the atlases the update pass sampled; passes are separate usage scopes.
        const update = encoder.beginComputePass({ label: 'ddgi/update', timestampWrites: context.timer.computePass('ddgi/update') });
        resources.updateKernel.bind(update);
        update.setBindGroup(1, resources.updateGroup);
        update.dispatchWorkgroups(groupsX, groupsY);
        update.end();

        const write = encoder.beginComputePass({ label: 'ddgi/write', timestampWrites: context.timer.computePass('ddgi/write') });
        resources.writeKernel.bind(write);
        write.setBindGroup(1, resources.writeGroup);
        write.dispatchWorkgroups(groupsX, groupsY);
        write.end();
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

    get counters(): Readonly<Record<string, number>> {
        return this.latestCounters;
    }

    get memoryBytes(): number {
        return this.resources?.bytes ?? 0;
    }

    /** Every probe goes back to "not updated yet": its next update replaces it outright, and sampling skips it until then. */
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

    private releaseResources(): void {
        const resources = this.resources;
        if (!resources) return;
        this.resources = null;
        for (const texture of [resources.irradiance, resources.distance, resources.probeData]) texture.destroy();
        for (const buffer of [resources.uniforms, resources.updateList, resources.scratchBuffer]) buffer.destroy();
    }

    /** Writes state 3 with zero offsets into every probe-data texel. Lands before the host's next submit. */
    private markAllNotUpdated(): void {
        const resources = this.resources;
        const context = this.context;
        if (!resources || !context) return;
        const { width, height, layers } = resources.layout.probeData;
        const texels = new Uint16Array(width * height * layers * 4);
        for (let texel = 0; texel < width * height * layers; texel++) texels[texel * 4 + 3] = HALF_THREE;
        context.device.queue.writeTexture(
            { texture: resources.probeData },
            texels,
            { bytesPerRow: width * PROBE_DATA_TEXEL_BYTES, rowsPerImage: height },
            { width, height, depthOrArrayLayers: layers },
        );
    }

    /** vec4s 0-3 of the uniforms: the volume, with the live biases. */
    private fillSampleParameters(grid: ProbeGrid, irradianceTexels: number, distanceTexels: number): void {
        const data = this.uniformData;
        const set = (vec4: number, values: readonly number[]): void => data.set(values, vec4 * 4);
        set(UNIFORM_VEC4.volumeOrigin, [...grid.origin, resolveBias(this.settings.normalBias, grid)]);
        set(UNIFORM_VEC4.volumeSpacing, [...grid.spacing, resolveBias(this.settings.viewBias, grid)]);
        set(UNIFORM_VEC4.volumeCounts, [...grid.counts, 0]);
        set(UNIFORM_VEC4.volumeTexels, [irradianceTexels, distanceTexels, 0, 0]);
    }

    private writeUniforms(resources: PreparedResources, frame: number, count: number, raysPerProbe: number): void {
        const { grid } = resources;
        this.fillSampleParameters(grid, resources.irradianceTexels, resources.distanceTexels);
        const data = this.uniformData;
        const rotation = frameRayRotation(frame);
        for (let column = 0; column < 3; column++) {
            data.set([rotation[column * 3], rotation[column * 3 + 1], rotation[column * 3 + 2], 0], (UNIFORM_VEC4.rayRotation + column) * 4);
        }
        const blend = blendSettings(this.settings);
        data.set([blend.hysteresis, blend.irradianceChangeThreshold, blend.maxBrightnessChange, blend.distanceSharpness], UNIFORM_VEC4.blend * 4);
        data.set([effectiveMaxRayDistance(this.settings.maxRayDistance), distanceClamp(grid), blend.bounceEnergy, 0], UNIFORM_VEC4.rays * 4);
        this.uniformWords.set([raysPerProbe, count, frame >>> 0, 0], UNIFORM_VEC4.counts * 4);
    }

    private countersFor(updated: number, raysPerProbe: number): Record<string, number> {
        const resources = this.resources;
        const notUpdatedYet = resources?.cursor.notUpdatedYet ?? 0;
        const probes = resources?.grid.probeCount ?? 0;
        // Until classification (T1.5b) every probe that has had an update is active.
        return {
            'probes updated': updated,
            'rays traced': updated * raysPerProbe,
            'probes active': probes - notUpdatedYet,
            'probes off: inside geometry': 0,
            'probes off: no surface nearby': 0,
            'probes not updated yet': notUpdatedYet,
        };
    }
}

function positiveInteger(value: number, name: string): number {
    if (!Number.isInteger(value) || value < 1) throw new Error(`@pmndrs/ddgi: ${name} must be a positive whole number, got ${value}`);
    return value;
}
