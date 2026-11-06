/**
 * The platform's own copy of the sky: whatever the host set with
 * `setEnvironment`, baked by compute into an engine-neutral form that hit
 * shading reads (contract/scene.ts, `EnvironmentDescriptor`):
 *
 * - a radiance map for rays that miss: equirectangular, 256 x 128,
 *   `rgba16float`, in the contract's texture order (first row the nadir),
 *   read times a scale (1 but for a procedural sky, whose map is written
 *   before its calibration is known; `giSkyRadiance` applies it);
 * - the sky's harmonics (`SKY_HARMONICS_LAYOUT`), integrated over that map:
 *   its irradiance as spherical harmonics, bands 0, 1, 2 and 4 (18
 *   coefficients, `spherical-harmonics.ts`), already convolved with the
 *   cosine lobe, its radiance SH, bands 0-2, which the instant sky reads
 *   (T2.6b), and the map's scale.
 *
 * The bake runs inside `encode`, only when the environment changed, never
 * every frame, at most once a frame. The host's texture is read at that
 * moment only; to pick up new contents, call `setEnvironment` again.
 *
 * Sources:
 * - `texture`, `equirect` or `cube`, times `intensity`, turned by `rotationY`;
 * - `color` (times `intensity`) and `none` (black), as constant radiance;
 * - `procedural`: the calibrated Preetham sky, evaluated by the bake's
 *   kernels on the GPU (T2.6a, `procedural-sky.ts`). The CPU code that builds
 *   the same sky as an image (`calibrated-sky.ts`) still writes the shared
 *   sky files, and the GPU test holds this map to it.
 *
 * **When a procedural sky re-bakes** (m2-lights.md §6.2): when the sun has
 * turned more than 0.05 degrees since the last bake, or the turbidity,
 * `intensity` or `sunIlluminance` changed. A host animating time of day can
 * call `setEnvironment` every frame; turns smaller than that cost nothing.
 * Every other source re-bakes on every `setEnvironment`.
 *
 * Nothing here is a storage buffer a tracing kernel binds: hit shading reads
 * the map as a texture and the harmonics from its uniform buffer (binding
 * budget, docs/api/platform.md).
 */

import type { EnvironmentDescriptor } from '../contract/scene.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import { PROCEDURAL_GRID_WIDTH, proceduralSkyNeedsRebake, proceduralSkyParameters, type GpuPerez, type ProceduralSkyParameters } from './procedural-sky.ts';
import {
    PARTIAL_SUM_COUNT,
    PEREZ_WORDS,
    PROJECTION_BINDING,
    PROJECTION_ENTRY_POINTS,
    PROJECTION_PARAMS_BYTES,
    PROJECTION_PARAMS_WORD,
    projectionWgsl,
    RADIANCE_MAP_HEIGHT,
    RADIANCE_MAP_WIDTH,
    REDUCTION_BYTES,
    RESAMPLE_WORKGROUP_SIZE,
    SKY_HARMONICS_LAYOUT,
    SKY_SOURCE_KIND,
    type SkyReduction,
} from './projection-wgsl.ts';

export { liftSunAboveMinimum } from './procedural-sky.ts';
export { RADIANCE_MAP_HEIGHT, RADIANCE_MAP_WIDTH } from './projection-wgsl.ts';

export const RADIANCE_MAP_FORMAT: GPUTextureFormat = 'rgba16float';

/** Most resample samples per texel along each axis. */
const MAX_SUPERSAMPLE = 8;

/** The name the bake's pass is timed under, in the platform's stats. */
export const SKY_BAKE_PASS = 'platform/sky bake';

/**
 * Formats a float texture can have but no filtering sampler may read: these
 * use the resample's unfilterable variant (nearest samples). Unrotated, for
 * a source a whole multiple of the map's size, the samples sit on texel
 * centres, so nearest samples are exact; with `rotationY` they shift the sky
 * by up to half a source texel. Bigger skies than the radiance map get a box
 * filter from the supersampling either way.
 */
const UNFILTERABLE_FLOAT_FORMATS: ReadonlySet<GPUTextureFormat> = new Set(['r32float', 'rg32float', 'rgba32float']);

/** `GPUTextureUsage` and `GPUBufferUsage` bits; the globals don't exist in Node, where tests import this file. */
const TEXTURE_USAGE = { textureBinding: 0x4, storageBinding: 0x8 } as const;
const BUFFER_USAGE = { copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const SHADER_STAGE_COMPUTE = 0x4;

export interface PlatformSkyOptions {
    readonly label?: string;
    /**
     * Which reduction to build. Defaults to `subgroups` when the device has
     * the feature, `workgroup` (the declared fallback) otherwise. Choosing
     * `workgroup` on a device with subgroups is how the test compares the two.
     */
    readonly reduction?: SkyReduction;
    /** Called once per distinct problem (an unusable environment texture, say). */
    readonly warn?: (message: string) => void;
}

/** Why a texture can't be a sky source, or null if it can. */
export function unusableEnvironmentTexture(texture: GPUTexture, layout: 'cube' | 'equirect'): string | null {
    const format = texture.format;
    const isFloatSampleable =
        !/(uint|sint)$/.test(format) && !format.startsWith('depth') && !format.startsWith('stencil');
    if (!isFloatSampleable) return `its format ${format} can't be sampled as float`;
    if ((texture.usage & TEXTURE_USAGE.textureBinding) === 0) return 'it lacks TEXTURE_BINDING usage';
    if (texture.dimension !== '2d') return `it is a ${texture.dimension} texture, not 2d`;
    if (layout === 'cube' && texture.depthOrArrayLayers !== 6) return `a cube needs 6 layers, it has ${texture.depthOrArrayLayers}`;
    if (layout === 'cube' && texture.width !== texture.height) return `cube faces must be square, it is ${texture.width} x ${texture.height}`;
    if (layout === 'equirect' && texture.depthOrArrayLayers !== 1) return `an equirect texture has 1 layer, it has ${texture.depthOrArrayLayers}`;
    return null;
}

/** What one bake reads, worked out on the CPU. */
interface BakeSource {
    readonly kind: number;
    readonly constantRadiance: readonly [number, number, number];
    readonly intensity: number;
    readonly rotationY: number;
    readonly equirect: GPUTexture | null;
    readonly cube: GPUTexture | null;
    readonly filterable: boolean;
    readonly supersample: number;
    /** Only for `procedural`. */
    readonly procedural: ProceduralSkyParameters | null;
}

/** The resample for one kind of source texture: filterable, or not (32-bit floats). */
interface ResamplePipeline {
    readonly sourceLayout: GPUBindGroupLayout;
    readonly pipeline: GPUComputePipeline;
}

interface BakePipelines {
    readonly sharedLayout: GPUBindGroupLayout;
    readonly resample: { readonly filterable: ResamplePipeline; readonly unfilterable: ResamplePipeline };
    /** The partial sums read the map through a group 1 of their own. */
    readonly mapInputLayout: GPUBindGroupLayout;
    readonly partialSums: GPUComputePipeline;
    readonly sum: GPUComputePipeline;
}

/** A bind group, and what it was made for, so a bake of the same textures (every procedural bake) makes none. */
interface Cached<Key> {
    readonly group: GPUBindGroup;
    readonly key: Key;
}

interface SourceKey {
    readonly equirect: GPUTexture | null;
    readonly cube: GPUTexture | null;
    readonly filterable: boolean;
}

interface HarmonicsKey {
    readonly buffer: GPUBuffer;
    readonly offset: number;
}

export class PlatformSky {
    /** The radiance map: equirect, contract texture order. Never reallocated. */
    readonly radianceMap: GPUTexture;
    readonly reduction: SkyReduction;

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly warn: (message: string) => void;
    private readonly params: GPUBuffer;
    /** The bake's partial sums and calibration scale (`GiSkyReduction`). */
    private readonly reductionBuffer: GPUBuffer;
    private readonly paramsData = new ArrayBuffer(PROJECTION_PARAMS_BYTES);
    private readonly filteringSampler: GPUSampler;
    private readonly nearestSampler: GPUSampler;
    private readonly placeholder2dView: GPUTextureView;
    private readonly placeholderCubeView: GPUTextureView;
    private readonly radianceMapView: GPUTextureView;
    private readonly placeholders: GPUTexture[];
    private pipelines: BakePipelines | null = null;
    private sharedGroup: Cached<HarmonicsKey> | null = null;
    private sourceGroup: Cached<SourceKey> | null = null;
    private mapInputGroup: GPUBindGroup | null = null;
    /** The `version` last looked at; null before the first bake. */
    private seenVersion: number | null = null;
    /** The procedural sky's numbers at the last bake; null when the last bake was another source. */
    private bakedProcedural: ProceduralSkyParameters | null = null;
    private bakeCount = 0;
    private disposed = false;

    constructor(device: GPUDevice, options: PlatformSkyOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'gi';
        this.warn = options.warn ?? ((message) => console.warn(message));
        this.reduction = options.reduction ?? (device.features.has('subgroups') ? 'subgroups' : 'workgroup');
        if (this.reduction === 'subgroups' && !device.features.has('subgroups')) {
            throw new Error("@pmndrs/ddgi: the sky's subgroup reduction needs a device with the 'subgroups' feature");
        }

        this.radianceMap = device.createTexture({
            label: `${this.label} sky radiance map`,
            size: { width: RADIANCE_MAP_WIDTH, height: RADIANCE_MAP_HEIGHT },
            format: RADIANCE_MAP_FORMAT,
            usage: TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.storageBinding,
        });
        this.radianceMapView = this.radianceMap.createView({ label: `${this.label} sky radiance map (bake)` });
        this.params = device.createBuffer({
            label: `${this.label} sky bake parameters`,
            size: PROJECTION_PARAMS_BYTES,
            usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst,
        });
        this.reductionBuffer = device.createBuffer({
            label: `${this.label} sky bake partial sums`,
            size: REDUCTION_BYTES,
            usage: BUFFER_USAGE.storage,
        });
        // Repeat across the -X seam of an equirect map, clamp at the poles.
        this.filteringSampler = device.createSampler({
            label: `${this.label} sky source (linear)`,
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'repeat',
            addressModeV: 'clamp-to-edge',
        });
        this.nearestSampler = device.createSampler({
            label: `${this.label} sky source (nearest)`,
            addressModeU: 'repeat',
            addressModeV: 'clamp-to-edge',
        });
        // The kernels declare both a 2D and a cube source; the one not in use (both, for
        // a colour or a procedural sky) gets a 1-texel stand-in.
        const placeholder2d = device.createTexture({
            label: `${this.label} sky placeholder 2d`,
            size: { width: 1, height: 1 },
            format: 'rgba16float',
            usage: TEXTURE_USAGE.textureBinding,
        });
        const placeholderCube = device.createTexture({
            label: `${this.label} sky placeholder cube`,
            size: { width: 1, height: 1, depthOrArrayLayers: 6 },
            format: 'rgba16float',
            usage: TEXTURE_USAGE.textureBinding,
            // Compatibility-mode devices bind a texture only with the view
            // dimension it was created for; 6 layers would default to 2d-array.
            textureBindingViewDimension: 'cube',
        });
        this.placeholders = [placeholder2d, placeholderCube];
        this.placeholder2dView = placeholder2d.createView({ dimension: '2d' });
        this.placeholderCubeView = placeholderCube.createView({ dimension: 'cube' });
    }

    /** How many times the environment has been baked (once per change that needs it). */
    get projections(): number {
        return this.bakeCount;
    }

    /** GPU bytes: the radiance map, the parameters and the partial sums. */
    get memoryBytes(): number {
        return RADIANCE_MAP_WIDTH * RADIANCE_MAP_HEIGHT * 8 + PROJECTION_PARAMS_BYTES + REDUCTION_BYTES;
    }

    /** Compiles the bake's kernels: the resample (for filterable and unfilterable sources), the partial sums, the sum. */
    async prepare(): Promise<void> {
        if (this.pipelines) return;
        const device = this.device;
        const module = device.createShaderModule({ label: `${this.label} sky bake (${this.reduction})`, code: projectionWgsl(this.reduction) });
        const sharedLayout = device.createBindGroupLayout({
            label: `${this.label} sky bake`,
            entries: [
                { binding: PROJECTION_BINDING.params, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
                { binding: PROJECTION_BINDING.reduction, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
                { binding: PROJECTION_BINDING.harmonics, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const sourceLayout = (filterable: boolean): GPUBindGroupLayout => {
            const sampleType: GPUTextureSampleType = filterable ? 'float' : 'unfilterable-float';
            return device.createBindGroupLayout({
                label: `${this.label} sky source (${filterable ? 'filterable' : 'unfilterable'})`,
                entries: [
                    { binding: PROJECTION_BINDING.sourceEquirect, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType, viewDimension: '2d' } },
                    { binding: PROJECTION_BINDING.sourceCube, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType, viewDimension: 'cube' } },
                    { binding: PROJECTION_BINDING.sourceSampler, visibility: SHADER_STAGE_COMPUTE, sampler: { type: filterable ? 'filtering' : 'non-filtering' } },
                    {
                        binding: PROJECTION_BINDING.radianceMapOutput,
                        visibility: SHADER_STAGE_COMPUTE,
                        storageTexture: { access: 'write-only', format: RADIANCE_MAP_FORMAT, viewDimension: '2d' },
                    },
                ],
            });
        };
        // The partial sums only load texels, so the map's binding needs no filtering.
        const mapInputLayout = device.createBindGroupLayout({
            label: `${this.label} sky map input`,
            entries: [{ binding: PROJECTION_BINDING.radianceMapInput, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } }],
        });
        const pipeline = (name: string, entryPoint: string, groups: GPUBindGroupLayout[]): Promise<GPUComputePipeline> =>
            device.createComputePipelineAsync({
                label: `${this.label} sky ${name}`,
                layout: device.createPipelineLayout({ label: `${this.label} sky ${name}`, bindGroupLayouts: groups }),
                compute: { module, entryPoint },
            });
        const resample = async (filterable: boolean): Promise<ResamplePipeline> => {
            const layout = sourceLayout(filterable);
            return { sourceLayout: layout, pipeline: await pipeline(`resample${filterable ? '' : ' (unfilterable)'}`, PROJECTION_ENTRY_POINTS.resample, [sharedLayout, layout]) };
        };
        // The partial sums first: the slowest of the four to compile.
        const [partialSums, sum, filterable, unfilterable] = await Promise.all([
            pipeline('partial sums', PROJECTION_ENTRY_POINTS.partialSums, [sharedLayout, mapInputLayout]),
            pipeline('SH sum', PROJECTION_ENTRY_POINTS.sum, [sharedLayout]),
            resample(true),
            resample(false),
        ]);
        this.pipelines = { sharedLayout, resample: { filterable, unfilterable }, mapInputLayout, partialSums, sum };
    }

    /**
     * Bakes the environment if `version` differs from the one last seen and
     * the change needs a bake (the file comment), writing the radiance map and
     * the sky's harmonics into `harmonics`, a range of a buffer with STORAGE
     * usage, {@link SKY_HARMONICS_LAYOUT}.byteLength long. Records into the
     * host's encoder, timed as {@link SKY_BAKE_PASS}; returns whether it baked.
     *
     * Why a version and not the descriptor's identity: the contract tells
     * hosts to call `setEnvironment` again after changing a texture's
     * contents, and they may pass the same object. The platform passes
     * `PlatformScene.environmentVersion`, which every call bumps.
     */
    encode(encoder: GPUCommandEncoder, environment: EnvironmentDescriptor, version: number, harmonics: GPUBufferBinding, timer?: GpuTimerScope): boolean {
        if (this.disposed) throw new Error('@pmndrs/ddgi: the sky was disposed');
        if (version === this.seenVersion) return false;
        const pipelines = this.pipelines;
        if (!pipelines) throw new Error('@pmndrs/ddgi: the sky bake is not compiled yet; call prepare() first');
        this.seenVersion = version;

        const procedural = environment.type === 'procedural' ? proceduralSkyParameters(environment) : null;
        if (procedural && this.bakedProcedural && !proceduralSkyNeedsRebake(this.bakedProcedural, procedural)) return false;

        const source = procedural ? this.proceduralSource(procedural) : this.sourceFor(environment as Exclude<EnvironmentDescriptor, { type: 'procedural' }>);
        this.writeParams(source);
        const resample = source.filterable ? pipelines.resample.filterable : pipelines.resample.unfilterable;

        const pass = encoder.beginComputePass({ label: `${this.label} sky bake`, timestampWrites: timer?.computePass(SKY_BAKE_PASS) });
        pass.setBindGroup(0, this.sharedGroupFor(pipelines, harmonics));
        // The map first: the sums integrate it.
        pass.setBindGroup(1, this.sourceGroupFor(source, resample));
        pass.setPipeline(resample.pipeline);
        pass.dispatchWorkgroups(Math.ceil(RADIANCE_MAP_WIDTH / RESAMPLE_WORKGROUP_SIZE), Math.ceil(RADIANCE_MAP_HEIGHT / RESAMPLE_WORKGROUP_SIZE));
        pass.setBindGroup(1, this.mapInputGroupFor(pipelines));
        pass.setPipeline(pipelines.partialSums);
        pass.dispatchWorkgroups(PARTIAL_SUM_COUNT);
        pass.setPipeline(pipelines.sum);
        pass.dispatchWorkgroups(1);
        pass.end();

        this.bakedProcedural = procedural;
        this.bakeCount++;
        return true;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.radianceMap.destroy();
        this.params.destroy();
        this.reductionBuffer.destroy();
        for (const texture of this.placeholders) texture.destroy();
        this.sharedGroup = null;
        this.sourceGroup = null;
        this.mapInputGroup = null;
    }

    // -----------------------------------------------------------------------

    private sharedGroupFor(pipelines: BakePipelines, harmonics: GPUBufferBinding): GPUBindGroup {
        const key: HarmonicsKey = { buffer: harmonics.buffer, offset: harmonics.offset ?? 0 };
        const cached = this.sharedGroup;
        if (cached && cached.key.buffer === key.buffer && cached.key.offset === key.offset) return cached.group;
        const group = this.device.createBindGroup({
            label: `${this.label} sky bake`,
            layout: pipelines.sharedLayout,
            entries: [
                { binding: PROJECTION_BINDING.params, resource: { buffer: this.params } },
                { binding: PROJECTION_BINDING.reduction, resource: { buffer: this.reductionBuffer } },
                { binding: PROJECTION_BINDING.harmonics, resource: { buffer: key.buffer, offset: key.offset, size: SKY_HARMONICS_LAYOUT.byteLength } },
            ],
        });
        this.sharedGroup = { group, key };
        return group;
    }

    private sourceGroupFor(source: BakeSource, resample: ResamplePipeline): GPUBindGroup {
        const cached = this.sourceGroup;
        if (cached && cached.key.equirect === source.equirect && cached.key.cube === source.cube && cached.key.filterable === source.filterable) return cached.group;
        const group = this.device.createBindGroup({
            label: `${this.label} sky source`,
            layout: resample.sourceLayout,
            entries: [
                { binding: PROJECTION_BINDING.sourceEquirect, resource: source.equirect ? source.equirect.createView({ dimension: '2d' }) : this.placeholder2dView },
                { binding: PROJECTION_BINDING.sourceCube, resource: source.cube ? source.cube.createView({ dimension: 'cube' }) : this.placeholderCubeView },
                { binding: PROJECTION_BINDING.sourceSampler, resource: source.filterable ? this.filteringSampler : this.nearestSampler },
                { binding: PROJECTION_BINDING.radianceMapOutput, resource: this.radianceMapView },
            ],
        });
        this.sourceGroup = { group, key: { equirect: source.equirect, cube: source.cube, filterable: source.filterable } };
        return group;
    }

    private mapInputGroupFor(pipelines: BakePipelines): GPUBindGroup {
        this.mapInputGroup ??= this.device.createBindGroup({
            label: `${this.label} sky map input`,
            layout: pipelines.mapInputLayout,
            entries: [{ binding: PROJECTION_BINDING.radianceMapInput, resource: this.radianceMapView }],
        });
        return this.mapInputGroup;
    }

    private proceduralSource(procedural: ProceduralSkyParameters): BakeSource {
        return {
            kind: SKY_SOURCE_KIND.procedural,
            constantRadiance: [0, 0, 0],
            intensity: 1,
            rotationY: 0,
            equirect: null,
            cube: null,
            filterable: true,
            // The CPU sky has 2 x 2 of its texels in each map texel; the resample evaluates
            // the sky at their centres, as it sampled the CPU sky's texture in M1.
            supersample: PROCEDURAL_GRID_WIDTH / RADIANCE_MAP_WIDTH,
            procedural,
        };
    }

    private sourceFor(environment: Exclude<EnvironmentDescriptor, { type: 'procedural' }>): BakeSource {
        const constant = (radiance: readonly [number, number, number]): BakeSource => ({
            kind: SKY_SOURCE_KIND.constant,
            constantRadiance: radiance,
            intensity: 1,
            rotationY: 0,
            equirect: null,
            cube: null,
            filterable: true,
            supersample: 1,
            procedural: null,
        });

        switch (environment.type) {
            case 'none':
                return constant([0, 0, 0]);
            case 'color': {
                const intensity = environment.intensity ?? 1;
                const [r, g, b] = environment.color;
                return constant([r * intensity, g * intensity, b * intensity]);
            }
            case 'texture': {
                const problem = unusableEnvironmentTexture(environment.texture, environment.layout);
                if (problem) {
                    this.warn(`@pmndrs/ddgi: the environment texture can't be used (${problem}); the sky is black until setEnvironment gets one that can.`);
                    return constant([0, 0, 0]);
                }
                const intensity = environment.intensity ?? 1;
                const rotationY = environment.rotationY ?? 0;
                if (environment.layout === 'equirect') return this.equirectSource(environment.texture, intensity, rotationY);
                const faceSize = environment.texture.width;
                return {
                    kind: SKY_SOURCE_KIND.cube,
                    constantRadiance: [0, 0, 0],
                    intensity,
                    rotationY,
                    equirect: null,
                    cube: environment.texture,
                    filterable: !UNFILTERABLE_FLOAT_FORMATS.has(environment.texture.format),
                    // A face spans 90 degrees, the radiance map 360 across 256 texels: face texels per map texel = size / 64.
                    supersample: clampSupersample(Math.ceil(faceSize / 64)),
                    procedural: null,
                };
            }
        }
    }

    private equirectSource(texture: GPUTexture, intensity: number, rotationY: number): BakeSource {
        return {
            kind: SKY_SOURCE_KIND.equirect,
            constantRadiance: [0, 0, 0],
            intensity,
            rotationY,
            equirect: texture,
            cube: null,
            filterable: !UNFILTERABLE_FLOAT_FORMATS.has(texture.format),
            supersample: clampSupersample(Math.ceil(texture.width / RADIANCE_MAP_WIDTH)),
            procedural: null,
        };
    }

    /** Writes the bake's parameters. */
    private writeParams(source: BakeSource): void {
        const floats = new Float32Array(this.paramsData);
        const words = new Uint32Array(this.paramsData);
        const sky = source.procedural;
        floats.set(source.constantRadiance, PROJECTION_PARAMS_WORD.constantRadiance);
        floats[PROJECTION_PARAMS_WORD.intensity] = source.intensity;
        floats.set(sky ? sky.sunDirection : [0, 1, 0], PROJECTION_PARAMS_WORD.sunDirection);
        words[PROJECTION_PARAMS_WORD.sourceKind] = source.kind;
        floats[PROJECTION_PARAMS_WORD.rotationCos] = Math.cos(source.rotationY);
        floats[PROJECTION_PARAMS_WORD.rotationSin] = Math.sin(source.rotationY);
        words[PROJECTION_PARAMS_WORD.supersample] = source.supersample;
        floats[PROJECTION_PARAMS_WORD.targetHorizontalLuminance] = sky?.targetHorizontalLuminance ?? 0;
        writePerez(floats, PROJECTION_PARAMS_WORD.perezLuminance, sky?.luminance);
        writePerez(floats, PROJECTION_PARAMS_WORD.perezX, sky?.chromaticityX);
        writePerez(floats, PROJECTION_PARAMS_WORD.perezY, sky?.chromaticityY);
        this.device.queue.writeBuffer(this.params, 0, this.paramsData);
    }
}

function writePerez(floats: Float32Array, offset: number, perez: GpuPerez | undefined): void {
    const values = perez ? [perez.a, perez.b, perez.c, perez.d, perez.e, perez.zenithScale] : [0, 0, 0, 0, 0, 0];
    floats.set(values, offset);
    floats.fill(0, offset + values.length, offset + PEREZ_WORDS);
}

function clampSupersample(samples: number): number {
    return Math.min(MAX_SUPERSAMPLE, Math.max(1, samples));
}
