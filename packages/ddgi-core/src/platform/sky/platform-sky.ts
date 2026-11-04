/**
 * The platform's own copy of the sky: whatever the host set with
 * `setEnvironment`, projected by compute into an engine-neutral form that
 * hit shading reads (contract/scene.ts, `EnvironmentDescriptor`):
 *
 * - a radiance map for rays that miss: equirectangular, 256 x 128,
 *   `rgba16float`, in the contract's texture order (first row the nadir);
 * - irradiance as spherical harmonics, bands 0, 1, 2 and 4 (18 coefficients,
 *   `spherical-harmonics.ts`), already convolved with the cosine lobe.
 *
 * The projection runs inside `encode`, only when the environment changed
 * (a new descriptor from `setEnvironment`), never every frame. The host's
 * texture is read at that moment only; to pick up new contents, call
 * `setEnvironment` again.
 *
 * Sources:
 * - `texture`, `equirect` or `cube`, times `intensity`, turned by `rotationY`;
 * - `color` (times `intensity`) and `none` (black), as constant radiance;
 * - `procedural`: the calibrated Preetham sky built on the CPU by the same
 *   code as the shared sky files (`calibrated-sky.ts`), uploaded to a texture
 *   of the platform's, then projected like an equirect texture.
 *
 * Nothing here is a storage buffer a tracing kernel binds: hit shading reads
 * the map as a texture and the SH from its uniform buffer (binding budget,
 * docs/api/platform.md).
 */

import type { EnvironmentDescriptor } from '../contract/scene.ts';
import { buildCalibratedSky, DEFAULT_TURBIDITY } from './calibrated-sky.ts';
import { MIN_SUN_ELEVATION_DEG } from './clear-sky-ratio.ts';
import { halfImageToTextureRgba } from './half-float.ts';
import { MAX_TURBIDITY, MIN_TURBIDITY } from './preetham.ts';
import {
    PROJECTION_BINDING,
    PROJECTION_PARAMS_BYTES,
    PROJECTION_WORKGROUP_SIZE,
    projectionWgsl,
    RESAMPLE_WORKGROUP_SIZE,
    SH_BYTES,
    SKY_GRID_KIND,
    SKY_SOURCE_KIND,
    type SkyReduction,
} from './projection-wgsl.ts';

/** Size of the radiance map rays see on a miss: 1.4 degrees per texel, enough for a sky without a sun disc. */
export const RADIANCE_MAP_WIDTH = 256;
export const RADIANCE_MAP_HEIGHT = 128;
export const RADIANCE_MAP_FORMAT: GPUTextureFormat = 'rgba16float';

/** Largest projection grid: source texels beyond this are averaged by the sampler's bilinear taps. */
const MAX_EQUIRECT_GRID_WIDTH = 1024;
const MAX_EQUIRECT_GRID_HEIGHT = 512;
const MAX_CUBE_GRID_SIZE = 256;
/** Grid for a constant sky: the SH of a constant are exact on any grid; this one just keeps the sum short. */
const CONSTANT_GRID_WIDTH = 64;
const CONSTANT_GRID_HEIGHT = 32;
/** Most resample samples per texel along each axis. */
const MAX_SUPERSAMPLE = 8;

/**
 * Formats a float texture can have but no filtering sampler may read: these
 * use the projection's unfilterable variant (nearest samples). Unrotated, the
 * projection's cells sit on texel centres, so nearest samples are exact; with
 * `rotationY` they shift the sky by up to half a source texel. Bigger skies
 * than the radiance map get a box filter from the supersampling either way.
 */
const UNFILTERABLE_FLOAT_FORMATS: ReadonlySet<GPUTextureFormat> = new Set(['r32float', 'rg32float', 'rgba32float']);

/** `GPUTextureUsage` and `GPUBufferUsage` bits; the globals don't exist in Node, where tests import this file. */
const TEXTURE_USAGE = { copyDst: 0x2, textureBinding: 0x4, storageBinding: 0x8 } as const;
const BUFFER_USAGE = { copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const SHADER_STAGE_COMPUTE = 0x4;

export interface PlatformSkyOptions {
    readonly label?: string;
    /**
     * Which SH reduction to build. Defaults to `subgroups` when the device has
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

/** What one projection reads, worked out on the CPU. */
interface ProjectionSource {
    readonly kind: number;
    readonly constantRadiance: readonly [number, number, number];
    readonly intensity: number;
    readonly rotationY: number;
    readonly equirect: GPUTexture | null;
    readonly cube: GPUTexture | null;
    readonly filterable: boolean;
    readonly gridKind: number;
    readonly gridWidth: number;
    readonly gridHeight: number;
    readonly supersample: number;
}

interface ProjectionPipelines {
    readonly layout: GPUBindGroupLayout;
    readonly resample: GPUComputePipeline;
    readonly project: GPUComputePipeline;
}

export class PlatformSky {
    /** The radiance map: equirect, contract texture order. Never reallocated. */
    readonly radianceMap: GPUTexture;
    readonly reduction: SkyReduction;

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly warn: (message: string) => void;
    private readonly params: GPUBuffer;
    private readonly filteringSampler: GPUSampler;
    private readonly nearestSampler: GPUSampler;
    private readonly placeholder2d: GPUTexture;
    private readonly placeholderCube: GPUTexture;
    private proceduralTexture: GPUTexture | null = null;
    private pipelines: { filterable: ProjectionPipelines; unfilterable: ProjectionPipelines } | null = null;
    /** The `version` last projected; null before the first projection. */
    private projectedVersion: number | null = null;
    private projectionCount = 0;
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
        this.params = device.createBuffer({
            label: `${this.label} sky projection parameters`,
            size: PROJECTION_PARAMS_BYTES,
            usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst,
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
        // The kernels declare both a 2D and a cube source; the one not in use gets a 1-texel stand-in.
        this.placeholder2d = device.createTexture({
            label: `${this.label} sky placeholder 2d`,
            size: { width: 1, height: 1 },
            format: 'rgba16float',
            usage: TEXTURE_USAGE.textureBinding,
        });
        this.placeholderCube = device.createTexture({
            label: `${this.label} sky placeholder cube`,
            size: { width: 1, height: 1, depthOrArrayLayers: 6 },
            format: 'rgba16float',
            usage: TEXTURE_USAGE.textureBinding,
            // Compatibility-mode devices bind a texture only with the view
            // dimension it was created for; 6 layers would default to 2d-array.
            textureBindingViewDimension: 'cube',
        });
    }

    /** How many times the environment has been projected (it should be once per change). */
    get projections(): number {
        return this.projectionCount;
    }

    /** GPU bytes: the radiance map, plus the procedural sky's source texture while one is in use. */
    get memoryBytes(): number {
        const proceduralBytes = this.proceduralTexture ? this.proceduralTexture.width * this.proceduralTexture.height * 8 : 0;
        return RADIANCE_MAP_WIDTH * RADIANCE_MAP_HEIGHT * 8 + PROJECTION_PARAMS_BYTES + proceduralBytes;
    }

    /** Compiles the projection kernels (both texture-sampling variants). */
    async prepare(): Promise<void> {
        if (this.pipelines) return;
        const module = this.device.createShaderModule({ label: `${this.label} sky projection (${this.reduction})`, code: projectionWgsl(this.reduction) });
        const build = async (filterable: boolean): Promise<ProjectionPipelines> => {
            const sampleType: GPUTextureSampleType = filterable ? 'float' : 'unfilterable-float';
            const layout = this.device.createBindGroupLayout({
                label: `${this.label} sky projection (${filterable ? 'filterable' : 'unfilterable'})`,
                entries: [
                    { binding: PROJECTION_BINDING.params, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
                    { binding: PROJECTION_BINDING.sourceEquirect, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType, viewDimension: '2d' } },
                    { binding: PROJECTION_BINDING.sourceCube, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType, viewDimension: 'cube' } },
                    { binding: PROJECTION_BINDING.sourceSampler, visibility: SHADER_STAGE_COMPUTE, sampler: { type: filterable ? 'filtering' : 'non-filtering' } },
                    {
                        binding: PROJECTION_BINDING.radianceMap,
                        visibility: SHADER_STAGE_COMPUTE,
                        storageTexture: { access: 'write-only', format: RADIANCE_MAP_FORMAT, viewDimension: '2d' },
                    },
                    { binding: PROJECTION_BINDING.irradianceSh, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
                ],
            });
            const pipelineLayout = this.device.createPipelineLayout({ label: layout.label, bindGroupLayouts: [layout] });
            const [resample, project] = await Promise.all([
                this.device.createComputePipelineAsync({ label: `${this.label} sky resample`, layout: pipelineLayout, compute: { module, entryPoint: 'giSkyResample' } }),
                this.device.createComputePipelineAsync({ label: `${this.label} sky SH projection`, layout: pipelineLayout, compute: { module, entryPoint: 'giSkyProject' } }),
            ]);
            return { layout, resample, project };
        };
        const [filterable, unfilterable] = await Promise.all([build(true), build(false)]);
        this.pipelines = { filterable, unfilterable };
    }

    /**
     * Projects the environment if `version` differs from the one last
     * projected, writing the radiance map and the irradiance SH (into
     * `irradianceSh`, a range of a buffer with STORAGE usage, {@link SH_BYTES}
     * long). Records into the host's encoder; returns whether it projected.
     *
     * Why a version and not the descriptor's identity: the contract tells
     * hosts to call `setEnvironment` again after changing a texture's
     * contents, and they may pass the same object. The platform passes
     * `PlatformScene.environmentVersion`, which every call bumps.
     */
    encode(encoder: GPUCommandEncoder, environment: EnvironmentDescriptor, version: number, irradianceSh: GPUBufferBinding): boolean {
        if (this.disposed) throw new Error('@pmndrs/ddgi: the sky was disposed');
        if (version === this.projectedVersion) return false;
        const pipelines = this.pipelines;
        if (!pipelines) throw new Error('@pmndrs/ddgi: the sky projection is not compiled yet; call prepare() first');

        const source = this.sourceFor(environment);
        this.writeParams(source);
        const variant = source.filterable ? pipelines.filterable : pipelines.unfilterable;
        const bindGroup = this.device.createBindGroup({
            label: `${this.label} sky projection`,
            layout: variant.layout,
            entries: [
                { binding: PROJECTION_BINDING.params, resource: { buffer: this.params } },
                { binding: PROJECTION_BINDING.sourceEquirect, resource: (source.equirect ?? this.placeholder2d).createView({ dimension: '2d' }) },
                { binding: PROJECTION_BINDING.sourceCube, resource: (source.cube ?? this.placeholderCube).createView({ dimension: 'cube' }) },
                { binding: PROJECTION_BINDING.sourceSampler, resource: source.filterable ? this.filteringSampler : this.nearestSampler },
                { binding: PROJECTION_BINDING.radianceMap, resource: this.radianceMap.createView() },
                { binding: PROJECTION_BINDING.irradianceSh, resource: { ...irradianceSh, size: SH_BYTES } },
            ],
        });

        const pass = encoder.beginComputePass({ label: `${this.label} sky projection` });
        pass.setBindGroup(0, bindGroup);
        pass.setPipeline(variant.resample);
        pass.dispatchWorkgroups(Math.ceil(RADIANCE_MAP_WIDTH / RESAMPLE_WORKGROUP_SIZE), Math.ceil(RADIANCE_MAP_HEIGHT / RESAMPLE_WORKGROUP_SIZE));
        pass.setPipeline(variant.project);
        pass.dispatchWorkgroups(1);
        pass.end();

        this.projectedVersion = version;
        this.projectionCount++;
        return true;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.radianceMap.destroy();
        this.params.destroy();
        this.placeholder2d.destroy();
        this.placeholderCube.destroy();
        this.proceduralTexture?.destroy();
        this.proceduralTexture = null;
    }

    // -----------------------------------------------------------------------

    private sourceFor(environment: EnvironmentDescriptor): ProjectionSource {
        const constant = (radiance: readonly [number, number, number]): ProjectionSource => ({
            kind: SKY_SOURCE_KIND.constant,
            constantRadiance: radiance,
            intensity: 1,
            rotationY: 0,
            equirect: null,
            cube: null,
            filterable: true,
            gridKind: SKY_GRID_KIND.equirect,
            gridWidth: CONSTANT_GRID_WIDTH,
            gridHeight: CONSTANT_GRID_HEIGHT,
            supersample: 1,
        });

        switch (environment.type) {
            case 'none':
                return constant([0, 0, 0]);
            case 'color': {
                const intensity = environment.intensity ?? 1;
                const [r, g, b] = environment.color;
                return constant([r * intensity, g * intensity, b * intensity]);
            }
            case 'procedural':
                return this.equirectSource(this.uploadProceduralSky(environment), 1, 0);
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
                    gridKind: SKY_GRID_KIND.cube,
                    gridWidth: Math.min(faceSize, MAX_CUBE_GRID_SIZE),
                    gridHeight: Math.min(faceSize, MAX_CUBE_GRID_SIZE),
                    // A face spans 90 degrees, the radiance map 360 across 256 texels: face texels per map texel = size / 64.
                    supersample: clampSupersample(Math.ceil(faceSize / 64)),
                };
            }
        }
    }

    private equirectSource(texture: GPUTexture, intensity: number, rotationY: number): ProjectionSource {
        return {
            kind: SKY_SOURCE_KIND.equirect,
            constantRadiance: [0, 0, 0],
            intensity,
            rotationY,
            equirect: texture,
            cube: null,
            filterable: !UNFILTERABLE_FLOAT_FORMATS.has(texture.format),
            gridKind: SKY_GRID_KIND.equirect,
            // At the source's own resolution (up to a cap), cell centres land on texel centres, so the
            // bilinear sampler returns each texel exactly and the sum is the texels' own integral.
            gridWidth: Math.min(texture.width, MAX_EQUIRECT_GRID_WIDTH),
            gridHeight: Math.min(texture.height, MAX_EQUIRECT_GRID_HEIGHT),
            supersample: clampSupersample(Math.ceil(texture.width / RADIANCE_MAP_WIDTH)),
        };
    }

    /**
     * Builds the calibrated Preetham sky for a `procedural` environment with
     * the sky files' own code and uploads it: calibrated to a sun of
     * `sunIlluminance`, times `intensity` (contract/scene.ts), the same two
     * numbers the sky files use. Values outside the
     * model's range are clamped: turbidity to 1.7..20, and the sun to at least
     * 5 degrees above the horizon (same azimuth); a lower sun is M2 (time of day).
     */
    private uploadProceduralSky(environment: Extract<EnvironmentDescriptor, { type: 'procedural' }>): GPUTexture {
        const turbidity = Math.min(MAX_TURBIDITY, Math.max(MIN_TURBIDITY, environment.turbidity ?? DEFAULT_TURBIDITY));
        const sky = buildCalibratedSky({
            sunDirection: liftSunAboveMinimum(environment.sunDirection),
            turbidity,
            sunIlluminance: environment.sunIlluminance ?? 1,
            intensity: environment.intensity ?? 1,
        });
        const { width, height } = sky.image;
        if (!this.proceduralTexture || this.proceduralTexture.width !== width || this.proceduralTexture.height !== height) {
            this.proceduralTexture?.destroy();
            this.proceduralTexture = this.device.createTexture({
                label: `${this.label} procedural sky`,
                size: { width, height },
                format: 'rgba16float',
                usage: TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copyDst,
            });
        }
        const bytesPerTexel = 8;
        this.device.queue.writeTexture(
            { texture: this.proceduralTexture },
            halfImageToTextureRgba(sky.image) as Uint16Array<ArrayBuffer>,
            { bytesPerRow: width * bytesPerTexel, rowsPerImage: height },
            { width, height },
        );
        return this.proceduralTexture;
    }

    private writeParams(source: ProjectionSource): void {
        const data = new ArrayBuffer(PROJECTION_PARAMS_BYTES);
        const floats = new Float32Array(data);
        const words = new Uint32Array(data);
        floats.set(source.constantRadiance, 0);
        floats[3] = source.intensity;
        words[4] = source.kind;
        words[5] = source.gridKind;
        words[6] = source.gridWidth;
        words[7] = source.gridHeight;
        floats[8] = Math.cos(source.rotationY);
        floats[9] = Math.sin(source.rotationY);
        words[10] = source.supersample;
        const cells = source.gridWidth * source.gridHeight * (source.gridKind === SKY_GRID_KIND.cube ? 6 : 1);
        words[11] = Math.ceil(cells / PROJECTION_WORKGROUP_SIZE);
        this.device.queue.writeBuffer(this.params, 0, data);
    }
}

function clampSupersample(samples: number): number {
    return Math.min(MAX_SUPERSAMPLE, Math.max(1, samples));
}

/** The sun direction, raised to the clear-sky model's lowest elevation if it is below it (same azimuth). */
export function liftSunAboveMinimum(sunDirection: readonly [number, number, number]): [number, number, number] {
    const length = Math.hypot(sunDirection[0], sunDirection[1], sunDirection[2]);
    // A thousandth of a degree above the model's limit, so rounding can't put it just under.
    const minimumElevation = ((MIN_SUN_ELEVATION_DEG + 0.001) * Math.PI) / 180;
    if (sunDirection[1] / length >= Math.sin(minimumElevation)) return [sunDirection[0], sunDirection[1], sunDirection[2]];
    const horizontal = Math.hypot(sunDirection[0], sunDirection[2]);
    // Straight down has no azimuth: put the sun towards +X.
    const [x, z] = horizontal > 0 ? [sunDirection[0] / horizontal, sunDirection[2] / horizontal] : [1, 0];
    return [x * Math.cos(minimumElevation), Math.sin(minimumElevation), z * Math.cos(minimumElevation)];
}
