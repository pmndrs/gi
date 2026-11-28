/**
 * Phase 2's second half (RC2.2): the passes from the blended probe atlas to
 * irradiance per pixel, as Capsaicin runs them after `BlendScreenProbes`
 * (`GI1::render`, gi1.cpp:2561-2636):
 *
 * 1. `FilterProbeMask`, once per mip level of the probe mask (gi1.cpp:2561-2585);
 * 2. `FilterScreenProbes` twice, across then down (gi1.cpp:2587-2612; RC4.2): the
 *    blended atlas is copied to a scratch atlas first (Capsaicin copies it into
 *    its other probe buffer, gi1.cpp:2540-2541), the first pass writes the
 *    scratch, the second a filtered atlas the projection reads; the blended
 *    atlas stays next frame's history (register D59);
 * 3. `ProjectScreenProbes`, over every possible ray (gi1.cpp:2618-2625);
 * 4. `InterpolateScreenProbes`, a thread per pixel (gi1.cpp:2629-2636), with the
 *    platform's SSGI occlusion, bent normal and near-field light when the
 *    technique runs SSGI (RC4.2), and then also the lit image next frame's SSGI
 *    reads (register D51).
 *
 * The kernels are in `wgsl/screen-irradiance.ts`. This file makes their
 * resources (the SH texture, the irradiance image, the filter's scratch atlas
 * and the lit image, per screen size, as Capsaicin's `ensureMemoryIsAllocated`
 * does) and records them. The radiance cache calls it at the end of its
 * screen stage; the shading module reads {@link ScreenIrradiance.irradiance}
 * by screen position.
 */

import type { FrameCamera } from '@pmndrs/gi-core/platform/contract/frame';
import type { TechniqueContext } from '@pmndrs/gi-core/platform/contract/technique';
import type { ComputeKernel } from '@pmndrs/gi-core/platform/contract/wgsl';
import type { ScreenSpaceGITextures } from '@pmndrs/gi-core/platform/contract/compute';
import { PROBE_SIZE, type ScreenProbeLayout } from './settings.ts';
import {
    BLUR_DIRECTION_BYTES,
    FILTER_GROUP,
    FILTER_WGSL,
    FILTER_PROBE_MASK_GROUP,
    FILTER_PROBE_MASK_WGSL,
    INTERPOLATE_GROUP,
    INTERPOLATE_WGSL,
    PROJECT_GROUP,
    PROJECT_WGSL,
    SCREEN_IRRADIANCE_ENTRY_POINT,
    SCREEN_IRRADIANCE_WORKGROUP,
    SCREEN_PROBES_CONSTANTS_BYTES,
    SCREEN_PROBES_CONSTANTS_OFFSETS,
} from './wgsl/screen-irradiance.ts';
import { SH_BLOCK_SIZE } from './wgsl/spherical-harmonics.ts';
import { beginTimedPass } from './timed-pass.ts';

const BUFFER_USAGE = { uniform: 0x40, copyDst: 0x08 } as const;
const TEXTURE_USAGE = { copySrc: 0x01, copyDst: 0x02, textureBinding: 0x04, storageBinding: 0x08 } as const;
const WORD_BYTES = 4;
/** `rgba16float`: four halves a texel. */
const IRRADIANCE_TEXEL_BYTES = 8;
/** `rg32uint`: one coefficient (`packHalf4`) a texel. */
const SH_TEXEL_BYTES = 8;
/** `kGI1_InvalidId` (gi1.hlsl:30), for the placeholder mask. */
const INVALID_ID = 0xffffffff;

const TECHNIQUE_ID = 'radiance-cache';

/** What the passes read from the probes and the host this frame. */
export interface ScreenIrradianceInputs {
    readonly layout: ScreenProbeLayout;
    /** The technique's `RcFrame` uniforms, written for this frame. */
    readonly frameUniforms: GPUBuffer;
    /** The compacted probes (`RcProbeSpawn`: the count, then the seeds). */
    readonly probeSpawn: GPUBuffer;
    /** The blended probe atlas (`rgba16float`). */
    readonly probeAtlas: GPUTexture;
    /** The probe mask (`r32uint`), level 0 written by `BlendScreenProbes`, with room for its mip chain. */
    readonly probeMask: GPUTexture;
    /** The host's depth and normals (`FrameScreen`). */
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    /** The platform's SSGI images for this frame (Capsaicin's `HAS_OCCLUSION`), or null to run without (ao 1, the shading normal, no near-field light). */
    readonly occlusion: ScreenSpaceGITextures | null;
    /**
     * `FrameScreen.lighting` and `FrameScreen.albedo`: with both, the
     * interpolation also writes {@link ScreenIrradiance.combined}, the lit
     * image next frame's SSGI reads (register D51). Null for none.
     */
    readonly lighting: GPUTexture | null;
    readonly albedo: GPUTexture | null;
}

/** Switches read every frame. */
export interface ScreenIrradianceOptions {
    /** Capsaicin's jittered lookup position (`RadianceCacheSettings.interpolationJitter`). */
    readonly interpolationJitter: boolean;
    /** Run `FilterScreenProbes` (`RadianceCacheSettings.probeFilter`). */
    readonly probeFilter: boolean;
    /** The second pass writes back into the atlas, the history, as Capsaicin's does (`probeFilterHistory`, D59). */
    readonly probeFilterHistory: boolean;
}

interface Kernels {
    readonly filterProbeMask: ComputeKernel;
    readonly filter: ComputeKernel;
    readonly project: ComputeKernel;
    readonly interpolate: ComputeKernel;
}

/** What one screen size needs. */
interface SizedResources {
    readonly layout: ScreenProbeLayout;
    /** `g_ScreenProbes_ProbeSHBuffer` as an `rg32uint` texture: each probe tile's nine coefficients in a 3 × 3 block (register D26). */
    readonly probeSH: GPUTexture;
    /** `g_GIDenoiser_ColorBuffer`: `rgba16float`, irradiance in rgb and the denoiser hint in alpha. */
    readonly irradiance: GPUTexture;
    /** The first filter pass's output: Capsaicin's other probe buffer (gi1.cpp:2604-2610). */
    readonly filterScratch: GPUTexture;
    /**
     * The second pass's output, which the projection reads (register D59: Capsaicin writes it back
     * into the atlas, which becomes next frame's history; ours keeps the blended atlas as history).
     */
    readonly filteredAtlas: GPUTexture;
    /** This frame's lit image (`FrameScreen.lighting` + albedo / pi x E), next frame's SSGI lighting (register D51). */
    readonly combined: GPUTexture;
    readonly bytes: number;
}

/** Bind groups over this frame's inputs, rebuilt when any input texture is another one. */
interface InputGroups {
    readonly sized: SizedResources;
    readonly inputs: ScreenIrradianceInputs;
    /** One group per mip level from 1: level − 1 read, level written. */
    readonly filterLevels: readonly GPUBindGroup[];
    /** FilterScreenProbes across (atlas to scratch), then down (scratch to atlas). */
    readonly filterAcross: GPUBindGroup;
    readonly filterDown: GPUBindGroup;
    /** The second pass into the atlas itself (Capsaicin's, D59). */
    readonly filterDownIntoAtlas: GPUBindGroup;
    /** The projection of the filtered atlas, and of the blended one when the filter is off. */
    readonly project: GPUBindGroup;
    readonly projectUnfiltered: GPUBindGroup;
    readonly interpolate: GPUBindGroup;
}

/** 1 × 1 textures the shading module reads before the first frame (see {@link ScreenIrradiance.placeholders}). */
export interface ScreenIrradiancePlaceholders {
    readonly irradiance: GPUTexture;
    readonly probeAtlas: GPUTexture;
    readonly probeMask: GPUTexture;
    readonly probeSH: GPUTexture;
    /** For the SSGI images, the host's lighting and albedo when the interpolation runs without them. */
    readonly screenImage: GPUTexture;
}

export class ScreenIrradiance {
    /** Textures the shading module binds until the first screen stage makes the real ones: no irradiance, no probes. */
    readonly placeholders: ScreenIrradiancePlaceholders;
    private sized: SizedResources | null = null;
    private groups: InputGroups | null = null;
    private readonly constantWords = new ArrayBuffer(SCREEN_PROBES_CONSTANTS_BYTES);

    private constructor(
        private readonly context: TechniqueContext,
        private readonly kernels: Kernels,
        private readonly constants: GPUBuffer,
        /** `g_BlurDirection` for the two filter passes: (1, 0) and (0, 1) (gi1.cpp:2601-2603). */
        private readonly blurDirections: readonly [GPUBuffer, GPUBuffer],
    ) {
        this.placeholders = makePlaceholders(context.device);
    }

    /** Compiles the three kernels. */
    static async create(context: TechniqueContext): Promise<ScreenIrradiance> {
        const kernel = (label: string, code: string, entryPoint: string, group: GPUBindGroupLayoutDescriptor) =>
            context.kernels.computeKernel({ label: `${TECHNIQUE_ID}/${label}`, modules: [], code, entryPoint, ownGroups: [group] });
        const entry = SCREEN_IRRADIANCE_ENTRY_POINT;
        const [filterProbeMask, filter, project, interpolate] = await Promise.all([
            kernel(entry.filterProbeMask, FILTER_PROBE_MASK_WGSL, entry.filterProbeMask, FILTER_PROBE_MASK_GROUP),
            kernel(entry.filter, FILTER_WGSL, entry.filter, FILTER_GROUP),
            kernel(entry.project, PROJECT_WGSL, entry.project, PROJECT_GROUP),
            kernel(entry.interpolate, INTERPOLATE_WGSL, entry.interpolate, INTERPOLATE_GROUP),
        ]);
        const constants = context.device.createBuffer({
            label: `${TECHNIQUE_ID} screen probes constants`,
            size: SCREEN_PROBES_CONSTANTS_BYTES,
            usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst,
        });
        const blurDirection = (x: number, y: number): GPUBuffer => {
            const buffer = context.device.createBuffer({ label: `${TECHNIQUE_ID} blur direction (${x}, ${y})`, size: BLUR_DIRECTION_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
            context.device.queue.writeBuffer(buffer, 0, new Int32Array([x, y, 0, 0]));
            return buffer;
        };
        return new ScreenIrradiance(context, { filterProbeMask, filter, project, interpolate }, constants, [blurDirection(1, 0), blurDirection(0, 1)]);
    }

    /** The latest frame's irradiance image (screen-sized), or null before the first {@link encode}. */
    get irradiance(): GPUTexture | null {
        return this.sized?.irradiance ?? null;
    }

    /** The latest frame's SH texture, or null before the first {@link encode}. */
    get probeSH(): GPUTexture | null {
        return this.sized?.probeSH ?? null;
    }

    /** The first filter pass's output (the probes filtered across only), or null before the first {@link encode}. For tests. */
    get filterScratch(): GPUTexture | null {
        return this.sized?.filterScratch ?? null;
    }

    /** The filtered probes the projection read (both passes), or null before the first {@link encode}. For tests. */
    get filteredAtlas(): GPUTexture | null {
        return this.sized?.filteredAtlas ?? null;
    }

    /**
     * The lit image the latest {@link encode} wrote (with lighting and
     * albedo; zeros otherwise), which the next frame's SSGI reads as its
     * lighting: the image for `layout`, made when the size changes.
     */
    combinedFor(layout: ScreenProbeLayout): GPUTexture {
        return this.sizedFor(layout).combined;
    }

    get memoryBytes(): number {
        return SCREEN_PROBES_CONSTANTS_BYTES + (this.sized?.bytes ?? 0);
    }

    /**
     * The SH texture for `layout`, made when the size changes. The screen
     * probes' reprojection (RC4.1) copies last frame's SH into it before
     * {@link encode} projects this frame's probes.
     */
    probeSHFor(layout: ScreenProbeLayout): GPUTexture {
        return this.sizedFor(layout).probeSH;
    }

    /**
     * Records the passes (each its own timed section), after `BlendScreenProbes` in the same
     * encoder. `camera` is the frame's (`TechniqueScreenFrame.camera`): its
     * projection sizes the interpolation's cells, its view gives the axis
     * linear depth is measured along.
     */
    encode(encoder: GPUCommandEncoder, inputs: ScreenIrradianceInputs, camera: FrameCamera, options: ScreenIrradianceOptions): void {
        const { layout } = inputs;
        const sized = this.sizedFor(layout);
        const groups = this.groupsFor(sized, inputs);
        const writeCombined = inputs.lighting !== null && inputs.albedo !== null;
        this.writeConstants(layout, camera, options.interpolationJitter, inputs.occlusion !== null, writeCombined);
        if (options.probeFilter) {
            // Every texel of the atlas into the scratch, so the second pass reads the probes the
            // first doesn't write (reprojected ones not in this frame's list) as they are (gi1.cpp:2540-2541).
            encoder.copyTextureToTexture({ texture: inputs.probeAtlas }, { texture: sized.filterScratch }, [layout.atlasSize[0], layout.atlasSize[1]]);
        }
        const { kernels } = this;
        const timer = this.context.timer;
        const maxPerDimension = this.context.device.limits.maxComputeWorkgroupsPerDimension;
        const tile = SCREEN_IRRADIANCE_WORKGROUP.tile;

        // FilterProbeMask, level by level (gi1.cpp:2572-2584): each dispatch reads the level the previous one wrote.
        const maskPass = beginTimedPass(encoder, timer, 'FilterProbeMask');
        kernels.filterProbeMask.bind(maskPass);
        groups.filterLevels.forEach((group, index) => {
            const level = index + 1;
            maskPass.setBindGroup(1, group);
            maskPass.dispatchWorkgroups(Math.ceil(Math.max(layout.probeCount[0] >> level, 1) / tile), Math.ceil(Math.max(layout.probeCount[1] >> level, 1) / tile));
        });
        maskPass.end();
        // FilterScreenProbes across, then down, over every possible ray (gi1.cpp:2590-2612).
        if (options.probeFilter) {
            const filterPass = beginTimedPass(encoder, timer, 'FilterScreenProbes');
            kernels.filter.bind(filterPass);
            for (const group of [groups.filterAcross, options.probeFilterHistory ? groups.filterDownIntoAtlas : groups.filterDown]) {
                filterPass.setBindGroup(1, group);
                dispatchSplit(filterPass, layout.maxRayCount / SCREEN_IRRADIANCE_WORKGROUP.probe, maxPerDimension);
            }
            filterPass.end();
        }
        // ProjectScreenProbes over every possible ray: one probe per workgroup (gi1.cpp:2618-2625).
        const projectPass = beginTimedPass(encoder, timer, 'ProjectScreenProbes');
        kernels.project.bind(projectPass);
        projectPass.setBindGroup(1, options.probeFilter && !options.probeFilterHistory ? groups.project : groups.projectUnfiltered);
        dispatchSplit(projectPass, layout.maxRayCount / SCREEN_IRRADIANCE_WORKGROUP.probe, maxPerDimension);
        projectPass.end();
        // InterpolateScreenProbes, a thread per pixel (gi1.cpp:2629-2636).
        const interpolatePass = beginTimedPass(encoder, timer, 'InterpolateScreenProbes');
        kernels.interpolate.bind(interpolatePass);
        interpolatePass.setBindGroup(1, groups.interpolate);
        interpolatePass.dispatchWorkgroups(Math.ceil(layout.width / tile), Math.ceil(layout.height / tile));
        interpolatePass.end();
    }

    dispose(): void {
        this.releaseSized();
        this.constants.destroy();
        for (const buffer of this.blurDirections) buffer.destroy();
        for (const texture of Object.values(this.placeholders)) texture.destroy();
    }

    /**
     * `RcScreenProbesConstants`: the camera's forward axis (linear depth),
     * Capsaicin's `cell_size` (gi1.cpp:1839-1843) and the mask's mip count.
     */
    private writeConstants(layout: ScreenProbeLayout, camera: FrameCamera, interpolationJitter: boolean, hasOcclusion: boolean, writeCombined: boolean): void {
        const words = this.constantWords;
        const floats = new Float32Array(words);
        const uints = new Uint32Array(words);
        const offsets = SCREEN_PROBES_CONSTANTS_OFFSETS;
        floats.set(viewForward(camera), offsets.viewForward / WORD_BYTES);
        floats[offsets.cellSize / WORD_BYTES] = screenProbeCellSize(camera, layout.width, layout.height);
        uints[offsets.probeMaskMipCount / WORD_BYTES] = layout.probeMaskMipCount;
        uints[offsets.interpolationJitter / WORD_BYTES] = interpolationJitter ? 1 : 0;
        uints[offsets.hasOcclusion / WORD_BYTES] = hasOcclusion ? 1 : 0;
        uints[offsets.writeCombined / WORD_BYTES] = writeCombined ? 1 : 0;
        this.context.device.queue.writeBuffer(this.constants, 0, words);
    }

    private sizedFor(layout: ScreenProbeLayout): SizedResources {
        const current = this.sized;
        if (current && current.layout.width === layout.width && current.layout.height === layout.height) return current;
        this.releaseSized();
        const device = this.context.device;
        const shSize: [number, number] = [layout.probeCount[0] * SH_BLOCK_SIZE, layout.probeCount[1] * SH_BLOCK_SIZE];
        const probeSH = device.createTexture({
            label: `${TECHNIQUE_ID} probe SH`,
            size: shSize,
            format: 'rg32uint',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc | TEXTURE_USAGE.copyDst,
        });
        const irradiance = device.createTexture({
            label: `${TECHNIQUE_ID} irradiance`,
            size: [layout.width, layout.height],
            format: 'rgba16float',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        const filterScratch = device.createTexture({
            label: `${TECHNIQUE_ID} probe filter scratch`,
            size: [layout.atlasSize[0], layout.atlasSize[1]],
            format: 'rgba16float',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc | TEXTURE_USAGE.copyDst,
        });
        const filteredAtlas = device.createTexture({
            label: `${TECHNIQUE_ID} filtered probe atlas`,
            size: [layout.atlasSize[0], layout.atlasSize[1]],
            format: 'rgba16float',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        const combined = device.createTexture({
            label: `${TECHNIQUE_ID} lit image for SSGI`,
            size: [layout.width, layout.height],
            format: 'rgba16float',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        const sized: SizedResources = {
            layout,
            probeSH,
            irradiance,
            filterScratch,
            filteredAtlas,
            combined,
            bytes:
                shSize[0] * shSize[1] * SH_TEXEL_BYTES +
                2 * layout.width * layout.height * IRRADIANCE_TEXEL_BYTES +
                2 * layout.atlasSize[0] * layout.atlasSize[1] * IRRADIANCE_TEXEL_BYTES,
        };
        this.sized = sized;
        return sized;
    }

    private groupsFor(sized: SizedResources, inputs: ScreenIrradianceInputs): InputGroups {
        const current = this.groups;
        const same =
            current &&
            current.sized === sized &&
            current.inputs.frameUniforms === inputs.frameUniforms &&
            current.inputs.probeSpawn === inputs.probeSpawn &&
            current.inputs.probeAtlas === inputs.probeAtlas &&
            current.inputs.probeMask === inputs.probeMask &&
            current.inputs.depth === inputs.depth &&
            current.inputs.normal === inputs.normal &&
            current.inputs.occlusion?.occlusion === inputs.occlusion?.occlusion &&
            current.inputs.occlusion?.nearField === inputs.occlusion?.nearField &&
            current.inputs.lighting === inputs.lighting &&
            current.inputs.albedo === inputs.albedo;
        if (same) return current;
        const device = this.context.device;
        const { kernels } = this;
        const group = (label: string, kernel: ComputeKernel, resources: GPUBindingResource[]): GPUBindGroup =>
            device.createBindGroup({ label: `${TECHNIQUE_ID}/${label}`, layout: kernel.ownLayouts[0], entries: resources.map((resource, binding) => ({ binding, resource })) });
        const maskLevel = (level: number): GPUTextureView => inputs.probeMask.createView({ baseMipLevel: level, mipLevelCount: 1 });
        const filterLevels: GPUBindGroup[] = [];
        for (let level = 1; level < inputs.layout.probeMaskMipCount; level++) {
            filterLevels.push(group(`FilterProbeMask level ${level}`, kernels.filterProbeMask, [maskLevel(level - 1), maskLevel(level)]));
        }
        const uniforms = { buffer: inputs.frameUniforms };
        const normal = screenTextureView(inputs.normal);
        const placeholderImage = this.placeholders.screenImage;
        const maskChain = (): GPUTextureView => inputs.probeMask.createView({ baseMipLevel: 0, mipLevelCount: inputs.layout.probeMaskMipCount });
        const filterPass = (label: string, from: GPUTexture, to: GPUTexture, direction: GPUBuffer): GPUBindGroup =>
            group(`${SCREEN_IRRADIANCE_ENTRY_POINT.filter} ${label}`, kernels.filter, [
                uniforms,
                { buffer: this.constants },
                screenTextureView(inputs.depth),
                normal,
                maskChain(),
                { buffer: inputs.probeSpawn },
                from.createView(),
                to.createView(),
                { buffer: direction },
            ]);
        const groups: InputGroups = {
            sized,
            inputs,
            filterLevels,
            filterAcross: filterPass('across', inputs.probeAtlas, sized.filterScratch, this.blurDirections[0]),
            filterDown: filterPass('down', sized.filterScratch, sized.filteredAtlas, this.blurDirections[1]),
            filterDownIntoAtlas: filterPass('down into the atlas', sized.filterScratch, inputs.probeAtlas, this.blurDirections[1]),
            project: group(SCREEN_IRRADIANCE_ENTRY_POINT.project, kernels.project, [uniforms, normal, { buffer: inputs.probeSpawn }, sized.filteredAtlas.createView(), sized.probeSH.createView()]),
            projectUnfiltered: group(SCREEN_IRRADIANCE_ENTRY_POINT.project, kernels.project, [uniforms, normal, { buffer: inputs.probeSpawn }, inputs.probeAtlas.createView(), sized.probeSH.createView()]),
            interpolate: group(SCREEN_IRRADIANCE_ENTRY_POINT.interpolate, kernels.interpolate, [
                uniforms,
                { buffer: this.constants },
                screenTextureView(inputs.depth),
                normal,
                // The whole mip chain: FindClosestProbe walks up it.
                maskChain(),
                sized.probeSH.createView(),
                sized.irradiance.createView(),
                (inputs.occlusion?.occlusion ?? placeholderImage).createView(),
                (inputs.occlusion?.nearField ?? placeholderImage).createView(),
                screenTextureView(inputs.lighting ?? placeholderImage),
                screenTextureView(inputs.albedo ?? placeholderImage),
                sized.combined.createView(),
            ]),
        };
        this.groups = groups;
        return groups;
    }

    private releaseSized(): void {
        const sized = this.sized;
        if (!sized) return;
        sized.probeSH.destroy();
        sized.irradiance.destroy();
        sized.filterScratch.destroy();
        sized.filteredAtlas.destroy();
        sized.combined.destroy();
        this.sized = null;
        this.groups = null;
    }
}

/**
 * The camera's forward axis in world space: the view matrix maps it to −z,
 * so it is minus the view rotation's third row (column-major: elements 2, 6, 10).
 */
export function viewForward(camera: Pick<FrameCamera, 'view'>): [number, number, number] {
    const view = camera.view;
    const x = -view[2];
    const y = -view[6];
    const z = -view[10];
    const length = Math.hypot(x, y, z);
    if (!(length > 0)) throw new Error('@pmndrs/gi: the radiance cache needs a camera view with a rotation');
    return [x / length, y / length, z / length];
}

/**
 * Capsaicin's `cell_size` (gi1.cpp:1839-1843): `tan(fovY × probe_size ×
 * max(1 / h, h / w²))`, the size per unit distance of about a probe tile's
 * pixels. The vertical field of view comes from the projection
 * (`projection[5] = 1 / tan(fovY / 2)`).
 */
export function screenProbeCellSize(camera: Pick<FrameCamera, 'projection'>, width: number, height: number): number {
    const fovY = 2 * Math.atan(1 / camera.projection[5]);
    return Math.tan(fovY * PROBE_SIZE * Math.max(1 / height, height / (width * width)));
}

/**
 * `workgroups` workgroups of a 1D kernel as `[x, y]`, so neither passes the
 * device's limit; the kernels count workgroups in rows (`rcWorkgroupIndex`).
 */
function dispatchSplit(pass: GPUComputePassEncoder, workgroups: number, maxPerDimension: number): void {
    const x = Math.max(1, Math.min(workgroups, maxPerDimension));
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
}

/** A depth texture is read through its depth aspect (as `unfilterable-float`); anything else whole. */
function screenTextureView(texture: GPUTexture): GPUTextureView {
    const isDepth = texture.format.startsWith('depth');
    return texture.createView({ aspect: isDepth ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
}

function makePlaceholders(device: GPUDevice): ScreenIrradiancePlaceholders {
    const usage = TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copyDst;
    const make = (label: string, format: GPUTextureFormat) => device.createTexture({ label: `${TECHNIQUE_ID} ${label} (before the first frame)`, size: [1, 1], format, usage });
    const probeMask = make('probe mask', 'r32uint');
    // "No probe" in the mask; the other placeholders start at zero (no irradiance, no radiance).
    device.queue.writeTexture({ texture: probeMask }, new Uint32Array([INVALID_ID]), { bytesPerRow: WORD_BYTES }, [1, 1]);
    return {
        irradiance: make('irradiance', 'rgba16float'),
        probeAtlas: make('probe atlas', 'rgba16float'),
        probeMask,
        probeSH: make('probe SH', 'rg32uint'),
        screenImage: make('screen image', 'rgba16float'),
    };
}
