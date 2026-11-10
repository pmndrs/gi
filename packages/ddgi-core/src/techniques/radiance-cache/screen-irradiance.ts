/**
 * Phase 2's second half (RC2.2): the passes from the blended probe atlas to
 * irradiance per pixel, as Capsaicin runs them after `BlendScreenProbes`
 * (`GI1::render`, gi1.cpp:2561-2636):
 *
 * 1. `FilterProbeMask`, once per mip level of the probe mask (gi1.cpp:2561-2585);
 * 2. `ProjectScreenProbes`, over every possible ray (gi1.cpp:2618-2625);
 * 3. `InterpolateScreenProbes`, a thread per pixel (gi1.cpp:2629-2636).
 *
 * The kernels are in `wgsl/screen-irradiance.ts`. This file makes their
 * resources (the SH texture and the irradiance image, per screen size, as
 * Capsaicin's `ensureMemoryIsAllocated` does) and records them. The radiance
 * cache calls it at the end of its screen stage; the shading module reads
 * {@link ScreenIrradiance.irradiance} by screen position.
 */

import type { FrameCamera } from '../../platform/contract/frame.ts';
import type { TechniqueContext } from '../../platform/contract/technique.ts';
import type { ComputeKernel } from '../../platform/contract/wgsl.ts';
import { PROBE_SIZE, type ScreenProbeLayout } from './settings.ts';
import {
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
}

interface Kernels {
    readonly filterProbeMask: ComputeKernel;
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
    readonly bytes: number;
}

/** Bind groups over this frame's inputs, rebuilt when any input texture is another one. */
interface InputGroups {
    readonly sized: SizedResources;
    readonly inputs: ScreenIrradianceInputs;
    /** One group per mip level from 1: level − 1 read, level written. */
    readonly filterLevels: readonly GPUBindGroup[];
    readonly project: GPUBindGroup;
    readonly interpolate: GPUBindGroup;
}

/** 1 × 1 textures the shading module reads before the first frame (see {@link ScreenIrradiance.placeholders}). */
export interface ScreenIrradiancePlaceholders {
    readonly irradiance: GPUTexture;
    readonly probeAtlas: GPUTexture;
    readonly probeMask: GPUTexture;
    readonly probeSH: GPUTexture;
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
    ) {
        this.placeholders = makePlaceholders(context.device);
    }

    /** Compiles the three kernels. */
    static async create(context: TechniqueContext): Promise<ScreenIrradiance> {
        const kernel = (label: string, code: string, entryPoint: string, group: GPUBindGroupLayoutDescriptor) =>
            context.kernels.computeKernel({ label: `${TECHNIQUE_ID}/${label}`, modules: [], code, entryPoint, ownGroups: [group] });
        const entry = SCREEN_IRRADIANCE_ENTRY_POINT;
        const [filterProbeMask, project, interpolate] = await Promise.all([
            kernel(entry.filterProbeMask, FILTER_PROBE_MASK_WGSL, entry.filterProbeMask, FILTER_PROBE_MASK_GROUP),
            kernel(entry.project, PROJECT_WGSL, entry.project, PROJECT_GROUP),
            kernel(entry.interpolate, INTERPOLATE_WGSL, entry.interpolate, INTERPOLATE_GROUP),
        ]);
        const constants = context.device.createBuffer({
            label: `${TECHNIQUE_ID} screen probes constants`,
            size: SCREEN_PROBES_CONSTANTS_BYTES,
            usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst,
        });
        return new ScreenIrradiance(context, { filterProbeMask, project, interpolate }, constants);
    }

    /** The latest frame's irradiance image (screen-sized), or null before the first {@link encode}. */
    get irradiance(): GPUTexture | null {
        return this.sized?.irradiance ?? null;
    }

    /** The latest frame's SH texture, or null before the first {@link encode}. */
    get probeSH(): GPUTexture | null {
        return this.sized?.probeSH ?? null;
    }

    get memoryBytes(): number {
        return SCREEN_PROBES_CONSTANTS_BYTES + (this.sized?.bytes ?? 0);
    }

    /**
     * Records the three passes, after `BlendScreenProbes` in the same
     * encoder. `camera` is the frame's (`TechniqueScreenFrame.camera`): its
     * projection sizes the interpolation's cells, its view gives the axis
     * linear depth is measured along.
     */
    encode(encoder: GPUCommandEncoder, inputs: ScreenIrradianceInputs, camera: FrameCamera): void {
        const { layout } = inputs;
        const sized = this.sizedFor(layout);
        const groups = this.groupsFor(sized, inputs);
        this.writeConstants(layout, camera);
        const { kernels } = this;
        const timer = this.context.timer;
        const maxPerDimension = this.context.device.limits.maxComputeWorkgroupsPerDimension;
        const tile = SCREEN_IRRADIANCE_WORKGROUP.tile;

        const pass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/irradiance`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/irradiance`) });
        // FilterProbeMask, level by level (gi1.cpp:2572-2584): each dispatch reads the level the previous one wrote.
        kernels.filterProbeMask.bind(pass);
        groups.filterLevels.forEach((group, index) => {
            const level = index + 1;
            pass.setBindGroup(1, group);
            pass.dispatchWorkgroups(Math.ceil(Math.max(layout.probeCount[0] >> level, 1) / tile), Math.ceil(Math.max(layout.probeCount[1] >> level, 1) / tile));
        });
        // ProjectScreenProbes over every possible ray: one probe per workgroup (gi1.cpp:2618-2625).
        kernels.project.bind(pass);
        pass.setBindGroup(1, groups.project);
        dispatchSplit(pass, layout.maxRayCount / SCREEN_IRRADIANCE_WORKGROUP.probe, maxPerDimension);
        // InterpolateScreenProbes, a thread per pixel (gi1.cpp:2629-2636).
        kernels.interpolate.bind(pass);
        pass.setBindGroup(1, groups.interpolate);
        pass.dispatchWorkgroups(Math.ceil(layout.width / tile), Math.ceil(layout.height / tile));
        pass.end();
    }

    dispose(): void {
        this.releaseSized();
        this.constants.destroy();
        for (const texture of Object.values(this.placeholders)) texture.destroy();
    }

    /**
     * `RcScreenProbesConstants`: the camera's forward axis (linear depth),
     * Capsaicin's `cell_size` (gi1.cpp:1839-1843) and the mask's mip count.
     */
    private writeConstants(layout: ScreenProbeLayout, camera: FrameCamera): void {
        const words = this.constantWords;
        const floats = new Float32Array(words);
        const uints = new Uint32Array(words);
        const offsets = SCREEN_PROBES_CONSTANTS_OFFSETS;
        floats.set(viewForward(camera), offsets.viewForward / WORD_BYTES);
        floats[offsets.cellSize / WORD_BYTES] = screenProbeCellSize(camera, layout.width, layout.height);
        uints[offsets.probeMaskMipCount / WORD_BYTES] = layout.probeMaskMipCount;
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
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        const irradiance = device.createTexture({
            label: `${TECHNIQUE_ID} irradiance`,
            size: [layout.width, layout.height],
            format: 'rgba16float',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        const sized: SizedResources = {
            layout,
            probeSH,
            irradiance,
            bytes: shSize[0] * shSize[1] * SH_TEXEL_BYTES + layout.width * layout.height * IRRADIANCE_TEXEL_BYTES,
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
            current.inputs.normal === inputs.normal;
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
        const groups: InputGroups = {
            sized,
            inputs,
            filterLevels,
            project: group(SCREEN_IRRADIANCE_ENTRY_POINT.project, kernels.project, [uniforms, normal, { buffer: inputs.probeSpawn }, inputs.probeAtlas.createView(), sized.probeSH.createView()]),
            interpolate: group(SCREEN_IRRADIANCE_ENTRY_POINT.interpolate, kernels.interpolate, [
                uniforms,
                { buffer: this.constants },
                screenTextureView(inputs.depth),
                normal,
                // The whole mip chain: FindClosestProbe walks up it.
                inputs.probeMask.createView({ baseMipLevel: 0, mipLevelCount: inputs.layout.probeMaskMipCount }),
                sized.probeSH.createView(),
                sized.irradiance.createView(),
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
    if (!(length > 0)) throw new Error('@pmndrs/ddgi: the radiance cache needs a camera view with a rotation');
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
    };
}
