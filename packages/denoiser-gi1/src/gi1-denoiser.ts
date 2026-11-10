/**
 * GI-1's image-space GI denoiser on a bare `GPUDevice`: Capsaicin v1.3's
 * `ReprojectGI` and two `FilterGI` passes (wgsl.ts), with their history kept
 * as Capsaicin keeps it (`GI1::GIDenoiser`, gi1.cpp:1176-1227; the dispatches
 * at gi1.cpp:2882-2927; the flip at 3102).
 *
 * Resources, as Capsaicin's (`ensureMemoryIsAllocated`, gi1.cpp:1193-1227):
 * two history colour textures (irradiance × samples in rgb, the sample count
 * in alpha) and two colour-delta textures, swapped every frame; the blur
 * mask; the denoised irradiance. Plus last frame's depth and normals, which
 * Capsaicin's framework keeps for it (register D68).
 *
 * Per frame, with `i` the history this frame writes:
 *
 * 1. `ReprojectGI`: this frame's input and the previous history `[1 − i]`
 *    into history `[i]`, the colour delta `[i]` and the blur mask;
 * 2. `FilterGI` horizontally: `[i]` into `[1 − i]` (the previous history is
 *    read by then, so its texture is free);
 * 3. `FilterGI` vertically: `[1 − i]` into `[i]`, and the irradiance. So the
 *    **filtered** history is what the next frame reprojects, as in Capsaicin;
 * 4. `KeepGuides`: this frame's depth and normals for the next frame.
 *
 * The interface follows `@pmndrs/denoiser`'s `TemporalDenoiser`
 * (`configure`, `dispatch`, `resetHistory`, `dispose`; types.ts), plus
 * `encode`, which records into the caller's encoder instead of submitting,
 * for a host (like the radiance cache) that runs it inside its own frame.
 */

import { DEFAULT_CELL_PIXELS, WORKGROUP_SIZE } from './constants.ts';
import { frameConstants } from './camera.ts';
import type { DiffuseInputs, FrameCamera, FrameGuides, PassTimer, TemporalConfig } from './types.ts';
import {
    CONSTANTS_BYTES,
    CONSTANTS_OFFSETS,
    ENTRY_POINT,
    FILTER_LAYOUT,
    FILTER_PASS_BYTES,
    FILTER_WGSL,
    FORMAT,
    KEEP_GUIDES_LAYOUT,
    KEEP_GUIDES_WGSL,
    REPROJECT_LAYOUT,
    REPROJECT_WGSL,
} from './wgsl.ts';

// WebGPU's flag values, so the module also loads where the GPU* globals don't exist (unit tests in Node).
const BUFFER_USAGE = { uniform: 0x40, copyDst: 0x08 } as const;
const TEXTURE_USAGE = { copySrc: 0x01, textureBinding: 0x04, storageBinding: 0x08, renderAttachment: 0x10 } as const;
const WORD_BYTES = 4;

/** Bytes per pixel of each texture the denoiser owns (for `memoryBytes`). */
const TEXEL_BYTES: Record<(typeof FORMAT)[keyof typeof FORMAT], number> = {
    rgba16float: 8,
    r32float: 4,
    rgba8snorm: 4,
};

export interface GI1DenoiserOptions {
    /**
     * Pixels across the reprojection's distance tolerance, at the pixel's
     * distance: GI-1 ties it to its screen-probe size (`probe_size` 8,
     * gi1.cpp:1839-1843). Default 8.
     */
    readonly cellPixels?: number;
    /** Label prefix for its GPU objects and timed pass. Default `'gi1-denoiser'`. */
    readonly label?: string;
    /** Timestamp writes for its compute pass, from the host's timer (optional). */
    readonly timer?: PassTimer;
}

/** The textures of the latest frame, for tests and debug views. */
export interface GI1DenoiserHistory {
    /** `g_GIDenoiser_ColorBuffer` ×2 (`rgba16float`): irradiance × samples, sample count. */
    readonly color: readonly [GPUTexture, GPUTexture];
    /** `g_GIDenoiser_ColorDeltaBuffer` ×2 (`r32float`): the smoothed luminance change. */
    readonly colorDelta: readonly [GPUTexture, GPUTexture];
    /** `g_GIDenoiser_BlurMask` (`rgba8snorm`, r): the blur radius over 8, −1/8 on sky pixels. */
    readonly blurMask: GPUTexture;
    /** Last frame's depth (`r32float`) and normals (`rgba16float`, decoded, 0 on sky). */
    readonly previousDepth: GPUTexture;
    readonly previousNormal: GPUTexture;
    /**
     * The history the latest frame wrote (`color[latest]`, after the second
     * filter pass), which the next frame reads as its previous; the other one
     * holds the first filter pass's result. Undefined before the first frame.
     */
    readonly latest: 0 | 1 | undefined;
}

interface Pipelines {
    readonly reproject: GPUComputePipeline;
    readonly filter: GPUComputePipeline;
    readonly keepGuides: GPUComputePipeline;
    readonly reprojectLayout: GPUBindGroupLayout;
    readonly filterLayout: GPUBindGroupLayout;
    readonly keepGuidesLayout: GPUBindGroupLayout;
}

/** What one size needs. */
interface Targets {
    readonly width: number;
    readonly height: number;
    readonly color: readonly [GPUTexture, GPUTexture];
    readonly colorDelta: readonly [GPUTexture, GPUTexture];
    readonly blurMask: GPUTexture;
    readonly irradiance: GPUTexture;
    readonly previousDepth: GPUTexture;
    readonly previousNormal: GPUTexture;
    readonly bytes: number;
}

/** Bind groups for one frame parity, over one set of input textures. */
interface FrameGroups {
    readonly reproject: GPUBindGroup;
    readonly filterHorizontal: GPUBindGroup;
    readonly filterVertical: GPUBindGroup;
    readonly keepGuides: GPUBindGroup;
}

interface GroupCache {
    readonly radiance: GPUTexture;
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    readonly motion: GPUTexture;
    readonly byHistory: [FrameGroups, FrameGroups];
}

export class GI1Denoiser {
    readonly name = 'gi1-denoiser';
    readonly device: GPUDevice;
    private readonly label: string;
    private readonly cellPixels: number;
    private readonly timer: PassTimer | undefined;
    private readonly pipelines: Pipelines;
    private readonly constants: GPUBuffer;
    /** `g_BlurDirection` for the two filter passes: (1, 0) then (0, 1) (gi1.cpp:2911). */
    private readonly filterPasses: readonly [GPUBuffer, GPUBuffer];
    private readonly constantWords = new ArrayBuffer(CONSTANTS_BYTES);
    private targets: Targets | null = null;
    private groups: GroupCache | null = null;
    /** Capsaicin's `color_buffer_index_`: the history this frame writes. */
    private historyIndex: 0 | 1 = 0;
    private latestHistory: 0 | 1 | undefined = undefined;
    private needsReset = true;
    /** Last frame's jitter in pixels: the previous depth was drawn with it. */
    private previousJitter: readonly [number, number] = [0, 0];

    private constructor(device: GPUDevice, options: GI1DenoiserOptions, pipelines: Pipelines) {
        this.device = device;
        this.label = options.label ?? 'gi1-denoiser';
        this.cellPixels = options.cellPixels ?? DEFAULT_CELL_PIXELS;
        this.timer = options.timer;
        this.pipelines = pipelines;
        this.constants = device.createBuffer({ label: `${this.label} constants`, size: CONSTANTS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const filterPass = (name: string, direction: readonly [number, number]): GPUBuffer => {
            const buffer = device.createBuffer({ label: `${this.label} ${name}`, size: FILTER_PASS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
            device.queue.writeBuffer(buffer, 0, new Int32Array([direction[0], direction[1], 0, 0]));
            return buffer;
        };
        this.filterPasses = [filterPass('FilterGI horizontal', [1, 0]), filterPass('FilterGI vertical', [0, 1])];
    }

    /** Compiles the three kernels. The host owns `device`; the denoiser never destroys it. */
    static async create(device: GPUDevice, options: GI1DenoiserOptions = {}): Promise<GI1Denoiser> {
        const label = options.label ?? 'gi1-denoiser';
        const pipeline = async (name: string, code: string, entryPoint: string, layoutDescriptor: GPUBindGroupLayoutDescriptor) => {
            const layout = device.createBindGroupLayout({ ...layoutDescriptor, label: `${label}/${name}` });
            const compute = await device.createComputePipelineAsync({
                label: `${label}/${name}`,
                layout: device.createPipelineLayout({ label: `${label}/${name}`, bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ label: `${label}/${name}`, code }), entryPoint },
            });
            return { compute, layout };
        };
        const [reproject, filter, keepGuides] = await Promise.all([
            pipeline(ENTRY_POINT.reproject, REPROJECT_WGSL, ENTRY_POINT.reproject, REPROJECT_LAYOUT),
            pipeline(ENTRY_POINT.filter, FILTER_WGSL, ENTRY_POINT.filter, FILTER_LAYOUT),
            pipeline(ENTRY_POINT.keepGuides, KEEP_GUIDES_WGSL, ENTRY_POINT.keepGuides, KEEP_GUIDES_LAYOUT),
        ]);
        return new GI1Denoiser(device, options, {
            reproject: reproject.compute,
            filter: filter.compute,
            keepGuides: keepGuides.compute,
            reprojectLayout: reproject.layout,
            filterLayout: filter.layout,
            keepGuidesLayout: keepGuides.layout,
        });
    }

    /** (Re)allocates for a size and drops the history. A size it already has keeps everything. */
    configure({ width, height }: TemporalConfig): void {
        if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
            throw new RangeError(`@pmndrs/denoiser-gi1: needs a size of at least 1 × 1 whole pixels, got ${width} × ${height}`);
        }
        if (this.targets && this.targets.width === width && this.targets.height === height) return;
        this.releaseTargets();
        const device = this.device;
        const sampledStorage = TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc;
        const texture = (name: string, format: GPUTextureFormat, extraUsage = 0): GPUTexture =>
            device.createTexture({ label: `${this.label} ${name}`, size: [width, height], format, usage: sampledStorage | extraUsage });
        // The history is cleared by a render pass on reset, so it can be a render attachment.
        const color: [GPUTexture, GPUTexture] = [texture('colour 0', FORMAT.color, TEXTURE_USAGE.renderAttachment), texture('colour 1', FORMAT.color, TEXTURE_USAGE.renderAttachment)];
        const colorDelta: [GPUTexture, GPUTexture] = [
            texture('colour delta 0', FORMAT.colorDelta, TEXTURE_USAGE.renderAttachment),
            texture('colour delta 1', FORMAT.colorDelta, TEXTURE_USAGE.renderAttachment),
        ];
        const formats = [FORMAT.color, FORMAT.color, FORMAT.colorDelta, FORMAT.colorDelta, FORMAT.blurMask, FORMAT.irradiance, FORMAT.previousDepth, FORMAT.previousNormal];
        this.targets = {
            width,
            height,
            color,
            colorDelta,
            blurMask: texture('blur mask', FORMAT.blurMask),
            irradiance: texture('irradiance', FORMAT.irradiance),
            previousDepth: texture('previous depth', FORMAT.previousDepth),
            previousNormal: texture('previous normal', FORMAT.previousNormal),
            bytes: width * height * formats.reduce((sum, format) => sum + TEXEL_BYTES[format], 0),
        };
        this.needsReset = true;
    }

    /** Drops the history at the next frame (a camera cut, a new scene). */
    resetHistory(): void {
        this.needsReset = true;
    }

    /**
     * Records one frame into `encoder`: `ReprojectGI`, `FilterGI` twice and
     * the guide copy, in one compute pass. Returns the denoised irradiance
     * (`rgba16float`, rgb; alpha 1), owned by the denoiser and valid until the
     * next `configure` or `dispose`. Every input must be the configured size.
     */
    encode(encoder: GPUCommandEncoder, inputs: DiffuseInputs, guides: FrameGuides, camera: FrameCamera): GPUTexture {
        const targets = this.targets;
        if (!targets) throw new Error('@pmndrs/denoiser-gi1: call configure() before encode()');
        const named: [string, GPUTexture][] = [
            ['radiance', inputs.radiance],
            ['depth', guides.depth],
            ['normal', guides.normal],
            ['motion', guides.motion],
        ];
        for (const [name, texture] of named) {
            if (texture.width !== targets.width || texture.height !== targets.height) {
                throw new Error(`@pmndrs/denoiser-gi1: ${name} is ${texture.width} × ${texture.height}, configured ${targets.width} × ${targets.height}`);
            }
        }
        if (this.needsReset) {
            this.encodeClear(encoder, targets);
            this.previousJitter = camera.jitter ?? [0, 0];
        }
        this.writeConstants(targets, guides, camera);
        const groups = this.groupsFor(targets, inputs, guides)[this.historyIndex];
        const workgroups: [number, number] = [Math.ceil(targets.width / WORKGROUP_SIZE), Math.ceil(targets.height / WORKGROUP_SIZE)];
        const { pipelines } = this;

        const pass = encoder.beginComputePass({ label: this.label, timestampWrites: this.timer?.computePass(this.label) });
        // ReprojectGI (gi1.cpp:2882-2897).
        pass.setPipeline(pipelines.reproject);
        pass.setBindGroup(0, groups.reproject);
        pass.dispatchWorkgroups(...workgroups);
        // FilterGI, horizontal then vertical (gi1.cpp:2899-2927): each dispatch is its own usage
        // scope, so a texture written by one is read by the next.
        pass.setPipeline(pipelines.filter);
        pass.setBindGroup(0, groups.filterHorizontal);
        pass.dispatchWorkgroups(...workgroups);
        pass.setBindGroup(0, groups.filterVertical);
        pass.dispatchWorkgroups(...workgroups);
        // This frame's depth and normals for the next frame's ReprojectGI.
        pass.setPipeline(pipelines.keepGuides);
        pass.setBindGroup(0, groups.keepGuides);
        pass.dispatchWorkgroups(...workgroups);
        pass.end();

        this.needsReset = false;
        this.latestHistory = this.historyIndex;
        // Flip the buffers (gi1.cpp:3102).
        this.historyIndex = this.historyIndex === 0 ? 1 : 0;
        this.previousJitter = camera.jitter ?? [0, 0];
        return targets.irradiance;
    }

    /**
     * `@pmndrs/denoiser`'s frame contract: records and submits one frame on
     * `device.queue` (no awaits, no readback) and returns the denoised
     * irradiance. Call it after the frame's inputs are submitted.
     */
    dispatch(inputs: DiffuseInputs, guides: FrameGuides, camera: FrameCamera): GPUTexture {
        const encoder = this.device.createCommandEncoder({ label: this.label });
        const output = this.encode(encoder, inputs, guides, camera);
        this.device.queue.submit([encoder.finish()]);
        return output;
    }

    /** The denoised irradiance after the latest frame, or null before `configure`. */
    get output(): GPUTexture | null {
        return this.targets?.irradiance ?? null;
    }

    /** The history and guide textures, for tests and debug views; null before `configure`. */
    get history(): GI1DenoiserHistory | null {
        const targets = this.targets;
        if (!targets) return null;
        return {
            color: targets.color,
            colorDelta: targets.colorDelta,
            blurMask: targets.blurMask,
            previousDepth: targets.previousDepth,
            previousNormal: targets.previousNormal,
            latest: this.latestHistory,
        };
    }

    /** GPU bytes it owns: its textures and uniforms. */
    get memoryBytes(): number {
        return (this.targets?.bytes ?? 0) + CONSTANTS_BYTES + 2 * FILTER_PASS_BYTES;
    }

    /** Releases its GPU resources. Never destroys the device. */
    dispose(): void {
        this.releaseTargets();
        this.constants.destroy();
        for (const buffer of this.filterPasses) buffer.destroy();
    }

    /**
     * Empties the history: both colour textures to 0, so every previous
     * sample counts 0 and is skipped (`c.w < 1`, gi1.comp:4041), as on the
     * first frame after Capsaicin allocates them; the deltas to 0.
     */
    private encodeClear(encoder: GPUCommandEncoder, targets: Targets): void {
        for (const texture of [...targets.color, ...targets.colorDelta]) {
            const pass = encoder.beginRenderPass({
                label: `${this.label} reset`,
                colorAttachments: [{ view: texture.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }],
            });
            pass.end();
        }
        this.latestHistory = undefined;
    }

    /** `GIDenoiserConstants` (wgsl.ts), by field name. */
    private writeConstants(targets: Targets, guides: FrameGuides, camera: FrameCamera): void {
        const words = this.constantWords;
        const floats = new Float32Array(words);
        const uints = new Uint32Array(words);
        const word = (field: keyof typeof CONSTANTS_OFFSETS): number => CONSTANTS_OFFSETS[field] / WORD_BYTES;
        const { width, height } = targets;
        const frame = frameConstants(camera, this.previousJitter, width, height, this.cellPixels);
        floats.set(frame.inverseViewProjection, word('inverseViewProjection'));
        floats.set(frame.previousInverseViewProjection, word('previousInverseViewProjection'));
        floats.set(frame.viewForward, word('viewForward'));
        floats[word('cellSize')] = frame.cellSize;
        uints.set([width, height], word('bufferDimensions'));
        uints[word('reversedDepth')] = frame.reversedDepth ? 1 : 0;
        uints[word('normalUnorm')] = guides.normalEncoding === 'unorm' ? 1 : 0;
        this.device.queue.writeBuffer(this.constants, 0, words);
    }

    /** Bind groups for both frame parities over these inputs, made again when an input is another texture. */
    private groupsFor(targets: Targets, inputs: DiffuseInputs, guides: FrameGuides): [FrameGroups, FrameGroups] {
        const current = this.groups;
        if (current && current.radiance === inputs.radiance && current.depth === guides.depth && current.normal === guides.normal && current.motion === guides.motion) {
            return current.byHistory;
        }
        const device = this.device;
        const { pipelines } = this;
        const view = (texture: GPUTexture): GPUTextureView => texture.createView({ baseMipLevel: 0, mipLevelCount: 1 });
        // A depth format is read through its depth aspect, as an unfilterable float.
        const depth = guides.depth.createView({ aspect: guides.depth.format.startsWith('depth') ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
        const normal = view(guides.normal);
        const motion = view(guides.motion);
        const radiance = view(inputs.radiance);
        const constants = { buffer: this.constants };
        const group = (name: string, layout: GPUBindGroupLayout, resources: GPUBindingResource[]): GPUBindGroup =>
            device.createBindGroup({ label: `${this.label}/${name}`, layout, entries: resources.map((resource, binding) => ({ binding, resource })) });
        const forHistory = (index: 0 | 1): FrameGroups => {
            const written = targets.color[index];
            const previous = targets.color[index === 0 ? 1 : 0];
            return {
                reproject: group(`ReprojectGI ${index}`, pipelines.reprojectLayout, [
                    constants,
                    radiance,
                    depth,
                    normal,
                    motion,
                    view(previous),
                    view(targets.colorDelta[index === 0 ? 1 : 0]),
                    view(targets.previousDepth),
                    view(targets.previousNormal),
                    view(targets.blurMask),
                    view(written),
                    view(targets.colorDelta[index]),
                ]),
                // gi1.cpp:2917-2922: the first pass reads [i] and writes [1 - i], the second back.
                filterHorizontal: group(`FilterGI horizontal ${index}`, pipelines.filterLayout, [
                    constants,
                    { buffer: this.filterPasses[0] },
                    view(written),
                    view(targets.blurMask),
                    depth,
                    normal,
                    view(previous),
                    view(targets.irradiance),
                ]),
                filterVertical: group(`FilterGI vertical ${index}`, pipelines.filterLayout, [
                    constants,
                    { buffer: this.filterPasses[1] },
                    view(previous),
                    view(targets.blurMask),
                    depth,
                    normal,
                    view(written),
                    view(targets.irradiance),
                ]),
                keepGuides: group(`KeepGuides ${index}`, pipelines.keepGuidesLayout, [constants, depth, normal, view(targets.previousDepth), view(targets.previousNormal)]),
            };
        };
        const byHistory: [FrameGroups, FrameGroups] = [forHistory(0), forHistory(1)];
        this.groups = { radiance: inputs.radiance, depth: guides.depth, normal: guides.normal, motion: guides.motion, byHistory };
        return byHistory;
    }

    private releaseTargets(): void {
        const targets = this.targets;
        if (!targets) return;
        for (const texture of [...targets.color, ...targets.colorDelta, targets.blurMask, targets.irradiance, targets.previousDepth, targets.previousNormal]) texture.destroy();
        this.targets = null;
        this.groups = null;
        this.historyIndex = 0;
        this.latestHistory = undefined;
    }
}
