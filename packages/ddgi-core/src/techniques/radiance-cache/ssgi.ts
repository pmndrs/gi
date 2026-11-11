/**
 * Capsaicin's SSGI technique as a module of the radiance cache (RC4.2;
 * `src/core/src/render_techniques/ssgi/`): its one kernel (wgsl/ssgi.ts) and
 * its two outputs, the ambient occlusion with the bent normal and the
 * near-field light, which `InterpolateScreenProbes` reads.
 *
 * Capsaicin runs SSGI as a technique of its own that writes shared textures
 * (`OcclusionAndBentNormal`, `NearFieldGlobalIllumination`, ssgi.cpp:74-81),
 * before GI-1 (gi1_renderer.cpp:60-61). Here the radiance cache owns it and
 * runs it first in its screen stage (register D57). It reads the host's
 * depth, shading normals and last frame's lit colour through `FrameScreen`
 * (`previousLighting`, D56), as Capsaicin's reads `VisibilityDepth`,
 * `ShadingNormal` and `PrevCombinedIllumination` (ssgi.cpp:79-81, 145-149).
 *
 * The outputs are made per screen size, as Capsaicin's shared textures are.
 */

import type { FrameCamera } from '../../platform/contract/frame.ts';
import type { TechniqueContext } from '../../platform/contract/technique.ts';
import type { ComputeKernel } from '../../platform/contract/wgsl.ts';
import { invertMat4 } from './camera.ts';
import { viewForward } from './screen-irradiance.ts';
import type { RadianceCacheSettings } from './settings.ts';
import { beginTimedPass, SSGI_SECTION } from './timed-pass.ts';
import { SSGI_CONSTANTS_BYTES, SSGI_CONSTANTS_OFFSETS, SSGI_ENTRY_POINT, SSGI_GROUP, SSGI_OUTPUT_FORMAT, SSGI_WGSL, SSGI_WORKGROUP_SIZE } from './wgsl/ssgi.ts';

const BUFFER_USAGE = { uniform: 0x40, copyDst: 0x08 } as const;
const TEXTURE_USAGE = { copySrc: 0x01, textureBinding: 0x04, storageBinding: 0x08 } as const;
const WORD_BYTES = 4;
/** `rgba16float`: four halves a texel. */
const OUTPUT_TEXEL_BYTES = 8;

const TECHNIQUE_ID = 'radiance-cache';

/** The SSGI settings by Capsaicin's option names (settings.ts). */
export type SSGISettings = Pick<RadianceCacheSettings, 'ssgiSliceCount' | 'ssgiStepCount' | 'ssgiViewRadius' | 'ssgiFalloffRange'>;

/** `SSGIConstants` as the kernel reads them (`RcSSGIConstants`), in plain numbers: what the CPU twin takes too. */
export interface SSGIConstants {
    /** `inv_view`, column-major. */
    readonly inverseView: readonly number[];
    /** `forward`: the camera's unit view axis in world space. */
    readonly forward: readonly [number, number, number];
    readonly uvRadius: number;
    readonly falloffMul: number;
    readonly falloffAdd: number;
    readonly sliceCount: number;
    readonly stepCount: number;
}

/**
 * Capsaicin's constants from the camera and the options (`SSGI::render`,
 * ssgi.cpp:113-135): the radius in uv per unit of linear depth, and the
 * falloff over the last `falloff_range` of the radius.
 */
export function ssgiConstants(camera: Pick<FrameCamera, 'view' | 'projection'>, settings: SSGISettings): SSGIConstants {
    const inverseView = invertMat4(camera.view);
    if (!inverseView) throw new Error('@pmndrs/ddgi: the radiance cache needs a camera view it can invert');
    const viewRadius = settings.ssgiViewRadius;
    // camera_matrices.projection[0][0] and [1][1]: column-major elements 0 and 5.
    const uvRadius = viewRadius * 0.5 * Math.max(camera.projection[0], camera.projection[5]);
    const falloffRange = viewRadius * settings.ssgiFalloffRange;
    const falloffFrom = viewRadius * (1 - settings.ssgiFalloffRange);
    return {
        inverseView,
        forward: viewForward(camera),
        uvRadius,
        falloffMul: -1 / falloffRange,
        falloffAdd: falloffFrom / falloffRange + 1,
        sliceCount: settings.ssgiSliceCount,
        stepCount: settings.ssgiStepCount,
    };
}

/** What the kernel reads this frame. */
export interface SSGIInputs {
    readonly width: number;
    readonly height: number;
    /** The technique's `RcFrame` uniforms, written for this frame. */
    readonly frameUniforms: GPUBuffer;
    /** `FrameScreen.depth`, `.normal` and `.previousLighting`. */
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    readonly previousLighting: GPUTexture;
}

interface Outputs {
    readonly width: number;
    readonly height: number;
    /** `OcclusionAndBentNormal`: rgb the bent normal as 0.5 n + 0.5, a the ambient occlusion. */
    readonly occlusionAndBentNormal: GPUTexture;
    /** `NearFieldGlobalIllumination`: rgb the near-field light. */
    readonly nearField: GPUTexture;
}

interface InputGroup {
    readonly outputs: Outputs;
    readonly inputs: SSGIInputs;
    readonly group: GPUBindGroup;
}

export class ScreenSpaceGI {
    private outputs: Outputs | null = null;
    private inputGroup: InputGroup | null = null;
    private readonly constantWords = new ArrayBuffer(SSGI_CONSTANTS_BYTES);

    private constructor(
        private readonly context: TechniqueContext,
        private readonly kernel: ComputeKernel,
        private readonly constants: GPUBuffer,
    ) {}

    static async create(context: TechniqueContext): Promise<ScreenSpaceGI> {
        const kernel = await context.kernels.computeKernel({ label: `${TECHNIQUE_ID}/${SSGI_ENTRY_POINT}`, modules: [], code: SSGI_WGSL, entryPoint: SSGI_ENTRY_POINT, ownGroups: [SSGI_GROUP] });
        const constants = context.device.createBuffer({ label: `${TECHNIQUE_ID} SSGI constants`, size: SSGI_CONSTANTS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        return new ScreenSpaceGI(context, kernel, constants);
    }

    /** The latest frame's occlusion and bent normal, or null before the first {@link encode}. */
    get occlusionAndBentNormal(): GPUTexture | null {
        return this.outputs?.occlusionAndBentNormal ?? null;
    }

    /** The latest frame's near-field light, or null before the first {@link encode}. */
    get nearField(): GPUTexture | null {
        return this.outputs?.nearField ?? null;
    }

    /** The outputs for a screen of this size, made now if they aren't (the interpolation binds them). */
    outputsFor(width: number, height: number): { readonly occlusionAndBentNormal: GPUTexture; readonly nearField: GPUTexture } {
        return this.sizedFor(width, height);
    }

    get memoryBytes(): number {
        const outputs = this.outputs;
        return SSGI_CONSTANTS_BYTES + (outputs ? 2 * outputs.width * outputs.height * OUTPUT_TEXEL_BYTES : 0);
    }

    /** Records the kernel (Capsaicin's "Main" section, ssgi.cpp:157-168): a thread per pixel. */
    encode(encoder: GPUCommandEncoder, inputs: SSGIInputs, camera: FrameCamera, settings: SSGISettings): void {
        const outputs = this.sizedFor(inputs.width, inputs.height);
        const group = this.groupFor(outputs, inputs);
        this.writeConstants(ssgiConstants(camera, settings));
        const timer = this.context.timer;
        // Capsaicin times this as the SSGI technique's "Main" section, beside GI-1's (ssgi.cpp:157).
        const pass = beginTimedPass(encoder, timer, SSGI_SECTION);
        this.kernel.bind(pass);
        pass.setBindGroup(1, group);
        pass.dispatchWorkgroups(Math.ceil(inputs.width / SSGI_WORKGROUP_SIZE), Math.ceil(inputs.height / SSGI_WORKGROUP_SIZE));
        pass.end();
    }

    dispose(): void {
        this.releaseOutputs();
        this.constants.destroy();
    }

    private writeConstants(constants: SSGIConstants): void {
        const words = this.constantWords;
        const floats = new Float32Array(words);
        const uints = new Uint32Array(words);
        const offsets = SSGI_CONSTANTS_OFFSETS;
        floats.set(constants.inverseView, offsets.inverseView / WORD_BYTES);
        floats.set(constants.forward, offsets.forward / WORD_BYTES);
        floats[offsets.uvRadius / WORD_BYTES] = constants.uvRadius;
        floats[offsets.falloffMul / WORD_BYTES] = constants.falloffMul;
        floats[offsets.falloffAdd / WORD_BYTES] = constants.falloffAdd;
        uints[offsets.sliceCount / WORD_BYTES] = constants.sliceCount;
        uints[offsets.stepCount / WORD_BYTES] = constants.stepCount;
        this.context.device.queue.writeBuffer(this.constants, 0, words);
    }

    private sizedFor(width: number, height: number): Outputs {
        const current = this.outputs;
        if (current && current.width === width && current.height === height) return current;
        this.releaseOutputs();
        const output = (label: string) =>
            this.context.device.createTexture({
                label: `${TECHNIQUE_ID} SSGI ${label}`,
                size: [width, height],
                format: SSGI_OUTPUT_FORMAT,
                usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
            });
        const outputs: Outputs = { width, height, occlusionAndBentNormal: output('occlusion and bent normal'), nearField: output('near-field light') };
        this.outputs = outputs;
        return outputs;
    }

    private groupFor(outputs: Outputs, inputs: SSGIInputs): GPUBindGroup {
        const current = this.inputGroup;
        if (
            current &&
            current.outputs === outputs &&
            current.inputs.frameUniforms === inputs.frameUniforms &&
            current.inputs.depth === inputs.depth &&
            current.inputs.normal === inputs.normal &&
            current.inputs.previousLighting === inputs.previousLighting
        ) {
            return current.group;
        }
        const resources: GPUBindingResource[] = [
            { buffer: inputs.frameUniforms },
            { buffer: this.constants },
            screenTextureView(inputs.depth),
            screenTextureView(inputs.normal),
            screenTextureView(inputs.previousLighting),
            outputs.occlusionAndBentNormal.createView(),
            outputs.nearField.createView(),
        ];
        const group = this.context.device.createBindGroup({
            label: `${TECHNIQUE_ID}/${SSGI_ENTRY_POINT}`,
            layout: this.kernel.ownLayouts[0],
            entries: resources.map((resource, binding) => ({ binding, resource })),
        });
        this.inputGroup = { outputs, inputs, group };
        return group;
    }

    private releaseOutputs(): void {
        const outputs = this.outputs;
        if (!outputs) return;
        outputs.occlusionAndBentNormal.destroy();
        outputs.nearField.destroy();
        this.outputs = null;
        this.inputGroup = null;
    }
}

/** A depth texture is read through its depth aspect (as `unfilterable-float`); anything else whole. */
function screenTextureView(texture: GPUTexture): GPUTextureView {
    const isDepth = texture.format.startsWith('depth');
    return texture.createView({ aspect: isDepth ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
}
