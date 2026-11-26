/**
 * The previous frame's depth and normals (RC4.1, `GpuScreenHistory` in
 * platform/contract/compute.ts): one compute pass at the end of a frame
 * copies `FrameScreen.depth` and `FrameScreen.normal` into textures the
 * platform owns, for the next frame's reprojection.
 *
 * A compute copy rather than `copyTextureToTexture`: a host's depth may be
 * `depth24plus`, which can't be copied, and its normals may be 0..1 encoded,
 * which the copy decodes so a reader needs no flag for last frame's encoding.
 */

import type { GpuScreenHistory, ScreenHistoryOptions, ScreenHistoryTextures } from '../contract/compute.ts';
import type { FrameScreen } from '../contract/frame.ts';

const SHADER_STAGE_COMPUTE = 0x4;
const TEXTURE_USAGE = { copySrc: 0x01, textureBinding: 0x04, storageBinding: 0x08 } as const;
const BUFFER_USAGE = { uniform: 0x40, copyDst: 0x08 } as const;
/** `r32float` and `rgba16float` bytes per texel. */
const DEPTH_TEXEL_BYTES = 4;
const NORMAL_TEXEL_BYTES = 8;
const WORKGROUP = 8;

const COPY_WGSL = /* wgsl */ `
struct GiScreenHistoryCopy {
    size: vec2u,
    normalUnorm: u32,   // 1: the host's normals are 0..1 encoded (2n - 1 decodes them)
    pad: u32,
}
@group(0) @binding(0) var<uniform> giCopy: GiScreenHistoryCopy;
@group(0) @binding(1) var giSourceDepth: texture_2d<f32>;
@group(0) @binding(2) var giSourceNormal: texture_2d<f32>;
@group(0) @binding(3) var giHistoryDepth: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var giHistoryNormal: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= giCopy.size)) {
        return;
    }
    let depth = textureLoad(giSourceDepth, id.xy, 0).x;
    let stored = textureLoad(giSourceNormal, id.xy, 0).xyz;
    // Decoded, not normalised. A stored zero (a pixel nothing was drawn on) stays zero in either
    // encoding: decoded, a 0..1-encoded zero would read (-1, -1, -1), a real direction.
    let decoded = select(stored, 2.0 * stored - 1.0, giCopy.normalUnorm != 0u);
    let normal = select(decoded, vec3f(0.0), all(stored == vec3f(0.0)));
    textureStore(giHistoryDepth, id.xy, vec4f(depth, 0.0, 0.0, 0.0));
    textureStore(giHistoryNormal, id.xy, vec4f(normal, 0.0));
}
`;

const COPY_LAYOUT: GPUBindGroupLayoutDescriptor = {
    label: 'gi/screen-history copy',
    entries: [
        { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: SHADER_STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
        { binding: 4, visibility: SHADER_STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
};

export interface ScreenHistoryKernel {
    readonly layout: GPUBindGroupLayout;
    readonly pipeline: GPUComputePipeline;
}

/** Compiles the copy kernel (once per platform; histories share it). */
export async function compileScreenHistoryKernel(device: GPUDevice, label: string): Promise<ScreenHistoryKernel> {
    const layout = device.createBindGroupLayout({ ...COPY_LAYOUT, label: `${label} gi/screen-history` });
    const pipeline = await device.createComputePipelineAsync({
        label: `${label} gi/screen-history`,
        layout: device.createPipelineLayout({ label: `${label} gi/screen-history`, bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ label: `${label} gi/screen-history`, code: COPY_WGSL }), entryPoint: 'main' },
    });
    return { layout, pipeline };
}

interface Sized {
    readonly width: number;
    readonly height: number;
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    valid: boolean;
}

/** A depth texture is read through its depth aspect; anything else whole. */
function sourceView(texture: GPUTexture): GPUTextureView {
    const isDepth = texture.format.startsWith('depth');
    return texture.createView({ aspect: isDepth ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
}

export class ScreenHistory implements GpuScreenHistory {
    private sized: Sized | null = null;
    private readonly uniforms: GPUBuffer;
    private readonly uniformWords = new Uint32Array(4);
    private group: { readonly key: readonly unknown[]; readonly bindGroup: GPUBindGroup } | null = null;

    constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ScreenHistoryKernel,
        private readonly options: ScreenHistoryOptions,
    ) {
        this.uniforms = device.createBuffer({ label: `${options.label} screen history`, size: 16, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
    }

    texturesFor(width: number, height: number): ScreenHistoryTextures {
        const current = this.sized;
        if (current && current.width === width && current.height === height) return current;
        this.releaseSized();
        const usage = TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.copySrc;
        const sized: Sized = {
            width,
            height,
            depth: this.device.createTexture({ label: `${this.options.label} previous depth`, size: [width, height], format: 'r32float', usage }),
            normal: this.device.createTexture({ label: `${this.options.label} previous normal`, size: [width, height], format: 'rgba16float', usage }),
            valid: false,
        };
        this.sized = sized;
        return sized;
    }

    encode(encoder: GPUCommandEncoder, screen: FrameScreen): void {
        const sized = this.texturesFor(screen.width, screen.height) as Sized;
        const words = this.uniformWords;
        words.set([screen.width, screen.height, screen.normalEncoding === 'unorm' ? 1 : 0, 0]);
        this.device.queue.writeBuffer(this.uniforms, 0, words);
        const key = [sized, screen.depth, screen.normal];
        if (!this.group || this.group.key.some((value, index) => value !== key[index])) {
            this.group = {
                key,
                bindGroup: this.device.createBindGroup({
                    label: `${this.options.label} screen history`,
                    layout: this.kernel.layout,
                    entries: [
                        { binding: 0, resource: { buffer: this.uniforms } },
                        { binding: 1, resource: sourceView(screen.depth) },
                        { binding: 2, resource: sourceView(screen.normal) },
                        { binding: 3, resource: sized.depth.createView() },
                        { binding: 4, resource: sized.normal.createView() },
                    ],
                }),
            };
        }
        const pass = encoder.beginComputePass({ label: `${this.options.label} screen history` });
        pass.setPipeline(this.kernel.pipeline);
        pass.setBindGroup(0, this.group.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(screen.width / WORKGROUP), Math.ceil(screen.height / WORKGROUP));
        pass.end();
        sized.valid = true;
    }

    get memoryBytes(): number {
        const sized = this.sized;
        return sized ? sized.width * sized.height * (DEPTH_TEXEL_BYTES + NORMAL_TEXEL_BYTES) : 0;
    }

    dispose(): void {
        this.releaseSized();
        this.uniforms.destroy();
    }

    private releaseSized(): void {
        const sized = this.sized;
        if (!sized) return;
        sized.depth.destroy();
        sized.normal.destroy();
        this.sized = null;
        this.group = null;
    }
}
