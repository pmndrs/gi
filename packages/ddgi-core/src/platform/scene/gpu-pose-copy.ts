/**
 * Deformed positions the host hands over on the GPU (`setDeformedPositions`
 * with a {@link DeformedPositions}), copied into the scene copy's vertex
 * buffer by `gi/pose-copy` (T3.3, docs/design/m3-moving.md §3 and §7).
 *
 * **Why a kernel, not `copyBufferToBuffer`.** The scene copy keeps a mesh's
 * positions as tight xyz floats (3 words a vertex, mesh-packing.ts), while a
 * host's buffer may have any stride (a vec4 per vertex is common), and the
 * texture form (three's `computeSkinning` writing a storage `ExternalTexture`,
 * NOTES) arrives as one rgba32float texel per vertex. One small kernel takes
 * any stride, so every GPU pose is one GPU copy into the scene copy, and the
 * tree's refit then reads the scene copy (ray-queries.ts): rays and hit
 * shading always see the same positions.
 *
 * **The texture form** first goes through `copyTextureToBuffer` into a
 * scratch buffer of this class (4 floats a vertex), then the same kernel.
 *
 * Every copy is recorded into the host's encoder in the scene's `encode`,
 * before anything that reads the vertex buffer, and before a growth of the
 * vertex buffer (whose copies then carry the new positions over).
 */

import type { DeformedPositions } from '../contract/scene.ts';

/** The timed pass's name (`stats.passes`). */
export const POSE_COPY_PASS = 'gi/pose-copy';

const WORKGROUP_SIZE = 64;
/** Each copy's parameters sit in their own 256-byte slot of one uniform buffer, picked with a dynamic offset. */
const PARAM_SLOT_BYTES = 256;
const PARAM_WORDS = 4;
/** rgba32float: one texel, 16 bytes, per vertex. */
const TEXEL_BYTES = 16;
const TEXEL_FLOATS = 4;
/** `copyTextureToBuffer` needs rows of a multiple of 256 bytes: 16 rgba32float texels. */
export const POSE_TEXTURE_WIDTH_MULTIPLE = 256 / TEXEL_BYTES;
/** WebGPU's usage bits, as numbers so the checks also run where the GPU globals don't exist (unit tests). */
const BUFFER_USAGE_STORAGE = 0x80;
const TEXTURE_USAGE_COPY_SRC = 0x1;
/** Workgroups along x in one dispatch: the default `maxComputeWorkgroupsPerDimension`. */
const MAX_WORKGROUPS_PER_DIMENSION = 65535;

const POSE_COPY_WGSL = /* wgsl */ `
struct PoseCopyParams {
    vertexCount: u32,
    sourceOffsetFloats: u32,   // float index of vertex 0's x in the source
    sourceStrideFloats: u32,   // floats from one vertex to the next in the source
    destinationWord: u32,      // word index of vertex 0's x in the vertex buffer
}

@group(0) @binding(0) var<storage, read> poseSource: array<f32>;
@group(0) @binding(1) var<storage, read_write> poseVertices: array<u32>;
@group(0) @binding(2) var<uniform> poseParams: PoseCopyParams;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn poseCopy(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let vertex = id.x + id.y * groups.x * ${WORKGROUP_SIZE}u;
    if (vertex >= poseParams.vertexCount) {
        return;
    }
    let source = poseParams.sourceOffsetFloats + vertex * poseParams.sourceStrideFloats;
    let destination = poseParams.destinationWord + vertex * 3u;
    // The scene copy stores floats as their bits (its buffer is array<u32>).
    poseVertices[destination] = bitcast<u32>(poseSource[source]);
    poseVertices[destination + 1u] = bitcast<u32>(poseSource[source + 1u]);
    poseVertices[destination + 2u] = bitcast<u32>(poseSource[source + 2u]);
}
`;

/** A checked GPU pose, as the scene keeps it until its `encode`. */
export type GpuPose =
    | { readonly kind: 'buffer'; readonly buffer: GPUBuffer; readonly offsetFloats: number; readonly strideFloats: number }
    | { readonly kind: 'texture'; readonly texture: GPUTexture };

/**
 * Checks a {@link DeformedPositions} for a mesh of `vertexCount` vertices and
 * returns it as a {@link GpuPose}; throws a plain-language error otherwise.
 */
export function checkedGpuPose(positions: DeformedPositions, vertexCount: number, what: string): GpuPose {
    if ('texture' in positions) {
        const texture = positions.texture;
        if (texture.format !== 'rgba32float') throw new TypeError(`@pmndrs/gi: ${what}: a positions texture must be rgba32float, got ${texture.format}`);
        if ((texture.usage & TEXTURE_USAGE_COPY_SRC) === 0) throw new TypeError(`@pmndrs/gi: ${what}: a positions texture needs COPY_SRC usage`);
        if (texture.width % POSE_TEXTURE_WIDTH_MULTIPLE !== 0) {
            throw new RangeError(`@pmndrs/gi: ${what}: a positions texture's width must be a multiple of ${POSE_TEXTURE_WIDTH_MULTIPLE} (whole 256-byte rows), got ${texture.width}`);
        }
        if (texture.width * texture.height < vertexCount) throw new RangeError(`@pmndrs/gi: ${what}: the positions texture holds ${texture.width * texture.height} texels for ${vertexCount} vertices`);
        return { kind: 'texture', texture };
    }
    const buffer = positions.buffer;
    const offset = positions.offset ?? 0;
    const stride = positions.stride ?? 12;
    if (!(Number.isInteger(offset) && offset >= 0 && offset % 4 === 0)) throw new RangeError(`@pmndrs/gi: ${what}: offset must be a whole number of 4-byte words, got ${offset}`);
    if (!(Number.isInteger(stride) && stride >= 12 && stride % 4 === 0)) throw new RangeError(`@pmndrs/gi: ${what}: stride must be at least 12 and a multiple of 4, got ${stride}`);
    if ((buffer.usage & BUFFER_USAGE_STORAGE) === 0) throw new TypeError(`@pmndrs/gi: ${what}: a positions buffer needs STORAGE usage`);
    const end = offset + (vertexCount - 1) * stride + 12;
    if (buffer.size < end) throw new RangeError(`@pmndrs/gi: ${what}: the positions buffer has ${buffer.size} bytes, but ${vertexCount} vertices need ${end}`);
    return { kind: 'buffer', buffer, offsetFloats: offset / 4, strideFloats: stride / 4 };
}

/** One pose to copy this frame. */
export interface PoseCopy {
    readonly pose: GpuPose;
    readonly vertexCount: number;
    /** Word index of the mesh's first position in the vertex buffer. */
    readonly destinationWord: number;
}

export class GpuPoseCopier {
    private readonly device: GPUDevice;
    private readonly label: string;
    private pipeline: GPUComputePipeline | null = null;
    private layout: GPUBindGroupLayout | null = null;
    private compiling: Promise<void> | null = null;
    private params: GPUBuffer | null = null;
    /** Where texture poses land before the kernel (4 floats a vertex); grows as needed. */
    private textureScratch: GPUBuffer | null = null;
    /** Buffers replaced by bigger ones: destroyed at the next copy, after the host's submit. */
    private retired: GPUBuffer[] = [];

    constructor(device: GPUDevice, label: string) {
        this.device = device;
        this.label = `${label} ${POSE_COPY_PASS}`;
    }

    /** Asks for the kernel's compile (once). */
    prepare(): Promise<void> {
        this.compiling ??= (async () => {
            const layout = this.device.createBindGroupLayout({
                label: this.label,
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
                ],
            });
            const module = this.device.createShaderModule({ label: this.label, code: POSE_COPY_WGSL });
            this.pipeline = await this.device.createComputePipelineAsync({
                label: this.label,
                layout: this.device.createPipelineLayout({ label: this.label, bindGroupLayouts: [layout] }),
                compute: { module, entryPoint: 'poseCopy' },
            });
            this.layout = layout;
        })();
        return this.compiling;
    }

    get ready(): boolean {
        return this.pipeline !== null;
    }

    /** GPU bytes held (parameters and the texture scratch). */
    get memoryBytes(): number {
        return (this.params?.size ?? 0) + (this.textureScratch?.size ?? 0);
    }

    /** Records every copy into `encoder`: texture poses' `copyTextureToBuffer` first, then one pass with a dispatch per pose. */
    encode(encoder: GPUCommandEncoder, vertexBuffer: GPUBuffer, copies: readonly PoseCopy[]): void {
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
        if (copies.length === 0) return;
        const pipeline = this.pipeline;
        const layout = this.layout;
        if (!pipeline || !layout) throw new Error('@pmndrs/gi: the pose copy kernel is not compiled yet; await prepare() first');

        // Texture poses share one scratch buffer, one after another.
        let scratchTexels = 0;
        for (const copy of copies) if (copy.pose.kind === 'texture') scratchTexels += texelsCopied(copy.pose.texture, copy.vertexCount);
        const scratch = scratchTexels > 0 ? this.scratchFor(scratchTexels * TEXEL_BYTES) : null;
        const params = this.paramsFor(copies.length);

        const words = new Uint32Array((copies.length * PARAM_SLOT_BYTES) / 4);
        const sources: { buffer: GPUBuffer; offsetFloats: number; strideFloats: number }[] = [];
        let scratchTexel = 0;
        copies.forEach((copy, index) => {
            const pose = copy.pose;
            let source: { buffer: GPUBuffer; offsetFloats: number; strideFloats: number };
            if (pose.kind === 'texture') {
                const texture = pose.texture;
                const rows = texelsCopied(texture, copy.vertexCount) / texture.width;
                encoder.copyTextureToBuffer(
                    { texture },
                    { buffer: scratch!, offset: scratchTexel * TEXEL_BYTES, bytesPerRow: texture.width * TEXEL_BYTES, rowsPerImage: rows },
                    [texture.width, rows],
                );
                source = { buffer: scratch!, offsetFloats: scratchTexel * TEXEL_FLOATS, strideFloats: TEXEL_FLOATS };
                scratchTexel += rows * texture.width;
            } else {
                source = pose;
            }
            sources.push(source);
            words.set([copy.vertexCount, source.offsetFloats, source.strideFloats, copy.destinationWord], (index * PARAM_SLOT_BYTES) / 4);
        });
        this.device.queue.writeBuffer(params, 0, words);

        const pass = encoder.beginComputePass({ label: this.label });
        pass.setPipeline(pipeline);
        copies.forEach((copy, index) => {
            // A bind group per pose: the source is the host's buffer (or the scratch), a few a frame at most.
            const group = this.device.createBindGroup({
                label: this.label,
                layout,
                entries: [
                    { binding: 0, resource: { buffer: sources[index].buffer } },
                    { binding: 1, resource: { buffer: vertexBuffer } },
                    { binding: 2, resource: { buffer: params, offset: 0, size: PARAM_WORDS * 4 } },
                ],
            });
            pass.setBindGroup(0, group, [index * PARAM_SLOT_BYTES]);
            const workgroups = Math.ceil(copy.vertexCount / WORKGROUP_SIZE);
            const x = Math.min(workgroups, MAX_WORKGROUPS_PER_DIMENSION);
            pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
        });
        pass.end();
    }

    dispose(): void {
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
        this.params?.destroy();
        this.textureScratch?.destroy();
        this.params = null;
        this.textureScratch = null;
    }

    private paramsFor(count: number): GPUBuffer {
        const bytes = count * PARAM_SLOT_BYTES;
        if (!this.params || this.params.size < bytes) {
            if (this.params) this.retired.push(this.params);
            this.params = this.device.createBuffer({ label: `${this.label} params`, size: Math.max(bytes, 4 * PARAM_SLOT_BYTES), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        }
        return this.params;
    }

    private scratchFor(bytes: number): GPUBuffer {
        if (!this.textureScratch || this.textureScratch.size < bytes) {
            if (this.textureScratch) this.retired.push(this.textureScratch);
            this.textureScratch = this.device.createBuffer({
                label: `${this.label} texture scratch`,
                size: Math.max(bytes, (this.textureScratch?.size ?? 0) * 2),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }
        return this.textureScratch;
    }
}

/** Texels a texture pose copies: the rows that hold its vertices, whole (so up to one row more than the vertices). */
function texelsCopied(texture: GPUTexture, vertexCount: number): number {
    return Math.ceil(vertexCount / texture.width) * texture.width;
}
