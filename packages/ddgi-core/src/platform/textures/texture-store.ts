/**
 * The platform's texture store (docs/design/m4-materials.md §2): one
 * `texture_2d_array` of 256 × 256 layers, each with a full mip chain, that
 * hit shading samples where rays hit (`giSurfaceAtCone`, shade/wgsl.ts).
 *
 * - **Format:** `rgba8unorm-srgb`. The texels are stored sRGB-encoded (8 bits
 *   keep dark tones), the GPU decodes them before filtering, so a sample is
 *   linear light; alpha is linear.
 * - **Layers:** a texture's layer is its slot from the scene's
 *   `SlotAllocator` (the scene hands it in), so destroyed textures' layers
 *   are reused first. Non-square and small textures are stretched to 256²:
 *   GI rays read coarse mips (a hit 3 m away on a Sponza wall reads mip ~6),
 *   so nobody sees it.
 * - **Growth:** when a layer past the end is needed, the array is replaced
 *   by one with twice the layers (at most the device's
 *   `maxTextureArrayLayers`), the old layers are copied over in the host's
 *   encoder, and `version` increments so kernels rebuild their bind group.
 *   Nothing recompiles.
 * - **Building:** `gi/texture-store` (texture-store-wgsl.ts) makes a layer
 *   and its mips from the source in linear light, in the next `encode`
 *   after `createTexture`, every pending texture at once (as the scene
 *   copy's mesh packing does, T2.13).
 *
 * The array is created on first use, so a scene without textures holds
 * two layers (0.7 MB). Two, not one: a one-layer texture defaults to a 2D
 * binding on compatibility-mode devices, and the store binds as an array.
 */

import type { TextureSource } from '../contract/scene.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import { stageSource, type StagedSource } from './texture-sources.ts';
import {
    BUILD_LAYERS_PER_PASS,
    BUILD_STEP_BYTES,
    BUILD_STEP_UNIFORM_BYTES,
    BUILD_WORKGROUP_SIZE,
    buildStepOffset,
    buildStepWords,
    FILL_ENCODED_ENTRY_POINT,
    FILL_LINEAR_ENTRY_POINT,
    LAYER_BUILD_BYTES,
    levelByteOffset,
    levelRowBytes,
    levelSize,
    LINEAR_SCRATCH_TEXELS,
    MIP_ENTRY_POINT,
    STORE_LAYER_SIZE,
    STORE_MIP_LEVELS,
    TEXTURE_STORE_WGSL,
} from './texture-store-wgsl.ts';

/** `GPUTextureUsage` and `GPUBufferUsage` bits; the globals don't exist in Node, where tests import this file. */
const TEXTURE_USAGE = { copySrc: 0x01, copyDst: 0x02, textureBinding: 0x04 } as const;
const BUFFER_USAGE = { copySrc: 0x04, copyDst: 0x08, uniform: 0x40, storage: 0x80 } as const;
/** `GPUShaderStage.COMPUTE`. */
const SHADER_STAGE_COMPUTE = 0x4;

/** The pass name in `stats.passes`. */
export const TEXTURE_STORE_PASS = 'gi/texture-store';

/** Layers the array starts with (see the file comment for why two). */
const FIRST_LAYER_CAPACITY = 2;

/** Bytes of one layer with its mips: 4 bytes a texel, 256² × 4/3. */
export const LAYER_BYTES = (() => {
    let texels = 0;
    for (let level = 0; level < STORE_MIP_LEVELS; level++) texels += levelSize(level) * levelSize(level);
    return texels * 4;
})();

/** One texture to make into a layer. */
export interface TextureLayerRequest {
    readonly layer: number;
    readonly source: TextureSource;
    /** The descriptor's `srgb`: the source's values are sRGB-encoded (unless its format decodes sRGB itself). */
    readonly srgb: boolean;
    readonly label?: string;
}

/** What one `encode` did with its requests: layers made, and layers that can't be (the store warned why). */
export interface TextureLayerResults {
    readonly built: readonly number[];
    readonly failed: readonly number[];
}

/**
 * What the scene needs from a store: make layers in `encode`. The platform's
 * store is {@link PlatformTextureStore}; unit tests in Node hand the scene a
 * stand-in (no GPU textures there).
 */
export interface TextureLayerStore {
    /** Its kernel is compiled: `encode` makes layers. Until then requests wait (the scene keeps them). */
    readonly ready: boolean;
    encode(encoder: GPUCommandEncoder, requests: readonly TextureLayerRequest[], timer?: GpuTimerScope): TextureLayerResults;
    /** GPU bytes: the array, and while textures are being made, the build buffers and staging textures. */
    readonly memoryBytes: number;
    dispose(): void;
}

/** What hit shading binds: the array as a sampled `2d-array` view, and when it was last replaced. */
export interface TextureStoreBinding {
    readonly view: GPUTextureView;
    readonly version: number;
}

export interface PlatformTextureStoreOptions {
    readonly label?: string;
    readonly warn?: (message: string) => void;
}

interface BuildPipelines {
    readonly fillEncoded: GPUComputePipeline;
    readonly fillLinear: GPUComputePipeline;
    readonly mip: GPUComputePipeline;
    readonly sharedLayout: GPUBindGroupLayout;
    readonly sourceLayout: GPUBindGroupLayout;
}

/** The buffers a pass builds into; made when textures arrive, destroyed after a frame with none. */
interface BuildBuffers {
    readonly linear: GPUBuffer;
    readonly bytes: GPUBuffer;
    readonly steps: GPUBuffer;
    readonly group: GPUBindGroup;
}

export class PlatformTextureStore implements TextureLayerStore, TextureStoreBinding {
    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly warn: (message: string) => void;
    private array: GPUTexture | null = null;
    private arrayView: GPUTextureView | null = null;
    private capacity = 0;
    private arrayVersion = 0;
    private pipelines: BuildPipelines | null = null;
    private compiling: Promise<void> | null = null;
    private buffers: BuildBuffers | null = null;
    /** Staging textures and replaced arrays, destroyed in the next `encode` (after the host submitted the frame using them). */
    private retired: GPUTexture[] = [];
    private retiredBuffers: GPUBuffer[] = [];
    private disposed = false;

    constructor(device: GPUDevice, options: PlatformTextureStoreOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'gi';
        this.warn = options.warn ?? ((message) => console.warn(message));
    }

    /** Compiles `gi/texture-store` (once). Called by the platform's `prepare`. */
    prepare(): Promise<void> {
        this.compiling ??= (async () => {
            const device = this.device;
            const label = `${this.label} ${TEXTURE_STORE_PASS}`;
            const sharedLayout = device.createBindGroupLayout({
                label: `${label} build`,
                entries: [
                    { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
                    { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
                    { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
                ],
            });
            // unfilterable-float accepts every float format, filterable or not (rgba32float, BC7, ...): the kernel only loads.
            const sourceLayout = device.createBindGroupLayout({
                label: `${label} source`,
                entries: [{ binding: 0, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } }],
            });
            const module = device.createShaderModule({ label, code: TEXTURE_STORE_WGSL });
            const fillLayout = device.createPipelineLayout({ label: `${label} fill`, bindGroupLayouts: [sharedLayout, sourceLayout] });
            const mipLayout = device.createPipelineLayout({ label: `${label} mip`, bindGroupLayouts: [sharedLayout] });
            const [fillEncoded, fillLinear, mip] = await Promise.all([
                device.createComputePipelineAsync({ label: `${label} fill (sRGB source)`, layout: fillLayout, compute: { module, entryPoint: FILL_ENCODED_ENTRY_POINT } }),
                device.createComputePipelineAsync({ label: `${label} fill (linear source)`, layout: fillLayout, compute: { module, entryPoint: FILL_LINEAR_ENTRY_POINT } }),
                device.createComputePipelineAsync({ label: `${label} mip`, layout: mipLayout, compute: { module, entryPoint: MIP_ENTRY_POINT } }),
            ]);
            this.pipelines = { fillEncoded, fillLinear, mip, sharedLayout, sourceLayout };
        })();
        return this.compiling;
    }

    get ready(): boolean {
        return this.pipelines !== null;
    }

    /** The array as hit shading samples it: every layer and level, linear light through the sRGB format. */
    get view(): GPUTextureView {
        this.ensureArray(FIRST_LAYER_CAPACITY);
        return this.arrayView!;
    }

    /** Increments when the array is replaced (it grew): bind groups holding {@link view} are stale. */
    get version(): number {
        return this.arrayVersion;
    }

    /** Layers the array holds now. */
    get layerCapacity(): number {
        return this.capacity;
    }

    /** The array itself, for tests that read layers back (`COPY_SRC`). Null until first used. */
    get texture(): GPUTexture | null {
        return this.array;
    }

    get memoryBytes(): number {
        let bytes = this.capacity * LAYER_BYTES;
        if (this.buffers) bytes += this.buffers.linear.size + this.buffers.bytes.size + this.buffers.steps.size;
        for (const texture of this.retired) bytes += stagingBytes(texture);
        return bytes;
    }

    /**
     * Makes each request's layer, into the host's encoder: the sources go
     * onto the GPU (texture-sources.ts), the array grows if a layer is past
     * its end, and `gi/texture-store` makes each layer and its mips. Requests
     * whose source can't be used are warned about and come back in `failed`.
     * Also destroys what the previous call retired (the host has submitted
     * by now).
     */
    encode(encoder: GPUCommandEncoder, requests: readonly TextureLayerRequest[], timer?: GpuTimerScope): TextureLayerResults {
        if (this.disposed) throw new Error('@pmndrs/ddgi: the texture store was disposed');
        for (const texture of this.retired) texture.destroy();
        this.retired = [];
        for (const buffer of this.retiredBuffers) buffer.destroy();
        this.retiredBuffers = [];
        if (requests.length === 0) {
            // A frame with nothing to make: the build buffers go (a load is over).
            this.releaseBuildBuffers();
            return { built: [], failed: [] };
        }
        const pipelines = this.pipelines;
        if (!pipelines) throw new Error('@pmndrs/ddgi: the texture store kernel is not compiled yet; await prepare() first');

        const layerLimit = this.device.limits.maxTextureArrayLayers;
        const failed: number[] = [];
        const staged: { readonly layer: number; readonly source: StagedSource }[] = [];
        for (const request of requests) {
            if (request.layer >= layerLimit) {
                this.warn(`@pmndrs/ddgi: the scene has more textures than the device's maxTextureArrayLayers (${layerLimit}); the materials using the rest keep their flat colour. requiredLimits(adapter) asks for the adapter's maximum.`);
                failed.push(request.layer);
                continue;
            }
            const label = `${this.label} texture${request.label ? ` "${request.label}"` : ''}`;
            const source = stageSource(this.device, request.source, request.srgb, STORE_LAYER_SIZE, label);
            if ('reason' in source) {
                this.warn(source.reason);
                failed.push(request.layer);
                continue;
            }
            if (source.staging) this.retired.push(source.staging);
            staged.push({ layer: request.layer, source });
        }
        if (staged.length === 0) return { built: [], failed };

        this.ensureArray(Math.max(...staged.map((entry) => entry.layer)) + 1, encoder);
        const buffers = this.buildBuffers(pipelines);
        for (let first = 0; first < staged.length; first += BUILD_LAYERS_PER_PASS) {
            const batch = staged.slice(first, first + BUILD_LAYERS_PER_PASS);
            // Only the frame's first pass is timed: a load can make hundreds of passes, and the
            // timer's passes per frame are better spent on the techniques.
            this.encodeBatch(encoder, pipelines, buffers, batch, first === 0 ? timer : undefined);
        }
        return { built: staged.map((entry) => entry.layer), failed };
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.releaseBuildBuffers();
        for (const texture of this.retired) texture.destroy();
        this.retired = [];
        for (const buffer of this.retiredBuffers) buffer.destroy();
        this.retiredBuffers = [];
        this.array?.destroy();
        this.array = null;
        this.arrayView = null;
    }

    /**
     * One pass making up to `BUILD_LAYERS_PER_PASS` layers into the byte
     * buffer, then the copies of each layer's levels into the array. The
     * next batch's pass comes after these copies in the encoder, so it may
     * overwrite the byte buffer.
     */
    private encodeBatch(
        encoder: GPUCommandEncoder,
        pipelines: BuildPipelines,
        buffers: BuildBuffers,
        batch: readonly { readonly layer: number; readonly source: StagedSource }[],
        timer: GpuTimerScope | undefined,
    ): void {
        const pass = encoder.beginComputePass({ label: `${this.label} ${TEXTURE_STORE_PASS}`, timestampWrites: timer?.computePass(TEXTURE_STORE_PASS) });
        batch.forEach(({ source }, place) => {
            const sourceGroup = this.device.createBindGroup({ label: `${this.label} ${TEXTURE_STORE_PASS} source`, layout: pipelines.sourceLayout, entries: [{ binding: 0, resource: source.view }] });
            // Level 0 from the source; the layers of a pass share the linear scratch, one after another.
            pass.setPipeline(source.encoded ? pipelines.fillEncoded : pipelines.fillLinear);
            pass.setBindGroup(0, buffers.group, [buildStepOffset(place, 0)]);
            pass.setBindGroup(1, sourceGroup);
            const tiles = Math.ceil(STORE_LAYER_SIZE / BUILD_WORKGROUP_SIZE);
            pass.dispatchWorkgroups(tiles, tiles);
            pass.setPipeline(pipelines.mip);
            for (let level = 1; level < STORE_MIP_LEVELS; level++) {
                pass.setBindGroup(0, buffers.group, [buildStepOffset(place, level)]);
                const levelTiles = Math.ceil(levelSize(level) / BUILD_WORKGROUP_SIZE);
                pass.dispatchWorkgroups(levelTiles, levelTiles);
            }
        });
        pass.end();
        const array = this.array!;
        batch.forEach(({ layer }, place) => {
            for (let level = 0; level < STORE_MIP_LEVELS; level++) {
                const size = levelSize(level);
                encoder.copyBufferToTexture(
                    { buffer: buffers.bytes, offset: place * LAYER_BUILD_BYTES + levelByteOffset(level), bytesPerRow: levelRowBytes(level), rowsPerImage: size },
                    { texture: array, mipLevel: level, origin: { x: 0, y: 0, z: layer } },
                    { width: size, height: size, depthOrArrayLayers: 1 },
                );
            }
        });
    }

    /**
     * Makes sure the array has at least `layers` layers. A bigger array gets
     * the old one's layers by copies recorded in `encoder` (none is needed
     * before the first build, when there is nothing to copy).
     */
    private ensureArray(layers: number, encoder?: GPUCommandEncoder): void {
        if (this.array && layers <= this.capacity) return;
        const limit = this.device.limits.maxTextureArrayLayers;
        // Doubling keeps the copies few while a scene loads.
        const capacity = Math.min(limit, Math.max(layers, FIRST_LAYER_CAPACITY, this.capacity * 2));
        const array = this.device.createTexture({
            label: `${this.label} texture store`,
            size: { width: STORE_LAYER_SIZE, height: STORE_LAYER_SIZE, depthOrArrayLayers: capacity },
            mipLevelCount: STORE_MIP_LEVELS,
            format: 'rgba8unorm-srgb',
            // COPY_DST for the layers made; COPY_SRC to copy them into a bigger array (and for tests to read them).
            usage: TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copyDst | TEXTURE_USAGE.copySrc,
        });
        const previous = this.array;
        if (previous) {
            if (!encoder) throw new Error('growing the texture store needs an encoder for the copies');
            for (let level = 0; level < STORE_MIP_LEVELS; level++) {
                const size = levelSize(level);
                encoder.copyTextureToTexture({ texture: previous, mipLevel: level }, { texture: array, mipLevel: level }, { width: size, height: size, depthOrArrayLayers: this.capacity });
            }
            this.retired.push(previous);
        }
        this.array = array;
        this.arrayView = array.createView({ label: `${this.label} texture store`, dimension: '2d-array' });
        this.capacity = capacity;
        if (previous) this.arrayVersion++;
    }

    private buildBuffers(pipelines: BuildPipelines): BuildBuffers {
        if (this.buffers) return this.buffers;
        const device = this.device;
        const label = `${this.label} ${TEXTURE_STORE_PASS}`;
        const linear = device.createBuffer({ label: `${label} linear scratch`, size: LINEAR_SCRATCH_TEXELS * 16, usage: BUFFER_USAGE.storage });
        const bytes = device.createBuffer({ label: `${label} texels`, size: BUILD_LAYERS_PER_PASS * LAYER_BUILD_BYTES, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc });
        const steps = device.createBuffer({ label: `${label} steps`, size: BUILD_STEP_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        device.queue.writeBuffer(steps, 0, buildStepWords());
        const group = device.createBindGroup({
            label: `${label} build`,
            layout: pipelines.sharedLayout,
            entries: [
                { binding: 0, resource: { buffer: linear } },
                { binding: 1, resource: { buffer: bytes } },
                { binding: 2, resource: { buffer: steps, size: BUILD_STEP_BYTES } },
            ],
        });
        this.buffers = { linear, bytes, steps, group };
        return this.buffers;
    }

    private releaseBuildBuffers(): void {
        if (!this.buffers) return;
        // Retired, not destroyed: the frame that used them may not have been submitted yet.
        this.retiredBuffers.push(this.buffers.linear, this.buffers.bytes, this.buffers.steps);
        this.buffers = null;
    }
}

/**
 * Rough bytes of a retired texture, for the memory report: 4 a texel and a
 * third more for mips when it has them (a replaced array). Compressed
 * staging textures are smaller; this is an upper bound.
 */
function stagingBytes(texture: GPUTexture): number {
    const texels = texture.width * texture.height * texture.depthOrArrayLayers;
    return texels * 4 * (texture.mipLevelCount > 1 ? 4 / 3 : 1);
}
