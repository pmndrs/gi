/**
 * The "GI data" of the spike: three textures a compute shader writes, shaped
 * like what a real technique publishes to a fragment shader.
 *
 * - `atlas`:   `texture_2d_array<f32>`, rgba16float, filterable (think: probe irradiance atlas).
 * - `volume`:  `texture_3d<f32>`, rgba16float, filterable (think: a voxel or probe-grid volume).
 * - `table`:   `texture_2d<f32>`, rgba32float, read with `textureLoad` (think: the volume table).
 *
 * Every texel of one texture holds the same constant, so the irradiance the
 * light node should add is a number we can compute by hand. Layer 0 of the
 * atlas holds a decoy value, so sampling the wrong layer would show.
 */

import { ExternalTexture, FloatType, LinearFilter } from 'three/webgpu';

export const ATLAS_SIZE = 4;
export const ATLAS_LAYERS = 2;
/** The layer the shader must read; layer 0 is the decoy. */
export const ATLAS_LAYER_USED = 1;
export const VOLUME_SIZE = 4;
export const TABLE_WIDTH = 2;

/** Constants the compute shader writes. All values are exact in rgba16float. */
export interface GiTextureValues {
    atlasDecoy: readonly [number, number, number, number];
    atlasUsed: readonly [number, number, number, number];
    volume: readonly [number, number, number, number];
    /** Texel 0 of the table: x weights the atlas, y weights the volume. */
    tableWeights: readonly [number, number, number, number];
}

export const FIRST_VALUES: GiTextureValues = {
    atlasDecoy: [9, 9, 9, 1],
    atlasUsed: [0.5, 0.25, 0.125, 1],
    volume: [0.25, 0.5, 1, 1],
    tableWeights: [2, 0.5, 0, 0],
};

/** A second set, to prove a later compute write reaches the next frame's shading. */
export const SECOND_VALUES: GiTextureValues = {
    ...FIRST_VALUES,
    tableWeights: [4, 0.5, 0, 0],
};

/**
 * The irradiance E the shading function returns, written the same way the
 * WGSL does: atlas value times the table's x weight plus volume value times y.
 */
export function expectedIrradiance(values: GiTextureValues): [number, number, number] {
    const [weightAtlas, weightVolume] = values.tableWeights;
    return [0, 1, 2].map(
        (channel) => values.atlasUsed[channel] * weightAtlas + values.volume[channel] * weightVolume,
    ) as [number, number, number];
}

const WRITE_SHADER = /* wgsl */ `
struct Values {
    atlasDecoy: vec4f,
    atlasUsed: vec4f,
    volume: vec4f,
    tableWeights: vec4f,
}

@group(0) @binding(0) var<uniform> values: Values;
@group(0) @binding(1) var atlas: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var volume: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var table: texture_storage_2d<rgba32float, write>;

// One 4x4x4 workgroup covers the whole volume; the atlas and table are
// smaller, so the extra invocations skip them.
@compute @workgroup_size(${VOLUME_SIZE}, ${VOLUME_SIZE}, ${VOLUME_SIZE})
fn writeAll(@builtin(global_invocation_id) id: vec3u) {
    textureStore(volume, id, values.volume);

    if (id.z < ${ATLAS_LAYERS}u) {
        let value = select(values.atlasDecoy, values.atlasUsed, id.z == ${ATLAS_LAYER_USED}u);
        textureStore(atlas, id.xy, id.z, value);
    }

    if (id.y == 0u && id.z == 0u && id.x < ${TABLE_WIDTH}u) {
        // Texel 1 is a decoy: the shader reads texel 0.
        textureStore(table, vec2u(id.x, 0u), select(vec4f(7.0), values.tableWeights, id.x == 0u));
    }
}`;

/** The textures plus the compute pipeline that fills them. */
export class GiTextures {
    readonly atlas: GPUTexture;
    readonly volume: GPUTexture;
    readonly table: GPUTexture;

    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly valuesBuffer: GPUBuffer;
    private readonly bindGroup: GPUBindGroup;

    constructor(device: GPUDevice) {
        this.device = device;
        const sampledAndStored = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        this.atlas = device.createTexture({
            label: 'spike atlas',
            size: { width: ATLAS_SIZE, height: ATLAS_SIZE, depthOrArrayLayers: ATLAS_LAYERS },
            format: 'rgba16float',
            usage: sampledAndStored,
        });
        this.volume = device.createTexture({
            label: 'spike volume',
            dimension: '3d',
            size: { width: VOLUME_SIZE, height: VOLUME_SIZE, depthOrArrayLayers: VOLUME_SIZE },
            format: 'rgba16float',
            usage: sampledAndStored,
        });
        this.table = device.createTexture({
            label: 'spike table',
            size: { width: TABLE_WIDTH, height: 1 },
            format: 'rgba32float',
            usage: sampledAndStored,
        });

        this.valuesBuffer = device.createBuffer({
            label: 'spike values',
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.pipeline = device.createComputePipeline({
            label: 'spike write textures',
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ label: 'spike write textures', code: WRITE_SHADER }),
                entryPoint: 'writeAll',
            },
        });
        this.bindGroup = device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.valuesBuffer } },
                { binding: 1, resource: this.atlas.createView({ dimension: '2d-array' }) },
                { binding: 2, resource: this.volume.createView({ dimension: '3d' }) },
                { binding: 3, resource: this.table.createView() },
            ],
        });
    }

    /** Fills all three textures by compute. The write is queued; no readback is needed. */
    write(values: GiTextureValues): void {
        this.device.queue.writeBuffer(
            this.valuesBuffer,
            0,
            new Float32Array([...values.atlasDecoy, ...values.atlasUsed, ...values.volume, ...values.tableWeights]),
        );
        const encoder = this.device.createCommandEncoder({ label: 'spike write textures' });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    destroy(): void {
        this.atlas.destroy();
        this.volume.destroy();
        this.table.destroy();
        this.valuesBuffer.destroy();
    }
}

/** The three textures wrapped for three. See {@link wrapForThree}. */
export interface WrappedGiTextures {
    atlas: ExternalTexture;
    volume: ExternalTexture;
    table: ExternalTexture;
}

/**
 * Wraps each GPUTexture in an ExternalTexture. three binds it without
 * allocating or destroying it. What three needs to be told, found in the r187
 * source and confirmed by this spike running:
 *
 * - `image` size, or three reads the size as 1x1.
 * - `isArrayTexture = true` for a 2D array (WGSL `texture_2d_array`, view dimension 2d-array).
 * - `is3DTexture = true` for a 3D texture. `texture3D(node)` sets the *binding*
 *   to 3D, but WGSLNodeBuilder decides the WGSL *type* from `texture.is3DTexture`
 *   and would otherwise declare `texture_2d<f32>`.
 * - `type = FloatType` for rgba32float, so the bind-group layout asks for an
 *   unfilterable (or, with float32-filterable, filterable) float sample type.
 *   rgba16float needs nothing: the default sample type is filterable float.
 */
/** The types declare `ExternalTexture.image` as null; three reads width, height and depth from it. */
function setImageSize(texture: ExternalTexture, width: number, height: number, depth: number): void {
    (texture as unknown as { image: unknown }).image = { width, height, depth };
}

export function wrapForThree(textures: GiTextures): WrappedGiTextures {
    const atlas = new ExternalTexture(textures.atlas as never);
    setImageSize(atlas, ATLAS_SIZE, ATLAS_SIZE, ATLAS_LAYERS);
    (atlas as unknown as { isArrayTexture: boolean }).isArrayTexture = true;

    const volume = new ExternalTexture(textures.volume as never);
    setImageSize(volume, VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE);
    (volume as unknown as { is3DTexture: boolean }).is3DTexture = true;

    const table = new ExternalTexture(textures.table as never);
    setImageSize(table, TABLE_WIDTH, 1, 1);
    table.type = FloatType;

    for (const wrapped of [atlas, volume]) {
        wrapped.magFilter = LinearFilter;
        wrapped.minFilter = LinearFilter;
        wrapped.generateMipmaps = false;
    }
    return { atlas, volume, table };
}
