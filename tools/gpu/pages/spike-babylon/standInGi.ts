/**
 * A stand-in for a GI technique, written against the platform's `ShadingModule`
 * contract. It owns three textures on the host's `GPUDevice` and publishes a
 * parameter-style `giIrradiance` that reads all of them:
 *
 *   - a 2D-array atlas (`rgba16float`), like DDGI's irradiance texture;
 *   - a 3D texture (`rgba16float`), like a voxel technique's radiance volume;
 *   - a small `rgba32float` table read with `textureLoad`, like the volume list;
 *   - two `vec4f` uniforms.
 *
 * The values are constants, so the irradiance E the shader returns is known
 * exactly and the readback test can compare against it. The function also uses
 * `normal` and `screenUV`, so a wrong handedness or a wrong screen UV shows up.
 *
 * Only the contract's types are imported (no runtime code from ddgi-core).
 */

import type { ShadingModule, ShadingParameter } from '../../../../packages/ddgi-core/src/platform/contract/shading-module.ts';

export type Rgb = readonly [number, number, number];

/** Which atlas layer holds the real value. The other layers hold decoys, to prove the layer is selected. */
export const ATLAS_LAYER_UNDER_TEST = 1;
export const ATLAS_LAYER_COUNT = 3;
const DECOY: Rgb = [0.9, 0.8, 0.7];

export const VOLUME_COLOR: Rgb = [0.02, 0.03, 0.04];
const VOLUME_SIZE = 4;

/** Table texel 0 is used on the left half of the screen, texel 1 on the right half. */
export type TableValues = readonly [readonly [number, number, number, number], readonly [number, number, number, number]];
export const DEFAULT_TABLE: TableValues = [
    [0.25, 0.5, 1.0, 1.0],
    [0.5, 0.25, 0.0, 1.0],
];

/** giParams0 = (table weight, direction xyz); giParams1 = (normal-facing colour, unused). */
export const TABLE_WEIGHT = 0.4;
/** Direction in the platform's right-handed space that a surface facing the camera has as its normal. */
export const FACING_DIRECTION: Rgb = [0, 0, 1];
export const FACING_COLOR: Rgb = [0.3, 0.2, 0.1];

export const GI_IRRADIANCE_WGSL = /* wgsl */ `
// Stand-in for a technique's shading module. Resources are parameters, in the
// order of ShadingModule.parameters, exactly as the contract describes.
fn giIrradiance(
    position: vec3f,
    normal: vec3f,
    viewDirection: vec3f,
    screenUV: vec2f,
    giAtlas: texture_2d_array<f32>,
    giAtlasSampler: sampler,
    giVolume: texture_3d<f32>,
    giVolumeSampler: sampler,
    giTable: texture_2d<f32>,
    giParams0: vec4f,
    giParams1: vec4f,
) -> vec3f {
    // Filtered reads: textureSampleLevel, because the call site is inside a
    // large engine shader and the level is known anyway.
    let atlas = textureSampleLevel(giAtlas, giAtlasSampler, fract(screenUV), ${ATLAS_LAYER_UNDER_TEST}, 0.0).rgb;
    let volumeCoordinate = clamp(position * 0.05 + vec3f(0.5), vec3f(0.0), vec3f(1.0));
    let volume = textureSampleLevel(giVolume, giVolumeSampler, volumeCoordinate, 0.0).rgb;

    // Unfilterable table: textureLoad only. The left half of the screen reads
    // texel 0, the right half texel 1.
    let tableIndex = clamp(i32(screenUV.x * 2.0), 0, 1);
    let table = textureLoad(giTable, vec2i(tableIndex, 0), 0).rgb;

    // Uses the normal: only surfaces facing giParams0.yzw receive this term.
    let facing = max(dot(normal, giParams0.yzw), 0.0);

    // viewDirection is part of the contract's signature; this stand-in has no use for it.
    return atlas + volume + table * giParams0.x + giParams1.rgb * facing;
}
`;

export interface StandInGi {
    readonly module: ShadingModule;
    /** The irradiance E the shader returns for the left or right half of the screen, for a surface facing FACING_DIRECTION. */
    expectedIrradiance(side: 'left' | 'right'): Rgb;
    /** Changes table contents in place. The GPUTexture stays the same. */
    writeTable(table: TableValues): void;
    /** Allocates a new atlas of a different size, as a technique does when it grows. Returns the old one, still alive. */
    reallocateAtlas(size: number, color: Rgb): GPUTexture;
    currentAtlas(): GPUTexture;
    destroy(): void;
}

export function createStandInGi(device: GPUDevice, initialAtlasSize: number, atlasColor: Rgb): StandInGi {
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
    let atlasColorNow = atlasColor;
    let tableNow = DEFAULT_TABLE;

    const volume = device.createTexture({
        label: 'stand-in volume',
        dimension: '3d',
        size: [VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE],
        format: 'rgba16float',
        usage,
    });
    const volumeTexels = new Uint16Array(VOLUME_SIZE ** 3 * 4);
    for (let texel = 0; texel < VOLUME_SIZE ** 3; texel++) {
        volumeTexels.set(rgbaToHalf([...VOLUME_COLOR, 1]), texel * 4);
    }
    device.queue.writeTexture(
        { texture: volume },
        volumeTexels,
        { bytesPerRow: VOLUME_SIZE * 8, rowsPerImage: VOLUME_SIZE },
        [VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE],
    );

    const table = device.createTexture({
        label: 'stand-in table',
        size: [2, 1],
        format: 'rgba32float',
        usage,
    });
    const writeTable = (values: TableValues): void => {
        tableNow = values;
        device.queue.writeTexture({ texture: table }, new Float32Array(values.flat()), { bytesPerRow: 32 }, [2, 1]);
    };
    writeTable(DEFAULT_TABLE);

    const createAtlas = (size: number, color: Rgb): GPUTexture => {
        const atlas = device.createTexture({
            label: `stand-in atlas ${size}x${size}`,
            size: [size, size, ATLAS_LAYER_COUNT],
            format: 'rgba16float',
            usage,
        });
        const texels = new Uint16Array(size * size * ATLAS_LAYER_COUNT * 4);
        for (let layer = 0; layer < ATLAS_LAYER_COUNT; layer++) {
            const layerColor = layer === ATLAS_LAYER_UNDER_TEST ? color : DECOY;
            for (let texel = 0; texel < size * size; texel++) {
                texels.set(rgbaToHalf([...layerColor, 1]), (layer * size * size + texel) * 4);
            }
        }
        device.queue.writeTexture(
            { texture: atlas },
            texels,
            { bytesPerRow: size * 8, rowsPerImage: size },
            [size, size, ATLAS_LAYER_COUNT],
        );
        return atlas;
    };
    let atlas = createAtlas(initialAtlasSize, atlasColor);

    const params0 = new Float32Array([TABLE_WEIGHT, ...FACING_DIRECTION]);
    const params1 = new Float32Array([...FACING_COLOR, 0]);

    const parameters: readonly ShadingParameter[] = [
        { kind: 'texture', name: 'giAtlas', dimension: '2d-array', sampleType: 'float', texture: () => atlas },
        { kind: 'sampler', name: 'giAtlasSampler', texture: 'giAtlas', filtering: 'linear', addressMode: 'clamp-to-edge' },
        { kind: 'texture', name: 'giVolume', dimension: '3d', sampleType: 'float', texture: () => volume },
        { kind: 'sampler', name: 'giVolumeSampler', texture: 'giVolume', filtering: 'linear', addressMode: 'clamp-to-edge' },
        { kind: 'texture', name: 'giTable', dimension: '2d', sampleType: 'unfilterable-float', texture: () => table },
        { kind: 'vec4f', name: 'giParams0', value: () => params0 },
        { kind: 'vec4f', name: 'giParams1', value: () => params1 },
    ];

    const module: ShadingModule = {
        techniqueId: 'stand-in',
        code: GI_IRRADIANCE_WGSL,
        parameters,
        provides: { radiance: false },
        globalsCode() {
            throw new Error('The globals path is not exercised by this spike.');
        },
        writeUniforms() {
            throw new Error('The globals path is not exercised by this spike.');
        },
        version: 1,
    };

    return {
        module,
        expectedIrradiance(side) {
            const texel = tableNow[side === 'left' ? 0 : 1];
            const channelSum = (channel: 0 | 1 | 2): number =>
                atlasColorNow[channel] + VOLUME_COLOR[channel] + texel[channel] * TABLE_WEIGHT + FACING_COLOR[channel];
            return [channelSum(0), channelSum(1), channelSum(2)];
        },
        writeTable,
        reallocateAtlas(size, color) {
            const previous = atlas;
            atlas = createAtlas(size, color);
            atlasColorNow = color;
            return previous;
        },
        currentAtlas: () => atlas,
        destroy() {
            atlas.destroy();
            volume.destroy();
            table.destroy();
        },
    };
}

/** Converts a JavaScript number to IEEE half-float bits (round to nearest even; values here are small and finite). */
function toHalf(value: number): number {
    const float32 = new Float32Array([value]);
    const bits = new Uint32Array(float32.buffer)[0]!;
    const sign = (bits >>> 16) & 0x8000;
    const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
    const mantissa = bits & 0x7fffff;
    if (exponent <= 0) return sign; // flushes tiny values to zero; none are used here
    if (exponent >= 31) return sign | 0x7c00;
    let half = sign | (exponent << 10) | (mantissa >>> 13);
    // Round to nearest: look at the dropped bits.
    const dropped = mantissa & 0x1fff;
    if (dropped > 0x1000 || (dropped === 0x1000 && (half & 1) === 1)) half++;
    return half;
}

function rgbaToHalf(values: readonly number[]): number[] {
    return values.map(toHalf);
}
