/**
 * three's cube maps, mirrored in x for the platform.
 *
 * three r187 on WebGPU looks every cube texture up at (−x, y, z)
 * (`CubeTextureNode.setupUV`), while the platform's `cube` layout uses
 * WebGPU's own addressing with the direction unchanged (scene contract,
 * "cube layout"). So what three shows along a world direction d is the
 * texture at M d, with M = diag(−1, 1, 1), and the platform needs a cube T'
 * with T'(d) = T(M d).
 *
 * Working through WebGPU's face table (major axis, then s and t per face):
 * - the +X face of T' is the −X face of T and vice versa (M swaps them);
 * - on every face, s runs the other way: +X uses s = −z/|x| and −X uses
 *   s = +z/|x|, while on ±Y and ±Z, s follows x, which M negates; t never
 *   changes, because M leaves y and z alone.
 *
 * So T' = T with the ±X faces swapped and every face flipped left to right.
 * That's what {@link mirroredCubeFaces} does, on CPU pixel data. The unit
 * test checks it against direct cube lookups in both conventions.
 */

/** One image's pixels: rows in memory order, `channels` values per texel. */
export interface CubeFacePixels<Data extends ArrayLike<number> = ArrayLike<number>> {
    readonly data: Data;
    readonly width: number;
    readonly height: number;
}

export type PixelArray = Float32Array | Uint16Array | Uint8Array | Uint8ClampedArray;

/** WebGPU cube layers, in order: +X, −X, +Y, −Y, +Z, −Z. */
export const CUBE_FACE_COUNT = 6;

/** For each face of the mirrored cube, the face of three's cube it comes from: the ±X faces swap. */
export const MIRRORED_SOURCE_FACE: readonly number[] = [1, 0, 2, 3, 4, 5];

/** A copy of an image, flipped left to right and/or top to bottom. */
export function flipPixels<Data extends PixelArray>(image: CubeFacePixels<Data>, channels: number, flipX: boolean, flipY: boolean): CubeFacePixels<Data> {
    const { width, height, data } = image;
    const flipped = new (data.constructor as new (length: number) => Data)(width * height * channels);
    for (let row = 0; row < height; row++) {
        const sourceRow = flipY ? height - 1 - row : row;
        for (let column = 0; column < width; column++) {
            const sourceColumn = flipX ? width - 1 - column : column;
            const target = (row * width + column) * channels;
            const source = (sourceRow * width + sourceColumn) * channels;
            for (let channel = 0; channel < channels; channel++) flipped[target + channel] = data[source + channel];
        }
    }
    return { data: flipped, width, height };
}

/**
 * With `texture.flipY`, three uploads image i into layer FLIP_Y_LAYER[i],
 * flipped top to bottom: turning every face upside down also swaps what is
 * up and down (r187 `WebGPUTextureUtils`, `_flipMap`).
 */
export const FLIP_Y_LAYER: readonly number[] = [0, 1, 3, 2, 4, 5];

/**
 * The six faces of the mirrored cube (layer order), made from the six images
 * given to three. `flipY` is three's `texture.flipY`: the copy first lays the
 * faces out as three uploads them (FLIP_Y_LAYER, each flipped top to
 * bottom), then mirrors that cube.
 */
export function mirroredCubeFaces<Data extends PixelArray>(faces: readonly CubeFacePixels<Data>[], channels: number, flipY = false): CubeFacePixels<Data>[] {
    if (faces.length !== CUBE_FACE_COUNT) throw new RangeError(`a cube map has ${CUBE_FACE_COUNT} faces, got ${faces.length}`);
    // FLIP_Y_LAYER swaps two entries, so it is its own inverse: layer L holds image FLIP_Y_LAYER[L].
    const threeLayer = (layer: number): number => (flipY ? FLIP_Y_LAYER[layer] : layer);
    return MIRRORED_SOURCE_FACE.map((source) => flipPixels(faces[threeLayer(source)], channels, true, flipY));
}
