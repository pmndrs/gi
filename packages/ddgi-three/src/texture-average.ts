/**
 * Average texture colours for GI, until the core samples textures itself (its
 * texture atlas is M4).
 *
 * Until then a textured material would bounce light in its flat `color`
 * alone, usually white, so a red brick wall would bounce white light, and a
 * material whose metalness comes from a texture (glTF's metallicRoughness,
 * three's `metalnessMap`) would count as all metal or all paint. So the GI
 * albedo is multiplied by an average over the texture:
 *
 * - **albedo map and metalness map on the same uv mapping:** the average of
 *   albedoTexel × (1 − metalness × metalnessTexel), texel by texel, over the
 *   albedo map's grid (the metalness map sampled at the same uv). This is
 *   the exact average of what three's diffuse term multiplies.
 * - **on different uv mappings:** the two averages, multiplied, which is
 *   exact only when the two are unrelated over the surface.
 * - **one of them alone:** its average (for metalness: 1 − metalness ×
 *   average texel).
 *
 * Every average is a box average in linear light, weighted by the albedo
 * map's alpha, so cut-out texels (alpha-tested leaves) don't count as black.
 * Images larger than {@link MAX_READ_SIZE} on a side are first drawn at that
 * size by the browser (its high-quality downscale, which filters in sRGB);
 * every texel of that copy is then decoded to linear and counted. Data
 * textures are read whole. Each texture is read once per version, never per
 * frame (scene-sync.ts keeps the results).
 *
 * **When M4's atlas samples textures, remove the fold** (convert.ts,
 * `materialDescriptor`): the core would then apply the texture a second time.
 */

import { FloatType, HalfFloatType, SRGBColorSpace, type Texture } from 'three';
import type { ColorRGB } from '@pmndrs/ddgi-core';

/** Images larger than this many pixels on a side are drawn at this size before averaging. */
export const MAX_READ_SIZE = 256;

/** sRGB-encoded value in 0..1 to linear. */
export function srgbToLinear(value: number): number {
    return value <= 0.04045 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
}

/** 8-bit sRGB to linear, one entry per byte value. */
const SRGB_BYTE_TO_LINEAR = Float32Array.from({ length: 256 }, (_, byte) => srgbToLinear(byte / 255));

/** A half float's bits to a number. */
function halfToFloat(bits: number): number {
    const exponent = (bits >> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    const sign = bits & 0x8000 ? -1 : 1;
    if (exponent === 0) return sign * Math.pow(2, -14) * (mantissa / 1024);
    if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
    return sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

/** A texture's texels, decoded: linear rgb and alpha 0..1, four floats per texel, rows as stored. */
export interface LinearPixels {
    readonly rgba: Float32Array;
    readonly width: number;
    readonly height: number;
    /** three's `texture.flipY`. Memory row r holds uv v = r / height, or 1 − r / height with flipY (three flips such images on upload). */
    readonly flipY: boolean;
}

interface PixelSource {
    readonly data: ArrayLike<number>;
    readonly width: number;
    readonly height: number;
}

function decodeBytes(source: PixelSource, srgb: boolean, flipY: boolean): LinearPixels {
    const count = source.width * source.height;
    const rgba = new Float32Array(count * 4);
    const data = source.data;
    for (let texel = 0; texel < count; texel++) {
        const base = texel * 4;
        for (let channel = 0; channel < 3; channel++) rgba[base + channel] = srgb ? SRGB_BYTE_TO_LINEAR[data[base + channel]] : data[base + channel] / 255;
        rgba[base + 3] = data[base + 3] / 255;
    }
    return { rgba, width: source.width, height: source.height, flipY };
}

function decodeFloats(source: PixelSource, decode: (value: number) => number, flipY: boolean): LinearPixels {
    const count = source.width * source.height;
    const rgba = new Float32Array(count * 4);
    for (let index = 0; index < count * 4; index++) rgba[index] = decode(source.data[index]);
    return { rgba, width: source.width, height: source.height, flipY };
}

type DrawableImage = ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas;

function isDrawable(image: unknown): image is DrawableImage {
    return (
        (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) ||
        (typeof HTMLImageElement !== 'undefined' && image instanceof HTMLImageElement && image.complete && image.naturalWidth > 0) ||
        (typeof HTMLCanvasElement !== 'undefined' && image instanceof HTMLCanvasElement) ||
        (typeof OffscreenCanvas !== 'undefined' && image instanceof OffscreenCanvas)
    );
}

/**
 * The texture's texels in linear light, or undefined when they can't be read
 * on the CPU (still loading, compressed, a render target, a format this
 * doesn't decode).
 */
export function readLinearPixels(texture: Texture): LinearPixels | undefined {
    const flags = texture as Texture & { isCompressedTexture?: boolean; isRenderTargetTexture?: boolean };
    if (flags.isCompressedTexture || flags.isRenderTargetTexture) return undefined;
    const image = texture.image as unknown;
    if (!image) return undefined;
    const srgb = texture.colorSpace === SRGBColorSpace;
    if (typeof ImageData !== 'undefined' && image instanceof ImageData) return decodeBytes(image, srgb, texture.flipY);
    if (isDrawable(image)) {
        const width = (image as HTMLImageElement).naturalWidth || image.width;
        const height = (image as HTMLImageElement).naturalHeight || image.height;
        const scale = Math.min(1, MAX_READ_SIZE / Math.max(width, height));
        const targetWidth = Math.max(1, Math.round(width * scale));
        const targetHeight = Math.max(1, Math.round(height * scale));
        const canvas = new OffscreenCanvas(targetWidth, targetHeight);
        const context = canvas.getContext('2d');
        if (!context) return undefined;
        context.imageSmoothingQuality = 'high';
        context.drawImage(image as CanvasImageSource, 0, 0, targetWidth, targetHeight);
        return decodeBytes(context.getImageData(0, 0, targetWidth, targetHeight), srgb, texture.flipY);
    }
    const data = image as PixelSource;
    if (!ArrayBuffer.isView(data.data) || !(data.width > 0 && data.height > 0) || (data.data as ArrayLike<number>).length < data.width * data.height * 4) return undefined;
    if (texture.type === FloatType && data.data instanceof Float32Array) return decodeFloats(data, (value) => value, texture.flipY);
    if (texture.type === HalfFloatType && data.data instanceof Uint16Array) return decodeFloats(data, halfToFloat, texture.flipY);
    if (data.data instanceof Uint8Array || data.data instanceof Uint8ClampedArray) return decodeBytes(data, srgb, texture.flipY);
    return undefined;
}

/** Box average of linear rgb, weighted by alpha; undefined if every texel is fully transparent. */
export function averageLinear(pixels: LinearPixels): ColorRGB | undefined {
    const { rgba } = pixels;
    let red = 0;
    let green = 0;
    let blue = 0;
    let weight = 0;
    for (let base = 0; base < rgba.length; base += 4) {
        const alpha = rgba[base + 3];
        red += rgba[base] * alpha;
        green += rgba[base + 1] * alpha;
        blue += rgba[base + 2] * alpha;
        weight += alpha;
    }
    if (!(weight > 0)) return undefined;
    return [red / weight, green / weight, blue / weight];
}

/** The texture's average linear colour (alpha-weighted box average), or undefined when its pixels can't be read. */
export function textureAverage(texture: Texture): ColorRGB | undefined {
    const pixels = readLinearPixels(texture);
    return pixels ? averageLinear(pixels) : undefined;
}

/** The blue channel's plain average: three reads metalness from `metalnessMap.b`. */
function averageBlue(pixels: LinearPixels): number {
    let sum = 0;
    for (let base = 2; base < pixels.rgba.length; base += 4) sum += pixels.rgba[base];
    return sum / (pixels.rgba.length / 4);
}

/** The texel of `pixels` holding uv (u, v) (nearest, clamped). */
function texelAt(pixels: LinearPixels, u: number, v: number): number {
    const column = Math.min(pixels.width - 1, Math.max(0, Math.floor(u * pixels.width)));
    // See LinearPixels.flipY for which row holds which v.
    const rowFraction = pixels.flipY ? 1 - v : v;
    const row = Math.min(pixels.height - 1, Math.max(0, Math.floor(rowFraction * pixels.height)));
    return row * pixels.width + column;
}

/** How a diffuse average was made (see the file comment). */
export type DiffuseAverageMethod = 'per-texel' | 'separate-averages' | 'albedo-map' | 'metalness-map';

export interface DiffuseAverage {
    /** What to multiply the material's colour by for the GI albedo: average albedo × (1 − metalness). */
    readonly factor: ColorRGB;
    readonly method: DiffuseAverageMethod;
}

/**
 * The GI albedo factor for a material with an albedo map and/or a metalness
 * map: the average over the surface of albedoTexel × (1 − metalness ×
 * metalnessTexel.b), as the file comment says. `sameUv` says whether the two
 * maps share a uv mapping (channel and transform), so texels line up.
 * Undefined when neither map can be read.
 */
export function diffuseAverage(map: Texture | null, metalnessMap: Texture | null, metalness: number, sameUv: boolean): DiffuseAverage | undefined {
    const albedo = map ? readLinearPixels(map) : undefined;
    const metal = metalnessMap ? readLinearPixels(metalnessMap) : undefined;
    if (albedo && metal && sameUv) {
        const { rgba, width, height } = albedo;
        let red = 0;
        let green = 0;
        let blue = 0;
        let weight = 0;
        for (let row = 0; row < height; row++) {
            for (let column = 0; column < width; column++) {
                const base = (row * width + column) * 4;
                const alpha = rgba[base + 3];
                if (alpha <= 0) continue;
                // This texel's uv centre, then the metalness map's texel there.
                const u = (column + 0.5) / width;
                const rowFraction = (row + 0.5) / height;
                const v = albedo.flipY ? 1 - rowFraction : rowFraction;
                const diffuse = 1 - metalness * metal.rgba[texelAt(metal, u, v) * 4 + 2];
                red += rgba[base] * diffuse * alpha;
                green += rgba[base + 1] * diffuse * alpha;
                blue += rgba[base + 2] * diffuse * alpha;
                weight += alpha;
            }
        }
        if (!(weight > 0)) return undefined;
        return { factor: [red / weight, green / weight, blue / weight], method: 'per-texel' };
    }
    const albedoAverage = albedo ? averageLinear(albedo) : undefined;
    const diffuse = metal ? 1 - metalness * averageBlue(metal) : 1 - metalness;
    if (albedoAverage) return { factor: [albedoAverage[0] * diffuse, albedoAverage[1] * diffuse, albedoAverage[2] * diffuse], method: metal ? 'separate-averages' : 'albedo-map' };
    if (metal) return { factor: [diffuse, diffuse, diffuse], method: 'metalness-map' };
    return undefined;
}
