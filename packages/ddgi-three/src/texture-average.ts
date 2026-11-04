/**
 * The average linear colour of a texture, for GI until the core samples
 * textures itself (its texture atlas is M4).
 *
 * Until then a textured material would bounce light in its flat `color`
 * alone, usually white, so a red brick wall would bounce white. Folding the
 * texture's average into the GI albedo (and an emissive map's into the
 * emission) gets the right colour of light on average. It is computed once
 * per texture version, never per frame, from at most
 * {@link AVERAGE_SAMPLES_PER_AXIS}² texels.
 *
 * **When M4's atlas samples textures, remove the fold** (convert.ts,
 * `materialDescriptor`): the core would then multiply by the texture again.
 */

import { HalfFloatType, FloatType, SRGBColorSpace, type Texture } from 'three';
import type { ColorRGB } from '@pmndrs/ddgi-core';

/** Texels read per axis at most: a 32 × 32 grid of samples, or a downsample to that size. */
export const AVERAGE_SAMPLES_PER_AXIS = 32;

/** sRGB-encoded value in 0..1 to linear. */
export function srgbToLinear(value: number): number {
    return value <= 0.04045 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
}

/** A half float's bits to a number. */
function halfToFloat(bits: number): number {
    const exponent = (bits >> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    const sign = bits & 0x8000 ? -1 : 1;
    if (exponent === 0) return sign * Math.pow(2, -14) * (mantissa / 1024);
    if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
    return sign * Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
}

interface PixelSource {
    readonly data: ArrayLike<number>;
    readonly width: number;
    readonly height: number;
}

/**
 * The average of an RGBA pixel array, sampled on a grid of at most
 * AVERAGE_SAMPLES_PER_AXIS² texels. `decode` turns a stored channel value
 * into a linear one.
 */
export function averagePixels(pixels: PixelSource, decode: (value: number) => number): ColorRGB {
    const { data, width, height } = pixels;
    const columns = Math.min(width, AVERAGE_SAMPLES_PER_AXIS);
    const rows = Math.min(height, AVERAGE_SAMPLES_PER_AXIS);
    const sum = [0, 0, 0];
    for (let row = 0; row < rows; row++) {
        // Sample at grid-cell centres, so the grid spreads over the whole image.
        const y = Math.min(height - 1, Math.floor(((row + 0.5) * height) / rows));
        for (let column = 0; column < columns; column++) {
            const x = Math.min(width - 1, Math.floor(((column + 0.5) * width) / columns));
            const base = (y * width + x) * 4;
            for (let channel = 0; channel < 3; channel++) sum[channel] += decode(data[base + channel]);
        }
    }
    const count = rows * columns;
    return [sum[0] / count, sum[1] / count, sum[2] / count];
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
 * The texture's average linear colour, or undefined when its pixels can't be
 * read on the CPU (still loading, compressed, a render target, a format this
 * doesn't decode). Callers then use no fold (×1).
 */
export function textureAverage(texture: Texture): ColorRGB | undefined {
    const flags = texture as Texture & { isCompressedTexture?: boolean; isRenderTargetTexture?: boolean };
    if (flags.isCompressedTexture || flags.isRenderTargetTexture) return undefined;
    const image = texture.image as unknown;
    if (!image) return undefined;
    const srgb = texture.colorSpace === SRGBColorSpace;
    const fromByte = srgb ? (value: number) => srgbToLinear(value / 255) : (value: number) => value / 255;

    if (typeof ImageData !== 'undefined' && image instanceof ImageData) return averagePixels(image, fromByte);
    if (isDrawable(image)) {
        // The browser downsamples while drawing; the small copy is then averaged exactly.
        const width = (image as HTMLImageElement).naturalWidth || image.width;
        const height = (image as HTMLImageElement).naturalHeight || image.height;
        const targetWidth = Math.max(1, Math.min(width, AVERAGE_SAMPLES_PER_AXIS));
        const targetHeight = Math.max(1, Math.min(height, AVERAGE_SAMPLES_PER_AXIS));
        const canvas = new OffscreenCanvas(targetWidth, targetHeight);
        const context = canvas.getContext('2d');
        if (!context) return undefined;
        context.drawImage(image as CanvasImageSource, 0, 0, targetWidth, targetHeight);
        return averagePixels(context.getImageData(0, 0, targetWidth, targetHeight), fromByte);
    }
    const data = image as PixelSource;
    if (!ArrayBuffer.isView(data.data) || !(data.width > 0 && data.height > 0) || data.data.length < data.width * data.height * 4) return undefined;
    if (texture.type === FloatType && data.data instanceof Float32Array) return averagePixels(data, (value) => value);
    if (texture.type === HalfFloatType && data.data instanceof Uint16Array) return averagePixels(data, halfToFloat);
    if (data.data instanceof Uint8Array || data.data instanceof Uint8ClampedArray) return averagePixels(data, fromByte);
    return undefined;
}
