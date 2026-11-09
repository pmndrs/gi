/**
 * Getting a texture's source onto the GPU in a form the store's kernel reads
 * (`gi/texture-store`, texture-store-wgsl.ts): a 2D view of one mip level
 * that `textureLoad` reads as float, and whether its values are still
 * sRGB-encoded (the kernel then decodes them before it averages).
 *
 * Each kind of source gets there one way (docs/design/m4-materials.md §2):
 *
 * - images (`ImageBitmap`, image elements, canvases): `copyExternalImageToTexture`
 *   into an `rgba8unorm` staging texture at full size;
 * - `PixelData` and `ImageData`: `writeTexture` into the same kind of staging texture;
 * - a same-device `GPUTexture`: read where it is;
 * - `CompressedPixelData`: one level (`chooseCompressedLevel`) written into a
 *   staging texture of its own format, which the GPU decodes when the kernel
 *   reads it.
 *
 * Staging textures belong to the store, which destroys them in the next
 * `encode` (after the host has submitted the frame that read them).
 */

import type { CompressedPixelData, PixelData, TextureSource } from '../contract/scene.ts';
import { chooseCompressedLevel, compressedBlock, compressionFeature, decodesSrgb, readableAsFloat } from './texture-formats.ts';

/** `GPUTextureUsage` bits; the global doesn't exist in Node, where tests import this file. */
const TEXTURE_USAGE = { copySrc: 0x01, copyDst: 0x02, textureBinding: 0x04, storageBinding: 0x08, renderAttachment: 0x10 } as const;

/** A source on the GPU, ready for the store's kernel. */
export interface StagedSource {
    /** A 2D view of one mip level and one layer, read as float. */
    readonly view: GPUTextureView;
    /** The values read are sRGB-encoded: the kernel decodes them to linear light before averaging. */
    readonly encoded: boolean;
    /** A texture the store made for this source, destroyed after the frame's submit; null for a host's `GPUTexture`. */
    readonly staging: GPUTexture | null;
}

/** Why a source can't be used. The store warns with it (once per message) and the texture's materials keep their flat colour. */
export interface UnusableSource {
    readonly reason: string;
}

export type SourceKind = 'gpu-texture' | 'compressed' | 'pixels' | 'image';

/**
 * Which way a source goes. Checked by shape, not `instanceof`, so the
 * checks run in Node (no `GPUTexture` there) and accept sources from
 * another realm (a worker's `ImageData`).
 */
export function sourceKind(source: TextureSource): SourceKind {
    const fields = source as unknown as Record<string, unknown>;
    if (typeof fields.createView === 'function' && typeof fields.format === 'string') return 'gpu-texture';
    if (typeof fields.format === 'string' && Array.isArray(fields.levels)) return 'compressed';
    // PixelData and ImageData both carry their bytes in `data`.
    if (ArrayBuffer.isView(fields.data) && typeof fields.width === 'number' && typeof fields.height === 'number') return 'pixels';
    return 'image';
}

/** A source's width and height in texels, as the store reads it (a compressed source's chosen level). */
export function sourceSize(source: TextureSource, layerSize: number): { readonly width: number; readonly height: number } {
    const kind = sourceKind(source);
    if (kind === 'compressed') {
        const levels = (source as CompressedPixelData).levels;
        const level = levels[chooseCompressedLevel(levels, layerSize)];
        return level ? { width: level.width, height: level.height } : { width: 0, height: 0 };
    }
    if (typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement) return { width: source.naturalWidth, height: source.naturalHeight };
    const sized = source as { readonly width: number; readonly height: number };
    return { width: sized.width, height: sized.height };
}

/**
 * Puts a source on the GPU (see the file comment). `srgb` is the texture
 * descriptor's: whether the source's values are sRGB-encoded, unless the
 * source's format decodes sRGB itself.
 */
export function stageSource(device: GPUDevice, source: TextureSource, srgb: boolean, layerSize: number, label: string): StagedSource | UnusableSource {
    const kind = sourceKind(source);
    if (kind === 'gpu-texture') return hostTexture(source as GPUTexture, srgb);
    if (kind === 'compressed') return compressedTexture(device, source as CompressedPixelData, srgb, layerSize, label);

    const { width, height } = sourceSize(source, layerSize);
    if (!(width > 0 && height > 0)) return { reason: `@pmndrs/ddgi: a texture source has no pixels yet (${width} × ${height}); the materials using it keep their flat colour. Create the texture once its image has loaded.` };
    const limit = device.limits.maxTextureDimension2D;
    if (width > limit || height > limit) return { reason: `@pmndrs/ddgi: a ${width} × ${height} texture is over the device's maxTextureDimension2D (${limit}); the materials using it keep their flat colour.` };
    if (kind === 'pixels') {
        const pixels = source as PixelData;
        if (pixels.data.byteLength < width * height * 4) return { reason: '@pmndrs/ddgi: a PixelData texture has fewer than width × height × 4 bytes; the materials using it keep their flat colour.' };
    }

    // COPY_DST for both uploads; RENDER_ATTACHMENT because copyExternalImageToTexture requires it.
    const staging = device.createTexture({
        label: `${label} staging`,
        size: { width, height },
        format: 'rgba8unorm',
        usage: TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copyDst | TEXTURE_USAGE.renderAttachment,
    });
    if (kind === 'pixels') {
        const pixels = source as PixelData;
        device.queue.writeTexture({ texture: staging }, pixels.data as Uint8Array<ArrayBuffer>, { bytesPerRow: width * 4, rowsPerImage: height }, { width, height });
    } else {
        // The image's own bytes, unconverted: no premultiplication, no flip, sRGB-encoded as image files are.
        device.queue.copyExternalImageToTexture({ source: source as GPUCopyExternalImageSource, flipY: false }, { texture: staging, premultipliedAlpha: false }, { width, height });
    }
    return { view: staging.createView({ label: `${label} staging`, dimension: '2d' }), encoded: srgb, staging };
}

function hostTexture(texture: GPUTexture, srgb: boolean): StagedSource | UnusableSource {
    if ((texture.usage & TEXTURE_USAGE.textureBinding) === 0) return { reason: '@pmndrs/ddgi: a GPUTexture texture source needs TEXTURE_BINDING usage; the materials using it keep their flat colour.' };
    if (texture.dimension !== '2d') return { reason: `@pmndrs/ddgi: a GPUTexture texture source must be 2D, not ${texture.dimension}; the materials using it keep their flat colour.` };
    if (!readableAsFloat(texture.format)) return { reason: `@pmndrs/ddgi: a GPUTexture texture source in ${texture.format} can't be read as a colour; the materials using it keep their flat colour.` };
    // Its first mip level and layer: the full image, whatever mips the host made (or didn't).
    const view = texture.createView({ dimension: '2d', baseMipLevel: 0, mipLevelCount: 1, baseArrayLayer: 0, arrayLayerCount: 1 });
    return { view, encoded: srgb && !decodesSrgb(texture.format), staging: null };
}

function compressedTexture(device: GPUDevice, source: CompressedPixelData, srgb: boolean, layerSize: number, label: string): StagedSource | UnusableSource {
    const format = source.format;
    const feature = compressionFeature(format);
    if (feature && !device.features.has(feature)) {
        return { reason: `@pmndrs/ddgi: compressed textures in ${format} need the device feature '${feature}', which this device wasn't created with; the materials using them keep their flat colour.` };
    }
    const block = compressedBlock(format) ?? { width: 1, height: 1, bytes: 4 };
    if (!feature && format !== 'rgba8unorm' && format !== 'rgba8unorm-srgb') {
        return { reason: `@pmndrs/ddgi: CompressedPixelData in ${format} isn't supported; use a compressed format, or rgba8unorm.` };
    }
    const level = source.levels[chooseCompressedLevel(source.levels, layerSize)];
    if (!level || !(level.width > 0 && level.height > 0)) return { reason: '@pmndrs/ddgi: a CompressedPixelData texture has no levels; the materials using it keep their flat colour.' };

    // A compressed texture's size must be whole blocks. A level whose size isn't (a small or odd level)
    // still holds whole blocks, the extra texels being padding: the staging texture covers them.
    const blocksWide = Math.ceil(level.width / block.width);
    const blocksHigh = Math.ceil(level.height / block.height);
    const bytesNeeded = blocksWide * blocksHigh * block.bytes;
    if (level.data.byteLength < bytesNeeded) {
        return { reason: `@pmndrs/ddgi: a ${format} level of ${level.width} × ${level.height} needs ${bytesNeeded} bytes, got ${level.data.byteLength}; the materials using it keep their flat colour.` };
    }
    const size = { width: blocksWide * block.width, height: blocksHigh * block.height };
    const staging = device.createTexture({ label: `${label} staging (${format})`, size, format, usage: TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copyDst });
    device.queue.writeTexture({ texture: staging }, level.data as Uint8Array<ArrayBuffer>, { bytesPerRow: blocksWide * block.bytes, rowsPerImage: blocksHigh }, size);
    return { view: staging.createView({ label: `${label} staging`, dimension: '2d' }), encoded: srgb && !decodesSrgb(format), staging };
}
