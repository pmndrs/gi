/**
 * What the texture store needs to know about a source's GPU format: whether
 * it is block-compressed (and how big its blocks are), which device feature
 * it needs, whether the GPU decodes sRGB for it, and whether the store's
 * kernel can read it at all (it reads sources as float with `textureLoad`).
 *
 * Pure functions of the format's name, so they run in Node tests.
 */

/** A block-compressed format's block: its size in texels and in bytes. */
export interface CompressedBlock {
    readonly width: number;
    readonly height: number;
    readonly bytes: number;
}

/** The device feature a block-compressed format family needs. */
export type CompressionFeature = 'texture-compression-bc' | 'texture-compression-etc2' | 'texture-compression-astc';

/** BC formats whose blocks are 8 bytes (one 64-bit block per 4×4); every other BC format has 16. */
const BC_EIGHT_BYTE_BLOCKS = new Set(['bc1-rgba-unorm', 'bc1-rgba-unorm-srgb', 'bc4-r-unorm', 'bc4-r-snorm']);

/** ETC2 / EAC formats whose blocks are 8 bytes; the rest (RGBA8 and the two-channel EAC) have 16. */
const ETC2_EIGHT_BYTE_BLOCKS = new Set(['etc2-rgb8unorm', 'etc2-rgb8unorm-srgb', 'etc2-rgb8a1unorm', 'etc2-rgb8a1unorm-srgb', 'eac-r11unorm', 'eac-r11snorm']);

/** The device feature a compressed format needs, or null for an uncompressed format. */
export function compressionFeature(format: string): CompressionFeature | null {
    if (format.startsWith('bc')) return 'texture-compression-bc';
    if (format.startsWith('etc2') || format.startsWith('eac')) return 'texture-compression-etc2';
    if (format.startsWith('astc')) return 'texture-compression-astc';
    return null;
}

/**
 * The block of a compressed format, or null for an uncompressed one.
 * Every BC, ETC2 and EAC format has 4×4 blocks; ASTC names its block in the
 * format (`astc-6x5-unorm`), always 16 bytes.
 */
export function compressedBlock(format: string): CompressedBlock | null {
    const family = compressionFeature(format);
    if (family === 'texture-compression-bc') return { width: 4, height: 4, bytes: BC_EIGHT_BYTE_BLOCKS.has(format) ? 8 : 16 };
    if (family === 'texture-compression-etc2') return { width: 4, height: 4, bytes: ETC2_EIGHT_BYTE_BLOCKS.has(format) ? 8 : 16 };
    if (family === 'texture-compression-astc') {
        const match = /^astc-(\d+)x(\d+)-/.exec(format);
        if (!match) return null;
        return { width: Number(match[1]), height: Number(match[2]), bytes: 16 };
    }
    return null;
}

/** Whether the GPU decodes sRGB when the format is read (`rgba8unorm-srgb`, `bc7-rgba-unorm-srgb`). */
export function decodesSrgb(format: string): boolean {
    return format.endsWith('-srgb');
}

/**
 * Whether the store's kernel can read the format: `textureLoad` on a
 * `texture_2d<f32>`, which takes every colour format whose values are float
 * or normalized. Depth and stencil formats, and integer formats (`uint`,
 * `sint`), are refused.
 */
export function readableAsFloat(format: string): boolean {
    if (format.startsWith('depth') || format.startsWith('stencil')) return false;
    return !(format.endsWith('uint') || format.endsWith('sint'));
}

/**
 * Of a compressed image's mip levels (largest first), the one the store
 * uploads: the smallest that still covers the store's layer on both axes,
 * so the store's kernel never upsamples a level when a finer one exists and
 * uploads as few bytes as it can (64 KB of BC7 at 256²). When even the
 * first level is smaller than the layer, the first (largest) level.
 */
export function chooseCompressedLevel(levels: readonly { readonly width: number; readonly height: number }[], layerSize: number): number {
    let chosen = 0;
    for (let index = 0; index < levels.length; index++) {
        if (levels[index].width >= layerSize && levels[index].height >= layerSize) chosen = index;
    }
    return chosen;
}
