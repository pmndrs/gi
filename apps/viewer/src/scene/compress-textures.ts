/**
 * Compresses a loaded model's textures on the GPU at load (BC7 for colour and
 * data maps, BC5 for normal maps), with their mip chains, through gputex
 * (verekia/gputex 0.14.0, MIT; THIRD_PARTY.md).
 *
 * Why: three keeps every image uncompressed (RGBA8 with mipmaps, 4/3 × 4 bytes
 * a pixel). Bistro exterior's images then take 5.0 GB, which with DDGI's
 * 0.6 GB is more than the 6 GB RTX 3060 Laptop holds; the driver moves memory
 * in and out every frame the GI update runs, and a frame takes about 0.5 s
 * instead of 0.05 s (T2.17, NOTES). BC7 and BC5 take 1 byte a pixel, a quarter.
 *
 * How: gputex's engine-free encoders run on three's own device (the host owns
 * the device; no second WebGPU device). Each image is uploaded once, its mip
 * chain is box-filtered on the GPU, every level is encoded in one submission,
 * and the blocks are read back once. `buildCompressedTexture` (gputex/three)
 * wraps them in a three `CompressedTexture`, which three uploads like any other.
 * gputex's `compressTexture()` does the same, but with a device of ours it asks
 * for a new adapter and builds a new encoder on every call, so the two
 * encoders are made here once per model instead.
 *
 * The compressed bytes stay in `texture.mipmaps` after three uploads them: M4
 * reads them for GI (docs/design/m4-materials.md §5).
 *
 * Until M4, the three adapter averages the colours of the base colour,
 * emissive and metalness maps on the CPU for GI's albedo and emission
 * (packages/ddgi-three/src/texture-average.ts), and it can't read a
 * `CompressedTexture`. Compressing those maps would make GI see the material's
 * plain colour instead (white on every Bistro material). So the default,
 * `'compressed'`, leaves those three maps as they are; `'all-compressed'`
 * compresses them too, for the smallest memory at the cost of GI's colours.
 */

import { BC5Encoder, BC7Encoder, generateGpuMipChain, TextureFormat, type Encoder } from 'gputex';
import { buildCompressedTexture } from 'gputex/three';
import { SRGBColorSpace, type CompressedTexture, type Material, type Mesh, type Object3D, type Texture } from 'three';
import type { TextureCompression } from './texture-compression-mode';

export type { TextureCompression } from './texture-compression-mode';

/**
 * Material slots whose textures the three adapter averages on the CPU for GI
 * (texture-average.ts: `map` and `metalnessMap` for the albedo, `emissiveMap`
 * for the emission). It skips a `CompressedTexture`, so in `'compressed'` mode
 * these stay uncompressed. Empty this list once M4's adapter reads compressed
 * mips (T4.3).
 */
export const GI_AVERAGED_SLOTS: readonly string[] = ['map', 'emissiveMap', 'metalnessMap'];

/** Material slots holding tangent-space normal maps: BC5 keeps their two channels, and three rebuilds z (`NormalRGPacking`). */
const NORMAL_MAP_SLOTS: readonly string[] = ['normalMap', 'clearcoatNormalMap'];

/** What to do with one texture, decided from every material slot that uses it. */
export type TexturePlan =
    | { readonly action: 'compress'; readonly kind: 'normal' | 'colour' }
    | { readonly action: 'keep'; readonly reason: 'read-by-gi' | 'not-an-image' | 'size' };

/** What `compressModelTextures` did, for the loading log and the benchmark. */
export interface TextureCompressionReport {
    readonly mode: TextureCompression;
    /** Why nothing was compressed, when the device can't (no `texture-compression-bc`). */
    readonly unavailable: string | null;
    /** Distinct textures compressed, and how many of them were normal maps (BC5). */
    readonly compressed: number;
    readonly normalMaps: number;
    /** Textures left uncompressed because GI reads their colours on the CPU until M4. */
    readonly keptForGi: number;
    /** Textures left uncompressed for other reasons: not a decoded image, or a side that isn't a power of two. */
    readonly keptOther: number;
    /** GPU bytes of the compressed textures as three would have held them (RGBA8 plus mips), and as compressed. */
    readonly uncompressedBytes: number;
    readonly compressedBytes: number;
    /** Wall time of the whole pass, in milliseconds. */
    readonly milliseconds: number;
}

type ImageSource = ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas;

function isTexture(value: unknown): value is Texture {
    return typeof value === 'object' && value !== null && (value as { isTexture?: boolean }).isTexture === true;
}

function isDecodedImage(image: unknown): image is ImageSource {
    return (
        (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) ||
        (typeof HTMLImageElement !== 'undefined' && image instanceof HTMLImageElement) ||
        (typeof HTMLCanvasElement !== 'undefined' && image instanceof HTMLCanvasElement) ||
        (typeof OffscreenCanvas !== 'undefined' && image instanceof OffscreenCanvas)
    );
}

function isPowerOfTwo(value: number): boolean {
    return value > 0 && (value & (value - 1)) === 0;
}

/** Every material of the meshes under `root`. */
function materialsUnder(root: Object3D): Set<Material> {
    const materials = new Set<Material>();
    root.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) materials.add(material);
    });
    return materials;
}

/** Each texture under `root` and the material slots (`map`, `normalMap`, ...) that use it. */
export function textureSlots(root: Object3D): Map<Texture, Set<string>> {
    const slots = new Map<Texture, Set<string>>();
    for (const material of materialsUnder(root)) {
        for (const [slot, value] of Object.entries(material)) {
            if (!isTexture(value)) continue;
            let used = slots.get(value);
            if (!used) slots.set(value, (used = new Set()));
            used.add(slot);
        }
    }
    return slots;
}

/**
 * Whether and how to compress a texture used in `slots`:
 * - kept when any slot is one GI averages and the mode is `'compressed'`;
 * - kept when it isn't a decoded image (data textures, canvases being drawn, already compressed);
 * - kept when a side isn't a power of two: gputex pads each level to whole 4×4 blocks, and only
 *   power-of-two sides give a padded chain that matches the mip sizes WebGPU expects;
 * - a normal map (BC5) only when every slot that uses it is a normal-map slot; otherwise colour (BC7).
 */
export function planTexture(texture: Texture, slots: ReadonlySet<string>, mode: Exclude<TextureCompression, 'uncompressed'>): TexturePlan {
    if (mode === 'compressed' && [...slots].some((slot) => GI_AVERAGED_SLOTS.includes(slot))) return { action: 'keep', reason: 'read-by-gi' };
    if ((texture as { isCompressedTexture?: boolean }).isCompressedTexture) return { action: 'keep', reason: 'not-an-image' };
    const image = texture.image as unknown;
    if (!isDecodedImage(image)) return { action: 'keep', reason: 'not-an-image' };
    if (!isPowerOfTwo(image.width) || !isPowerOfTwo(image.height)) return { action: 'keep', reason: 'size' };
    const allNormal = [...slots].every((slot) => NORMAL_MAP_SLOTS.includes(slot));
    return { action: 'compress', kind: allNormal ? 'normal' : 'colour' };
}

/** Bytes three holds for an RGBA8 image with its full mip chain (4 bytes a pixel, plus a third for the mips). */
function rgba8WithMipsBytes(width: number, height: number): number {
    let bytes = 0;
    for (let w = width, h = height; ; w = Math.max(1, w >> 1), h = Math.max(1, h >> 1)) {
        bytes += w * h * 4;
        if (w === 1 && h === 1) break;
    }
    return bytes;
}

/**
 * Copies how `original` is sampled and placed onto its compressed copy: wrap,
 * filters, anisotropy, the uv transform and channel, the name and userData.
 * The colour space comes from the compressed format (sRGB BC7 for sRGB maps),
 * and `flipY` stays false: compressed textures can't be flipped on upload, and
 * the blocks keep the decoded image's row order, as three drew it (glTF
 * textures have `flipY = false`).
 */
function copySampling(original: Texture, compressed: CompressedTexture): void {
    compressed.name = original.name;
    compressed.wrapS = original.wrapS;
    compressed.wrapT = original.wrapT;
    compressed.magFilter = original.magFilter;
    compressed.minFilter = original.minFilter;
    compressed.anisotropy = original.anisotropy;
    compressed.channel = original.channel;
    compressed.offset.copy(original.offset);
    compressed.repeat.copy(original.repeat);
    compressed.center.copy(original.center);
    compressed.rotation = original.rotation;
    compressed.matrixAutoUpdate = original.matrixAutoUpdate;
    compressed.matrix.copy(original.matrix);
    compressed.userData = { ...original.userData, ...compressed.userData };
}

/**
 * Compresses the textures of the model under `root` on `device` (three's), and
 * puts the compressed copies into its materials. Returns what it did. Does
 * nothing in `'uncompressed'` mode, or when the device has no
 * `texture-compression-bc` (say, a mobile GPU: gputex would pick ASTC there,
 * which puts normal maps in `.ra`, a layout three's normal map node doesn't
 * read).
 */
export async function compressModelTextures(
    root: Object3D,
    device: GPUDevice,
    mode: TextureCompression,
    report: (done: number, total: number) => void = () => {},
): Promise<TextureCompressionReport> {
    const startedAt = performance.now();
    const empty = { mode, unavailable: null, compressed: 0, normalMaps: 0, keptForGi: 0, keptOther: 0, uncompressedBytes: 0, compressedBytes: 0 };
    if (mode === 'uncompressed') return { ...empty, milliseconds: 0 };
    if (!device.features.has('texture-compression-bc')) {
        return { ...empty, unavailable: 'the device has no texture-compression-bc', milliseconds: performance.now() - startedAt };
    }

    const slots = textureSlots(root);
    const plans = new Map<Texture, TexturePlan>();
    for (const [texture, used] of slots) plans.set(texture, planTexture(texture, used, mode));
    const toCompress = [...plans].filter(([, plan]) => plan.action === 'compress') as Array<[Texture, Extract<TexturePlan, { action: 'compress' }>]>;

    // One encoder of each kind, on three's device; `ownsDevice: false` leaves the device alone on destroy.
    const colourEncoder: Encoder = new BC7Encoder({ device, ownsDevice: false });
    const normalEncoder: Encoder = new BC5Encoder({ device, ownsDevice: false });
    const replacements = new Map<Texture, CompressedTexture>();
    // GLTFLoader shares one decoded image between textures that use the same file: encode it once per kind.
    const encodedImages = new Map<string, CompressedTexture>();
    const imageIds = new Map<ImageSource, number>();
    const handedOut = new Set<CompressedTexture>();
    let uncompressedBytes = 0;
    let compressedBytes = 0;
    try {
        for (const [index, [texture, plan]] of toCompress.entries()) {
            report(index, toCompress.length);
            const image = texture.image as ImageSource;
            if (!imageIds.has(image)) imageIds.set(image, imageIds.size);
            const srgb = plan.kind === 'colour' && texture.colorSpace === SRGBColorSpace;
            const key = `${imageIds.get(image)}:${plan.kind}:${srgb}`;
            let encoded = encodedImages.get(key);
            if (!encoded) {
                // generateGpuMipChain takes bitmaps and canvases, not image elements.
                const source = image instanceof HTMLImageElement ? await createImageBitmap(image, { premultiplyAlpha: 'none', colorSpaceConversion: 'none' }) : image;
                const chain = await generateGpuMipChain(device, source, { flipY: false });
                try {
                    const { levels } = await (plan.kind === 'normal' ? normalEncoder : colourEncoder).encodeMipChainFromTexture(chain);
                    const format = plan.kind === 'normal' ? TextureFormat.BC5 : srgb ? TextureFormat.BC7_SRGB : TextureFormat.BC7;
                    encoded = buildCompressedTexture(levels, format);
                } finally {
                    chain.destroy();
                    if (source !== image) (source as ImageBitmap).close();
                }
                encodedImages.set(key, encoded);
                uncompressedBytes += rgba8WithMipsBytes(image.width, image.height);
                for (const level of encoded.mipmaps as Array<{ data: Uint8Array }>) compressedBytes += level.data.byteLength;
            }
            // Each three texture gets its own texture object (its own wrap and uv transform); the first
            // takes the encoded one, later ones a clone, which shares the blocks (`source` and `mipmaps`).
            const compressed = handedOut.has(encoded) ? (encoded.clone() as CompressedTexture) : encoded;
            handedOut.add(encoded);
            copySampling(texture, compressed);
            compressed.needsUpdate = true;
            replacements.set(texture, compressed);
        }
        report(toCompress.length, toCompress.length);
    } finally {
        colourEncoder.destroy();
        normalEncoder.destroy();
    }

    for (const material of materialsUnder(root)) {
        let changed = false;
        const fields = material as unknown as Record<string, unknown>;
        for (const [slot, value] of Object.entries(material)) {
            const replacement = isTexture(value) ? replacements.get(value) : undefined;
            if (!replacement) continue;
            fields[slot] = replacement;
            changed = true;
        }
        // A normal map in BC5 changes the material's shader (two channels, z rebuilt).
        if (changed) material.needsUpdate = true;
    }

    // The decoded images are no longer drawn by any texture. Close a bitmap only when every texture using it was replaced.
    const imagesStillUsed = new Set<unknown>();
    for (const [texture, plan] of plans) if (plan.action === 'keep') imagesStillUsed.add(texture.image);
    for (const texture of replacements.keys()) {
        const image = texture.image as unknown;
        texture.dispose();
        if (image instanceof ImageBitmap && !imagesStillUsed.has(image)) image.close();
    }

    let keptForGi = 0;
    let keptOther = 0;
    for (const plan of plans.values()) {
        if (plan.action !== 'keep') continue;
        if (plan.reason === 'read-by-gi') keptForGi++;
        else keptOther++;
    }
    return {
        mode,
        unavailable: null,
        compressed: replacements.size,
        normalMaps: toCompress.filter(([, plan]) => plan.kind === 'normal').length,
        keptForGi,
        keptOther,
        uncompressedBytes,
        compressedBytes,
        milliseconds: performance.now() - startedAt,
    };
}
