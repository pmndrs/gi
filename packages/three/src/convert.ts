/**
 * three.js objects to the platform's plain scene descriptors (`SceneApi`,
 * packages/core/src/platform/contract/scene.ts).
 *
 * Pure conversions: nothing here touches the GPU or keeps state, so each rule
 * has a unit test (convert.test.ts). The scene sync (scene-sync.ts) decides
 * *when* to call them.
 *
 * Units pass straight through: the platform uses three's physical light units
 * (directional intensity is irradiance in lux, point and spot intensity is
 * candela, emission is radiance), and three's `Color` already holds linear
 * working-space values.
 */

import {
    ClampToEdgeWrapping,
    Color,
    ColorManagement,
    MirroredRepeatWrapping,
    R11_EAC_Format,
    RED_GREEN_RGTC2_Format,
    RED_RGTC1_Format,
    RG11_EAC_Format,
    RGBA_ASTC_10x10_Format,
    RGBA_ASTC_10x5_Format,
    RGBA_ASTC_10x6_Format,
    RGBA_ASTC_10x8_Format,
    RGBA_ASTC_12x10_Format,
    RGBA_ASTC_12x12_Format,
    RGBA_ASTC_4x4_Format,
    RGBA_ASTC_5x4_Format,
    RGBA_ASTC_5x5_Format,
    RGBA_ASTC_6x5_Format,
    RGBA_ASTC_6x6_Format,
    RGBA_ASTC_8x5_Format,
    RGBA_ASTC_8x6_Format,
    RGBA_ASTC_8x8_Format,
    RGBA_BPTC_Format,
    RGBA_ETC2_EAC_Format,
    RGBA_S3TC_DXT1_Format,
    RGBA_S3TC_DXT3_Format,
    RGBA_S3TC_DXT5_Format,
    RGB_ETC1_Format,
    RGB_ETC2_Format,
    RGB_S3TC_DXT1_Format,
    SRGBTransfer,
    type BufferAttribute,
    type BufferGeometry,
    type CompressedPixelFormat,
    type DirectionalLight,
    type InterleavedBufferAttribute,
    type Light,
    type Material,
    type PointLight,
    type SpotLight,
    type Texture,
    type Wrapping,
} from 'three';
import type {
    AlphaMode,
    ColorRGB,
    CompressedPixelData,
    CompressedPixelLevel,
    LightDescriptor,
    MaterialDescriptor,
    MeshDescriptor,
    MeshGroup,
    MetalnessTexture,
    PixelData,
    TextureDescriptor,
    TextureHandle,
    TextureSource,
    UvMapping,
    UvWrap,
    Vec3,
} from '@pmndrs/gi-core';
import { blocksLightOptIn, castsShadowsForGI, giAlbedoMapOverride, giAlbedoOverride, giEmissiveOverride, lightGroup, singleSidedForGI } from './user-data.ts';

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

type Attribute = BufferAttribute | InterleavedBufferAttribute;

/**
 * An attribute as tightly packed float32, `itemSize` floats per vertex.
 *
 * Plain float attributes are passed through without a copy (the platform
 * packs meshes as soon as `createMesh` is called, so sharing the array is
 * safe). Interleaved and normalized-integer attributes (glTF quantization)
 * are read through three's accessors, which undo the normalization.
 */
export function readFloatAttribute(attribute: Attribute, itemSize: number): Float32Array {
    const count = attribute.count;
    const plain = attribute as BufferAttribute;
    if (
        !(attribute as InterleavedBufferAttribute).isInterleavedBufferAttribute &&
        plain.array instanceof Float32Array &&
        !plain.normalized &&
        plain.itemSize === itemSize
    ) {
        return plain.array.length === count * itemSize ? plain.array : plain.array.subarray(0, count * itemSize);
    }
    const values = new Float32Array(count * itemSize);
    for (let vertex = 0; vertex < count; vertex++) {
        values[vertex * itemSize] = attribute.getX(vertex);
        if (itemSize > 1) values[vertex * itemSize + 1] = attribute.getY(vertex);
        if (itemSize > 2) values[vertex * itemSize + 2] = attribute.getZ(vertex);
    }
    return values;
}

/** The triangle list: the geometry's own index, or `0..n-1` for a non-indexed geometry (the contract asks the adapter for that). */
export function readIndices(geometry: BufferGeometry, vertexCount: number): Uint16Array | Uint32Array {
    const index = geometry.index;
    if (index) {
        const array = index.array;
        if (array instanceof Uint16Array || array instanceof Uint32Array) {
            return array.length === index.count ? array : array.subarray(0, index.count);
        }
        // Uint8 or another integer type: widen.
        const widened = new Uint32Array(index.count);
        for (let entry = 0; entry < index.count; entry++) widened[entry] = index.getX(entry);
        return widened;
    }
    const sequential = vertexCount <= 0x10000 ? new Uint16Array(vertexCount) : new Uint32Array(vertexCount);
    for (let vertex = 0; vertex < vertexCount; vertex++) sequential[vertex] = vertex;
    return sequential;
}

/**
 * Material runs from `geometry.groups`. A group's `start` and `count` count
 * indices for an indexed geometry and vertices for a non-indexed one, which is
 * the same thing once `readIndices` has made the sequential index. Counts are
 * clipped to the index list (three allows `Infinity`) and rounded down to whole
 * triangles. Returns undefined when the geometry has no groups.
 */
export function readGroups(geometry: BufferGeometry, indexCount: number): MeshGroup[] | undefined {
    if (geometry.groups.length === 0) return undefined;
    const groups: MeshGroup[] = [];
    for (const group of geometry.groups) {
        const start = Math.min(group.start, indexCount);
        const available = indexCount - start;
        const count = Math.min(group.count, available);
        const whole = count - (count % 3);
        if (whole <= 0) continue;
        groups.push({ start, count: whole, materialSlot: group.materialIndex ?? 0 });
    }
    return groups;
}

/** How many material slots a geometry's groups use (1 without groups). */
export function materialSlotCount(geometry: BufferGeometry): number {
    let slots = 1;
    for (const group of geometry.groups) slots = Math.max(slots, (group.materialIndex ?? 0) + 1);
    return slots;
}

export interface MeshConversionOptions {
    /** Pass the second uv set (`uv1`). Only when a material samples channel 1: it costs memory on the GPU (NOTES: 16 MB on Sponza). */
    readonly includeUv1?: boolean;
    readonly label?: string;
}

/**
 * A `MeshDescriptor` for a geometry: positions, indices (made up for
 * non-indexed geometry), vertex normals, uv and uv1, and groups.
 * Returns null for a geometry without positions.
 */
export function meshDescriptor(geometry: BufferGeometry, options: MeshConversionOptions = {}): MeshDescriptor | null {
    const position = geometry.getAttribute('position');
    if (!position || position.count === 0) return null;
    const positions = readFloatAttribute(position, 3);
    const indices = readIndices(geometry, position.count);
    const normal = geometry.getAttribute('normal');
    const uv = geometry.getAttribute('uv');
    const uv1 = options.includeUv1 ? geometry.getAttribute('uv1') : undefined;
    return {
        positions,
        indices,
        normals: normal ? readFloatAttribute(normal, 3) : undefined,
        uvs: uv ? readFloatAttribute(uv, 2) : undefined,
        uvs1: uv1 ? readFloatAttribute(uv1, 2) : undefined,
        groups: readGroups(geometry, indices.length),
        label: options.label ?? (geometry.name || undefined),
    };
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

/** The texture flags convert.ts reads; three sets them on the classes, so a missing one is false. */
type TextureFlags = Texture & {
    isCompressedTexture?: boolean;
    isCompressedArrayTexture?: boolean;
    isCompressedCubeTexture?: boolean;
    isRenderTargetTexture?: boolean;
    isDataTexture?: boolean;
    isVideoTexture?: boolean;
};

/** three's wrap modes as the platform's. */
export function uvWrap(wrap: Wrapping): UvWrap {
    if (wrap === ClampToEdgeWrapping) return 'clamp';
    if (wrap === MirroredRepeatWrapping) return 'mirror';
    return 'repeat';
}

/**
 * Whether three turns the texture upside down when it uploads it: `flipY`
 * images and data textures. three never flips compressed textures (their
 * blocks can't be flipped) or render targets.
 */
export function flippedOnUpload(texture: Texture): boolean {
    const flags = texture as TextureFlags;
    return texture.flipY && !flags.isCompressedTexture && !flags.isCompressedArrayTexture && !flags.isRenderTargetTexture;
}

/**
 * Where a texture is sampled: three's `texture.matrix` (offset, repeat,
 * rotation, centre) and `texture.channel`, as the contract's 3×2
 * `[e[0], e[1], e[3], e[4], e[6], e[7]]`, after `updateMatrix()` when three
 * would update it itself.
 *
 * **`flipY`.** The platform's v = 0 is the source's first row (the top of an
 * image), and it never flips. three's WebGPU backend samples at
 * `M · uv` in its uploaded copy, which for a `flipY` texture is the image
 * upside down: three's v = 0 is the image's bottom row. So for those the
 * mapping is followed by v → 1 − v, which negates the matrix's second row
 * and moves its offset: `[a, −b, c, −d, e, 1 − f]`. glTF textures have
 * `flipY = false` and pass through unchanged.
 *
 * The platform has two uv sets; a texture on channel 2 or 3 falls back to
 * channel 0, and `channelSupported` says so, so the caller can warn.
 */
export function uvMapping(texture: Texture): UvMapping & { readonly channelSupported: boolean } {
    if (texture.matrixAutoUpdate) texture.updateMatrix();
    const e = texture.matrix.elements;
    const channelSupported = texture.channel === 0 || texture.channel === 1;
    const matrix: [number, number, number, number, number, number] = flippedOnUpload(texture) ? [e[0], -e[1], e[3], -e[4], e[6], 1 - e[7]] : [e[0], e[1], e[3], e[4], e[6], e[7]];
    return {
        channel: channelSupported ? (texture.channel as 0 | 1) : 0,
        matrix,
        wrapU: uvWrap(texture.wrapS),
        wrapV: uvWrap(texture.wrapT),
        channelSupported,
    };
}

/** The platform's mapping without the adapter's extra flag. */
export function contractUvMapping(texture: Texture): UvMapping {
    const { channel, matrix, wrapU, wrapV } = uvMapping(texture);
    return { channel, matrix, wrapU, wrapV };
}

/** A three compressed format's WebGPU formats: linear, and sRGB where WebGPU has one. */
interface CompressedFormatPair {
    readonly linear: GPUTextureFormat;
    readonly srgb?: GPUTextureFormat;
}

/** ASTC's block sizes, each with its three constant. WebGPU names them `astc-<w>x<h>-unorm[-srgb]`. */
const ASTC_BLOCKS: readonly (readonly [CompressedPixelFormat, string])[] = [
    [RGBA_ASTC_4x4_Format, '4x4'],
    [RGBA_ASTC_5x4_Format, '5x4'],
    [RGBA_ASTC_5x5_Format, '5x5'],
    [RGBA_ASTC_6x5_Format, '6x5'],
    [RGBA_ASTC_6x6_Format, '6x6'],
    [RGBA_ASTC_8x5_Format, '8x5'],
    [RGBA_ASTC_8x6_Format, '8x6'],
    [RGBA_ASTC_8x8_Format, '8x8'],
    [RGBA_ASTC_10x5_Format, '10x5'],
    [RGBA_ASTC_10x6_Format, '10x6'],
    [RGBA_ASTC_10x8_Format, '10x8'],
    [RGBA_ASTC_10x10_Format, '10x10'],
    [RGBA_ASTC_12x10_Format, '12x10'],
    [RGBA_ASTC_12x12_Format, '12x12'],
];

/**
 * three's compressed formats that map onto a WebGPU format GI can read, as
 * three's own WebGPU backend maps them (`WebGPUTextureUtils.getFormat`).
 * Left out: BC6H (half-float HDR; the texture store is 8-bit) and the signed
 * formats (normal maps, never an albedo or emissive map).
 */
const COMPRESSED_FORMATS: ReadonlyMap<CompressedPixelFormat, CompressedFormatPair> = new Map<CompressedPixelFormat, CompressedFormatPair>([
    [RGB_S3TC_DXT1_Format, { linear: 'bc1-rgba-unorm', srgb: 'bc1-rgba-unorm-srgb' }],
    [RGBA_S3TC_DXT1_Format, { linear: 'bc1-rgba-unorm', srgb: 'bc1-rgba-unorm-srgb' }],
    [RGBA_S3TC_DXT3_Format, { linear: 'bc2-rgba-unorm', srgb: 'bc2-rgba-unorm-srgb' }],
    [RGBA_S3TC_DXT5_Format, { linear: 'bc3-rgba-unorm', srgb: 'bc3-rgba-unorm-srgb' }],
    [RED_RGTC1_Format, { linear: 'bc4-r-unorm' }],
    [RED_GREEN_RGTC2_Format, { linear: 'bc5-rg-unorm' }],
    [RGBA_BPTC_Format, { linear: 'bc7-rgba-unorm', srgb: 'bc7-rgba-unorm-srgb' }],
    [RGB_ETC1_Format, { linear: 'etc2-rgb8unorm', srgb: 'etc2-rgb8unorm-srgb' }],
    [RGB_ETC2_Format, { linear: 'etc2-rgb8unorm', srgb: 'etc2-rgb8unorm-srgb' }],
    [RGBA_ETC2_EAC_Format, { linear: 'etc2-rgba8unorm', srgb: 'etc2-rgba8unorm-srgb' }],
    [R11_EAC_Format, { linear: 'eac-r11unorm' }],
    [RG11_EAC_Format, { linear: 'eac-rg11unorm' }],
    ...ASTC_BLOCKS.map(([format, block]): [CompressedPixelFormat, CompressedFormatPair] => [
        format,
        { linear: `astc-${block}-unorm` as GPUTextureFormat, srgb: `astc-${block}-unorm-srgb` as GPUTextureFormat },
    ]),
]);

/** Whether three decodes the texture's values as sRGB (its colour space has the sRGB transfer, as three's backend asks). */
function hasSrgbTransfer(texture: Texture): boolean {
    return ColorManagement.getTransfer(texture.colorSpace) === SRGBTransfer;
}

/**
 * The WebGPU format of a three `CompressedTexture`, chosen as three chooses
 * it: the sRGB variant when the texture's colour space has the sRGB transfer
 * and WebGPU has one. Null for formats GI doesn't read ({@link COMPRESSED_FORMATS}).
 */
export function compressedFormat(texture: Texture): GPUTextureFormat | null {
    const pair = COMPRESSED_FORMATS.get(texture.format as CompressedPixelFormat);
    if (!pair) return null;
    return hasSrgbTransfer(texture) && pair.srgb ? pair.srgb : pair.linear;
}

/** A typed array's bytes, without a copy. */
function bytesOf(view: ArrayBufferView): Uint8Array {
    return view instanceof Uint8Array ? view : new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
}

/**
 * A `CompressedTexture`'s mips as the platform's `CompressedPixelData`, from
 * `texture.mipmaps` (what KTX2Loader and gputex fill), largest first. The
 * platform uploads one of them (the smallest still covering its 256² layer)
 * and the GPU decodes it. Null when the format isn't read or there are no
 * mips yet.
 */
export function compressedSource(texture: Texture): CompressedPixelData | null {
    const format = compressedFormat(texture);
    if (!format) return null;
    const levels: CompressedPixelLevel[] = [];
    for (const mip of (texture.mipmaps ?? []) as { data?: ArrayBufferView; width: number; height: number }[]) {
        // A level without data ends the usable chain: the levels after it would be out of order.
        if (!mip || !ArrayBuffer.isView(mip.data) || !(mip.width > 0 && mip.height > 0)) break;
        levels.push({ data: bytesOf(mip.data), width: mip.width, height: mip.height });
    }
    return levels.length > 0 ? { format, levels } : null;
}

interface DataImage {
    readonly data: ArrayBufferView | null;
    readonly width: number;
    readonly height: number;
}

/**
 * The texture's pixels in a form `createTexture` accepts, or null when there
 * are none yet (an image still loading) or GI can't read them
 * ({@link unreadableTextureReason} says why):
 * - images: `ImageBitmap` (GLTFLoader), image elements once loaded,
 *   canvases, `ImageData`;
 * - 8-bit RGBA data textures, as `PixelData`;
 * - `CompressedTexture`s, as `CompressedPixelData` ({@link compressedSource}).
 */
export function textureSource(texture: Texture): TextureSource | null {
    const flags = texture as TextureFlags;
    if (flags.isRenderTargetTexture || flags.isCompressedArrayTexture || flags.isCompressedCubeTexture) return null;
    if (flags.isCompressedTexture) return compressedSource(texture);
    const image = texture.image as unknown;
    if (!image) return null;
    if (typeof ImageBitmap !== 'undefined' && image instanceof ImageBitmap) return image;
    if (typeof HTMLImageElement !== 'undefined' && image instanceof HTMLImageElement) return image.complete && image.naturalWidth > 0 ? image : null;
    if (typeof HTMLCanvasElement !== 'undefined' && image instanceof HTMLCanvasElement) return image;
    if (typeof OffscreenCanvas !== 'undefined' && image instanceof OffscreenCanvas) return image;
    if (typeof ImageData !== 'undefined' && image instanceof ImageData) return image;
    const data = image as DataImage;
    if ((data.data instanceof Uint8Array || data.data instanceof Uint8ClampedArray) && data.data.length === data.width * data.height * 4) {
        const pixels: PixelData = { data: data.data, width: data.width, height: data.height };
        return pixels;
    }
    return null;
}

/**
 * Why GI can't read a texture, as a warning, or null when it can (or its
 * image simply hasn't loaded yet). Messages name the kind, not the texture,
 * so a warn-once sink shows each kind once.
 */
export function unreadableTextureReason(texture: Texture): string | null {
    const flags = texture as TextureFlags;
    const fallback = 'materials using one shade with their colour only.';
    if (flags.isRenderTargetTexture) return `render-target textures are not read for GI; ${fallback}`;
    if (flags.isCompressedArrayTexture || flags.isCompressedCubeTexture) return `compressed array and cube textures are not read for GI; ${fallback}`;
    if (flags.isCompressedTexture) {
        if (!compressedFormat(texture)) return `compressed texture format ${texture.format} is not read for GI (BC1-5, BC7, ETC2, EAC and ASTC are); ${fallback}`;
        return null;
    }
    if (flags.isVideoTexture) return `video textures are not read for GI yet; ${fallback}`;
    if (textureSource(texture)) return null;
    const data = texture.image as DataImage | null;
    if (data && ArrayBuffer.isView(data.data)) return `data textures that aren't 8-bit RGBA (float, half-float, one- or two-channel) are not read for GI; ${fallback}`;
    return null;
}

/**
 * The platform texture for a three texture: its pixels ({@link textureSource})
 * and whether they are sRGB-encoded, which follows three: the texture's
 * colour space for images and data textures, the chosen format for
 * compressed ones (a format with no sRGB variant, BC4/BC5/EAC, is read
 * linear as three reads it). Null while GI can't read it.
 */
export function textureDescriptor(texture: Texture): TextureDescriptor | null {
    const source = textureSource(texture);
    if (!source) return null;
    const compressed = (source as CompressedPixelData).levels !== undefined;
    const srgb = compressed ? (source as CompressedPixelData).format.endsWith('-srgb') : hasSrgbTransfer(texture);
    return { source, srgb, label: texture.name || undefined };
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

/** The material properties GI reads, as three's materials name them (all optional). */
interface MaterialFields {
    color?: { r: number; g: number; b: number };
    metalness?: number;
    emissive?: { r: number; g: number; b: number };
    emissiveIntensity?: number;
    transmission?: number;
    alphaHash?: boolean;
    map?: Texture | null;
    emissiveMap?: Texture | null;
    metalnessMap?: Texture | null;
    alphaMap?: Texture | null;
    colorNode?: unknown;
    emissiveNode?: unknown;
    isNodeMaterial?: boolean;
    isShaderMaterial?: boolean;
    isShadowMaterial?: boolean;
}

function fields(material: Material): MaterialFields {
    return material as unknown as MaterialFields;
}

/** The material's metalness, 0..1, for materials that have one (Standard, Physical and their node versions). */
export function metalnessOf(material: Material): number {
    const metalness = fields(material).metalness;
    return typeof metalness === 'number' ? Math.min(Math.max(metalness, 0), 1) : 0;
}

/**
 * How a material's colour is made, as far as GI can tell:
 * - `plain`: from its properties (`color`, `map`, `emissive`...), which GI
 *   reads. Every built-in material, and node materials without a
 *   `colorNode` (an `emissiveNode` alone leaves the albedo plain: GI reads
 *   `color` and `map`, and the emission from `emissive`);
 * - `node graph`: a node material with a `colorNode`, which replaces
 *   `color` × `map` entirely in three's shader (NOTES);
 * - `shader`: a `ShaderMaterial` or `RawShaderMaterial`.
 * GI can't run a graph or a shader on its rays, so for the last two it
 * guesses the albedo ({@link materialDescriptor}).
 */
export type MaterialShading = 'plain' | 'node graph' | 'shader';

export function materialShading(material: Material): MaterialShading {
    const flags = fields(material);
    if (flags.isShaderMaterial) return 'shader';
    if (flags.isNodeMaterial && flags.colorNode != null) return 'node graph';
    return 'plain';
}

/** The grey GI gives a material whose colour comes from a graph or shader and that has no `color` to go by. */
export const CUSTOM_SHADING_ALBEDO: ColorRGB = [0.5, 0.5, 0.5];

/**
 * Whether a custom-shaded material's `color` says something: a node
 * material always has one (white unless set), so only a colour other than
 * white counts; any other material has one only if someone put it there.
 */
function colorIsSet(material: Material, color: { r: number; g: number; b: number } | undefined): boolean {
    if (!color) return false;
    if (!fields(material).isNodeMaterial) return true;
    return color.r !== 1 || color.g !== 1 || color.b !== 1;
}

/**
 * Whether three draws the material so that light passes through it
 * entirely, so GI's rays should too (`blocksLight: false`), before the
 * `giBlocksLight` opt-in:
 * - glass: `transmission` > 0 (tinted glass is later);
 * - a `ShadowMaterial`: an invisible ground that only catches shadows.
 *
 * Hidden materials (`visible = false`) let light through too, whatever the
 * opt-in ({@link materialDescriptor}). Partly see-through surfaces don't
 * belong here: blended ones are `alphaMode: 'stochastic'` and alpha-tested
 * ones `'cutout'`, which block light by their alpha, triangle by triangle
 * ({@link alphaModeOf}).
 */
export function seeThroughForGI(material: Material): boolean {
    const { transmission, isShadowMaterial } = fields(material);
    return (typeof transmission === 'number' && transmission > 0) || isShadowMaterial === true;
}

/**
 * How the material's alpha (`opacity` × the albedo map's alpha) affects GI's
 * rays (DECISIONS 2026-11-09, "M4 answers"):
 * - `alphaTest` > 0 (glTF `MASK`): `cutout` at `alphaTest`;
 * - blended (`transparent` without `alphaTest`, glTF `BLEND`), and three's
 *   `alphaHash`: `stochastic`, a ray passes with probability 1 − alpha;
 * - otherwise `opaque`.
 * `userData.giBlocksLight` makes a blended material opaque for GI.
 */
export function alphaModeOf(material: Material): AlphaMode {
    if (material.alphaTest > 0) return 'cutout';
    if (blocksLightOptIn(material.userData)) return 'opaque';
    if (material.transparent || fields(material).alphaHash === true) return 'stochastic';
    return 'opaque';
}

export interface MaterialConversionOptions {
    /** Where warnings about the material go (custom shading, alpha maps, uv channels). The scene sync's shows each once. */
    readonly warn?: (message: string) => void;
}

/** A material's name for a warning. */
function materialName(material: Material): string {
    return material.name ? `"${material.name}"` : `${material.type} ${material.uuid}`;
}

/**
 * The platform material for a three material (docs/design/m4-materials.md
 * §4). The platform samples the textures at ray hits, so nothing is folded
 * or averaged here.
 *
 * - **albedo:** `color` × `map`. Standard and Physical also send
 *   `metalness` and `metalnessMap` (its blue channel, glTF's
 *   metallic-roughness layout, with its own uv mapping): the platform scales
 *   the albedo by 1 − metalness × map, as three does. Lambert, Phong, Toon
 *   and Basic have no metalness.
 * - **emission:** `emissive` × `emissiveIntensity` × `emissiveMap`. Basic
 *   materials have none.
 * - **custom shading** ({@link materialShading}): a `colorNode` graph, or a
 *   `ShaderMaterial`, can't be read. GI uses `color` if it's set, otherwise
 *   grey {@link CUSTOM_SHADING_ALBEDO}, and `emissive` if it's there, with a
 *   warning naming the material (unless `userData.giAlbedo` or `giAlbedoMap`
 *   says what to use). `map` is left out: three ignores it under a
 *   `colorNode`, so it says nothing about what is drawn; `giAlbedoMap`
 *   brings a texture back.
 * - **overrides** (user-data.ts): `userData.giAlbedo` replaces `color`,
 *   `giAlbedoMap` replaces `map`, `giEmissive` replaces `emissive`.
 * - **alpha** ({@link alphaModeOf}): `alphaMode`, `alphaCutoff` and
 *   `opacity`. A blended material blocks light by its alpha, no longer
 *   all-or-nothing.
 * - **blocks light** (`blocksLight: false`) only for glass and shadow
 *   catchers ({@link seeThroughForGI}) and hidden materials;
 *   `userData.giBlocksLight = true` keeps glass blocking, not a hidden
 *   material.
 * - double-sided whatever three's `side` (back-face hits are how probes find
 *   out they sit inside geometry), unless `material.userData.giSingleSided`.
 * - textures as handles from `textureHandle`, with their uv mappings
 *   ({@link uvMapping}). `alphaMap` isn't read yet (a warning).
 */
export function materialDescriptor(
    material: Material,
    textureHandle: (texture: Texture) => TextureHandle | undefined,
    options: MaterialConversionOptions = {},
): MaterialDescriptor {
    const warn = options.warn ?? (() => {});
    const { color, emissive, emissiveIntensity, metalness, map, emissiveMap, metalnessMap, alphaMap } = fields(material);
    const shading = materialShading(material);
    const userData = material.userData;

    // Albedo: the colour and the map, each replaceable from userData.
    const albedoOverride = giAlbedoOverride(userData, scratchColor);
    const mapOverride = giAlbedoMapOverride(userData);
    // Under custom shading three ignores `map` (a colorNode replaces color × map), so only the override counts there.
    const albedoMap = mapOverride ?? (shading === 'plain' ? (map ?? null) : null);
    let albedo: ColorRGB | undefined;
    if (albedoOverride) {
        albedo = [albedoOverride.r, albedoOverride.g, albedoOverride.b];
    } else if (shading === 'plain') {
        albedo = color ? [color.r, color.g, color.b] : undefined;
    } else {
        // A graph or shader makes the colour: go by color if it's set, otherwise grey (white × giAlbedoMap with an override map).
        const useColor = colorIsSet(material, color);
        albedo = useColor ? [color!.r, color!.g, color!.b] : mapOverride ? [1, 1, 1] : [...CUSTOM_SHADING_ALBEDO];
        if (!mapOverride) {
            const used = useColor ? 'its color' : `grey ${CUSTOM_SHADING_ALBEDO[0]}`;
            const by = shading === 'shader' ? `a ${material.type}` : 'a node graph (colorNode)';
            warn(`material ${materialName(material)} is shaded by ${by}, which GI can't read; GI uses ${used}. Set material.userData.giAlbedo (or giAlbedoMap, giEmissive) to say what GI should see.`);
        }
    }
    const albedoTexture = albedoMap ? textureHandle(albedoMap) : undefined;

    // Metalness: Standard and Physical. The map's blue channel, sampled with its own mapping.
    const hasMetalness = typeof metalness === 'number';
    const metalnessHandle = hasMetalness && metalnessMap ? textureHandle(metalnessMap) : undefined;
    const metalnessTexture: MetalnessTexture | undefined =
        metalnessHandle !== undefined && metalnessMap ? { texture: metalnessHandle, channel: 'b', uv: contractUvMapping(metalnessMap) } : undefined;

    // Emission.
    const emissiveOverride = giEmissiveOverride(userData, scratchColor);
    const emissiveColor = emissiveOverride ? ([emissiveOverride.r, emissiveOverride.g, emissiveOverride.b] as ColorRGB) : emissive ? ([emissive.r, emissive.g, emissive.b] as ColorRGB) : undefined;
    const emissiveTexture = emissiveMap ? textureHandle(emissiveMap) : undefined;

    // Alpha, and whether light passes at all.
    const alphaMode = alphaModeOf(material);
    const blocksLight = material.visible && (blocksLightOptIn(userData) || !seeThroughForGI(material));
    if (alphaMap) warn('alphaMap is not read for GI yet; the alpha GI uses is opacity × the albedo map\'s alpha.');
    for (const texture of [albedoMap, emissiveMap, hasMetalness ? metalnessMap : null]) {
        if (texture && !uvMapping(texture).channelSupported) warn(`GI has two uv sets; a texture on uv channel ${texture.channel} is sampled with channel 0.`);
    }

    return {
        albedo,
        albedoTexture,
        albedoUv: albedoTexture !== undefined && albedoMap ? contractUvMapping(albedoMap) : undefined,
        metalness: hasMetalness ? metalnessOf(material) : undefined,
        metalnessTexture,
        emissive: emissiveColor,
        emissiveIntensity: emissiveColor ? (emissiveIntensity ?? 1) : undefined,
        emissiveTexture,
        emissiveUv: emissiveTexture !== undefined && emissiveMap ? contractUvMapping(emissiveMap) : undefined,
        alphaMode,
        alphaCutoff: alphaMode === 'cutout' ? Math.min(material.alphaTest, 1) : undefined,
        opacity: alphaMode === 'opaque' ? undefined : Math.min(Math.max(material.opacity, 0), 1),
        blocksLight,
        doubleSided: !singleSidedForGI(userData),
        label: material.name || material.type,
    };
}

/** Reused by the userData colour reads, so reading a material allocates nothing for them. */
const scratchColor = new Color();

/** The material's own `envMap`, if it has one. */
export function envMapOf(material: Material): Texture | null {
    const envMap = (material as Material & { envMap?: Texture | null }).envMap;
    return envMap ?? null;
}

/** Whether any of the material's textures samples uv channel 1, so its meshes need `uv1`. */
export function usesUvChannel1(material: Material): boolean {
    const { map, emissiveMap, metalnessMap } = fields(material);
    const mapOverride = giAlbedoMapOverride(material.userData);
    return map?.channel === 1 || emissiveMap?.channel === 1 || metalnessMap?.channel === 1 || mapOverride?.channel === 1;
}

/** Numbers a {@link MaterialSnapshot} keeps per texture (see `readTextureNumbers`). */
const TEXTURE_SNAPSHOT_NUMBERS = 13;
/** Where the texture numbers start in a snapshot; the material's own numbers come first. */
const FIRST_TEXTURE_SNAPSHOT_NUMBER = 23;
/** Textures a snapshot keeps: `map`, `emissiveMap`, `metalnessMap` and `userData.giAlbedoMap`. */
const SNAPSHOT_TEXTURES = 4;

/**
 * Everything {@link materialDescriptor} reads, as numbers and references, so
 * the scene sync can tell each frame whether a material changed without
 * building a descriptor. three bumps `material.version` only on
 * `needsUpdate`, not when `color` or `metalness` are set, so the version
 * alone would miss most edits. Each texture's `version` is in it too, so new
 * pixels (`texture.needsUpdate`) re-send the material and its texture.
 */
export class MaterialSnapshot {
    readonly numbers = new Float64Array(FIRST_TEXTURE_SNAPSHOT_NUMBER + SNAPSHOT_TEXTURES * TEXTURE_SNAPSHOT_NUMBERS);
    map: Texture | null = null;
    emissiveMap: Texture | null = null;
    metalnessMap: Texture | null = null;
    /** `userData.giAlbedoMap`. */
    albedoMapOverride: Texture | null = null;
    /** The material's own environment map, which three lights it with instead of `scene.environment`. */
    envMap: Texture | null = null;

    /** Slots in `numbers`, named so the reads below say what they are. */
    static readonly SLOT = {
        colorR: 0,
        colorG: 1,
        colorB: 2,
        metalness: 3,
        emissiveR: 4,
        emissiveG: 5,
        emissiveB: 6,
        emissiveIntensity: 7,
        alphaTest: 8,
        transparent: 9,
        opacity: 10,
        transmission: 11,
        singleSided: 12,
        visible: 13,
        blocksLightOptIn: 14,
        alphaHash: 15,
        /** {@link materialShading}: 0 plain, 1 node graph, 2 shader. */
        shading: 16,
        albedoOverrideR: 17,
        albedoOverrideG: 18,
        albedoOverrideB: 19,
        emissiveOverrideR: 20,
        emissiveOverrideG: 21,
        emissiveOverrideB: 22,
        /** Then the texture numbers for `map`, `emissiveMap`, `metalnessMap` and `giAlbedoMap`, in that order. */
        mapFirst: FIRST_TEXTURE_SNAPSHOT_NUMBER,
    } as const;

    read(material: Material): void {
        const slot = MaterialSnapshot.SLOT;
        const values = this.numbers;
        const { color, emissive, emissiveIntensity, transmission, map, emissiveMap, metalnessMap, alphaHash } = fields(material);
        const userData = material.userData;
        values[slot.colorR] = color ? color.r : -1;
        values[slot.colorG] = color ? color.g : -1;
        values[slot.colorB] = color ? color.b : -1;
        values[slot.metalness] = metalnessOf(material);
        values[slot.emissiveR] = emissive ? emissive.r : -1;
        values[slot.emissiveG] = emissive ? emissive.g : -1;
        values[slot.emissiveB] = emissive ? emissive.b : -1;
        values[slot.emissiveIntensity] = emissiveIntensity ?? 1;
        values[slot.alphaTest] = material.alphaTest;
        values[slot.transparent] = material.transparent ? 1 : 0;
        values[slot.opacity] = material.opacity;
        values[slot.transmission] = typeof transmission === 'number' ? transmission : 0;
        values[slot.singleSided] = singleSidedForGI(userData) ? 1 : 0;
        values[slot.visible] = material.visible ? 1 : 0;
        values[slot.blocksLightOptIn] = blocksLightOptIn(userData) ? 1 : 0;
        values[slot.alphaHash] = alphaHash === true ? 1 : 0;
        const shading = materialShading(material);
        values[slot.shading] = shading === 'plain' ? 0 : shading === 'node graph' ? 1 : 2;
        readColorNumbers(giAlbedoOverride(userData, scratchColor), values, slot.albedoOverrideR);
        readColorNumbers(giEmissiveOverride(userData, scratchColor), values, slot.emissiveOverrideR);
        this.map = map ?? null;
        this.emissiveMap = emissiveMap ?? null;
        this.metalnessMap = metalnessMap ?? null;
        this.albedoMapOverride = giAlbedoMapOverride(userData);
        this.envMap = envMapOf(material);
        readTextureNumbers(this.map, values, slot.mapFirst);
        readTextureNumbers(this.emissiveMap, values, slot.mapFirst + TEXTURE_SNAPSHOT_NUMBERS);
        readTextureNumbers(this.metalnessMap, values, slot.mapFirst + 2 * TEXTURE_SNAPSHOT_NUMBERS);
        readTextureNumbers(this.albedoMapOverride, values, slot.mapFirst + 3 * TEXTURE_SNAPSHOT_NUMBERS);
    }

    equals(other: MaterialSnapshot): boolean {
        if (
            this.map !== other.map ||
            this.emissiveMap !== other.emissiveMap ||
            this.metalnessMap !== other.metalnessMap ||
            this.albedoMapOverride !== other.albedoMapOverride ||
            this.envMap !== other.envMap
        ) {
            return false;
        }
        const a = this.numbers;
        const b = other.numbers;
        for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return false;
        return true;
    }
}

/** A userData colour override's three numbers, or −1s when there is none. */
function readColorNumbers(color: Color | null, target: Float64Array, first: number): void {
    target[first] = color ? color.r : -1;
    target[first + 1] = color ? color.g : -1;
    target[first + 2] = color ? color.b : -1;
}

/**
 * A texture's version, channel, wraps, `flipY` and the inputs of its uv
 * matrix (or the matrix itself when three doesn't update it), so a change
 * to any of them is seen without recomputing the matrix every frame.
 */
function readTextureNumbers(texture: Texture | null, target: Float64Array, first: number): void {
    if (!texture) {
        target.fill(0, first, first + TEXTURE_SNAPSHOT_NUMBERS);
        return;
    }
    target[first] = texture.version;
    target[first + 1] = texture.channel;
    target[first + 2] = texture.wrapS;
    target[first + 3] = texture.wrapT;
    if (texture.matrixAutoUpdate) {
        target[first + 4] = 1;
        target[first + 5] = texture.offset.x;
        target[first + 6] = texture.offset.y;
        target[first + 7] = texture.repeat.x;
        target[first + 8] = texture.repeat.y;
        target[first + 9] = texture.rotation;
        target[first + 10] = texture.center.x;
        target[first + 11] = texture.center.y;
    } else {
        const e = texture.matrix.elements;
        target[first + 4] = 0;
        target[first + 5] = e[0];
        target[first + 6] = e[1];
        target[first + 7] = e[3];
        target[first + 8] = e[4];
        target[first + 9] = e[6];
        target[first + 10] = e[7];
        target[first + 11] = 0;
    }
    target[first + 12] = texture.flipY ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Lights
// ---------------------------------------------------------------------------

/** three's physically based lights fall off with distance squared; the platform assumes it. */
export const PHYSICAL_DECAY = 2;

/** A light's world position from its `matrixWorld` (which must be current). */
function worldPosition(light: Light): [number, number, number] {
    const e = light.matrixWorld.elements;
    return [e[12], e[13], e[14]];
}

/**
 * The direction from a light to its target, normalized: the direction the
 * light travels (contract: "Direction the light travels"). For a degenerate
 * target (on the light), straight down, as three's default target would give.
 */
function directionToTarget(light: DirectionalLight | SpotLight): Vec3 {
    const from = light.matrixWorld.elements;
    const to = light.target.matrixWorld.elements;
    const x = to[12] - from[12];
    const y = to[13] - from[13];
    const z = to[14] - from[14];
    const length = Math.hypot(x, y, z);
    if (!(length > 0)) return [0, -1, 0];
    return [x / length, y / length, z / length];
}

/** Largest cone the platform accepts (contract: outer ≤ π/2); three's own maximum for `SpotLight.angle`. */
const MAX_SPOT_ANGLE = Math.PI / 2;

/**
 * The platform light for a three directional, point or spot light, or null
 * for other kinds. World matrices (the light's and its target's) must be
 * current.
 *
 * - colour and intensity pass through (same units);
 * - directional: direction = target − position (the way the light travels);
 * - point and spot: `distance` becomes `range`, so `distance = 0` (no limit)
 *   arrives as `range: 0`;
 * - spot: `angle` is the outer cone; three fades over the outer `penumbra`
 *   share of it, so the inner cone is `angle × (1 − penumbra)`;
 * - `castShadow` becomes `castsShadows`, and `userData.giLightGroup` the group.
 *
 * `intensityScale` multiplies the intensity (the sync passes 0 for hidden lights).
 */
export function lightDescriptor(light: Light, intensityScale = 1): LightDescriptor | null {
    const common = {
        color: [light.color.r, light.color.g, light.color.b] as ColorRGB,
        intensity: light.intensity * intensityScale,
        castsShadows: castsShadowsForGI(light),
        group: lightGroup(light),
        label: light.name || light.type,
    };
    if ((light as DirectionalLight).isDirectionalLight) {
        return { ...common, type: 'directional', direction: directionToTarget(light as DirectionalLight) };
    }
    if ((light as SpotLight).isSpotLight) {
        const spot = light as SpotLight;
        const outer = Math.min(Math.max(spot.angle, 0), MAX_SPOT_ANGLE);
        const penumbra = Math.min(Math.max(spot.penumbra, 0), 1);
        return {
            ...common,
            type: 'spot',
            position: worldPosition(spot),
            direction: directionToTarget(spot),
            range: Math.max(spot.distance, 0),
            outerConeAngle: outer,
            innerConeAngle: outer * (1 - penumbra),
        };
    }
    if ((light as PointLight).isPointLight) {
        const point = light as PointLight;
        return { ...common, type: 'point', position: worldPosition(point), range: Math.max(point.distance, 0) };
    }
    return null;
}

/** Slots of {@link readLightNumbers}. */
const LIGHT_SLOT = { colorR: 0, colorG: 1, colorB: 2, intensity: 3, castShadow: 4, group: 5, positionX: 6, positionY: 7, positionZ: 8, targetX: 9, targetY: 10, targetZ: 11, distance: 12, angle: 13, penumbra: 14 } as const;
export const LIGHT_NUMBER_COUNT = 15;

/**
 * Everything {@link lightDescriptor} reads, as numbers in `target`, so a
 * light can be compared with its last-sent state every frame without
 * building a descriptor (or allocating). World matrices must be current.
 */
export function readLightNumbers(light: Light, intensityScale: number, target: Float64Array): void {
    const position = light.matrixWorld.elements;
    const lightTarget = (light as DirectionalLight | SpotLight).target;
    const targetPosition = lightTarget ? lightTarget.matrixWorld.elements : position;
    const ranged = light as PointLight & Partial<Pick<SpotLight, 'angle' | 'penumbra'>>;
    target[LIGHT_SLOT.colorR] = light.color.r;
    target[LIGHT_SLOT.colorG] = light.color.g;
    target[LIGHT_SLOT.colorB] = light.color.b;
    target[LIGHT_SLOT.intensity] = light.intensity * intensityScale;
    target[LIGHT_SLOT.castShadow] = castsShadowsForGI(light) ? 1 : 0;
    target[LIGHT_SLOT.group] = lightGroup(light);
    target[LIGHT_SLOT.positionX] = position[12];
    target[LIGHT_SLOT.positionY] = position[13];
    target[LIGHT_SLOT.positionZ] = position[14];
    target[LIGHT_SLOT.targetX] = targetPosition[12];
    target[LIGHT_SLOT.targetY] = targetPosition[13];
    target[LIGHT_SLOT.targetZ] = targetPosition[14];
    target[LIGHT_SLOT.distance] = ranged.distance ?? 0;
    target[LIGHT_SLOT.angle] = ranged.angle ?? 0;
    target[LIGHT_SLOT.penumbra] = ranged.penumbra ?? 0;
}

/** Rec. 709 luminance of a linear colour. */
export function luminance(color: { r: number; g: number; b: number }): number {
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}
