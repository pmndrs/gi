/**
 * Content fingerprints of the scene (SPEC, "content fingerprints for saved
 * bakes"): short hashes of what changes the GI answer, so a saved field can
 * tell whether it was baked for this scene. Engine-neutral and CPU-only.
 *
 * What goes in, per part of {@link SceneFingerprint}:
 * - **geometry:** every instance that contributes to GI: its mesh's content
 *   (positions, indices, normals, uv sets, groups, triangle count and box),
 *   its transform and whether it is visible. Hashed when the mesh is created
 *   or updated, because the scene keeps no CPU copy of the geometry after
 *   upload. A deforming mesh counts with its rest pose, not its current pose.
 * - **materials:** each GI instance's material list, every material by its
 *   numbers (colours, metalness, opacity, alpha mode and cut-off, emission,
 *   uv mappings, `blocksLight`, `doubleSided`) and its textures by their
 *   content hash. Labels are left out.
 * - **textures:** the content hash of every texture a GI instance's
 *   materials use. Sources that carry bytes (`PixelData`, `ImageData`,
 *   compressed levels) are hashed from their bytes (a texture over
 *   {@link FULL_HASH_WORDS} words is hashed from an even sample of that many
 *   words, plus its length). Sources the CPU can't read without drawing them
 *   (`ImageBitmap`, canvases, a `GPUTexture`) count by their size, format
 *   and colour space only, and image elements by their URL and size: the
 *   fingerprint misses a repaint that keeps those.
 * - **lights:** every light's numbers, its shadow parameters (`castsShadows`)
 *   and its group; labels left out. The platform has no shadow matrices, so
 *   none are hashed (SPEC: parameters, not matrices).
 * - **sky:** the environment's settings. A `texture` environment counts by
 *   its texture's size and format, layout, intensity and rotation, not its
 *   texels (they are on the GPU).
 *
 * Each part is order-free: instances, lights and textures are hashed one by
 * one and their hashes sorted, so two loads of the same scene that create
 * objects in another order (textures load asynchronously) agree.
 *
 * Bump {@link FINGERPRINT_VERSION} whenever what a part covers or how it is
 * hashed changes: fingerprints of different versions are never compared.
 */

import type { SceneFingerprint } from '../contract/scene-copy.ts';
import type { CompressedPixelData, EnvironmentDescriptor, LightDescriptor, MaterialDescriptor, MeshDescriptor, PixelData, TextureSource } from '../contract/scene.ts';

/** The fingerprint's own version (the SPEC's "cache version"). 1: T5.2's parts, as the file comment lists them. */
export const FINGERPRINT_VERSION = 1;

/**
 * A texture's bytes are hashed in full up to this many 32-bit words (4 MB,
 * a 2048 x 2048 BC7 image), and past it from an even sample of this many.
 * Why a cap: Bistro's textures are hundreds of MB, and a texture edited
 * anywhere almost always changes many sampled words.
 */
export const FULL_HASH_WORDS = 1 << 20;

/**
 * A 64-bit hash in two independent 32-bit lanes (multiply-xorshift mixing,
 * as murmur3's finaliser), written as 16 hex digits. Not cryptographic: it
 * only has to tell a changed scene from the same one, quickly. A few ns a
 * word, so a 50 MB scene's geometry hashes in tens of milliseconds.
 */
export class ContentHasher {
    private lane1 = 0x9e3779b9;
    private lane2 = 0x85ebca6b;
    private count = 0;
    private readonly scratch = new DataView(new ArrayBuffer(8));

    /** Adds one 32-bit word. */
    word(value: number): this {
        // Two different multipliers and rotations, so the lanes don't collide together.
        this.lane1 = Math.imul(rotateLeft(this.lane1 ^ value, 13), 0xcc9e2d51) >>> 0;
        this.lane2 = Math.imul(rotateLeft(this.lane2 ^ value, 17), 0x1b873593) >>> 0;
        this.lane1 = (this.lane1 + this.lane2) >>> 0;
        this.count++;
        return this;
    }

    /** Adds every word of `words` (or an even sample of `limit` of them, plus the length). */
    words(words: Uint32Array, limit = Number.POSITIVE_INFINITY): this {
        this.word(words.length);
        const step = words.length > limit ? Math.ceil(words.length / limit) : 1;
        for (let index = 0; index < words.length; index += step) this.word(words[index]);
        return this;
    }

    /** Adds any typed array's bytes (sampled past `limit` words, as {@link words}). */
    bytes(view: ArrayBufferView, limit = Number.POSITIVE_INFINITY): this {
        const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
        const wholeWords = Math.floor(bytes.byteLength / 4);
        // A Uint32Array needs a 4-byte-aligned start; copy the rare unaligned view.
        const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
        this.word(bytes.byteLength);
        this.words(new Uint32Array(aligned.buffer, aligned.byteOffset, wholeWords), limit);
        for (let index = wholeWords * 4; index < bytes.byteLength; index++) this.word(bytes[index]);
        return this;
    }

    /** Adds a number as its 64-bit float bits (so 0.1 and 0.10000001 differ, and -0 and 0 do too). */
    number(value: number): this {
        this.scratch.setFloat64(0, value, true);
        return this.word(this.scratch.getUint32(0, true)).word(this.scratch.getUint32(4, true));
    }

    /** Adds a string's UTF-16 code units, after its length. */
    string(text: string): this {
        this.word(text.length);
        for (let index = 0; index < text.length; index++) this.word(text.charCodeAt(index));
        return this;
    }

    /** The hash so far, as 16 hex digits (lane 1, then lane 2, each finalised with the word count). */
    digest(): string {
        return hex32(finalise(this.lane1 ^ this.count)) + hex32(finalise(this.lane2 ^ Math.imul(this.count, 0x27d4eb2d)));
    }
}

function rotateLeft(value: number, bits: number): number {
    return ((value << bits) | (value >>> (32 - bits))) >>> 0;
}

/** murmur3's 32-bit finaliser: every input bit reaches every output bit. */
function finalise(value: number): number {
    let mixed = value >>> 0;
    mixed = Math.imul(mixed ^ (mixed >>> 16), 0x85ebca6b) >>> 0;
    mixed = Math.imul(mixed ^ (mixed >>> 13), 0xc2b2ae35) >>> 0;
    return (mixed ^ (mixed >>> 16)) >>> 0;
}

function hex32(value: number): string {
    return value.toString(16).padStart(8, '0');
}

/** The hash of a list of hashes, whatever their order. */
export function unorderedDigest(digests: readonly string[]): string {
    const hasher = new ContentHasher();
    hasher.word(digests.length);
    for (const digest of [...digests].sort()) hasher.string(digest);
    return hasher.digest();
}

/**
 * A plain value as text with its object keys sorted, so two descriptors with
 * the same numbers give the same text however they were built. Numbers keep
 * every digit (JSON writes the shortest text that reads back the same double).
 */
export function stableText(value: unknown): string {
    if (value === undefined) return 'undefined';
    if (value === null || typeof value !== 'object') return JSON.stringify(value);
    if (Array.isArray(value) || ArrayBuffer.isView(value)) return `[${Array.from(value as ArrayLike<unknown>, stableText).join(',')}]`;
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record)
        .filter((key) => record[key] !== undefined)
        .sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${stableText(record[key])}`).join(',')}}`;
}

/** A mesh's content hash (geometry part): every array GI reads, its groups and its declared deform box. */
export function meshContentHash(descriptor: MeshDescriptor): string {
    const hasher = new ContentHasher();
    hasher.bytes(descriptor.positions);
    // The index width is part of the content: Uint16 and Uint32 indices of the same numbers hash apart (a harmless miss).
    hasher.bytes(descriptor.indices);
    for (const optional of [descriptor.normals, descriptor.uvs, descriptor.uvs1]) {
        if (optional) hasher.bytes(optional);
        else hasher.word(0);
    }
    hasher.string(stableText({ groups: descriptor.groups ?? null, deforming: descriptor.deforming === true, deformBounds: descriptor.deformBounds ?? null }));
    return hasher.digest();
}

/** A texture's content hash (textures part; the file comment says what each kind of source contributes). */
export function textureContentHash(source: TextureSource, srgb: boolean): string {
    const hasher = new ContentHasher();
    hasher.word(srgb ? 1 : 0);
    const compressed = source as CompressedPixelData;
    const pixels = source as PixelData;
    if (Array.isArray(compressed.levels) && typeof compressed.format === 'string') {
        hasher.string('compressed').string(compressed.format);
        // The levels share the sample budget, so a long mip chain costs no more than one big image.
        const perLevel = Math.max(1, Math.floor(FULL_HASH_WORDS / Math.max(1, compressed.levels.length)));
        for (const level of compressed.levels) hasher.word(level.width).word(level.height).bytes(level.data, perLevel);
    } else if (ArrayBuffer.isView(pixels.data) && Number.isFinite(pixels.width) && Number.isFinite(pixels.height)) {
        // PixelData and ImageData: 8-bit RGBA rows.
        hasher.string('pixels').word(pixels.width).word(pixels.height).bytes(pixels.data, FULL_HASH_WORDS);
    } else if (typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement) {
        hasher.string('image').string(source.currentSrc || source.src).word(source.naturalWidth).word(source.naturalHeight);
    } else {
        // ImageBitmap, canvases, a GPUTexture: no bytes the CPU can read cheaply.
        const sized = source as { width?: number; height?: number; format?: string; depthOrArrayLayers?: number };
        hasher.string('unread').word(sized.width ?? 0).word(sized.height ?? 0).word(sized.depthOrArrayLayers ?? 1).string(sized.format ?? '');
    }
    return hasher.digest();
}

/**
 * A material's hash (materials part): its numbers as given, with each
 * texture replaced by that texture's content hash. `textureHash` maps a
 * texture handle to it (null for an unknown handle).
 */
export function materialHash(descriptor: MaterialDescriptor, textureHash: (handle: number) => string | null): string {
    const { albedoTexture, emissiveTexture, metalnessTexture } = descriptor;
    return new ContentHasher()
        .string(
            stableText({
                // stableText leaves out undefined keys, so the label counts for nothing.
                ...descriptor,
                label: undefined,
                albedoTexture: albedoTexture === undefined ? null : textureHash(albedoTexture),
                emissiveTexture: emissiveTexture === undefined ? null : textureHash(emissiveTexture),
                metalnessTexture: metalnessTexture === undefined ? null : { ...metalnessTexture, texture: textureHash(metalnessTexture.texture) },
            }),
        )
        .digest();
}

/** A light's hash (lights part): every number and its shadow parameter, not its label. */
export function lightHash(descriptor: LightDescriptor): string {
    return new ContentHasher().string(stableText({ ...descriptor, label: undefined })).digest();
}

/** The sky's hash (sky part). A texture environment counts by its texture's shape and format, not its texels. */
export function environmentHash(environment: EnvironmentDescriptor): string {
    if (environment.type !== 'texture') return new ContentHasher().string(stableText(environment)).digest();
    const { texture, ...settings } = environment;
    const shape = { width: texture.width, height: texture.height, layers: texture.depthOrArrayLayers, format: texture.format };
    return new ContentHasher().string(stableText({ ...settings, texture: shape })).digest();
}

/** The parts of two fingerprints that differ, in the order they are listed; every part when the versions differ. */
export function fingerprintDifferences(saved: SceneFingerprint, current: SceneFingerprint): (keyof Omit<SceneFingerprint, 'version'>)[] {
    const parts = ['geometry', 'materials', 'textures', 'lights', 'sky'] as const;
    if (saved.version !== current.version) return [...parts];
    return parts.filter((part) => saved[part] !== current[part]);
}
