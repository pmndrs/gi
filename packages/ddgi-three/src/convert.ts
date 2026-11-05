/**
 * three.js objects to the platform's plain scene descriptors (`SceneApi`,
 * packages/ddgi-core/src/platform/contract/scene.ts).
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
    MirroredRepeatWrapping,
    SRGBColorSpace,
    type BufferAttribute,
    type BufferGeometry,
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
    ColorRGB,
    LightDescriptor,
    MaterialDescriptor,
    MeshDescriptor,
    MeshGroup,
    PixelData,
    TextureHandle,
    TextureSource,
    UvMapping,
    UvWrap,
    Vec3,
} from '@pmndrs/ddgi-core';
import { blocksLightOptIn, lightGroup, singleSidedForGI } from './user-data.ts';

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

/** three's wrap modes as the platform's. */
export function uvWrap(wrap: Wrapping): UvWrap {
    if (wrap === ClampToEdgeWrapping) return 'clamp';
    if (wrap === MirroredRepeatWrapping) return 'mirror';
    return 'repeat';
}

/**
 * Where a texture is sampled: three's `texture.matrix` (offset, repeat,
 * rotation, centre) and `texture.channel`, passed straight through as the
 * contract says: `[e[0], e[1], e[3], e[4], e[6], e[7]]` of the 3x3, after
 * `updateMatrix()` when three would update it itself.
 *
 * The platform has two uv sets; a texture on channel 2 or 3 falls back to
 * channel 0, and `channelSupported` says so, so the caller can warn.
 */
export function uvMapping(texture: Texture): UvMapping & { readonly channelSupported: boolean } {
    if (texture.matrixAutoUpdate) texture.updateMatrix();
    const e = texture.matrix.elements;
    const channelSupported = texture.channel === 0 || texture.channel === 1;
    return {
        channel: channelSupported ? (texture.channel as 0 | 1) : 0,
        matrix: [e[0], e[1], e[3], e[4], e[6], e[7]],
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

interface DataImage {
    readonly data: ArrayBufferView | null;
    readonly width: number;
    readonly height: number;
}

/**
 * The texture's pixels in a form `createTexture` accepts, or null when there
 * are none yet (an image still loading) or they can't be read on the CPU:
 * compressed textures (docs/api/platform.md gap 5, M4), render-target
 * textures, and data textures that aren't 8-bit RGBA.
 */
export function textureSource(texture: Texture): TextureSource | null {
    const flags = texture as Texture & { isCompressedTexture?: boolean; isRenderTargetTexture?: boolean };
    if (flags.isCompressedTexture || flags.isRenderTargetTexture) return null;
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

export function isSrgbTexture(texture: Texture): boolean {
    return texture.colorSpace === SRGBColorSpace;
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

/**
 * Below this `opacity`, a `transparent` material that is alpha-tested
 * (`alphaTest` > 0) still lets light through for GI (`blocksLight: false`):
 * a mostly see-through cutout would otherwise cast full shadows and bounce
 * like a wall. (Alpha-blended materials let light through at any opacity.)
 */
export const SEE_THROUGH_OPACITY = 0.5;

/**
 * The alpha cutoff given to an alpha-blended material with an albedo texture
 * that opted in to blocking light (`userData.giBlocksLight`): foliage and
 * decals drawn blended then block light where they are mostly opaque and let
 * it through where they are mostly not (once the core samples textures, M4).
 */
export const BLENDED_ALPHA_CUTOFF = 0.5;

/** The material properties GI reads, as three's materials name them (all optional). */
interface MaterialFields {
    color?: { r: number; g: number; b: number };
    metalness?: number;
    emissive?: { r: number; g: number; b: number };
    emissiveIntensity?: number;
    transmission?: number;
    map?: Texture | null;
    emissiveMap?: Texture | null;
    metalnessMap?: Texture | null;
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
 * GI albedo: base colour × (1 − metalness).
 *
 * Why: a metal reflects light specularly, tinted, and has no diffuse
 * reflection, and DDGI bounces diffuse light only. Without this, a metal
 * would bounce light like paint (docs/api/platform.md, "Metals bounce no
 * diffuse light"). A metalness *texture* is folded in when the atlas is built
 * (M4); here only the scalar.
 */
export function giAlbedo(material: Material): ColorRGB | undefined {
    const color = fields(material).color;
    if (!color) return undefined;
    const diffuse = 1 - metalnessOf(material);
    return [color.r * diffuse, color.g * diffuse, color.b * diffuse];
}

/**
 * The platform material for a three material.
 *
 * - albedo: `color` × (1 − `metalness`) ({@link giAlbedo}); materials without
 *   a `color` keep the platform's default;
 * - emission: `emissive` and `emissiveIntensity`, passed separately (the
 *   platform multiplies them);
 * - cutout: `alphaTest` > 0 becomes `alphaCutoff`;
 * - light passes through (`blocksLight: false`) for `transmission` > 0
 *   (glass), for alpha-blended materials (`transparent` and no `alphaTest`;
 *   glTF's `alphaMode: BLEND` loads that way), for alpha-tested `transparent`
 *   ones with `opacity` below {@link SEE_THROUGH_OPACITY}, and for materials
 *   with `visible = false`. `userData.giBlocksLight = true` keeps a
 *   see-through material blocking (user-data.ts), but not a hidden one;
 * - double-sided whatever three's `side` (back-face hits are how probes find
 *   out they sit inside geometry), unless `material.userData.giSingleSided`;
 * - textures (`map`, `emissiveMap`) as handles from `textureHandle`, with
 *   their uv mappings. The atlas that samples them is M4; until then each
 *   the albedo is multiplied by the average of albedo map × (1 − metalness ×
 *   metalness map), and the emission by the emissive map's average
 *   (texture-average.ts, `averages`).
 *
 * Node materials' custom nodes (`colorNode`, `emissiveNode`) are not read.
 */
/**
 * Texture averages folded into GI while the core samples no textures
 * (texture-average.ts; remove in M4). The scene sync supplies cached ones.
 */
export interface TextureAverages {
    /** A texture's average linear colour (emissive maps). */
    color(texture: Texture): ColorRGB | undefined;
    /** The GI albedo factor from an albedo map and/or a metalness map (texture-average.ts, `diffuseAverage`). */
    diffuse(map: Texture | null, metalnessMap: Texture | null, metalness: number, sameUv: boolean): ColorRGB | undefined;
}

/** Whether two textures are sampled at the same uv: the same channel and the same uv transform. */
export function sameUvMapping(a: Texture, b: Texture): boolean {
    const first = uvMapping(a);
    const second = uvMapping(b);
    if (first.channel !== second.channel) return false;
    for (let index = 0; index < 6; index++) if (first.matrix![index] !== second.matrix![index]) return false;
    return true;
}

/**
 * Whether three draws the material see-through, so GI should let light pass
 * (before the `giBlocksLight` opt-in):
 * - glass: `transmission` > 0;
 * - alpha-blended: `transparent` and not alpha-tested. Decals, glass panes and
 *   fading objects are drawn this way; tracing them as opaque turns a decal
 *   with no diffuse colour into a black sheet (Sponza's `dirt_decal`);
 * - alpha-tested and `transparent` but mostly see-through: `opacity` below
 *   {@link SEE_THROUGH_OPACITY}.
 */
function seeThroughForGI(material: Material, alphaTested: boolean, transmission: number | undefined): boolean {
    if (typeof transmission === 'number' && transmission > 0) return true;
    if (!material.transparent) return false;
    return !alphaTested || material.opacity < SEE_THROUGH_OPACITY;
}

export function materialDescriptor(material: Material, textureHandle: (texture: Texture) => TextureHandle | undefined, averages?: TextureAverages): MaterialDescriptor {
    const { color, emissive, emissiveIntensity, transmission, map, emissiveMap, metalnessMap } = fields(material);
    const albedoTexture = map ? textureHandle(map) : undefined;
    const emissiveTexture = emissiveMap ? textureHandle(emissiveMap) : undefined;

    const alphaTested = material.alphaTest > 0;
    let alphaCutoff: number | undefined = alphaTested ? Math.min(material.alphaTest, 1) : undefined;
    const blocksLight = material.visible && (blocksLightOptIn(material.userData) || !seeThroughForGI(material, alphaTested, transmission));
    // An opted-in blended material with a texture: cut it at half alpha, so its clear parts still let light through (M4).
    if (blocksLight && material.transparent && !alphaTested && albedoTexture !== undefined) alphaCutoff = BLENDED_ALPHA_CUTOFF;

    // Textures folded in by their averages while the core samples none (texture-average.ts; remove in M4).
    // The albedo map and the metalness map (three: metalness × metalnessMap.b) together give what
    // multiplies the colour; without either, the scalar metalness alone (giAlbedo).
    let albedo = giAlbedo(material);
    const metalnessTexture = metalnessMap && typeof (material as MaterialFields).metalness === 'number' ? metalnessMap : null;
    const factor =
        averages && (map || metalnessTexture)
            ? averages.diffuse(map ?? null, metalnessTexture, metalnessOf(material), map && metalnessTexture ? sameUvMapping(map, metalnessTexture) : false)
            : undefined;
    if (factor) albedo = color ? [color.r * factor[0], color.g * factor[1], color.b * factor[2]] : factor;
    let emission: ColorRGB | undefined = emissive ? [emissive.r, emissive.g, emissive.b] : undefined;
    const emissiveColor = emissiveMap && averages ? averages.color(emissiveMap) : undefined;
    if (emission && emissiveColor) emission = [emission[0] * emissiveColor[0], emission[1] * emissiveColor[1], emission[2] * emissiveColor[2]];

    return {
        albedo,
        albedoTexture,
        albedoUv: albedoTexture !== undefined && map ? contractUvMapping(map) : undefined,
        emissive: emission,
        emissiveIntensity: emissive ? (emissiveIntensity ?? 1) : undefined,
        emissiveTexture,
        emissiveUv: emissiveTexture !== undefined && emissiveMap ? contractUvMapping(emissiveMap) : undefined,
        alphaCutoff,
        blocksLight,
        doubleSided: !singleSidedForGI(material.userData),
        label: material.name || material.type,
    };
}

/** The material's own `envMap`, if it has one. */
export function envMapOf(material: Material): Texture | null {
    const envMap = (material as Material & { envMap?: Texture | null }).envMap;
    return envMap ?? null;
}

/** Whether any of the material's textures samples uv channel 1, so its meshes need `uv1`. */
export function usesUvChannel1(material: Material): boolean {
    const { map, emissiveMap } = fields(material);
    return map?.channel === 1 || emissiveMap?.channel === 1;
}

/**
 * Everything {@link materialDescriptor} reads, as numbers and references, so
 * the scene sync can tell each frame whether a material changed without
 * building a descriptor. three bumps `material.version` only on
 * `needsUpdate`, not when `color` or `metalness` are set, so the version
 * alone would miss most edits.
 */
/** Numbers a {@link MaterialSnapshot} keeps per texture (see `readTextureNumbers`). */
const TEXTURE_SNAPSHOT_NUMBERS = 12;
/** Where the texture numbers start in a snapshot; the material's own numbers come first. */
const FIRST_TEXTURE_SNAPSHOT_NUMBER = 15;

export class MaterialSnapshot {
    readonly numbers = new Float64Array(FIRST_TEXTURE_SNAPSHOT_NUMBER + 3 * TEXTURE_SNAPSHOT_NUMBERS);
    map: Texture | null = null;
    emissiveMap: Texture | null = null;
    metalnessMap: Texture | null = null;
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
        /** Then the texture numbers for `map`, then the same for `emissiveMap`. */
        mapFirst: FIRST_TEXTURE_SNAPSHOT_NUMBER,
    } as const;

    read(material: Material): void {
        const slot = MaterialSnapshot.SLOT;
        const values = this.numbers;
        const { color, emissive, emissiveIntensity, transmission, map, emissiveMap } = fields(material);
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
        values[slot.singleSided] = singleSidedForGI(material.userData) ? 1 : 0;
        values[slot.visible] = material.visible ? 1 : 0;
        values[slot.blocksLightOptIn] = blocksLightOptIn(material.userData) ? 1 : 0;
        this.map = map ?? null;
        this.emissiveMap = emissiveMap ?? null;
        this.metalnessMap = fields(material).metalnessMap ?? null;
        this.envMap = envMapOf(material);
        readTextureNumbers(this.map, values, slot.mapFirst);
        readTextureNumbers(this.emissiveMap, values, slot.mapFirst + TEXTURE_SNAPSHOT_NUMBERS);
        readTextureNumbers(this.metalnessMap, values, slot.mapFirst + 2 * TEXTURE_SNAPSHOT_NUMBERS);
    }

    equals(other: MaterialSnapshot): boolean {
        if (this.map !== other.map || this.emissiveMap !== other.emissiveMap || this.metalnessMap !== other.metalnessMap || this.envMap !== other.envMap) return false;
        const a = this.numbers;
        const b = other.numbers;
        for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return false;
        return true;
    }
}

/**
 * A texture's version, channel, wraps and the inputs of its uv matrix (or the
 * matrix itself when three doesn't update it), so a change to any of them is
 * seen without recomputing the matrix every frame.
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
        castsShadows: light.castShadow,
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
    target[LIGHT_SLOT.castShadow] = light.castShadow ? 1 : 0;
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
