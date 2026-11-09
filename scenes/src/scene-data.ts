/**
 * Engine-free scene data produced by the procedural builders.
 *
 * The shapes deliberately mirror the platform's scene contract
 * (`packages/ddgi-core/src/platform/contract/scene.ts`) so a procedural scene
 * can be handed to the platform later without translation: positions and
 * indices for meshes, a small material model (albedo, emissive, alpha cutoff,
 * double-sided), instances with column-major transforms, and lights in the
 * three.js physical units.
 *
 * Differences from the contract, on purpose:
 * - Meshes, materials and instances refer to each other by array index, not by
 *   GPU handle, because nothing here has been uploaded anywhere.
 * - Meshes carry normals, which the platform derives itself but glTF and
 *   three.js want.
 * - `emissive` is the final emitted radiance (colour times intensity) as one
 *   linear RGB triple.
 * - Textures are pixels made in code (`ProceduralScene.textures`), and a
 *   material's texture says where it is sampled in glTF / three.js terms
 *   (offset, repeat, rotation, wrap, uv set) rather than as the contract's
 *   3×2 matrix; `uvMappingMatrix` turns one into the other.
 *
 * Everything is metres, right-handed, Y up, linear-light colours.
 */

/** `[x, y, z]`. */
export type Vector3Tuple = readonly [number, number, number];

/** Linear-light RGB, not sRGB-encoded. */
export type LinearColor = readonly [number, number, number];

/** A run of triangles that uses one material slot (the contract's `MeshGroup`). */
export interface SceneMeshGroup {
    /** First index (not triangle) of the run. */
    readonly start: number;
    /** Number of indices in the run; a multiple of 3. */
    readonly count: number;
    /** Which entry of the instance's `materials` the run uses. */
    readonly materialSlot: number;
}

export interface SceneMeshData {
    readonly name: string;
    /** xyz per vertex, local space. */
    readonly positions: Float32Array;
    /** Unit normals, one per vertex. */
    readonly normals: Float32Array;
    /** uv per vertex (uv set 0). */
    readonly uvs: Float32Array;
    /** A second uv set (uv set 1, glTF `TEXCOORD_1`, three's `uv1`), for textures with `channel: 1`. */
    readonly uvs1?: Float32Array;
    /** Triangle list. */
    readonly indices: Uint32Array;
    /**
     * Material runs, for a mesh with more than one material (a wall with a
     * glass pane). Omit for one run of every triangle in slot 0.
     */
    readonly groups?: readonly SceneMeshGroup[];
}

/**
 * Pixels made in code: 8-bit RGBA, rows of `width` texels. Row 0 is v = 0,
 * which is the top row of the image in glTF terms (glTF's uv origin is the
 * image's top-left corner) and three's layout for a `DataTexture` with
 * `flipY` false, so the same bytes mean the same thing in both.
 */
export interface SceneTextureData {
    readonly name: string;
    readonly width: number;
    readonly height: number;
    /** `width * height * 4` bytes, RGBA, alpha not premultiplied. */
    readonly data: Uint8Array;
    /** The colour channels are sRGB-encoded (true for albedo, as glTF requires). Alpha is always linear. */
    readonly srgb: boolean;
}

/** How a texture wraps outside 0..1 on one axis (the contract's `UvWrap`). `mirror` is three's `MirroredRepeatWrapping`. */
export type SceneUvWrap = 'repeat' | 'clamp' | 'mirror';

/**
 * Where a texture is sampled, as glTF's `KHR_texture_transform` and three's
 * texture fields say it:
 *
 *   uv' = offset + R(rotation) · (repeat ⊙ uv)
 *
 * about the uv origin (three's `center` stays (0, 0)). Rotation is in
 * radians and means exactly three's `Texture.rotation`, which is also the
 * number glTF stores (three's GLTFLoader and GLTFExporter copy it unchanged;
 * `uvMappingMatrix` has the formula). glTF scales before rotating and three
 * scales after, which agree only when `repeat` is the same on both axes, so
 * a rotated mapping must use a uniform repeat (`uvMappingMatrix` checks).
 */
export interface SceneUvMapping {
    /** Which uv set: 0 = `uvs`, 1 = `uvs1`. Default 0. */
    readonly channel?: 0 | 1;
    /** Default [0, 0]. */
    readonly offset?: readonly [number, number];
    /** Default [1, 1]. */
    readonly repeat?: readonly [number, number];
    /** Radians. Default 0. */
    readonly rotation?: number;
    /** Default `repeat`. */
    readonly wrapU?: SceneUvWrap;
    /** Default `repeat`. */
    readonly wrapV?: SceneUvWrap;
}

/** A texture on a material: an index into `ProceduralScene.textures` and where it is sampled. */
export interface SceneTextureReference {
    readonly texture: number;
    readonly uv?: SceneUvMapping;
}

export interface SceneMaterialData {
    readonly name: string;
    /** Diffuse albedo, linear. Multiplied with `albedoMap` when both are set. */
    readonly albedo: LinearColor;
    /** An albedo texture (glTF base colour texture, three's `map`); its alpha is what `alphaCutoff` tests. */
    readonly albedoMap?: SceneTextureReference;
    /** Emitted radiance, linear, may exceed 1. `[0, 0, 0]` for non-emitters. */
    readonly emissive: LinearColor;
    /** Alpha below this is a hole for rays; needs an albedo texture's alpha. Omit for opaque. */
    readonly alphaCutoff?: number;
    /**
     * How alpha (`opacity` × the albedo map's alpha) affects rays, as the
     * contract's `AlphaMode`: `cutout` is glTF MASK (three's `alphaTest`),
     * `stochastic` is glTF BLEND (three's `transparent`). Defaults to `cutout`
     * with an `alphaCutoff`, else `opaque`.
     */
    readonly alphaMode?: 'opaque' | 'cutout' | 'stochastic';
    /** 0 to 1, multiplied with the albedo map's alpha. Defaults to 1. */
    readonly opacity?: number;
    /**
     * 0 to 1: a metal has no diffuse colour, so GI's albedo is scaled by (1 − metalness).
     * Defaults to 0, or to 1 with a `metalnessMap` (glTF's `metallicFactor` default).
     */
    readonly metalness?: number;
    /**
     * A metalness map, multiplied with `metalness`. glTF and three read the
     * blue channel of the metallic-roughness texture, so only `b` exports.
     */
    readonly metalnessMap?: SceneTextureReference & { readonly channel: 'r' | 'g' | 'b' | 'a' };
    /**
     * Glass: 1 lets light through as a clear pane (glTF
     * `KHR_materials_transmission`, three's `transmission`). Omit or 0 for
     * an opaque surface. Set `blocksLight: false` with it, which is what the
     * platform's GI makes of glass.
     */
    readonly transmission?: number;
    /** False for surfaces rays pass through (glass). Defaults to true, as in the contract. */
    readonly blocksLight?: boolean;
    /** Rays hitting the back side still count as hits. */
    readonly doubleSided: boolean;
}

export interface SceneInstanceData {
    readonly name: string;
    /** Index into {@link ProceduralScene.meshes}. */
    readonly mesh: number;
    /** Index into {@link ProceduralScene.materials}: the material of slot 0 (and of the whole mesh when it has no groups). */
    readonly material: number;
    /**
     * One material per slot, for a mesh with `groups`; `materials[0]` equals
     * `material`. Omit for a one-material mesh. A reader that only knows
     * `material` draws the whole mesh with slot 0's material.
     */
    readonly materials?: readonly number[];
    /** Local-to-world, 16 numbers, column-major (the same order as three.js `Matrix4.elements`). */
    readonly transform: readonly number[];
}

interface SceneLightCommon {
    readonly label?: string;
    /** Linear colour. */
    readonly color: LinearColor;
    /** Directional: lux (irradiance at normal incidence). Point and spot: candela. */
    readonly intensity: number;
    /** Defaults to true. */
    readonly castsShadows?: boolean;
}

export interface SceneDirectionalLight extends SceneLightCommon {
    readonly type: 'directional';
    /** Direction the light travels. */
    readonly direction: Vector3Tuple;
}

export interface ScenePointLight extends SceneLightCommon {
    readonly type: 'point';
    readonly position: Vector3Tuple;
    /** Distance at which the light reaches zero, or 0 for no range window (plain inverse-square falloff, three.js `distance = 0`). */
    readonly range: number;
}

export interface SceneSpotLight extends SceneLightCommon {
    readonly type: 'spot';
    readonly position: Vector3Tuple;
    readonly direction: Vector3Tuple;
    /** As for point lights: a distance, or 0 for no range window. */
    readonly range: number;
    /** Half-angle where falloff starts, radians. */
    readonly innerConeAngle: number;
    /** Half-angle where the light reaches zero, radians. */
    readonly outerConeAngle: number;
}

export type SceneLightData = SceneDirectionalLight | ScenePointLight | SceneSpotLight;

/** What a procedural builder returns. */
export interface ProceduralScene {
    readonly meshes: readonly SceneMeshData[];
    /** Textures the materials' maps point at. Omitted by untextured scenes. */
    readonly textures?: readonly SceneTextureData[];
    readonly materials: readonly SceneMaterialData[];
    readonly instances: readonly SceneInstanceData[];
    readonly lights: readonly SceneLightData[];
}

/** Options a manifest can pass to a builder (`parts[].options`). */
export type BuilderOptions = Readonly<Record<string, string | number | boolean>>;

/** A procedural builder: plain function, no engine, no I/O. */
export type ProceduralBuilder = (options?: BuilderOptions) => ProceduralScene;
