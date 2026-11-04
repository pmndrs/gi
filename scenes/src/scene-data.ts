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
 *
 * Everything is metres, right-handed, Y up, linear-light colours.
 */

/** `[x, y, z]`. */
export type Vector3Tuple = readonly [number, number, number];

/** Linear-light RGB, not sRGB-encoded. */
export type LinearColor = readonly [number, number, number];

export interface SceneMeshData {
    readonly name: string;
    /** xyz per vertex, local space. */
    readonly positions: Float32Array;
    /** Unit normals, one per vertex. */
    readonly normals: Float32Array;
    /** uv per vertex. */
    readonly uvs: Float32Array;
    /** Triangle list. */
    readonly indices: Uint32Array;
}

export interface SceneMaterialData {
    readonly name: string;
    /** Diffuse albedo, linear. */
    readonly albedo: LinearColor;
    /** Emitted radiance, linear, may exceed 1. `[0, 0, 0]` for non-emitters. */
    readonly emissive: LinearColor;
    /** Alpha below this is a hole for rays; needs an albedo texture's alpha. Omit for opaque. */
    readonly alphaCutoff?: number;
    /** Rays hitting the back side still count as hits. */
    readonly doubleSided: boolean;
}

export interface SceneInstanceData {
    readonly name: string;
    /** Index into {@link ProceduralScene.meshes}. */
    readonly mesh: number;
    /** Index into {@link ProceduralScene.materials}. */
    readonly material: number;
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
    readonly materials: readonly SceneMaterialData[];
    readonly instances: readonly SceneInstanceData[];
    readonly lights: readonly SceneLightData[];
}

/** Options a manifest can pass to a builder (`parts[].options`). */
export type BuilderOptions = Readonly<Record<string, string | number | boolean>>;

/** A procedural builder: plain function, no engine, no I/O. */
export type ProceduralBuilder = (options?: BuilderOptions) => ProceduralScene;
