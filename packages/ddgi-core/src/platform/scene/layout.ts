/**
 * Memory layout of the platform's GPU scene copy, in one place.
 *
 * The scene copy is two storage buffers, so that the scene copy and hit
 * shading together stay within 2 of the 8 storage buffers a default device
 * allows per shader stage (docs/api/platform.md, "After T0.10"):
 *
 * 1. **The vertex buffer** (`giSceneVertices`, `array<u32>`): every mesh's
 *    data, one mesh after another, at word offsets kept in the mesh's record.
 *    Per mesh: triangle indices, positions, then the optional normals, uvs,
 *    second uvs and per-triangle material slots.
 * 2. **The table buffer** (`giSceneTables`, `array<vec4u>`): a header, then
 *    fixed-capacity sections of instance, mesh, material and light records,
 *    then each instance's list of material indices.
 *
 * The CPU writes these layouts and the WGSL reads them, so every number both
 * sides must agree on lives here, and the WGSL source is generated from these
 * constants (see `wgsl.ts`). The layout is private: techniques read the scene
 * only through the accessor functions, so it may change at any time.
 *
 * Everything is 32-bit words. Floats are stored as their bit patterns and
 * read with `bitcast<f32>`, so an integer never passes through a float
 * register on the GPU (a NaN-shaped integer could be changed on the way).
 */

/** Marks an absent offset or texture: "this mesh has no normals", "no albedo texture". */
export const NONE = 0xffffffff;

// ---------------------------------------------------------------------------
// Vertex buffer: per-mesh data
// ---------------------------------------------------------------------------

/**
 * Vertex normals are stored octahedral-encoded as two snorm16 values in one
 * word (4 bytes, against 12 for three floats). WGSL decodes them with
 * `unpack2x16snorm`. The angular error is about 0.01 degrees, far below
 * anything probe lighting can show. Chosen by measured memory (T1.2, whole
 * vertex buffer): Intel Sponza 104.1 -> 87.7 MB, Bistro exterior 134.8 ->
 * 111.4 MB. With full floats Bistro exterior's scene copy would not fit one
 * 128 MB storage binding, the default limit.
 */
export const NORMAL_WORDS_PER_VERTEX = 1;
export const POSITION_WORDS_PER_VERTEX = 3;
export const UV_WORDS_PER_VERTEX = 2;

/**
 * Triangle indices are stored as 16 bits when every index of the mesh fits
 * (65,536 vertices or fewer), else as 32 bits. Every mesh of Intel Sponza and
 * Bistro interior fits, so this halves their largest array (measured, whole
 * vertex buffer: Sponza 110.2 -> 87.7 MB).
 */
export const MAX_VERTICES_FOR_16_BIT_INDICES = 0x10000;

/**
 * Per-triangle material slots are stored only for meshes whose triangles use
 * more than one slot, as 16-bit values, two per word. A mesh with one slot
 * stores it in its record instead.
 */
export const MAX_MATERIAL_SLOTS = 0x10000;

// ---------------------------------------------------------------------------
// Table buffer
// ---------------------------------------------------------------------------

/** Everything in the table buffer is addressed in 16-byte `vec4u` units. */
export const TABLE_VEC4_BYTES = 16;

/**
 * The header: counts, where each section starts, and the environment.
 * 8 vec4s (128 bytes). Section bases are vec4 indices into the table buffer;
 * the material-list base is too, and lists inside it are addressed in words.
 */
export const HEADER_VEC4S = 8;

/** Word positions inside the header (vec4 index × 4 + component). */
export const HEADER_WORD = {
    /** One more than the highest instance slot in use. Free slots in between read as empty. */
    instanceSlots: 0,
    meshSlots: 1,
    materialSlots: 2,
    /** Lights are packed: `[0, lightCount)` are all live. */
    lightCount: 3,

    instancesBase: 4,
    meshesBase: 5,
    materialsBase: 6,
    lightsBase: 7,

    materialListsBase: 8,
    /** Lights `[0, directionalLightCount)` are directional, the rest point or spot. */
    directionalLightCount: 9,
    environmentType: 10,
    // word 11 unused

    /** Environment radiance for `color` (already times intensity): rgb, then intensity. */
    environmentColor: 12,
    environmentIntensity: 15,

    /** Procedural sky: unit direction toward the sun, then turbidity. */
    environmentSunDirection: 16,
    environmentTurbidity: 19,

    /** Rotation of a texture environment about +Y, radians. */
    environmentRotationY: 20,
} as const;

/** `environmentType` values. */
export const ENVIRONMENT_TYPE = {
    none: 0,
    color: 1,
    texture: 2,
    procedural: 3,
} as const;

/**
 * Instance record, 8 vec4s (128 bytes):
 *
 *     objectToWorldRow0..2: vec4f   rows of the 3x4 object-to-world matrix (xyz linear, w translation)
 *     normalRow0..2.xyz             rows of inverse(linear part); as matrix *columns* they are the
 *                                   normal matrix transpose(inverse(L))
 *     normalRow0.w: u32             first word of this instance's material list (relative to the list section)
 *     normalRow1.w: u32             number of entries in the list
 *     normalRow2.w: u32             reserved (written as 0)
 *     boundsMin.xyz, w: u32         world-space box (min), mesh slot
 *     boundsMax.xyz, w: u32         world-space box (max), flags
 *
 * A free slot is all zeros: flags 0 (not live) and mesh 0, and the accessors
 * report 0 triangles for it.
 */
export const INSTANCE_RECORD_VEC4S = 8;
export const INSTANCE_WORD = {
    objectToWorldRow0: 0,
    objectToWorldRow1: 4,
    objectToWorldRow2: 8,
    normalRow0: 12,
    materialListStart: 15,
    normalRow1: 16,
    materialListCount: 19,
    normalRow2: 20,
    /** The w lane of normalRow2: nothing uses it yet; kept 0. */
    reserved: 23,
    boundsMin: 24,
    mesh: 27,
    boundsMax: 28,
    flags: 31,
} as const;

/** Instance `flags` bits. */
export const INSTANCE_FLAG = {
    /** The slot holds an instance. */
    live: 1,
    /** Rays see this instance (`contributesToGI`). */
    contributesToGi: 2,
    visible: 4,
    /**
     * The transform has a negative determinant. World-space winding is then
     * the reverse of the mesh's own winding.
     */
    mirrored: 8,
    /** The transform cannot be inverted; the instance has no area and rays skip it. */
    singular: 16,
} as const;

/**
 * Mesh record, 4 vec4s (64 bytes). Offsets are word indices into the vertex buffer.
 *
 *     indices, positions, triangleCount, vertexCount
 *     normals, uvs0, uvs1, materialSlots     (NONE when absent)
 *     boundsMin.xyz, constantMaterialSlot    (used when materialSlots is NONE)
 *     boundsMax.xyz, flags
 */
export const MESH_RECORD_VEC4S = 4;
export const MESH_WORD = {
    indices: 0,
    positions: 1,
    triangleCount: 2,
    vertexCount: 3,
    normals: 4,
    uvs0: 5,
    uvs1: 6,
    materialSlots: 7,
    boundsMin: 8,
    constantMaterialSlot: 11,
    boundsMax: 12,
    flags: 15,
} as const;

/** Mesh `flags` bits. */
export const MESH_FLAG = {
    live: 1,
    /** Indices are 16-bit, two per word. */
    indices16: 2,
    /** Positions will come from the host each frame (M3). */
    deforming: 4,
} as const;

/**
 * Material record, 9 vec4s (144 bytes):
 *
 *     albedo.rgb, alphaCutoff (0 unless the alpha mode is cutout)
 *     emission.rgb (emissive × emissiveIntensity), flags
 *     albedoUv matrix m0..m3
 *     albedoUv m4, m5, albedoTexture (store layer or NONE), albedoUv packed (channel, wrapU, wrapV)
 *     emissiveUv matrix m0..m3
 *     emissiveUv m4, m5, emissiveTexture, emissiveUv packed
 *     metalness, opacity, alphaMode, reserved (0)
 *     metalnessUv matrix m0..m3
 *     metalnessUv m4, m5, metalnessTexture, metalnessUv packed (and the texture channel metalness is read from)
 *
 * The uv matrix is the contract's 3×2 column-major transform:
 * `uv' = (m0·u + m2·v + m4, m1·u + m3·v + m5)`.
 *
 * A texture word holds the texture's layer in the texture store
 * (textures/texture-store.ts), or NONE when the material has no such texture
 * or its layer hasn't been made yet: the material then shades as if the
 * texture were white, and the platform rewrites the record when the layer
 * lands (M4, T4.1).
 */
export const MATERIAL_RECORD_VEC4S = 9;
export const MATERIAL_WORD = {
    albedo: 0,
    alphaCutoff: 3,
    emission: 4,
    flags: 7,
    albedoUvMatrix: 8,
    albedoUvTranslation: 12,
    albedoTexture: 14,
    albedoUvPacked: 15,
    emissiveUvMatrix: 16,
    emissiveUvTranslation: 20,
    emissiveTexture: 22,
    emissiveUvPacked: 23,
    metalness: 24,
    opacity: 25,
    alphaMode: 26,
    /** Nothing uses it yet; kept 0. */
    reserved: 27,
    metalnessUvMatrix: 28,
    metalnessUvTranslation: 32,
    metalnessTexture: 34,
    metalnessUvPacked: 35,
} as const;

/** Material `flags` bits. */
export const MATERIAL_FLAG = {
    live: 1,
    blocksLight: 2,
    doubleSided: 4,
    /** The alpha mode is cutout and there is an albedo texture to read alpha from (M4). */
    alphaTest: 8,
} as const;

/** `alphaMode` values (contract: `MaterialDescriptor.alphaMode`). */
export const ALPHA_MODE = { opaque: 0, cutout: 1, stochastic: 2 } as const;

/** The cutoff a cutout material gets when it names none (glTF's `alphaCutoff` default). */
export const DEFAULT_ALPHA_CUTOFF = 0.5;

/**
 * Packed uv mapping: bits 0-3 the uv channel, 4-7 wrapU, 8-11 wrapV, and
 * 12-15 which texture channel holds the value (`TEXTURE_CHANNEL`; only the
 * metalness map uses it, the other maps read rgb or rgba).
 */
export const UV_PACK_SHIFT = { channel: 0, wrapU: 4, wrapV: 8, textureChannel: 12 } as const;
export const UV_WRAP = { repeat: 0, clamp: 1, mirror: 2 } as const;
export const TEXTURE_CHANNEL = { r: 0, g: 1, b: 2, a: 3 } as const;

/**
 * Light record, 4 vec4s (64 bytes):
 *
 *     position.xyz, type
 *     direction.xyz (unit, the way the light travels), range (0 = no limit)
 *     intensity × color (rgb), flags
 *     cos(innerConeAngle), cos(outerConeAngle), radius (angular radius for directional), group
 *
 * The cone angles are stored as cosines because the shader compares the
 * cosine of the angle to the axis, never the angle itself.
 */
export const LIGHT_RECORD_VEC4S = 4;
export const LIGHT_WORD = {
    position: 0,
    type: 3,
    direction: 4,
    range: 7,
    intensity: 8,
    flags: 11,
    cosInnerCone: 12,
    cosOuterCone: 13,
    radius: 14,
    group: 15,
} as const;

export const LIGHT_TYPE = { directional: 0, point: 1, spot: 2 } as const;
export const LIGHT_FLAG = { castsShadows: 1 } as const;

/** Light groups 0..3 (contract: `LightCommon.group`). */
export { LIGHT_GROUP_COUNT } from '../contract/scene.ts';

/** Where the table buffer's sections start, in vec4s, for given capacities. */
export interface TableLayout {
    readonly instanceCapacity: number;
    readonly meshCapacity: number;
    readonly materialCapacity: number;
    readonly lightCapacity: number;
    /** Words available for instance material lists. A multiple of 4. */
    readonly materialListCapacity: number;
    readonly instancesBase: number;
    readonly meshesBase: number;
    readonly materialsBase: number;
    readonly lightsBase: number;
    readonly materialListsBase: number;
    /** Size of the whole table buffer in vec4s. */
    readonly totalVec4s: number;
}

export function tableLayout(capacities: {
    instanceCapacity: number;
    meshCapacity: number;
    materialCapacity: number;
    lightCapacity: number;
    materialListCapacity: number;
}): TableLayout {
    const materialListCapacity = Math.ceil(capacities.materialListCapacity / 4) * 4;
    const instancesBase = HEADER_VEC4S;
    const meshesBase = instancesBase + capacities.instanceCapacity * INSTANCE_RECORD_VEC4S;
    const materialsBase = meshesBase + capacities.meshCapacity * MESH_RECORD_VEC4S;
    const lightsBase = materialsBase + capacities.materialCapacity * MATERIAL_RECORD_VEC4S;
    const materialListsBase = lightsBase + capacities.lightCapacity * LIGHT_RECORD_VEC4S;
    return {
        ...capacities,
        materialListCapacity,
        instancesBase,
        meshesBase,
        materialsBase,
        lightsBase,
        materialListsBase,
        totalVec4s: materialListsBase + materialListCapacity / 4,
    };
}
