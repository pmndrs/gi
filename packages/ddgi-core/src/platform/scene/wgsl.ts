/**
 * WGSL for the scene copy: accessor functions over the two scene buffers.
 *
 * Techniques and hit shading read the scene only through these functions, so
 * the record layouts in `layout.ts` can change without breaking them. The
 * source is generated from the layout constants, so the shader can never
 * disagree with the CPU about a word offset.
 *
 * What the module defines (all names start with `gi` / `Gi`):
 *
 * ```wgsl
 * struct GiAabb { min: vec3f, max: vec3f }
 * struct GiTriangle { vertex0: vec3f, vertex1: vec3f, vertex2: vec3f }
 * struct GiUvMapping, GiMaterial, GiLight, GiEnvironment
 *
 * // Instances (slots; a free slot reports 0 triangles and isn't traceable)
 * fn giInstanceCount() -> u32
 * fn giInstanceTransform(instance: u32) -> mat4x4f
 * fn giInstanceNormalMatrix(instance: u32) -> mat3x3f
 * fn giInstanceBounds(instance: u32) -> GiAabb
 * fn giInstanceMesh(instance: u32) -> u32
 * fn giInstanceTriangleCount(instance: u32) -> u32
 * fn giInstanceContributesToGI(instance: u32) -> bool
 * fn giInstanceTriangle(instance: u32, triangle: u32) -> GiTriangle      // world-space corners
 * fn giInstanceGeometricNormal(instance: u32, triangle: u32) -> vec3f    // world, unit
 * fn giInstanceNormalAt(instance: u32, triangle: u32, barycentrics: vec2f) -> vec3f
 * fn giInstanceUvAt(instance: u32, triangle: u32, barycentrics: vec2f, channel: u32) -> vec2f
 *
 * // Meshes (object space)
 * fn giMeshTriangleVertices(mesh: u32, triangle: u32) -> vec3u
 * fn giMeshVertexPosition(mesh: u32, vertex: u32) -> vec3f
 *
 * // Materials, lights, environment
 * fn giMaterialAt(instance: u32, triangle: u32) -> u32
 * fn giMaterial(material: u32) -> GiMaterial
 * fn giLightCount() -> u32
 * fn giDirectionalLightCount() -> u32                 // lights [0, this) are directional
 * fn giLight(light: u32) -> GiLight
 * fn giEnvironment() -> GiEnvironment
 * ```
 *
 * `triangle` is always the triangle's index in the mesh as the host gave it
 * (the order of `MeshDescriptor.indices`), the same index `GiHit.triangle`
 * returns. Barycentrics are the weights of vertices 1 and 2, as in `GiHit`.
 */

import type { BindingSlot, ModuleBindingEntry } from '../contract/wgsl.ts';
import {
    ENVIRONMENT_TYPE,
    HEADER_WORD,
    INSTANCE_FLAG,
    INSTANCE_RECORD_VEC4S,
    INSTANCE_WORD,
    LIGHT_FLAG,
    LIGHT_RECORD_VEC4S,
    LIGHT_TYPE,
    LIGHT_WORD,
    MATERIAL_FLAG,
    MATERIAL_RECORD_VEC4S,
    MATERIAL_WORD,
    MESH_FLAG,
    MESH_RECORD_VEC4S,
    MESH_WORD,
    NONE,
    UV_PACK_SHIFT,
} from './layout.ts';

/** `GPUShaderStage.COMPUTE`. The global does not exist in Node, where the tests import this file. */
const SHADER_STAGE_COMPUTE = 0x4;

/** The two read-only storage buffers the module binds: vertex data, then tables. */
export const SCENE_MODULE_ENTRIES: readonly ModuleBindingEntry[] = [
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
];

const hex = (value: number): string => `0x${value.toString(16)}u`;

/** Which vec4 of a record a word is in, and which component. */
function vec4Of(word: number): number {
    return Math.floor(word / 4);
}

const COMPONENTS = ['x', 'y', 'z', 'w'] as const;
function componentOf(word: number): string {
    return COMPONENTS[word % 4];
}

/** Vec3 fields must start a vec4 so `.xyz` reads them. */
function checkVec3(name: string, word: number): number {
    if (word % 4 !== 0) throw new Error(`layout: ${name} must start a vec4`);
    return vec4Of(word);
}

/** The scene module's source, with its bindings at `slot`: vertices, then tables. */
export function sceneModuleWgsl(slot: BindingSlot): string {
    const group = slot.group;
    const verticesBinding = slot.firstBinding;
    const tablesBinding = slot.firstBinding + 1;

    // Header vec4s.
    const headerCounts = vec4Of(HEADER_WORD.instanceSlots);
    const headerBases = vec4Of(HEADER_WORD.instancesBase);
    const headerExtra = vec4Of(HEADER_WORD.materialListsBase);
    const headerColor = checkVec3('environment colour', HEADER_WORD.environmentColor);
    const headerSun = checkVec3('sun direction', HEADER_WORD.environmentSunDirection);
    const headerRotation = vec4Of(HEADER_WORD.environmentRotationY);

    return /* wgsl */ `
// ---- Scene copy (platform/scene, generated from layout.ts) ---------------------------

struct GiAabb {
    min: vec3f,
    max: vec3f,
}

struct GiTriangle {
    vertex0: vec3f,
    vertex1: vec3f,
    vertex2: vec3f,
}

// Where a material's texture is sampled: uv' = transform * vec3f(uv, 1).
struct GiUvMapping {
    transform: mat3x2f,
    channel: u32,     // 0 = first uv set, 1 = second
    wrapU: u32,       // GI_UV_WRAP_*
    wrapV: u32,
}

struct GiMaterial {
    albedo: vec3f,          // diffuse albedo (multiplies the albedo texture when there is one)
    alphaCutoff: f32,       // alpha below this is a hole; 0 = opaque
    emission: vec3f,        // emitted radiance, emissive x emissiveIntensity
    blocksLight: bool,
    doubleSided: bool,
    albedoTexture: u32,     // GI_NO_TEXTURE when none; sampling arrives with the atlas (M4)
    albedoUv: GiUvMapping,
    emissiveTexture: u32,
    emissiveUv: GiUvMapping,
}

struct GiLight {
    kind: u32,              // GI_LIGHT_DIRECTIONAL, GI_LIGHT_POINT or GI_LIGHT_SPOT
    position: vec3f,        // world space (point and spot)
    direction: vec3f,       // unit, the way the light travels (directional and spot)
    range: f32,             // distance where point and spot lights reach zero; 0 = no limit
    intensity: vec3f,       // color x intensity, in the platform's light units
    castsShadows: bool,
    cosInnerCone: f32,      // spot: cosine of the angle where the falloff starts
    cosOuterCone: f32,      // spot: cosine of the angle where the light reaches zero
    radius: f32,            // emitter radius; angular radius in radians for directional lights
    group: u32,             // light group 0..3
}

struct GiEnvironment {
    kind: u32,              // GI_ENVIRONMENT_*
    radiance: vec3f,        // GI_ENVIRONMENT_COLOR: color x intensity
    intensity: f32,
    sunDirection: vec3f,    // GI_ENVIRONMENT_PROCEDURAL: unit vector toward the sun
    turbidity: f32,
    rotationY: f32,         // GI_ENVIRONMENT_TEXTURE: rotation about +Y, radians
}

const GI_NO_TEXTURE: u32 = ${hex(NONE)};

const GI_LIGHT_DIRECTIONAL: u32 = ${LIGHT_TYPE.directional}u;
const GI_LIGHT_POINT: u32 = ${LIGHT_TYPE.point}u;
const GI_LIGHT_SPOT: u32 = ${LIGHT_TYPE.spot}u;

const GI_ENVIRONMENT_NONE: u32 = ${ENVIRONMENT_TYPE.none}u;
const GI_ENVIRONMENT_COLOR: u32 = ${ENVIRONMENT_TYPE.color}u;
const GI_ENVIRONMENT_TEXTURE: u32 = ${ENVIRONMENT_TYPE.texture}u;
const GI_ENVIRONMENT_PROCEDURAL: u32 = ${ENVIRONMENT_TYPE.procedural}u;

const GI_UV_WRAP_REPEAT: u32 = 0u;
const GI_UV_WRAP_CLAMP: u32 = 1u;
const GI_UV_WRAP_MIRROR: u32 = 2u;

// ---- Bindings and layout ---------------------------------------------------------------

// Every mesh's indices, positions, normals, uvs and material slots, at word
// offsets kept in the mesh records. Floats are read with bitcast<f32>.
@group(${group}) @binding(${verticesBinding}) var<storage, read> giSceneVertices: array<u32>;
// Header, then instance, mesh, material and light records, then material lists.
@group(${group}) @binding(${tablesBinding}) var<storage, read> giSceneTables: array<vec4u>;

const GI_SCENE_NONE: u32 = ${hex(NONE)};

const GI_SCENE_INSTANCE_VEC4S: u32 = ${INSTANCE_RECORD_VEC4S}u;
const GI_SCENE_MESH_VEC4S: u32 = ${MESH_RECORD_VEC4S}u;
const GI_SCENE_MATERIAL_VEC4S: u32 = ${MATERIAL_RECORD_VEC4S}u;
const GI_SCENE_LIGHT_VEC4S: u32 = ${LIGHT_RECORD_VEC4S}u;

const GI_SCENE_INSTANCE_LIVE: u32 = ${INSTANCE_FLAG.live}u;
// An instance rays see: live, visible, contributing, and with an invertible transform.
const GI_SCENE_INSTANCE_TRACEABLE_MASK: u32 = ${hex(INSTANCE_FLAG.live | INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible | INSTANCE_FLAG.singular)};
const GI_SCENE_INSTANCE_TRACEABLE: u32 = ${hex(INSTANCE_FLAG.live | INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible)};

const GI_SCENE_MESH_INDICES16: u32 = ${MESH_FLAG.indices16}u;
const GI_SCENE_MATERIAL_BLOCKS_LIGHT: u32 = ${MATERIAL_FLAG.blocksLight}u;
const GI_SCENE_MATERIAL_DOUBLE_SIDED: u32 = ${MATERIAL_FLAG.doubleSided}u;
const GI_SCENE_LIGHT_CASTS_SHADOWS: u32 = ${LIGHT_FLAG.castsShadows}u;

// ---- Header ------------------------------------------------------------------------------

// (instance slots, mesh slots, material slots, light count)
fn giSceneCounts() -> vec4u {
    return giSceneTables[${headerCounts}];
}

// Where the instance, mesh, material and light sections start, in vec4s.
fn giSceneBases() -> vec4u {
    return giSceneTables[${headerBases}];
}

// One u32 of the table buffer, by word index. The component is picked with
// select() rather than by indexing the vector with a runtime value, which can
// push the vector into slow per-thread memory (NOTES, "Tracing").
fn giSceneTableWord(word: u32) -> u32 {
    let quad = giSceneTables[word >> 2u];
    let component = word & 3u;
    return select(select(quad.x, quad.y, component == 1u), select(quad.z, quad.w, component == 3u), component >= 2u);
}

fn giSceneVertexFloat(word: u32) -> f32 {
    return bitcast<f32>(giSceneVertices[word]);
}

// ---- Instances ---------------------------------------------------------------------------

fn giSceneInstanceBase(instance: u32) -> u32 {
    return giSceneBases().x + instance * GI_SCENE_INSTANCE_VEC4S;
}

fn giSceneInstanceFlags(instance: u32) -> u32 {
    return giSceneTables[giSceneInstanceBase(instance) + ${vec4Of(INSTANCE_WORD.flags)}u].${componentOf(INSTANCE_WORD.flags)};
}

// Instance slots in use: indices 0 .. giInstanceCount() - 1. Destroyed
// instances leave free slots, which report 0 triangles and aren't traceable.
fn giInstanceCount() -> u32 {
    return giSceneCounts().x;
}

// Local-to-world transform.
fn giInstanceTransform(instance: u32) -> mat4x4f {
    let base = giSceneInstanceBase(instance);
    // Stored as the three rows of the 3x4 matrix; WGSL matrices are built from columns.
    let row0 = bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.objectToWorldRow0)}u]);
    let row1 = bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.objectToWorldRow1)}u]);
    let row2 = bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.objectToWorldRow2)}u]);
    return transpose(mat4x4f(row0, row1, row2, vec4f(0.0, 0.0, 0.0, 1.0)));
}

// Takes object-space normals to world space: transpose(inverse(linear part)).
// The record holds the rows of inverse(L), and those rows are the columns of
// its transpose, so they go straight in as columns. The result is not unit
// length; normalize after multiplying.
fn giInstanceNormalMatrix(instance: u32) -> mat3x3f {
    let base = giSceneInstanceBase(instance);
    return mat3x3f(
        bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.normalRow0)}u]).xyz,
        bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.normalRow1)}u]).xyz,
        bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.normalRow2)}u]).xyz,
    );
}

// World-space box around the instance (slightly padded, as the tracer's top level is).
fn giInstanceBounds(instance: u32) -> GiAabb {
    let base = giSceneInstanceBase(instance);
    return GiAabb(
        bitcast<vec4f>(giSceneTables[base + ${checkVec3('instance boundsMin', INSTANCE_WORD.boundsMin)}u]).xyz,
        bitcast<vec4f>(giSceneTables[base + ${checkVec3('instance boundsMax', INSTANCE_WORD.boundsMax)}u]).xyz,
    );
}

fn giInstanceMesh(instance: u32) -> u32 {
    return giSceneTables[giSceneInstanceBase(instance) + ${vec4Of(INSTANCE_WORD.mesh)}u].${componentOf(INSTANCE_WORD.mesh)};
}

// True when rays see this instance: it exists, is visible, contributes to GI
// and has an invertible transform.
fn giInstanceContributesToGI(instance: u32) -> bool {
    if (instance >= giInstanceCount()) {
        return false;
    }
    return (giSceneInstanceFlags(instance) & GI_SCENE_INSTANCE_TRACEABLE_MASK) == GI_SCENE_INSTANCE_TRACEABLE;
}

// Triangles in the instance's mesh; 0 for a free slot or an out-of-range index.
fn giInstanceTriangleCount(instance: u32) -> u32 {
    if (instance >= giInstanceCount() || (giSceneInstanceFlags(instance) & GI_SCENE_INSTANCE_LIVE) == 0u) {
        return 0u;
    }
    return giMeshTriangleCount(giInstanceMesh(instance));
}

fn giSceneObjectToWorld(instance: u32, point: vec3f) -> vec3f {
    let base = giSceneInstanceBase(instance);
    let row0 = bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.objectToWorldRow0)}u]);
    let row1 = bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.objectToWorldRow1)}u]);
    let row2 = bitcast<vec4f>(giSceneTables[base + ${vec4Of(INSTANCE_WORD.objectToWorldRow2)}u]);
    return vec3f(
        dot(row0.xyz, point) + row0.w,
        dot(row1.xyz, point) + row1.w,
        dot(row2.xyz, point) + row2.w,
    );
}

// World-space corners of one triangle, for voxelizers, debug drawing and hit shading.
fn giInstanceTriangle(instance: u32, triangle: u32) -> GiTriangle {
    let mesh = giInstanceMesh(instance);
    let vertices = giMeshTriangleVertices(mesh, triangle);
    return GiTriangle(
        giSceneObjectToWorld(instance, giMeshVertexPosition(mesh, vertices.x)),
        giSceneObjectToWorld(instance, giMeshVertexPosition(mesh, vertices.y)),
        giSceneObjectToWorld(instance, giMeshVertexPosition(mesh, vertices.z)),
    );
}

// The triangle's own unit normal in world space, from its world-space winding
// (corner 0 -> 1 -> 2, counter-clockwise seen from the front). The tracer's
// GiHit.frontFace uses the same winding, so frontFace is true exactly when
// dot(ray.direction, giInstanceGeometricNormal(...)) < 0. A mirrored instance
// reverses the winding, so this points the other way from its vertex normals.
fn giInstanceGeometricNormal(instance: u32, triangle: u32) -> vec3f {
    let corners = giInstanceTriangle(instance, triangle);
    let normal = cross(corners.vertex1 - corners.vertex0, corners.vertex2 - corners.vertex0);
    let lengthSquared = dot(normal, normal);
    // A degenerate triangle has no normal; rays can't hit it, so any unit vector will do.
    return select(vec3f(0.0, 0.0, 1.0), normal * inverseSqrt(lengthSquared), lengthSquared > 0.0);
}

// The interpolated vertex normal at a point of the triangle, in world space,
// unit length. Falls back to the geometric normal when the mesh has no
// normals or the interpolated normal vanishes (opposite normals at the corners).
fn giInstanceNormalAt(instance: u32, triangle: u32, barycentrics: vec2f) -> vec3f {
    let mesh = giInstanceMesh(instance);
    let normals = giSceneMeshRecord(mesh, ${vec4Of(MESH_WORD.normals)}u).${componentOf(MESH_WORD.normals)};
    if (normals == GI_SCENE_NONE) {
        return giInstanceGeometricNormal(instance, triangle);
    }
    let vertices = giMeshTriangleVertices(mesh, triangle);
    let objectNormal =
        giSceneDecodeNormal(giSceneVertices[normals + vertices.x]) * (1.0 - barycentrics.x - barycentrics.y) +
        giSceneDecodeNormal(giSceneVertices[normals + vertices.y]) * barycentrics.x +
        giSceneDecodeNormal(giSceneVertices[normals + vertices.z]) * barycentrics.y;
    let worldNormal = giInstanceNormalMatrix(instance) * objectNormal;
    let lengthSquared = dot(worldNormal, worldNormal);
    if (!(lengthSquared > 1.0e-20)) {
        return giInstanceGeometricNormal(instance, triangle);
    }
    return worldNormal * inverseSqrt(lengthSquared);
}

// The interpolated uv of set 'channel' (0 or 1) at a point of the triangle.
// (0, 0) when the mesh has no such uv set.
fn giInstanceUvAt(instance: u32, triangle: u32, barycentrics: vec2f, channel: u32) -> vec2f {
    let mesh = giInstanceMesh(instance);
    let uvSets = giSceneMeshRecord(mesh, ${vec4Of(MESH_WORD.uvs0)}u);
    let uvs = select(uvSets.${componentOf(MESH_WORD.uvs0)}, uvSets.${componentOf(MESH_WORD.uvs1)}, channel == 1u);
    if (uvs == GI_SCENE_NONE) {
        return vec2f(0.0);
    }
    let vertices = giMeshTriangleVertices(mesh, triangle);
    return giSceneUv(uvs, vertices.x) * (1.0 - barycentrics.x - barycentrics.y) +
        giSceneUv(uvs, vertices.y) * barycentrics.x +
        giSceneUv(uvs, vertices.z) * barycentrics.y;
}

// ---- Meshes ----------------------------------------------------------------------------

fn giSceneMeshRecord(mesh: u32, vec4Index: u32) -> vec4u {
    return giSceneTables[giSceneBases().y + mesh * GI_SCENE_MESH_VEC4S + vec4Index];
}

fn giMeshTriangleCount(mesh: u32) -> u32 {
    return giSceneMeshRecord(mesh, ${vec4Of(MESH_WORD.triangleCount)}u).${componentOf(MESH_WORD.triangleCount)};
}

// The three vertex indices of a triangle, in the mesh's own order.
fn giMeshTriangleVertices(mesh: u32, triangle: u32) -> vec3u {
    let record = giSceneMeshRecord(mesh, ${vec4Of(MESH_WORD.indices)}u);
    let indices = record.${componentOf(MESH_WORD.indices)};
    let flags = giSceneMeshRecord(mesh, ${vec4Of(MESH_WORD.flags)}u).${componentOf(MESH_WORD.flags)};
    let first = 3u * triangle;
    if ((flags & GI_SCENE_MESH_INDICES16) != 0u) {
        // Two 16-bit indices per word, even index in the low half. A triangle's
        // three indices span two words: low, high, low when 'first' is even;
        // high, low, high when it is odd. Both words always belong to this mesh.
        let word = indices + (first >> 1u);
        let a = giSceneVertices[word];
        let b = giSceneVertices[word + 1u];
        let evenStart = vec3u(a & 0xffffu, a >> 16u, b & 0xffffu);
        let oddStart = vec3u(a >> 16u, b & 0xffffu, b >> 16u);
        return select(evenStart, oddStart, (first & 1u) == 1u);
    }
    let word = indices + first;
    return vec3u(giSceneVertices[word], giSceneVertices[word + 1u], giSceneVertices[word + 2u]);
}

// A vertex position in the mesh's own (object) space.
fn giMeshVertexPosition(mesh: u32, vertex: u32) -> vec3f {
    let positions = giSceneMeshRecord(mesh, ${vec4Of(MESH_WORD.positions)}u).${componentOf(MESH_WORD.positions)};
    let word = positions + 3u * vertex;
    return vec3f(giSceneVertexFloat(word), giSceneVertexFloat(word + 1u), giSceneVertexFloat(word + 2u));
}

fn giSceneUv(uvs: u32, vertex: u32) -> vec2f {
    let word = uvs + 2u * vertex;
    return vec2f(giSceneVertexFloat(word), giSceneVertexFloat(word + 1u));
}

// Octahedral decode (Cigolle et al. 2014), the inverse of octahedral.ts. The
// square [-1, 1]^2 unfolds onto the octahedron |x| + |y| + |z| = 1: points
// outside the diamond |x| + |y| <= 1 came from the lower half (z < 0) and are
// folded back across the diagonals.
fn giSceneDecodeNormal(word: u32) -> vec3f {
    let encoded = unpack2x16snorm(word);
    var normal = vec3f(encoded.x, encoded.y, 1.0 - abs(encoded.x) - abs(encoded.y));
    let fold = max(-normal.z, 0.0);
    normal.x += select(fold, -fold, normal.x >= 0.0);
    normal.y += select(fold, -fold, normal.y >= 0.0);
    return normalize(normal);
}

// ---- Materials -------------------------------------------------------------------------

// The material slot the mesh's groups give this triangle.
fn giSceneTriangleMaterialSlot(mesh: u32, triangle: u32) -> u32 {
    let slots = giSceneMeshRecord(mesh, ${vec4Of(MESH_WORD.materialSlots)}u).${componentOf(MESH_WORD.materialSlots)};
    if (slots == GI_SCENE_NONE) {
        return giSceneMeshRecord(mesh, ${vec4Of(MESH_WORD.constantMaterialSlot)}u).${componentOf(MESH_WORD.constantMaterialSlot)};
    }
    // 16-bit slots, two per word, even triangle in the low half.
    let word = giSceneVertices[slots + (triangle >> 1u)];
    return (word >> ((triangle & 1u) * 16u)) & 0xffffu;
}

// The material (an index for giMaterial) this triangle of this instance uses.
fn giMaterialAt(instance: u32, triangle: u32) -> u32 {
    let base = giSceneInstanceBase(instance);
    let listStart = giSceneTables[base + ${vec4Of(INSTANCE_WORD.materialListStart)}u].${componentOf(INSTANCE_WORD.materialListStart)};
    let listCount = giSceneTables[base + ${vec4Of(INSTANCE_WORD.materialListCount)}u].${componentOf(INSTANCE_WORD.materialListCount)};
    // The CPU checks that an instance names a material for every slot. If the
    // mesh changes shape later and uses more slots, the last material stands in.
    let slot = min(giSceneTriangleMaterialSlot(giInstanceMesh(instance), triangle), max(listCount, 1u) - 1u);
    let listsBase = giSceneTables[${headerExtra}].${componentOf(HEADER_WORD.materialListsBase)};
    return giSceneTableWord(listsBase * 4u + listStart + slot);
}

fn giSceneUvMapping(transform: vec4u, translationAndPacked: vec4u) -> GiUvMapping {
    let linear = bitcast<vec4f>(transform);
    let translation = bitcast<vec2f>(translationAndPacked.xy);
    let packed = translationAndPacked.w;
    return GiUvMapping(
        mat3x2f(linear.xy, linear.zw, translation),
        (packed >> ${UV_PACK_SHIFT.channel}u) & 0xfu,
        (packed >> ${UV_PACK_SHIFT.wrapU}u) & 0xfu,
        (packed >> ${UV_PACK_SHIFT.wrapV}u) & 0xfu,
    );
}

fn giMaterial(material: u32) -> GiMaterial {
    let base = giSceneBases().z + material * GI_SCENE_MATERIAL_VEC4S;
    let albedo = bitcast<vec4f>(giSceneTables[base + ${checkVec3('albedo', MATERIAL_WORD.albedo)}u]);
    let emission = giSceneTables[base + ${checkVec3('emission', MATERIAL_WORD.emission)}u];
    let flags = emission.${componentOf(MATERIAL_WORD.flags)};
    let albedoTranslation = giSceneTables[base + ${vec4Of(MATERIAL_WORD.albedoUvTranslation)}u];
    let emissiveTranslation = giSceneTables[base + ${vec4Of(MATERIAL_WORD.emissiveUvTranslation)}u];
    return GiMaterial(
        albedo.xyz,
        albedo.${componentOf(MATERIAL_WORD.alphaCutoff)},
        bitcast<vec3f>(emission.xyz),
        (flags & GI_SCENE_MATERIAL_BLOCKS_LIGHT) != 0u,
        (flags & GI_SCENE_MATERIAL_DOUBLE_SIDED) != 0u,
        albedoTranslation.${componentOf(MATERIAL_WORD.albedoTexture)},
        giSceneUvMapping(giSceneTables[base + ${checkVec3('albedo uv', MATERIAL_WORD.albedoUvMatrix)}u], albedoTranslation),
        emissiveTranslation.${componentOf(MATERIAL_WORD.emissiveTexture)},
        giSceneUvMapping(giSceneTables[base + ${checkVec3('emissive uv', MATERIAL_WORD.emissiveUvMatrix)}u], emissiveTranslation),
    );
}

// ---- Lights and environment ------------------------------------------------------------

// Lights are packed: every index below giLightCount() is a live light.
fn giLightCount() -> u32 {
    return giSceneCounts().w;
}

// Lights 0 .. giDirectionalLightCount() - 1 are directional; the rest are point and spot lights.
fn giDirectionalLightCount() -> u32 {
    return giSceneTables[${headerExtra}].${componentOf(HEADER_WORD.directionalLightCount)};
}

fn giLight(light: u32) -> GiLight {
    let base = giSceneBases().w + light * GI_SCENE_LIGHT_VEC4S;
    let position = giSceneTables[base + ${checkVec3('light position', LIGHT_WORD.position)}u];
    let direction = bitcast<vec4f>(giSceneTables[base + ${checkVec3('light direction', LIGHT_WORD.direction)}u]);
    let intensity = giSceneTables[base + ${checkVec3('light intensity', LIGHT_WORD.intensity)}u];
    let cone = giSceneTables[base + ${vec4Of(LIGHT_WORD.cosInnerCone)}u];
    return GiLight(
        position.${componentOf(LIGHT_WORD.type)},
        bitcast<vec3f>(position.xyz),
        direction.xyz,
        direction.${componentOf(LIGHT_WORD.range)},
        bitcast<vec3f>(intensity.xyz),
        (intensity.${componentOf(LIGHT_WORD.flags)} & GI_SCENE_LIGHT_CASTS_SHADOWS) != 0u,
        bitcast<f32>(cone.${componentOf(LIGHT_WORD.cosInnerCone)}),
        bitcast<f32>(cone.${componentOf(LIGHT_WORD.cosOuterCone)}),
        bitcast<f32>(cone.${componentOf(LIGHT_WORD.radius)}),
        cone.${componentOf(LIGHT_WORD.group)},
    );
}

fn giEnvironment() -> GiEnvironment {
    let color = bitcast<vec4f>(giSceneTables[${headerColor}]);
    let sun = bitcast<vec4f>(giSceneTables[${headerSun}]);
    let rotation = bitcast<vec4f>(giSceneTables[${headerRotation}]);
    return GiEnvironment(
        giSceneTables[${headerExtra}].${componentOf(HEADER_WORD.environmentType)},
        color.xyz,
        color.${componentOf(HEADER_WORD.environmentIntensity)},
        sun.xyz,
        sun.${componentOf(HEADER_WORD.environmentTurbidity)},
        rotation.${componentOf(HEADER_WORD.environmentRotationY)},
    );
}
`;
}
