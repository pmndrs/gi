/**
 * A small binary glTF (GLB) writer for {@link ProceduralScene}.
 *
 * The native harness and Blender both read glTF, so procedural scenes get
 * exported to it. Writing the file by hand (about 150 lines) is simpler and
 * more predictable than driving three's GLTFExporter in Node, which expects a
 * DOM for some paths and rewrites materials and transforms on the way.
 *
 * What the file contains:
 * - one glTF mesh per (source mesh, material) pair, sharing vertex data;
 * - one node per instance, with its column-major matrix as-is;
 * - materials as metallic-roughness with the albedo as the base colour factor
 *   (metallic 0, roughness 1) and emission through `KHR_materials_emissive_strength`
 *   so values above 1 survive;
 * - lights through `KHR_lights_punctual`, whose units (lux, candela) match
 *   three.js physical lights.
 */

import type { ProceduralScene, SceneLightData, SceneMeshData, Vector3Tuple } from './scene-data';

const GLB_MAGIC = 0x46546c67; // "glTF"
const GLB_VERSION = 2;
const CHUNK_JSON = 0x4e4f534a; // "JSON"
const CHUNK_BIN = 0x004e4942; // "BIN\0"

const COMPONENT_FLOAT = 5126;
const COMPONENT_UNSIGNED_INT = 5125;
const TARGET_ARRAY_BUFFER = 34962;
const TARGET_ELEMENT_ARRAY_BUFFER = 34963;
const MODE_TRIANGLES = 4;

export interface GlbOptions {
    /** Stored as the scene name. */
    name: string;
    /** Copied into `asset.extras`, for example the manifest id. */
    extras?: Record<string, unknown>;
}

export interface GltfJson {
    asset: Record<string, unknown>;
    extensionsUsed?: string[];
    scene: number;
    scenes: Array<{ name: string; nodes: number[] }>;
    nodes: Array<Record<string, unknown>>;
    meshes: Array<Record<string, unknown>>;
    materials: Array<Record<string, unknown>>;
    accessors: Array<Record<string, unknown>>;
    bufferViews: Array<Record<string, unknown>>;
    buffers: Array<{ byteLength: number }>;
    extensions?: Record<string, unknown>;
}

/** Appends 4-byte-aligned binary blobs and describes each as a buffer view. */
class BinaryBuilder {
    private readonly chunks: Uint8Array[] = [];
    private byteLength = 0;

    /** Returns the byte offset of the data just added. */
    add(data: ArrayBufferView): number {
        const offset = this.byteLength;
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        this.chunks.push(bytes);
        // Float32 and Uint32 data are already multiples of 4 bytes, so offsets stay aligned.
        this.byteLength += bytes.byteLength;
        return offset;
    }

    get length(): number {
        return this.byteLength;
    }

    toBytes(): Uint8Array {
        const out = new Uint8Array(this.byteLength);
        let cursor = 0;
        for (const chunk of this.chunks) {
            out.set(chunk, cursor);
            cursor += chunk.byteLength;
        }
        return out;
    }
}

function positionBounds(positions: Float32Array): { min: number[]; max: number[] } {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
        for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis], positions[i + axis]);
            max[axis] = Math.max(max[axis], positions[i + axis]);
        }
    }
    return { min, max };
}

/**
 * Quaternion `[x, y, z, w]` that turns glTF's light axis (-Z) onto `direction`.
 * glTF directional and spot lights shine along their local -Z.
 */
export function quaternionFromLightDirection(direction: Vector3Tuple): [number, number, number, number] {
    const length = Math.hypot(...direction);
    const d = [direction[0] / length, direction[1] / length, direction[2] / length];
    const from = [0, 0, -1];
    const dot = from[0] * d[0] + from[1] * d[1] + from[2] * d[2];
    if (dot < -0.999999) {
        // Pointing straight along +Z: any half-turn about a perpendicular axis works.
        return [0, 1, 0, 0];
    }
    // q = (from x d, 1 + from . d), normalised, is the shortest-arc rotation.
    const x = from[1] * d[2] - from[2] * d[1];
    const y = from[2] * d[0] - from[0] * d[2];
    const z = from[0] * d[1] - from[1] * d[0];
    const w = 1 + dot;
    const norm = Math.hypot(x, y, z, w);
    return [x / norm, y / norm, z / norm, w / norm];
}

function lightToGltf(light: SceneLightData): Record<string, unknown> {
    const common = {
        ...(light.label ? { name: light.label } : {}),
        color: [...light.color],
        intensity: light.intensity,
    };
    switch (light.type) {
        case 'directional':
            return { type: 'directional', ...common };
        case 'point':
            return { type: 'point', range: light.range, ...common };
        case 'spot':
            return {
                type: 'spot',
                range: light.range,
                spot: { innerConeAngle: light.innerConeAngle, outerConeAngle: light.outerConeAngle },
                ...common,
            };
    }
}

/** Writes the scene as a GLB and returns the file's bytes. */
export function writeGlb(scene: ProceduralScene, options: GlbOptions): Uint8Array {
    const bin = new BinaryBuilder();
    const bufferViews: GltfJson['bufferViews'] = [];
    const accessors: GltfJson['accessors'] = [];

    const addAccessor = (
        data: Float32Array | Uint32Array,
        target: number,
        componentType: number,
        type: 'VEC3' | 'VEC2' | 'SCALAR',
        extra: Record<string, unknown> = {},
    ): number => {
        const byteOffset = bin.add(data);
        bufferViews.push({ buffer: 0, byteOffset, byteLength: data.byteLength, target });
        const componentsPerElement = type === 'VEC3' ? 3 : type === 'VEC2' ? 2 : 1;
        accessors.push({
            bufferView: bufferViews.length - 1,
            componentType,
            count: data.length / componentsPerElement,
            type,
            ...extra,
        });
        return accessors.length - 1;
    };

    // Vertex data is written once per source mesh and shared by every material using it.
    const meshAccessors = scene.meshes.map((mesh: SceneMeshData) => {
        const bounds = positionBounds(mesh.positions);
        return {
            position: addAccessor(mesh.positions, TARGET_ARRAY_BUFFER, COMPONENT_FLOAT, 'VEC3', bounds),
            normal: addAccessor(mesh.normals, TARGET_ARRAY_BUFFER, COMPONENT_FLOAT, 'VEC3'),
            uv: addAccessor(mesh.uvs, TARGET_ARRAY_BUFFER, COMPONENT_FLOAT, 'VEC2'),
            indices: addAccessor(mesh.indices, TARGET_ELEMENT_ARRAY_BUFFER, COMPONENT_UNSIGNED_INT, 'SCALAR'),
        };
    });

    const usesEmission = scene.materials.some((m) => m.emissive.some((channel) => channel > 0));
    const materials = scene.materials.map((material) => {
        const peak = Math.max(...material.emissive);
        const gltfMaterial: Record<string, unknown> = {
            name: material.name,
            pbrMetallicRoughness: {
                baseColorFactor: [material.albedo[0], material.albedo[1], material.albedo[2], 1],
                metallicFactor: 0,
                roughnessFactor: 1,
            },
            doubleSided: material.doubleSided,
        };
        if (material.alphaCutoff !== undefined) {
            gltfMaterial.alphaMode = 'MASK';
            gltfMaterial.alphaCutoff = material.alphaCutoff;
        }
        if (peak > 0) {
            // emissiveFactor is limited to 0..1, so store the colour normalised and the scale separately.
            gltfMaterial.emissiveFactor = material.emissive.map((channel) => channel / peak);
            gltfMaterial.extensions = { KHR_materials_emissive_strength: { emissiveStrength: peak } };
        }
        return gltfMaterial;
    });

    // One glTF mesh per (source mesh, material) pair.
    const meshes: GltfJson['meshes'] = [];
    const meshIndexByPair = new Map<string, number>();
    const nodes: GltfJson['nodes'] = [];
    for (const instance of scene.instances) {
        const pairKey = `${instance.mesh}:${instance.material}`;
        let gltfMeshIndex = meshIndexByPair.get(pairKey);
        if (gltfMeshIndex === undefined) {
            const accessorSet = meshAccessors[instance.mesh];
            meshes.push({
                name: `${scene.meshes[instance.mesh].name}-${scene.materials[instance.material].name}`,
                primitives: [
                    {
                        attributes: { POSITION: accessorSet.position, NORMAL: accessorSet.normal, TEXCOORD_0: accessorSet.uv },
                        indices: accessorSet.indices,
                        material: instance.material,
                        mode: MODE_TRIANGLES,
                    },
                ],
            });
            gltfMeshIndex = meshes.length - 1;
            meshIndexByPair.set(pairKey, gltfMeshIndex);
        }
        nodes.push({ name: instance.name, mesh: gltfMeshIndex, matrix: [...instance.transform] });
    }

    const lights = scene.lights.map(lightToGltf);
    scene.lights.forEach((light, index) => {
        const node: Record<string, unknown> = {
            name: light.label ?? `light-${index}`,
            extensions: { KHR_lights_punctual: { light: index } },
        };
        if (light.type !== 'directional') node.translation = [...light.position];
        if (light.type !== 'point') node.rotation = quaternionFromLightDirection(light.direction);
        nodes.push(node);
    });

    const extensionsUsed: string[] = [];
    if (usesEmission) extensionsUsed.push('KHR_materials_emissive_strength');
    if (lights.length > 0) extensionsUsed.push('KHR_lights_punctual');

    const json: GltfJson = {
        asset: { version: '2.0', generator: '@pmndrs/ddgi-scenes', copyright: 'MIT, this repo', extras: options.extras ?? {} },
        ...(extensionsUsed.length > 0 ? { extensionsUsed } : {}),
        scene: 0,
        scenes: [{ name: options.name, nodes: nodes.map((_, index) => index) }],
        nodes,
        meshes,
        materials,
        accessors,
        bufferViews,
        buffers: [{ byteLength: bin.length }],
        ...(lights.length > 0 ? { extensions: { KHR_lights_punctual: { lights } } } : {}),
    };

    return packGlb(json, bin.toBytes());
}

/** Lays out the 12-byte header, a JSON chunk padded with spaces and a BIN chunk padded with zeros. */
function packGlb(json: GltfJson, binary: Uint8Array): Uint8Array {
    const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
    const jsonPadded = Math.ceil(jsonBytes.length / 4) * 4;
    const binPadded = Math.ceil(binary.length / 4) * 4;
    const totalLength = 12 + 8 + jsonPadded + 8 + binPadded;

    const out = new Uint8Array(totalLength);
    const view = new DataView(out.buffer);
    view.setUint32(0, GLB_MAGIC, true);
    view.setUint32(4, GLB_VERSION, true);
    view.setUint32(8, totalLength, true);

    view.setUint32(12, jsonPadded, true);
    view.setUint32(16, CHUNK_JSON, true);
    out.set(jsonBytes, 20);
    out.fill(0x20, 20 + jsonBytes.length, 20 + jsonPadded); // spaces

    const binHeader = 20 + jsonPadded;
    view.setUint32(binHeader, binPadded, true);
    view.setUint32(binHeader + 4, CHUNK_BIN, true);
    out.set(binary, binHeader + 8);
    return out;
}

/** Splits a GLB into its JSON and binary chunks. Throws on a malformed header. Used by tests and checks. */
export function readGlbChunks(glb: Uint8Array): { json: GltfJson; binary: Uint8Array } {
    const view = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
    if (view.getUint32(0, true) !== GLB_MAGIC) throw new Error('Not a GLB file: bad magic');
    if (view.getUint32(4, true) !== GLB_VERSION) throw new Error('Unsupported GLB version');
    if (view.getUint32(8, true) !== glb.byteLength) throw new Error('GLB length field does not match the file size');

    const jsonLength = view.getUint32(12, true);
    if (view.getUint32(16, true) !== CHUNK_JSON) throw new Error('First GLB chunk is not JSON');
    const json = JSON.parse(new TextDecoder().decode(glb.subarray(20, 20 + jsonLength))) as GltfJson;

    const binHeader = 20 + jsonLength;
    if (view.getUint32(binHeader + 4, true) !== CHUNK_BIN) throw new Error('Second GLB chunk is not BIN');
    const binLength = view.getUint32(binHeader, true);
    return { json, binary: glb.subarray(binHeader + 8, binHeader + 8 + binLength) };
}
