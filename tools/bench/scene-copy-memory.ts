/**
 * Scene-copy memory on the real scenes: the bytes the platform's vertex
 * buffer holds for each scene's meshes, in the chosen layout (octahedral
 * normals, 16-bit indices where they fit) and in the two alternatives that
 * were measured against it (float normals; 32-bit indices everywhere).
 *
 *     node tools/bench/scene-copy-memory.ts [scene id ...]
 *
 * Defaults to sponza, bistro-exterior and bistro-interior. Every model part
 * of a scene is read (Sponza is the base package plus the curtains), and the
 * output lists the files, then the scene's total and each part. Models must be
 * downloaded first (`pnpm assets:fetch`). Meshes are packed with the
 * platform's own `packMesh`, so the chosen-layout number is exactly what
 * `PlatformScene` uploads; the alternatives add the bytes those layouts
 * would take on top. One platform mesh per glTF primitive, as three.js's
 * GLTFLoader makes one Mesh per primitive. CPU only: no GPU, no bench lock.
 *
 * The table buffer (instances, materials, lights) is left out: it is under
 * 1 MB for these scenes.
 */

import { readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NONE } from '../../packages/ddgi-core/src/platform/scene/layout.ts';
import { packMesh } from '../../packages/ddgi-core/src/platform/scene/mesh-packing.ts';

const repository = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

interface GltfAccessor {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
    normalized?: boolean;
    sparse?: unknown;
}

interface GltfJson {
    meshes: { primitives: { attributes: Record<string, number>; indices?: number; mode?: number }[] }[];
    accessors: GltfAccessor[];
    bufferViews: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
    buffers: { uri?: string; byteLength: number }[];
}

interface Gltf {
    json: GltfJson;
    buffers: Buffer[];
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };
const COMPONENT_BYTES: Record<number, number> = { 5121: 1, 5123: 2, 5125: 4, 5126: 4 };

function loadGltf(path: string): Gltf {
    const bytes = readFileSync(path);
    if (path.endsWith('.gltf')) {
        const json = JSON.parse(bytes.toString('utf8')) as GltfJson;
        const buffers = json.buffers.map((buffer) => {
            if (!buffer.uri || buffer.uri.startsWith('data:')) throw new Error(`${path}: only external .bin buffers are supported`);
            return readFileSync(join(dirname(path), decodeURIComponent(buffer.uri)));
        });
        return { json, buffers };
    }
    // GLB: 12-byte header, then a JSON chunk and a BIN chunk, each with an 8-byte chunk header.
    const jsonLength = bytes.readUInt32LE(12);
    const json = JSON.parse(bytes.subarray(20, 20 + jsonLength).toString('utf8')) as GltfJson;
    return { json, buffers: [bytes.subarray(20 + jsonLength + 8)] };
}

/** Reads an accessor as floats (positions, normals, uvs) or as integers (indices). */
function readAccessor(gltf: Gltf, index: number): number[] {
    const accessor = gltf.json.accessors[index];
    if (accessor.sparse || accessor.bufferView === undefined) throw new Error(`accessor ${index}: sparse or bufferless accessors are not supported`);
    const size = COMPONENT_BYTES[accessor.componentType];
    if (!size || (accessor.componentType !== 5126 && accessor.normalized)) throw new Error(`accessor ${index}: component type ${accessor.componentType} is not supported`);
    const components = COMPONENTS[accessor.type];
    const view = gltf.json.bufferViews[accessor.bufferView];
    const buffer = gltf.buffers[view.buffer];
    const stride = view.byteStride ?? components * size;
    const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const values = new Array<number>(accessor.count * components);
    for (let element = 0; element < accessor.count; element++) {
        for (let component = 0; component < components; component++) {
            const at = start + element * stride + component * size;
            values[element * components + component] =
                accessor.componentType === 5126
                    ? buffer.readFloatLE(at)
                    : accessor.componentType === 5125
                      ? buffer.readUInt32LE(at)
                      : accessor.componentType === 5123
                        ? buffer.readUInt16LE(at)
                        : buffer.readUInt8(at);
        }
    }
    return values;
}

interface ModelFile {
    /** The asset id from tools/assets/sources.json. */
    asset: string;
    /** The file, relative to the asset's download folder. */
    path: string;
    absolutePath: string;
}

/**
 * Every model part of a scene, in manifest order. Sponza is several Intel
 * packages (the base plus add-ons), so a scene's memory is the sum over its parts.
 */
function modelFiles(sceneId: string): ModelFile[] {
    const manifest = JSON.parse(readFileSync(join(repository, 'scenes/manifests', `${sceneId}.json`), 'utf8')) as {
        parts: { kind: string; asset?: string; path?: string }[];
    };
    const files: ModelFile[] = [];
    for (const part of manifest.parts) {
        if (part.kind !== 'model') continue;
        if (!part.asset || !part.path) throw new Error(`${sceneId}: a model part has no asset or path`);
        files.push({ asset: part.asset, path: part.path, absolutePath: join(repository, 'assets/downloads', part.asset, part.path) });
    }
    if (files.length === 0) throw new Error(`${sceneId}: the manifest has no model part`);
    return files;
}

interface Memory {
    label: string;
    meshes: number;
    vertices: number;
    triangles: number;
    chosenBytes: number;
    floatNormalsBytes: number;
    indices32Bytes: number;
}

const emptyMemory = (label: string): Memory => ({ label, meshes: 0, vertices: 0, triangles: 0, chosenBytes: 0, floatNormalsBytes: 0, indices32Bytes: 0 });

/** The scene copy's bytes for one glTF file. */
function measureFile(file: ModelFile): Memory {
    const gltf = loadGltf(file.absolutePath);
    const result = emptyMemory(file.asset);
    for (const mesh of gltf.json.meshes) {
        for (const primitive of mesh.primitives) {
            if ((primitive.mode ?? 4) !== 4) continue; // triangles only, as the platform takes them
            const attribute = (name: string) => (primitive.attributes[name] !== undefined ? Float32Array.from(readAccessor(gltf, primitive.attributes[name])) : undefined);
            const positions = attribute('POSITION')!;
            const vertexCount = positions.length / 3;
            const indices =
                primitive.indices !== undefined ? Uint32Array.from(readAccessor(gltf, primitive.indices)) : Uint32Array.from({ length: vertexCount }, (_, index) => index);
            const packed = packMesh({ positions, indices, normals: attribute('NORMAL'), uvs: attribute('TEXCOORD_0'), uvs1: attribute('TEXCOORD_1') });
            const bytes = packed.words.byteLength;
            result.meshes++;
            result.vertices += vertexCount;
            result.triangles += packed.triangleCount;
            result.chosenBytes += bytes;
            // Float normals: 12 bytes per vertex instead of 4.
            result.floatNormalsBytes += bytes + (packed.offsets.normals !== NONE ? vertexCount * 8 : 0);
            // 32-bit indices: 3 words per triangle instead of ceil(3T / 2).
            const indexCount = packed.triangleCount * 3;
            result.indices32Bytes += bytes + (packed.indices16 ? (indexCount - Math.ceil(indexCount / 2)) * 4 : 0);
        }
    }
    return result;
}

/** One row per model part, then the scene's total (the sum of its parts). */
function measureScene(sceneId: string): { parts: Memory[]; total: Memory } {
    const parts = modelFiles(sceneId).map(measureFile);
    const total = emptyMemory(parts.length > 1 ? `${sceneId} (total of ${parts.length} parts)` : sceneId);
    for (const part of parts) {
        total.meshes += part.meshes;
        total.vertices += part.vertices;
        total.triangles += part.triangles;
        total.chosenBytes += part.chosenBytes;
        total.floatNormalsBytes += part.floatNormalsBytes;
        total.indices32Bytes += part.indices32Bytes;
    }
    return { parts, total };
}

const megabytes = (bytes: number) => (bytes / 1e6).toFixed(1);
const LABEL_WIDTH = 40;

function formatRow(row: Memory): string {
    return [
        row.label.padEnd(LABEL_WIDTH),
        String(row.meshes).padStart(7),
        String(row.vertices).padStart(9),
        String(row.triangles).padStart(11),
        megabytes(row.chosenBytes).padStart(10),
        megabytes(row.floatNormalsBytes).padStart(17),
        megabytes(row.indices32Bytes).padStart(18),
    ].join(' ');
}

const scenes = process.argv.slice(2).length > 0 ? process.argv.slice(2) : ['sponza', 'bistro-exterior', 'bistro-interior'];
// Say exactly which files were read, so a scene's line is not mistaken for one package's.
for (const sceneId of scenes) {
    for (const file of modelFiles(sceneId)) console.log(`${sceneId}: reads ${file.asset}/${file.path}`);
}
console.log('');
console.log(`${'scene (and its parts)'.padEnd(LABEL_WIDTH)}  meshes  vertices   triangles  chosen MB  float-normals MB  32-bit-indices MB`);
for (const sceneId of scenes) {
    const { parts, total } = measureScene(sceneId);
    console.log(formatRow(total));
    if (parts.length > 1) for (const part of parts) console.log(formatRow({ ...part, label: `  ${part.label}` }));
}
