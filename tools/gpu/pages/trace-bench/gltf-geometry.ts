/**
 * Reads only the triangles and instance transforms out of a glTF / GLB file.
 *
 * The benchmark needs geometry and nothing else: no textures, no materials.
 * Three's GLTFLoader would decode every texture (the Bistro GLBs carry 4K
 * ones), which wastes minutes and memory on a 6 GB GPU, so this reads the JSON
 * and the binary buffers directly.
 *
 * What it produces:
 * - one "mesh" per glTF mesh: all of its triangle primitives merged into one
 *   position array and one index array (so a mesh with three materials is one
 *   BVH, as a platform mesh would be);
 * - one "instance" per scene-graph node that uses a mesh, carrying that node's
 *   world matrix. Several nodes sharing a glTF mesh share the BVH.
 *
 * Not handled, on purpose: skins, morph targets, sparse accessors, and
 * compressed meshes (Draco, meshopt). They throw a clear error.
 */

/** Column-major 4x4. */
export type Matrix4 = number[];

export interface BenchMeshGeometry {
    /** `x, y, z` per vertex. */
    positions: Float32Array;
    /** Three vertex indices per triangle, always 32-bit. */
    indices: Uint32Array;
}

export interface BenchInstance {
    /** Position in {@link GltfGeometry.meshes}. */
    meshIndex: number;
    objectToWorld: Matrix4;
}

export interface GltfGeometry {
    meshes: BenchMeshGeometry[];
    instances: BenchInstance[];
    /** Primitives left out (lines, points, strips, fans), for the record. */
    skippedPrimitives: number;
}

interface GltfAccessor {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
    sparse?: unknown;
}

interface GltfBufferView {
    buffer: number;
    byteOffset?: number;
    byteLength: number;
    byteStride?: number;
}

interface GltfPrimitive {
    attributes: { POSITION?: number };
    indices?: number;
    mode?: number;
}

interface GltfNode {
    mesh?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
}

export interface GltfJson {
    scene?: number;
    scenes?: { nodes?: number[] }[];
    nodes?: GltfNode[];
    meshes?: { primitives: GltfPrimitive[] }[];
    accessors?: GltfAccessor[];
    bufferViews?: GltfBufferView[];
    buffers?: { uri?: string; byteLength: number }[];
    extensionsRequired?: string[];
}

const COMPONENT = {
    byte: 5120,
    unsignedByte: 5121,
    short: 5122,
    unsignedShort: 5123,
    unsignedInt: 5125,
    float: 5126,
} as const;

const MODE_TRIANGLES = 4;
/**
 * Extensions that change how geometry is stored or instanced. A file that requires
 * one cannot be read here. Texture and material extensions (EXT_texture_webp, ...)
 * do not matter, since textures are never loaded.
 */
const UNSUPPORTED_GEOMETRY_EXTENSIONS = new Set([
    'KHR_draco_mesh_compression',
    'EXT_meshopt_compression',
    'KHR_mesh_quantization',
    'EXT_mesh_gpu_instancing',
]);

export const IDENTITY_MATRIX: Matrix4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Column-major `a * b`. */
export function multiplyMatrices(a: Matrix4, b: Matrix4): Matrix4 {
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
        }
    }
    return out;
}

/** A node's local matrix: its `matrix`, or translation * rotation * scale. */
export function nodeLocalMatrix(node: GltfNode): Matrix4 {
    if (node.matrix) return node.matrix;
    const [tx, ty, tz] = node.translation ?? [0, 0, 0];
    const [qx, qy, qz, qw] = node.rotation ?? [0, 0, 0, 1];
    const [sx, sy, sz] = node.scale ?? [1, 1, 1];
    return [
        (1 - 2 * (qy * qy + qz * qz)) * sx, 2 * (qx * qy + qz * qw) * sx, 2 * (qx * qz - qy * qw) * sx, 0,
        2 * (qx * qy - qz * qw) * sy, (1 - 2 * (qx * qx + qz * qz)) * sy, 2 * (qy * qz + qx * qw) * sy, 0,
        2 * (qx * qz + qy * qw) * sz, 2 * (qy * qz - qx * qw) * sz, (1 - 2 * (qx * qx + qy * qy)) * sz, 0,
        tx, ty, tz, 1,
    ];
}

/**
 * The manifest's model transform (position, Euler XYZ degrees, uniform scale)
 * as a column-major matrix. Same convention as three's `Object3D` with
 * rotation order 'XYZ', which is what the scene check page applies.
 */
export function modelTransformMatrix(transform: {
    position: readonly number[];
    rotationDeg: readonly number[];
    scale: number;
}): Matrix4 {
    const [rx, ry, rz] = transform.rotationDeg.map((degrees) => (degrees * Math.PI) / 180);
    const [cx, cy, cz] = [Math.cos(rx), Math.cos(ry), Math.cos(rz)];
    const [sx, sy, sz] = [Math.sin(rx), Math.sin(ry), Math.sin(rz)];
    const s = transform.scale;
    return [
        cy * cz * s, (cx * sz + sx * sy * cz) * s, (sx * sz - cx * sy * cz) * s, 0,
        -cy * sz * s, (cx * cz - sx * sy * sz) * s, (sx * cz + cx * sy * sz) * s, 0,
        sy * s, -sx * cy * s, cx * cy * s, 0,
        transform.position[0], transform.position[1], transform.position[2], 1,
    ];
}

function componentSize(componentType: number): number {
    switch (componentType) {
        case COMPONENT.byte:
        case COMPONENT.unsignedByte:
            return 1;
        case COMPONENT.short:
        case COMPONENT.unsignedShort:
            return 2;
        case COMPONENT.unsignedInt:
        case COMPONENT.float:
            return 4;
        default:
            throw new Error(`unsupported accessor component type ${componentType}`);
    }
}

const COMPONENTS_PER_TYPE: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

/** Reads one component with the right width and signedness. */
function readComponent(view: DataView, byteOffset: number, componentType: number): number {
    switch (componentType) {
        case COMPONENT.byte:
            return view.getInt8(byteOffset);
        case COMPONENT.unsignedByte:
            return view.getUint8(byteOffset);
        case COMPONENT.short:
            return view.getInt16(byteOffset, true);
        case COMPONENT.unsignedShort:
            return view.getUint16(byteOffset, true);
        case COMPONENT.unsignedInt:
            return view.getUint32(byteOffset, true);
        case COMPONENT.float:
            return view.getFloat32(byteOffset, true);
        default:
            throw new Error(`unsupported accessor component type ${componentType}`);
    }
}

/**
 * Copies an accessor's data into a new, tightly packed typed array of numbers
 * (`Float32Array` for float data, `Uint32Array` otherwise).
 */
function readAccessor(gltf: GltfJson, buffers: readonly Uint8Array[], accessorIndex: number): Float32Array | Uint32Array {
    const accessor = gltf.accessors?.[accessorIndex];
    if (!accessor) throw new Error(`accessor ${accessorIndex} does not exist`);
    if (accessor.sparse) throw new Error(`accessor ${accessorIndex} is sparse, which this reader does not support`);
    if (accessor.bufferView === undefined) throw new Error(`accessor ${accessorIndex} has no buffer view`);
    const bufferView = gltf.bufferViews?.[accessor.bufferView];
    if (!bufferView) throw new Error(`buffer view ${accessor.bufferView} does not exist`);
    const buffer = buffers[bufferView.buffer];
    if (!buffer) throw new Error(`buffer ${bufferView.buffer} was not loaded`);

    const components = COMPONENTS_PER_TYPE[accessor.type];
    if (components === undefined) throw new Error(`unsupported accessor type ${accessor.type}`);
    const bytesPerComponent = componentSize(accessor.componentType);
    const elementBytes = components * bytesPerComponent;
    const stride = bufferView.byteStride ?? elementBytes;
    const start = buffer.byteOffset + (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const isFloat = accessor.componentType === COMPONENT.float;

    // Fast path: tightly packed, 4-byte aligned 32-bit data copies straight over.
    if (stride === elementBytes && bytesPerComponent === 4 && start % 4 === 0) {
        const length = accessor.count * components;
        return isFloat
            ? new Float32Array(buffer.buffer, start, length).slice()
            : new Uint32Array(buffer.buffer, start, length).slice();
    }

    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const base = start - buffer.byteOffset;
    const out = isFloat ? new Float32Array(accessor.count * components) : new Uint32Array(accessor.count * components);
    for (let element = 0; element < accessor.count; element++) {
        for (let component = 0; component < components; component++) {
            out[element * components + component] = readComponent(view, base + element * stride + component * bytesPerComponent, accessor.componentType);
        }
    }
    return out;
}

/** The merged geometry of one glTF mesh, or null when it has no triangles. */
function readMesh(gltf: GltfJson, buffers: readonly Uint8Array[], meshIndex: number, counters: { skipped: number }): BenchMeshGeometry | null {
    const mesh = gltf.meshes?.[meshIndex];
    if (!mesh) throw new Error(`mesh ${meshIndex} does not exist`);

    const positionParts: Float32Array[] = [];
    const indexParts: Uint32Array[] = [];
    let vertexBase = 0;
    let indexTotal = 0;
    for (const primitive of mesh.primitives) {
        if ((primitive.mode ?? MODE_TRIANGLES) !== MODE_TRIANGLES) {
            counters.skipped++;
            continue;
        }
        const positionAccessor = primitive.attributes.POSITION;
        if (positionAccessor === undefined) continue;
        const accessor = gltf.accessors?.[positionAccessor];
        if (accessor?.componentType !== COMPONENT.float) throw new Error('only float POSITION data is supported');
        const positions = readAccessor(gltf, buffers, positionAccessor) as Float32Array;
        const vertexCount = positions.length / 3;

        let indices: Uint32Array;
        if (primitive.indices === undefined) {
            indices = new Uint32Array(vertexCount);
            for (let i = 0; i < vertexCount; i++) indices[i] = i;
        } else {
            indices = readAccessor(gltf, buffers, primitive.indices) as Uint32Array;
        }
        if (indices.length % 3 !== 0) throw new Error(`mesh ${meshIndex} has an index count that is not a multiple of 3`);
        if (indices.length === 0) continue;

        if (vertexBase !== 0) for (let i = 0; i < indices.length; i++) indices[i] += vertexBase;
        positionParts.push(positions);
        indexParts.push(indices);
        vertexBase += vertexCount;
        indexTotal += indices.length;
    }
    if (indexTotal === 0) return null;

    if (positionParts.length === 1) return { positions: positionParts[0], indices: indexParts[0] };
    const positions = new Float32Array(vertexBase * 3);
    const indices = new Uint32Array(indexTotal);
    let positionCursor = 0;
    for (const part of positionParts) {
        positions.set(part, positionCursor);
        positionCursor += part.length;
    }
    let indexCursor = 0;
    for (const part of indexParts) {
        indices.set(part, indexCursor);
        indexCursor += part.length;
    }
    return { positions, indices };
}

/**
 * Walks the default scene and returns every mesh once and every use of a mesh
 * as an instance. `rootTransform` is applied on top of the node transforms.
 */
export function extractGltfGeometry(
    gltf: GltfJson,
    buffers: readonly Uint8Array[],
    rootTransform: Matrix4 = IDENTITY_MATRIX,
): GltfGeometry {
    for (const extension of gltf.extensionsRequired ?? []) {
        if (UNSUPPORTED_GEOMETRY_EXTENSIONS.has(extension)) {
            throw new Error(`the file requires the glTF extension ${extension}, which this reader cannot decode`);
        }
    }
    const nodes = gltf.nodes ?? [];
    const counters = { skipped: 0 };
    const meshes: BenchMeshGeometry[] = [];
    const meshIndexByGltfMesh = new Map<number, number | null>();
    const instances: BenchInstance[] = [];

    const benchMeshFor = (gltfMesh: number): number | null => {
        if (meshIndexByGltfMesh.has(gltfMesh)) return meshIndexByGltfMesh.get(gltfMesh) ?? null;
        const geometry = readMesh(gltf, buffers, gltfMesh, counters);
        const index = geometry ? meshes.push(geometry) - 1 : null;
        meshIndexByGltfMesh.set(gltfMesh, index);
        return index;
    };

    // Iterative walk: a deep hierarchy must not hit the JS call-stack limit. Nodes are pushed in
    // reverse so they pop in file order, which keeps the instance list repeatable.
    const roots = gltf.scenes?.[gltf.scene ?? 0]?.nodes ?? [];
    const stack: { node: number; parent: Matrix4 }[] = [...roots].reverse().map((node) => ({ node, parent: rootTransform }));
    while (stack.length > 0) {
        const { node: nodeIndex, parent } = stack.pop()!;
        const node = nodes[nodeIndex];
        if (!node) throw new Error(`node ${nodeIndex} does not exist`);
        const world = multiplyMatrices(parent, nodeLocalMatrix(node));
        if (node.mesh !== undefined) {
            const meshIndex = benchMeshFor(node.mesh);
            if (meshIndex !== null) instances.push({ meshIndex, objectToWorld: world });
        }
        const children = node.children ?? [];
        for (let child = children.length - 1; child >= 0; child--) stack.push({ node: children[child], parent: world });
    }

    return { meshes, instances, skippedPrimitives: counters.skipped };
}

const GLB_MAGIC = 0x46546c67;
const GLB_CHUNK_JSON = 0x4e4f534a;
const GLB_CHUNK_BIN = 0x004e4942;

/** Splits a GLB file into its JSON and its binary chunk (a view into the same memory). */
export function parseGlb(file: ArrayBuffer): { gltf: GltfJson; binary: Uint8Array | null } {
    const view = new DataView(file);
    if (file.byteLength < 20 || view.getUint32(0, true) !== GLB_MAGIC) throw new Error('not a .glb file');
    let gltf: GltfJson | null = null;
    let binary: Uint8Array | null = null;
    let cursor = 12;
    while (cursor + 8 <= file.byteLength) {
        const chunkLength = view.getUint32(cursor, true);
        const chunkType = view.getUint32(cursor + 4, true);
        const chunkStart = cursor + 8;
        if (chunkType === GLB_CHUNK_JSON) {
            gltf = JSON.parse(new TextDecoder().decode(new Uint8Array(file, chunkStart, chunkLength))) as GltfJson;
        } else if (chunkType === GLB_CHUNK_BIN && binary === null) {
            binary = new Uint8Array(file, chunkStart, chunkLength);
        }
        cursor = chunkStart + chunkLength;
    }
    if (!gltf) throw new Error('the .glb file has no JSON chunk');
    return { gltf, binary };
}

/** Downloads a model and returns its geometry, with `rootTransform` applied. */
export async function fetchGltfGeometry(url: string, rootTransform: Matrix4): Promise<GltfGeometry> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not download ${url}: HTTP ${response.status}`);

    if (url.toLowerCase().endsWith('.glb')) {
        const { gltf, binary } = parseGlb(await response.arrayBuffer());
        if (!binary) throw new Error('the .glb file has no binary chunk');
        return extractGltfGeometry(gltf, [binary], rootTransform);
    }

    const gltf = (await response.json()) as GltfJson;
    const buffers: Uint8Array[] = [];
    for (const entry of gltf.buffers ?? []) {
        if (!entry.uri) throw new Error('a .gltf buffer has no uri');
        if (entry.uri.startsWith('data:')) throw new Error('data: URIs are not supported by this reader');
        const bufferResponse = await fetch(new URL(entry.uri, new URL(url, location.href)));
        if (!bufferResponse.ok) throw new Error(`could not download ${entry.uri}: HTTP ${bufferResponse.status}`);
        buffers.push(new Uint8Array(await bufferResponse.arrayBuffer()));
    }
    return extractGltfGeometry(gltf, buffers, rootTransform);
}
