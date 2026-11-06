/**
 * The candle set for `sponza-candles` (T2.1), from Intel's 10,000-candles
 * add-on (package 726676; download-only, never committed).
 *
 *   node tools/assets/convert/sponza-candles.ts <package glTF> <output folder>
 *
 * Run by `pnpm assets:fetch` as the asset's `node` convert step
 * (tools/assets/sources.json). It writes two new files and never touches its
 * inputs:
 *
 * - `sponza-candle.glb`: one candle (wax, wick and flame, Intel's three
 *   materials), placed so its light is at the origin. 1,954 triangles.
 * - `sponza-candles.json`: the 10,000 candles' light positions (metres, the
 *   add-on's shared frame with the base Sponza), in the selection order
 *   below, with the light's colour and intensity.
 *
 * **Why one candle and positions, not the candles' mesh.** The package's
 * glTF is one 19.5 M-triangle mesh: the same candle copied 10,000 times,
 * each copy moved so its flame sits on one of 10,000 point lights. This
 * script checks that (every copy within CANDLE_COPY_TOLERANCE of the first,
 * moved by its light's offset) and fails if it isn't so. A scene that wants
 * N candles then loads one candle and N positions: only the chosen candles'
 * geometry, as T2.1's done-when asks, and as instances, as an app would draw
 * them.
 *
 * **The selection order** (DECISIONS 2026-11-06, T2.1): the candles are
 * sorted along a Morton (Z-order) curve through the box around their lights,
 * and listed in the bit-reversed order of their place on that curve: a van
 * der Corput sequence over the next power of two, skipping places past the
 * last candle. Any first N of the list are spread through the scene in
 * proportion to where Intel put the candles (dense where they are dense),
 * and the first 1,024 are `sponza-candles`' default. The order depends only
 * on the light positions, so it is the same on every machine.
 *
 * Plain Node with type stripping: no imports from the repo.
 */

import { readFile, mkdir, writeFile, open } from 'node:fs/promises';
import { dirname, join, resolve, relative, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

/** How far (metres) a copy's vertex may sit from the first candle's, moved by its light's offset. The package's copies agree to about 1 µm (float rounding at 10 m). */
export const CANDLE_COPY_TOLERANCE = 1e-5;
/** The add-on's candle count. */
export const PACKAGE_CANDLES = 10_000;
/** Bits per axis of the Morton code: 1,024 cells across the candles' box, about 2.5 cm along Sponza's long axis. */
const MORTON_BITS = 10;

const GLTF_FLOAT = 5126;
const GLTF_UNSIGNED_INT = 5125;
const GLTF_UNSIGNED_SHORT = 5123;
const GLTF_ARRAY_BUFFER = 34962;
const GLTF_ELEMENT_ARRAY_BUFFER = 34963;

type Vec3 = [number, number, number];
type Quaternion = [number, number, number, number];

interface GltfAccessor {
    bufferView: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
}

interface GltfNode {
    name?: string;
    mesh?: number;
    children?: number[];
    translation?: Vec3;
    rotation?: Quaternion;
    scale?: Vec3;
    extensions?: { KHR_lights_punctual?: { light: number } };
}

interface GltfDocument {
    scenes: { nodes: number[] }[];
    nodes: GltfNode[];
    meshes: { name?: string; primitives: { attributes: Record<string, number>; indices: number; material: number }[] }[];
    materials: Record<string, unknown>[];
    accessors: GltfAccessor[];
    bufferViews: { buffer: number; byteOffset?: number; byteLength: number }[];
    buffers: { uri: string; byteLength: number }[];
    extensions?: { KHR_lights_punctual?: { lights: { type: string; color?: Vec3; intensity?: number; range?: number }[] } };
}

/** The light all candles share, in the package's (three's) physical units: candela. */
export interface CandleLight {
    readonly type: 'point';
    readonly color: Vec3;
    readonly intensity: number;
    /** 0: no range window. The package's lights have none, so three's distance is 0 (DECISIONS 2026-11-06: lights with no range). */
    readonly range: number;
}

/** What `sponza-candles.json` holds. */
export interface CandleSetFile {
    readonly description: string;
    readonly source: string;
    readonly prototype: string;
    readonly count: number;
    readonly trianglesPerCandle: number;
    readonly order: string;
    readonly light: CandleLight;
    /** x, y, z of each candle's light, in the selection order. */
    readonly positions: number[];
    /** Each candle's number in the package (its light node `Instance<n>`), in the same order. */
    readonly packageIndices: number[];
}

// ---------------------------------------------------------------------------
// The selection order (pure, unit-tested in candle-order.test.ts)
// ---------------------------------------------------------------------------

/** Spreads the low 10 bits of `value` to every third bit. */
function spreadBits(value: number): number {
    let x = value & 0x3ff;
    x = (x | (x << 16)) & 0x30000ff;
    x = (x | (x << 8)) & 0x300f00f;
    x = (x | (x << 4)) & 0x30c30c3;
    x = (x | (x << 2)) & 0x9249249;
    return x;
}

/** The Morton code of a point in a box, MORTON_BITS per axis. */
export function mortonCode(point: readonly number[], min: readonly number[], max: readonly number[]): number {
    const cells = (1 << MORTON_BITS) - 1;
    const cell = (axis: number) => {
        const extent = max[axis] - min[axis];
        const share = extent > 0 ? (point[axis] - min[axis]) / extent : 0;
        return Math.min(cells, Math.max(0, Math.floor(share * (cells + 1))));
    };
    return spreadBits(cell(0)) | (spreadBits(cell(1)) << 1) | (spreadBits(cell(2)) << 2);
}

/** `value` with its lowest `bits` bits in reverse order. */
function reverseBits(value: number, bits: number): number {
    let result = 0;
    for (let bit = 0; bit < bits; bit++) result |= ((value >> bit) & 1) << (bits - 1 - bit);
    return result;
}

/**
 * The selection order (the file comment): indices into `positions` (x, y, z
 * each), sorted along the Morton curve (ties by index), then taken in the
 * bit-reversed order of their place on the curve.
 */
export function selectionOrder(positions: ArrayLike<number>): number[] {
    const count = positions.length / 3;
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let index = 0; index < count; index++) {
        for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis], positions[index * 3 + axis]);
            max[axis] = Math.max(max[axis], positions[index * 3 + axis]);
        }
    }
    const codes = Array.from({ length: count }, (_, index) => mortonCode([positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]], min, max));
    const alongCurve = Array.from({ length: count }, (_, index) => index).sort((a, b) => codes[a] - codes[b] || a - b);
    let bits = 0;
    while (1 << bits < count) bits++;
    const order: number[] = [];
    for (let slot = 0; slot < 1 << bits; slot++) {
        const place = reverseBits(slot, bits);
        if (place < count) order.push(alongCurve[place]);
    }
    return order;
}

// ---------------------------------------------------------------------------
// Reading the package
// ---------------------------------------------------------------------------

function rotate(q: Quaternion, v: Vec3): Vec3 {
    // v' = q v q*, written out (glTF quaternions are x, y, z, w).
    const [x, y, z, w] = q;
    const tx = 2 * (y * v[2] - z * v[1]);
    const ty = 2 * (z * v[0] - x * v[2]);
    const tz = 2 * (x * v[1] - y * v[0]);
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

/** A node's transform applied to a point: scale, then rotation, then translation (glTF's TRS). */
function transformPoint(node: GltfNode, point: Vec3): Vec3 {
    const scale = node.scale ?? [1, 1, 1];
    const scaled: Vec3 = [point[0] * scale[0], point[1] * scale[1], point[2] * scale[2]];
    const rotated = node.rotation ? rotate(node.rotation, scaled) : scaled;
    const translation = node.translation ?? [0, 0, 0];
    return [rotated[0] + translation[0], rotated[1] + translation[1], rotated[2] + translation[2]];
}

class PackageReader {
    readonly document: GltfDocument;
    private readonly binary: Buffer;

    // Plain fields, not parameter properties: Node's type stripping can't run those.
    constructor(document: GltfDocument, binary: Buffer) {
        this.document = document;
        this.binary = binary;
    }

    floats(accessorIndex: number, components: number): Float32Array {
        const accessor = this.document.accessors[accessorIndex];
        if (accessor.componentType !== GLTF_FLOAT) throw new Error(`accessor ${accessorIndex} is not float`);
        const view = this.document.bufferViews[accessor.bufferView];
        const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
        const copy = new Float32Array(accessor.count * components);
        new Uint8Array(copy.buffer).set(this.binary.subarray(start, start + copy.byteLength));
        return copy;
    }

    indices(accessorIndex: number): Uint32Array {
        const accessor = this.document.accessors[accessorIndex];
        if (accessor.componentType !== GLTF_UNSIGNED_INT) throw new Error(`accessor ${accessorIndex} is not 32-bit indices`);
        const view = this.document.bufferViews[accessor.bufferView];
        const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
        const copy = new Uint32Array(accessor.count);
        new Uint8Array(copy.buffer).set(this.binary.subarray(start, start + copy.byteLength));
        return copy;
    }
}

/** One primitive of the first candle, its light at the origin. */
interface CandlePart {
    readonly material: number;
    readonly positions: Float32Array;
    readonly normals: Float32Array;
    readonly uvs: Float32Array;
    readonly indices: Uint16Array;
}

/** The lights' world positions by package number (`Instance<n>`), and the light they share. */
function readLights(document: GltfDocument): { positions: Float64Array; light: CandleLight } {
    const lights = document.extensions?.KHR_lights_punctual?.lights ?? [];
    if (lights.length !== 1 || lights[0].type !== 'point') throw new Error(`expected one shared point light, found ${lights.length}`);
    const parents = new Map<number, number>();
    document.nodes.forEach((node, index) => node.children?.forEach((child) => parents.set(child, index)));
    const positions = new Float64Array(PACKAGE_CANDLES * 3).fill(Number.NaN);
    let found = 0;
    document.nodes.forEach((node, index) => {
        const match = /^Instance(\d+)$/.exec(node.name ?? '');
        if (!match || node.extensions?.KHR_lights_punctual === undefined) return;
        if (node.extensions.KHR_lights_punctual.light !== 0) throw new Error(`light node ${node.name} uses another light`);
        const number = Number(match[1]);
        if (!(number < PACKAGE_CANDLES) || !Number.isNaN(positions[number * 3])) throw new Error(`light node ${node.name} is out of range or repeated`);
        // A point light sits at its node's origin: walk the parents up.
        let point: Vec3 = [0, 0, 0];
        for (let at: number | undefined = index; at !== undefined; at = parents.get(at)) point = transformPoint(document.nodes[at], point);
        positions.set(point, number * 3);
        found++;
    });
    if (found !== PACKAGE_CANDLES) throw new Error(`expected ${PACKAGE_CANDLES} candle lights, found ${found}`);
    const light = lights[0];
    return { positions, light: { type: 'point', color: light.color ?? [1, 1, 1], intensity: light.intensity ?? 1, range: light.range ?? 0 } };
}

/**
 * The first candle of each primitive, moved so its light is at the origin,
 * after checking that every candle is a copy of it moved to its own light.
 * Each primitive holds the candles one after another (triangles and
 * vertices), in package order; the copies' vertices may be numbered
 * differently (the wicks' are), so they are compared corner by corner.
 */
function readCandle(reader: PackageReader, lights: Float64Array): CandlePart[] {
    const document = reader.document;
    const meshNodes = document.nodes.filter((node) => node.mesh !== undefined);
    if (meshNodes.length !== 1 || document.meshes.length !== 1) throw new Error('expected one mesh node with the candles');
    if (meshNodes[0].translation || meshNodes[0].rotation || meshNodes[0].scale) throw new Error('the candles mesh is expected to sit at the origin');
    return document.meshes[0].primitives.map((primitive, primitiveIndex) => {
        const indices = reader.indices(primitive.indices);
        const positions = reader.floats(primitive.attributes.POSITION, 3);
        const normals = reader.floats(primitive.attributes.NORMAL, 3);
        const uvs = reader.floats(primitive.attributes.TEXCOORD_0, 2);
        if (indices.length % (3 * PACKAGE_CANDLES) !== 0) throw new Error(`primitive ${primitiveIndex}: ${indices.length / 3} triangles is not a whole number per candle`);
        const cornersPerCandle = indices.length / PACKAGE_CANDLES;
        let worst = 0;
        for (let candle = 0; candle < PACKAGE_CANDLES; candle++) {
            for (let corner = 0; corner < cornersPerCandle; corner++) {
                const vertex = indices[candle * cornersPerCandle + corner];
                const first = indices[corner];
                for (let axis = 0; axis < 3; axis++) {
                    const relativeHere = positions[vertex * 3 + axis] - lights[candle * 3 + axis];
                    const relativeFirst = positions[first * 3 + axis] - lights[axis];
                    worst = Math.max(worst, Math.abs(relativeHere - relativeFirst));
                }
            }
        }
        if (!(worst <= CANDLE_COPY_TOLERANCE)) throw new Error(`primitive ${primitiveIndex}: a candle differs from the first by ${worst} m (more than ${CANDLE_COPY_TOLERANCE} m)`);
        // The first candle's own vertices: those its triangles use, renumbered from 0.
        let last = 0;
        for (let corner = 0; corner < cornersPerCandle; corner++) last = Math.max(last, indices[corner]);
        const vertexCount = last + 1;
        if (vertexCount > 0xffff) throw new Error(`primitive ${primitiveIndex}: ${vertexCount} vertices do not fit 16-bit indices`);
        const moved = new Float32Array(vertexCount * 3);
        for (let vertex = 0; vertex < vertexCount; vertex++) for (let axis = 0; axis < 3; axis++) moved[vertex * 3 + axis] = positions[vertex * 3 + axis] - lights[axis];
        console.log(`  primitive ${primitiveIndex} (material ${primitive.material}): ${cornersPerCandle / 3} triangles a candle, ${vertexCount} vertices; copies agree to ${worst.toExponential(2)} m`);
        return {
            material: primitive.material,
            positions: moved,
            normals: normals.slice(0, vertexCount * 3),
            uvs: uvs.slice(0, vertexCount * 2),
            indices: Uint16Array.from(indices.subarray(0, cornersPerCandle)),
        };
    });
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/** A .glb of the candle: one node, one mesh, a primitive per material, the package's materials as they are. */
export function candleGlb(parts: readonly CandlePart[], materials: readonly Record<string, unknown>[]): Uint8Array {
    const chunks: Uint8Array[] = [];
    const bufferViews: Record<string, unknown>[] = [];
    const accessors: Record<string, unknown>[] = [];
    let offset = 0;
    const addView = (bytes: Uint8Array, target: number): number => {
        bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.byteLength, target });
        chunks.push(bytes);
        offset += bytes.byteLength;
        // Every view starts on 4 bytes.
        const padding = (4 - (offset % 4)) % 4;
        if (padding > 0) {
            chunks.push(new Uint8Array(padding));
            offset += padding;
        }
        return bufferViews.length - 1;
    };
    const addFloats = (values: Float32Array, components: number, withBounds: boolean): number => {
        const view = addView(new Uint8Array(values.buffer, values.byteOffset, values.byteLength), GLTF_ARRAY_BUFFER);
        const accessor: Record<string, unknown> = { bufferView: view, componentType: GLTF_FLOAT, count: values.length / components, type: components === 3 ? 'VEC3' : 'VEC2' };
        if (withBounds) {
            const min = [Infinity, Infinity, Infinity];
            const max = [-Infinity, -Infinity, -Infinity];
            for (let index = 0; index < values.length; index++) {
                min[index % 3] = Math.min(min[index % 3], values[index]);
                max[index % 3] = Math.max(max[index % 3], values[index]);
            }
            accessor.min = min;
            accessor.max = max;
        }
        accessors.push(accessor);
        return accessors.length - 1;
    };
    const usedMaterials = [...new Set(parts.map((part) => part.material))];
    const primitives = parts.map((part) => {
        const position = addFloats(part.positions, 3, true);
        const normal = addFloats(part.normals, 3, false);
        const uv = addFloats(part.uvs, 2, false);
        const view = addView(new Uint8Array(part.indices.buffer, part.indices.byteOffset, part.indices.byteLength), GLTF_ELEMENT_ARRAY_BUFFER);
        accessors.push({ bufferView: view, componentType: GLTF_UNSIGNED_SHORT, count: part.indices.length, type: 'SCALAR' });
        return { attributes: { POSITION: position, NORMAL: normal, TEXCOORD_0: uv }, indices: accessors.length - 1, material: usedMaterials.indexOf(part.material) };
    });
    const document = {
        asset: { version: '2.0', generator: 'webgpu-ddgi tools/assets/convert/sponza-candles.ts' },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ name: 'candle', mesh: 0 }],
        meshes: [{ name: 'candle', primitives }],
        materials: usedMaterials.map((index) => materials[index]),
        accessors,
        bufferViews,
        buffers: [{ byteLength: offset }],
    };
    const json = new TextEncoder().encode(JSON.stringify(document));
    const jsonPadded = new Uint8Array(Math.ceil(json.byteLength / 4) * 4).fill(0x20);
    jsonPadded.set(json);
    const total = 12 + 8 + jsonPadded.byteLength + 8 + offset;
    const glb = new Uint8Array(total);
    const header = new DataView(glb.buffer);
    header.setUint32(0, 0x46546c67, true); // "glTF"
    header.setUint32(4, 2, true);
    header.setUint32(8, total, true);
    header.setUint32(12, jsonPadded.byteLength, true);
    header.setUint32(16, 0x4e4f534a, true); // "JSON"
    glb.set(jsonPadded, 20);
    const binStart = 20 + jsonPadded.byteLength;
    header.setUint32(binStart, offset, true);
    header.setUint32(binStart + 4, 0x004e4942, true); // "BIN\0"
    let at = binStart + 8;
    for (const chunk of chunks) {
        glb.set(chunk, at);
        at += chunk.byteLength;
    }
    return glb;
}

/** Metres to the micrometre: plenty for a light's place, and the same text on every machine. */
function micrometres(value: number): number {
    return Math.round(value * 1e6) / 1e6;
}

async function main(gltfPath: string, outputDirectory: string): Promise<void> {
    const document = JSON.parse(await readFile(gltfPath, 'utf8')) as GltfDocument;
    if (document.buffers.length !== 1) throw new Error('expected one buffer');
    const binaryPath = join(dirname(gltfPath), document.buffers[0].uri);
    // 650 MB: read it whole, once (only the candle's parts are kept).
    const handle = await open(binaryPath, 'r');
    const binary = Buffer.alloc(document.buffers[0].byteLength);
    await handle.read(binary, 0, binary.byteLength, 0);
    await handle.close();
    const reader = new PackageReader(document, binary);

    const { positions, light } = readLights(document);
    const parts = readCandle(reader, positions);
    const trianglesPerCandle = parts.reduce((sum, part) => sum + part.indices.length / 3, 0);
    const order = selectionOrder(positions);

    const ordered: number[] = [];
    for (const index of order) for (let axis = 0; axis < 3; axis++) ordered.push(micrometres(positions[index * 3 + axis]));
    const set: CandleSetFile = {
        description:
            "Intel's 10,000 candles (Sponza add-on 726676) as one candle and its light positions, in the selection order: any first N are spread through the scene (DECISIONS 2026-11-06, T2.1). Made by tools/assets/convert/sponza-candles.ts.",
        source: basename(gltfPath),
        prototype: 'sponza-candle.glb',
        count: PACKAGE_CANDLES,
        trianglesPerCandle,
        order: 'Morton curve over the candles\' box (10 bits an axis), taken in bit-reversed order of place on the curve (van der Corput over 16,384 places, skipping empty ones)',
        light,
        positions: ordered,
        packageIndices: order,
    };

    await mkdir(outputDirectory, { recursive: true });
    await writeFile(join(outputDirectory, 'sponza-candle.glb'), candleGlb(parts, document.materials));
    await writeFile(join(outputDirectory, 'sponza-candles.json'), `${JSON.stringify(set)}\n`);
    console.log(`  ${PACKAGE_CANDLES} candles, ${trianglesPerCandle} triangles each (${(PACKAGE_CANDLES * trianglesPerCandle).toLocaleString('en')} in the package); written to ${relative(process.cwd(), outputDirectory)}`);
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    const [gltfPath, outputDirectory] = process.argv.slice(2);
    if (!gltfPath || !outputDirectory) {
        console.error('usage: node tools/assets/convert/sponza-candles.ts <package glTF> <output folder>');
        process.exit(2);
    }
    main(resolve(gltfPath), resolve(outputDirectory)).catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
