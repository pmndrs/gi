/**
 * Generates the wind tree: a game-style tree for the M3 wind scenario
 * (docs/design/m3-moving.md section 8.1), written as one GLB.
 *
 *   node tools/assets/convert/wind-tree.ts --seed 20261109 <out.glb>
 *
 * It is a generated asset in tools/assets/sources.json (`generated`), so
 * `pnpm assets:fetch --only wind-tree` runs it and checks the file's sha256.
 * Ours, MIT. Nothing is downloaded.
 *
 * The tree: about 10 m tall (a trunk that leans a little, tapering to a thin
 * leader), three levels of side branches, and opaque leaf cards on the last
 * level. Alpha cutouts are M4's job, so the cards are plain two-triangle quads
 * with no texture; they block light as solid squares.
 *
 * The file has three meshes with three materials, so the viewer can sway them
 * differently:
 *   - `trunk`    the first level of tubes (the stem and its leader),
 *   - `branches` every other tube,
 *   - `leaves`   the cards (double-sided).
 *
 * Wind data: every vertex has COLOR_0 as float RGBA. RGB is 1,1,1 (neutral, so
 * a renderer that multiplies by vertex colour still shows the material's own
 * colour). The wind weight is in the ALPHA channel, 0 to 1:
 *   - 0 at the foot of the trunk, rising with height above the ground and with
 *     how many branchings the tube is away from the trunk (at most 0.6);
 *   - 0.7 to 1 on the leaves, higher for higher cards (always above any tube,
 *     so a viewer may add its extra flutter to cards by weight alone).
 * Every vertex of a card has the same weight, so a card sways as one piece.
 *
 * The same seed gives the same bytes: the only randomness is a seeded
 * generator, and the output is written in a fixed order.
 */

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export const TREE_HEIGHT_M = 10;
const TRUNK_BASE_RADIUS_M = 0.28;
const TRUNK_TIP_RADIUS_M = 0.03;
/** Side branches per level: how many grow from each parent branch. */
const CHILDREN_PER_BRANCH = [22, 6, 4] as const;
/** Radial sides of a tube at each branching level (the trunk is round, twigs are rougher). */
const SIDES_BY_LEVEL = [12, 7, 5, 4] as const;
const MAX_LEVEL = 3;
/** Leaf cards on each level-3 twig. */
const CARDS_PER_TWIG = 26;
const CARD_WIDTH_M = 0.24;
const CARD_HEIGHT_M = 0.3;
const BARK_COLOR: Vec3 = [0.1, 0.06, 0.03];
const LEAF_COLOR: Vec3 = [0.05, 0.2, 0.04];

type Vec3 = [number, number, number];

// ---------------------------------------------------------------------------
// Small vector helpers and the seeded random generator
// ---------------------------------------------------------------------------

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function normalize(a: Vec3): Vec3 {
    const length = Math.hypot(a[0], a[1], a[2]);
    return length > 0 ? scale(a, 1 / length) : [0, 1, 0];
}
/** Any unit vector perpendicular to `direction`. */
function anyPerpendicular(direction: Vec3): Vec3 {
    const helper: Vec3 = Math.abs(direction[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    return normalize(cross(direction, helper));
}
/** Rotates `v` around the unit `axis` by `angle` (Rodrigues). */
function rotate(v: Vec3, axis: Vec3, angle: number): Vec3 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1 - c)));
}

/** mulberry32: a small seeded generator with the same output on every machine. */
function makeRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ---------------------------------------------------------------------------
// Mesh pieces
// ---------------------------------------------------------------------------

/** Vertex data for one glTF mesh. COLOR_0 is RGBA; see the file header. */
export interface TreeMesh {
    name: string;
    positions: number[];
    normals: number[];
    uvs: number[];
    colors: number[];
    indices: number[];
}

function newMesh(name: string): TreeMesh {
    return { name, positions: [], normals: [], uvs: [], colors: [], indices: [] };
}

function pushVertex(mesh: TreeMesh, position: Vec3, normal: Vec3, u: number, v: number, weight: number): number {
    mesh.positions.push(...position);
    mesh.normals.push(...normal);
    mesh.uvs.push(u, v);
    mesh.colors.push(1, 1, 1, weight);
    return mesh.positions.length / 3 - 1;
}

/** Wind weight of a tube vertex: grows with height and with branching level, never past 0.6. */
function tubeWeight(height: number, level: number): number {
    return Math.min(0.6, 0.45 * Math.pow(Math.max(0, height) / TREE_HEIGHT_M, 1.3) + 0.05 * level);
}

/** Wind weight of a leaf card: always above any tube, higher up is higher. */
function leafWeight(height: number): number {
    return Math.min(1, 0.7 + 0.3 * Math.max(0, height) / TREE_HEIGHT_M);
}

interface Branch {
    points: Vec3[];
    radii: number[];
    level: number;
}

/** Adds a tapered tube along `branch`, with a fan cap at the tip (and at the foot, for the trunk). */
function addTube(mesh: TreeMesh, branch: Branch): void {
    const sides = SIDES_BY_LEVEL[branch.level];
    const { points, radii, level } = branch;
    let along = 0;
    let previousRingStart = -1;
    let firstRingStart = -1;
    let frameSide = anyPerpendicular(sub(points[1], points[0]));
    for (let i = 0; i < points.length; i++) {
        const before = points[Math.max(0, i - 1)];
        const after = points[Math.min(points.length - 1, i + 1)];
        const direction = normalize(sub(after, before));
        // Carry the frame along the branch so the rings do not twist.
        frameSide = normalize(sub(frameSide, scale(direction, dot(frameSide, direction))));
        const frameUp = cross(direction, frameSide);
        if (i > 0) along += Math.hypot(...sub(points[i], points[i - 1]));
        const ringStart = mesh.positions.length / 3;
        const weight = tubeWeight(points[i][1], level);
        for (let s = 0; s <= sides; s++) {
            const angle = (s / sides) * Math.PI * 2;
            const radial = add(scale(frameSide, Math.cos(angle)), scale(frameUp, Math.sin(angle)));
            pushVertex(mesh, add(points[i], scale(radial, radii[i])), radial, s / sides, along, weight);
        }
        if (previousRingStart >= 0) {
            for (let s = 0; s < sides; s++) {
                const a = previousRingStart + s;
                const b = ringStart + s;
                mesh.indices.push(a, a + 1, b, a + 1, b + 1, b);
            }
        }
        previousRingStart = ringStart;
        if (firstRingStart < 0) firstRingStart = ringStart;
    }
    // Tip: a fan to a point just beyond the last ring.
    const last = points.length - 1;
    const tipDirection = normalize(sub(points[last], points[last - 1]));
    addCap(mesh, previousRingStart, sides, add(points[last], scale(tipDirection, radii[last])), tipDirection, tubeWeight(points[last][1], level), false);
    if (level === 0) {
        // The foot: the trunk is closed underneath, so it is a solid thing standing on the ground.
        const footDirection = normalize(sub(points[1], points[0]));
        addCap(mesh, firstRingStart, sides, sub(points[0], scale(footDirection, radii[0])), scale(footDirection, -1), tubeWeight(points[0][1], level), true);
    }
}

/** A cone from a ring of `sides + 1` vertices (starting at `ringStart`) to `apex`. */
function addCap(mesh: TreeMesh, ringStart: number, sides: number, apex: Vec3, outward: Vec3, weight: number, flip: boolean): void {
    const apexIndex = pushVertex(mesh, apex, outward, 0.5, 0, weight);
    for (let s = 0; s < sides; s++) {
        const a = ringStart + s;
        if (flip) mesh.indices.push(a + 1, a, apexIndex);
        else mesh.indices.push(a, a + 1, apexIndex);
    }
}

// ---------------------------------------------------------------------------
// The tree
// ---------------------------------------------------------------------------

export interface WindTree {
    meshes: TreeMesh[];
    materials: { name: string; color: Vec3 }[];
}

/** Builds the tree for one seed. */
export function generateWindTree(seed: number): WindTree {
    const random = makeRandom(seed);
    const between = (low: number, high: number) => low + (high - low) * random();

    const trunk = newMesh('trunk');
    const branches = newMesh('branches');
    const leaves = newMesh('leaves');

    /**
     * Grows one branch from `start` heading `direction`: a gently bending
     * polyline that tapers from `radiusStart` to `radiusEnd` and is drawn as a
     * tube, then its children.
     */
    function grow(start: Vec3, direction: Vec3, length: number, radiusStart: number, radiusEnd: number, level: number): void {
        const segmentLength = level === 0 ? 0.5 : level === 1 ? 0.4 : 0.3;
        const segments = Math.max(3, Math.round(length / segmentLength));
        const points: Vec3[] = [start];
        const radii: number[] = [radiusStart];
        let heading = direction;
        for (let i = 1; i <= segments; i++) {
            // A random bend, and a pull upwards that is stronger for the stem and weaker for twigs.
            const bendAxis = rotate(anyPerpendicular(heading), heading, between(0, Math.PI * 2));
            heading = normalize(rotate(heading, bendAxis, between(0.02, level === 0 ? 0.08 : 0.2)));
            heading = normalize(add(heading, [0, level === 0 ? 0.05 : 0.04, 0]));
            points.push(add(points[i - 1], scale(heading, length / segments)));
            const t = i / segments;
            radii.push(radiusStart + (radiusEnd - radiusStart) * Math.pow(t, level === 0 ? 0.8 : 1));
        }
        addTube(level === 0 ? trunk : branches, { points, radii, level });

        if (level === MAX_LEVEL) {
            addLeafCards(points);
            return;
        }
        const count = CHILDREN_PER_BRANCH[level];
        const childLevel = level + 1;
        for (let c = 0; c < count; c++) {
            // Children start along the outer part of the parent, spread around it by the golden angle.
            const t = level === 0 ? 0.3 + 0.68 * ((c + random() * 0.6) / count) : 0.25 + 0.7 * ((c + random() * 0.6) / count);
            const position = Math.min(points.length - 1, t * (points.length - 1));
            const index = Math.floor(position);
            const fraction = position - index;
            const next = Math.min(points.length - 1, index + 1);
            const origin = add(scale(points[index], 1 - fraction), scale(points[next], fraction));
            const parentRadius = radii[index] * (1 - fraction) + radii[next] * fraction;
            const parentDirection = normalize(sub(points[next], points[Math.max(0, next === index ? index - 1 : index)]));
            const around = c * 2.399963 + between(-0.4, 0.4);
            const pitch = between(0.7, 1.15) - (level === 0 ? 0.25 * t : 0);
            const side = rotate(anyPerpendicular(parentDirection), parentDirection, around);
            const childDirection = normalize(rotate(parentDirection, cross(parentDirection, side), -pitch));
            // Lower side branches of the stem are longer, which gives a rounded crown.
            const crown = level === 0 ? Math.sin(Math.PI * Math.min(1, 0.15 + 0.85 * t)) * 1.05 + 0.15 : 1;
            const childLength = length * (level === 0 ? 0.3 * crown : between(0.42, 0.6)) * (level === 0 ? 1 : 1);
            const childRadius = Math.max(0.02, parentRadius * between(0.5, 0.65));
            grow(origin, childDirection, Math.max(0.5, childLength), childRadius, Math.max(0.008, childRadius * 0.2), childLevel);
        }
    }

    /** Cards along a twig, scattered around it, each facing a random way that favours the sky. */
    function addLeafCards(twig: Vec3[]): void {
        for (let c = 0; c < CARDS_PER_TWIG; c++) {
            const position = ((c + random()) / CARDS_PER_TWIG) * (twig.length - 1);
            const index = Math.min(twig.length - 2, Math.floor(position));
            const fraction = position - index;
            const along = add(scale(twig[index], 1 - fraction), scale(twig[index + 1], fraction));
            const twigDirection = normalize(sub(twig[index + 1], twig[index]));
            // Offset from the twig, so the cards form a clump rather than a line.
            const offsetDirection = rotate(anyPerpendicular(twigDirection), twigDirection, between(0, Math.PI * 2));
            const centre = add(along, scale(offsetDirection, between(0.03, 0.16)));
            const facing = normalize([between(-1, 1), between(0.1, 1.4), between(-1, 1)]);
            const widthAxis = anyPerpendicular(facing);
            const heightAxis = cross(facing, widthAxis);
            const spin = between(0, Math.PI * 2);
            const u = rotate(widthAxis, facing, spin);
            const v = rotate(heightAxis, facing, spin);
            const weight = leafWeight(centre[1]);
            const halfU = scale(u, CARD_WIDTH_M / 2);
            const halfV = scale(v, CARD_HEIGHT_M / 2);
            const corners: Vec3[] = [sub(sub(centre, halfU), halfV), sub(add(centre, halfU), halfV), add(add(centre, halfU), halfV), add(sub(centre, halfU), halfV)];
            const uvs: [number, number][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
            const first = leaves.positions.length / 3;
            corners.forEach((corner, i) => pushVertex(leaves, corner, facing, uvs[i][0], uvs[i][1], weight));
            leaves.indices.push(first, first + 1, first + 2, first, first + 2, first + 3);
        }
    }

    const lean = between(-0.05, 0.05);
    grow([0, 0, 0], normalize([lean, 1, -lean * 0.5]), TREE_HEIGHT_M * 0.8, TRUNK_BASE_RADIUS_M, TRUNK_TIP_RADIUS_M, 0);

    return {
        meshes: [trunk, branches, leaves],
        materials: [
            { name: 'bark', color: BARK_COLOR },
            { name: 'bark', color: BARK_COLOR },
            { name: 'leaves', color: LEAF_COLOR },
        ],
    };
}

/** Triangles in a tree, all meshes. */
export function triangleCount(tree: WindTree): number {
    return tree.meshes.reduce((total, mesh) => total + mesh.indices.length / 3, 0);
}

// ---------------------------------------------------------------------------
// GLB writer
// ---------------------------------------------------------------------------

/** Writes the tree as a binary glTF 2.0 file. Same input, same bytes. */
export function writeWindTreeGlb(tree: WindTree): Uint8Array {
    const chunks: Uint8Array[] = [];
    let byteLength = 0;
    const bufferViews: object[] = [];
    const accessors: object[] = [];

    function addView(data: ArrayBufferView, target: number): number {
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        const padding = (4 - (bytes.length % 4)) % 4;
        bufferViews.push({ buffer: 0, byteOffset: byteLength, byteLength: bytes.length, target });
        chunks.push(bytes, new Uint8Array(padding));
        byteLength += bytes.length + padding;
        return bufferViews.length - 1;
    }
    function addAccessor(view: number, componentType: number, count: number, type: string, extra: object = {}): number {
        accessors.push({ bufferView: view, componentType, count, type, ...extra });
        return accessors.length - 1;
    }

    const ARRAY_BUFFER = 34962;
    const ELEMENT_ARRAY_BUFFER = 34963;
    const FLOAT = 5126;
    const UNSIGNED_INT = 5125;

    const meshes = tree.meshes.map((mesh, meshIndex) => {
        const positions = new Float32Array(mesh.positions);
        const minimum = [Infinity, Infinity, Infinity];
        const maximum = [-Infinity, -Infinity, -Infinity];
        for (let i = 0; i < positions.length; i++) {
            minimum[i % 3] = Math.min(minimum[i % 3], positions[i]);
            maximum[i % 3] = Math.max(maximum[i % 3], positions[i]);
        }
        const vertexCount = positions.length / 3;
        const position = addAccessor(addView(positions, ARRAY_BUFFER), FLOAT, vertexCount, 'VEC3', { min: minimum, max: maximum });
        const normal = addAccessor(addView(new Float32Array(mesh.normals), ARRAY_BUFFER), FLOAT, vertexCount, 'VEC3');
        const uv = addAccessor(addView(new Float32Array(mesh.uvs), ARRAY_BUFFER), FLOAT, vertexCount, 'VEC2');
        const color = addAccessor(addView(new Float32Array(mesh.colors), ARRAY_BUFFER), FLOAT, vertexCount, 'VEC4');
        const index = addAccessor(addView(new Uint32Array(mesh.indices), ELEMENT_ARRAY_BUFFER), UNSIGNED_INT, mesh.indices.length, 'SCALAR');
        return {
            name: mesh.name,
            primitives: [{ attributes: { POSITION: position, NORMAL: normal, TEXCOORD_0: uv, COLOR_0: color }, indices: index, material: meshIndex === 2 ? 1 : 0 }],
        };
    });

    const json = {
        asset: { version: '2.0', generator: 'webgpu-ddgi tools/assets/convert/wind-tree.ts' },
        scene: 0,
        scenes: [{ name: 'wind-tree', nodes: meshes.map((_, i) => i) }],
        nodes: meshes.map((mesh, i) => ({ name: mesh.name, mesh: i })),
        meshes,
        // Bark and leaves. Both are dielectric (metallic 0) and rough. Leaves are double-sided: a card has two faces.
        materials: [
            { name: 'bark', pbrMetallicRoughness: { baseColorFactor: [...BARK_COLOR, 1], metallicFactor: 0, roughnessFactor: 0.95 } },
            { name: 'leaves', doubleSided: true, pbrMetallicRoughness: { baseColorFactor: [...LEAF_COLOR, 1], metallicFactor: 0, roughnessFactor: 0.8 } },
        ],
        accessors,
        bufferViews,
        buffers: [{ byteLength }],
    };

    const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
    const jsonPadding = (4 - (jsonBytes.length % 4)) % 4;
    const jsonChunkLength = jsonBytes.length + jsonPadding;
    const total = 12 + 8 + jsonChunkLength + 8 + byteLength;
    const output = new Uint8Array(total);
    const header = new DataView(output.buffer);
    header.setUint32(0, 0x46546c67, true); // "glTF"
    header.setUint32(4, 2, true);
    header.setUint32(8, total, true);
    header.setUint32(12, jsonChunkLength, true);
    header.setUint32(16, 0x4e4f534a, true); // "JSON"
    output.set(jsonBytes, 20);
    output.fill(0x20, 20 + jsonBytes.length, 20 + jsonChunkLength); // JSON pads with spaces
    const binHeader = 20 + jsonChunkLength;
    header.setUint32(binHeader, byteLength, true);
    header.setUint32(binHeader + 4, 0x004e4942, true); // "BIN\0"
    let offset = binHeader + 8;
    for (const chunk of chunks) {
        output.set(chunk, offset);
        offset += chunk.length;
    }
    return output;
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    const args = process.argv.slice(2);
    const seedFlag = args.indexOf('--seed');
    const seed = Number(args[seedFlag + 1]);
    const outputPath = args.filter((_, i) => i !== seedFlag && i !== seedFlag + 1)[0];
    if (seedFlag < 0 || !Number.isInteger(seed) || !outputPath) {
        console.error('usage: node tools/assets/convert/wind-tree.ts --seed <whole number> <out.glb>');
        process.exit(2);
    }
    const tree = generateWindTree(seed);
    await mkdir(dirname(resolve(outputPath)), { recursive: true });
    await writeFile(outputPath, writeWindTreeGlb(tree));
    console.log(`wind-tree: seed ${seed}, ${triangleCount(tree)} triangles, written to ${outputPath}`);
}
