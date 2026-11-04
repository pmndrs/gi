/// <reference types="node" />
/**
 * How much of a floor the sun reaches, found by casting a ray from sample
 * points toward the sun and checking whether any triangle is in the way. Used to
 * pick a sun direction for a scene with a courtyard or an atrium, where a
 * badly chosen sun leaves the floor in shadow.
 *
 *   pnpm --filter @pmndrs/ddgi-scenes exec tsx src/sun-visibility.ts <scene id>
 *       [--grid x0,x1,dx,z0,z1,dz,y] [--sun x,y,z]...
 *
 * Without `--sun` it tries the scene manifest's sun and prints the lit
 * fraction. Alpha cutouts are treated as solid and the test is brute force
 * (a few seconds per direction for Sponza), which is fine for a design check.
 */

import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadManifests } from './export';
import { modelTransformMatrix, multiplyMatrices, readModelJson } from './model-bounds';

type Vec = [number, number, number];

/** True when nothing blocks the segment from `origin` along `toSun` (a unit vector) over `maxDistance`. */
export function isLit(triangles: Float32Array, origin: Vec, toSun: Vec, maxDistance = 1000): boolean {
    for (let t = 0; t < triangles.length; t += 9) {
        const e1x = triangles[t + 3] - triangles[t];
        const e1y = triangles[t + 4] - triangles[t + 1];
        const e1z = triangles[t + 5] - triangles[t + 2];
        const e2x = triangles[t + 6] - triangles[t];
        const e2y = triangles[t + 7] - triangles[t + 1];
        const e2z = triangles[t + 8] - triangles[t + 2];
        const px = toSun[1] * e2z - toSun[2] * e2y;
        const py = toSun[2] * e2x - toSun[0] * e2z;
        const pz = toSun[0] * e2y - toSun[1] * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        if (Math.abs(det) < 1e-12) continue;
        const inverse = 1 / det;
        const tx = origin[0] - triangles[t];
        const ty = origin[1] - triangles[t + 1];
        const tz = origin[2] - triangles[t + 2];
        const u = (tx * px + ty * py + tz * pz) * inverse;
        if (u < 0 || u > 1) continue;
        const qx = ty * e1z - tz * e1y;
        const qy = tz * e1x - tx * e1z;
        const qz = tx * e1y - ty * e1x;
        const v = (toSun[0] * qx + toSun[1] * qy + toSun[2] * qz) * inverse;
        if (v < 0 || u + v > 1) continue;
        const distance = (e2x * qx + e2y * qy + e2z * qz) * inverse;
        if (distance > 0.01 && distance < maxDistance) return false;
    }
    return true;
}

/** Fraction of `points` that see the sun. `sunDirection` is the way the light travels. */
export function litFraction(triangles: Float32Array, points: readonly Vec[], sunDirection: Vec): number {
    const length = Math.hypot(...sunDirection);
    const toSun: Vec = [-sunDirection[0] / length, -sunDirection[1] / length, -sunDirection[2] / length];
    let lit = 0;
    for (const point of points) if (isLit(triangles, point, toSun)) lit++;
    return lit / points.length;
}

const COMPONENT_BYTES: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };

interface Accessor {
    bufferView: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
}

/** All triangles of a .gltf or .glb with external or embedded .bin, in world space, as flat xyz triples. */
export async function loadWorldTriangles(path: string, rootTransform: number[]): Promise<Float32Array> {
    const gltf = (await readModelJson(path)) as unknown as {
        scene?: number;
        scenes: Array<{ nodes: number[] }>;
        nodes: Array<{ mesh?: number; children?: number[]; matrix?: number[]; translation?: number[]; rotation?: number[]; scale?: number[] }>;
        meshes: Array<{ primitives: Array<{ attributes: { POSITION: number }; indices?: number; mode?: number }> }>;
        accessors: Accessor[];
        bufferViews: Array<{ buffer: number; byteOffset?: number; byteStride?: number }>;
        buffers: Array<{ uri?: string }>;
    };
    if (!path.toLowerCase().endsWith('.gltf')) throw new Error('Only .gltf files with an external .bin are supported here.');
    const buffers = await Promise.all(gltf.buffers.map(async (buffer) => new Uint8Array(await readFile(join(dirname(path), decodeURIComponent(buffer.uri ?? ''))))));

    const read = (accessorIndex: number): number[] => {
        const accessor = gltf.accessors[accessorIndex];
        const view = gltf.bufferViews[accessor.bufferView];
        const components = accessor.type === 'VEC3' ? 3 : 1;
        const bytes = COMPONENT_BYTES[accessor.componentType];
        const stride = view.byteStride ?? bytes * components;
        const base = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
        const data = new DataView(buffers[view.buffer].buffer, buffers[view.buffer].byteOffset);
        const out: number[] = [];
        for (let i = 0; i < accessor.count; i++) {
            for (let c = 0; c < components; c++) {
                const at = base + i * stride + c * bytes;
                out.push(
                    accessor.componentType === 5126 ? data.getFloat32(at, true)
                    : accessor.componentType === 5125 ? data.getUint32(at, true)
                    : accessor.componentType === 5123 ? data.getUint16(at, true)
                    : data.getUint8(at),
                );
            }
        }
        return out;
    };

    const triangles: number[] = [];
    const visit = (index: number, parent: number[]): void => {
        const node = gltf.nodes[index];
        const local = node.matrix ?? localMatrix(node);
        const world = multiplyMatrices(parent, local);
        if (node.mesh !== undefined) {
            for (const primitive of gltf.meshes[node.mesh].primitives) {
                if (primitive.mode !== undefined && primitive.mode !== 4) continue;
                const positions = read(primitive.attributes.POSITION);
                const indices = primitive.indices !== undefined ? read(primitive.indices) : Array.from({ length: positions.length / 3 }, (_, i) => i);
                for (const vertex of indices) {
                    const x = positions[vertex * 3];
                    const y = positions[vertex * 3 + 1];
                    const z = positions[vertex * 3 + 2];
                    triangles.push(
                        world[0] * x + world[4] * y + world[8] * z + world[12],
                        world[1] * x + world[5] * y + world[9] * z + world[13],
                        world[2] * x + world[6] * y + world[10] * z + world[14],
                    );
                }
            }
        }
        for (const child of node.children ?? []) visit(child, world);
    };
    for (const root of gltf.scenes[gltf.scene ?? 0].nodes) visit(root, rootTransform);
    return Float32Array.from(triangles);
}

function localMatrix(node: { translation?: number[]; rotation?: number[]; scale?: number[] }): number[] {
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

/** Evenly spaced points on a horizontal grid: `x0,x1,dx,z0,z1,dz,y`. */
export function floorGrid(spec: string): Vec[] {
    const [x0, x1, dx, z0, z1, dz, y] = spec.split(',').map(Number);
    const points: Vec[] = [];
    for (let x = x0; x <= x1 + 1e-9; x += dx) for (let z = z0; z <= z1 + 1e-9; z += dz) points.push([x, y, z]);
    return points;
}

async function main(): Promise<void> {
    const argv = process.argv.slice(2);
    const sceneId = argv[0];
    const flagValues = (flag: string): string[] => argv.flatMap((arg, i) => (arg === flag ? [argv[i + 1]] : []));
    const manifest = (await loadManifests()).find((candidate) => candidate.id === sceneId);
    if (!manifest) throw new Error(`Usage: sun-visibility.ts <scene id> [--grid ...] [--sun x,y,z]...`);
    const part = manifest.parts.find((candidate) => candidate.kind === 'model');
    if (!part || part.kind !== 'model') throw new Error(`${sceneId} has no model part`);

    const downloads = process.env.DDGI_DOWNLOADS_DIR?.split(';')[0] ?? resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'assets', 'downloads');
    const triangles = await loadWorldTriangles(join(downloads, part.asset, part.path), modelTransformMatrix(part.transform));
    console.log(`${triangles.length / 9} triangles loaded`);

    const grid = floorGrid(flagValues('--grid')[0] ?? '-10,10.5,1,-3,3,0.75,0.02');
    const suns: Vec[] = flagValues('--sun').map((text) => text.split(',').map(Number) as Vec);
    if (suns.length === 0) {
        for (const light of manifest.lights) if (light.type === 'directional') suns.push([...light.direction]);
    }
    for (const sun of suns) {
        const started = Date.now();
        const fraction = litFraction(triangles, grid, sun);
        const elevation = (Math.asin(-sun[1] / Math.hypot(...sun)) * 180) / Math.PI;
        console.log(`sun direction [${sun.join(', ')}] (elevation ${elevation.toFixed(0)} deg): ${(fraction * 100).toFixed(0)}% of ${grid.length} floor points lit  (${((Date.now() - started) / 1000).toFixed(1)} s)`);
    }
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
