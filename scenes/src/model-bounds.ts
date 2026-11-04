/// <reference types="node" />
/**
 * World-space bounds of glTF models, read from the file's JSON alone (accessor
 * min/max and the node tree), so a 270 MB model costs a few kilobytes of I/O.
 *
 *   pnpm --filter @pmndrs/ddgi-scenes exec tsx src/model-bounds.ts [--downloads <dir>]
 *
 * Prints, for every model part in the manifests, the model's bounds after the
 * manifest transform, and whether the suggested probe volume covers them.
 */

import { open, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadManifests } from './export';

export interface Bounds {
    min: [number, number, number];
    max: [number, number, number];
}

interface LooseNode {
    mesh?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
}

interface LooseGltf {
    scene?: number;
    scenes?: Array<{ nodes?: number[] }>;
    nodes?: LooseNode[];
    meshes?: Array<{ primitives: Array<{ attributes: { POSITION?: number } }> }>;
    accessors?: Array<{ min?: number[]; max?: number[] }>;
}

type Matrix = number[];

const IDENTITY: Matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Column-major a * b. */
export function multiplyMatrices(a: Matrix, b: Matrix): Matrix {
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
        }
    }
    return out;
}

function nodeMatrix(node: LooseNode): Matrix {
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

/** Bounds of every mesh in the default scene, through the node hierarchy. Null when there is no geometry. */
export function gltfWorldBounds(gltf: LooseGltf, rootTransform: Matrix = IDENTITY): Bounds | null {
    const min: Bounds['min'] = [Infinity, Infinity, Infinity];
    const max: Bounds['max'] = [-Infinity, -Infinity, -Infinity];
    const nodes = gltf.nodes ?? [];

    const visit = (index: number, parent: Matrix): void => {
        const node = nodes[index];
        const world = multiplyMatrices(parent, nodeMatrix(node));
        if (node.mesh !== undefined) {
            for (const primitive of gltf.meshes?.[node.mesh].primitives ?? []) {
                const accessor = gltf.accessors?.[primitive.attributes.POSITION ?? -1];
                if (!accessor?.min || !accessor.max) continue;
                for (const x of [accessor.min[0], accessor.max[0]]) {
                    for (const y of [accessor.min[1], accessor.max[1]]) {
                        for (const z of [accessor.min[2], accessor.max[2]]) {
                            for (let axis = 0; axis < 3; axis++) {
                                const value = world[axis] * x + world[4 + axis] * y + world[8 + axis] * z + world[12 + axis];
                                min[axis] = Math.min(min[axis], value);
                                max[axis] = Math.max(max[axis], value);
                            }
                        }
                    }
                }
            }
        }
        for (const child of node.children ?? []) visit(child, world);
    };

    for (const root of gltf.scenes?.[gltf.scene ?? 0]?.nodes ?? []) visit(root, rootTransform);
    return Number.isFinite(min[0]) ? { min, max } : null;
}

/** Reads the JSON of a .gltf file, or only the JSON chunk of a .glb file. */
export async function readModelJson(path: string): Promise<LooseGltf> {
    if (path.toLowerCase().endsWith('.gltf')) return JSON.parse(await readFile(path, 'utf8')) as LooseGltf;
    const file = await open(path, 'r');
    try {
        const header = Buffer.alloc(20);
        await file.read(header, 0, 20, 0);
        if (header.readUInt32LE(0) !== 0x46546c67) throw new Error(`${path} is not a .glb file`);
        const jsonLength = header.readUInt32LE(12);
        const json = Buffer.alloc(jsonLength);
        await file.read(json, 0, jsonLength, 20);
        return JSON.parse(json.toString('utf8')) as LooseGltf;
    } finally {
        await file.close();
    }
}

/** Manifest transform (position, XYZ Euler degrees, uniform scale) as a column-major matrix. */
export function modelTransformMatrix(transform: { position: number[]; rotationDeg: number[]; scale: number }): Matrix {
    const [rx, ry, rz] = transform.rotationDeg.map((degrees) => (degrees * Math.PI) / 180);
    const [cx, cy, cz] = [Math.cos(rx), Math.cos(ry), Math.cos(rz)];
    const [sx, sy, sz] = [Math.sin(rx), Math.sin(ry), Math.sin(rz)];
    const s = transform.scale;
    // Three.js 'XYZ' order: R = Rx * Ry * Rz.
    return [
        cy * cz * s, (cx * sz + sx * sy * cz) * s, (sx * sz - cx * sy * cz) * s, 0,
        -cy * sz * s, (cx * cz - sx * sy * sz) * s, (sx * cz + cx * sy * sz) * s, 0,
        sy * s, -sx * cy * s, cx * cy * s, 0,
        transform.position[0], transform.position[1], transform.position[2], 1,
    ];
}

async function main(): Promise<void> {
    const argv = process.argv.slice(2);
    const downloadsIndex = argv.indexOf('--downloads');
    const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
    const downloads = downloadsIndex >= 0 ? resolve(argv[downloadsIndex + 1]) : join(repoRoot, 'assets', 'downloads');

    for (const manifest of await loadManifests()) {
        for (const part of manifest.parts) {
            if (part.kind !== 'model') continue;
            const path = join(downloads, part.asset, part.path);
            try {
                const bounds = gltfWorldBounds(await readModelJson(path), modelTransformMatrix(part.transform));
                const fmt = (v: number[]) => v.map((n) => n.toFixed(2)).join(', ');
                console.log(`${manifest.id} / ${part.asset}`);
                console.log(`  min [${fmt(bounds?.min ?? [])}]  max [${fmt(bounds?.max ?? [])}]`);
                const volume = manifest.volume;
                const covers = bounds
                    ? [0, 1, 2].every(
                          (axis) =>
                              volume.center[axis] - volume.size[axis] / 2 <= bounds.min[axis] + 1e-6 &&
                              volume.center[axis] + volume.size[axis] / 2 >= bounds.max[axis] - 1e-6,
                      )
                    : false;
                console.log(`  volume [center ${fmt(volume.center)}, size ${fmt(volume.size)}] covers the model: ${covers}`);
            } catch (error) {
                console.log(`${manifest.id} / ${part.asset}: ${error instanceof Error ? error.message : String(error)}`);
            }
        }
    }
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
