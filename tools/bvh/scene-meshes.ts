/// <reference types="node" />
/**
 * Reads a test scene's meshes in Node, for the CPU tools in this folder: every
 * model part of the scene's manifest, each mesh in object space (instances
 * don't matter to a mesh tree). Needs the downloaded models (`pnpm assets:fetch`).
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import type { ModelPart, SceneManifest } from '../../scenes/src/manifest.ts';
import { extractGltfGeometry, parseGlb, type BenchMeshGeometry, type GltfJson } from '../gpu/pages/trace-bench/gltf-geometry.ts';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '..', '..');
/** Folders searched for downloaded models, in order (as the trace bench does). */
const DOWNLOAD_FOLDERS = ['assets/downloads', 'assets/downloads.bak'];

export interface SceneMesh {
    /** `model path` and the mesh's number in that model, for messages. */
    readonly name: string;
    readonly geometry: BenchMeshGeometry;
}

function readManifest(sceneId: string): SceneManifest {
    const file = join(REPOSITORY_ROOT, 'scenes', 'manifests', `${sceneId}.json`);
    if (!existsSync(file)) throw new Error(`no manifest ${file}`);
    return JSON.parse(readFileSync(file, 'utf8')) as SceneManifest;
}

function findModelFile(part: ModelPart): string {
    for (const folder of DOWNLOAD_FOLDERS) {
        const file = join(REPOSITORY_ROOT, folder, part.asset, part.path);
        if (existsSync(file)) return file;
    }
    throw new Error(`model ${part.asset}/${part.path} not found; run "pnpm assets:fetch --only ${part.asset}"`);
}

/** Every mesh of a model file, in object space. */
function readModelMeshes(file: string): BenchMeshGeometry[] {
    if (file.toLowerCase().endsWith('.glb')) {
        const bytes = readFileSync(file);
        const { gltf, binary } = parseGlb(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        if (!binary) throw new Error(`${file} has no binary chunk`);
        return extractGltfGeometry(gltf, [binary]).meshes;
    }
    const gltf = JSON.parse(readFileSync(file, 'utf8')) as GltfJson & { buffers?: { uri?: string }[] };
    const buffers = (gltf.buffers ?? []).map((buffer) => {
        if (!buffer.uri || buffer.uri.startsWith('data:')) throw new Error(`${file}: only external buffers are read`);
        return new Uint8Array(readFileSync(join(dirname(file), decodeURIComponent(buffer.uri))));
    });
    return extractGltfGeometry(gltf, buffers).meshes;
}

/** Every mesh of every model part of a scene (manifest id), in manifest order. */
export function readSceneMeshes(sceneId: string): SceneMesh[] {
    const parts = readManifest(sceneId).parts.filter((part): part is ModelPart => part.kind === 'model');
    if (parts.length === 0) throw new Error(`${sceneId} has no model parts`);
    return parts.flatMap((part) => readModelMeshes(findModelFile(part)).map((geometry, index) => ({ name: `${part.path} mesh ${index}`, geometry })));
}
