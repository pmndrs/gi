/// <reference types="node" />
/**
 * Checks, on real scenes, that tinybvh's WASM writes exactly the GPU layout the
 * JavaScript conversion makes (stage B0b). For every mesh of each scene it
 * builds tinybvh's tree twice:
 *
 * - `build` (the tree's shape) followed by ddgi-core's `buildMeshBvhFromTopology`;
 * - `buildLayout` (the layout written inside the WASM);
 *
 * and compares the nodes, triangles, parents and box byte for byte, plus the
 * node, leaf and depth counts. tinybvh is deterministic, so both builds see the
 * same tree (`mesh-bvh-tinybvh-layout.test.ts` checks that).
 *
 *   node tools/bvh/layout-identity.ts                  Sponza and both Bistro scenes
 *   node tools/bvh/layout-identity.ts sponza ...       the scenes named (manifest ids)
 *
 * Needs the downloaded models (`pnpm assets:fetch`). Prints one line per scene
 * and exits with 1 if any mesh differs or a scene can't be loaded. CPU only, and
 * done in well under a minute.
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { createTinyBvhBuilder, type GpuLayoutMeshBvh, type TinyBvhBuilder } from '../../packages/bvh/src/index.ts';
import type { MeshBvh } from '../../packages/ddgi-core/src/platform/trace/mesh-bvh.ts';
import { buildMeshBvhFromTopology } from '../../packages/ddgi-core/src/platform/trace/mesh-bvh-topology.ts';
import type { ModelPart, SceneManifest } from '../../scenes/src/manifest.ts';
import { extractGltfGeometry, parseGlb, type BenchMeshGeometry, type GltfJson } from '../gpu/pages/trace-bench/gltf-geometry.ts';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '..', '..');
const DEFAULT_SCENES = ['sponza', 'bistro-exterior', 'bistro-interior'];
/** Folders searched for downloaded models, in order (as the trace bench does). */
const DOWNLOAD_FOLDERS = ['assets/downloads', 'assets/downloads.bak'];

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

/** Every mesh of a model file, in object space (instances don't matter here). */
function readMeshes(file: string): BenchMeshGeometry[] {
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

function sameBytes(a: ArrayBuffer | ArrayBufferView, b: ArrayBuffer | ArrayBufferView): boolean {
    const bytesOf = (data: ArrayBuffer | ArrayBufferView) =>
        data instanceof ArrayBuffer ? Buffer.from(data) : Buffer.from(data.buffer, data.byteOffset, data.byteLength);
    return bytesOf(a).equals(bytesOf(b));
}

/** The first field that differs between the two layouts, or null if none does. */
function firstDifference(fromWasm: GpuLayoutMeshBvh, fromJavaScript: MeshBvh): string | null {
    const counts = ['nodeCount', 'triangleCount', 'maxDepth', 'leafCount'] as const;
    for (const count of counts) {
        if (fromWasm[count] !== fromJavaScript[count]) return `${count} ${fromWasm[count]} against ${fromJavaScript[count]}`;
    }
    const arrays = ['nodes', 'triangles', 'parents', 'bounds'] as const;
    for (const array of arrays) {
        if (!sameBytes(fromWasm[array], fromJavaScript[array])) return `${array} bytes`;
    }
    return null;
}

interface SceneResult {
    meshes: number;
    triangles: number;
    identical: number;
    differing: string[];
}

function checkScene(tinyBvh: TinyBvhBuilder, sceneId: string): SceneResult {
    const parts = readManifest(sceneId).parts.filter((part): part is ModelPart => part.kind === 'model');
    if (parts.length === 0) throw new Error(`${sceneId} has no model parts`);
    const result: SceneResult = { meshes: 0, triangles: 0, identical: 0, differing: [] };
    for (const part of parts) {
        readMeshes(findModelFile(part)).forEach((mesh, index) => {
            const fromJavaScript = buildMeshBvhFromTopology(mesh, tinyBvh.build(mesh).topology);
            const fromWasm = tinyBvh.buildLayout(mesh).layout;
            const difference = firstDifference(fromWasm, fromJavaScript);
            result.meshes++;
            result.triangles += fromWasm.triangleCount;
            if (difference === null) result.identical++;
            else result.differing.push(`${part.path} mesh ${index}: ${difference}`);
        });
    }
    return result;
}

async function main(): Promise<void> {
    const scenes = process.argv.length > 2 ? process.argv.slice(2) : DEFAULT_SCENES;
    const tinyBvh = await createTinyBvhBuilder();
    let failed = false;
    for (const sceneId of scenes) {
        try {
            const result = checkScene(tinyBvh, sceneId);
            console.log(
                `${sceneId}: ${result.meshes} meshes, ${result.triangles} triangles; ` +
                    `${result.identical} identical, ${result.differing.length} differing`,
            );
            for (const line of result.differing.slice(0, 10)) console.log(`  ${line}`);
            if (result.differing.length > 0) failed = true;
        } catch (error) {
            console.log(`${sceneId}: not checked: ${error instanceof Error ? error.message : String(error)}`);
            failed = true;
        }
    }
    process.exit(failed ? 1 : 0);
}

await main();
