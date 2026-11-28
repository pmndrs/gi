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

import { createTinyBvhBuilder, type GpuLayoutMeshBvh, type TinyBvhBuilder } from '../../packages/bvh/src/index.ts';
import type { MeshBvh } from '../../packages/core/src/platform/trace/mesh-bvh.ts';
import { buildMeshBvhFromTopology } from '../../packages/core/src/platform/trace/mesh-bvh-topology.ts';
import { readSceneMeshes } from './scene-meshes.ts';

const DEFAULT_SCENES = ['sponza', 'bistro-exterior', 'bistro-interior'];

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
    const result: SceneResult = { meshes: 0, triangles: 0, identical: 0, differing: [] };
    for (const { name, geometry } of readSceneMeshes(sceneId)) {
        const fromJavaScript = buildMeshBvhFromTopology(geometry, tinyBvh.build(geometry).topology);
        const fromWasm = tinyBvh.buildLayout(geometry).layout;
        const difference = firstDifference(fromWasm, fromJavaScript);
        result.meshes++;
        result.triangles += fromWasm.triangleCount;
        if (difference === null) result.identical++;
        else result.differing.push(`${name}: ${difference}`);
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
