/// <reference types="node" />
/**
 * Tree quality of the CPU builders and twins on real scenes, without a GPU:
 * the SAH cost of every mesh's tree, summed over the scene (each mesh's cost
 * relative to its root box times that box's half-area, as the trace bench
 * sums them), against tinybvh's default build (B0, the tracer's trees).
 *
 *   node tools/bvh/tree-quality.ts                   Sponza
 *   node tools/bvh/tree-quality.ts sponza-full ...   the scenes named (manifest ids)
 *
 * Builders: tinybvh's default (B0) and its `BuildHPLOC` (the oracle for B2),
 * and the CPU twins of the GPU builders: the LBVH (B1) and H-PLOC with B1's
 * 30-bit codes and with tinybvh's 63-bit codes. H-PLOC is also built with
 * one-triangle leaves (no SAH leaf collapse), the shape tinybvh's `BuildHPLOC`
 * makes, which is what "matches tinybvh" compares. CPU only; a few minutes for
 * Sponza, most of it tinybvh's scalar `BuildHPLOC`.
 */

import { treeSahCost } from '../../packages/bvh/src/layout-check.ts';
import { buildHplocOnCpu } from '../../packages/bvh/src/hploc-cpu.ts';
import { buildLbvhOnCpu } from '../../packages/bvh/src/lbvh-cpu.ts';
import { createTinyBvhBuilder, type GpuLayoutMeshBvh, type TinyBvhBuilder, type TinyBvhInput } from '../../packages/bvh/src/index.ts';
import { readSceneMeshes } from './scene-meshes.ts';

interface Builder {
    readonly name: string;
    build(geometry: TinyBvhInput): GpuLayoutMeshBvh;
}

function builders(tinyBvh: TinyBvhBuilder): Builder[] {
    return [
        { name: 'tinybvh (B0)', build: (geometry) => tinyBvh.buildLayout(geometry).layout },
        { name: 'tinybvh BuildHPLOC, leaves of 1', build: (geometry) => tinyBvh.buildLayout(geometry, { builder: 'hploc', maxLeafPrimitives: 1 }).layout },
        {
            name: 'H-PLOC twin, tinybvh-63 and ties, leaves of 1',
            build: (geometry) => buildHplocOnCpu(geometry, { mortonCodes: 'tinybvh-63', maxLeafPrimitives: 1, tieBreak: 'tinybvh' }),
        },
        { name: 'H-PLOC twin, tinybvh-63, leaves of 1', build: (geometry) => buildHplocOnCpu(geometry, { mortonCodes: 'tinybvh-63', maxLeafPrimitives: 1 }) },
        { name: 'H-PLOC twin, cubic-30, leaves of 1', build: (geometry) => buildHplocOnCpu(geometry, { mortonCodes: 'cubic-30', maxLeafPrimitives: 1 }) },
        { name: 'H-PLOC twin, tinybvh-63', build: (geometry) => buildHplocOnCpu(geometry, { mortonCodes: 'tinybvh-63' }) },
        { name: 'H-PLOC twin, cubic-30', build: (geometry) => buildHplocOnCpu(geometry, { mortonCodes: 'cubic-30' }) },
        { name: 'LBVH twin (B1)', build: (geometry) => buildLbvhOnCpu(geometry) },
    ];
}

/** A tree's SAH cost in absolute terms (relative cost times its box's half-area), so meshes can be summed. */
function absoluteSahCost(layout: GpuLayoutMeshBvh): number {
    const b = layout.bounds;
    const size = [b[3] - b[0], b[4] - b[1], b[5] - b[2]];
    return treeSahCost(layout) * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]);
}

async function main(): Promise<void> {
    const scenes = process.argv.length > 2 ? process.argv.slice(2) : ['sponza'];
    const tinyBvh = await createTinyBvhBuilder();
    for (const sceneId of scenes) {
        const meshes = readSceneMeshes(sceneId);
        const triangles = meshes.reduce((sum, mesh) => sum + (mesh.geometry.indices ? mesh.geometry.indices.length / 3 : mesh.geometry.positions.length / 9), 0);
        console.log(`${sceneId}: ${meshes.length} meshes, ${triangles} triangles`);
        const results = builders(tinyBvh).map((builder) => {
            const start = performance.now();
            let cost = 0;
            let nodes = 0;
            let depth = 0;
            for (const mesh of meshes) {
                const layout = builder.build(mesh.geometry);
                cost += absoluteSahCost(layout);
                nodes += layout.nodeCount;
                depth = Math.max(depth, layout.maxDepth);
            }
            return { name: builder.name, cost, nodes, depth, seconds: (performance.now() - start) / 1000 };
        });
        const reference = results[0].cost;
        for (const result of results) {
            console.log(
                `  ${result.name.padEnd(40)} SAH ${(result.cost / reference).toFixed(4)}x B0, ${result.nodes} nodes, depth ${result.depth}, ${result.seconds.toFixed(1)} s`,
            );
        }
    }
}

await main();
