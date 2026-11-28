/**
 * The synthetic many-instance scene: copies of one small mesh at random
 * transforms. Only used to time the top-level rebuild at 10,000 instances,
 * which is what a scene full of moving objects would cost every frame.
 */

import {
    buildTopLevel,
    type MeshGeometry,
    type TraceInstance,
} from '../../../../packages/core/src/platform/trace/index.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../packages/core/src/platform/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '../../../../packages/bvh/src/index.ts';
import { createRandom } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { SYNTHETIC_INSTANCE_COUNT, SYNTHETIC_SCENE_ID, TOP_LEVEL_REBUILD_RUNS } from './config.ts';
import type { SyntheticResult } from './results.ts';
import { maximum, median } from './statistics.ts';
import { timeTopLevelRebuilds } from './ours-bench.ts';

// Mesh trees come from tinybvh (WASM), the only mesh builder since B0.
const tinyBvh = await createTinyBvhBuilder();

/** A latitude-longitude sphere of radius 1: about `2 * segments * (rings - 1)` triangles. */
export function makeSphereMesh(segments = 16, rings = 12): MeshGeometry {
    const positions: number[] = [];
    for (let ring = 0; ring <= rings; ring++) {
        const polar = (ring / rings) * Math.PI;
        for (let segment = 0; segment < segments; segment++) {
            const azimuth = (segment / segments) * 2 * Math.PI;
            positions.push(Math.sin(polar) * Math.cos(azimuth), Math.cos(polar), Math.sin(polar) * Math.sin(azimuth));
        }
    }
    const indices: number[] = [];
    for (let ring = 0; ring < rings; ring++) {
        for (let segment = 0; segment < segments; segment++) {
            const next = (segment + 1) % segments;
            const a = ring * segments + segment;
            const b = ring * segments + next;
            const c = (ring + 1) * segments + segment;
            const d = (ring + 1) * segments + next;
            if (ring > 0) indices.push(a, c, b);
            if (ring < rings - 1) indices.push(b, c, d);
        }
    }
    return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

/** Column-major affine transform: a random rotation, a uniform scale of 0.5 to 2, a position inside a cube. */
function randomTransform(random: () => number, extent: number): number[] {
    // A random unit quaternion by Shoemake's method.
    const u1 = random();
    const u2 = random() * 2 * Math.PI;
    const u3 = random() * 2 * Math.PI;
    const a = Math.sqrt(1 - u1);
    const b = Math.sqrt(u1);
    const [x, y, z, w] = [a * Math.sin(u2), a * Math.cos(u2), b * Math.sin(u3), b * Math.cos(u3)];
    const scale = 0.5 + 1.5 * random();
    return [
        (1 - 2 * (y * y + z * z)) * scale, 2 * (x * y + z * w) * scale, 2 * (x * z - y * w) * scale, 0,
        2 * (x * y - z * w) * scale, (1 - 2 * (x * x + z * z)) * scale, 2 * (y * z + x * w) * scale, 0,
        2 * (x * z + y * w) * scale, 2 * (y * z - x * w) * scale, (1 - 2 * (x * x + y * y)) * scale, 0,
        (random() * 2 - 1) * extent, (random() * 2 - 1) * extent, (random() * 2 - 1) * extent, 1,
    ];
}

export async function runSynthetic(instanceCount = SYNTHETIC_INSTANCE_COUNT, runs = TOP_LEVEL_REBUILD_RUNS): Promise<SyntheticResult> {
    const mesh = makeSphereMesh();
    const bvh = buildMeshBvhWithTinyBvh(mesh, tinyBvh);
    const random = createRandom(0x10_000);
    const instances: TraceInstance[] = [];
    for (let i = 0; i < instanceCount; i++) {
        instances.push({
            instanceIndex: i,
            nodeOffset: 0,
            triangleOffset: 0,
            meshBounds: bvh.bounds,
            objectToWorld: randomTransform(random, 100),
        });
    }
    const times = await timeTopLevelRebuilds(instances, runs);
    const topLevel = buildTopLevel(instances);
    return {
        scene: SYNTHETIC_SCENE_ID,
        instanceCount,
        meshTriangles: bvh.triangleCount,
        topLevelRebuildMsMedian: median(times),
        topLevelRebuildMsMax: maximum(times),
        topLevelRebuildRuns: times.length,
        topLevelNodes: topLevel.nodeCount,
        topLevelMaxDepth: topLevel.maxDepth,
        topLevelBytes: topLevel.buffer.byteLength,
    };
}
