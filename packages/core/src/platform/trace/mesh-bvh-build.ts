/**
 * One entry point for building a mesh tree, for the worker and the inline
 * builder in `mesh-bvh-async.ts`. tinybvh (WASM) is the only mesh builder
 * (B0, DECISIONS 2026-11-04); its WASM is started once per thread, on first use.
 */

import { createTinyBvhBuilder, type TinyBvhBuilder } from '@pmndrs/bvh';
import type { MeshBvh, MeshBvhBuildOptions, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';

let tinyBvh: Promise<TinyBvhBuilder> | null = null;

export async function buildMeshBvhAsync(geometry: MeshGeometry, options: MeshBvhBuildOptions = {}): Promise<MeshBvh> {
    tinyBvh ??= createTinyBvhBuilder();
    return buildMeshBvhWithTinyBvh(geometry, await tinyBvh, { maxLeafPrimitives: options.maxLeafPrimitives });
}
