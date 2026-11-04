/**
 * One entry point for every mesh builder, for the worker and the inline
 * builder in `mesh-bvh-async.ts`. tinybvh's WASM is started once per thread,
 * on first use.
 */

import { createTinyBvhBuilder, type TinyBvhBuilder } from '@pmndrs/bvh';
import { buildMeshBvh, type MeshBvh, type MeshBvhBuildOptions, type MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';

let tinyBvh: Promise<TinyBvhBuilder> | null = null;

/** Builds with the builder `options.builder` names (default `sah`). */
export async function buildMeshBvhAsync(geometry: MeshGeometry, options: MeshBvhBuildOptions = {}): Promise<MeshBvh> {
    const { builder = 'sah', maxLeafPrimitives } = options;
    if (builder === 'sah') return buildMeshBvh(geometry, options);
    tinyBvh ??= createTinyBvhBuilder();
    return buildMeshBvhWithTinyBvh(geometry, await tinyBvh, { maxLeafPrimitives });
}
