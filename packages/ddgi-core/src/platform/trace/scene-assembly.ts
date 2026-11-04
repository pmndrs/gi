/**
 * Puts finished mesh BVHs and the instance list together into the three
 * buffers the tracer reads: all nodes, all triangles, and the top level.
 *
 * Mesh nodes keep mesh-relative indices, so a mesh can be built once, in a
 * worker, and placed anywhere in the shared buffers. The instance record
 * carries the two offsets the traversal adds.
 */

import type { Mat4 } from '../contract/math.ts';
import { BVH_NODE_BYTES, DEFAULT_TRIANGLE_TEST, TRACE_STACK_SIZE, TRIANGLE_BYTES, type TriangleTest } from './layout.ts';
import type { MeshBvh } from './mesh-bvh.ts';
import { buildTopLevel, type TopLevelBvh, type TraceInstance } from './top-level.ts';
import type { SahBuildOptions } from './sah-builder.ts';

/** One placed copy of one of the meshes passed to {@link assembleTraceScene}. */
export interface TraceSceneInstance {
    /** Position in the `meshes` array. */
    readonly meshIndex: number;
    /** The platform instance index, returned in `GiHit.instance`. */
    readonly instanceIndex: number;
    readonly objectToWorld: Mat4;
    readonly contributesToGi?: boolean;
    readonly visible?: boolean;
}

export interface TraceSceneData {
    /** Every mesh's nodes, one after another. */
    readonly nodes: ArrayBuffer;
    /** Every mesh's triangles, one after another. */
    readonly triangles: ArrayBuffer;
    readonly topLevel: TopLevelBvh;
    /** The test every mesh's triangles were built for. */
    readonly triangleTest: TriangleTest;
    /** Where each input mesh landed. */
    readonly meshes: readonly { readonly nodeOffset: number; readonly triangleOffset: number }[];
    /** Deepest mesh tree among the meshes the instances use. */
    readonly maxMeshDepth: number;
    /** Stack entries the worst case needs: top-level depth + one marker + mesh depth. */
    readonly requiredStackSize: number;
    /** Bytes the three buffers hold. */
    readonly byteLength: number;
}

export function assembleTraceScene(
    meshes: readonly MeshBvh[],
    instances: readonly TraceSceneInstance[],
    topLevelOptions: Partial<SahBuildOptions> = {},
): TraceSceneData {
    const triangleTest = meshes[0]?.triangleTest ?? DEFAULT_TRIANGLE_TEST;
    if (meshes.some((mesh) => mesh.triangleTest !== triangleTest)) {
        throw new RangeError('all meshes in a scene must be built for the same triangle test');
    }
    let totalNodes = 0;
    let totalTriangles = 0;
    const placements = meshes.map((mesh) => {
        const placement = { nodeOffset: totalNodes, triangleOffset: totalTriangles };
        totalNodes += mesh.nodeCount;
        totalTriangles += mesh.triangleCount;
        return placement;
    });

    const nodes = new ArrayBuffer(Math.max(1, totalNodes) * BVH_NODE_BYTES);
    const triangles = new ArrayBuffer(Math.max(1, totalTriangles) * TRIANGLE_BYTES);
    meshes.forEach((mesh, index) => {
        new Uint8Array(nodes, placements[index].nodeOffset * BVH_NODE_BYTES, mesh.nodeCount * BVH_NODE_BYTES).set(
            new Uint8Array(mesh.nodes),
        );
        new Uint8Array(triangles, placements[index].triangleOffset * TRIANGLE_BYTES, mesh.triangleCount * TRIANGLE_BYTES).set(
            new Uint8Array(mesh.triangles),
        );
    });

    let maxMeshDepth = 0;
    const traceInstances: TraceInstance[] = instances.map((instance) => {
        const mesh = meshes[instance.meshIndex];
        if (!mesh) throw new RangeError(`instance ${instance.instanceIndex} uses mesh ${instance.meshIndex}, which does not exist`);
        maxMeshDepth = Math.max(maxMeshDepth, mesh.maxDepth);
        return {
            instanceIndex: instance.instanceIndex,
            nodeOffset: placements[instance.meshIndex].nodeOffset,
            triangleOffset: placements[instance.meshIndex].triangleOffset,
            meshBounds: mesh.bounds,
            objectToWorld: instance.objectToWorld,
            contributesToGi: instance.contributesToGi,
            visible: instance.visible,
        };
    });
    const topLevel = buildTopLevel(traceInstances, topLevelOptions);

    const requiredStackSize = topLevel.maxDepth + 1 + maxMeshDepth;
    if (requiredStackSize > TRACE_STACK_SIZE) {
        throw new RangeError(
            `the trace stack holds ${TRACE_STACK_SIZE} entries but this scene can need ${requiredStackSize} ` +
                `(top level ${topLevel.maxDepth} + marker + deepest mesh ${maxMeshDepth})`,
        );
    }

    return {
        nodes,
        triangles,
        topLevel,
        triangleTest,
        meshes: placements,
        maxMeshDepth,
        requiredStackSize,
        byteLength: nodes.byteLength + triangles.byteLength + topLevel.buffer.byteLength,
    };
}
