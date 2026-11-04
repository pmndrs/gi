/**
 * Worker entry: builds mesh BVHs off the main thread so a large mesh does not
 * freeze the page. Started by `mesh-bvh-async.ts`; nothing imports this file.
 */

import { buildMeshBvh, meshBvhTransferables, type MeshBvh, type MeshBvhBuildOptions, type MeshGeometry } from './mesh-bvh.ts';

export interface BuildRequest {
    readonly id: number;
    readonly geometry: MeshGeometry;
    readonly options: MeshBvhBuildOptions;
}

export type BuildResponse =
    | { readonly id: number; readonly bvh: MeshBvh }
    | { readonly id: number; readonly error: string };

/** The few worker-scope members we use. The project's tsconfig only has the DOM library. */
interface WorkerScope {
    onmessage: ((event: MessageEvent<BuildRequest>) => void) | null;
    postMessage(message: BuildResponse, transfer: Transferable[]): void;
}

const scope = self as unknown as WorkerScope;

scope.onmessage = (event) => {
    const { id, geometry, options } = event.data;
    try {
        const bvh = buildMeshBvh(geometry, options);
        scope.postMessage({ id, bvh }, meshBvhTransferables(bvh));
    } catch (error) {
        scope.postMessage({ id, error: error instanceof Error ? error.message : String(error) }, []);
    }
};
