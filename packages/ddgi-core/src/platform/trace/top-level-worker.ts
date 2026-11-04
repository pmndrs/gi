/**
 * Worker entry: builds top-level tree shapes off the main thread (B1a). A
 * shape for 10,000 instances takes about 50-120 ms of CPU, which must not
 * land in a frame. Started by `top-level-async.ts`; nothing imports this file.
 *
 * It is a worker of its own, not the mesh-tree worker, so a rebuild for
 * moving instances never waits behind a mesh build that takes seconds.
 */

import { buildTopLevelShape, topLevelShapeTransferables, type TopLevelBuildInput, type TopLevelShape } from './top-level-shape.ts';

export interface TopLevelBuildRequest {
    readonly id: number;
    readonly input: TopLevelBuildInput;
}

export type TopLevelBuildResponse =
    | { readonly id: number; readonly shape: TopLevelShape; readonly buildMs: number }
    | { readonly id: number; readonly error: string };

/** The few worker-scope members we use. The project's tsconfig only has the DOM library. */
interface WorkerScope {
    onmessage: ((event: MessageEvent<TopLevelBuildRequest>) => void) | null;
    postMessage(message: TopLevelBuildResponse, transfer: Transferable[]): void;
}

const scope = self as unknown as WorkerScope;

scope.onmessage = (event) => {
    const { id, input } = event.data;
    try {
        const start = performance.now();
        const shape = buildTopLevelShape(input);
        scope.postMessage({ id, shape, buildMs: performance.now() - start }, topLevelShapeTransferables(shape));
    } catch (error) {
        scope.postMessage({ id, error: error instanceof Error ? error.message : String(error) }, []);
    }
};
