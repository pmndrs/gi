/**
 * Worker entry: builds and checks top-level tree shapes off the main thread
 * (B1a). A shape for 10,000 instances takes about 55 ms of CPU, which must
 * not land in a frame. Started by `top-level-async.ts`; nothing imports this
 * file.
 *
 * It is a worker of its own, not the mesh-tree worker, so a rebuild for
 * moving instances never waits behind a mesh build that takes seconds. It
 * keeps the last shape it built, which quality checks compare against
 * (`TopLevelShapeKeeper`).
 */

import { topLevelShapeTransferables, TopLevelShapeKeeper, type TopLevelBuildInput, type TopLevelShape } from './top-level-shape.ts';

export interface TopLevelRequest {
    readonly id: number;
    /** `build`: a new shape now. `check`: a new shape only if the last one's cost passed `ratio` times its as-built cost. */
    readonly mode: 'build' | 'check';
    readonly input: TopLevelBuildInput;
    readonly ratio: number;
    /** The generation a new shape gets. */
    readonly generation: number;
    /** For checks: the generation of the shape the GPU uses now. */
    readonly currentGeneration: number;
}

export type TopLevelResponse =
    | { readonly id: number; readonly shape: TopLevelShape | null; readonly relativeCost: number; readonly builtCost: number; readonly workerMs: number }
    | { readonly id: number; readonly error: string };

/** The few worker-scope members we use. The project's tsconfig only has the DOM library. */
interface WorkerScope {
    onmessage: ((event: MessageEvent<TopLevelRequest>) => void) | null;
    postMessage(message: TopLevelResponse, transfer: Transferable[]): void;
}

const scope = self as unknown as WorkerScope;
const keeper = new TopLevelShapeKeeper();

scope.onmessage = (event) => {
    const { id, mode, input, ratio, generation, currentGeneration } = event.data;
    try {
        const start = performance.now();
        const result = mode === 'build' ? { shape: keeper.build(input, generation), relativeCost: 0, builtCost: 0 } : keeper.check(input, ratio, currentGeneration, generation);
        const response = { id, ...result, workerMs: performance.now() - start };
        scope.postMessage(response, result.shape ? topLevelShapeTransferables(result.shape) : []);
    } catch (error) {
        scope.postMessage({ id, error: error instanceof Error ? error.message : String(error) }, []);
    }
};
