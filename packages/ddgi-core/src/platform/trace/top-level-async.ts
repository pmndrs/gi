/**
 * Builds and checks top-level tree shapes in a module worker when the
 * environment has one, and on the calling thread (on a later turn) otherwise:
 * Node tests, or a host that blocks workers. The same pattern as
 * `mesh-bvh-async.ts`.
 */

import { topLevelInputTransferables, TopLevelShapeKeeper, type TopLevelBuildInput, type TopLevelShape } from './top-level-shape.ts';
import type { TopLevelRequest, TopLevelResponse } from './top-level-worker.ts';

export interface TopLevelResult {
    /** A new shape, or null when a check found the old one still good enough. */
    readonly shape: TopLevelShape | null;
    /** For checks: the old shape's relative SAH cost at the current transforms, and as built. */
    readonly relativeCost: number;
    readonly builtCost: number;
    /** CPU time of the request (in the worker when there is one). */
    readonly workerMs: number;
}

export interface TopLevelBuilder {
    readonly usesWorker: boolean;
    /**
     * `build` makes a new shape; `check` makes one only if the cost of shape
     * `generations.current` (the one in use) at these transforms passed
     * `ratio` times its as-built cost. A new shape is remembered as
     * `generations.next`. The input's arrays move to the worker: they are
     * unusable afterwards.
     */
    request(mode: 'build' | 'check', input: TopLevelBuildInput, ratio: number, generations: { current: number; next: number }): Promise<TopLevelResult>;
    /** Stops the worker. Pending requests reject. */
    dispose(): void;
}

export interface TopLevelBuilderConfig {
    /** `'auto'` (default) uses a worker when `Worker` exists and starts; `'never'` always works inline. */
    readonly worker?: 'auto' | 'never';
}

export function createTopLevelBuilder(config: TopLevelBuilderConfig = {}): TopLevelBuilder {
    const worker = config.worker === 'never' ? null : tryStartWorker();
    return worker === null ? createInlineBuilder() : createWorkerBuilder(worker);
}

function tryStartWorker(): Worker | null {
    if (typeof Worker === 'undefined') return null;
    try {
        // The URL must stay inline in `new Worker(new URL(...))` so bundlers can find the worker file.
        return new Worker(new URL('./top-level-worker.ts', import.meta.url), { type: 'module' });
    } catch {
        return null;
    }
}

function createInlineBuilder(): TopLevelBuilder {
    const keeper = new TopLevelShapeKeeper();
    return {
        usesWorker: false,
        request: (mode, input, ratio, generations) =>
            new Promise((resolve, reject) => {
                // A later turn, so callers see the same asynchronous behaviour as with a worker.
                setTimeout(() => {
                    try {
                        const start = performance.now();
                        const result =
                            mode === 'build'
                                ? { shape: keeper.build(input, generations.next), relativeCost: 0, builtCost: 0 }
                                : keeper.check(input, ratio, generations.current, generations.next);
                        resolve({ ...result, workerMs: performance.now() - start });
                    } catch (error) {
                        reject(error instanceof Error ? error : new Error(String(error)));
                    }
                }, 0);
            }),
        dispose: () => {},
    };
}

function createWorkerBuilder(worker: Worker): TopLevelBuilder {
    const pending = new Map<number, { resolve(result: TopLevelResult): void; reject(error: Error): void }>();
    let nextId = 1;
    const failAll = (message: string): void => {
        for (const entry of pending.values()) entry.reject(new Error(message));
        pending.clear();
    };
    worker.onmessage = (event: MessageEvent<TopLevelResponse>) => {
        const response = event.data;
        const entry = pending.get(response.id);
        if (!entry) return;
        pending.delete(response.id);
        if ('error' in response) entry.reject(new Error(response.error));
        else entry.resolve(response);
    };
    worker.onerror = (event) => failAll(`top-level worker failed: ${event.message}`);
    return {
        usesWorker: true,
        request(mode, input, ratio, generations) {
            return new Promise<TopLevelResult>((resolve, reject) => {
                const id = nextId++;
                pending.set(id, { resolve, reject });
                const message: TopLevelRequest = { id, mode, input, ratio, generation: generations.next, currentGeneration: generations.current };
                worker.postMessage(message, topLevelInputTransferables(input));
            });
        },
        dispose() {
            worker.terminate();
            failAll('the top-level builder was disposed');
        },
    };
}
