/**
 * Builds top-level tree shapes in a module worker when the environment has
 * one, and on the calling thread (on a later turn) otherwise: Node tests, or
 * a host that blocks workers. The same pattern as `mesh-bvh-async.ts`.
 */

import { buildTopLevelShape, topLevelInputTransferables, type TopLevelBuildInput, type TopLevelShape } from './top-level-shape.ts';
import type { TopLevelBuildRequest, TopLevelBuildResponse } from './top-level-worker.ts';

export interface TopLevelBuild {
    readonly shape: TopLevelShape;
    /** CPU time of the build itself (in the worker when there is one). */
    readonly buildMs: number;
}

export interface TopLevelBuilder {
    readonly usesWorker: boolean;
    /** Builds over the instances in `input`, whose arrays move to the worker: they are unusable afterwards. */
    build(input: TopLevelBuildInput): Promise<TopLevelBuild>;
    /** Stops the worker. Pending builds reject. */
    dispose(): void;
}

export interface TopLevelBuilderConfig {
    /** `'auto'` (default) uses a worker when `Worker` exists and starts; `'never'` always builds inline. */
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
    return {
        usesWorker: false,
        build: (input) =>
            new Promise((resolve, reject) => {
                // A later turn, so callers see the same asynchronous behaviour as with a worker.
                setTimeout(() => {
                    try {
                        const start = performance.now();
                        const shape = buildTopLevelShape(input);
                        resolve({ shape, buildMs: performance.now() - start });
                    } catch (error) {
                        reject(error instanceof Error ? error : new Error(String(error)));
                    }
                }, 0);
            }),
        dispose: () => {},
    };
}

function createWorkerBuilder(worker: Worker): TopLevelBuilder {
    const pending = new Map<number, { resolve(build: TopLevelBuild): void; reject(error: Error): void }>();
    let nextId = 1;
    const failAll = (message: string): void => {
        for (const entry of pending.values()) entry.reject(new Error(message));
        pending.clear();
    };
    worker.onmessage = (event: MessageEvent<TopLevelBuildResponse>) => {
        const response = event.data;
        const entry = pending.get(response.id);
        if (!entry) return;
        pending.delete(response.id);
        if ('error' in response) entry.reject(new Error(response.error));
        else entry.resolve({ shape: response.shape, buildMs: response.buildMs });
    };
    worker.onerror = (event) => failAll(`top-level worker failed: ${event.message}`);
    return {
        usesWorker: true,
        build(input) {
            return new Promise<TopLevelBuild>((resolve, reject) => {
                const id = nextId++;
                pending.set(id, { resolve, reject });
                const request: TopLevelBuildRequest = { id, input };
                worker.postMessage(request, topLevelInputTransferables(input));
            });
        },
        dispose() {
            worker.terminate();
            failAll('the top-level builder was disposed');
        },
    };
}
