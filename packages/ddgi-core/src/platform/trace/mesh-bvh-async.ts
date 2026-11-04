/**
 * Builds mesh BVHs in a module worker when the environment has one, and on the
 * calling thread otherwise (Node tests, or a host that blocks workers).
 */

import type { MeshBvh, MeshBvhBuildOptions, MeshGeometry } from './mesh-bvh.ts';
import { buildMeshBvhAsync } from './mesh-bvh-build.ts';
import type { BuildRequest, BuildResponse } from './mesh-bvh-worker.ts';

export interface MeshBvhBuilderConfig {
    /**
     * `'auto'` (default) uses a worker when `Worker` exists and falls back to
     * the calling thread if starting it fails. `'never'` always builds inline;
     * `'always'` fails instead of falling back.
     */
    readonly worker?: 'auto' | 'never' | 'always';
    /**
     * Move the input buffers into the worker instead of copying them. The
     * caller's arrays become unusable (empty) afterwards. Off by default.
     */
    readonly transferInput?: boolean;
}

export interface MeshBvhBuilder {
    /** Whether builds run in a worker. */
    readonly usesWorker: boolean;
    build(geometry: MeshGeometry, options?: MeshBvhBuildOptions): Promise<MeshBvh>;
    /** Stops the worker. Pending builds reject. */
    dispose(): void;
}

interface PendingBuild {
    resolve(bvh: MeshBvh): void;
    reject(error: Error): void;
}

export function createMeshBvhBuilder(config: MeshBvhBuilderConfig = {}): MeshBvhBuilder {
    const mode = config.worker ?? 'auto';
    let worker: Worker | null = null;
    if (mode !== 'never') {
        worker = tryStartWorker();
        if (worker === null && mode === 'always') {
            throw new Error('BVH builder: a worker was required but could not be started');
        }
    }
    if (worker === null) return createInlineBuilder();
    return createWorkerBuilder(worker, config.transferInput ?? false);
}

function tryStartWorker(): Worker | null {
    if (typeof Worker === 'undefined') return null;
    try {
        // The URL must stay inline in `new Worker(new URL(...))` so bundlers can find the worker file.
        return new Worker(new URL('./mesh-bvh-worker.ts', import.meta.url), { type: 'module' });
    } catch {
        return null;
    }
}

function createInlineBuilder(): MeshBvhBuilder {
    return {
        usesWorker: false,
        build: (geometry, options = {}) =>
            new Promise((resolve, reject) => {
                // Build on a later turn so callers get the same "asynchronous" behaviour as with a worker.
                setTimeout(() => {
                    buildMeshBvhAsync(geometry, options).then(resolve, (error: unknown) =>
                        reject(error instanceof Error ? error : new Error(String(error))),
                    );
                }, 0);
            }),
        dispose: () => {},
    };
}

function createWorkerBuilder(worker: Worker, transferInput: boolean): MeshBvhBuilder {
    const pending = new Map<number, PendingBuild>();
    let nextId = 1;

    const failAll = (message: string): void => {
        for (const entry of pending.values()) entry.reject(new Error(message));
        pending.clear();
    };
    worker.onmessage = (event: MessageEvent<BuildResponse>) => {
        const response = event.data;
        const entry = pending.get(response.id);
        if (!entry) return;
        pending.delete(response.id);
        if ('error' in response) entry.reject(new Error(response.error));
        else entry.resolve(response.bvh);
    };
    worker.onerror = (event) => failAll(`BVH worker failed: ${event.message}`);

    return {
        usesWorker: true,
        build(geometry, options = {}) {
            return new Promise<MeshBvh>((resolve, reject) => {
                const id = nextId++;
                pending.set(id, { resolve, reject });
                const request: BuildRequest = { id, geometry, options };
                const transfer: Transferable[] = [];
                if (transferInput) {
                    transfer.push(geometry.positions.buffer);
                    if (geometry.indices) transfer.push(geometry.indices.buffer);
                    if (geometry.triangleFlags) transfer.push(geometry.triangleFlags.buffer);
                }
                worker.postMessage(request, transfer);
            });
        },
        dispose() {
            worker.terminate();
            failAll('BVH builder was disposed');
        },
    };
}
