/**
 * The `rays` query backend over the platform's scene copy: what techniques
 * get as `context.queries.rays`.
 *
 * It keeps the tracer's GPU buffers in step with the scene:
 *
 * - **Mesh trees.** When the scene uploads a mesh, tinybvh (WASM) builds its
 *   tree off the main thread, and the tree is placed into the shared node and triangle buffers when it arrives, at
 *   the start of a later `encode`. Until then rays see the mesh's previous
 *   shape, or miss it if it is new (`building` is true meanwhile).
 * - **The top level.** Rebuilt on the CPU when an instance, a transform or a
 *   tree changed. B1a moves this to the GPU.
 *
 * `GiHit.instance` is the scene copy's instance slot, so a hit reads its
 * surface straight through the scene module (`giInstanceTriangle(hit.instance,
 * hit.triangle)`), and `GiHit.triangle` is the triangle's index in the mesh as
 * the host gave it.
 *
 * Nothing here recompiles: the WGSL is fixed, and buffers that grow bump
 * `module.version` so kernels rebuild their bind group only.
 */

import { GILimitError, GINotReadyError } from '../contract/errors.ts';
import type { RayQueries } from '../contract/queries.ts';
import type { BindingSlot, ComputeKernel, KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import type { PlatformScene, SceneUpload, UploadedMesh } from '../scene/platform-scene.ts';
import { packedIndices, packedPositions } from '../scene/mesh-packing.ts';
import { GrowableStorage, type ResidentRange } from './growable-storage.ts';
import { BVH_NODE_BYTES, TOP_LEVEL_HEADER_BYTES, TRACE_STACK_SIZE, TRIANGLE_BYTES } from './layout.ts';
import type { MeshBvh } from './mesh-bvh.ts';
import { createMeshBvhBuilder, type MeshBvhBuilder } from './mesh-bvh-async.ts';
import { buildTopLevel, type TraceInstance } from './top-level.ts';
import { RAY_BYTES } from './trace-gpu.ts';
import { BATCH_WORKGROUP_SIZE, batchEntryWgsl, HIT_RECORD_BYTES, TRACE_MODULE_ENTRIES, traceModuleWgsl } from './wgsl.ts';

export interface PlatformRayQueriesOptions {
    /** Prefix for GPU object labels. */
    readonly label?: string;
    /** `worker` (default) builds mesh trees in a module worker; `main` on this thread. */
    readonly buildIn?: 'worker' | 'main';
}

/** A mesh tree placed in the shared buffers. */
interface PlacedTree {
    readonly nodeOffset: number;
    readonly nodeCount: number;
    readonly triangleOffset: number;
    readonly triangleCount: number;
    readonly maxDepth: number;
    readonly bounds: Float32Array;
}

/** A finished build waiting to be placed at the next `update`. */
interface ArrivedTree {
    readonly slot: number;
    readonly generation: number;
    readonly bvh: MeshBvh;
}

/** GPU bytes the tracer holds, by buffer. */
export interface AccelerationMemory {
    readonly nodes: number;
    readonly triangles: number;
    readonly topLevel: number;
    readonly total: number;
}

/** Starting sizes; both buffers double when they fill. */
const INITIAL_NODES = 1024;
const INITIAL_TRIANGLES = 2048;

/** Bindings of a batch kernel's own group (group 1): rays in, results out. */
const BATCH_GROUP = 1;

export class PlatformRayQueries implements RayQueries {
    readonly kind = 'rays';
    readonly capabilities = { alphaCutouts: false, backFaceHits: true } as const;
    readonly module: WgslModule;

    private readonly device: GPUDevice;
    private readonly scene: PlatformScene;
    private readonly label: string;
    private readonly builder: MeshBvhBuilder;

    private readonly nodes: GrowableStorage;
    private readonly triangles: GrowableStorage;
    private topLevelBuffer: GPUBuffer;
    private retiredTopLevel: GPUBuffer[] = [];
    private bufferVersion = 0;

    /** Bumped every time a mesh slot is uploaded or removed, so a late build for an old version is dropped. */
    private readonly generations = new Map<number, number>();
    private readonly placed = new Map<number, PlacedTree>();
    private arrived: ArrivedTree[] = [];
    private pendingBuilds = 0;
    private buildFailure: Error | null = null;
    /** Set when the top level no longer matches the instances or the placed trees. */
    private topLevelStale = false;
    private builtWaiters: (() => void)[] = [];

    private closestKernel: ComputeKernel | null = null;
    private anyKernel: ComputeKernel | null = null;
    private batchLayout: GPUBindGroupLayout | null = null;
    private disposed = false;

    /** Instances in the current top level; for stats and tests. */
    topLevelInstanceCount = 0;

    constructor(device: GPUDevice, scene: PlatformScene, options: PlatformRayQueriesOptions = {}) {
        this.device = device;
        this.scene = scene;
        this.label = options.label ?? 'gi';
        this.builder = createMeshBvhBuilder({
            worker: options.buildIn === 'main' ? 'never' : 'auto',
            // The inputs are copies made for the build (startBuild), so they move instead of being cloned.
            transferInput: true,
        });

        this.nodes = new GrowableStorage(device, `${this.label} trace nodes`, BVH_NODE_BYTES, INITIAL_NODES);
        this.triangles = new GrowableStorage(device, `${this.label} trace triangles`, TRIANGLE_BYTES, INITIAL_TRIANGLES);
        // An empty top level: the header says 0 nodes, so every ray misses.
        this.topLevelBuffer = this.createTopLevelBuffer(TOP_LEVEL_HEADER_BYTES);

        const currentVersion = (): number => this.bufferVersion;
        this.module = {
            name: 'gi-trace',
            dependencies: [],
            code: (slot: BindingSlot) => traceModuleWgsl(slot),
            entries: TRACE_MODULE_ENTRIES,
            resources: () => [{ buffer: this.nodes.buffer }, { buffer: this.triangles.buffer }, { buffer: this.topLevelBuffer }],
            get version() {
                return currentVersion();
            },
            storageBufferCount: TRACE_MODULE_ENTRIES.length,
        };
    }

    /** True while a mesh tree is still being built. Rays meanwhile see the mesh's previous shape, or miss it if it is new. */
    get building(): boolean {
        return this.pendingBuilds > 0;
    }

    get memoryBytes(): number {
        return this.memory.total;
    }

    get memory(): AccelerationMemory {
        const nodes = this.nodes.byteLength;
        const triangles = this.triangles.byteLength;
        const topLevel = this.topLevelBuffer.size;
        return { nodes, triangles, topLevel, total: nodes + triangles + topLevel };
    }

    /** Whether builds run in a worker. */
    get usesWorker(): boolean {
        return this.builder.usesWorker;
    }

    /**
     * Resolves once no mesh tree is building. The trees still reach the GPU
     * only at the next `update` (the platform's next `encode`).
     */
    whenBuilt(): Promise<void> {
        if (this.pendingBuilds === 0) return Promise.resolve();
        return new Promise((resolve) => this.builtWaiters.push(resolve));
    }

    /** Compiles the batch kernels. Called by the platform's `prepare`. */
    async prepare(kernels: KernelBuilder): Promise<void> {
        if (this.closestKernel && this.anyKernel) return;
        const compute = 0x4; // GPUShaderStage.COMPUTE, spelled out so Node tests need no globals
        const batchLayout = this.device.createBindGroupLayout({
            label: `${this.label} trace batch`,
            entries: [
                { binding: 0, visibility: compute, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: compute, buffer: { type: 'storage' } },
            ],
        });
        const kernelFor = (mode: 'closest' | 'any') =>
            kernels.computeKernel({
                label: `trace batch ${mode}`,
                modules: [this.module],
                code: batchEntryWgsl(mode, BATCH_GROUP),
                entryPoint: 'giBatch',
                ownLayouts: [batchLayout],
            });
        [this.closestKernel, this.anyKernel] = await Promise.all([kernelFor('closest'), kernelFor('any')]);
        this.batchLayout = batchLayout;
    }

    /**
     * Brings the tracer's buffers up to date with this frame's scene upload.
     * Called by the platform right after the scene copy's own `encode`, in the
     * same encoder, before any technique runs.
     */
    update(encoder: GPUCommandEncoder, upload: SceneUpload): void {
        this.checkLive();
        this.nodes.releaseRetired();
        this.triangles.releaseRetired();
        for (const buffer of this.retiredTopLevel) buffer.destroy();
        this.retiredTopLevel = [];

        if (upload.instancesChanged) this.topLevelStale = true;
        // Removals first: a slot may be removed and reused by a new mesh in the same frame.
        for (const slot of upload.meshesRemoved) {
            this.bumpGeneration(slot);
            this.releaseTree(slot);
            this.topLevelStale = true;
        }
        for (const mesh of upload.meshesUploaded) this.startBuild(mesh);

        // Placing trees can fail (a buffer over the device's binding limit). The top level
        // is still rebuilt from what is placed, so it never points at a released range, and
        // then the error is thrown.
        let failure: unknown = null;
        const arrived = this.takeArrived();
        if (arrived.length > 0) {
            const buffersBefore = [this.nodes.buffer, this.triangles.buffer];
            try {
                this.placeTrees(encoder, arrived);
            } catch (error) {
                failure = error;
            }
            if (this.nodes.buffer !== buffersBefore[0] || this.triangles.buffer !== buffersBefore[1]) this.bufferVersion++;
            this.topLevelStale = true;
        }
        if (this.topLevelStale) {
            this.rebuildTopLevel();
            this.topLevelStale = false;
        }

        if (failure === null && this.buildFailure) {
            // Reported once; the mesh stays out of the top level (rays miss it) until it is uploaded again.
            failure = this.buildFailure;
            this.buildFailure = null;
        }
        if (failure !== null) throw failure;
    }

    /**
     * Traces `count` rays from `rays` (packed `GiRay`, 32 bytes each) into
     * `hits`: `GiHitRecord`s (24 bytes) for `closest`, or one u32 per ray
     * (1 = blocked) for `any`. `timestampWrites` is for benchmarks.
     */
    encodeBatch(
        encoder: GPUCommandEncoder,
        batch: { readonly mode: 'closest' | 'any'; readonly rays: GPUBuffer; readonly hits: GPUBuffer; readonly count: number },
        timestampWrites?: GPUComputePassTimestampWrites,
    ): void {
        this.checkLive();
        const kernel = batch.mode === 'closest' ? this.closestKernel : this.anyKernel;
        if (!kernel || !this.batchLayout) throw new GINotReadyError('preparing');
        if (batch.count <= 0) return;
        const resultBytes = batch.mode === 'closest' ? HIT_RECORD_BYTES : 4;
        if (batch.rays.size < batch.count * RAY_BYTES) throw new RangeError(`the ray buffer holds fewer than ${batch.count} rays`);
        if (batch.hits.size < batch.count * resultBytes) throw new RangeError(`the hit buffer holds fewer than ${batch.count} results`);

        // The kernel takes the ray count from the size of the bound range, so no uniform is needed.
        const group = this.device.createBindGroup({
            label: `${this.label} trace batch ${batch.mode}`,
            layout: this.batchLayout,
            entries: [
                { binding: 0, resource: { buffer: batch.rays, offset: 0, size: batch.count * RAY_BYTES } },
                { binding: 1, resource: { buffer: batch.hits, offset: 0, size: batch.count * resultBytes } },
            ],
        });
        // Dispatches are 2D so a very large batch stays under the per-dimension workgroup limit.
        const workgroups = Math.ceil(batch.count / BATCH_WORKGROUP_SIZE);
        const x = Math.min(workgroups, this.device.limits.maxComputeWorkgroupsPerDimension);
        const y = Math.ceil(workgroups / x);
        const pass = encoder.beginComputePass({ label: `${this.label} trace batch ${batch.mode}`, timestampWrites });
        kernel.bind(pass);
        pass.setBindGroup(BATCH_GROUP, group);
        pass.dispatchWorkgroups(x, y);
        pass.end();
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.builder.dispose();
        this.nodes.destroy();
        this.triangles.destroy();
        for (const buffer of this.retiredTopLevel) buffer.destroy();
        this.topLevelBuffer.destroy();
        for (const resolve of this.builtWaiters.splice(0)) resolve();
    }

    // -----------------------------------------------------------------------
    // Mesh trees
    // -----------------------------------------------------------------------

    private bumpGeneration(slot: number): number {
        const generation = (this.generations.get(slot) ?? 0) + 1;
        this.generations.set(slot, generation);
        return generation;
    }

    private startBuild(mesh: UploadedMesh): void {
        const generation = this.bumpGeneration(mesh.slot);
        // Compact copies: the packed words also hold normals and uvs, and the scene drops them after upload.
        const geometry = { positions: packedPositions(mesh.packed).slice(), indices: packedIndices(mesh.packed) };
        this.pendingBuilds++;
        const finished = (bvh: MeshBvh | null): void => {
            this.pendingBuilds--;
            if (bvh && !this.disposed && this.generations.get(mesh.slot) === generation) {
                this.arrived.push({ slot: mesh.slot, generation, bvh });
            }
            if (this.pendingBuilds === 0) for (const resolve of this.builtWaiters.splice(0)) resolve();
        };
        this.builder.build(geometry).then(finished, (error: unknown) => {
            if (!this.disposed) this.buildFailure = new Error(`@pmndrs/ddgi: building a mesh tree failed: ${describe(error)}`);
            finished(null);
        });
    }

    /** Finished builds for meshes that still exist in the version they were built from. */
    private takeArrived(): ArrivedTree[] {
        const current = this.arrived.filter((tree) => this.generations.get(tree.slot) === tree.generation);
        this.arrived = [];
        return current;
    }

    private releaseTree(slot: number): void {
        const tree = this.placed.get(slot);
        if (!tree) return;
        this.nodes.release(tree.nodeOffset, tree.nodeCount);
        this.triangles.release(tree.triangleOffset, tree.triangleCount);
        this.placed.delete(slot);
    }

    /** Copies arrived trees into the shared buffers, growing them first if needed. */
    private placeTrees(encoder: GPUCommandEncoder, arrived: readonly ArrivedTree[]): void {
        // A rebuilt mesh replaces its old tree: free that first so the space can be reused.
        for (const tree of arrived) this.releaseTree(tree.slot);

        const nodeCounts = arrived.map((tree) => tree.bvh.nodeCount);
        const triangleCounts = arrived.map((tree) => tree.bvh.triangleCount);
        let nodeOffsets: number[] | null = null;
        let triangleOffsets: number[];
        try {
            nodeOffsets = this.allocateAll(this.nodes, encoder, nodeCounts, (placed) => ({ offset: placed.nodeOffset, count: placed.nodeCount }));
            triangleOffsets = this.allocateAll(this.triangles, encoder, triangleCounts, (placed) => ({
                offset: placed.triangleOffset,
                count: placed.triangleCount,
            }));
        } catch (error) {
            // Nothing was written. Keep the trees queued, so the next frame tries again (and
            // reports the same error) instead of losing them.
            nodeOffsets?.forEach((offset, index) => this.nodes.release(offset, nodeCounts[index]));
            this.arrived.push(...arrived);
            throw error;
        }

        arrived.forEach(({ slot, bvh }, index) => {
            this.nodes.write(nodeOffsets[index], bvh.nodes);
            this.triangles.write(triangleOffsets[index], bvh.triangles);
            this.placed.set(slot, {
                nodeOffset: nodeOffsets[index],
                nodeCount: bvh.nodeCount,
                triangleOffset: triangleOffsets[index],
                triangleCount: bvh.triangleCount,
                maxDepth: bvh.maxDepth,
                bounds: bvh.bounds,
            });
        });
    }

    /**
     * Allocates one range per entry of `counts`. If they don't all fit, the
     * storage grows once (copying the trees already placed) and they are
     * allocated again.
     */
    private allocateAll(
        storage: GrowableStorage,
        encoder: GPUCommandEncoder,
        counts: readonly number[],
        rangeOf: (tree: PlacedTree) => ResidentRange,
    ): number[] {
        const offsets: number[] = [];
        for (const count of counts) {
            const offset = storage.allocate(count);
            if (offset === null) {
                // Undo, grow once for everything, then allocate again.
                offsets.forEach((placedOffset, index) => storage.release(placedOffset, counts[index]));
                const total = counts.reduce((sum, value) => sum + value, 0);
                storage.grow(encoder, total, [...this.placed.values()].map(rangeOf));
                return counts.map((value) => {
                    const grown = storage.allocate(value);
                    if (grown === null) throw new Error('trace buffer allocation failed after growing');
                    return grown;
                });
            }
            offsets.push(offset);
        }
        return offsets;
    }

    // -----------------------------------------------------------------------
    // Top level
    // -----------------------------------------------------------------------

    private rebuildTopLevel(): void {
        const instances: TraceInstance[] = [];
        let deepestMesh = 0;
        for (const instance of this.scene.instanceSnapshot()) {
            // An instance whose mesh tree hasn't arrived yet is left out until it does.
            const tree = this.placed.get(instance.meshSlot);
            if (!tree) continue;
            if (!instance.visible || !instance.contributesToGi) continue;
            deepestMesh = Math.max(deepestMesh, tree.maxDepth);
            instances.push({
                instanceIndex: instance.slot,
                nodeOffset: tree.nodeOffset,
                triangleOffset: tree.triangleOffset,
                meshBounds: tree.bounds,
                objectToWorld: instance.objectToWorld,
            });
        }
        const topLevel = buildTopLevel(instances);
        const requiredStack = topLevel.maxDepth + 1 + deepestMesh;
        if (requiredStack > TRACE_STACK_SIZE) {
            throw new Error(
                `@pmndrs/ddgi: the trace stack holds ${TRACE_STACK_SIZE} entries, but this scene can need ${requiredStack} ` +
                    `(top level ${topLevel.maxDepth} + 1 + deepest mesh tree ${deepestMesh}).`,
            );
        }

        const bytes = topLevel.buffer.byteLength;
        if (bytes > this.topLevelBuffer.size) {
            const limit = this.device.limits.maxStorageBufferBindingSize;
            if (bytes > limit) throw new GILimitError('maxStorageBufferBindingSize', bytes, limit);
            this.retiredTopLevel.push(this.topLevelBuffer);
            // Room to grow, so a scene that keeps adding instances doesn't reallocate every frame.
            this.topLevelBuffer = this.createTopLevelBuffer(Math.min(limit, Math.max(bytes, this.topLevelBuffer.size * 2)));
            this.bufferVersion++;
        }
        this.device.queue.writeBuffer(this.topLevelBuffer, 0, topLevel.buffer);
        this.topLevelInstanceCount = topLevel.instanceCount;
    }

    private createTopLevelBuffer(size: number): GPUBuffer {
        return this.device.createBuffer({
            label: `${this.label} trace top level`,
            size: Math.ceil(size / 16) * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/ddgi: the ray queries were disposed');
    }
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}
