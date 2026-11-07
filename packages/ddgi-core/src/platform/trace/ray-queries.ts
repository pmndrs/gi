/**
 * The `rays` query backend over the platform's scene copy: what techniques
 * get as `context.queries.rays`.
 *
 * It keeps the tracer's GPU buffers in step with the scene:
 *
 * - **Mesh trees.** When the scene uploads a mesh, tinybvh (WASM) builds its
 *   tree off the main thread, and the tree is placed into the shared node and
 *   triangle buffers when it arrives, at the start of a later `encode`. Until
 *   then rays see the mesh's previous shape, or miss it if it is new
 *   (`building` is true meanwhile).
 * - **First trees from the GPU** (T2.12, "fast first GI"). On a device with
 *   subgroups, the meshes of the scene as it is first loaded are also built
 *   on the GPU with H-PLOC, all in one batch (gpu-mesh-trees.ts). Their
 *   trees arrive a few frames later, after a small stats readback, and are
 *   copied into the same shared buffers, so rays see the scene long before
 *   tinybvh is done. tinybvh's tree then replaces each mesh's GPU tree as it
 *   arrives (the "swap"): only that mesh's range and its mesh-table entry
 *   change, so nothing recompiles and the top level keeps its shape. A GPU
 *   tree that was refused, or is too deep for the trace stack, is never used:
 *   that mesh waits for tinybvh. Meshes added later get tinybvh trees only
 *   (GPU builds at runtime are M3's T3.2). `building` is true until every
 *   mesh has its first tree; `refining` while tinybvh trees are still to come.
 * - **The top level** (stage B1a). On a device with subgroups it lives on the
 *   GPU (top-level-gpu.ts): refit every frame something moved, from the
 *   scene copy's transforms, with no CPU work per instance; a new shape is
 *   built in a worker now and then. Without subgroups, the declared fallback
 *   (DECISIONS 2026-11-04): it is rebuilt on the CPU, on this thread, when an
 *   instance, a transform or a tree changed. `topLevelPath` says which.
 *
 * `GiHit.instance` is the scene copy's instance slot, so a hit reads its
 * surface straight through the scene module (`giInstanceTriangle(hit.instance,
 * hit.triangle)`), and `GiHit.triangle` is the triangle's index in the mesh as
 * the host gave it.
 *
 * Nothing here recompiles: the WGSL is fixed, and buffers that grow bump
 * `module.version` so kernels rebuild their bind group only.
 */

import { GpuBvhRefit } from '@pmndrs/bvh/gpu-refit';
import { GILimitError, GINotReadyError } from '../contract/errors.ts';
import type { RayQueries } from '../contract/queries.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { BindingSlot, ComputeKernel, KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import { SHADER_STAGE_COMPUTE } from '../kernel-builder.ts';
import type { PlatformScene, SceneUpload, UploadedMesh } from '../scene/platform-scene.ts';
import { packedIndices, packedPositions } from '../scene/mesh-packing.ts';
import { GpuMeshTrees, type FirstTreeBuilder, type GpuBuiltTree, type GpuTreeBatchResult, type GpuTreeOutputs, type GpuTreeRequest } from './gpu-mesh-trees.ts';
import { GrowableStorage, type ResidentRange } from './growable-storage.ts';
import { BVH_NODE_BYTES, TOP_LEVEL_HEADER_BYTES, TRACE_STACK_SIZE, TRIANGLE_BYTES } from './layout.ts';
import type { MeshBvh, MeshGeometry } from './mesh-bvh.ts';
import { createMeshBvhBuilder, type MeshBvhBuilder } from './mesh-bvh-async.ts';
import { buildTopLevel, type TraceInstance } from './top-level.ts';
import { GpuTopLevel } from './top-level-gpu.ts';
import { RAY_BYTES } from './trace-gpu.ts';
import { BATCH_WORKGROUP_SIZE, batchEntryWgsl, HIT_RECORD_BYTES, TRACE_MODULE_ENTRIES, traceModuleWgsl } from './wgsl.ts';

export interface PlatformRayQueriesOptions {
    /** Prefix for GPU object labels. */
    readonly label?: string;
    /** `worker` (default) builds mesh trees and top-level shapes in module workers; `main` on this thread. */
    readonly buildIn?: 'worker' | 'main';
    /**
     * Read the GPU top level's refit stats back, for counters and warnings.
     * Off by default (hosts are promised no readbacks); the platform turns it
     * on with `gpuTiming`.
     */
    readonly diagnostics?: boolean;
    /** Where the GPU top level reports a refit it had to repair, and a GPU tree build that failed (once each). Defaults to `console.warn`. */
    readonly warn?: (message: string) => void;
    /** How mesh trees are made (T2.12); the defaults are what hosts get. */
    readonly meshTrees?: MeshTreeOptions;
}

/**
 * How mesh trees are made. Not host-facing: the platform passes these through
 * from its implementation options, for the repository's tools and tests.
 */
export interface MeshTreeOptions {
    /** First trees from the GPU (H-PLOC) for the scene as first loaded, on a device with subgroups. Default true; false is M1's path, tinybvh only. */
    readonly gpuFirst?: boolean;
    /**
     * tinybvh's trees wait instead of replacing a mesh's GPU tree, until
     * {@link PlatformRayQueries.releaseRefinements}. A mesh with no tree yet
     * still takes tinybvh's at once. For tools that check a scene mid-swap or
     * time the swap frame. Default false.
     */
    readonly holdRefinements?: boolean;
    /** GPU trees deeper than this are refused, and the mesh waits for tinybvh. Default {@link GPU_TREE_MAX_DEPTH}. */
    readonly gpuMaxDepth?: number;
    /**
     * Builds the first trees instead of the GPU H-PLOC. For unit tests, which
     * run in Node without a GPU; it is used whatever the device's features.
     */
    readonly firstTreeBuilder?: FirstTreeBuilder;
}

/**
 * Levels kept free on the trace stack for the top level and the instance
 * marker. The stack (64 entries) is shared by both levels, so a GPU tree
 * deeper than {@link GPU_TREE_MAX_DEPTH} waits for tinybvh's rather than
 * risk a scene that can't be traced. 16 levels hold a top level over tens of
 * thousands of instances.
 */
const TOP_LEVEL_STACK_RESERVE = 16;
export const GPU_TREE_MAX_DEPTH = TRACE_STACK_SIZE - 1 - TOP_LEVEL_STACK_RESERVE;

/** Where the top level is kept up to date: refit on the GPU, or the CPU fallback without subgroups. */
export type TopLevelPath = 'gpu-refit' | 'cpu';

/** Which builder made a mesh's tree. */
export type MeshTreeSource = 'gpu' | 'tinybvh';

/** A mesh tree placed in the shared buffers. */
interface PlacedTree {
    readonly nodeOffset: number;
    readonly nodeCount: number;
    readonly triangleOffset: number;
    readonly triangleCount: number;
    readonly maxDepth: number;
    readonly bounds: Float32Array;
    readonly source: MeshTreeSource;
    /** The mesh version the tree was built from. */
    readonly generation: number;
}

/** A finished build waiting to be placed at a later `update`: tinybvh's, from the worker, or the GPU's, in its batch's buffers. */
type ArrivedTree =
    | { readonly source: 'tinybvh'; readonly slot: number; readonly generation: number; readonly bvh: MeshBvh }
    | { readonly source: 'gpu'; readonly slot: number; readonly generation: number; readonly built: GpuBuiltTree };

/**
 * What one version of a mesh (one upload of a slot) has from each builder.
 * `gpu` is `'none'` when the mesh wasn't built on the GPU: no subgroups, or
 * uploaded after the scene's first trees.
 */
interface MeshTreeState {
    readonly generation: number;
    gpu: 'none' | 'building' | 'built' | 'refused';
    tinybvh: 'building' | 'built' | 'failed';
}

/** Counts of the mesh trees, for `stats.counters` and the tools. */
export interface MeshTreeCounts {
    /** Meshes whose rays see a GPU-built tree now. */
    readonly onGpuTrees: number;
    /** Meshes whose rays see tinybvh's tree now. */
    readonly onTinybvhTrees: number;
    /** Meshes with no tree yet, and one to come (rays miss them meanwhile). */
    readonly waitingForFirstTree: number;
    /** tinybvh builds still running. */
    readonly tinybvhBuilding: number;
    /** GPU trees built and usable, since the platform was made. */
    readonly gpuTreesBuilt: number;
    /** GPU trees refused (the builder's checks, too deep, too big, or a failed build), since the platform was made. */
    readonly gpuTreesRefused: number;
    /** GPU trees replaced by tinybvh's, since the platform was made. */
    readonly swaps: number;
    /** Milliseconds from the latest GPU batch's submit to its stats on the CPU; 0 before any. */
    readonly lastGpuBuildMs: number;
}

/** GPU bytes the tracer holds, by buffer. */
export interface AccelerationMemory {
    readonly nodes: number;
    readonly triangles: number;
    readonly topLevel: number;
    /** GPU-built trees in their batches' own buffers, until they are copied into `nodes` and `triangles` (a few frames at load). */
    readonly gpuBuildOutputs: number;
    readonly total: number;
}

/** Starting sizes; both buffers double when they fill. */
const INITIAL_NODES = 1024;
const INITIAL_TRIANGLES = 2048;

/**
 * Room kept free in the node buffer when a scene's trees arrive at once
 * (growable-storage.ts). tinybvh's trees replace the GPU's first trees and
 * can have more nodes (Sponza: 0.6% more in all, more for single meshes;
 * the triangles are the same). Without room, the first replacement that
 * doesn't fit its old range doubled the buffer: Sponza's nodes went from
 * 193 MB to 386 MB (T2.12). An eighth covers the difference with margin.
 */
const NODE_HEADROOM = 1 / 8;

/** Bindings of a batch kernel's own group (group 1): rays in, results out. */
const BATCH_GROUP = 1;

export class PlatformRayQueries implements RayQueries {
    readonly kind = 'rays';
    readonly capabilities = { alphaCutouts: false, backFaceHits: true } as const;
    readonly module: WgslModule;

    private readonly device: GPUDevice;
    private readonly scene: PlatformScene;
    private readonly label: string;
    private readonly warn: (message: string) => void;
    private readonly builder: MeshBvhBuilder;

    private readonly nodes: GrowableStorage;
    private readonly triangles: GrowableStorage;
    private topLevelBuffer: GPUBuffer;
    private retiredTopLevel: GPUBuffer[] = [];
    private bufferVersion = 0;

    /** Bumped every time a mesh slot is uploaded or removed, so a late build for an old version is dropped. */
    private readonly generations = new Map<number, number>();
    /** Each live mesh slot's current version: what each builder has given it. */
    private readonly treeStates = new Map<number, MeshTreeState>();
    private readonly placed = new Map<number, PlacedTree>();
    private arrived: ArrivedTree[] = [];
    /** tinybvh builds still running (any version). */
    private pendingBuilds = 0;
    private buildFailure: Error | null = null;
    /** Set when the top level no longer matches the instances or the placed trees. */
    private topLevelStale = false;
    private builtWaiters: (() => void)[] = [];
    private refinedWaiters: (() => void)[] = [];

    /** The GPU's first trees (T2.12): null without subgroups, with `gpuFirst: false`, or when its kernels failed to compile. */
    private firstTrees: FirstTreeBuilder | null;
    private firstTreesReady = false;
    /**
     * GPU builds are for the scene as first loaded: open until its first trees
     * are all in, then meshes added later go to tinybvh only (runtime GPU
     * builds are M3's T3.2).
     */
    private gpuBuildsOpen: boolean;
    private readonly gpuMaxDepth: number;
    /** Slots whose tinybvh tree waits instead of replacing their GPU tree; null holds none. */
    private refinementHold: ((slot: number) => boolean) | null;
    /** Batch output buffers alive until their trees are copied out (or dropped). */
    private readonly liveOutputs = new Set<GpuTreeOutputs>();
    /** Outputs whose trees this frame copied out or dropped: released at the next update, after the host submitted the copies. */
    private outputsToRelease: GpuTreeOutputs[] = [];
    private gpuTreesBuilt = 0;
    private gpuTreesRefused = 0;
    private swaps = 0;
    private lastGpuBuildMs = 0;

    private closestKernel: ComputeKernel | null = null;
    private anyKernel: ComputeKernel | null = null;
    /** A `prepare` has finished, so missing batch kernels mean nobody asked for them. */
    private preparedOnce = false;
    private disposed = false;

    /** `gpu-refit` on a device with subgroups; `cpu` otherwise. Fixed for the backend's life. */
    readonly topLevelPath: TopLevelPath;
    /** The GPU top level, on the `gpu-refit` path. */
    readonly gpuTopLevel: GpuTopLevel | null;
    /** Instances in the CPU top level (the `cpu` path). */
    private cpuTopLevelInstanceCount = 0;

    constructor(device: GPUDevice, scene: PlatformScene, options: PlatformRayQueriesOptions = {}) {
        this.device = device;
        this.scene = scene;
        this.label = options.label ?? 'gi';
        this.builder = createMeshBvhBuilder({
            worker: options.buildIn === 'main' ? 'never' : 'auto',
            // The inputs are copies made for the build (startBuild), so they move instead of being cloned.
            transferInput: true,
        });

        this.nodes = new GrowableStorage(device, `${this.label} trace nodes`, BVH_NODE_BYTES, INITIAL_NODES, { headroom: NODE_HEADROOM });
        this.triangles = new GrowableStorage(device, `${this.label} trace triangles`, TRIANGLE_BYTES, INITIAL_TRIANGLES);
        // An empty top level: the header says 0 nodes, so every ray misses.
        this.topLevelBuffer = this.createTopLevelBuffer(TOP_LEVEL_HEADER_BYTES);
        this.topLevelPath = GpuBvhRefit.isSupported(device) ? 'gpu-refit' : 'cpu';
        this.gpuTopLevel =
            this.topLevelPath === 'gpu-refit'
                ? new GpuTopLevel(device, scene, { label: this.label, buildIn: options.buildIn, diagnostics: options.diagnostics, warn: options.warn })
                : null;

        const meshTrees = options.meshTrees ?? {};
        this.warn = options.warn ?? ((message) => console.warn(message));
        this.gpuMaxDepth = meshTrees.gpuMaxDepth ?? GPU_TREE_MAX_DEPTH;
        this.refinementHold = meshTrees.holdRefinements ? () => true : null;
        if (meshTrees.gpuFirst === false) this.firstTrees = null;
        else if (meshTrees.firstTreeBuilder) this.firstTrees = meshTrees.firstTreeBuilder;
        else this.firstTrees = GpuMeshTrees.isSupported(device) ? new GpuMeshTrees(device, { label: this.label }) : null;
        this.gpuBuildsOpen = this.firstTrees !== null;

        // The GPU top level replaces its buffer when it grows, so its version counts too.
        const currentVersion = (): number => this.bufferVersion + (this.gpuTopLevel?.version ?? 0);
        this.module = {
            name: 'gi-trace',
            dependencies: [],
            code: (slot: BindingSlot) => traceModuleWgsl(slot),
            entries: TRACE_MODULE_ENTRIES,
            resources: () => [{ buffer: this.nodes.buffer }, { buffer: this.triangles.buffer }, { buffer: this.currentTopLevelBuffer }],
            get version() {
                return currentVersion();
            },
            storageBufferCount: TRACE_MODULE_ENTRIES.length,
        };
    }

    /**
     * True while a mesh waits for its first tree (from the GPU or tinybvh) or
     * a top-level shape is still being built. Rays meanwhile see a mesh's
     * previous shape, or miss it if it is new, and miss instances created (or
     * shown) since the current top-level shape. tinybvh trees still to come
     * for meshes that already have a GPU tree don't count: see `refining`.
     */
    get building(): boolean {
        return this.meshesWaitingForFirstTree() > 0 || (this.gpuTopLevel?.building ?? false);
    }

    /** True while tinybvh trees are still being built (each will replace a mesh's GPU tree, or be its first). */
    get refining(): boolean {
        return this.pendingBuilds > 0;
    }

    /** Which builders the meshes' trees come from now, and what the GPU builds gave. */
    get meshTreeCounts(): MeshTreeCounts {
        let onGpuTrees = 0;
        let onTinybvhTrees = 0;
        for (const tree of this.placed.values()) {
            if (tree.source === 'gpu') onGpuTrees++;
            else onTinybvhTrees++;
        }
        return {
            onGpuTrees,
            onTinybvhTrees,
            waitingForFirstTree: this.meshesWaitingForFirstTree(),
            tinybvhBuilding: this.pendingBuilds,
            gpuTreesBuilt: this.gpuTreesBuilt,
            gpuTreesRefused: this.gpuTreesRefused,
            swaps: this.swaps,
            lastGpuBuildMs: this.lastGpuBuildMs,
        };
    }

    /** `meshTreeCounts` in plain words, for `stats.counters`. */
    get meshTreeCounters(): Readonly<Record<string, number>> {
        const counts = this.meshTreeCounts;
        return {
            'meshes on GPU trees': counts.onGpuTrees,
            'meshes on tinybvh trees': counts.onTinybvhTrees,
            'meshes waiting for a first tree': counts.waitingForFirstTree,
            'tinybvh builds running': counts.tinybvhBuilding,
            'GPU trees built': counts.gpuTreesBuilt,
            'GPU trees refused': counts.gpuTreesRefused,
            'GPU trees replaced by tinybvh': counts.swaps,
            'last GPU build ms (submit to stats)': counts.lastGpuBuildMs,
        };
    }

    /** Instances in the current top level; for stats and tests. */
    get topLevelInstanceCount(): number {
        return this.gpuTopLevel ? this.gpuTopLevel.instanceCount : this.cpuTopLevelInstanceCount;
    }

    /** The GPU top level's counters in plain words, for `stats.counters`; null on the CPU path. */
    get topLevelCounters(): Readonly<Record<string, number>> | null {
        const counters = this.gpuTopLevel?.counters;
        if (!counters) return null;
        return {
            'refit passes': counters.refits,
            'shapes built': counters.shapesBuilt,
            'quality checks': counters.qualityChecks,
            'quality rebuilds': counters.qualityRebuilds,
            'last shape build ms (worker)': counters.lastBuildMs,
            'last snapshot ms (main thread)': counters.lastSnapshotMs,
            'refit reads that gave up (diagnostics)': counters.refitGaveUp,
            'refit repairs (diagnostics)': counters.refitRepairs,
        };
    }

    private get currentTopLevelBuffer(): GPUBuffer {
        return this.gpuTopLevel ? this.gpuTopLevel.buffer : this.topLevelBuffer;
    }

    get memoryBytes(): number {
        return this.memory.total;
    }

    get memory(): AccelerationMemory {
        const nodes = this.nodes.byteLength;
        const triangles = this.triangles.byteLength;
        const topLevel = this.gpuTopLevel ? this.gpuTopLevel.memoryBytes : this.topLevelBuffer.size;
        let gpuBuildOutputs = 0;
        for (const outputs of this.liveOutputs) gpuBuildOutputs += outputs.bytes;
        return { nodes, triangles, topLevel, gpuBuildOutputs, total: nodes + triangles + topLevel + gpuBuildOutputs };
    }

    /** Whether builds run in a worker. */
    get usesWorker(): boolean {
        return this.builder.usesWorker;
    }

    /** Whether the scene's first trees are built on the GPU (a device with subgroups, and `gpuFirst` not turned off). */
    get gpuFirstTrees(): boolean {
        return this.firstTrees !== null;
    }

    /**
     * Resolves once every mesh has its first tree (or none is coming) and no
     * top-level shape is building. They still reach the GPU only at the next
     * `update` (the platform's next `encode`). tinybvh trees that will
     * replace GPU trees may still be building: see `whenRefined`.
     */
    async whenBuilt(): Promise<void> {
        if (this.meshesWaitingForFirstTree() > 0) await new Promise<void>((resolve) => this.builtWaiters.push(resolve));
        await this.gpuTopLevel?.whenIdle();
    }

    /** Resolves once every tinybvh build has finished, as {@link whenBuilt} does; held trees still wait for `releaseRefinements`. */
    async whenRefined(): Promise<void> {
        if (this.pendingBuilds > 0) await new Promise<void>((resolve) => this.refinedWaiters.push(resolve));
        await this.whenBuilt();
    }

    /**
     * Lets tinybvh trees held by `holdRefinements` replace GPU trees from the
     * next `update`: all of them, or only those of the mesh slots `which`
     * accepts (the rest stay held).
     */
    releaseRefinements(which?: (meshSlot: number) => boolean): void {
        const held = this.refinementHold;
        if (!held) return;
        this.refinementHold = which ? (slot) => held(slot) && !which(slot) : null;
    }

    /**
     * Compiles the first-tree builder's kernels (asked for first: the scene's
     * first trees wait on them), the GPU top level's, and the batch kernels
     * when `batches` is true (a technique asked for `encodeBatch`, or a tool
     * that calls it itself). Called by the platform's `prepare`. Everything is
     * asked for at once, so no kernel waits for another to finish compiling.
     */
    async prepare(kernels: KernelBuilder, options: { readonly batches: boolean }): Promise<void> {
        await Promise.all([this.prepareFirstTrees(), this.gpuTopLevel?.prepare(kernels), options.batches ? this.prepareBatches(kernels) : undefined]);
        this.preparedOnce = true;
    }

    /**
     * The first-tree builder's kernels. If they fail to compile, the meshes
     * wait for tinybvh instead (the fallback always stays, DECISIONS
     * 2026-11-05): GI starts later, but it starts.
     */
    private async prepareFirstTrees(): Promise<void> {
        if (!this.firstTrees || this.firstTreesReady || !this.gpuBuildsOpen) return;
        try {
            await this.firstTrees.prepare();
            this.firstTreesReady = true;
        } catch (error) {
            this.firstTrees = null;
            this.gpuBuildsOpen = false;
            this.warn(`@pmndrs/ddgi: the GPU mesh tree builder could not be compiled, so mesh trees come from tinybvh only: ${describe(error)}`);
        }
    }

    /** Whether `encodeBatch` can run: its kernels were compiled by a `prepare` that asked for them. */
    get batchesReady(): boolean {
        return this.closestKernel !== null && this.anyKernel !== null;
    }

    private async prepareBatches(kernels: KernelBuilder): Promise<void> {
        if (this.batchesReady) return;
        const batchGroup: GPUBindGroupLayoutDescriptor = {
            label: `${this.label} trace batch`,
            entries: [
                { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
            ],
        };
        const kernelFor = (mode: 'closest' | 'any') =>
            kernels.computeKernel({
                label: `trace batch ${mode}`,
                modules: [this.module],
                code: batchEntryWgsl(mode, BATCH_GROUP),
                entryPoint: 'giBatch',
                ownGroups: [batchGroup],
            });
        [this.closestKernel, this.anyKernel] = await Promise.all([kernelFor('closest'), kernelFor('any')]);
    }

    /**
     * Brings the tracer's buffers up to date with this frame's scene upload.
     * Called by the platform right after the scene copy's own `encode`, in the
     * same encoder, before any technique runs. `timer` times the top-level
     * refit pass (`platform/top-level refit`) when given.
     */
    update(encoder: GPUCommandEncoder, upload: SceneUpload, timer?: GpuTimerScope): void {
        this.checkLive();
        this.nodes.releaseRetired();
        this.triangles.releaseRetired();
        for (const buffer of this.retiredTopLevel) buffer.destroy();
        this.retiredTopLevel = [];
        // The host has submitted the frame that copied these GPU trees out (or dropped them).
        for (const outputs of this.outputsToRelease) this.releaseOutputs(outputs);
        this.outputsToRelease = [];

        if (upload.instancesChanged) this.topLevelStale = true;
        // Removals first: a slot may be removed and reused by a new mesh in the same frame.
        let meshesChanged = upload.meshesRemoved.length > 0;
        for (const slot of upload.meshesRemoved) {
            this.bumpGeneration(slot);
            this.treeStates.delete(slot);
            this.releaseTree(slot);
            this.topLevelStale = true;
        }
        for (const mesh of upload.meshesUploaded) this.gpuTopLevel?.meshUploaded(mesh.slot, mesh.packed.bounds);
        if (upload.meshesUploaded.length > 0) this.startBuilds(upload.meshesUploaded);
        // A removed mesh may have been the last one waiting.
        if (upload.meshesRemoved.length > 0) this.notifyWaiters();

        // Placing trees can fail (a buffer over the device's binding limit). The top level
        // is still brought up to date with what is placed (the CPU path rebuilds it; the
        // GPU path writes the mesh table, where released trees are marked absent, and
        // refits), so it never points at a released range, and then the error is thrown.
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
            meshesChanged = true;
        }
        // Once the scene's first trees are all placed, meshes added later get tinybvh trees only.
        if (this.gpuBuildsOpen && this.placed.size > 0 && this.meshesWaitingForFirstTree() === 0) this.gpuBuildsOpen = false;
        if (this.gpuTopLevel) {
            // Refit (and swap in a new shape) on the GPU: moves never rebuild on the CPU here.
            try {
                this.gpuTopLevel.update(
                    encoder,
                    {
                        instancesChanged: upload.instancesChanged,
                        instanceSetChanged: upload.instanceSetChanged,
                        meshesChanged,
                        deepestMesh: this.deepestPlacedMesh(),
                    },
                    timer,
                );
            } catch (error) {
                // The placement failure, if any, is the one to report first.
                if (failure === null) failure = error;
            }
            this.topLevelStale = false;
        } else if (this.topLevelStale) {
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
        // Never compiled here on first use: that would be a compile while the platform is
        // ready, which `stats.recompiles` exists to catch (docs/design/m3-moving.md §1).
        if (!kernel) {
            if (!this.preparedOnce) throw new GINotReadyError('preparing');
            throw new Error(
                '@pmndrs/ddgi: rays.encodeBatch needs its kernels compiled in prepare: give the technique that calls it `requirements.rayBatches: true` (tools: the platform option `rayBatches`), then prepare',
            );
        }
        if (batch.count <= 0) return;
        const resultBytes = batch.mode === 'closest' ? HIT_RECORD_BYTES : 4;
        if (batch.rays.size < batch.count * RAY_BYTES) throw new RangeError(`the ray buffer holds fewer than ${batch.count} rays`);
        if (batch.hits.size < batch.count * resultBytes) throw new RangeError(`the hit buffer holds fewer than ${batch.count} results`);

        // The kernel takes the ray count from the size of the bound range, so no uniform is needed.
        const group = this.device.createBindGroup({
            label: `${this.label} trace batch ${batch.mode}`,
            layout: kernel.ownLayouts[0],
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
        this.gpuTopLevel?.dispose();
        this.nodes.destroy();
        this.triangles.destroy();
        for (const buffer of this.retiredTopLevel) buffer.destroy();
        this.topLevelBuffer.destroy();
        for (const outputs of this.liveOutputs) outputs.destroy();
        this.liveOutputs.clear();
        for (const resolve of this.builtWaiters.splice(0)) resolve();
        for (const resolve of this.refinedWaiters.splice(0)) resolve();
    }

    // -----------------------------------------------------------------------
    // Mesh trees
    // -----------------------------------------------------------------------

    private bumpGeneration(slot: number): number {
        const generation = (this.generations.get(slot) ?? 0) + 1;
        this.generations.set(slot, generation);
        return generation;
    }

    /**
     * Meshes whose current version has no tree yet and still has one coming.
     * A mesh's first tree is whichever builder's arrives first (tinybvh's
     * small meshes often beat the GPU's batch, and a busy GPU may be slower
     * than the worker), except where tinybvh's trees are held: there a
     * running GPU build goes first (`takeArrived`).
     */
    private meshesWaitingForFirstTree(): number {
        let waiting = 0;
        for (const [slot, state] of this.treeStates) {
            const tinybvhUsable = state.tinybvh === 'built' && !(state.gpu === 'building' && this.holdsRefinement(slot));
            const hasTree = state.gpu === 'built' || tinybvhUsable;
            const treeComing = state.gpu === 'building' || state.tinybvh === 'building';
            if (!hasTree && treeComing) waiting++;
        }
        return waiting;
    }

    /** Whether tinybvh's tree for this slot waits for `releaseRefinements` (the `holdRefinements` option). */
    private holdsRefinement(slot: number): boolean {
        return this.refinementHold !== null && this.refinementHold(slot);
    }

    /** Resolves `whenBuilt` and `whenRefined` waiters whose condition now holds. */
    private notifyWaiters(): void {
        if (this.meshesWaitingForFirstTree() === 0) for (const resolve of this.builtWaiters.splice(0)) resolve();
        if (this.pendingBuilds === 0) for (const resolve of this.refinedWaiters.splice(0)) resolve();
    }

    /** The slot's state, if `generation` is still its current version. */
    private currentState(slot: number, generation: number): MeshTreeState | undefined {
        const state = this.treeStates.get(slot);
        return state && state.generation === generation ? state : undefined;
    }

    /**
     * Starts the trees of this frame's uploaded meshes: one GPU build for all
     * of them while the scene's first trees are still to come
     * (gpu-mesh-trees.ts), and tinybvh in the worker for every mesh.
     */
    private startBuilds(meshes: readonly UploadedMesh[]): void {
        const builder = this.gpuBuildsOpen && this.firstTreesReady ? this.firstTrees : null;
        const gpuRequests: GpuTreeRequest[] = [];
        for (const mesh of meshes) {
            const generation = this.bumpGeneration(mesh.slot);
            this.treeStates.set(mesh.slot, { generation, gpu: builder ? 'building' : 'none', tinybvh: 'building' });
            // Views of the packed words, no copies: the build writes them to its own buffers before it returns.
            if (builder) gpuRequests.push({ slot: mesh.slot, generation, geometry: packedGeometryView(mesh) });
        }
        if (builder && gpuRequests.length > 0) this.startGpuBuilds(builder, gpuRequests);
        // tinybvh after the GPU's submit, so its copies (main-thread work) don't hold the GPU build back.
        for (const mesh of meshes) this.startTinybvhBuild(mesh);
    }

    private startGpuBuilds(builder: FirstTreeBuilder, requests: readonly GpuTreeRequest[]): void {
        let builds;
        try {
            builds = builder.build(requests);
        } catch (error) {
            this.warn(`@pmndrs/ddgi: the GPU mesh tree build failed, so the meshes wait for tinybvh: ${describe(error)}`);
            for (const { slot, generation } of requests) this.gpuRefused(slot, generation);
            return;
        }
        for (const { slot, generation } of builds.notBuilt) this.gpuRefused(slot, generation);
        for (const batch of builds.batches) void batch.then((result) => this.gpuBatchFinished(result));
    }

    /** A batch's stats are back: usable trees queue for placement; refused ones leave their meshes to tinybvh. */
    private gpuBatchFinished(result: GpuTreeBatchResult): void {
        this.lastGpuBuildMs = result.buildMs;
        for (const outputs of new Set(result.built.map((built) => built.outputs))) {
            if (this.disposed) outputs.destroy();
            else this.liveOutputs.add(outputs);
        }
        if (this.disposed) return;
        for (const { slot, generation } of result.refused) this.gpuRefused(slot, generation);
        if (result.failure) this.warn(`@pmndrs/ddgi: ${result.failure}, so the meshes wait for tinybvh`);
        for (const built of result.built) {
            const state = this.currentState(built.slot, built.generation);
            if (!state || built.tree.maxDepth > this.gpuMaxDepth) {
                // A mesh removed or replaced meanwhile, or a tree too deep for the trace stack.
                if (state) this.gpuRefused(built.slot, built.generation);
                this.outputsToRelease.push(built.outputs);
                continue;
            }
            state.gpu = 'built';
            this.gpuTreesBuilt++;
            this.arrived.push({ source: 'gpu', slot: built.slot, generation: built.generation, built });
        }
        this.notifyWaiters();
    }

    private gpuRefused(slot: number, generation: number): void {
        const state = this.currentState(slot, generation);
        if (!state) return;
        state.gpu = 'refused';
        this.gpuTreesRefused++;
    }

    private startTinybvhBuild(mesh: UploadedMesh): void {
        const generation = this.generations.get(mesh.slot) ?? 0;
        // Compact copies: the packed words also hold normals and uvs, and the scene drops them after upload.
        const geometry = { positions: packedPositions(mesh.packed).slice(), indices: packedIndices(mesh.packed) };
        this.pendingBuilds++;
        const finished = (bvh: MeshBvh | null): void => {
            this.pendingBuilds--;
            const state = this.disposed ? undefined : this.currentState(mesh.slot, generation);
            if (state) {
                state.tinybvh = bvh ? 'built' : 'failed';
                if (bvh) this.arrived.push({ source: 'tinybvh', slot: mesh.slot, generation, bvh });
            }
            this.notifyWaiters();
        };
        this.builder.build(geometry).then(finished, (error: unknown) => {
            if (!this.disposed) this.buildFailure = new Error(`@pmndrs/ddgi: building a mesh tree failed: ${describe(error)}`);
            finished(null);
        });
    }

    /**
     * The finished builds to place this frame, for meshes that still exist in
     * the version they were built from. A GPU tree whose mesh already has
     * tinybvh's (placed, or placed this frame) is dropped: it would only be
     * worse. Where tinybvh's trees are held (`holdRefinements`, for tools),
     * tinybvh's tree waits while the mesh's GPU build runs or would replace
     * its GPU tree, until `releaseRefinements`: the GPU's tree, when there is
     * one, is then always the first, and the swap comes when the tool says.
     */
    private takeArrived(): ArrivedTree[] {
        const taken: ArrivedTree[] = [];
        const waiting: ArrivedTree[] = [];
        const tinybvhNow = new Set<number>();
        // GPU trees of the meshes' current versions only: a stale one must not hold back a newer version's tinybvh tree.
        const gpuQueued = new Set(this.arrived.filter((tree) => tree.source === 'gpu' && this.currentState(tree.slot, tree.generation)).map((tree) => tree.slot));
        for (const tree of this.arrived) {
            const state = this.currentState(tree.slot, tree.generation);
            if (!state || tree.source !== 'tinybvh') continue;
            const placed = this.placed.get(tree.slot);
            const gpuTreeFirst = state.gpu === 'building' || (placed?.source === 'gpu' && placed.generation === tree.generation) || gpuQueued.has(tree.slot);
            if (gpuTreeFirst && this.holdsRefinement(tree.slot)) {
                waiting.push(tree);
            } else {
                taken.push(tree);
                tinybvhNow.add(tree.slot);
            }
        }
        for (const tree of this.arrived) {
            if (tree.source !== 'gpu') continue;
            const placed = this.placed.get(tree.slot);
            const hasTinybvhTree = tinybvhNow.has(tree.slot) || (placed?.source === 'tinybvh' && placed.generation === tree.generation);
            if (this.currentState(tree.slot, tree.generation) && !hasTinybvhTree) taken.push(tree);
            else this.outputsToRelease.push(tree.built.outputs);
        }
        this.arrived = waiting;
        return taken;
    }

    private releaseTree(slot: number): void {
        const tree = this.placed.get(slot);
        if (!tree) return;
        this.nodes.release(tree.nodeOffset, tree.nodeCount);
        this.triangles.release(tree.triangleOffset, tree.triangleCount);
        this.placed.delete(slot);
        this.gpuTopLevel?.setMeshTree(slot, null);
    }

    /** One tree is done with its batch's outputs; they are freed when no tree needs them any more. */
    private releaseOutputs(outputs: GpuTreeOutputs): void {
        outputs.release();
        if (outputs.isDestroyed) this.liveOutputs.delete(outputs);
    }

    /**
     * Places arrived trees into the shared buffers, growing them first if
     * needed: tinybvh's are written from the CPU, GPU-built ones copied from
     * their batch's buffers in `encoder`, after any growth copies (the
     * allocation records those first).
     */
    private placeTrees(encoder: GPUCommandEncoder, arrived: readonly ArrivedTree[]): void {
        // A rebuilt mesh replaces its old tree, and tinybvh's replaces the GPU's: free the old one
        // first so the space can be reused. Queue order makes reuse safe: frames already submitted
        // read the old tree before any write or copy below runs.
        const swaps = arrived.filter((tree) => tree.source === 'tinybvh' && this.placed.get(tree.slot)?.source === 'gpu').length;
        for (const tree of arrived) this.releaseTree(tree.slot);

        const sizes = arrived.map(treeSize);
        const nodeCounts = sizes.map((size) => size.nodeCount);
        const triangleCounts = sizes.map((size) => size.triangleCount);
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
        this.swaps += swaps;

        arrived.forEach((tree, index) => {
            const size = sizes[index];
            if (tree.source === 'tinybvh') {
                this.nodes.write(nodeOffsets[index], tree.bvh.nodes);
                this.triangles.write(triangleOffsets[index], tree.bvh.triangles);
            } else {
                const { outputs, tree: built } = tree.built;
                encoder.copyBufferToBuffer(
                    outputs.nodes,
                    built.nodeOffset * BVH_NODE_BYTES,
                    this.nodes.buffer,
                    nodeOffsets[index] * BVH_NODE_BYTES,
                    size.nodeCount * BVH_NODE_BYTES,
                );
                encoder.copyBufferToBuffer(
                    outputs.triangles,
                    built.triangleOffset * TRIANGLE_BYTES,
                    this.triangles.buffer,
                    triangleOffsets[index] * TRIANGLE_BYTES,
                    size.triangleCount * TRIANGLE_BYTES,
                );
                // Let go of at the next update, once the host has submitted this frame's copies.
                this.outputsToRelease.push(outputs);
            }
            const placed: PlacedTree = {
                nodeOffset: nodeOffsets[index],
                nodeCount: size.nodeCount,
                triangleOffset: triangleOffsets[index],
                triangleCount: size.triangleCount,
                maxDepth: size.maxDepth,
                bounds: size.bounds,
                source: tree.source,
                generation: tree.generation,
            };
            this.placed.set(tree.slot, placed);
            this.gpuTopLevel?.setMeshTree(tree.slot, placed);
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

    private deepestPlacedMesh(): number {
        let deepest = 0;
        for (const tree of this.placed.values()) deepest = Math.max(deepest, tree.maxDepth);
        return deepest;
    }

    /** The CPU fallback: rebuilds the whole top level on this thread. */
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
            // One binding and one buffer must both hold it; name whichever limit is the smaller.
            const limits = this.device.limits;
            const limitBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
            if (bytes > limitBytes) {
                const limit = limits.maxStorageBufferBindingSize <= limits.maxBufferSize ? 'maxStorageBufferBindingSize' : 'maxBufferSize';
                throw new GILimitError(limit, bytes, limitBytes);
            }
            this.retiredTopLevel.push(this.topLevelBuffer);
            // Room to grow, so a scene that keeps adding instances doesn't reallocate every frame.
            // The cap is rounded down to whole vec4s, since the buffer size is rounded up to them.
            const cap = Math.floor(limitBytes / 16) * 16;
            this.topLevelBuffer = this.createTopLevelBuffer(Math.min(cap, Math.max(bytes, this.topLevelBuffer.size * 2)));
            this.bufferVersion++;
        }
        this.device.queue.writeBuffer(this.topLevelBuffer, 0, topLevel.buffer);
        this.cpuTopLevelInstanceCount = topLevel.instanceCount;
    }

    private createTopLevelBuffer(size: number): GPUBuffer {
        return this.device.createBuffer({
            label: `${this.label} trace top level`,
            size: Math.ceil(size / 16) * 16,
            // COPY_SRC lets tests and debug tools read the tree back.
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }

    private checkLive(): void {
        if (this.disposed) throw new Error('@pmndrs/ddgi: the ray queries were disposed');
    }
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

/** A tree's counts, depth and box, whichever builder made it. */
function treeSize(tree: ArrivedTree): { readonly nodeCount: number; readonly triangleCount: number; readonly maxDepth: number; readonly bounds: Float32Array } {
    return tree.source === 'tinybvh' ? tree.bvh : tree.built.tree;
}

/**
 * A mesh's positions and indices as views of its packed words (mesh-packing.ts), for the GPU
 * build: no copies. 16-bit indices sit two to a word, low half first, which is exactly a
 * little-endian `Uint16Array`.
 */
function packedGeometryView(mesh: UploadedMesh): MeshGeometry {
    const { words, offsets, triangleCount, indices16 } = mesh.packed;
    const indexCount = triangleCount * 3;
    const indexByteStart = words.byteOffset + offsets.indices * 4;
    const indices = indices16 ? new Uint16Array(words.buffer, indexByteStart, indexCount) : new Uint32Array(words.buffer, indexByteStart, indexCount);
    return { positions: packedPositions(mesh.packed), indices };
}
