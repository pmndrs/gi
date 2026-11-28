/**
 * The `rays` query backend over the platform's scene copy: what techniques
 * get as `context.queries.rays`.
 *
 * It keeps the tracer's GPU buffers in step with the scene:
 *
 * - **Mesh trees come from the GPU** (T2.12, "fast first GI"; T3.2 at
 *   runtime; final since #26). On a device with subgroups, every mesh the
 *   scene uploads is built on the GPU with H-PLOC: the scene as it is first
 *   loaded, and meshes added or replaced (`updateMesh`) while it runs. A
 *   frame's uploads go in one batch (gpu-mesh-trees.ts). Their trees arrive a
 *   few frames later, after a small stats readback, and are copied into the
 *   shared node and triangle buffers at the start of a later `encode`. Until
 *   then rays see the mesh's previous shape, or miss it if it is new
 *   (`building` is true meanwhile). The GPU's tree is the mesh's tree: it
 *   traces as fast as tinybvh's (DECISIONS 2026-11-28), so nothing replaces it.
 * - **tinybvh, the declared fallback** (DECISIONS 2026-11-04 and
 *   2026-11-09). tinybvh (WASM) builds a mesh's tree off the main thread only
 *   when the GPU gives that version of the mesh no usable tree: no subgroups,
 *   the GPU builder off (`gpuFirst: false`) or its kernels failed to compile,
 *   or a GPU tree refused (the builder's checks, too big for the device's
 *   buffers, too deep for the trace stack, a failed batch, or a deforming
 *   mesh whose layout readback failed). The scene's CPU copy of the mesh is
 *   kept until the GPU's answer, so the fallback can start then.
 * - **One tinybvh build per mesh at a time.** A mesh replaced again while
 *   its tinybvh build runs keeps only its newest version waiting behind it.
 *   Why: the worker builds in order, and a host that reshapes a mesh every
 *   few frames would otherwise queue builds faster than the worker clears
 *   them, and the mesh's tree would come later and later.
 * - **Deforming meshes** (T3.3, deforming-trees.ts). A pose doesn't build a
 *   tree: the mesh's placed tree is refit on the GPU in the same frame, from
 *   the positions the scene copy just took, and its top-level box is the
 *   mesh's `deformBounds` box. Their GPU builds read the tree's layout back
 *   (the refit needs its shape), in a batch of their own. A tree whose refit
 *   has drifted for about a second is rebuilt with H-PLOC from the scene
 *   copy's positions and swapped in like any GPU tree.
 * - **The top level** (stage B1a). On a device with subgroups it lives on the
 *   GPU (top-level-gpu.ts): refit every frame something moved, from the
 *   scene copy's transforms, with no CPU work per instance; a new shape is
 *   built in a worker now and then. Without subgroups, the declared fallback
 *   (DECISIONS 2026-11-04): it is rebuilt on the CPU, on this thread, when an
 *   instance, a transform or a tree changed. `topLevelPath` says which.
 *
 * - **Materials in the walk** (T4.2). The walk consults a triangle's material
 *   only where the triangle's flag bit 0 says so (material-test-wgsl.ts), and
 *   `gi/triangle-flags` (triangle-flags.ts) keeps that bit right: it runs on
 *   every mesh whose tree is placed (`requestTriangleFlags`, the install hook
 *   every path that writes tree triangles calls), and on the meshes of
 *   instances whose materials changed. So the `rays` module depends on the
 *   scene module, whose material tables the walk reads.
 *
 * `GiHit.instance` is the scene copy's instance slot, so a hit reads its
 * surface straight through the scene module (`giInstanceTriangle(hit.instance,
 * hit.triangle)`), and `GiHit.triangle` is the triangle's index in the mesh as
 * the host gave it.
 *
 * Nothing here recompiles: the WGSL is fixed, and buffers that grow bump
 * `module.version` so kernels rebuild their bind group only.
 */

import { GpuBvhRefit } from '@pmndrs/bvh';
import { GILimitError, GINotReadyError } from '../contract/errors.ts';
import type { InstanceHandle } from '../contract/handles.ts';
import type { RayQueries } from '../contract/queries.ts';
import type { SceneChanges } from '../contract/scene-copy.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { BindingSlot, ComputeKernel, KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import { SHADER_STAGE_COMPUTE } from '../kernel-builder.ts';
import { MATERIAL_FLAG, MATERIAL_WORD, NONE } from '../scene/layout.ts';
import type { PlatformScene, PosedMesh, SceneUpload, UploadedMesh } from '../scene/platform-scene.ts';
import { DeformingTrees, type DeformingTreeCounts } from './deforming-trees.ts';
import { GpuMeshTrees, type FirstTreeBuilder, type GpuBuiltTree, type GpuTreeBatchResult, type GpuTreeOutputs, type GpuTreeRequest } from './gpu-mesh-trees.ts';
import { GrowableStorage, type ResidentRange } from './growable-storage.ts';
import { BVH_NODE_BYTES, TOP_LEVEL_HEADER_BYTES, TRACE_STACK_SIZE, TRIANGLE_BYTES } from './layout.ts';
import type { MeshBvh } from './mesh-bvh.ts';
import { createMeshBvhBuilder, type MeshBvhBuilder } from './mesh-bvh-async.ts';
import { createMaterialTestModule, MATERIAL_TEST_UNIFORM_BYTES } from './material-test-wgsl.ts';
import { buildTopLevel, type TraceInstance } from './top-level.ts';
import { GpuTopLevel } from './top-level-gpu.ts';
import { RAY_BYTES } from './trace-gpu.ts';
import { TriangleFlagsKernels, type MeshTriangleRange } from './triangle-flags.ts';
import { BATCH_WORKGROUP_SIZE, batchEntryWgsl, HIT_RECORD_BYTES, TRACE_MODULE_ENTRIES, traceModuleWgsl } from './wgsl.ts';

export interface PlatformRayQueriesOptions {
    /** Prefix for GPU object labels. */
    readonly label?: string;
    /** `worker` (default) builds mesh trees and top-level shapes in module workers; `main` on this thread. */
    readonly buildIn?: 'worker' | 'main';
    /**
     * Read the GPU top level's refit stats back, for counters and warnings.
     * Off by default (hosts are promised no readbacks frame after frame); the platform turns it
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
    /**
     * Mesh trees from the GPU (H-PLOC) for every uploaded mesh (at load, and added or replaced
     * later), on a device with subgroups. Default true; false forces the fallback, tinybvh only
     * (M1's path), so tools can test it on a device that has subgroups.
     */
    readonly gpuFirst?: boolean;
    /** GPU trees deeper than this are refused, and the mesh falls back to tinybvh. Default {@link GPU_TREE_MAX_DEPTH}. */
    readonly gpuMaxDepth?: number;
    /**
     * Builds the trees instead of the GPU H-PLOC. For unit tests, which
     * run in Node without a GPU; it is used whatever the device's features.
     */
    readonly firstTreeBuilder?: FirstTreeBuilder;
}

/**
 * Levels kept free on the trace stack for the top level and the instance
 * marker. The stack (64 entries) is shared by both levels, so a GPU tree
 * deeper than {@link GPU_TREE_MAX_DEPTH} falls back to tinybvh's rather than
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

/** A finished build waiting to be placed at a later `update`: the GPU's, in its batch's buffers, or tinybvh's (the fallback), from the worker. */
type ArrivedTree =
    | { readonly source: 'tinybvh'; readonly slot: number; readonly generation: number; readonly bvh: MeshBvh }
    | {
          readonly source: 'gpu';
          readonly slot: number;
          readonly generation: number;
          readonly built: GpuBuiltTree;
          /** A deforming mesh's drift rebuild: it replaces whatever tree the mesh has, tinybvh's too. */
          readonly rebuild: boolean;
      };

/**
 * What one version of a mesh (one upload of a slot) has from each builder.
 * `gpu` is `'none'` when the mesh wasn't built on the GPU: no subgroups, the
 * GPU builder off, or its kernels not compiled. tinybvh is `'none'` unless
 * the GPU gave this version no tree (the fallback); its `'building'`
 * includes a build waiting for the slot's previous one to end.
 */
interface MeshTreeState {
    readonly generation: number;
    gpu: 'none' | 'building' | 'built' | 'refused';
    tinybvh: 'none' | 'building' | 'built' | 'failed';
}

/** Counts of the mesh trees, for `stats.counters` and the tools. */
export interface MeshTreeCounts {
    /** Meshes whose rays see a GPU-built tree now. */
    readonly onGpuTrees: number;
    /** Meshes whose rays see tinybvh's tree now (the fallback). */
    readonly onTinybvhTrees: number;
    /** Meshes with no tree yet, and one to come (rays miss them meanwhile). */
    readonly waitingForFirstTree: number;
    /** tinybvh builds still running, or waiting for the same mesh's running build to end. */
    readonly tinybvhBuilding: number;
    /** GPU trees built and usable, since the platform was made. */
    readonly gpuTreesBuilt: number;
    /** GPU trees refused (the builder's checks, too deep, too big, a failed build, a lost layout), since the platform was made. Each such mesh falls back to tinybvh. */
    readonly gpuTreesRefused: number;
    /** Milliseconds from the latest GPU batch's submit to its stats on the CPU; 0 before any. */
    readonly lastGpuBuildMs: number;
}

/** GPU bytes the tracer holds, by buffer. */
export interface AccelerationMemory {
    readonly nodes: number;
    /** Of `nodes`, the bytes holding placed trees (the rest is free room). */
    readonly nodesInUse: number;
    readonly triangles: number;
    /** Of `triangles`, the bytes holding placed trees' triangle records. */
    readonly trianglesInUse: number;
    readonly topLevel: number;
    /** GPU-built trees in their batches' own buffers, until they are copied into `nodes` and `triangles` (a few frames at load). */
    readonly gpuBuildOutputs: number;
    /** Deforming meshes' refits: their topology, scratch and stats (the trees are in `nodes` and `triangles`). */
    readonly meshRefits: number;
    readonly total: number;
}

/** Starting sizes; both buffers double when they fill. */
const INITIAL_NODES = 1024;
const INITIAL_TRIANGLES = 2048;

/** Bindings of a batch kernel's own group (group 1): rays in, results out. */
const BATCH_GROUP = 1;

export class PlatformRayQueries implements RayQueries {
    readonly kind = 'rays';
    readonly capabilities = { alphaCutouts: true, perTriangleMaterials: true, backFaceHits: true } as const;
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
    /** tinybvh builds still running or waiting to start (any version). */
    private pendingBuilds = 0;
    /** Mesh slots with a tinybvh build in the worker now. */
    private readonly tinybvhRunning = new Set<number>();
    /** The newest upload of a slot whose tinybvh build waits for that slot's running one to end. */
    private readonly tinybvhWaiting = new Map<number, { readonly mesh: UploadedMesh; readonly generation: number }>();
    /**
     * Uploads whose GPU build is still running, kept for the tinybvh fallback
     * should the GPU refuse them: the scene's CPU copy of the mesh
     * (`UploadedMesh.geometry`), let go of once the GPU's tree is usable.
     */
    private readonly awaitingGpu = new Map<number, { readonly mesh: UploadedMesh; readonly generation: number }>();
    private buildFailure: Error | null = null;
    /** Set when the top level no longer matches the instances or the placed trees. */
    private topLevelStale = false;
    private builtWaiters: (() => void)[] = [];

    /**
     * The GPU's mesh trees (T2.12 at load, T3.2 at runtime): null without
     * subgroups, with `gpuFirst: false`, or when its kernels failed to compile.
     */
    private firstTrees: FirstTreeBuilder | null;
    private firstTreesReady = false;
    private readonly gpuMaxDepth: number;
    /** Batch output buffers alive until their trees are copied out (or dropped). */
    private readonly liveOutputs = new Set<GpuTreeOutputs>();
    /** Outputs whose trees this frame copied out or dropped: released at the next update, after the host submitted the copies. */
    private outputsToRelease: GpuTreeOutputs[] = [];
    private gpuTreesBuilt = 0;
    private gpuTreesRefused = 0;
    private lastGpuBuildMs = 0;

    /** `gi/triangle-flags`: keeps triangle flag bit 0 ("consult the material") right (T4.2). */
    private readonly triangleFlags: TriangleFlagsKernels;
    /** Mesh slots whose triangle flags the next `update` rewrites (`requestTriangleFlags`). */
    private readonly flagRequests = new Set<number>();
    private flagRewrites = 0;
    /**
     * Per instance handle, what the flags last saw of it: its mesh slot and the
     * record words the walk's rules read (`walkKey`). A material change that
     * leaves these alone (a colour, an emissive, a uv transform) rewrites nothing.
     */
    private readonly walkKeys = new Map<number, string>();
    /** The walk's frame number for the stochastic hash (material-test-wgsl.ts), one more each `update`. */
    private readonly walkFrameUniform: GPUBuffer;
    private walkFrame = 0;

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
    /** Deforming meshes' refits and drift rebuilds; null without subgroups (they stay in their rest pose then). */
    private readonly deforming: DeformingTrees | null;
    /** `update` calls so far: the clock the drift rule counts frames with. */
    private updates = 0;

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
        this.topLevelPath = GpuBvhRefit.isSupported(device) ? 'gpu-refit' : 'cpu';
        this.gpuTopLevel =
            this.topLevelPath === 'gpu-refit'
                ? new GpuTopLevel(device, scene, { label: this.label, buildIn: options.buildIn, diagnostics: options.diagnostics, warn: options.warn })
                : null;

        const meshTrees = options.meshTrees ?? {};
        this.warn = options.warn ?? ((message) => console.warn(message));
        this.gpuMaxDepth = meshTrees.gpuMaxDepth ?? GPU_TREE_MAX_DEPTH;
        if (meshTrees.gpuFirst === false) this.firstTrees = null;
        else if (meshTrees.firstTreeBuilder) this.firstTrees = meshTrees.firstTreeBuilder;
        else this.firstTrees = GpuMeshTrees.isSupported(device) ? new GpuMeshTrees(device, { label: this.label }) : null;
        this.deforming = DeformingTrees.isSupported(device) ? new DeformingTrees(device, { label: this.label }) : null;

        // The walk reads materials from the scene copy (T4.2), through the binding-free material test
        // module, and cutout alpha from the texture store through its sampling module: the same
        // object hit shading lists, so a kernel with both binds the store once.
        this.walkFrameUniform = device.createBuffer({
            label: `${this.label} trace walk frame`,
            size: MATERIAL_TEST_UNIFORM_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const materialTest = createMaterialTestModule(scene.module, scene.textureStore.samplingModule(scene.module), this.walkFrameUniform);
        this.triangleFlags = new TriangleFlagsKernels(device, materialTest, this.label);

        // The GPU top level replaces its buffer when it grows, so its version counts too.
        const currentVersion = (): number => this.bufferVersion + (this.gpuTopLevel?.version ?? 0);
        this.module = {
            name: 'gi-trace',
            dependencies: [materialTest],
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
     * True while a mesh waits for its tree (from the GPU, or the tinybvh
     * fallback) or a top-level shape is still being built. Rays meanwhile see
     * a mesh's previous shape, or miss it if it is new, and miss instances
     * created (or shown) since the current top-level shape. A deforming
     * mesh's drift rebuild doesn't count: the mesh traces on its refit tree.
     */
    get building(): boolean {
        return this.meshesWaitingForFirstTree() > 0 || (this.gpuTopLevel?.building ?? false);
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
            'last GPU build ms (submit to stats)': counts.lastGpuBuildMs,
        };
    }

    /** Deforming meshes' refits and drift rebuilds, or null without subgroups. */
    get deformingTreeCounts(): DeformingTreeCounts | null {
        return this.deforming?.counts ?? null;
    }

    /** `deformingTreeCounts` in plain words, for `stats.counters`; null without subgroups. */
    get deformingTreeCounters(): Readonly<Record<string, number>> | null {
        const counts = this.deforming?.counts;
        if (!counts) return null;
        return {
            'deforming meshes refit': counts.refitting,
            'mesh refits': counts.refits,
            'deforming trees rebuilt after drift': counts.driftRebuilds,
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
        const meshRefits = this.deforming?.memoryBytes ?? 0;
        return {
            nodes,
            nodesInUse: this.nodes.usedBytes,
            triangles,
            trianglesInUse: this.triangles.usedBytes,
            topLevel,
            gpuBuildOutputs,
            meshRefits,
            total: nodes + triangles + topLevel + gpuBuildOutputs + meshRefits,
        };
    }

    /** Whether builds run in a worker. */
    get usesWorker(): boolean {
        return this.builder.usesWorker;
    }

    /** Whether meshes' trees are built on the GPU (a device with subgroups, `gpuFirst` not turned off, and the kernels compiled). */
    get gpuFirstTrees(): boolean {
        return this.firstTrees !== null;
    }

    /**
     * Resolves once every mesh has its tree (or none is coming), the tinybvh
     * fallback's included, and no top-level shape is building. They still
     * reach the GPU only at the next `update` (the platform's next `encode`).
     */
    async whenBuilt(): Promise<void> {
        if (this.meshesWaitingForFirstTree() > 0) await new Promise<void>((resolve) => this.builtWaiters.push(resolve));
        await this.gpuTopLevel?.whenIdle();
    }

    /** Meshes whose triangle flags `gi/triangle-flags` has rewritten, since the platform was made (a mesh rewritten twice counts twice). */
    get triangleFlagRewrites(): number {
        return this.flagRewrites;
    }

    /**
     * **The install hook** (T4.2; triangle-flags.ts says why). Every path that
     * writes a mesh's tree triangles into the shared triangle buffer calls
     * this with the mesh's slot: the next `update` (this frame's, when called
     * before it or from inside it) then rewrites the mesh's triangle flags
     * from the scene copy's materials, after every placement, in the frame's
     * encoder. A slot with no placed tree by then is skipped: its placement
     * asks again.
     */
    requestTriangleFlags(meshSlot: number): void {
        this.flagRequests.add(meshSlot);
    }

    /**
     * Compiles the first-tree builder's kernels (asked for first: the scene's
     * first trees wait on them), `gi/triangle-flags`, the GPU top level's, and the batch kernels
     * when `batches` is true (a technique asked for `encodeBatch`, or a tool
     * that calls it itself). Called by the platform's `prepare`. Everything is
     * asked for at once, so no kernel waits for another to finish compiling.
     */
    async prepare(kernels: KernelBuilder, options: { readonly batches: boolean }): Promise<void> {
        await Promise.all([
            this.prepareFirstTrees(),
            this.triangleFlags.prepare(kernels),
            this.gpuTopLevel?.prepare(kernels),
            this.deforming?.prepare(),
            options.batches ? this.prepareBatches(kernels) : undefined,
        ]);
        this.preparedOnce = true;
    }

    /**
     * The first-tree builder's kernels. If they fail to compile, the meshes
     * wait for tinybvh instead (the fallback always stays, DECISIONS
     * 2026-11-05): GI starts later, but it starts.
     */
    private async prepareFirstTrees(): Promise<void> {
        if (!this.firstTrees || this.firstTreesReady) return;
        try {
            await this.firstTrees.prepare();
            this.firstTreesReady = true;
        } catch (error) {
            this.firstTrees = null;
            this.warn(`@pmndrs/gi: the GPU mesh tree builder could not be compiled, so mesh trees come from tinybvh only: ${describe(error)}`);
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
        this.updates++;
        // Stats copied in earlier frames (the host has submitted them since): the drift rule reads them.
        this.deforming?.pollStats();
        this.nodes.releaseRetired();
        this.triangles.releaseRetired();
        for (const buffer of this.retiredTopLevel) buffer.destroy();
        this.retiredTopLevel = [];
        // The host has submitted the frame that copied these GPU trees out (or dropped them).
        for (const outputs of this.outputsToRelease) this.releaseOutputs(outputs);
        this.outputsToRelease = [];

        // A new frame number for the stochastic hash: rays that repeat exactly get new answers each frame.
        this.walkFrame = (this.walkFrame + 1) >>> 0;
        this.device.queue.writeBuffer(this.walkFrameUniform, 0, new Uint32Array([this.walkFrame, 0, 0, 0]));

        if (upload.instancesChanged) this.topLevelStale = true;
        // Removals first: a slot may be removed and reused by a new mesh in the same frame.
        let meshesChanged = upload.meshesRemoved.length > 0;
        for (const slot of upload.meshesRemoved) {
            this.bumpGeneration(slot);
            this.treeStates.delete(slot);
            this.awaitingGpu.delete(slot);
            this.dropWaitingTinybvhBuild(slot);
            this.releaseTree(slot);
            this.deforming?.meshRemoved(slot);
            this.topLevelStale = true;
        }
        for (const mesh of upload.meshesUploaded) {
            this.gpuTopLevel?.meshUploaded(mesh.slot, mesh.layout.bounds);
            this.deforming?.meshUploaded(mesh.slot, mesh.layout, mesh.vertexOffset);
        }
        if (this.posesTaken(upload.meshesPosed)) meshesChanged = true;
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
        // The frame's poses and the trees placed above, refit from the scene copy's positions before anything traces.
        this.deforming?.encode(encoder, { vertices: this.scene.vertexStorage, nodes: this.nodes.buffer, triangles: this.triangles.buffer }, this.updates, timer);
        this.startDriftRebuild(upload);
        // Material changes can flip a mesh's slots between "consult" and not; placed trees
        // asked already (placeTrees). Then one rewrite for all of them, after every write.
        this.requestTriangleFlagsForChanges(upload.changes);
        this.encodeTriangleFlags(encoder);
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
                '@pmndrs/gi: rays.encodeBatch needs its kernels compiled in prepare: give the technique that calls it `requirements.rayBatches: true` (tools: the platform option `rayBatches`), then prepare',
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
        this.triangleFlags.dispose();
        this.walkFrameUniform.destroy();
        this.deforming?.dispose();
        this.nodes.destroy();
        this.triangles.destroy();
        for (const buffer of this.retiredTopLevel) buffer.destroy();
        this.topLevelBuffer.destroy();
        for (const outputs of this.liveOutputs) outputs.destroy();
        this.liveOutputs.clear();
        this.tinybvhWaiting.clear();
        this.awaitingGpu.clear();
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

    /**
     * Meshes whose current version has no tree yet and still has one coming:
     * the GPU's, or tinybvh's after the GPU refused it (one builder at a time,
     * so never both).
     */
    private meshesWaitingForFirstTree(): number {
        let waiting = 0;
        for (const state of this.treeStates.values()) {
            const hasTree = state.gpu === 'built' || state.tinybvh === 'built';
            const treeComing = state.gpu === 'building' || state.tinybvh === 'building';
            if (!hasTree && treeComing) waiting++;
        }
        return waiting;
    }

    /** Resolves `whenBuilt` waiters once no mesh waits for its tree. */
    private notifyWaiters(): void {
        if (this.meshesWaitingForFirstTree() === 0) for (const resolve of this.builtWaiters.splice(0)) resolve();
    }

    /** The slot's state, if `generation` is still its current version. */
    private currentState(slot: number, generation: number): MeshTreeState | undefined {
        const state = this.treeStates.get(slot);
        return state && state.generation === generation ? state : undefined;
    }

    /**
     * Starts the trees of this frame's uploaded meshes: one GPU build for all
     * of them (gpu-mesh-trees.ts), whether the scene is loading or running.
     * Without the GPU builder, tinybvh builds every mesh in the worker (the
     * fallback); with it, tinybvh builds only the meshes the GPU refuses.
     */
    private startBuilds(meshes: readonly UploadedMesh[]): void {
        const builder = this.firstTreesReady ? this.firstTrees : null;
        const gpuRequests: GpuTreeRequest[] = [];
        for (const mesh of meshes) {
            const generation = this.bumpGeneration(mesh.slot);
            this.treeStates.set(mesh.slot, { generation, gpu: builder ? 'building' : 'none', tinybvh: 'none' });
            if (!builder) {
                this.startTinybvhBuild(mesh, generation);
                continue;
            }
            // The scene's copies of the mesh's positions and indices (UploadedMesh.geometry). The GPU build
            // writes them into its own buffers (queue.writeBuffer) before build() returns; they are kept
            // until the GPU's answer, for the fallback (a refusal can come from inside build()).
            this.awaitingGpu.set(mesh.slot, { mesh, generation });
            gpuRequests.push({ slot: mesh.slot, generation, geometry: mesh.geometry });
        }
        if (builder && gpuRequests.length > 0) {
            // Deforming meshes' trees are read back too (the refit needs their shape), so they get a batch
            // of their own: the readback then holds only their trees, never a whole scene's.
            const deforming = gpuRequests.filter((request) => this.deforming?.has(request.slot));
            const others = gpuRequests.filter((request) => !this.deforming?.has(request.slot));
            if (others.length > 0) this.startGpuBuilds(builder, others, { readLayouts: false, rebuild: false });
            if (deforming.length > 0) this.startGpuBuilds(builder, deforming, { readLayouts: true, rebuild: false });
        }
    }

    /**
     * Starts one GPU build. `rebuild` marks a deforming mesh's drift rebuild:
     * its mesh already has a tree, so a refused rebuild leaves the mesh as it
     * is instead of falling back to tinybvh. Returns whether anything started.
     * Refusals are handled only after `builder.build` returns, so the GPU's
     * `writeBuffer`s have copied every mesh before a fallback build transfers
     * (detaches) its arrays to the worker.
     */
    private startGpuBuilds(builder: FirstTreeBuilder, requests: readonly GpuTreeRequest[], kind: { readonly readLayouts: boolean; readonly rebuild: boolean }): boolean {
        const refuse = (slot: number, generation: number): void => {
            if (kind.rebuild) this.deforming?.rebuildRefused(slot);
            else this.gpuRefused(slot, generation);
        };
        let builds;
        try {
            builds = builder.build(requests, { readLayouts: kind.readLayouts });
        } catch (error) {
            const what = kind.rebuild ? 'the mesh keeps its refit tree' : 'the meshes fall back to tinybvh';
            this.warn(`@pmndrs/gi: the GPU mesh tree build failed, so ${what}: ${describe(error)}`);
            for (const { slot, generation } of requests) refuse(slot, generation);
            return false;
        }
        for (const { slot, generation } of builds.notBuilt) refuse(slot, generation);
        for (const batch of builds.batches) void batch.then((result) => this.gpuBatchFinished(result, kind.rebuild));
        return builds.batches.length > 0;
    }

    /**
     * Takes the frame's poses: each deforming mesh's refit is due, and a mesh
     * whose box grew (a CPU pose that left its `deformBounds`) gets the bigger
     * box in the top level's mesh table, so rays find the whole pose. Returns
     * whether a placed tree's box changed.
     */
    private posesTaken(posed: readonly PosedMesh[]): boolean {
        let boxChanged = false;
        for (const mesh of posed) {
            this.deforming?.posed(mesh.slot, mesh.layout, mesh.vertexOffset);
            const tree = this.placed.get(mesh.slot);
            if (!tree || sameBox(tree.bounds, mesh.layout.bounds)) continue;
            const grown: PlacedTree = { ...tree, bounds: Float32Array.from(mesh.layout.bounds) };
            this.placed.set(mesh.slot, grown);
            this.gpuTopLevel?.setMeshTree(mesh.slot, grown);
            this.topLevelStale = true;
            boxChanged = true;
        }
        return boxChanged;
    }

    /**
     * Starts the drift rebuild of the deforming mesh whose refit has drifted
     * longest, once it has held (deforming-trees.ts), with H-PLOC from the
     * scene copy's positions. Skipped while that mesh still has its tree
     * coming (from the GPU or the fallback), and in a frame
     * whose scene upload made a new vertex buffer: the build copies the
     * positions in its own submit, before the host's, and a new buffer holds
     * nothing until the host's growth copies run.
     */
    private startDriftRebuild(upload: SceneUpload): void {
        const deforming = this.deforming;
        if (!deforming) return;
        const slot = deforming.rebuildDue(this.updates);
        const builder = this.firstTreesReady ? this.firstTrees : null;
        if (slot === null || !builder || upload.reallocated) return;
        const state = this.treeStates.get(slot);
        if (!state || state.gpu === 'building' || state.tinybvh === 'building') return;
        const geometry = deforming.rebuildGeometry(slot, this.scene.vertexStorage);
        if (!geometry) return;
        const started = this.startGpuBuilds(builder, [{ slot, generation: state.generation, geometry }], { readLayouts: true, rebuild: true });
        deforming.rebuildStarted(slot, started);
    }

    /** A batch's stats are back: usable trees queue for placement; refused ones fall back to tinybvh (or, rebuilt, keep their refit tree). */
    private gpuBatchFinished(result: GpuTreeBatchResult, rebuild: boolean): void {
        this.lastGpuBuildMs = result.buildMs;
        for (const outputs of new Set(result.built.map((built) => built.outputs))) {
            if (this.disposed) outputs.destroy();
            else this.liveOutputs.add(outputs);
        }
        if (this.disposed) return;
        const refuse = (slot: number, generation: number): void => {
            if (rebuild) this.deforming?.rebuildRefused(slot);
            else this.gpuRefused(slot, generation);
        };
        for (const { slot, generation } of result.refused) refuse(slot, generation);
        if (result.failure) this.warn(`@pmndrs/gi: ${result.failure}, so ${rebuild ? 'the mesh keeps its refit tree' : 'the meshes fall back to tinybvh'}`);
        for (const built of result.built) {
            const state = this.currentState(built.slot, built.generation);
            // A deforming mesh's refit needs the tree's layout; if its readback failed, the tree can't follow poses.
            const layoutLost = built.layout === null && (this.deforming?.has(built.slot) ?? false);
            if (!state || built.tree.maxDepth > this.gpuMaxDepth || layoutLost) {
                // A mesh removed or replaced meanwhile, a tree too deep for the trace stack, or one the refit can't use.
                if (state) refuse(built.slot, built.generation);
                this.outputsToRelease.push(built.outputs);
                continue;
            }
            // A rebuild's mesh already had its first tree: its state stays as the first builds left it.
            if (!rebuild) {
                state.gpu = 'built';
                // The GPU's tree is final: the fallback won't be needed, so let go of the CPU copy.
                if (this.awaitingGpu.get(built.slot)?.generation === built.generation) this.awaitingGpu.delete(built.slot);
            }
            this.gpuTreesBuilt++;
            this.arrived.push({ source: 'gpu', slot: built.slot, generation: built.generation, built, rebuild });
        }
        this.notifyWaiters();
    }

    /** The GPU gave this version of the mesh no tree: tinybvh builds it from the CPU copy kept for this (the fallback). */
    private gpuRefused(slot: number, generation: number): void {
        const state = this.currentState(slot, generation);
        if (!state) return;
        state.gpu = 'refused';
        this.gpuTreesRefused++;
        const kept = this.awaitingGpu.get(slot);
        if (kept?.generation !== generation) return;
        this.awaitingGpu.delete(slot);
        this.startTinybvhBuild(kept.mesh, generation);
    }

    /**
     * Starts tinybvh's build of this upload, or, while the slot's previous
     * build still runs, leaves it waiting behind that one in place of any
     * older upload already waiting (see the file comment).
     */
    private startTinybvhBuild(mesh: UploadedMesh, generation: number): void {
        const state = this.currentState(mesh.slot, generation);
        if (state) state.tinybvh = 'building';
        if (this.tinybvhRunning.has(mesh.slot)) {
            // An older upload waiting here is out of date: its build is never started, so it stops counting.
            if (!this.tinybvhWaiting.has(mesh.slot)) this.pendingBuilds++;
            this.tinybvhWaiting.set(mesh.slot, { mesh, generation });
            return;
        }
        this.pendingBuilds++;
        this.runTinybvhBuild(mesh, generation);
    }

    /** A removed slot's waiting tinybvh build is dropped (its running one ends by itself and is ignored). */
    private dropWaitingTinybvhBuild(slot: number): void {
        if (this.tinybvhWaiting.delete(slot)) this.pendingBuilds--;
    }

    /** Sends one build to the worker; `pendingBuilds` already counts it. */
    private runTinybvhBuild(mesh: UploadedMesh, generation: number): void {
        this.tinybvhRunning.add(mesh.slot);
        // The scene's own copies, handed over once (UploadedMesh.geometry): the worker builder transfers them.
        const geometry = mesh.geometry;
        const finished = (bvh: MeshBvh | null): void => {
            this.pendingBuilds--;
            this.tinybvhRunning.delete(mesh.slot);
            const state = this.disposed ? undefined : this.currentState(mesh.slot, generation);
            if (state) {
                state.tinybvh = bvh ? 'built' : 'failed';
                if (bvh) this.arrived.push({ source: 'tinybvh', slot: mesh.slot, generation, bvh });
            }
            const next = this.tinybvhWaiting.get(mesh.slot);
            if (next && !this.disposed) {
                this.tinybvhWaiting.delete(mesh.slot);
                this.runTinybvhBuild(next.mesh, next.generation);
            }
            this.notifyWaiters();
        };
        this.builder.build(geometry).then(finished, (error: unknown) => {
            if (!this.disposed) this.buildFailure = new Error(`@pmndrs/gi: building a mesh tree failed: ${describe(error)}`);
            finished(null);
        });
    }

    /**
     * The finished builds to place this frame, for meshes that still exist in
     * the version they were built from. A version gets one tree, from the GPU
     * or else from the fallback, so nothing here replaces a tree of the same
     * version, except a deforming mesh's drift rebuild: it is newer than any
     * tree its mesh has (it was built from a later pose), tinybvh's too.
     */
    private takeArrived(): ArrivedTree[] {
        const taken: ArrivedTree[] = [];
        for (const tree of this.arrived) {
            if (this.currentState(tree.slot, tree.generation)) {
                taken.push(tree);
            } else if (tree.source === 'gpu') {
                this.outputsToRelease.push(tree.built.outputs);
                if (tree.rebuild) this.deforming?.rebuildRefused(tree.slot);
            }
        }
        this.arrived = [];
        return taken;
    }

    private releaseTree(slot: number): void {
        const tree = this.placed.get(slot);
        if (!tree) return;
        this.nodes.release(tree.nodeOffset, tree.nodeCount);
        this.triangles.release(tree.triangleOffset, tree.triangleCount);
        this.placed.delete(slot);
        this.gpuTopLevel?.setMeshTree(slot, null);
        this.deforming?.treeReleased(slot);
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
        // A replaced or rebuilt mesh's new tree takes the place of its old one: free the old one
        // first so the space can be reused. Queue order makes reuse safe: frames already submitted
        // read the old tree before any write or copy below runs.
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
                // A deforming mesh's box in the top level is the box every pose stays in, not this tree's
                // (the pose it was built in): then a pose never changes the top level (deforming-trees.ts).
                bounds: this.deforming?.deformBox(tree.slot) ?? size.bounds,
                source: tree.source,
                generation: tree.generation,
            };
            this.placed.set(tree.slot, placed);
            this.gpuTopLevel?.setMeshTree(tree.slot, placed);
            // Builders leave flag bit 0 clear: the material lookup flags come from the scene copy.
            this.requestTriangleFlags(tree.slot);
            // Its refit takes the new shape and refits it this frame, from the positions the scene copy holds now.
            this.deforming?.treePlaced(tree.slot, placed, tree.source === 'tinybvh' ? tree.bvh : tree.built.layout);
        });
    }

    /**
     * Allocates one range per entry of `counts`. If they don't all fit, the
     * storage grows once (copying the trees already placed), and they are
     * allocated again.
     */
    private allocateAll(storage: GrowableStorage, encoder: GPUCommandEncoder, counts: readonly number[], rangeOf: (tree: PlacedTree) => ResidentRange): number[] {
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
    // Triangle flags (T4.2)
    // -----------------------------------------------------------------------

    /**
     * Asks for the flags of the meshes whose instances' materials may have
     * changed in a way the walk sees this frame: a material edited or an
     * instance given other materials (`material` regions, also when a
     * texture's layer lands), and an instance created, shown or given another
     * mesh (`geometry` regions that are not moves; the masks count hidden
     * instances, but one created hidden made no region then). For each such
     * instance, its mesh is rewritten only when the words the rules read
     * changed since the flags last saw it (`walkKey`), so an edit of a colour
     * or an emissive rewrites nothing (T4.2's review). Moves change no
     * material, so they never ask. Destroying or hiding an instance can only
     * leave a flag that is no longer needed, which costs an extra lookup until
     * the mesh's next rewrite. Regions merged in a busy frame name no
     * instance: then every placed mesh is rewritten.
     */
    private requestTriangleFlagsForChanges(changes: SceneChanges): void {
        for (const region of changes.regions) {
            const mayFlipSlots = region.reason === 'material' || (region.reason === 'geometry' && region.moved === undefined);
            if (!mayFlipSlots) continue;
            if (region.source === undefined) {
                for (const slot of this.placed.keys()) this.requestTriangleFlags(slot);
                this.walkKeys.clear();
                return;
            }
            const handle = region.source as number;
            const state = this.scene.instanceMaterialRecords(region.source as InstanceHandle);
            if (!state) {
                // Destroyed: nothing to ask (see above); forget it.
                this.walkKeys.delete(handle);
                continue;
            }
            const key = walkKey(state.meshSlot, state.records);
            if (this.walkKeys.get(handle) === key) continue;
            this.walkKeys.set(handle, key);
            this.requestTriangleFlags(state.meshSlot);
        }
    }

    /** Records `gi/triangle-flags` over the asked-for meshes that have a placed tree. */
    private encodeTriangleFlags(encoder: GPUCommandEncoder): void {
        // Before prepare the kernels don't exist and the requests wait (the platform never encodes before prepare).
        if (this.flagRequests.size === 0 || !this.triangleFlags.ready) return;
        const ranges: MeshTriangleRange[] = [];
        for (const slot of this.flagRequests) {
            const tree = this.placed.get(slot);
            if (tree) ranges.push({ meshSlot: slot, triangleOffset: tree.triangleOffset, triangleCount: tree.triangleCount });
        }
        this.flagRequests.clear();
        if (ranges.length === 0) return;
        this.triangleFlags.encode(encoder, this.triangles.buffer, ranges, this.scene.counts.instances);
        this.flagRewrites += ranges.length;
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
                `@pmndrs/gi: the trace stack holds ${TRACE_STACK_SIZE} entries, but this scene can need ${requiredStack} ` +
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
        if (this.disposed) throw new Error('@pmndrs/gi: the ray queries were disposed');
    }
}

/**
 * What the walk's rules read of an instance, as a string to compare: its mesh
 * slot, and per material slot the record words `giMaterialNeedsLookup` and
 * `giMaterialPasses` decide on (the blocks-light and double-sided flags, the
 * cutoff, the opacity, the alpha mode, and whether there is an albedo
 * texture). Coarser than "the mask changed" (a new opacity of a stochastic
 * material asks again), never finer: whatever can flip a slot changes it.
 */
function walkKey(meshSlot: number, records: readonly Uint32Array[]): string {
    const parts: number[] = [meshSlot];
    for (const record of records) {
        parts.push(
            record[MATERIAL_WORD.flags] & (MATERIAL_FLAG.blocksLight | MATERIAL_FLAG.doubleSided),
            record[MATERIAL_WORD.alphaCutoff],
            record[MATERIAL_WORD.opacity],
            record[MATERIAL_WORD.alphaMode],
            record[MATERIAL_WORD.albedoTexture] === NONE ? 0 : 1,
        );
    }
    return parts.join(',');
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function sameBox(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
    for (let index = 0; index < 6; index++) if (a[index] !== b[index]) return false;
    return true;
}

/** A tree's counts, depth and box, whichever builder made it. */
function treeSize(tree: ArrivedTree): { readonly nodeCount: number; readonly triangleCount: number; readonly maxDepth: number; readonly bounds: Float32Array } {
    return tree.source === 'tinybvh' ? tree.bvh : tree.built.tree;
}
