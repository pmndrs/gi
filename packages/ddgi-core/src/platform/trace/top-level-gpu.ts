/**
 * The top-level tree on the GPU (stage B1a, DECISIONS 2026-11-04): refit
 * every frame from the instances' transforms, rebuilt now and then.
 *
 * **Every frame something moved**, one compute pass, no CPU work per instance:
 *
 * 1. *Gather* (this file's kernel), one thread per instance record: reads the
 *    instance's transform, inverse and flags from the scene copy's table, and
 *    its mesh tree's offsets and object box from the tracer's mesh table;
 *    writes the record the walk reads (world-to-object rows, offsets, flags)
 *    and the instance's world box (`bvhInstanceWorldBox`, `@pmndrs/bvh`).
 *    An instance that is now hidden, removed or has a singular transform, or
 *    whose mesh tree is gone, gets an empty box: rays skip it at once.
 * 2. *Refit* (`GpuBvhRefit`, `@pmndrs/bvh`): recomputes every node box
 *    bottom-up from those world boxes, keeping the tree's shape. If any read
 *    in its climb gave up waiting, its repair dispatch refits the whole tree
 *    again in the same pass, so a wrong box is never kept.
 *
 * **Now and then, a new shape**, built on the CPU in a worker
 * (top-level-async.ts) and swapped in at the start of a later frame;
 * refits continue on the old shape meanwhile. A new shape is built when:
 *
 * - an instance that contributes to GI was created or destroyed (the scene
 *   copy's `instanceSetChanged`). A new instance becomes visible to rays when
 *   the shape arrives, usually a frame or two later, as a new mesh does when
 *   its tree arrives;
 * - the refit tree's quality has decayed. While things move, every
 *   {@link QUALITY_CHECK_INTERVAL_MS} the worker refits the current shape on
 *   the CPU from the current transforms and builds a new one if its relative
 *   SAH cost is more than {@link TOP_LEVEL_REBUILD_COST_RATIO} times its cost
 *   as built. Nothing is read back from the GPU for this: the platform
 *   promises hosts no readbacks frame after frame.
 *
 * Shapes include hidden instances too (their boxes are empty while hidden),
 * so hiding and showing take effect in the same frame. Removed instances
 * cost nothing until the next shape: their empty boxes are skipped, and a slot
 * reused by a new instance is traced correctly because the gather reads
 * whatever instance the slot holds now. Mesh trees that arrive or are rebuilt
 * only change the mesh table, so they need no new shape either.
 *
 * **Diagnostics** (`diagnostics: true`, which the platform turns on with
 * `gpuTiming`): the refit's stats are read back a frame or two late. A climb
 * that gave up (and was repaired) is then counted and reported once through
 * `warn`. Without diagnostics there are no readbacks at all.
 */

import { BVH_INSTANCE_BOX_WGSL } from '@pmndrs/bvh/gpu-instance-box';
import { GpuBvhRefit, type BvhRefitStats } from '@pmndrs/bvh/gpu-refit';
import { BVH_FLOAT_MAX, PRIMITIVE_BOX_FLOATS } from '@pmndrs/bvh/refit-cpu';
import { GILimitError } from '../contract/errors.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { ComputeKernel, KernelBuilder } from '../contract/wgsl.ts';
import { SHADER_STAGE_COMPUTE } from '../kernel-builder.ts';
import type { PlatformScene } from '../scene/platform-scene.ts';
import {
    BVH_NODE_BYTES,
    INSTANCE_FLAG,
    INSTANCE_RECORD_BYTES,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_BYTES,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
} from './layout.ts';
import { createTopLevelBuilder, type TopLevelBuilder, type TopLevelResult } from './top-level-async.ts';
import type { TopLevelBuildInput, TopLevelShape } from './top-level-shape.ts';

/**
 * Rebuild when the refit tree's relative SAH cost exceeds the as-built cost
 * by this factor (a planning number; DECISIONS 2026-11-05, B1a).
 *
 * Why 1.25: the cost is the expected number of top-level nodes a ray visits,
 * and the top level is a small part of a ray's work (a shadow ray in Bistro
 * interior visits about 44 nodes in all, T1.3), so 25% more top-level visits
 * cost a few percent of trace time at most. A check or rebuild costs no frame
 * time on the GPU and runs in the worker; on the main thread it costs packing
 * the transforms (`counters.lastSnapshotMs`): about 1.3 ms for 10,000
 * instances, 2.6-4.4 ms the first time while the code is still cold (RTX
 * 3060 and Intel UHD laptop, bench/results/bvh.csv). Only one runs at a
 * time. A lower ratio would rebuild continuously in a scene that never stops
 * moving.
 */
export const TOP_LEVEL_REBUILD_COST_RATIO = 1.25;

/**
 * While instances move, the worker checks the shape's quality at most this
 * often. Each check packs every instance's transform on the main thread
 * (about 1.3 ms for 10,000), so it isn't done every frame: at most about
 * 0.5% of the main thread while everything moves.
 */
export const QUALITY_CHECK_INTERVAL_MS = 250;

/** Gather kernel workgroup size. */
const GATHER_WORKGROUP_SIZE = 64;

/** Mesh table entry: two vec4u per mesh slot, `(boundsMin.xyz, rootNodeOffset)` and `(boundsMax.xyz, triangleOffset)`. */
const MESH_ENTRY_WORDS = 8;
/** Word positions inside a mesh table entry (the bounds as f32 bits). */
const MESH_ENTRY_WORD = {
    boundsMin: 0,
    rootNodeOffset: 3,
    boundsMax: 4,
    triangleOffset: 7,
} as const;
/** `rootNodeOffset` of a mesh slot with no tree: its instances get empty boxes. */
const MESH_ABSENT = 0xffffffff;

/** A mesh tree placed in the shared buffers, as the mesh table needs it. */
export interface PlacedMeshTree {
    readonly nodeOffset: number;
    readonly triangleOffset: number;
    /** Object-space box `[min, max]` of the tree's root. */
    readonly bounds: ArrayLike<number>;
}

/** What the tracer tells the top level about this frame. */
export interface TopLevelFrame {
    /** The scene copy reported an instance, transform or mesh change. */
    readonly instancesChanged: boolean;
    /** An instance that contributes to GI was created or destroyed (the scene copy's flag). */
    readonly instanceSetChanged: boolean;
    /** Mesh trees were placed or released this frame (the mesh table changed). */
    readonly meshesChanged: boolean;
    /** Deepest placed mesh tree, in levels, for the traversal-stack check. */
    readonly deepestMesh: number;
}

export interface GpuTopLevelOptions {
    readonly label?: string;
    /** `worker` (default) builds shapes in a module worker; `main` on this thread (on a later turn). */
    readonly buildIn?: 'worker' | 'main';
    /** Read the refit's stats back for counters and warnings (see the top of this file). Off by default. */
    readonly diagnostics?: boolean;
    /** Where a refit that had to be repaired is reported (once). Defaults to `console.warn`. */
    readonly warn?: (message: string) => void;
}

/** Counters for tests, stats and the benchmark. */
export interface GpuTopLevelCounters {
    /** Shapes built (each one swapped in). */
    shapesBuilt: number;
    /** Quality checks run in the worker. */
    qualityChecks: number;
    /** Of the shapes, those a quality check built. */
    qualityRebuilds: number;
    /** Worker CPU time of the last shape build. */
    lastBuildMs: number;
    /** Main-thread time to pack the last build's or check's input. */
    lastSnapshotMs: number;
    /** Refit passes recorded. */
    refits: number;
    /** Diagnostics only: climb reads that gave up waiting (each repaired in the same pass). */
    refitGaveUp: number;
    /** Diagnostics only: refits the repair pass redid. */
    refitRepairs: number;
}

interface ArrivedShape {
    readonly shape: TopLevelShape;
    readonly generation: number;
}

interface CurrentShape {
    readonly recordCount: number;
    readonly nodeCount: number;
    readonly maxDepth: number;
    readonly generation: number;
}

const USAGE = { COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const;

const hex = (value: number): string => `0x${value.toString(16)}u`;

/**
 * The gather kernel's own WGSL (group 1). The scene module (group 0) comes
 * from the kernel builder and provides the `giInstance*` accessors.
 */
export function topLevelGatherWgsl(): string {
    return /* wgsl */ `${BVH_INSTANCE_BOX_WGSL}
struct GiGatherParams {
    recordCount: u32,
    meshCapacity: u32,
    unused0: u32,
    unused1: u32,
}

// The tracer's top level: [header][instance records][nodes], as vec4u so no
// integer passes through a float register.
@group(1) @binding(0) var<storage, read_write> giGatherTopLevel: array<vec4u>;
// Two vec4u per mesh slot: (boundsMin bits, rootNodeOffset), (boundsMax bits, triangleOffset).
@group(1) @binding(1) var<storage, read> giGatherMeshes: array<vec4u>;
// The scene instance slot each record holds.
@group(1) @binding(2) var<storage, read> giGatherRecordSlots: array<u32>;
// The refit's input: two vec4f per record, (min.xyz, 0), (max.xyz, 0).
@group(1) @binding(3) var<storage, read_write> giGatherBoxes: array<vec4f>;
@group(1) @binding(4) var<uniform> giGatherParams: GiGatherParams;

const GI_GATHER_WORKGROUP_SIZE: u32 = ${GATHER_WORKGROUP_SIZE}u;
const GI_GATHER_HEADER_VEC4S: u32 = ${TOP_LEVEL_HEADER_BYTES / 16}u;
const GI_GATHER_RECORD_VEC4S: u32 = ${INSTANCE_RECORD_BYTES / 16}u;
const GI_GATHER_BOX_VEC4S: u32 = ${PRIMITIVE_BOX_FLOATS / 4}u;
const GI_GATHER_MESH_ABSENT: u32 = ${hex(MESH_ABSENT)};
const GI_GATHER_FLOAT_MAX: f32 = 0x1.fffffep+127f;   // ${BVH_FLOAT_MAX}
const GI_GATHER_FLAG_TRACED: u32 = ${INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible}u;
const GI_GATHER_FLAG_MIRRORED: u32 = ${INSTANCE_FLAG.mirrored}u;

@compute @workgroup_size(${GATHER_WORKGROUP_SIZE})
fn giTopLevelGather(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    // 2D dispatch, so very many instances stay under the per-dimension workgroup limit.
    let record = id.y * groups.x * GI_GATHER_WORKGROUP_SIZE + id.x;
    if (record >= giGatherParams.recordCount) {
        return;
    }
    let slot = giGatherRecordSlots[record];
    let recordBase = GI_GATHER_HEADER_VEC4S + record * GI_GATHER_RECORD_VEC4S;
    let boxBase = record * GI_GATHER_BOX_VEC4S;

    // Rays see the instance only if the scene says so now and its mesh tree is placed.
    var traced = giInstanceContributesToGI(slot);
    let meshSlot = giInstanceMesh(slot);
    var meshLower = vec4u(0u);
    var meshUpper = vec4u(0u);
    if (traced && meshSlot < giGatherParams.meshCapacity) {
        meshLower = giGatherMeshes[meshSlot * 2u];
        meshUpper = giGatherMeshes[meshSlot * 2u + 1u];
        traced = meshLower.w != GI_GATHER_MESH_ABSENT;
    } else {
        traced = false;
    }
    if (!traced) {
        // An empty box: the refit stores it where no ray reaches, so the record is never entered.
        giGatherTopLevel[recordBase] = vec4u(0u);
        giGatherTopLevel[recordBase + 1u] = vec4u(0u);
        giGatherTopLevel[recordBase + 2u] = vec4u(0u);
        giGatherTopLevel[recordBase + 3u] = vec4u(0u, 0u, slot, 0u);
        giGatherBoxes[boxBase] = vec4f(vec3f(GI_GATHER_FLOAT_MAX), 0.0);
        giGatherBoxes[boxBase + 1u] = vec4f(vec3f(-GI_GATHER_FLOAT_MAX), 0.0);
        return;
    }

    // The scene stores object-to-world as three rows; transposing the matrix it
    // builds from them gives the rows back as columns.
    let rows = transpose(giInstanceTransform(slot));
    let translation = vec3f(rows[0].w, rows[1].w, rows[2].w);
    // The scene also stores the rows of inverse(L), the linear part's inverse,
    // computed in doubles on the CPU; they are the normal matrix's columns.
    let inverseRows = giInstanceNormalMatrix(slot);
    // p_object = inverse(L) * (p_world - t), so the inverse's translation is -inverse(L) * t.
    let worldToObject0 = vec4f(inverseRows[0], -dot(inverseRows[0], translation));
    let worldToObject1 = vec4f(inverseRows[1], -dot(inverseRows[1], translation));
    let worldToObject2 = vec4f(inverseRows[2], -dot(inverseRows[2], translation));
    let mirrored = (giSceneInstanceFlags(slot) & GI_SCENE_INSTANCE_MIRRORED) != 0u;
    let flags = GI_GATHER_FLAG_TRACED | select(0u, GI_GATHER_FLAG_MIRRORED, mirrored);
    giGatherTopLevel[recordBase] = bitcast<vec4u>(worldToObject0);
    giGatherTopLevel[recordBase + 1u] = bitcast<vec4u>(worldToObject1);
    giGatherTopLevel[recordBase + 2u] = bitcast<vec4u>(worldToObject2);
    giGatherTopLevel[recordBase + 3u] = vec4u(meshLower.w, meshUpper.w, slot, flags);

    let box = bvhInstanceWorldBox(rows[0], rows[1], rows[2], bitcast<vec3f>(meshLower.xyz), bitcast<vec3f>(meshUpper.xyz));
    giGatherBoxes[boxBase] = vec4f(box.lower, 0.0);
    giGatherBoxes[boxBase + 1u] = vec4f(box.upper, 0.0);
}
`;
}

const GATHER_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'gi top level gather',
    entries: [
        { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    ],
};

export class GpuTopLevel {
    readonly counters: GpuTopLevelCounters = {
        shapesBuilt: 0,
        qualityChecks: 0,
        qualityRebuilds: 0,
        lastBuildMs: 0,
        lastSnapshotMs: 0,
        refits: 0,
        refitGaveUp: 0,
        refitRepairs: 0,
    };

    private readonly device: GPUDevice;
    private readonly scene: PlatformScene;
    private readonly label: string;
    private readonly buildIn: 'worker' | 'main';
    private readonly warn: (message: string) => void;
    private builder: TopLevelBuilder | null = null;
    private readonly refit: GpuBvhRefit;
    private gatherKernel: ComputeKernel | null = null;

    private topLevelBuffer: GPUBuffer;
    private retired: GPUBuffer[] = [];
    private recordSlots: GPUBuffer;
    private primitiveBoxes: GPUBuffer;
    private meshTable: GPUBuffer;
    private meshMirror: Uint32Array<ArrayBuffer>;
    private meshTableDirty = true;
    private readonly gatherParams: GPUBuffer;
    private gatherGroup: { group: GPUBindGroup; buffers: GPUBuffer[] } | null = null;
    /**
     * Object-space mesh boxes by mesh slot, six floats each: the packed bounds
     * from the scene upload, then the tree's root box once placed. The input
     * of shape builds and checks.
     */
    private meshBounds = new Float32Array(16 * 6);

    private current: CurrentShape = { recordCount: 0, nodeCount: 0, maxDepth: 0, generation: 0 };
    private arrived: ArrivedShape | null = null;
    /** The request the worker is busy with, if any. */
    private inFlight: 'build' | 'check' | null = null;
    /** A build was asked for while the worker was busy: start it when the worker is free. */
    private buildQueued = false;
    private lastRequestTime = -Infinity;
    private nextGeneration = 1;
    private buildFailure: Error | null = null;
    private idleWaiters: (() => void)[] = [];
    private seenStats: BvhRefitStats | null = null;
    /** A give-up was reported already: the warning is shown once per top level. */
    private warnedGaveUp = false;
    /** Set when a shape couldn't be swapped in; cleared when instances or meshes change. */
    private qualityRebuildsPaused = false;
    private bufferVersion = 0;
    private disposed = false;

    /** Relative SAH cost of the refit tree: from the last quality check, or the diagnostics readback when on. */
    latestRelativeCost = 0;

    constructor(device: GPUDevice, scene: PlatformScene, options: GpuTopLevelOptions = {}) {
        this.device = device;
        this.scene = scene;
        this.label = options.label ?? 'gi';
        this.buildIn = options.buildIn ?? 'worker';
        this.warn = options.warn ?? ((message) => console.warn(message));
        this.refit = new GpuBvhRefit(device, { label: `${this.label} top level refit`, diagnostics: options.diagnostics ?? false });
        // An empty top level: the header says 0 nodes, so every ray misses.
        this.topLevelBuffer = this.createTopLevelBuffer(TOP_LEVEL_HEADER_BYTES);
        this.recordSlots = this.createStorage('top level record slots', 64);
        this.primitiveBoxes = this.createStorage('top level instance boxes', 64);
        this.meshMirror = new Uint32Array(16 * MESH_ENTRY_WORDS).fill(MESH_ABSENT);
        this.meshTable = this.createStorage('top level mesh table', this.meshMirror.byteLength);
        this.gatherParams = device.createBuffer({ label: `${this.label} top level gather params`, size: 16, usage: USAGE.UNIFORM | USAGE.COPY_DST });
    }

    /** The tracer's top-level buffer: header, instance records, nodes. */
    get buffer(): GPUBuffer {
        return this.topLevelBuffer;
    }

    /** Increments when {@link buffer} is replaced. */
    get version(): number {
        return this.bufferVersion;
    }

    /** Records in the current shape. */
    get instanceCount(): number {
        return this.current.recordCount;
    }

    /** True while a shape that adds or removes instances is being built or waits to be swapped in. Quality checks don't count. */
    get building(): boolean {
        return this.buildPending || this.arrived !== null;
    }

    get memoryBytes(): number {
        return this.topLevelBuffer.size + this.recordSlots.size + this.primitiveBoxes.size + this.meshTable.size + this.gatherParams.size + this.refit.memoryBytes;
    }

    /** Resolves once no shape build is pending (the last may still wait for the next `update` to be swapped in). */
    whenIdle(): Promise<void> {
        if (!this.buildPending) return Promise.resolve();
        return new Promise((resolve) => this.idleWaiters.push(resolve));
    }

    async prepare(kernels: KernelBuilder): Promise<void> {
        // The shape worker starts here, so its module loads while the kernels compile instead of
        // when the first frame asks for a shape: the first shape is on the way to first GI (T2.12).
        this.builder ??= createTopLevelBuilder({ worker: this.buildIn === 'main' ? 'never' : 'auto' });
        if (this.gatherKernel && this.refit.ready) return;
        const [gather] = await Promise.all([
            kernels.computeKernel({
                label: 'top level gather',
                modules: [this.scene.module],
                code: topLevelGatherWgsl(),
                entryPoint: 'giTopLevelGather',
                ownGroups: [GATHER_GROUP],
            }),
            this.refit.prepare(),
        ]);
        this.gatherKernel = gather;
    }

    /** A mesh was uploaded: its object box is what builds use until its tree is placed. */
    meshUploaded(slot: number, bounds: ArrayLike<number>): void {
        this.setMeshBounds(slot, bounds);
    }

    /** A mesh's tree was placed (or replaced), or released (`null`). The mesh table reaches the GPU at the next `update`. */
    setMeshTree(slot: number, tree: PlacedMeshTree | null): void {
        if (tree) this.setMeshBounds(slot, tree.bounds);
        if ((slot + 1) * MESH_ENTRY_WORDS > this.meshMirror.length) {
            let capacity = this.meshMirror.length / MESH_ENTRY_WORDS;
            while (capacity <= slot) capacity *= 2;
            const grown = new Uint32Array(capacity * MESH_ENTRY_WORDS).fill(MESH_ABSENT);
            grown.set(this.meshMirror);
            this.meshMirror = grown;
            // Destroyed at the next update, after the host submitted the frames that used it.
            this.retired.push(this.meshTable);
            this.meshTable = this.createStorage('top level mesh table', grown.byteLength);
        }
        const base = slot * MESH_ENTRY_WORDS;
        if (tree === null) {
            this.meshMirror.fill(MESH_ABSENT, base, base + MESH_ENTRY_WORDS);
        } else {
            const floats = new Float32Array(this.meshMirror.buffer, base * 4, MESH_ENTRY_WORDS);
            floats.set([tree.bounds[0], tree.bounds[1], tree.bounds[2]], MESH_ENTRY_WORD.boundsMin);
            floats.set([tree.bounds[3], tree.bounds[4], tree.bounds[5]], MESH_ENTRY_WORD.boundsMax);
            this.meshMirror[base + MESH_ENTRY_WORD.rootNodeOffset] = tree.nodeOffset;
            this.meshMirror[base + MESH_ENTRY_WORD.triangleOffset] = tree.triangleOffset;
        }
        this.meshTableDirty = true;
    }

    /**
     * Brings the top level up to this frame: swaps in an arrived shape,
     * starts a build or quality check if one is due, and records the refit
     * pass (gather, then refit) when anything it depends on changed. Called
     * by the tracer in the platform's `encode`, before any technique traces.
     */
    update(encoder: GPUCommandEncoder, frame: TopLevelFrame, timer?: GpuTimerScope): void {
        if (this.disposed) throw new Error('@pmndrs/ddgi: the top level was disposed');
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];

        this.refit.pollStats();
        this.collectDiagnostics();
        const moved = frame.instancesChanged || frame.meshesChanged;
        // A shape that couldn't be swapped in pauses quality rebuilds (they would make the same
        // shape again) until the instances or meshes change.
        if (frame.instanceSetChanged || frame.meshesChanged) this.qualityRebuildsPaused = false;
        const checkDue = performance.now() - this.lastRequestTime >= QUALITY_CHECK_INTERVAL_MS;
        if (frame.instanceSetChanged) this.request('build');
        else if (moved && checkDue && this.current.recordCount > 0 && !this.qualityRebuildsPaused) this.request('check');

        // The mesh table goes first, so even a frame that throws below leaves no record
        // pointing at a released mesh tree.
        let changed = moved;
        if (this.meshTableDirty) {
            this.device.queue.writeBuffer(this.meshTable, 0, this.meshMirror);
            this.writeGatherParams();
            this.meshTableDirty = false;
            changed = true;
        }
        // A failure is thrown at the end, once this frame's refit is recorded on the shape in use.
        let failure: unknown = this.buildFailure;
        this.buildFailure = null;
        if (this.arrived) {
            // Tried once. A shape that can't be used (a limit, or too deep for the trace stack)
            // is dropped: swap checks before writing anything, so the old shape stays whole and
            // in use, and the worker keeps judging that one (checks name its generation).
            const arrived = this.arrived;
            this.arrived = null;
            try {
                this.swap(arrived, frame.deepestMesh);
                changed = true;
            } catch (error) {
                failure ??= error;
                this.qualityRebuildsPaused = true;
            }
        }
        this.checkStack(this.current.maxDepth, frame.deepestMesh);
        if (changed && this.current.recordCount > 0) this.encodeRefit(encoder, timer);
        if (failure) throw failure;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.builder?.dispose();
        this.refit.dispose();
        for (const buffer of [...this.retired, this.topLevelBuffer, this.recordSlots, this.primitiveBoxes, this.meshTable, this.gatherParams]) buffer.destroy();
        this.retired = [];
        for (const resolve of this.idleWaiters.splice(0)) resolve();
    }

    // -----------------------------------------------------------------------
    // Shapes
    // -----------------------------------------------------------------------

    /** A build is running or queued (quality checks aside). */
    private get buildPending(): boolean {
        return this.inFlight === 'build' || this.buildQueued;
    }

    private setMeshBounds(slot: number, bounds: ArrayLike<number>): void {
        if ((slot + 1) * 6 > this.meshBounds.length) {
            const grown = new Float32Array(Math.max((slot + 1) * 6, this.meshBounds.length * 2));
            grown.set(this.meshBounds);
            this.meshBounds = grown;
        }
        for (let index = 0; index < 6; index++) this.meshBounds[slot * 6 + index] = bounds[index];
    }

    private request(mode: 'build' | 'check'): void {
        if (this.inFlight !== null) {
            if (mode === 'build') this.buildQueued = true;
            return;
        }
        this.send(mode);
    }

    /** Packs every instance that contributes to GI and sends it to the worker. */
    private send(mode: 'build' | 'check'): void {
        const start = performance.now();
        const packed = this.scene.packGiInstances();
        const count = packed.slots.length;
        const meshBounds = new Float32Array(count * 6);
        for (let index = 0; index < count; index++) {
            const meshSlot = packed.meshSlots[index];
            // A mesh always reaches the tracer in the same upload as (or before) its first instance.
            if ((meshSlot + 1) * 6 <= this.meshBounds.length) meshBounds.set(this.meshBounds.subarray(meshSlot * 6, meshSlot * 6 + 6), index * 6);
        }
        const input: TopLevelBuildInput = { slots: packed.slots, visible: packed.visible, objectToWorld: packed.objectToWorld, meshBounds };
        this.counters.lastSnapshotMs = performance.now() - start;
        this.lastRequestTime = performance.now();
        if (mode === 'check') this.counters.qualityChecks++;

        const generation = this.nextGeneration++;
        this.inFlight = mode;
        this.builder ??= createTopLevelBuilder({ worker: this.buildIn === 'main' ? 'never' : 'auto' });
        this.builder.request(mode, input, TOP_LEVEL_REBUILD_COST_RATIO, { current: this.current.generation, next: generation }).then(
            (result) => this.finish(mode, generation, result),
            (error: unknown) => {
                if (this.disposed) return;
                this.buildFailure = new Error(`@pmndrs/ddgi: building the top level failed: ${error instanceof Error ? error.message : String(error)}`);
                this.finish(mode, generation, null);
            },
        );
    }

    private finish(mode: 'build' | 'check', generation: number, result: TopLevelResult | null): void {
        if (this.disposed) return;
        this.inFlight = null;
        if (result) {
            if (mode === 'check') this.latestRelativeCost = result.relativeCost;
            if (result.shape) {
                this.arrived = { shape: result.shape, generation };
                this.counters.lastBuildMs = result.workerMs;
                if (mode === 'check') this.counters.qualityRebuilds++;
            }
        }
        if (this.buildQueued) {
            // Instances came or went while the worker was busy: build again right away, so
            // `whenIdle` only resolves once the shape covers the scene as it is now.
            this.buildQueued = false;
            this.send('build');
            return;
        }
        for (const resolve of this.idleWaiters.splice(0)) resolve();
    }

    private checkStack(topLevelDepth: number, deepestMesh: number): void {
        const requiredStack = topLevelDepth + 1 + deepestMesh;
        if (topLevelDepth > 0 && requiredStack > TRACE_STACK_SIZE) {
            throw new Error(
                `@pmndrs/ddgi: the trace stack holds ${TRACE_STACK_SIZE} entries, but this scene can need ${requiredStack} ` +
                    `(top level ${topLevelDepth} + 1 + deepest mesh tree ${deepestMesh}).`,
            );
        }
    }

    /**
     * Puts an arrived shape in place. The stack check and the size check come
     * first, so a shape that can't be used leaves every buffer as it was.
     * Every write lands before this frame's commands run.
     */
    private swap(arrived: ArrivedShape, deepestMesh: number): void {
        const { shape, generation } = arrived;
        this.checkStack(shape.maxDepth, deepestMesh);
        const recordCount = shape.recordSlots.length;
        const nodeCount = shape.nodeCount;
        const nodeByteOffset = TOP_LEVEL_HEADER_BYTES + recordCount * INSTANCE_RECORD_BYTES;
        this.ensureTopLevelBytes(nodeByteOffset + nodeCount * BVH_NODE_BYTES);

        const header = new Uint32Array(TOP_LEVEL_HEADER_WORDS);
        header[TOP_HEADER_WORD.instanceCount] = recordCount;
        header[TOP_HEADER_WORD.nodeCount] = nodeCount;
        header[TOP_HEADER_WORD.maxDepth] = shape.maxDepth;
        this.device.queue.writeBuffer(this.topLevelBuffer, 0, header);
        if (nodeCount > 0) this.device.queue.writeBuffer(this.topLevelBuffer, nodeByteOffset, shape.nodes, 0, nodeCount * BVH_NODE_BYTES);

        // Records are in tree order; record r holds the instance in scene slot recordSlots[r].
        const recordSlots = new Uint32Array(Math.max(1, recordCount));
        recordSlots.set(shape.recordSlots);
        if (this.recordSlots.size < recordSlots.byteLength) {
            this.retired.push(this.recordSlots);
            this.recordSlots = this.createStorage('top level record slots', recordSlots.byteLength * 2);
        }
        this.device.queue.writeBuffer(this.recordSlots, 0, recordSlots);
        const boxBytes = Math.max(1, recordCount) * PRIMITIVE_BOX_FLOATS * 4;
        if (this.primitiveBoxes.size < boxBytes) {
            this.retired.push(this.primitiveBoxes);
            this.primitiveBoxes = this.createStorage('top level instance boxes', boxBytes * 2);
        }

        this.refit.setShape({
            nodeWordOffset: nodeByteOffset / 4,
            nodeCount,
            parentSlots: shape.parentSlots,
            leafSlots: shape.leafSlots,
            generation,
        });
        this.current = { recordCount, nodeCount, maxDepth: shape.maxDepth, generation };
        this.writeGatherParams();
        this.counters.shapesBuilt++;
    }

    /** Diagnostics only: counts and reports (once) refits that gave up and were repaired. */
    private collectDiagnostics(): void {
        const stats = this.refit.latestStats;
        if (!stats || stats === this.seenStats) return;
        this.seenStats = stats;
        if (stats.generation === this.current.generation) this.latestRelativeCost = stats.relativeCost;
        if (stats.incomplete === 0) return;
        this.counters.refitGaveUp += stats.incomplete;
        this.counters.refitRepairs += stats.repairs;
        if (this.warnedGaveUp) return;
        this.warnedGaveUp = true;
        this.warn(
            '@pmndrs/ddgi: the top-level refit gave up waiting for a box on this GPU and refit the tree again in a slower pass ' +
                '(the result is correct). Counted in stats.counters["platform top level"].',
        );
    }

    // -----------------------------------------------------------------------
    // The refit pass
    // -----------------------------------------------------------------------

    private encodeRefit(encoder: GPUCommandEncoder, timer?: GpuTimerScope): void {
        const gather = this.gatherKernel;
        if (!gather || !this.refit.ready) throw new Error('@pmndrs/ddgi: the top level was not prepared');
        const pass = encoder.beginComputePass({ label: `${this.label} top level refit`, timestampWrites: timer?.computePass('platform/top-level refit') });
        gather.bind(pass);
        pass.setBindGroup(1, this.gatherBindGroup(gather));
        const workgroups = Math.ceil(this.current.recordCount / GATHER_WORKGROUP_SIZE);
        const x = Math.min(workgroups, this.device.limits.maxComputeWorkgroupsPerDimension);
        pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
        this.refit.encode(pass, this.topLevelBuffer, this.primitiveBoxes);
        pass.end();
        this.refit.copyStats(encoder);
        this.counters.refits++;
    }

    private gatherBindGroup(gather: ComputeKernel): GPUBindGroup {
        const buffers = [this.topLevelBuffer, this.meshTable, this.recordSlots, this.primitiveBoxes];
        const cached = this.gatherGroup;
        if (cached && cached.buffers.every((buffer, index) => buffer === buffers[index])) return cached.group;
        const group = this.device.createBindGroup({
            label: `${this.label} top level gather`,
            layout: gather.ownLayouts[0],
            entries: [...buffers.map((buffer, binding) => ({ binding, resource: { buffer } })), { binding: 4, resource: { buffer: this.gatherParams } }],
        });
        this.gatherGroup = { group, buffers };
        return group;
    }

    private writeGatherParams(): void {
        this.device.queue.writeBuffer(this.gatherParams, 0, new Uint32Array([this.current.recordCount, this.meshMirror.length / MESH_ENTRY_WORDS, 0, 0]));
    }

    // -----------------------------------------------------------------------
    // Buffers
    // -----------------------------------------------------------------------

    private ensureTopLevelBytes(bytes: number): void {
        if (bytes <= this.topLevelBuffer.size) return;
        // One binding and one buffer must both hold it; name whichever limit is the smaller.
        const limits = this.device.limits;
        const limitBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
        if (bytes > limitBytes) {
            const limit = limits.maxStorageBufferBindingSize <= limits.maxBufferSize ? 'maxStorageBufferBindingSize' : 'maxBufferSize';
            throw new GILimitError(limit, bytes, limitBytes);
        }
        this.retired.push(this.topLevelBuffer);
        // Room to grow, rounded down to whole vec4s under the cap (the size is rounded up to them).
        const cap = Math.floor(limitBytes / 16) * 16;
        this.topLevelBuffer = this.createTopLevelBuffer(Math.min(cap, Math.max(bytes, this.topLevelBuffer.size * 2)));
        this.bufferVersion++;
    }

    private createTopLevelBuffer(size: number): GPUBuffer {
        return this.device.createBuffer({
            label: `${this.label} trace top level`,
            size: Math.ceil(size / 16) * 16,
            // COPY_SRC lets tests and debug tools read the tree back.
            usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC,
        });
    }

    private createStorage(name: string, size: number): GPUBuffer {
        return this.device.createBuffer({ label: `${this.label} ${name}`, size: Math.ceil(size / 16) * 16, usage: USAGE.STORAGE | USAGE.COPY_DST });
    }
}
