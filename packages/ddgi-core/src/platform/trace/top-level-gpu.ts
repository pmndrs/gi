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
 *    bottom-up from those world boxes, keeping the tree's shape.
 *
 * **Now and then, a new shape**, built on the CPU in a worker
 * (top-level-async.ts) and swapped in at the start of a later frame;
 * refits continue on the old shape meanwhile. A new shape is built when:
 *
 * - an instance that contributes to GI isn't in the tree yet (created since,
 *   or made to contribute). It becomes visible to rays when the shape
 *   arrives, usually a frame or two later, as a new mesh does when its tree
 *   arrives;
 * - the refit tree's quality has decayed: its relative SAH cost
 *   (`relativeSahCost`, read back from the refit a frame or two late) is more
 *   than {@link TOP_LEVEL_REBUILD_COST_RATIO} times the shape's cost as built.
 *
 * Shapes include hidden instances too (their boxes are empty while hidden),
 * so hiding and showing take effect in the same frame. Removed instances
 * don't need a new shape either: their empty boxes cost nothing to skip, and
 * a slot reused by a new instance is traced correctly because the gather
 * reads whatever instance the slot holds now.
 *
 * Mesh trees that arrive or are rebuilt only change the mesh table, so they
 * need no new shape either.
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
import { createTopLevelBuilder, type TopLevelBuild, type TopLevelBuilder } from './top-level-async.ts';

/**
 * Rebuild when the refit tree's relative SAH cost exceeds the as-built cost
 * by this factor (a planning number; DECISIONS 2026-11-05, B1a).
 *
 * Why 1.25: the cost is the expected number of top-level nodes a ray visits,
 * and the top level is a small part of a ray's work (a shadow ray in Bistro
 * interior visits about 44 nodes in all, T1.3), so 25% more top-level visits
 * cost a few percent of trace time at most. A rebuild costs no frame time
 * (worker), only a main-thread copy of the transforms (well under a
 * millisecond for 10,000 instances), and only one runs at a time. A lower
 * ratio would rebuild continuously in a scene that never stops moving.
 */
export const TOP_LEVEL_REBUILD_COST_RATIO = 1.25;

/** Gather kernel workgroup size. */
const GATHER_WORKGROUP_SIZE = 64;

/** Mesh table entry: `(boundsMin.xyz, rootNodeOffset)`, `(boundsMax.xyz, triangleOffset)` as u32 words. */
const MESH_ENTRY_WORDS = 8;
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
    /** Mesh trees were placed or released this frame (the mesh table changed). */
    readonly meshesChanged: boolean;
    /** Deepest placed mesh tree, in levels, for the traversal-stack check. */
    readonly deepestMesh: number;
}

export interface GpuTopLevelOptions {
    readonly label?: string;
    /** `worker` (default) builds shapes in a module worker; `main` on this thread (on a later turn). */
    readonly buildIn?: 'worker' | 'main';
}

/** Counters for tests, stats and the benchmark. */
export interface GpuTopLevelCounters {
    /** Shapes built (each one swapped in). */
    shapesBuilt: number;
    /** Of those, started because the refit tree's quality decayed. */
    qualityRebuilds: number;
    /** CPU time of the last shape build, in the worker. */
    lastBuildMs: number;
    /** Main-thread time to gather the last build's input. */
    lastSnapshotMs: number;
    /** Refits recorded. */
    refits: number;
    /** Box reads the refit gave up on (see `@pmndrs/bvh` gpu-refit.ts). Must stay 0. */
    refitIncomplete: number;
}

interface ArrivedShape {
    readonly build: TopLevelBuild;
    /** Scene instance slot of each build input, in input order. */
    readonly slots: Uint32Array;
    readonly generation: number;
}

interface CurrentShape {
    readonly recordCount: number;
    readonly nodeCount: number;
    readonly maxDepth: number;
    readonly generation: number;
    /** Relative SAH cost as built. */
    readonly builtCost: number;
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
        qualityRebuilds: 0,
        lastBuildMs: 0,
        lastSnapshotMs: 0,
        refits: 0,
        refitIncomplete: 0,
    };

    private readonly device: GPUDevice;
    private readonly scene: PlatformScene;
    private readonly label: string;
    private readonly buildIn: 'worker' | 'main';
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
    /** Object-space mesh boxes by mesh slot, from the scene upload: the build's input for meshes without a tree yet. */
    private readonly meshBounds = new Map<number, Float32Array>();

    private current: CurrentShape = { recordCount: 0, nodeCount: 0, maxDepth: 0, generation: 0, builtCost: 0 };
    private arrived: ArrivedShape | null = null;
    /** Slots in the newest shape requested (built, in flight or swapped in). */
    private requestedSlots = new Set<number>();
    private buildInFlight = false;
    private buildAgain = false;
    private nextGeneration = 1;
    private buildFailure: Error | null = null;
    private idleWaiters: (() => void)[] = [];
    private seenStats: BvhRefitStats | null = null;
    private bufferVersion = 0;
    private disposed = false;

    /** Relative SAH cost of the refit tree, as last read back (0 before the first). */
    latestRelativeCost = 0;

    constructor(device: GPUDevice, scene: PlatformScene, options: GpuTopLevelOptions = {}) {
        this.device = device;
        this.scene = scene;
        this.label = options.label ?? 'gi';
        this.buildIn = options.buildIn ?? 'worker';
        this.refit = new GpuBvhRefit(device, { label: `${this.label} top level refit` });
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

    /** Records in the current shape (traceable instances when it was built). */
    get instanceCount(): number {
        return this.current.recordCount;
    }

    /** True while a shape is being built or waits to be swapped in. */
    get building(): boolean {
        return this.buildInFlight || this.arrived !== null;
    }

    get memoryBytes(): number {
        return this.topLevelBuffer.size + this.recordSlots.size + this.primitiveBoxes.size + this.meshTable.size + this.gatherParams.size + this.refit.memoryBytes;
    }

    /** Resolves once no shape is being built (one may still wait for the next `update` to be swapped in). */
    whenIdle(): Promise<void> {
        if (!this.buildInFlight) return Promise.resolve();
        return new Promise((resolve) => this.idleWaiters.push(resolve));
    }

    async prepare(kernels: KernelBuilder): Promise<void> {
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
        this.meshBounds.set(slot, Float32Array.from({ length: 6 }, (_, index) => bounds[index]));
    }

    /** A mesh's tree was placed (or replaced), or released (`null`). Takes effect at the next refit. */
    setMeshTree(slot: number, tree: PlacedMeshTree | null): void {
        if (tree === null) this.meshBounds.delete(slot);
        else this.meshBounds.set(slot, Float32Array.from({ length: 6 }, (_, index) => tree.bounds[index]));
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
            floats.set([tree.bounds[0], tree.bounds[1], tree.bounds[2]], 0);
            floats.set([tree.bounds[3], tree.bounds[4], tree.bounds[5]], 4);
            this.meshMirror[base + 3] = tree.nodeOffset;
            this.meshMirror[base + 7] = tree.triangleOffset;
        }
        this.meshTableDirty = true;
    }

    /**
     * Brings the top level up to this frame: swaps in an arrived shape,
     * starts a build if one is due, and records the refit pass (gather, then
     * refit) when anything it depends on changed. Called by the tracer in the
     * platform's `encode`, before any technique traces.
     */
    update(encoder: GPUCommandEncoder, frame: TopLevelFrame, timer?: GpuTimerScope): void {
        if (this.disposed) throw new Error('@pmndrs/ddgi: the top level was disposed');
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
        if (this.buildFailure) {
            const failure = this.buildFailure;
            this.buildFailure = null;
            throw failure;
        }

        this.refit.pollStats();
        this.checkQuality();
        if (frame.instancesChanged && this.hasUntracedInstance()) this.requestBuild();

        let changed = frame.instancesChanged || frame.meshesChanged;
        if (this.arrived) {
            this.swap(this.arrived);
            this.arrived = null;
            changed = true;
        }
        const requiredStack = this.current.maxDepth + 1 + frame.deepestMesh;
        if (this.current.recordCount > 0 && requiredStack > TRACE_STACK_SIZE) {
            throw new Error(
                `@pmndrs/ddgi: the trace stack holds ${TRACE_STACK_SIZE} entries, but this scene can need ${requiredStack} ` +
                    `(top level ${this.current.maxDepth} + 1 + deepest mesh tree ${frame.deepestMesh}).`,
            );
        }
        if (this.meshTableDirty) {
            this.device.queue.writeBuffer(this.meshTable, 0, this.meshMirror);
            this.writeGatherParams();
            this.meshTableDirty = false;
            changed = true;
        }
        if (changed && this.current.recordCount > 0) this.encodeRefit(encoder, timer);
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

    /**
     * Whether an instance that contributes to GI (shown or hidden) is missing from the newest shape. One pass
     * over the scene's instances, run only on frames the scene reported an
     * instance change (DECISIONS proposal in PROGRESS, B1a: a scene flag would
     * make it free).
     */
    private hasUntracedInstance(): boolean {
        for (const instance of this.scene.instanceSnapshot()) {
            if (instance.contributesToGi && !this.requestedSlots.has(instance.slot)) return true;
        }
        return false;
    }

    private requestBuild(): void {
        if (this.buildInFlight) {
            this.buildAgain = true;
            return;
        }
        this.startBuild();
    }

    private startBuild(): void {
        const start = performance.now();
        const instances = this.scene
            .instanceSnapshot()
            // Hidden ones too: the gather gives them empty boxes, and showing them again needs no new shape.
            .filter((instance) => instance.contributesToGi && this.meshBounds.has(instance.meshSlot));
        const slots = new Uint32Array(instances.length);
        const objectToWorld = new Float64Array(instances.length * 16);
        const meshBounds = new Float32Array(instances.length * 6);
        instances.forEach((instance, index) => {
            slots[index] = instance.slot;
            objectToWorld.set(instance.objectToWorld.subarray(0, 16), index * 16);
            meshBounds.set(this.meshBounds.get(instance.meshSlot) as Float32Array, index * 6);
        });
        this.requestedSlots = new Set(slots);
        this.counters.lastSnapshotMs = performance.now() - start;

        const generation = this.nextGeneration++;
        this.buildInFlight = true;
        this.builder ??= createTopLevelBuilder({ worker: this.buildIn === 'main' ? 'never' : 'auto' });
        this.builder.build({ objectToWorld, meshBounds }).then(
            (build) => {
                if (this.disposed) return;
                this.arrived = { build, slots, generation };
                this.finishBuild();
            },
            (error: unknown) => {
                if (this.disposed) return;
                this.buildFailure = new Error(`@pmndrs/ddgi: building the top level failed: ${error instanceof Error ? error.message : String(error)}`);
                this.finishBuild();
            },
        );
    }

    private finishBuild(): void {
        this.buildInFlight = false;
        if (this.buildAgain) {
            // Something changed while this one was building: start the next right away, so
            // `whenIdle` only resolves once the shape covers the scene as it is now.
            this.buildAgain = false;
            this.startBuild();
            return;
        }
        for (const resolve of this.idleWaiters.splice(0)) resolve();
    }

    /** Puts an arrived shape in place. Every write lands before this frame's commands run. */
    private swap(arrived: ArrivedShape): void {
        const { shape, buildMs } = arrived.build;
        const recordCount = arrived.slots.length;
        const nodeCount = shape.nodeCount;
        const nodeByteOffset = TOP_LEVEL_HEADER_BYTES + recordCount * INSTANCE_RECORD_BYTES;
        this.ensureTopLevelBytes(nodeByteOffset + nodeCount * BVH_NODE_BYTES);

        const header = new Uint32Array(TOP_LEVEL_HEADER_WORDS);
        header[TOP_HEADER_WORD.instanceCount] = recordCount;
        header[TOP_HEADER_WORD.nodeCount] = nodeCount;
        header[TOP_HEADER_WORD.maxDepth] = shape.maxDepth;
        this.device.queue.writeBuffer(this.topLevelBuffer, 0, header);
        if (nodeCount > 0) this.device.queue.writeBuffer(this.topLevelBuffer, nodeByteOffset, shape.nodes, 0, nodeCount * BVH_NODE_BYTES);

        // Records are in tree order; record r holds the instance at build input position order[r].
        const recordSlots = new Uint32Array(Math.max(1, recordCount));
        for (let record = 0; record < recordCount; record++) recordSlots[record] = arrived.slots[shape.order[record]];
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
            generation: arrived.generation,
        });
        this.current = { recordCount, nodeCount, maxDepth: shape.maxDepth, generation: arrived.generation, builtCost: shape.relativeCost };
        this.writeGatherParams();
        this.counters.shapesBuilt++;
        this.counters.lastBuildMs = buildMs;
    }

    /** Reads the latest refit stats: counts incomplete refits, and asks for a new shape when quality has decayed. */
    private checkQuality(): void {
        const stats = this.refit.latestStats;
        if (!stats || stats === this.seenStats) return;
        this.seenStats = stats;
        this.counters.refitIncomplete += stats.incomplete;
        if (stats.generation !== this.current.generation) return;
        this.latestRelativeCost = stats.relativeCost;
        const decayed = this.current.builtCost > 0 && stats.relativeCost > TOP_LEVEL_REBUILD_COST_RATIO * this.current.builtCost;
        if (decayed && !this.buildInFlight && this.arrived === null) {
            this.counters.qualityRebuilds++;
            this.requestBuild();
        }
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
