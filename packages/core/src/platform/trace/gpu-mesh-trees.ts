/**
 * Mesh trees from the GPU (T2.12 at load, T3.2 at runtime;
 * docs/design/m3-moving.md §2.2-2.3, "fast first GI"; final since #26):
 * every mesh the scene uploads, when it is loaded and when meshes are added
 * or replaced while it runs, has its tree built on the GPU with H-PLOC
 * (`@pmndrs/bvh`, B2, the default GPU builder), so rays can see it a few
 * frames later instead of after tinybvh's time in the worker (seconds for a
 * whole scene). The GPU's tree is the mesh's tree from then on; tinybvh
 * builds only the meshes the GPU refuses (ray-queries.ts).
 *
 * **One batch for all of a frame's meshes.** H-PLOC builds every mesh of a
 * batch with the same dispatches; building meshes one by one is
 * dispatch-bound (NOTES, LBVH: 97 ms against 32-42 ms for Sponza). The
 * meshes are split into more batches only when one doesn't fit the device's
 * buffers (`planGpuHplocBatches`).
 *
 * **Its own submit, and one small readback,** in each frame that uploads
 * meshes (DECISIONS 2026-11-08 for load, 2026-11-09 for runtime). A batch
 * is recorded into an encoder of its own and submitted at once, then its stats (node count,
 * depth, refused or not: a few words per mesh) are read back with `mapAsync`;
 * never the trees. Why not the host's encoder: the readback must come after
 * the build on the queue, and the platform can't know when the host submits
 * its frame. Submitting here also starts the build before the host's frame,
 * and the geometry is the platform's own CPU copy (written with
 * `queue.writeBuffer`, which lands before this submit), so nothing in the
 * host's encoder has to run first.
 *
 * **Deforming meshes (T3.3)** also need the tree's layout on the CPU, so the
 * refit can take its shape (deforming-trees.ts): their builds ask for
 * `readLayouts`, which reads their batch's trees back once, after the stats.
 * The tracer puts them in a batch of their own, so the readback is only
 * theirs. A drift rebuild builds from the scene copy's positions on the GPU
 * (`GpuMeshBuffers`), which the batch copies in its own encoder.
 *
 * Without subgroups there is no GPU builder: every mesh's tree comes from
 * tinybvh, as in M1 (the declared fallback, DECISIONS 2026-11-04).
 */

import { GpuHplocBuilder, planGpuHplocBatches } from '@pmndrs/bvh';
import type { GpuLayoutMeshBvh } from '@pmndrs/bvh';
import type { GpuMeshBuffers, GpuMeshTreeRange, GpuTreeBuffers } from '@pmndrs/bvh';
import type { MeshGeometry } from './mesh-bvh.ts';

/**
 * A mesh to build: its slot and version in the scene copy, and its geometry,
 * read once when the build starts: CPU arrays, or a range of a GPU buffer
 * (copied in the build's own commands, which are submitted at once).
 */
export interface GpuTreeRequest {
    readonly slot: number;
    readonly generation: number;
    readonly geometry: MeshGeometry | GpuMeshBuffers;
}

/** Options of one {@link FirstTreeBuilder.build}. */
export interface GpuTreeBuildOptions {
    /** Read every usable tree's layout back too (deforming meshes' refit needs it), once, after the stats. Default false. */
    readonly readLayouts?: boolean;
}

/**
 * The output buffers of one batch, shared by its trees. Each tree is copied
 * out of them into the tracer's buffers (or dropped); the buffers are freed
 * once every tree has let go of them.
 */
export class GpuTreeOutputs {
    private readonly trees: GpuTreeBuffers;
    private users: number;
    private destroyed = false;

    constructor(trees: GpuTreeBuffers, users: number) {
        this.trees = trees;
        this.users = users;
        if (users === 0) this.destroy();
    }

    get nodes(): GPUBuffer {
        return this.trees.nodes;
    }

    get triangles(): GPUBuffer {
        return this.trees.triangles;
    }

    /** GPU bytes held until every tree is released. */
    get bytes(): number {
        return this.destroyed ? 0 : this.trees.nodes.size + this.trees.triangles.size + this.trees.parents.size;
    }

    get isDestroyed(): boolean {
        return this.destroyed;
    }

    /** One tree is done with the buffers (copied out in a frame the host has submitted since, or dropped). */
    release(): void {
        if (this.destroyed) return;
        this.users--;
        if (this.users <= 0) this.destroy();
    }

    /** Frees the buffers whatever still uses them (the platform is being disposed). */
    destroy(): void {
        if (this.destroyed) return;
        this.destroyed = true;
        this.trees.destroy();
    }
}

/** One mesh's tree built on the GPU, inside its batch's output buffers. */
export interface GpuBuiltTree {
    readonly slot: number;
    readonly generation: number;
    readonly outputs: GpuTreeOutputs;
    /** Where the tree sits in `outputs` (nodes and triangles), its counts, depth and box. */
    readonly tree: GpuMeshTreeRange;
    /** The tree's layout on the CPU, when the build asked for `readLayouts` and the readback worked; else null. */
    readonly layout: GpuLayoutMeshBvh | null;
}

/** A mesh the GPU didn't give a usable tree: it falls back to tinybvh's. */
export interface GpuRefusedTree {
    readonly slot: number;
    readonly generation: number;
    readonly reason: string;
}

/** What one batch gave, once its stats were read back. */
export interface GpuTreeBatchResult {
    readonly built: readonly GpuBuiltTree[];
    readonly refused: readonly GpuRefusedTree[];
    /** Milliseconds from the submit to the stats on the CPU (the build plus the readback's latency). */
    readonly buildMs: number;
    /** Why the whole batch failed (it couldn't start, or its readback failed), or null. Its meshes are then all in `refused`. */
    readonly failure: string | null;
}

/** What {@link FirstTreeBuilder.build} started. */
export interface GpuTreeBuilds {
    /** Meshes too big for any batch on this device (or for H-PLOC): they fall back to tinybvh. */
    readonly notBuilt: readonly GpuRefusedTree[];
    /** One promise per batch. They never reject: a failed batch reports its meshes as refused. */
    readonly batches: readonly Promise<GpuTreeBatchResult>[];
}

/** Builds mesh trees for many meshes at once. The tracer's only GPU builder today is H-PLOC ({@link GpuMeshTrees}). */
export interface FirstTreeBuilder {
    /** Compiles the builder's kernels. */
    prepare(): Promise<void>;
    /** Starts building every request's tree; see {@link GpuTreeBuilds}. Needs `prepare` to have finished. */
    build(requests: readonly GpuTreeRequest[], options?: GpuTreeBuildOptions): GpuTreeBuilds;
}

/** H-PLOC mesh trees ({@link FirstTreeBuilder}) on a device with subgroups. */
export class GpuMeshTrees implements FirstTreeBuilder {
    static isSupported(device: GPUDevice): boolean {
        return GpuHplocBuilder.isSupported(device);
    }

    private readonly device: GPUDevice;
    private readonly hploc: GpuHplocBuilder;
    private readonly label: string;

    constructor(device: GPUDevice, options: { readonly label?: string } = {}) {
        this.device = device;
        this.label = `${options.label ?? 'gi'} mesh trees`;
        this.hploc = new GpuHplocBuilder(device, { label: this.label });
    }

    prepare(): Promise<void> {
        return this.hploc.prepare();
    }

    build(requests: readonly GpuTreeRequest[], options: GpuTreeBuildOptions = {}): GpuTreeBuilds {
        const geometries = requests.map((request) => request.geometry);
        const plan = planGpuHplocBatches(this.device, geometries);
        const limitBytes = Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
        const notBuilt = plan.tooBig.map((index) => ({
            slot: requests[index].slot,
            generation: requests[index].generation,
            reason: `too big for a GPU build on this device (buffers of at most ${limitBytes} bytes)`,
        }));

        // Every batch goes into one encoder and one submit; each then reads its own stats back.
        const encoder = this.device.createCommandEncoder({ label: this.label });
        const recorded: { members: GpuTreeRequest[]; finish: () => Promise<GpuTreeBuffers> }[] = [];
        const failed: { members: GpuTreeRequest[]; reason: string }[] = [];
        for (const indices of plan.batches) {
            const members = indices.map((index) => requests[index]);
            try {
                const batch = this.hploc.createBatch(members.map((member) => member.geometry));
                batch.encode(encoder);
                recorded.push({ members, finish: () => batch.finish() });
            } catch (error) {
                failed.push({ members, reason: `the GPU mesh tree build could not start: ${describe(error)}` });
            }
        }
        const submittedAt = performance.now();
        this.device.queue.submit([encoder.finish()]);

        const batches = recorded.map(async ({ members, finish }): Promise<GpuTreeBatchResult> => {
            let trees: GpuTreeBuffers;
            try {
                trees = await finish();
            } catch (error) {
                const failure = `the GPU mesh tree build failed: ${describe(error)}`;
                return { built: [], refused: refuseAll(members, failure), buildMs: performance.now() - submittedAt, failure };
            }
            const buildMs = performance.now() - submittedAt;
            // A failed layout readback leaves the trees usable for tracing; only the refit goes without.
            let layouts: (GpuLayoutMeshBvh | null)[] = [];
            if (options.readLayouts) layouts = await this.hploc.readLayouts(trees).catch(() => []);
            const usable = trees.meshes.filter((tree) => tree.problem === null).length;
            const outputs = new GpuTreeOutputs(trees, usable);
            const built: GpuBuiltTree[] = [];
            const refused: GpuRefusedTree[] = [];
            trees.meshes.forEach((tree, index) => {
                const { slot, generation } = members[index];
                if (tree.problem === null) built.push({ slot, generation, outputs, tree, layout: layouts[index] ?? null });
                else refused.push({ slot, generation, reason: tree.problem });
            });
            return { built, refused, buildMs, failure: null };
        });
        for (const { members, reason } of failed) batches.push(Promise.resolve({ built: [], refused: refuseAll(members, reason), buildMs: 0, failure: reason }));
        return { notBuilt, batches };
    }
}

function refuseAll(members: readonly GpuTreeRequest[], reason: string): GpuRefusedTree[] {
    return members.map(({ slot, generation }) => ({ slot, generation, reason }));
}

function describe(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}
