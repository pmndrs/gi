/**
 * Deforming meshes' trees (T3.3; docs/design/m3-moving.md §3 and §3.1):
 * every pose refits the mesh's tree on the GPU, and a tree whose refit has
 * drifted for about a second is rebuilt.
 *
 * **The refit.** A mesh created `deforming` keeps the shape of the tree its
 * builder made (H-PLOC's or tinybvh's, ray-queries.ts). Each frame that gives
 * it a pose, `GpuMeshRefit` (`@pmndrs/bvh`) rewrites the corners of its
 * triangle records from the scene copy's positions and refits its boxes
 * bottom-up, in the frame's own encoder: no submit and no CPU work per
 * vertex. The scene copy got the pose earlier in the same frame
 * (platform-scene.ts), so rays and hit shading always see the same positions.
 * A tree placed (or replaced) is refit at once from the positions the scene
 * copy holds then, so a tree built from an older pose never shows.
 *
 * The refit leaves each record's flags word (word 7: M4's alpha test and
 * glass) as it was, so a flag written after the tree's install survives
 * every pose.
 *
 * **The top level** keeps the mesh's `deformBounds` box (the box every pose
 * stays in, ray-queries.ts), not the refit root's, so a pose never needs a
 * readback or a new top-level shape (DECISIONS 2026-11-09).
 *
 * **Held drift** (DECISIONS 2026-11-09, changing §3.1's plain 1.1× rule). A
 * refit tree's quality depends only on how far the pose is from the one it
 * was built in (T3.0: a refit keeps no history). The refit's stats (its
 * relative SAH cost) are read back a few frames late, a few words a mesh
 * (the readback §3.1's DECISIONS entry allows); when the cost has stayed
 * above {@link DRIFT_THRESHOLD} times the built tree's own for
 * {@link DRIFT_HOLD_FRAMES} frames, the tracer rebuilds the mesh with H-PLOC
 * from the scene copy's current positions, through the same runtime path as
 * an uploaded mesh (T3.2), and this refit takes the new tree's shape. A sway
 * passes through drifted poses and back, so it rarely holds long enough; a
 * bend that stays (an arm raised) does.
 *
 * Needs subgroups (the refit's pairing step). Without them the platform keeps
 * deforming meshes in their rest pose (platform-scene.ts, `posesTraced`).
 */

import { GpuMeshRefit, GpuMeshRefitKernels } from '@pmndrs/bvh/gpu-mesh-refit';
import type { GpuLayoutMeshBvh } from '@pmndrs/bvh/gpu-layout';
import type { GpuLbvhBufferGeometry } from '@pmndrs/bvh/gpu-lbvh';
import { relativeSahCost } from '@pmndrs/bvh/refit-cpu';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { MeshLayout } from '../scene/mesh-packing.ts';
import { BVH_NODE_BYTES, TRIANGLE_BYTES } from './layout.ts';

/** The timed pass's name (`stats.passes`). */
export const MESH_REFIT_PASS = 'platform/mesh refit';

/**
 * Refit cost over the built tree's, above which a pose counts as drifted.
 * §3.1's number: a mesh tree is most of a ray's walk, so its cost tracks
 * trace speed closely, and 1.1× matches the BVH done-when (within 10% of a
 * fresh build).
 */
export const DRIFT_THRESHOLD = 1.1;

/** Frames the drift must hold, about a second at 60 Hz, before a rebuild (DECISIONS 2026-11-09). */
export const DRIFT_HOLD_FRAMES = 60;

const NODE_WORDS = BVH_NODE_BYTES / 4;
const TRIANGLE_WORDS = TRIANGLE_BYTES / 4;

/** Where a placed tree sits in the tracer's shared buffers, in nodes and triangles. */
export interface DeformingTreePlacement {
    readonly nodeOffset: number;
    readonly triangleOffset: number;
}

/** Counts for `stats.counters`. */
export interface DeformingTreeCounts {
    /** Deforming meshes whose tree is refit from their poses now. */
    readonly refitting: number;
    /** Mesh refits recorded, since the platform was made. */
    readonly refits: number;
    /** Trees rebuilt because their refit had drifted, since the platform was made. */
    readonly driftRebuilds: number;
}

interface DeformingMesh {
    readonly slot: number;
    readonly refit: GpuMeshRefit;
    /** Where the mesh's positions and indices are in the scene copy's vertex buffer. */
    layout: MeshLayout;
    vertexOffset: number;
    /** The placed tree the refit follows, or null while none is placed (or the mesh was re-uploaded and its new tree hasn't come). */
    placement: DeformingTreePlacement | null;
    /** The placed tree's relative SAH cost as it was built: what drift is measured against. */
    builtCost: number;
    /** Tells this tree's stats from an earlier tree's (`GpuBvhRefit`'s generation). */
    shapeGeneration: number;
    /** The tree must be refit this frame: a new pose, or a tree just placed. */
    refitDue: boolean;
    /** The first frame of the current run of drifted stats, or -1. */
    driftSince: number;
    /** A rebuild of this mesh is running. */
    rebuilding: boolean;
}

export class DeformingTrees {
    static isSupported(device: GPUDevice): boolean {
        return GpuMeshRefit.isSupported(device);
    }

    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly kernels: GpuMeshRefitKernels;
    private readonly meshes = new Map<number, DeformingMesh>();
    private nextShapeGeneration = 1;
    private refits = 0;
    private driftRebuilds = 0;

    constructor(device: GPUDevice, options: { readonly label?: string } = {}) {
        this.device = device;
        this.label = `${options.label ?? 'gi'} mesh refit`;
        this.kernels = new GpuMeshRefitKernels(device, this.label);
    }

    prepare(): Promise<void> {
        return this.kernels.prepare();
    }

    get counts(): DeformingTreeCounts {
        let refitting = 0;
        for (const mesh of this.meshes.values()) if (mesh.placement) refitting++;
        return { refitting, refits: this.refits, driftRebuilds: this.driftRebuilds };
    }

    /** GPU bytes of the refits' own buffers (the trees are in the tracer's). */
    get memoryBytes(): number {
        let bytes = 0;
        for (const mesh of this.meshes.values()) bytes += mesh.refit.memoryBytes;
        return bytes;
    }

    /** Whether the slot holds a deforming mesh. */
    has(slot: number): boolean {
        return this.meshes.has(slot);
    }

    /** The box every pose of the mesh stays in (its layout's), or null for a mesh that doesn't deform. */
    deformBox(slot: number): Float32Array | null {
        return this.meshes.get(slot)?.layout.bounds ?? null;
    }

    /**
     * A mesh was created or reshaped (`meshesUploaded`). A deforming one gets
     * (or keeps) its refit, and stops refitting its old tree: the scene copy
     * now holds the new shape, which the old tree's triangle numbers don't
     * describe. One that no longer deforms is forgotten.
     */
    meshUploaded(slot: number, layout: MeshLayout, vertexOffset: number): void {
        if (!layout.deforming) {
            this.meshRemoved(slot);
            return;
        }
        const existing = this.meshes.get(slot);
        if (existing) {
            existing.layout = layout;
            existing.vertexOffset = vertexOffset;
            existing.placement = null;
            existing.refitDue = false;
            existing.driftSince = -1;
            return;
        }
        this.meshes.set(slot, {
            slot,
            refit: new GpuMeshRefit(this.device, { label: `${this.label} ${slot}`, diagnostics: true, kernels: this.kernels }),
            layout,
            vertexOffset,
            placement: null,
            builtCost: 0,
            shapeGeneration: 0,
            refitDue: false,
            driftSince: -1,
            rebuilding: false,
        });
    }

    meshRemoved(slot: number): void {
        const mesh = this.meshes.get(slot);
        if (!mesh) return;
        mesh.refit.dispose();
        this.meshes.delete(slot);
    }

    /** A pose reached the scene copy in this frame's encode: the tree follows it in the same frame. */
    posed(slot: number, layout: MeshLayout, vertexOffset: number): void {
        const mesh = this.meshes.get(slot);
        if (!mesh) return;
        mesh.layout = layout;
        mesh.vertexOffset = vertexOffset;
        mesh.refitDue = true;
    }

    /**
     * A tree of the mesh's current version was placed. With its layout on the
     * CPU (tinybvh's always; a GPU build's when it was read back), the refit
     * takes its shape and refits it this frame from the scene copy's
     * positions. Without one the mesh can't be refit: it keeps the pose the
     * tree was built in until a tree with a layout arrives.
     */
    treePlaced(slot: number, placement: DeformingTreePlacement, layout: GpuLayoutMeshBvh | null): void {
        const mesh = this.meshes.get(slot);
        if (!mesh) return;
        mesh.rebuilding = false;
        mesh.driftSince = -1;
        if (!layout) {
            mesh.placement = null;
            return;
        }
        mesh.shapeGeneration = this.nextShapeGeneration++;
        mesh.refit.setTree(layout, {
            nodeWordOffset: placement.nodeOffset * NODE_WORDS,
            triangleWordOffset: placement.triangleOffset * TRIANGLE_WORDS,
            generation: mesh.shapeGeneration,
        });
        const words = new Uint32Array(layout.nodes, 0, layout.nodeCount * NODE_WORDS);
        const floats = new Float32Array(layout.nodes, 0, layout.nodeCount * NODE_WORDS);
        mesh.builtCost = relativeSahCost(floats, words, 0, layout.nodeCount);
        mesh.placement = placement;
        mesh.refitDue = true;
    }

    /** The mesh's tree was released (removed or replaced): nothing to refit until the next is placed. */
    treeReleased(slot: number): void {
        const mesh = this.meshes.get(slot);
        if (mesh) mesh.placement = null;
    }

    /** Starts reading back stats copied in earlier frames (call at the frame's start; never waits). */
    pollStats(): void {
        for (const mesh of this.meshes.values()) mesh.refit.pollStats();
    }

    /**
     * Records every due refit into one compute pass in `encoder`, then the
     * stats copies. Called after the frame's tree placements (and any growth
     * of the tracer's buffers), before anything traces.
     */
    encode(encoder: GPUCommandEncoder, buffers: { readonly vertices: GPUBuffer; readonly nodes: GPUBuffer; readonly triangles: GPUBuffer }, timer?: GpuTimerScope): void {
        if (!this.kernels.ready) return;
        const due: DeformingMesh[] = [];
        for (const mesh of this.meshes.values()) if (mesh.refitDue && mesh.placement) due.push(mesh);
        if (due.length === 0) return;
        const pass = encoder.beginComputePass({ label: this.label, timestampWrites: timer?.computePass(MESH_REFIT_PASS) });
        for (const mesh of due) {
            const layout = mesh.layout;
            mesh.refit.encode(pass, { nodes: buffers.nodes, triangles: buffers.triangles }, sceneCopyVertices(buffers.vertices, layout, mesh.vertexOffset));
            mesh.refitDue = false;
            this.refits++;
        }
        pass.end();
        for (const mesh of due) mesh.refit.copyStats(encoder);
    }

    /**
     * The mesh to rebuild this frame, if any: the one whose refit has drifted
     * longest, once it has held for {@link DRIFT_HOLD_FRAMES}. At most one a
     * frame. `frame` counts the tracer's updates.
     */
    rebuildDue(frame: number): number | null {
        let chosen: DeformingMesh | null = null;
        for (const mesh of this.meshes.values()) {
            this.readDrift(mesh, frame);
            if (mesh.rebuilding || !mesh.placement || mesh.driftSince < 0) continue;
            if (frame - mesh.driftSince < DRIFT_HOLD_FRAMES) continue;
            if (!chosen || mesh.driftSince < chosen.driftSince) chosen = mesh;
        }
        return chosen?.slot ?? null;
    }

    /** The scene copy's positions and indices of the mesh, as a GPU builder reads them (the drift rebuild). */
    rebuildGeometry(slot: number, vertices: GPUBuffer): GpuLbvhBufferGeometry | null {
        const mesh = this.meshes.get(slot);
        if (!mesh) return null;
        return { ...sceneCopyVertices(vertices, mesh.layout, mesh.vertexOffset), triangleCount: mesh.layout.triangleCount };
    }

    /** The tracer started (or failed to start) a rebuild of the mesh. */
    rebuildStarted(slot: number, started: boolean): void {
        const mesh = this.meshes.get(slot);
        if (!mesh) return;
        mesh.driftSince = -1;
        if (!started) return;
        mesh.rebuilding = true;
        this.driftRebuilds++;
    }

    /** The rebuild ended without a usable tree: the mesh goes on refitting the tree it has. */
    rebuildRefused(slot: number): void {
        const mesh = this.meshes.get(slot);
        if (mesh) mesh.rebuilding = false;
    }

    dispose(): void {
        for (const mesh of this.meshes.values()) mesh.refit.dispose();
        this.meshes.clear();
    }

    /** Updates the mesh's drift run from its latest stats of the current tree. */
    private readDrift(mesh: DeformingMesh, frame: number): void {
        const stats = mesh.refit.latestStats;
        if (!stats || stats.generation !== mesh.shapeGeneration || !(mesh.builtCost > 0)) return;
        const drifted = stats.relativeCost > DRIFT_THRESHOLD * mesh.builtCost;
        if (!drifted) mesh.driftSince = -1;
        else if (mesh.driftSince < 0) mesh.driftSince = frame;
    }
}

/**
 * A mesh's positions and indices in the scene copy's vertex buffer, as the
 * refit and the GPU builders take them: positions 3 floats a vertex at
 * `vertexOffset + offsets.positions` words; indices 16-bit two to a word
 * (counted in indices, so twice the word offset) or 32-bit.
 */
function sceneCopyVertices(vertices: GPUBuffer, layout: MeshLayout, vertexOffset: number) {
    const indexWord = vertexOffset + layout.offsets.indices;
    return {
        positionBuffer: vertices,
        vertexCount: layout.vertexCount,
        positionStrideFloats: 3,
        positionOffsetFloats: vertexOffset + layout.offsets.positions,
        indexBuffer: vertices,
        indexFormat: layout.indices16 ? ('uint16' as const) : ('uint32' as const),
        indexOffset: layout.indices16 ? indexWord * 2 : indexWord,
    };
}
