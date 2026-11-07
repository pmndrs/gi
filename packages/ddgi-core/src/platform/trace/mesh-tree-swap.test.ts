/**
 * First trees from the GPU and tinybvh's swapped in (T2.12): which tree a mesh
 * gets, when, and that rays hit the same triangles throughout. Node has no
 * GPU, so a stand-in first-tree builder makes the "GPU" trees: tinybvh trees
 * with one triangle per leaf (a different shape from the worker's, as
 * H-PLOC's is), written into fake buffers in the tracer's layout. The GPU's
 * own builds are checked on real GPUs (tools/gpu/pages/trace-oracle).
 */

import type { GpuLbvhMeshTree } from '@pmndrs/bvh/gpu-lbvh';
import { describe, expect, it } from 'vitest';
import { PlatformScene } from '../scene/platform-scene.ts';
import { createFakeGpu, type FakeBuffer, type FakeGpu } from '../scene/testing/fake-gpu.ts';
import { createCpuTracer } from './cpu-traversal.ts';
import { GpuTreeOutputs, type FirstTreeBuilder, type GpuTreeBatchResult, type GpuTreeBuilds, type GpuTreeRequest } from './gpu-mesh-trees.ts';
import { TOP_HEADER_WORD } from './layout.ts';
import { buildMeshBvhAsync } from './mesh-bvh-build.ts';
import { PlatformRayQueries, type MeshTreeOptions } from './ray-queries.ts';
import type { TraceSceneData } from './scene-assembly.ts';
import type { TopLevelBvh } from './top-level.ts';

/** The fake GPU plus what the tracer needs: features (none, so the top level is the CPU fallback) and layouts. */
function createGpu(): FakeGpu {
    const fake = createFakeGpu();
    Object.assign(fake.device as unknown as Record<string, unknown>, {
        features: new Set<string>(),
        createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ descriptor }),
        createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ descriptor }),
    });
    return fake;
}

/**
 * A first-tree builder that answers when the test says so. Each `build` call
 * is one batch; `finishBatch` resolves the oldest, building its trees then
 * (one triangle per leaf, so their shape differs from tinybvh's in the worker).
 */
class StandInGpuBuilder implements FirstTreeBuilder {
    readonly requests: GpuTreeRequest[][] = [];
    private readonly pending: { members: { slot: number; generation: number; positions: Float32Array; indices: Uint32Array }[]; resolve: (result: GpuTreeBatchResult) => void }[] = [];
    private readonly device: GPUDevice;
    /** Slots whose tree the builder's own checks refuse. */
    readonly refuse = new Set<number>();
    outputs: GpuTreeOutputs[] = [];

    constructor(device: GPUDevice) {
        this.device = device;
    }

    /** Loads tinybvh's WASM now, so `finishBatch` answers within the task that calls it, ahead of the worker's later tasks. */
    async prepare(): Promise<void> {
        await buildMeshBvhAsync({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]) });
    }

    build(requests: readonly GpuTreeRequest[]): GpuTreeBuilds {
        this.requests.push([...requests]);
        // The geometry is read when the build starts, as the real builder writes it to its buffers.
        const members = requests.map(({ slot, generation, geometry }) => ({
            slot,
            generation,
            positions: geometry.positions.slice(),
            indices: Uint32Array.from(geometry.indices ?? []),
        }));
        const batch = new Promise<GpuTreeBatchResult>((resolve) => this.pending.push({ members, resolve }));
        return { notBuilt: [], batches: [batch] };
    }

    get waiting(): number {
        return this.pending.length;
    }

    /** Builds and reports the oldest batch. */
    async finishBatch(): Promise<void> {
        const batch = this.pending.shift();
        if (!batch) throw new Error('no batch is waiting');
        const trees = await Promise.all(batch.members.map((member) => buildMeshBvhAsync({ positions: member.positions, indices: member.indices }, { maxLeafPrimitives: 1 })));
        const nodeBytes = trees.reduce((sum, tree) => sum + tree.nodes.byteLength, 0);
        const triangleBytes = trees.reduce((sum, tree) => sum + tree.triangles.byteLength, 0);
        const nodes = this.device.createBuffer({ label: 'stand-in nodes', size: nodeBytes, usage: 0 });
        const triangles = this.device.createBuffer({ label: 'stand-in triangles', size: triangleBytes, usage: 0 });
        const parents = this.device.createBuffer({ label: 'stand-in parents', size: 4, usage: 0 });
        const meshes: GpuLbvhMeshTree[] = [];
        let nodeOffset = 0;
        let triangleOffset = 0;
        trees.forEach((tree, index) => {
            this.device.queue.writeBuffer(nodes, nodeOffset * 64, tree.nodes);
            this.device.queue.writeBuffer(triangles, triangleOffset * 48, tree.triangles);
            const refused = this.refuse.has(batch.members[index].slot);
            meshes.push({ ...tree, nodeOffset, triangleOffset, problem: refused ? 'refused by the test' : null });
            nodeOffset += tree.nodeCount;
            triangleOffset += tree.triangleCount;
        });
        const usable = meshes.filter((mesh) => mesh.problem === null).length;
        const outputs = new GpuTreeOutputs({ nodes, triangles, parents, meshes, destroy: () => [nodes, triangles, parents].forEach((buffer) => buffer.destroy()) }, usable);
        this.outputs.push(outputs);
        batch.resolve({
            built: meshes.flatMap((tree, index) => (tree.problem ? [] : [{ slot: batch.members[index].slot, generation: batch.members[index].generation, outputs, tree }])),
            refused: meshes.flatMap((tree, index) => (tree.problem ? [{ slot: batch.members[index].slot, generation: batch.members[index].generation, reason: tree.problem }] : [])),
            buildMs: 1,
            failure: null,
        });
        // Let the platform's `then` run.
        await Promise.resolve();
        await Promise.resolve();
    }
}

/** A square of `size` metres in the z = 0 plane at x = `x`, cut into a grid of triangles (so trees have a few levels). */
function grid(x: number, size = 1, cells = 4): { positions: Float32Array; indices: Uint32Array } {
    const positions: number[] = [];
    const indices: number[] = [];
    for (let row = 0; row <= cells; row++) {
        for (let column = 0; column <= cells; column++) positions.push(x + (column / cells - 0.5) * size, (row / cells - 0.5) * size, 0);
    }
    for (let row = 0; row < cells; row++) {
        for (let column = 0; column < cells; column++) {
            const corner = row * (cells + 1) + column;
            indices.push(corner, corner + 1, corner + cells + 2, corner, corner + cells + 2, corner + cells + 1);
        }
    }
    return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

async function setUp(options: { meshes?: number; meshTrees?: MeshTreeOptions; stillBuilding?: boolean } = {}) {
    const gpu = createGpu();
    const scene = new PlatformScene(gpu.device);
    const gpuBuilder = new StandInGpuBuilder(gpu.device);
    const rays = new PlatformRayQueries(gpu.device, scene, { buildIn: 'main', meshTrees: { firstTreeBuilder: gpuBuilder, ...options.meshTrees } });
    await rays.prepare({} as never, { batches: false });
    const frame = () => {
        const encoder = gpu.createEncoder();
        rays.update(encoder, scene.encode(encoder));
        gpu.submit(encoder);
    };
    const material = scene.createMaterial({});
    const meshes = Array.from({ length: options.meshes ?? 3 }, (_, index) => scene.createMesh(grid(index * 3)));
    const instances = meshes.map((mesh) => scene.createInstance({ mesh, materials: [material], transform: IDENTITY }));
    return { gpu, scene, rays, gpuBuilder, frame, material, meshes, instances };
}

/** The tracer's buffers as the fake GPU holds them, for the CPU walk. */
function traceScene(rays: PlatformRayQueries): TraceSceneData {
    const [nodes, triangles, topLevel] = rays.module.resources().map((resource) => (resource as GPUBufferBinding).buffer as unknown as FakeBuffer);
    const topWords = new Uint32Array(topLevel.bytes.slice().buffer);
    return {
        nodes: nodes.bytes.slice().buffer,
        triangles: triangles.bytes.slice().buffer,
        topLevel: { buffer: topWords.buffer, instanceCount: topWords[TOP_HEADER_WORD.instanceCount], nodeCount: topWords[TOP_HEADER_WORD.nodeCount] } as TopLevelBvh,
        meshes: [],
        maxMeshDepth: 0,
        requiredStackSize: 0,
        byteLength: 0,
    };
}

/** Rays straight down onto every mesh: which instance and triangle each hits (or -1). */
function hitsDown(rays: PlatformRayQueries, meshes: number): string[] {
    const tracer = createCpuTracer(traceScene(rays));
    const hits: string[] = [];
    for (let mesh = 0; mesh < meshes; mesh++) {
        for (const [dx, dy] of [[0.1, 0.2], [-0.3, 0.37], [0.41, -0.13]]) {
            const hit = tracer.closest({ origin: [mesh * 3 + dx, dy, 5], direction: [0, 0, -1], tMin: 0, tMax: 100 });
            hits.push(hit ? `${hit.instanceIndex}/${hit.triangleIndex}/${hit.t.toFixed(5)}` : '-1');
        }
    }
    return hits;
}

describe('first trees from the GPU, tinybvh swapped in (T2.12)', () => {
    it("gives each mesh the first tree to arrive, and drops a GPU tree that comes after tinybvh's", async () => {
        const { rays, gpuBuilder, frame, gpu } = await setUp();
        frame();
        // One batch for all meshes.
        expect(gpuBuilder.requests).toHaveLength(1);
        expect(gpuBuilder.requests[0].map((request) => request.slot)).toEqual([0, 1, 2]);
        // tinybvh's trees finish while the GPU build still runs (the stand-in waits for the test): they go in.
        await rays.whenBuilt();
        expect(gpuBuilder.waiting).toBe(1);
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 3 });
        expect(hitsDown(rays, 3).filter((hit) => hit === '-1')).toHaveLength(0);

        // The GPU's trees come later and are dropped; their batch's buffers go at the next update.
        await gpuBuilder.finishBatch();
        expect(rays.memory.gpuBuildOutputs).toBeGreaterThan(0);
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 3, gpuTreesBuilt: 3, swaps: 0 });
        frame();
        expect(gpuBuilder.outputs[0].isDestroyed).toBe(true);
        expect(rays.memory.gpuBuildOutputs).toBe(0);
        expect(gpu.buffers.filter((buffer) => buffer.label.startsWith('stand-in') && !buffer.destroyed)).toHaveLength(0);
        rays.dispose();
    });

    it("swaps tinybvh's trees in by themselves when the GPU's came first, with the same hits", async () => {
        const { rays, gpuBuilder, frame } = await setUp();
        frame();
        // The stand-in answers within this task, before the worker's (here: a later task's) trees arrive.
        await gpuBuilder.finishBatch();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 3, onTinybvhTrees: 0, tinybvhBuilding: 3 });
        expect(rays.building).toBe(false);
        expect(rays.refining).toBe(true);
        const onGpuTrees = hitsDown(rays, 3);
        expect(onGpuTrees.filter((hit) => hit === '-1')).toHaveLength(0);
        await rays.whenRefined();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 3, swaps: 3 });
        expect(rays.refining).toBe(false);
        expect(hitsDown(rays, 3)).toEqual(onGpuTrees);
        rays.dispose();
    });

    it('holds tinybvh trees while asked, and releases some (mid-swap) or all, with the same hits in every state', async () => {
        const { rays, gpuBuilder, frame } = await setUp({ meshes: 4, meshTrees: { holdRefinements: true } });
        frame();
        await gpuBuilder.finishBatch();
        frame();
        await rays.whenRefined();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 4, onTinybvhTrees: 0, tinybvhBuilding: 0 });
        const onGpuTrees = hitsDown(rays, 4);
        expect(onGpuTrees.filter((hit) => hit === '-1')).toHaveLength(0);

        rays.releaseRefinements((slot) => slot % 2 === 0);
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 2, swaps: 2 });
        expect(hitsDown(rays, 4)).toEqual(onGpuTrees);

        rays.releaseRefinements();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 4, swaps: 4 });
        expect(hitsDown(rays, 4)).toEqual(onGpuTrees);
        // The GPU trees were copied out long ago: their batch's buffers went one update after that.
        expect(gpuBuilder.outputs[0].isDestroyed).toBe(true);
        rays.dispose();
    });

    it("leaves a refused GPU tree's mesh to tinybvh, and refuses trees deeper than gpuMaxDepth", async () => {
        const { rays, gpuBuilder, frame } = await setUp({ meshTrees: { holdRefinements: true } });
        gpuBuilder.refuse.add(1);
        frame();
        await gpuBuilder.finishBatch();
        frame();
        await rays.whenBuilt();
        frame();
        // Mesh 1 waited for tinybvh; the held refinements don't apply to a mesh with no GPU tree.
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 1, gpuTreesRefused: 1 });
        expect(hitsDown(rays, 3).filter((hit) => hit === '-1')).toHaveLength(0);
        rays.dispose();

        const deep = await setUp({ meshTrees: { holdRefinements: true, gpuMaxDepth: 1 } });
        deep.frame();
        await deep.gpuBuilder.finishBatch();
        await deep.rays.whenBuilt();
        deep.frame();
        expect(deep.rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 3, gpuTreesRefused: 3 });
        // Too deep: the outputs are let go of at once.
        deep.frame();
        expect(deep.gpuBuilder.outputs[0].isDestroyed).toBe(true);
        deep.rays.dispose();
    });

    it('builds meshes added after the first trees with tinybvh only, and drops a GPU tree whose mesh changed meanwhile', async () => {
        const { rays, gpuBuilder, frame, scene, material, meshes } = await setUp();
        frame();
        // Mesh 2 is reshaped while its GPU build runs: that tree is for an old version.
        scene.updateMesh(meshes[2], grid(6, 2));
        frame();
        expect(gpuBuilder.requests).toHaveLength(2);
        await gpuBuilder.finishBatch();
        await gpuBuilder.finishBatch();
        await rays.whenRefined();
        frame();
        frame();
        // Mesh 2's first build was for its old version: not counted, and its tree never placed.
        expect(rays.meshTreeCounts.gpuTreesBuilt).toBe(3);
        expect(rays.meshTreeCounts.onTinybvhTrees).toBe(3);

        // The scene's first trees are in: a new mesh goes to tinybvh only.
        const added = scene.createMesh(grid(9));
        scene.createInstance({ mesh: added, materials: [material], transform: IDENTITY });
        frame();
        expect(gpuBuilder.requests).toHaveLength(2);
        expect(gpuBuilder.waiting).toBe(0);
        await rays.whenBuilt();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 4 });
        expect(hitsDown(rays, 4).filter((hit) => hit === '-1')).toHaveLength(0);
        rays.dispose();
    });

    it('uses tinybvh alone without subgroups, when the GPU builder fails to compile, or with gpuFirst off', async () => {
        const gpu = createGpu();
        const scene = new PlatformScene(gpu.device);
        // No stand-in builder: the device's features decide, and it has no subgroups.
        const rays = new PlatformRayQueries(gpu.device, scene, { buildIn: 'main' });
        expect(rays.gpuFirstTrees).toBe(false);
        await rays.prepare({} as never, { batches: false });
        scene.createInstance({ mesh: scene.createMesh(grid(0)), materials: [scene.createMaterial({})], transform: IDENTITY });
        const frame = () => {
            const encoder = gpu.createEncoder();
            rays.update(encoder, scene.encode(encoder));
            gpu.submit(encoder);
        };
        frame();
        await rays.whenBuilt();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 1, gpuTreesBuilt: 0 });
        expect(hitsDown(rays, 1).filter((hit) => hit === '-1')).toHaveLength(0);
        rays.dispose();

        // A GPU builder whose kernels fail to compile: tinybvh alone, with one warning, and prepare still succeeds.
        const broken = createGpu();
        const brokenScene = new PlatformScene(broken.device);
        const warnings: string[] = [];
        const failing: FirstTreeBuilder = { prepare: () => Promise.reject(new Error('no compile')), build: () => ({ notBuilt: [], batches: [] }) };
        const brokenRays = new PlatformRayQueries(broken.device, brokenScene, { buildIn: 'main', warn: (message) => warnings.push(message), meshTrees: { firstTreeBuilder: failing } });
        await brokenRays.prepare({} as never, { batches: false });
        expect(brokenRays.gpuFirstTrees).toBe(false);
        expect(warnings).toEqual([expect.stringContaining('no compile')]);
        brokenScene.createInstance({ mesh: brokenScene.createMesh(grid(0)), materials: [brokenScene.createMaterial({})], transform: IDENTITY });
        const brokenFrame = () => {
            const encoder = broken.createEncoder();
            brokenRays.update(encoder, brokenScene.encode(encoder));
            broken.submit(encoder);
        };
        brokenFrame();
        await brokenRays.whenBuilt();
        brokenFrame();
        expect(brokenRays.meshTreeCounts.onTinybvhTrees).toBe(1);
        brokenRays.dispose();

        const off = await setUp({ meshTrees: { gpuFirst: false } });
        expect(off.rays.gpuFirstTrees).toBe(false);
        off.frame();
        expect(off.gpuBuilder.requests).toHaveLength(0);
        await off.rays.whenBuilt();
        off.frame();
        expect(off.rays.meshTreeCounts.onTinybvhTrees).toBe(3);
        off.rays.dispose();
    });
});
