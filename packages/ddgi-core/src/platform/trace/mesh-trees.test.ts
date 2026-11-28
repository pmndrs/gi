/**
 * Mesh trees from the GPU, with tinybvh as the declared fallback (T2.12,
 * T3.2, #26): which tree a mesh gets, when, and that rays hit the same
 * triangles whichever builder made it. Node has no GPU, so a stand-in builder
 * makes the "GPU" trees: tinybvh trees with one triangle per leaf (a
 * different shape from the worker's, as H-PLOC's is), written into fake
 * buffers in the tracer's layout. The GPU's own builds are checked on real
 * GPUs (tools/gpu/pages/trace-oracle).
 */

import type { GpuLbvhMeshTree } from '@pmndrs/bvh/gpu-lbvh';
import { describe, expect, it } from 'vitest';
import { PlatformKernelBuilder } from '../kernel-builder.ts';
import { PlatformScene } from '../scene/platform-scene.ts';
import { createFakeGpu, type FakeBuffer, type FakeGpu } from '../scene/testing/fake-gpu.ts';
import { createCpuTracer } from './cpu-traversal.ts';
import { GpuTreeOutputs, type FirstTreeBuilder, type GpuTreeBatchResult, type GpuTreeBuilds, type GpuTreeRequest } from './gpu-mesh-trees.ts';
import { TOP_HEADER_WORD } from './layout.ts';
import { buildMeshBvhAsync } from './mesh-bvh-build.ts';
import type { MeshGeometry } from './mesh-bvh.ts';
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
 * A GPU tree builder that answers when the test says so. Each `build` call
 * is one batch; `finishBatch` resolves the oldest, building its trees then
 * (one triangle per leaf, so their shape differs from tinybvh's in the worker).
 */
class StandInGpuBuilder implements FirstTreeBuilder {
    readonly requests: GpuTreeRequest[][] = [];
    private readonly pending: {
        members: { slot: number; generation: number; positions: Float32Array; indices: Uint32Array }[];
        fail: boolean;
        resolve: (result: GpuTreeBatchResult) => void;
    }[] = [];
    private readonly device: GPUDevice;
    /** Slots whose tree the builder's own checks refuse. */
    readonly refuse = new Set<number>();
    /** Slots too big to start a build for: `build` reports them itself, as `planGpuHplocBatches` makes the real one do. */
    readonly tooBig = new Set<number>();
    /** Whether the next batch fails as a whole (as a failed stats readback does). */
    failNextBatch = false;
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
        const notBuilt = requests.filter(({ slot }) => this.tooBig.has(slot)).map(({ slot, generation }) => ({ slot, generation, reason: 'too big for the test' }));
        // The geometry is read when the build starts, as the real builder writes it to its buffers.
        const members = requests
            .filter(({ slot }) => !this.tooBig.has(slot))
            .map(({ slot, generation, geometry }) => {
                // This stand-in builds from CPU arrays only (the tracer's drift rebuilds read the GPU; GPU pages test them).
                const cpu = geometry as MeshGeometry;
                return { slot, generation, positions: cpu.positions.slice(), indices: Uint32Array.from(cpu.indices ?? []) };
            });
        const fail = this.failNextBatch;
        this.failNextBatch = false;
        if (members.length === 0) return { notBuilt, batches: [] };
        const batch = new Promise<GpuTreeBatchResult>((resolve) => this.pending.push({ members, fail, resolve }));
        return { notBuilt, batches: [batch] };
    }

    get waiting(): number {
        return this.pending.length;
    }

    /** Builds and reports the oldest batch (or fails it whole, after `failNextBatch`). */
    async finishBatch(): Promise<void> {
        const batch = this.pending.shift();
        if (!batch) throw new Error('no batch is waiting');
        if (batch.fail) {
            const failure = 'the test failed the batch';
            batch.resolve({ built: [], refused: batch.members.map(({ slot, generation }) => ({ slot, generation, reason: failure })), buildMs: 1, failure });
        } else {
            batch.resolve(await this.buildBatch(batch.members));
        }
        // Let the platform's `then` run.
        await Promise.resolve();
        await Promise.resolve();
    }

    private async buildBatch(members: readonly { slot: number; generation: number; positions: Float32Array; indices: Uint32Array }[]): Promise<GpuTreeBatchResult> {
        const trees = await Promise.all(members.map((member) => buildMeshBvhAsync({ positions: member.positions, indices: member.indices }, { maxLeafPrimitives: 1 })));
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
            const refused = this.refuse.has(members[index].slot);
            meshes.push({ ...tree, nodeOffset, triangleOffset, problem: refused ? 'refused by the test' : null });
            nodeOffset += tree.nodeCount;
            triangleOffset += tree.triangleCount;
        });
        const usable = meshes.filter((mesh) => mesh.problem === null).length;
        const outputs = new GpuTreeOutputs({ nodes, triangles, parents, meshes, destroy: () => [nodes, triangles, parents].forEach((buffer) => buffer.destroy()) }, usable);
        this.outputs.push(outputs);
        return {
            built: meshes.flatMap((tree, index) => (tree.problem ? [] : [{ slot: members[index].slot, generation: members[index].generation, outputs, tree, layout: trees[index] }])),
            refused: meshes.flatMap((tree, index) => (tree.problem ? [{ slot: members[index].slot, generation: members[index].generation, reason: tree.problem }] : [])),
            buildMs: 1,
            failure: null,
        };
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

async function setUp(options: { meshes?: number; meshTrees?: MeshTreeOptions; cells?: number } = {}) {
    const gpu = createGpu();
    const scene = new PlatformScene(gpu.device);
    const gpuBuilder = new StandInGpuBuilder(gpu.device);
    const warnings: string[] = [];
    const rays = new PlatformRayQueries(gpu.device, scene, {
        buildIn: 'main',
        warn: (message) => warnings.push(message),
        meshTrees: { firstTreeBuilder: gpuBuilder, ...options.meshTrees },
    });
    await rays.prepare(new PlatformKernelBuilder(gpu.device, 'test'), { batches: false });
    const frame = () => {
        const encoder = gpu.createEncoder();
        rays.update(encoder, scene.encode(encoder));
        gpu.submit(encoder);
    };
    const material = scene.createMaterial({});
    const meshes = Array.from({ length: options.meshes ?? 3 }, (_, index) => scene.createMesh(grid(index * 3, 1, options.cells)));
    const instances = meshes.map((mesh) => scene.createInstance({ mesh, materials: [material], transform: IDENTITY }));
    return { gpu, scene, rays, gpuBuilder, frame, material, meshes, instances, warnings };
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

/** Waits until no tinybvh build runs (a removed mesh's running build ends by itself, after `whenBuilt`). */
async function tinybvhIdle(rays: PlatformRayQueries): Promise<void> {
    while (rays.meshTreeCounts.tinybvhBuilding > 0) await new Promise((resolve) => setTimeout(resolve, 1));
}

/** The trees in and settled: every batch answered, the trees placed and the batches' buffers let go of. */
async function gpuTreesIn(setup: Awaited<ReturnType<typeof setUp>>): Promise<void> {
    setup.frame();
    while (setup.gpuBuilder.waiting > 0) await setup.gpuBuilder.finishBatch();
    await setup.rays.whenBuilt();
    setup.frame();
    setup.frame();
}

describe('mesh trees from the GPU, tinybvh only as the fallback (T2.12, #26)', () => {
    it("gives every mesh the GPU's tree, builds nothing in the worker, and hits what tinybvh's trees hit", async () => {
        const setup = await setUp();
        const { rays, gpuBuilder, frame, gpu } = setup;
        frame();
        // One batch for all meshes, and no tinybvh build beside it.
        expect(gpuBuilder.requests).toHaveLength(1);
        expect(gpuBuilder.requests[0].map((request) => request.slot)).toEqual([0, 1, 2]);
        expect(rays.meshTreeCounts).toMatchObject({ waitingForFirstTree: 3, tinybvhBuilding: 0 });
        expect(rays.building).toBe(true);

        await gpuBuilder.finishBatch();
        await rays.whenBuilt();
        frame();
        expect(rays.building).toBe(false);
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 3, onTinybvhTrees: 0, tinybvhBuilding: 0, gpuTreesBuilt: 3, gpuTreesRefused: 0 });
        const onGpuTrees = hitsDown(rays, 3);
        expect(onGpuTrees.filter((hit) => hit === '-1')).toHaveLength(0);

        // The fallback's trees (tinybvh, another shape) hit the same triangles.
        const tinybvhOnly = await setUp({ meshTrees: { gpuFirst: false } });
        await gpuTreesIn(tinybvhOnly);
        expect(tinybvhOnly.rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 3 });
        expect(hitsDown(tinybvhOnly.rays, 3)).toEqual(onGpuTrees);
        tinybvhOnly.rays.dispose();

        // The batch's buffers go one update after the copy, and nothing replaces the GPU's trees later.
        frame();
        expect(gpuBuilder.outputs[0].isDestroyed).toBe(true);
        expect(rays.memory.gpuBuildOutputs).toBe(0);
        expect(gpu.buffers.filter((buffer) => buffer.label.startsWith('stand-in') && !buffer.destroyed)).toHaveLength(0);
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 3, onTinybvhTrees: 0 });
        rays.dispose();
    });

    it('sizes the node buffer by what fits when GPU trees make it jump, with no headroom', async () => {
        // Meshes big enough that their trees outgrow the starting buffer several times over.
        const setup = await setUp({ cells: 48 });
        await gpuTreesIn(setup);
        expect(setup.rays.meshTreeCounts.onGpuTrees).toBe(3);
        const memory = setup.rays.memory;
        expect(memory.nodes).toBe(memory.nodesInUse);
        setup.rays.dispose();
    });

    it('falls back to tinybvh for a GPU tree the builder refuses, and for trees deeper than gpuMaxDepth', async () => {
        const { rays, gpuBuilder, frame } = await setUp();
        gpuBuilder.refuse.add(1);
        frame();
        // Nothing starts in the worker until the GPU has answered.
        expect(rays.meshTreeCounts.tinybvhBuilding).toBe(0);
        await gpuBuilder.finishBatch();
        expect(rays.meshTreeCounts).toMatchObject({ tinybvhBuilding: 1, gpuTreesRefused: 1, waitingForFirstTree: 1 });
        expect(rays.building).toBe(true);
        await rays.whenBuilt();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 1, gpuTreesRefused: 1, tinybvhBuilding: 0 });
        expect(hitsDown(rays, 3).filter((hit) => hit === '-1')).toHaveLength(0);
        rays.dispose();

        const deep = await setUp({ meshTrees: { gpuMaxDepth: 1 } });
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

    it("falls back to tinybvh for a mesh the GPU can't start (too big) and for a batch that fails, with one warning", async () => {
        const { rays, gpuBuilder, frame, warnings } = await setUp();
        // Refused inside build() itself: the fallback starts in the same update.
        gpuBuilder.tooBig.add(2);
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ tinybvhBuilding: 1, gpuTreesRefused: 1 });
        await gpuBuilder.finishBatch();
        await rays.whenBuilt();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 2, onTinybvhTrees: 1 });
        expect(hitsDown(rays, 3).filter((hit) => hit === '-1')).toHaveLength(0);
        expect(warnings).toEqual([]);
        rays.dispose();

        const failing = await setUp();
        failing.gpuBuilder.failNextBatch = true;
        failing.frame();
        await failing.gpuBuilder.finishBatch();
        expect(failing.rays.meshTreeCounts).toMatchObject({ tinybvhBuilding: 3, gpuTreesRefused: 3 });
        await failing.rays.whenBuilt();
        failing.frame();
        expect(failing.rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 3 });
        expect(failing.warnings).toEqual([expect.stringContaining('fall back to tinybvh')]);
        failing.rays.dispose();
    });

    it('drops a GPU tree whose mesh changed while it was building', async () => {
        const { rays, gpuBuilder, frame, scene, meshes } = await setUp();
        frame();
        // Mesh 2 is lowered 1 m while its GPU build runs: that tree is for an old version.
        const lowered = grid(6, 2);
        for (let at = 2; at < lowered.positions.length; at += 3) lowered.positions[at] = -1;
        scene.updateMesh(meshes[2], lowered);
        frame();
        expect(gpuBuilder.requests).toHaveLength(2);
        await gpuBuilder.finishBatch();
        await gpuBuilder.finishBatch();
        await rays.whenBuilt();
        frame();
        frame();
        // Mesh 2's first build was for its old version: not counted, and its tree never placed.
        expect(rays.meshTreeCounts).toMatchObject({ gpuTreesBuilt: 3, onGpuTrees: 3, onTinybvhTrees: 0, tinybvhBuilding: 0 });
        for (const hit of hitsDown(rays, 3).slice(6)) expect(hit).toMatch(/\/6\.00000$/);
        rays.dispose();
    });
});

describe('runtime mesh trees: meshes added or replaced while the scene runs (T3.2)', () => {
    /** A scene whose trees are all in (the GPU's, or the fallback's with `gpuFirst: false`), with the builders idle. */
    async function settledScene(options: { meshes?: number; meshTrees?: MeshTreeOptions } = {}) {
        const setup = await setUp(options);
        await gpuTreesIn(setup);
        expect(setup.rays.meshTreeCounts).toMatchObject({ waitingForFirstTree: 0, tinybvhBuilding: 0 });
        expect(setup.gpuBuilder.waiting).toBe(0);
        return setup;
    }

    it('builds a mesh added at runtime on the GPU and traces it on that tree, with no tinybvh build', async () => {
        const { rays, gpuBuilder, frame, scene, material, gpu } = await settledScene();
        const nodesBefore = rays.memory.nodesInUse;
        const added = scene.createMesh(grid(9));
        scene.createInstance({ mesh: added, materials: [material], transform: IDENTITY });
        frame();
        // One GPU batch with only the new mesh.
        expect(gpuBuilder.requests).toHaveLength(2);
        expect(gpuBuilder.requests[1].map((request) => request.slot)).toEqual([3]);
        expect(rays.building).toBe(true);
        expect(rays.meshTreeCounts.tinybvhBuilding).toBe(0);

        await gpuBuilder.finishBatch();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 4, onTinybvhTrees: 0, waitingForFirstTree: 0, tinybvhBuilding: 0 });
        expect(rays.building).toBe(false);
        expect(hitsDown(rays, 4).filter((hit) => hit === '-1')).toHaveLength(0);
        expect(rays.memory.nodesInUse).toBeGreaterThan(nodesBefore);
        // The batch's own buffers go one update after the copy.
        frame();
        expect(rays.memory.gpuBuildOutputs).toBe(0);
        expect(gpu.buffers.filter((buffer) => buffer.label.startsWith('stand-in') && !buffer.destroyed)).toHaveLength(0);
        rays.dispose();
    });

    it('gives a replaced mesh (updateMesh on an existing instance) a GPU tree of its new shape, while rays see the old tree until then', async () => {
        const { rays, gpuBuilder, frame, scene, meshes } = await settledScene();
        const before = hitsDown(rays, 3);
        // Mesh 1 becomes a bigger, finer square, 1 m lower: rays straight down hit it further away.
        const lowered = grid(3, 1.5, 6);
        for (let at = 2; at < lowered.positions.length; at += 3) lowered.positions[at] = -1;
        scene.updateMesh(meshes[1], lowered);
        frame();
        expect(gpuBuilder.requests[1].map((request) => request.slot)).toEqual([1]);
        // Until its new tree is in, rays see the old one.
        expect(rays.building).toBe(true);
        expect(hitsDown(rays, 3)).toEqual(before);

        await gpuBuilder.finishBatch();
        frame();
        expect(rays.building).toBe(false);
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 3, onTinybvhTrees: 0 });
        const after = hitsDown(rays, 3);
        expect(after.slice(0, 3)).toEqual(before.slice(0, 3));
        expect(after.slice(6)).toEqual(before.slice(6));
        for (const hit of after.slice(3, 6)) expect(hit).toMatch(/\/6\.00000$/);
        rays.dispose();
    });

    it('keeps one fallback build per mesh: a mesh replaced every frame queues only its newest version', async () => {
        const { rays, gpuBuilder, frame, scene, meshes } = await settledScene({ meshTrees: { gpuFirst: false } });
        // Five new shapes in five frames, with no await between them, so the first build is still running.
        let last = grid(3);
        for (let version = 1; version <= 5; version++) {
            last = grid(3, 1 + version * 0.1);
            for (let at = 2; at < last.positions.length; at += 3) last.positions[at] = -version * 0.25;
            scene.updateMesh(meshes[1], last);
            frame();
            // Running plus at most one waiting, however many uploads came.
            expect(rays.meshTreeCounts.tinybvhBuilding).toBeLessThanOrEqual(2);
        }
        expect(gpuBuilder.requests).toHaveLength(0);
        await rays.whenBuilt();
        await tinybvhIdle(rays);
        frame();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 3, tinybvhBuilding: 0 });
        // The last shape is what rays see: 1.25 m below the plane the rays start 5 m above.
        for (const hit of hitsDown(rays, 3).slice(3, 6)) expect(hit).toMatch(/\/6\.25000$/);
        rays.dispose();
    });

    it('drops the waiting fallback build of a mesh removed meanwhile', async () => {
        const { rays, frame, scene, meshes, instances } = await settledScene({ meshTrees: { gpuFirst: false } });
        scene.updateMesh(meshes[2], grid(6, 1.2));
        frame();
        scene.updateMesh(meshes[2], grid(6, 1.4));
        frame();
        expect(rays.meshTreeCounts.tinybvhBuilding).toBe(2);
        scene.destroyInstance(instances[2]);
        scene.destroyMesh(meshes[2]);
        frame();
        // Only the running build is left, and it ends by itself.
        expect(rays.meshTreeCounts.tinybvhBuilding).toBe(1);
        await tinybvhIdle(rays);
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onTinybvhTrees: 2, waitingForFirstTree: 0, tinybvhBuilding: 0 });
        rays.dispose();
    });

    it('uses tinybvh alone without subgroups, when the GPU builder fails to compile, or with gpuFirst off', async () => {
        const gpu = createGpu();
        const scene = new PlatformScene(gpu.device);
        // No stand-in builder: the device's features decide, and it has no subgroups.
        const rays = new PlatformRayQueries(gpu.device, scene, { buildIn: 'main' });
        expect(rays.gpuFirstTrees).toBe(false);
        await rays.prepare(new PlatformKernelBuilder(gpu.device, 'test'), { batches: false });
        const material = scene.createMaterial({});
        const firstMesh = scene.createMesh(grid(0));
        scene.createInstance({ mesh: firstMesh, materials: [material], transform: IDENTITY });
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

        // At runtime too (T3.2): a mesh added and a mesh replaced while the scene runs get tinybvh trees.
        const added = scene.createMesh(grid(3));
        scene.createInstance({ mesh: added, materials: [material], transform: IDENTITY });
        const reshaped = grid(0, 1.5);
        for (let at = 2; at < reshaped.positions.length; at += 3) reshaped.positions[at] = -1;
        scene.updateMesh(firstMesh, reshaped);
        frame();
        expect(rays.building).toBe(true);
        await rays.whenBuilt();
        frame();
        expect(rays.meshTreeCounts).toMatchObject({ onGpuTrees: 0, onTinybvhTrees: 2, gpuTreesBuilt: 0, waitingForFirstTree: 0 });
        const runtimeHits = hitsDown(rays, 2);
        for (const hit of runtimeHits.slice(0, 3)) expect(hit).toMatch(/\/6\.00000$/);
        for (const hit of runtimeHits.slice(3)) expect(hit).toMatch(/\/5\.00000$/);
        rays.dispose();

        // A GPU builder whose kernels fail to compile: tinybvh alone, with one warning, and prepare still succeeds.
        const broken = createGpu();
        const brokenScene = new PlatformScene(broken.device);
        const warnings: string[] = [];
        const failing: FirstTreeBuilder = { prepare: () => Promise.reject(new Error('no compile')), build: () => ({ notBuilt: [], batches: [] }) };
        const brokenRays = new PlatformRayQueries(broken.device, brokenScene, { buildIn: 'main', warn: (message) => warnings.push(message), meshTrees: { firstTreeBuilder: failing } });
        await brokenRays.prepare(new PlatformKernelBuilder(broken.device, 'test'), { batches: false });
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
