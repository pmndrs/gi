/**
 * three-mesh-bvh's side of the benchmark (v0.9.18, `three-mesh-bvh/webgpu`).
 *
 * That library only runs inside three's node system, so this goes through a
 * `WebGPURenderer` that shares our device:
 *
 * - each mesh gets a `MeshBVH` built with `{ strategy: SAH }` (its default is
 *   CENTER, which would make it look slower than it is) and the library's own
 *   default leaf size unless `?tmLeaf=` says otherwise;
 * - `BVHComputeData` packs the trees, builds its clustered top level, and
 *   provides the WGSL functions `bvh_RaycastFirstHit` / `bvh_RaycastAnyHit`;
 * - a TSL compute kernel reads the same rays we give our tracer (copied once
 *   into a three storage attribute, because three has no way to bind a
 *   foreign buffer), traces them, and writes one `vec4f` per ray.
 *
 * Timing: with `trackTimestamp`, three wraps each `renderer.compute()` pass in
 * its own timestamp pair. Resolving after every dispatch
 * (`resolveTimestampsAsync('compute')`) gives that one pass's GPU time. Three's
 * own overhead (bind-group work, the first buffer upload) happens outside the
 * pass and is not counted; the warm-up dispatches absorb the uploads.
 */

import { BufferAttribute, BufferGeometry, Mesh, MeshBasicMaterial } from 'three';
import { Fn, If, instanceIndex, storage, wgslFn } from 'three/tsl';
import { StorageBufferAttribute, WebGPURenderer } from 'three/webgpu';
import { MeshBVH, SAH } from 'three-mesh-bvh';
import { BVHComputeData } from 'three-mesh-bvh/webgpu';
import { measureDispatches } from './dispatch-timer.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import { yieldToBrowser, type TimingPlan } from './ours-bench.ts';
import type { ModeResult, TimingSource, TmbvhBuildResult, TraceMode } from './results.ts';
import { maximum, megaRaysPerSecond, sum } from './statistics.ts';

/** Floats per ray: two `vec4f` (origin + tMin, direction + tMax). */
const VEC4_PER_RAY = 2;

export const TMBVH_TIMING_METHOD =
    'three trackTimestamp: GPU timestamps at the start and end of each renderer.compute() pass, resolved after every dispatch';

export interface TmbvhRayInput {
    closestRays: Float32Array;
    closestCount: number;
    shadowRays: Float32Array | null;
    shadowCount: number;
}

export interface TmbvhRun {
    build: TmbvhBuildResult;
    modes: ModeResult[];
    closestT: Float32Array;
    blocked: Uint32Array | null;
}

/** A shared renderer for the page: three on top of our device, with timestamps if the device has them. */
export async function createTmbvhRenderer(device: GPUDevice): Promise<WebGPURenderer> {
    const renderer = new WebGPURenderer({ device, trackTimestamp: true });
    await renderer.init();
    return renderer;
}

function storageBytes(node: unknown): number {
    const attribute = (node as { proxyNode: { value: { array: { byteLength: number } } } }).proxyNode.value;
    return attribute.array.byteLength;
}

interface BuiltTrees {
    geometries: BufferGeometry[];
    perMeshMs: number[];
}

/** Builds each mesh's SAH tree on the main thread (that is where the library builds), yielding between meshes. */
async function buildTrees(meshes: readonly BenchMeshGeometry[], leafSize: number): Promise<BuiltTrees> {
    const geometries: BufferGeometry[] = [];
    const perMeshMs: number[] = [];
    for (const mesh of meshes) {
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(mesh.positions, 3));
        // MeshBVH reorders the index in place, so it gets its own copy; our tracer's indices must stay as they are.
        geometry.setIndex(new BufferAttribute(mesh.indices.slice(), 1));
        const start = performance.now();
        (geometry as unknown as { boundsTree: MeshBVH }).boundsTree = new MeshBVH(geometry, {
            strategy: SAH,
            targetLeafSize: leafSize,
            verbose: false,
        });
        perMeshMs.push(performance.now() - start);
        geometries.push(geometry);
        await yieldToBrowser();
    }
    return { geometries, perMeshMs };
}

interface Kernel {
    mode: TraceMode;
    rays: number;
    node: ReturnType<typeof makeKernelNode>;
    results: StorageBufferAttribute;
    input: StorageBufferAttribute;
}

/**
 * One thread per ray: read the ray, call the library's WGSL function, write
 * `vec4f(result, 0, 0, 0)`. For closest-hit the result is the hit distance
 * (negative for a miss); for any-hit it is 1 when blocked.
 */
function makeKernelNode(bvhData: BVHComputeData, mode: TraceMode, input: StorageBufferAttribute, results: StorageBufferAttribute, rayCount: number) {
    const functions = bvhData.fns as unknown as Record<string, unknown>;
    const trace =
        mode === 'closest'
            ? wgslFn(
                  /* wgsl */ `
                  fn traceClosest( origin: vec3f, direction: vec3f, maxDist: f32 ) -> vec4f {
                      var ray: Ray;
                      ray.origin = origin;
                      ray.direction = direction;
                      ray.maxDist = maxDist;
                      var hit: IntersectionResult;
                      bvh_RaycastFirstHit( ray, &hit );
                      return vec4f( select( -1.0, hit.dist, hit.didHit ), 0.0, 0.0, 0.0 );
                  }
              `,
                  [functions.raycastFirstHit as never],
              )
            : wgslFn(
                  /* wgsl */ `
                  fn traceAny( origin: vec3f, direction: vec3f, maxDist: f32 ) -> vec4f {
                      var ray: Ray;
                      ray.origin = origin;
                      ray.direction = direction;
                      ray.maxDist = maxDist;
                      return vec4f( select( 0.0, 1.0, bvh_RaycastAnyHit( ray ) ), 0.0, 0.0, 0.0 );
                  }
              `,
                  [functions.raycastAnyHit as never],
              );

    const rays = storage(input, 'vec4', rayCount * VEC4_PER_RAY).toReadOnly();
    const output = storage(results, 'vec4', rayCount);
    return Fn(() => {
        If(instanceIndex.lessThan(rayCount), () => {
            const originAndMin = rays.element(instanceIndex.mul(VEC4_PER_RAY));
            const directionAndMax = rays.element(instanceIndex.mul(VEC4_PER_RAY).add(1));
            output.element(instanceIndex).assign(trace({ origin: originAndMin.xyz, direction: directionAndMax.xyz, maxDist: directionAndMax.w }));
        });
    })().compute(rayCount, [64]);
}

function makeKernel(bvhData: BVHComputeData, mode: TraceMode, packedRays: Float32Array, rayCount: number): Kernel {
    const input = new StorageBufferAttribute(packedRays, 4);
    const results = new StorageBufferAttribute(rayCount, 4);
    return { mode, rays: rayCount, node: makeKernelNode(bvhData, mode, input, results, rayCount), results, input };
}

async function timeKernel(renderer: WebGPURenderer, kernel: Kernel, plan: TimingPlan, source: TimingSource): Promise<ModeResult> {
    // The first dispatch compiles the pipeline and uploads the buffers.
    await renderer.computeAsync(kernel.node);
    if (source === 'timestamp-query') await renderer.resolveTimestampsAsync('compute');

    const timeOnce = async (): Promise<number> => {
        if (source === 'timestamp-query') {
            renderer.compute(kernel.node);
            const milliseconds = await renderer.resolveTimestampsAsync('compute');
            if (typeof milliseconds !== 'number') throw new Error('three returned no compute timestamp');
            return milliseconds;
        }
        const start = performance.now();
        await renderer.computeAsync(kernel.node);
        return performance.now() - start;
    };
    const timing = await measureDispatches(timeOnce, plan.warmups, plan.timed);
    // A stale value would repeat exactly; real timestamps jitter.
    if (timing.samplesMs.length > 1 && new Set(timing.samplesMs).size === 1) {
        throw new Error('three-mesh-bvh timings are all identical, so the timestamps look stale');
    }
    return {
        tracer: 'tmbvh',
        mode: kernel.mode,
        rays: kernel.rays,
        timing,
        timingSource: source,
        mraysPerSecond: megaRaysPerSecond(kernel.rays, timing.medianMs),
    };
}

async function readResults(renderer: WebGPURenderer, kernel: Kernel): Promise<Float32Array> {
    const bytes = await renderer.getArrayBufferAsync(kernel.results);
    const vec4s = new Float32Array(bytes);
    const values = new Float32Array(kernel.rays);
    for (let ray = 0; ray < kernel.rays; ray++) values[ray] = vec4s[ray * 4];
    return values;
}

/** Builds the library's structures for a scene, traces the shared rays, and releases everything. */
export async function runTmbvh(
    renderer: WebGPURenderer,
    sourceMeshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
    rays: TmbvhRayInput,
    plan: TimingPlan,
    leafSize: number,
): Promise<TmbvhRun> {
    const timestamps = renderer.hasFeature('timestamp-query');
    const source: TimingSource = timestamps ? 'timestamp-query' : 'wall-clock';

    const { geometries, perMeshMs } = await buildTrees(sourceMeshes, leafSize);
    const sharedMaterial = new MeshBasicMaterial();
    const objects = instances.map((instance) => {
        const mesh = new Mesh(geometries[instance.meshIndex], sharedMaterial);
        mesh.matrixAutoUpdate = false;
        mesh.matrix.fromArray(instance.objectToWorld);
        mesh.matrixWorld.copy(mesh.matrix);
        return mesh;
    });

    // autogenerateBvh is off so a mesh without our SAH tree fails loudly instead of getting a CENTER tree.
    const bvhData = new BVHComputeData(objects, { attributes: { position: 'vec4f' }, autogenerateBvh: false });
    await yieldToBrowser();
    const updateStart = performance.now();
    bvhData.update();
    const dataUpdateMs = performance.now() - updateStart;

    const storageNodes = bvhData.storage as unknown as Record<'nodes' | 'index' | 'attributes' | 'transforms', unknown>;
    const bytesNodes = storageBytes(storageNodes.nodes);
    const bytesIndex = storageBytes(storageNodes.index);
    const bytesAttributes = storageBytes(storageNodes.attributes);
    const bytesTransforms = storageBytes(storageNodes.transforms);
    const build: TmbvhBuildResult = {
        meshCount: sourceMeshes.length,
        buildMsTotal: sum(perMeshMs),
        buildMsMaxMesh: maximum(perMeshMs),
        dataUpdateMs,
        leafSize,
        strategy: 'SAH',
        bytesNodes,
        bytesIndex,
        bytesAttributes,
        bytesTransforms,
        bytesTotal: bytesNodes + bytesIndex + bytesAttributes + bytesTransforms,
    };

    const kernels: Kernel[] = [];
    try {
        const modes: ModeResult[] = [];
        const closest = makeKernel(bvhData, 'closest', rays.closestRays, rays.closestCount);
        kernels.push(closest);
        modes.push(await timeKernel(renderer, closest, plan, source));
        const closestT = await readResults(renderer, closest);

        let blocked: Uint32Array | null = null;
        if (rays.shadowRays && rays.shadowCount > 0) {
            const any = makeKernel(bvhData, 'any', rays.shadowRays, rays.shadowCount);
            kernels.push(any);
            modes.push(await timeKernel(renderer, any, plan, source));
            blocked = Uint32Array.from(await readResults(renderer, any), (value) => (value > 0.5 ? 1 : 0));
        }
        return { build, modes, closestT, blocked };
    } finally {
        for (const kernel of kernels) {
            kernel.node.dispose();
            kernel.input.dispose();
            kernel.results.dispose();
        }
        (bvhData as unknown as { dispose(): void }).dispose();
        for (const geometry of geometries) geometry.dispose();
        sharedMaterial.dispose();
    }
}
