/**
 * Our tracer's side of the benchmark: build the trees, time the top-level
 * rebuild, upload, and time the batch kernels.
 */

import {
    assembleTraceScene,
    buildTopLevel,
    DEFAULT_SAH_OPTIONS,
    HIT_RECORD_BYTES,
    RAY_FLOATS,
    TraceBatchKernel,
    uploadTraceScene,
    type MeshBvh,
    type MeshBvhBuilder,
    type MeshBvhBuilderKind,
    type TraceInstance,
    type TraceSceneData,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { TOP_LEVEL_REBUILD_RUNS, type TracerId } from './config.ts';
import { measureDispatches, type DispatchTimer } from './dispatch-timer.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import type { ModeResult, OursBuildResult, TmbuildBuildResult, TraceMode } from './results.ts';
import { buildMeshBvhWithTmbvh } from './tmbvh-convert.ts';
import { maximum, median, megaRaysPerSecond, sum } from './statistics.ts';

export interface OursBuild {
    bvhs: MeshBvh[];
    data: TraceSceneData;
    traceInstances: TraceInstance[];
    result: OursBuildResult;
}

/** Lets the page breathe so the runner's polling never sees one 30-second block. */
export function yieldToBrowser(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Builds every mesh's tree in the worker, one mesh after another, so each
 * mesh's time is its own.
 */
export async function buildOursScene(
    builder: MeshBvhBuilder,
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
    kind: MeshBvhBuilderKind = 'sah',
): Promise<OursBuild> {
    const perMeshMs: number[] = [];
    const bvhs: MeshBvh[] = [];
    for (const mesh of meshes) {
        const start = performance.now();
        bvhs.push(await builder.build({ positions: mesh.positions, indices: mesh.indices }, { builder: kind }));
        perMeshMs.push(performance.now() - start);
    }
    return assembleOursScene(bvhs, perMeshMs, instances, builder.usesWorker);
}

export interface TmbuildScene extends OursBuild {
    result: TmbuildBuildResult;
}

/**
 * The `ours-tmbuild` trees: three-mesh-bvh builds each mesh (SAH, on this
 * thread, where that library builds), and tmbvh-convert.ts turns the tree into
 * our layout. Yields between meshes.
 */
export async function buildTmbuildScene(
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
    targetLeafSize: number,
): Promise<TmbuildScene> {
    const perMeshMs: number[] = [];
    const bvhs: MeshBvh[] = [];
    let tmbvhBuildMsTotal = 0;
    let convertMsTotal = 0;
    for (const mesh of meshes) {
        const built = buildMeshBvhWithTmbvh({ positions: mesh.positions, indices: mesh.indices }, targetLeafSize);
        bvhs.push(built.bvh);
        perMeshMs.push(built.tmbvhMs + built.convertMs);
        tmbvhBuildMsTotal += built.tmbvhMs;
        convertMsTotal += built.convertMs;
        await yieldToBrowser();
    }
    const scene = await assembleOursScene(bvhs, perMeshMs, instances, false);
    return { ...scene, result: { ...scene.result, tmbvhBuildMsTotal, convertMsTotal, targetLeafSize } };
}

async function assembleOursScene(
    bvhs: MeshBvh[],
    perMeshMs: number[],
    instances: readonly BenchInstance[],
    usesWorker: boolean,
): Promise<OursBuild> {
    const assembleStart = performance.now();
    const data = assembleTraceScene(
        bvhs,
        instances.map((instance, index) => ({ meshIndex: instance.meshIndex, instanceIndex: index, objectToWorld: instance.objectToWorld })),
    );
    const assembleMs = performance.now() - assembleStart;

    const traceInstances: TraceInstance[] = instances.map((instance, index) => ({
        instanceIndex: index,
        nodeOffset: data.meshes[instance.meshIndex].nodeOffset,
        triangleOffset: data.meshes[instance.meshIndex].triangleOffset,
        meshBounds: bvhs[instance.meshIndex].bounds,
        objectToWorld: instance.objectToWorld,
    }));
    const rebuildMs = await timeTopLevelRebuilds(traceInstances);

    let maxMeshIndex = 0;
    perMeshMs.forEach((ms, index) => {
        if (ms > perMeshMs[maxMeshIndex]) maxMeshIndex = index;
    });
    const result: OursBuildResult = {
        usesWorker,
        meshCount: bvhs.length,
        triangles: sum(bvhs.map((bvh) => bvh.triangleCount)),
        buildMsTotal: sum(perMeshMs),
        buildMsMaxMesh: maximum(perMeshMs),
        maxMeshTriangles: bvhs[maxMeshIndex]?.triangleCount ?? 0,
        assembleMs,
        topLevelRebuildMsMedian: median(rebuildMs),
        topLevelRebuildMsMax: maximum(rebuildMs),
        topLevelRebuildRuns: rebuildMs.length,
        topLevelInstances: data.topLevel.instanceCount,
        topLevelNodes: data.topLevel.nodeCount,
        nodeCount: sum(bvhs.map((bvh) => bvh.nodeCount)),
        maxMeshDepth: data.maxMeshDepth,
        requiredStackSize: data.requiredStackSize,
        leafSize: DEFAULT_SAH_OPTIONS.maxLeafPrimitives,
        bytesNodes: data.nodes.byteLength,
        bytesTriangles: data.triangles.byteLength,
        bytesTopLevel: data.topLevel.buffer.byteLength,
        bytesTotal: data.byteLength,
    };
    return { bvhs, data, traceInstances, result };
}

/** The top level is rebuilt on the CPU whenever a transform changes; this is what one rebuild costs. */
export async function timeTopLevelRebuilds(instances: readonly TraceInstance[], runs = TOP_LEVEL_REBUILD_RUNS): Promise<number[]> {
    const times: number[] = [];
    for (let run = 0; run < runs; run++) {
        const start = performance.now();
        buildTopLevel(instances);
        times.push(performance.now() - start);
        if (run % 4 === 3) await yieldToBrowser();
    }
    return times;
}

/** GPU memory the benchmark owns while a tracer runs; freed together. */
class Resources {
    private readonly owned: { destroy(): void }[] = [];

    add<T extends { destroy(): void }>(resource: T): T {
        this.owned.push(resource);
        return resource;
    }

    destroyAll(): void {
        for (const resource of this.owned.splice(0)) resource.destroy();
    }
}

export async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({
        label: 'trace-bench readback',
        size: byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    try {
        const encoder = device.createCommandEncoder({ label: 'trace-bench readback' });
        encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const copy = readback.getMappedRange().slice(0);
        readback.unmap();
        return copy;
    } finally {
        readback.destroy();
    }
}

/** Ray and result buffers shared by the tracers of one scene. */
export interface OursRayBuffers {
    closestRays: GPUBuffer;
    closestCount: number;
    /** Null when no hit faced the sun. */
    shadowRays: GPUBuffer | null;
    shadowCount: number;
}

export interface OursTracerRun {
    modes: ModeResult[];
    /** Closest-hit records (`GiHitRecord`) of the last closest dispatch. */
    hitRecords: ArrayBuffer;
    /** Closest `t` per ray, negative for a miss. */
    closestT: Float32Array;
    /** Blocked flags (1 = blocked) per shadow ray, or null when none were traced. */
    blocked: Uint32Array | null;
}

export interface TimingPlan {
    warmups: number;
    timed: number;
}

function summarizeMode(tracer: TracerId, mode: TraceMode, rays: number, timer: DispatchTimer, timing: Awaited<ReturnType<typeof measureDispatches>>): ModeResult {
    return { tracer, mode, rays, timing, timingSource: timer.source, mraysPerSecond: megaRaysPerSecond(rays, timing.medianMs) };
}

/** Pulls `t` out of an array of `GiHitRecord`s. */
export function extractHitDistances(hitRecords: ArrayBuffer): Float32Array {
    const recordWords = HIT_RECORD_BYTES / 4;
    const floats = new Float32Array(hitRecords);
    const distances = new Float32Array(floats.length / recordWords);
    for (let ray = 0; ray < distances.length; ray++) distances[ray] = floats[ray * recordWords];
    return distances;
}

/**
 * Uploads `data`, times the closest-hit and any-hit kernels on the shared
 * rays, reads the answers back, and frees everything it made on the GPU.
 */
export async function runOursTracer(
    device: GPUDevice,
    timer: DispatchTimer,
    tracer: 'ours-wt' | 'ours-tinybvh' | 'ours-tmbuild',
    data: TraceSceneData,
    rays: OursRayBuffers,
    plan: TimingPlan,
): Promise<OursTracerRun> {
    const resources = new Resources();
    try {
        const kernel = await TraceBatchKernel.create(device, tracer);
        const scene = resources.add(uploadTraceScene(device, data, tracer));

        const hits = resources.add(
            device.createBuffer({
                label: `${tracer} hits`,
                size: rays.closestCount * HIT_RECORD_BYTES,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            }),
        );
        const modes: ModeResult[] = [];

        const closestTiming = await measureDispatches(
            () =>
                timer.time((encoder, timestampWrites) =>
                    kernel.encode(encoder, scene, { mode: 'closest', rays: rays.closestRays, results: hits, count: rays.closestCount, timestampWrites }),
                ),
            plan.warmups,
            plan.timed,
        );
        modes.push(summarizeMode(tracer, 'closest', rays.closestCount, timer, closestTiming));
        const hitRecords = await readBuffer(device, hits, rays.closestCount * HIT_RECORD_BYTES);

        let blocked: Uint32Array | null = null;
        if (rays.shadowRays && rays.shadowCount > 0) {
            const shadowRays = rays.shadowRays;
            const occlusion = resources.add(
                device.createBuffer({
                    label: `${tracer} occlusion`,
                    size: rays.shadowCount * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                }),
            );
            const anyTiming = await measureDispatches(
                () =>
                    timer.time((encoder, timestampWrites) =>
                        kernel.encode(encoder, scene, { mode: 'any', rays: shadowRays, results: occlusion, count: rays.shadowCount, timestampWrites }),
                    ),
                plan.warmups,
                plan.timed,
            );
            modes.push(summarizeMode(tracer, 'any', rays.shadowCount, timer, anyTiming));
            blocked = new Uint32Array(await readBuffer(device, occlusion, rays.shadowCount * 4));
        }
        return { modes, hitRecords, closestT: extractHitDistances(hitRecords), blocked };
    } finally {
        resources.destroyAll();
    }
}

/** Creates a storage buffer from packed rays. The caller owns (and destroys) it. */
export function uploadRays(device: GPUDevice, label: string, packed: Float32Array): GPUBuffer {
    const buffer = device.createBuffer({ label, size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, packed as Float32Array<ArrayBuffer>);
    return buffer;
}

/**
 * Times a small batch of rays on our tracer and returns GPU
 * nanoseconds per ray. Launch overhead is included, so for a small sample this
 * overestimates a little, which is the safe direction for a time limit.
 */
export async function estimateNsPerRay(device: GPUDevice, timer: DispatchTimer, data: TraceSceneData, sampleRays: Float32Array): Promise<number> {
    const resources = new Resources();
    try {
        const count = sampleRays.length / RAY_FLOATS;
        const kernel = await TraceBatchKernel.create(device, 'ours-wt calibrate');
        const scene = resources.add(uploadTraceScene(device, data, 'ours-wt calibrate'));
        const rays = resources.add(uploadRays(device, 'calibration rays', sampleRays));
        const hits = resources.add(
            device.createBuffer({ label: 'calibration hits', size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE }),
        );
        const timing = await measureDispatches(
            () =>
                timer.time((encoder, timestampWrites) =>
                    kernel.encode(encoder, scene, { mode: 'closest', rays, results: hits, count, timestampWrites }),
                ),
            1,
            3,
        );
        return (timing.medianMs * 1e6) / count;
    } finally {
        resources.destroyAll();
    }
}

/**
 * One untimed closest-hit dispatch on our scene, for building the
 * shadow rays. Returns the hit records.
 */
export async function traceClosestOnce(
    device: GPUDevice,
    data: TraceSceneData,
    closestRays: GPUBuffer,
    closestCount: number,
): Promise<ArrayBuffer> {
    const resources = new Resources();
    try {
        const kernel = await TraceBatchKernel.create(device, 'ours-wt prepare');
        const scene = resources.add(uploadTraceScene(device, data, 'ours-wt prepare'));
        const hits = resources.add(
            device.createBuffer({
                label: 'prepare hits',
                size: closestCount * HIT_RECORD_BYTES,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            }),
        );
        const encoder = device.createCommandEncoder({ label: 'prepare closest' });
        kernel.encode(encoder, scene, { mode: 'closest', rays: closestRays, results: hits, count: closestCount });
        device.queue.submit([encoder.finish()]);
        return await readBuffer(device, hits, closestCount * HIT_RECORD_BYTES);
    } finally {
        resources.destroyAll();
    }
}

