/**
 * Our tracer's side of the benchmark: build the trees, time the top-level
 * rebuild, upload, and time the batch kernels.
 */

import {
    assembleTraceScene,
    buildTopLevel,
    MAX_LEAF_PRIMITIVES,
    HIT_RECORD_BYTES,
    RAY_FLOATS,
    TraceBatchKernel,
    uploadTraceScene,
    type MeshBvh,
    type MeshBvhBuilder,
    type TraceInstance,
    type TraceSceneData,
} from '../../../../packages/core/src/platform/trace/index.ts';
import { GpuHplocBuilder } from '@pmndrs/bvh';
import { GpuLbvhBuilder, type GpuMeshGeometry } from '@pmndrs/bvh';
import { treeSahCost } from '@pmndrs/bvh';
import { createTinyBvhBuilder, type TinyBvhBuilder } from '@pmndrs/bvh';
import { GPU_BUILD_LOADED_MS, GPU_BUILD_LOADED_RUNS, GPU_BUILD_TIMED_RUNS, RC_LOADED_DISPATCHES, TOP_LEVEL_REBUILD_RUNS, type TracerId } from './config.ts';
import { measureDispatches, type DispatchTimer } from './dispatch-timer.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import type { GpuBuildResult, LoadedTiming, ModeResult, OursBuildResult, TmbuildBuildResult, TraceMode } from './results.ts';
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
 * Builds every mesh's tree in the worker (tinybvh, the only mesh builder),
 * one mesh after another, so each mesh's time is its own.
 */
export async function buildOursScene(
    builder: MeshBvhBuilder,
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
): Promise<OursBuild> {
    const perMeshMs: number[] = [];
    const bvhs: MeshBvh[] = [];
    for (const mesh of meshes) {
        const start = performance.now();
        bvhs.push(await builder.build({ positions: mesh.positions, indices: mesh.indices }));
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

export interface GpuBuiltScene extends OursBuild {
    result: GpuBuildResult;
}

/** The GPU builders behind the Builder contract, as the bench uses them. */
export type GpuBuilderKind = 'lbvh' | 'hploc';

function createGpuBuilder(device: GPUDevice, kind: GpuBuilderKind): GpuLbvhBuilder | GpuHplocBuilder {
    return kind === 'lbvh' ? new GpuLbvhBuilder(device, { label: 'trace-bench lbvh' }) : new GpuHplocBuilder(device, { label: 'trace-bench hploc' });
}

/** A tree's SAH cost in absolute terms (relative cost times its box's half-area), so meshes can be summed. */
function absoluteSahCost(bvh: MeshBvh): number {
    const size = [bvh.bounds[3] - bvh.bounds[0], bvh.bounds[4] - bvh.bounds[1], bvh.bounds[5] - bvh.bounds[2]];
    return treeSahCost(bvh) * (size[0] * size[1] + size[1] * size[2] + size[2] * size[0]);
}

/**
 * The `ours-lbvh` (B1) or `ours-hploc` (B2) trees: every mesh built on the
 * GPU in one batch. The batch is recorded once and built
 * `1 + GPU_BUILD_TIMED_RUNS` times through the dispatch timer (one compute
 * pass, so the timestamps cover exactly the build); the first is the warm-up.
 * The trees are then read back and assembled like the others, and their SAH
 * cost is compared with tinybvh's trees of the same meshes (`reference`).
 */
export async function buildGpuScene(
    device: GPUDevice,
    timer: DispatchTimer,
    kind: GpuBuilderKind,
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
    reference: readonly MeshBvh[],
): Promise<GpuBuiltScene> {
    const builder = createGpuBuilder(device, kind);
    await builder.prepare();
    const geometries = meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices }));

    const wallStart = performance.now();
    const batch = builder.createBatch(geometries);
    const gpuMs: number[] = [];
    let loadedMs: number[] = [];
    let loadedSeconds = 0;
    let wallBuildMs = 0;
    let builds;
    try {
        for (let run = 0; run <= GPU_BUILD_TIMED_RUNS; run++) {
            const ms = await timer.time((encoder, timestampWrites) => batch.encode(encoder, timestampWrites));
            if (run === 0) wallBuildMs = performance.now() - wallStart;
            else gpuMs.push(ms);
        }
        const loadedRuns = Math.min(GPU_BUILD_LOADED_RUNS.max, Math.max(GPU_BUILD_LOADED_RUNS.min, Math.round(GPU_BUILD_LOADED_MS / Math.max(1, median(gpuMs)))));
        const loadedStart = performance.now();
        loadedMs = await timePassesBackToBack(device, (encoder, timestampWrites) => batch.encode(encoder, timestampWrites), loadedRuns);
        loadedSeconds = (performance.now() - loadedStart) / 1000;
        const statsStart = performance.now();
        builds = await batch.finish();
        wallBuildMs += performance.now() - statsStart;
    } finally {
        batch.destroy();
    }

    const readbackStart = performance.now();
    const layouts = await builder.readLayouts(builds);
    builds.destroy();
    const readbackMs = performance.now() - readbackStart;
    const bvhs: MeshBvh[] = [];
    let meshesFallenBack = 0;
    let tinyBvh: TinyBvhBuilder | null = null;
    for (const [index, layout] of layouts.entries()) {
        if (layout) {
            bvhs.push(layout);
            continue;
        }
        meshesFallenBack++;
        tinyBvh ??= await createTinyBvhBuilder();
        bvhs.push(tinyBvh.buildLayout(geometries[index]).layout);
    }

    const scene = await assembleOursScene(bvhs, bvhs.map(() => 0), instances, false);
    const sahCostVsTinybvh = sum(bvhs.map(absoluteSahCost)) / sum(reference.map(absoluteSahCost));
    const match = builder instanceof GpuHplocBuilder ? await compareWithTinybvhHploc(builder, geometries) : null;
    return {
        ...scene,
        result: {
            ...scene.result,
            buildMsTotal: wallBuildMs,
            // One batch for every mesh: there is no per-mesh time.
            buildMsMaxMesh: Number.NaN,
            gpuBuildMsMedian: median(gpuMs),
            gpuBuildMsMin: Math.min(...gpuMs),
            gpuBuildMsMax: Math.max(...gpuMs),
            gpuBuildRuns: gpuMs.length,
            gpuBuildTimestampTimed: timer.source === 'timestamp-query' ? 1 : 0,
            gpuBuildLoadedMsMedian: median(loadedMs),
            gpuBuildLoadedMsMin: Math.min(...loadedMs),
            gpuBuildLoadedMsMax: Math.max(...loadedMs),
            gpuBuildLoadedRuns: loadedMs.length,
            gpuBuildLoadedSeconds: loadedSeconds,
            wallBuildMs,
            readbackMs,
            sahCostVsTinybvh,
            meshesFallenBack,
            ...(match ? { sahCostVsTinybvhHploc: match.sahCostRatio, hplocMatchMeshesRefused: match.refused } : {}),
        },
    };
}

/**
 * Times `count` passes (GPU builds, trace dispatches) submitted back to back,
 * without waiting between them, so the GPU stays loaded: each pass is its own
 * compute pass with timestamps, and the passes go in submits of a few (a
 * single submit of seconds of work could trip Windows' GPU timeout). Returns
 * each pass's ms; without timestamp queries, one value, the wall time over
 * the count.
 */
export async function timePassesBackToBack(
    device: GPUDevice,
    encode: (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites) => void,
    count: number,
): Promise<number[]> {
    if (!device.features.has('timestamp-query')) {
        const start = performance.now();
        for (let run = 0; run < count; run++) {
            const encoder = device.createCommandEncoder({ label: 'loaded build' });
            encode(encoder);
            device.queue.submit([encoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();
        return [(performance.now() - start) / count];
    }
    const perSubmit = 8;
    const reads: Promise<number[]>[] = [];
    for (let first = 0; first < count; first += perSubmit) {
        const builds = Math.min(perSubmit, count - first);
        const querySet = device.createQuerySet({ type: 'timestamp', count: builds * 2 });
        const resolve = device.createBuffer({ size: builds * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        const readback = device.createBuffer({ size: builds * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder({ label: 'loaded builds' });
        for (let build = 0; build < builds; build++) encode(encoder, { querySet, beginningOfPassWriteIndex: build * 2, endOfPassWriteIndex: build * 2 + 1 });
        encoder.resolveQuerySet(querySet, 0, builds * 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, readback, 0, builds * 16);
        device.queue.submit([encoder.finish()]);
        reads.push(
            readback.mapAsync(GPUMapMode.READ).then(() => {
                const stamps = new BigUint64Array(readback.getMappedRange().slice(0));
                readback.destroy();
                resolve.destroy();
                querySet.destroy();
                return Array.from({ length: builds }, (_, build) => Number(stamps[build * 2 + 1] - stamps[build * 2]) / 1e6);
            }),
        );
    }
    return (await Promise.all(reads)).flat();
}

/**
 * B2's match: the scene built on the GPU with tinybvh's codes, tie rule and
 * one-triangle leaves (not timed), against tinybvh's scalar `BuildHPLOC` of
 * each mesh, by summed SAH cost. tinybvh runs here on the page, a mesh at a
 * time with a yield between meshes (Sponza: tens of seconds). A mesh the GPU
 * refuses is left out of both sums and counted.
 */
async function compareWithTinybvhHploc(builder: GpuHplocBuilder, geometries: readonly GpuMeshGeometry[]): Promise<{ sahCostRatio: number; refused: number }> {
    const built = await builder.buildMany(geometries, { mortonCodes: 'tinybvh-63', tieBreak: 'tinybvh', maxLeafPrimitives: 1 });
    let layouts;
    try {
        layouts = await builder.readLayouts(built);
    } finally {
        built.destroy();
    }
    const tinyBvh = await createTinyBvhBuilder();
    let ours = 0;
    let theirs = 0;
    let refused = 0;
    for (const [index, layout] of layouts.entries()) {
        if (!layout) {
            refused++;
            continue;
        }
        const geometry = geometries[index] as Parameters<TinyBvhBuilder['buildLayout']>[0];
        ours += absoluteSahCost(layout);
        theirs += absoluteSahCost(tinyBvh.buildLayout(geometry, { builder: 'hploc', maxLeafPrimitives: 1 }).layout);
        await yieldToBrowser();
    }
    return { sahCostRatio: ours / theirs, refused };
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
        leafSize: MAX_LEAF_PRIMITIVES,
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
    /**
     * After the timed dispatches, submit the same dispatch back to back for
     * about this long and time each (`ModeResult.loaded`). 0 or absent: no
     * loaded section (the probe-grid bench's behaviour).
     */
    loadedMs?: number;
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
    tracer: 'ours-tinybvh' | 'ours-tmbuild' | 'ours-lbvh' | 'ours-hploc',
    data: TraceSceneData,
    rays: OursRayBuffers,
    plan: TimingPlan,
): Promise<OursTracerRun> {
    return probeGridRun(await runOursTracerOnSets(device, timer, tracer, data, probeGridSets(rays), plan));
}

/**
 * Uploads `data`, times our batch kernels on each set (see
 * {@link timeRaySets}), and frees everything it made on the GPU.
 */
export async function runOursTracerOnSets(
    device: GPUDevice,
    timer: DispatchTimer,
    tracer: 'ours-tinybvh' | 'ours-tmbuild' | 'ours-lbvh' | 'ours-hploc',
    data: TraceSceneData,
    sets: readonly RaySetBuffer[],
    plan: TimingPlan,
): Promise<RaySetRun[]> {
    const resources = new Resources();
    try {
        const kernel = await TraceBatchKernel.create(device, tracer);
        const scene = resources.add(uploadTraceScene(device, data, tracer));
        return await timeRaySets(device, timer, tracer, sets, plan, (encoder, set, results, timestampWrites) =>
            kernel.encode(encoder, scene, { mode: set.mode, rays: set.rays, results, count: set.count, timestampWrites }),
        );
    } finally {
        resources.destroyAll();
    }
}

/** One batch of rays to time: closest-hit or any-hit. */
export interface RaySetBuffer {
    /** The set's name in the results (`ModeResult.raySet`); absent for the probe-grid bench's two sets. */
    raySet?: string;
    mode: TraceMode;
    rays: GPUBuffer;
    count: number;
}

/**
 * Records one batch dispatch of `set` writing into `results` (hit records for
 * closest, one word per ray for any), passing `timestampWrites` to its pass.
 */
export type EncodeRaySet = (
    encoder: GPUCommandEncoder,
    set: RaySetBuffer,
    results: GPUBuffer,
    timestampWrites: GPUComputePassTimestampWrites | undefined,
) => void;

/** What one tracer gave for one set: its timing, and the answers of its last dispatch. */
export interface RaySetRun {
    mode: ModeResult;
    /** `GiHitRecord`s for closest, one word per ray (1 = blocked) for any. */
    results: ArrayBuffer;
}

/** The probe-grid bench's two sets: closest, then any when there are shadow rays. */
export function probeGridSets(rays: OursRayBuffers): RaySetBuffer[] {
    const sets: RaySetBuffer[] = [{ mode: 'closest', rays: rays.closestRays, count: rays.closestCount }];
    if (rays.shadowRays && rays.shadowCount > 0) sets.push({ mode: 'any', rays: rays.shadowRays, count: rays.shadowCount });
    return sets;
}

/** The probe-grid bench's two set runs as the record its agreement check reads. */
export function probeGridRun(runs: readonly RaySetRun[]): OursTracerRun {
    const closest = runs.find((run) => run.mode.mode === 'closest');
    if (!closest) throw new Error('no closest-hit set was timed');
    const any = runs.find((run) => run.mode.mode === 'any');
    return {
        modes: runs.map((run) => run.mode),
        hitRecords: closest.results,
        closestT: extractHitDistances(closest.results),
        blocked: any ? new Uint32Array(any.results) : null,
    };
}

/**
 * The loaded section: the set's dispatch submitted back to back for about
 * `loadedMs`, each timed on its own, so the GPU runs at its loaded clock
 * (DECISIONS 2026-11-05: pass/fail timing is taken GPU-bound).
 */
async function timeLoaded(
    device: GPUDevice,
    encode: (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites) => void,
    rays: number,
    singleMedianMs: number,
    loadedMs: number,
): Promise<LoadedTiming> {
    const dispatches = Math.min(
        RC_LOADED_DISPATCHES.max,
        Math.max(RC_LOADED_DISPATCHES.min, Math.round(loadedMs / Math.max(singleMedianMs, 1e-3))),
    );
    const start = performance.now();
    const samplesMs = await timePassesBackToBack(device, encode, dispatches);
    const medianMs = median(samplesMs);
    return {
        medianMs,
        minMs: Math.min(...samplesMs),
        maxMs: Math.max(...samplesMs),
        dispatches,
        seconds: (performance.now() - start) / 1000,
        mraysPerSecond: megaRaysPerSecond(rays, medianMs),
    };
}

/**
 * Times each set with one tracer, in order: warm-ups, then the timed
 * dispatches (each submitted and waited for alone, as the bench always has),
 * then the loaded section when the plan asks for one, then one read of the
 * answers. Each set's result buffer is freed before the next set starts.
 */
export async function timeRaySets(
    device: GPUDevice,
    timer: DispatchTimer,
    tracer: TracerId,
    sets: readonly RaySetBuffer[],
    plan: TimingPlan,
    encode: EncodeRaySet,
): Promise<RaySetRun[]> {
    const runs: RaySetRun[] = [];
    for (const set of sets) {
        const resultBytes = set.count * (set.mode === 'closest' ? HIT_RECORD_BYTES : 4);
        const results = device.createBuffer({
            label: `${tracer} ${set.raySet ?? set.mode} results`,
            size: resultBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        try {
            const timing = await measureDispatches(
                () => timer.time((encoder, timestampWrites) => encode(encoder, set, results, timestampWrites)),
                plan.warmups,
                plan.timed,
            );
            const mode: ModeResult = {
                tracer,
                mode: set.mode,
                rays: set.count,
                timing,
                timingSource: timer.source,
                mraysPerSecond: megaRaysPerSecond(set.count, timing.medianMs),
            };
            if (set.raySet !== undefined) mode.raySet = set.raySet;
            const loadedMs = plan.loadedMs ?? 0;
            if (loadedMs > 0) {
                mode.loaded = await timeLoaded(
                    device,
                    (encoder, timestampWrites) => encode(encoder, set, results, timestampWrites),
                    set.count,
                    timing.medianMs,
                    loadedMs,
                );
            }
            runs.push({ mode, results: await readBuffer(device, results, resultBytes) });
        } finally {
            results.destroy();
        }
    }
    return runs;
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
        const kernel = await TraceBatchKernel.create(device, 'ours calibrate');
        const scene = resources.add(uploadTraceScene(device, data, 'ours calibrate'));
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

/** Our scene on the GPU for untimed dispatches whose answers make the next rays. */
export interface OursPrepareTracer {
    /** One untimed dispatch; returns hit records (closest) or one word per ray, 1 = blocked (any). */
    trace(mode: TraceMode, rays: GPUBuffer, count: number): Promise<ArrayBuffer>;
    dispose(): void;
}

/**
 * Uploads `data` once for several untimed dispatches (the `rc` sets, where
 * each set is aimed from the hits of the one before). The caller disposes it.
 */
export async function openOursPrepareTracer(device: GPUDevice, data: TraceSceneData): Promise<OursPrepareTracer> {
    const resources = new Resources();
    try {
        const kernel = await TraceBatchKernel.create(device, 'ours prepare');
        const scene = resources.add(uploadTraceScene(device, data, 'ours prepare'));
        return {
            async trace(mode, rays, count) {
                const bytes = count * (mode === 'closest' ? HIT_RECORD_BYTES : 4);
                const results = device.createBuffer({ label: `prepare ${mode}`, size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                try {
                    const encoder = device.createCommandEncoder({ label: `prepare ${mode}` });
                    kernel.encode(encoder, scene, { mode, rays, results, count });
                    device.queue.submit([encoder.finish()]);
                    return await readBuffer(device, results, bytes);
                } finally {
                    results.destroy();
                }
            },
            dispose: () => resources.destroyAll(),
        };
    } catch (error) {
        resources.destroyAll();
        throw error;
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
        const kernel = await TraceBatchKernel.create(device, 'ours prepare');
        const scene = resources.add(uploadTraceScene(device, data, 'ours prepare'));
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

