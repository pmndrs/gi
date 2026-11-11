/**
 * The surface-ray investigation (RC5.4, lever 20). RC1.5 found that rays
 * leaving visible surfaces (Capsaicin-shaped screen-probe rays) trace at
 * 17-35 Mrays/s on the RTX 3060, where the probe-grid bench's rays from
 * points in the air reach about 87. This page splits that gap: it makes
 * RC1.5's probe set ("surface") and a probe-grid set of the same size
 * ("air-grid"), then variants that change one thing at a time between them
 * (origin push, tMin, direction spread, origins swapped, ray order, ray
 * length, closest against any hit), times each through the production batch
 * kernel (`ours-tinybvh`: tinybvh's trees and the platform's walk, as RC1.5),
 * and counts the walk's work on each with a bench-only counted copy of the
 * walk (counted-walk-wgsl.ts).
 *
 * One scene per page; the counts don't depend on the GPU, the rates do.
 */

import { createMeshBvhBuilder, RAY_FLOATS, HIT_RECORD_BYTES, HIT_RECORD_WORD, uploadTraceScene, type TraceSceneBuffers } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { createRandom } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import type { AdapterSummary } from '../../contract.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { BENCH_SCENE_SUBSETS, RAY_DIRECTION_SEED, RC_HALTON_FIRST_INDEX, RC_RAYS_PER_PROBE, RC_SEED, RC_TILE_SIZE, RC_VIEW } from './config.ts';
import { COUNTED_WORD, COUNTED_WORDS, countedKernelWgsl } from './counted-walk-wgsl.ts';
import { createDispatchTimer } from './dispatch-timer.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import { buildOursScene, openOursPrepareTracer, readBuffer, runOursTracerOnSets, uploadRays, yieldToBrowser, type RaySetBuffer } from './ours-bench.ts';
import { makePlacementRays, rayProblems, tileGrid } from './rc-rays.ts';
import { makeProbeGrid, makeProbeRays as makeGridRays, makeSphereDirections, probePosition, type Box, type Vec3 } from './rays.ts';
import { fullBrowserVersion, requestBenchDevice } from './run.ts';
import { findManifest, loadBenchScene } from './scene-source.ts';
import {
    cellMajorOrder,
    makeVariantProbeRays,
    meanInstanceBoxesAtOrigins,
    probeOrigins,
    rayOriginIds,
    reorder,
    shuffledOrder,
    sortedOrder,
    summariseCounts,
    surfaceProbes,
    transplantOrigins,
    withInterval,
    type CountSummary,
    type DirectionKind,
} from './surface-ray-variants.ts';

/** Timing per set: warm-ups, single dispatches waited for one by one, then a loaded section of about this long. */
const SURFACE_TIMING = { warmups: 3, timed: 10, loadedMs: 800 } as const;
/** Every this-many-th ray's origin is checked against every instance's world box. */
const INSTANCE_BOX_SAMPLE_STRIDE = 29;
/** The counted copy of the walk must find the production kernel's answer on at least this share of rays. */
const MIN_COUNTED_AGREEMENT = 0.9999;

/** Pushes off the surface (m) for the offset variants. The baseline is the platform's 256 rounding steps. */
const OFFSET_PUSHES_M = [0.001, 0.01, 0.05, 0.2, 0.5] as const;
/** tMin (m) for the tMin variants: the origin stays where the platform puts it, the walk ignores the first stretch. */
const T_MIN_M = [0.001, 0.01, 0.05] as const;
/** tMax (m) for the length variants. */
const T_MAX_M = [0.5, 1, 2, 5, 10] as const;

interface Variant {
    name: string;
    /** Which question it answers. */
    group: 'baseline' | 'offset' | 't-min' | 'direction' | 'origin-swap' | 'order' | 'length' | 'any-hit' | 'repeat';
    /** What differs from the set it is compared with, in words. */
    change: string;
    mode: 'closest' | 'any';
    rays: Float32Array;
    /** Per ray, `vec2u`: the instance and triangle the ray leaves from, or NO_ORIGIN. */
    originIds: Uint32Array;
}

export interface SurfaceSetResult extends CountSummary {
    name: string;
    group: Variant['group'];
    change: string;
    mode: 'closest' | 'any';
    singleMedianMs: number;
    singleMrays: number;
    loadedMedianMs: number;
    loadedMrays: number;
    loadedDispatches: number;
    /** Share of rays where the counted copy of the walk found the production kernel's answer. */
    countedAgreement: number;
    /** Instances whose world box contains the ray's origin (sampled). */
    meanInstanceBoxesAtOrigin: number;
}

export interface SurfaceRaysData {
    benchmark: 'surface-rays';
    scene: string;
    meshCount: number;
    instanceCount: number;
    instancedTriangles: number;
    camera: string;
    probesPlaced: number;
    gridProbes: number;
    gridDimensions: number[];
    raysPerProbe: number;
    timing: typeof SURFACE_TIMING;
    userAgent: string;
    browserVersion: string | null;
    sets: SurfaceSetResult[];
}

/** Each instance's world-space box, from its mesh's object-space box. */
function instanceWorldBoxes(meshes: readonly BenchMeshGeometry[], instances: readonly BenchInstance[]): Box[] {
    const meshBoxes = meshes.map((mesh) => {
        const min: Vec3 = [Infinity, Infinity, Infinity];
        const max: Vec3 = [-Infinity, -Infinity, -Infinity];
        for (let index = 0; index < mesh.positions.length; index += 3) {
            for (let axis = 0; axis < 3; axis++) {
                min[axis] = Math.min(min[axis], mesh.positions[index + axis]);
                max[axis] = Math.max(max[axis], mesh.positions[index + axis]);
            }
        }
        return { min, max };
    });
    return instances.map((instance) => {
        const box = meshBoxes[instance.meshIndex];
        const m = instance.objectToWorld;
        const min: Vec3 = [Infinity, Infinity, Infinity];
        const max: Vec3 = [-Infinity, -Infinity, -Infinity];
        for (let corner = 0; corner < 8; corner++) {
            const x = corner & 1 ? box.max[0] : box.min[0];
            const y = corner & 2 ? box.max[1] : box.min[1];
            const z = corner & 4 ? box.max[2] : box.min[2];
            for (let axis = 0; axis < 3; axis++) {
                const value = m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis];
                min[axis] = Math.min(min[axis], value);
                max[axis] = Math.max(max[axis], value);
            }
        }
        return { min, max };
    });
}

/** The counted copy of the walk as a pipeline per mode, on one uploaded scene. */
class CountedKernel {
    private constructor(
        private readonly device: GPUDevice,
        private readonly pipelines: Record<'closest' | 'any', GPUComputePipeline>,
        private readonly scene: TraceSceneBuffers,
    ) {}

    static async create(device: GPUDevice, scene: TraceSceneBuffers): Promise<CountedKernel> {
        const make = (mode: 'closest' | 'any') =>
            device.createComputePipelineAsync({
                label: `counted walk ${mode}`,
                layout: 'auto',
                compute: { module: device.createShaderModule({ label: `counted walk ${mode}`, code: countedKernelWgsl(mode) }), entryPoint: 'benchCounted' },
            });
        const [closest, any] = await Promise.all([make('closest'), make('any')]);
        return new CountedKernel(device, { closest, any }, scene);
    }

    /** One dispatch over `count` rays; returns {@link COUNTED_WORDS} words per ray. */
    async count(mode: 'closest' | 'any', rays: GPUBuffer, originIds: Uint32Array, count: number): Promise<ArrayBuffer> {
        const device = this.device;
        const pipeline = this.pipelines[mode];
        const origins = device.createBuffer({ label: 'counted origins', size: originIds.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(origins, 0, originIds as Uint32Array<ArrayBuffer>);
        const bytes = count * COUNTED_WORDS * 4;
        const output = device.createBuffer({ label: 'counted output', size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        try {
            const encoder = device.createCommandEncoder({ label: 'counted walk' });
            const pass = encoder.beginComputePass({ label: 'counted walk' });
            pass.setPipeline(pipeline);
            pass.setBindGroup(
                0,
                device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: this.scene.nodes } },
                        { binding: 1, resource: { buffer: this.scene.triangles } },
                        { binding: 2, resource: { buffer: this.scene.topLevel } },
                    ],
                }),
            );
            pass.setBindGroup(
                1,
                device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(1),
                    entries: [
                        { binding: 0, resource: { buffer: rays, offset: 0, size: count * RAY_FLOATS * 4 } },
                        { binding: 1, resource: { buffer: origins } },
                        { binding: 2, resource: { buffer: output } },
                    ],
                }),
            );
            const workgroups = Math.ceil(count / 64);
            const x = Math.min(workgroups, device.limits.maxComputeWorkgroupsPerDimension);
            pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
            pass.end();
            device.queue.submit([encoder.finish()]);
            return await readBuffer(device, output, bytes);
        } finally {
            origins.destroy();
            output.destroy();
        }
    }
}

/** Share of rays where the counted walk's answer is the production kernel's. */
function countedAgreement(mode: 'closest' | 'any', production: ArrayBuffer, counted: ArrayBuffer, count: number): number {
    const countedFloats = new Float32Array(counted);
    const countedWords = new Uint32Array(counted);
    let agree = 0;
    if (mode === 'closest') {
        const floats = new Float32Array(production);
        const words = new Uint32Array(production);
        const recordWords = HIT_RECORD_BYTES / 4;
        for (let ray = 0; ray < count; ray++) {
            const t = floats[ray * recordWords + HIT_RECORD_WORD.t];
            const countedT = countedFloats[ray * COUNTED_WORDS + COUNTED_WORD.t];
            const bothMiss = t < 0 && countedT < 0;
            const sameHit =
                t >= 0 &&
                countedT >= 0 &&
                Math.abs(t - countedT) <= 1e-6 * Math.max(1, t) &&
                words[ray * recordWords + HIT_RECORD_WORD.instance] === countedWords[ray * COUNTED_WORDS + COUNTED_WORD.instance] &&
                words[ray * recordWords + HIT_RECORD_WORD.triangle] === countedWords[ray * COUNTED_WORDS + COUNTED_WORD.triangle];
            if (bothMiss || sameHit) agree++;
        }
    } else {
        const blocked = new Uint32Array(production);
        for (let ray = 0; ray < count; ray++) if ((blocked[ray] !== 0) === countedFloats[ray * COUNTED_WORDS + COUNTED_WORD.t] > 0) agree++;
    }
    return count > 0 ? agree / count : 1;
}

/** Makes every variant from the scene's probes and grid. */
function makeVariants(
    probes: ReturnType<typeof surfaceProbes>,
    grid: ReturnType<typeof makeProbeGrid>,
    sharedDirections: Float32Array,
    sceneBox: Box,
): Variant[] {
    const raysPerProbe = RC_RAYS_PER_PROBE;
    const surfaceIds = rayOriginIds(probes.originIds, probes.count, raysPerProbe);
    const platformOrigins = probeOrigins(probes, 'platform');
    const noIdsFor = (rays: Float32Array) => rayOriginIds(null, rays.length / RAY_FLOATS / raysPerProbe, raysPerProbe);

    // The baseline: RC1.5's probe set (the same generator, seed and order).
    const surface = makeVariantProbeRays({
        origins: platformOrigins,
        normals: probes.normals,
        count: probes.count,
        raysPerProbe,
        kind: 'cosine-stratified',
        random: createRandom(RC_SEED),
    });
    // The probe-grid bench's rays at the same size: grid points, one shared sphere of directions.
    const airGrid = makeGridRays(grid, sharedDirections);
    const airIds = noIdsFor(airGrid);

    const variants: Variant[] = [
        { name: 'surface', group: 'baseline', change: "RC1.5's probe rays: 64 stratified cosine rays per screen probe, platform push, probe-major", mode: 'closest', rays: surface, originIds: surfaceIds },
        { name: 'air-grid', group: 'baseline', change: "the probe-grid bench's rays: grid points in the manifest volume, 64 shared sphere directions, probe-major", mode: 'closest', rays: airGrid, originIds: airIds },
    ];

    // Origin push: the same directions, the origin moved off the surface along the normal.
    for (const push of OFFSET_PUSHES_M) {
        variants.push({
            name: `offset-${push * 1000}mm`,
            group: 'offset',
            change: `surface, origins pushed ${push * 1000} mm off the surface instead of the platform's 256 rounding steps`,
            mode: 'closest',
            rays: transplantOrigins(surface, raysPerProbe, probeOrigins(probes, push)),
            originIds: surfaceIds,
        });
    }
    for (const tMin of T_MIN_M) {
        variants.push({
            name: `tmin-${tMin * 1000}mm`,
            group: 't-min',
            change: `surface, tMin ${tMin * 1000} mm (origin unchanged)`,
            mode: 'closest',
            rays: withInterval(surface, { tMin }),
            originIds: surfaceIds,
        });
    }

    // Direction spread from the same surface origins.
    const directionKinds: DirectionKind[] = ['cosine-random', 'uniform-hemisphere', 'uniform-sphere', 'shared-sphere'];
    directionKinds.forEach((kind, index) => {
        variants.push({
            name: `dir-${kind}`,
            group: 'direction',
            change: `surface origins, directions ${kind}${kind === 'shared-sphere' ? " (the air grid's 64 directions)" : ''}`,
            mode: 'closest',
            rays: makeVariantProbeRays({
                origins: platformOrigins,
                normals: probes.normals,
                count: probes.count,
                raysPerProbe,
                kind,
                random: createRandom(RC_SEED + 1 + index),
                sharedDirections,
            }),
            originIds: surfaceIds,
        });
    });

    // Origins swapped: the surface set's directions from the grid's points.
    const gridOrigins = new Float64Array(probes.count * 3);
    for (let probe = 0; probe < probes.count; probe++) gridOrigins.set(probePosition(grid, probe % grid.probeCount), probe * 3);
    variants.push({
        name: 'air-origins-surface-dirs',
        group: 'origin-swap',
        change: "surface's directions (cosine about each probe's normal), each probe's origin moved to a grid point",
        mode: 'closest',
        rays: transplantOrigins(surface, raysPerProbe, gridOrigins),
        originIds: noIdsFor(surface),
    });

    // Order: the same rays, threads grouped differently.
    const orders: [string, string, Uint32Array][] = [
        ['order-shuffled', 'surface, rays in a random order (no two neighbours from one probe)', shuffledOrder(probes.count * raysPerProbe, createRandom(RC_SEED + 11))],
        ['order-cell-major', 'surface, ray k of every probe together (neighbouring tiles, same stratum)', cellMajorOrder(probes.count, raysPerProbe)],
        ['order-dir-then-morton', 'surface, sorted by direction bin (8x8 octahedral) then origin Morton code', sortedOrder(surface, sceneBox, 'direction-then-origin')],
        ['order-morton-then-dir', 'surface, sorted by origin Morton code then direction bin', sortedOrder(surface, sceneBox, 'origin-then-direction')],
    ];
    for (const [name, change, order] of orders) {
        variants.push({ name, group: 'order', change, mode: 'closest', rays: reorder(surface, order, RAY_FLOATS), originIds: reorder(surfaceIds, order, 2) });
    }
    const airOrder = shuffledOrder(airGrid.length / RAY_FLOATS, createRandom(RC_SEED + 12));
    variants.push({ name: 'air-grid-shuffled', group: 'order', change: 'air-grid, rays in a random order', mode: 'closest', rays: reorder(airGrid, airOrder, RAY_FLOATS), originIds: airIds });

    // Length: Capsaicin's probe rays are unbounded (FLT_MAX); these cut them.
    for (const tMax of T_MAX_M) {
        variants.push({ name: `tmax-${tMax}m`, group: 'length', change: `surface, tMax ${tMax} m`, mode: 'closest', rays: withInterval(surface, { tMax }), originIds: surfaceIds });
    }

    variants.push({ name: 'surface-any', group: 'any-hit', change: 'surface traced any-hit (far child first), unbounded', mode: 'any', rays: surface, originIds: surfaceIds });
    variants.push({ name: 'air-grid-any', group: 'any-hit', change: 'air-grid traced any-hit (far child first), unbounded', mode: 'any', rays: airGrid, originIds: airIds });

    // The two baselines again, last, to see drift over the run.
    variants.push({ name: 'surface-again', group: 'repeat', change: 'surface, timed again at the end', mode: 'closest', rays: surface, originIds: surfaceIds });
    variants.push({ name: 'air-grid-again', group: 'repeat', change: 'air-grid, timed again at the end', mode: 'closest', rays: airGrid, originIds: airIds });
    return variants;
}

export async function runSurfaceRaysPage(sceneId: string): Promise<void> {
    const startTime = performance.now();
    const lines: string[] = [];
    const progress = (message: string): void => {
        lines.push(message);
        console.info(`[surface-rays] ${message}`);
    };
    let deviceErrors: string[] = [];
    let adapterSummary: AdapterSummary | null = null;
    let data: SurfaceRaysData | null = null;
    let status: 'pass' | 'fail' = 'pass';
    let message: string | undefined;
    const gpuBuffers: GPUBuffer[] = [];
    try {
        const { device } = await requestBenchDevice();
        deviceErrors = watchDevice(device);
        adapterSummary = summarizeAdapterInfo(device.adapterInfo, device.features);
        const timer = await createDispatchTimer(device);
        const builder = createMeshBvhBuilder({ worker: 'auto' });
        try {
            progress(`${sceneId}: loading geometry`);
            const scene = await loadBenchScene(sceneId, 'manifest');
            const manifest = findManifest(BENCH_SCENE_SUBSETS[sceneId]?.manifest ?? sceneId);
            const camera = manifest.cameras[0];
            progress(`${sceneId}: ${scene.meshes.length} meshes, ${scene.instances.length} instances, ${scene.instancedTriangles} triangles; camera "${camera.name}"`);
            const ours = await buildOursScene(builder, scene.meshes, scene.instances);
            await yieldToBrowser();

            // Placement, as RC1.5: one primary ray per 16x16 tile from the first bookmark.
            const view = { width: RC_VIEW.width, height: RC_VIEW.height, tileSize: RC_TILE_SIZE };
            const placementRays = makePlacementRays({ position: camera.position, target: camera.target, fovYDeg: camera.fovYDeg }, view, RC_HALTON_FIRST_INDEX);
            const prepare = await openOursPrepareTracer(device, ours.data);
            let probes: ReturnType<typeof surfaceProbes>;
            try {
                const placementBuffer = uploadRays(device, 'placement', placementRays);
                gpuBuffers.push(placementBuffer);
                const placementHits = await prepare.trace('closest', placementBuffer, tileGrid(view).count);
                probes = surfaceProbes(placementRays, placementHits, scene.meshes, scene.instances);
            } finally {
                prepare.dispose();
            }
            const grid = makeProbeGrid(scene.probeBox, probes.count);
            const sharedDirections = makeSphereDirections(RC_RAYS_PER_PROBE, RAY_DIRECTION_SEED);
            progress(`${sceneId}: ${probes.count} screen probes; air grid ${grid.dimensions.join(' x ')} = ${grid.probeCount} points in the manifest volume`);

            const variants = makeVariants(probes, grid, sharedDirections, scene.geometryBounds);
            const problems = variants.flatMap((variant) => rayProblems(variant.name, variant.rays));
            if (problems.length > 0) throw new Error(`bad rays: ${problems.join('; ')}`);

            const buffers = variants.map((variant) => {
                const buffer = uploadRays(device, `surface-rays ${variant.name}`, variant.rays);
                gpuBuffers.push(buffer);
                return buffer;
            });
            const timedSets: RaySetBuffer[] = variants.map((variant, index) => ({
                raySet: variant.name,
                mode: variant.mode,
                rays: buffers[index],
                count: variant.rays.length / RAY_FLOATS,
            }));

            progress(`${sceneId}: timing ${variants.length} sets through the production batch kernel`);
            const runs = await runOursTracerOnSets(device, timer, 'ours-tinybvh', ours.data, timedSets, SURFACE_TIMING);

            progress(`${sceneId}: counting the walk's work on each set (bench-only counted copy)`);
            const instanceBoxes = instanceWorldBoxes(scene.meshes, scene.instances);
            const sceneBuffers = uploadTraceScene(device, ours.data, 'counted');
            const results: SurfaceSetResult[] = [];
            try {
                const counted = await CountedKernel.create(device, sceneBuffers);
                for (const [index, variant] of variants.entries()) {
                    const count = variant.rays.length / RAY_FLOATS;
                    const run = runs[index];
                    const counts = await counted.count(variant.mode, buffers[index], variant.originIds, count);
                    const loaded = run.mode.loaded;
                    const result: SurfaceSetResult = {
                        name: variant.name,
                        group: variant.group,
                        change: variant.change,
                        mode: variant.mode,
                        singleMedianMs: run.mode.timing.medianMs,
                        singleMrays: run.mode.mraysPerSecond,
                        loadedMedianMs: loaded?.medianMs ?? Number.NaN,
                        loadedMrays: loaded?.mraysPerSecond ?? Number.NaN,
                        loadedDispatches: loaded?.dispatches ?? 0,
                        countedAgreement: countedAgreement(variant.mode, run.results, counts, count),
                        meanInstanceBoxesAtOrigin: meanInstanceBoxesAtOrigins(variant.rays, instanceBoxes, INSTANCE_BOX_SAMPLE_STRIDE),
                        ...summariseCounts(counts, count),
                    };
                    results.push(result);
                    progress(
                        `${sceneId}: ${variant.name.padEnd(26)} ${result.loadedMrays.toFixed(1).padStart(6)} Mrays/s loaded (${result.singleMrays.toFixed(1)} single); ` +
                            `steps ${result.meanSteps.toFixed(1)} (p90 ${result.p90Steps}), mesh nodes ${result.meanMeshNodes.toFixed(1)}, top ${result.meanTopLevelNodes.toFixed(1)}, ` +
                            `instances ${result.meanInstanceEntries.toFixed(2)}, tris ${result.meanTriangleTests.toFixed(1)}, origin boxes ${result.meanOriginBoxes.toFixed(1)}, ` +
                            `steps before 1st hit ${result.meanStepsBeforeFirstHit.toFixed(1)}, in origin instance ${result.meanStepsInOriginInstance.toFixed(1)}, ` +
                            `simd32 ${(result.simd32Efficiency * 100).toFixed(0)}%, hit ${(result.hitFraction * 100).toFixed(1)}%, median t ${result.medianHitT.toFixed(2)}, ` +
                            `self-hit ${(result.selfHitFraction * 100).toFixed(2)}%, agree ${(result.countedAgreement * 100).toFixed(3)}%`,
                    );
                    await yieldToBrowser();
                }
            } finally {
                sceneBuffers.destroy();
            }

            data = {
                benchmark: 'surface-rays',
                scene: sceneId,
                meshCount: scene.meshes.length,
                instanceCount: scene.instances.length,
                instancedTriangles: scene.instancedTriangles,
                camera: camera.name,
                probesPlaced: probes.count,
                gridProbes: grid.probeCount,
                gridDimensions: [...grid.dimensions],
                raysPerProbe: RC_RAYS_PER_PROBE,
                timing: SURFACE_TIMING,
                userAgent: navigator.userAgent,
                browserVersion: await fullBrowserVersion(),
                sets: results,
            };
            const disagreeing = results.filter((result) => result.countedAgreement < MIN_COUNTED_AGREEMENT);
            if (disagreeing.length > 0) {
                status = 'fail';
                message = `the counted copy of the walk disagrees with the production kernel: ${disagreeing.map((result) => `${result.name} ${(result.countedAgreement * 100).toFixed(3)}%`).join(', ')}`;
            }
        } finally {
            builder.dispose();
            timer.dispose();
            for (const buffer of gpuBuffers) buffer.destroy();
        }
        await drainDevice(device);
    } catch (error) {
        status = 'fail';
        message = error instanceof Error ? error.message : String(error);
    }
    reportGpuTest({
        status,
        message,
        data: (data ?? { benchmark: 'surface-rays' }) as unknown as Record<string, unknown>,
        adapter: adapterSummary,
        deviceErrors,
        startTime,
        logLines: lines,
    });
}
