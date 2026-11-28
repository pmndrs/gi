/**
 * The surface-ray investigation (P6.6, from the Mac track's RC5.4). RC1.5 found that rays
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

import {
    createMeshBvhBuilder,
    RAY_FLOATS,
    HIT_RECORD_BYTES,
    HIT_RECORD_WORD,
    TraceBatchKernel,
    uploadTraceScene,
    type TraceSceneBuffers,
    type TraceSceneData,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { createRandom } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import type { AdapterSummary } from '../../contract.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { BENCH_SCENE_SUBSETS, RAY_DIRECTION_SEED, RC_HALTON_FIRST_INDEX, RC_RAYS_PER_PROBE, RC_SEED, RC_TILE_SIZE, RC_VIEW } from './config.ts';
import { COUNTED_WORD, COUNTED_WORDS, countedKernelWgsl, NO_ORIGIN, TIE_BREAKS, tieBreakKernelWgsl, type TieBreak } from './counted-walk-wgsl.ts';
import { createDispatchTimer } from './dispatch-timer.ts';
import { IDENTITY_MATRIX, type BenchInstance, type BenchMeshGeometry } from './gltf-geometry.ts';
import { buildOursScene, openOursPrepareTracer, readBuffer, runOursTracerOnSets, timeRaySets, uploadRays, yieldToBrowser, type RaySetBuffer } from './ours-bench.ts';
import { makePlacementRays, rayProblems, tileGrid } from './rc-rays.ts';
import { makeProbeGrid, makeProbeRays as makeGridRays, makeSphereDirections, probePosition, type Box, type Vec3 } from './rays.ts';
import { fullBrowserVersion, requestBenchDevice } from './run.ts';
import { findManifest, loadBenchScene } from './scene-source.ts';
import {
    cellMajorOrder,
    mortonCode,
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
/**
 * The merged tree holds the same triangles in world space, so its hits must
 * match; rounding differs (the transform is applied before the f32 test, not
 * inside it), so a few edge-grazing rays may differ.
 */
const MIN_MERGED_AGREEMENT = 0.999;

/** Pushes off the surface (m) for the offset variants. The baseline is the platform's 256 rounding steps. */
const OFFSET_PUSHES_M = [0.001, 0.01, 0.05, 0.2, 0.5] as const;
/** tMin (m) for the tMin variants: the origin stays where the platform puts it, the walk ignores the first stretch. */
const T_MIN_M = [0.001, 0.01, 0.05] as const;
/** tMax (m) for the length variants. */
const T_MAX_M = [0.5, 1, 2, 5, 10] as const;

interface Variant {
    name: string;
    /** Which question it answers. */
    group: 'baseline' | 'offset' | 't-min' | 'direction' | 'origin-swap' | 'region' | 'order' | 'length' | 'any-hit' | 'repeat' | 'tie-break';
    /** What differs from the set it is compared with, in words. */
    change: string;
    mode: 'closest' | 'any';
    rays: Float32Array;
    /** Per ray, `vec2u`: the instance and triangle the ray leaves from, or NO_ORIGIN. */
    originIds: Uint32Array;
}

/**
 * Which trees the scene is traced through: `instanced` is the platform's
 * (a tree per mesh under a top level over the instances); `merged` is every
 * instance's triangles in world space in one mesh tree under a one-instance
 * top level, to see what the top level's overlapping instance boxes cost;
 * `split-N` is the instanced layout with every mesh cut into compact pieces
 * of at most N triangles (see {@link SPLIT_TRIANGLES}).
 */
export type TreeLayout = 'instanced' | 'merged' | `split-${number}`;

/**
 * Triangles per piece for the split layouts: every mesh cut into pieces of
 * at most this many triangles (consecutive along the Morton order of their
 * centroids, so each piece is compact), each piece its own mesh and every
 * instance one instance per piece. Instancing stays; only the top level's
 * boxes get smaller.
 */
const SPLIT_TRIANGLES = [32_768, 4_096] as const;

/** The sets also traced through the merged tree. */
const MERGED_SETS = ['surface', 'air-grid', 'offset-500mm', 'air-origins-surface-dirs', 'air-region', 'tmax-2m', 'surface-any', 'air-grid-any', 'surface-again'] as const;

/**
 * The paired section: the surface set through the instanced and the merged
 * trees, and through the tie-break walks, timed in turn for this many
 * rounds, the order reversed every other round, so the GPU's drift over the
 * run falls on both sides of every ratio (the batches above are timed one
 * after the other, and the M5 drifted 13-29% over them).
 */
const PAIRED_ROUNDS = 5;
const PAIRED_TIMING = { warmups: 2, timed: 5, loadedMs: 500 } as const;

export interface SurfaceSetResult extends CountSummary {
    tree: TreeLayout;
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
    /** Merged and split trees only: share of rays whose closest hit (or blocked flag) matches the instanced tree's. */
    agreementWithInstanced: number | null;
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
    /** The region grid: points filling the box around the screen probes' surface points. */
    regionGridProbes: number;
    regionBox: Box;
    raysPerProbe: number;
    timing: typeof SURFACE_TIMING;
    userAgent: string;
    browserVersion: string | null;
    sets: SurfaceSetResult[];
    paired: PairedResult;
}

/** One kernel on one tree in the paired section: its loaded rate in every round. */
export interface PairedEntry {
    name: string;
    tree: 'instanced' | 'merged';
    walk: TieBreak;
    mode: 'closest' | 'any';
    mraysPerRound: number[];
    /** Share of rays with the production instanced kernel's answer (first round). */
    agreementWithProduction: number;
}

/** A ratio of two paired entries' rates, per round. */
export interface PairedRatio {
    name: string;
    perRound: number[];
    median: number;
    min: number;
    max: number;
}

export interface PairedResult {
    rounds: number;
    entries: PairedEntry[];
    ratios: PairedRatio[];
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

    /** `tieBreak` changes the closest-hit walk's order for children entered at the same distance (any-hit keeps production's). */
    static async create(device: GPUDevice, scene: TraceSceneBuffers, tieBreak: TieBreak = 'none'): Promise<CountedKernel> {
        const make = (mode: 'closest' | 'any') =>
            device.createComputePipelineAsync({
                label: `counted walk ${mode}`,
                layout: 'auto',
                compute: { module: device.createShaderModule({ label: `counted walk ${mode}`, code: countedKernelWgsl(mode, tieBreak) }), entryPoint: 'benchCounted' },
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
    regionGrid: ReturnType<typeof makeProbeGrid>,
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

    // Region: points in the air, but only in the box around the screen probes' surfaces
    // (the part of the scene the camera sees), not the whole manifest volume.
    const regionRays = makeGridRays(regionGrid, sharedDirections);
    variants.push({
        name: 'air-region',
        group: 'region',
        change: "air-grid's shared directions from grid points filling the box around the screen probes' surface points",
        mode: 'closest',
        rays: regionRays,
        originIds: noIdsFor(regionRays),
    });
    const regionOrigins = new Float64Array(probes.count * 3);
    for (let probe = 0; probe < probes.count; probe++) regionOrigins.set(probePosition(regionGrid, probe % regionGrid.probeCount), probe * 3);
    variants.push({
        name: 'air-region-surface-dirs',
        group: 'region',
        change: "surface's directions, each probe's origin moved to a point of the region grid",
        mode: 'closest',
        rays: transplantOrigins(surface, raysPerProbe, regionOrigins),
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

/** The box around a list of `x, y, z` points. */
function boxAround(points: ArrayLike<number>): Box {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (let index = 0; index < points.length; index += 3) {
        for (let axis = 0; axis < 3; axis++) {
            min[axis] = Math.min(min[axis], points[index + axis]);
            max[axis] = Math.max(max[axis], points[index + axis]);
        }
    }
    return { min, max };
}

/** Every instance's triangles in world space as one mesh, and where each instance's triangles start in it. */
function mergeInstances(meshes: readonly BenchMeshGeometry[], instances: readonly BenchInstance[]): { mesh: BenchMeshGeometry; triangleOffsets: number[]; box: Box } {
    const vertexCount = instances.reduce((total, instance) => total + meshes[instance.meshIndex].positions.length / 3, 0);
    const indexCount = instances.reduce((total, instance) => total + meshes[instance.meshIndex].indices.length, 0);
    const positions = new Float32Array(vertexCount * 3);
    const indices = new Uint32Array(indexCount);
    const triangleOffsets: number[] = [];
    let vertexBase = 0;
    let indexBase = 0;
    for (const instance of instances) {
        const mesh = meshes[instance.meshIndex];
        const m = instance.objectToWorld;
        triangleOffsets.push(indexBase / 3);
        for (let vertex = 0; vertex < mesh.positions.length / 3; vertex++) {
            const x = mesh.positions[vertex * 3];
            const y = mesh.positions[vertex * 3 + 1];
            const z = mesh.positions[vertex * 3 + 2];
            for (let axis = 0; axis < 3; axis++) positions[(vertexBase + vertex) * 3 + axis] = m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis];
        }
        for (let index = 0; index < mesh.indices.length; index++) indices[indexBase + index] = mesh.indices[index] + vertexBase;
        vertexBase += mesh.positions.length / 3;
        indexBase += mesh.indices.length;
    }
    const box = boxAround(positions);
    return { mesh: { positions, indices }, triangleOffsets, box };
}

/**
 * Every mesh cut into compact pieces of at most `piece` triangles (sorted
 * by the Morton code of their centroids in the mesh's box, then cut in
 * runs), and every instance repeated once per piece of its mesh with the
 * same transform. Returns where each original (instance, triangle) went.
 */
function splitMeshes(
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
    piece: number,
): { meshes: BenchMeshGeometry[]; instances: BenchInstance[]; mapId(instance: number, triangle: number): [number, number] } {
    const pieces: BenchMeshGeometry[] = [];
    /** Per original mesh: the first piece's index, and per triangle its piece (relative) and index in the piece. */
    const meshPieces = meshes.map((mesh) => {
        const triangleCount = mesh.indices.length / 3;
        const box = boxAround(mesh.positions);
        const codes = new Float64Array(triangleCount);
        const centroid = [0, 0, 0];
        for (let triangle = 0; triangle < triangleCount; triangle++) {
            for (let axis = 0; axis < 3; axis++) {
                centroid[axis] =
                    (mesh.positions[mesh.indices[triangle * 3] * 3 + axis] +
                        mesh.positions[mesh.indices[triangle * 3 + 1] * 3 + axis] +
                        mesh.positions[mesh.indices[triangle * 3 + 2] * 3 + axis]) /
                    3;
            }
            codes[triangle] = mortonCode(centroid, box);
        }
        const order = Array.from({ length: triangleCount }, (_, index) => index).sort((a, b) => codes[a] - codes[b] || a - b);
        const firstPiece = pieces.length;
        const pieceOf = new Uint32Array(triangleCount);
        const indexInPiece = new Uint32Array(triangleCount);
        for (let start = 0; start < triangleCount; start += piece) {
            const end = Math.min(triangleCount, start + piece);
            const vertexMap = new Map<number, number>();
            const positions: number[] = [];
            const indices = new Uint32Array((end - start) * 3);
            for (let slot = start; slot < end; slot++) {
                const triangle = order[slot];
                pieceOf[triangle] = pieces.length - firstPiece;
                indexInPiece[triangle] = slot - start;
                for (let corner = 0; corner < 3; corner++) {
                    const vertex = mesh.indices[triangle * 3 + corner];
                    let mapped = vertexMap.get(vertex);
                    if (mapped === undefined) {
                        mapped = vertexMap.size;
                        vertexMap.set(vertex, mapped);
                        positions.push(mesh.positions[vertex * 3], mesh.positions[vertex * 3 + 1], mesh.positions[vertex * 3 + 2]);
                    }
                    indices[(slot - start) * 3 + corner] = mapped;
                }
            }
            pieces.push({ positions: Float32Array.from(positions), indices });
        }
        return { firstPiece, pieceCount: pieces.length - firstPiece, pieceOf, indexInPiece };
    });
    const splitInstances: BenchInstance[] = [];
    const firstInstanceOf: number[] = [];
    for (const instance of instances) {
        const { firstPiece, pieceCount } = meshPieces[instance.meshIndex];
        firstInstanceOf.push(splitInstances.length);
        for (let offset = 0; offset < pieceCount; offset++) splitInstances.push({ meshIndex: firstPiece + offset, objectToWorld: instance.objectToWorld });
    }
    return {
        meshes: pieces,
        instances: splitInstances,
        mapId: (instance, triangle) => {
            const mesh = meshPieces[instances[instance].meshIndex];
            return [firstInstanceOf[instance] + mesh.pieceOf[triangle], mesh.indexInPiece[triangle]];
        },
    };
}

/** Per-ray origin records through a renumbering of (instance, triangle). */
function mappedOriginIds(originIds: Uint32Array, mapId: (instance: number, triangle: number) => [number, number]): Uint32Array {
    const ids = originIds.slice();
    for (let slot = 0; slot < ids.length; slot += 2) {
        if (ids[slot] === NO_ORIGIN) continue;
        const [instance, triangle] = mapId(ids[slot], ids[slot + 1]);
        ids[slot] = instance;
        ids[slot + 1] = triangle;
    }
    return ids;
}

/** Per-ray origin records for the merged tree: instance 0, the instance's triangle offset plus the triangle. */
function mergedOriginIds(originIds: Uint32Array, triangleOffsets: readonly number[]): Uint32Array {
    const ids = originIds.slice();
    for (let slot = 0; slot < ids.length; slot += 2) {
        if (ids[slot] === NO_ORIGIN) continue;
        ids[slot + 1] = triangleOffsets[ids[slot]] + ids[slot + 1];
        ids[slot] = 0;
    }
    return ids;
}

/** Share of rays with the same answer in two scenes whose triangle numbering differs: closest by `t`, any by the blocked flag. */
function answerAgreement(mode: 'closest' | 'any', a: ArrayBuffer, b: ArrayBuffer, count: number): number {
    let agree = 0;
    if (mode === 'closest') {
        const recordWords = HIT_RECORD_BYTES / 4;
        const floatsA = new Float32Array(a);
        const floatsB = new Float32Array(b);
        for (let ray = 0; ray < count; ray++) {
            const tA = floatsA[ray * recordWords + HIT_RECORD_WORD.t];
            const tB = floatsB[ray * recordWords + HIT_RECORD_WORD.t];
            if ((tA < 0 && tB < 0) || (tA >= 0 && tB >= 0 && Math.abs(tA - tB) <= 1e-4 * Math.max(1, tA))) agree++;
        }
    } else {
        const blockedA = new Uint32Array(a);
        const blockedB = new Uint32Array(b);
        for (let ray = 0; ray < count; ray++) if ((blockedA[ray] !== 0) === (blockedB[ray] !== 0)) agree++;
    }
    return count > 0 ? agree / count : 1;
}

interface MeasureContext {
    device: GPUDevice;
    timer: Awaited<ReturnType<typeof createDispatchTimer>>;
    sceneId: string;
    progress(message: string): void;
}

/**
 * Times every set through the production batch kernel on `data`, then
 * counts the walk's work on each with the counted copy. With
 * `instancedAnswers`, also compares each set's answers with the instanced
 * tree's (the merged tree must find the same hits).
 */
async function measureTree(
    context: MeasureContext,
    tree: TreeLayout,
    data: TraceSceneData,
    variants: readonly Variant[],
    buffers: readonly GPUBuffer[],
    instanceBoxes: readonly Box[],
    instancedAnswers: ReadonlyMap<string, ArrayBuffer> | null,
): Promise<{ results: SurfaceSetResult[]; answers: Map<string, ArrayBuffer> }> {
    const { device, timer, sceneId, progress } = context;
    const timedSets: RaySetBuffer[] = variants.map((variant, index) => ({
        raySet: variant.name,
        mode: variant.mode,
        rays: buffers[index],
        count: variant.rays.length / RAY_FLOATS,
    }));
    progress(`${sceneId} (${tree}): timing ${variants.length} sets through the production batch kernel`);
    const runs = await runOursTracerOnSets(device, timer, 'ours-tinybvh', data, timedSets, SURFACE_TIMING);

    progress(`${sceneId} (${tree}): counting the walk's work on each set (bench-only counted copy)`);
    const sceneBuffers = uploadTraceScene(device, data, `counted ${tree}`);
    const results: SurfaceSetResult[] = [];
    const answers = new Map<string, ArrayBuffer>();
    try {
        const counted = await CountedKernel.create(device, sceneBuffers);
        for (const [index, variant] of variants.entries()) {
            const count = variant.rays.length / RAY_FLOATS;
            const run = runs[index];
            answers.set(variant.name, run.results);
            const reference = instancedAnswers?.get(variant.name) ?? null;
            const counts = await counted.count(variant.mode, buffers[index], variant.originIds, count);
            const loaded = run.mode.loaded;
            const result: SurfaceSetResult = {
                tree,
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
                agreementWithInstanced: reference ? answerAgreement(variant.mode, reference, run.results, count) : null,
                ...summariseCounts(counts, count),
            };
            results.push(result);
            progress(
                `${sceneId} (${tree}): ${variant.name.padEnd(26)} ${result.loadedMrays.toFixed(1).padStart(6)} Mrays/s loaded (${result.singleMrays.toFixed(1)} single); ` +
                    `steps ${result.meanSteps.toFixed(1)} (p90 ${result.p90Steps}), mesh nodes ${result.meanMeshNodes.toFixed(1)}, top ${result.meanTopLevelNodes.toFixed(1)}, ` +
                    `instances ${result.meanInstanceEntries.toFixed(2)} (${result.meanInstanceEntriesWithoutHit.toFixed(2)} without a hit), tris ${result.meanTriangleTests.toFixed(1)}, ` +
                    `origin boxes ${result.meanOriginBoxes.toFixed(1)}, simd32 ${(result.simd32Efficiency * 100).toFixed(0)}%, hit ${(result.hitFraction * 100).toFixed(1)}%, ` +
                    `median t ${result.medianHitT.toFixed(2)}, self-hit ${(result.selfHitFraction * 100).toFixed(2)}%, counted agrees ${(result.countedAgreement * 100).toFixed(3)}%` +
                    (result.agreementWithInstanced === null ? '' : `, agrees with instanced ${(result.agreementWithInstanced * 100).toFixed(3)}%`),
            );
            await yieldToBrowser();
        }
    } finally {
        sceneBuffers.destroy();
    }
    return { results, answers };
}

/** Records one dispatch of the tie-break walk (same bindings and hit record as the production batch kernel). */
function encodeTieBreak(
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    scene: TraceSceneBuffers,
    encoder: GPUCommandEncoder,
    set: RaySetBuffer,
    results: GPUBuffer,
    timestampWrites: GPUComputePassTimestampWrites | undefined,
): void {
    const pass = encoder.beginComputePass({ label: 'tie-break walk', timestampWrites });
    pass.setPipeline(pipeline);
    pass.setBindGroup(
        0,
        device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: scene.nodes } },
                { binding: 1, resource: { buffer: scene.triangles } },
                { binding: 2, resource: { buffer: scene.topLevel } },
            ],
        }),
    );
    pass.setBindGroup(
        1,
        device.createBindGroup({
            layout: pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: set.rays, offset: 0, size: set.count * RAY_FLOATS * 4 } },
                { binding: 1, resource: { buffer: results } },
            ],
        }),
    );
    const workgroups = Math.ceil(set.count / 64);
    const x = Math.min(workgroups, device.limits.maxComputeWorkgroupsPerDimension);
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
    pass.end();
}

function ratioOf(name: string, numerator: PairedEntry, denominator: PairedEntry): PairedRatio {
    const perRound = numerator.mraysPerRound.map((rate, round) => rate / denominator.mraysPerRound[round]);
    const sorted = [...perRound].sort((a, b) => a - b);
    return { name, perRound, median: sorted[sorted.length >> 1], min: sorted[0], max: sorted[sorted.length - 1] };
}

/**
 * The paired section (see {@link PAIRED_ROUNDS}): the surface set's closest
 * and any-hit rays through the instanced and merged trees with the
 * production kernel, and its closest-hit rays through every tie-break walk
 * on both trees, each timed once per round, the order reversed every other
 * round. Ratios are taken within a round.
 */
async function pairedTiming(
    context: MeasureContext,
    instancedData: TraceSceneData,
    mergedData: TraceSceneData,
    surface: RaySetBuffer,
    surfaceAny: RaySetBuffer,
): Promise<PairedResult> {
    const { device, timer, sceneId, progress } = context;
    const scenes = { instanced: uploadTraceScene(device, instancedData, 'paired instanced'), merged: uploadTraceScene(device, mergedData, 'paired merged') };
    try {
        const production = await TraceBatchKernel.create(device, 'paired');
        const tiePipelines = new Map<TieBreak, GPUComputePipeline>();
        for (const tieBreak of TIE_BREAKS.filter((entry) => entry !== 'none')) {
            tiePipelines.set(
                tieBreak,
                await device.createComputePipelineAsync({
                    label: `tie-break ${tieBreak}`,
                    layout: 'auto',
                    compute: { module: device.createShaderModule({ label: `tie-break ${tieBreak}`, code: tieBreakKernelWgsl(tieBreak) }), entryPoint: 'benchTieBreak' },
                }),
            );
        }
        interface Plan {
            entry: PairedEntry;
            set: RaySetBuffer;
            encode: Parameters<typeof timeRaySets>[5];
        }
        const plans: Plan[] = [];
        for (const tree of ['instanced', 'merged'] as const) {
            for (const walk of TIE_BREAKS) {
                const pipeline = tiePipelines.get(walk);
                plans.push({
                    entry: { name: walk === 'none' ? 'surface' : `surface-tie-${walk}`, tree, walk, mode: 'closest', mraysPerRound: [], agreementWithProduction: Number.NaN },
                    set: surface,
                    encode: pipeline
                        ? (encoder, set, results, timestampWrites) => encodeTieBreak(device, pipeline, scenes[tree], encoder, set, results, timestampWrites)
                        : (encoder, set, results, timestampWrites) => production.encode(encoder, scenes[tree], { mode: set.mode, rays: set.rays, results, count: set.count, timestampWrites }),
                });
            }
            plans.push({
                entry: { name: 'surface-any', tree, walk: 'none', mode: 'any', mraysPerRound: [], agreementWithProduction: Number.NaN },
                set: surfaceAny,
                encode: (encoder, set, results, timestampWrites) => production.encode(encoder, scenes[tree], { mode: set.mode, rays: set.rays, results, count: set.count, timestampWrites }),
            });
        }
        const firstAnswers = new Map<Plan, ArrayBuffer>();
        for (let round = 0; round < PAIRED_ROUNDS; round++) {
            const order = round % 2 === 0 ? plans : [...plans].reverse();
            for (const plan of order) {
                const [run] = await timeRaySets(device, timer, 'ours-tinybvh', [plan.set], PAIRED_TIMING, plan.encode);
                plan.entry.mraysPerRound.push(run.mode.loaded?.mraysPerSecond ?? Number.NaN);
                if (round === 0) firstAnswers.set(plan, run.results);
                await yieldToBrowser();
            }
            progress(`${sceneId}: paired round ${round + 1} of ${PAIRED_ROUNDS}: ${plans.map((plan) => `${plan.entry.tree} ${plan.entry.name} ${plan.entry.mraysPerRound[round].toFixed(1)}`).join(', ')}`);
        }
        const find = (tree: 'instanced' | 'merged', name: string): Plan => {
            const plan = plans.find((candidate) => candidate.entry.tree === tree && candidate.entry.name === name);
            if (!plan) throw new Error(`no paired entry ${tree} ${name}`);
            return plan;
        };
        for (const plan of plans) {
            const reference = firstAnswers.get(find('instanced', plan.entry.mode === 'closest' ? 'surface' : 'surface-any'));
            const answers = firstAnswers.get(plan);
            if (reference && answers) plan.entry.agreementWithProduction = answerAgreement(plan.entry.mode, reference, answers, plan.set.count);
        }
        const ratios = [
            ratioOf('merged_over_instanced_closest', find('merged', 'surface').entry, find('instanced', 'surface').entry),
            ratioOf('merged_over_instanced_any', find('merged', 'surface-any').entry, find('instanced', 'surface-any').entry),
        ];
        for (const tree of ['instanced', 'merged'] as const) {
            for (const walk of TIE_BREAKS.filter((entry) => entry !== 'none')) {
                ratios.push(ratioOf(`${tree}_tie_${walk}_over_none`, find(tree, `surface-tie-${walk}`).entry, find(tree, 'surface').entry));
            }
        }
        for (const ratio of ratios) progress(`${sceneId}: paired ${ratio.name}: median ${ratio.median.toFixed(3)} (${ratio.min.toFixed(3)}-${ratio.max.toFixed(3)} over ${PAIRED_ROUNDS} rounds)`);
        return { rounds: PAIRED_ROUNDS, entries: plans.map((plan) => plan.entry), ratios };
    } finally {
        scenes.instanced.destroy();
        scenes.merged.destroy();
    }
}

/**
 * The walk's work on the surface set with each tie-break, on one tree:
 * counted results named `surface-tie-<rule>` (group `tie-break`), with the
 * paired section's median rate.
 */
async function countTieBreaks(
    context: MeasureContext,
    tree: 'instanced' | 'merged',
    data: TraceSceneData,
    surface: Variant,
    buffer: GPUBuffer,
    paired: PairedResult,
    reference: ArrayBuffer,
): Promise<SurfaceSetResult[]> {
    const { device } = context;
    const sceneBuffers = uploadTraceScene(device, data, `tie-break ${tree}`);
    const results: SurfaceSetResult[] = [];
    try {
        const count = surface.rays.length / RAY_FLOATS;
        for (const walk of TIE_BREAKS.filter((entry) => entry !== 'none')) {
            const kernel = await CountedKernel.create(device, sceneBuffers, walk);
            const counts = await kernel.count('closest', buffer, surface.originIds, count);
            const name = `surface-tie-${walk}`;
            const entry = paired.entries.find((candidate) => candidate.tree === tree && candidate.name === name);
            const rates = [...(entry?.mraysPerRound ?? [])].sort((a, b) => a - b);
            const rate = rates.length > 0 ? rates[rates.length >> 1] : Number.NaN;
            // The counted record's first words are a hit record's t, instance and triangle, so agreement is read as for the production answers.
            const summary = summariseCounts(counts, count);
            results.push({
                tree,
                name,
                group: 'tie-break',
                change: `surface, children entered at the same distance ordered by ${walk} instead of left first (bench-only walk)`,
                mode: 'closest',
                singleMedianMs: Number.NaN,
                singleMrays: Number.NaN,
                loadedMedianMs: Number.NaN,
                loadedMrays: rate,
                loadedDispatches: 0,
                countedAgreement: answerAgreementCounted(reference, counts, count),
                meanInstanceBoxesAtOrigin: Number.NaN,
                agreementWithInstanced: null,
                ...summary,
            });
            context.progress(`${context.sceneId} (${tree}): ${name}: steps ${summary.meanSteps.toFixed(1)}, median paired rate ${rate.toFixed(1)} Mrays/s, same closest hit as production on ${(results[results.length - 1].countedAgreement * 100).toFixed(3)}%`);
        }
    } finally {
        sceneBuffers.destroy();
    }
    return results;
}

/** Share of rays whose counted closest `t` matches a production hit record's (hit and miss alike, `t` within 1e-4 relative). */
function answerAgreementCounted(production: ArrayBuffer, counted: ArrayBuffer, count: number): number {
    const recordWords = HIT_RECORD_BYTES / 4;
    const floats = new Float32Array(production);
    const countedFloats = new Float32Array(counted);
    let agree = 0;
    for (let ray = 0; ray < count; ray++) {
        const t = floats[ray * recordWords + HIT_RECORD_WORD.t];
        const countedT = countedFloats[ray * COUNTED_WORDS + COUNTED_WORD.t];
        if ((t < 0 && countedT < 0) || (t >= 0 && countedT >= 0 && Math.abs(t - countedT) <= 1e-4 * Math.max(1, t))) agree++;
    }
    return count > 0 ? agree / count : 1;
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

            const regionBox = boxAround(probes.positions);
            const regionGrid = makeProbeGrid(regionBox, probes.count);
            progress(`${sceneId}: region grid ${regionGrid.dimensions.join(' x ')} = ${regionGrid.probeCount} points in the box around the screen probes (${regionBox.min.map((v) => v.toFixed(1)).join(', ')} to ${regionBox.max.map((v) => v.toFixed(1)).join(', ')})`);
            const variants = makeVariants(probes, grid, regionGrid, sharedDirections, scene.geometryBounds);
            const problems = variants.flatMap((variant) => rayProblems(variant.name, variant.rays));
            if (problems.length > 0) throw new Error(`bad rays: ${problems.join('; ')}`);

            const buffers = variants.map((variant) => {
                const buffer = uploadRays(device, `surface-rays ${variant.name}`, variant.rays);
                gpuBuffers.push(buffer);
                return buffer;
            });
            const instanceBoxes = instanceWorldBoxes(scene.meshes, scene.instances);
            const measureContext = { device, timer, sceneId, progress };
            const instanced = await measureTree(measureContext, 'instanced', ours.data, variants, buffers, instanceBoxes, null);
            const results = instanced.results;

            // The same rays through one tree over the whole scene: no top level to speak of.
            progress(`${sceneId}: building one tree over every instance's triangles in world space (tinybvh, in the worker)`);
            const merged = mergeInstances(scene.meshes, scene.instances);
            const mergedScene = await buildOursScene(builder, [merged.mesh], [{ meshIndex: 0, objectToWorld: IDENTITY_MATRIX }]);
            const mergedIndices = variants.flatMap((variant, index) => ((MERGED_SETS as readonly string[]).includes(variant.name) ? [index] : []));
            const mergedVariants = mergedIndices.map((index) => ({ ...variants[index], originIds: mergedOriginIds(variants[index].originIds, merged.triangleOffsets) }));
            const mergedRun = await measureTree(
                measureContext,
                'merged',
                mergedScene.data,
                mergedVariants,
                mergedIndices.map((index) => buffers[index]),
                [merged.box],
                instanced.answers,
            );
            results.push(...mergedRun.results);

            // Instanced against merged, and the tie-break walks, timed in turn so drift cancels.
            const surfaceIndex = variants.findIndex((variant) => variant.name === 'surface');
            const surfaceAnyIndex = variants.findIndex((variant) => variant.name === 'surface-any');
            const asSet = (index: number): RaySetBuffer => ({ raySet: variants[index].name, mode: variants[index].mode, rays: buffers[index], count: variants[index].rays.length / RAY_FLOATS });
            progress(`${sceneId}: paired section, ${PAIRED_ROUNDS} rounds`);
            const paired = await pairedTiming(measureContext, ours.data, mergedScene.data, asSet(surfaceIndex), asSet(surfaceAnyIndex));
            const surfaceAnswers = instanced.answers.get('surface');
            if (!surfaceAnswers) throw new Error('the instanced surface set has no answers');
            results.push(...(await countTieBreaks(measureContext, 'instanced', ours.data, variants[surfaceIndex], buffers[surfaceIndex], paired, surfaceAnswers)));
            const mergedSurface = mergedVariants.find((variant) => variant.name === 'surface');
            if (!mergedSurface) throw new Error('the merged sets have no surface set');
            // Each tree's tie-break walks against that tree's own production answers. The merged tree's
            // answers differ from the instanced tree's on a few edge-grazing rays (rounding, checked above
            // against MIN_MERGED_AGREEMENT), so comparing its walks with the instanced answers measured that
            // rounding, not the walk: 99.989% on Bistro interior on the RTX 3060, the merged tree's own
            // agreement with the instanced one, to the ray.
            const mergedSurfaceAnswers = mergedRun.answers.get('surface');
            if (!mergedSurfaceAnswers) throw new Error('the merged surface set has no answers');
            results.push(...(await countTieBreaks(measureContext, 'merged', mergedScene.data, mergedSurface, buffers[surfaceIndex], paired, mergedSurfaceAnswers)));

            // The instanced layout with every mesh cut into compact pieces: instancing kept, smaller top-level boxes.
            for (const piece of SPLIT_TRIANGLES) {
                const split = splitMeshes(scene.meshes, scene.instances, piece);
                progress(`${sceneId}: split into pieces of at most ${piece} triangles: ${split.meshes.length} meshes, ${split.instances.length} instances`);
                const splitScene = await buildOursScene(builder, split.meshes, split.instances);
                const splitVariants = mergedIndices.map((index) => ({ ...variants[index], originIds: mappedOriginIds(variants[index].originIds, split.mapId) }));
                const splitRun = await measureTree(
                    measureContext,
                    `split-${piece}`,
                    splitScene.data,
                    splitVariants,
                    mergedIndices.map((index) => buffers[index]),
                    instanceWorldBoxes(split.meshes, split.instances),
                    instanced.answers,
                );
                results.push(...splitRun.results);
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
                regionGridProbes: regionGrid.probeCount,
                regionBox,
                raysPerProbe: RC_RAYS_PER_PROBE,
                timing: SURFACE_TIMING,
                userAgent: navigator.userAgent,
                browserVersion: await fullBrowserVersion(),
                sets: results,
                paired,
            };
            const disagreeing = results.filter((result) => result.countedAgreement < MIN_COUNTED_AGREEMENT);
            const pairedDisagreeing = paired.entries.filter((entry) => !(entry.agreementWithProduction >= MIN_MERGED_AGREEMENT));
            const mergedDisagreeing = results.filter((result) => result.agreementWithInstanced !== null && result.agreementWithInstanced < MIN_MERGED_AGREEMENT);
            const failures = [
                ...disagreeing.map((result) => `the counted copy of the walk disagrees with the production kernel on ${result.tree} ${result.name}: ${(result.countedAgreement * 100).toFixed(3)}%`),
                ...pairedDisagreeing.map((entry) => `paired ${entry.tree} ${entry.name} disagrees with the production instanced kernel: ${(entry.agreementWithProduction * 100).toFixed(3)}%`),
                ...mergedDisagreeing.map((result) => `the ${result.tree} tree disagrees with the instanced one on ${result.name}: ${((result.agreementWithInstanced ?? 0) * 100).toFixed(3)}%`),
            ];
            if (failures.length > 0) {
                status = 'fail';
                message = failures.join(' | ');
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
