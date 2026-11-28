/**
 * T3.0 spike, question 3 (docs/design/m3-moving.md §3 and §3.1): after 600
 * frames of refits only on a swaying ~100,000-triangle tree, how much slower
 * is tracing than on a fresh build of the same pose?
 *
 * Two meshes, each swaying in a breeze and in a gust:
 * - a procedural tree (tree.ts; 99,696 triangles: a trunk, 60 branches and
 *   47,400 leaf quads), written before T3.11's tree was ready;
 * - T3.11's wind tree (wind-tree.ts; 50,474 triangles, one copy), swayed by
 *   its own wind weights.
 * `VITE_SPIKE_MESHES=procedural` or `=wind-tree` runs one of them.
 *
 * For each mesh and strength:
 * - a tinybvh tree of the rest pose goes into a one-instance trace scene
 *   (ddgi-core's assembly and batch kernel, the tracer DDGI uses);
 * - 600 frames at 60 fps: the CPU writes the frame's pose, `GpuMeshRefit`
 *   refits the tree in the scene's own buffers (nothing else rebuilds it);
 * - every 100 frames (and at 600) the refit tree and a fresh tinybvh build of
 *   the same pose trace the same probe-style rays (512 probes x 128
 *   directions, closest hit and any hit), timed in alternating blocks, and
 *   their hit distances are compared (they must agree: same triangles);
 * - at frame 600 also against a fresh GPU H-PLOC build (the rebuild T3.3
 *   would use), and the refit tree is compared byte for byte with a single
 *   refit from the rest pose straight to frame 600's pose: a refit keeps no
 *   history, so its quality depends on the pose only, not on the frame count.
 *
 * Timing is rough (the GPU is shared with other projects; no timing gate,
 * DECISIONS 2026-11-09). The SAH-cost ratio is reported beside it, as it
 * doesn't depend on the clock.
 */

import {
    assembleTraceScene,
    HIT_RECORD_BYTES,
    packRays,
    RAY_BYTES,
    TraceBatchKernel,
    type TraceRay,
    type TraceSceneBuffers,
} from '../../../../packages/core/src/platform/trace/index.ts';
import { GpuHplocBuilder } from '../../../../packages/bvh/src/gpu-hploc.ts';
import type { GpuLayoutMeshBvh } from '../../../../packages/bvh/src/gpu-layout.ts';
import { GpuMeshRefit } from '../../../../packages/bvh/src/gpu-mesh-refit.ts';
import { treeSahCost } from '../../../../packages/bvh/src/layout-check.ts';
import { createTinyBvhBuilder } from '../../../../packages/bvh/src/tinybvh.ts';
import { assert, runGpuTest } from '../../client.ts';
import { createDispatchTimer, measureDispatches, type DispatchTimer } from '../trace-bench/dispatch-timer.ts';
import { makeProceduralTree, SWAY_STRENGTHS, type SwayingMesh, type SwayStrength } from './tree.ts';
import { loadWindTree } from './wind-tree.ts';

const FRAMES = 600;
const FRAMES_PER_SECOND = 60;
const CHECK_EVERY = 100;
const PROBES_PER_AXIS = 8;
const DIRECTIONS_PER_PROBE = 128;
/** Any-hit rays stop here, about a probe spacing (as DDGI's visibility rays would). */
const ANY_HIT_RANGE = 1;
/** Alternating blocks of timed dispatches per tree, and dispatches per block. */
const TIMING_BLOCKS = 6;
const DISPATCHES_PER_BLOCK = 8;
const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function seededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

/** Probe-style rays: a grid of probes over the tree's box, uniform random directions from each. */
function probeRays(bounds: Float32Array, range: number): TraceRay[] {
    const random = seededRandom(99);
    const rays: TraceRay[] = [];
    for (let x = 0; x < PROBES_PER_AXIS; x++) {
        for (let y = 0; y < PROBES_PER_AXIS; y++) {
            for (let z = 0; z < PROBES_PER_AXIS; z++) {
                const cell = [x, y, z];
                const origin = [0, 1, 2].map((axis) => bounds[axis] + ((cell[axis] + 0.5) / PROBES_PER_AXIS) * (bounds[axis + 3] - bounds[axis])) as [number, number, number];
                for (let direction = 0; direction < DIRECTIONS_PER_PROBE; direction++) {
                    const cosTheta = 2 * random() - 1;
                    const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
                    const phi = 2 * Math.PI * random();
                    rays.push({ origin, direction: [sinTheta * Math.cos(phi), cosTheta, sinTheta * Math.sin(phi)], tMin: 0, tMax: range });
                }
            }
        }
    }
    return rays;
}

/**
 * A one-instance trace scene, uploaded like `uploadTraceScene` but with
 * COPY_SRC too, so the checks can read the refit tree back. The top level's
 * mesh box is `bounds` (every pose's box), so it holds while the mesh sways.
 */
function uploadScene(device: GPUDevice, layout: GpuLayoutMeshBvh, bounds: Float32Array, label: string): TraceSceneBuffers {
    const data = assembleTraceScene([{ ...layout, bounds }], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY }]);
    const buffers = ([['nodes', data.nodes], ['triangles', data.triangles], ['top level', data.topLevel.buffer]] as const).map(([name, bytes]) => {
        const buffer = device.createBuffer({ label: `${label} ${name}`, size: bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
        device.queue.writeBuffer(buffer, 0, bytes);
        return buffer;
    });
    const [nodes, triangles, topLevel] = buffers;
    return { nodes, triangles, topLevel, byteLength: data.byteLength, destroy: () => buffers.forEach((buffer) => buffer.destroy()) };
}

async function readBuffer(device: GPUDevice, buffer: GPUBuffer, size = buffer.size): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.destroy();
    return copy;
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

interface RayBatch {
    readonly rays: GPUBuffer;
    readonly count: number;
    readonly results: GPUBuffer;
}

/** The refit tree's nodes as a layout, for its SAH cost. */
async function readLayout(device: GPUDevice, scene: TraceSceneBuffers, shape: GpuLayoutMeshBvh): Promise<GpuLayoutMeshBvh> {
    return { ...shape, nodes: await readBuffer(device, scene.nodes, shape.nodeCount * 64) };
}

/**
 * Times scenes on the same rays in alternating blocks, so a clock change hits
 * them alike. Returns each scene's fastest dispatch, and the median for the
 * record. The fastest, because this GPU is shared: another process's work
 * lands inside some timed passes (seen as ~10 ms samples next to ~0.4 ms
 * ones), and that only ever adds time.
 */
async function timeAlternating(
    timer: DispatchTimer,
    kernel: TraceBatchKernel,
    scenes: readonly TraceSceneBuffers[],
    mode: 'closest' | 'any',
    batch: RayBatch,
): Promise<{ fastest: number; median: number }[]> {
    const samples: number[][] = scenes.map(() => []);
    for (let block = 0; block < TIMING_BLOCKS; block++) {
        for (const [index, scene] of scenes.entries()) {
            const timing = await measureDispatches(
                () => timer.time((encoder, timestampWrites) => kernel.encode(encoder, scene, { mode, rays: batch.rays, results: batch.results, count: batch.count, timestampWrites })),
                block === 0 ? 2 : 0,
                DISPATCHES_PER_BLOCK,
            );
            samples[index].push(...timing.samplesMs);
        }
    }
    return samples.map((values) => ({ fastest: Math.min(...values), median: median(values) }));
}

/** Closest-hit distances of a scene (negative for a miss). */
async function closestDistances(device: GPUDevice, kernel: TraceBatchKernel, scene: TraceSceneBuffers, batch: RayBatch): Promise<Float32Array> {
    const encoder = device.createCommandEncoder();
    kernel.encode(encoder, scene, { mode: 'closest', rays: batch.rays, results: batch.results, count: batch.count });
    device.queue.submit([encoder.finish()]);
    const records = new Float32Array(await readBuffer(device, batch.results, batch.count * HIT_RECORD_BYTES));
    const words = HIT_RECORD_BYTES / 4;
    return Float32Array.from({ length: batch.count }, (_, ray) => records[ray * words]);
}

function disagreements(a: Float32Array, b: Float32Array): number {
    let count = 0;
    for (let ray = 0; ray < a.length; ray++) {
        const hitA = a[ray] >= 0;
        const hitB = b[ray] >= 0;
        if (hitA !== hitB || (hitA && Math.abs(a[ray] - b[ray]) > 1e-5 * Math.max(1, a[ray]))) count++;
    }
    return count;
}

interface Checkpoint {
    frame: number;
    sahRatio: number;
    refitSignalRatio: number | null;
    /** Each tree's fastest dispatch and their ratio; the ratio of medians for the record. */
    closestMs: { refit: number; fresh: number; ratio: number; medianRatio: number };
    anyMs: { refit: number; fresh: number; ratio: number; medianRatio: number };
    hitDisagreements: number;
}

async function runStrength(
    device: GPUDevice,
    timer: DispatchTimer,
    kernel: TraceBatchKernel,
    tree: SwayingMesh,
    strength: SwayStrength,
    batches: { closest: RayBatch; any: RayBatch },
    log: (line: string) => void,
): Promise<Record<string, unknown>> {
    const tinyBvh = await createTinyBvhBuilder();
    const restLayout = tinyBvh.buildLayout({ positions: tree.restPositions, indices: tree.indices }).layout;
    const restSah = treeSahCost(restLayout);
    const refitScene = uploadScene(device, restLayout, tree.swayBounds, `refit ${strength.name}`);
    const indexBuffer = device.createBuffer({ label: 'tree indices', size: tree.indices.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(indexBuffer, 0, tree.indices);
    const positionBuffer = device.createBuffer({ label: 'tree positions', size: tree.restPositions.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const refit = new GpuMeshRefit(device, { label: `refit ${strength.name}`, diagnostics: true });
    await refit.prepare();
    refit.setTree(restLayout);
    const vertexCount = tree.restPositions.length / 3;
    const pose: Float32Array<ArrayBuffer> = new Float32Array(tree.restPositions.length);

    /** Writes `pose` and refits `scene`'s tree from it. `timestampWrites` times the refit's pass alone. */
    const refitInto = (target: GpuMeshRefit, scene: TraceSceneBuffers, encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites): void => {
        const pass = encoder.beginComputePass({ label: 'tree refit', timestampWrites });
        target.encode(pass, { nodes: scene.nodes, triangles: scene.triangles }, { positionBuffer, vertexCount, indexBuffer, indexFormat: 'uint32' });
        pass.end();
    };

    /** Polls until the refit's stats for the work submitted so far have arrived. */
    const settledSignal = async (): Promise<number | null> => {
        await device.queue.onSubmittedWorkDone();
        for (let tries = 0; tries < 20; tries++) {
            refit.pollStats();
            await new Promise((resolve) => setTimeout(resolve, 10));
        }
        return refit.latestStats?.relativeCost ?? null;
    };

    // The refit's quality signal at the rest pose, where the tree was built: the base T3.3's trigger would compare against.
    device.queue.writeBuffer(positionBuffer, 0, tree.restPositions);
    const restEncoder = device.createCommandEncoder({ label: 'rest refit' });
    refitInto(refit, refitScene, restEncoder);
    refit.copyStats(restEncoder);
    device.queue.submit([restEncoder.finish()]);
    const restSignal = await settledSignal();

    const checkpoints: Checkpoint[] = [];
    const refitMs: number[] = [];
    for (let frame = 0; frame <= FRAMES; frame++) {
        tree.swayInto(pose, frame / FRAMES_PER_SECOND, strength);
        device.queue.writeBuffer(positionBuffer, 0, pose);
        refit.pollStats();
        const checking = frame % CHECK_EVERY === 0;
        if (checking) {
            // This frame's refit timed on its own (rough: the GPU is shared).
            refitMs.push(await timer.time((encoder, writes) => {
                refitInto(refit, refitScene, encoder, writes);
                refit.copyStats(encoder);
            }));
        } else {
            const encoder = device.createCommandEncoder({ label: 'tree refit' });
            refitInto(refit, refitScene, encoder);
            refit.copyStats(encoder);
            device.queue.submit([encoder.finish()]);
        }
        if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
        if (!checking) continue;

        // The refit's own quality signal (what T3.3 would watch), relative to the rest pose's.
        const signal = await settledSignal();

        const freshLayout = tinyBvh.buildLayout({ positions: pose, indices: tree.indices }).layout;
        const freshScene = uploadScene(device, freshLayout, tree.swayBounds, `fresh ${frame}`);
        const refitLayout = await readLayout(device, refitScene, restLayout);
        const sahRatio = treeSahCost(refitLayout) / treeSahCost(freshLayout);
        const [refitClosest, freshClosest] = await timeAlternating(timer, kernel, [refitScene, freshScene], 'closest', batches.closest);
        const [refitAny, freshAny] = await timeAlternating(timer, kernel, [refitScene, freshScene], 'any', batches.any);
        const hitDisagreements = disagreements(await closestDistances(device, kernel, refitScene, batches.closest), await closestDistances(device, kernel, freshScene, batches.closest));
        freshScene.destroy();
        const checkpoint: Checkpoint = {
            frame,
            sahRatio,
            refitSignalRatio: signal !== null && restSignal ? signal / restSignal : null,
            closestMs: { refit: refitClosest.fastest, fresh: freshClosest.fastest, ratio: refitClosest.fastest / freshClosest.fastest, medianRatio: refitClosest.median / freshClosest.median },
            anyMs: { refit: refitAny.fastest, fresh: freshAny.fastest, ratio: refitAny.fastest / freshAny.fastest, medianRatio: refitAny.median / freshAny.median },
            hitDisagreements,
        };
        checkpoints.push(checkpoint);
        log(
            `${strength.name} frame ${frame}: SAH refit/fresh ${sahRatio.toFixed(3)}, refit signal ${checkpoint.refitSignalRatio?.toFixed(3) ?? 'n/a'}; ` +
                `fastest closest ${checkpoint.closestMs.refit.toFixed(3)} / ${checkpoint.closestMs.fresh.toFixed(3)} ms (${checkpoint.closestMs.ratio.toFixed(3)}x), ` +
                `any ${checkpoint.anyMs.refit.toFixed(3)} / ${checkpoint.anyMs.fresh.toFixed(3)} ms (${checkpoint.anyMs.ratio.toFixed(3)}x); hit disagreements ${hitDisagreements}`,
        );
        assert(hitDisagreements <= batches.closest.count * 1e-4, `${strength.name} frame ${frame}: the refit tree and a fresh tree disagree on ${hitDisagreements} closest hits`);
    }

    // Frame 600's pose (still in `pose`) against a fresh H-PLOC tree, the rebuild T3.3 would use.
    const hploc = new GpuHplocBuilder(device, { label: 'spike hploc' });
    await hploc.prepare();
    const hplocTrees = await hploc.build({ positions: pose, indices: tree.indices });
    const [hplocLayout] = await hploc.readLayouts(hplocTrees);
    hplocTrees.destroy();
    let hplocResult: Record<string, number> | null = null;
    if (hplocLayout) {
        const hplocScene = uploadScene(device, hplocLayout, tree.swayBounds, 'hploc');
        const [refitClosest, hplocClosest] = await timeAlternating(timer, kernel, [refitScene, hplocScene], 'closest', batches.closest);
        const refitLayout = await readLayout(device, refitScene, restLayout);
        hplocResult = {
            sahRatio: treeSahCost(refitLayout) / treeSahCost(hplocLayout),
            closestRatio: refitClosest.fastest / hplocClosest.fastest,
            refitClosestMs: refitClosest.fastest,
            hplocClosestMs: hplocClosest.fastest,
        };
        hplocScene.destroy();
        log(`${strength.name} frame ${FRAMES} vs H-PLOC: SAH ${hplocResult.sahRatio.toFixed(3)}, closest ${hplocResult.closestRatio.toFixed(3)}x`);
    }

    // A refit has no memory: one refit from the rest tree straight to frame 600's pose gives the same bytes.
    const oneStepScene = uploadScene(device, restLayout, tree.swayBounds, 'one step');
    const oneStep = new GpuMeshRefit(device, { label: 'one-step refit' });
    await oneStep.prepare();
    oneStep.setTree(restLayout);
    const encoder = device.createCommandEncoder();
    refitInto(oneStep, oneStepScene, encoder);
    device.queue.submit([encoder.finish()]);
    const [after600, afterOne] = await Promise.all([readBuffer(device, refitScene.nodes), readBuffer(device, oneStepScene.nodes)]);
    const after600Triangles = new Uint8Array(await readBuffer(device, refitScene.triangles));
    const afterOneTriangles = new Uint8Array(await readBuffer(device, oneStepScene.triangles));
    const sameNodes = new Uint8Array(after600).every((byte, index) => byte === new Uint8Array(afterOne)[index]);
    const sameTriangles = after600Triangles.every((byte, index) => byte === afterOneTriangles[index]);
    oneStep.dispose();
    oneStepScene.destroy();
    log(`${strength.name}: 600 frames of refits vs one refit from rest: nodes ${sameNodes ? 'identical' : 'differ'}, triangles ${sameTriangles ? 'identical' : 'differ'}`);

    refit.dispose();
    refitScene.destroy();
    indexBuffer.destroy();
    positionBuffer.destroy();
    const last = checkpoints[checkpoints.length - 1];
    return {
        strength,
        restSahCost: restSah,
        checkpoints,
        frame600: { sahRatio: last.sahRatio, closestRatio: last.closestMs.ratio, anyRatio: last.anyMs.ratio },
        worst: {
            sahRatio: Math.max(...checkpoints.map((checkpoint) => checkpoint.sahRatio)),
            closestRatio: Math.max(...checkpoints.map((checkpoint) => checkpoint.closestMs.ratio)),
            anyRatio: Math.max(...checkpoints.map((checkpoint) => checkpoint.anyMs.ratio)),
        },
        hploc: hplocResult,
        refitGpuMs: { fastest: Math.min(...refitMs), median: median(refitMs), samples: refitMs },
        refitHasNoHistory: sameNodes && sameTriangles,
    };
}

await runGpuTest(
    async ({ device, log }) => {
        assert(GpuMeshRefit.isSupported(device), 'the refit needs subgroups');
        // VITE_SPIKE_MESHES=procedural or =wind-tree runs one of the two.
        const only = (import.meta.env as Record<string, unknown>).VITE_SPIKE_MESHES;
        const meshes: SwayingMesh[] = [];
        if (only !== 'wind-tree') meshes.push(makeProceduralTree());
        if (only !== 'procedural') meshes.push(await loadWindTree());
        const timer = await createDispatchTimer(device);
        const kernel = await TraceBatchKernel.create(device, 'spike refit decay');
        const makeBatch = (rays: TraceRay[], mode: 'closest' | 'any'): RayBatch => {
            const packed = packRays(rays) as Float32Array<ArrayBuffer>;
            const rayBuffer = device.createBuffer({ label: `${mode} rays`, size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(rayBuffer, 0, packed);
            const results = device.createBuffer({ label: `${mode} results`, size: rays.length * (mode === 'closest' ? HIT_RECORD_BYTES : 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
            return { rays: rayBuffer, count: packed.byteLength / RAY_BYTES, results };
        };
        const perMesh = [];
        for (const tree of meshes) {
            log(`${tree.name}: ${tree.triangleCount} triangles, ${tree.restPositions.length / 3} vertices`);
            const batches = { closest: makeBatch(probeRays(tree.swayBounds, 1e30), 'closest'), any: makeBatch(probeRays(tree.swayBounds, ANY_HIT_RANGE), 'any') };
            const results = [];
            for (const strength of SWAY_STRENGTHS) results.push(await runStrength(device, timer, kernel, tree, strength, batches, log));
            for (const batch of Object.values(batches)) {
                batch.rays.destroy();
                batch.results.destroy();
            }
            for (const result of results) assert(result.refitHasNoHistory, `${tree.name}, ${(result.strength as SwayStrength).name}: 600 frames of refits differ from one refit of the same pose`);
            perMesh.push({ mesh: tree.name, triangles: tree.triangleCount, rays: batches.closest.count, results });
        }
        timer.dispose();
        return { timingSource: timer.source, meshes: perMesh };
    },
    { optionalFeatures: ['subgroups', 'timestamp-query'] },
);
