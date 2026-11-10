/**
 * B1a timing: the GPU top level with 10,000 instances, all moving every frame.
 *
 * The scene is `synthetic-10k` (the trace bench's: copies of one small sphere
 * at random rotations, non-uniform scales and positions in a 200-unit cube).
 * Every frame every instance moves a small step, then the scene copy uploads
 * and the tracer's `update` records the top-level refit pass: the
 * per-instance gather and the bottom-up refit, timed together by timestamp
 * queries around that one pass. Each frame is submitted and waited for alone,
 * so no other GPU work overlaps the timed pass.
 *
 * Also reported: the main thread's time in `setTransform` (the scene copy's
 * own work per moved instance) and in the tracer's `update`; how many new
 * shapes the quality signal asked for and what one costs in the worker; and
 * the GPU refit's own counters. Afterwards the whole tree is read back and
 * checked node by node (top-level-check.ts), and no refit may have given up
 * waiting for a box.
 *
 * Then the cost of the refit's safety path: the same 10,000 instances in a
 * standalone `GpuBvhRefit` whose leaf-slot list names one leaf twice, so
 * one climb waits for a box nobody writes, gives up after its bounded spin,
 * and the repair pass refits the whole tree in one workgroup. Timed as one
 * pass (clear, climb, the spin, the repair), median of 20.
 *
 * Done-when (TASKS, B1a): the median refit pass is at most 0.5 ms on the RTX
 * 3060, over 3 runs under the bench lock (to-csv.ts writes the rows).
 */

import type { InstanceHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { GpuBvhRefit } from '../../../../packages/bvh/src/gpu-refit.ts';
import { PRIMITIVE_BOX_FLOATS } from '../../../../packages/bvh/src/refit-cpu.ts';
import { BVH_NODE_BYTES } from '../../../../packages/ddgi-core/src/platform/trace/layout.ts';
import { transformBounds } from '../../../../packages/ddgi-core/src/platform/trace/top-level.ts';
import { buildTopLevelShape } from '../../../../packages/ddgi-core/src/platform/trace/top-level-shape.ts';
import type { DispatchTimer } from '../trace-bench/dispatch-timer.ts';
import { PlatformKernelBuilder } from '../../../../packages/ddgi-core/src/platform/kernel-builder.ts';
import { PlatformScene } from '../../../../packages/ddgi-core/src/platform/scene/platform-scene.ts';
import { PlatformRayQueries } from '../../../../packages/ddgi-core/src/platform/trace/ray-queries.ts';
import { TOP_LEVEL_REBUILD_COST_RATIO } from '../../../../packages/ddgi-core/src/platform/trace/top-level-gpu.ts';
import { createRandom, randomAffine } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { assert, runGpuTest } from '../../client.ts';
import { createDispatchTimer } from '../trace-bench/dispatch-timer.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';
import { median } from '../trace-bench/statistics.ts';
import { makeSphereMesh } from '../trace-bench/synthetic.ts';
import { checkTopLevel, objectBoxOf, type CheckedInstance } from '../tlas-oracle/top-level-check.ts';

const SCENE = 'synthetic-10k';
const INSTANCES = 10_000;
const EXTENT = 100;
/** Each instance moves up to this far per axis per frame. */
const STEP = 0.25;
const WARMUP_FRAMES = 10;
const TIMED_FRAMES = 60;
/** Timed runs of the forced repair. */
const REPAIR_RUNS = 20;

/**
 * GPU time of one refit pass that has to repair: a 10,000-instance tree whose
 * leaf-slot list names one leaf twice (see the top of this file). Checks that
 * the repair really ran.
 */
async function timeForcedRepair(device: GPUDevice, timer: DispatchTimer, matrices: readonly number[][], objectBox: readonly number[]): Promise<{ medianMs: number; gaveUp: number }> {
    const count = matrices.length;
    const objectToWorld = new Float64Array(count * 16);
    const meshBounds = new Float32Array(count * 6);
    matrices.forEach((matrix, index) => {
        objectToWorld.set(matrix, index * 16);
        meshBounds.set(objectBox, index * 6);
    });
    const shape = buildTopLevelShape({
        slots: Uint32Array.from({ length: count }, (_, index) => index),
        visible: new Uint8Array(count).fill(1),
        objectToWorld: objectToWorld.slice(),
        meshBounds: meshBounds.slice(),
    });
    const primitiveBoxes = new Float32Array(count * PRIMITIVE_BOX_FLOATS);
    const box = new Float32Array(6);
    for (let record = 0; record < count; record++) {
        const input = shape.recordSlots[record];
        transformBounds(meshBounds.subarray(input * 6, input * 6 + 6), objectToWorld.subarray(input * 16, input * 16 + 16), box, 0);
        primitiveBoxes.set(box.subarray(0, 3), record * PRIMITIVE_BOX_FLOATS);
        primitiveBoxes.set(box.subarray(3, 6), record * PRIMITIVE_BOX_FLOATS + 4);
    }
    const leafSlots = shape.leafSlots.slice();
    const position = leafSlots.findIndex((slot, index) => (slot & 1) === 0 && leafSlots[index + 1] === slot + 1);
    assert(position >= 0, 'no node with two leaf children');
    leafSlots[position + 1] = leafSlots[position];

    const nodeBytes = shape.nodeCount * BVH_NODE_BYTES;
    const tree = device.createBuffer({ label: 'forced repair tree', size: nodeBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(tree, 0, shape.nodes, 0, nodeBytes);
    const boxes = device.createBuffer({ label: 'forced repair boxes', size: primitiveBoxes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(boxes, 0, primitiveBoxes);
    const refit = new GpuBvhRefit(device, { label: 'forced repair', diagnostics: true });
    try {
        await refit.prepare();
        refit.setShape({ nodeWordOffset: 0, nodeCount: shape.nodeCount, parentSlots: shape.parentSlots, leafSlots, generation: 1 });
        const samples: number[] = [];
        for (let run = 0; run < WARMUP_FRAMES + REPAIR_RUNS; run++) {
            const ms = await timer.time((encoder, writes) => {
                const pass = encoder.beginComputePass({ label: 'forced repair', timestampWrites: writes });
                refit.encode(pass, tree, boxes);
                pass.end();
                refit.copyStats(encoder);
            });
            if (run >= WARMUP_FRAMES) samples.push(ms);
            refit.pollStats();
        }
        for (let waited = 0; refit.latestStats === null || refit.latestStats.repairs === 0; waited++) {
            assert(waited < 200, 'the forced repair never reported a repair');
            await new Promise((resolve) => setTimeout(resolve, 5));
            refit.pollStats();
        }
        const stats = refit.latestStats;
        assert(stats.incomplete > 0 && stats.repairs === 1, `the forced repair did not run (gave up ${stats.incomplete}, repairs ${stats.repairs})`);
        return { medianMs: median(samples), gaveUp: stats.incomplete };
    } finally {
        refit.dispose();
        tree.destroy();
        boxes.destroy();
    }
}

/** Done-when bar on the RTX 3060 (TASKS, B1a). */
const BAR_MS = 0.5;

await runGpuTest(
    async ({ device, log }) => {
        assert(device.features.has('subgroups'), 'this adapter has no subgroups, so it has no GPU top level to time');
        assert(device.features.has('timestamp-query'), 'this adapter has no timestamp queries');
        const random = createRandom(0x10_000);
        const scene = new PlatformScene(device, { label: 'tlas-bench' });
        // Diagnostics on, so a refit that gave up is counted. Its 64-byte stats copy runs after the timed pass.
        const rays = new PlatformRayQueries(device, scene, { label: 'tlas-bench', diagnostics: true });
        const timer = await createDispatchTimer(device);
        try {
            assert(rays.topLevelPath === 'gpu-refit', `expected the GPU top level, got ${rays.topLevelPath}`);
            const top = rays.gpuTopLevel;
            assert(top, 'no GPU top level');
            await rays.prepare(new PlatformKernelBuilder(device, 'tlas-bench'), { batches: false });

            const sphere = makeSphereMesh();
            const objectBox = objectBoxOf(sphere.positions);
            assert(sphere.indices, 'the sphere mesh is indexed');
            const mesh = scene.createMesh({ positions: sphere.positions, indices: sphere.indices });
            const material = scene.createMaterial({ albedo: [0.8, 0.8, 0.8] });
            const matrices: number[][] = [];
            const handles: InstanceHandle[] = [];
            for (let index = 0; index < INSTANCES; index++) {
                const matrix = randomAffine(random, { extent: EXTENT, mirror: index % 4 === 0 });
                matrices.push(matrix);
                handles.push(scene.createInstance({ mesh, materials: [material], transform: matrix }));
            }

            /** Main-thread milliseconds of the scene copy's upload and of the tracer's update. */
            const encodeFrame = (encoder: GPUCommandEncoder, writes?: GPUComputePassTimestampWrites): { sceneMs: number; updateMs: number } => {
                const start = performance.now();
                const upload = scene.encode(encoder);
                const uploaded = performance.now();
                rays.update(encoder, upload, { computePass: () => writes });
                return { sceneMs: uploaded - start, updateMs: performance.now() - uploaded };
            };
            const untimedFrame = async (): Promise<void> => {
                const encoder = device.createCommandEncoder({ label: 'tlas-bench frame' });
                encodeFrame(encoder);
                device.queue.submit([encoder.finish()]);
                await device.queue.onSubmittedWorkDone();
            };
            // The mesh tree and the first shape, then one frame to put them in place.
            await untimedFrame();
            for (let waited = 0; rays.building; waited++) {
                assert(waited < 200, 'the top level still builds after 200 frames');
                await rays.whenBuilt();
                await untimedFrame();
            }
            assert(rays.topLevelInstanceCount === INSTANCES, `the top level holds ${rays.topLevelInstanceCount} of ${INSTANCES} instances`);
            const firstBuildMs = top.counters.lastBuildMs;
            const snapshotMs = top.counters.lastSnapshotMs;

            const moveAll = (): number => {
                const start = performance.now();
                for (let index = 0; index < INSTANCES; index++) {
                    const matrix = matrices[index];
                    for (let axis = 0; axis < 3; axis++) matrix[12 + axis] += (random() * 2 - 1) * STEP;
                    scene.setTransform(handles[index], matrix);
                }
                return performance.now() - start;
            };
            const refitsBefore = top.counters.refits;
            const shapesBefore = top.counters.shapesBuilt;
            const qualityBefore = top.counters.qualityRebuilds;
            const checksBefore = top.counters.qualityChecks;
            const gpuMs: number[] = [];
            const updateCpuMs: number[] = [];
            const sceneEncodeCpuMs: number[] = [];
            const setTransformCpuMs: number[] = [];
            for (let frame = 0; frame < WARMUP_FRAMES + TIMED_FRAMES; frame++) {
                const moveMs = moveAll();
                let cpu = { sceneMs: 0, updateMs: 0 };
                const passMs = await timer.time((encoder, writes) => {
                    cpu = encodeFrame(encoder, writes);
                });
                if (frame < WARMUP_FRAMES) continue;
                gpuMs.push(passMs);
                updateCpuMs.push(cpu.updateMs);
                sceneEncodeCpuMs.push(cpu.sceneMs);
                setTransformCpuMs.push(moveMs);
            }
            const refits = top.counters.refits - refitsBefore;
            assert(refits === WARMUP_FRAMES + TIMED_FRAMES, `expected a refit every frame, got ${refits}`);

            // Let the last stats come back, then check the tree as it stands.
            for (let extra = 0; extra < 4; extra++) {
                const encoder = device.createCommandEncoder();
                encodeFrame(encoder);
                device.queue.submit([encoder.finish()]);
                await device.queue.onSubmittedWorkDone();
            }
            const bySlot = new Map<number, CheckedInstance>(
                handles.map((handle, index) => [scene.instanceSlot(handle), { id: index, objectBox, objectToWorld: matrices[index], visible: true }]),
            );
            const tree = await checkTopLevel(device, top.buffer, bySlot);
            assert(tree.failures.length === 0, tree.failures.join('\n'));
            assert(top.counters.refitGaveUp === 0, `${top.counters.refitGaveUp} refit box reads gave up waiting (repaired)`);

            const repair = await timeForcedRepair(device, timer, matrices, objectBox);
            const result = {
                benchmark: 'tlas-bench',
                scene: SCENE,
                instances: INSTANCES,
                timingSource: timer.source,
                refitGpuMsMedian: median(gpuMs),
                refitGpuMsMin: Math.min(...gpuMs),
                refitGpuMsMax: Math.max(...gpuMs),
                refitWithRepairGpuMsMedian: repair.medianMs,
                repairGaveUpReads: repair.gaveUp,
                timedFrames: gpuMs.length,
                updateCpuMsMedian: median(updateCpuMs),
                sceneEncodeCpuMsMedian: median(sceneEncodeCpuMs),
                setTransformCpuMsMedian: median(setTransformCpuMs),
                shapeBuildMsFirst: firstBuildMs,
                shapeSnapshotMsFirst: snapshotMs,
                shapeBuildMsLast: top.counters.lastBuildMs,
                shapesBuiltWhileMoving: top.counters.shapesBuilt - shapesBefore,
                qualityRebuildsWhileMoving: top.counters.qualityRebuilds - qualityBefore,
                qualityChecksWhileMoving: top.counters.qualityChecks - checksBefore,
                snapshotMsLast: top.counters.lastSnapshotMs,
                rebuildCostRatio: TOP_LEVEL_REBUILD_COST_RATIO,
                latestRelativeCost: top.latestRelativeCost,
                slotsChecked: tree.checked,
                topLevelBytes: rays.memory.topLevel,
                subgroupMinSize: (device.adapterInfo as GPUAdapterInfo & { subgroupMinSize?: number }).subgroupMinSize ?? null,
                subgroupMaxSize: (device.adapterInfo as GPUAdapterInfo & { subgroupMaxSize?: number }).subgroupMaxSize ?? null,
                userAgent: navigator.userAgent,
                browserVersion: await fullBrowserVersion(),
            };
            log(
                `refit pass ${result.refitGpuMsMedian.toFixed(4)} ms median (${result.refitGpuMsMin.toFixed(4)}-${result.refitGpuMsMax.toFixed(4)}), ` +
                    `tracer update ${result.updateCpuMsMedian.toFixed(3)} ms CPU, scene upload ${result.sceneEncodeCpuMsMedian.toFixed(3)} ms CPU, setTransform x${INSTANCES} ${result.setTransformCpuMsMedian.toFixed(2)} ms CPU, ` +
                    `${result.qualityRebuildsWhileMoving} quality rebuilds, shape build ${firstBuildMs.toFixed(1)} ms in the worker (bar ${BAR_MS} ms on the RTX 3060)`,
            );
            return result;
        } finally {
            timer.dispose();
            rays.dispose();
            scene.dispose();
        }
    },
    { requiredFeatures: ['subgroups', 'timestamp-query'] },
);
