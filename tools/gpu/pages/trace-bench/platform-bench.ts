/**
 * The `platform` tracer of the benchmark: the same geometry and rays, but
 * through the platform object (T1.3): the scene API (`createMesh`,
 * `createMaterial`, `createInstance`), `prepare`, one `encode` per frame until
 * every mesh tree has arrived from the worker, then the `rays` backend's
 * `encodeBatch`.
 *
 * It is not quite a host's path: it constructs `GIPlatformImpl` (the class
 * `createGI` returns) to reach internals a host can't, namely
 * `rays.whenBuilt()` to wait for the trees, the batch's `timestampWrites`, and
 * the per-buffer memory. A technique gets the same `rays` object as
 * `context.queries.rays`. It shows that the platform path traces as fast as
 * the low-level one (`ours-tinybvh`, the same trees and the same WGSL), and it
 * reports the platform's memory: the scene copy and the acceleration buffers.
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import type { TracerId } from './config.ts';
import type { DispatchTimer } from './dispatch-timer.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import {
    probeGridRun,
    probeGridSets,
    timeRaySets,
    yieldToBrowser,
    type OursRayBuffers,
    type OursTracerRun,
    type RaySetBuffer,
    type RaySetRun,
    type TimingPlan,
} from './ours-bench.ts';
import type { PlatformBuildResult } from './results.ts';

const TRACER: TracerId = 'platform';

/** A frame with nothing for techniques to do: the platform only uploads and builds. */
const IDLE_FRAME: FrameInput = {
    deltaTime: 0,
    budget: { units: 0 },
    camera: {
        position: [0, 0, 0],
        view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        projection: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        near: 0.1,
        far: 100,
    },
};

function encodeFrame(device: GPUDevice, platform: GIPlatformImpl): void {
    const encoder = device.createCommandEncoder({ label: 'platform frame' });
    platform.encode(encoder, IDLE_FRAME);
    device.queue.submit([encoder.finish()]);
}

export interface PlatformRun extends OursTracerRun {
    build: PlatformBuildResult;
}

export interface PlatformSetsRun {
    sets: RaySetRun[];
    build: PlatformBuildResult;
}

/**
 * Describes the scene through the platform, waits for the trees, then times
 * each set through `rays.encodeBatch` (see `timeRaySets`). Disposes the
 * platform before returning.
 */
export async function runPlatformTracerOnSets(
    device: GPUDevice,
    timer: DispatchTimer,
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
    sets: readonly RaySetBuffer[],
    plan: TimingPlan,
): Promise<PlatformSetsRun> {
    const platform = new GIPlatformImpl(device, { label: 'bench platform', rayBatches: true });
    try {
        const start = performance.now();
        const material = platform.scene.createMaterial({ albedo: [0.8, 0.8, 0.8] });
        const meshHandles = meshes.map((mesh) => platform.scene.createMesh({ positions: mesh.positions, indices: mesh.indices }));
        // The instance's slot is its position here, the same instance index the other tracers report.
        instances.forEach((instance) =>
            platform.scene.createInstance({ mesh: meshHandles[instance.meshIndex], materials: [material], transform: instance.objectToWorld }),
        );
        const describeMs = performance.now() - start;

        await platform.prepare();
        encodeFrame(device, platform); // uploads the scene; the worker starts building
        await platform.rays.whenBuilt();
        encodeFrame(device, platform); // places the trees and builds the top level
        await device.queue.onSubmittedWorkDone();
        const readyMs = performance.now() - start;
        if (platform.rays.topLevelInstanceCount !== instances.length) {
            throw new Error(`the platform's top level holds ${platform.rays.topLevelInstanceCount} of ${instances.length} instances`);
        }
        await yieldToBrowser();

        const runs = await timeRaySets(device, timer, TRACER, sets, plan, (encoder, set, results, timestampWrites) =>
            platform.rays.encodeBatch(encoder, { mode: set.mode, rays: set.rays, hits: results, count: set.count }, timestampWrites),
        );

        const stats = platform.stats;
        const acceleration = platform.rays.memory;
        return {
            sets: runs,
            build: {
                describeMs,
                readyMs,
                usesWorker: platform.rays.usesWorker,
                bytesSceneCopy: stats.memory.sceneCopy,
                bytesAcceleration: stats.memory.acceleration,
                bytesNodes: acceleration.nodes,
                bytesTriangles: acceleration.triangles,
                bytesTopLevel: acceleration.topLevel,
            },
        };
    } finally {
        platform.dispose();
    }
}

/**
 * Describes the scene through the platform, waits for the trees, then times
 * closest and any on the shared rays. Disposes the platform before returning.
 */
export async function runPlatformTracer(
    device: GPUDevice,
    timer: DispatchTimer,
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
    rays: OursRayBuffers,
    plan: TimingPlan,
): Promise<PlatformRun> {
    const run = await runPlatformTracerOnSets(device, timer, meshes, instances, probeGridSets(rays), plan);
    return { ...probeGridRun(run.sets), build: run.build };
}
