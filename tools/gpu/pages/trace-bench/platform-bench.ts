/**
 * The `platform` tracer of the benchmark: the same geometry and rays, but
 * through the platform API a host uses (T1.3). `createGI(device)`, then
 * `scene.createMesh` / `createMaterial` / `createInstance`, one `encode` per
 * frame until every mesh tree has arrived from the worker, then
 * `queries.rays.encodeBatch`. It shows that the platform path traces as fast as
 * the low-level one (`ours-tinybvh`, the same trees and the same WGSL), and it
 * reports the platform's memory: the scene copy and the acceleration buffers.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { HIT_RECORD_BYTES } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import type { TracerId } from './config.ts';
import { measureDispatches, type DispatchTimer } from './dispatch-timer.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import { readBuffer, extractHitDistances, yieldToBrowser, type OursRayBuffers, type OursTracerRun, type TimingPlan } from './ours-bench.ts';
import type { ModeResult, PlatformBuildResult, TraceMode } from './results.ts';
import { megaRaysPerSecond } from './statistics.ts';

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
    const platform = new GIPlatformImpl(device, { label: 'bench platform' });
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

        const closestHits = device.createBuffer({
            label: 'platform hits',
            size: rays.closestCount * HIT_RECORD_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const modes: ModeResult[] = [];
        const summarize = (mode: TraceMode, count: number, timing: Awaited<ReturnType<typeof measureDispatches>>): ModeResult => ({
            tracer: TRACER,
            mode,
            rays: count,
            timing,
            timingSource: timer.source,
            mraysPerSecond: megaRaysPerSecond(count, timing.medianMs),
        });
        try {
            const closestTiming = await measureDispatches(
                () =>
                    timer.time((encoder, timestampWrites) =>
                        platform.rays.encodeBatch(encoder, { mode: 'closest', rays: rays.closestRays, hits: closestHits, count: rays.closestCount }, timestampWrites),
                    ),
                plan.warmups,
                plan.timed,
            );
            modes.push(summarize('closest', rays.closestCount, closestTiming));
            const hitRecords = await readBuffer(device, closestHits, rays.closestCount * HIT_RECORD_BYTES);

            let blocked: Uint32Array | null = null;
            if (rays.shadowRays && rays.shadowCount > 0) {
                const shadowRays = rays.shadowRays;
                const occlusion = device.createBuffer({
                    label: 'platform occlusion',
                    size: rays.shadowCount * 4,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
                });
                try {
                    const anyTiming = await measureDispatches(
                        () =>
                            timer.time((encoder, timestampWrites) =>
                                platform.rays.encodeBatch(encoder, { mode: 'any', rays: shadowRays, hits: occlusion, count: rays.shadowCount }, timestampWrites),
                            ),
                        plan.warmups,
                        plan.timed,
                    );
                    modes.push(summarize('any', rays.shadowCount, anyTiming));
                    blocked = new Uint32Array(await readBuffer(device, occlusion, rays.shadowCount * 4));
                } finally {
                    occlusion.destroy();
                }
            }

            const stats = platform.stats;
            const acceleration = platform.rays.memory;
            return {
                modes,
                hitRecords,
                closestT: extractHitDistances(hitRecords),
                blocked,
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
            closestHits.destroy();
        }
    } finally {
        platform.dispose();
    }
}
