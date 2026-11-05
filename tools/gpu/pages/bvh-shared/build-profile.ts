/**
 * Profile of a GPU builder (B1 LBVH, B2 H-PLOC): where its build time goes, on
 * a real scene. Used by the bvh-lbvh-profile and bvh-hploc-profile pages.
 *
 * Loads a scene's meshes as the trace bench does (`?scene=`, default sponza),
 * then times the batch build three ways, each a median of a few runs:
 *
 * - whole: one compute pass, as the trace bench times it;
 * - by step: each step (for every mesh at once) in its own timed pass (the
 *   extra passes add a little time, so the sum is above "whole");
 * - by size: the largest mesh alone, and every other mesh together.
 *
 * Not a recorded number; the trace-bench-lbvh and trace-bench-hploc pages are.
 */

import { GpuHplocBuilder, HPLOC_STAGES } from '../../../../packages/bvh/src/gpu-hploc.ts';
import { GpuLbvhBuilder, LBVH_STAGES, type GpuLbvhGeometry } from '../../../../packages/bvh/src/gpu-lbvh.ts';
import { loadBenchScene } from '../trace-bench/scene-source.ts';
import { median } from '../trace-bench/statistics.ts';
import { assert, runGpuTest } from '../../client.ts';

/** What the profile needs of a batch; both builders' batches have it. */
interface ProfiledBatch {
    encode(encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites, stageTimestamps?: (stage: string) => GPUComputePassTimestampWrites): void;
    finish(): Promise<unknown>;
    destroy(): void;
}

interface ProfiledBuilder {
    readonly stages: readonly string[];
    prepare(): Promise<void>;
    createBatch(geometries: readonly GpuLbvhGeometry[]): ProfiledBatch;
}

function profiledBuilder(device: GPUDevice, kind: 'lbvh' | 'hploc'): ProfiledBuilder {
    const builder = kind === 'lbvh' ? new GpuLbvhBuilder(device) : new GpuHplocBuilder(device);
    return {
        stages: kind === 'lbvh' ? LBVH_STAGES : HPLOC_STAGES,
        prepare: () => builder.prepare(),
        createBatch: (geometries) => builder.createBatch(geometries) as ProfiledBatch,
    };
}

const RUNS = 5;

/** Times `record` (which gets timestamp writes for any number of passes) on the GPU; returns ms per pass. */
async function timePasses(device: GPUDevice, passCount: number, record: (encoder: GPUCommandEncoder, writes: (pass: number) => GPUComputePassTimestampWrites) => void): Promise<number[]> {
    const querySet = device.createQuerySet({ type: 'timestamp', count: passCount * 2 });
    const resolve = device.createBuffer({ size: passCount * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: passCount * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    record(encoder, (pass) => ({ querySet, beginningOfPassWriteIndex: pass * 2, endOfPassWriteIndex: pass * 2 + 1 }));
    encoder.resolveQuerySet(querySet, 0, passCount * 2, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, readback, 0, passCount * 16);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const stamps = new BigUint64Array(readback.getMappedRange().slice(0));
    readback.destroy();
    resolve.destroy();
    querySet.destroy();
    return Array.from({ length: passCount }, (_, pass) => Number(stamps[pass * 2 + 1] - stamps[pass * 2]) / 1e6);
}

async function timeWhole(device: GPUDevice, batch: ProfiledBatch): Promise<number> {
    const samples: number[] = [];
    for (let run = 0; run <= RUNS; run++) {
        const [ms] = await timePasses(device, 1, (encoder, writes) => batch.encode(encoder, writes(0)));
        if (run > 0) samples.push(ms);
    }
    return median(samples);
}

export async function runBuildProfile(kind: 'lbvh' | 'hploc'): Promise<void> {
    await runGpuTest(
        async ({ device, log }) => {
            assert(device.features.has('timestamp-query'), 'the profile needs timestamp-query');
            const sceneId = new URLSearchParams(location.search).get('scene') ?? 'sponza';
            const scene = await loadBenchScene(sceneId, 'manifest');
            const geometries = scene.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices }));
            const triangles = geometries.map((geometry) => geometry.indices.length / 3);
            log(`${sceneId}: ${geometries.length} meshes, ${triangles.reduce((a, b) => a + b, 0)} triangles, largest ${Math.max(...triangles)}`);

            const builder = profiledBuilder(device, kind);
            await builder.prepare();
            const batch = builder.createBatch(geometries);
            const wholeMs = await timeWhole(device, batch);
            log(`whole: ${wholeMs.toFixed(2)} ms`);

            const perStage: Record<string, number[]> = {};
            for (let run = 0; run <= RUNS; run++) {
                const order: string[] = [];
                const times = await timePasses(device, builder.stages.length, (encoder, writes) =>
                    batch.encode(encoder, undefined, (stage) => {
                        order.push(stage);
                        return writes(order.length - 1);
                    }),
                );
                if (run === 0) continue;
                const sums: Record<string, number> = {};
                order.forEach((stage, pass) => (sums[stage] = (sums[stage] ?? 0) + times[pass]));
                for (const [stage, ms] of Object.entries(sums)) (perStage[stage] ??= []).push(ms);
            }
            const byStage = Object.fromEntries(Object.entries(perStage).map(([stage, samples]) => [stage, Number(median(samples).toFixed(3))]));
            log(`by step: ${JSON.stringify(byStage)}`);
            await batch.finish();

            const largest = triangles.indexOf(Math.max(...triangles));
            const sizes: Record<string, number> = {};
            for (const [name, subset] of [
                ['largest mesh', [geometries[largest]]],
                ['all but the largest', geometries.filter((_, index) => index !== largest)],
            ] as const) {
                const part = builder.createBatch(subset);
                sizes[name] = Number((await timeWhole(device, part)).toFixed(3));
                part.destroy();
            }
            log(`by size: ${JSON.stringify(sizes)}`);
            return { scene: sceneId, meshes: geometries.length, wholeMs: Number(wholeMs.toFixed(3)), byStage, sizes };
        },
        { optionalFeatures: ['subgroups', 'timestamp-query'], requiredLimits: { maxStorageBufferBindingSize: 1 << 30, maxBufferSize: 1 << 30 } },
    );
}
