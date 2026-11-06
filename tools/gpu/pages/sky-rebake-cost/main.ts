/**
 * T2.6a check 2 (docs/design/m2-lights.md §6.2, "Estimated GPU cost under
 * 0.1 ms per re-bake; CPU under 0.05 ms"): what re-baking a procedural sky
 * costs, as time of day re-bakes it.
 *
 * - **GPU:** two-room with DDGI updating every probe every frame, plus a
 *   page-only busy pass (BUSY_ITERATIONS of arithmetic per thread), GPU-bound
 *   (frames submitted back to back, the GPU waited for every
 *   FRAMES_IN_FLIGHT frames), so the clocks are up as Budget v2's
 *   measurements are (Dennis 2026-11-05). Two-room alone is too light to
 *   keep this GPU busy. Every frame the sun turns 0.1
 *   degrees (more than the 0.05-degree re-bake threshold), so every frame
 *   re-bakes; the median of the `platform/sky bake` pass over BAKE_FRAMES
 *   frames must be at most 0.1 ms. Its p90 and largest are reported (other
 *   programs' GPU work lands in some frames, docs/NOTES.md). A control run of
 *   the same frames with the sun still must bake nothing.
 * - **CPU:** what a re-bake costs the platform's `encode`: the sky's own
 *   `encode` called CPU_BAKES times in a row on one encoder (a sun turned
 *   0.1 degrees each time, so each is a bake), the mean per bake, which must
 *   be at most 0.05 ms. Timed in bulk because the page's clock is coarse
 *   (browsers round `performance.now()` to 0.1 ms without cross-origin
 *   isolation). Also reported: `setEnvironment` plus the whole
 *   `platform.encode` per frame, with and without a bake (frame means).
 *
 * Manual, under the bench lock (a timing page):
 *   node tools/reference/bench-lock.ts run --what "T2.6a re-bake cost" -- pnpm test:gpu --filter sky-rebake-cost --gpu nvidia --json <file>
 */

import { assert, runGpuTest } from '../../client.ts';
import type { EnvironmentDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { PlatformSky, SKY_BAKE_PASS } from '../../../../packages/ddgi-core/src/platform/sky/platform-sky.ts';
import { SKY_HARMONICS_LAYOUT } from '../../../../packages/ddgi-core/src/platform/sky/projection-wgsl.ts';

import { createDdgiScene, frameInput, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';

const GPU_BAR_MS = 0.1;
const CPU_BAR_MS = 0.05;
/** Frames before measuring, so the clocks are up and the field has started. */
const WARM_FRAMES = 120;
const BAKE_FRAMES = 600;
const FRAMES_IN_FLIGHT = 2;
const CPU_BAKES = 2000;
/** More than the 0.05-degree threshold, so every step re-bakes. */
const STEP_DEGREES = 0.1;
const SUN_ELEVATION_DEGREES = 35;
/** The busy pass: workgroups of 64 threads, and multiply-adds per thread (about 1 ms on the RTX 3060). */
const BUSY_WORKGROUPS = 1024;
const BUSY_ITERATIONS = 40000;

const BUSY_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> sink: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    var value = f32(id.x) * 1.0e-6;
    for (var index = 0u; index < ${BUSY_ITERATIONS}u; index++) {
        value = fma(value, 0.999999, 1.0e-7);
    }
    sink[id.x] = value;
}
`;

type Vec3 = [number, number, number];

function sunAt(elevationDeg: number, azimuthDeg: number): Vec3 {
    const e = (elevationDeg * Math.PI) / 180;
    const a = (azimuthDeg * Math.PI) / 180;
    return [Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)];
}

const skyAt = (azimuthDeg: number): EnvironmentDescriptor => ({ type: 'procedural', sunDirection: sunAt(SUN_ELEVATION_DEGREES, azimuthDeg), turbidity: 3, sunIlluminance: 3 });

function quantile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))] ?? Number.NaN;
}

const mean = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const results: Record<string, unknown> = {};
        const scene = await createDdgiScene(device, 'two-room', {}, { gpuTiming: true });
        try {
            const platform = scene.platform;
            const units = scene.ddgi.field!.grid.probeCount;
            results.unitsPerFrame = units;
            platform.scene.setEnvironment(skyAt(0));

            const sink = device.createBuffer({ label: 'busy pass sink', size: BUSY_WORKGROUPS * 64 * 4, usage: GPUBufferUsage.STORAGE });
            const busyPipeline = await device.createComputePipelineAsync({ label: 'busy pass', layout: 'auto', compute: { module: device.createShaderModule({ code: BUSY_WGSL }), entryPoint: 'main' } });
            const busyGroup = device.createBindGroup({ layout: busyPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: sink } }] });
            /** One frame: the platform's encode (timed on the CPU), then the busy pass, submitted together. */
            const frame = (): number => {
                const encoder = device.createCommandEncoder({ label: 'rebake cost frame' });
                const start = performance.now();
                platform.encode(encoder, frameInput(units));
                const cpuMs = performance.now() - start;
                const busy = encoder.beginComputePass({ label: 'busy pass' });
                busy.setPipeline(busyPipeline);
                busy.setBindGroup(0, busyGroup);
                busy.dispatchWorkgroups(BUSY_WORKGROUPS);
                busy.end();
                device.queue.submit([encoder.finish()]);
                return cpuMs;
            };

            /** Frames back to back; per frame the sky (or not), the encode's CPU ms, and the bake pass's GPU ms once timed. */
            const run = async (frames: number, turnSun: boolean, firstAzimuth: number): Promise<{ bakes: number; bakeMs: number[]; encodeCpuMs: number[] }> => {
                const bakesBefore = platform.shade.sky.projections;
                const bakeMs = new Map<number, number>();
                const encodeCpuMs: number[] = [];
                for (let index = 0; index < frames; index++) {
                    const start = performance.now();
                    if (turnSun) platform.scene.setEnvironment(skyAt(firstAzimuth + STEP_DEGREES * (index + 1)));
                    const setMs = performance.now() - start;
                    encodeCpuMs.push(setMs + frame());
                    // The stats hold the newest timed frame; with frames in flight that is every other one.
                    for (const pass of platform.stats.passes) if (pass.name === SKY_BAKE_PASS && pass.gpuMs !== undefined && pass.gpuMs >= 0) bakeMs.set(pass.frame, pass.gpuMs);
                    if (index % FRAMES_IN_FLIGHT === FRAMES_IN_FLIGHT - 1) await device.queue.onSubmittedWorkDone();
                }
                await device.queue.onSubmittedWorkDone();
                return { bakes: platform.shade.sky.projections - bakesBefore, bakeMs: [...bakeMs.values()], encodeCpuMs };
            };

            await run(WARM_FRAMES, true, 0);
            const still = await run(BAKE_FRAMES / 2, false, 0);
            const turning = await run(BAKE_FRAMES, true, WARM_FRAMES * STEP_DEGREES);
            assert(still.bakes === 0, `the sky baked ${still.bakes} times with the sun still`);
            assert(turning.bakes === BAKE_FRAMES, `the sky baked ${turning.bakes} times in ${BAKE_FRAMES} frames of a turning sun (expected every frame)`);
            assert(turning.bakeMs.length >= BAKE_FRAMES / 4, `only ${turning.bakeMs.length} bake passes were timed`);
            sink.destroy();
            const gpu = { median: quantile(turning.bakeMs, 0.5), p90: quantile(turning.bakeMs, 0.9), largest: Math.max(...turning.bakeMs), timed: turning.bakeMs.length };
            results.gpuMs = gpu;
            results.encodeCpuMs = { stillFrameMean: mean(still.encodeCpuMs), bakeFrameMean: mean(turning.encodeCpuMs) };
            log(
                `GPU: the sky bake pass median ${gpu.median.toFixed(4)} ms (bar ${GPU_BAR_MS}), p90 ${gpu.p90.toFixed(4)}, largest ${gpu.largest.toFixed(4)}, over ${gpu.timed} timed bakes; ` +
                    `setEnvironment + platform.encode per frame: still ${mean(still.encodeCpuMs).toFixed(4)} ms, baking ${mean(turning.encodeCpuMs).toFixed(4)} ms`,
            );

            // CPU: the sky's own encode, CPU_BAKES bakes in a row on a stand-alone sky.
            const sky = new PlatformSky(device, { label: 'cpu cost' });
            await sky.prepare();
            const harmonics = device.createBuffer({ label: 'cpu cost harmonics', size: SKY_HARMONICS_LAYOUT.byteLength, usage: GPUBufferUsage.STORAGE });
            const environments = Array.from({ length: CPU_BAKES }, (_, index) => skyAt(STEP_DEGREES * index));
            // One bake first, so the bind group the rest reuse exists (as in a running platform).
            const first = device.createCommandEncoder();
            sky.encode(first, environments[0], 0, { buffer: harmonics });
            device.queue.submit([first.finish()]);
            const encoder = device.createCommandEncoder({ label: 'cpu cost bakes' });
            const start = performance.now();
            for (let index = 1; index < CPU_BAKES; index++) sky.encode(encoder, environments[index], index, { buffer: harmonics });
            const perBake = (performance.now() - start) / (CPU_BAKES - 1);
            const bakes = sky.projections;
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            sky.dispose();
            harmonics.destroy();
            assert(bakes === CPU_BAKES, `the stand-alone sky baked ${bakes} of ${CPU_BAKES}`);
            results.cpuMsPerBake = perBake;
            log(`CPU: ${perBake.toFixed(4)} ms per bake (the sky's encode, mean of ${CPU_BAKES - 1}; bar ${CPU_BAR_MS})`);

            const failures: string[] = [];
            if (!(gpu.median <= GPU_BAR_MS)) failures.push(`the bake's GPU median is ${gpu.median.toFixed(4)} ms (bar ${GPU_BAR_MS})`);
            if (!(perBake <= CPU_BAR_MS)) failures.push(`a bake costs ${perBake.toFixed(4)} ms of CPU (bar ${CPU_BAR_MS})`);
            assert(failures.length === 0, failures.join('; '));
            return results;
        } finally {
            scene.platform.dispose();
        }
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
