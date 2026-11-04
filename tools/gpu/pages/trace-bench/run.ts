/**
 * The tracer benchmark (T0.10 parts B and C): ours, on trees from three builders,
 * against three-mesh-bvh on probe-style rays, per scene.
 *
 * Per scene, in order:
 *   1. load the geometry (positions, indices, instance transforms; no textures);
 *   2. build our trees in the worker, assemble, time top-level rebuilds;
 *   3. generate the probe rays, trace them once to find the hits, and turn the
 *      hits into shadow rays toward the sun (the same rays go to every tracer);
 *   4. time ours-wt, then ours-tinybvh (tinybvh's trees, our walk), then
 *      ours-tmbuild (three-mesh-bvh's trees in our layout, our walk), then
 *      three-mesh-bvh, each on closest and any;
 *   5. compare the closest hits between tracers.
 *
 * Everything the GPU holds for a tracer is freed before the next one starts, so
 * a scene's peak is one tracer's structures plus the rays.
 */

import { createMeshBvhBuilder, type MeshBvhBuilder } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import type { AdapterSummary } from '../../contract.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import {
    BENCH_PROFILES,
    CALIBRATION_SAMPLE_RAYS,
    BenchConfigError,
    MIN_AGREEMENT_RATE,
    parseBenchConfig,
    SHADOW_RAY_OFFSET_FRACTION,
    RAY_DIRECTION_SEED,
    SYNTHETIC_SCENE_ID,
    type BenchConfig,
    type BenchProfileName,
} from './config.ts';
import { createDispatchTimer, type DispatchTimer } from './dispatch-timer.ts';
import {
    buildOursScene,
    buildTmbuildScene,
    estimateNsPerRay,
    runOursTracer,
    traceClosestOnce,
    uploadRays,
    yieldToBrowser,
    type OursRayBuffers,
    type OursTracerRun,
    type TimingPlan,
} from './ours-bench.ts';
import { makeProbeGrid, makeProbeRays, makeShadowRays, makeSphereDirections, type ShadowRays } from './rays.ts';
import type { AgreementResult, DeviceRecord, ModeResult, SceneResult, TraceBenchData } from './results.ts';
import { loadBenchScene } from './scene-source.ts';
import { compareBlocked, compareClosestDistances } from './statistics.ts';
import { runSynthetic } from './synthetic.ts';
import { createTmbvhRenderer, runTmbvh, TMBVH_TIMING_METHOD, type TmbvhRun } from './tmbvh-bench.ts';
import type { WebGPURenderer } from 'three/webgpu';

interface BenchContext {
    device: GPUDevice;
    timer: DispatchTimer;
    builder: MeshBvhBuilder;
    config: BenchConfig;
    plan: TimingPlan;
    /** Created on first use, since ours-only runs do not need three. */
    renderer: () => Promise<WebGPURenderer>;
    progress(message: string): void;
}

/**
 * Asks for the biggest storage bindings and buffers the adapter offers: Bistro's
 * triangle buffer (about 136 MB) is already over the 128 MB default, and Sponza's is not far behind.
 * Every feature the adapter has is requested because three reads `device.features` at init
 * and drops to a reduced mode without `core-features-and-limits`.
 */
async function requestBenchDevice(): Promise<{ adapter: GPUAdapter; device: GPUDevice; record: DeviceRecord }> {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('requestAdapter() returned null (no usable GPU).');
    const requiredLimits = {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
    };
    const device = await adapter.requestDevice({
        requiredFeatures: [...adapter.features] as GPUFeatureName[],
        requiredLimits,
    });
    return {
        adapter,
        device,
        record: {
            maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
            maxBufferSize: device.limits.maxBufferSize,
            maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
            maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
            timestampQuery: device.features.has('timestamp-query'),
        },
    };
}

function compareTracers(
    nameA: string,
    runA: { closestT: Float32Array; blocked: Uint32Array | null },
    nameB: string,
    runB: { closestT: Float32Array; blocked: Uint32Array | null },
): AgreementResult {
    const closest = compareClosestDistances(runA.closestT, runB.closestT);
    const any = runA.blocked && runB.blocked ? compareBlocked(runA.blocked, runB.blocked).rate : null;
    return {
        pair: `${nameA} vs ${nameB}`,
        closestRate: closest.rate,
        closestRays: closest.rays,
        hitMissMismatch: closest.hitMissMismatch,
        distanceMismatch: closest.distanceMismatch,
        anyRate: any,
    };
}

async function runScene(context: BenchContext, sceneId: string): Promise<SceneResult> {
    const { device, config, plan, progress } = context;
    progress(`${sceneId}: loading geometry`);
    const scene = await loadBenchScene(sceneId, config.volume);
    progress(`${sceneId}: ${scene.meshes.length} meshes, ${scene.instances.length} instances, ${scene.instancedTriangles} triangles (${Math.round(scene.loadMs)} ms to load)`);
    await yieldToBrowser();

    const result: SceneResult = {
        scene: sceneId,
        meshCount: scene.meshes.length,
        instanceCount: scene.instances.length,
        uniqueTriangles: scene.uniqueTriangles,
        instancedTriangles: scene.instancedTriangles,
        probeBoxSource: scene.probeBoxSource,
        probeBox: { min: scene.probeBox.min, max: scene.probeBox.max },
        probeDimensions: [],
        probeCount: 0,
        raysPerProbe: config.raysPerProbe,
        closestRays: 0,
        closestRaysRequested: 0,
        estimatedNsPerRay: null,
        closestHits: 0,
        shadowRays: 0,
        towardSun: scene.towardSun,
        sunSource: scene.sunSource,
        modelFiles: scene.modelFiles,
        loadMs: scene.loadMs,
        ours: null,
        oursTmbuild: null,
        oursTinybvh: null,
        tmbvh: null,
        modes: [],
        agreement: [],
        skipped: [],
    };

    progress(`${sceneId}: building our trees`);
    const ours = await buildOursScene(context.builder, scene.meshes, scene.instances);
    result.ours = ours.result;
    progress(
        `${sceneId}: ours built in ${Math.round(ours.result.buildMsTotal)} ms (slowest mesh ${Math.round(ours.result.buildMsMaxMesh)} ms), ` +
            `${(ours.result.bytesTotal / 1e6).toFixed(1)} MB, top level rebuild ${ours.result.topLevelRebuildMsMedian.toFixed(2)} ms`,
    );
    await yieldToBrowser();

    // The rays: one probe grid, one direction set, shared by every tracer.
    const directions = makeSphereDirections(config.raysPerProbe, RAY_DIRECTION_SEED);
    const requestedProbes = Math.max(1, Math.round(config.targetRays / config.raysPerProbe));
    let probeTarget = requestedProbes;
    if (config.maxDispatchMs > 0) {
        // Time a sample first: a dispatch longer than the OS's GPU watchdog would reset the device.
        try {
            const sampleProbes = Math.min(requestedProbes, Math.max(1, Math.ceil(CALIBRATION_SAMPLE_RAYS / config.raysPerProbe)));
            const sampleRays = makeProbeRays(makeProbeGrid(scene.probeBox, sampleProbes), directions);
            const nsPerRay = await estimateNsPerRay(device, context.timer, ours.data, sampleRays);
            result.estimatedNsPerRay = nsPerRay;
            const affordableRays = Math.floor((config.maxDispatchMs * 1e6) / nsPerRay);
            probeTarget = Math.max(1, Math.min(requestedProbes, Math.floor(affordableRays / config.raysPerProbe)));
            progress(`${sceneId}: about ${nsPerRay.toFixed(0)} ns per ray, so a ${config.maxDispatchMs} ms dispatch holds about ${affordableRays} rays`);
        } catch (error) {
            result.skipped.push(`ray-count check: ${error instanceof Error ? error.message : String(error)}`);
        }
    }
    const grid = makeProbeGrid(scene.probeBox, probeTarget);
    const closestPacked = makeProbeRays(grid, directions);
    const closestCount = grid.probeCount * config.raysPerProbe;
    result.probeDimensions = grid.dimensions;
    result.probeCount = grid.probeCount;
    result.closestRays = closestCount;
    result.closestRaysRequested = requestedProbes * config.raysPerProbe;
    result.probesRequested = requestedProbes;
    result.thinned = probeTarget < requestedProbes;
    progress(
        `${sceneId}: ${grid.probeCount} probes (${grid.dimensions.join('x')}), ${closestCount} closest-hit rays` +
            (probeTarget < requestedProbes ? ` (thinned from about ${result.closestRaysRequested} to keep one dispatch short)` : ''),
    );

    const gpuBuffers: GPUBuffer[] = [];
    const runs = new Map<string, { closestT: Float32Array; blocked: Uint32Array | null }>();
    try {
        // Trace once with our tracer to find where the rays hit, then aim shadow rays from there.
        const closestGpu = uploadRays(device, 'closest rays', closestPacked);
        gpuBuffers.push(closestGpu);
        let shadow: ShadowRays | null = null;
        try {
            const hitRecords = await traceClosestOnce(device, ours.data, closestGpu, closestCount);
            const diagonal = Math.hypot(
                scene.geometryBounds.max[0] - scene.geometryBounds.min[0],
                scene.geometryBounds.max[1] - scene.geometryBounds.min[1],
                scene.geometryBounds.max[2] - scene.geometryBounds.min[2],
            );
            shadow = makeShadowRays({
                closestRays: closestPacked,
                hitRecords,
                meshes: scene.meshes,
                instances: scene.instances,
                towardSun: scene.towardSun,
                normalOffset: diagonal * SHADOW_RAY_OFFSET_FRACTION,
            });
            result.closestHits = shadow.hitCount;
            result.shadowRays = shadow.count;
            progress(`${sceneId}: ${shadow.hitCount} of ${closestCount} rays hit; ${shadow.count} shadow rays (${shadow.facingAwayCount} hits face away from the sun)`);
        } catch (error) {
            result.skipped.push(`ours (preparing rays): ${error instanceof Error ? error.message : String(error)}`);
        }
        await yieldToBrowser();

        const shadowGpu = shadow && shadow.count > 0 ? uploadRays(device, 'shadow rays', shadow.rays) : null;
        if (shadowGpu) gpuBuffers.push(shadowGpu);
        const oursRays: OursRayBuffers = {
            closestRays: closestGpu,
            closestCount,
            shadowRays: shadowGpu,
            shadowCount: shadow?.count ?? 0,
        };

        const oursAvailable = shadow !== null;
        const finish = (tracer: string, run: OursTracerRun | TmbvhRun): void => {
            result.modes.push(...run.modes);
            runs.set(tracer, { closestT: run.closestT, blocked: run.blocked });
            for (const mode of run.modes) progress(describeMode(sceneId, mode));
        };

        for (const tracer of ['ours-wt'] as const) {
            if (!config.tracers.includes(tracer)) continue;
            if (!oursAvailable) continue;
            try {
                finish(tracer, await runOursTracer(device, context.timer, tracer, ours.data, oursRays, plan));
            } catch (error) {
                result.skipped.push(`${tracer}: ${error instanceof Error ? error.message : String(error)}`);
                progress(`${sceneId}: ${tracer} skipped (${error instanceof Error ? error.message : String(error)})`);
            }
            await yieldToBrowser();
        }

        // tinybvh's trees (WASM, built in the same worker as ours), traced by our watertight walk.
        if (config.tracers.includes('ours-tinybvh') && oursAvailable) {
            try {
                progress(`${sceneId}: building with tinybvh (WASM, in the worker)`);
                const tinybvh = await buildOursScene(context.builder, scene.meshes, scene.instances, 'tinybvh');
                result.oursTinybvh = tinybvh.result;
                progress(`${sceneId}: ours-tinybvh built in ${Math.round(tinybvh.result.buildMsTotal)} ms, ${(tinybvh.result.bytesTotal / 1e6).toFixed(1)} MB`);
                finish('ours-tinybvh', await runOursTracer(device, context.timer, 'ours-tinybvh', tinybvh.data, oursRays, plan));
            } catch (error) {
                result.skipped.push(`ours-tinybvh: ${error instanceof Error ? error.message : String(error)}`);
                progress(`${sceneId}: ours-tinybvh skipped (${error instanceof Error ? error.message : String(error)})`);
            }
            await yieldToBrowser();
        }

        // three-mesh-bvh's trees in our layout, traced by our watertight walk. Built after the others are
        // done with the GPU; the trees are dropped once traced.
        if (config.tracers.includes('ours-tmbuild') && oursAvailable) {
            try {
                progress(`${sceneId}: building with three-mesh-bvh (SAH, target leaf ${config.tmbuildLeafSize}) and converting to our layout`);
                const tmbuild = await buildTmbuildScene(scene.meshes, scene.instances, config.tmbuildLeafSize);
                result.oursTmbuild = tmbuild.result;
                progress(
                    `${sceneId}: ours-tmbuild built in ${Math.round(tmbuild.result.buildMsTotal)} ms ` +
                        `(three-mesh-bvh ${Math.round(tmbuild.result.tmbvhBuildMsTotal)} ms + converting ${Math.round(tmbuild.result.convertMsTotal)} ms), ` +
                        `${(tmbuild.result.bytesTotal / 1e6).toFixed(1)} MB`,
                );
                finish('ours-tmbuild', await runOursTracer(device, context.timer, 'ours-tmbuild', tmbuild.data, oursRays, plan));
            } catch (error) {
                result.skipped.push(`ours-tmbuild: ${error instanceof Error ? error.message : String(error)}`);
                progress(`${sceneId}: ours-tmbuild skipped (${error instanceof Error ? error.message : String(error)})`);
            }
            await yieldToBrowser();
        }

        if (config.tracers.includes('tmbvh')) {
            try {
                progress(`${sceneId}: three-mesh-bvh building (SAH, leaf ${config.tmbvhLeafSize})`);
                const run = await runTmbvh(
                    await context.renderer(),
                    scene.meshes,
                    scene.instances,
                    {
                        closestRays: closestPacked,
                        closestCount,
                        shadowRays: shadow && shadow.count > 0 ? shadow.rays : null,
                        shadowCount: shadow?.count ?? 0,
                    },
                    plan,
                    config.tmbvhLeafSize,
                );
                result.tmbvh = run.build;
                progress(
                    `${sceneId}: three-mesh-bvh built in ${Math.round(run.build.buildMsTotal)} ms (+${Math.round(run.build.dataUpdateMs)} ms packing), ` +
                        `${(run.build.bytesTotal / 1e6).toFixed(1)} MB`,
                );
                finish('tmbvh', run);
            } catch (error) {
                result.skipped.push(`tmbvh: ${error instanceof Error ? error.message : String(error)}`);
                progress(`${sceneId}: tmbvh skipped (${error instanceof Error ? error.message : String(error)})`);
            }
        }
    } finally {
        for (const buffer of gpuBuffers) buffer.destroy();
    }

    const pairs: [string, string][] = [
        ['ours-wt', 'tmbvh'],
        ['ours-wt', 'ours-tinybvh'],
        ['ours-tinybvh', 'tmbvh'],
        ['ours-wt', 'ours-tmbuild'],
        ['ours-tmbuild', 'tmbvh'],
    ];
    for (const [a, b] of pairs) {
        const runA = runs.get(a);
        const runB = runs.get(b);
        if (runA && runB) result.agreement.push(compareTracers(a, runA, b, runB));
    }
    for (const agreement of result.agreement) {
        progress(
            `${sceneId}: ${agreement.pair} closest-hit agreement ${(agreement.closestRate * 100).toFixed(3)}%` +
                (agreement.anyRate === null ? '' : `, any-hit ${(agreement.anyRate * 100).toFixed(3)}%`),
        );
    }
    return result;
}

function describeMode(sceneId: string, mode: ModeResult): string {
    return (
        `${sceneId}: ${mode.tracer} ${mode.mode}: ${mode.timing.medianMs.toFixed(3)} ms median over ${mode.timing.samplesMs.length}, ` +
        `${mode.mraysPerSecond.toFixed(1)} Mrays/s (${mode.timingSource})`
    );
}

/** The user-agent string freezes the version at 156.0.0.0; the client hints still know the real one. */
async function fullBrowserVersion(): Promise<string | null> {
    const hints = (
        navigator as unknown as {
            userAgentData?: { getHighEntropyValues(hints: string[]): Promise<{ fullVersionList?: { brand: string; version: string }[] }> };
        }
    ).userAgentData;
    if (!hints) return null;
    try {
        const { fullVersionList } = await hints.getHighEntropyValues(['fullVersionList']);
        return fullVersionList?.find((entry) => entry.brand === 'Google Chrome' || entry.brand === 'Chromium')?.version ?? null;
    } catch {
        return null;
    }
}

export function newBenchData(config: BenchConfig, record: DeviceRecord, browserVersion: string | null): TraceBenchData {
    return {
        benchmark: 'trace-bench',
        profile: config.profile,
        raysPerProbe: config.raysPerProbe,
        targetRays: config.targetRays,
        warmupDispatches: config.warmupDispatches,
        timedDispatches: config.timedDispatches,
        tracers: config.tracers,
        userAgent: navigator.userAgent,
        browserVersion,
        device: record,
        tmbvhTimingMethod: TMBVH_TIMING_METHOD,
        scenes: [],
        synthetic: [],
    };
}

/**
 * Runs the whole benchmark for `config` on `device`, adding each scene's result
 * to `data` as it finishes, so a failure late in the run still reports what
 * came before.
 */
export async function runTraceBench(
    device: GPUDevice,
    data: TraceBenchData,
    config: BenchConfig,
    progress: (message: string) => void,
): Promise<void> {
    const timer = await createDispatchTimer(device);
    const builder = createMeshBvhBuilder({ worker: 'auto' });
    const shared: { renderer: WebGPURenderer | null } = { renderer: null };
    const context: BenchContext = {
        device,
        timer,
        builder,
        config,
        plan: { warmups: config.warmupDispatches, timed: config.timedDispatches },
        renderer: async () => (shared.renderer ??= await createTmbvhRenderer(device)),
        progress,
    };

    try {
        for (const sceneId of config.scenes) {
            if (sceneId === SYNTHETIC_SCENE_ID) {
                progress(`${SYNTHETIC_SCENE_ID}: timing top-level rebuilds`);
                const synthetic = await runSynthetic();
                data.synthetic.push(synthetic);
                progress(`${SYNTHETIC_SCENE_ID}: ${synthetic.instanceCount} instances, rebuild ${synthetic.topLevelRebuildMsMedian.toFixed(2)} ms median`);
                continue;
            }
            data.scenes.push(await runScene(context, sceneId));
            await yieldToBrowser();
        }
    } finally {
        builder.dispose();
        timer.dispose();
        if (shared.renderer) await shared.renderer.dispose();
    }
}

/** What each tracer must agree on before a speed number means anything. */
export function disagreements(data: TraceBenchData): string[] {
    const problems: string[] = [];
    for (const scene of data.scenes) {
        for (const agreement of scene.agreement) {
            if (agreement.closestRate < MIN_AGREEMENT_RATE) {
                problems.push(
                    `${scene.scene}: ${agreement.pair} agree on only ${(agreement.closestRate * 100).toFixed(2)}% of closest hits ` +
                        `(${agreement.hitMissMismatch} hit/miss, ${agreement.distanceMismatch} distance mismatches); that is a bug, not a speed result`,
                );
            }
        }
    }
    return problems;
}

/**
 * Entry point for a page folder. `defaults` say what the folder runs when the
 * URL has no `scene=` or `profile=`. `query` adds default parameters (for
 * example `rays=16`) that the real URL can still override. The test runner
 * cannot add parameters to a page URL, which is why each scene has a folder.
 */
export async function runTraceBenchPage(defaults: { scenes: string[]; profile: BenchProfileName; query?: string }): Promise<void> {
    const startTime = performance.now();
    const lines: string[] = [];
    const progress = (message: string): void => {
        lines.push(message);
        console.info(`[trace-bench] ${message}`);
    };

    const asyncProblems: string[] = [];
    window.addEventListener('error', (event) => asyncProblems.push(`Uncaught error: ${event.message}`));
    window.addEventListener('unhandledrejection', (event) => asyncProblems.push(`Unhandled rejection: ${String(event.reason)}`));

    let deviceErrors: string[] = [];
    let adapterSummary: AdapterSummary | null = null;
    let device: GPUDevice | null = null;
    let data: TraceBenchData | null = null;
    let status: 'pass' | 'fail' = 'pass';
    let message: string | undefined;
    try {
        const search = new URLSearchParams(defaults.query ?? '');
        for (const [name, value] of new URLSearchParams(location.search)) search.set(name, value);
        const config = parseBenchConfig(search.toString(), defaults);
        const requested = await requestBenchDevice();
        device = requested.device;
        deviceErrors = watchDevice(device);
        adapterSummary = summarizeAdapterInfo(device.adapterInfo, device.features);
        progress(
            `profile ${config.profile}: ${config.raysPerProbe} rays per probe, about ${config.targetRays} rays, ` +
                `${config.warmupDispatches} warm-up + ${config.timedDispatches} timed dispatches; scenes ${config.scenes.join(', ')}; tracers ${config.tracers.join(', ')}`,
        );
        progress(
            `device limits: maxStorageBufferBindingSize ${requested.record.maxStorageBufferBindingSize}, maxBufferSize ${requested.record.maxBufferSize}, timestamps ${requested.record.timestampQuery}`,
        );
        data = newBenchData(config, requested.record, await fullBrowserVersion());
        await runTraceBench(device, data, config, progress);
        const problems = disagreements(data);
        if (problems.length > 0) {
            status = 'fail';
            message = problems.join(' | ');
        }
        await drainDevice(device);
    } catch (error) {
        status = 'fail';
        message = error instanceof BenchConfigError || error instanceof Error ? error.message : String(error);
    }
    if (status === 'pass' && asyncProblems.length > 0) {
        status = 'fail';
        message = asyncProblems.join(' | ');
    }
    reportGpuTest({
        status,
        message,
        data: (data ?? { benchmark: 'trace-bench', profiles: Object.keys(BENCH_PROFILES) }) as unknown as Record<string, unknown>,
        adapter: adapterSummary,
        deviceErrors,
        startTime,
        logLines: lines,
    });
}
