/**
 * radiance-cache-mac-limits (pmndrs/gi#49): the radiance cache on a device
 * with at most 10 storage buffers per shader stage, what Chrome gives on
 * Apple GPUs (the M5 Pro). WebGPU enforces the limits a device was created
 * with, so on the RTX this stands for the Mac:
 *
 * 1. The device has 10, and a bind group layout with 11 storage buffers is
 *    refused (the cap is enforced, not only reported).
 * 2. The platform reports the cache's need (`limitNeeds`: 10).
 * 3. Every configuration that changes what the cache's kernels bind
 *    prepares and runs four frames with no validation error, and its
 *    irradiance is finite and lit: the default (GI-1.2's hash grid),
 *    `worldCache: 'sharc'`, `cellShading: 'budgeted'`, `secondBounce:
 *    'gi1-rays'`, `'fallback'` with DDGI stacked under it, `'shared-cell'`
 *    (pmndrs/gi#54), `neuralCache:
 *    'tail'` (where this checkout has the optional network; it says so in
 *    the data otherwise), SSGI on, the denoiser off, and `useMultibounce`
 *    off at prepare (register D67: the second-bounce passes aren't made and
 *    the rest are built without their indirect lines).
 *
 * The scene is built in: a floor, a wall and a box under a constant sky with
 * a sun, so it needs no downloaded assets. Correctness only; the analytic
 * gates are the other radiance-cache pages' (run them under
 * `pnpm test:gpu --max-storage-buffers 10` for the Mac's limit there too).
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi/src/ddgi-technique.ts';
import { RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE, RadianceCacheTechnique } from '../../../../packages/cache/src/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/cache/src/settings.ts';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { PageScreen, frameCamera, type PageCamera } from '../radiance-cache-probes/screen.ts';

/** Apple GPUs' storage buffers per stage in Chrome (M5 Pro, 2026-11-26). */
const APPLE_STORAGE_BUFFERS_PER_STAGE = 10;
const WIDTH = 640;
const HEIGHT = 360;
const FRAMES = 4;
const CAMERA: PageCamera = { position: [0, 2, 4], target: [0, 1, -2], fovYDeg: 60, near: 0.1, far: 500 };
const SKY = 1.2;

interface Configuration {
    readonly name: string;
    readonly settings: Partial<RadianceCacheSettings>;
    readonly ddgiBelow?: boolean;
}

const CONFIGURATIONS: readonly Configuration[] = [
    { name: 'default', settings: {} },
    { name: "worldCache 'sharc'", settings: { worldCache: 'sharc' } },
    { name: "cellShading 'budgeted'", settings: { cellShading: 'budgeted' } },
    { name: "secondBounce 'gi1-rays'", settings: { secondBounce: 'gi1-rays' } },
    { name: "secondBounce 'fallback', DDGI below", settings: { secondBounce: 'fallback' }, ddgiBelow: true },
    // pmndrs/gi#54: cells lit on the budget though cellShading is left per hit, and its two traced passes.
    { name: "secondBounce 'shared-cell'", settings: { secondBounce: 'shared-cell' } },
    { name: "neuralCache 'tail'", settings: { neuralCache: 'tail', secondBounce: 'gi1-rays' } },
    { name: 'ssgi on', settings: { ssgi: true } },
    { name: 'denoiser off', settings: { denoiser: false } },
    { name: 'useMultibounce off at prepare', settings: { useMultibounce: false } },
];

function quad(corners: readonly (readonly [number, number, number])[], normal: readonly [number, number, number]): { positions: Float32Array; normals: Float32Array; indices: Uint32Array } {
    return {
        positions: new Float32Array(corners.flat()),
        normals: new Float32Array([...normal, ...normal, ...normal, ...normal]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** A floor, a back wall and a box's top and front under a constant sky and a sun. */
function describeScene(platform: GIPlatformImpl): void {
    const scene = platform.scene;
    const material = scene.createMaterial({ albedo: [0.6, 0.6, 0.6], label: 'grey' });
    const add = (label: string, mesh: ReturnType<typeof quad>): void => {
        scene.createInstance({ mesh: scene.createMesh({ ...mesh, label }), materials: [material], transform: IDENTITY, label });
    };
    add('floor', quad([[-20, 0, 20], [20, 0, 20], [20, 0, -20], [-20, 0, -20]], [0, 1, 0]));
    add('wall', quad([[-20, 0, -6], [20, 0, -6], [20, 6, -6], [-20, 6, -6]], [0, 0, 1]));
    add('box top', quad([[-1, 1, 0], [1, 1, 0], [1, 1, -2], [-1, 1, -2]], [0, 1, 0]));
    add('box front', quad([[-1, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 1, 0]], [0, 0, 1]));
    scene.setEnvironment({ type: 'color', color: [SKY, SKY, SKY], intensity: 1 });
    scene.createLight({ type: 'directional', direction: [-0.4, -1, -0.3], color: [1, 0.95, 0.9], intensity: 3 });
}

async function runConfiguration(device: GPUDevice, configuration: Configuration): Promise<Record<string, unknown>> {
    const platform = new GIPlatformImpl(device, { label: `radiance-cache-mac-limits ${configuration.name}` });
    const warnings: string[] = [];
    const warn = console.warn;
    console.warn = (...args: unknown[]) => {
        warnings.push(args.map(String).join(' '));
    };
    try {
        const technique = platform.addTechnique(new RadianceCacheTechnique(configuration.settings));
        platform.setOutput(technique);
        if (configuration.ddgiBelow) {
            const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: [0, 2, -1], size: [12, 4, 10], spacing: 1 } }));
            platform.setFallback(technique, ddgi);
        }
        const limitNeeds = platform.limitNeeds;
        await platform.prepare();
        describeScene(platform);
        const screen = await PageScreen.create(platform, device, WIDTH, HEIGHT);
        try {
            const input = (): FrameInput => ({ deltaTime: 1 / 60, budget: { units: 256 }, camera: frameCamera(CAMERA, WIDTH, HEIGHT) });
            const frame = (): void => {
                const encoder = device.createCommandEncoder({ label: `radiance-cache-mac-limits ${configuration.name}` });
                platform.encode(encoder, input());
                screen.encode(encoder, CAMERA);
                platform.encodeScreen(encoder, screen.screen);
                device.queue.submit([encoder.finish()]);
            };
            frame();
            await platform.rays.whenBuilt();
            for (let index = 0; index < FRAMES; index++) frame();
            await device.queue.onSubmittedWorkDone();
            const probes = technique.probes;
            if (!probes) throw new Error(`${configuration.name}: no probes after encodeScreen`);
            const output = probes.denoisedIrradiance ?? probes.irradiance;
            const values = halves(await readTexture(device, output, 8, WIDTH, HEIGHT));
            let sum = 0;
            let bad = 0;
            for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
                const rgb = [values[pixel * 4], values[pixel * 4 + 1], values[pixel * 4 + 2]];
                if (rgb.some((value) => !Number.isFinite(value) || value < 0)) bad++;
                else sum += (rgb[0] + rgb[1] + rgb[2]) / 3;
            }
            return {
                limitNeeds,
                meanIrradiance: sum / (WIDTH * HEIGHT - bad),
                badPixels: bad,
                worldCache: technique.worldCacheBackend?.id ?? null,
                neuralTail: configuration.settings.neuralCache === 'tail' ? !warnings.some((message) => message.includes("neuralCache 'tail' is unavailable")) : undefined,
                warnings: warnings.length > 0 ? warnings : undefined,
            };
        } finally {
            screen.dispose();
        }
    } finally {
        console.warn = warn;
        platform.dispose();
    }
}

const startTime = performance.now();
let deviceErrors: string[] = [];
try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU is not available, or requestAdapter() returned null');
    const limits = requiredLimits(adapter);
    const cap = Math.min(adapter.limits.maxStorageBuffersPerShaderStage, APPLE_STORAGE_BUFFERS_PER_STAGE);
    const device = await adapter.requestDevice({ requiredLimits: { ...limits, maxStorageBuffersPerShaderStage: cap }, requiredFeatures: optionalFeatures(adapter) });
    deviceErrors = watchDevice(device);
    const failures: string[] = [];
    const data: Record<string, unknown> = { adapterStorageBuffers: adapter.limits.maxStorageBuffersPerShaderStage, deviceStorageBuffers: device.limits.maxStorageBuffersPerShaderStage };

    // 1. The cap holds and is enforced.
    if (device.limits.maxStorageBuffersPerShaderStage !== APPLE_STORAGE_BUFFERS_PER_STAGE) {
        failures.push(`the device has ${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage, not ${APPLE_STORAGE_BUFFERS_PER_STAGE} (adapter ${adapter.limits.maxStorageBuffersPerShaderStage})`);
    }
    device.pushErrorScope('validation');
    const eleven = Array.from({ length: APPLE_STORAGE_BUFFERS_PER_STAGE + 1 }, (_, binding): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }));
    device.createBindGroupLayout({ label: 'eleven storage buffers', entries: eleven });
    const refusal = await device.popErrorScope();
    data.elevenRefused = refusal !== null;
    if (!refusal) failures.push(`a layout with ${APPLE_STORAGE_BUFFERS_PER_STAGE + 1} storage buffers was accepted: the cap isn't enforced`);

    // 2 and 3. Every configuration prepares, runs and lights the scene.
    if (RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE > APPLE_STORAGE_BUFFERS_PER_STAGE) failures.push(`the cache declares ${RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE}`);
    const runs: Record<string, unknown> = {};
    for (const configuration of CONFIGURATIONS) {
        try {
            const run = await runConfiguration(device, configuration);
            runs[configuration.name] = run;
            const needs = run.limitNeeds as { maxStorageBuffersPerShaderStage?: number };
            if (needs.maxStorageBuffersPerShaderStage !== RADIANCE_CACHE_STORAGE_BUFFERS_PER_STAGE) failures.push(`${configuration.name}: limitNeeds ${JSON.stringify(needs)}`);
            if ((run.badPixels as number) > 0) failures.push(`${configuration.name}: ${run.badPixels} pixels not finite or negative`);
            if (!((run.meanIrradiance as number) > 0.05)) failures.push(`${configuration.name}: mean irradiance ${run.meanIrradiance}`);
        } catch (error) {
            failures.push(`${configuration.name}: ${error instanceof Error ? error.message : String(error)}`);
        }
        await device.queue.onSubmittedWorkDone();
    }
    data.configurations = runs;

    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data,
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        deviceErrors,
        startTime,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
}
