/**
 * T1.1: the shared sky file reads the same way in every GPU consumer.
 *
 * Looks up known world directions (towards the sun's glow, away from it, to
 * either side, along +X and -Z, high up) in `scenes/sky/open-ground.exr` and
 * checks each against a CPU bilinear lookup of the decoded file under the
 * layout of `scenes/src/sky/equirect.ts`:
 *
 *  1. the platform path: the engine-free loader (`loadSkyFile` +
 *     `createSkyTexture`) to an `rgba16float` GPUTexture in the platform's
 *     equirect texture order (zenith in the last row, since T1.4), sampled by
 *     a compute shader through `SKY_EQUIRECT_WGSL`;
 *  2. the viewer path: three.js's `EXRLoader` and the viewer's sky dome,
 *     rendered by a 1-degree camera looking along each direction.
 *
 * A turned or mirrored sky misses the glow by a factor of 2 or more; the
 * tolerance (1%) only allows for filtering precision. Cycles is checked the
 * same way by `node tools/reference/cycles/run.ts --calibrate`.
 */

import { FloatType, PerspectiveCamera, RenderTarget, Scene, WebGPURenderer } from 'three/webgpu';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { createSkyDome, loadSkyTexture } from '../../../../apps/viewer/src/scene/sky.ts';
import { sampleEquirectBilinear, type Direction } from '../../../../scenes/src/sky/equirect.ts';
import { halfImageToFloats } from '../../../../scenes/src/sky/exr-half.ts';
import { loadSkyFile } from '../../../../scenes/src/sky/sky-file.ts';
import { createSkyTexture, SKY_EQUIRECT_WGSL } from '../../../../scenes/src/sky/sky-texture.ts';

const SCENE_ID = 'open-ground';
const TOLERANCE = 0.01;
const skyUrl = Object.values(import.meta.glob('../../../../scenes/sky/open-ground.exr', { query: '?url', import: 'default', eager: true }))[0] as string;
const manifestUrl = Object.values(import.meta.glob('../../../../scenes/manifests/open-ground.json', { query: '?url', import: 'default', eager: true }))[0] as string;

interface Probe {
    name: string;
    direction: Direction;
}

function probesFor(sunDirection: Direction): Probe[] {
    const sunAzimuth = Math.atan2(sunDirection[2], sunDirection[0]);
    const at = (azimuth: number, elevationDeg: number): Direction => {
        const elevation = (elevationDeg * Math.PI) / 180;
        return [Math.cos(elevation) * Math.cos(azimuth), Math.sin(elevation), Math.cos(elevation) * Math.sin(azimuth)];
    };
    return [
        { name: 'towards the sun glow, 15 up', direction: at(sunAzimuth, 15) },
        { name: 'away from the sun, 15 up', direction: at(sunAzimuth + Math.PI, 15) },
        { name: 'sun azimuth + 90, 15 up', direction: at(sunAzimuth + Math.PI / 2, 15) },
        { name: 'sun azimuth - 90, 15 up', direction: at(sunAzimuth - Math.PI / 2, 15) },
        { name: '+X, 30 up', direction: at(0, 30) },
        { name: '-Z, 30 up', direction: at(-Math.PI / 2, 30) },
        { name: 'towards the sun, 60 up', direction: at(sunAzimuth, 60) },
        { name: 'away from the sun, 80 up', direction: at(sunAzimuth + Math.PI, 80) },
    ];
}

function luminance(rgb: ArrayLike<number>): number {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

/** Platform path: sample the loader's GPUTexture in a compute shader at each direction. */
async function samplePlatformTexture(device: GPUDevice, texture: GPUTexture, probes: Probe[]): Promise<Float32Array> {
    const directions = new Float32Array(probes.length * 4);
    probes.forEach((probe, index) => directions.set(probe.direction, index * 4));
    const directionBuffer = device.createBuffer({ size: directions.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(directionBuffer, 0, directions);
    const resultBuffer = device.createBuffer({ size: directions.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuffer = device.createBuffer({ size: directions.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const module = device.createShaderModule({
        label: 'sky lookup',
        code: /* wgsl */ `
${SKY_EQUIRECT_WGSL}
@group(0) @binding(0) var skyTexture: texture_2d<f32>;
@group(0) @binding(1) var skySampler: sampler;
@group(0) @binding(2) var<storage, read> directions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&directions)) { return; }
    results[id.x] = textureSampleLevel(skyTexture, skySampler, skyEquirectUv(directions[id.x].xyz), 0.0);
}`,
    });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'clamp-to-edge' });
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: texture.createView() },
            { binding: 1, resource: sampler },
            { binding: 2, resource: { buffer: directionBuffer } },
            { binding: 3, resource: { buffer: resultBuffer } },
        ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(probes.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(resultBuffer, 0, readBuffer, 0, directions.byteLength);
    device.queue.submit([encoder.finish()]);
    await readBuffer.mapAsync(GPUMapMode.READ);
    const results = new Float32Array(readBuffer.getMappedRange().slice(0));
    readBuffer.unmap();
    for (const buffer of [directionBuffer, resultBuffer, readBuffer]) buffer.destroy();
    return results;
}

interface WebGpuBackendInternals {
    device: GPUDevice;
    get(texture: unknown): { texture: GPUTexture };
}

/** Viewer path: the viewer's sky dome, rendered with a 1-degree camera along each direction; the centre 2x2 pixels. */
async function renderViewerDome(renderer: WebGPURenderer, probes: Probe[]): Promise<number[][]> {
    const backend = renderer.backend as unknown as WebGpuBackendInternals;
    const device = backend.device;
    const sky = await loadSkyTexture(SCENE_ID);
    const scene = new Scene();
    scene.add(createSkyDome(sky));
    const size = 8;
    const target = new RenderTarget(size, size, { type: FloatType });
    const camera = new PerspectiveCamera(1, 1, 0.1, 1_000_000);
    const results: number[][] = [];
    for (const probe of probes) {
        camera.position.set(0, 0, 0);
        camera.lookAt(...probe.direction);
        camera.updateMatrixWorld(true);
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);

        const bytesPerRow = 256;
        const buffer = device.createBuffer({ size: bytesPerRow * size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer({ texture: backend.get(target.texture).texture }, { buffer, bytesPerRow }, { width: size, height: size });
        device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(buffer.getMappedRange().slice(0));
        buffer.unmap();
        buffer.destroy();
        const centre = [0, 0, 0];
        for (const y of [size / 2 - 1, size / 2]) {
            for (const x of [size / 2 - 1, size / 2]) {
                for (let channel = 0; channel < 3; channel++) centre[channel] += values[(y * bytesPerRow) / 4 + x * 4 + channel] / 4;
            }
        }
        results.push(centre);
    }
    target.dispose();
    sky.dispose();
    return results;
}

const startTime = performance.now();
const renderer = new WebGPURenderer({ antialias: false });
await renderer.init();
const device = (renderer.backend as unknown as WebGpuBackendInternals).device;
const deviceErrors = watchDevice(device);
try {
    const manifest = (await (await fetch(manifestUrl)).json()) as { environment: { sunDirection: Direction } };
    const probes = probesFor(manifest.environment.sunDirection);
    const image = await loadSkyFile(skyUrl);
    const floats = halfImageToFloats(image);
    const expected = probes.map((probe) => sampleEquirectBilinear(floats, image.width, image.height, probe.direction));

    const texture = createSkyTexture(device, image, 'shared sky (platform loader)');
    const platform = await samplePlatformTexture(device, texture, probes);
    texture.destroy();
    const viewer = await renderViewerDome(renderer, probes);

    const rows: Record<string, number>[] = [];
    const failures: string[] = [];
    probes.forEach((probe, index) => {
        const wanted = luminance(expected[index]);
        const fromPlatform = luminance(platform.subarray(index * 4, index * 4 + 3));
        const fromViewer = luminance(viewer[index]);
        const platformError = fromPlatform / wanted - 1;
        const viewerError = fromViewer / wanted - 1;
        rows.push({ expected: wanted, platform: fromPlatform, viewer: fromViewer, platformError, viewerError });
        if (Math.abs(platformError) > TOLERANCE) failures.push(`platform loader, ${probe.name}: ${fromPlatform.toFixed(5)} vs file ${wanted.toFixed(5)}`);
        if (Math.abs(viewerError) > TOLERANCE) failures.push(`viewer sky dome, ${probe.name}: ${fromViewer.toFixed(5)} vs file ${wanted.toFixed(5)}`);
    });
    // The glow must really be there, or the check proves nothing.
    const glowRatio = luminance(expected[0]) / luminance(expected[1]);
    if (glowRatio < 2) failures.push(`the file's sun glow is only ${glowRatio.toFixed(2)}x the opposite sky; the probes can't tell orientation apart`);

    await drainDevice(device);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join('; '),
        data: {
            probes: probes.map((probe) => probe.name),
            worstPlatformError: Math.max(...rows.map((row) => Math.abs(row.platformError))),
            worstViewerError: Math.max(...rows.map((row) => Math.abs(row.viewerError))),
            glowRatio,
            rows,
        },
        deviceErrors,
        startTime,
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), deviceErrors, startTime });
}
