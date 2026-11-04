/**
 * Measures three's PMREM environment diffuse; this is NOT the viewer's sky check.
 * The viewer never lights surfaces with `scene.environment` (three r187 would
 * add it on top of probe GI); it only shows the sky file on a dome, which the
 * `sky-orientation` GPU page checks.
 *
 * What this renders: a manifest scene built by the viewer's `loadScene` (the
 * same geometry, materials and sky file), lit by the sky only through three's
 * environment lighting (`applySkyEnvironmentLighting`, PMREM), into a linear
 * float image. Driven by `tools/sky/pmrem-env-render.ts`; see `tools/sky/README.md`.
 *
 * `material`:
 *  - `diffuse`: every material made pure Lambert (MeshPhysicalMaterial,
 *    roughness 1, metalness 0, specularIntensity 0), as the references are;
 *  - `standard`: the viewer's own materials (MeshStandardMaterial, roughness
 *    1, metalness 0, F0 0.04), to measure what the specular lobe adds.
 *
 * No tone mapping and no exposure: the float target holds scene radiance.
 */

import { FloatType, PerspectiveCamera, RenderTarget, WebGPURenderer } from 'three/webgpu';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../gpu/client.ts';
import { makeDiffuseOnly } from '../../reference/pathtracer/page/diffuse-material.ts';
import type { SceneManifest } from '../../../scenes/src/manifest.ts';
import { loadScene } from '../../../apps/viewer/src/scene/load-scene.ts';
import { applySkyEnvironmentLighting } from '../../../apps/viewer/src/scene/sky.ts';

interface Job {
    scene: string;
    camera: string;
    width: number;
    height: number;
    material: 'diffuse' | 'standard';
}

/** Renders before reading back: PMREM and background pipeline compiles settle within the first few frames. */
const WARM_UP_FRAMES = 8;

interface WebGpuBackendInternals {
    device: GPUDevice;
    get(texture: unknown): { texture: GPUTexture };
}

function pickCamera(manifest: SceneManifest, wanted: string) {
    const camera = /^\d+$/.test(wanted) ? manifest.cameras[Number(wanted) - 1] : manifest.cameras.find((candidate) => candidate.name === wanted);
    if (!camera) throw new Error(`unknown camera ${wanted}`);
    return camera;
}

/** Copies an rgba32float texture to the CPU as interleaved RGB, row 0 at the top (as OpenEXR stores it). */
async function readTextureRgb(device: GPUDevice, texture: GPUTexture, width: number, height: number): Promise<Float32Array> {
    const bytesPerPixel = 16;
    // Texture-to-buffer copies need a row pitch that is a multiple of 256 bytes.
    const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
    const buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    buffer.destroy();
    const rowValues = bytesPerRow / 4;
    const rgb = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            for (let channel = 0; channel < 3; channel++) rgb[(y * width + x) * 3 + channel] = values[y * rowValues + x * 4 + channel];
        }
    }
    return rgb;
}

function toBase64(values: Float32Array): string {
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    let text = '';
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) text += String.fromCharCode(...bytes.subarray(i, i + chunk));
    return btoa(text);
}

async function main(): Promise<void> {
    const startTime = performance.now();
    const job = (await (await fetch('/job.json')).json()) as Job;
    const manifest = (await (await fetch(`/manifest/${job.scene}.json`)).json()) as SceneManifest;
    const bookmark = pickCamera(manifest, job.camera);

    const renderer = new WebGPURenderer({ antialias: false });
    renderer.setSize(job.width, job.height);
    await renderer.init();
    const backend = renderer.backend as unknown as WebGpuBackendInternals;
    const device = backend.device;
    const deviceErrors = watchDevice(device);

    try {
        const loaded = await loadScene(manifest, () => {});
        if (!loaded.sky) throw new Error(`scene ${manifest.id} has no sky file (environment "${manifest.environment.type}")`);
        // three PMREM env diffuse, not used by the viewer: measured here on its own.
        applySkyEnvironmentLighting(loaded.scene, loaded.sky);
        if (job.material === 'diffuse') makeDiffuseOnly(loaded.content);

        const camera = new PerspectiveCamera(bookmark.fovYDeg, job.width / job.height, 0.05, 1000);
        camera.position.set(...bookmark.position);
        camera.lookAt(...bookmark.target);
        camera.updateMatrixWorld(true);

        const target = new RenderTarget(job.width, job.height, { type: FloatType, depthBuffer: true });
        await renderer.compileAsync(loaded.scene, camera);
        renderer.setRenderTarget(target);
        for (let frame = 0; frame < WARM_UP_FRAMES; frame++) {
            renderer.render(loaded.scene, camera);
            await device.queue.onSubmittedWorkDone();
        }
        renderer.setRenderTarget(null);

        const rgb = await readTextureRgb(device, backend.get(target.texture).texture, job.width, job.height);
        let sum = 0;
        for (let i = 0; i < rgb.length; i += 3) sum += 0.2126 * rgb[i] + 0.7152 * rgb[i + 1] + 0.0722 * rgb[i + 2];
        await drainDevice(device);
        reportGpuTest({
            status: 'pass',
            data: { meanLuminance: sum / (job.width * job.height), rgbBase64: toBase64(rgb) },
            deviceErrors,
            startTime,
            adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        });
    } catch (error) {
        reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), deviceErrors, startTime });
    }
}

void main();
