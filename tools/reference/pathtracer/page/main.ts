/**
 * Path-traced reference page: renders a manifest scene with three-gpu-pathtracer
 * (WebGPU) and hands the linear float image back to the driver (../run.ts)
 * through the GPU test runner's report.
 *
 * Reads its job from /job.json (served by the driver), renders two images:
 *   full    all bounces (job.bounces)
 *   direct  emission seen by the camera plus light that reaches the visible
 *           surface without an earlier bounce (rendered as two runs, see Job)
 * The driver writes full, and full - direct as the indirect-only image.
 * Only procedural manifest scenes are supported (model parts need a glTF loader).
 */

import { DirectionalLight, PerspectiveCamera, PointLight, Scene, SpotLight, WebGPURenderer, type Light } from 'three/webgpu';
import { WebGPUPathTracer } from 'three-gpu-pathtracer/webgpu';
import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../../gpu/client.ts';
import { makeDiffuseOnly } from './diffuse-material.ts';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import type { CameraBookmark, SceneManifest } from '../../../../scenes/src/manifest.ts';
import { manifestLightToSceneLight } from '../../../../scenes/src/manifest.ts';
import { lightToThree, proceduralSceneToThree } from '../../../../scenes/src/three-scene.ts';

interface Job {
    scene: string;
    camera: string;
    samples: number;
    width: number;
    height: number;
    bounces: number;
    /**
     * `maxBounces` for the two parts of the direct image, found by measurement
     * (see "What full and indirect-only mean" in tools/reference/README.md).
     * The library counts the camera hit as bounce 1, and it finds emissive
     * surfaces only by following a scattered ray, but it samples point,
     * directional and spot lights explicitly at every hit. So:
     *   - lights only (emission switched off): 1 bounce = light reaching the
     *     visible surface directly;
     *   - emission only (lights switched off): 2 bounces = the emitter seen by
     *     the camera plus the emitter reached by the first scattered ray.
     * Light adds up, so direct = lights part + emission part.
     */
    directBouncesLights: number;
    directBouncesEmission: number;
    /** Path steps per renderSample call; keeps each GPU submission short. */
    frameBudget: number;
    /**
     * three.js fades a point or spot light to zero at its `range`; Cycles has no
     * such window. With this set the range is dropped (distance 0 = infinite), which
     * makes the path tracer the plain inverse-square answer, like the Cycles reference.
     */
    ignoreLightRange: boolean;
    /** When set, also denoise the full image with @pmndrs/denoiser, loaded from `moduleUrl`. */
    denoiser: { moduleUrl: string; weightsUrl: string } | null;
}

/** The part of the `denoiser` package this page calls. */
interface DenoiserLike {
    denoiseTextures(options: { color: GPUTexture; hdr: boolean; output: GPUTexture }): Promise<GPUTexture | undefined>;
    modelName?: string;
}

const startTime = performance.now();

function pickCamera(manifest: SceneManifest, wanted: string): CameraBookmark {
    if (/^\d+$/.test(wanted)) {
        const camera = manifest.cameras[Number(wanted) - 1];
        if (!camera) throw new Error(`camera ${wanted} is out of range (${manifest.cameras.length} bookmarks)`);
        return camera;
    }
    const camera = manifest.cameras.find((candidate) => candidate.name === wanted);
    if (!camera) throw new Error(`no camera named ${wanted}`);
    return camera;
}

function bytesToBase64(bytes: Uint8Array): string {
    let binary = '';
    const chunk = 0x8000;
    for (let start = 0; start < bytes.length; start += chunk) {
        binary += String.fromCharCode(...bytes.subarray(start, start + chunk));
    }
    return btoa(binary);
}

function halfToFloat(half: number): number {
    const sign = (half & 0x8000) !== 0 ? -1 : 1;
    const exponent = (half >> 10) & 0x1f;
    const fraction = half & 0x3ff;
    if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
    if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
    return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** Copies an rgba32float or rgba16float texture to the CPU as interleaved RGB floats. */
async function readTextureRgb(device: GPUDevice, texture: GPUTexture, width: number, height: number): Promise<Float32Array> {
    const bytesPerPixel = texture.format === 'rgba16float' ? 8 : 16;
    // Row pitch must be a multiple of 256 bytes when copying a texture to a buffer.
    const bytesPerRow = Math.ceil((width * bytesPerPixel) / 256) * 256;
    const buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const mapped = buffer.getMappedRange().slice(0);
    buffer.unmap();
    buffer.destroy();
    const channels = bytesPerPixel / (texture.format === 'rgba16float' ? 2 : 4);
    const half = texture.format === 'rgba16float' ? new Uint16Array(mapped) : null;
    const full = texture.format === 'rgba16float' ? null : new Float32Array(mapped);
    const rowValues = bytesPerRow / (bytesPerPixel / channels);
    const rgb = new Float32Array(width * height * 3);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            for (let channel = 0; channel < 3; channel++) {
                const source = y * rowValues + x * channels + channel;
                rgb[(y * width + x) * 3 + channel] = half ? halfToFloat(half[source]) : (full as Float32Array)[source];
            }
        }
    }
    return rgb;
}

/** three's WebGPU backend, which has no public typings for what we need from it. */
interface WebGpuBackendInternals {
    device: GPUDevice;
    get(texture: unknown): { texture: GPUTexture };
}

function backendOf(renderer: WebGPURenderer): WebGpuBackendInternals {
    return renderer.backend as unknown as WebGpuBackendInternals;
}

/** The path tracer's accumulation target (rgba32float, running mean of the samples). */
function accumulationTexture(renderer: WebGPURenderer, pathTracer: WebGPUPathTracer): GPUTexture {
    return backendOf(renderer).get(pathTracer.target).texture;
}

async function accumulate(pathTracer: WebGPUPathTracer, samples: number, label: string): Promise<{ seconds: number }> {
    const started = performance.now();
    pathTracer.maxSamples = samples;
    pathTracer.reset();
    let frames = 0;
    let lastProgress = performance.now();
    for (;;) {
        pathTracer.renderSample();
        frames++;
        // Yield so the page stays responsive and the runner's polling gets answered.
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (frames % 4 === 0) {
            const counts = await pathTracer.getSampleCountsAsync();
            if (counts.min >= samples) break;
            if (performance.now() - lastProgress > 10_000) {
                lastProgress = performance.now();
                console.log(`${label}: ${counts.min.toFixed(0)}/${samples} samples`);
            }
        }
    }
    return { seconds: (performance.now() - started) / 1000 };
}

async function main(): Promise<void> {
    const job = (await (await fetch('/job.json')).json()) as Job;
    const manifest = (await (await fetch(`/manifest/${job.scene}.json`)).json()) as SceneManifest;
    const camera = pickCamera(manifest, job.camera);

    const renderer = new WebGPURenderer({ antialias: false });
    renderer.setSize(job.width, job.height);
    await renderer.init();
    const device = backendOf(renderer).device;
    const deviceErrors = watchDevice(device);

    try {
        const scene = new Scene();
        const geometry = proceduralSceneToThree(mergeScenes(buildProceduralParts(manifest)), manifest.id);
        const emissiveMaterials = makeDiffuseOnly(geometry);
        scene.add(geometry);
        // Lights come from the manifest, never from the builder's own list, so
        // the Cycles and path-traced references use exactly the same ones.
        const threeLights: Light[] = [];
        for (const light of manifest.lights) {
            const threeLight = lightToThree(manifestLightToSceneLight(light), { center: manifest.volume.center, radius: 20 });
            scene.add(threeLight);
            threeLights.push(threeLight);
            if (threeLight instanceof SpotLight || threeLight instanceof DirectionalLight) scene.add(threeLight.target);
            if (threeLight instanceof PointLight || threeLight instanceof SpotLight) {
                threeLight.decay = 2;
                if (job.ignoreLightRange) threeLight.distance = 0;
            }
        }
        if (manifest.environment.type !== 'none') {
            throw new Error(`environment "${manifest.environment.type}" is not supported by the path-traced reference yet`);
        }

        // The camera basis comes straight from the manifest: position, target, vertical fov.
        const threeCamera = new PerspectiveCamera(camera.fovYDeg, job.width / job.height, 0.1, 200);
        threeCamera.position.set(...camera.position);
        threeCamera.lookAt(...camera.target);
        threeCamera.updateMatrixWorld(true);

        const pathTracer = new WebGPUPathTracer(renderer);
        pathTracer.renderToCanvas = false;
        pathTracer.synchronizeRenderSize = false;
        pathTracer.setSize(job.width, job.height);
        pathTracer.dynamicLowRes = false;
        pathTracer.renderDelay = 0;
        pathTracer.fadeDuration = 0;
        pathTracer.adaptiveFrameBudget = false;
        pathTracer.frameBudget = job.frameBudget;
        // No clamping and no glossy filtering, as in the Cycles reference: both bias the image.
        pathTracer.clampDirect = 0;
        pathTracer.clampIndirect = 0;
        pathTracer.filterGlossyFactor = 0;
        pathTracer.maxTransparentBounces = job.bounces;
        pathTracer.setScene(scene, threeCamera);

        pathTracer.maxBounces = job.bounces;
        const full = await accumulate(pathTracer, job.samples, 'full');
        await device.queue.onSubmittedWorkDone();
        const fullRgb = await readTextureRgb(device, accumulationTexture(renderer, pathTracer), job.width, job.height);

        // Optional: denoise the full image with @pmndrs/denoiser (OIDN networks in WGSL), sharing this device.
        let denoisedRgb: Float32Array | null = null;
        let denoiseSeconds = 0;
        let denoiserModel = '';
        if (job.denoiser) {
            const denoiseStarted = performance.now();
            const module = (await import(/* @vite-ignore */ job.denoiser.moduleUrl)) as {
                Denoiser: { create(options: Record<string, unknown>): Promise<DenoiserLike> };
            };
            const denoiser = await module.Denoiser.create({ device, tzaUrl: job.denoiser.weightsUrl, precision: 'fp32' });
            const output = device.createTexture({
                size: [job.width, job.height],
                format: 'rgba16float',
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING,
            });
            // No albedo or normal guides: the color-only HDR network. The accumulation texture is linear HDR, top row first.
            await denoiser.denoiseTextures({ color: accumulationTexture(renderer, pathTracer), hdr: true, output });
            await device.queue.onSubmittedWorkDone();
            denoisedRgb = await readTextureRgb(device, output, job.width, job.height);
            denoiserModel = denoiser.modelName ?? '';
            denoiseSeconds = (performance.now() - denoiseStarted) / 1000;
        }

        // The direct image, as the sum of two runs (see Job.directBouncesLights).
        const directParts: string[] = [];
        let directSeconds = 0;
        const directRgb = new Float32Array(job.width * job.height * 3);
        const addDirectRun = async (label: string, bounces: number): Promise<void> => {
            pathTracer.maxBounces = bounces;
            const run = await accumulate(pathTracer, job.samples, `direct (${label})`);
            await device.queue.onSubmittedWorkDone();
            const rgb = await readTextureRgb(device, accumulationTexture(renderer, pathTracer), job.width, job.height);
            for (let index = 0; index < directRgb.length; index++) directRgb[index] += rgb[index];
            directParts.push(`${label}: ${bounces} bounces`);
            directSeconds += run.seconds;
        };

        if (threeLights.length > 0) {
            const emissions = emissiveMaterials.map((material) => material.emissiveIntensity);
            for (const material of emissiveMaterials) material.emissiveIntensity = 0;
            pathTracer.updateMaterials();
            await addDirectRun('lights only', job.directBouncesLights);
            emissiveMaterials.forEach((material, index) => (material.emissiveIntensity = emissions[index]));
            pathTracer.updateMaterials();
        }
        if (emissiveMaterials.length > 0) {
            const intensities = threeLights.map((light) => light.intensity);
            for (const light of threeLights) light.intensity = 0;
            pathTracer.updateLights();
            await addDirectRun('emission only', job.directBouncesEmission);
            threeLights.forEach((light, index) => (light.intensity = intensities[index]));
            pathTracer.updateLights();
        }

        await drainDevice(device);
        reportGpuTest({
            status: 'pass',
            data: {
                width: job.width,
                height: job.height,
                samples: job.samples,
                maxBounces: job.bounces,
                directParts,
                fullSeconds: full.seconds,
                directSeconds,
                fullBase64: bytesToBase64(new Uint8Array(fullRgb.buffer)),
                directBase64: bytesToBase64(new Uint8Array(directRgb.buffer)),
                denoisedBase64: denoisedRgb ? bytesToBase64(new Uint8Array(denoisedRgb.buffer)) : null,
                denoiseSeconds,
                denoiserModel,
            },
            deviceErrors,
            startTime,
            adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        });
    } catch (error) {
        reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors, startTime });
    }
}

main().catch((error) => {
    reportGpuTest({ status: 'fail', message: String(error), startTime });
});
