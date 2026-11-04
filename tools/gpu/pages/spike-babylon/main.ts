/**
 * T0.9 spike: can Babylon's WebGPU engine and a material plugin read a GI
 * module's textures, using the platform's `ShadingModule` contract?
 *
 * Setup: a white PBR plane, no lights, no environment, in front of a camera
 * that renders into a 64x64 float render target (so the result is linear and
 * not clamped or gamma-encoded). A stand-in technique (standInGi.ts) holds
 * constant textures, so the irradiance E it returns is known. A material plugin
 * (shadingModulePlugin.ts) adds E/pi * albedo to the PBR diffuse.
 *
 * Babylon's conventions, which fix the expected value:
 *   - Babylon's environment diffuse is stored as E/pi ("L = (1/pi) * E * rho",
 *     sphericalPolynomial.pure.js:187-199) and multiplied by albedo in the shader.
 *     So a surface lit only by irradiance E with white albedo reads E/pi.
 *   - No lights, no ambient colour, no emissive, so nothing else adds to the pixel.
 *   - The render target is a float texture and image processing is applied by
 *     post-process (which does not exist here), so the shader writes linear values.
 *   - With metallicF0Factor = 0 the material has no specular energy, so Babylon's
 *     "diffuse gets what specular leaves" factor is exactly 1.
 *
 * Tolerance: 0.002 absolute plus 1% of the expected value. The textures are
 * rgba16float (about 0.05% error) and the arithmetic is float32.
 *
 * It runs twice:
 *   - "host-owned": the device is created here, first, and Babylon is made to use
 *     it by wrapping requestAdapter/requestDevice. This is NOT a supported
 *     Babylon path; it shows nothing in the engine rejects a foreign device.
 *     This device has no float32-filterable, so the rgba32float table is bound
 *     as unfilterable.
 *   - "babylon-owned": Babylon creates the device from a deviceDescriptor (the
 *     supported path; the core would take `engine._device`). This device has
 *     float32-filterable.
 */

import { Constants } from '@babylonjs/core/Engines/constants.js';
import { WebGPUEngine } from '@babylonjs/core/Engines/webgpuEngine.js';
import { FreeCamera } from '@babylonjs/core/Cameras/freeCamera.js';
import { Color3, Color4 } from '@babylonjs/core/Maths/math.color.js';
import { Vector3 } from '@babylonjs/core/Maths/math.vector.js';
import { PBRMaterial } from '@babylonjs/core/Materials/PBR/pbrMaterial.js';
import '@babylonjs/core/Materials/Textures/baseTexture.polynomial.js'; // adds BaseTexture.sphericalPolynomial
import { BaseTexture } from '@babylonjs/core/Materials/Textures/baseTexture.js';
import { RawCubeTexture } from '@babylonjs/core/Materials/Textures/rawCubeTexture.js';
import { RenderTargetTexture } from '@babylonjs/core/Materials/Textures/renderTargetTexture.js';
import { CubeMapToSphericalPolynomialTools } from '@babylonjs/core/Misc/HighDynamicRange/cubemapToSphericalPolynomial.js';
import { CreatePlane } from '@babylonjs/core/Meshes/Builders/planeBuilder.js';
import { Scene } from '@babylonjs/core/scene.js';
import { drainDevice, reportGpuTest, summarizeAdapter, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import type { AdapterSummary } from '../../contract.ts';
import { createStandInGi, type Rgb, type TableValues } from './standInGi.ts';
import { ShadingModulePlugin } from './shadingModulePlugin.ts';

const TARGET_SIZE = 64;
/** Pixel columns well inside the left and right halves of the screen, on the middle row. */
const LEFT_PIXEL = { x: 16, y: 32 };
const RIGHT_PIXEL = { x: 48, y: 32 };
const ABSOLUTE_TOLERANCE = 0.002;
const RELATIVE_TOLERANCE = 0.01;

/** Limits the host asks for, to see them arrive on the device Babylon uses. */
const WANTED_LIMITS = { maxSampledTexturesPerShaderStage: 20, maxStorageBuffersPerShaderStage: 10 } as const;

/** A uniform white furnace: every direction has this radiance. */
const ENVIRONMENT_RADIANCE: Rgb = [0.6, 0.3, 0.9];

interface Check {
    name: string;
    ok: boolean;
    /** Numbers or words that explain the result, shown in the run output. */
    detail: string;
}

const checks: Check[] = [];
const notes: string[] = [];
const deviceErrors: string[] = [];

function expect(name: string, ok: boolean, detail = ''): void {
    checks.push({ name, ok, detail });
}

function isClose(got: Rgb, expected: Rgb): boolean {
    return [0, 1, 2].every((channel) => {
        const wanted = expected[channel]!;
        return Math.abs(got[channel]! - wanted) <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * Math.abs(wanted);
    });
}

function expectClose(name: string, got: Rgb, expected: Rgb): void {
    expect(name, isClose(got, expected), `got ${fmt(got)}, expected ${fmt(expected)}`);
}

const startTime = performance.now();
let adapterSummary: AdapterSummary | null = null;
let failure: string | undefined;
const data: Record<string, unknown> = {};

try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('requestAdapter() returned null.');
    adapterSummary = summarizeAdapter(adapter);

    const limits = pickSupportedLimits(adapter);
    data['requestedLimits'] = limits;

    // Scenario A: the host created the device first and Babylon adopts it.
    const hostFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
    const hostDevice = await adapter.requestDevice({ requiredLimits: limits, requiredFeatures: hostFeatures });
    deviceErrors.push(...watchDevice(hostDevice));
    data['hostOwnedDevice'] = await runScenario('host-owned', {
        adapter,
        hostDevice,
        limits,
        features: hostFeatures,
        showPreview: false,
    });

    // Scenario B: Babylon creates the device (our limits and features go in its deviceDescriptor).
    const babylonFeatures: GPUFeatureName[] = (['float32-filterable', 'timestamp-query'] as const).filter((feature) =>
        adapter.features.has(feature),
    );
    data['babylonOwnedDevice'] = await runScenario('babylon-owned', {
        adapter,
        limits,
        features: babylonFeatures,
        showPreview: true,
    });

    const failed = checks.filter((check) => !check.ok);
    if (failed.length > 0) {
        failure = failed.map((check) => `${check.name}: ${check.detail}`).join(' | ');
    }
} catch (error) {
    failure = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
}

data['checks'] = checks;
data['notes'] = notes;

reportGpuTest({
    status: failure ? 'fail' : 'pass',
    message: failure,
    data,
    adapter: adapterSummary,
    deviceErrors,
    startTime,
    logLines: [
        ...checks.map((check) => `${check.ok ? 'ok  ' : 'FAIL'} ${check.name}${check.detail ? ': ' + check.detail : ''}`),
        ...notes,
    ],
});

// ---------------------------------------------------------------------------

interface ScenarioOptions {
    adapter: GPUAdapter;
    /** Present when the host made the device itself; absent when Babylon should make it. */
    hostDevice?: GPUDevice;
    limits: Record<string, number>;
    features: GPUFeatureName[];
    showPreview: boolean;
}

async function runScenario(label: string, options: ScenarioOptions): Promise<Record<string, unknown>> {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    canvas.style.cssText = 'display:block; width:128px; height:128px;';
    document.body.append(canvas);

    // ----- The engine, and whose device it runs on -----
    let engine: WebGPUEngine;
    if (options.hostDevice) {
        engine = await createEngineOnHostDevice(canvas, options.adapter, options.hostDevice);
    } else {
        engine = new WebGPUEngine(canvas, {
            antialias: false,
            deviceDescriptor: { requiredLimits: options.limits, requiredFeatures: options.features },
        });
        await engine.initAsync();
    }

    // The device the core would be given: the host's own, or the one Babylon made.
    const device: GPUDevice = engine._device;
    if (options.hostDevice) {
        expect(`${label}: engine runs on the host's GPUDevice`, device === options.hostDevice);
    } else {
        deviceErrors.push(...watchDevice(device));
    }
    const deviceLimits = device.limits as unknown as Record<string, number>;
    for (const [name, wanted] of Object.entries(options.limits)) {
        expect(`${label}: device limit ${name} is ${wanted}`, deviceLimits[name] === wanted, `got ${deviceLimits[name]}`);
    }
    for (const feature of options.features) {
        expect(`${label}: device has feature ${feature}`, device.features.has(feature));
    }
    const hasFloat32Filterable = device.features.has('float32-filterable');

    // ----- Scene: one white PBR plane facing the camera, no lights, no environment -----
    const scene = new Scene(engine);
    scene.clearColor = new Color4(0, 0, 0, 1);
    scene.imageProcessingConfiguration.applyByPostProcess = true; // linear output, see the header
    const camera = new FreeCamera('camera', new Vector3(0, 0, -3), scene);
    camera.setTarget(Vector3.Zero());
    const plane = CreatePlane('plane', { size: 20 }, scene);
    const material = new PBRMaterial('white', scene);
    material.albedoColor = Color3.White();
    material.metallic = 0;
    material.roughness = 1;
    material.metallicF0Factor = 0; // no specular energy, so the diffuse share is exactly 1
    material.backFaceCulling = false;
    plane.material = material;

    const target = new RenderTargetTexture('target', TARGET_SIZE, scene, {
        type: Constants.TEXTURETYPE_FLOAT,
        generateMipMaps: false,
        generateDepthBuffer: true,
    });
    camera.outputRenderTarget = target;

    const gi = createStandInGi(device, 8, [0.2, 0.1, 0.05]);
    const plugin = new ShadingModulePlugin(material, gi.module, engine, scene);

    const render = async (): Promise<Float32Array> => {
        await waitUntilReady(scene);
        engine.beginFrame();
        scene.render();
        engine.endFrame();
        const pixels = await target.readPixels();
        if (!pixels) throw new Error('readPixels returned nothing.');
        return pixels as Float32Array;
    };
    const pixelAt = (pixels: Float32Array, point: { x: number; y: number }): Rgb => {
        const index = (point.y * TARGET_SIZE + point.x) * 4;
        return [pixels[index]!, pixels[index + 1]!, pixels[index + 2]!];
    };
    const outgoing = (irradiance: Rgb): Rgb => [irradiance[0] / Math.PI, irradiance[1] / Math.PI, irradiance[2] / Math.PI];

    // 1. Control: plugin off. Nothing but the plugin adds light, so the plane is black.
    plugin.isEnabled = false;
    let pixels = await render();
    expectClose(`${label}: plugin off reads black`, pixelAt(pixels, LEFT_PIXEL), [0, 0, 0]);

    // 2. Plugin on: the added irradiance, on both halves of the screen (the halves read different table texels).
    plugin.isEnabled = true;
    pixels = await render();
    expectClose(`${label}: plugin on, left half = E/pi`, pixelAt(pixels, LEFT_PIXEL), outgoing(gi.expectedIrradiance('left')));
    expectClose(`${label}: plugin on, right half = E/pi`, pixelAt(pixels, RIGHT_PIXEL), outgoing(gi.expectedIrradiance('right')));
    const wrapsAfterFirstFrame = plugin.wrapCount;
    expect(`${label}: the 3 textures were wrapped once each`, plugin.wrapCount === 3, `wrapCount=${plugin.wrapCount}`);

    // 3. Content changes in place: no re-wrapping needed.
    const changedTable: TableValues = [
        [0.0, 0.0, 0.5, 1],
        [1.0, 0.5, 0.25, 1],
    ];
    gi.writeTable(changedTable);
    pixels = await render();
    expectClose(`${label}: table rewritten in place, left`, pixelAt(pixels, LEFT_PIXEL), outgoing(gi.expectedIrradiance('left')));
    expectClose(`${label}: table rewritten in place, right`, pixelAt(pixels, RIGHT_PIXEL), outgoing(gi.expectedIrradiance('right')));
    expect(`${label}: an in-place write needed no new wrapper`, plugin.wrapCount === wrapsAfterFirstFrame, `wrapCount=${plugin.wrapCount}`);

    // 4. The technique reallocates its atlas at a different size: the plugin sees a new GPUTexture and re-wraps.
    const oldAtlas = gi.reallocateAtlas(16, [0.4, 0.05, 0.3]);
    pixels = await render();
    expectClose(`${label}: atlas reallocated 8 to 16, left`, pixelAt(pixels, LEFT_PIXEL), outgoing(gi.expectedIrradiance('left')));
    expect(
        `${label}: reallocation re-wrapped exactly one texture`,
        plugin.wrapCount === wrapsAfterFirstFrame + 1,
        `wrapCount=${plugin.wrapCount}`,
    );
    // Dropping Babylon's wrapper must not have destroyed our old GPUTexture: writing to a destroyed texture is a validation error.
    device.queue.writeTexture({ texture: oldAtlas }, new Uint16Array(4), { bytesPerRow: 8 }, [1, 1, 1]);
    oldAtlas.destroy();

    // 5. Babylon's default dielectric: how much Babylon's own energy conservation takes away from added diffuse.
    material.metallicF0Factor = 1;
    pixels = await render();
    const withSpecular = pixelAt(pixels, LEFT_PIXEL);
    const noSpecular = outgoing(gi.expectedIrradiance('left'));
    const keptShare = withSpecular[1] / noSpecular[1];
    notes.push(`${label}: default dielectric F0 (metallicF0Factor=1, roughness 1) keeps ${(keptShare * 100).toFixed(1)}% of E/pi`);
    expect(`${label}: default F0 keeps 85 to 100% of E/pi`, keptShare > 0.85 && keptShare <= 1.0, `${(keptShare * 100).toFixed(1)}%`);
    const giOnly = withSpecular; // the GI term with the default material, no environment

    // 6. Environment: keep Babylon's environment diffuse and add ours, or replace the diffuse part and keep environment specular.
    const environment = createUniformEnvironment(scene, ENVIRONMENT_RADIANCE);
    scene.environmentTexture = environment;
    plugin.isEnabled = false;
    const environmentOnly = pixelAt(await render(), LEFT_PIXEL);
    // A uniform environment of radiance L lights a white diffuse+specular surface with about L in total.
    // Babylon's energy conservation is not exact: measured about 97.4%, so allow 95 to 100%.
    const furnaceShare = environmentOnly.map((value, c) => value / ENVIRONMENT_RADIANCE[c]!);
    notes.push(`${label}: environment alone reads ${furnaceShare.map((v) => (v * 100).toFixed(1)).join('/')}% of its radiance (white furnace)`);
    expect(
        `${label}: environment alone reads about its own radiance (white furnace)`,
        furnaceShare.every((share) => share > 0.95 && share <= 1.0),
        `share = ${furnaceShare.map((v) => v.toFixed(3)).join(', ')}`,
    );

    plugin.isEnabled = true;
    const added = pixelAt(await render(), LEFT_PIXEL);
    expectClose(
        `${label}: environment + GI, both kept = environment + GI-only`,
        added,
        add(environmentOnly, giOnly),
    );

    plugin.replaceEnvironmentDiffuse = true;
    const replaced = pixelAt(await render(), LEFT_PIXEL);
    // What is left after removing environment diffuse is environment specular + GI.
    const specularLeft = subtract(replaced, giOnly);
    const specularShare = specularLeft.map((value, c) => value / ENVIRONMENT_RADIANCE[c]!);
    notes.push(
        `${label}: environment specular alone is ${specularShare.map((v) => (v * 100).toFixed(1)).join('/')}% of L; diffuse took the rest`,
    );
    expect(
        `${label}: replacing environment diffuse keeps a small environment specular and drops the diffuse`,
        specularShare.every((share) => share > 0.005 && share < 0.1),
        `specular share of L = ${specularShare.map((v) => v.toFixed(3)).join(', ')}`,
    );
    plugin.replaceEnvironmentDiffuse = false;
    scene.environmentTexture = null;
    environment.dispose();
    material.metallicF0Factor = 0;

    // 7. A wrapped texture that is disposed the plain Babylon way takes the GPUTexture with it.
    await probeDisposeDestroysWrappedTexture(label, engine, scene, device);

    if (options.showPreview) {
        // For the screenshot: draw to the canvas with Babylon's normal gamma output.
        camera.outputRenderTarget = null;
        scene.imageProcessingConfiguration.applyByPostProcess = false;
        await waitUntilReady(scene);
        engine.beginFrame();
        scene.render();
        engine.endFrame();
        await drainDevice(device);
    }

    const summary: Record<string, unknown> = {
        adapter: summarizeAdapterInfo(device.adapterInfo, device.features),
        hasFloat32Filterable,
        wrapCount: plugin.wrapCount,
        keptShareOfIrradianceWithDefaultF0: keptShare,
    };

    if (options.hostDevice) {
        // Babylon's dispose destroys the device it was given (webgpuEngine.pure.js:3176).
        // Count the calls, and keep the host's device alive for the next scenario.
        const hostDevice = options.hostDevice;
        const realDestroy = hostDevice.destroy.bind(hostDevice);
        let destroyCalls = 0;
        hostDevice.destroy = () => {
            destroyCalls++;
        };
        plugin.dispose();
        gi.destroy();
        engine.dispose();
        expect(`${label}: engine.dispose() calls device.destroy() (which would kill a host's device)`, destroyCalls === 1, `calls=${destroyCalls}`);
        hostDevice.destroy = realDestroy;
        canvas.remove();
        await drainDevice(hostDevice);
        summary['destroyCallsFromEngineDispose'] = destroyCalls;
    }
    return summary;
}

/**
 * Makes `new WebGPUEngine().initAsync()` run on a device the host created.
 * WebGPUEngine calls navigator.gpu.requestAdapter(options) and then
 * adapter.requestDevice(descriptor) (webgpuEngine.pure.js:443,481) and has no
 * option to pass a device in, so for the duration of initAsync we hand it an
 * adapter whose requestDevice returns the host's device.
 */
async function createEngineOnHostDevice(canvas: HTMLCanvasElement, adapter: GPUAdapter, hostDevice: GPUDevice): Promise<WebGPUEngine> {
    const adapterThatReturnsHostDevice = new Proxy(adapter, {
        get(target, property) {
            if (property === 'requestDevice') return async () => hostDevice;
            const value = Reflect.get(target, property, target) as unknown;
            return typeof value === 'function' ? value.bind(target) : value;
        },
    });
    const gpu = navigator.gpu;
    const originalRequestAdapter = gpu.requestAdapter;
    gpu.requestAdapter = async () => adapterThatReturnsHostDevice;
    try {
        const engine = new WebGPUEngine(canvas, { antialias: false });
        await engine.initAsync();
        return engine;
    } finally {
        gpu.requestAdapter = originalRequestAdapter;
    }
}

/** A cube map with the same radiance in every direction, plus the spherical harmonics Babylon uses for diffuse. */
function createUniformEnvironment(scene: Scene, radiance: Rgb): RawCubeTexture {
    const size = 4;
    const face = (): Float32Array => {
        const texels = new Float32Array(size * size * 4);
        for (let texel = 0; texel < size * size; texel++) texels.set([...radiance, 1], texel * 4);
        return texels;
    };
    const faces = [face(), face(), face(), face(), face(), face()];
    const cube = new RawCubeTexture(scene, faces, size, Constants.TEXTUREFORMAT_RGBA, Constants.TEXTURETYPE_FLOAT, false, false);
    cube.sphericalPolynomial = CubeMapToSphericalPolynomialTools.ConvertCubeMapToSphericalPolynomial({
        size,
        right: faces[0]!,
        left: faces[1]!,
        up: faces[2]!,
        down: faces[3]!,
        front: faces[4]!,
        back: faces[5]!,
        format: Constants.TEXTUREFORMAT_RGBA,
        type: Constants.TEXTURETYPE_FLOAT,
        gammaSpace: false,
    });
    return cube;
}

/**
 * Wraps a throwaway GPUTexture and disposes it the plain Babylon way. Babylon
 * destroys the GPUTexture (deferred to the end of the frame), which is why the
 * plugin detaches it first. The validation error is caught here, not reported.
 */
async function probeDisposeDestroysWrappedTexture(label: string, engine: WebGPUEngine, scene: Scene, device: GPUDevice): Promise<void> {
    const throwaway = device.createTexture({ size: [4, 4], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const wrapper = new BaseTexture(scene, engine.wrapWebGPUTexture(throwaway));
    wrapper.dispose();
    engine.beginFrame();
    engine.endFrame(); // deferred releases happen here (webgpuEngine.pure.js endFrame -> destroyDeferredTextures)
    device.pushErrorScope('validation');
    device.queue.writeTexture({ texture: throwaway }, new Uint8Array(4), { bytesPerRow: 4 }, [1, 1, 1]);
    const error = await device.popErrorScope();
    expect(`${label}: disposing Babylon's wrapper destroys the wrapped GPUTexture`, error !== null, error ? error.message.slice(0, 80) : 'texture survived');
}

async function waitUntilReady(scene: Scene): Promise<void> {
    for (let attempt = 0; attempt < 500; attempt++) {
        if (scene.isReady()) return;
        await new Promise((resolve) => setTimeout(resolve, 10));
    }
    throw new Error('The Babylon scene did not become ready within 5 s (shader compile problem?).');
}

function pickSupportedLimits(adapter: GPUAdapter): Record<string, number> {
    const picked: Record<string, number> = {};
    for (const [name, wanted] of Object.entries(WANTED_LIMITS)) {
        const supported = (adapter.limits as unknown as Record<string, number>)[name] ?? 0;
        picked[name] = Math.min(wanted, supported);
    }
    return picked;
}

function add(a: Rgb, b: Rgb): Rgb {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function subtract(a: Rgb, b: Rgb): Rgb {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function fmt(rgb: Rgb): string {
    return `(${rgb.map((value) => value.toFixed(4)).join(', ')})`;
}
