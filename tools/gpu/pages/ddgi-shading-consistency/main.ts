/**
 * T1.6: DDGI's shading module, checked on a real GPU three ways, all on one
 * converged field (the point-light Cornell box, bookmark 1, 600 frames with
 * every probe each frame, as ddgi-test-view):
 *
 * 1. **The module's shape:** textures, samplers and `vec4f` uniforms only, at
 *    most 6 textures, each sampler named `<texture>Sampler` and belonging to a
 *    filterable texture, no `@group` in `code`, `writeUniforms` equal to the
 *    field's sampling parameters, `version` unchanged by frames.
 * 2. **The parameter path** (how an engine such as three's `wgslFn` calls
 *    it): a compute kernel declares the module's resources under its own
 *    names and calls `giIrradiance` / `giCoverage` with them in parameter
 *    order, at the engine-free test view's primary-ray hits. Its
 *    `albedo / π × giIrradiance` must equal the test view's indirect image
 *    (the same function on the same inputs). Coverage is reported.
 * 3. **The consistency check** (the globals path): the hello example's
 *    forward renderer, `globalsCode(1)` bound as globals, GI-only view
 *    (`albedo / π × giIrradianceAt(...)`, rasterized), against the test
 *    view's indirect image (primary rays) at the same camera and size. They
 *    share `ddgiSampleIrradiance`, so they differ only where rasterization
 *    and ray tracing disagree about which surface a pixel centre sees.
 *    Tolerances: `MEAN_TOLERANCE`, `PIXEL_TOLERANCE`, `MAX_DIFFERING_SHARE`.
 *
 * It also renders the hello example's full image (direct light with its
 * shadow map, plus GI) and DDGI's three screen debug views, checks they are
 * finite and not black, and draws the full image on the page for the
 * runner's screenshot.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { ShadingModule, ShadingParameter } from '../../../../packages/ddgi-core/src/platform/contract/shading-module.ts';
import { DDGI_DEBUG_VIEW } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/shading.ts';
import { ForwardRenderer, type RenderView } from '../../../../examples/webgpu/hello/src/renderer.ts';
import { imageMean, releaseDeviceOnPageHide, runFrames, TestView } from '../ddgi-shared/harness.ts';
import { createHelloScene, readRgba32FloatAsRgb } from '../ddgi-shading-shared/hello-scene.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';

const SCENE = 'cornell-box-point';
const CAMERA = 1;
const SIZE = 512;
const FRAMES = 600;
/** The contract's rule for now (docs/api/platform.md, walk-through gap 1). */
const MAX_TEXTURES = 6;

/** Raster against rays: the images' means may differ by at most this share. */
const MEAN_TOLERANCE = 0.002;
/** A pixel "differs" when it is off by more than this share of its value (plus a floor, below). */
const PIXEL_TOLERANCE = 0.01;
/** ... or by more than this share of the image mean, so near-black pixels don't count as different over rounding. */
const PIXEL_FLOOR_OF_MEAN = 0.01;
/** At most this share of pixels may differ: silhouettes, where raster and rays see different surfaces. */
const MAX_DIFFERING_SHARE = 0.01;
/** The parameter path against the test view: the same function on the same inputs. */
const PARAMETER_PATH_TOLERANCE = 1e-5;

function checkModuleShape(shading: ShadingModule): Record<string, unknown> {
    const textures = shading.parameters.filter((parameter): parameter is Extract<ShadingParameter, { kind: 'texture' }> => parameter.kind === 'texture');
    const samplers = shading.parameters.filter((parameter): parameter is Extract<ShadingParameter, { kind: 'sampler' }> => parameter.kind === 'sampler');
    const uniforms = shading.parameters.filter((parameter) => parameter.kind === 'vec4f');
    assert(textures.length <= MAX_TEXTURES, `${textures.length} textures, more than ${MAX_TEXTURES}`);
    for (const sampler of samplers) {
        assert(sampler.name === `${sampler.texture}Sampler`, `sampler ${sampler.name} should be called ${sampler.texture}Sampler`);
        const owner = textures.find((texture) => texture.name === sampler.texture);
        assert(owner && owner.sampleType === 'float', `sampler ${sampler.name} must belong to a filterable texture`);
    }
    assert(!/@group|@binding|var<storage/.test(shading.code), 'the module code declares bindings or storage');
    assert(!shading.globalsCode(1).includes('var<storage'), 'globalsCode declares a storage buffer');
    return {
        textures: textures.map((texture) => `${texture.name} (${texture.dimension}, ${texture.sampleType})`),
        samplers: samplers.map((sampler) => sampler.name),
        uniforms: uniforms.map((uniform) => uniform.name),
        providesRadiance: shading.provides.radiance,
        version: shading.version,
    };
}

/**
 * The parameter path: a kernel that declares the module's resources under
 * names of its own (as an engine would), in parameter order, and calls the
 * entry points with them at each pixel's primary-ray hit.
 */
async function parameterPathImage(device: GPUDevice, platform: GIPlatformImpl, shading: ShadingModule, camera: CameraBookmark): Promise<{ indirect: Float32Array; coverage: Float32Array }> {
    const declarations: string[] = [];
    const layoutEntries: GPUBindGroupLayoutEntry[] = [];
    const callArguments: string[] = [];
    const uniformFields: string[] = [];
    for (const parameter of shading.parameters) {
        const engineName = `engineResource${callArguments.length}`;
        if (parameter.kind === 'vec4f') {
            uniformFields.push(`${engineName}: vec4f,`);
            callArguments.push(`engineUniforms.${engineName}`);
            continue;
        }
        const binding = layoutEntries.length;
        if (parameter.kind === 'texture') {
            declarations.push(`@group(1) @binding(${binding}) var ${engineName}: texture_2d_array<f32>;`);
            layoutEntries.push({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: parameter.sampleType, viewDimension: parameter.dimension } });
        } else {
            declarations.push(`@group(1) @binding(${binding}) var ${engineName}: sampler;`);
            layoutEntries.push({ binding, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } });
        }
        callArguments.push(engineName);
    }
    const uniformBinding = layoutEntries.length;
    layoutEntries.push({ binding: uniformBinding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
    const resources = callArguments.join(', ');
    const kernel = await platform.kernels.computeKernel({
        label: 'ddgi shading parameter path',
        modules: [platform.shade.module],
        code: /* wgsl */ `
${shading.code}
${declarations.join('\n')}
struct EngineUniforms {
    ${uniformFields.join('\n    ')}
}
@group(1) @binding(${uniformBinding}) var<uniform> engineUniforms: EngineUniforms;

struct TestCamera {
    position: vec4f,
    forward: vec4f,   // xyz: unit forward; w: tan(fovY / 2)
    right: vec4f,     // xyz: unit right; w: width / height
    up: vec4f,
    size: vec4u,
}
@group(2) @binding(0) var<storage, read> testCamera: TestCamera;
@group(2) @binding(1) var<storage, read_write> testPixels: array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = testCamera.size.xy;
    if (any(id.xy >= size)) {
        return;
    }
    // The same rays as the engine-free test view (ddgi-shared/harness.ts).
    let ndc = vec2f((f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(size.y) * 2.0);
    let tanHalfFov = testCamera.forward.w;
    let direction = normalize(testCamera.forward.xyz + testCamera.right.xyz * ndc.x * tanHalfFov * testCamera.right.w + testCamera.up.xyz * ndc.y * tanHalfFov);
    let ray = GiRay(testCamera.position.xyz, 0.0, direction, 1.0e27);
    let hit = giTraceClosest(ray);
    var result = vec4f(0.0);
    if (hit.t >= 0.0) {
        let surface = giSurfaceAt(ray, hit);
        let screenUV = (vec2f(id.xy) + 0.5) / vec2f(size);
        let irradiance = giIrradiance(surface.position, surface.normal, direction, screenUV, ${resources});
        let coverage = giCoverage(surface.position, surface.normal, ${resources});
        result = vec4f(surface.albedo * 0.318309886183791 * irradiance, coverage);
    }
    testPixels[id.y * size.x + id.x] = result;
}
`,
        entryPoint: 'main',
        ownGroups: [
            { label: 'engine resources', entries: layoutEntries },
            {
                label: 'test io',
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                ],
            },
        ],
    });

    // Resources, the way an engine would wrap them.
    const uniformFloats = new Float32Array(shading.parameters.filter((parameter) => parameter.kind === 'vec4f').length * 4);
    shading.writeUniforms(uniformFloats);
    const uniformBuffer = device.createBuffer({ size: uniformFloats.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniformBuffer, 0, uniformFloats);
    const entries: GPUBindGroupEntry[] = [];
    for (const parameter of shading.parameters) {
        if (parameter.kind === 'texture') entries.push({ binding: entries.length, resource: parameter.texture().createView({ dimension: parameter.dimension }) });
        else if (parameter.kind === 'sampler') entries.push({ binding: entries.length, resource: device.createSampler({ magFilter: parameter.filtering, minFilter: parameter.filtering, addressModeU: parameter.addressMode, addressModeV: parameter.addressMode }) });
    }
    entries.push({ binding: entries.length, resource: { buffer: uniformBuffer } });

    // The camera, built exactly as TestView builds it.
    const forward = normalize3(subtract(camera.target, camera.position));
    const right = normalize3(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const cameraData = new ArrayBuffer(80);
    new Float32Array(cameraData).set([...camera.position, 0, ...forward, Math.tan((camera.fovYDeg * Math.PI) / 360), ...right, 1, ...up, 0]);
    new Uint32Array(cameraData, 64, 2).set([SIZE, SIZE]);
    const cameraBuffer = device.createBuffer({ size: cameraData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(cameraBuffer, 0, cameraData);
    const outputBytes = SIZE * SIZE * 16;
    const output = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    const encoder = device.createCommandEncoder({ label: 'ddgi shading parameter path' });
    const pass = encoder.beginComputePass();
    kernel.bind(pass);
    pass.setBindGroup(1, device.createBindGroup({ layout: kernel.ownLayouts[0], entries }));
    pass.setBindGroup(
        2,
        device.createBindGroup({
            layout: kernel.ownLayouts[1],
            entries: [
                { binding: 0, resource: { buffer: cameraBuffer } },
                { binding: 1, resource: { buffer: output } },
            ],
        }),
    );
    pass.dispatchWorkgroups(SIZE / 8, SIZE / 8);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    for (const gpuBuffer of [uniformBuffer, cameraBuffer, output, readback]) gpuBuffer.destroy();

    const indirect = new Float32Array(SIZE * SIZE * 3);
    const coverage = new Float32Array(SIZE * SIZE);
    for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
        indirect.set(pixels.subarray(pixel * 4, pixel * 4 + 3), pixel * 3);
        coverage[pixel] = pixels[pixel * 4 + 3];
    }
    return { indirect, coverage };
}

type Tuple3 = readonly [number, number, number];
function subtract(a: Tuple3, b: Tuple3): [number, number, number] {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a: Tuple3, b: Tuple3): [number, number, number] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize3(a: Tuple3): [number, number, number] {
    const length = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / length, a[1] / length, a[2] / length];
}

/** Per-pixel comparison of two rgb images (pixel luminance-free: the channel sum). */
function compareImages(candidate: Float32Array, reference: Float32Array): { meanDifference: number; differingShare: number; worstDifference: number; differingPixels: number } {
    const referenceMean = imageMean(reference);
    const candidateMean = imageMean(candidate);
    const floor = PIXEL_FLOOR_OF_MEAN * referenceMean * 3;
    let differing = 0;
    let worst = 0;
    const pixels = reference.length / 3;
    for (let pixel = 0; pixel < pixels; pixel++) {
        const a = candidate[pixel * 3] + candidate[pixel * 3 + 1] + candidate[pixel * 3 + 2];
        const b = reference[pixel * 3] + reference[pixel * 3 + 1] + reference[pixel * 3 + 2];
        const difference = Math.abs(a - b);
        worst = Math.max(worst, difference / Math.max(b, floor));
        if (difference > Math.max(PIXEL_TOLERANCE * b, floor)) differing++;
    }
    return { meanDifference: candidateMean / referenceMean - 1, differingShare: differing / pixels, worstDifference: worst, differingPixels: differing };
}

function checkFinite(label: string, image: Float32Array): void {
    let bad = 0;
    for (const value of image) if (!Number.isFinite(value)) bad++;
    assert(bad === 0, `${label}: ${bad} non-finite values`);
    assert(imageMean(image) > 0, `${label}: the image is black`);
}

async function renderView(device: GPUDevice, renderer: ForwardRenderer, camera: CameraBookmark, view: RenderView): Promise<Float32Array> {
    const encoder = device.createCommandEncoder({ label: 'hello render' });
    renderer.render(encoder, renderer.camera(camera), view);
    device.queue.submit([encoder.finish()]);
    return readRgba32FloatAsRgb(device, renderer.hdrTexture!);
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const { platform, ddgi, scene } = await createHelloScene(device, SCENE, { ignoreLightRange: true });
        try {
            const field = ddgi.field;
            assert(field, 'DDGI has no field after prepare');
            const camera = scene.manifest.cameras[CAMERA - 1];
            const shading = platform.shading;
            assert(shading === ddgi.shading, 'gi.shading is not the output technique\'s module');
            const versionBefore = shading.version;
            await runFrames(device, platform, FRAMES, field.grid.probeCount);
            assert(shading.version === versionBefore, 'version changed although nothing changed shape');

            // 1. Shape, and the uniforms against the field's own parameters.
            const shape = checkModuleShape(shading);
            const uniforms = new Float32Array(16);
            shading.writeUniforms(uniforms);
            const fieldParameters = ddgi.field!.sampleParameters;
            assert(uniforms.every((value, index) => value === fieldParameters[index]), `writeUniforms ${[...uniforms]} differs from the field's ${[...fieldParameters]}`);
            for (const parameter of shading.parameters) {
                if (parameter.kind === 'vec4f') assert(parameter.value() === parameter.value(), `${parameter.name}.value() should return the same array each time`);
            }

            // The engine-free test view: primary rays, albedo / pi x ddgiSampleIrradiance.
            const testView = await (await TestView.create(device, platform)).render(field, camera, SIZE, SIZE);

            // 2. The parameter path.
            const parameterPath = await parameterPathImage(device, platform, shading, camera);
            let parameterWorst = 0;
            for (let index = 0; index < testView.indirect.length; index++) {
                parameterWorst = Math.max(parameterWorst, Math.abs(parameterPath.indirect[index] - testView.indirect[index]));
            }
            let covered = 0;
            let fullyCovered = 0;
            let coverageSum = 0;
            for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
                // Pixels that see a surface (the background has no albedo and no coverage).
                if (testView.indirect[pixel * 3] + testView.indirect[pixel * 3 + 1] + testView.indirect[pixel * 3 + 2] <= 0) continue;
                covered++;
                coverageSum += parameterPath.coverage[pixel];
                if (parameterPath.coverage[pixel] >= 0.999) fullyCovered++;
            }
            log(`parameter path: worst difference from the test view ${parameterWorst.toExponential(2)}; coverage on ${covered} surface pixels: mean ${(coverageSum / covered).toFixed(4)}, ${((100 * fullyCovered) / covered).toFixed(2)}% at 1`);
            assert(parameterWorst <= PARAMETER_PATH_TOLERANCE * Math.max(1, imageMean(testView.indirect)), `the parameter path differs from the test view by up to ${parameterWorst}`);

            // 3. The consistency check: the hello renderer's GI-only image against the test view.
            const renderer = new ForwardRenderer(device, scene, { shading, hdrFormat: 'rgba32float' });
            renderer.resize(SIZE, SIZE);
            const raster = await renderView(device, renderer, camera, { mode: 'gi' });
            checkFinite('hello GI only', raster);
            const comparison = compareImages(raster, testView.indirect);
            log(
                `raster against rays: means ${imageMean(raster).toFixed(6)} / ${imageMean(testView.indirect).toFixed(6)} (${(comparison.meanDifference * 100).toFixed(3)}%), ` +
                    `${comparison.differingPixels} pixels (${(comparison.differingShare * 100).toFixed(3)}%) differ by more than ${PIXEL_TOLERANCE * 100}%`,
            );
            assert(Math.abs(comparison.meanDifference) <= MEAN_TOLERANCE, `raster and ray-traced GI means differ by ${(comparison.meanDifference * 100).toFixed(3)}%, more than ${MEAN_TOLERANCE * 100}%`);
            assert(comparison.differingShare <= MAX_DIFFERING_SHARE, `${(comparison.differingShare * 100).toFixed(3)}% of pixels differ, more than ${MAX_DIFFERING_SHARE * 100}%`);

            // The full image and the debug views: finite, not black.
            const full = await renderView(device, renderer, camera, { mode: 'full' });
            checkFinite('hello full', full);
            const debugMeans: Record<string, number> = {};
            for (const debugView of ddgi.debugViews) {
                if (debugView.kind !== 'screen') continue;
                const image = await renderView(device, renderer, camera, { mode: 'debug', debugView: debugView.id });
                checkFinite(`debug view ${debugView.label}`, image);
                debugMeans[debugView.label] = imageMean(image);
            }
            assert(Object.keys(debugMeans).length === 3, 'expected 3 screen debug views');
            const ids = ddgi.debugViews.map((debugView) => debugView.id);
            assert(new Set(ids).size === ids.length && ids.includes(DDGI_DEBUG_VIEW.coverage), `debug view ids ${ids} must be distinct and include coverage`);

            // The full image on the page, for the runner's screenshot.
            const encoder = device.createCommandEncoder({ label: 'hello screenshot' });
            renderer.render(encoder, renderer.camera(camera), { mode: 'full' });
            const canvas = document.createElement('canvas');
            canvas.width = SIZE;
            canvas.height = SIZE;
            document.body.prepend(canvas);
            const context = canvas.getContext('webgpu')!;
            context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: 'opaque' });
            renderer.present(encoder, context.getCurrentTexture());
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            renderer.destroy();

            return {
                scene: SCENE,
                camera: camera.name,
                size: SIZE,
                frames: FRAMES,
                probes: field.grid.counts,
                shape,
                testViewIndirectMean: imageMean(testView.indirect),
                parameterPathWorstDifference: parameterWorst,
                coverage: { surfacePixels: covered, mean: coverageSum / covered, shareAtOne: fullyCovered / covered },
                rasterGiMean: imageMean(raster),
                rasterMeanDifference: comparison.meanDifference,
                rasterDifferingShare: comparison.differingShare,
                rasterDifferingPixels: comparison.differingPixels,
                rasterWorstRelativeDifference: comparison.worstDifference,
                fullMean: imageMean(full),
                debugViewMeans: debugMeans,
            };
        } finally {
            platform.dispose();
        }
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
