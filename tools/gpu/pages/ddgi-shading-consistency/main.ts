/**
 * T1.6: DDGI's shading module, checked on a real GPU, all on one converged
 * field (the point-light Cornell box, bookmark 1, 600 frames with every
 * probe each frame, as ddgi-test-view):
 *
 * 1. **The module's shape:** textures, samplers and `vec4f` uniforms only, at
 *    most 6 textures, each sampler named `<texture>Sampler` and belonging to a
 *    filterable texture, no `@group` in `code`, the volume table's one row
 *    (read back) equal to the field's sampling parameters (since T3.5 the
 *    volume is in the table, not in uniforms), `version` unchanged by frames.
 * 2. **The parameter path** (how an engine such as three's `wgslFn` calls
 *    it): compute kernels declare the module's resources under their own
 *    names and call the entry points with them in parameter order.
 *    - At the engine-free test view's primary-ray hits,
 *      `albedo / π × giIrradiance` must equal the test view's indirect image
 *      (the same function on the same inputs).
 *    - **Coverage** at chosen points: inside the volume (1), in the fade band
 *      past the outermost probes (the edge formula), past the margin (0), and
 *      inside a solid block, where no active probe counts (the Σw term
 *      fades it to 0). Every point must also match
 *      `edge × saturate(Σw / COVERAGE_FULL_WEIGHT)` worked out from the
 *      kernel's own Σw.
 * 3. **The consistency check** (the globals path): the hello example's
 *    forward renderer, `globalsCode(1)` bound as globals, GI-only view
 *    (`albedo / π × giIrradianceAt(...)`, rasterized), against the test
 *    view's indirect image (primary rays) at the same camera and size. They
 *    share `ddgiSampleVolumes`, so they differ only where rasterization
 *    and ray tracing disagree about which surface a pixel centre sees.
 *    Tolerances: `MEAN_TOLERANCE`, `PIXEL_TOLERANCE`, `MAX_DIFFERING_SHARE`.
 * 4. **The debug views,** rasterized by the hello renderer, each against what
 *    it should show at the test view's hits:
 *    - "GI only": `giIrradiance` seen head-on, ÷ π (from the parameter path);
 *    - "Coverage": the coverage colours mixed by `giCoverage` (parameter path);
 *    - "Probes near this point": worked out on the CPU from the probe-data
 *      texture read back, the hit points and the cage rule; it must also
 *      show states other than active somewhere (Cornell's solid blocks), so
 *      it can't pass by looking like the coverage view.
 *
 * It also renders the full image (direct light with its shadow map, plus
 * GI), checks it is finite and not black, and draws it on the page for the
 * runner's screenshot.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { ShadingModule, ShadingParameter } from '../../../../packages/ddgi-core/src/platform/contract/shading-module.ts';
import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { DDGIField } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { PROBE_STATE } from '../../../../packages/ddgi-core/src/techniques/ddgi/layout.ts';
import { VOLUME_TABLE_ROW_FLOATS, VOLUME_TABLE_TEXEL } from '../../../../packages/ddgi-core/src/techniques/ddgi/volumes.ts';
import { COVERAGE_FADE_SPACINGS, COVERAGE_FULL_MARGIN_SPACINGS, COVERAGE_FULL_WEIGHT, DDGI_DEBUG_COLORS, DDGI_DEBUG_VIEW } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/shading.ts';
import { ForwardRenderer, type RenderView } from '../../../../examples/webgpu/hello/src/renderer.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { DDGI_SHADING_PARAMETER_NAMES } from '../../../../packages/ddgi-core/src/techniques/ddgi/shading.ts';
import { imageMean, readProbeData, readTextureBytes, releaseDeviceOnPageHide, runFrames, TestView, type ProbeRecord } from '../ddgi-shared/harness.ts';
import { createHelloScene, readRgba32FloatAsRgb } from '../ddgi-shading-shared/hello-scene.ts';

const SCENE = 'cornell-box-point';
const CAMERA = 1;
const SIZE = 512;
const FRAMES = 600;
/** The contract's rule for now (docs/api/platform.md, walk-through gap 1). */
const MAX_TEXTURES = 6;

/** Raster against rays: the images' means may differ by at most this share. */
const MEAN_TOLERANCE = 0.002;
/** A pixel "differs" when a channel is off by more than this share of its value (plus a floor, below). */
const PIXEL_TOLERANCE = 0.01;
/** ... or by more than this share of the image mean, so near-black pixels don't count as different over rounding. */
const PIXEL_FLOOR_OF_MEAN = 0.01;
/** Debug colours (0..1): a pixel differs when a channel is off by more than this. */
const COLOR_TOLERANCE = 0.01;
/** At most this share of pixels may differ: silhouettes, where raster and rays see different surfaces. */
const MAX_DIFFERING_SHARE = 0.01;
/** The parameter path against the test view: the same function on the same inputs. */
const PARAMETER_PATH_TOLERANCE = 1e-5;
/** Coverage against the formula worked out from the kernel's own Σw. */
const COVERAGE_TOLERANCE = 1e-5;
/** The probe-state view must show a non-active probe on at least this many pixels (Cornell's blocks). */
const MIN_NON_ACTIVE_PIXELS = 100;

type Vec3 = readonly [number, number, number];

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

type ComputeKernel = Awaited<ReturnType<GIPlatformImpl['kernels']['computeKernel']>>;

const IO_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'test io',
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
};

/** What the pixel kernel writes per pixel, in vec4s. */
const PIXEL_VEC4S = 4;

/** Per pixel of the test view's camera, from the parameter path. */
interface ParameterPathPixels {
    /** albedo / π × giIrradiance(p, n, ray direction), rgb. */
    readonly indirect: Float32Array;
    /** giCoverage(p, n); 0 where the ray misses. */
    readonly coverage: Float32Array;
    /** giIrradiance(p, n, -n) / π: what the "GI only" debug view should show; 0 where the ray misses. */
    readonly giOnlyHeadOn: Float32Array;
    /** The hit point and shading normal, xyz; `hit` 1 where the ray hit something. */
    readonly positions: Float32Array;
    readonly normals: Float32Array;
    readonly hit: Uint8Array;
}

/**
 * Kernels that call the module the way an engine would: its resources
 * declared under names of the engine's own (`engineResource<i>`), in
 * parameter order, at group 1.
 */
class ParameterPath {
    private constructor(
        private readonly device: GPUDevice,
        private readonly shading: ShadingModule,
        private readonly pixelKernel: ComputeKernel,
        private readonly pointKernel: ComputeKernel,
    ) {}

    static async create(device: GPUDevice, platform: GIPlatformImpl, shading: ShadingModule): Promise<ParameterPath> {
        const declarations: string[] = [];
        const layoutEntries: GPUBindGroupLayoutEntry[] = [];
        const callArguments: string[] = [];
        const uniformFields: string[] = [];
        // The engine's name for each module parameter, for calls that pass some of them (the point kernel).
        const engineNames = new Map<string, string>();
        for (const parameter of shading.parameters) {
            const engineName = `engineResource${callArguments.length}`;
            engineNames.set(parameter.name, engineName);
            if (parameter.kind === 'vec4f') {
                uniformFields.push(`${engineName}: vec4f,`);
                callArguments.push(`engineUniforms.${engineName}`);
                continue;
            }
            const binding = layoutEntries.length;
            if (parameter.kind === 'texture') {
                const type = parameter.dimension === '2d-array' ? 'texture_2d_array<f32>' : 'texture_2d<f32>';
                declarations.push(`@group(1) @binding(${binding}) var ${engineName}: ${type};`);
                layoutEntries.push({ binding, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: parameter.sampleType, viewDimension: parameter.dimension } });
            } else {
                declarations.push(`@group(1) @binding(${binding}) var ${engineName}: sampler;`);
                layoutEntries.push({ binding, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } });
            }
            callArguments.push(engineName);
        }
        // WGSL has no empty structs: a uniform buffer only when the module has vec4f parameters (DDGI's has none since T3.5).
        const uniformBinding = layoutEntries.length;
        if (uniformFields.length > 0) layoutEntries.push({ binding: uniformBinding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
        const resources = callArguments.join(', ');
        const uniformDeclarations =
            uniformFields.length > 0
                ? `struct EngineUniforms {\n    ${uniformFields.join('\n    ')}\n}\n@group(1) @binding(${uniformBinding}) var<uniform> engineUniforms: EngineUniforms;`
                : '';
        const engineDeclarations = /* wgsl */ `
${shading.code}
${declarations.join('\n')}
${uniformDeclarations}
`;
        // The summed probe weight the coverage fades by, from the one volume's (the table's row 0) field, seen head-on.
        const names = DDGI_SHADING_PARAMETER_NAMES;
        const atlasArguments = [names.irradiance, names.irradianceSampler, names.distance, names.distanceSampler, names.probeData].map((name) => engineNames.get(name)).join(', ');
        const firstVolume = `ddgiTableVolume(${engineNames.get(names.volumeTable)}, 0u)`;
        const ownGroups = [{ label: 'engine resources', entries: layoutEntries }, IO_GROUP];

        const pixelKernel = await platform.kernels.computeKernel({
            label: 'ddgi shading parameter path (pixels)',
            modules: [platform.shade.module],
            code: /* wgsl */ `
${engineDeclarations}
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
    let base = (id.y * size.x + id.x) * ${PIXEL_VEC4S}u;
    if (hit.t < 0.0) {
        for (var slot = 0u; slot < ${PIXEL_VEC4S}u; slot++) {
            testPixels[base + slot] = vec4f(0.0);
        }
        return;
    }
    let surface = giSurfaceAt(ray, hit);
    let screenUV = (vec2f(id.xy) + 0.5) / vec2f(size);
    let irradiance = giIrradiance(surface.position, surface.normal, direction, screenUV, ${resources});
    let coverage = giCoverage(surface.position, surface.normal, ${resources});
    let headOn = giIrradiance(surface.position, surface.normal, -surface.normal, screenUV, ${resources});
    testPixels[base] = vec4f(surface.albedo * 0.318309886183791 * irradiance, coverage);
    testPixels[base + 1u] = vec4f(surface.position, 1.0);
    testPixels[base + 2u] = vec4f(surface.normal, 0.0);
    testPixels[base + 3u] = vec4f(headOn * 0.318309886183791, 0.0);
}
`,
            entryPoint: 'main',
            ownGroups,
        });

        const pointKernel = await platform.kernels.computeKernel({
            label: 'ddgi shading parameter path (points)',
            modules: [],
            code: /* wgsl */ `
${engineDeclarations}
struct TestPoint {
    position: vec4f,
    normal: vec4f,
}
@group(2) @binding(0) var<storage, read> testPoints: array<TestPoint>;
// Per point: coverage, then the summed probe weight seen head-on (what giCoverage fades by).
@group(2) @binding(1) var<storage, read_write> testResults: array<vec2f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&testPoints)) {
        return;
    }
    let point = testPoints[id.x];
    let normal = normalize(point.normal.xyz);
    let coverage = giCoverage(point.position.xyz, normal, ${resources});
    let field = ddgiAccumulateVolume(${firstVolume}, point.position.xyz, normal, -normal, ${atlasArguments});
    testResults[id.x] = vec2f(coverage, field.weight);
}
`,
            entryPoint: 'main',
            ownGroups,
        });
        return new ParameterPath(device, shading, pixelKernel, pointKernel);
    }

    /** The module's resources, the way an engine would wrap them (group 1). */
    private resourceGroup(layout: GPUBindGroupLayout): { group: GPUBindGroup; uniformBuffer: GPUBuffer | null } {
        const device = this.device;
        const uniformFloats = new Float32Array(this.shading.parameters.filter((parameter) => parameter.kind === 'vec4f').length * 4);
        this.shading.writeUniforms(uniformFloats);
        const uniformBuffer = uniformFloats.length > 0 ? device.createBuffer({ size: uniformFloats.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }) : null;
        if (uniformBuffer) device.queue.writeBuffer(uniformBuffer, 0, uniformFloats);
        const entries: GPUBindGroupEntry[] = [];
        for (const parameter of this.shading.parameters) {
            if (parameter.kind === 'texture') entries.push({ binding: entries.length, resource: parameter.texture().createView({ dimension: parameter.dimension }) });
            else if (parameter.kind === 'sampler') entries.push({ binding: entries.length, resource: device.createSampler({ magFilter: parameter.filtering, minFilter: parameter.filtering, addressModeU: parameter.addressMode, addressModeV: parameter.addressMode }) });
        }
        if (uniformBuffer) entries.push({ binding: entries.length, resource: { buffer: uniformBuffer } });
        return { group: device.createBindGroup({ layout, entries }), uniformBuffer };
    }

    private async run(kernel: ComputeKernel, input: ArrayBuffer, outputBytes: number, workgroups: [number, number]): Promise<ArrayBuffer> {
        const device = this.device;
        const { group, uniformBuffer } = this.resourceGroup(kernel.ownLayouts[0]);
        const inputBuffer = device.createBuffer({ size: input.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(inputBuffer, 0, input);
        const output = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readback = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder({ label: 'ddgi shading parameter path' });
        const pass = encoder.beginComputePass();
        kernel.bind(pass);
        pass.setBindGroup(1, group);
        pass.setBindGroup(
            2,
            device.createBindGroup({
                layout: kernel.ownLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: output } },
                ],
            }),
        );
        pass.dispatchWorkgroups(...workgroups);
        pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const result = readback.getMappedRange().slice(0);
        readback.unmap();
        for (const gpuBuffer of [inputBuffer, output, readback]) gpuBuffer.destroy();
        uniformBuffer?.destroy();
        return result;
    }

    /** Every pixel of the camera, with the camera built exactly as TestView builds it. */
    async pixels(camera: CameraBookmark): Promise<ParameterPathPixels> {
        const forward = normalize3(subtract(camera.target, camera.position));
        const right = normalize3(cross(forward, [0, 1, 0]));
        const up = cross(right, forward);
        const cameraData = new ArrayBuffer(80);
        new Float32Array(cameraData).set([...camera.position, 0, ...forward, Math.tan((camera.fovYDeg * Math.PI) / 360), ...right, 1, ...up, 0]);
        new Uint32Array(cameraData, 64, 2).set([SIZE, SIZE]);
        const pixels = new Float32Array(await this.run(this.pixelKernel, cameraData, SIZE * SIZE * PIXEL_VEC4S * 16, [SIZE / 8, SIZE / 8]));
        const count = SIZE * SIZE;
        const result = {
            indirect: new Float32Array(count * 3),
            coverage: new Float32Array(count),
            giOnlyHeadOn: new Float32Array(count * 3),
            positions: new Float32Array(count * 3),
            normals: new Float32Array(count * 3),
            hit: new Uint8Array(count),
        };
        for (let pixel = 0; pixel < count; pixel++) {
            const base = pixel * PIXEL_VEC4S * 4;
            result.indirect.set(pixels.subarray(base, base + 3), pixel * 3);
            result.coverage[pixel] = pixels[base + 3];
            result.positions.set(pixels.subarray(base + 4, base + 7), pixel * 3);
            result.hit[pixel] = pixels[base + 7] > 0 ? 1 : 0;
            result.normals.set(pixels.subarray(base + 8, base + 11), pixel * 3);
            result.giOnlyHeadOn.set(pixels.subarray(base + 12, base + 15), pixel * 3);
        }
        return result;
    }

    /** giCoverage and the head-on Σw at each point. */
    async coverageAt(points: readonly { position: Vec3; normal: Vec3 }[]): Promise<{ coverage: number; weight: number }[]> {
        const input = new Float32Array(points.length * 8);
        points.forEach((point, index) => {
            input.set(point.position, index * 8);
            input.set(point.normal, index * 8 + 4);
        });
        const values = new Float32Array(await this.run(this.pointKernel, input.buffer, points.length * 8, [Math.ceil(points.length / 64), 1]));
        return points.map((_, index) => ({ coverage: values[index * 2], weight: values[index * 2 + 1] }));
    }
}

function subtract(a: Vec3, b: Vec3): [number, number, number] {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a: Vec3, b: Vec3): [number, number, number] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize3(a: Vec3): [number, number, number] {
    const length = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / length, a[1] / length, a[2] / length];
}

/** The volume as the shading module's four vec4f parameters give it. */
interface VolumeParameters {
    readonly origin: Vec3;
    readonly spacing: Vec3;
    readonly counts: Vec3;
    readonly normalBias: number;
    readonly viewBias: number;
}

function volumeParameters(field: DDGIField): VolumeParameters {
    const p = field.sampleParameters;
    return { origin: [p[0], p[1], p[2]], normalBias: p[3], spacing: [p[4], p[5], p[6]], viewBias: p[7], counts: [p[8], p[9], p[10]] };
}

/** The coverage edge term, on the CPU: 1 within the margin past the outermost probes, then a linear fade. */
function expectedEdgeCoverage(volume: VolumeParameters, position: Vec3): number {
    let farthest = 0;
    for (let axis = 0; axis < 3; axis++) {
        const first = volume.origin[axis];
        const last = volume.origin[axis] + (volume.counts[axis] - 1) * volume.spacing[axis];
        const outside = Math.max(first - position[axis], position[axis] - last, 0) / volume.spacing[axis];
        farthest = Math.max(farthest, outside);
    }
    return Math.min(1, Math.max(0, 1 - (farthest - COVERAGE_FULL_MARGIN_SPACINGS) / COVERAGE_FADE_SPACINGS));
}

const STATE_COLORS: Readonly<Record<number, readonly [number, number, number]>> = {
    [PROBE_STATE.active]: DDGI_DEBUG_COLORS.stateActive,
    [PROBE_STATE.offInsideGeometry]: DDGI_DEBUG_COLORS.stateInsideGeometry,
    [PROBE_STATE.offNoSurfaceNearby]: DDGI_DEBUG_COLORS.stateNoSurfaceNearby,
    [PROBE_STATE.notUpdatedYet]: DDGI_DEBUG_COLORS.stateNotUpdatedYet,
};

/**
 * What "Probes near this point" should show, on the CPU: the head-on biased
 * point's cage (clamped to the grid), each corner's state colour weighted
 * trilinearly (design doc, "Sampling the field", steps 1-2).
 */
function expectedProbeStatesColor(volume: VolumeParameters, probes: readonly ProbeRecord[], position: Vec3, normal: Vec3): [number, number, number] {
    const bias = volume.normalBias + volume.viewBias;
    const counts = volume.counts;
    const base: number[] = [];
    const alpha: number[] = [];
    for (let axis = 0; axis < 3; axis++) {
        const grid = (position[axis] + normal[axis] * bias - volume.origin[axis]) / volume.spacing[axis];
        base[axis] = Math.min(Math.max(Math.floor(grid), 0), Math.max(counts[axis] - 2, 0));
        alpha[axis] = Math.min(Math.max(grid - base[axis], 0), 1);
    }
    const color: [number, number, number] = [0, 0, 0];
    for (let corner = 0; corner < 8; corner++) {
        const step = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
        const probe = step.map((offset, axis) => Math.min(base[axis] + offset, counts[axis] - 1));
        let weight = 1;
        for (let axis = 0; axis < 3; axis++) weight *= step[axis] ? alpha[axis] : 1 - alpha[axis];
        // Probe index p = x + nx * (z + nz * y).
        const state = probes[probe[0] + counts[0] * (probe[2] + counts[2] * probe[1])].state;
        const stateColor = STATE_COLORS[state] ?? DDGI_DEBUG_COLORS.stateNotUpdatedYet;
        for (let channel = 0; channel < 3; channel++) color[channel] += weight * stateColor[channel];
    }
    return color;
}

interface Comparison {
    meanDifference: number;
    differingShare: number;
    differingPixels: number;
}

/**
 * Per-pixel comparison of two rgb images: a pixel differs when any channel
 * is off by more than `relative` of the reference's value or `absolute`,
 * whichever is larger. Per channel, so red and green of the same brightness
 * never pass for each other.
 */
function compareImages(candidate: Float32Array, reference: Float32Array, relative: number, absolute: number): Comparison {
    let differing = 0;
    const pixels = reference.length / 3;
    for (let pixel = 0; pixel < pixels; pixel++) {
        for (let channel = 0; channel < 3; channel++) {
            const a = candidate[pixel * 3 + channel];
            const b = reference[pixel * 3 + channel];
            if (Math.abs(a - b) > Math.max(relative * Math.abs(b), absolute)) {
                differing++;
                break;
            }
        }
    }
    return { meanDifference: imageMean(candidate) / imageMean(reference) - 1, differingShare: differing / pixels, differingPixels: differing };
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

function percent(share: number): string {
    return `${(share * 100).toFixed(3)}%`;
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
            assert(shading === ddgi.shading, "gi.shading is not the output technique's module");
            const versionBefore = shading.version;
            await runFrames(device, platform, FRAMES, field.grid.probeCount);
            assert(shading.version === versionBefore, 'version changed although nothing changed shape');

            // 1. Shape, and the volume table (what the module reads the volume from) against the field's own parameters.
            const shape = checkModuleShape(shading);
            const tableTexture = shading.parameters.find((parameter) => parameter.name === DDGI_SHADING_PARAMETER_NAMES.volumeTable);
            assert(tableTexture?.kind === 'texture', 'the module has no volume table');
            const tableRow = new Float32Array((await readTextureBytes(device, tableTexture.texture(), 16)).buffer);
            const fieldParameters = ddgi.field!.sampleParameters;
            // A row: the four sampling vec4s, then the window (0 for a fixed volume) and the valid extent (its first and last probes); T3.6.
            assert(tableRow.length === VOLUME_TABLE_ROW_FLOATS, `the volume table holds ${tableRow.length} floats, a row is ${VOLUME_TABLE_ROW_FLOATS}`);
            const sampling = tableRow.subarray(0, fieldParameters.length);
            assert(sampling.every((value, index) => value === fieldParameters[index]), `the volume table ${[...sampling]} differs from the field's ${[...fieldParameters]}`);
            const counts = field.grid.counts;
            const windowAndExtent = [...tableRow.subarray(VOLUME_TABLE_TEXEL.window * 4)];
            const expectedWindowAndExtent = [0, 0, 0, 0, 0, 0, 0, 0, counts[0] - 1, counts[1] - 1, counts[2] - 1, 0];
            assert(windowAndExtent.every((value, index) => value === expectedWindowAndExtent[index]), `a fixed volume's window and extent read ${windowAndExtent}, not ${expectedWindowAndExtent}`);
            for (const parameter of shading.parameters) {
                if (parameter.kind === 'vec4f') assert(parameter.value() === parameter.value(), `${parameter.name}.value() should return the same array each time`);
            }
            const volume = volumeParameters(ddgi.field!);

            // The engine-free test view: primary rays, albedo / pi x ddgiSampleIrradiance.
            const testView = await (await TestView.create(device, platform)).render(field, camera, SIZE, SIZE);

            // 2. The parameter path: the same image as the test view.
            const parameterPath = await ParameterPath.create(device, platform, shading);
            const pixels = await parameterPath.pixels(camera);
            let parameterWorst = 0;
            for (let index = 0; index < testView.indirect.length; index++) {
                parameterWorst = Math.max(parameterWorst, Math.abs(pixels.indirect[index] - testView.indirect[index]));
            }
            assert(parameterWorst <= PARAMETER_PATH_TOLERANCE * Math.max(1, imageMean(testView.indirect)), `the parameter path differs from the test view by up to ${parameterWorst}`);
            let surfacePixels = 0;
            let fullyCovered = 0;
            let coverageSum = 0;
            for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
                if (!pixels.hit[pixel]) continue;
                surfacePixels++;
                coverageSum += pixels.coverage[pixel];
                if (pixels.coverage[pixel] >= 0.999) fullyCovered++;
            }
            log(`parameter path: worst difference from the test view ${parameterWorst.toExponential(2)}; coverage on ${surfacePixels} surface pixels: mean ${(coverageSum / surfacePixels).toFixed(4)}, ${((100 * fullyCovered) / surfacePixels).toFixed(2)}% at 1`);

            // Coverage at chosen points. The open front of the box faces +z; the
            // outermost probe layer there is at lastZ, with nothing beyond it.
            // The band points sit at floor height: probes in the middle of the
            // room have no surface nearby and are off, so they'd see nothing.
            const lastZ = volume.origin[2] + (volume.counts[2] - 1) * volume.spacing[2];
            const middleX = 2.78;
            const floorY = 0;
            const fadeBandOffset = (COVERAGE_FULL_MARGIN_SPACINGS + COVERAGE_FADE_SPACINGS / 2) * volume.spacing[2];
            const pastMarginOffset = (COVERAGE_FULL_MARGIN_SPACINGS + COVERAGE_FADE_SPACINGS * 1.5) * volume.spacing[2];
            const coveragePoints = [
                { label: 'inside (floor, middle of the box)', position: [middleX, floorY, 2.8] as Vec3, normal: [0, 1, 0] as Vec3, expected: 'one' },
                { label: 'fade band (in front of the open box)', position: [middleX, floorY, lastZ + fadeBandOffset] as Vec3, normal: [0, 1, 0] as Vec3, expected: 'between' },
                { label: 'past the margin', position: [middleX, floorY, lastZ + pastMarginOffset] as Vec3, normal: [0, 1, 0] as Vec3, expected: 'zero' },
                // The tall block is solid; its centre's whole cage is off (inside geometry).
                { label: 'unseen (inside the tall block)', position: [1.865, 1.65, 2.08] as Vec3, normal: [0, 1, 0] as Vec3, expected: 'unseen' },
            ] as const;
            const coverageResults = await parameterPath.coverageAt(coveragePoints);
            const coverageReport: Record<string, unknown> = {};
            coveragePoints.forEach((point, index) => {
                const { coverage, weight } = coverageResults[index];
                const edge = expectedEdgeCoverage(volume, point.position);
                const seen = Math.min(1, Math.max(0, weight / COVERAGE_FULL_WEIGHT));
                const formula = edge * seen;
                coverageReport[point.label] = { coverage, edge, weight, formula };
                log(`coverage ${point.label}: ${coverage.toFixed(6)} (edge ${edge.toFixed(4)}, summed weight ${weight.toExponential(3)})`);
                assert(Math.abs(coverage - formula) <= COVERAGE_TOLERANCE, `coverage at ${point.label} is ${coverage}, the formula gives ${formula}`);
                if (point.expected === 'one') assert(coverage === 1, `coverage inside the volume is ${coverage}, not 1`);
                if (point.expected === 'between') {
                    assert(seen === 1, `the fade-band point should be seen by its probes (summed weight ${weight})`);
                    assert(edge > 0 && edge < 1 && Math.abs(coverage - edge) <= COVERAGE_TOLERANCE, `coverage in the fade band is ${coverage}, expected ${edge}`);
                }
                if (point.expected === 'zero') assert(edge === 0 && coverage === 0, `coverage past the margin is ${coverage}`);
                if (point.expected === 'unseen') assert(edge === 1 && weight < COVERAGE_FULL_WEIGHT && coverage < 1, `a point no probe sees has coverage ${coverage} (summed weight ${weight})`);
            });

            // 3. The consistency check: the hello renderer's GI-only image against the test view.
            const renderer = new ForwardRenderer(device, scene, { shading, hdrFormat: 'rgba32float' });
            renderer.resize(SIZE, SIZE);
            const raster = await renderView(device, renderer, camera, { mode: 'gi' });
            checkFinite('hello GI only', raster);
            const comparison = compareImages(raster, testView.indirect, PIXEL_TOLERANCE, PIXEL_FLOOR_OF_MEAN * imageMean(testView.indirect));
            log(`raster against rays: means ${imageMean(raster).toFixed(6)} / ${imageMean(testView.indirect).toFixed(6)} (${percent(comparison.meanDifference)}), ${comparison.differingPixels} pixels (${percent(comparison.differingShare)}) differ`);
            assert(Math.abs(comparison.meanDifference) <= MEAN_TOLERANCE, `raster and ray-traced GI means differ by ${percent(comparison.meanDifference)}, more than ${MEAN_TOLERANCE * 100}%`);
            assert(comparison.differingShare <= MAX_DIFFERING_SHARE, `${percent(comparison.differingShare)} of pixels differ, more than ${MAX_DIFFERING_SHARE * 100}%`);

            // 4. The debug views against what each should show.
            const screenViews = ddgi.debugViews.filter((debugView) => debugView.kind === 'screen');
            const ids = ddgi.debugViews.map((debugView) => debugView.id);
            assert(new Set(ids).size === ids.length, `debug view ids ${ids} must be distinct`);
            assert(screenViews.length === 3, 'expected 3 screen debug views');
            const probes = await readProbeData(device, field);
            const expectedCoverageImage = new Float32Array(SIZE * SIZE * 3);
            const expectedStatesImage = new Float32Array(SIZE * SIZE * 3);
            let nonActivePixels = 0;
            for (let pixel = 0; pixel < SIZE * SIZE; pixel++) {
                if (!pixels.hit[pixel]) continue;
                const coverage = pixels.coverage[pixel];
                const position: Vec3 = [pixels.positions[pixel * 3], pixels.positions[pixel * 3 + 1], pixels.positions[pixel * 3 + 2]];
                const normal: Vec3 = [pixels.normals[pixel * 3], pixels.normals[pixel * 3 + 1], pixels.normals[pixel * 3 + 2]];
                const states = expectedProbeStatesColor(volume, probes, position, normal);
                for (let channel = 0; channel < 3; channel++) {
                    const none = DDGI_DEBUG_COLORS.noCoverage[channel];
                    expectedCoverageImage[pixel * 3 + channel] = none + (DDGI_DEBUG_COLORS.fullCoverage[channel] - none) * coverage;
                    expectedStatesImage[pixel * 3 + channel] = states[channel];
                }
                if (Math.max(...states.map((value, channel) => Math.abs(value - DDGI_DEBUG_COLORS.stateActive[channel]))) > 0.05) nonActivePixels++;
            }
            assert(nonActivePixels >= MIN_NON_ACTIVE_PIXELS, `only ${nonActivePixels} pixels have a non-active probe nearby; the probe-state check would prove little`);
            const expectations: Record<number, { image: Float32Array; relative: number; absolute: number }> = {
                [DDGI_DEBUG_VIEW.giOnly]: { image: pixels.giOnlyHeadOn, relative: PIXEL_TOLERANCE, absolute: PIXEL_FLOOR_OF_MEAN * imageMean(pixels.giOnlyHeadOn) },
                [DDGI_DEBUG_VIEW.coverage]: { image: expectedCoverageImage, relative: 0, absolute: COLOR_TOLERANCE },
                [DDGI_DEBUG_VIEW.probeStates]: { image: expectedStatesImage, relative: 0, absolute: COLOR_TOLERANCE },
            };
            const debugResults: Record<string, unknown> = {};
            for (const debugView of screenViews) {
                const expected = expectations[debugView.id];
                assert(expected, `no expectation for debug view ${debugView.id} (${debugView.label})`);
                const image = await renderView(device, renderer, camera, { mode: 'debug', debugView: debugView.id });
                checkFinite(`debug view ${debugView.label}`, image);
                const result = compareImages(image, expected.image, expected.relative, expected.absolute);
                debugResults[debugView.label] = { differingShare: result.differingShare, differingPixels: result.differingPixels, mean: imageMean(image) };
                log(`debug view ${debugView.label}: ${result.differingPixels} pixels (${percent(result.differingShare)}) differ from what it should show`);
                assert(result.differingShare <= MAX_DIFFERING_SHARE, `debug view ${debugView.label}: ${percent(result.differingShare)} of pixels differ from what it should show`);
            }

            // The full image: finite, not black, and on the page for the runner's screenshot.
            const full = await renderView(device, renderer, camera, { mode: 'full' });
            checkFinite('hello full', full);
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
                coverage: { surfacePixels, mean: coverageSum / surfacePixels, shareAtOne: fullyCovered / surfacePixels, points: coverageReport },
                rasterGiMean: imageMean(raster),
                rasterMeanDifference: comparison.meanDifference,
                rasterDifferingShare: comparison.differingShare,
                rasterDifferingPixels: comparison.differingPixels,
                debugViews: debugResults,
                probeStatePixelsWithNonActiveProbes: nonActivePixels,
                fullMean: imageMean(full),
            };
        } finally {
            platform.dispose();
        }
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
