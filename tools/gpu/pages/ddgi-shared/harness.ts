/**
 * Shared by the DDGI GPU pages (T1.5a): a platform with a manifest scene and
 * a `DDGITechnique`, frames, and two test kernels that read the field
 * through `ddgiSampleIrradiance`, the same function the update kernel uses:
 *
 * - `FieldProbe`: E at given surface points (the furnace check);
 * - `TestView`: the engine-free test view. Per pixel of a manifest camera, a
 *   primary ray through the platform tracer finds the surface, and the pixel
 *   is `albedo / π × ddgiSampleIrradiance(...)` (indirect), and also
 *   `emission + albedo / π × (direct + field)` (full). Primary rays here are a
 *   test tool only, never a shading path.
 *
 * Not a page itself (no index.html).
 */

import type { FrameBudget, FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { InstanceHandle, LightHandle, MaterialHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { EnvironmentDescriptor, LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique, type DDGIField, type DDGITechniqueOptions } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import type { DDGISettingsInput } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { DDGI_SAMPLE_WGSL } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/sample.ts';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import type { CameraBookmark, ManifestLight, SceneManifest } from '../../../../scenes/src/manifest.ts';
import { describeProceduralScene } from '../../../../scenes/src/platform-scene.ts';
import type { ProceduralScene } from '../../../../scenes/src/scene-data.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';

const manifests = import.meta.glob('../../../../scenes/manifests/*.json', { eager: true, import: 'default' }) as Record<string, SceneManifest>;

export function loadManifest(id: string): SceneManifest {
    const manifest = Object.entries(manifests).find(([path]) => path.endsWith(`/${id}.json`))?.[1];
    if (!manifest) throw new Error(`no manifest '${id}'`);
    return manifest;
}

export { CORNELL_POINT_IN_AIR_VOLUME } from './volumes.ts';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** A frame with this budget: a number of units, or a whole FrameBudget (gpuMs). */
export function frameInput(budget: number | FrameBudget): FrameInput {
    return { deltaTime: 1 / 60, budget: typeof budget === 'number' ? { units: budget } : budget, camera: { position: [0, 0, 0], view: IDENTITY, projection: IDENTITY, near: 0.1, far: 100 } };
}

export interface SceneOptions {
    /**
     * Light point and spot lights with no range window (`range: 0`), as Cycles
     * does: the Cycles references have no window (tools/reference/README.md).
     */
    readonly ignoreLightRange?: boolean;
}

function toPlatformLight(light: ManifestLight, options: SceneOptions): LightDescriptor {
    const common = { color: light.color, intensity: light.intensity, castsShadows: light.castsShadows ?? true, label: light.label };
    if (light.type === 'directional') return { ...common, type: 'directional', direction: light.direction };
    const range = options.ignoreLightRange ? 0 : light.range;
    if (light.type === 'point') return { ...common, type: 'point', position: light.position, range };
    const radians = Math.PI / 180;
    return {
        ...common,
        type: 'spot',
        position: light.position,
        direction: light.direction,
        range,
        innerConeAngle: light.innerConeDeg * radians,
        outerConeAngle: light.outerConeDeg * radians,
    };
}

function toPlatformEnvironment(manifest: SceneManifest): EnvironmentDescriptor {
    const environment = manifest.environment;
    if (environment.type === 'none') return { type: 'none' };
    if (environment.type === 'color') return { type: 'color', color: environment.color, intensity: environment.intensity };
    return platformSkyEnvironment(manifest);
}

/**
 * Describes a procedural manifest scene through the scene API, with its
 * textures, groups and slot materials (scenes/src/platform-scene.ts). Lights come
 * from the manifest, as every renderer takes them; their handles are pushed
 * to `lightHandles` (in manifest order) when given, for pages that change them.
 */
export function describeScene(
    platform: GIPlatformImpl,
    manifest: SceneManifest,
    options: SceneOptions = {},
    lightHandles: LightHandle[] = [],
    materialHandles: MaterialHandle[] = [],
    instanceHandles: Map<string, InstanceHandle> = new Map(),
): ProceduralScene {
    const data = mergeScenes(buildProceduralParts(manifest));
    const scene = platform.scene;
    // Everything the scene data says (textures, uv sets, groups, slot materials, alpha,
    // metalness, glass); a field the conversion can't pass throws, naming it.
    const handles = describeProceduralScene(scene, data);
    materialHandles.push(...handles.materials);
    for (const [name, instance] of handles.instances) instanceHandles.set(name, instance);
    for (const light of manifest.lights) lightHandles.push(scene.createLight(toPlatformLight(light, options)));
    scene.setEnvironment(toPlatformEnvironment(manifest));
    return data;
}

export interface DdgiScene {
    readonly platform: GIPlatformImpl;
    readonly ddgi: DDGITechnique;
    readonly manifest: SceneManifest;
    readonly data: ProceduralScene;
    /** The manifest's lights, in manifest order. */
    readonly lights: readonly LightHandle[];
    /** The scene's materials, in `data.materials` order. */
    readonly materials: readonly MaterialHandle[];
    /** The scene's instances by name (`data.instances`), for pages that move them. */
    readonly instances: ReadonlyMap<string, InstanceHandle>;
}

/**
 * A platform with the scene and DDGI on the manifest's volume (unless settings say otherwise), prepared, trees built.
 * The technique's warm-up is off unless `settings.warmUpPasses` asks for it, so a page's frames run at the
 * units it gives them from the first (the pages measure from black, or converge in their own way).
 */
export async function createDdgiScene(
    device: GPUDevice,
    manifestId: string,
    settings: Partial<DDGISettingsInput> = {},
    options: SceneOptions & { readonly gpuTiming?: boolean; readonly technique?: DDGITechniqueOptions } = {},
): Promise<DdgiScene> {
    const manifest = loadManifest(manifestId);
    const platform = new GIPlatformImpl(device, { label: `ddgi ${manifestId}`, gpuTiming: options.gpuTiming });
    const lights: LightHandle[] = [];
    const materials: MaterialHandle[] = [];
    const instances = new Map<string, InstanceHandle>();
    const data = describeScene(platform, manifest, options, lights, materials, instances);
    const volume = settings.volume ?? { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
    const ddgi = platform.addTechnique(new DDGITechnique({ warmUpPasses: 0, ...settings, volume }, options.technique));
    platform.setOutput(ddgi);
    await platform.prepare();
    // A frame with no probe updates places the scene; then wait for the mesh trees and place them.
    runFrame(device, platform, 0);
    await platform.rays.whenBuilt();
    runFrame(device, platform, 0);
    if (platform.rays.building) throw new Error('the platform still reports trees building');
    return { platform, ddgi, manifest, data, lights, materials, instances };
}

export function runFrame(device: GPUDevice, platform: GIPlatformImpl, budget: number | FrameBudget): void {
    const encoder = device.createCommandEncoder({ label: 'ddgi test frame' });
    platform.encode(encoder, frameInput(budget));
    device.queue.submit([encoder.finish()]);
}

/** Runs frames, waiting for the GPU every few so the page never sits in one long task. */
export async function runFrames(device: GPUDevice, platform: GIPlatformImpl, frames: number, units: number): Promise<void> {
    for (let index = 0; index < frames; index++) {
        runFrame(device, platform, units);
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();
}

export async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'ddgi test readback', size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'ddgi test readback' });
    encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** The field's resources as test kernels bind them (group 1). */
export const FIELD_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi test field',
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        // The volume table (rgba32float, read with textureLoad).
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
    ],
};

/**
 * Declarations matching FIELD_GROUP, and a call of the sampling function
 * renderers use (`ddgiSampleVolumes`: every volume, blended by priority; one
 * volume's answer is exactly `ddgiSampleIrradiance`'s). `testField` holds the
 * first volume's four parameters, for the diagnostics that read one volume.
 */
const FIELD_WGSL = /* wgsl */ `
${DDGI_SAMPLE_WGSL}

struct TestFieldParameters {
    volumeOrigin: vec4f,
    volumeSpacing: vec4f,
    volumeCounts: vec4f,
    volumeTexels: vec4f,
}
@group(1) @binding(0) var<uniform> testField: TestFieldParameters;
@group(1) @binding(1) var testIrradiance: texture_2d_array<f32>;
@group(1) @binding(2) var testDistance: texture_2d_array<f32>;
@group(1) @binding(3) var testProbeData: texture_2d_array<f32>;
@group(1) @binding(4) var testSampler: sampler;
@group(1) @binding(5) var testVolumeTable: texture_2d<f32>;

fn testFieldSample(position: vec3f, normal: vec3f, viewDirection: vec3f) -> DdgiVolumesSample {
    return ddgiSampleVolumes(
        position, normal, viewDirection,
        testIrradiance, testSampler, testDistance, testSampler, testProbeData, testVolumeTable,
    );
}

fn testFieldIrradiance(position: vec3f, normal: vec3f, viewDirection: vec3f) -> vec3f {
    return testFieldSample(position, normal, viewDirection).irradiance;
}
`;

export function fieldBindGroup(device: GPUDevice, layout: GPUBindGroupLayout, field: DDGIField, parameters: GPUBuffer): GPUBindGroup {
    device.queue.writeBuffer(parameters, 0, field.sampleParameters as Float32Array<ArrayBuffer>);
    const view = (texture: GPUTexture): GPUTextureView => texture.createView({ dimension: '2d-array' });
    return device.createBindGroup({
        layout,
        entries: [
            { binding: 0, resource: { buffer: parameters } },
            { binding: 1, resource: view(field.irradiance) },
            { binding: 2, resource: view(field.distance) },
            { binding: 3, resource: view(field.probeData) },
            { binding: 4, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }) },
            { binding: 5, resource: field.volumeTable.createView() },
        ],
    });
}

export const IO_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi test io',
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
};

type ComputeKernel = Awaited<ReturnType<GIPlatformImpl['kernels']['computeKernel']>>;

export interface SurfacePoint {
    readonly position: readonly [number, number, number];
    readonly normal: readonly [number, number, number];
    /** Direction the point is seen along (towards the point). */
    readonly viewDirection: readonly [number, number, number];
}

/** Evaluates `ddgiSampleIrradiance` at surface points. */
export class FieldProbe {
    private constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ComputeKernel,
    ) {}

    static async create(device: GPUDevice, platform: GIPlatformImpl): Promise<FieldProbe> {
        const kernel = await platform.kernels.computeKernel({
            label: 'ddgi test field probe',
            modules: [],
            code: /* wgsl */ `
${FIELD_WGSL}
struct TestPoint { position: vec4f, normal: vec4f, viewDirection: vec4f }
@group(2) @binding(0) var<storage, read> testPoints: array<TestPoint>;
@group(2) @binding(1) var<storage, read_write> testResults: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&testPoints)) {
        return;
    }
    let point = testPoints[id.x];
    testResults[id.x] = vec4f(testFieldIrradiance(point.position.xyz, normalize(point.normal.xyz), normalize(point.viewDirection.xyz)), 0.0);
}
`,
            entryPoint: 'main',
            ownGroups: [FIELD_GROUP, IO_GROUP],
        });
        return new FieldProbe(device, kernel);
    }

    /** Irradiance E (rgb) at each point. */
    async sample(field: DDGIField, points: readonly SurfacePoint[]): Promise<[number, number, number][]> {
        const device = this.device;
        const inputs = new Float32Array(points.length * 12);
        points.forEach((point, index) => {
            inputs.set(point.position, index * 12);
            inputs.set(point.normal, index * 12 + 4);
            inputs.set(point.viewDirection, index * 12 + 8);
        });
        const inputBuffer = device.createBuffer({ size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(inputBuffer, 0, inputs);
        const outputBytes = points.length * 16;
        const outputBuffer = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const parameters = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder({ label: 'ddgi test field probe' });
        const pass = encoder.beginComputePass();
        this.kernel.bind(pass);
        pass.setBindGroup(1, fieldBindGroup(device, this.kernel.ownLayouts[0], field, parameters));
        pass.setBindGroup(
            2,
            device.createBindGroup({
                layout: this.kernel.ownLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                ],
            }),
        );
        pass.dispatchWorkgroups(Math.ceil(points.length / 64));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const values = new Float32Array(await readBuffer(device, outputBuffer, outputBytes));
        inputBuffer.destroy();
        outputBuffer.destroy();
        parameters.destroy();
        return points.map((_, index) => [values[index * 4], values[index * 4 + 1], values[index * 4 + 2]]);
    }
}

export interface TestViewImages {
    readonly width: number;
    readonly height: number;
    /** albedo / π × field, rgb per pixel, row 0 at the top. */
    readonly indirect: Float32Array;
    /** emission + albedo / π × (direct + field). */
    readonly full: Float32Array;
    /** The surface's GI albedo (0 on sky pixels), e.g. to tell materials apart in a comparison. */
    readonly albedo: Float32Array;
    /** The surface point in world space (0 on sky pixels), e.g. to split a comparison by region. */
    readonly position: Float32Array;
}

/** Images the test view writes per pixel (indirect, full, albedo, position), one vec4f each. */
const TEST_VIEW_IMAGES = 4;

/**
 * How the test view reads the field:
 * - `field`: `ddgiSampleIrradiance`, as renderers do (the default);
 * - `tracedVisibility`: a diagnostic. The same weights, except that each
 *   probe's Chebyshev test is replaced by a ray from the lookup point to the
 *   probe (a blocked probe gets no weight). It tells how much of an error is
 *   the distance moments' fault and how much the probes' own values;
 * - `shortRangeOcclusion`: a diagnostic. The field's value times a
 *   short-range occlusion factor (SHORT_RANGE_OCCLUSION_WGSL).
 */
export type TestViewLookup = 'field' | 'tracedVisibility' | 'shortRangeOcclusion';

/** Diagnostic only (TestViewLookup 'tracedVisibility'); never a shading path. */
const TRACED_VISIBILITY_WGSL = /* wgsl */ `
fn testFieldIrradianceTracedVisibility(position: vec3f, normal: vec3f, viewDirection: vec3f) -> vec3f {
    let volume = ddgiVolume(testField.volumeOrigin, testField.volumeSpacing, testField.volumeCounts, testField.volumeTexels);
    let samplePoint = ddgiBiasedPoint(volume, position, normal, viewDirection);
    let cage = ddgiCage(volume, samplePoint);
    let irradianceScale = ddgiAtlasScale(volume, volume.irradianceTexels);
    let normalInTile = ddgiTileTexel(normal, volume.irradianceTexels);
    var irradianceSum = vec3f(0.0);
    var weightSum = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let step = ddgiCageStep(corner);
        let probe = min(cage.baseProbe + step, volume.counts - 1);
        let data = ddgiReadProbeData(testProbeData, volume, probe);
        if (data.state != DDGI_STATE_ACTIVE) {
            continue;
        }
        let probePosition = volume.origin + (vec3f(probe) + data.offset) * volume.spacing;
        let toProbe = probePosition - samplePoint;
        let probeDistance = length(toProbe);
        if (probeDistance > DDGI_MIN_PROBE_DISTANCE && giTraceAny(GiRay(samplePoint, 0.0, toProbe / probeDistance, probeDistance))) {
            continue;
        }
        let facing = (dot(normalize(probePosition - position), normal) + 1.0) * 0.5;
        let weight = (facing * facing + DDGI_DIRECTION_WEIGHT_FLOOR) * ddgiTrilinearWeight(cage, step);
        irradianceSum += weight * textureSampleLevel(testIrradiance, testSampler, ddgiTileUv(normalInTile, probe, irradianceScale), ddgiAtlasLayer(volume, probe), 0.0).rgb;
        weightSum += weight;
    }
    return irradianceSum / max(weightSum, DDGI_MIN_WEIGHT_SUM);
}
`;

/**
 * Diagnostic only (TestViewLookup 'shortRangeOcclusion'); never a shading
 * path. The field's value times a short-range occlusion factor: of
 * SHORT_RANGE_RAYS cosine-distributed rays out to half the smallest
 * spacing, each one that hits something takes away (1 − the albedo there),
 * a one-bounce guess at what the occluder sends back. It estimates what a
 * near-surface term (ambient occlusion at the scale the probes can't
 * resolve) would do.
 */
const SHORT_RANGE_RAYS = 32;
const SHORT_RANGE_OCCLUSION_WGSL = /* wgsl */ `
// The ray count arrives at run time (testCamera.size.z), not as a constant: a constant
// bound invites the shader compiler to unroll a loop with a whole traversal in
// it, which can make the compile very slow.
fn testShortRangeRays() -> u32 {
    return testCamera.size.z;
}
const TEST_GOLDEN_ANGLE: f32 = 2.39996322972865;

fn testFieldIrradianceShortRangeOcclusion(position: vec3f, normal: vec3f, viewDirection: vec3f) -> vec3f {
    let field = testFieldIrradiance(position, normal, viewDirection);
    let volume = ddgiVolume(testField.volumeOrigin, testField.volumeSpacing, testField.volumeCounts, testField.volumeTexels);
    let radius = 0.5 * min(volume.spacing.x, min(volume.spacing.y, volume.spacing.z));
    let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
    let tangent = normalize(cross(helper, normal));
    let bitangent = cross(normal, tangent);
    let origin = giShadeOffsetRayOrigin(position, normal);
    // A per-pixel rotation of the spiral, from the position's bits.
    let rotation = f32(bitcast<u32>(position.x * 7.0 + position.y * 13.0 + position.z * 17.0) & 1023u) / 1024.0 * 6.28318530717959;
    var occlusion = 0.0;
    for (var index = 0u; index < testShortRangeRays(); index++) {
        // Cosine-distributed spiral (Malley's method on a Fibonacci disc).
        let u = (f32(index) + 0.5) / f32(testShortRangeRays());
        let angle = f32(index) * TEST_GOLDEN_ANGLE + rotation;
        let discRadius = sqrt(u);
        let direction = normalize(tangent * (discRadius * cos(angle)) + bitangent * (discRadius * sin(angle)) + normal * sqrt(max(0.0, 1.0 - u)));
        let ray = GiRay(origin, 0.0, direction, radius);
        let hit = giTraceClosest(ray);
        if (hit.t >= 0.0) {
            let albedo = giSurfaceAt(ray, hit).albedo;
            occlusion += 1.0 - (albedo.r + albedo.g + albedo.b) / 3.0;
        }
    }
    return field * (1.0 - occlusion / f32(testShortRangeRays()));
}
`;

/**
 * The engine-free test view. Camera rays come from an explicit basis
 * (forward, right, up from position, target and vertical fov), not an
 * inverse view-projection matrix, which loses precision (NOTES).
 */
export class TestView {
    private constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ComputeKernel,
        // Rays per pixel of the shortRangeOcclusion lookup (unused by the others).
        private readonly shortRangeRays: number,
    ) {}

    static async create(device: GPUDevice, platform: GIPlatformImpl, lookup: TestViewLookup = 'field', shortRangeRays = SHORT_RANGE_RAYS): Promise<TestView> {
        const lookupFunction = { field: 'testFieldIrradiance', tracedVisibility: 'testFieldIrradianceTracedVisibility', shortRangeOcclusion: 'testFieldIrradianceShortRangeOcclusion' }[lookup];
        if (!lookupFunction) throw new Error(`unknown test view lookup '${String(lookup)}'`);
        const kernel = await platform.kernels.computeKernel({
            label: `ddgi test view (${lookup})`,
            modules: [platform.shade.module],
            code: /* wgsl */ `
${FIELD_WGSL}
${{ field: '', tracedVisibility: TRACED_VISIBILITY_WGSL, shortRangeOcclusion: SHORT_RANGE_OCCLUSION_WGSL }[lookup]}
struct TestCamera {
    position: vec4f,
    forward: vec4f,   // xyz: unit forward; w: tan(fovY / 2)
    right: vec4f,     // xyz: unit right; w: width / height
    up: vec4f,        // xyz: unit up
    size: vec4u,      // width, height, short-range occlusion rays (shortRangeOcclusion only)
}
@group(2) @binding(0) var<storage, read> testCamera: TestCamera;
@group(2) @binding(1) var<storage, read_write> testPixels: array<vec4f>;

const TEST_INV_PI: f32 = 0.318309886183791;
// Images written per pixel, interleaved: indirect, full, albedo, position.
const TEST_IMAGES: u32 = ${TEST_VIEW_IMAGES}u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = testCamera.size.xy;
    if (any(id.xy >= size)) {
        return;
    }
    // Pixel centre in [-1, 1], y up; row 0 is the top of the image.
    let ndc = vec2f((f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(size.y) * 2.0);
    let tanHalfFov = testCamera.forward.w;
    let direction = normalize(testCamera.forward.xyz + testCamera.right.xyz * ndc.x * tanHalfFov * testCamera.right.w + testCamera.up.xyz * ndc.y * tanHalfFov);
    let ray = GiRay(testCamera.position.xyz, 0.0, direction, 1.0e27);
    let hit = giTraceClosest(ray);
    var indirect = vec3f(0.0);
    var full = vec3f(0.0);
    var albedo = vec3f(0.0);
    var position = vec3f(0.0);
    if (hit.t >= 0.0) {
        let surface = giSurfaceAt(ray, hit);
        let field = ${lookupFunction}(surface.position, surface.normal, direction);
        let random = vec2f(fract(f32(id.x) * 0.7548776662), fract(f32(id.y) * 0.5698402910));
        let direct = giDirectIrradiance(surface, 0xfu, random);
        indirect = surface.albedo * TEST_INV_PI * field;
        full = surface.emission + surface.albedo * TEST_INV_PI * (direct + field);
        albedo = surface.albedo;
        position = surface.position;
    }
    let pixel = id.y * size.x + id.x;
    testPixels[TEST_IMAGES * pixel] = vec4f(indirect, 0.0);
    testPixels[TEST_IMAGES * pixel + 1u] = vec4f(full, 0.0);
    testPixels[TEST_IMAGES * pixel + 2u] = vec4f(albedo, 0.0);
    testPixels[TEST_IMAGES * pixel + 3u] = vec4f(position, 0.0);
}
`,
            entryPoint: 'main',
            ownGroups: [FIELD_GROUP, IO_GROUP],
        });
        return new TestView(device, kernel, shortRangeRays);
    }

    async render(field: DDGIField, camera: CameraBookmark, width: number, height: number): Promise<TestViewImages> {
        const device = this.device;
        const forward = normalize3(subtract(camera.target, camera.position));
        const right = normalize3(cross(forward, [0, 1, 0]));
        const up = cross(right, forward);
        const cameraData = new ArrayBuffer(80);
        const floats = new Float32Array(cameraData);
        floats.set([...camera.position, 0, ...forward, Math.tan((camera.fovYDeg * Math.PI) / 360), ...right, width / height, ...up, 0]);
        new Uint32Array(cameraData, 64, 3).set([width, height, this.shortRangeRays]);
        const cameraBuffer = device.createBuffer({ size: cameraData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(cameraBuffer, 0, cameraData);
        const outputBytes = width * height * TEST_VIEW_IMAGES * 16;
        const outputBuffer = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const parameters = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder({ label: 'ddgi test view' });
        const pass = encoder.beginComputePass();
        this.kernel.bind(pass);
        pass.setBindGroup(1, fieldBindGroup(device, this.kernel.ownLayouts[0], field, parameters));
        pass.setBindGroup(
            2,
            device.createBindGroup({
                layout: this.kernel.ownLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: cameraBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                ],
            }),
        );
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const pixels = new Float32Array(await readBuffer(device, outputBuffer, outputBytes));
        cameraBuffer.destroy();
        outputBuffer.destroy();
        parameters.destroy();
        const [indirect, full, albedo, position] = Array.from({ length: TEST_VIEW_IMAGES }, () => new Float32Array(width * height * 3));
        const images = [indirect, full, albedo, position];
        for (let pixel = 0; pixel < width * height; pixel++) {
            images.forEach((image, imageIndex) => {
                const base = (pixel * TEST_VIEW_IMAGES + imageIndex) * 4;
                for (let channel = 0; channel < 3; channel++) image[pixel * 3 + channel] = pixels[base + channel];
            });
        }
        return { width, height, indirect, full, albedo, position };
    }
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

/** Mean of an rgb image, per channel and overall (luminance-free: the plain average of all values). */
export function imageMean(rgb: Float32Array): number {
    let sum = 0;
    for (const value of rgb) sum += value;
    return sum / rgb.length;
}

/** Floats as base64, for handing an image to Node through the report. */
export function floatsToBase64(values: Float32Array): string {
    const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
    let binary = '';
    const chunk = 0x8000;
    for (let offset = 0; offset < bytes.length; offset += chunk) binary += String.fromCharCode(...bytes.subarray(offset, offset + chunk));
    return btoa(binary);
}

/**
 * A whole texture (every layer) read back as raw bytes, rows packed tightly
 * (the copy itself pads rows to 256 bytes, which this removes).
 */
export async function readTextureBytes(device: GPUDevice, texture: GPUTexture, bytesPerTexel: number): Promise<Uint8Array> {
    const rowBytes = texture.width * bytesPerTexel;
    const paddedRowBytes = Math.ceil(rowBytes / 256) * 256;
    const layers = texture.depthOrArrayLayers;
    const buffer = device.createBuffer({ size: paddedRowBytes * texture.height * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder({ label: 'ddgi test texture readback' });
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: paddedRowBytes, rowsPerImage: texture.height }, { width: texture.width, height: texture.height, depthOrArrayLayers: layers });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buffer.getMappedRange());
    const packed = new Uint8Array(rowBytes * texture.height * layers);
    for (let row = 0; row < texture.height * layers; row++) packed.set(padded.subarray(row * paddedRowBytes, row * paddedRowBytes + rowBytes), row * rowBytes);
    buffer.unmap();
    buffer.destroy();
    return packed;
}

export function halfToFloat(half: number): number {
    const sign = half & 0x8000 ? -1 : 1;
    const exponent = (half >> 10) & 0x1f;
    const fraction = half & 0x3ff;
    if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
    if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
    return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export interface ProbeRecord {
    /** Relocation offset in spacings. */
    readonly offset: [number, number, number];
    readonly state: number;
}

/** Every probe's offset and state, in probe-index order (p = x + nx * (z + nz * y)). */
export async function readProbeData(device: GPUDevice, field: DDGIField): Promise<ProbeRecord[]> {
    const halves = new Uint16Array((await readTextureBytes(device, field.probeData, 8)).buffer);
    const [nx, ny, nz] = field.grid.counts;
    const probes: ProbeRecord[] = [];
    for (let y = 0; y < ny; y++) {
        for (let z = 0; z < nz; z++) {
            for (let x = 0; x < nx; x++) {
                // Layer y, row z, column x.
                const base = ((y * nz + z) * nx + x) * 4;
                probes.push({ offset: [halfToFloat(halves[base]), halfToFloat(halves[base + 1]), halfToFloat(halves[base + 2])], state: Math.round(halfToFloat(halves[base + 3])) });
            }
        }
    }
    return probes;
}

/**
 * Destroys the page's device when the runner moves on, so the next page in
 * the same browser doesn't wait on this one's GPU memory (NOTES, "Release the
 * device between pages").
 */
export function releaseDeviceOnPageHide(device: GPUDevice): void {
    window.addEventListener('pagehide', () => device.destroy(), { once: true });
}
