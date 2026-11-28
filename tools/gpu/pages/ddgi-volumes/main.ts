/**
 * T3.5 (docs/design/m3-moving.md §4.1, "Draft done-whens"): several DDGI
 * volumes in one technique, on a real GPU.
 *
 * The scene is two-room with two volumes: the manifest's coarse one (0.5 m,
 * the whole of both rooms) and a dense one (0.25 m, priority 1) in the lit
 * room. Two cases (CASES), each its own technique:
 *
 * - **walls in the band:** the dense volume fills the lit room, placed so
 *   that the partition (x = 4 m) and the north wall (z = 4 m) lie in the
 *   middle of its fade band; the camera faces the partition, so about half
 *   the view is in the band (the cost is measured here);
 * - **band across the room:** the dense volume fills the room's west half,
 *   so its band crosses the floor, the ceiling and both side walls at
 *   x ≈ 2.5 m, where a seam would show as a line; the camera looks down the
 *   room across it.
 *
 * In each case, after the technique's warm-up and SETTLE_FRAMES frames at
 * UNITS probes a frame (a 60 Hz clock, so the history settles), the page
 * checks:
 *
 * 1. **The priority blend.** The view is shaded three times from one field:
 *    through the full volume table, and through tables holding the dense or
 *    the coarse volume alone (the same atlases, one row each). Inside the
 *    dense volume the blend must give the dense volume's answer exactly;
 *    where the dense volume doesn't reach, the coarse one's; and in the band,
 *    `(c_d E_d + (1 − c_d) c_c E_c) / (c_d + (1 − c_d) c_c)` with coverage
 *    `c_d + (1 − c_d) c_c`, from the single-volume answers (BLEND_TOLERANCE).
 * 2. **No seam over time** (done-when 1): MEASURE_FRAMES frames, each one
 *    platform update and one shading of the view; per frame pair and pixel
 *    `mean over rgb |GI(t) − GI(t−1)| / mean(GI(t))`, as `ddgi-flicker`
 *    measures it, over the band's pixels (the mean taken over the band too),
 *    averaged over the pairs: mean ≤ 0.1% and p95 ≤ 0.5% (the stability
 *    threshold, DECISIONS 2026-11-06). The whole view and the dense-only and
 *    coarse-only parts are reported beside it.
 * 3. **No recompiles:** the platform's counter doesn't move over the frames.
 * 4. **The cost** (done-when 3, recorded, not gated; the first case only):
 *    the renderer's per-pixel calls (`giIrradianceAt` and `giCoverageAt`,
 *    the globals path) at 2560 × 1440 on the same view, timed with timestamp
 *    queries (median of COST_REPEATS), through the full table and through
 *    each volume alone, with the share of pixels in the band.
 *
 * The page draws, per case, panels for the eye (done-when 4): the GI image
 * with both volumes; the same through the coarse volume alone; which volume
 * answers (green the dense one, blue the coarse one, the band in between);
 * and the frame-to-frame change over the measured frames, brightened, so a
 * seam that flickers would show as a bright line.
 *
 *   VITE_DDGI_FRAMES=120 (measured frames), VITE_DDGI_SETTLE=1200
 */

import { assert, runGpuTest } from '../../client.ts';
import type { ShadingModule, ShadingParameter } from '../../../../packages/core/src/platform/contract/shading-module.ts';
import type { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import type { DDGIField } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import type { DDGIVolumeSettings } from '../../../../packages/core/src/techniques/ddgi/settings.ts';
import { volumeEdgeCoverage, VOLUME_TABLE_FORMAT, VOLUME_TABLE_TEXELS } from '../../../../packages/core/src/techniques/ddgi/volumes.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { createDdgiScene, readBuffer, releaseDeviceOnPageHide, runFrame } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
function countFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' ? Number(value) : NaN;
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

const SCENE = 'two-room';
/** The manifest's volume: both rooms at 0.5 m. */
const COARSE: DDGIVolumeSettings = { center: [4.1, 1.4, 2.0], size: [8.4, 3.0, 4.2], spacing: 0.5 };
/** One placement of the dense volume, and the camera that looks at its fade band. */
interface Case {
    readonly name: string;
    readonly dense: DDGIVolumeSettings;
    readonly camera: CameraBookmark;
    /** Measure the sampling cost at 1440p on this case's view. */
    readonly cost: boolean;
}
const CASES: readonly Case[] = [
    {
        name: 'walls in the band',
        // The lit room at 0.25 m: 15 × 12 × 15 probes. Its last probes on x and z sit at 3.8125 m, so the partition
        // (x = 4 m) and the north wall (z = 4 m) are 0.75 spacings out, in the middle of the fade band (0.5 to 1
        // spacing out): the dense volume answers half there. Floor and ceiling are within its margin.
        dense: { center: [2.0625, 1.4, 2.0625], size: [3.5, 2.75, 3.5], spacing: 0.25, priority: 1 },
        // Inside the lit room, facing the partition (the band) with the doorway into the unlit room (the coarse volume only).
        camera: { name: 'facing the partition', position: [0.6, 1.5, 1.0], target: [4.0, 1.2, 2.4], fovYDeg: 70 },
        cost: true,
    },
    {
        name: 'band across the room',
        // The lit room's west half at 0.25 m: 9 × 12 × 15 probes, the last on x at 2.3125 m, so the band runs
        // across the floor, the ceiling and both side walls from x = 2.4375 to 2.5625 m.
        dense: { center: [1.3125, 1.4, 2.0625], size: [2.0, 2.75, 3.5], spacing: 0.25, priority: 1 },
        // From the west wall down the room towards the partition, across the band.
        camera: { name: 'down the room', position: [0.3, 1.7, 1.2], target: [4.0, 0.9, 2.2], fovYDeg: 75 },
        cost: false,
    },
];
/** Two-room's calibrated batch on the RTX at 2 ms (T1.9b), shared between the volumes by their probe counts. */
const UNITS = 707;
const SETTLE_FRAMES = countFromEnv(env.VITE_DDGI_SETTLE, 1200);
const MEASURE_FRAMES = countFromEnv(env.VITE_DDGI_FRAMES, 120);
const WIDTH = 640;
const HEIGHT = 360;
const COST_WIDTH = 2560;
const COST_HEIGHT = 1440;
const COST_WARM_UP = 5;
const COST_REPEATS = 21;

/** The stability threshold (DECISIONS 2026-11-06): per-pixel change a frame. */
const STABILITY_MEAN = 0.001;
const STABILITY_P95 = 0.005;
/** The blend against the formula from the single-volume answers, relative to the pixel's value (plus a small floor). */
const BLEND_TOLERANCE = 1e-4;
/** At least this many of the view's pixels must be in the band (1% of 640 × 360), or check 2 would prove little. */
const MIN_BAND_PIXELS = 2304;

type Vec3 = readonly [number, number, number];
type ComputeKernel = Awaited<ReturnType<GIPlatformImpl['kernels']['computeKernel']>>;

const COMPUTE = GPUShaderStage.COMPUTE;
const GBUFFER_VEC4S = 2;

/** The camera as the engine-free test view builds it (ddgi-shared/harness.ts): an explicit basis, 80 bytes. */
function cameraData(camera: CameraBookmark, width: number, height: number): ArrayBuffer {
    const forward = normalize(subtract(camera.target, camera.position));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const data = new ArrayBuffer(80);
    new Float32Array(data).set([...camera.position, 0, ...forward, Math.tan((camera.fovYDeg * Math.PI) / 360), ...right, width / height, ...up, 0]);
    new Uint32Array(data, 64, 2).set([width, height]);
    return data;
}

/**
 * The view's surfaces, once (nothing moves): per pixel the hit point (w: 1
 * on a hit) and the shading normal (w: the albedo, pack4x8unorm). Two vec4s,
 * so 2560 × 1440 fits a default storage binding (118 MB).
 */
const GBUFFER_WGSL = /* wgsl */ `
struct TestCamera {
    position: vec4f,
    forward: vec4f,   // xyz: unit forward; w: tan(fovY / 2)
    right: vec4f,     // xyz: unit right; w: width / height
    up: vec4f,
    size: vec4u,
}
@group(1) @binding(0) var<storage, read> testCamera: TestCamera;
@group(1) @binding(1) var<storage, read_write> testSurfaces: array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = testCamera.size.xy;
    if (any(id.xy >= size)) {
        return;
    }
    let ndc = vec2f((f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(size.y) * 2.0);
    let tanHalfFov = testCamera.forward.w;
    let direction = normalize(testCamera.forward.xyz + testCamera.right.xyz * ndc.x * tanHalfFov * testCamera.right.w + testCamera.up.xyz * ndc.y * tanHalfFov);
    let ray = GiRay(testCamera.position.xyz, 0.0, direction, 1.0e27);
    let hit = giTraceClosest(ray);
    let pixel = (id.y * size.x + id.x) * ${GBUFFER_VEC4S}u;
    if (hit.t < 0.0) {
        testSurfaces[pixel] = vec4f(0.0);
        testSurfaces[pixel + 1u] = vec4f(0.0);
        return;
    }
    let surface = giSurfaceAt(ray, hit);
    testSurfaces[pixel] = vec4f(surface.position, 1.0);
    testSurfaces[pixel + 1u] = vec4f(surface.normal, bitcast<f32>(pack4x8unorm(vec4f(surface.albedo, 0.0))));
}
`;

/**
 * Shading from the G-buffer through the module's globals (`globalsCode(1)`),
 * two entry points:
 * - `shadeImage`: one `ddgiSampleVolumes` call with the true view direction,
 *   writing albedo / π × E and the blend's coverage (what checks 1 and 2 read);
 * - `shadeRenderer`: what a renderer pays per pixel, `giIrradianceAt` and
 *   `giCoverageAt` (three's per-fragment path asks for both), for the cost.
 */
function shadingWgsl(shading: ShadingModule): string {
    return /* wgsl */ `
${shading.globalsCode(1)}
struct TestCamera {
    position: vec4f,
    forward: vec4f,
    right: vec4f,
    up: vec4f,
    size: vec4u,
}
@group(2) @binding(0) var<storage, read> testCamera: TestCamera;
@group(2) @binding(1) var<storage, read> testSurfaces: array<vec4f>;
@group(2) @binding(2) var<storage, read_write> testShaded: array<vec4f>;

const TEST_INV_PI: f32 = 0.318309886183791;

struct TestSurface {
    hit: bool,
    position: vec3f,
    normal: vec3f,
    albedo: vec3f,
    viewDirection: vec3f,
}

fn testSurface(id: vec3u) -> TestSurface {
    let pixel = (id.y * testCamera.size.x + id.x) * ${GBUFFER_VEC4S}u;
    let positionAndHit = testSurfaces[pixel];
    let normalAndAlbedo = testSurfaces[pixel + 1u];
    let viewDirection = normalize(positionAndHit.xyz - testCamera.position.xyz);
    let albedo = unpack4x8unorm(bitcast<u32>(normalAndAlbedo.w)).rgb;
    return TestSurface(positionAndHit.w > 0.5, positionAndHit.xyz, normalAndAlbedo.xyz, albedo, viewDirection);
}

@compute @workgroup_size(8, 8)
fn shadeImage(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= testCamera.size.xy)) {
        return;
    }
    let surface = testSurface(id);
    var result = vec4f(0.0);
    if (surface.hit) {
        let field = ddgiSampleVolumes(
            surface.position, surface.normal, surface.viewDirection,
            ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData, ddgiVolumeTable,
        );
        result = vec4f(surface.albedo * TEST_INV_PI * field.irradiance, field.coverage);
    }
    testShaded[id.y * testCamera.size.x + id.x] = result;
}

@compute @workgroup_size(8, 8)
fn shadeRenderer(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= testCamera.size.xy)) {
        return;
    }
    let surface = testSurface(id);
    var result = vec4f(0.0);
    if (surface.hit) {
        let screenUV = (vec2f(id.xy) + 0.5) / vec2f(testCamera.size.xy);
        let irradiance = giIrradianceAt(surface.position, surface.normal, surface.viewDirection, screenUV);
        result = vec4f(irradiance, giCoverageAt(surface.position, surface.normal));
    }
    testShaded[id.y * testCamera.size.x + id.x] = result;
}
`;
}

/** The module's resources as globals at group 1, in parameter order (globalsCode's bindings). */
function moduleGroupLayout(shading: ShadingModule): GPUBindGroupLayoutDescriptor {
    const entries: GPUBindGroupLayoutEntry[] = [];
    for (const parameter of shading.parameters) {
        if (parameter.kind === 'texture') entries.push({ binding: entries.length, visibility: COMPUTE, texture: { sampleType: parameter.sampleType, viewDimension: parameter.dimension } });
        else if (parameter.kind === 'sampler') entries.push({ binding: entries.length, visibility: COMPUTE, sampler: { type: parameter.filtering === 'linear' ? 'filtering' : 'non-filtering' } });
    }
    assert(!shading.parameters.some((parameter) => parameter.kind === 'vec4f'), 'the DDGI module should have no vec4f parameters since T3.5');
    return { label: 'ddgi-volumes module', entries };
}

/** The module's resources bound, with `volumeTable` standing in for the module's own table. */
function moduleGroup(device: GPUDevice, layout: GPUBindGroupLayout, shading: ShadingModule, volumeTable: GPUTexture): GPUBindGroup {
    const entries: GPUBindGroupEntry[] = [];
    for (const parameter of shading.parameters as readonly ShadingParameter[]) {
        if (parameter.kind === 'texture') {
            const texture = parameter.name === 'ddgiVolumeTable' ? volumeTable : parameter.texture();
            entries.push({ binding: entries.length, resource: texture.createView({ dimension: parameter.dimension }) });
        } else if (parameter.kind === 'sampler') {
            entries.push({
                binding: entries.length,
                resource: device.createSampler({ magFilter: parameter.filtering, minFilter: parameter.filtering, addressModeU: parameter.addressMode, addressModeV: parameter.addressMode }),
            });
        }
    }
    return device.createBindGroup({ layout, entries });
}

/** A table holding one row of the field's volume table: that volume alone, read from the same atlases. */
function singleVolumeTable(device: GPUDevice, field: DDGIField, row: number): GPUTexture {
    const table = device.createTexture({ label: `ddgi-volumes table row ${row}`, size: { width: VOLUME_TABLE_TEXELS, height: 1 }, format: VOLUME_TABLE_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'ddgi-volumes table copy' });
    encoder.copyTextureToTexture({ texture: field.volumeTable, origin: { x: 0, y: row } }, { texture: table }, { width: VOLUME_TABLE_TEXELS, height: 1 });
    device.queue.submit([encoder.finish()]);
    return table;
}

/** A view's surfaces and the buffers its shading writes into. */
interface View {
    readonly width: number;
    readonly height: number;
    readonly camera: GPUBuffer;
    readonly surfaces: GPUBuffer;
    readonly shaded: GPUBuffer;
}

class Shader {
    private constructor(
        private readonly device: GPUDevice,
        private readonly shading: ShadingModule,
        private readonly gbufferKernel: ComputeKernel,
        private readonly imageKernel: ComputeKernel,
        private readonly rendererKernel: ComputeKernel,
    ) {}

    static async create(device: GPUDevice, platform: GIPlatformImpl): Promise<Shader> {
        const shading = platform.shading;
        assert(shading, 'the platform has no shading module');
        const ioGroup = (label: string): GPUBindGroupLayoutDescriptor => ({
            label,
            entries: [
                { binding: 0, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const gbufferKernel = await platform.kernels.computeKernel({
            label: 'ddgi-volumes surfaces',
            modules: [platform.shade.module],
            code: GBUFFER_WGSL,
            entryPoint: 'main',
            ownGroups: [
                {
                    label: 'ddgi-volumes surfaces io',
                    entries: [
                        { binding: 0, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
                        { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
                    ],
                },
            ],
        });
        const code = shadingWgsl(shading);
        const ownGroups = [moduleGroupLayout(shading), ioGroup('ddgi-volumes shading io')];
        const imageKernel = await platform.kernels.computeKernel({ label: 'ddgi-volumes shade image', modules: [], code, entryPoint: 'shadeImage', ownGroups });
        const rendererKernel = await platform.kernels.computeKernel({ label: 'ddgi-volumes shade renderer', modules: [], code, entryPoint: 'shadeRenderer', ownGroups });
        return new Shader(device, shading, gbufferKernel, imageKernel, rendererKernel);
    }

    /** The camera's surfaces at this size (traced once). */
    async view(camera: CameraBookmark, width: number, height: number): Promise<View> {
        const device = this.device;
        const cameraBuffer = device.createBuffer({ size: 80, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(cameraBuffer, 0, cameraData(camera, width, height));
        const surfaces = device.createBuffer({ size: width * height * GBUFFER_VEC4S * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const shaded = device.createBuffer({ size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const encoder = device.createCommandEncoder({ label: 'ddgi-volumes surfaces' });
        const pass = encoder.beginComputePass();
        this.gbufferKernel.bind(pass);
        pass.setBindGroup(
            1,
            device.createBindGroup({
                layout: this.gbufferKernel.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: cameraBuffer } },
                    { binding: 1, resource: { buffer: surfaces } },
                ],
            }),
        );
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        return { width, height, camera: cameraBuffer, surfaces, shaded };
    }

    private groups(kernel: ComputeKernel, view: View, volumeTable: GPUTexture): [GPUBindGroup, GPUBindGroup] {
        const device = this.device;
        return [
            moduleGroup(device, kernel.ownLayouts[0], this.shading, volumeTable),
            device.createBindGroup({
                layout: kernel.ownLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: view.camera } },
                    { binding: 1, resource: { buffer: view.surfaces } },
                    { binding: 2, resource: { buffer: view.shaded } },
                ],
            }),
        ];
    }

    private dispatch(pass: GPUComputePassEncoder, kernel: ComputeKernel, groups: [GPUBindGroup, GPUBindGroup], view: View): void {
        kernel.bind(pass);
        pass.setBindGroup(1, groups[0]);
        pass.setBindGroup(2, groups[1]);
        pass.dispatchWorkgroups(Math.ceil(view.width / 8), Math.ceil(view.height / 8));
    }

    /** albedo / π × E and the blend's coverage per pixel (rgba), through `volumeTable`. */
    async image(view: View, volumeTable: GPUTexture): Promise<Float32Array> {
        const device = this.device;
        const encoder = device.createCommandEncoder({ label: 'ddgi-volumes shade image' });
        const pass = encoder.beginComputePass();
        this.dispatch(pass, this.imageKernel, this.groups(this.imageKernel, view, volumeTable), view);
        pass.end();
        device.queue.submit([encoder.finish()]);
        return new Float32Array(await readBuffer(device, view.shaded, view.width * view.height * 16));
    }

    /** The renderer's per-pixel calls over the whole view, timed: the median of COST_REPEATS, in ms; null without timestamps. */
    async rendererMs(view: View, volumeTable: GPUTexture): Promise<number | null> {
        const device = this.device;
        if (!device.features.has('timestamp-query')) return null;
        const runs = COST_WARM_UP + COST_REPEATS;
        const querySet = device.createQuerySet({ type: 'timestamp', count: 2 * runs });
        const resolved = device.createBuffer({ size: 16 * runs, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        const groups = this.groups(this.rendererKernel, view, volumeTable);
        const encoder = device.createCommandEncoder({ label: 'ddgi-volumes shading cost' });
        for (let run = 0; run < runs; run++) {
            const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * run, endOfPassWriteIndex: 2 * run + 1 } });
            this.dispatch(pass, this.rendererKernel, groups, view);
            pass.end();
        }
        encoder.resolveQuerySet(querySet, 0, 2 * runs, resolved, 0);
        device.queue.submit([encoder.finish()]);
        const stamps = new BigUint64Array(await readBuffer(device, resolved, 16 * runs));
        querySet.destroy();
        resolved.destroy();
        const times: number[] = [];
        for (let run = COST_WARM_UP; run < runs; run++) times.push(Number(stamps[2 * run + 1] - stamps[2 * run]) / 1e6);
        return median(times);
    }
}

/** The view's surface points and which volumes answer each: the dense volume's edge coverage per pixel (NaN on sky). */
async function denseEdgeCoverage(device: GPUDevice, view: View, field: DDGIField): Promise<Float32Array> {
    const surfaces = new Float32Array(await readBuffer(device, view.surfaces, view.width * view.height * GBUFFER_VEC4S * 16));
    const dense = field.volumes[0];
    const edges = new Float32Array(view.width * view.height);
    for (let pixel = 0; pixel < edges.length; pixel++) {
        const base = pixel * GBUFFER_VEC4S * 4;
        edges[pixel] = surfaces[base + 3] > 0.5 ? volumeEdgeCoverage(dense.grid, [surfaces[base], surfaces[base + 1], surfaces[base + 2]]) : Number.NaN;
    }
    return edges;
}

type Region = 'band' | 'dense' | 'coarse' | 'view';
const REGIONS: readonly Region[] = ['band', 'dense', 'coarse', 'view'];

/** Which region a surface pixel is in by the dense volume's edge coverage; null on sky. */
function regionOf(edge: number): Exclude<Region, 'view'> | null {
    if (Number.isNaN(edge)) return null;
    if (edge >= 1) return 'dense';
    if (edge <= 0) return 'coarse';
    return 'band';
}

/**
 * One frame pair over one region's pixels, as `ddgi-flicker` measures it:
 * per pixel the mean over rgb of |GI(t) − GI(t−1)| over the region's mean GI
 * at t; the mean and the 95th percentile over the pixels.
 */
function frameChange(previous: Float32Array, current: Float32Array, pixels: readonly number[]): { mean: number; p95: number } {
    let sum = 0;
    for (const pixel of pixels) sum += current[pixel * 4] + current[pixel * 4 + 1] + current[pixel * 4 + 2];
    const regionMean = sum / (3 * pixels.length);
    const changes = new Float32Array(pixels.length);
    pixels.forEach((pixel, index) => {
        let change = 0;
        for (let channel = 0; channel < 3; channel++) change += Math.abs(current[pixel * 4 + channel] - previous[pixel * 4 + channel]);
        changes[index] = change / 3 / regionMean;
    });
    let total = 0;
    for (const change of changes) total += change;
    changes.sort();
    return { mean: total / pixels.length, p95: changes[Math.min(changes.length - 1, Math.max(0, Math.ceil(0.95 * changes.length) - 1))] };
}

/**
 * Check 1 at every surface pixel: the blend against the formula from the
 * single-volume answers. Returns the worst relative differences by region.
 */
function checkBlend(both: Float32Array, dense: Float32Array, coarse: Float32Array, edges: Float32Array): Record<string, number> {
    const worst = { insideDense: 0, outsideDense: 0, bandIrradiance: 0, bandCoverage: 0 };
    const floor = 1e-6;
    for (let pixel = 0; pixel < edges.length; pixel++) {
        const region = regionOf(edges[pixel]);
        if (!region) continue;
        const coverageDense = dense[pixel * 4 + 3];
        const coverageCoarse = coarse[pixel * 4 + 3];
        const coarseShare = (1 - coverageDense) * coverageCoarse;
        const gathered = coverageDense + coarseShare;
        for (let channel = 0; channel < 3; channel++) {
            const value = both[pixel * 4 + channel];
            const scale = Math.max(Math.abs(value), floor);
            if (region === 'dense' && coverageDense >= 0.99) {
                // The visit stops after the dense volume once it covers 99%: its answer, as it is.
                worst.insideDense = Math.max(worst.insideDense, Math.abs(value - dense[pixel * 4 + channel]) / scale);
            } else if (region === 'coarse') {
                worst.outsideDense = Math.max(worst.outsideDense, Math.abs(value - coarse[pixel * 4 + channel]) / scale);
            } else if (region === 'band' && gathered > 0 && coverageDense < 0.99) {
                const expected = (coverageDense * dense[pixel * 4 + channel] + coarseShare * coarse[pixel * 4 + channel]) / gathered;
                worst.bandIrradiance = Math.max(worst.bandIrradiance, Math.abs(value - expected) / scale);
            }
        }
        if (region === 'band' && coverageDense < 0.99) worst.bandCoverage = Math.max(worst.bandCoverage, Math.abs(both[pixel * 4 + 3] - gathered));
    }
    return worst;
}

/** What one case measured, and its images for the panels. */
interface CaseResult {
    readonly report: Record<string, unknown>;
    readonly images: PanelImages;
}

/** One case (the file comment): its own platform and technique, checked and measured. */
async function runCase(device: GPUDevice, testCase: Case, log: (line: string) => void): Promise<CaseResult> {
    const say = (line: string): void => log(`${testCase.name}: ${line}`);
    const scene = await createDdgiScene(device, SCENE, { volumes: [COARSE, testCase.dense], warmUpPasses: 16 }, { gpuTiming: false });
    const { platform, ddgi } = scene;
    try {
        const field = ddgi.field;
        assert(field, 'DDGI has no field after prepare');
        assert(field.volumes.length === 2, `expected 2 volumes, got ${field.volumes.length}`);
        const [dense, coarse] = field.volumes;
        assert(dense.priority === 1 && dense.grid.spacing[0] === 0.25, 'the dense volume should come first (priority 1)');
        say(
            `dense ${dense.grid.counts.join(' x ')} (layers ${dense.layerBase}-${dense.layerBase + dense.grid.counts[1] - 1}), ` +
                `coarse ${coarse.grid.counts.join(' x ')} (layers ${coarse.layerBase}-${coarse.layerBase + coarse.grid.counts[1] - 1}); atlas ${field.layout.irradiance.width} x ${field.layout.irradiance.height} x ${field.layout.irradiance.layers}`,
        );

        // The test's own kernels first, so the recompile counter afterwards counts only what the frames compile.
        const shader = await Shader.create(device, platform);
        const recompilesBefore = platform.stats.recompiles;

        // The technique's warm-up, then settling at the batch with a 60 Hz clock.
        let warmUpFrames = 0;
        while (ddgi.warmingUp) {
            runFrame(device, platform, UNITS);
            if (++warmUpFrames % 8 === 0) await device.queue.onSubmittedWorkDone();
        }
        for (let index = 0; index < SETTLE_FRAMES; index++) {
            runFrame(device, platform, UNITS);
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();
        const counters = ddgi.counters;
        say(`warm-up ${warmUpFrames} frames (${counters['warm-up passes'].toFixed(2)} passes), then ${SETTLE_FRAMES} frames at ${counters['probes updated']} probes a frame`);

        const view = await shader.view(testCase.camera, WIDTH, HEIGHT);
        const edges = await denseEdgeCoverage(device, view, field);
        const pixelsBy: Record<Region, number[]> = { band: [], dense: [], coarse: [], view: [] };
        edges.forEach((edge, pixel) => {
            const region = regionOf(edge);
            if (!region) return;
            pixelsBy[region].push(pixel);
            pixelsBy.view.push(pixel);
        });
        const bandShare = pixelsBy.band.length / pixelsBy.view.length;
        say(`surface pixels ${pixelsBy.view.length}: in the dense volume's fade band ${(100 * bandShare).toFixed(1)}%, dense only ${pixelsBy.dense.length}, coarse only ${pixelsBy.coarse.length}`);
        assert(pixelsBy.band.length >= MIN_BAND_PIXELS, `only ${pixelsBy.band.length} pixels of the view are in the fade band`);

        // 1. The priority blend against each volume alone.
        const denseTable = singleVolumeTable(device, field, 0);
        const coarseTable = singleVolumeTable(device, field, 1);
        const both = await shader.image(view, field.volumeTable);
        const denseOnly = await shader.image(view, denseTable);
        const coarseOnly = await shader.image(view, coarseTable);
        const blend = checkBlend(both, denseOnly, coarseOnly, edges);
        say(
            `blend against the single volumes, worst relative difference: inside the dense volume ${blend.insideDense.toExponential(2)}, outside it ${blend.outsideDense.toExponential(2)}, ` +
                `band irradiance ${blend.bandIrradiance.toExponential(2)}, band coverage ${blend.bandCoverage.toExponential(2)}`,
        );
        for (const [what, value] of Object.entries(blend)) assert(value <= BLEND_TOLERANCE, `${testCase.name}: the blend's ${what} differs from the formula by ${value}`);
        // Information: how far apart the two volumes' answers are where the band hands over (what the blend smooths).
        let bandDifference = 0;
        let bandValue = 0;
        for (const pixel of pixelsBy.band) {
            for (let channel = 0; channel < 3; channel++) {
                bandDifference += Math.abs(denseOnly[pixel * 4 + channel] - coarseOnly[pixel * 4 + channel]);
                bandValue += coarseOnly[pixel * 4 + channel];
            }
        }
        const denseAgainstCoarseInBand = bandDifference / bandValue;
        say(`in the band the dense and coarse volumes' answers differ by ${(100 * denseAgainstCoarseInBand).toFixed(2)}% (mean |dense - coarse| / mean coarse)`);

        // 2. Frame to frame, every frame one update and one shading.
        const sums: Record<Region, { mean: number; p95: number }> = { band: { mean: 0, p95: 0 }, dense: { mean: 0, p95: 0 }, coarse: { mean: 0, p95: 0 }, view: { mean: 0, p95: 0 } };
        const changeMap = new Float32Array(WIDTH * HEIGHT);
        let previous = both;
        let nonFinite = 0;
        for (let frame = 0; frame < MEASURE_FRAMES; frame++) {
            runFrame(device, platform, UNITS);
            const current = await shader.image(view, field.volumeTable);
            for (const value of current) if (!Number.isFinite(value)) nonFinite++;
            for (const region of REGIONS) {
                if (pixelsBy[region].length === 0) continue;
                const change = frameChange(previous, current, pixelsBy[region]);
                sums[region].mean += change.mean / MEASURE_FRAMES;
                sums[region].p95 += change.p95 / MEASURE_FRAMES;
            }
            for (const pixel of pixelsBy.view) {
                let change = 0;
                for (let channel = 0; channel < 3; channel++) change += Math.abs(current[pixel * 4 + channel] - previous[pixel * 4 + channel]);
                changeMap[pixel] += change / 3 / MEASURE_FRAMES;
            }
            previous = current;
        }
        assert(nonFinite === 0, `${testCase.name}: ${nonFinite} non-finite values in the measured frames`);
        for (const region of REGIONS) {
            say(`frame-to-frame change, ${region}: mean ${(100 * sums[region].mean).toFixed(4)}%, p95 ${(100 * sums[region].p95).toFixed(4)}% per pixel a frame (${pixelsBy[region].length} pixels)`);
        }
        const probesPerFrame = ddgi.counters['probes updated'];

        // 3. Nothing compiled over the frames.
        const recompiles = platform.stats.recompiles - recompilesBefore;
        assert(recompiles === 0, `${testCase.name}: ${recompiles} kernels compiled during the frames`);

        // 4. The cost at 1440p, on the same view.
        let cost: Record<string, number | null> | null = null;
        if (testCase.cost) {
            const costView = await shader.view(testCase.camera, COST_WIDTH, COST_HEIGHT);
            const costEdges = await denseEdgeCoverage(device, costView, field);
            let costSurface = 0;
            let costBand = 0;
            for (const edge of costEdges) {
                const region = regionOf(edge);
                if (!region) continue;
                costSurface++;
                if (region === 'band') costBand++;
            }
            cost = {
                bothVolumesMs: await shader.rendererMs(costView, field.volumeTable),
                denseOnlyMs: await shader.rendererMs(costView, denseTable),
                coarseOnlyMs: await shader.rendererMs(costView, coarseTable),
                bandShareOfSurface: costBand / costSurface,
                surfaceShareOfView: costSurface / costEdges.length,
            };
            say(
                cost.bothVolumesMs === null
                    ? 'cost: no timestamp-query on this device'
                    : `cost at ${COST_WIDTH} x ${COST_HEIGHT} (giIrradiance + giCoverage per pixel, ${(100 * costBand / costSurface).toFixed(1)}% of surface pixels in the band): ` +
                          `both volumes ${cost.bothVolumesMs.toFixed(3)} ms, dense alone ${cost.denseOnlyMs?.toFixed(3)} ms, coarse alone ${cost.coarseOnlyMs?.toFixed(3)} ms`,
            );
            for (const buffer of [costView.camera, costView.surfaces, costView.shaded]) buffer.destroy();
        }
        for (const buffer of [view.camera, view.surfaces, view.shaded]) buffer.destroy();
        denseTable.destroy();
        coarseTable.destroy();

        assert(sums.band.mean <= STABILITY_MEAN, `${testCase.name}: the fade band changes ${(100 * sums.band.mean).toFixed(4)}% a frame (mean), over ${STABILITY_MEAN * 100}%`);
        assert(sums.band.p95 <= STABILITY_P95, `${testCase.name}: the fade band changes ${(100 * sums.band.p95).toFixed(4)}% a frame (p95), over ${STABILITY_P95 * 100}%`);
        return {
            report: {
                camera: testCase.camera,
                volumes: field.volumes.map((volume) => ({ counts: volume.grid.counts, spacing: volume.grid.spacing, priority: volume.priority, layerBase: volume.layerBase, probeBase: volume.probeBase })),
                probesPerFrame,
                bandShare,
                pixels: { band: pixelsBy.band.length, dense: pixelsBy.dense.length, coarse: pixelsBy.coarse.length, view: pixelsBy.view.length },
                stability: sums,
                blend,
                denseAgainstCoarseInBand,
                recompiles,
                cost,
            },
            images: { title: testCase.name, both, coarseOnly, denseOnly, edges, changeMap },
        };
    } finally {
        platform.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const cases: Record<string, unknown> = {};
        const images: PanelImages[] = [];
        for (const testCase of CASES) {
            const result = await runCase(device, testCase, log);
            cases[testCase.name] = result.report;
            images.push(result.images);
        }
        drawPanels(images);
        return {
            scene: SCENE,
            units: UNITS,
            settleFrames: SETTLE_FRAMES,
            measuredFrames: MEASURE_FRAMES,
            cases,
            costNote: 'recorded, not gated (prototype level, DECISIONS 2026-11-09)',
        };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);

/** Display value for a linear GI value: the image mean maps to 0.25, then sRGB. */
function display(value: number, scale: number): number {
    const linear = Math.min(1, Math.max(0, value * scale));
    const srgb = linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
    return Math.round(255 * srgb);
}

/** One case's images for its row of panels. */
interface PanelImages {
    readonly title: string;
    readonly both: Float32Array;
    readonly coarseOnly: Float32Array;
    readonly denseOnly: Float32Array;
    readonly edges: Float32Array;
    readonly changeMap: Float32Array;
}

/**
 * A row of four panels per case, with captions: both volumes; the coarse
 * volume alone; which volume answers; the mean change a frame, brightened
 * 50× (white at 2% of the image mean a frame). Shown scaled to fit the page,
 * so the runner's screenshot holds them all.
 */
function drawPanels(rows: readonly PanelImages[]): void {
    const caption = 22;
    const canvas = document.createElement('canvas');
    canvas.width = 4 * WIDTH;
    canvas.height = rows.length * (HEIGHT + caption);
    canvas.style.width = '1240px';
    const context = canvas.getContext('2d');
    if (!context) return;
    context.fillStyle = '#111';
    context.fillRect(0, 0, canvas.width, canvas.height);
    rows.forEach((images, row) => {
        let sum = 0;
        let count = 0;
        for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
            if (Number.isNaN(images.edges[pixel])) continue;
            sum += images.both[pixel * 4] + images.both[pixel * 4 + 1] + images.both[pixel * 4 + 2];
            count++;
        }
        const imageMean = sum / (3 * count);
        const scale = 0.25 / imageMean;
        const changeGain = 50 / imageMean;
        const shown = (image: Float32Array) => (p: number) => [0, 1, 2].map((c) => display(image[p * 4 + c], scale)) as [number, number, number];
        const panels: { title: string; pixel: (pixel: number) => [number, number, number] }[] = [
            { title: `${images.title}: GI, both volumes`, pixel: shown(images.both) },
            { title: 'GI, the coarse volume alone', pixel: shown(images.coarseOnly) },
            {
                title: 'Which answers: green dense, blue coarse, mixed: fade band',
                pixel: (p) => {
                    if (Number.isNaN(images.edges[p])) return [0, 0, 0];
                    const denseShare = images.denseOnly[p * 4 + 3];
                    const coarseShare = (1 - denseShare) * images.coarseOnly[p * 4 + 3];
                    return [Math.round(40 * (1 - denseShare - coarseShare)), Math.round(200 * denseShare), Math.round(230 * coarseShare)];
                },
            },
            {
                title: 'Change a frame, 50x (a flickering seam shows bright)',
                pixel: (p) => {
                    const grey = Math.round(255 * Math.min(1, images.changeMap[p] * changeGain));
                    return [grey, grey, grey];
                },
            },
        ];
        panels.forEach((panel, column) => {
            const image = context.createImageData(WIDTH, HEIGHT);
            for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
                const [r, g, b] = panel.pixel(pixel);
                image.data.set([r, g, b, 255], pixel * 4);
            }
            const x = column * WIDTH;
            const y = row * (HEIGHT + caption);
            context.putImageData(image, x, y + caption);
            context.fillStyle = '#eee';
            context.font = '18px sans-serif';
            context.fillText(panel.title, x + 6, y + 17);
        });
    });
    document.body.prepend(canvas);
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function subtract(a: Vec3, b: Vec3): [number, number, number] {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function cross(a: Vec3, b: Vec3): [number, number, number] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function normalize(a: Vec3): [number, number, number] {
    const length = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / length, a[1] / length, a[2] / length];
}
