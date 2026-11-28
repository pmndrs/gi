/**
 * T3.6 (docs/design/m3-moving.md §4.2, "Draft done-whens"): the
 * camera-following DDGI volume on Bistro exterior's fly-through, on a real
 * GPU.
 *
 * Two volumes in one technique: the camera-following one (`cameraVolume`,
 * 32 x 12 x 32 probes at 1.5 m, priority 1, Dennis's pick of 2026-11-05) and
 * the manifest's fixed 4 m volume behind it, which covers the camera volume's
 * fade band as §4.2 plans ("a plane that enters next to a fixed volume is
 * covered by that volume meanwhile"). The scene is loaded the viewer's way
 * (ddgi-shared/viewer-scene.ts) and every frame goes through the three
 * adapter, so the camera reaches the technique as a host gives it.
 *
 * The flight (`planFlight`): from the `street` bookmark along the street at
 * SPEED_M_S, then right-angle turns to the freer side (up to MAX_LEGS legs or FLIGHT_MAX_M), each leg as long as the street is free (measured by
 * a ray), so the window moves on both x and z. Before it: the technique's
 * warm-up and SETTLE_FRAMES frames at rest; after it, the rest test.
 *
 * Checks (done-whens 1-6):
 * 1. **The edge band holds still.** Fixed observers (OBSERVERS, cameras along
 *    the flight looking along it) see the camera volume's leading fade band
 *    sweep over the street. Every frame each observer's view is shaded through
 *    the volume table (`ddgiSampleVolumes`, what renderers read), and per frame
 *    pair and pixel the change is `mean over rgb |GI(t) − GI(t−1)| / mean GI`,
 *    as `ddgi-flicker` and `ddgi-volumes` measure it, over the pixels in the
 *    camera volume's fade band at either frame (its edge coverage between 0
 *    and 1), averaged over the pairs: mean ≤ 0.1% and p95 ≤ 0.5% (the
 *    stability threshold, DECISIONS 2026-11-06). The same at rest before the
 *    flight is reported beside it.
 * 2. **Entering probes are never used before they're warm.** Every frame of
 *    the flight the camera volume's probe data and history are read back:
 *    the slots of planes that entered that frame must be in state 3 with at
 *    most one blended update (`ddgi/scroll` reset them, and the frame may have
 *    updated them once); and no slot may be active with fewer than 2 blended
 *    updates since its reset (k, the history's update count). Sampling skips
 *    every probe that isn't active (`ddgiCornerWeight`).
 * 3. **Memory stays fixed:** the technique's and the platform's bytes are the
 *    same before the flight, after it and after the rest.
 * 4. **Resting on a plane boundary clears 0 planes:** the camera is put
 *    WINDOW_MOVE_SPACINGS from the window's centre (where the window moves)
 *    with a millimetre of jitter, and after arriving holds there REST_FRAMES
 *    frames (10 s at 60 Hz): the "planes cleared (total)" counter must not move.
 * 5. **A frame strip** of the flight from the moving camera, for the eye:
 *    the GI the field gives, and which volume answers (green the camera
 *    volume, blue the fixed one, dark neither).
 * 6. **No recompiles** over the flight and the rest.
 *
 * The scroll pass's GPU time is recorded, not gated (prototype level,
 * DECISIONS 2026-11-09).
 *
 *   VITE_DDGI_UNITS=264 (probes a frame over both volumes), VITE_DDGI_SETTLE=300,
 *   VITE_DDGI_MAX_TEXTURE_SIZE=1024 (scale model textures down at load; three never draws here)
 */

import type { PerspectiveCamera } from 'three/webgpu';
import { assert, runGpuTest } from '../../client.ts';
import type { ShadingModule, ShadingParameter } from '../../../../packages/core/src/platform/contract/shading-module.ts';
import type { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import type { DDGIFieldVolume } from '../../../../packages/ddgi/src/ddgi-technique.ts';
import { PROBE_HISTORY_FLAG, PROBE_HISTORY_UPDATES, PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS, PROBE_STATE } from '../../../../packages/ddgi/src/layout.ts';
import type { DDGIVolumeSettings } from '../../../../packages/ddgi/src/settings.ts';
import { WINDOW_MOVE_SPACINGS, windowProbeOf } from '../../../../packages/ddgi/src/scroll.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { halfToFloat, loadManifest, readBuffer, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { startViewerScene, warmUp, yieldToCallbacks, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';

const env = import.meta.env as Record<string, unknown>;
function countFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' ? Number(value) : NaN;
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

type Vec3 = readonly [number, number, number];

const SCENE = 'bistro-exterior';
/** The fly-through's speed and spacing (m3-moving.md, "Still open" 1: 5 m/s at 1.5 m). */
const SPEED_M_S = 5;
const SPACING = 1.5;
const FRAMES_PER_SECOND = 60;
/** The legs' longest length, and how far short of the first wall a leg stops. */
const LEG_MAX_M = 40;
const WALL_CLEARANCE_M = 4;
/** The flight turns into a new leg until it has this many legs or is this long; a leg shorter than MIN_LEG_M ends it. */
const MAX_LEGS = 4;
const FLIGHT_MAX_M = 60;
const MIN_LEG_M = 5;
/** Camera height above the bookmark's ground, metres (the bookmark is at 2 m). */
const EYE_HEIGHT_M = 2;
/**
 * Probes a frame over both volumes (each gets its share by probe count and a
 * floor of its own, volumes.ts): twice Budget v2's 132 on Bistro (T1.9b,
 * m3-moving.md §4.2), so the camera volume gets about the 43 a frame the
 * design's arithmetic asks for 5 m/s at 1.5 m, and more.
 */
const UNITS = countFromEnv(env.VITE_DDGI_UNITS, 264);
const SETTLE_FRAMES = countFromEnv(env.VITE_DDGI_SETTLE, 300);
/** At rest before the flight, the same measurement as check 1, for comparison. */
const STILL_FRAMES = 60;
/** 10 s at 60 Hz, resting on the plane boundary (check 4); arrival frames before them. */
const REST_FRAMES = 600;
const ARRIVAL_FRAMES = 30;
const REST_JITTER_M = 0.001;
const MAX_TEXTURE_SIZE = typeof env.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' && env.VITE_DDGI_MAX_TEXTURE_SIZE !== '' ? Number(env.VITE_DDGI_MAX_TEXTURE_SIZE) : undefined;

/** The stability threshold (DECISIONS 2026-11-06): per-pixel change a frame. */
const STABILITY_MEAN = 0.001;
const STABILITY_P95 = 0.005;
/** A frame pair counts toward check 1 when this many observer pixels are in the band (else its mean says little). */
const MIN_BAND_PIXELS = 200;

const OBSERVER_WIDTH = 320;
const OBSERVER_HEIGHT = 180;
const STRIP_WIDTH = 320;
const STRIP_HEIGHT = 180;
const STRIP_FRAMES = 8;
const FOV_Y_DEG = 65;

const COMPUTE = GPUShaderStage.COMPUTE;
const GBUFFER_VEC4S = 2;
const SHADED_VEC4S = 2;

type ComputeKernel = Awaited<ReturnType<GIPlatformImpl['kernels']['computeKernel']>>;

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

/** Per pixel the hit point (w: 1 on a hit) and the shading normal (w: the albedo, pack4x8unorm). */
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
 * Shading from the G-buffer through the module's globals: per pixel
 * albedo / π × E and the blend's coverage (one `ddgiSampleVolumes` call, as a
 * renderer reads the field), then the camera volume's own edge coverage (its
 * fade band) and its share of the blend (which volume answers).
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
// The camera-following volume has the higher priority: the table's first row.
const TEST_CAMERA_VOLUME_ROW: u32 = 0u;

@compute @workgroup_size(8, 8)
fn shadeImage(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= testCamera.size.xy)) {
        return;
    }
    let pixel = id.y * testCamera.size.x + id.x;
    let positionAndHit = testSurfaces[pixel * ${GBUFFER_VEC4S}u];
    let normalAndAlbedo = testSurfaces[pixel * ${GBUFFER_VEC4S}u + 1u];
    var image = vec4f(0.0);
    var cameraVolume = vec4f(-1.0, 0.0, 0.0, 0.0);
    if (positionAndHit.w > 0.5) {
        let position = positionAndHit.xyz;
        let normal = normalAndAlbedo.xyz;
        let albedo = unpack4x8unorm(bitcast<u32>(normalAndAlbedo.w)).rgb;
        let viewDirection = normalize(position - testCamera.position.xyz);
        let field = ddgiSampleVolumes(position, normal, viewDirection, ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData, ddgiVolumeTable);
        image = vec4f(albedo * TEST_INV_PI * field.irradiance, field.coverage);
        let volume = ddgiTableVolume(ddgiVolumeTable, TEST_CAMERA_VOLUME_ROW);
        let edge = ddgiVolumeEdgeCoverage(volume, position);
        // The camera volume's share of the blend: its edge coverage times how fully its probes see the point.
        let own = ddgiAccumulateVolume(volume, position, normal, viewDirection, ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData);
        cameraVolume = vec4f(edge, ddgiBlendShare(edge, own.weight, 0.0), 0.0, 0.0);
    }
    testShaded[pixel * ${SHADED_VEC4S}u] = image;
    testShaded[pixel * ${SHADED_VEC4S}u + 1u] = cameraVolume;
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
    return { label: 'ddgi-camera-volume module', entries };
}

function moduleGroup(device: GPUDevice, layout: GPUBindGroupLayout, shading: ShadingModule): GPUBindGroup {
    const entries: GPUBindGroupEntry[] = [];
    for (const parameter of shading.parameters as readonly ShadingParameter[]) {
        if (parameter.kind === 'texture') {
            entries.push({ binding: entries.length, resource: parameter.texture().createView({ dimension: parameter.dimension }) });
        } else if (parameter.kind === 'sampler') {
            entries.push({
                binding: entries.length,
                resource: device.createSampler({ magFilter: parameter.filtering, minFilter: parameter.filtering, addressModeU: parameter.addressMode, addressModeV: parameter.addressMode }),
            });
        }
    }
    return device.createBindGroup({ layout, entries });
}

/** A view's surfaces and the buffer its shading writes into. */
interface View {
    readonly camera: CameraBookmark;
    readonly width: number;
    readonly height: number;
    readonly cameraBuffer: GPUBuffer;
    readonly surfaces: GPUBuffer;
    readonly shaded: GPUBuffer;
}

/** One shading of a view: GI (rgba: albedo / π × E, blend coverage) and the camera volume's edge coverage and share per pixel (−1 on sky). */
interface Shaded {
    readonly image: Float32Array;
    readonly edge: Float32Array;
    readonly share: Float32Array;
}

class Shader {
    private constructor(
        private readonly device: GPUDevice,
        private readonly shading: ShadingModule,
        private readonly gbufferKernel: ComputeKernel,
        private readonly imageKernel: ComputeKernel,
    ) {}

    static async create(device: GPUDevice, platform: GIPlatformImpl): Promise<Shader> {
        const shading = platform.shading;
        assert(shading, 'the platform has no shading module');
        const gbufferKernel = await platform.kernels.computeKernel({
            label: 'ddgi-camera-volume surfaces',
            modules: [platform.shade.module],
            code: GBUFFER_WGSL,
            entryPoint: 'main',
            ownGroups: [
                {
                    label: 'ddgi-camera-volume surfaces io',
                    entries: [
                        { binding: 0, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
                        { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
                    ],
                },
            ],
        });
        const imageKernel = await platform.kernels.computeKernel({
            label: 'ddgi-camera-volume shade',
            modules: [],
            code: shadingWgsl(shading),
            entryPoint: 'shadeImage',
            ownGroups: [
                moduleGroupLayout(shading),
                {
                    label: 'ddgi-camera-volume shading io',
                    entries: [
                        { binding: 0, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
                        { binding: 1, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
                        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
                    ],
                },
            ],
        });
        return new Shader(device, shading, gbufferKernel, imageKernel);
    }

    /** The camera's surfaces at this size (traced once; nothing in the scene moves). */
    async view(camera: CameraBookmark, width: number, height: number): Promise<View> {
        const device = this.device;
        const cameraBuffer = device.createBuffer({ size: 80, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(cameraBuffer, 0, cameraData(camera, width, height));
        const surfaces = device.createBuffer({ size: width * height * GBUFFER_VEC4S * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const shaded = device.createBuffer({ size: width * height * SHADED_VEC4S * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const encoder = device.createCommandEncoder({ label: 'ddgi-camera-volume surfaces' });
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
        return { camera, width, height, cameraBuffer, surfaces, shaded };
    }

    /** The first surface point along a ray from `position` (NaN when it hits nothing): a 1 × 1 view. */
    async firstHit(position: Vec3, direction: Vec3): Promise<number> {
        const view = await this.view({ name: 'probe ray', position: [...position], target: add(position, direction), fovYDeg: 1 }, 1, 1);
        const surfaces = new Float32Array(await readBuffer(this.device, view.surfaces, GBUFFER_VEC4S * 16));
        destroyView(view);
        if (surfaces[3] < 0.5) return Number.NaN;
        return Math.hypot(surfaces[0] - position[0], surfaces[1] - position[1], surfaces[2] - position[2]);
    }

    /** Shades a view through the volume table now. */
    async shade(view: View): Promise<Shaded> {
        const device = this.device;
        const encoder = device.createCommandEncoder({ label: 'ddgi-camera-volume shade' });
        const pass = encoder.beginComputePass();
        this.imageKernel.bind(pass);
        pass.setBindGroup(1, moduleGroup(device, this.imageKernel.ownLayouts[0], this.shading));
        pass.setBindGroup(
            2,
            device.createBindGroup({
                layout: this.imageKernel.ownLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: view.cameraBuffer } },
                    { binding: 1, resource: { buffer: view.surfaces } },
                    { binding: 2, resource: { buffer: view.shaded } },
                ],
            }),
        );
        pass.dispatchWorkgroups(Math.ceil(view.width / 8), Math.ceil(view.height / 8));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const raw = new Float32Array(await readBuffer(device, view.shaded, view.width * view.height * SHADED_VEC4S * 16));
        const pixels = view.width * view.height;
        const image = new Float32Array(pixels * 4);
        const edge = new Float32Array(pixels);
        const share = new Float32Array(pixels);
        for (let pixel = 0; pixel < pixels; pixel++) {
            image.set(raw.subarray(pixel * 8, pixel * 8 + 4), pixel * 4);
            edge[pixel] = raw[pixel * 8 + 4];
            share[pixel] = raw[pixel * 8 + 5];
        }
        return { image, edge, share };
    }
}

function destroyView(view: View): void {
    for (const buffer of [view.cameraBuffer, view.surfaces, view.shaded]) buffer.destroy();
}

/** A straight leg of the flight at SPEED_M_S. */
interface Leg {
    readonly from: Vec3;
    readonly direction: Vec3;
    readonly length: number;
}

/** Where the flight is after `metres`, and the way it faces. */
function along(legs: readonly Leg[], metres: number): { position: Vec3; direction: Vec3 } {
    let left = metres;
    for (const [index, leg] of legs.entries()) {
        if (left <= leg.length || index === legs.length - 1) {
            const distance = Math.min(left, leg.length);
            return { position: add(leg.from, scale(leg.direction, distance)), direction: leg.direction };
        }
        left -= leg.length;
    }
    throw new Error('no legs');
}

/** A horizontal direction at `degrees` from +x toward +z. */
function heading(degrees: number): Vec3 {
    const radians = (degrees * Math.PI) / 180;
    return [Math.cos(radians), 0, Math.sin(radians)];
}

/**
 * The flight: from the street bookmark toward its target along the street,
 * as far as the street is free (up to LEG_MAX_M, stopping WALL_CLEARANCE_M
 * short of what the ray hits), then turning to whichever side is freer.
 */
async function planFlight(shader: Shader, bookmark: CameraBookmark, log: (line: string) => void): Promise<Leg[]> {
    const start: Vec3 = [bookmark.position[0], bookmark.position[1] - 2 + EYE_HEIGHT_M, bookmark.position[2]];
    const toTarget = normalize([bookmark.target[0] - start[0], 0, bookmark.target[2] - start[2]]);
    const freeLength = async (from: Vec3, direction: Vec3): Promise<number> => {
        const hit = await shader.firstHit(from, direction);
        return Math.min(LEG_MAX_M, (Number.isNaN(hit) ? Number.POSITIVE_INFINITY : hit) - WALL_CLEARANCE_M);
    };
    const first: Leg = { from: start, direction: toTarget, length: await freeLength(start, toTarget) };
    const legs: Leg[] = [first];
    // Each next leg turns a right angle to whichever side is freer, until MAX_LEGS or FLIGHT_MAX_M.
    while (legs.length < MAX_LEGS && legs.reduce((sum, leg) => sum + leg.length, 0) < FLIGHT_MAX_M) {
        const last = legs[legs.length - 1];
        const turnFrom = add(last.from, scale(last.direction, last.length));
        const lastDegrees = (Math.atan2(last.direction[2], last.direction[0]) * 180) / Math.PI;
        const sides = await Promise.all([lastDegrees + 90, lastDegrees - 90].map(async (degrees) => ({ direction: heading(degrees), length: await freeLength(turnFrom, heading(degrees)) })));
        const turn = sides[0].length >= sides[1].length ? sides[0] : sides[1];
        if (turn.length < MIN_LEG_M) break;
        legs.push({ from: turnFrom, direction: turn.direction, length: turn.length });
    }
    for (const leg of legs) log(`leg from ${leg.from.map((value) => value.toFixed(1)).join(', ')} heading ${leg.direction.map((value) => value.toFixed(2)).join(', ')}, ${leg.length.toFixed(1)} m`);
    assert(legs.length > 0 && legs[0].length >= 10, `the street isn't free ahead of the bookmark (${first.length.toFixed(1)} m)`);
    return legs;
}

/** Puts the adapter's camera at `position` looking along `direction` (a little down the street). */
function placeCamera(camera: PerspectiveCamera, position: Vec3, direction: Vec3): void {
    camera.position.set(position[0], position[1], position[2]);
    camera.lookAt(position[0] + direction[0], position[1] + direction[1] - 0.05, position[2] + direction[2]);
    camera.updateMatrixWorld();
}

/** Per pixel the mean over rgb of |GI(t) − GI(t−1)| over `pixels`, divided by their mean GI at t: the mean and the 95th percentile. */
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

const inBand = (edge: number): boolean => edge > 0 && edge < 1;

/** The mean GI over an image's surface pixels (edge coverage −1 marks sky), cached per image. */
const imageMeans = new WeakMap<Float32Array, number>();
function imageMean(image: Float32Array, edge: Float32Array): number {
    const cached = imageMeans.get(image);
    if (cached !== undefined) return cached;
    let sum = 0;
    let count = 0;
    for (let pixel = 0; pixel < edge.length; pixel++) {
        if (edge[pixel] < 0) continue;
        sum += image[pixel * 4] + image[pixel * 4 + 1] + image[pixel * 4 + 2];
        count++;
    }
    const mean = Math.max(1e-9, sum / (3 * Math.max(1, count)));
    imageMeans.set(image, mean);
    return mean;
}
/** A band pixel whose edge coverage changes by more than this between two frames is in the band that moved. */
const MOVING_EDGE_STEP = 1e-4;

/** Frame-to-frame change over a run of frames, in the band and over the whole view, averaged over the pairs. */
class ChangeTally {
    band = { mean: 0, p95: 0, pairs: 0, pixels: 0 };
    view = { mean: 0, p95: 0, pairs: 0 };
    worstBand = { mean: 0, p95: 0 };
    /** Per observer pixel, its largest change a frame while in the band, over the image's mean GI (the change map). */
    readonly changeMaps: Float32Array[];

    constructor(observers: readonly View[]) {
        this.changeMaps = observers.map((view) => new Float32Array(view.width * view.height));
    }

    add(observer: number, previous: Shaded, current: Shaded): void {
        const band: number[] = [];
        const surface: number[] = [];
        for (let pixel = 0; pixel < current.edge.length; pixel++) {
            if (current.edge[pixel] < 0) continue;
            surface.push(pixel);
            if (inBand(current.edge[pixel]) || inBand(previous.edge[pixel])) band.push(pixel);
        }
        if (surface.length > 0) {
            const change = frameChange(previous.image, current.image, surface);
            this.view.mean += change.mean;
            this.view.p95 += change.p95;
            this.view.pairs++;
        }
        if (band.length >= MIN_BAND_PIXELS) {
            const change = frameChange(previous.image, current.image, band);
            this.band.mean += change.mean;
            this.band.p95 += change.p95;
            this.band.pairs++;
            this.band.pixels += band.length;
            this.worstBand.mean = Math.max(this.worstBand.mean, change.mean);
            this.worstBand.p95 = Math.max(this.worstBand.p95, change.p95);
            const map = this.changeMaps[observer];
            for (const pixel of band) {
                let difference = 0;
                for (let channel = 0; channel < 3; channel++) difference += Math.abs(current.image[pixel * 4 + channel] - previous.image[pixel * 4 + channel]);
                map[pixel] = Math.max(map[pixel], difference / 3 / imageMean(current.image, current.edge));
                // A pop would be the camera volume's share jumping by about 1 in one frame; a fade moves it a little.
                this.largestShareStep = Math.max(this.largestShareStep, Math.abs(current.share[pixel] - previous.share[pixel]));
            }
        }
        // The band that moved this frame (its edge coverage changed), apart from the band that stays put (the window's
        // faces it isn't moving across, such as its top while the camera flies level): reported, not gated.
        const moving = band.filter((pixel) => Math.abs(current.edge[pixel] - previous.edge[pixel]) > MOVING_EDGE_STEP);
        if (moving.length >= MIN_BAND_PIXELS) {
            const change = frameChange(previous.image, current.image, moving);
            this.movingBand.mean += change.mean;
            this.movingBand.p95 += change.p95;
            this.movingBand.pairs++;
            this.movingBand.pixels += moving.length;
            let shareStep = 0;
            for (const pixel of moving) shareStep += Math.abs(current.share[pixel] - previous.share[pixel]);
            this.movingBand.shareStep += shareStep / moving.length;
        }
    }

    /** The band that moved: frame pairs, summed means, pixels, and the mean step in the camera volume's share. */
    movingBand = { mean: 0, p95: 0, pairs: 0, pixels: 0, shareStep: 0 };
    /** The largest step in the camera volume's share at a band pixel between two frames. */
    largestShareStep = 0;

    summary(): Record<string, number> {
        const pairs = Math.max(1, this.band.pairs);
        const viewPairs = Math.max(1, this.view.pairs);
        const movingPairs = Math.max(1, this.movingBand.pairs);
        return {
            bandMean: this.band.mean / pairs,
            bandP95: this.band.p95 / pairs,
            bandPairs: this.band.pairs,
            bandPixelsPerPair: this.band.pixels / pairs,
            worstPairBandMean: this.worstBand.mean,
            worstPairBandP95: this.worstBand.p95,
            movingBandMean: this.movingBand.mean / movingPairs,
            movingBandP95: this.movingBand.p95 / movingPairs,
            movingBandPairs: this.movingBand.pairs,
            movingBandPixelsPerPair: this.movingBand.pixels / movingPairs,
            movingBandShareStep: this.movingBand.shareStep / movingPairs,
            largestShareStep: this.largestShareStep,
            viewMean: this.view.mean / viewPairs,
            viewP95: this.view.p95 / viewPairs,
        };
    }
}

/** The camera volume's slots, read back: state and blended-update count (k) per slot, in probe-index order. */
interface SlotRecords {
    readonly state: Uint8Array;
    readonly updates: Uint16Array;
    readonly warmedOnce: Uint8Array;
}

/** Reads the camera volume's probe data (its layers of the shared atlas) and history. */
async function readSlots(device: GPUDevice, scene: ViewerScene, volume: DDGIFieldVolume): Promise<SlotRecords> {
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const [nx, ny, nz] = volume.grid.counts;
    // rgba16float texels; rows padded to 256 bytes for the copy.
    const bytesPerRow = Math.ceil((nx * 8) / 256) * 256;
    const staging = device.createBuffer({ size: bytesPerRow * nz * ny, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder({ label: 'ddgi-camera-volume slots' });
    encoder.copyTextureToBuffer({ texture: field.probeData, origin: { x: 0, y: 0, z: volume.layerBase } }, { buffer: staging, bytesPerRow, rowsPerImage: nz }, { width: nx, height: nz, depthOrArrayLayers: ny });
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    const history = new Uint32Array(await readBuffer(device, volume.probeHistory, volume.grid.probeCount * PROBE_HISTORY_WORDS * 4));
    const count = volume.grid.probeCount;
    const state = new Uint8Array(count);
    const updates = new Uint16Array(count);
    const warmedOnce = new Uint8Array(count);
    for (let index = 0; index < count; index++) {
        const x = index % nx;
        const z = Math.floor(index / nx) % nz;
        const y = Math.floor(index / (nx * nz));
        const texel = (y * nz + z) * (bytesPerRow / 2) + x * 4;
        state[index] = Math.round(halfToFloat(halves[texel + 3]));
        const base = index * PROBE_HISTORY_WORDS;
        updates[index] = history[base + PROBE_HISTORY_WORD.updates] & PROBE_HISTORY_UPDATES.countMask;
        warmedOnce[index] = (history[base + PROBE_HISTORY_WORD.scheduleAndFlags] & PROBE_HISTORY_FLAG.warmedOnce) !== 0 ? 1 : 0;
    }
    return { state, updates, warmedOnce };
}

/** The slots whose window probes entered after the window moved from `before` to `after` (scroll.ts's rule). */
function enteredSlots(counts: Vec3, before: Vec3, after: Vec3): number[] {
    const moved = [0, 1, 2].map((axis) => after[axis] - before[axis]);
    if (moved.every((planes) => planes === 0)) return [];
    const [nx, , nz] = counts;
    const slots: number[] = [];
    const total = counts[0] * counts[1] * counts[2];
    for (let index = 0; index < total; index++) {
        const slot = [index % nx, Math.floor(index / (nx * nz)), Math.floor(index / nx) % nz];
        const entered = slot.some((value, axis) => {
            const windowProbe = windowProbeOf(value, after[axis], counts[axis]);
            return (moved[axis] > 0 && windowProbe >= counts[axis] - moved[axis]) || (moved[axis] < 0 && windowProbe < -moved[axis]);
        });
        if (entered) slots.push(index);
    }
    return slots;
}

/** Check 2's tally over the flight. */
class WarmingTally {
    enteringChecked = 0;
    enteringNotReset = 0;
    /** Entering slots whose first update this frame switched them off (inside geometry, or no surface nearby). */
    enteringSwitchedOff = 0;
    activeTooEarly = 0;
    framesChecked = 0;
    firstProblem = '';

    checkFrame(records: SlotRecords, entered: readonly number[], frame: number): void {
        this.framesChecked++;
        for (const slot of entered) {
            this.enteringChecked++;
            // Reset this frame and updated at most once since: still warming (state 3 ends only after the second
            // update), or switched off by that first update's classification (off probes are never sampled either).
            const state = records.state[slot];
            const warmingStill = state === PROBE_STATE.notUpdatedYet && records.updates[slot] <= 1 && (records.updates[slot] === 0 || records.warmedOnce[slot] === 1);
            const switchedOff = state === PROBE_STATE.offInsideGeometry || state === PROBE_STATE.offNoSurfaceNearby;
            const fresh = warmingStill || switchedOff;
            if (switchedOff) this.enteringSwitchedOff++;
            if (!fresh) {
                this.enteringNotReset++;
                this.firstProblem ||= `frame ${frame}: entering slot ${slot} is in state ${records.state[slot]} with k ${records.updates[slot]}`;
            }
        }
        for (let slot = 0; slot < records.state.length; slot++) {
            if (records.state[slot] === PROBE_STATE.active && records.updates[slot] < 2) {
                this.activeTooEarly++;
                this.firstProblem ||= `frame ${frame}: slot ${slot} is active after ${records.updates[slot]} blended updates`;
            }
        }
    }
}

/** The bytes that must not change while flying: the technique's, and everything the platform holds. */
function memoryNow(scene: ViewerScene): { ddgi: number; platform: number } {
    return { ddgi: scene.ddgi.memoryBytes, platform: scene.platform.stats.memory.total };
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const manifest = loadManifest(SCENE);
        const street = manifest.cameras[0];
        const coarse: DDGIVolumeSettings = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
        // The window starts centred on the flight's start (scroll.ts): 32 x 12 x 32 probes at 1.5 m.
        const start: Vec3 = [street.position[0], street.position[1] - 2 + EYE_HEIGHT_M, street.position[2]];
        const cameraVolume: DDGIVolumeSettings = { center: start, size: [31 * SPACING, 11 * SPACING, 31 * SPACING], spacing: SPACING, follow: 'camera', priority: 1 };

        const scene = await startViewerScene(device, manifest, {
            gpuTiming: true,
            label: 'ddgi-camera-volume',
            settings: { volumes: [coarse, cameraVolume] },
            load: MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE },
        });
        try {
            const { ddgi, platform, camera } = scene;
            camera.fov = FOV_Y_DEG;
            camera.updateProjectionMatrix();
            const field = ddgi.field;
            assert(field, 'DDGI has no field after prepare');
            assert(field.volumes.length === 2 && field.volumes[0].followsCamera, 'the camera-following volume should come first (priority 1)');
            const cameraField = (): DDGIFieldVolume => {
                const volumes = ddgi.field?.volumes;
                assert(volumes, 'no field');
                return volumes[0];
            };
            log(
                `volumes: camera ${field.volumes[0].grid.counts.join(' x ')} at ${SPACING} m (layers ${field.volumes[0].layerBase}+), fixed ${field.volumes[1].grid.counts.join(' x ')} at ${field.volumes[1].grid.spacing[0]} m; ` +
                    `${UNITS} probes a frame over both`,
            );

            const shader = await Shader.create(device, platform);
            const legs = await planFlight(shader, street, log);
            const flightMetres = legs.reduce((sum, leg) => sum + leg.length, 0);
            const flightFrames = Math.round((flightMetres / SPEED_M_S) * FRAMES_PER_SECOND);

            // Warm-up and settling at the start, the camera still.
            placeCamera(camera, start, legs[0].direction);
            scene.frame({ units: UNITS });
            const warm = await warmUp(scene);
            for (let index = 0; index < SETTLE_FRAMES; index++) {
                scene.frame({ units: UNITS });
                if (index % 8 === 7) {
                    await device.queue.onSubmittedWorkDone();
                    await yieldToCallbacks();
                }
            }
            await device.queue.onSubmittedWorkDone();
            log(`warm-up ${warm.passes.toFixed(2)} passes in ${(warm.ms / 1000).toFixed(1)} s, then ${SETTLE_FRAMES} frames at rest; window at ${cameraField().window.join(', ')}`);

            // Fixed observers along the flight, looking along it: they see the leading band sweep over the street.
            const observerPlaces = [0.3, 0.6].map((share) => along(legs, share * legs[0].length));
            if (legs.length > 1) observerPlaces.push({ position: legs[1].from, direction: legs[1].direction });
            const observers = await Promise.all(
                observerPlaces.map(({ position, direction }, index) =>
                    shader.view({ name: `observer ${index + 1}`, position: [...position], target: add(position, direction), fovYDeg: FOV_Y_DEG }, OBSERVER_WIDTH, OBSERVER_HEIGHT),
                ),
            );
            const recompilesBefore = platform.stats.recompiles;

            // At rest: the same measurement, for comparison.
            const still = new ChangeTally(observers);
            let previous = await Promise.all(observers.map((view) => shader.shade(view)));
            for (let frame = 0; frame < STILL_FRAMES; frame++) {
                scene.frame({ units: UNITS });
                const current = await Promise.all(observers.map((view) => shader.shade(view)));
                current.forEach((shaded, observer) => still.add(observer, previous[observer], shaded));
                previous = current;
            }
            const memoryBefore = memoryNow(scene);
            const planesBefore = ddgi.counters['planes cleared (total)'] ?? 0;

            // The flight.
            const flight = new ChangeTally(observers);
            const warming = new WarmingTally();
            const strip: { frame: number; image: Float32Array; share: Float32Array; edge: Float32Array }[] = [];
            const stripEvery = Math.max(1, Math.floor(flightFrames / STRIP_FRAMES));
            const scrollMs: number[] = [];
            let movedFrames = 0;
            for (let frame = 0; frame <= flightFrames; frame++) {
                const metres = (frame / FRAMES_PER_SECOND) * SPEED_M_S;
                const { position, direction } = along(legs, metres);
                placeCamera(camera, position, direction);
                const windowBefore = cameraField().window;
                scene.frame({ units: UNITS });
                const windowAfter = cameraField().window;
                const entered = enteredSlots(cameraField().grid.counts, windowBefore, windowAfter);
                if (entered.length > 0) movedFrames++;
                const records = await readSlots(device, scene, cameraField());
                warming.checkFrame(records, entered, frame);
                const current = await Promise.all(observers.map((view) => shader.shade(view)));
                current.forEach((shaded, observer) => flight.add(observer, previous[observer], shaded));
                previous = current;
                if (frame % stripEvery === 0 && strip.length < STRIP_FRAMES) {
                    const view = await shader.view({ name: `flight ${frame}`, position: [...position], target: add(position, [direction[0], direction[1] - 0.05, direction[2]]), fovYDeg: FOV_Y_DEG }, STRIP_WIDTH, STRIP_HEIGHT);
                    const shaded = await shader.shade(view);
                    strip.push({ frame, image: shaded.image, share: shaded.share, edge: shaded.edge });
                    destroyView(view);
                }
                await yieldToCallbacks();
                const pass = platform.stats.passes.find((entry) => entry.name === 'ddgi/scroll');
                if (pass?.gpuMs !== undefined && Number.isFinite(pass.gpuMs)) scrollMs.push(pass.gpuMs);
            }
            const memoryAfterFlight = memoryNow(scene);
            const planesAfterFlight = ddgi.counters['planes cleared (total)'] ?? 0;
            log(`flight: ${flightMetres.toFixed(1)} m in ${flightFrames} frames at ${SPEED_M_S} m/s; the window moved in ${movedFrames} frames, ${planesAfterFlight - planesBefore} planes cleared; window now at ${cameraField().window.join(', ')}`);

            // Check 4: resting on the plane boundary where the window moves (+x), with a millimetre of jitter.
            const restVolume = cameraField();
            const restGrid = restVolume.grid;
            const centre = [0, 1, 2].map((axis) => restGrid.origin[axis] + (restVolume.window[axis] + (restGrid.counts[axis] - 1) / 2) * restGrid.spacing[axis]);
            const boundary: Vec3 = [centre[0] + WINDOW_MOVE_SPACINGS * restGrid.spacing[0], centre[1], centre[2]];
            const restDirection = legs[legs.length - 1].direction;
            const jittered = (frame: number): Vec3 => [boundary[0] + REST_JITTER_M * Math.sin(frame * 1.7), boundary[1] + REST_JITTER_M * Math.sin(frame * 2.3), boundary[2] - REST_JITTER_M * Math.cos(frame * 1.1)];
            for (let frame = 0; frame < ARRIVAL_FRAMES; frame++) {
                placeCamera(camera, jittered(frame), restDirection);
                scene.frame({ units: UNITS });
            }
            await device.queue.onSubmittedWorkDone();
            const planesOnArrival = ddgi.counters['planes cleared (total)'] ?? 0;
            for (let frame = 0; frame < REST_FRAMES; frame++) {
                placeCamera(camera, jittered(ARRIVAL_FRAMES + frame), restDirection);
                scene.frame({ units: UNITS });
                if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
            }
            await device.queue.onSubmittedWorkDone();
            await yieldToCallbacks();
            const planesWhileResting = (ddgi.counters['planes cleared (total)'] ?? 0) - planesOnArrival;
            const memoryAfterRest = memoryNow(scene);
            const recompiles = platform.stats.recompiles - recompilesBefore;

            const stillSummary = still.summary();
            const flightSummary = flight.summary();
            log(
                `at rest: band ${(100 * stillSummary.bandMean).toFixed(4)}% / p95 ${(100 * stillSummary.bandP95).toFixed(4)}% a frame (${stillSummary.bandPairs} pairs); ` +
                    `view ${(100 * stillSummary.viewMean).toFixed(4)}% / ${(100 * stillSummary.viewP95).toFixed(4)}%`,
            );
            log(
                `flying: band ${(100 * flightSummary.bandMean).toFixed(4)}% / p95 ${(100 * flightSummary.bandP95).toFixed(4)}% a frame over ${flightSummary.bandPairs} pairs ` +
                    `(${flightSummary.bandPixelsPerPair.toFixed(0)} band pixels a pair; worst pair ${(100 * flightSummary.worstPairBandMean).toFixed(4)}% / ${(100 * flightSummary.worstPairBandP95).toFixed(4)}%); ` +
                    `view ${(100 * flightSummary.viewMean).toFixed(4)}% / ${(100 * flightSummary.viewP95).toFixed(4)}%`,
            );
            log(
                `flying, the band that moved: ${(100 * flightSummary.movingBandMean).toFixed(4)}% / p95 ${(100 * flightSummary.movingBandP95).toFixed(4)}% a frame over ${flightSummary.movingBandPairs} pairs ` +
                    `(${flightSummary.movingBandPixelsPerPair.toFixed(0)} pixels a pair); the camera volume's share steps ${flightSummary.movingBandShareStep.toFixed(4)} a frame there on average, ` +
                    `largest step at a band pixel ${flightSummary.largestShareStep.toFixed(4)} (a pop would be about 1)`,
            );
            log(
                `warming: ${warming.enteringChecked} entering slots checked over ${warming.framesChecked} frames, ${warming.enteringNotReset} not reset; ` +
                    `${warming.enteringSwitchedOff} of them switched off by their first update; ${warming.activeTooEarly} slot-frames active before 2 updates${warming.firstProblem ? ` (first: ${warming.firstProblem})` : ''}`,
            );
            log(
                `memory: DDGI ${memoryBefore.ddgi} / ${memoryAfterFlight.ddgi} / ${memoryAfterRest.ddgi} bytes, platform ${memoryBefore.platform} / ${memoryAfterFlight.platform} / ${memoryAfterRest.platform} (before, after the flight, after the rest)`,
            );
            log(`resting on the boundary: ${planesOnArrival - planesAfterFlight} planes on arrival, ${planesWhileResting} over ${REST_FRAMES} frames; recompiles ${recompiles}`);
            const scrollMedian = scrollMs.length > 0 ? median(scrollMs) : null;
            if (scrollMedian !== null) log(`ddgi/scroll: median ${scrollMedian.toFixed(4)} ms a frame (recorded, not gated)`);

            drawPanels(strip, observers, flight);
            for (const view of observers) destroyView(view);

            const report = {
                scene: SCENE,
                units: UNITS,
                speedMetresPerSecond: SPEED_M_S,
                spacing: SPACING,
                legs,
                flightFrames,
                movedFrames,
                planesCleared: planesAfterFlight - planesBefore,
                warmUp: warm,
                still: stillSummary,
                flight: flightSummary,
                warming: { enteringChecked: warming.enteringChecked, enteringNotReset: warming.enteringNotReset, enteringSwitchedOff: warming.enteringSwitchedOff, activeTooEarly: warming.activeTooEarly, framesChecked: warming.framesChecked },
                memory: { before: memoryBefore, afterFlight: memoryAfterFlight, afterRest: memoryAfterRest },
                rest: { boundary, arrivalPlanes: planesOnArrival - planesAfterFlight, planesWhileResting, frames: REST_FRAMES },
                recompiles,
                scrollMsMedian: scrollMedian,
                counters: ddgi.counters,
                costNote: 'recorded, not gated (prototype level, DECISIONS 2026-11-09)',
            };
            // The checks, after everything is logged, so a failed run can still be read.
            const failures: string[] = [];
            if (flightSummary.bandPairs === 0) failures.push('no frame pair had the band in view');
            if (flightSummary.bandMean > STABILITY_MEAN) failures.push(`the edge band changes ${(100 * flightSummary.bandMean).toFixed(4)}% a frame (mean), over ${100 * STABILITY_MEAN}%`);
            if (flightSummary.bandP95 > STABILITY_P95) failures.push(`the edge band changes ${(100 * flightSummary.bandP95).toFixed(4)}% a frame (p95), over ${100 * STABILITY_P95}%`);
            if (movedFrames === 0) failures.push('the window never moved');
            if (warming.enteringNotReset > 0 || warming.activeTooEarly > 0) failures.push(`entering probes: ${warming.firstProblem}`);
            if (memoryAfterFlight.ddgi !== memoryBefore.ddgi || memoryAfterRest.ddgi !== memoryBefore.ddgi) failures.push('DDGI memory changed while flying');
            if (memoryAfterFlight.platform !== memoryBefore.platform || memoryAfterRest.platform !== memoryBefore.platform) failures.push("the platform's memory changed while flying");
            if (planesWhileResting !== 0) failures.push(`resting on the boundary cleared ${planesWhileResting} planes`);
            if (recompiles !== 0) failures.push(`${recompiles} kernels compiled during the flight`);
            if (failures.length > 0) {
                log(`report: ${JSON.stringify(report)}`);
                // The runner prints only the message of a failed page, so it carries every check's numbers too.
                const outcome = {
                    flight: flightSummary,
                    still: { bandMean: stillSummary.bandMean, bandP95: stillSummary.bandP95 },
                    warming: report.warming,
                    memoryFixed: failures.every((failure) => !failure.includes('memory')),
                    rest: report.rest,
                    recompiles,
                    scrollMsMedian: scrollMedian,
                    movedFrames,
                    flightFrames,
                };
                throw new Error(`${failures.join('; ')}. Every check: ${JSON.stringify(outcome)}`);
            }
            return report;
        } finally {
            scene.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    },
    {
        optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'],
        // Two volumes on Bistro need a little over the default 128 MB storage binding (136 MB on the first run).
        requiredLimits: { maxStorageBufferBindingSize: 512 * 1024 * 1024, maxBufferSize: 512 * 1024 * 1024 },
    },
);

/** Display value for a linear GI value: `scale` maps the mean to 0.25, then sRGB. */
function display(value: number, scale: number): number {
    const linear = Math.min(1, Math.max(0, value * scale));
    const srgb = linear <= 0.0031308 ? 12.92 * linear : 1.055 * linear ** (1 / 2.4) - 0.055;
    return Math.round(255 * srgb);
}

/**
 * The frame strip (two rows: the GI the field gives from the moving camera,
 * and which volume answers: green the camera volume, blue the fixed one), and
 * a row of the observers' change maps over the flight's band pixels, 50×
 * (white at 2% of the image mean a frame). Shown scaled to fit the page, so
 * the runner's screenshot holds them all.
 */
function drawPanels(strip: readonly { frame: number; image: Float32Array; share: Float32Array; edge: Float32Array }[], observers: readonly View[], flight: ChangeTally): void {
    const caption = 20;
    const columns = Math.max(strip.length, observers.length);
    const canvas = document.createElement('canvas');
    canvas.width = columns * STRIP_WIDTH;
    canvas.height = 3 * (STRIP_HEIGHT + caption);
    canvas.style.width = '1240px';
    const context = canvas.getContext('2d');
    if (!context) return;
    context.fillStyle = '#111';
    context.fillRect(0, 0, canvas.width, canvas.height);
    const put = (pixel: (index: number) => [number, number, number], width: number, height: number, column: number, row: number, title: string): void => {
        const image = context.createImageData(width, height);
        for (let index = 0; index < width * height; index++) image.data.set([...pixel(index), 255], index * 4);
        const x = column * STRIP_WIDTH;
        const y = row * (STRIP_HEIGHT + caption);
        context.putImageData(image, x, y + caption);
        context.fillStyle = '#eee';
        context.font = '14px sans-serif';
        context.fillText(title, x + 4, y + 15);
    };
    strip.forEach((shot, column) => {
        let sum = 0;
        let count = 0;
        for (let index = 0; index < shot.edge.length; index++) {
            if (shot.edge[index] < 0) continue;
            sum += shot.image[index * 4] + shot.image[index * 4 + 1] + shot.image[index * 4 + 2];
            count++;
        }
        const scale = 0.25 / Math.max(1e-6, sum / (3 * Math.max(1, count)));
        put((index) => [0, 1, 2].map((channel) => display(shot.image[index * 4 + channel], scale)) as [number, number, number], STRIP_WIDTH, STRIP_HEIGHT, column, 0, `GI, frame ${shot.frame} (${((shot.frame / FRAMES_PER_SECOND) * SPEED_M_S).toFixed(1)} m)`);
        put(
            (index) => {
                if (shot.edge[index] < 0) return [0, 0, 0];
                const cameraShare = shot.share[index];
                const coverage = shot.image[index * 4 + 3];
                return [Math.round(40 * (1 - coverage)), Math.round(200 * cameraShare), Math.round(230 * Math.max(0, coverage - cameraShare))];
            },
            STRIP_WIDTH,
            STRIP_HEIGHT,
            column,
            1,
            'green camera volume, blue fixed',
        );
    });
    observers.forEach((view, column) => {
        const map = flight.changeMaps[column];
        put(
            (index) => {
                // White at 2% of the image's mean GI in one frame.
                const grey = Math.round(255 * Math.min(1, map[index] / 0.02));
                return [grey, grey, grey];
            },
            view.width,
            view.height,
            column,
            2,
            `${view.camera.name}: largest band change a frame (white 2%)`,
        );
    });
    document.body.prepend(canvas);
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function add(a: Vec3, b: Vec3): [number, number, number] {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function scale(a: Vec3, factor: number): [number, number, number] {
    return [a[0] * factor, a[1] * factor, a[2] * factor];
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
