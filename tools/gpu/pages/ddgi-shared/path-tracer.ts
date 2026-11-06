/**
 * A small GPU path tracer over the platform's own tracer and hit shading,
 * for DDGI diagnostics (T1.9 Sponza investigation; grown from
 * ddgi-irradiance-check's kernel). It shares DDGI's scene, materials, lights
 * and sky exactly, so it answers "what should the field read here" without
 * a Cycles render, and a scene change (a part left out) needs no new Cycles
 * reference once the tracer has been checked against Cycles on the full scene.
 *
 * Two uses:
 * - `irradianceAt(points)`: irradiance E at points with a normal (cosine-
 *   sampled paths), e.g. at probe positions for each irradiance direction;
 * - `renderIndirect(camera)`: the test view's "indirect" quantity per pixel,
 *   albedo / π × E at the primary hit, where E counts sky and emitters seen
 *   straight, but not the lights' direct light at the primary hit (DDGI's
 *   field definition, DECISIONS 2026-11-05).
 *
 * Paths add `giDirectIrradiance` (shadow rays) at every vertex after the
 * first and the sky on a miss, with no Russian roulette, up to `bounces`
 * rays per path.
 *
 * `backFaces` chooses what a path vertex on a back face does:
 * - `black`: the path ends there with no light, as DDGI's probe rays treat
 *   back faces (docs/design/ddgi.md, "Back-face hit");
 * - `shaded`: the back face is shaded like a front face (normal turned to the
 *   ray, as `giSurfaceAt` does), as Cycles shades every surface on both sides.
 * The primary hit of `renderIndirect` is always shaded, as the test view does.
 *
 * Not a page itself (no index.html).
 */

import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { readBuffer, type SurfacePoint } from './harness.ts';

export interface PathTraceOptions {
    /** Paths per point or pixel. */
    readonly paths: number;
    /** Rays per path, the first included. */
    readonly bounces: number;
    readonly backFaces: 'black' | 'shaded';
    /** Varies the random numbers between runs. */
    readonly seed?: number;
    /**
     * `renderIndirect` only: start the paths from DDGI's biased lookup point
     * (the surface point + normal × `normal` − view direction × `view`, metres)
     * instead of the surface, still over the surface normal's hemisphere. That
     * is what an exact field, read where DDGI reads it, would return.
     */
    readonly lookupBias?: { readonly normal: number; readonly view: number };
}

/** Paths per point or pixel in one dispatch: keeps each submit well under the GPU watchdog on Sponza. */
const PATHS_PER_DISPATCH = 2;
/** Threads sharing one point's paths in `irradianceAt`, so a few thousand points still fill the GPU. */
const LANES_PER_POINT = 64;

const PATH_TRACER_WGSL = /* wgsl */ `
struct PtSettings {
    pathsPerDispatch: u32,
    bounces: u32,
    seed: u32,
    shadeBackFaces: u32,     // 0: a back face ends the path with no light; 1: shaded like a front face
    imageSize: vec2u,        // renderIndirect only
    pointCount: u32,         // irradianceAt only
    lanesPerPoint: u32,
    lookupNormalBias: f32,   // renderIndirect only: see PathTraceOptions.lookupBias
    lookupViewBias: f32,
    useLookupBias: u32,
}

struct PtCamera {
    position: vec4f,
    forward: vec4f,   // xyz: unit forward; w: tan(fovY / 2)
    right: vec4f,     // xyz: unit right; w: width / height
    up: vec4f,
}

struct PtPoint { position: vec4f, normal: vec4f }

@group(1) @binding(0) var<uniform> pt: PtSettings;
@group(1) @binding(1) var<storage, read> ptInputs: array<vec4f>;
// Per pixel or per (point, lane): rgb sum of path results, w the number of paths.
@group(1) @binding(2) var<storage, read_write> ptSums: array<vec4f>;

const PT_INV_PI: f32 = 0.318309886183791;
const PT_TWO_PI: f32 = 6.28318530717959;

// PCG hash (Jarzynski and Olano 2020).
fn ptHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
var<private> ptState: u32;
fn ptRandom() -> f32 {
    ptState = ptHash(ptState);
    return f32(ptState >> 8u) * (1.0 / 16777216.0);
}

// Cosine-weighted direction around a unit normal (Malley's method).
fn ptCosineDirection(normal: vec3f) -> vec3f {
    let u = ptRandom();
    let v = ptRandom();
    let radius = sqrt(u);
    let angle = PT_TWO_PI * v;
    let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
    let tangent = normalize(cross(helper, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle)) + normal * sqrt(max(0.0, 1.0 - u)));
}

// Radiance arriving at 'origin' along -direction: the sky on a miss, or the
// first surface's emission and direct light, plus its own bounces.
fn ptIncomingRadiance(origin: vec3f, direction: vec3f) -> vec3f {
    var rayOrigin = origin;
    var rayDirection = direction;
    var throughput = vec3f(1.0);
    var sum = vec3f(0.0);
    for (var bounce = 0u; bounce < pt.bounces; bounce++) {
        let ray = GiRay(rayOrigin, 0.0, rayDirection, 1.0e27);
        let hit = giTraceClosest(ray);
        if (hit.t < 0.0) {
            sum += throughput * giSkyRadiance(rayDirection);
            break;
        }
        if (!hit.frontFace && pt.shadeBackFaces == 0u) {
            break;
        }
        let surface = giSurfaceAt(ray, hit);
        sum += throughput * (surface.emission + surface.albedo * PT_INV_PI * giDirectIrradiance(surface, 0xfu, vec2f(ptRandom(), ptRandom())));
        throughput *= surface.albedo;
        if (all(throughput == vec3f(0.0))) {
            break;
        }
        rayOrigin = giShadeOffsetRayOrigin(surface.position, giShadeRaySide(surface));
        rayDirection = ptCosineDirection(surface.normal);
    }
    return sum;
}

// One lane of one point: paths from the point over the hemisphere of its normal.
// E = pi × the mean incoming radiance under cosine sampling.
@compute @workgroup_size(64)
fn ptPoints(@builtin(global_invocation_id) id: vec3u) {
    let slot = id.x;
    let point = slot / pt.lanesPerPoint;
    if (point >= pt.pointCount) {
        return;
    }
    ptState = ptHash(slot * 1000003u + pt.seed * 7919u + 17u);
    let position = ptInputs[2u * point].xyz;
    let normal = normalize(ptInputs[2u * point + 1u].xyz);
    var sum = vec3f(0.0);
    for (var path = 0u; path < pt.pathsPerDispatch; path++) {
        sum += ptIncomingRadiance(position, ptCosineDirection(normal));
    }
    ptSums[slot] += vec4f(sum, f32(pt.pathsPerDispatch));
}

// One pixel: the primary hit at the pixel centre (as the DDGI test view),
// then albedo × the mean incoming radiance = albedo / pi × E.
@compute @workgroup_size(8, 8)
fn ptCameraView(@builtin(global_invocation_id) id: vec3u) {
    let size = pt.imageSize;
    if (any(id.xy >= size)) {
        return;
    }
    let pixel = id.y * size.x + id.x;
    ptState = ptHash(pixel * 1000003u + pt.seed * 7919u + 29u);
    let position = ptInputs[0];
    let forward = ptInputs[1];
    let right = ptInputs[2];
    let up = ptInputs[3];
    let ndc = vec2f((f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(size.y) * 2.0);
    let direction = normalize(forward.xyz + right.xyz * ndc.x * forward.w * right.w + up.xyz * ndc.y * forward.w);
    let primary = GiRay(position.xyz, 0.0, direction, 1.0e27);
    let hit = giTraceClosest(primary);
    if (hit.t < 0.0) {
        ptSums[pixel] += vec4f(0.0, 0.0, 0.0, f32(pt.pathsPerDispatch));
        return;
    }
    let surface = giSurfaceAt(primary, hit);
    var origin = giShadeOffsetRayOrigin(surface.position, giShadeRaySide(surface));
    if (pt.useLookupBias != 0u) {
        origin = surface.position + surface.normal * pt.lookupNormalBias - direction * pt.lookupViewBias;
    }
    var sum = vec3f(0.0);
    for (var path = 0u; path < pt.pathsPerDispatch; path++) {
        sum += ptIncomingRadiance(origin, ptCosineDirection(surface.normal));
    }
    ptSums[pixel] += vec4f(surface.albedo * sum, f32(pt.pathsPerDispatch));
}
`;

type ComputeKernel = Awaited<ReturnType<GIPlatformImpl['kernels']['computeKernel']>>;

const OWN_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi path tracer',
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
};

export class PathTracer {
    private constructor(
        private readonly device: GPUDevice,
        private readonly pointsKernel: ComputeKernel,
        private readonly cameraKernel: ComputeKernel,
    ) {}

    static async create(device: GPUDevice, platform: GIPlatformImpl): Promise<PathTracer> {
        const make = (entryPoint: string) =>
            platform.kernels.computeKernel({ label: `ddgi path tracer ${entryPoint}`, modules: [platform.shade.module], code: PATH_TRACER_WGSL, entryPoint, ownGroups: [OWN_GROUP] });
        const [pointsKernel, cameraKernel] = await Promise.all([make('ptPoints'), make('ptCameraView')]);
        return new PathTracer(device, pointsKernel, cameraKernel);
    }

    /** Irradiance E (rgb) at each point, over the hemisphere of its normal. */
    async irradianceAt(points: readonly Pick<SurfacePoint, 'position' | 'normal'>[], options: PathTraceOptions): Promise<[number, number, number][]> {
        const inputs = new Float32Array(Math.max(1, points.length) * 8);
        points.forEach((point, index) => {
            inputs.set(point.position, index * 8);
            inputs.set(point.normal, index * 8 + 4);
        });
        const slots = points.length * LANES_PER_POINT;
        // Each of a point's lanes traces its share of the paths.
        const pathsPerLane = Math.ceil(options.paths / LANES_PER_POINT);
        const sums = await this.run(this.pointsKernel, inputs, slots, [Math.ceil(slots / 64), 1], { ...options, paths: pathsPerLane }, [0, 0], points.length);
        return points.map((_, index) => {
            const total = [0, 0, 0];
            let paths = 0;
            for (let lane = 0; lane < LANES_PER_POINT; lane++) {
                const base = (index * LANES_PER_POINT + lane) * 4;
                for (let channel = 0; channel < 3; channel++) total[channel] += sums[base + channel];
                paths += sums[base + 3];
            }
            return [0, 1, 2].map((channel) => (Math.PI * total[channel]) / paths) as [number, number, number];
        });
    }

    /** The test view's indirect image (rgb per pixel, row 0 at the top) from paths. */
    async renderIndirect(camera: CameraBookmark, width: number, height: number, options: PathTraceOptions): Promise<Float32Array> {
        const forward = normalize3(subtract(camera.target, camera.position));
        const right = normalize3(cross(forward, [0, 1, 0]));
        const up = cross(right, forward);
        const inputs = new Float32Array([...camera.position, 0, ...forward, Math.tan((camera.fovYDeg * Math.PI) / 360), ...right, width / height, ...up, 0]);
        const pixels = width * height;
        const sums = await this.run(this.cameraKernel, inputs, pixels, [Math.ceil(width / 8), Math.ceil(height / 8)], options, [width, height], 0);
        const image = new Float32Array(pixels * 3);
        for (let pixel = 0; pixel < pixels; pixel++) {
            const paths = sums[pixel * 4 + 3];
            for (let channel = 0; channel < 3; channel++) image[pixel * 3 + channel] = paths > 0 ? sums[pixel * 4 + channel] / paths : 0;
        }
        return image;
    }

    private async run(
        kernel: ComputeKernel,
        inputs: Float32Array,
        slots: number,
        workgroups: [number, number],
        options: PathTraceOptions,
        imageSize: [number, number],
        pointCount: number,
    ): Promise<Float32Array> {
        const device = this.device;
        const inputBuffer = device.createBuffer({ label: 'path tracer inputs', size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(inputBuffer, 0, inputs as Float32Array<ArrayBuffer>);
        const sumBytes = Math.max(16, slots * 16);
        const sumBuffer = device.createBuffer({ label: 'path tracer sums', size: sumBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const settingsBuffer = device.createBuffer({ label: 'path tracer settings', size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bindGroup = device.createBindGroup({
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: settingsBuffer } },
                { binding: 1, resource: { buffer: inputBuffer } },
                { binding: 2, resource: { buffer: sumBuffer } },
            ],
        });
        const dispatches = Math.ceil(options.paths / PATHS_PER_DISPATCH);
        for (let dispatch = 0; dispatch < dispatches; dispatch++) {
            const settings = new ArrayBuffer(48);
            new Uint32Array(settings, 0, 8).set([
                PATHS_PER_DISPATCH,
                options.bounces,
                (options.seed ?? 0) * 65_537 + dispatch,
                options.backFaces === 'shaded' ? 1 : 0,
                imageSize[0],
                imageSize[1],
                pointCount,
                LANES_PER_POINT,
            ]);
            new Float32Array(settings, 32, 2).set([options.lookupBias?.normal ?? 0, options.lookupBias?.view ?? 0]);
            new Uint32Array(settings, 40, 1).set([options.lookupBias ? 1 : 0]);
            device.queue.writeBuffer(settingsBuffer, 0, settings);
            const encoder = device.createCommandEncoder({ label: 'path tracer' });
            const pass = encoder.beginComputePass();
            kernel.bind(pass);
            pass.setBindGroup(1, bindGroup);
            pass.dispatchWorkgroups(workgroups[0], workgroups[1]);
            pass.end();
            device.queue.submit([encoder.finish()]);
            // One dispatch in flight at a time: each is a fraction of a second, and the page stays responsive.
            await device.queue.onSubmittedWorkDone();
        }
        const sums = new Float32Array(await readBuffer(device, sumBuffer, sumBytes));
        inputBuffer.destroy();
        sumBuffer.destroy();
        settingsBuffer.destroy();
        return sums;
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
