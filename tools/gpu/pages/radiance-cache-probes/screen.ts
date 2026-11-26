/**
 * An engine-free `FrameScreen` for the radiance cache's test page: a primary
 * ray per pixel through the platform's tracer writes depth (`r32float`, the
 * hit projected with the camera's own projection × view, 1 for a miss) and
 * the hit triangle's geometric normal turned towards the camera
 * (`rgba16float`, signed floats), as Capsaicin's visibility buffer gives its
 * geometry normals, and the motion of each hit since the previous `encode`'s
 * camera (`rg16float` in an `rgba16float`: NDC, current − previous, y up, as
 * `FrameScreen.velocity` says; 0 for the sky; RC4.1), and for SSGI (RC4.2) the
 * hit's albedo and its own light (front-face emission plus one sample of the
 * platform's direct light through albedo / π: `FrameScreen.albedo` and
 * `.lighting`). A test tool only, never a shading path; the three
 * adapter's screen inputs (RC1.1) take its place in RC2.2.
 *
 * The camera is built here in f64: a look-at view and WebGPU's 0..1-depth
 * perspective, column-major, the same matrices the page hands to
 * `platform.encode` as `FrameCamera`.
 */

import type { FrameCamera, FrameScreen } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { ComputeKernel } from '../../../../packages/ddgi-core/src/platform/contract/wgsl.ts';
import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';

type Vector = [number, number, number];

export interface PageCamera {
    readonly position: Vector;
    readonly target: Vector;
    readonly fovYDeg: number;
    readonly near: number;
    readonly far: number;
}

const subtract = (a: Vector, b: Vector): Vector => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vector, b: Vector): Vector => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: Vector, b: Vector): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const normalize = (a: Vector): Vector => {
    const length = Math.hypot(...a);
    return [a[0] / length, a[1] / length, a[2] / length];
};

/** The camera's basis: forward, right and up, unit length. */
function basis(camera: PageCamera): { forward: Vector; right: Vector; up: Vector } {
    const forward = normalize(subtract(camera.target, camera.position));
    // Looking straight up or down, y can't be the up hint.
    const hint: Vector = Math.abs(forward[1]) > 0.999 ? [0, 0, -1] : [0, 1, 0];
    const right = normalize(cross(forward, hint));
    return { forward, right, up: cross(right, forward) };
}

/** `FrameCamera` for `platform.encode`: look-at view and a WebGPU (0..1 depth) perspective. */
export function frameCamera(camera: PageCamera, width: number, height: number): FrameCamera {
    const { forward, right, up } = basis(camera);
    const eye = camera.position;
    const view = [right[0], up[0], -forward[0], 0, right[1], up[1], -forward[1], 0, right[2], up[2], -forward[2], 0, -dot(right, eye), -dot(up, eye), dot(forward, eye), 1];
    const f = 1 / Math.tan((camera.fovYDeg * Math.PI) / 360);
    const { near, far } = camera;
    const projection = [f / (width / height), 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0];
    return { position: eye, view, projection, near, far };
}

function multiply(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
        }
    }
    return out;
}

const SCREEN_WGSL = /* wgsl */ `
struct PageCamera {
    viewProjection: mat4x4f,
    previousViewProjection: mat4x4f,
    position: vec4f,
    forward: vec4f,   // xyz: unit forward; w: tan(fovY / 2)
    right: vec4f,     // xyz: unit right; w: width / height
    up: vec4f,        // xyz: unit up
    size: vec4u,
    // x: a normal map's tilt in radians (0: none), y: its wavelength along world x, in metres. The
    // written normal turns about world z by x * sin(2 pi hit.x / y), as a normal map would; depth stays.
    normalBump: vec4f,
}
@group(1) @binding(0) var<uniform> pageCamera: PageCamera;
@group(1) @binding(1) var pageDepth: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var pageNormal: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var pageVelocity: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var pageAlbedo: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var pageLighting: texture_storage_2d<rgba16float, write>;

// A random pair per pixel and frame for the direct light's sample.
fn pageHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// The largest f32 below 1: a hit never reads as the far plane (the sky).
const PAGE_DEPTH_BELOW_FAR: f32 = 0.99999994;

// The pixel's primary ray, from the camera through the pixel centre.
fn pageRay(id: vec2u) -> GiRay {
    let size = pageCamera.size.xy;
    // Pixel centre in [-1, 1], y up; row 0 is the top.
    let ndc = vec2f((f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(size.y) * 2.0);
    let tanHalfFov = pageCamera.forward.w;
    let direction = normalize(pageCamera.forward.xyz + pageCamera.right.xyz * ndc.x * tanHalfFov * pageCamera.right.w + pageCamera.up.xyz * ndc.y * tanHalfFov);
    return GiRay(pageCamera.position.xyz, 0.0, direction, 1.0e27);
}

// Depth, normal and velocity (the default device has four storage textures a stage, so albedo and
// lighting are a second entry point over the same rays).
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= pageCamera.size.xy)) {
        return;
    }
    let ray = pageRay(id.xy);
    let direction = ray.direction;
    let hit = giTraceClosest(ray);
    var depth = 1.0;
    var normal = vec3f(0.0);
    var velocity = vec2f(0.0);
    if (hit.t >= 0.0) {
        let surface = giSurfaceAt(ray, hit);
        let point = vec4f(pageCamera.position.xyz + direction * hit.t, 1.0);
        let clip = pageCamera.viewProjection * point;
        let previousClip = pageCamera.previousViewProjection * point;
        depth = min(clip.z / clip.w, PAGE_DEPTH_BELOW_FAR);
        velocity = clip.xy / clip.w - previousClip.xy / previousClip.w;
        normal = select(-surface.geometricNormal, surface.geometricNormal, dot(surface.geometricNormal, direction) < 0.0);
        if (pageCamera.normalBump.x != 0.0) {
            let tilt = pageCamera.normalBump.x * sin(6.28318530717959 * point.x / pageCamera.normalBump.y);
            normal = vec3f(cos(tilt) * normal.x - sin(tilt) * normal.y, sin(tilt) * normal.x + cos(tilt) * normal.y, normal.z);
        }
    }
    textureStore(pageDepth, id.xy, vec4f(depth, 0.0, 0.0, 0.0));
    textureStore(pageNormal, id.xy, vec4f(normal, 0.0));
    textureStore(pageVelocity, id.xy, vec4f(velocity, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) id: vec3u) {
    let size = pageCamera.size.xy;
    if (any(id.xy >= size)) {
        return;
    }
    let ray = pageRay(id.xy);
    let hit = giTraceClosest(ray);
    var albedo = vec3f(0.0);
    var lighting = vec3f(0.0);
    if (hit.t >= 0.0) {
        let surface = giSurfaceAt(ray, hit);
        // FrameScreen.albedo and .lighting: the host's own light, emission and one sample of direct light.
        albedo = surface.albedo;
        let first = pageHash(id.x + id.y * size.x + pageHash(pageCamera.size.z));
        let random = vec2f(f32(first >> 8u), f32(pageHash(first) >> 8u)) / 16777216.0;
        let emission = select(vec3f(0.0), surface.emission, surface.frontFace);
        lighting = emission + surface.albedo / 3.14159265358979 * giDirectIrradiance(surface, 0xffffffffu, random);
    }
    textureStore(pageAlbedo, id.xy, vec4f(albedo, 1.0));
    textureStore(pageLighting, id.xy, vec4f(lighting, 1.0));
}
`;

const STAGE_COMPUTE = 0x4;
/** Bytes of `PageCamera`: two matrices, four vec4s, the size, the normal map. */
const UNIFORM_BYTES = 224;

const SCREEN_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache-probes screen group 1',
    entries: [
        { binding: 0, visibility: STAGE_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
        { binding: 2, visibility: STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 3, visibility: STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
};

const SHADE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache-probes screen shade group 1',
    entries: [
        { binding: 0, visibility: STAGE_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 4, visibility: STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 5, visibility: STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
};

/** Draws the page's screen buffers for a camera, on the platform's tracer. */
export class PageScreen {
    readonly screen: FrameScreen;
    private readonly uniforms: GPUBuffer;
    private readonly group: GPUBindGroup;
    private readonly shadeGroup: GPUBindGroup;
    /** The camera of the previous `encode`, for the velocity. */
    private previousViewProjection: number[] | null = null;
    /** Frames drawn: the direct light's random numbers change with it. */
    private frames = 0;
    /** A normal map on every surface, for tests (`normalBump` in the WGSL): tilt in radians and wavelength in metres; 0 for none. */
    normalBump: { readonly tilt: number; readonly wavelength: number } = { tilt: 0, wavelength: 1 };

    private constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ComputeKernel,
        private readonly shadeKernel: ComputeKernel,
        width: number,
        height: number,
    ) {
        const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
        const depth = device.createTexture({ label: 'page depth', size: [width, height], format: 'r32float', usage });
        const normal = device.createTexture({ label: 'page normal', size: [width, height], format: 'rgba16float', usage });
        const velocity = device.createTexture({ label: 'page velocity', size: [width, height], format: 'rgba16float', usage });
        const albedo = device.createTexture({ label: 'page albedo', size: [width, height], format: 'rgba16float', usage });
        const lighting = device.createTexture({ label: 'page lighting', size: [width, height], format: 'rgba16float', usage });
        this.screen = { width, height, depth, normal, normalEncoding: 'snorm-float', velocity, albedo, lighting };
        this.uniforms = device.createBuffer({ label: 'page camera', size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.group = device.createBindGroup({
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.uniforms } },
                { binding: 1, resource: depth.createView() },
                { binding: 2, resource: normal.createView() },
                { binding: 3, resource: velocity.createView() },
            ],
        });
        this.shadeGroup = device.createBindGroup({
            layout: shadeKernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.uniforms } },
                { binding: 4, resource: albedo.createView() },
                { binding: 5, resource: lighting.createView() },
            ],
        });
    }

    static async create(platform: GIPlatformImpl, device: GPUDevice, width: number, height: number): Promise<PageScreen> {
        const make = (entryPoint: string, group: GPUBindGroupLayoutDescriptor) =>
            platform.kernels.computeKernel({ label: `radiance-cache-probes screen ${entryPoint}`, modules: [platform.shade.module], code: SCREEN_WGSL, entryPoint, ownGroups: [group] });
        const [kernel, shadeKernel] = await Promise.all([make('main', SCREEN_GROUP), make('shade', SHADE_GROUP)]);
        return new PageScreen(device, kernel, shadeKernel, width, height);
    }

    /**
     * Records the primary rays into `encoder` for `camera` (whose `frameCamera` the page gives `encode`).
     * The velocity is the motion since the previous `encode`'s camera (none on the first).
     */
    encode(encoder: GPUCommandEncoder, camera: PageCamera): void {
        const { width, height } = this.screen;
        const frame = frameCamera(camera, width, height);
        const viewProjection = multiply(frame.projection, frame.view);
        const { forward, right, up } = basis(camera);
        const words = new ArrayBuffer(UNIFORM_BYTES);
        const floats = new Float32Array(words);
        floats.set(viewProjection, 0);
        floats.set(this.previousViewProjection ?? viewProjection, 16);
        floats.set([...camera.position, 0, ...forward, Math.tan((camera.fovYDeg * Math.PI) / 360), ...right, width / height, ...up, 0], 32);
        new Uint32Array(words, 192, 4).set([width, height, this.frames++, 0]);
        new Float32Array(words, 208, 4).set([this.normalBump.tilt, this.normalBump.wavelength, 0, 0]);
        this.previousViewProjection = viewProjection;
        this.device.queue.writeBuffer(this.uniforms, 0, words);
        const pass = encoder.beginComputePass({ label: 'radiance-cache-probes screen' });
        this.kernel.bind(pass);
        pass.setBindGroup(1, this.group);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        this.shadeKernel.bind(pass);
        pass.setBindGroup(1, this.shadeGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    dispose(): void {
        this.screen.depth.destroy();
        this.screen.normal.destroy();
        this.screen.velocity?.destroy();
        this.screen.albedo?.destroy();
        this.screen.lighting?.destroy();
        this.uniforms.destroy();
    }
}
