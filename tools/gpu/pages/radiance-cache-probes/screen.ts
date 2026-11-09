/**
 * An engine-free `FrameScreen` for the radiance cache's test page: a primary
 * ray per pixel through the platform's tracer writes depth (`r32float`, the
 * hit projected with the camera's own projection × view, 1 for a miss) and
 * the hit triangle's geometric normal turned towards the camera
 * (`rgba16float`, signed floats), as Capsaicin's visibility buffer gives its
 * geometry normals. A test tool only, never a shading path; the three
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
    position: vec4f,
    forward: vec4f,   // xyz: unit forward; w: tan(fovY / 2)
    right: vec4f,     // xyz: unit right; w: width / height
    up: vec4f,        // xyz: unit up
    size: vec4u,
}
@group(1) @binding(0) var<uniform> pageCamera: PageCamera;
@group(1) @binding(1) var pageDepth: texture_storage_2d<r32float, write>;
@group(1) @binding(2) var pageNormal: texture_storage_2d<rgba16float, write>;

// The largest f32 below 1: a hit never reads as the far plane (the sky).
const PAGE_DEPTH_BELOW_FAR: f32 = 0.99999994;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = pageCamera.size.xy;
    if (any(id.xy >= size)) {
        return;
    }
    // Pixel centre in [-1, 1], y up; row 0 is the top.
    let ndc = vec2f((f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(size.y) * 2.0);
    let tanHalfFov = pageCamera.forward.w;
    let direction = normalize(pageCamera.forward.xyz + pageCamera.right.xyz * ndc.x * tanHalfFov * pageCamera.right.w + pageCamera.up.xyz * ndc.y * tanHalfFov);
    let ray = GiRay(pageCamera.position.xyz, 0.0, direction, 1.0e27);
    let hit = giTraceClosest(ray);
    var depth = 1.0;
    var normal = vec3f(0.0);
    if (hit.t >= 0.0) {
        let surface = giSurfaceAt(ray, hit);
        let clip = pageCamera.viewProjection * vec4f(pageCamera.position.xyz + direction * hit.t, 1.0);
        depth = min(clip.z / clip.w, PAGE_DEPTH_BELOW_FAR);
        normal = select(-surface.geometricNormal, surface.geometricNormal, dot(surface.geometricNormal, direction) < 0.0);
    }
    textureStore(pageDepth, id.xy, vec4f(depth, 0.0, 0.0, 0.0));
    textureStore(pageNormal, id.xy, vec4f(normal, 0.0));
}
`;

const STAGE_COMPUTE = 0x4;

const SCREEN_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache-probes screen group 1',
    entries: [
        { binding: 0, visibility: STAGE_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
        { binding: 2, visibility: STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
    ],
};

/** Draws the page's screen buffers for a camera, on the platform's tracer. */
export class PageScreen {
    readonly screen: FrameScreen;
    private readonly uniforms: GPUBuffer;
    private readonly group: GPUBindGroup;

    private constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ComputeKernel,
        width: number,
        height: number,
    ) {
        const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
        const depth = device.createTexture({ label: 'page depth', size: [width, height], format: 'r32float', usage });
        const normal = device.createTexture({ label: 'page normal', size: [width, height], format: 'rgba16float', usage });
        this.screen = { width, height, depth, normal, normalEncoding: 'snorm-float' };
        this.uniforms = device.createBuffer({ label: 'page camera', size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.group = device.createBindGroup({
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.uniforms } },
                { binding: 1, resource: depth.createView() },
                { binding: 2, resource: normal.createView() },
            ],
        });
    }

    static async create(platform: GIPlatformImpl, device: GPUDevice, width: number, height: number): Promise<PageScreen> {
        const kernel = await platform.kernels.computeKernel({
            label: 'radiance-cache-probes screen',
            modules: [platform.shade.module],
            code: SCREEN_WGSL,
            entryPoint: 'main',
            ownGroups: [SCREEN_GROUP],
        });
        return new PageScreen(device, kernel, width, height);
    }

    /** Records the primary rays into `encoder` for `camera` (whose `frameCamera` the page gives `encode`). */
    encode(encoder: GPUCommandEncoder, camera: PageCamera): void {
        const { width, height } = this.screen;
        const frame = frameCamera(camera, width, height);
        const { forward, right, up } = basis(camera);
        const words = new ArrayBuffer(160);
        const floats = new Float32Array(words);
        floats.set(multiply(frame.projection, frame.view), 0);
        floats.set([...camera.position, 0, ...forward, Math.tan((camera.fovYDeg * Math.PI) / 360), ...right, width / height, ...up, 0], 16);
        new Uint32Array(words, 128, 4).set([width, height, 0, 0]);
        this.device.queue.writeBuffer(this.uniforms, 0, words);
        const pass = encoder.beginComputePass({ label: 'radiance-cache-probes screen' });
        this.kernel.bind(pass);
        pass.setBindGroup(1, this.group);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    dispose(): void {
        this.screen.depth.destroy();
        this.screen.normal.destroy();
        this.uniforms.destroy();
    }
}
