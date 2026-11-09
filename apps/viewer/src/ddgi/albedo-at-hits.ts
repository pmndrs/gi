/**
 * The "GI albedo at hits" debug view (M4, T4.4): the albedo GI's rays read
 * where they hit, for every pixel of the view.
 *
 * Each pixel traces one ray from the camera through the platform's own
 * tracer (`giTraceClosest`) and shades the hit with the platform's hit
 * shading (`giSurfaceAt`), the same calls DDGI's probe rays make, and shows
 * the surface's albedo: material colour × albedo texture × (1 − metalness),
 * with the texture sampled at the mip the hit shading picks. So the picture
 * is what GI sees of the scene's materials: a texture GI doesn't read shows
 * flat, a cutout GI ignores shows solid, glass GI lets through shows what is
 * behind it. Rays that miss show black. Camera rays are a debug tool here,
 * never a shading path.
 *
 * It runs on the platform's implementation (its kernel builder and hit
 * shading module, `GIPlatformImpl.kernels` and `.shade`), which the public
 * `GIPlatform` interface doesn't list, so the platform is checked for them
 * (`albedoViewPlatform`). The image is an rgba16float storage texture the
 * size of the drawing buffer, drawn over the picture as display colours
 * (albedo is a reflectance, not light: no exposure or tone mapping).
 */

import { screenUV, sRGBTransferOETF, texture, vec4 } from 'three/tsl';
import { Vector2, Vector3, type Node, type PerspectiveCamera, type WebGPURenderer } from 'three/webgpu';
import { wrapGpuTexture } from './library';

/** What the view needs of the platform implementation, read by duck typing (see the file comment). */
interface KernelPlatform {
    readonly kernels: {
        computeKernel(options: {
            label: string;
            modules: readonly unknown[];
            code: string;
            entryPoint: string;
            ownGroups?: readonly GPUBindGroupLayoutDescriptor[];
        }): Promise<{ readonly ownLayouts: readonly GPUBindGroupLayout[]; bind(pass: GPUComputePassEncoder): void }>;
    };
    readonly shade: { readonly module: unknown; readonly settings: { sampleTextures: boolean } };
}

/** The platform as the view needs it, or null when it isn't the platform implementation. */
export function albedoViewPlatform(platform: unknown): KernelPlatform | null {
    const candidate = platform as Partial<KernelPlatform> | null;
    const kernels = candidate?.kernels as { computeKernel?: unknown } | undefined;
    const shade = candidate?.shade as { module?: unknown; settings?: { sampleTextures?: unknown } } | undefined;
    if (typeof kernels?.computeKernel !== 'function' || !shade?.module || typeof shade.settings?.sampleTextures !== 'boolean') return null;
    return candidate as KernelPlatform;
}

const IMAGE_FORMAT: GPUTextureFormat = 'rgba16float';
/** position, forward (w: tan of half the vertical field of view), right (w: width / height), up, then the size. */
const CAMERA_BYTES = 80;

const KERNEL_WGSL = /* wgsl */ `
struct AlbedoCamera {
    position: vec4f,
    forward: vec4f,
    right: vec4f,
    up: vec4f,
    size: vec4u,
}
@group(1) @binding(0) var<uniform> albedoCamera: AlbedoCamera;
@group(1) @binding(1) var albedoImage: texture_storage_2d<${IMAGE_FORMAT}, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = albedoCamera.size.xy;
    if (any(id.xy >= size)) {
        return;
    }
    // Pixel centre in [-1, 1], y up; row 0 is the top of the picture, as screenUV reads it.
    let ndc = vec2f((f32(id.x) + 0.5) / f32(size.x) * 2.0 - 1.0, 1.0 - (f32(id.y) + 0.5) / f32(size.y) * 2.0);
    let tanHalfFov = albedoCamera.forward.w;
    let direction = normalize(
        albedoCamera.forward.xyz
        + albedoCamera.right.xyz * ndc.x * tanHalfFov * albedoCamera.right.w
        + albedoCamera.up.xyz * ndc.y * tanHalfFov
    );
    let ray = GiRay(albedoCamera.position.xyz, 0.0, direction, 1.0e27);
    let hit = giTraceClosest(ray);
    var albedo = vec3f(0.0);
    if (hit.t >= 0.0) {
        albedo = giSurfaceAt(ray, hit).albedo;
    }
    textureStore(albedoImage, vec2i(id.xy), vec4f(albedo, 1.0));
}
`;

const drawingBufferSize = new Vector2();
const axisX = new Vector3();
const axisY = new Vector3();
const axisZ = new Vector3();
const eye = new Vector3();

type TextureNode = ReturnType<typeof texture>;

export class AlbedoAtHitsView {
    private kernel: Awaited<ReturnType<KernelPlatform['kernels']['computeKernel']>> | null = null;
    /** The hit shading module the kernel was built with; a new one (a setting changed its code) means a new kernel. */
    private kernelModule: unknown = null;
    private building = false;
    private image: GPUTexture | null = null;
    private readonly cameraBuffer: GPUBuffer;
    private readonly cameraData = new ArrayBuffer(CAMERA_BYTES);
    private bindGroup: GPUBindGroup | null = null;
    private readonly node: TextureNode;
    private disposed = false;

    constructor(
        private readonly device: GPUDevice,
        private readonly platform: KernelPlatform,
    ) {
        this.cameraBuffer = device.createBuffer({ label: 'viewer albedo at hits camera', size: CAMERA_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.node = texture(wrapGpuTexture(this.ensureImage(1, 1), { dimension: '2d', sampleType: 'float' }), screenUV) as TextureNode;
    }

    /** `display` (the picture as the screen shows it) replaced by the albedo image, as display colours. */
    overlay(display: Node): Node {
        void display;
        const albedo = (this.node as unknown as Node<'vec4'>).rgb;
        return vec4(sRGBTransferOETF(albedo as unknown as Node<'vec3'>) as unknown as Node<'vec3'>, 1) as unknown as Node;
    }

    /**
     * Traces the view for this frame, after GI's update and before the picture is drawn. Builds the
     * kernel the first time (and again if the hit shading's code changed); until it is ready the
     * image stays as it was.
     */
    render(renderer: WebGPURenderer, camera: PerspectiveCamera): void {
        if (this.disposed) return;
        const module = this.platform.shade.module;
        if (module !== this.kernelModule && !this.building) this.build(module);
        if (!this.kernel || module !== this.kernelModule) return;

        renderer.getDrawingBufferSize(drawingBufferSize);
        const width = Math.max(1, Math.floor(drawingBufferSize.width));
        const height = Math.max(1, Math.floor(drawingBufferSize.height));
        const image = this.ensureImage(width, height);

        camera.updateMatrixWorld();
        camera.matrixWorld.extractBasis(axisX, axisY, axisZ);
        eye.setFromMatrixPosition(camera.matrixWorld);
        const floats = new Float32Array(this.cameraData, 0, 16);
        const tanHalfFov = Math.tan((camera.getEffectiveFOV() * Math.PI) / 360);
        // A three camera looks down its local -z.
        floats.set([eye.x, eye.y, eye.z, 0, -axisZ.x, -axisZ.y, -axisZ.z, tanHalfFov, axisX.x, axisX.y, axisX.z, width / height, axisY.x, axisY.y, axisY.z, 0]);
        new Uint32Array(this.cameraData, 64, 4).set([width, height, 0, 0]);
        this.device.queue.writeBuffer(this.cameraBuffer, 0, this.cameraData);

        if (!this.bindGroup) {
            this.bindGroup = this.device.createBindGroup({
                label: 'viewer albedo at hits',
                layout: this.kernel.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: this.cameraBuffer } },
                    { binding: 1, resource: image.createView() },
                ],
            });
        }
        const encoder = this.device.createCommandEncoder({ label: 'viewer albedo at hits' });
        const pass = encoder.beginComputePass({ label: 'viewer albedo at hits' });
        this.kernel.bind(pass);
        pass.setBindGroup(1, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    dispose(): void {
        this.disposed = true;
        this.image?.destroy();
        this.cameraBuffer.destroy();
    }

    private build(module: unknown): void {
        this.building = true;
        void this.platform.kernels
            .computeKernel({
                label: 'viewer albedo at hits',
                modules: [module],
                code: KERNEL_WGSL,
                entryPoint: 'main',
                ownGroups: [
                    {
                        label: 'viewer albedo at hits',
                        entries: [
                            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                            { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: IMAGE_FORMAT } },
                        ],
                    },
                ],
            })
            .then((kernel) => {
                this.kernel = kernel;
                this.kernelModule = module;
                this.bindGroup = null;
            })
            .catch((error: unknown) => console.error('the GI albedo view could not be built:', error))
            .finally(() => {
                this.building = false;
            });
    }

    /** The image at this size, made again (and wrapped again for three) when the size changes. */
    private ensureImage(width: number, height: number): GPUTexture {
        if (this.image && this.image.width === width && this.image.height === height) return this.image;
        this.image?.destroy();
        this.image = this.device.createTexture({
            label: 'viewer albedo at hits',
            size: [width, height],
            format: IMAGE_FORMAT,
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.bindGroup = null;
        // A reallocated texture needs a new ExternalTexture on the node (NOTES: needsUpdate throws).
        if (this.node) this.node.value = wrapGpuTexture(this.image, { dimension: '2d', sampleType: 'float' });
        return this.image;
    }
}
