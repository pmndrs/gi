/**
 * A test technique for the `three-screen-inputs` page (RC1.1): it asks for
 * every screen channel and, when armed, reads them at chosen pixels inside
 * its own `encodeScreen`, which is exactly when and how a screen technique
 * would read them. What it reads is compared with three's own read-back of
 * the same targets, so a match shows the platform got this frame's targets.
 *
 * Not a page itself (no index.html).
 */

import type { FrameScreen } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { GITechnique, TechniqueContext, TechniqueScreenFrame } from '../../../../packages/core/src/platform/contract/technique.ts';

/** One pixel's channels as the technique read them. */
export interface ScreenSample {
    readonly pixel: readonly [number, number];
    readonly depth: number;
    readonly normal: readonly [number, number, number];
    readonly velocity: readonly [number, number];
    readonly albedo: readonly [number, number, number];
    readonly instance: number;
}

/** Three vec4s per sample: [depth, normal.xyz], [velocity.xy, albedo.rg], [albedo.b, instance bits, 0, 0]. */
const VEC4_PER_SAMPLE = 3;

const SAMPLE_KERNEL = /* wgsl */ `
@group(0) @binding(0) var depthTexture: texture_depth_2d;
@group(0) @binding(1) var normalTexture: texture_2d<f32>;
@group(0) @binding(2) var velocityTexture: texture_2d<f32>;
@group(0) @binding(3) var albedoTexture: texture_2d<f32>;
@group(0) @binding(4) var instanceTexture: texture_2d<u32>;
@group(0) @binding(5) var<storage, read> pixels: array<vec2u>;
@group(0) @binding(6) var<storage, read_write> samples: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&pixels)) {
        return;
    }
    let pixel = pixels[index];
    let normal = textureLoad(normalTexture, pixel, 0);
    let velocity = textureLoad(velocityTexture, pixel, 0);
    let albedo = textureLoad(albedoTexture, pixel, 0);
    let instance = textureLoad(instanceTexture, pixel, 0).r;
    samples[index * 3u] = vec4f(textureLoad(depthTexture, pixel, 0), normal.xyz);
    samples[index * 3u + 1u] = vec4f(velocity.xy, albedo.xy);
    samples[index * 3u + 2u] = vec4f(albedo.z, bitcast<f32>(instance), 0.0, 0.0);
}
`;

export class ScreenProbeTechnique implements GITechnique {
    readonly id = 'screen-probe';
    readonly requirements = { queries: [], screen: { velocity: true, albedo: true, instance: true } } satisfies GITechnique['requirements'];
    readonly settings = {};
    readonly debugViews = [];
    readonly counters: Record<string, number> = {};
    readonly memoryBytes = 0;
    readonly msPerUnit = undefined;
    /** Screen stages seen, and the latest screen. */
    stages = 0;
    lastScreen: FrameScreen | null = null;

    private device: GPUDevice | null = null;
    private pipeline: GPUComputePipeline | null = null;
    private armed: Array<readonly [number, number]> | null = null;
    private pending: { staging: GPUBuffer; pixels: Array<readonly [number, number]> } | null = null;

    get shading(): never {
        throw new Error('the screen probe has no shading module');
    }

    attach(context: TechniqueContext): void {
        this.device = context.device;
    }

    async prepare(): Promise<void> {
        const device = this.device!;
        this.pipeline = await device.createComputePipelineAsync({
            label: 'screen probe',
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: 'screen probe', code: SAMPLE_KERNEL }), entryPoint: 'main' },
        });
    }

    encode(): number {
        return 0;
    }

    /** The next screen stage reads these pixels; `collect` returns what it read. */
    arm(pixels: Array<readonly [number, number]>): void {
        this.armed = pixels;
    }

    encodeScreen(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame): void {
        this.stages++;
        this.lastScreen = frame.screen;
        const pixels = this.armed;
        if (!pixels) return;
        this.armed = null;
        const device = this.device!;
        const screen = frame.screen;
        const pixelBuffer = device.createBuffer({ size: Math.max(16, pixels.length * 8), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(pixelBuffer, 0, new Uint32Array(pixels.flatMap(([x, y]) => [x, y])));
        const bytes = pixels.length * VEC4_PER_SAMPLE * 16;
        const samples = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const pipeline = this.pipeline!;
        const view = (texture: GPUTexture | undefined, name: string) => {
            if (!texture) throw new Error(`the screen has no ${name}`);
            return texture.createView();
        };
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: screen.depth.createView({ aspect: 'depth-only' }) },
                { binding: 1, resource: screen.normal.createView() },
                { binding: 2, resource: view(screen.velocity, 'velocity') },
                { binding: 3, resource: view(screen.albedo, 'albedo') },
                { binding: 4, resource: view(screen.instance, 'instance') },
                { binding: 5, resource: { buffer: pixelBuffer } },
                { binding: 6, resource: { buffer: samples } },
            ],
        });
        const pass = encoder.beginComputePass({ label: 'screen probe' });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(pixels.length / 64));
        pass.end();
        encoder.copyBufferToBuffer(samples, 0, staging, 0, bytes);
        this.pending = { staging, pixels };
    }

    /** What the last armed stage read (call after the frame that ran it was submitted). */
    async collect(): Promise<ScreenSample[]> {
        const pending = this.pending;
        if (!pending) throw new Error('the screen probe was armed but no screen stage ran');
        this.pending = null;
        await pending.staging.mapAsync(GPUMapMode.READ);
        const floats = new Float32Array(pending.staging.getMappedRange().slice(0));
        pending.staging.destroy();
        const words = new Uint32Array(floats.buffer);
        return pending.pixels.map((pixel, index) => {
            const at = index * VEC4_PER_SAMPLE * 4;
            return {
                pixel,
                depth: floats[at],
                normal: [floats[at + 1], floats[at + 2], floats[at + 3]],
                velocity: [floats[at + 4], floats[at + 5]],
                albedo: [floats[at + 6], floats[at + 7], floats[at + 8]],
                instance: words[at + 9],
            };
        });
    }

    resetHistory(): void {}
    dispose(): void {}
}
