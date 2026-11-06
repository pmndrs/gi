/**
 * Shared by the hit-shading GPU pages (T1.4): a platform with a scene, frames
 * until its trees are placed, and test kernels built by the platform's own
 * kernel builder, so the hit-shading module is compiled exactly as a
 * technique would compile it (scene copy, rays and hit shading in group 0).
 *
 * Not a page itself (no index.html).
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** A frame with no technique work: the platform uploads the scene, places trees, and projects the sky if it changed. */
export const IDLE_FRAME: FrameInput = {
    deltaTime: 0,
    budget: { units: 0 },
    camera: { position: [0, 0, 0], view: IDENTITY, projection: IDENTITY, near: 0.1, far: 100 },
};

export function frame(device: GPUDevice, platform: GIPlatformImpl): void {
    const encoder = device.createCommandEncoder({ label: 'hit shading test frame' });
    platform.encode(encoder, IDLE_FRAME);
    device.queue.submit([encoder.finish()]);
}

/** `prepare`, then frames until every mesh tree has arrived and is placed. */
export async function ready(device: GPUDevice, platform: GIPlatformImpl): Promise<void> {
    await platform.prepare();
    frame(device, platform);
    await platform.rays.whenBuilt();
    frame(device, platform);
    if (platform.rays.building) throw new Error('the platform still reports trees building');
}

export async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'hit shading readback', size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'hit shading readback' });
    encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** Group 1 of every test kernel: inputs (read-only) and outputs, both `array<vec4f>` unless the code says otherwise. */
const TEST_GROUP: GPUBindGroupLayoutDescriptor = {
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
};

/**
 * A compiled test kernel over the platform's hit-shading module. Its WGSL
 * (after the platform modules) declares `inputs` at @group(1) @binding(0),
 * `outputs` at @binding(1), and the entry point, with @workgroup_size(64).
 * `run` can be called again after scene or setting changes (and a `frame`).
 */
type ComputeKernel = Awaited<ReturnType<GIPlatformImpl['kernels']['computeKernel']>>;

export class ShadeKernel {
    private readonly device: GPUDevice;
    private readonly kernel: ComputeKernel;

    private constructor(device: GPUDevice, kernel: ComputeKernel) {
        this.device = device;
        this.kernel = kernel;
    }

    static async create(device: GPUDevice, platform: GIPlatformImpl, label: string, code: string, entryPoint = 'main'): Promise<ShadeKernel> {
        const kernel = await platform.kernels.computeKernel({
            label,
            modules: [platform.shade.module],
            code,
            entryPoint,
            ownGroups: [TEST_GROUP],
        });
        return new ShadeKernel(device, kernel);
    }

    async run(inputs: Float32Array | Uint32Array, outputBytes: number, invocations: number): Promise<ArrayBuffer> {
        const device = this.device;
        // A storage binding can't be empty: keep at least one vec4.
        const inputBuffer = device.createBuffer({ label: 'hit shading test inputs', size: Math.max(16, inputs.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(inputBuffer, 0, inputs as Float32Array<ArrayBuffer>);
        const outputBuffer = device.createBuffer({ label: 'hit shading test outputs', size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const group = device.createBindGroup({
            layout: this.kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } },
            ],
        });
        const encoder = device.createCommandEncoder({ label: 'hit shading test' });
        const pass = encoder.beginComputePass({ label: 'hit shading test' });
        this.kernel.bind(pass);
        pass.setBindGroup(1, group);
        pass.dispatchWorkgroups(Math.ceil(invocations / 64));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const result = await readBuffer(device, outputBuffer, outputBytes);
        inputBuffer.destroy();
        outputBuffer.destroy();
        return result;
    }
}

export type Vec3 = [number, number, number];

export function normalize(vector: readonly number[]): Vec3 {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    return [vector[0] / length, vector[1] / length, vector[2] / length];
}

/** Relative difference, with a floor on the denominator so values near 0 are judged in absolute terms. */
export function relativeError(found: number, expected: number, floor: number): number {
    return Math.abs(found - expected) / Math.max(Math.abs(expected), floor);
}
