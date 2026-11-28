/**
 * The fixed workload the device page times: one compute pass and one render
 * pass of known size. Both are arithmetic-bound (a chain of fused multiply-adds
 * whose result is written out, so the compiler cannot remove it), which makes
 * the GPU time depend on the GPU and not on memory or driver details.
 *
 * Sizes are chosen so the NVIDIA laptop GPU needs a few milliseconds per pass
 * (Chrome may round timestamp-query values to 100 microseconds, so much
 * shorter passes would read as noise). The Intel GPU takes about ten times
 * longer and phones are somewhere in between; a run stays under ten seconds
 * on the laptop's Intel GPU.
 */

import { type GpuTimer } from '../../../../packages/core/src/platform/timing/index.ts';

export interface WorkloadConfig {
    /** Compute threads are `workgroupsX * workgroupsY * 64`. */
    computeWorkgroupsX: number;
    computeWorkgroupsY: number;
    /** Loop count per compute thread. */
    computeIterations: number;
    renderWidth: number;
    renderHeight: number;
    /** Loop count per fragment. */
    renderIterations: number;
}

export const DEFAULT_WORKLOAD: WorkloadConfig = {
    computeWorkgroupsX: 256,
    computeWorkgroupsY: 256,
    computeIterations: 256,
    renderWidth: 1920,
    renderHeight: 1080,
    renderIterations: 256,
};

export const COMPUTE_PASS_NAME = 'device/compute';
export const RENDER_PASS_NAME = 'device/render';

const COMPUTE_WORKGROUP_SIZE = 64;

// The loop reads its length from a uniform so the shader compiler cannot
// unroll or fold it. The coefficients keep the values finite for the loop
// lengths used here.
const COMPUTE_SHADER = /* wgsl */ `
struct Params {
    computeIterations: u32,
    renderIterations: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> results: array<f32>;

@compute @workgroup_size(${COMPUTE_WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) globalId: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
) {
    let index = globalId.y * workgroups.x * ${COMPUTE_WORKGROUP_SIZE}u + globalId.x;
    var a = f32(index) * 1.0e-6 + 0.5;
    var b = 0.25;
    for (var i = 0u; i < params.computeIterations; i++) {
        a = fma(a, 0.999, b);
        b = fma(b, 0.998, a * 0.0001);
    }
    results[index] = a + b;
}`;

const RENDER_SHADER = /* wgsl */ `
struct Params {
    computeIterations: u32,
    renderIterations: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
    // One oversized triangle that covers the whole target.
    let corner = vec2f(f32((vertexIndex << 1u) & 2u), f32(vertexIndex & 2u));
    return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fragmentMain(@builtin(position) position: vec4f) -> @location(0) vec4f {
    var a = position.x * 1.0e-3 + 0.5;
    var b = position.y * 1.0e-3 + 0.25;
    for (var i = 0u; i < params.renderIterations; i++) {
        a = fma(a, 0.999, b);
        b = fma(b, 0.998, a * 0.0001);
    }
    return vec4f(fract(a), fract(b), 0.5, 1.0);
}`;

export interface Workload {
    readonly config: WorkloadConfig;
    /** Encodes both passes into `encoder`; they are timed when a frame has been begun on `timer`. */
    encode(encoder: GPUCommandEncoder, timer: GpuTimer | null): void;
    destroy(): void;
}

export function createWorkload(device: GPUDevice, config: WorkloadConfig): Workload {
    const paramsBuffer = device.createBuffer({
        label: 'device workload params',
        size: 8,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([config.computeIterations, config.renderIterations]));

    const threadCount = config.computeWorkgroupsX * config.computeWorkgroupsY * COMPUTE_WORKGROUP_SIZE;
    const resultsBuffer = device.createBuffer({
        label: 'device workload results',
        size: threadCount * 4,
        usage: GPUBufferUsage.STORAGE,
    });

    const computePipeline = device.createComputePipeline({
        label: 'device workload compute',
        layout: 'auto',
        compute: {
            module: device.createShaderModule({ label: 'device compute shader', code: COMPUTE_SHADER }),
            entryPoint: 'main',
        },
    });
    const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: resultsBuffer } },
        ],
    });

    const renderModule = device.createShaderModule({ label: 'device render shader', code: RENDER_SHADER });
    const renderPipeline = device.createRenderPipeline({
        label: 'device workload render',
        layout: 'auto',
        vertex: { module: renderModule, entryPoint: 'vertexMain' },
        fragment: { module: renderModule, entryPoint: 'fragmentMain', targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
    });
    const renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: paramsBuffer } }],
    });
    // Offscreen on purpose: no canvas means the page behaves the same headless, in a hidden tab and on a phone.
    const target = device.createTexture({
        label: 'device workload target',
        size: [config.renderWidth, config.renderHeight],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const targetView = target.createView();

    return {
        config,
        encode(encoder, timer) {
            const computePass = encoder.beginComputePass({
                label: COMPUTE_PASS_NAME,
                timestampWrites: timer?.computePass(COMPUTE_PASS_NAME),
            });
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, computeBindGroup);
            computePass.dispatchWorkgroups(config.computeWorkgroupsX, config.computeWorkgroupsY);
            computePass.end();

            const renderPass = encoder.beginRenderPass({
                label: RENDER_PASS_NAME,
                colorAttachments: [{ view: targetView, clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }],
                timestampWrites: timer?.renderPass(RENDER_PASS_NAME),
            });
            renderPass.setPipeline(renderPipeline);
            renderPass.setBindGroup(0, renderBindGroup);
            renderPass.draw(3);
            renderPass.end();
        },
        destroy() {
            paramsBuffer.destroy();
            resultsBuffer.destroy();
            target.destroy();
        },
    };
}
