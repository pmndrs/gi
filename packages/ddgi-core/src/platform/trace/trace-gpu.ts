/**
 * GPU side of the `rays` backend: uploads the BVH buffers, exposes the trace
 * code as a platform `WgslModule`, and runs batches of rays from buffers.
 *
 * The host owns the device (nothing here creates one) and errors are the
 * host's to listen for with `addEventListener('uncapturederror')`.
 */

import { GILimitError } from '../contract/errors.ts';
import type { BindingSlot, ModuleBindingEntry, WgslModule } from '../contract/wgsl.ts';
import { DEFAULT_TRIANGLE_TEST, type TriangleTest } from './layout.ts';
import type { TraceRay } from './reference-tracer.ts';
import type { TraceSceneData } from './scene-assembly.ts';
import { BATCH_WORKGROUP_SIZE, batchKernelWgsl, TRACE_MODULE_ENTRIES, traceModuleWgsl } from './wgsl.ts';

/** Bytes per ray in a batch: `GiRay` is 32 bytes. */
export const RAY_BYTES = 32;

/** The three storage buffers the tracer reads. */
export interface TraceSceneBuffers {
    readonly nodes: GPUBuffer;
    readonly triangles: GPUBuffer;
    readonly topLevel: GPUBuffer;
    /** Bytes held on the GPU. */
    readonly byteLength: number;
    /** The test the triangles were built for. */
    readonly triangleTest: TriangleTest;
    destroy(): void;
}

function checkStorageBinding(device: GPUDevice, name: string, byteLength: number): void {
    const bindingLimit = device.limits.maxStorageBufferBindingSize;
    if (byteLength > bindingLimit) throw new GILimitError('maxStorageBufferBindingSize', byteLength, bindingLimit);
    const bufferLimit = device.limits.maxBufferSize;
    if (byteLength > bufferLimit) throw new GILimitError('maxBufferSize', byteLength, bufferLimit);
    if (byteLength === 0) throw new RangeError(`trace buffer '${name}' is empty`);
}

/**
 * Copies an assembled scene to the GPU. Throws `GILimitError` naming the
 * limit to request if a buffer is larger than one storage binding may be
 * (128 MB on a default device, about 2.7 M triangles).
 */
export function uploadTraceScene(device: GPUDevice, scene: TraceSceneData, label = 'trace'): TraceSceneBuffers {
    const parts: [string, ArrayBuffer][] = [
        ['nodes', scene.nodes],
        ['triangles', scene.triangles],
        ['top level', scene.topLevel.buffer],
    ];
    for (const [name, data] of parts) checkStorageBinding(device, name, data.byteLength);

    const buffers = parts.map(([name, data]) => {
        const buffer = device.createBuffer({
            label: `${label} ${name}`,
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    });
    const [nodes, triangles, topLevel] = buffers;
    return {
        nodes,
        triangles,
        topLevel,
        byteLength: scene.byteLength,
        triangleTest: scene.triangleTest,
        destroy: () => buffers.forEach((buffer) => buffer.destroy()),
    };
}

/**
 * The trace code and its three bindings as a platform module, so a technique's
 * kernel can include it and call `giTraceClosest` / `giTraceAny` inline.
 */
export function createTraceModule(
    getBuffers: () => TraceSceneBuffers,
    getVersion: () => number = () => 0,
    triangleTest: TriangleTest = DEFAULT_TRIANGLE_TEST,
): WgslModule {
    return {
        name: 'gi-trace',
        dependencies: [],
        code: (slot: BindingSlot) => traceModuleWgsl(slot, triangleTest),
        entries: TRACE_MODULE_ENTRIES as readonly ModuleBindingEntry[],
        resources: () => {
            const buffers = getBuffers();
            return [{ buffer: buffers.nodes }, { buffer: buffers.triangles }, { buffer: buffers.topLevel }];
        },
        get version() {
            return getVersion();
        },
        storageBufferCount: TRACE_MODULE_ENTRIES.length,
    };
}

export interface TraceBatch {
    readonly mode: 'closest' | 'any';
    /** At least `count` rays of 32 bytes (`GiRay`). */
    readonly rays: GPUBuffer;
    /** Closest: `count` records of `HIT_RECORD_BYTES`. Any: `count` u32, 1 = blocked. */
    readonly results: GPUBuffer;
    readonly count: number;
}

interface SceneBindGroups {
    readonly closest: GPUBindGroup;
    readonly any: GPUBindGroup;
}

/**
 * Runs batches of rays through the trace module. Group 0 is the scene; group 1
 * is the batch's rays and results. One kernel serves any number of scenes.
 */
export class TraceBatchKernel {
    private readonly sceneGroups = new WeakMap<TraceSceneBuffers, SceneBindGroups>();

    private constructor(
        private readonly device: GPUDevice,
        private readonly triangleTest: TriangleTest,
        private readonly closestPipeline: GPUComputePipeline,
        private readonly anyPipeline: GPUComputePipeline,
    ) {}

    /** `triangleTest` must match the scenes this kernel will trace (`TraceSceneData.triangleTest`). */
    static async create(
        device: GPUDevice,
        triangleTest: TriangleTest = DEFAULT_TRIANGLE_TEST,
        label = 'trace',
    ): Promise<TraceBatchKernel> {
        const pipelineFor = (mode: 'closest' | 'any') =>
            device.createComputePipelineAsync({
                label: `${label} batch ${mode} ${triangleTest}`,
                layout: 'auto',
                compute: {
                    module: device.createShaderModule({ label: `${label} batch ${mode}`, code: batchKernelWgsl(mode, triangleTest) }),
                    entryPoint: 'giBatch',
                },
            });
        const [closestPipeline, anyPipeline] = await Promise.all([pipelineFor('closest'), pipelineFor('any')]);
        return new TraceBatchKernel(device, triangleTest, closestPipeline, anyPipeline);
    }

    private groupsFor(scene: TraceSceneBuffers): SceneBindGroups {
        let groups = this.sceneGroups.get(scene);
        if (!groups) {
            const make = (pipeline: GPUComputePipeline) =>
                this.device.createBindGroup({
                    label: 'trace scene',
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: scene.nodes } },
                        { binding: 1, resource: { buffer: scene.triangles } },
                        { binding: 2, resource: { buffer: scene.topLevel } },
                    ],
                });
            groups = { closest: make(this.closestPipeline), any: make(this.anyPipeline) };
            this.sceneGroups.set(scene, groups);
        }
        return groups;
    }

    /** Records one dispatch into `encoder`. */
    encode(encoder: GPUCommandEncoder, scene: TraceSceneBuffers, batch: TraceBatch): void {
        if (batch.count <= 0) return;
        if (scene.triangleTest !== this.triangleTest) {
            throw new Error(`trace kernel was built for the ${this.triangleTest} triangle test, but the scene uses ${scene.triangleTest}`);
        }
        const pipeline = batch.mode === 'closest' ? this.closestPipeline : this.anyPipeline;
        // The ray count is the size of the bound range, so no uniform is needed.
        const batchGroup = this.device.createBindGroup({
            label: `trace batch ${batch.mode}`,
            layout: pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: { buffer: batch.rays, offset: 0, size: batch.count * RAY_BYTES } },
                { binding: 1, resource: { buffer: batch.results } },
            ],
        });
        const workgroups = Math.ceil(batch.count / BATCH_WORKGROUP_SIZE);
        const x = Math.min(workgroups, this.device.limits.maxComputeWorkgroupsPerDimension);
        const y = Math.ceil(workgroups / x);
        const groups = this.groupsFor(scene);
        const pass = encoder.beginComputePass({ label: `trace batch ${batch.mode}` });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, batch.mode === 'closest' ? groups.closest : groups.any);
        pass.setBindGroup(1, batchGroup);
        pass.dispatchWorkgroups(x, y);
        pass.end();
    }
}

/** Floats per ray in a packed ray array: `origin (3), tMin, direction (3), tMax`. */
export const RAY_FLOATS = RAY_BYTES / 4;

/** Packs rays into the `GiRay` layout, ready for `writeBuffer`. */
export function packRays(rays: readonly TraceRay[]): Float32Array {
    const packed = new Float32Array(rays.length * RAY_FLOATS);
    rays.forEach((ray, index) => {
        const base = index * RAY_FLOATS;
        packed.set(ray.origin, base);
        packed[base + 3] = ray.tMin;
        packed.set(ray.direction, base + 4);
        packed[base + 7] = ray.tMax;
    });
    return packed;
}

export function unpackRay(packed: Float32Array, index: number): TraceRay {
    const base = index * RAY_FLOATS;
    return {
        origin: [packed[base], packed[base + 1], packed[base + 2]],
        tMin: packed[base + 3],
        direction: [packed[base + 4], packed[base + 5], packed[base + 6]],
        tMax: packed[base + 7],
    };
}
