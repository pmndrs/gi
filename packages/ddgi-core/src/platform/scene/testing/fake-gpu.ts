/**
 * A stand-in GPUDevice for unit tests in Node, just enough for the scene copy:
 * buffers with real bytes, `queue.writeBuffer`, and command encoders whose
 * buffer copies run when the test "submits" them.
 *
 * The order matches WebGPU's queue timeline: `writeBuffer` lands at once
 * (it is queued ahead of any later submit), while copies recorded in an
 * encoder happen at `submit`. It also counts shader modules and pipelines,
 * so a test can show that scene changes never create any. Test-only: not
 * exported from the package.
 */

/** WebGPU's usage flags, which Node does not define. */
const BUFFER_USAGE = {
    MAP_READ: 0x0001,
    MAP_WRITE: 0x0002,
    COPY_SRC: 0x0004,
    COPY_DST: 0x0008,
    INDEX: 0x0010,
    VERTEX: 0x0020,
    UNIFORM: 0x0040,
    STORAGE: 0x0080,
    INDIRECT: 0x0100,
    QUERY_RESOLVE: 0x0200,
};

export function installWebGpuGlobals(): void {
    const scope = globalThis as unknown as Record<string, unknown>;
    scope.GPUBufferUsage ??= BUFFER_USAGE;
    scope.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
}

export class FakeBuffer {
    readonly label: string;
    readonly size: number;
    readonly usage: number;
    readonly bytes: Uint8Array;
    destroyed = false;

    constructor(descriptor: GPUBufferDescriptor) {
        this.label = descriptor.label ?? '';
        this.size = descriptor.size;
        this.usage = descriptor.usage;
        this.bytes = new Uint8Array(descriptor.size);
    }

    destroy(): void {
        this.destroyed = true;
    }

    get words(): Uint32Array {
        return new Uint32Array(this.bytes.buffer);
    }
}

interface RecordedCopy {
    source: FakeBuffer;
    sourceOffset: number;
    destination: FakeBuffer;
    destinationOffset: number;
    size: number;
}

/** A bound buffer range, as a kernel stand-in sees it. */
export interface FakeBinding {
    readonly buffer: FakeBuffer;
    readonly offset: number;
    readonly size: number;
}

/**
 * A CPU stand-in for a compute kernel, keyed by entry point: called at
 * submit with the dispatch's bindings (group 0, by binding number) and its
 * workgroup counts.
 */
export type FakeKernel = (bindings: ReadonlyMap<number, FakeBinding>, workgroupsX: number, workgroupsY: number) => void;

interface RecordedDispatch {
    entryPoint: string;
    bindings: ReadonlyMap<number, FakeBinding>;
    workgroupsX: number;
    workgroupsY: number;
}

type RecordedCommand = { copy: RecordedCopy } | { dispatch: RecordedDispatch };

interface FakePipeline {
    readonly entryPoint: string;
}

interface FakeBindGroup {
    readonly entries: readonly GPUBindGroupEntry[];
}

export class FakeEncoder {
    /** Copies and dispatches in recording order: submit runs them in this order. */
    readonly commands: RecordedCommand[] = [];

    get copies(): RecordedCopy[] {
        return this.commands.flatMap((command) => ('copy' in command ? [command.copy] : []));
    }

    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void {
        if (sourceOffset % 4 !== 0 || destinationOffset % 4 !== 0 || size % 4 !== 0) throw new Error('copy offsets and size must be multiples of 4');
        this.commands.push({
            copy: {
                source: source as unknown as FakeBuffer,
                sourceOffset,
                destination: destination as unknown as FakeBuffer,
                destinationOffset,
                size,
            },
        });
    }

    /** Records dispatches; bind group 0's buffers are what a kernel stand-in gets. */
    beginComputePass(): GPUComputePassEncoder {
        let pipeline: FakePipeline | null = null;
        let group: FakeBindGroup | null = null;
        const pass = {
            setPipeline: (value: GPUComputePipeline) => {
                pipeline = value as unknown as FakePipeline;
            },
            setBindGroup: (index: number, value: GPUBindGroup) => {
                if (index === 0) group = value as unknown as FakeBindGroup;
            },
            dispatchWorkgroups: (workgroupsX: number, workgroupsY = 1) => {
                if (!pipeline || !group) throw new Error('dispatch without a pipeline and bind group 0');
                const bindings = new Map<number, FakeBinding>();
                for (const entry of group.entries) {
                    const resource = entry.resource as GPUBufferBinding;
                    const buffer = resource.buffer as unknown as FakeBuffer;
                    const offset = resource.offset ?? 0;
                    bindings.set(entry.binding, { buffer, offset, size: resource.size ?? buffer.size - offset });
                }
                this.commands.push({ dispatch: { entryPoint: pipeline.entryPoint, bindings, workgroupsX, workgroupsY } });
            },
            end: () => {},
        };
        return pass as unknown as GPUComputePassEncoder;
    }
}

export interface FakeGpu {
    readonly device: GPUDevice;
    readonly buffers: FakeBuffer[];
    readonly counters: { shaderModules: number; pipelines: number; bindGroups: number };
    createEncoder(): GPUCommandEncoder;
    /** Runs the encoder's copies and dispatches, failing like WebGPU would if a buffer it uses was destroyed. */
    submit(encoder: GPUCommandEncoder): void;
}

export interface FakeGpuOptions {
    /** Kernels to run on the CPU at submit, by entry point. A dispatch of any other kernel does nothing. */
    readonly kernels?: Readonly<Record<string, FakeKernel>>;
}

export function createFakeGpu(limits: Partial<GPUSupportedLimits> = {}, options: FakeGpuOptions = {}): FakeGpu {
    installWebGpuGlobals();
    const buffers: FakeBuffer[] = [];
    const counters = { shaderModules: 0, pipelines: 0, bindGroups: 0 };
    const fullLimits = {
        maxStorageBufferBindingSize: 128 * 1024 * 1024,
        maxBufferSize: 256 * 1024 * 1024,
        maxStorageBuffersPerShaderStage: 8,
        // Core defaults, which the platform's kernels need (KERNEL_COMPUTE_LIMITS).
        maxComputeInvocationsPerWorkgroup: 256,
        maxComputeWorkgroupSizeX: 256,
        ...limits,
    };
    const device = {
        limits: fullLimits,
        createBuffer(descriptor: GPUBufferDescriptor) {
            if (descriptor.size > fullLimits.maxBufferSize) throw new Error(`buffer of ${descriptor.size} bytes is over maxBufferSize`);
            const buffer = new FakeBuffer(descriptor);
            buffers.push(buffer);
            return buffer;
        },
        createShaderModule() {
            counters.shaderModules++;
            return {};
        },
        createComputePipeline(descriptor: GPUComputePipelineDescriptor) {
            counters.pipelines++;
            return { entryPoint: descriptor.compute.entryPoint ?? '' } satisfies FakePipeline;
        },
        createComputePipelineAsync(descriptor: GPUComputePipelineDescriptor) {
            counters.pipelines++;
            return Promise.resolve({ entryPoint: descriptor.compute.entryPoint ?? '' } satisfies FakePipeline);
        },
        createBindGroupLayout(descriptor: GPUBindGroupLayoutDescriptor) {
            return { descriptor };
        },
        createPipelineLayout(descriptor: GPUPipelineLayoutDescriptor) {
            return { descriptor };
        },
        createBindGroup(descriptor: GPUBindGroupDescriptor) {
            counters.bindGroups++;
            return { entries: [...descriptor.entries] } satisfies FakeBindGroup;
        },
        queue: {
            writeBuffer(buffer: GPUBuffer, bufferOffset: number, data: ArrayBufferView | ArrayBuffer, dataOffset = 0, size?: number) {
                const target = buffer as unknown as FakeBuffer;
                if (target.destroyed) throw new Error(`writeBuffer into destroyed buffer "${target.label}"`);
                let source: Uint8Array;
                if (ArrayBuffer.isView(data)) {
                    // Typed arrays: offset and size count elements.
                    const elementSize = (data as unknown as { BYTES_PER_ELEMENT?: number }).BYTES_PER_ELEMENT ?? 1;
                    const elements = size ?? data.byteLength / elementSize - dataOffset;
                    source = new Uint8Array(data.buffer, data.byteOffset + dataOffset * elementSize, elements * elementSize);
                } else {
                    source = new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
                }
                if (bufferOffset + source.byteLength > target.size) throw new Error(`writeBuffer past the end of "${target.label}"`);
                target.bytes.set(source, bufferOffset);
            },
        },
    };
    return {
        device: device as unknown as GPUDevice,
        buffers,
        counters,
        createEncoder: () => new FakeEncoder() as unknown as GPUCommandEncoder,
        submit(encoder) {
            for (const command of (encoder as unknown as FakeEncoder).commands) {
                if ('dispatch' in command) {
                    const { entryPoint, bindings, workgroupsX, workgroupsY } = command.dispatch;
                    for (const binding of bindings.values()) if (binding.buffer.destroyed) throw new Error('submit uses a destroyed buffer');
                    options.kernels?.[entryPoint]?.(bindings, workgroupsX, workgroupsY);
                    continue;
                }
                const copy = command.copy;
                if (copy.source.destroyed || copy.destination.destroyed) throw new Error('submit uses a destroyed buffer');
                if (copy.sourceOffset + copy.size > copy.source.size || copy.destinationOffset + copy.size > copy.destination.size) {
                    throw new Error('copy out of range');
                }
                copy.destination.bytes.set(copy.source.bytes.subarray(copy.sourceOffset, copy.sourceOffset + copy.size), copy.destinationOffset);
            }
        },
    };
}
