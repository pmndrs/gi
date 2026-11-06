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

export class FakeEncoder {
    readonly copies: RecordedCopy[] = [];

    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number): void {
        if (sourceOffset % 4 !== 0 || destinationOffset % 4 !== 0 || size % 4 !== 0) throw new Error('copy offsets and size must be multiples of 4');
        this.copies.push({
            source: source as unknown as FakeBuffer,
            sourceOffset,
            destination: destination as unknown as FakeBuffer,
            destinationOffset,
            size,
        });
    }
}

export interface FakeGpu {
    readonly device: GPUDevice;
    readonly buffers: FakeBuffer[];
    readonly counters: { shaderModules: number; pipelines: number; bindGroups: number };
    createEncoder(): GPUCommandEncoder;
    /** Runs the encoder's copies, failing like WebGPU would if a buffer it uses was destroyed. */
    submit(encoder: GPUCommandEncoder): void;
}

export function createFakeGpu(limits: Partial<GPUSupportedLimits> = {}): FakeGpu {
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
        createComputePipeline() {
            counters.pipelines++;
            return {};
        },
        createComputePipelineAsync() {
            counters.pipelines++;
            return Promise.resolve({});
        },
        createBindGroup() {
            counters.bindGroups++;
            return {};
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
            for (const copy of (encoder as unknown as FakeEncoder).copies) {
                if (copy.source.destroyed || copy.destination.destroyed) throw new Error('submit uses a destroyed buffer');
                if (copy.sourceOffset + copy.size > copy.source.size || copy.destinationOffset + copy.size > copy.destination.size) {
                    throw new Error('copy out of range');
                }
                copy.destination.bytes.set(copy.source.bytes.subarray(copy.sourceOffset, copy.sourceOffset + copy.size), copy.destinationOffset);
            }
        },
    };
}
