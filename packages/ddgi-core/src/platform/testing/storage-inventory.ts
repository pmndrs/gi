/**
 * Counts the storage buffers every compute pipeline binds in its compute
 * stage, across all its bind groups: the number WebGPU checks against
 * `maxStorageBuffersPerShaderStage` when it makes the pipeline.
 *
 * Why it exists: Chrome on Apple GPUs allows 10 storage buffers per stage
 * (the M5 Pro), the Windows GPUs 16 and a device three.js makes itself 8
 * (docs/NOTES.md). A pipeline over 10 fails only on the Mac, so unit tests
 * make every pipeline on a fake device in Node and count them here.
 *
 * {@link recordComputePipelines} wraps a device: every
 * `createComputePipeline(Async)` is recorded with the storage buffers of its
 * layout. An explicit layout is counted from its bind group layout entries
 * (storage and read-only storage buffers visible to compute), each named
 * after the WGSL variable at that group and binding. An `'auto'` layout is
 * counted from the WGSL: the `var<storage>` declarations the entry point
 * reaches through the functions it calls, which is what WebGPU puts in an
 * automatic layout. Test-only: not exported from the package.
 */

import { createFakeGpu, type FakeGpu } from '../scene/testing/fake-gpu.ts';

/** `GPUShaderStage.COMPUTE`: the global does not exist in Node. */
const COMPUTE_STAGE = 0x4;

/** One storage buffer a pipeline binds. */
export interface InventoryBuffer {
    readonly group: number;
    readonly binding: number;
    /** The WGSL variable at that group and binding, or `group:binding` when the code declares none. */
    readonly name: string;
}

/** One compute pipeline and the storage buffers its compute stage binds. */
export interface InventoryPipeline {
    readonly label: string;
    readonly entryPoint: string;
    readonly storageBuffers: number;
    readonly buffers: readonly InventoryBuffer[];
    /** Bound buffers the entry point never reaches (its layout lists them for other passes, or a module brings them). */
    readonly unused: readonly string[];
}

/** A WGSL binding declaration: `@group(g) @binding(b) var<...> name`, the attributes in either order. */
const BINDING_DECLARATION = /@(group|binding)\((\d+)\)\s*@(group|binding)\((\d+)\)\s*var(<[^>]*>)?\s+(\w+)/g;

export interface WgslBinding {
    readonly group: number;
    readonly binding: number;
    readonly name: string;
    readonly storage: boolean;
}

/** Every binding the WGSL declares, and whether it is a storage buffer. */
export function wgslBindings(code: string): WgslBinding[] {
    const bindings: WgslBinding[] = [];
    for (const match of code.matchAll(BINDING_DECLARATION)) {
        const [, firstKind, firstValue, , secondValue, addressSpace, name] = match;
        const group = Number(firstKind === 'group' ? firstValue : secondValue);
        const binding = Number(firstKind === 'group' ? secondValue : firstValue);
        bindings.push({ group, binding, name, storage: /^<\s*storage\b/.test(addressSpace ?? '') });
    }
    return bindings;
}

/** The body of each `fn` in the WGSL, by name (comments removed first). */
function functionBodies(code: string): Map<string, string> {
    const source = code.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    const bodies = new Map<string, string>();
    for (const match of source.matchAll(/\bfn\s+(\w+)\s*\(/g)) {
        const open = source.indexOf('{', match.index);
        if (open < 0) continue;
        let depth = 0;
        let close = open;
        for (; close < source.length; close++) {
            if (source[close] === '{') depth++;
            else if (source[close] === '}' && --depth === 0) break;
        }
        bodies.set(match[1], source.slice(match.index, close + 1));
    }
    return bodies;
}

/** Identifiers the entry point uses, directly or through the functions it calls. */
function identifiersReachedFrom(code: string, entryPoint: string): Set<string> {
    const bodies = functionBodies(code);
    const reached = new Set<string>();
    const pending = [entryPoint];
    const visited = new Set<string>();
    while (pending.length > 0) {
        const name = pending.pop()!;
        if (visited.has(name)) continue;
        visited.add(name);
        const body = bodies.get(name);
        if (body === undefined) continue;
        for (const [identifier] of body.matchAll(/\b[A-Za-z_]\w*\b/g)) {
            reached.add(identifier);
            if (bodies.has(identifier) && !visited.has(identifier)) pending.push(identifier);
        }
    }
    return reached;
}

/** The fake layouts the device hands out keep their descriptors. */
interface DescribedLayout<T> {
    readonly descriptor: T;
}

/**
 * The storage buffers a pipeline's compute stage binds: from the explicit
 * layout's entries when it has one, from what the entry point reaches in the
 * WGSL for `'auto'`.
 */
export function pipelineStorageBuffers(layout: GPUPipelineLayout | 'auto', code: string, entryPoint: string): InventoryBuffer[] {
    const declared = wgslBindings(code);
    const nameAt = (group: number, binding: number): string =>
        declared.find((entry) => entry.group === group && entry.binding === binding)?.name ?? `${group}:${binding}`;
    if (layout === 'auto') {
        const reached = identifiersReachedFrom(code, entryPoint);
        return declared.filter((entry) => entry.storage && reached.has(entry.name)).map(({ group, binding, name }) => ({ group, binding, name }));
    }
    const groups = [...(layout as unknown as DescribedLayout<GPUPipelineLayoutDescriptor>).descriptor.bindGroupLayouts];
    const buffers: InventoryBuffer[] = [];
    groups.forEach((groupLayout, group) => {
        if (!groupLayout) return;
        for (const entry of (groupLayout as unknown as DescribedLayout<GPUBindGroupLayoutDescriptor>).descriptor.entries) {
            const type = entry.buffer ? (entry.buffer.type ?? 'uniform') : null;
            const isStorage = type === 'storage' || type === 'read-only-storage';
            if (isStorage && (entry.visibility & COMPUTE_STAGE) !== 0) buffers.push({ group, binding: entry.binding, name: nameAt(group, entry.binding) });
        }
    });
    return buffers;
}

/**
 * Wraps `device` so every compute pipeline it makes is recorded in the
 * returned list. The device's `createBindGroupLayout` and
 * `createPipelineLayout` must keep their descriptors (the fake GPU's do).
 */
export function recordComputePipelines(device: GPUDevice): InventoryPipeline[] {
    const pipelines: InventoryPipeline[] = [];
    const codeOf = new WeakMap<object, string>();
    const target = device as unknown as Record<string, unknown>;
    const createShaderModule = (target.createShaderModule as (descriptor: GPUShaderModuleDescriptor) => object).bind(device);
    target.createShaderModule = (descriptor: GPUShaderModuleDescriptor) => {
        // The fake GPU hands out plain objects, so each module gets its own wrapper to key on.
        // It reports no messages, as a module that compiled does (the neural tail's network checks).
        const module = { getCompilationInfo: () => Promise.resolve({ messages: [] }), ...(createShaderModule(descriptor) ?? {}) };
        codeOf.set(module, descriptor.code);
        return module;
    };
    const record = (descriptor: GPUComputePipelineDescriptor): void => {
        const code = codeOf.get(descriptor.compute.module as unknown as object) ?? '';
        const entryPoint = descriptor.compute.entryPoint ?? '';
        const buffers = pipelineStorageBuffers(descriptor.layout, code, entryPoint);
        const reached = new Set(pipelineStorageBuffers('auto', code, entryPoint).map((buffer) => buffer.name));
        const unused = buffers.filter((buffer) => !reached.has(buffer.name)).map((buffer) => buffer.name);
        pipelines.push({ label: descriptor.label ?? '', entryPoint, storageBuffers: buffers.length, buffers, unused });
    };
    for (const name of ['createComputePipeline', 'createComputePipelineAsync'] as const) {
        const original = (target[name] as (descriptor: GPUComputePipelineDescriptor) => unknown).bind(device);
        target[name] = (descriptor: GPUComputePipelineDescriptor) => {
            record(descriptor);
            // An 'auto' layout's groups are what a caller binds (the neural tail's network does).
            const withGroups = (pipeline: unknown): unknown => Object.assign(pipeline as object, { getBindGroupLayout: (index: number) => ({ label: `${descriptor.label ?? ''} group ${index}` }) });
            const pipeline = original(descriptor);
            return pipeline instanceof Promise ? pipeline.then(withGroups) : withGroups(pipeline);
        };
    }
    return pipelines;
}

function fakeTexture(descriptor: GPUTextureDescriptor): unknown {
    const size = descriptor.size as GPUExtent3DDict;
    return {
        label: descriptor.label ?? '',
        width: size.width,
        height: size.height ?? 1,
        depthOrArrayLayers: size.depthOrArrayLayers ?? 1,
        mipLevelCount: descriptor.mipLevelCount ?? 1,
        format: descriptor.format,
        usage: descriptor.usage,
        dimension: descriptor.dimension ?? '2d',
        createView: () => ({}),
        destroy() {},
    };
}

/** An encoder, pass or command buffer whose every method does nothing and returns another such. */
const recordsNothing: unknown = new Proxy(() => {}, {
    get: (_target, key) => (key === 'then' ? undefined : recordsNothing),
    apply: () => recordsNothing,
});

/** Limits of the inventory device: our GPUs' storage limits, with room for any count of storage buffers. */
const INVENTORY_LIMITS: Partial<GPUSupportedLimits> = {
    maxStorageBuffersPerShaderStage: 64,
    maxStorageBufferBindingSize: 2 ** 31 - 4,
    maxBufferSize: 2 ** 31,
    maxBindGroups: 4,
    maxComputeWorkgroupsPerDimension: 65535,
    maxComputeWorkgroupStorageSize: 32768,
    maxTextureDimension2D: 8192,
    maxTextureArrayLayers: 2048,
};

/**
 * A fake device that can prepare whole techniques without a GPU: buffers
 * hold no bytes (the hash grid alone is 1 GiB), the queue does nothing, and
 * the limits are our GPUs' with any number of storage buffers allowed, so
 * every pipeline is made and counted. `features` are what the device
 * reports: with and without `subgroups`, the kernels take their two paths.
 */
export function createInventoryGpu(features: readonly string[] = ['subgroups', 'shader-f16', 'timestamp-query']): FakeGpu {
    const fake = createFakeGpu(INVENTORY_LIMITS);
    const device = fake.device as unknown as Record<string, unknown>;
    const events = new EventTarget();
    Object.assign(device, {
        addEventListener: events.addEventListener.bind(events),
        removeEventListener: events.removeEventListener.bind(events),
        dispatchEvent: events.dispatchEvent.bind(events),
        lost: new Promise(() => {}),
        features: new Set<string>(features),
        createTexture: fakeTexture,
        createSampler: () => ({}),
        createQuerySet: () => ({ destroy() {} }),
        pushErrorScope: () => {},
        popErrorScope: () => Promise.resolve(null),
        createBuffer: (descriptor: GPUBufferDescriptor) => ({
            label: descriptor.label ?? '',
            size: descriptor.size,
            usage: descriptor.usage,
            destroy() {},
            unmap() {},
            mapAsync: () => Promise.resolve(),
            getMappedRange: (offset = 0, size?: number) => new ArrayBuffer(size ?? descriptor.size - offset),
        }),
        // Commands recorded while preparing (the neural tail's network fills its weights) do nothing.
        createCommandEncoder: () => recordsNothing,
        queue: { writeBuffer() {}, writeTexture() {}, submit() {}, onSubmittedWorkDone: () => Promise.resolve() },
    });
    Object.assign(device.limits as object, INVENTORY_LIMITS);
    return fake;
}

/** The pipeline that binds the most storage buffers (the first such), or null for none. */
export function worstPipeline(pipelines: readonly InventoryPipeline[]): InventoryPipeline | null {
    return pipelines.reduce<InventoryPipeline | null>((worst, pipeline) => (worst === null || pipeline.storageBuffers > worst.storageBuffers ? pipeline : worst), null);
}

/** One line per pipeline over `limit`: count, label, entry point and its buffers. */
export function describeOver(pipelines: readonly InventoryPipeline[], limit: number): string[] {
    return pipelines
        .filter((pipeline) => pipeline.storageBuffers > limit)
        .map((pipeline) => {
            const unused = pipeline.unused.length > 0 ? ` (unused: ${pipeline.unused.join(', ')})` : '';
            return `${pipeline.storageBuffers}  ${pipeline.label} [${pipeline.entryPoint}]: ${pipeline.buffers.map((buffer) => buffer.name).join(', ')}${unused}`;
        });
}
