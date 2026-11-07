/**
 * The scene copy's meshes packed on the GPU (T2.13): the raw upload, and the
 * `gi/pack-mesh` kernel that lays the words out (pack-mesh-wgsl.ts).
 *
 * **Why.** Packing on the CPU (octahedral normals, 16-bit indices, laying the
 * parts out) cost the main thread 0.2-0.4 s on Sponza and Bistro exterior at
 * every load. The GPU does it in a few milliseconds; the main thread only
 * copies the host's arrays once, with `queue.writeBuffer`, which it has to do
 * anyway to get them to the GPU.
 *
 * **When the copy happens.** At `createMesh` / `updateMesh`, so the host may
 * reuse its arrays straight after the call, as before (the contract), and so
 * the upload overlaps `prepare`'s compiles when a host copies its scene in
 * while preparing (DECISIONS 2026-11-08, T2.14's `preparing`). The staged
 * arrays wait in staging buffers until the scene's next `encode`, where the
 * kernel writes them into the vertex buffer.
 *
 * **The staging format** (one mesh after another in a staging buffer; all
 * offsets in 4-byte words, every part starting on a word):
 *
 *     indices        as the host gave them: a Uint16Array two to a word (low half first,
 *                    the last high half 0 when the count is odd), a Uint32Array one to a word
 *     positions      3 floats per vertex
 *     uvs (set 0)    2 floats per vertex, when given
 *     uvs (set 1)    2 floats per vertex, when given
 *     materialSlots  the packed words the CPU made (mesh-packing.ts), when stored
 *     normals        3 floats per vertex, when given (last: they are the one part that changes size)
 *
 * The first five parts are in the vertex buffer's order (layout.ts), so the
 * parts that are already packed become one or two copy jobs per mesh.
 *
 * **Staging buffers** are made as needed, growing from 4 MB by doubling to
 * 64 MB while a scene loads (a mesh larger than that gets a buffer of its
 * own size, up to the device's storage binding limit). A buffer
 * is destroyed at the `encode` after the one that packed its last mesh (the
 * host has submitted the packing by then). A mesh whose raw arrays are larger
 * than one staging buffer may be packed on the CPU instead (the fallback).
 */

import type { MeshDescriptor } from '../contract/scene.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import { NONE } from './layout.ts';
import type { MeshLayout } from './mesh-packing.ts';
import {
    PACK_HEADER_WORD,
    PACK_HEADER_WORDS,
    PACK_JOB_KIND,
    PACK_JOB_WORD,
    PACK_JOB_WORDS,
    PACK_MESH_ENTRY_POINT,
    PACK_MESH_WGSL,
    PACK_MESH_WORKGROUP_SIZE,
} from './pack-mesh-wgsl.ts';

/** The timed pass's name (`stats.passes`). */
export const PACK_MESH_PASS = 'gi/pack-mesh';

/** The first staging buffer of a load; later ones double, up to {@link LARGEST_STAGING_BYTES}. */
const FIRST_STAGING_BYTES = 4 * 1024 * 1024;

/**
 * Where the doubling stops (unless one mesh needs more, up to the binding
 * limit): a buffer is allocated whole however little of it a load fills, so
 * this bounds the waste. Sponza's 144 MB of raw arrays take 188 MB of staging.
 */
const LARGEST_STAGING_BYTES = 64 * 1024 * 1024;

/** A storage binding's offset must be a multiple of this on every device (`minStorageBufferOffsetAlignment` is at most 256). */
const BINDING_OFFSET_ALIGNMENT = 256;

/** Workgroups along x in one dispatch: the default `maxComputeWorkgroupsPerDimension`. */
const MAX_WORKGROUPS_PER_DIMENSION = 65535;

interface StagingBuffer {
    readonly buffer: GPUBuffer;
    readonly words: number;
    /** Words written so far. */
    used: number;
    /** Staged meshes in it that have not been packed yet. */
    waiting: number;
}

/** One part of a staged mesh, before the mesh has a place in the vertex buffer. */
interface StagedPart {
    readonly kind: number;
    /** Word offset in the staging buffer. */
    readonly source: number;
    /** Word offset from the mesh's first word in the vertex buffer. */
    readonly destination: number;
    readonly items: number;
    readonly indexCount: number;
}

/** A mesh's raw arrays in a staging buffer, and what the kernel must do with them. */
export interface StagedMesh {
    readonly staging: StagingBuffer;
    readonly parts: readonly StagedPart[];
    /** Bytes the staging copied. */
    readonly bytes: number;
}

/** A staged mesh with its place in the vertex buffer, ready to pack. */
export interface MeshToPack {
    readonly staged: StagedMesh;
    /** Word offset of the mesh in the vertex buffer. */
    readonly vertexOffset: number;
}

/** Raw words a mesh takes in a staging buffer (see the file comment). */
export function stagedWordCount(descriptor: MeshDescriptor, layout: MeshLayout): number {
    const indexWords = descriptor.indices instanceof Uint16Array ? Math.ceil(descriptor.indices.length / 2) : descriptor.indices.length;
    let words = indexWords + descriptor.positions.length;
    if (descriptor.uvs) words += descriptor.uvs.length;
    if (descriptor.uvs1) words += descriptor.uvs1.length;
    if (layout.offsets.materialSlots !== NONE) words += Math.ceil(layout.triangleCount / 2);
    if (descriptor.normals) words += descriptor.normals.length;
    return words;
}

/**
 * How a mesh's staged parts become its packed words: the jobs, with
 * destinations relative to the mesh's first word. Pure, so the unit tests
 * check it on the CPU (testing/pack-mesh-cpu.ts runs the jobs as the kernel does).
 */
export function stagedParts(descriptor: MeshDescriptor, layout: MeshLayout, first: number): StagedPart[] {
    const { offsets, vertexCount, triangleCount } = layout;
    const parts: StagedPart[] = [];
    const indexCount = triangleCount * 3;
    let cursor = first;

    const hostIndices16 = descriptor.indices instanceof Uint16Array;
    const indexWords = hostIndices16 ? Math.ceil(indexCount / 2) : indexCount;
    if (hostIndices16 === layout.indices16) {
        parts.push({ kind: PACK_JOB_KIND.copy, source: cursor, destination: offsets.indices, items: indexWords, indexCount });
    } else if (layout.indices16) {
        parts.push({ kind: PACK_JOB_KIND.narrow, source: cursor, destination: offsets.indices, items: Math.ceil(indexCount / 2), indexCount });
    } else {
        parts.push({ kind: PACK_JOB_KIND.widen, source: cursor, destination: offsets.indices, items: indexCount, indexCount });
    }
    cursor += indexWords;

    const copy = (destination: number, words: number): void => {
        parts.push({ kind: PACK_JOB_KIND.copy, source: cursor, destination, items: words, indexCount: 0 });
        cursor += words;
    };
    copy(offsets.positions, vertexCount * 3);
    if (offsets.uvs0 !== NONE) copy(offsets.uvs0, vertexCount * 2);
    if (offsets.uvs1 !== NONE) copy(offsets.uvs1, vertexCount * 2);
    if (offsets.materialSlots !== NONE) copy(offsets.materialSlots, Math.ceil(triangleCount / 2));
    if (offsets.normals !== NONE) {
        parts.push({ kind: PACK_JOB_KIND.normals, source: cursor, destination: offsets.normals, items: vertexCount, indexCount: 0 });
        cursor += vertexCount * 3;
    }
    return mergeCopies(parts);
}

/** Neighbouring copies that are contiguous on both sides become one job (fewer jobs to search). */
function mergeCopies(parts: StagedPart[]): StagedPart[] {
    const merged: StagedPart[] = [];
    for (const part of parts) {
        const previous = merged[merged.length - 1];
        if (
            previous &&
            previous.kind === PACK_JOB_KIND.copy &&
            part.kind === PACK_JOB_KIND.copy &&
            previous.source + previous.items === part.source &&
            previous.destination + previous.items === part.destination
        ) {
            merged[merged.length - 1] = { ...previous, items: previous.items + part.items };
        } else {
            merged.push(part);
        }
    }
    return merged;
}

/**
 * The job table of one dispatch: the header, then one record per job
 * (pack-mesh-wgsl.ts), destinations made absolute. Returns the words and
 * how many threads the jobs need.
 */
export function packJobTable(meshes: readonly MeshToPack[]): { readonly words: Uint32Array<ArrayBuffer>; readonly totalItems: number; readonly jobCount: number } {
    let jobCount = 0;
    for (const mesh of meshes) jobCount += mesh.staged.parts.length;
    const words = new Uint32Array(PACK_HEADER_WORDS + jobCount * PACK_JOB_WORDS);
    let job = 0;
    let firstItem = 0;
    for (const mesh of meshes) {
        for (const part of mesh.staged.parts) {
            const base = PACK_HEADER_WORDS + job * PACK_JOB_WORDS;
            words[base + PACK_JOB_WORD.kind] = part.kind;
            words[base + PACK_JOB_WORD.source] = part.source;
            words[base + PACK_JOB_WORD.destination] = mesh.vertexOffset + part.destination;
            words[base + PACK_JOB_WORD.items] = part.items;
            words[base + PACK_JOB_WORD.firstItem] = firstItem;
            words[base + PACK_JOB_WORD.indexCount] = part.indexCount;
            firstItem += part.items;
            job++;
        }
    }
    words[PACK_HEADER_WORD.jobCount] = jobCount;
    words[PACK_HEADER_WORD.totalItems] = firstItem;
    return { words, totalItems: firstItem, jobCount };
}

/** Workgroups along x and y for `items` threads, and the threads in one row (the kernel's `itemOf`). */
export function packDispatchSize(items: number): { readonly x: number; readonly y: number; readonly threadsPerRow: number } {
    const workgroups = Math.max(1, Math.ceil(items / PACK_MESH_WORKGROUP_SIZE));
    const x = Math.min(workgroups, MAX_WORKGROUPS_PER_DIMENSION);
    const y = Math.ceil(workgroups / x);
    return { x, y, threadsPerRow: x * PACK_MESH_WORKGROUP_SIZE };
}

export interface GpuMeshPackerOptions {
    readonly label: string;
}

/**
 * Stages meshes' raw arrays and packs them on the GPU. Owned by the
 * `PlatformScene`, which decides per mesh whether to use it (see
 * `PlatformScene.prepareGpuPacking`).
 */
export class GpuMeshPacker {
    private readonly device: GPUDevice;
    private readonly label: string;
    private readonly limitWords: number;
    private pipeline: GPUComputePipeline | null = null;
    private bindGroupLayout: GPUBindGroupLayout | null = null;
    private compiling: Promise<void> | null = null;

    /** The buffer new meshes are staged into, while it has room. */
    private open: StagingBuffer | null = null;
    /** Every staging buffer still alive. */
    private readonly buffers = new Set<StagingBuffer>();
    /** Bytes of the next staging buffer; doubles while meshes keep coming, back to the first size after an encode. */
    private nextBytes = FIRST_STAGING_BYTES;
    /** Buffers whose meshes were all packed in the last encode: destroyed in the next one, after the host's submit. */
    private retired: GPUBuffer[] = [];
    private jobBuffer: GPUBuffer | null = null;

    constructor(device: GPUDevice, options: GpuMeshPackerOptions) {
        this.device = device;
        this.label = options.label;
        const limits = device.limits;
        // A staging buffer is bound whole as storage, so it must fit one binding.
        this.limitWords = Math.floor(Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize) / 4);
    }

    /** Asks for the kernel's compile (once). */
    prepare(): Promise<void> {
        this.compiling ??= (async () => {
            const layout = this.device.createBindGroupLayout({
                label: `${this.label} ${PACK_MESH_PASS}`,
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                ],
            });
            const module = this.device.createShaderModule({ label: `${this.label} ${PACK_MESH_PASS}`, code: PACK_MESH_WGSL });
            const pipeline = await this.device.createComputePipelineAsync({
                label: `${this.label} ${PACK_MESH_PASS}`,
                layout: this.device.createPipelineLayout({ label: `${this.label} ${PACK_MESH_PASS}`, bindGroupLayouts: [layout] }),
                compute: { module, entryPoint: PACK_MESH_ENTRY_POINT },
            });
            this.bindGroupLayout = layout;
            this.pipeline = pipeline;
        })();
        return this.compiling;
    }

    get ready(): boolean {
        return this.pipeline !== null;
    }

    /** Bytes of staging buffers alive now (meshes waiting to be packed, or buffers waiting to be destroyed). */
    get stagingBytes(): number {
        let bytes = 0;
        for (const staging of this.buffers) bytes += staging.buffer.size;
        for (const buffer of this.retired) bytes += buffer.size;
        return bytes;
    }

    /** Whether a mesh's raw arrays fit one staging buffer (else it must be packed on the CPU). */
    fits(descriptor: MeshDescriptor, layout: MeshLayout): boolean {
        return stagedWordCount(descriptor, layout) <= this.limitWords;
    }

    /**
     * Copies a mesh's raw arrays into a staging buffer (`queue.writeBuffer`:
     * the only copy of them the main thread makes). Call only when
     * {@link fits} says it does.
     */
    stage(descriptor: MeshDescriptor, layout: MeshLayout, materialSlotWords: Uint32Array<ArrayBuffer> | null): StagedMesh {
        const words = stagedWordCount(descriptor, layout);
        const staging = this.stagingWithRoom(words);
        const first = staging.used;
        const queue = this.device.queue;
        let cursor = first;
        // The host's arrays go in as they are. Their type says ArrayBufferLike (they may sit on a
        // SharedArrayBuffer), which writeBuffer accepts (GPUAllowSharedBufferSource) but the DOM
        // typings narrow to ArrayBuffer, hence the casts.
        const write = (data: Float32Array | Uint32Array): void => {
            queue.writeBuffer(staging.buffer, cursor * 4, data as Uint32Array<ArrayBuffer>);
            cursor += data.length;
        };

        const indices = descriptor.indices;
        if (indices instanceof Uint16Array) {
            // writeBuffer takes whole words: the even part straight from the host's array,
            // an odd last index as a word of its own with a high half of 0, as the CPU packer leaves it.
            const even = indices.length & ~1;
            if (even > 0) queue.writeBuffer(staging.buffer, cursor * 4, indices as Uint16Array<ArrayBuffer>, 0, even);
            if (even < indices.length) queue.writeBuffer(staging.buffer, (cursor + even / 2) * 4, new Uint32Array([indices[even]]));
            cursor += Math.ceil(indices.length / 2);
        } else {
            write(indices);
        }
        write(descriptor.positions);
        if (descriptor.uvs) write(descriptor.uvs);
        if (descriptor.uvs1) write(descriptor.uvs1);
        if (materialSlotWords) write(materialSlotWords);
        if (descriptor.normals) write(descriptor.normals);

        staging.used = cursor;
        staging.waiting++;
        return { staging, parts: stagedParts(descriptor, layout, first), bytes: (cursor - first) * 4 };
    }

    /** A staged mesh that will never be packed (destroyed or replaced before its upload). */
    forget(staged: StagedMesh): void {
        staged.staging.waiting--;
    }

    /**
     * Records the packing of these meshes into the host's encoder: one
     * dispatch per staging buffer. Then retires the staging buffers no
     * waiting mesh needs any more. Throws if the kernel isn't compiled yet
     * (the platform only encodes after `prepare`).
     */
    encode(encoder: GPUCommandEncoder, vertexBuffer: GPUBuffer, meshes: readonly MeshToPack[], timer?: GpuTimerScope): void {
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
        // Meshes staged after this encode start a new buffer, small again.
        this.open = null;
        this.nextBytes = FIRST_STAGING_BYTES;
        if (meshes.length === 0) {
            this.retireUnused();
            return;
        }
        if (!this.pipeline || !this.bindGroupLayout) throw new Error('@pmndrs/ddgi: the mesh packing kernel is not compiled yet; await prepare() first');

        const byStaging = new Map<StagingBuffer, MeshToPack[]>();
        for (const mesh of meshes) {
            const list = byStaging.get(mesh.staged.staging);
            if (list) list.push(mesh);
            else byStaging.set(mesh.staged.staging, [mesh]);
        }

        // Every dispatch's job table in one buffer, each at a binding-aligned offset.
        const tables = [...byStaging].map(([staging, list]) => ({ staging, count: list.length, ...packJobTable(list) }));
        let tableBytes = 0;
        const tableOffsets = tables.map((table) => {
            const offset = tableBytes;
            tableBytes += Math.ceil(table.words.byteLength / BINDING_OFFSET_ALIGNMENT) * BINDING_OFFSET_ALIGNMENT;
            return offset;
        });
        const jobBuffer = this.jobBufferOfAtLeast(tableBytes);

        const pass = encoder.beginComputePass({ label: `${this.label} ${PACK_MESH_PASS}`, timestampWrites: timer?.computePass(PACK_MESH_PASS) });
        pass.setPipeline(this.pipeline);
        tables.forEach((table, index) => {
            const dispatch = packDispatchSize(table.totalItems);
            table.words[PACK_HEADER_WORD.threadsPerRow] = dispatch.threadsPerRow;
            this.device.queue.writeBuffer(jobBuffer, tableOffsets[index], table.words);
            const bindGroup = this.device.createBindGroup({
                label: `${this.label} ${PACK_MESH_PASS}`,
                layout: this.bindGroupLayout!,
                entries: [
                    { binding: 0, resource: { buffer: table.staging.buffer } },
                    { binding: 1, resource: { buffer: vertexBuffer } },
                    { binding: 2, resource: { buffer: jobBuffer, offset: tableOffsets[index], size: table.words.byteLength } },
                ],
            });
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(dispatch.x, dispatch.y);
            table.staging.waiting -= table.count;
        });
        pass.end();
        this.retireUnused();
    }

    dispose(): void {
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
        for (const staging of this.buffers) staging.buffer.destroy();
        this.buffers.clear();
        this.open = null;
        this.jobBuffer?.destroy();
        this.jobBuffer = null;
    }

    private retireUnused(): void {
        for (const staging of [...this.buffers]) {
            if (staging.waiting > 0 || staging === this.open) continue;
            this.buffers.delete(staging);
            this.retired.push(staging.buffer);
        }
    }

    private stagingWithRoom(words: number): StagingBuffer {
        if (this.open && this.open.words - this.open.used >= words) return this.open;
        const bytes = Math.min(this.limitWords * 4, Math.max(words * 4, this.nextBytes));
        this.nextBytes = Math.min(LARGEST_STAGING_BYTES, this.nextBytes * 2);
        const staging: StagingBuffer = {
            buffer: this.device.createBuffer({
                label: `${this.label} mesh staging`,
                size: Math.ceil(bytes / 4) * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
            words: Math.floor(bytes / 4),
            used: 0,
            waiting: 0,
        };
        this.buffers.add(staging);
        this.open = staging;
        return staging;
    }

    private jobBufferOfAtLeast(bytes: number): GPUBuffer {
        if (this.jobBuffer && this.jobBuffer.size >= bytes) return this.jobBuffer;
        // The old one may still be read by the last submit: destroy it after the next one.
        if (this.jobBuffer) this.retired.push(this.jobBuffer);
        this.jobBuffer = this.device.createBuffer({
            label: `${this.label} mesh packing jobs`,
            size: Math.max(BINDING_OFFSET_ALIGNMENT, Math.ceil(bytes / BINDING_OFFSET_ALIGNMENT) * BINDING_OFFSET_ALIGNMENT * 2),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        return this.jobBuffer;
    }
}
