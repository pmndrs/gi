/**
 * A GPU storage buffer holding variable-sized ranges (one per mesh tree) that
 * grows by doubling, for the tracer's node and triangle buffers.
 *
 * Growing makes a new buffer and records copies of the ranges still in use
 * into the host's encoder. Ranges keep their offsets, so a mesh's place never
 * changes and nothing that points at it needs rewriting. The old buffer is
 * kept until the next {@link GrowableStorage.releaseRetired} (the next frame),
 * because this frame's commands still copy out of it.
 *
 * Order on the queue matters here (NOTES, "Platform scene copy"): a
 * `queue.writeBuffer` made while encoding lands before the host submits, and
 * the copies recorded in the encoder run after it. So the growth copies only
 * cover ranges that were already resident, never one being written this frame.
 */

import { GILimitError } from '../contract/errors.ts';
import { RangeAllocator } from '../scene/range-allocator.ts';

export interface ResidentRange {
    readonly offset: number;
    readonly count: number;
}

export class GrowableStorage {
    private readonly device: GPUDevice;
    private readonly label: string;
    /** Bytes per unit (a node, a triangle). */
    private readonly unitBytes: number;
    private current: GPUBuffer;
    private readonly allocator: RangeAllocator;
    private retired: GPUBuffer[] = [];
    /** Share of room kept free when a growth is sized by what must fit (see the constructor). */
    private readonly headroom: number;

    /**
     * `headroom` (default 0): when the buffer grows by more than doubling (a
     * scene's trees arriving at once), it is made this share larger than
     * what must fit, so ranges that are replaced by slightly larger ones
     * soon after still fit without doubling it.
     */
    constructor(device: GPUDevice, label: string, unitBytes: number, initialUnits: number, options: { readonly headroom?: number } = {}) {
        this.device = device;
        this.label = label;
        this.unitBytes = unitBytes;
        this.headroom = options.headroom ?? 0;
        this.allocator = new RangeAllocator(initialUnits);
        this.current = this.createBuffer(initialUnits);
    }

    get buffer(): GPUBuffer {
        return this.current;
    }

    /** Bytes of the GPU buffer (allocated, not only used). */
    get byteLength(): number {
        return this.current.size;
    }

    /** Bytes holding live ranges. */
    get usedBytes(): number {
        return this.allocator.used * this.unitBytes;
    }

    allocate(count: number): number | null {
        return this.allocator.allocate(count);
    }

    release(offset: number, count: number): void {
        this.allocator.release(offset, count);
    }

    /**
     * Grows so that `extraUnits` more units fit after the current high-water
     * mark, copying `resident` ranges into the new buffer. Throws a
     * `GILimitError` naming the limit when one storage binding can't hold it.
     */
    grow(encoder: GPUCommandEncoder, extraUnits: number, resident: readonly ResidentRange[]): void {
        const limits = this.device.limits;
        const limitBytes = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
        const limitUnits = Math.floor(limitBytes / this.unitBytes);
        const minimumUnits = this.allocator.capacityToFit(extraUnits);
        if (minimumUnits > limitUnits) {
            const limit = limits.maxStorageBufferBindingSize <= limits.maxBufferSize ? 'maxStorageBufferBindingSize' : 'maxBufferSize';
            throw new GILimitError(limit, minimumUnits * this.unitBytes, limitBytes);
        }
        // Doubling keeps reallocations logarithmic while a scene loads; a bigger jump gets its headroom.
        const units = Math.min(limitUnits, Math.max(Math.ceil(minimumUnits * (1 + this.headroom)), this.allocator.capacity * 2));
        const buffer = this.createBuffer(units);

        // Neighbouring ranges are copied as one.
        const sorted = [...resident].sort((a, b) => a.offset - b.offset);
        let runStart = -1;
        let runEnd = -1;
        const flush = (): void => {
            if (runStart < 0) return;
            encoder.copyBufferToBuffer(this.current, runStart * this.unitBytes, buffer, runStart * this.unitBytes, (runEnd - runStart) * this.unitBytes);
        };
        for (const range of sorted) {
            if (range.offset === runEnd) {
                runEnd += range.count;
            } else {
                flush();
                runStart = range.offset;
                runEnd = range.offset + range.count;
            }
        }
        flush();

        this.retired.push(this.current);
        this.current = buffer;
        this.allocator.grow(units);
    }

    /** Writes `data` at a range this storage handed out. */
    write(offset: number, data: ArrayBuffer): void {
        this.device.queue.writeBuffer(this.current, offset * this.unitBytes, data);
    }

    /** Destroys buffers replaced by an earlier `grow`. Call once the frame that copied out of them was submitted. */
    releaseRetired(): void {
        for (const buffer of this.retired) buffer.destroy();
        this.retired = [];
    }

    destroy(): void {
        this.releaseRetired();
        this.current.destroy();
    }

    private createBuffer(units: number): GPUBuffer {
        return this.device.createBuffer({
            label: this.label,
            // A binding can't be empty, so an empty storage still holds one unit.
            size: Math.max(1, units) * this.unitBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
    }
}
