/**
 * First-fit allocation of ranges inside a buffer of fixed capacity, with
 * freed neighbours merged back together.
 *
 * Used for mesh data in the vertex buffer and for instance material lists in
 * the table buffer. Units are whatever the caller counts in (words here).
 * When nothing fits, `allocate` returns null and the caller grows the buffer
 * with {@link RangeAllocator.grow}; existing ranges keep their offsets, so
 * growing never moves data.
 */

interface FreeRange {
    offset: number;
    size: number;
}

export class RangeAllocator {
    /** Free ranges, sorted by offset, never touching each other. */
    private readonly free: FreeRange[] = [];
    private capacityUnits: number;
    private usedUnits = 0;

    constructor(capacity: number) {
        this.capacityUnits = capacity;
        if (capacity > 0) this.free.push({ offset: 0, size: capacity });
    }

    get capacity(): number {
        return this.capacityUnits;
    }

    /** Units handed out and not yet released. */
    get used(): number {
        return this.usedUnits;
    }

    /** End of the last allocated range: everything at or past it is free. */
    get highWaterMark(): number {
        const last = this.free[this.free.length - 1];
        if (last && last.offset + last.size === this.capacityUnits) return last.offset;
        return this.capacityUnits;
    }

    /** The offset of a new range of `size` units, or null if no free range is large enough. */
    allocate(size: number): number | null {
        if (!(size > 0) || !Number.isInteger(size)) throw new RangeError(`allocation size must be a positive integer, got ${size}`);
        const index = this.free.findIndex((range) => range.size >= size);
        if (index < 0) return null;
        const range = this.free[index];
        const offset = range.offset;
        if (range.size === size) {
            this.free.splice(index, 1);
        } else {
            range.offset += size;
            range.size -= size;
        }
        this.usedUnits += size;
        return offset;
    }

    /** Returns a range to the free list. The caller must pass exactly what `allocate` gave. */
    release(offset: number, size: number): void {
        if (offset < 0 || offset + size > this.capacityUnits) {
            throw new RangeError(`release of [${offset}, ${offset + size}) is outside the capacity ${this.capacityUnits}`);
        }
        // Insert in offset order, then merge with the neighbours it touches.
        let index = this.free.findIndex((range) => range.offset > offset);
        if (index < 0) index = this.free.length;
        const previous = this.free[index - 1];
        const next = this.free[index];
        if ((previous && previous.offset + previous.size > offset) || (next && offset + size > next.offset)) {
            throw new RangeError(`release of [${offset}, ${offset + size}) overlaps a free range: double release?`);
        }
        this.usedUnits -= size;
        const touchesPrevious = previous !== undefined && previous.offset + previous.size === offset;
        const touchesNext = next !== undefined && offset + size === next.offset;
        if (touchesPrevious && touchesNext) {
            previous.size += size + next.size;
            this.free.splice(index, 1);
        } else if (touchesPrevious) {
            previous.size += size;
        } else if (touchesNext) {
            next.offset = offset;
            next.size += size;
        } else {
            this.free.splice(index, 0, { offset, size });
        }
    }

    /** Extends the capacity; the new space joins the free range at the end. */
    grow(newCapacity: number): void {
        if (newCapacity < this.capacityUnits) throw new RangeError('a range allocator cannot shrink');
        const added = newCapacity - this.capacityUnits;
        if (added === 0) return;
        const last = this.free[this.free.length - 1];
        if (last && last.offset + last.size === this.capacityUnits) {
            last.size += added;
        } else {
            this.free.push({ offset: this.capacityUnits, size: added });
        }
        this.capacityUnits = newCapacity;
    }

    /** Capacity needed so that a range of `size` fits, if the allocator grew. */
    capacityToFit(size: number): number {
        return this.highWaterMark + size;
    }
}
