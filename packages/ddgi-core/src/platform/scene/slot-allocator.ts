/**
 * Hands out record slots (table indices), lowest free slot first, so holes
 * left by destroyed objects are refilled before the table grows and the
 * slot range the GPU walks stays short.
 */
export class SlotAllocator {
    /** Free slots below {@link slotCount}, sorted ascending. */
    private readonly free: number[] = [];
    private end = 0;

    /** One more than the highest slot in use. */
    get slotCount(): number {
        return this.end;
    }

    take(): number {
        const reused = this.free.shift();
        if (reused !== undefined) return reused;
        return this.end++;
    }

    release(slot: number): void {
        if (slot < 0 || slot >= this.end) throw new RangeError(`slot ${slot} is not in use`);
        if (slot === this.end - 1) {
            // Shrink past the released slot and any free slots right below it.
            this.end--;
            while (this.free.length > 0 && this.free[this.free.length - 1] === this.end - 1) {
                this.free.pop();
                this.end--;
            }
            return;
        }
        let low = 0;
        let high = this.free.length;
        while (low < high) {
            const middle = (low + high) >> 1;
            if (this.free[middle] < slot) low = middle + 1;
            else high = middle;
        }
        if (this.free[low] === slot) throw new RangeError(`slot ${slot} released twice`);
        this.free.splice(low, 0, slot);
    }
}
