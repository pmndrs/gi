/**
 * Bookkeeping for the GPU timer's ring of timestamp slots. No GPU objects are
 * touched here, so the whole lifecycle can be unit-tested without a device.
 *
 * Why a ring: a frame's timestamps can only be read on the CPU a few frames
 * after the frame was submitted. While slot 3 is waiting for its readback,
 * the next frames must write into other slots, or they would overwrite results
 * that haven't been read yet.
 *
 * A slot moves through three states:
 *
 *     idle --acquire--> encoding --submit--> pending --release--> idle
 *
 * Each timed pass uses two consecutive query indices in the slot's query set:
 * `2 * n` for the start and `2 * n + 1` for the end of the n-th pass.
 */

export type TimestampSlotState = 'idle' | 'encoding' | 'pending';

export interface TimestampSlot {
    readonly index: number;
    state: TimestampSlotState;
    /** Pass names in the order their queries were handed out. */
    passNames: string[];
    /** The caller's frame number, echoed back with the results. */
    frame: number;
    /** Increases by one for every acquired slot, so results can be ordered. */
    sequence: number;
    /** The ring's epoch when the slot was acquired; see {@link TimestampRing.invalidate}. */
    epoch: number;
}

/** The two query indices a pass writes its start and end timestamps to. */
export interface PassQueryIndices {
    readonly beginIndex: number;
    readonly endIndex: number;
}

export class TimestampRing {
    readonly slots: readonly TimestampSlot[];
    readonly maxPassesPerSlot: number;

    private nextSlotToTry = 0;
    private nextSequence = 0;
    private epoch = 0;

    constructor(slotCount: number, maxPassesPerSlot: number) {
        if (!Number.isInteger(slotCount) || slotCount < 1) {
            throw new RangeError(`TimestampRing needs at least one slot (got ${slotCount}).`);
        }
        if (!Number.isInteger(maxPassesPerSlot) || maxPassesPerSlot < 1) {
            throw new RangeError(`TimestampRing needs room for at least one pass (got ${maxPassesPerSlot}).`);
        }
        this.maxPassesPerSlot = maxPassesPerSlot;
        this.slots = Array.from({ length: slotCount }, (_unused, index) => ({
            index,
            state: 'idle' as TimestampSlotState,
            passNames: [],
            frame: -1,
            sequence: -1,
            epoch: 0,
        }));
    }

    /** Number of timestamp queries one slot's query set needs. */
    get queriesPerSlot(): number {
        return this.maxPassesPerSlot * 2;
    }

    get idleCount(): number {
        return this.slots.filter((slot) => slot.state === 'idle').length;
    }

    get pendingCount(): number {
        return this.slots.filter((slot) => slot.state === 'pending').length;
    }

    /**
     * Takes the next idle slot for a new frame, or returns `undefined` when
     * every slot is still waiting for its readback. Dropping a frame's timing
     * is the right answer then: stalling the frame would cost far more than
     * one missing sample.
     */
    acquire(frame?: number): TimestampSlot | undefined {
        for (let offset = 0; offset < this.slots.length; offset++) {
            const slot = this.slots[(this.nextSlotToTry + offset) % this.slots.length];
            if (slot.state !== 'idle') continue;
            this.nextSlotToTry = (slot.index + 1) % this.slots.length;
            slot.state = 'encoding';
            slot.passNames = [];
            slot.sequence = this.nextSequence++;
            slot.frame = frame ?? slot.sequence;
            slot.epoch = this.epoch;
            return slot;
        }
        return undefined;
    }

    /** Hands out the query indices for one more pass, or `undefined` when the slot is full. */
    addPass(slot: TimestampSlot, name: string): PassQueryIndices | undefined {
        if (slot.state !== 'encoding') return undefined;
        if (slot.passNames.length >= this.maxPassesPerSlot) return undefined;
        const passNumber = slot.passNames.length;
        slot.passNames.push(name);
        return { beginIndex: passNumber * 2, endIndex: passNumber * 2 + 1 };
    }

    /**
     * Marks the slot as submitted. Returns how many queries were written (and
     * so must be resolved and read back), or 0 if no pass used the slot; then
     * it goes straight back to idle because there is nothing to wait for.
     */
    submit(slot: TimestampSlot): number {
        if (slot.state !== 'encoding') return 0;
        const queryCount = slot.passNames.length * 2;
        slot.state = queryCount === 0 ? 'idle' : 'pending';
        return queryCount;
    }

    /** The readback is finished (or abandoned); the slot can be used again. */
    release(slot: TimestampSlot): void {
        slot.state = 'idle';
    }

    /** True if the slot was acquired before the last {@link invalidate}, so its results are outdated. */
    isStale(slot: TimestampSlot): boolean {
        return slot.epoch !== this.epoch;
    }

    /**
     * Throws away work in flight: results from slots acquired earlier are
     * marked stale, and slots that were only being encoded become idle again.
     * Pending slots stay pending until their readback ends; they are reused
     * then.
     */
    invalidate(): void {
        this.epoch++;
        for (const slot of this.slots) {
            if (slot.state === 'encoding') slot.state = 'idle';
        }
    }
}
