/**
 * The hot movers (docs/design/m3-moving.md §5.1, "Probes that saw it last
 * update"): the instances that moved lately, each with a bit of the probes'
 * 32-bit mover mask and its current world box.
 *
 * `ddgi/blend` tests each blended ray's hit point against these boxes and
 * keeps the bits it hit in the probe's record (`moverMask`); when one of them
 * moves again, `ddgi/mark` adds every probe within the far radius whose mask
 * has its bit to the far set. That catches probes the geometric test misses
 * (a thin or odd-shaped object whose box centre the probe can't see).
 *
 * - An object joins the table when a region of the frame's changes says it
 *   moved (`DirtyRegion.moved`, with its slot), and keeps its entry for
 *   {@link MOVER_HOT_FRAMES} frames after its last move.
 * - Its first move finds no bit in any mask yet: the masks hold only hits on
 *   objects already in the table. From its second move on they do.
 * - With more than 32 movers, the one that moved longest ago gives up its
 *   bit. A bit handed to another object only over-marks (probes that saw the
 *   old owner), never misses.
 * - An object created, destroyed, shown, hidden or given a new mesh leaves
 *   the table: its box and slot may no longer mean the same object.
 */

import type { SceneChanges } from '@pmndrs/gi-core/platform/contract/scene-copy';
import { MOVER_BOX_WORD, MOVER_BOX_WORDS, MOVER_TABLE_CAPACITY, MOVER_TABLE_HEADER_WORDS, MOVER_TABLE_WORDS } from './layout.ts';

/** How long an object that stopped moving keeps its entry: 5 s at 60 Hz, so a door swung again soon after is still known. */
export const MOVER_HOT_FRAMES = 300;

interface MoverEntry {
    /** The instance's slot (`GiHit.instance`). */
    slot: number;
    /** Its current world box, min then max. */
    readonly box: Float32Array;
    lastMovedFrame: number;
}

export class MoverTable {
    /** Entry i owns bit i; null where no object holds it. */
    private readonly entries: (MoverEntry | null)[] = Array.from({ length: MOVER_TABLE_CAPACITY }, () => null);
    private readonly bySlot = new Map<number, number>();
    /** The table as `ddgi/blend` reads it (MOVER_TABLE_* in layout.ts). */
    readonly words = new Uint32Array(MOVER_TABLE_WORDS);
    private readonly floats = new Float32Array(this.words.buffer);
    /** Something changed since `words` was last read by the GPU upload. */
    dirty = true;

    /** How many objects hold an entry now. */
    get count(): number {
        return this.bySlot.size;
    }

    /** The mask bit of an instance, or 0 when it isn't a hot mover. */
    bitFor(slot: number): number {
        const index = this.bySlot.get(slot);
        return index === undefined ? 0 : bitOf(index);
    }

    /** Takes in a frame's changes (call before the mark table is built, so a mover's bit is known in the frame it moves). */
    update(changes: SceneChanges, frame: number): void {
        for (const region of changes.regions) {
            if (region.reason !== 'geometry' || region.instance === undefined) continue;
            if (region.moved === undefined) {
                this.remove(region.instance);
                continue;
            }
            // A transform set to what it was moves nothing.
            if (!(region.moved > 0)) continue;
            // A move logs where the object was and then where it is: the last region it has this frame is its box now.
            const entry = this.entryFor(region.instance, frame);
            entry.box.set(region.bounds.min, 0);
            entry.box.set(region.bounds.max, 3);
            entry.lastMovedFrame = frame;
            this.dirty = true;
        }
        for (let index = 0; index < this.entries.length; index++) {
            const entry = this.entries[index];
            if (entry && frame - entry.lastMovedFrame > MOVER_HOT_FRAMES) this.remove(entry.slot);
        }
        if (this.dirty) this.pack();
    }

    /** Forgets every entry (the technique's history was reset). */
    clear(): void {
        this.entries.fill(null);
        this.bySlot.clear();
        this.dirty = true;
        this.pack();
    }

    private entryFor(slot: number, frame: number): MoverEntry {
        const existing = this.bySlot.get(slot);
        if (existing !== undefined) return this.entries[existing]!;
        let index = this.entries.indexOf(null);
        if (index < 0) {
            // Full: the object that moved longest ago gives up its bit.
            index = 0;
            for (let candidate = 1; candidate < this.entries.length; candidate++) {
                if (this.entries[candidate]!.lastMovedFrame < this.entries[index]!.lastMovedFrame) index = candidate;
            }
            this.bySlot.delete(this.entries[index]!.slot);
        }
        const entry: MoverEntry = { slot, box: new Float32Array(6), lastMovedFrame: frame };
        this.entries[index] = entry;
        this.bySlot.set(slot, index);
        return entry;
    }

    private remove(slot: number): void {
        const index = this.bySlot.get(slot);
        if (index === undefined) return;
        this.entries[index] = null;
        this.bySlot.delete(slot);
        this.dirty = true;
    }

    /** Writes the live entries, packed from the start, into `words`. */
    private pack(): void {
        this.words.fill(0);
        let count = 0;
        for (let index = 0; index < this.entries.length; index++) {
            const entry = this.entries[index];
            if (!entry) continue;
            const base = MOVER_TABLE_HEADER_WORDS + count * MOVER_BOX_WORDS;
            this.floats.set(entry.box.subarray(0, 3), base + MOVER_BOX_WORD.min);
            this.words[base + MOVER_BOX_WORD.bit] = bitOf(index);
            this.floats.set(entry.box.subarray(3, 6), base + MOVER_BOX_WORD.max);
            count++;
        }
        this.words[0] = count;
    }
}

/** Entry i's mask bit (unsigned, so bit 31 stays positive). */
function bitOf(index: number): number {
    return (1 << index) >>> 0;
}
