import { describe, expect, it } from 'vitest';
import type { DirtyRegion, SceneChanges } from '../../platform/contract/scene-copy.ts';
import { MOVER_BOX_WORD, MOVER_BOX_WORDS, MOVER_TABLE_CAPACITY, MOVER_TABLE_HEADER_WORDS } from './layout.ts';
import { MOVER_HOT_FRAMES, MoverTable } from './mover-table.ts';

const NONE: SceneChanges = { regions: [], global: [], receivers: [], lights: [], lightsOverflowed: false };

function move(instance: number, x: number, moved = 0.1): DirtyRegion[] {
    // A move logs where it was, then where it is.
    return [
        { bounds: { min: [x - 1, 0, 0], max: [x, 1, 1] }, reason: 'geometry', moved, instance },
        { bounds: { min: [x, 0, 0], max: [x + 1, 1, 1] }, reason: 'geometry', moved, instance },
    ];
}

function packedBoxes(table: MoverTable): { min: number[]; max: number[]; bit: number }[] {
    const floats = new Float32Array(table.words.buffer);
    return Array.from({ length: table.words[0] }, (_, entry) => {
        const base = MOVER_TABLE_HEADER_WORDS + entry * MOVER_BOX_WORDS;
        return {
            min: [...floats.subarray(base + MOVER_BOX_WORD.min, base + MOVER_BOX_WORD.min + 3)],
            max: [...floats.subarray(base + MOVER_BOX_WORD.max, base + MOVER_BOX_WORD.max + 3)],
            bit: table.words[base + MOVER_BOX_WORD.bit],
        };
    });
}

describe('the mover table', () => {
    it('gives a moved object a bit and its newest box, and forgets it after MOVER_HOT_FRAMES still frames', () => {
        const table = new MoverTable();
        table.update({ ...NONE, regions: move(4, 2) }, 1);
        expect(table.bitFor(4)).toBe(1);
        expect(table.bitFor(5)).toBe(0);
        expect(packedBoxes(table)).toEqual([{ min: [2, 0, 0], max: [3, 1, 1], bit: 1 }]);
        table.update({ ...NONE, regions: move(5, 6) }, 2);
        expect(table.bitFor(5)).toBe(2);
        table.update(NONE, 1 + MOVER_HOT_FRAMES);
        expect(table.count).toBe(2);
        table.update(NONE, 2 + MOVER_HOT_FRAMES);
        expect(table.bitFor(4)).toBe(0);
        expect(table.bitFor(5)).toBe(2);
        expect(table.words[0]).toBe(1);
    });

    it('leaves out moves of nothing, and drops an object that was added, removed, shown or hidden', () => {
        const table = new MoverTable();
        table.update({ ...NONE, regions: move(4, 2, 0) }, 1);
        expect(table.count).toBe(0);
        table.update({ ...NONE, regions: move(4, 2) }, 2);
        table.update({ ...NONE, regions: [{ bounds: { min: [0, 0, 0], max: [1, 1, 1] }, reason: 'geometry', instance: 4 }] }, 3);
        expect(table.bitFor(4)).toBe(0);
    });

    it('hands the bit of the object that moved longest ago to a new one when full', () => {
        const table = new MoverTable();
        for (let slot = 0; slot < MOVER_TABLE_CAPACITY; slot++) table.update({ ...NONE, regions: move(slot, slot) }, 10 + slot);
        expect(table.count).toBe(MOVER_TABLE_CAPACITY);
        // Bit 31 stays a positive number.
        expect(table.bitFor(MOVER_TABLE_CAPACITY - 1)).toBe(2 ** 31);
        table.update({ ...NONE, regions: move(100, 50) }, 100);
        expect(table.bitFor(0)).toBe(0);
        expect(table.bitFor(100)).toBe(1);
        expect(table.count).toBe(MOVER_TABLE_CAPACITY);
    });
});
