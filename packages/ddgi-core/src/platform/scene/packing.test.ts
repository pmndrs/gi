/** The building blocks of the scene copy: normal encoding, mesh packing and the two allocators. */

import { describe, expect, it } from 'vitest';
import { createRandom, randomUnitVector } from '../trace/oracle-fixtures.ts';
import { NONE } from './layout.ts';
import { packedIndices, packedPositions, packedTriangleVertices, packMesh } from './mesh-packing.ts';
import { decodeOctahedral, encodeOctahedral } from './octahedral.ts';
import { RangeAllocator } from './range-allocator.ts';
import { SlotAllocator } from './slot-allocator.ts';

describe('octahedral normals', () => {
    it('round-trip any direction within 1e-4 radians', () => {
        const random = createRandom(1);
        let worst = 0;
        for (let sample = 0; sample < 100000; sample++) {
            const n = randomUnitVector(random);
            const decoded = decodeOctahedral(encodeOctahedral(n[0], n[1], n[2]));
            const cosine = Math.min(1, n[0] * decoded[0] + n[1] * decoded[1] + n[2] * decoded[2]);
            worst = Math.max(worst, Math.acos(cosine));
        }
        expect(worst).toBeLessThan(1e-4);
    });

    it('keeps the axes exact and turns a zero vector into +Z', () => {
        for (const axis of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) {
            const decoded = decodeOctahedral(encodeOctahedral(axis[0], axis[1], axis[2]));
            decoded.forEach((value, index) => expect(value).toBeCloseTo(axis[index], 6));
        }
        expect(decodeOctahedral(encodeOctahedral(0, 0, 0))).toEqual([0, 0, 1]);
    });
});

describe('mesh packing', () => {
    const quad = {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0.5, 0.5, 1]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3, 2, 4, 3]),
    };

    it('stores small meshes with 16-bit indices and reads every triangle back', () => {
        const packed = packMesh(quad);
        expect(packed.indices16).toBe(true);
        expect(packed.offsets.indices).toBe(0);
        expect(packed.offsets.positions).toBe(5); // ceil(9 / 2) words
        for (let triangle = 0; triangle < 3; triangle++) {
            expect(packedTriangleVertices(packed, triangle)).toEqual(Array.from(quad.indices.subarray(triangle * 3, triangle * 3 + 3)));
        }
        expect(Array.from(packedIndices(packed))).toEqual(Array.from(quad.indices));
        expect(Array.from(packedPositions(packed))).toEqual(Array.from(quad.positions));
        expect(Array.from(packed.bounds)).toEqual([0, 0, 0, 1, 1, 1]);
        expect(packed.offsets.normals).toBe(NONE);
        expect(packed.constantMaterialSlot).toBe(0);
    });

    it('switches to 32-bit indices above 65,536 vertices', () => {
        const positions = new Float32Array(65537 * 3);
        const packed = packMesh({ positions, indices: new Uint32Array([65536, 1, 0]) });
        expect(packed.indices16).toBe(false);
        expect(packedTriangleVertices(packed, 0)).toEqual([65536, 1, 0]);
        expect(packMesh({ positions: new Float32Array(65536 * 3), indices: new Uint32Array([65535, 1, 0]) }).indices16).toBe(true);
    });

    it('stores per-triangle slots only when they differ, and asks for enough materials', () => {
        const uniform = packMesh({ ...quad, groups: [{ start: 0, count: 9, materialSlot: 4 }] });
        expect(uniform.offsets.materialSlots).toBe(NONE);
        expect(uniform.constantMaterialSlot).toBe(4);
        expect(uniform.materialSlotCount).toBe(5);

        const mixed = packMesh({ ...quad, groups: [{ start: 0, count: 3, materialSlot: 2 }, { start: 6, count: 3, materialSlot: 1 }] });
        expect(mixed.constantMaterialSlot).toBe(NONE);
        expect(mixed.materialSlotCount).toBe(3);
        const slots = mixed.words[mixed.offsets.materialSlots];
        // Triangle 0: slot 2 (low half), triangle 1: uncovered, slot 0 (high half), triangle 2: slot 1.
        expect(slots & 0xffff).toBe(2);
        expect(slots >>> 16).toBe(0);
        expect(mixed.words[mixed.offsets.materialSlots + 1] & 0xffff).toBe(1);
    });

    it('rejects broken meshes with a message that says what is wrong', () => {
        expect(() => packMesh({ positions: new Float32Array(8), indices: new Uint32Array(3) })).toThrow(/three floats/);
        expect(() => packMesh({ positions: new Float32Array(9), indices: new Uint32Array(0) })).toThrow(/at least one triangle/);
        expect(() => packMesh({ ...quad, uvs: new Float32Array(4) })).toThrow(/uvs/);
        expect(() => packMesh({ ...quad, groups: [{ start: 1, count: 3, materialSlot: 0 }] })).toThrow(/multiples of 3/);
        expect(() => packMesh({ ...quad, groups: [{ start: 6, count: 6, materialSlot: 0 }] })).toThrow(/runs past/);
        expect(() => packMesh({ positions: new Float32Array([0, 0, NaN, 1, 0, 0, 0, 1, 0]), indices: new Uint32Array([0, 1, 2]) })).toThrow(/non-finite/);
    });
});

describe('range allocator', () => {
    it('reuses freed ranges, merges neighbours and grows at the end', () => {
        const allocator = new RangeAllocator(100);
        const a = allocator.allocate(30)!;
        const b = allocator.allocate(30)!;
        const c = allocator.allocate(30)!;
        expect([a, b, c]).toEqual([0, 30, 60]);
        expect(allocator.allocate(20)).toBeNull();
        allocator.release(b, 30);
        allocator.release(a, 30);
        expect(allocator.allocate(50)).toBe(0); // a and b merged
        expect(allocator.used).toBe(80);
        expect(allocator.capacityToFit(40)).toBe(130);
        allocator.grow(130);
        expect(allocator.allocate(40)).toBe(90);
        expect(() => allocator.release(0, 10)).not.toThrow();
        expect(() => allocator.release(0, 10)).toThrow(/double release/);
    });
});

describe('slot allocator', () => {
    it('hands out the lowest free slot and shrinks when the top is freed', () => {
        const slots = new SlotAllocator();
        expect([slots.take(), slots.take(), slots.take(), slots.take()]).toEqual([0, 1, 2, 3]);
        slots.release(2);
        slots.release(1);
        expect(slots.slotCount).toBe(4);
        expect(slots.take()).toBe(1);
        slots.release(3);
        expect(slots.slotCount).toBe(2); // 3 freed, and 2 below it was already free
        expect(slots.take()).toBe(2);
        expect(() => slots.release(5)).toThrow();
    });
});
