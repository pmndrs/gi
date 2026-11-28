/** The tracer's growable node and triangle buffers: doubling, and the headroom a growth may ask for (T2.12). */

import { describe, expect, it } from 'vitest';
import { createFakeGpu } from '../scene/testing/fake-gpu.ts';
import { GrowableStorage } from './growable-storage.ts';

describe('growable storage', () => {
    it('doubles when a little more is needed, keeping the ranges in use', () => {
        const gpu = createFakeGpu();
        const storage = new GrowableStorage(gpu.device, 'test', 4, 100);
        const offset = storage.allocate(80);
        expect(offset).toBe(0);
        storage.write(0, new Uint32Array(80).fill(7).buffer);
        const encoder = gpu.createEncoder();
        storage.grow(encoder, 40, [{ offset: 0, count: 80 }]);
        gpu.submit(encoder);
        expect(storage.byteLength).toBe(200 * 4);
        // The range in use was copied into the new buffer at its offset.
        expect(new Uint32Array((gpu.buffers.at(-1) as unknown as { bytes: Uint8Array }).bytes.buffer, 0, 80).every((word) => word === 7)).toBe(true);
        expect(storage.allocate(40)).toBe(80);
    });

    it('sizes a big jump by what must fit, plus the headroom a growth asks for, so a slightly larger replacement still fits', () => {
        const gpu = createFakeGpu();
        const exact = new GrowableStorage(gpu.device, 'exact', 4, 10);
        const roomy = new GrowableStorage(gpu.device, 'roomy', 4, 10);
        exact.grow(gpu.createEncoder(), 1000, []);
        roomy.grow(gpu.createEncoder(), 1000, [], 1 / 8);
        expect(exact.byteLength).toBe(1000 * 4);
        expect(roomy.byteLength).toBe(1125 * 4);

        // A scene's first trees fill it; then they are replaced by trees 5% larger.
        for (const storage of [exact, roomy]) {
            expect(storage.allocate(1000)).toBe(0);
            storage.release(0, 1000);
            storage.allocate(950);
        }
        expect(exact.allocate(100)).toBeNull();
        expect(roomy.allocate(100)).toBe(950);

        // A growth that doubles needs no headroom: the doubling is the larger.
        const small = new GrowableStorage(gpu.device, 'small', 4, 100);
        small.grow(gpu.createEncoder(), 50, [], 1 / 8);
        expect(small.byteLength).toBe(200 * 4);
    });
});
