import { describe, expect, it } from 'vitest';
import { StepReadback, type StepReadbackBuffer } from './step-readback.ts';

/** A readback buffer whose map resolves when the test says, holding the GPU's step count at the copy. */
function fakeBuffer(): StepReadbackBuffer & { value: number; resolve: () => Promise<void>; maps: number } {
    let pending: (() => void) | null = null;
    const buffer = {
        value: 0,
        maps: 0,
        mapAsync: () => {
            buffer.maps++;
            return new Promise<void>((resolve) => {
                pending = resolve;
            });
        },
        getMappedRange: () => new Uint32Array([buffer.value]).buffer,
        unmap: () => {},
        resolve: async () => {
            pending?.();
            pending = null;
            await Promise.resolve();
            await Promise.resolve();
        },
    };
    return buffer;
}

describe("the neural tail's step readback (P6.4)", () => {
    it('reads the GPU count a frame after the copy, and takes one copy at a time', async () => {
        const buffer = fakeBuffer();
        const readback = new StepReadback(buffer);
        expect(readback.wantsCopy).toBe(true);
        buffer.value = 32;
        readback.copied();
        expect(readback.wantsCopy).toBe(false);
        readback.poll();
        readback.poll(); // mapping already: no second map
        expect(buffer.maps).toBe(1);
        await buffer.resolve();
        expect(readback.steps).toBe(32);
        expect(readback.wantsCopy).toBe(true);
    });

    it("drops a count copied before a restart, even when it is mapped after it (the review's stale gate)", async () => {
        const buffer = fakeBuffer();
        const readback = new StepReadback(buffer);
        buffer.value = 600; // the old network's Adam step: past the gate
        readback.copied();
        readback.restart(); // the box centred again, or resetHistory, before the next frame maps it
        readback.poll();
        await buffer.resolve();
        expect(readback.steps).toBe(0);
        // The next copy is the new network's, and is kept.
        buffer.value = 16;
        readback.copied();
        readback.poll();
        await buffer.resolve();
        expect(readback.steps).toBe(16);
    });

    it('drops a count whose map was already asked for when the restart came', async () => {
        const buffer = fakeBuffer();
        const readback = new StepReadback(buffer);
        buffer.value = 600;
        readback.copied();
        readback.poll();
        readback.restart();
        await buffer.resolve();
        expect(readback.steps).toBe(0);
    });
});
