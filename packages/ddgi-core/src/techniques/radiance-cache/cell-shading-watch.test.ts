import { describe, expect, it } from 'vitest';
import { CellShadingAgeWatch } from './cell-shading-watch.ts';

/** A buffer whose mapAsync resolves with `value` in its first word. */
function fakeDevice(value: number): { device: GPUDevice } {
    const device = {
        createBuffer: () => ({
            mapAsync: () => Promise.resolve(),
            getMappedRange: () => new Uint32Array([value]).buffer,
            unmap() {},
            destroy() {},
        }),
    } as unknown as GPUDevice;
    return { device };
}

describe("lever 1's age watch", () => {
    it('reads the counter back without waiting, and warns once when cells passed 255 frames unlit', async () => {
        const { device } = fakeDevice(12);
        const warnings: string[] = [];
        const watch = new CellShadingAgeWatch(device, (message) => warnings.push(message));
        let copies = 0;
        const encoder = { copyBufferToBuffer: () => copies++ } as unknown as GPUCommandEncoder;
        const counter = { buffer: {} as GPUBuffer, offset: 60 };
        for (let frame = 0; frame < 200; frame++) {
            watch.record(encoder, counter, 1000);
            await Promise.resolve();
        }
        expect(copies).toBeGreaterThanOrEqual(3);
        expect(watch.agedCells).toBe(12);
        expect(warnings).toHaveLength(1);
        expect(warnings[0]).toMatch(/cellShadingBudget 1000 is too small/);
    });

    it('stays quiet while no cell passes 255 frames', async () => {
        const { device } = fakeDevice(0);
        const warnings: string[] = [];
        const watch = new CellShadingAgeWatch(device, (message) => warnings.push(message));
        const encoder = { copyBufferToBuffer() {} } as unknown as GPUCommandEncoder;
        for (let frame = 0; frame < 200; frame++) {
            watch.record(encoder, { buffer: {} as GPUBuffer, offset: 60 }, 1000);
            await Promise.resolve();
        }
        expect(watch.agedCells).toBe(0);
        expect(warnings).toHaveLength(0);
    });
});
