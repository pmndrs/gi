import { describe, expect, it } from 'vitest';
import { checkGpuLayout } from './layout-check.ts';
import { createMeshLayoutBuilder } from './mesh-builder.ts';

function randomSoup(count: number, seed: number): Float32Array {
    let state = seed >>> 0;
    const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    return Float32Array.from({ length: count * 9 }, () => random() * 10);
}

describe('createMeshLayoutBuilder', () => {
    it('falls back to tinybvh on a device without subgroups, and the trees are valid', async () => {
        // Only `features` is read before the fallback is chosen.
        const device = { features: new Set<string>() } as unknown as GPUDevice;
        const builder = await createMeshLayoutBuilder(device);
        expect(builder.kind).toBe('tinybvh');
        expect(builder.fallbackReason).toMatch(/subgroups/);
        const geometries = [{ positions: randomSoup(1, 1) }, { positions: randomSoup(500, 2) }];
        const results = await builder.buildLayouts(geometries);
        results.forEach((result, index) => {
            expect(result.builtBy).toBe('tinybvh');
            expect(checkGpuLayout(result.layout, geometries[index])).toEqual([]);
        });
    });

    it('falls back without a device, or when the CPU builder is asked for', async () => {
        expect((await createMeshLayoutBuilder(null)).kind).toBe('tinybvh');
        const device = { features: new Set(['subgroups']) } as unknown as GPUDevice;
        const forced = await createMeshLayoutBuilder(device, { prefer: 'tinybvh' });
        expect(forced.kind).toBe('tinybvh');
        expect(forced.fallbackReason).toMatch(/asked/);
        // Asking for either GPU builder on a device without subgroups still gives tinybvh.
        const withoutSubgroups = { features: new Set<string>() } as unknown as GPUDevice;
        for (const prefer of ['gpu-hploc', 'gpu-lbvh'] as const) expect((await createMeshLayoutBuilder(withoutSubgroups, { prefer })).kind).toBe('tinybvh');
    });
});
