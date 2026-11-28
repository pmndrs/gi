import { REVISION, Texture } from 'three';
import { describe, expect, it } from 'vitest';
import { PINNED_THREE_REVISION, checkThreeRevision, threeGpuTexture } from './three-gpu-texture.ts';

/** A WebGPU backend's data map, as three r187 keeps it: a WeakMap from object to its data. */
function fakeBackend(isWebGPUBackend = true) {
    const data = new WeakMap<object, Record<string, unknown>>();
    return {
        isWebGPUBackend,
        get(object: object): Record<string, unknown> {
            let entry = data.get(object);
            if (!entry) {
                entry = {};
                data.set(object, entry);
            }
            return entry;
        },
    };
}

const fakeGpuTexture = { width: 4, height: 2, format: 'rgba16float', createView: () => ({}) };

describe('the backend step (three-gpu-texture.ts, D13)', () => {
    it('is pinned to the installed three, and fails loudly on any other revision', () => {
        expect(REVISION).toBe(PINNED_THREE_REVISION);
        expect(() => checkThreeRevision(REVISION)).not.toThrow();
        expect(() => checkThreeRevision('188')).toThrow(/checked on three r187; this is r188/);
        const backend = fakeBackend();
        expect(() => threeGpuTexture({ backend }, new Texture(), '186')).toThrow(/r186/);
    });

    it('reads the GPUTexture three stored for a texture, and null before three made one', () => {
        const backend = fakeBackend();
        const texture = new Texture();
        expect(threeGpuTexture({ backend }, texture)).toBeNull();
        backend.get(texture).texture = fakeGpuTexture;
        expect(threeGpuTexture({ backend }, texture)).toBe(fakeGpuTexture);
    });

    it('refuses a backend that is not WebGPU, and backend data of another shape', () => {
        expect(() => threeGpuTexture({ backend: fakeBackend(false) }, new Texture())).toThrow(/WebGPU backend/);
        const backend = fakeBackend();
        const texture = new Texture();
        texture.name = 'giNormal';
        backend.get(texture).texture = { not: 'a texture' };
        expect(() => threeGpuTexture({ backend }, texture)).toThrow(/'giNormal' isn't a GPUTexture/);
    });
});
