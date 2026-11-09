import { describe, expect, it } from 'vitest';
import { DataTexture, Group, Mesh, BoxGeometry, MeshStandardMaterial, Texture } from 'three';
import { planTexture, textureSlots } from './compress-textures';

/** Node has no OffscreenCanvas; the plan's image test uses `instanceof OffscreenCanvas`, so the tests install this one. */
class OffscreenCanvasStandIn {
    width = 0;
    height = 0;
}
(globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas ??= OffscreenCanvasStandIn;

/** A texture whose image passes for a decoded canvas of this size (the plan reads only the image's type and size). */
function imageTexture(width: number, height: number): Texture {
    const canvas = Object.create(OffscreenCanvasStandIn.prototype) as OffscreenCanvasStandIn;
    canvas.width = width;
    canvas.height = height;
    return new Texture(canvas as unknown as OffscreenCanvas);
}

describe('texture compression plan (T3.12)', () => {
    it('finds each texture with every slot that uses it', () => {
        const shared = imageTexture(256, 256);
        const first = new MeshStandardMaterial({ map: imageTexture(64, 64), normalMap: shared });
        const second = new MeshStandardMaterial({ roughnessMap: shared });
        const root = new Group();
        root.add(new Mesh(new BoxGeometry(), first), new Mesh(new BoxGeometry(), [second, first]));
        const slots = textureSlots(root);
        expect(slots.size).toBe(2);
        expect([...slots.get(shared)!].sort()).toEqual(['normalMap', 'roughnessMap']);
        expect([...slots.get(first.map!)!]).toEqual(['map']);
    });

    it('compresses the maps GI reads in both compressed modes since M4 (GI reads the compressed mips, T4.3)', () => {
        const texture = imageTexture(512, 512);
        for (const slot of ['map', 'emissiveMap', 'metalnessMap']) {
            expect(planTexture(texture, new Set([slot]), 'compressed')).toEqual({ action: 'compress', kind: 'colour' });
            expect(planTexture(texture, new Set([slot]), 'all-compressed')).toEqual({ action: 'compress', kind: 'colour' });
        }
        expect(planTexture(texture, new Set(['roughnessMap', 'metalnessMap']), 'compressed')).toEqual({ action: 'compress', kind: 'colour' });
    });

    it('makes a normal map BC5 only when every use is a normal map', () => {
        const texture = imageTexture(1024, 1024);
        expect(planTexture(texture, new Set(['normalMap']), 'compressed')).toEqual({ action: 'compress', kind: 'normal' });
        expect(planTexture(texture, new Set(['normalMap', 'aoMap']), 'compressed')).toEqual({ action: 'compress', kind: 'colour' });
    });

    it('keeps what it cannot compress: sides that are not powers of two, data textures, compressed ones', () => {
        expect(planTexture(imageTexture(1000, 1024), new Set(['normalMap']), 'all-compressed')).toEqual({ action: 'keep', reason: 'size' });
        expect(planTexture(imageTexture(1, 1), new Set(['normalMap']), 'all-compressed')).toEqual({ action: 'compress', kind: 'normal' });
        const data = new DataTexture(new Uint8Array(16), 2, 2);
        expect(planTexture(data, new Set(['map']), 'all-compressed')).toEqual({ action: 'keep', reason: 'not-an-image' });
    });
});
