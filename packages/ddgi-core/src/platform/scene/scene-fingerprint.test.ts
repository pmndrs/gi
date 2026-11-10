/**
 * The scene fingerprint (T5.2, scene-fingerprint.ts): the same scene gives
 * the same fingerprint however it was built, and each kind of change moves
 * its own part.
 */

import { describe, expect, it } from 'vitest';
import type { InstanceHandle, LightHandle, MaterialHandle, MeshHandle, TextureHandle } from '../contract/handles.ts';
import type { MeshDescriptor } from '../contract/scene.ts';
import { PlatformScene } from './platform-scene.ts';
import { ContentHasher, FULL_HASH_WORDS, fingerprintDifferences, stableText, textureContentHash, unorderedDigest } from './scene-fingerprint.ts';
import { createFakeGpu } from './testing/fake-gpu.ts';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function boxMesh(size = 1): MeshDescriptor {
    const h = size / 2;
    return {
        positions: new Float32Array([-h, -h, -h, h, -h, -h, h, h, -h, -h, h, -h]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
}

interface Built {
    readonly scene: PlatformScene;
    readonly instances: InstanceHandle[];
    readonly meshes: MeshHandle[];
    readonly material: MaterialHandle;
    readonly texture: TextureHandle;
    readonly light: LightHandle;
}

/** Two boxes, a textured material, a light and a sky; `reversed` creates the instances and lights in the other order. */
function build(reversed = false): Built {
    const scene = new PlatformScene(createFakeGpu().device);
    const pixels = new Uint8Array(4 * 4 * 4).map((_, index) => index);
    const texture = scene.createTexture({ source: { data: pixels, width: 4, height: 4 }, srgb: true, label: 'checker' });
    const material = scene.createMaterial({ albedo: [0.8, 0.7, 0.6], albedoTexture: texture, label: 'wall' });
    const meshes = [scene.createMesh(boxMesh(1)), scene.createMesh(boxMesh(2))];
    const placements = [
        { mesh: meshes[0], at: [0, 0, 0] },
        { mesh: meshes[1], at: [3, 0, 0] },
    ];
    const instances: InstanceHandle[] = [];
    for (const placement of reversed ? [...placements].reverse() : placements) {
        const transform = [...IDENTITY];
        transform[12] = placement.at[0];
        instances.push(scene.createInstance({ mesh: placement.mesh, materials: [material], transform }));
    }
    const lights = [
        { type: 'point', position: [0, 2, 0], color: [1, 1, 1], intensity: 10, range: 0 } as const,
        { type: 'directional', direction: [0, -1, 0], color: [1, 0.9, 0.8], intensity: 3, castsShadows: true } as const,
    ];
    const handles = (reversed ? [...lights].reverse() : lights).map((light) => scene.createLight(light));
    scene.setEnvironment({ type: 'color', color: [0.2, 0.3, 0.4], intensity: 1 });
    return { scene, instances, meshes, material, texture, light: handles[reversed ? 1 : 0] };
}

describe('the scene fingerprint', () => {
    it('is the same for the same scene built in another order', () => {
        expect(build(true).scene.fingerprint()).toEqual(build(false).scene.fingerprint());
    });

    it('moves only the geometry part when an instance moves, and comes back when it does', () => {
        const { scene, instances } = build();
        const before = scene.fingerprint();
        const moved = [...IDENTITY];
        moved[13] = 0.25;
        scene.setTransform(instances[0], moved);
        expect(fingerprintDifferences(before, scene.fingerprint())).toEqual(['geometry']);
        scene.setTransform(instances[0], IDENTITY);
        expect(scene.fingerprint()).toEqual(before);
    });

    it('moves only the geometry part when an instance is hidden', () => {
        const { scene, instances } = build();
        const before = scene.fingerprint();
        scene.setVisible(instances[1], false);
        expect(fingerprintDifferences(before, scene.fingerprint())).toEqual(['geometry']);
    });

    it('moves only the geometry part when a mesh changes shape, and comes back with the old shape', () => {
        const { scene, meshes } = build();
        const before = scene.fingerprint();
        expect(before.geometry).toMatch(/^[0-9a-f]{16}$/);
        const altered = boxMesh(1);
        altered.positions[0] = -0.6;
        scene.updateMesh(meshes[0], altered);
        expect(fingerprintDifferences(before, scene.fingerprint())).toEqual(['geometry']);
        scene.updateMesh(meshes[0], boxMesh(1));
        expect(scene.fingerprint()).toEqual(before);
    });

    it('moves only the materials part when a material colour changes, and ignores labels', () => {
        const { scene, material } = build();
        const before = scene.fingerprint();
        scene.updateMaterial(material, { label: 'renamed' });
        expect(scene.fingerprint()).toEqual(before);
        scene.updateMaterial(material, { albedo: [0.8, 0.7, 0.5] });
        expect(fingerprintDifferences(before, scene.fingerprint())).toEqual(['materials']);
    });

    it('moves only the lights part when a light changes, including its shadow setting', () => {
        const { scene, light } = build();
        const before = scene.fingerprint();
        scene.updateLight(light, { intensity: 11 });
        expect(fingerprintDifferences(before, scene.fingerprint())).toEqual(['lights']);
        scene.updateLight(light, { intensity: 10 });
        expect(scene.fingerprint()).toEqual(before);
        scene.updateLight(light, { castsShadows: false });
        expect(fingerprintDifferences(before, scene.fingerprint())).toEqual(['lights']);
    });

    it('moves only the sky part when the environment changes', () => {
        const { scene } = build();
        const before = scene.fingerprint();
        scene.setEnvironment({ type: 'color', color: [0.2, 0.3, 0.5], intensity: 1 });
        expect(fingerprintDifferences(before, scene.fingerprint())).toEqual(['sky']);
    });

    it('moves the materials and textures parts when a texture has other pixels', () => {
        const { scene, material } = build();
        const before = scene.fingerprint();
        const pixels = new Uint8Array(4 * 4 * 4).map((_, index) => index);
        pixels[17] = 0;
        const repainted = scene.createTexture({ source: { data: pixels, width: 4, height: 4 }, srgb: true, label: 'checker' });
        scene.updateMaterial(material, { albedoTexture: repainted });
        expect(fingerprintDifferences(before, scene.fingerprint())).toEqual(['materials', 'textures']);
    });

    it('says every part differs when the versions differ', () => {
        const { scene } = build();
        const now = scene.fingerprint();
        expect(fingerprintDifferences({ ...now, version: now.version + 1 }, now)).toEqual(['geometry', 'materials', 'textures', 'lights', 'sky']);
    });
});

describe('content hashing', () => {
    it('gives 16 hex digits, the same for the same input', () => {
        const words = new Uint32Array([1, 2, 3]);
        expect(new ContentHasher().words(words).digest()).toMatch(/^[0-9a-f]{16}$/);
        expect(new ContentHasher().words(words).digest()).toEqual(new ContentHasher().words(new Uint32Array([1, 2, 3])).digest());
        expect(new ContentHasher().words(words).digest()).not.toEqual(new ContentHasher().words(new Uint32Array([1, 3, 2])).digest());
    });

    it('hashes an unaligned byte view the same as an aligned copy', () => {
        const backing = new Uint8Array(19).map((_, index) => index * 7);
        const unaligned = backing.subarray(1, 18);
        expect(new ContentHasher().bytes(unaligned).digest()).toEqual(new ContentHasher().bytes(unaligned.slice()).digest());
    });

    it('samples big textures, so a changed sampled word still counts', () => {
        const size = FULL_HASH_WORDS * 4 + 4096;
        const data = new Uint8Array(size);
        const source = { data, width: size / 4, height: 1 };
        const before = textureContentHash(source, true);
        data[0] = 1;
        expect(textureContentHash(source, true)).not.toEqual(before);
        expect(textureContentHash(source, false)).not.toEqual(textureContentHash(source, true));
    });

    it('orders nothing in unordered digests and everything in stable text', () => {
        expect(unorderedDigest(['a', 'b'])).toEqual(unorderedDigest(['b', 'a']));
        expect(stableText({ b: 1, a: [1, 2], c: undefined })).toEqual(stableText({ a: [1, 2], b: 1 }));
    });
});
