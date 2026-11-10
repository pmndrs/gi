/**
 * Scenes the radiance-cache pages share (radiance-cache-probes,
 * radiance-cache-temporal, radiance-cache-filter). Not a page itself.
 */

import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { SceneSync } from '../../../../packages/ddgi-three/src/scene-sync.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import type { PageCamera } from './screen.ts';

/** The sky-plane scene's sky radiance. */
export const SKY_RADIANCE = 1;

/** A 400 m floor at y = 0 under a constant white sky, no lights. */
export async function describeSkyPlane(platform: GIPlatformImpl): Promise<void> {
    const half = 200;
    const scene = platform.scene;
    const mesh = scene.createMesh({
        positions: new Float32Array([-half, 0, -half, half, 0, -half, half, 0, half, -half, 0, half]),
        normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
        indices: new Uint32Array([0, 2, 1, 0, 3, 2]),
        label: 'floor',
    });
    const material = scene.createMaterial({ albedo: [0.5, 0.5, 0.5], label: 'floor' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'floor' });
    scene.setEnvironment({ type: 'color', color: [SKY_RADIANCE, SKY_RADIANCE, SKY_RADIANCE], intensity: 1 });
}

type Vector = [number, number, number];

/** The sky plane and a 0.6 × 2.5 × 0.6 m box (albedo 0.7, 0.3, 0.2) standing on it 5 m ahead of the origin, at (0, 0, -5). */
export async function describePillar(platform: GIPlatformImpl): Promise<void> {
    await describeSkyPlane(platform);
    const scene = platform.scene;
    // A 0.6 x 2.5 x 0.6 m box standing on the floor 5 m ahead (each face its own four vertices, normals out).
    const [cx, cz, half, height] = [0, -5, 0.3, 2.5];
    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];
    const face = (normal: Vector, corners: Vector[]): void => {
        const base = positions.length / 3;
        for (const corner of corners) {
            positions.push(...corner);
            normals.push(...normal);
        }
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    };
    const [x0, x1, z0, z1] = [cx - half, cx + half, cz - half, cz + half];
    face([0, 0, 1], [[x0, 0, z1], [x1, 0, z1], [x1, height, z1], [x0, height, z1]]);
    face([0, 0, -1], [[x1, 0, z0], [x0, 0, z0], [x0, height, z0], [x1, height, z0]]);
    face([1, 0, 0], [[x1, 0, z1], [x1, 0, z0], [x1, height, z0], [x1, height, z1]]);
    face([-1, 0, 0], [[x0, 0, z0], [x0, 0, z1], [x0, height, z1], [x0, height, z0]]);
    face([0, 1, 0], [[x0, height, z1], [x1, height, z1], [x1, height, z0], [x0, height, z0]]);
    const mesh = scene.createMesh({ positions: new Float32Array(positions), normals: new Float32Array(normals), indices: new Uint32Array(indices), label: 'pillar' });
    const material = scene.createMaterial({ albedo: [0.7, 0.3, 0.2], label: 'pillar' });
    scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], label: 'pillar' });
}

/** The manifest's first bookmark as a page camera. */
export function bookmarkCamera(manifest: SceneManifest, far: number): PageCamera {
    const bookmark = manifest.cameras[0];
    return { position: [...bookmark.position], target: [...bookmark.target], fovYDeg: bookmark.fovYDeg, near: 0.05, far };
}

/** Sponza through the viewer's loader, copied into the platform by the three adapter's scene sync (a loader here, nothing more). */
export async function describeSponza(platform: GIPlatformImpl, device: GPUDevice, manifest: SceneManifest): Promise<() => void> {
    const loaded = await loadScene(manifest, () => {});
    // Only the model is GI geometry; the sky dome is the viewer's background (GI gets the sky as its environment).
    const content = new Set<object>();
    loaded.content.traverse((object) => content.add(object));
    loaded.scene.traverse((object) => {
        if ((object as { isMesh?: boolean }).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    loaded.scene.updateMatrixWorld();
    const sync = new SceneSync(platform.scene, { device, warn: () => {} });
    sync.environment = platformSkyEnvironment(manifest);
    return () => {
        sync.update(loaded.scene);
    };
}
