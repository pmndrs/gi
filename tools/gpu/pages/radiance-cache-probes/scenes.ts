/**
 * Scenes the radiance-cache pages share (radiance-cache-probes,
 * radiance-cache-temporal). Not a page itself.
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
