/**
 * T1.7: how much GI Sponza gets through the three adapter. Most of Sponza's
 * materials are glTF metallic-roughness with metalness factor 1 and the real
 * metalness in the metalnessMap; reading only the factor makes their GI
 * albedo 0 and the GI near black. Since T4.3 the platform samples the maps
 * itself (the adapter's texture averages are gone), so the albedo reported
 * is the colour factor without textures: colour × (1 − metalness) for
 * materials without a metalnessMap, the colour alone for those with one.
 *
 * Runs DDGI on the manifest's volume for `FRAMES` frames, then renders the
 * GI alone: direct lights hidden, every surface a white Lambert material
 * (`scene.overrideMaterial`), so a pixel reads E / π and the image mean is
 * the camera's mean GI irradiance / π. Also reports the materials' mean GI
 * albedo and how each diffuse factor was made.
 *
 * `VITE_DDGI_NO_METALNESS_MAPS=1` drops every metalnessMap first, which is
 * what the adapter did before the metalness fold: the "before" number.
 * Manual: it needs the downloaded Sponza.
 */

import { MeshLambertMaterial, PerspectiveCamera, RenderTarget, type Material, type Mesh, type Object3D } from 'three/webgpu';
import { FloatType } from 'three';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/three/src/three-gi.ts';
import { materialDescriptor } from '../../../../packages/three/src/convert.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { runSteps, runThreePage } from '../three-shared/harness.ts';

const FRAMES = 300;
const PROBES_PER_FRAME = 4096;
const SIZE = 256;

const env = import.meta.env as Record<string, unknown>;
const withoutMetalnessMaps = env.VITE_DDGI_NO_METALNESS_MAPS === '1';

function luminance(color: readonly number[]): number {
    return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
}

await runThreePage(async ({ device, renderer, data, log }) => {
    const manifest = loadAllManifests().find((candidate) => candidate.id === 'sponza');
    assert(manifest, 'no sponza manifest');
    const loaded = await loadScene(manifest, () => {});
    const scene = loaded.scene;
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    const notContent: Object3D[] = [];
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) {
            object.userData.giContributes = false;
            notContent.push(object);
        }
    });

    // The materials, and their GI albedo as the adapter makes it.
    const materials = new Set<Material>();
    loaded.content.traverse((object) => {
        const material = (object as Mesh).material;
        if (material) for (const each of Array.isArray(material) ? material : [material]) materials.add(each);
    });
    if (withoutMetalnessMaps) for (const material of materials) (material as Material & { metalnessMap: unknown }).metalnessMap = null;
    let albedoSum = 0;
    let withMetalnessMap = 0;
    for (const material of materials) {
        const descriptor = materialDescriptor(material, () => undefined);
        const hasMetalnessMap = Boolean((material as Material & { metalnessMap?: unknown }).metalnessMap);
        albedoSum += luminance(descriptor.albedo ?? [0, 0, 0]) * (hasMetalnessMap ? 1 : 1 - (descriptor.metalness ?? 0));
        if ((material as Material & { metalnessMap?: unknown }).metalnessMap) withMetalnessMap++;
    }
    data.metalnessMaps = withoutMetalnessMaps ? 'dropped (before the fold)' : 'kept';
    data.materials = materials.size;
    data.materialsWithMetalnessMap = withMetalnessMap;
    data.meanGiAlbedoLuminance = albedoSum / materials.size;

    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);

    const platform = new GIPlatformImpl(device, { label: 'three sponza gi' });
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing }, warmUpPasses: 0 }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 }, environment: platformSkyEnvironment(manifest) });
    threeGI.update(camera);
    await platform.rays.whenBuilt();
    threeGI.update(camera);
    threeGI.budget = { units: PROBES_PER_FRAME };
    await runSteps(device, FRAMES, () => threeGI.update(camera));

    // GI alone on white Lambert: each pixel is E / pi.
    for (const light of loaded.directLights) light.visible = false;
    for (const object of notContent) object.visible = false;
    scene.overrideMaterial = new MeshLambertMaterial({ color: 0xffffff });
    const target = new RenderTarget(SIZE, SIZE, { type: FloatType, depthBuffer: true });
    await renderer.compileAsync(scene, camera);
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    const pixels = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, SIZE, SIZE)) as Float32Array;
    let sum = 0;
    for (let pixel = 0; pixel < SIZE * SIZE; pixel++) sum += luminance([pixels[pixel * 4], pixels[pixel * 4 + 1], pixels[pixel * 4 + 2]]);
    data.meanGiOnlyRadiance = sum / (SIZE * SIZE);
    data.meanGiIrradiance = (Math.PI * sum) / (SIZE * SIZE);
    log(`metalness maps ${data.metalnessMaps}: mean GI albedo ${(data.meanGiAlbedoLuminance as number).toFixed(3)}, mean GI irradiance ${(data.meanGiIrradiance as number).toFixed(3)} lux at bookmark 1`);
    // Show it on the page for the screenshot.
    renderer.render(scene, camera);
    target.dispose();
    threeGI.dispose();
    platform.dispose();
});
