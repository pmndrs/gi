/**
 * Contact detail: the radiance cache and DDGI on one view, a wipe between
 * them. The scene is the cabinet row with an open shelf on legs and small
 * boxes where light meets a contact (`cabinet-row-objects`).
 *
 * - Left: the radiance cache on a platform made by hand (`createGI`), with
 *   SHaRC as its world cache, so it needs no volume at all (P6.5's choice
 *   when there is none; docs/design/radiance-cache-variants.md).
 * - Right: DDGI, a `DDGIVolume` over the room with its probes 0.25 m or
 *   0.5 m apart. A new spacing is a new volume: `start` it, swap it in,
 *   dispose the old one.
 * - One scene pass feeds both. Its depth, normals and motion go to the
 *   cache (`screenInputs`); each technique's `ddgi()` composite adds its own
 *   GI to the same pass colour, and `wipe.node` picks a side per pixel (or
 *   shows their difference). three's tone mapping applies to both alike.
 *
 * Two platforms on one scene pass need two things kept apart, because each
 * platform's shading functions have the same names:
 * - each composite is drawn into its own texture (`rtt`), so no shader holds
 *   both;
 * - the cache's GI light is on a layer the camera doesn't see, so the scene
 *   pass's materials hold only DDGI's. The composites light every surface
 *   here (all opaque Standard materials), so the light isn't needed.
 *
 * The pass is single-sampled (`antialias: false`): the screen inputs need it.
 */

import * as THREE from 'three/webgpu';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { mrt, normalView, output, packNormalToRGB, pass, rtt } from 'three/tsl';
import { createGI } from '@pmndrs/gi/core';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { DDGIVolume, ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';
import { createRendererAndDevice } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { createWipe } from '../../shared/wipe.ts';
import { showCacheStats } from '../../shared/cache-stats.ts';

/** A layer the camera doesn't see. */
const HIDDEN_LAYER = 1;
/** The composites' textures: half float, as they are HDR. */
const PICTURE = { type: THREE.HalfFloatType };

const { renderer, device } = await createRendererAndDevice({ antialias: false });
const { scene, camera, controls, probes } = loadScene('cabinet-row-objects', renderer);

// Left: the radiance cache. SHaRC carries every bounce on its own, so there is no volume to place.
const gi = createGI(device);
const cache = gi.addTechnique(new RadianceCacheTechnique({ worldCache: 'sharc' }));
gi.setOutput(cache);
const preparing = gi.prepare();
const cacheGI = new ThreeGI(gi, scene, { preparing });
cacheGI.light.layers.set(HIDDEN_LAYER);
await preparing;

// Right: DDGI, a probe grid over the room, `spacing` metres apart.
async function startDDGI(spacing: number): Promise<DDGIVolume> {
    const next = DDGIVolume.fromSettings({ ...probes, spacing });
    scene.add(next);
    await next.start(renderer, scene);
    return next;
}
const settings = { spacing: 0.25 }; // the scene's own spacing; the panel offers 0.5 m too
let volume = await startDDGI(settings.spacing);

// One scene pass for both pictures. The cache reads its screen inputs; both composites read its colour, albedo and normal.
const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const screen = screenInputs(scenePass, cacheGI);
scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
const cachePicture = rtt(screen.encodeBefore(ddgi(scenePass, cacheGI)), null, null, PICTURE);
const wipe = createWipe(renderer.domElement, { labels: ['radiance cache', 'DDGI'], controls });
function showPictures(): void {
    pipeline.outputNode = wipe.node(cachePicture, rtt(ddgi(scenePass, volume), null, null, PICTURE));
    pipeline.needsUpdate = true;
}
showPictures();

const gui = new GUI({ title: 'Contact detail' });
gui.add(wipe, 'position', 0.02, 0.98, 0.01).name('wipe position').listen();
gui.add(wipe, 'difference').name('difference |cache − DDGI|');
const spacing = gui.add(settings, 'spacing', { '0.25 m': 0.25, '0.5 m': 0.5 }).name('DDGI probe spacing');
spacing.onChange(async (metres: number) => {
    spacing.disable(); // one new volume at a time
    // The old field keeps showing, frozen, while the new one warms up: its light leaves the pass.
    const old = volume;
    old.light.layers.set(HIDDEN_LAYER);
    volume = await startDDGI(metres);
    showPictures();
    scene.remove(old);
    old.dispose();
    spacing.enable();
});

showCacheStats(device, gi, cache, () => `left: radiance cache (SHaRC) · right: DDGI, probes ${(volume.width / volume.resolution.x).toFixed(2)} m apart`);
renderer.setAnimationLoop(() => {
    cacheGI.update(camera); // the cache's world-space work; DDGI updates itself inside the scene pass
    pipeline.render();
});
