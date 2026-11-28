/**
 * C1 · No volume needed: the radiance cache (`@pmndrs/gi/radiance-cache`)
 * lights a sunlit courtyard with no probe box to place. It lights what the
 * camera sees: screen probes on the visible surfaces, whose rays end in a
 * world cache of cells keyed by position.
 *
 * - The platform is made by hand (`createGI`) with the cache as its output,
 *   and nothing under it.
 * - `worldCache: 'sharc'`: with no DDGI volume, SHaRC carries every bounce
 *   (docs/design/radiance-cache-variants.md). It becomes the default with no
 *   volume in TASKS P6.7; until then the page asks for it.
 * - three draws the scene pass with `mrt()`: three's colour, `giAlbedo`, the
 *   normal and the screen inputs the cache reads (depth, normal, velocity).
 * - `screen.encodeBefore` runs the cache's screen work once a frame, then
 *   `ddgi()` adds `giAlbedo × E / π` per pixel, E read by screen position.
 *
 * The pass must be single-sampled (`antialias: false`), and the device needs
 * the platform's limits (`createGIRenderer`).
 */

import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { createGI } from '@pmndrs/gi/core';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';
import { createRendererAndDevice } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { showCacheStats } from '../../shared/cache-stats.ts';

// The screen inputs need a single-sampled pass.
const { renderer, device } = await createRendererAndDevice({ antialias: false });
const { scene, camera, lights, sky } = loadScene('courtyard-tree', renderer);
const sun = lights[0] as THREE.DirectionalLight;
if (!sky) throw new Error('The courtyard has a sky.');

// The cache alone: no volume, nothing under it.
const gi = createGI(device);
const cache = gi.addTechnique(new RadianceCacheTechnique({ worldCache: 'sharc' }));
gi.setOutput(cache);
const preparing = gi.prepare();
const threeGI = new ThreeGI(gi, scene, { preparing });
await preparing;

const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const screen = screenInputs(scenePass, threeGI);
scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
pipeline.outputNode = screen.encodeBefore(ddgi(scenePass, threeGI));

// GI on or off (the composite's intensity), and the sun's height and compass direction (the scene's own sun to start).
const settings = { gi: true, height: 58, direction: 53 };
const gui = new GUI({ title: 'C1 · No volume needed' });
gui.add(settings, 'gi').name('GI');
gui.add(settings, 'height', 3, 85, 1).name('sun height (°)');
gui.add(settings, 'direction', 0, 360, 1).name('sun direction (°)');

const sunDistance = sun.position.distanceTo(sun.target.position);
const toSun = new THREE.Vector3();
showCacheStats(device, gi, cache, () => 'world cache: SHaRC, no volume');
renderer.setAnimationLoop(() => {
    threeGI.light.intensity = settings.gi ? 1 : 0;
    toSun.setFromSphericalCoords(1, THREE.MathUtils.degToRad(90 - settings.height), THREE.MathUtils.degToRad(settings.direction));
    // A DirectionalLight shines from its position toward its target; the sky's sun follows it.
    sun.position.copy(sun.target.position).addScaledVector(toSun, sunDistance);
    sky.sunPosition.value.copy(toSun);

    threeGI.update(camera); // the frame's GI work; the screen stage runs inside the pipeline
    pipeline.render();
});
