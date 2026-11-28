/**
 * The radiance cache (`@pmndrs/gi/radiance-cache`) on 01's scene, from
 * public paths only. Built from AMD's GI-1.0/1.2: screen probes on the
 * surfaces the camera sees, over a world-space hash grid that lights the
 * probes' rays. There is no volume to place for the cache itself.
 *
 * - The platform is made by hand (`createGI`) with the cache as its output.
 * - A DDGI volume is stacked under it (`setFallback`), so the world cache's
 *   second bounce, and every later one, comes from DDGI's field
 *   (`secondBounce: 'auto'`, the default). `?ddgi=off` runs the cache alone,
 *   with GI-1.2's own second-bounce rays.
 * - three draws the scene pass with `mrt()`: three's colour, `giAlbedo`, the
 *   normal and the screen inputs the cache reads (depth, normal, velocity).
 * - `screen.encodeBefore` runs the cache's screen work once a frame, then
 *   `ddgi()` adds `giAlbedo × E / π` per pixel, E read by screen position.
 *
 * The pass must be single-sampled (`antialias: false`), and the device needs
 * the platform's limits (`createGIRenderer`): the cache binds up to 10
 * storage buffers per shader stage and holds about 1 GiB of world cache.
 */

import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { createGI } from '@pmndrs/gi/core';
import { DDGITechnique } from '@pmndrs/gi/ddgi';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';
import { createRendererAndDevice, orbitCamera } from '../../shared/renderer.ts';
import { createLightProbesScene } from '../../shared/lightprobes-scene.ts';
import { showCacheStats } from '../../shared/cache-stats.ts';

const withDDGI = new URLSearchParams(location.search).get('ddgi') !== 'off';

// The screen inputs need a single-sampled pass.
const { renderer, device } = await createRendererAndDevice({ antialias: false });
const { scene, camera } = createLightProbesScene(window.innerWidth / window.innerHeight);
orbitCamera(renderer, camera, new THREE.Vector3(0, 2.5, 0));

const gi = createGI(device, { gpuTiming: device.features.has('timestamp-query') });
const cache = gi.addTechnique(new RadianceCacheTechnique());
gi.setOutput(cache);
let under: DDGITechnique | null = null;
if (withDDGI) {
    // The room, 6 × 5 × 6 m, with a probe every metre.
    under = gi.addTechnique(new DDGITechnique({ volume: { center: [0, 2.5, 0], size: [6, 5, 6], spacing: 1 } }));
    gi.setFallback(cache, under); // before prepare: the cache builds its DDGI lookup then
}
const preparing = gi.prepare();
// The cache ignores the budget; it hands DDGI its probes a frame.
const threeGI = new ThreeGI(gi, scene, { preparing, budget: { units: 256 } });
await preparing;

// DDGI warms up first (full passes over its probes), so the cache's later bounces don't start from a black field.
while (under?.warmingUp) {
    threeGI.update(camera, 0);
    await device.queue.onSubmittedWorkDone();
}

const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const screen = screenInputs(scenePass, threeGI);
scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
pipeline.outputNode = screen.encodeBefore(ddgi(scenePass, threeGI));

showCacheStats(device, gi, cache, () => (withDDGI ? 'second bounce from DDGI' : "second bounce from GI-1.2's rays"));
renderer.setAnimationLoop(() => {
    threeGI.update(camera); // the frame's GI work; the screen stage runs inside the pipeline
    pipeline.render();
});
