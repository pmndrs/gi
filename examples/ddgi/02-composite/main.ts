/**
 * GI the fast way for dense scenes, the intended pipeline: 01's scene and
 * volume, plus three's own post-processing pipeline and `@pmndrs/upscaler`.
 *
 * - The scene pass renders at the upscaler's render resolution (its
 *   Performance mode: half the display in each direction) and writes
 *   `giAlbedo`, the normal and three's `velocity` with `mrt()`.
 * - `ddgi()` adds GI there once per pixel, instead of once per fragment.
 *   It runs no occlusion on the GI term by default. Scenes with small folds
 *   the probes can't follow (Sponza's curtains) can opt in to three's GTAO
 *   at half that resolution: `ddgi(scenePass, volume, { occlusion: {
 *   resolutionScale: 0.5 } })`.
 * - The upscaler brings the result to the display.
 *
 * Transparent objects keep GI per fragment. Something opaque seen through a
 * transparent one gets its composited GI worked out at the transparent
 * surface, so draw transparents after the composite where that matters.
 */

import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass, rtt, velocity } from 'three/tsl';
import { QualityMode, getQualityModeRatio, upscale } from '@pmndrs/upscaler';
import { DDGIVolume, ddgi, giAlbedo } from '@pmndrs/gi/three';
import { createRenderer, orbitCamera } from '../../shared/renderer.ts';
import { createLightProbesScene } from '../../shared/lightprobes-scene.ts';
import { showStats } from '../../shared/stats.ts';

// No multisampling: the upscaler's temporal reconstruction is the anti-aliasing.
const renderer = await createRenderer({ antialias: false });
const { scene, camera } = createLightProbesScene(window.innerWidth / window.innerHeight);
orbitCamera(renderer, camera, new THREE.Vector3(0, 2.5, 0));

const volume = new DDGIVolume(6, 5, 6, 6, 6, 6);
volume.position.set(0, 2.5, 0);
scene.add(volume);
await volume.start(renderer, scene);

const renderScale = 1 / getQualityModeRatio(QualityMode.Performance);
const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView), velocity }));
scenePass.setResolutionScale(renderScale);
// The composite as a texture at the render resolution (half float: it is HDR), then upscaled to the display.
const gi = rtt(ddgi(scenePass, volume), null, null, { type: THREE.HalfFloatType, resolutionScale: renderScale });
pipeline.outputNode = upscale(gi, scenePass.getTextureNode('depth'), scenePass.getTextureNode('velocity'), camera, { quality: QualityMode.Performance });

showStats(volume);
renderer.setAnimationLoop(() => pipeline.render()); // instead of renderer.render(scene, camera)
