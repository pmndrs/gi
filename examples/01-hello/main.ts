/**
 * GI the simple way: a `DDGIVolume` in place of three's `LightProbeGrid`, on
 * three's own `webgpu_lightprobes.html` scene. The GI lines match
 * LightProbeGrid's one for one (`new`, `position.set`, `scene.add`, then
 * `start` for `bake`). After `start`, GI updates itself inside
 * `renderer.render`: there is no per-frame call.
 */

import * as THREE from 'three/webgpu';
import { DDGIVolume } from '@pmndrs/gi/three';
import { createRenderer, orbitCamera } from '../shared/renderer.ts';
import { createLightProbesScene } from '../shared/lightprobes-scene.ts';
import { showStats } from '../shared/stats.ts';

const renderer = await createRenderer();
const { scene, camera } = createLightProbesScene(window.innerWidth / window.innerHeight);
orbitCamera(renderer, camera, new THREE.Vector3(0, 2.5, 0));

const volume = new DDGIVolume(6, 5, 6, 6, 6, 6); // like new LightProbeGrid(6, 5, 6, 6, 6, 6): a 6 × 5 × 6 m box, 6 probes a side
volume.position.set(0, 2.5, 0);
scene.add(volume);
await volume.start(renderer, scene); // like probes.bake(renderer, scene)

showStats(volume);
renderer.setAnimationLoop(() => renderer.render(scene, camera)); // no per-frame GI call
