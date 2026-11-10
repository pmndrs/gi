/**
 * Lights change, GI follows. The Cornell box with one point lamp: move it,
 * dim it, recolour it or switch it off, and GI catches up within a few
 * frames. Nothing is called on change: the adapter sees the light's new
 * position, colour and intensity on the next render, and the probes the
 * light reaches are traced again, nearest the camera first. Until they are,
 * they keep their old light, so the room never goes black.
 *
 * For changes that must show in the very same frame (dimming a whole group
 * of lights, the sun's colour), see light groups and the sun layer in
 * docs/api/three.md; 05-time-of-day uses the sun layer.
 */

import * as THREE from 'three/webgpu';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { DDGIVolume } from '@pmndrs/gi/three';
import { createRenderer } from '../shared/renderer.ts';
import { loadScene } from '../shared/scene.ts';
import { showStats } from '../shared/stats.ts';

const renderer = await createRenderer();
const { scene, camera, lights, probes } = loadScene('cornell-box-point', renderer);
const lamp = lights[0] as THREE.PointLight;

const volume = DDGIVolume.fromDDGIVolume(probes); // the box the scene suggests: probes 0.5 m apart
scene.add(volume);
await volume.start(renderer, scene);

// The panel edits the light itself; that is all GI needs.
const settings = { sweep: true, on: true, color: `#${lamp.color.getHexString()}` };
const gui = new GUI({ title: 'Lamp' });
gui.add(settings, 'sweep').name('sweep across the box');
gui.add(lamp.position, 'x', 0.4, 5.1, 0.01).listen();
gui.add(lamp.position, 'y', 0.4, 5.1, 0.01).listen();
gui.add(lamp.position, 'z', 0.4, 5.1, 0.01).listen();
gui.add(lamp, 'intensity', 0, 40, 0.1);
gui.addColor(settings, 'color').onChange((color: string) => lamp.color.set(color));
gui.add(settings, 'on').onChange((on: boolean) => (lamp.visible = on));

const timer = new THREE.Timer();
showStats(volume);
renderer.setAnimationLoop(() => {
    timer.update();
    if (settings.sweep) {
        // Left to right and back across the box, every 8 seconds.
        const swing = Math.sin(timer.getElapsed() * ((2 * Math.PI) / 8));
        lamp.position.set(2.78 + 2.1 * swing, 4.2, 2.8);
    }
    renderer.render(scene, camera);
});
