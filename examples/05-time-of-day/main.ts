/**
 * Time of day: the sun crosses the sky over a walled courtyard.
 *
 * Three things move together each frame: three's `DirectionalLight` (the
 * sun), three's `SkyMesh` (the sky the camera sees), and GI, which reads
 * both. The adapter takes the `SkyMesh` as GI's sky, its brightness from the
 * sun light, and makes its own copy of the sky again whenever the sun has
 * turned more than a twentieth of a degree.
 *
 * `sunLayer: true` gives the sun a layer of its own in GI, so its colour
 * and brightness (the warm, dim sun near the horizon here) show exactly in
 * the same frame, with nothing traced again. Its direction still re-traces
 * the probes, nearest the camera first.
 */

import * as THREE from 'three/webgpu';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { DDGIVolume } from '@pmndrs/gi/three';
import { createRenderer } from '../shared/renderer.ts';
import { loadScene } from '../shared/scene.ts';
import { showStats } from '../shared/stats.ts';

const renderer = await createRenderer();
const { scene, camera, lights, sky, probes } = loadScene('courtyard-tree', renderer, { cameraBookmark: 1 });
const sun = lights[0] as THREE.DirectionalLight;
if (!sky) throw new Error('The courtyard has a sky.');

// Something in the middle of the yard for the sun to light and shade.
const stone = new THREE.MeshStandardMaterial({ color: 0xb8ab98 });
const red = new THREE.MeshStandardMaterial({ color: 0xc8402c });
const pieces: Array<[THREE.BufferGeometry, THREE.Material, [number, number, number]]> = [
    [new THREE.BoxGeometry(0.8, 3.2, 0.8), stone, [0.5, 1.6, -0.5]],
    [new THREE.BoxGeometry(2.4, 0.9, 1.2), red, [-1.2, 0.45, 1.0]],
    [new THREE.SphereGeometry(0.6, 48, 24), stone, [1.8, 0.6, 1.6]],
];
for (const [geometry, material, position] of pieces) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...position);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    scene.add(mesh);
}

const volume = DDGIVolume.fromSettings(probes);
scene.add(volume);
await volume.start(renderer, scene, { ddgi: { sunLayer: true } });

const settings = { hour: 8, play: true };
const gui = new GUI({ title: 'Time of day' });
gui.add(settings, 'hour', 6, 18, 0.01).listen();
gui.add(settings, 'play').name('play (a 30 s day)');

const noonIntensity = sun.intensity;
const sunDistance = sun.position.distanceTo(sun.target.position);
const toSun = new THREE.Vector3();
const warm = new THREE.Color(1, 0.55, 0.3);
const white = new THREE.Color(1, 0.96, 0.9);

const timer = new THREE.Timer();
showStats(volume, { note: () => `${Math.floor(settings.hour)}:${String(Math.floor((settings.hour % 1) * 60)).padStart(2, '0')}` });
renderer.setAnimationLoop(() => {
    timer.update();
    if (settings.play) settings.hour = 6 + ((settings.hour - 6 + timer.getDelta() * (12 / 30)) % 12);

    // From the east at 6 am, overhead (and a little south) at noon, to the west at 6 pm.
    const angle = ((settings.hour - 6) / 12) * Math.PI;
    toSun.set(Math.cos(angle), Math.sin(angle), 0.35).normalize();
    // A DirectionalLight shines from its position toward its target.
    sun.position.copy(sun.target.position).addScaledVector(toSun, sunDistance);
    // Low sun: dimmer and warmer, gone at the horizon.
    const height = THREE.MathUtils.smoothstep(toSun.y, 0, 0.35);
    sun.intensity = noonIntensity * height;
    sun.color.lerpColors(warm, white, height);
    sky.sunPosition.value.copy(toSun);

    renderer.render(scene, camera);
});
