/**
 * GI the simple way: a `GIVolume` in place of three's `LightProbeGrid`, on
 * three's `webgpu_lightprobes.html` scene. The GI lines are marked; they
 * match LightProbeGrid's one for one (`new`, `position.set`, `scene.add`,
 * `bake` → `start`). After `start`, GI updates itself inside
 * `renderer.render`: no per-frame call.
 */

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GIVolume } from '@pmndrs/ddgi/three'; // GI
import { createCornellScene } from './cornell.ts';

export interface ExampleHandle {
    readonly renderer: THREE.WebGPURenderer;
    readonly scene: THREE.Scene;
    readonly camera: THREE.PerspectiveCamera;
    readonly volume: GIVolume;
    /** Draws one frame (the animation loop calls it too). */
    render(): void;
}

export async function main(container: HTMLElement = document.body): Promise<ExampleHandle> {
    const { scene, camera } = createCornellScene(window.innerWidth / window.innerHeight);

    const renderer = new THREE.WebGPURenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    container.appendChild(renderer.domElement);
    await renderer.init();

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 2.5, 0);
    controls.update();

    const volume = new GIVolume(6, 5, 6, 6, 6, 6); // GI: like new LightProbeGrid(6, 5, 6, 6, 6, 6)
    volume.position.set(0, 2.5, 0); // GI
    scene.add(volume); // GI
    await volume.start(renderer, scene); // GI: like probes.bake(renderer, scene)

    const render = () => renderer.render(scene, camera);
    renderer.setAnimationLoop(render);
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    return { renderer, scene, camera, volume, render };
}
