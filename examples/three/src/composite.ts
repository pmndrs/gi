/**
 * GI the fast way for dense scenes: the simple example plus three's own
 * post-processing pipeline, as in three's SSGI and GTAO examples. The scene
 * pass writes `giAlbedo` and the normal with `mrt()`, and `ddgi()` adds GI
 * once per pixel instead of once per fragment, with three's GTAO on the GI
 * term (1 m). The GI lines are marked.
 *
 * Transparent objects keep GI per fragment. Something opaque seen through a
 * transparent one gets its composited GI evaluated at the transparent
 * surface, so draw transparents after the composite where that matters.
 */

import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GIVolume, ddgi, giAlbedo } from '@pmndrs/ddgi/three'; // GI
import { createCornellScene } from './cornell.ts';
import type { ExampleHandle } from './simple.ts';

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

    const volume = new GIVolume(6, 5, 6, 6, 6, 6); // GI
    volume.position.set(0, 2.5, 0); // GI
    scene.add(volume); // GI
    await volume.start(renderer, scene); // GI

    const pipeline = new THREE.RenderPipeline(renderer); // GI: the pipeline
    const scenePass = pass(scene, camera); // GI
    scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView) })); // GI
    pipeline.outputNode = ddgi(scenePass, volume); // GI

    const render = () => pipeline.render(); // GI: instead of renderer.render(scene, camera)
    renderer.setAnimationLoop(render);
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    return { renderer, scene, camera, volume, render };
}
