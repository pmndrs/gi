/**
 * GI the fast way for dense scenes, the intended pipeline (DECISIONS
 * 2026-11-06): the simple example plus three's own post-processing pipeline
 * and `@pmndrs/upscaler`. The scene pass renders at the upscaler's render
 * resolution (its Performance preset: half the display in each direction)
 * and writes `giAlbedo`, the normal and three's `velocity` with `mrt()`;
 * `ddgi()` adds GI once per pixel there, with three's GTAO on the GI term at
 * half that resolution (its default on a reduced pass); the upscaler brings
 * the result to the display. The GI lines are marked.
 *
 * Measured on Sponza at 2560 × 1440 (RTX 3060 laptop): showing GI costs
 * 0.73 ms this way, plus the upscaler's own 1.55 ms per frame (which it costs
 * with or without GI).
 *
 * Transparent objects keep GI per fragment. Something opaque seen through a
 * transparent one gets its composited GI evaluated at the transparent
 * surface, so draw transparents after the composite where that matters.
 */

import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass, rtt, velocity } from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { QualityMode, getQualityModeRatio, upscale } from '@pmndrs/upscaler';
import { GIVolume, ddgi, giAlbedo } from '@pmndrs/ddgi/three'; // GI
import { createCornellScene } from './cornell.ts';
import type { ExampleHandle } from './simple.ts';

export async function main(container: HTMLElement = document.body): Promise<ExampleHandle> {
    const { scene, camera } = createCornellScene(window.innerWidth / window.innerHeight);

    // No MSAA: the upscaler's temporal reconstruction is the anti-aliasing.
    const renderer = new THREE.WebGPURenderer({ antialias: false });
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

    const renderScale = 1 / getQualityModeRatio(QualityMode.Performance); // GI: the upscaler's render resolution
    const pipeline = new THREE.RenderPipeline(renderer); // GI: the pipeline
    const scenePass = pass(scene, camera); // GI
    scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView), velocity })); // GI
    scenePass.setResolutionScale(renderScale); // GI
    // The composite as a texture at the render resolution (half float: it is HDR), then upscaled.
    const gi = rtt(ddgi(scenePass, volume), null, null, { type: THREE.HalfFloatType, resolutionScale: renderScale }); // GI
    pipeline.outputNode = upscale(gi, scenePass.getTextureNode('depth'), scenePass.getTextureNode('velocity'), camera, { quality: QualityMode.Performance }); // GI

    const render = () => pipeline.render(); // GI: instead of renderer.render(scene, camera)
    renderer.setAnimationLoop(render);
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    return { renderer, scene, camera, volume, render };
}
