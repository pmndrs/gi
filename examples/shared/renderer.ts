/**
 * The renderer and camera set-up every example shares: three's
 * `WebGPURenderer` on a device with the limits GI needs
 * (`createGIRenderer`), filling the window, and an orbiting camera.
 */

import * as THREE from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createGIRenderer } from '@pmndrs/ddgi/three';
import { reportError } from './status.ts';

export interface RendererOptions {
    /** Multisampling. Off for the upscaler examples: its temporal pass is the anti-aliasing. Default on. */
    readonly antialias?: boolean;
}

/** A `WebGPURenderer` filling the window, with shadows and ACES tone mapping, its canvas in the page. */
export async function createRenderer(options: RendererOptions = {}): Promise<THREE.WebGPURenderer> {
    if (!navigator.gpu) throw new Error('This example needs WebGPU: Chrome or Edge 113+, or Safari 26+.');
    const { renderer, device } = await createGIRenderer({ renderer: { antialias: options.antialias ?? true } });
    // The host owns the device, so the host listens for its errors.
    device.addEventListener('uncapturederror', (event) => {
        console.error('WebGPU error:', event.error.message);
        reportError(event.error.message);
    });
    // Free the GPU as soon as the page goes away, rather than whenever it is collected. A page the
    // browser keeps in its back/forward cache comes back without a device, so it reloads.
    window.addEventListener('pagehide', () => {
        renderer.setAnimationLoop(null);
        device.destroy();
    });
    window.addEventListener('pageshow', (event) => {
        if (event.persisted) location.reload();
    });
    // Two device pixels per CSS pixel at most: GI costs per pixel too.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    document.body.append(renderer.domElement);
    return renderer;
}

/** Orbit controls around `target`, and the camera and canvas following the window's size. */
export function orbitCamera(renderer: THREE.WebGPURenderer, camera: THREE.PerspectiveCamera, target: THREE.Vector3Like): OrbitControls {
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(target.x, target.y, target.z);
    controls.update();
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    return controls;
}
