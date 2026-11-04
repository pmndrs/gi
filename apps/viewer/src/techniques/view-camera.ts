/**
 * Each technique renders with its own camera, so the techniques can see
 * different layers in the same frame (the split-screen wipe). This copies the
 * viewer's camera onto one of them.
 */

import { PerspectiveCamera } from 'three';

export function createViewCamera(): PerspectiveCamera {
    return new PerspectiveCamera(60, 1, 0.1, 1000);
}

export function copyViewerCamera(viewer: PerspectiveCamera, target: PerspectiveCamera): void {
    target.position.copy(viewer.position);
    target.quaternion.copy(viewer.quaternion);
    // Only rebuild the projection when something that feeds it changed. VXGI's TRAA and the
    // upscaler jitter the projection each frame and own it between frames.
    if (target.fov !== viewer.fov || target.aspect !== viewer.aspect || target.near !== viewer.near || target.far !== viewer.far) {
        target.fov = viewer.fov;
        target.aspect = viewer.aspect;
        target.near = viewer.near;
        target.far = viewer.far;
        target.updateProjectionMatrix();
    }
    target.updateMatrixWorld(true);
}
