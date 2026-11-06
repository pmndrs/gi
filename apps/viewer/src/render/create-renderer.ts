/**
 * Creates the three.js WebGPU renderer on a device we create ourselves,
 * through the library's `createGIRenderer`, so three and the GI platform
 * share one device (the host owns the device).
 *
 * Why not let three create it: three asks for a compatibility-mode adapter
 * with default limits (a 128 MB storage-buffer binding, 8 storage buffers per
 * stage). DDGI on Sponza needs about 276 MB of triangle records in one
 * binding, and the Bistro exterior alone needs about 136 MB of triangles for
 * VXGI's voxelizer. `createGIRenderer` asks for the platform's limits
 * (`requiredLimits`); the viewer adds every other limit the adapter offers,
 * as it did before DDGI arrived, so the three.js baselines keep their room.
 */

import type { WebGPURenderer } from 'three/webgpu';
import { createGIRenderer } from '../ddgi/library';

export interface RendererBundle {
    renderer: WebGPURenderer;
    device: GPUDevice;
    adapterInfo: GPUAdapterInfo;
    /** Whether GPU timestamp queries are available on this device. */
    hasTimestamps: boolean;
    /** Every uncaptured GPU error and device loss seen so far, newest last. */
    gpuErrors: string[];
}

/** Every limit the adapter offers, so the device gets the adapter's maximum. */
function maximumLimits(adapter: GPUAdapter): Record<string, number> {
    const limits: Record<string, number> = {};
    // GPUSupportedLimits exposes its values as prototype getters, so `Object.entries` finds none.
    for (const name in adapter.limits) {
        const value = (adapter.limits as unknown as Record<string, unknown>)[name];
        if (typeof value === 'number') limits[name] = value;
    }
    return limits;
}

export async function createRenderer(canvas: HTMLCanvasElement): Promise<RendererBundle> {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser. Use a recent Chrome or Edge.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU is available but no graphics adapter was found.');

    // No canvas antialiasing: the picture goes through a render pipeline, so MSAA on the canvas would do
    // nothing, and VXGI's depth copy fails when the renderer's default sample count is 4. The scene
    // targets of the scene-pass techniques ask for MSAA themselves.
    const { renderer, device } = await createGIRenderer({
        adapter,
        label: 'ddgi viewer',
        requiredLimits: maximumLimits(adapter),
        renderer: { canvas, antialias: false, trackTimestamp: adapter.features.has('timestamp-query') },
    });

    // addEventListener, never `device.onuncapturederror =`: three assigns that slot.
    const gpuErrors: string[] = [];
    device.addEventListener('uncapturederror', (event) => {
        const message = (event as GPUUncapturedErrorEvent).error.message;
        gpuErrors.push(message);
        console.error(`GPU error: ${message}`);
    });
    void device.lost.then((info) => {
        // `destroy` on unload is a normal end, not a problem.
        if (info.reason === 'destroyed') return;
        const message = `device lost (${info.reason}): ${info.message}`;
        gpuErrors.push(message);
        console.error(`GPU error: ${message}`);
    });

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setSize(canvas.clientWidth || window.innerWidth, canvas.clientHeight || window.innerHeight, false);
    renderer.shadowMap.enabled = true;

    return { renderer, device, adapterInfo: device.adapterInfo, hasTimestamps: device.features.has('timestamp-query'), gpuErrors };
}
