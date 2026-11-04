/**
 * The smallest plain-WebGPU use of @pmndrs/ddgi: no engine, one scene, DDGI.
 *
 *   pnpm --filter @pmndrs/ddgi-example-hello dev
 *
 * URL parameters:
 * - `scene`: a procedural test scene (default `cornell-box`; `cornell-box-point` has a point light);
 * - `camera`: the manifest's camera bookmark, 1-based (default 1);
 * - `view`: `full` (default), `gi` (indirect light only), or one of DDGI's
 *   debug views: `gi-only`, `coverage`, `probes`;
 * - `exposure`: EV, default 0.
 *
 * The steps, in order: the host makes the device, the platform gets the scene
 * through its scene API, DDGI is added, and each frame records the GI update
 * and then the renderer's own passes into one encoder.
 */

import { createGI, DDGI_DEBUG_VIEW, DDGITechnique, optionalFeatures, requiredLimits } from '@pmndrs/ddgi/core';
import { ForwardRenderer, type RenderView } from './renderer.ts';
import { describeScene, loadManifest } from './scene.ts';

const status = document.getElementById('status') as HTMLElement;
const canvas = document.getElementById('view') as HTMLCanvasElement;

/** URL `view` values for DDGI's screen debug views (ids from `technique.debugViews`). */
const DEBUG_VIEW_PARAMETERS: Readonly<Record<string, number>> = {
    'gi-only': DDGI_DEBUG_VIEW.giOnly,
    coverage: DDGI_DEBUG_VIEW.coverage,
    probes: DDGI_DEBUG_VIEW.probeStates,
};

function viewFromUrl(value: string | null): RenderView {
    if (value === null || value === 'full') return { mode: 'full' };
    if (value === 'gi') return { mode: 'gi' };
    const debugView = DEBUG_VIEW_PARAMETERS[value];
    if (debugView === undefined) throw new Error(`Unknown view "${value}". Use full, gi, ${Object.keys(DEBUG_VIEW_PARAMETERS).join(', ')}.`);
    return { mode: 'debug', debugView };
}

/** Readiness for the screenshot script (scripts/screenshot.ts) and anyone watching. */
interface HelloState {
    ready: boolean;
    frames: number;
    error: string | null;
    adapter: string;
}

declare global {
    interface Window {
        helloState?: HelloState;
    }
}

const state: HelloState = { ready: false, frames: 0, error: null, adapter: '' };
window.helloState = state;

async function main(): Promise<void> {
    const parameters = new URLSearchParams(location.search);
    const sceneId = parameters.get('scene') ?? 'cornell-box';
    const cameraIndex = Number(parameters.get('camera') ?? '1');
    const view = viewFromUrl(parameters.get('view'));
    const exposure = Number(parameters.get('exposure') ?? '0');

    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found.');
    state.adapter = `${adapter.info.vendor} ${adapter.info.architecture}`;

    // 1. The host owns the device. The platform asks for raised limits (large
    //    scenes need them) and a few optional features, where the adapter has them.
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    device.addEventListener('uncapturederror', (event) => {
        state.error = event.error.message;
        console.error('WebGPU error:', event.error.message);
    });
    // Free the GPU promptly when the page goes away (NOTES: release the device between pages).
    window.addEventListener('pagehide', () => device.destroy(), { once: true });

    // 2. The platform, and the scene described through its scene API.
    const gi = createGI(device, { gpuTiming: device.features.has('timestamp-query'), label: 'hello' });
    const manifest = loadManifest(sceneId);
    const scene = describeScene(gi.scene, manifest);
    const camera = manifest.cameras[cameraIndex - 1];
    if (!camera) throw new Error(`${sceneId} has ${manifest.cameras.length} cameras; camera=${cameraIndex} is not one of them.`);

    // 3. DDGI on the manifest's volume, as the output renderers read.
    const ddgi = gi.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing } }));
    gi.setOutput(ddgi);
    await gi.prepare();

    // 4. The renderer reads GI only through the shading module.
    const renderer = new ForwardRenderer(device, scene, { shading: gi.shading });
    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('The canvas has no WebGPU context.');
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    let previousTime = performance.now();
    const frame = (time: number): void => {
        const width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
        const height = Math.max(1, Math.round(canvas.clientHeight * devicePixelRatio));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        renderer.resize(width, height);
        const matrices = renderer.camera(camera);

        const encoder = device.createCommandEncoder({ label: 'hello frame' });
        // GI first: this frame's probe updates, within 2 ms of GPU time.
        gi.encode(encoder, {
            deltaTime: (time - previousTime) / 1000,
            budget: { gpuMs: 2 },
            camera: { position: matrices.position, view: matrices.view, projection: matrices.projection, near: matrices.near, far: matrices.far },
        });
        renderer.render(encoder, matrices, view);
        renderer.present(encoder, context.getCurrentTexture(), exposure);
        device.queue.submit([encoder.finish()]);

        previousTime = time;
        state.frames++;
        // Probes start out "not updated yet"; give the field a few hundred frames to settle.
        state.ready = state.frames >= 300;
        const counters = gi.stats.counters.ddgi ?? {};
        status.textContent = `${manifest.title} · ${camera.name} · ${state.adapter} · frame ${state.frames} · probes updated ${counters['probes updated'] ?? 0}${state.error ? ` · ERROR ${state.error}` : ''}`;
        requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
}

main().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    state.error = message;
    status.textContent = message;
    console.error(message);
});
