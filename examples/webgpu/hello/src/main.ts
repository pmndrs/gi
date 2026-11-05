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

/**
 * Updates per probe before the field counts as settled. Why 100: chosen for
 * the fixed 0.97 hysteresis (0.97^100 = 0.048 of where a probe started), and
 * light has had as many bounces to spread. Under the adaptive hysteresis
 * (the default since T1.12) the first updates are a running average, so
 * after 100 a probe's noise is down to a tenth, but bounce light can still
 * be a few percent low for some seconds (docs/design/ddgi.md).
 */
const SETTLED_UPDATES_PER_PROBE = 100;

/** Readiness for the screenshot script (scripts/screenshot.ts) and anyone watching. */
interface HelloState {
    /** Every probe has been updated at least SETTLED_UPDATES_PER_PROBE times. */
    ready: boolean;
    /** The fewest times any probe has been updated so far. */
    updatesPerProbe: number;
    frames: number;
    error: string | null;
    adapter: string;
}

declare global {
    interface Window {
        helloState?: HelloState;
    }
}

const state: HelloState = { ready: false, updatesPerProbe: 0, frames: 0, error: null, adapter: '' };
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
    // Free the GPU promptly when the page goes away (NOTES: release the device
    // between pages), and stop drawing with it. A page the browser keeps in
    // its back/forward cache comes back without a device, so it reloads.
    let animationFrame = 0;
    let stopped = false;
    window.addEventListener(
        'pagehide',
        () => {
            stopped = true;
            cancelAnimationFrame(animationFrame);
            device.destroy();
        },
        { once: true },
    );
    window.addEventListener('pageshow', (event) => {
        if (event.persisted) location.reload();
    });

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

    // DDGI walks every probe in a fixed order, a budget of probes per frame,
    // so all probes have been updated floor(total / count) times or more.
    const probeCount = ddgi.field?.grid.probeCount ?? 1;
    let probesUpdatedInTotal = 0;

    let previousTime = performance.now();
    const frame = (time: number): void => {
        if (stopped) return;
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
        const probesThisFrame = ddgi.counters['probes updated'] ?? 0;
        probesUpdatedInTotal += probesThisFrame;
        state.updatesPerProbe = Math.floor(probesUpdatedInTotal / probeCount);
        state.ready = state.updatesPerProbe >= SETTLED_UPDATES_PER_PROBE;
        const settling = state.ready
            ? `field settled (every probe updated ${SETTLED_UPDATES_PER_PROBE}+ times)`
            : `field settling: ${state.updatesPerProbe} of ${SETTLED_UPDATES_PER_PROBE} updates per probe`;
        status.textContent = `${manifest.title} · ${camera.name} · ${state.adapter} · ${settling} · ${probesThisFrame} probes this frame${state.error ? ` · ERROR ${state.error}` : ''}`;
        animationFrame = requestAnimationFrame(frame);
    };
    animationFrame = requestAnimationFrame(frame);
}

main().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    state.error = message;
    status.textContent = message;
    console.error(message);
});
