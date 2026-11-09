/**
 * M4, T4.3 done-when 4's picture: the viewer's own core (`ViewerCore`, as the
 * viewer page draws it) opens Sponza with the curtains, GI only, at the
 * first bookmark, warms DDGI up as the viewer does, runs a few more frames
 * and leaves the picture on the page for the runner's screenshot. Reports
 * the picture's mean (DDGI's own read-back of the view).
 *
 * Nothing here depends on T4.3, so the same page runs on the commit before
 * it for a before-and-after pair. `VITE_DDGI_SCENES=<id>` picks another
 * scene.
 */

import { Vector2 } from 'three';
import { reportGpuTest } from '../../client.ts';
import { ViewerCore } from '../../../../apps/viewer/src/core/viewer-core.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { DEFAULT_DDGI_VIEWER_SETTINGS } from '../../../../apps/viewer/src/techniques/ddgi-technique.ts';

const env = import.meta.env as Record<string, unknown>;
const SCENE = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES.trim() !== '' ? env.VITE_DDGI_SCENES.trim() : 'sponza';
const WIDTH = 1280;
const HEIGHT = 720;
/** Ordinary frames after DDGI's warm-up, before the picture is kept. */
const SETTLE_FRAMES = 120;
const WARM_UP_LIMIT_MS = 180_000;

const startTime = performance.now();
const logLines: string[] = [];
const pageErrors: string[] = [];
window.addEventListener('error', (event) => pageErrors.push(`Uncaught error: ${event.message}`));
window.addEventListener('unhandledrejection', (event) => pageErrors.push(`Unhandled rejection: ${String(event.reason)}`));

try {
    const canvas = document.getElementById('view') as HTMLCanvasElement;
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = `${WIDTH}px`;
    canvas.style.height = `${HEIGHT}px`;
    const core = await ViewerCore.create(canvas);
    core.resize(WIDTH, HEIGHT);
    window.addEventListener('pagehide', () => core.dispose());
    const manifest = loadAllManifests().find((candidate) => candidate.id === SCENE);
    if (!manifest) throw new Error(`no manifest ${SCENE}`);
    core.layout = { left: 'none', right: null, giMode: 'only', split: 0.5 };
    core.configureDdgi({ ...DEFAULT_DDGI_VIEWER_SETTINGS });
    await core.loadScene(manifest, () => {});
    const bookmark = manifest.cameras[0];
    core.camera.position.set(...bookmark.position);
    core.camera.fov = bookmark.fovYDeg;
    core.camera.lookAt(...bookmark.target);
    core.camera.updateProjectionMatrix();
    core.camera.updateMatrixWorld(true);
    core.setLayout({ left: 'ddgi', right: null, giMode: 'only' });
    // The scene's own exposure, as the viewer app sets it (manifest `exposureEv`).
    const exposureEv = manifest.exposureEv ?? 0;
    core.setExposureEv(exposureEv);
    const ddgi = core.ddgi;
    if (!ddgi) throw new Error('DDGI was not created');
    while (!ddgi.running) {
        if (ddgi.failed) throw new Error(`DDGI failed: ${ddgi.failed}`);
        if (performance.now() - startTime > WARM_UP_LIMIT_MS) throw new Error('DDGI did not warm up in time');
        await core.renderThroughLoop();
        await core.pumpPrepare(40);
    }
    for (let frame = 0; frame < SETTLE_FRAMES; frame++) await core.renderThroughLoop();
    const size = core.renderer.getDrawingBufferSize(new Vector2());
    const pictureMean = await ddgi.readRegionMean(0, 0, size.x, size.y);
    logLines.push(`${SCENE} at '${bookmark.name}', GI only, exposure ${exposureEv} EV: picture mean ${pictureMean} (linear, before exposure)`);
    // The last frame stays on the canvas for the screenshot.
    await core.renderThroughLoop();
    const info = core.adapterInfo;
    reportGpuTest({
        status: pageErrors.length === 0 ? 'pass' : 'fail',
        message: pageErrors.length === 0 ? undefined : pageErrors.join(' | '),
        data: { scene: SCENE, camera: bookmark.name, exposureEv, pictureMean, recompiles: ddgi.recompiles },
        deviceErrors: [...core.gpuErrors],
        startTime,
        adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description, device: info.device, features: [...core.device.features].sort(), isFallbackAdapter: false },
        logLines,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), deviceErrors: [], startTime, logLines });
}
