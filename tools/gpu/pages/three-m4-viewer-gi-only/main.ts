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
 *
 * T4.5 (the M4 note's screenshots and costs): `VITE_DDGI_GI_MODE=on` shows
 * the full picture (direct and GI) instead of GI only;
 * `VITE_DDGI_MAX_TEXTURE_SIZE=1024` is T2.17's cap on three's own textures
 * (Bistro on the 6 GB RTX); and the report has, for the record, the probes
 * DDGI updated per frame over the settle frames at the viewer's default
 * budget, and the texture store's memory (`stats.memory.textures`).
 */

import { Vector2 } from 'three';
import { reportGpuTest } from '../../client.ts';
import { ViewerCore } from '../../../../apps/viewer/src/core/viewer-core.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { DEFAULT_DDGI_VIEWER_SETTINGS } from '../../../../apps/viewer/src/techniques/ddgi-technique.ts';

const env = import.meta.env as Record<string, unknown>;
const SCENE = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES.trim() !== '' ? env.VITE_DDGI_SCENES.trim() : 'sponza';
const GI_MODE: 'only' | 'on' = env.VITE_DDGI_GI_MODE === 'on' ? 'on' : 'only';
const MAX_TEXTURE_SIZE = typeof env.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' && env.VITE_DDGI_MAX_TEXTURE_SIZE !== '' ? Number(env.VITE_DDGI_MAX_TEXTURE_SIZE) : undefined;
// The page's own view (the runner's 1280 × 720 window less the browser's frame), so the screenshot shows the whole picture.
const WIDTH = window.innerWidth > 0 ? window.innerWidth : 1280;
const HEIGHT = window.innerHeight > 0 ? window.innerHeight : 720;
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
    core.layout = { left: 'none', right: null, giMode: GI_MODE, split: 0.5 };
    core.configureDdgi({ ...DEFAULT_DDGI_VIEWER_SETTINGS });
    await core.loadScene(manifest, () => {}, MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE });
    const bookmark = manifest.cameras[0];
    core.camera.position.set(...bookmark.position);
    core.camera.fov = bookmark.fovYDeg;
    core.camera.lookAt(...bookmark.target);
    core.camera.updateProjectionMatrix();
    core.camera.updateMatrixWorld(true);
    core.setLayout({ left: 'ddgi', right: null, giMode: GI_MODE });
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
    const probesPerFrame: number[] = [];
    for (let frame = 0; frame < SETTLE_FRAMES; frame++) {
        await core.renderThroughLoop();
        probesPerFrame.push(ddgi.probesUpdated);
    }
    const sortedProbes = [...probesPerFrame].sort((a, b) => a - b);
    const probesPerFrameMedian = sortedProbes[Math.floor(sortedProbes.length / 2)];
    const probesPerFrameMean = probesPerFrame.reduce((sum, value) => sum + value, 0) / probesPerFrame.length;
    const stats = ddgi.statsSnapshot();
    // The snapshot holds the platform's whole MemoryReport; the viewer's stats type names only the parts it draws.
    const memory = stats?.memory as { readonly textures?: number } | undefined;
    const textureStoreBytes = memory?.textures ?? null;
    const size = core.renderer.getDrawingBufferSize(new Vector2());
    const pictureMean = await ddgi.readRegionMean(0, 0, size.x, size.y);
    logLines.push(
        `${SCENE} at '${bookmark.name}', GI ${GI_MODE === 'only' ? 'only' : 'on (full picture)'}, exposure ${exposureEv} EV: picture mean ${pictureMean} (linear, before exposure); ` +
            `${probesPerFrameMedian} probes a frame (median of ${SETTLE_FRAMES}) of ${stats?.probeCount ?? '?'}; texture store ${textureStoreBytes === null ? '?' : (textureStoreBytes / 2 ** 20).toFixed(1)} MB`,
    );
    // The last frame stays on the canvas for the screenshot.
    await core.renderThroughLoop();
    const info = core.adapterInfo;
    reportGpuTest({
        status: pageErrors.length === 0 ? 'pass' : 'fail',
        message: pageErrors.length === 0 ? undefined : pageErrors.join(' | '),
        data: {
            scene: SCENE,
            camera: bookmark.name,
            giMode: GI_MODE,
            maxTextureSize: MAX_TEXTURE_SIZE ?? null,
            exposureEv,
            pictureMean,
            recompiles: ddgi.recompiles,
            probeCount: stats?.probeCount ?? null,
            probesPerFrameMedian,
            probesPerFrameMean,
            budgetMs: DEFAULT_DDGI_VIEWER_SETTINGS.budgetMs,
            textureStoreBytes,
        },
        deviceErrors: [...core.gpuErrors],
        startTime,
        adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description, device: info.device, features: [...core.device.features].sort(), isFallbackAdapter: false },
        logLines,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), deviceErrors: [], startTime, logLines });
}
