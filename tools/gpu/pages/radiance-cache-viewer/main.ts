/**
 * RC2.2: the radiance cache in the viewer, on the viewer's own code
 * (`ViewerCore` and its `rc` technique, apps/viewer/src/techniques/radiance-cache-technique.ts),
 * on `cornell-box`, `two-room` and `sponza` at their first bookmark, 1280 × 720:
 *
 * - it starts (the platform prepares, the scene is copied) and draws GI on,
 *   GI only and GI off;
 * - each of its debug views (probe atlas, probe mask, probe SH; since RC4.2 SSGI occlusion and bent normal) draws;
 * - the split wipe against DDGI draws, DDGI warmed up the viewer's way;
 * - no device error, nothing logged as an error, the technique never fails,
 *   and its platform compiles nothing after `prepare` (`stats.recompiles`).
 *
 * The page ends on Sponza's wipe (radiance cache left, DDGI right, GI on)
 * for the runner's screenshot. The picture's values are checked by the
 * adapter page (radiance-cache-three); this one checks the viewer's wiring.
 *
 * Correctness only (DECISIONS 2026-11-09):
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc2.2 viewer" -- pnpm test:gpu --filter radiance-cache-viewer --gpu nvidia --out bench/screenshots/rc2.2`.
 */

import { reportGpuTest, summarizeAdapterInfo } from '../../client.ts';
import { ViewerCore, type ViewLayout } from '../../../../apps/viewer/src/core/viewer-core.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { RADIANCE_CACHE_SCREEN_VIEWS, RadianceCacheViewerTechnique, type RadianceCacheScreenView } from '../../../../apps/viewer/src/techniques/radiance-cache-technique.ts';
import type { GiMode } from '../../../../apps/viewer/src/scene/layers.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const SCENES = ['cornell-box', 'two-room', 'sponza'];
/** Frames drawn for each view checked. */
const FRAMES_PER_VIEW = 4;
/** How long the techniques get to start, per scene. */
const START_TIMEOUT_MS = 180000;

const startTime = performance.now();
const logLines: string[] = [];
const log = (message: string): void => {
    logLines.push(message);
};
const consoleErrors: string[] = [];
const originalError = console.error.bind(console);
console.error = (...args: unknown[]) => {
    consoleErrors.push(args.map(String).join(' '));
    originalError(...args);
};

async function drawFrames(core: ViewerCore, frames: number): Promise<void> {
    for (let frame = 0; frame < frames; frame++) await core.renderThroughLoop();
    await core.device.queue.onSubmittedWorkDone();
}

/** Sets the layout and runs set-up steps (DDGI's warm-up) until every technique on screen is ready. */
async function showReady(core: ViewerCore, layout: Partial<ViewLayout>): Promise<void> {
    core.setLayout(layout);
    const started = performance.now();
    while (core.preparing) {
        if (performance.now() - started > START_TIMEOUT_MS) throw new Error(`${core.loaded?.manifest.id}: the techniques on screen did not start in ${START_TIMEOUT_MS / 1000} s`);
        await core.pumpPrepare(50);
        await core.renderThroughLoop();
    }
}

const failures: string[] = [];
const data: Record<string, unknown> = {};
let core: ViewerCore | null = null;
try {
    const canvas = document.createElement('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = `${WIDTH}px`;
    canvas.style.height = `${HEIGHT}px`;
    document.body.style.margin = '0';
    document.body.append(canvas);
    core = await ViewerCore.create(canvas);
    core.resize(WIDTH, HEIGHT);
    const manifests = loadAllManifests();

    for (const sceneId of SCENES) {
        const manifest = manifests.find((candidate) => candidate.id === sceneId);
        if (!manifest) throw new Error(`no manifest '${sceneId}'`);
        core.layout = { left: 'rc', right: null, giMode: 'on', split: 0.5 };
        core.configureRadianceCache({ screenView: null });
        await core.loadScene(manifest, () => {});
        const bookmark = manifest.cameras[0];
        core.camera.position.set(...bookmark.position);
        core.camera.fov = bookmark.fovYDeg;
        core.camera.lookAt(...bookmark.target);
        core.camera.updateProjectionMatrix();
        core.camera.updateMatrixWorld(true);
        const errorsBefore = core.gpuErrors.length;

        await showReady(core, { left: 'rc', right: null, giMode: 'on' });
        const technique = core.getTechnique('rc');
        if (!(technique instanceof RadianceCacheViewerTechnique)) throw new Error('the rc technique is not the radiance cache');
        const views: string[] = [];
        for (const mode of ['on', 'only', 'off', 'on'] as GiMode[]) {
            core.setLayout({ giMode: mode });
            await drawFrames(core, FRAMES_PER_VIEW);
            views.push(`gi ${mode}`);
        }
        for (const view of Object.keys(RADIANCE_CACHE_SCREEN_VIEWS) as RadianceCacheScreenView[]) {
            core.configureRadianceCache({ screenView: view });
            await drawFrames(core, FRAMES_PER_VIEW);
            views.push(view);
        }
        core.configureRadianceCache({ screenView: null });
        await showReady(core, { left: 'rc', right: 'ddgi', giMode: 'on', split: 0.5 });
        await drawFrames(core, FRAMES_PER_VIEW);
        views.push('wipe rc | ddgi');

        const newErrors = core.gpuErrors.slice(errorsBefore);
        if (newErrors.length > 0) failures.push(`${sceneId}: device errors: ${newErrors.slice(0, 3).join(' | ')}`);
        if (technique.failed) failures.push(`${sceneId}: the radiance cache failed: ${technique.failed}`);
        // Every view above drew after prepare: none may have compiled a compute pipeline on first use.
        const recompiles = technique.recompiles;
        if (recompiles !== 0) failures.push(`${sceneId}: the radiance cache's platform compiled ${recompiles ?? 'unknown'} pipelines after prepare (stats.recompiles)`);
        const status = core.statusLines();
        data[sceneId] = { views, status, gpuErrors: newErrors.length, recompiles };
        log(`${sceneId}: ${status.join(' / ')}`);
        // Leave Sponza's wipe on screen for the screenshot.
        if (sceneId !== SCENES[SCENES.length - 1]) core.unloadScene();
    }
    if (consoleErrors.length > 0) failures.push(`logged errors: ${consoleErrors.slice(0, 3).join(' | ')}`);
    reportGpuTest({
        status: failures.length === 0 ? 'pass' : 'fail',
        message: failures.length === 0 ? undefined : failures.join(' | '),
        data,
        adapter: summarizeAdapterInfo(core.device.adapterInfo, core.device.features),
        deviceErrors: core.gpuErrors,
        startTime,
        logLines,
    });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? (error.stack ?? error.message) : String(error), deviceErrors: core?.gpuErrors ?? [], startTime, logLines });
}
