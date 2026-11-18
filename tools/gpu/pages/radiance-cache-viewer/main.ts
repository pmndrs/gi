/**
 * RC2.2: the radiance cache in the viewer, on the viewer's own code
 * (`ViewerCore` and its `rc` technique, apps/viewer/src/techniques/radiance-cache-technique.ts),
 * on `cornell-box`, `two-room` and `sponza` at their first bookmark, 1280 × 720:
 *
 * - it starts (the platform prepares, the scene is copied) and draws GI on,
 *   GI only and GI off;
 * - each of its debug views (probe atlas, probe mask, probe SH) draws;
 * - the split wipe against DDGI draws, DDGI warmed up the viewer's way;
 * - no device error, nothing logged as an error, the technique never fails;
 * - with GI on from the first frame after it starts, its per-pixel output
 *   (the interpolated irradiance E, which the composite adds as
 *   `giAlbedo × E / π`) and its probe atlas hold no NaN, infinite or negative
 *   value after frames 1, 30 and 60 (RC3.2's deferred check, through the
 *   viewer's own path: the world cache lights the probe rays since RC3.2).
 *
 * The page ends on the last scene's radiance cache, GI on, for the runner's
 * screenshot. The composited picture's values are checked by the adapter
 * page (radiance-cache-three) too; this one checks the viewer's wiring.
 *
 * `VITE_RC_VIEWER_SCENES` picks the scenes (comma-separated; default all
 * three), so a run per scene leaves a screenshot per scene.
 *
 * Correctness only (DECISIONS 2026-11-09), through the lock, e.g.:
 * `VITE_RC_VIEWER_SCENES=sponza node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-viewer --out bench/screenshots/rc2.2/sponza`.
 */

import { reportGpuTest, summarizeAdapterInfo } from '../../client.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { ViewerCore, type ViewLayout } from '../../../../apps/viewer/src/core/viewer-core.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { RADIANCE_CACHE_SCREEN_VIEWS, RadianceCacheViewerTechnique, type RadianceCacheScreenView } from '../../../../apps/viewer/src/techniques/radiance-cache-technique.ts';
import type { GiMode } from '../../../../apps/viewer/src/scene/layers.ts';

const WIDTH = 1280;
const HEIGHT = 720;
const env = import.meta.env as Record<string, unknown>;
const ALL_SCENES = ['cornell-box', 'two-room', 'sponza'];
const SCENES = typeof env.VITE_RC_VIEWER_SCENES === 'string' && env.VITE_RC_VIEWER_SCENES.trim() !== '' ? env.VITE_RC_VIEWER_SCENES.split(',').map((id) => id.trim()) : ALL_SCENES;
/** Frames after the start, GI on, at which the output is scanned for NaN, infinite and negative values. */
const SCAN_FRAMES = [1, 30, 60];
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

/** NaN, infinite and negative values in a texture of four halves a texel (rgba16float); alpha is checked for finite only. */
async function scanHalves(device: GPUDevice, texture: GPUTexture): Promise<{ nonFinite: number; negative: number; meanRgb: number }> {
    const values = halves(await readTexture(device, texture, 8, texture.width, texture.height));
    let nonFinite = 0;
    let negative = 0;
    let sum = 0;
    for (let index = 0; index < values.length; index++) {
        const value = values[index];
        if (!Number.isFinite(value)) nonFinite++;
        else if (index % 4 !== 3) {
            if (value < 0) negative++;
            sum += value;
        }
    }
    return { nonFinite, negative, meanRgb: sum / ((values.length / 4) * 3) };
}

const failures: string[] = [];
const data: Record<string, unknown> = {};
let core: ViewerCore | null = null;
try {
    const canvas = document.createElement('canvas');
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    // Drawn at 1280 × 720 (resize below); shown scaled to fit the runner's window, so the screenshot holds all of it.
    canvas.style.display = 'block';
    canvas.style.width = `min(100vw, calc(100vh * ${WIDTH} / ${HEIGHT}))`;
    canvas.style.aspectRatio = `${WIDTH} / ${HEIGHT}`;
    document.body.style.margin = '0';
    document.body.style.overflow = 'hidden';
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

        // The scan: frames 1, 30 and 60 after the start, GI on.
        const scans: Record<string, unknown>[] = [];
        let drawn = 0;
        for (const frame of SCAN_FRAMES) {
            await drawFrames(core, frame - drawn);
            drawn = frame;
            const probes = technique.radianceCache?.probes;
            if (!probes) {
                failures.push(`${sceneId}: no probes after frame ${frame}`);
                continue;
            }
            const irradiance = await scanHalves(core.device, probes.irradiance);
            const atlas = await scanHalves(core.device, probes.probeAtlas);
            scans.push({ frame, frameIndex: probes.frameIndex, irradiance, probeAtlas: atlas });
            for (const [name, scan] of [['irradiance', irradiance], ['probe atlas', atlas]] as const) {
                if (scan.nonFinite > 0 || scan.negative > 0) failures.push(`${sceneId} frame ${frame}: the ${name} has ${scan.nonFinite} values not finite and ${scan.negative} negative`);
            }
            if (!(irradiance.meanRgb > 0)) failures.push(`${sceneId} frame ${frame}: no irradiance anywhere`);
        }

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
        const status = core.statusLines();
        data[sceneId] = { views, scans, status, gpuErrors: newErrors.length };
        log(`${sceneId}: ${status.join(' / ')}`);
        if (sceneId !== SCENES[SCENES.length - 1]) {
            core.unloadScene();
        } else {
            // The radiance cache alone, GI on, for the runner's screenshot.
            core.setLayout({ left: 'rc', right: null, giMode: 'on' });
            await drawFrames(core, FRAMES_PER_VIEW);
        }
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
