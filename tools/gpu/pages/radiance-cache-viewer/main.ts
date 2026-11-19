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
 *   viewer's own path: the world cache lights the probe rays since RC3.2);
 * - per pixel, on one more frame drawn by hand (the viewer's loop stopped so
 *   every readback is that frame's): the technique's GI picture minus its
 *   scene pass's own colour equals `giAlbedo × E / π` with E the irradiance
 *   image's texel at the same pixel (RC2.2 review: a flipped or offset
 *   `screenUV` fails).
 *
 * - RC4.5, the compare toggles, through the panel's own section
 *   (`RcCompareControls`, apps/viewer/src/ui/rc-compare-controls.ts) shown
 *   beside the picture and clicked as a user would (checkbox clicks, select
 *   and slider change events): each toggle set to Capsaicin's value (or, where
 *   that is ours, to its other value) from ours, `COMPARE_FRAMES` frames
 *   drawn, then back to ours; and "All Capsaicin". Each must reach the core
 *   technique's settings, leave no NaN, infinite or negative value in the
 *   irradiance and probe atlas, and log no error; the change of the picture
 *   against ours is recorded beside ours against itself a few frames later
 *   (the noise), and "All Capsaicin" must change it by more than that noise.
 *   The panel's readout (GI change per frame, frames since a toggle changed)
 *   is recorded too.
 *
 * The page ends on the last scene's radiance cache, GI on, for the runner's
 * screenshot, with the compare section beside it: at ours, or with
 * `VITE_RC_VIEWER_END=capsaicin` at every Capsaicin value (a before/after
 * pair: two runs). The composited picture's values are checked by the adapter
 * page (radiance-cache-three) too; this one checks the viewer's wiring.
 *
 * `VITE_RC_VIEWER_SCENES` picks the scenes (comma-separated; default all
 * three), so a run per scene leaves a screenshot per scene. `VITE_RC_VIEWER_COMPARE=0`
 * skips the compare toggles.
 *
 * Correctness only (DECISIONS 2026-11-09), through the lock, e.g.:
 * `VITE_RC_VIEWER_SCENES=sponza node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-viewer --out bench/screenshots/rc2.2/sponza`.
 */

import { reportGpuTest, summarizeAdapterInfo } from '../../client.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { HdrReader } from '../three-composite-shared/composite-check.ts';
import { isBackgroundDepth, uv, vec4 } from 'three/tsl';
import type { Node, WebGPURenderer } from 'three/webgpu';
import { GI_ALBEDO_OUTPUT } from '../../../../packages/ddgi-three/src/composite.ts';
import { ViewerCore, type ViewLayout } from '../../../../apps/viewer/src/core/viewer-core.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { RADIANCE_CACHE_SCREEN_VIEWS, RadianceCacheViewerTechnique, type RadianceCacheScreenView } from '../../../../apps/viewer/src/techniques/radiance-cache-technique.ts';
import type { GiMode } from '../../../../apps/viewer/src/scene/layers.ts';
import { RcCompareControls } from '../../../../apps/viewer/src/ui/rc-compare-controls.ts';
import { compareSettingValue, RADIANCE_CACHE_COMPARE_TOGGLES, type CompareToggle, type CompareValues } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/compare-toggles.ts';

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
/** RC4.5: frames drawn after each toggle before its picture is read, and between the two reads of ours (the noise). */
const COMPARE_FRAMES = 30;
const COMPARE = env.VITE_RC_VIEWER_COMPARE !== '0';
const END_AT_CAPSAICIN = env.VITE_RC_VIEWER_END === 'capsaicin';

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

/** The composite check's tolerances: a share of the expected GI, a share of the picture (f32 sums of half inputs), and the pixels allowed past them. */
const COMPOSITE_RELATIVE_TOLERANCE = 1e-3;
const COMPOSITE_PICTURE_TOLERANCE = 1e-4;
const COMPOSITE_OUTLIER_FRACTION = 0.001;

/**
 * One frame drawn by hand with the viewer's loop stopped: the technique's GI
 * picture, its pass's colour and giAlbedo, and the irradiance image, all of
 * that frame; then per pixel, picture − pass colour against giAlbedo × E / π.
 */
async function checkComposite(core: ViewerCore, technique: RadianceCacheViewerTechnique): Promise<{ pixels: number; outliers: number; worstRelative: number }> {
    const nodes = technique.giNodes;
    const probes = () => technique.radianceCache?.probes ?? null;
    if (!nodes || !probes()) throw new Error('the radiance cache has no GI pipeline to check');
    const renderer = core.renderer as unknown as WebGPURenderer;
    const loop = (core as unknown as { tick(time: number): void }).tick.bind(core);
    renderer.setAnimationLoop(null);
    const reader = new HdrReader(renderer, WIDTH, HEIGHT);
    try {
        technique.renderBeforePipeline();
        const picture = await reader.read(nodes.output, true);
        const passColor = await reader.read(nodes.scenePass.getTextureNode('output') as unknown as Node, false);
        const albedo = await reader.read(nodes.scenePass.getTextureNode(GI_ALBEDO_OUTPUT) as unknown as Node, false);
        const surfaceNode = vec4(isBackgroundDepth(nodes.scenePass.getTextureNode('depth').sample(uv()).r).select(0, 1), 0, 0, 1) as unknown as Node;
        const surfaces = await reader.read(surfaceNode, false);
        const irradiance = halves(await readTexture(core.device, probes()!.irradiance, 8, WIDTH, HEIGHT));
        let pixels = 0;
        let outliers = 0;
        let worstRelative = 0;
        for (let pixel = 0; pixel < WIDTH * HEIGHT; pixel++) {
            if (surfaces[pixel * 4] < 0.5) continue;
            if (!(albedo[pixel * 4] + albedo[pixel * 4 + 1] + albedo[pixel * 4 + 2] > 0)) continue;
            pixels++;
            let bad = false;
            for (let channel = 0; channel < 3; channel++) {
                const index = pixel * 4 + channel;
                const expected = (albedo[index] * irradiance[index]) / Math.PI;
                const error = Math.abs(picture[index] - passColor[index] - expected);
                if (error > COMPOSITE_RELATIVE_TOLERANCE * expected + COMPOSITE_PICTURE_TOLERANCE * Math.abs(picture[index]) + 1e-6) bad = true;
                if (expected > 1e-4) worstRelative = Math.max(worstRelative, error / expected);
            }
            if (bad) outliers++;
        }
        return { pixels, outliers, worstRelative };
    } finally {
        reader.dispose();
        renderer.setAnimationLoop(loop);
    }
}

/** The compare section beside the picture, wired to the viewer core as the viewer app wires it. */
function showCompareControls(core: ViewerCore): RcCompareControls {
    const controls = new RcCompareControls({ onCompare: (values: CompareValues) => core.configureRadianceCache({ screenView: null, compare: values }) });
    // Beside the picture, as the viewer's panel is: fixed, never wider than the window, its own vertical scroll.
    const style = document.createElement('style');
    style.textContent = `
        html, body { overflow: hidden; background: #0b0d10; }
        .rc-compare-box { position: fixed; top: 0; right: 0; bottom: 0; width: min(300px, 100vw); box-sizing: border-box; overflow-y: auto; overflow-x: hidden;
            padding: 6px 10px; background: #10141a; color: #d8dee6; font: 12px/1.4 system-ui, sans-serif; }
        .rc-compare-box h2 { font-size: 13px; margin: 4px 0 6px; }
        .rc-compare-box select, .rc-compare-box input[type='range'], .rc-compare-box button { width: 100%; box-sizing: border-box; }
        .rc-compare-box .button-row { display: flex; gap: 4px; }
        .rc-compare-box .label { margin: 6px 0 2px; color: #8f9bab; }
        .rc-compare-box label.inline { display: inline-flex; align-items: center; gap: 4px; margin: 2px 0; }
        .rc-compare-box .muted { color: #8f9bab; }
        .rc-compare-box .changed { border-left: 2px solid #4da3ff; padding-left: 4px; }`;
    document.head.append(style);
    const box = document.createElement('div');
    box.className = 'rc-compare-box';
    box.append(controls.section);
    document.body.append(box);
    return controls;
}

/** The value a toggle is clicked to: Capsaicin's, or where that is ours, the other one. */
function compareValueOf(entry: CompareToggle): boolean | number | string {
    if (entry.capsaicin !== undefined && entry.capsaicin !== entry.defaultValue) return entry.capsaicin;
    if (entry.kind === 'boolean') return !entry.defaultValue;
    if (entry.kind === 'number') return entry.defaultValue === entry.min ? entry.max : entry.min;
    return entry.choices.find((choice) => choice.value !== entry.defaultValue)!.value;
}

/** Sets a toggle in the compare section the way a user does: a click on a checkbox, a change of a select or slider. */
function clickToggle(section: HTMLElement, entry: CompareToggle, value: boolean | number | string): void {
    const row = section.querySelector<HTMLElement>(`[data-compare-setting="${entry.setting}"]`);
    const control = row?.querySelector<HTMLInputElement | HTMLSelectElement>('input, select');
    if (!row || !control) throw new Error(`the compare section has no control for ${entry.setting}`);
    if (control instanceof HTMLInputElement && control.type === 'checkbox') {
        if (control.checked !== value) control.click();
        return;
    }
    control.value = String(value);
    control.dispatchEvent(new Event('input', { bubbles: true }));
    control.dispatchEvent(new Event('change', { bubbles: true }));
}

function clickButton(section: HTMLElement, name: 'all-capsaicin' | 'all-ours'): void {
    const button = section.querySelector<HTMLButtonElement>(`button[data-compare="${name}"]`);
    if (!button) throw new Error(`the compare section has no ${name} button`);
    button.click();
}

/** The interpolated irradiance's mean over rgb per pixel. */
async function irradianceImage(core: ViewerCore, technique: RadianceCacheViewerTechnique): Promise<Float32Array> {
    const probes = technique.radianceCache?.probes;
    if (!probes) throw new Error('no probes');
    const values = halves(await readTexture(core.device, probes.irradiance, 8, probes.irradiance.width, probes.irradiance.height));
    const image = new Float32Array(values.length / 4);
    for (let pixel = 0; pixel < image.length; pixel++) image[pixel] = (values[pixel * 4] + values[pixel * 4 + 1] + values[pixel * 4 + 2]) / 3;
    return image;
}

/** Sum |a - b| / sum a over pixels, and sum b / sum a. */
function imageChange(a: Float32Array, b: Float32Array): { change: number; ratio: number } {
    let difference = 0;
    let sumA = 0;
    let sumB = 0;
    for (let pixel = 0; pixel < a.length; pixel++) {
        difference += Math.abs(a[pixel] - b[pixel]);
        sumA += a[pixel];
        sumB += b[pixel];
    }
    return { change: Number((difference / Math.max(sumA, 1e-30)).toFixed(4)), ratio: Number((sumB / Math.max(sumA, 1e-30)).toFixed(4)) };
}

/** RC4.5: every compare toggle clicked in the panel section, its picture against ours (the page comment). */
async function checkCompareToggles(core: ViewerCore, controls: RcCompareControls, sceneId: string): Promise<Record<string, unknown>> {
    const section = controls.section;
    // The radiance cache on screen: a toggle read at prepare (the world-cache backend) makes a new one.
    const rc = (): RadianceCacheViewerTechnique => core.getTechnique('rc') as RadianceCacheViewerTechnique;
    const valueOf = (entry: CompareToggle): unknown => {
        const settings = rc().radianceCache?.settings;
        return settings ? compareSettingValue(settings, entry.setting) : undefined;
    };
    // After a click: a restarted radiance cache prepares again before frames are drawn.
    const settle = async (): Promise<void> => {
        await showReady(core, { left: 'rc', right: null, giMode: 'on' });
        if (rc().failed) failures.push(`${sceneId}: the radiance cache failed: ${rc().failed}`);
    };
    clickButton(section, 'all-ours');
    await drawFrames(core, COMPARE_FRAMES);
    const ours = await irradianceImage(core, rc());
    await drawFrames(core, COMPARE_FRAMES);
    const noise = imageChange(ours, await irradianceImage(core, rc()));
    const results: Record<string, unknown> = { noise };
    const scan = async (name: string): Promise<void> => {
        const probes = rc().radianceCache?.probes;
        if (!probes) throw new Error('no probes');
        for (const [label, texture] of [['irradiance', probes.irradiance], ['probe atlas', probes.probeAtlas]] as const) {
            const values = await scanHalves(core.device, texture);
            if (values.nonFinite > 0 || values.negative > 0) failures.push(`${sceneId}, ${name}: the ${label} has ${values.nonFinite} values not finite and ${values.negative} negative`);
        }
    };
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
        const value = compareValueOf(entry);
        clickToggle(section, entry, value);
        await settle();
        if (valueOf(entry) !== value) failures.push(`${sceneId}: clicking ${entry.setting} to ${String(value)} left the technique at ${String(valueOf(entry))}`);
        await drawFrames(core, COMPARE_FRAMES);
        const image = await irradianceImage(core, rc());
        await scan(`${entry.setting} = ${String(value)}`);
        results[`${entry.setting}=${String(value)}`] = { ...imageChange(ours, image), readout: rc().changeReadout };
        clickToggle(section, entry, entry.defaultValue);
        await settle();
        if (valueOf(entry) !== entry.defaultValue) failures.push(`${sceneId}: clicking ${entry.setting} back left the technique at ${String(valueOf(entry))}`);
        await drawFrames(core, COMPARE_FRAMES);
    }
    clickButton(section, 'all-capsaicin');
    await settle();
    for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
        if (entry.capsaicin !== undefined && valueOf(entry) !== entry.capsaicin) failures.push(`${sceneId}: "All Capsaicin" left ${entry.setting} at ${String(valueOf(entry))}`);
    }
    await drawFrames(core, COMPARE_FRAMES);
    const capsaicin = imageChange(ours, await irradianceImage(core, rc()));
    await scan('all Capsaicin');
    results.allCapsaicin = { ...capsaicin, readout: rc().changeReadout };
    if (!(capsaicin.change > 2 * noise.change)) failures.push(`${sceneId}: "All Capsaicin" changed the picture by ${capsaicin.change} of ours, not over twice the noise ${noise.change}`);
    clickButton(section, 'all-ours');
    await settle();
    await drawFrames(core, COMPARE_FRAMES);
    return results;
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
    // Left of the compare section (300 px), as the viewer draws beside its panel.
    canvas.style.width = `min(calc(100vw - min(300px, 100vw)), calc(100vh * ${WIDTH} / ${HEIGHT}))`;
    canvas.style.aspectRatio = `${WIDTH} / ${HEIGHT}`;
    document.body.style.margin = '0';
    document.body.style.overflow = 'hidden';
    document.body.append(canvas);
    core = await ViewerCore.create(canvas);
    core.resize(WIDTH, HEIGHT);
    const compareControls = showCompareControls(core);
    const manifests = loadAllManifests();

    for (const sceneId of SCENES) {
        const manifest = manifests.find((candidate) => candidate.id === sceneId);
        if (!manifest) throw new Error(`no manifest '${sceneId}'`);
        core.layout = { left: 'rc', right: null, giMode: 'on', split: 0.5 };
        core.configureRadianceCache({ screenView: null, compare: {} });
        await core.loadScene(manifest, () => {});
        const bookmark = manifest.cameras[0];
        core.camera.position.set(...bookmark.position);
        core.camera.fov = bookmark.fovYDeg;
        core.camera.lookAt(...bookmark.target);
        core.camera.updateProjectionMatrix();
        core.camera.updateMatrixWorld(true);
        const errorsBefore = core.gpuErrors.length;

        await showReady(core, { left: 'rc', right: null, giMode: 'on' });
        const started = core.getTechnique('rc');
        if (!(started instanceof RadianceCacheViewerTechnique)) throw new Error('the rc technique is not the radiance cache');
        let technique: RadianceCacheViewerTechnique = started;

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

        const composite = await checkComposite(core, technique);
        if (!(composite.pixels > (WIDTH * HEIGHT) / 4)) failures.push(`${sceneId}: only ${composite.pixels} pixels for the composite check`);
        if (composite.outliers > COMPOSITE_OUTLIER_FRACTION * composite.pixels) {
            failures.push(`${sceneId}: on ${composite.outliers} of ${composite.pixels} pixels the picture minus the pass colour isn't giAlbedo x E / pi of the same pixel`);
        }

        const views: string[] = [];
        for (const mode of ['on', 'only', 'off', 'on'] as GiMode[]) {
            core.setLayout({ giMode: mode });
            await drawFrames(core, FRAMES_PER_VIEW);
            views.push(`gi ${mode}`);
        }
        for (const view of Object.keys(RADIANCE_CACHE_SCREEN_VIEWS) as RadianceCacheScreenView[]) {
            core.configureRadianceCache({ screenView: view, compare: {} });
            await drawFrames(core, FRAMES_PER_VIEW);
            views.push(view);
        }
        core.configureRadianceCache({ screenView: null, compare: {} });
        const compare = COMPARE ? await checkCompareToggles(core, compareControls, sceneId) : null;
        // A restart toggle made a new radiance cache: check that one from here on.
        technique = core.getTechnique('rc') as RadianceCacheViewerTechnique;
        await showReady(core, { left: 'rc', right: 'ddgi', giMode: 'on', split: 0.5 });
        await drawFrames(core, FRAMES_PER_VIEW);
        views.push('wipe rc | ddgi');

        const newErrors = core.gpuErrors.slice(errorsBefore);
        if (newErrors.length > 0) failures.push(`${sceneId}: device errors: ${newErrors.slice(0, 3).join(' | ')}`);
        if (technique.failed) failures.push(`${sceneId}: the radiance cache failed: ${technique.failed}`);
        const status = core.statusLines();
        data[sceneId] = { views, scans, composite, compare, status, gpuErrors: newErrors.length };
        log(`${sceneId}: ${status.join(' / ')}`);
        if (sceneId !== SCENES[SCENES.length - 1]) {
            core.unloadScene();
        } else {
            // The radiance cache alone, GI on, for the runner's screenshot: at ours, or at Capsaicin's values.
            core.setLayout({ left: 'rc', right: null, giMode: 'on' });
            clickButton(compareControls.section, END_AT_CAPSAICIN ? 'all-capsaicin' : 'all-ours');
            await drawFrames(core, END_AT_CAPSAICIN ? 2 * COMPARE_FRAMES : FRAMES_PER_VIEW);
            compareControls.setReadout(technique.changeReadout);
            data.endedAt = END_AT_CAPSAICIN ? 'all Capsaicin' : 'ours';
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
