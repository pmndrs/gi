/**
 * T5.2's done-when (docs/design/m5-coverage.md §2): saved bakes.
 *
 * Per scene (Cornell, two-room), DDGI with its default settings (the
 * technique's own warm-up on):
 * 1. **Save:** the field is warmed up and converged (CONVERGE_FRAMES more
 *    frames), the test view's indirect mean at the scene's first bookmark is
 *    read (the saved page's mean), and `saveField` writes the file.
 * 2. **Load into a fresh platform:** a new platform on the same scene, with
 *    the same default settings (so its own warm-up would run if the load
 *    didn't skip it), loads the file right after `prepare`. Checks: the
 *    load isn't stale; the first frame that updates probes (after the trees
 *    are built) gives a test-view mean within 1% of the saved page's; no
 *    warm-up pass ran ("warm-up passes" stays 0 and the technique isn't
 *    warming once the trees are built); and `stats.recompiles` doesn't move
 *    from before the load to the end (the test view's own kernel is
 *    compiled before the load).
 * 3. **A layout mismatch is refused:** a platform with 8 x 8 irradiance
 *    texels instead of 6 refuses the file with a `SavedFieldError` that
 *    names `irradianceTexels`.
 * 4. **A changed scene loads stale and settles:** the scene changed (Cornell:
 *    a point light added under the ceiling, as its `light: 'point'` mode
 *    has it; two-room: the lamp at half intensity)
 *    loads the file as stale, with the fingerprint part that changed named.
 *    Its view's mean must come within 5% of the changed scene's settled
 *    mean, and stay there, within the usual response: the same change made
 *    live on the saved platform, measured the same way (frames until within
 *    5% and staying), times STALE_RESPONSE_FACTOR plus STALE_RESPONSE_SLACK
 *    frames (DECISIONS 2026-11-10, "Saved bakes"). The settled mean is the
 *    live run's after SETTLE_FRAMES more frames.
 *
 * And a camera-following volume (two-room, the manifest's fixed volume plus
 * a 7 x 5 x 7 window at priority 1, the camera at the lit-room bookmark):
 * its window is saved and put back, so the fresh platform's first frame
 * doesn't move it (no teleport, no warm-up) and its mean is within 1%.
 *
 * Every frame has the bookmark's camera (the scheduler and the window
 * follow it) and a 60 Hz `deltaTime`. Diagnostics are logged beside the
 * checks: each page's mean history sample count, the saved page's own next
 * frame (how much one update moves a converged field), the loaded page's
 * next 30 frames and its mark and feedback totals (a still scene marks
 * nothing after a load), and each response's series. `pnpm test:gpu --filter ddgi-saved-bake`.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { InstanceHandle, LightHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { SavedFieldError } from '../../../../packages/ddgi-core/src/techniques/ddgi/saved-field.ts';
import type { DDGISettingsInput, DDGIVolumeSettings } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import type { ProceduralScene } from '../../../../scenes/src/scene-data.ts';
import { describeScene, imageMean, loadManifest, releaseDeviceOnPageHide, TestView } from '../ddgi-shared/harness.ts';
import { bookmarkCamera, readHistory, runCameraFrame } from '../ddgi-shared/response.ts';

const WIDTH = 320;
const HEIGHT = 180;
/** Frames at the case's batch after the warm-up, before the save: the field's histories reach their steady counts. */
const CONVERGE_FRAMES = 300;
/** The first-frame bar: the loaded page's test-view mean within 1% of the saved page's. */
const FIRST_FRAME_TOLERANCE = 0.01;
/** "Settled": the view's mean within 5% of the settled mean, as T2.4a's response checks count it. */
const SETTLED_TOLERANCE = 0.05;
/** Frames the responses are followed for, and the live run's frames after that to its settled mean. */
const RESPONSE_FRAMES = 240;
const SETTLE_FRAMES = 600;
/** The stale load settles within the usual response: at most this times the live change's frames, plus this many (DECISIONS 2026-11-10). */
const STALE_RESPONSE_FACTOR = 1.5;
const STALE_RESPONSE_SLACK = 10;

/** A scene edit made live (on the saved platform) or before the load (on the stale one). */
type SceneChange = (scene: BuiltScene) => void;

interface Case {
    readonly id: string;
    readonly scene: string;
    /** Probes a frame after the warm-up (the RTX's calibrated batch at 2 ms, as `ddgi-change-still` uses). */
    readonly units: number;
    readonly settings?: Partial<DDGISettingsInput>;
    /** The changed scene for check 4, and the fingerprint part it changes; none for the camera-volume case. */
    readonly change?: { readonly label: string; readonly part: string; readonly apply: SceneChange };
}

interface BuiltScene {
    readonly platform: GIPlatformImpl;
    readonly ddgi: DDGITechnique;
    readonly manifest: SceneManifest;
    readonly data: ProceduralScene;
    readonly lights: LightHandle[];
    readonly instances: Map<string, InstanceHandle>;
}

/**
 * Cornell with a point light added under the middle of the ceiling, beside
 * its emitter (scenes/src/procedural/cornell-box.ts, `light: 'point'`). A
 * big change of the view, which `ddgi/mark` marks when made live. (Moving a
 * block changed the view's mean by 1.5%, inside the 5% bar from the start.)
 */
const addCornellLight: SceneChange = (scene) => {
    scene.lights.push(scene.platform.scene.createLight({ type: 'point', position: [2.775, 5.188, 2.796], color: [1, 0.72, 0.28], intensity: 15, range: 20, castsShadows: true }));
};

/** Two-room's lamp at half its manifest intensity. */
const halveLamp: SceneChange = (scene) => {
    const lamp = scene.manifest.lights[0];
    scene.platform.scene.updateLight(scene.lights[0], { intensity: lamp.intensity / 2 });
};

/** A camera-following window over two-room, 7 x 5 x 7 probes at 0.5 m, starting at the manifest's volume centre (the camera is 3.7 m away in x). */
function cameraVolumes(manifest: SceneManifest): DDGIVolumeSettings[] {
    const fixed: DDGIVolumeSettings = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
    return [fixed, { center: manifest.volume.center, size: [3, 2, 3], spacing: 0.5, priority: 1, follow: 'camera' }];
}

const CASES: readonly Case[] = [
    { id: 'cornell-box', scene: 'cornell-box', units: 1574, change: { label: 'a point light added', part: 'lights', apply: addCornellLight } },
    { id: 'two-room', scene: 'two-room', units: 707, change: { label: 'the lamp at half intensity', part: 'lights', apply: halveLamp } },
    { id: 'two-room camera volume', scene: 'two-room', units: 707, settings: { volumes: cameraVolumes(loadManifest('two-room')) } },
];

/** A platform with the scene and DDGI at its defaults (the warm-up on), prepared; no frame yet. */
async function buildScene(device: GPUDevice, testCase: Case, settings: Partial<DDGISettingsInput> = {}, change?: SceneChange): Promise<BuiltScene> {
    const manifest = loadManifest(testCase.scene);
    const platform = new GIPlatformImpl(device, { label: `saved bake ${testCase.id}`, gpuTiming: true });
    const lights: LightHandle[] = [];
    const instances = new Map<string, InstanceHandle>();
    const data = describeScene(platform, manifest, {}, lights, [], instances);
    const volume = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
    const ddgi = platform.addTechnique(new DDGITechnique({ volume, ...testCase.settings, ...settings }));
    platform.setOutput(ddgi);
    const scene = { platform, ddgi, manifest, data, lights, instances };
    change?.(scene);
    await platform.prepare();
    return scene;
}

/** Frames at `units` with the first bookmark's camera, waiting for the GPU every few. */
async function frames(device: GPUDevice, scene: BuiltScene, count: number, units: number): Promise<void> {
    const camera = bookmarkCamera(scene.manifest.cameras[0], WIDTH / HEIGHT);
    for (let index = 0; index < count; index++) {
        runCameraFrame(device, scene.platform, units, camera);
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }
    await device.queue.onSubmittedWorkDone();
}

/**
 * The first frame places the scene and starts the trees; then wait for them,
 * and one more frame places them (no probe work: units 0). The test view
 * traces its primary rays through them too.
 */
async function waitForTrees(device: GPUDevice, scene: BuiltScene): Promise<void> {
    await frames(device, scene, 1, 0);
    await scene.platform.rays.whenBuilt();
    await frames(device, scene, 1, 0);
}

/** The test view's indirect mean at the first bookmark. */
async function viewMean(scene: BuiltScene, view: TestView): Promise<number> {
    const field = scene.ddgi.field;
    assert(field, 'no field');
    return imageMean((await view.render(field, scene.manifest.cameras[0], WIDTH, HEIGHT)).indirect);
}

/** The mean of the first volume's probes' history sample counts (active probes only). */
async function meanSamples(device: GPUDevice, scene: BuiltScene): Promise<number> {
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const records = await readHistory(device, field);
    const counted = records.filter((record) => record.samples > 0);
    return counted.reduce((sum, record) => sum + record.samples, 0) / Math.max(1, counted.length);
}

/** The view's mean after each of `count` frames. */
async function responseSeries(device: GPUDevice, scene: BuiltScene, view: TestView, count: number, units: number): Promise<number[]> {
    const means: number[] = [];
    for (let frame = 0; frame < count; frame++) {
        await frames(device, scene, 1, units);
        means.push(await viewMean(scene, view));
    }
    return means;
}

/** Frames (1-based) until the series comes within `tolerance` of `settled` and stays; null if it never does. */
function framesToSettle(series: readonly number[], settled: number, tolerance: number): number | null {
    for (let index = series.length - 1; index >= 0; index--) {
        if (Math.abs(series[index] / settled - 1) > tolerance) return index + 1 < series.length ? index + 2 : null;
    }
    return 1;
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, unknown> = {};
        const failures: string[] = [];
        for (const testCase of CASES) {
            const fail = (what: string): void => {
                failures.push(`${testCase.id}: ${what}`);
                log(`  FAIL ${what}`);
            };
            // 1. Save a converged field.
            const saved = await buildScene(device, testCase);
            const savedView = await TestView.create(device, saved.platform);
            let warmUpFrames = 0;
            await waitForTrees(device, saved);
            while (saved.ddgi.warmingUp) {
                await frames(device, saved, 1, testCase.units);
                warmUpFrames++;
                assert(warmUpFrames < 10_000, `${testCase.id}: the saved page's warm-up never ended`);
            }
            await frames(device, saved, CONVERGE_FRAMES, testCase.units);
            const savedMean = await viewMean(saved, savedView);
            const saveStart = performance.now();
            const bytes = await saved.ddgi.saveField();
            const saveMs = performance.now() - saveStart;
            const savedSamples = await meanSamples(device, saved);
            // The control: the saved page's own next frame, one update later from the same field.
            await frames(device, saved, 1, testCase.units);
            const savedNextChange = (await viewMean(saved, savedView)) / savedMean - 1;
            log(`  diagnostics: saved page mean samples ${savedSamples.toFixed(2)}; its next frame moves its mean ${(savedNextChange * 100).toFixed(3)}%`);
            log(`${testCase.id}: saved after a ${warmUpFrames}-frame warm-up and ${CONVERGE_FRAMES} frames; view mean ${savedMean.toPrecision(6)}; ${bytes.byteLength} bytes in ${saveMs.toFixed(1)} ms`);

            // 2. Load into a fresh platform: no warm-up, the first frame's mean within 1%, nothing compiled.
            const fresh = await buildScene(device, testCase);
            const freshView = await TestView.create(device, fresh.platform);
            const recompilesBefore = fresh.platform.stats.recompiles;
            const loadStart = performance.now();
            const loaded = fresh.ddgi.loadField(bytes);
            const loadMs = performance.now() - loadStart;
            await waitForTrees(device, fresh);
            const meanAtLoad = await viewMean(fresh, freshView);
            const loadedSamples = await meanSamples(device, fresh);
            await frames(device, fresh, 1, testCase.units);
            const samplesAfterFirst = await meanSamples(device, fresh);
            log(`  diagnostics: loaded mean samples ${loadedSamples.toFixed(2)}, after the first frame ${samplesAfterFirst.toFixed(2)}`);
            const firstFrameMean = await viewMean(fresh, freshView);
            const afterFirst = fresh.ddgi.counters;
            const warmingAfterFirst = fresh.ddgi.warmingUp;
            // A few more frames, so a recompile or a late warm-up would show.
            const laterMeans = await responseSeries(device, fresh, freshView, 30, testCase.units);
            const afterMore = fresh.ddgi.counters;
            log(
                `  diagnostics: loaded page's mean over 30 more frames ${laterMeans.filter((_, index) => index % 5 === 4).map((mean) => ((mean / savedMean - 1) * 100).toFixed(3) + '%').join(' ')}; ` +
                    ['feedback requests (total)', 'probes marked (total)', 'probes blended (total)', 'probes propagated (total)'].map((name) => `${name} ${afterMore[name]}`).join(', '),
            );
            const recompiles = fresh.platform.stats.recompiles - recompilesBefore;
            const firstFrameChange = firstFrameMean / savedMean - 1;
            const windowMoves = afterMore['window moves'];
            log(
                `  loaded in ${loadMs.toFixed(1)} ms (stale ${loaded.stale}); view mean at load ${meanAtLoad.toPrecision(6)}, after the first frame ${firstFrameMean.toPrecision(6)} (${(firstFrameChange * 100).toFixed(3)}% from the saved page); ` +
                    `probes updated ${afterFirst['probes updated']}, warm-up passes ${afterMore['warm-up passes']}, warming after the first frame ${warmingAfterFirst}, recompiles ${recompiles}` +
                    (windowMoves !== undefined ? `, window moves ${windowMoves}, teleports ${afterMore['teleports']}` : ''),
            );
            if (loaded.stale) fail(`the same scene loaded as stale (${loaded.changedParts.join(', ')} differ)`);
            if (Math.abs(firstFrameChange) > FIRST_FRAME_TOLERANCE) fail(`the first frame's mean is ${(firstFrameChange * 100).toFixed(3)}% from the saved page's (bar 1%)`);
            if ((afterMore['warm-up passes'] ?? 0) !== 0 || warmingAfterFirst) fail(`a warm-up ran after the load (${afterMore['warm-up passes']} passes)`);
            if (recompiles !== 0) fail(`${recompiles} pipelines were compiled after the load`);
            if (!((afterFirst['probes updated'] ?? 0) > 0)) fail('the first frame after the trees updated no probes: the loaded field did not carry on updating');
            if ((afterMore['teleports'] ?? 0) !== 0) fail('the camera window teleported after the load: its saved place was not restored');
            fresh.platform.dispose();

            // 3. A layout mismatch is refused, naming the field.
            const other = await buildScene(device, testCase, { irradianceTexels: 8 });
            let refusal = '';
            try {
                other.ddgi.loadField(bytes);
            } catch (error) {
                refusal = error instanceof SavedFieldError && error.problem === 'layout' ? error.message : `the wrong error: ${String(error)}`;
            }
            other.platform.dispose();
            log(`  8 x 8 irradiance texels: ${refusal || 'loaded (it should have been refused)'}`);
            if (!refusal.includes('irradianceTexels')) fail(`a layout mismatch was not refused by name (${refusal || 'it loaded'})`);

            const result: Record<string, unknown> = {
                units: testCase.units,
                warmUpFrames,
                bytes: bytes.byteLength,
                saveMs,
                loadMs,
                savedMean,
                meanAtLoad,
                firstFrameMean,
                firstFrameChange,
                probesUpdatedFirstFrame: afterFirst['probes updated'],
                warmUpPassesAfterLoad: afterMore['warm-up passes'],
                recompilesAfterLoad: recompiles,
                refusal,
            };

            // 4. A changed scene loads stale and settles within the usual response.
            const change = testCase.change;
            if (change) {
                change.apply(saved);
                const live = await responseSeries(device, saved, savedView, RESPONSE_FRAMES, testCase.units);
                await frames(device, saved, SETTLE_FRAMES, testCase.units);
                const settled = await viewMean(saved, savedView);
                const stale = await buildScene(device, testCase, {}, change.apply);
                const staleView = await TestView.create(device, stale.platform);
                const staleLoad = stale.ddgi.loadField(bytes);
                await waitForTrees(device, stale);
                const atLoad = await viewMean(stale, staleView);
                const staleSeries = await responseSeries(device, stale, staleView, RESPONSE_FRAMES, testCase.units);
                const staleCounters = stale.ddgi.counters;
                stale.platform.dispose();
                const liveFrames = framesToSettle(live, settled, SETTLED_TOLERANCE);
                const staleFrames = framesToSettle(staleSeries, settled, SETTLED_TOLERANCE);
                const allowed = liveFrames === null ? null : Math.ceil(liveFrames * STALE_RESPONSE_FACTOR + STALE_RESPONSE_SLACK);
                log(
                    `  ${change.label}: loaded stale ${staleLoad.stale} (${staleLoad.changedParts.join(', ') || 'nothing'} differ); settled mean ${settled.toPrecision(6)} (saved ${savedMean.toPrecision(6)}), stale at load ${atLoad.toPrecision(6)}; ` +
                        `within 5% and staying: live change ${liveFrames ?? 'never'} frames, stale load ${staleFrames ?? 'never'} (allowed ${allowed ?? '-'}); probes marked (total) after the stale load ${staleCounters['probes marked (total)']}`,
                );
                const share = (series: number[]) => series.filter((_, index) => index < 10 || index % 20 === 19).map((mean) => ((mean / settled - 1) * 100).toFixed(1)).join(' ');
                log(`  diagnostics: from settled (%), frames 1-10 then every 20th: live ${share(live)}; stale ${share(staleSeries)}`);
                if (!staleLoad.stale || !staleLoad.changedParts.includes(change.part)) fail(`the changed scene didn't load as stale with ${change.part} named (${staleLoad.changedParts.join(', ') || 'nothing'})`);
                if (liveFrames === null) fail(`the live change never settled within 5% in ${RESPONSE_FRAMES} frames`);
                else if (staleFrames === null || staleFrames > (allowed ?? 0)) fail(`the stale load took ${staleFrames ?? 'more than ' + RESPONSE_FRAMES} frames to settle, the live change ${liveFrames} (allowed ${allowed})`);
                Object.assign(result, {
                    change: change.label,
                    changedParts: staleLoad.changedParts,
                    settledMean: settled,
                    staleMeanAtLoad: atLoad,
                    liveResponseFrames: liveFrames,
                    staleResponseFrames: staleFrames,
                    allowedFrames: allowed,
                    liveSeries: live.filter((_, index) => index % 10 === 0),
                    staleSeries: staleSeries.filter((_, index) => index % 10 === 0),
                });
            }
            saved.platform.dispose();
            results[testCase.id] = result;
        }
        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
