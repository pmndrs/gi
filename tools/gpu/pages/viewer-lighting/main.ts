/**
 * T2.8, the viewer's lighting controls (docs/design/m2-lights.md "Draft
 * done-whens", T2.8), on the viewer's own code: `ViewerCore` draws the scene
 * with DDGI exactly as the viewer page does, and the viewer's own
 * `MotionPlayer`, `TimeOfDay`, `LightEditor` and `ResponseMeter` drive it.
 *
 * - **motions** (check 1): on two-room, point-light Cornell and Sponza, each
 *   scene's `light-sweep` and `light-step` play for MOTION_SECONDS each. The
 *   lamp must be where `motionState` says in every frame, DDGI's picture of
 *   the lamp's room (GI only, the first bookmark) must change while it
 *   plays, and `gi.stats.recompiles` must stay 0 (the lead's rule: no
 *   control compiles anything in GI). Stopping must put the lamp back.
 * - **controls** (the same rule for the rest of the panel, on two-room and
 *   Sponza): the lamp's brightness, colour, reach, group and shadows edited
 *   and put back; on Sponza time of day turned on (DDGI starts over once
 *   with its sun layer, a new platform that counts from 0) and a day played,
 *   and on two-room 2 light groups with their weights changed. The counter
 *   must stay 0 throughout.
 * - **meter** (check 2): the response meter's CSV for two-room's step (the
 *   `light-step` motion's second jump: the lamp back from x = 6.2 to x = 1 m,
 *   into the `lit-room` view; the first jump, out of it, is reported beside
 *   it, METER_JUMPS says why) against the T2.4b test page's measurement of the same step
 *   (`ddgi-shared/response.ts`: fields converged before and after, the step
 *   at the same probes a frame as the viewer ran, its test view's mean): the
 *   first frame within 5% of settled, and the frame from which it stays
 *   there, must agree within MATCH_FRAMES frames.
 *
 *   VITE_DDGI_SECTIONS=motions,controls,meter   (default all; Sponza needs the download)
 *   VITE_DDGI_SCENES=two-room,cornell-box-point,sponza   (the motions section's scenes)
 *
 * `pnpm test:gpu --filter viewer-lighting --gpu nvidia`.
 */

import { Vector2, Vector3 } from 'three';
import { reportGpuTest } from '../../client.ts';
import { ViewerCore } from '../../../../apps/viewer/src/core/viewer-core.ts';
import { LightEditor } from '../../../../apps/viewer/src/lighting/light-editor.ts';
import { MotionPlayer } from '../../../../apps/viewer/src/lighting/motion-player.ts';
import { ResponseMeter, RECORD_FRAMES } from '../../../../apps/viewer/src/lighting/response-meter.ts';
import { TimeOfDay } from '../../../../apps/viewer/src/lighting/time-of-day.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { parseResponseCsv } from '../../../../apps/viewer/src/stats/response-curve.ts';
import { type DdgiTechnique, DEFAULT_DDGI_VIEWER_SETTINGS, type DdgiViewerSettings } from '../../../../apps/viewer/src/techniques/ddgi-technique.ts';
import { motionState, type CameraBookmark, type SceneManifest } from '../../../../scenes/src/manifest.ts';
import { createDdgiScene, imageMean, TestView } from '../ddgi-shared/harness.ts';
import { bookmarkCamera, convergeField, describeErrors, runCameraFrame, stepResponse, type ConvergeOptions } from '../ddgi-shared/response.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SECTIONS = new Set(listFromEnv(env.VITE_DDGI_SECTIONS, ['motions', 'controls', 'meter']));
const MOTION_SCENES = listFromEnv(env.VITE_DDGI_SCENES, ['two-room', 'cornell-box-point', 'sponza']);
const WIDTH = 1280;
const HEIGHT = 720;
/** Each motion plays this long (a sweep's lamp travels 4 m; a step jumps once, at 3 s). */
const MOTION_SECONDS = 4;
/** The lamp's place must match `motionState` to this many metres every frame. */
const POSITION_TOLERANCE_M = 1e-6;
/**
 * While a motion plays, DDGI's picture must change by more than this many
 * times its own change over the same time with nothing moving (DDGI's
 * noise), and by more than MIN_PICTURE_CHANGE. Sponza's arcade lamp (3 cd)
 * in its sunlit courtyard moves the whole picture by about 1%.
 */
const NOISE_MULTIPLE = 3;
const MIN_PICTURE_CHANGE = 0.002;
/** Check 2: the meter and the T2.4b page agree within this many frames. */
const MATCH_FRAMES = 2;
/** The T2.4b page's test view size and convergence (ddgi-schedule-step). */
const TEST_VIEW_WIDTH = 320;
const TEST_VIEW_HEIGHT = 180;
const CONVERGE: Omit<ConvergeOptions, 'settleUnits'> = {
    fastFrames: 96,
    fastHysteresis: 0.5,
    blockFrames: 32,
    convergedTolerance: 0.001,
    maxBlocks: 24,
    settleFrames: 1200,
    convergeUnits: 4096,
};
const TOLERANCE = 0.05;
/**
 * The meter's frames are paced at 60 Hz: headless Chrome draws as fast as it
 * can (about 150 frames a second here), and DDGI's history clock advances by
 * each frame's real time, while the T2.4b page advances it 1/60 s a frame.
 */
const METER_FRAME_MS = 1000 / 60;
/** Frames the T2.4b procedure measures after the step (the meter records RECORD_FRAMES). */
const MEASURED_FRAMES = RECORD_FRAMES;

type Vec3 = [number, number, number];

const startTime = performance.now();
const logElement = document.getElementById('log') as HTMLElement;
const logLines: string[] = [];
const log = (message: string): void => {
    logLines.push(message);
    logElement.textContent = logLines.join('\n');
    console.log(message);
};
const pageErrors: string[] = [];
window.addEventListener('error', (event) => pageErrors.push(`page error: ${event.message}`));
window.addEventListener('unhandledrejection', (event) => pageErrors.push(`unhandled rejection: ${String(event.reason)}`));

const manifests = loadAllManifests();
function manifestById(id: string): SceneManifest {
    const manifest = manifests.find((candidate) => candidate.id === id);
    if (!manifest) throw new Error(`no manifest ${id}`);
    return manifest;
}

function lookFrom(core: ViewerCore, bookmark: CameraBookmark): void {
    core.camera.position.set(...bookmark.position);
    core.camera.fov = bookmark.fovYDeg;
    core.camera.lookAt(...bookmark.target);
    core.camera.updateProjectionMatrix();
    core.camera.updateMatrixWorld(true);
}

/** Loads a scene into the viewer core with DDGI in GI-only mode at a bookmark, and runs frames until DDGI is warm. */
async function openScene(core: ViewerCore, sceneId: string, cameraName: string | null, settings: Partial<DdgiViewerSettings> = {}): Promise<DdgiTechnique> {
    const manifest = manifestById(sceneId);
    core.layout = { left: 'none', right: null, giMode: 'only', split: 0.5 };
    core.configureDdgi({ ...DEFAULT_DDGI_VIEWER_SETTINGS, ...settings });
    await core.loadScene(manifest, () => {});
    lookFrom(core, manifest.cameras.find((camera) => camera.name === cameraName) ?? manifest.cameras[0]);
    core.setLayout({ left: 'ddgi', right: null, giMode: 'only' });
    return waitForDdgi(core);
}

async function waitForDdgi(core: ViewerCore): Promise<DdgiTechnique> {
    const ddgi = core.ddgi;
    if (!ddgi) throw new Error('DDGI was not created');
    const startedAt = performance.now();
    while (!ddgi.running) {
        if (ddgi.failed) throw new Error(`DDGI failed: ${ddgi.failed}`);
        if (performance.now() - startedAt > 180_000) throw new Error('DDGI did not warm up within 180 s');
        await core.renderThroughLoop();
        await core.pumpPrepare(40);
    }
    // A few ordinary frames after the start-up.
    for (let frame = 0; frame < 10; frame++) await core.renderThroughLoop();
    return ddgi;
}

/** DDGI's picture of the whole view, read back (GI only). */
async function pictureMean(core: ViewerCore, ddgi: DdgiTechnique): Promise<number> {
    const size = core.renderer.getDrawingBufferSize(new Vector2());
    const reading = ddgi.readRegionMean(0, 0, size.x, size.y);
    if (!reading) throw new Error('the picture could not be read (upscaler on?)');
    return reading;
}

interface MotionResult {
    scene: string;
    motion: string;
    frames: number;
    worstPositionErrorM: number;
    pictureChange: number;
    /** The picture's own change with nothing moving, over the same time. */
    stillChange: number;
    recompiles: number;
    putBack: boolean;
}

async function runMotions(core: ViewerCore, failures: string[]): Promise<MotionResult[]> {
    const results: MotionResult[] = [];
    for (const sceneId of MOTION_SCENES) {
        log(`motions: ${sceneId}`);
        const ddgi = await openScene(core, sceneId, null);
        const loaded = core.loaded!;
        const player = new MotionPlayer(loaded);
        for (const motionId of ['light-sweep', 'light-step']) {
            const motion = player.motions.find((candidate) => candidate.id === motionId);
            if (!motion) {
                failures.push(`${sceneId} has no ${motionId}`);
                continue;
            }
            const light = player.lightFor(motion)!;
            const before = light.position.clone();
            const beforeIntensity = light.intensity;
            const meanBefore = await pictureMean(core, ddgi);
            // The picture's own change with nothing moving, over the same time and sampling.
            let stillChange = 0;
            for (let frame = 1, startedStill = performance.now(); performance.now() - startedStill < MOTION_SECONDS * 1000; frame++) {
                await core.renderThroughLoop();
                if (frame % 15 === 0) stillChange = Math.max(stillChange, Math.abs((await pictureMean(core, ddgi)) / meanBefore - 1));
            }
            let worst = 0;
            let pictureChange = 0;
            let frames = 0;
            const startedAt = performance.now();
            player.play(motionId, startedAt);
            ddgi.markChanged(light);
            while (performance.now() - startedAt < MOTION_SECONDS * 1000) {
                const now = performance.now();
                player.update(now);
                const expected = motionState(motion, player.seconds(now)).position as Vec3 | undefined;
                if (expected) worst = Math.max(worst, light.position.distanceTo(new Vector3(...expected)));
                await core.renderThroughLoop();
                frames++;
                if (frames % 15 === 0) pictureChange = Math.max(pictureChange, Math.abs((await pictureMean(core, ddgi)) / meanBefore - 1));
            }
            player.stop();
            const putBack = light.position.distanceTo(before) < 1e-9 && light.intensity === beforeIntensity && (beforeIntensity > 0 || light.parent === null);
            await core.renderThroughLoop();
            const result: MotionResult = { scene: sceneId, motion: motionId, frames, worstPositionErrorM: worst, pictureChange, stillChange, recompiles: ddgi.recompiles, putBack };
            results.push(result);
            log(
                `  ${motionId}: ${frames} frames, lamp within ${worst.toExponential(1)} m of the script, picture moved up to ${(pictureChange * 100).toFixed(2)}% (still: ${(stillChange * 100).toFixed(2)}%), recompiles ${ddgi.recompiles}, put back ${putBack}`,
            );
            if (worst > POSITION_TOLERANCE_M) failures.push(`${sceneId} ${motionId}: the lamp was ${worst} m from the script`);
            if (!(pictureChange > Math.max(MIN_PICTURE_CHANGE, NOISE_MULTIPLE * stillChange))) {
                failures.push(`${sceneId} ${motionId}: DDGI's picture moved only ${(pictureChange * 100).toFixed(2)}% (${(stillChange * 100).toFixed(2)}% with nothing moving)`);
            }
            if (ddgi.recompiles !== 0) failures.push(`${sceneId} ${motionId}: ${ddgi.recompiles} recompiles`);
            if (!putBack) failures.push(`${sceneId} ${motionId}: stopping did not put the lamp back`);
        }
    }
    return results;
}

async function frames(core: ViewerCore, count: number): Promise<void> {
    for (let frame = 0; frame < count; frame++) await core.renderThroughLoop();
}

async function runControls(core: ViewerCore, failures: string[]): Promise<Record<string, number>> {
    const counts: Record<string, number> = {};
    const check = (label: string, ddgi: DdgiTechnique) => {
        counts[label] = ddgi.recompiles;
        log(`  ${label}: recompiles ${ddgi.recompiles}`);
        if (ddgi.recompiles !== 0) failures.push(`controls, ${label}: ${ddgi.recompiles} recompiles`);
    };

    // Two-room: the lamp's controls, then light groups and their weights.
    log('controls: two-room');
    let ddgi = await openScene(core, 'two-room', 'lit-room');
    const canvas = core.renderer.domElement;
    let editor = new LightEditor(core.loaded!, core.camera, canvas, { onEdited: () => {}, onDragging: () => {}, onSelected: () => {} });
    editor.select(0);
    const lamp = editor.selected!;
    const settings = editor.settings(lamp);
    const edits: Array<[string, Partial<typeof settings>]> = [
        ['brightness', { intensity: settings.intensity * 4 }],
        ['colour', { color: [1, 0.3, 0.1] }],
        ['reach', { range: 4 }],
        ['group', { group: 2 }],
        ['off', { on: false }],
        ['on', { on: true }],
        ['shadows off', { shadows: false }],
        ['shadows on', { shadows: true }],
    ];
    for (const [label, edit] of edits) {
        ddgi.markChanged(editor.apply(lamp, { ...editor.settings(lamp), ...edit }));
        await frames(core, 20);
        check(`two-room lamp ${label}`, ddgi);
    }
    // The gizmo's own move: the marker dragged 1 m, the lamp following.
    lamp.marker!.position.x += 1;
    editor.followMarker();
    ddgi.markChanged(lamp.light);
    await frames(core, 20);
    check('two-room lamp moved by its marker', ddgi);
    ddgi.markChanged(editor.putBack(lamp));
    await frames(core, 20);
    check('two-room lamp put back', ddgi);
    editor.dispose();

    core.configureDdgi({ ...DEFAULT_DDGI_VIEWER_SETTINGS, lightGroups: 2 });
    ddgi = await waitForDdgi(core);
    check('two-room with 2 light groups (a new DDGI)', ddgi);
    for (const weights of [[{ color: [1, 0.5, 0.2], intensity: 0.5 }], [{ color: [1, 1, 1], intensity: 0 }, { color: [0.2, 0.4, 1], intensity: 2 }]] as const) {
        core.configureDdgi({ ...DEFAULT_DDGI_VIEWER_SETTINGS, lightGroups: 2, groupWeights: weights.map((weight) => ({ color: [...weight.color] as Vec3, intensity: weight.intensity })), environmentWeight: { color: [1, 1, 1], intensity: 0.5 } });
        await frames(core, 10);
        check('two-room group weights', ddgi);
    }

    // Sponza: time of day on (DDGI starts over with the sun layer), the hour moved, a day played.
    if (MOTION_SCENES.includes('sponza')) {
        log('controls: sponza');
        ddgi = await openScene(core, 'sponza', null);
        const timeOfDay = new TimeOfDay(core.loaded!);
        const environment = timeOfDay.setEnabled(true, performance.now());
        core.configureDdgi({ ...DEFAULT_DDGI_VIEWER_SETTINGS, sunLayer: true });
        core.ddgi!.setEnvironment(environment);
        ddgi = await waitForDdgi(core);
        log(`  sun layer: ${ddgi.sunLayerOn ? 'on' : 'off'}`);
        if (!ddgi.sunLayerOn) failures.push('time of day did not turn the sun layer on');
        check('sponza time of day on (a new DDGI, with the sun layer)', ddgi);
        ddgi.setEnvironment(timeOfDay.setTime(timeOfDay.hour + 2, timeOfDay.dayOfYear, performance.now()));
        await frames(core, 30);
        check('sponza two hours later', ddgi);
        timeOfDay.playDay(performance.now());
        let played = 0;
        while (timeOfDay.playing) {
            const changed = timeOfDay.update(performance.now());
            if (changed !== undefined) ddgi.setEnvironment(changed);
            await core.renderThroughLoop();
            played++;
        }
        log(`  played a day in ${played} frames`);
        check('sponza a played day', ddgi);
        editor = new LightEditor(core.loaded!, core.camera, canvas, { onEdited: () => {}, onDragging: () => {}, onSelected: () => {} });
        const sun = editor.lights.find((entry) => entry.kind === 'directional')!;
        ddgi.markChanged(editor.apply(sun, { ...editor.settings(sun), intensity: 1, color: [1, 0.6, 0.35] }));
        await frames(core, 20);
        check('sponza sun dimmed and recoloured', ddgi);
        editor.dispose();
        timeOfDay.dispose();
    }
    return counts;
}

interface MeterComparison {
    jump: number;
    judged: boolean;
    units: number;
    meter: { enteredFrame: number | null; settledFrame: number | null; firstChangedFrame: number | null; before: number; settled: number; settleCurve: ReadonlyArray<{ seconds: number; mean: number }> };
    page: { enteredFrame: number | null; settledFrame: number | null; firstFrameMove: number; before: number; settled: number; errors: string };
    csv: string;
}

/**
 * The step's jumps, in the order measured: jump 2 (the lamp back into the lit
 * room, the view brightening tenfold) is check 2's; jump 1 (the lamp leaving,
 * the view ending at a tenth) is reported beside it. Leaving, 5% of settled
 * is 0.5% of the picture before the step, where the old light's bounce fades
 * by about 0.3% a second, so both methods' "within 5%" frame there depends on
 * that slow tail more than on the response (DECISIONS 2026-11-08).
 */
const METER_JUMPS: ReadonlyArray<{ jump: number; judged: boolean }> = [
    { jump: 1, judged: true },
    { jump: 0, judged: false },
];

/** Check 2: the meter's CSV on two-room's step against the T2.4b page's measurement of the same step. */
async function runMeter(core: ViewerCore, failures: string[]): Promise<MeterComparison[]> {
    const comparisons: MeterComparison[] = [];
    for (const { jump, judged } of METER_JUMPS) {
        log(`meter: two-room, lit-room, light-step jump ${jump + 1}${judged ? '' : ' (reported, not judged)'}`);
        comparisons.push(await compareJump(core, jump, judged, failures));
    }
    return comparisons;
}

async function compareJump(core: ViewerCore, jump: number, judged: boolean, failures: string[]): Promise<MeterComparison> {
    await openScene(core, 'two-room', 'lit-room');
    const player = new MotionPlayer(core.loaded!);
    const motion = player.motions.find((candidate) => candidate.id === 'light-step')!;
    const meter = new ResponseMeter({
        ddgi: () => core.ddgi,
        player,
        drawingBufferSize: () => {
            const size = core.renderer.getDrawingBufferSize(new Vector2());
            return { width: size.x, height: size.y };
        },
        markChanged: (light) => core.ddgi?.markChanged(light),
        describe: () => ({ scene: 'two-room', camera: 'lit-room', adapter: `${core.adapterInfo.vendor} ${core.adapterInfo.architecture}` }),
    });
    // A few seconds at 60 Hz first, so the frame-rate-dependent parts (the history clock, the frame-time window) are at 60 Hz too.
    let lastFrameAt = performance.now();
    const pacedFrame = async () => {
        const wait = lastFrameAt + METER_FRAME_MS - performance.now();
        if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
        lastFrameAt = performance.now();
        await core.renderThroughLoop();
    };
    for (let frame = 0; frame < 120; frame++) await pacedFrame();
    meter.start(motion, performance.now(), jump);
    while (meter.running) {
        meter.beforeFrame(performance.now());
        await pacedFrame();
        meter.afterFrame(performance.now());
    }
    const result = meter.result;
    if (!result) throw new Error(`the meter stopped: ${meter.failed}`);
    const parsed = parseResponseCsv(result.csv);
    const unitsByFrame = result.samples.map((sample) => sample.probesUpdated).sort((a, b) => a - b);
    const units = unitsByFrame[Math.floor(unitsByFrame.length / 2)];
    log(`  meter: ${units} probes a frame; first change frame ${parsed.firstChangedFrame}, within ${TOLERANCE * 100}% from frame ${parsed.enteredFrame}, for good from ${parsed.settledFrame}`);
    log(`  meter curve: ${describeErrors(result.samples.map((sample) => sample.mean / result.marks.settled - 1))}`);
    // How the test mode's settled picture moved: the mean at whole seconds, against the final settled.
    const settleBySecond = result.settleCurve.filter((reading, index, all) => index === 0 || Math.floor(reading.seconds) !== Math.floor(all[index - 1].seconds));
    log(`  meter settling: ${settleBySecond.map((reading) => `${reading.seconds.toFixed(1)} s ${((reading.mean / result.marks.settled - 1) * 100).toFixed(2)}%`).join(', ')}`);

    // The T2.4b page's procedure on the same jump, at the same probes a frame.
    const device = core.device;
    const manifest = manifestById('two-room');
    const bookmark = manifest.cameras.find((camera) => camera.name === 'lit-room')!;
    const camera = bookmarkCamera(bookmark, TEST_VIEW_WIDTH / TEST_VIEW_HEIGHT);
    const from = motion.keyframes[jump].position as Vec3;
    const to = motion.keyframes[jump + 1].position as Vec3;
    const converged = async (position: Vec3, label: string) => {
        const scene = await createDdgiScene(device, 'two-room', {}, { gpuTiming: true });
        scene.platform.scene.updateLight(scene.lights[0], { position });
        runCameraFrame(device, scene.platform, 0, camera);
        const view = await TestView.create(device, scene.platform);
        const field = scene.ddgi.field!;
        const mean = async () => imageMean((await view.render(field, bookmark, TEST_VIEW_WIDTH, TEST_VIEW_HEIGHT)).indirect);
        const settled = await convergeField(device, scene.platform, scene.ddgi.settings, camera, mean, { ...CONVERGE, settleUnits: units }, label, log);
        return { scene, mean, settled };
    };
    const after = await converged(to, `two-room, lamp at x = ${to[0]}`);
    after.scene.platform.dispose();
    const before = await converged(from, `two-room, lamp at x = ${from[0]}`);
    before.scene.platform.scene.updateLight(before.scene.lights[0], { position: to });
    const means: number[] = [];
    for (let frame = 1; frame <= MEASURED_FRAMES; frame++) {
        runCameraFrame(device, before.scene.platform, units, camera);
        means.push(await before.mean());
    }
    before.scene.platform.dispose();
    const response = stepResponse(before.settled, after.settled, means, TOLERANCE);
    const firstFrameMove = Math.abs(means[0] - before.settled) / Math.max(before.settled, after.settled);
    log(`  T2.4b page: within ${TOLERANCE * 100}% from frame ${response.enteredFrame}, for good from ${response.settledFrame}; frame 1 moved ${(firstFrameMove * 100).toFixed(1)}%`);
    log(`  T2.4b curve: ${describeErrors(response.errors)}`);

    if (judged) {
        const differs = (a: number | null, b: number | null) => a === null || b === null || Math.abs(a - b) > MATCH_FRAMES;
        if (differs(parsed.enteredFrame, response.enteredFrame)) failures.push(`meter, jump ${jump + 1}: within 5% from frame ${parsed.enteredFrame}, the T2.4b page ${response.enteredFrame} (more than ${MATCH_FRAMES} apart)`);
        if (differs(parsed.settledFrame, response.settledFrame)) failures.push(`meter, jump ${jump + 1}: for good from frame ${parsed.settledFrame}, the T2.4b page ${response.settledFrame} (more than ${MATCH_FRAMES} apart)`);
    }
    return {
        jump: jump + 1,
        judged,
        units,
        meter: {
            enteredFrame: parsed.enteredFrame,
            settledFrame: parsed.settledFrame,
            firstChangedFrame: parsed.firstChangedFrame,
            before: result.marks.before,
            settled: result.marks.settled,
            settleCurve: settleBySecond,
        },
        page: { enteredFrame: response.enteredFrame, settledFrame: response.settledFrame, firstFrameMove, before: before.settled, settled: after.settled, errors: describeErrors(response.errors) },
        csv: result.csv,
    };
}

try {
    const canvas = document.getElementById('view') as HTMLCanvasElement;
    canvas.width = WIDTH;
    canvas.height = HEIGHT;
    canvas.style.width = `${WIDTH}px`;
    canvas.style.height = `${HEIGHT}px`;
    const core = await ViewerCore.create(canvas);
    core.resize(WIDTH, HEIGHT);
    window.addEventListener('pagehide', () => core.dispose());
    const failures: string[] = [];
    const data: Record<string, unknown> = {};
    if (SECTIONS.has('motions')) data.motions = await runMotions(core, failures);
    if (SECTIONS.has('controls')) data.controls = await runControls(core, failures);
    if (SECTIONS.has('meter')) data.meter = await runMeter(core, failures);
    const problems = [...pageErrors, ...failures];
    const info = core.adapterInfo;
    reportGpuTest({
        status: problems.length === 0 ? 'pass' : 'fail',
        message: problems.length === 0 ? undefined : problems.join(' | '),
        data,
        deviceErrors: [...core.gpuErrors],
        startTime,
        adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description, device: info.device, features: [...core.device.features].sort(), isFallbackAdapter: false },
        logLines,
    });
} catch (error) {
    const message = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
    reportGpuTest({ status: 'fail', message: [message, ...pageErrors].join(' | '), startTime, logLines });
}
