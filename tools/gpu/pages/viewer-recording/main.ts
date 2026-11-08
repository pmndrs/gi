/**
 * T2.8 check 3: records a light motion in the viewer the way the native
 * harness records it (T2.7, tools/native/README.md "Motions and
 * recordings"), for the compare page (apps/viewer/compare.html) to play side
 * by side with native's recording.
 *
 * The viewer's own code draws it: `ViewerCore` with DDGI at its default
 * budget, GI only, at the scene's first bookmark (two-room: `lit-room`, as
 * native's), and the viewer's `MotionPlayer` poses the light. Like native:
 *
 * - the motion on a fixed clock: frame k shows it at k / 60 s;
 * - 600 frames at the motion's start first, so the probes have settled
 *   (native's `app.motionWarmupFrames`);
 * - each frame's brightness is the response meter's reading over the whole
 *   view (`DdgiTechnique.readRegionMean`: the mean of red, green and blue of
 *   DDGI's GI-only picture, linear);
 * - an image every 4th frame: the picture on screen (GI only, after exposure
 *   and tone mapping), as a JPEG at 960 x 540.
 *
 * What isn't fixed: DDGI ages each probe's history by the real time between
 * frames (`ThreeGI.update` measures it; the blend's memory is in seconds),
 * where native's blend has no time in it (a fixed hysteresis a frame,
 * tools/native/README.md). So the frames are paced at 60 Hz (headless Chrome
 * otherwise draws about 150 a second, NOTES), and nothing slow runs between
 * them: a picture frame only snapshots the canvas (`createImageBitmap`), and
 * the JPEGs are encoded after the run. An encode inside the loop made each
 * picture frame about 31 ms and the next 5 ms. The real intervals are in
 * `frames.csv` (`wall_time_s`, from each frame's start), and their median,
 * mean and largest in `recording.json` (`frameIntervalMs`).
 *
 * The page hands the recording back in its report (data.recordings, images
 * as data URLs); `node tools/native/save-viewer-recording.ts <report.json>`
 * writes it out as a recording folder in bench/recordings/viewer (git-ignored).
 * One motion a run keeps the report small:
 *
 *   VITE_DDGI_MOTION=light-sweep   (or light-step; default light-sweep)
 *   VITE_DDGI_SCENE=two-room       (default two-room)
 *   VITE_DDGI_FRAMES=625           (default: as native's recordings, MOTION_FRAMES)
 *   VITE_DDGI_TAG=run1             (optional: the folder is <scene>-<motion>-<tag>, as play.ps1 -Tag)
 *   VITE_DDGI_SETTLE_KEY=1         (optional: hold the light at this key of the motion instead of
 *                                   playing it, as play.ps1 -SettleKey: 3,600 frames held, then 61
 *                                   recorded and no images; tools/native/response.ts takes the mean
 *                                   of those as the settled level, the same rule as native's)
 *
 * `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter viewer-recording --gpu nvidia --json <report.json>`.
 */

import { Vector2 } from 'three';
import { reportGpuTest } from '../../client.ts';
import { ViewerCore } from '../../../../apps/viewer/src/core/viewer-core.ts';
import { framesCsv, type RecordingInfo } from '../../../../apps/viewer/src/compare/recording.ts';
import { MotionPlayer } from '../../../../apps/viewer/src/lighting/motion-player.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { defaultExposureEv } from '../../../../scenes/src/manifest.ts';
import { DEFAULT_DDGI_VIEWER_SETTINGS, type DdgiTechnique } from '../../../../apps/viewer/src/techniques/ddgi-technique.ts';

const env = import.meta.env as Record<string, unknown>;
const SCENE = typeof env.VITE_DDGI_SCENE === 'string' && env.VITE_DDGI_SCENE !== '' ? env.VITE_DDGI_SCENE : 'two-room';
const MOTION = typeof env.VITE_DDGI_MOTION === 'string' && env.VITE_DDGI_MOTION !== '' ? env.VITE_DDGI_MOTION : 'light-sweep';
/**
 * Frames recorded, as native's two-room recordings have them: the sweep one
 * loop and its last frame (10.4 s, 625), the step 12 s (`-RecordSeconds 12`, 721).
 */
const MOTION_FRAMES: Readonly<Record<string, number>> = { 'light-sweep': 625, 'light-step': 721 };
const TAG = typeof env.VITE_DDGI_TAG === 'string' ? env.VITE_DDGI_TAG : '';
/** -1: play the motion. A key index: hold the light there (native's `play.ps1 -SettleKey`, with its frame counts). */
const SETTLE_KEY = env.VITE_DDGI_SETTLE_KEY !== undefined && env.VITE_DDGI_SETTLE_KEY !== '' ? Number(env.VITE_DDGI_SETTLE_KEY) : -1;
const SETTLING = SETTLE_KEY >= 0;
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Number(env.VITE_DDGI_FRAMES) : SETTLING ? 61 : (MOTION_FRAMES[MOTION] ?? 625);
const FRAME_RATE = 60;
const WARMUP_FRAMES = SETTLING ? 3600 : 600;
/** No pictures while settling (native's settle recordings use -RecordImageEvery 0); only the brightness is wanted. */
const IMAGE_EVERY = SETTLING ? 0 : 4;
const WIDTH = 1280;
const HEIGHT = 720;
const IMAGE_WIDTH = 960;
const IMAGE_HEIGHT = 540;
const JPEG_QUALITY = 0.85;

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
    return ddgi;
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

    const manifest = loadAllManifests().find((candidate) => candidate.id === SCENE);
    if (!manifest) throw new Error(`no scene ${SCENE}`);
    const bookmark = manifest.cameras[0];
    core.layout = { left: 'none', right: null, giMode: 'only', split: 0.5 };
    core.configureDdgi(DEFAULT_DDGI_VIEWER_SETTINGS);
    await core.loadScene(manifest, () => {});
    // The scene's own exposure, as the viewer applies it (viewer-app.ts); ViewerCore starts at 0 EV. Only the pictures
    // change: the brightness is read before exposure.
    const exposureEv = defaultExposureEv(manifest);
    core.setExposureEv(exposureEv);
    core.camera.position.set(...bookmark.position);
    core.camera.fov = bookmark.fovYDeg;
    core.camera.lookAt(...bookmark.target);
    core.camera.updateProjectionMatrix();
    core.camera.updateMatrixWorld(true);
    // GI only, as the response meter reads it (three's direct light changes at once and would hide GI's response).
    core.setLayout({ left: 'ddgi', right: null, giMode: 'only' });
    const ddgi = await waitForDdgi(core);
    log(`${SCENE}: DDGI running`);

    const player = new MotionPlayer(core.loaded!);
    const motion = player.motions.find((candidate) => candidate.id === MOTION);
    if (!motion) throw new Error(`${SCENE} has no motion ${MOTION}`);
    const light = player.lightFor(motion)!;
    if (SETTLING && !(SETTLE_KEY < motion.keyframes.length)) throw new Error(`${MOTION} has no key ${SETTLE_KEY}`);
    // The motion's time each frame: its own clock, or the settle key's time held (a step motion shows that key from then on)
    const motionTime = (frame: number): number => (SETTLING ? motion.keyframes[SETTLE_KEY].t : frame / FRAME_RATE);

    // Paced at 60 Hz (the file comment says why) on a fixed schedule, so a late frame (the browser's next animation frame
    // comes a few ms after the timer) doesn't push every later frame back: the intervals average 1/60 s. Returns when the
    // frame began drawing (`renderThroughLoop`), just before DDGI's update measures its time.
    const frameMs = 1000 / FRAME_RATE;
    let nextFrameAt = performance.now();
    const pacedFrame = async (): Promise<number> => {
        const wait = nextFrameAt - performance.now();
        if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
        // After a long stall start the schedule again rather than racing to catch up.
        nextFrameAt = Math.max(nextFrameAt + frameMs, performance.now());
        return core.renderThroughLoop();
    };

    // Native's warm-up: the motion's start for 600 frames.
    player.pose(motion, motionTime(0));
    ddgi.markChanged(light);
    for (let frame = 0; frame < WARMUP_FRAMES; frame++) await pacedFrame();
    log(`${SCENE} ${MOTION}: ${WARMUP_FRAMES} frames ${SETTLING ? `held at key ${SETTLE_KEY}` : 'at the start'}, now recording ${FRAMES}`);

    const size = core.renderer.getDrawingBufferSize(new Vector2());
    const rows: Array<{ frame: number; time: number; wallTime: number; image: string | null; value: number; probesUpdated: number; light: readonly [number, number, number]; mean: Promise<number> }> = [];
    const snapshots: Array<{ path: string; bitmap: Promise<ImageBitmap> }> = [];
    const frameStarts: number[] = [];
    for (let frame = 0; frame < FRAMES; frame++) {
        const time = motionTime(frame);
        player.pose(motion, time);
        const startedAt = await pacedFrame();
        frameStarts.push(startedAt);
        const reading = ddgi.readRegionMean(0, 0, size.x, size.y);
        if (!reading) throw new Error('the picture could not be read');
        let image: string | null = null;
        if (IMAGE_EVERY > 0 && frame % IMAGE_EVERY === 0) {
            // Right after the frame, before the browser shows the next: a snapshot of the picture on screen, encoded after the run.
            image = `frames/frame-${String(frame).padStart(6, '0')}.jpg`;
            snapshots.push({ path: image, bitmap: createImageBitmap(canvas, { resizeWidth: IMAGE_WIDTH, resizeHeight: IMAGE_HEIGHT, resizeQuality: 'high' }) });
        }
        rows.push({ frame, time, wallTime: (startedAt - frameStarts[0]) / 1000, image, value: 0, probesUpdated: ddgi.probesUpdated, light: [light.position.x, light.position.y, light.position.z], mean: reading });
    }
    for (const row of rows) row.value = await row.mean;

    // The real time between frames: what DDGI aged its histories by.
    const intervals = frameStarts.slice(1).map((start, index) => start - frameStarts[index]);
    const sorted = [...intervals].sort((a, b) => a - b);
    const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
    // Intervals ending at the frame after a picture frame, and the rest: the snapshot must not slow the picture frames.
    const afterPicture = IMAGE_EVERY > 0 ? intervals.filter((_, index) => index % IMAGE_EVERY === 0) : [];
    const others = IMAGE_EVERY > 0 ? intervals.filter((_, index) => index % IMAGE_EVERY !== 0) : intervals;
    const frameIntervalMs = {
        median: sorted[Math.floor(sorted.length / 2)],
        mean: mean(intervals),
        largest: sorted[sorted.length - 1],
        meanAfterPicture: mean(afterPicture),
        meanOthers: mean(others),
        over25: intervals.filter((interval) => interval > 25).length,
    };
    log(
        `frame intervals: median ${frameIntervalMs.median.toFixed(2)} ms, mean ${frameIntervalMs.mean.toFixed(2)} ms, largest ${frameIntervalMs.largest.toFixed(2)} ms; ` +
            `after a picture frame ${frameIntervalMs.meanAfterPicture.toFixed(2)} ms, others ${frameIntervalMs.meanOthers.toFixed(2)} ms; ${frameIntervalMs.over25} over 25 ms`,
    );

    // The JPEGs, now that the timed frames are over.
    const capture = document.createElement('canvas');
    capture.width = IMAGE_WIDTH;
    capture.height = IMAGE_HEIGHT;
    const captureContext = capture.getContext('2d')!;
    const images: Array<{ path: string; dataUrl: string }> = [];
    for (const snapshot of snapshots) {
        const bitmap = await snapshot.bitmap;
        captureContext.drawImage(bitmap, 0, 0);
        bitmap.close();
        images.push({ path: snapshot.path, dataUrl: capture.toDataURL('image/jpeg', JPEG_QUALITY) });
    }
    log(`recorded ${rows.length} frames, ${images.length} images; brightness ${rows[0].value.toPrecision(4)} at the start`);

    const info: RecordingInfo = {
        format: 'webgpu-ddgi native motion recording, version 1 (tools/native/README.md, "Motions and recordings"); written by the viewer (tools/gpu/pages/viewer-recording)',
        source: 'viewer',
        scene: SCENE,
        motion: MOTION,
        motionSeconds: motion.keyframes[motion.keyframes.length - 1].t,
        camera: bookmark.name,
        cameraBookmark: 1,
        width: IMAGE_WIDTH,
        height: IMAGE_HEIGHT,
        view: 'indirect',
        // The pictures' exposure in stops (the scene's own, manifest `exposureEv`)
        exposureEv,
        clock: 'fixed',
        frameSeconds: 1 / FRAME_RATE,
        // The motion's clock is fixed; DDGI's histories follow these real intervals (the file comment).
        frameIntervalMs,
        // DDGI's own probe grid (native's recording.json says which grid it used; without one, its own `cells`).
        gridRule: 'ddgi',
        frameRate: FRAME_RATE,
        warmupFrames: WARMUP_FRAMES,
        // As native's recording.json: -1 when the motion played, else the key the light was held at
        settleKey: SETTLE_KEY,
        frames: FRAMES,
        imageEvery: IMAGE_EVERY,
        images: images.length,
        framesCsv: 'frames.csv',
        probeUpdate: `DDGI at the viewer's default budget (${DEFAULT_DDGI_VIEWER_SETTINGS.budgetMs} ms a frame): ${rows[Math.floor(rows.length / 2)].probesUpdated} probes a frame`,
        probes: ddgi.describeSettings().probes,
        // The adapter's description, or what else it says about itself when that is empty.
        device: core.adapterInfo.description || [core.adapterInfo.vendor, core.adapterInfo.architecture, core.adapterInfo.device].filter(Boolean).join(' '),
        // The browser, for response.ts's rows (headless Chrome says HeadlessChrome/<version>)
        browser: navigator.userAgent,
        recorded: new Date().toISOString(),
    };
    const recording = { name: `${SCENE}-${MOTION}${TAG ? `-${TAG}` : ''}`, info, framesCsv: framesCsv(rows, light.name), images };
    const problems = [...pageErrors];
    const adapter = core.adapterInfo;
    reportGpuTest({
        status: problems.length === 0 ? 'pass' : 'fail',
        message: problems.length === 0 ? undefined : problems.join(' | '),
        data: { recordings: [recording] },
        deviceErrors: [...core.gpuErrors],
        startTime,
        adapter: { vendor: adapter.vendor, architecture: adapter.architecture, description: adapter.description, device: adapter.device, features: [...core.device.features].sort(), isFallbackAdapter: false },
        logLines,
    });
} catch (error) {
    const message = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
    reportGpuTest({ status: 'fail', message: [message, ...pageErrors].join(' | '), startTime, logLines });
}
