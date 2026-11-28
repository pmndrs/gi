/**
 * T2.4a done-whens 2 and 4: a lamp step on two-room, in a test mode that
 * updates every probe every frame (the 16 ms test budget; 1,071 probes), so
 * this tests the history without a scheduler (docs/design/m2-lights.md,
 * "Draft done-whens", T2.4a).
 *
 * - **Check 2 (response):** the lamp moves 2 m in one frame. The GI-only
 *   image at the `lit-room` bookmark (the engine-free test view's indirect
 *   image, its mean as the region of interest) must be within 5% of the
 *   settled image for the new position within 0.5 s (30 frames at 60 Hz),
 *   and stay there. Settled: a second scene with the lamp at the new place
 *   from the start, converged the same way. The same step with
 *   `changeDrivenHistory: false` (M1's rule alone) is measured beside it.
 * - **Check 4 (the change estimate):** right after the step, which probes
 *   `ddgi/mark` cut (pendingChange > ε) against the probes whose light
 *   really changed by more than 5%, from the two converged fields (each
 *   probe's mean irradiance, the history's words 6-7, relative per colour
 *   channel, the largest). Recall (cut among really changed) and precision,
 *   for the whole volume and for the lit room (room A, x < 4 m), at the
 *   default `lightChangeScale` and, worked out from the same marks, at
 *   other scales (one light, so each probe's change is exactly scale × its
 *   unscaled value).
 *
 * Converging, the same for every scene. Two-room bounces most of its light,
 * and bounce light settles by about (1 - h)(1 - rho) an update, so at 0.97
 * or 0.995 it takes thousands of frames (the leak page runs 3,000); a field
 * that isn't converged reads low, and a field whose cut histories converge
 * fast then looks as if it overshoots. So: FAST_FRAMES frames at a fixed
 * h = FAST_HYSTERESIS (bounce light settles about 17% an update, noisily),
 * then blocks of 32 frames at the fixed 0.97 until the view's mean moves
 * under 0.1% between blocks, then SETTLE_FRAMES frames of the default rule,
 * so each probe's history reaches its steady count (n = 200, h = 0.995)
 * before the step: the slowest case for a step, a probe that remembers 3.3 s.
 *
 * The frame of the step runs `ddgi/mark` with no probe updates, so the marks
 * can be read before the blend takes them in; it counts in the response
 * time (frame 1). DDGI's clock advances 1/60 s a frame (a 60 Hz host).
 *
 * Manual: `pnpm test:gpu --filter ddgi-change-step --gpu nvidia`.
 *   VITE_DDGI_LIGHT_CHANGE_SCALE=0.25   (instead of the default, for tuning)
 */

import { assert, runGpuTest } from '../../client.ts';
import { PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS } from '../../../../packages/core/src/techniques/ddgi/layout.ts';
import { createDdgiScene, halfToFloat, imageMean, readBuffer, readProbeData, releaseDeviceOnPageHide, runFrames, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';
import { applyHysteresisMode, FIXED_097, parseHysteresisMode } from '../ddgi-shared/hysteresis-modes.ts';

const env = import.meta.env as Record<string, unknown>;

const SCENE = 'two-room';
const CAMERA = 'lit-room';
/** The lamp's step: 2 m from its manifest place, diagonally toward room A's south-west corner (it stays 0.59 m from both walls). */
const STEP_METRES = 2;
const STEP_DIRECTION: readonly [number, number, number] = [-Math.SQRT1_2, 0, -Math.SQRT1_2];
/** Room A is x 0..4 m (scenes/src/procedural/two-rooms.ts). */
const LIT_ROOM_MAX_X = 4;
const WIDTH = 320;
const HEIGHT = 180;
/** DDGI's clock per frame: a 60 Hz host. */
const FRAME_SECONDS = 1 / 60;
/** The response target: within TOLERANCE of settled in RESPONSE_SECONDS. */
const TOLERANCE = 0.05;
const RESPONSE_SECONDS = 0.5;
/** Frames measured after the step (2 s): long enough to see the response settle, or not. */
const RESPONSE_FRAMES = 120;
/** Converging (the file comment). */
const FAST_FRAMES = 96;
const FAST_HYSTERESIS = 0.5;
const BLOCK_FRAMES = 32;
const CONVERGED_TOLERANCE = 0.001;
const MIN_BLOCKS = 2;
const MAX_BLOCKS = 16;
/** Four memory times of the default rule (τ = 5 s): every probe's history at its steady count. */
const SETTLE_FRAMES = 1200;
/** A probe "really changed" by more than this (check 4), the same 5% as the tolerance. */
const TRUE_CHANGE = 0.05;
/** The lit-room recall the done-when asks for. */
const RECALL_TARGET = 0.95;
/** Scales the recall is also worked out for, from the same marks. */
const SCALES = [0.1, 0.15, 0.25, 0.5, 1];
const LIGHT_CHANGE_SCALE = Number(env.VITE_DDGI_LIGHT_CHANGE_SCALE) > 0 ? Number(env.VITE_DDGI_LIGHT_CHANGE_SCALE) : undefined;

type Vec3 = [number, number, number];

interface ProbeSnapshot {
    readonly mean: Vec3[];
    readonly pendingChange: number[];
    readonly active: boolean[];
    readonly position: Vec3[];
}

async function probeSnapshot(scene: DdgiScene): Promise<ProbeSnapshot> {
    const device = scene.platform.device;
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const probeCount = field.grid.probeCount;
    const words = new Uint32Array(await readBuffer(device, field.probeHistory, probeCount * PROBE_HISTORY_WORDS * 4));
    const floats = new Float32Array(words.buffer);
    const probes = await readProbeData(device, field);
    const [nx, , nz] = field.grid.counts;
    const mean: Vec3[] = [];
    const pendingChange: number[] = [];
    const position: Vec3[] = [];
    for (let probe = 0; probe < probeCount; probe++) {
        const base = probe * PROBE_HISTORY_WORDS;
        const redGreen = words[base + PROBE_HISTORY_WORD.meanRedGreen];
        const blueNoise = words[base + PROBE_HISTORY_WORD.meanBlueNoise];
        mean.push([halfToFloat(redGreen & 0xffff), halfToFloat(redGreen >>> 16), halfToFloat(blueNoise & 0xffff)]);
        pendingChange.push(floats[base + PROBE_HISTORY_WORD.pendingChange]);
        const x = probe % nx;
        const z = Math.floor(probe / nx) % nz;
        const y = Math.floor(probe / (nx * nz));
        const offset = probes[probe].offset;
        position.push([0, 1, 2].map((axis) => field.grid.origin[axis] + ([x, y, z][axis] + offset[axis]) * field.grid.spacing[axis]) as Vec3);
    }
    return { mean, pendingChange, active: probes.map((probe) => probe.state === 0), position };
}

function viewOf(scene: DdgiScene) {
    const bookmark = scene.manifest.cameras.find((camera) => camera.name === CAMERA);
    assert(bookmark, `${SCENE} has no camera ${CAMERA}`);
    return bookmark;
}

/** The view's GI-only mean. */
async function viewMean(scene: DdgiScene, view: TestView): Promise<number> {
    const field = scene.ddgi.field;
    assert(field, 'no field');
    return imageMean((await view.render(field, viewOf(scene), WIDTH, HEIGHT)).indirect);
}

/** Converges a scene (the file comment) and returns the view's mean. */
async function converge(scene: DdgiScene, view: TestView, log: (line: string) => void, label: string): Promise<number> {
    const device = scene.platform.device;
    const probeCount = scene.ddgi.field!.grid.probeCount;
    const settings = scene.ddgi.settings;
    const defaults = parseHysteresisMode('default');
    applyHysteresisMode(settings, { name: 'fast', settings: { ...FIXED_097.settings, hysteresis: FAST_HYSTERESIS } });
    await runFrames(device, scene.platform, FAST_FRAMES, probeCount);
    applyHysteresisMode(settings, FIXED_097);
    let mean = await viewMean(scene, view);
    let settled = false;
    for (let block = 1; block <= MAX_BLOCKS; block++) {
        await runFrames(device, scene.platform, BLOCK_FRAMES, probeCount);
        const next = await viewMean(scene, view);
        const change = next / mean - 1;
        mean = next;
        if (block >= MIN_BLOCKS && Math.abs(change) < CONVERGED_TOLERANCE) {
            settled = true;
            log(`${label}: converged after ${block} blocks, view mean ${next.toPrecision(6)}`);
            break;
        }
    }
    assert(settled, `${label}: the view's mean did not settle within ${MAX_BLOCKS} blocks`);
    applyHysteresisMode(settings, defaults);
    await runFrames(device, scene.platform, SETTLE_FRAMES, probeCount);
    return viewMean(scene, view);
}

async function startScene(device: GPUDevice, changeDriven: boolean, lampAt: Vec3 | null): Promise<DdgiScene> {
    const scene = await createDdgiScene(device, SCENE, { changeDrivenHistory: changeDriven, ...(LIGHT_CHANGE_SCALE !== undefined ? { lightChangeScale: LIGHT_CHANGE_SCALE } : {}) }, { gpuTiming: true });
    if (lampAt) {
        scene.platform.scene.updateLight(scene.lights[0], { position: lampAt });
        // The move lands before any probe has been updated, so it marks nothing (every probe is still new).
        await runFrames(device, scene.platform, 1, 0);
    }
    return scene;
}

interface StepResult {
    readonly means: number[];
    /** The first frame from which every measured frame is within the tolerance of settled; null if none. */
    readonly settledFrame: number | null;
    readonly settledSeconds: number | null;
    readonly firstFrameError: number;
    readonly marks: ProbeSnapshot | null;
}

/** The step from A to B on a converged scene; the response against `settled`. */
async function step(scene: DdgiScene, view: TestView, to: Vec3, settled: number): Promise<StepResult> {
    const device = scene.platform.device;
    const probeCount = scene.ddgi.field!.grid.probeCount;
    scene.platform.scene.updateLight(scene.lights[0], { position: to });
    // Frame 1: the mark alone, so its marks can be read before any blend takes them in.
    await runFrames(device, scene.platform, 1, 0);
    const marks = scene.ddgi.settings.changeDrivenHistory ? await probeSnapshot(scene) : null;
    const means = [await viewMean(scene, view)];
    for (let frame = 2; frame <= RESPONSE_FRAMES; frame++) {
        await runFrames(device, scene.platform, 1, probeCount);
        means.push(await viewMean(scene, view));
    }
    const errors = means.map((mean) => Math.abs(mean / settled - 1));
    let settledFrame: number | null = null;
    for (let index = errors.length - 1; index >= 0 && errors[index] <= TOLERANCE; index--) settledFrame = index + 1;
    return { means, settledFrame, settledSeconds: settledFrame === null ? null : settledFrame * FRAME_SECONDS, firstFrameError: errors[0], marks };
}

/**
 * How the marks compare with the true change in size, not only in who is
 * marked: per probe (active, marked, really changed by more than the
 * tolerance), the true relative change over the mark's unscaled value. A
 * `lightChangeScale` at the median makes r an unbiased guess of the true
 * change for the median probe; quartiles show the spread.
 */
function scaleCalibration(marks: ProbeSnapshot, before: ProbeSnapshot, after: ProbeSnapshot, scale: number, include: (probe: number) => boolean) {
    const ratios: number[] = [];
    for (let probe = 0; probe < marks.pendingChange.length; probe++) {
        if (!include(probe) || !before.active[probe] || !after.active[probe] || !marks.active[probe]) continue;
        const unscaled = marks.pendingChange[probe] / scale;
        const trueChange = relativeChange(before.mean[probe], after.mean[probe]);
        if (unscaled > 0 && trueChange > TRUE_CHANGE) ratios.push(trueChange / unscaled);
    }
    ratios.sort((a, b) => a - b);
    const at = (share: number) => ratios[Math.min(ratios.length - 1, Math.floor(share * ratios.length))] ?? Number.NaN;
    return { probes: ratios.length, quartile1: at(0.25), median: at(0.5), quartile3: at(0.75) };
}

function relativeChange(before: Vec3, after: Vec3): number {
    let largest = 0;
    for (let channel = 0; channel < 3; channel++) largest = Math.max(largest, Math.abs(after[channel] - before[channel]) / Math.max(before[channel], 1e-6));
    return largest;
}

/** Check 4: the marks against the probes that really changed, over the probes `include` picks. */
function recall(marks: ProbeSnapshot, before: ProbeSnapshot, after: ProbeSnapshot, tolerance: number, scaleRatio: number, include: (probe: number) => boolean) {
    let changed = 0;
    let caught = 0;
    let cut = 0;
    let cutAndChanged = 0;
    for (let probe = 0; probe < marks.pendingChange.length; probe++) {
        if (!include(probe) || !before.active[probe] || !after.active[probe] || !marks.active[probe]) continue;
        const reallyChanged = relativeChange(before.mean[probe], after.mean[probe]) > TRUE_CHANGE;
        const isCut = marks.pendingChange[probe] * scaleRatio > tolerance;
        if (reallyChanged) changed++;
        if (isCut) cut++;
        if (reallyChanged && isCut) caught++;
        if (reallyChanged && isCut) cutAndChanged++;
    }
    return { changed, cut, caught, recall: changed > 0 ? caught / changed : 1, precision: cut > 0 ? cutAndChanged / cut : 1 };
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, unknown> = { scene: SCENE, camera: CAMERA, width: WIDTH, height: HEIGHT, responseFrames: RESPONSE_FRAMES };

        // The settled answer at the new place.
        const probe = await startScene(device, true, null);
        const lampA = (probe.manifest.lights[0] as { position: Vec3 }).position;
        const lampB: Vec3 = [0, 1, 2].map((axis) => lampA[axis] + STEP_METRES * STEP_DIRECTION[axis]) as Vec3;
        probe.platform.dispose();
        results.lampFrom = lampA;
        results.lampTo = lampB;

        const settledScene = await startScene(device, true, lampB);
        const settledView = await TestView.create(device, settledScene.platform);
        const settledB = await converge(settledScene, settledView, log, 'settled at the new place');
        const after = await probeSnapshot(settledScene);
        settledScene.platform.dispose();

        const runs: Record<string, StepResult & { settledBefore: number }> = {};
        let before: ProbeSnapshot | null = null;
        let tolerance = 0;
        let scale = 0;
        for (const changeDriven of [true, false]) {
            const scene = await startScene(device, changeDriven, null);
            const view = await TestView.create(device, scene.platform);
            const settledA = await converge(scene, view, log, changeDriven ? 'change-driven, before the step' : 'M1 rule, before the step');
            if (changeDriven) {
                before = await probeSnapshot(scene);
                tolerance = scene.ddgi.settings.changeTolerance;
                scale = scene.ddgi.settings.lightChangeScale;
            }
            const result = step(scene, view, lampB, settledB);
            const measured = { ...(await result), settledBefore: settledA };
            runs[changeDriven ? 'changeDriven' : 'm1Rule'] = measured;
            log(
                `${changeDriven ? 'change-driven history' : "M1's rule alone"}: view mean ${settledA.toPrecision(5)} before, settled ${settledB.toPrecision(5)} after; ` +
                    `frame 1 off by ${(measured.firstFrameError * 100).toFixed(1)}%; within ${TOLERANCE * 100}% from frame ${measured.settledFrame ?? 'never'} (${measured.settledSeconds?.toFixed(3) ?? '-'} s)`,
            );
            // The curve, so a failing run still shows it: the error against settled, frames 1-12 then every 6th.
            const curve = measured.means.map((mean, index) => ({ frame: index + 1, error: mean / settledB - 1 })).filter(({ frame }) => frame <= 12 || frame % 6 === 0);
            log(`  error by frame: ${curve.map(({ frame, error }) => `${frame}: ${(error * 100).toFixed(1)}%`).join(', ')}`);
            scene.platform.dispose();
        }

        const changeDriven = runs.changeDriven;
        assert(before && changeDriven.marks, 'no marks read');
        const marks = changeDriven.marks;
        const litRoom = (index: number) => marks.position[index][0] < LIT_ROOM_MAX_X;
        const everyProbe = () => true;
        const atScale = (candidate: number) => ({
            scale: candidate,
            litRoom: recall(marks, before!, after, tolerance, candidate / scale, litRoom),
            volume: recall(marks, before!, after, tolerance, candidate / scale, everyProbe),
        });
        const estimate = atScale(scale);
        results.lightChangeScale = scale;
        results.changeEstimate = estimate;
        results.otherScales = SCALES.map(atScale);
        const calibration = { litRoom: scaleCalibration(marks, before, after, scale, litRoom), volume: scaleCalibration(marks, before, after, scale, everyProbe) };
        results.scaleCalibration = calibration;
        log(
            `true change over the unscaled mark: lit room median ${calibration.litRoom.median.toFixed(3)} (quartiles ${calibration.litRoom.quartile1.toFixed(3)} - ${calibration.litRoom.quartile3.toFixed(3)}, ${calibration.litRoom.probes} probes); ` +
                `volume median ${calibration.volume.median.toFixed(3)} (${calibration.volume.quartile1.toFixed(3)} - ${calibration.volume.quartile3.toFixed(3)})`,
        );
        results.probesMarked = marks.pendingChange.filter((value) => value > 0).length;
        results.probesCut = marks.pendingChange.filter((value) => value > tolerance).length;
        results.responses = Object.fromEntries(Object.entries(runs).map(([name, run]) => [name, { settledBefore: run.settledBefore, settledAfter: settledB, settledFrame: run.settledFrame, settledSeconds: run.settledSeconds, firstFrameError: run.firstFrameError, means: run.means }]));
        log(
            `change estimate at lightChangeScale ${scale}: lit room recall ${(estimate.litRoom.recall * 100).toFixed(1)}% (${estimate.litRoom.caught} of ${estimate.litRoom.changed}), precision ${(estimate.litRoom.precision * 100).toFixed(1)}%; ` +
                `whole volume recall ${(estimate.volume.recall * 100).toFixed(1)}% (${estimate.volume.caught} of ${estimate.volume.changed}), precision ${(estimate.volume.precision * 100).toFixed(1)}%`,
        );
        for (const other of results.otherScales as ReturnType<typeof atScale>[]) {
            log(`  at scale ${other.scale}: lit room recall ${(other.litRoom.recall * 100).toFixed(1)}%, precision ${(other.litRoom.precision * 100).toFixed(1)}%; volume recall ${(other.volume.recall * 100).toFixed(1)}%, precision ${(other.volume.precision * 100).toFixed(1)}%`);
        }

        const failures: string[] = [];
        if (changeDriven.settledSeconds === null || changeDriven.settledSeconds > RESPONSE_SECONDS) {
            failures.push(`the lit-room view took ${changeDriven.settledSeconds?.toFixed(3) ?? 'more than the measured'} s to get within ${TOLERANCE * 100}% (target ${RESPONSE_SECONDS} s)`);
        }
        if (estimate.litRoom.recall < RECALL_TARGET) failures.push(`lit-room recall ${(estimate.litRoom.recall * 100).toFixed(1)}% is under ${RECALL_TARGET * 100}%`);
        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
