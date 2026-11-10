/**
 * T2.4b, the GPU scheduler at the calibrated batch (docs/design/m2-lights.md
 * "Draft done-whens" T2.4b, and TASKS "M2 done-when changes" → T2.4b), GI
 * only (the engine-free test view's indirect image at a bookmark, its mean
 * as the region of interest). Response times are counted in frames: 0.5 s is
 * 30 frames at the batch (DDGI's clock advances 1/60 s a frame).
 *
 * - **steps** (check 1, and the larger step with its overshoot): two-room
 *   (the `lit-room` view, 880 probes a frame) and point-light Cornell (the
 *   `front` view, 1,463), the RTX 3060's calibrated batches at 2 ms on this
 *   code (PROGRESS, T2.4b). Two steps each, both changing the view by far
 *   more than 25%: the lamp switched on (0 → full) and halved (full → half).
 *   Each must show in the step's own frame (the view moves by more than 1%
 *   of the larger of before and settled) and be within 5% of settled within
 *   30 frames, and stay there: a view that enters the band and leaves it
 *   again isn't settled until it is back for good. The overshoot (how far
 *   past settled the view goes, as a share of the step) is reported.
 *   Settled: a second scene converged in the new state from the start.
 * - **sweep** (the steady-change rule): the two-room lamp glides along x at
 *   SWEEP_SPEED m/s; at 3 moments the view is compared with a field
 *   converged with the lamp where it is at that moment, and must be within
 *   5% of it, as a mean and per pixel (the mean over pixels of |view −
 *   converged|, over the converged mean; TASKS, T2.4c: the swinging lamp
 *   hardly moves the view's mean).
 * - **census** (check 3, on two-room): the lamp swings back and forth for
 *   60 s (3,600 frames); no probe may go more than 2 full passes of the round
 *   robin without being listed (ScheduleGapCensus). A pass of the round
 *   robin is the probes over the list's share it always keeps.
 *
 * Converging (ConvergeOptions): every probe a frame at a fast fixed
 * hysteresis, then the fixed 0.97 until the view settles, then 1,200 frames
 * of the default rule at the batch, so every probe's history is at its
 * steady count before the step.
 *
 *   VITE_DDGI_SECTIONS=steps,sweep,census   (default all)
 *   VITE_DDGI_STEP_SCENES=two-room,cornell-box-point
 *
 * `pnpm test:gpu --filter ddgi-schedule-step --gpu nvidia`.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { FrameCamera } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { DEFAULT_PRIORITY_SHARE } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { priorityCount } from '../../../../packages/ddgi-core/src/techniques/ddgi/schedule.ts';
import { createDdgiScene, imageMean, loadManifest, releaseDeviceOnPageHide, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';
import { bookmarkCamera, convergeField, describeErrors, runCameraFrame, runCameraFrames, ScheduleGapCensus, stepResponse, type ConvergeOptions, type StepResponse } from '../ddgi-shared/response.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SECTIONS = new Set(listFromEnv(env.VITE_DDGI_SECTIONS, ['steps', 'sweep', 'census']));
/** The scenes, their view, and their calibrated batch (the RTX 3060 at 2 ms, VALID run on this code: PROGRESS 2026-11-06, T2.4b). */
const STEP_SCENES: Readonly<Record<string, { readonly camera: string; readonly units: number }>> = {
    'two-room': { camera: 'lit-room', units: 880 },
    'cornell-box-point': { camera: 'front', units: 1463 },
};
const SCENES = listFromEnv(env.VITE_DDGI_STEP_SCENES, Object.keys(STEP_SCENES));
const WIDTH = 320;
const HEIGHT = 180;
const TOLERANCE = 0.05;
/** 0.5 s at 60 Hz. */
const RESPONSE_FRAMES = 30;
/** The view must move by more than this in the step's own frame. */
const NEXT_FRAME_CHANGE = 0.01;
/** Frames measured after a step (2 s): long enough to see the view leave the band again, or not. */
const MEASURED_FRAMES = 120;
/** The steps must change the view by at least this (TASKS, the larger step). */
const MIN_STEP = 0.25;

/** The sweep: two-room's lamp along x, from SWEEP_FROM to SWEEP_TO, at SWEEP_SPEED m/s; compared at these seconds. */
const SWEEP_FROM_X = 0.8;
const SWEEP_TO_X = 3.2;
const SWEEP_SPEED = 1;
const SWEEP_CHECK_SECONDS = [0.8, 1.6, 2.4];
/** The census: 60 s of a lamp swinging over SWEEP_FROM..SWEEP_TO and back. */
const CENSUS_FRAMES = 3600;
const CENSUS_PASSES = 2;

const CONVERGE: Omit<ConvergeOptions, 'settleUnits'> = {
    fastFrames: 96,
    fastHysteresis: 0.5,
    blockFrames: 32,
    convergedTolerance: 0.001,
    maxBlocks: 24,
    settleFrames: 1200,
    convergeUnits: 4096,
};

type Vec3 = [number, number, number];

interface SceneUnderTest {
    readonly scene: DdgiScene;
    readonly view: TestView;
    readonly camera: FrameCamera;
    readonly mean: () => Promise<number>;
    /** The view's indirect image (rgb per pixel). */
    readonly image: () => Promise<Float32Array>;
}

/** The mean over pixels of |a − b|, over the mean of b: a per-pixel error no averaging over the view can hide. */
function perPixelError(image: Float32Array, reference: Float32Array): number {
    let difference = 0;
    let sum = 0;
    for (let index = 0; index < image.length; index++) {
        difference += Math.abs(image[index] - reference[index]);
        sum += reference[index];
    }
    return sum > 0 ? difference / sum : 0;
}

async function startScene(device: GPUDevice, sceneId: string, lamp: { intensity?: number; position?: Vec3 }, settings: { changeDrivenHistory?: boolean } = {}): Promise<SceneUnderTest> {
    const scene = await createDdgiScene(device, sceneId, settings, { gpuTiming: true });
    const bookmark = scene.manifest.cameras.find((camera) => camera.name === STEP_SCENES[sceneId].camera);
    assert(bookmark, `${sceneId} has no camera ${STEP_SCENES[sceneId].camera}`);
    const camera = bookmarkCamera(bookmark, WIDTH / HEIGHT);
    if (lamp.intensity !== undefined || lamp.position !== undefined) {
        scene.platform.scene.updateLight(scene.lights[0], lamp);
        // Before any probe is updated: it marks nothing (every probe is still new).
        runCameraFrame(device, scene.platform, 0, camera);
    }
    const view = await TestView.create(device, scene.platform);
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const image = async () => (await view.render(field, bookmark, WIDTH, HEIGHT)).indirect;
    const mean = async () => imageMean(await image());
    return { scene, view, camera, mean, image };
}

async function converged(
    device: GPUDevice,
    sceneId: string,
    lamp: { intensity?: number; position?: Vec3 },
    label: string,
    log: (line: string) => void,
    settings: { changeDrivenHistory?: boolean } = {},
): Promise<SceneUnderTest & { settled: number }> {
    const test = await startScene(device, sceneId, lamp, settings);
    const settled = await convergeField(device, test.scene.platform, test.scene.ddgi.settings, test.camera, test.mean, { ...CONVERGE, settleUnits: STEP_SCENES[sceneId].units }, label, log);
    return { ...test, settled };
}

/** The step on a converged scene, measured frame by frame against `settled`. */
async function measureStep(device: GPUDevice, test: SceneUnderTest, units: number, change: { intensity?: number; position?: Vec3 }, before: number, settled: number): Promise<StepResponse & { means: number[] }> {
    test.scene.platform.scene.updateLight(test.scene.lights[0], change);
    const means: number[] = [];
    for (let frame = 1; frame <= MEASURED_FRAMES; frame++) {
        runCameraFrame(device, test.scene.platform, units, test.camera);
        means.push(await test.mean());
    }
    return { ...stepResponse(before, settled, means, TOLERANCE), means };
}

function describeStep(name: string, response: StepResponse, before: number, settled: number, firstFrameMove: number): string {
    return (
        `${name}: view ${before.toPrecision(4)} -> settled ${settled.toPrecision(4)}; ` +
        `frame 1 moved ${(firstFrameMove * 100).toFixed(1)}%; ` +
        `within ${TOLERANCE * 100}% from frame ${response.enteredFrame ?? 'never'}, for good from frame ${response.settledFrame ?? 'never'}${response.leftBandAgain ? ' (left the band again)' : ''}; ` +
        `overshoot ${(response.overshootShareOfStep * 100).toFixed(1)}% of the step (${(response.overshoot * 100).toFixed(2)}% of settled)\n    error by frame: ${describeErrors(response.errors)}`
    );
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, unknown> = { responseFrames: RESPONSE_FRAMES, tolerance: TOLERANCE };
        const failures: string[] = [];

        if (SECTIONS.has('steps')) {
            for (const sceneId of SCENES) {
                const { units } = STEP_SCENES[sceneId];
                const fullIntensity = loadManifest(sceneId).lights[0].intensity;
                const half = await converged(device, sceneId, { intensity: fullIntensity / 2 }, `${sceneId} at half`, log);
                const halfSettled = half.settled;
                half.scene.platform.dispose();
                const full = await converged(device, sceneId, {}, `${sceneId} at full`, log);
                const fullSettled = full.settled;
                // Dark (the lamp is the only light): nothing to converge, only the histories to settle at the batch, as the others did.
                const off = await startScene(device, sceneId, { intensity: 0 });
                await runCameraFrames(device, off.scene.platform, CONVERGE.settleFrames, units, off.camera);
                const cases: Array<{ name: string; test: SceneUnderTest; change: { intensity: number }; before: number; settled: number }> = [
                    { name: `${sceneId}: the lamp switched on`, test: off, change: { intensity: fullIntensity }, before: await off.mean(), settled: fullSettled },
                    { name: `${sceneId}: the lamp halved`, test: full, change: { intensity: fullIntensity / 2 }, before: fullSettled, settled: halfSettled },
                ];
                const sceneResults: Record<string, unknown> = { units, priorityShare: priorityCount(units, DEFAULT_PRIORITY_SHARE) };
                for (const step of cases) {
                    const response = await measureStep(device, step.test, units, step.change, step.before, step.settled);
                    const firstFrameMove = Math.abs(response.means[0] - step.before) / Math.max(step.before, step.settled);
                    const stepSize = Math.abs(step.settled - step.before) / Math.max(step.before, step.settled);
                    log(describeStep(step.name, response, step.before, step.settled, firstFrameMove));
                    sceneResults[step.name] = { before: step.before, settled: step.settled, firstFrameMove, stepSize, ...response };
                    if (stepSize < MIN_STEP) failures.push(`${step.name}: the step changed the view by only ${(stepSize * 100).toFixed(1)}%`);
                    if (!(firstFrameMove > NEXT_FRAME_CHANGE)) failures.push(`${step.name}: the step's own frame moved the view by ${(firstFrameMove * 100).toFixed(2)}%`);
                    if (response.settledFrame === null || response.settledFrame > RESPONSE_FRAMES) failures.push(`${step.name}: within ${TOLERANCE * 100}% for good only from frame ${response.settledFrame ?? 'never'} (bar ${RESPONSE_FRAMES})`);
                }
                results[sceneId] = sceneResults;
                full.scene.platform.dispose();
                off.scene.platform.dispose();
            }
        }

        if (SECTIONS.has('sweep')) {
            const sceneId = 'two-room';
            const { units } = STEP_SCENES[sceneId];
            const lamp = loadManifest(sceneId).lights[0] as { position: Vec3 };
            const at = (x: number): Vec3 => [x, lamp.position[1], lamp.position[2]];
            const references: Array<{ mean: number; image: Float32Array }> = [];
            for (const seconds of SWEEP_CHECK_SECONDS) {
                const reference = await converged(device, sceneId, { position: at(SWEEP_FROM_X + SWEEP_SPEED * seconds) }, `${sceneId}, the lamp at x = ${(SWEEP_FROM_X + SWEEP_SPEED * seconds).toFixed(2)}`, log);
                references.push({ mean: reference.settled, image: await reference.image() });
                reference.scene.platform.dispose();
            }
            // The sweep with change-driven history, and under M1's rule beside it (does the check tell them apart?).
            const sweeps: Record<string, Array<{ seconds: number; mean: number; reference: number; error: number; perPixelError: number }>> = {};
            for (const changeDrivenHistory of [true, false]) {
                const sweep = await converged(device, sceneId, { position: at(SWEEP_FROM_X) }, `${sceneId}, the sweep's start${changeDrivenHistory ? '' : " (M1's rule)"}`, log, { changeDrivenHistory });
                const checks: Array<{ seconds: number; mean: number; reference: number; error: number; perPixelError: number }> = [];
                const frames = Math.round(SWEEP_CHECK_SECONDS[SWEEP_CHECK_SECONDS.length - 1] * 60);
                for (let frame = 1; frame <= frames; frame++) {
                    sweep.scene.platform.scene.updateLight(sweep.scene.lights[0], { position: at(SWEEP_FROM_X + (SWEEP_SPEED * frame) / 60) });
                    runCameraFrame(device, sweep.scene.platform, units, sweep.camera);
                    const index = SWEEP_CHECK_SECONDS.findIndex((seconds) => Math.round(seconds * 60) === frame);
                    if (index >= 0) {
                        const image = await sweep.image();
                        const mean = imageMean(image);
                        const reference = references[index];
                        checks.push({ seconds: SWEEP_CHECK_SECONDS[index], mean, reference: reference.mean, error: mean / reference.mean - 1, perPixelError: perPixelError(image, reference.image) });
                    } else if (frame % 8 === 0) {
                        await device.queue.onSubmittedWorkDone();
                    }
                }
                sweep.scene.platform.dispose();
                sweeps[changeDrivenHistory ? 'changeDriven' : 'm1Rule'] = checks;
                log(
                    `sweep at ${SWEEP_SPEED} m/s, ${units} a frame${changeDrivenHistory ? '' : ", M1's rule"}: ` +
                        checks.map((check) => `at ${check.seconds} s view ${(check.error * 100).toFixed(1)}%, per pixel ${(check.perPixelError * 100).toFixed(1)}%`).join('; ') +
                        ' (against fields converged there)',
                );
            }
            results.sweep = { speedMetresPerSecond: SWEEP_SPEED, units, ...sweeps };
            for (const check of sweeps.changeDriven) {
                if (Math.abs(check.error) > TOLERANCE) failures.push(`sweep: ${(check.error * 100).toFixed(1)}% from the converged field at ${check.seconds} s`);
                // Per pixel too (TASKS, T2.4c): the swinging lamp hardly moves the view's mean, so the mean alone can't tell rules apart.
                if (check.perPixelError > TOLERANCE) failures.push(`sweep: per pixel ${(check.perPixelError * 100).toFixed(1)}% from the converged field at ${check.seconds} s`);
            }
        }

        if (SECTIONS.has('census')) {
            const sceneId = 'two-room';
            const { units } = STEP_SCENES[sceneId];
            const test = await startScene(device, sceneId, {});
            const field = test.scene.ddgi.field;
            assert(field, 'no field');
            const lamp = test.scene.manifest.lights[0] as { position: Vec3 };
            const census = await ScheduleGapCensus.create(device, test.scene.platform, field);
            // Two full passes first, so every probe has a stamp.
            await runCameraFrames(device, test.scene.platform, 2 * Math.ceil(field.grid.probeCount / units) + 2, units, test.camera);
            const span = SWEEP_TO_X - SWEEP_FROM_X;
            for (let frame = 1; frame <= CENSUS_FRAMES; frame++) {
                // Back and forth at SWEEP_SPEED.
                const travelled = ((SWEEP_SPEED * frame) / 60) % (2 * span);
                const x = SWEEP_FROM_X + (travelled < span ? travelled : 2 * span - travelled);
                test.scene.platform.scene.updateLight(test.scene.lights[0], { position: [x, lamp.position[1], lamp.position[2]] });
                runCameraFrame(device, test.scene.platform, units, test.camera);
                census.measure(test.scene.platform.stats.frame);
                if (frame % 8 === 0) await device.queue.onSubmittedWorkDone();
            }
            const gap = await census.read();
            census.dispose();
            test.scene.platform.dispose();
            const roundRobinShare = units - priorityCount(units, DEFAULT_PRIORITY_SHARE);
            const passFrames = Math.ceil(field.grid.probeCount / roundRobinShare);
            results.census = { units, frames: CENSUS_FRAMES, longestGapFrames: gap.frames, atProbe: gap.probe, roundRobinPassFrames: passFrames };
            log(`census, ${CENSUS_FRAMES} frames of a swinging lamp at ${units} a frame: the longest any probe waited was ${gap.frames} frames (probe ${gap.probe}); a pass of the round robin is ${passFrames} frames`);
            if (gap.frames > CENSUS_PASSES * passFrames) failures.push(`census: a probe waited ${gap.frames} frames, more than ${CENSUS_PASSES} passes of the round robin (${CENSUS_PASSES * passFrames})`);
        }

        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
