/**
 * T2.1 (TASKS, "M2 done-when changes"): a Sponza lamp step's response on the
 * key-probe path. T2.4c's review found that only the Sponza lamp's recall had
 * been checked, not how fast the view follows it.
 *
 * The lamp is Sponza's own `arcade-lamp` (off in the manifest), played by the
 * manifest's `light-step` motion (`motionState`): it jumps along the north
 * arcade from keyframe 0 to keyframe 1, then back 3 s later. Sponza runs at
 * the batch the platform calibrates for 2 ms, as the viewer runs it; on a
 * volume this size every change takes the key-probe path there (19,425
 * probes x K updates don't fit in 120 frames), which the page checks: the
 * frame's `keyProbes` path is on, and keys pass the change on.
 *
 * The view is one the step really moves, GI only (the test view's indirect
 * light, so three's own direct light doesn't count): from the arcade's east
 * end toward the wall behind the lamp's second place. Reported: how much the
 * step moves the view (wanted at least 25%, as T2.4b's steps), the first
 * frame within 5% of settled for good, the overshoot as a share of the step,
 * and the same for the step back.
 *
 * "Settled" for each place is a field converged with the lamp there (fast
 * frames at 4,096 probes, then the fixed 0.97 until the view stops moving,
 * as T2.4b/c's Sponza pages converge).
 *
 * Manual (needs the Sponza download):
 *   node tools/reference/bench-lock.ts run --no-cool-wait --what "T2.1 Sponza lamp step" -- pnpm test:gpu --gpu nvidia --filter ddgi-sponza-lamp-step
 */

import type { PointLight } from 'three/webgpu';
import { assert } from '../../client.ts';
import { BUILT_IN_MOTIONS, motionState, STEP_SECONDS, type CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { FAR_RULE_FRAMES } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { imageMean } from '../ddgi-shared/harness.ts';
import { describeErrors, stepResponse } from '../ddgi-shared/response.ts';
import { convergeSponza, sponzaFrames, startSponza, VIEW_HEIGHT, VIEW_WIDTH, type SponzaUnderTest } from '../ddgi-shared/sponza-response.ts';
import { yieldToCallbacks } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const BUDGET_MS = 2;
const WARM_UP_PASSES = 4;
const CALIBRATION_MAX_FRAMES = 600;
const TOLERANCE = 0.05;
/** A step the view must really see (T2.4b's rule for its steps). */
const MIN_STEP = 0.25;
/** The motion's own time between jumps, in frames at 60 Hz. */
const STEP_FRAMES = Math.round(STEP_SECONDS * 60);
/** From the arcade's east end, toward the arcade wall behind the lamp's second place (x 4 m). */
const VIEW: CameraBookmark = { name: 'arcade-east', position: [8.5, 1.7, 3.9], target: [3.5, 1.2, 5.9], fovYDeg: 70 };

type Vec3 = [number, number, number];

/** Sponza with its arcade lamp lit where the step motion puts it at `seconds`, and the view above. */
async function start(device: GPUDevice, label: string, seconds: number): Promise<{ test: SponzaUnderTest; lamp: PointLight; place: (time: number) => void }> {
    const base = await startSponza(device, label);
    const manifest = base.scene.manifest;
    const motion = manifest.motions?.find((candidate) => candidate.id === BUILT_IN_MOTIONS.step);
    assert(motion, 'Sponza has no light-step motion');
    // Off in the manifest, so loaded aside (`offLights`); the motion lights it, as a player would.
    const lamp = base.scene.loaded.offLights.find((light) => light.name === motion.target.slice('light:'.length)) as PointLight | undefined;
    assert(lamp && lamp.isPointLight, `no ${motion.target} among the loaded scene's off lights`);
    base.scene.loaded.scene.add(lamp);
    const place = (time: number) => {
        const state = motionState(motion, time);
        if (state.position) lamp.position.set(...(state.position as Vec3));
        if (state.intensity !== undefined) lamp.intensity = state.intensity;
        lamp.updateMatrixWorld();
        base.scene.threeGI.markChanged(lamp);
    };
    place(seconds);
    // The scheduler orders probes by the adapter's camera: the one the view is measured from.
    base.scene.camera.fov = VIEW.fovYDeg;
    base.scene.camera.position.set(...VIEW.position);
    base.scene.camera.lookAt(...VIEW.target);
    base.scene.camera.updateProjectionMatrix();
    base.scene.camera.updateMatrixWorld();
    const field = base.scene.ddgi.field!;
    const images = async () => base.view.render(field, VIEW, VIEW_WIDTH, VIEW_HEIGHT);
    const test: SponzaUnderTest = { ...base, images, mean: async () => imageMean((await images()).indirect) };
    return { test, lamp, place };
}

/** Warm-up and Budget v2's calibration at 2 ms, as the viewer runs; returns the batch. */
async function calibrate(test: SponzaUnderTest): Promise<number> {
    const { scene } = test;
    scene.ddgi.settings.warmUpPasses = WARM_UP_PASSES;
    scene.ddgi.resetHistory();
    let steps = 0;
    while (scene.ddgi.warmingUp) {
        scene.frame({ paused: false }, 0);
        if (++steps % 4 === 0) {
            await scene.device.queue.onSubmittedWorkDone();
            await yieldToCallbacks();
        }
    }
    scene.ddgi.settings.warmUpPasses = 0;
    for (let frame = 0; frame < CALIBRATION_MAX_FRAMES; frame++) {
        scene.frame({ gpuMs: BUDGET_MS }, 0);
        const report = scene.platform.budgetReport(scene.ddgi);
        if (!(report?.state === 'calibrating' || report?.confirming === true)) break;
        if (frame % 4 === 3) {
            await scene.device.queue.onSubmittedWorkDone();
            await yieldToCallbacks();
        }
    }
    await scene.device.queue.onSubmittedWorkDone();
    const calibration = scene.platform.budgetReport(scene.ddgi)?.calibration;
    assert(calibration?.measured, 'the platform did not calibrate');
    return calibration.units;
}

await runThreePage(async ({ device, data, log: logLine }) => {
    const log = (line: string): void => {
        logLine(line);
        console.log(line);
    };
    const results = data;
    const failures: string[] = [];

    // The settled view with the lamp at the motion's second place.
    const reference = await start(device, 'sponza lamp at its second place', STEP_SECONDS);
    const settledSecond = await convergeSponza(reference.test, 'the lamp at its second place', log, null);
    reference.test.scene.dispose();

    // The step itself, at the calibrated batch.
    const { test, place } = await start(device, 'sponza lamp step', 0);
    const units = await calibrate(test);
    const probes = test.scene.probeCount;
    const fullUpdates = test.scene.ddgi.settings.changeUpdates;
    const framesForEveryProbe = (probes * fullUpdates) / units;
    results.units = units;
    results.keyPath = { probes, framesForEveryProbe: Number(framesForEveryProbe.toFixed(1)), ruleFrames: FAR_RULE_FRAMES, on: framesForEveryProbe > FAR_RULE_FRAMES && test.scene.ddgi.settings.keyProbes !== false };
    if (!(results.keyPath as { on: boolean }).on) failures.push(`the key-probe path is off at ${units} probes a frame (${framesForEveryProbe.toFixed(0)} frames for every probe's K updates)`);
    const settledFirst = await convergeSponza(test, 'the lamp at its first place', log, units);
    const passedBefore = test.scene.ddgi.counters['key changes passed (total)'] ?? 0;

    const steps: Record<string, unknown>[] = [];
    for (const [name, time, before, settled] of [
        ['to the second place', STEP_SECONDS, settledFirst, settledSecond],
        ['back to the first place', 2 * STEP_SECONDS, settledSecond, settledFirst],
    ] as const) {
        // The view as it is before the jump (the field at the batch, not the converged value).
        const now = await test.mean();
        place(time);
        const means: number[] = [];
        for (let frame = 1; frame <= STEP_FRAMES; frame++) {
            test.scene.frame({ units });
            means.push(await test.mean());
            if (frame % 4 === 0) await yieldToCallbacks();
        }
        const response = stepResponse(now, settled, means, TOLERANCE);
        const stepSize = Math.abs(settled - before) / Math.max(before, settled);
        steps.push({ name, before: now, settled, stepSize, ...response, means });
        log(
            `the lamp jumps ${name}: the view ${now.toPrecision(4)} -> ${settled.toPrecision(4)} (the step is ${(stepSize * 100).toFixed(0)}% of the view); ` +
                `within ${TOLERANCE * 100}% for good from frame ${response.settledFrame ?? 'never'} (first in the band at frame ${response.enteredFrame ?? 'never'}${response.leftBandAgain ? ', left it again' : ''}); ` +
                `overshoot ${(response.overshootShareOfStep * 100).toFixed(1)}% of the step\n    error by frame: ${describeErrors(response.errors)}`,
        );
        if (stepSize < MIN_STEP) failures.push(`the step ${name} moves the view by only ${(stepSize * 100).toFixed(1)}%`);
        if (response.settledFrame === null) failures.push(`the step ${name}: the view never settled within 5% in ${STEP_FRAMES} frames`);
        await sponzaFrames(test, 0, units);
    }
    const passedAfter = test.scene.ddgi.counters['key changes passed (total)'] ?? 0;
    results.keyProbesPassingAChange = passedAfter - passedBefore;
    if (!(passedAfter > passedBefore)) failures.push('no key probe passed a change on: the step did not take the key-probe path');
    log(`calibrated at ${BUDGET_MS} ms to ${units} probes a frame; key path on (${framesForEveryProbe.toFixed(0)} frames for every probe's K updates, over ${FAR_RULE_FRAMES}); key probes passed a change on ${passedAfter - passedBefore} times`);
    results.steps = steps;
    test.scene.dispose();
    assert(failures.length === 0, failures.join('; '));
}, { width: 320, height: 180 });
