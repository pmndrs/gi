/**
 * T2.4c on Sponza (docs/design/m2-lights.md §4.5, the T2.4c done-when):
 * key probes and propagation after a global change, through the three
 * adapter as the viewer runs it, at the RTX 3060's calibrated batch at 2 ms
 * (316 probes a frame, T2.4b's VALID calibration, so the numbers compare
 * with T2.4b's), GI only, bookmark 1 (`colonnade-from-the-floor`), whose
 * camera the scheduler orders probes by. Counted in frames (DDGI's clock
 * advances 1/60 s a frame: 15 frames is 0.25 s, 120 is 2 s).
 *
 * - **turn** (check 1): the sun turned 30° lower with key probes on. The
 *   view must be within 5% of settled for good within 15 frames, and the
 *   field's light-weighted error over all probes (T2.4b's measure,
 *   DECISIONS 2026-11-06) within 5% within 120 frames; the p95 of the
 *   per-probe errors is recorded, and when the last fine probe stopped
 *   waiting for its keys (every fine probe traced once since the step).
 *   Also check 4's measurement for this path (below) and the side-by-side
 *   image's left half.
 * - **halved** (TASKS, beside the turn): the sun halved with key probes on,
 *   T2.4b's check 2 (the view within 5% for good within 30 frames), reported.
 * - **hold** (check 2, the decision rule): in the test mode that holds fine
 *   probes back from their own updates (`holdFineProbes`), propagation alone
 *   must remove at least 60% of the change at the fine probes it serves
 *   (the light-weighted error the change left, over the volume), for the sun
 *   turn and for a sky dimming (the sky at a quarter).
 * - **fallback** (check 4, and the image's right half): the same turn with
 *   key probes off ("far probes owe 2"). At 3 s after the step (frames
 *   180-209), the out-of-view fine probes' noise (the standard deviation of
 *   each probe's mean irradiance over 30 frames, as a share of its mean,
 *   light-weighted: Σ std / Σ mean) with key probes on must be no worse than
 *   this.
 *
 * The side-by-side image: the GI-only view with and without propagation at
 * the frame the view settled with it, from bookmark 1 and from the upper
 * gallery, whose far probes the scheduler leaves to the keys (images in the
 * JSON, `images`).
 *
 * Converged references: the sun turned, the sun halved and the sky dimmed,
 * each converged from the start (ddgi-shared/sponza-response.ts).
 *
 *   VITE_DDGI_SECTIONS=turn,halved,hold,fallback   (default all)
 *   VITE_DDGI_SETTINGS=<json>                      (DDGI settings for the turn's run, Object.assign-ed: for comparing rules)
 *
 * Manual (needs the Sponza download): `pnpm test:gpu --filter ddgi-key-probes-sponza --gpu nvidia`.
 */

import { assert } from '../../client.ts';
import { priorityCount } from '../../../../packages/ddgi/src/schedule.ts';
import { DEFAULT_PRIORITY_SHARE } from '../../../../packages/ddgi/src/settings.ts';
import { keyProbeCount, PROBE_HISTORY_FLAG } from '../../../../packages/ddgi/src/layout.ts';
import { imageMean } from '../ddgi-shared/harness.ts';
import { describeErrors, stepResponse } from '../ddgi-shared/response.ts';
import {
    aimSun,
    awaitingCount,
    changeTakenIn,
    convergeSponza,
    dimSky,
    lightWeightedError,
    meanOf,
    pngBase64,
    probeClasses,
    probeErrors,
    probeMeans,
    quantile,
    startSponza,
    sunDirection,
    turnedSun,
    VIEW_HEIGHT,
    VIEW_WIDTH,
    type ProbeMeans,
    type SponzaUnderTest,
} from '../ddgi-shared/sponza-response.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SECTIONS = new Set(listFromEnv(env.VITE_DDGI_SECTIONS, ['turn', 'halved', 'hold', 'fallback']));
const UNITS = 316;
const TOLERANCE = 0.05;
/** Check 1's bounds, in frames. */
const VIEW_FRAMES = 15;
const VOLUME_FRAMES = 120;
/** ...and the bound past which the main session decides with Dennis (50% over). */
const VOLUME_FRAMES_DECISION = 180;
/** The sun halved: T2.4b's bound. */
const HALVED_VIEW_FRAMES = 30;
const MEASURED_FRAMES = 240;
/** The volume's error is read every this many frames (and at frame 1). */
const VOLUME_EVERY = 6;
const SUN_TURN_DEGREES = 30;
/** Check 2: the decision rule, and how long the held run lasts (the keys' owed updates are served well within it). */
const PROPAGATION_REMOVES = 0.6;
const HOLD_FRAMES = 240;
const SKY_DIMMING = 0.25;
/** Check 4: the noise window, frames after the step. */
const NOISE_FIRST_FRAME = 180;
const NOISE_FRAMES = 30;
/** The side-by-side image's exposure (Sponza's GI-only view is dark). */
const IMAGE_EXPOSURE = 8;

interface StepRun {
    readonly viewMeans: number[];
    readonly volume: Array<{ frame: number; weighted: number; weightedFine: number; weightedKeys: number; mean: number; p95: number; awaiting: number }>;
    readonly lastAwaitingFrame: number | null;
    readonly noise: { lightWeighted: number; meanRelative: number; probes: number } | null;
    /** The view's image at a frame, from bookmark 1 (the scheduler's camera) and from the upper gallery (out of its view). */
    readonly imageAt: (frame: number) => { readonly view: string; readonly elsewhere: string } | undefined;
    readonly counters: Readonly<Record<string, number>>;
}

/**
 * The sun turned on a converged, settled field, `frames` frames at UNITS,
 * the view's mean every frame, the field's error every VOLUME_EVERY frames,
 * the noise window's probe means, and the view's image every frame up to
 * `imageFrames` (kept as PNGs, for the side-by-side).
 */
async function sunTurnRun(test: SponzaUnderTest, reference: ProbeMeans, direction: ReturnType<typeof sunDirection>, imageFrames: number): Promise<StepRun> {
    const field = test.scene.ddgi.field!;
    const classes = probeClasses(test, field);
    aimSun(test, turnedSun(direction, SUN_TURN_DEGREES));
    const viewMeans: number[] = [];
    const volume: StepRun['volume'] = [];
    const images = new Map<number, { view: string; elsewhere: string }>();
    const elsewhere = test.scene.manifest.cameras[1];
    const noiseMeans: ProbeMeans[] = [];
    let lastAwaitingFrame: number | null = null;
    const fine = (probe: number) => !classes.key(probe);
    for (let frame = 1; frame <= MEASURED_FRAMES; frame++) {
        test.scene.frame({ units: UNITS });
        const view = await test.images();
        viewMeans.push(imageMean(view.indirect));
        if (frame <= imageFrames) {
            const other = await test.view.render(field, elsewhere, VIEW_WIDTH, VIEW_HEIGHT);
            images.set(frame, { view: await pngBase64(view.indirect, VIEW_WIDTH, VIEW_HEIGHT, IMAGE_EXPOSURE), elsewhere: await pngBase64(other.indirect, VIEW_WIDTH, VIEW_HEIGHT, IMAGE_EXPOSURE) });
        }
        const inNoiseWindow = frame >= NOISE_FIRST_FRAME && frame < NOISE_FIRST_FRAME + NOISE_FRAMES;
        if (frame % VOLUME_EVERY === 0 || frame === 1 || inNoiseWindow) {
            const now = await probeMeans(test);
            if (inNoiseWindow) noiseMeans.push(now);
            if (frame % VOLUME_EVERY === 0 || frame === 1) {
                const { errors } = probeErrors(now, reference);
                const awaiting = awaitingCount(now);
                if (awaiting > 0) lastAwaitingFrame = frame;
                volume.push({
                    frame,
                    weighted: lightWeightedError(now, reference),
                    weightedFine: lightWeightedError(now, reference, fine),
                    weightedKeys: lightWeightedError(now, reference, classes.key),
                    mean: meanOf(errors),
                    p95: quantile(errors, 0.95),
                    awaiting,
                });
            }
        }
    }
    // Check 4: the out-of-view fine probes' noise over the window.
    let noise: StepRun['noise'] = null;
    if (noiseMeans.length === NOISE_FRAMES) {
        let stdSum = 0;
        let meanSum = 0;
        const relative: number[] = [];
        const probeCount = reference.mean.length;
        for (let probe = 0; probe < probeCount; probe++) {
            if (classes.key(probe) || classes.near(probe) || !noiseMeans.every((means) => means.active[probe])) continue;
            const luminance = noiseMeans.map((means) => (means.mean[probe][0] + means.mean[probe][1] + means.mean[probe][2]) / 3);
            const average = meanOf(luminance);
            const variance = meanOf(luminance.map((value) => (value - average) ** 2));
            stdSum += Math.sqrt(variance);
            meanSum += average;
            if (average > 0) relative.push(Math.sqrt(variance) / average);
        }
        noise = { lightWeighted: meanSum > 0 ? stdSum / meanSum : 0, meanRelative: meanOf(relative), probes: relative.length };
    }
    return { viewMeans, volume, lastAwaitingFrame: lastAwaitingFrame === null ? null : lastAwaitingFrame + VOLUME_EVERY, noise, imageAt: (frame) => images.get(frame), counters: test.scene.ddgi.counters };
}

await runThreePage(async ({ device, data, log: logLine }) => {
    const log = (line: string): void => {
        logLine(line);
        console.log(line);
    };
    const results = data;
    results.units = UNITS;
    results.priorityShare = priorityCount(UNITS, DEFAULT_PRIORITY_SHARE);
    const failures: string[] = [];
    const decisions: string[] = [];
    const images: Array<{ name: string; base64: string }> = [];
    results.images = images;

    // The references, converged from the start.
    const needsTurn = SECTIONS.has('turn') || SECTIONS.has('hold') || SECTIONS.has('fallback');
    // The sun as the scene has it, from the first scene started.
    const first = await startSponza(device, needsTurn ? 'sponza sun turned reference' : 'sponza');
    const startDirection = sunDirection(first.sun);
    const fullSun = first.sun.intensity;
    const grid = first.scene.ddgi.field!.grid;
    results.keyProbes = keyProbeCount(grid.counts);
    results.probes = grid.probeCount;
    let turnedReference: ProbeMeans | null = null;
    let turnedSettled = Number.NaN;
    if (!needsTurn) first.scene.dispose();
    if (needsTurn) {
        const turned = first;
        aimSun(turned, turnedSun(startDirection, SUN_TURN_DEGREES));
        turnedSettled = await convergeSponza(turned, `the sun ${SUN_TURN_DEGREES} degrees lower, from the start`, log, null);
        turnedReference = await probeMeans(turned);
        turned.scene.dispose();
    }

    let keyRun: StepRun | null = null;
    let keyViewSettled: number | null = null;
    if (SECTIONS.has('turn')) {
        assert(turnedReference, 'no reference');
        const test = await startSponza(device, 'sponza sun turned, key probes');
        await convergeSponza(test, 'the sun as it was', log, UNITS);
        const before = await test.mean();
        if (typeof env.VITE_DDGI_SETTINGS === 'string' && env.VITE_DDGI_SETTINGS !== '') Object.assign(test.scene.ddgi.settings, JSON.parse(env.VITE_DDGI_SETTINGS));
        keyRun = await sunTurnRun(test, turnedReference, startDirection, 60);
        test.scene.dispose();
        const response = stepResponse(before, turnedSettled, keyRun.viewMeans, TOLERANCE);
        keyViewSettled = response.settledFrame;
        const volumeWithin = keyRun.volume.find((entry, index) => keyRun!.volume.slice(index).every((later) => later.weighted <= TOLERANCE));
        const atBound = keyRun.volume.find((entry) => entry.frame === VOLUME_FRAMES);
        results.turn = {
            degrees: SUN_TURN_DEGREES,
            before,
            settled: turnedSettled,
            viewSettledFrame: response.settledFrame,
            viewOvershootShareOfStep: response.overshootShareOfStep,
            volumeWithinFrame: volumeWithin?.frame ?? null,
            atVolumeBound: atBound,
            lastFineProbeWaitingFrame: keyRun.lastAwaitingFrame,
            noise: keyRun.noise,
            volume: keyRun.volume,
            viewErrors: response.errors,
            counters: keyRun.counters,
        };
        log(
            `key probes, the sun ${SUN_TURN_DEGREES} degrees lower: view ${before.toPrecision(4)} -> ${turnedSettled.toPrecision(4)}, within ${TOLERANCE * 100}% for good from frame ${response.settledFrame ?? 'never'} (bar ${VIEW_FRAMES}); ` +
                `the field's light-weighted error within ${TOLERANCE * 100}% for good from frame ${volumeWithin?.frame ?? 'never'} (bar ${VOLUME_FRAMES}; at ${VOLUME_FRAMES}: ${((atBound?.weighted ?? NaN) * 100).toFixed(2)}%, fine ${((atBound?.weightedFine ?? NaN) * 100).toFixed(2)}%, keys ${((atBound?.weightedKeys ?? NaN) * 100).toFixed(2)}%, per-probe mean ${((atBound?.mean ?? NaN) * 100).toFixed(1)}%, p95 ${((atBound?.p95 ?? NaN) * 100).toFixed(1)}%); ` +
                `the last fine probe waiting for its keys by frame ${keyRun.lastAwaitingFrame ?? 'none waited'}; out-of-view fine probes' noise at 3 s ${((keyRun.noise?.lightWeighted ?? NaN) * 100).toFixed(2)}% (light-weighted), ${((keyRun.noise?.meanRelative ?? NaN) * 100).toFixed(2)}% (mean of ${keyRun.noise?.probes ?? 0})\n` +
                `    view error by frame: ${describeErrors(response.errors)}\n    light-weighted error by frame: ${keyRun.volume.map((entry) => `${entry.frame}: ${(entry.weighted * 100).toFixed(1)}% (${entry.awaiting} waiting)`).join(', ')}`,
        );
        if (response.settledFrame === null || response.settledFrame > VIEW_FRAMES) failures.push(`check 1: the view within 5% for good only from frame ${response.settledFrame ?? 'never'} (bar ${VIEW_FRAMES})`);
        if (!volumeWithin || volumeWithin.frame > VOLUME_FRAMES) {
            failures.push(`check 1: the field within 5% for good only from frame ${volumeWithin?.frame ?? 'never'} (bar ${VOLUME_FRAMES})`);
            if (!volumeWithin || volumeWithin.frame > VOLUME_FRAMES_DECISION) decisions.push(`check 1 misses by more than 50% (${volumeWithin?.frame ?? 'never'} frames against ${VOLUME_FRAMES}): the main session decides with Dennis`);
        }
    }

    if (SECTIONS.has('halved')) {
        const reference = await startSponza(device, 'sponza sun halved reference');
        reference.sun.intensity = fullSun / 2;
        reference.scene.threeGI.markChanged(reference.sun);
        const halfSettled = await convergeSponza(reference, 'the sun halved, from the start', log, null);
        reference.scene.dispose();
        const test = await startSponza(device, 'sponza sun halved, key probes');
        const before = await convergeSponza(test, 'the sun at full', log, UNITS);
        test.sun.intensity = fullSun / 2;
        test.scene.threeGI.markChanged(test.sun);
        const means: number[] = [];
        for (let frame = 1; frame <= MEASURED_FRAMES / 2; frame++) {
            test.scene.frame({ units: UNITS });
            means.push(await test.mean());
        }
        test.scene.dispose();
        const response = stepResponse(before, halfSettled, means, TOLERANCE);
        results.halved = { before, settled: halfSettled, settledFrame: response.settledFrame, overshootShareOfStep: response.overshootShareOfStep, errors: response.errors };
        log(`key probes, the sun halved: view ${before.toPrecision(4)} -> ${halfSettled.toPrecision(4)}, within ${TOLERANCE * 100}% for good from frame ${response.settledFrame ?? 'never'} (T2.4b's bar ${HALVED_VIEW_FRAMES}; reported)\n    error by frame: ${describeErrors(response.errors)}`);
    }

    if (SECTIONS.has('hold')) {
        assert(turnedReference, 'no reference');
        const skyReference = await startSponza(device, 'sponza sky dimmed reference');
        dimSky(skyReference, SKY_DIMMING);
        await convergeSponza(skyReference, `the sky at ${SKY_DIMMING}, from the start`, log, null);
        const dimmedReference = await probeMeans(skyReference);
        skyReference.scene.dispose();
        const holdResults: Record<string, unknown> = {};
        for (const change of ['sun turned', 'sky dimmed'] as const) {
            const test = await startSponza(device, `sponza ${change}, fine probes held`);
            await convergeSponza(test, `before the ${change}`, log, UNITS);
            const before = await probeMeans(test);
            const classes = probeClasses(test, test.scene.ddgi.field!);
            test.scene.ddgi.settings.holdFineProbes = true;
            if (change === 'sun turned') aimSun(test, turnedSun(startDirection, SUN_TURN_DEGREES));
            else dimSky(test, SKY_DIMMING);
            const reference = change === 'sun turned' ? turnedReference : dimmedReference;
            const curve: Array<{ frame: number; removed: number; along: number }> = [];
            let now = before;
            // The fine probes the hold keeps from their own updates: waiting for their keys after the mark.
            let held: Set<number> | null = null;
            for (let frame = 1; frame <= HOLD_FRAMES; frame++) {
                test.scene.frame({ units: UNITS });
                if (frame === 1) {
                    now = await probeMeans(test);
                    held = new Set(now.history.flatMap((record, probe) => ((record.flags & PROBE_HISTORY_FLAG.awaitingChange) !== 0 && !classes.key(probe) ? [probe] : [])));
                }
                if (frame % 30 === 0) {
                    await device.queue.onSubmittedWorkDone();
                    now = await probeMeans(test);
                    const taken = changeTakenIn(before, now, reference, (probe) => held!.has(probe));
                    curve.push({ frame, removed: taken.removed, along: taken.along });
                }
            }
            const final = changeTakenIn(before, now, reference, (probe) => held!.has(probe));
            const stillHeld = now.history.filter((record, probe) => held!.has(probe) && (record.flags & PROBE_HISTORY_FLAG.awaitingChange) !== 0).length;
            const counters = test.scene.ddgi.counters;
            test.scene.dispose();
            holdResults[change] = { ...final, held: held?.size ?? 0, stillHeld, curve, counters };
            log(
                `held fine probes, ${change}: propagation alone removed ${(final.removed * 100).toFixed(1)}% of the change at ${final.probes} fine probes (moved ${(final.along * 100).toFixed(1)}% along it; the change ${(final.change * 100).toFixed(1)}% of their light; ${stillHeld} of ${held?.size ?? 0} still held at frame ${HOLD_FRAMES}); ` +
                    `by frame: ${curve.map((entry) => `${entry.frame}: ${(entry.removed * 100).toFixed(1)}%`).join(', ')}; probes propagated (total) ${counters['probes propagated (total)'] ?? 'n/a'}`,
            );
            if (!(final.removed >= PROPAGATION_REMOVES)) {
                failures.push(`check 2 (${change}): propagation alone removed ${(final.removed * 100).toFixed(1)}% of the change (rule: at least ${PROPAGATION_REMOVES * 100}%)`);
                decisions.push(`check 2's rule (${change}): propagation removes less than ${PROPAGATION_REMOVES * 100}%, so it is switched off and "far probes owe 2" stays`);
            }
        }
        results.hold = holdResults;
    }

    if (SECTIONS.has('fallback')) {
        assert(turnedReference, 'no reference');
        const test = await startSponza(device, 'sponza sun turned, far probes owe 2');
        test.scene.ddgi.settings.keyProbes = false;
        await convergeSponza(test, 'the sun as it was (far probes owe 2)', log, UNITS);
        const before = await test.mean();
        const imageFrame = keyViewSettled ?? VIEW_FRAMES;
        const run = await sunTurnRun(test, turnedReference, startDirection, imageFrame);
        test.scene.dispose();
        const response = stepResponse(before, turnedSettled, run.viewMeans, TOLERANCE);
        const volumeWithin = run.volume.find((entry, index) => run.volume.slice(index).every((later) => later.weighted <= TOLERANCE));
        const atBound = run.volume.find((entry) => entry.frame === VOLUME_FRAMES);
        results.fallback = { viewSettledFrame: response.settledFrame, volumeWithinFrame: volumeWithin?.frame ?? null, atVolumeBound: atBound, noise: run.noise, volume: run.volume, viewErrors: response.errors };
        log(
            `far probes owe 2, the same turn: view within ${TOLERANCE * 100}% for good from frame ${response.settledFrame ?? 'never'}; field within 5% from frame ${volumeWithin?.frame ?? 'never'} (at ${VOLUME_FRAMES}: ${((atBound?.weighted ?? NaN) * 100).toFixed(2)}%); ` +
                `out-of-view fine probes' noise at 3 s ${((run.noise?.lightWeighted ?? NaN) * 100).toFixed(2)}% (light-weighted), ${((run.noise?.meanRelative ?? NaN) * 100).toFixed(2)}% (mean of ${run.noise?.probes ?? 0})\n    light-weighted error by frame: ${run.volume.map((entry) => `${entry.frame}: ${(entry.weighted * 100).toFixed(1)}%`).join(', ')}`,
        );
        if (keyRun?.noise && run.noise) {
            const worse = keyRun.noise.lightWeighted > run.noise.lightWeighted;
            log(`check 4: out-of-view fine probes' noise at 3 s, key probes ${(keyRun.noise.lightWeighted * 100).toFixed(2)}% against far probes owe 2 ${(run.noise.lightWeighted * 100).toFixed(2)}%: ${worse ? 'worse' : 'no worse'}`);
            if (worse) failures.push(`check 4: key probes leave the out-of-view fine probes noisier (${(keyRun.noise.lightWeighted * 100).toFixed(2)}% against ${(run.noise.lightWeighted * 100).toFixed(2)}%)`);
        }
        // The side-by-side: both at the frame the view settled with propagation.
        const withPropagation = keyRun?.imageAt(imageFrame);
        const without = run.imageAt(imageFrame);
        if (withPropagation) {
            images.push({ name: `sun-turn-frame${imageFrame}-colonnade-key-probes`, base64: withPropagation.view });
            images.push({ name: `sun-turn-frame${imageFrame}-upper-gallery-key-probes`, base64: withPropagation.elsewhere });
        }
        if (without) {
            images.push({ name: `sun-turn-frame${imageFrame}-colonnade-far-probes-owe-2`, base64: without.view });
            images.push({ name: `sun-turn-frame${imageFrame}-upper-gallery-far-probes-owe-2`, base64: without.elsewhere });
        }
    }

    results.decisions = decisions;
    if (decisions.length > 0) log(`decisions for the main session: ${decisions.join('; ')}`);
    assert(failures.length === 0, failures.join('; '));
});
