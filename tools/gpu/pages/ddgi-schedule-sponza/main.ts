/**
 * T2.4b on Sponza through the three adapter, as the viewer runs it, at the
 * RTX 3060's calibrated batch at 2 ms on this code (316 probes a frame,
 * PROGRESS 2026-11-06, T2.4b), GI only. Response times are counted in frames
 * (0.5 s is 30, 3 s is 180; DDGI's clock advances 1/60 s a frame). The view
 * is bookmark 1 (`colonnade-from-the-floor`), whose camera the scheduler
 * orders probes by.
 *
 * - **lamp** (TASKS, a lamp inside sunlit Sponza): a point light with no
 *   range (`range: 0`, so its influence radius is set against the sun,
 *   DECISIONS 2026-11-06) is switched on in the colonnade. The probes it
 *   marks (r above the tolerance, in the step's mark-only frame) against
 *   the probes whose light really changed by more than 5% (two converged
 *   fields, each probe's stored mean): recall must be at least 95%. Two
 *   lamps, LAMP_INTENSITIES.
 * - **sun** (checks 2 and 5): the sun halved (check 2: the view must move by
 *   more than 1% in the step's frame and be within 5% of settled within 30
 *   frames, for good), and turned 30° lower (check 5: the mean over all
 *   active probes of each probe's error against a converged field within 5%
 *   within 180 frames, and the view within 5% within 30; far probes owe 2).
 *   Reported: the p95 of the probe errors, and the error of near (camera
 *   weight ≥ 0.5) and far probes at 3 s against the same before the step.
 * - **census** (check 3): the sun turns 1.5° a second (an hour of sun in
 *   10 s) for 60 s; no probe may wait more than 2 passes of the round robin
 *   (ScheduleGapCensus). Also the `ddgi/schedule` pass's GPU time in these
 *   frames (information; the gate is ddgi-schedule-reduction).
 *
 * Converged fields: FAST_FRAMES at a fast fixed hysteresis and 4,096 probes
 * a frame, then the fixed 0.97 in blocks until the view settles; a field the
 * step runs on then has SETTLE_FRAMES of the default rule at the batch, so
 * its histories are at their steady count.
 *
 *   VITE_DDGI_SECTIONS=lamp,sun,census   (default all)
 *
 * Manual (needs the Sponza download): `pnpm test:gpu --filter ddgi-schedule-sponza --gpu nvidia`.
 */

import { PointLight, Vector3, type DirectionalLight } from 'three/webgpu';
import { assert } from '../../client.ts';
import { cameraWeight, frustumPlanes } from '../../../../packages/ddgi-core/src/techniques/ddgi/camera.ts';
import { priorityCount } from '../../../../packages/ddgi-core/src/techniques/ddgi/schedule.ts';
import { DEFAULT_PRIORITY_SHARE, NEAR_CAMERA_WEIGHT } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { imageMean, readProbeData, TestView } from '../ddgi-shared/harness.ts';
import { applyHysteresisMode, FIXED_097, parseHysteresisMode } from '../ddgi-shared/hysteresis-modes.ts';
import { describeErrors, probeCoordinates, readHistory, ScheduleGapCensus, stepResponse } from '../ddgi-shared/response.ts';
import { startViewerScene, yieldToCallbacks, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SECTIONS = new Set(listFromEnv(env.VITE_DDGI_SECTIONS, ['lamp', 'sun', 'census']));
const UNITS = 316;
const WIDTH = 320;
const HEIGHT = 180;
const TOLERANCE = 0.05;
const VIEW_FRAMES = 30;
const VOLUME_FRAMES = 180;
const MEASURED_FRAMES = 240;
const NEXT_FRAME_CHANGE = 0.01;
const MIN_STEP = 0.25;
const RECALL_TARGET = 0.95;
const TRUE_CHANGE = 0.05;
/** The lamp, in the colonnade in front of bookmark 1; two strengths (platform units: irradiance at 1 m). */
const LAMP_POSITION: [number, number, number] = [3, 1.5, 0.5];
const LAMP_INTENSITIES = [0.3, 3];
const SUN_TURN_DEGREES = 30;
const CENSUS_FRAMES = 3600;
const SUN_DEGREES_PER_SECOND = 1.5;
const CENSUS_PASSES = 2;
/** Converging (the file comment). */
const CONVERGE_UNITS = 4096;
const FAST_FRAMES = 160;
const FAST_HYSTERESIS = 0.5;
const BLOCK_FRAMES = 40;
const MAX_BLOCKS = 16;
const CONVERGED_TOLERANCE = 0.002;
const SETTLE_FRAMES = 1200;

type Vec3 = [number, number, number];

interface SponzaUnderTest {
    readonly scene: ViewerScene;
    readonly view: TestView;
    readonly sun: DirectionalLight;
    readonly lamp: PointLight;
    readonly mean: () => Promise<number>;
}

async function start(device: GPUDevice, label: string): Promise<SponzaUnderTest> {
    const manifest = loadAllManifests().find((entry) => entry.id === 'sponza');
    assert(manifest, 'no sponza manifest');
    let lamp: PointLight | null = null;
    const scene = await startViewerScene(device, manifest, {
        gpuTiming: true,
        label,
        prepareScene: (loaded) => {
            // Off until the page switches it on; no range (three's distance 0).
            lamp = new PointLight(0xffffff, 0, 0, 2);
            lamp.position.set(...LAMP_POSITION);
            loaded.scene.add(lamp);
        },
    });
    assert(lamp, 'no lamp');
    const sun = scene.loaded.directLights.find((light) => (light as DirectionalLight).isDirectionalLight) as DirectionalLight | undefined;
    assert(sun, 'Sponza has no sun');
    const view = await TestView.create(device, scene.platform);
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const bookmark = manifest.cameras[0];
    const mean = async () => imageMean((await view.render(field, bookmark, WIDTH, HEIGHT)).indirect);
    return { scene, view, sun, lamp, mean };
}

async function frames(test: SponzaUnderTest, count: number, units: number): Promise<void> {
    for (let index = 0; index < count; index++) {
        test.scene.frame({ units });
        if (index % 4 === 3) {
            await test.scene.device.queue.onSubmittedWorkDone();
            await yieldToCallbacks();
        }
    }
    await test.scene.device.queue.onSubmittedWorkDone();
}

async function converge(test: SponzaUnderTest, label: string, log: (line: string) => void, settle: boolean): Promise<number> {
    const settings = test.scene.ddgi.settings;
    applyHysteresisMode(settings, { name: 'fast', settings: { ...FIXED_097.settings, hysteresis: FAST_HYSTERESIS } });
    await frames(test, FAST_FRAMES, CONVERGE_UNITS);
    applyHysteresisMode(settings, FIXED_097);
    let mean = await test.mean();
    let settled = false;
    for (let block = 1; block <= MAX_BLOCKS && !settled; block++) {
        await frames(test, BLOCK_FRAMES, CONVERGE_UNITS);
        const next = await test.mean();
        settled = block >= 2 && Math.abs(next / mean - 1) < CONVERGED_TOLERANCE;
        mean = next;
        if (settled) log(`${label}: converged after ${block} blocks, view mean ${next.toPrecision(5)}`);
    }
    assert(settled, `${label}: the view did not settle`);
    applyHysteresisMode(settings, parseHysteresisMode('default'));
    if (settle) await frames(test, SETTLE_FRAMES, UNITS);
    return test.mean();
}

/** The sun's direction (from the light toward its target). */
function sunDirection(sun: DirectionalLight): Vector3 {
    return new Vector3().subVectors(sun.target.getWorldPosition(new Vector3()), sun.getWorldPosition(new Vector3())).normalize();
}

/** Points the sun along `direction`, keeping its target; the adapter sees it at the next update. */
function aimSun(test: SponzaUnderTest, direction: Vector3): void {
    const target = test.sun.target.getWorldPosition(new Vector3());
    const distance = test.sun.getWorldPosition(new Vector3()).distanceTo(target);
    test.sun.position.copy(target).addScaledVector(direction, -distance);
    test.sun.updateMatrixWorld();
    test.scene.threeGI.markChanged(test.sun);
}

/** The sun turned `degrees` lower (about the horizontal axis across its direction). */
function turnedSun(direction: Vector3, degrees: number): Vector3 {
    const axis = new Vector3().crossVectors(direction, new Vector3(0, 1, 0)).normalize();
    return direction.clone().applyAxisAngle(axis, (-degrees * Math.PI) / 180).normalize();
}

interface ProbeMeans {
    readonly mean: Vec3[];
    readonly active: boolean[];
}

async function probeMeans(test: SponzaUnderTest): Promise<ProbeMeans> {
    const field = test.scene.ddgi.field!;
    const [history, data] = await Promise.all([readHistory(test.scene.device, field), readProbeData(test.scene.device, field)]);
    return { mean: history.map((record) => record.mean), active: data.map((probe) => probe.state === 0) };
}

function relativeChange(before: Vec3, after: Vec3): number {
    let largest = 0;
    for (let channel = 0; channel < 3; channel++) largest = Math.max(largest, Math.abs(after[channel] - before[channel]) / Math.max(before[channel], 1e-6));
    return largest;
}

/** Each active probe's error against a reference (relative, its largest channel; dark probes against a floor of 1% of the volume's mean). */
function probeErrors(now: ProbeMeans, reference: ProbeMeans): { errors: number[]; probes: number[] } {
    const luminance = (rgb: Vec3) => (rgb[0] + rgb[1] + rgb[2]) / 3;
    const activeMeans = reference.mean.filter((_, probe) => reference.active[probe]).map(luminance);
    const floor = 0.01 * (activeMeans.reduce((sum, value) => sum + value, 0) / Math.max(1, activeMeans.length));
    const errors: number[] = [];
    const probes: number[] = [];
    for (let probe = 0; probe < reference.mean.length; probe++) {
        if (!reference.active[probe] || !now.active[probe]) continue;
        let largest = 0;
        for (let channel = 0; channel < 3; channel++) largest = Math.max(largest, Math.abs(now.mean[probe][channel] - reference.mean[probe][channel]) / Math.max(reference.mean[probe][channel], floor));
        errors.push(largest);
        probes.push(probe);
    }
    return { errors, probes };
}

/**
 * The field's error weighted by its light: the sum of |now - reference| over
 * the sum of the reference, every active probe and channel. Dark probes,
 * whose relative error is large and says little about the picture, count by
 * their light. Also the error of the volume's mean light.
 */
function lightWeightedError(now: ProbeMeans, reference: ProbeMeans, include: (probe: number) => boolean = () => true): { weighted: number; ofMean: number } {
    let difference = 0;
    let sum = 0;
    let nowSum = 0;
    for (let probe = 0; probe < reference.mean.length; probe++) {
        if (!reference.active[probe] || !now.active[probe] || !include(probe)) continue;
        for (let channel = 0; channel < 3; channel++) {
            difference += Math.abs(now.mean[probe][channel] - reference.mean[probe][channel]);
            sum += reference.mean[probe][channel];
            nowSum += now.mean[probe][channel];
        }
    }
    return { weighted: sum > 0 ? difference / sum : 0, ofMean: sum > 0 ? Math.abs(nowSum / sum - 1) : 0 };
}

const meanOf = (values: readonly number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
function quantile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))] ?? Number.NaN;
}

await runThreePage(
    async ({ device, data, log: logLine }) => {
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results = data;
        results.units = UNITS;
        results.priorityShare = priorityCount(UNITS, DEFAULT_PRIORITY_SHARE);
        const failures: string[] = [];

        if (SECTIONS.has('lamp')) {
            const lampResults: unknown[] = [];
            for (const intensity of LAMP_INTENSITIES) {
                const on = await start(device, `sponza lamp ${intensity} on`);
                on.lamp.intensity = intensity;
                on.scene.threeGI.markChanged(on.lamp);
                await converge(on, `lamp ${intensity} on from the start`, log, false);
                const after = await probeMeans(on);
                on.scene.dispose();
                const off = await start(device, `sponza lamp ${intensity} step`);
                await converge(off, `lamp ${intensity} off`, log, false);
                const before = await probeMeans(off);
                off.lamp.intensity = intensity;
                off.scene.threeGI.markChanged(off.lamp);
                // The mark-only frame: no probe updates, so the marks can be read before a blend takes them in.
                off.scene.frame({ units: 0 });
                await device.queue.onSubmittedWorkDone();
                const history = await readHistory(device, off.scene.ddgi.field!);
                const tolerance = off.scene.ddgi.settings.changeTolerance;
                off.scene.dispose();
                let changed = 0;
                let caught = 0;
                let cut = 0;
                let cutAndChanged = 0;
                const missedChanges: number[] = [];
                for (let probe = 0; probe < history.length; probe++) {
                    if (!before.active[probe] || !after.active[probe]) continue;
                    const reallyChanged = relativeChange(before.mean[probe], after.mean[probe]) > TRUE_CHANGE;
                    const isCut = history[probe].pendingChange > tolerance;
                    if (reallyChanged) changed++;
                    if (isCut) cut++;
                    if (reallyChanged && isCut) {
                        caught++;
                        cutAndChanged++;
                    }
                    if (reallyChanged && !isCut) missedChanges.push(relativeChange(before.mean[probe], after.mean[probe]));
                }
                const recall = changed > 0 ? caught / changed : 1;
                const precision = cut > 0 ? cutAndChanged / cut : 1;
                missedChanges.sort((a, b) => b - a);
                lampResults.push({ intensity, changed, caught, cut, recall, precision, largestMissed: missedChanges.slice(0, 5) });
                log(
                    `lamp ${intensity} switched on in sunlit Sponza: recall ${(recall * 100).toFixed(1)}% (${caught} of ${changed} probes that changed by more than ${TRUE_CHANGE * 100}%), ` +
                        `precision ${(precision * 100).toFixed(1)}%; the largest changes missed: ${missedChanges.slice(0, 5).map((value) => `${(value * 100).toFixed(0)}%`).join(', ') || 'none'}`,
                );
                if (recall < RECALL_TARGET) failures.push(`lamp ${intensity}: recall ${(recall * 100).toFixed(1)}% is under ${RECALL_TARGET * 100}%`);
            }
            results.lamp = lampResults;
        }

        if (SECTIONS.has('sun')) {
            // The references: converged with the sun halved, and with the sun turned.
            const probe = await start(device, 'sponza sun reference');
            const startDirection = sunDirection(probe.sun);
            const fullSun = probe.sun.intensity;
            probe.sun.intensity = fullSun / 2;
            probe.scene.threeGI.markChanged(probe.sun);
            const halfSettled = await converge(probe, 'the sun halved, from the start', log, false);
            probe.scene.dispose();
            const turned = await start(device, 'sponza sun turned reference');
            aimSun(turned, turnedSun(startDirection, SUN_TURN_DEGREES));
            const turnedSettled = await converge(turned, `the sun ${SUN_TURN_DEGREES} degrees lower, from the start`, log, false);
            const turnedReference = await probeMeans(turned);
            turned.scene.dispose();

            // Check 2: the sun halved.
            {
                const test = await start(device, 'sponza sun halved');
                const before = await converge(test, 'the sun at full', log, true);
                test.sun.intensity = fullSun / 2;
                test.scene.threeGI.markChanged(test.sun);
                const means: number[] = [];
                for (let frame = 1; frame <= MEASURED_FRAMES / 2; frame++) {
                    test.scene.frame({ units: UNITS });
                    means.push(await test.mean());
                }
                test.scene.dispose();
                const response = stepResponse(before, halfSettled, means, TOLERANCE);
                const firstFrameMove = Math.abs(means[0] - before) / Math.max(before, halfSettled);
                const stepSize = Math.abs(halfSettled - before) / Math.max(before, halfSettled);
                results.sunHalved = { before, settled: halfSettled, firstFrameMove, stepSize, ...response, means };
                log(
                    `the sun halved: view ${before.toPrecision(4)} -> ${halfSettled.toPrecision(4)} (${(stepSize * 100).toFixed(0)}%); frame 1 moved ${(firstFrameMove * 100).toFixed(1)}%; ` +
                        `within ${TOLERANCE * 100}% for good from frame ${response.settledFrame ?? 'never'}; overshoot ${(response.overshootShareOfStep * 100).toFixed(1)}% of the step\n    error by frame: ${describeErrors(response.errors)}`,
                );
                if (stepSize < MIN_STEP) failures.push(`the sun halved changed the view by only ${(stepSize * 100).toFixed(1)}%`);
                if (!(firstFrameMove > NEXT_FRAME_CHANGE)) failures.push(`the sun halved: frame 1 moved the view ${(firstFrameMove * 100).toFixed(2)}%`);
                if (response.settledFrame === null || response.settledFrame > VIEW_FRAMES) failures.push(`the sun halved: the view within 5% for good only from frame ${response.settledFrame ?? 'never'} (bar ${VIEW_FRAMES})`);
            }

            // Check 5: the sun 30 degrees lower.
            {
                const test = await start(device, 'sponza sun turned');
                const field = test.scene.ddgi.field!;
                const converged = await converge(test, 'the sun as it was', log, false);
                const convergedMeans = await probeMeans(test);
                await frames(test, SETTLE_FRAMES, UNITS);
                const before = await test.mean();
                // The noise floor: the same field after the settle frames at the batch, against itself converged.
                const settledMeans = await probeMeans(test);
                const floor = probeErrors(settledMeans, convergedMeans);
                const weightedFloor = lightWeightedError(settledMeans, convergedMeans);
                aimSun(test, turnedSun(startDirection, SUN_TURN_DEGREES));
                // Near and far probes, by their camera weight from bookmark 1.
                const camera = test.scene.camera;
                camera.updateMatrixWorld();
                const view = camera.matrixWorldInverse.elements;
                const planes = frustumPlanes({ view: [...view], projection: [...camera.projectionMatrix.elements] });
                const grid = field.grid;
                const near = (probeIndex: number) => {
                    const point = probeCoordinates(probeIndex, grid.counts).map((value, axis) => grid.origin[axis] + value * grid.spacing[axis]) as Vec3;
                    return cameraWeight(point, { position: [camera.position.x, camera.position.y, camera.position.z] }, planes, grid) >= NEAR_CAMERA_WEIGHT;
                };
                const viewMeans: number[] = [];
                const volume: Array<{ frame: number; mean: number; p95: number; near: number; far: number; weighted: number; ofMean: number; weightedFar: number }> = [];
                for (let frame = 1; frame <= MEASURED_FRAMES; frame++) {
                    test.scene.frame({ units: UNITS });
                    viewMeans.push(await test.mean());
                    if (frame % 6 === 0 || frame === 1) {
                        const now = await probeMeans(test);
                        const { errors, probes } = probeErrors(now, turnedReference);
                        const nearErrors = errors.filter((_, index) => near(probes[index]));
                        const farErrors = errors.filter((_, index) => !near(probes[index]));
                        const light = lightWeightedError(now, turnedReference);
                        const farLight = lightWeightedError(now, turnedReference, (probeIndex) => !near(probeIndex));
                        volume.push({ frame, mean: meanOf(errors), p95: quantile(errors, 0.95), near: meanOf(nearErrors), far: meanOf(farErrors), weighted: light.weighted, ofMean: light.ofMean, weightedFar: farLight.weighted });
                    }
                }
                test.scene.dispose();
                const response = stepResponse(before, turnedSettled, viewMeans, TOLERANCE);
                // Check 5's measure: the field's error weighted by its light (the per-probe relative error is reported, with its noise floor).
                const volumeWithin = volume.find((entry, index) => volume.slice(index).every((later) => later.weighted <= TOLERANCE));
                const atThreeSeconds = volume.find((entry) => entry.frame === VOLUME_FRAMES);
                const floorNear = meanOf(floor.errors.filter((_, index) => near(floor.probes[index])));
                const floorFar = meanOf(floor.errors.filter((_, index) => !near(floor.probes[index])));
                results.sunTurned = {
                    degrees: SUN_TURN_DEGREES,
                    convergedView: converged,
                    before,
                    settled: turnedSettled,
                    viewSettledFrame: response.settledFrame,
                    viewOvershootShareOfStep: response.overshootShareOfStep,
                    volumeWithinFrame: volumeWithin?.frame ?? null,
                    atThreeSeconds,
                    noiseFloor: { mean: meanOf(floor.errors), p95: quantile(floor.errors, 0.95), near: floorNear, far: floorFar, weighted: weightedFloor.weighted, ofMean: weightedFloor.ofMean },
                    volume,
                    viewErrors: response.errors,
                };
                log(
                    `the sun ${SUN_TURN_DEGREES} degrees lower: the view ${before.toPrecision(4)} -> ${turnedSettled.toPrecision(4)}, within ${TOLERANCE * 100}% for good from frame ${response.settledFrame ?? 'never'}; ` +
                        `the field's light-weighted error within ${TOLERANCE * 100}% for good from frame ${volumeWithin?.frame ?? 'never'} (at ${VOLUME_FRAMES} frames ${((atThreeSeconds?.weighted ?? NaN) * 100).toFixed(1)}%, far probes ${((atThreeSeconds?.weightedFar ?? NaN) * 100).toFixed(1)}%, the volume's mean ${((atThreeSeconds?.ofMean ?? NaN) * 100).toFixed(1)}%; floor ${(weightedFloor.weighted * 100).toFixed(1)}%); ` +
                        `per-probe relative error at ${VOLUME_FRAMES} frames mean ${((atThreeSeconds?.mean ?? NaN) * 100).toFixed(1)}%, p95 ${((atThreeSeconds?.p95 ?? NaN) * 100).toFixed(1)}%, ` +
                        `near ${((atThreeSeconds?.near ?? NaN) * 100).toFixed(1)}%, far ${((atThreeSeconds?.far ?? NaN) * 100).toFixed(1)}% (before the step, against itself converged: near ${(floorNear * 100).toFixed(1)}%, far ${(floorFar * 100).toFixed(1)}%)\n` +
                        `    view error by frame: ${describeErrors(response.errors)}\n    light-weighted error by frame: ${volume.map((entry) => `${entry.frame}: ${(entry.weighted * 100).toFixed(1)}%`).join(', ')}`,
                );
                if (response.settledFrame === null || response.settledFrame > VIEW_FRAMES) failures.push(`the sun turned: the view within 5% for good only from frame ${response.settledFrame ?? 'never'} (bar ${VIEW_FRAMES})`);
                if (!volumeWithin || volumeWithin.frame > VOLUME_FRAMES) failures.push(`the sun turned: the field's light-weighted error within 5% for good only from frame ${volumeWithin?.frame ?? 'never'} (bar ${VOLUME_FRAMES})`);
            }
        }

        if (SECTIONS.has('census')) {
            const test = await start(device, 'sponza census');
            const field = test.scene.ddgi.field!;
            await converge(test, 'the census field', log, false);
            const census = await ScheduleGapCensus.create(device, test.scene.platform, field);
            const roundRobinShare = UNITS - priorityCount(UNITS, DEFAULT_PRIORITY_SHARE);
            const passFrames = Math.ceil(field.grid.probeCount / roundRobinShare);
            // Every probe listed at least once before the census starts.
            await frames(test, passFrames + 4, UNITS);
            let direction = sunDirection(test.sun);
            const scheduleMs: number[] = [];
            for (let frame = 1; frame <= CENSUS_FRAMES; frame++) {
                direction = turnedSun(direction, SUN_DEGREES_PER_SECOND / 60);
                aimSun(test, direction);
                test.scene.frame({ units: UNITS });
                census.measure(test.scene.platform.stats.frame);
                const pass = test.scene.platform.stats.passes.find((entry) => entry.name === 'ddgi/schedule');
                if (pass?.gpuMs !== undefined && pass.gpuMs >= 0 && frame % 10 === 0) scheduleMs.push(pass.gpuMs);
                if (frame % 4 === 0) {
                    await device.queue.onSubmittedWorkDone();
                    await yieldToCallbacks();
                }
            }
            const gap = await census.read();
            const counters = test.scene.ddgi.counters;
            census.dispose();
            test.scene.dispose();
            results.census = { frames: CENSUS_FRAMES, degreesPerSecond: SUN_DEGREES_PER_SECOND, longestGapFrames: gap.frames, atProbe: gap.probe, roundRobinPassFrames: passFrames, scheduleMsMedian: quantile(scheduleMs, 0.5), counters };
            log(
                `census, the sun turning ${SUN_DEGREES_PER_SECOND} degrees a second for ${CENSUS_FRAMES} frames at ${UNITS} a frame: the longest any probe waited was ${gap.frames} frames (probe ${gap.probe}); ` +
                    `a pass of the round robin is ${passFrames} frames; ddgi/schedule median ${quantile(scheduleMs, 0.5).toFixed(4)} ms (information)`,
            );
            if (gap.frames > CENSUS_PASSES * passFrames) failures.push(`census: a probe waited ${gap.frames} frames, more than ${CENSUS_PASSES} passes of the round robin (${CENSUS_PASSES * passFrames})`);
        }

        assert(failures.length === 0, failures.join('; '));
    },
);
