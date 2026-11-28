/**
 * T2.6a check 4 (docs/design/m2-lights.md §6.4, the T2.6a done-when): by
 * re-tracing alone (no instant sky yet), the sun 30° lower (in elevation,
 * same azimuth) on the furnished room, the sun light and the procedural sky
 * moved together as a time-of-day slider moves them, through the three adapter as the viewer
 * runs it, GI only. Counted in frames at a fixed batch (DDGI's clock
 * advances 1/60 s a frame: 30 frames is 0.5 s).
 *
 * For each bookmark in VITE_DDGI_BOOKMARKS (1-based; default both), a scene
 * of its own whose camera, the one the scheduler orders probes by, is that
 * bookmark's:
 * - **the step's size** on that view: its GI-only mean before and settled
 *   (settled / before − 1), and per pixel (the mean over pixels of
 *   |settled − before| over the settled mean). T2.2's review found Sponza's
 *   30° turn moved its bookmark view by only 4%, which makes "within 5%"
 *   hard to fail, so the check is judged on a view the step moves by at
 *   least 25% (STEP_AT_LEAST; DECISIONS 2026-11-06 records which);
 * - **the view's response:** the first frame from which the view's mean
 *   stays within 5% of settled (bar 30 frames), and the same for the
 *   per-pixel error (reported);
 * - **the whole volume:** the field's light-weighted error over every active
 *   probe against a field converged from the start in the new light (T2.4b's
 *   measure, DECISIONS 2026-11-06), every VOLUME_EVERY frames, and the first
 *   frame from which it stays within 5% (recorded, no bar);
 * - **the sky re-baked** once, in the step's frame (the sky bake counter).
 *
 * Settled: a scene converged in the new light from the start
 * (ddgi-shared/sponza-response.ts's converging, which is not Sponza-specific).
 *
 *   VITE_DDGI_UNITS=<probes a frame>   (default DEFAULT_UNITS: the furnished room's calibrated batch at 2 ms)
 *   VITE_DDGI_BOOKMARKS=1,2
 *   VITE_DDGI_LAMP=on                  (the pendant lamp as the manifest has it; default off, see LAMP_ON)
 *   VITE_DDGI_SUN_STEP_DEGREES=30
 *   VITE_DDGI_SECTIONS=explore         (only the step's size on 24 candidate views)
 *   VITE_DDGI_SECTIONS=rho             (the room's bounce share, every probe every frame at h = 0)
 *   VITE_DDGI_BOUNCE_SHARE=<rho>       (changeBounceShare for the step's run)
 *   VITE_DDGI_SETTINGS=<json>          (DDGI settings for the step's run, Object.assign-ed: for comparing rules)
 *
 * The step's run also follows two groups of 12 probes near and in the view
 * (the largest change, and the largest relative change of the rest) at
 * TRACK_FRAMES: their error against the converged field, updates since the
 * step, history count n, owed updates, and how many still hold a cut.
 * And the field by class (T2.4d, 'byClass'): keys, fine probes near the
 * view, in view farther off, and the rest, each with its light-weighted
 * error, updates since the step, history count, and how many wait for keys.
 *
 * Also reported per bookmark: the probes' marked r against their true
 * change (medians), which tells a mark that underestimates the change from
 * a budget that can't serve it.
 *
 * Manual (needs the furnished-room download):
 *   pnpm test:gpu --filter ddgi-sky-sun-step --gpu nvidia --json <file>
 */

import { PointLight, Vector3, type DirectionalLight } from 'three/webgpu';
import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { imageMean, TestView } from '../ddgi-shared/harness.ts';
import { describeErrors, probeCoordinates, stepResponse } from '../ddgi-shared/response.ts';
import { cameraWeight, frustumPlanes } from '../../../../packages/ddgi/src/camera.ts';
import { applyHysteresisMode, parseHysteresisMode } from '../ddgi-shared/hysteresis-modes.ts';
import { PROBE_HISTORY_FLAG } from '../../../../packages/ddgi/src/layout.ts';
import { convergeSponza, lightWeightedError, probeClasses, probeMeans, sunDirection, turnedSun, VIEW_HEIGHT, VIEW_WIDTH, type ProbeMeans, type SponzaUnderTest } from '../ddgi-shared/sponza-response.ts';
import { startViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SCENE_ID = 'furnished-room';
/** The furnished room's batch at Budget v2's 2 ms on the RTX 3060 (T2.6a's VALID calibration, PROGRESS). */
const DEFAULT_UNITS = 571;
const UNITS = Number(env.VITE_DDGI_UNITS ?? DEFAULT_UNITS);
const BOOKMARKS = listFromEnv(env.VITE_DDGI_BOOKMARKS, ['1', '2']).map(Number);
/** How much lower the sun goes, in elevation (same azimuth): the evening direction, the sunlight reaching deeper into the room. */
const SUN_STEP_DEGREES = Number(env.VITE_DDGI_SUN_STEP_DEGREES ?? 30);
const TOLERANCE = 0.05;
/** The bar: 0.5 s. */
const VIEW_FRAMES = 30;
/** A view the check is judged on must move by at least this much (TASKS, T2.6a). */
const STEP_AT_LEAST = 0.25;
const MEASURED_FRAMES = 180;
/**
 * The pendant lamp is off unless `VITE_DDGI_LAMP=on`, so the room is lit by
 * the sun and the sky alone (DECISIONS 2026-11-06). With it on, its bounce
 * light is most of every view's GI, and no view of the room moves by
 * STEP_AT_LEAST: the bookmarks by 5-9%, 24 candidate views (`explore`) by
 * 13.5% at most.
 */
const LAMP_ON = env.VITE_DDGI_LAMP === 'on';

/** `VITE_DDGI_BOUNCE_SHARE`: the technique's `changeBounceShare` (ρ) for the step's run; its default (0.8) otherwise. */
const BOUNCE_SHARE = env.VITE_DDGI_BOUNCE_SHARE === undefined || env.VITE_DDGI_BOUNCE_SHARE === '' ? null : Number(env.VITE_DDGI_BOUNCE_SHARE);
/**
 * `rho`: the room's bounce share measured as T2.4b measured two-room's: after
 * the step, every probe every frame at h = 0 (each update keeps nothing of
 * the old value), so the field's error shrinks only by what comes back
 * through the field: the light-weighted error against the converged field,
 * less its noise floor, shrinks by ρ a frame.
 */
const MEASURE_RHO = env.VITE_DDGI_SECTIONS === 'rho';
/** Probes followed after the step: in each group, this many. */
const TRACKED_PER_GROUP = 12;
/** Frames the tracked probes are summarised at. */
const TRACK_FRAMES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180];

/** `explore`: only the step's size on candidate views (EXPLORE_VIEWS), from two converged fields, no response runs. */
const EXPLORE = env.VITE_DDGI_SECTIONS === 'explore';

/** Views around the room for `explore`: from four places, toward the window wall, the far wall, the floor and the ceiling. */
const EXPLORE_VIEWS: CameraBookmark[] = (() => {
    const places: Array<[string, [number, number, number]]> = [
        ['west', [-1.5, 1.4, 0]],
        ['east', [1.5, 1.4, 0]],
        ['south', [0, 1.4, 2]],
        ['north', [0, 1.4, -2]],
    ];
    const targets: Array<[string, [number, number, number]]> = [
        ['window wall', [2.05, 1.4, -0.5]],
        ['far wall', [-2.05, 1.2, -0.5]],
        ['floor by the window', [1.5, 0.0, -0.5]],
        ['floor in the middle', [0, 0.0, 0]],
        ['ceiling by the window', [1.5, 3.2, -0.5]],
        ['ceiling in the middle', [0, 3.2, 0]],
    ];
    return places.flatMap(([placeName, position]) => targets.map(([targetName, target]) => ({ name: `${placeName} to ${targetName}`, position, target, fovYDeg: 60 })));
})();
const VOLUME_EVERY = 6;

/** The mean over pixels of |a − b|, over the mean of b (as ddgi-schedule-step measures a sweep). */
function perPixelError(image: Float32Array, reference: Float32Array): number {
    let difference = 0;
    let sum = 0;
    for (let index = 0; index < image.length; index++) {
        difference += Math.abs(image[index] - reference[index]);
        sum += reference[index];
    }
    return sum > 0 ? difference / sum : 0;
}

/**
 * What the mark gave the probes against how much their light really
 * changed: over the active probes not updated in the step's frame (still
 * holding their pendingChange), the median marked r, and the median true
 * change (|settled − before| over before, the largest channel; a darkening
 * against what is left, as the mark measures it).
 */
function markedAgainstTrue(before: ProbeMeans, afterMark: ProbeMeans, settled: ProbeMeans): { probes: number; medianMarked: number; medianTrue: number } {
    const marked: number[] = [];
    const truth: number[] = [];
    for (let probe = 0; probe < before.mean.length; probe++) {
        const pending = afterMark.history[probe].pendingChange;
        if (!before.active[probe] || !settled.active[probe] || !(pending > 0)) continue;
        marked.push(pending);
        let largest = 0;
        for (let channel = 0; channel < 3; channel++) {
            const old = before.mean[probe][channel];
            const next = settled.mean[probe][channel];
            const against = next < old ? next : old;
            if (against > 0) largest = Math.max(largest, Math.abs(next - old) / against);
        }
        truth.push(largest);
    }
    const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? Number.NaN;
    return { probes: marked.length, medianMarked: median(marked), medianTrue: median(truth) };
}

/** The probes followed after the step: two groups of the probes near and in the view (camera weight ≥ 0.5). */
interface TrackedProbes {
    /** The largest change in light (light-weighted): the sunlit floor and what it lights. */
    readonly largest: number[];
    /** The largest relative change, of the rest: probes the window's light reaches mostly by bounce. */
    readonly relative: number[];
}

function trackedProbes(room: FurnishedRoom, before: ProbeMeans, settled: ProbeMeans): TrackedProbes {
    const classes = probeClasses(room, room.scene.ddgi.field!);
    const luminance = (rgb: readonly number[]) => (rgb[0] + rgb[1] + rgb[2]) / 3;
    const candidates = before.mean.flatMap((mean, probe) => (before.active[probe] && settled.active[probe] && classes.near(probe) ? [probe] : []));
    const absolute = (probe: number) => Math.abs(luminance(settled.mean[probe]) - luminance(before.mean[probe]));
    const relative = (probe: number) => absolute(probe) / Math.max(luminance(before.mean[probe]), 1e-6);
    const largest = [...candidates].sort((a, b) => absolute(b) - absolute(a)).slice(0, TRACKED_PER_GROUP);
    const rest = candidates.filter((probe) => !largest.includes(probe));
    return { largest, relative: rest.sort((a, b) => relative(b) - relative(a)).slice(0, TRACKED_PER_GROUP) };
}

/** One group's summary at one frame. */
interface TrackRow {
    readonly group: string;
    readonly frame: number;
    /** Median of the probes' errors against the converged field (largest channel, relative). */
    readonly medianError: number;
    /** Updates since the step (the record's count k, less its count before). */
    readonly meanUpdates: number;
    /** Median history count n after the update (the next update's h is about n / (n + 1)). */
    readonly medianSamples: number;
    readonly meanOwed: number;
    /** Share still holding a cut (taking a change, updates owed). */
    readonly holdingShare: number;
}

function trackRows(frame: number, now: ProbeMeans, before: ProbeMeans, settled: ProbeMeans, tracked: TrackedProbes): TrackRow[] {
    const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? Number.NaN;
    const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
    return (['largest', 'relative'] as const).map((group) => {
        const probes = tracked[group];
        const error = (probe: number) =>
            Math.max(...[0, 1, 2].map((channel) => Math.abs(now.mean[probe][channel] - settled.mean[probe][channel]) / Math.max(settled.mean[probe][channel], 1e-6)));
        return {
            group: group === 'largest' ? 'largest change' : 'largest relative change',
            frame,
            medianError: median(probes.map(error)),
            meanUpdates: mean(probes.map((probe) => now.history[probe].updates - before.history[probe].updates)),
            medianSamples: median(probes.map((probe) => now.history[probe].samples)),
            meanOwed: mean(probes.map((probe) => now.history[probe].owed)),
            holdingShare: mean(probes.map((probe) => ((now.history[probe].flags & PROBE_HISTORY_FLAG.takingChange) !== 0 && now.history[probe].owed > 0 ? 1 : 0))),
        };
    });
}

/**
 * The field by class of probe, to see where a response's error sits: keys,
 * fine probes near and in view (camera weight ≥ 0.5), fine probes in view
 * farther off (0.1..0.5: what most of the view's pixels sample), and the rest.
 */
type ProbeClass = 'keys' | 'near fine' | 'view fine' | 'other fine';
const PROBE_CLASSES: readonly ProbeClass[] = ['keys', 'near fine', 'view fine', 'other fine'];
/** Fine probes with a camera weight from this up to "near" count as in view, farther off. */
const VIEW_CAMERA_WEIGHT = 0.1;

function classifyProbes(room: FurnishedRoom): (probe: number) => ProbeClass {
    const field = room.scene.ddgi.field!;
    const classes = probeClasses(room, field);
    const camera = room.scene.camera;
    camera.updateMatrixWorld();
    const planes = frustumPlanes({ view: [...camera.matrixWorldInverse.elements], projection: [...camera.projectionMatrix.elements] });
    const grid = field.grid;
    return (probe) => {
        if (classes.key(probe)) return 'keys';
        if (classes.near(probe)) return 'near fine';
        const point = probeCoordinates(probe, grid.counts).map((value, axis) => grid.origin[axis] + value * grid.spacing[axis]) as [number, number, number];
        return cameraWeight(point, { position: [camera.position.x, camera.position.y, camera.position.z] }, planes, grid) >= VIEW_CAMERA_WEIGHT ? 'view fine' : 'other fine';
    };
}

interface ClassRow {
    readonly frame: number;
    readonly probeClass: ProbeClass;
    readonly probes: number;
    /** Light-weighted error against the converged field. */
    readonly error: number;
    readonly meanUpdates: number;
    readonly medianSamples: number;
    readonly awaitingShare: number;
    readonly owedShare: number;
}

function classRows(frame: number, now: ProbeMeans, before: ProbeMeans, settled: ProbeMeans, classOf: (probe: number) => ProbeClass): ClassRow[] {
    const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] ?? Number.NaN;
    return PROBE_CLASSES.map((probeClass) => {
        const members = now.mean.flatMap((_, probe) => (now.active[probe] && settled.active[probe] && classOf(probe) === probeClass ? [probe] : []));
        const share = (test: (probe: number) => boolean) => members.filter(test).length / Math.max(1, members.length);
        return {
            frame,
            probeClass,
            probes: members.length,
            error: lightWeightedError(now, settled, (probe) => classOf(probe) === probeClass),
            meanUpdates: members.reduce((sum, probe) => sum + now.history[probe].updates - before.history[probe].updates, 0) / Math.max(1, members.length),
            medianSamples: median(members.map((probe) => now.history[probe].samples)),
            awaitingShare: share((probe) => (now.history[probe].flags & PROBE_HISTORY_FLAG.awaitingChange) !== 0),
            owedShare: share((probe) => now.history[probe].owed > 0),
        };
    });
}

/** The first frame from which every value stays at or under the tolerance; null if the last is over. */
function settledFrom(values: ReadonlyArray<{ frame: number; value: number }>): number | null {
    let settled: number | null = null;
    for (let index = values.length - 1; index >= 0 && values[index].value <= TOLERANCE; index--) settled = values[index].frame;
    return settled;
}

interface FurnishedRoom extends SponzaUnderTest {
    /** The GI-only image of a bookmark (1-based). */
    readonly imageOf: (bookmark: number) => Promise<Float32Array>;
}

/** The furnished room as the viewer starts it, its camera at `bookmark` (1-based). */
async function startRoom(device: GPUDevice, label: string, bookmark: number): Promise<FurnishedRoom> {
    const manifest = loadAllManifests().find((entry) => entry.id === SCENE_ID);
    assert(manifest, `no ${SCENE_ID} manifest`);
    const scene = await startViewerScene(device, manifest, {
        gpuTiming: true,
        label,
        prepareScene: (loaded) => {
            // Before the adapter's first sync, so the field never sees the lamp.
            if (!LAMP_ON) loaded.scene.traverse((object) => (object as PointLight).isPointLight && ((object as PointLight).intensity = 0));
        },
    });
    scene.lookFrom(bookmark);
    const sun = scene.loaded.directLights.find((light) => (light as DirectionalLight).isDirectionalLight) as DirectionalLight | undefined;
    assert(sun, `${SCENE_ID} has no sun`);
    const view = await TestView.create(device, scene.platform);
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const imageOf = async (index: number) => (await view.render(field, manifest.cameras[index - 1], VIEW_WIDTH, VIEW_HEIGHT)).indirect;
    const images = async () => view.render(field, manifest.cameras[bookmark - 1], VIEW_WIDTH, VIEW_HEIGHT);
    // The converging helpers take a lamp; this page switches none on.
    return { scene, view, sun, lamp: new PointLight(0xffffff, 0), images, mean: async () => imageMean((await images()).indirect), imageOf };
}

/**
 * The time-of-day step: the sun light turned `degrees` lower, and the
 * procedural sky's sun with it (the adapter sends the environment again).
 */
function stepSun(room: FurnishedRoom, toward: Vector3): void {
    const target = room.sun.target.getWorldPosition(new Vector3());
    const distance = room.sun.getWorldPosition(new Vector3()).distanceTo(target);
    room.sun.position.copy(target).addScaledVector(toward, -distance);
    room.sun.updateMatrixWorld();
    room.scene.threeGI.markChanged(room.sun);
    const environment = room.scene.threeGI.environment;
    assert(environment && environment.type === 'procedural', 'the furnished room has no procedural sky');
    // The light travels along `toward`; the sky's sun is where it comes from.
    room.scene.threeGI.environment = { ...environment, sunDirection: [-toward.x, -toward.y, -toward.z] };
}

await runThreePage(async ({ device, data, log: logLine }) => {
    const log = (line: string): void => {
        logLine(line);
        console.log(line);
    };
    const results = data;
    results.units = UNITS;
    results.pendantLamp = LAMP_ON ? 'on' : 'off';
    results.bounceShare = BOUNCE_SHARE ?? 'default';
    results.sunStepDegrees = SUN_STEP_DEGREES;
    const failures: string[] = [];

    // Settled: converged from the start in the new light.
    const reference = await startRoom(device, 'furnished room, the sun stepped, from the start', BOOKMARKS[0]);
    const startDirection = sunDirection(reference.sun);
    const stepped = turnedSun(startDirection, SUN_STEP_DEGREES);
    results.sunElevationDegrees = { before: (Math.asin(-startDirection.y) * 180) / Math.PI, after: (Math.asin(-stepped.y) * 180) / Math.PI };
    stepSun(reference, stepped);
    await convergeSponza(reference, `the sun ${SUN_STEP_DEGREES} degrees lower, from the start`, log, null);
    if (EXPLORE) {
        const view = reference.view;
        const field = reference.scene.ddgi.field!;
        const after = await Promise.all(EXPLORE_VIEWS.map(async (camera) => (await view.render(field, camera, VIEW_WIDTH, VIEW_HEIGHT)).indirect));
        reference.scene.dispose();
        const room = await startRoom(device, 'furnished room, the sun as it was', BOOKMARKS[0]);
        await convergeSponza(room, 'the sun as it was, from the start', log, null);
        const roomField = room.scene.ddgi.field!;
        const explored: Array<{ name: string; stepShare: number; stepPerPixel: number; position: number[]; target: number[] }> = [];
        for (const [index, camera] of EXPLORE_VIEWS.entries()) {
            const before = (await room.view.render(roomField, camera, VIEW_WIDTH, VIEW_HEIGHT)).indirect;
            explored.push({ name: camera.name, stepShare: imageMean(after[index]) / imageMean(before) - 1, stepPerPixel: perPixelError(before, after[index]), position: camera.position, target: camera.target });
        }
        room.scene.dispose();
        explored.sort((a, b) => Math.abs(b.stepShare) - Math.abs(a.stepShare));
        results.explored = explored;
        log(`the step's size on candidate views: ${explored.map((entry) => `${entry.name} ${(entry.stepShare * 100).toFixed(1)}% (per pixel ${(entry.stepPerPixel * 100).toFixed(1)}%)`).join('; ')}`);
        return;
    }
    const settledImages = new Map<number, Float32Array>();
    for (const bookmark of BOOKMARKS) settledImages.set(bookmark, await reference.imageOf(bookmark));
    const settledField: ProbeMeans = await probeMeans(reference);
    if (MEASURE_RHO) {
        reference.scene.dispose();
        const room = await startRoom(device, 'furnished room, rho', BOOKMARKS[0]);
        await convergeSponza(room, 'the sun as it was, from the start', log, null);
        applyHysteresisMode(room.scene.ddgi.settings, parseHysteresisMode('fixed:0'));
        stepSun(room, stepped);
        const errors: number[] = [];
        const all = room.scene.probeCount;
        for (let frame = 1; frame <= 60; frame++) {
            room.scene.frame({ units: all });
            errors.push(lightWeightedError(await probeMeans(room), settledField));
        }
        room.scene.dispose();
        // The floor: one update's noise, where the error stops shrinking (the last 20 frames).
        const floor = errors.slice(40).reduce((sum, value) => sum + value, 0) / 20;
        const ratios = errors.slice(1, 12).map((value, index) => (value - floor) / (errors[index] - floor));
        const sortedRatios = [...ratios].sort((a, b) => a - b);
        results.rho = { errors, floor, ratios, median: sortedRatios[Math.floor(sortedRatios.length / 2)] };
        log(`rho: light-weighted error by frame ${errors.slice(0, 15).map((value) => (value * 100).toFixed(1)).join(', ')} ... floor ${(floor * 100).toFixed(2)}%; ratios ${ratios.map((value) => value.toFixed(3)).join(', ')}`);
        return;
    }
    results.probes = reference.scene.probeCount;
    reference.scene.dispose();

    const views: Record<string, unknown> = {};
    for (const bookmark of BOOKMARKS) {
        const room = await startRoom(device, `furnished room, bookmark ${bookmark}`, bookmark);
        await convergeSponza(room, `bookmark ${bookmark}: the sun as it was`, log, UNITS);
        const beforeImage = await room.imageOf(bookmark);
        const settledImage = settledImages.get(bookmark)!;
        const before = imageMean(beforeImage);
        const settled = imageMean(settledImage);
        const stepPerPixel = perPixelError(beforeImage, settledImage);
        const bakesBefore = room.scene.platform.shade.sky.projections;
        const fieldBefore = await probeMeans(room);

        if (BOUNCE_SHARE !== null) room.scene.ddgi.settings.changeBounceShare = BOUNCE_SHARE;
        if (typeof env.VITE_DDGI_SETTINGS === 'string' && env.VITE_DDGI_SETTINGS !== '') Object.assign(room.scene.ddgi.settings, JSON.parse(env.VITE_DDGI_SETTINGS));
        const tracked = trackedProbes(room, fieldBefore, settledField);
        const tracking: TrackRow[] = [];
        const classOf = classifyProbes(room);
        const byClass: ClassRow[] = [];
        stepSun(room, stepped);
        const means: number[] = [];
        const perPixel: Array<{ frame: number; value: number }> = [];
        const volume: Array<{ frame: number; value: number }> = [];
        let bakesAtStep = 0;
        let marks: ReturnType<typeof markedAgainstTrue> | null = null;
        for (let frame = 1; frame <= MEASURED_FRAMES; frame++) {
            room.scene.frame({ units: UNITS });
            if (frame === 1) {
                bakesAtStep = room.scene.platform.shade.sky.projections - bakesBefore;
                marks = markedAgainstTrue(fieldBefore, await probeMeans(room), settledField);
            }
            const image = await room.imageOf(bookmark);
            means.push(imageMean(image));
            perPixel.push({ frame, value: perPixelError(image, settledImage) });
            const needsField = frame === 1 || frame % VOLUME_EVERY === 0 || TRACK_FRAMES.includes(frame);
            const now = needsField ? await probeMeans(room) : null;
            if (now && (frame === 1 || frame % VOLUME_EVERY === 0)) volume.push({ frame, value: lightWeightedError(now, settledField) });
            if (now && TRACK_FRAMES.includes(frame)) tracking.push(...trackRows(frame, now, fieldBefore, settledField, tracked));
            if (now && TRACK_FRAMES.includes(frame)) byClass.push(...classRows(frame, now, fieldBefore, settledField, classOf));
        }
        const bakesAfter = room.scene.platform.shade.sky.projections - bakesBefore;
        const counters = room.scene.ddgi.counters;
        room.scene.dispose();

        const response = stepResponse(before, settled, means, TOLERANCE);
        const perPixelSettled = settledFrom(perPixel);
        const volumeSettled = settledFrom(volume);
        const judged = Math.abs(response.stepShare) >= STEP_AT_LEAST;
        views[`bookmark ${bookmark}`] = {
            name: reference.scene.manifest.cameras[bookmark - 1].name,
            before,
            settled,
            stepShare: response.stepShare,
            stepPerPixel,
            judged,
            viewSettledFrame: response.settledFrame,
            viewEnteredFrame: response.enteredFrame,
            overshootShareOfStep: response.overshootShareOfStep,
            perPixelSettledFrame: perPixelSettled,
            perPixelAtViewBound: perPixel[VIEW_FRAMES - 1]?.value,
            perPixelAt3s: perPixel[MEASURED_FRAMES - 1]?.value,
            viewAt3s: response.errors[MEASURED_FRAMES - 1],
            probesPropagatedTotal: counters['probes propagated (total)'],
            volumeSettledFrame: volumeSettled,
            volume,
            viewErrors: response.errors,
            perPixel: perPixel.map((entry) => entry.value),
            skyBakes: { inTheStepFrame: bakesAtStep, inAll: bakesAfter },
            marks,
            tracking,
            byClass,
            probesMarkedTotal: counters['probes marked (total)'],
        };
        log(
            `bookmark ${bookmark} (${reference.scene.manifest.cameras[bookmark - 1].name}): the step moves the view by ${(response.stepShare * 100).toFixed(1)}% (per pixel ${(stepPerPixel * 100).toFixed(1)}%)${judged ? '' : `, under ${STEP_AT_LEAST * 100}%: reported, not judged`}; ` +
                `view within ${TOLERANCE * 100}% for good from frame ${response.settledFrame ?? 'never'} (bar ${VIEW_FRAMES}), per pixel from frame ${perPixelSettled ?? 'never'} (at frame ${VIEW_FRAMES}: ${((perPixel[VIEW_FRAMES - 1]?.value ?? NaN) * 100).toFixed(1)}%); ` +
                `whole volume within 5% from frame ${volumeSettled ?? 'never'}; the sky baked ${bakesAtStep} time(s) in the step's frame, ${bakesAfter} in all; ` +
                `marked r against the true change (medians over ${marks?.probes ?? 0} probes not updated in the step's frame): ${marks?.medianMarked.toFixed(3)} against ${marks?.medianTrue.toFixed(3)}\n` +
                `    view error by frame: ${describeErrors(response.errors)}\n    per pixel: ${perPixel.filter((entry) => entry.frame <= 12 || entry.frame % 6 === 0).map((entry) => `${entry.frame}: ${(entry.value * 100).toFixed(1)}%`).join(', ')}\n` +
                `    volume (light-weighted): ${volume.map((entry) => `${entry.frame}: ${(entry.value * 100).toFixed(1)}%`).join(', ')}\n` +
                tracking.map((row) => `    ${row.group} frame ${row.frame}: error ${(row.medianError * 100).toFixed(1)}%, updates ${row.meanUpdates.toFixed(1)}, n ${row.medianSamples.toFixed(2)}, owed ${row.meanOwed.toFixed(1)}, holding ${(row.holdingShare * 100).toFixed(0)}%`).join('\n'),
        );
        if (bakesAtStep !== 1 || bakesAfter !== 1) failures.push(`bookmark ${bookmark}: the sky baked ${bakesAtStep} time(s) in the step's frame and ${bakesAfter} in all (expected 1 and 1)`);
        if (judged && (response.settledFrame === null || response.settledFrame > VIEW_FRAMES)) {
            failures.push(`bookmark ${bookmark}: the view within 5% for good only from frame ${response.settledFrame ?? 'never'} (bar ${VIEW_FRAMES})`);
        }
    }
    results.views = views;
    const judgedViews = Object.values(views).filter((view) => (view as { judged: boolean }).judged);
    if (judgedViews.length === 0) failures.push(`no bookmark is moved by ${STEP_AT_LEAST * 100}% or more by the step, so check 4 can't be judged on these views`);
    assert(failures.length === 0, failures.join('; '));
});
