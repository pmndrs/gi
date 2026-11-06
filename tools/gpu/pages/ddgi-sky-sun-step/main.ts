/**
 * T2.6a check 4 (docs/design/m2-lights.md §6.4, the T2.6a done-when): by
 * re-tracing alone (no instant sky yet), a 30° step of the sun on the
 * furnished room, the sun light and the procedural sky turned together as a
 * time-of-day slider moves them, through the three adapter as the viewer
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
 *
 * Manual (needs the furnished-room download):
 *   pnpm test:gpu --filter ddgi-sky-sun-step --gpu nvidia --json <file>
 */

import { PointLight, Vector3, type DirectionalLight } from 'three/webgpu';
import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { imageMean, TestView } from '../ddgi-shared/harness.ts';
import { describeErrors, stepResponse } from '../ddgi-shared/response.ts';
import { convergeSponza, lightWeightedError, probeMeans, sunDirection, turnedSun, VIEW_HEIGHT, VIEW_WIDTH, type ProbeMeans, type SponzaUnderTest } from '../ddgi-shared/sponza-response.ts';
import { startViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SCENE_ID = 'furnished-room';
/** The furnished room's batch at Budget v2's 2 ms on the RTX 3060 (T2.6a's VALID calibration, PROGRESS). */
const DEFAULT_UNITS = 571;
const UNITS = Number(env.VITE_DDGI_UNITS ?? DEFAULT_UNITS);
const BOOKMARKS = listFromEnv(env.VITE_DDGI_BOOKMARKS, ['1', '2']).map(Number);
const SUN_STEP_DEGREES = 30;
const TOLERANCE = 0.05;
/** The bar: 0.5 s. */
const VIEW_FRAMES = 30;
/** A view the check is judged on must move by at least this much (TASKS, T2.6a). */
const STEP_AT_LEAST = 0.25;
const MEASURED_FRAMES = 180;
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
    const scene = await startViewerScene(device, manifest, { gpuTiming: true, label });
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
    results.sunStepDegrees = SUN_STEP_DEGREES;
    const failures: string[] = [];

    // Settled: converged from the start in the new light.
    const reference = await startRoom(device, 'furnished room, the sun stepped, from the start', BOOKMARKS[0]);
    const startDirection = sunDirection(reference.sun);
    const stepped = turnedSun(startDirection, SUN_STEP_DEGREES);
    results.sunElevationDegrees = { before: (Math.asin(-startDirection.y) * 180) / Math.PI, after: (Math.asin(-stepped.y) * 180) / Math.PI };
    stepSun(reference, stepped);
    await convergeSponza(reference, `the sun ${SUN_STEP_DEGREES} degrees lower, from the start`, log, null);
    const settledImages = new Map<number, Float32Array>();
    for (const bookmark of BOOKMARKS) settledImages.set(bookmark, await reference.imageOf(bookmark));
    const settledField: ProbeMeans = await probeMeans(reference);
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

        stepSun(room, stepped);
        const means: number[] = [];
        const perPixel: Array<{ frame: number; value: number }> = [];
        const volume: Array<{ frame: number; value: number }> = [];
        let bakesAtStep = 0;
        for (let frame = 1; frame <= MEASURED_FRAMES; frame++) {
            room.scene.frame({ units: UNITS });
            if (frame === 1) bakesAtStep = room.scene.platform.shade.sky.projections - bakesBefore;
            const image = await room.imageOf(bookmark);
            means.push(imageMean(image));
            perPixel.push({ frame, value: perPixelError(image, settledImage) });
            if (frame === 1 || frame % VOLUME_EVERY === 0) volume.push({ frame, value: lightWeightedError(await probeMeans(room), settledField) });
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
            volumeSettledFrame: volumeSettled,
            volume,
            viewErrors: response.errors,
            perPixel: perPixel.map((entry) => entry.value),
            skyBakes: { inTheStepFrame: bakesAtStep, inAll: bakesAfter },
            probesMarkedTotal: counters['probes marked (total)'],
        };
        log(
            `bookmark ${bookmark} (${reference.scene.manifest.cameras[bookmark - 1].name}): the step moves the view by ${(response.stepShare * 100).toFixed(1)}% (per pixel ${(stepPerPixel * 100).toFixed(1)}%)${judged ? '' : `, under ${STEP_AT_LEAST * 100}%: reported, not judged`}; ` +
                `view within ${TOLERANCE * 100}% for good from frame ${response.settledFrame ?? 'never'} (bar ${VIEW_FRAMES}), per pixel from frame ${perPixelSettled ?? 'never'} (at frame ${VIEW_FRAMES}: ${((perPixel[VIEW_FRAMES - 1]?.value ?? NaN) * 100).toFixed(1)}%); ` +
                `whole volume within 5% from frame ${volumeSettled ?? 'never'}; the sky baked ${bakesAtStep} time(s) in the step's frame, ${bakesAfter} in all\n` +
                `    view error by frame: ${describeErrors(response.errors)}\n    per pixel: ${perPixel.filter((entry) => entry.frame <= 12 || entry.frame % 6 === 0).map((entry) => `${entry.frame}: ${(entry.value * 100).toFixed(1)}%`).join(', ')}\n` +
                `    volume (light-weighted): ${volume.map((entry) => `${entry.frame}: ${(entry.value * 100).toFixed(1)}%`).join(', ')}`,
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
