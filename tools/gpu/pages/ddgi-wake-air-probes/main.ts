/**
 * T3.4 items 3 and 5 (docs/design/m3-moving.md §5.2, §5.3): an object moving
 * through empty space gets GI. Classification switches off probes with no
 * surface within reach ("no surface nearby", 347 of Cornell's 1,728), so a
 * ball in mid-air used to read no probes at all. Probes near a receiver are
 * now woken: they come back on, warm over 2 blended updates, and the ball's
 * lighting follows it.
 *
 * The scene: the Cornell box, its field converged with every probe a frame.
 * A ball of radius BALL_RADIUS that rays don't see (`contributesToGI: false`,
 * a receiver) is created at the red wall's side and moved across the box's
 * open middle, one `setTransforms` a frame, for PATH_FRAMES frames, at
 * UNITS probes a frame (about a seventh of the volume, so the woken probes'
 * priority is what serves them). Then it is destroyed and the field runs on
 * until every wake has run out.
 *
 * Checks:
 * - **item 3, every frame:** every active probe has had at least 2 blended
 *   updates (k ≥ 2), and each air probe that came back on was seen in state 3
 *   (warming) first and became active with exactly k = 2;
 * - **woken:** the air probes around the ball's path (in the cells it passed
 *   through) were all woken and were active within WAKE_DEADLINE_FRAMES of
 *   the ball's first frame near them;
 * - **item 5, the ball's GI:** its mean irradiance over its surface, in each
 *   frame from JUDGED_FROM_FRAME on, is within BALL_GI_TOLERANCE (mean) of a
 *   reference: the same ball position in a field converged with
 *   classification off (every probe on). Its colour (red against green, as
 *   it crosses from one wall towards the other) follows the reference's
 *   (correlation ≥ COLOUR_CORRELATION);
 * - **asleep again:** after the ball is gone and WAKE_FRAMES have passed,
 *   the probes off for "no surface nearby" are exactly those at the start,
 *   and no probe is awake: a still scene is as it was (item 4's "air probes
 *   still asleep").
 *
 * **The screenshot** (item 5 is judged by eye): three rows of the ball,
 * shaded with the field's irradiance (white albedo) at six points of its
 * path, seen from the front camera: the field as the ball passed (woken
 * probes), the reference (every probe on), and the field before the ball
 * came (air probes asleep: what M2 gave).
 */

import { assert, runGpuTest } from '../../client.ts';
import type { InstanceHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { PROBE_STATE } from '../../../../packages/ddgi-core/src/techniques/ddgi/layout.ts';
import { WAKE_FRAMES } from '../../../../packages/ddgi-core/src/techniques/ddgi/mark-table.ts';
import { createDdgiScene, FieldProbe, releaseDeviceOnPageHide, type DdgiScene, type SurfacePoint } from '../ddgi-shared/harness.ts';
import { probeSnapshot, type ProbeSnapshot } from '../ddgi-shared/moving.ts';
import { bookmarkCamera, runCameraFrame, runCameraFrames } from '../ddgi-shared/response.ts';

type Vec3 = [number, number, number];

const SCENE = 'cornell-box';
/** Probes a frame while the ball moves: about a seventh of Cornell's 1,728. */
const UNITS = 256;
/** Frames with every probe a frame to converge each field. */
const CONVERGE_FRAMES = 300;
/** The ball's path: across the box's open middle, above the tall block (3.3 m), from near one side wall to near the other. */
const PATH_START: Vec3 = [1.0, 3.9, 2.8];
const PATH_END: Vec3 = [4.6, 3.9, 2.8];
/** 2.5 s at 60 Hz: 1.44 m/s. */
const PATH_FRAMES = 150;
const BALL_RADIUS = 0.3;
/** The ball's GI is judged from this frame of its path on (its first frames wait for the probes around it to warm). */
const JUDGED_FROM_FRAME = 4;
/** An air probe in a cell the ball passes must be active within this many frames of the ball's first frame within reach of it. */
const WAKE_DEADLINE_FRAMES = 4;
/** The ball's GI against the reference: the mean relative difference over the judged frames... */
const BALL_GI_TOLERANCE = 0.15;
/** ...and no judged frame below this share of the reference (no frame without GI). */
const BALL_GI_FLOOR = 0.5;
/** The correlation of the ball's colour (red - green over red + green) with the reference's along the path. */
const COLOUR_CORRELATION = 0.9;
/** Points on the ball's surface its mean irradiance is taken over. */
const BALL_POINTS = 128;
/** The screenshot: the ball at these frames of its path. */
const SHOWN_FRAMES = [10, 36, 62, 88, 114, 140];
const DISC_PIXELS = 96;
const DISC_GAP = 12;
const LABEL_WIDTH = 150;
/** The albedo the screenshot shades the ball with (white). */
const DISPLAY_ALBEDO = 0.8;

function pathPoint(frame: number): Vec3 {
    const t = Math.min(1, Math.max(0, frame / PATH_FRAMES));
    return [0, 1, 2].map((axis) => PATH_START[axis] + (PATH_END[axis] - PATH_START[axis]) * t) as Vec3;
}

function ballTransform(centre: Vec3): number[] {
    return [BALL_RADIUS, 0, 0, 0, 0, BALL_RADIUS, 0, 0, 0, 0, BALL_RADIUS, 0, centre[0], centre[1], centre[2], 1];
}

/** A unit sphere, latitude and longitude bands. Rays never see it (a receiver), so its winding doesn't matter. */
function unitSphere(bands = 12, segments = 24): { positions: Float32Array; indices: Uint32Array } {
    const positions: number[] = [];
    const indices: number[] = [];
    for (let band = 0; band <= bands; band++) {
        const polar = (band / bands) * Math.PI;
        for (let segment = 0; segment <= segments; segment++) {
            const azimuth = (segment / segments) * 2 * Math.PI;
            positions.push(Math.sin(polar) * Math.cos(azimuth), Math.cos(polar), Math.sin(polar) * Math.sin(azimuth));
        }
    }
    const row = segments + 1;
    for (let band = 0; band < bands; band++) {
        for (let segment = 0; segment < segments; segment++) {
            const a = band * row + segment;
            indices.push(a, a + row, a + 1, a + 1, a + row, a + row + 1);
        }
    }
    return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

/** Points spread evenly over the ball's surface (spherical Fibonacci), each seen from straight outside. */
function ballSurface(centre: Vec3): SurfacePoint[] {
    const points: SurfacePoint[] = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let index = 0; index < BALL_POINTS; index++) {
        const y = 1 - (2 * (index + 0.5)) / BALL_POINTS;
        const ring = Math.sqrt(1 - y * y);
        const normal: Vec3 = [ring * Math.cos(golden * index), y, ring * Math.sin(golden * index)];
        points.push({ position: [0, 1, 2].map((axis) => centre[axis] + BALL_RADIUS * normal[axis]) as Vec3, normal, viewDirection: [-normal[0], -normal[1], -normal[2]] });
    }
    return points;
}

/** The ball as the front camera sees it (from +z): one point per pixel of a DISC_PIXELS disc, null outside it. */
function ballDisc(centre: Vec3): (SurfacePoint | null)[] {
    const pixels: (SurfacePoint | null)[] = [];
    for (let row = 0; row < DISC_PIXELS; row++) {
        for (let column = 0; column < DISC_PIXELS; column++) {
            const x = ((column + 0.5) / DISC_PIXELS) * 2 - 1;
            const y = 1 - ((row + 0.5) / DISC_PIXELS) * 2;
            const squared = x * x + y * y;
            if (squared > 1) {
                pixels.push(null);
                continue;
            }
            const normal: Vec3 = [x, y, Math.sqrt(1 - squared)];
            pixels.push({ position: [0, 1, 2].map((axis) => centre[axis] + BALL_RADIUS * normal[axis]) as Vec3, normal, viewDirection: [0, 0, -1] });
        }
    }
    return pixels;
}

/** Irradiance per disc pixel (null outside the disc). */
async function shadeDisc(probe: FieldProbe, scene: DdgiScene, centre: Vec3): Promise<(Vec3 | null)[]> {
    const disc = ballDisc(centre);
    const points = disc.filter((point): point is SurfacePoint => point !== null);
    const values = await probe.sample(scene.ddgi.field!, points);
    let next = 0;
    return disc.map((point) => (point ? (values[next++] as Vec3) : null));
}

function meanOf(values: readonly Vec3[]): Vec3 {
    const sum: Vec3 = [0, 0, 0];
    for (const value of values) for (let channel = 0; channel < 3; channel++) sum[channel] += value[channel];
    return sum.map((channel) => channel / Math.max(values.length, 1)) as Vec3;
}

function luminance(rgb: Vec3): number {
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

/** Red against green, -1..1: which side wall's bounce the ball sees more of. */
function redGreen(rgb: Vec3): number {
    return (rgb[0] - rgb[1]) / Math.max(rgb[0] + rgb[1], 1e-9);
}

function correlation(a: readonly number[], b: readonly number[]): number {
    const meanA = a.reduce((sum, value) => sum + value, 0) / a.length;
    const meanB = b.reduce((sum, value) => sum + value, 0) / b.length;
    let product = 0;
    let squaredA = 0;
    let squaredB = 0;
    for (let index = 0; index < a.length; index++) {
        product += (a[index] - meanA) * (b[index] - meanB);
        squaredA += (a[index] - meanA) ** 2;
        squaredB += (b[index] - meanB) ** 2;
    }
    return product / Math.sqrt(squaredA * squaredB);
}

/** The probe indices of the 8 corners of the cell a point is in (the probes shading reads there), clamped to the grid. */
function cellProbes(scene: DdgiScene, point: Vec3): number[] {
    const grid = scene.ddgi.field!.grid;
    const base = [0, 1, 2].map((axis) => Math.min(Math.max(Math.floor((point[axis] - grid.origin[axis]) / grid.spacing[axis]), 0), grid.counts[axis] - 2));
    const probes: number[] = [];
    for (let corner = 0; corner < 8; corner++) {
        const [x, y, z] = [base[0] + (corner & 1), base[1] + ((corner >> 1) & 1), base[2] + ((corner >> 2) & 1)];
        probes.push(x + grid.counts[0] * (z + grid.counts[2] * y));
    }
    return probes;
}

/** Every active probe has had at least 2 blended updates (item 3); returns those that break it. */
function brokenRule(snapshot: ProbeSnapshot): number[] {
    const broken: number[] = [];
    snapshot.state.forEach((state, probe) => {
        if (state === PROBE_STATE.active && snapshot.updates[probe] < 2) broken.push(probe);
    });
    return broken;
}

/** Draws the three rows of discs on a canvas at the top of the page, for the runner's screenshot. */
function drawRows(rows: readonly { label: string; discs: (Vec3 | null)[][] }[], exposure: number): void {
    const width = LABEL_WIDTH + SHOWN_FRAMES.length * (DISC_PIXELS + DISC_GAP);
    const height = 28 + rows.length * (DISC_PIXELS + DISC_GAP);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d')!;
    context.fillStyle = '#202020';
    context.fillRect(0, 0, width, height);
    context.fillStyle = '#e0e0e0';
    context.font = '12px sans-serif';
    SHOWN_FRAMES.forEach((frame, index) => context.fillText(`frame ${frame}, x ${pathPoint(frame)[0].toFixed(2)}`, LABEL_WIDTH + index * (DISC_PIXELS + DISC_GAP), 16));
    rows.forEach((row, rowIndex) => {
        const top = 28 + rowIndex * (DISC_PIXELS + DISC_GAP);
        context.fillStyle = '#e0e0e0';
        row.label.split('\n').forEach((line, lineIndex) => context.fillText(line, 6, top + 20 + lineIndex * 15));
        row.discs.forEach((disc, discIndex) => {
            const image = context.createImageData(DISC_PIXELS, DISC_PIXELS);
            disc.forEach((irradiance, pixel) => {
                // Outgoing radiance of a white diffuse ball, albedo / pi x E, exposed and gamma-encoded.
                const rgb = irradiance ? irradiance.map((channel) => 1 - Math.exp((-exposure * DISPLAY_ALBEDO * channel) / Math.PI)) : [0.125, 0.125, 0.125];
                for (let channel = 0; channel < 3; channel++) image.data[pixel * 4 + channel] = Math.round(255 * Math.pow(Math.min(1, Math.max(0, rgb[channel])), 1 / 2.2));
                image.data[pixel * 4 + 3] = 255;
            });
            context.putImageData(image, LABEL_WIDTH + discIndex * (DISC_PIXELS + DISC_GAP), top);
        });
    });
    document.body.prepend(canvas);
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const scene = await createDdgiScene(device, SCENE);
        const field = scene.ddgi.field;
        assert(field, 'no field');
        const probeCount = field.grid.probeCount;
        const frontCamera = scene.manifest.cameras.find((camera) => camera.name === 'front');
        assert(frontCamera, 'Cornell has no front camera');
        const camera = bookmarkCamera(frontCamera, 16 / 9);
        const fieldProbe = await FieldProbe.create(device, scene.platform);

        // The still field: converged, its air probes asleep.
        await runCameraFrames(device, scene.platform, CONVERGE_FRAMES, probeCount, camera);
        const still = await probeSnapshot(device, field);
        const asleepAtStart = still.state.flatMap((state, probe) => (state === PROBE_STATE.offNoSurfaceNearby ? [probe] : []));
        const awakeAtStart = still.wakeUntil.filter((until) => until !== 0).length;
        log(`${SCENE}: ${probeCount} probes; still and converged: ${asleepAtStart.length} off for "no surface nearby", ${awakeAtStart} awake`);
        assert(asleepAtStart.length >= 100, `only ${asleepAtStart.length} probes are off for "no surface nearby"; the check needs Cornell's air probes`);
        assert(awakeAtStart === 0, `${awakeAtStart} probes are awake in a still scene`);

        // The ball before it comes, at the shown points: the field with its air probes asleep.
        const asleepDiscs: (Vec3 | null)[][] = [];
        const asleepBall: Vec3[] = [];
        for (const frame of SHOWN_FRAMES) {
            asleepDiscs.push(await shadeDisc(fieldProbe, scene, pathPoint(frame)));
            asleepBall.push(meanOf(await fieldProbe.sample(field, ballSurface(pathPoint(frame)))));
        }

        // The ball: a receiver, moved across the box one frame at a time.
        const sceneApi = scene.platform.scene;
        const sphere = unitSphere();
        const mesh = sceneApi.createMesh({ positions: sphere.positions, indices: sphere.indices, label: 'ball' });
        const ball: InstanceHandle = sceneApi.createInstance({ mesh, materials: [scene.materials[0]], transform: ballTransform(pathPoint(0)), contributesToGI: false, label: 'ball' });
        const asleep = new Set(asleepAtStart);
        const warmingFrom = new Map<number, number>();
        const activeFrom = new Map<number, number>();
        const wokenFrom = new Map<number, number>();
        // Per air probe of a cell the ball passed: the first frame the ball was within reach of it (its cell's corners).
        const neededFrom = new Map<number, number>();
        const ballGi: Vec3[] = [];
        const wokenDiscs: (Vec3 | null)[][] = [];
        for (let frame = 1; frame <= PATH_FRAMES; frame++) {
            const centre = pathPoint(frame);
            sceneApi.setTransforms(Uint32Array.of(ball), new Float32Array(ballTransform(centre)));
            runCameraFrame(device, scene.platform, UNITS, camera);
            const now = await probeSnapshot(device, field);
            const broken = brokenRule(now);
            assert(broken.length === 0, `frame ${frame}: ${broken.length} active probes have fewer than 2 blended updates (first: probe ${broken[0]}, k = ${now.updates[broken[0]]})`);
            for (const probe of asleep) {
                if (now.wakeUntil[probe] !== 0 && !wokenFrom.has(probe)) wokenFrom.set(probe, frame);
                if (now.state[probe] === PROBE_STATE.notUpdatedYet && !warmingFrom.has(probe)) warmingFrom.set(probe, frame);
                if (now.state[probe] === PROBE_STATE.active && !activeFrom.has(probe)) {
                    activeFrom.set(probe, frame);
                    assert(now.updates[probe] === 2, `air probe ${probe} became active with k = ${now.updates[probe]}, not 2`);
                    assert((warmingFrom.get(probe) ?? Infinity) < frame, `air probe ${probe} became active in frame ${frame} without being seen warming first`);
                }
            }
            for (const probe of cellProbes(scene, centre)) if (asleep.has(probe) && !neededFrom.has(probe)) neededFrom.set(probe, frame);
            ballGi.push(meanOf(await fieldProbe.sample(field, ballSurface(centre))));
            if (SHOWN_FRAMES.includes(frame)) wokenDiscs.push(await shadeDisc(fieldProbe, scene, centre));
        }

        // Each air probe the ball needed was woken and active within the deadline.
        const late: string[] = [];
        for (const [probe, needed] of neededFrom) {
            const active = activeFrom.get(probe);
            if (active === undefined || active - needed > WAKE_DEADLINE_FRAMES) late.push(`probe ${probe}: needed from frame ${needed}, active ${active ?? 'never'}`);
        }
        const wakeDelays = [...neededFrom].map(([probe, needed]) => (activeFrom.get(probe) ?? Infinity) - needed);
        log(
            `the ball's path: ${wokenFrom.size} air probes woken, ${activeFrom.size} of them active again; ${neededFrom.size} were corners of cells the ball passed through, ` +
                `active ${Math.min(...wakeDelays)} to ${Math.max(...wakeDelays)} frames after the ball first needed them (the first ones at frame ${Math.min(...activeFrom.values())})`,
        );
        assert(neededFrom.size >= 4, `the ball passed only ${neededFrom.size} air probes; the path doesn't test waking`);
        assert(late.length === 0, `${late.length} air probes the ball needed were not active within ${WAKE_DEADLINE_FRAMES} frames: ${late.slice(0, 4).join('; ')}`);

        // Gone: after its last wake runs out, every woken probe goes back to sleep.
        sceneApi.destroyInstance(ball);
        sceneApi.destroyMesh(mesh);
        const sleepFrames = WAKE_FRAMES + 3 * Math.ceil(probeCount / UNITS);
        await runCameraFrames(device, scene.platform, sleepFrames, UNITS, camera);
        const after = await probeSnapshot(device, field);
        const asleepAfter = new Set(after.state.flatMap((state, probe) => (state === PROBE_STATE.offNoSurfaceNearby ? [probe] : [])));
        const stillAwake = after.wakeUntil.filter((until) => until !== 0).length;
        const notBack = asleepAtStart.filter((probe) => !asleepAfter.has(probe)).length;
        const newlyAsleep = [...asleepAfter].filter((probe) => !asleep.has(probe)).length;
        log(`${sleepFrames} frames after the ball left: ${asleepAfter.size} off for "no surface nearby" (${notBack} of the start's not back asleep, ${newlyAsleep} new), ${stillAwake} awake`);
        assert(notBack === 0 && newlyAsleep === 0, `after the ball left, ${notBack} air probes are still on and ${newlyAsleep} others are off`);
        assert(stillAwake === 0, `${stillAwake} probes are still awake ${sleepFrames} frames after the ball left`);
        scene.platform.dispose();

        // The reference: the same scene converged with every probe on (classification off), the ball's GI at the same points.
        const reference = await createDdgiScene(device, SCENE, { classification: false });
        const referenceProbe = await FieldProbe.create(device, reference.platform);
        await runCameraFrames(device, reference.platform, CONVERGE_FRAMES, probeCount, camera);
        const referenceGi: Vec3[] = [];
        for (let frame = 1; frame <= PATH_FRAMES; frame++) referenceGi.push(meanOf(await referenceProbe.sample(reference.ddgi.field!, ballSurface(pathPoint(frame)))));
        const referenceDiscs: (Vec3 | null)[][] = [];
        for (const frame of SHOWN_FRAMES) referenceDiscs.push(await shadeDisc(referenceProbe, reference, pathPoint(frame)));
        reference.platform.dispose();

        // The ball's GI against the reference, from JUDGED_FROM_FRAME on.
        const judged = ballGi.map((gi, index) => ({ frame: index + 1, ratio: luminance(gi) / luminance(referenceGi[index]), ours: redGreen(gi), theirs: redGreen(referenceGi[index]) })).filter(({ frame }) => frame >= JUDGED_FROM_FRAME);
        const meanDifference = judged.reduce((sum, { ratio }) => sum + Math.abs(ratio - 1), 0) / judged.length;
        const lowest = judged.reduce((worst, entry) => (entry.ratio < worst.ratio ? entry : worst));
        const colourCorrelation = correlation(
            judged.map(({ ours }) => ours),
            judged.map(({ theirs }) => theirs),
        );
        const colourRange = [Math.min(...judged.map(({ theirs }) => theirs)), Math.max(...judged.map(({ theirs }) => theirs))];
        const asleepShare = asleepBall.map((gi, index) => luminance(gi) / luminance(referenceGi[SHOWN_FRAMES[index] - 1]));
        log(
            `the ball's GI against every probe on, frames ${JUDGED_FROM_FRAME}-${PATH_FRAMES}: mean difference ${(meanDifference * 100).toFixed(1)}%, lowest ${(lowest.ratio * 100).toFixed(1)}% at frame ${lowest.frame}; ` +
                `frames 1-3: ${ballGi
                    .slice(0, 3)
                    .map((gi, index) => `${((luminance(gi) / luminance(referenceGi[index])) * 100).toFixed(0)}%`)
                    .join(', ')}`,
        );
        log(
            `its colour (red - green) / (red + green) follows the reference's with correlation ${colourCorrelation.toFixed(3)} (reference from ${colourRange[0].toFixed(3)} to ${colourRange[1].toFixed(3)}); ` +
                `before it came (air probes asleep) the shown points read ${asleepShare.map((share) => `${(share * 100).toFixed(0)}%`).join(', ')} of the reference`,
        );

        // The screenshot: exposed so the reference's brightest ball reads about 0.85.
        const brightest = Math.max(...referenceDiscs.flat().map((value) => (value ? Math.max(...value) : 0)));
        drawRows(
            [
                { label: 'woken probes\n(as the ball passed)', discs: wokenDiscs },
                { label: 'reference\n(every probe on)', discs: referenceDiscs },
                { label: 'before it came\n(air probes asleep)', discs: asleepDiscs },
            ],
            (2 * Math.PI) / (DISPLAY_ALBEDO * Math.max(brightest, 1e-6)),
        );

        assert(meanDifference <= BALL_GI_TOLERANCE, `the ball's GI differs from the reference by ${(meanDifference * 100).toFixed(1)}% on average (limit ${BALL_GI_TOLERANCE * 100}%)`);
        assert(lowest.ratio >= BALL_GI_FLOOR, `the ball's GI fell to ${(lowest.ratio * 100).toFixed(1)}% of the reference at frame ${lowest.frame}`);
        assert(colourCorrelation >= COLOUR_CORRELATION, `the ball's colour follows the reference's with correlation ${colourCorrelation.toFixed(3)} only`);
        return {
            probes: probeCount,
            units: UNITS,
            asleepAtStart: asleepAtStart.length,
            woken: wokenFrom.size,
            wokenActive: activeFrom.size,
            neededByTheBall: neededFrom.size,
            wakeDelayFrames: { min: Math.min(...wakeDelays), max: Math.max(...wakeDelays) },
            ballGiMeanDifference: meanDifference,
            ballGiLowest: lowest,
            ballGiFirstFrames: ballGi.slice(0, 3).map((gi, index) => luminance(gi) / luminance(referenceGi[index])),
            colourCorrelation,
            asleepShareOfReference: asleepShare,
            sleepFrames,
            asleepAfter: asleepAfter.size,
        };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
