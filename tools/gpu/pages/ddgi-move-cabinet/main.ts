/**
 * T2.M1 items 3 and 8 (docs/design/m3-moving.md §5.1, §8 "Drag furniture";
 * T3.4 check 1): on the furnished room, a 1 m cabinet move is within 5% of
 * settled in ≤ 1 s, at the default budget, GI only; and the probes the move
 * marks against the probes whose light really changed by more than 5% (the
 * far-set guess, m3-moving.md risk 4).
 *
 * The scene is the viewer's (`loadScene`, the three adapter mirroring it),
 * with a cabinet the breakfast room doesn't have: a 0.9 × 0.85 × 0.45 m box
 * of brown wood standing against the north wall. DDGI runs on the manifest's
 * volume at the default budget: the batch Budget v2 calibrates at 2 ms in
 * this run (VITE_DDGI_UNITS gives a fixed one instead).
 *
 * 1. The field is converged with the cabinet where it stands
 *    (ddgi-shared/response.ts `convergeField`), and the GI-only test view
 *    (`CAMERA`, looking at the wall) is taken: before.
 * 2. The cabinet slides 1 m along the wall in one step, through three (its
 *    position, `markChanged` as the viewer's drag gizmo does), and the adapter
 *    sends it in its next update. That frame updates no probe, so each
 *    probe's `pendingChange` is what the move marked (item 8).
 * 3. RESPONSE_FRAMES frames at the batch; the view each frame.
 * 4. The field is converged with the cabinet in its new place: settled.
 *
 * **What "within 5% of settled" is measured on** (DECISIONS 2026-11-06,
 * T2.M1): the pixels the move changed, those whose GI differs between
 * before and settled by more than 5% of settled; over them, the GI's sum
 * against settled's. A cabinet changes a part of the room, so the whole
 * view's mean moves by far less than the 25% T2.4b asked of a step, and
 * would be "within 5%" from the first frame whatever the field did. The
 * whole view's mean error and the region's per-pixel error (the mean over
 * its pixels of |now − settled| over settled's mean, which has a floor of
 * DDGI's own noise against a converged field) are reported beside it. It
 * must enter the band within RESPONSE_DEADLINE_FRAMES (1 s) and stay.
 *
 *   VITE_DDGI_UNITS=<probes a frame>   (default: calibrated at 2 ms)
 *
 * Needs the downloaded furnished room. `pnpm test:gpu --gpu nvidia --filter ddgi-move-cabinet`.
 */

import { BoxGeometry, Box3, Mesh, MeshStandardMaterial, type Object3D } from 'three/webgpu';
import { assert, runGpuTest } from '../../client.ts';
import { NEAR_SET_SPACINGS } from '../../../../packages/ddgi-core/src/techniques/ddgi/mark-table.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { TestView } from '../ddgi-shared/harness.ts';
import { markRecall, missedByDistance, nearBox, perPixelError, probeSnapshot } from '../ddgi-shared/moving.ts';
import { bookmarkCamera, convergeField, describeErrors, stepResponse, type ConvergeOptions } from '../ddgi-shared/response.ts';
import { startViewerScene, yieldToCallbacks } from '../ddgi-shared/viewer-scene.ts';

const env = import.meta.env as Record<string, unknown>;
const FIXED_UNITS = Number(env.VITE_DDGI_UNITS) > 0 ? Math.round(Number(env.VITE_DDGI_UNITS)) : null;
const SCENE = 'furnished-room';
const CABINET_SIZE: [number, number, number] = [0.9, 0.85, 0.45];
/** The move: 1 m along the north wall (+x). */
const MOVE: [number, number, number] = [1, 0, 0];
/** The view: from the room's south side, looking at the north wall where the cabinet stands. */
const CAMERA = { name: 'towards-the-cabinet', position: [0.2, 1.3, 1.6] as [number, number, number], target: [-0.3, 0.6, -2.3] as [number, number, number], fovYDeg: 65 };
const WIDTH = 320;
const HEIGHT = 180;
const TOLERANCE = 0.05;
/** 1 s at 60 Hz. */
const RESPONSE_DEADLINE_FRAMES = 60;
/** Frames measured after the move (3 s), to see the region leave the band again, or not. */
const RESPONSE_FRAMES = 180;
/** A pixel counts as changed by the move when its GI moved by more than this share of settled. */
const CHANGED_PIXEL = 0.05;
const CALIBRATION_FRAMES = 400;
const CONVERGE: Omit<ConvergeOptions, 'settleUnits'> = {
    fastFrames: 96,
    fastHysteresis: 0.5,
    blockFrames: 32,
    convergedTolerance: 0.001,
    maxBlocks: 24,
    settleFrames: 1200,
    convergeUnits: 4096,
};

/** The pixels (rgb triples' indices) whose GI changed by more than CHANGED_PIXEL between two images. */
function changedPixels(before: Float32Array, settled: Float32Array): number[] {
    const pixels: number[] = [];
    for (let pixel = 0; pixel < before.length / 3; pixel++) {
        const a = before[pixel * 3] + before[pixel * 3 + 1] + before[pixel * 3 + 2];
        const b = settled[pixel * 3] + settled[pixel * 3 + 1] + settled[pixel * 3 + 2];
        if (Math.abs(a - b) > CHANGED_PIXEL * Math.max(b, 1e-6)) pixels.push(pixel);
    }
    return pixels;
}

function regionSum(image: Float32Array, pixels: readonly number[]): number {
    let sum = 0;
    for (const pixel of pixels) sum += image[pixel * 3] + image[pixel * 3 + 1] + image[pixel * 3 + 2];
    return sum;
}

function regionImage(image: Float32Array, pixels: readonly number[]): Float32Array {
    const out = new Float32Array(pixels.length * 3);
    pixels.forEach((pixel, index) => out.set(image.subarray(pixel * 3, pixel * 3 + 3), index * 3));
    return out;
}

function mean(values: Float32Array): number {
    let sum = 0;
    for (const value of values) sum += value;
    return sum / values.length;
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const manifest = loadAllManifests().find((candidate) => candidate.id === SCENE);
        assert(manifest, `no manifest ${SCENE}`);
        let cabinet: Mesh | null = null;
        const scene = await startViewerScene(device, manifest, {
            gpuTiming: true,
            label: 'move cabinet',
            prepareScene: (loaded) => {
                // The room's floor and north wall, from the content's world box.
                const box = new Box3().setFromObject(loaded.content as Object3D);
                cabinet = new Mesh(new BoxGeometry(...CABINET_SIZE), new MeshStandardMaterial({ color: 0x73502e, roughness: 1, metalness: 0 }));
                cabinet.name = 'cabinet';
                cabinet.position.set(-1.0, box.min.y + CABINET_SIZE[1] / 2 + 0.002, box.min.z + CABINET_SIZE[2] / 2 + 0.03);
                loaded.content.add(cabinet);
                loaded.content.updateMatrixWorld(true);
            },
        });
        assert(cabinet, 'the cabinet was not added');
        const moving = cabinet as Mesh;
        const field = scene.ddgi.field;
        assert(field, 'no field');
        const view = await TestView.create(device, scene.platform);
        const image = async () => (await view.render(field, CAMERA, WIDTH, HEIGHT)).indirect;
        const frameCamera = bookmarkCamera(CAMERA, WIDTH / HEIGHT);
        // The adapter's camera (the scheduler's camera weight) looks the same way.
        scene.camera.position.set(...CAMERA.position);
        scene.camera.lookAt(...CAMERA.target);
        scene.camera.updateMatrixWorld();

        // The default budget: Budget v2's calibration at 2 ms, unless a run gives the batch.
        let units = FIXED_UNITS;
        if (units === null) {
            for (let frame = 0; frame < CALIBRATION_FRAMES && scene.platform.budgetReport(scene.ddgi)?.calibratedUnits === undefined; frame++) {
                scene.frame({ gpuMs: 2 });
                await device.queue.onSubmittedWorkDone();
                await yieldToCallbacks();
            }
            units = scene.platform.budgetReport(scene.ddgi)?.calibratedUnits ?? null;
            assert(units !== null, `no calibration after ${CALIBRATION_FRAMES} frames`);
        }
        log(`${SCENE}: ${scene.probeCount} probes; ${units} probes a frame (${FIXED_UNITS === null ? 'calibrated at 2 ms in this run' : 'given'})`);

        await convergeField(device, scene.platform, scene.ddgi.settings, frameCamera, async () => mean(await image()), { ...CONVERGE, settleUnits: units }, 'before the move', log);
        const before = await image();
        const probesBefore = await probeSnapshot(device, field);
        const boxBefore = new Box3().setFromObject(moving);

        // The move: through three, sent by the adapter in its next update; that frame updates no probe.
        moving.position.add({ x: MOVE[0], y: MOVE[1], z: MOVE[2] } as never);
        moving.updateMatrixWorld();
        scene.threeGI.markChanged(moving);
        scene.frame({ units: 0 });
        assert(scene.threeGI.report.transforms === 1, `the adapter sent ${scene.threeGI.report.transforms} transforms for the move`);
        const marked = await probeSnapshot(device, field);
        const boxAfter = new Box3().setFromObject(moving);

        const images: Float32Array[] = [];
        for (let frame = 1; frame <= RESPONSE_FRAMES; frame++) {
            scene.frame({ units });
            images.push(await image());
        }
        await convergeField(device, scene.platform, scene.ddgi.settings, frameCamera, async () => mean(await image()), { ...CONVERGE, settleUnits: units }, 'after the move', log);
        const settled = await image();
        const probesAfter = await probeSnapshot(device, field);

        // The response, on the pixels the move changed (the file comment).
        const region = changedPixels(before, settled);
        assert(region.length > 0.005 * WIDTH * HEIGHT, `the move changed only ${region.length} pixels of the view`);
        const settledRegion = regionSum(settled, region);
        const regionResponse = stepResponse(regionSum(before, region), settledRegion, images.map((frame) => regionSum(frame, region)), TOLERANCE);
        const viewResponse = stepResponse(mean(before), mean(settled), images.map(mean), TOLERANCE);
        const perPixel = [1, 10, 30, 60, 120, 180].map((frame) => ({ frame, error: perPixelError(regionImage(images[frame - 1], region), regionImage(settled, region)) }));
        log(
            `the move changed ${region.length} pixels (${((100 * region.length) / (WIDTH * HEIGHT)).toFixed(1)}% of the view) by ${(regionResponse.stepShare * 100).toFixed(1)}% in all; the view's mean by ${(viewResponse.stepShare * 100).toFixed(2)}%`,
        );
        log(`changed pixels against settled: ${describeErrors(regionResponse.errors)}; within ${TOLERANCE * 100}% for good from frame ${regionResponse.settledFrame} (first ${regionResponse.enteredFrame}), overshoot ${(regionResponse.overshootShareOfStep * 100).toFixed(1)}% of the step`);
        log(`whole view: within ${TOLERANCE * 100}% for good from frame ${viewResponse.settledFrame}; changed pixels per pixel: ${perPixel.map(({ frame, error }) => `${frame}: ${(error * 100).toFixed(1)}%`).join(', ')}`);

        // Item 8: the marks against the probes whose light really changed.
        const margin = field.grid.spacing.map((spacing) => NEAR_SET_SPACINGS * spacing);
        const isNear = (probe: number) =>
            [boxBefore, boxAfter].some((box) => nearBox(probesBefore.position[probe], box.min.toArray(), box.max.toArray(), margin));
        const recall = markRecall(marked, probesBefore, probesAfter, scene.ddgi.settings.changeTolerance, isNear);
        const missed = missedByDistance(recall.missed, probesBefore.position, [boxBefore, boxAfter].map((box) => ({ min: box.min.toArray(), max: box.max.toArray() })), field.grid.spacing);
        log(
            `marks: ${recall.caught} of ${recall.changed} probes that changed by over 5% were marked (recall ${(recall.recall * 100).toFixed(1)}%), precision ${(recall.precision * 100).toFixed(1)}% of ${recall.marked}; ` +
                `near ${recall.near.caught}/${recall.near.changed}, far ${recall.far.caught}/${recall.far.changed} (far marked ${recall.far.marked}); missed by distance from the cabinet in spacings: ${JSON.stringify(missed)}`,
        );
        scene.dispose();

        assert(
            regionResponse.settledFrame !== null && regionResponse.settledFrame <= RESPONSE_DEADLINE_FRAMES,
            `the changed pixels were within ${TOLERANCE * 100}% of settled for good only from frame ${regionResponse.settledFrame} (deadline ${RESPONSE_DEADLINE_FRAMES})`,
        );
        return {
            units,
            probes: scene.probeCount,
            changedPixels: region.length,
            regionStep: regionResponse.stepShare,
            regionSettledFrame: regionResponse.settledFrame,
            regionEnteredFrame: regionResponse.enteredFrame,
            regionOvershootShareOfStep: regionResponse.overshootShareOfStep,
            regionErrors: regionResponse.errors,
            viewStep: viewResponse.stepShare,
            viewSettledFrame: viewResponse.settledFrame,
            perPixel,
            marks: { ...recall, missed: recall.missed.length, missedByDistance: missed },
        };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
