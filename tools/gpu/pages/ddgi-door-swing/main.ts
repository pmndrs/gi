/**
 * T2.M1 items 4 and 8 (docs/design/m3-moving.md §5.1, T3.4 check 2): on the
 * closed-door scene with its door moving, the dark room's leak is ≤ 1%
 * within 1 s of the door closing and stays there; and the probes the door's
 * moves mark against the probes whose light really changed by more than 5%
 * (the far-set guess, m3-moving.md risk 4).
 *
 * **The leak** is DECISIONS 2026-11-04's for the closed door, whose true
 * answer is zero: the mean of the engine-free test view's GI image at
 * `dark-room-leak-view` as a share of its mean at `lit-room` (as
 * `ddgi-leak` measures it, lights without a range window, as in Cycles).
 *
 * **The swing:** the door panel turns about its hinge (the lit side's face,
 * at the doorway's z0 edge) from 90° open into room A to closed, over
 * SWING_FRAMES frames (1 s), eased at both ends, one `setTransforms` a frame,
 * at the default budget: `units` probes a frame, the RTX 3060's calibrated
 * batch at 2 ms for this grid (VITE_DDGI_UNITS overrides it), with the
 * `dark-room-leak-view` camera for the scheduler. Before it, the open state
 * is converged (ddgi-shared/response.ts `convergeField`); after it, the leak
 * is measured every frame for HOLD_FRAMES (10 s). It must be ≤ 1% from frame
 * CLOSE_DEADLINE_FRAMES after the door closed (1 s) and in every frame after.
 * Then the field is converged closed and its leak reported (the floor).
 *
 * **The marks** (item 8), two ways, each from a converged open field:
 * - *step:* the door closed in one step, one frame with no probe updates
 *   (only `ddgi/mark` runs, so each probe's `pendingChange` is what it was
 *   marked);
 * - *swing:* the same 60-frame swing with no probe updates, so each probe's
 *   marks add up over the swing.
 * Against the field converged closed: per probe, the largest channel's
 * relative change of its mean irradiance (`ddgi-shared/moving.ts`). Recall
 * and precision are reported (not judged), with the near set (within 1.5
 * spacings of the door's boxes) and the far set apart.
 *
 *   VITE_DDGI_SECTIONS=swing,marks   (default both)
 *   VITE_DDGI_UNITS=880
 */

import { assert, runGpuTest } from '../../client.ts';
import type { FrameCamera } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { NEAR_SET_SPACINGS } from '../../../../packages/ddgi-core/src/techniques/ddgi/mark-table.ts';
import { createDdgiScene, imageMean, releaseDeviceOnPageHide, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';
import { markRecall, missedByDistance, nearBox, probeSnapshot, type MarkRecall } from '../ddgi-shared/moving.ts';
import { bookmarkCamera, convergeField, runCameraFrame, type ConvergeOptions } from '../ddgi-shared/response.ts';

const env = import.meta.env as Record<string, unknown>;
const SECTIONS = new Set(typeof env.VITE_DDGI_SECTIONS === 'string' && env.VITE_DDGI_SECTIONS !== '' ? env.VITE_DDGI_SECTIONS.split(',').map((item) => item.trim()) : ['swing', 'marks']);
/** Two-room's calibrated batch at 2 ms on the RTX 3060 (T2.4b, VALID run); the closed door has the same 17 x 7 x 9 grid. */
const DEFAULT_UNITS = 880;
const UNITS = Number(env.VITE_DDGI_UNITS) > 0 ? Math.round(Number(env.VITE_DDGI_UNITS)) : DEFAULT_UNITS;
const SWING_FRAMES = 60;
/** The door must be ≤ LEAK_LIMIT from this many frames after it closed (1 s at 60 Hz)... */
const CLOSE_DEADLINE_FRAMES = 60;
/** ...and stay there for this long (10 s). */
const HOLD_FRAMES = 600;
/** SPEC's leak target, as a share of the lit room's GI. */
const LEAK_LIMIT = 0.01;
const OPEN_DEGREES = -90;
const WIDTH = 320;
const HEIGHT = 180;
const CONVERGE: Omit<ConvergeOptions, 'settleUnits'> = {
    fastFrames: 96,
    fastHysteresis: 0.5,
    blockFrames: 32,
    convergedTolerance: 0.001,
    maxBlocks: 24,
    settleFrames: 1200,
    convergeUnits: 4096,
};

type Mat4 = number[];

function multiply(a: Mat4, b: Mat4): Mat4 {
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[column * 4 + k];
            out[column * 4 + row] = sum;
        }
    }
    return out;
}

function translation(x: number, y: number, z: number): Mat4 {
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

/** A turn about +y by `degrees`, as the scene builder's yaw (geometry.ts composeTransform). */
function yaw(degrees: number): Mat4 {
    const angle = (degrees * Math.PI) / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return [cos, 0, -sin, 0, 0, 1, 0, 0, sin, 0, cos, 0, 0, 0, 0, 1];
}

/** The door: its handle, its closed transform, and the transform turned about its hinge (the lit-side face's z0 edge). */
interface Door {
    readonly handle: number;
    readonly closed: Mat4;
    at(degrees: number): Mat4;
    /** The world box of the door at an angle, for telling the near set. */
    boxAt(degrees: number): { min: number[]; max: number[] };
}

function doorOf(scene: DdgiScene): Door {
    const handle = scene.instances.get('door-panel');
    assert(handle !== undefined, 'the closed-door scene has no door-panel');
    const closed = scene.data.instances.find((instance) => instance.name === 'door-panel')!.transform as Mat4;
    // A unit box scaled and placed: its min corner is centre - size / 2 (the builder makes no turn for the door).
    const size = [closed[0], closed[5], closed[10]];
    const hinge = [closed[12] - size[0] / 2, 0, closed[14] - size[2] / 2];
    const at = (degrees: number): Mat4 => multiply(translation(hinge[0], hinge[1], hinge[2]), multiply(yaw(degrees), multiply(translation(-hinge[0], -hinge[1], -hinge[2]), closed)));
    const boxAt = (degrees: number) => {
        const matrix = at(degrees);
        const min = [Infinity, Infinity, Infinity];
        const max = [-Infinity, -Infinity, -Infinity];
        for (let corner = 0; corner < 8; corner++) {
            const local = [corner & 1 ? 0.5 : -0.5, corner & 2 ? 0.5 : -0.5, corner & 4 ? 0.5 : -0.5];
            for (let axis = 0; axis < 3; axis++) {
                const value = matrix[axis] * local[0] + matrix[4 + axis] * local[1] + matrix[8 + axis] * local[2] + matrix[12 + axis];
                min[axis] = Math.min(min[axis], value);
                max[axis] = Math.max(max[axis], value);
            }
        }
        return { min, max };
    };
    return { handle, closed, at, boxAt };
}

/** The angle at frame `frame` of the swing (1..SWING_FRAMES): open to closed, eased at both ends. */
function swingAngle(frame: number): number {
    const t = Math.min(1, frame / SWING_FRAMES);
    const eased = t * t * (3 - 2 * t);
    return OPEN_DEGREES * (1 - eased);
}

/**
 * What convergence is judged on: both rooms' GI means added. Not the dark
 * room's alone: closed, it goes to about 0, and a relative change of almost
 * nothing never settles.
 */
function convergedOn(views: { lit: number; dark: number }): number {
    return views.lit + views.dark;
}

interface DoorScene {
    readonly scene: DdgiScene;
    readonly door: Door;
    readonly view: TestView;
    readonly camera: FrameCamera;
    readonly leak: () => Promise<{ leak: number; lit: number; dark: number }>;
    setDoor(degrees: number): void;
}

/** The closed-door scene with the door open, converged at the batch. */
async function openAndConverged(device: GPUDevice, label: string, log: (line: string) => void): Promise<DoorScene> {
    const scene = await createDdgiScene(device, 'closed-door', {}, { ignoreLightRange: true });
    const door = doorOf(scene);
    const setDoor = (degrees: number) => scene.platform.scene.setTransforms(Uint32Array.of(door.handle), new Float32Array(door.at(degrees)));
    const bookmark = (name: string) => {
        const found = scene.manifest.cameras.find((camera) => camera.name === name);
        assert(found, `closed-door has no camera ${name}`);
        return found;
    };
    const darkView = bookmark('dark-room-leak-view');
    const litView = bookmark('lit-room');
    const camera = bookmarkCamera(darkView, WIDTH / HEIGHT);
    const view = await TestView.create(device, scene.platform);
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const leak = async () => {
        const dark = imageMean((await view.render(field, darkView, WIDTH, HEIGHT)).indirect);
        const lit = imageMean((await view.render(field, litView, WIDTH, HEIGHT)).indirect);
        return { leak: dark / lit, lit, dark };
    };
    // Open before any probe is updated: it marks nothing (every probe is new). The trees are built already.
    setDoor(OPEN_DEGREES);
    runCameraFrame(device, scene.platform, 0, camera);
    await scene.platform.rays.whenBuilt();
    await convergeField(device, scene.platform, scene.ddgi.settings, camera, async () => convergedOn(await leak()), { ...CONVERGE, settleUnits: UNITS }, `${label} (open)`, log);
    return { scene, door, view, camera, leak, setDoor };
}

async function convergeClosed(device: GPUDevice, test: DoorScene, label: string, log: (line: string) => void): Promise<number> {
    await convergeField(device, test.scene.platform, test.scene.ddgi.settings, test.camera, async () => convergedOn(await test.leak()), { ...CONVERGE, settleUnits: UNITS }, `${label} (closed)`, log);
    return (await test.leak()).leak;
}

async function swing(device: GPUDevice, log: (line: string) => void): Promise<Record<string, unknown>> {
    const test = await openAndConverged(device, 'swing', log);
    const open = await test.leak();
    log(`swing: open, converged: dark room ${(open.leak * 100).toFixed(2)}% of the lit room`);
    const during: number[] = [];
    for (let frame = 1; frame <= SWING_FRAMES; frame++) {
        test.setDoor(swingAngle(frame));
        runCameraFrame(device, test.scene.platform, UNITS, test.camera);
        during.push((await test.leak()).leak);
    }
    const after: number[] = [];
    for (let frame = 1; frame <= HOLD_FRAMES; frame++) {
        runCameraFrame(device, test.scene.platform, UNITS, test.camera);
        after.push((await test.leak()).leak);
    }
    // From the deadline on, every frame must be within the limit.
    const lateFrames = after.slice(CLOSE_DEADLINE_FRAMES - 1);
    const worstLate = Math.max(...lateFrames);
    const firstWithin = after.findIndex((leak) => leak <= LEAK_LIMIT);
    let lastAbove = -1;
    after.forEach((leak, index) => {
        if (leak > LEAK_LIMIT) lastAbove = index;
    });
    const settled = await convergeClosed(device, test, 'swing', log);
    const curve = [1, 2, 5, 10, 20, 30, 45, 60, 90, 120, 300, 600].filter((frame) => frame <= after.length).map((frame) => `${frame}: ${(after[frame - 1] * 100).toFixed(2)}%`).join(', ');
    log(`swing: after the door closed (frames at ${UNITS} probes): ${curve}; within ${LEAK_LIMIT * 100}% from frame ${lastAbove + 2} (first ${firstWithin + 1}); worst from frame ${CLOSE_DEADLINE_FRAMES} on ${(worstLate * 100).toFixed(3)}%; converged closed ${(settled * 100).toFixed(3)}%`);
    test.scene.platform.dispose();
    assert(worstLate <= LEAK_LIMIT, `the dark room's leak was ${(worstLate * 100).toFixed(2)}% at its worst from ${CLOSE_DEADLINE_FRAMES} frames after the door closed (limit ${LEAK_LIMIT * 100}%)`);
    return {
        units: UNITS,
        openLeak: open.leak,
        swingLeak: during,
        afterLeak: after,
        firstFrameWithin: firstWithin + 1,
        withinForGoodFrom: lastAbove + 2,
        worstFromDeadline: worstLate,
        settledClosedLeak: settled,
    };
}

/** Item 8: marks against true changes, for one way of closing the door (the file comment). */
async function marks(device: GPUDevice, how: 'step' | 'swing', log: (line: string) => void): Promise<MarkRecall> {
    const test = await openAndConverged(device, `marks ${how}`, log);
    const field = test.scene.ddgi.field!;
    const before = await probeSnapshot(device, field);
    if (how === 'step') {
        test.setDoor(0);
        runCameraFrame(device, test.scene.platform, 0, test.camera);
    } else {
        for (let frame = 1; frame <= SWING_FRAMES; frame++) {
            test.setDoor(swingAngle(frame));
            runCameraFrame(device, test.scene.platform, 0, test.camera);
        }
    }
    const marked = await probeSnapshot(device, field);
    await convergeClosed(device, test, `marks ${how}`, log);
    const after = await probeSnapshot(device, field);
    const spacing = field.grid.spacing;
    const margin = spacing.map((value) => NEAR_SET_SPACINGS * value);
    const boxes = [test.door.boxAt(OPEN_DEGREES), test.door.boxAt(0), ...(how === 'swing' ? [-60, -30].map((degrees) => test.door.boxAt(degrees)) : [])];
    const isNear = (probe: number) => boxes.some((box) => nearBox(before.position[probe], box.min, box.max, margin));
    const result = markRecall(marked, before, after, test.scene.ddgi.settings.changeTolerance, isNear);
    const missed = missedByDistance(result.missed, before.position, boxes, spacing);
    log(
        `marks (${how}): ${result.caught} of ${result.changed} probes that changed by over 5% were marked (recall ${(result.recall * 100).toFixed(1)}%), precision ${(result.precision * 100).toFixed(1)}% of ${result.marked}; ` +
            `near ${result.near.caught}/${result.near.changed}, far ${result.far.caught}/${result.far.changed} (far marked ${result.far.marked}); missed by distance from the door in spacings: ${JSON.stringify(missed)}`,
    );
    test.scene.platform.dispose();
    return { ...result, missed: result.missed.length, missedByDistance: missed } as unknown as MarkRecall;
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, unknown> = {};
        if (SECTIONS.has('marks')) {
            results.marksStep = await marks(device, 'step', log);
            results.marksSwing = await marks(device, 'swing', log);
        }
        if (SECTIONS.has('swing')) results.swing = await swing(device, log);
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
