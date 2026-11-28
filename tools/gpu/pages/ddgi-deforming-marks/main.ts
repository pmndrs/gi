/**
 * T3.4, deforming movers (docs/design/m3-moving.md §5.1): a deforming mesh's
 * poses mark probes the way a move does, from the region the platform logs
 * for each pose (T3.1): its `deformBounds` box in world space, with how far
 * its vertices travelled (`moved`) and its instance.
 *
 * The scene: two-room, with a 1 m sheet created `deforming: true` hanging
 * in the middle of room A, its field converged flat (every probe a frame).
 *
 * 1. **First pose** (a wave), in a frame that updates no probe, so each
 *    probe's `pendingChange` is what the pose marked:
 *    - every probe within the near set (1.5 spacings of the deform box)
 *      that has been updated, off ones included, is marked with exactly
 *      r = min(1, moved / spacing), moved being the largest vertex travel
 *      (worked out here from the poses);
 *    - no probe farther than the far set's reach is marked;
 *    - the far set (probes that see the box) is reported.
 * 2. Frames run until the pose's tree is in and every probe is updated, so
 *    the probes' mover masks hold the sheet's bit (it became a hot mover
 *    with its first pose).
 * 3. **Second pose**, again in a frame with no updates: every active probe
 *    within the far reach whose mask has the sheet's bit is marked (the
 *    "probes that saw it last update" rule), so a probe the geometric test
 *    misses is still caught.
 *
 * Prototype level: until T3.3's refit, a pose reaches rays through a tree
 * rebuild (T3.1), a few frames after its marks. That gap is reported, not
 * judged (the marks wait in the history for each probe's next update).
 */

import { assert, runGpuTest } from '../../client.ts';
import { tessellatedQuad } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { PROBE_STATE } from '../../../../packages/core/src/techniques/ddgi/layout.ts';
import { FAR_SET_SPACINGS, NEAR_SET_SPACINGS } from '../../../../packages/core/src/techniques/ddgi/mark-table.ts';
import { createDdgiScene, releaseDeviceOnPageHide, runFrame, runFrames } from '../ddgi-shared/harness.ts';
import { nearBox, probeSnapshot } from '../ddgi-shared/moving.ts';

type Vec3 = [number, number, number];

const SCENE = 'two-room';
/** The sheet: CELLS x CELLS squares from -HALF to HALF in x and y, hanging upright (its wave goes along z). */
const CELLS = 16;
const HALF = 0.5;
const WAVE_HEIGHT = 0.3;
/** Where the sheet hangs: the middle of room A. */
const SHEET_CENTRE: Vec3 = [2.0, 1.4, 2.0];
const CONVERGE_FRAMES = 120;
/** Frames between the poses: the pose's tree is placed, and every probe is updated a few times. */
const BETWEEN_POSES_FRAMES = 12;
/** The marked change is a float sum on the GPU (one mark here): equal to this much. */
const MARK_TOLERANCE = 1e-5;

/** The sheet at wave phase `phase`: z = height x sin(2 pi x + phase) x cos(pi y). */
function sheetPose(rest: Float32Array, phase: number): Float32Array {
    const pose = new Float32Array(rest.length);
    for (let at = 0; at < rest.length; at += 3) {
        const x = rest[at];
        const y = rest[at + 1];
        pose[at] = x;
        pose[at + 1] = y;
        pose[at + 2] = Math.fround(WAVE_HEIGHT * Math.sin(2 * Math.PI * x + phase) * Math.cos(Math.PI * y));
    }
    return pose;
}

/** The largest distance any vertex travelled between two poses (what the platform logs as `moved`, at scale 1). */
function largestTravel(before: Float32Array, after: Float32Array): number {
    let largest = 0;
    for (let at = 0; at < before.length; at += 3) {
        largest = Math.max(largest, Math.hypot(after[at] - before[at], after[at + 1] - before[at + 1], after[at + 2] - before[at + 2]));
    }
    return largest;
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
        const spacing = field.grid.spacing;
        const smallestSpacing = Math.min(...spacing);

        const sheet = tessellatedQuad(CELLS, HALF);
        const rest = Float32Array.from(sheet.positions);
        const deformBounds = { min: [-HALF, -HALF, -WAVE_HEIGHT] as Vec3, max: [HALF, HALF, WAVE_HEIGHT] as Vec3 };
        const sceneApi = scene.platform.scene;
        const mesh = sceneApi.createMesh({ positions: rest, indices: sheet.indices!, deforming: true, deformBounds, label: 'waving sheet' });
        const transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...SHEET_CENTRE, 1];
        sceneApi.createInstance({ mesh, materials: [scene.materials[0]], transform, label: 'waving sheet' });
        runFrame(device, scene.platform, 0);
        await scene.platform.rays.whenBuilt();
        await runFrames(device, scene.platform, CONVERGE_FRAMES, probeCount);
        const box = { min: deformBounds.min.map((value, axis) => value + SHEET_CENTRE[axis]), max: deformBounds.max.map((value, axis) => value + SHEET_CENTRE[axis]) };
        const nearMargin = spacing.map((value) => NEAR_SET_SPACINGS * value);
        // The kernel's far reach is FAR_SET_SPACINGS from the box, where relocation put the probe; 'beyond' leaves half a spacing more.
        const reachMargin = spacing.map((value) => FAR_SET_SPACINGS * value);
        const farMargin = spacing.map((value) => FAR_SET_SPACINGS * value + 0.5 * value);
        assert(!scene.platform.rays.building, 'trees are still building before the first pose');

        // 1. The first pose, marked in a frame that updates nothing.
        const firstPose = sheetPose(rest, Math.PI / 2);
        const firstMoved = largestTravel(rest, firstPose);
        const expected = Math.min(1, firstMoved / smallestSpacing);
        sceneApi.setDeformedPositions(mesh, firstPose);
        runFrame(device, scene.platform, 0);
        const marked = await probeSnapshot(device, field);
        let near = 0;
        let nearWrong = 0;
        let farMarked = 0;
        let beyondMarked = 0;
        for (let probe = 0; probe < probeCount; probe++) {
            const position = marked.position[probe];
            const change = marked.pendingChange[probe];
            if (nearBox(position, box.min, box.max, nearMargin)) {
                if (marked.state[probe] === PROBE_STATE.notUpdatedYet) continue;
                near++;
                if (Math.abs(change - expected) > MARK_TOLERANCE) nearWrong++;
            } else if (!nearBox(position, box.min, box.max, farMargin)) {
                if (change > 0) beyondMarked++;
            } else if (change > 0) {
                farMarked++;
            }
        }
        log(`first pose: vertices travelled up to ${firstMoved.toFixed(3)} m, so r = ${expected.toFixed(3)} near; ${near} near probes (${nearWrong} not marked so), ${farMarked} far probes marked, ${beyondMarked} beyond the far reach marked`);
        assert(near > 0, 'no probe lies near the sheet');
        assert(nearWrong === 0, `${nearWrong} of ${near} near probes weren't marked with r = ${expected.toFixed(4)}`);
        assert(beyondMarked === 0, `${beyondMarked} probes beyond the far reach were marked`);
        assert(farMarked > 0, 'the pose marked no far probe');

        // 2. The pose's tree goes in, and every probe is updated: their masks now hold the sheet's bit.
        let framesToTree = 0;
        while (scene.platform.rays.building) {
            runFrame(device, scene.platform, probeCount);
            framesToTree++;
            await device.queue.onSubmittedWorkDone();
            if (framesToTree % 4 === 0) await scene.platform.rays.whenBuilt();
        }
        await runFrames(device, scene.platform, BETWEEN_POSES_FRAMES, probeCount);
        const sawIt = await probeSnapshot(device, field);
        const withBit = sawIt.moverMask.map((mask, probe) => ({ mask, probe })).filter(({ mask, probe }) => mask !== 0 && sawIt.state[probe] === PROBE_STATE.active && nearBox(sawIt.position[probe], box.min, box.max, reachMargin));

        // 3. The second pose: every probe that saw the sheet at its last update is marked.
        const secondPose = sheetPose(rest, Math.PI);
        sceneApi.setDeformedPositions(mesh, secondPose);
        runFrame(device, scene.platform, 0);
        const marked2 = await probeSnapshot(device, field);
        const missed = withBit.filter(({ probe }) => !(marked2.pendingChange[probe] > 0));
        const markedTotal = marked2.pendingChange.filter((change) => change > 0).length;
        log(
            `the pose's tree was placed after ${framesToTree} frame(s) of rays on the old tree; second pose: ${withBit.length} active probes within reach saw the sheet at their last update, ` +
                `${withBit.length - missed.length} of them marked (${markedTotal} probes marked in all)`,
        );
        assert(withBit.length > 0, "no probe's mover mask holds the sheet's bit after its first pose");
        const gap = (probe: number) => Math.max(...[0, 1, 2].map((axis) => Math.max(box.min[axis] - sawIt.position[probe][axis], 0, sawIt.position[probe][axis] - box.max[axis]) / spacing[axis]));
        const missedDetail = missed.slice(0, 6).map(({ probe, mask }) => `probe ${probe} (${gap(probe).toFixed(2)} spacings out, mask ${mask}, flags ${marked2.flags[probe]})`);
        assert(missed.length === 0, `${missed.length} probes that saw the sheet at their last update weren't marked by its second pose: ${missedDetail.join('; ')}`);
        scene.platform.dispose();
        return { firstMoved, nearProbes: near, farMarked, framesToTree, sawTheSheet: withBit.length, secondPoseMarked: markedTotal };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
