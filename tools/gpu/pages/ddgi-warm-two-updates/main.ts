/**
 * T2.M1 item 5 (docs/design/m3-moving.md §5.2): no probe is used in shading
 * before it has had 2 blended updates. One noisy estimate is the pop SPEC
 * forbids, so a probe in state 3 (new, reset, or back on) stays out of
 * shading after its first blended update and becomes active after its
 * second.
 *
 * Shading uses active probes only (state 0, `ddgiSampleIrradiance`), so the
 * check is on states and counts, after every frame:
 * - **the rule:** every active probe has k ≥ 2 (k: its blended updates since
 *   it was new, reset or back on; history word 2);
 * - **from new** (two-room, warm-up off, every probe a frame): after frame 1
 *   no probe is active and the test view's GI is exactly 0; every probe that
 *   isn't off is in state 3 with k = 1; after frame 2 they are active;
 * - **back on** (a crate set down over 24 grid points switches the probes
 *   relocation can't move out of it off; moved
 *   away, they come back on): at a small budget (FREED_UNITS a frame) each
 *   freed probe goes through state 3 for exactly 2 blended updates, and
 *   because it keeps the updates the move promised it, all of them are
 *   active again within FREED_FRAMES frames.
 *
 * Both reductions (subgroups, and the declared fallback) run the same steps.
 */

import { assert, runGpuTest } from '../../client.ts';
import { PROBE_STATE } from '../../../../packages/ddgi/src/layout.ts';
import type { PhaseBReduction } from '../../../../packages/ddgi/src/wgsl/blend.ts';
import { createDdgiScene, imageMean, releaseDeviceOnPageHide, runFrame, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';
import { probeSnapshot, type ProbeSnapshot } from '../ddgi-shared/moving.ts';
import { bookmarkCamera, runCameraFrame } from '../ddgi-shared/response.ts';

/**
 * Probes a frame while freed probes warm: about a tenth of the volume, so
 * the priority they keep is what gets them served (a round robin alone would
 * take about 8 frames for each of their 3 updates).
 */
const FREED_UNITS = 128;
/**
 * All freed probes are active again within this many frames: a fixed-ray
 * trace and 2 blended updates, each at the front of the list (off probes in
 * a move's near set go first, FREED_PROBE_URGENCY_SHARE). On the RTX 3060
 * the last is active in frame 3, the least it can be.
 */
const FREED_FRAMES = 6;
/** Frames with every probe a frame for the crate's probes to switch off and the field to settle. */
const SETTLE_FRAMES = 12;
const WIDTH = 160;
const HEIGHT = 90;
/** A crate around 24 of two-room's grid points (x 1.6-2.6, y 0.4-0.9, z 1.0-2.5); relocation frees those near its faces. */
const CRATE_CENTRE: [number, number, number] = [2.1, 0.65, 1.75];
const CRATE_SIZE: [number, number, number] = [1.6, 1.1, 1.6];
/** At least this many probes must be switched off by it for the check to mean something. */
const MIN_SWITCHED_OFF = 4;
/** Moved 1.5 m along the room, off the probes. */
const CRATE_AWAY: [number, number, number] = [2.1, 0.65, 3.25];

function boxTransform(centre: readonly number[], size: readonly number[]): number[] {
    return [size[0], 0, 0, 0, 0, size[1], 0, 0, 0, 0, size[2], 0, centre[0], centre[1], centre[2], 1];
}

/** Every active probe has had at least 2 blended updates; returns the probes that break it. */
function brokenRule(snapshot: ProbeSnapshot): number[] {
    const broken: number[] = [];
    snapshot.state.forEach((state, probe) => {
        if (state === PROBE_STATE.active && snapshot.updates[probe] < 2) broken.push(probe);
    });
    return broken;
}

async function checkReduction(device: GPUDevice, reduction: PhaseBReduction, log: (line: string) => void): Promise<Record<string, unknown>> {
    const scene: DdgiScene = await createDdgiScene(device, 'two-room', {}, { technique: { reduction } });
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const probeCount = field.grid.probeCount;
    const view = await TestView.create(device, scene.platform);
    const camera = scene.manifest.cameras[0];
    const snapshot = async (label: string): Promise<ProbeSnapshot> => {
        await device.queue.onSubmittedWorkDone();
        const taken = await probeSnapshot(device, field);
        const broken = brokenRule(taken);
        assert(broken.length === 0, `${reduction}, ${label}: ${broken.length} active probes have fewer than 2 blended updates (first: probe ${broken[0]}, k = ${taken.updates[broken[0]]})`);
        return taken;
    };

    // From new: one blended update each, then two.
    runFrame(device, scene.platform, probeCount);
    const afterOne = await snapshot('after frame 1');
    const active1 = afterOne.state.filter((state) => state === PROBE_STATE.active).length;
    const warming1 = afterOne.state.filter((state, probe) => state === PROBE_STATE.notUpdatedYet && afterOne.updates[probe] === 1).length;
    const off1 = afterOne.state.filter((state) => state === PROBE_STATE.offInsideGeometry || state === PROBE_STATE.offNoSurfaceNearby).length;
    const viewAfterOne = imageMean((await view.render(field, camera, WIDTH, HEIGHT)).indirect);
    assert(active1 === 0, `${reduction}: ${active1} probes are active after one blended update`);
    assert(warming1 + off1 === probeCount, `${reduction}: after frame 1, ${probeCount - warming1 - off1} probes are neither warming (state 3, k = 1) nor off`);
    assert(viewAfterOne === 0, `${reduction}: the view's GI is ${viewAfterOne} after one update; no probe should be used yet`);
    runFrame(device, scene.platform, probeCount);
    const afterTwo = await snapshot('after frame 2');
    const active2 = afterTwo.state.filter((state) => state === PROBE_STATE.active).length;
    const viewAfterTwo = imageMean((await view.render(field, camera, WIDTH, HEIGHT)).indirect);
    assert(active2 > 0.8 * warming1, `${reduction}: only ${active2} of ${warming1} warming probes are active after their second update`);
    assert(viewAfterTwo > 0, `${reduction}: the view is still black after two updates`);
    log(`${reduction}: after 1 update ${active1} active, ${warming1} warming, ${off1} off, view ${viewAfterOne}; after 2 updates ${active2} active, view ${viewAfterTwo.toFixed(4)}`);

    // Back on: a crate over 12 probes switches them off; moved away, they come back.
    const sceneApi = scene.platform.scene;
    // The scene's own unit box (the procedural builder's, as its crates use), so its faces point outwards.
    const boxMesh = scene.data.meshes[scene.data.instances.find((instance) => instance.name === 'crate-red')!.mesh];
    const mesh = sceneApi.createMesh({ positions: boxMesh.positions, indices: boxMesh.indices, label: 'crate' });
    const crate = sceneApi.createInstance({ mesh, materials: [scene.materials[0]], transform: boxTransform(CRATE_CENTRE, CRATE_SIZE), label: 'crate' });
    runFrame(device, scene.platform, 0);
    await scene.platform.rays.whenBuilt();
    for (let frame = 0; frame < SETTLE_FRAMES; frame++) runFrame(device, scene.platform, probeCount);
    const covered = await snapshot('with the crate down');
    // The probes the crate switched off: inside geometry now, active before it came.
    const switchedOff = covered.state.map((state, probe) => ({ state, probe })).filter(({ state, probe }) => state === PROBE_STATE.offInsideGeometry && afterTwo.state[probe] === PROBE_STATE.active).map(({ probe }) => probe);
    assert(switchedOff.length >= MIN_SWITCHED_OFF, `${reduction}: the crate switched only ${switchedOff.length} probes off`);

    sceneApi.setTransforms(Uint32Array.of(crate), new Float32Array(boxTransform(CRATE_AWAY, CRATE_SIZE)));
    // The scheduler orders by nearness to the camera: the lit-room view, as a viewer would look.
    const frameCamera = bookmarkCamera(camera, WIDTH / HEIGHT);
    // Per freed probe: the frame it was first seen in state 3 and the frame it became active.
    const warmingFrom = new Map<number, number>();
    const activeFrom = new Map<number, number>();
    for (let frame = 1; frame <= FREED_FRAMES + 30 && activeFrom.size < switchedOff.length; frame++) {
        runCameraFrame(device, scene.platform, FREED_UNITS, frameCamera);
        const now = await snapshot(`frame ${frame} after the crate moved`);
        for (const probe of switchedOff) {
            if (now.state[probe] === PROBE_STATE.notUpdatedYet && !warmingFrom.has(probe)) warmingFrom.set(probe, frame);
            if (now.state[probe] === PROBE_STATE.active && !activeFrom.has(probe)) {
                activeFrom.set(probe, frame);
                assert(now.updates[probe] === 2, `${reduction}: freed probe ${probe} became active with k = ${now.updates[probe]}, not 2`);
            }
        }
    }
    const slowest = Math.max(...activeFrom.values());
    assert(activeFrom.size === switchedOff.length, `${reduction}: ${switchedOff.length - activeFrom.size} freed probes never became active`);
    assert(slowest <= FREED_FRAMES, `${reduction}: the last freed probe became active in frame ${slowest}, after ${FREED_FRAMES}`);
    // Each went through state 3 for its 2 blended updates: seen warming before it was seen active.
    for (const probe of switchedOff) assert((warmingFrom.get(probe) ?? Infinity) < activeFrom.get(probe)!, `${reduction}: freed probe ${probe} skipped its warming`);
    log(`${reduction}: ${switchedOff.length} probes under the crate came back on; active again in frames ${Math.min(...activeFrom.values())}-${slowest} at ${FREED_UNITS} probes a frame, each after exactly 2 blended updates`);
    scene.platform.dispose();
    return { activeAfterOne: active1, warmingAfterOne: warming1, offAfterOne: off1, viewAfterOne, activeAfterTwo: active2, viewAfterTwo, freedProbes: switchedOff.length, lastFreedActiveFrame: slowest };
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const results: Record<string, unknown> = {};
        const reductions: PhaseBReduction[] = device.features.has('subgroups') ? ['subgroups', 'workgroup'] : ['workgroup'];
        for (const reduction of reductions) results[reduction] = await checkReduction(device, reduction, log);
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
