/**
 * T2.4b check 7, the feedback rule (docs/design/m2-lights.md §4.6): a probe
 * whose own update lands further from its stored value than its noise
 * allows asks for one more update, and so do its neighbours. It only
 * schedules; it never changes how an update blends.
 *
 * - **Still scenes** (Cornell, two-room; the RTX's calibrated batches at
 *   2 ms, T1.9b, as the still pages use): the technique's warm-up, then
 *   FRAMES frames. The rule must fire on under 0.1% of the probe updates of
 *   those frames (the census's "feedback requests (total)" over "probes
 *   blended (total)"), and the test view's mean with feedback on must be
 *   within 0.2% of the same run with it off.
 * - **An unreported change** (two-room at the design's test budget of 64
 *   probes a frame, so the round robin takes 17 frames a pass): after the
 *   warm-up and SETTLE_FRAMES frames, the blue crate in the dark room starts
 *   to glow through `updateMaterial`. DDGI's change estimate knows nothing
 *   of materials, so only feedback can find it. For the probes nearest the
 *   crate (grid points within one spacing of its box), each probe's first update after
 *   the change and its next one (from the history's scheduled-frame stamp,
 *   read every frame): with feedback on, every one of them whose light the
 *   glow changed by more than 5% (its stored mean, by the end of the watch)
 *   must be revisited within 0.5 s (30 frames) of the change, and they must
 *   get more updates than without it.
 *
 * DDGI's clock advances 1/60 s a frame (a 60 Hz host).
 *
 * `pnpm test:gpu --filter ddgi-feedback --gpu nvidia` (or intel).
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, imageMean, readProbeData, releaseDeviceOnPageHide, runFrames, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';
import { probeCoordinates, readHistory } from '../ddgi-shared/response.ts';

const env = import.meta.env as Record<string, unknown>;

/** Frames measured on the still scenes: the done-when's 600. */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 600;
const STILL_CASES: readonly { readonly scene: string; readonly units: number }[] = [
    { scene: 'cornell-box', units: 1574 },
    { scene: 'two-room', units: 707 },
];
/** Check 7's bars. */
const MAX_FIRED_SHARE = 0.001;
const MAX_MEAN_CHANGE = 0.002;
const WIDTH = 320;
const HEIGHT = 180;

/** The unreported change (the file comment). */
const CHANGE_SCENE = 'two-room';
const CHANGE_UNITS = 64;
const CHANGE_MATERIAL = 'crate-blue';
/** The crate's box in room B (scenes/src/procedural/two-rooms.ts, 'crate-blue-in-room-b'; rotated 15 degrees, so its box is a little larger). */
const CRATE_CENTRE: readonly [number, number, number] = [6.4, 0.4, 2.6];
const CRATE_HALF_EXTENT = 0.55;
const GLOW: readonly [number, number, number] = [3, 3, 3];
/** Frames of the default rule before the change, so every probe's history and noise are settled at 64 a frame. */
const SETTLE_FRAMES = 1200;
const RESPONSE_FRAMES = 30;
/** A probe's light changed when its stored mean moved by more than this (the 5% tolerance) by the end of the watch. */
const CHANGED = 0.05;
const WATCH_FRAMES = 60;

/** The technique's own warm-up (16 passes), run back to back as volume.start does. */
async function warmUp(scene: DdgiScene): Promise<void> {
    const device = scene.platform.device;
    let steps = 0;
    while (scene.ddgi.warmingUp) {
        await runFrames(device, scene.platform, 1, 0);
        steps++;
        assert(steps < 10_000, 'the warm-up did not end');
    }
}

async function stillRun(device: GPUDevice, sceneId: string, units: number, feedback: boolean) {
    const scene = await createDdgiScene(device, sceneId, { feedback, warmUpPasses: 16 }, { gpuTiming: true });
    await warmUp(scene);
    // Let the census of the warm-up's last frames land, then take the totals the measured frames start from.
    await runFrames(device, scene.platform, 4, 0);
    const before = { ...scene.ddgi.counters };
    await runFrames(device, scene.platform, FRAMES, units);
    await runFrames(device, scene.platform, 4, 0);
    const after = { ...scene.ddgi.counters };
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const view = await TestView.create(device, scene.platform);
    const viewMean = imageMean((await view.render(field, scene.manifest.cameras[0], WIDTH, HEIGHT)).indirect);
    scene.platform.dispose();
    const fired = (after['feedback requests (total)'] ?? 0) - (before['feedback requests (total)'] ?? 0);
    const blended = (after['probes blended (total)'] ?? 0) - (before['probes blended (total)'] ?? 0);
    return { fired, blended, firedShare: blended > 0 ? fired / blended : 0, viewMean };
}

/** The unreported change (the file comment): per nearest probe, the frames it was updated in after the change. */
async function changeRun(device: GPUDevice, feedback: boolean, log: (line: string) => void) {
    const scene = await createDdgiScene(device, CHANGE_SCENE, { feedback, warmUpPasses: 16 }, { gpuTiming: true });
    const field = scene.ddgi.field;
    assert(field, 'no field');
    await warmUp(scene);
    await runFrames(device, scene.platform, SETTLE_FRAMES, CHANGE_UNITS);
    const material = scene.materials[scene.data.materials.findIndex((entry) => entry.name === CHANGE_MATERIAL)];
    assert(material, `${CHANGE_SCENE} has no material ${CHANGE_MATERIAL}`);
    // The nearest probes: active, grid point within one spacing of the crate's box (by distance).
    const grid = field.grid;
    const probeData = await readProbeData(device, field);
    const nearest: number[] = [];
    for (let probe = 0; probe < grid.probeCount; probe++) {
        const coordinates = probeCoordinates(probe, grid.counts);
        const position = coordinates.map((value, axis) => grid.origin[axis] + value * grid.spacing[axis]);
        const outside = position.map((value, axis) => Math.max(0, Math.abs(value - CRATE_CENTRE[axis]) - CRATE_HALF_EXTENT));
        const near = Math.hypot(...outside) <= Math.min(...grid.spacing);
        if (near && probeData[probe].state === 0) nearest.push(probe);
    }
    const before = await readHistory(device, field);
    const startFrame = scene.platform.stats.frame;
    scene.platform.scene.updateMaterial(material, { emissive: [...GLOW] });
    const updatesOf = new Map<number, number[]>(nearest.map((probe) => [probe, []]));
    let fired = 0;
    for (let frame = 1; frame <= WATCH_FRAMES; frame++) {
        await runFrames(device, scene.platform, 1, CHANGE_UNITS);
        const history = await readHistory(device, field);
        const stamp = (startFrame + frame) & 0xffffff;
        for (const probe of nearest) if (history[probe].scheduledFrame === stamp) updatesOf.get(probe)?.push(frame);
        fired = Math.max(fired, scene.ddgi.counters['feedback requests (total)'] ?? 0);
    }
    const after = await readHistory(device, field);
    scene.platform.dispose();
    // How much each nearest probe's light changed by the end of the watch (its stored mean, the largest channel).
    const changeOf = (probe: number) => Math.max(...[0, 1, 2].map((channel) => Math.abs(after[probe].mean[channel] - before[probe].mean[channel]) / Math.max(before[probe].mean[channel], 1e-6)));
    // Revisited: a second update after the change, within RESPONSE_FRAMES of it.
    const revisitFrames = nearest.map((probe) => updatesOf.get(probe)?.[1] ?? null);
    const revisitedInTime = revisitFrames.filter((frame) => frame !== null && frame <= RESPONSE_FRAMES).length;
    // The probes whose light the glow really changed (by more than the 5% tolerance): those must be revisited.
    const changed = nearest.filter((probe) => changeOf(probe) > CHANGED);
    const changedRevisited = changed.filter((probe) => (updatesOf.get(probe)?.[1] ?? Infinity) <= RESPONSE_FRAMES).length;
    const missed = nearest
        .filter((probe) => !((updatesOf.get(probe)?.[1] ?? Infinity) <= RESPONSE_FRAMES))
        .map((probe) => ({ probe, at: probeCoordinates(probe, grid.counts), change: changeOf(probe), updates: updatesOf.get(probe) ?? [] }));
    const known = revisitFrames.filter((frame): frame is number => frame !== null).sort((a, b) => a - b);
    const medianRevisit = known.length > 0 ? known[Math.floor(known.length / 2)] : null;
    const updatesInTime = nearest.reduce((sum, probe) => sum + (updatesOf.get(probe)?.filter((frame) => frame <= RESPONSE_FRAMES).length ?? 0), 0);
    log(
        `unreported change, feedback ${feedback ? 'on' : 'off'}: ${nearest.length} nearest probes; revisited within ${RESPONSE_FRAMES} frames: ${revisitedInTime}; ` +
            `of the ${changed.length} whose light changed by more than ${CHANGED * 100}%: ${changedRevisited}; ` +
            `median revisit at frame ${medianRevisit ?? 'never'}; their updates in ${RESPONSE_FRAMES} frames: ${updatesInTime}; ` +
            `not revisited: ${missed.map((entry) => `probe ${entry.at.join(',')} changed ${(entry.change * 100).toFixed(1)}%, updated at frames ${entry.updates.join(' ') || 'none'}`).join('; ') || 'none'}`,
    );
    return { nearest: nearest.length, revisitedInTime, changed: changed.length, changedRevisited, medianRevisit, updatesInTime, missed, revisitFrames };
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, unknown> = { frames: FRAMES };
        const failures: string[] = [];

        for (const { scene, units } of STILL_CASES) {
            const on = await stillRun(device, scene, units, true);
            const off = await stillRun(device, scene, units, false);
            const meanChange = on.viewMean / off.viewMean - 1;
            results[scene] = { units, on, off, meanChange };
            log(
                `${scene} (${units} a frame, ${FRAMES} frames after the warm-up): feedback fired on ${on.fired} of ${on.blended} updates (${(on.firedShare * 100).toFixed(3)}%); ` +
                    `view mean on ${on.viewMean.toPrecision(6)}, off ${off.viewMean.toPrecision(6)} (${(meanChange * 100).toFixed(3)}%)`,
            );
            if (on.firedShare >= MAX_FIRED_SHARE) failures.push(`${scene}: feedback fired on ${(on.firedShare * 100).toFixed(3)}% of updates (bar ${MAX_FIRED_SHARE * 100}%)`);
            if (Math.abs(meanChange) > MAX_MEAN_CHANGE) failures.push(`${scene}: the view's mean moved ${(meanChange * 100).toFixed(3)}% with feedback on (bar ${MAX_MEAN_CHANGE * 100}%)`);
        }

        const on = await changeRun(device, true, log);
        const off = await changeRun(device, false, log);
        results.unreportedChange = { units: CHANGE_UNITS, responseFrames: RESPONSE_FRAMES, on, off };
        if (on.changedRevisited < on.changed) failures.push(`unreported change: ${on.changed - on.changedRevisited} of the ${on.changed} nearest probes whose light changed were not revisited within ${RESPONSE_FRAMES} frames`);
        if (!(on.updatesInTime > off.updatesInTime)) failures.push(`unreported change: feedback gave the nearest probes no more updates (${on.updatesInTime} against ${off.updatesInTime})`);

        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
