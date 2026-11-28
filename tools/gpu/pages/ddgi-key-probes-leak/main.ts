/**
 * T2.4c check 3: no light crosses a wall through propagation
 * (docs/design/m2-lights.md §4.5, "The visibility test, from both ends").
 *
 * Two-room at a test budget of UNITS probes a frame, so the volume takes the
 * key-probe path (1,071 probes × 8 updates / 64 > 120 frames) and
 * propagation does the work, with fine probes held back from their own
 * updates (`holdFineProbes`), so only propagation moves them. The scheduler
 * orders by the `lit-room` camera: the player is in the lit room, and the
 * dark room's fine probes wait for their keys. The lamp, the scene's only
 * light, starts off; the histories settle at the batch; then it is switched
 * on.
 *
 * Each frame for STEP_FRAMES frames, the dark room's view
 * (`dark-room-leak-view`, GI only) against a field converged with the lamp
 * on from the start: the dark room's light beyond that settled answer (which
 * has the real light through the doorway), as a share of the lit room's
 * settled light (the `lit-room` view), the leak as DECISIONS 2026-11-04
 * defines it against its own reference. It must stay at or under SPEC's 1%
 * in every frame. And per probe, because held, the dark room's fine probes
 * stay under their settled light for a while, which could hide a leak in the
 * view's mean: the dark room's fine probes' light beyond their settled
 * values, over the lit room's probes' light, also at most 1%. Reported beside
 * it: the same run with key probes off (far probes owe 2, no propagation).
 *
 * `pnpm test:gpu --filter ddgi-key-probes-leak --gpu nvidia`.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { FrameCamera } from '../../../../packages/core/src/platform/contract/frame.ts';
import { isKeyProbe } from '../../../../packages/ddgi/src/layout.ts';
import { createDdgiScene, imageMean, loadManifest, releaseDeviceOnPageHide, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';
import { bookmarkCamera, convergeField, probeCoordinates, readHistory, runCameraFrame, runCameraFrames, type ConvergeOptions } from '../ddgi-shared/response.ts';

const UNITS = 64;
const WIDTH = 320;
const HEIGHT = 180;
/** SPEC's leak bar, as a share of the lit room's light. */
const LEAK_LIMIT = 0.01;
const STEP_FRAMES = 240;
/** The histories settle at the batch with the lamp off before the step. */
const SETTLE_FRAMES = 600;
/** The wall between the rooms is at x = 4.1 m (the volume's centre); probes this far past it count as the dark room's. */
const WALL_X = 4.1;
const ROOM_MARGIN = 0.2;

const CONVERGE: ConvergeOptions = {
    fastFrames: 96,
    fastHysteresis: 0.5,
    blockFrames: 32,
    convergedTolerance: 0.001,
    maxBlocks: 24,
    settleFrames: 0,
    convergeUnits: 4096,
    settleUnits: UNITS,
};

type Vec3 = [number, number, number];

interface TwoRoom {
    readonly scene: DdgiScene;
    readonly view: TestView;
    readonly camera: FrameCamera;
    readonly litMean: () => Promise<number>;
    readonly darkMean: () => Promise<number>;
    readonly probeMeans: () => Promise<Vec3[]>;
}

async function startTwoRoom(device: GPUDevice, settings: { keyProbes?: boolean; holdFineProbes?: boolean }, lampIntensity: number): Promise<TwoRoom> {
    const scene = await createDdgiScene(device, 'two-room', settings, { gpuTiming: true });
    const cameras = scene.manifest.cameras;
    const lit = cameras.find((camera) => camera.name === 'lit-room');
    const dark = cameras.find((camera) => camera.name === 'dark-room-leak-view');
    assert(lit && dark, 'two-room has no lit-room or dark-room-leak-view camera');
    const camera = bookmarkCamera(lit, WIDTH / HEIGHT);
    scene.platform.scene.updateLight(scene.lights[0], { intensity: lampIntensity });
    // Before any probe is updated: it marks nothing (every probe is still new).
    runCameraFrame(device, scene.platform, 0, camera);
    const view = await TestView.create(device, scene.platform);
    const field = scene.ddgi.field;
    assert(field, 'no field');
    const litMean = async () => imageMean((await view.render(field, lit, WIDTH, HEIGHT)).indirect);
    const darkMean = async () => imageMean((await view.render(field, dark, WIDTH, HEIGHT)).indirect);
    const probeMeans = async () => (await readHistory(device, field)).map((record) => record.mean);
    return { scene, view, camera, litMean, darkMean, probeMeans };
}

const luminance = (rgb: Vec3) => (rgb[0] + rgb[1] + rgb[2]) / 3;

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const lampIntensity = loadManifest('two-room').lights[0].intensity;

        // The settled answer with the lamp on, converged from the start.
        const reference = await startTwoRoom(device, {}, lampIntensity);
        await convergeField(device, reference.scene.platform, reference.scene.ddgi.settings, reference.camera, reference.litMean, CONVERGE, 'two-room, the lamp on', log);
        const litSettled = await reference.litMean();
        const darkSettled = await reference.darkMean();
        const settledProbes = await reference.probeMeans();
        const grid = reference.scene.ddgi.field!.grid;
        reference.scene.platform.dispose();
        const probeX = (probe: number) => grid.origin[0] + probeCoordinates(probe, grid.counts)[0] * grid.spacing[0];
        const darkRoom = (probe: number) => probeX(probe) > WALL_X + ROOM_MARGIN;
        const litRoom = (probe: number) => probeX(probe) < WALL_X - ROOM_MARGIN;
        const litProbeLight = settledProbes.reduce((sum, mean, probe) => sum + (litRoom(probe) ? luminance(mean) : 0), 0);

        const results: Record<string, unknown> = { units: UNITS, litSettled, darkSettled, leakLimit: LEAK_LIMIT };
        const failures: string[] = [];
        for (const path of [
            { name: 'key probes, fine probes held', settings: { keyProbes: true, holdFineProbes: true } },
            { name: 'far probes owe 2', settings: { keyProbes: false } },
        ]) {
            const test = await startTwoRoom(device, path.settings, 0);
            await runCameraFrames(device, test.scene.platform, SETTLE_FRAMES, UNITS, test.camera);
            test.scene.platform.scene.updateLight(test.scene.lights[0], { intensity: lampIntensity });
            let worstView = Number.NEGATIVE_INFINITY;
            let worstViewFrame = 0;
            let worstProbes = Number.NEGATIVE_INFINITY;
            let worstProbesFrame = 0;
            const curve: Array<{ frame: number; viewLeak: number; probeLeak: number }> = [];
            for (let frame = 1; frame <= STEP_FRAMES; frame++) {
                runCameraFrame(device, test.scene.platform, UNITS, test.camera);
                const viewLeak = ((await test.darkMean()) - darkSettled) / litSettled;
                if (viewLeak > worstView) {
                    worstView = viewLeak;
                    worstViewFrame = frame;
                }
                if (frame % 10 === 0 || frame <= 10) {
                    const means = await test.probeMeans();
                    let beyond = 0;
                    for (let probe = 0; probe < means.length; probe++) {
                        if (darkRoom(probe) && !isKeyProbe(probeCoordinates(probe, grid.counts), grid.counts)) beyond += Math.max(0, luminance(means[probe]) - luminance(settledProbes[probe]));
                    }
                    const probeLeak = beyond / litProbeLight;
                    if (probeLeak > worstProbes) {
                        worstProbes = probeLeak;
                        worstProbesFrame = frame;
                    }
                    curve.push({ frame, viewLeak, probeLeak });
                }
            }
            const counters = test.scene.ddgi.counters;
            test.scene.platform.dispose();
            results[path.name] = { worstViewLeak: worstView, worstViewFrame, worstProbeLeak: worstProbes, worstProbeFrame: worstProbesFrame, curve, counters };
            log(
                `${path.name}: the dark room's light beyond its settled value, as a share of the lit room's: at most ${(worstView * 100).toFixed(3)}% (frame ${worstViewFrame}, the view); ` +
                    `its fine probes beyond their settled light at most ${(worstProbes * 100).toFixed(3)}% of the lit room's probes' light (frame ${worstProbesFrame}); probes propagated (total) ${counters['probes propagated (total)'] ?? 'n/a'}\n` +
                    `    by frame: ${curve.map((entry) => `${entry.frame}: ${(entry.viewLeak * 100).toFixed(2)}%`).join(', ')}`,
            );
            if (path.settings.holdFineProbes && worstView > LEAK_LIMIT) failures.push(`${path.name}: the dark room rose ${(worstView * 100).toFixed(2)}% of the lit room's light past its settled value (bar ${LEAK_LIMIT * 100}%)`);
            // Held, the dark room's fine probes stay under their settled light, which could hide a leak in the view's mean; per probe it can't hide.
            if (path.settings.holdFineProbes && worstProbes > LEAK_LIMIT) failures.push(`${path.name}: the dark room's fine probes rose ${(worstProbes * 100).toFixed(2)}% of the lit room's probes' light past their settled values (bar ${LEAK_LIMIT * 100}%)`);
            if (path.settings.holdFineProbes && !((counters['probes propagated (total)'] ?? 0) > 0)) failures.push(`${path.name}: propagation never ran`);
        }
        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
