/**
 * T2.1 on `sponza-candles` (Sponza at night with 1,024 of Intel's candles,
 * each a point light with no range), loaded the way the viewer loads it and
 * run through the three adapter.
 *
 * - **counts** (check 4): the scene has 1,024 candle lights in three and in
 *   GI, 1,954 triangles a candle drawn by three, and the candles out of GI's
 *   rays (scenes/src/three-scene.ts, `loadLightSet`, says why).
 * - **mark** (the TASKS addition to check 5; timed, one VALID run under the
 *   clock rule): `ddgi/mark`'s GPU time with every candle changing at once,
 *   which the folded rule bounds (DECISIONS 2026-11-06; mark-table.ts,
 *   MAX_UNRANGED_LIGHT_ENTRIES): all 1,024 flickering every frame, all
 *   drifting 1 cm a frame, and all switched off and on. Beside it, the
 *   per-light rule's cost (T2.4b: one thread per probe per light) measured
 *   at 1 and at 8 candles changing a frame (the most it still serves light
 *   by light) and carried to 1,024 by its per-light slope: what an all-candle
 *   frame would cost without the fold. Still frames must mark nothing.
 * - Also reported, not gated (T2.1: "`gi/direct` keeps its current light
 *   selection"): the calibrated batch at 2 ms and `ddgi/gi-direct`'s time
 *   with 1,024 lights, against the same on `sponza` lit by its sun (run the
 *   page with VITE_DDGI_SCENES=sponza for that; the candle sections skip).
 *
 * - **cost** (information, T2.3's starting point): the calibration's own
 *   samples, and `ddgi/trace`, `ddgi/gi-direct` and `ddgi/blend` at fixed
 *   batches of COST_BATCHES probes (the same on `sponza` with
 *   VITE_DDGI_SCENES=sponza).
 *
 *   VITE_DDGI_SECTIONS=counts,mark,cost   (default counts and mark)
 *
 * Manual (needs the Sponza and candle downloads):
 *   node tools/reference/bench-lock.ts run --what "T2.1 sponza-candles" -- pnpm test:gpu --gpu nvidia --filter ddgi-sponza-candles
 */

import type { PointLight } from 'three/webgpu';
import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { MAX_UNRANGED_LIGHT_ENTRIES } from '../../../../packages/core/src/techniques/ddgi/mark-table.ts';
import { startViewerScene, yieldToCallbacks, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
const SCENE_ID = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES !== '' ? env.VITE_DDGI_SCENES.split(',')[0].trim() : 'sponza-candles';
const SECTIONS = new Set(typeof env.VITE_DDGI_SECTIONS === 'string' && env.VITE_DDGI_SECTIONS !== '' ? env.VITE_DDGI_SECTIONS.split(',').map((item) => item.trim()) : ['counts', 'mark']);

/** The scene's default candle count (the manifest's `candles` setting) and Intel's candle (tools/assets/convert/sponza-candles.ts). */
const CANDLES = 1024;
const TRIANGLES_PER_CANDLE = 1954;
const BUDGET_MS = 2;
const WARM_UP_PASSES = 4;
const CALIBRATION_MAX_FRAMES = 600;
const STILL_FRAMES = 120;
const CHANGE_FRAMES = 120;
const TOGGLES = 8;
const TOGGLE_GAP_FRAMES = 12;
const FRAMES_IN_FLIGHT = 2;
const MARK_PASS = 'ddgi/mark';
const BLEND_PASS = 'ddgi/blend';
const COST_BATCHES = [16, 64, 256];
const COST_FRAMES = 40;
const DIRECT_PASS = 'ddgi/gi-direct';
const TRACE_PASS = 'ddgi/trace';
/** A candle's flicker: its intensity a few percent either way, frame by frame. */
const FLICKER = 0.05;
const DRIFT_M = 0.01;

/** Keeps up to FRAMES_IN_FLIGHT frames queued (GPU-bound), as ddgi-batch-cost does. */
class GpuBoundPacing {
    private readonly pending: Promise<void>[] = [];
    constructor(private readonly device: GPUDevice) {}
    async after(): Promise<void> {
        this.pending.push(this.device.queue.onSubmittedWorkDone());
        if (this.pending.length >= FRAMES_IN_FLIGHT) await this.pending.shift();
        await yieldToCallbacks();
    }
    async drain(): Promise<void> {
        await Promise.all(this.pending.splice(0));
    }
}

/** Every pass timing seen, by platform frame (the platform keeps only the newest timed frame's). */
class FrameTimings {
    readonly byFrame = new Map<number, Map<string, number>>();
    note(scene: ViewerScene): void {
        for (const pass of scene.platform.stats.passes) {
            if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
            let passes = this.byFrame.get(pass.frame);
            if (!passes) {
                passes = new Map();
                this.byFrame.set(pass.frame, passes);
            }
            passes.set(pass.name, pass.gpuMs);
        }
    }
    /** A pass's times over frames [first, last], where it ran. */
    pass(name: string, first: number, last: number): number[] {
        const times: number[] = [];
        for (let frame = first; frame <= last; frame++) {
            const ms = this.byFrame.get(frame)?.get(name);
            if (ms !== undefined) times.push(ms);
        }
        return times;
    }
}

function quantile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))] ?? Number.NaN;
}
const round = (value: number, digits = 4) => Number(value.toFixed(digits));
const summary = (values: readonly number[]) => ({ frames: values.length, median: round(quantile(values, 0.5)), p90: round(quantile(values, 0.9)), largest: round(Math.max(...values)) });

await runThreePage(async ({ device, data, log: logLine }) => {
    const log = (line: string): void => {
        logLine(line);
        console.log(line);
    };
    const results = data;
    const failures: string[] = [];
    const manifest = loadAllManifests().find((entry) => entry.id === SCENE_ID);
    assert(manifest, `no manifest ${SCENE_ID}`);
    results.scene = SCENE_ID;
    const scene = await startViewerScene(device, manifest, { gpuTiming: true, label: SCENE_ID });
    const candles: PointLight[] = scene.loaded.lightSets.flatMap((set) => set.lights);
    const isCandleScene = scene.loaded.lightSets.length > 0;

    if (SECTIONS.has('counts') && isCandleScene) {
        const set = scene.loaded.lightSets[0];
        const copy = scene.platform.sceneCopy.counts;
        const counts = {
            setting: scene.loaded.settings.candles,
            candlesInThree: set.count,
            candleLightsInThree: candles.length,
            candleTrianglesInThree: set.triangles,
            sceneTrianglesInThree: scene.loaded.stats.triangles,
            lightsInGi: copy.lights,
            // The scene copy counts each mesh once: the candle is one mesh, whatever the count.
            meshTrianglesInGi: copy.triangles,
            instancesInGi: copy.instances,
            instancesInGiRays: scene.platform.sceneCopy.packGiInstances().slots.length,
            candleInstances: set.group.children.length * set.count,
            candlesInPackage: set.set.count,
        };
        results.counts = counts;
        log(
            `counts: ${counts.candlesInThree} candles of ${counts.candlesInPackage} (setting ${counts.setting}); three: ${counts.candleLightsInThree} candle lights, ` +
                `${counts.sceneTrianglesInThree.toLocaleString('en')} triangles (${counts.candleTrianglesInThree.toLocaleString('en')} of them candles); ` +
                `GI: ${counts.lightsInGi} lights, ${counts.meshTrianglesInGi.toLocaleString('en')} mesh triangles, ${counts.instancesInGi} instances of which ${counts.instancesInGiRays} in GI's rays (${counts.candleInstances} candle instances left out)`,
        );
        if (counts.candlesInThree !== CANDLES || counts.candleLightsInThree !== CANDLES) failures.push(`${counts.candlesInThree} candles and ${counts.candleLightsInThree} lights in three, not ${CANDLES}`);
        if (counts.candleTrianglesInThree !== CANDLES * TRIANGLES_PER_CANDLE) failures.push(`the candles have ${counts.candleTrianglesInThree} triangles, not ${CANDLES} x ${TRIANGLES_PER_CANDLE}`);
        if (counts.lightsInGi !== CANDLES + manifest.lights.length) failures.push(`GI has ${counts.lightsInGi} lights, not ${CANDLES + manifest.lights.length}`);
        if (counts.meshTrianglesInGi !== counts.sceneTrianglesInThree - counts.candleTrianglesInThree + TRIANGLES_PER_CANDLE) failures.push(`GI's meshes have ${counts.meshTrianglesInGi} triangles: not the scene's plus one candle`);
        if (!(counts.instancesInGiRays <= counts.instancesInGi - counts.candleInstances)) failures.push(`${counts.instancesInGiRays} of ${counts.instancesInGi} instances are in GI's rays: the candles are among them`);
        if (!candles.every((light) => light.distance === 0 && light.castShadow === false && light.userData.giCastsShadows === true)) failures.push('a candle light has a range, or three shadows, or GI no shadows');
    }

    if (SECTIONS.has('mark')) {
        const timings = new FrameTimings();
        const pacing = new GpuBoundPacing(device);
        const run = async (frames: number, budget: Parameters<typeof scene.frame>[0], before?: (index: number) => void): Promise<{ first: number; last: number }> => {
            const first = scene.platform.stats.frame + 1;
            for (let index = 0; index < frames; index++) {
                before?.(index);
                scene.frame(budget, 1 / 60);
                timings.note(scene);
                await pacing.after();
            }
            await pacing.drain();
            // Timings arrive a few frames late: frames with no work let the last ones land.
            for (let extra = 0; extra < 6; extra++) {
                scene.frame({ paused: true }, 0);
                timings.note(scene);
                await device.queue.onSubmittedWorkDone();
                await yieldToCallbacks();
            }
            return { first, last: first + frames - 1 };
        };

        // Warm-up and calibration at 2 ms, as the viewer runs.
        scene.ddgi.settings.warmUpPasses = WARM_UP_PASSES;
        scene.ddgi.resetHistory();
        while (scene.ddgi.warmingUp) {
            scene.frame({ paused: false }, 0);
            await pacing.after();
        }
        await pacing.drain();
        scene.ddgi.settings.warmUpPasses = 0;
        let calibrationFrames = 0;
        for (;;) {
            scene.frame({ gpuMs: BUDGET_MS }, 0);
            calibrationFrames++;
            const report = scene.platform.budgetReport(scene.ddgi);
            if (!(report?.state === 'calibrating' || report?.confirming === true) || calibrationFrames >= CALIBRATION_MAX_FRAMES) break;
            await pacing.after();
        }
        await pacing.drain();
        const calibration = scene.platform.budgetReport(scene.ddgi)?.calibration;
        assert(calibration?.measured, `the platform did not calibrate within ${calibrationFrames} frames`);
        const units = calibration.units;
        results.calibration = { units, frames: calibrationFrames, samples: calibration.samples.map((sample) => ({ units: sample.units, ms: round(sample.ms) })) };

        if (SECTIONS.has('cost')) {
            const batches: Record<string, unknown>[] = [];
            for (const batch of COST_BATCHES) {
                const range = await run(COST_FRAMES, { units: batch });
                const entry = {
                    probes: batch,
                    trace: round(quantile(timings.pass(TRACE_PASS, range.first, range.last), 0.5)),
                    direct: round(quantile(timings.pass(DIRECT_PASS, range.first, range.last), 0.5)),
                    blend: round(quantile(timings.pass(BLEND_PASS, range.first, range.last), 0.5)),
                    hitsLit: scene.ddgi.counters['hits lit by gi/direct'],
                };
                batches.push(entry);
                log(`${SCENE_ID} at ${batch} probes a frame: ddgi/trace ${entry.trace} ms, ddgi/gi-direct ${entry.direct} ms, ddgi/blend ${entry.blend} ms (medians; hits lit ${entry.hitsLit})`);
            }
            results.cost = batches;
            log(`calibration at ${BUDGET_MS} ms: ${units} probes; its samples: ${calibration.samples.map((sample) => `${sample.units}: ${sample.ms.toFixed(3)} ms`).join(', ')}`);
        }

        // Still: nothing marked; gi/direct's cost at the calibrated batch.
        const markedBefore = scene.ddgi.counters['frames with changes marked'];
        const still = await run(STILL_FRAMES, { units });
        const stillMarked = scene.ddgi.counters['frames with changes marked'] - markedBefore;
        results.still = {
            calibratedUnits: units,
            framesMarked: stillMarked,
            trace: summary(timings.pass(TRACE_PASS, still.first, still.last)),
            direct: summary(timings.pass(DIRECT_PASS, still.first, still.last)),
        };
        log(
            `${SCENE_ID}: calibrated at ${BUDGET_MS} ms to ${units} probes a frame; still frames marked ${stillMarked}; ` +
                `ddgi/trace median ${round(quantile(timings.pass(TRACE_PASS, still.first, still.last), 0.5))} ms, ddgi/gi-direct median ${round(quantile(timings.pass(DIRECT_PASS, still.first, still.last), 0.5))} ms (information)`,
        );
        if (stillMarked !== 0) failures.push(`still frames marked changes in ${stillMarked} frames`);

        if (isCandleScene) {
            const base = candles.map((light) => light.intensity);
            const home = candles.map((light) => light.position.clone());
            const send = (light: PointLight) => {
                light.updateMatrixWorld();
                scene.threeGI.markChanged(light);
            };
            const markSummary = (range: { first: number; last: number }) => summary(timings.pass(MARK_PASS, range.first, range.last));

            // The per-light rule: 1 and 8 candles changing a frame, each over every probe.
            const perLight: Record<string, ReturnType<typeof summary>> = {};
            for (const count of [1, MAX_UNRANGED_LIGHT_ENTRIES]) {
                let next = 0;
                const range = await run(CHANGE_FRAMES, { units }, (frame) => {
                    for (let index = 0; index < count; index++) {
                        const light = candles[next++ % candles.length];
                        light.intensity = base[candles.indexOf(light)] * (1 + (frame % 2 === 0 ? FLICKER : -FLICKER));
                        send(light);
                    }
                });
                perLight[String(count)] = markSummary(range);
            }
            candles.forEach((light, index) => {
                light.intensity = base[index];
                send(light);
            });
            await run(30, { units });
            const slope = (perLight[String(MAX_UNRANGED_LIGHT_ENTRIES)].median - perLight['1'].median) / (MAX_UNRANGED_LIGHT_ENTRIES - 1);
            const perLightEstimate = perLight['1'].median + slope * (CANDLES - 1);

            // The folded rule: every candle at once.
            const flicker = await run(CHANGE_FRAMES, { units }, (frame) => {
                candles.forEach((light, index) => {
                    light.intensity = base[index] * (1 + ((frame + index) % 2 === 0 ? FLICKER : -FLICKER));
                    send(light);
                });
            });
            const drift = await run(CHANGE_FRAMES, { units }, (frame) => {
                candles.forEach((light, index) => {
                    light.intensity = base[index];
                    light.position.copy(home[index]).setY(home[index].y + DRIFT_M * Math.sin(frame * 0.3 + index));
                    send(light);
                });
            });
            candles.forEach((light, index) => {
                light.position.copy(home[index]);
                send(light);
            });
            await run(30, { units });
            const toggleFrames: number[] = [];
            for (let toggle = 0; toggle < TOGGLES; toggle++) {
                const range = await run(1, { units }, () => candles.forEach((light, index) => {
                    light.intensity = toggle % 2 === 0 ? 0 : base[index];
                    send(light);
                }));
                toggleFrames.push(...timings.pass(MARK_PASS, range.first, range.last));
                await run(TOGGLE_GAP_FRAMES, { units });
            }
            candles.forEach((light, index) => {
                light.intensity = base[index];
                send(light);
            });

            const folded = { flicker: markSummary(flicker), drift: markSummary(drift), switchedOffAndOn: summary(toggleFrames) };
            results.mark = { probes: scene.probeCount, perLight, perLightSlopeMs: round(slope, 5), perLightEstimateFor1024Ms: round(perLightEstimate, 3), folded, counters: scene.ddgi.counters };
            log(
                `ddgi/mark, every one of ${CANDLES} candles changing in a frame (folded): flicker median ${folded.flicker.median} ms (p90 ${folded.flicker.p90}), drift ${folded.drift.median} ms, ` +
                    `switched off and on ${folded.switchedOffAndOn.median} ms (largest ${folded.switchedOffAndOn.largest}); ` +
                    `the per-light rule: 1 candle ${perLight['1'].median} ms, ${MAX_UNRANGED_LIGHT_ENTRIES} candles ${perLight[String(MAX_UNRANGED_LIGHT_ENTRIES)].median} ms, ` +
                    `so ${CANDLES} would be about ${round(perLightEstimate, 2)} ms (${(CANDLES * scene.probeCount).toLocaleString('en')} threads)`,
            );
            for (const [name, entry] of Object.entries(folded)) if (!(entry.frames > 0)) failures.push(`no ddgi/mark timings for ${name}`);
        }
    }

    scene.dispose();
    assert(failures.length === 0, failures.join('; '));
});
