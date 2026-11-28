/**
 * DDGI's GPU time per probe, through the platform's own timer scopes
 * (`stats.passes`: `ddgi/trace`, `ddgi/gi-direct`, `ddgi/blend`, `ddgi/write`), the probes per
 * frame the 2 ms update budget (SPEC) buys, and the platform's conversion of
 * that budget into probes.
 *
 * Per scene: DDGI on the manifest's volume with default settings (256 rays),
 * WARMUP_FRAMES untimed, then MEASURED_FRAMES at each probe count in UNITS;
 * the medians per frame are divided by the probes updated. Then the budget
 * check: frames with `budget.gpuMs = 2` and no units, so the platform picks the
 * probe count from its measured cost per probe; their median DDGI time is
 * reported. For hand use under the bench lock (`manual`); run it three times
 * and take the median.
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, runFrame, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import type { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import type { FrameBudget } from '../../../../packages/core/src/platform/contract/frame.ts';

const SCENES = ['cornell-box-point', 'two-room'] as const;
const UNITS = [128, 512, 1024] as const;
const WARMUP_FRAMES = 30;
const MEASURED_FRAMES = 120;
const BUDGET_MS = 2;
// With direct light deferred (the default since T2.2) the platform's gi/direct is part of the update, timed as DDGI's.
// UNITS stay at or below 1,024 probes, one deferred round, so each pass runs once a frame.
const PASSES = ['ddgi/trace', 'ddgi/gi-direct', 'ddgi/blend', 'ddgi/write'] as const;

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Runs frames with this budget and collects each DDGI pass's GPU ms by frame, and the probes updated. */
async function timeFrames(device: GPUDevice, platform: GIPlatformImpl, budget: number | FrameBudget, frames: number, probesUpdated: () => number) {
    const byPass = new Map<string, Map<number, number>>(PASSES.map((name) => [name, new Map()]));
    const updated: number[] = [];
    // Timestamps arrive a few frames late; keep going until enough frames reported every pass.
    for (let frame = 0; byPass.get('ddgi/trace')!.size < frames && frame < frames * 3; frame++) {
        runFrame(device, platform, budget);
        updated.push(probesUpdated());
        await device.queue.onSubmittedWorkDone();
        // Let the readback's mapAsync callbacks run before the next encode collects them.
        await new Promise((resolve) => setTimeout(resolve, 0));
        for (const pass of platform.stats.passes) {
            if (pass.gpuMs === undefined || pass.gpuMs < 0) continue;
            byPass.get(pass.name)?.set(pass.frame, pass.gpuMs);
        }
    }
    assert(byPass.get('ddgi/trace')!.size >= frames / 2, `only ${byPass.get('ddgi/trace')!.size} timed frames`);
    const medians = Object.fromEntries(PASSES.map((name) => [name, median([...byPass.get(name)!.values()])])) as Record<(typeof PASSES)[number], number>;
    return { medians, totalMs: PASSES.reduce((sum, name) => sum + medians[name], 0), medianUpdated: median(updated) };
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        assert(device.features.has('timestamp-query'), 'timing needs the timestamp-query feature');
        const results: Record<string, unknown> = {};
        for (const sceneId of SCENES) {
            const scene = await createDdgiScene(device, sceneId, {}, { gpuTiming: true });
            const probes = scene.ddgi.field?.grid.probeCount ?? 0;
            const probesUpdated = (): number => scene.ddgi.counters['probes updated'];
            const perUnits: Record<string, unknown> = {};
            for (const units of UNITS) {
                for (let frame = 0; frame < WARMUP_FRAMES; frame++) {
                    runFrame(device, scene.platform, units);
                    await device.queue.onSubmittedWorkDone();
                }
                const updated = Math.min(units, probes);
                const timed = await timeFrames(device, scene.platform, units, MEASURED_FRAMES, probesUpdated);
                const perProbeMs = timed.totalMs / updated;
                perUnits[String(units)] = {
                    probesUpdated: updated,
                    traceMs: timed.medians['ddgi/trace'],
                    blendMs: timed.medians['ddgi/blend'],
                    writeMs: timed.medians['ddgi/write'],
                    msPerProbe: perProbeMs,
                    probesIn2Ms: Math.floor(BUDGET_MS / perProbeMs),
                };
                log(`${sceneId}, ${updated} probes: trace ${timed.medians['ddgi/trace'].toFixed(3)}, blend ${timed.medians['ddgi/blend'].toFixed(3)}, write ${timed.medians['ddgi/write'].toFixed(4)} ms; ${(perProbeMs * 1000).toFixed(2)} µs per probe; ${Math.floor(BUDGET_MS / perProbeMs)} probes in ${BUDGET_MS} ms`);
            }

            // The platform's own conversion: gpuMs only, probes from its measured cost.
            const budget: FrameBudget = { gpuMs: BUDGET_MS };
            for (let frame = 0; frame < WARMUP_FRAMES; frame++) {
                runFrame(device, scene.platform, budget);
                await device.queue.onSubmittedWorkDone();
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
            const budgeted = await timeFrames(device, scene.platform, budget, MEASURED_FRAMES, probesUpdated);
            const budgetCheck = { gpuMs: BUDGET_MS, medianProbesUpdated: budgeted.medianUpdated, medianDdgiMs: budgeted.totalMs, costModel: scene.platform.costModel(scene.ddgi) };
            log(`${sceneId}, budget ${BUDGET_MS} ms: ${budgeted.medianUpdated} probes per frame, DDGI ${budgeted.totalMs.toFixed(3)} ms`);

            // With gpuTiming on, the census runs: the per-state counters are there.
            results[sceneId] = { probes, raysPerProbe: scene.ddgi.settings.raysPerProbe, units: perUnits, budgetCheck, counters: scene.ddgi.counters };
            scene.platform.dispose();
        }
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
