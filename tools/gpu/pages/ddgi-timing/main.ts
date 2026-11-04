/**
 * T1.5a: GPU time of `ddgi/update` and `ddgi/write` per probe, through the
 * platform's own timer scopes (`stats.passes`), and the probes per frame
 * that the 2 ms update budget (SPEC) buys.
 *
 * Per scene: DDGI on the manifest's volume with default settings (256 rays),
 * WARMUP_FRAMES untimed, then MEASURED_FRAMES at each probe count in
 * UNITS. The medians per frame are divided by the probes updated. For hand
 * use under the bench lock (`manual`); run it three times and take the median.
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, runFrame } from '../ddgi-shared/harness.ts';

const SCENES = ['cornell-box-point', 'two-room'] as const;
const UNITS = [128, 512, 1024] as const;
const WARMUP_FRAMES = 30;
const MEASURED_FRAMES = 120;
const BUDGET_MS = 2;

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

await runGpuTest(
    async ({ device, log }) => {
        assert(device.features.has('timestamp-query'), 'timing needs the timestamp-query feature');
        const results: Record<string, unknown> = {};
        for (const sceneId of SCENES) {
            const scene = await createDdgiScene(device, sceneId, {}, { gpuTiming: true });
            const probes = scene.ddgi.field?.grid.probeCount ?? 0;
            const perUnits: Record<string, unknown> = {};
            for (const units of UNITS) {
                const updated = Math.min(units, probes);
                for (let frame = 0; frame < WARMUP_FRAMES; frame++) {
                    runFrame(device, scene.platform, units);
                    await device.queue.onSubmittedWorkDone();
                }
                const update = new Map<number, number>();
                const write = new Map<number, number>();
                // Timestamps arrive a few frames late; keep going until enough frames reported both passes.
                for (let frame = 0; update.size < MEASURED_FRAMES && frame < MEASURED_FRAMES * 3; frame++) {
                    runFrame(device, scene.platform, units);
                    await device.queue.onSubmittedWorkDone();
                    // Let the readback's mapAsync callbacks run before the next encode collects them.
                    await new Promise((resolve) => setTimeout(resolve, 0));
                    for (const pass of scene.platform.stats.passes) {
                        if (pass.gpuMs === undefined || pass.gpuMs < 0) continue;
                        if (pass.name === 'ddgi/update') update.set(pass.frame, pass.gpuMs);
                        if (pass.name === 'ddgi/write') write.set(pass.frame, pass.gpuMs);
                    }
                }
                assert(update.size >= MEASURED_FRAMES / 2, `${sceneId}: only ${update.size} timed frames`);
                const updateMs = median([...update.values()]);
                const writeMs = median([...write.values()]);
                const perProbeMs = (updateMs + writeMs) / updated;
                perUnits[String(units)] = {
                    probesUpdated: updated,
                    frames: update.size,
                    updateMs,
                    writeMs,
                    updateMsPerProbe: updateMs / updated,
                    writeMsPerProbe: writeMs / updated,
                    probesIn2Ms: Math.floor(BUDGET_MS / perProbeMs),
                };
                log(`${sceneId}, ${updated} probes: update ${updateMs.toFixed(3)} ms, write ${writeMs.toFixed(3)} ms; ${(perProbeMs * 1000).toFixed(2)} µs per probe; ${Math.floor(BUDGET_MS / perProbeMs)} probes in ${BUDGET_MS} ms`);
            }
            results[sceneId] = { probes, raysPerProbe: scene.ddgi.settings.raysPerProbe, units: perUnits };
            scene.platform.dispose();
        }
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
