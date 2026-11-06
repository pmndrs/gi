/**
 * T2.4a: `ddgi/mark` counts its marks subgroups first, with one declared
 * fallback (one atomic per marking thread); both must give the same answer
 * on the same input (the M1 rule for every reduction, DECISIONS 2026-11-04).
 *
 * Two two-room scenes, one per reduction (`DDGITechniqueOptions.reduction`,
 * which also picks phase B's reduction in `ddgi/blend`; both reductions give
 * the same states and offsets, so the two fields are the same). Each runs
 * WARM_FRAMES frames of every probe, then the lamp moves 2 m and one frame
 * runs `ddgi/mark` alone. Checked:
 * - the census's "probes marked (total)" is the same for both, and equals
 *   the probes whose history now holds a change (one light, so each probe is
 *   marked at most once);
 * - every probe's pendingChange and urgency are bit for bit the same.
 *
 * Also the global changes' paths on the GPU (one scene, the default
 * reduction): an environment change marks every active probe with r = 1
 * times its openness factor (0.25 to 1), and nothing else; a sun switched on
 * marks every active probe with MAX_GLOBAL_CHANGE times its openness factor.
 *
 * Runs on any GPU; without the `subgroups` feature only the fallback runs,
 * and the page says so.
 */

import { assert, runGpuTest } from '../../client.ts';
import { PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS } from '../../../../packages/ddgi-core/src/techniques/ddgi/layout.ts';
import type { PhaseBReduction } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/blend.ts';
import { ENVIRONMENT_CHANGE, MAX_GLOBAL_CHANGE } from '../../../../packages/ddgi-core/src/techniques/ddgi/mark-table.ts';
import { createDdgiScene, readBuffer, readProbeData, releaseDeviceOnPageHide, runFrames } from '../ddgi-shared/harness.ts';

const WARM_FRAMES = 64;
const MARKED_TOTAL = 'probes marked (total)';
/** Frames with one probe update after the mark, so census copies made after it are read back. */
const READBACK_FRAMES = 6;

interface MarkResult {
    readonly markedTotal: number | undefined;
    readonly pending: Uint32Array;
    readonly urgency: Uint32Array;
    readonly probesWithChange: number;
}

async function markOnce(device: GPUDevice, reduction: PhaseBReduction): Promise<MarkResult> {
    const scene = await createDdgiScene(device, 'two-room', {}, { gpuTiming: true, technique: { reduction } });
    try {
        assert(scene.ddgi.reduction === reduction, `asked for the ${reduction} reduction, got ${scene.ddgi.reduction}`);
        const field = scene.ddgi.field;
        assert(field, 'no field');
        const probeCount = field.grid.probeCount;
        await runFrames(device, scene.platform, WARM_FRAMES, probeCount);
        const lamp = scene.manifest.lights[0] as { position: [number, number, number] };
        scene.platform.scene.updateLight(scene.lights[0], { position: [lamp.position[0] - Math.SQRT2, lamp.position[1], lamp.position[2] - Math.SQRT2] });
        await runFrames(device, scene.platform, 1, 0);
        const words = new Uint32Array(await readBuffer(device, field.probeHistory, probeCount * PROBE_HISTORY_WORDS * 4));
        const pending = new Uint32Array(probeCount);
        const urgency = new Uint32Array(probeCount);
        for (let probe = 0; probe < probeCount; probe++) {
            pending[probe] = words[probe * PROBE_HISTORY_WORDS + PROBE_HISTORY_WORD.pendingChange];
            urgency[probe] = words[probe * PROBE_HISTORY_WORDS + PROBE_HISTORY_WORD.urgency];
        }
        // One probe a frame never reaches the marked probes' records before they're read above,
        // and the census copies these frames make carry the running total.
        for (let frame = 0; frame < READBACK_FRAMES; frame++) {
            await runFrames(device, scene.platform, 1, 1);
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
        const probesWithChange = [...new Float32Array(pending.buffer)].filter((value) => value > 0).length;
        return { markedTotal: scene.ddgi.counters[MARKED_TOTAL], pending, urgency, probesWithChange };
    } finally {
        scene.platform.dispose();
    }
}

/** Float rounding allowance on the openness factor's bounds. */
const FACTOR_SLACK = 1e-5;
/** The openness factor's floor (DDGI_OPENNESS_FLOOR in wgsl/mark.ts). */
const OPENNESS_FLOOR = 0.25;

/** The global kinds' marks on the GPU: every active probe within its factor's bounds, every other probe untouched. */
async function globalMarks(device: GPUDevice): Promise<Record<string, unknown>> {
    const scene = await createDdgiScene(device, 'two-room', {}, { gpuTiming: true });
    try {
        const field = scene.ddgi.field;
        assert(field, 'no field');
        const probeCount = field.grid.probeCount;
        await runFrames(device, scene.platform, WARM_FRAMES, probeCount);
        const pending = async (): Promise<Float32Array> => {
            const words = new Float32Array(await readBuffer(device, field.probeHistory, probeCount * PROBE_HISTORY_WORDS * 4));
            return Float32Array.from({ length: probeCount }, (_, probe) => words[probe * PROBE_HISTORY_WORDS + PROBE_HISTORY_WORD.pendingChange]);
        };
        const active = (await readProbeData(device, field)).map((probe) => probe.state === 0);
        const check = (label: string, before: Float32Array, after: Float32Array, change: number): Record<string, number> => {
            let marked = 0;
            let outOfBounds = 0;
            let strayMarks = 0;
            let smallest = Infinity;
            let largest = 0;
            for (let probe = 0; probe < probeCount; probe++) {
                const added = after[probe] - before[probe];
                if (!active[probe]) {
                    if (added !== 0) strayMarks++;
                    continue;
                }
                marked++;
                smallest = Math.min(smallest, added);
                largest = Math.max(largest, added);
                if (added < change * OPENNESS_FLOOR * (1 - FACTOR_SLACK) || added > change * (1 + FACTOR_SLACK)) outOfBounds++;
            }
            assert(outOfBounds === 0 && strayMarks === 0, `${label}: ${outOfBounds} active probes outside ${change * OPENNESS_FLOOR}..${change}, ${strayMarks} inactive probes marked`);
            return { marked, smallest, largest };
        };
        const start = await pending();
        scene.platform.scene.setEnvironment({ type: 'color', color: [0.2, 0.3, 0.4] });
        await runFrames(device, scene.platform, 1, 0);
        const afterSky = await pending();
        const sky = check('environment', start, afterSky, ENVIRONMENT_CHANGE);
        scene.platform.scene.createLight({ type: 'directional', color: [1, 1, 1], intensity: 2, direction: [0.3, -1, 0.2] });
        await runFrames(device, scene.platform, 1, 0);
        const sun = check('sun switched on', afterSky, await pending(), MAX_GLOBAL_CHANGE);
        return { environment: sky, sunSwitchedOn: sun };
    } finally {
        scene.platform.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const hasSubgroups = device.features.has('subgroups');
        const fallback = await markOnce(device, 'workgroup');
        assert(fallback.markedTotal !== undefined, 'the census was not read back');
        assert(fallback.probesWithChange > 0, 'the lamp step marked nothing');
        assert(fallback.markedTotal === fallback.probesWithChange, `the fallback counted ${fallback.markedTotal} marks for ${fallback.probesWithChange} marked probes`);
        log(`fallback: ${fallback.markedTotal} marks, ${fallback.probesWithChange} probes with a change`);
        const global = await globalMarks(device);
        log(`global marks: ${JSON.stringify(global)}`);
        if (!hasSubgroups) {
            log('no subgroups on this device: only the fallback ran');
            return { subgroups: false, fallbackMarks: fallback.markedTotal, global };
        }
        const subgroups = await markOnce(device, 'subgroups');
        log(`subgroups: ${subgroups.markedTotal} marks, ${subgroups.probesWithChange} probes with a change`);
        assert(subgroups.markedTotal === fallback.markedTotal, `subgroups counted ${subgroups.markedTotal} marks, the fallback ${fallback.markedTotal}`);
        let differing = 0;
        for (let probe = 0; probe < fallback.pending.length; probe++) {
            if (subgroups.pending[probe] !== fallback.pending[probe] || subgroups.urgency[probe] !== fallback.urgency[probe]) differing++;
        }
        assert(differing === 0, `${differing} probes' marks differ between the two reductions`);
        return { subgroups: true, marks: fallback.markedTotal, probesWithChange: fallback.probesWithChange, global };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
