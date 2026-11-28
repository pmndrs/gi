/**
 * T2.4a: `ddgi/mark` counts its marks subgroups first, with one declared
 * fallback (one atomic per marking thread); both must give the same answer
 * on the same input (the M1 rule for every reduction, DECISIONS 2026-11-04).
 *
 * Two two-room scenes, one per reduction (`DDGITechniqueOptions.reduction`,
 * which also picks phase B's reduction in `ddgi/blend`; both reductions give
 * the same states and offsets). The feedback rule is off in them: its
 * requests put probes in the scheduler's priority share, whose list slots
 * the two reductions may fill in a different order, and since T2.2 a ray's
 * random numbers are keyed by its slot (`giDirectRandom(rayRecord, frame)`),
 * so the two fields would be two different, equally valid realizations
 * (T2.M1, DECISIONS 2026-11-06). Without it every list is the round robin's,
 * slot for slot, and the fields must be the same to the bit (checked on the
 * irradiance atlas before the mark). Each runs
 * WARM_FRAMES frames of every probe, then the lamp moves 2 m and one frame
 * runs `ddgi/mark` alone. Checked:
 * - the census's "probes marked (total)" is the same for both, and equals
 *   the probes whose history now holds a change (one light, so each probe is
 *   marked at most once);
 * - every probe's pendingChange and urgency are bit for bit the same.
 *
 * Also the global changes' paths on the GPU (one scene, the default
 * reduction): an environment change marks every active probe with the sky's
 * own change times its openness factor (0.25 to 1), and nothing else
 * (T2.6a: the change of the sky's band-0 and band-1 radiance SH against its
 * mean; for a constant colour, the colour's change per channel, where it got
 * darker against what is left). A sky switched on from black marks with
 * MAX_GLOBAL_CHANGE, a colour with its red halved with 1, and the same
 * colour set again marks nothing. A sun switched on marks every active
 * probe with MAX_GLOBAL_CHANGE times its openness factor.
 *
 * Runs on any GPU; without the `subgroups` feature only the fallback runs,
 * and the page says so.
 */

import { assert, runGpuTest } from '../../client.ts';
import { PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS } from '../../../../packages/ddgi/src/layout.ts';
import type { PhaseBReduction } from '../../../../packages/ddgi/src/wgsl/blend.ts';
import { ENVIRONMENT_CHANGE, MAX_GLOBAL_CHANGE } from '../../../../packages/ddgi/src/mark-table.ts';
import type { EnvironmentDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import { createDdgiScene, readBuffer, readProbeData, readTextureBytes, releaseDeviceOnPageHide, runFrames } from '../ddgi-shared/harness.ts';

const WARM_FRAMES = 64;
const MARKED_TOTAL = 'probes marked (total)';
/** Frames with one probe update after the mark, so census copies made after it are read back. */
const READBACK_FRAMES = 6;

interface MarkResult {
    /** The irradiance atlas just before the mark, for checking that both scenes start from the same field. */
    readonly atlas: Uint8Array;
    readonly markedTotal: number | undefined;
    readonly pending: Uint32Array;
    readonly urgency: Uint32Array;
    readonly probesWithChange: number;
}

async function markOnce(device: GPUDevice, reduction: PhaseBReduction): Promise<MarkResult> {
    // Feedback off: the file comment says why.
    const scene = await createDdgiScene(device, 'two-room', { feedback: false }, { gpuTiming: true, technique: { reduction } });
    try {
        assert(scene.ddgi.reduction === reduction, `asked for the ${reduction} reduction, got ${scene.ddgi.reduction}`);
        const field = scene.ddgi.field;
        assert(field, 'no field');
        const probeCount = field.grid.probeCount;
        await runFrames(device, scene.platform, WARM_FRAMES, probeCount);
        const atlas = await readTextureBytes(device, field.irradiance, 8);
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
        return { atlas, markedTotal: scene.ddgi.counters[MARKED_TOTAL], pending, urgency, probesWithChange };
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
        // Constant skies (two-room starts with none): black to a colour, its red halved, the same again.
        const skySteps: Array<[string, EnvironmentDescriptor, number]> = [
            ['a sky switched on from black', { type: 'color', color: [0.2, 0.3, 0.4] }, MAX_GLOBAL_CHANGE],
            ["the sky's red halved", { type: 'color', color: [0.1, 0.3, 0.4] }, 1],
            ['the same sky set again', { type: 'color', color: [0.1, 0.3, 0.4] }, 0],
        ];
        let before = await pending();
        const environment: Record<string, unknown> = {};
        for (const [label, sky, change] of skySteps) {
            scene.platform.scene.setEnvironment(sky);
            await runFrames(device, scene.platform, 1, 0);
            const after = await pending();
            environment[label] = check(`environment, ${label}`, before, after, change * ENVIRONMENT_CHANGE);
            before = after;
        }
        scene.platform.scene.createLight({ type: 'directional', color: [1, 1, 1], intensity: 2, direction: [0.3, -1, 0.2] });
        await runFrames(device, scene.platform, 1, 0);
        const sun = check('sun switched on', before, await pending(), MAX_GLOBAL_CHANGE);
        return { environment, sunSwitchedOn: sun };
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
        let atlasDiffers = 0;
        for (let index = 0; index < fallback.atlas.length; index++) if (subgroups.atlas[index] !== fallback.atlas[index]) atlasDiffers++;
        assert(atlasDiffers === 0, `the two scenes' fields differ before the mark (${atlasDiffers} bytes of the irradiance atlas), so their marks can't be compared`);
        assert(subgroups.markedTotal === fallback.markedTotal,
 `subgroups counted ${subgroups.markedTotal} marks, the fallback ${fallback.markedTotal}`);
        let differing = 0;
        for (let probe = 0; probe < fallback.pending.length; probe++) {
            if (subgroups.pending[probe] !== fallback.pending[probe] || subgroups.urgency[probe] !== fallback.urgency[probe]) differing++;
        }
        assert(differing === 0, `${differing} probes' marks differ between the two reductions`);
        return { subgroups: true, marks: fallback.markedTotal, probesWithChange: fallback.probesWithChange, global };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
