/**
 * T2.4c: `ddgi/propagate-find` compacts its list subgroups first, with one
 * declared fallback (one atomic per listed probe); both must give the same
 * field (the M1 rule for every reduction, DECISIONS 2026-11-04).
 *
 * Two two-room scenes, one per reduction (`DDGITechniqueOptions.reduction`,
 * which picks every DDGI kernel's form). The list's order differs between
 * the two, but each listed fine probe's result doesn't depend on its place
 * in the list, so the fields must match bit for bit. The scheduler's
 * priority share is set to 0 for this page: its boundary bucket's members
 * are whichever reserve a slot first, which is allowed to differ between the
 * forms, and with the round robin alone the update lists are the same.
 *
 * Each scene: WARM_FRAMES frames of every probe; the lamp switched off; then
 * STEP_FRAMES frames at UNITS probes, a batch at which two-room takes the
 * key-probe path (1,071 probes × 8 updates / 64 > 120 frames). Checked:
 * propagation ran (key changes passed and probes propagated, from the
 * census), the two reductions' totals are equal, and the irradiance atlas
 * and every probe's history record are the same bit for bit.
 *
 * Runs on any GPU; without the `subgroups` feature only the fallback runs,
 * and the page says so.
 */

import { assert, runGpuTest } from '../../client.ts';
import { PROBE_HISTORY_WORDS } from '../../../../packages/ddgi/src/layout.ts';
import type { PhaseBReduction } from '../../../../packages/ddgi/src/wgsl/blend.ts';
import { createDdgiScene, readBuffer, readTextureBytes, releaseDeviceOnPageHide, runFrames } from '../ddgi-shared/harness.ts';

const WARM_FRAMES = 64;
const STEP_FRAMES = 90;
const UNITS = 64;
/** Frames with one probe update after the run, so census copies made after it are read back. */
const READBACK_FRAMES = 6;
const IRRADIANCE_TEXEL_BYTES = 8;

interface PropagationRun {
    readonly propagated: number | undefined;
    readonly keyChanges: number | undefined;
    readonly irradiance: Uint8Array;
    readonly history: Uint32Array;
}

async function propagationRun(device: GPUDevice, reduction: PhaseBReduction): Promise<PropagationRun> {
    const scene = await createDdgiScene(device, 'two-room', { priorityShare: 0 }, { gpuTiming: true, technique: { reduction } });
    try {
        assert(scene.ddgi.reduction === reduction, `asked for the ${reduction} reduction, got ${scene.ddgi.reduction}`);
        const field = scene.ddgi.field;
        assert(field, 'no field');
        const probeCount = field.grid.probeCount;
        await runFrames(device, scene.platform, WARM_FRAMES, probeCount);
        scene.platform.scene.updateLight(scene.lights[0], { intensity: 0 });
        await runFrames(device, scene.platform, STEP_FRAMES, UNITS);
        const irradiance = await readTextureBytes(device, field.irradiance, IRRADIANCE_TEXEL_BYTES);
        const history = new Uint32Array(await readBuffer(device, field.probeHistory, probeCount * PROBE_HISTORY_WORDS * 4));
        for (let frame = 0; frame < READBACK_FRAMES; frame++) {
            await runFrames(device, scene.platform, 1, 0);
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
        const counters = scene.ddgi.counters;
        return { propagated: counters['probes propagated (total)'], keyChanges: counters['key changes passed (total)'], irradiance, history };
    } finally {
        scene.platform.dispose();
    }
}

function firstDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
    if (a.length !== b.length) return Math.min(a.length, b.length);
    for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return index;
    return -1;
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const fallback = await propagationRun(device, 'workgroup');
        assert(fallback.propagated !== undefined && fallback.keyChanges !== undefined, 'the census was not read back');
        assert(fallback.keyChanges > 0 && fallback.propagated > 0, `propagation didn't run: ${fallback.keyChanges} key changes passed, ${fallback.propagated} probes propagated`);
        log(`fallback: ${fallback.keyChanges} key changes passed, ${fallback.propagated} probes propagated`);
        if (!device.features.has('subgroups')) {
            log('no subgroups on this device: only the fallback ran');
            return { subgroups: false, fallback: { keyChanges: fallback.keyChanges, propagated: fallback.propagated } };
        }
        const subgroups = await propagationRun(device, 'subgroups');
        log(`subgroups: ${subgroups.keyChanges} key changes passed, ${subgroups.propagated} probes propagated`);
        assert(subgroups.propagated === fallback.propagated && subgroups.keyChanges === fallback.keyChanges, 'the two reductions propagated differently');
        const atlas = firstDifference(subgroups.irradiance, fallback.irradiance);
        const history = firstDifference(subgroups.history, fallback.history);
        assert(atlas < 0, `the irradiance atlases differ from byte ${atlas}`);
        assert(history < 0, `the probe histories differ from word ${history}`);
        return { subgroups: true, keyChanges: fallback.keyChanges, propagated: fallback.propagated, atlasBytes: fallback.irradiance.length };
    },
    { optionalFeatures: ['subgroups', 'timestamp-query', 'texture-formats-tier1'] },
);
