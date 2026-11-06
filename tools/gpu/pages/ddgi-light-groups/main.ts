/**
 * T2.5 light groups, the correctness checks (docs/design/m2-lights.md §5;
 * TASKS T2.5 done-when 1). Two-room with 2 light groups: its lamp in room A
 * (group 0) and a second lamp added in room B (group 1), every probe updated
 * every frame, GI only (the engine-free test view's indirect image) at the
 * scene's three bookmarks.
 *
 * 1. **Linearity:** after FRAMES frames with both weights 1, the weights
 *    change (group 0 to grey 0.5, group 1 to the colour WEIGHT_B) in a frame
 *    that updates no probe. In that same frame the image must match a field
 *    re-converged from scratch, with groups off, with the lamps themselves
 *    dimmed and recoloured by those weights: mean within 1% at every
 *    bookmark (the done-when). The image must also have moved (more than 1%):
 *    the change shows in the frame it happens.
 * 2. **A compose, not a light change:** that frame marks nothing and cuts no
 *    history: the probe history and the layered atlas are byte for byte what
 *    they were, and the composed atlas alone changed. The frames after it
 *    mark no probe either (the GPU's own count).
 * 3. **Layers add up:** with both weights 1 the image matches groups off with
 *    the lamps as they are (within 1%; information on how the layers' sums
 *    differ from one field's, reported and checked).
 * 4. **Both reductions:** the same scene with the declared fallbacks (phase
 *    B's and propagation's workgroup reductions, gi/direct's one-atomic-a-hit
 *    work list) gives the same layered field after FALLBACK_FRAMES frames
 *    (atlas mean within 0.1%, T2.2's bar for two compiled variants).
 *
 * Also recorded: GPU memory with 2 groups against groups off (per layer).
 * Runs on the RTX and, for correctness, the Intel UHD.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { LightDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { DDGIField } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { defaultLightGroups } from '../../../../packages/ddgi-core/src/techniques/ddgi/light-groups.ts';
import { createDdgiScene, halfToFloat, imageMean, readBuffer, readTextureBytes, releaseDeviceOnPageHide, runFrame, runFrames, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

/** Frames each field converges for, every probe every frame. */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 900;
/** Frames for the reduction comparison. */
const FALLBACK_FRAMES = 300;
const WIDTH = 480;
const HEIGHT = 270;
/** Done-when 1: the image within 1% mean of a re-converged field. */
const LINEARITY_TOLERANCE = 0.01;
/** The weight change must show: the image moves by more than this in its frame. */
const SHOWS_THRESHOLD = 0.01;
/** Two compiled variants of the kernels: atlas mean within 0.1% (T2.2, DECISIONS 2026-11-06). */
const REDUCTION_TOLERANCE = 0.001;

/** Room B's lamp, in group 1 (two-room's own lamp lights room A and is in group 0). Cooler than room A's. */
const LAMP_B: LightDescriptor = { type: 'point', color: [0.75, 0.85, 1.0], intensity: 8, position: [6.2, 2.45, 2.0], range: 14, group: 1, label: 'room-b-lamp' };
/** The weights after the change: room A's lamp at half, room B's recoloured (and brighter in blue). */
const WEIGHT_A: [number, number, number] = [0.5, 0.5, 0.5];
const WEIGHT_B: [number, number, number] = [0.3, 0.6, 1.5];

/** Room B's lamp as a lamp of its own with WEIGHT_B folded in: colour × weight, brightest channel 1. */
function weightedLamp(lamp: LightDescriptor & { type: 'point' }, weight: readonly number[]): LightDescriptor {
    const color = lamp.color.map((channel, index) => channel * weight[index]);
    const brightest = Math.max(...color);
    return { ...lamp, color: color.map((channel) => channel / brightest) as [number, number, number], intensity: lamp.intensity * brightest };
}

async function views(device: GPUDevice, scene: DdgiScene, field: DDGIField): Promise<number[]> {
    const view = await TestView.create(device, scene.platform);
    const means: number[] = [];
    for (const camera of scene.manifest.cameras) means.push(imageMean((await view.render(field, camera, WIDTH, HEIGHT)).indirect));
    return means;
}

/** Mean of rgb over every texel of an rgba16float atlas. */
async function atlasMean(device: GPUDevice, texture: GPUTexture): Promise<number> {
    const halves = new Uint16Array((await readTextureBytes(device, texture, 8)).buffer);
    let sum = 0;
    for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
    return sum / ((halves.length / 4) * 3);
}

function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
    if (a.length !== b.length) return false;
    for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return false;
    return true;
}

const relative = (value: number, reference: number): number => value / reference - 1;
const percent = (share: number): string => `${(share * 100).toFixed(3)}%`;

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const failures: string[] = [];
        const results: Record<string, unknown> = { frames: FRAMES };

        // ---- The scene with 2 light groups ----
        const groups = await createDdgiScene(device, 'two-room', { lightGroups: defaultLightGroups(2) }, { gpuTiming: true });
        groups.platform.scene.createLight(LAMP_B);
        const probes = groups.ddgi.field!.grid.probeCount;
        await runFrames(device, groups.platform, FALLBACK_FRAMES, probes);
        const groupsField = groups.ddgi.field!;
        const layeredAt300 = await atlasMean(device, groupsField.layeredIrradiance);
        await runFrames(device, groups.platform, FRAMES - FALLBACK_FRAMES, probes);
        assert(groups.ddgi.counters['irradiance layers'] === 3, `expected 3 layers, got ${groups.ddgi.counters['irradiance layers']}`);
        const beforeChange = await views(device, groups, groupsField);
        const historyBefore = new Uint8Array(await readBuffer(device, groupsField.probeHistory, groupsField.probeHistory.size));
        const layeredBefore = await readTextureBytes(device, groupsField.layeredIrradiance, 8);
        const composedBefore = await readTextureBytes(device, groupsField.irradiance, 8);
        const markedFramesBefore = groups.ddgi.counters['frames with changes marked'];
        const fullComposesBefore = groups.ddgi.counters['frames composed in full'];

        // The weight change, in a frame that updates no probe.
        groups.ddgi.setGroupWeight(0, WEIGHT_A);
        groups.ddgi.setGroupWeight(1, WEIGHT_B);
        runFrame(device, groups.platform, 0);
        await device.queue.onSubmittedWorkDone();
        const afterChange = await views(device, groups, groupsField);
        const historyAfter = new Uint8Array(await readBuffer(device, groupsField.probeHistory, groupsField.probeHistory.size));
        const layeredAfter = await readTextureBytes(device, groupsField.layeredIrradiance, 8);
        const composedAfter = await readTextureBytes(device, groupsField.irradiance, 8);
        const changeCounters = { ...groups.ddgi.counters };
        // Frames after it, with probes updated at the new weights: the GPU's own mark count.
        const markedTotalBefore = groups.ddgi.counters['probes marked (total)'];
        await runFrames(device, groups.platform, 30, probes);
        await runFrames(device, groups.platform, 4, 0);
        const markedTotalAfter = groups.ddgi.counters['probes marked (total)'];
        const groupsMemory = groups.ddgi.memoryBytes;
        groups.platform.dispose();

        // ---- References, re-converged from scratch with groups off ----
        const reference = async (label: string, lampA: number, lampB: LightDescriptor): Promise<{ means: number[]; memory: number }> => {
            const scene = await createDdgiScene(device, 'two-room', {}, { gpuTiming: true });
            scene.platform.scene.updateLight(scene.lights[0], { intensity: lampA });
            scene.platform.scene.createLight(lampB);
            await runFrames(device, scene.platform, FRAMES, probes);
            const means = await views(device, scene, scene.ddgi.field!);
            const memory = scene.ddgi.memoryBytes;
            scene.platform.dispose();
            log(`${label}: ${means.map((mean) => mean.toPrecision(6)).join(' / ')}`);
            return { means, memory };
        };
        const lampAIntensity = groups.manifest.lights[0].intensity;
        const asTheyAre = await reference('groups off, lamps as they are', lampAIntensity, LAMP_B);
        const weighted = await reference('groups off, lamps dimmed and recoloured by the weights', lampAIntensity * WEIGHT_A[0], weightedLamp(LAMP_B as LightDescriptor & { type: 'point' }, WEIGHT_B));

        // 1. Linearity and the same frame.
        const cameras = groups.manifest.cameras.map((camera) => camera.name);
        const linearity = cameras.map((name, index) => ({
            bookmark: name,
            composedBefore: beforeChange[index],
            composedAfter: afterChange[index],
            reconverged: weighted.means[index],
            againstReconverged: relative(afterChange[index], weighted.means[index]),
            movedInItsFrame: relative(afterChange[index], beforeChange[index]),
            weightsOneAgainstGroupsOff: relative(beforeChange[index], asTheyAre.means[index]),
        }));
        for (const row of linearity) {
            log(
                `${row.bookmark}: weights changed ${percent(row.againstReconverged)} from the re-converged field (bar ${percent(LINEARITY_TOLERANCE)}), moved ${percent(row.movedInItsFrame)} in its frame; ` +
                    `weights 1 against groups off ${percent(row.weightsOneAgainstGroupsOff)}`,
            );
            if (!(Math.abs(row.againstReconverged) <= LINEARITY_TOLERANCE)) failures.push(`${row.bookmark}: the weighted image is ${percent(row.againstReconverged)} from the re-converged field`);
            if (!(Math.abs(row.movedInItsFrame) > SHOWS_THRESHOLD)) failures.push(`${row.bookmark}: the weight change moved the image only ${percent(row.movedInItsFrame)} in its frame`);
            if (!(Math.abs(row.weightsOneAgainstGroupsOff) <= LINEARITY_TOLERANCE)) failures.push(`${row.bookmark}: with weights 1 the layers add up to ${percent(row.weightsOneAgainstGroupsOff)} from groups off`);
        }
        results.linearity = linearity;

        // 2. A compose, not a light change.
        const composeOnly = {
            historyUnchanged: sameBytes(historyBefore, historyAfter),
            layersUnchanged: sameBytes(layeredBefore, layeredAfter),
            composedChanged: !sameBytes(composedBefore, composedAfter),
            changesMarkedInItsFrame: changeCounters['changes marked this frame'],
            framesWithChangesMarked: changeCounters['frames with changes marked'] - markedFramesBefore,
            fullComposes: changeCounters['frames composed in full'] - fullComposesBefore,
            probesMarkedAfter: markedTotalAfter - markedTotalBefore,
        };
        log(`the weight change's frame: ${JSON.stringify(composeOnly)}`);
        if (!composeOnly.historyUnchanged) failures.push('the weight change changed the probe history');
        if (!composeOnly.layersUnchanged) failures.push('the weight change changed the layered atlas');
        if (!composeOnly.composedChanged) failures.push('the weight change did not change the composed atlas');
        if (composeOnly.changesMarkedInItsFrame !== 0 || composeOnly.framesWithChangesMarked !== 0) failures.push('the weight change ran ddgi/mark');
        if (composeOnly.fullComposes !== 1) failures.push(`the weight change composed every tile ${composeOnly.fullComposes} times, not once`);
        if (composeOnly.probesMarkedAfter !== 0) failures.push(`${composeOnly.probesMarkedAfter} probes were marked in the frames after the weight change`);
        results.composeOnly = composeOnly;

        // 4. Both reductions give the same layered field.
        const fallback = await createDdgiScene(device, 'two-room', { lightGroups: defaultLightGroups(2) }, { technique: { reduction: 'workgroup', directWorkList: 'atomics' } });
        fallback.platform.scene.createLight(LAMP_B);
        await runFrames(device, fallback.platform, FALLBACK_FRAMES, probes);
        const fallbackAt300 = await atlasMean(device, fallback.ddgi.field!.layeredIrradiance);
        fallback.platform.dispose();
        const reductionDifference = relative(fallbackAt300, layeredAt300);
        log(`layered atlas after ${FALLBACK_FRAMES} frames: default ${layeredAt300.toPrecision(7)}, fallbacks ${fallbackAt300.toPrecision(7)} (${percent(reductionDifference)}; bar ${percent(REDUCTION_TOLERANCE)})`);
        if (!(Math.abs(reductionDifference) <= REDUCTION_TOLERANCE)) failures.push(`the fallback reductions' layered field is ${percent(reductionDifference)} from the default's`);
        results.reductions = { defaultMean: layeredAt300, fallbackMean: fallbackAt300, difference: reductionDifference, hasSubgroups: device.features.has('subgroups') };

        // Memory: 2 groups (3 layers and the composed atlas) against groups off.
        results.memory = { groupsOffBytes: asTheyAre.memory, twoGroupsBytes: groupsMemory, perExtraLayerBytes: (groupsMemory - asTheyAre.memory) / 2, probes };
        log(`memory: groups off ${(asTheyAre.memory / 1e6).toFixed(2)} MB, 2 groups ${(groupsMemory / 1e6).toFixed(2)} MB`);

        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
