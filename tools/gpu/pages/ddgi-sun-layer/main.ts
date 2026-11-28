/**
 * T2.6b, the sun layer (TASKS, "T2.6b, re-scoped to the sun layer";
 * docs/design/m2-lights.md §6.1): the correctness checks. The sun is traced
 * at a white reference in an irradiance layer of its own and weighed by its
 * colour × intensity, so a change of its colour or brightness is a new
 * weight, composed in the frame it happens.
 *
 * The scene is `courtyard-tree` as `createDdgiScene` builds it: its
 * procedural courtyard (10 m square, walls, one gateway) without the tree
 * model, lit by its sun and its procedural sky. Every probe is updated every
 * frame; the images are GI only (the engine-free test view's indirect image)
 * at the scene's three bookmarks. DDGI seeds its rays by frame number, so
 * two fields run for the same frames trace the same rays.
 *
 * Sections (`VITE_DDGI_SECTIONS`, default all):
 *
 * - **`colour`, done-when 1.** With the sun layer on, after FRAMES frames the
 *   sun changes colour and brightness (NEW_SUN) in a frame that updates no
 *   probe. In that same frame the image must match a field re-converged from
 *   scratch with the sun layer off and the sun at NEW_SUN: mean within 1% at
 *   every bookmark (as T2.5's check 1). The image must have moved by more
 *   than 1% somewhere: the change shows in its frame. That frame marks
 *   nothing and cuts no history: the probe history and the layered atlas are
 *   byte for byte what they were, the composed atlas changed, every tile was
 *   composed once, and the frames after it mark no probe (the GPU's count).
 *   Also: before the change the layers add up to the field with the sun
 *   layer off (within 1%), and the memory the sun layer costs.
 * - **`off`, done-when 2.** With the sun layer off (the default), the field
 *   after FRAMES frames matches `m2`'s (bcb1af8, measured with this section
 *   on that commit: M2_REFERENCE) within 0.1% mean: the irradiance atlas's
 *   mean and each bookmark's view.
 * - **`cap`, the 6-layer cap.** 4 light groups, the environment and the sun:
 *   6 layers, the most the blend holds (three lamps added in groups 1-3, the
 *   sun in group 0). After CAP_FRAMES frames the sun changes colour and
 *   group 0's weight halves in one frame that updates no probe; the image
 *   must match a field converged with groups and the sun layer off, the lamps
 *   as they are and the sun at NEW_SUN at half its intensity, within 1%. The
 *   declared fallbacks (phase B's and propagation's workgroup reductions,
 *   gi/direct's one-atomic-a-hit work list) give the same layered field
 *   (atlas mean within 0.1%, T2.2's bar for two compiled variants).
 *
 * Runs on the RTX and, for correctness, the Intel UHD (`VITE_DDGI_FRAMES` to
 * shorten it there; M2_REFERENCE is per GPU and frame count).
 */

import { assert, runGpuTest } from '../../client.ts';
import type { LightHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import type { ColorRGB } from '../../../../packages/core/src/platform/contract/math.ts';
import type { LightDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import type { DDGIField, DDGITechniqueOptions } from '../../../../packages/ddgi/src/ddgi-technique.ts';
import type { DDGISettingsInput } from '../../../../packages/ddgi/src/settings.ts';
import { createDdgiScene, halfToFloat, imageMean, readBuffer, readTextureBytes, releaseDeviceOnPageHide, runFrame, runFrames, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

const SCENE = 'courtyard-tree';
/** Frames each field converges for, every probe every frame. */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 600;
/** Frames for the 6-layer runs (half, as T2.5's reduction check). */
const CAP_FRAMES = Math.max(1, Math.round(FRAMES / 2));
const SECTIONS = new Set(typeof env.VITE_DDGI_SECTIONS === 'string' && env.VITE_DDGI_SECTIONS !== '' ? env.VITE_DDGI_SECTIONS.split(',').map((part) => part.trim()) : ['colour', 'off', 'cap']);
const WIDTH = 480;
const HEIGHT = 270;
/** Done-when 1: the image within 1% mean of a re-converged field. */
const LINEARITY_TOLERANCE = 0.01;
/** The change must show: the image moves by more than this in its frame (at one bookmark at least). */
const SHOWS_THRESHOLD = 0.01;
/** Done-when 2: the field with the sun layer off within 0.1% mean of m2's. */
const OFF_TOLERANCE = 0.001;
/** Two compiled variants of the kernels: atlas mean within 0.1% (T2.2, DECISIONS 2026-11-06). */
const REDUCTION_TOLERANCE = 0.001;
/** Frames run after the change, every probe, whose marks are counted. */
const AFTER_FRAMES = 30;

/** The sun after the change: an evening sun, half as bright and much redder (the manifest's is (1, 0.95, 0.85) × 3). */
const NEW_SUN = { color: [1, 0.6, 0.35] as ColorRGB, intensity: 1.5 };
/** Group 0's weight in the `cap` section's change (the sun is in group 0). */
const CAP_GROUP_WEIGHT: ColorRGB = [0.5, 0.5, 0.5];
/** Three lamps in groups 1-3 for the `cap` section: one layer each. */
const CAP_LAMPS: readonly LightDescriptor[] = [
    { type: 'point', color: [1, 0.8, 0.6], intensity: 3, position: [3, 1.5, 3], range: 8, group: 1, label: 'lamp-1' },
    { type: 'point', color: [0.7, 0.85, 1], intensity: 3, position: [-3, 1.5, 3], range: 8, group: 2, label: 'lamp-2' },
    { type: 'point', color: [0.8, 1, 0.8], intensity: 3, position: [0, 3, -3], range: 8, group: 3, label: 'lamp-3' },
];

/**
 * `off` at m2 (bcb1af8), per GPU vendor and frame count: the irradiance
 * atlas's mean and each bookmark's GI-only view mean after FRAMES frames
 * with the default settings. Measured with this page's `off` section on that
 * commit (PROGRESS 2026-11-07, T2.6b).
 */
const M2_REFERENCE: Readonly<Record<string, Readonly<Record<number, { atlas: number; views: readonly number[] }>>>> = {
    // RTX 3060 Laptop, Chrome headless, 600 frames: 4,840 probes, 30.4 MB of DDGI.
    nvidia: { 600: { atlas: 0.23075826089244245, views: [0.04927994447835978, 0.05041530817066829, 0.046508910088438674] } },
};

/** Mean of rgb over every texel of an rgba16float atlas. */
async function atlasMean(device: GPUDevice, texture: GPUTexture): Promise<number> {
    const halves = new Uint16Array((await readTextureBytes(device, texture, 8)).buffer);
    let sum = 0;
    for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
    return sum / ((halves.length / 4) * 3);
}

async function views(device: GPUDevice, scene: DdgiScene, field: DDGIField): Promise<number[]> {
    const view = await TestView.create(device, scene.platform);
    const means: number[] = [];
    for (const camera of scene.manifest.cameras) means.push(imageMean((await view.render(field, camera, WIDTH, HEIGHT)).indirect));
    return means;
}

function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
    if (a.length !== b.length) return false;
    for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return false;
    return true;
}

const relative = (value: number, reference: number): number => value / reference - 1;
const percent = (share: number): string => `${(share * 100).toFixed(3)}%`;

/** The courtyard with DDGI made with `settings`; its sun is the manifest's first light. */
async function courtyard(device: GPUDevice, settings: Partial<DDGISettingsInput>, technique?: DDGITechniqueOptions): Promise<{ scene: DdgiScene; sun: LightHandle; probes: number }> {
    const scene = await createDdgiScene(device, SCENE, settings, { gpuTiming: true, technique });
    const sun = scene.lights[0];
    assert(scene.manifest.lights[0].type === 'directional', `${SCENE}'s first light is not its sun`);
    return { scene, sun, probes: scene.ddgi.field!.grid.probeCount };
}

await runGpuTest(
    async ({ adapter, device, log }) => {
        releaseDeviceOnPageHide(device);
        const vendor = /nvidia/i.test(adapter.info.vendor) ? 'nvidia' : /intel/i.test(adapter.info.vendor) ? 'intel' : adapter.info.vendor;
        const failures: string[] = [];
        const results: Record<string, unknown> = { frames: FRAMES, vendor, sections: [...SECTIONS] };

        // ---- off: done-when 2 (runs first, so it also runs on m2's code, where the other sections can't) ----
        let offViews: number[] | null = null;
        let offMemory = 0;
        if (SECTIONS.has('off') || SECTIONS.has('colour')) {
            const { scene, probes } = await courtyard(device, {});
            await runFrames(device, scene.platform, FRAMES, probes);
            const field = scene.ddgi.field!;
            const atlas = await atlasMean(device, field.irradiance);
            offViews = await views(device, scene, field);
            offMemory = scene.ddgi.memoryBytes;
            scene.platform.dispose();
            const reference = M2_REFERENCE[vendor]?.[FRAMES] ?? null;
            const atlasChange = reference ? relative(atlas, reference.atlas) : null;
            const viewChanges = reference ? offViews.map((mean, index) => relative(mean, reference.views[index])) : null;
            log(
                `sun layer off: atlas mean ${atlas.toPrecision(9)}, views ${offViews.map((mean) => mean.toPrecision(9)).join(' / ')}, ${probes} probes` +
                    (reference ? ` (m2: ${percent(atlasChange!)} / ${viewChanges!.map(percent).join(' / ')}; bar ${percent(OFF_TOLERANCE)})` : ` (no m2 number for ${vendor} at ${FRAMES} frames)`),
            );
            if (SECTIONS.has('off')) {
                if (atlasChange !== null && !(Math.abs(atlasChange) <= OFF_TOLERANCE)) failures.push(`sun layer off: the atlas mean is ${percent(atlasChange)} from m2's`);
                viewChanges?.forEach((change, index) => {
                    if (!(Math.abs(change) <= OFF_TOLERANCE)) failures.push(`sun layer off: bookmark ${index + 1} is ${percent(change)} from m2's`);
                });
            }
            results.off = { atlasMean: atlas, viewMeans: offViews, m2: reference, atlasChange, viewChanges, probes, memoryBytes: offMemory };
        }

        // ---- colour: done-when 1 ----
        if (SECTIONS.has('colour') && offViews) {
            const { scene, sun, probes } = await courtyard(device, { sunLayer: true });
            const ddgi = scene.ddgi;
            assert(ddgi.counters['irradiance layers'] === 2 && ddgi.counters['sun layer'] === 1, `expected the sun layer as the second of 2 layers, got ${ddgi.counters['irradiance layers']}`);
            await runFrames(device, scene.platform, FRAMES, probes);
            const field = ddgi.field!;
            const before = await views(device, scene, field);
            const historyBefore = new Uint8Array(await readBuffer(device, field.probeHistory, field.probeHistory.size));
            const layeredBefore = await readTextureBytes(device, field.layeredIrradiance, 8);
            const composedBefore = await readTextureBytes(device, field.irradiance, 8);
            const markedFramesBefore = ddgi.counters['frames with changes marked'];
            const fullComposesBefore = ddgi.counters['frames composed in full'];

            // The sun's new colour and brightness, in a frame that updates no probe.
            scene.platform.scene.updateLight(sun, NEW_SUN);
            runFrame(device, scene.platform, 0);
            await device.queue.onSubmittedWorkDone();
            const after = await views(device, scene, field);
            const historyAfter = new Uint8Array(await readBuffer(device, field.probeHistory, field.probeHistory.size));
            const layeredAfter = await readTextureBytes(device, field.layeredIrradiance, 8);
            const composedAfter = await readTextureBytes(device, field.irradiance, 8);
            const changeCounters = { ...ddgi.counters };
            // Frames after it, every probe updated with the new sun: the GPU's own mark count.
            const markedTotalBefore = ddgi.counters['probes marked (total)'];
            await runFrames(device, scene.platform, AFTER_FRAMES, probes);
            await runFrames(device, scene.platform, 4, 0);
            const markedTotalAfter = ddgi.counters['probes marked (total)'];
            const sunLayerMemory = ddgi.memoryBytes;
            scene.platform.dispose();

            // The reference: re-converged from scratch, sun layer off, the sun at its new colour from the start.
            const reference = await courtyard(device, {});
            reference.scene.platform.scene.updateLight(reference.sun, NEW_SUN);
            await runFrames(device, reference.scene.platform, FRAMES, reference.probes);
            const reconverged = await views(device, reference.scene, reference.scene.ddgi.field!);
            reference.scene.platform.dispose();

            const rows = scene.manifest.cameras.map((camera, index) => ({
                bookmark: camera.name,
                before: before[index],
                after: after[index],
                reconverged: reconverged[index],
                againstReconverged: relative(after[index], reconverged[index]),
                movedInItsFrame: relative(after[index], before[index]),
                layersAgainstOff: relative(before[index], offViews![index]),
            }));
            for (const row of rows) {
                log(
                    `${row.bookmark}: the sun's change ${percent(row.againstReconverged)} from the re-converged field (bar ${percent(LINEARITY_TOLERANCE)}), moved ${percent(row.movedInItsFrame)} in its frame; ` +
                        `before it, the layers ${percent(row.layersAgainstOff)} from the sun layer off`,
                );
                if (!(Math.abs(row.againstReconverged) <= LINEARITY_TOLERANCE)) failures.push(`${row.bookmark}: after the sun's change the image is ${percent(row.againstReconverged)} from the re-converged field`);
                if (!(Math.abs(row.layersAgainstOff) <= LINEARITY_TOLERANCE)) failures.push(`${row.bookmark}: the layers add up to ${percent(row.layersAgainstOff)} from the sun layer off`);
            }
            const largestMove = Math.max(...rows.map((row) => Math.abs(row.movedInItsFrame)));
            if (!(largestMove > SHOWS_THRESHOLD)) failures.push(`the sun's change moved the image at most ${percent(largestMove)} in its frame`);

            const composeOnly = {
                historyUnchanged: sameBytes(historyBefore, historyAfter),
                layersUnchanged: sameBytes(layeredBefore, layeredAfter),
                composedChanged: !sameBytes(composedBefore, composedAfter),
                changesMarkedInItsFrame: changeCounters['changes marked this frame'],
                framesWithChangesMarked: changeCounters['frames with changes marked'] - markedFramesBefore,
                fullComposes: changeCounters['frames composed in full'] - fullComposesBefore,
                probesMarkedAfter: markedTotalAfter - markedTotalBefore,
            };
            log(`the sun change's frame: ${JSON.stringify(composeOnly)}`);
            if (!composeOnly.historyUnchanged) failures.push("the sun's change changed the probe history");
            if (!composeOnly.layersUnchanged) failures.push("the sun's change changed the layered atlas");
            if (!composeOnly.composedChanged) failures.push("the sun's change did not change the composed atlas");
            if (composeOnly.changesMarkedInItsFrame !== 0 || composeOnly.framesWithChangesMarked !== 0) failures.push("the sun's change ran ddgi/mark");
            if (composeOnly.fullComposes !== 1) failures.push(`the sun's change composed every tile ${composeOnly.fullComposes} times, not once`);
            if (composeOnly.probesMarkedAfter !== 0) failures.push(`${composeOnly.probesMarkedAfter} probes were marked in the frames after the sun's change`);
            const memory = { sunLayerOffBytes: offMemory, sunLayerBytes: sunLayerMemory, extraBytes: sunLayerMemory - offMemory, probes };
            log(`memory: sun layer off ${(offMemory / 1e6).toFixed(2)} MB, on ${(sunLayerMemory / 1e6).toFixed(2)} MB (+${((sunLayerMemory - offMemory) / 1e6).toFixed(2)} MB)`);
            results.colour = { rows, largestMove, composeOnly, memory };
        }

        // ---- cap: 4 groups, the environment and the sun (6 layers) ----
        if (SECTIONS.has('cap')) {
            const settings: Partial<DDGISettingsInput> = { lightGroups: { count: 4, weights: [0, 1, 2, 3].map(() => [1, 1, 1] as ColorRGB), environmentWeight: [1, 1, 1] }, sunLayer: true };
            const six = await courtyard(device, settings);
            for (const lamp of CAP_LAMPS) six.scene.platform.scene.createLight(lamp);
            const sixDdgi = six.scene.ddgi;
            assert(sixDdgi.counters['irradiance layers'] === 6, `expected 6 layers, got ${sixDdgi.counters['irradiance layers']}`);
            await runFrames(device, six.scene.platform, CAP_FRAMES, six.probes);
            const sixField = sixDdgi.field!;
            const layeredMean = await atlasMean(device, sixField.layeredIrradiance);
            sixDdgi.setGroupWeight(0, CAP_GROUP_WEIGHT);
            six.scene.platform.scene.updateLight(six.sun, NEW_SUN);
            runFrame(device, six.scene.platform, 0);
            await device.queue.onSubmittedWorkDone();
            const sixAfter = await views(device, six.scene, sixField);
            const sixMemory = sixDdgi.memoryBytes;
            six.scene.platform.dispose();

            // The reference: groups and the sun layer off, the lamps as they are, the sun at its new colour × group 0's weight.
            const reference = await courtyard(device, {});
            for (const lamp of CAP_LAMPS) reference.scene.platform.scene.createLight(lamp);
            reference.scene.platform.scene.updateLight(reference.sun, { color: NEW_SUN.color, intensity: NEW_SUN.intensity * CAP_GROUP_WEIGHT[0] });
            await runFrames(device, reference.scene.platform, CAP_FRAMES, reference.probes);
            const referenceViews = await views(device, reference.scene, reference.scene.ddgi.field!);
            reference.scene.platform.dispose();

            // The declared fallbacks give the same layered field.
            const fallback = await courtyard(device, settings, { reduction: 'workgroup', directWorkList: 'atomics' });
            for (const lamp of CAP_LAMPS) fallback.scene.platform.scene.createLight(lamp);
            await runFrames(device, fallback.scene.platform, CAP_FRAMES, fallback.probes);
            const fallbackMean = await atlasMean(device, fallback.scene.ddgi.field!.layeredIrradiance);
            fallback.scene.platform.dispose();

            const rows = six.scene.manifest.cameras.map((camera, index) => ({ bookmark: camera.name, sixLayers: sixAfter[index], reference: referenceViews[index], difference: relative(sixAfter[index], referenceViews[index]) }));
            for (const row of rows) {
                log(`6 layers, ${row.bookmark}: the sun's change and group 0 at half ${percent(row.difference)} from a field converged that way (bar ${percent(LINEARITY_TOLERANCE)})`);
                if (!(Math.abs(row.difference) <= LINEARITY_TOLERANCE)) failures.push(`6 layers, ${row.bookmark}: ${percent(row.difference)} from the converged field`);
            }
            const reductionDifference = relative(fallbackMean, layeredMean);
            log(`6 layers, layered atlas after ${CAP_FRAMES} frames: default ${layeredMean.toPrecision(7)}, fallbacks ${fallbackMean.toPrecision(7)} (${percent(reductionDifference)}; bar ${percent(REDUCTION_TOLERANCE)})`);
            if (!(Math.abs(reductionDifference) <= REDUCTION_TOLERANCE)) failures.push(`6 layers: the fallbacks' layered field is ${percent(reductionDifference)} from the default's`);
            log(`6 layers: ${(sixMemory / 1e6).toFixed(2)} MB`);
            results.cap = { rows, reductions: { defaultMean: layeredMean, fallbackMean, difference: reductionDifference, hasSubgroups: device.features.has('subgroups') }, memoryBytes: sixMemory };
        }

        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
