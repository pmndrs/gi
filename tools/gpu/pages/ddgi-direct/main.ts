/**
 * T2.2 checks 1 and 2 on the DDGI field: with direct light deferred to the
 * platform's `gi/direct`, the field after FRAMES frames matches M1's inline
 * path, and the subgroup work list gives the same field as its declared
 * fallback. A correctness check: nothing here is timed.
 *
 * Per scene (`VITE_DDGI_SCENES`, default cornell-box-point and two-room,
 * lit by a point light and a lamp; `sponza` needs the downloads and is lit by
 * the sun), one fresh platform per path, started as the viewer starts it
 * (ddgi-shared/viewer-scene.ts), then FRAMES frames of a fixed batch
 * (BATCH_PROBES, or every probe when there are fewer: several 1,024-probe
 * rounds of the deferred trace a frame), `deltaTime` 1/60 s:
 *
 * - `inline`: M1's `ddgi/trace`, which traces the shadow rays itself;
 * - `deferred-subgroups` (devices with subgroups) and `deferred-atomics`.
 *
 * Both paths light a ray with the same random numbers (`giDirectRandom`, by
 * the ray's index and the frame), so the fields differ only by what the
 * surface record and the second half-float store round, not by ray noise,
 * and can be compared texel by texel:
 *
 * - **the irradiance atlas, per texel:** over every interior texel brighter
 *   than TEXEL_FLOOR × the atlas mean, the mean and p95 of
 *   |deferred − inline| / inline (luminance), because a mean can hide errors
 *   that cancel (TASKS, T2.2);
 * - **the atlas mean** and the engine-free test view's GI image mean per
 *   camera bookmark, deferred against inline (MAX_MEAN_DIFFERENCE);
 * - **check 2:** `deferred-subgroups` against `deferred-atomics`, atlas mean
 *   within 0.1% (MAX_WORK_LIST_DIFFERENCE). `gi/direct`'s random numbers
 *   follow the ray, not the list's order, so they should be identical.
 */

import { DDGITechnique, type DDGITechniqueOptions } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { assert } from '../../client.ts';
import { halfToFloat, imageMean, readTextureBytes, TestView } from '../ddgi-shared/harness.ts';
import { startViewerScene, yieldToCallbacks } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
const SCENES = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES.trim() !== '' ? env.VITE_DDGI_SCENES.split(',').map((id) => id.trim()) : ['cornell-box-point', 'two-room'];

/** The done-when's frame count. */
const FRAMES = 300;
/** Probes per frame: the technique's default maximum (4 rounds of the deferred trace). */
const BATCH_PROBES = 4096;
/** Check 1's bar, on the field's means: within 0.5% of inline. */
const MAX_MEAN_DIFFERENCE = 0.005;
/** The per-texel bars (DECISIONS 2026-11-06, T2.2): mean and p95 of the per-texel relative difference. */
const MAX_TEXEL_MEAN = 0.002;
const MAX_TEXEL_P95 = 0.005;
/** Texels darker than this share of the atlas mean are left out of the per-texel numbers (relative differences of near-black texels mean little). */
const TEXEL_FLOOR = 0.01;
/** Check 2's bar: the two work lists' fields within 0.1% (mean). */
const MAX_WORK_LIST_DIFFERENCE = 0.001;
/** The test view's size: the flicker and settle pages' 640 × 360. */
const VIEW_WIDTH = 640;
const VIEW_HEIGHT = 360;
/** rgba16float: 4 halves a texel. */
const IRRADIANCE_TEXEL_BYTES = 8;

interface PathUnderTest {
    readonly name: string;
    readonly options: DDGITechniqueOptions;
}

interface PathResult {
    /** The GI image's mean per camera bookmark, in bookmark order. */
    readonly imageMeans: number[];
    /** Every irradiance atlas texel's luminance (borders included; they copy interior texels). */
    readonly texels: Float32Array;
    readonly atlasMean: number;
    readonly directHits: number | undefined;
}

/** Luminance of every texel of an rgba16float array texture. */
async function atlasLuminance(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
    const halves = new Uint16Array((await readTextureBytes(device, texture, IRRADIANCE_TEXEL_BYTES)).buffer);
    const texels = new Float32Array(halves.length / 4);
    for (let texel = 0; texel < texels.length; texel++) {
        texels[texel] = 0.2126 * halfToFloat(halves[texel * 4]) + 0.7152 * halfToFloat(halves[texel * 4 + 1]) + 0.0722 * halfToFloat(halves[texel * 4 + 2]);
    }
    return texels;
}

function mean(values: Float32Array): number {
    let sum = 0;
    for (const value of values) sum += value;
    return sum / values.length;
}

function relativeDifference(value: number, reference: number): number {
    return reference === 0 ? (value === 0 ? 0 : Infinity) : value / reference - 1;
}

/** Per texel: the mean, p95 and largest |found − reference| / reference over texels above the floor. */
function texelDifferences(found: Float32Array, reference: Float32Array, referenceMean: number): { mean: number; p95: number; max: number; texels: number } {
    const floor = TEXEL_FLOOR * referenceMean;
    const differences: number[] = [];
    for (let texel = 0; texel < reference.length; texel++) {
        if (reference[texel] > floor) differences.push(Math.abs(found[texel] - reference[texel]) / reference[texel]);
    }
    differences.sort((a, b) => a - b);
    const at = (share: number): number => differences[Math.min(differences.length - 1, Math.floor(share * differences.length))] ?? 0;
    return { mean: differences.reduce((sum, value) => sum + value, 0) / Math.max(1, differences.length), p95: at(0.95), max: at(1), texels: differences.length };
}

await runThreePage(async ({ device, data, log }) => {
    const manifests = loadAllManifests();
    const paths: PathUnderTest[] = [
        { name: 'inline', options: { directLight: 'inline' } },
        ...(device.features.has('subgroups') ? [{ name: 'deferred-subgroups', options: { directLight: 'deferred', directWorkList: 'subgroups' } } as const] : []),
        { name: 'deferred-atomics', options: { directLight: 'deferred', directWorkList: 'atomics' } },
    ];
    data.frames = FRAMES;
    data.paths = paths.map((path) => path.name);
    const results: Record<string, unknown> = {};
    data.scenes = results;
    const failures: string[] = [];

    for (const id of SCENES) {
        const manifest = manifests.find((candidate) => candidate.id === id);
        assert(manifest, `no manifest '${id}'`);
        const byPath = new Map<string, PathResult>();
        let units = 0;
        for (const path of paths) {
            const scene = await startViewerScene(device, manifest, { gpuTiming: true, label: `direct ${id} ${path.name}`, technique: path.options });
            assert(scene.ddgi instanceof DDGITechnique, 'not a DDGI technique');
            units = Math.min(scene.probeCount, BATCH_PROBES);
            for (let index = 0; index < FRAMES; index++) {
                scene.frame({ units });
                // Let the GPU catch up now and then, so the page never sits in one long task.
                if (index % 8 === 7) {
                    await device.queue.onSubmittedWorkDone();
                    await yieldToCallbacks();
                }
            }
            // The census arrives a few frames late: a few frames with no updates let the last counts in.
            for (let index = 0; index < 4; index++) {
                scene.frame({ units: 0 });
                await device.queue.onSubmittedWorkDone();
                await yieldToCallbacks();
            }
            const field = scene.ddgi.field;
            assert(field, 'DDGI has no field');
            const view = await TestView.create(device, scene.platform);
            const imageMeans: number[] = [];
            for (const camera of manifest.cameras) imageMeans.push(imageMean((await view.render(field, camera, VIEW_WIDTH, VIEW_HEIGHT)).indirect));
            const texels = await atlasLuminance(device, field.irradiance);
            const result: PathResult = { imageMeans, texels, atlasMean: mean(texels), directHits: scene.ddgi.counters['hits lit by gi/direct'] };
            byPath.set(path.name, result);
            log(`${id} ${path.name}: image means ${imageMeans.map((value) => value.toPrecision(6)).join(', ')}; atlas mean ${result.atlasMean.toPrecision(6)} (${units} probes a frame, ${FRAMES} frames); hits lit by gi/direct in the last frame ${result.directHits ?? 'none (inline)'}`);
            scene.dispose();
            await device.queue.onSubmittedWorkDone();
        }

        const inline = byPath.get('inline')!;
        const sceneResult: Record<string, unknown> = { probes_per_frame: units, inline_atlas_mean: inline.atlasMean };
        inline.imageMeans.forEach((value, camera) => (sceneResult[`inline_image_mean_camera${camera + 1}`] = value));
        for (const [name, result] of byPath) {
            if (name === 'inline') continue;
            const atlasDifference = relativeDifference(result.atlasMean, inline.atlasMean);
            const perTexel = texelDifferences(result.texels, inline.texels, inline.atlasMean);
            Object.assign(sceneResult, {
                [`${name}_atlas_mean`]: result.atlasMean,
                [`${name}_atlas_vs_inline`]: atlasDifference,
                [`${name}_texel_mean_difference`]: perTexel.mean,
                [`${name}_texel_p95_difference`]: perTexel.p95,
                [`${name}_texel_max_difference`]: perTexel.max,
                [`${name}_texels_compared`]: perTexel.texels,
                [`${name}_hits_last_frame`]: result.directHits,
            });
            log(`${id} ${name} against inline: atlas mean ${(atlasDifference * 100).toFixed(4)}%; per texel mean ${(perTexel.mean * 100).toFixed(4)}%, p95 ${(perTexel.p95 * 100).toFixed(4)}%, max ${(perTexel.max * 100).toFixed(3)}% over ${perTexel.texels} texels`);
            if (!(Math.abs(atlasDifference) <= MAX_MEAN_DIFFERENCE)) failures.push(`${id} ${name}: atlas mean ${(atlasDifference * 100).toFixed(3)}% from inline`);
            if (!(perTexel.mean <= MAX_TEXEL_MEAN)) failures.push(`${id} ${name}: per-texel mean difference ${(perTexel.mean * 100).toFixed(3)}% from inline`);
            if (!(perTexel.p95 <= MAX_TEXEL_P95)) failures.push(`${id} ${name}: per-texel p95 difference ${(perTexel.p95 * 100).toFixed(3)}% from inline`);
            result.imageMeans.forEach((value, camera) => {
                const difference = relativeDifference(value, inline.imageMeans[camera]);
                sceneResult[`${name}_image_vs_inline_camera${camera + 1}`] = difference;
                if (!(Math.abs(difference) <= MAX_MEAN_DIFFERENCE)) failures.push(`${id} ${name}: camera ${camera + 1} image mean ${(difference * 100).toFixed(3)}% from inline`);
            });
        }
        // Check 2: the subgroup work list against its declared fallback.
        const subgroups = byPath.get('deferred-subgroups');
        const atomics = byPath.get('deferred-atomics');
        if (subgroups && atomics) {
            const difference = relativeDifference(subgroups.atlasMean, atomics.atlasMean);
            const perTexel = texelDifferences(subgroups.texels, atomics.texels, atomics.atlasMean);
            Object.assign(sceneResult, { subgroups_vs_atomics_atlas: difference, subgroups_vs_atomics_texel_max: perTexel.max });
            log(`${id} subgroups against atomics: atlas mean ${(difference * 100).toFixed(5)}%, largest per-texel difference ${(perTexel.max * 100).toFixed(5)}%`);
            if (!(Math.abs(difference) <= MAX_WORK_LIST_DIFFERENCE)) failures.push(`${id}: the subgroup work list's field is ${(difference * 100).toFixed(4)}% from the fallback's`);
        }
        results[id] = sceneResult;
    }
    assert(failures.length === 0, failures.join('; '));
});
