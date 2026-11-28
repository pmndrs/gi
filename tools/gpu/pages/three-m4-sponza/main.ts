/**
 * M4, T4.3 done-when 4: Sponza (Intel New Sponza with the curtains) through
 * the three adapter, as the viewer starts it, with real textures.
 *
 * - **Traced, no mesh left out:** every mesh of the model, the `dirt_decal`
 *   meshes and the curtains included, is sent into GI's rays
 *   (`SceneSync.sentContributes`). Before T4.3 the adapter left out the
 *   meshes whose materials all let light through (the decal).
 * - **Real textures:** the materials are sent with their maps as textures
 *   (no averages), and the platform built a store layer for each.
 * - **The field:** the irradiance atlas's mean, exactly as
 *   `ddgi-field-unchanged` measures it (tinybvh trees, 600 frames at 1,024
 *   probes, then the mean over 8 frames), so it compares with that page run
 *   on `m4` before T4.3 (PROGRESS, T4.3). Reported, not gated: the change is
 *   meant to move it (textures were applied twice before).
 * - The adapter's still-frame CPU cost (`SceneSync.update` on a scene that
 *   doesn't change), noted, not gated.
 *
 * Manual: needs the Sponza download.
 */

import { Mesh, type Material } from 'three';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import type { DDGITechnique } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { assert } from '../../client.ts';
import { halfToFloat, readTextureBytes } from '../ddgi-shared/harness.ts';
import { startViewerScene, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

/** As ddgi-field-unchanged, so the means compare. */
const UNITS = 1024;
const CONVERGE_FRAMES = 600;
const MEAN_FRAMES = 8;
const STILL_FRAMES = 200;

/** Mean of rgb over every texel of the rgba16float irradiance atlas (borders included); ddgi-field-unchanged's. */
async function atlasMean(device: GPUDevice, ddgi: DDGITechnique): Promise<number> {
    const field = ddgi.field;
    assert(field, 'no field');
    const halves = new Uint16Array((await readTextureBytes(device, field.irradiance, 8)).buffer);
    let sum = 0;
    for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
    return sum / ((halves.length / 4) * 3);
}

async function frames(scene: ViewerScene, count: number, units: number): Promise<void> {
    for (let index = 0; index < count; index++) {
        scene.frame({ units });
        if (index % 8 === 7) await scene.device.queue.onSubmittedWorkDone();
    }
    await scene.device.queue.onSubmittedWorkDone();
}

function materialsOf(mesh: Mesh): Material[] {
    return Array.isArray(mesh.material) ? mesh.material : [mesh.material];
}

await runThreePage(async ({ device, log, data }) => {
    const manifest = loadAllManifests().find((entry) => entry.id === 'sponza');
    assert(manifest, 'no sponza manifest');
    const scene = await startViewerScene(device, manifest, { gpuTiming: false, label: 'three-m4 sponza', meshTrees: { gpuFirst: false } });
    try {
        const sync = scene.threeGI.sync;
        // Every mesh of the model, and the ones that matter here.
        const meshes: Mesh[] = [];
        scene.loaded.content.traverse((object) => {
            if (object instanceof Mesh) meshes.push(object);
        });
        const leftOut = meshes.filter((mesh) => sync.sentContributes(mesh) === false).map((mesh) => mesh.name);
        const notSent = meshes.filter((mesh) => sync.sentContributes(mesh) === undefined && mesh.visible).map((mesh) => mesh.name);
        const named = (pattern: RegExp) => meshes.filter((mesh) => pattern.test(mesh.name) || materialsOf(mesh).some((material) => pattern.test(material.name)));
        const decals = named(/decal/i);
        const curtains = named(/curtain/i);
        log(`${meshes.length} meshes; ${decals.length} decal and ${curtains.length} curtain meshes; left out of GI's rays: ${leftOut.length}; not mirrored: ${notSent.length}`);
        assert(decals.length > 0 && curtains.length > 0, `the decal (${decals.length}) or the curtains (${curtains.length}) weren't found`);
        for (const mesh of [...decals, ...curtains]) assert(sync.sentContributes(mesh) === true, `mesh '${mesh.name}' is not in GI's rays (${sync.sentContributes(mesh)})`);
        assert(leftOut.length === 0, `meshes left out of GI's rays: ${leftOut.join(', ')}`);

        // Materials and textures as sent.
        // Each distinct material of the meshes, once.
        const describe = (group: readonly Mesh[]) =>
            [...new Set(group.flatMap(materialsOf))].map((material) => {
                const sent = sync.sentMaterial(material);
                assert(sent, `material '${material.name}' wasn't sent`);
                const layer = sent.albedoTexture !== undefined ? scene.platform.sceneCopy.textureLayer(sent.albedoTexture) : null;
                return { material: material.name, alphaMode: sent.alphaMode, opacity: sent.opacity ?? null, alphaCutoff: sent.alphaCutoff ?? null, blocksLight: sent.blocksLight, albedo: sent.albedo, metalness: sent.metalness ?? null, albedoLayer: layer, metalnessMap: sent.metalnessTexture !== undefined };
            });
        const decalMaterials = describe(decals);
        const curtainMaterials = describe(curtains);
        log(`decal meshes' materials: ${JSON.stringify(decalMaterials)}`);
        log(`curtain meshes' materials: ${JSON.stringify(curtainMaterials)}`);
        for (const material of [...decalMaterials, ...curtainMaterials]) assert(material.albedoLayer !== null, `material '${material.material}' has no albedo layer in the store`);
        const textureCounts = scene.platform.sceneCopy.textureCounts();
        const counts = sync.counts;
        log(`textures sent ${counts.textures} for ${counts.materials} materials; store layers built ${textureCounts.built}, pending ${textureCounts.pending}, refused ${textureCounts.failed}; store ${(scene.platform.stats.memory.textures / 1e6).toFixed(1)} MB`);
        assert(counts.textures > 0 && textureCounts.built === counts.textures && textureCounts.failed === 0, `not every texture became a layer: ${JSON.stringify(textureCounts)} of ${counts.textures}`);

        // The field, as ddgi-field-unchanged measures it.
        const units = Math.min(UNITS, scene.probeCount);
        await frames(scene, CONVERGE_FRAMES, units);
        const means: number[] = [];
        for (let index = 0; index < MEAN_FRAMES; index++) {
            await frames(scene, 1, units);
            means.push(await atlasMean(device, scene.ddgi));
        }
        const mean = means.reduce((sum, value) => sum + value, 0) / means.length;
        log(`atlas mean ${mean.toPrecision(8)} over ${MEAN_FRAMES} frames after ${CONVERGE_FRAMES} at ${units} probes`);

        // The adapter's CPU time on still frames (the scene doesn't change).
        // (Timed over the whole run: the page's clock is coarse, 0.1 ms, against a frame's cost.)
        const stillStart = performance.now();
        for (let index = 0; index < STILL_FRAMES; index++) sync.update(scene.loaded.scene);
        const stillFrameMs = (performance.now() - stillStart) / STILL_FRAMES;
        log(`adapter still frame: ${stillFrameMs.toFixed(3)} ms on average over ${STILL_FRAMES}`);

        Object.assign(data, {
            meshes: meshes.length,
            decalMeshes: decals.map((mesh) => mesh.name),
            curtainMeshes: curtains.length,
            leftOut,
            decalMaterials,
            curtainMaterials,
            texturesSent: counts.textures,
            materials: counts.materials,
            textureCounts,
            storeMegabytes: scene.platform.stats.memory.textures / 1e6,
            units,
            convergeFrames: CONVERGE_FRAMES,
            atlasMean: mean,
            means,
            stillFrameMs,
            recompiles: scene.platform.stats.recompiles,
        });
    } finally {
        scene.dispose();
    }
});
