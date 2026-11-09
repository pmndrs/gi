/**
 * T4.2 done-when 3: a scene with no flagged triangles gives the same field as
 * before materials reached the walk. This page only measures; the comparison
 * is between two runs of it, one on the commit before T4.2 and one after
 * (PROGRESS, T4.2), with the bar "means within 0.1%".
 *
 * Per scene (two-room and Sponza by default; `VITE_DDGI_SCENES=a,b`), as the
 * viewer starts it (three adapter, `startViewerScene`), tinybvh trees only so
 * both runs trace the same trees: CONVERGE_FRAMES frames at UNITS probes a
 * frame, then the irradiance atlas's mean (rgb over every texel) averaged
 * over MEAN_FRAMES more frames. Also reported: the meshes whose materials mix
 * see-through and blocking ones, the only ones the three adapter sends with
 * flaggable triangles (it leaves all-see-through meshes out of GI until T4.3),
 * and how many mesh flag rewrites the platform made (absent before T4.2).
 *
 * T4.2's second half: Sponza has alpha-tested materials (leaves, the
 * chains), which the walk now cuts out, so its field changes on purpose.
 * `VITE_DDGI_NO_CUTOUTS=1` sets every material's `alphaTest` to 0 before the
 * adapter's first sync, so the same scene without cutouts can be compared
 * across the commits; the materials it changed are reported either way.
 */

import { Mesh, type Material } from 'three';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import type { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { materialDescriptor } from '../../../../packages/ddgi-three/src/convert.ts';
import { assert } from '../../client.ts';
import { halfToFloat, readTextureBytes } from '../ddgi-shared/harness.ts';
import { startViewerScene, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
const SCENES = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES.trim() !== '' ? env.VITE_DDGI_SCENES.split(',').map((id) => id.trim()) : ['two-room', 'sponza'];
/** Probes a frame, capped by the scene's probe count. */
const UNITS = 1024;
const CONVERGE_FRAMES = 600;
/** Every material's alphaTest set to 0 before the adapter sees it (see the file comment). */
const NO_CUTOUTS = env.VITE_DDGI_NO_CUTOUTS === '1' || env.VITE_DDGI_NO_CUTOUTS === 'true';
const MEAN_FRAMES = 8;

/** Mean of rgb over every texel of the rgba16float irradiance atlas (borders included). */
async function atlasMean(device: GPUDevice, ddgi: DDGITechnique): Promise<number> {
    const field = ddgi.field;
    assert(field, 'no field');
    const halves = new Uint16Array((await readTextureBytes(device, field.irradiance, 8)).buffer);
    let sum = 0;
    for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
    return sum / ((halves.length / 4) * 3);
}

/** Meshes whose materials mix see-through and blocking ones, by the adapter's own rule. */
function mixedMeshes(scene: ViewerScene): string[] {
    const names: string[] = [];
    scene.loaded.scene.traverse((object) => {
        if (!(object instanceof Mesh)) return;
        const materials: Material[] = Array.isArray(object.material) ? object.material : [object.material];
        const blocks = materials.map((material) => materialDescriptor(material, () => undefined).blocksLight !== false);
        if (blocks.includes(true) && blocks.includes(false)) names.push(object.name || object.uuid);
    });
    return names;
}

async function frames(scene: ViewerScene, count: number, units: number): Promise<void> {
    for (let index = 0; index < count; index++) {
        scene.frame({ units });
        if (index % 8 === 7) await scene.device.queue.onSubmittedWorkDone();
    }
    await scene.device.queue.onSubmittedWorkDone();
}

await runThreePage(async ({ device, log, data }) => {
    const manifests = loadAllManifests();
    for (const sceneId of SCENES) {
        const manifest = manifests.find((entry) => entry.id === sceneId);
        assert(manifest, `no manifest '${sceneId}'`);
        // Alpha-tested materials, as loaded; with NO_CUTOUTS their alphaTest goes to 0 before the adapter sees them.
        const cutoutMaterials = new Set<Material>();
        const prepareScene = (loaded: ViewerScene['loaded']): void => {
            loaded.scene.traverse((object) => {
                if (!(object instanceof Mesh)) return;
                for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
                    if (material.alphaTest > 0) cutoutMaterials.add(material);
                }
            });
            if (NO_CUTOUTS) for (const material of cutoutMaterials) material.alphaTest = 0;
        };
        const scene = await startViewerScene(device, manifest, { gpuTiming: false, label: `field ${sceneId}`, meshTrees: { gpuFirst: false }, prepareScene });
        try {
            const units = Math.min(UNITS, scene.probeCount);
            await frames(scene, CONVERGE_FRAMES, units);
            const means: number[] = [];
            for (let index = 0; index < MEAN_FRAMES; index++) {
                await frames(scene, 1, units);
                means.push(await atlasMean(device, scene.ddgi));
            }
            const mean = means.reduce((sum, value) => sum + value, 0) / means.length;
            const mixed = mixedMeshes(scene);
            // Absent before T4.2: read without the type, so this page runs on the older commit too.
            const flagRewrites = (scene.platform.rays as { readonly triangleFlagRewrites?: number }).triangleFlagRewrites ?? null;
            const cutouts = `${cutoutMaterials.size} alpha-tested materials${NO_CUTOUTS ? ' (made opaque)' : ''}`;
            log(`${sceneId}: atlas mean ${mean.toPrecision(8)} over ${MEAN_FRAMES} frames after ${CONVERGE_FRAMES} at ${units} probes; ${mixed.length} mixed meshes; ${cutouts}; flag rewrites ${flagRewrites}`);
            data[sceneId] = {
                units,
                convergeFrames: CONVERGE_FRAMES,
                atlasMean: mean,
                means,
                mixedMeshes: mixed,
                cutoutMaterials: cutoutMaterials.size,
                cutoutsMadeOpaque: NO_CUTOUTS,
                flagRewrites,
                recompiles: scene.platform.stats.recompiles,
            };
        } finally {
            scene.dispose();
        }
    }
});
