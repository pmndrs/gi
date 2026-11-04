/**
 * T1.7: the three adapter's CPU time per frame, on a real scene in the
 * browser. Default: `sponza` (Intel's New Sponza with the curtains);
 * `VITE_DDGI_SCENES=bistro-interior` (or another manifest id) picks another.
 *
 * The scene is the viewer's own (`apps/viewer/src/scene/load-scene.ts`,
 * GLTFLoader with textures), mirrored by `ThreeGI` into a platform running
 * DDGI on the manifest's volume, rendered by three each frame at 256².
 *
 * What is timed is the adapter's own work: `SceneSync.update`, the walk over
 * the scene that compares world matrices, materials, geometry versions and
 * lights and sends what changed. Reported separately, not part of it: three's
 * own `updateMatrixWorld` (which `render` does; ThreeGI doesn't repeat it by
 * default) and the platform's `encode`.
 *
 * Two ways, for each of a still scene and one object moving every frame:
 * - **per frame:** `threeGI.timings.syncMs` over `FRAMES` real frames
 *   (median and p95). `performance.now()` may step in 0.1 ms here (NOTES),
 *   which is coarse next to the targets;
 * - **batched:** the mean of `BATCH` back-to-back `sync.update` calls, five
 *   times, median of the five. Each call is a full frame's sync (for the
 *   moving case the object moves before every call), so this is the same
 *   work, measured below the timer's step, but with warm caches: a real
 *   frame's sync starts cold after three's render and reads 2-4x slower.
 * Both must meet the targets.
 *
 * Targets (TASKS T1.7): still ≤ 0.2 ms, one object moving ≤ 0.5 ms.
 *
 * A timing run takes the bench lock:
 *   node tools/reference/bench-lock.ts run --what "T1.7 adapter timing" -- pnpm test:gpu --gpu nvidia --filter three-adapter-timing
 */

import { Box3, PerspectiveCamera, Vector3, type Mesh, type Object3D } from 'three/webgpu';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import type { EnvironmentDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const FRAMES = 600;
const WARMUP_FRAMES = 30;
const BATCH = 200;
const BATCH_REPEATS = 5;
const STILL_TARGET_MS = 0.2;
const MOVING_TARGET_MS = 0.5;

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function percentile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
}

function summary(values: readonly number[]): { median: number; p95: number; mean: number } {
    return { median: median(values), p95: percentile(values, 0.95), mean: values.reduce((sum, value) => sum + value, 0) / values.length };
}

/** A mesh near the middle of the scene's meshes by size: an ordinary object, not the whole building. */
function pickMovingMesh(root: Object3D): Mesh {
    const meshes: { mesh: Mesh; size: number }[] = [];
    root.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh || mesh.userData.giContributes === false) return;
        meshes.push({ mesh, size: new Box3().setFromObject(mesh).getSize(new Vector3()).length() });
    });
    meshes.sort((a, b) => a.size - b.size);
    return meshes[Math.floor(meshes.length / 2)].mesh;
}

const env = import.meta.env as Record<string, unknown>;
const sceneId = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES !== '' ? env.VITE_DDGI_SCENES.split(',')[0].trim() : 'sponza';

await runThreePage(async ({ device, renderer, data, log }) => {
    const manifest = loadAllManifests().find((candidate) => candidate.id === sceneId);
    assert(manifest, `no manifest "${sceneId}"`);
    data.scene = sceneId;
    const loaded = await loadScene(manifest, () => {});
    const scene = loaded.scene;
    // The viewer's sky dome is a picture of the sky, not geometry (T1.8 marks it the same way).
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });

    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);

    const platform = new GIPlatformImpl(device, { label: `three timing ${sceneId}` });
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing } }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const environment: EnvironmentDescriptor | null = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const warnings: string[] = [];
    const threeGI = new ThreeGI(platform, scene, { budget: { gpuMs: 2 }, environment, warn: (message) => warnings.push(message) });

    const firstStart = performance.now();
    threeGI.update(camera);
    data.firstSync = { ms: threeGI.timings.syncMs, totalMs: performance.now() - firstStart, report: { ...threeGI.report } };
    data.counts = threeGI.sync.counts;
    let objectsInScene = 0;
    scene.traverse(() => objectsInScene++);
    data.objectsInScene = objectsInScene;
    log(`first sync ${threeGI.timings.syncMs.toFixed(1)} ms; ${JSON.stringify(threeGI.sync.counts)}`);
    await platform.rays.whenBuilt();

    // The loop ThreeGI documents: update, then render.
    const frame = (): void => {
        threeGI.update(camera);
        renderer.render(scene, camera);
    };
    for (let index = 0; index < WARMUP_FRAMES; index++) {
        frame();
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }

    const measureFrames = async (beforeFrame: (index: number) => void) => {
        const sync: number[] = [];
        const matrices: number[] = [];
        const encode: number[] = [];
        const sent: number[] = [];
        for (let index = 0; index < FRAMES; index++) {
            beforeFrame(index);
            frame();
            sync.push(threeGI.timings.syncMs);
            matrices.push(threeGI.timings.matrixWorldMs);
            encode.push(threeGI.timings.encodeMs);
            sent.push(threeGI.report.transforms + threeGI.report.materials + threeGI.report.lights + threeGI.report.meshes);
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        return { syncMs: summary(sync), matrixWorldMs: summary(matrices), encodeMs: summary(encode), changesSentPerFrame: summary(sent) };
    };

    /** Mean ms of BATCH back-to-back syncs, BATCH_REPEATS times; the median of those means. */
    const measureBatched = (beforeSync: (index: number) => void) => {
        const means: number[] = [];
        let call = 0;
        for (let repeat = 0; repeat < BATCH_REPEATS; repeat++) {
            const start = performance.now();
            for (let index = 0; index < BATCH; index++) {
                beforeSync(call);
                // The world matrix of whatever moved, as ThreeGI.update would have it.
                moving.updateMatrixWorld();
                threeGI.sync.update(scene);
                call++;
            }
            means.push((performance.now() - start) / BATCH);
        }
        return { meanMsPerSync: median(means), repeats: means };
    };

    const moving = pickMovingMesh(loaded.content);
    const home = moving.position.clone();
    const move = (index: number) => moving.position.set(home.x, home.y + 0.05 * Math.sin(index * 0.1), home.z);

    // Still. updateMatrixWorld in the batch loop touches only the picked mesh, which doesn't move here.
    const stillFrames = await measureFrames(() => {});
    const stillBatched = measureBatched(() => {});
    // One object moving every frame.
    const movingFrames = await measureFrames(move);
    const movingBatched = measureBatched(move);
    moving.position.copy(home);

    // For scale: three's own updateMatrixWorld over the same scene (render does it every frame; ThreeGI doesn't repeat it by default).
    const matrixStart = performance.now();
    for (let index = 0; index < 50; index++) scene.updateMatrixWorld();
    data.threeUpdateMatrixWorldMs = (performance.now() - matrixStart) / 50;

    data.movingObject = moving.name || moving.uuid;
    data.still = { frames: stillFrames, batched: stillBatched };
    data.moving = { frames: movingFrames, batched: movingBatched };
    data.warnings = warnings;
    log(`still: sync per frame median ${stillFrames.syncMs.median.toFixed(3)} ms (p95 ${stillFrames.syncMs.p95.toFixed(3)}), batched ${stillBatched.meanMsPerSync.toFixed(4)} ms`);
    log(`moving: sync per frame median ${movingFrames.syncMs.median.toFixed(3)} ms (p95 ${movingFrames.syncMs.p95.toFixed(3)}), batched ${movingBatched.meanMsPerSync.toFixed(4)} ms`);
    log(`three's own updateMatrixWorld ${(data.threeUpdateMatrixWorldMs as number).toFixed(3)} ms; platform encode per frame median ${stillFrames.encodeMs.median.toFixed(3)} ms`);

    threeGI.dispose();
    platform.dispose();

    assert(stillFrames.changesSentPerFrame.mean === 0, `a still scene sent ${stillFrames.changesSentPerFrame.mean} changes per frame on average`);
    assert(movingFrames.changesSentPerFrame.median === 1, `one moving object sent a median of ${movingFrames.changesSentPerFrame.median} changes per frame`);
    assert(stillBatched.meanMsPerSync <= STILL_TARGET_MS, `still scene: ${stillBatched.meanMsPerSync.toFixed(3)} ms per sync (batched), over ${STILL_TARGET_MS} ms`);
    assert(movingBatched.meanMsPerSync <= MOVING_TARGET_MS, `one object moving: ${movingBatched.meanMsPerSync.toFixed(3)} ms per sync (batched), over ${MOVING_TARGET_MS} ms`);
    // The done-when's own wording: the median over many real frames (timer steps of 0.1 ms included).
    assert(stillFrames.syncMs.median <= STILL_TARGET_MS + 1e-6, `still scene: per-frame median ${stillFrames.syncMs.median.toFixed(3)} ms, over ${STILL_TARGET_MS} ms`);
    assert(movingFrames.syncMs.median <= MOVING_TARGET_MS + 1e-6, `one object moving: per-frame median ${movingFrames.syncMs.median.toFixed(3)} ms, over ${MOVING_TARGET_MS} ms`);
});
