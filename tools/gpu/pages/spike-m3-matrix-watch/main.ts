/**
 * T3.0 spike, question 2 (docs/design/m3-moving.md §6): what the
 * `updateMatrixWorld` override costs on Bistro interior, per object, in real
 * frames, and whether it finds an unmarked move in the frame it happens.
 *
 * The scene is the viewer's (the same loader as three-adapter-timing), drawn
 * by three every frame at 256². No DDGI: the question is the override's own
 * cost inside three's matrix update, so the GPU only draws.
 *
 * Each frame: `scene.updateMatrixWorld()` timed on its own (the update three's
 * render would do first), then `renderer.render` (which repeats it, warm, and
 * is timed separately so a slowdown the override causes elsewhere in three's
 * render shows too). Data is cold right after the previous render, as the
 * real adapter would meet it.
 *
 * Phases, in order: without the watch, with it on every object
 * (`MatrixWatch.installTree(scene)`), and without it again after `dispose`,
 * `FRAMES` each, after a warm-up of `WARMUP_FRAMES`. Also a
 * warm, batched number for each (back-to-back updates), because the browser's
 * timer steps in 0.1 ms with jitter (NOTES): means over many frames are fine,
 * single frames are not.
 *
 * Rough numbers only (DECISIONS 2026-11-09: no timing gates in M3).
 */

import { PerspectiveCamera, type Mesh, type Object3D } from 'three/webgpu';
import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { sceneLighting } from '../../../../apps/viewer/src/render/scene-lighting.ts';
import { MatrixWatch } from '../../../spikes/m3-t3.0/matrix-watch.ts';
import { runThreePage } from '../three-shared/harness.ts';

const FRAMES = 300;
const WARMUP_FRAMES = 200;
const BATCH = 50;
const BATCH_REPEATS = 5;

function mean(values: readonly number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

const env = import.meta.env as Record<string, unknown>;
const sceneId = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES !== '' ? env.VITE_DDGI_SCENES.split(',')[0].trim() : 'bistro-interior';

await runThreePage(async ({ device, renderer, data, log }) => {
    const manifest = loadAllManifests().find((candidate) => candidate.id === sceneId);
    assert(manifest, `no manifest "${sceneId}"`);
    data.scene = sceneId;
    const loaded = await loadScene(manifest, () => {});
    const scene = loaded.scene;
    renderer.lighting = sceneLighting(loaded.directLights, renderer.lighting);
    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    let objects = 0;
    scene.traverse(() => objects++);
    data.objects = objects;

    /** `FRAMES` real frames: the matrix update timed alone, then three's render timed. */
    const measureFrames = async (): Promise<{ updateMs: number; updateMedianMs: number; renderMs: number }> => {
        const update: number[] = [];
        const render: number[] = [];
        for (let frame = 0; frame < FRAMES; frame++) {
            const start = performance.now();
            scene.updateMatrixWorld();
            const updated = performance.now();
            renderer.render(scene, camera);
            render.push(performance.now() - updated);
            update.push(updated - start);
            if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        return { updateMs: mean(update), updateMedianMs: median(update), renderMs: mean(render) };
    };
    /** Back-to-back updates (warm caches): the median of `BATCH_REPEATS` means. */
    const measureBatched = (): number => {
        const means: number[] = [];
        for (let repeat = 0; repeat < BATCH_REPEATS; repeat++) {
            const start = performance.now();
            for (let index = 0; index < BATCH; index++) scene.updateMatrixWorld();
            means.push((performance.now() - start) / BATCH);
        }
        return median(means);
    };
    const warmUp = async (): Promise<void> => {
        for (let frame = 0; frame < WARMUP_FRAMES; frame++) {
            renderer.render(scene, camera);
            if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
    };

    await warmUp();
    const without = { frames: await measureFrames(), batchedMs: measureBatched() };

    const watch = new MatrixWatch();
    const installStart = performance.now();
    watch.installTree(scene);
    const installMs = performance.now() - installStart;
    await warmUp();
    assert(watch.takeMoved().length === 0, 'a still scene reported moved objects');
    const withWatch = { frames: await measureFrames(), batchedMs: measureBatched() };
    const stillMoved = watch.takeMoved().length;

    // An unmarked move: found by the update of the frame it happens in (the adapter would send it in the next update).
    let someMesh: Mesh | null = null;
    loaded.content.traverse((object: Object3D) => {
        if (!someMesh && (object as Mesh).isMesh) someMesh = object as Mesh;
    });
    assert(someMesh, 'no mesh in the scene');
    const mover = someMesh as Mesh;
    mover.position.y += 0.01;
    renderer.render(scene, camera);
    const found = watch.takeMoved();
    watch.dispose();

    // Without it again, after `uninstall` (which deletes the own property): brackets any drift
    // across the run, and shows whether the delete left three's objects slower.
    await warmUp();
    const withoutAfter = { frames: await measureFrames(), batchedMs: measureBatched() };

    // The cost against the mean of the two runs without it.
    const baseUpdateMs = (without.frames.updateMs + withoutAfter.frames.updateMs) / 2;
    const baseBatchedMs = (without.batchedMs + withoutAfter.batchedMs) / 2;
    const extraUpdateMs = withWatch.frames.updateMs - baseUpdateMs;
    const extraBatchedMs = withWatch.batchedMs - baseBatchedMs;
    const result = {
        objects,
        watched: objects,
        installMs,
        without,
        withWatch,
        withoutAfter,
        nsPerObjectInFrames: (extraUpdateMs * 1e6) / objects,
        nsPerObjectBatched: (extraBatchedMs * 1e6) / objects,
        renderChangeMs: withWatch.frames.renderMs - (without.frames.renderMs + withoutAfter.frames.renderMs) / 2,
        stillFramesReportedMoved: stillMoved,
        unmarkedMoveFoundSameFrame: found.length === 1 && found[0] === mover,
    };
    Object.assign(data, result);
    log(
        `${sceneId}, ${objects} objects: updateMatrixWorld per frame ${without.frames.updateMs.toFixed(3)} / ${withWatch.frames.updateMs.toFixed(3)} / ${withoutAfter.frames.updateMs.toFixed(3)} ms ` +
            `(without / with / without again; ${result.nsPerObjectInFrames.toFixed(0)} ns per object), ` +
            `batched ${without.batchedMs.toFixed(3)} / ${withWatch.batchedMs.toFixed(3)} / ${withoutAfter.batchedMs.toFixed(3)} ms (${result.nsPerObjectBatched.toFixed(0)} ns per object); ` +
            `render ${without.frames.renderMs.toFixed(2)} / ${withWatch.frames.renderMs.toFixed(2)} / ${withoutAfter.frames.renderMs.toFixed(2)} ms; install ${installMs.toFixed(2)} ms`,
    );
    assert(stillMoved === 0, `the still frames reported ${stillMoved} moved objects`);
    assert(result.unmarkedMoveFoundSameFrame, `an unmarked move wasn't found in its frame (found ${found.length})`);
});
