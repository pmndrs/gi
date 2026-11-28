/**
 * T1.7: the three adapter's CPU time per frame, on a real scene in the
 * browser. Default: `sponza` (Intel's New Sponza with the curtains);
 * `VITE_DDGI_SCENES=bistro-interior` (or another manifest id) picks another.
 *
 * The scene is the viewer's own (`apps/viewer/src/scene/load-scene.ts`,
 * GLTFLoader with textures), mirrored by `ThreeGI` into a platform running
 * DDGI on the manifest's volume, rendered by three each frame at 256².
 *
 * What is timed is the adapter's own work: `SceneSync.update` (marked and
 * hot objects, this frame's share of the walk, lights, edit checks, the
 * environment) and what it sends. Reported separately, not part of it: three's
 * own `updateMatrixWorld` (which `render` does; ThreeGI doesn't repeat it by
 * default) and the platform's `encode`.
 *
 * Two ways, for each of a still scene and one object moving every frame:
 * - **per frame:** `threeGI.timings.syncMs` over `FRAMES` real frames. The
 *   browser's `performance.now()` steps in 0.1 ms with random jitter (NOTES),
 *   so single frames are coarse but their mean over many frames isn't: the
 *   targets are judged on the mean (median and p95 are reported too);
 * - **batched:** the mean of `BATCH` back-to-back `sync.update` calls, five
 *   times, median of the five. Each call is a full frame's sync (for the
 *   moving case the object moves before every call), so this is the same
 *   work, measured below the timer's step, but with warm caches: a real
 *   frame's sync starts cold after three's render and reads 2-4x slower.
 * Both must meet the targets.
 *
 * Targets (TASKS T1.7): still ≤ 0.2 ms, one object moving ≤ 0.5 ms.
 * T2.1 checks the still target on `sponza-candles` (1,024 lights) as the
 * median of 300 frames, reported as `still.frames.syncMs.medianOf300`; the
 * page draws that scene with three's clustered lighting, as the viewer does
 * (apps/viewer/src/render/scene-lighting.ts). `VITE_DDGI_ADAPTER_CHECK=still`
 * measures only that: 300 still frames and the batched still syncs, no
 * moving object or detection latency, with DDGI's updates paused (the
 * budget `{ paused: true }`). On `sponza-candles` every DDGI frame is heavy
 * GPU work (its light selection with 1,024 lights costs about 7 ms at any
 * batch, NOTES), and two runs of it throttled the RTX laptop, which the
 * clock rule doesn't accept. Paused, the adapter's sync does the same work
 * (it runs before the budget is looked at, and a still scene sends
 * nothing), three still renders every frame, and the GPU stays cool.
 *
 * Also measured: how many frames until a static object's first move reaches
 * GI without `markChanged` (the walk's detection latency), and with it.
 *
 * `VITE_DDGI_ADAPTER_CHECK=set-transforms` (T2.10) times `setTransforms` with
 * 10,000 transforms inside these real frames instead (set-transforms.ts).
 *
 * A timing run takes the bench lock:
 *   node tools/reference/bench-lock.ts run --what "T1.7 adapter timing" -- pnpm test:gpu --gpu nvidia --filter three-adapter-timing
 */

import { Box3, PerspectiveCamera, Vector3, type Mesh, type Object3D } from 'three/webgpu';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/three/src/three-gi.ts';
import { DEFAULT_OBJECTS_PER_FRAME, HOT_FRAMES } from '../../../../packages/three/src/scene-sync.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { sceneLighting } from '../../../../apps/viewer/src/render/scene-lighting.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import type { EnvironmentDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import { runThreePage } from '../three-shared/harness.ts';
import { addMovingInstances, SET_TRANSFORMS_INSTANCES, SET_TRANSFORMS_TARGET_MS, timeSetTransforms } from './set-transforms.ts';

/** `VITE_DDGI_ADAPTER_CHECK=still`: the still frames only (the file comment). */
const STILL_ONLY = (import.meta.env as Record<string, unknown>).VITE_DDGI_ADAPTER_CHECK === 'still';
/** `VITE_DDGI_ADAPTER_CHECK=set-transforms`: 10,000 moving instances, `setTransforms` timed in real frames (T2.10). */
const SET_TRANSFORMS = (import.meta.env as Record<string, unknown>).VITE_DDGI_ADAPTER_CHECK === 'set-transforms';
/** Frames timed in the set-transforms check (after the warm-up frames). */
const SET_TRANSFORMS_FRAMES = 200;
const FRAMES = STILL_ONLY ? 300 : 600;
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
    renderer.lighting = sceneLighting(loaded.directLights, renderer.lighting);
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

    const instances = SET_TRANSFORMS ? addMovingInstances(scene, manifest.volume.center, manifest.volume.size) : null;

    const platform = new GIPlatformImpl(device, { label: `three timing ${sceneId}` });
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing }, warmUpPasses: 0 }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const environment: EnvironmentDescriptor | null = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const warnings: string[] = [];
    // VITE_DDGI_OBJECTS_PER_FRAME and VITE_DDGI_EDIT_CHECKS try other budgets than the defaults.
    const objectsPerFrame = Number(env.VITE_DDGI_OBJECTS_PER_FRAME) || DEFAULT_OBJECTS_PER_FRAME;
    const editChecksPerFrame = Number(env.VITE_DDGI_EDIT_CHECKS) || undefined;
    data.budgets = { objectsPerFrame, editChecksPerFrame: editChecksPerFrame ?? 'default' };
    const threeGI = new ThreeGI(platform, scene, { budget: STILL_ONLY ? { paused: true } : { gpuMs: 2 }, environment, warn: (message) => warnings.push(message), objectsPerFrame, editChecksPerFrame });

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
        instances?.move(index);
        frame();
        if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
    }

    if (instances) {
        // T2.10: every instance slides every frame; each frame's setTransforms call timed (set-transforms.ts).
        const timed = timeSetTransforms(platform);
        const sync: number[] = [];
        const encode: number[] = [];
        const callsPerFrame: number[] = [];
        for (let index = 0; index < SET_TRANSFORMS_FRAMES; index++) {
            instances.move(WARMUP_FRAMES + index);
            const callsBefore = timed.calls.length;
            frame();
            callsPerFrame.push(timed.calls.length - callsBefore);
            sync.push(threeGI.timings.syncMs);
            encode.push(threeGI.timings.encodeMs);
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        const bigCalls = timed.calls.filter((call) => call.transforms >= SET_TRANSFORMS_INSTANCES);
        const callMs = bigCalls.map((call) => call.ms);
        data.setTransforms = {
            instances: SET_TRANSFORMS_INSTANCES,
            frames: SET_TRANSFORMS_FRAMES,
            callsWithAllInstances: bigCalls.length,
            callsPerFrame: summary(callsPerFrame),
            transformsPerCall: summary(timed.calls.map((call) => call.transforms)),
            callMs: summary(callMs),
            syncMs: summary(sync),
            encodeMs: summary(encode),
            targetMs: SET_TRANSFORMS_TARGET_MS,
        };
        data.warnings = warnings;
        log(
            `setTransforms with ${SET_TRANSFORMS_INSTANCES} transforms in real frames (three renders every frame, DDGI at 2 ms): ${bigCalls.length} calls, median ${median(callMs).toFixed(3)} ms, p95 ${percentile(callMs, 0.95).toFixed(3)}, mean ${summary(callMs).mean.toFixed(3)} (bar ${SET_TRANSFORMS_TARGET_MS}); ` +
                `the adapter's whole sync median ${median(sync).toFixed(3)} ms, encode ${median(encode).toFixed(3)} ms`,
        );
        threeGI.dispose();
        platform.dispose();
        assert(bigCalls.length >= SET_TRANSFORMS_FRAMES * 0.9, `only ${bigCalls.length} of ${SET_TRANSFORMS_FRAMES} frames sent all ${SET_TRANSFORMS_INSTANCES} transforms in one call`);
        assert(median(callMs) <= SET_TRANSFORMS_TARGET_MS, `setTransforms median ${median(callMs).toFixed(3)} ms in real frames, over ${SET_TRANSFORMS_TARGET_MS} ms`);
        return;
    }

    const measureFrames = async (beforeFrame: (index: number) => void) => {
        const sync: number[] = [];
        const matrices: number[] = [];
        const encode: number[] = [];
        const sent: number[] = [];
        // The timer's own reading of an empty interval at the same point of the frame:
        // what a per-frame mean costs before any work (coarse, jittered timer; interruptions).
        const empty: number[] = [];
        for (let index = 0; index < FRAMES; index++) {
            beforeFrame(index);
            const emptyStart = performance.now();
            empty.push(performance.now() - emptyStart);
            frame();
            sync.push(threeGI.timings.syncMs);
            matrices.push(threeGI.timings.matrixWorldMs);
            encode.push(threeGI.timings.encodeMs);
            sent.push(threeGI.report.transforms + threeGI.report.materials + threeGI.report.lights + threeGI.report.meshes);
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        return { syncMs: { ...summary(sync), medianOf300: median(sync.slice(0, 300)) }, emptyIntervalMs: summary(empty), matrixWorldMs: summary(matrices), encodeMs: summary(encode), changesSentPerFrame: summary(sent) };
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

    // Detection latency: a static object moved once, without and with markChanged.
    const framesUntilSent = async (other: Mesh, mark: boolean): Promise<number> => {
        other.position.y += 0.01;
        other.updateMatrixWorld();
        if (mark) threeGI.markChanged(other);
        for (let frames = 1; frames <= 1000; frames++) {
            frame();
            if (threeGI.report.transforms > 0) return frames;
            if (frames % 8 === 0) await device.queue.onSubmittedWorkDone();
        }
        return Number.POSITIVE_INFINITY;
    };
    const settle = async (): Promise<void> => {
        // Long enough for everything to leave the hot set.
        for (let index = 0; index < HOT_FRAMES + 10; index++) {
            frame();
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
    };
    const staticMeshes: Mesh[] = [];
    loaded.content.traverse((object) => {
        if ((object as Mesh).isMesh && object !== moving) staticMeshes.push(object as Mesh);
    });
    const unmarked = STILL_ONLY ? 0 : await framesUntilSent(staticMeshes[Math.floor(staticMeshes.length * 0.3)], false);
    const marked = STILL_ONLY ? 1 : await framesUntilSent(staticMeshes[Math.floor(staticMeshes.length * 0.7)], true);
    if (!STILL_ONLY) await settle();
    if (!STILL_ONLY) data.detection = {
        framesUntilFirstMoveSentUnmarked: unmarked,
        framesUntilFirstMoveSentMarked: marked,
        boundFrames: Math.ceil(objectsInScene / objectsPerFrame),
        objectsPerFrame,
    };
    log(`detection: first move of a static object sent after ${unmarked} frames (bound ${Math.ceil(objectsInScene / objectsPerFrame)}), ${marked} with markChanged`);
    const home = moving.position.clone();
    const move = (index: number) => moving.position.set(home.x, home.y + 0.05 * Math.sin(index * 0.1), home.z);

    // Still. updateMatrixWorld in the batch loop touches only the picked mesh, which doesn't move here.
    const stillFrames = await measureFrames(() => {});
    const stillBatched = measureBatched(() => {});
    // One object moving every frame.
    if (STILL_ONLY) {
        data.still = { frames: stillFrames, batched: stillBatched };
        data.warnings = warnings;
        log(`still: sync per frame median of ${FRAMES} ${stillFrames.syncMs.medianOf300.toFixed(3)} ms (mean ${stillFrames.syncMs.mean.toFixed(3)}, p95 ${stillFrames.syncMs.p95.toFixed(3)}), batched ${stillBatched.meanMsPerSync.toFixed(4)} ms; ${loaded.directLights.length} lights`);
        threeGI.dispose();
        platform.dispose();
        assert(stillFrames.changesSentPerFrame.mean === 0, `a still scene sent ${stillFrames.changesSentPerFrame.mean} changes per frame on average`);
        assert(stillBatched.meanMsPerSync <= STILL_TARGET_MS, `still scene: ${stillBatched.meanMsPerSync.toFixed(3)} ms per sync (batched), over ${STILL_TARGET_MS} ms`);
        assert(stillFrames.syncMs.medianOf300 <= STILL_TARGET_MS, `still scene: median of 300 frames ${stillFrames.syncMs.medianOf300.toFixed(3)} ms, over ${STILL_TARGET_MS} ms`);
        return;
    }
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
    log(`still: sync per frame mean ${stillFrames.syncMs.mean.toFixed(3)} ms (median ${stillFrames.syncMs.median.toFixed(3)}, median of the first 300 ${stillFrames.syncMs.medianOf300.toFixed(3)}, p95 ${stillFrames.syncMs.p95.toFixed(3)}), batched ${stillBatched.meanMsPerSync.toFixed(4)} ms; ${loaded.directLights.length} lights`);
    log(`moving: sync per frame mean ${movingFrames.syncMs.mean.toFixed(3)} ms (median ${movingFrames.syncMs.median.toFixed(3)}, p95 ${movingFrames.syncMs.p95.toFixed(3)}), batched ${movingBatched.meanMsPerSync.toFixed(4)} ms`);
    log(`three's own updateMatrixWorld ${(data.threeUpdateMatrixWorldMs as number).toFixed(3)} ms; platform encode per frame median ${stillFrames.encodeMs.median.toFixed(3)} ms`);

    threeGI.dispose();
    platform.dispose();

    assert(stillFrames.changesSentPerFrame.mean === 0, `a still scene sent ${stillFrames.changesSentPerFrame.mean} changes per frame on average`);
    assert(movingFrames.changesSentPerFrame.median === 1, `one moving object sent a median of ${movingFrames.changesSentPerFrame.median} changes per frame`);
    assert(stillBatched.meanMsPerSync <= STILL_TARGET_MS, `still scene: ${stillBatched.meanMsPerSync.toFixed(3)} ms per sync (batched), over ${STILL_TARGET_MS} ms`);
    assert(movingBatched.meanMsPerSync <= MOVING_TARGET_MS, `one object moving: ${movingBatched.meanMsPerSync.toFixed(3)} ms per sync (batched), over ${MOVING_TARGET_MS} ms`);
    assert(stillFrames.syncMs.medianOf300 <= STILL_TARGET_MS, `still scene: median of 300 frames ${stillFrames.syncMs.medianOf300.toFixed(3)} ms, over ${STILL_TARGET_MS} ms`);
    assert(stillFrames.syncMs.mean <= STILL_TARGET_MS, `still scene: ${stillFrames.syncMs.mean.toFixed(3)} ms per frame (mean of ${FRAMES}), over ${STILL_TARGET_MS} ms`);
    assert(movingFrames.syncMs.mean <= MOVING_TARGET_MS, `one object moving: ${movingFrames.syncMs.mean.toFixed(3)} ms per frame (mean of ${FRAMES}), over ${MOVING_TARGET_MS} ms`);
    assert(unmarked <= Math.ceil(objectsInScene / objectsPerFrame) + 1, `a static object's first move took ${unmarked} frames to reach GI`);
    assert(marked === 1, `with markChanged a move took ${marked} frames to reach GI; it should be the next update`);
});
