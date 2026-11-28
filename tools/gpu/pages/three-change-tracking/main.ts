/**
 * T3.7 checks 1, 2 and 5 (docs/design/m3-moving.md "Draft done-whens"; TASKS
 * M3, prototype level): the three adapter's change tracking v2 on a real
 * scene, three rendering every frame in the documented order (`update`, then
 * `render`). Default scene: Bistro interior (2,622 objects, where the walk
 * alone takes 41 frames a pass); `VITE_DDGI_SCENES=<manifest id>` picks
 * another. Nothing calls `markChanged`.
 *
 * 1. **Added and removed objects.** A 0.5 m box is added to the scene with
 *    plain `scene.add`, at a point in front of the first camera bookmark. The
 *    adapter must create it in the very next update (its report), and GI's
 *    rays must then hit it: one ray from the camera at the box, traced
 *    through the platform each frame, until its distance is the box's (the
 *    platform builds the new mesh's tree in between; recorded, not judged).
 *    The box is then removed with `removeFromParent`: destroyed in the next
 *    update, and the ray goes back to what it hit before.
 * 2. **An unmarked first move.** The same box, still for longer than the hot
 *    set keeps it (so only its `updateMatrixWorld` hook can find the move),
 *    slides 0.6 m sideways. The move must be sent within 10 frames (the bar),
 *    and the ray at its new place must hit it. Also one of the scene's own
 *    meshes, moved unmarked: frames until sent.
 * 5. **Costs, recorded and not judged** (DECISIONS 2026-11-09, M3 has no
 *    timing gates): the adapter's sync per frame, still and with one object
 *    moving every frame (median of 300 real frames); three's own
 *    `scene.updateMatrixWorld()` with the hooks in place and after they are
 *    removed (the hooks' cost is paid there, inside three's render); the
 *    first sync, which now also attaches the listeners and hooks.
 *
 * Also: the platform's recompile counter stays at 0 through all of it.
 *
 * Needs the downloaded scene; manual. Under the bench lock:
 *   node tools/reference/bench-lock.ts run --no-cool-wait --what "T3.7 change tracking" -- pnpm test:gpu --gpu nvidia --filter three-change-tracking
 */

import { BoxGeometry, Mesh, MeshStandardMaterial, PerspectiveCamera, Vector3, type Object3D } from 'three/webgpu';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi/src/ddgi-technique.ts';
import { HIT_RECORD_BYTES, packRays, RAY_BYTES } from '../../../../packages/core/src/platform/trace/index.ts';
import { toFloat32Ray } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import type { EnvironmentDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import { ThreeGI } from '../../../../packages/three/src/three-gi.ts';
import { HOT_FRAMES } from '../../../../packages/three/src/scene-sync.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { sceneLighting } from '../../../../apps/viewer/src/render/scene-lighting.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { readBuffer } from '../tlas-oracle/top-level-check.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
const sceneId = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES !== '' ? env.VITE_DDGI_SCENES.split(',')[0].trim() : 'bistro-interior';

/** The done-when's bar for an unmarked first move (m3-moving.md T3.7 item 3, with the override adopted). */
const MOVE_BAR_FRAMES = 10;
/** How long rays may take to see a change the adapter sent (the platform's tree build); generous, recorded. */
const RAY_WAIT_FRAMES = 240;
/** Frames timed for the costs (the done-when's 300). */
const COST_FRAMES = 300;
const BOX_SIZE = 0.5;
/** How far in front of the camera the box goes, at most. */
const BOX_DISTANCE = 1.5;
const SLIDE = 0.6;
/** A ray's distance counts as "at the box" within this of the box's near face. */
const HIT_TOLERANCE = 0.02;

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

await runThreePage(async ({ device, renderer, data, log }) => {
    const manifest = loadAllManifests().find((candidate) => candidate.id === sceneId);
    assert(manifest, `no manifest "${sceneId}"`);
    data.scene = sceneId;
    const loaded = await loadScene(manifest, () => {});
    const scene = loaded.scene;
    renderer.lighting = sceneLighting(loaded.directLights, renderer.lighting);
    // The viewer's sky dome is a picture of the sky, not geometry.
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    let objectsInScene = 0;
    scene.traverse(() => objectsInScene++);
    data.objectsInScene = objectsInScene;

    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld();

    // rayBatches: the page traces its own rays through the platform's trees.
    const platform = new GIPlatformImpl(device, { label: `three change tracking ${sceneId}`, rayBatches: true });
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing }, warmUpPasses: 0 }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const environment: EnvironmentDescriptor | null = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const warnings: string[] = [];
    const threeGI = new ThreeGI(platform, scene, { budget: { gpuMs: 2 }, environment, warn: (message) => warnings.push(message) });
    const recompilesAtStart = platform.stats.recompiles;

    const firstStart = performance.now();
    threeGI.update(camera);
    data.firstSync = { ms: threeGI.timings.syncMs, totalMs: performance.now() - firstStart, counts: threeGI.sync.counts };
    await platform.rays.whenBuilt();

    // The loop ThreeGI documents: update, then render.
    const frame = (): void => {
        threeGI.update(camera);
        renderer.render(scene, camera);
    };
    const frames = async (count: number): Promise<void> => {
        for (let index = 0; index < count; index++) {
            frame();
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();
    };
    await frames(30);

    // One ray, traced through the platform as GI's own rays are: its closest hit distance, or Infinity.
    const rayBuffer = device.createBuffer({ label: 'change-tracking ray', size: RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const hitBuffer = device.createBuffer({ label: 'change-tracking hit', size: HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const traceDistance = async (origin: Vector3, target: Vector3): Promise<number> => {
        const direction = target.clone().sub(origin).normalize();
        device.queue.writeBuffer(rayBuffer, 0, packRays([toFloat32Ray(origin.toArray(), direction.toArray(), 0, 1e6)]) as Float32Array<ArrayBuffer>);
        const encoder = device.createCommandEncoder({ label: 'change-tracking trace' });
        platform.rays.encodeBatch(encoder, { mode: 'closest', rays: rayBuffer, hits: hitBuffer, count: 1 });
        device.queue.submit([encoder.finish()]);
        const t = new Float32Array(await readBuffer(device, hitBuffer, HIT_RECORD_BYTES))[0];
        return t >= 0 ? t : Number.POSITIVE_INFINITY;
    };
    /** Frames (update + render each) until the ray's distance passes `test`; Infinity if it never does. */
    const framesUntilRay = async (origin: Vector3, target: Vector3, test: (t: number) => boolean): Promise<{ frames: number; t: number }> => {
        let t = Number.NaN;
        for (let count = 1; count <= RAY_WAIT_FRAMES; count++) {
            frame();
            t = await traceDistance(origin, target);
            if (test(t)) return { frames: count, t };
        }
        return { frames: Number.POSITIVE_INFINITY, t };
    };

    // ---- 1. Added and removed objects ----
    const eye = camera.position.clone();
    const forward = camera.getWorldDirection(new Vector3());
    const aheadWithout = await traceDistance(eye, eye.clone().add(forward));
    // In free air: halfway to whatever the camera looks at, at most BOX_DISTANCE.
    const boxDistance = Math.min(BOX_DISTANCE, aheadWithout / 2);
    assert(boxDistance > BOX_SIZE, `the first camera looks at a surface ${aheadWithout.toFixed(2)} m away: no room for the box`);
    const boxCentre = eye.clone().addScaledVector(forward, boxDistance);
    const box = new Mesh(new BoxGeometry(BOX_SIZE, BOX_SIZE, BOX_SIZE), new MeshStandardMaterial({ color: 0xc0c0c0 }));
    box.name = 't3.7 added box';
    box.position.copy(boxCentre);
    // Facing the camera, so the ray meets a face square on: the near face is BOX_SIZE / 2 before the centre.
    box.lookAt(eye);
    const nearFace = boxDistance - BOX_SIZE / 2;
    const atBox = (t: number) => Math.abs(t - nearFace) < HIT_TOLERANCE;

    scene.add(box);
    frame();
    const addedReport = { ...threeGI.report };
    const addedRay = await framesUntilRay(eye, boxCentre, atBox);
    log(`added: created ${addedReport.created} instance(s) in the next update; rays hit the box ${addedRay.frames} frame(s) later at ${addedRay.t.toFixed(3)} m (near face ${nearFace.toFixed(3)} m; before: ${aheadWithout.toFixed(3)} m)`);

    box.removeFromParent();
    frame();
    const removedReport = { ...threeGI.report };
    const removedRay = await framesUntilRay(eye, boxCentre, (t) => !atBox(t));
    log(`removed: destroyed ${removedReport.destroyed} instance(s) in the next update; rays pass ${removedRay.frames} frame(s) later (${removedRay.t.toFixed(3)} m)`);
    data.added = { createdInNextUpdate: addedReport.created, framesUntilRaysHit: addedRay.frames, rayT: addedRay.t, nearFace, before: aheadWithout };
    data.removed = { destroyedInNextUpdate: removedReport.destroyed, framesUntilRaysPass: removedRay.frames, rayT: removedRay.t };

    // ---- 2. An unmarked first move ----
    scene.add(box);
    await framesUntilRay(eye, boxCentre, atBox);
    // Longer than the hot set keeps it: from here only its hook can find a move.
    await frames(HOT_FRAMES + 10);
    const right = new Vector3().crossVectors(forward, camera.up).normalize();
    box.position.addScaledVector(right, SLIDE);
    const movedCentre = box.position.clone();
    let framesUntilSent = Number.POSITIVE_INFINITY;
    for (let count = 1; count <= 2 * MOVE_BAR_FRAMES; count++) {
        frame();
        if (threeGI.report.transforms > 0) {
            framesUntilSent = count;
            break;
        }
    }
    // The ray at the box's new place (it still faces the camera's old line; the slide keeps it in front of the eye).
    const movedRay = await framesUntilRay(eye, movedCentre, (t) => t < eye.distanceTo(movedCentre) && t > eye.distanceTo(movedCentre) - BOX_SIZE);
    log(`moved box: sent after ${framesUntilSent} frame(s) (bar ${MOVE_BAR_FRAMES}); rays hit it at its new place ${movedRay.frames} frame(s) later`);

    // One of the scene's own meshes, never moved before: frames until its move is sent.
    const sceneMeshes: Mesh[] = [];
    loaded.content.traverse((object) => {
        if ((object as Mesh).isMesh) sceneMeshes.push(object as Mesh);
    });
    await frames(HOT_FRAMES + 10);
    const sceneMover = sceneMeshes[Math.floor(sceneMeshes.length * 0.3)];
    sceneMover.position.y += 0.01;
    let sceneMoverFrames = Number.POSITIVE_INFINITY;
    for (let count = 1; count <= 2 * MOVE_BAR_FRAMES; count++) {
        frame();
        if (threeGI.report.transforms > 0) {
            sceneMoverFrames = count;
            break;
        }
    }
    log(`a scene mesh's first unmarked move ("${sceneMover.name}"): sent after ${sceneMoverFrames} frame(s); the walk alone: up to ${Math.ceil(objectsInScene / 64)}`);
    data.move = { boxFramesUntilSent: framesUntilSent, boxFramesUntilRaysHit: movedRay.frames, sceneMeshFramesUntilSent: sceneMoverFrames, barFrames: MOVE_BAR_FRAMES, walkPassFrames: Math.ceil(objectsInScene / 64) };

    // ---- 5. Costs (recorded) ----
    box.removeFromParent();
    await frames(HOT_FRAMES + 10);
    const measure = async (beforeFrame: (index: number) => void) => {
        const sync: number[] = [];
        const sent: number[] = [];
        for (let index = 0; index < COST_FRAMES; index++) {
            beforeFrame(index);
            frame();
            sync.push(threeGI.timings.syncMs);
            sent.push(threeGI.report.transforms);
            if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        return { syncMs: summary(sync), transformsSentPerFrame: summary(sent) };
    };
    const still = await measure(() => {});
    const moverHome = sceneMover.position.clone();
    const mover = await measure((index) => sceneMover.position.set(moverHome.x, moverHome.y + 0.05 * Math.sin(index * 0.1), moverHome.z));
    sceneMover.position.copy(moverHome);
    await frames(HOT_FRAMES + 10);

    // three's own matrix update, which the hooks ride in: with them, then after they're gone.
    const timeThreeMatrices = (): number => {
        for (let index = 0; index < 10; index++) scene.updateMatrixWorld(true);
        const start = performance.now();
        for (let index = 0; index < 50; index++) scene.updateMatrixWorld(true);
        return (performance.now() - start) / 50;
    };
    const withHooks = timeThreeMatrices();
    const hookedMeshes = threeGI.sync.counts.objects;
    const recompiles = platform.stats.recompiles - recompilesAtStart;
    threeGI.dispose();
    const withoutHooks = timeThreeMatrices();
    platform.dispose();
    rayBuffer.destroy();
    hitBuffer.destroy();

    data.costs = { still, oneMover: mover, threeUpdateMatrixWorldMs: { withHooks, withoutHooks, hookedMeshes, perHookedMeshNs: ((withHooks - withoutHooks) * 1e6) / hookedMeshes } };
    data.recompiles = recompiles;
    data.warnings = warnings;
    log(`costs (recorded, not judged): still sync median ${still.syncMs.median.toFixed(3)} ms (p95 ${still.syncMs.p95.toFixed(3)}), one mover ${mover.syncMs.median.toFixed(3)} ms (p95 ${mover.syncMs.p95.toFixed(3)}); three's updateMatrixWorld ${withHooks.toFixed(3)} ms with hooks, ${withoutHooks.toFixed(3)} ms without`);
    log(`recompiles ${recompiles}`);

    assert(addedReport.created >= 1, `the added box wasn't created in the next update (created ${addedReport.created})`);
    assert(Number.isFinite(addedRay.frames), `rays never hit the added box (last t ${addedRay.t})`);
    assert(removedReport.destroyed >= 1, `the removed box wasn't destroyed in the next update (destroyed ${removedReport.destroyed})`);
    assert(Number.isFinite(removedRay.frames), `rays still hit the removed box after ${RAY_WAIT_FRAMES} frames`);
    assert(framesUntilSent <= MOVE_BAR_FRAMES, `the box's unmarked first move took ${framesUntilSent} frames to be sent (bar ${MOVE_BAR_FRAMES})`);
    assert(Number.isFinite(movedRay.frames), 'rays never hit the moved box at its new place');
    assert(sceneMoverFrames <= MOVE_BAR_FRAMES, `a scene mesh's unmarked first move took ${sceneMoverFrames} frames to be sent (bar ${MOVE_BAR_FRAMES})`);
    assert(still.transformsSentPerFrame.mean === 0, `a still scene sent ${still.transformsSentPerFrame.mean} transforms per frame on average`);
    assert(mover.transformsSentPerFrame.median === 1, `one moving object sent a median of ${mover.transformsSentPerFrame.median} transforms per frame`);
    assert(recompiles === 0, `the platform compiled ${recompiles} pipeline(s) after prepare`);
});
