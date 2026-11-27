/**
 * #26 (was T2.12 check 2, `ddgi-tree-swap`): DDGI on the GPU's H-PLOC mesh
 * trees, which hosts get on a device with subgroups and which nothing
 * replaces, against tinybvh's trees (the declared fallback).
 *
 * Per scene (Sponza by default), two platforms on the same scene, each
 * started as the viewer starts it (`startViewerScene`) and warmed up:
 *
 * - **gpu**, the defaults: every mesh on its GPU tree; tinybvh builds only
 *   the meshes the GPU refused, and nothing else runs in the worker;
 * - **tinybvh-only** (`gpuFirst: false`): the fallback, M1's path.
 *
 * Then, at a fixed batch of probes a frame (`VITE_DDGI_UNITS`, default 512),
 * GPU-bound (two frames in flight):
 *
 * 1. the trees: `gpu` has a GPU tree for every mesh it didn't refuse, a
 *    fallback tree for each refused one, and no tinybvh build left running;
 * 2. both converge for CONVERGE_FRAMES frames;
 * 3. **trace speed**: ROUNDS blocks of BLOCK_FRAMES frames of each in turn,
 *    each first in half the rounds (so a clock that drifts, and the second
 *    timing's slowdown, affect both alike); the median GPU ms of the passes
 *    that walk the trees (`ddgi/trace` and, with direct light deferred,
 *    `ddgi/gi-direct`) per frame. The done-when: `gpu` at most 5% slower
 *    than `tinybvh-only` (one-sided since #26: T2.12's two-sided bar compared
 *    the same trees after a swap, and faster GPU trees are no failure),
 *    judged on Sponza on the RTX under the clock rule
 *    (DECISIONS 2026-11-05); other scenes and GPUs report it. A small
 *    scene's tree passes take about 2 ms a frame, and in a correctness run
 *    on a hot RTX two-room read 0.80x with the same trees on both sides;
 * 4. `stats.recompiles` stays 0 over all of it, the GPU build's own buffers
 *    are gone, and the node buffers of both are reported.
 *
 * VITE_DDGI_SCENES=a,b picks scenes (Sponza and Bistro need the downloads; a
 * small scene such as two-room checks correctness quickly; Bistro on the
 * 6 GB RTX wants `VITE_DDGI_MAX_TEXTURE_SIZE=1024`, T2.17). For hand use
 * under the bench lock: `node tools/reference/bench-lock.ts run --what
 * "gpu trees" -- pnpm test:gpu --gpu nvidia --filter ddgi-gpu-trees`.
 */

import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { assert } from '../../client.ts';
import { startViewerScene, warmUp, yieldToCallbacks, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SCENES = listFromEnv(env.VITE_DDGI_SCENES, ['sponza']);
/** Probes a frame: a little over Sponza's batch at 2 ms (T2.4b calibrated 316), so the trace dominates the frame. */
const UNITS = Number(env.VITE_DDGI_UNITS) > 0 ? Math.round(Number(env.VITE_DDGI_UNITS)) : 512;
const CONVERGE_FRAMES = 240;
/** Even, so each side is timed first in half the rounds (see `timeInTurn`). */
const ROUNDS = 6;
const BLOCK_FRAMES = 120;
/** The done-when's bar. */
const TRACE_SPEED_TOLERANCE = 0.05;
/** The scene the speed bar is judged on (see the file comment). */
const SPEED_SCENE = 'sponza';
/** The passes that walk the mesh trees. */
const TREE_PASSES = ['ddgi/trace', 'ddgi/gi-direct'];
/** Frames kept queued so the GPU never waits for the CPU. */
const FRAMES_IN_FLIGHT = 2;
/** T2.17: scale model textures down to this side at load (`LoadSceneOptions.maxTextureSize`); undefined keeps them. */
const MAX_TEXTURE_SIZE = typeof env.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' && env.VITE_DDGI_MAX_TEXTURE_SIZE !== '' ? Number(env.VITE_DDGI_MAX_TEXTURE_SIZE) : undefined;
const LOAD = MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE };

function median(values: readonly number[]): number {
    if (values.length === 0) return NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function round(value: number, digits = 4): number {
    return Number.isFinite(value) ? Number(value.toFixed(digits)) : value;
}

/** The tree-walking passes' GPU ms per platform frame, as their timestamps arrive. */
class TreePassTimes {
    private readonly byFrame = new Map<number, Map<string, number>>();

    note(platform: GIPlatformImpl): void {
        // Passes of one name in a frame add up (the deferred trace runs one of each per round of probes).
        const sums = new Map<number, Map<string, number>>();
        for (const pass of platform.stats.passes) {
            if (!TREE_PASSES.includes(pass.name) || pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
            let passes = sums.get(pass.frame);
            if (!passes) {
                passes = new Map();
                sums.set(pass.frame, passes);
            }
            passes.set(pass.name, (passes.get(pass.name) ?? 0) + pass.gpuMs);
        }
        for (const [frame, passes] of sums) this.byFrame.set(frame, passes);
    }

    /** Each listed frame's tree-walking ms, for frames whose timings arrived. */
    msOf(frames: readonly number[]): number[] {
        const values: number[] = [];
        for (const frame of frames) {
            const passes = this.byFrame.get(frame);
            if (passes?.has('ddgi/trace')) values.push([...passes.values()].reduce((sum, ms) => sum + ms, 0));
        }
        return values;
    }
}

interface Side {
    readonly name: 'gpu' | 'tinybvh-only';
    readonly scene: ViewerScene;
    readonly times: TreePassTimes;
}

/** `count` frames of one side at UNITS probes, GPU-bound; returns the platform frames they were. */
async function runFrames(side: Side, count: number): Promise<number[]> {
    const { scene, times } = side;
    const pending: Promise<void>[] = [];
    const frames: number[] = [];
    for (let index = 0; index < count; index++) {
        scene.frame({ units: UNITS }, 1 / 60);
        frames.push(scene.platform.stats.frame);
        times.note(scene.platform);
        pending.push(scene.device.queue.onSubmittedWorkDone());
        if (pending.length >= FRAMES_IN_FLIGHT) await pending.shift();
        // Lets the timestamp readbacks land before the next frame collects them.
        await yieldToCallbacks();
    }
    await Promise.all(pending);
    // A few empty frames so the last frames' timestamps arrive.
    for (let index = 0; index < 3; index++) {
        scene.frame({ units: 0 }, 0);
        await scene.device.queue.onSubmittedWorkDone();
        await yieldToCallbacks();
        times.note(scene.platform);
    }
    return frames;
}

/**
 * ROUNDS blocks of each side in turn: the median tree-walking ms per frame of
 * each. The side that goes first changes every round: on the power-limited
 * RTX the second of two back-to-back timings runs about 6% slow (NOTES, B3),
 * and a first version that always timed one side first read it 0.93x of the
 * other with the same trees in both.
 */
async function timeInTurn(sides: readonly Side[]): Promise<Record<string, number>> {
    const frames = new Map<string, number[]>(sides.map((side) => [side.name, []]));
    for (let round = 0; round < ROUNDS; round++) {
        const order = round % 2 === 0 ? sides : [...sides].reverse();
        for (const side of order) frames.get(side.name)?.push(...(await runFrames(side, BLOCK_FRAMES)));
    }
    return Object.fromEntries(sides.map((side) => [side.name, median(side.times.msOf(frames.get(side.name) ?? []))]));
}

// The adapter's own device and limits (Sponza outgrows the default 128 MiB storage binding), as the viewer makes them.
await runThreePage(async ({ device, adapterSummary, log, data }) => {
    const onRtx = /nvidia/i.test(adapterSummary.vendor);
    const manifests = loadAllManifests();
    const failures: string[] = [];
    Object.assign(data, { units: UNITS, rounds: ROUNDS, blockFrames: BLOCK_FRAMES, ...(MAX_TEXTURE_SIZE === undefined ? {} : { maxTextureSize: MAX_TEXTURE_SIZE }) });

    for (const sceneId of SCENES) {
        const manifest = manifests.find((entry) => entry.id === sceneId);
        assert(manifest, `no manifest '${sceneId}'`);
        const gpu: Side = {
            name: 'gpu',
            scene: await startViewerScene(device, manifest, { gpuTiming: true, label: `gpu trees ${sceneId} gpu`, load: LOAD }),
            times: new TreePassTimes(),
        };
        const tinybvhOnly: Side = {
            name: 'tinybvh-only',
            scene: await startViewerScene(device, manifest, { gpuTiming: true, label: `gpu trees ${sceneId} tinybvh-only`, meshTrees: { gpuFirst: false }, load: LOAD }),
            times: new TreePassTimes(),
        };
        const rays = gpu.scene.platform.rays;
        assert(rays.gpuFirstTrees, `${sceneId}: the platform builds no trees on the GPU on this device`);
        const recompilesBefore = gpu.scene.platform.stats.recompiles;

        // 1. The trees: the GPU's, and the fallback's only where the GPU refused.
        const counts = rays.meshTreeCounts;
        log(
            `${sceneId}: ${counts.onGpuTrees} meshes on GPU trees, ${counts.onTinybvhTrees} on tinybvh's (${counts.gpuTreesRefused} GPU trees refused), ` +
                `${counts.tinybvhBuilding} tinybvh builds running; GPU build ${round(counts.lastGpuBuildMs, 1)} ms to its stats`,
        );
        if (counts.onGpuTrees === 0) failures.push(`${sceneId}: no mesh got a GPU tree`);
        if (counts.onTinybvhTrees !== counts.gpuTreesRefused) failures.push(`${sceneId}: ${counts.onTinybvhTrees} meshes on tinybvh trees, but ${counts.gpuTreesRefused} GPU trees refused`);
        if (counts.tinybvhBuilding !== 0) failures.push(`${sceneId}: ${counts.tinybvhBuilding} tinybvh builds still run with every mesh on its tree`);

        // 2. and 3. Converge, then trace speed in turns.
        for (const side of [gpu, tinybvhOnly]) await warmUp(side.scene);
        for (const side of [gpu, tinybvhOnly]) await runFrames(side, CONVERGE_FRAMES);
        const treePassMs = await timeInTurn([gpu, tinybvhOnly]);
        const ratio = treePassMs.gpu / treePassMs['tinybvh-only'];
        log(`${sceneId}: tree passes on GPU trees ${round(treePassMs.gpu)} ms, tinybvh only ${round(treePassMs['tinybvh-only'])} ms (${round(ratio)}x)`);
        const judgeSpeed = onRtx && sceneId === SPEED_SCENE;
        if (judgeSpeed && ratio - 1 > TRACE_SPEED_TOLERANCE) failures.push(`${sceneId}: on GPU trees the tree passes take ${round(ratio)}x tinybvh-only's (bar: at most 5% slower)`);

        // 4. Nothing replaced a tree or recompiled, and the GPU build's buffers are gone.
        const after = rays.meshTreeCounts;
        const recompiles = gpu.scene.platform.stats.recompiles - recompilesBefore;
        const memory = rays.memory;
        const tinybvhMemory = tinybvhOnly.scene.platform.rays.memory;
        log(
            `${sceneId}: recompiles ${recompiles}; node buffer ${round(memory.nodes / 1e6, 1)} MB (${round(memory.nodesInUse / 1e6, 1)} MB in use), ` +
                `tinybvh only ${round(tinybvhMemory.nodes / 1e6, 1)} MB (${round(tinybvhMemory.nodesInUse / 1e6, 1)} MB); GPU build buffers ${memory.gpuBuildOutputs} bytes`,
        );
        if (after.onGpuTrees !== counts.onGpuTrees) failures.push(`${sceneId}: ${counts.onGpuTrees} meshes were on GPU trees at the start, ${after.onGpuTrees} at the end`);
        if (recompiles !== 0) failures.push(`${sceneId}: ${recompiles} compute pipelines were made while the scene ran`);
        if (memory.gpuBuildOutputs !== 0) failures.push(`${sceneId}: the GPU build's buffers (${memory.gpuBuildOutputs} bytes) were still alive`);

        data[sceneId] = {
            meshes: counts.onGpuTrees + counts.onTinybvhTrees,
            trees: counts,
            treePassMs,
            ratio: round(ratio),
            speedJudged: judgeSpeed,
            recompiles,
            memory,
            memoryTinybvhOnly: tinybvhMemory,
        };
        gpu.scene.dispose();
        tinybvhOnly.scene.dispose();
    }
    assert(failures.length === 0, failures.join('\n'));
});
