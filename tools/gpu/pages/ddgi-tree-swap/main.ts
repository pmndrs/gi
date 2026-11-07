/**
 * T2.12 check 2: the swap from the GPU's first trees (H-PLOC) to tinybvh's.
 *
 * Per scene (Sponza by default), two platforms on the same scene, each
 * started as the viewer starts it (`startViewerScene`) and warmed up:
 *
 * - **A**, first trees from the GPU, with tinybvh's held back
 *   (`holdRefinements`), so the swap happens in one chosen frame;
 * - **B**, tinybvh only (`gpuFirst: false`): M1's path, the reference.
 *
 * Then, at a fixed batch of probes a frame (`VITE_DDGI_UNITS`, default 512),
 * GPU-bound (two frames in flight):
 *
 * 1. both converge for CONVERGE_FRAMES frames;
 * 2. **trace speed on GPU trees**: ROUNDS blocks of BLOCK_FRAMES frames of A
 *    and of B in turn, each first in half the rounds (so a clock that drifts,
 *    and the second timing's slowdown, affect both alike); the
 *    median GPU ms of the passes that walk the trees (`ddgi/trace` and,
 *    with direct light deferred, `ddgi/gi-direct`) per frame;
 * 3. **the swap frame**: once every tinybvh tree is built, A runs a few
 *    frames, reading the irradiance atlas's mean after each (the normal
 *    change from one frame to the next), then every held tree is released
 *    and the next frame swaps them all in. The done-when: the atlas's mean
 *    changes by at most 0.5% across that frame. Also reported: the mean 60
 *    frames later, and `stats.recompiles` (must stay 0: a swap changes
 *    buffers and the mesh table, never a kernel);
 * 4. **trace speed after the swap**, as in 2. The done-when: A within 5% of
 *    B, judged on Sponza (the done-when's scene) on the RTX, under the clock
 *    rule (DECISIONS 2026-11-05); other scenes and GPUs report it. A small
 *    scene's tree passes take about 2 ms a frame, and in a correctness run on
 *    a hot RTX two-room read 0.80x with the same trees on both sides.
 *    The ratio before the swap is reported for the lead (H-PLOC trees alone).
 *
 * 5. **The natural swap**, for the node buffer's size: a third platform with
 *    the defaults (no hold), frames at UNITS probes from its first trees
 *    until every tinybvh tree is in, so the swaps spread over the frames
 *    as the worker finishes them. Reported: the swaps and the frames they
 *    fell in, and the node buffer against tinybvh-only's.
 *
 * VITE_DDGI_SCENES=a,b picks scenes (Sponza needs the downloads; a small
 * scene such as two-room checks the swap's correctness quickly). For hand
 * use under the bench lock: `node tools/reference/bench-lock.ts run --what
 * "tree swap" -- pnpm test:gpu --gpu nvidia --filter ddgi-tree-swap`.
 */

import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { AccelerationMemory } from '../../../../packages/ddgi-core/src/platform/trace/ray-queries.ts';
import type { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { assert } from '../../client.ts';
import { halfToFloat, readTextureBytes } from '../ddgi-shared/harness.ts';
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
/** Frames before the swap whose atlas mean is read, for the normal change from one frame to the next. */
const FRAMES_BEFORE_SWAP = 4;
const FRAMES_AFTER_SWAP = 60;
/** The done-when's bars. */
const SWAP_MEAN_TOLERANCE = 0.005;
const TRACE_SPEED_TOLERANCE = 0.05;
/** The scene the speed bar is judged on (see the file comment). */
const SPEED_SCENE = 'sponza';
/** The passes that walk the mesh trees. */
const TREE_PASSES = ['ddgi/trace', 'ddgi/gi-direct'];
/** Frames kept queued so the GPU never waits for the CPU. */
const FRAMES_IN_FLIGHT = 2;

function median(values: readonly number[]): number {
    if (values.length === 0) return NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function round(value: number, digits = 4): number {
    return Number.isFinite(value) ? Number(value.toFixed(digits)) : value;
}

/** Mean of rgb over every texel of an rgba16float atlas (borders included). */
async function atlasMean(device: GPUDevice, ddgi: DDGITechnique): Promise<number> {
    const field = ddgi.field;
    assert(field, 'no field');
    const halves = new Uint16Array((await readTextureBytes(device, field.irradiance, 8)).buffer);
    let sum = 0;
    for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
    return sum / ((halves.length / 4) * 3);
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
    readonly name: string;
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
 * each. The side that goes first swaps every round: on the power-limited RTX
 * the second of two back-to-back timings runs about 6% slow (NOTES, B3), and
 * a first version that always timed A first read A 0.93x of B with the same
 * trees in both.
 */
async function timeInTurn(sides: readonly Side[]): Promise<Record<string, number>> {
    const frames = new Map<string, number[]>(sides.map((side) => [side.name, []]));
    for (let round = 0; round < ROUNDS; round++) {
        const order = round % 2 === 0 ? sides : [...sides].reverse();
        for (const side of order) frames.get(side.name)?.push(...(await runFrames(side, BLOCK_FRAMES)));
    }
    return Object.fromEntries(sides.map((side) => [side.name, median(side.times.msOf(frames.get(side.name) ?? []))]));
}

/** What the natural swap did (step 5). */
interface NaturalSwap {
    readonly swaps: number;
    readonly framesWithSwaps: number;
    readonly frames: number;
    readonly firstSwapMs: number;
    readonly lastSwapMs: number;
    readonly gpuTreesAtStart: number;
    readonly memory: AccelerationMemory;
}

/** Step 5: the defaults, frames until tinybvh's trees are all in, the swaps' spread and the node buffer. */
async function naturalSwap(device: GPUDevice, manifest: Parameters<typeof startViewerScene>[1]): Promise<NaturalSwap> {
    const scene = await startViewerScene(device, manifest, { gpuTiming: true, label: `swap ${manifest.id} natural` });
    const side: Side = { name: 'natural', scene, times: new TreePassTimes() };
    const rays = scene.platform.rays;
    const startedAt = performance.now();
    const gpuTreesAtStart = rays.meshTreeCounts.onGpuTrees;
    let swaps = rays.meshTreeCounts.swaps;
    let framesWithSwaps = 0;
    let frames = 0;
    let firstSwapMs = NaN;
    let lastSwapMs = NaN;
    // Until no tinybvh build runs and none waits to be placed (a frame after the last arrives), at most a minute.
    let quietFrames = 0;
    while (quietFrames < 2 && performance.now() - startedAt < 60_000) {
        await runFrames(side, 1);
        frames++;
        const now = rays.meshTreeCounts.swaps;
        if (now > swaps) {
            framesWithSwaps++;
            const at = performance.now() - startedAt;
            if (Number.isNaN(firstSwapMs)) firstSwapMs = at;
            lastSwapMs = at;
        }
        swaps = now;
        quietFrames = rays.refining || rays.meshTreeCounts.onGpuTrees > 0 ? 0 : quietFrames + 1;
    }
    const result = { swaps, framesWithSwaps, frames, firstSwapMs, lastSwapMs, gpuTreesAtStart, memory: rays.memory };
    scene.dispose();
    return result;
}

// The adapter's own device and limits (Sponza outgrows the default 128 MiB storage binding), as the viewer makes them.
await runThreePage(async ({ device, adapterSummary, log, data }) => {
    const onRtx = /nvidia/i.test(adapterSummary.vendor);
    const manifests = loadAllManifests();
    const failures: string[] = [];
    Object.assign(data, { units: UNITS, rounds: ROUNDS, blockFrames: BLOCK_FRAMES });

    for (const sceneId of SCENES) {
        const manifest = manifests.find((entry) => entry.id === sceneId);
        assert(manifest, `no manifest '${sceneId}'`);
        const gpuFirst: Side = {
            name: 'gpu-first',
            scene: await startViewerScene(device, manifest, { gpuTiming: true, label: `swap ${sceneId} gpu-first`, meshTrees: { holdRefinements: true } }),
            times: new TreePassTimes(),
        };
        const tinybvhOnly: Side = {
            name: 'tinybvh-only',
            scene: await startViewerScene(device, manifest, { gpuTiming: true, label: `swap ${sceneId} tinybvh-only`, meshTrees: { gpuFirst: false } }),
            times: new TreePassTimes(),
        };
        const rays = gpuFirst.scene.platform.rays;
        const firstCounts = rays.meshTreeCounts;
        // The GPU trees were just copied in: their batch's buffers are freed at the next update.
        const memoryAtFirstTrees = rays.memory;
        log(
            `${sceneId}: first trees ${firstCounts.onGpuTrees} on the GPU, ${firstCounts.onTinybvhTrees} from tinybvh ` +
                `(${firstCounts.gpuTreesRefused} GPU trees refused); GPU build ${round(firstCounts.lastGpuBuildMs, 1)} ms to its stats`,
        );
        assert(rays.gpuFirstTrees, `${sceneId}: the platform builds no first trees on the GPU on this device`);
        assert(firstCounts.onGpuTrees > 0, `${sceneId}: no mesh got a GPU tree`);

        for (const side of [gpuFirst, tinybvhOnly]) await warmUp(side.scene);
        for (const side of [gpuFirst, tinybvhOnly]) await runFrames(side, CONVERGE_FRAMES);
        const beforeSwap = await timeInTurn([gpuFirst, tinybvhOnly]);
        const gpuTreesRatio = beforeSwap['gpu-first'] / beforeSwap['tinybvh-only'];
        log(`${sceneId}: tree passes on GPU trees ${round(beforeSwap['gpu-first'])} ms, tinybvh only ${round(beforeSwap['tinybvh-only'])} ms (${round(gpuTreesRatio)}x)`);

        // The swap frame.
        await rays.whenRefined();
        assert(rays.meshTreeCounts.onGpuTrees === firstCounts.onGpuTrees, `${sceneId}: held trees were swapped in before the release`);
        const recompilesBefore = gpuFirst.scene.platform.stats.recompiles;
        const means: number[] = [];
        for (let index = 0; index < FRAMES_BEFORE_SWAP; index++) {
            await runFrames(gpuFirst, 1);
            means.push(await atlasMean(device, gpuFirst.scene.ddgi));
        }
        rays.releaseRefinements();
        await runFrames(gpuFirst, 1);
        const afterSwapFrame = await atlasMean(device, gpuFirst.scene.ddgi);
        const swapCounts = rays.meshTreeCounts;
        await runFrames(gpuFirst, FRAMES_AFTER_SWAP);
        const later = await atlasMean(device, gpuFirst.scene.ddgi);
        const lastBefore = means[means.length - 1];
        const swapChange = afterSwapFrame / lastBefore - 1;
        const normalChanges = means.slice(1).map((mean, index) => Math.abs(mean / means[index] - 1));
        const recompiles = gpuFirst.scene.platform.stats.recompiles - recompilesBefore;
        log(
            `${sceneId}: swap frame: ${swapCounts.swaps} trees swapped, atlas mean ${lastBefore.toPrecision(6)} -> ${afterSwapFrame.toPrecision(6)} ` +
                `(${(swapChange * 100).toFixed(4)}%; normal frames ${normalChanges.map((change) => (change * 100).toFixed(4)).join(', ')}%), ` +
                `${FRAMES_AFTER_SWAP} frames later ${later.toPrecision(6)}; recompiles ${recompiles}`,
        );
        if (swapCounts.onGpuTrees !== 0) failures.push(`${sceneId}: ${swapCounts.onGpuTrees} meshes were still on GPU trees after the swap frame`);
        if (Math.abs(swapChange) > SWAP_MEAN_TOLERANCE) failures.push(`${sceneId}: the field's mean changed ${(swapChange * 100).toFixed(3)}% across the swap frame (bar 0.5%)`);
        if (recompiles !== 0) failures.push(`${sceneId}: ${recompiles} compute pipelines were made across the swap`);

        const afterSwap = await timeInTurn([gpuFirst, tinybvhOnly]);
        const afterRatio = afterSwap['gpu-first'] / afterSwap['tinybvh-only'];
        log(`${sceneId}: tree passes after the swap ${round(afterSwap['gpu-first'])} ms, tinybvh only ${round(afterSwap['tinybvh-only'])} ms (${round(afterRatio)}x)`);
        const judgeSpeed = onRtx && sceneId === SPEED_SCENE;
        if (judgeSpeed && Math.abs(afterRatio - 1) > TRACE_SPEED_TOLERANCE) failures.push(`${sceneId}: after the swap the tree passes take ${round(afterRatio)}x tinybvh-only's (bar within 5%)`);

        const tinybvhNodes = tinybvhOnly.scene.platform.rays.memory.nodes;
        data[sceneId] = {
            meshes: firstCounts.onGpuTrees + firstCounts.onTinybvhTrees,
            firstTrees: firstCounts,
            treePassMs: { beforeSwap, afterSwap },
            speedJudged: judgeSpeed,
            gpuTreesRatio: round(gpuTreesRatio),
            afterSwapRatio: round(afterRatio),
            swap: { atlasMeanBefore: lastBefore, atlasMeanAfter: afterSwapFrame, change: swapChange, normalChanges, atlasMeanLater: later, swaps: swapCounts.swaps, recompiles },
            memoryAtFirstTrees,
            memoryAfterSwap: gpuFirst.scene.platform.rays.memory,
            memoryTinybvhOnly: tinybvhOnly.scene.platform.rays.memory,
        };
        gpuFirst.scene.dispose();
        tinybvhOnly.scene.dispose();

        // 5. The natural swap: a third platform with the defaults, as a host gets them.
        const natural = await naturalSwap(device, manifest);
        log(
            `${sceneId}: natural swap: ${natural.swaps} trees swapped over ${natural.framesWithSwaps} frames, ${round(natural.firstSwapMs)}-${round(natural.lastSwapMs)} ms after the first trees; ` +
                `node buffer ${round(natural.memory.nodes / 1e6, 1)} MB (${round(natural.memory.nodesInUse / 1e6, 1)} MB in use), tinybvh only ${round(tinybvhNodes / 1e6, 1)} MB`,
        );
        (data[sceneId] as Record<string, unknown>).naturalSwap = { ...natural, tinybvhOnlyNodes: tinybvhNodes };
    }
    assert(failures.length === 0, failures.join('\n'));
});
