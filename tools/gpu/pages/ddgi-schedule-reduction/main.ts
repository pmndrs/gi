/**
 * T2.4b check 4: the GPU scheduler (wgsl/schedule.ts) counts its histogram
 * and compacts its list subgroups first, with one declared fallback, and
 * both must agree exactly on the same input; and the scheduler costs at most
 * 0.05 ms at 20,000 probes.
 *
 * The kernels run on their own here (the technique's four dispatches, built
 * by the platform's kernel builder from the same WGSL), on a synthetic
 * volume of 40 × 20 × 25 = 20,000 probes, with a camera inside it and
 * history records written by the page. Scenarios:
 * - `still`: nothing owed. The list must be exactly M1's round robin: the
 *   scrambled order's first entries from the cursor (CPU twin, schedule.ts).
 * - `local`: 0.5% of the probes owed (a light change), fewer than the
 *   priority share, plus feedback requests. Every owed probe is listed, and
 *   the two reductions give the same list, entry for entry in the round
 *   robin's part, and the same set in the priority part.
 * - `global`: every probe owed at random urgencies (a sun change). The share
 *   fills up inside a bucket: the histograms, the boundary, the counts and
 *   the set above the boundary must be the same; which probes of the
 *   boundary bucket get in is whichever reserve first, so only their number
 *   is compared. The list must hold no probe twice.
 * In every scenario the histogram is checked against the probes' buckets
 * counted on the CPU from the same records, after the kernels ran.
 *
 * With VITE_DDGI_SCHEDULE_TIMING=1 (run it under the bench lock for the
 * clock rule): each scenario's four dispatches, in one pass as the technique
 * runs them, timed with timestamps over TIMED_ROUNDS × TIMED_FRAMES
 * back-to-back frames, median per reduction. Check 4's bar applies to the default reduction
 * (subgroups where the device has them).
 *
 * `pnpm test:gpu --filter ddgi-schedule-reduction --gpu nvidia` (or intel).
 */

import { assert, runGpuTest } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { cameraFalloffDistance, cameraWeight, frustumPlanes, inViewMargin } from '../../../../packages/core/src/techniques/ddgi/camera.ts';
import {
    FRUSTUM_PLANE_WORDS,
    PRIORITY_BUCKETS,
    PROBE_HISTORY_UPDATES,
    PROBE_HISTORY_WORD,
    PROBE_HISTORY_WORDS,
    SCHEDULE_STATE_WORD,
    SCHEDULE_STATE_WORDS,
    UNIFORM_BYTES,
    UNIFORM_WORD,
    UNIFORM_WORDS,
} from '../../../../packages/core/src/techniques/ddgi/layout.ts';
import { priorityBucket, priorityCount, scheduleOnCpu, scrambleStride } from '../../../../packages/core/src/techniques/ddgi/schedule.ts';
import { DEFAULT_PRIORITY_SHARE, probeGrid } from '../../../../packages/core/src/techniques/ddgi/settings.ts';
import { SCHEDULE_ENTRY_POINTS, SCHEDULE_WORKGROUP_SIZE, scheduleKernelWgsl, type ScheduleReduction } from '../../../../packages/core/src/techniques/ddgi/wgsl/schedule.ts';
import { readBuffer, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { bookmarkCamera } from '../ddgi-shared/response.ts';

const env = import.meta.env as Record<string, unknown>;
const TIMING = env.VITE_DDGI_SCHEDULE_TIMING === '1';

const VOLUME = { center: [0, 0, 0] as [number, number, number], size: [39, 19, 24] as [number, number, number], spacing: 1 };
/** Sponza's calibrated batch at 2 ms on the RTX (T2.0's worktree: 293): the list the kernels fill. */
const UPDATE_COUNT = 293;
const FRAME = 1234;
/**
 * Timed frames per round (one query set: 2 timestamps a frame, at most 4,096
 * queries), and rounds. Frames go to the GPU FRAMES_PER_SUBMIT at a time so
 * it runs them back to back: one 30 µs frame per submit would leave the GPU
 * idle between them and its clocks low (the clock rule needs it loaded).
 */
const TIMED_FRAMES = 2000;
const TIMED_ROUNDS = 12;
const FRAMES_PER_SUBMIT = 250;
/** The local scenario's owed probes (a light change's box): 100 of 20,000, under the priority share. */
const LOCAL_OWED_SHARE = 0.005;
/** Check 4's bar. */
const MAX_SCHEDULE_MS = 0.05;

type Scenario = 'still' | 'local' | 'global';
const SCENARIOS: readonly Scenario[] = ['still', 'local', 'global'];

const SCHEDULE_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'ddgi/schedule group 1 (test)',
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
};

/** A seeded random number in [0, 1) (an LCG), so both reductions get the same input. */
function seeded(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

const grid = probeGrid(VOLUME);
const camera = bookmarkCamera({ name: 'test', position: [-15, -5, -8], target: [5, 0, 3], fovYDeg: 60 }, 16 / 9);
const planes = frustumPlanes(camera);

/** The scenario's history records and feedback requests. */
function scenarioInput(scenario: Scenario): { history: Uint32Array<ArrayBuffer>; requests: Uint32Array<ArrayBuffer> } {
    const history = new Uint32Array(grid.probeCount * PROBE_HISTORY_WORDS);
    const floats = new Float32Array(history.buffer);
    const requests = new Uint32Array(grid.probeCount);
    const random = seeded(scenario === 'local' ? 7 : 11);
    for (let probe = 0; probe < grid.probeCount; probe++) {
        const base = probe * PROBE_HISTORY_WORDS;
        // Every probe stamped by an earlier frame (the scheduler skips this frame's only).
        history[base + PROBE_HISTORY_WORD.scheduleAndFlags] = ((FRAME - 1 - (probe % 50)) & 0xffffff) << 8;
        const owed = scenario === 'global' ? 1 + Math.floor(random() * 8) : scenario === 'local' && random() < LOCAL_OWED_SHARE ? 8 : 0;
        history[base + PROBE_HISTORY_WORD.updates] = 20 | (owed << PROBE_HISTORY_UPDATES.owedShift);
        // Urgencies from the tolerance (0.05) to MAX_GLOBAL_CHANGE (100), spread on a log scale.
        if (owed > 0) floats[base + PROBE_HISTORY_WORD.urgency] = 0.05 * Math.pow(2000, random());
        if (scenario === 'local' && random() < 0.002) requests[probe] = new Uint32Array(new Float32Array([0.2 * random()]).buffer)[0];
    }
    return { history, requests };
}

function uniformData(cursorStride: number): ArrayBuffer {
    const buffer = new ArrayBuffer(UNIFORM_BYTES);
    const floats = new Float32Array(buffer, 0, UNIFORM_WORDS);
    const words = new Uint32Array(buffer, 0, UNIFORM_WORDS);
    floats.set([...grid.origin, 0], UNIFORM_WORD.volumeOrigin);
    floats.set([...grid.spacing, 0], UNIFORM_WORD.volumeSpacing);
    floats.set([...grid.counts, 0], UNIFORM_WORD.volumeCounts);
    words[UNIFORM_WORD.updateCount] = UPDATE_COUNT;
    words[UNIFORM_WORD.frame] = FRAME;
    words[UNIFORM_WORD.priorityCount] = priorityCount(UPDATE_COUNT, DEFAULT_PRIORITY_SHARE);
    words[UNIFORM_WORD.scrambleStride] = cursorStride;
    floats[UNIFORM_WORD.cameraFalloffDistance] = cameraFalloffDistance(grid);
    floats.set([...camera.position, inViewMargin(grid)], UNIFORM_WORD.cameraPosition);
    planes.forEach((plane, index) => floats.set(plane, FRUSTUM_PLANE_WORDS[index]));
    return buffer;
}

interface Run {
    readonly state: Uint32Array;
    readonly list: Uint32Array;
    readonly history: Uint32Array;
}

async function runScheduler(device: GPUDevice, platform: GIPlatformImpl, reduction: ScheduleReduction, scenario: Scenario, cursorProbe: number, timing: boolean) {
    const code = scheduleKernelWgsl(reduction);
    const kernels = await Promise.all(
        Object.values(SCHEDULE_ENTRY_POINTS).map((entryPoint) => platform.kernels.computeKernel({ label: `test ${entryPoint} (${reduction})`, modules: [], code, entryPoint, ownGroups: [SCHEDULE_GROUP] })),
    );
    const stride = scrambleStride(grid.probeCount);
    const input = scenarioInput(scenario);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const uniforms = device.createBuffer({ size: UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const history = device.createBuffer({ size: input.history.byteLength, usage });
    const requests = device.createBuffer({ size: input.requests.byteLength, usage });
    const state = device.createBuffer({ size: SCHEDULE_STATE_WORDS * 4, usage });
    const list = device.createBuffer({ size: UPDATE_COUNT * 4, usage });
    device.queue.writeBuffer(uniforms, 0, uniformData(stride));
    const group = device.createBindGroup({
        layout: kernels[0].ownLayouts[0],
        entries: [uniforms, history, requests, state, list].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const groups = Math.ceil(grid.probeCount / SCHEDULE_WORKGROUP_SIZE);
    const reset = () => {
        device.queue.writeBuffer(history, 0, input.history);
        device.queue.writeBuffer(requests, 0, input.requests);
        const stateWords = new Uint32Array(SCHEDULE_STATE_WORDS);
        stateWords[SCHEDULE_STATE_WORD.cursorProbe] = cursorProbe;
        device.queue.writeBuffer(state, 0, stateWords);
    };
    const encodeFrame = (encoder: GPUCommandEncoder, timestampWrites?: GPUComputePassTimestampWrites) => {
        encoder.clearBuffer(state, SCHEDULE_STATE_WORD.histogram * 4, PRIORITY_BUCKETS * 4);
        const pass = encoder.beginComputePass({ timestampWrites });
        kernels.forEach((kernel, index) => {
            kernel.bind(pass);
            pass.setBindGroup(1, group);
            // count and write over every probe; select and fill one workgroup.
            pass.dispatchWorkgroups(index === 0 || index === 2 ? groups : 1);
        });
        pass.end();
    };
    reset();
    const encoder = device.createCommandEncoder();
    encodeFrame(encoder);
    device.queue.submit([encoder.finish()]);
    const run: Run = {
        state: new Uint32Array(await readBuffer(device, state, SCHEDULE_STATE_WORDS * 4)),
        list: new Uint32Array(await readBuffer(device, list, UPDATE_COUNT * 4)),
        history: new Uint32Array(await readBuffer(device, history, input.history.byteLength)),
    };

    let medianMs: number | null = null;
    if (timing) {
        // Back to back, the same input every frame (the requests are taken by the first, which is how a still frame looks).
        const querySet = device.createQuerySet({ type: 'timestamp', count: 2 * TIMED_FRAMES });
        const resolve = device.createBuffer({ size: 16 * TIMED_FRAMES, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        reset();
        const ms: number[] = [];
        for (let round = 0; round < TIMED_ROUNDS; round++) {
            for (let start = 0; start < TIMED_FRAMES; start += FRAMES_PER_SUBMIT) {
                const timed = device.createCommandEncoder();
                for (let frame = start; frame < Math.min(TIMED_FRAMES, start + FRAMES_PER_SUBMIT); frame++) {
                    encodeFrame(timed, { querySet, beginningOfPassWriteIndex: 2 * frame, endOfPassWriteIndex: 2 * frame + 1 });
                }
                if (start + FRAMES_PER_SUBMIT >= TIMED_FRAMES) timed.resolveQuerySet(querySet, 0, 2 * TIMED_FRAMES, resolve, 0);
                device.queue.submit([timed.finish()]);
            }
            const stamps = new BigUint64Array(await readBuffer(device, resolve, 16 * TIMED_FRAMES));
            for (let frame = 0; frame < TIMED_FRAMES; frame++) {
                const value = Number(stamps[2 * frame + 1] - stamps[2 * frame]) / 1e6;
                if (value > 0 && value < 100) ms.push(value);
            }
        }
        ms.sort((a, b) => a - b);
        medianMs = ms[Math.floor(ms.length / 2)];
        querySet.destroy();
        resolve.destroy();
    }
    for (const buffer of [uniforms, history, requests, state, list]) buffer.destroy();
    return { run, input, stride, medianMs };
}

/** The buckets the CPU counts from the records after the kernels ran (count took the requests in). */
function cpuBuckets(history: Uint32Array): number[] {
    const floats = new Float32Array(history.buffer);
    const buckets: number[] = [];
    for (let probe = 0; probe < grid.probeCount; probe++) {
        const base = probe * PROBE_HISTORY_WORDS;
        const owed = (history[base + PROBE_HISTORY_WORD.updates] >>> PROBE_HISTORY_UPDATES.owedShift) & PROBE_HISTORY_UPDATES.owedMask;
        if (owed === 0) {
            buckets.push(-1);
            continue;
        }
        const x = probe % grid.counts[0];
        const z = Math.floor(probe / grid.counts[0]) % grid.counts[2];
        const y = Math.floor(probe / (grid.counts[0] * grid.counts[2]));
        const point = [x, y, z].map((value, axis) => grid.origin[axis] + value * grid.spacing[axis]) as [number, number, number];
        buckets.push(priorityBucket(floats[base + PROBE_HISTORY_WORD.urgency] * cameraWeight(point, camera, planes, grid)));
    }
    return buckets;
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const platform = new GIPlatformImpl(device, { label: 'schedule reduction' });
        const reductions: ScheduleReduction[] = device.features.has('subgroups') ? ['subgroups', 'workgroup'] : ['workgroup'];
        if (reductions.length === 1) log('no subgroups on this device: only the declared fallback runs');
        const results: Record<string, unknown> = { probes: grid.probeCount, updateCount: UPDATE_COUNT, reductions };
        const failures: string[] = [];
        const cursorProbe = 4321;
        for (const scenario of SCENARIOS) {
            const runs = [];
            for (const reduction of reductions) runs.push({ reduction, ...(await runScheduler(device, platform, reduction, scenario, cursorProbe, TIMING)) });
            const [first] = runs;
            const histogram = (run: Run) => Array.from(run.state.subarray(SCHEDULE_STATE_WORD.histogram, SCHEDULE_STATE_WORD.histogram + PRIORITY_BUCKETS));
            const word = (run: Run, name: keyof typeof SCHEDULE_STATE_WORD) => run.state[SCHEDULE_STATE_WORD[name]];
            const listed = word(first.run, 'listed');
            const aboveTotal = word(first.run, 'aboveTotal');
            const summary: Record<string, unknown> = {
                owed: word(first.run, 'owed'),
                listed,
                aboveTotal,
                boundaryBucket: new Int32Array(first.run.state.buffer)[SCHEDULE_STATE_WORD.boundaryBucket],
                cursorAfter: word(first.run, 'cursorProbe'),
                histogram: histogram(first.run),
                medianMs: Object.fromEntries(runs.map((run) => [run.reduction, run.medianMs])),
            };
            results[scenario] = summary;

            // The histogram against the CPU's count of the same records.
            const cpuHistogram = new Array<number>(PRIORITY_BUCKETS).fill(0);
            for (const bucket of cpuBuckets(first.run.history)) if (bucket >= 0) cpuHistogram[bucket]++;
            const cpuOwed = cpuHistogram.reduce((sum, count) => sum + count, 0);
            if (cpuOwed !== word(first.run, 'owed')) failures.push(`${scenario}: the GPU counted ${word(first.run, 'owed')} owed probes, the CPU ${cpuOwed}`);
            // Bucket edges can round either way between the CPU's doubles and the GPU's f32: allow a probe or two to change bucket.
            const histogramMoved = cpuHistogram.reduce((sum, count, bucket) => sum + Math.abs(count - histogram(first.run)[bucket]), 0);
            summary.cpuHistogramDifference = histogramMoved;
            if (histogramMoved > 4) failures.push(`${scenario}: the histogram differs from the CPU's by ${histogramMoved} probes`);

            for (const other of runs.slice(1)) {
                const label = `${scenario}, ${first.reduction} against ${other.reduction}`;
                if (histogram(first.run).join(',') !== histogram(other.run).join(',')) failures.push(`${label}: the histograms differ`);
                for (const name of ['owed', 'listed', 'aboveTotal', 'boundaryBucket', 'boundaryRoom', 'cursorProbe'] as const) {
                    if (word(first.run, name) !== word(other.run, name)) failures.push(`${label}: ${name} ${word(first.run, name)} against ${word(other.run, name)}`);
                }
                const aboveA = [...first.run.list.subarray(0, aboveTotal)].sort((a, b) => a - b).join(',');
                const aboveB = [...other.run.list.subarray(0, aboveTotal)].sort((a, b) => a - b).join(',');
                if (aboveA !== aboveB) failures.push(`${label}: the probes above the boundary differ`);
                // With no partial bucket, the stamped set is the same, so the round robin's part matches entry for entry.
                if (scenario !== 'global' && first.run.list.subarray(listed).join(',') !== other.run.list.subarray(listed).join(',')) failures.push(`${label}: the round robin's part differs`);
            }
            for (const run of runs) {
                if (new Set(run.run.list).size !== UPDATE_COUNT) failures.push(`${scenario} (${run.reduction}): a probe is listed twice`);
            }
            if (scenario !== 'global') {
                // Every owed probe fits: the list is the owed probes, then the round robin from the cursor skipping them (the CPU twin).
                const owedProbes = new Set([...first.run.list.subarray(0, listed)]);
                const probes = Array.from({ length: grid.probeCount }, (_, probe) => ({ owed: owedProbes.has(probe) ? 1 : 0, priority: 1 }));
                const cpu = scheduleOnCpu(probes, UPDATE_COUNT, DEFAULT_PRIORITY_SHARE, cursorProbe, first.stride);
                if (cpu.roundRobin.join(',') !== [...first.run.list.subarray(listed)].join(',')) failures.push(`${scenario}: the round robin's part is not the CPU twin's`);
                if (cpu.cursorProbe !== word(first.run, 'cursorProbe')) failures.push(`${scenario}: the cursor moved to ${word(first.run, 'cursorProbe')}, the CPU twin to ${cpu.cursorProbe}`);
                if (listed !== word(first.run, 'owed')) failures.push(`${scenario}: ${word(first.run, 'owed')} owed but ${listed} listed`);
            }
            log(
                `${scenario}: ${summary.owed} owed, ${listed} listed at priority (above the boundary ${aboveTotal}, boundary bucket ${summary.boundaryBucket}); ` +
                    `cursor ${cursorProbe} -> ${summary.cursorAfter}; CPU histogram difference ${histogramMoved}` +
                    (TIMING ? `; median ms ${runs.map((run) => `${run.reduction} ${run.medianMs?.toFixed(4)}`).join(', ')}` : ''),
            );
            if (TIMING && first.medianMs !== null && first.medianMs > MAX_SCHEDULE_MS) failures.push(`${scenario}: the scheduler took ${first.medianMs.toFixed(4)} ms (${first.reduction}; bar ${MAX_SCHEDULE_MS} ms)`);
        }
        platform.dispose();
        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['subgroups', 'timestamp-query'] },
);
