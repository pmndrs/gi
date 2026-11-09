/**
 * RC1.3: the platform's GPU utilities (packages/ddgi-core/src/platform/compute)
 * against their CPU twins, on both paths.
 *
 * For every size (0, 1, one scan tile - 1, one tile, one tile + 1, a few
 * seeded random sizes, one tile of tiles + 1, where the across-tiles kernel
 * takes a second round, and 10 M):
 * - the prefix scan into an output buffer and in place: every prefix and the
 *   total equal the CPU's;
 * - the stream compaction of random flags (half kept; kept flags are 1 or 3,
 *   since any non-zero flag keeps): the kept indices and their count equal the
 *   CPU's;
 * - the indirect arguments sized from the compaction's GPU count for a
 *   64-thread kernel equal the CPU's (at 10 M: 78,000 workgroups, split over
 *   two rows), and a real `dispatchWorkgroupsIndirect` of a kernel built by
 *   the platform's kernel builder with the args module counts its
 *   invocations: exactly `count` within the count, every kept index read
 *   once, in increasing order, each with a non-zero flag.
 *
 * Paths: the declared fallback (workgroup memory) on a second device made
 * from a new adapter without `subgroups` (NOTES: one adapter makes one
 * device), run first and destroyed before the main device works (the Intel
 * UHD has hung with two devices busy, ddgi-distance-format); then subgroups
 * on the main device. The two paths' results must be identical, size by size.
 *
 * With VITE_PLATFORM_COMPUTE_TIMING=1 (run it under the bench lock): the
 * scan, the compaction and the arguments writer at 1 M and 10 M items, each
 * its own pass timed with timestamps, back to back so the GPU stays loaded;
 * median and p95 per reduction (the fallback forced on the main device too).
 * `to-csv.ts` turns the run into rows of bench/results/compute.csv.
 *
 * `pnpm test:gpu --filter platform-compute --gpu nvidia` (or intel).
 */

import { SCAN_TILE_ITEMS } from '../../../../packages/bvh/src/gpu-scan.ts';
import type { ComputeReduction, GpuCompaction, GpuIndirectArgs, GpuPrefixScan } from '../../../../packages/ddgi-core/src/platform/contract/compute.ts';
import { compactOnCpu, exclusiveScanOnCpu, indirectArgsOnCpu } from '../../../../packages/ddgi-core/src/platform/compute/cpu-reference.ts';
import { PlatformGpuCompute } from '../../../../packages/ddgi-core/src/platform/compute/gpu-compute.ts';
import { INDIRECT_ARGS_COUNT_WORD } from '../../../../packages/ddgi-core/src/platform/compute/wgsl.ts';
import { PlatformKernelBuilder } from '../../../../packages/ddgi-core/src/platform/kernel-builder.ts';
import type { ComputeKernel } from '../../../../packages/ddgi-core/src/platform/contract/wgsl.ts';
import { assert, drainDevice, runGpuTest, watchDevice } from '../../client.ts';
import { readBuffer, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';

const env = import.meta.env as Record<string, unknown>;
const TIMING = env.VITE_PLATFORM_COMPUTE_TIMING === '1';

const TEN_MILLION = 10_000_000;
/** Threads per workgroup of the consuming kernel, and so items per workgroup for the arguments. */
const CONSUMER_WORKGROUP = 64;
/** Timed passes per round (2 timestamps each, one query set), rounds, and passes per submit. */
const TIMED_PASSES = 400;
const TIMED_ROUNDS = 5;
const PASSES_PER_SUBMIT = 100;

/** A seeded random number in [0, 1) (an LCG), so both paths get the same input. */
function seeded(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

function testSizes(): number[] {
    const random = seeded(2026);
    const randoms = [0, 1, 2].map(() => 2 + Math.floor(random() * 300_000));
    return [0, 1, SCAN_TILE_ITEMS - 1, SCAN_TILE_ITEMS, SCAN_TILE_ITEMS + 1, ...randoms, SCAN_TILE_ITEMS * SCAN_TILE_ITEMS + 1, TEN_MILLION];
}

/** The same input for every size and path: values 0..15 to scan, and flags (half kept, as 1 or 3). */
function makeInput(capacity: number): { values: Uint32Array<ArrayBuffer>; flags: Uint32Array<ArrayBuffer> } {
    const random = seeded(7);
    const values = new Uint32Array(capacity);
    const flags = new Uint32Array(capacity);
    for (let index = 0; index < capacity; index++) {
        values[index] = Math.floor(random() * 16);
        const draw = random();
        flags[index] = draw < 0.25 ? 1 : draw < 0.5 ? 3 : 0;
    }
    return { values, flags };
}

/** FNV-1a over words, so two paths' outputs can be compared without keeping both. */
function digest(words: Uint32Array): string {
    let hash = 0x811c9dc5;
    for (let index = 0; index < words.length; index++) {
        hash = Math.imul(hash ^ words[index], 0x01000193) >>> 0;
    }
    return `${words.length}:${hash.toString(16)}`;
}

function firstDifference(gpu: Uint32Array, cpu: Uint32Array): string | null {
    if (gpu.length !== cpu.length) return `${gpu.length} words against ${cpu.length}`;
    for (let index = 0; index < cpu.length; index++) {
        if (gpu[index] !== cpu[index]) return `word ${index}: ${gpu[index]} against ${cpu[index]}`;
    }
    return null;
}

/** The consuming kernel: counts what an indirect dispatch over the kept indices runs. */
const CONSUMER_WGSL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> consumerArgs: GiIndirectArgs;
@group(1) @binding(1) var<storage, read> consumerKept: array<u32>;
@group(1) @binding(2) var<storage, read> consumerFlags: array<u32>;
// Invocations within the count, invocations launched, sum of the kept indices read (wrapping), problems seen.
@group(1) @binding(3) var<storage, read_write> consumerTally: array<atomic<u32>, 4>;

var<workgroup> workgroupTally: array<atomic<u32>, 4>;

@compute @workgroup_size(${CONSUMER_WORKGROUP})
fn consumeKept(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let item = giIndirectItemIndex(workgroup, groups, localIndex, ${CONSUMER_WORKGROUP}u);
    atomicAdd(&workgroupTally[1], 1u);
    if (item < consumerArgs.count) {
        atomicAdd(&workgroupTally[0], 1u);
        let kept = consumerKept[item];
        atomicAdd(&workgroupTally[2], kept);
        // Every kept item has a non-zero flag, and the list increases (stable order).
        let outOfOrder = item > 0u && consumerKept[item - 1u] >= kept;
        if (consumerFlags[kept] == 0u || outOfOrder) {
            atomicAdd(&workgroupTally[3], 1u);
        }
    }
    // One global atomic per workgroup and word, not one per invocation.
    workgroupBarrier();
    if (localIndex < 4u) {
        atomicAdd(&consumerTally[localIndex], atomicLoad(&workgroupTally[localIndex]));
    }
}
`;

const CONSUMER_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'platform-compute consumer group 1 (test)',
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
};

/** Everything one device's runs share: the utilities at 10 M capacity and their buffers. */
interface Rig {
    readonly device: GPUDevice;
    readonly compute: PlatformGpuCompute;
    readonly values: GPUBuffer;
    readonly scanned: GPUBuffer;
    readonly inPlace: GPUBuffer;
    readonly flags: GPUBuffer;
    readonly tally: GPUBuffer;
    readonly scan: GpuPrefixScan;
    readonly scanInPlace: GpuPrefixScan;
    readonly compaction: GpuCompaction;
    readonly args: GpuIndirectArgs;
    readonly consumer: ComputeKernel;
    readonly consumerGroup: GPUBindGroup;
    readonly buffers: readonly GPUBuffer[];
}

async function makeRig(device: GPUDevice, reduction: ComputeReduction, capacity: number): Promise<Rig> {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const buffer = (label: string, size: number) => device.createBuffer({ label: `platform-compute ${label}`, size, usage });
    const values = buffer('values', capacity * 4);
    const scanned = buffer('scanned', capacity * 4);
    const inPlace = buffer('in place', capacity * 4);
    const flags = buffer('flags', capacity * 4);
    const tally = buffer('tally', 16);
    const compute = new PlatformGpuCompute(device, { label: 'platform-compute', reduction });
    const kernels = new PlatformKernelBuilder(device, 'platform-compute');
    const [scan, scanInPlace, compaction] = await Promise.all([
        compute.createPrefixScan({ label: 'platform-compute scan', input: values, output: scanned, capacity }),
        compute.createPrefixScan({ label: 'platform-compute scan in place', input: inPlace, capacity }),
        compute.createCompaction({ label: 'platform-compute compaction', flags, capacity }),
    ]);
    const args = await compute.createIndirectArgs({ label: 'platform-compute args', count: compaction.count, itemsPerWorkgroup: CONSUMER_WORKGROUP });
    const consumer = await kernels.computeKernel({
        label: 'platform-compute consumer',
        modules: [compute.indirectModule],
        code: CONSUMER_WGSL,
        entryPoint: 'consumeKept',
        ownGroups: [CONSUMER_GROUP],
    });
    const consumerGroup = device.createBindGroup({
        label: 'platform-compute consumer',
        layout: consumer.ownLayouts[0],
        entries: [args.buffer, compaction.indices, flags, tally].map((resource, binding) => ({ binding, resource: { buffer: resource } })),
    });
    return { device, compute, values, scanned, inPlace, flags, tally, scan, scanInPlace, compaction, args, consumer, consumerGroup, buffers: [values, scanned, inPlace, flags, tally] };
}

function disposeRig(rig: Rig): void {
    rig.scan.dispose();
    rig.scanInPlace.dispose();
    rig.compaction.dispose();
    rig.args.dispose();
    for (const buffer of rig.buffers) buffer.destroy();
}

interface SizeResult {
    readonly count: number;
    readonly scanTotal: number;
    readonly kept: number;
    readonly args: number[];
    readonly invocations: number;
    /** Digests of the outputs, compared between the paths. */
    readonly digests: Record<string, string>;
}

/** Runs every utility at one size and checks it against the CPU; failures go to `failures`. */
async function runSize(rig: Rig, count: number, input: ReturnType<typeof makeInput>, failures: string[], path: string): Promise<SizeResult> {
    const { device } = rig;
    const values = input.values.subarray(0, count);
    const flags = input.flags.subarray(0, count);
    if (count > 0) {
        device.queue.writeBuffer(rig.values, 0, values);
        device.queue.writeBuffer(rig.inPlace, 0, values);
        device.queue.writeBuffer(rig.flags, 0, flags);
    }
    const encoder = device.createCommandEncoder({ label: 'platform-compute run' });
    encoder.clearBuffer(rig.tally);
    // Stale totals must not pass for new ones.
    for (const word of [rig.scan.total, rig.scanInPlace.total, rig.compaction.count]) encoder.clearBuffer(word.buffer, word.offset, 4);
    const pass = encoder.beginComputePass({ label: 'platform-compute utilities' });
    rig.scan.encode(pass, count);
    rig.scanInPlace.encode(pass, count);
    rig.compaction.encode(pass, count);
    rig.args.encode(pass);
    rig.consumer.bind(pass);
    pass.setBindGroup(1, rig.consumerGroup);
    pass.dispatchWorkgroupsIndirect(rig.args.buffer, 0);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const word = async (buffer: GPUBuffer, offset = 0) => new Uint32Array(await readBuffer(device, buffer, offset + 4))[offset / 4];
    const scanTotal = await word(rig.scan.total.buffer, rig.scan.total.offset);
    const inPlaceTotal = await word(rig.scanInPlace.total.buffer, rig.scanInPlace.total.offset);
    const kept = await word(rig.compaction.count.buffer, rig.compaction.count.offset);
    const argsWords = new Uint32Array(await readBuffer(device, rig.args.buffer, 16));
    const tally = new Uint32Array(await readBuffer(device, rig.tally, 16));
    const read = async (buffer: GPUBuffer, words: number) => (words === 0 ? new Uint32Array(0) : new Uint32Array(await readBuffer(device, buffer, words * 4)));
    const scanned = await read(rig.scanned, count);
    const inPlace = await read(rig.inPlace, count);
    const indices = await read(rig.compaction.indices, kept);

    const label = `${path}, ${count} items`;
    const cpuScan = exclusiveScanOnCpu(values);
    const cpuKept = compactOnCpu(flags);
    if (scanTotal !== cpuScan.total) failures.push(`${label}: scan total ${scanTotal}, CPU ${cpuScan.total}`);
    if (inPlaceTotal !== cpuScan.total) failures.push(`${label}: in-place scan total ${inPlaceTotal}, CPU ${cpuScan.total}`);
    const scanDifference = firstDifference(scanned, cpuScan.prefix);
    if (scanDifference) failures.push(`${label}: scan into output differs at ${scanDifference}`);
    const inPlaceDifference = firstDifference(inPlace, cpuScan.prefix);
    if (inPlaceDifference) failures.push(`${label}: scan in place differs at ${inPlaceDifference}`);
    if (kept !== cpuKept.length) failures.push(`${label}: kept ${kept}, CPU ${cpuKept.length}`);
    else {
        const keptDifference = firstDifference(indices, cpuKept);
        if (keptDifference) failures.push(`${label}: kept indices differ at ${keptDifference}`);
    }

    const cpuArgs = indirectArgsOnCpu(cpuKept.length, CONSUMER_WORKGROUP, device.limits.maxComputeWorkgroupsPerDimension);
    const args = [...argsWords.subarray(0, 3)];
    if (args.join(',') !== cpuArgs.join(',')) failures.push(`${label}: indirect args [${args}], CPU [${cpuArgs}]`);
    if (argsWords[INDIRECT_ARGS_COUNT_WORD] !== cpuKept.length) failures.push(`${label}: the args' count is ${argsWords[INDIRECT_ARGS_COUNT_WORD]}, CPU ${cpuKept.length}`);
    // The consuming kernel: every workgroup the arguments asked for ran, and exactly the kept items were in range.
    const launched = cpuArgs[0] * cpuArgs[1] * cpuArgs[2] * CONSUMER_WORKGROUP;
    let keptSum = 0;
    for (const index of cpuKept) keptSum = (keptSum + index) >>> 0;
    if (tally[0] !== cpuKept.length) failures.push(`${label}: the indirect kernel saw ${tally[0]} items in range, expected ${cpuKept.length}`);
    if (tally[1] !== launched) failures.push(`${label}: the indirect dispatch ran ${tally[1]} invocations, expected ${launched}`);
    if (tally[2] !== keptSum) failures.push(`${label}: the indirect kernel's sum of kept indices is ${tally[2]}, expected ${keptSum}`);
    if (tally[3] !== 0) failures.push(`${label}: the indirect kernel found ${tally[3]} kept indices out of order or with a zero flag`);

    return {
        count,
        scanTotal,
        kept,
        args,
        invocations: tally[1],
        digests: { scan: digest(scanned), inPlace: digest(inPlace), kept: digest(indices), args: digest(argsWords), tally: digest(tally) },
    };
}

async function runPath(rig: Rig, input: ReturnType<typeof makeInput>, failures: string[], log: (line: string) => void): Promise<SizeResult[]> {
    const results: SizeResult[] = [];
    for (const count of testSizes()) {
        const result = await runSize(rig, count, input, failures, rig.compute.reduction);
        results.push(result);
        log(`${rig.compute.reduction}: ${count} items: total ${result.scanTotal}, kept ${result.kept}, args [${result.args}], ${result.invocations} invocations`);
    }
    return results;
}

interface TimingStats {
    readonly medianMs: number;
    readonly p95Ms: number;
    readonly samples: number;
}

/** One operation recorded back to back, each in its own timed pass; median and p95 of the pass times. */
async function timePasses(device: GPUDevice, record: (pass: GPUComputePassEncoder) => void): Promise<TimingStats> {
    const querySet = device.createQuerySet({ type: 'timestamp', count: 2 * TIMED_PASSES });
    const resolve = device.createBuffer({ size: 16 * TIMED_PASSES, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const ms: number[] = [];
    for (let round = 0; round < TIMED_ROUNDS; round++) {
        for (let start = 0; start < TIMED_PASSES; start += PASSES_PER_SUBMIT) {
            const encoder = device.createCommandEncoder({ label: 'platform-compute timing' });
            for (let index = start; index < Math.min(TIMED_PASSES, start + PASSES_PER_SUBMIT); index++) {
                const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * index, endOfPassWriteIndex: 2 * index + 1 } });
                record(pass);
                pass.end();
            }
            if (start + PASSES_PER_SUBMIT >= TIMED_PASSES) encoder.resolveQuerySet(querySet, 0, 2 * TIMED_PASSES, resolve, 0);
            device.queue.submit([encoder.finish()]);
        }
        const stamps = new BigUint64Array(await readBuffer(device, resolve, 16 * TIMED_PASSES));
        for (let index = 0; index < TIMED_PASSES; index++) {
            const value = Number(stamps[2 * index + 1] - stamps[2 * index]) / 1e6;
            if (value > 0 && value < 1000) ms.push(value);
        }
    }
    querySet.destroy();
    resolve.destroy();
    assert(ms.length > TIMED_PASSES, `only ${ms.length} usable timestamps`);
    ms.sort((a, b) => a - b);
    return { medianMs: ms[Math.floor(ms.length / 2)], p95Ms: ms[Math.floor(ms.length * 0.95)], samples: ms.length };
}

async function timeRig(rig: Rig, input: ReturnType<typeof makeInput>, log: (line: string) => void): Promise<Record<string, Record<string, TimingStats>>> {
    const timings: Record<string, Record<string, TimingStats>> = {};
    for (const count of [1_000_000, TEN_MILLION]) {
        rig.device.queue.writeBuffer(rig.values, 0, input.values.subarray(0, count));
        rig.device.queue.writeBuffer(rig.flags, 0, input.flags.subarray(0, count));
        // One encode first so the count reaches the parameters before the timed submits.
        const warm = rig.device.createCommandEncoder();
        const pass = warm.beginComputePass();
        rig.scan.encode(pass, count);
        rig.compaction.encode(pass, count);
        rig.args.encode(pass);
        pass.end();
        rig.device.queue.submit([warm.finish()]);
        await rig.device.queue.onSubmittedWorkDone();
        const scan = await timePasses(rig.device, (timed) => rig.scan.encode(timed, count));
        const compaction = await timePasses(rig.device, (timed) => rig.compaction.encode(timed, count));
        const args = await timePasses(rig.device, (timed) => rig.args.encode(timed));
        const name = count === TEN_MILLION ? 'items10M' : 'items1M';
        timings[name] = { scan, compaction, args };
        log(`timing ${rig.compute.reduction} ${name}: scan ${scan.medianMs.toFixed(4)} ms, compaction ${compaction.medianMs.toFixed(4)} ms, args ${args.medianMs.toFixed(4)} ms (medians)`);
    }
    return timings;
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        assert(device.features.has('subgroups'), 'the main device has no subgroups, so the subgroup path cannot be tested here');
        const capacity = TEN_MILLION;
        const input = makeInput(capacity);
        const failures: string[] = [];

        // The fallback first, on its own device without subgroups, destroyed before the main device works.
        const fallbackAdapter = await navigator.gpu.requestAdapter();
        assert(fallbackAdapter, 'no second adapter');
        const fallbackDevice = await fallbackAdapter.requestDevice();
        const fallbackErrors = watchDevice(fallbackDevice);
        assert(!fallbackDevice.features.has('subgroups'), 'the second device has subgroups although they were not requested');
        const fallbackRig = await makeRig(fallbackDevice, 'workgroup', capacity);
        assert(fallbackRig.compute.reduction === 'workgroup', `without subgroups the utilities chose ${fallbackRig.compute.reduction}`);
        const fallback = await runPath(fallbackRig, input, failures, log);
        disposeRig(fallbackRig);
        await drainDevice(fallbackDevice);
        fallbackDevice.destroy();
        assert(fallbackErrors.length === 0, `the device without subgroups reported errors: ${fallbackErrors.join(' | ')}`);

        const rig = await makeRig(device, 'subgroups', capacity);
        const subgroups = await runPath(rig, input, failures, log);

        // The split over y must be exercised: an indirect dispatch with x past the limit runs
        // nothing and raises no error (RC1.4, Chrome 157), so an overflow would otherwise pass silently.
        const limit = device.limits.maxComputeWorkgroupsPerDimension;
        const split = subgroups.find((result) => result.kept > limit * CONSUMER_WORKGROUP);
        assert(split, `no size kept more than ${limit} x ${CONSUMER_WORKGROUP} items, so the split over y went untested`);
        if (!(split.args[0] <= limit && split.args[1] > 1 && split.args[1] <= limit)) failures.push(`${split.kept} kept items gave the arguments [${split.args}], not a split within the limit`);

        // The two paths, size by size.
        subgroups.forEach((result, index) => {
            const other = fallback[index];
            for (const [name, value] of Object.entries(result.digests)) {
                if (other.digests[name] !== value) failures.push(`${result.count} items: ${name} differs between subgroups and the fallback`);
            }
        });

        const data: Record<string, unknown> = {
            subgroupSizes: [device.adapterInfo.subgroupMinSize, device.adapterInfo.subgroupMaxSize],
            maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
            sizes: subgroups.map((result) => result.count),
            kept: subgroups.map((result) => result.kept),
            args: subgroups.map((result) => result.args),
            memoryBytes: { scan: rig.scan.memoryBytes, compaction: rig.compaction.memoryBytes, args: rig.args.memoryBytes },
            pathsIdentical: failures.every((failure) => !failure.includes('differs between')),
        };

        if (TIMING) {
            assert(device.features.has('timestamp-query'), 'timing needs timestamp-query');
            const timing: Record<string, unknown> = { subgroups: await timeRig(rig, input, log) };
            disposeRig(rig);
            // The fallback timed on the same device: same GPU, same clocks, only the path differs.
            const forced = await makeRig(device, 'workgroup', capacity);
            timing.workgroup = await timeRig(forced, input, log);
            disposeRig(forced);
            data.timing = timing;
            data.userAgent = navigator.userAgent;
            data.browserVersion = await fullBrowserVersion();
        } else {
            disposeRig(rig);
        }
        assert(failures.length === 0, failures.slice(0, 20).join('; '));
        return data;
    },
    { optionalFeatures: ['subgroups', 'timestamp-query'] },
);
