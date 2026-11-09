/**
 * RC1.4's functional checks: each feature the radiance cache needs is used
 * for real on a given device and its result read back, not only looked up in
 * `device.features`. The page runs them on a device our code makes and on
 * the device three's `WebGPURenderer` makes for itself.
 *
 * Every check runs inside error scopes, so a validation or out-of-memory
 * error is the check's own failure (with the message), not an uncaptured
 * error somewhere later.
 */

import {
    F16_EXPECTED,
    F16_INPUTS,
    halfToFloat,
    splitDispatch,
    subgroupInput,
    workgroupsRun,
    wrongSubgroupSums,
    type DispatchArguments,
} from './expected.ts';

/** The features RC1.4 checks, by the name used in the page's data. */
export const CHECKED_FEATURES = ['indirect-dispatch', 'texture-formats-tier2', 'subgroups', 'shader-f16', 'timestamp-query'] as const;
export type CheckedFeature = (typeof CHECKED_FEATURES)[number];

/**
 * The `GPUFeatureName` a check needs on the device. Indirect compute
 * dispatch is core WebGPU, so it needs none.
 */
export const FEATURE_NEEDED: Readonly<Record<CheckedFeature, GPUFeatureName | null>> = {
    'indirect-dispatch': null,
    'texture-formats-tier2': 'texture-formats-tier2',
    subgroups: 'subgroups',
    'shader-f16': 'shader-f16',
    'timestamp-query': 'timestamp-query',
};

export interface FeatureResult {
    /** The adapter lists the feature (a core feature always counts as offered). */
    offered: boolean;
    /** The device has it enabled, so it was requested and granted. */
    enabled: boolean;
    /** The functional check ran (it needs the feature enabled). */
    ran: boolean;
    pass: boolean;
    /** What the check read back; numbers and booleans become checks.csv rows. */
    measured: Record<string, number | boolean>;
    /** Why it failed, one sentence each. */
    problems: string[];
}

/** The limits RC1.4 reports: the radiance cache's hash cache wants about 1 GiB in about 30 packed buffers. */
export const REPORTED_LIMITS = [
    'maxStorageBufferBindingSize',
    'maxBufferSize',
    'maxStorageBuffersPerShaderStage',
    'maxStorageTexturesPerShaderStage',
    'maxComputeWorkgroupsPerDimension',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupStorageSize',
] as const satisfies readonly (keyof GPUSupportedLimits)[];

export function reportedLimits(limits: GPUSupportedLimits): Record<string, number> {
    return Object.fromEntries(REPORTED_LIMITS.map((name) => [name, limits[name]]));
}

/** Runs every check the device can run; `offered` is the adapter's feature list. */
export async function runFeatureChecks(device: GPUDevice, offered: ReadonlySet<string>): Promise<Record<CheckedFeature, FeatureResult>> {
    const results = {} as Record<CheckedFeature, FeatureResult>;
    const bodies: Record<CheckedFeature, (result: FeatureResult) => Promise<void>> = {
        'indirect-dispatch': (result) => checkIndirectDispatch(device, result),
        'texture-formats-tier2': (result) => checkReadWriteRgba16Float(device, result),
        subgroups: (result) => checkSubgroups(device, result),
        'shader-f16': (result) => checkShaderF16(device, result),
        'timestamp-query': (result) => checkTimestamps(device, result),
    };
    for (const feature of CHECKED_FEATURES) {
        const needed = FEATURE_NEEDED[feature];
        const result: FeatureResult = {
            offered: needed === null || offered.has(needed),
            enabled: needed === null || device.features.has(needed),
            ran: false,
            pass: false,
            measured: {},
            problems: [],
        };
        if (result.enabled) {
            result.ran = true;
            const errors = await withErrorScopes(device, () => bodies[feature](result));
            result.problems.push(...errors);
            result.pass = result.problems.length === 0;
        }
        results[feature] = result;
    }
    return results;
}

/**
 * Runs `body` inside validation, out-of-memory and internal error scopes and
 * returns every error they caught, plus anything `body` threw.
 */
export async function withErrorScopes(device: GPUDevice, body: () => Promise<void>): Promise<string[]> {
    const filters: GPUErrorFilter[] = ['validation', 'out-of-memory', 'internal'];
    for (const filter of filters) device.pushErrorScope(filter);
    const problems: string[] = [];
    try {
        await body();
    } catch (error) {
        problems.push(error instanceof Error ? error.message : String(error));
    }
    for (const filter of [...filters].reverse()) {
        const error = await device.popErrorScope();
        if (error) problems.push(`${filter} error: ${error.message}`);
    }
    return problems;
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const staging = device.createBuffer({ label: 'rc1.4 readback', size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'rc1.4 readback' });
    encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return copy;
}

/** Texture-to-buffer copies need rows padded to 256 bytes (NOTES, WebGPU). */
const COPY_ROW_ALIGNMENT = 256;

async function readTexture(device: GPUDevice, texture: GPUTexture, bytesPerTexel: number): Promise<{ data: ArrayBuffer; bytesPerRow: number }> {
    const bytesPerRow = Math.ceil((texture.width * bytesPerTexel) / COPY_ROW_ALIGNMENT) * COPY_ROW_ALIGNMENT;
    const size = bytesPerRow * texture.height;
    const staging = device.createBuffer({ label: 'rc1.4 texture readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'rc1.4 texture readback' });
    encoder.copyTextureToBuffer({ texture }, { buffer: staging, bytesPerRow }, [texture.width, texture.height]);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return { data, bytesPerRow };
}

function storageBuffer(device: GPUDevice, label: string, contents: ArrayBufferView<ArrayBuffer> | number, extraUsage = 0): GPUBuffer {
    const size = typeof contents === 'number' ? contents : contents.byteLength;
    const buffer = device.createBuffer({ label, size: Math.max(16, Math.ceil(size / 4) * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage });
    if (typeof contents !== 'number') device.queue.writeBuffer(buffer, 0, contents);
    return buffer;
}

function computePipeline(device: GPUDevice, label: string, code: string, entryPoint: string): GPUComputePipeline {
    const module = device.createShaderModule({ label, code });
    return device.createComputePipeline({ label, layout: 'auto', compute: { module, entryPoint } });
}

// --- Indirect compute dispatch --------------------------------------------------------------

const COUNT_WORKGROUP_SIZE = 64;

/**
 * One kernel writes indirect arguments from a GPU-side count (split over x
 * and y past the per-dimension limit, as RC1.3's writer will); a second,
 * dispatched indirectly from them, tallies how many workgroups and threads
 * actually ran and the largest workgroup id seen in x and y.
 */
const INDIRECT_SHADER = /* wgsl */ `
struct DispatchRequest {
    groups: u32,
    maxPerDimension: u32,
}

@group(0) @binding(0) var<storage, read> requests: array<DispatchRequest>;
@group(0) @binding(1) var<storage, read_write> dispatchArguments: array<u32>;

@compute @workgroup_size(64)
fn writeArguments(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&requests)) {
        return;
    }
    let request = requests[index];
    // y rounds up, so x * y can exceed the count; zero groups still writes y = 1.
    let x = min(request.groups, request.maxPerDimension);
    let y = max(1u, (request.groups + request.maxPerDimension - 1u) / request.maxPerDimension);
    dispatchArguments[index * 3u + 0u] = x;
    dispatchArguments[index * 3u + 1u] = y;
    dispatchArguments[index * 3u + 2u] = 1u;
}

struct Tally {
    workgroups: atomic<u32>,
    threads: atomic<u32>,
    // Largest workgroup id + 1, so 0 means "nothing ran".
    extentX: atomic<u32>,
    extentY: atomic<u32>,
}

@group(0) @binding(0) var<storage, read_write> tally: Tally;

var<workgroup> threadsInGroup: atomic<u32>;

@compute @workgroup_size(${COUNT_WORKGROUP_SIZE})
fn countWorkgroups(@builtin(workgroup_id) workgroupId: vec3u, @builtin(local_invocation_index) thread: u32) {
    // Count threads in workgroup memory first: one global atomic per workgroup, not per thread.
    atomicAdd(&threadsInGroup, 1u);
    workgroupBarrier();
    if (thread == 0u) {
        atomicAdd(&tally.workgroups, 1u);
        atomicAdd(&tally.threads, atomicLoad(&threadsInGroup));
        atomicMax(&tally.extentX, workgroupId.x + 1u);
        atomicMax(&tally.extentY, workgroupId.y + 1u);
    }
}`;

interface IndirectCase {
    name: string;
    groups: number;
    maxPerDimension: number;
}

async function checkIndirectDispatch(device: GPUDevice, result: FeatureResult): Promise<void> {
    const limit = device.limits.maxComputeWorkgroupsPerDimension;
    const cases: IndirectCase[] = [
        { name: 'none', groups: 0, maxPerDimension: limit },
        { name: 'one', groups: 1, maxPerDimension: limit },
        { name: 'x37', groups: 37, maxPerDimension: limit },
        // y > 1 with a small split: 5 groups at 2 per row are [2, 3, 1], 6 workgroups.
        { name: 'split_small', groups: 5, maxPerDimension: 2 },
        // Past the real limit: [limit, 3, 1].
        { name: 'split_at_limit', groups: limit * 2 + 5, maxPerDimension: limit },
        // x one over the limit: the spec makes the dispatch a no-op.
        { name: 'over_limit', groups: limit + 1, maxPerDimension: limit + 1 },
    ];

    const requests = new Uint32Array(cases.flatMap((entry) => [entry.groups, entry.maxPerDimension]));
    const requestBuffer = storageBuffer(device, 'rc1.4 dispatch requests', requests);
    const argumentBuffer = storageBuffer(device, 'rc1.4 dispatch arguments', cases.length * 3 * 4, GPUBufferUsage.INDIRECT);
    const tallies = cases.map((entry) => storageBuffer(device, `rc1.4 tally ${entry.name}`, 16));

    const writer = computePipeline(device, 'rc1.4 write dispatch arguments', INDIRECT_SHADER, 'writeArguments');
    const counter = computePipeline(device, 'rc1.4 count workgroups', INDIRECT_SHADER, 'countWorkgroups');
    const writerBindings = device.createBindGroup({
        layout: writer.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: requestBuffer } },
            { binding: 1, resource: { buffer: argumentBuffer } },
        ],
    });

    const encoder = device.createCommandEncoder({ label: 'rc1.4 indirect dispatch' });
    const pass = encoder.beginComputePass({ label: 'rc1.4 indirect dispatch' });
    pass.setPipeline(writer);
    pass.setBindGroup(0, writerBindings);
    pass.dispatchWorkgroups(1);
    // Each dispatch is its own usage scope, so the arguments written above can be read as INDIRECT here (NOTES).
    pass.setPipeline(counter);
    cases.forEach((_, index) => {
        pass.setBindGroup(0, device.createBindGroup({ layout: counter.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: tallies[index] } }] }));
        pass.dispatchWorkgroupsIndirect(argumentBuffer, index * 3 * 4);
    });
    pass.end();
    device.queue.submit([encoder.finish()]);

    const writtenArguments = new Uint32Array(await readBuffer(device, argumentBuffer, cases.length * 3 * 4));
    for (const [index, entry] of cases.entries()) {
        const expected = splitDispatch(entry.groups, entry.maxPerDimension);
        const written: DispatchArguments = { x: writtenArguments[index * 3], y: writtenArguments[index * 3 + 1], z: writtenArguments[index * 3 + 2] };
        const [workgroups, threads, extentX, extentY] = new Uint32Array(await readBuffer(device, tallies[index], 16));
        const expectedWorkgroups = workgroupsRun(expected, limit);
        const ran = expectedWorkgroups > 0;
        result.measured[`${entry.name}/x`] = written.x;
        result.measured[`${entry.name}/y`] = written.y;
        result.measured[`${entry.name}/workgroups`] = workgroups;
        result.measured[`${entry.name}/expected_workgroups`] = expectedWorkgroups;
        if (written.x !== expected.x || written.y !== expected.y || written.z !== expected.z) {
            result.problems.push(`${entry.name}: the GPU wrote [${written.x}, ${written.y}, ${written.z}], expected [${expected.x}, ${expected.y}, ${expected.z}]`);
        }
        if (workgroups !== expectedWorkgroups || threads !== expectedWorkgroups * COUNT_WORKGROUP_SIZE) {
            result.problems.push(`${entry.name}: ${workgroups} workgroups (${threads} threads) ran, expected ${expectedWorkgroups} (${expectedWorkgroups * COUNT_WORKGROUP_SIZE})`);
        }
        const expectedExtentX = ran ? expected.x : 0;
        const expectedExtentY = ran ? expected.y : 0;
        if (extentX !== expectedExtentX || extentY !== expectedExtentY) {
            result.problems.push(`${entry.name}: workgroup ids reached ${extentX} x ${extentY}, expected ${expectedExtentX} x ${expectedExtentY}`);
        }
    }
    for (const buffer of [requestBuffer, argumentBuffer, ...tallies]) buffer.destroy();
}

// --- texture-formats-tier2: read-write rgba16float ------------------------------------------

const ACCUMULATE_SIZE = 16;
const ACCUMULATE_STEPS = 4;
/** Added to every texel each step: exact in f16, so the sum is exact too. */
const ACCUMULATE_STEP = [1, 0.5, 0.25, 2] as const;

/** A texel's starting value, from its position; small multiples of 1/4, exact in f16. */
function accumulateStart(x: number, y: number): [number, number, number, number] {
    return [x * 0.25, y * 0.5, 1, -2];
}

/**
 * Each thread reads its own texel of a read-write `rgba16float` storage
 * texture and writes it back plus a step: the access pattern the radiance
 * cache's probe blending uses (D5). Run several times, the readback must be
 * the start plus that many steps.
 */
const ACCUMULATE_SHADER = /* wgsl */ `
@group(0) @binding(0) var accumulator: texture_storage_2d<rgba16float, read_write>;

const STEP = vec4f(${ACCUMULATE_STEP.join(', ')});

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= textureDimensions(accumulator))) {
        return;
    }
    let previous = textureLoad(accumulator, id.xy);
    textureStore(accumulator, id.xy, previous + STEP);
}`;

/** f32 to binary16 bits, for values known to be exact in f16 (the page's start values). */
function exactFloatToHalf(value: number): number {
    for (let bits = 0; bits < 0x10000; bits++) {
        if (halfToFloat(bits) === value && !(value === 0 && bits !== 0)) return bits;
    }
    throw new Error(`${value} is not exact in f16`);
}

async function checkReadWriteRgba16Float(device: GPUDevice, result: FeatureResult): Promise<void> {
    const texture = device.createTexture({
        label: 'rc1.4 read-write rgba16float',
        size: [ACCUMULATE_SIZE, ACCUMULATE_SIZE],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
    const halfBits = new Map<number, number>();
    const toHalf = (value: number): number => {
        if (!halfBits.has(value)) halfBits.set(value, exactFloatToHalf(value));
        return halfBits.get(value)!;
    };
    const start = new Uint16Array(ACCUMULATE_SIZE * ACCUMULATE_SIZE * 4);
    for (let y = 0; y < ACCUMULATE_SIZE; y++) {
        for (let x = 0; x < ACCUMULATE_SIZE; x++) {
            accumulateStart(x, y).forEach((value, channel) => (start[(y * ACCUMULATE_SIZE + x) * 4 + channel] = toHalf(value)));
        }
    }
    device.queue.writeTexture({ texture }, start, { bytesPerRow: ACCUMULATE_SIZE * 8 }, [ACCUMULATE_SIZE, ACCUMULATE_SIZE]);

    const pipeline = computePipeline(device, 'rc1.4 accumulate rgba16float', ACCUMULATE_SHADER, 'accumulate');
    const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: texture.createView() }] });
    const encoder = device.createCommandEncoder({ label: 'rc1.4 accumulate' });
    const pass = encoder.beginComputePass({ label: 'rc1.4 accumulate' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    for (let step = 0; step < ACCUMULATE_STEPS; step++) pass.dispatchWorkgroups(ACCUMULATE_SIZE / 8, ACCUMULATE_SIZE / 8);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const { data, bytesPerRow } = await readTexture(device, texture, 8);
    let wrong = 0;
    let firstWrong = '';
    for (let y = 0; y < ACCUMULATE_SIZE; y++) {
        const row = new Uint16Array(data, y * bytesPerRow, ACCUMULATE_SIZE * 4);
        for (let x = 0; x < ACCUMULATE_SIZE; x++) {
            const expected = accumulateStart(x, y).map((value, channel) => value + ACCUMULATE_STEPS * ACCUMULATE_STEP[channel]);
            const got = [0, 1, 2, 3].map((channel) => halfToFloat(row[x * 4 + channel]));
            if (got.some((value, channel) => value !== expected[channel])) {
                if (wrong === 0) firstWrong = `texel (${x}, ${y}) read (${got.join(', ')}), expected (${expected.join(', ')})`;
                wrong++;
            }
        }
    }
    result.measured.texels = ACCUMULATE_SIZE * ACCUMULATE_SIZE;
    result.measured.steps = ACCUMULATE_STEPS;
    result.measured.wrong_texels = wrong;
    if (wrong > 0) result.problems.push(`${wrong} of ${ACCUMULATE_SIZE * ACCUMULATE_SIZE} texels wrong after ${ACCUMULATE_STEPS} read-write steps; first: ${firstWrong}`);
    texture.destroy();
}

// --- subgroups ------------------------------------------------------------------------------

const SUBGROUP_WORKGROUP_SIZE = 128;
const SUBGROUP_WORKGROUPS = 64;

/**
 * Each thread adds its value with `subgroupAdd` and records the sum, its
 * subgroup's first active thread (`subgroupBroadcastFirst` of the global
 * index: the subgroup's name), its lane and the subgroup size. The CPU sums
 * each subgroup by that name, so no lane order is assumed.
 */
const SUBGROUP_SHADER = /* wgsl */ `
enable subgroups;

struct ThreadRecord {
    sum: u32,
    leader: u32,
    lane: u32,
    size: u32,
}

@group(0) @binding(0) var<storage, read> values: array<u32>;
@group(0) @binding(1) var<storage, read_write> records: array<ThreadRecord>;

@compute @workgroup_size(${SUBGROUP_WORKGROUP_SIZE})
fn sum(
    @builtin(global_invocation_id) id: vec3u,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) size: u32,
) {
    let index = id.x;
    records[index] = ThreadRecord(subgroupAdd(values[index]), subgroupBroadcastFirst(index), lane, size);
}`;

async function checkSubgroups(device: GPUDevice, result: FeatureResult): Promise<void> {
    const count = SUBGROUP_WORKGROUP_SIZE * SUBGROUP_WORKGROUPS;
    const inputs = Uint32Array.from({ length: count }, (_, index) => subgroupInput(index));
    const valueBuffer = storageBuffer(device, 'rc1.4 subgroup values', inputs);
    const recordBuffer = storageBuffer(device, 'rc1.4 subgroup records', count * 16);
    const pipeline = computePipeline(device, 'rc1.4 subgroupAdd', SUBGROUP_SHADER, 'sum');
    const bindings = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: valueBuffer } },
            { binding: 1, resource: { buffer: recordBuffer } },
        ],
    });
    const encoder = device.createCommandEncoder({ label: 'rc1.4 subgroups' });
    const pass = encoder.beginComputePass({ label: 'rc1.4 subgroups' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(SUBGROUP_WORKGROUPS);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const records = new Uint32Array(await readBuffer(device, recordBuffer, count * 16));
    const field = (offset: number): Uint32Array => Uint32Array.from({ length: count }, (_, index) => records[index * 4 + offset]);
    const sums = field(0);
    const leaders = field(1);
    const lanes = field(2);
    const sizes = field(3);

    const wrong = wrongSubgroupSums(inputs, sums, leaders);
    const subgroupSizes = new Set(sizes);
    const observedMin = Math.min(...subgroupSizes);
    const observedMax = Math.max(...subgroupSizes);
    // Threads in a subgroup and the size the builtin reports must agree (every lane is active here).
    const members = new Map<number, number>();
    for (const leader of leaders) members.set(leader, (members.get(leader) ?? 0) + 1);
    const partial = [...leaders].filter((leader, index) => members.get(leader) !== sizes[index]).length;
    // Information only (NOTES: WebGPU doesn't promise it): do lanes hold consecutive threads?
    const consecutive = [...lanes].every((lane, index) => leaders[index] + lane === index);
    const info = device.adapterInfo as GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number };

    result.measured.threads = count;
    result.measured.wrong_sums = wrong;
    result.measured.subgroups = members.size;
    result.measured.observed_size_min = observedMin;
    result.measured.observed_size_max = observedMax;
    result.measured.reported_size_min = info.subgroupMinSize ?? -1;
    result.measured.reported_size_max = info.subgroupMaxSize ?? -1;
    result.measured.lanes_consecutive = consecutive;
    if (wrong > 0) result.problems.push(`${wrong} of ${count} threads read a wrong subgroupAdd sum`);
    if (partial > 0) result.problems.push(`${partial} threads' subgroup_size differs from their subgroup's member count`);
    if (info.subgroupMinSize !== undefined && info.subgroupMaxSize !== undefined && (observedMin < info.subgroupMinSize || observedMax > info.subgroupMaxSize)) {
        result.problems.push(`subgroup sizes ${observedMin}-${observedMax} fall outside the reported ${info.subgroupMinSize}-${info.subgroupMaxSize}`);
    }
    for (const buffer of [valueBuffer, recordBuffer]) buffer.destroy();
}

// --- shader-f16 ------------------------------------------------------------------------------

/**
 * Computes F16_EXPECTED's cases in f16 from f32 inputs read at run time (so
 * nothing folds at compile time) and stores them to an `array<f16>` in a
 * storage buffer, which `shader-f16` also allows. The last two cases fail
 * if the compiler does the math in f32 and only stores f16.
 */
const F16_SHADER = /* wgsl */ `
enable f16;

struct Inputs {
    one: f32,
    tiny: f32,
    multiplicandA: f32,
    multiplicandB: f32,
    squareOf: f32,
    largestFinite: f32,
    dividend: f32,
    divisor: f32,
    threeHundred: f32,
    thousand: f32,
}

@group(0) @binding(0) var<storage, read> inputs: Inputs;
@group(0) @binding(1) var<storage, read_write> results: array<f16>;

@compute @workgroup_size(1)
fn computeHalves() {
    results[0] = f16(inputs.multiplicandA) * f16(inputs.multiplicandB);
    results[1] = f16(inputs.one) + f16(inputs.tiny);
    results[2] = sqrt(f16(inputs.squareOf));
    results[3] = f16(inputs.largestFinite);
    let oneHalf = f16(inputs.one) / 2.0h;
    results[4] = dot(vec4h(oneHalf, 1.5h, -2.0h, 4.0h), vec4h(2.0h, 2.0h, 0.25h, 0.125h) * f16(inputs.one));
    results[5] = f16(inputs.dividend) / f16(inputs.divisor);
    // The two cases whose f16 answer differs from f32 math stored as f16 (F16_INPUTS).
    let onePlusTiny = f16(inputs.one) + f16(inputs.tiny);
    results[6] = onePlusTiny - f16(inputs.one);
    let squared = f16(inputs.threeHundred) * f16(inputs.threeHundred);
    results[7] = squared / f16(inputs.thousand);
}`;

async function checkShaderF16(device: GPUDevice, result: FeatureResult): Promise<void> {
    const inputBuffer = storageBuffer(device, 'rc1.4 f16 inputs', new Float32Array(Object.values(F16_INPUTS)));
    const resultBytes = Math.ceil((F16_EXPECTED.length * 2) / 4) * 4;
    const resultBuffer = storageBuffer(device, 'rc1.4 f16 results', resultBytes);
    const pipeline = computePipeline(device, 'rc1.4 f16', F16_SHADER, 'computeHalves');
    const bindings = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: resultBuffer } },
        ],
    });
    const encoder = device.createCommandEncoder({ label: 'rc1.4 f16' });
    const pass = encoder.beginComputePass({ label: 'rc1.4 f16' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindings);
    pass.dispatchWorkgroups(1);
    pass.end();
    device.queue.submit([encoder.finish()]);

    const halves = new Uint16Array(await readBuffer(device, resultBuffer, resultBytes));
    F16_EXPECTED.forEach(({ name, value }, index) => {
        const got = halfToFloat(halves[index]);
        // Infinity isn't a CSV number, so each case also records whether it matched.
        if (Number.isFinite(got)) result.measured[name] = got;
        else result.measured[`${name}_infinite`] = got === Infinity ? 1 : got === -Infinity ? -1 : 0;
        result.measured[`${name}_matches`] = got === value;
        if (got !== value) result.problems.push(`${name}: read ${got}, expected ${value}`);
    });
    for (const buffer of [inputBuffer, resultBuffer]) buffer.destroy();
}

// --- timestamp-query -------------------------------------------------------------------------

const BUSY_WORKGROUPS = 4096;
const BUSY_ITERATIONS = 4000;

/** Enough arithmetic that a pass takes well over Chrome's timestamp quantum (100 µs when it rounds). */
const BUSY_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> sink: array<f32>;

@compute @workgroup_size(64)
fn busy(@builtin(global_invocation_id) id: vec3u) {
    var value = f32(id.x) * 1e-6;
    for (var iteration = 0u; iteration < ${BUSY_ITERATIONS}u; iteration++) {
        value = fract(value * 1.0001 + 0.37);
    }
    sink[id.x] = value;
}`;

async function checkTimestamps(device: GPUDevice, result: FeatureResult): Promise<void> {
    const querySet = device.createQuerySet({ label: 'rc1.4 timestamps', type: 'timestamp', count: 4 });
    const resolveBuffer = device.createBuffer({ label: 'rc1.4 timestamps', size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const sink = storageBuffer(device, 'rc1.4 busy sink', BUSY_WORKGROUPS * 64 * 4);
    const pipeline = computePipeline(device, 'rc1.4 busy', BUSY_SHADER, 'busy');
    const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: sink } }] });

    const encoder = device.createCommandEncoder({ label: 'rc1.4 timestamps' });
    for (const pair of [0, 1]) {
        const pass = encoder.beginComputePass({
            label: `rc1.4 timed pass ${pair}`,
            timestampWrites: { querySet, beginningOfPassWriteIndex: pair * 2, endOfPassWriteIndex: pair * 2 + 1 },
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindings);
        pass.dispatchWorkgroups(BUSY_WORKGROUPS);
        pass.end();
    }
    encoder.resolveQuerySet(querySet, 0, 4, resolveBuffer, 0);
    device.queue.submit([encoder.finish()]);

    const stamps = Array.from(new BigUint64Array(await readBuffer(device, resolveBuffer, 32)));
    const [firstStart, firstEnd, secondStart, secondEnd] = stamps;
    const nanoseconds = (from: bigint, to: bigint): number => Number(to - from);
    result.measured.zero_stamps = stamps.filter((stamp) => stamp === 0n).length;
    result.measured.pass1_ms = nanoseconds(firstStart, firstEnd) / 1e6;
    result.measured.pass2_ms = nanoseconds(secondStart, secondEnd) / 1e6;
    result.measured.gap_ms = nanoseconds(firstEnd, secondStart) / 1e6;
    if (stamps.some((stamp) => stamp === 0n)) result.problems.push(`a timestamp read 0: ${stamps.join(', ')}`);
    // Each pass did real work, so its end must be after its start; the second pass can't start before the first ends.
    if (!(firstStart < firstEnd && firstEnd <= secondStart && secondStart < secondEnd)) {
        result.problems.push(`timestamps out of order: ${stamps.join(', ')}`);
    }
    querySet.destroy();
    for (const buffer of [resolveBuffer, sink]) buffer.destroy();
}

// --- Without texture-formats-tier2 ----------------------------------------------------------

export interface WithoutTier2Result {
    /** The device has `texture-formats-tier1` (and not tier2). */
    tier1Enabled: boolean;
    /** tier1: a write-only `rg16float` storage texture (DDGI's distance atlas) stores and reads back. */
    rg16floatWriteOnlyPass: boolean | null;
    /** A read-write `rgba16float` storage binding is refused with a validation error, captured. */
    rgba16floatReadWriteRefused: boolean;
    refusalMessage: string;
    /** Control: read-write `r32float` (core WebGPU) is accepted. */
    r32floatReadWriteAccepted: boolean;
    /** D5's declared fallback works: read `rgba16float` sampled, write it to a write-only storage texture. */
    pingPongFallbackPass: boolean;
    problems: string[];
}

const RG16FLOAT_SHADER = /* wgsl */ `
@group(0) @binding(0) var destination: texture_storage_2d<rg16float, write>;

@compute @workgroup_size(8, 8)
fn store(@builtin(global_invocation_id) id: vec3u) {
    textureStore(destination, id.xy, vec4f(f32(id.x) * 0.5, -f32(id.y), 0.0, 0.0));
}`;

/** D5's fallback: last frame's value from a sampled texture, this frame's into a write-only storage texture. */
const PING_PONG_SHADER = /* wgsl */ `
@group(0) @binding(0) var previous: texture_2d<f32>;
@group(0) @binding(1) var next: texture_storage_2d<rgba16float, write>;

const STEP = vec4f(${ACCUMULATE_STEP.join(', ')});

@compute @workgroup_size(8, 8)
fn advance(@builtin(global_invocation_id) id: vec3u) {
    textureStore(next, id.xy, textureLoad(previous, id.xy, 0) + STEP);
}`;

/**
 * On a device without tier2: what tier1 gives, that read-write
 * `rgba16float` fails cleanly (a captured validation error from the bind
 * group layout, no crash and no shader compile), and that D5's ping-pong
 * fallback works.
 */
export async function checkWithoutTier2(device: GPUDevice): Promise<WithoutTier2Result> {
    const problems: string[] = [];
    const tier1Enabled = device.features.has('texture-formats-tier1');
    const layoutError = async (format: GPUTextureFormat): Promise<GPUError | null> => {
        device.pushErrorScope('validation');
        device.createBindGroupLayout({
            label: `rc1.4 read-write ${format}`,
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'read-write', format } }],
        });
        return device.popErrorScope();
    };
    const refusal = await layoutError('rgba16float');
    const control = await layoutError('r32float');

    let rg16floatWriteOnlyPass: boolean | null = null;
    if (tier1Enabled) {
        const errors = await withErrorScopes(device, async () => {
            const size = 8;
            const texture = device.createTexture({ label: 'rc1.4 rg16float', size: [size, size], format: 'rg16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
            const pipeline = computePipeline(device, 'rc1.4 rg16float store', RG16FLOAT_SHADER, 'store');
            const encoder = device.createCommandEncoder({ label: 'rc1.4 rg16float' });
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: texture.createView() }] }));
            pass.dispatchWorkgroups(1);
            pass.end();
            device.queue.submit([encoder.finish()]);
            const { data, bytesPerRow } = await readTexture(device, texture, 4);
            for (let y = 0; y < size; y++) {
                const row = new Uint16Array(data, y * bytesPerRow, size * 2);
                for (let x = 0; x < size; x++) {
                    if (halfToFloat(row[x * 2]) !== x * 0.5 || halfToFloat(row[x * 2 + 1]) !== -y) throw new Error(`rg16float texel (${x}, ${y}) read wrong`);
                }
            }
            texture.destroy();
        });
        rg16floatWriteOnlyPass = errors.length === 0;
        problems.push(...errors.map((error) => `rg16float write-only: ${error}`));
    }

    const pingPongErrors = await withErrorScopes(device, async () => {
        const size = 8;
        const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST;
        const textures = [0, 1].map((index) => device.createTexture({ label: `rc1.4 ping-pong ${index}`, size: [size, size], format: 'rgba16float', usage }));
        const start = new Uint16Array(size * size * 4).fill(exactFloatToHalf(1));
        device.queue.writeTexture({ texture: textures[0] }, start, { bytesPerRow: size * 8 }, [size, size]);
        const pipeline = computePipeline(device, 'rc1.4 ping-pong', PING_PONG_SHADER, 'advance');
        const encoder = device.createCommandEncoder({ label: 'rc1.4 ping-pong' });
        for (let step = 0; step < ACCUMULATE_STEPS; step++) {
            const [from, to] = step % 2 === 0 ? textures : [textures[1], textures[0]];
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(
                0,
                device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: from.createView() },
                        { binding: 1, resource: to.createView() },
                    ],
                }),
            );
            pass.dispatchWorkgroups(1);
            pass.end();
        }
        device.queue.submit([encoder.finish()]);
        // An even number of steps ends in the first texture.
        const { data, bytesPerRow } = await readTexture(device, textures[ACCUMULATE_STEPS % 2], 8);
        const expected = ACCUMULATE_STEP.map((step) => 1 + ACCUMULATE_STEPS * step);
        for (let y = 0; y < size; y++) {
            const row = new Uint16Array(data, y * bytesPerRow, size * 4);
            for (let x = 0; x < size; x++) {
                const got = [0, 1, 2, 3].map((channel) => halfToFloat(row[x * 4 + channel]));
                if (got.some((value, channel) => value !== expected[channel])) throw new Error(`ping-pong texel (${x}, ${y}) read (${got.join(', ')}), expected (${expected.join(', ')})`);
            }
        }
        for (const texture of textures) texture.destroy();
    });
    problems.push(...pingPongErrors.map((error) => `ping-pong fallback: ${error}`));

    if (refusal === null) problems.push('a read-write rgba16float storage binding was accepted without texture-formats-tier2');
    if (control !== null) problems.push(`read-write r32float (core) was refused: ${control.message}`);
    return {
        tier1Enabled,
        rg16floatWriteOnlyPass,
        rgba16floatReadWriteRefused: refusal !== null,
        refusalMessage: refusal?.message ?? '',
        r32floatReadWriteAccepted: control === null,
        pingPongFallbackPass: pingPongErrors.length === 0,
        problems,
    };
}

/** Whether a device accepts a read-write `rgba16float` storage binding (it should with tier2). */
export async function acceptsReadWriteRgba16Float(device: GPUDevice): Promise<boolean> {
    device.pushErrorScope('validation');
    device.createBindGroupLayout({
        label: 'rc1.4 read-write rgba16float',
        entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'read-write', format: 'rgba16float' } }],
    });
    return (await device.popErrorScope()) === null;
}
