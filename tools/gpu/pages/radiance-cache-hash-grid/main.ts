/**
 * RC3.1: the radiance cache's hash grid (packages/core/src/techniques/
 * radiance-cache/world-cache) on the GPU, at Capsaicin's default capacity
 * (2^14 buckets × 16 tiles × 85 cells, about 1 GiB).
 *
 * 1. **Limits:** a device with WebGPU's default limits (a second adapter,
 *    made and destroyed first) gets `GILimitError` naming
 *    `maxStorageBufferBindingSize`; the main device asks for the adapter's
 *    limits (`requiredLimits`) and holds the cache. Capacity and memory are
 *    reported.
 * 2. **Hashes and descriptor:** `pcgHash`, `xxHash` and `HashGridCache_GetDesc`
 *    on the GPU give the CPU port's bits, on fixed hits and on seeded random
 *    hits at Capsaicin's cell size (60°, 1280 × 720, 32 pixels). Random hits
 *    within a rounding step of a floor in the CPU's f64 arithmetic are left
 *    out (WGSL lets `log2`, `/` and `distance` be a few ULP off) and counted.
 * 3. **Insert then find:** 256 random hits inserted and found again, each in
 *    its own bucket's slots, with the CPU's cell index, the packed tile list
 *    and decay stamps right.
 * 1b. **Probe-ray layout:** a ray written at byte 16 + 48 × index reads back
 *    field by field through `WorldCacheProbeRays` (the world cache's struct).
 * 3b. **Near-boundary hits** (the ones section 2 leaves out): insert and find
 *    land on the same cell for every one; how many the GPU keys differently
 *    from the CPU is reported as information. Then `reset()` empties the
 *    cache: every find misses, the packed list is empty, an insert is new.
 * 4. **Collision:** two tiles that share a bucket (found by the CPU) both
 *    inserted and both found, in slots 0 and 1.
 * 5. **Full bucket:** 16 tiles fill a bucket; the 17th is refused, counted
 *    once in the overflow counter, and not found.
 * 6. **Many threads, one tile:** 65,536 threads insert the same tile; one
 *    claims a slot, all get it, the bucket holds it once.
 * 7. **Purge:** a tile untouched for 50 frames is freed by `PurgeTiles` at
 *    exactly the 50th, one touched at frame 30 lives to its own 50th, and
 *    the same across the frame counter's wrap (from 2^32 − 20).
 * 8. **Find stops at the first empty slot** (Capsaicin's behaviour, kept):
 *    with slot 0's tile purged, the tile in slot 1 isn't found, and
 *    inserting it again claims slot 0.
 * 9. **Halves as f32tof16 makes them** (register D31): `HashGridCache_PackHalf2`
 *    gives the nearest half on random values and exact ties (to the even
 *    one, both parities and signs, normal and subnormal), and clamps past
 *    65,504; the CPU's `floatToHalfBits` is the reference.
 *
 * Correctness only, nothing timed (DECISIONS 2026-11-09).
 * `pnpm test:gpu --filter radiance-cache-hash-grid --gpu nvidia` (or intel).
 */

import { GILimitError } from '../../../../packages/core/src/platform/contract/errors.ts';
import type { FrameCamera } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { Vec3 } from '../../../../packages/core/src/platform/contract/math.ts';
import { optionalFeatures, requiredLimits } from '../../../../packages/core/src/platform/device-requirements.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import type { GpuTimerScope } from '../../../../packages/core/src/platform/contract/timing.ts';
import { cellResetFrameWords, DEFAULT_HASH_GRID_CACHE_OPTIONS, hashGridCacheCapacity, type HashGridCacheOptions } from '../../../../packages/core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { HashGridCache } from '../../../../packages/core/src/techniques/radiance-cache/world-cache/hash-grid-cache.ts';
import {
    hashGridCacheCellIndex,
    hashGridCacheCellSizeConstant,
    hashGridCacheGetDesc,
    INVALID_ID,
    pcgHash,
    TILE_DECAY,
    verticalFieldOfView,
    xxHash,
    type HashGridCacheDesc,
    type HashGridCacheDescConstants,
} from '../../../../packages/core/src/techniques/radiance-cache/world-cache/hash-grid-cpu.ts';
import {
    WORLD_CACHE_PROBE_RAY_BYPASS,
    WORLD_CACHE_PROBE_RAY_BYTES,
    WORLD_CACHE_PROBE_RAYS_HEADER_BYTES,
    type WorldCacheContext,
    type WorldCacheFrame,
    type WorldCacheProbeRays,
} from '../../../../packages/core/src/techniques/radiance-cache/world-cache/world-cache.ts';
import { floatToHalfBits } from '../../../../packages/core/src/techniques/radiance-cache/capsaicin-math.ts';
import { assert, drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import { releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { COUNTER, createProbeRayReader, createTestIo, createTestKernels, HIT_WORDS, PROBE_RAY_READ_WORDS, RESULT_WORDS, TEST_WORKGROUP_SIZE, type TestIo, type TestKernel, type TestKernelName } from './test-kernels.ts';

const f32 = Math.fround;

/** The cache's passes untimed: this page times nothing. */
const NO_TIMER: GpuTimerScope = { computePass: () => undefined };

const RENDER_WIDTH = 1280;
const RENDER_HEIGHT = 720;
const FOV_Y = Math.PI / 3;
const EYE: Vec3 = [f32(1.25), f32(1.75), f32(-2.5)];
/** Random hits for the descriptor check, before the boundary filter. */
const DESCRIPTOR_CANDIDATES = 8192;
/** Hits inserted and found in the insert-then-find check. */
const INSERT_HITS = 256;
/** Threads inserting one tile at once. */
const CONCURRENT_THREADS = 65536;
const HASH_INPUTS = 4096;
const MAX_INPUT_WORDS = Math.max(DESCRIPTOR_CANDIDATES * HIT_WORDS, HASH_INPUTS);
const MAX_RESULT_WORDS = CONCURRENT_THREADS * RESULT_WORDS.insert;

/** A seeded LCG in [0, 1), so every run and both GPUs see the same hits. */
function seeded(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

interface TestHit {
    readonly position: Vec3;
    readonly direction: Vec3;
    readonly hitDistance: number;
}

function toF32Hit(position: readonly number[], direction: readonly number[], hitDistance: number): TestHit {
    const length = Math.hypot(direction[0], direction[1], direction[2]);
    return {
        position: [f32(position[0]), f32(position[1]), f32(position[2])],
        direction: [f32(direction[0] / length), f32(direction[1] / length), f32(direction[2] / length)],
        hitDistance: f32(hitDistance),
    };
}

/** Distance from `value` to the nearest whole number. */
function toWhole(value: number): number {
    return Math.abs(value - Math.round(value));
}

/**
 * Whether every floor and comparison in `HashGridCache_GetDesc` is clear of
 * its boundary by more than a rounding step, worked out in f64: such a hit
 * gives the same bits whichever way the GPU rounds.
 */
function clearOfBoundaries(hit: TestHit, constants: HashGridCacheDescConstants): boolean {
    const offset = [hit.position[0] - EYE[0], hit.position[1] - EYE[1], hit.position[2] - EYE[2]];
    const scaled = Math.hypot(offset[0], offset[1], offset[2]) * constants.cellSize;
    if (Math.abs(scaled - constants.minCellSize) < 1e-5) return false;
    const step = Math.max(scaled, constants.minCellSize);
    const levelValue = Math.log2(1000 * step);
    if (toWhole(levelValue) < 1e-4) return false;
    const cell = f32(f32(1e-3) * 2 ** Math.floor(levelValue));
    const tile = cell * constants.tileCellRatio;
    for (const value of hit.position) {
        if (toWhole(value / tile) < 2e-3 || toWhole(value / cell) < 2e-3) return false;
    }
    for (const value of hit.direction) {
        if (toWhole(0.5 + (0.5 * value + 0.5) * 4) < 1e-4) return false;
    }
    if (Math.abs(hit.hitDistance - tile) < 1e-4 * tile) return false;
    const sorted = hit.direction.map(Math.abs).sort((a, b) => b - a);
    return sorted[0] - sorted[1] > 1e-5;
}

function encodeHits(hits: readonly TestHit[]): Uint32Array<ArrayBuffer> {
    const bytes = new ArrayBuffer(hits.length * HIT_WORDS * 4);
    const floats = new Float32Array(bytes);
    hits.forEach((hit, index) => {
        floats.set([...hit.position, hit.hitDistance, ...hit.direction, 0], index * HIT_WORDS);
    });
    return new Uint32Array(bytes);
}

function perspective(fovY: number, aspect: number, near: number, far: number): number[] {
    const focal = 1 / Math.tan(fovY / 2);
    const projection = new Array<number>(16).fill(0);
    projection[0] = focal / aspect;
    projection[5] = focal;
    projection[10] = far / (near - far);
    projection[11] = -1;
    projection[14] = (far * near) / (near - far);
    return projection;
}

const CAMERA: FrameCamera = {
    position: EYE,
    view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -EYE[0], -EYE[1], -EYE[2], 1],
    projection: perspective(FOV_Y, RENDER_WIDTH / RENDER_HEIGHT, 0.1, 1000),
    near: 0.1,
    far: 1000,
};

async function readWords(device: GPUDevice, source: GPUBuffer, offsetBytes: number, words: number): Promise<Uint32Array> {
    const readback = device.createBuffer({ label: 'hash-grid readback', size: Math.max(4, words * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'hash-grid readback' });
    if (words > 0) encoder.copyBufferToBuffer(source, offsetBytes, readback, 0, words * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = new Uint32Array(readback.getMappedRange().slice(0, words * 4));
    readback.unmap();
    readback.destroy();
    return copy;
}

/** Finds `count` hits whose tiles land in one bucket, all different tiles, at the minimum cell size (a cell-size option of 0). */
function hitsSharingABucket(constants: HashGridCacheDescConstants, count: number): { hits: TestHit[]; descs: HashGridCacheDesc[] } {
    const tile = f32(f32(f32(1e-3) * 64) * 8);
    const byBucket = new Map<number, { hits: TestHit[]; descs: HashGridCacheDesc[] }>();
    for (let x = -64; x < 64; x++) {
        for (let y = -64; y < 64; y++) {
            for (let z = 0; z < 32; z++) {
                // Off the tile's centre lines, so every floor is far from a boundary.
                const hit = toF32Hit([(x + 0.53) * tile, (y + 0.41) * tile, (z + 0.67) * tile], [0.1, -0.2, -1], 10);
                const desc = hashGridCacheGetDesc({ eyePosition: EYE, hitPosition: hit.position, direction: hit.direction, hitDistance: hit.hitDistance }, constants);
                const group = byBucket.get(desc.bucketIndex) ?? { hits: [], descs: [] };
                if (group.descs.some((other) => other.tileHash === desc.tileHash)) continue;
                group.hits.push(hit);
                group.descs.push(desc);
                byBucket.set(desc.bucketIndex, group);
                if (group.hits.length === count) return group;
            }
        }
    }
    throw new Error(`no bucket reached ${count} tiles in the search`);
}

const startTime = performance.now();
const logLines: string[] = [];
let deviceErrors: string[] = [];
let adapterSummary = null;
const data: Record<string, unknown> = {};
const log = (line: string): void => {
    logLines.push(line);
};

try {
    assert(navigator.gpu, 'WebGPU is not available in this browser.');

    // ---- 1. A device with WebGPU's default limits can't hold the cache (register D14) ----
    {
        const defaultAdapter = await navigator.gpu.requestAdapter();
        assert(defaultAdapter, 'requestAdapter() returned null');
        const defaultDevice = await defaultAdapter.requestDevice();
        const defaultErrors = watchDevice(defaultDevice);
        let refusal: unknown = null;
        try {
            const defaultPlatform = new GIPlatformImpl(defaultDevice, { label: 'hash-grid default limits' });
            const context: WorldCacheContext = { device: defaultDevice, kernels: defaultPlatform.kernels, compute: defaultPlatform.compute, shade: defaultPlatform.shade, timer: NO_TIMER };
            await new HashGridCache().prepare(context);
        } catch (error) {
            refusal = error;
        }
        await drainDevice(defaultDevice);
        // Checked before destroy(), which resolves device.lost (watchDevice records that too).
        assert(defaultErrors.length === 0, `the refused prepare left device errors: ${defaultErrors.join(' | ')}`);
        defaultDevice.destroy();
        assert(refusal instanceof GILimitError, `a default-limits device should get GILimitError, got ${String(refusal)}`);
        assert(refusal.limit === 'maxStorageBufferBindingSize', `the error should name maxStorageBufferBindingSize, named ${refusal.limit}`);
        data.defaultLimitsRefusal = refusal.message;
        log(`default limits: refused with ${refusal.limit} (needs ${refusal.needed}, has ${refusal.available})`);
    }

    // ---- The main device, with the adapter's limits ----
    const adapter = await navigator.gpu.requestAdapter();
    assert(adapter, 'requestAdapter() returned null');
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter), requiredFeatures: optionalFeatures(adapter) });
    deviceErrors = watchDevice(device);
    releaseDeviceOnPageHide(device);
    adapterSummary = summarizeAdapterInfo(device.adapterInfo, device.features);
    data.limits = { maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize, maxBufferSize: device.limits.maxBufferSize };

    const options: HashGridCacheOptions = { ...DEFAULT_HASH_GRID_CACHE_OPTIONS };
    // An empty platform: the cache's population kernels light hits through its hit shading
    // (none here: this page's probe rays are empty, so they run on no rays).
    const platform = new GIPlatformImpl(device, { label: 'hash-grid' });
    await platform.prepare();
    {
        const encoder = device.createCommandEncoder({ label: 'hash-grid platform frame' });
        platform.encode(encoder, { deltaTime: 1 / 60, budget: { units: 0 }, camera: CAMERA });
        device.queue.submit([encoder.finish()]);
    }
    const kernels = platform.kernels;
    const cache = new HashGridCache(options);
    await cache.prepare({ device, kernels, compute: platform.compute, shade: platform.shade, timer: NO_TIMER });

    const capacity = cache.capacity;
    const expected = hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS);
    assert(capacity.numBuckets === 16384 && capacity.numTilesPerBucket === 16 && capacity.numCellsPerTile === 85, 'not Capsaicin\'s default capacity');
    assert(capacity.numCells === expected.numCells && capacity.numCells === 22282240, `cells ${capacity.numCells}`);
    const layout = cache.layout;
    assert(cache.memoryBytes >= layout.totalBytes && cache.memoryBytes < layout.totalBytes + 4096, `memoryBytes ${cache.memoryBytes} against the buffers' ${layout.totalBytes}`);
    data.capacity = cache.stats;
    data.memory = {
        words: layout.words.bytes,
        tileWords: layout.tileWords.bytes,
        // E9's frames: in the per-ray lists while E8 or E9 is used (made with the first frame's lists).
        resetFramesWhenUsed: cellResetFrameWords(capacity) * 4,
        updateScratch: layout.updateScratch.bytes,
        cellValues: layout.cellValues.bytes,
        total: cache.memoryBytes,
        totalMiB: cache.memoryBytes / 2 ** 20,
    };
    log(`capacity: ${capacity.numBuckets} buckets x ${capacity.numTilesPerBucket} tiles x ${capacity.numCellsPerTile} cells = ${capacity.numCells} cells; memory ${(cache.memoryBytes / 2 ** 20).toFixed(1)} MiB`);

    const io: TestIo = createTestIo(device, MAX_INPUT_WORDS, MAX_RESULT_WORDS);
    const testKernels = await createTestKernels(device, kernels, cache, io);

    // ---- 1b. The probe-ray layout: a ray written at its byte offset reads back through WorldCacheProbeRays ----
    {
        const rayCount = 7;
        const rayIndex = 5;
        const bytes = new ArrayBuffer(WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + rayCount * WORLD_CACHE_PROBE_RAY_BYTES);
        const view = new DataView(bytes);
        view.setUint32(0, rayCount, true);
        for (let word = 1; word < WORLD_CACHE_PROBE_RAYS_HEADER_BYTES / 4; word++) view.setUint32(word * 4, 0xdeadbeef, true);
        // Every ray distinct, so a ray read at the wrong offset shows.
        const half = { half: 0x3800, minusHalf: 0xb800, z: 0x39a8 } as const; // 0.5, -0.5, 0.70703125 (exact in halves)
        for (let ray = 0; ray < rayCount; ray++) {
            const base = WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES;
            view.setFloat32(base + 0, 10 + ray, true);
            view.setFloat32(base + 4, 20 + ray, true);
            view.setFloat32(base + 8, 30 + ray, true);
            view.setUint32(base + 12, ray === rayIndex ? WORLD_CACHE_PROBE_RAY_BYPASS : 0, true);
            view.setUint32(base + 16, half.half | (half.minusHalf << 16), true);
            view.setUint32(base + 20, ray === rayIndex ? half.z : 0, true);
            view.setFloat32(base + 24, 1.5 + ray, true);
            view.setUint32(base + 28, 100 + ray, true);
            view.setUint32(base + 32, 200 + ray, true);
            view.setUint32(base + 36, ray % 2, true);
            view.setFloat32(base + 40, 0.25, true);
            view.setFloat32(base + 44, 0.125 * ray, true);
        }
        const rays = device.createBuffer({ label: 'probe-ray layout rays', size: bytes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const index = device.createBuffer({ label: 'probe-ray layout index', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const words = device.createBuffer({ label: 'probe-ray layout words', size: PROBE_RAY_READ_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        device.queue.writeBuffer(rays, 0, bytes);
        device.queue.writeBuffer(index, 0, new Uint32Array([rayIndex, 0, 0, 0]));
        const reader = await createProbeRayReader(kernels);
        const encoder = device.createCommandEncoder({ label: 'probe-ray layout' });
        const pass = encoder.beginComputePass({ label: 'probe-ray layout' });
        reader.bind(pass);
        pass.setBindGroup(1, device.createBindGroup({
            label: 'probe-ray layout',
            layout: reader.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: index } },
                { binding: 1, resource: { buffer: rays } },
                { binding: 2, resource: { buffer: words } },
            ],
        }));
        pass.dispatchWorkgroups(1);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const read = await readWords(device, words, 0, PROBE_RAY_READ_WORDS);
        const asFloat = (word: number): number => new Float32Array(new Uint32Array([word]).buffer)[0];
        const got = {
            count: read[0],
            origin: [asFloat(read[1]), asFloat(read[2]), asFloat(read[3])],
            flags: read[4],
            direction: [asFloat(read[5]), asFloat(read[6]), asFloat(read[7])],
            hitT: asFloat(read[8]),
            hitInstance: read[9],
            hitTriangle: read[10],
            hitFrontFace: read[11],
            hitBarycentrics: [asFloat(read[12]), asFloat(read[13])],
        };
        const expected = {
            count: rayCount,
            origin: [10 + rayIndex, 20 + rayIndex, 30 + rayIndex],
            flags: WORLD_CACHE_PROBE_RAY_BYPASS,
            direction: [0.5, -0.5, 0.70703125],
            hitT: 1.5 + rayIndex,
            hitInstance: 100 + rayIndex,
            hitTriangle: 200 + rayIndex,
            hitFrontFace: rayIndex % 2,
            hitBarycentrics: [0.25, 0.125 * rayIndex],
        };
        assert(JSON.stringify(got) === JSON.stringify(expected), `ray ${rayIndex} read through WorldCacheProbeRays as ${JSON.stringify(got)}, written as ${JSON.stringify(expected)}`);
        rays.destroy();
        index.destroy();
        words.destroy();
        data.probeRayLayout = { headerBytes: WORLD_CACHE_PROBE_RAYS_HEADER_BYTES, rayBytes: WORLD_CACHE_PROBE_RAY_BYTES, rayIndex, readBack: 'equal' };
        log(`probe-ray layout: ray ${rayIndex} written at byte ${WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + rayIndex * WORLD_CACHE_PROBE_RAY_BYTES} reads back through WorldCacheProbeRays field by field`);
    }

    // Copy source too: the cache copies the ray count where its first pass's indirect arguments read it.
    const rayBuffer = device.createBuffer({ label: 'no probe rays', size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const radianceBuffer = device.createBuffer({ label: 'no radiance', size: 16, usage: GPUBufferUsage.STORAGE });
    const noRays: WorldCacheProbeRays = { buffer: rayBuffer, count: { buffer: rayBuffer, offset: 0 }, capacity: 1, radiance: { buffer: radianceBuffer } };

    const encodeFrame = (frame: number): void => {
        const input: WorldCacheFrame = { frame, camera: CAMERA, width: RENDER_WIDTH, height: RENDER_HEIGHT };
        const encoder = device.createCommandEncoder({ label: `hash-grid frame ${frame}` });
        cache.encodeFrame(encoder, input, noRays);
        device.queue.submit([encoder.finish()]);
    };
    /** Empties the cache: every slot, stamp, list and counter to zero. */
    const clearCache = (): void => {
        const encoder = device.createCommandEncoder({ label: 'hash-grid clear' });
        // The tile words alone: the words' later sections (the cells' last-reset frames and the update scratch) stay, as cache.reset leaves them.
        encoder.clearBuffer(cache.buffers.tileWords, 0, cache.layout.tileWords.bytes);
        device.queue.submit([encoder.finish()]);
    };
    const runKernel = (name: TestKernelName, input: Uint32Array<ArrayBuffer>, itemCount: number, threadCount = itemCount): void => {
        const kernel: TestKernel = testKernels[name];
        device.queue.writeBuffer(io.params, 0, new Uint32Array([itemCount, threadCount, 0, 0]));
        device.queue.writeBuffer(io.input, 0, input);
        const encoder = device.createCommandEncoder({ label: `hash-grid ${name}` });
        encoder.clearBuffer(io.results);
        encoder.clearBuffer(io.counters);
        const pass = encoder.beginComputePass({ label: `hash-grid ${name}` });
        kernel.kernel.bind(pass);
        pass.setBindGroup(1, kernel.cacheGroup);
        pass.setBindGroup(2, kernel.ioGroup);
        pass.dispatchWorkgroups(Math.ceil(threadCount / TEST_WORKGROUP_SIZE));
        pass.end();
        device.queue.submit([encoder.finish()]);
    };
    const results = (words: number): Promise<Uint32Array> => readWords(device, io.results, 0, words);
    const hashSlots = (bucket: number): Promise<Uint32Array> =>
        readWords(device, cache.buffers.tileWords, (layout.tileWords.hashBuffer + bucket * capacity.numTilesPerBucket) * 4, capacity.numTilesPerBucket);
    const hashSlot = async (tileIndex: number): Promise<number> => (await readWords(device, cache.buffers.tileWords, (layout.tileWords.hashBuffer + tileIndex) * 4, 1))[0];
    const packedTiles = async (): Promise<number[]> => {
        const list = cache.packedTiles;
        const count = (await readWords(device, list.count.buffer, list.count.offset, 1))[0];
        return [...(await readWords(device, cache.buffers.tileWords, list.firstWord * 4, count))];
    };
    const descConstants = (): HashGridCacheDescConstants => ({
        // From the projection, as the cache works it out, so the f32 constant is the same bits.
        cellSize: f32(hashGridCacheCellSizeConstant(verticalFieldOfView(CAMERA.projection), RENDER_WIDTH, RENDER_HEIGHT, options.hashGridCacheCellSize)),
        minCellSize: options.hashGridCacheMinCellSize,
        tileCellRatio: options.hashGridCacheTileCellRatio,
        numBuckets: capacity.numBuckets,
    });
    const descOf = (hit: TestHit): HashGridCacheDesc =>
        hashGridCacheGetDesc({ eyePosition: EYE, hitPosition: hit.position, direction: hit.direction, hitDistance: hit.hitDistance }, descConstants());

    /** The random hits section 2 leaves out near a rounding boundary; section 3b inserts them. */
    let nearBoundaryHits: TestHit[] = [];

    // ---- 2. Hashes and descriptors against the CPU port ----
    {
        const random = seeded(31);
        const inputs = new Uint32Array(HASH_INPUTS);
        inputs.set([0, 1, 2, 3, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff]);
        for (let index = 8; index < HASH_INPUTS; index++) inputs[index] = Math.floor(random() * 2 ** 32);
        runKernel('hashes', inputs, HASH_INPUTS);
        const hashes = await results(HASH_INPUTS * RESULT_WORDS.hashes);
        let hashMismatches = 0;
        for (let index = 0; index < HASH_INPUTS; index++) {
            if (hashes[index * 2] !== pcgHash(inputs[index]) || hashes[index * 2 + 1] !== xxHash(inputs[index])) hashMismatches++;
        }
        assert(hashMismatches === 0, `${hashMismatches} of ${HASH_INPUTS} inputs hash differently on the GPU`);

        encodeFrame(1);
        const constants = descConstants();
        const fixed: TestHit[] = [
            toF32Hit([0.33, 0.21, -9.7], [0, 0, -1], 7.3),
            toF32Hit([-3.3, 1.1, 4.4], [1, 0, 0], 0.05),
            toF32Hit([1.3, 1.8, -2.45], [0, 1, 0], 0.3),
            toF32Hit([-17.3, -4.6, 22.9], [-0.3, 0.2, 0.9], 30.1),
            toF32Hit([120.7, 3.3, -250.1], [0.5, -0.1, -0.8], 260.3),
            toF32Hit([-0.71, -0.33, -0.27], [0.2, -0.9, 0.3], 1.9),
        ];
        const candidates: TestHit[] = [...fixed];
        for (let index = 0; index < DESCRIPTOR_CANDIDATES - fixed.length; index++) {
            const position = [EYE[0] + (random() - 0.5) * 80, EYE[1] + (random() - 0.5) * 80, EYE[2] + (random() - 0.5) * 80];
            candidates.push(toF32Hit(position, [random() - 0.5, random() - 0.5, random() - 0.5], random() * 8));
        }
        const hits = candidates.filter((hit) => clearOfBoundaries(hit, constants));
        nearBoundaryHits = candidates.filter((hit) => !hits.includes(hit));
        for (const hit of fixed) assert(hits.includes(hit), `fixed hit ${JSON.stringify(hit.position)} is near a boundary; move it`);
        runKernel('descriptors', encodeHits(hits), hits.length);
        const descriptors = await results(hits.length * RESULT_WORDS.descriptors);
        const mismatches: string[] = [];
        const levels = new Set<number>();
        let shortRays = 0;
        hits.forEach((hit, index) => {
            const desc = descOf(hit);
            levels.add(desc.level);
            shortRays += desc.shortRay;
            const gpu = descriptors.subarray(index * 4, index * 4 + 4);
            if (gpu[0] !== desc.bucketIndex || gpu[1] !== desc.tileHash || gpu[2] !== desc.cellOffset[0] || gpu[3] !== desc.cellOffset[1]) {
                mismatches.push(`hit ${index} ${JSON.stringify(hit)}: GPU [${[...gpu].join(', ')}], CPU [${desc.bucketIndex}, ${desc.tileHash}, ${desc.cellOffset.join(', ')}]`);
            }
        });
        data.descriptors = { hashInputs: HASH_INPUTS, hits: hits.length, leftOutNearBoundaries: candidates.length - hits.length, levels: [...levels].sort((a, b) => a - b), shortRays, mismatches: mismatches.length };
        assert(mismatches.length === 0, `${mismatches.length} of ${hits.length} descriptors differ from the CPU: ${mismatches.slice(0, 3).join(' | ')}`);
        log(`descriptors: ${hits.length} hits equal the CPU (levels ${[...levels].join(',')}, ${shortRays} short rays; ${candidates.length - hits.length} left out near a boundary); ${HASH_INPUTS} hash inputs equal`);
    }

    // ---- 3. Insert then find ----
    {
        clearCache();
        const frame = 1000;
        encodeFrame(frame);
        const random = seeded(77);
        const constants = descConstants();
        const hits: TestHit[] = [];
        while (hits.length < INSERT_HITS) {
            const hit = toF32Hit([(random() - 0.5) * 60, (random() - 0.5) * 60, (random() - 0.5) * 60], [random() - 0.5, random() - 0.5, random() - 0.5], 1 + random() * 20);
            if (clearOfBoundaries(hit, constants)) hits.push(hit);
        }
        const input = encodeHits(hits);
        runKernel('insert', input, hits.length);
        const inserted = await results(hits.length * RESULT_WORDS.insert);
        const newTiles = (await readWords(device, io.counters, 0, 1))[COUNTER.newTiles];
        runKernel('find', input, hits.length);
        const found = await results(hits.length * RESULT_WORDS.find);

        const tileOfKey = new Map<string, number>();
        for (let index = 0; index < hits.length; index++) {
            const desc = descOf(hits[index]);
            const [cellIndex, tileIndex] = [inserted[index * 3], inserted[index * 3 + 1]];
            assert(cellIndex !== INVALID_ID, `hit ${index} was refused`);
            assert(Math.floor(tileIndex / capacity.numTilesPerBucket) === desc.bucketIndex, `hit ${index} went to tile ${tileIndex}, outside bucket ${desc.bucketIndex}`);
            assert(cellIndex === hashGridCacheCellIndex(desc.cellOffset, tileIndex, 0, capacity), `hit ${index}: cell ${cellIndex}, CPU ${hashGridCacheCellIndex(desc.cellOffset, tileIndex, 0, capacity)}`);
            const key = `${desc.bucketIndex}/${desc.tileHash}`;
            const known = tileOfKey.get(key);
            assert(known === undefined || known === tileIndex, `hits of one tile went to tiles ${known} and ${tileIndex}`);
            tileOfKey.set(key, tileIndex);
            assert(found[index * 2] === cellIndex && found[index * 2 + 1] === tileIndex, `hit ${index} found at cell ${found[index * 2]} tile ${found[index * 2 + 1]}, inserted at ${cellIndex} ${tileIndex}`);
            assert((await hashSlot(tileIndex)) === desc.tileHash, `tile ${tileIndex}'s slot doesn't hold its checksum`);
        }
        assert(newTiles === tileOfKey.size, `${newTiles} inserts claimed a new tile for ${tileOfKey.size} different tiles`);
        const packed = await packedTiles();
        const tiles = [...tileOfKey.values()].sort((a, b) => a - b);
        assert(JSON.stringify([...packed].sort((a, b) => a - b)) === JSON.stringify(tiles), `the packed list holds ${packed.length} tiles, not the ${tiles.length} inserted`);
        const stamps = await Promise.all(tiles.slice(0, 16).map((tile) => readWords(device, cache.buffers.tileWords, (layout.tileWords.decayTileBuffer + tile) * 4, 1)));
        assert(stamps.every((stamp) => stamp[0] === frame), 'a decay stamp is not the frame that touched it');

        // A hit never inserted isn't found.
        const stranger = toF32Hit([500.3, 500.3, 500.3], [0.3, 0.4, 0.86], 50.1);
        runKernel('find', encodeHits([stranger]), 1);
        assert((await results(2))[0] === INVALID_ID, 'a tile never inserted was found');
        data.insertThenFind = { hits: hits.length, tiles: tileOfKey.size };
        log(`insert then find: ${hits.length} hits in ${tileOfKey.size} tiles, all found where inserted, packed list and stamps right`);
    }

    // ---- 3b. Hits near a rounding boundary: insert and find agree; then a reset empties the cache ----
    {
        clearCache();
        encodeFrame(1500);
        const hits = nearBoundaryHits;
        assert(hits.length > 0, 'section 2 left no hits out near a boundary');
        const input = encodeHits(hits);
        runKernel('insert', input, hits.length);
        const inserted = await results(hits.length * RESULT_WORDS.insert);
        runKernel('find', input, hits.length);
        const found = await results(hits.length * RESULT_WORDS.find);
        // Insert and find run the same descriptor code on the same GPU, so they must land on the same cell
        // whichever way a floor near its boundary rounds.
        for (let index = 0; index < hits.length; index++) {
            assert(inserted[index * 3] !== INVALID_ID, `near-boundary hit ${index} was refused`);
            assert(found[index * 2] === inserted[index * 3] && found[index * 2 + 1] === inserted[index * 3 + 1], `near-boundary hit ${index}: inserted at cell ${inserted[index * 3]}, found at ${found[index * 2]}`);
        }
        // As information: how many of them the GPU keys differently from the CPU port.
        runKernel('descriptors', input, hits.length);
        const descriptors = await results(hits.length * RESULT_WORDS.descriptors);
        let differFromCpu = 0;
        hits.forEach((hit, index) => {
            const desc = descOf(hit);
            const gpu = descriptors.subarray(index * 4, index * 4 + 4);
            if (gpu[0] !== desc.bucketIndex || gpu[1] !== desc.tileHash || gpu[2] !== desc.cellOffset[0] || gpu[3] !== desc.cellOffset[1]) differFromCpu++;
        });

        // reset(): every hit misses afterwards, the packed list is empty, and an insert claims a new tile.
        const encoder = device.createCommandEncoder({ label: 'hash-grid reset' });
        cache.reset(encoder);
        device.queue.submit([encoder.finish()]);
        runKernel('find', input, hits.length);
        const afterReset = await results(hits.length * RESULT_WORDS.find);
        for (let index = 0; index < hits.length; index++) assert(afterReset[index * 2] === INVALID_ID, `near-boundary hit ${index} was found after reset()`);
        const packedCount = cache.packedTiles.count;
        assert((await readWords(device, packedCount.buffer, packedCount.offset, 1))[0] === 0, 'the packed list survived reset()');
        encodeFrame(1501);
        runKernel('insert', encodeHits([hits[0]]), 1);
        assert((await results(3))[2] === 1, 'after reset() an insert didn\'t claim a new tile');
        data.nearBoundaryHits = { hits: hits.length, insertAndFindAgree: hits.length, differFromCpu, missAfterReset: hits.length };
        log(`near-boundary hits: ${hits.length} inserted and found at the same cell (${differFromCpu} keyed differently from the CPU port, information); after reset() all miss`);
    }

    // The rest work at the minimum cell (a cell-size option of 0), so tiles can be placed by hand.
    options.hashGridCacheCellSize = 0;
    const minimumCell = descConstants();
    const sharing = hitsSharingABucket(minimumCell, capacity.numTilesPerBucket + 1);
    const bucket = sharing.descs[0].bucketIndex;
    const firstSlot = bucket * capacity.numTilesPerBucket;

    // ---- 4. Two tiles colliding in a bucket ----
    {
        clearCache();
        encodeFrame(2000);
        runKernel('insert', encodeHits([sharing.hits[0]]), 1);
        const first = await results(3);
        runKernel('insert', encodeHits([sharing.hits[1]]), 1);
        const second = await results(3);
        assert(first[1] === firstSlot && first[2] === 1, `the first tile took tile ${first[1]} (new ${first[2]}), not slot 0 of bucket ${bucket}`);
        assert(second[1] === firstSlot + 1 && second[2] === 1, `the colliding tile took tile ${second[1]} (new ${second[2]}), not slot 1`);
        runKernel('find', encodeHits(sharing.hits.slice(0, 2)), 2);
        const found = await results(4);
        assert(found[1] === firstSlot && found[3] === firstSlot + 1, `found the colliding tiles at ${found[1]} and ${found[3]}`);
        assert(found[0] === first[0] && found[2] === second[0], 'found cells differ from the inserted ones');
        data.collision = { bucket, tiles: [first[1], second[1]] };
        log(`collision: two tiles in bucket ${bucket} at slots 0 and 1, both found`);
    }

    // ---- 5. A full bucket refuses the 17th tile, counted ----
    {
        clearCache();
        encodeFrame(3000);
        const sixteen = sharing.hits.slice(0, capacity.numTilesPerBucket);
        runKernel('insert', encodeHits(sixteen), sixteen.length);
        const filled = await results(sixteen.length * 3);
        const slots = new Set<number>();
        for (let index = 0; index < sixteen.length; index++) {
            assert(filled[index * 3] !== INVALID_ID, `tile ${index} of 16 was refused`);
            slots.add(filled[index * 3 + 1]);
        }
        assert(slots.size === 16 && Math.min(...slots) === firstSlot && Math.max(...slots) === firstSlot + 15, 'the 16 tiles don\'t fill the bucket\'s 16 slots');
        const overflowBefore = (await readWords(device, cache.bucketOverflowCount.buffer, cache.bucketOverflowCount.offset, 1))[0];
        const seventeenth = sharing.hits[capacity.numTilesPerBucket];
        runKernel('insert', encodeHits([seventeenth]), 1);
        const refused = await results(3);
        const overflow = (await readWords(device, cache.bucketOverflowCount.buffer, cache.bucketOverflowCount.offset, 1))[0];
        assert(refused[0] === INVALID_ID && refused[2] === 0, `the 17th tile got cell ${refused[0]}`);
        assert(overflowBefore === 0 && overflow === 1, `overflow count ${overflowBefore} before, ${overflow} after (expected 0 and 1)`);
        runKernel('find', encodeHits([seventeenth]), 1);
        assert((await results(2))[0] === INVALID_ID, 'the refused tile was found');
        const slotHashes = await hashSlots(bucket);
        for (let index = 0; index < sixteen.length; index++) {
            const slot = filled[index * 3 + 1] - firstSlot;
            assert(slotHashes[slot] === sharing.descs[index].tileHash, `slot ${slot} doesn't hold tile ${index}'s checksum`);
        }
        // The next frame's ClearCounters zeroes the count.
        encodeFrame(3001);
        assert((await readWords(device, cache.bucketOverflowCount.buffer, cache.bucketOverflowCount.offset, 1))[0] === 0, 'ClearCounters left the overflow count');
        data.fullBucket = { bucket, overflowCount: overflow };
        log(`full bucket: 16 tiles fill bucket ${bucket}; the 17th is refused, overflow count 1, then 0 after ClearCounters`);
    }

    // ---- 6. Many threads inserting one tile get one slot ----
    // This can't tell the weak exchange's retry (register D22) from no retry: on D3D12 the weak
    // exchange is likely a strong one underneath, so a spurious failure may never happen here.
    {
        clearCache();
        encodeFrame(4000);
        const hit = sharing.hits[0];
        runKernel('insert', encodeHits([hit]), 1, CONCURRENT_THREADS);
        const all = await results(CONCURRENT_THREADS * 3);
        const newTiles = (await readWords(device, io.counters, 0, 1))[COUNTER.newTiles];
        let newFlags = 0;
        for (let thread = 0; thread < CONCURRENT_THREADS; thread++) {
            assert(all[thread * 3] === all[0] && all[thread * 3 + 1] === firstSlot, `thread ${thread} got cell ${all[thread * 3]} tile ${all[thread * 3 + 1]}`);
            newFlags += all[thread * 3 + 2];
        }
        assert(newFlags === 1 && newTiles === 1, `${newFlags} threads saw a new tile (counter ${newTiles}); one should`);
        const slotHashes = await hashSlots(bucket);
        assert(slotHashes[0] === sharing.descs[0].tileHash && slotHashes.slice(1).every((hash) => hash === 0), `the bucket holds [${[...slotHashes].join(', ')}]`);
        assert((await packedTiles()).length === 1, 'the tile joined the packed list more than once');
        data.concurrentInsert = { threads: CONCURRENT_THREADS, newTiles };
        log(`concurrent insert: ${CONCURRENT_THREADS} threads, one tile, one slot, one new`);
    }

    // ---- 7. Purge after 50 idle frames, and across the frame counter's wrap ----
    const purgeRun = async (startFrame: number, label: string): Promise<Record<string, number>> => {
        clearCache();
        encodeFrame(startFrame);
        const idle = sharing.hits[0];
        const touched = sharing.hits[1];
        runKernel('insert', encodeHits([idle, touched]), 2);
        const inserted = await results(6);
        const idleTile = inserted[1];
        const touchedTile = inserted[4];
        assert(inserted[0] !== INVALID_ID && inserted[3] !== INVALID_ID, `${label}: insert failed`);
        for (let age = 1; age <= 2 * TILE_DECAY; age++) {
            encodeFrame(startFrame + age);
            if (age === 30) runKernel('touch', encodeHits([touched]), 1);
            if (age === TILE_DECAY - 1) {
                assert((await hashSlot(idleTile)) !== 0, `${label}: the idle tile was purged after ${age} frames`);
                assert((await packedTiles()).length === 2, `${label}: the packed list lost a tile before 50 frames`);
            }
            if (age === TILE_DECAY) {
                assert((await hashSlot(idleTile)) === 0, `${label}: the idle tile survived ${age} frames`);
                assert((await hashSlot(touchedTile)) !== 0, `${label}: the touched tile was purged with the idle one`);
                const packed = await packedTiles();
                assert(packed.length === 1 && packed[0] === touchedTile, `${label}: the packed list after the purge is [${packed.join(', ')}]`);
            }
            if (age === 30 + TILE_DECAY - 1) assert((await hashSlot(touchedTile)) !== 0, `${label}: the touched tile was purged before its 50th frame`);
            if (age === 30 + TILE_DECAY) {
                assert((await hashSlot(touchedTile)) === 0, `${label}: the touched tile survived its 50th frame`);
                assert((await packedTiles()).length === 0, `${label}: the packed list isn't empty`);
            }
        }
        return { startFrameIndex: startFrame >>> 0, idlePurgedAtAge: TILE_DECAY, touchedAtAge: 30, touchedPurgedAtAge: 30 + TILE_DECAY };
    };
    data.purge = await purgeRun(5000, 'purge');
    // The idle tile's stamp is 2^32 - 20; it is purged at frame index 30, after the wrap.
    data.purgeAcrossWrap = await purgeRun(2 ** 32 - 20, 'purge across the wrap');
    log(`purge: a tile untouched for ${TILE_DECAY} frames is freed at exactly the ${TILE_DECAY}th, a touched one lives on; the same from frame 2^32 - 20 across the wrap`);

    // ---- 8. Find stops at the bucket's first empty slot (Capsaicin's behaviour, kept) ----
    {
        clearCache();
        encodeFrame(9000);
        const [first, second] = sharing.hits;
        runKernel('insert', encodeHits([first]), 1);
        runKernel('insert', encodeHits([second]), 1);
        const secondTile = (await results(3))[1];
        assert(secondTile === firstSlot + 1, 'the second tile isn\'t in slot 1');
        for (let age = 1; age <= TILE_DECAY; age++) {
            encodeFrame(9000 + age);
            runKernel('touch', encodeHits([second]), 1);
        }
        assert((await hashSlot(firstSlot)) === 0 && (await hashSlot(firstSlot + 1)) !== 0, 'slot 0 should be purged and slot 1 kept');
        runKernel('find', encodeHits([second]), 1);
        const found = await results(2);
        assert(found[0] === INVALID_ID, 'FindCell looked past an empty slot; Capsaicin\'s stops there');
        runKernel('insert', encodeHits([second]), 1);
        const again = await results(3);
        assert(again[1] === firstSlot && again[2] === 1, `inserting it again took tile ${again[1]} (new ${again[2]}); Capsaicin's insert claims the empty slot 0`);
        data.findStopsAtEmptySlot = { purgedSlot: 0, keptSlot: 1, reinsertedSlot: again[1] - firstSlot };
        log('find stops at the first empty slot: slot 0 purged, slot 1\'s tile not found, re-inserted into slot 0 (Capsaicin\'s behaviour)');
    }

    // ---- 9. Halves rounded to nearest, ties to even, clamped (register D31) ----
    {
        const step = 2 ** -10;
        const values: number[] = [
            // Ties: halfway between two halves; the even one wins (the old rule kept the truncated one).
            1 + step / 2, 1 + 3 * step / 2, -(1 + step / 2), -(1 + 3 * step / 2), 1000.25, 1000.75, 0.5 + step / 4, 3 * 2 ** -25, -(5 * 2 ** -25),
            // Exact halves, and the clamp.
            0, 1, -2, 65504, 70000, -1e9, 65520,
        ];
        const random = seeded(91);
        while (values.length < HASH_INPUTS) values.push((random() - 0.5) * 2 ** (random() * 40 - 24));
        const floats = new Float32Array(values);
        runKernel('packHalves', new Uint32Array(floats.buffer), floats.length / 2);
        const packed = await results(floats.length / 2);
        const clampHalf = (value: number): number => Math.min(Math.max(value, -65504), 65504);
        const mismatches: string[] = [];
        for (let index = 0; index < floats.length; index++) {
            const gpu = (packed[index >> 1] >>> ((index & 1) * 16)) & 0xffff;
            const expected = floatToHalfBits(clampHalf(floats[index]));
            if (gpu !== expected) mismatches.push(`${floats[index]}: GPU 0x${gpu.toString(16)}, f32tof16 0x${expected.toString(16)}`);
        }
        data.packHalves = { values: floats.length, ties: 9, mismatches: mismatches.length };
        assert(mismatches.length === 0, `${mismatches.length} of ${floats.length} values pack to another half than f32tof16's: ${mismatches.slice(0, 4).join(' | ')}`);
        log(`halves: ${floats.length} values (9 exact ties, the clamp) pack as f32tof16 does, ties to even`);
    }

    rayBuffer.destroy();
    radianceBuffer.destroy();
    cache.dispose();
    await drainDevice(device);
    reportGpuTest({ status: 'pass', data, adapter: adapterSummary, deviceErrors, startTime, logLines });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), data, adapter: adapterSummary, deviceErrors, startTime, logLines });
}
