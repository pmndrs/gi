/**
 * B1: the GPU LBVH builder's pieces and trees, on this GPU.
 *
 * 1. **Scan:** random arrays of several sizes (one tile, several, many)
 *    against a CPU prefix sum.
 * 2. **Sort:** random 30-bit keys with many repeats, in segments of awkward
 *    sizes, against a stable CPU sort of each segment: keys in order, equal
 *    keys in their original order, and no key leaving its segment.
 * 3. **Trees:** meshes of every awkward kind (one, two and three triangles,
 *    tile boundaries, a grid with 16-bit indices, all-equal triangles, a flat
 *    mesh, flags, a mesh already on the GPU with interleaved vertices, a large
 *    soup) built in one batch, read back, and run through the layout checker
 *    (packages/bvh/src/layout-check.ts). Each tree's SAH cost is compared with
 *    the CPU twin's (lbvh-cpu.ts) and with tinybvh's (B0).
 *
 * The ray oracle for B1 trees is the trace-oracle page (variant `watertight-lbvh`).
 */

import { createRadixSort, createRadixSortPipelines, sortTileTable } from '../../../../packages/bvh/src/gpu-radix-sort.ts';
import { createScanBindGroups, createScanPipelines, encodeScan, scanParams, scanTileSumWords } from '../../../../packages/bvh/src/gpu-scan.ts';
import { GpuLbvhBuilder, type GpuLbvhGeometry } from '../../../../packages/bvh/src/gpu-lbvh.ts';
import { checkGpuLayout, treeSahCost } from '../../../../packages/bvh/src/layout-check.ts';
import { buildLbvhOnCpu } from '../../../../packages/bvh/src/lbvh-cpu.ts';
import { createTinyBvhBuilder, type TinyBvhInput } from '../../../../packages/bvh/src/tinybvh.ts';
import { assert, runGpuTest } from '../../client.ts';

/** Deterministic random numbers in [0, 1). */
function randomSource(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
    };
}

function randomSoup(count: number, seed: number, spread = 20): Float32Array {
    const random = randomSource(seed);
    const positions = new Float32Array(count * 9);
    for (let triangle = 0; triangle < count; triangle++) {
        const center = [random() * spread - spread / 2, random() * spread - spread / 2, random() * spread - spread / 2];
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) positions[triangle * 9 + corner * 3 + axis] = center[axis] + random() - 0.5;
        }
    }
    return positions;
}

/** A wavy grid of `size` x `size` quads, indexed with 16-bit indices when they fit. */
function grid(size: number): TinyBvhInput {
    const positions = new Float32Array((size + 1) * (size + 1) * 3);
    for (let z = 0; z <= size; z++) {
        for (let x = 0; x <= size; x++) positions.set([x * 0.1, Math.sin(x * 0.3) * Math.cos(z * 0.2), z * 0.1], (z * (size + 1) + x) * 3);
    }
    const vertexCount = (size + 1) * (size + 1);
    const indices = vertexCount <= 0xffff ? new Uint16Array(size * size * 6) : new Uint32Array(size * size * 6);
    for (let z = 0; z < size; z++) {
        for (let x = 0; x < size; x++) {
            const corner = z * (size + 1) + x;
            indices.set([corner, corner + 1, corner + size + 1, corner + 1, corner + size + 2, corner + size + 1], (z * size + x) * 6);
        }
    }
    return { positions, indices };
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'bvh-lbvh readback', size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.destroy();
    return copy;
}

function storageBuffer(device: GPUDevice, label: string, data: Uint32Array | Float32Array | number): GPUBuffer {
    const size = Math.max(16, typeof data === 'number' ? data : data.byteLength);
    const buffer = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data as Uint32Array<ArrayBuffer>);
    return buffer;
}

async function checkScan(device: GPUDevice, log: (message: string) => void): Promise<string[]> {
    const failures: string[] = [];
    const pipelines = await createScanPipelines(device, 'bvh-lbvh');
    for (const count of [1, 7, 2047, 2048, 2049, 100_000, 3_000_000]) {
        const random = randomSource(count);
        const input = Uint32Array.from({ length: count }, () => Math.floor(random() * 4));
        const data = storageBuffer(device, 'scan data', input);
        const tileSums = storageBuffer(device, 'scan tile sums', scanTileSumWords(count) * 4);
        const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(params, 0, scanParams(count));
        const groups = createScanBindGroups(device, pipelines, { data: { buffer: data }, tileSums: { buffer: tileSums }, params: { buffer: params }, count }, 'scan');
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        encodeScan(pass, pipelines, groups, device.limits.maxComputeWorkgroupsPerDimension);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const result = new Uint32Array(await readBuffer(device, data, count * 4));
        const total = new Uint32Array(await readBuffer(device, tileSums, scanTileSumWords(count) * 4))[Math.ceil(count / 2048)];
        let running = 0;
        let wrong = -1;
        for (let index = 0; index < count; index++) {
            if (result[index] !== running && wrong < 0) wrong = index;
            running += input[index];
        }
        if (wrong >= 0) failures.push(`scan of ${count}: item ${wrong} is ${result[wrong]}`);
        if (total !== running) failures.push(`scan of ${count}: total ${total}, expected ${running}`);
        log(`scan of ${count}: ${wrong < 0 && total === running ? 'right' : 'WRONG'}`);
        data.destroy();
        tileSums.destroy();
        params.destroy();
    }
    return failures;
}

async function checkSort(device: GPUDevice, log: (message: string) => void): Promise<string[]> {
    const failures: string[] = [];
    const scan = await createScanPipelines(device, 'bvh-lbvh');
    const pipelines = await createRadixSortPipelines(device, scan, 'bvh-lbvh');
    const segmentSets = [[1], [5], [2048], [2049], [50_000], [1_000_000], [1, 2047, 2049, 3, 70_000, 1, 4096, 10]];
    for (const segments of segmentSets) {
        const count = segments.reduce((total, size) => total + size, 0);
        const random = randomSource(count + 99);
        // Few distinct keys, so stability is tested; all 30 bits used.
        const palette = Uint32Array.from({ length: 64 }, () => Math.floor(random() * 2 ** 30));
        const keys = Uint32Array.from({ length: count }, () => palette[Math.floor(random() * palette.length)]);
        const values = Uint32Array.from({ length: count }, (_, index) => index);
        const table = sortTileTable(segments);
        const buffers = {
            keysA: storageBuffer(device, 'keys a', keys),
            valuesA: storageBuffer(device, 'values a', values),
            keysB: storageBuffer(device, 'keys b', count * 4),
            valuesB: storageBuffer(device, 'values b', count * 4),
            offsets: storageBuffer(device, 'offsets', table.offsetWords * 4),
            tileSums: storageBuffer(device, 'tile sums', scanTileSumWords(table.offsetWords) * 4),
            tables: storageBuffer(device, 'tile table', table.tiles),
            tilesOffset: 0,
            params: device.createBuffer({ size: 256 * 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
            paramsOffset: 0,
            paramsStride: 256,
        };
        const sort = createRadixSort(device, pipelines, buffers, table, 30, 'sort');
        sort.writeParams(device.queue);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        sort.encode(pass);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const sortedKeys = new Uint32Array(await readBuffer(device, buffers.keysA, count * 4));
        const sortedValues = new Uint32Array(await readBuffer(device, buffers.valuesA, count * 4));
        // Each segment sorted on its own: a key never leaves its segment.
        const expected = new Uint32Array(count);
        let start = 0;
        for (const size of segments) {
            expected.set(values.subarray(start, start + size).sort((a, b) => keys[a] - keys[b] || a - b), start);
            start += size;
        }
        let wrong = -1;
        for (let index = 0; index < count && wrong < 0; index++) {
            if (sortedValues[index] !== expected[index] || sortedKeys[index] !== keys[expected[index]]) wrong = index;
        }
        const name = `sort of ${segments.length === 1 ? count : `${segments.length} segments (${segments.join(', ')})`}`;
        if (wrong >= 0) failures.push(`${name}: position ${wrong} holds key ${sortedKeys[wrong]} value ${sortedValues[wrong]}, expected value ${expected[wrong]}`);
        log(`${name}: ${wrong < 0 ? 'right and stable' : 'WRONG'}`);
        for (const buffer of [buffers.keysA, buffers.valuesA, buffers.keysB, buffers.valuesB, buffers.offsets, buffers.tileSums, buffers.tables, buffers.params]) buffer.destroy();
    }
    return failures;
}

interface TreeCase {
    name: string;
    /** What the checker compares with (CPU arrays). */
    geometry: TinyBvhInput;
    /** What the GPU builds from; the CPU arrays unless the case puts the mesh on the GPU first. */
    input?: GpuLbvhGeometry;
}

function treeCases(device: GPUDevice): { cases: TreeCase[]; owned: GPUBuffer[] } {
    const owned: GPUBuffer[] = [];
    const cases: TreeCase[] = [
        { name: 'one triangle', geometry: { positions: randomSoup(1, 1) } },
        { name: 'two triangles', geometry: { positions: randomSoup(2, 2) } },
        { name: 'three triangles', geometry: { positions: randomSoup(3, 3) } },
        { name: 'one sort tile', geometry: { positions: randomSoup(2048, 4) } },
        { name: 'one past a tile', geometry: { positions: randomSoup(2049, 5) } },
        { name: 'grid, 16-bit indices', geometry: grid(150) },
        {
            name: 'all triangles equal',
            geometry: { positions: Float32Array.from({ length: 3000 * 9 }, (_, index) => [0, 0, 0, 1, 0, 0, 0, 1, 0][index % 9]) },
        },
        { name: 'flat (no height)', geometry: { positions: randomSoup(4000, 6).map((value, index) => (index % 3 === 1 ? 0 : value)) } },
        { name: 'flags', geometry: { positions: randomSoup(500, 7), triangleFlags: Uint32Array.from({ length: 500 }, (_, index) => index % 3) } },
        { name: 'random soup 300k', geometry: { positions: randomSoup(300_000, 8, 200) } },
    ];

    // The same grid again, but from GPU buffers: interleaved vertices (position, then 3 unused floats) and 32-bit indices.
    const source = grid(120);
    const vertexCount = source.positions.length / 3;
    const interleaved = new Float32Array(vertexCount * 6);
    for (let vertex = 0; vertex < vertexCount; vertex++) interleaved.set(source.positions.subarray(vertex * 3, vertex * 3 + 3), vertex * 6 + 1);
    const indices = Uint32Array.from(source.indices as Uint16Array);
    const positionBuffer = storageBuffer(device, 'interleaved vertices', interleaved);
    const indexBuffer = storageBuffer(device, 'indices', indices);
    owned.push(positionBuffer, indexBuffer);
    cases.push({
        name: 'GPU buffers, interleaved',
        geometry: { positions: source.positions, indices },
        input: { positionBuffer, vertexCount, positionStrideFloats: 6, positionOffsetFloats: 1, indexBuffer, indexFormat: 'uint32', triangleCount: indices.length / 3 },
    });
    return { cases, owned };
}

await runGpuTest(
    async ({ device, log }) => {
        assert(GpuLbvhBuilder.isSupported(device), 'this device has no subgroups; the fallback is tested by bvh-lbvh-fallback');
        const failures: string[] = [];
        failures.push(...(await checkScan(device, log)));
        failures.push(...(await checkSort(device, log)));

        const builder = new GpuLbvhBuilder(device);
        await builder.prepare();
        const tinyBvh = await createTinyBvhBuilder();
        const { cases, owned } = treeCases(device);
        // Every case in one batch, so meshes of every size share the dispatches.
        const start = performance.now();
        const built = await builder.buildMany(cases.map((entry) => entry.input ?? entry.geometry));
        const batchMs = performance.now() - start;
        const layouts = await builder.readLayouts(built);
        built.destroy();
        const trees: Record<string, unknown> = {};
        for (const [index, layout] of layouts.entries()) {
            const { name, geometry } = cases[index];
            if (!layout) {
                failures.push(`${name}: ${built.meshes[index].problem}`);
                continue;
            }
            const problems = checkGpuLayout(layout, geometry);
            for (const problem of problems) failures.push(`${name}: ${problem}`);
            const cost = treeSahCost(layout);
            const twin = buildLbvhOnCpu(geometry);
            const twinCost = treeSahCost(twin);
            const tinyCost = treeSahCost(tinyBvh.buildLayout(geometry).layout);
            // The CPU twin computes the same tree but for float rounding (lbvh-cpu.ts); 1% covers that.
            if (Math.abs(cost - twinCost) > 0.01 * twinCost) failures.push(`${name}: SAH cost ${cost.toFixed(3)}, the CPU twin's ${twinCost.toFixed(3)}`);
            trees[name] = {
                triangles: layout.triangleCount,
                nodes: layout.nodeCount,
                twinNodes: twin.nodeCount,
                leaves: layout.leafCount,
                depth: layout.maxDepth,
                sahCost: Number(cost.toFixed(4)),
                sahCostVsTinyBvh: Number((cost / tinyCost).toFixed(4)),
            };
            log(
                `${name}: ${layout.triangleCount} triangles, ${layout.nodeCount} nodes (twin ${twin.nodeCount}), depth ${layout.maxDepth}, ` +
                    `SAH ${cost.toFixed(2)} (twin ${twinCost.toFixed(2)}, tinybvh ${tinyCost.toFixed(2)}), ${problems.length === 0 ? 'valid' : 'INVALID'}`,
            );
        }
        for (const buffer of owned) buffer.destroy();
        assert(failures.length === 0, failures.slice(0, 15).join('\n'));
        return { batchMs: Math.round(batchMs), trees, subgroupMinSize: device.adapterInfo.subgroupMinSize, subgroupMaxSize: device.adapterInfo.subgroupMaxSize };
    },
    { optionalFeatures: ['subgroups'] },
);
