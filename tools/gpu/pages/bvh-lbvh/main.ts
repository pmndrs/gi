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
import { GpuLbvhBuilder } from '../../../../packages/bvh/src/gpu-lbvh.ts';
import { checkGpuLayout, treeSahCost } from '../../../../packages/bvh/src/layout-check.ts';
import { buildLbvhOnCpu } from '../../../../packages/bvh/src/lbvh-cpu.ts';
import { createTinyBvhBuilder } from '../../../../packages/bvh/src/tinybvh.ts';
import { assert, runGpuTest } from '../../client.ts';
import { randomSource, storageBuffer, treeCases } from '../bvh-shared/tree-cases.ts';

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
