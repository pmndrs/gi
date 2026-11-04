/**
 * Smoke test: the whole chain works. A compute shader writes values we can
 * compute on the CPU, the page copies them to a mappable buffer and compares.
 */

import { assert, runGpuTest } from '../../client.ts';

const WORKGROUP_SIZE = 64;
const WORKGROUP_COUNT = 16;
const ELEMENT_COUNT = WORKGROUP_SIZE * WORKGROUP_COUNT;

// Each invocation writes one u32 and one f32. The f32 value is a multiple of
// 0.25 below 2^24, so it is exactly representable and must match bit for bit.
// A workgroup-wide sum through shared memory checks barriers as well.
const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> integers: array<u32>;
@group(0) @binding(1) var<storage, read_write> floats: array<f32>;
@group(0) @binding(2) var<storage, read_write> workgroupSums: array<u32>;

var<workgroup> shared_values: array<u32, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) group_id: vec3u,
) {
    let index = global_id.x;
    integers[index] = index * index + 7u;
    floats[index] = f32(index) * 0.25 + 1.0;

    shared_values[local_id.x] = local_id.x;
    workgroupBarrier();
    if (local_id.x == 0u) {
        var sum = 0u;
        for (var i = 0u; i < ${WORKGROUP_SIZE}u; i++) {
            sum += shared_values[i];
        }
        workgroupSums[group_id.x] = sum;
    }
}`;

await runGpuTest(async ({ adapter, device, log }) => {
    const module = device.createShaderModule({ label: 'smoke shader', code: SHADER });
    const pipeline = device.createComputePipeline({
        label: 'smoke pipeline',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
    });

    const integerBytes = ELEMENT_COUNT * 4;
    const floatBytes = ELEMENT_COUNT * 4;
    const sumBytes = WORKGROUP_COUNT * 4;
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const integers = device.createBuffer({ size: integerBytes, usage: storageUsage });
    const floats = device.createBuffer({ size: floatBytes, usage: storageUsage });
    const sums = device.createBuffer({ size: sumBytes, usage: storageUsage });
    const readback = device.createBuffer({
        size: integerBytes + floatBytes + sumBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: integers } },
            { binding: 1, resource: { buffer: floats } },
            { binding: 2, resource: { buffer: sums } },
        ],
    });

    const encoder = device.createCommandEncoder({ label: 'smoke encoder' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(WORKGROUP_COUNT);
    pass.end();
    encoder.copyBufferToBuffer(integers, 0, readback, 0, integerBytes);
    encoder.copyBufferToBuffer(floats, 0, readback, integerBytes, floatBytes);
    encoder.copyBufferToBuffer(sums, 0, readback, integerBytes + floatBytes, sumBytes);
    device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange();
    const gotIntegers = new Uint32Array(mapped, 0, ELEMENT_COUNT);
    const gotFloats = new Float32Array(mapped, integerBytes, ELEMENT_COUNT);
    const gotSums = new Uint32Array(mapped, integerBytes + floatBytes, WORKGROUP_COUNT);

    let wrongIntegers = 0;
    let wrongFloats = 0;
    for (let index = 0; index < ELEMENT_COUNT; index++) {
        if (gotIntegers[index] !== (index * index + 7) >>> 0) wrongIntegers++;
        if (gotFloats[index] !== index * 0.25 + 1) wrongFloats++;
    }
    const expectedSum = (WORKGROUP_SIZE * (WORKGROUP_SIZE - 1)) / 2;
    const wrongSums = gotSums.filter((sum) => sum !== expectedSum).length;
    log(`read back ${ELEMENT_COUNT} u32, ${ELEMENT_COUNT} f32, ${WORKGROUP_COUNT} workgroup sums`);

    readback.unmap();
    for (const buffer of [integers, floats, sums, readback]) buffer.destroy();

    assert(wrongIntegers === 0, `${wrongIntegers} of ${ELEMENT_COUNT} u32 values were wrong`);
    assert(wrongFloats === 0, `${wrongFloats} of ${ELEMENT_COUNT} f32 values were wrong`);
    assert(wrongSums === 0, `${wrongSums} of ${WORKGROUP_COUNT} workgroup sums were wrong (barrier problem?)`);

    return {
        elements: ELEMENT_COUNT,
        maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup,
        maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
        adapterHasTimestampQuery: adapter.features.has('timestamp-query'),
    };
});
