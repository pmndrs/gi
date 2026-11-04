/**
 * GPU unit tests of two pieces of DDGI's WGSL, run exactly as the kernels
 * include them:
 *
 * 1. `ddgiStochasticHalf` (wgsl/half.ts): its result is always one of the two
 *    half floats around the value (so packing is exact), and on average it
 *    equals the value (unbiased), checked at ordinary values, at both sides
 *    of exponent boundaries, at exact powers of two, among subnormal halves,
 *    and at 0;
 * 2. `ddgiBorderSource` (wgsl/border.ts): for every texel of 6- and 14-texel
 *    tiles, the same source texel as `borderSource` in octahedral.ts, whose
 *    rule a CPU unit test checks against the sphere.
 */

import { assert, runGpuTest } from '../../client.ts';
import { borderSource } from '../../../../packages/ddgi-core/src/techniques/ddgi/octahedral.ts';
import { DDGI_BORDER_SOURCE_WGSL } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/border.ts';
import { DDGI_STOCHASTIC_HALF_WGSL } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/half.ts';
import { halfToFloat } from '../ddgi-shared/harness.ts';

/** Random numbers per value: the mean of this many roundings is checked against the value. */
const DRAWS = 16384;

const SMALLEST_SUBNORMAL_HALF = 2 ** -24;
const SMALLEST_NORMAL_HALF = 2 ** -14;

/** Values to round: ordinary ones, both sides of exponent boundaries, exact powers of two, subnormals, zero. */
const VALUES = [
    1.2345, 0.4987, 1.5707963, 3.14159, 100.37, 6789.123,
    1 - 1e-5, 1 + 1e-4, 2 - 1e-4, 2 + 1e-3, 0.5 - 1e-6, 0.25 + 1e-6,
    1, 2, 0.5, 1024,
    3.3 * SMALLEST_SUBNORMAL_HALF, SMALLEST_NORMAL_HALF * 0.75, SMALLEST_NORMAL_HALF * 1.0001,
    0,
];

const HALF_WGSL = /* wgsl */ `
${DDGI_STOCHASTIC_HALF_WGSL}
@group(0) @binding(0) var<storage, read> values: array<f32>;
// Per value: the smallest and largest result, how many rounded up, and how many weren't exact halves.
@group(0) @binding(1) var<storage, read_write> rounded: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&values)) {
        return;
    }
    let value = values[id.x];
    var smallest = 3.0e38;
    var largest = -3.0e38;
    var roundedUp = 0.0;
    var notExact = 0.0;
    for (var draw = 0u; draw < ${DRAWS}u; draw++) {
        // Evenly spaced random numbers: the exact expected value of the rounding.
        let random = (f32(draw) + 0.5) / f32(${DRAWS}u);
        let result = ddgiStochasticHalf(value, random);
        // Exact when packed: a half float survives a round trip unchanged.
        if (unpack2x16float(pack2x16float(vec2f(result, 0.0))).x != result) {
            notExact += 1.0;
        }
        smallest = min(smallest, result);
        largest = max(largest, result);
        if (result > value) {
            roundedUp += 1.0;
        }
    }
    rounded[id.x] = vec4f(smallest, largest, roundedUp, notExact);
}
`;

const BORDER_WGSL = /* wgsl */ `
${DDGI_BORDER_SOURCE_WGSL}
@group(0) @binding(0) var<storage, read_write> sources: array<vec2u>;

// One invocation per tile texel; x: column, y: row, z: 0 for 6-texel tiles, 1 for 14.
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let interior = select(6u, 14u, id.z == 1u);
    let tile = interior + 2u;
    if (id.x >= tile || id.y >= tile) {
        return;
    }
    sources[(id.z * 16u + id.y) * 16u + id.x] = ddgiBorderSource(id.xy, interior);
}
`;

async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** The two half floats around a value (equal when it is one), from the bit patterns. */
function halfNeighbours(value: number): [number, number] {
    let below = 0;
    for (let bits = 0; bits < 0x7c00; bits++) {
        const half = halfToFloat(bits);
        if (half <= value) below = half;
        else return [below, half];
    }
    return [below, below];
}

await runGpuTest(async ({ device, log }) => {
    const failures: string[] = [];

    // 1. Stochastic rounding.
    const values = new Float32Array(VALUES);
    const valueBuffer = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(valueBuffer, 0, values);
    const roundedBuffer = device.createBuffer({ size: values.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const halfPipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: HALF_WGSL }), entryPoint: 'main' } });
    let encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(halfPipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: halfPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: valueBuffer } }, { binding: 1, resource: { buffer: roundedBuffer } }] }));
    pass.dispatchWorkgroups(Math.ceil(values.length / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);
    const rounded = new Float32Array(await readBuffer(device, roundedBuffer, values.length * 16));
    const halfResults: Record<string, unknown>[] = [];
    values.forEach((value, index) => {
        const [smallest, largest, roundedUp, notExact] = rounded.subarray(index * 4, index * 4 + 4);
        const [below, above] = halfNeighbours(value);
        const step = above - below;
        // Unbiased: a value between two halves rounds up with probability (value - below) / step,
        // so the mean result is the value. DRAWS evenly spaced random numbers get that share to within 1 / DRAWS.
        const mean = step > 0 ? below + (roundedUp / DRAWS) * step : below;
        const error = Math.abs(mean - value);
        halfResults.push({ value, below, above, roundedUpShare: roundedUp / DRAWS, mean, errorInSteps: step > 0 ? error / step : 0 });
        if (notExact > 0) failures.push(`${value}: ${notExact} results were not half floats`);
        if (smallest < below || largest > (step > 0 ? above : below)) failures.push(`${value}: results ${smallest}..${largest} leave its neighbours ${below}, ${above}`);
        if (step > 0 ? error > (1.5 * step) / DRAWS : error > 0) failures.push(`${value}: mean ${mean} off by ${error} (step ${step})`);
    });
    log(`stochastic rounding: ${values.length} values, worst mean error ${Math.max(...halfResults.map((result) => result.errorInSteps as number)).toExponential(2)} of a step`);

    // 2. The border rule against octahedral.ts.
    const sourceBuffer = device.createBuffer({ size: 2 * 16 * 16 * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const borderPipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: BORDER_WGSL }), entryPoint: 'main' } });
    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(borderPipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: borderPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: sourceBuffer } }] }));
    pass.dispatchWorkgroups(16, 16, 2);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const sources = new Uint32Array(await readBuffer(device, sourceBuffer, 2 * 16 * 16 * 8));
    let borderTexels = 0;
    for (const [layer, interior] of [[0, 6], [1, 14]]) {
        for (let row = 0; row < interior + 2; row++) {
            for (let column = 0; column < interior + 2; column++) {
                const base = ((layer * 16 + row) * 16 + column) * 2;
                const expected = borderSource(column, row, interior);
                if (sources[base] !== expected[0] || sources[base + 1] !== expected[1]) {
                    failures.push(`border ${interior}: tile texel (${column}, ${row}) copies (${sources[base]}, ${sources[base + 1]}) on the GPU, (${expected.join(', ')}) in octahedral.ts`);
                }
                borderTexels++;
            }
        }
    }
    log(`border rule: ${borderTexels} tile texels agree with octahedral.ts`);
    valueBuffer.destroy();
    roundedBuffer.destroy();
    sourceBuffer.destroy();
    assert(failures.length === 0, failures.slice(0, 10).join('\n'));
    return { stochasticHalf: halfResults, borderTexels };
});
