/**
 * Reads what one DDGI probe stores for a direction: its irradiance tile,
 * filtered bilinearly at the direction's octahedral spot (as sampling does
 * for the surface normal), with no visibility or interpolation between
 * probes. For comparing probes against a path-traced irradiance at the
 * probe's own position (T1.9 Sponza investigation): if the probes agree but
 * surfaces read high, the excess comes from interpolation and visibility;
 * if the probes are high, it comes from the update.
 *
 * Not a page itself (no index.html).
 */

import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { DDGIField } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { DDGI_SAMPLE_WGSL } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/sample.ts';
import { FIELD_GROUP, fieldBindGroup, IO_GROUP, readBuffer } from './harness.ts';

export interface ProbeDirection {
    /** Probe index, p = x + nx × (z + nz × y). */
    readonly probe: number;
    /** Unit direction (the irradiance tile's normal). */
    readonly direction: readonly [number, number, number];
}

const PROBE_VALUES_WGSL = /* wgsl */ `
${DDGI_SAMPLE_WGSL}

struct ProbeValueParameters {
    volumeOrigin: vec4f,
    volumeSpacing: vec4f,
    volumeCounts: vec4f,
    volumeTexels: vec4f,
}
@group(1) @binding(0) var<uniform> probeField: ProbeValueParameters;
@group(1) @binding(1) var probeIrradiance: texture_2d_array<f32>;
@group(1) @binding(2) var probeDistance: texture_2d_array<f32>;
@group(1) @binding(3) var probeDataAtlas: texture_2d_array<f32>;
@group(1) @binding(4) var probeSampler: sampler;
// Per query: xyz the direction, w the probe index (exact in f32 below 2^24).
@group(2) @binding(0) var<storage, read> probeQueries: array<vec4f>;
@group(2) @binding(1) var<storage, read_write> probeResults: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&probeQueries)) {
        return;
    }
    let query = probeQueries[id.x];
    let volume = ddgiVolume(probeField.volumeOrigin, probeField.volumeSpacing, probeField.volumeCounts, probeField.volumeTexels);
    let index = i32(query.w);
    let counts = volume.counts;
    let probe = vec3i(index % counts.x, index / (counts.x * counts.z), (index / counts.x) % counts.z);
    let scale = ddgiAtlasScale(volume, volume.irradianceTexels);
    let uv = ddgiTileUv(ddgiTileTexel(normalize(query.xyz), volume.irradianceTexels), probe, scale);
    let irradiance = textureSampleLevel(probeIrradiance, probeSampler, uv, ddgiAtlasLayer(volume, probe), 0.0).rgb;
    let data = ddgiReadProbeData(probeDataAtlas, volume, probe);
    // w: the probe's state, for checking the selection.
    probeResults[id.x] = vec4f(irradiance, f32(data.state));
}
`;

type ComputeKernel = Awaited<ReturnType<GIPlatformImpl['kernels']['computeKernel']>>;

export class ProbeValues {
    private constructor(
        private readonly device: GPUDevice,
        private readonly kernel: ComputeKernel,
    ) {}

    static async create(device: GPUDevice, platform: GIPlatformImpl): Promise<ProbeValues> {
        const kernel = await platform.kernels.computeKernel({ label: 'ddgi probe values', modules: [], code: PROBE_VALUES_WGSL, entryPoint: 'main', ownGroups: [FIELD_GROUP, IO_GROUP] });
        return new ProbeValues(device, kernel);
    }

    /** Irradiance E (rgb) each probe stores for each direction, and the probe's state. */
    async read(field: DDGIField, queries: readonly ProbeDirection[]): Promise<{ irradiance: [number, number, number]; state: number }[]> {
        const device = this.device;
        const inputs = new Float32Array(queries.length * 4);
        queries.forEach((query, index) => inputs.set([...query.direction, query.probe], index * 4));
        const inputBuffer = device.createBuffer({ size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(inputBuffer, 0, inputs);
        const outputBytes = queries.length * 16;
        const outputBuffer = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const parameters = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const encoder = device.createCommandEncoder({ label: 'ddgi probe values' });
        const pass = encoder.beginComputePass();
        this.kernel.bind(pass);
        pass.setBindGroup(1, fieldBindGroup(device, this.kernel.ownLayouts[0], field, parameters));
        pass.setBindGroup(
            2,
            device.createBindGroup({
                layout: this.kernel.ownLayouts[1],
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: outputBuffer } },
                ],
            }),
        );
        pass.dispatchWorkgroups(Math.ceil(queries.length / 64));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const values = new Float32Array(await readBuffer(device, outputBuffer, outputBytes));
        inputBuffer.destroy();
        outputBuffer.destroy();
        parameters.destroy();
        return queries.map((_, index) => ({ irradiance: [values[index * 4], values[index * 4 + 1], values[index * 4 + 2]], state: Math.round(values[index * 4 + 3]) }));
    }
}
