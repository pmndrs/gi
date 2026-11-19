import { NativeRadianceCascades, type NativeRcTransport } from '../../../../packages/ddgi-core/src/techniques/radiance-cascades/native.ts';
import { halfToFloat } from '../../../../packages/ddgi-core/src/platform/sky/half-float.ts';
import { assert, runGpuTest } from '../../client.ts';

const fixtureCode = /* wgsl */ `
    struct Ray { a:vec4f, b:vec4f }
    @group(0) @binding(0) var<storage,read> rays:array<Ray>;
    @group(0) @binding(1) var<storage,read_write> hits:array<vec4f>;
    @group(0) @binding(2) var<uniform> mode:vec4u;
    @compute @workgroup_size(64) fn trace(@builtin(global_invocation_id) id:vec3u) {
        if (id.x >= arrayLength(&hits)) { return; }
        // Uniform sky, opaque near surface, then split near red / distant blue hemispheres.
        var hit = vec4f(0.4,0.2,0.1,-1);
        if (mode.x == 1u) { hit = vec4f(0.05,0.4,0.1,0.25); }
        if (mode.x == 2u) {
            hit = select(vec4f(0,0,0.4,3), vec4f(0.6,0,0,0.25), rays[id.x].b.x > 0);
        }
        // At the exact near/middle boundary exactly one interval owns this hit.
        if (mode.x == 3u) { hit = vec4f(0.1,0.6,0.3,0.5); }
        hits[id.x] = hit;
    }`;

await runGpuTest(async ({ device }) => {
    const solver = await NativeRadianceCascades.create(device, {
        bounds: { min: [-1,-1,-1], max: [1,1,1] }, grid: [3, 2, 3],
        firstInterval: 0.5, maxDistance: 8, rayStride: 2, history: 0,
    });
    const pipeline = await device.createComputePipelineAsync({
        layout: 'auto', compute: { module: device.createShaderModule({ code: fixtureCode }), entryPoint: 'trace' },
    });
    const mode = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const transport: NativeRcTransport = {
        beginFrame() {},
        encode(encoder, { rays, results, count }) {
            const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
                { binding: 0, resource: { buffer: rays } },
                { binding: 1, resource: { buffer: results } },
                { binding: 2, resource: { buffer: mode } },
            ] });
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(Math.ceil(count / 64));
            pass.end();
        },
    };
    async function check(width: number, height: number, caseId: number, expected: number[], external = false) {
        solver.resize(width, height);
        device.queue.writeBuffer(mode, 0, new Uint32Array([caseId, 0, 0, 0]));
        const makeInput = (positions: boolean) => {
            const texture = device.createTexture({
                size: [width, height], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            const pixels = new Float32Array(width * height * 4);
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const index = (x + y * width) * 4;
                    pixels.set(positions ? [(x / width - 0.5) * 1.6, 0, (y / height - 0.5) * 1.6, x === 0 ? 0 : 1] : [0, 1, 0, 0], index);
                }
            }
            device.queue.writeTexture({ texture }, pixels, { bytesPerRow: width * 16 }, [width, height]);
            return texture;
        };
        const position = makeInput(true);
        const normal = makeInput(false);
        const screen = { worldPosition: position.createView(), worldNormal: normal.createView() };
        const target = external ? device.createTexture({
            size: [width, height], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
        }) : solver.outputTexture;
        const expectedOwnedBytes = external ? 0 : width * height * 8;
        assert(solver.ownedOutputBytes === expectedOwnedBytes, 'unexpected solver-owned output allocation after resize');
        const targetView = target.createView();
        const bytesPerRow = Math.ceil(width * 8 / 256) * 256;
        const read = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        try {
            const encoder = device.createCommandEncoder();
            // Multiple frames in the same submission exercise deterministic GPU-side frame sequencing.
            for (let i = 0; i < 24; i++) solver.encode(encoder, screen, transport, external ? targetView : undefined);
            assert(solver.ownedOutputBytes === expectedOwnedBytes, 'encoding to the supplied view allocated a second output');
            encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow }, [width, height]);
            device.queue.submit([encoder.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            const values = new Uint16Array(read.getMappedRange());
            const mean = [0, 0, 0];
            let valid = 0;
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const index = y * bytesPerRow / 2 + 4 * x;
                    if (x === 0) {
                        assert(values[index + 3] === 0, 'invalid position produced irradiance');
                        continue;
                    }
                    assert(halfToFloat(values[index + 3]) === 1, 'unlit/unallocated valid surface');
                    for (let c = 0; c < 3; c++) mean[c] += halfToFloat(values[index + c]) / Math.PI;
                    valid++;
                }
            }
            for (let c = 0; c < 3; c++) {
                mean[c] /= valid;
                assert(Math.abs(mean[c] - expected[c]) < 0.025, `case ${caseId}, color ${c}: E/pi ${mean[c]} expected ${expected[c]}`);
            }
            return mean;
        } finally {
            read.destroy();
            position.destroy();
            normal.destroy();
            if (external) target.destroy();
        }
    }
    try {
        const sky = await check(48, 24, 0, [0.4, 0.2, 0.1]);
        const blocked = await check(48, 24, 1, [0.05, 0.4, 0.1]);
        solver.reset();
        const split = await check(40, 21, 2, [0.3, 0, 0.2], true);
        const boundary = await check(40, 21, 3, [0.1, 0.6, 0.3]);
        return { sky, blocked, split, boundary, units: 'linear E/pi', probeBytes: solver.probeBytes, rayScratchBytes: solver.rayScratchBytes };
    } finally {
        solver.dispose();
        mode.destroy();
    }
});
