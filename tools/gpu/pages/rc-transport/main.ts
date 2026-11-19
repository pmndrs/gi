import { createRcTransport } from '../../../../packages/ddgi-core/src/techniques/radiance-cascades/transport.ts';
import { assert, runGpuTest } from '../../client.ts';

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function triangle(x: number, z: number, width: number) {
    return { positions: Float32Array.of(x - width, -width, z, x, width, z, x + width, -width, z) };
}

await runGpuTest(async ({ device }) => {
    // The light-to-receiver segment crosses the blue shutter exactly halfway.
    // The first ray sees the receiver, the second sees the shutter, and the third sees sky.
    const transport = await createRcTransport(device, {
        meshes: [triangle(0, 2, 0.8), triangle(-1, 1, 0.45)],
        instances: [
            { meshIndex: 0, instanceIndex: 0, objectToWorld: identity },
            { meshIndex: 1, instanceIndex: 1, objectToWorld: identity, material: 1 },
        ],
        materials: [
            { albedo: [1, 1, 1], emission: [0, 0.25, 0] },
            { albedo: [0, 0, 0], emission: [0, 0, 1] },
        ],
        lights: [{ position: [-2, 0, 0], intensity: [32, 0, 0] }],
    });
    const rays = device.createBuffer({ size: 96, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const results = device.createBuffer({ size: 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const queryCount = 3;
    device.queue.writeBuffer(rays, 0, Float32Array.from([0, -1, 3].flatMap(x => [x, 0, 0, 0, 0, 0, 1, 100])));
    async function trace() {
        transport.beginFrame();
        const encoder = device.createCommandEncoder();
        transport.encode(encoder, { rays, results, count: queryCount });
        encoder.copyBufferToBuffer(results, 0, readback, 0, readback.size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const actual = Array.from(new Float32Array(readback.getMappedRange()));
        readback.unmap();
        return actual;
    }
    function expectWords(name: string, actual: number[], expected: number[]) {
        expected.forEach((value, i) => {
            assert(Math.abs(actual[i] - value) < 1e-4, name + ' word ' + i + ': expected ' + value + ', got ' + actual[i]);
        });
    }
    try {
        const closed = await trace();
        expectWords('closed', closed, [0, 0.25, 0, 2, 0, 0, 1, 1, 0, 0, 0, -1]);
        const moved = [...identity];
        moved[12] = 6;
        transport.setTransform(1, moved);
        const open = await trace();
        // Unit-albedo Lambertian receiver, red point light of intensity 32, distance²=8, cosine=1/√2.
        const diffuse = 32 / (8 * Math.sqrt(2) * Math.PI);
        expectWords('open', open, [diffuse, 0.25, 0, 2, 0, 0, 0, -1, 0, 0, 0, -1]);
        return { closed, open, expectedDiffuse: diffuse };
    } finally {
        transport.dispose();
        rays.destroy();
        results.destroy();
        readback.destroy();
    }
});
