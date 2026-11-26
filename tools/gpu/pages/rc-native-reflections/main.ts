import { NativeRadianceCascades, type NativeRcTransport } from '../../../../packages/ddgi-core/src/techniques/radiance-cascades/native.ts';
import { halfToFloat } from '../../../../packages/ddgi-core/src/platform/sky/half-float.ts';
import { assert, runGpuTest } from '../../client.ts';

const width = 32, height = 16;
const origin = [0.12, 0.04, 0.02];
const roughnesses = [0, 0.55, 1];
const columns = [4, 16, 27];
const boundaryDirection = [1, 0.55, 0.4998];
const boundaryOffset = 4 * 0.0004 / Math.hypot(...boundaryDirection);
type Color = [number, number, number, number];
const fieldCode = /* wgsl */ `
struct Ray { origin: vec4f, direction: vec4f }
@group(0) @binding(0) var<storage,read> rays: array<Ray>;
@group(0) @binding(1) var<storage,read_write> hits: array<vec4f>;
@group(0) @binding(2) var<uniform> mode: vec4u;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= arrayLength(&hits)) { return; }
    let d = rays[id.x].direction.xyz;
    var result = vec4f(0.4, 0.2, 0.1, -1); // radiance furnace
    if (mode.x == 1u) { result = select(vec4f(0,0,1,0.25), vec4f(1,0,0,0.25), d.x > 0); }
    if (mode.x == 2u) { result = select(vec4f(0,0,1,3), vec4f(1,0,0,3), d.z > 0.5 * d.x); }
    if (mode.x == 3u) { result = vec4f(0.23,0.1,0.38,0.25); }
    hits[id.x] = result;
}`;

/** Independent CPU quadrature on the 24 near bin centers for the split red/blue field. */
function expectedNearRed(reflectedX: number, roughness: number) {
    let red = 0, total = 0;
    for (let face = 0; face < 6; face++) {
        const sign = face % 2 === 0 ? 1 : -1;
        for (const u of [-0.5, 0.5]) for (const v of [-0.5, 0.5]) {
            const local = face < 2 ? [sign,u,v] : face < 4 ? [u,sign,v] : [u,v,sign];
            const [x, y] = local.map(c => c / Math.sqrt(1.5));
            if (y <= 0) continue;
            const cosine = reflectedX * x + 0.5 * y;
            const weight = y * 2 ** (6 * (1 - roughness ** 2) * (cosine - 1));
            total += weight;
            if (x > 0) red += weight;
        }
    }
    return red / total;
}

await runGpuTest(async ({ device }) => {
    const solver = await NativeRadianceCascades.create(device, {
        bounds: { min: [-1,-1,-1], max: [1,1,1] }, grid: [1,1,1],
        firstInterval: 0.5, maxDistance: 8, rayStride: 1, history: 0, roughSpecular: true,
    });
    solver.resize(width, height);
    const makeInput = (normal: boolean) => {
        const texture = device.createTexture({ size: [width,height], format: 'rgba32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        const pixels = new Float32Array(width * height * 4);
        for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
            const material = x < 11 ? roughnesses[0] : x < 22 ? roughnesses[1] : roughnesses[2];
            const positionZ = origin[2] + (x === 5 ? boundaryOffset : 0);
            pixels.set(normal ? [0,1,0,material] : [origin[0],origin[1],positionZ,x === 0 ? 0 : 1], (y * width + x) * 4);
        }
        device.queue.writeTexture({ texture }, pixels, { bytesPerRow: width * 16 }, [width,height]);
        return texture;
    };
    const position = makeInput(false), normal = makeInput(true);
    const screen = { worldPosition: position.createView(), worldNormal: normal.createView() };
    const mode = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const view = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const external = device.createTexture({ size: [width,height], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const outputView = external.createView();
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code: fieldCode }) } });
    let batches = 0, frames = 0;
    const transport: NativeRcTransport = {
        beginFrame() {},
        encode(encoder, { rays, results, count }) {
            batches++;
            const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
                { binding: 0, resource: { buffer: rays } }, { binding: 1, resource: { buffer: results } }, { binding: 2, resource: { buffer: mode } },
            ] });
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(Math.ceil(count / 64));
            pass.end();
        },
    };
    async function samples(texture: GPUTexture, readColumns = columns) {
        // A 32-texel rgba16float row meets the 256-byte copy alignment exactly.
        const read = device.createBuffer({ size: width * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
            const encoder = device.createCommandEncoder();
            encoder.copyTextureToBuffer({ texture, origin: { x: 0, y: 5 } }, { buffer: read, bytesPerRow: width * 8 }, [width,1]);
            device.queue.submit([encoder.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            const words = new Uint16Array(read.getMappedRange());
            assert(words.slice(0, 4).every(value => value === 0), 'invalid G-buffer pixel must have zero light and support');
            return readColumns.map(x => Array.from(words.slice(x * 4, x * 4 + 4), halfToFloat) as Color);
        } finally { read.destroy(); }
    }
    function warm(field: number) {
        device.queue.writeBuffer(mode, 0, new Uint32Array([field,0,0,0]));
        const encoder = device.createCommandEncoder();
        for (let frame = 0; frame < 40; frame++) solver.encode(encoder, screen, transport);
        frames += 40;
        device.queue.submit([encoder.finish()]);
    }
    async function query(reflected: number[], perspective = false, owned = false, readColumns = columns) {
        // On a horizontal plane, the camera direction producing this reflected
        // vector is (-x,+y,-z). Use nonzero world positions to test the subtraction.
        const len = Math.hypot(...reflected);
        const v = reflected.map((x, i) => x * (i === 1 ? 1 : -1) / len);
        const xyz = perspective ? v.map((x,i) => origin[i] + 4 * x) : v;
        device.queue.writeBuffer(view, 0, new Float32Array([...xyz, perspective ? 1 : 0]));
        const encoder = device.createCommandEncoder();
        solver.encode(encoder, { ...screen, specular: { view, outputView: owned ? undefined : outputView } }, transport);
        frames++;
        device.queue.submit([encoder.finish()]);
        return samples(owned ? solver.specularTexture : external, readColumns);
    }
    const near = (actual: number, expected: number, tolerance: number, label: string) =>
        assert(Math.abs(actual - expected) <= tolerance, `${label}: got ${actual}, expected ${expected} ± ${tolerance}`);
    try {
        warm(0);
        const diffuseBefore = await samples(solver.outputTexture);
        assert(solver.ownedSpecularBytes === 0, 'skipped reflections allocated a target');
        const furnace = await query([1,1,0]);
        assert(solver.ownedSpecularBytes === 0, 'external reflections allocated a duplicate target');
        for (const c of furnace) {
            [0.4,0.2,0.1,1].forEach((expected, i) => near(c[i], expected, 0.003, 'constant radiance or support'));
        }
        const diffuseAfter = await samples(solver.outputTexture);
        assert(JSON.stringify(diffuseBefore) === JSON.stringify(diffuseAfter), 'specular query changed diffuse output');
        const backside = await query([0,-1,0]);
        assert(backside.every(color => color.every(value => value === 0)), 'back-facing surface returned specular radiance');

        warm(1);
        const rx = Math.sqrt(3) / 2;
        const right = await query([rx,0.5,0]);
        const left = await query([-rx,0.5,0], true);
        for (let i = 0; i < roughnesses.length; i++) {
            near(right[i][0], expectedNearRed(rx, roughnesses[i]), 0.008, 'right-facing near lobe');
            near(left[i][0], expectedNearRed(-rx, roughnesses[i]), 0.008, 'left-facing near lobe');
        }

        solver.reset();
        warm(2);
        // Both directions address the same +X/y+/z+ near bin, on opposite
        // sides of a distant red/blue boundary at z/x = 0.5.
        const lowZ = await query([1,0.55,0.1]);
        const highZ = await query([1,0.55,0.92]);
        assert(highZ[0][0] - lowZ[0][0] > 0.45, 'fine far directional contrast was lost');
        // Sweep through an outer-bin boundary. Its center and finite cone
        // should move continuously while the two flanking views stay distinct.
        // Two positions under one perspective camera share an identical
        // interval field while their reflected z/x straddles 0.5 by ±0.0002.
        const [beforeBoundary, afterBoundary] = await query(boundaryDirection, true, false, [4,5]);
        const boundaryJump = Math.abs(beforeBoundary[0] - afterBoundary[0]);
        near(boundaryJump, 0, 0.015, 'polished query jump across a far cube bin');
        near(lowZ[2][0], highZ[2][0], 0.02, 'maximum roughness view independence');
        // Retain the previous far radiance while replacing every near bin with an opaque neutral hit.
        warm(3);
        const blockedLow = await query([1,0.55,0.1], false, true);
        const blockedHigh = await query([1,0.55,0.92], false, true);
        for (const c of [...blockedLow, ...blockedHigh]) {
            [0.23,0.1,0.38,1].forEach((expected,i) => near(c[i], expected, 0.004, 'opaque near interval blocks far reflection'));
        }
        assert(solver.ownedSpecularBytes === width * height * 8, 'owned reflection target size differs');
        solver.resize(17,13);
        assert(Number(solver.ownedSpecularBytes) === 0, 'resize kept the obsolete optional output');
        assert(batches === frames, 'reflection query dispatched additional rays');
        return { furnace, nearRoughness: roughnesses, nearRightRed: right.map(v => v[0]), nearLeftRed: left.map(v => v[0]),
            farLowRed: lowZ.map(v => v[0]), farHighRed: highZ.map(v => v[0]), boundaryJump, blocked: blockedHigh[0], frames, transportBatches: batches };
    } finally {
        mode.destroy(); view.destroy(); external.destroy(); position.destroy(); normal.destroy(); solver.dispose();
    }
});
