/**
 * T2.3 check 1: many-light sampling through the light grid is unbiased
 * (docs/design/m2-lights.md §3, "T2.3 Many-light sampling").
 *
 * The test scene: a 24 m square floor with 16 pillars (1 x 3 x 1 m) and 8
 * low blocks, lit by 1,024 point lights between 0.2 and 3.5 m up, a third of
 * them with no range, the rest ranged 3 to 10 m, all casting hard shadows
 * (radius 0, so the exhaustive answer has no noise). 1,000 test points: 700
 * seen from above (floor, pillar and block tops) and 300 on the pillars'
 * sides.
 *
 * 1. **The exhaustive answer:** `giDirectIrradiance` with `exactLightLimit`
 *    and `maxShadowedLightsPerHit` above the light count, so every light is
 *    evaluated with its shadow ray at every point.
 * 2. **Through `gi/direct` with the grid** (the defaults: grid above 8
 *    point and spot lights, shadow rays to K = 4 of a cell's 16 picks): a test kernel traces
 *    the points' rays and lists them as a technique's tracing kernel does,
 *    `gi/direct` lights them; the grid is rebuilt every frame (new picks) by
 *    the platform's frame. Averaged over FRAMES frames, the done-when:
 *    the mean over all points within 1% of the exhaustive one, and 99% of
 *    the points within 5% each.
 * 3. **Noise at K = 1, 2, 4** (recorded, not gated): each point's standard
 *    deviation over frames, relative to its exhaustive value (median, mean,
 *    p95 over points), and the same accuracy figures.
 * 4. **The two scans:** `gi/light-grid` built with the subgroup scan and with
 *    the declared fallback (workgroup memory) on the same input hand out the
 *    same picks (at least 99.9% of them; float sums added in a different
 *    order can move a pick that sits on a boundary), and every pick slot is
 *    written (a pick that counts 0 is "no light" or a repeat of the light
 *    before it).
 */

import type { DirectWorkList, RayScratch } from '../../../../packages/ddgi-core/src/platform/contract/shade.ts';
import type { GpuTimerScope } from '../../../../packages/ddgi-core/src/platform/contract/timing.ts';
import type { ComputeKernel } from '../../../../packages/ddgi-core/src/platform/contract/wgsl.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { RAY_RESULT_BYTES } from '../../../../packages/ddgi-core/src/platform/shade/direct.ts';
import {
    LIGHT_GRID_NO_LIGHT,
    LIGHT_GRID_PICKS_PER_CELL,
    LIGHT_GRID_TEXTURE_WIDTH,
    PlatformLightGrid,
    type LightGridReduction,
} from '../../../../packages/ddgi-core/src/platform/shade/light-grid.ts';
import { createRandom } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { assert, runGpuTest } from '../../client.ts';
import { boxMesh } from '../trace-oracle/cases.ts';
import { halfToFloat } from '../ddgi-shared/harness.ts';
import { frame, normalize, ready, type Vec3 } from '../hit-shading-shared/harness.ts';

const LIGHTS = 1024;
const POINTS_FROM_ABOVE = 700;
const POINTS_ON_SIDES = 300;
/** Frames averaged (the done-when: 1,024). */
const FRAMES = 1024;
/** The done-when's bars. */
const MAX_MEAN_DIFFERENCE = 0.01;
const POINT_TOLERANCE = 0.05;
const POINTS_WITHIN_SHARE = 0.99;
/** The two scans must agree on at least this share of picks. */
const MIN_SAME_PICKS = 0.999;
const WORKGROUP_SIZE = 64;
const ALL_GROUPS = 0xf;
const NO_TIMER: GpuTimerScope = { computePass: () => undefined };

const RAYS_GROUP: GPUBindGroupLayoutDescriptor = {
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
};

/** A technique's tracing side for the test points: trace, ask giNeedsDirect, list the hit. */
function recordKernelWgsl(scratchWgsl: string): string {
    return /* wgsl */ `
${scratchWgsl}

struct TestFrame {
    frame: u32,
    groupMask: u32,
    reserved0: u32,
    reserved1: u32,
}

@group(1) @binding(0) var<storage, read> testRays: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> testScratch: GiRayScratch;
@group(1) @binding(2) var<uniform> testFrame: TestFrame;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
    let index = group.x * ${WORKGROUP_SIZE}u + thread;
    var deferred = false;
    var record = GiSurfaceRecord();
    if (index < arrayLength(&testRays) / 2u) {
        let ray = GiRay(testRays[2u * index].xyz, testRays[2u * index].w, testRays[2u * index + 1u].xyz, testRays[2u * index + 1u].w);
        let hit = giTraceClosest(ray);
        if (hit.t >= 0.0) {
            let surface = giSurfaceAt(ray, hit);
            deferred = giNeedsDirect(surface);
            if (deferred) {
                record = giSurfaceRecord(surface);
            }
        }
        giStoreRayResult(index, vec3f(0.0), 0.0);
    }
    giDeferDirect(deferred, record, index);
}
`;
}

/** The exhaustive side: giDirectIrradiance inline, in f32; w is -1 for a miss. */
const INLINE_KERNEL_WGSL = /* wgsl */ `
struct TestFrame {
    frame: u32,
    groupMask: u32,
    reserved0: u32,
    reserved1: u32,
}

@group(1) @binding(0) var<storage, read> testRays: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> testOutputs: array<vec4f>;
@group(1) @binding(2) var<uniform> testFrame: TestFrame;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
    let index = group.x * ${WORKGROUP_SIZE}u + thread;
    if (index >= arrayLength(&testRays) / 2u) {
        return;
    }
    let ray = GiRay(testRays[2u * index].xyz, testRays[2u * index].w, testRays[2u * index + 1u].xyz, testRays[2u * index + 1u].w);
    let hit = giTraceClosest(ray);
    if (hit.t < 0.0) {
        testOutputs[index] = vec4f(0.0, 0.0, 0.0, -1.0);
        return;
    }
    let surface = giSurfaceAt(ray, hit);
    testOutputs[index] = vec4f(giDirectIrradiance(surface, testFrame.groupMask, giDirectRandom(index, testFrame.frame)), 1.0);
}
`;

async function readBack(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, offset, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

async function readTexture(device: GPUDevice, texture: GPUTexture): Promise<Uint32Array> {
    const bytesPerRow = texture.width * 8;
    const buffer = device.createBuffer({ size: bytesPerRow * texture.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width: texture.width, height: texture.height });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    buffer.destroy();
    return words;
}

const luminance = (r: number, g: number, b: number): number => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** The test points' rays and the two kernels over them. */
class PointBench {
    private readonly rays: GPUBuffer;
    private readonly scratch: RayScratch;
    private readonly outputs: GPUBuffer;
    private readonly frameUniform: GPUBuffer;

    private constructor(
        private readonly device: GPUDevice,
        private readonly platform: GIPlatformImpl,
        private readonly recordKernel: ComputeKernel,
        private readonly inlineKernel: ComputeKernel,
        readonly pointCount: number,
        packedRays: Float32Array,
    ) {
        this.rays = device.createBuffer({ size: packedRays.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.rays, 0, packedRays as Float32Array<ArrayBuffer>);
        this.scratch = platform.shade.direct.createScratch({ label: 'many lights scratch', rays: pointCount, workItems: pointCount });
        this.outputs = device.createBuffer({ size: pointCount * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.frameUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    static async create(device: GPUDevice, platform: GIPlatformImpl, packedRays: Float32Array): Promise<PointBench> {
        await platform.shade.direct.prepare();
        const workList: DirectWorkList = platform.shade.direct.defaultWorkList;
        const recordKernel = await platform.kernels.computeKernel({
            label: 'many lights records',
            modules: [platform.shade.module],
            code: recordKernelWgsl(platform.shade.direct.traceWgsl('testScratch', workList)),
            entryPoint: 'main',
            ownGroups: [RAYS_GROUP],
        });
        const inlineKernel = await platform.kernels.computeKernel({ label: 'many lights inline', modules: [platform.shade.module], code: INLINE_KERNEL_WGSL, entryPoint: 'main', ownGroups: [RAYS_GROUP] });
        return new PointBench(device, platform, recordKernel, inlineKernel, packedRays.length / 8, packedRays);
    }

    /** Direct irradiance at each point through gi/direct (luminance; NaN where the ray missed is not tracked here). */
    async deferred(frameNumber: number): Promise<Float64Array> {
        const device = this.device;
        device.queue.writeBuffer(this.frameUniform, 0, new Uint32Array([frameNumber, ALL_GROUPS, 0, 0]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        this.recordKernel.bind(pass);
        pass.setBindGroup(1, this.group(this.recordKernel, this.scratch.buffer));
        pass.dispatchWorkgroups(Math.ceil(this.pointCount / WORKGROUP_SIZE));
        pass.end();
        this.platform.shade.direct.encode(encoder, this.scratch, { frame: frameNumber, groupMask: ALL_GROUPS, owner: 'test', timer: NO_TIMER });
        device.queue.submit([encoder.finish()]);
        const results = this.scratch.rayResults;
        const words = new Uint32Array(await readBack(device, results.buffer, results.offset ?? 0, this.pointCount * RAY_RESULT_BYTES));
        const values = new Float64Array(this.pointCount);
        for (let point = 0; point < this.pointCount; point++) {
            // White materials: π × the result is the direct irradiance.
            const r = halfToFloat(words[point * 2] & 0xffff);
            const g = halfToFloat(words[point * 2] >>> 16);
            const b = halfToFloat(words[point * 2 + 1] & 0xffff);
            values[point] = Math.PI * luminance(r, g, b);
        }
        return values;
    }

    /** giDirectIrradiance inline (luminance), and which points hit. */
    async inline(frameNumber: number): Promise<{ values: Float64Array; hit: boolean[] }> {
        const device = this.device;
        device.queue.writeBuffer(this.frameUniform, 0, new Uint32Array([frameNumber, ALL_GROUPS, 0, 0]));
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        this.inlineKernel.bind(pass);
        pass.setBindGroup(1, this.group(this.inlineKernel, this.outputs));
        pass.dispatchWorkgroups(Math.ceil(this.pointCount / WORKGROUP_SIZE));
        pass.end();
        device.queue.submit([encoder.finish()]);
        const out = new Float32Array(await readBack(device, this.outputs, 0, this.pointCount * 16));
        const values = new Float64Array(this.pointCount);
        const hit: boolean[] = [];
        for (let point = 0; point < this.pointCount; point++) {
            values[point] = luminance(out[point * 4], out[point * 4 + 1], out[point * 4 + 2]);
            hit.push(out[point * 4 + 3] >= 0);
        }
        return { values, hit };
    }

    private group(kernel: ComputeKernel, second: GPUBuffer): GPUBindGroup {
        return this.device.createBindGroup({
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.rays } },
                { binding: 1, resource: { buffer: second } },
                { binding: 2, resource: { buffer: this.frameUniform } },
            ],
        });
    }

    destroy(): void {
        this.rays.destroy();
        this.scratch.destroy();
        this.outputs.destroy();
        this.frameUniform.destroy();
    }
}

const PILLARS: Vec3[] = [];
for (let x = -9; x <= 9; x += 6) for (let z = -9; z <= 9; z += 6) PILLARS.push([x, 1.5, z]);
const BLOCKS: Vec3[] = [
    [-6, 0.25, 0],
    [6, 0.25, 0],
    [0, 0.25, -6],
    [0, 0.25, 6],
    [-3, 0.25, -3],
    [3, 0.25, 3],
    [-3, 0.25, 3],
    [3, 0.25, -3],
];

function insidePillar(point: Vec3, margin: number): boolean {
    return PILLARS.some((centre) => Math.abs(point[0] - centre[0]) < 0.5 + margin && Math.abs(point[2] - centre[2]) < 0.5 + margin && point[1] < 3 + margin);
}

function buildScene(platform: GIPlatformImpl): void {
    const scene = platform.scene;
    const white = scene.createMaterial({ albedo: [1, 1, 1] });
    const h = 12;
    const ground = scene.createMesh({
        positions: new Float32Array([-h, 0, -h, -h, 0, h, h, 0, h, h, 0, -h]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
    });
    scene.createInstance({ mesh: ground, materials: [white], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
    const pillar = boxMesh([0.5, 1.5, 0.5]);
    const pillarMesh = scene.createMesh({ positions: pillar.positions, indices: pillar.indices as Uint32Array });
    for (const [x, y, z] of PILLARS) scene.createInstance({ mesh: pillarMesh, materials: [white], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1] });
    const block = boxMesh([0.75, 0.25, 0.75]);
    const blockMesh = scene.createMesh({ positions: block.positions, indices: block.indices as Uint32Array });
    for (const [x, y, z] of BLOCKS) scene.createInstance({ mesh: blockMesh, materials: [white], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1] });

    const random = createRandom(2303);
    for (let index = 0; index < LIGHTS; index++) {
        let position: Vec3;
        do position = [random() * 23 - 11.5, 0.2 + random() * 3.3, random() * 23 - 11.5];
        while (insidePillar(position, 0.05));
        scene.createLight({
            type: 'point',
            color: [1, 0.5 + random() * 0.4, 0.2 + random() * 0.4],
            intensity: 0.5 + random() * 1.5,
            position,
            range: index % 3 === 0 ? 0 : 3 + random() * 7,
            castsShadows: true,
        });
    }
}

function testRays(): Float32Array {
    const random = createRandom(77);
    const rays: Array<{ origin: Vec3; direction: Vec3 }> = [];
    for (let index = 0; index < POINTS_FROM_ABOVE; index++) rays.push({ origin: [random() * 23 - 11.5, 10, random() * 23 - 11.5], direction: [0, -1, 0] });
    for (let index = 0; index < POINTS_ON_SIDES; index++) {
        const pillar = PILLARS[index % PILLARS.length];
        const angle = random() * 2 * Math.PI;
        const target: Vec3 = [pillar[0] + (random() - 0.5) * 0.9, 0.05 + random() * 2.9, pillar[2] + (random() - 0.5) * 0.9];
        const origin: Vec3 = [pillar[0] + Math.cos(angle) * 2.5, target[1], pillar[2] + Math.sin(angle) * 2.5];
        rays.push({ origin, direction: normalize([target[0] - origin[0], 0, target[2] - origin[2]]) });
    }
    const packed = new Float32Array(rays.length * 8);
    rays.forEach((ray, index) => {
        packed.set([...ray.origin, 0], index * 8);
        packed.set([...normalize(ray.direction), 1e6], index * 8 + 4);
    });
    return packed;
}

interface Accuracy {
    meanDifference: number;
    withinTolerance: number;
    p99Error: number;
    worstError: number;
    noiseMedian: number;
    noiseMean: number;
    noiseP95: number;
    points: number;
    /** The points furthest off: their exhaustive value, the average's error, the noise a frame and the error in standard errors (bias or noise). */
    worstPoints: string;
}

function quantile(sorted: number[], share: number): number {
    return sorted.length === 0 ? 0 : sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
}

async function sample(device: GPUDevice, platform: GIPlatformImpl, bench: PointBench, reference: Float64Array, lit: boolean[], frameBase: number): Promise<Accuracy> {
    const sums = new Float64Array(bench.pointCount);
    const squares = new Float64Array(bench.pointCount);
    for (let index = 0; index < FRAMES; index++) {
        // A platform frame rebuilds the grid with new picks.
        frame(device, platform);
        const values = await bench.deferred(frameBase + index);
        for (let point = 0; point < bench.pointCount; point++) {
            sums[point] += values[point];
            squares[point] += values[point] * values[point];
        }
    }
    let estimateTotal = 0;
    let referenceTotal = 0;
    const errors: number[] = [];
    const noise: number[] = [];
    const byPoint: Array<{ point: number; error: number; noise: number }> = [];
    for (let point = 0; point < bench.pointCount; point++) {
        if (!lit[point]) continue;
        const mean = sums[point] / FRAMES;
        const variance = Math.max(0, squares[point] / FRAMES - mean * mean);
        estimateTotal += mean;
        referenceTotal += reference[point];
        errors.push(Math.abs(mean / reference[point] - 1));
        noise.push(Math.sqrt(variance) / reference[point]);
        byPoint.push({ point, error: mean / reference[point] - 1, noise: Math.sqrt(variance) / reference[point] });
    }
    byPoint.sort((a, b) => Math.abs(b.error) - Math.abs(a.error));
    const worstPoints = byPoint
        .slice(0, 6)
        .map(({ point, error, noise: pointNoise }) => `#${point} E ${reference[point].toPrecision(3)} err ${(error * 100).toFixed(1)}% noise ${(pointNoise * 100).toFixed(0)}% (${(error / (pointNoise / Math.sqrt(FRAMES))).toFixed(1)} s.e.)`)
        .join('; ');
    errors.sort((a, b) => a - b);
    noise.sort((a, b) => a - b);
    return {
        meanDifference: estimateTotal / referenceTotal - 1,
        withinTolerance: errors.filter((error) => error <= POINT_TOLERANCE).length / errors.length,
        p99Error: quantile(errors, 0.99),
        worstError: quantile(errors, 1),
        noiseMedian: quantile(noise, 0.5),
        noiseMean: noise.reduce((sum, value) => sum + value, 0) / noise.length,
        noiseP95: quantile(noise, 0.95),
        points: errors.length,
        worstPoints,
    };
}

function describe(accuracy: Accuracy): string {
    const percent = (value: number, digits = 2) => `${(value * 100).toFixed(digits)}%`;
    return (
        `mean ${percent(accuracy.meanDifference, 3)} from exhaustive; ${percent(accuracy.withinTolerance, 1)} of ${accuracy.points} points within ${percent(POINT_TOLERANCE, 0)} ` +
        `(p99 ${percent(accuracy.p99Error)}, worst ${percent(accuracy.worstError)}); noise a frame median ${percent(accuracy.noiseMedian, 0)}, mean ${percent(accuracy.noiseMean, 0)}, p95 ${percent(accuracy.noiseP95, 0)}`
    );
}

/** Builds one standalone grid per scan on the platform's scene, with the same input, and compares their picks. */
async function compareScans(device: GPUDevice, platform: GIPlatformImpl, log: (line: string) => void): Promise<Record<string, unknown>> {
    const reductions: LightGridReduction[] = device.features.has('subgroups') ? ['subgroups', 'workgroup'] : ['workgroup'];
    const picks = new Map<LightGridReduction, Uint32Array>();
    for (const reduction of reductions) {
        const grid = new PlatformLightGrid(device, platform.kernels, platform.sceneCopy.module, platform.shade.lightFalloff, { label: `many lights ${reduction}`, reduction });
        await grid.prepare();
        const encoder = device.createCommandEncoder();
        const lookup = grid.encode(encoder, { frame: 12345, bounds: platform.scene.bounds, localLights: LIGHTS, exactLightLimit: 8, maxCells: 4096 });
        device.queue.submit([encoder.finish()]);
        assert(lookup.active, `the ${reduction} grid did not build`);
        const words = await readTexture(device, grid.texture);
        const cells = lookup.layout.cellCount + 1;
        // Every slot written: a pick that counts 0 is either "no light" or a repeat of the light before it
        // (a light picked m times counts m / (R p) on its first pick and 0 on the rest).
        let unwritten = 0;
        const shares = new Float32Array(words.buffer);
        for (let slot = 0; slot < cells * LIGHT_GRID_PICKS_PER_CELL; slot++) {
            const light = words[slot * 2];
            const firstOfCell = slot % LIGHT_GRID_PICKS_PER_CELL === 0;
            const repeat = !firstOfCell && words[(slot - 1) * 2] === light;
            if (light !== LIGHT_GRID_NO_LIGHT && !(shares[slot * 2 + 1] > 0) && !repeat) unwritten++;
        }
        let emptyCells = 0;
        for (let cell = 0; cell < cells; cell++) if (words[cell * LIGHT_GRID_PICKS_PER_CELL * 2] === LIGHT_GRID_NO_LIGHT) emptyCells++;
        assert(unwritten === 0, `${reduction} scan: ${unwritten} pick slots left unwritten`);
        // With 1,024 unranged and ranged lights over a 24 m floor, every cell has some light.
        assert(emptyCells === 0, `${reduction} scan: ${emptyCells} of ${cells} cells have no light`);
        picks.set(reduction, words.slice(0, cells * LIGHT_GRID_PICKS_PER_CELL * 2));
        grid.dispose();
    }
    const result: Record<string, unknown> = { reductions };
    if (reductions.length === 2) {
        const [first, second] = [picks.get('subgroups')!, picks.get('workgroup')!];
        const slots = first.length / 2;
        let same = 0;
        for (let slot = 0; slot < slots; slot++) if (first[slot * 2] === second[slot * 2]) same++;
        const share = same / slots;
        result.samePicks = share;
        result.picks = slots;
        log(`the subgroup and workgroup scans hand out the same light in ${(share * 100).toFixed(3)}% of ${slots} picks (texture ${LIGHT_GRID_TEXTURE_WIDTH} wide)`);
        assert(share >= MIN_SAME_PICKS, `the two scans agree on only ${(share * 100).toFixed(3)}% of picks`);
    } else {
        log('no subgroups on this device: only the workgroup scan ran');
    }
    return result;
}

await runGpuTest(
    async ({ device, log }) => {
        const results: Record<string, unknown> = {};
        const platform = new GIPlatformImpl(device, { label: 'many lights' });
        try {
            buildScene(platform);
            await ready(device, platform);
            const bench = await PointBench.create(device, platform, testRays());
            // The two scans first: they read the grid's picks back.
            results.scans = await compareScans(device, platform, log);

            // 1. The exhaustive answer: every light, every point, with its shadow ray.
            platform.shade.settings.exactLightLimit = Infinity;
            platform.shade.settings.maxShadowedLightsPerHit = 4096;
            frame(device, platform);
            const exhaustive = await bench.inline(1);
            const lit = Array.from(exhaustive.values, (value, point) => exhaustive.hit[point] && value > 0);
            const litPoints = lit.filter(Boolean).length;
            results.points = { tested: bench.pointCount, lit: litPoints };
            log(`${bench.pointCount} points, ${litPoints} lit by the exhaustive evaluation of ${LIGHTS} point lights`);
            assert(litPoints >= 0.95 * bench.pointCount, `only ${litPoints} points are lit`);

            // 2 and 3. Through gi/direct with the grid, at K = 4 (the default; gated), 2 and 1.
            platform.shade.settings.exactLightLimit = 8;
            const byK: Record<string, Accuracy> = {};
            for (const k of [4, 2, 1]) {
                platform.shade.settings.maxShadowedLightsPerHit = k;
                const accuracy = await sample(device, platform, bench, exhaustive.values, lit, 1000 * k);
                byK[`K${k}`] = accuracy;
                log(`K = ${k}, ${FRAMES} frames: ${describe(accuracy)}`);
            }
            results.accuracy = byK;
            const gated = byK.K4;
            assert(Math.abs(gated.meanDifference) <= MAX_MEAN_DIFFERENCE, `K = 4: the mean is ${(gated.meanDifference * 100).toFixed(3)}% from exhaustive (bar ${MAX_MEAN_DIFFERENCE * 100}%)`);
            assert(gated.withinTolerance >= POINTS_WITHIN_SHARE, `K = 4: only ${(gated.withinTolerance * 100).toFixed(1)}% of points within ${POINT_TOLERANCE * 100}% (bar ${POINTS_WITHIN_SHARE * 100}%). ${Object.entries(byK).map(([k, accuracy]) => `${k}: ${describe(accuracy)}; worst ${accuracy.worstPoints}`).join(' | ')}`);

            bench.destroy();
        } finally {
            platform.dispose();
        }
        return results;
    },
    { optionalFeatures: ['subgroups'] },
);
