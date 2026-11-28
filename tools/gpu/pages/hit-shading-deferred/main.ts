/**
 * T2.2 check 1: direct light through the platform's deferred pass
 * (`gi/direct`) matches `giDirectIrradiance` called inline, on the GPU.
 *
 * Each hit is lit twice from the same surface and the same random numbers
 * (`giDirectRandom(ray, frame)`, what `gi/direct` uses):
 *
 * - **deferred:** a test kernel traces the ray, asks `giNeedsDirect`, writes
 *   the hit's surface record with `giSurfaceRecord` and appends it with
 *   `giDeferDirect` (the platform's tracing-kernel side, `traceWgsl`); then
 *   `shade.direct.encode` runs `gi/direct`, which adds albedo / π × direct
 *   into the ray's result. Materials are white (albedo 1, exact in the
 *   record's half floats) and results start at 0, so π × the result is the
 *   direct irradiance;
 * - **inline:** a test kernel traces the same ray and calls
 *   `giDirectIrradiance` on `giSurfaceAt`'s surface, in f32.
 *
 * So the two differ only by what the record rounds (octahedral 16-bit normals)
 * and the half-float ray result. Cases:
 *
 * 1. **T1.4's oracle points:** every trace-oracle scene's rays (thin walls,
 *    mirrored boxes, a closed thin box, instancing, random soups, a deep
 *    mesh), under each light group alone as in `hit-shading-shadows` (a point
 *    light outside, one inside the first instance, a sun) and a fourth group,
 *    a point light of radius 0.2 m outside (soft shadows, so the shadow ray
 *    follows the random numbers). Per case and group: the sum over every hit
 *    (the mean, MAX_MEAN_DIFFERENCE), and the per-point difference (mean and
 *    p95 of |deferred − inline| / max(inline, IRRADIANCE_FLOOR) over hits
 *    either path lights), because a mean can hide errors that cancel.
 * 2. **The stochastic light choice:** 12 shadowed point lights and a sun
 *    around a box, at most 2 shadowed local lights per hit, over FRAMES
 *    frames of random numbers: the per-point means compared the same way.
 * 3. **A sun on a plane:** E cos θ at 0 to 89 degrees through `gi/direct`,
 *    within MAX_SUN_ERROR.
 * 4. **Only hits a light can reach are listed:** the work list's length
 *    (the scratch's count of hits lit) equals the hits where
 *    `giDirectReaches` is true, and is below the hit count where some face
 *    away (the ground seen from below, the oracle's back faces).
 * 5. **The work lists:** on a device with subgroups, the subgroup list and
 *    the declared fallback (`atomics`) give the same results, bit for bit.
 */

import type { DirectWorkList, RayScratch } from '../../../../packages/core/src/platform/contract/shade.ts';
import type { GpuTimerScope } from '../../../../packages/core/src/platform/contract/timing.ts';
import type { ComputeKernel } from '../../../../packages/core/src/platform/contract/wgsl.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { RAY_RESULT_BYTES, RAY_SCRATCH_HEADER_BYTES, RAY_SCRATCH_WORD } from '../../../../packages/core/src/platform/shade/direct.ts';
import { createRandom } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { assert, runGpuTest } from '../../client.ts';
import { boxMesh, buildOracleCases, FULL_SIZES, type OracleCase } from '../trace-oracle/cases.ts';
import { frame, normalize, readBuffer, ready, type Vec3 } from '../hit-shading-shared/harness.ts';
import { lightsFor, SUN_DIRECTION } from '../hit-shading-shared/oracle-lights.ts';
import { halfToFloat } from '../ddgi-shared/harness.ts';

/** The done-when's bar: the deferred mean within 0.5% of the inline one. */
const MAX_MEAN_DIFFERENCE = 0.005;
/**
 * The per-point bars (DECISIONS 2026-11-06, T2.2): the mean and p95 of the
 * per-point relative difference. The half-float result alone rounds by up to
 * 2^-12 (0.024%) a channel.
 */
const MAX_POINT_MEAN = 0.001;
const MAX_POINT_P95 = 0.002;
/** Sun on a plane: within 0.5% of E cos θ (the done-when). */
const MAX_SUN_ERROR = 0.005;
/** Below this irradiance a point is judged in absolute terms (half floats lose relative precision near 0). */
const IRRADIANCE_FLOOR = 1e-3;
/** Frames of random numbers for the stochastic light choice. */
const FRAMES = 64;
const WORKGROUP_SIZE = 64;

const NO_TIMER: GpuTimerScope = { computePass: () => undefined };

/** Rays in, one GiRay per two vec4 (origin, tMin; direction, tMax), as the oracle packs them. */
const RAYS_GROUP: GPUBindGroupLayoutDescriptor = {
    entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ],
};

/** The deferred side: what a technique's tracing kernel does, for oracle rays. */
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
    // Every thread, in uniform control flow (the subgroup list needs it).
    giDeferDirect(deferred, record, index);
}
`;
}

/** The inline side: outputs (direct irradiance, reaches, hit) per ray. */
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
    let direct = giDirectIrradiance(surface, testFrame.groupMask, giDirectRandom(index, testFrame.frame));
    // w: 1 for a hit a light of the groups reaches, 0 for one none does.
    testOutputs[index] = vec4f(direct, select(0.0, 1.0, giDirectReaches(surface, testFrame.groupMask)));
}
`;

interface Lit {
    /** Direct irradiance per ray, rgb, from each path. */
    readonly deferred: Float32Array;
    readonly inline: Float32Array;
    /** Rays that hit, and hits a light of the groups reaches (from the inline kernel). */
    readonly hits: number;
    readonly reached: number;
    /** The hits this dispatch's gi/direct ran over (the header's `listed`), and the hits dropped. */
    readonly listed: number;
    readonly dropped: number;
}

/** The kernels and buffers for one scene's rays. */
class DeferredBench {
    private readonly device: GPUDevice;
    private readonly platform: GIPlatformImpl;
    private readonly recordKernels: Map<DirectWorkList, ComputeKernel>;
    private readonly inlineKernel: ComputeKernel;
    private readonly rays: GPUBuffer;
    private readonly rayCount: number;
    private readonly scratch: RayScratch;
    private readonly outputs: GPUBuffer;
    private readonly frameUniform: GPUBuffer;
    /** The header's per-frame dropped count at the last read (calls here reuse frame numbers). */
    private droppedSoFar = 0;

    private constructor(device: GPUDevice, platform: GIPlatformImpl, recordKernels: Map<DirectWorkList, ComputeKernel>, inlineKernel: ComputeKernel, rays: Float32Array) {
        this.device = device;
        this.platform = platform;
        this.recordKernels = recordKernels;
        this.inlineKernel = inlineKernel;
        this.rayCount = rays.length / 8;
        this.rays = device.createBuffer({ label: 'deferred test rays', size: rays.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.rays, 0, rays as Float32Array<ArrayBuffer>);
        // Room for every ray to be listed.
        this.scratch = platform.shade.direct.createScratch({ label: 'deferred test scratch', rays: this.rayCount, workItems: this.rayCount });
        this.outputs = device.createBuffer({ label: 'deferred test inline outputs', size: this.rayCount * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.frameUniform = device.createBuffer({ label: 'deferred test frame', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    static async create(device: GPUDevice, platform: GIPlatformImpl, rays: Float32Array): Promise<DeferredBench> {
        await platform.shade.direct.prepare();
        const workLists: DirectWorkList[] = device.features.has('subgroups') ? ['subgroups', 'atomics'] : ['atomics'];
        const recordKernels = new Map<DirectWorkList, ComputeKernel>();
        for (const workList of workLists) {
            recordKernels.set(
                workList,
                await platform.kernels.computeKernel({
                    label: `deferred test records (${workList})`,
                    modules: [platform.shade.module],
                    code: recordKernelWgsl(platform.shade.direct.traceWgsl('testScratch', workList)),
                    entryPoint: 'main',
                    ownGroups: [RAYS_GROUP],
                }),
            );
        }
        const inlineKernel = await platform.kernels.computeKernel({ label: 'deferred test inline', modules: [platform.shade.module], code: INLINE_KERNEL_WGSL, entryPoint: 'main', ownGroups: [RAYS_GROUP] });
        return new DeferredBench(device, platform, recordKernels, inlineKernel, rays);
    }

    get workLists(): DirectWorkList[] {
        return [...this.recordKernels.keys()];
    }

    async light(groupMask: number, frameNumber: number, workList: DirectWorkList): Promise<Lit> {
        const device = this.device;
        device.queue.writeBuffer(this.frameUniform, 0, new Uint32Array([frameNumber, groupMask, 0, 0]));
        const encoder = device.createCommandEncoder({ label: 'deferred test' });
        const groups = Math.ceil(this.rayCount / WORKGROUP_SIZE);

        const recordKernel = this.recordKernels.get(workList);
        assert(recordKernel, `no ${workList} record kernel`);
        const records = encoder.beginComputePass({ label: 'deferred test records' });
        recordKernel.bind(records);
        records.setBindGroup(1, this.group(recordKernel, { buffer: this.scratch.buffer }));
        records.dispatchWorkgroups(groups);
        records.end();
        this.platform.shade.direct.encode(encoder, this.scratch, { frame: frameNumber, groupMask, owner: 'test', timer: NO_TIMER });

        const inline = encoder.beginComputePass({ label: 'deferred test inline' });
        this.inlineKernel.bind(inline);
        inline.setBindGroup(1, this.group(this.inlineKernel, { buffer: this.outputs }));
        inline.dispatchWorkgroups(groups);
        inline.end();
        device.queue.submit([encoder.finish()]);

        const results = this.scratch.rayResults;
        const [header, rayResults, inlineOutputs] = await Promise.all([
            readBuffer(device, this.scratch.buffer, RAY_SCRATCH_HEADER_BYTES),
            readRange(device, results.buffer, results.offset ?? 0, this.rayCount * RAY_RESULT_BYTES),
            readBuffer(device, this.outputs, this.rayCount * 16),
        ]);
        const headerWords = new Uint32Array(header);
        const droppedThisFrame = headerWords[RAY_SCRATCH_WORD.droppedThisFrame];
        const dropped = droppedThisFrame - this.droppedSoFar;
        this.droppedSoFar = droppedThisFrame;
        const words = new Uint32Array(rayResults);
        const out = new Float32Array(inlineOutputs);
        const deferred = new Float32Array(this.rayCount * 3);
        const inlineLight = new Float32Array(this.rayCount * 3);
        let hits = 0;
        let reached = 0;
        for (let ray = 0; ray < this.rayCount; ray++) {
            // π × the result: albedo 1, so albedo / π × E comes back as E.
            deferred[ray * 3] = Math.PI * halfToFloat(words[ray * 2] & 0xffff);
            deferred[ray * 3 + 1] = Math.PI * halfToFloat(words[ray * 2] >>> 16);
            deferred[ray * 3 + 2] = Math.PI * halfToFloat(words[ray * 2 + 1] & 0xffff);
            inlineLight.set(out.subarray(ray * 4, ray * 4 + 3), ray * 3);
            if (out[ray * 4 + 3] >= 0) hits++;
            if (out[ray * 4 + 3] === 1) reached++;
        }
        return { deferred, inline: inlineLight, hits, reached, listed: headerWords[RAY_SCRATCH_WORD.listed], dropped };
    }

    private group(kernel: ComputeKernel, second: GPUBufferBinding): GPUBindGroup {
        return this.device.createBindGroup({
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.rays } },
                { binding: 1, resource: second },
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

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'deferred test readback', size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'deferred test readback' });
    encoder.copyBufferToBuffer(source, offset, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

const luminance = (rgb: Float32Array, ray: number): number => 0.2126 * rgb[ray * 3] + 0.7152 * rgb[ray * 3 + 1] + 0.0722 * rgb[ray * 3 + 2];

interface Comparison {
    /** Σ deferred / Σ inline − 1 over every hit (luminance). */
    meanDifference: number;
    /** Over hits either path lights: the per-point relative difference's mean, p95 and largest. */
    pointMean: number;
    pointP95: number;
    pointMax: number;
    points: number;
    /** Points more than 1% apart (a shadow ray that went the other way at an edge, say). */
    pointsOver1Percent: number;
}

function compare(deferred: Float32Array, inline: Float32Array): Comparison {
    let deferredSum = 0;
    let inlineSum = 0;
    const differences: number[] = [];
    const rays = deferred.length / 3;
    for (let ray = 0; ray < rays; ray++) {
        const found = luminance(deferred, ray);
        const expected = luminance(inline, ray);
        deferredSum += found;
        inlineSum += expected;
        if (found > 0 || expected > 0) differences.push(Math.abs(found - expected) / Math.max(expected, IRRADIANCE_FLOOR));
    }
    differences.sort((a, b) => a - b);
    const at = (share: number): number => (differences.length === 0 ? 0 : differences[Math.min(differences.length - 1, Math.floor(share * differences.length))]);
    return {
        meanDifference: inlineSum > 0 ? deferredSum / inlineSum - 1 : deferredSum === 0 ? 0 : Infinity,
        pointMean: differences.length === 0 ? 0 : differences.reduce((sum, value) => sum + value, 0) / differences.length,
        pointP95: at(0.95),
        pointMax: at(1),
        points: differences.length,
        pointsOver1Percent: differences.filter((value) => value > 0.01).length,
    };
}

function judge(label: string, comparison: Comparison, failures: string[]): void {
    if (!(Math.abs(comparison.meanDifference) <= MAX_MEAN_DIFFERENCE)) failures.push(`${label}: mean ${(comparison.meanDifference * 100).toFixed(4)}% from inline`);
    if (!(comparison.pointMean <= MAX_POINT_MEAN)) failures.push(`${label}: per-point mean difference ${(comparison.pointMean * 100).toFixed(4)}%`);
    if (!(comparison.pointP95 <= MAX_POINT_P95)) failures.push(`${label}: per-point p95 difference ${(comparison.pointP95 * 100).toFixed(4)}%`);
}

function describe(comparison: Comparison): string {
    return (
        `mean ${(comparison.meanDifference * 100).toFixed(4)}%, per point mean ${(comparison.pointMean * 100).toFixed(4)}% / p95 ${(comparison.pointP95 * 100).toFixed(4)}% / max ${(comparison.pointMax * 100).toFixed(3)}% ` +
        `over ${comparison.points} lit points (${comparison.pointsOver1Percent} over 1%)`
    );
}

/** Both work lists must give the very same results (gi/direct's random numbers follow the ray). */
function sameBits(a: Float32Array, b: Float32Array): boolean {
    return a.length === b.length && a.every((value, index) => Object.is(value, b[index]));
}

/** Lights each group, both work lists; checks the list's length. Returns the subgroup (or only) list's results per group. */
async function lightGroups(bench: DeferredBench, label: string, groups: readonly number[], frameNumber: number, failures: string[]): Promise<Lit[]> {
    const lit: Lit[] = [];
    for (const groupMask of groups) {
        const byList = new Map<DirectWorkList, Lit>();
        for (const workList of bench.workLists) byList.set(workList, await bench.light(groupMask, frameNumber, workList));
        const first = byList.get(bench.workLists[0])!;
        for (const [workList, result] of byList) {
            if (result.listed !== result.reached) failures.push(`${label}, groups ${groupMask}, ${workList}: ${result.listed} hits listed, ${result.reached} reached by a light`);
            if (result.dropped !== 0) failures.push(`${label}, groups ${groupMask}, ${workList}: ${result.dropped} hits dropped`);
            if (!sameBits(result.deferred, first.deferred)) failures.push(`${label}, groups ${groupMask}: the ${workList} work list lit differently from the ${bench.workLists[0]} one`);
        }
        lit.push(first);
    }
    return lit;
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const translation = (x: number, y: number, z: number): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

/** Rays packed as the oracle packs them: origin, tMin, direction, tMax. */
function packRays(rays: ReadonlyArray<{ origin: Vec3; direction: Vec3 }>): Float32Array {
    const packed = new Float32Array(rays.length * 8);
    rays.forEach((ray, index) => {
        packed.set([...ray.origin, 0], index * 8);
        packed.set([...normalize(ray.direction), 1e6], index * 8 + 4);
    });
    return packed;
}

async function oracleCase(device: GPUDevice, oracle: OracleCase, failures: string[], log: (line: string) => void): Promise<Record<string, unknown>> {
    const platform = new GIPlatformImpl(device, { label: `deferred ${oracle.name}` });
    try {
        const scene = platform.scene;
        const white = scene.createMaterial({ albedo: [1, 1, 1] });
        const meshes = oracle.meshes.map((mesh) => scene.createMesh({ positions: mesh.positions, indices: mesh.indices ?? Uint32Array.from({ length: mesh.positions.length / 3 }, (_, index) => index) }));
        for (const instance of oracle.instances) {
            scene.createInstance({ mesh: meshes[instance.meshIndex], materials: [white], transform: instance.objectToWorld, visible: instance.visible, contributesToGI: instance.contributesToGi });
        }
        const lights = lightsFor(oracle);
        scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 10, position: lights.outside, range: 0, group: 0, castsShadows: true });
        scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 10, position: lights.inside, range: 0, group: 1, castsShadows: true });
        scene.createLight({ type: 'directional', color: [1, 1, 1], intensity: 3, direction: SUN_DIRECTION, group: 2, castsShadows: true });
        // Soft shadows: the shadow ray aims at a random point of a 0.2 m disc.
        scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 10, position: [lights.outside[0] - 0.9, lights.outside[1], lights.outside[2] + 0.4], radius: 0.2, range: 0, group: 3, castsShadows: true });
        await ready(device, platform);
        const bench = await DeferredBench.create(device, platform, oracle.rays);
        const groups = [1, 2, 4, 8];
        const lit = await lightGroups(bench, oracle.name, groups, 7, failures);
        const result: Record<string, unknown> = { rays: oracle.rays.length / 8, hits: lit[0].hits };
        lit.forEach((group, index) => {
            const comparison = compare(group.deferred, group.inline);
            judge(`${oracle.name}, group ${index}`, comparison, failures);
            result[`group${index}`] = { ...comparison, listed: group.listed, reached: group.reached };
            log(`${oracle.name}, group ${index}: ${describe(comparison)}; listed ${group.listed} of ${group.hits} hits`);
        });
        bench.destroy();
        return result;
    } finally {
        platform.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        const failures: string[] = [];
        const results: Record<string, unknown> = { workLists: device.features.has('subgroups') ? ['subgroups', 'atomics'] : ['atomics'] };

        // 1. T1.4's oracle points.
        const oracle: Record<string, unknown> = {};
        for (const oracleCaseData of buildOracleCases(FULL_SIZES)) {
            oracle[oracleCaseData.name] = await oracleCase(device, oracleCaseData, failures, log);
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
        results.oracle = oracle;

        // 2 and 3: a ground, a box, a sun, then 12 point lights.
        const platform = new GIPlatformImpl(device, { label: 'deferred lights' });
        try {
            const scene = platform.scene;
            const h = 200;
            const ground = scene.createMesh({
                positions: new Float32Array([-h, 0, -h, -h, 0, h, h, 0, h, h, 0, -h]),
                indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
                normals: new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]),
            });
            const white = scene.createMaterial({ albedo: [1, 1, 1] });
            scene.createInstance({ mesh: ground, materials: [white], transform: IDENTITY });

            // 3. A sun on a plane, through gi/direct. The last point is seen from below (the
            // ground's back), which no light reaches: it must not be listed.
            const sunColor: Vec3 = [1, 0.9, 0.8];
            const sunIrradiance = 3;
            const sun = scene.createLight({ type: 'directional', color: sunColor, intensity: sunIrradiance, direction: [0, -1, 0], castsShadows: true });
            const downward = [[0, 0], [3.7, -1.2], [-20, 40], [150, -150], [0.01, 0.02]].map(([x, z]) => ({ origin: [x, 10, z] as Vec3, direction: [0, -1, 0] as Vec3 }));
            const sunRays = packRays([...downward, { origin: [1, -5, 2], direction: [0, 1, 0] }]);
            await ready(device, platform);
            const sunBench = await DeferredBench.create(device, platform, sunRays);
            let worstSun = 0;
            for (const degrees of [0, 30, 60, 75, 89]) {
                const angle = (degrees * Math.PI) / 180;
                scene.updateLight(sun, { direction: [Math.sin(angle) * 0.6, -Math.cos(angle), -Math.sin(angle) * 0.8] });
                frame(device, platform);
                const [lit] = await lightGroups(sunBench, `sun at ${degrees} degrees`, [0xf], 1, failures);
                for (let ray = 0; ray < downward.length; ray++) {
                    for (let channel = 0; channel < 3; channel++) {
                        const expected = sunIrradiance * sunColor[channel] * Math.cos(angle);
                        worstSun = Math.max(worstSun, Math.abs(lit.deferred[ray * 3 + channel] - expected) / expected);
                    }
                }
                if (lit.listed !== downward.length) failures.push(`sun at ${degrees} degrees: ${lit.listed} hits listed, expected the ${downward.length} on the lit side`);
                if (lit.deferred.subarray(downward.length * 3).some((value) => value !== 0)) failures.push(`sun at ${degrees} degrees: the back of the ground was lit`);
            }
            results.sunOnPlaneWorstRelativeError = worstSun;
            log(`sun on a plane through gi/direct, 0-89 degrees: worst relative error ${(worstSun * 100).toFixed(4)}% (bar ${MAX_SUN_ERROR * 100}%)`);
            if (!(worstSun <= MAX_SUN_ERROR)) failures.push(`sun on a plane: ${(worstSun * 100).toFixed(4)}% from E cos(theta)`);
            sunBench.destroy();
            scene.destroyLight(sun);

            // 2. The stochastic light choice: hit-shading-lights' 12 point lights and a sun around a box.
            const boxGeometry = boxMesh([0.6, 0.6, 0.6]);
            const box = scene.createMesh({ positions: boxGeometry.positions, indices: boxGeometry.indices as Uint32Array });
            scene.createInstance({ mesh: box, materials: [white], transform: translation(0.4, 0.6, -0.2) });
            const random = createRandom(1234);
            for (let index = 0; index < 12; index++) {
                const angle = (index / 12) * 2 * Math.PI + random() * 0.4;
                const radius = 1.2 + random() * 3;
                scene.createLight({
                    type: 'point',
                    color: [0.5 + random() * 0.5, 0.5 + random() * 0.5, 0.5 + random() * 0.5],
                    intensity: 1 + random() * 29,
                    position: [Math.cos(angle) * radius, 0.4 + random() * 2.5, Math.sin(angle) * radius],
                    range: index % 3 === 0 ? 0 : 6 + random() * 6,
                    radius: 0.05,
                    castsShadows: true,
                });
            }
            scene.createLight({ type: 'directional', color: [1, 1, 1], intensity: 2, direction: normalize([0.3, -1, 0.2]), castsShadows: true });
            platform.shade.settings.maxShadowedLightsPerHit = 2;
            await ready(device, platform);
            // A grid of points on the ground around the box, and some on the box.
            const points: Array<{ origin: Vec3; direction: Vec3 }> = [];
            for (let x = -3; x <= 3; x += 0.25) for (let z = -3; z <= 3; z += 0.25) points.push({ origin: [x, 10, z], direction: [0, -1, 0] });
            const choiceBench = await DeferredBench.create(device, platform, packRays(points));
            const deferredMean = new Float32Array(points.length * 3);
            const inlineMean = new Float32Array(points.length * 3);
            for (let frameNumber = 0; frameNumber < FRAMES; frameNumber++) {
                const [lit] = await lightGroups(choiceBench, `stochastic choice, frame ${frameNumber}`, [0xf], frameNumber, failures);
                for (let index = 0; index < deferredMean.length; index++) {
                    deferredMean[index] += lit.deferred[index] / FRAMES;
                    inlineMean[index] += lit.inline[index] / FRAMES;
                }
            }
            const choice = compare(deferredMean, inlineMean);
            judge('stochastic choice', choice, failures);
            results.stochasticChoice = choice;
            log(`stochastic choice (12 shadowed point lights of radius 5 cm, at most 2 a hit, ${FRAMES} frames): ${describe(choice)}`);
            choiceBench.destroy();
        } finally {
            platform.dispose();
        }

        results.failures = failures;
        assert(failures.length === 0, failures.slice(0, 12).join('; '));
        return results;
    },
    { optionalFeatures: ['subgroups'] },
);
