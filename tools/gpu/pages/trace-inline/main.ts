/**
 * Inline tracing from a technique's own kernel, the way DDGI traces (T1.3).
 *
 * Only the public API: `createGI(device)`, the scene API, `addTechnique` with a
 * minimal test technique, `prepare`, `encode`. The technique builds its kernel
 * through `context.kernels.computeKernel` with the `rays` and scene modules in
 * group 0 and three storage buffers of its own in group 1: DDGI's binding shape
 * (tracer 3 + scene copy 2 + technique 3 = 8, the default per-stage limit). The
 * kernel calls `giTraceClosest` and `giTraceAny` inline, then reads the hit's
 * surface through the scene copy: the material (whose albedo carries the case's
 * instance index, so hits map back without any platform internals) and the
 * geometric normal (front faces must agree with it).
 *
 * The answers are checked against the CPU reference on the oracle scenes,
 * mirrored boxes and thin walls included, plus rays at shared edges. The
 * kernel builder must also refuse a 9th storage buffer with a GILimitError.
 * Default device limits, on purpose.
 */

import { GILimitError } from '../../../../packages/ddgi-core/src/platform/contract/errors.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { MaterialHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame } from '../../../../packages/ddgi-core/src/platform/contract/technique.ts';
import type { ComputeKernel } from '../../../../packages/ddgi-core/src/platform/contract/wgsl.ts';
import { createGI } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { RAY_BYTES } from '../../../../packages/ddgi-core/src/platform/trace/trace-gpu.ts';
import type { TracedHit } from '../../../../packages/ddgi-core/src/platform/trace/oracle-compare.ts';
import { assert, runGpuTest } from '../../client.ts';
import { buildOracleCases, FULL_SIZES, sharedEdgeCase, type OracleCase } from '../trace-oracle/cases.ts';
import {
    computeReferenceAnswersYielding,
    evaluateCase,
    evaluateSharedEdgeCase,
    sharedEdgeFailures,
    type TracerAnswers,
} from '../trace-oracle/check.ts';

/** Rays per shared-edge scene; the oracle page runs a million, this checks the inline path. */
const SHARED_EDGE_RAYS = 200_000;
const SHARED_EDGE_REFERENCE_RAYS = 5000;

const WORKGROUP_SIZE = 64;
/** 32-bit words the kernel writes per ray. */
const RESULT_WORDS = 8;
/** Written as the case instance for a miss. */
const NO_INSTANCE = 0xffffffff;
/** `GPUShaderStage.COMPUTE`. */
const COMPUTE = 0x4;

/** The technique's group 1: rays in, hits out, any-hit answers out. Three storage buffers, as DDGI's update kernel. */
const OWN_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'trace-inline group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: COMPUTE, buffer: { type: 'storage' } },
    ],
};

const KERNEL_WGSL = /* wgsl */ `
struct InlineResult {
    t: f32,
    caseInstance: u32,   // the case's instance index, read back from the hit material's albedo
    triangle: u32,
    frontFace: u32,
    barycentrics: vec2f,
    facing: f32,         // dot(ray direction, scene copy's geometric normal)
    padding: u32,
}

@group(1) @binding(0) var<storage, read> inlineRays: array<GiRay>;
@group(1) @binding(1) var<storage, read_write> inlineHits: array<InlineResult>;
@group(1) @binding(2) var<storage, read_write> inlineBlocked: array<u32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    // 2D dispatch, so a large batch stays under the per-dimension workgroup limit.
    let index = id.y * groups.x * ${WORKGROUP_SIZE}u + id.x;
    if (index >= arrayLength(&inlineRays)) {
        return;
    }
    let ray = inlineRays[index];
    let hit = giTraceClosest(ray);
    var result = InlineResult(hit.t, ${NO_INSTANCE}u, hit.triangle, select(0u, 1u, hit.frontFace), hit.barycentrics, 0.0, 0u);
    if (hit.t >= 0.0) {
        result.caseInstance = u32(giMaterial(giMaterialAt(hit.instance, hit.triangle)).albedo.x + 0.5);
        result.facing = dot(ray.direction, giInstanceGeometricNormal(hit.instance, hit.triangle));
    }
    inlineHits[index] = result;
    inlineBlocked[index] = select(0u, 1u, giTraceAny(ray));
}
`;

interface InlineJob {
    rays: GPUBuffer;
    hits: GPUBuffer;
    blocked: GPUBuffer;
    count: number;
}

/** The smallest technique that traces inline: one kernel, run when the page gives it a job. */
class InlineTraceTechnique implements GITechnique {
    readonly id = 'trace-inline';
    readonly requirements = { queries: ['rays' as const] };
    readonly settings = {};
    readonly debugViews = [];
    readonly counters: Record<string, number> = {};
    readonly msPerUnit = undefined;
    readonly memoryBytes = 0;
    context: TechniqueContext | null = null;
    kernel: ComputeKernel | null = null;
    job: InlineJob | null = null;

    get shading(): never {
        throw new Error('the inline-trace test technique has no shading module');
    }

    attach(context: TechniqueContext): void {
        this.context = context;
    }

    async prepare(): Promise<void> {
        const context = this.context;
        assert(context, 'prepare before attach');
        this.kernel = await context.kernels.computeKernel({
            label: 'trace-inline',
            modules: [context.queries.rays.module, context.scene.module],
            code: KERNEL_WGSL,
            entryPoint: 'main',
            ownGroups: [OWN_GROUP],
        });
    }

    encode(encoder: GPUCommandEncoder, _frame: TechniqueFrame): void {
        const job = this.job;
        const kernel = this.kernel;
        if (!job || !kernel || !this.context) return;
        this.job = null;
        const group = this.context.device.createBindGroup({
            label: 'trace-inline group 1',
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: job.rays, size: job.count * RAY_BYTES } },
                { binding: 1, resource: { buffer: job.hits } },
                { binding: 2, resource: { buffer: job.blocked } },
            ],
        });
        const workgroups = Math.ceil(job.count / WORKGROUP_SIZE);
        const x = Math.min(workgroups, this.context.device.limits.maxComputeWorkgroupsPerDimension);
        const pass = encoder.beginComputePass({ label: 'trace-inline', timestampWrites: this.context.timer.computePass('trace-inline/trace') });
        kernel.bind(pass);
        pass.setBindGroup(1, group);
        pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
        pass.end();
    }

    resetHistory(): void {}
    dispose(): void {}
}

const FRAME: FrameInput = {
    deltaTime: 1 / 60,
    budget: {},
    camera: { position: [0, 0, 0], view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], projection: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], near: 0.1, far: 100 },
};

async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'trace-inline readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

interface InlineAnswers extends TracerAnswers {
    /** Hits whose frontFace disagrees with the sign of dot(direction, geometric normal), grazing hits left out. */
    facingDisagreements: number;
}

/**
 * Describes one oracle case through the scene API, lets the platform build it,
 * then traces its rays in the technique's kernel during an ordinary `encode`.
 */
async function traceCase(device: GPUDevice, oracleCase: OracleCase): Promise<InlineAnswers> {
    const gi = createGI(device, { label: `inline ${oracleCase.name}` });
    try {
        const technique = gi.addTechnique(new InlineTraceTechnique());
        const meshes = oracleCase.meshes.map((mesh) =>
            gi.scene.createMesh({
                positions: mesh.positions,
                indices: mesh.indices ?? Uint32Array.from({ length: mesh.positions.length / 3 }, (_, index) => index),
            }),
        );
        for (const instance of oracleCase.instances) {
            // The albedo's red channel carries the case's instance index; the kernel reads it back from the hit.
            const material: MaterialHandle = gi.scene.createMaterial({ albedo: [instance.instanceIndex, 0, 0] });
            gi.scene.createInstance({
                mesh: meshes[instance.meshIndex],
                materials: [material],
                transform: instance.objectToWorld,
                visible: instance.visible,
                contributesToGI: instance.contributesToGi,
            });
        }
        await gi.prepare();
        const rays = technique.context?.queries.rays;
        assert(rays, 'the technique was not attached');

        const frame = async (): Promise<void> => {
            const encoder = device.createCommandEncoder({ label: 'trace-inline frame' });
            gi.encode(encoder, FRAME);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        };
        // Frames until every mesh tree has arrived from the worker, then one more to place them.
        await frame();
        for (let waited = 0; rays.building; waited++) {
            assert(waited < 600, 'mesh trees still building after 600 frames');
            await new Promise((resolve) => setTimeout(resolve, 5));
            await frame();
        }
        await frame();

        const count = oracleCase.rays.length / 8;
        const rayBuffer = device.createBuffer({ label: 'trace-inline rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(rayBuffer, 0, oracleCase.rays as Float32Array<ArrayBuffer>);
        const hitBytes = count * RESULT_WORDS * 4;
        const hits = device.createBuffer({ label: 'trace-inline hits', size: hitBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const blocked = device.createBuffer({ label: 'trace-inline blocked', size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        technique.job = { rays: rayBuffer, hits, blocked, count };
        await frame();
        assert(technique.job === null, 'the technique did not run in encode');

        const hitData = await readBuffer(device, hits, hitBytes);
        const blockedWords = new Uint32Array(await readBuffer(device, blocked, count * 4));
        rayBuffer.destroy();
        hits.destroy();
        blocked.destroy();

        const floats = new Float32Array(hitData);
        const words = new Uint32Array(hitData);
        let facingDisagreements = 0;
        const closest = (index: number): TracedHit | null => {
            const base = index * RESULT_WORDS;
            const t = floats[base];
            if (!(t >= 0)) return null;
            return {
                hit: true,
                t,
                instanceIndex: words[base + 1],
                triangleIndex: words[base + 2],
                frontFace: words[base + 3] !== 0,
                u: floats[base + 4],
                v: floats[base + 5],
            };
        };
        for (let index = 0; index < count; index++) {
            const base = index * RESULT_WORDS;
            const facing = floats[base + 6];
            if (floats[base] >= 0 && Math.abs(facing) > 1e-4 && (words[base + 3] !== 0) !== facing < 0) facingDisagreements++;
        }
        return { closest, any: (index) => blockedWords[index] !== 0, facingDisagreements };
    } finally {
        gi.dispose();
    }
}

/** A ninth storage buffer must be refused before compiling, naming the limit. */
async function checkBindingLimit(device: GPUDevice): Promise<string> {
    const gi = createGI(device, { label: 'inline limit' });
    try {
        const attached = gi.addTechnique(new InlineTraceTechnique()).context;
        assert(attached, 'the technique was not attached');
        const tooMany: GPUBindGroupLayoutDescriptor = {
            entries: [...OWN_GROUP.entries, { binding: 3, visibility: COMPUTE, buffer: { type: 'read-only-storage' } }],
        };
        const error = await attached.kernels
            .computeKernel({
                label: 'nine storage buffers',
                modules: [attached.queries.rays.module, attached.scene.module],
                code: KERNEL_WGSL,
                entryPoint: 'main',
                ownGroups: [tooMany],
            })
            .then(() => null, (caught: unknown) => caught);
        assert(error instanceof GILimitError, `a kernel with 9 storage buffers gave ${String(error)}, not a GILimitError`);
        assert(error.limit === 'maxStorageBuffersPerShaderStage' && error.needed === 9, `wrong GILimitError: ${error.message}`);
        return error.message;
    } finally {
        gi.dispose();
    }
}

await runGpuTest(async ({ device, log }) => {
    // runGpuTest asks for no limits, so this is a default device: DDGI's real budget.
    assert(device.limits.maxStorageBuffersPerShaderStage === 8, `expected a default-limits device, got ${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage`);
    const failures: string[] = [];
    const cases: Record<string, unknown> = {};

    for (const oracleCase of buildOracleCases(FULL_SIZES)) {
        const reference = await computeReferenceAnswersYielding(oracleCase);
        const answers = await traceCase(device, oracleCase);
        const result = evaluateCase(oracleCase, answers, reference);
        cases[oracleCase.name] = {
            rays: result.rays,
            closest: { mismatches: result.closest.mismatches, ambiguous: result.closest.ambiguous },
            any: { mismatches: result.any.mismatches, ambiguous: result.any.ambiguous },
            escapes: result.closestEscapes.ordinary + result.closestEscapes.edgeAimed + result.anyEscapes.ordinary + result.anyEscapes.edgeAimed,
            facingDisagreements: answers.facingDisagreements,
        };
        log(`${oracleCase.name}: ${result.rays} rays, closest ${result.closest.mismatches} mismatches, any ${result.any.mismatches}, facing ${answers.facingDisagreements}`);
        for (const failure of result.failures) failures.push(`${oracleCase.name}: ${failure}`);
        if (answers.facingDisagreements > 0) failures.push(`${oracleCase.name}: ${answers.facingDisagreements} hits whose frontFace disagrees with the geometric normal`);
    }

    const sharedEdges: Record<string, unknown> = {};
    for (const transformed of [false, true]) {
        const oracleCase = sharedEdgeCase(SHARED_EDGE_RAYS, transformed);
        const report = evaluateSharedEdgeCase(oracleCase, await traceCase(device, oracleCase), SHARED_EDGE_REFERENCE_RAYS);
        sharedEdges[report.name] = { rays: report.rays, slipped: report.slipped, slippedAny: report.slippedAny };
        log(`${report.name}: ${report.slipped} of ${report.rays} slipped (${report.slippedAny} any-hit)`);
        for (const failure of sharedEdgeFailures(report)) failures.push(`${report.name}: ${failure}`);
    }

    const limitMessage = await checkBindingLimit(device);
    log(`9 storage buffers: ${limitMessage}`);

    assert(failures.length === 0, failures.slice(0, 12).join('\n'));
    return {
        storageBuffers: { tracer: 3, sceneCopy: 2, technique: 3, limit: device.limits.maxStorageBuffersPerShaderStage },
        cases,
        sharedEdges,
        ninthStorageBuffer: limitMessage,
    };
}, {
    // As a host asking optionalFeatures(adapter): the platform then keeps its top level on the GPU (B1a).
    optionalFeatures: ['subgroups'],
});
