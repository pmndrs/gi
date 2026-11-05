/**
 * Trace oracle: runs every oracle case through the real WGSL tracer on this
 * GPU and compares with the CPU reference (and with answers worked out by hand).
 *
 * Cases: thin walls, mirrored and non-uniform transforms, instancing, a closed
 * thin box no interior ray may escape, random scenes, and about two million
 * rays aimed at the seams of a tessellated quad (they must not slip through).
 * Everything runs once per variant (see VARIANTS), always with the watertight
 * test of Woop et al. (the tracer's only triangle test):
 * - `platform`: through the platform object (T1.3): the scene API, `prepare`,
 *   frames, then the `rays` backend's `encodeBatch`. Trees come from tinybvh
 *   in the worker, and hits report scene-copy instance slots, which the page
 *   maps back to the case's instance indices. The page reaches platform
 *   internals directly (it constructs `GIPlatformImpl` for `rays.whenBuilt()`
 *   and `sceneCopy.instanceSlot()`); inline tracing from a technique's own
 *   kernel, the way DDGI traces, is checked by `trace-inline`;
 * - `watertight-tinybvh`: the same trees assembled and traced directly;
 * - `watertight-tmbuild`: trees built by three-mesh-bvh and converted;
 * - `watertight-lbvh`: trees built on the GPU by the LBVH builder (B1,
 *   `@pmndrs/bvh`'s `createMeshLayoutBuilder`), read back and assembled the
 *   same way. It runs only when the device has subgroups; without them the
 *   builder would fall back to tinybvh, which the variants above already cover;
 * - `watertight-hploc`: the same with the GPU H-PLOC builder (B2).
 * The default device limits are used on purpose: the oracle scenes are tiny and
 * the core has to work there.
 *
 * URL option: `?sharedRays=100000` shrinks the shared-edge scenes for a quick look.
 */

import {
    assembleTraceScene,
    createCpuTracer,
    createMeshBvhBuilder,
    HIT_RECORD_BYTES,
    newTraceStats,
    RAY_BYTES,
    TRACE_MODULE_ENTRIES,
    TraceBatchKernel,
    uploadTraceScene,
    type MeshBvhBuilder,
    type TraceSceneData,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { type TracedHit } from '../../../../packages/ddgi-core/src/platform/trace/oracle-compare.ts';
import { createMeshLayoutBuilder, createTinyBvhBuilder, type MeshLayoutBuilder } from '../../../../packages/bvh/src/index.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../packages/ddgi-core/src/platform/trace/mesh-bvh-topology.ts';
import { assert, runGpuTest } from '../../client.ts';
import {
    buildOracleCases,
    FULL_SIZES,
    sharedEdgeCase,
    unpackRay,
    type OracleCase,
    type SharedEdgeCase,
} from './cases.ts';
import { IDENTITY_MATRIX } from '../trace-bench/gltf-geometry.ts';
import { buildMeshBvhWithTmbvh } from '../trace-bench/tmbvh-convert.ts';
import {
    computeReferenceAnswersYielding,
    evaluateCase,
    evaluateSharedEdgeCase,
    sharedEdgeFailures,
    type CaseResult,
    type SharedEdgeReport,
    type TracerAnswers,
} from './check.ts';

/**
 * What each pass of the oracle traces: the triangle test, and who built the
 * per-mesh trees. `three-mesh-bvh` trees are built by that library and
 * converted to our layout (../trace-bench/tmbvh-convert.ts), the T0.10 part C
 * experiment; they are traced by the same WGSL walk.
 */
interface OracleVariant {
    name: string;
    treeSource: 'platform' | 'tinybvh' | 'three-mesh-bvh' | 'gpu-lbvh' | 'gpu-hploc';
}

const VARIANTS: OracleVariant[] = [
    { name: 'platform', treeSource: 'platform' },
    { name: 'watertight-tinybvh', treeSource: 'tinybvh' },
    { name: 'watertight-tmbuild', treeSource: 'three-mesh-bvh' },
    { name: 'watertight-lbvh', treeSource: 'gpu-lbvh' },
    { name: 'watertight-hploc', treeSource: 'gpu-hploc' },
];

/** The GPU builders (B1 LBVH, B2 H-PLOC), made once the device is known. */
const gpuBuilders = new Map<'gpu-lbvh' | 'gpu-hploc', MeshLayoutBuilder>();

/** three-mesh-bvh target leaf size for the converted trees, as in the benchmark's default. */
const TMBUILD_LEAF_SIZE = 4;

/** Rays per shared-edge scene. Two scenes, so 2 million in total for each triangle test. */
const DEFAULT_SHARED_EDGE_RAYS = 1_000_000;

/** Rays the CPU port of the walk traces per case, to report the stack depth it needs. */
const STACK_PROBE_RAYS = 2000;

/** Rays of each shared-edge scene also checked against the brute-force reference. */
const SHARED_EDGE_REFERENCE_RAYS = 20000;

const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;

interface GpuAnswers {
    closest: Float32Array;
    closestWords: Uint32Array;
    blocked: Uint32Array;
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({
        label: 'oracle readback',
        size: byteLength,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = device.createCommandEncoder({ label: 'oracle readback' });
    encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** Records one batch: the low-level kernel or the platform's `encodeBatch`. */
type EncodeBatch = (encoder: GPUCommandEncoder, batch: { mode: 'closest' | 'any'; rays: GPUBuffer; results: GPUBuffer; count: number }) => void;

/** Traces the packed rays on the GPU in both modes and reads the results back. */
async function traceOnGpu(device: GPUDevice, encodeBatch: EncodeBatch, packedRays: Float32Array): Promise<GpuAnswers> {
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'oracle rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hitBytes = count * HIT_RECORD_BYTES;
    const anyBytes = count * 4;
    const hits = device.createBuffer({ label: 'oracle hits', size: hitBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const occlusion = device.createBuffer({ label: 'oracle occlusion', size: anyBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const encoder = device.createCommandEncoder({ label: 'oracle trace' });
    encodeBatch(encoder, { mode: 'closest', rays, results: hits, count });
    encodeBatch(encoder, { mode: 'any', rays, results: occlusion, count });
    device.queue.submit([encoder.finish()]);

    const hitData = await readBuffer(device, hits, hitBytes);
    const anyData = await readBuffer(device, occlusion, anyBytes);
    rays.destroy();
    hits.destroy();
    occlusion.destroy();
    return {
        closest: new Float32Array(hitData),
        closestWords: new Uint32Array(hitData),
        blocked: new Uint32Array(anyData),
    };
}


/** One ray's closest-hit record as the comparison code wants it. */
function hitAt(answers: GpuAnswers, index: number): TracedHit | null {
    const base = index * HIT_RECORD_WORDS;
    const t = answers.closest[base];
    if (!(t >= 0)) return null;
    return {
        hit: true,
        t,
        instanceIndex: answers.closestWords[base + 1],
        triangleIndex: answers.closestWords[base + 2],
        frontFace: answers.closestWords[base + 3] !== 0,
        u: answers.closest[base + 4],
        v: answers.closest[base + 5],
    };
}

/** The read-back results in the form the checks (check.ts) take. */
function tracerAnswers(answers: GpuAnswers): TracerAnswers {
    return {
        closest: (index) => hitAt(answers, index),
        any: (index) => answers.blocked[index] !== 0,
    };
}

interface BuiltScene {
    data: TraceSceneData;
    buildMs: number;
}

async function buildScene(builder: MeshBvhBuilder, oracleCase: OracleCase, variant: OracleVariant): Promise<BuiltScene> {
    const start = performance.now();
    let bvhs;
    if (variant.treeSource === 'three-mesh-bvh') {
        bvhs = oracleCase.meshes.map((mesh) => buildMeshBvhWithTmbvh(mesh, TMBUILD_LEAF_SIZE).bvh);
    } else if (variant.treeSource === 'gpu-lbvh' || variant.treeSource === 'gpu-hploc') {
        const gpuBuilder = gpuBuilders.get(variant.treeSource);
        assert(gpuBuilder, `the ${variant.treeSource} builder was not made`);
        const results = await gpuBuilder.buildLayouts(oracleCase.meshes);
        for (const result of results) assert(result.builtBy === variant.treeSource, `a mesh fell back to ${result.builtBy}: ${result.fallbackReason}`);
        bvhs = results.map((result) => result.layout);
    } else {
        bvhs = await Promise.all(oracleCase.meshes.map((mesh) => builder.build(mesh)));
    }
    const data = assembleTraceScene(bvhs, oracleCase.instances);
    return { data, buildMs: performance.now() - start };
}

/** Deepest stack the CPU port of the walk used over the first rays of a case. */
function probeStackDepth(scene: TraceSceneData, oracleCase: OracleCase): number {
    const tracer = createCpuTracer(scene);
    const stats = newTraceStats();
    const rayCount = Math.min(STACK_PROBE_RAYS, oracleCase.rays.length / 8);
    for (let index = 0; index < rayCount; index++) {
        const ray = unpackRay(oracleCase.rays, index);
        tracer.closest(ray, stats);
        tracer.any(ray, stats);
    }
    return stats.maxStackSize;
}

/** A frame with no technique work: the platform only uploads the scene and places trees. */
const IDLE_FRAME: FrameInput = {
    deltaTime: 0,
    budget: { units: 0 },
    camera: { position: [0, 0, 0], view: IDENTITY_MATRIX, projection: IDENTITY_MATRIX, near: 0.1, far: 100 },
};

/**
 * Traces a case through the platform object: the scene API, `prepare`, frames
 * until every tree has arrived, then the `rays` backend's `encodeBatch`. Uses
 * two internals a host doesn't have: `rays.whenBuilt()` and
 * `sceneCopy.instanceSlot()`. Hits report scene-copy instance slots; they
 * are mapped back to the case's own instance indices for the checks.
 */
async function traceThroughPlatform(device: GPUDevice, oracleCase: OracleCase): Promise<GpuAnswers> {
    const platform = new GIPlatformImpl(device, { label: `oracle ${oracleCase.name}` });
    try {
        const material = platform.scene.createMaterial({ albedo: [1, 1, 1] });
        const meshes = oracleCase.meshes.map((mesh) =>
            platform.scene.createMesh({
                positions: mesh.positions,
                indices: mesh.indices ?? Uint32Array.from({ length: mesh.positions.length / 3 }, (_, index) => index),
            }),
        );
        const caseIndexBySlot = new Map<number, number>();
        for (const instance of oracleCase.instances) {
            const handle = platform.scene.createInstance({
                mesh: meshes[instance.meshIndex],
                materials: [material],
                transform: instance.objectToWorld,
                visible: instance.visible,
                contributesToGI: instance.contributesToGi,
            });
            caseIndexBySlot.set(platform.sceneCopy.instanceSlot(handle), instance.instanceIndex);
        }
        await platform.prepare();
        const frame = (): void => {
            const encoder = device.createCommandEncoder({ label: 'oracle platform frame' });
            platform.encode(encoder, IDLE_FRAME);
            device.queue.submit([encoder.finish()]);
        };
        frame(); // uploads the scene; the worker starts building
        await platform.rays.whenBuilt();
        frame(); // places the trees and builds the top level
        assert(!platform.rays.building, 'the platform still reports trees building');

        const answers = await traceOnGpu(device, (encoder, batch) => {
            platform.rays.encodeBatch(encoder, { mode: batch.mode, rays: batch.rays, hits: batch.results, count: batch.count });
        }, oracleCase.rays);
        // GiHit.instance is the scene-copy slot; the checks know the case's instance indices.
        for (let ray = 0; ray < answers.blocked.length; ray++) {
            const base = ray * HIT_RECORD_WORDS;
            if (!(answers.closest[base] >= 0)) continue;
            const caseIndex = caseIndexBySlot.get(answers.closestWords[base + 1]);
            answers.closestWords[base + 1] = caseIndex ?? UNKNOWN_SLOT;
        }
        return answers;
    } finally {
        platform.dispose();
    }
}

/** Reported for a hit on a slot that holds none of the case's instances, so the checks fail it. */
const UNKNOWN_SLOT = 0xffffffff;

/** Traces a case in a variant and returns the answers plus the trees (for the stack and size report). */
async function traceCase(
    device: GPUDevice,
    kernel: TraceBatchKernel,
    builder: MeshBvhBuilder,
    oracleCase: OracleCase,
    variant: OracleVariant,
): Promise<{ answers: GpuAnswers; built: BuiltScene }> {
    const built = await buildScene(builder, oracleCase, variant);
    if (variant.treeSource === 'platform') return { answers: await traceThroughPlatform(device, oracleCase), built };
    const buffers = uploadTraceScene(device, built.data, oracleCase.name);
    try {
        return { answers: await traceOnGpu(device, (encoder, batch) => kernel.encode(encoder, buffers, batch), oracleCase.rays), built };
    } finally {
        buffers.destroy();
    }
}

async function runSharedEdgeCase(
    device: GPUDevice,
    kernel: TraceBatchKernel,
    builder: MeshBvhBuilder,
    oracleCase: SharedEdgeCase,
    variant: OracleVariant,
): Promise<SharedEdgeReport> {
    const { answers } = await traceCase(device, kernel, builder, oracleCase, variant);
    return evaluateSharedEdgeCase(oracleCase, tracerAnswers(answers), SHARED_EDGE_REFERENCE_RAYS);
}

function summarize(result: CaseResult, stackUsed: number, scene: TraceSceneData, buildMs: number) {
    return {
        rays: result.rays,
        closest: {
            matches: result.closest.matches,
            ambiguous: result.closest.ambiguous,
            mismatches: result.closest.mismatches,
            maxTError: Number(result.closest.maxTError.toExponential(2)),
        },
        any: { matches: result.any.matches, ambiguous: result.any.ambiguous, mismatches: result.any.mismatches },
        closestEscapes: result.closestEscapes,
        anyEscapes: result.anyEscapes,
        referenceEscapes: result.referenceEscapes,
        expectationFailures: result.expectationFailures.length,
        topLevelDepth: scene.topLevel.maxDepth,
        maxMeshDepth: scene.maxMeshDepth,
        requiredStack: scene.requiredStackSize,
        stackUsedByCpuPort: stackUsed,
        sceneBytes: scene.byteLength,
        buildMs: Math.round(buildMs),
    };
}

const sharedRays = Number(new URLSearchParams(location.search).get('sharedRays')) || DEFAULT_SHARED_EDGE_RAYS;

await runGpuTest(async ({ device, log }) => {
    const builder = createMeshBvhBuilder({ worker: 'auto' });
    assert(builder.usesWorker, 'the BVH builder did not start its worker');

    // The worker must give the same tree as building on this thread.
    {
        const sample = buildOracleCases(FULL_SIZES).find((entry) => entry.name === 'random-soup');
        assert(sample !== undefined, 'random-soup case missing');
        const fromWorker = await builder.build(sample.meshes[0]);
        const inline = buildMeshBvhWithTinyBvh(sample.meshes[0], await createTinyBvhBuilder());
        const inlineNodes = new Uint8Array(inline.nodes);
        assert(
            fromWorker.nodeCount === inline.nodeCount &&
                fromWorker.maxDepth === inline.maxDepth &&
                new Uint8Array(fromWorker.nodes).every((byte, position) => byte === inlineNodes[position]),
            'the worker built a different tree than the main thread',
        );
        log(`worker build matches the inline build (${inline.nodeCount} nodes, depth ${inline.maxDepth})`);
    }

    const failures: string[] = [];
    const byTest: Record<string, { cases: Record<string, unknown>; sharedEdges: SharedEdgeReport[] }> = {};
    let deepestStackUsed = 0;
    let deepestRequired = 0;

    // The brute-force answers are the slow part of the page, so compute them once per case.
    const cases = buildOracleCases(FULL_SIZES);
    const references = [];
    for (const oracleCase of cases) references.push(await computeReferenceAnswersYielding(oracleCase));

    for (const kind of ['gpu-lbvh', 'gpu-hploc'] as const) gpuBuilders.set(kind, await createMeshLayoutBuilder(device, { prefer: kind, label: `oracle ${kind}` }));
    const skipped: string[] = [];
    for (const variant of VARIANTS) {
        const gpuBuilder = variant.treeSource === 'gpu-lbvh' || variant.treeSource === 'gpu-hploc' ? gpuBuilders.get(variant.treeSource) : undefined;
        if (gpuBuilder && gpuBuilder.kind === 'tinybvh') {
            skipped.push(`${variant.name}: ${gpuBuilder.fallbackReason}`);
            log(`[${variant.name}] skipped: ${gpuBuilder.fallbackReason}`);
            continue;
        }
        const kernel = await TraceBatchKernel.create(device);
        const results = { cases: {} as Record<string, unknown>, sharedEdges: [] as SharedEdgeReport[] };
        byTest[variant.name] = results;

        for (const [caseIndex, oracleCase] of cases.entries()) {
            const { answers, built } = await traceCase(device, kernel, builder, oracleCase, variant);

            const result = evaluateCase(oracleCase, tracerAnswers(answers), references[caseIndex]);
            const stackUsed = probeStackDepth(built.data, oracleCase);
            deepestStackUsed = Math.max(deepestStackUsed, stackUsed);
            deepestRequired = Math.max(deepestRequired, built.data.requiredStackSize);
            results.cases[oracleCase.name] = summarize(result, stackUsed, built.data, built.buildMs);
            log(
                `[${variant.name}] ${oracleCase.name}: ${result.rays} rays, closest ${result.closest.mismatches} mismatches / ${result.closest.ambiguous} ambiguous, ` +
                    `any ${result.any.mismatches} / ${result.any.ambiguous}, ` +
                    `escapes closest ${result.closestEscapes.ordinary} + ${result.closestEscapes.edgeAimed} at corners, ` +
                    `any ${result.anyEscapes.ordinary} + ${result.anyEscapes.edgeAimed} at corners`,
            );
            for (const failure of result.failures) failures.push(`[${variant.name}] ${oracleCase.name}: ${failure}`);
        }

        for (const transformed of [false, true]) {
            const report = await runSharedEdgeCase(device, kernel, builder, sharedEdgeCase(sharedRays, transformed), variant);
            results.sharedEdges.push(report);
            log(
                `[${variant.name}] ${report.name}: ${report.slipped} of ${report.rays} rays slipped between triangles ` +
                    `(${report.slippedAny} any-hit)`,
            );
            for (const failure of sharedEdgeFailures(report)) failures.push(`[${variant.name}] ${report.name}: ${failure}`);
        }
    }
    builder.dispose();

    const limits = device.limits;
    assert(failures.length === 0, failures.slice(0, 12).join('\n'));
    const slippedTotals: Record<string, number> = {};
    for (const variant of VARIANTS) {
        if (!byTest[variant.name]) continue;
        slippedTotals[variant.name] = byTest[variant.name].sharedEdges.reduce((sum, report) => sum + report.slipped, 0);
    }
    return {
        skipped,
        sharedEdgeRaysPerTest: byTest[VARIANTS[0].name].sharedEdges.reduce((sum, report) => sum + report.rays, 0),
        slippedTotals,
        byTest,
        traceStorageBuffers: TRACE_MODULE_ENTRIES.length,
        batchKernelStorageBuffers: TRACE_MODULE_ENTRIES.length + 2,
        maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        deepestStackUsedByCpuPort: deepestStackUsed,
        deepestStackRequired: deepestRequired,
    };
}, {
    // As a host asking optionalFeatures(adapter): the platform then keeps its top level on the GPU (B1a).
    optionalFeatures: ['subgroups'],
});
