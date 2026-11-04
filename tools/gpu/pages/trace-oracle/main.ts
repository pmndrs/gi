/**
 * Trace oracle: runs every oracle case through the real WGSL tracer on this
 * GPU and compares with the CPU reference (and with answers worked out by hand).
 *
 * Cases: thin walls, mirrored and non-uniform transforms, instancing, a closed
 * thin box no interior ray may escape, random scenes, and about two million
 * rays aimed at the seams of a tessellated quad (they must not slip through).
 * Everything runs twice, once per ray-triangle test: Moller-Trumbore (the
 * design's default) and the watertight test of Woop et al. (the fallback the
 * design names for rays that slip between triangles).
 * The default device limits are used on purpose: the oracle scenes are tiny and
 * the core has to work there.
 *
 * URL option: `?sharedRays=100000` shrinks the shared-edge scenes for a quick look.
 */

import {
    assembleTraceScene,
    buildMeshBvh,
    buildReferenceScene,
    createCpuTracer,
    createMeshBvhBuilder,
    HIT_RECORD_BYTES,
    newTraceStats,
    RAY_BYTES,
    TRACE_MODULE_ENTRIES,
    TraceBatchKernel,
    traceClosestReference,
    uploadTraceScene,
    type MeshBvhBuilder,
    type TraceSceneBuffers,
    type TraceSceneData,
    type TracedHit,
    type TriangleTest,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { assert, runGpuTest } from '../../client.ts';
import {
    buildOracleCases,
    FULL_SIZES,
    SHARED_EDGE_CATEGORIES,
    sharedEdgeCase,
    unpackRay,
    type OracleCase,
    type SharedEdgeCase,
} from './cases.ts';
import { computeReferenceAnswersYielding, evaluateCase, type CaseResult } from './check.ts';

const TRIANGLE_TESTS: TriangleTest[] = ['moller-trumbore', 'watertight'];

/** Rays per shared-edge scene. Two scenes, so 2 million in total for each triangle test. */
const DEFAULT_SHARED_EDGE_RAYS = 1_000_000;

/** Rays the CPU port of the walk traces per case, to report the stack depth it needs. */
const STACK_PROBE_RAYS = 2000;

/** Rays of each shared-edge scene also checked against the brute-force reference. */
const SHARED_EDGE_REFERENCE_RAYS = 20000;

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

/** Traces the packed rays on the GPU in both modes and reads the results back. */
async function traceOnGpu(
    device: GPUDevice,
    kernel: TraceBatchKernel,
    scene: TraceSceneBuffers,
    packedRays: Float32Array,
): Promise<GpuAnswers> {
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'oracle rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hitBytes = count * HIT_RECORD_BYTES;
    const anyBytes = count * 4;
    const hits = device.createBuffer({ label: 'oracle hits', size: hitBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const occlusion = device.createBuffer({ label: 'oracle occlusion', size: anyBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const encoder = device.createCommandEncoder({ label: 'oracle trace' });
    kernel.encode(encoder, scene, { mode: 'closest', rays, results: hits, count });
    kernel.encode(encoder, scene, { mode: 'any', rays, results: occlusion, count });
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

const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;

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

interface BuiltScene {
    data: TraceSceneData;
    buildMs: number;
}

async function buildScene(builder: MeshBvhBuilder, oracleCase: OracleCase, triangleTest: TriangleTest): Promise<BuiltScene> {
    const start = performance.now();
    const bvhs = await Promise.all(oracleCase.meshes.map((mesh) => builder.build(mesh, { triangleTest })));
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

interface SharedEdgeReport {
    name: string;
    rays: number;
    /** Closest-hit rays that reported a miss although they cross the quad. */
    slipped: number;
    /** Any-hit rays that reported "clear" although they cross the quad. */
    slippedAny: number;
    byCategory: Record<string, { rays: number; slipped: number }>;
    /** Rays among the first few thousand for which the double-precision reference found no hit: must be 0. */
    referenceMisses: number;
    firstSlippedRays: number[];
}

async function runSharedEdgeCase(
    device: GPUDevice,
    kernel: TraceBatchKernel,
    builder: MeshBvhBuilder,
    oracleCase: SharedEdgeCase,
    triangleTest: TriangleTest,
): Promise<SharedEdgeReport> {
    const built = await buildScene(builder, oracleCase, triangleTest);
    const buffers = uploadTraceScene(device, built.data, oracleCase.name);
    const answers = await traceOnGpu(device, kernel, buffers, oracleCase.rays);
    buffers.destroy();

    const rayCount = oracleCase.rays.length / 8;
    const report: SharedEdgeReport = {
        name: oracleCase.name,
        rays: rayCount,
        slipped: 0,
        slippedAny: 0,
        byCategory: {},
        referenceMisses: 0,
        firstSlippedRays: [],
    };
    for (const name of SHARED_EDGE_CATEGORIES) report.byCategory[name] = { rays: 0, slipped: 0 };

    for (let index = 0; index < rayCount; index++) {
        const entry = report.byCategory[SHARED_EDGE_CATEGORIES[oracleCase.categories[index]]];
        entry.rays++;
        // Every ray crosses the quad, so a miss is a slip.
        if (hitAt(answers, index) === null) {
            entry.slipped++;
            report.slipped++;
            if (report.firstSlippedRays.length < 5) report.firstSlippedRays.push(index);
        }
        if (answers.blocked[index] === 0) report.slippedAny++;
    }

    // The setup itself: the double-precision reference finds a hit for a sample of the rays.
    const reference = buildReferenceScene(oracleCase.meshes, oracleCase.instances);
    for (let index = 0; index < Math.min(rayCount, SHARED_EDGE_REFERENCE_RAYS); index++) {
        if (traceClosestReference(reference, unpackRay(oracleCase.rays, index)) === null) report.referenceMisses++;
    }
    return report;
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
        leaks: result.leaks,
        cornerLeaks: result.edgeLeaks,
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
        const inline = buildMeshBvh(sample.meshes[0]);
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

    for (const triangleTest of TRIANGLE_TESTS) {
        const kernel = await TraceBatchKernel.create(device, triangleTest);
        const results = { cases: {} as Record<string, unknown>, sharedEdges: [] as SharedEdgeReport[] };
        byTest[triangleTest] = results;

        for (const [caseIndex, oracleCase] of cases.entries()) {
            const built = await buildScene(builder, oracleCase, triangleTest);
            const buffers = uploadTraceScene(device, built.data, oracleCase.name);
            const answers = await traceOnGpu(device, kernel, buffers, oracleCase.rays);
            buffers.destroy();

            const result = evaluateCase(
                oracleCase,
                { closest: (index) => hitAt(answers, index), any: (index) => answers.blocked[index] !== 0 },
                references[caseIndex],
            );
            const stackUsed = probeStackDepth(built.data, oracleCase);
            deepestStackUsed = Math.max(deepestStackUsed, stackUsed);
            deepestRequired = Math.max(deepestRequired, built.data.requiredStackSize);
            results.cases[oracleCase.name] = summarize(result, stackUsed, built.data, built.buildMs);
            log(
                `[${triangleTest}] ${oracleCase.name}: ${result.rays} rays, closest ${result.closest.mismatches} mismatches / ${result.closest.ambiguous} ambiguous, ` +
                    `any ${result.any.mismatches} / ${result.any.ambiguous}, leaks ${result.leaks}, corner leaks ${result.edgeLeaks}`,
            );
            for (const failure of result.failures) failures.push(`[${triangleTest}] ${oracleCase.name}: ${failure}`);
        }

        for (const transformed of [false, true]) {
            const report = await runSharedEdgeCase(device, kernel, builder, sharedEdgeCase(sharedRays, transformed), triangleTest);
            results.sharedEdges.push(report);
            log(`[${triangleTest}] ${report.name}: ${report.slipped} of ${report.rays} rays slipped between triangles`);
            // Moller-Trumbore is known to let some edge-aimed rays through (the design expects this and
            // names the watertight test as the fix), so its count is reported, not failed. The
            // watertight test has to be perfect.
            if (triangleTest === 'watertight' && report.slipped > 0) {
                failures.push(`[${triangleTest}] ${report.name}: ${report.slipped} rays slipped between triangles`);
            }
            // Shadow rays go through the any-hit walk, which has its own shortcuts; it must not leak either.
            if (triangleTest === 'watertight' && report.slippedAny > 0) {
                failures.push(`[${triangleTest}] ${report.name}: ${report.slippedAny} any-hit rays slipped between triangles`);
            }
            if (report.referenceMisses > 0) {
                failures.push(`[${triangleTest}] ${report.name}: the reference missed ${report.referenceMisses} rays, so the setup is wrong`);
            }
        }
    }
    builder.dispose();

    const limits = device.limits;
    assert(failures.length === 0, failures.slice(0, 12).join('\n'));
    const slippedTotals: Record<string, number> = {};
    for (const triangleTest of TRIANGLE_TESTS) {
        slippedTotals[triangleTest] = byTest[triangleTest].sharedEdges.reduce((sum, report) => sum + report.slipped, 0);
    }
    return {
        sharedEdgeRaysPerTest: byTest[TRIANGLE_TESTS[0]].sharedEdges.reduce((sum, report) => sum + report.rays, 0),
        slippedTotals,
        byTest,
        traceStorageBuffers: TRACE_MODULE_ENTRIES.length,
        batchKernelStorageBuffers: TRACE_MODULE_ENTRIES.length + 2,
        maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
        deepestStackUsedByCpuPort: deepestStackUsed,
        deepestStackRequired: deepestRequired,
    };
});
