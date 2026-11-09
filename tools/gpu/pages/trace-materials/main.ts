/**
 * T4.2 (first half): see-through and single-sided materials in the walk,
 * through the platform on this GPU, against the brute-force CPU reference
 * with the same materials (docs/design/m4-materials.md §3).
 *
 * 1. **Agreement** (done-when 1): the three material cases
 *    (`buildMaterialCases` in oracle-fixtures.ts: a glass pane in an opaque
 *    wall mesh, a single-sided quad hit from both sides, one mesh with an
 *    opaque and a see-through instance), each through the scene API,
 *    `prepare`, frames, and `rays.encodeBatch` in closest-hit and any-hit
 *    mode. Hits, misses and the closest-hit triangle must agree with the
 *    reference on at least 99.9% of rays. Run on tinybvh's trees (after the
 *    swap) and, on a device with subgroups, on the GPU's H-PLOC first trees
 *    with tinybvh's held back: the flags must be right on both install paths.
 * 2. **Run-time changes** (done-when 4): a material edited from opaque to
 *    see-through, a material made single-sided, and an instance given a
 *    see-through material (`setMaterials`). After one frame the flags are
 *    rewritten (`rays.triangleFlagRewrites` grows) and the answers agree with
 *    the reference for the new materials; `stats.recompiles` stays 0.
 * 3. **Cost** (noted, not gated): any-hit rays/s on the glass scene with the
 *    pane see-through (its triangles flagged) against opaque (no flags), GPU
 *    timestamps when the device has them.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { InstanceHandle, MaterialHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { HIT_RECORD_BYTES, RAY_BYTES, packRays, type TraceRay } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import type { WalkMaterial } from '../../../../packages/ddgi-core/src/platform/trace/material-test-wgsl.ts';
import { addToTally, compareAny, compareClosest, newTally, type ComparisonTally, type TracedHit } from '../../../../packages/ddgi-core/src/platform/trace/oracle-compare.ts';
import {
    buildMaterialCases,
    createRandom,
    materialLookup,
    OPAQUE,
    SEE_THROUGH,
    SINGLE_SIDED,
    slotRuns,
    toFloat32Ray,
    type MaterialCase,
} from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from '../../../../packages/ddgi-core/src/platform/trace/reference-tracer.ts';
import { assert, runGpuTest } from '../../client.ts';
import { IDENTITY_MATRIX } from '../trace-bench/gltf-geometry.ts';
import { createDispatchTimer } from '../trace-bench/dispatch-timer.ts';

/** Rays per case for the agreement check. */
const RAYS_PER_CASE = 50_000;
/** The done-when: hits, misses and closest-hit triangles agree on at least this share of rays. */
const MIN_AGREEMENT = 0.999;
/** Any-hit rays for the cost note, and timed dispatches per side (the median is reported). */
const COST_RAYS = 1_000_000;
const COST_RUNS = 7;

const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;
const NO_CASE_INSTANCE = 0xffffffff;

const IDLE_FRAME: FrameInput = {
    deltaTime: 0,
    budget: { units: 0 },
    camera: { position: [0, 0, 0], view: IDENTITY_MATRIX, projection: IDENTITY_MATRIX, near: 0.1, far: 100 },
};

type TreeSource = 'tinybvh' | 'gpu';

/** A case placed on a platform: handles to edit, and the scene-copy slot of each case instance. */
interface PlacedCase {
    readonly platform: GIPlatformImpl;
    readonly materialCase: MaterialCase;
    /** One platform material per distinct WalkMaterial object of the case (OPAQUE, SEE_THROUGH, ...), plus any the page adds. */
    readonly materialHandles: Map<WalkMaterial, MaterialHandle>;
    readonly instances: InstanceHandle[];
    readonly caseIndexBySlot: Map<number, number>;
}

function descriptorOf(material: WalkMaterial) {
    return { albedo: [0.5, 0.5, 0.5] as [number, number, number], blocksLight: material.blocksLight, doubleSided: material.doubleSided };
}

function frame(device: GPUDevice, platform: GIPlatformImpl): void {
    const encoder = device.createCommandEncoder({ label: 'trace-materials frame' });
    platform.encode(encoder, IDLE_FRAME);
    device.queue.submit([encoder.finish()]);
}

/**
 * Puts a case on a new platform and waits until every mesh has the trees
 * `trees` names. `materials` stands in for the case's own lists; each
 * distinct material object becomes one platform material, so editing it
 * edits every instance that uses it.
 */
async function placeCase(device: GPUDevice, materialCase: MaterialCase, trees: TreeSource, materials = materialCase.materials): Promise<PlacedCase> {
    const platform = new GIPlatformImpl(device, {
        label: `trace-materials ${materialCase.name}`,
        rayBatches: true,
        meshTrees: trees === 'gpu' ? { holdRefinements: true } : {},
    });
    const materialHandles = new Map<WalkMaterial, MaterialHandle>();
    const handleFor = (material: WalkMaterial): MaterialHandle => {
        let handle = materialHandles.get(material);
        if (handle === undefined) {
            handle = platform.scene.createMaterial(descriptorOf(material));
            materialHandles.set(material, handle);
        }
        return handle;
    };
    const meshes = materialCase.meshes.map((mesh, meshIndex) => {
        assert(mesh.indices, `${materialCase.name}: mesh ${meshIndex} is not indexed`);
        return platform.scene.createMesh({ positions: mesh.positions, indices: mesh.indices, groups: slotRuns(materialCase.triangleSlots[meshIndex]) });
    });
    const caseIndexBySlot = new Map<number, number>();
    const instances = materialCase.instances.map((instance, position) => {
        const handle = platform.scene.createInstance({
            mesh: meshes[instance.meshIndex],
            materials: materials[position].map(handleFor),
            transform: instance.objectToWorld,
        });
        caseIndexBySlot.set(platform.sceneCopy.instanceSlot(handle), instance.instanceIndex);
        return handle;
    });
    await platform.prepare();
    frame(device, platform); // uploads the scene; the trees start building
    await platform.rays.whenRefined();
    frame(device, platform); // places the trees (and rewrites their flags)
    await platform.rays.whenBuilt();
    frame(device, platform);
    assert(!platform.rays.building, `${materialCase.name}: trees still building`);
    const counts = platform.rays.meshTreeCounts;
    const meshCount = materialCase.meshes.length;
    if (trees === 'gpu') assert(counts.onGpuTrees === meshCount, `${materialCase.name}: ${counts.onGpuTrees} of ${meshCount} meshes are on GPU trees`);
    else assert(counts.onTinybvhTrees === meshCount, `${materialCase.name}: ${counts.onTinybvhTrees} of ${meshCount} meshes are on tinybvh trees`);
    return { platform, materialCase, materialHandles, instances, caseIndexBySlot };
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'trace-materials readback', size: byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'trace-materials readback' });
    encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

interface Answers {
    readonly closest: (TracedHit | null)[];
    readonly blocked: boolean[];
}

/** Traces the case's rays through the platform in both modes; hits name the case's instance indices. */
async function trace(device: GPUDevice, placed: PlacedCase, rays: readonly TraceRay[]): Promise<Answers> {
    const count = rays.length;
    const rayBuffer = device.createBuffer({ label: 'trace-materials rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rayBuffer, 0, packRays(rays) as Float32Array<ArrayBuffer>);
    const hits = device.createBuffer({ label: 'trace-materials hits', size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const blocked = device.createBuffer({ label: 'trace-materials blocked', size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'trace-materials trace' });
    placed.platform.rays.encodeBatch(encoder, { mode: 'closest', rays: rayBuffer, hits, count });
    placed.platform.rays.encodeBatch(encoder, { mode: 'any', rays: rayBuffer, hits: blocked, count });
    device.queue.submit([encoder.finish()]);
    const hitData = await readBuffer(device, hits, count * HIT_RECORD_BYTES);
    const blockedWords = new Uint32Array(await readBuffer(device, blocked, count * 4));
    rayBuffer.destroy();
    hits.destroy();
    blocked.destroy();

    const floats = new Float32Array(hitData);
    const words = new Uint32Array(hitData);
    const closest = rays.map((_, index): TracedHit | null => {
        const base = index * HIT_RECORD_WORDS;
        if (!(floats[base] >= 0)) return null;
        return {
            hit: true,
            t: floats[base],
            instanceIndex: placed.caseIndexBySlot.get(words[base + 1]) ?? NO_CASE_INSTANCE,
            triangleIndex: words[base + 2],
            frontFace: words[base + 3] !== 0,
            u: floats[base + 4],
            v: floats[base + 5],
        };
    });
    return { closest, blocked: Array.from(blockedWords, (word) => word !== 0) };
}

interface Agreement {
    readonly rays: number;
    readonly closest: ComparisonTally;
    readonly any: ComparisonTally;
    /** Share of rays whose closest hit (or miss) and any-hit answer both agree or are excusably ambiguous. */
    readonly agreement: number;
    /** Rays whose answer differs from the case with every material opaque: what the material rules changed. */
    readonly raysChangedByMaterials: number;
}

/** Compares the platform's answers with the reference for `materials`. */
function compare(materialCase: MaterialCase, answers: Answers, materials: MaterialCase['materials']): Agreement {
    const reference = buildReferenceScene(materialCase.meshes, materialCase.instances, materialLookup(materialCase, materials));
    const opaqueReference = buildReferenceScene(materialCase.meshes, materialCase.instances);
    const closest = newTally();
    const any = newTally();
    let disagreeing = 0;
    let changed = 0;
    materialCase.rays.forEach((ray, index) => {
        const closestBefore = closest.mismatches;
        const anyBefore = any.mismatches;
        const expected = traceClosestReference(reference, ray);
        addToTally(closest, compareClosest(answers.closest[index], expected, ray), `ray ${index}`);
        addToTally(any, compareAny(answers.blocked[index], traceAnyReference(reference, ray)), `ray ${index}`);
        if (closest.mismatches > closestBefore || any.mismatches > anyBefore) disagreeing++;
        const opaque = traceClosestReference(opaqueReference, ray);
        if ((opaque === null) !== (expected === null) || (opaque && expected && (opaque.instanceIndex !== expected.instanceIndex || opaque.triangleIndex !== expected.triangleIndex))) changed++;
    });
    return { rays: materialCase.rays.length, closest, any, agreement: 1 - disagreeing / materialCase.rays.length, raysChangedByMaterials: changed };
}

function summary(result: Agreement) {
    return {
        rays: result.rays,
        agreement: Number(result.agreement.toFixed(6)),
        raysChangedByMaterials: result.raysChangedByMaterials,
        closest: { matches: result.closest.matches, ambiguous: result.closest.ambiguous, mismatches: result.closest.mismatches, examples: result.closest.examples },
        any: { matches: result.any.matches, ambiguous: result.any.ambiguous, mismatches: result.any.mismatches, examples: result.any.examples },
    };
}

/** Rays aimed at the glass case's wall from both sides, for the cost note (the case's own rays are fewer). */
function costRays(count: number): TraceRay[] {
    const random = createRandom(0xc057);
    const rays: TraceRay[] = [];
    for (let index = 0; index < count; index++) {
        const side = random() < 0.5 ? 1 : -1;
        const origin = [-2 + 4 * random(), 3 * random(), side * (0.2 + 1.6 * random())];
        const target = [-1.2 + 2.4 * random(), 0.3 + 2.4 * random(), 0];
        const direction = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
        const length = Math.hypot(direction[0], direction[1], direction[2]);
        rays.push(toFloat32Ray(origin, direction.map((component) => component / length), 0, 1e6));
    }
    return rays;
}

await runGpuTest(
    async ({ device, log }) => {
        const failures: string[] = [];
        const cases = buildMaterialCases(RAYS_PER_CASE);
        const treeSources: TreeSource[] = device.features.has('subgroups') ? ['tinybvh', 'gpu'] : ['tinybvh'];
        const results: Record<string, unknown> = {};

        // 1. Agreement with the reference, on both install paths.
        for (const trees of treeSources) {
            for (const materialCase of cases) {
                const placed = await placeCase(device, materialCase, trees);
                try {
                    const result = compare(materialCase, await trace(device, placed, materialCase.rays), materialCase.materials);
                    const name = `${materialCase.name} (${trees})`;
                    results[name] = { ...summary(result), flagRewrites: placed.platform.rays.triangleFlagRewrites };
                    log(
                        `${name}: ${(result.agreement * 100).toFixed(3)}% agree; closest ${result.closest.mismatches} mismatches / ${result.closest.ambiguous} ambiguous, ` +
                            `any ${result.any.mismatches} / ${result.any.ambiguous}; ${result.raysChangedByMaterials} rays changed by the materials`,
                    );
                    if (result.agreement < MIN_AGREEMENT) failures.push(`${name}: ${(result.agreement * 100).toFixed(3)}% agree (${[...result.closest.examples, ...result.any.examples].slice(0, 3).join('; ')})`);
                    if (result.raysChangedByMaterials < materialCase.rays.length * 0.05) failures.push(`${name}: the materials changed only ${result.raysChangedByMaterials} rays`);
                } finally {
                    placed.platform.dispose();
                }
            }
        }

        // 2. Run-time changes: flags rewritten, answers follow, nothing recompiles.
        const changes: Record<string, unknown> = {};
        const runChange = async (
            name: string,
            materialCase: MaterialCase,
            before: MaterialCase['materials'],
            change: (placed: PlacedCase) => void,
            after: MaterialCase['materials'],
        ): Promise<void> => {
            const placed = await placeCase(device, materialCase, 'tinybvh', before);
            try {
                const resultBefore = compare(materialCase, await trace(device, placed, materialCase.rays), before);
                const rewritesBefore = placed.platform.rays.triangleFlagRewrites;
                change(placed);
                frame(device, placed.platform);
                const resultAfter = compare(materialCase, await trace(device, placed, materialCase.rays), after);
                const rewrites = placed.platform.rays.triangleFlagRewrites - rewritesBefore;
                const recompiles = placed.platform.stats.recompiles;
                changes[name] = { before: summary(resultBefore), after: summary(resultAfter), meshesRewritten: rewrites, recompiles };
                log(
                    `${name}: before ${(resultBefore.agreement * 100).toFixed(3)}%, after ${(resultAfter.agreement * 100).toFixed(3)}% agree; ` +
                        `${rewrites} mesh flag rewrites; recompiles ${recompiles}`,
                );
                if (resultBefore.agreement < MIN_AGREEMENT) failures.push(`${name}: ${(resultBefore.agreement * 100).toFixed(3)}% agree before the change`);
                if (resultAfter.agreement < MIN_AGREEMENT) failures.push(`${name}: ${(resultAfter.agreement * 100).toFixed(3)}% agree after the change`);
                if (rewrites === 0) failures.push(`${name}: the change rewrote no mesh's flags`);
                if (recompiles !== 0) failures.push(`${name}: ${recompiles} recompiles`);
            } finally {
                placed.platform.dispose();
            }
        };

        const [glass, singleSided, perInstance] = cases;
        // The pane's own material, opaque at first (no triangle of the scene flagged), then see-through.
        const pane: WalkMaterial = { ...OPAQUE };
        await runChange(
            'pane opaque to see-through (updateMaterial)',
            glass,
            [[OPAQUE, pane], [OPAQUE], [OPAQUE]],
            (placed) => placed.platform.scene.updateMaterial(placed.materialHandles.get(pane)!, { blocksLight: false }),
            [[OPAQUE, SEE_THROUGH], [OPAQUE], [OPAQUE]],
        );
        const quadMaterial: WalkMaterial = { ...OPAQUE };
        await runChange(
            'quad double-sided to single-sided (updateMaterial)',
            singleSided,
            [[quadMaterial], [quadMaterial], [quadMaterial], [OPAQUE], [OPAQUE]],
            (placed) => placed.platform.scene.updateMaterial(placed.materialHandles.get(quadMaterial)!, { doubleSided: false }),
            [[SINGLE_SIDED], [SINGLE_SIDED], [SINGLE_SIDED], [OPAQUE], [OPAQUE]],
        );
        await runChange(
            'instance given a see-through slot (setMaterials)',
            perInstance,
            [[OPAQUE, OPAQUE], [OPAQUE, OPAQUE], [OPAQUE], [OPAQUE]],
            (placed) => {
                const glassHandle = placed.platform.scene.createMaterial(descriptorOf(SEE_THROUGH));
                placed.platform.scene.setMaterials(placed.instances[1], [placed.materialHandles.get(OPAQUE)!, glassHandle]);
            },
            [[OPAQUE, OPAQUE], [OPAQUE, SEE_THROUGH], [OPAQUE], [OPAQUE]],
        );

        // 3. Cost of flagged rays (noted, not gated): any-hit on the glass scene, pane see-through vs opaque.
        const timer = await createDispatchTimer(device);
        const rays = costRays(COST_RAYS);
        const rayBuffer = device.createBuffer({ label: 'trace-materials cost rays', size: rays.length * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(rayBuffer, 0, packRays(rays) as Float32Array<ArrayBuffer>);
        const blocked = device.createBuffer({ label: 'trace-materials cost blocked', size: rays.length * 4, usage: GPUBufferUsage.STORAGE });
        const cost: Record<string, unknown> = { timing: timer.source, rays: rays.length };
        for (const [side, paneMaterial] of [
            ['flagged (pane see-through)', SEE_THROUGH],
            ['unflagged (pane opaque)', OPAQUE],
        ] as const) {
            const placed = await placeCase(device, glass, 'tinybvh', [[OPAQUE, paneMaterial], [OPAQUE], [OPAQUE]]);
            try {
                const times: number[] = [];
                for (let run = 0; run < COST_RUNS + 1; run++) {
                    const ms = await timer.time((encoder, writes) => placed.platform.rays.encodeBatch(encoder, { mode: 'any', rays: rayBuffer, hits: blocked, count: rays.length }, writes));
                    if (run > 0) times.push(ms); // the first is a warm-up
                }
                times.sort((a, b) => a - b);
                const medianMs = times[Math.floor(times.length / 2)];
                cost[side] = { medianMs: Number(medianMs.toFixed(4)), megaRaysPerSecond: Number((rays.length / medianMs / 1000).toFixed(1)) };
                log(`cost, any-hit on the glass scene, ${side}: ${medianMs.toFixed(3)} ms, ${(rays.length / medianMs / 1000).toFixed(1)} Mrays/s (${timer.source})`);
            } finally {
                placed.platform.dispose();
            }
        }
        rayBuffer.destroy();
        blocked.destroy();
        timer.dispose();

        assert(failures.length === 0, failures.join('\n'));
        return { treeSources, agreement: results, changes, cost };
    },
    { optionalFeatures: ['subgroups', 'timestamp-query'] },
);
