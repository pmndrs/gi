/**
 * T4.2: materials in the walk, through the platform on this GPU, against the
 * brute-force CPU reference with the same materials
 * (docs/design/m4-materials.md §3).
 *
 * 1. **Agreement** (done-when 1 of both halves). Each case goes through the
 *    scene API (textures as `PixelData`, meshes with their uv sets),
 *    `prepare`, frames, and `rays.encodeBatch` in closest-hit and any-hit
 *    mode. Hits, misses and the closest-hit triangle must agree with the
 *    reference on at least 99.9% of the rays the reference is certain of:
 *    rays within a texel of an alpha edge (at the alpha-test mip), and rays
 *    a stochastic surface decided, are left out and counted. The cases:
 *    - the first half's (`buildMaterialCases`): a glass pane in an opaque
 *      wall mesh, a single-sided quad hit from both sides, one mesh with an
 *      opaque and a see-through instance;
 *    - the second half's (`buildAlphaCases`): a wall cut out by a 16 x 16
 *      checkerboard (one texel = 1/16 of the wall), a cutout quad through uv
 *      set 1, a turned, scaled, offset transform and mirror wrap (and a
 *      mirrored instance), and a stochastic quad at alpha 0.25.
 *    The stochastic quad is judged on the share of the rays meeting it that
 *    pass, against the reference's hash twin: within 1 percentage point over
 *    `STOCHASTIC_RAYS` rays. The reference reads alpha from store layers the
 *    CPU makes from the cases' own pixels, never from the GPU's.
 *    Run on the GPU's H-PLOC trees (on a device with subgroups) and on
 *    tinybvh's (the fallback, `gpuFirst: false`): the flags must be right
 *    on both install paths. Parts 2 and 3 run on the trees hosts get
 *    (the GPU's where there are subgroups).
 * 2. **Run-time changes** (first half's done-when 4, and the review's "only
 *    when the walk sees it"): a material edited from opaque to see-through,
 *    made single-sided, an instance given a see-through material
 *    (`setMaterials`), the checker wall from opaque to cutout; after one
 *    frame the flags are rewritten and the answers agree for the new
 *    materials. An emissive edit of the cutout rewrites nothing.
 *    `stats.recompiles` stays 0 throughout.
 * 3. **Cost** (noted, not gated): any-hit rays/s with flagged triangles
 *    against unflagged ones, on the glass scene (pane see-through or opaque)
 *    and the checker wall (cutout or opaque), GPU timestamps when the device
 *    has them.
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { InstanceHandle, MaterialHandle, TextureHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import type { MaterialDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import type { CpuStoreLayer } from '../../../../packages/core/src/platform/textures/texture-store-cpu.ts';
import { HIT_RECORD_BYTES, RAY_BYTES, packRays, type TraceRay } from '../../../../packages/core/src/platform/trace/index.ts';
import type { WalkMaterial } from '../../../../packages/core/src/platform/trace/material-test-wgsl.ts';
import { addToTally, compareAny, compareClosest, newTally, type ComparisonTally, type TracedHit } from '../../../../packages/core/src/platform/trace/oracle-compare.ts';
import {
    buildAlphaCases,
    buildMaterialCases,
    createRandom,
    materialLookup,
    OPAQUE,
    SEE_THROUGH,
    SINGLE_SIDED,
    slotRuns,
    toFloat32Ray,
    uvLookup,
    type MaterialCase,
} from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReferenceWithMiss } from '../../../../packages/core/src/platform/trace/reference-tracer.ts';
import { assert, runGpuTest } from '../../client.ts';
import { IDENTITY_MATRIX } from '../trace-bench/gltf-geometry.ts';
import { createDispatchTimer } from '../trace-bench/dispatch-timer.ts';

/** Rays per case for the agreement check. */
const RAYS_PER_CASE = 50_000;
/** Rays of the stochastic case: its bar is a share of them (done-when 1: 100k rays). */
const STOCHASTIC_RAYS = 100_000;
/** The stochastic quad's alpha (done-when 1). */
const STOCHASTIC_ALPHA = 0.25;
/** The done-when: hits, misses and closest-hit triangles agree on at least this share of the rays the reference is certain of. */
const MIN_AGREEMENT = 0.999;
/** The stochastic quad's pass share against the reference's hash twin: within 1 percentage point. */
const MAX_PASS_RATE_DIFFERENCE = 0.01;
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
    readonly textureHandles: Map<CpuStoreLayer, TextureHandle>;
    readonly instances: InstanceHandle[];
    readonly caseIndexBySlot: Map<number, number>;
}

/** The descriptor a walk material stands for; `textureFor` gives the platform texture of a case texture's layer. */
function descriptorOf(material: WalkMaterial, textureFor: (layer: CpuStoreLayer) => TextureHandle | undefined): MaterialDescriptor {
    const texture = material.albedoTexture;
    return {
        albedo: [0.5, 0.5, 0.5],
        blocksLight: material.blocksLight,
        doubleSided: material.doubleSided,
        alphaMode: material.alphaMode ?? 'opaque',
        alphaCutoff: material.alphaMode === 'cutout' ? material.alphaCutoff : undefined,
        opacity: material.opacity ?? 1,
        albedoTexture: texture ? textureFor(texture.layer) : undefined,
        albedoUv: texture?.uv,
    };
}

function frame(device: GPUDevice, platform: GIPlatformImpl): void {
    const encoder = device.createCommandEncoder({ label: 'trace-materials frame' });
    platform.encode(encoder, IDLE_FRAME);
    device.queue.submit([encoder.finish()]);
}

/**
 * Puts a case on a new platform and waits until every mesh has the trees
 * `trees` names and every texture its layer. `materials` stands in for the
 * case's own lists; each distinct material object becomes one platform
 * material, so editing it edits every instance that uses it.
 */
async function placeCase(device: GPUDevice, materialCase: MaterialCase, trees: TreeSource, materials = materialCase.materials): Promise<PlacedCase> {
    const platform = new GIPlatformImpl(device, {
        label: `trace-materials ${materialCase.name}`,
        rayBatches: true,
        meshTrees: trees === 'gpu' ? {} : { gpuFirst: false },
    });
    const textureHandles = new Map<CpuStoreLayer, TextureHandle>();
    for (const texture of materialCase.textures ?? []) {
        const pixels = texture.pixels;
        textureHandles.set(texture.layer, platform.scene.createTexture({ source: { data: pixels.data, width: pixels.width, height: pixels.height }, srgb: texture.srgb, label: texture.name }));
    }
    const materialHandles = new Map<WalkMaterial, MaterialHandle>();
    const handleFor = (material: WalkMaterial): MaterialHandle => {
        let handle = materialHandles.get(material);
        if (handle === undefined) {
            handle = platform.scene.createMaterial(descriptorOf(material, (layer) => textureHandles.get(layer)));
            materialHandles.set(material, handle);
        }
        return handle;
    };
    const meshes = materialCase.meshes.map((mesh, meshIndex) => {
        assert(mesh.indices, `${materialCase.name}: mesh ${meshIndex} is not indexed`);
        const [uvs, uvs1] = materialCase.uvs?.[meshIndex] ?? [null, null];
        return platform.scene.createMesh({
            positions: mesh.positions,
            indices: mesh.indices,
            uvs: uvs ?? undefined,
            uvs1: uvs1 ?? undefined,
            groups: slotRuns(materialCase.triangleSlots[meshIndex]),
        });
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
    frame(device, platform); // uploads the scene and makes the texture layers; the trees start building
    await platform.rays.whenBuilt();
    frame(device, platform); // places the trees (and rewrites their flags)
    await platform.rays.whenBuilt();
    frame(device, platform);
    assert(!platform.rays.building, `${materialCase.name}: trees still building`);
    const counts = platform.rays.meshTreeCounts;
    const meshCount = materialCase.meshes.length;
    if (trees === 'gpu') assert(counts.onGpuTrees === meshCount, `${materialCase.name}: ${counts.onGpuTrees} of ${meshCount} meshes are on GPU trees`);
    else assert(counts.onTinybvhTrees === meshCount, `${materialCase.name}: ${counts.onTinybvhTrees} of ${meshCount} meshes are on tinybvh trees`);
    const textures = platform.sceneCopy.textureCounts();
    assert(textures.built === textureHandles.size && textures.pending === 0, `${materialCase.name}: textures ${JSON.stringify(textures)}`);
    return { platform, materialCase, materialHandles, textureHandles, instances, caseIndexBySlot };
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

interface PassRates {
    /** Rays whose first surface, with every material opaque, is a surface with an alpha rule. */
    readonly met: number;
    /** Share of those the GPU let through, and the reference. */
    readonly gpu: number;
    readonly reference: number;
}

interface Agreement {
    readonly rays: number;
    readonly closest: ComparisonTally;
    readonly any: ComparisonTally;
    /** Rays the reference is not certain of (an alpha edge within a texel, a stochastic decision, an f32 edge or tie): left out of `agreement`. */
    readonly uncertain: number;
    /** Share of the certain rays whose closest hit (or miss) and any-hit answer both agree; 1 when no ray is certain (the stochastic quad: every ray meets it or a backdrop behind it). */
    readonly agreement: number;
    /** Share of every ray, the uncertain ones included, on which both answers agree outright. */
    readonly agreementOfAllRays: number;
    /** Rays whose answer differs from the case with every material opaque: what the material rules changed. */
    readonly raysChangedByMaterials: number;
    readonly passRates: PassRates;
}

/** Whether a material has an alpha rule (cutout or stochastic). */
function hasAlphaRule(material: WalkMaterial): boolean {
    return material.alphaMode === 'cutout' || material.alphaMode === 'stochastic';
}

/** Compares the platform's answers with the reference for `materials`. */
function compare(materialCase: MaterialCase, answers: Answers, materials: MaterialCase['materials']): Agreement {
    const lookup = materialLookup(materialCase, materials);
    const uvs = uvLookup(materialCase);
    const reference = buildReferenceScene(materialCase.meshes, materialCase.instances, lookup, uvs);
    const opaqueReference = buildReferenceScene(materialCase.meshes, materialCase.instances);
    const alphaInstances = new Set(materials.flatMap((list, instance) => (list.some(hasAlphaRule) ? [instance] : [])));
    const closest = newTally();
    const any = newTally();
    let uncertain = 0;
    let certainAgreeing = 0;
    let allAgreeing = 0;
    let changed = 0;
    let met = 0;
    let gpuPassed = 0;
    let referencePassed = 0;
    materialCase.rays.forEach((ray, index) => {
        const label = `ray ${index}`;
        const expected = traceClosestReferenceWithMiss(reference, ray);
        const expectedAny = traceAnyReference(reference, ray);
        const closestComparison = compareClosest(answers.closest[index], expected.hit, ray, expected.missAmbiguous);
        const anyComparison = compareAny(answers.blocked[index], expectedAny);
        addToTally(closest, closestComparison, label);
        addToTally(any, anyComparison, label);
        const agrees = closestComparison.verdict === 'match' && anyComparison.verdict === 'match';
        if (agrees) allAgreeing++;
        const certain = !(expected.hit?.ambiguous ?? expected.missAmbiguous) && !expectedAny.ambiguous;
        if (!certain) uncertain++;
        else if (agrees) certainAgreeing++;

        const opaque = traceClosestReferenceWithMiss(opaqueReference, ray).hit;
        const hit = expected.hit;
        if ((opaque === null) !== (hit === null) || (opaque && hit && (opaque.instanceIndex !== hit.instanceIndex || opaque.triangleIndex !== hit.triangleIndex))) changed++;
        // Pass shares: rays whose first surface is one with an alpha rule.
        if (opaque && alphaInstances.has(opaque.instanceIndex)) {
            met++;
            const gpu = answers.closest[index];
            if (!(gpu && gpu.instanceIndex === opaque.instanceIndex && gpu.triangleIndex === opaque.triangleIndex)) gpuPassed++;
            if (!(hit && hit.instanceIndex === opaque.instanceIndex && hit.triangleIndex === opaque.triangleIndex)) referencePassed++;
        }
    });
    const rays = materialCase.rays.length;
    return {
        rays,
        closest,
        any,
        uncertain,
        agreement: rays > uncertain ? certainAgreeing / (rays - uncertain) : 1,
        agreementOfAllRays: allAgreeing / rays,
        raysChangedByMaterials: changed,
        passRates: { met, gpu: met > 0 ? gpuPassed / met : 0, reference: met > 0 ? referencePassed / met : 0 },
    };
}

function summary(result: Agreement) {
    return {
        rays: result.rays,
        uncertain: result.uncertain,
        agreement: Number(result.agreement.toFixed(6)),
        agreementOfAllRays: Number(result.agreementOfAllRays.toFixed(6)),
        raysChangedByMaterials: result.raysChangedByMaterials,
        passRates: { met: result.passRates.met, gpu: Number(result.passRates.gpu.toFixed(5)), reference: Number(result.passRates.reference.toFixed(5)) },
        closest: { matches: result.closest.matches, ambiguous: result.closest.ambiguous, mismatches: result.closest.mismatches, examples: result.closest.examples },
        any: { matches: result.any.matches, ambiguous: result.any.ambiguous, mismatches: result.any.mismatches, examples: result.any.examples },
    };
}

function describeAgreement(result: Agreement): string {
    const rates = result.passRates.met > 0 ? `; of ${result.passRates.met} rays meeting an alpha surface, GPU passed ${(result.passRates.gpu * 100).toFixed(2)}%, reference ${(result.passRates.reference * 100).toFixed(2)}%` : '';
    return (
        `${(result.agreement * 100).toFixed(3)}% of ${result.rays - result.uncertain} certain rays agree (${result.uncertain} uncertain left out; ${(result.agreementOfAllRays * 100).toFixed(3)}% of all); ` +
        `closest ${result.closest.mismatches} mismatches / ${result.closest.ambiguous} ambiguous, any ${result.any.mismatches} / ${result.any.ambiguous}; ` +
        `${result.raysChangedByMaterials} rays changed by the materials${rates}`
    );
}

/** Rays aimed at an area of the plane z = 0 from both sides, for the cost note. */
function costRays(count: number, target: { readonly x: readonly [number, number]; readonly y: readonly [number, number] }): TraceRay[] {
    const random = createRandom(0xc057);
    const rays: TraceRay[] = [];
    for (let index = 0; index < count; index++) {
        const side = random() < 0.5 ? 1 : -1;
        const origin = [-2 + 4 * random(), 3 * random(), side * (0.2 + 1.6 * random())];
        const aim = [target.x[0] + (target.x[1] - target.x[0]) * random(), target.y[0] + (target.y[1] - target.y[0]) * random(), 0];
        const direction = [aim[0] - origin[0], aim[1] - origin[1], aim[2] - origin[2]];
        const length = Math.hypot(direction[0], direction[1], direction[2]);
        rays.push(toFloat32Ray(origin, direction.map((component) => component / length), 0, 1e6));
    }
    return rays;
}

await runGpuTest(
    async ({ device, log }) => {
        const failures: string[] = [];
        const firstHalf = buildMaterialCases(RAYS_PER_CASE);
        const alphaCases = buildAlphaCases(RAYS_PER_CASE, STOCHASTIC_RAYS, STOCHASTIC_ALPHA);
        const cases = [...firstHalf, ...alphaCases];
        const treeSources: TreeSource[] = device.features.has('subgroups') ? ['tinybvh', 'gpu'] : ['tinybvh'];
        // What hosts get: the GPU's trees with subgroups, the fallback's without (#26).
        const hostTrees: TreeSource = treeSources[treeSources.length - 1];
        const results: Record<string, unknown> = {};

        // 1. Agreement with the reference, on both install paths.
        for (const trees of treeSources) {
            for (const materialCase of cases) {
                const placed = await placeCase(device, materialCase, trees);
                try {
                    const result = compare(materialCase, await trace(device, placed, materialCase.rays), materialCase.materials);
                    const name = `${materialCase.name} (${trees})`;
                    results[name] = { ...summary(result), flagRewrites: placed.platform.rays.triangleFlagRewrites };
                    log(`${name}: ${describeAgreement(result)}`);
                    if (result.agreement < MIN_AGREEMENT) failures.push(`${name}: ${(result.agreement * 100).toFixed(3)}% agree (${[...result.closest.examples, ...result.any.examples].slice(0, 3).join('; ')})`);
                    if (result.raysChangedByMaterials < materialCase.rays.length * 0.05) failures.push(`${name}: the materials changed only ${result.raysChangedByMaterials} rays`);
                    if (materialCase.name === 'stochastic-quad') {
                        const difference = Math.abs(result.passRates.gpu - result.passRates.reference);
                        if (difference > MAX_PASS_RATE_DIFFERENCE) failures.push(`${name}: the GPU passed ${(result.passRates.gpu * 100).toFixed(2)}%, the reference ${(result.passRates.reference * 100).toFixed(2)}%`);
                    }
                } finally {
                    placed.platform.dispose();
                }
            }
        }

        // 2. Run-time changes: flags rewritten when the walk sees the change, answers follow, nothing recompiles.
        const changes: Record<string, unknown> = {};
        const runChange = async (
            name: string,
            materialCase: MaterialCase,
            before: MaterialCase['materials'],
            change: (placed: PlacedCase) => void,
            after: MaterialCase['materials'],
            expectRewrites = true,
        ): Promise<void> => {
            const placed = await placeCase(device, materialCase, hostTrees, before);
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
                if (expectRewrites && rewrites === 0) failures.push(`${name}: the change rewrote no mesh's flags`);
                if (!expectRewrites && rewrites !== 0) failures.push(`${name}: the change rewrote ${rewrites} meshes' flags, but the walk can't see it`);
                if (recompiles !== 0) failures.push(`${name}: ${recompiles} recompiles`);
            } finally {
                placed.platform.dispose();
            }
        };

        const [glass, singleSided, perInstance] = firstHalf;
        const [checkerWall] = alphaCases;
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
                const glassHandle = placed.platform.scene.createMaterial(descriptorOf(SEE_THROUGH, () => undefined));
                placed.platform.scene.setMaterials(placed.instances[1], [placed.materialHandles.get(OPAQUE)!, glassHandle]);
            },
            [[OPAQUE, OPAQUE], [OPAQUE, SEE_THROUGH], [OPAQUE], [OPAQUE]],
        );
        const cutout = checkerWall.materials[0][0];
        // The same textured material with no alpha rule: what the cutout is before the edit, and the cost's unflagged side.
        const texturedOpaque: WalkMaterial = { ...cutout, alphaMode: 'opaque' };
        await runChange(
            'checker wall opaque to cutout (updateMaterial alphaMode)',
            checkerWall,
            [[texturedOpaque], [OPAQUE], [OPAQUE]],
            (placed) => placed.platform.scene.updateMaterial(placed.materialHandles.get(texturedOpaque)!, { alphaMode: 'cutout', alphaCutoff: 0.5 }),
            checkerWall.materials,
        );
        await runChange(
            'cutout given an emissive colour (rewrites nothing)',
            checkerWall,
            checkerWall.materials,
            (placed) => placed.platform.scene.updateMaterial(placed.materialHandles.get(cutout)!, { emissive: [1, 0.5, 0], emissiveIntensity: 2 }),
            checkerWall.materials,
            false,
        );

        // 3. Cost of flagged rays (noted, not gated): any-hit, flagged against unflagged.
        const timer = await createDispatchTimer(device);
        const cost: Record<string, unknown> = { timing: timer.source, rays: COST_RAYS };
        const costScenes: readonly {
            readonly scene: MaterialCase;
            readonly target: { readonly x: readonly [number, number]; readonly y: readonly [number, number] };
            readonly sides: readonly (readonly [string, MaterialCase['materials']])[];
        }[] = [
            {
                scene: glass,
                target: { x: [-1.2, 1.2], y: [0.3, 2.7] },
                sides: [
                    ['glass: flagged (pane see-through)', [[OPAQUE, SEE_THROUGH], [OPAQUE], [OPAQUE]]],
                    ['glass: unflagged (pane opaque)', [[OPAQUE, OPAQUE], [OPAQUE], [OPAQUE]]],
                ],
            },
            {
                scene: checkerWall,
                target: { x: [-1.5, 1.5], y: [0, 3] },
                sides: [
                    ['checker wall: cutout', checkerWall.materials],
                    ['checker wall: opaque (same texture)', [[texturedOpaque], [OPAQUE], [OPAQUE]]],
                ],
            },
        ];
        for (const { scene, target, sides } of costScenes) {
            const rays = costRays(COST_RAYS, target);
            const rayBuffer = device.createBuffer({ label: 'trace-materials cost rays', size: rays.length * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(rayBuffer, 0, packRays(rays) as Float32Array<ArrayBuffer>);
            const blocked = device.createBuffer({ label: 'trace-materials cost blocked', size: rays.length * 4, usage: GPUBufferUsage.STORAGE });
            for (const [side, materials] of sides) {
                const placed = await placeCase(device, scene, hostTrees, materials);
                try {
                    const times: number[] = [];
                    for (let run = 0; run < COST_RUNS + 1; run++) {
                        const ms = await timer.time((encoder, writes) => placed.platform.rays.encodeBatch(encoder, { mode: 'any', rays: rayBuffer, hits: blocked, count: rays.length }, writes));
                        if (run > 0) times.push(ms); // the first is a warm-up
                    }
                    times.sort((a, b) => a - b);
                    const medianMs = times[Math.floor(times.length / 2)];
                    cost[side] = { medianMs: Number(medianMs.toFixed(4)), megaRaysPerSecond: Number((rays.length / medianMs / 1000).toFixed(1)) };
                    log(`cost, any-hit, ${side}: ${medianMs.toFixed(3)} ms, ${(rays.length / medianMs / 1000).toFixed(1)} Mrays/s (${timer.source})`);
                } finally {
                    placed.platform.dispose();
                }
            }
            rayBuffer.destroy();
            blocked.destroy();
        }
        timer.dispose();

        assert(failures.length === 0, failures.join('\n'));
        return { treeSources, agreement: results, changes, cost };
    },
    { optionalFeatures: ['subgroups', 'timestamp-query'] },
);
