/**
 * Alpha in the walk (T4.2, second half), on the CPU: cutouts and stochastic
 * surfaces in the CPU port of the walk against the brute-force reference,
 * both reading the store layers the CPU makes from the cases' own pixels;
 * the rules' CPU twins; and the flag upkeep the review asked for (each
 * encode's own upload, rewrites only for changes the walk sees). The GPU
 * side of the agreement is the `trace-materials` page.
 */

import { createTinyBvhBuilder } from '@pmndrs/bvh';
import { describe, expect, it } from 'vitest';
import type { ComputeKernel, KernelBuilder } from '../contract/wgsl.ts';
import { PlatformKernelBuilder } from '../kernel-builder.ts';
import { PlatformScene } from '../scene/platform-scene.ts';
import { createFakeGpu, type FakeBuffer, type FakeEncoder } from '../scene/testing/fake-gpu.ts';
import { createCpuTracer } from './cpu-traversal.ts';
import {
    createMaterialTestModule,
    materialAlphaAt,
    materialLetsRayPass,
    materialNeedsLookup,
    stochasticHashValue,
    walkMaterialOf,
    type WalkMaterial,
} from './material-test-wgsl.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { buildAlphaCases, materialLookup, OPAQUE, stochasticMaterial, uvLookup, type MaterialCase } from './oracle-fixtures.ts';
import { PlatformRayQueries } from './ray-queries.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReferenceWithMiss } from './reference-tracer.ts';
import { assembleTraceScene, type TraceSceneData } from './scene-assembly.ts';
import { instanceSlotMask, TriangleFlagsKernels, writeTriangleFlagsOnCpu } from './triangle-flags.ts';

const tinyBvh = await createTinyBvhBuilder();

/** The case's trees, with the consult flags as `gi/triangle-flags` writes them (its CPU twin). */
function flaggedScene(materialCase: MaterialCase): TraceSceneData {
    const scene = assembleTraceScene(
        materialCase.meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)),
        materialCase.instances,
    );
    const words = new Uint32Array(scene.triangles);
    materialCase.meshes.forEach((mesh, meshIndex) => {
        let mask = 0;
        materialCase.instances.forEach((instance, position) => {
            if (instance.meshIndex === meshIndex) mask |= instanceSlotMask(materialCase.materials[position]);
        });
        const slots = materialCase.triangleSlots[meshIndex];
        writeTriangleFlagsOnCpu(words, { triangleOffset: scene.meshes[meshIndex].triangleOffset, triangleCount: (mesh.indices?.length ?? 0) / 3 }, mask, (original) => slots[original]);
    });
    return scene;
}

/** The CPU walk against the reference, and how often the alpha surface (instance `surface`) let a ray through. */
function compareWithReference(materialCase: MaterialCase, surface = 0) {
    const materials = materialLookup(materialCase);
    const uvs = uvLookup(materialCase);
    const tracer = createCpuTracer(flaggedScene(materialCase), { materialAt: materials, uvAt: uvs });
    const reference = buildReferenceScene(materialCase.meshes, materialCase.instances, materials, uvs);
    const opaque = buildReferenceScene(materialCase.meshes, materialCase.instances);
    const closest = newTally();
    const any = newTally();
    const passes = { walk: 0, reference: 0, met: 0 };
    materialCase.rays.forEach((ray, index) => {
        const hit = tracer.closest(ray);
        const traced = hit ? { hit: true, t: hit.t, instanceIndex: hit.instanceIndex, triangleIndex: hit.triangleIndex, u: hit.u, v: hit.v, frontFace: hit.frontFace } : null;
        const expected = traceClosestReferenceWithMiss(reference, ray);
        addToTally(closest, compareClosest(traced, expected.hit, ray, expected.missAmbiguous), `ray ${index}`);
        addToTally(any, compareAny(tracer.any(ray), traceAnyReference(reference, ray)), `ray ${index}`);
        // Rays whose first surface, all opaque, is the alpha surface: did they go through it?
        const first = createFirstHit(opaque, ray);
        if (first !== surface) return;
        passes.met++;
        if (hit?.instanceIndex !== surface) passes.walk++;
        if (expected.hit?.instanceIndex !== surface) passes.reference++;
    });
    return { closest, any, passes };
}

function createFirstHit(scene: ReturnType<typeof buildReferenceScene>, ray: MaterialCase['rays'][number]): number {
    return traceClosestReferenceWithMiss(scene, ray).hit?.instanceIndex ?? -1;
}

describe('the alpha rules (CPU twins)', () => {
    it('resolve a descriptor as the scene copy writes its record', () => {
        expect(walkMaterialOf({ alphaCutoff: 0.3 })).toMatchObject({ alphaMode: 'cutout', alphaCutoff: 0.3, opacity: 1 });
        expect(walkMaterialOf({ alphaMode: 'cutout' })).toMatchObject({ alphaMode: 'cutout', alphaCutoff: 0.5 });
        expect(walkMaterialOf({ alphaMode: 'stochastic', alphaCutoff: 0.3, opacity: 0.4 })).toMatchObject({ alphaMode: 'stochastic', alphaCutoff: 0, opacity: 0.4 });
        expect(walkMaterialOf({})).toMatchObject({ alphaMode: 'opaque', alphaCutoff: 0 });
    });

    it('need a lookup only where alpha can open a hole', () => {
        const [checkerWall] = buildAlphaCases(1);
        const cutout = checkerWall.materials[0][0];
        expect(materialNeedsLookup(cutout)).toBe(true);
        // No texture: a cutout opens holes only if its opacity is below the cutoff.
        expect(materialNeedsLookup({ ...cutout, albedoTexture: undefined })).toBe(false);
        expect(materialNeedsLookup({ ...cutout, albedoTexture: undefined, opacity: 0.4 })).toBe(true);
        expect(materialNeedsLookup({ ...cutout, alphaCutoff: 0 })).toBe(false);
        expect(materialNeedsLookup(stochasticMaterial(1))).toBe(false);
        expect(materialNeedsLookup(stochasticMaterial(0.99))).toBe(true);
        expect(materialNeedsLookup({ ...OPAQUE, alphaMode: 'opaque', albedoTexture: cutout.albedoTexture })).toBe(false);
    });

    it('cut a hole where the texture alpha times the opacity is below the cutoff', () => {
        const [checkerWall] = buildAlphaCases(1);
        const cutout = checkerWall.materials[0][0];
        // The wall's first triangle: corners at uv (0,1) (0.25,1) (0.25,0.75) of the 16 x 16 checker.
        const uvs = uvLookup(checkerWall)(0, 0, 0)!;
        const uvsOf = () => uvs;
        // Weights of corners 1 and 2 that put the hit in the middle of checker texel (0, 15) (opaque: 0 + 15 is odd... clear)
        // and texel (1, 15) (opaque). Texel centres are at ((x + 0.5) / 16, (y + 0.5) / 16).
        const at = (texelX: number, texelY: number): [number, number] => {
            // Solve uv = c0 + u (c1 - c0) + v (c2 - c0) for this triangle: u along +x, v along -y once past u.
            const targetU = (texelX + 0.5) / 16;
            const targetV = (texelY + 0.5) / 16;
            const v = (1 - targetV) / 0.25;
            const u = targetU / 0.25 - v;
            return [u, v];
        };
        const [clearU, clearV] = at(2, 15); // a texel inside the triangle: 2 + 15 is odd, so clear
        const [opaqueU, opaqueV] = at(3, 15); // 3 + 15 even: opaque
        expect(materialAlphaAt(cutout, clearU, clearV, uvsOf)).toBeLessThan(0.1);
        expect(materialAlphaAt(cutout, opaqueU, opaqueV, uvsOf)).toBeGreaterThan(0.9);
        const alpha = (u: number, v: number) => () => materialAlphaAt(cutout, u, v, uvsOf);
        expect(materialLetsRayPass(cutout, true, alpha(clearU, clearV))).toBe(true);
        expect(materialLetsRayPass(cutout, true, alpha(opaqueU, opaqueV))).toBe(false);
        // Half the opacity puts the opaque texel below the cutoff too.
        expect(materialLetsRayPass({ ...cutout, opacity: 0.4 }, true, () => 0.4 * materialAlphaAt(cutout, opaqueU, opaqueV, uvsOf))).toBe(true);
    });

    it('let a share 1 - alpha of stochastic hits through, from a hash that spreads evenly', () => {
        const material: WalkMaterial = stochasticMaterial(0.25);
        let passed = 0;
        const count = 20000;
        for (let index = 0; index < count; index++) {
            const u = Math.fround((index % 141) / 141);
            const v = Math.fround(Math.floor(index / 141) / 145);
            if (materialLetsRayPass(material, true, () => 0.25, () => stochasticHashValue(3, index % 7, u, v))) passed++;
        }
        expect(passed / count).toBeGreaterThan(0.74);
        expect(passed / count).toBeLessThan(0.76);
        // The hash's known values (the WGSL twin must give the same; the GPU page compares rates).
        expect(stochasticHashValue(0, 0, 0, 0)).toBeGreaterThanOrEqual(0);
        expect(stochasticHashValue(0, 0, 0, 0)).toBeLessThan(1);
        expect(stochasticHashValue(1, 2, 0.25, 0.5)).not.toBe(stochasticHashValue(1, 2, 0.5, 0.25));
    });
});

describe('the CPU walk with alpha against the brute-force reference', () => {
    const cases = buildAlphaCases(4000, 20000);

    for (const materialCase of cases.slice(0, 2)) {
        it(`agrees on ${materialCase.name}, rays within a texel of an alpha edge aside`, () => {
            const { closest, any, passes } = compareWithReference(materialCase);
            expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
            expect(any.mismatches, any.examples.join('\n')).toBe(0);
            // The cutout really cuts: a good share of the rays that meet it pass, and a good share don't.
            expect(passes.walk).toBeGreaterThan(passes.met * 0.2);
            expect(passes.walk).toBeLessThan(passes.met * 0.8);
        });
    }

    it('lets the same share through a stochastic quad as the reference, within a point', () => {
        const stochastic = cases[2];
        const { closest, passes } = compareWithReference(stochastic);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(passes.met).toBeGreaterThan(5000);
        const walkRate = passes.walk / passes.met;
        const referenceRate = passes.reference / passes.met;
        expect(Math.abs(walkRate - referenceRate)).toBeLessThan(0.01);
        expect(Math.abs(walkRate - 0.75)).toBeLessThan(0.02);
    });
});

/** A kernel builder whose kernels record nothing: enough to drive `TriangleFlagsKernels.encode` on the fake GPU. */
const RECORDING_KERNELS: KernelBuilder = {
    computeKernel: async (options) =>
        ({
            pipeline: {} as GPUComputePipeline,
            ownLayouts: [{} as GPUBindGroupLayout],
            // The fake encoder records a dispatch only with a pipeline and a group 0 set.
            bind: (pass) => {
                pass.setPipeline({ entryPoint: options.entryPoint } as unknown as GPUComputePipeline);
                pass.setBindGroup(0, { entries: [] } as unknown as GPUBindGroup);
            },
        }) satisfies ComputeKernel,
};

describe('gi/triangle-flags uploads (T4.2 review)', () => {
    it("gives each encode its own ranges and clears the masks in that encode's encoder, with no queue writes", async () => {
        const gpu = createFakeGpu();
        const writes: unknown[] = [];
        const queue = gpu.device.queue as unknown as { writeBuffer: (...args: unknown[]) => void };
        const write = queue.writeBuffer.bind(queue);
        queue.writeBuffer = (...args: unknown[]) => {
            writes.push(args[0]);
            write(...args);
        };
        const sceneModule = new PlatformScene(gpu.device).module;
        const flags = new TriangleFlagsKernels(gpu.device, createMaterialTestModule(sceneModule, sceneModule), 'test');
        await flags.prepare(RECORDING_KERNELS);
        const triangles = gpu.device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE });
        const groups: GPUBindGroupEntry[][] = [];
        const createBindGroup = gpu.device.createBindGroup.bind(gpu.device);
        (gpu.device as unknown as { createBindGroup: unknown }).createBindGroup = (descriptor: GPUBindGroupDescriptor) => {
            groups.push([...descriptor.entries]);
            return createBindGroup(descriptor);
        };

        // Two encodes before one submit, of different meshes.
        const first = gpu.createEncoder();
        flags.encode(first, triangles, [{ meshSlot: 0, triangleOffset: 0, triangleCount: 10 }], 1);
        const second = gpu.createEncoder();
        flags.encode(second, triangles, [{ meshSlot: 3, triangleOffset: 10, triangleCount: 5 }], 1);

        expect(writes).toEqual([]);
        for (const encoder of [first, second]) {
            expect((encoder as unknown as FakeEncoder).commands.some((command) => 'clear' in command)).toBe(true);
        }
        // Each encode's write group binds its own ranges buffer, holding that encode's range.
        const rangesOf = (entries: GPUBindGroupEntry[]) => ((entries[2]?.resource as GPUBufferBinding).buffer as unknown as FakeBuffer).words;
        const writeGroups = groups.filter((entries) => entries.length === 3);
        expect(writeGroups).toHaveLength(2);
        expect(Array.from(rangesOf(writeGroups[0]))).toEqual([0, 0, 10, 0]);
        expect(Array.from(rangesOf(writeGroups[1]))).toEqual([3, 10, 5, 0]);
        // And both still exist when the two are submitted.
        expect(() => gpu.submit(first)).not.toThrow();
        expect(() => gpu.submit(second)).not.toThrow();
        flags.dispose();
    });
});

describe('triangle flag rewrites follow what the walk reads (T4.2 review)', () => {
    it('rewrite a mesh for a change of blocksLight, doubleSided or the alpha fields, not for a colour or emissive edit', async () => {
        const gpu = createFakeGpu();
        Object.assign(gpu.device as unknown as Record<string, unknown>, {
            features: new Set<string>(),
            createBindGroupLayout: (descriptor: GPUBindGroupLayoutDescriptor) => ({ descriptor }),
            createPipelineLayout: (descriptor: GPUPipelineLayoutDescriptor) => ({ descriptor }),
        });
        const scene = new PlatformScene(gpu.device);
        const rays = new PlatformRayQueries(gpu.device, scene, { buildIn: 'main', meshTrees: { gpuFirst: false } });
        await rays.prepare(new PlatformKernelBuilder(gpu.device, 'test'), { batches: false });
        const frame = () => {
            const encoder = gpu.createEncoder();
            rays.update(encoder, scene.encode(encoder));
            gpu.submit(encoder);
        };
        const material = scene.createMaterial({ albedo: [0.5, 0.5, 0.5] });
        const mesh = scene.createMesh({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), indices: new Uint32Array([0, 1, 2]) });
        scene.createInstance({ mesh, materials: [material], transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] });
        frame();
        await rays.whenBuilt();
        frame();
        const rewritesAfter = (change: () => void): number => {
            const before = rays.triangleFlagRewrites;
            change();
            frame();
            return rays.triangleFlagRewrites - before;
        };
        expect(rewritesAfter(() => scene.updateMaterial(material, { albedo: [0.9, 0.1, 0.1] }))).toBe(0);
        expect(rewritesAfter(() => scene.updateMaterial(material, { emissive: [1, 1, 1], emissiveIntensity: 3 }))).toBe(0);
        expect(rewritesAfter(() => scene.updateMaterial(material, { alphaMode: 'stochastic', opacity: 0.5 }))).toBe(1);
        expect(rewritesAfter(() => scene.updateMaterial(material, { emissiveIntensity: 1 }))).toBe(0);
        expect(rewritesAfter(() => scene.updateMaterial(material, { opacity: 0.25 }))).toBe(1);
        expect(rewritesAfter(() => scene.updateMaterial(material, { blocksLight: false }))).toBe(1);
        expect(rewritesAfter(() => scene.updateMaterial(material, { doubleSided: false }))).toBe(1);
        // Another material in its place, the same as far as the walk can tell: nothing to rewrite.
        const twin = scene.createMaterial({ albedo: [0.1, 0.1, 0.1], alphaMode: 'stochastic', opacity: 0.25, blocksLight: false, doubleSided: false });
        const [instance] = [...scene.instanceSnapshot()].map((entry) => entry.handle);
        expect(rewritesAfter(() => scene.setMaterials(instance, [twin]))).toBe(0);
        rays.dispose();
    });
});
