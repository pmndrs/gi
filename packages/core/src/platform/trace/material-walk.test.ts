/**
 * Materials in the walk (T4.2), on the CPU: the CPU port of the walk, with
 * the consult flags written by the CPU twin of `gi/triangle-flags`, against
 * the brute-force reference with the same materials. The GPU side of the
 * same check is the `trace-materials` page.
 */

import { createTinyBvhBuilder } from '@pmndrs/bvh';
import { describe, expect, it } from 'vitest';
import { createCpuTracer } from './cpu-traversal.ts';
import { TRIANGLE_WORD, TRIANGLE_WORDS } from './layout.ts';
import { materialLetsRayPass, materialNeedsLookup, TRIANGLE_FLAG_CONSULT_MATERIAL, type WalkMaterial } from './material-test-wgsl.ts';
import { buildMeshBvhWithTinyBvh } from './mesh-bvh-topology.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { buildMaterialCases, materialLookup, OPAQUE, SEE_THROUGH, SINGLE_SIDED, slotRuns, type MaterialCase } from './oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene, type TraceSceneData } from './scene-assembly.ts';
import { instanceSlotMask, slotMaskBit, writeTriangleFlagsOnCpu } from './triangle-flags.ts';

const tinyBvh = await createTinyBvhBuilder();

/** The case's trees, with the consult flags as `gi/triangle-flags` writes them (its CPU twin). */
function flaggedScene(materialCase: MaterialCase, materials = materialCase.materials): TraceSceneData {
    const scene = assembleTraceScene(
        materialCase.meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)),
        materialCase.instances,
    );
    const words = new Uint32Array(scene.triangles);
    materialCase.meshes.forEach((mesh, meshIndex) => {
        let mask = 0;
        materialCase.instances.forEach((instance, position) => {
            if (instance.meshIndex === meshIndex) mask |= instanceSlotMask(materials[position]);
        });
        const triangleCount = (mesh.indices?.length ?? 0) / 3;
        const slots = materialCase.triangleSlots[meshIndex];
        writeTriangleFlagsOnCpu(words, { triangleOffset: scene.meshes[meshIndex].triangleOffset, triangleCount }, mask, (original) => slots[original]);
    });
    return scene;
}

function compareWithReference(materialCase: MaterialCase, scene: TraceSceneData, materials = materialCase.materials) {
    const lookup = materialLookup(materialCase, materials);
    const tracer = createCpuTracer(scene, { materialAt: lookup });
    const reference = buildReferenceScene(materialCase.meshes, materialCase.instances, lookup);
    const closest = newTally();
    const any = newTally();
    materialCase.rays.forEach((ray, index) => {
        const hit = tracer.closest(ray);
        const traced = hit ? { hit: true, t: hit.t, instanceIndex: hit.instanceIndex, triangleIndex: hit.triangleIndex, u: hit.u, v: hit.v, frontFace: hit.frontFace } : null;
        addToTally(closest, compareClosest(traced, traceClosestReference(reference, ray), ray), `ray ${index}`);
        addToTally(any, compareAny(tracer.any(ray), traceAnyReference(reference, ray)), `ray ${index}`);
    });
    return { closest, any };
}

describe('the material rules', () => {
    it('let rays through see-through materials and single-sided back faces only', () => {
        expect(materialLetsRayPass(OPAQUE, true)).toBe(false);
        expect(materialLetsRayPass(OPAQUE, false)).toBe(false);
        expect(materialLetsRayPass(SEE_THROUGH, true)).toBe(true);
        expect(materialLetsRayPass(SEE_THROUGH, false)).toBe(true);
        expect(materialLetsRayPass(SINGLE_SIDED, true)).toBe(false);
        expect(materialLetsRayPass(SINGLE_SIDED, false)).toBe(true);
        expect([OPAQUE, SEE_THROUGH, SINGLE_SIDED].map(materialNeedsLookup)).toEqual([false, true, true]);
    });
});

describe('slot masks (gi/triangle-flags pass 1)', () => {
    it('set one bit per slot that needs a lookup, and the last entry covers every slot after it', () => {
        expect(instanceSlotMask([OPAQUE, OPAQUE])).toBe(0);
        expect(instanceSlotMask([SEE_THROUGH, OPAQUE])).toBe(0b1);
        // The last entry stands in for slots past the list's end too.
        expect(instanceSlotMask([OPAQUE, SEE_THROUGH])).toBe(~0b1 >>> 0);
        expect(instanceSlotMask([OPAQUE, SINGLE_SIDED, OPAQUE])).toBe(0b10);
    });

    it('share bit 31 for slots 31 and up', () => {
        const materials: WalkMaterial[] = Array.from({ length: 40 }, () => OPAQUE);
        materials[35] = SEE_THROUGH;
        const mask = instanceSlotMask(materials);
        expect(mask).toBe(slotMaskBit(31));
        expect(slotMaskBit(35)).toBe(slotMaskBit(31));
        expect(slotMaskBit(5)).toBe(32);
    });

    it('give runs that cover every triangle in order', () => {
        const runs = slotRuns(Uint32Array.from([0, 0, 1, 1, 0]));
        expect(runs).toEqual([
            { start: 0, count: 6, materialSlot: 0 },
            { start: 6, count: 6, materialSlot: 1 },
            { start: 12, count: 3, materialSlot: 0 },
        ]);
    });
});

describe('the CPU walk with materials against the brute-force reference', () => {
    const cases = buildMaterialCases(3000);

    for (const materialCase of cases) {
        it(`agrees on ${materialCase.name}`, () => {
            const scene = flaggedScene(materialCase);
            const words = new Uint32Array(scene.triangles);
            let flagged = 0;
            for (let triangle = 0; triangle < words.length / TRIANGLE_WORDS; triangle++) {
                if (words[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.flags] & TRIANGLE_FLAG_CONSULT_MATERIAL) flagged++;
            }
            expect(flagged).toBeGreaterThan(0);

            const { closest, any } = compareWithReference(materialCase, scene);
            expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
            expect(any.mismatches, any.examples.join('\n')).toBe(0);
            expect(closest.ambiguous).toBeLessThan(closest.rays * 0.01);
        });
    }

    it('would disagree without the flags: the cases test the material rules', () => {
        for (const materialCase of cases) {
            // Trees straight from the builder: every flag clear, so every triangle blocks.
            const unflagged = assembleTraceScene(
                materialCase.meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)),
                materialCase.instances,
            );
            const { closest, any } = compareWithReference(materialCase, unflagged);
            expect(closest.mismatches).toBeGreaterThan(materialCase.rays.length * 0.05);
            expect(any.mismatches).toBeGreaterThan(0);
        }
    });

    it('keeps an opaque instance of a flagged mesh blocking (the per-instance case)', () => {
        const perInstance = cases.find((entry) => entry.name === 'per-instance-see-through');
        expect(perInstance).toBeDefined();
        const lookup = materialLookup(perInstance!);
        // Instance 0 is all opaque; its right half is flagged through instance 1's see-through slot.
        expect(lookup(0, perInstance!.triangleSlots[0].indexOf(1)).blocksLight).toBe(true);
        expect(lookup(1, perInstance!.triangleSlots[0].indexOf(1)).blocksLight).toBe(false);
    });

    it('follows a material edited from opaque to see-through once the flags are rewritten', () => {
        const glass = cases.find((entry) => entry.name === 'glass-pane-in-wall')!;
        const opaquePane = glass.materials.map((list) => list.map(() => OPAQUE));
        const before = flaggedScene(glass, opaquePane);
        const beforeWords = new Uint32Array(before.triangles);
        for (let triangle = 0; triangle < beforeWords.length / TRIANGLE_WORDS; triangle++) {
            expect(beforeWords[triangle * TRIANGLE_WORDS + TRIANGLE_WORD.flags] & TRIANGLE_FLAG_CONSULT_MATERIAL).toBe(0);
        }
        const opaqueResult = compareWithReference(glass, before, opaquePane);
        expect(opaqueResult.closest.mismatches).toBe(0);
        // Rewritten with the edited materials: the pane's triangles are flagged and rays go through.
        const after = flaggedScene(glass);
        const afterResult = compareWithReference(glass, after);
        expect(afterResult.closest.mismatches).toBe(0);
        expect(afterResult.any.mismatches).toBe(0);
    });
});
