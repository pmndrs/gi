/**
 * Runs the oracle cases through the CPU port of the traversal, with no GPU.
 * If these fail, the builder or the traversal logic is wrong; the GPU page
 * then checks that the WGSL does what the port does.
 */

import { describe, expect, it } from 'vitest';
import {
    assembleTraceScene,
    createCpuTracer,
    newTraceStats,
} from '../../../../packages/core/src/platform/trace/index.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../packages/core/src/platform/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '@pmndrs/bvh';
import { buildReferenceScene, traceClosestReference } from '../../../../packages/core/src/platform/trace/reference-tracer.ts';
import { type TracedHit } from '../../../../packages/core/src/platform/trace/oracle-compare.ts';
import { buildOracleCases, sharedEdgeCase, SMALL_SIZES, unpackRay, type OracleCase } from './cases.ts';
import { computeReferenceAnswers, evaluateCase } from './check.ts';

// Mesh trees come from tinybvh (WASM), the only mesh builder since B0.
const tinyBvh = await createTinyBvhBuilder();

function traceOnCpu(oracleCase: OracleCase, reference: ReturnType<typeof computeReferenceAnswers>) {
    const scene = assembleTraceScene(
        oracleCase.meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)),
        oracleCase.instances,
    );
    const tracer = createCpuTracer(scene);
    const stats = newTraceStats();
    const result = evaluateCase(
        oracleCase,
        {
            closest(index): TracedHit | null {
                const hit = tracer.closest(unpackRay(oracleCase.rays, index), stats);
                return hit && { hit: true, ...hit };
            },
            any: (index) => tracer.any(unpackRay(oracleCase.rays, index)),
        },
        reference,
    );
    return { result, stats, scene };
}

describe('oracle cases on the CPU traversal', () => {
    for (const oracleCase of buildOracleCases(SMALL_SIZES)) {
        it(`${oracleCase.name}: ${oracleCase.description}`, { timeout: 60_000 }, () => {
            const reference = computeReferenceAnswers(oracleCase);
            const { result, stats, scene } = traceOnCpu(oracleCase, reference);
            expect(result.failures, result.failures.join('\n')).toEqual([]);
            expect(stats.maxStackSize).toBeLessThanOrEqual(scene.requiredStackSize);
        });
    }
});

describe('shared edges on the CPU traversal', () => {
    for (const transformed of [false, true]) {
        it(`no ray slips between triangles (${transformed ? 'transformed' : 'plain'} quad)`, { timeout: 60_000 }, () => {
            const oracleCase = sharedEdgeCase(20000, transformed);
            const scene = assembleTraceScene(
                oracleCase.meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)),
                oracleCase.instances,
            );
            const tracer = createCpuTracer(scene);
            const reference = buildReferenceScene(oracleCase.meshes, oracleCase.instances);
            let slipped = 0;
            let referenceMisses = 0;
            for (let index = 0; index < 20000; index++) {
                const ray = unpackRay(oracleCase.rays, index);
                if (tracer.closest(ray) === null) slipped++;
                // The setup itself: every ray really does cross the quad.
                if (index < 2000 && traceClosestReference(reference, ray) === null) referenceMisses++;
            }
            expect(referenceMisses).toBe(0);
            // Doubles leave far more room than f32; the GPU page reports the f32 count.
            expect(slipped).toBe(0);
        });
    }
});
