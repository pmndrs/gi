/**
 * Shows the oracle's checks can fail. Each test takes a tracer that passes (the
 * CPU port of the walk), breaks one answer the way a real tracer bug would, and
 * expects the check to fail. The faults are injected here, into the answers the
 * checks read; the production shaders carry no test hooks.
 */

import { describe, expect, it } from 'vitest';
import { assembleTraceScene, createCpuTracer } from '../../../../packages/core/src/platform/trace/index.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../packages/core/src/platform/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '../../../../packages/bvh/src/index.ts';
import { type TracedHit } from '../../../../packages/core/src/platform/trace/oracle-compare.ts';
import { buildOracleCases, sharedEdgeCase, SMALL_SIZES, unpackRay, type OracleCase } from './cases.ts';
import {
    computeReferenceAnswers,
    evaluateCase,
    evaluateSharedEdgeCase,
    sharedEdgeFailures,
    type ReferenceAnswers,
    type TracerAnswers,
} from './check.ts';

// Mesh trees come from tinybvh (WASM), the only mesh builder since B0.
const tinyBvh = await createTinyBvhBuilder();

/** Answers from the CPU port of the walk: the tracer that passes every case (see cases.test.ts). */
function cpuAnswers(oracleCase: OracleCase): TracerAnswers {
    const scene = assembleTraceScene(
        oracleCase.meshes.map((mesh) => buildMeshBvhWithTinyBvh(mesh, tinyBvh)),
        oracleCase.instances,
    );
    const tracer = createCpuTracer(scene);
    return {
        closest(index): TracedHit | null {
            const hit = tracer.closest(unpackRay(oracleCase.rays, index));
            return hit && { hit: true, ...hit };
        },
        any: (index) => tracer.any(unpackRay(oracleCase.rays, index)),
    };
}

/** One broken answer: what the closest-hit and/or any-hit result of ray `index` becomes. */
interface Fault {
    readonly index: number;
    readonly closest?: (correct: TracedHit | null) => TracedHit | null;
    readonly any?: boolean;
}

function withFault(answers: TracerAnswers, fault: Fault): TracerAnswers {
    return {
        closest(index) {
            const correct = answers.closest(index);
            return index === fault.index && fault.closest ? fault.closest(correct) : correct;
        },
        any(index) {
            return index === fault.index && fault.any !== undefined ? fault.any : answers.any(index);
        },
    };
}

function requireHit(hit: TracedHit | null): TracedHit {
    if (hit === null || !hit.hit) throw new Error('the fault needs a ray the correct tracer hits');
    return hit;
}

const miss = (): null => null;

const cases = new Map(buildOracleCases(SMALL_SIZES).map((oracleCase) => [oracleCase.name, oracleCase]));
const references = new Map<string, ReferenceAnswers>();

function setup(name: string): { oracleCase: OracleCase; reference: ReferenceAnswers; answers: TracerAnswers } {
    const oracleCase = cases.get(name);
    if (!oracleCase) throw new Error(`no oracle case "${name}"`);
    let reference = references.get(name);
    if (!reference) {
        reference = computeReferenceAnswers(oracleCase);
        references.set(name, reference);
    }
    return { oracleCase, reference, answers: cpuAnswers(oracleCase) };
}

function curatedIndex(oracleCase: OracleCase, label: string): number {
    const index = oracleCase.curated?.findIndex((entry) => entry.label === label) ?? -1;
    if (index < 0) throw new Error(`no curated ray "${label}" in ${oracleCase.name}`);
    return index;
}

/** The failures with `fault` applied, after checking the unbroken tracer passes (so the failure is the fault's). */
function failuresWith(name: string, fault: Fault): string {
    const { oracleCase, reference, answers } = setup(name);
    expect(evaluateCase(oracleCase, answers, reference).failures).toEqual([]);
    return evaluateCase(oracleCase, withFault(answers, fault), reference).failures.join('\n');
}

/** A random-soup ray the reference hits without doubt, so any disagreement is a mismatch. */
function clearRandomHit(): number {
    const { reference } = setup('random-soup');
    const index = reference.closest.findIndex((hit, rayIndex) => hit !== null && !hit.ambiguous && !reference.any[rayIndex].ambiguous);
    expect(index).toBeGreaterThanOrEqual(0);
    return index;
}

/**
 * A thin-box ray aimed at a corner whose reference hit is ambiguous: a tie,
 * the one place the checks excuse a different triangle. Faults here show the
 * excuse covers nothing more.
 */
function tiedCornerRay(): number {
    const { oracleCase, reference } = setup('thin-box');
    const from = oracleCase.edgeAimedFrom ?? Infinity;
    const index = reference.closest.findIndex((hit, rayIndex) => rayIndex >= from && hit !== null && hit.ambiguous);
    expect(index).toBeGreaterThanOrEqual(from);
    return index;
}

describe('the oracle checks fail on a faulty tracer', () => {
    describe('a slipped edge ray', () => {
        const oracleCase = sharedEdgeCase(2000, false);
        const answers = cpuAnswers(oracleCase);

        it('closest-hit miss on a shared edge fails the shared-edge check', () => {
            expect(sharedEdgeFailures(evaluateSharedEdgeCase(oracleCase, answers, 200))).toEqual([]);
            const report = evaluateSharedEdgeCase(oracleCase, withFault(answers, { index: 17, closest: miss }), 200);
            expect(report.slipped).toBe(1);
            expect(report.firstSlippedRays).toEqual([17]);
            expect(sharedEdgeFailures(report).join('\n')).toMatch(/1 rays slipped between triangles/);
        });

        it('any-hit "clear" on a shared edge fails the shared-edge check', () => {
            const report = evaluateSharedEdgeCase(oracleCase, withFault(answers, { index: 17, any: false }), 200);
            expect(report.slippedAny).toBe(1);
            expect(sharedEdgeFailures(report).join('\n')).toMatch(/1 any-hit rays slipped/);
        });

        it('a miss through the seam of a thin wall fails the hand-worked expectation', () => {
            const index = curatedIndex(setup('thin-wall').oracleCase, 'through the diagonal seam');
            expect(failuresWith('thin-wall', { index, closest: miss })).toMatch(/through the diagonal seam: expected a hit, got a miss/);
        });
    });

    describe('a corner escape from the sealed box', () => {
        it('closest-hit miss at a corner fails, though the reference calls the ray ambiguous', () => {
            const index = tiedCornerRay();
            const { oracleCase, reference, answers } = setup('thin-box');
            const result = evaluateCase(oracleCase, withFault(answers, { index, closest: miss }), reference);
            expect(result.closestEscapes).toEqual({ ordinary: 0, edgeAimed: 1 });
            expect(result.failures.join('\n')).toMatch(/closest: 1 rays escaped a closed shell/);
        });

        it('any-hit "clear" at a corner fails and is counted as an escape', () => {
            const index = tiedCornerRay();
            const { oracleCase, reference, answers } = setup('thin-box');
            const result = evaluateCase(oracleCase, withFault(answers, { index, any: false }), reference);
            expect(result.anyEscapes).toEqual({ ordinary: 0, edgeAimed: 1 });
            expect(result.failures.join('\n')).toMatch(/any: 1 rays escaped a closed shell/);
        });

        it('a random-direction escape is counted apart from corner escapes', () => {
            const { oracleCase, reference, answers } = setup('thin-box');
            const result = evaluateCase(oracleCase, withFault(answers, { index: 5, closest: miss, any: false }), reference);
            expect(result.closestEscapes).toEqual({ ordinary: 1, edgeAimed: 0 });
            expect(result.anyEscapes).toEqual({ ordinary: 1, edgeAimed: 0 });
        });
    });

    describe('a wrong closest t', () => {
        const later = (offset: number) => (correct: TracedHit | null): TracedHit => {
            const hit = requireHit(correct);
            return { ...hit, t: hit.t + offset };
        };

        it('fails the hand-worked expectation', () => {
            const index = curatedIndex(setup('analytic').oracleCase, 'nearest');
            expect(failuresWith('analytic', { index, closest: later(0.5) })).toMatch(/nearest: t 3\.5, expected 3/);
        });

        it('fails against the reference', () => {
            expect(failuresWith('random-soup', { index: clearRandomHit(), closest: later(0.05) })).toMatch(/closest: 1 mismatches/);
        });

        it('fails at a corner tie: tied faces are crossed at the same t', () => {
            expect(failuresWith('thin-box', { index: tiedCornerRay(), closest: later(0.01) })).toMatch(/tied faces are crossed at the same t/);
        });
    });

    describe('an any-hit that says "clear" where the reference hits', () => {
        it('fails the hand-worked expectation', () => {
            const index = curatedIndex(setup('analytic').oracleCase, 'nearest');
            expect(failuresWith('analytic', { index, any: false })).toMatch(/nearest: any-hit said clear, expected blocked/);
        });

        it('fails against the reference', () => {
            expect(failuresWith('random-soup', { index: clearRandomHit(), any: false })).toMatch(/any: 1 mismatches/);
        });
    });

    describe('a wrong front-face flag', () => {
        const flipped = (correct: TracedHit | null): TracedHit => {
            const hit = requireHit(correct);
            return { ...hit, frontFace: !hit.frontFace };
        };

        it('fails the hand-worked expectation', () => {
            const index = curatedIndex(setup('analytic').oracleCase, 'back side');
            expect(failuresWith('analytic', { index, closest: flipped })).toMatch(/back side: frontFace true, expected false/);
        });

        it('fails against the reference', () => {
            expect(failuresWith('random-soup', { index: clearRandomHit(), closest: flipped })).toMatch(/closest: 1 mismatches/);
        });

        it('fails at a corner tie: a closed shell shows one side to an interior ray', () => {
            expect(failuresWith('thin-box', { index: tiedCornerRay(), closest: flipped })).toMatch(/shows one side to an interior ray/);
        });
    });
});
