/**
 * Checks shared by the CPU tests and the GPU page: a tracer's answers against
 * the hand-worked expectations and against the brute-force reference.
 */

import {
    addToTally,
    buildReferenceScene,
    compareAny,
    compareClosest,
    newTally,
    traceAnyReference,
    traceClosestReference,
    T_TOLERANCE,
    type ComparisonTally,
    type ReferenceHit,
    type ReferenceOcclusion,
    type TracedHit,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { unpackRay, type ExpectedHit, type OracleCase } from './cases.ts';

/** Accepted difference for barycentrics against hand-worked values. */
const BARYCENTRIC_TOLERANCE = 1e-3;

/** Returns a description of what differs from the expectation, or null if it matches. */
export function describeExpectationFailure(actual: TracedHit | null, expected: ExpectedHit): string | null {
    const hit = actual !== null && actual.hit;
    if (hit !== expected.hit) return hit ? `expected a miss, got a hit at t=${actual?.t}` : 'expected a hit, got a miss';
    if (!hit || actual === null) return null;
    const problems: string[] = [];
    if (expected.instanceIndex !== undefined && actual.instanceIndex !== expected.instanceIndex) {
        problems.push(`instance ${actual.instanceIndex}, expected ${expected.instanceIndex}`);
    }
    if (expected.triangleIndex !== undefined && actual.triangleIndex !== expected.triangleIndex) {
        problems.push(`triangle ${actual.triangleIndex}, expected ${expected.triangleIndex}`);
    }
    if (expected.frontFace !== undefined && actual.frontFace !== expected.frontFace) {
        problems.push(`frontFace ${actual.frontFace}, expected ${expected.frontFace}`);
    }
    if (expected.t !== undefined && Math.abs(actual.t - expected.t) > T_TOLERANCE * Math.max(1, expected.t)) {
        problems.push(`t ${actual.t}, expected ${expected.t}`);
    }
    if (expected.u !== undefined && Math.abs(actual.u - expected.u) > BARYCENTRIC_TOLERANCE) {
        problems.push(`u ${actual.u}, expected ${expected.u}`);
    }
    if (expected.v !== undefined && Math.abs(actual.v - expected.v) > BARYCENTRIC_TOLERANCE) {
        problems.push(`v ${actual.v}, expected ${expected.v}`);
    }
    return problems.length > 0 ? problems.join('; ') : null;
}

/** The brute-force reference's answer for every ray of a case, computed once and reused for each tracer variant. */
export interface ReferenceAnswers {
    readonly closest: readonly (ReferenceHit | null)[];
    readonly any: readonly ReferenceOcclusion[];
}

export function computeReferenceAnswers(oracleCase: OracleCase): ReferenceAnswers {
    const scene = buildReferenceScene(oracleCase.meshes, oracleCase.instances);
    const rayCount = oracleCase.rays.length / 8;
    const closest: (ReferenceHit | null)[] = [];
    const any: ReferenceOcclusion[] = [];
    for (let index = 0; index < rayCount; index++) {
        const ray = unpackRay(oracleCase.rays, index);
        closest.push(traceClosestReference(scene, ray));
        any.push(traceAnyReference(scene, ray));
    }
    return { closest, any };
}

/**
 * The same, but hands control back to the browser every few hundred rays, so a
 * page stays responsive (the test runner talks to it over DevTools and gives
 * up on a page that is busy for 30 seconds).
 */
export async function computeReferenceAnswersYielding(oracleCase: OracleCase): Promise<ReferenceAnswers> {
    const scene = buildReferenceScene(oracleCase.meshes, oracleCase.instances);
    const rayCount = oracleCase.rays.length / 8;
    const closest: (ReferenceHit | null)[] = [];
    const any: ReferenceOcclusion[] = [];
    for (let index = 0; index < rayCount; index++) {
        const ray = unpackRay(oracleCase.rays, index);
        closest.push(traceClosestReference(scene, ray));
        any.push(traceAnyReference(scene, ray));
        if (index % 500 === 499) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return { closest, any };
}

export interface CaseResult {
    readonly name: string;
    readonly rays: number;
    readonly closest: ComparisonTally;
    readonly any: ComparisonTally;
    /** Disagreements with the hand-worked expectations (curated cases). */
    readonly expectationFailures: string[];
    /** Ordinary rays that had to hit but reported a miss. Any is a failure. */
    readonly leaks: number;
    /** Edge- or corner-aimed rays that had to hit but missed. Reported, not failed. */
    readonly edgeLeaks: number;
    /** Plain-language failures; empty means the case passed. */
    readonly failures: string[];
}

/**
 * Runs a case's rays through `answers` (the tracer under test) and compares
 * with the hand-worked expectations and with the reference.
 */
export function evaluateCase(
    oracleCase: OracleCase,
    answers: { closest(index: number): TracedHit | null; any(index: number): boolean },
    reference: ReferenceAnswers = computeReferenceAnswers(oracleCase),
): CaseResult {
    const rayCount = oracleCase.rays.length / 8;
    const closest = newTally();
    const any = newTally();
    const expectationFailures: string[] = [];
    let leaks = 0;
    let edgeLeaks = 0;

    for (let index = 0; index < rayCount; index++) {
        const ray = unpackRay(oracleCase.rays, index);
        const curated = oracleCase.curated?.[index];
        const label = curated?.label ?? `ray ${index}`;
        const traced = answers.closest(index);
        const blocked = answers.any(index);

        if (curated) {
            const failure = describeExpectationFailure(traced, curated.expected);
            if (failure) expectationFailures.push(`${label}: ${failure}`);
            // Any-hit must agree with whether the hand-worked answer is a hit.
            if (blocked !== curated.expected.hit) {
                expectationFailures.push(`${label}: any-hit said ${blocked ? 'blocked' : 'clear'}, expected ${curated.expected.hit ? 'blocked' : 'clear'}`);
            }
        }
        if (curated?.onEdge) continue;

        addToTally(closest, compareClosest(traced, reference.closest[index], ray), label);
        addToTally(any, compareAny(blocked, reference.any[index]), label);
        if (oracleCase.everyRayMustHit && !(traced !== null && traced.hit)) {
            if (index >= (oracleCase.edgeAimedFrom ?? rayCount)) edgeLeaks++;
            else leaks++;
        }
    }

    const failures: string[] = [];
    failures.push(...expectationFailures);
    if (leaks > 0) failures.push(`${leaks} rays escaped a closed shell`);
    for (const [mode, tally] of [['closest', closest], ['any', any]] as const) {
        if (tally.mismatches > 0) failures.push(`${mode}: ${tally.mismatches} mismatches, e.g. ${tally.examples.join(' | ')}`);
        const share = tally.ambiguous / Math.max(1, tally.rays);
        if (share > oracleCase.maxAmbiguousShare) {
            failures.push(`${mode}: ${(share * 100).toFixed(2)}% ambiguous, allowed ${(oracleCase.maxAmbiguousShare * 100).toFixed(2)}%`);
        }
    }
    return { name: oracleCase.name, rays: rayCount, closest, any, expectationFailures, leaks, edgeLeaks, failures };
}
