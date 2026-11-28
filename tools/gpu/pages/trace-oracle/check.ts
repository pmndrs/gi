/**
 * The oracle's checks, kept free of GPU code so the CPU tests (`cases.test.ts`),
 * the fault tests (`check.test.ts`) and the GPU page (`main.ts`) all judge a
 * tracer's answers by exactly the same rules.
 *
 * A tracer's answers come in through {@link TracerAnswers}: the page reads them
 * from GPU buffers, the tests from the CPU port of the walk (or from a copy of
 * it with a deliberate fault). Nothing here knows which.
 */

import { type TraceRay } from '../../../../packages/core/src/platform/trace/index.ts';
import {
    buildReferenceScene,
    traceAnyReference,
    traceClosestReference,
    type ReferenceHit,
    type ReferenceOcclusion,
} from '../../../../packages/core/src/platform/trace/reference-tracer.ts';
import {
    addToTally,
    compareAny,
    compareClosest,
    newTally,
    T_TOLERANCE,
    type Comparison,
    type ComparisonTally,
    type TracedHit,
} from '../../../../packages/core/src/platform/trace/oracle-compare.ts';
import { SHARED_EDGE_CATEGORIES, unpackRay, type ExpectedHit, type OracleCase, type SharedEdgeCase } from './cases.ts';

/** Accepted difference for barycentrics against hand-worked values. */
const BARYCENTRIC_TOLERANCE = 1e-3;

/** A tracer's answers for the rays of one case, by ray index. */
export interface TracerAnswers {
    /** Closest hit, or null (or `hit: false`) for a miss. */
    closest(index: number): TracedHit | null;
    /** Any-hit: true when something blocks the ray within its interval. */
    any(index: number): boolean;
}

function isHit(traced: TracedHit | null): traced is TracedHit {
    return traced !== null && traced.hit;
}

/** Returns a description of what differs from the expectation, or null if it matches. */
export function describeExpectationFailure(actual: TracedHit | null, expected: ExpectedHit): string | null {
    const hit = isHit(actual);
    if (hit !== expected.hit) return hit ? `expected a miss, got a hit at t=${actual?.t}` : 'expected a hit, got a miss';
    if (!hit) return null;
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

/** Rays of a closed-shell case that reported a miss, split by how they were aimed. Every one is a failure. */
export interface EscapeCount {
    /** Rays in random directions. */
    ordinary: number;
    /** Rays aimed at edges or corners (from `edgeAimedFrom` on). Counted apart only so the report says where the hole is. */
    edgeAimed: number;
}

export interface CaseResult {
    readonly name: string;
    readonly rays: number;
    readonly closest: ComparisonTally;
    readonly any: ComparisonTally;
    /** Disagreements with the hand-worked expectations (curated cases). */
    readonly expectationFailures: string[];
    /** Closed-shell cases: closest-hit rays that reported a miss. */
    readonly closestEscapes: EscapeCount;
    /** Closed-shell cases: any-hit rays that reported "clear". */
    readonly anyEscapes: EscapeCount;
    /** Closed-shell cases: rays for which the reference itself found no wall, which means the setup is wrong. */
    readonly referenceEscapes: number;
    /** Plain-language failures; empty means the case passed. */
    readonly failures: string[];
}

/**
 * Closest-hit comparison for a ray that starts inside a closed shell.
 *
 * The reference calls a hit ambiguous when it lands within rounding distance of
 * an edge, and then accepts a different triangle. In a closed shell that is a
 * real tie and the only thing that can't be decided: the faces meeting at the
 * edge or corner share the point the ray crosses, so which triangle gets the
 * credit is rounding. The distance is not: every tied face is crossed at the
 * same t. And a closed shell's faces are wound consistently, so an interior
 * ray sees the same side of each of them. So a tie must still agree on t and
 * on the side, and a miss is an escape, never a tie.
 */
function compareClosestInShell(traced: TracedHit | null, reference: ReferenceHit | null, ray: TraceRay): Comparison {
    const comparison = compareClosest(traced, reference, ray);
    if (comparison.verdict !== 'ambiguous') return comparison;
    if (!isHit(traced)) return { verdict: 'mismatch', reason: 'escaped the closed shell' };
    // A reference miss is a setup problem, reported through `referenceEscapes`.
    if (reference === null) return comparison;
    if ((comparison.tError ?? 0) > T_TOLERANCE) {
        return {
            verdict: 'mismatch',
            reason: `t ${traced.t} vs ${reference.t}: tied faces are crossed at the same t`,
            tError: comparison.tError,
        };
    }
    if (traced.frontFace !== reference.frontFace) {
        return {
            verdict: 'mismatch',
            reason: `frontFace ${traced.frontFace} vs ${reference.frontFace}: a closed shell shows one side to an interior ray`,
        };
    }
    return comparison;
}

/**
 * Any-hit for a ray that starts inside a closed shell. The answer is known
 * without the reference: something always blocks the ray. A corner ray the
 * reference calls ambiguous is no exception, so nothing here is ambiguous.
 */
function judgeAnyInShell(blocked: boolean): Comparison {
    return blocked ? { verdict: 'match' } : { verdict: 'mismatch', reason: 'any-hit said clear inside a closed shell' };
}

/**
 * Runs a case's rays through `answers` (the tracer under test) and compares
 * with the hand-worked expectations and with the reference.
 */
export function evaluateCase(
    oracleCase: OracleCase,
    answers: TracerAnswers,
    reference: ReferenceAnswers = computeReferenceAnswers(oracleCase),
): CaseResult {
    const rayCount = oracleCase.rays.length / 8;
    const closedShell = oracleCase.everyRayMustHit === true;
    const edgeAimedFrom = oracleCase.edgeAimedFrom ?? rayCount;
    const closest = newTally();
    const any = newTally();
    const expectationFailures: string[] = [];
    const closestEscapes: EscapeCount = { ordinary: 0, edgeAimed: 0 };
    const anyEscapes: EscapeCount = { ordinary: 0, edgeAimed: 0 };
    let referenceEscapes = 0;

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

        if (closedShell) {
            addToTally(closest, compareClosestInShell(traced, reference.closest[index], ray), label);
            addToTally(any, judgeAnyInShell(blocked), label);
            const aim = index >= edgeAimedFrom ? 'edgeAimed' : 'ordinary';
            if (!isHit(traced)) closestEscapes[aim]++;
            if (!blocked) anyEscapes[aim]++;
            if (reference.closest[index] === null) referenceEscapes++;
        } else {
            addToTally(closest, compareClosest(traced, reference.closest[index], ray), label);
            addToTally(any, compareAny(blocked, reference.any[index]), label);
        }
    }

    const failures: string[] = [...expectationFailures];
    for (const [mode, escapes] of [['closest', closestEscapes], ['any', anyEscapes]] as const) {
        const total = escapes.ordinary + escapes.edgeAimed;
        if (total > 0) {
            failures.push(
                `${mode}: ${total} rays escaped a closed shell (${escapes.ordinary} in random directions, ${escapes.edgeAimed} aimed at edges or corners)`,
            );
        }
    }
    if (referenceEscapes > 0) {
        failures.push(`the reference found no wall for ${referenceEscapes} rays inside a closed shell, so the setup is wrong`);
    }
    for (const [mode, tally] of [['closest', closest], ['any', any]] as const) {
        if (tally.mismatches > 0) failures.push(`${mode}: ${tally.mismatches} mismatches, e.g. ${tally.examples.join(' | ')}`);
        const allowed = oracleCase.maxAmbiguousShare[mode];
        const share = tally.ambiguous / Math.max(1, tally.rays);
        if (share > allowed) {
            failures.push(`${mode}: ${(share * 100).toFixed(2)}% ambiguous, allowed ${(allowed * 100).toFixed(2)}%`);
        }
    }
    return { name: oracleCase.name, rays: rayCount, closest, any, expectationFailures, closestEscapes, anyEscapes, referenceEscapes, failures };
}

// ---------------------------------------------------------------------------
// Shared edges
// ---------------------------------------------------------------------------

export interface SharedEdgeReport {
    name: string;
    rays: number;
    /** Closest-hit rays that reported a miss although they cross the quad. */
    slipped: number;
    /** Any-hit rays that reported "clear" although they cross the quad. */
    slippedAny: number;
    byCategory: Record<string, { rays: number; slipped: number }>;
    /** Sampled rays for which the double-precision reference found no hit: must be 0. */
    referenceMisses: number;
    firstSlippedRays: number[];
}

/**
 * Counts the shared-edge rays that slipped between triangles. Every ray
 * crosses the quad well inside its border, so any miss is a slip, in either
 * mode. There is no tie to excuse here: whichever triangle gets the credit,
 * one of them must be hit. The first `referenceSampleRays` rays also go through
 * the double-precision reference, to show the setup really does cross the quad.
 */
export function evaluateSharedEdgeCase(oracleCase: SharedEdgeCase, answers: TracerAnswers, referenceSampleRays: number): SharedEdgeReport {
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
        if (!isHit(answers.closest(index))) {
            entry.slipped++;
            report.slipped++;
            if (report.firstSlippedRays.length < 5) report.firstSlippedRays.push(index);
        }
        if (!answers.any(index)) report.slippedAny++;
    }

    const reference = buildReferenceScene(oracleCase.meshes, oracleCase.instances);
    for (let index = 0; index < Math.min(rayCount, referenceSampleRays); index++) {
        if (traceClosestReference(reference, unpackRay(oracleCase.rays, index)) === null) report.referenceMisses++;
    }
    return report;
}

/** Plain-language failures for a shared-edge report; empty means it passed. */
export function sharedEdgeFailures(report: SharedEdgeReport): string[] {
    const failures: string[] = [];
    // The watertight test has to be perfect: no ray may slip between triangles.
    if (report.slipped > 0) failures.push(`${report.slipped} rays slipped between triangles`);
    // Shadow rays go through the any-hit walk, which has its own shortcuts; it must not leak either.
    if (report.slippedAny > 0) failures.push(`${report.slippedAny} any-hit rays slipped between triangles`);
    if (report.referenceMisses > 0) failures.push(`the reference missed ${report.referenceMisses} rays, so the setup is wrong`);
    return failures;
}
