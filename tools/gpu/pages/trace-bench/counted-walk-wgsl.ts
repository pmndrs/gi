/**
 * Bench-only copies of the platform tracer's walk (RC5.4, lever 20: why rays
 * leaving surfaces trace slower than rays from points in the air):
 *
 * - a **counted** walk, which adds counters of what the walk does;
 * - a **tie-break** walk (closest hit, no counters, timed), which changes one
 *   thing: when both child boxes are entered at the same distance (every box
 *   that holds the ray's origin is entered at tMin), it picks the first child
 *   by a rule instead of always the left one.
 *
 * The production walk (`packages/ddgi-core/src/platform/trace/wgsl.ts`,
 * `walkWgsl`) has no counters and must not get any: production shaders carry
 * no test hooks. {@link benchWalkWgsl} repeats its loop line for line,
 * reusing every helper of the production trace module (node and triangle
 * loads, the box test, the watertight triangle test, the ray setup). Two
 * checks keep the copy honest: with counters off and no tie-break its text
 * equals the production function's, comments and whitespace aside
 * (`surface-ray-variants.test.ts`), so traversal order is the same; and the
 * page checks that every copy finds the production kernel's hits.
 *
 * The counted walk's times mean nothing (it writes more words per ray); the
 * bench times the production kernel and the tie-break walk.
 */

import { traceModuleWgsl } from '../../../../packages/ddgi-core/src/platform/trace/wgsl.ts';

/** Words per ray the counted kernel writes, in the order of {@link COUNTED_WORD}. */
export const COUNTED_WORDS = 17;

/** Where each counter sits in a ray's {@link COUNTED_WORDS} words. */
export const COUNTED_WORD = {
    /** The closest hit's `t` (f32 bits), negative for a miss; any-hit: 1.0 blocked, -1.0 clear. */
    t: 0,
    /** The hit's platform instance index. */
    instance: 1,
    /** The hit's triangle within its mesh. */
    triangle: 2,
    /** Iterations of the walk's loop: one per node or leaf processed. */
    steps: 3,
    /** Inner nodes of a mesh tree whose two child boxes were tested. */
    meshNodes: 4,
    /** Inner nodes of the top level whose two child boxes were tested. */
    topLevelNodes: 5,
    /** Instances entered (top-level leaves reached). */
    instanceEntries: 6,
    /** Ray-triangle tests. */
    triangleTests: 7,
    /** Triangle hits that became the new closest hit (closest only). */
    hitsAccepted: 8,
    /** Child boxes the walk went into whose entry was the ray's `tMin`: boxes the origin sits in. */
    originBoxes: 9,
    /** Steps taken before the first accepted hit (all steps when the ray misses). */
    stepsBeforeFirstHit: 10,
    /** Steps taken while inside the instance the ray left from (0 when the origin is not on a surface). */
    stepsInOriginInstance: 11,
    /** Ray-triangle tests on the triangle the ray left from. */
    originTriangleTests: 12,
    /** 1 when the first accepted hit was the triangle the ray left from. */
    firstHitIsOrigin: 13,
    /** Deepest the stack got. */
    maxStack: 14,
    /** Steps while the ray's current closest `t` was still `tMax` (nothing found yet), inside any instance. */
    meshStepsBeforeFirstHit: 15,
    /** Instances entered in which the walk accepted at least one hit (closest) or found the blocker (any). */
    instancesWithHit: 16,
} as const;

/** "This ray did not leave from a surface" in the per-ray origin record. */
export const NO_ORIGIN = 0xffffffff;

/**
 * How the closest-hit walk orders two children entered at the same distance:
 *
 * - `none`: the production rule, the left child first;
 * - `exit-near`: the child the ray leaves first;
 * - `exit-far`: the child the ray leaves last;
 * - `direction`: the child whose centre comes first along the ray's direction.
 */
export const TIE_BREAKS = ['none', 'exit-near', 'exit-far', 'direction'] as const;
export type TieBreak = (typeof TIE_BREAKS)[number];

export interface BenchWalkOptions {
    kind: 'closest' | 'any';
    /** The function's name. */
    name: string;
    /** Count the walk's work (returns `BenchCounts`, takes the ray's origin record). */
    counters: boolean;
    /** Closest hit only; any-hit keeps the production order. */
    tieBreak: TieBreak;
}

/** The child-order condition: true sends the walk into the left child first. */
function leftFirstCondition(kind: 'closest' | 'any', tieBreak: TieBreak): string {
    if (kind === 'any') return 'leftEntry > rightEntry';
    if (tieBreak === 'none') return 'leftEntry <= rightEntry';
    const tie =
        tieBreak === 'direction'
            ? 'dot(node.rightMin + node.rightMax - node.leftMin - node.leftMax, 1.0 / setup.inverseDirection) >= 0.0'
            : `benchBoxExit(node.leftMin, node.leftMax, setup) ${tieBreak === 'exit-near' ? '<=' : '>='} benchBoxExit(node.rightMin, node.rightMax, setup)`;
    return `leftEntry < rightEntry || (leftEntry == rightEntry && ${tie})`;
}

/**
 * One bench walk. With `counters: false` and `tieBreak: 'none'` the text is
 * the production `walkWgsl`'s, comments and whitespace aside; every other
 * line here is an added counter (`count(...)`) or the tie-break condition.
 */
export function benchWalkWgsl(options: BenchWalkOptions): string {
    const { kind, name, counters, tieBreak } = options;
    const closest = kind === 'closest';
    const count = (statement: string): string => (counters ? statement : '');
    const finish = (value: string): string => (counters ? `benchFinish${closest ? 'Closest' : 'Any'}(counts, ${value})` : value);
    const parameters = counters ? 'ray: GiRay, origin: vec2u' : 'ray: GiRay';
    const returnType = counters ? 'BenchCounts' : closest ? 'GiHit' : 'bool';
    const noHit = closest ? 'result' : 'false';
    const resultDeclaration = closest ? 'var result = GiHit(-1.0, 0u, 0u, vec2f(0.0), true);' : '';
    const farDeclaration = closest ? 'var farthest = ray.tMax;' : 'let farthest = ray.tMax;';
    const onTriangleHit = closest
        ? `${count(`if (counts.hitsAccepted == 0u) {
                        counts.stepsBeforeFirstHit = counts.steps;
                        counts.firstHitIsOrigin = select(0u, 1u, isOriginTriangle);
                    }
                    counts.hitsAccepted += 1u;
                    if (!hitInThisInstance) {
                        counts.instancesWithHit += 1u;
                        hitInThisInstance = true;
                    }`)}
                    result = GiHit(candidate.t, instance.instanceIndex, triangle.originalIndex, candidate.barycentrics, candidate.frontFacing);
                    farthest = candidate.t;`
        : `${count(`counts.stepsBeforeFirstHit = counts.steps;
                    counts.firstHitIsOrigin = select(0u, 1u, isOriginTriangle);
                    counts.hitsAccepted = 1u;
                    counts.instancesWithHit += 1u;`)}
                    return ${finish('true')};`;
    return /* wgsl */ `
fn ${name}(${parameters}) -> ${returnType} {
    ${count('var counts = BenchCounts();')}
    ${resultDeclaration}
    let header = giTraceTopLevel[0];
    let instanceCount = header.x;
    let topLevelNodeCount = header.y;
    if (topLevelNodeCount == 0u) {
        return ${finish(noHit)};
    }
    let firstTopLevelNode = GI_TOP_HEADER_VEC4S + instanceCount * GI_TOP_VEC4_PER_RECORD;

    let worldSetup = giMakeRaySetup(ray.origin, ray.direction);
    var setup = worldSetup;

    var insideInstance = false;
    var instance: GiInstance;
    ${count('var hitInThisInstance = false;')}
    ${farDeclaration}

    var stack: array<u32, GI_STACK_SIZE>;
    var stackSize = 0u;
    var current = giMakeReference(0u, 0u);

    loop {
        ${count(`counts.steps += 1u;
        if (insideInstance && instance.instanceIndex == origin.x) {
            counts.stepsInOriginInstance += 1u;
        }
        if (insideInstance && counts.hitsAccepted == 0u) {
            counts.meshStepsBeforeFirstHit += 1u;
        }
        counts.maxStack = max(counts.maxStack, stackSize);`)}
        let count = current >> GI_REF_COUNT_SHIFT;
        let index = current & GI_REF_INDEX_MASK;

        if (count == 0u) {
            var node: GiBvhNode;
            if (insideInstance) {
                node = giLoadMeshNode(instance.rootNodeOffset + index);
                ${count('counts.meshNodes += 1u;')}
            } else {
                node = giLoadTopLevelNode(firstTopLevelNode, index);
                ${count('counts.topLevelNodes += 1u;')}
            }

            let leftEntry = giBoxEntry(node.leftMin, node.leftMax, setup, ray.tMin, farthest);
            var rightEntry = GI_BOX_MISS;
            if (node.rightIndex != GI_EMPTY_CHILD) {
                rightEntry = giBoxEntry(node.rightMin, node.rightMax, setup, ray.tMin, farthest);
            }
            let leftReference = giMakeReference(node.leftIndex, node.leftCount);
            let rightReference = giMakeReference(node.rightIndex, node.rightCount);
            let hitLeft = leftEntry < GI_BOX_MISS;
            let hitRight = rightEntry < GI_BOX_MISS;
            ${count(`// A box entered at tMin is one the ray starts inside.
            counts.originBoxes += select(0u, 1u, hitLeft && leftEntry <= ray.tMin) + select(0u, 1u, hitRight && rightEntry <= ray.tMin);`)}

            if (hitLeft && hitRight) {
                if (${leftFirstCondition(kind, tieBreak)}) {
                    stack[stackSize] = rightReference;
                    current = leftReference;
                } else {
                    stack[stackSize] = leftReference;
                    current = rightReference;
                }
                stackSize += 1u;
                continue;
            }
            if (hitLeft) {
                current = leftReference;
                continue;
            }
            if (hitRight) {
                current = rightReference;
                continue;
            }
        } else if (!insideInstance) {
            instance = giLoadInstance(index);
            ${count('counts.instanceEntries += 1u;')}
            setup = giMakeRaySetup(
                giTransformPoint(instance, ray.origin),
                giTransformDirection(instance, ray.direction),
            );
            stack[stackSize] = GI_REF_INSTANCE_MARKER;
            stackSize += 1u;
            insideInstance = true;
            ${count('hitInThisInstance = false;')}
            current = giMakeReference(0u, 0u);
            continue;
        } else {
            for (var i = 0u; i < count; i += 1u) {
                let triangle = giLoadTriangle(instance.triangleOffset + index + i);
                ${count(`let isOriginTriangle = instance.instanceIndex == origin.x && triangle.originalIndex == origin.y;
                counts.triangleTests += 1u;
                counts.originTriangleTests += select(0u, 1u, isOriginTriangle);`)}
                let candidate = giIntersectWatertight(triangle, setup, ray.tMin, farthest);
                if (candidate.hit) {
                    ${onTriangleHit}
                }
            }
        }

        loop {
            if (stackSize == 0u) {
                return ${finish(noHit)};
            }
            stackSize -= 1u;
            let entry = stack[stackSize];
            if (entry == GI_REF_INSTANCE_MARKER) {
                insideInstance = false;
                setup = worldSetup;
                continue;
            }
            current = entry;
            break;
        }
    }
}
`;
}

/** Bench helpers the walks above call: the box exit distance (tie-breaks) and the counters' record. */
const BENCH_HELPERS_WGSL = /* wgsl */ `
// The distance at which the ray leaves the box (the slab test's far side), for the exit tie-breaks.
fn benchBoxExit(boxMin: vec3f, boxMax: vec3f, setup: GiRaySetup) -> f32 {
    let toMin = (boxMin - setup.origin) * setup.inverseDirection;
    let toMax = (boxMax - setup.origin) * setup.inverseDirection;
    let farSides = max(toMin, toMax);
    return min(farSides.x, min(farSides.y, farSides.z));
}

struct BenchCounts {
    t: f32,
    instance: u32,
    triangle: u32,
    steps: u32,
    meshNodes: u32,
    topLevelNodes: u32,
    instanceEntries: u32,
    triangleTests: u32,
    hitsAccepted: u32,
    originBoxes: u32,
    stepsBeforeFirstHit: u32,
    stepsInOriginInstance: u32,
    originTriangleTests: u32,
    firstHitIsOrigin: u32,
    maxStack: u32,
    meshStepsBeforeFirstHit: u32,
    instancesWithHit: u32,
}

fn benchFinishClosest(countsIn: BenchCounts, hit: GiHit) -> BenchCounts {
    var counts = countsIn;
    counts.t = hit.t;
    counts.instance = hit.instance;
    counts.triangle = hit.triangle;
    if (counts.hitsAccepted == 0u) {
        counts.stepsBeforeFirstHit = counts.steps;
    }
    return counts;
}

fn benchFinishAny(countsIn: BenchCounts, blocked: bool) -> BenchCounts {
    var counts = countsIn;
    counts.t = select(-1.0, 1.0, blocked);
    if (counts.hitsAccepted == 0u) {
        counts.stepsBeforeFirstHit = counts.steps;
    }
    return counts;
}
`;

/** Dispatches are 2D, as the production batch kernel's, so a large batch stays under 65535 workgroups in x. */
const RAY_INDEX_WGSL = 'let rayIndex = (workgroup.y * workgroupCount.x + workgroup.x) * 64u + localIndex;';

/**
 * The counted batch kernel: the production trace module in group 0, then in
 * group 1 the rays, one origin record per ray (`vec2u`: the instance and
 * triangle the ray leaves from, or {@link NO_ORIGIN}), and the counters out.
 */
export function countedKernelWgsl(mode: 'closest' | 'any', tieBreak: TieBreak = 'none'): string {
    return /* wgsl */ `${traceModuleWgsl({ group: 0, firstBinding: 0 })}
${BENCH_HELPERS_WGSL}
${benchWalkWgsl({ kind: mode, name: 'benchCountedWalk', counters: true, tieBreak })}

@group(1) @binding(0) var<storage, read> benchRays: array<GiRay>;
@group(1) @binding(1) var<storage, read> benchOrigins: array<vec2u>;
@group(1) @binding(2) var<storage, read_write> benchCounts: array<BenchCounts>;

@compute @workgroup_size(64)
fn benchCounted(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    ${RAY_INDEX_WGSL}
    if (rayIndex >= arrayLength(&benchRays)) {
        return;
    }
    benchCounts[rayIndex] = benchCountedWalk(benchRays[rayIndex], benchOrigins[rayIndex]);
}
`;
}

/**
 * The tie-break batch kernel for timing: closest hit, no counters, the
 * production batch kernel's bindings and hit record (group 1: rays, hit
 * records), so the bench's timing code runs it as it runs the production one.
 */
export function tieBreakKernelWgsl(tieBreak: TieBreak): string {
    return /* wgsl */ `${traceModuleWgsl({ group: 0, firstBinding: 0 })}
${BENCH_HELPERS_WGSL}
${benchWalkWgsl({ kind: 'closest', name: 'benchTieBreakWalk', counters: false, tieBreak })}

struct BenchHitRecord {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,
    barycentrics: vec2f,
}

@group(1) @binding(0) var<storage, read> benchRays: array<GiRay>;
@group(1) @binding(1) var<storage, read_write> benchResults: array<BenchHitRecord>;

@compute @workgroup_size(64)
fn benchTieBreak(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    ${RAY_INDEX_WGSL}
    if (rayIndex >= arrayLength(&benchRays)) {
        return;
    }
    let hit = benchTieBreakWalk(benchRays[rayIndex]);
    benchResults[rayIndex] = BenchHitRecord(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.barycentrics);
}
`;
}
