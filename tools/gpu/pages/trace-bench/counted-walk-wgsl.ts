/**
 * A bench-only copy of the platform tracer's walk that counts what it does
 * (RC5.4, lever 20: why rays leaving surfaces trace slower than rays from
 * points in the air).
 *
 * The production walk (`packages/ddgi-core/src/platform/trace/wgsl.ts`,
 * `walkWgsl`) has no counters and must not get any: production shaders carry
 * no test hooks. This file repeats its loop step for step and adds counters,
 * reusing every helper of the production trace module (node and triangle
 * loads, the box test, the watertight triangle test, the ray setup), so the
 * only thing copied is the loop itself. The page checks that this copy finds
 * exactly the production kernel's hits on every ray before it reports a
 * count; if the production walk changes, that check fails.
 *
 * Its times mean nothing (it writes ten more words per ray); the bench times
 * the production kernel and uses this one only for the counts.
 */

import { traceModuleWgsl } from '../../../../packages/ddgi-core/src/platform/trace/wgsl.ts';

/** Words per ray the counted kernel writes, in the order of {@link COUNTED_WORD}. */
export const COUNTED_WORDS = 16;

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
} as const;

/** "This ray did not leave from a surface" in the per-ray origin record. */
export const NO_ORIGIN = 0xffffffff;

/**
 * The counted walk for one mode, generated from one template like the
 * production `walkWgsl`, with the same child order (closest: nearer first;
 * any: farther first) and the same culling.
 */
function countedWalkWgsl(kind: 'closest' | 'any'): string {
    const closest = kind === 'closest';
    const name = closest ? 'benchCountedClosest' : 'benchCountedAny';
    const farDeclaration = closest ? 'var farthest = ray.tMax;' : 'let farthest = ray.tMax;';
    const onTriangleHit = closest
        ? `if (counts.hitsAccepted == 0u) {
                        counts.stepsBeforeFirstHit = counts.steps;
                        counts.firstHitIsOrigin = select(0u, 1u, isOriginTriangle);
                    }
                    counts.hitsAccepted += 1u;
                    hitT = candidate.t;
                    hitInstance = instance.instanceIndex;
                    hitTriangle = triangle.originalIndex;
                    farthest = candidate.t;`
        : `counts.stepsBeforeFirstHit = counts.steps;
                    counts.firstHitIsOrigin = select(0u, 1u, isOriginTriangle);
                    counts.hitsAccepted = 1u;
                    hitT = 1.0;
                    return benchFinish(counts, hitT, hitInstance, hitTriangle);`;
    return /* wgsl */ `
fn ${name}(ray: GiRay, origin: vec2u) -> BenchCounts {
    var counts = BenchCounts();
    var hitT = -1.0;
    var hitInstance = 0u;
    var hitTriangle = 0u;
    let header = giTraceTopLevel[0];
    let instanceCount = header.x;
    let topLevelNodeCount = header.y;
    if (topLevelNodeCount == 0u) {
        return benchFinish(counts, hitT, hitInstance, hitTriangle);
    }
    let firstTopLevelNode = GI_TOP_HEADER_VEC4S + instanceCount * GI_TOP_VEC4_PER_RECORD;

    let worldSetup = giMakeRaySetup(ray.origin, ray.direction);
    var setup = worldSetup;

    var insideInstance = false;
    var instance: GiInstance;
    ${farDeclaration}

    var stack: array<u32, GI_STACK_SIZE>;
    var stackSize = 0u;
    var current = giMakeReference(0u, 0u);

    loop {
        counts.steps += 1u;
        if (insideInstance && instance.instanceIndex == origin.x) {
            counts.stepsInOriginInstance += 1u;
        }
        if (insideInstance && counts.hitsAccepted == 0u) {
            counts.meshStepsBeforeFirstHit += 1u;
        }
        counts.maxStack = max(counts.maxStack, stackSize);
        let count = current >> GI_REF_COUNT_SHIFT;
        let index = current & GI_REF_INDEX_MASK;

        if (count == 0u) {
            var node: GiBvhNode;
            if (insideInstance) {
                node = giLoadMeshNode(instance.rootNodeOffset + index);
                counts.meshNodes += 1u;
            } else {
                node = giLoadTopLevelNode(firstTopLevelNode, index);
                counts.topLevelNodes += 1u;
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
            // A box entered at tMin is one the ray starts inside.
            counts.originBoxes += select(0u, 1u, hitLeft && leftEntry <= ray.tMin) + select(0u, 1u, hitRight && rightEntry <= ray.tMin);

            if (hitLeft && hitRight) {
                if (${closest ? 'leftEntry <= rightEntry' : 'leftEntry > rightEntry'}) {
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
            counts.instanceEntries += 1u;
            setup = giMakeRaySetup(
                giTransformPoint(instance, ray.origin),
                giTransformDirection(instance, ray.direction),
            );
            stack[stackSize] = GI_REF_INSTANCE_MARKER;
            stackSize += 1u;
            insideInstance = true;
            current = giMakeReference(0u, 0u);
            continue;
        } else {
            for (var i = 0u; i < count; i += 1u) {
                let triangle = giLoadTriangle(instance.triangleOffset + index + i);
                let isOriginTriangle = instance.instanceIndex == origin.x && triangle.originalIndex == origin.y;
                counts.triangleTests += 1u;
                counts.originTriangleTests += select(0u, 1u, isOriginTriangle);
                let candidate = giIntersectWatertight(triangle, setup, ray.tMin, farthest);
                if (candidate.hit) {
                    ${onTriangleHit}
                }
            }
        }

        loop {
            if (stackSize == 0u) {
                return benchFinish(counts, hitT, hitInstance, hitTriangle);
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

/**
 * The counted batch kernel: the production trace module in group 0, then in
 * group 1 the rays, one origin record per ray (`vec2u`: the instance and
 * triangle the ray leaves from, or {@link NO_ORIGIN}), and the counters out.
 */
export function countedKernelWgsl(mode: 'closest' | 'any'): string {
    return /* wgsl */ `${traceModuleWgsl({ group: 0, firstBinding: 0 })}

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
}

fn benchFinish(countsIn: BenchCounts, t: f32, instanceIndex: u32, triangle: u32) -> BenchCounts {
    var counts = countsIn;
    counts.t = t;
    counts.instance = instanceIndex;
    counts.triangle = triangle;
    if (counts.hitsAccepted == 0u) {
        counts.stepsBeforeFirstHit = counts.steps;
    }
    return counts;
}

${countedWalkWgsl(mode)}

@group(1) @binding(0) var<storage, read> benchRays: array<GiRay>;
@group(1) @binding(1) var<storage, read> benchOrigins: array<vec2u>;
@group(1) @binding(2) var<storage, read_write> benchCounts: array<BenchCounts>;

@compute @workgroup_size(64)
fn benchCounted(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let rayIndex = (workgroup.y * workgroupCount.x + workgroup.x) * 64u + localIndex;
    if (rayIndex >= arrayLength(&benchRays)) {
        return;
    }
    benchCounts[rayIndex] = ${mode === 'closest' ? 'benchCountedClosest' : 'benchCountedAny'}(benchRays[rayIndex], benchOrigins[rayIndex]);
}
`;
}
