import { describe, expect, it } from 'vitest';
import { createCpuTracer, newTraceStats } from './cpu-traversal.ts';
import {
    EMPTY_CHILD_INDEX,
    INSTANCE_FLAG,
    INSTANCE_RECORD_WORDS,
    INSTANCE_WORD,
    NODE_WORD,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
} from './layout.ts';
import { buildMeshBvh } from './mesh-bvh.ts';
import { addToTally, compareAny, compareClosest, newTally } from './oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad } from './oracle-fixtures.ts';
import { buildReferenceScene, traceAnyReference, traceClosestReference } from './reference-tracer.ts';
import { assembleTraceScene, type TraceSceneInstance } from './scene-assembly.ts';
import { buildTopLevel, invertAffine, transformBounds } from './top-level.ts';

function multiplyAffine(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
    // Column-major 4x4 product a * b.
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
        }
    }
    return out;
}

describe('invertAffine', () => {
    it('inverts random rotation/scale/translation transforms', () => {
        const random = createRandom(1);
        for (let i = 0; i < 50; i++) {
            const matrix = randomAffine(random, { extent: 10, mirror: i % 2 === 1 });
            const inverse = invertAffine(matrix);
            expect(inverse).not.toBeNull();
            const rows = inverse!.rows;
            const full = [
                rows[0], rows[4], rows[8], 0,
                rows[1], rows[5], rows[9], 0,
                rows[2], rows[6], rows[10], 0,
                rows[3], rows[7], rows[11], 1,
            ];
            const identity = multiplyAffine(full, matrix);
            for (let k = 0; k < 16; k++) expect(identity[k]).toBeCloseTo(k % 5 === 0 ? 1 : 0, 9);
            expect(inverse!.determinant < 0).toBe(i % 2 === 1);
        }
    });

    it('returns null for a collapsed transform', () => {
        const flat = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        expect(invertAffine(flat)).toBeNull();
    });
});

describe('buildTopLevel', () => {
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const bounds = [-1, -1, -1, 1, 1, 1];

    it('keeps the platform instance index, offsets and mirror flag in each record', () => {
        const random = createRandom(2);
        const instances = Array.from({ length: 40 }, (_, i) => ({
            instanceIndex: 1000 + i * 7,
            nodeOffset: i * 3,
            triangleOffset: i * 11,
            meshBounds: bounds,
            objectToWorld: randomAffine(random, { extent: 30, mirror: i % 3 === 0 }),
        }));
        const top = buildTopLevel(instances);
        const words = new Uint32Array(top.buffer);
        expect(words[TOP_HEADER_WORD.instanceCount]).toBe(40);
        expect(top.instanceCount).toBe(40);
        expect(top.nodeCount).toBe(39);

        const seen = new Set<number>();
        for (let record = 0; record < 40; record++) {
            const base = TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS;
            const source = top.recordOrder[record];
            seen.add(source);
            expect(words[base + INSTANCE_WORD.instanceIndex]).toBe(instances[source].instanceIndex);
            expect(words[base + INSTANCE_WORD.rootNodeOffset]).toBe(instances[source].nodeOffset);
            expect(words[base + INSTANCE_WORD.triangleOffset]).toBe(instances[source].triangleOffset);
            const mirrored = (words[base + INSTANCE_WORD.flags] & INSTANCE_FLAG.mirrored) !== 0;
            expect(mirrored).toBe(source % 3 === 0);
        }
        expect(seen.size).toBe(40);
    });

    it('puts every instance in exactly one leaf and every leaf box around its instance', () => {
        const random = createRandom(3);
        const instances = Array.from({ length: 100 }, (_, i) => ({
            instanceIndex: i,
            nodeOffset: 0,
            triangleOffset: 0,
            meshBounds: bounds,
            objectToWorld: randomAffine(random, { extent: 50 }),
        }));
        const top = buildTopLevel(instances);
        const floats = new Float32Array(top.buffer);
        const words = new Uint32Array(top.buffer);
        const firstNodeWord = TOP_LEVEL_HEADER_WORDS + top.instanceCount * INSTANCE_RECORD_WORDS;
        const leafRecords: number[] = [];
        const stack = [0];
        while (stack.length > 0) {
            const node = stack.pop() as number;
            const base = firstNodeWord + node * 16;
            for (const [minWord, maxWord, indexWord, countWord] of [
                [NODE_WORD.leftMin, NODE_WORD.leftMax, NODE_WORD.leftIndex, NODE_WORD.leftCount],
                [NODE_WORD.rightMin, NODE_WORD.rightMax, NODE_WORD.rightIndex, NODE_WORD.rightCount],
            ]) {
                const index = words[base + indexWord];
                const count = words[base + countWord];
                if (count === 0) {
                    stack.push(index);
                    continue;
                }
                expect(count).toBe(1);
                leafRecords.push(index);
                const expected = new Float32Array(6);
                transformBounds(bounds, instances[top.recordOrder[index]].objectToWorld, expected, 0);
                for (let axis = 0; axis < 3; axis++) {
                    expect(floats[base + minWord + axis]).toBe(expected[axis]);
                    expect(floats[base + maxWord + axis]).toBe(expected[3 + axis]);
                }
            }
        }
        expect(leafRecords.sort((a, b) => a - b)).toEqual(Array.from({ length: 100 }, (_, i) => i));
    });

    it('leaves out hidden, non-GI and collapsed instances, and handles an empty scene', () => {
        const make = (extra: object, instanceIndex: number) => ({
            instanceIndex, nodeOffset: 0, triangleOffset: 0, meshBounds: bounds, objectToWorld: identity, ...extra,
        });
        const collapsed = identity.slice();
        collapsed[0] = 0;
        const top = buildTopLevel([
            make({}, 0),
            make({ visible: false }, 1),
            make({ contributesToGi: false }, 2),
            make({ objectToWorld: collapsed }, 3),
        ]);
        expect(top.instanceCount).toBe(1);
        expect(top.skippedSingular).toBe(1);

        const empty = buildTopLevel([]);
        expect(empty.instanceCount).toBe(0);
        expect(empty.nodeCount).toBe(0);
        expect(new Uint32Array(empty.buffer)[TOP_HEADER_WORD.nodeCount]).toBe(0);
    });

    it('stores a lone instance as a root with an empty right child', () => {
        const top = buildTopLevel([{ instanceIndex: 5, nodeOffset: 0, triangleOffset: 0, meshBounds: bounds, objectToWorld: identity }]);
        const words = new Uint32Array(top.buffer);
        const rootWord = TOP_LEVEL_HEADER_WORDS + INSTANCE_RECORD_WORDS;
        expect(words[rootWord + NODE_WORD.rightIndex]).toBe(EMPTY_CHILD_INDEX);
        expect(words[rootWord + NODE_WORD.leftCount]).toBe(1);
    });
});

/** Runs both tracers over the same rays and returns the tallies. */
function compareCpuTracerWithReference(
    meshes: ReturnType<typeof randomTriangleSoup>[],
    instances: TraceSceneInstance[],
    rays: ReturnType<typeof randomRays>,
) {
    const scene = assembleTraceScene(meshes.map((mesh) => buildMeshBvh(mesh)), instances);
    const tracer = createCpuTracer(scene);
    const reference = buildReferenceScene(meshes, instances);
    const closest = newTally();
    const any = newTally();
    const stats = newTraceStats();
    rays.forEach((ray, index) => {
        const actual = tracer.closest(ray, stats);
        const traced = actual && { hit: true, ...actual };
        addToTally(closest, compareClosest(traced, traceClosestReference(reference, ray), ray), `ray ${index}`);
        addToTally(any, compareAny(tracer.any(ray), traceAnyReference(reference, ray)), `ray ${index}`);
    });
    return { scene, closest, any, stats };
}

describe('CPU traversal against the brute-force reference', () => {
    it('matches on a single mesh with an identity instance', { timeout: 60_000 }, () => {
        const mesh = randomTriangleSoup(createRandom(10), 1500, 10, 1);
        const rays = randomRays(createRandom(11), 3000, [-10, -10, -10, 10, 10, 10], 4);
        const { closest, any } = compareCpuTracerWithReference(
            [mesh],
            [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }],
            rays,
        );
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
        expect(closest.matches).toBeGreaterThan(2900);
    });

    it('matches with several meshes, non-uniform scales, rotations and mirrors, and random ray intervals', { timeout: 60_000 }, () => {
        const random = createRandom(20);
        const meshes = [randomTriangleSoup(random, 400, 3, 0.8), randomTriangleSoup(random, 60, 2, 1), tessellatedQuad(6, 2)];
        const instances: TraceSceneInstance[] = Array.from({ length: 24 }, (_, i) => ({
            meshIndex: i % 3,
            // Instance indices deliberately unrelated to the order they are placed in.
            instanceIndex: 500 - i * 3,
            objectToWorld: randomAffine(random, { extent: 12, mirror: i % 4 === 0 }),
        }));
        const rays = randomRays(createRandom(21), 4000, [-14, -14, -14, 14, 14, 14], 6, { randomInterval: true });
        const { closest, any, scene } = compareCpuTracerWithReference(meshes, instances, rays);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
        expect(closest.matches + closest.ambiguous).toBe(4000);
        // Ambiguous results are only the rare edge-grazing rays.
        expect(closest.ambiguous + any.ambiguous).toBeLessThan(40);
        expect(scene.requiredStackSize).toBeLessThanOrEqual(TRACE_STACK_SIZE);
    });

    it('matches with many copies of one small mesh and reports stack use under the limit', { timeout: 60_000 }, () => {
        const random = createRandom(30);
        const meshes = [randomTriangleSoup(random, 40, 1, 0.5)];
        const instances: TraceSceneInstance[] = Array.from({ length: 150 }, (_, i) => ({
            meshIndex: 0,
            instanceIndex: i,
            objectToWorld: randomAffine(random, { extent: 25, mirror: i % 5 === 0 }),
        }));
        const rays = randomRays(createRandom(31), 1500, [-25, -25, -25, 25, 25, 25], 5);
        const { closest, any, scene, stats } = compareCpuTracerWithReference(meshes, instances, rays);
        expect(closest.mismatches, closest.examples.join('\n')).toBe(0);
        expect(any.mismatches, any.examples.join('\n')).toBe(0);
        expect(stats.maxStackSize).toBeGreaterThan(0);
        expect(stats.maxStackSize).toBeLessThanOrEqual(scene.requiredStackSize);
    });

    it('gives front faces from world-space winding, including for mirrored instances', () => {
        // One triangle facing +Z, seen from above and below, plain and mirrored in x.
        const mesh = { positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]) };
        const plain = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        const mirror = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1];
        const scene = assembleTraceScene([buildMeshBvh(mesh)], [
            { meshIndex: 0, instanceIndex: 0, objectToWorld: plain },
            { meshIndex: 0, instanceIndex: 1, objectToWorld: mirror },
        ]);
        const tracer = createCpuTracer(scene);
        const down = { direction: [0, 0, -1] as const, tMin: 0, tMax: 100 };
        const up = { direction: [0, 0, 1] as const, tMin: 0, tMax: 100 };
        expect(tracer.closest({ ...down, origin: [0, 0, 5] })?.frontFace).toBe(true);
        expect(tracer.closest({ ...up, origin: [0, 0, -5] })?.frontFace).toBe(false);
        expect(tracer.closest({ ...down, origin: [10, 0, 5] })?.frontFace).toBe(false);
        expect(tracer.closest({ ...up, origin: [10, 0, -5] })?.frontFace).toBe(true);
    });

    it('refuses a scene whose trees could overflow the stack', () => {
        // 70 instances in a line build a top-level tree far shallower than 64, so force it
        // with a degenerate mesh depth: the check compares the sum against the stack size.
        const mesh = buildMeshBvh(randomTriangleSoup(createRandom(1), 50, 5, 1));
        const deepMesh = { ...mesh, maxDepth: TRACE_STACK_SIZE };
        expect(() =>
            assembleTraceScene([deepMesh], [{ meshIndex: 0, instanceIndex: 0, objectToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }]),
        ).toThrow(/stack/);
    });
});
