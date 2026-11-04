/**
 * A CPU port of the WGSL walk in `wgsl.ts`, step for step.
 *
 * It exists so the builder and the traversal logic (nearer child first,
 * two-level descent, object-space rays, stack use) can be tested in plain Node
 * against the brute-force reference, with no GPU. It also reports the deepest
 * stack the walk used, which the GPU cannot tell us without a debug counter.
 * It runs in doubles, so it matches the GPU's decisions but not its rounding.
 */

import {
    INSTANCE_FLAG,
    INSTANCE_RECORD_WORDS,
    INSTANCE_WORD,
    BVH_NODE_WORDS,
    EMPTY_CHILD_INDEX,
    NODE_WORD,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_WORDS,
    TRACE_STACK_SIZE,
    TRIANGLE_WORD,
    TRIANGLE_WORDS,
    REF_INSTANCE_MARKER,
    makeReference,
    referenceCount,
    referenceIndex,
    type TriangleTest,
} from './layout.ts';
import type { TraceRay } from './reference-tracer.ts';
import type { TraceSceneData } from './scene-assembly.ts';

export interface CpuHit {
    readonly t: number;
    readonly instanceIndex: number;
    readonly triangleIndex: number;
    readonly u: number;
    readonly v: number;
    readonly frontFace: boolean;
}

/** Counters the walk fills in when asked. */
export interface CpuTraceStats {
    maxStackSize: number;
    nodesVisited: number;
    trianglesTested: number;
}

export function newTraceStats(): CpuTraceStats {
    return { maxStackSize: 0, nodesVisited: 0, trianglesTested: 0 };
}

export interface CpuTracer {
    closest(ray: TraceRay, stats?: CpuTraceStats): CpuHit | null;
    any(ray: TraceRay, stats?: CpuTraceStats): boolean;
}

const BOX_MISS = Infinity;
const MIN_DIRECTION = 1e-30;
const BOX_EXIT_PADDING = 1.0000004;

function safeInverse(component: number): number {
    const magnitude = Math.max(Math.abs(component), MIN_DIRECTION);
    return 1 / (component < 0 ? -magnitude : magnitude);
}

/** A ray in the space of the level being walked, with what the triangle tests need precomputed. */
interface RaySetup {
    readonly origin: readonly number[];
    readonly direction: readonly number[];
    readonly inverseDirection: readonly number[];
    /** Watertight test: the axis the ray runs along most (z'), and the two others (x', y'), winding-preserving. */
    readonly axes: readonly [number, number, number];
    /** Watertight test: shear that turns the ray into +z'. */
    readonly shear: readonly [number, number, number];
}

function makeRaySetup(origin: readonly number[], direction: readonly number[]): RaySetup {
    const magnitude = direction.map(Math.abs);
    let kz = 2;
    if (magnitude[0] > magnitude[1] && magnitude[0] > magnitude[2]) kz = 0;
    else if (magnitude[1] > magnitude[2]) kz = 1;
    let kx = (kz + 1) % 3;
    let ky = (kx + 1) % 3;
    // Swapping x' and y' for a ray pointing along -z' keeps triangle winding.
    if (direction[kz] < 0) [kx, ky] = [ky, kx];
    return {
        origin,
        direction,
        inverseDirection: direction.map(safeInverse),
        axes: [kx, ky, kz],
        shear: [direction[kx] / direction[kz], direction[ky] / direction[kz], 1 / direction[kz]],
    };
}

interface TriangleHit {
    t: number;
    u: number;
    v: number;
    frontFace: boolean;
}

export function createCpuTracer(scene: TraceSceneData): CpuTracer {
    const nodeFloats = new Float32Array(scene.nodes);
    const nodeWords = new Uint32Array(scene.nodes);
    const triangleFloats = new Float32Array(scene.triangles);
    const triangleWords = new Uint32Array(scene.triangles);
    const topFloats = new Float32Array(scene.topLevel.buffer);
    const topWords = new Uint32Array(scene.topLevel.buffer);
    const triangleTest: TriangleTest = scene.triangleTest;

    const instanceCount = topWords[TOP_HEADER_WORD.instanceCount];
    const topNodeCount = topWords[TOP_HEADER_WORD.nodeCount];
    const firstTopLevelNodeWord = TOP_LEVEL_HEADER_WORDS + instanceCount * INSTANCE_RECORD_WORDS;

    /** Entry distance into the box stored at `floats[minWord..]`/`floats[maxWord..]`, or BOX_MISS. */
    function boxEntry(
        floats: Float32Array,
        minWord: number,
        maxWord: number,
        setup: RaySetup,
        tMin: number,
        tMax: number,
    ): number {
        let enter = tMin;
        let exit = Infinity;
        for (let axis = 0; axis < 3; axis++) {
            const toMin = (floats[minWord + axis] - setup.origin[axis]) * setup.inverseDirection[axis];
            const toMax = (floats[maxWord + axis] - setup.origin[axis]) * setup.inverseDirection[axis];
            enter = Math.max(enter, Math.min(toMin, toMax));
            exit = Math.min(exit, Math.max(toMin, toMax));
        }
        exit = Math.min(exit * BOX_EXIT_PADDING, tMax);
        return enter <= exit ? enter : BOX_MISS;
    }

    function traverse(ray: TraceRay, stopAtFirstHit: boolean, stats?: CpuTraceStats): CpuHit | null {
        if (topNodeCount === 0) return null;

        const worldSetup = makeRaySetup(ray.origin, ray.direction);
        let setup = worldSetup;

        let insideInstance = false;
        let rootNodeOffset = 0;
        let triangleOffset = 0;
        let instanceIndex = 0;
        let mirrored = false;

        let closestT = ray.tMax;
        let result: CpuHit | null = null;

        const stack: number[] = [];
        let current = makeReference(0, 0);

        for (;;) {
            const count = referenceCount(current);
            const index = referenceIndex(current);
            let descended = false;

            if (count === 0) {
                if (stats) stats.nodesVisited++;
                const floats = insideInstance ? nodeFloats : topFloats;
                const words = insideInstance ? nodeWords : topWords;
                const nodeWord = insideInstance
                    ? (rootNodeOffset + index) * BVH_NODE_WORDS
                    : firstTopLevelNodeWord + index * BVH_NODE_WORDS;

                const leftEntry = boxEntry(floats, nodeWord + NODE_WORD.leftMin, nodeWord + NODE_WORD.leftMax, setup, ray.tMin, closestT);
                const rightIndex = words[nodeWord + NODE_WORD.rightIndex];
                const rightEntry =
                    rightIndex === EMPTY_CHILD_INDEX
                        ? BOX_MISS
                        : boxEntry(floats, nodeWord + NODE_WORD.rightMin, nodeWord + NODE_WORD.rightMax, setup, ray.tMin, closestT);
                const leftReference = makeReference(words[nodeWord + NODE_WORD.leftIndex], words[nodeWord + NODE_WORD.leftCount]);
                const rightReference = makeReference(rightIndex, words[nodeWord + NODE_WORD.rightCount]);
                const hitLeft = leftEntry < BOX_MISS;
                const hitRight = rightEntry < BOX_MISS;

                if (hitLeft && hitRight) {
                    if (leftEntry <= rightEntry) {
                        stack.push(rightReference);
                        current = leftReference;
                    } else {
                        stack.push(leftReference);
                        current = rightReference;
                    }
                    descended = true;
                } else if (hitLeft) {
                    current = leftReference;
                    descended = true;
                } else if (hitRight) {
                    current = rightReference;
                    descended = true;
                }
            } else if (!insideInstance) {
                const record = TOP_LEVEL_HEADER_WORDS + index * INSTANCE_RECORD_WORDS;
                rootNodeOffset = topWords[record + INSTANCE_WORD.rootNodeOffset];
                triangleOffset = topWords[record + INSTANCE_WORD.triangleOffset];
                instanceIndex = topWords[record + INSTANCE_WORD.instanceIndex];
                mirrored = (topWords[record + INSTANCE_WORD.flags] & INSTANCE_FLAG.mirrored) !== 0;
                const row = (rowStart: number): number[] => [
                    topFloats[record + rowStart],
                    topFloats[record + rowStart + 1],
                    topFloats[record + rowStart + 2],
                    topFloats[record + rowStart + 3],
                ];
                const rows = [row(INSTANCE_WORD.worldToObjectRow0), row(INSTANCE_WORD.worldToObjectRow1), row(INSTANCE_WORD.worldToObjectRow2)];
                const origin = rows.map((r) => r[0] * ray.origin[0] + r[1] * ray.origin[1] + r[2] * ray.origin[2] + r[3]);
                // Not renormalized: t stays in world units.
                const direction = rows.map((r) => r[0] * ray.direction[0] + r[1] * ray.direction[1] + r[2] * ray.direction[2]);
                setup = makeRaySetup(origin, direction);
                stack.push(REF_INSTANCE_MARKER);
                insideInstance = true;
                current = makeReference(0, 0);
                descended = true;
            } else {
                for (let i = 0; i < count; i++) {
                    if (stats) stats.trianglesTested++;
                    const base = (triangleOffset + index + i) * TRIANGLE_WORDS;
                    const hit =
                        triangleTest === 'watertight'
                            ? intersectWatertight(triangleFloats, base, setup, ray.tMin, closestT)
                            : intersectMollerTrumbore(triangleFloats, base, setup, ray.tMin, closestT);
                    if (hit === null) continue;
                    result = {
                        t: hit.t,
                        instanceIndex,
                        triangleIndex: triangleWords[base + TRIANGLE_WORD.originalIndex],
                        u: hit.u,
                        v: hit.v,
                        // A mirrored instance flips the winding as seen in world space.
                        frontFace: hit.frontFace !== mirrored,
                    };
                    closestT = hit.t;
                    if (stopAtFirstHit) return result;
                }
            }

            if (descended) {
                if (stats) stats.maxStackSize = Math.max(stats.maxStackSize, stack.length);
                if (stack.length > TRACE_STACK_SIZE) {
                    throw new RangeError(`the trace stack overflowed: ${stack.length} entries, room for ${TRACE_STACK_SIZE}`);
                }
                continue;
            }

            // Pop.
            for (;;) {
                if (stack.length === 0) return result;
                const entry = stack.pop() as number;
                if (entry === REF_INSTANCE_MARKER) {
                    insideInstance = false;
                    setup = worldSetup;
                    continue;
                }
                current = entry;
                break;
            }
        }
    }

    return {
        closest: (ray, stats) => traverse(ray, false, stats),
        any: (ray, stats) => traverse(ray, true, stats) !== null,
    };
}

/** Moller-Trumbore with no epsilon on the edge tests, as in the shader. Triangles store `v0`, `v1 - v0`, `v2 - v0`. */
function intersectMollerTrumbore(
    floats: Float32Array,
    base: number,
    setup: RaySetup,
    tMin: number,
    tMax: number,
): TriangleHit | null {
    const v0 = base + TRIANGLE_WORD.vertex0;
    const e1 = base + TRIANGLE_WORD.secondVector;
    const e2 = base + TRIANGLE_WORD.thirdVector;
    const direction = setup.direction;
    const origin = setup.origin;
    const e1x = floats[e1], e1y = floats[e1 + 1], e1z = floats[e1 + 2];
    const e2x = floats[e2], e2y = floats[e2 + 1], e2z = floats[e2 + 2];
    const px = direction[1] * e2z - direction[2] * e2y;
    const py = direction[2] * e2x - direction[0] * e2z;
    const pz = direction[0] * e2y - direction[1] * e2x;
    const determinant = e1x * px + e1y * py + e1z * pz;
    if (determinant === 0) return null;
    const inverseDeterminant = 1 / determinant;
    const sx = origin[0] - floats[v0];
    const sy = origin[1] - floats[v0 + 1];
    const sz = origin[2] - floats[v0 + 2];
    const u = (sx * px + sy * py + sz * pz) * inverseDeterminant;
    if (u < 0 || u > 1) return null;
    const qx = sy * e1z - sz * e1y;
    const qy = sz * e1x - sx * e1z;
    const qz = sx * e1y - sy * e1x;
    const v = (direction[0] * qx + direction[1] * qy + direction[2] * qz) * inverseDeterminant;
    if (v < 0 || u + v > 1) return null;
    const t = (e2x * qx + e2y * qy + e2z * qz) * inverseDeterminant;
    if (t < tMin || t > tMax) return null;
    return { t, u, v, frontFace: determinant > 0 };
}

/** Woop, Benthin and Wald 2013. Triangles store `v0`, `v1`, `v2`. Edge-function zeros count for both sides. */
function intersectWatertight(
    floats: Float32Array,
    base: number,
    setup: RaySetup,
    tMin: number,
    tMax: number,
): TriangleHit | null {
    const [kx, ky, kz] = setup.axes;
    const [shearX, shearY, shearZ] = setup.shear;
    const relative = (word: number): number[] => [
        floats[word] - setup.origin[0],
        floats[word + 1] - setup.origin[1],
        floats[word + 2] - setup.origin[2],
    ];
    const a = relative(base + TRIANGLE_WORD.vertex0);
    const b = relative(base + TRIANGLE_WORD.secondVector);
    const c = relative(base + TRIANGLE_WORD.thirdVector);

    const ax = a[kx] - shearX * a[kz];
    const ay = a[ky] - shearY * a[kz];
    const bx = b[kx] - shearX * b[kz];
    const by = b[ky] - shearY * b[kz];
    const cx = c[kx] - shearX * c[kz];
    const cy = c[ky] - shearY * c[kz];

    const u = cx * by - cy * bx;
    const v = ax * cy - ay * cx;
    const w = bx * ay - by * ax;
    if ((u < 0 || v < 0 || w < 0) && (u > 0 || v > 0 || w > 0)) return null;

    const determinant = u + v + w;
    if (determinant === 0) return null;
    const scaledT = u * (shearZ * a[kz]) + v * (shearZ * b[kz]) + w * (shearZ * c[kz]);
    const inverseDeterminant = 1 / determinant;
    const t = scaledT * inverseDeterminant;
    if (t < tMin || t > tMax) return null;
    return { t, u: v * inverseDeterminant, v: w * inverseDeterminant, frontFace: determinant > 0 };
}
