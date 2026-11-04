/**
 * Brute-force reference tracer, used only to check the BVH and the shaders.
 *
 * It tests every triangle against every ray in double precision, in world
 * space, with no BVH and none of the object-space tricks the GPU path uses.
 * That independence is the point: a bug in the builder, the instance
 * transforms or the traversal shows up as a disagreement with this file.
 * Never use it for rendering.
 *
 * Besides the answer it reports how *certain* the answer is. An f32 tracer and
 * a double-precision one may legitimately differ for a ray that lands within
 * rounding distance of a triangle edge or of the ray's own `tMin`/`tMax`, so
 * results carry an `ambiguous` flag and comparisons only accept a difference
 * when it is set.
 */

import type { MeshGeometry } from './mesh-bvh.ts';
import { countTriangles } from './mesh-bvh.ts';
import { invertAffine } from './top-level.ts';
import type { TraceSceneInstance } from './scene-assembly.ts';

export interface TraceRay {
    readonly origin: readonly [number, number, number];
    /** Normalized. */
    readonly direction: readonly [number, number, number];
    readonly tMin: number;
    readonly tMax: number;
}

export interface ReferenceHit {
    readonly t: number;
    readonly instanceIndex: number;
    readonly triangleIndex: number;
    /** Weights of vertices 1 and 2. */
    readonly u: number;
    readonly v: number;
    readonly frontFace: boolean;
    /** The ray is within rounding distance of an edge or of its interval end, or two triangles tie. A different f32 answer is acceptable. */
    readonly ambiguous: boolean;
}

/**
 * How close to an edge (in barycentric weight) or to `tMin`/`tMax` (relative)
 * a hit may be before an f32 tracer is allowed to disagree. Far larger than
 * f32 rounding for well-shaped triangles, small enough to hide almost nothing.
 */
export const REFERENCE_TOLERANCE = 1e-3;

/** World-space triangles, flattened for speed: per triangle `a` (3), `edge1` (3), `edge2` (3). */
export interface ReferenceScene {
    readonly triangleCount: number;
    readonly geometry: Float64Array;
    readonly instanceIndices: Uint32Array;
    readonly triangleIndices: Uint32Array;
}

/** Transforms every triangle of every traceable instance to world space. */
export function buildReferenceScene(
    meshes: readonly MeshGeometry[],
    instances: readonly TraceSceneInstance[],
): ReferenceScene {
    let total = 0;
    const traceable = instances.filter((instance) => {
        if (instance.visible === false || instance.contributesToGi === false) return false;
        return invertAffine(instance.objectToWorld) !== null;
    });
    for (const instance of traceable) total += countTriangles(meshes[instance.meshIndex]);

    const geometry = new Float64Array(total * 9);
    const instanceIndices = new Uint32Array(total);
    const triangleIndices = new Uint32Array(total);
    let slot = 0;
    for (const instance of traceable) {
        const mesh = meshes[instance.meshIndex];
        const e = instance.objectToWorld;
        const toWorld = (vertex: number): [number, number, number] => {
            const x = mesh.positions[vertex * 3];
            const y = mesh.positions[vertex * 3 + 1];
            const z = mesh.positions[vertex * 3 + 2];
            return [
                e[0] * x + e[4] * y + e[8] * z + e[12],
                e[1] * x + e[5] * y + e[9] * z + e[13],
                e[2] * x + e[6] * y + e[10] * z + e[14],
            ];
        };
        const triangles = countTriangles(mesh);
        for (let triangle = 0; triangle < triangles; triangle++) {
            const corner = (index: number): number => (mesh.indices ? mesh.indices[triangle * 3 + index] : triangle * 3 + index);
            const a = toWorld(corner(0));
            const b = toWorld(corner(1));
            const c = toWorld(corner(2));
            const base = slot * 9;
            for (let axis = 0; axis < 3; axis++) {
                geometry[base + axis] = a[axis];
                geometry[base + 3 + axis] = b[axis] - a[axis];
                geometry[base + 6 + axis] = c[axis] - a[axis];
            }
            instanceIndices[slot] = instance.instanceIndex;
            triangleIndices[slot] = triangle;
            slot++;
        }
    }
    return { triangleCount: total, geometry, instanceIndices, triangleIndices };
}

/** Möller-Trumbore in doubles. Writes `t, u, v, determinant` to `out` and returns false when the ray misses the triangle's plane region. */
function intersect(scene: ReferenceScene, triangle: number, ray: TraceRay, out: Float64Array): boolean {
    const g = scene.geometry;
    const base = triangle * 9;
    const e1x = g[base + 3], e1y = g[base + 4], e1z = g[base + 5];
    const e2x = g[base + 6], e2y = g[base + 7], e2z = g[base + 8];
    const dx = ray.direction[0];
    const dy = ray.direction[1];
    const dz = ray.direction[2];

    const px = dy * e2z - dz * e2y;
    const py = dz * e2x - dx * e2z;
    const pz = dx * e2y - dy * e2x;
    const determinant = e1x * px + e1y * py + e1z * pz;
    const scale = Math.hypot(e1x, e1y, e1z) * Math.hypot(e2x, e2y, e2z);
    // Parallel to the triangle's plane to within rounding: no meaningful hit.
    if (Math.abs(determinant) <= 64 * Number.EPSILON * scale) return false;

    const inverse = 1 / determinant;
    const sx = ray.origin[0] - g[base];
    const sy = ray.origin[1] - g[base + 1];
    const sz = ray.origin[2] - g[base + 2];
    const u = (sx * px + sy * py + sz * pz) * inverse;
    const qx = sy * e1z - sz * e1y;
    const qy = sz * e1x - sx * e1z;
    const qz = sx * e1y - sy * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * inverse;
    const t = (e2x * qx + e2y * qy + e2z * qz) * inverse;
    out[0] = t;
    out[1] = u;
    out[2] = v;
    out[3] = determinant;
    return true;
}

/** Smallest of the three barycentric weights: negative means outside the triangle. */
function edgeMargin(u: number, v: number): number {
    return Math.min(u, v, 1 - u - v);
}

function nearIntervalEnd(t: number, ray: TraceRay): boolean {
    const slack = REFERENCE_TOLERANCE * Math.max(1, Math.abs(t));
    return Math.abs(t - ray.tMin) <= slack || Math.abs(t - ray.tMax) <= slack;
}

const scratch = new Float64Array(4);

/** Nearest hit within `[tMin, tMax]`, or null. Back faces count as hits. */
export function traceClosestReference(scene: ReferenceScene, ray: TraceRay): ReferenceHit | null {
    let bestT = Infinity;
    let best = -1;
    let bestU = 0;
    let bestV = 0;
    let bestDeterminant = 0;
    const hitTimes: number[] = [];

    for (let triangle = 0; triangle < scene.triangleCount; triangle++) {
        if (!intersect(scene, triangle, ray, scratch)) continue;
        const t = scratch[0];
        const u = scratch[1];
        const v = scratch[2];
        const determinant = scratch[3];
        if (edgeMargin(u, v) < 0 || t < ray.tMin || t > ray.tMax) continue;
        hitTimes.push(t);
        if (t < bestT) {
            bestT = t;
            best = triangle;
            bestU = u;
            bestV = v;
            bestDeterminant = determinant;
        }
    }
    if (best < 0) return null;

    // Ambiguous when the closest hit is near an edge or the interval end, or when
    // another triangle is almost as close (which one is "first" is rounding).
    let bestIsAmbiguous = edgeMargin(bestU, bestV) < REFERENCE_TOLERANCE || nearIntervalEnd(bestT, ray);
    const tieSlack = 1e-5 * Math.max(1, bestT);
    if (!bestIsAmbiguous) {
        let closeHits = 0;
        for (const t of hitTimes) if (t - bestT <= tieSlack) closeHits++;
        bestIsAmbiguous = closeHits > 1;
    }

    return {
        t: bestT,
        instanceIndex: scene.instanceIndices[best],
        triangleIndex: scene.triangleIndices[best],
        u: bestU,
        v: bestV,
        // World-space winding: front when the ray runs against the geometric normal.
        frontFace: bestDeterminant > 0,
        ambiguous: bestIsAmbiguous,
    };
}

export interface ReferenceOcclusion {
    readonly blocked: boolean;
    /** Not trusted either way: a candidate hit sits within rounding distance of an edge or interval end. */
    readonly ambiguous: boolean;
}

/** Is anything within `[tMin, tMax]`? Used for shadow rays. */
export function traceAnyReference(scene: ReferenceScene, ray: TraceRay): ReferenceOcclusion {
    let blocked = false;
    let robustlyBlocked = false;
    let ambiguousCandidate = false;
    for (let triangle = 0; triangle < scene.triangleCount; triangle++) {
        if (!intersect(scene, triangle, ray, scratch)) continue;
        const t = scratch[0];
        const u = scratch[1];
        const v = scratch[2];
        const margin = edgeMargin(u, v);
        const slack = REFERENCE_TOLERANCE * Math.max(1, Math.abs(t));
        const inside = margin >= 0 && t >= ray.tMin && t <= ray.tMax;
        const nearMiss = margin >= -REFERENCE_TOLERANCE && t >= ray.tMin - slack && t <= ray.tMax + slack;
        if (!nearMiss) continue;
        if (inside) blocked = true;
        if (inside && margin >= REFERENCE_TOLERANCE && !nearIntervalEnd(t, ray)) {
            robustlyBlocked = true;
            break;
        }
        ambiguousCandidate = true;
    }
    return { blocked, ambiguous: !robustlyBlocked && ambiguousCandidate };
}
