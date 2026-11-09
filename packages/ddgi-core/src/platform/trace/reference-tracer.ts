/**
 * Brute-force reference tracer, used only to check the BVH and the shaders.
 *
 * It tests every triangle against every ray in double precision, in world
 * space, with no BVH and none of the object-space tricks the GPU path uses.
 * That independence is the point: a bug in the builder, the instance
 * transforms or the traversal shows up as a disagreement with this file.
 * Never use it for rendering.
 *
 * Materials (T4.2): given a material lookup, a triangle whose material lets
 * the ray through (see-through, the back face of a single-sided one, a
 * cutout's hole, a stochastic surface's hash; `materialLetsRayPass` in
 * material-test-wgsl.ts) is skipped, as if it weren't there. Every triangle
 * is looked at: the reference has no flags. Cutout alpha is read from the
 * material's texture as the CPU makes the store's layer from the test's own
 * pixels (textures/texture-store-cpu.ts), never from the GPU's copy; the
 * stochastic rule uses the hash's CPU twin on the reference's own
 * barycentrics, so its answers differ ray by ray from an f32 tracer's and
 * only the share that passes compares. A hit within a texel of an alpha edge
 * (at the alpha-test mip), and any stochastic decision, makes the answer
 * ambiguous.
 *
 * Besides the answer it reports how *certain* the answer is. An f32 tracer and
 * a double-precision one may legitimately differ for a ray that lands within
 * rounding distance of a triangle edge or of the ray's own `tMin`/`tMax`, so
 * results carry an `ambiguous` flag and comparisons only accept a difference
 * when it is set.
 */

import { STORE_LAYER_SIZE } from '../textures/texture-store-wgsl.ts';
import { mappedTextureUv, sampleStoreLayer, wrapTextureCoordinate } from '../textures/texture-store-cpu.ts';
import {
    ALPHA_TEST_LOD,
    materialAlphaAt,
    materialLetsRayPass,
    stochasticHashValue,
    type WalkMaterial,
    type WalkMaterialLookup,
    type WalkUvLookup,
} from './material-test-wgsl.ts';
import type { MeshGeometry } from './mesh-bvh.ts';
import { countTriangles } from './mesh-bvh.ts';
import { invertAffine } from './top-level.ts';
import type { TraceRay } from './trace-gpu.ts';

export type { TraceRay } from './trace-gpu.ts';
import type { TraceSceneInstance } from './scene-assembly.ts';


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
    /** 1 for triangles of a mirrored instance, whose world-space winding is the reverse of the authored one. */
    readonly mirrored: Uint8Array;
    /** Each triangle's material as the walk sees it; undefined (opaque, double-sided) without a material lookup. */
    readonly materials: readonly (WalkMaterial | undefined)[];
    /** Each triangle's corner uvs per uv set, `[u0, v0, u1, v1, u2, v2]` per triangle; null without a uv lookup or set. */
    readonly uvs: readonly [Float64Array | null, Float64Array | null];
}

/**
 * Transforms every triangle of every traceable instance to world space.
 * `materialAt` gives each triangle's material (T4.2); without it every
 * triangle is opaque and double-sided. `uvAt` gives its corner uvs, for the
 * alpha rules (cutouts with a texture).
 */
export function buildReferenceScene(
    meshes: readonly MeshGeometry[],
    instances: readonly TraceSceneInstance[],
    materialAt?: WalkMaterialLookup,
    uvAt?: WalkUvLookup,
): ReferenceScene {
    let total = 0;
    const traceable = instances.filter((instance) => {
        if (instance.visible === false || instance.contributesToGi === false) return false;
        return invertAffine(instance.objectToWorld) !== null;
    });
    const isMirrored = (instance: TraceSceneInstance): boolean => (invertAffine(instance.objectToWorld)?.determinant ?? 1) < 0;
    for (const instance of traceable) total += countTriangles(meshes[instance.meshIndex]);

    const geometry = new Float64Array(total * 9);
    const instanceIndices = new Uint32Array(total);
    const triangleIndices = new Uint32Array(total);
    const mirrored = new Uint8Array(total);
    const materials: (WalkMaterial | undefined)[] = new Array<WalkMaterial | undefined>(total);
    const uvs: [Float64Array | null, Float64Array | null] = [null, null];
    let slot = 0;
    for (const instance of traceable) {
        const mesh = meshes[instance.meshIndex];
        const instanceMirrored = isMirrored(instance) ? 1 : 0;
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
            mirrored[slot] = instanceMirrored;
            materials[slot] = materialAt?.(instance.instanceIndex, triangle);
            for (const channel of [0, 1] as const) {
                const corners = uvAt?.(instance.instanceIndex, triangle, channel);
                if (!corners) continue;
                const channelUvs = (uvs[channel] ??= new Float64Array(total * 6));
                for (let index = 0; index < 6; index++) channelUvs[slot * 6 + index] = corners[index];
            }
            slot++;
        }
    }
    return { triangleCount: total, geometry, instanceIndices, triangleIndices, mirrored, materials, uvs };
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

/** A material's decision at a candidate hit, and whether an f32 tracer may decide otherwise. */
interface MaterialDecision {
    readonly passes: boolean;
    /** Within a texel of an alpha edge, or a stochastic decision (its hash takes f32 barycentrics the reference doesn't have). */
    readonly ambiguous: boolean;
}

const BLOCKS: MaterialDecision = { passes: false, ambiguous: false };

/** One texel of the alpha-test mip, in texture coordinates. */
const ALPHA_TEST_TEXEL = Math.pow(2, ALPHA_TEST_LOD) / STORE_LAYER_SIZE;

/** The eight neighbours a texel away, around a cutout hit. */
const NEIGHBOUR_OFFSETS: readonly (readonly [number, number])[] = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
];

const NO_UVS = [0, 0, 0, 0, 0, 0];

/**
 * Whether a cutout's decision changes within a texel of the hit, at the
 * alpha-test mip: some point a texel away (in texture coordinates, so after
 * the uv transform, then wrapped) is on the other side of the cutoff. There
 * the GPU's filter weights, its f32 uvs and the texture's bytes may each put
 * the edge a little elsewhere.
 */
function nearAlphaEdge(material: WalkMaterial, passes: boolean, cornerUvs: ArrayLike<number>, u: number, v: number): boolean {
    const texture = material.albedoTexture;
    if (!texture) return false;
    const opacity = material.opacity ?? 1;
    const cutoff = material.alphaCutoff ?? 0;
    const clamp = (weight: number): number => Math.min(Math.max(weight, 0), 1);
    const [centreU, centreV] = mappedTextureUv(texture.uv, cornerUvs, [clamp(1 - u - v), clamp(u), clamp(v)]);
    const halfTexel = (0.5 * Math.pow(2, Math.ceil(ALPHA_TEST_LOD))) / STORE_LAYER_SIZE;
    for (const [du, dv] of NEIGHBOUR_OFFSETS) {
        const sampleU = wrapTextureCoordinate(centreU + du * ALPHA_TEST_TEXEL, texture.uv?.wrapU, halfTexel);
        const sampleV = wrapTextureCoordinate(centreV + dv * ALPHA_TEST_TEXEL, texture.uv?.wrapV, halfTexel);
        const alpha = opacity * sampleStoreLayer(texture.layer, sampleU, sampleV, ALPHA_TEST_LOD)[3];
        if (alpha < cutoff !== passes) return true;
    }
    return false;
}

/**
 * The triangle's material at a candidate hit (u, v the weights of corners 1
 * and 2): whether the ray goes on through, and whether that is certain. The
 * side is the authored one: the world-space winding test, flipped back for a
 * mirrored instance (as `frontFace` below).
 */
function materialDecision(scene: ReferenceScene, triangle: number, determinant: number, u: number, v: number): MaterialDecision {
    const material = scene.materials[triangle];
    if (!material) return BLOCKS;
    const frontFace = determinant > 0 !== (scene.mirrored[triangle] === 1);
    const uvsOf = (channel: 0 | 1): Float64Array | null => scene.uvs[channel]?.subarray(triangle * 6, triangle * 6 + 6) ?? null;
    let stochastic = false;
    const hashAt = (): number => {
        stochastic = true;
        return stochasticHashValue(scene.instanceIndices[triangle], scene.triangleIndices[triangle], u, v);
    };
    const passes = materialLetsRayPass(material, frontFace, () => materialAlphaAt(material, u, v, uvsOf), hashAt);
    // The alpha rules decide only for blocking surfaces met on a side they block from.
    const alphaDecides = material.blocksLight && (material.doubleSided || frontFace);
    if (!alphaDecides) return { passes, ambiguous: false };
    if (stochastic) return { passes, ambiguous: true };
    if (material.alphaMode === 'cutout') {
        return { passes, ambiguous: nearAlphaEdge(material, passes, uvsOf(material.albedoTexture?.uv?.channel ?? 0) ?? NO_UVS, u, v) };
    }
    return { passes, ambiguous: false };
}

function nearIntervalEnd(t: number, ray: TraceRay): boolean {
    const slack = REFERENCE_TOLERANCE * Math.max(1, Math.abs(t));
    return Math.abs(t - ray.tMin) <= slack || Math.abs(t - ray.tMax) <= slack;
}

const scratch = new Float64Array(4);

/** The closest-hit reference's answer: the hit (null for a miss), and whether a miss is certain. */
export interface ReferenceClosest {
    readonly hit: ReferenceHit | null;
    /** For a miss: a surface whose material let the ray through was within a texel of an alpha edge, or stochastic. */
    readonly missAmbiguous: boolean;
}

/** Nearest hit within `[tMin, tMax]`, or null. Back faces count as hits. */
export function traceClosestReference(scene: ReferenceScene, ray: TraceRay): ReferenceHit | null {
    return traceClosestReferenceWithMiss(scene, ray).hit;
}

/** {@link traceClosestReference}, also saying whether a miss could be a hit for an f32 tracer (alpha edges, stochastic surfaces). */
export function traceClosestReferenceWithMiss(scene: ReferenceScene, ray: TraceRay): ReferenceClosest {
    let bestT = Infinity;
    let best = -1;
    let bestU = 0;
    let bestV = 0;
    let bestDeterminant = 0;
    const hitTimes: number[] = [];
    // The nearest blocking triangle the ray misses by less than the tolerance (just past an edge).
    let nearestNearMiss = Infinity;
    // The nearest surface whose material decision is uncertain (an alpha edge, a stochastic surface).
    let nearestUncertainMaterial = Infinity;

    for (let triangle = 0; triangle < scene.triangleCount; triangle++) {
        if (!intersect(scene, triangle, ray, scratch)) continue;
        const t = scratch[0];
        const u = scratch[1];
        const v = scratch[2];
        const determinant = scratch[3];
        if (t < ray.tMin || t > ray.tMax) continue;
        const margin = edgeMargin(u, v);
        if (margin < -REFERENCE_TOLERANCE) continue;
        const decision = materialDecision(scene, triangle, determinant, u, v);
        if (decision.ambiguous) nearestUncertainMaterial = Math.min(nearestUncertainMaterial, t);
        if (decision.passes) continue;
        if (margin < 0) {
            nearestNearMiss = Math.min(nearestNearMiss, t);
            continue;
        }
        hitTimes.push(t);
        if (t < bestT) {
            bestT = t;
            best = triangle;
            bestU = u;
            bestV = v;
            bestDeterminant = determinant;
        }
    }
    if (best < 0) return { hit: null, missAmbiguous: nearestUncertainMaterial < Infinity };

    // Ambiguous when the closest hit is near an edge or the interval end, or when
    // another triangle is almost as close (which one is "first" is rounding).
    let bestIsAmbiguous = edgeMargin(bestU, bestV) < REFERENCE_TOLERANCE || nearIntervalEnd(bestT, ray);
    const tieSlack = 1e-5 * Math.max(1, bestT);
    if (!bestIsAmbiguous) {
        let closeHits = 0;
        for (const t of hitTimes) if (t - bestT <= tieSlack) closeHits++;
        bestIsAmbiguous = closeHits > 1;
    }
    // A blocker missed by a hair in front of the hit: an f32 tracer may hit it instead. Without
    // materials its neighbour across the edge is usually hit too (a tie, above); next to a
    // see-through triangle (a glass pane in a wall) nothing else stands in for it.
    if (nearestNearMiss < bestT + tieSlack) bestIsAmbiguous = true;
    // A surface in front of (or at) the hit whose material an f32 tracer may decide otherwise.
    if (nearestUncertainMaterial < bestT + tieSlack) bestIsAmbiguous = true;

    const hit: ReferenceHit = {
        t: bestT,
        instanceIndex: scene.instanceIndices[best],
        triangleIndex: scene.triangleIndices[best],
        u: bestU,
        v: bestV,
        // The authored side (DECISIONS 2026-11-04): the world-space winding test, flipped
        // back for a mirrored instance, whose transform reverses the winding.
        frontFace: bestDeterminant > 0 !== (scene.mirrored[best] === 1),
        ambiguous: bestIsAmbiguous,
    };
    return { hit, missAmbiguous: false };
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
        const decision = materialDecision(scene, triangle, scratch[3], u, v);
        if (decision.ambiguous) ambiguousCandidate = true;
        if (decision.passes) continue;
        if (inside) blocked = true;
        if (inside && margin >= REFERENCE_TOLERANCE && !nearIntervalEnd(t, ray) && !decision.ambiguous) {
            robustlyBlocked = true;
            break;
        }
        ambiguousCandidate = true;
    }
    return { blocked, ambiguous: !robustlyBlocked && ambiguousCandidate };
}
