/**
 * The `rc` ray sets (RC1.5): rays shaped like Capsaicin GI-1.2's defaults, so
 * the trace bench can predict what the radiance-cache baseline's tracing
 * will cost before any of the technique exists.
 *
 * From the scene's first camera bookmark at 1280x720:
 *
 * 1. **Placement:** one primary closest-hit ray per 16x16-pixel tile, through
 *    a Halton (2, 3) jittered point in the tile. A tile whose ray hits
 *    nothing places no probe. Capsaicin reads the probe's pixel from its
 *    visibility buffer instead of tracing, so these rays are reported but are
 *    not part of the per-frame sum.
 * 2. **Probe rays:** 64 per probe, cosine-weighted about the probe's normal:
 *    an 8x8 stratified grid (Capsaicin's 8x8 hemi-octahedral cells), jittered
 *    inside each cell and turned by a random angle about the normal, both per
 *    probe. Closest hit, no length limit.
 * 3. **Multi-bounce rays:** 0.3x the probe-ray count, each from a random
 *    probe-ray hit (only rays that hit geometry), one cosine ray about that
 *    hit's normal. *This approximates Capsaicin's multi-bounce cells*, which
 *    trace from hash-grid cache cells as they are populated, not from probe
 *    hits; the count and the kind of ray (diffuse, from a surface, closest
 *    hit) are what matter for cost.
 * 4. **Shadow rays (any hit):** one per hit of sets 2 and 3, toward one light
 *    sampled per hit: the sun where the scene has one, otherwise one of the
 *    manifest's point and spot lights chosen uniformly, with the ray ending at
 *    the light. A hit facing away from its light gets none, since shading
 *    would skip it.
 *
 * Every random choice comes from one seeded generator (mulberry32, the
 * oracle fixtures'), so every tracer and every run sees the same rays.
 *
 * Conventions are the platform tracer's: rays are packed as `GiRay`
 * (`origin (3), tMin, direction (3), tMax`), hit points are rebuilt from the
 * barycentrics on the world-space triangle and moved slightly inside it
 * (`giSurfaceAt`), and rays leave a surface from a point pushed off it along
 * the geometric normal of the side the ray arrived on, by 256 rounding steps
 * of the point's largest coordinate (`giShadeOffsetRayOrigin`).
 */

import { HIT_RECORD_BYTES, HIT_RECORD_WORD, RAY_FLOATS } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import type { Random } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import type { ManifestLight } from '../../../../scenes/src/manifest.ts';
import { RAY_FAR } from './config.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import { normalizeVector, type Vec3 } from './rays.ts';

// ---- The platform's surface conventions, mirrored from platform/shade/wgsl.ts ----------

/** One rounding step (ulp) of 1.0 in f32, 2^-23 (`GI_SHADE_ULP`, whose literal rounds to exactly this in f32). */
const F32_ULP = 2 ** -23;
/** Below this distance from the origin the rounding step stays that of 1/16 m, as ulps vanish at 0. */
const ROUNDING_STEP_MIN_MAGNITUDE = 0.0625;
/** How far a hit point moves into its triangle, in rounding steps (`GI_SHADE_IN_SURFACE_STEPS`). */
const IN_SURFACE_STEPS = 64;
/** How far a ray's origin moves off the surface, in rounding steps (`GI_SHADE_NORMAL_STEPS`). */
const NORMAL_OFFSET_STEPS = 256;
/** A shadow ray toward a directional light runs this far (`GI_SHADE_UNBOUNDED_DISTANCE`). */
export const SUN_SHADOW_RAY_LENGTH = 1.0e27;

/** The size of one f32 rounding step of a point's largest coordinate (`giShadeRoundingStep`). */
export function roundingStep(point: ArrayLike<number>): number {
    const magnitude = Math.max(Math.abs(point[0]), Math.abs(point[1]), Math.abs(point[2]), ROUNDING_STEP_MIN_MAGNITUDE);
    return magnitude * F32_ULP;
}

/** Where a ray leaving a surface starts (`giShadeOffsetRayOrigin`): pushed off along the normal of the side it leaves from. */
export function offsetRayOrigin(position: ArrayLike<number>, normal: ArrayLike<number>): Vec3 {
    const push = NORMAL_OFFSET_STEPS * roundingStep(position);
    return [position[0] + normal[0] * push, position[1] + normal[1] * push, position[2] + normal[2] * push];
}

// ---- Small helpers -------------------------------------------------------------------------

/** The radical inverse of `index` in `base`: the Halton sequence's coordinate for that base. */
export function halton(index: number, base: number): number {
    let result = 0;
    let fraction = 1 / base;
    let remaining = index;
    while (remaining > 0) {
        result += (remaining % base) * fraction;
        remaining = Math.floor(remaining / base);
        fraction /= base;
    }
    return result;
}

function cross(a: readonly number[], b: readonly number[]): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/**
 * Two unit tangents that make a right-handed frame with the unit normal `n`
 * (Duff et al., "Building an Orthonormal Basis, Revisited", JCGT 2017): no
 * branch on which axis to cross with, and no loss of precision near the poles.
 */
export function tangentFrame(n: ArrayLike<number>): { tangent: Vec3; bitangent: Vec3 } {
    const sign = n[2] >= 0 ? 1 : -1;
    const a = -1 / (sign + n[2]);
    const b = n[0] * n[1] * a;
    return {
        tangent: [1 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]],
        bitangent: [b, sign + n[1] * n[1] * a, -n[1]],
    };
}

/**
 * Writes one `GiRay` at ray slot `slot`. The direction is rounded to f32 and
 * then normalised again, so what the GPU reads is a unit vector to f32
 * precision (rounding each component can leave the length a hair off 1).
 */
function writeRay(output: Float32Array, slot: number, origin: ArrayLike<number>, direction: ArrayLike<number>, tMax: number): void {
    const dx = Math.fround(direction[0]);
    const dy = Math.fround(direction[1]);
    const dz = Math.fround(direction[2]);
    const length = Math.hypot(dx, dy, dz);
    const base = slot * RAY_FLOATS;
    output[base] = origin[0];
    output[base + 1] = origin[1];
    output[base + 2] = origin[2];
    output[base + 3] = 0;
    output[base + 4] = dx / length;
    output[base + 5] = dy / length;
    output[base + 6] = dz / length;
    output[base + 7] = tMax;
}

/**
 * A cosine-weighted direction about the unit normal `n` from two numbers in
 * [0, 1): `u1` picks the height (cos θ = sqrt(1 − u1), so the density is
 * cos θ / π, Malley's method), `u2` the angle around the normal.
 */
export function cosineDirection(n: ArrayLike<number>, frame: { tangent: Vec3; bitangent: Vec3 }, u1: number, u2: number): Vec3 {
    const radius = Math.sqrt(u1);
    const angle = 2 * Math.PI * u2;
    const x = radius * Math.cos(angle);
    const y = radius * Math.sin(angle);
    const z = Math.sqrt(Math.max(0, 1 - u1));
    const { tangent, bitangent } = frame;
    return [
        tangent[0] * x + bitangent[0] * y + n[0] * z,
        tangent[1] * x + bitangent[1] * y + n[1] * z,
        tangent[2] * x + bitangent[2] * y + n[2] * z,
    ];
}

// ---- 1. Probe placement ------------------------------------------------------------------

export interface RcCamera {
    position: Vec3;
    target: Vec3;
    /** Vertical field of view, degrees. */
    fovYDeg: number;
}

export interface RcView {
    width: number;
    height: number;
    /** Pixels per tile side; one probe per tile. */
    tileSize: number;
}

export interface TileGrid {
    columns: number;
    rows: number;
    count: number;
}

export function tileGrid(view: RcView): TileGrid {
    const columns = Math.ceil(view.width / view.tileSize);
    const rows = Math.ceil(view.height / view.tileSize);
    return { columns, rows, count: columns * rows };
}

/**
 * One primary ray per tile, row by row from the top-left tile, through the
 * tile's Halton (2, 3) point of index `tile + haltonFirstIndex`. A pinhole
 * camera at the bookmark's position looking at its target, y up (the
 * manifests' `upAxis`), with the bookmark's vertical field of view.
 */
export function makePlacementRays(camera: RcCamera, view: RcView, haltonFirstIndex: number): Float32Array {
    const grid = tileGrid(view);
    const forward = normalizeVector([
        camera.target[0] - camera.position[0],
        camera.target[1] - camera.position[1],
        camera.target[2] - camera.position[2],
    ]);
    const right = normalizeVector(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const tanHalfY = Math.tan((camera.fovYDeg * Math.PI) / 360);
    const tanHalfX = tanHalfY * (view.width / view.height);

    const rays = new Float32Array(grid.count * RAY_FLOATS);
    for (let tile = 0; tile < grid.count; tile++) {
        const column = tile % grid.columns;
        const row = Math.floor(tile / grid.columns);
        const jitterIndex = tile + haltonFirstIndex;
        // A point inside the tile, clipped to the view for a partial last tile.
        const pixelX = Math.min(view.width, (column + halton(jitterIndex, 2)) * view.tileSize);
        const pixelY = Math.min(view.height, (row + halton(jitterIndex, 3)) * view.tileSize);
        const screenX = (2 * pixelX) / view.width - 1;
        const screenY = 1 - (2 * pixelY) / view.height;
        const direction = [0, 1, 2].map(
            (axis) => forward[axis] + right[axis] * screenX * tanHalfX + up[axis] * screenY * tanHalfY,
        );
        writeRay(rays, tile, camera.position, normalizeVector(direction), RAY_FAR);
    }
    return rays;
}

// ---- Surfaces at hits ----------------------------------------------------------------------

/** Where a set of closest-hit rays landed. */
export interface HitSurfaces {
    /** Per surface, the ray that found it. */
    rayIndex: Uint32Array;
    /** Per surface, `x, y, z`: the point on the triangle, moved slightly inside it (`giSurfaceAt`). */
    positions: Float64Array;
    /** Per surface, `x, y, z`: the unit geometric normal, turned to the side the ray arrived from. */
    normals: Float64Array;
    count: number;
    /** Rays that hit nothing. */
    missCount: number;
    /** Hits on a collapsed triangle, which has no normal; dropped. */
    degenerateCount: number;
}

/**
 * Rebuilds each hit's surface from the hit records (`GiHitRecord`) and the
 * scene's geometry, as the platform's `giSurfaceAt` does: the point from the
 * barycentrics on the world-space triangle, moved into the triangle by 64
 * rounding steps (toward its centroid), and the geometric normal turned to
 * the side the ray came from.
 */
export function hitSurfaces(
    rays: Float32Array,
    hitRecords: ArrayBuffer,
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
): HitSurfaces {
    const rayCount = rays.length / RAY_FLOATS;
    if (hitRecords.byteLength < rayCount * HIT_RECORD_BYTES) throw new RangeError('fewer hit records than rays');
    const recordWords = HIT_RECORD_BYTES / 4;
    const floats = new Float32Array(hitRecords);
    const words = new Uint32Array(hitRecords);

    const rayIndex = new Uint32Array(rayCount);
    const positions = new Float64Array(rayCount * 3);
    const normals = new Float64Array(rayCount * 3);
    let count = 0;
    let missCount = 0;
    let degenerateCount = 0;
    const corner = new Float64Array(9);
    const onTriangle = new Float64Array(3);

    for (let ray = 0; ray < rayCount; ray++) {
        const record = ray * recordWords;
        const t = floats[record + HIT_RECORD_WORD.t];
        if (!(t >= 0)) {
            missCount++;
            continue;
        }
        const instanceIndex = words[record + HIT_RECORD_WORD.instance];
        const instance = instances[instanceIndex];
        if (!instance) throw new RangeError(`ray ${ray} reports instance ${instanceIndex}, which does not exist`);
        const mesh = meshes[instance.meshIndex];
        const triangle = words[record + HIT_RECORD_WORD.triangle];
        const m = instance.objectToWorld;
        for (let c = 0; c < 3; c++) {
            const vertex = mesh.indices[triangle * 3 + c] * 3;
            const x = mesh.positions[vertex];
            const y = mesh.positions[vertex + 1];
            const z = mesh.positions[vertex + 2];
            for (let axis = 0; axis < 3; axis++) corner[c * 3 + axis] = m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis];
        }

        const e1 = [corner[3] - corner[0], corner[4] - corner[1], corner[5] - corner[2]];
        const e2 = [corner[6] - corner[0], corner[7] - corner[1], corner[8] - corner[2]];
        const normal = cross(e1, e2);
        const normalLength = Math.hypot(normal[0], normal[1], normal[2]);
        if (!(normalLength > 0)) {
            degenerateCount++;
            continue;
        }
        const base = ray * RAY_FLOATS;
        // Turn the normal to the side the ray arrived from.
        const facing = normal[0] * rays[base + 4] + normal[1] * rays[base + 5] + normal[2] * rays[base + 6] > 0 ? -1 : 1;

        // The point from clamped barycentrics, then nudged toward the centroid (giSurfaceAt).
        const u = Math.min(Math.max(floats[record + HIT_RECORD_WORD.barycentrics], 0), 1);
        const v = Math.min(Math.max(floats[record + HIT_RECORD_WORD.barycentrics + 1], 0), 1);
        const w = Math.min(Math.max(1 - u - v, 0), 1);
        for (let axis = 0; axis < 3; axis++) onTriangle[axis] = corner[axis] * w + corner[3 + axis] * u + corner[6 + axis] * v;
        const toCentroid = [0, 1, 2].map((axis) => (corner[axis] + corner[3 + axis] + corner[6 + axis]) / 3 - onTriangle[axis]);
        const centroidDistance = Math.hypot(toCentroid[0], toCentroid[1], toCentroid[2]);
        const nudge = Math.min(IN_SURFACE_STEPS * roundingStep(onTriangle), centroidDistance);
        const nudgeScale = nudge / Math.max(centroidDistance, 1e-30);

        for (let axis = 0; axis < 3; axis++) {
            positions[count * 3 + axis] = onTriangle[axis] + toCentroid[axis] * nudgeScale;
            normals[count * 3 + axis] = (facing * normal[axis]) / normalLength;
        }
        rayIndex[count] = ray;
        count++;
    }
    return {
        rayIndex: rayIndex.slice(0, count),
        positions: positions.slice(0, count * 3),
        normals: normals.slice(0, count * 3),
        count,
        missCount,
        degenerateCount,
    };
}

export interface RcProbes {
    /** Per probe, `x, y, z`: the surface point pushed off the surface toward the camera (`offsetRayOrigin`). */
    origins: Float64Array;
    /** Per probe, the unit geometric normal on the camera's side. */
    normals: Float64Array;
    /** Per probe, the tile it belongs to. */
    tiles: Uint32Array;
    count: number;
    /** Tiles whose ray saw sky or nothing (no probe). */
    missCount: number;
    /** Tiles whose ray hit a collapsed triangle (no probe). */
    degenerateCount: number;
}

/**
 * Places one probe per tile whose placement ray hit something: at the hit,
 * pushed off the surface by the platform's ray offset, with the surface's
 * normal faced toward the camera (the side the primary ray came from).
 */
export function placeProbes(
    placementRays: Float32Array,
    hitRecords: ArrayBuffer,
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
): RcProbes {
    const surfaces = hitSurfaces(placementRays, hitRecords, meshes, instances);
    const origins = new Float64Array(surfaces.count * 3);
    for (let probe = 0; probe < surfaces.count; probe++) {
        const position = surfaces.positions.subarray(probe * 3, probe * 3 + 3);
        const normal = surfaces.normals.subarray(probe * 3, probe * 3 + 3);
        origins.set(offsetRayOrigin(position, normal), probe * 3);
    }
    return {
        origins,
        normals: surfaces.normals,
        tiles: surfaces.rayIndex,
        count: surfaces.count,
        missCount: surfaces.missCount,
        degenerateCount: surfaces.degenerateCount,
    };
}

// ---- 2. Probe rays -------------------------------------------------------------------------

/**
 * `raysPerProbe` cosine-weighted closest-hit rays per probe, probe-major.
 * The rays of one probe cover a side x side stratified grid over the
 * (height, angle) square (8x8 for 64, like Capsaicin's 8x8 cells), jittered
 * inside each cell; and the whole set is turned by a random angle about the
 * normal. Jitter and angle are drawn per probe, so no two probes share
 * directions.
 */
export function makeProbeRays(probes: Pick<RcProbes, 'origins' | 'normals' | 'count'>, raysPerProbe: number, random: Random): Float32Array {
    const side = Math.round(Math.sqrt(raysPerProbe));
    if (side * side !== raysPerProbe) throw new RangeError(`rays per probe must be a square number (a side x side grid), got ${raysPerProbe}`);
    const rays = new Float32Array(probes.count * raysPerProbe * RAY_FLOATS);
    for (let probe = 0; probe < probes.count; probe++) {
        const origin = probes.origins.subarray(probe * 3, probe * 3 + 3);
        const normal = probes.normals.subarray(probe * 3, probe * 3 + 3);
        const frame = tangentFrame(normal);
        const rotation = random();
        for (let cell = 0; cell < raysPerProbe; cell++) {
            const heightCell = Math.floor(cell / side);
            const angleCell = cell % side;
            const u1 = (heightCell + random()) / side;
            const u2 = (angleCell + random()) / side + rotation;
            writeRay(rays, probe * raysPerProbe + cell, origin, cosineDirection(normal, frame, u1, u2 - Math.floor(u2)), RAY_FAR);
        }
    }
    return rays;
}

// ---- 3. Multi-bounce rays ------------------------------------------------------------------

/** How many multi-bounce rays go with `probeRayCount` probe rays. */
export function multiBounceRayCount(probeRayCount: number, ratio: number): number {
    return Math.round(probeRayCount * ratio);
}

/**
 * `count` multi-bounce rays, each from a random probe-ray hit (`surfaces`),
 * one cosine-weighted ray about that hit's normal, leaving from the
 * platform's offset origin. Hits are drawn without repeats while there are
 * enough; past that (fewer hits than `count`, as in a mostly open view) they
 * are drawn again at random, so the count stays 0.3x the probe rays.
 *
 * An approximation of Capsaicin's multi-bounce cells (rays from hash-grid
 * cache cells being populated), not the exact source.
 */
export function makeBounceRays(surfaces: Pick<HitSurfaces, 'positions' | 'normals' | 'count'>, count: number, random: Random): Float32Array {
    if (surfaces.count === 0 || count === 0) return new Float32Array(0);
    const rays = new Float32Array(count * RAY_FLOATS);
    // A partial Fisher-Yates shuffle: the first `distinct` entries become a random subset.
    const order = new Uint32Array(surfaces.count);
    for (let index = 0; index < order.length; index++) order[index] = index;
    const distinct = Math.min(count, surfaces.count);
    for (let index = 0; index < distinct; index++) {
        const swap = index + Math.floor(random() * (order.length - index));
        const chosen = order[swap];
        order[swap] = order[index];
        order[index] = chosen;
    }
    for (let ray = 0; ray < count; ray++) {
        const surface = ray < distinct ? order[ray] : Math.floor(random() * surfaces.count);
        const position = surfaces.positions.subarray(surface * 3, surface * 3 + 3);
        const normal = surfaces.normals.subarray(surface * 3, surface * 3 + 3);
        const direction = cosineDirection(normal, tangentFrame(normal), random(), random());
        writeRay(rays, ray, offsetRayOrigin(position, normal), direction, RAY_FAR);
    }
    return rays;
}

// ---- 4. Shadow rays ------------------------------------------------------------------------

/** Which light a hit's shadow ray goes to. */
export type RcLights =
    | { kind: 'sun'; /** Unit vector from a surface toward the sun. */ towardSun: Vec3 }
    | { kind: 'local'; /** Point and spot light positions; one is chosen uniformly per hit. */ positions: Vec3[] };

/**
 * The sun where the manifest has a directional light (the first), else every
 * point and spot light (a hit picks one uniformly). Capsaicin samples one
 * light per hit from its light sampler; uniform is the simplest stand-in
 * with the same one-ray-per-hit cost.
 */
export function rcLightsFromManifest(lights: readonly ManifestLight[]): RcLights {
    const sun = lights.find((light) => light.type === 'directional');
    if (sun?.type === 'directional') {
        const travel = normalizeVector(sun.direction);
        return { kind: 'sun', towardSun: [-travel[0], -travel[1], -travel[2]] };
    }
    const positions = lights.flatMap((light) => (light.type === 'directional' ? [] : [[...light.position] as Vec3]));
    if (positions.length === 0) throw new Error('the scene has no light to aim shadow rays at');
    return { kind: 'local', positions };
}

export interface RcShadowRays {
    rays: Float32Array;
    count: number;
    /** Hits that got no shadow ray because they face away from the light sampled for them. */
    facingAwayCount: number;
}

/**
 * One any-hit ray per surface toward one light: from the platform's offset
 * origin, toward the sun (length `SUN_SHADOW_RAY_LENGTH`, as the platform's
 * shadow rays) or toward a uniformly chosen local light's position (ending
 * at the light). Surfaces facing away from their light get none.
 */
export function makeLightShadowRays(surfaces: Pick<HitSurfaces, 'positions' | 'normals' | 'count'>, lights: RcLights, random: Random): RcShadowRays {
    const output = new Float32Array(surfaces.count * RAY_FLOATS);
    let written = 0;
    let facingAwayCount = 0;
    for (let surface = 0; surface < surfaces.count; surface++) {
        const position = surfaces.positions.subarray(surface * 3, surface * 3 + 3);
        const normal = surfaces.normals.subarray(surface * 3, surface * 3 + 3);
        const origin = offsetRayOrigin(position, normal);
        let direction: Vec3;
        let length: number;
        if (lights.kind === 'sun') {
            direction = lights.towardSun;
            length = SUN_SHADOW_RAY_LENGTH;
        } else {
            const light = lights.positions[Math.min(lights.positions.length - 1, Math.floor(random() * lights.positions.length))];
            const toLight: Vec3 = [light[0] - origin[0], light[1] - origin[1], light[2] - origin[2]];
            length = Math.hypot(toLight[0], toLight[1], toLight[2]);
            if (!(length > 0)) {
                facingAwayCount++;
                continue;
            }
            direction = [toLight[0] / length, toLight[1] / length, toLight[2] / length];
        }
        if (normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2] <= 0) {
            facingAwayCount++;
            continue;
        }
        writeRay(output, written, origin, direction, length);
        written++;
    }
    return { rays: output.slice(0, written * RAY_FLOATS), count: written, facingAwayCount };
}

// ---- Checks --------------------------------------------------------------------------------

/** How far a stored direction's length may be from 1 (f32 rounding is about 1e-7). */
const UNIT_LENGTH_TOLERANCE = 1e-5;

/**
 * Problems with a packed ray set, at most `limit` of them: a component that
 * isn't finite, a direction that isn't unit length, or an empty or inverted
 * interval. Empty means every ray is usable.
 */
export function rayProblems(name: string, rays: Float32Array, limit = 5): string[] {
    const problems: string[] = [];
    const count = rays.length / RAY_FLOATS;
    for (let ray = 0; ray < count && problems.length < limit; ray++) {
        const base = ray * RAY_FLOATS;
        const values = rays.subarray(base, base + RAY_FLOATS);
        if (!values.every(Number.isFinite)) {
            problems.push(`${name} ray ${ray} has a value that isn't finite (${Array.from(values).join(', ')})`);
            continue;
        }
        const length = Math.hypot(values[4], values[5], values[6]);
        if (Math.abs(length - 1) > UNIT_LENGTH_TOLERANCE) problems.push(`${name} ray ${ray}'s direction has length ${length}`);
        if (!(values[3] >= 0 && values[7] > values[3])) problems.push(`${name} ray ${ray} has the interval [${values[3]}, ${values[7]}]`);
    }
    return problems;
}
