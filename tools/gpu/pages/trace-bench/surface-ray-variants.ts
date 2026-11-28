/**
 * Ray sets for the surface-ray investigation (P6.6, from the Mac track's RC5.4): variants
 * that change one thing at a time between RC1.5's screen-probe rays (64
 * cosine rays per probe from visible surfaces) and the probe-grid bench's
 * rays (a shared sphere of directions from a grid of points in the air).
 *
 * Every function here is pure and CPU-only, so it can be unit-tested; the
 * page (surface-rays-bench.ts) makes the sets, times them and counts the
 * walk's work on each.
 */

import { HIT_RECORD_BYTES, HIT_RECORD_WORD, RAY_FLOATS } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import type { Random } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { RAY_FAR } from './config.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';
import { cosineDirection, hitSurfaces, offsetRayOrigin, tangentFrame } from './rc-rays.ts';
import type { Box, Vec3 } from './rays.ts';
import { COUNTED_WORD, COUNTED_WORDS, NO_ORIGIN } from './counted-walk-wgsl.ts';

// ---- Probes on surfaces --------------------------------------------------------------------

/** Screen probes with what the variants need beyond rc-rays' `RcProbes`: the point on the surface and the triangle it is on. */
export interface SurfaceProbes {
    /** Per probe, `x, y, z` on the triangle (moved slightly inside it, `giSurfaceAt`), before any push off the surface. */
    positions: Float64Array;
    /** Per probe, the unit geometric normal on the camera's side. */
    normals: Float64Array;
    /** Per probe, the platform instance and the triangle within its mesh that the probe sits on. */
    originIds: Uint32Array;
    count: number;
}

/** The probes the placement rays found, as rc-rays' `placeProbes` places them, with their triangles. */
export function surfaceProbes(
    placementRays: Float32Array,
    hitRecords: ArrayBuffer,
    meshes: readonly BenchMeshGeometry[],
    instances: readonly BenchInstance[],
): SurfaceProbes {
    const surfaces = hitSurfaces(placementRays, hitRecords, meshes, instances);
    const words = new Uint32Array(hitRecords);
    const recordWords = HIT_RECORD_BYTES / 4;
    const originIds = new Uint32Array(surfaces.count * 2);
    for (let probe = 0; probe < surfaces.count; probe++) {
        const record = surfaces.rayIndex[probe] * recordWords;
        originIds[probe * 2] = words[record + HIT_RECORD_WORD.instance];
        originIds[probe * 2 + 1] = words[record + HIT_RECORD_WORD.triangle];
    }
    return { positions: surfaces.positions, normals: surfaces.normals, originIds, count: surfaces.count };
}

/**
 * Where each probe's rays start. `'platform'` is the platform's push
 * (`giShadeOffsetRayOrigin`, 256 rounding steps of the point's largest
 * coordinate, about 0.2 mm in Sponza); a number is a push of that many
 * metres along the normal instead.
 */
export function probeOrigins(probes: SurfaceProbes, push: 'platform' | number): Float64Array {
    const origins = new Float64Array(probes.count * 3);
    for (let probe = 0; probe < probes.count; probe++) {
        const position = probes.positions.subarray(probe * 3, probe * 3 + 3);
        const normal = probes.normals.subarray(probe * 3, probe * 3 + 3);
        if (push === 'platform') {
            origins.set(offsetRayOrigin(position, normal), probe * 3);
        } else {
            for (let axis = 0; axis < 3; axis++) origins[probe * 3 + axis] = position[axis] + normal[axis] * push;
        }
    }
    return origins;
}

// ---- Directions ------------------------------------------------------------------------------

/**
 * How a probe's rays are spread:
 *
 * - `cosine-stratified`: RC1.5's probe rays (rc-rays' `makeProbeRays`): an
 *   8x8 grid over the cosine hemisphere, jittered, turned by a random angle per probe;
 * - `cosine-random`: cosine-weighted about the normal, each ray drawn on its own;
 * - `uniform-hemisphere`: uniform over the hemisphere about the normal;
 * - `uniform-sphere`: uniform over the whole sphere (half of them go into the surface);
 * - `shared-sphere`: the probe-grid bench's directions: one fixed set of
 *   sphere directions, the same for every probe.
 */
export type DirectionKind = 'cosine-stratified' | 'cosine-random' | 'uniform-hemisphere' | 'uniform-sphere' | 'shared-sphere';

/** A uniform direction on the hemisphere about `normal` (cos θ uniform in [0, 1)). */
function uniformHemisphereDirection(normal: ArrayLike<number>, frame: { tangent: Vec3; bitangent: Vec3 }, u1: number, u2: number): Vec3 {
    const z = u1;
    const radius = Math.sqrt(Math.max(0, 1 - z * z));
    const angle = 2 * Math.PI * u2;
    const x = radius * Math.cos(angle);
    const y = radius * Math.sin(angle);
    const { tangent, bitangent } = frame;
    return [0, 1, 2].map((axis) => tangent[axis] * x + bitangent[axis] * y + normal[axis] * z) as Vec3;
}

/** A uniform direction on the sphere. */
function uniformSphereDirection(u1: number, u2: number): Vec3 {
    const z = 1 - 2 * u1;
    const radius = Math.sqrt(Math.max(0, 1 - z * z));
    const angle = 2 * Math.PI * u2;
    return [radius * Math.cos(angle), radius * Math.sin(angle), z];
}

/** Writes one `GiRay`, the direction rounded to f32 and normalised again (as rc-rays does). */
function writeRay(output: Float32Array, slot: number, origin: ArrayLike<number>, direction: ArrayLike<number>, tMin: number, tMax: number): void {
    const dx = Math.fround(direction[0]);
    const dy = Math.fround(direction[1]);
    const dz = Math.fround(direction[2]);
    const length = Math.hypot(dx, dy, dz);
    const base = slot * RAY_FLOATS;
    output[base] = origin[0];
    output[base + 1] = origin[1];
    output[base + 2] = origin[2];
    output[base + 3] = tMin;
    output[base + 4] = dx / length;
    output[base + 5] = dy / length;
    output[base + 6] = dz / length;
    output[base + 7] = tMax;
}

export interface ProbeRayOptions {
    /** Per probe `x, y, z`. */
    origins: Float64Array;
    /** Per probe, the unit normal the hemisphere kinds are spread about. */
    normals: Float64Array;
    count: number;
    raysPerProbe: number;
    kind: DirectionKind;
    random: Random;
    /** `shared-sphere` only: `raysPerProbe` unit directions, `x, y, z` each. */
    sharedDirections?: Float32Array;
}

/**
 * `raysPerProbe` closest-hit rays per probe, probe-major, from `tMin` 0 to
 * the bench's far end. `cosine-stratified` consumes the generator exactly as
 * rc-rays' `makeProbeRays` does, so with RC1.5's seed and probes it makes
 * RC1.5's probe set.
 */
export function makeVariantProbeRays(options: ProbeRayOptions): Float32Array {
    const { origins, normals, count, raysPerProbe, kind, random } = options;
    const side = Math.round(Math.sqrt(raysPerProbe));
    if (kind === 'cosine-stratified' && side * side !== raysPerProbe) throw new RangeError(`rays per probe must be a square number, got ${raysPerProbe}`);
    if (kind === 'shared-sphere' && (options.sharedDirections?.length ?? 0) !== raysPerProbe * 3) {
        throw new RangeError('shared-sphere needs one shared direction per ray of a probe');
    }
    const rays = new Float32Array(count * raysPerProbe * RAY_FLOATS);
    for (let probe = 0; probe < count; probe++) {
        const origin = origins.subarray(probe * 3, probe * 3 + 3);
        const normal = normals.subarray(probe * 3, probe * 3 + 3);
        const frame = tangentFrame(normal);
        const rotation = kind === 'cosine-stratified' ? random() : 0;
        for (let cell = 0; cell < raysPerProbe; cell++) {
            let direction: Vec3;
            if (kind === 'cosine-stratified') {
                const u1 = (Math.floor(cell / side) + random()) / side;
                const u2 = ((cell % side) + random()) / side + rotation;
                direction = cosineDirection(normal, frame, u1, u2 - Math.floor(u2));
            } else if (kind === 'cosine-random') {
                direction = cosineDirection(normal, frame, random(), random());
            } else if (kind === 'uniform-hemisphere') {
                direction = uniformHemisphereDirection(normal, frame, random(), random());
            } else if (kind === 'uniform-sphere') {
                direction = uniformSphereDirection(random(), random());
            } else {
                const shared = options.sharedDirections as Float32Array;
                direction = [shared[cell * 3], shared[cell * 3 + 1], shared[cell * 3 + 2]];
            }
            writeRay(rays, probe * raysPerProbe + cell, origin, direction, 0, RAY_FAR);
        }
    }
    return rays;
}

/** Per ray, the instance and triangle it leaves from, for probe-major rays of `raysPerProbe` per probe. */
export function rayOriginIds(probeOriginIds: Uint32Array | null, probeCount: number, raysPerProbe: number): Uint32Array {
    const ids = new Uint32Array(probeCount * raysPerProbe * 2).fill(NO_ORIGIN);
    if (!probeOriginIds) return ids;
    for (let probe = 0; probe < probeCount; probe++) {
        for (let ray = 0; ray < raysPerProbe; ray++) {
            const slot = (probe * raysPerProbe + ray) * 2;
            ids[slot] = probeOriginIds[probe * 2];
            ids[slot + 1] = probeOriginIds[probe * 2 + 1];
        }
    }
    return ids;
}

// ---- Changing a set ---------------------------------------------------------------------------

/** A copy of `rays` with every ray's interval set (either end left alone when undefined). */
export function withInterval(rays: Float32Array, interval: { tMin?: number; tMax?: number }): Float32Array {
    const copy = rays.slice();
    for (let base = 0; base < copy.length; base += RAY_FLOATS) {
        if (interval.tMin !== undefined) copy[base + 3] = interval.tMin;
        if (interval.tMax !== undefined) copy[base + 7] = interval.tMax;
    }
    return copy;
}

/** Rays (and their per-ray records of `wordsPerRay` words) in a new order: slot `i` takes ray `order[i]`. */
export function reorder<T extends Float32Array | Uint32Array>(values: T, order: Uint32Array, wordsPerRay: number): T {
    const copy = values.slice() as T;
    for (let slot = 0; slot < order.length; slot++) {
        const from = order[slot] * wordsPerRay;
        copy.set(values.subarray(from, from + wordsPerRay), slot * wordsPerRay);
    }
    return copy;
}

/** A uniformly random order of `count` rays (Fisher-Yates). */
export function shuffledOrder(count: number, random: Random): Uint32Array {
    const order = new Uint32Array(count);
    for (let index = 0; index < count; index++) order[index] = index;
    for (let index = count - 1; index > 0; index--) {
        const swap = Math.floor(random() * (index + 1));
        const value = order[swap];
        order[swap] = order[index];
        order[index] = value;
    }
    return order;
}

/**
 * Ray `k` of every probe next to each other: probe-major becomes cell-major.
 * Neighbouring threads then trace the same stratum of neighbouring tiles'
 * probes (similar directions where the normals agree, nearby origins).
 */
export function cellMajorOrder(probeCount: number, raysPerProbe: number): Uint32Array {
    const order = new Uint32Array(probeCount * raysPerProbe);
    for (let cell = 0; cell < raysPerProbe; cell++) {
        for (let probe = 0; probe < probeCount; probe++) order[cell * probeCount + probe] = probe * raysPerProbe + cell;
    }
    return order;
}

/** Spreads the low 10 bits of `value` to every third bit (for a 30-bit Morton code). */
function spreadBits(value: number): number {
    let x = value & 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    return x >>> 0;
}

/** The 30-bit Morton code of a point quantised to 1024 steps per axis of `box`. */
export function mortonCode(point: ArrayLike<number>, box: Box): number {
    const quantised = [0, 1, 2].map((axis) => {
        const extent = Math.max(box.max[axis] - box.min[axis], 1e-9);
        return Math.min(1023, Math.max(0, Math.floor(((point[axis] - box.min[axis]) / extent) * 1024)));
    });
    return (spreadBits(quantised[0]) | (spreadBits(quantised[1]) << 1) | (spreadBits(quantised[2]) << 2)) >>> 0;
}

/**
 * A direction's bin: the octahedral map of the unit direction, quantised to
 * `resolution` x `resolution` cells. Nearby directions share a bin.
 */
export function directionBin(dx: number, dy: number, dz: number, resolution: number): number {
    const sum = Math.abs(dx) + Math.abs(dy) + Math.abs(dz);
    let u = dx / sum;
    let v = dy / sum;
    if (dz < 0) {
        const foldedU = (1 - Math.abs(v)) * (u >= 0 ? 1 : -1);
        const foldedV = (1 - Math.abs(u)) * (v >= 0 ? 1 : -1);
        u = foldedU;
        v = foldedV;
    }
    const column = Math.min(resolution - 1, Math.floor(((u + 1) / 2) * resolution));
    const row = Math.min(resolution - 1, Math.floor(((v + 1) / 2) * resolution));
    return row * resolution + column;
}

/**
 * An order sorted by a key per ray. `'direction-then-origin'`: by direction
 * bin (8 x 8 octahedral), then by the origin's Morton code, so threads next
 * to each other run nearly parallel from nearby points.
 * `'origin-then-direction'`: by Morton code first, then by direction bin.
 */
export function sortedOrder(rays: Float32Array, box: Box, kind: 'direction-then-origin' | 'origin-then-direction'): Uint32Array {
    const count = rays.length / RAY_FLOATS;
    const keys = new Float64Array(count);
    const directionBins = 64;
    for (let ray = 0; ray < count; ray++) {
        const base = ray * RAY_FLOATS;
        const morton = mortonCode(rays.subarray(base, base + 3), box);
        const bin = directionBin(rays[base + 4], rays[base + 5], rays[base + 6], 8);
        // Both keys fit in a double exactly: 6 bits of bin and 30 of Morton code.
        keys[ray] = kind === 'direction-then-origin' ? bin * 2 ** 30 + morton : morton * directionBins + bin;
    }
    const order = Uint32Array.from({ length: count }, (_, index) => index);
    // Stable on ties: Array.prototype.sort is stable, and the index breaks nothing.
    return Uint32Array.from(Array.from(order).sort((a, b) => keys[a] - keys[b] || a - b));
}

/** Replaces each probe's origin (rays probe-major, `raysPerProbe` each) with `origins[probe]`. */
export function transplantOrigins(rays: Float32Array, raysPerProbe: number, origins: Float64Array): Float32Array {
    const copy = rays.slice();
    const probes = copy.length / RAY_FLOATS / raysPerProbe;
    if (origins.length < probes * 3) throw new RangeError(`${probes} probes need ${probes} origins, got ${origins.length / 3}`);
    for (let probe = 0; probe < probes; probe++) {
        for (let ray = 0; ray < raysPerProbe; ray++) {
            const base = (probe * raysPerProbe + ray) * RAY_FLOATS;
            copy[base] = origins[probe * 3];
            copy[base + 1] = origins[probe * 3 + 1];
            copy[base + 2] = origins[probe * 3 + 2];
        }
    }
    return copy;
}

// ---- What the counted walk found ---------------------------------------------------------------

/** Means of the counted walk's counters over a set, and how evenly they fall within SIMD groups. */
export interface CountSummary {
    rays: number;
    hitFraction: number;
    /** Median closest-hit distance over the rays that hit (m). */
    medianHitT: number;
    meanSteps: number;
    p50Steps: number;
    p90Steps: number;
    p99Steps: number;
    meanMeshNodes: number;
    meanTopLevelNodes: number;
    meanInstanceEntries: number;
    meanTriangleTests: number;
    meanHitsAccepted: number;
    meanOriginBoxes: number;
    meanStepsBeforeFirstHit: number;
    meanMeshStepsBeforeFirstHit: number;
    meanStepsInOriginInstance: number;
    /** Share of rays that tested the triangle they left from at least once. */
    originTriangleTestedFraction: number;
    /** Share of rays whose first accepted hit was the triangle they left from (a self-hit). */
    selfHitFraction: number;
    meanMaxStack: number;
    /** Instances entered in which the walk found nothing to keep. */
    meanInstanceEntriesWithoutHit: number;
    /**
     * Per group of `width` consecutive rays (one SIMD group, or one
     * workgroup), the steps of its slowest ray: the group runs that long.
     * Σ mean / Σ max is the share of lanes doing work, 1 when every ray of a
     * group takes the same number of steps.
     */
    simd32Efficiency: number;
    workgroup64Efficiency: number;
    /** Σ max over SIMD groups of 32, per ray: the steps the hardware pays for. */
    simd32StepsPaidPerRay: number;
}

function percentile(sorted: Uint32Array, fraction: number): number {
    if (sorted.length === 0) return Number.NaN;
    return sorted[Math.min(sorted.length - 1, Math.floor(fraction * sorted.length))];
}

function groupEfficiency(steps: Uint32Array, width: number): { efficiency: number; paidPerRay: number } {
    let sumOfSteps = 0;
    let sumOfPaid = 0;
    for (let first = 0; first < steps.length; first += width) {
        let maximum = 0;
        const end = Math.min(steps.length, first + width);
        for (let ray = first; ray < end; ray++) {
            sumOfSteps += steps[ray];
            if (steps[ray] > maximum) maximum = steps[ray];
        }
        // A partial last group still occupies a whole SIMD group.
        sumOfPaid += maximum * width;
    }
    return { efficiency: sumOfPaid > 0 ? sumOfSteps / sumOfPaid : 1, paidPerRay: steps.length > 0 ? sumOfPaid / steps.length : 0 };
}

/** Summarises the counted kernel's output (`COUNTED_WORDS` per ray). */
export function summariseCounts(counts: ArrayBuffer, rayCount: number): CountSummary {
    const words = new Uint32Array(counts);
    const floats = new Float32Array(counts);
    if (words.length < rayCount * COUNTED_WORDS) throw new RangeError('fewer counter records than rays');
    const steps = new Uint32Array(rayCount);
    const sums = new Float64Array(COUNTED_WORDS);
    const hitDistances: number[] = [];
    let hits = 0;
    let originTested = 0;
    let selfHits = 0;
    for (let ray = 0; ray < rayCount; ray++) {
        const base = ray * COUNTED_WORDS;
        for (let word = COUNTED_WORD.steps; word < COUNTED_WORDS; word++) sums[word] += words[base + word];
        steps[ray] = words[base + COUNTED_WORD.steps];
        const t = floats[base + COUNTED_WORD.t];
        if (t >= 0) {
            hits++;
            hitDistances.push(t);
        }
        if (words[base + COUNTED_WORD.originTriangleTests] > 0) originTested++;
        if (words[base + COUNTED_WORD.firstHitIsOrigin] > 0) selfHits++;
    }
    const mean = (word: number): number => (rayCount > 0 ? sums[word] / rayCount : 0);
    const sortedSteps = steps.slice().sort();
    hitDistances.sort((a, b) => a - b);
    const simd = groupEfficiency(steps, 32);
    return {
        rays: rayCount,
        hitFraction: rayCount > 0 ? hits / rayCount : 0,
        medianHitT: hitDistances.length > 0 ? hitDistances[hitDistances.length >> 1] : Number.NaN,
        meanSteps: mean(COUNTED_WORD.steps),
        p50Steps: percentile(sortedSteps, 0.5),
        p90Steps: percentile(sortedSteps, 0.9),
        p99Steps: percentile(sortedSteps, 0.99),
        meanMeshNodes: mean(COUNTED_WORD.meshNodes),
        meanTopLevelNodes: mean(COUNTED_WORD.topLevelNodes),
        meanInstanceEntries: mean(COUNTED_WORD.instanceEntries),
        meanTriangleTests: mean(COUNTED_WORD.triangleTests),
        meanHitsAccepted: mean(COUNTED_WORD.hitsAccepted),
        meanOriginBoxes: mean(COUNTED_WORD.originBoxes),
        meanStepsBeforeFirstHit: mean(COUNTED_WORD.stepsBeforeFirstHit),
        meanMeshStepsBeforeFirstHit: mean(COUNTED_WORD.meshStepsBeforeFirstHit),
        meanStepsInOriginInstance: mean(COUNTED_WORD.stepsInOriginInstance),
        originTriangleTestedFraction: rayCount > 0 ? originTested / rayCount : 0,
        selfHitFraction: rayCount > 0 ? selfHits / rayCount : 0,
        meanMaxStack: mean(COUNTED_WORD.maxStack),
        meanInstanceEntriesWithoutHit: mean(COUNTED_WORD.instanceEntries) - mean(COUNTED_WORD.instancesWithHit),
        simd32Efficiency: simd.efficiency,
        workgroup64Efficiency: groupEfficiency(steps, 64).efficiency,
        simd32StepsPaidPerRay: simd.paidPerRay,
    };
}

/**
 * How many instances' world boxes contain each sampled ray origin (every
 * `stride`-th ray): the top-level leaves a ray starts inside, which it must
 * enter whatever its direction.
 */
export function meanInstanceBoxesAtOrigins(rays: Float32Array, instanceBoxes: readonly Box[], stride: number): number {
    const count = rays.length / RAY_FLOATS;
    let total = 0;
    let sampled = 0;
    for (let ray = 0; ray < count; ray += stride) {
        const base = ray * RAY_FLOATS;
        const x = rays[base];
        const y = rays[base + 1];
        const z = rays[base + 2];
        for (const box of instanceBoxes) {
            if (x >= box.min[0] && x <= box.max[0] && y >= box.min[1] && y <= box.max[1] && z >= box.min[2] && z <= box.max[2]) total++;
        }
        sampled++;
    }
    return sampled > 0 ? total / sampled : 0;
}
