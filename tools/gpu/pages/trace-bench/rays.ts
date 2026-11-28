/**
 * The benchmark's rays: DDGI-shaped, generated once and shared by every tracer.
 *
 * - Closest-hit rays come from a grid of probes that fills a box. Every probe
 *   fires the same set of fixed-seed random directions, which is what DDGI does
 *   within one frame (the set is rotated between frames, not between probes).
 * - Any-hit (shadow) rays start at each closest hit, nudged off the surface
 *   along its normal, and point at the sun. DDGI traces exactly these to
 *   decide whether a hit point is lit. Hits facing away from the sun get no
 *   shadow ray, since the shading would skip them too.
 *
 * Rays are packed as the tracer's `GiRay`: `origin (3), tMin, direction (3), tMax`
 * as eight 32-bit floats. That is also two `vec4f` per ray, which is how the
 * three-mesh-bvh kernel reads the same buffer.
 */

import { HIT_RECORD_BYTES, RAY_FLOATS } from '../../../../packages/core/src/platform/trace/index.ts';
import { createRandom, randomUnitVector } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { RAY_FAR } from './config.ts';
import type { BenchInstance, BenchMeshGeometry } from './gltf-geometry.ts';

export type Vec3 = [number, number, number];

export interface Box {
    min: Vec3;
    max: Vec3;
}

export interface ProbeGrid {
    /** Probes along x, y and z. */
    dimensions: Vec3;
    probeCount: number;
    /** Centre of the first probe. */
    firstProbe: Vec3;
    /** Distance between neighbouring probes on each axis. */
    spacing: Vec3;
}

/**
 * A grid of about `targetProbeCount` probes with roughly cubic cells that
 * fills `box`. Probes sit at cell centres, so none lies exactly on the box.
 */
export function makeProbeGrid(box: Box, targetProbeCount: number): ProbeGrid {
    const size = [0, 1, 2].map((axis) => Math.max(box.max[axis] - box.min[axis], 1e-6));
    const volume = size[0] * size[1] * size[2];
    const cellEdge = Math.cbrt(volume / Math.max(1, targetProbeCount));
    const dimensions = size.map((extent) => Math.max(1, Math.round(extent / cellEdge))) as Vec3;
    const spacing = size.map((extent, axis) => extent / dimensions[axis]) as Vec3;
    const firstProbe = [0, 1, 2].map((axis) => box.min[axis] + spacing[axis] / 2) as Vec3;
    return {
        dimensions,
        probeCount: dimensions[0] * dimensions[1] * dimensions[2],
        firstProbe,
        spacing,
    };
}

export function probePosition(grid: ProbeGrid, probeIndex: number): Vec3 {
    const [nx, ny] = grid.dimensions;
    const x = probeIndex % nx;
    const y = Math.floor(probeIndex / nx) % ny;
    const z = Math.floor(probeIndex / (nx * ny));
    return [
        grid.firstProbe[0] + x * grid.spacing[0],
        grid.firstProbe[1] + y * grid.spacing[1],
        grid.firstProbe[2] + z * grid.spacing[2],
    ];
}

/** `count` uniformly distributed unit vectors from a fixed seed, `x, y, z` each. */
export function makeSphereDirections(count: number, seed: number): Float32Array {
    const random = createRandom(seed);
    const directions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const direction = randomUnitVector(random);
        // Rounding to f32 can leave the length a hair off 1, so normalize what is stored.
        const stored = direction.map(Math.fround);
        const length = Math.hypot(stored[0], stored[1], stored[2]);
        directions.set([stored[0] / length, stored[1] / length, stored[2] / length], i * 3);
    }
    return directions;
}

/** Closest-hit rays: every probe fires every direction. Probe-major order. */
export function makeProbeRays(grid: ProbeGrid, directions: Float32Array): Float32Array {
    const directionCount = directions.length / 3;
    const rays = new Float32Array(grid.probeCount * directionCount * RAY_FLOATS);
    for (let probe = 0; probe < grid.probeCount; probe++) {
        const origin = probePosition(grid, probe);
        for (let d = 0; d < directionCount; d++) {
            const base = (probe * directionCount + d) * RAY_FLOATS;
            rays[base] = origin[0];
            rays[base + 1] = origin[1];
            rays[base + 2] = origin[2];
            rays[base + 3] = 0;
            rays[base + 4] = directions[d * 3];
            rays[base + 5] = directions[d * 3 + 1];
            rays[base + 6] = directions[d * 3 + 2];
            rays[base + 7] = RAY_FAR;
        }
    }
    return rays;
}

export function normalizeVector(vector: readonly number[]): Vec3 {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    if (!(length > 0)) throw new RangeError('cannot normalize a zero-length vector');
    return [vector[0] / length, vector[1] / length, vector[2] / length];
}

/** The box around all instances of all meshes, in world space. */
export function worldBounds(meshes: readonly BenchMeshGeometry[], instances: readonly BenchInstance[]): Box {
    const meshBoxes = meshes.map((mesh) => {
        const low: Vec3 = [Infinity, Infinity, Infinity];
        const high: Vec3 = [-Infinity, -Infinity, -Infinity];
        const { positions } = mesh;
        for (let i = 0; i < positions.length; i += 3) {
            for (let axis = 0; axis < 3; axis++) {
                const value = positions[i + axis];
                if (value < low[axis]) low[axis] = value;
                if (value > high[axis]) high[axis] = value;
            }
        }
        return { min: low, max: high };
    });
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (const instance of instances) {
        const box = meshBoxes[instance.meshIndex];
        const m = instance.objectToWorld;
        for (let corner = 0; corner < 8; corner++) {
            const x = corner & 1 ? box.max[0] : box.min[0];
            const y = corner & 2 ? box.max[1] : box.min[1];
            const z = corner & 4 ? box.max[2] : box.min[2];
            for (let axis = 0; axis < 3; axis++) {
                const value = m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis];
                if (value < min[axis]) min[axis] = value;
                if (value > max[axis]) max[axis] = value;
            }
        }
    }
    return { min, max };
}

export interface ShadowRayInput {
    /** The closest-hit rays, packed. */
    closestRays: Float32Array;
    /** One `GiHitRecord` per ray (`HIT_RECORD_BYTES` each), as read back from the GPU. */
    hitRecords: ArrayBuffer;
    meshes: readonly BenchMeshGeometry[];
    /** Indexed by the instance index the tracer reports. */
    instances: readonly BenchInstance[];
    /** Unit vector from the surface toward the sun. */
    towardSun: Vec3;
    /** How far to move the start along the surface normal. */
    normalOffset: number;
}

export interface ShadowRays {
    rays: Float32Array;
    count: number;
    /** Closest-hit rays that hit something. */
    hitCount: number;
    /** Hits that got no shadow ray because they face away from the sun. */
    facingAwayCount: number;
}

/**
 * Builds the any-hit rays from the closest-hit results. The surface normal is
 * the triangle's geometric normal in world space, turned to the side the ray
 * came from.
 */
export function makeShadowRays(input: ShadowRayInput): ShadowRays {
    const { closestRays, hitRecords, meshes, instances, towardSun, normalOffset } = input;
    const rayCount = closestRays.length / RAY_FLOATS;
    const recordWords = HIT_RECORD_BYTES / 4;
    if (hitRecords.byteLength < rayCount * HIT_RECORD_BYTES) throw new RangeError('fewer hit records than rays');
    const floats = new Float32Array(hitRecords);
    const words = new Uint32Array(hitRecords);

    const output = new Float32Array(rayCount * RAY_FLOATS);
    let written = 0;
    let hitCount = 0;
    let facingAwayCount = 0;

    const corner = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let ray = 0; ray < rayCount; ray++) {
        const record = ray * recordWords;
        const t = floats[record];
        if (!(t >= 0)) continue;
        hitCount++;

        const instance = instances[words[record + 1]];
        if (!instance) throw new RangeError(`ray ${ray} reports instance ${words[record + 1]}, which does not exist`);
        const mesh = meshes[instance.meshIndex];
        const triangle = words[record + 2];
        const m = instance.objectToWorld;
        for (let c = 0; c < 3; c++) {
            const vertex = mesh.indices[triangle * 3 + c] * 3;
            const x = mesh.positions[vertex];
            const y = mesh.positions[vertex + 1];
            const z = mesh.positions[vertex + 2];
            for (let axis = 0; axis < 3; axis++) corner[c * 3 + axis] = m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis];
        }
        const ex = corner[3] - corner[0];
        const ey = corner[4] - corner[1];
        const ez = corner[5] - corner[2];
        const fx = corner[6] - corner[0];
        const fy = corner[7] - corner[1];
        const fz = corner[8] - corner[2];
        let nx = ey * fz - ez * fy;
        let ny = ez * fx - ex * fz;
        let nz = ex * fy - ey * fx;
        const length = Math.hypot(nx, ny, nz);
        if (!(length > 0)) continue; // a collapsed triangle has no normal
        nx /= length;
        ny /= length;
        nz /= length;

        const base = ray * RAY_FLOATS;
        const dx = closestRays[base + 4];
        const dy = closestRays[base + 5];
        const dz = closestRays[base + 6];
        // Turn the normal to the side the ray arrived from.
        if (nx * dx + ny * dy + nz * dz > 0) {
            nx = -nx;
            ny = -ny;
            nz = -nz;
        }
        if (nx * towardSun[0] + ny * towardSun[1] + nz * towardSun[2] <= 0) {
            facingAwayCount++;
            continue;
        }

        const out = written * RAY_FLOATS;
        output[out] = closestRays[base] + dx * t + nx * normalOffset;
        output[out + 1] = closestRays[base + 1] + dy * t + ny * normalOffset;
        output[out + 2] = closestRays[base + 2] + dz * t + nz * normalOffset;
        output[out + 3] = 0;
        output[out + 4] = towardSun[0];
        output[out + 5] = towardSun[1];
        output[out + 6] = towardSun[2];
        output[out + 7] = RAY_FAR;
        written++;
    }
    return { rays: output.slice(0, written * RAY_FLOATS), count: written, hitCount, facingAwayCount };
}
