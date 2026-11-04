/**
 * Deterministic test scenes and rays for the BVH tests and the GPU oracle
 * page. Not part of the tracer; nothing in production imports this file.
 *
 * Everything is seeded so a failing case can be reproduced exactly.
 */

import type { MeshGeometry } from './mesh-bvh.ts';
import type { TraceRay } from './trace-gpu.ts';

export type Random = () => number;

/** mulberry32: a small seeded generator returning numbers in [0, 1). */
export function createRandom(seed: number): Random {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let mixed = state;
        mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
        mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}

export type Vector3 = [number, number, number];

export function randomUnitVector(random: Random): Vector3 {
    const z = 1 - 2 * random();
    const angle = 2 * Math.PI * random();
    const radius = Math.sqrt(Math.max(0, 1 - z * z));
    return [radius * Math.cos(angle), radius * Math.sin(angle), z];
}

/** Rounds to f32 so the CPU oracle and the GPU see exactly the same ray. */
export function toFloat32Ray(origin: readonly number[], direction: readonly number[], tMin: number, tMax: number): TraceRay {
    return {
        origin: [Math.fround(origin[0]), Math.fround(origin[1]), Math.fround(origin[2])],
        direction: [Math.fround(direction[0]), Math.fround(direction[1]), Math.fround(direction[2])],
        tMin: Math.fround(tMin),
        tMax: Math.fround(tMax),
    };
}

/** Unindexed triangles scattered in a cube of half-size `extent`, each about `size` wide. */
export function randomTriangleSoup(random: Random, triangleCount: number, extent: number, size: number): MeshGeometry {
    const positions = new Float32Array(triangleCount * 9);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
        const center = [0, 1, 2].map(() => (random() * 2 - 1) * extent);
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) {
                positions[triangle * 9 + corner * 3 + axis] = center[axis] + (random() * 2 - 1) * size;
            }
        }
    }
    return { positions };
}

/**
 * A flat grid of `cells` x `cells` squares in the XY plane (z = 0) spanning
 * `[-half, half]`, each square split into two triangles along the same
 * diagonal. Vertices are shared, so every interior edge is shared by exactly
 * two triangles.
 */
export function tessellatedQuad(cells: number, half: number): MeshGeometry {
    const verticesPerRow = cells + 1;
    const positions = new Float32Array(verticesPerRow * verticesPerRow * 3);
    for (let row = 0; row <= cells; row++) {
        for (let column = 0; column <= cells; column++) {
            const base = (row * verticesPerRow + column) * 3;
            positions[base] = Math.fround(-half + (2 * half * column) / cells);
            positions[base + 1] = Math.fround(-half + (2 * half * row) / cells);
            positions[base + 2] = 0;
        }
    }
    const indices = new Uint32Array(cells * cells * 6);
    let slot = 0;
    for (let row = 0; row < cells; row++) {
        for (let column = 0; column < cells; column++) {
            const a = row * verticesPerRow + column;
            const b = a + 1;
            const c = a + verticesPerRow;
            const d = c + 1;
            // Counter-clockwise seen from +Z, diagonal from a to d.
            indices.set([a, b, d, a, d, c], slot);
            slot += 6;
        }
    }
    return { positions, indices };
}

/**
 * A random affine object-to-world transform, column-major: rotation, then
 * per-axis scale (non-uniform), optionally mirrored on x, then translation.
 */
export function randomAffine(random: Random, options: { extent: number; mirror?: boolean }): number[] {
    // Random rotation from a unit quaternion.
    const [qx, qy, qz] = randomUnitVector(random);
    const angle = random() * Math.PI * 2;
    const s = Math.sin(angle / 2);
    const x = qx * s, y = qy * s, z = qz * s, w = Math.cos(angle / 2);
    const rotation = [
        1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
        2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y),
    ];
    const scale = [0.5 + random() * 1.5, 0.5 + random() * 1.5, 0.5 + random() * 1.5];
    if (options.mirror) scale[0] = -scale[0];
    const translation = [0, 1, 2].map(() => (random() * 2 - 1) * options.extent);
    // Each column of the rotation is scaled by that axis's scale.
    return [
        rotation[0] * scale[0], rotation[1] * scale[0], rotation[2] * scale[0], 0,
        rotation[3] * scale[1], rotation[4] * scale[1], rotation[5] * scale[1], 0,
        rotation[6] * scale[2], rotation[7] * scale[2], rotation[8] * scale[2], 0,
        translation[0], translation[1], translation[2], 1,
    ];
}

/** Rays from random points inside the box (grown by `margin`) in random directions. */
export function randomRays(
    random: Random,
    count: number,
    bounds: readonly [number, number, number, number, number, number],
    margin: number,
    options: { randomInterval?: boolean } = {},
): TraceRay[] {
    const rays: TraceRay[] = [];
    for (let i = 0; i < count; i++) {
        const origin = [0, 1, 2].map((axis) => bounds[axis] - margin + random() * (bounds[axis + 3] - bounds[axis] + 2 * margin));
        const direction = randomUnitVector(random);
        const tMin = options.randomInterval ? random() * 2 : 0;
        const tMax = options.randomInterval ? tMin + random() * 30 : 1e6;
        rays.push(toFloat32Ray(origin, direction, tMin, tMax));
    }
    return rays;
}
