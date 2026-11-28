/**
 * The per-frame random rotation of the blended rays (docs/design/ddgi.md,
 * "Rays"). One rotation per frame, shared by every probe updated that frame,
 * so successive updates of a probe sample different directions and the
 * hysteresis blend averages them into a smooth estimate.
 *
 * The rotation is a uniform random quaternion (Shoemake 1992, "Uniform
 * random rotations", Graphics Gems III) from a small seeded generator,
 * seeded by the frame number. That makes a run repeatable on one machine for
 * debugging, but nothing may require byte-identical results: the GPU's
 * rounding differs between devices.
 */

/** Mulberry32: a tiny 32-bit generator with a good spread, enough for picking rotations. */
export function seededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let mixed = state;
        mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
        mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}

/** A unit quaternion `[x, y, z, w]` drawn uniformly from all rotations. */
export function uniformRandomQuaternion(random: () => number): [number, number, number, number] {
    const u1 = random();
    const u2 = random();
    const u3 = random();
    const a = Math.sqrt(1 - u1);
    const b = Math.sqrt(u1);
    return [a * Math.sin(2 * Math.PI * u2), a * Math.cos(2 * Math.PI * u2), b * Math.sin(2 * Math.PI * u3), b * Math.cos(2 * Math.PI * u3)];
}

/** The rotation matrix of a unit quaternion, column-major, 9 numbers. */
export function quaternionToMatrix([x, y, z, w]: readonly [number, number, number, number]): number[] {
    return [
        1 - 2 * (y * y + z * z),
        2 * (x * y + z * w),
        2 * (x * z - y * w),
        2 * (x * y - z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z + x * w),
        2 * (x * z + y * w),
        2 * (y * z - x * w),
        1 - 2 * (x * x + y * y),
    ];
}

/** The rotation for a frame, column-major 3×3. The frame number is mixed so neighbouring frames get unrelated seeds. */
export function frameRayRotation(frame: number): number[] {
    const seed = Math.imul(frame >>> 0, 0x9e3779b1) ^ 0x5eed_dd61;
    return quaternionToMatrix(uniformRandomQuaternion(seededRandom(seed)));
}
