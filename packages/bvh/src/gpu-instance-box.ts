/**
 * WGSL for an instance's world-space box: the box of a mesh's object-space
 * box after an affine object-to-world transform, padded so f32 rounding can
 * never make it too small. The top-level refit's per-instance pass calls it
 * once per instance per frame.
 *
 * **Arvo's method** (Graphics Gems, 1990): for a 3x4 row `r`, the world
 * centre is `dot(r.xyz, centre) + r.w` and the half-extent is
 * `dot(abs(r.xyz), halfExtent)`. That is exactly the box around the eight
 * transformed corners, at the cost of two dot products per axis instead of
 * eight corner transforms. It holds for rotations, shears, non-uniform scales
 * and mirrors alike, because `abs` takes care of the sign.
 *
 * **Padding.** In f32, each of those sums can be off by a few units in the
 * last place of the largest term it adds, which with cancellation (a large
 * translation, say) can exceed the result itself. So each axis is grown by
 * `BVH_INSTANCE_BOX_PADDING` times the summed magnitude of every term,
 * `dot(abs(r.xyz), max(abs(lower), abs(upper))) + abs(r.w)`: 2^-20, sixteen
 * f32 steps, comfortably more than the handful of roundings involved. The CPU
 * top level pads by 2^-21 of the result for the same reason (the tracer's
 * `transformBounds`); an instance box only decides which instances a ray
 * enters, so a slightly loose one costs nothing measurable.
 */

/** Relative padding of an instance's world box, as a power of two. */
export const BVH_INSTANCE_BOX_PADDING = 2 ** -20;

/**
 * Declares `BvhWorldBox` and `bvhInstanceWorldBox(row0, row1, row2, lower, upper)`:
 * rows of the object-to-world 3x4 matrix (`xyz` linear part, `w` translation)
 * and the object-space box.
 */
export const BVH_INSTANCE_BOX_WGSL = /* wgsl */ `
struct BvhWorldBox {
    lower: vec3f,
    upper: vec3f,
}

// Relative padding: 2^-20 of each axis's summed term magnitude (see gpu-instance-box.ts).
const BVH_INSTANCE_BOX_PADDING: f32 = ${BVH_INSTANCE_BOX_PADDING};

fn bvhInstanceWorldBox(row0: vec4f, row1: vec4f, row2: vec4f, objectLower: vec3f, objectUpper: vec3f) -> BvhWorldBox {
    let center = (objectLower + objectUpper) * 0.5;
    let halfExtent = (objectUpper - objectLower) * 0.5;
    let reach = max(abs(objectLower), abs(objectUpper));
    let worldCenter = vec3f(
        dot(row0.xyz, center) + row0.w,
        dot(row1.xyz, center) + row1.w,
        dot(row2.xyz, center) + row2.w,
    );
    let worldHalfExtent = vec3f(
        dot(abs(row0.xyz), halfExtent),
        dot(abs(row1.xyz), halfExtent),
        dot(abs(row2.xyz), halfExtent),
    );
    let magnitude = vec3f(
        dot(abs(row0.xyz), reach) + abs(row0.w),
        dot(abs(row1.xyz), reach) + abs(row1.w),
        dot(abs(row2.xyz), reach) + abs(row2.w),
    );
    let padding = magnitude * BVH_INSTANCE_BOX_PADDING;
    return BvhWorldBox(worldCenter - worldHalfExtent - padding, worldCenter + worldHalfExtent + padding);
}
`;
