/**
 * Plain-data math types used across the contract.
 *
 * The platform works in metres, right-handed, Y-up, with linear-light colors.
 * Adapters convert from their engine's conventions before calling in.
 */

/** A 3-component vector as a tuple: `[x, y, z]`. */
export type Vec3 = readonly [number, number, number];

/** A linear-light RGB color, not sRGB-encoded. Components are usually 0..1. */
export type ColorRGB = readonly [number, number, number];

/**
 * A 4×4 affine transform, column-major (the same order as three.js
 * `Matrix4.elements` and WGSL `mat4x4f`). Mirrored or non-uniform scales are
 * allowed; the platform derives normal matrices itself.
 */
export type Mat4 = ArrayLike<number>;

/** An axis-aligned box in world space. */
export interface Aabb {
    readonly min: Vec3;
    readonly max: Vec3;
}
