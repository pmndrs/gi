/**
 * Octahedral encoding of unit vectors into one 32-bit word (two snorm16).
 *
 * The unit sphere is projected onto the octahedron |x| + |y| + |z| = 1, and
 * the lower half (z < 0) is folded over the diagonals onto the square
 * |x| + |y| <= 1, so every direction maps to a point of [-1, 1]². That point
 * is stored as two signed 16-bit values. Cigolle et al., "A Survey of
 * Efficient Representations for Independent Unit Vectors" (JCGT 2014).
 *
 * The WGSL decoder in `wgsl.ts` (`giSceneDecodeNormal`) must stay the exact
 * inverse of {@link encodeOctahedral}; the GPU readback page checks that.
 */

const SNORM16_MAX = 32767;

/** +1 for zero and positive values, so the fold never collapses an axis to 0. */
function signNotZero(value: number): number {
    return value >= 0 ? 1 : -1;
}

function toSnorm16(value: number): number {
    const clamped = Math.min(1, Math.max(-1, value));
    return Math.round(clamped * SNORM16_MAX) & 0xffff;
}

function fromSnorm16(bits: number): number {
    const signed = bits >= 0x8000 ? bits - 0x10000 : bits;
    // WGSL's unpack2x16snorm clamps -32768 to -1 the same way.
    return Math.max(-1, signed / SNORM16_MAX);
}

/**
 * Encodes a direction (need not be unit length) as a word. A zero vector
 * encodes as +Z, so a bad input normal still decodes to a unit vector.
 * The first component goes in the low 16 bits, as `pack2x16snorm` does.
 */
export function encodeOctahedral(x: number, y: number, z: number): number {
    const sum = Math.abs(x) + Math.abs(y) + Math.abs(z);
    if (!(sum > 0) || !Number.isFinite(sum)) return toSnorm16(0) | (toSnorm16(0) << 16);
    let u = x / sum;
    let v = y / sum;
    if (z < 0) {
        // Fold the lower hemisphere: reflect across the diagonals of the square.
        const foldedU = (1 - Math.abs(v)) * signNotZero(u);
        const foldedV = (1 - Math.abs(u)) * signNotZero(v);
        u = foldedU;
        v = foldedV;
    }
    return (toSnorm16(u) | (toSnorm16(v) << 16)) >>> 0;
}

/** The inverse of {@link encodeOctahedral}: a unit vector. */
export function decodeOctahedral(word: number): [number, number, number] {
    const u = fromSnorm16(word & 0xffff);
    const v = fromSnorm16(word >>> 16);
    let x = u;
    let y = v;
    const z = 1 - Math.abs(u) - Math.abs(v);
    // Unfold: points outside the diamond |u| + |v| <= 1 came from z < 0.
    const fold = Math.max(-z, 0);
    x += x >= 0 ? -fold : fold;
    y += y >= 0 ? -fold : fold;
    const length = Math.hypot(x, y, z);
    return [x / length, y / length, z / length];
}
