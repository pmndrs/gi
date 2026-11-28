/**
 * Octahedral direction maps (Cigolle et al. 2014, "A Survey of Efficient
 * Representations for Independent Unit Vectors", JCGT 3(2)), on the CPU.
 *
 * The WGSL in `wgsl/sample.ts` and `wgsl/write.ts` does the same; these
 * mirror it so the mapping and the border rule are unit-tested without a GPU.
 *
 * The map folds the sphere onto the square [-1, 1]²: the upper hemisphere
 * (z ≥ 0) is the inner diamond, the lower one the four corner triangles.
 * Each edge of the square is mirrored about its midpoint (the point at u on
 * the top edge is the same direction as -u), and all four corners are the
 * same direction (-z). So a tile's 1-texel border copies the interior texel
 * mirrored across the edge, and a border corner copies the diagonally
 * opposite interior corner. Bilinear filtering across a tile's edge then
 * blends the texels that really are neighbours on the sphere.
 */

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

function signNotZero(value: number): number {
    return value >= 0 ? 1 : -1;
}

/** Unit direction → point in [-1, 1]². */
export function octEncode(direction: readonly [number, number, number]): Vec2 {
    const [x, y, z] = direction;
    const sum = Math.abs(x) + Math.abs(y) + Math.abs(z);
    const u = x / sum;
    const v = y / sum;
    if (z >= 0) return [u, v];
    return [(1 - Math.abs(v)) * signNotZero(u), (1 - Math.abs(u)) * signNotZero(v)];
}

/** Point in [-1, 1]² → unit direction. */
export function octDecode([u, v]: readonly [number, number]): Vec3 {
    let x = u;
    let y = v;
    const z = 1 - Math.abs(u) - Math.abs(v);
    if (z < 0) {
        x = (1 - Math.abs(v)) * signNotZero(u);
        y = (1 - Math.abs(u)) * signNotZero(v);
    }
    const length = Math.hypot(x, y, z);
    return [x / length, y / length, z / length];
}

/** Direction at the centre of interior texel (column, row) of a tile with `interior` texels per side. */
export function texelDirection(column: number, row: number, interior: number): Vec3 {
    return octDecode([((column + 0.5) / interior) * 2 - 1, ((row + 0.5) / interior) * 2 - 1]);
}

/**
 * The interior texel a tile texel copies. Tile texels run 0..interior+1,
 * with the border at 0 and interior+1; interior texel i is tile texel i + 1.
 */
export function borderSource(tileColumn: number, tileRow: number, interior: number): Vec2 {
    const last = interior - 1;
    const column = tileColumn - 1;
    const row = tileRow - 1;
    const borderColumn = column < 0 || column > last;
    const borderRow = row < 0 || row > last;
    const clampedColumn = Math.min(last, Math.max(0, column));
    const clampedRow = Math.min(last, Math.max(0, row));
    // A corner copies the diagonally opposite interior corner.
    if (borderColumn && borderRow) return [column < 0 ? last : 0, row < 0 ? last : 0];
    // Top and bottom rows mirror across the tile's vertical centre line.
    if (borderRow) return [last - clampedColumn, clampedRow];
    // Left and right columns mirror across the horizontal centre line.
    if (borderColumn) return [clampedColumn, last - clampedRow];
    return [column, row];
}
