import type { Vector3Tuple } from '../scene-data';
import type { SceneBuilder } from './geometry';

/** Materials for the six surfaces of a room. Pass the same index for all of them for a one-material room. */
export interface RoomMaterials {
    readonly wall: number;
    readonly floor: number;
    readonly ceiling: number;
}

/**
 * Adds a closed box room made of six solid slabs, so every surface is a thick
 * box rather than a single quad. Thick walls keep the outside from showing
 * through at corners, and each slab has a real inside and outside face for
 * back-face tests.
 *
 * `interiorMin` and `interiorMax` are the inside faces of the room. The slabs
 * extend outwards by `thickness`.
 *
 * Slabs overlap where they meet (walls run into the floor and ceiling, and
 * the north and south walls run into the west and east walls) instead of
 * touching face to face. A flush seam can let a ray slip through the crack
 * between two triangles' shared edge; overlapping solids have no crack.
 */
export function addRoomShell(
    builder: SceneBuilder,
    name: string,
    interiorMin: Vector3Tuple,
    interiorMax: Vector3Tuple,
    thickness: number,
    materials: RoomMaterials,
): void {
    const [x0, y0, z0] = interiorMin;
    const [x1, y1, z1] = interiorMax;
    const t = thickness;

    builder.addBox(`${name}-floor`, materials.floor, [x0 - t, y0 - t, z0 - t], [x1 + t, y0, z1 + t]);
    builder.addBox(`${name}-ceiling`, materials.ceiling, [x0 - t, y1, z0 - t], [x1 + t, y1 + t, z1 + t]);
    // Walls reach into the floor and ceiling slabs (y0 - t to y1 + t) and each
    // other's corners (all four span the full outer footprint).
    builder.addBox(`${name}-wall-west`, materials.wall, [x0 - t, y0 - t, z0 - t], [x0, y1 + t, z1 + t]);
    builder.addBox(`${name}-wall-east`, materials.wall, [x1, y0 - t, z0 - t], [x1 + t, y1 + t, z1 + t]);
    builder.addBox(`${name}-wall-south`, materials.wall, [x0 - t, y0 - t, z0 - t], [x1 + t, y1 + t, z0]);
    builder.addBox(`${name}-wall-north`, materials.wall, [x0 - t, y0 - t, z1], [x1 + t, y1 + t, z1 + t]);
}
