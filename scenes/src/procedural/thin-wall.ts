/**
 * A single thin wall between a lit side and a dark side.
 *
 * One room, 8 m by 2.8 m by 4 m, split down the middle by a wall that is only
 * 3 cm thick and has no doorway. It runs 1 cm into the floor, ceiling and side
 * walls so there is no flush seam to leak through. Only the west side has a light, so the east
 * side must stay dark. Any light there is a leak through the wall. The wall is
 * far thinner than any sensible probe spacing, which is the point.
 *
 * Change `THIN_WALL_THICKNESS_M` to make the test easier or harder.
 */

import type { ProceduralScene } from '../scene-data';
import { SceneBuilder } from './geometry';
import { addRoomShell } from './room-shell';

export const THIN_WALL_THICKNESS_M = 0.03;
/** How far the wall runs into the floor, ceiling and side walls, so it never just touches them. */
export const THIN_WALL_EMBED_M = 0.01;
export const THIN_WALL_ROOM_SIZE_M: readonly [number, number, number] = [8, 2.8, 4];

export function buildThinWall(): ProceduralScene {
    const builder = new SceneBuilder();
    const wall = builder.addDiffuse('outer-wall', [0.72, 0.7, 0.66]);
    const floor = builder.addDiffuse('floor', [0.45, 0.4, 0.35]);
    const ceiling = builder.addDiffuse('ceiling', [0.8, 0.8, 0.8]);
    const partition = builder.addDiffuse('thin-partition', [0.8, 0.8, 0.78]);
    const crateRed = builder.addDiffuse('crate-red', [0.7, 0.12, 0.1]);
    const crateYellow = builder.addDiffuse('crate-yellow', [0.75, 0.6, 0.1]);

    const [sizeX, sizeY, sizeZ] = THIN_WALL_ROOM_SIZE_M;
    addRoomShell(builder, 'room', [0, 0, 0], [sizeX, sizeY, sizeZ], 0.2, { wall, floor, ceiling });

    const middleX = sizeX / 2;
    builder.addBox(
        'thin-wall',
        partition,
        [middleX - THIN_WALL_THICKNESS_M / 2, -THIN_WALL_EMBED_M, -THIN_WALL_EMBED_M],
        [middleX + THIN_WALL_THICKNESS_M / 2, sizeY + THIN_WALL_EMBED_M, sizeZ + THIN_WALL_EMBED_M],
    );

    // A crate on each side. Seeing the east one at all means light got through.
    builder.addBoxCentered('crate-lit-side', crateRed, [2.2, 0.35, 1.2], [0.7, 0.7, 0.7], 20);
    builder.addBoxCentered('crate-dark-side', crateYellow, [6.0, 0.35, 2.4], [0.7, 0.7, 0.7], -20);

    builder.addLight({
        type: 'point',
        label: 'west-lamp',
        color: [1, 0.96, 0.88],
        intensity: 12,
        position: [2, 2.45, 2],
        range: 14,
    });

    return builder.build();
}
