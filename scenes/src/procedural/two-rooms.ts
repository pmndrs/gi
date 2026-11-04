/**
 * Two rooms joined by a doorway: the classic light-leak test.
 *
 * The same shell serves two scenes:
 * - `two-room`: the doorway is open.
 * - `closed-door`: a door panel closes the doorway.
 *
 * Room A (x 0..4 m) has a lamp and is the lit room. Room B (x 4.2..8.2 m) has
 * no light at all, so any light there arrives through the doorway or leaks
 * through a wall. The partition between them is 0.2 m thick, which is thinner
 * than a typical probe spacing on purpose.
 *
 * Dimensions: each room is 4 m by 4 m and 2.8 m high; the doorway is 0.9 m
 * wide and 2.1 m high, centred on the partition.
 */

import type { ProceduralScene } from '../scene-data';
import { SceneBuilder } from './geometry';

export const ROOM_SIZE_M = 4;
export const ROOM_HEIGHT_M = 2.8;
export const PARTITION_THICKNESS_M = 0.2;
export const DOORWAY_WIDTH_M = 0.9;
export const DOORWAY_HEIGHT_M = 2.1;
const OUTER_WALL_THICKNESS_M = 0.2;

const DOORWAY_CENTRE_Z_M = ROOM_SIZE_M / 2;
const PARTITION_X0_M = ROOM_SIZE_M;
const PARTITION_X1_M = ROOM_SIZE_M + PARTITION_THICKNESS_M;
const ROOM_B_X1_M = PARTITION_X1_M + ROOM_SIZE_M;

/** A real door slab is a little larger than the opening it closes, so it overlaps the frame and leaves no gap. */
const DOOR_OVERLAP_M = 0.004;
const DOOR_THICKNESS_M = 0.04;

function buildRooms(withDoor: boolean): ProceduralScene {
    const builder = new SceneBuilder();
    const wallA = builder.addDiffuse('wall-room-a-warm', [0.75, 0.7, 0.62]);
    const wallB = builder.addDiffuse('wall-room-b-cool', [0.6, 0.68, 0.75]);
    const partition = builder.addDiffuse('partition', [0.8, 0.8, 0.78]);
    const floor = builder.addDiffuse('floor', [0.45, 0.4, 0.35]);
    const ceiling = builder.addDiffuse('ceiling', [0.8, 0.8, 0.8]);
    const door = builder.addDiffuse('door-wood', [0.45, 0.3, 0.18]);
    const table = builder.addDiffuse('table-wood', [0.55, 0.4, 0.25]);
    const crateRed = builder.addDiffuse('crate-red', [0.7, 0.12, 0.1]);
    const crateBlue = builder.addDiffuse('crate-blue', [0.1, 0.2, 0.7]);

    // The outer shell around both rooms. Each long wall is split at the
    // partition so room A's walls and room B's walls can differ in colour.
    // Solids overlap where they meet (walls run into the floor and ceiling, the
    // two halves of a long wall overlap inside the partition) so no seam is
    // flush: a ray cannot slip through a crack between touching faces.
    const t = OUTER_WALL_THICKNESS_M;
    const h = ROOM_HEIGHT_M;
    const d = ROOM_SIZE_M;
    const seamX = (PARTITION_X0_M + PARTITION_X1_M) / 2;
    builder.addBox('floor', floor, [-t, -t, -t], [ROOM_B_X1_M + t, 0, d + t]);
    builder.addBox('ceiling', ceiling, [-t, h, -t], [ROOM_B_X1_M + t, h + t, d + t]);
    builder.addBox('wall-west', wallA, [-t, -t, -t], [0, h + t, d + t]);
    builder.addBox('wall-east', wallB, [ROOM_B_X1_M, -t, -t], [ROOM_B_X1_M + t, h + t, d + t]);
    builder.addBox('wall-south-room-a', wallA, [-t, -t, -t], [seamX + DOOR_OVERLAP_M, h + t, 0]);
    builder.addBox('wall-south-room-b', wallB, [seamX - DOOR_OVERLAP_M, -t, -t], [ROOM_B_X1_M + t, h + t, 0]);
    builder.addBox('wall-north-room-a', wallA, [-t, -t, d], [seamX + DOOR_OVERLAP_M, h + t, d + t]);
    builder.addBox('wall-north-room-b', wallB, [seamX - DOOR_OVERLAP_M, -t, d], [ROOM_B_X1_M + t, h + t, d + t]);

    // The partition: two pillars either side of the doorway plus a lintel above
    // it. They run 1 cm into the shell, and the lintel overlaps the pillars.
    const embed = 0.01;
    const doorwayZ0 = DOORWAY_CENTRE_Z_M - DOORWAY_WIDTH_M / 2;
    const doorwayZ1 = DOORWAY_CENTRE_Z_M + DOORWAY_WIDTH_M / 2;
    builder.addBox('partition-south', partition, [PARTITION_X0_M, -embed, -embed], [PARTITION_X1_M, h + embed, doorwayZ0]);
    builder.addBox('partition-north', partition, [PARTITION_X0_M, -embed, doorwayZ1], [PARTITION_X1_M, h + embed, d + embed]);
    builder.addBox(
        'partition-lintel',
        partition,
        [PARTITION_X0_M, DOORWAY_HEIGHT_M, doorwayZ0 - embed],
        [PARTITION_X1_M, h + embed, doorwayZ1 + embed],
    );

    if (withDoor) {
        const middleX = (PARTITION_X0_M + PARTITION_X1_M) / 2;
        builder.addBox(
            'door-panel',
            door,
            [middleX - DOOR_THICKNESS_M / 2, -DOOR_OVERLAP_M, doorwayZ0 - DOOR_OVERLAP_M],
            [middleX + DOOR_THICKNESS_M / 2, DOORWAY_HEIGHT_M + DOOR_OVERLAP_M, doorwayZ1 + DOOR_OVERLAP_M],
        );
    }

    // Furniture, so there is something to see and to cast shadows.
    builder.addBox('table-top', table, [0.8, 0.72, 0.6], [2.0, 0.76, 1.4]);
    for (const [index, [x, z]] of ([[0.85, 0.65], [1.95, 0.65], [0.85, 1.35], [1.95, 1.35]] as const).entries()) {
        builder.addBox(`table-leg-${index}`, table, [x - 0.03, 0, z - 0.03], [x + 0.03, 0.72, z + 0.03]);
    }
    builder.addBoxCentered('crate-red', crateRed, [3.0, 0.3, 3.0], [0.6, 0.6, 0.6], 25);
    builder.addBoxCentered('crate-blue-in-room-b', crateBlue, [6.4, 0.4, 2.6], [0.8, 0.8, 0.8], -15);

    builder.addLight({
        type: 'point',
        label: 'room-a-lamp',
        color: [1, 0.96, 0.88],
        intensity: 12,
        position: [2, 2.45, 2],
        range: 14,
    });

    return builder.build();
}

/** Two rooms, doorway open, only room A lit. */
export function buildTwoRoom(): ProceduralScene {
    return buildRooms(false);
}

/** The same two rooms with a door panel in the doorway. */
export function buildClosedDoor(): ProceduralScene {
    return buildRooms(true);
}
