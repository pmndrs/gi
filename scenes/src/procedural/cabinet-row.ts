/**
 * A row of hollow kitchen cabinets against a wall, made of 18 mm panels.
 *
 * This is the thin-geometry stress test in a realistic setting: every cabinet
 * is a hollow box of 18 mm boards (bottom, top, two sides, back, door), so
 * there are many surfaces 18 mm apart, enclosed air volumes where probes can
 * land inside, and thin shelves of shadow under the wall units.
 *
 * Layout (metres). The room interior is 4.8 wide (x), 2.6 high, 3.2 deep (z).
 * The back wall is at z = 0, and everything is built against it:
 * - one tall unit at x 0..0.6, 2.2 m high;
 * - four base units, 0.6 wide, x 0.6..3.0, 0.72 m high on a 0.1 m kick space,
 *   with a worktop over them;
 * - four wall units above the worktop, 0.35 deep.
 *
 * Options:
 * - `closedFronts` (default true): doors on every unit, so the insides are
 *   closed hollow volumes. Set false to leave the fronts open.
 */

import type { BuilderOptions, ProceduralScene, Vector3Tuple } from '../scene-data';
import { SceneBuilder } from './geometry';
import { addRoomShell } from './room-shell';

export const PANEL_THICKNESS_M = 0.018;
/** Boards run this far into the boards they meet, like a real door overlaps its frame, so no seam is flush. */
export const PANEL_OVERLAP_M = 0.004;
const WORKTOP_THICKNESS_M = 0.038;
const KICK_SPACE_M = 0.1;
const BASE_UNIT_HEIGHT_M = 0.72;
const BASE_UNIT_DEPTH_M = 0.58;
const WALL_UNIT_DEPTH_M = 0.35;
const WALL_UNIT_BOTTOM_M = 1.45;
const WALL_UNIT_TOP_M = 2.15;
const TALL_UNIT_TOP_M = 2.2;
const UNIT_WIDTH_M = 0.6;

export const CABINET_ROOM_SIZE_M: readonly [number, number, number] = [4.8, 2.6, 3.2];

/**
 * Adds one hollow cabinet from `min` to `max` (the outer box), made of 18 mm
 * boards. The door, when present, covers the whole front.
 */
function addHollowCabinet(
    builder: SceneBuilder,
    name: string,
    material: number,
    min: Vector3Tuple,
    max: Vector3Tuple,
    closedFront: boolean,
): void {
    const p = PANEL_THICKNESS_M;
    const o = PANEL_OVERLAP_M;
    const [x0, y0, z0] = min;
    const [x1, y1, z1] = max;
    // The carcass stops where the door starts, plus a little into the door.
    const carcassFront = closedFront ? z1 - p + o : z1;

    // Bottom and top are full width. The sides, back and front door run a few
    // millimetres into the boards they meet, so the inside faces stay where
    // they were but no two boards just touch.
    builder.addBox(`${name}-bottom`, material, [x0, y0, z0], [x1, y0 + p, carcassFront]);
    builder.addBox(`${name}-top`, material, [x0, y1 - p, z0], [x1, y1, carcassFront]);
    builder.addBox(`${name}-side-left`, material, [x0, y0 + p - o, z0], [x0 + p, y1 - p + o, carcassFront]);
    builder.addBox(`${name}-side-right`, material, [x1 - p, y0 + p - o, z0], [x1, y1 - p + o, carcassFront]);
    builder.addBox(`${name}-back`, material, [x0 + p - o, y0 + p - o, z0], [x1 - p + o, y1 - p + o, z0 + p]);
    if (closedFront) {
        builder.addBox(`${name}-door`, material, [x0, y0, z1 - p], [x1, y1, z1]);
    }
}

export function buildCabinetRow(options: BuilderOptions = {}): ProceduralScene {
    const closedFronts = options.closedFronts ?? true;
    if (typeof closedFronts !== 'boolean') {
        throw new Error(`cabinet-row: option "closedFronts" must be a boolean, got ${JSON.stringify(closedFronts)}`);
    }

    const builder = new SceneBuilder();
    const wall = builder.addDiffuse('wall-paint', [0.75, 0.74, 0.7]);
    const floor = builder.addDiffuse('floor-boards', [0.5, 0.38, 0.27]);
    const ceiling = builder.addDiffuse('ceiling', [0.8, 0.8, 0.8]);
    const cabinet = builder.addDiffuse('cabinet-white', [0.78, 0.78, 0.75]);
    const worktop = builder.addDiffuse('worktop-dark', [0.12, 0.12, 0.13]);
    const kickboard = builder.addDiffuse('kickboard', [0.2, 0.2, 0.2]);

    const [roomX, roomY, roomZ] = CABINET_ROOM_SIZE_M;
    addRoomShell(builder, 'room', [0, 0, 0], [roomX, roomY, roomZ], 0.15, { wall, floor, ceiling });

    // The tall unit stands on the floor and is the left end of the run.
    addHollowCabinet(
        builder,
        'tall-unit',
        cabinet,
        [0, 0, 0],
        [UNIT_WIDTH_M, TALL_UNIT_TOP_M, BASE_UNIT_DEPTH_M],
        closedFronts,
    );

    // Base units sit on a kick space; a kickboard closes the gap at the front.
    const baseFirstX = UNIT_WIDTH_M;
    const baseUnitCount = 4;
    const baseLastX = baseFirstX + baseUnitCount * UNIT_WIDTH_M;
    const baseTop = KICK_SPACE_M + BASE_UNIT_HEIGHT_M;
    for (let i = 0; i < baseUnitCount; i++) {
        const x0 = baseFirstX + i * UNIT_WIDTH_M;
        addHollowCabinet(
            builder,
            `base-unit-${i}`,
            cabinet,
            [x0, KICK_SPACE_M, 0],
            [x0 + UNIT_WIDTH_M, baseTop, BASE_UNIT_DEPTH_M],
            closedFronts,
        );
    }
    builder.addBox(
        'kickboard',
        kickboard,
        [baseFirstX, 0, BASE_UNIT_DEPTH_M - 0.08 - PANEL_THICKNESS_M],
        [baseLastX, KICK_SPACE_M, BASE_UNIT_DEPTH_M - 0.08],
    );
    builder.addBox(
        'worktop',
        worktop,
        [baseFirstX, baseTop, 0],
        [baseLastX, baseTop + WORKTOP_THICKNESS_M, BASE_UNIT_DEPTH_M + 0.03],
    );

    for (let i = 0; i < baseUnitCount; i++) {
        const x0 = baseFirstX + i * UNIT_WIDTH_M;
        addHollowCabinet(
            builder,
            `wall-unit-${i}`,
            cabinet,
            [x0, WALL_UNIT_BOTTOM_M, 0],
            [x0 + UNIT_WIDTH_M, WALL_UNIT_TOP_M, WALL_UNIT_DEPTH_M],
            closedFronts,
        );
    }

    builder.addLight({
        type: 'point',
        label: 'ceiling-lamp',
        color: [1, 0.96, 0.88],
        intensity: 12,
        position: [2.4, 2.35, 1.7],
        range: 12,
    });

    return builder.build();
}
