/**
 * A dark room for fast changes (examples, C4): one closed room with no sky,
 * a ceiling lamp, a red box to drag about and a flashlight. The manifest
 * (`scenes/manifests/dark-room.json`) lists the lamp and the flashlight; the
 * C4 page switches the lamp, moves the flashlight with its camera and drags
 * the box. With both lights off the room is black, so any light left on
 * screen after a switch is the cache's history.
 *
 * The lights are here too, as in the manifest (the scene tests compare
 * them): the lamp 0.3 m under its shade, and the flashlight where the first
 * camera bookmark stands, pointing where it looks.
 *
 * Layout (metres). The interior is 5 wide (x), 2.7 high, 5 deep (z). The
 * red box (`drag-box`, 0.6 m) stands on the floor in the middle; a pale
 * cabinet stands against the west wall and a low blue bench against the
 * south wall (z = 0), so the box's bounce and shadow land on something.
 */

import type { ProceduralScene } from '../scene-data';
import { SceneBuilder } from './geometry';
import { addRoomShell } from './room-shell';

const DEGREES_TO_RADIANS = Math.PI / 180;

export const DARK_ROOM_SIZE_M: readonly [number, number, number] = [5, 2.7, 5];
/** The box the page drags: its name, edge length and where it starts (its centre). */
export const DRAG_BOX_NAME = 'drag-box';
export const DRAG_BOX_SIZE_M = 0.6;
export const DRAG_BOX_START: readonly [number, number, number] = [2.9, DRAG_BOX_SIZE_M / 2, 2.0];

export function buildDarkRoom(): ProceduralScene {
    const builder = new SceneBuilder();
    const wall = builder.addDiffuse('wall-paint', [0.72, 0.7, 0.66]);
    const floor = builder.addDiffuse('floor', [0.42, 0.4, 0.38]);
    const ceiling = builder.addDiffuse('ceiling', [0.8, 0.8, 0.8]);
    const red = builder.addDiffuse('box-red', [0.75, 0.1, 0.07]);
    const cabinet = builder.addDiffuse('cabinet-pale', [0.78, 0.76, 0.7]);
    const bench = builder.addDiffuse('bench-blue', [0.12, 0.22, 0.6]);
    const shade = builder.addDiffuse('lamp-shade', [0.3, 0.3, 0.3]);

    const [x, y, z] = DARK_ROOM_SIZE_M;
    addRoomShell(builder, 'room', [0, 0, 0], [x, y, z], 0.2, { wall, floor, ceiling });

    builder.addBoxCentered(DRAG_BOX_NAME, red, DRAG_BOX_START, [DRAG_BOX_SIZE_M, DRAG_BOX_SIZE_M, DRAG_BOX_SIZE_M]);
    builder.addBox('cabinet', cabinet, [0, 0, 0.6], [0.5, 1.8, 1.8]);
    builder.addBox('bench', bench, [2.0, 0, 0], [4.2, 0.45, 0.5]);
    // The lamp's shade hangs 5 cm under the ceiling, the lamp 0.3 m below it.
    builder.addBox('lamp-shade', shade, [2.3, y - 0.05, 2.3], [2.7, y, 2.7]);
    builder.addLight({ type: 'point', label: 'ceiling-lamp', color: [1, 0.94, 0.84], intensity: 10, position: [2.5, 2.35, 2.5], range: 12 });
    builder.addLight({
        type: 'spot',
        label: 'flashlight',
        color: [1, 0.97, 0.92],
        intensity: 40,
        position: [1.3, 1.5, 4.2],
        direction: [1.2, -0.7, -1.7],
        range: 12,
        innerConeAngle: 14 * DEGREES_TO_RADIANS,
        outerConeAngle: 22 * DEGREES_TO_RADIANS,
    });

    return builder.build();
}
