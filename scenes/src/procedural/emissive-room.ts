/**
 * A living room lit only by its emissive surfaces: two light strips under the
 * ceiling and a screen on the far wall. There are no light objects at all,
 * so every bit of light on the walls, floor and furniture is light that GI's
 * rays found on an emitter. The C5 example ("Light from surfaces") changes
 * the strips' colour and brightness and puts a picture on the screen.
 *
 * Layout (metres; y up, the screen on the north wall at z = 5):
 * - the room is 6 m wide (x), 2.8 m high and 5 m deep (z), with 15 cm slabs;
 * - the strips run front to back under the ceiling, 25 cm from the west and
 *   east walls, each 20 cm wide and 4.4 m long, glowing down;
 * - the screen is 1.9 m by 1.05 m, its bottom 0.85 m above the floor, on a
 *   dark body against the north wall, over a low cabinet;
 * - a sofa faces it, with a low table between them, a tall shelf on the east
 *   wall and a plant stand by the screen, so the emitters cast shadows and
 *   the furniture's colour bleeds onto the floor.
 *
 * The strips and the screen are thin and bright on purpose: an emitter
 * reaches GI only through the rays that hit it, so the smaller and brighter
 * it is, the noisier its light (docs/api/platform.md, "Emissive surfaces
 * aren't light sources to hit shading").
 */

import type { LinearColor, ProceduralScene } from '../scene-data';
import { SceneBuilder } from './geometry';
import { addRoomShell } from './room-shell';

/** The room's inside: x, y, z. */
export const EMISSIVE_ROOM_SIZE_M: readonly [number, number, number] = [6, 2.8, 5];
const SLAB_M = 0.15;

/** The strips' emitted radiance: a warm white. */
export const EMISSIVE_ROOM_STRIP_RADIANCE: LinearColor = [9, 7.6, 5.6];
/** The screen's emitted radiance with nothing on it: a plain white glow. */
export const EMISSIVE_ROOM_SCREEN_RADIANCE: LinearColor = [3, 3, 3];

/** Material names a page can look up to change the emitters (C5). */
export const EMISSIVE_ROOM_MATERIALS = { strip: 'strip-light', screen: 'screen' } as const;

const STRIP_WIDTH_M = 0.2;
const STRIP_DEPTH_M = 0.04;
const STRIP_INSET_M = 0.25;
const STRIP_Z0_M = 0.3;
const STRIP_Z1_M = 4.7;
/** How far an attached part reaches into what holds it, so no two faces are flush (room-shell.ts). */
const OVERLAP_M = 0.01;

export function buildEmissiveRoom(): ProceduralScene {
    const [sizeX, sizeY, sizeZ] = EMISSIVE_ROOM_SIZE_M;
    const builder = new SceneBuilder();
    const wall = builder.addDiffuse('wall', [0.7, 0.7, 0.68]);
    const floor = builder.addDiffuse('floor', [0.45, 0.33, 0.22]);
    const ceiling = builder.addDiffuse('ceiling', [0.75, 0.75, 0.75]);
    const sofa = builder.addDiffuse('sofa', [0.18, 0.28, 0.5]);
    const wood = builder.addDiffuse('wood', [0.55, 0.42, 0.3]);
    const plant = builder.addDiffuse('plant-stand', [0.25, 0.45, 0.2]);
    const body = builder.addDiffuse('screen-body', [0.04, 0.04, 0.04]);
    const strip = builder.addMaterial({ name: EMISSIVE_ROOM_MATERIALS.strip, albedo: [0.8, 0.8, 0.8], emissive: EMISSIVE_ROOM_STRIP_RADIANCE });
    const screen = builder.addMaterial({ name: EMISSIVE_ROOM_MATERIALS.screen, albedo: [0.04, 0.04, 0.04], emissive: EMISSIVE_ROOM_SCREEN_RADIANCE });

    addRoomShell(builder, 'room', [0, 0, 0], [sizeX, sizeY, sizeZ], SLAB_M, { wall, floor, ceiling });

    // The strips hang from the ceiling, reaching into its slab.
    for (const [side, x0] of [
        ['west', STRIP_INSET_M],
        ['east', sizeX - STRIP_INSET_M - STRIP_WIDTH_M],
    ] as const) {
        builder.addBox(`strip-${side}`, strip, [x0, sizeY - STRIP_DEPTH_M, STRIP_Z0_M], [x0 + STRIP_WIDTH_M, sizeY + OVERLAP_M, STRIP_Z1_M]);
    }

    // The screen: a dark body reaching into the north wall, the glowing panel set into its front.
    builder.addBox('screen-body', body, [1.95, 0.8, sizeZ - 0.12], [4.05, 1.95, sizeZ + OVERLAP_M]);
    builder.addBox('screen-panel', screen, [2.05, 0.85, sizeZ - 0.14], [3.95, 1.9, sizeZ - 0.1]);
    builder.addBox('cabinet', wood, [1.6, 0, sizeZ - 0.5], [4.4, 0.5, sizeZ + OVERLAP_M]);

    // The sofa, facing the screen.
    builder.addBox('sofa-seat', sofa, [1.8, 0, 1.2], [4.2, 0.45, 2.1]);
    builder.addBox('sofa-back', sofa, [1.8, 0, 0.95], [4.2, 0.9, 1.25]);
    builder.addBox('sofa-arm-west', sofa, [1.6, 0, 0.95], [1.85, 0.65, 2.1]);
    builder.addBox('sofa-arm-east', sofa, [4.15, 0, 0.95], [4.4, 0.65, 2.1]);

    builder.addBox('table', wood, [2.5, 0, 2.6], [3.5, 0.4, 3.3]);
    builder.addBox('shelf', wood, [sizeX - 0.45, 0, 1.0], [sizeX + OVERLAP_M, 2.0, 2.2]);
    builder.addBox('plant-stand', plant, [0.6, 0, 3.9], [0.95, 1.2, 4.25]);

    return builder.build();
}
