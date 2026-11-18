/**
 * Two rooms sharing a wall that light partly passes through (M4, T4.4): a lit
 * room A (x 0..4 m) with a lamp and a dark room B (x 4..8 m) with none, so
 * whatever light reaches room B came through the shared wall.
 *
 * - `alpha-checker`: the shared wall is a cut-out checkerboard (alpha test
 *   0.5), exactly half of it open. Room B's light comes through the holes.
 * - `glass-wall`: the shared wall is one mesh with two materials, an opaque
 *   wall with a window and a clear glass pane in the window (transmission 1).
 *   Room B's light comes through the glass only.
 *
 * Both rooms are 4 m by 4 m and 2.8 m high, inside one closed shell of 0.2 m
 * slabs that overlap at their seams (room-shell.ts). The shared wall runs
 * 0.1 m into the shell on every side, so nothing gets round its edges.
 */

import type { ProceduralScene } from '../scene-data';
import { createGroupedBoxMesh, SceneBuilder, type BoxExtent } from './geometry';
import { addRoomShell } from './room-shell';
import { createAlphaChecker } from './textures';

export const SEE_THROUGH_ROOM_SIZE_M = 4;
export const SEE_THROUGH_ROOM_HEIGHT_M = 2.8;
/** The shared wall's plane (the alpha checker) or middle (the glass wall). */
export const SHARED_WALL_X_M = SEE_THROUGH_ROOM_SIZE_M;
const SHELL_THICKNESS_M = 0.2;
/** How far the shared wall runs into the shell's slabs. */
const SHARED_WALL_EMBED_M = 0.1;

/** The alpha checker's cutoff: alpha below it is a hole. */
export const ALPHA_CHECKER_CUTOFF = 0.5;

/** The glass wall's slab thickness and its window, in the wall's z and y. */
export const GLASS_WALL_THICKNESS_M = 0.1;
export const GLASS_WINDOW_Z_M: readonly [number, number] = [0.8, 3.2];
export const GLASS_WINDOW_Y_M: readonly [number, number] = [0.6, 2.2];
const GLASS_PANE_THICKNESS_M = 0.01;
/** The pane and the window's side boxes run this far into the boxes around them, so no seam is flush. */
const GLASS_OVERLAP_M = 0.01;

/** The shell, the furniture and room A's lamp; the shared wall is added by the caller. */
function addRooms(builder: SceneBuilder): void {
    const wall = builder.addDiffuse('wall-plaster', [0.72, 0.7, 0.66]);
    const floor = builder.addDiffuse('floor', [0.45, 0.4, 0.35]);
    const ceiling = builder.addDiffuse('ceiling', [0.8, 0.8, 0.8]);
    const crateRed = builder.addDiffuse('crate-red', [0.7, 0.12, 0.1]);
    const crateBlue = builder.addDiffuse('crate-blue', [0.1, 0.2, 0.7]);

    const size = SEE_THROUGH_ROOM_SIZE_M;
    addRoomShell(builder, 'rooms', [0, 0, 0], [2 * size, SEE_THROUGH_ROOM_HEIGHT_M, size], SHELL_THICKNESS_M, { wall, floor, ceiling });
    builder.addBoxCentered('crate-red-room-a', crateRed, [1.2, 0.3, 3.0], [0.6, 0.6, 0.6], 25);
    builder.addBoxCentered('crate-blue-room-b', crateBlue, [6.2, 0.4, 2.4], [0.8, 0.8, 0.8], -15);

    builder.addLight({
        type: 'point',
        label: 'room-a-lamp',
        color: [1, 0.96, 0.88],
        intensity: 12,
        position: [2, 2.45, 2],
        range: 0,
    });
}

/**
 * Two rooms whose shared wall is a checkerboard of holes: a double-sided quad
 * at x = 4 whose albedo texture's alpha is 1 on half the squares and 0 on the
 * other half (`alphaCutoff` 0.5).
 *
 * The quad is 4.2 m by 3.0 m (it runs into the shell), and its uv mapping
 * stretches the texture so that exactly the room's 4 m by 2.8 m opening holds
 * the 8 × 8 squares: 50% of the opening is open, not just 50% of the quad.
 */
export function buildAlphaChecker(): ProceduralScene {
    const builder = new SceneBuilder();
    addRooms(builder);

    const width = SEE_THROUGH_ROOM_SIZE_M + 2 * SHARED_WALL_EMBED_M;
    const height = SEE_THROUGH_ROOM_HEIGHT_M + 2 * SHARED_WALL_EMBED_M;
    const checker = builder.addTexture(createAlphaChecker([0.75, 0.75, 0.72]));
    const material = builder.addMaterial({
        name: 'alpha-checker-wall',
        albedo: [1, 1, 1],
        // The quad's uv runs 0..1 over its full 4.2 m by 3.0 m; this maps the opening
        // (from 0.1 m in on each side) onto 0..1: uv' = uv * full / opening - embed / opening.
        albedoMap: {
            texture: checker,
            uv: {
                repeat: [width / SEE_THROUGH_ROOM_SIZE_M, height / SEE_THROUGH_ROOM_HEIGHT_M],
                offset: [-SHARED_WALL_EMBED_M / SEE_THROUGH_ROOM_SIZE_M, -SHARED_WALL_EMBED_M / SEE_THROUGH_ROOM_HEIGHT_M],
            },
        },
        alphaCutoff: ALPHA_CHECKER_CUTOFF,
        doubleSided: true,
    });
    // The unit quad faces +z; this turns its local x onto world +z and its normal onto -x
    // (a proper rotation: +z cross +y = -x), so u runs along z and v up the wall.
    builder.addQuad('alpha-checker-wall', material, [
        0, 0, width, 0,
        0, height, 0, 0,
        -1, 0, 0, 0,
        SHARED_WALL_X_M, SEE_THROUGH_ROOM_HEIGHT_M / 2, SEE_THROUGH_ROOM_SIZE_M / 2, 1,
    ]);
    return builder.build();
}

/** The glass wall's boxes: the wall around the window (slot 0) and the pane in it (slot 1), in world space. */
export function glassWallBoxes(): { wall: BoxExtent[]; pane: BoxExtent[] } {
    const x0 = SHARED_WALL_X_M - GLASS_WALL_THICKNESS_M / 2;
    const x1 = SHARED_WALL_X_M + GLASS_WALL_THICKNESS_M / 2;
    const zMin = -SHARED_WALL_EMBED_M;
    const zMax = SEE_THROUGH_ROOM_SIZE_M + SHARED_WALL_EMBED_M;
    const yMin = -SHARED_WALL_EMBED_M;
    const yMax = SEE_THROUGH_ROOM_HEIGHT_M + SHARED_WALL_EMBED_M;
    const [windowZ0, windowZ1] = GLASS_WINDOW_Z_M;
    const [windowY0, windowY1] = GLASS_WINDOW_Y_M;
    const overlap = GLASS_OVERLAP_M;
    const wall: BoxExtent[] = [
        { min: [x0, yMin, zMin], max: [x1, windowY0, zMax] },
        { min: [x0, windowY1, zMin], max: [x1, yMax, zMax] },
        { min: [x0, windowY0 - overlap, zMin], max: [x1, windowY1 + overlap, windowZ0] },
        { min: [x0, windowY0 - overlap, windowZ1], max: [x1, windowY1 + overlap, zMax] },
    ];
    const paneHalf = GLASS_PANE_THICKNESS_M / 2;
    const pane: BoxExtent[] = [
        {
            min: [SHARED_WALL_X_M - paneHalf, windowY0 - overlap, windowZ0 - overlap],
            max: [SHARED_WALL_X_M + paneHalf, windowY1 + overlap, windowZ1 + overlap],
        },
    ];
    return { wall, pane };
}

/**
 * Two rooms whose shared wall is a single mesh with two material slots: an
 * opaque wall with a window (slot 0) and a clear glass pane filling the
 * window (slot 1, transmission 1, `blocksLight: false`). Light reaches room B
 * through the glass only; a renderer that treats the whole mesh as one
 * material gets either no light there or light through the whole wall.
 */
export function buildGlassWall(): ProceduralScene {
    const builder = new SceneBuilder();
    addRooms(builder);

    const wallMaterial = builder.addDiffuse('partition-wall', [0.8, 0.8, 0.78]);
    const glassMaterial = builder.addMaterial({ name: 'glass-pane', albedo: [1, 1, 1], transmission: 1, blocksLight: false, doubleSided: true });
    const { wall, pane } = glassWallBoxes();
    const mesh = builder.addMesh(
        createGroupedBoxMesh('glass-wall', [
            { materialSlot: 0, boxes: wall },
            { materialSlot: 1, boxes: pane },
        ]),
    );
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    builder.addInstance('glass-wall', mesh, [wallMaterial, glassMaterial], identity);
    return builder.build();
}
