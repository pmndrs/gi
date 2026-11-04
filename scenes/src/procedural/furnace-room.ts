/**
 * A furnace room: a closed box where every surface has the same albedo and the
 * same emission. The right answer is known exactly, so it tests the energy of
 * the whole GI pipeline with no need for a reference renderer.
 *
 * The closed-form answer
 * ----------------------
 * Let rho be the albedo and Le the emitted radiance of every surface. In a
 * closed room every ray leaving a surface hits another surface, so the
 * incoming radiance is the outgoing radiance of whatever it hits. Try a
 * uniform radiance L everywhere. At any surface point:
 *
 *   L = Le + (rho / pi) * integral over the hemisphere of L cos(theta) dw
 *     = Le + (rho / pi) * L * pi
 *     = Le + rho * L
 *
 * which gives L = Le / (1 - rho). That is the sum of the bounce series
 * Le * (1 + rho + rho^2 + ...). It holds for any closed geometry, including
 * the blocks inside this room, because the hemisphere integral of cos(theta)
 * is always pi.
 *
 * With rho = 0.5 and Le = 0.25:
 *   - radiance seen from anywhere:                  L = 0.25 / 0.5 = 0.5
 *   - irradiance E on a surface facing any way:     E = pi * L = 1.5708
 *   - the probe's irradiance divided by pi:         0.5
 *
 * A GI result that is not uniformly 0.5 is wrong: lower means energy is lost
 * (too few bounces, a recursion cap, leaks to a missing sky), higher means
 * energy is created.
 *
 * rho is kept at 0.5, below NVIDIA's native recursion cap of 0.9, so the cap
 * does not change the answer.
 */

import type { ProceduralScene } from '../scene-data';
import { SceneBuilder } from './geometry';
import { addRoomShell } from './room-shell';

export const FURNACE_ALBEDO = 0.5;
export const FURNACE_EMISSION = 0.25;
/** The exact radiance everywhere in the room: Le / (1 - rho). */
export const FURNACE_EXPECTED_RADIANCE = FURNACE_EMISSION / (1 - FURNACE_ALBEDO);

export const FURNACE_SIZE_M: readonly [number, number, number] = [5, 2.8, 4];

export function buildFurnaceRoom(): ProceduralScene {
    const builder = new SceneBuilder();
    const everything = builder.addMaterial({
        name: 'furnace-surface',
        albedo: [FURNACE_ALBEDO, FURNACE_ALBEDO, FURNACE_ALBEDO],
        emissive: [FURNACE_EMISSION, FURNACE_EMISSION, FURNACE_EMISSION],
    });

    const [sizeX, sizeY, sizeZ] = FURNACE_SIZE_M;
    addRoomShell(builder, 'furnace', [0, 0, 0], [sizeX, sizeY, sizeZ], 0.1, {
        wall: everything,
        floor: everything,
        ceiling: everything,
    });

    // Objects inside make occlusion matter. Because they use the same material
    // the answer is still the same everywhere.
    builder.addBox('block-low', everything, [0.6, 0, 0.6], [1.8, 0.8, 1.6]);
    builder.addBoxCentered('block-rotated', everything, [3.6, 0.6, 2.6], [1, 1.2, 1], 30);
    builder.addBox('pillar', everything, [2.3, 0, 3.2], [2.6, sizeY, 3.5]);

    return builder.build();
}
