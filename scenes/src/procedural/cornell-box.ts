/**
 * The Cornell box, at the standard proportions.
 *
 * The standard data is in millimetre-ish units, 555 across. Here one standard
 * unit is one centimetre, so the box is 5.55 m wide: room-sized, which suits
 * probe spacings in metres.
 *
 * Layout: the back wall is at z = 0 and the front is open (z = depth), the way
 * the box is usually photographed. +x is to the right as seen from the front,
 * so the red wall is at x = 0 (left) and the green wall at x = width (right).
 * The tall block stands back-left, the short block front-right.
 *
 * Colours are the common Mitsuba/PBRT reflectances (linear). The ceiling light
 * is 1.3 m by 1.05 m with radiance (17, 12, 4), as in the standard file.
 *
 * Options:
 * - `light: 'emissive'` (default): the ceiling emitter lights the box.
 * - `light: 'point'`: no emitter; one isotropic point light under the ceiling.
 */

import type { BuilderOptions, ProceduralScene, SceneLightData, Vector3Tuple } from '../scene-data';
import { SceneBuilder } from './geometry';

export const CORNELL_WIDTH_M = 5.55;
export const CORNELL_HEIGHT_M = 5.488;
export const CORNELL_DEPTH_M = 5.592;
const WALL_THICKNESS_M = 0.1;

const LIGHT_SIZE_X_M = 1.3;
const LIGHT_SIZE_Z_M = 1.05;
const LIGHT_RADIANCE: Vector3Tuple = [17, 12, 4];

/** Where the point light sits in `light: 'point'` mode: under the middle of the ceiling. */
const POINT_LIGHT_POSITION: Vector3Tuple = [CORNELL_WIDTH_M / 2, CORNELL_HEIGHT_M - 0.3, CORNELL_DEPTH_M / 2];

export function buildCornellBox(options: BuilderOptions = {}): ProceduralScene {
    const lightMode = options.light ?? 'emissive';
    if (lightMode !== 'emissive' && lightMode !== 'point') {
        throw new Error(`cornell-box: option "light" must be "emissive" or "point", got "${String(lightMode)}"`);
    }

    const w = CORNELL_WIDTH_M;
    const h = CORNELL_HEIGHT_M;
    const d = CORNELL_DEPTH_M;
    const t = WALL_THICKNESS_M;

    const builder = new SceneBuilder();
    const white = builder.addDiffuse('cornell-white', [0.725, 0.71, 0.68]);
    const red = builder.addDiffuse('cornell-red', [0.63, 0.065, 0.05]);
    const green = builder.addDiffuse('cornell-green', [0.14, 0.45, 0.091]);

    builder.addBox('floor', white, [0, -t, 0], [w, 0, d]);
    builder.addBox('ceiling', white, [0, h, 0], [w, h + t, d]);
    builder.addBox('back-wall', white, [0, 0, -t], [w, h, 0]);
    builder.addBox('left-wall-red', red, [-t, -t, -t], [0, h + t, d]);
    builder.addBox('right-wall-green', green, [w, -t, -t], [w + t, h + t, d]);

    // Blocks, 165 standard units on a side; the tall one is 330 high.
    const blockSide = 1.65;
    builder.addBoxCentered('tall-block', white, [1.865, 1.65, 2.08], [blockSide, 3.3, blockSide], 17);
    builder.addBoxCentered('short-block', white, [3.7, 0.825, 3.9], [blockSide, blockSide, blockSide], -18);

    const lights: SceneLightData[] = [];
    if (lightMode === 'emissive') {
        const emitter = builder.addMaterial({
            name: 'cornell-light',
            albedo: [0, 0, 0],
            emissive: LIGHT_RADIANCE,
        });
        // 2 mm below the ceiling so the emitter is never coplanar with it.
        builder.addDownwardQuad('ceiling-light', emitter, [w / 2, h - 0.002, d / 2], LIGHT_SIZE_X_M, LIGHT_SIZE_Z_M);
    } else {
        lights.push({
            type: 'point',
            label: 'cornell-point-light',
            color: [1, 0.72, 0.28],
            // Chosen so the box looks comparably lit to the emissive version;
            // an isotropic point light with the emitter's exact power would be dimmer.
            intensity: 15,
            position: POINT_LIGHT_POSITION,
            range: 20,
        });
    }

    const scene = builder.build();
    return { ...scene, lights };
}
