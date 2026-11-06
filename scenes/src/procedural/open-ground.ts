/**
 * Open ground: a large flat grey ground under the sky, with one 1 m cube, and
 * no sun. It checks the shared sky's brightness and orientation.
 *
 * With nothing above it, the ground's top face receives exactly the sky's
 * horizontal irradiance E, the number the sky file is calibrated to
 * (`scenes/src/sky/sky-image.ts`). A diffuse surface of albedo rho then shows
 * radiance rho * E / pi. So every renderer that reads the sky file must show
 * the same ground brightness, and it can be checked against the sky file's
 * own sidecar (`scenes/sky/open-ground.json`).
 *
 * The ground is 200 m square so that, from the bookmarks, it fills the view
 * and the sky it sees is hardly cut by its edges. The cube stands at the
 * origin, well away from the first bookmark's view, so its shadow and its
 * bounce light don't reach the measured patch; the second bookmark looks past
 * it towards the sun's glow, to see orientation by eye.
 */

import type { ProceduralScene } from '../scene-data';
import { SceneBuilder } from './geometry';

export const OPEN_GROUND_ALBEDO = 0.5;
export const OPEN_GROUND_HALF_SIZE_M = 100;
const GROUND_THICKNESS_M = 0.1;

export function buildOpenGround(): ProceduralScene {
    const builder = new SceneBuilder();
    const grey = builder.addDiffuse('ground-grey', [OPEN_GROUND_ALBEDO, OPEN_GROUND_ALBEDO, OPEN_GROUND_ALBEDO]);
    const half = OPEN_GROUND_HALF_SIZE_M;
    // Top face at y = 0.
    builder.addBox('ground', grey, [-half, -GROUND_THICKNESS_M, -half], [half, 0, half]);
    builder.addBox('cube', grey, [-0.5, 0, -0.5], [0.5, 1, 0.5]);
    return builder.build();
}
