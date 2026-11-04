/**
 * A small walled courtyard for the tree to stand in.
 *
 * A 10 m by 10 m paved yard centred on the origin (x and z from -5 to 5), with
 * 2.6 m plaster walls on all four sides and a gateway in the south wall
 * (1.4 m wide, 2.2 m high, at x = -2). A stone bench stands against the north
 * wall. The tree model is placed at the origin by the manifest.
 *
 * The sun is the only light, so this scene shows how a DDGI field handles a
 * sunlit pocket (bright ground, shadowed wall bases) with foliage in the middle
 * and sky light arriving only from above and through the gateway.
 *
 * Solids overlap where they meet (walls run into the ground, the gateway's
 * lintel into its jambs) so no seam is flush.
 */

import type { ProceduralScene } from '../scene-data';
import { SceneBuilder } from './geometry';

export const COURTYARD_HALF_SIZE_M = 5;
export const COURTYARD_WALL_HEIGHT_M = 2.6;
export const COURTYARD_WALL_THICKNESS_M = 0.3;
export const GATEWAY_WIDTH_M = 1.4;
export const GATEWAY_HEIGHT_M = 2.2;
export const GATEWAY_CENTRE_X_M = -2;

export function buildCourtyard(): ProceduralScene {
    const builder = new SceneBuilder();
    const paving = builder.addDiffuse('paving-stone', [0.42, 0.39, 0.34]);
    const plaster = builder.addDiffuse('plaster-wall', [0.72, 0.66, 0.55]);
    const bench = builder.addDiffuse('stone-bench', [0.5, 0.5, 0.48]);

    const half = COURTYARD_HALF_SIZE_M;
    const t = COURTYARD_WALL_THICKNESS_M;
    const h = COURTYARD_WALL_HEIGHT_M;
    const outer = half + t;

    // Ground: thick, so it has a real underside, and extends under the walls.
    builder.addBox('ground', paving, [-outer, -0.3, -outer], [outer, 0, outer]);

    // West, east and north walls, deep enough to run into the ground and into each other at the corners.
    builder.addBox('wall-west', plaster, [-outer, -0.3, -outer], [-half, h, outer]);
    builder.addBox('wall-east', plaster, [half, -0.3, -outer], [outer, h, outer]);
    builder.addBox('wall-north', plaster, [-outer, -0.3, -outer], [outer, h, -half]);

    // South wall with the gateway: a jamb either side and a lintel across the top.
    const gatewayX0 = GATEWAY_CENTRE_X_M - GATEWAY_WIDTH_M / 2;
    const gatewayX1 = GATEWAY_CENTRE_X_M + GATEWAY_WIDTH_M / 2;
    const overlap = 0.01;
    builder.addBox('wall-south-west-of-gateway', plaster, [-outer, -0.3, half], [gatewayX0, h, outer]);
    builder.addBox('wall-south-east-of-gateway', plaster, [gatewayX1, -0.3, half], [outer, h, outer]);
    builder.addBox(
        'gateway-lintel',
        plaster,
        [gatewayX0 - overlap, GATEWAY_HEIGHT_M, half],
        [gatewayX1 + overlap, h, outer],
    );

    // A bench against the north wall, sunk 1 cm into the paving and the wall.
    builder.addBox('bench-seat', bench, [-4.4, 0.4, -half - 0.01], [-2.6, 0.48, -4.5]);
    builder.addBox('bench-support-left', bench, [-4.3, -0.01, -half - 0.01], [-4.1, 0.41, -4.55]);
    builder.addBox('bench-support-right', bench, [-2.9, -0.01, -half - 0.01], [-2.7, 0.41, -4.55]);

    return builder.build();
}
