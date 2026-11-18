/**
 * The furnace room with a checkerboard albedo texture (M4, T4.4).
 *
 * Same geometry and emission as `furnace-room` (every surface emits 0.25),
 * but the albedo is a grey checkerboard of about 0.26 and 0.74 in linear
 * light whose mean is 0.5: the material's colour factor times the texture's
 * mean is 0.5 to float precision (textures.ts says why a factor is needed).
 *
 * The answer is the flat furnace's: a probe's irradiance / pi is 0.5
 * everywhere. Light leaving a point is Le + albedo(x) * E(x) / pi, so it
 * differs between dark and light squares, but what reaches any point is an
 * average over many squares of the room it sees, whose mean albedo is 0.5,
 * so E / pi = Le / (1 - 0.5) as before. That holds to the extent the squares
 * average out over a view (16 squares a side, about 0.3 m on a 5 m wall); a
 * renderer that samples the texture's mean at coarse mips gets it exactly.
 *
 * What it tests: textured albedo at ray hits, in linear light. A renderer that
 * filters the sRGB-encoded values before decoding them sees a mean albedo of
 * 0.465 and settles at 0.25 / 0.535 = 0.467 instead of 0.5 (7% dark); one
 * that skips the decode sees 0.712 and settles at 0.87; one that ignores the
 * texture and uses the factor alone (about 1.0) never settles.
 */

import type { ProceduralScene } from '../scene-data';
import { addFurnaceGeometry, FURNACE_ALBEDO, FURNACE_EMISSION } from './furnace-room';
import { SceneBuilder } from './geometry';
import { createFurnaceChecker, FURNACE_CHECKER_MEAN } from './textures';

/** The material's colour factor: scales the checkerboard's mean onto the furnace's 0.5 exactly. */
export const TEXTURED_FURNACE_FACTOR = FURNACE_ALBEDO / FURNACE_CHECKER_MEAN;

export function buildTexturedFurnace(): ProceduralScene {
    const builder = new SceneBuilder();
    const checker = builder.addTexture(createFurnaceChecker());
    const everything = builder.addMaterial({
        name: 'furnace-checker-surface',
        albedo: [TEXTURED_FURNACE_FACTOR, TEXTURED_FURNACE_FACTOR, TEXTURED_FURNACE_FACTOR],
        albedoMap: { texture: checker },
        emissive: [FURNACE_EMISSION, FURNACE_EMISSION, FURNACE_EMISSION],
    });
    addFurnaceGeometry(builder, everything);
    return builder.build();
}
