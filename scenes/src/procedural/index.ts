import type { BuilderOptions, ProceduralBuilder, ProceduralScene } from '../scene-data';
import { buildCabinetRow } from './cabinet-row';
import { buildCornellBox } from './cornell-box';
import { buildCourtyard } from './courtyard';
import { buildEmissiveRoom } from './emissive-room';
import { buildFurnaceRoom } from './furnace-room';
import { buildOpenGround } from './open-ground';
import { buildAlphaChecker, buildGlassWall } from './see-through-walls';
import { buildTexturedFurnace } from './textured-furnace';
import { buildThinWall } from './thin-wall';
import { buildClosedDoor, buildTwoRoom } from './two-rooms';
import { buildUvBoard } from './uv-board';

/**
 * Builders by the name scene manifests use in `parts[].builder`. Names are
 * kebab-case and stable: renaming one breaks every manifest that mentions it.
 */
export const proceduralBuilders: Readonly<Record<string, ProceduralBuilder>> = {
    'cornell-box': buildCornellBox,
    courtyard: buildCourtyard,
    'two-room': buildTwoRoom,
    'furnace-room': buildFurnaceRoom,
    'thin-wall': buildThinWall,
    'closed-door': buildClosedDoor,
    'cabinet-row': buildCabinetRow,
    'open-ground': buildOpenGround,
    // M4's textured scenes (T4.4).
    'textured-furnace': buildTexturedFurnace,
    'uv-board': buildUvBoard,
    'alpha-checker': buildAlphaChecker,
    'glass-wall': buildGlassWall,
    // C5's room lit by emissive surfaces only.
    'emissive-room': buildEmissiveRoom,
};

export function buildProcedural(name: string, options?: BuilderOptions): ProceduralScene {
    const builder = proceduralBuilders[name];
    if (!builder) {
        const known = Object.keys(proceduralBuilders).join(', ');
        throw new Error(`Unknown procedural builder "${name}". Known builders: ${known}`);
    }
    return builder(options);
}

export {
    buildAlphaChecker,
    buildCabinetRow,
    buildClosedDoor,
    buildCornellBox,
    buildCourtyard,
    buildEmissiveRoom,
    buildFurnaceRoom,
    buildGlassWall,
    buildOpenGround,
    buildTexturedFurnace,
    buildThinWall,
    buildTwoRoom,
    buildUvBoard,
};
