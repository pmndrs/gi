import type { BuilderOptions, ProceduralBuilder, ProceduralScene } from '../scene-data';
import { buildCabinetRow } from './cabinet-row';
import { buildCornellBox } from './cornell-box';
import { buildCourtyard } from './courtyard';
import { buildFurnaceRoom } from './furnace-room';
import { buildThinWall } from './thin-wall';
import { buildClosedDoor, buildTwoRoom } from './two-rooms';

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
    buildCabinetRow,
    buildClosedDoor,
    buildCornellBox,
    buildCourtyard,
    buildFurnaceRoom,
    buildThinWall,
    buildTwoRoom,
};
