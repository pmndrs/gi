/**
 * The uv board (M4, T4.4): eight textured panels on a wall, each sampling its
 * texture a different way, out in the sun.
 *
 * Every panel has its own orientation texture (four coloured quadrants, a
 * marker at the uv origin, a bar along u, a grid; see textures.ts), so a
 * mapping done wrong shows as the wrong colour in the wrong place: a flipped
 * axis, a rotation the wrong way, a wrap mode ignored, the wrong uv set.
 *
 * Panels, left to right, top row then bottom row:
 *   repeat      tiled 3 × 2
 *   mirror      tiled 3 × 2, every other tile mirrored (three's MirroredRepeatWrapping)
 *   clamp       shrunk to the middle half, its edge texels smeared outwards
 *   offset      shifted by (0.25, 0.5), wrapping round
 *   rotation    turned 30 degrees, tiled 1.5 × 1.5
 *   uv set 1    sampled through the second uv set, which runs across the
 *               first (u and v swapped) and twice as fast
 *   mixed wrap  clamped along u and mirrored along v, so each axis has its own mode
 *   combined    offset, repeat, a -45 degree turn, mirror along u and repeat along v together
 *
 * The panels are quads 1 cm in front of a light wall, facing +z. The sun and
 * the sky light them from the front, so their colours bounce onto the ground
 * in front of them: GI sees the textures there.
 */

import type { LinearColor, ProceduralScene, SceneMeshData, SceneUvMapping } from '../scene-data';
import { composeTransform, createUnitQuad, SceneBuilder } from './geometry';
import { createOrientationTexture, type QuadrantPalette } from './textures';

export const UV_BOARD_PANEL_SIZE_M = 1.2;
/** Towards the sun: in front of the board (+z), a little to the right and high. The manifest's sky uses the same. */
export const UV_BOARD_TO_SUN: readonly [number, number, number] = [0.35, 0.75, 0.56];
/** How far the panels stand in front of the wall's face (z = 0). */
const PANEL_GAP_M = 0.01;

/** One panel: its name, its texture's colours and how the texture is sampled. */
interface PanelSpec {
    readonly name: string;
    readonly palette: QuadrantPalette;
    readonly uv: SceneUvMapping;
}

function palette(nearOrigin: LinearColor, alongU: LinearColor, alongV: LinearColor, farCorner: LinearColor): QuadrantPalette {
    return { nearOrigin, alongU, alongV, farCorner };
}

/** The eight panels, in the order the file comment lists them. */
export const UV_BOARD_PANELS: readonly PanelSpec[] = [
    { name: 'repeat', palette: palette([0.7, 0.08, 0.05], [0.05, 0.45, 0.08], [0.06, 0.1, 0.6], [0.75, 0.6, 0.05]), uv: { repeat: [3, 2] } },
    {
        name: 'mirror',
        palette: palette([0.6, 0.1, 0.45], [0.08, 0.5, 0.5], [0.7, 0.35, 0.05], [0.25, 0.25, 0.25]),
        uv: { repeat: [3, 2], wrapU: 'mirror', wrapV: 'mirror' },
    },
    {
        name: 'clamp',
        palette: palette([0.05, 0.3, 0.7], [0.7, 0.7, 0.1], [0.5, 0.05, 0.05], [0.1, 0.6, 0.2]),
        uv: { repeat: [2, 2], offset: [-0.5, -0.5], wrapU: 'clamp', wrapV: 'clamp' },
    },
    { name: 'offset', palette: palette([0.75, 0.4, 0.3], [0.2, 0.2, 0.55], [0.3, 0.55, 0.1], [0.55, 0.1, 0.3]), uv: { offset: [0.25, 0.5] } },
    {
        name: 'rotation',
        palette: palette([0.1, 0.55, 0.35], [0.65, 0.2, 0.1], [0.35, 0.15, 0.6], [0.6, 0.55, 0.4]),
        uv: { rotation: Math.PI / 6, repeat: [1.5, 1.5] },
    },
    { name: 'uv-set-1', palette: palette([0.75, 0.55, 0.1], [0.1, 0.1, 0.1], [0.1, 0.45, 0.65], [0.6, 0.1, 0.1]), uv: { channel: 1 } },
    {
        name: 'mixed-wrap',
        palette: palette([0.3, 0.6, 0.6], [0.6, 0.3, 0.05], [0.15, 0.3, 0.05], [0.7, 0.7, 0.7]),
        uv: { repeat: [2.5, 2.5], offset: [-0.75, -0.75], wrapU: 'clamp', wrapV: 'mirror' },
    },
    {
        name: 'combined',
        palette: palette([0.5, 0.5, 0.1], [0.05, 0.25, 0.45], [0.6, 0.25, 0.4], [0.2, 0.5, 0.25]),
        uv: { offset: [0.1, 0.3], repeat: [2, 2], rotation: -Math.PI / 4, wrapU: 'mirror', wrapV: 'repeat' },
    },
];

/**
 * The unit quad with a second uv set that runs across the first: uv1 =
 * (2v, 2u). A renderer that reads uv set 0 instead shows the texture once,
 * upright; uv set 1 shows it 2 × 2 times, mirrored across the diagonal.
 */
export function createQuadWithSecondUvSet(): SceneMeshData {
    const quad = createUnitQuad('unit-quad-uv1');
    const uvs1 = new Float32Array(quad.uvs.length);
    for (let vertex = 0; vertex < quad.uvs.length / 2; vertex++) {
        uvs1[vertex * 2] = 2 * quad.uvs[vertex * 2 + 1];
        uvs1[vertex * 2 + 1] = 2 * quad.uvs[vertex * 2];
    }
    return { ...quad, uvs1 };
}

/** Where panel `index` stands: four columns, two rows, centred on x = 0. */
export function panelCentre(index: number): [number, number, number] {
    const column = index % 4;
    const row = Math.floor(index / 4);
    const pitch = 1.5;
    return [(column - 1.5) * pitch, row === 0 ? 2.45 : 0.95, PANEL_GAP_M];
}

export function buildUvBoard(): ProceduralScene {
    const builder = new SceneBuilder();
    const ground = builder.addDiffuse('ground-light-grey', [0.6, 0.6, 0.58]);
    const wall = builder.addDiffuse('board-wall', [0.7, 0.7, 0.68]);
    const quadWithSecondUvSet = builder.addMesh(createQuadWithSecondUvSet());

    builder.addBox('ground', ground, [-6, -0.2, -3], [6, 0, 7]);
    // The wall's front face is at z = 0; it runs into the ground.
    builder.addBox('board-wall', wall, [-3.3, -0.05, -0.3], [3.3, 3.3, 0]);

    UV_BOARD_PANELS.forEach((panel, index) => {
        const texture = builder.addTexture(createOrientationTexture(`uv-board-${panel.name}`, panel.palette));
        const material = builder.addMaterial({ name: `panel-${panel.name}`, albedo: [1, 1, 1], albedoMap: { texture, uv: panel.uv } });
        const transform = composeTransform(panelCentre(index), [UV_BOARD_PANEL_SIZE_M, UV_BOARD_PANEL_SIZE_M, 1]);
        if (panel.uv.channel === 1) builder.addInstance(`panel-${panel.name}`, quadWithSecondUvSet, [material], transform);
        else builder.addQuad(`panel-${panel.name}`, material, transform);
    });

    builder.addLight({
        type: 'directional',
        label: 'sun',
        color: [1, 0.95, 0.85],
        intensity: 3,
        direction: [-UV_BOARD_TO_SUN[0], -UV_BOARD_TO_SUN[1], -UV_BOARD_TO_SUN[2]],
    });

    return builder.build();
}
