/**
 * DDGI's probe atlases drawn over the picture: the irradiance atlas (the
 * light each probe has gathered, one small square per probe) or the distance
 * atlas (how far each probe sees in every direction), as a picture in the
 * corner or filling the view.
 *
 * What the atlases hold (docs/design/ddgi.md, "The textures"): one array
 * layer per horizontal row of probes (y), and inside a layer probe (x, z)
 * owns tile (x, z). Each tile is an octahedral map of directions with a
 * one-texel border. This draws every layer side by side in a grid, the
 * lowest row of probes first (top left), with a dark gap between layers, one
 * screen pixel per texel or a whole multiple of it, so tiles stay sharp.
 *
 * It is drawn in display space, over the picture as the screen shows it
 * (after tone mapping and output colour; the viewer's pipeline calls
 * `overlay`), so it sits on top of the change-per-frame view too.
 *
 * Colours:
 * - irradiance: E / π, the light a white matte surface facing that way would
 *   show, times 2^(atlas exposure), then the viewer's exposure, tone mapping
 *   and output colour, as the picture gets them;
 * - distance: the mean distance as a display grey, black at the probe and
 *   white at DISTANCE_WHITE_CELL_DIAGONALS cell diagonals, the most the atlas
 *   stores; no tone mapping, so the greys mean the same in every scene.
 *
 * The overlay reads the same `GPUTexture`s the shading module hands three
 * (through `ShadingNodes`), so it shows exactly what the picture is lit from.
 */

import { mix, renderOutput, screenUV, uniform, vec4, wgslFn } from 'three/tsl';
import { Vector2, Vector4, type Node, type WebGPURenderer } from 'three/webgpu';
import type { DDGIField, ShadingModule } from './library';
import { ShadingNodes } from './shading-nodes';

export type AtlasChoice = 'irradiance' | 'distance';

/** The shading module's names for the two atlases (DDGI_SHADING_PARAMETER_NAMES). */
const ATLAS_PARAMETER: Record<AtlasChoice, string> = {
    irradiance: 'ddgiIrradiance',
    distance: 'ddgiDistance',
};

/**
 * Distances are capped at 1.5 cell diagonals when DDGI stores them
 * (`distanceClamp` in ddgi-core's settings.ts), so white means "at the cap".
 */
const DISTANCE_WHITE_CELL_DIAGONALS = 1.5;

/** Dark texels between two layers, so the rows of probes read apart. */
const LAYER_GAP_TEXELS = 2;

/** In the corner, the overlay is at most this share of the view's width and height. */
const CORNER_MAX_SHARE = { width: 0.5, height: 0.6 } as const;
/** Pixels between the overlay and the edges of the room it has. */
const MARGIN_PIXELS = 12;

/** Grey behind the layers and in the gaps: display grey for distance, linear (before tone mapping) for irradiance. */
const BACKGROUND_GREY = 0.02;

const OVERLAY_WGSL = /* wgsl */ `
// One texel of the atlas overlay at this screen position, or alpha 0 outside it.
//  placement:   the overlay's rectangle in screen UV: x, y of its top-left corner, then width, height
//  arrangement: one layer's width and height in texels, the layer count, the columns of layers
//  display:     x: 0 irradiance, 1 distance; y: the irradiance scale (2^stops / pi);
//               z: 1 / the distance shown white; w: unused
// Irradiance comes back linear (the caller tone maps it), distance as a display grey.
fn viewerDdgiAtlasOverlay(
    screenPosition: vec2f,
    atlas: texture_2d_array<f32>,
    placement: vec4f,
    arrangement: vec4f,
    display: vec4f,
) -> vec4f {
    let local = (screenPosition - placement.xy) / placement.zw;
    if (any(local < vec2f(0.0)) || any(local >= vec2f(1.0))) {
        return vec4f(0.0);
    }
    let background = vec4f(vec3f(${BACKGROUND_GREY}), 1.0);
    let layerSize = arrangement.xy;
    let layers = u32(arrangement.z + 0.5);
    let columns = u32(arrangement.w + 0.5);
    let rows = (layers + columns - 1u) / columns;
    // Overlay texels, layers plus the gaps between them.
    let cell = layerSize + vec2f(${LAYER_GAP_TEXELS}.0);
    let canvas = vec2f(f32(columns), f32(rows)) * cell - vec2f(${LAYER_GAP_TEXELS}.0);
    let position = local * canvas;
    let column = u32(position.x / cell.x);
    let row = u32(position.y / cell.y);
    let inLayer = position - vec2f(f32(column), f32(row)) * cell;
    let layer = row * columns + column;
    if (layer >= layers || any(inLayer >= layerSize)) {
        return background;
    }
    let texel = textureLoad(atlas, vec2u(inLayer), layer, 0);
    if (display.x < 0.5) {
        return vec4f(texel.rgb * display.y, 1.0);
    }
    // The distance atlas holds the mean distance and its square; show the mean.
    return vec4f(vec3f(saturate(texel.r * display.z)), 1.0);
}
`;

const drawingBufferSize = new Vector2();

export class AtlasOverlay {
    private readonly nodes: ShadingNodes;
    private readonly entry = wgslFn(OVERLAY_WGSL);
    private readonly placement = uniform(new Vector4(0, 0, 0, 0));
    private readonly arrangement = uniform(new Vector4(1, 1, 1, 1));
    private readonly display = uniform(new Vector4(0, 1, 1, 1));
    private choice: AtlasChoice = 'irradiance';

    constructor(shading: ShadingModule) {
        this.nodes = new ShadingNodes(shading);
    }

    /**
     * `display` (the picture as the screen shows it) with the chosen atlas drawn over it. Build
     * again when the atlas choice changes.
     */
    overlay(display: Node, choice: AtlasChoice): Node {
        this.choice = choice;
        const atlas = this.nodes.byName[ATLAS_PARAMETER[choice]];
        const texel = this.entry({
            screenPosition: screenUV,
            atlas,
            placement: this.placement,
            arrangement: this.arrangement,
            display: this.display,
        }) as unknown as Node<'vec4'>;
        // Irradiance is light, so it gets the picture's exposure, tone mapping and output colour; distance is already a display grey.
        const shown = choice === 'irradiance' ? (renderOutput(vec4(texel.rgb, 1)) as unknown as Node<'vec4'>) : texel;
        return mix(display as Node<'vec4'>, vec4(shown.rgb, 1), texel.a) as unknown as Node;
    }

    /**
     * Once per frame, before drawing: follows a reallocated atlas, and places the overlay for the
     * current view size. `fill` fills the view; otherwise it sits in the bottom-right corner.
     * `topInsetShare` is the share of the view's height at the top that something else covers (the
     * stats box): the overlay stays below it.
     */
    refresh(renderer: WebGPURenderer, field: DDGIField, options: { fill: boolean; exposureStops: number; topInsetShare: number }): void {
        this.nodes.refresh();
        const atlas = this.choice === 'irradiance' ? field.layout.irradiance : field.layout.distance;
        // Layers in a near-square grid of columns.
        const columns = Math.ceil(Math.sqrt(atlas.layers));
        const rows = Math.ceil(atlas.layers / columns);
        this.arrangement.value.set(atlas.width, atlas.height, atlas.layers, columns);

        const canvasWidth = columns * (atlas.width + LAYER_GAP_TEXELS) - LAYER_GAP_TEXELS;
        const canvasHeight = rows * (atlas.height + LAYER_GAP_TEXELS) - LAYER_GAP_TEXELS;
        renderer.getDrawingBufferSize(drawingBufferSize);
        const viewWidth = Math.max(1, drawingBufferSize.width);
        const viewHeight = Math.max(1, drawingBufferSize.height);
        // The room it has: below the covered top, inside the margins.
        const roomTop = Math.min(0.9, Math.max(0, options.topInsetShare)) * viewHeight;
        const roomHeight = viewHeight - roomTop;
        const boxWidth = options.fill ? viewWidth - 2 * MARGIN_PIXELS : viewWidth * CORNER_MAX_SHARE.width;
        const boxHeight = options.fill ? roomHeight - 2 * MARGIN_PIXELS : Math.min(viewHeight * CORNER_MAX_SHARE.height, roomHeight - 2 * MARGIN_PIXELS);
        const fitted = Math.max(1e-3, Math.min(boxWidth / canvasWidth, boxHeight / canvasHeight));
        // Whole screen pixels per texel when there is room, so every texel is a sharp square.
        const pixelsPerTexel = fitted >= 1 ? Math.floor(fitted) : fitted;
        const width = canvasWidth * pixelsPerTexel;
        const height = canvasHeight * pixelsPerTexel;
        // Screen UV runs from the top-left corner.
        const left = options.fill ? (viewWidth - width) / 2 : viewWidth - width - MARGIN_PIXELS;
        const top = options.fill ? roomTop + (roomHeight - height) / 2 : viewHeight - height - MARGIN_PIXELS;
        this.placement.value.set(left / viewWidth, top / viewHeight, width / viewWidth, height / viewHeight);

        const cellDiagonal = Math.hypot(...field.grid.spacing);
        this.display.value.set(
            this.choice === 'irradiance' ? 0 : 1,
            2 ** options.exposureStops / Math.PI,
            1 / (DISTANCE_WHITE_CELL_DIAGONALS * cellDiagonal),
            0,
        );
    }
}
