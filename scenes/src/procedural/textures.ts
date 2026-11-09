/**
 * Textures made in code for the textured test scenes (M4, T4.4), and the
 * small colour and uv helpers they need. No files, no engine: every texture
 * is 8-bit RGBA pixels in a {@link SceneTextureData}.
 *
 * Colour: albedo textures are stored sRGB-encoded, as glTF requires for a
 * base colour texture, and every GPU decodes them with the standard sRGB
 * curve (`srgbByteToLinear`). Means "in linear light" are means of the
 * decoded values, which is what a renderer averages when it filters or
 * builds mips correctly.
 */

import type { LinearColor, SceneTextureData, SceneUvMapping } from '../scene-data';

// ---------------------------------------------------------------------------
// sRGB
// ---------------------------------------------------------------------------

/** The sRGB decode (IEC 61966-2-1) of one 8-bit value, as GPUs apply it to `*-srgb` formats. */
export function srgbByteToLinear(byte: number): number {
    const encoded = byte / 255;
    return encoded <= 0.04045 ? encoded / 12.92 : ((encoded + 0.055) / 1.055) ** 2.4;
}

/** The nearest 8-bit sRGB value for a linear value in 0..1. */
export function linearToSrgbByte(linear: number): number {
    const clamped = Math.min(Math.max(linear, 0), 1);
    const encoded = clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
}

/** Mean of each channel over every texel, in linear light (decoded first when `srgb`). Alpha is always linear. */
export function textureMeanLinear(texture: SceneTextureData): [number, number, number, number] {
    const sums = [0, 0, 0, 0];
    const texels = texture.width * texture.height;
    for (let texel = 0; texel < texels; texel++) {
        for (let channel = 0; channel < 4; channel++) {
            const byte = texture.data[texel * 4 + channel];
            const isColour = channel < 3;
            sums[channel] += isColour && texture.srgb ? srgbByteToLinear(byte) : byte / 255;
        }
    }
    return [sums[0] / texels, sums[1] / texels, sums[2] / texels, sums[3] / texels];
}

// ---------------------------------------------------------------------------
// Pixel helpers
// ---------------------------------------------------------------------------

/** 8-bit RGBA, colour channels sRGB-encoded. */
type Rgba8 = readonly [number, number, number, number];

function encodeLinear(color: LinearColor, alpha = 1): Rgba8 {
    return [linearToSrgbByte(color[0]), linearToSrgbByte(color[1]), linearToSrgbByte(color[2]), Math.round(alpha * 255)];
}

/** Builds a texture by asking `texelAt(x, y)` for every texel; y is the row, row 0 is v = 0. */
function paint(name: string, size: number, texelAt: (x: number, y: number) => Rgba8): SceneTextureData {
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            data.set(texelAt(x, y), (y * size + x) * 4);
        }
    }
    return { name, width: size, height: size, data, srgb: true };
}

/** The square a texel is in, on a board of `squares` × `squares`: true for the squares whose row + column is even. */
function isEvenSquare(x: number, y: number, size: number, squares: number): boolean {
    const squareSize = size / squares;
    return (Math.floor(x / squareSize) + Math.floor(y / squareSize)) % 2 === 0;
}

// ---------------------------------------------------------------------------
// The textured furnace's checkerboard
// ---------------------------------------------------------------------------

/**
 * The two grey levels of the textured furnace's checkerboard, as 8-bit sRGB:
 * the pair near 0.25 and 0.75 in linear light whose mean is closest to 0.5
 * from above (found by search over every pair). No 8-bit sRGB pair other
 * than 0 and 255 averages to exactly 0.5, so the material's colour factor
 * (`FURNACE_CHECKER_FACTOR`, just under 1) scales the 0.016% left over away.
 *
 *   140 decodes to 0.262251, 223 to 0.737910; mean 0.500081.
 */
export const FURNACE_CHECKER_DARK_SRGB = 140;
export const FURNACE_CHECKER_LIGHT_SRGB = 223;

/** The checkerboard's mean albedo in linear light, before the material's factor. */
export const FURNACE_CHECKER_MEAN = (srgbByteToLinear(FURNACE_CHECKER_DARK_SRGB) + srgbByteToLinear(FURNACE_CHECKER_LIGHT_SRGB)) / 2;

/**
 * Squares along each side of the furnace texture. Fine enough that any
 * point's view of the room averages many squares, coarse enough to read in
 * the viewer: on the 5 m walls a square is about 0.3 m.
 */
export const FURNACE_CHECKER_SQUARES = 16;
export const FURNACE_CHECKER_SIZE = 256;

/**
 * A grey checkerboard of equal light and dark squares. Both levels are grey,
 * so the mean is the same in every colour channel; alpha is 1. The square
 * count divides the size, so the two levels cover exactly half the texels each.
 */
export function createFurnaceChecker(): SceneTextureData {
    const dark: Rgba8 = [FURNACE_CHECKER_DARK_SRGB, FURNACE_CHECKER_DARK_SRGB, FURNACE_CHECKER_DARK_SRGB, 255];
    const light: Rgba8 = [FURNACE_CHECKER_LIGHT_SRGB, FURNACE_CHECKER_LIGHT_SRGB, FURNACE_CHECKER_LIGHT_SRGB, 255];
    return paint('furnace-checker', FURNACE_CHECKER_SIZE, (x, y) => (isEvenSquare(x, y, FURNACE_CHECKER_SIZE, FURNACE_CHECKER_SQUARES) ? light : dark));
}

// ---------------------------------------------------------------------------
// The alpha checker's cutout
// ---------------------------------------------------------------------------

export const ALPHA_CHECKER_SQUARES = 8;
export const ALPHA_CHECKER_SIZE = 256;

/**
 * A checkerboard of holes: alpha 1 on even squares, 0 on odd ones, so with
 * `alphaTest` 0.5 exactly half the board is open. The colour is the same
 * everywhere (so filtering at a hole's edge brings in no other colour), and
 * square edges fall on texel edges, where bilinear alpha crosses 0.5 exactly.
 */
export function createAlphaChecker(color: LinearColor): SceneTextureData {
    const solid = encodeLinear(color, 1);
    const hole = encodeLinear(color, 0);
    return paint('alpha-checker', ALPHA_CHECKER_SIZE, (x, y) => (isEvenSquare(x, y, ALPHA_CHECKER_SIZE, ALPHA_CHECKER_SQUARES) ? solid : hole));
}

// ---------------------------------------------------------------------------
// The uv board's orientation pattern
// ---------------------------------------------------------------------------

/** Four colours for the four quadrants of an orientation texture, in linear light. */
export interface QuadrantPalette {
    /** u < 0.5, v < 0.5: the quadrant at the uv origin, which also holds the white marker. */
    readonly nearOrigin: LinearColor;
    /** u ≥ 0.5, v < 0.5. */
    readonly alongU: LinearColor;
    /** u < 0.5, v ≥ 0.5. */
    readonly alongV: LinearColor;
    /** u ≥ 0.5, v ≥ 0.5. */
    readonly farCorner: LinearColor;
}

export const ORIENTATION_TEXTURE_SIZE = 256;
/** Grid lines every 1/8 of the texture, so a tiled or scaled mapping can be counted. */
const ORIENTATION_GRID_CELLS = 8;

/**
 * A texture whose every flip, turn and shift looks different: four
 * quadrants of different colours (any mirror or rotation moves at least two
 * of them), a white square marker near the uv origin (it says which corner is
 * (0, 0)), a white bar along the u axis from the marker (it says which way u
 * runs), and dark grid lines every eighth inside it (they show tiling and
 * scale).
 */
export function createOrientationTexture(name: string, palette: QuadrantPalette): SceneTextureData {
    const size = ORIENTATION_TEXTURE_SIZE;
    const cell = size / ORIENTATION_GRID_CELLS;
    const quadrants: Rgba8[] = [palette.nearOrigin, palette.alongU, palette.alongV, palette.farCorner].map((color) => encodeLinear(color));
    const white = encodeLinear([0.9, 0.9, 0.9]);
    const gridLine = encodeLinear([0.03, 0.03, 0.03]);
    const markerMin = cell * 0.5;
    const markerMax = cell * 1.5;
    const barEnd = size * 0.45;
    const barHalfWidth = cell * 0.15;
    const barCentre = cell;
    return paint(name, size, (x, y) => {
        const inMarker = x >= markerMin && x < markerMax && y >= markerMin && y < markerMax;
        const inBar = x >= markerMax && x < barEnd && Math.abs(y + 0.5 - barCentre) < barHalfWidth;
        if (inMarker || inBar) return white;
        // Grid lines one texel wide on each side of every inner cell edge. Not on the texture's own
        // edges: a clamped mapping smears its edge texels outwards, and those should show the quadrants.
        const onGridLine = (coordinate: number) => (coordinate % cell === 0 && coordinate > 0) || (coordinate % cell === cell - 1 && coordinate < size - 1);
        const onGrid = onGridLine(x) || onGridLine(y);
        if (onGrid) return gridLine;
        const quadrant = (x < size / 2 ? 0 : 1) + (y < size / 2 ? 0 : 2);
        return quadrants[quadrant];
    });
}

// ---------------------------------------------------------------------------
// uv mappings
// ---------------------------------------------------------------------------

/**
 * The 3×2 affine matrix of a mapping, in the contract's layout
 * (`UvMapping.matrix`: `uv' = (m[0]·u + m[2]·v + m[4], m[1]·u + m[3]·v + m[5])`),
 * built the way three's `Texture.updateMatrix` builds it with `center` (0, 0):
 *
 *   u' = sx·( cos r·u + sin r·v) + tx
 *   v' = sy·(−sin r·u + cos r·v) + ty
 *
 * Throws on a rotated mapping with an uneven repeat, where three (scale after
 * rotation) and glTF's `KHR_texture_transform` (scale before) would disagree.
 */
export function uvMappingMatrix(mapping: SceneUvMapping = {}): [number, number, number, number, number, number] {
    const [offsetU, offsetV] = mapping.offset ?? [0, 0];
    const [repeatU, repeatV] = mapping.repeat ?? [1, 1];
    const rotation = mapping.rotation ?? 0;
    if (rotation !== 0 && repeatU !== repeatV) {
        throw new Error(`a rotated uv mapping needs the same repeat on both axes (got ${repeatU} and ${repeatV}): three and glTF apply them in different orders`);
    }
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);
    return [repeatU * cos, -repeatV * sin, repeatU * sin, repeatV * cos, offsetU, offsetV];
}
