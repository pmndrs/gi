/**
 * The octahedral tile border rule in WGSL. Its own module so a GPU test can
 * check exactly this code against `octahedral.ts` (tools/gpu/pages/ddgi-wgsl-units).
 */

import { TILE_BORDER } from '../layout.ts';

export const DDGI_BORDER_SOURCE_WGSL = /* wgsl */ `
const DDGI_BORDER_WIDTH: i32 = ${TILE_BORDER};

// The interior texel a tile texel copies. Tile texels run 0..interior+1 with
// the border at both ends. Each edge of an octahedral map mirrors about its
// midpoint and all four corners are one direction, so a border texel copies
// the interior texel mirrored across its edge, and a border corner copies the
// diagonally opposite interior corner (octahedral.ts has the same on the CPU).
fn ddgiBorderSource(tileTexel: vec2u, interior: u32) -> vec2u {
    let last = i32(interior) - 1;
    let inside = vec2i(tileTexel) - DDGI_BORDER_WIDTH;
    let low = inside < vec2i(0);
    let high = inside > vec2i(last);
    let onBorder = low | high;
    let clamped = vec2u(clamp(inside, vec2i(0), vec2i(last)));
    if (all(onBorder)) {
        return select(vec2u(0u), vec2u(u32(last)), low);
    }
    // Top and bottom rows mirror across the tile's vertical centre line.
    if (onBorder.y) {
        return vec2u(u32(last) - clamped.x, clamped.y);
    }
    // Left and right columns mirror across the horizontal centre line.
    if (onBorder.x) {
        return vec2u(clamped.x, u32(last) - clamped.y);
    }
    return clamped;
}
`;
