/**
 * The one equirectangular layout every consumer of the sky file uses.
 *
 * World space is the manifest's (three.js's): right-handed, +Y up. A direction
 * d maps to the image like three.js's `equirectUV`:
 *
 *   u = atan2(d.z, d.x) / (2 pi) + 0.5        0 and 1 are -X, 0.25 is -Z, 0.5 is +X, 0.75 is +Z
 *   elevation = asin(d.y)                      +90 degrees (zenith) is the top row of the file
 *
 * The file stores rows top to bottom (row 0 is the zenith, the last row the
 * nadir), as OpenEXR does. So in texture coordinates where v = 0 is the first
 * row in memory (WebGPU, and three.js after `flipY`), v = acos(d.y) / pi.
 *
 * Blender's equirectangular environment texture uses the same layout once
 * directions go through the manifest-to-Blender rotation (x, y, z) -> (x, -z, y);
 * `tools/reference/cycles` checks that by rendering known directions.
 */

export type Direction = readonly [number, number, number];

/** Texture coordinates in the file's own order: u across, v down from the top row (the zenith). */
export interface EquirectUv {
    u: number;
    /** 0 at the zenith (first row), 1 at the nadir (last row). */
    v: number;
}

/** Direction (need not be normalised) to file texture coordinates. */
export function directionToEquirectUv(direction: Direction): EquirectUv {
    const length = Math.hypot(direction[0], direction[1], direction[2]);
    const y = Math.min(1, Math.max(-1, direction[1] / length));
    return {
        u: Math.atan2(direction[2], direction[0]) / (2 * Math.PI) + 0.5,
        v: Math.acos(y) / Math.PI,
    };
}

/** File texture coordinates to a unit direction; the inverse of {@link directionToEquirectUv}. */
export function equirectUvToDirection(uv: EquirectUv): [number, number, number] {
    const azimuth = (uv.u - 0.5) * 2 * Math.PI;
    const polar = uv.v * Math.PI;
    const horizontal = Math.sin(polar);
    return [horizontal * Math.cos(azimuth), Math.cos(polar), horizontal * Math.sin(azimuth)];
}

/** The direction through the centre of texel (column, row), row 0 at the top. */
export function texelCentreDirection(column: number, row: number, width: number, height: number): [number, number, number] {
    return equirectUvToDirection({ u: (column + 0.5) / width, v: (row + 0.5) / height });
}

/**
 * Exact integrals over one row of texels (every texel in a row covers the
 * same band of polar angle theta, from row/height * pi to (row+1)/height * pi):
 *  - solid angle of one texel: (2 pi / width) (cos theta0 - cos theta1);
 *  - its weight for irradiance on a horizontal, upward-facing surface,
 *    the integral of cos(theta) d(omega): (2 pi / width) (cos^2 theta0 - cos^2 theta1) / 2.
 * The second is only meaningful above the horizon; it is 0 for rows below it.
 * Treating each texel as constant over its area, the sum of radiance times
 * this weight is the horizontal irradiance the file delivers.
 */
export function rowSolidAngle(row: number, width: number, height: number): number {
    const theta0 = (row / height) * Math.PI;
    const theta1 = ((row + 1) / height) * Math.PI;
    return ((2 * Math.PI) / width) * (Math.cos(theta0) - Math.cos(theta1));
}

export function rowHorizontalIrradianceWeight(row: number, width: number, height: number): number {
    const theta0 = (row / height) * Math.PI;
    const theta1 = Math.min(((row + 1) / height) * Math.PI, Math.PI / 2);
    if (theta1 <= theta0) return 0;
    const cos0 = Math.cos(theta0);
    const cos1 = Math.cos(theta1);
    return ((2 * Math.PI) / width) * ((cos0 * cos0 - cos1 * cos1) / 2);
}

/**
 * Bilinear lookup in an equirect RGB image (row 0 at the top), with texel
 * centres at (i + 0.5) / size, wrapping across the -X seam and clamping at the
 * poles: what a GPU sampler with linear filtering and `repeat` on u returns.
 * Used to check renderers against the file.
 */
export function sampleEquirectBilinear(rgb: ArrayLike<number>, width: number, height: number, direction: Direction): [number, number, number] {
    const { u, v } = directionToEquirectUv(direction);
    const x = u * width - 0.5;
    const y = v * height - 0.5;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const result: [number, number, number] = [0, 0, 0];
    const corners: [number, number, number][] = [
        [0, 0, (1 - fx) * (1 - fy)],
        [1, 0, fx * (1 - fy)],
        [0, 1, (1 - fx) * fy],
        [1, 1, fx * fy],
    ];
    for (const [dx, dy, weight] of corners) {
        const column = (((x0 + dx) % width) + width) % width;
        const row = Math.min(height - 1, Math.max(0, y0 + dy));
        const base = (row * width + column) * 3;
        for (let channel = 0; channel < 3; channel++) result[channel] += weight * rgb[base + channel];
    }
    return result;
}
