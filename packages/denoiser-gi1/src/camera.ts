/**
 * The camera matrices the kernels rebuild positions with, worked out on the
 * CPU in f64.
 *
 * Capsaicin rebuilds world positions with `transformPointProjection(uv,
 * depth, inverse(view_projection))` for this frame and the previous one
 * (gi1.comp:4023, 4049). In f32 that matrix's translation cancels against a
 * far point's coordinates when the camera is far from the origin, so both
 * inverses here are taken **relative to this frame's eye**: `inverse(P × V ×
 * translate(eye))`. Points come out as offsets from the eye, distances
 * between them are the same, and the large numbers stay on the CPU.
 */

import { DEFAULT_CELL_PIXELS } from './constants.ts';
import type { Mat4 } from './types.ts';

export type Vec3 = [number, number, number];

/** Column-major 4 × 4 product `a × b`. */
export function multiplyMat4(a: Mat4, b: Mat4): number[] {
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[column * 4 + k];
            out[column * 4 + row] = sum;
        }
    }
    return out;
}

/** The inverse of a column-major 4 × 4 matrix (cofactors, f64), or null when it has none. */
export function invertMat4(m: Mat4): number[] | null {
    const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
    const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
    const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
    const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
    // 2 × 2 sub-determinants of the top two and the bottom two rows.
    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;
    const determinant = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!Number.isFinite(determinant) || determinant === 0) return null;
    const inverse = 1 / determinant;
    return [
        (a11 * b11 - a12 * b10 + a13 * b09) * inverse,
        (a02 * b10 - a01 * b11 - a03 * b09) * inverse,
        (a31 * b05 - a32 * b04 + a33 * b03) * inverse,
        (a22 * b04 - a21 * b05 - a23 * b03) * inverse,
        (a12 * b08 - a10 * b11 - a13 * b07) * inverse,
        (a00 * b11 - a02 * b08 + a03 * b07) * inverse,
        (a32 * b02 - a30 * b05 - a33 * b01) * inverse,
        (a20 * b05 - a22 * b02 + a23 * b01) * inverse,
        (a10 * b10 - a11 * b08 + a13 * b06) * inverse,
        (a01 * b08 - a00 * b10 - a03 * b06) * inverse,
        (a30 * b04 - a31 * b02 + a33 * b00) * inverse,
        (a21 * b02 - a20 * b04 - a23 * b00) * inverse,
        (a11 * b07 - a10 * b09 - a12 * b06) * inverse,
        (a00 * b09 - a01 * b07 + a02 * b06) * inverse,
        (a31 * b01 - a30 * b03 - a32 * b00) * inverse,
        (a20 * b03 - a21 * b01 + a22 * b00) * inverse,
    ];
}

/** The camera position: the translation of the view matrix's inverse. */
export function eyeFromView(view: Mat4): Vec3 {
    const inverse = invertMat4(view);
    if (!inverse) throw new Error('@pmndrs/denoiser-gi1: the view matrix has no inverse');
    return [inverse[12], inverse[13], inverse[14]];
}

/**
 * A jitter in pixels (x right, y down) as the NDC offset it adds to every
 * projected point (x right, y up): `(2x / width, −2y / height)`.
 */
export function jitterToNdc(jitter: readonly [number, number] | undefined, width: number, height: number): [number, number] {
    const [x, y] = jitter ?? [0, 0];
    return [(2 * x) / width, (-2 * y) / height];
}

/**
 * The projection the depth was drawn with: `translate(jitter, 0) ×
 * projection`, which adds `jitter × w` to clip x and y, so each column gains
 * `jitter ×` its w row.
 */
export function jitteredProjection(projection: Mat4, ndcJitter: readonly [number, number]): number[] {
    const out = Array.from(projection);
    const [jitterX, jitterY] = ndcJitter;
    if (jitterX === 0 && jitterY === 0) return out;
    for (let column = 0; column < 4; column++) {
        const w = out[column * 4 + 3];
        out[column * 4 + 0] += jitterX * w;
        out[column * 4 + 1] += jitterY * w;
    }
    return out;
}

/**
 * `inverse(projection × view × translate(origin))`: with it, `(M × (ndc,
 * 1)).xyz / w` is a world point minus `origin`. Throws when projection × view
 * can't be inverted.
 */
export function relativeInverseViewProjection(projection: Mat4, view: Mat4, origin: readonly [number, number, number]): number[] {
    const [x, y, z] = origin;
    const translate = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
    const inverse = invertMat4(multiplyMat4(multiplyMat4(projection, view), translate));
    if (!inverse) throw new Error('@pmndrs/denoiser-gi1: the camera needs a projection × view that can be inverted');
    return inverse;
}

/**
 * The camera's forward axis in world space, unit length: the view matrix maps
 * it to −z, so it is minus the view rotation's third row (elements 2, 6, 10).
 * Linear depth is measured along it (FilterGI's `toLinearDepth`, gi1.comp:4120).
 */
export function viewForward(view: Mat4): Vec3 {
    const x = -view[2];
    const y = -view[6];
    const z = -view[10];
    const length = Math.hypot(x, y, z);
    if (!(length > 0)) throw new Error('@pmndrs/denoiser-gi1: the view matrix has no rotation');
    return [x / length, y / length, z / length];
}

/**
 * Capsaicin's `cell_size` (gi1.cpp:1839-1843): `tan(fovY × probe_size ×
 * max(1 / h, h / w²))`, about the size per unit distance of `cellPixels`
 * pixels. The vertical field of view comes from the projection
 * (`projection[5] = 1 / tan(fovY / 2)`).
 */
export function reprojectionCellSize(projection: Mat4, width: number, height: number, cellPixels = DEFAULT_CELL_PIXELS): number {
    const fovY = 2 * Math.atan(1 / projection[5]);
    return Math.tan(fovY * cellPixels * Math.max(1 / height, height / (width * width)));
}

/** What the kernels' `GIDenoiserConstants` hold for one frame (wgsl.ts), worked out from the host's camera. */
export interface FrameConstants {
    /** `inverse(P × V × translate(eye))` with this frame's jitter: uv and depth to a point relative to the eye. */
    readonly inverseViewProjection: number[];
    /** The previous frame's, with its jitter, relative to this frame's eye. */
    readonly previousInverseViewProjection: number[];
    readonly viewForward: Vec3;
    /** Capsaicin's `cell_size`. */
    readonly cellSize: number;
    readonly reversedDepth: boolean;
}

/**
 * The frame's constants. `previousJitter` is the previous frame's jitter in
 * pixels (the denoiser remembers it; `FrameCamera` carries only this frame's).
 */
export function frameConstants(
    camera: {
        readonly view: Mat4;
        readonly projection: Mat4;
        readonly prevView: Mat4;
        readonly prevProjection: Mat4;
        readonly jitter?: readonly [number, number];
        readonly position?: readonly [number, number, number];
        readonly reversedDepth?: boolean;
    },
    previousJitter: readonly [number, number],
    width: number,
    height: number,
    cellPixels = DEFAULT_CELL_PIXELS,
): FrameConstants {
    const eye = camera.position ?? eyeFromView(camera.view);
    // Both frames' depths were drawn with their own jitter (Capsaicin's camera matrices are
    // jittered when TAA is on, gi1.cpp:1816-1820).
    const projection = jitteredProjection(camera.projection, jitterToNdc(camera.jitter, width, height));
    const previousProjection = jitteredProjection(camera.prevProjection, jitterToNdc(previousJitter, width, height));
    return {
        inverseViewProjection: relativeInverseViewProjection(projection, camera.view, eye),
        previousInverseViewProjection: relativeInverseViewProjection(previousProjection, camera.prevView, eye),
        viewForward: viewForward(camera.view),
        cellSize: reprojectionCellSize(camera.projection, width, height, cellPixels),
        reversedDepth: camera.reversedDepth === true,
    };
}
