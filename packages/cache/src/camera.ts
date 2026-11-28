/**
 * The camera matrix the screen-probe kernels reconstruct positions with
 * (`RcFrame.inverseViewProjection`): Capsaicin's `g_ViewProjectionInverse`
 * (gi1.comp:97, gi1.cpp:1819), made camera-relative.
 *
 * `transformPointProjection(uv, depth, inverse(P × V))` gives a world point,
 * but in f32 the matrix's translation part cancels against a far point's
 * coordinates when the camera is far from the origin. So the matrix is
 * `inverse(P × V × T(eye))`, worked out here in f64, and the kernels add
 * `eye` after the divide: the same point, with the large numbers kept apart.
 */

import type { FrameCamera } from '@pmndrs/gi-core/platform/contract/frame';
import type { Mat4 } from '@pmndrs/gi-core/platform/contract/math';

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

/** The inverse of a column-major 4 × 4 matrix (cofactors, in f64), or null when it has none. */
export function invertMat4(m: Mat4): number[] | null {
    const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
    const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
    const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
    const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
    // 2 × 2 sub-determinants of the top two and bottom two rows of columns pairs.
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

/**
 * The projection the host drew its depth with: `translate(jitter, 0) ×
 * projection` (`FrameCamera.jitter`, an NDC offset). In clip space that adds
 * `jitter × w` to x and y, so each column gains `jitter × ` its w row.
 */
export function jitteredProjection(camera: Pick<FrameCamera, 'projection' | 'jitter'>): number[] {
    const projection = Array.from(camera.projection);
    const [jitterX, jitterY] = camera.jitter ?? [0, 0];
    if (jitterX === 0 && jitterY === 0) return projection;
    for (let column = 0; column < 4; column++) {
        const w = projection[column * 4 + 3];
        projection[column * 4 + 0] += jitterX * w;
        projection[column * 4 + 1] += jitterY * w;
    }
    return projection;
}

/**
 * `inverse(projection × view × translate(eye))`, column-major, for
 * `world = eye + (M × (ndc, 1)).xyz / w`, with the frame's jitter in the
 * projection (`jitteredProjection`), so a point rebuilt from the host's
 * depth lands where it was drawn. Throws for a camera whose projection ×
 * view can't be inverted (a zero-size view, say).
 */
export function cameraRelativeInverseViewProjection(camera: Pick<FrameCamera, 'position' | 'view' | 'projection' | 'jitter'>): number[] {
    const inverse = invertMat4(cameraRelativeViewProjection(camera));
    if (!inverse) throw new Error('@pmndrs/gi: the radiance cache needs a camera whose projection × view can be inverted');
    return inverse;
}

/**
 * `projection × view × translate(eye)`, column-major, with the frame's jitter:
 * Capsaicin's `g_ViewProjection` (gi1.comp:95) for a point given relative to
 * the eye, `clip = M × (world − eye, 1)` (`CountScreenProbes` projects the
 * cached probes with it, RC4.1).
 */
export function cameraRelativeViewProjection(camera: Pick<FrameCamera, 'position' | 'view' | 'projection' | 'jitter'>): number[] {
    const [x, y, z] = camera.position;
    const translateToEye = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
    return multiplyMat4(multiplyMat4(jitteredProjection(camera), camera.view), translateToEye);
}
