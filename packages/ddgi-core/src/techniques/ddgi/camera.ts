/**
 * The camera as the GPU scheduler sees it (docs/design/m2-lights.md §4.3):
 * the view's six planes, for "is this probe in view", and the camera weight
 * `1 / (1 + (d / (4 × spacing))²)`, times 0.25 outside the view. The WGSL
 * twin is `ddgiCameraWeight` in wgsl/common.ts; these CPU versions feed the
 * uniforms and the unit tests.
 */

import type { FrameCamera } from '../../platform/contract/frame.ts';
import { CAMERA_FALLOFF_SPACINGS, OUT_OF_VIEW_WEIGHT, type ProbeGrid } from './settings.ts';

/** A plane `[x, y, z, w]`: a point p is on its inner side when `x·px + y·py + z·pz + w ≥ 0`. */
export type Plane = [number, number, number, number];

/**
 * The view's planes, left, right, bottom, top, near, far, pointing inwards
 * and normalised (so `dot` gives metres), from `projection × view`
 * (Gribb and Hartmann 2001: each plane is the clip matrix's last row plus
 * or minus one of the others).
 *
 * Near is taken as `z ≥ −w`, the loosest of the depth conventions (OpenGL's
 * −1..1, WebGPU's 0..1, reversed depth), so no host's near plane cuts probes
 * the camera sees; points behind the camera fail the side planes anyway.
 * A degenerate plane (all zero) becomes "always inside".
 */
export function frustumPlanes(camera: Pick<FrameCamera, 'view' | 'projection'>): Plane[] {
    const clip = multiply(camera.projection, camera.view);
    // Row i of a column-major matrix: elements i, 4 + i, 8 + i, 12 + i.
    const row = (index: number): Plane => [clip[index], clip[4 + index], clip[8 + index], clip[12 + index]];
    const [x, y, z, w] = [row(0), row(1), row(2), row(3)];
    const add = (a: Plane, b: Plane, sign: number): Plane => [a[0] + sign * b[0], a[1] + sign * b[1], a[2] + sign * b[2], a[3] + sign * b[3]];
    return [add(w, x, 1), add(w, x, -1), add(w, y, 1), add(w, y, -1), add(w, z, 1), add(w, z, -1)].map(normalizedPlane);
}

function normalizedPlane(plane: Plane): Plane {
    const length = Math.hypot(plane[0], plane[1], plane[2]);
    if (!(length > 0) || !Number.isFinite(length)) return [0, 0, 0, 1];
    return [plane[0] / length, plane[1] / length, plane[2] / length, plane[3] / length];
}

/** `a × b` for column-major 4×4 matrices. */
function multiply(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
    const result = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let rowIndex = 0; rowIndex < 4; rowIndex++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) sum += a[k * 4 + rowIndex] * b[column * 4 + k];
            result[column * 4 + rowIndex] = sum;
        }
    }
    return result;
}

/**
 * How far outside the planes a probe still counts as in view: a cell's
 * diagonal, because a probe lights the points up to one cell away from it
 * (its 8 cells of trilinear interpolation).
 */
export function inViewMargin(grid: ProbeGrid): number {
    return Math.hypot(...grid.spacing);
}

/** The camera weight's falloff distance: 4 spacings (the mean of the three axes). */
export function cameraFalloffDistance(grid: ProbeGrid): number {
    return (CAMERA_FALLOFF_SPACINGS * (grid.spacing[0] + grid.spacing[1] + grid.spacing[2])) / 3;
}

/** The camera weight of a point (the CPU twin of `ddgiCameraWeight`). */
export function cameraWeight(point: readonly [number, number, number], camera: Pick<FrameCamera, 'position'>, planes: readonly Plane[], grid: ProbeGrid): number {
    const distance = Math.hypot(point[0] - camera.position[0], point[1] - camera.position[1], point[2] - camera.position[2]);
    const scaled = distance / cameraFalloffDistance(grid);
    const weight = 1 / (1 + scaled * scaled);
    const margin = inViewMargin(grid);
    const inView = planes.every((plane) => plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3] >= -margin);
    return inView ? weight : weight * OUT_OF_VIEW_WEIGHT;
}
