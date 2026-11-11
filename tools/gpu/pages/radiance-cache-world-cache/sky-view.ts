/**
 * How much of a constant sky a point of the corner sees (RC6.1's sky check):
 * the irradiance a constant sky of radiance L gives a surface is
 * π L (1 - F), where F is the view factor of whatever blocks the sky. In the
 * corner the floor's points are blocked only by the wall, and the wall's only
 * by the part of the floor in front of it.
 *
 * F of a polygon from a point with unit normal n is Lambert's contour
 * formula, exact for a polygon on the normal's side of the tangent plane
 * (touching it is fine):
 *
 *     F = (1 / 2π) |Σ_i θ_i (n · ĝ_i)|
 *
 * over the polygon's edges (v_i, v_i+1), with r_i = v_i - p, θ_i the angle
 * between r_i and r_i+1, and ĝ_i the unit normal of the plane through p and
 * the edge, r_i × r_i+1 normalised (Howell, "A Catalog of Radiation Heat
 * Transfer Configuration Factors", the contour-integral form).
 */
import type { Vec3 } from '../../../../packages/ddgi-core/src/platform/contract/math.ts';
import { FLOOR_HALF_SIDE, WALL } from './corner.ts';

function subtract(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function length(a: Vec3): number {
    return Math.sqrt(dot(a, a));
}

/** The view factor of `polygon` (its corners in order) from `point` with unit `normal`. */
export function polygonViewFactor(point: Vec3, normal: Vec3, polygon: readonly Vec3[]): number {
    let sum = 0;
    for (let index = 0; index < polygon.length; index++) {
        const from = subtract(polygon[index], point);
        const to = subtract(polygon[(index + 1) % polygon.length], point);
        const plane = cross(from, to);
        const planeLength = length(plane);
        // An edge whose line passes through the point spans no angle seen from it.
        if (!(planeLength > 0)) continue;
        const cosine = Math.min(1, Math.max(-1, dot(from, to) / (length(from) * length(to))));
        sum += Math.acos(cosine) * (dot(normal, plane) / planeLength);
    }
    return Math.abs(sum) / (2 * Math.PI);
}

/** The wall (z = 0, x from -20 to 20, y from 0 to 10). */
const WALL_POLYGON: readonly Vec3[] = [
    [WALL.minX, 0, 0],
    [WALL.maxX, 0, 0],
    [WALL.maxX, WALL.height, 0],
    [WALL.minX, WALL.height, 0],
];

/** The floor in front of the wall (z from 0 to 200): the part a point of the wall can see. */
const FLOOR_IN_FRONT_POLYGON: readonly Vec3[] = [
    [-FLOOR_HALF_SIDE, 0, 0],
    [-FLOOR_HALF_SIDE, 0, FLOOR_HALF_SIDE],
    [FLOOR_HALF_SIDE, 0, FLOOR_HALF_SIDE],
    [FLOOR_HALF_SIDE, 0, 0],
];

/**
 * The share of a constant sky's irradiance that reaches a point of the
 * corner's floor (normal +y, in front of the wall) or wall (normal +z): 1 - F
 * of the one surface that blocks it.
 */
export function cornerSkyVisibility(point: Vec3, surface: 'floor' | 'wall'): number {
    if (surface === 'floor') return 1 - polygonViewFactor(point, [0, 1, 0], WALL_POLYGON);
    return 1 - polygonViewFactor(point, [0, 0, 1], FLOOR_IN_FRONT_POLYGON);
}
