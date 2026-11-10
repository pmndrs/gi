/**
 * A small scene ray-cast on the CPU for the GI denoiser's own checks: the
 * guides (depth, normal, motion) the denoiser reads, computed exactly in f64
 * for a camera and the previous frame's camera, and a world-fixed checker
 * of irradiance on it. Not a page itself.
 *
 * The scene: a floor (y = 0), a back wall (z = −8, up to y = 4) and a pillar
 * (x ±0.4, y 0..2.5, z −3.4..−2.6); everything else is sky (depth 1,
 * normal 0, motion 0). Depth is the hit projected with the camera's own
 * projection × view (WebGPU's 0..1), motion its NDC now minus its NDC
 * through the previous camera (the scene doesn't move).
 */

import type { FrameCamera } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { frameCamera, type PageCamera } from '../radiance-cache-probes/screen.ts';

type Vector = [number, number, number];

const subtract = (a: Vector, b: Vector): Vector => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vector, b: Vector): Vector => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize = (a: Vector): Vector => {
    const length = Math.hypot(...a);
    return [a[0] / length, a[1] / length, a[2] / length];
};

/** Axis-aligned faces: `axis` is the coordinate that is constant (`value`), within `bounds` on the other two. */
interface Face {
    readonly id: number;
    readonly axis: 0 | 1 | 2;
    readonly value: number;
    readonly normal: Vector;
    /** [min, max] of each axis (the face's own axis ignored). */
    readonly bounds: readonly [readonly [number, number], readonly [number, number], readonly [number, number]];
}

const INFINITE: readonly [number, number] = [-1e9, 1e9];
const PILLAR = { x: [-0.4, 0.4], y: [0, 2.5], z: [-3.4, -2.6] } as const;
const FACES: readonly Face[] = [
    { id: 0, axis: 1, value: 0, normal: [0, 1, 0], bounds: [INFINITE, INFINITE, [-8, 1e9]] },
    { id: 1, axis: 2, value: -8, normal: [0, 0, 1], bounds: [INFINITE, [0, 4], INFINITE] },
    { id: 2, axis: 0, value: PILLAR.x[0], normal: [-1, 0, 0], bounds: [INFINITE, PILLAR.y, PILLAR.z] },
    { id: 3, axis: 0, value: PILLAR.x[1], normal: [1, 0, 0], bounds: [INFINITE, PILLAR.y, PILLAR.z] },
    { id: 4, axis: 2, value: PILLAR.z[1], normal: [0, 0, 1], bounds: [PILLAR.x, PILLAR.y, INFINITE] },
    { id: 5, axis: 2, value: PILLAR.z[0], normal: [0, 0, -1], bounds: [PILLAR.x, PILLAR.y, INFINITE] },
    { id: 6, axis: 1, value: PILLAR.y[1], normal: [0, 1, 0], bounds: [PILLAR.x, INFINITE, PILLAR.z] },
];

export interface Hit {
    readonly t: number;
    readonly point: Vector;
    readonly face: Face;
}

/** The closest face the ray hits from its front, or null (sky). */
export function castRay(origin: Vector, direction: Vector): Hit | null {
    let best: Hit | null = null;
    for (const face of FACES) {
        const d = direction[face.axis];
        if (Math.abs(d) < 1e-12) continue;
        const t = (face.value - origin[face.axis]) / d;
        if (!(t > 1e-6) || (best && t >= best.t)) continue;
        // Only the face's front: the ray comes towards its normal.
        if (d * face.normal[face.axis] >= 0) continue;
        const point: Vector = [origin[0] + direction[0] * t, origin[1] + direction[1] * t, origin[2] + direction[2] * t];
        const inside = [0, 1, 2].every((axis) => axis === face.axis || (point[axis] >= face.bounds[axis][0] && point[axis] <= face.bounds[axis][1]));
        if (inside) best = { t, point, face };
    }
    return best;
}

/** The checker cell of a hit: its face and the `size` m squares of its two other coordinates. */
export function checkerCell(hit: Hit, size = 1): { id: string; value: number } {
    const axes = [0, 1, 2].filter((axis) => axis !== hit.face.axis);
    const a = Math.floor(hit.point[axes[0]] / size);
    const b = Math.floor(hit.point[axes[1]] / size);
    return { id: `${hit.face.id}:${a}:${b}`, value: (a + b) % 2 === 0 ? 1.0 : 0.25 };
}

export interface Guides {
    readonly width: number;
    readonly height: number;
    /** WebGPU depth (0..1), 1 on sky. */
    readonly depth: Float32Array;
    /** Unit normals (xyz, 0), 0 on sky. */
    readonly normal: Float32Array;
    /** NDC motion (current − previous) in xy. */
    readonly motion: Float32Array;
    /** The hit per pixel, or null on sky. */
    readonly hits: readonly (Hit | null)[];
}

function multiply(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
        }
    }
    return out;
}

function project(matrix: readonly number[], point: Vector): [number, number, number] {
    const clip = [0, 1, 2, 3].map((row) => matrix[row] * point[0] + matrix[4 + row] * point[1] + matrix[8 + row] * point[2] + matrix[12 + row]);
    return [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
}

/** The largest f32 below 1: a hit never reads as the far plane. */
const DEPTH_BELOW_FAR = 0.99999994;

/** Ray-casts every pixel centre of `camera`, with motion against `previous`. */
export function renderGuides(camera: PageCamera, previous: PageCamera, width: number, height: number): Guides {
    const current = frameCamera(camera, width, height);
    const before = frameCamera(previous, width, height);
    const viewProjection = multiply(current.projection, current.view);
    const previousViewProjection = multiply(before.projection, before.view);
    const forward = normalize(subtract(camera.target, camera.position));
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = cross(right, forward);
    const tanHalfFov = Math.tan((camera.fovYDeg * Math.PI) / 360);
    const aspect = width / height;
    const pixels = width * height;
    const depth = new Float32Array(pixels).fill(1);
    const normal = new Float32Array(pixels * 4);
    const motion = new Float32Array(pixels * 4);
    const hits: (Hit | null)[] = new Array(pixels).fill(null);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel = y * width + x;
            const ndcX = ((x + 0.5) / width) * 2 - 1;
            const ndcY = 1 - ((y + 0.5) / height) * 2;
            const direction = normalize([0, 1, 2].map((axis) => forward[axis] + right[axis] * ndcX * tanHalfFov * aspect + up[axis] * ndcY * tanHalfFov) as Vector);
            const hit = castRay(camera.position, direction);
            if (!hit) continue;
            hits[pixel] = hit;
            const now = project(viewProjection, hit.point);
            const then = project(previousViewProjection, hit.point);
            depth[pixel] = Math.min(now[2], DEPTH_BELOW_FAR);
            normal.set([...hit.face.normal, 0], pixel * 4);
            motion.set([now[0] - then[0], now[1] - then[1], 0, 0], pixel * 4);
        }
    }
    return { width, height, depth, normal, motion, hits };
}

/** The camera as the denoiser takes it (column-major matrices, f64). */
export function denoiserCameraOf(camera: PageCamera, previous: PageCamera, width: number, height: number): {
    view: number[];
    projection: number[];
    prevView: number[];
    prevProjection: number[];
    near: number;
    far: number;
    position: [number, number, number];
} {
    const current: FrameCamera = frameCamera(camera, width, height);
    const before: FrameCamera = frameCamera(previous, width, height);
    return {
        view: Array.from(current.view),
        projection: Array.from(current.projection),
        prevView: Array.from(before.view),
        prevProjection: Array.from(before.projection),
        near: camera.near,
        far: camera.far,
        position: [...camera.position],
    };
}
