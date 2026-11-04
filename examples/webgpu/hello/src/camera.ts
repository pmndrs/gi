/**
 * Camera matrices for the hello renderer: column-major 4×4 (WGSL's
 * `mat4x4f` layout), right-handed, Y up, looking down −Z, with WebGPU's
 * 0..1 clip depth.
 */

export type Vec3 = readonly [number, number, number];
/** 16 numbers, column-major. */
export type Mat4 = Float32Array;

export interface CameraPose {
    readonly position: Vec3;
    readonly target: Vec3;
    /** Vertical field of view, degrees. */
    readonly fovYDeg: number;
}

export interface CameraMatrices {
    readonly position: Vec3;
    readonly view: Mat4;
    readonly projection: Mat4;
    readonly viewProjection: Mat4;
    readonly near: number;
    readonly far: number;
}

export const WORLD_UP: Vec3 = [0, 1, 0];

export function subtract(a: Vec3, b: Vec3): [number, number, number] {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function cross(a: Vec3, b: Vec3): [number, number, number] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function dot(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function normalize(a: Vec3): [number, number, number] {
    const length = Math.hypot(a[0], a[1], a[2]);
    return [a[0] / length, a[1] / length, a[2] / length];
}

/**
 * World-to-view: the camera at `eye` looking at `target`. The basis is the
 * same one the engine-free test view builds its rays from (forward, right =
 * forward × up, up = right × forward), so both see the same pixels.
 */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3 = WORLD_UP): Mat4 {
    const forward = normalize(subtract(target, eye));
    const right = normalize(cross(forward, up));
    const trueUp = cross(right, forward);
    // The rotation's rows are the camera axes (right, up, back), so view space
    // has x right, y up and the camera looking down -z.
    // prettier-ignore
    return new Float32Array([
        right[0], trueUp[0], -forward[0], 0,
        right[1], trueUp[1], -forward[1], 0,
        right[2], trueUp[2], -forward[2], 0,
        -dot(right, eye), -dot(trueUp, eye), dot(forward, eye), 1,
    ]);
}

/** Perspective projection to WebGPU clip space (depth 0 at `near`, 1 at `far`). */
export function perspective(fovYRadians: number, aspect: number, near: number, far: number): Mat4 {
    const focal = 1 / Math.tan(fovYRadians / 2);
    const depthScale = far / (near - far);
    // prettier-ignore
    return new Float32Array([
        focal / aspect, 0, 0, 0,
        0, focal, 0, 0,
        0, 0, depthScale, -1,
        0, 0, near * depthScale, 0,
    ]);
}

export function multiply(a: Mat4, b: Mat4): Mat4 {
    const out = new Float32Array(16);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            let sum = 0;
            for (let k = 0; k < 4; k++) sum += a[k * 4 + row] * b[column * 4 + k];
            out[column * 4 + row] = sum;
        }
    }
    return out;
}

/** Near and far planes: the procedural scenes are a few metres across. */
export const CAMERA_NEAR = 0.05;
export const CAMERA_FAR = 200;

export function cameraMatrices(pose: CameraPose, aspect: number): CameraMatrices {
    const view = lookAt(pose.position, pose.target);
    const projection = perspective((pose.fovYDeg * Math.PI) / 180, aspect, CAMERA_NEAR, CAMERA_FAR);
    return { position: pose.position, view, projection, viewProjection: multiply(projection, view), near: CAMERA_NEAR, far: CAMERA_FAR };
}

/**
 * The upper 3×3 of `inverse(transpose(model))`, as three vec4 columns (WGSL's
 * `mat3x3f` layout), so normals stay perpendicular under non-uniform scale.
 * Also returns whether the transform mirrors (negative determinant), which
 * flips triangle winding on screen.
 */
export function normalMatrix(model: readonly number[]): { columns: Float32Array; mirrored: boolean } {
    const m = (column: number, row: number): number => model[column * 4 + row];
    const a00 = m(0, 0), a01 = m(1, 0), a02 = m(2, 0);
    const a10 = m(0, 1), a11 = m(1, 1), a12 = m(2, 1);
    const a20 = m(0, 2), a21 = m(1, 2), a22 = m(2, 2);
    // Cofactors of the 3x3 (row, column), which are inverse(transpose(A)) times the determinant.
    const c00 = a11 * a22 - a12 * a21, c01 = -(a10 * a22 - a12 * a20), c02 = a10 * a21 - a11 * a20;
    const c10 = -(a01 * a22 - a02 * a21), c11 = a00 * a22 - a02 * a20, c12 = -(a00 * a21 - a01 * a20);
    const c20 = a01 * a12 - a02 * a11, c21 = -(a00 * a12 - a02 * a10), c22 = a00 * a11 - a01 * a10;
    const determinant = a00 * c00 + a01 * c01 + a02 * c02;
    const scale = 1 / determinant;
    // inverse(transpose(A))[row][column] = cofactor[row][column] / det; stored by columns.
    // prettier-ignore
    const columns = new Float32Array([
        c00 * scale, c10 * scale, c20 * scale, 0,
        c01 * scale, c11 * scale, c21 * scale, 0,
        c02 * scale, c12 * scale, c22 * scale, 0,
    ]);
    return { columns, mirrored: determinant < 0 };
}
