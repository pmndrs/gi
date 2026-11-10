/**
 * A CPU twin of Capsaicin's SSGI kernel (ssgi.comp:132-252; the WGSL is
 * wgsl/ssgi.ts), in f64, for the GPU checks and unit tests (RC4.2): the
 * same random numbers, the same samples and the same integrals, from the
 * screen the GPU read.
 *
 * The kernel's results move a lot with a few of its f32 values: a sample a
 * few pixels away gives a horizon direction from the difference of two
 * nearly equal positions (Capsaicin's own "BE CAREFUL: main source of
 * precision errors", ssgi.comp:199), the horizon truncation intersects from
 * the world's origin with its denominator held at 1e-5 (up to 1e5 times an
 * input's error), and some comparisons pick a branch. So
 * {@link ssgiTwinPixel} marks a pixel **sensitive** where a decision sits
 * within a small margin of its threshold, or where the twin itself moves by
 * more than the comparison's tolerance when its positions are rounded to
 * f32 as the GPU holds them; the GPU checks compare the other pixels.
 */

import { makeRandom, rand2, type Direction } from './capsaicin-math.ts';
import type { SSGIConstants } from './ssgi.ts';

type Vector = [number, number, number];

/** Each component of normalize(2 × 0 − 1): Capsaicin's decode of a sky pixel's cleared normal (register D58). */
const SKY_NORMAL = -1 / Math.sqrt(3);

/** When the twin calls a pixel sensitive (see the file comment). */
export const SSGI_SENSITIVITY = {
    /** `sample_cos >= prev_horizon_cos` and the facing test `step(0, ...)`: cosines this close to their threshold. */
    cosine: 1e-4,
    /** A sample's uv this close (in texels) to a texel edge, where the point sampler's texel may differ. */
    texel: 1e-3,
    /** `sign(dot(ortho_world_dir, proj_world_normal))` this close to 0. */
    sign: 1e-6,
    /** `RotFromToMatrix`'s identity test (`f > 1 - 0.0003`) this close. */
    rotation: 1e-6,
} as const;

/** The screen SSGI reads, as the GPU had it. */
export interface SSGIScreen {
    readonly width: number;
    readonly height: number;
    /** One per pixel. */
    readonly depth: Float32Array;
    /** Four per pixel, signed (rgb used). */
    readonly normal: Float32Array;
    /** Last frame's lit colour, four per texel, `lightingWidth` × `lightingHeight`. */
    readonly lighting: Float32Array;
    readonly lightingWidth: number;
    readonly lightingHeight: number;
    /** The far plane's depth: 1, or 0 with reversed depth. */
    readonly farDepth: number;
    /** `RcFrame`'s camera-relative inverse view-projection (`cameraRelativeInverseViewProjection`) and eye. */
    readonly inverseViewProjection: readonly number[];
    readonly eye: Direction;
    readonly frameIndex: number;
}

export interface SSGIPixel {
    readonly ambientOcclusion: number;
    readonly bentNormal: Vector;
    readonly nearField: Vector;
    /** A decision within its margin, or (from {@link ssgiTwinPixel}) a result f32 positions move (file comment). */
    readonly sensitive: boolean;
    /** Samples skipped by the zero-length guard (register D58). */
    readonly zeroLengthSamples: number;
}

const dot = (a: Direction, b: Direction): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const subtract = (a: Direction, b: Direction): Vector => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Direction, s: number): Vector => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: Direction, b: Direction): Vector => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (a: Direction): number => Math.hypot(a[0], a[1], a[2]);
const normalize = (a: Direction): Vector => scale(a, 1 / length(a));
const saturate = (value: number): number => Math.min(Math.max(value, 0), 1);
const clamp = (value: number, low: number, high: number): number => Math.min(Math.max(value, low), high);

/** A direction through a column-major 4 × 4 matrix (w = 0). */
function transformDirection(matrix: readonly number[], v: Direction): Vector {
    return [0, 1, 2].map((row) => matrix[row] * v[0] + matrix[4 + row] * v[1] + matrix[8 + row] * v[2]) as Vector;
}

/** `IntersectDirPlaneOneSided` (ssgi.comp:72-77). */
export function intersectDirPlaneOneSided(dir: Direction, normal: Direction, pt: Direction): number {
    const d = -dot(pt, normal);
    return d / Math.max(1e-5, -dot(dir, normal));
}

/** `RotFromToMatrix` (ssgi.comp:81-111) applied to `v`: the rotation taking `from` onto `to`. */
export function rotateFromTo(from: Direction, to: Direction, v: Direction): Vector {
    const e = dot(from, to);
    const f = Math.abs(e);
    if (f > 1 - 0.0003) return [v[0], v[1], v[2]];
    const c = cross(from, to);
    const h = 1 / (1 + e);
    const hvx = h * c[0];
    const hvz = h * c[2];
    const hvxy = hvx * c[1];
    const hvxz = hvx * c[2];
    const hvyz = hvz * c[1];
    const rows: Vector[] = [
        [e + hvx * c[0], hvxy - c[2], hvxz + c[1]],
        [hvxy + c[2], e + h * c[1] * c[1], hvyz - c[0]],
        [hvxz - c[1], hvyz + c[0], e + hvz * c[2]],
    ];
    return [dot(rows[0], v), dot(rows[1], v), dot(rows[2], v)];
}

/** `IntegrateHalfArc` (ssgi.comp:113-116). */
export function integrateHalfArc(horizonAngle: number, normalAngle: number): number {
    return (Math.cos(normalAngle) + 2 * horizonAngle * Math.sin(normalAngle) - Math.cos(2 * horizonAngle - normalAngle)) / 4;
}

/** `IntegrateBentNormal` (ssgi.comp:118-130). */
export function integrateBentNormal(h0: number, h1: number, n: number, worldEyeDir: Direction, sliceViewDir: Direction, constants: Pick<SSGIConstants, 'inverseView' | 'forward'>): Vector {
    const t0 =
        (6 * Math.sin(h0 - n) - Math.sin(3 * h0 - n) + 6 * Math.sin(h1 - n) - Math.sin(3 * h1 - n) + 16 * Math.sin(n) - 3 * (Math.sin(h0 + n) + Math.sin(h1 + n))) / 12;
    const t1 = (-Math.cos(3 * h0 - n) - Math.cos(3 * h1 - n) + 8 * Math.cos(n) - 3 * (Math.cos(h0 + n) + Math.cos(h1 + n))) / 12;
    const worldBentNormal = transformDirection(constants.inverseView, [sliceViewDir[0] * t0, sliceViewDir[1] * t0, t1]);
    return rotateFromTo(scale(constants.forward, -1), worldEyeDir, worldBentNormal);
}

class Screen {
    constructor(private readonly screen: SSGIScreen) {}

    isSky(x: number, y: number): boolean {
        return this.screen.depth[x + y * this.screen.width] === this.screen.farDepth;
    }

    normalAt(x: number, y: number): Vector {
        const index = 4 * (x + y * this.screen.width);
        const n = this.screen.normal;
        return normalize([n[index], n[index + 1], n[index + 2]]);
    }

    /**
     * `rcTransformPointProjection(uv, depth)`: in f64, or with the
     * camera-relative point and the world point rounded to f32 as the GPU
     * holds them (`f32Positions`).
     */
    pointAt(u: number, v: number, depth: number, f32Positions: boolean): Vector {
        const m = this.screen.inverseViewProjection;
        const ndc = [2 * u - 1, 2 * (1 - v) - 1, depth, 1];
        const h = [0, 1, 2, 3].map((row) => m[row] * ndc[0] + m[4 + row] * ndc[1] + m[8 + row] * ndc[2] + m[12 + row] * ndc[3]);
        const eye = this.screen.eye;
        const round = f32Positions ? Math.fround : (value: number) => value;
        return [0, 1, 2].map((axis) => round(eye[axis] + round(h[axis] / h[3]))) as Vector;
    }

    /** The point sampler's texel (clamped) and how near uv is to a texel edge, in texels. */
    texel(u: number, v: number, width: number, height: number): { x: number; y: number; edge: number } {
        const fx = u * width;
        const fy = v * height;
        const x = clamp(Math.floor(fx), 0, width - 1);
        const y = clamp(Math.floor(fy), 0, height - 1);
        const edgeX = fx > 0 && fx < width ? Math.abs(fx - Math.round(fx)) : Infinity;
        const edgeY = fy > 0 && fy < height ? Math.abs(fy - Math.round(fy)) : Infinity;
        return { x, y, edge: Math.min(edgeX, edgeY) };
    }

    lightingAt(u: number, v: number): Vector {
        const { lightingWidth, lightingHeight, lighting } = this.screen;
        const { x, y } = this.texel(u, v, lightingWidth, lightingHeight);
        const index = 4 * (x + y * lightingWidth);
        return [lighting[index], lighting[index + 1], lighting[index + 2]];
    }
}

/**
 * The SSGI kernel at one pixel, or null for a sky pixel (it writes zeros
 * there). `sensitive` covers the decisions only; {@link ssgiTwinPixel} adds
 * the f32 test.
 */
export function ssgiPixel(screen: SSGIScreen, constants: SSGIConstants, x: number, y: number, f32Positions = false): SSGIPixel | null {
    const view = new Screen(screen);
    if (view.isSky(x, y)) return null;
    const { width, height } = screen;
    const limits = SSGI_SENSITIVITY;
    let sensitive = false;
    let zeroLengthSamples = 0;

    const u = (x + 0.5) / width;
    const v = (y + 0.5) / height;
    const worldPos = view.pointAt(u, v, screen.depth[x + y * width], f32Positions);
    const linearDepth = dot(subtract(worldPos, screen.eye), constants.forward);
    const worldNormal = view.normalAt(x, y);
    const worldEyeDir = normalize(subtract(screen.eye, worldPos));
    const rotationTest = Math.abs(Math.abs(dot(scale(constants.forward, -1), worldEyeDir)) - (1 - 0.0003));
    if (rotationTest < limits.rotation) sensitive = true;

    const random = makeRandom(x + y * width + width * height, screen.frameIndex);
    const noise = rand2(random);

    const { sliceCount, stepCount } = constants;
    const sliceUvRadius = constants.uvRadius / linearDepth;
    let ambientOcclusion = 0;
    let bentNormal: Vector = [0, 0, 0];
    let globalLighting: Vector = [0, 0, 0];
    for (let sliceIndex = 0; sliceIndex < sliceCount; sliceIndex++) {
        const sliceAngle = ((sliceIndex + noise[0]) / sliceCount) * Math.PI;
        const sliceCos = Math.cos(sliceAngle);
        const sliceSin = Math.sin(sliceAngle);
        const sliceUvDir = [sliceCos * sliceUvRadius, -sliceSin * sliceUvRadius];
        const sliceViewDir: Vector = [sliceCos, sliceSin, 0];
        const sliceWorldDir = transformDirection(constants.inverseView, sliceViewDir);
        const orthoWorldDir = subtract(sliceWorldDir, scale(worldEyeDir, dot(sliceWorldDir, worldEyeDir)));
        const projAxisDir = normalize(cross(orthoWorldDir, worldEyeDir));
        const projWorldNormal = subtract(worldNormal, scale(projAxisDir, dot(worldNormal, projAxisDir)));
        const projWorldNormalLen = length(projWorldNormal);
        const projWorldNormalCos = saturate(dot(projWorldNormal, worldEyeDir) / projWorldNormalLen);
        const signTest = dot(orthoWorldDir, projWorldNormal);
        if (Math.abs(signTest) < limits.sign) sensitive = true;
        const projWorldNormalAngle = Math.sign(signTest) * Math.acos(projWorldNormalCos);

        const horizonAngles = [0, 0];
        for (let sideIndex = 0; sideIndex < 2; sideIndex++) {
            const sideSign = sideIndex === 0 ? 1 : -1;
            const horizonMin = Math.cos(projWorldNormalAngle + sideSign * (Math.PI / 2));
            let horizonCos = horizonMin;
            let prevSampleWorldPos: Vector = [0, 0, 0];
            for (let stepIndex = 0; stepIndex < stepCount; stepIndex++) {
                const sampleStep = (stepIndex + noise[1]) / stepCount;
                const sampleU = u + sideSign * sampleStep * sliceUvDir[0];
                const sampleV = v + sideSign * sampleStep * sliceUvDir[1];
                const texel = view.texel(sampleU, sampleV, width, height);
                if (texel.edge < limits.texel) sensitive = true;
                const sampleWorldPos = view.pointAt(sampleU, sampleV, screen.depth[texel.x + texel.y * width], f32Positions);
                const offset = subtract(sampleWorldPos, worldPos);
                const horizonWorldLen = length(offset);
                if (horizonWorldLen > 0) {
                    const horizonWorldDir = scale(offset, 1 / horizonWorldLen);
                    const sampleWeight = saturate(horizonWorldLen * constants.falloffMul + constants.falloffAdd);
                    const sampleCos = horizonMin + sampleWeight * (dot(horizonWorldDir, worldEyeDir) - horizonMin);
                    let prevHorizonCos = horizonCos;
                    horizonCos = Math.max(horizonCos, sampleCos);
                    if (Math.abs(sampleCos - prevHorizonCos) < limits.cosine) sensitive = true;
                    if (sampleCos >= prevHorizonCos) {
                        const sampleLighting = view.lightingAt(sampleU, sampleV);
                        // A sky sample's normal is Capsaicin's decoded cleared normal (register D58).
                        const sampleWorldNormal: Vector = view.isSky(texel.x, texel.y) ? [SKY_NORMAL, SKY_NORMAL, SKY_NORMAL] : view.normalAt(texel.x, texel.y);
                        if (stepIndex > 0) {
                            const t = Math.min(intersectDirPlaneOneSided(prevSampleWorldPos, sampleWorldNormal, sampleWorldPos), intersectDirPlaneOneSided(prevSampleWorldPos, worldNormal, worldPos));
                            const closestWorldPos = scale(prevSampleWorldPos, t);
                            prevHorizonCos = clamp(dot(normalize(subtract(closestWorldPos, worldPos)), worldEyeDir), prevHorizonCos, horizonCos);
                        }
                        const horizonAngle0 = projWorldNormalAngle + Math.max(sideSign * Math.acos(prevHorizonCos) - projWorldNormalAngle, -Math.PI / 2);
                        const horizonAngle1 = projWorldNormalAngle + Math.min(sideSign * Math.acos(horizonCos) - projWorldNormalAngle, Math.PI / 2);
                        const sampleOcclusion = integrateHalfArc(horizonAngle0, projWorldNormalAngle) - integrateHalfArc(horizonAngle1, projWorldNormalAngle);
                        const facing = dot(scale(horizonWorldDir, -1), sampleWorldNormal);
                        if (Math.abs(facing) < limits.cosine) sensitive = true;
                        const visible = facing >= 0 ? 1 : 0;
                        globalLighting = [0, 1, 2].map((axis) => globalLighting[axis] + sampleLighting[axis] * sampleOcclusion * visible) as Vector;
                    }
                } else {
                    zeroLengthSamples++;
                }
                prevSampleWorldPos = sampleWorldPos;
            }
            horizonAngles[sideIndex] = sideSign * Math.acos(horizonCos);
            ambientOcclusion += projWorldNormalLen * integrateHalfArc(horizonAngles[sideIndex], projWorldNormalAngle);
            // Capsaicin scales the sum so far on each side (wgsl/ssgi.ts says why it is kept).
            globalLighting = scale(globalLighting, projWorldNormalLen);
        }
        const bent = integrateBentNormal(horizonAngles[0], horizonAngles[1], projWorldNormalAngle, worldEyeDir, sliceViewDir, constants);
        bentNormal = [0, 1, 2].map((axis) => bentNormal[axis] + projWorldNormalLen * bent[axis]) as Vector;
    }
    return {
        ambientOcclusion: ambientOcclusion / sliceCount,
        bentNormal: normalize(bentNormal),
        nearField: scale(globalLighting, 1 / sliceCount),
        sensitive,
        zeroLengthSamples,
    };
}

/** How far apart two twin results may be and still count as the same (the GPU checks' tolerances). */
export interface SSGITolerance {
    readonly occlusion: number;
    readonly bentNormal: number;
    readonly nearFieldRelative: number;
    readonly nearFieldAbsolute: number;
}

/** Whether two results agree within `tolerance` (the bent normal compared as stored, 0.5 n + 0.5). */
export function ssgiResultsAgree(a: Pick<SSGIPixel, 'ambientOcclusion' | 'bentNormal' | 'nearField'>, b: Pick<SSGIPixel, 'ambientOcclusion' | 'bentNormal' | 'nearField'>, tolerance: SSGITolerance): boolean {
    if (Math.abs(a.ambientOcclusion - b.ambientOcclusion) > tolerance.occlusion) return false;
    for (let axis = 0; axis < 3; axis++) {
        if (Math.abs(0.5 * (a.bentNormal[axis] - b.bentNormal[axis])) > tolerance.bentNormal) return false;
        const near = Math.abs(a.nearField[axis] - b.nearField[axis]);
        if (near > tolerance.nearFieldRelative * Math.abs(b.nearField[axis]) + tolerance.nearFieldAbsolute) return false;
    }
    return true;
}

/**
 * The twin at one pixel for a GPU check: the f64 result, sensitive where a
 * decision sits at its threshold or where rounding the positions to f32
 * moves the result past half of `tolerance` (the file comment).
 */
export function ssgiTwinPixel(screen: SSGIScreen, constants: SSGIConstants, x: number, y: number, tolerance: SSGITolerance): SSGIPixel | null {
    const exact = ssgiPixel(screen, constants, x, y, false);
    if (!exact) return null;
    const rounded = ssgiPixel(screen, constants, x, y, true);
    const half: SSGITolerance = {
        occlusion: tolerance.occlusion / 2,
        bentNormal: tolerance.bentNormal / 2,
        nearFieldRelative: tolerance.nearFieldRelative / 2,
        nearFieldAbsolute: tolerance.nearFieldAbsolute / 2,
    };
    const moves = !rounded || rounded.sensitive || !ssgiResultsAgree(rounded, exact, half);
    return { ...exact, sensitive: exact.sensitive || moves };
}
