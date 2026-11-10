/**
 * A CPU reference of the kernels (wgsl.ts), line for line, in f64: what the
 * GPU checks compare against, and what the unit tests run Capsaicin's rules
 * on. Not used by the denoiser itself.
 *
 * Images are `Float32Array`s, row by row: four channels a pixel for colours,
 * normals and motion, one for depth and the colour delta. Normals are
 * decoded (signed), and the previous normals hold 0 on sky pixels, as
 * `KeepGuides` writes them. The camera is `frameConstants`' (camera.ts).
 */

import type { FrameConstants, Vec3 } from './camera.ts';
import {
    COLOR_DELTA_SMOOTHING,
    FILTER_DEPTH_SHARPNESS_WITH_HISTORY,
    FILTER_DEPTH_SHARPNESS_WITHOUT_HISTORY,
    GRAZING_CELL_EXPONENT,
    GRAZING_CELL_SCALE,
    HISTORY_SAMPLES_PER_COUNT,
    LUMINANCE_WEIGHTS,
    MAX_BLUR_MASK,
    MIN_HISTORY_LUMINANCE,
    MIN_HISTORY_SAMPLES,
    ONE_OVER_SQRT_OF_TWO,
    REMOVE_NANS_FLOOR,
    REPROJECTION_NORMAL_COSINE,
    VIGNETTE_EXPONENT,
    VIGNETTE_SCALE,
} from './constants.ts';

type Vec4 = [number, number, number, number];

/** One frame's inputs to `ReprojectGI`. */
export interface ReprojectInputs {
    readonly width: number;
    readonly height: number;
    /** This frame's noisy irradiance, alpha the hint. */
    readonly radiance: Float32Array;
    readonly depth: Float32Array;
    readonly normal: Float32Array;
    /** NDC motion (current − previous) in rg. */
    readonly motion: Float32Array;
    readonly previousColor: Float32Array;
    readonly previousColorDelta: Float32Array;
    readonly previousDepth: Float32Array;
    readonly previousNormal: Float32Array;
}

export interface ReprojectResult {
    /** History: irradiance × samples, sample count. */
    readonly color: Float32Array;
    readonly colorDelta: Float32Array;
    /** The blur mask as written, before the texture's snorm rounding: radius / 8, −1/8 on sky pixels. */
    readonly blurMask: Float32Array;
    /** Pixels the history was found for (some tap kept). */
    readonly reprojected: Uint8Array;
    /**
     * Pixels where a test sat within `margin` of its threshold (the bounds of
     * the previous uv, the distance and normal tests, a tap's count against
     * 1), so f32 on the GPU may decide it the other way.
     */
    readonly undecided: Uint8Array;
}

const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const length3 = (a: ArrayLike<number>): number => Math.sqrt(dot3(a, a));
const saturate = (x: number): number => Math.min(Math.max(x, 0), 1);
const mix = (a: number, b: number, t: number): number => a + (b - a) * t;
const luminance = (rgb: ArrayLike<number>): number => LUMINANCE_WEIGHTS[0] * rgb[0] + LUMINANCE_WEIGHTS[1] * rgb[1] + LUMINANCE_WEIGHTS[2] * rgb[2];

/** `GIDenoiser_RemoveNaNs` (gi_denoiser.hlsl:64-92) on one value. */
export function removeNaNs(value: number): number {
    const compressed = saturate(value / (1 + value));
    return compressed / Math.max(1 - compressed, REMOVE_NANS_FLOOR);
}

/** `transformPointProjection(uv, depth, inverse)`: a point relative to this frame's eye. */
export function transformPointProjection(u: number, v: number, depth: number, inverse: readonly number[]): Vec3 {
    const ndc = [2 * u - 1, 2 * (1 - v) - 1, depth, 1];
    const h = [0, 1, 2, 3].map((row) => inverse[row] * ndc[0] + inverse[4 + row] * ndc[1] + inverse[8 + row] * ndc[2] + inverse[12 + row] * ndc[3]);
    return [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
}

/** The far plane's depth: where sky pixels sit (register D67). */
export function skyDepth(reversedDepth: boolean): number {
    return reversedDepth ? 0 : 1;
}

/** `ReprojectGI` (gi1.comp:3996-4102) over the whole image. */
export function reprojectGI(inputs: ReprojectInputs, frame: FrameConstants, margin = 1e-3): ReprojectResult {
    const { width, height } = inputs;
    const pixels = width * height;
    const color = new Float32Array(pixels * 4);
    const colorDelta = new Float32Array(pixels);
    const blurMask = new Float32Array(pixels);
    const reprojected = new Uint8Array(pixels);
    const undecided = new Uint8Array(pixels);
    const far = skyDepth(frame.reversedDepth);
    const near = (value: number, threshold: number): boolean => Math.abs(value - threshold) <= margin * Math.max(Math.abs(threshold), 1e-6);

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel = y * width + x;
            const input: Vec4 = [inputs.radiance[pixel * 4], inputs.radiance[pixel * 4 + 1], inputs.radiance[pixel * 4 + 2], inputs.radiance[pixel * 4 + 3]];
            const lighting: Vec4 = [0, 0, 0, 0];
            const depth = inputs.depth[pixel];
            let alphaBlend = 1;
            let delta = 0;
            const u = (x + 0.5) / width;
            const v = (y + 0.5) / height;
            const isSky = depth === far;

            if (!isSky) {
                const previousU = u - inputs.motion[pixel * 4] * 0.5;
                const previousV = v + inputs.motion[pixel * 4 + 1] * 0.5;
                const normal = inputs.normal.subarray(pixel * 4, pixel * 4 + 3);
                const bounds = [previousU, previousV];
                if (bounds.some((value) => near(value, 0) || near(value, 1))) undecided[pixel] = 1;
                if (previousU > 0 && previousU < 1 && previousV > 0 && previousV < 1) {
                    const world = transformPointProjection(u, v, depth, frame.inverseViewProjection);
                    const worldLength = length3(world);
                    const normalLength = length3(normal);
                    const facing = Math.max(-dot3(world, normal) / (worldLength * normalLength), 0);
                    const cellSize = worldLength * frame.cellSize * mix(1, GRAZING_CELL_SCALE, Math.pow(1 - facing, GRAZING_CELL_EXPONENT));
                    let weight = 0;
                    for (let ty = -1; ty <= 1; ty++) {
                        for (let tx = -1; tx <= 1; tx++) {
                            const s = previousU + tx / width;
                            const t = previousV + ty / height;
                            const texelX = Math.min(Math.max(Math.floor(s * width), 0), width - 1);
                            const texelY = Math.min(Math.max(Math.floor(t * height), 0), height - 1);
                            const tap = texelY * width + texelX;
                            const c = inputs.previousColor.subarray(tap * 4, tap * 4 + 4);
                            const previousNormal = inputs.previousNormal.subarray(tap * 4, tap * 4 + 3);
                            if (near(c[3], 1)) undecided[pixel] = 1;
                            if (c[3] < 1 || dot3(previousNormal, previousNormal) === 0) continue;
                            const previousWorld = transformPointProjection(s, t, inputs.previousDepth[tap], frame.previousInverseViewProjection);
                            const distance = Math.hypot(world[0] - previousWorld[0], world[1] - previousWorld[1], world[2] - previousWorld[2]);
                            const cosine = dot3(normal, previousNormal);
                            if (near(distance, cellSize) || near(cosine, REPROJECTION_NORMAL_COSINE)) undecided[pixel] = 1;
                            if (distance < cellSize && cosine > REPROJECTION_NORMAL_COSINE) {
                                const subpixelDistance = Math.hypot(Math.floor(s * width) + 0.5 - previousU * width, Math.floor(t * height) + 0.5 - previousV * height);
                                const w = saturate(1 - subpixelDistance * ONE_OVER_SQRT_OF_TWO);
                                delta += w * inputs.previousColorDelta[tap];
                                for (let channel = 0; channel < 4; channel++) lighting[channel] += w * c[channel];
                                weight += w;
                            }
                        }
                    }
                    if (weight > 0) {
                        reprojected[pixel] = 1;
                        delta = removeNaNs(Math.abs(delta) / weight) * Math.sign(delta);
                        for (let channel = 0; channel < 4; channel++) lighting[channel] = removeNaNs(lighting[channel] / weight);
                    }
                    if (input[3] > 0) {
                        const lumaA = luminance(input);
                        const history = Math.max(lighting[3], 1);
                        const lumaB = luminance([lighting[0] / history, lighting[1] / history, lighting[2] / history]);
                        delta = mix(delta, lumaA - lumaB, COLOR_DELTA_SMOOTHING);
                        alphaBlend = saturate(1 - Math.abs(delta) / Math.max(lumaB, MIN_HISTORY_LUMINANCE));
                    }
                }
            }

            const mask = isSky ? -1 : Math.max(MAX_BLUR_MASK - lighting[3], 0);
            if (input[3] > 0 || lighting[3] < 1) {
                for (let channel = 0; channel < 3; channel++) lighting[channel] += input[channel];
                lighting[3] += input[3] > 0 ? 1 : -1;
            }
            const vignette = Math.pow(VIGNETTE_SCALE * u * (1 - v) * v * (1 - u), VIGNETTE_EXPONENT);
            const maxSampleCount = Math.max(mix(MIN_HISTORY_SAMPLES, HISTORY_SAMPLES_PER_COUNT * Math.min(Math.abs(lighting[3]), MAX_BLUR_MASK), alphaBlend) * vignette, 1);
            if (lighting[3] > maxSampleCount) {
                const scale = maxSampleCount / lighting[3];
                for (let channel = 0; channel < 4; channel++) lighting[channel] *= scale;
            }
            color.set(lighting, pixel * 4);
            colorDelta[pixel] = delta;
            blurMask[pixel] = mask / MAX_BLUR_MASK;
        }
    }
    return { color, colorDelta, blurMask, reprojected, undecided };
}

/** `GIDenoiser_GetBlurRadius` (gi_denoiser.hlsl:43-54) from a mask value as the texture holds it. */
export function blurRadius(mask: number): number {
    const radius = mask * MAX_BLUR_MASK;
    return radius > 0 ? Math.trunc(Math.max(radius, 1) + 0.5) : 0;
}

/** The view-axis depth of every pixel (FilterGI's `toLinearDepth`, register D70). */
export function linearDepths(depth: Float32Array, width: number, height: number, frame: FrameConstants): Float64Array {
    const out = new Float64Array(width * height);
    const forward: Vec3 = frame.viewForward;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const point = transformPointProjection((x + 0.5) / width, (y + 0.5) / height, depth[y * width + x], frame.inverseViewProjection);
            out[y * width + x] = dot3(point, forward);
        }
    }
    return out;
}

export interface FilterResult {
    /** The pass's history colour. */
    readonly color: Float32Array;
    /** `g_IrradianceBuffer` (second pass only: rgb / count, alpha 1); null on the first pass. */
    readonly irradiance: Float32Array | null;
}

/**
 * `FilterGI` (gi1.comp:4104-4154), one pass along `direction`: (1, 0) or
 * (0, 1). `radius` per pixel from the blur mask (`blurRadius`), `linear`
 * from `linearDepths`, `normal` decoded.
 */
export function filterGI(
    input: Float32Array,
    radius: Int32Array,
    linear: Float64Array,
    normal: Float32Array,
    width: number,
    height: number,
    direction: readonly [number, number],
): FilterResult {
    const color = new Float32Array(input.length);
    const irradiance = direction[1] > 0 ? new Float32Array(input.length) : null;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pixel = y * width + x;
            const lighting: Vec4 = [input[pixel * 4], input[pixel * 4 + 1], input[pixel * 4 + 2], input[pixel * 4 + 3]];
            const r = radius[pixel];
            if (r > 0) {
                let weight = 1;
                const count = Math.max(lighting[3], 1);
                const sum = [lighting[0] / count, lighting[1] / count, lighting[2] / count];
                const centerDepth = linear[pixel];
                const centerNormal = normal.subarray(pixel * 4, pixel * 4 + 3);
                const hasSamples = lighting[3] > 0;
                const sharpness = hasSamples ? FILTER_DEPTH_SHARPNESS_WITH_HISTORY : FILTER_DEPTH_SHARPNESS_WITHOUT_HISTORY;
                for (let offset = -r; offset <= r; offset++) {
                    const px = Math.min(Math.max(x + offset * direction[0], 0), width - 1);
                    const py = Math.min(Math.max(y + offset * direction[1], 0), height - 1);
                    const tap = py * width + px;
                    const c = input.subarray(tap * 4, tap * 4 + 4);
                    if (!(c[3] > 0)) continue;
                    const depthDiff = 1 - centerDepth / linear[tap];
                    const depthFactor = Math.pow(2, -sharpness * Math.abs(depthDiff));
                    const normalFactor = Math.max(dot3(normal.subarray(tap * 4, tap * 4 + 3), centerNormal), 0) ** 4;
                    const w = depthFactor * (hasSamples ? normalFactor : 1);
                    const tapCount = Math.max(c[3], 1);
                    for (let channel = 0; channel < 3; channel++) sum[channel] += (w * c[channel]) / tapCount;
                    weight += w;
                }
                for (let channel = 0; channel < 3; channel++) lighting[channel] = (sum[channel] / weight) * count;
            }
            color.set(lighting, pixel * 4);
            if (irradiance) {
                const count = Math.max(lighting[3], 1);
                irradiance.set([lighting[0] / count, lighting[1] / count, lighting[2] / count, 1], pixel * 4);
            }
        }
    }
    return { color, irradiance };
}
