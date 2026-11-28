/**
 * The texture store on the CPU: a twin of `gi/texture-store`'s layer build
 * (texture-store-wgsl.ts) and of sampling it (texture-sampling-wgsl.ts), for
 * references and tests. Production never uses it.
 *
 * It starts from the test's own pixels, not from what the GPU made, so a
 * reference built on it checks the store's build as well as the sampling:
 * the CPU reference tracer reads cutout alpha from it (T4.2), and the unit
 * tests check the build's rules (alpha-weighted mips) without a GPU.
 *
 * Step for step as the kernel:
 *
 * - level 0 from the source: a box over the source texels under each store
 *   texel when the source is larger, a bilinear blend of the nearest four
 *   when smaller, in linear light (sRGB sources decoded first), colour
 *   weighted by alpha;
 * - level k from level k - 1: the alpha-weighted average of 2 × 2 texels,
 *   from the unrounded linear values of the level above;
 * - each level stored as rgba8, colour sRGB-encoded, alpha linear.
 *
 * Sampling decodes the stored bytes and filters bilinearly within a level,
 * linearly between levels, with repeat addressing, as the store's sampler
 * does. The GPU's filter weights have a few bits of precision (8 on many
 * GPUs), so samples near a sharp edge can differ a little: compare with a
 * tolerance, or call answers near an edge ambiguous.
 */

import type { UvMapping, UvWrap } from '../contract/scene.ts';
import { STORE_LAYER_SIZE, STORE_MIP_LEVELS } from './texture-store-wgsl.ts';

/** RGBA8 pixels, rows top first (the `PixelData` source kind). */
export interface CpuPixelSource {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8Array | Uint8ClampedArray;
}

/** One store layer made on the CPU: each level as stored, rgba8 with sRGB-encoded colour and linear alpha, rows unpadded. */
export interface CpuStoreLayer {
    readonly levels: readonly Uint8Array[];
}

/** RGBA in linear light. */
export type LinearRgba = [number, number, number, number];

/** The sRGB decoding curve (IEC 61966-2-1), one channel in 0..1. */
export function srgbToLinear(encoded: number): number {
    return encoded <= 0.04045 ? encoded / 12.92 : Math.pow((encoded + 0.055) / 1.055, 2.4);
}

/** The sRGB encoding curve, one channel in 0..1. */
export function linearToSrgb(linear: number): number {
    return linear <= 0.0031308 ? linear * 12.92 : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
}

/** WGSL's `pack4x8unorm` for one channel: clamped, rounded to the nearest of 0..255. */
function toUnorm8(value: number): number {
    return Math.floor(0.5 + Math.min(Math.max(value, 0), 1) * 255);
}

/**
 * Sums for an alpha-weighted average (the kernel's `GiBuildAverage`): colour
 * weighted by alpha, alpha the plain average, the plain colour where every
 * texel is clear, and exactly the plain average where every texel is opaque.
 */
class AlphaWeightedAverage {
    private readonly weightedColour = [0, 0, 0];
    private alphaWeight = 0;
    private readonly plainColour = [0, 0, 0];
    private weight = 0;

    add(value: ArrayLike<number>, weight: number): void {
        const alpha = Math.min(Math.max(value[3], 0), 1);
        for (let channel = 0; channel < 3; channel++) {
            this.weightedColour[channel] += value[channel] * weight * alpha;
            this.plainColour[channel] += value[channel] * weight;
        }
        this.alphaWeight += weight * alpha;
        this.weight += weight;
    }

    result(): LinearRgba {
        if (this.alphaWeight === this.weight) return [this.plainColour[0], this.plainColour[1], this.plainColour[2], this.alphaWeight];
        if (this.alphaWeight > 0) {
            const scale = this.weight / this.alphaWeight;
            return [this.weightedColour[0] * scale, this.weightedColour[1] * scale, this.weightedColour[2] * scale, this.alphaWeight];
        }
        return [this.plainColour[0], this.plainColour[1], this.plainColour[2], 0];
    }
}

/** The source texels one store texel reads along one axis, and their weights (the kernel's `giBuildAxisTaps`). */
function axisTaps(texel: number, sourceSize: number): { readonly first: number; readonly weights: readonly number[] } {
    const scale = sourceSize / STORE_LAYER_SIZE;
    if (sourceSize >= STORE_LAYER_SIZE) {
        // A box over the source texels whose centres fall inside the store texel's footprint.
        const first = Math.ceil(texel * scale - 0.5);
        const end = Math.ceil((texel + 1) * scale - 0.5);
        const count = Math.max(end - first, 1);
        return { first, weights: new Array<number>(count).fill(1 / count) };
    }
    // Bilinear between the two texels around the store texel's centre.
    const centre = (texel + 0.5) * scale - 0.5;
    const first = Math.floor(centre);
    const firstWeight = 1 - (centre - first);
    return { first, weights: [firstWeight, 1 - firstWeight] };
}

/**
 * Makes one store layer from RGBA8 pixels, as `gi/texture-store` does.
 * `srgb`: the pixels' colour is sRGB-encoded (the texture descriptor's `srgb`).
 */
export function buildStoreLayerOnCpu(source: CpuPixelSource, srgb: boolean): CpuStoreLayer {
    const levels: Uint8Array[] = [];
    let linear = new Float64Array(STORE_LAYER_SIZE * STORE_LAYER_SIZE * 4);

    // Level 0 from the source.
    const readSource = (x: number, y: number): number[] => {
        const column = Math.min(Math.max(x, 0), source.width - 1);
        const row = Math.min(Math.max(y, 0), source.height - 1);
        const at = (row * source.width + column) * 4;
        const value = [0, 1, 2, 3].map((channel) => source.data[at + channel] / 255);
        if (srgb) for (let channel = 0; channel < 3; channel++) value[channel] = srgbToLinear(value[channel]);
        return value;
    };
    for (let y = 0; y < STORE_LAYER_SIZE; y++) {
        const tapsY = axisTaps(y, source.height);
        for (let x = 0; x < STORE_LAYER_SIZE; x++) {
            const tapsX = axisTaps(x, source.width);
            const sum = new AlphaWeightedAverage();
            tapsY.weights.forEach((rowWeight, row) => {
                tapsX.weights.forEach((columnWeight, column) => sum.add(readSource(tapsX.first + column, tapsY.first + row), rowWeight * columnWeight));
            });
            linear.set(clampedTexel(sum.result()), (y * STORE_LAYER_SIZE + x) * 4);
        }
    }
    levels.push(encodeLevel(linear));

    // Each coarser level from the unrounded linear values of the one above.
    for (let level = 1; level < STORE_MIP_LEVELS; level++) {
        const size = STORE_LAYER_SIZE >> level;
        const parentSize = size * 2;
        const next = new Float64Array(size * size * 4);
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const sum = new AlphaWeightedAverage();
                for (const [dx, dy] of [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [1, 1],
                ]) {
                    const at = ((2 * y + dy) * parentSize + 2 * x + dx) * 4;
                    sum.add(linear.subarray(at, at + 4), 0.25);
                }
                next.set(clampedTexel(sum.result()), (y * size + x) * 4);
            }
        }
        linear = next;
        levels.push(encodeLevel(linear));
    }
    return { levels };
}

/** A texel as the kernel writes it: clamped to 0..1 (giBuildWrite). */
function clampedTexel(value: LinearRgba): LinearRgba {
    return value.map((channel) => Math.min(Math.max(channel, 0), 1)) as LinearRgba;
}

/** One level as the layer stores it: rgba8, colour sRGB-encoded, alpha linear. */
function encodeLevel(linear: Float64Array): Uint8Array {
    const bytes = new Uint8Array(linear.length);
    for (let at = 0; at < linear.length; at++) bytes[at] = toUnorm8(at % 4 === 3 ? linear[at] : linearToSrgb(linear[at]));
    return bytes;
}

/** One stored texel, decoded as the sRGB format does (alpha linear), with repeat addressing. */
function storedTexel(layer: CpuStoreLayer, level: number, x: number, y: number): LinearRgba {
    const size = STORE_LAYER_SIZE >> level;
    const wrap = (value: number): number => ((value % size) + size) % size;
    const at = (wrap(y) * size + wrap(x)) * 4;
    const bytes = layer.levels[level];
    return [srgbToLinear(bytes[at] / 255), srgbToLinear(bytes[at + 1] / 255), srgbToLinear(bytes[at + 2] / 255), bytes[at + 3] / 255];
}

function bilinear(layer: CpuStoreLayer, level: number, u: number, v: number): LinearRgba {
    const size = STORE_LAYER_SIZE >> level;
    const x = u * size - 0.5;
    const y = v * size - 0.5;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const result: LinearRgba = [0, 0, 0, 0];
    const corners: readonly [number, number, number][] = [
        [x0, y0, (1 - fx) * (1 - fy)],
        [x0 + 1, y0, fx * (1 - fy)],
        [x0, y0 + 1, (1 - fx) * fy],
        [x0 + 1, y0 + 1, fx * fy],
    ];
    for (const [tx, ty, weight] of corners) {
        const texel = storedTexel(layer, level, tx, ty);
        for (let channel = 0; channel < 4; channel++) result[channel] += texel[channel] * weight;
    }
    return result;
}

/** `textureSampleLevel` with the store's sampler: bilinear within a level, linear between the two around `lod`, repeating. */
export function sampleStoreLayer(layer: CpuStoreLayer, u: number, v: number, lod: number): LinearRgba {
    const clamped = Math.min(Math.max(lod, 0), STORE_MIP_LEVELS - 1);
    const lower = Math.floor(clamped);
    const upper = Math.min(lower + 1, STORE_MIP_LEVELS - 1);
    const blend = clamped - lower;
    const a = bilinear(layer, lower, u, v);
    if (blend === 0) return a;
    const b = bilinear(layer, upper, u, v);
    return a.map((channel, index) => channel * (1 - blend) + b[index] * blend) as LinearRgba;
}

/** One axis of a mapped uv, wrapped (CPU twin of `giTextureWrap`). */
export function wrapTextureCoordinate(coordinate: number, wrap: UvWrap | undefined, halfTexel: number): number {
    if (wrap === 'clamp') return Math.min(Math.max(coordinate, halfTexel), 1 - halfTexel);
    if (wrap === 'mirror') {
        const half = coordinate * 0.5;
        const folded = 1 - Math.abs(2 * (half - Math.floor(half)) - 1);
        return Math.min(Math.max(folded, halfTexel), 1 - halfTexel);
    }
    return coordinate;
}

const IDENTITY_UV_MATRIX: readonly number[] = [1, 0, 0, 1, 0, 0];

/**
 * The texture coordinate of a point of a triangle before wrapping (CPU twin
 * of `giTextureMappedUv`): corner uvs `[u0, v0, u1, v1, u2, v2]` blended by
 * `weights` (of corners 0, 1, 2), then the mapping's 3×2 transform.
 */
export function mappedTextureUv(mapping: UvMapping | undefined, cornerUvs: ArrayLike<number>, weights: readonly [number, number, number]): [number, number] {
    const u = cornerUvs[0] * weights[0] + cornerUvs[2] * weights[1] + cornerUvs[4] * weights[2];
    const v = cornerUvs[1] * weights[0] + cornerUvs[3] * weights[1] + cornerUvs[5] * weights[2];
    const m = mapping?.matrix ?? IDENTITY_UV_MATRIX;
    return [m[0] * u + m[2] * v + m[4], m[1] * u + m[3] * v + m[5]];
}

/** A material texture at a point of a triangle, through its mapping, at `lod` (CPU twin of `giTextureSampleMapped`). */
export function sampleMappedOnCpu(layer: CpuStoreLayer, mapping: UvMapping | undefined, cornerUvs: ArrayLike<number>, weights: readonly [number, number, number], lod: number): LinearRgba {
    const [u, v] = mappedTextureUv(mapping, cornerUvs, weights);
    const halfTexel = (0.5 * Math.pow(2, Math.ceil(lod))) / STORE_LAYER_SIZE;
    return sampleStoreLayer(layer, wrapTextureCoordinate(u, mapping?.wrapU, halfTexel), wrapTextureCoordinate(v, mapping?.wrapV, halfTexel), lod);
}
