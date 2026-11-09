/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.hlsl,
 * screen_probes.hlsl and gi1.comp, and src/core/src/math/sampling.hlsl and pack.hlsl.
 *
 * Copyright © 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files(the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions :
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/**
 * CPU ports of the Capsaicin functions the screen probes use, written to give
 * the same numbers as the WGSL in `wgsl/capsaicin.ts` (f32 arithmetic through
 * `Math.fround` where it decides a result, such as the Halton jitter that
 * picks a pixel). The unit tests check them, and the GPU test page
 * (`tools/gpu/pages/radiance-cache-probes`) compares the GPU's spawn, its
 * compaction and its ray directions with them.
 */

import { PROBE_SIZE, PROBE_SPAWN_TILE_SIZE } from './settings.ts';

/** `kGI1_InvalidId` (gi1.hlsl:30): no probe in a probe-mask texel. */
export const INVALID_ID = 0xffffffff;

const f32 = Math.fround;

/** `CalculateHaltonNumber` (gi1.hlsl:44-57): the radical inverse of `index` in `base`, in f32 steps as the GPU takes them. */
export function calculateHaltonNumber(index: number, base: number): number {
    let f = 1;
    let result = 0;
    for (let i = index >>> 0; i > 0; ) {
        f = f32(f / base);
        result = f32(result + f32(f * (i % base)));
        // `i = uint(i / float(base))`: a float division, truncated.
        i = Math.trunc(f32(i / base)) >>> 0;
    }
    return result;
}

/** `CalculateHaltonSequence` (gi1.hlsl:59-63): bases 2 and 3, repeating every 256 frames. */
export function calculateHaltonSequence(index: number): [number, number] {
    const wrapped = ((index >>> 0) & 0xff) + 1;
    return [calculateHaltonNumber(wrapped, 2), calculateHaltonNumber(wrapped, 3)];
}

/** `ScreenProbes_PackSeed` (screen_probes.hlsl:93-96): the seed pixel as x << 16 | y. */
export function packSeed(x: number, y: number): number {
    return ((x << 16) | y) >>> 0;
}

/** `ScreenProbes_UnpackSeed` (screen_probes.hlsl:99-102). */
export function unpackSeed(packed: number): [number, number] {
    return [packed >>> 16, packed & 0xffff];
}

/**
 * The jitter of this frame's spawn tiles, in pixels (`SpawnScreenProbes`,
 * gi1.comp:567): the Halton point scaled to the tile and kept inside it.
 */
export function spawnJitter(frameIndex: number): [number, number] {
    const [hx, hy] = calculateHaltonSequence(frameIndex);
    const limit = PROBE_SPAWN_TILE_SIZE - 1;
    return [Math.trunc(Math.min(f32(hx * PROBE_SPAWN_TILE_SIZE), limit)), Math.trunc(Math.min(f32(hy * PROBE_SPAWN_TILE_SIZE), limit))];
}

/** Spawn tiles across and down a `width` × `height` screen (`SpawnScreenProbes`, gi1.comp:558-559). */
export function spawnTileCounts(width: number, height: number): [number, number] {
    return [Math.ceil(width / PROBE_SPAWN_TILE_SIZE), Math.ceil(height / PROBE_SPAWN_TILE_SIZE)];
}

/** What `SpawnScreenProbes` decides for one spawn tile. */
export interface SpawnedTile {
    /** The jittered pixel, clamped to the screen. */
    readonly seed: readonly [number, number];
    /** False on a sky pixel: the tile spawns no probe (its scan flag is 0). */
    readonly spawned: boolean;
}

/**
 * The spawn rule on the CPU (`SpawnScreenProbes`, gi1.comp:555-579): one
 * candidate per 16 × 16 tile at the frame's Halton jitter, kept unless the
 * pixel is sky. Tiles in row order, as the kernel's thread index.
 */
export function spawnScreenProbes(width: number, height: number, frameIndex: number, isSkyPixel: (x: number, y: number) => boolean): SpawnedTile[] {
    const [tilesX, tilesY] = spawnTileCounts(width, height);
    const [jitterX, jitterY] = spawnJitter(frameIndex);
    const tiles: SpawnedTile[] = [];
    for (let tile = 0; tile < tilesX * tilesY; tile++) {
        const x = Math.min((tile % tilesX) * PROBE_SPAWN_TILE_SIZE + jitterX, width - 1);
        const y = Math.min(Math.floor(tile / tilesX) * PROBE_SPAWN_TILE_SIZE + jitterY, height - 1);
        tiles.push({ seed: [x, y], spawned: !isSkyPixel(x, y) });
    }
    return tiles;
}

/** A unit 3-vector. */
export type Direction = readonly [number, number, number];

/**
 * `mapToHemiOctahedron` (sampling.hlsl:61-83): a point of the unit square to a
 * direction on the +z hemisphere, area-preserving (Clarberg's equal-area map,
 * rotated so the hemisphere fills the square).
 */
export function mapToHemiOctahedron(u: number, v: number): [number, number, number] {
    // From [0, 1] to [-1, 1], then the square turned 45° onto the diamond of the +z hemisphere.
    const sx = 2 * u - 1;
    const sy = 2 * v - 1;
    const stX = (sx + sy) * 0.5;
    const stY = (sx - sy) * 0.5;
    const absX = Math.abs(stX);
    const absY = Math.abs(stY);
    const distance = 1 - (absX + absY);
    const radius = 1 - Math.abs(distance);
    const phi = radius === 0 ? 0 : (Math.PI / 4) * ((absY - absX) / radius + 1);
    const radiusSquared = radius * radius;
    const sinTheta = radius * Math.sqrt(2 - radiusSquared);
    return [sinTheta * Math.sign(stX) * Math.cos(phi), sinTheta * Math.sign(stY) * Math.sin(phi), Math.sign(distance) * (1 - radiusSquared)];
}

/** `mapToHemiOctahedronInverse` (sampling.hlsl:90-116): a +z-hemisphere direction back to the unit square. */
export function mapToHemiOctahedronInverse(direction: Direction): [number, number] {
    const absX = Math.abs(direction[0]);
    const absY = Math.abs(direction[1]);
    const absZ = Math.abs(direction[2]);
    const radius = Math.sqrt(1 - absZ);
    const a = Math.max(absX, absY);
    const b = a === 0 ? 0 : Math.min(absX, absY) / a;
    let phi = Math.atan(b) * (2 / Math.PI);
    phi = absX >= absY ? phi : 1 - phi;
    const t = phi * radius;
    const s = radius - t;
    const stX = s * Math.sign(direction[0]);
    const stY = t * Math.sign(direction[1]);
    // The diamond back onto the square, then [-1, 1] to [0, 1].
    return [0.5 * (stX + stY) + 0.5, 0.5 * (stX - stY) + 0.5];
}

/** `GetOrthoVectors` (sampling.hlsl:118-126): two tangents completing `n` to an orthonormal basis. */
export function getOrthoVectors(n: Direction): { b1: [number, number, number]; b2: [number, number, number] } {
    const selected = Math.abs(n[2]) > 0;
    const p2z = selected ? n[2] : n[0];
    const k = 1 / Math.sqrt(p2z * p2z + n[1] * n[1]);
    let b1: [number, number, number] = [0, -p2z * k, n[1] * k];
    if (!selected) b1 = [b1[2], b1[1], b1[0]];
    const b2: [number, number, number] = [n[1] * b1[2] - n[2] * b1[1], n[2] * b1[0] - n[0] * b1[2], n[0] * b1[1] - n[1] * b1[0]];
    return { b1, b2 };
}

/** A world direction in the probe's frame (`mul(direction, CreateTBN(normal))`, sampling.hlsl:364-375): (·b1, ·b2, ·n). */
export function toProbeFrame(direction: Direction, normal: Direction): [number, number, number] {
    const { b1, b2 } = getOrthoVectors(normal);
    const dot = (a: Direction, b: Direction): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    return [dot(direction, b1), dot(direction, b2), dot(direction, normal)];
}

/** The probe cell (0..63, x + y × 8) a world direction falls in (`BlendScreenProbes`, gi1.comp:1201-1202). */
export function probeCellOf(direction: Direction, normal: Direction): number {
    const [u, v] = mapToHemiOctahedronInverse(toProbeFrame(direction, normal));
    const x = Math.min(Math.trunc(u * PROBE_SIZE), PROBE_SIZE - 1);
    const y = Math.min(Math.trunc(v * PROBE_SIZE), PROBE_SIZE - 1);
    return x + y * PROBE_SIZE;
}

const conversionFloats = new Float32Array(1);
const conversionWords = new Uint32Array(conversionFloats.buffer);

/** A number to IEEE half bits through f32, rounding to nearest even (what `f32tof16` and `pack2x16float` do in range). */
export function floatToHalfBits(value: number): number {
    conversionFloats[0] = value;
    const bits = conversionWords[0];
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    let mantissa = bits & 0x7fffff;
    if (exponent === 0xff) return sign | 0x7c00 | (mantissa === 0 ? 0 : 0x200);
    const halfExponent = exponent - 127 + 15;
    if (halfExponent >= 0x1f) return sign | 0x7c00;
    if (halfExponent <= 0) {
        // A half subnormal (or zero): the implicit bit joins the mantissa, shifted into place.
        if (halfExponent < -10) return sign;
        mantissa |= 0x800000;
        const shift = 14 - halfExponent;
        let half = mantissa >>> shift;
        const remainder = mantissa & ((1 << shift) - 1);
        const halfway = 1 << (shift - 1);
        if (remainder > halfway || (remainder === halfway && (half & 1) === 1)) half++;
        return sign | half;
    }
    let half = (halfExponent << 10) | (mantissa >>> 13);
    const remainder = mantissa & 0x1fff;
    // A carry out of the mantissa steps the exponent up, which is the right rounding (to infinity at the top).
    if (remainder > 0x1000 || (remainder === 0x1000 && (half & 1) === 1)) half++;
    return sign | half;
}

/** IEEE half bits to a number. */
export function halfBitsToFloat(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) return sign * mantissa * 2 ** -24;
    if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : NaN;
    return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** `packHalf3` (pack.hlsl:411-416): x and y in the first word, z in the low half of the second. */
export function packHalf3(value: Direction): [number, number] {
    return [(floatToHalfBits(value[0]) | (floatToHalfBits(value[1]) << 16)) >>> 0, floatToHalfBits(value[2])];
}

/** `unpackHalf3` (pack.hlsl:445-449). */
export function unpackHalf3(packed: readonly [number, number]): [number, number, number] {
    return [halfBitsToFloat(packed[0] & 0xffff), halfBitsToFloat(packed[0] >>> 16), halfBitsToFloat(packed[1] & 0xffff)];
}

/** `ScreenProbes_QuantizeRadiance` (screen_probes.hlsl:231-234): radiance × 10⁴ rounded, for atomic sums in workgroup memory. */
export function quantizeRadiance(value: number): number {
    return roundHalfToEven(1e4 * value);
}

function roundHalfToEven(value: number): number {
    const floor = Math.floor(value);
    const fraction = value - floor;
    if (fraction > 0.5) return floor + 1;
    if (fraction < 0.5) return floor;
    return floor % 2 === 0 ? floor : floor + 1;
}

/** `SH_GetCoefficients` (spherical_harmonics.hlsl:28-48): the nine real SH basis values at a unit direction, Capsaicin's order and signs. */
export function shGetCoefficients(direction: Direction): number[] {
    const [x, y, z] = direction;
    // fC1 = x·x − y·y and fS1 = x·y + y·x, as the HLSL builds them.
    return [
        0.2820947917738781,
        -0.48860251190292 * y,
        0.4886025119029199 * z,
        -0.48860251190292 * x,
        0.5462742152960395 * (x * y + y * x),
        -1.092548430592079 * z * y,
        0.9461746957575601 * z * z - 0.3153915652525201,
        -1.092548430592079 * z * x,
        0.5462742152960395 * (x * x - y * y),
    ];
}

/**
 * `SH_GetCoefficients_ClampedCosine_Cone` (spherical_harmonics.hlsl:78-104):
 * the clamped cosine lobe about `direction`, cut to a cone of half-angle
 * `coneThetaMax`, as convolved SH (its sum with a radiance's coefficients is
 * the irradiance).
 */
export function shGetCoefficientsClampedCosineCone(direction: Direction, coneThetaMax: number): number[] {
    const [x, y, z] = direction;
    const sinThetaMax = Math.sin(coneThetaMax);
    const cosThetaMax = Math.cos(coneThetaMax);
    const band1 = 1.023326707946489 * (1 - cosThetaMax ** 3);
    const band2 = (4 - 3 * sinThetaMax ** 3) * sinThetaMax ** 2;
    return [
        0.886226925452758 * sinThetaMax ** 2,
        -band1 * y,
        band1 * z,
        -band1 * x,
        0.8580855308097834 * band2 * x * y,
        -0.8580855308097834 * band2 * y * z,
        0.2477079561003757 * band2 * (3 * z * z - 1),
        -0.8580855308097834 * band2 * x * z,
        0.4290427654048917 * band2 * (x * x - y * y),
    ];
}

/** The solid angle of one probe cell: 2π over 64 equal-area cells (`ProjectScreenProbes`' weight here, register D25). */
export const SH_CELL_SOLID_ANGLE = (2 * Math.PI) / (PROBE_SIZE * PROBE_SIZE);
/** Capsaicin's weight per cell in `ProjectScreenProbes`: `1 / probe_size` (gi1.comp:1468), 4/π times the solid angle. */
export const CAPSAICIN_SH_CELL_WEIGHT = 1 / PROBE_SIZE;

/** The world direction of a probe cell's centre about `normal` (`ProjectScreenProbes`, gi1.comp:1468-1473). */
export function probeCellDirection(cell: number, normal: Direction): Direction {
    const { b1, b2 } = getOrthoVectors(normal);
    const [x, y, z] = mapToHemiOctahedron(((cell % PROBE_SIZE) + 0.5) / PROBE_SIZE, (Math.floor(cell / PROBE_SIZE) + 0.5) / PROBE_SIZE);
    return [x * b1[0] + y * b2[0] + z * normal[0], x * b1[1] + y * b2[1] + z * normal[1], x * b1[2] + y * b2[2] + z * normal[2]];
}

/** Points across a cell, each way, that `ProjectScreenProbes` averages the SH basis over (register D25). */
export const SH_CELL_SUBSAMPLES = 4;

/**
 * A probe cell's mean SH basis about `normal`: the basis averaged over
 * `subsamples²` points spread evenly over the cell in the hemi-octahedral
 * square (equal-area, so evenly over its solid angle too). With 1 it is the
 * basis at the cell centre, as Capsaicin takes it (gi1.comp:1468-1473).
 */
export function probeCellSH(cell: number, normal: Direction, subsamples = SH_CELL_SUBSAMPLES): number[] {
    const { b1, b2 } = getOrthoVectors(normal);
    const sum = new Array<number>(9).fill(0);
    for (let row = 0; row < subsamples; row++) {
        for (let column = 0; column < subsamples; column++) {
            const u = ((cell % PROBE_SIZE) + (column + 0.5) / subsamples) / PROBE_SIZE;
            const v = (Math.floor(cell / PROBE_SIZE) + (row + 0.5) / subsamples) / PROBE_SIZE;
            const [x, y, z] = mapToHemiOctahedron(u, v);
            const direction: Direction = [x * b1[0] + y * b2[0] + z * normal[0], x * b1[1] + y * b2[1] + z * normal[1], x * b1[2] + y * b2[2] + z * normal[2]];
            shGetCoefficients(direction).forEach((value, index) => (sum[index] += value));
        }
    }
    return sum.map((value) => value / (subsamples * subsamples));
}

/**
 * `ProjectScreenProbes` (gi1.comp:1462-1511) on the CPU, as the WGSL runs it:
 * each cell's radiance times `weight` and the cell's mean SH basis
 * ({@link probeCellSH}), summed over the 64 cells (cells `x + y × 8`).
 * `cellRadiance` gets the cell centre's direction. Capsaicin's own projection
 * is `weight` {@link CAPSAICIN_SH_CELL_WEIGHT} with `subsamples` 1.
 */
export function projectProbe(
    cellRadiance: (cell: number, direction: Direction) => number,
    normal: Direction,
    weight = SH_CELL_SOLID_ANGLE,
    subsamples = SH_CELL_SUBSAMPLES,
): number[] {
    const sum = new Array<number>(9).fill(0);
    for (let cell = 0; cell < PROBE_SIZE * PROBE_SIZE; cell++) {
        const radiance = cellRadiance(cell, probeCellDirection(cell, normal)) * weight;
        probeCellSH(cell, normal, subsamples).forEach((value, index) => (sum[index] += value * radiance));
    }
    return sum;
}

/** `ScreenProbes_CalculateSHIrradiance_BentCone` (screen_probes.hlsl:200-216) on the CPU: irradiance along `normal` from nine coefficients, never negative. */
export function shIrradiance(coefficients: readonly number[], normal: Direction, ao = 1): number {
    const lobe = shGetCoefficientsClampedCosineCone(normal, Math.acos(Math.sqrt(Math.min(Math.max(1 - ao, 0), 1))));
    let irradiance = 0;
    lobe.forEach((value, index) => (irradiance += value * coefficients[index]));
    return Math.max(irradiance, 0);
}

/**
 * The SH projection of a radiance over the hemisphere about `normal`
 * (zero below), integrated finely: `steps²` equal-area points (the
 * hemi-octahedral map is equal-area, so each weighs 2π / steps²). The
 * reference the GPU's 64-cell projection is compared against.
 */
export function integrateHemisphereSH(radiance: (direction: Direction) => number, normal: Direction, steps = 256): number[] {
    const { b1, b2 } = getOrthoVectors(normal);
    const sum = new Array<number>(9).fill(0);
    const weight = (2 * Math.PI) / (steps * steps);
    for (let i = 0; i < steps; i++) {
        for (let j = 0; j < steps; j++) {
            const [x, y, z] = mapToHemiOctahedron((i + 0.5) / steps, (j + 0.5) / steps);
            const direction: Direction = [x * b1[0] + y * b2[0] + z * normal[0], x * b1[1] + y * b2[1] + z * normal[1], x * b1[2] + y * b2[2] + z * normal[2]];
            const value = radiance(direction) * weight;
            shGetCoefficients(direction).forEach((basis, index) => (sum[index] += basis * value));
        }
    }
    return sum;
}
