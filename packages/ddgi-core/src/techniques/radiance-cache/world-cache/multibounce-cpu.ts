/**********************************************************************
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
********************************************************************/

/**
 * The CPU twin of GI-1.2's multibounce arithmetic (RC3.3,
 * `hash-grid-population-wgsl.ts`): Capsaicin's random numbers
 * (`random_number_generator.hlsl`), the second-bounce direction
 * (`sampleBRDFAndEvaluateDiffuse` with its quaternion frame and
 * `mapToCosineHemisphere`), the discard, and the sample
 * `UpdateMultibounceCells` adds to a first-bounce cell. Every float step is
 * rounded to f32 as the GPU rounds it, so the GPU page can predict which
 * first-bounce hits trace a second ray and in which direction.
 */

import type { Vec3 } from '../../../platform/contract/math.ts';
import { pcgHash } from './hash-grid-cpu.ts';

const f32 = Math.fround;

/** `INV_PI` and `TWO_PI` (math/math_constants.hlsl), as f32. */
const INV_PI = f32(0.318309886183791);
const TWO_PI = f32(6.28318530717959);

// ---- components/random_number_generator/random_number_generator.hlsl ------------------------------

/** Capsaicin's `Random` (random_number_generator.hlsl:30-86): a PCG generator's state. */
export interface CapsaicinRandom {
    state: number;
}

/**
 * `MakeRandom(seed, index)` (random_number_generator.hlsl:108-116), started
 * from `pcgHash(seed)` where Capsaicin reads its seed table (register D18), as
 * the WGSL `rcMakeRandom` does.
 */
export function makeRandom(seed: number, index: number): CapsaicinRandom {
    const increment = ((index << 1) | 1) >>> 0;
    return { state: (Math.imul((pcgHash(seed >>> 0) + increment) >>> 0, 747796405) + increment) >>> 0 };
}

/** `Random::randInt` (random_number_generator.hlsl:38-46). */
export function randInt(random: CapsaicinRandom): number {
    const state = random.state;
    random.state = (Math.imul(random.state, 747796405) + 2891336453) >>> 0;
    const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
    return ((word >>> 22) ^ word) >>> 0;
}

/** `Random::rand` (random_number_generator.hlsl:62-67): the upper 24 bits, exact in f32. */
export function rand(random: CapsaicinRandom): number {
    return (randInt(random) >>> 8) / 16777216;
}

/** `Random::rand2` (random_number_generator.hlsl:73-76). */
export function rand2(random: CapsaicinRandom): [number, number] {
    const first = rand(random);
    return [first, rand(random)];
}

// ---- math/quaternion.hlsl and math/sampling.hlsl ----------------------------------------------------

export type Quaternion = readonly [number, number, number, number];

function normalize4(value: readonly [number, number, number, number]): Quaternion {
    const length = f32(Math.sqrt(f32(f32(f32(value[0] * value[0]) + f32(value[1] * value[1])) + f32(f32(value[2] * value[2]) + f32(value[3] * value[3])))));
    return [f32(value[0] / length), f32(value[1] / length), f32(value[2] / length), f32(value[3] / length)];
}

/** `QuaternionRotationZ` (quaternion.hlsl:58-64): the rotation taking `direction` to +z. */
export function quaternionRotationZ(direction: Vec3): Quaternion {
    return direction[2] >= -0.99999 ? normalize4([direction[1], -direction[0], 0, f32(1 + direction[2])]) : [1, 0, 0, 0];
}

/** `Quaternion::inverse` (quaternion.hlsl:37-42). */
export function quaternionInverse(quaternion: Quaternion): Quaternion {
    return [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]];
}

function dot3(a: Vec3, b: Vec3): number {
    return f32(f32(f32(a[0] * b[0]) + f32(a[1] * b[1])) + f32(a[2] * b[2]));
}

/** `Quaternion::transform` (quaternion.hlsl:49-55): 2 (q·d) q + (w² - q·q) d + 2 w (q × d). */
export function quaternionTransform(quaternion: Quaternion, direction: Vec3): [number, number, number] {
    const axis: Vec3 = [quaternion[0], quaternion[1], quaternion[2]];
    const w = quaternion[3];
    const along = f32(2 * dot3(axis, direction));
    const scale = f32(f32(w * w) - dot3(axis, axis));
    const cross: Vec3 = [
        f32(f32(axis[1] * direction[2]) - f32(axis[2] * direction[1])),
        f32(f32(axis[2] * direction[0]) - f32(axis[0] * direction[2])),
        f32(f32(axis[0] * direction[1]) - f32(axis[1] * direction[0])),
    ];
    const twoW = f32(2 * w);
    return [0, 1, 2].map((i) => f32(f32(f32(along * axis[i]) + f32(scale * direction[i])) + f32(twoW * cross[i]))) as [number, number, number];
}

/** `mapToCosineHemisphere` (sampling.hlsl:161-169): a cosine-weighted direction about +z. */
export function mapToCosineHemisphere(samples: readonly [number, number]): [number, number, number] {
    const sinTheta = f32(Math.sqrt(samples[0]));
    const phi = f32(TWO_PI * samples[1]);
    return [f32(sinTheta * f32(Math.cos(phi))), f32(sinTheta * f32(Math.sin(phi))), f32(Math.sqrt(f32(1 - samples[0])))];
}

/** A second-bounce direction and its reflectance (cosine included) and pdf. */
export interface BRDFSample {
    readonly direction: [number, number, number];
    readonly brdf: [number, number, number];
    readonly pdf: number;
}

/**
 * `sampleBRDFAndEvaluateDiffuse` (material_sampling.hlsl:741-773) on a
 * Lambert surface (register D8), as `HashGridCache_SampleBRDFAndEvaluateDiffuse`:
 * a cosine-weighted direction about `normal` in Capsaicin's quaternion frame,
 * reflectance albedo / π × cos, pdf cos / π.
 */
export function sampleBRDFAndEvaluateDiffuse(albedo: Vec3, samples: readonly [number, number], normal: Vec3): BRDFSample {
    const localRotation = quaternionRotationZ(normal);
    const newLight = mapToCosineHemisphere(samples);
    const cosine = Math.min(Math.max(newLight[2], 0), 1);
    const brdf = albedo.map((channel) => f32(f32(channel * INV_PI) * cosine)) as [number, number, number];
    const pdf = f32(cosine * INV_PI);
    const world = quaternionTransform(quaternionInverse(localRotation), newLight);
    const length = f32(Math.sqrt(dot3(world, world)));
    return { direction: [f32(world[0] / length), f32(world[1] / length), f32(world[2] / length)], brdf, pdf };
}

// ---- The multibounce passes' arithmetic ----------------------------------------------------------

/**
 * What `PopulateMultibounceCells` (gi1.comp:1869-1940) draws for the
 * first-bounce visibility entry `did` in frame `frameIndex`: whether it stops
 * there (`rand() <= discard_multibounce_ray_probability`), and otherwise the
 * two numbers its direction is sampled from.
 */
export function multibounceDraw(did: number, frameIndex: number, discardProbability: number): { readonly discarded: true } | { readonly discarded: false; readonly samples: [number, number] } {
    const random = makeRandom(did, frameIndex);
    if (rand(random) <= f32(discardProbability)) return { discarded: true };
    return { discarded: false, samples: rand2(random) };
}

/**
 * The sample `UpdateMultibounceCells` (gi1.comp:1958-1966) adds to the
 * first-bounce cell for one second-bounce hit: the second-bounce cell's
 * filtered direct radiance (rgb over its count) × BRDF / pdf, divided by
 * 1 - the discard probability.
 */
export function multibounceSample(directMean: Vec3, brdf: Vec3, pdf: number, discardProbability: number): [number, number, number] {
    const keep = f32(1 - f32(discardProbability));
    return [0, 1, 2].map((i) => f32(f32(f32(directMean[i] * brdf[i]) / pdf) / keep)) as [number, number, number];
}

/**
 * The indirect value a first-bounce cell settles at when every second-bounce
 * hit sees the same direct radiance `secondDirect` off Lambert surfaces of
 * albedo `albedo`: each kept sample is albedo × secondDirect / (1 - p)
 * (BRDF / pdf is the albedo), and `UpdateTiles` averages the samples it gets.
 * Only the kept rays add a sample, so the 1 / (1 - p) isn't undone by a count
 * that includes the discarded ones: the value is 1 / (1 - p) times the
 * second bounce's expected value (3.33 × at p = 0.7). Capsaicin behaviour,
 * to be fixed under DECISIONS 2026-11-19 (NOTES, RC3.3).
 */
export function multibounceSettledIndirect(albedo: number, secondDirect: number, discardProbability: number): number {
    return (albedo * secondDirect) / (1 - discardProbability);
}
