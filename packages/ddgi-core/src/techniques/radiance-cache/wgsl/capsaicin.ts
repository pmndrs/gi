/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.hlsl,
 * screen_probes.hlsl and gi1.comp; src/core/src/math/sampling.hlsl, pack.hlsl and
 * transform.hlsl; src/core/src/components/random_number_generator/random_number_generator.hlsl.
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
 * Capsaicin's shared functions the screen-probe kernels call, in WGSL, each
 * named after its HLSL original with an `rc` prefix (WGSL has no overloads
 * or includes, and the prefix keeps them apart from the platform's `gi`
 * names). Their CPU twins are in `capsaicin-math.ts`.
 *
 * Code only: nothing bound. The kernels in `screen-probes.ts` include it.
 */

import { PROBE_CELLS, PROBE_SIZE, PROBE_SPAWN_TILE_SIZE } from '../settings.ts';

export const CAPSAICIN_WGSL = /* wgsl */ `
// ---- Capsaicin constants -------------------------------------------------------------------

// probe_size_ (gi1.h:191): cells across a probe, and pixels across a probe tile.
const RC_PROBE_SIZE: u32 = ${PROBE_SIZE}u;
const RC_PROBE_CELLS: u32 = ${PROBE_CELLS}u;
// probe_spawn_tile_size_ in QuarterSpp (gi1.h:195-198): one probe per 16 x 16 pixels per frame.
const RC_PROBE_SPAWN_TILE_SIZE: u32 = ${PROBE_SPAWN_TILE_SIZE}u;
// kGI1_InvalidId (gi1.hlsl:30): no probe in a probe-mask texel.
const RC_INVALID_ID: u32 = 0xffffffffu;
// kGI1_FloatQuantize (screen_probes.hlsl:33): radiance as integers for atomic sums.
const RC_FLOAT_QUANTIZE: f32 = 1e4;
// MAX_HIT_DISTANCE (gi1.comp:127): a probe ray's length; also the hit distance
// BlendScreenProbes gives a cell no ray reached (gi1.comp:1232).
const RC_MAX_HIT_DISTANCE: f32 = 1e9;
// The hit distance stored for a ray that missed, and the largest one stored at all.
// Capsaicin packs FLT_MAX with f32tof16 (gi1.comp:1094, 1156); WGSL's pack2x16float
// and rgba16float stores give an indeterminate value outside the half range, so we
// store the largest finite half instead (register D19).
const RC_LARGEST_HALF: f32 = 65504.0;
const RC_QUARTER_PI: f32 = 0.785398163397448;
const RC_TWO_INV_PI: f32 = 0.636619772367581;
const RC_INV_PI: f32 = 0.318309886183791;

// ---- gi1.hlsl -------------------------------------------------------------------------------

// CalculateHaltonNumber (gi1.hlsl:44-57): the radical inverse of index in base.
fn rcCalculateHaltonNumber(index: u32, base: u32) -> f32 {
    var f = 1.0;
    var result = 0.0;
    var i = index;
    loop {
        if (i == 0u) {
            break;
        }
        f = f / f32(base);
        result = result + f * f32(i % base);
        // HLSL divides as float and truncates: uint(i / float(base)).
        i = u32(f32(i) / f32(base));
    }
    return result;
}

// CalculateHaltonSequence (gi1.hlsl:59-63): bases 2 and 3, repeating every 256 frames.
fn rcCalculateHaltonSequence(index: u32) -> vec2f {
    return vec2f(rcCalculateHaltonNumber((index & 0xffu) + 1u, 2u), rcCalculateHaltonNumber((index & 0xffu) + 1u, 3u));
}

// ---- screen_probes.hlsl ---------------------------------------------------------------------

// ScreenProbes_PackSeed (screen_probes.hlsl:93-96): the seed pixel as x << 16 | y.
fn rcPackSeed(seed: vec2u) -> u32 {
    return (seed.x << 16u) | seed.y;
}

// ScreenProbes_UnpackSeed (screen_probes.hlsl:99-102).
fn rcUnpackSeed(packedSeed: u32) -> vec2u {
    return vec2u(packedSeed >> 16u, packedSeed & 0xffffu);
}

// ScreenProbes_GetCellAndProbeIndex (screen_probes.hlsl:262-266): ray 'query' is cell x of probe y.
fn rcGetCellAndProbeIndex(query: u32) -> vec2u {
    return vec2u(query % RC_PROBE_CELLS, query / RC_PROBE_CELLS);
}

// ScreenProbes_PackSample / UnpackSample (screen_probes.hlsl:269-278): a direction as three halves (packHalf3).
fn rcPackSample(direction: vec3f) -> vec2u {
    return vec2u(pack2x16float(direction.xy), pack2x16float(vec2f(direction.z, 0.0)));
}

fn rcUnpackSample(packedSample: vec2u) -> vec3f {
    return vec3f(unpack2x16float(packedSample.x), unpack2x16float(packedSample.y).x);
}

// ScreenProbes_PackRadiance / UnpackRadiance (screen_probes.hlsl:219-228): radiance and hit distance as four halves.
fn rcUnpackRadiance(packedRadiance: vec2u) -> vec4f {
    return vec4f(unpack2x16float(packedRadiance.x), unpack2x16float(packedRadiance.y));
}

// ScreenProbes_QuantizeRadiance (screen_probes.hlsl:231-234).
fn rcQuantizeRadiance(radiance: vec4f) -> vec4u {
    return vec4u(round(RC_FLOAT_QUANTIZE * radiance));
}

// ScreenProbes_RecoverRadiance (screen_probes.hlsl:237-246).
fn rcRecoverRadiance(quantized: vec4u) -> vec4f {
    return vec4f(quantized) / RC_FLOAT_QUANTIZE;
}

// f32tof16 (HLSL): the half's bits in the low 16 bits, for a value within the half range.
fn rcF32ToF16Bits(value: f32) -> u32 {
    return pack2x16float(vec2f(clamp(value, -RC_LARGEST_HALF, RC_LARGEST_HALF), 0.0)) & 0xffffu;
}

// The half nearest a value (a tie to the even one), as an f32 that an rgba16float store keeps
// exactly. pack2x16float and rgba16float stores truncate on our GPUs (NOTES), and the probe
// atlas is stored, read and stored again every frame (the reprojection, the taps, the running
// mean), so truncation would drag it down by up to a step each time (register D48; the same
// rule as D31's HashGridCache_RoundHalfBits for the world cache).
fn rcNearestHalf(value: f32) -> f32 {
    let clamped = clamp(value, -RC_LARGEST_HALF, RC_LARGEST_HALF);
    let bits = pack2x16float(vec2f(clamped, 0.0)) & 0xffffu;
    let magnitude = abs(clamped);
    let packedMagnitude = abs(unpack2x16float(bits).x);
    if (magnitude == packedMagnitude) {
        return unpack2x16float(bits).x;
    }
    var lower = bits;
    var upper = bits + 1u;
    if (packedMagnitude > magnitude) {
        lower = bits - 1u;
        upper = bits;
    } else if ((bits & 0x7fffu) >= 0x7bffu) {
        return unpack2x16float(bits).x;
    }
    let below = magnitude - abs(unpack2x16float(lower).x);
    let above = abs(unpack2x16float(upper).x) - magnitude;
    let upperIsEven = (upper & 1u) == 0u;
    return unpack2x16float(select(lower, upper, above < below || (above == below && upperIsEven))).x;
}

fn rcNearestHalf4(value: vec4f) -> vec4f {
    return vec4f(rcNearestHalf(value.x), rcNearestHalf(value.y), rcNearestHalf(value.z), rcNearestHalf(value.w));
}

// ---- math/color.hlsl, math/pack.hlsl --------------------------------------------------------

// luminance (color.hlsl): Rec. 709 weights.
fn rcLuminance(color: vec3f) -> f32 {
    return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

// packNormal (pack.hlsl): three 10-bit snorms in one word.
fn rcPackNormal(value: vec3f) -> u32 {
    let packedValue = vec3u(vec3i(clamp(value, vec3f(-1.0), vec3f(1.0)) * 511.0 + 0.5 * sign(value))) & vec3u(0x3ffu);
    return packedValue.x | (packedValue.y << 10u) | (packedValue.z << 20u);
}

// unpackNormal (pack.hlsl): the three 10-bit snorms back, sign-extended.
fn rcUnpackNormal(packedValue: u32) -> vec3f {
    let value = vec3i(bitcast<i32>(packedValue << 22u), bitcast<i32>(packedValue << 12u), bitcast<i32>(packedValue << 2u)) >> vec3u(22u);
    return vec3f(value) * (1.0 / 511.0);
}

// ---- math/sampling.hlsl ---------------------------------------------------------------------

// mapToHemiOctahedron (sampling.hlsl:61-83): a point of the unit square to a +z-hemisphere
// direction, area-preserving (Clarberg's equal-area map, turned 45 degrees so the
// hemisphere fills the square). Equal areas make every probe cell the same solid angle.
fn rcMapToHemiOctahedron(samples: vec2f) -> vec3f {
    // From [0, 1] to [-1, 1], then from the square onto the diamond of the +z hemisphere.
    var st = 2.0 * samples - 1.0;
    st = vec2f(st.x + st.y, st.x - st.y) * 0.5;

    let absMapped = abs(st);
    let distance = 1.0 - (absMapped.x + absMapped.y);
    let radius = 1.0 - abs(distance);

    let phi = select(RC_QUARTER_PI * ((absMapped.y - absMapped.x) / radius + 1.0), 0.0, radius == 0.0);
    let radiusSquared = radius * radius;
    let sinTheta = radius * sqrt(2.0 - radiusSquared);
    let x = sinTheta * sign(st.x) * cos(phi);
    let y = sinTheta * sign(st.y) * sin(phi);
    let z = sign(distance) * (1.0 - radiusSquared);
    return vec3f(x, y, z);
}

// mapToHemiOctahedronInverse (sampling.hlsl:90-116): a +z-hemisphere direction back to the unit square.
fn rcMapToHemiOctahedronInverse(direction: vec3f) -> vec2f {
    // Modified version of "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD" (Clarberg).
    let absDirection = abs(direction);
    let radius = sqrt(1.0 - absDirection.z);
    let a = max(absDirection.x, absDirection.y);
    var b = min(absDirection.x, absDirection.y);
    b = select(b / a, 0.0, a == 0.0);

    var phi = atan(b) * RC_TWO_INV_PI;
    phi = select(1.0 - phi, phi, absDirection.x >= absDirection.y);

    let t = phi * radius;
    let s = radius - t;
    var st = vec2f(s, t) * sign(direction.xy);

    // Only the hemisphere above the surface is mapped, so it is turned back to fill the square.
    st = vec2f(st.x + st.y, st.x - st.y);
    return 0.5 * st + 0.5;
}

// GetOrthoVectors (sampling.hlsl:118-126): two tangents completing n to an orthonormal basis.
struct RcOrthoVectors {
    b1: vec3f,
    b2: vec3f,
}

fn rcGetOrthoVectors(n: vec3f) -> RcOrthoVectors {
    let selected = abs(n.z) > 0.0;
    let p2 = select(n.zyx, n, selected);
    let k = 1.0 / sqrt(p2.z * p2.z + n.y * n.y);
    var b1 = vec3f(0.0, -p2.z * k, n.y * k);
    b1 = select(b1.zyx, b1, selected);
    return RcOrthoVectors(b1, cross(n, b1));
}

// A probe-frame direction (x along b1, y along b2, z along the normal) in world space:
// direction.x * b1 + direction.y * b2 + direction.z * normal (gi1.comp:975-977).
fn rcFromProbeFrame(direction: vec3f, normal: vec3f) -> vec3f {
    let basis = rcGetOrthoVectors(normal);
    return direction.x * basis.b1 + direction.y * basis.b2 + direction.z * normal;
}

// A world direction in the probe's frame: mul(direction, CreateTBN(normal)) (sampling.hlsl:364-375),
// whose columns are b1, b2 and the normal.
fn rcToProbeFrame(direction: vec3f, normal: vec3f) -> vec3f {
    let basis = rcGetOrthoVectors(normal);
    return vec3f(dot(direction, basis.b1), dot(direction, basis.b2), dot(direction, normal));
}

// ---- components/random_number_generator/random_number_generator.hlsl -----------------------

// Capsaicin's Random (random_number_generator.hlsl:30-86): a PCG generator.
struct RcRandom {
    state: u32,
}

// Random::randInt (random_number_generator.hlsl:38-46).
fn rcRandInt(random: ptr<function, RcRandom>) -> u32 {
    let state = (*random).state;
    (*random).state = (*random).state * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return word;
}

// Random::randInt(max) (random_number_generator.hlsl:48-56): with its slight modulo bias.
fn rcRandIntBelow(random: ptr<function, RcRandom>, maximum: u32) -> u32 {
    return rcRandInt(random) % maximum;
}

// Random::rand (random_number_generator.hlsl:62-67): the upper 24 bits, so the float is exact.
fn rcRand(random: ptr<function, RcRandom>) -> f32 {
    return f32(rcRandInt(random) >> 8u) * (1.0 / 16777216.0);
}

// Random::rand2 (random_number_generator.hlsl:73-76).
fn rcRand2(random: ptr<function, RcRandom>) -> vec2f {
    let first = rcRand(random);
    return vec2f(first, rcRand(random));
}

// pcgHash (math/hash.hlsl:33-39).
fn rcPcgHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// MakeRandom(seed, index) (random_number_generator.hlsl:108-116). Capsaicin starts from
// g_RandomSeedBuffer[seed % size], a table of mt19937 numbers made on the CPU; we hash
// the seed instead, which needs no table (register D18).
fn rcMakeRandom(seed: u32, index: u32) -> RcRandom {
    let increment = (index << 1u) | 1u;
    return RcRandom((rcPcgHash(seed) + increment) * 747796405u + increment);
}
`;
