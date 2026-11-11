/*
 * Ported from AMD Capsaicin v1.3 (914b915): src/core/src/render_techniques/gi1/gi1.hlsl,
 * screen_probes.hlsl and gi1.comp; src/core/src/math/sampling.hlsl, pack.hlsl, color.hlsl,
 * math.hlsl and transform.hlsl; src/core/src/geometry/geometry.hlsl;
 * src/core/src/components/random_number_generator/random_number_generator.hlsl.
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

/**
 * The largest quantized hit distance one sample adds to a probe cell's sum
 * (register D48): 64 of them, a whole probe's rays in one cell, still fit a
 * `u32`. `floor((2^32 - 1) / 64)`, about 6,711 m at `kGI1_FloatQuantize`.
 */
export const LARGEST_QUANTIZED_HIT_DISTANCE = Math.floor(0xffffffff / PROBE_CELLS);

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
// The smallest normal half, 2^-14: below it halves are evenly spaced, 2^-24 apart.
const RC_SMALLEST_NORMAL_HALF: f32 = 6.103515625e-5;

// Ours (register D59): each component rounded to its nearest half (ties to even), for the probe
// atlas's rgba16float stores. WGSL lets such a store take either neighbouring half: the 16float
// write is quantizeToF16 (WGSL 6.6.1, Channel Formats), which converts as 15.7.6 says, "WGSL does
// not specify whether the higher or lower representable value is chosen". The M5 takes the one
// toward zero, and the filter's min then keeps every lost step of the hit distance. A value that
// already is a half is stored exactly on every GPU (15.7.6), so this makes the store's rounding
// the same everywhere. Halves have 10 fraction bits: neighbours are 2^(e - 10) apart for a
// magnitude in [2^e, 2^(e+1)), and frexp gives e + 1 exactly. Scaling by a power of two with
// ldexp and a multiply is exact (both are correctly rounded in WGSL, where exp2 and division
// aren't), and round() takes ties to even. Values past the half range are left to the store.
fn rcNearestHalf(value: vec4f) -> vec4f {
    let stepExponent = frexp(max(abs(value), vec4f(RC_SMALLEST_NORMAL_HALF))).exp - vec4i(11);
    return round(value * ldexp(vec4f(1.0), -stepExponent)) * ldexp(vec4f(1.0), stepExponent);
}
const RC_QUARTER_PI: f32 = 0.785398163397448;
const RC_TWO_INV_PI: f32 = 0.636619772367581;

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

// The largest quantized hit distance one sample adds to a cell (register D48): a cell's sum of
// 64 such stays below 2^32.
const RC_LARGEST_QUANTIZED_HIT_DISTANCE: u32 = ${LARGEST_QUANTIZED_HIT_DISTANCE}u;

// ScreenProbes_QuantizeRadiance (screen_probes.hlsl:231-234), with the hit distance capped
// (register D48). A missed ray's distance is 65,504 here (D19), 655,040,000 once quantized, so
// seven of them in one cell wrapped the u32 sum; ray guiding can send all 64 rays of a probe into
// one sky cell. Capsaicin's own miss is f32tof16(FLT_MAX), infinity, which D3D converts to
// 0xffffffff, so its sum wraps from the second ray.
fn rcQuantizeRadiance(radiance: vec4f) -> vec4u {
    let quantized = vec4u(round(RC_FLOAT_QUANTIZE * radiance));
    return vec4u(quantized.xyz, min(quantized.w, RC_LARGEST_QUANTIZED_HIT_DISTANCE));
}

// ScreenProbes_RecoverRadiance (screen_probes.hlsl:237-246).
fn rcRecoverRadiance(quantized: vec4u) -> vec4f {
    return vec4f(quantized) / RC_FLOAT_QUANTIZE;
}

// ---- math/color.hlsl, math/math.hlsl, math/pack.hlsl ----------------------------------------

// luminance (color.hlsl:34-37): Rec. 709 weights.
fn rcLuminance(color: vec3f) -> f32 {
    return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

// squared (math.hlsl:63-66).
fn rcSquared(value: f32) -> f32 {
    return value * value;
}

// packNormal (pack.hlsl:468-472): three 10-bit snorms, rounded away from zero.
fn rcPackNormal(value: vec3f) -> u32 {
    let scaled = vec3i(clamp(value, vec3f(-1.0), vec3f(1.0)) * 511.0 + 0.5 * sign(value));
    let packedValue = bitcast<vec3u>(scaled) & vec3u(0x3ffu);
    return packedValue.x | (packedValue.y << 10u) | (packedValue.z << 20u);
}

// unpackNormal (pack.hlsl:479-483): each 10-bit field sign-extended by an arithmetic shift.
fn rcUnpackNormal(packedValue: u32) -> vec3f {
    let value = bitcast<vec3i>(vec3u(packedValue << 22u, packedValue << 12u, packedValue << 2u)) >> vec3u(22u);
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
    // Register D34: the one change from Capsaicin's sqrt(1 - |z|). BlendScreenProbes passes a
    // direction read back from halves (gi1.comp:1197-1201), which can be up to about 3e-4 longer
    // than 1 where pack2x16float rounds up (the Apple M5 rounds to nearest), so a ray near the
    // normal has z past 1. WGSL gives sqrt of a negative and u32() of the NaN that follows no
    // defined value (sqrt's domain is [0, +inf], WGSL 17.5.58; 15.7.2, 15.7.6), and on the M5
    // such rays all landed in cell 0, a grazing corner. Clamped, they map to the pole (the centre
    // of the square), next to the cell they were sampled in. A unit direction is unchanged.
    let radius = sqrt(max(1.0 - absDirection.z, 0.0));
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

// ---- geometry/geometry.hlsl -----------------------------------------------------------------

// offsetPosition (geometry.hlsl:35-47): Wachter and Binder's offset from Ray Tracing Gems
// (chapter 6, "A Fast and Robust Method for Avoiding Self-Intersection"). Each coordinate moves
// along the normal by 256 x its normal component in steps of its own f32 rounding (an integer
// step on its bits), so the push scales with the coordinate's error; within 1/32 of the origin,
// where those steps vanish, by 1/65536 along the normal instead. Capsaicin passes the hit's
// interpolated normal, turned to the side the ray came from (gi1.comp:1903, 1933). Used for the
// second-bounce ray only (register D39); the platform's other rays use giShadeOffsetRayOrigin,
// which pushes along the normal exactly (NOTES: this one's per-coordinate steps tilt along the
// surface, which carried 8 corner points of the trace oracle's thin box past the next face).
fn rcOffsetPosition(position: vec3f, normal: vec3f) -> vec3f {
    let origin = 1.0 / 32.0;
    let floatScale = 1.0 / 65536.0;
    let intScale = 256.0;

    // int3(normal * intScale): truncated toward zero, as HLSL's conversion.
    let intN = vec3i(normal * intScale);
    // asfloat(asint(position) + (position < 0 ? -intN : intN)): a step away from zero on the
    // bits moves a negative coordinate further negative, hence the sign flip.
    let positionI = bitcast<vec3f>(bitcast<vec3i>(position) + select(intN, -intN, position < vec3f(0.0)));

    let posOffset = position + normal * floatScale;
    return select(positionI, posOffset, abs(position) < vec3f(origin));
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
